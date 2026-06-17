"""
DES-LOC Heterogeneous MXFP8 Refit Engine
==========================================

Upstream design intent (Megatron fca1679):
    Megatron's MXFP8 refit (#3742) introduced a pluggable ``ReshardTransform``
    abstraction so that weight transfers between training and inference workers
    can transparently convert BF16 parameters to FlashInfer's MXFP8 block-
    floating-point format.  Three key design decisions drive the upstream code:

    1. **Persistent buffer address stability**: CUDA graphs capture device
       pointers at warmup time.  If quantize_params_to_mxfp8 creates new
       MXFP8Tensor objects on every refit, the graph's captured pointers
       become stale, causing silent data corruption or segfaults.  The
       ``persistent_buffers`` dict is therefore created once and mutated
       in-place on all subsequent refits (copy_ into existing tensors).

    2. **1D vs 2D scale layout**: FlashInfer uses a "swizzled" 1D scale
       layout for certain tile sizes.  A partial slice update corrupts the
       swizzle pattern, so partial BF16 slices must be accumulated and the
       full weight requantized atomically once all shards arrive.

    3. **Barrier ordering**: The original code had a race where
       torch.cuda.synchronize() fired *before* the writeback loop, leaving
       async copy_() kernels in flight when execute_reshard_plan returned.
       The fix adds a second synchronize() after writebacks.

DES-LOC adaptation (Neuron_SP M3478-BF):
    DES-LOC (Decoupled Execution with Shared LOcality Cache) separates
    training execution (2x A6000 48 GB, SM86) from inference execution
    (1x H100 NVL 96 GB, SM90) across a PCIe fabric with 1.5 TB CPU DRAM
    as the shared locality cache (SLC).

    The heterogeneous hardware introduces two precision-routing concerns
    absent in Megatron's homogeneous GPU fleet:

    A. **SM86 cannot execute native MXFP8 kernels** (requires SM89+).
       Weights resident on A6000 training replicas must be kept in BF16.
       Only the H100 inference replica holds MXFP8 persistent buffers.
       The ``HeteroMXFP8RefitEngine`` checks SM version at runtime and
       routes accordingly via ``DeviceRole``.

    B. **PCIe bandwidth bottleneck (no NVLink)**: Unlike NVLink-connected
       DGX pods, PCIe limits inter-GPU bandwidth to ~32 GB/s aggregate.
       To avoid saturating the PCIe bus during refit, the SLC (CPU DRAM)
       is used as a staging area: training workers DMA weights to pinned
       CPU buffers; the inference worker pulls from CPU rather than from
       peer GPU memory.  This is the "Shared LOcality Cache" in DES-LOC.

    C. **Deferred quantization on H100**: The H100 inference worker
       receives BF16 from CPU staging (receiver-side conversion), converts
       to MXFP8, and writes into persistent CUDA-graph-safe buffers.
       The 1D-scale accumulation logic from upstream is preserved intact.

    D. **Stream wrapper naming**: Upstream renamed ``torch_*_stream`` →
       ``torch_*_stream_wrapper`` to avoid shadowing the underlying CUDA
       stream object.  DES-LOC follows the same convention for all internal
       stream handles.

Hardware topology assumed:
    rank 0,1 : A6000 48 GB (SM86)  — training / source
    rank 2    : H100 NVL 96 GB (SM90) — inference / destination
    CPU DRAM  : 1.5 TB pinned staging (SLC)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardware / capability detection
# ---------------------------------------------------------------------------

def _sm_version(device: Optional[torch.device] = None) -> int:
    """Return the SM major*10 + minor for *device* (or the current device)."""
    idx = (device.index if device is not None else torch.cuda.current_device())
    props = torch.cuda.get_device_properties(idx)
    return props.major * 10 + props.minor


def _supports_mxfp8(device: Optional[torch.device] = None) -> bool:
    """MXFP8 hardware support requires SM ≥ 89 (Ada / H100 and later)."""
    return _sm_version(device) >= 89


class DeviceRole(Enum):
    """Role of the current rank in the DES-LOC heterogeneous cluster."""
    TRAINING_SOURCE = auto()   # A6000 SM86 — produces BF16 weights
    INFERENCE_SINK  = auto()   # H100  SM90 — consumes MXFP8 weights
    IDLE            = auto()   # participates in collectives only


# ---------------------------------------------------------------------------
# Shared LOcality Cache (SLC) — pinned CPU staging buffers
# ---------------------------------------------------------------------------

class SharedLocalityCache:
    """PCIe-aware CPU DRAM staging layer for DES-LOC weight transfers.

    In a homogeneous NVLink cluster, GPU-to-GPU weight transfers happen via
    NCCL P2P or NVSHMEM.  On the A6000 + H100 PCIe topology, there is no
    NVLink, so direct GPU-to-GPU copies share PCIe bandwidth with all other
    traffic.

    DES-LOC instead uses CPU DRAM (1.5 TB available) as a locality cache:
      - Training workers write BF16 shards into pinned CPU buffers.
      - The inference worker reads from CPU, converts to MXFP8, and loads
        into its persistent GPU buffers.
    This decouples the A6000→CPU phase from the CPU→H100 phase, allowing
    both to overlap with training computation on the A6000s.

    The cache is keyed by (param_name, shard_index) so that partial TP
    shards can be written and read independently.
    """

    def __init__(self, capacity_bytes: int = 32 * 1024**3):
        """
        Args:
            capacity_bytes: Maximum pinned CPU memory to allocate.  Default
                32 GB, well within the 1.5 TB available.
        """
        self._capacity = capacity_bytes
        self._used: int = 0
        self._store: Dict[Tuple[str, int], torch.Tensor] = {}
        self._lock = threading.Lock()
        logger.info(
            "SharedLocalityCache initialised, capacity=%.1f GB",
            capacity_bytes / 1024**3,
        )

    # -- write side (training workers / A6000) --------------------------------

    def put(self, param_name: str, shard_idx: int, tensor: torch.Tensor) -> None:
        """Stage *tensor* (GPU or CPU) into a pinned CPU buffer.

        If *tensor* is already on CPU it is copied into a pinned allocation.
        GPU tensors are first DMA'd to CPU via a non-blocking copy.

        The operation is thread-safe; concurrent writes from TP workers are
        serialised per-key so the reader always sees a consistent shard.
        """
        key = (param_name, shard_idx)
        pinned = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
        if tensor.is_cuda:
            # Non-blocking H2D → D2H; caller must synchronize before reading.
            pinned.copy_(tensor.cpu(), non_blocking=False)
        else:
            pinned.copy_(tensor)

        with self._lock:
            old = self._store.get(key)
            if old is not None:
                self._used -= old.numel() * old.element_size()
            self._store[key] = pinned
            self._used += pinned.numel() * pinned.element_size()
            logger.debug(
                "SLC.put %s[%d] shape=%s used=%.2f GB",
                param_name, shard_idx, list(tensor.shape),
                self._used / 1024**3,
            )

    def get(self, param_name: str, shard_idx: int) -> Optional[torch.Tensor]:
        """Retrieve a staged shard (or None if not yet written)."""
        return self._store.get((param_name, shard_idx))

    def wait_for(
        self,
        param_name: str,
        shard_idx: int,
        timeout: float = 60.0,
        poll_interval: float = 0.005,
    ) -> torch.Tensor:
        """Block until the shard is available, then return it.

        Args:
            timeout: Maximum seconds to wait before raising RuntimeError.
            poll_interval: Seconds between cache polls.
        """
        deadline = time.monotonic() + timeout
        while True:
            t = self.get(param_name, shard_idx)
            if t is not None:
                return t
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"SLC timeout waiting for {param_name}[{shard_idx}] "
                    f"after {timeout:.1f}s"
                )
            time.sleep(poll_interval)

    def evict(self, param_name: str, shard_idx: int) -> None:
        """Remove a shard from the cache (free pinned memory)."""
        key = (param_name, shard_idx)
        with self._lock:
            t = self._store.pop(key, None)
            if t is not None:
                self._used -= t.numel() * t.element_size()

    def stats(self) -> Dict[str, object]:
        return {
            "num_shards": len(self._store),
            "used_bytes": self._used,
            "capacity_bytes": self._capacity,
        }


# ---------------------------------------------------------------------------
# MXFP8 tensor shim (graceful degradation when FlashInfer unavailable)
# ---------------------------------------------------------------------------

try:
    from flashinfer import mxfp8_quantize as _fi_quantize  # noqa: F401
    _HAVE_FLASHINFER = True
except ImportError:
    _HAVE_FLASHINFER = False
    logger.warning(
        "FlashInfer not found — MXFP8 quantization disabled.  "
        "Inference replica will receive BF16 weights."
    )


@dataclass
class DESLOC_MXFP8Tensor:
    """Lightweight MXFP8 container used by the DES-LOC refit engine.

    Wraps a FlashInfer MXFP8Tensor when available; falls back to storing
    plain BF16 data + a unit scale for SM86 training replicas or when
    FlashInfer is absent.  The ``persistent`` flag indicates whether the
    underlying buffers must not be reallocated (CUDA graph constraint).

    Fields:
        data:       quantized uint8 tensor (MXFP8) or BF16 fallback
        scale:      per-block scale tensor (1D swizzled or 2D row-wise)
        persistent: if True, write via copy_() not assignment
        param_name: FQN of the originating parameter (for logging)
    """
    data:       torch.Tensor
    scale:      torch.Tensor
    persistent: bool = False
    param_name: str  = ""

    @classmethod
    def from_bf16(
        cls,
        bf16: torch.Tensor,
        param_name: str = "",
        persistent: bool = False,
    ) -> "DESLOC_MXFP8Tensor":
        """Quantize a BF16 tensor to MXFP8 using FlashInfer.

        Falls back to BF16 identity when FlashInfer is unavailable or the
        current device is SM86 (A6000) which lacks native MXFP8 support.
        """
        if not _HAVE_FLASHINFER or not _supports_mxfp8():
            # SM86 path: store BF16, unit scale
            scale = torch.ones(
                bf16.shape[0], dtype=torch.float32, device=bf16.device
            )
            return cls(
                data=bf16.contiguous(),
                scale=scale,
                persistent=persistent,
                param_name=param_name,
            )
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
        fi = MXFP8Tensor.from_bf16(bf16)
        return cls(
            data=fi.data,
            scale=fi.scale,
            persistent=persistent,
            param_name=param_name,
        )

    def update_inplace(self, new_bf16: torch.Tensor) -> None:
        """Requantize *new_bf16* into this tensor's existing buffers.

        Preserves device pointers for CUDA graph compatibility.
        Raises RuntimeError if buffer shapes mismatch.
        """
        if not _HAVE_FLASHINFER or not _supports_mxfp8():
            if self.data.shape != new_bf16.shape:
                raise RuntimeError(
                    f"[SLC] Buffer shape mismatch for {self.param_name}: "
                    f"got {new_bf16.shape}, expected {self.data.shape}"
                )
            self.data.copy_(new_bf16)
            return

        from flashinfer import mxfp8_quantize
        new_data, new_scale = mxfp8_quantize(new_bf16.to(torch.bfloat16))
        self.data.copy_(new_data)
        self.scale.copy_(new_scale)

    def dequantize(self) -> torch.Tensor:
        """Reconstruct BF16 from MXFP8 (or return BF16 data on SM86)."""
        if not _HAVE_FLASHINFER or not _supports_mxfp8():
            return self.data.to(torch.bfloat16)
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
        fi = MXFP8Tensor.__new__(MXFP8Tensor)
        fi.data  = self.data
        fi.scale = self.scale
        return fi.dequantize()


# ---------------------------------------------------------------------------
# Scale-slice helper (mirrors upstream _scale_slice_from_data_slice)
# ---------------------------------------------------------------------------

def _scale_slice_from_data_slice(
    data_slice: Tuple,
    block_size: int = 32,
) -> Tuple:
    """Compute the scale tensor slice corresponding to a data tensor slice.

    In MXFP8, ``block_size`` consecutive elements along the K (last) dimension
    share one scale value.  This function divides the last slice's start/stop
    by ``block_size``.  Alignment is enforced to catch mis-sharded TP plans.
    """
    adjusted = list(data_slice)
    last = adjusted[-1]
    if isinstance(last, slice):
        if last.start is not None and last.start % block_size != 0:
            raise ValueError(
                f"Data slice last-dim start {last.start} not aligned to "
                f"block_size={block_size}"
            )
        if last.stop is not None and last.stop % block_size != 0:
            raise ValueError(
                f"Data slice last-dim stop {last.stop} not aligned to "
                f"block_size={block_size}"
            )
        adjusted[-1] = slice(
            (last.start // block_size) if last.start is not None else None,
            (last.stop  // block_size) if last.stop  is not None else None,
        )
    elif isinstance(last, int):
        adjusted[-1] = last // block_size
    return tuple(adjusted)


# ---------------------------------------------------------------------------
# DES-LOC Reshard Transform
# ---------------------------------------------------------------------------

class DESLOCReshardTransform:
    """Pluggable weight-transfer transform for DES-LOC heterogeneous refit.

    Extends Megatron's ReshardTransform contract to add:

    * **SM-aware routing**: training ranks (SM86) send BF16 via the SLC;
      the inference rank (SM90) receives BF16 and converts to MXFP8.
    * **SLC staging**: instead of direct GPU↔GPU transfers over PCIe,
      each shard is written to pinned CPU DRAM by the sender and pulled by
      the receiver.  This avoids saturating the PCIe bus and allows overlap
      with the next training microbatch.
    * **1D scale accumulation**: preserves the upstream logic for assembling
      partial shards before re-quantizing with a 1D swizzled scale layout.
    * **Persistent buffer preservation**: all MXFP8Tensor writes go through
      copy_() so CUDA-graph device-pointer captures remain valid.

    Args:
        convertible_params: FQN set of parameters handled by this transform.
        persistent_buffers: dict mapping bare param name (without prefix) to
            DESLOC_MXFP8Tensor.  Must be pre-allocated by the inference rank.
        slc: SharedLocalityCache instance shared between all ranks in the job.
        local_role: DeviceRole of the current process.
        buffer_key_prefix: prefix stripped from param FQN for buffer lookup.
        tp_world_size: tensor-parallel world size; used to track shard arrival.
    """

    def __init__(
        self,
        convertible_params: Set[str],
        persistent_buffers: Dict[str, DESLOC_MXFP8Tensor],
        slc: SharedLocalityCache,
        local_role: DeviceRole,
        buffer_key_prefix: str = "decoder.",
        tp_world_size: int = 1,
    ):
        self.convertible_params  = convertible_params
        self.persistent_buffers  = persistent_buffers
        self.slc                 = slc
        self.local_role          = local_role
        self.buffer_key_prefix   = buffer_key_prefix
        self.tp_world_size       = tp_world_size

        # 1D-scale accumulation state: buf_key → [accum_bf16, shards_written]
        self._pending_1d: Dict[str, List] = {}
        # Shard arrival tracking: buf_key → count of slices finalised
        self._shard_count: Dict[str, int] = {}

    # -- predicate -----------------------------------------------------------

    def should_transform(self, param_name: str) -> bool:
        return param_name in self.convertible_params

    # -- send side (A6000 / SM86 training ranks) ----------------------------

    def prepare_send(
        self,
        param_name: str,
        src_slice: Tuple,
        src_param: torch.Tensor,
        shard_idx: int = 0,
    ) -> List[torch.Tensor]:
        """Stage a BF16 weight shard into the SLC for the inference rank.

        Because A6000 (SM86) cannot run MXFP8 kernels, conversion is always
        deferred to the receiver (H100 SM90).  The shard is written to the
        SharedLocalityCache; the returned list contains the CPU-pinned tensor
        so that callers using a CopyService backend can also submit it
        directly if desired.

        Args:
            param_name: Fully-qualified parameter name.
            src_slice: Slice into the source parameter.
            src_param: Source parameter tensor (GPU).
            shard_idx: TP shard index (0 for unsharded).
        """
        if self.local_role != DeviceRole.TRAINING_SOURCE:
            raise RuntimeError(
                "prepare_send called on non-training rank "
                f"(role={self.local_role})"
            )

        # Dequantize if the training model uses TE MXFP8 params
        data = src_param
        try:
            from transformer_engine.pytorch.tensor.mxfp8_tensor import (
                MXFP8Tensor as _TEMXFP8,
            )
            if isinstance(src_param, _TEMXFP8):
                data = src_param.dequantize()
        except ImportError:
            pass

        shard_bf16 = data[src_slice].contiguous().to(torch.bfloat16)
        self.slc.put(param_name, shard_idx, shard_bf16)
        logger.debug(
            "prepare_send: staged %s shard=%d shape=%s to SLC",
            param_name, shard_idx, list(shard_bf16.shape),
        )
        return [shard_bf16]

    # -- recv side (H100 / SM90 inference rank) -----------------------------

    def prepare_recv(
        self,
        param_name: str,
        dst_slice: Tuple,
        shard_idx: int = 0,
    ) -> List[torch.Tensor]:
        """Pull a BF16 shard from the SLC into a pinned staging buffer.

        Blocks until the training rank has written the shard.  Returns a
        single BF16 tensor on the inference device; finalize_recv performs
        the actual MXFP8 conversion.
        """
        if self.local_role != DeviceRole.INFERENCE_SINK:
            raise RuntimeError(
                "prepare_recv called on non-inference rank "
                f"(role={self.local_role})"
            )
        buf_key = param_name.removeprefix(self.buffer_key_prefix)
        buf = self.persistent_buffers[buf_key]
        device = buf.data.device

        # Wait for the shard from the training replica
        cpu_shard = self.slc.wait_for(param_name, shard_idx)
        gpu_shard = cpu_shard.to(device, non_blocking=False)
        logger.debug(
            "prepare_recv: pulled %s shard=%d shape=%s from SLC",
            param_name, shard_idx, list(gpu_shard.shape),
        )
        return [gpu_shard]

    # -- writeback (inference rank only) ------------------------------------

    def finalize_recv(
        self,
        param_name: str,
        dst_slice: Tuple,
        recv_buffers: List[torch.Tensor],
        shard_idx: int = 0,
    ) -> None:
        """Convert received BF16 → MXFP8 and write into persistent buffers.

        Handles two scale layouts:

        * **2D scale**: each scale row is independent; the slice can be
          quantized and written immediately.
        * **1D swizzled scale**: the swizzle layout spans the full weight
          tensor; partial updates corrupt the scale.  Slices are accumulated
          in a BF16 staging tensor and the full weight is requantized once
          all TP shards have arrived.

        After writing, the SLC entry is evicted to free pinned memory.
        """
        buf_key = param_name.removeprefix(self.buffer_key_prefix)
        buf = self.persistent_buffers[buf_key]
        [bf16_recv] = recv_buffers

        if buf.scale.ndim == 1:
            # ---- 1D swizzled scale path ------------------------------------
            # Cannot update partial slices — accumulate into a full BF16 buffer
            # then requantize the complete weight atomically.
            if buf_key not in self._pending_1d:
                self._pending_1d[buf_key] = [
                    torch.zeros(
                        buf.data.shape,
                        dtype=torch.bfloat16,
                        device=buf.data.device,
                    ),
                    0,   # elements written
                ]
            accum, written = self._pending_1d[buf_key]
            accum[dst_slice].copy_(bf16_recv)
            written += bf16_recv.numel()
            self._pending_1d[buf_key][1] = written

            total_elems = buf.data.numel()
            logger.debug(
                "1D-scale accumulation %s: %d / %d elements written",
                param_name, written, total_elems,
            )
            if written >= total_elems:
                if written != total_elems:
                    raise AssertionError(
                        f"1D-scale param {param_name!r}: received {written} "
                        f"elements, expected {total_elems} "
                        f"(duplicate or missing shards?)"
                    )
                # All shards arrived — requantize the full weight
                buf.update_inplace(accum)
                del self._pending_1d[buf_key]
                logger.info(
                    "1D-scale finalized %s  shape=%s",
                    param_name, list(buf.data.shape),
                )
        else:
            # ---- 2D scale path (per-row scales, slice-independent) ----------
            if not _HAVE_FLASHINFER or not _supports_mxfp8():
                buf.data[dst_slice].copy_(bf16_recv)
            else:
                from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
                mxfp8_slice = MXFP8Tensor.from_bf16(bf16_recv.to(torch.bfloat16))
                buf.data[dst_slice].copy_(mxfp8_slice.data)
                sc_slice = _scale_slice_from_data_slice(dst_slice)
                buf.scale[sc_slice].copy_(mxfp8_slice.scale)
            logger.debug(
                "2D-scale slice written %s  dst_slice=%s",
                param_name, dst_slice,
            )

        # Evict SLC entry to reclaim pinned CPU memory
        self.slc.evict(param_name, shard_idx)


# ---------------------------------------------------------------------------
# Persistent buffer factory
# ---------------------------------------------------------------------------

def build_persistent_mxfp8_buffers(
    named_params: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, DESLOC_MXFP8Tensor]:
    """Allocate persistent DESLOC_MXFP8Tensor buffers for the inference replica.

    Called **once** during initialisation, before any CUDA graph warmup.
    The buffers' device pointers are subsequently captured by CUDA graphs and
    must never be reallocated.

    On SM86 devices (A6000), returns BF16 tensors with unit scales because
    the hardware cannot execute MXFP8 kernels.  On SM90 (H100 NVL) the full
    MXFP8 quantisation is applied.

    Args:
        named_params: dict of {bare_name: bf16_tensor} from the inference
            model's decoder (before any format conversion).
        device: CUDA device for the inference replica.

    Returns:
        dict mapping bare param name → DESLOC_MXFP8Tensor (persistent=True)
    """
    buffers: Dict[str, DESLOC_MXFP8Tensor] = {}
    for name, param in named_params.items():
        bf16 = param.detach().to(device, dtype=torch.bfloat16)
        buf = DESLOC_MXFP8Tensor.from_bf16(bf16, param_name=name, persistent=True)
        buffers[name] = buf
        logger.debug(
            "persistent buffer: %s  data=%s  scale=%s",
            name, list(buf.data.shape), list(buf.scale.shape),
        )
    logger.info(
        "Built %d persistent MXFP8 buffers on %s (SM%d)",
        len(buffers),
        device,
        _sm_version(device),
    )
    return buffers


# ---------------------------------------------------------------------------
# Refit engine
# ---------------------------------------------------------------------------

@dataclass
class HeteroRefitConfig:
    """Configuration for the DES-LOC heterogeneous refit engine.

    Attributes:
        tp_world_size: Tensor-parallel group size (typically 2 for A6000 pair).
        param_names:   List of bare parameter names (without prefix) that the
                       engine should refit on each call to ``execute_refit``.
        buffer_prefix: Module prefix prepended to bare names for FQN lookup.
        slc_capacity_gb: GB of CPU DRAM reserved for SLC staging.
        sync_barrier:  Whether to insert a dist.barrier() after all refits.
        stream_timeout: Seconds to wait for each SLC shard before raising.
    """
    tp_world_size:    int   = 2
    param_names:      List[str] = field(default_factory=list)
    buffer_prefix:    str   = "decoder."
    slc_capacity_gb:  float = 32.0
    sync_barrier:     bool  = True
    stream_timeout:   float = 60.0


class HeteroMXFP8RefitEngine:
    """DES-LOC heterogeneous refit engine for SM86/SM90 mixed clusters.

    Orchestrates weight transfer from A6000 training replicas (SM86, BF16)
    to the H100 inference replica (SM90, MXFP8) via the SharedLocalityCache.

    Lifecycle:
        1. Call ``__init__`` on all ranks simultaneously (collective comm).
        2. Training ranks call ``stage_weights(model)`` after each update
           step; this is the A6000-side write into the SLC.
        3. The inference rank calls ``apply_refit()`` to pull from SLC,
           convert to MXFP8, and write into persistent CUDA-graph buffers.
        4. Repeat from step 2.

    The engine maintains one DESLOCReshardTransform per direction (source
    ranks share a transform state for shard counting; the sink rank has its
    own transform with the persistent buffers).

    Args:
        config: HeteroRefitConfig.
        local_role: DeviceRole of the calling process.
        persistent_buffers: Pre-built DESLOC_MXFP8Tensor dict (inference
            rank only; training ranks pass empty dict or None).
        group: Optional dist.ProcessGroup for barrier synchronisation.
    """

    def __init__(
        self,
        config: HeteroRefitConfig,
        local_role: DeviceRole,
        persistent_buffers: Optional[Dict[str, DESLOC_MXFP8Tensor]] = None,
        group: Optional[dist.ProcessGroup] = None,
    ):
        self.config       = config
        self.local_role   = local_role
        self.group        = group
        self._refit_count = 0

        self.slc = SharedLocalityCache(
            capacity_bytes=int(config.slc_capacity_gb * 1024**3)
        )

        # Build the FQN set (prefix + bare name)
        self._convertible: Set[str] = {
            f"{config.buffer_prefix}{n}" for n in config.param_names
        }

        # Only the inference sink needs persistent buffers
        _pb: Dict[str, DESLOC_MXFP8Tensor] = persistent_buffers or {}

        self._transform = DESLOCReshardTransform(
            convertible_params=self._convertible,
            persistent_buffers=_pb,
            slc=self.slc,
            local_role=local_role,
            buffer_key_prefix=config.buffer_prefix,
            tp_world_size=config.tp_world_size,
        )

        logger.info(
            "HeteroMXFP8RefitEngine initialised: role=%s  params=%d  "
            "tp=%d  SM=%d",
            local_role.name,
            len(self._convertible),
            config.tp_world_size,
            _sm_version(),
        )

    # -- training-side API ---------------------------------------------------

    def stage_weights(
        self,
        model: torch.nn.Module,
        shard_idx: int = 0,
    ) -> None:
        """Write current BF16 parameter values into the SLC (A6000 ranks).

        Should be called after the optimiser step and before
        ``apply_refit()``.  Safe to call from all training ranks in parallel;
        each rank tags its shards with *shard_idx*.

        Args:
            model: Training model (state-dict must be on GPU).
            shard_idx: TP shard index of this rank (0 or 1 for tp_world=2).
        """
        if self.local_role != DeviceRole.TRAINING_SOURCE:
            logger.debug("stage_weights: skipped (role=%s)", self.local_role.name)
            return

        t0 = time.monotonic()
        staged = 0
        for bare_name in self.config.param_names:
            fqn = f"{self.config.buffer_prefix}{bare_name}"
            # Navigate the module hierarchy
            parts = bare_name.split(".")
            obj = model
            for p in parts:
                obj = getattr(obj, p, None)
                if obj is None:
                    logger.warning("stage_weights: %s not found in model", fqn)
                    break
            if obj is None:
                continue
            param_tensor = obj.data if isinstance(obj, torch.nn.Parameter) else obj
            self._transform.prepare_send(fqn, (slice(None),) * param_tensor.dim(),
                                         param_tensor, shard_idx)
            staged += 1

        elapsed = time.monotonic() - t0
        logger.info(
            "stage_weights: staged %d params to SLC in %.3fs (shard=%d)",
            staged, elapsed, shard_idx,
        )

    # -- inference-side API --------------------------------------------------

    def apply_refit(self, shard_indices: Optional[List[int]] = None) -> None:
        """Pull staged shards from SLC and write into persistent MXFP8 buffers.

        Blocks until all shards for each parameter have been staged by the
        training ranks.  Should be called on the inference rank (H100) before
        each decode batch.

        Args:
            shard_indices: List of TP shard indices to pull.  Defaults to
                ``list(range(tp_world_size))``.
        """
        if self.local_role != DeviceRole.INFERENCE_SINK:
            logger.debug("apply_refit: skipped (role=%s)", self.local_role.name)
            return

        if shard_indices is None:
            shard_indices = list(range(self.config.tp_world_size))

        t0 = time.monotonic()
        for bare_name in self.config.param_names:
            fqn = f"{self.config.buffer_prefix}{bare_name}"
            for shard_idx in shard_indices:
                recv_bufs = self._transform.prepare_recv(
                    fqn, (slice(None),), shard_idx
                )
                self._transform.finalize_recv(
                    fqn, (slice(None),), recv_bufs, shard_idx
                )

        # Ensure all writeback copies are visible before returning.
        # (Matches upstream's second torch.cuda.synchronize() fix.)
        torch.cuda.synchronize()

        self._refit_count += 1
        elapsed = time.monotonic() - t0
        logger.info(
            "apply_refit #%d: %d params x %d shards in %.3fs",
            self._refit_count,
            len(self.config.param_names),
            len(shard_indices),
            elapsed,
        )

    # -- collective barrier --------------------------------------------------

    def barrier(self) -> None:
        """Optional collective barrier across all DES-LOC ranks."""
        if self.config.sync_barrier and dist.is_initialized():
            dist.barrier(group=self.group)
            logger.debug("HeteroMXFP8RefitEngine: barrier complete")

    # -- diagnostics ---------------------------------------------------------

    def slc_stats(self) -> Dict[str, object]:
        return self.slc.stats()

    def pending_1d_params(self) -> List[str]:
        """Return list of param keys still awaiting 1D-scale accumulation."""
        return list(self._transform._pending_1d.keys())


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sm = _sm_version(device) if device.type == "cuda" else 0
    logger.info("Smoke test on %s  SM=%d  FlashInfer=%s", device, sm, _HAVE_FLASHINFER)

    # 1) SLC round-trip
    slc = SharedLocalityCache(capacity_bytes=1 * 1024**3)
    t_orig = torch.randn(64, 128, dtype=torch.bfloat16)
    slc.put("layer.weight", 0, t_orig)
    t_back = slc.get("layer.weight", 0)
    assert t_back is not None and t_back.shape == t_orig.shape, "SLC put/get failed"
    logger.info("  [PASS] SLC round-trip")

    # 2) DESLOC_MXFP8Tensor creation and update
    bf16 = torch.randn(32, 64, dtype=torch.bfloat16, device=device)
    buf = DESLOC_MXFP8Tensor.from_bf16(bf16, param_name="test.weight", persistent=True)
    ptr_before = buf.data.data_ptr()
    new_bf16 = torch.randn(32, 64, dtype=torch.bfloat16, device=device)
    buf.update_inplace(new_bf16)
    assert buf.data.data_ptr() == ptr_before, "persistent buffer address changed!"
    logger.info("  [PASS] DESLOC_MXFP8Tensor persistent update")

    # 3) Scale-slice helper alignment check
    s = _scale_slice_from_data_slice((slice(0, 64), slice(0, 128)), block_size=32)
    assert s == (slice(0, 2), slice(0, 4)), f"Unexpected scale slice: {s}"
    logger.info("  [PASS] _scale_slice_from_data_slice 2D")

    # 4) SM-version detection
    if device.type == "cuda":
        assert isinstance(_sm_version(device), int), "SM version should be int"
        logger.info("  [PASS] SM version detection: SM%d", _sm_version(device))

    # 5) DESLOCReshardTransform should_transform
    fake_buf = DESLOC_MXFP8Tensor.from_bf16(
        torch.zeros(16, 32, dtype=torch.bfloat16, device=device),
        persistent=True,
    )
    transform = DESLOCReshardTransform(
        convertible_params={"decoder.mlp.weight"},
        persistent_buffers={"mlp.weight": fake_buf},
        slc=slc,
        local_role=DeviceRole.IDLE,
        buffer_key_prefix="decoder.",
    )
    assert transform.should_transform("decoder.mlp.weight")
    assert not transform.should_transform("decoder.mlp.bias")
    logger.info("  [PASS] DESLOCReshardTransform.should_transform")

    logger.info("All smoke tests passed.")
    sys.exit(0)
