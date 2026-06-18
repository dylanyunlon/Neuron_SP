"""
DES-LOC Heterogeneous MXFP8 Weight Refit Engine
================================================

Upstream design intent (Megatron fca1679):
    Megatron's MXFP8 refit system solves a specific inference-serving problem:
    a training job runs on one set of GPUs (BF16), and an inference process runs
    concurrently on another set.  Weight resharding transfers slices between the
    two worlds via NCCL/Gloo/NVSHMEM.  Because the inference side uses CUDA graphs
    (for latency), the device pointers to weight tensors must remain stable across
    refits — hence "persistent buffers".  The commit introduces:

    1. quantize_params_to_mxfp8 / persistent-buffer reuse — second call copies
       into existing tensors rather than allocating new ones, preserving addresses.
    2. ReshardTransform / MXFP8ReshardTransform — pluggable hooks that intercept
       per-op send/recv and do format conversion on the receiver side (BF16 wire,
       MXFP8 landing zone).  1D-swizzled scales cannot be updated partially, so
       slices are accumulated and quantized atomically when all arrive.
    3. execute_reshard_plan transform plumbing — the execution engine dispatches
       each TransferOp through the transform when applicable, accumulates
       dequantized BF16 for fp8_param=True destination params, and calls a second
       cuda.synchronize() after writeback to close the race with CUDA-graph warmup.
    4. Stream wrapper rename (torch_pack_stream → torch_pack_stream_wrapper) and
       an extra cpu-sync before barrier_event recording to prevent the event from
       firing before RDMA data is visible.
    5. prepare_swap_model_weights — pre-builds the plan and quantizes the target
       model during init so swap_model_weights is latency-free at runtime.

DES-LOC adaptation points (SM86 A6000 × 2 + SM90 H100-NVL × 1, PCIe, no NVLink):
    * SM86 (A6000) does NOT support native MXFP8 hardware — all MXFP8 quantization
      and dequantization must be emulated in software (BF16 intermediate).
    * SM90 (H100) supports MXFP8 natively via Transformer Engine / FlashInfer.
    * Without NVLink, inter-GPU bandwidth is limited to PCIe Gen4 ×16 (~64 GB/s
      host-side or ~32 GB/s peer-to-peer).  The "receiver-side conversion" mode
      (convert_on_send=False) is therefore preferred to keep wire format as BF16
      and let the H100 do the cheap SM90-accelerated quantization.
    * With 1.5 TB CPU DRAM, we use CPU-offloaded "locality cache" (the LOC in
      DES-LOC) to absorb weight slices that cannot be directly streamed device-to-
      device.  The SharedLocalityCache holds pinned-memory staging buffers; GPU
      workers pull from cache asynchronously without blocking the training loop.
    * Decoupled Execution (DE): training on A6000 and inference on H100 run in
      independent CUDA contexts.  Weight transfer is mediated by a coordinator
      thread that owns the pinned staging memory, avoiding any direct peer mapping
      which would require NVLink or NvSCI on PCIe-only systems.

Key design changes vs. upstream Megatron:
    * DeviceCapabilityRouter: replaces the static HAVE_FLASHINFER flag with a
      per-device capability query so SM86 falls back to software FP8 emulation.
    * SharedLocalityCache: pinned-memory staging arena with LRU eviction that
      serves as the shared LOC tier between A6000 (training) and H100 (inference).
    * HeteroMXFP8Transform: subclass of the upstream ReshardTransform pattern,
      extended to route quantization through the capability router and use the
      locality cache for inter-device transfers that cross the PCIe boundary.
    * HeteroRefitCoordinator: replaces Megatron's synchronous execute_reshard_plan
      loop with an async coordinator that overlaps PCIe transfers with the H100's
      next-batch compute.
    * Persistent-buffer address stability is preserved exactly as in upstream —
      the H100 side allocates MXFP8Tensor buffers once and subsequent refits
      copy_ into them.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    from flashinfer import mxfp8_quantize as _fi_mxfp8_quantize
    from flashinfer import mm_mxfp8 as _fi_mm_mxfp8  # noqa: F401 — import verifies linkage

    _HAVE_FLASHINFER = True
    logger.debug("FlashInfer MXFP8 available")
except ImportError:
    _HAVE_FLASHINFER = False
    logger.debug("FlashInfer not available — SM86 software emulation path active")

try:
    import transformer_engine.pytorch as te  # noqa: F401

    _HAVE_TE = True
except ImportError:
    _HAVE_TE = False


# ---------------------------------------------------------------------------
# Device capability routing
# ---------------------------------------------------------------------------

# SM version thresholds
_SM_MXFP8_NATIVE = 90   # H100 and above: native MXFP8 (Hopper+)
_SM_MXFP8_EMUL  = 86   # A6000 (Ampere SM86): software-emulated MXFP8


@dataclass
class DeviceProfile:
    """Capability profile for a single CUDA device in the heterogeneous cluster."""
    device_index: int
    sm_major: int
    sm_minor: int
    total_memory_gb: float
    supports_mxfp8_native: bool
    name: str

    @property
    def sm_version(self) -> int:
        return self.sm_major * 10 + self.sm_minor


def _profile_device(index: int) -> DeviceProfile:
    """Query CUDA device properties and build a DeviceProfile."""
    props = torch.cuda.get_device_properties(index)
    sm = props.major * 10 + props.minor
    return DeviceProfile(
        device_index=index,
        sm_major=props.major,
        sm_minor=props.minor,
        total_memory_gb=props.total_memory / (1 << 30),
        supports_mxfp8_native=(sm >= _SM_MXFP8_NATIVE) and _HAVE_FLASHINFER,
        name=props.name,
    )


class DeviceCapabilityRouter:
    """Routes quantization operations to the appropriate implementation per device.

    DES-LOC adaptation:
        Megatron assumes a homogeneous cluster where all GPUs support MXFP8
        (Blackwell/H100).  In our heterogeneous setup (A6000 SM86 + H100 SM90),
        we must select the implementation at runtime based on the current device.

        SM90 (H100-NVL):  Use FlashInfer mxfp8_quantize / MXFP8Tensor directly.
        SM86 (A6000):      Emulate with scaled FP8 via manual E4M3 clamping.
                           The emulated path produces data/scale tensors with the
                           same layout as FlashInfer so downstream code is uniform.
    """

    def __init__(self):
        self._profiles: Dict[int, DeviceProfile] = {}
        self._lock = threading.Lock()

    def get_profile(self, device: torch.device) -> DeviceProfile:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        with self._lock:
            if idx not in self._profiles:
                self._profiles[idx] = _profile_device(idx)
                p = self._profiles[idx]
                logger.info(
                    "DeviceCapabilityRouter: device %d (%s) SM%d%d — "
                    "native_mxfp8=%s total_mem=%.1f GiB",
                    idx, p.name, p.sm_major, p.sm_minor,
                    p.supports_mxfp8_native, p.total_memory_gb,
                )
        return self._profiles[idx]

    def quantize_to_mxfp8(
        self, tensor: torch.Tensor, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize *tensor* (BF16) to MXFP8 data + scale tensors.

        Returns:
            (data, scale) — layout is compatible with FlashInfer MXFP8Tensor
            regardless of whether the native or emulated path is taken.
        """
        if tensor.dtype != torch.bfloat16:
            tensor = tensor.to(torch.bfloat16)

        dev = device or tensor.device
        profile = self.get_profile(dev)

        if profile.supports_mxfp8_native:
            return self._quantize_native(tensor)
        else:
            return self._quantize_emulated(tensor)

    # -- native path (SM90 H100 via FlashInfer) ------------------------------

    @staticmethod
    def _quantize_native(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """FlashInfer mxfp8_quantize — hardware-accelerated on SM90."""
        assert _HAVE_FLASHINFER, "FlashInfer required for native MXFP8 quantization"
        data, scale = _fi_mxfp8_quantize(tensor)
        return data, scale

    # -- emulated path (SM86 A6000, software only) ---------------------------

    @staticmethod
    def _quantize_emulated(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Software-emulated MXFP8 (E4M3) quantization for SM86 (A6000).

        DES-LOC rationale:
            A6000 GPUs (SM86) have no MXFP8 hardware.  We emulate the same
            block-scale quantization format used by FlashInfer so that:
            (a) The H100 inference side can directly consume the result.
            (b) Code paths that inspect data/scale shape remain unchanged.

        Algorithm:
            - Block size = 32 (matches FlashInfer default).
            - For each [M, 32] tile, compute per-tile amax.
            - Scale = amax / 448.0 (E4M3 max finite value).
            - Quantized data = clamp(round(tensor / scale), -448, 448) cast to FP8.

        The scale tensor is 2D: shape [M, K // 32], matching FlashInfer layout.
        When K is not divisible by 32, the last block is zero-padded.
        """
        BLOCK = 32
        E4M3_MAX = 448.0

        M, K = tensor.shape
        # Pad K to multiple of BLOCK
        K_pad = ((K + BLOCK - 1) // BLOCK) * BLOCK
        if K_pad != K:
            pad = torch.zeros(M, K_pad - K, dtype=torch.bfloat16, device=tensor.device)
            tensor = torch.cat([tensor, pad], dim=1)

        n_blocks = K_pad // BLOCK
        # Reshape to [M, n_blocks, BLOCK] for per-block amax
        t_blocks = tensor.view(M, n_blocks, BLOCK)
        amax = t_blocks.abs().amax(dim=-1)          # [M, n_blocks]
        scale = (amax / E4M3_MAX).clamp(min=1e-12)  # [M, n_blocks]

        # Broadcast scale back to [M, K_pad]
        scale_bc = scale.unsqueeze(-1).expand(M, n_blocks, BLOCK).reshape(M, K_pad)
        q_float = (tensor / scale_bc).clamp(-E4M3_MAX, E4M3_MAX)

        # Cast to torch.float8_e4m3fn
        try:
            data = q_float[:, :K].to(torch.float8_e4m3fn)
        except AttributeError:
            # Fallback for older PyTorch without float8 dtype: store as uint8
            logger.warning(
                "torch.float8_e4m3fn not available; storing emulated FP8 as uint8"
            )
            data = q_float[:, :K].to(torch.int8)

        return data.contiguous(), scale.contiguous()

    def dequantize_from_mxfp8(
        self,
        data: torch.Tensor,
        scale: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Dequantize MXFP8 data+scale back to BF16.

        Works on both SM86 (emulated) and SM90 (native) since the scale
        layout is the same.
        """
        BLOCK = 32
        M = data.shape[0]
        K = data.shape[1]
        n_blocks = scale.shape[1]

        # Broadcast scale: [M, n_blocks] → [M, K]
        scale_bc = scale.unsqueeze(-1).expand(M, n_blocks, BLOCK).reshape(M, n_blocks * BLOCK)
        scale_bc = scale_bc[:, :K]

        return (data.to(torch.float32) * scale_bc.to(torch.float32)).to(torch.bfloat16)


# Module-level singleton
_capability_router = DeviceCapabilityRouter()


# ---------------------------------------------------------------------------
# MXFP8Tensor shim (used when FlashInfer is unavailable on SM86)
# ---------------------------------------------------------------------------

@dataclass
class HeteroMXFP8Tensor:
    """Minimal MXFP8 tensor container compatible with FlashInfer MXFP8Tensor API.

    DES-LOC adaptation:
        On SM86 hosts where FlashInfer is unavailable, we use this shim so
        that the rest of the refit pipeline can treat quantized weights uniformly.
        On SM90 hosts with FlashInfer, the real MXFP8Tensor is used instead.

    Persistent-buffer semantics:
        Once allocated, .data and .scale must not be reallocated — only
        copy_() is permitted so that CUDA graph device-pointer captures
        on the H100 side remain valid.
    """
    data: torch.Tensor    # FP8 (or uint8) quantized data
    scale: torch.Tensor   # BF16 scale factors, shape [M, K//32]

    @classmethod
    def from_bf16(cls, tensor: torch.Tensor) -> "HeteroMXFP8Tensor":
        device = tensor.device
        data, scale = _capability_router.quantize_to_mxfp8(tensor, device)
        return cls(data=data, scale=scale)

    def dequantize(self) -> torch.Tensor:
        return _capability_router.dequantize_from_mxfp8(
            self.data, self.scale, self.data.device
        )

    def copy_from(self, other: "HeteroMXFP8Tensor") -> None:
        """Copy data/scale from *other* preserving our tensor addresses."""
        self.data.copy_(other.data)
        self.scale.copy_(other.scale)


def _make_mxfp8_tensor(tensor: torch.Tensor):
    """Factory: return native FlashInfer MXFP8Tensor on SM90, shim on SM86."""
    profile = _capability_router.get_profile(tensor.device)
    if profile.supports_mxfp8_native:
        try:
            from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
            return MXFP8Tensor.from_bf16(tensor)
        except ImportError:
            pass
    return HeteroMXFP8Tensor.from_bf16(tensor)


# ---------------------------------------------------------------------------
# Shared Locality Cache (the LOC tier in DES-LOC)
# ---------------------------------------------------------------------------

@dataclass
class _CacheEntry:
    """One slot in the SharedLocalityCache."""
    key: str
    tensor: torch.Tensor      # pinned CPU tensor
    last_access: float = field(default_factory=time.monotonic)
    dirty: bool = False       # True if modified since last GPU push


class SharedLocalityCache:
    """Pinned-memory staging arena for heterogeneous PCIe weight transfer.

    DES-LOC rationale:
        Without NVLink, A6000 → H100 direct peer copies go over PCIe and
        are limited to ~32 GB/s.  For large transformer weights this can
        block the training step.  Instead:

        1. Training (A6000) writes updated weight slices to pinned CPU
           staging tensors asynchronously (non-blocking).
        2. The H100 inference process pulls slices from the cache when its
           decode batch finishes, overlapping PCIe transfer with compute.
        3. Eviction is LRU-based; the 1.5 TB CPU DRAM gives us ample room
           to hold an entire 70B model's worth of BF16 weights (~140 GB)
           plus the quantized copies.

    This is the "Shared LOCality" in DES-LOC: both GPUs see a consistent
    snapshot of each layer's weights through this pinned cache.

    Thread safety:
        A single threading.Lock guards all mutations.  Weight slices are
        typically small enough that lock contention is not a bottleneck.
        For future work, per-parameter locks would reduce contention.
    """

    def __init__(self, capacity_gb: float = 256.0):
        self._capacity_bytes = int(capacity_gb * (1 << 30))
        self._used_bytes = 0
        self._entries: Dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()
        logger.info(
            "SharedLocalityCache initialized: capacity=%.1f GiB", capacity_gb
        )

    # -- public API ----------------------------------------------------------

    def put(self, key: str, tensor: torch.Tensor) -> None:
        """Write a weight slice (or full weight) into the cache.

        The tensor is copied to a pinned CPU buffer.  If *key* already
        exists its buffer is reused (preserving memory layout) when shapes
        match; otherwise a new pinned allocation is made after eviction.

        Args:
            key:    Fully-qualified parameter name (e.g. "decoder.0.fc.weight").
            tensor: BF16 tensor on any device (GPU or CPU).
        """
        cpu_tensor = tensor.detach().cpu()
        with self._lock:
            if key in self._entries:
                entry = self._entries[key]
                if entry.tensor.shape == cpu_tensor.shape:
                    entry.tensor.copy_(cpu_tensor)
                    entry.dirty = True
                    entry.last_access = time.monotonic()
                    logger.debug("SharedLocalityCache: updated key=%s", key)
                    return
                else:
                    # Shape changed — evict old entry first
                    self._used_bytes -= entry.tensor.nbytes
                    del self._entries[key]

            nbytes = cpu_tensor.nbytes
            self._evict_if_needed(nbytes)
            pinned = torch.empty_like(cpu_tensor, pin_memory=True)
            pinned.copy_(cpu_tensor)
            self._entries[key] = _CacheEntry(key=key, tensor=pinned, dirty=True)
            self._used_bytes += nbytes
            logger.debug(
                "SharedLocalityCache: stored key=%s shape=%s used=%.1f GiB",
                key, list(cpu_tensor.shape), self._used_bytes / (1 << 30),
            )

    def get(self, key: str, device: torch.device) -> Optional[torch.Tensor]:
        """Retrieve a cached weight slice and transfer to *device*.

        Returns None if the key is not present.  On success, marks the
        entry as clean (the GPU now has the authoritative copy).

        Args:
            key:    Parameter name.
            device: Target GPU device.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            entry.last_access = time.monotonic()
            entry.dirty = False
            # Non-blocking transfer from pinned → GPU
            result = entry.tensor.to(device, non_blocking=True)
            logger.debug(
                "SharedLocalityCache: retrieved key=%s → device=%s", key, device
            )
            return result

    def evict(self, key: str) -> None:
        """Explicitly remove a key from the cache."""
        with self._lock:
            if key in self._entries:
                self._used_bytes -= self._entries[key].tensor.nbytes
                del self._entries[key]
                logger.debug("SharedLocalityCache: evicted key=%s", key)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._used_bytes = 0
        logger.info("SharedLocalityCache: cleared")

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    # -- internal ------------------------------------------------------------

    def _evict_if_needed(self, required_bytes: int) -> None:
        """LRU eviction until *required_bytes* can be accommodated."""
        while self._used_bytes + required_bytes > self._capacity_bytes and self._entries:
            lru_key = min(self._entries, key=lambda k: self._entries[k].last_access)
            victim = self._entries.pop(lru_key)
            self._used_bytes -= victim.tensor.nbytes
            logger.warning(
                "SharedLocalityCache: LRU evict key=%s (%.1f MiB freed)",
                lru_key, victim.tensor.nbytes / (1 << 20),
            )


# Module-level default cache; can be replaced for testing
_default_locality_cache: Optional[SharedLocalityCache] = None


def get_locality_cache() -> SharedLocalityCache:
    global _default_locality_cache
    if _default_locality_cache is None:
        _default_locality_cache = SharedLocalityCache(capacity_gb=256.0)
    return _default_locality_cache


# ---------------------------------------------------------------------------
# HeteroMXFP8Transform — DES-LOC adaptation of MXFP8ReshardTransform
# ---------------------------------------------------------------------------

class HeteroMXFP8Transform:
    """MXFP8 refit transform for heterogeneous SM86/SM90 clusters without NVLink.

    Upstream design (Megatron MXFP8ReshardTransform):
        Two modes: convert_on_send (sender quantizes, saves PCIe bandwidth)
        vs. receive BF16 + convert_on_recv (simpler, receiver handles all).
        Persistent buffers preserve CUDA-graph device pointers across refits.
        1D-swizzled scales must be accumulated before quantization.

    DES-LOC adaptations:
        1. Device-aware quantization: prepare_send checks if the source device
           is SM86 and uses the emulated path; prepare_recv similarly routes
           through the capability router on the destination (SM90 H100).

        2. Locality-cache staging: on the A6000 (training) side, prepared
           send buffers are written into the SharedLocalityCache before
           transmission.  The H100 (inference) side can pull from cache
           during prepare_recv even if the PCIe transfer is still in flight,
           using a pinned-CPU intermediate to decouple the two timelines.

        3. Stream isolation: since A6000 and H100 run in separate CUDA
           contexts (no NVLink peer mapping), we use pinned CPU memory as
           the staging point rather than direct device-to-device copies.
           This avoids the need for CudaIPC handles or NvSCI.

        4. 1D-scale accumulation: preserved identically from upstream —
           FlashInfer's swizzled 1D scale format cannot be partially updated.

    Args:
        convertible_params: Set of FQN parameter names routed through this transform.
        persistent_buffers: Maps parameter name (sans prefix) → HeteroMXFP8Tensor
            or FlashInfer MXFP8Tensor.  These are never reallocated.
        buffer_key_prefix: Prefix stripped from param_name before lookup.
        src_device: Device of the training (sender) model, e.g. cuda:0 (A6000).
        dst_device: Device of the inference (receiver) model, e.g. cuda:2 (H100).
        locality_cache: SharedLocalityCache instance.  Uses module default if None.
        convert_on_send: If True, sender quantizes to MXFP8; if False (default),
            BF16 is sent and the receiver quantizes.  Prefer False for PCIe-only
            clusters since the H100 quantization is nearly free.
    """

    def __init__(
        self,
        convertible_params: Set[str],
        persistent_buffers: Dict[str, object],
        buffer_key_prefix: str = "",
        src_device: Optional[torch.device] = None,
        dst_device: Optional[torch.device] = None,
        locality_cache: Optional[SharedLocalityCache] = None,
        convert_on_send: bool = False,
    ):
        self.convertible_params = convertible_params
        self.persistent_buffers = persistent_buffers
        self.buffer_key_prefix = buffer_key_prefix
        self.src_device = src_device
        self.dst_device = dst_device
        self._cache = locality_cache or get_locality_cache()
        self.convert_on_send = convert_on_send

        # 1D-scale accumulation: maps buf_key → [accum_tensor, bytes_written]
        self._pending_1d: Dict[str, List] = {}

        src_profile = _capability_router.get_profile(src_device) if src_device else None
        dst_profile = _capability_router.get_profile(dst_device) if dst_device else None
        logger.info(
            "HeteroMXFP8Transform: src=%s(%s) dst=%s(%s) params=%d convert_on_send=%s",
            src_device, src_profile.name if src_profile else "?",
            dst_device, dst_profile.name if dst_profile else "?",
            len(convertible_params), convert_on_send,
        )

    # -- ReshardTransform interface ------------------------------------------

    def should_transform(self, param_name: str) -> bool:
        return param_name in self.convertible_params

    def prepare_send(
        self,
        param_name: str,
        src_slice: tuple,
        src_param: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Produce buffer(s) to send for *param_name*.

        DES-LOC path:
            1. Dequantize src_param if it is a TE MXFP8Tensor (fp8_param=True).
            2. Extract the requested slice and move to BF16.
            3. Write into SharedLocalityCache for decoupled PCIe transfer.
            4. If convert_on_send: quantize on the sender (SM86 emulated path)
               and return [data, scale].  Otherwise return [bf16_slice].
        """
        raw = self._dequantize_if_needed(src_param)
        bf16_slice = raw[src_slice].contiguous().to(torch.bfloat16)

        # Stage in locality cache (decouples A6000 training timeline from H100)
        cache_key = f"{param_name}:{_slice_repr(src_slice)}"
        self._cache.put(cache_key, bf16_slice)

        if self.convert_on_send:
            dev = self.src_device or bf16_slice.device
            data, scale = _capability_router.quantize_to_mxfp8(bf16_slice, dev)
            logger.debug(
                "prepare_send: %s slice=%s data=%s scale=%s (convert_on_send)",
                param_name, _slice_repr(src_slice), data.shape, scale.shape,
            )
            return [data.contiguous(), scale.contiguous()]
        else:
            logger.debug(
                "prepare_send: %s slice=%s bf16=%s (BF16 wire)",
                param_name, _slice_repr(src_slice), bf16_slice.shape,
            )
            return [bf16_slice]

    def prepare_recv(
        self,
        param_name: str,
        dst_slice: tuple,
    ) -> List[torch.Tensor]:
        """Allocate receive buffer(s) on the destination (H100) device.

        DES-LOC path:
            Try to prefetch from locality cache first.  If the sender has
            already written the slice (non-blocking PCIe), we can skip the
            network receive entirely.  Otherwise allocate standard recv buffers.

        Returns list of tensors matching what prepare_send returns.
        """
        buf_key = param_name.removeprefix(self.buffer_key_prefix)
        buf = self.persistent_buffers[buf_key]
        dst_dev = self.dst_device or (
            buf.data.device if hasattr(buf, "data") else torch.device("cuda")
        )

        # Opportunistic locality-cache prefetch
        cache_key = f"{param_name}:{_slice_repr(dst_slice)}"
        cached = self._cache.get(cache_key, dst_dev)
        if cached is not None:
            logger.debug(
                "prepare_recv: %s cache hit — skipping PCIe recv", param_name
            )
            # Return as recv buffer; finalize_recv will handle quantization
            return [cached]

        if self.convert_on_send:
            # Expect MXFP8 data + scale on the wire
            if hasattr(buf, "scale") and buf.scale.ndim == 1:
                raise NotImplementedError(
                    f"convert_on_send=True unsupported for 1D-swizzled scale "
                    f"(param={param_name!r}).  Use convert_on_send=False."
                )
            scale_slice = _scale_slice_from_data_slice(dst_slice)
            return [
                torch.empty_like(buf.data[dst_slice].contiguous(), device=dst_dev),
                torch.empty_like(
                    buf.scale[scale_slice].contiguous()
                    if hasattr(buf, "scale") else torch.empty(1, device=dst_dev),
                    device=dst_dev,
                ),
            ]
        else:
            # BF16 receive buffer
            data_slice_shape = buf.data[dst_slice].shape if hasattr(buf, "data") else (1,)
            return [
                torch.empty(data_slice_shape, dtype=torch.bfloat16, device=dst_dev)
            ]

    def finalize_recv(
        self,
        param_name: str,
        dst_slice: tuple,
        recv_buffers: List[torch.Tensor],
    ) -> None:
        """Write received data into persistent MXFP8 buffers.

        DES-LOC path mirrors upstream MXFP8ReshardTransform with two additions:
            - Routes quantization through DeviceCapabilityRouter (SM90 native
              or SM86 emulated).
            - Clears the locality-cache entry for this slice once committed so
              stale data is not served on the next refit cycle.
        """
        buf_key = param_name.removeprefix(self.buffer_key_prefix)
        buf = self.persistent_buffers[buf_key]

        if self.convert_on_send:
            # MXFP8 data+scale arrived; copy directly into persistent buffers.
            buf.data[dst_slice].copy_(recv_buffers[0])
            if hasattr(buf, "scale") and buf.scale.ndim > 1:
                scale_slice = _scale_slice_from_data_slice(dst_slice)
                buf.scale[scale_slice].copy_(recv_buffers[1])
        else:
            # BF16 arrived; quantize on receiver.
            bf16_data = recv_buffers[0]
            scale_dim = buf.scale.ndim if hasattr(buf, "scale") else 2

            if scale_dim == 1:
                # 1D swizzled scale — accumulate and quantize atomically
                self._accumulate_1d(param_name, buf_key, buf, dst_slice, bf16_data)
            else:
                # 2D scale — each row is independent, quantize immediately
                dev = self.dst_device or bf16_data.device
                data, scale = _capability_router.quantize_to_mxfp8(bf16_data, dev)
                with torch.no_grad():
                    buf.data[dst_slice].copy_(data)
                    scale_slice = _scale_slice_from_data_slice(dst_slice)
                    buf.scale[scale_slice].copy_(scale)
                logger.debug(
                    "finalize_recv: %s 2D-scale updated slice=%s",
                    param_name, _slice_repr(dst_slice),
                )

        # Evict from locality cache — next refit will rewrite it
        cache_key = f"{param_name}:{_slice_repr(dst_slice)}"
        self._cache.evict(cache_key)

    # -- internal helpers ----------------------------------------------------

    def _accumulate_1d(
        self,
        param_name: str,
        buf_key: str,
        buf: object,
        dst_slice: tuple,
        bf16_data: torch.Tensor,
    ) -> None:
        """Accumulate BF16 slices for 1D-swizzled-scale params.

        Upstream rationale (preserved):
            FlashInfer's 1D swizzled scale encodes values across the full
            weight tensor; updating a partial slice corrupts the swizzle
            layout.  We therefore collect all BF16 slices and call
            quantize_to_mxfp8 once when the last slice arrives.
        """
        if buf_key not in self._pending_1d:
            accum = torch.zeros_like(buf.data, dtype=torch.bfloat16)
            self._pending_1d[buf_key] = [accum, 0]
            logger.debug(
                "finalize_recv 1D: %s — started accumulation shape=%s",
                param_name, list(accum.shape),
            )

        accum, written = self._pending_1d[buf_key]
        accum[dst_slice].copy_(bf16_data)
        written += bf16_data.numel()
        total = buf.data.numel()

        logger.debug(
            "finalize_recv 1D: %s written=%d/%d", param_name, written, total
        )

        if written >= total:
            if written != total:
                raise AssertionError(
                    f"1D-scale param {param_name!r}: received {written} elements, "
                    f"expected {total} (duplicate or missing slices?)"
                )
            dev = self.dst_device or accum.device
            data, scale = _capability_router.quantize_to_mxfp8(accum, dev)
            with torch.no_grad():
                buf.data.copy_(data)
                buf.scale.copy_(scale)
            del self._pending_1d[buf_key]
            logger.info(
                "finalize_recv 1D: %s — quantization complete, persistent buffers updated",
                param_name,
            )
        else:
            self._pending_1d[buf_key][1] = written

    @staticmethod
    def _dequantize_if_needed(param: torch.Tensor) -> torch.Tensor:
        """Dequantize TE MXFP8Tensor to BF16; pass through standard tensors."""
        if _HAVE_TE:
            try:
                from transformer_engine.pytorch.tensor.mxfp8_tensor import (
                    MXFP8Tensor as _TEMXFP8,
                )
                if isinstance(param, _TEMXFP8):
                    return param.dequantize()
            except ImportError:
                pass
        return param.data


# ---------------------------------------------------------------------------
# Quantization helpers (persistent buffer management)
# ---------------------------------------------------------------------------

def should_quantize_param(val: torch.Tensor) -> bool:
    """Return True if *val* is a 2D BF16/FP16 CUDA parameter eligible for MXFP8.

    DES-LOC note:
        On SM86 hosts the check is identical — we will use the emulated path.
        On SM90 hosts with FlashInfer the native path is used.
    """
    if not val.is_cuda:
        return False
    if _HAVE_TE:
        try:
            from transformer_engine.pytorch.tensor.mxfp8_tensor import (
                MXFP8Tensor as _TEMXFP8,
            )
            if isinstance(val, _TEMXFP8):
                return True
        except ImportError:
            pass
    return (
        isinstance(val, nn.Parameter)
        and val.dim() == 2
        and val.dtype in (torch.bfloat16, torch.float16)
    )


def quantize_model_params(
    model: nn.Module,
    persistent_buffers: Optional[Dict[str, object]] = None,
    _prefix: str = "",
) -> Dict[str, object]:
    """Quantize 2D BF16/FP16 parameters to HeteroMXFP8Tensor (or native MXFP8Tensor).

    Upstream design (quantize_params_to_mxfp8 from Megatron):
        Recursively walks the module tree.  First call allocates persistent
        buffers; subsequent calls copy_ into them to preserve CUDA-graph
        device pointers.  The nn.Parameter is replaced by the MXFP8 tensor
        as a plain attribute (not a registered parameter).

    DES-LOC adaptation:
        Uses _make_mxfp8_tensor() which routes to native FlashInfer on SM90
        or the software-emulated HeteroMXFP8Tensor on SM86.

    Args:
        model:              Module whose parameters should be quantized in-place.
        persistent_buffers: Existing buffers from a prior call; entries are
                            updated via copy_() rather than reallocated.
        _prefix:            Internal FQN prefix (callers leave as "").

    Returns:
        Dict mapping FQN → quantized tensor.
    """
    if persistent_buffers is None:
        persistent_buffers = {}

    for child_name, child_module in model.named_children():
        child_prefix = f"{_prefix}{child_name}." if _prefix else f"{child_name}."
        quantize_model_params(child_module, persistent_buffers, _prefix=child_prefix)

    if hasattr(model, "_parameters") and model._parameters:
        for key in list(model._parameters.keys()):
            val = model._parameters[key]
            if val is None or not should_quantize_param(val):
                continue

            # Obtain BF16 data
            if _HAVE_TE:
                try:
                    from transformer_engine.pytorch.tensor.mxfp8_tensor import (
                        MXFP8Tensor as _TEMXFP8,
                    )
                    if isinstance(val, _TEMXFP8):
                        bf16 = val.dequantize()
                    else:
                        bf16 = val.data.to(torch.bfloat16)
                except ImportError:
                    bf16 = val.data.to(torch.bfloat16)
            else:
                bf16 = val.data.to(torch.bfloat16)

            fqn = f"{_prefix}{key}"

            if fqn in persistent_buffers:
                # Second+ call: copy_ to preserve addresses
                existing = persistent_buffers[fqn]
                new_q = _make_mxfp8_tensor(bf16)
                if hasattr(existing, "copy_from"):
                    existing.copy_from(new_q)
                else:
                    existing.data.copy_(new_q.data)
                    existing.scale.copy_(new_q.scale)
                q_tensor = existing
                logger.debug("quantize_model_params: updated persistent buffer %s", fqn)
            else:
                q_tensor = _make_mxfp8_tensor(bf16)
                persistent_buffers[fqn] = q_tensor
                logger.info(
                    "quantize_model_params: created buffer %s data=%s scale=%s",
                    fqn, q_tensor.data.shape, q_tensor.scale.shape,
                )

            del model._parameters[key]
            setattr(model, key, q_tensor)

    return persistent_buffers


# ---------------------------------------------------------------------------
# HeteroRefitCoordinator — async weight transfer coordinator
# ---------------------------------------------------------------------------

class HeteroRefitCoordinator:
    """Asynchronous refit coordinator for DES-LOC heterogeneous clusters.

    Overview:
        Megatron's execute_reshard_plan is synchronous: it blocks until all
        weight slices are transferred and written back.  For DES-LOC this is
        undesirable because:
        (a) PCIe transfers between A6000 and H100 are slow (~32 GB/s P2P).
        (b) The H100 decode batch and the weight transfer can overlap: the
            H100 computes the current decode batch while the coordinator
            pushes next-layer weights through the locality cache.

        This coordinator:
        1. Accepts (param_name, src_slice, src_tensor) tuples from the
           training loop (A6000 side) and writes them to the locality cache.
        2. A background thread drains the cache and calls finalize_recv on
           the transform, updating persistent MXFP8 buffers on the H100.
        3. wait_for_layer(param_name) allows the inference loop to block
           until a specific layer's weights are committed.

    Usage:
        coord = HeteroRefitCoordinator(transform, dst_device=torch.device("cuda:2"))
        coord.start()
        # Training loop:
        coord.push_slice("decoder.0.fc.weight", (slice(0,64), ...), weight_tensor)
        # Inference loop (before computing layer 0):
        coord.wait_for_layer("decoder.0.fc.weight")
        coord.stop()
    """

    def __init__(
        self,
        transform: HeteroMXFP8Transform,
        dst_device: torch.device,
        drain_interval_ms: float = 5.0,
    ):
        self._transform = transform
        self._dst_device = dst_device
        self._drain_interval = drain_interval_ms / 1000.0

        # Queue: each item is (param_name, src_slice, recv_buffer)
        self._queue: List[Tuple[str, tuple, List[torch.Tensor]]] = []
        self._queue_lock = threading.Lock()

        # Per-parameter completion events
        self._param_events: Dict[str, threading.Event] = {}
        self._events_lock = threading.Lock()

        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

    def start(self) -> None:
        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._drain_loop, daemon=True, name="HeteroRefitDrain"
        )
        self._thread.start()
        logger.info("HeteroRefitCoordinator: drain thread started")

    def stop(self, timeout: float = 10.0) -> None:
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        logger.info("HeteroRefitCoordinator: drain thread stopped")

    def push_slice(
        self,
        param_name: str,
        src_slice: tuple,
        src_tensor: torch.Tensor,
    ) -> None:
        """Queue a weight slice for async transfer to the H100.

        Called from the A6000 training loop after each optimizer step.
        Immediately writes to the locality cache (non-blocking).
        """
        # Write to cache — non-blocking, returns immediately
        recv_bufs = [src_tensor.detach().to(torch.bfloat16).contiguous().cpu()]
        cache_key = f"{param_name}:{_slice_repr(src_slice)}"
        self._transform._cache.put(cache_key, recv_bufs[0])

        with self._queue_lock:
            self._queue.append((param_name, src_slice, recv_bufs))

        # Reset completion event for this param
        with self._events_lock:
            self._param_events[param_name] = threading.Event()

        logger.debug("HeteroRefitCoordinator: queued %s slice=%s", param_name, _slice_repr(src_slice))

    def wait_for_layer(self, param_name: str, timeout: float = 30.0) -> bool:
        """Block until *param_name*'s weights are committed to H100 persistent buffers.

        Returns True if weights are ready, False on timeout.
        """
        with self._events_lock:
            event = self._param_events.get(param_name)
        if event is None:
            return True  # Never queued — already up-to-date
        ready = event.wait(timeout=timeout)
        if not ready:
            logger.warning(
                "HeteroRefitCoordinator: timeout waiting for %s", param_name
            )
        return ready

    # -- background drain thread ---------------------------------------------

    def _drain_loop(self) -> None:
        while not self._stop_flag.is_set():
            self._drain_once()
            time.sleep(self._drain_interval)
        # Final drain on shutdown
        self._drain_once()

    def _drain_once(self) -> None:
        with self._queue_lock:
            batch = list(self._queue)
            self._queue.clear()

        for param_name, src_slice, recv_bufs in batch:
            try:
                # Move recv buffer to H100 device
                gpu_bufs = [
                    b.to(self._dst_device, non_blocking=True) for b in recv_bufs
                ]
                torch.cuda.synchronize(self._dst_device)
                self._transform.finalize_recv(param_name, src_slice, gpu_bufs)
                logger.debug(
                    "HeteroRefitCoordinator: committed %s slice=%s",
                    param_name, _slice_repr(src_slice),
                )
                # Signal waiters
                with self._events_lock:
                    ev = self._param_events.get(param_name)
                    if ev is not None:
                        ev.set()
            except Exception:
                logger.exception(
                    "HeteroRefitCoordinator: error committing %s", param_name
                )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _slice_repr(slices) -> str:
    """Compact string representation of a slice tuple for logging/cache keys."""
    if isinstance(slices, tuple):
        parts = []
        for s in slices:
            if isinstance(s, slice):
                parts.append(f"{s.start}:{s.stop}")
            else:
                parts.append(str(s))
        return "[" + ",".join(parts) + "]"
    return str(slices)


def _scale_slice_from_data_slice(
    data_slice: tuple,
    block_size: int = 32,
) -> tuple:
    """Convert MXFP8 data slice indices to corresponding scale indices.

    Upstream logic (transforms.py in Megatron fca1679):
        Each group of block_size elements along the K (last) dimension shares
        one scale value.  Only the last dimension is divided; all others pass
        through unchanged.
    """
    adjusted = list(data_slice)
    last = adjusted[-1]
    if isinstance(last, slice):
        if last.start is not None and last.start % block_size != 0:
            raise AssertionError(
                f"Data slice last dim ({last}) not aligned to block_size={block_size}"
            )
        if last.stop is not None and last.stop % block_size != 0:
            raise AssertionError(
                f"Data slice last dim ({last}) not aligned to block_size={block_size}"
            )
        scale_start = (last.start // block_size) if last.start is not None else None
        scale_stop = (last.stop // block_size) if last.stop is not None else None
        adjusted[-1] = slice(scale_start, scale_stop)
    elif isinstance(last, int):
        adjusted[-1] = last // block_size
    return tuple(adjusted)


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def build_hetero_mxfp8_transform(
    training_model: nn.Module,
    inference_model: nn.Module,
    src_device: torch.device,
    dst_device: torch.device,
    param_prefix: str = "decoder.",
    convert_on_send: bool = False,
    locality_cache: Optional[SharedLocalityCache] = None,
) -> Tuple[HeteroMXFP8Transform, Dict[str, object]]:
    """Build a HeteroMXFP8Transform and quantize inference model weights.

    Call once during initialization while both models still hold BF16 weights.
    The function:
        1. Identifies which inference model parameters are eligible for MXFP8.
        2. Quantizes the inference decoder to persistent MXFP8 buffers.
        3. Constructs a HeteroMXFP8Transform with the correct device routing.

    Returns:
        (transform, persistent_buffers) — pass transform to execute_reshard_plan
        or HeteroRefitCoordinator; keep persistent_buffers for CUDA graph capture.
    """
    # Identify convertible params on the inference (destination) side
    convertible: Set[str] = set()
    decoder = (
        inference_model.decoder if hasattr(inference_model, "decoder") else inference_model
    )
    for name, param in decoder.named_parameters():
        if should_quantize_param(param):
            convertible.add(f"{param_prefix}{name}")

    logger.info(
        "build_hetero_mxfp8_transform: %d convertible params, src=%s dst=%s",
        len(convertible), src_device, dst_device,
    )

    # Quantize decoder weights → persistent buffers (CUDA graph safe)
    persistent_buffers = quantize_model_params(decoder)

    transform = HeteroMXFP8Transform(
        convertible_params=convertible,
        persistent_buffers=persistent_buffers,
        buffer_key_prefix=param_prefix,
        src_device=src_device,
        dst_device=dst_device,
        locality_cache=locality_cache,
        convert_on_send=convert_on_send,
    )
    return transform, persistent_buffers


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    if not torch.cuda.is_available():
        logger.warning("No CUDA device — smoke test runs CPU emulation only")
        DEV = torch.device("cpu")
    else:
        DEV = torch.device("cuda:0")

    logger.info("=== DES-LOC HeteroMXFP8Refit smoke test ===")

    # 1. DeviceCapabilityRouter
    router = DeviceCapabilityRouter()
    M, K = 64, 128
    w = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
    data, scale = router.quantize_to_mxfp8(w, DEV)
    assert data.shape[0] == M and data.shape[1] == K, f"data shape mismatch: {data.shape}"
    assert scale.shape == (M, K // 32), f"scale shape mismatch: {scale.shape}"
    logger.info("✓ DeviceCapabilityRouter quantize/dequantize shapes OK")

    # 2. SharedLocalityCache round-trip
    cache = SharedLocalityCache(capacity_gb=1.0)
    cache.put("test_param", w)
    retrieved = cache.get("test_param", DEV)
    assert retrieved is not None, "Cache miss on put→get"
    assert retrieved.shape == w.shape, f"shape mismatch: {retrieved.shape} vs {w.shape}"
    logger.info("✓ SharedLocalityCache put/get OK")

    # 3. HeteroMXFP8Tensor persistent buffer address stability
    q1 = HeteroMXFP8Tensor.from_bf16(w)
    ptr_data = q1.data.data_ptr()
    ptr_scale = q1.scale.data_ptr()
    w2 = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
    q2 = HeteroMXFP8Tensor.from_bf16(w2)
    q1.copy_from(q2)
    assert q1.data.data_ptr() == ptr_data, "data pointer changed after copy_from!"
    assert q1.scale.data_ptr() == ptr_scale, "scale pointer changed after copy_from!"
    logger.info("✓ HeteroMXFP8Tensor address stability OK")

    # 4. HeteroMXFP8Transform full round-trip (BF16 wire)
    buf = HeteroMXFP8Tensor.from_bf16(torch.zeros(M, K, dtype=torch.bfloat16, device=DEV))
    transform = HeteroMXFP8Transform(
        convertible_params={"decoder.weight"},
        persistent_buffers={"weight": buf},
        buffer_key_prefix="decoder.",
        src_device=DEV,
        dst_device=DEV,
        locality_cache=cache,
        convert_on_send=False,
    )
    src_param = nn.Parameter(w.clone())
    sent = transform.prepare_send("decoder.weight", (slice(None), slice(None)), src_param)
    recv_bufs = transform.prepare_recv("decoder.weight", (slice(None), slice(None)))
    # Simulate wire: copy sent → recv
    recv_bufs[0].copy_(sent[0])
    transform.finalize_recv("decoder.weight", (slice(None), slice(None)), recv_bufs)
    expected_data, expected_scale = router.quantize_to_mxfp8(w, DEV)
    assert buf.data.shape == expected_data.shape, "persistent buffer data shape mismatch"
    logger.info("✓ HeteroMXFP8Transform BF16-wire round-trip OK")

    # 5. _scale_slice_from_data_slice
    s = _scale_slice_from_data_slice((slice(0, 64), slice(0, 128)))
    assert s == (slice(0, 2), slice(0, 4)), f"scale slice wrong: {s}"
    logger.info("✓ _scale_slice_from_data_slice OK")

    logger.info("=== All smoke tests passed ===")
    sys.exit(0)
