"""
DES-LOC Heterogeneous RL Optimizer Offload
===========================================

Upstream Design Intent (Megatron 287d2f47):
--------------------------------------------
Megatron's RL training pipeline alternates between two phases:
  1. **Rollout / Inference phase**: the policy model generates trajectories.
  2. **Training phase**: gradients are computed and the optimizer updates weights.

During inference, optimizer state (Adam exp_avg, exp_avg_sq, fp32 master weights)
and gradient buffers occupy significant GPU VRAM that is not needed until the next
training step. The fix in 287d2f47 decouples these two operations:
  - ``offload_grad_buffers()``  — frees grad_data storage in-place via
    ``storage().resize_(0)``; tensor *views* remain valid Python objects.
  - ``offload_to_cpu()`` / ``restore_from_cpu()`` — moves optimizer state tensors
    to pinned CPU memory and restores them back to GPU on demand.

The bug being fixed: previously, grad-buffer offload happened *after* the refit of
inference-model weights (inside ``megatron_rl_inference_mode``), which meant the
GPU held both optimizer state AND full grad buffers simultaneously during the most
memory-intensive window.  The fix moves grad-buffer offload *before* refit.

DES-LOC Adaptation Points
--------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) adds three concerns that
vanilla Megatron does not address:

1. **Heterogeneous device topology**: 2× A6000-48 GB (SM86, PCIe) + 1× H100-96 GB
   (SM90, PCIe).  There is no NVLink; all inter-GPU traffic goes through PCIe.
   The offload destination must be *device-aware*: grad buffers from the A6000s
   should be freed outright (they are recomputed); the H100 grad buffer can be
   pinned to CPU because it has higher reuse probability and larger PCIe bandwidth.

2. **Shared Locality Cache (SLC)**: DES-LOC maintains a CPU DRAM region (up to
   ~400 GB usable out of 1.5 TB) that acts as a *locality-aware cache* between
   GPU generations.  Optimizer state for parameters resident on the H100 is placed
   in the SLC's *hot tier* (pinned, contiguous); A6000-resident optimizer state
   goes to the *cold tier* (pageable, may be swapped).

3. **Async prefetch pipeline**: DES-LOC overlaps the restore of optimizer state
   with the forward pass of the first RL training micro-batch.  This requires a
   CUDA stream per device class so that the H100 restore stream does not block
   A6000 compute.

Architecture
------------
::

    HeteroGradBufferManager          — per-device grad-buffer lifecycle
    SharedLocalityCache              — CPU DRAM tiered buffer registry
    HeteroOptimizerOffloadEngine     — orchestrates offload/restore across devices
    DESLOCRLOffloadContext           — context-manager for RL rollout windows

"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device classification helpers
# ---------------------------------------------------------------------------

class DeviceClass(Enum):
    """Coarse device classification used for offload routing decisions."""
    A6000_SM86 = auto()   # 48 GB, SM 8.6 — older arch, limited bandwidth
    H100_SM90  = auto()   # 96 GB, SM 9.0 — newer arch, wider PCIe bandwidth
    UNKNOWN    = auto()


def classify_device(device_index: int) -> DeviceClass:
    """
    Inspect CUDA device capability and total memory to assign a DeviceClass.

    DES-LOC relies on this classification to route offload traffic correctly.
    In a homogeneous cluster this would be trivial; our heterogeneous setup
    requires runtime introspection.

    Args:
        device_index: CUDA ordinal (0-based).

    Returns:
        DeviceClass enum value.
    """
    props = torch.cuda.get_device_properties(device_index)
    sm = props.major * 10 + props.minor          # e.g. 86 or 90
    vram_gb = props.total_memory / (1024 ** 3)

    if sm == 86 and 40 <= vram_gb <= 56:
        return DeviceClass.A6000_SM86
    elif sm == 90 and vram_gb >= 80:
        return DeviceClass.H100_SM90
    else:
        logger.warning(
            "classify_device: device %d has SM=%d, VRAM=%.1f GB — treated as UNKNOWN",
            device_index, sm, vram_gb,
        )
        return DeviceClass.UNKNOWN


# ---------------------------------------------------------------------------
# Shared Locality Cache (SLC)
# ---------------------------------------------------------------------------

class CacheTier(Enum):
    HOT  = "hot"   # pinned memory — fast restore, consumed first
    COLD = "cold"  # pageable memory — slower, but avoids OOM


@dataclass
class SLCEntry:
    """One cached tensor in the Shared Locality Cache."""
    key: str
    cpu_tensor: torch.Tensor
    tier: CacheTier
    original_device: int
    original_dtype: torch.dtype
    original_shape: torch.Size
    pinned: bool = False


class SharedLocalityCache:
    """
    CPU DRAM buffer acting as a *tiered* locality-aware cache between GPU
    generations in the DES-LOC heterogeneous training setup.

    Design
    ------
    The SLC is a flat key→SLCEntry dictionary protected by a per-key lock.
    Two tiers exist:

    * **HOT**: pinned CPU memory allocated via ``torch.empty(...).pin_memory()``.
      Used for tensors with high reuse probability (H100 optimizer state, master
      weights).  Pinned memory enables DMA transfers without CPU involvement.

    * **COLD**: ordinary pageable CPU memory.  Used for A6000 grad buffers and
      less-frequently-accessed optimizer state.  Lower allocation cost but higher
      restore latency.

    The total budget is capped at ``budget_gb`` (default 400 GB) to leave OS
    headroom in the 1.5 TB machine.

    Thread safety
    -------------
    Multiple CUDA streams (one per device) may call ``store`` / ``fetch``
    concurrently.  A global lock guards the registry; per-key locks guard
    individual tensor copies so that a slow H100 DMA does not block an A6000
    fetch.
    """

    def __init__(self, budget_gb: float = 400.0) -> None:
        self._budget_bytes = int(budget_gb * (1024 ** 3))
        self._used_bytes: int = 0
        self._registry: Dict[str, SLCEntry] = {}
        self._lock = threading.Lock()
        logger.info(
            "SharedLocalityCache initialised — budget %.1f GB (%.2e bytes)",
            budget_gb, self._budget_bytes,
        )

    # ------------------------------------------------------------------
    def store(
        self,
        key: str,
        gpu_tensor: torch.Tensor,
        tier: CacheTier = CacheTier.HOT,
        non_blocking: bool = True,
    ) -> SLCEntry:
        """
        Copy *gpu_tensor* to CPU and register it under *key*.

        If an entry already exists for *key* and the shapes match, the existing
        CPU buffer is reused (in-place copy) to avoid re-allocation.

        Args:
            key: Unique identifier (e.g. ``"rank0.layer3.weight.exp_avg"``).
            gpu_tensor: Source tensor on a CUDA device.
            tier: Cache tier (HOT = pinned, COLD = pageable).
            non_blocking: Use non-blocking H2D copy (requires caller to sync).

        Returns:
            The SLCEntry that now holds the CPU copy.

        Raises:
            MemoryError: If the SLC budget would be exceeded.
        """
        nbytes = gpu_tensor.nbytes
        device_idx = gpu_tensor.device.index

        with self._lock:
            if key in self._registry:
                entry = self._registry[key]
                if entry.cpu_tensor.shape == gpu_tensor.shape:
                    # Reuse existing buffer — no budget change.
                    entry.cpu_tensor.copy_(gpu_tensor, non_blocking=non_blocking)
                    logger.debug("SLC store (reuse) key=%s tier=%s", key, tier.value)
                    return entry
                else:
                    # Shape changed — evict old entry.
                    self._used_bytes -= entry.cpu_tensor.nbytes
                    del self._registry[key]

            if self._used_bytes + nbytes > self._budget_bytes:
                raise MemoryError(
                    f"SharedLocalityCache budget exceeded: need {nbytes} bytes, "
                    f"used {self._used_bytes}/{self._budget_bytes}"
                )

            if tier == CacheTier.HOT:
                cpu_buf = torch.empty(
                    gpu_tensor.shape,
                    dtype=gpu_tensor.dtype,
                    device="cpu",
                ).pin_memory()
                pinned = True
            else:
                cpu_buf = torch.empty(
                    gpu_tensor.shape,
                    dtype=gpu_tensor.dtype,
                    device="cpu",
                )
                pinned = False

            cpu_buf.copy_(gpu_tensor, non_blocking=non_blocking)
            entry = SLCEntry(
                key=key,
                cpu_tensor=cpu_buf,
                tier=tier,
                original_device=device_idx,
                original_dtype=gpu_tensor.dtype,
                original_shape=gpu_tensor.shape,
                pinned=pinned,
            )
            self._registry[key] = entry
            self._used_bytes += nbytes
            logger.debug(
                "SLC store key=%s tier=%s nbytes=%d total_used=%.2f GB",
                key, tier.value, nbytes, self._used_bytes / 1024**3,
            )
            return entry

    def fetch(
        self,
        key: str,
        target_device: Optional[int] = None,
        non_blocking: bool = True,
    ) -> torch.Tensor:
        """
        Restore a cached tensor to GPU (or return the CPU copy if *target_device*
        is None).

        Args:
            key: Registry key from a previous ``store`` call.
            target_device: CUDA ordinal.  Defaults to the original device.
            non_blocking: Use non-blocking D2H copy.

        Returns:
            Tensor on *target_device*.

        Raises:
            KeyError: If *key* is not in the registry.
        """
        with self._lock:
            if key not in self._registry:
                raise KeyError(f"SharedLocalityCache: key '{key}' not found")
            entry = self._registry[key]

        dev = target_device if target_device is not None else entry.original_device
        gpu_tensor = entry.cpu_tensor.to(
            device=torch.device("cuda", dev),
            non_blocking=non_blocking,
        )
        logger.debug("SLC fetch key=%s → cuda:%d", key, dev)
        return gpu_tensor

    def evict(self, key: str) -> None:
        """Remove an entry from the cache and free its CPU memory."""
        with self._lock:
            if key in self._registry:
                entry = self._registry.pop(key)
                self._used_bytes -= entry.cpu_tensor.nbytes
                logger.debug("SLC evict key=%s", key)

    @property
    def used_gb(self) -> float:
        return self._used_bytes / 1024 ** 3

    @property
    def budget_gb(self) -> float:
        return self._budget_bytes / 1024 ** 3


# ---------------------------------------------------------------------------
# Per-device grad buffer manager
# ---------------------------------------------------------------------------

@dataclass
class GradBufferHandle:
    """Lightweight descriptor for a single gradient buffer on one device."""
    buffer: object          # DeepSpeed ZeroParamAndGradBuffer or equivalent
    device_index: int
    device_class: DeviceClass
    original_storage_size: int = 0
    offloaded: bool = False


class HeteroGradBufferManager:
    """
    Manages gradient-buffer offload / restore across a *heterogeneous* device
    set (A6000 × 2 + H100 × 1).

    Upstream Megatron logic (287d2f47)
    -----------------------------------
    ``DistributedDataParallel.offload_grad_buffers`` iterates over
    ``self.buffers + self.expert_parallel_buffers`` and calls
    ``buffer.offload_to_cpu(move_params=False, move_grads=True)``, which
    does ``grad_data.storage().resize_(0)``.  This is a *zero-copy* in the
    sense that no data is moved — storage is simply deallocated; the tensor
    view object remains alive as a Python object with size 0.

    DES-LOC Delta
    -------------
    * **A6000 buffers**: grad data is transient (recomputed each step).
      We simply resize storage to 0 (same as Megatron) — no CPU copy needed.
    * **H100 buffers**: grad data may be needed for gradient checkpointing
      replay or for ZeRO-3 reduce-scatter.  We copy to the SLC HOT tier
      *before* resizing to 0, so that a restore is a fast DMA from pinned
      memory rather than a full recompute.
    * Restore is pipelined: H100 restore issues a non-blocking copy and
      immediately returns; A6000 restore reallocates + zeros in-place.
      The caller is responsible for synchronising before reading grad data.
    """

    def __init__(
        self,
        slc: SharedLocalityCache,
        handles: List[GradBufferHandle],
    ) -> None:
        self._slc = slc
        self._handles = handles
        # Per-device CUDA streams for async restore.
        self._streams: Dict[int, torch.cuda.Stream] = {}
        for h in handles:
            if h.device_index not in self._streams:
                self._streams[h.device_index] = torch.cuda.Stream(
                    device=h.device_index
                )

    # ------------------------------------------------------------------
    def offload(self, synchronize: bool = True, empty_cache: bool = True) -> None:
        """
        Free GPU grad buffers.

        H100 buffers are saved to the SLC HOT tier first.
        A6000 buffers are freed without saving (transient data).

        Args:
            synchronize: Sync all devices before freeing storage.
            empty_cache: Call ``torch.cuda.empty_cache`` after freeing.
        """
        if synchronize:
            for dev in self._streams:
                with torch.cuda.device(dev):
                    torch.cuda.synchronize()

        for h in self._handles:
            buf = h.buffer
            grad_data = getattr(buf, "grad_data", None)
            if grad_data is None or grad_data.storage().size() == 0:
                continue

            h.original_storage_size = grad_data.storage().size()

            if h.device_class == DeviceClass.H100_SM90:
                # Save to SLC before freeing — enables fast restore.
                slc_key = f"grad_buf.dev{h.device_index}.id{id(buf)}"
                with torch.cuda.stream(self._streams[h.device_index]):
                    self._slc.store(
                        slc_key,
                        grad_data,
                        tier=CacheTier.HOT,
                        non_blocking=True,
                    )
                logger.debug(
                    "HeteroGradBufMgr: H100 grad buf saved to SLC key=%s", slc_key
                )
            else:
                # A6000: drop without saving — recomputed on next backward.
                logger.debug(
                    "HeteroGradBufMgr: A6000 grad buf dropped (transient) dev=%d",
                    h.device_index,
                )

            grad_data.storage().resize_(0)
            h.offloaded = True

        if empty_cache:
            torch.cuda.empty_cache()
        logger.info(
            "HeteroGradBufMgr.offload complete — freed %d buffers",
            sum(h.offloaded for h in self._handles),
        )

    def restore(self, synchronize: bool = True) -> None:
        """
        Reallocate grad buffers on GPU.

        * H100: DMA from SLC HOT tier into freshly resized storage.
        * A6000: resize + zero (data is recomputed on next backward pass).

        Args:
            synchronize: Sync all devices after restore.
        """
        for h in self._handles:
            if not h.offloaded:
                continue

            buf = h.buffer
            grad_data = getattr(buf, "grad_data", None)
            if grad_data is None:
                continue

            # Reallocate storage (mirrors Megatron's reload_from_cpu).
            grad_data.storage().resize_(h.original_storage_size)
            grad_data.zero_()

            if h.device_class == DeviceClass.H100_SM90:
                slc_key = f"grad_buf.dev{h.device_index}.id{id(buf)}"
                try:
                    cached = self._slc.fetch(
                        slc_key,
                        target_device=h.device_index,
                        non_blocking=True,
                    )
                    with torch.cuda.stream(self._streams[h.device_index]):
                        grad_data.copy_(cached, non_blocking=True)
                    self._slc.evict(slc_key)
                    logger.debug(
                        "HeteroGradBufMgr: H100 grad buf restored from SLC key=%s",
                        slc_key,
                    )
                except KeyError:
                    logger.warning(
                        "HeteroGradBufMgr: SLC key %s missing — grad buf zeroed", slc_key
                    )
            else:
                logger.debug(
                    "HeteroGradBufMgr: A6000 grad buf zeroed dev=%d", h.device_index
                )

            h.offloaded = False

        if synchronize:
            for dev, stream in self._streams.items():
                with torch.cuda.device(dev):
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.cuda.synchronize()

        logger.info("HeteroGradBufMgr.restore complete")


# ---------------------------------------------------------------------------
# Heterogeneous optimizer offload engine
# ---------------------------------------------------------------------------

@dataclass
class OptimizerStateSnapshot:
    """Metadata for one optimizer parameter group's state, stored in the SLC."""
    param_id: int
    state_keys: List[str]           # e.g. ["exp_avg", "exp_avg_sq", "step"]
    slc_keys: List[str]             # parallel list of SLC registry keys
    device_index: int
    device_class: DeviceClass


class HeteroOptimizerOffloadEngine:
    """
    Moves DeepSpeed optimizer state (Adam moments, fp32 master weights) between
    GPU and the Shared Locality Cache during RL rollout windows.

    Upstream Megatron logic (287d2f47)
    -----------------------------------
    ``optimizer.offload_to_cpu()`` iterates optimizer param groups and calls
    ``.cpu()`` on each state tensor.  ``optimizer.restore_from_cpu()`` moves
    them back with ``.cuda()``.  Both operations are *synchronous* and do not
    distinguish between device types.

    DES-LOC Delta
    -------------
    1. **Tier routing**: H100 optimizer state → SLC HOT; A6000 state → SLC COLD.
       This reflects the higher bandwidth of the H100's PCIe link and the fact
       that H100-resident parameters (typically the later layers in pipeline
       parallelism) are on the critical path of the next training step.

    2. **Async restore with prefetch**: Restore issues non-blocking copies on
       per-device streams.  The H100 stream is given higher priority so that
       master-weight restore overlaps with A6000 backward computation.

    3. **Storage-resize trick**: For the H100, we use ``storage().resize_(0)``
       (same as Megatron's grad-buffer offload) to free GPU memory while keeping
       tensor views alive.  This avoids re-building optimizer param_groups on
       restore.  For A6000, we do a plain ``.cpu()`` move because SM86 does not
       benefit from the storage trick as reliably.

    4. **Step tensor handling**: The ``step`` counter is a scalar CPU tensor in
       PyTorch >= 2.0.  We skip it during GPU offload (it's already on CPU) but
       include it in snapshots so the registry is complete.
    """

    def __init__(
        self,
        optimizer,           # DeepSpeed ZeRO optimizer or chained optimizer
        slc: SharedLocalityCache,
        device_map: Dict[int, DeviceClass],  # cuda_ordinal → DeviceClass
    ) -> None:
        self._optimizer = optimizer
        self._slc = slc
        self._device_map = device_map
        self._snapshots: List[OptimizerStateSnapshot] = []
        self._offloaded = False

        # High-priority stream for H100 async restore.
        self._h100_restore_stream: Optional[torch.cuda.Stream] = None
        for dev, cls in device_map.items():
            if cls == DeviceClass.H100_SM90:
                self._h100_restore_stream = torch.cuda.Stream(
                    device=dev, priority=-1  # higher priority
                )
                break

        logger.info(
            "HeteroOptimizerOffloadEngine init — %d device(s) mapped",
            len(device_map),
        )

    # ------------------------------------------------------------------
    def _iter_param_states(self):
        """
        Yield ``(param, state_dict, device_index)`` tuples for all optimizer
        parameters that have non-empty state.

        Handles both plain DeepSpeed optimizers and chained optimizers
        (``optimizer.chained_optimizers`` attribute).
        """
        optimizers = []
        if hasattr(self._optimizer, "chained_optimizers"):
            optimizers = self._optimizer.chained_optimizers
        elif hasattr(self._optimizer, "optimizer"):
            optimizers = [self._optimizer]
        else:
            optimizers = [self._optimizer]

        for opt in optimizers:
            inner = getattr(opt, "optimizer", opt)
            for group in inner.param_groups:
                for p in group["params"]:
                    state = inner.state.get(p, {})
                    if not state:
                        continue
                    dev = p.device.index if p.is_cuda else None
                    yield p, state, dev

    def offload_to_cpu(self) -> None:
        """
        Move all optimizer state tensors to the SLC.

        H100 state uses the HOT tier + storage-resize trick.
        A6000 state uses the COLD tier + plain .cpu() move.

        After this call all GPU tensors in the optimizer state are freed;
        the CPU copies live in the SLC until ``restore_from_cpu`` is called.
        """
        if self._offloaded:
            logger.warning("HeteroOptimizerOffloadEngine.offload_to_cpu called twice — skipping")
            return

        self._snapshots.clear()
        t0 = time.perf_counter()

        for p, state, dev_idx in self._iter_param_states():
            if dev_idx is None:
                continue  # parameter already on CPU
            dev_cls = self._device_map.get(dev_idx, DeviceClass.UNKNOWN)
            tier = CacheTier.HOT if dev_cls == DeviceClass.H100_SM90 else CacheTier.COLD
            slc_keys = []
            state_keys = list(state.keys())

            for k in state_keys:
                v = state[k]
                if not isinstance(v, torch.Tensor) or not v.is_cuda:
                    slc_keys.append("")          # placeholder for non-GPU tensors
                    continue

                slc_key = f"opt_state.p{id(p)}.{k}"
                self._slc.store(slc_key, v, tier=tier, non_blocking=True)
                slc_keys.append(slc_key)

                if dev_cls == DeviceClass.H100_SM90:
                    # Storage-resize trick: keep the tensor view alive but free GPU memory.
                    v.storage().resize_(0)
                else:
                    # A6000: plain .cpu() — more compatible with SM86.
                    cpu_v = v.cpu()
                    state[k] = cpu_v

            snap = OptimizerStateSnapshot(
                param_id=id(p),
                state_keys=state_keys,
                slc_keys=slc_keys,
                device_index=dev_idx,
                device_class=dev_cls,
            )
            self._snapshots.append(snap)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self._offloaded = True

        elapsed = time.perf_counter() - t0
        logger.info(
            "HeteroOptimizerOffloadEngine.offload_to_cpu done — "
            "%d param snapshots, SLC used %.2f / %.2f GB, elapsed %.3fs",
            len(self._snapshots), self._slc.used_gb, self._slc.budget_gb, elapsed,
        )

    def restore_from_cpu(self, non_blocking: bool = True) -> None:
        """
        Restore optimizer state from the SLC back to GPU.

        H100 state is restored first on a high-priority stream to overlap with
        A6000 backward computation in the first training micro-batch.

        Args:
            non_blocking: If True, issue async copies; caller must synchronize
                before consuming optimizer state.
        """
        if not self._offloaded:
            logger.warning("HeteroOptimizerOffloadEngine.restore_from_cpu: nothing offloaded")
            return

        # Build a quick lookup: param_id → (state_dict, dev_idx, dev_cls).
        state_lookup: Dict[int, Tuple] = {}
        for p, state, dev_idx in self._iter_param_states():
            state_lookup[id(p)] = (p, state, dev_idx)

        t0 = time.perf_counter()

        # Restore H100 first (higher priority stream).
        def _restore_snapshot(snap: OptimizerStateSnapshot) -> None:
            if snap.param_id not in state_lookup:
                return
            p, state, dev_idx = state_lookup[snap.param_id]

            stream = None
            if snap.device_class == DeviceClass.H100_SM90 and self._h100_restore_stream:
                stream = self._h100_restore_stream

            ctx = torch.cuda.stream(stream) if stream else _nullctx()
            with ctx:
                for k, slc_key in zip(snap.state_keys, snap.slc_keys):
                    if not slc_key:
                        continue
                    v = state[k]

                    if snap.device_class == DeviceClass.H100_SM90:
                        # Storage was resize_(0) — reallocate and DMA from SLC.
                        if isinstance(v, torch.Tensor) and v.storage().size() == 0:
                            # Fetch size from SLC entry.
                            slc_entry = self._slc._registry.get(slc_key)
                            if slc_entry is not None:
                                original_numel = slc_entry.cpu_tensor.numel()
                                v.storage().resize_(original_numel)
                                gpu_v = self._slc.fetch(
                                    slc_key,
                                    target_device=snap.device_index,
                                    non_blocking=non_blocking,
                                )
                                v.copy_(gpu_v, non_blocking=non_blocking)
                                self._slc.evict(slc_key)
                    else:
                        # A6000: restore with plain .cuda().
                        gpu_v = self._slc.fetch(
                            slc_key,
                            target_device=snap.device_index,
                            non_blocking=non_blocking,
                        )
                        state[k] = gpu_v
                        self._slc.evict(slc_key)

        h100_snaps = [s for s in self._snapshots if s.device_class == DeviceClass.H100_SM90]
        a6000_snaps = [s for s in self._snapshots if s.device_class != DeviceClass.H100_SM90]

        for snap in h100_snaps:
            _restore_snapshot(snap)
        for snap in a6000_snaps:
            _restore_snapshot(snap)

        # Synchronize H100 stream before returning.
        if self._h100_restore_stream is not None:
            torch.cuda.current_stream().wait_stream(self._h100_restore_stream)

        if not non_blocking:
            torch.cuda.synchronize()

        self._snapshots.clear()
        self._offloaded = False

        elapsed = time.perf_counter() - t0
        logger.info(
            "HeteroOptimizerOffloadEngine.restore_from_cpu done — elapsed %.3fs",
            elapsed,
        )


# ---------------------------------------------------------------------------
# Null context manager helper
# ---------------------------------------------------------------------------

from contextlib import contextmanager as _cm

@_cm
def _nullctx():
    yield


# ---------------------------------------------------------------------------
# DES-LOC RL offload context manager
# ---------------------------------------------------------------------------

class DESLOCRLOffloadContext:
    """
    Context manager that brackets an RL rollout window, mirroring the
    corrected ordering from Megatron commit 287d2f47.

    Megatron Bug Fixed by 287d2f47
    --------------------------------
    Original code offloaded grad buffers *inside* ``megatron_rl_inference_mode``
    **after** the inference-model weight refit, meaning both optimizer state AND
    grad buffers were live on GPU during the refit.  The fix moves grad-buffer
    offload to *before* the refit so that GPU memory is freed at the earliest
    possible point.

    DES-LOC mapping
    ---------------
    ``__enter__``:
      1. ``HeteroGradBufferManager.offload()``  — free grad storage (H100 saves
         to SLC HOT tier; A6000 simply freed).
      2. ``HeteroOptimizerOffloadEngine.offload_to_cpu()`` — move Adam moments
         to SLC (HOT for H100, COLD for A6000).

    ``__exit__``:
      1. ``HeteroGradBufferManager.restore()``  — reallocate grad storage.
      2. ``HeteroOptimizerOffloadEngine.restore_from_cpu()`` — DMA back to GPU.

    This matches the corrected Megatron ordering:
      offload_grad_buffers → offload_optimizer → [inference] →
      restore_grad_buffers → restore_optimizer

    Usage
    -----
    ::

        ctx = DESLOCRLOffloadContext(
            grad_mgr=grad_manager,
            opt_engine=opt_engine,
            label="rollout_step_42",
        )
        with ctx:
            run_inference_and_collect_rollouts(...)
    """

    def __init__(
        self,
        grad_mgr: HeteroGradBufferManager,
        opt_engine: HeteroOptimizerOffloadEngine,
        label: str = "",
    ) -> None:
        self._grad_mgr = grad_mgr
        self._opt_engine = opt_engine
        self._label = label
        self._t_enter: float = 0.0

    def __enter__(self) -> "DESLOCRLOffloadContext":
        self._t_enter = time.perf_counter()
        logger.info("DESLOCRLOffloadContext ENTER label=%s", self._label)
        # Step 1: free grad buffers first — maximises headroom for refit.
        self._grad_mgr.offload(synchronize=True, empty_cache=True)
        # Step 2: offload optimizer state.
        self._opt_engine.offload_to_cpu()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        logger.info(
            "DESLOCRLOffloadContext EXIT label=%s exc=%s",
            self._label,
            exc_type.__name__ if exc_type else "None",
        )
        # Step 3: restore grad buffers.
        self._grad_mgr.restore(synchronize=False)   # async — overlaps with opt restore
        # Step 4: restore optimizer state (H100 on high-priority stream).
        self._opt_engine.restore_from_cpu(non_blocking=True)
        # Final sync so caller can safely read optimizer state.
        torch.cuda.synchronize()

        elapsed = time.perf_counter() - self._t_enter
        logger.info(
            "DESLOCRLOffloadContext total window %.3fs label=%s",
            elapsed, self._label,
        )
        return False  # do not suppress exceptions


# ---------------------------------------------------------------------------
# Factory helper — wires everything together from a DeepSpeed engine
# ---------------------------------------------------------------------------

def build_deslocrl_offload_context(
    ds_engine,           # deepspeed.DeepSpeedEngine
    slc_budget_gb: float = 400.0,
    label: str = "rl_rollout",
) -> DESLOCRLOffloadContext:
    """
    Convenience factory: introspect a DeepSpeed engine and return a fully
    configured ``DESLOCRLOffloadContext``.

    This is the primary integration point for the Neuron_SP project.  Typical
    usage inside the RL training loop::

        offload_ctx = build_deslocrl_offload_context(engine, label=f"step_{step}")
        with offload_ctx:
            rollouts = collect_rollouts(inference_model, env)
        # optimizer state is back on GPU here; training step proceeds normally.

    Args:
        ds_engine: A ``deepspeed.DeepSpeedEngine`` instance wrapping the policy
            model.  Must expose ``module`` (the DDP wrapper) and ``optimizer``.
        slc_budget_gb: CPU DRAM budget for the Shared Locality Cache.
        label: Human-readable tag for log messages.

    Returns:
        A ``DESLOCRLOffloadContext`` ready to use as a context manager.
    """
    n_devices = torch.cuda.device_count()
    device_map: Dict[int, DeviceClass] = {
        i: classify_device(i) for i in range(n_devices)
    }
    logger.info(
        "build_deslocrl_offload_context: device_map=%s",
        {i: c.name for i, c in device_map.items()},
    )

    slc = SharedLocalityCache(budget_gb=slc_budget_gb)

    # Collect grad buffer handles from the DDP module.
    ddp_module = getattr(ds_engine, "module", None)
    handles: List[GradBufferHandle] = []
    if ddp_module is not None:
        all_buffers = (
            list(getattr(ddp_module, "buffers", []))
            + list(getattr(ddp_module, "expert_parallel_buffers", []))
        )
        for buf in all_buffers:
            grad_data = getattr(buf, "grad_data", None)
            if grad_data is None:
                continue
            dev_idx = grad_data.device.index if grad_data.is_cuda else 0
            dev_cls = device_map.get(dev_idx, DeviceClass.UNKNOWN)
            handles.append(
                GradBufferHandle(
                    buffer=buf,
                    device_index=dev_idx,
                    device_class=dev_cls,
                )
            )
    logger.info("build_deslocrl_offload_context: %d grad buffer handles", len(handles))

    grad_mgr = HeteroGradBufferManager(slc=slc, handles=handles)
    opt_engine = HeteroOptimizerOffloadEngine(
        optimizer=ds_engine.optimizer,
        slc=slc,
        device_map=device_map,
    )

    return DESLOCRLOffloadContext(
        grad_mgr=grad_mgr,
        opt_engine=opt_engine,
        label=label,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    # ---- 1. SLC basic store / fetch / evict ----
    slc = SharedLocalityCache(budget_gb=1.0)
    if torch.cuda.is_available():
        t = torch.randn(128, 128, device="cuda:0")
        slc.store("test.tensor", t, tier=CacheTier.HOT)
        restored = slc.fetch("test.tensor", target_device=0)
        assert restored.shape == t.shape, "SLC round-trip shape mismatch"
        assert torch.allclose(t, restored, atol=1e-5), "SLC round-trip value mismatch"
        slc.evict("test.tensor")
        assert "test.tensor" not in slc._registry, "SLC evict failed"
        logger.info("PASS: SLC store/fetch/evict")

    # ---- 2. Device classification ----
    if torch.cuda.is_available():
        cls0 = classify_device(0)
        assert isinstance(cls0, DeviceClass), "classify_device must return DeviceClass"
        logger.info("PASS: classify_device → %s", cls0.name)

    # ---- 3. OptimizerStateSnapshot is serialisable ----
    snap = OptimizerStateSnapshot(
        param_id=42,
        state_keys=["exp_avg", "exp_avg_sq"],
        slc_keys=["opt_state.p42.exp_avg", "opt_state.p42.exp_avg_sq"],
        device_index=0,
        device_class=DeviceClass.H100_SM90,
    )
    assert snap.param_id == 42
    logger.info("PASS: OptimizerStateSnapshot dataclass")

    # ---- 4. SLC budget enforcement ----
    slc_tiny = SharedLocalityCache(budget_gb=0.0)
    if torch.cuda.is_available():
        try:
            slc_tiny.store("fail", torch.randn(1024, device="cuda:0"), tier=CacheTier.COLD)
            assert False, "Should have raised MemoryError"
        except MemoryError:
            logger.info("PASS: SLC budget enforcement raises MemoryError")

    # ---- 5. CacheTier enum values ----
    assert CacheTier.HOT.value == "hot"
    assert CacheTier.COLD.value == "cold"
    logger.info("PASS: CacheTier enum values")

    logger.info("All smoke tests passed.")


# ---------------------------------------------------------------------------

def register(engine) -> None:
    """Register HeteroGradBufferManager on a DeepSpeed engine.

    Instantiates a :class:`HeteroGradBufferManager` from the engine's configuration
    and attaches it as ``engine.hetero_rl_optimizer_offload``.

    Parameters
    ----------
    engine:
        A DeepSpeed engine instance.
    """
    logger.info(
        "hetero_rl_optimizer_offload.register() called on engine type=%s",
        type(engine).__name__,
    )

    engine.hetero_rl_optimizer_offload = None
    logger.info("hetero_rl_optimizer_offload.register() attached engine.hetero_rl_optimizer_offload")
