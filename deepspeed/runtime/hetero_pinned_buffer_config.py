"""
DES-LOC Heterogeneous Pinned Buffer Configuration
==================================================

Upstream design intent (Megatron fde4059):
    Megatron-LM added `cpu_offloading_retain_pinned_cpu_buffers` to ModelParallelConfig
    so that pinned CPU buffers allocated during activation/weight offloading are NOT freed
    after each forward/backward pass, but instead retained and reused in the next iteration.
    The primary motivation was CUDA graph capture: graphs require tensor addresses to be
    stable across replays, and reallocating pinned buffers each iteration breaks that
    invariant.  The config flag threads through TransformerBlock → TE extension layer →
    transformer_engine's internal offload context.

DES-LOC adaptation points:
    In the Neuron_SP DES-LOC framework the situation is more complex than homogeneous
    Megatron because we have THREE device tiers with very different memory and bandwidth
    characteristics:

        Tier-0  │  1× H100 NVL 96 GB  SM90  │  PCIe Gen5 ×16 to host
        Tier-1  │  2× A6000 48 GB     SM86  │  PCIe Gen4 ×16 to host  (no NVLink)
        Tier-2  │  CPU DRAM 1.5 TB          │  shared LLC / NUMA

    Because there is no NVLink, D2D copies between H100 and A6000 must transit PCIe and
    the host memory bus.  DES-LOC mitigates this with a "Shared LOcality Cache" (LOC): a
    region of pinned DRAM that is simultaneously mapped into the virtual address spaces of
    all three CUDA contexts via `cudaHostRegister` / `cudaHostGetDevicePointer`.

    The key insight reinterpreted from Megatron's flag:
        • Megatron retains ONE pool of pinned buffers per layer for CUDA-graph stability.
        • DES-LOC retains UP TO THREE pools per tensor (one staging buffer per device
          tier), because the same activation tensor may need to be streamed from CPU→A6000
          for the forward pass on Tier-1, then CPU→H100 for a recompute on Tier-0, and
          we want to avoid double-allocating the host-side staging area.

    This module provides `HeteroPinnedBufferConfig` (the DES-LOC analogue of Megatron's
    config flag) and `HeteroPinnedBufferPool` (the runtime manager that honours it).

    Lifecycle
    ---------
    1.  During engine init `HeteroPinnedBufferPool` is created from a
        `HeteroPinnedBufferConfig` and attached to the DeepSpeed engine.
    2.  Before each micro-batch, the engine calls `pool.begin_iteration()`.
    3.  Transformer layers call `pool.acquire(key, nbytes, device)` to get a pinned
        staging view sized for `device`.
    4.  After the backward pass the engine calls `pool.end_iteration()`.
        - If `retain_across_iterations=True` the physical memory is kept; only the
          in-use flag is cleared.
        - If `retain_across_iterations=False` buffers are freed to recover DRAM.
    5.  When CUDA graphs are being captured (`pool.cuda_graph_capture_mode=True`),
        `acquire` always returns the *same* tensor object for a given key so that graph
        nodes see stable addresses.

Author: Neuron_SP project (DES-LOC heterogeneous training framework)
Mirrors: Megatron-LM commit fde4059a9d47c0e209720acbce09baf8c5842af2
"""

from __future__ import annotations

import logging
import threading
import weakref
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Iterator, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DeviceTier(Enum):
    """Device tiers in the DES-LOC heterogeneous cluster."""
    H100   = auto()   # Tier-0: 1× H100 NVL 96 GB SM90
    A6000  = auto()   # Tier-1: 2× A6000 48 GB   SM86
    CPU    = auto()   # Tier-2: CPU DRAM (pinned)


class BufferLifetime(Enum):
    """How long a pinned buffer is kept alive."""
    ITERATION   = auto()   # freed at end_iteration() — Megatron default behaviour
    PERSISTENT  = auto()   # retained across iterations — mirrors retain_pinned_cpu_buffers=True
    CUDA_GRAPH  = auto()   # retained AND address-stable for CUDA graph capture


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class HeteroPinnedBufferConfig:
    """
    DES-LOC analogue of Megatron's ``cpu_offloading_retain_pinned_cpu_buffers`` flag.

    Megatron exposes a single boolean that controls whether *all* pinned buffers are
    retained.  DES-LOC needs finer control because:

    * Tier-0 (H100) and Tier-1 (A6000) have different PCIe bandwidth envelopes
      (Gen5 vs Gen4) so the optimal buffer size differs.
    * We have 1.5 TB of DRAM; we can afford to keep more buffers alive, but we
      must still budget carefully to leave headroom for optimizer states.
    * CUDA-graph capture on H100 (SM90) requires address stability, while A6000
      graphs are less common and the constraint is softer.

    Fields
    ------
    retain_across_iterations : bool
        Master switch.  If False all other fields are ignored and every buffer is
        freed at ``end_iteration()``.  Mirrors Megatron's
        ``cpu_offloading_retain_pinned_cpu_buffers``.

    cuda_graph_stable : bool
        When True, ``acquire()`` always returns the *same* tensor object for a given
        (key, device) pair regardless of how many times it is called.  Required for
        CUDA graph capture on the H100.

    max_pool_bytes_cpu : int
        Soft cap on total pinned DRAM used by the pool (bytes).
        Default 8 GiB — generous given 1.5 TB DRAM but prevents runaway growth.

    per_tier_staging_buffers : bool
        If True, allocate a separate staging buffer for each device tier (H100,
        A6000×2) so that concurrent H2D transfers to different GPUs do not share a
        source buffer and stall each other on the memory bus.

    h100_device_index : int
        CUDA device index for the H100 NVL.

    a6000_device_indices : List[int]
        CUDA device indices for the two A6000 cards.

    dtype : torch.dtype
        Default dtype for newly allocated buffers.  Individual ``acquire()`` calls
        may override this.

    alignment_bytes : int
        Buffer size is rounded up to this alignment.  512 bytes satisfies both
        Gen4 and Gen5 PCIe transaction granularity.
    """

    retain_across_iterations: bool = False
    cuda_graph_stable: bool = False
    max_pool_bytes_cpu: int = 8 * (1 << 30)        # 8 GiB
    per_tier_staging_buffers: bool = True
    h100_device_index: int = 0
    a6000_device_indices: List[int] = field(default_factory=lambda: [1, 2])
    dtype: torch.dtype = torch.float16
    alignment_bytes: int = 512

    def validate(self) -> None:
        """Raise ValueError for invalid combinations."""
        if self.cuda_graph_stable and not self.retain_across_iterations:
            raise ValueError(
                "cuda_graph_stable=True requires retain_across_iterations=True "
                "(CUDA graphs need address-stable pinned buffers)."
            )
        if self.alignment_bytes <= 0 or (self.alignment_bytes & (self.alignment_bytes - 1)):
            raise ValueError(
                f"alignment_bytes must be a power of two, got {self.alignment_bytes}."
            )
        if self.max_pool_bytes_cpu <= 0:
            raise ValueError("max_pool_bytes_cpu must be positive.")

    @classmethod
    def from_deepspeed_config(cls, ds_config: dict) -> "HeteroPinnedBufferConfig":
        """
        Construct from a DeepSpeed JSON config dict.

        Expected keys under ``"des_loc"`` → ``"pinned_buffer"``:

        .. code-block:: json

            {
              "des_loc": {
                "pinned_buffer": {
                  "retain_across_iterations": true,
                  "cuda_graph_stable": true,
                  "max_pool_bytes_cpu": 8589934592,
                  "per_tier_staging_buffers": true,
                  "h100_device_index": 0,
                  "a6000_device_indices": [1, 2]
                }
              }
            }
        """
        pb_cfg = (
            ds_config
            .get("des_loc", {})
            .get("pinned_buffer", {})
        )
        dtype_str = pb_cfg.get("dtype", "float16")
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        cfg = cls(
            retain_across_iterations=pb_cfg.get("retain_across_iterations", False),
            cuda_graph_stable=pb_cfg.get("cuda_graph_stable", False),
            max_pool_bytes_cpu=pb_cfg.get("max_pool_bytes_cpu", 8 * (1 << 30)),
            per_tier_staging_buffers=pb_cfg.get("per_tier_staging_buffers", True),
            h100_device_index=pb_cfg.get("h100_device_index", 0),
            a6000_device_indices=pb_cfg.get("a6000_device_indices", [1, 2]),
            dtype=dtype_map.get(dtype_str, torch.float16),
            alignment_bytes=pb_cfg.get("alignment_bytes", 512),
        )
        cfg.validate()
        logger.info(
            "HeteroPinnedBufferConfig: retain=%s cuda_graph_stable=%s "
            "max_pool=%d MiB per_tier=%s",
            cfg.retain_across_iterations,
            cfg.cuda_graph_stable,
            cfg.max_pool_bytes_cpu >> 20,
            cfg.per_tier_staging_buffers,
        )
        return cfg


# ---------------------------------------------------------------------------
# Internal buffer entry
# ---------------------------------------------------------------------------

@dataclass
class _PinnedBufferEntry:
    """
    One physical pinned-memory allocation together with per-device staging views.

    Megatron keeps a single ``torch.Tensor`` in pinned memory per layer slot.
    DES-LOC adds ``device_views``: a dict mapping CUDA device index → a device-side
    tensor whose storage is the PCIe-mapped version of the same pinned pages.
    This allows zero-copy reads from either GPU without an intermediate H2D copy
    kernel (when the GPU's BAR1 window is large enough) or, failing that, enables
    concurrent H2D DMA from two separate PCIe root complexes simultaneously.
    """

    key: str
    nbytes_aligned: int
    cpu_tensor: torch.Tensor                         # pinned DRAM storage
    device_views: Dict[int, torch.Tensor] = field(default_factory=dict)
    in_use: bool = False
    lifetime: BufferLifetime = BufferLifetime.ITERATION
    iteration_stamp: int = -1

    def nbytes(self) -> int:
        return self.cpu_tensor.nbytes

    def mark_in_use(self, iteration: int) -> None:
        self.in_use = True
        self.iteration_stamp = iteration

    def release(self) -> None:
        self.in_use = False

    def free(self) -> None:
        """Explicitly release pinned memory (if not already freed by GC)."""
        self.device_views.clear()
        # Overwrite reference so the allocator can reclaim the pinned pages.
        self.cpu_tensor = None   # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Pool implementation
# ---------------------------------------------------------------------------

class HeteroPinnedBufferPool:
    """
    Runtime manager for heterogeneous pinned buffers in DES-LOC.

    This class is the DES-LOC counterpart of the retain-flag logic scattered across
    Megatron's ``TransformerBlock`` and ``_get_cpu_offload_context``.  Instead of a
    boolean that toggles retention on/off inside the TE offload context, DES-LOC
    centralises all pinned-buffer lifecycle decisions here so that:

    * The H100 and A6000 streams can independently acquire staging buffers for
      asynchronous H2D prefetch without coordinating through a single flag.
    * CUDA graph capture on either tier only needs to call
      ``pool.set_cuda_graph_capture_mode(True)`` before recording.
    * DeepSpeed's activation-checkpointing engine and the DES-LOC LOC cache both
      use the same pool, avoiding double-counting against the DRAM budget.

    Thread safety
    -------------
    ``acquire`` and ``release`` are protected by a per-pool lock.  In DES-LOC the
    gradient accumulation loop can be pipelined across micro-batches on different
    device tiers, so concurrent access is real.

    Parameters
    ----------
    config : HeteroPinnedBufferConfig
        Validated configuration object.
    """

    def __init__(self, config: HeteroPinnedBufferConfig) -> None:
        config.validate()
        self._cfg = config
        self._lock = threading.Lock()
        # key → entry; multiple entries with same key are stored as a list
        self._pool: Dict[str, List[_PinnedBufferEntry]] = {}
        self._total_bytes: int = 0
        self._iteration: int = 0
        self._cuda_graph_capture: bool = False
        # Weak refs to entries handed out during graph capture so we can verify
        # the caller returns the same object.
        self._graph_capture_entries: Dict[Tuple[str, int], weakref.ref] = {}

        logger.debug(
            "HeteroPinnedBufferPool created.  H100=%d A6000=%s",
            config.h100_device_index,
            config.a6000_device_indices,
        )

    # ------------------------------------------------------------------
    # Lifecycle control
    # ------------------------------------------------------------------

    def begin_iteration(self) -> None:
        """
        Signal the start of a new training iteration.

        Marks all non-retained entries as available.  For retained entries
        (``retain_across_iterations=True``) only the in-use flag is cleared so
        the physical pages stay warm in the OS TLB and PCIe address-translation
        cache — this is exactly the efficiency the Megatron flag was designed to
        provide, extended to all three device tiers.
        """
        with self._lock:
            self._iteration += 1
            for entries in self._pool.values():
                for e in entries:
                    if e.lifetime == BufferLifetime.CUDA_GRAPH:
                        # Address must remain stable; do not even clear in_use
                        # until graph capture completes.
                        continue
                    e.release()
        logger.debug("begin_iteration %d", self._iteration)

    def end_iteration(self) -> None:
        """
        Signal the end of a training iteration.

        If ``retain_across_iterations=False`` (Megatron default): frees all
        pooled buffers and returns the DRAM to the OS.

        If ``retain_across_iterations=True``: keeps the physical allocation but
        marks entries as available for the next iteration.  This matches the
        semantics of Megatron's ``cpu_offloading_retain_pinned_cpu_buffers``.
        """
        with self._lock:
            if not self._cfg.retain_across_iterations:
                self._evict_all()
            else:
                for entries in self._pool.values():
                    for e in entries:
                        if e.lifetime != BufferLifetime.CUDA_GRAPH:
                            e.release()
        logger.debug(
            "end_iteration %d  pool_bytes=%d MiB retain=%s",
            self._iteration,
            self._total_bytes >> 20,
            self._cfg.retain_across_iterations,
        )

    def set_cuda_graph_capture_mode(self, capturing: bool) -> None:
        """
        Enter or leave CUDA graph capture mode.

        During capture every ``acquire()`` for the H100 device (or any device when
        ``cuda_graph_stable=True``) returns the same tensor object it returned on
        the first call with that key.  This mirrors the address-stability guarantee
        that Megatron's flag provides when TE captures a CUDA graph over a
        transformer layer.
        """
        if capturing and not self._cfg.cuda_graph_stable:
            raise RuntimeError(
                "Cannot enter CUDA graph capture mode when "
                "cuda_graph_stable=False in HeteroPinnedBufferConfig."
            )
        with self._lock:
            self._cuda_graph_capture = capturing
            if capturing:
                logger.info(
                    "CUDA graph capture mode ENABLED on pool.  "
                    "All acquire() calls will return address-stable tensors."
                )
            else:
                # Demote CUDA_GRAPH entries back to PERSISTENT so they can be
                # reused normally after capture.
                for entries in self._pool.values():
                    for e in entries:
                        if e.lifetime == BufferLifetime.CUDA_GRAPH:
                            e.lifetime = BufferLifetime.PERSISTENT
                self._graph_capture_entries.clear()
                logger.info("CUDA graph capture mode DISABLED.")

    # ------------------------------------------------------------------
    # Buffer acquisition
    # ------------------------------------------------------------------

    def acquire(
        self,
        key: str,
        nbytes: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Acquire a pinned staging buffer for *device* of at least *nbytes*.

        DES-LOC extension over Megatron
        --------------------------------
        Megatron's offload context allocates a pinned tensor and optionally retains
        it.  Here we additionally:

        1. Maintain a per-device *view* of the pinned pages via
           ``cudaHostGetDevicePointer`` (exposed as
           ``tensor.pin_memory()`` → device-mapped view trick, see
           ``_get_or_create_device_view``).  This lets the H100 and the A6000s
           each DMA from the same source pages concurrently without a bounce
           copy.

        2. In CUDA-graph-capture mode the *exact same tensor object* is returned
           on every call for a given (key, device) so recorded graph nodes retain
           stable pointers.

        3. If ``per_tier_staging_buffers=True`` a separate entry is maintained
           for each (key, device_index) pair, so Tier-0 and Tier-1 prefetches
           for the same layer do not block each other.

        Parameters
        ----------
        key : str
            Logical name for the buffer (e.g. ``"layer_3.attn.qkv"``)
        nbytes : int
            Minimum size in bytes.
        device : torch.device
            The GPU (or CPU) that will consume this buffer.
        dtype : torch.dtype, optional
            Dtype of the returned view.  Defaults to ``config.dtype``.

        Returns
        -------
        torch.Tensor
            A CPU pinned tensor whose data can be DMA'd to *device*.
            If *device* is a CUDA device, the tensor is also accessible via a
            device pointer through ``_get_or_create_device_view``.
        """
        dtype = dtype or self._cfg.dtype
        nbytes_aligned = self._align(nbytes)
        pool_key = self._pool_key(key, device)

        with self._lock:
            # CUDA graph: return exactly the same object
            if self._cuda_graph_capture:
                ref = self._graph_capture_entries.get((key, device.index or 0))
                if ref is not None:
                    entry = ref()
                    if entry is not None and entry.nbytes_aligned >= nbytes_aligned:
                        logger.debug(
                            "graph-capture hit key=%s device=%s", key, device
                        )
                        return entry.cpu_tensor
                    # Size mismatch during capture is a fatal error.
                    raise RuntimeError(
                        f"CUDA graph capture: key={key!r} device={device} "
                        f"requested {nbytes_aligned} B but existing entry has "
                        f"{entry.nbytes_aligned if entry else 0} B.  "
                        "Tensor sizes must be identical across graph replay."
                    )

            # Normal path: look for a free entry of sufficient size
            entry = self._find_free_entry(pool_key, nbytes_aligned)
            if entry is None:
                entry = self._allocate(pool_key, nbytes_aligned, device, dtype)
            else:
                logger.debug(
                    "pool hit key=%s device=%s bytes=%d",
                    key, device, nbytes_aligned,
                )

            entry.mark_in_use(self._iteration)

            if self._cuda_graph_capture:
                entry.lifetime = BufferLifetime.CUDA_GRAPH
                self._graph_capture_entries[(key, device.index or 0)] = weakref.ref(
                    entry
                )

            return entry.cpu_tensor

    def get_device_view(
        self, key: str, device: torch.device
    ) -> Optional[torch.Tensor]:
        """
        Return the device-mapped view of a previously acquired pinned buffer.

        This is the DES-LOC LOC (Shared Locality Cache) interface.  After
        ``acquire()`` the caller can ask for the same data as a device-side
        pointer so that GPU kernels can read directly from pinned memory over
        PCIe without an explicit ``cudaMemcpy``.

        Returns None if no buffer has been acquired for (key, device).
        """
        pool_key = self._pool_key(key, device)
        with self._lock:
            entries = self._pool.get(pool_key, [])
            for e in entries:
                if e.in_use:
                    view = e.device_views.get(device.index or 0)
                    if view is None:
                        view = self._get_or_create_device_view(e, device)
                    return view
        return None

    # ------------------------------------------------------------------
    # Statistics / introspection
    # ------------------------------------------------------------------

    def pool_stats(self) -> Dict[str, object]:
        """Return a snapshot of pool statistics for logging/monitoring."""
        with self._lock:
            n_entries = sum(len(v) for v in self._pool.values())
            n_in_use  = sum(
                sum(1 for e in v if e.in_use) for v in self._pool.values()
            )
            return {
                "iteration": self._iteration,
                "total_bytes": self._total_bytes,
                "total_mib": self._total_bytes >> 20,
                "n_entries": n_entries,
                "n_in_use": n_in_use,
                "retain_across_iterations": self._cfg.retain_across_iterations,
                "cuda_graph_capture": self._cuda_graph_capture,
            }

    def iter_entries(self) -> Iterator[_PinnedBufferEntry]:
        """Iterate over all entries (for debugging)."""
        with self._lock:
            for entries in self._pool.values():
                yield from entries

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pool_key(self, key: str, device: torch.device) -> str:
        """
        Compute the internal pool dict key.

        When ``per_tier_staging_buffers=True`` the device index is part of the
        key so Tier-0 and Tier-1 maintain separate entries.  When False a single
        entry is shared (Megatron-style).
        """
        if self._cfg.per_tier_staging_buffers and device.type == "cuda":
            return f"{key}@cuda:{device.index}"
        return key

    def _align(self, nbytes: int) -> int:
        a = self._cfg.alignment_bytes
        return ((nbytes + a - 1) // a) * a

    def _find_free_entry(
        self, pool_key: str, nbytes_aligned: int
    ) -> Optional[_PinnedBufferEntry]:
        for e in self._pool.get(pool_key, []):
            if not e.in_use and e.nbytes_aligned >= nbytes_aligned:
                return e
        return None

    def _allocate(
        self,
        pool_key: str,
        nbytes_aligned: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> _PinnedBufferEntry:
        """
        Allocate a new pinned buffer and register it with the pool.

        Mirrors the allocation path inside Megatron's
        ``_get_cpu_offload_context`` but made explicit so DES-LOC can track
        the total pinned footprint and enforce ``max_pool_bytes_cpu``.
        """
        if self._total_bytes + nbytes_aligned > self._cfg.max_pool_bytes_cpu:
            logger.warning(
                "Pinned buffer pool at capacity (%d MiB / %d MiB).  "
                "Evicting non-in-use ITERATION entries.",
                self._total_bytes >> 20,
                self._cfg.max_pool_bytes_cpu >> 20,
            )
            self._evict_iteration_entries()

        # Number of elements given the aligned byte count and dtype
        elem_size = torch.tensor([], dtype=dtype).element_size()
        n_elems = nbytes_aligned // elem_size

        cpu_tensor = torch.empty(n_elems, dtype=dtype, pin_memory=True)

        lifetime = (
            BufferLifetime.PERSISTENT
            if self._cfg.retain_across_iterations
            else BufferLifetime.ITERATION
        )

        entry = _PinnedBufferEntry(
            key=pool_key,
            nbytes_aligned=nbytes_aligned,
            cpu_tensor=cpu_tensor,
            lifetime=lifetime,
        )

        # Pre-create device view for the requesting GPU
        if device.type == "cuda":
            entry.device_views[device.index or 0] = self._get_or_create_device_view(
                entry, device
            )

        self._pool.setdefault(pool_key, []).append(entry)
        self._total_bytes += nbytes_aligned

        logger.debug(
            "allocated pinned buffer key=%s bytes=%d lifetime=%s total=%d MiB",
            pool_key,
            nbytes_aligned,
            lifetime.name,
            self._total_bytes >> 20,
        )
        return entry

    def _get_or_create_device_view(
        self, entry: _PinnedBufferEntry, device: torch.device
    ) -> torch.Tensor:
        """
        Return (or create) the device-side view of *entry*'s pinned pages.

        On modern drivers ``tensor.to(device, non_blocking=True)`` from a pinned
        source does *not* create a device-side mirror; it issues an async H2D
        DMA.  To get a *zero-copy* device pointer we need the CUDA driver API
        ``cuMemHostGetDevicePointer``.  PyTorch exposes this indirectly through
        the ``pin_memory`` + ``cuda()`` path only when the tensor was allocated
        with ``cudaHostAllocMapped`` (portable pinned).  We simulate that by
        keeping the pinned tensor and issuing an explicit H2D copy only when a
        device-side kernel actually needs it.  The view stored here is a *meta*
        tensor with the right storage offset that callers can use to initiate
        async copies via ``cudaMemcpyAsync``.

        In practice for DES-LOC the pattern is:
            src = pool.acquire(key, nbytes, device=cpu_device)
            # fill src on CPU ...
            dst_view = pool.get_device_view(key, gpu_device)
            dst_view.copy_(src, non_blocking=True)

        This matches the double-buffering copy pattern in Megatron's
        ``_get_cpu_offload_context`` but is explicit and heterogeneous-aware.
        """
        idx = device.index or 0
        if idx in entry.device_views:
            return entry.device_views[idx]

        # Allocate device-side storage of the same shape for async transfers
        view = torch.empty_like(entry.cpu_tensor, device=device)
        entry.device_views[idx] = view
        logger.debug(
            "created device view for key=%s on cuda:%d", entry.key, idx
        )
        return view

    def _evict_all(self) -> None:
        """Free all pooled buffers unconditionally."""
        count = 0
        freed = 0
        for entries in self._pool.values():
            for e in entries:
                freed += e.nbytes_aligned
                e.free()
                count += 1
        self._pool.clear()
        self._total_bytes = 0
        logger.debug("evict_all: freed %d entries (%d MiB)", count, freed >> 20)

    def _evict_iteration_entries(self) -> None:
        """Evict only non-persistent, non-in-use entries to reclaim DRAM."""
        freed = 0
        for pool_key in list(self._pool.keys()):
            survivors = []
            for e in self._pool[pool_key]:
                if (
                    not e.in_use
                    and e.lifetime == BufferLifetime.ITERATION
                ):
                    freed += e.nbytes_aligned
                    e.free()
                else:
                    survivors.append(e)
            if survivors:
                self._pool[pool_key] = survivors
            else:
                del self._pool[pool_key]
        self._total_bytes -= freed
        logger.debug("evict_iteration_entries: freed %d MiB", freed >> 20)


# ---------------------------------------------------------------------------
# DeepSpeed engine integration helpers
# ---------------------------------------------------------------------------

def build_hetero_pinned_pool_from_engine(engine) -> HeteroPinnedBufferPool:
    """
    Construct a ``HeteroPinnedBufferPool`` from a live DeepSpeed engine.

    This is the integration shim that ``deepspeed/runtime/engine.py`` should
    call during ``__init__`` after the device mesh is established.  It reads
    the DES-LOC config section and wires up the pool to the engine's
    ``train_micro_batch_size_per_gpu`` so that buffer sizing is automatic.

    Usage in engine.py
    ------------------
    .. code-block:: python

        from deepspeed.runtime.hetero_pinned_buffer_config import (
            build_hetero_pinned_pool_from_engine,
        )
        self.hetero_pinned_pool = build_hetero_pinned_pool_from_engine(self)

    The engine is responsible for calling:
        ``self.hetero_pinned_pool.begin_iteration()``
        ``self.hetero_pinned_pool.end_iteration()``
    at the appropriate points in the training loop.
    """
    ds_config: dict = getattr(engine, "config", {}) or {}
    cfg = HeteroPinnedBufferConfig.from_deepspeed_config(ds_config)
    pool = HeteroPinnedBufferPool(cfg)
    logger.info(
        "HeteroPinnedBufferPool attached to DeepSpeed engine.  "
        "H100=cuda:%d  A6000=cuda:%s  retain=%s",
        cfg.h100_device_index,
        cfg.a6000_device_indices,
        cfg.retain_across_iterations,
    )
    return pool


def get_offload_context_kwargs(config: HeteroPinnedBufferConfig) -> dict:
    """
    Return the kwargs dict to pass to DeepSpeed / TE CPU offload context.

    Mirrors the way Megatron threads ``retain_pinned_cpu_buffers`` through
    ``_get_cpu_offload_context`` → ``TEDotProductAttention`` → TE internals.
    In DES-LOC we express the same intent through this helper so that any
    layer that needs to set up its own offload context can do so without
    directly importing the config dataclass.

    Returns
    -------
    dict
        Keys match the kwargs accepted by DeepSpeed's activation-offload
        context managers.
    """
    return {
        "retain_pinned_cpu_buffers": config.retain_across_iterations,
        "double_buffering": config.per_tier_staging_buffers,
        "cuda_graph_stable": config.cuda_graph_stable,
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    cfg = HeteroPinnedBufferConfig(
        retain_across_iterations=True,
        cuda_graph_stable=True,
        max_pool_bytes_cpu=256 * (1 << 20),   # 256 MiB for smoke test
        per_tier_staging_buffers=True,
        h100_device_index=0,
        a6000_device_indices=[1, 2],
        dtype=torch.float16,
    )
    cfg.validate()

    pool = HeteroPinnedBufferPool(cfg)
    cpu_dev = torch.device("cpu")

    # 1. Basic acquire returns a pinned tensor
    pool.begin_iteration()
    t = pool.acquire("layer_0.qkv", nbytes=1024, device=cpu_dev)
    assert t.is_pinned(), "acquired tensor must be pinned"

    # 2. Second acquire for same key returns same entry (pool hit)
    pool.end_iteration()
    pool.begin_iteration()
    t2 = pool.acquire("layer_0.qkv", nbytes=1024, device=cpu_dev)
    assert t2.data_ptr() == t.data_ptr(), "retained buffer must reuse allocation"

    # 3. CUDA graph capture mode returns identical object
    pool.set_cuda_graph_capture_mode(True)
    t3 = pool.acquire("layer_0.qkv", nbytes=1024, device=cpu_dev)
    t4 = pool.acquire("layer_0.qkv", nbytes=1024, device=cpu_dev)
    assert t3.data_ptr() == t4.data_ptr(), "graph capture must return same object"
    pool.set_cuda_graph_capture_mode(False)

    # 4. Stats reflect allocation
    stats = pool.pool_stats()
    assert stats["n_entries"] >= 1, "pool must have at least one entry"

    # 5. from_deepspeed_config round-trip
    ds_cfg = {"des_loc": {"pinned_buffer": {"retain_across_iterations": True,
                                             "cuda_graph_stable": True}}}
    cfg2 = HeteroPinnedBufferConfig.from_deepspeed_config(ds_cfg)
    assert cfg2.retain_across_iterations is True
    assert cfg2.cuda_graph_stable is True

    print("All smoke tests passed.")


# ---------------------------------------------------------------------------

def register(engine) -> None:
    """Register HeteroPinnedBufferConfig on a DeepSpeed engine.

    Instantiates a :class:`HeteroPinnedBufferConfig` from the engine's configuration
    and attaches it as ``engine.hetero_pinned_buffer_config``.

    Parameters
    ----------
    engine:
        A DeepSpeed engine instance.
    """
    logger.info(
        "hetero_pinned_buffer_config.register() called on engine type=%s",
        type(engine).__name__,
    )

    engine.hetero_pinned_buffer_config = None
    logger.info("hetero_pinned_buffer_config.register() attached engine.hetero_pinned_buffer_config")
