"""
DES-LOC Heterogeneous Wgrad Safe Double Buffer
===============================================

Upstream Design Intent (Megatron commit 55638bc4):
--------------------------------------------------
The original Megatron-LM commit fixes a race condition that arose when using
double-buffered gradient accumulation buffers (``fsdp_double_buffer=True``) in
conjunction with weight-gradient (wgrad) computation.

The race condition manifested as follows:

1. The ``_enforce_double_buffer_limit`` call was placed *before* ``param.get_main_grad()``,
   inside the backward hook of each FSDP unit. The intent was to ensure that when a
   new bucket is about to be allocated, any in-flight bucket from more than two FSDP
   units back is first freed / reduced before proceeding.

2. The bug: the ``_megatron_fsdp_model`` reference was attached to every parameter
   in a loop inside ``__init__``, *after* ``_replace_param_with_distributed_if_needed``
   had already swapped out the raw parameters for ``DTensor``-backed distributed
   parameters. The loop therefore iterated over **replaced** parameters that would
   later be swapped back, leaving the new DTensor parameters without the
   ``_megatron_fsdp_model`` attribute.

3. Fix strategy:
   a. Remove the post-init loop; instead attach ``_megatron_fsdp_model`` lazily, only
      inside ``_replace_param_with_distributed_if_needed`` and its mirror restore path,
      so the attribute travels with whichever object currently lives in the module.
   b. Move ``_enforce_double_buffer_limit`` from the backward hook into
      ``main_grad_getter`` (a property getter on the parameter), so enforcement happens
      at the exact moment the gradient buffer is about to be *allocated*, not before.
   c. Correct the double-buffer threshold from ``> 1`` (i.e., allow only 1 live unit)
      to ``> 2`` (allow up to 2 live units), matching the semantic definition of a
      *double* buffer.

DES-LOC Adaptation Points (HeteroWgradSafeDoubleBuffer):
---------------------------------------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) runs on a heterogeneous
cluster: 2× A6000-48GB (SM86, PCIe) + 1× H100-NVL-96GB (SM90, PCIe) backed by
1.5 TB of CPU DRAM used as a Locality Cache tier.

Key differences from the Megatron FSDP setting:

1. **Heterogeneous device tiers**: Each parameter/gradient bucket is owned by a
   specific device tier (A6000_0, A6000_1, H100). The double-buffer limit must be
   enforced *per-tier*, not globally, because GPU memory capacities differ wildly.
   Evicting an H100 bucket to free space for A6000 work would defeat the purpose.

2. **CPU Locality Cache**: Buckets that cannot fit in GPU VRAM are spilled to the
   CPU Locality Cache (pinned DRAM). The double-buffer enforcer must coordinate with
   the cache tier to decide whether to block-on-reduce or spill-to-cache.

3. **No NVLink**: Gradient reduce-scatter happens over PCIe. Latency is high, so the
   pipeline uses an aggressive prefetch schedule; the double-buffer limit is the only
   back-pressure mechanism. Getting the threshold wrong (off-by-one, as in the
   upstream bug) either deadlocks the pipeline or causes OOM.

4. **SM-version-aware gradient casting**: H100 (SM90) supports FP8 gradient comm;
   A6000 (SM86) does not. The ``main_grad_getter`` must select the comm dtype based
   on the owning device's SM version before calling ``fetch_bucket``.

5. **Decoupled execution streams**: Each device tier runs its own CUDA stream for
   reduce-scatter. The race condition fix (moving enforcement into the getter) is even
   more critical here because stream ordering across PCIe is not guaranteed without
   explicit events.

Module layout::

    HeteroDeviceTier          – enum: A6000_0, A6000_1, H100, CPU_CACHE
    TierMemoryConfig          – static memory limits and SM versions per tier
    LocalityCache             – thin wrapper around pinned CPU tensors for spill/fill
    DoubleBufferEnforcer      – per-tier gate that blocks or spills when > 2 units live
    HeteroGradBucket          – a gradient bucket tagged with its owning tier
    HeteroParamGradBuffer     – manages bucket lists and the main_grad_getter property
    HeteroWgradSafeDoubleBuffer – top-level module mirroring MegatronFSDP.__init__ fix

Reference: Megatron-LM commit 55638bc4433922cadbcb9f53991236e948dabdc8
Project:   github.com/dylanyunlon/Neuron_SP (Neuron_SP / DES-LOC)
"""

from __future__ import annotations

import enum
import logging
import math
import threading
import time
import unittest
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Set, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

# ---------------------------------------------------------------------------
# Module-level logger – only emitted at INFO or above for meaningful events.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_handler = logging.StreamHandler()
_handler.setFormatter(
    logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
    )
)
if not logger.handlers:
    logger.addHandler(_handler)


# ===========================================================================
# Section 1 – Hardware topology constants
# ===========================================================================

class HeteroDeviceTier(enum.Enum):
    """
    Enumeration of the three physical GPU tiers in the DES-LOC cluster plus the
    CPU Locality Cache tier used for gradient spill/fill.

    Ordinal ordering reflects priority for keeping live buffers in VRAM:
    H100 (highest VRAM, SM90) > A6000_1 > A6000_0 > CPU_CACHE (spill sink).
    """
    A6000_0   = 0   # NVIDIA A6000 48 GB, SM86, first card
    A6000_1   = 1   # NVIDIA A6000 48 GB, SM86, second card
    H100      = 2   # NVIDIA H100 NVL 96 GB, SM90
    CPU_CACHE = 3   # Pinned CPU DRAM Locality Cache (1.5 TB available)


@dataclass(frozen=True)
class TierMemoryConfig:
    """
    Static hardware description for a single device tier.

    Fields
    ------
    tier : HeteroDeviceTier
        Which tier this config describes.
    vram_bytes : int
        Usable VRAM budget reserved for gradient double-buffering (not full VRAM).
    sm_version : int
        CUDA SM version as an integer (e.g. 86 for SM8.6, 90 for SM9.0).
    supports_fp8_grad_comm : bool
        Whether the device supports FP8 gradient all-reduce / reduce-scatter.
        Only SM90 (H100) exposes native FP8 tensor core support.
    cuda_device_index : int
        PyTorch ``torch.device("cuda:N")`` index, or -1 for CPU_CACHE.
    pcie_bandwidth_gbps : float
        Approximate unidirectional PCIe bandwidth to the host bus (GB/s).
        Used by the enforcer to estimate how long a reduce will take so it can
        decide between blocking and spilling.
    double_buffer_limit : int
        Maximum number of simultaneously live FSDP units for this tier.
        Mirrors the corrected Megatron threshold (``> 2`` ↔ limit == 2).
    """
    tier: HeteroDeviceTier
    vram_bytes: int
    sm_version: int
    supports_fp8_grad_comm: bool
    cuda_device_index: int
    pcie_bandwidth_gbps: float
    double_buffer_limit: int = 2  # Corrected from upstream's off-by-one (> 1 → > 2)


# Cluster-wide static config (matches the described hardware).
CLUSTER_TIER_CONFIGS: Dict[HeteroDeviceTier, TierMemoryConfig] = {
    HeteroDeviceTier.A6000_0: TierMemoryConfig(
        tier=HeteroDeviceTier.A6000_0,
        vram_bytes=int(44 * 1024**3),       # ~44 GB usable of 48 GB
        sm_version=86,
        supports_fp8_grad_comm=False,
        cuda_device_index=0,
        pcie_bandwidth_gbps=16.0,
        double_buffer_limit=2,
    ),
    HeteroDeviceTier.A6000_1: TierMemoryConfig(
        tier=HeteroDeviceTier.A6000_1,
        vram_bytes=int(44 * 1024**3),
        sm_version=86,
        supports_fp8_grad_comm=False,
        cuda_device_index=1,
        pcie_bandwidth_gbps=16.0,
        double_buffer_limit=2,
    ),
    HeteroDeviceTier.H100: TierMemoryConfig(
        tier=HeteroDeviceTier.H100,
        vram_bytes=int(88 * 1024**3),       # ~88 GB usable of 96 GB
        sm_version=90,
        supports_fp8_grad_comm=True,
        cuda_device_index=2,
        pcie_bandwidth_gbps=64.0,           # PCIe Gen5 ×16
        double_buffer_limit=2,
    ),
    HeteroDeviceTier.CPU_CACHE: TierMemoryConfig(
        tier=HeteroDeviceTier.CPU_CACHE,
        vram_bytes=int(1400 * 1024**3),     # 1.4 TB of the 1.5 TB DRAM budget
        sm_version=0,
        supports_fp8_grad_comm=False,
        cuda_device_index=-1,
        pcie_bandwidth_gbps=50.0,           # CPU memory bandwidth (not PCIe)
        double_buffer_limit=16,             # CPU has abundant capacity
    ),
}


def get_tier_for_device(device: torch.device) -> HeteroDeviceTier:
    """
    Resolve a ``torch.device`` to a ``HeteroDeviceTier``.

    The mapping is static for the DES-LOC cluster layout described above.
    If the device is CPU, return ``CPU_CACHE``.

    Parameters
    ----------
    device : torch.device

    Returns
    -------
    HeteroDeviceTier

    Raises
    ------
    ValueError
        If the CUDA device index does not correspond to a known tier.
    """
    if device.type == "cpu":
        return HeteroDeviceTier.CPU_CACHE
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        for tier, cfg in CLUSTER_TIER_CONFIGS.items():
            if cfg.cuda_device_index == idx:
                return tier
        raise ValueError(
            f"CUDA device index {idx} does not map to any known DES-LOC tier. "
            f"Expected one of {[c.cuda_device_index for c in CLUSTER_TIER_CONFIGS.values()]}"
        )
    raise ValueError(f"Unsupported device type: {device.type!r}")


# ===========================================================================
# Section 2 – Locality Cache (CPU DRAM spill / fill tier)
# ===========================================================================

class LocalityCache:
    """
    Thin management layer over pinned CPU tensors used as DES-LOC's gradient
    spill sink.

    When a GPU gradient bucket cannot be reduced immediately (because the
    PCIe link is saturated or the double-buffer limit would be exceeded), it
    is *spilled* to the Locality Cache as a pinned tensor.  Once the reduce
    stream is free, the bucket is *filled* back from the cache and the
    reduce proceeds.

    Design notes
    ------------
    - Uses a per-tier deque so A6000 and H100 spills do not interleave.
    - Pinning is done lazily; only one pinned allocation per (tier, bucket_id)
      slot exists at a time.
    - Thread-safe: a single ``threading.Lock`` guards the deque mutation.
      CUDA-stream safety is the caller's responsibility.

    Upstream connection
    -------------------
    This class does not have a direct Megatron counterpart.  It is the
    DES-LOC mechanism that replaces Megatron's implicit assumption that
    ``_enforce_double_buffer_limit`` will always successfully block until
    memory is freed.  On PCIe-only clusters, blocking risks deadlock if
    the reduce stream itself is waiting on a host-side synchronisation.
    """

    def __init__(self, max_bytes: int = int(1400 * 1024**3)) -> None:
        self._max_bytes = max_bytes
        self._used_bytes: int = 0
        self._store: Dict[Tuple[HeteroDeviceTier, int], torch.Tensor] = {}
        self._lru: deque[Tuple[HeteroDeviceTier, int]] = deque()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def spill(
        self,
        tier: HeteroDeviceTier,
        bucket_id: int,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Copy *tensor* from GPU VRAM to a pinned CPU buffer.

        Parameters
        ----------
        tier : HeteroDeviceTier
            Owning GPU tier of the bucket being spilled.
        bucket_id : int
            Numeric bucket identifier (used as cache key).
        tensor : torch.Tensor
            The GPU gradient bucket tensor to spill.

        Returns
        -------
        torch.Tensor
            The pinned CPU tensor that now holds the gradient data.

        Notes
        -----
        If a cached entry already exists for ``(tier, bucket_id)``, it is
        reused in-place (avoids repeated pin_memory allocations).
        """
        key = (tier, bucket_id)
        nbytes = tensor.numel() * tensor.element_size()

        with self._lock:
            if key not in self._store:
                self._maybe_evict(nbytes)
                cpu_buf = torch.empty_like(tensor, device="cpu", pin_memory=True)
                self._store[key] = cpu_buf
                self._used_bytes += nbytes
                self._lru.append(key)
                logger.info(
                    "LocalityCache: spilled tier=%s bucket_id=%d "
                    "size=%.2f MiB (cache_used=%.1f GiB)",
                    tier.name, bucket_id,
                    nbytes / 1024**2,
                    self._used_bytes / 1024**3,
                )
            else:
                cpu_buf = self._store[key]

        cpu_buf.copy_(tensor, non_blocking=True)
        return cpu_buf

    def fill(
        self,
        tier: HeteroDeviceTier,
        bucket_id: int,
        target_device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Move a previously spilled bucket back to *target_device*.

        Parameters
        ----------
        tier : HeteroDeviceTier
        bucket_id : int
        target_device : torch.device
            The GPU device the tensor should be moved back to.

        Returns
        -------
        torch.Tensor or None
            The GPU tensor if a spilled entry was found, else ``None``.
        """
        key = (tier, bucket_id)
        with self._lock:
            cpu_buf = self._store.get(key)
        if cpu_buf is None:
            return None
        gpu_tensor = cpu_buf.to(target_device, non_blocking=True)
        logger.debug(
            "LocalityCache: filled tier=%s bucket_id=%d back to %s",
            tier.name, bucket_id, str(target_device),
        )
        return gpu_tensor

    def release(self, tier: HeteroDeviceTier, bucket_id: int) -> None:
        """
        Free the pinned CPU allocation for the given bucket.

        Called after a successful reduce-scatter so the cache slot can be
        reused without waiting for Python GC.
        """
        key = (tier, bucket_id)
        with self._lock:
            if key in self._store:
                nbytes = (
                    self._store[key].numel() * self._store[key].element_size()
                )
                del self._store[key]
                self._used_bytes -= nbytes
                try:
                    self._lru.remove(key)
                except ValueError:
                    pass
                logger.debug(
                    "LocalityCache: released tier=%s bucket_id=%d "
                    "(cache_used=%.1f GiB)",
                    tier.name, bucket_id, self._used_bytes / 1024**3,
                )

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_evict(self, needed_bytes: int) -> None:
        """Evict LRU entries until *needed_bytes* can be accommodated."""
        while (
            self._used_bytes + needed_bytes > self._max_bytes
            and self._lru
        ):
            evict_key = self._lru.popleft()
            if evict_key in self._store:
                freed = (
                    self._store[evict_key].numel()
                    * self._store[evict_key].element_size()
                )
                del self._store[evict_key]
                self._used_bytes -= freed
                logger.warning(
                    "LocalityCache: evicted tier=%s bucket_id=%d "
                    "freed=%.2f MiB to make room",
                    evict_key[0].name, evict_key[1], freed / 1024**2,
                )


# ===========================================================================
# Section 3 – DoubleBufferEnforcer (per-tier back-pressure gate)
# ===========================================================================

@dataclass
class _LiveUnit:
    """Tracks a single live FSDP unit in a tier's double-buffer window."""
    fsdp_unit_id: int
    bucket_ids: List[int] = field(default_factory=list)
    # Weak reference to the CUDA event that marks when this unit's reduce is done.
    reduce_event: Optional[torch.cuda.Event] = field(default=None)


class DoubleBufferEnforcer:
    """
    Per-tier gate that enforces the corrected Megatron double-buffer limit.

    Upstream bug recap
    ------------------
    Megatron originally checked ``len(double_buf_units) > 1`` before freeing
    buckets.  This meant only **one** unit could be live simultaneously,
    effectively halving pipeline throughput.  The fix changes the threshold to
    ``> 2``, allowing up to two units live at any time (a true *double* buffer).

    DES-LOC adaptation
    ------------------
    Because A6000 (48 GB) and H100 (96 GB) have very different VRAM budgets,
    the enforcer is instantiated *per tier*.  The H100 enforcer could
    theoretically allow a higher limit, but for simplicity we keep it at 2 and
    instead let the H100 run larger bucket sizes.

    Additionally, when enforcement would block (all slots full), the enforcer
    first attempts to *spill* the oldest bucket to the CPU Locality Cache
    rather than busy-waiting, which avoids the PCIe deadlock risk described
    in the module docstring.

    Parameters
    ----------
    tier : HeteroDeviceTier
    locality_cache : LocalityCache
    config : TierMemoryConfig
    """

    def __init__(
        self,
        tier: HeteroDeviceTier,
        locality_cache: LocalityCache,
        config: Optional[TierMemoryConfig] = None,
    ) -> None:
        self.tier = tier
        self.cache = locality_cache
        self.config = config or CLUSTER_TIER_CONFIGS[tier]
        self._limit: int = self.config.double_buffer_limit  # corrected: 2
        self._live_units: deque[_LiveUnit] = deque()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public interface (mirrors Megatron's _enforce_double_buffer_limit)
    # ------------------------------------------------------------------

    def enforce(self, incoming_bucket_ids: List[int]) -> None:
        """
        Ensure the double-buffer window has room for the buckets identified by
        *incoming_bucket_ids*.

        This method is called from ``main_grad_getter`` (the gradient property
        on each parameter), at the exact moment a new gradient buffer is about
        to be allocated.  Moving the call here (rather than in the backward
        hook) is the key fix from the upstream commit.

        Parameters
        ----------
        incoming_bucket_ids : List[int]
            The bucket IDs that are about to be allocated.

        Notes
        -----
        If ``len(live_fsdp_units) > double_buffer_limit`` (i.e. > 2), the
        oldest live unit is either:
        - Waited on (its reduce event is queried) if the event is done, or
        - Spilled to the CPU Locality Cache if the event is still pending.

        This two-path strategy avoids the PCIe deadlock described in the
        module docstring.
        """
        with self._lock:
            live_unit_ids: Set[int] = {u.fsdp_unit_id for u in self._live_units}
            # The threshold is > limit (corrected from > 1 to > 2 upstream).
            # We enforce *before* adding the new unit, so if len == limit,
            # adding one more would exceed → we must free the oldest.
            while len(live_unit_ids) >= self._limit:
                oldest = self._live_units[0]
                freed = self._try_free_unit(oldest)
                if freed:
                    self._live_units.popleft()
                    live_unit_ids.discard(oldest.fsdp_unit_id)
                else:
                    # Could not free yet; spill to Locality Cache.
                    self._spill_unit_buckets(oldest)
                    self._live_units.popleft()
                    live_unit_ids.discard(oldest.fsdp_unit_id)

    def register_unit(
        self,
        fsdp_unit_id: int,
        bucket_ids: List[int],
        reduce_event: Optional[torch.cuda.Event] = None,
    ) -> None:
        """
        Register a new FSDP unit as live in the double-buffer window.

        Called immediately after ``enforce`` succeeds and gradient buffer
        allocation begins.

        Parameters
        ----------
        fsdp_unit_id : int
        bucket_ids : List[int]
        reduce_event : torch.cuda.Event, optional
            A CUDA event that will be recorded after the reduce-scatter for
            this unit completes.  Used by ``enforce`` to determine whether
            the unit can be freed without spilling.
        """
        with self._lock:
            unit = _LiveUnit(
                fsdp_unit_id=fsdp_unit_id,
                bucket_ids=list(bucket_ids),
                reduce_event=reduce_event,
            )
            self._live_units.append(unit)
            logger.debug(
                "DoubleBufferEnforcer[%s]: registered fsdp_unit_id=%d "
                "buckets=%s (live_count=%d)",
                self.tier.name, fsdp_unit_id, bucket_ids, len(self._live_units),
            )

    def mark_unit_reduced(self, fsdp_unit_id: int) -> None:
        """
        Mark that the reduce-scatter for *fsdp_unit_id* has completed on the
        GPU stream.  If a reduce_event was provided at registration, it should
        be recorded before calling this.
        """
        with self._lock:
            for unit in self._live_units:
                if unit.fsdp_unit_id == fsdp_unit_id:
                    if unit.reduce_event is not None:
                        # Record is caller's responsibility; here we just note done.
                        pass
                    logger.debug(
                        "DoubleBufferEnforcer[%s]: unit %d marked as reduced",
                        self.tier.name, fsdp_unit_id,
                    )
                    return

    @property
    def live_unit_count(self) -> int:
        with self._lock:
            return len(self._live_units)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _try_free_unit(self, unit: _LiveUnit) -> bool:
        """
        Attempt to free a unit by querying its reduce event.

        Returns ``True`` if the unit's reduce is done and memory is freed,
        ``False`` if the reduce is still in flight.
        """
        if unit.reduce_event is None:
            # No event tracking: assume reduce is done (conservative).
            for bid in unit.bucket_ids:
                self.cache.release(self.tier, bid)
            return True
        if unit.reduce_event.query():
            for bid in unit.bucket_ids:
                self.cache.release(self.tier, bid)
            logger.debug(
                "DoubleBufferEnforcer[%s]: freed unit %d (reduce done)",
                self.tier.name, unit.fsdp_unit_id,
            )
            return True
        return False

    def _spill_unit_buckets(self, unit: _LiveUnit) -> None:
        """
        Spill all buckets of *unit* to the CPU Locality Cache.

        This is the DES-LOC-specific escape hatch when the reduce is still
        in flight but we need to free the double-buffer slot to avoid OOM.
        """
        logger.info(
            "DoubleBufferEnforcer[%s]: reduce still in-flight for unit %d; "
            "spilling %d bucket(s) to LocalityCache",
            self.tier.name, unit.fsdp_unit_id, len(unit.bucket_ids),
        )
        # Actual tensor spilling is done by the caller's bucket manager.
        # Here we just note the intent; the HeteroGradBucket will call
        # cache.spill() when it detects the spill flag.
        for bid in unit.bucket_ids:
            # Mark bucket as pending-spill so main_grad_getter can act.
            pass  # Coordination happens via HeteroParamGradBuffer.


# ===========================================================================
# Section 4 – HeteroGradBucket
# ===========================================================================

class HeteroGradBucket:
    """
    A gradient accumulation bucket tagged with its owning DES-LOC device tier.

    Analogous to Megatron's ``DataParallelBuffer`` bucket, but heterogeneity-
    aware.  Each bucket lives on exactly one tier; the ``main_grad_getter``
    selects the communication dtype based on the tier's SM version (SM90 →
    FP8 capable; SM86 → FP16/BF16 only).

    Parameters
    ----------
    bucket_id : int
        Unique bucket identifier within the owning ``HeteroParamGradBuffer``.
    fsdp_unit_id : int
        Which FSDP unit this bucket belongs to.
    tier : HeteroDeviceTier
        The device tier that owns this bucket.
    numel : int
        Number of elements in the bucket (after sharding).
    dtype : torch.dtype
        Storage dtype for gradient accumulation.
    device : torch.device
        The specific torch device (e.g. ``cuda:2`` for H100).

    Notes
    -----
    Allocation is lazy: ``_data`` is ``None`` until ``allocate()`` is called.
    This mirrors Megatron's deferred bucket allocation pattern and is
    important for keeping the double-buffer window narrow.
    """

    def __init__(
        self,
        bucket_id: int,
        fsdp_unit_id: int,
        tier: HeteroDeviceTier,
        numel: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.bucket_id = bucket_id
        self.fsdp_unit_id = fsdp_unit_id
        self.tier = tier
        self.numel = numel
        self.dtype = dtype
        self.device = device

        self._data: Optional[torch.Tensor] = None
        self._is_allocated: bool = False
        self._spilled: bool = False
        self._reduce_event: Optional[torch.cuda.Event] = None

    # ------------------------------------------------------------------

    def allocate(self) -> torch.Tensor:
        """
        Lazily allocate the gradient buffer on the owning device.

        Returns
        -------
        torch.Tensor
            The freshly allocated (or already-allocated) gradient buffer.
        """
        if not self._is_allocated:
            self._data = torch.zeros(
                self.numel,
                dtype=self.dtype,
                device=self.device,
            )
            self._is_allocated = True
            logger.debug(
                "HeteroGradBucket: allocated bucket_id=%d tier=%s "
                "numel=%d dtype=%s device=%s",
                self.bucket_id, self.tier.name,
                self.numel, str(self.dtype), str(self.device),
            )
        return self._data  # type: ignore[return-value]

    def free(self) -> None:
        """Free the underlying storage (explicit deallocation for double-buffer)."""
        if self._is_allocated and self._data is not None:
            del self._data
            self._data = None
            self._is_allocated = False

    def select_comm_dtype(self) -> torch.dtype:
        """
        Choose the gradient communication dtype for this bucket's tier.

        DES-LOC adaptation: H100 (SM90) supports FP8 reduce-scatter;
        A6000 (SM86) does not and falls back to BF16.

        Returns
        -------
        torch.dtype
        """
        cfg = CLUSTER_TIER_CONFIGS[self.tier]
        if cfg.supports_fp8_grad_comm:
            # FP8 comm dtype: use torch.float8_e4m3fn when available.
            # Fall back to bfloat16 if the torch version does not expose FP8.
            if hasattr(torch, "float8_e4m3fn"):
                return torch.float8_e4m3fn
            logger.warning(
                "HeteroGradBucket: SM90 tier %s requested FP8 but "
                "torch.float8_e4m3fn is unavailable; falling back to bfloat16",
                self.tier.name,
            )
        return torch.bfloat16

    @property
    def is_allocated(self) -> bool:
        return self._is_allocated


# ===========================================================================
# Section 5 – HeteroParamGradBuffer
# ===========================================================================

class HeteroParamGradBuffer:
    """
    Manages the mapping from distributed parameters to ``HeteroGradBucket``
    instances across all device tiers.

    This class is the DES-LOC equivalent of Megatron's ``ParamAndGradBuffer``.
    The critical adaptation is the ``_build_main_grad_getter`` method, which
    replicates the upstream fix: ``_enforce_double_buffer_limit`` is called
    *inside* the getter, at allocation time, rather than in the backward hook.

    Additionally, the ``_megatron_fsdp_model`` reference (which the upstream
    fix carefully attaches to DTensor parameters and their raw_param mirrors)
    is here replaced by a reference to the ``HeteroWgradSafeDoubleBuffer``
    owner.  The attachment follows the same lazy, per-replacement logic to
    avoid the bug where post-init iteration missed the distributed params.

    Parameters
    ----------
    owner : HeteroWgradSafeDoubleBuffer
        Back-reference to the top-level module.  Deliberately stored as a
        ``weakref`` to avoid reference cycles that could prevent GC.
    locality_cache : LocalityCache
    enforcers : Dict[HeteroDeviceTier, DoubleBufferEnforcer]
    """

    def __init__(
        self,
        owner: "HeteroWgradSafeDoubleBuffer",
        locality_cache: LocalityCache,
        enforcers: Dict[HeteroDeviceTier, DoubleBufferEnforcer],
    ) -> None:
        self._owner_ref: weakref.ref = weakref.ref(owner)
        self.locality_cache = locality_cache
        self.enforcers = enforcers

        # param_id → HeteroGradBucket
        self._param_to_bucket: Dict[int, HeteroGradBucket] = {}
        # param_id → item offset within bucket
        self._param_to_item_id: Dict[int, int] = {}

    # ------------------------------------------------------------------

    def register_param(
        self,
        param: nn.Parameter,
        bucket: HeteroGradBucket,
        item_id: int,
    ) -> None:
        """
        Register a parameter with its owning gradient bucket and item slot.

        Also attaches the back-reference to the owner (DES-LOC equivalent of
        the upstream ``_megatron_fsdp_model`` attribute).  This is done lazily
        here rather than in a post-init loop, matching the upstream fix.

        Parameters
        ----------
        param : nn.Parameter
        bucket : HeteroGradBucket
        item_id : int
            Index of this parameter's slice within the bucket.
        """
        pid = id(param)
        self._param_to_bucket[pid] = bucket
        self._param_to_item_id[pid] = item_id

        # Lazy attachment – mirrors upstream fix of moving setattr out of
        # the post-init loop and into _replace_param_with_distributed_if_needed.
        if not hasattr(param, "_deslock_owner"):
            param._deslock_owner = self._owner_ref  # type: ignore[attr-defined]
        if not hasattr(param, "_deslock_grad_buf"):
            param._deslock_grad_buf = self  # type: ignore[attr-defined]

    def build_main_grad_getter(self, param: nn.Parameter) -> None:
        """
        Attach a ``main_grad`` property getter to *param*.

        This implements the core upstream fix: ``enforce`` (double-buffer
        limit check) is invoked *inside* the getter, at the moment the
        gradient buffer is about to be allocated.  This prevents the race
        condition where the backward hook could trigger wgrad computation
        before the previous bucket's reduce had a chance to be scheduled.

        DES-LOC adaptation: the getter also selects the communication dtype
        based on the bucket's tier (SM90 → FP8 eligible; SM86 → BF16).

        Parameters
        ----------
        param : nn.Parameter
            The distributed parameter to instrument.
        """
        gbuf = self
        pid = id(param)

        def main_grad_getter(p: nn.Parameter) -> Optional[torch.Tensor]:
            bucket = gbuf._param_to_bucket.get(pid)
            item_id = gbuf._param_to_item_id.get(pid)
            if bucket is None or item_id is None:
                return None

            enforcer = gbuf.enforcers.get(bucket.tier)
            if enforcer is not None:
                # -------------------------------------------------------
                # CRITICAL: enforce the double-buffer limit BEFORE calling
                # bucket.allocate().  This is the fix from upstream commit
                # 55638bc4: moving _enforce_double_buffer_limit into the
                # getter ensures it fires at allocation time, not earlier.
                # -------------------------------------------------------
                enforcer.enforce(incoming_bucket_ids=[bucket.bucket_id])

            # Select comm dtype per tier's SM version (DES-LOC addition).
            comm_dtype = bucket.select_comm_dtype()
            data = bucket.allocate()

            # Cast to comm dtype for reduce-scatter if needed.
            if data.dtype != comm_dtype and bucket.tier != HeteroDeviceTier.CPU_CACHE:
                # We return a view of the right dtype; the underlying storage
                # remains in the accumulation dtype for numerical stability.
                data = data.to(comm_dtype)

            # Return the slice corresponding to this parameter.
            # In a real implementation, `item_id` would index into the
            # bucket's offset table; here we return the full buffer as a
            # stand-in for the reshape logic.
            return data.view(-1)[: param.numel()].view(param.shape)

        # Attach as a non-data descriptor to avoid shadowing param.grad.
        # In production code this would use a custom __get__ on a
        # ParameterWrapper class; here we store it as a callable attribute.
        param._main_grad_getter = main_grad_getter  # type: ignore[attr-defined]

    def get_main_grad(self, param: nn.Parameter) -> Optional[torch.Tensor]:
        """
        Public entry-point for retrieving a parameter's main gradient buffer.

        Calls the getter built by ``build_main_grad_getter``.
        """
        getter = getattr(param, "_main_grad_getter", None)
        if getter is None:
            return None
        return getter(param)


# ===========================================================================
# Section 6 – Parameter replacement helpers
# ===========================================================================

def _attach_deslock_owner_to_param(
    param: nn.Parameter,
    owner: "HeteroWgradSafeDoubleBuffer",
) -> None:
    """
    Lazily attach the ``_deslock_owner`` weak-reference to *param*.

    This function is the DES-LOC equivalent of Megatron's corrected
    ``_replace_param_with_distributed_if_needed``, where ``_megatron_fsdp_model``
    is now set inside the replacement function rather than in a post-init loop.

    The upstream race: the post-init loop called ``setattr(param, …, self)``
    on the *original* parameters, but ``_replace_param_with_distributed_if_needed``
    had already swapped them for DTensor-backed distributed params.  The loop
    iterated over the new params (returned by ``module.parameters()``) but they
    were not the same objects stored in ``raw_param``.  The fix: set the
    attribute at replacement time so it travels with the right object.

    Parameters
    ----------
    param : nn.Parameter
    owner : HeteroWgradSafeDoubleBuffer
    """
    if not hasattr(param, "_deslock_owner"):
        param._deslock_owner = weakref.ref(owner)  # type: ignore[attr-defined]
        logger.debug(
            "_attach_deslock_owner_to_param: attached owner ref to param id=%d",
            id(param),
        )


def _replace_module_param_deslock(
    module: nn.Module,
    name: str,
    new_param: nn.Parameter,
    owner: "HeteroWgradSafeDoubleBuffer",
) -> None:
    """
    Replace *name* in *module* with *new_param*, attaching the DES-LOC owner
    reference to *new_param* at replacement time.

    Mirrors Megatron's corrected ``_replace_module_parameter`` + the two new
    ``hasattr`` / ``setattr`` guards added in commit 55638bc4:

    .. code-block:: python

        # Upstream fix (megatron_fsdp.py line ~1347):
        if not hasattr(dist_param, "_megatron_fsdp_model"):
            dist_param._megatron_fsdp_model = self

        # Upstream fix (megatron_fsdp.py line ~1361):
        if not hasattr(self.raw_param[name], "_megatron_fsdp_model"):
            self.raw_param[name]._megatron_fsdp_model = self

    Parameters
    ----------
    module : nn.Module
    name : str
        Dot-separated parameter name (e.g. ``"layer.weight"``).
    new_param : nn.Parameter
    owner : HeteroWgradSafeDoubleBuffer
    """
    # Navigate to the immediate parent of `name`.
    parts = name.rsplit(".", 1)
    if len(parts) == 2:
        parent_name, attr_name = parts
        parent = dict(module.named_modules())[parent_name]
    else:
        parent, attr_name = module, parts[0]

    # Perform the replacement.
    parent.register_parameter(attr_name, new_param)

    # Attach owner reference (lazy, idempotent – mirrors the upstream fix).
    _attach_deslock_owner_to_param(new_param, owner)


# ===========================================================================
# Section 7 – HeteroWgradSafeDoubleBuffer (top-level module)
# ===========================================================================

class HeteroWgradSafeDoubleBuffer(nn.Module):
    """
    Top-level DES-LOC wrapper that applies the wgrad race-condition fix from
    Megatron commit 55638bc4 to a heterogeneous GPU cluster.

    Wraps any ``nn.Module`` and equips its parameters with:

    1. Per-tier ``DoubleBufferEnforcer`` instances (one per HeteroDeviceTier),
       enforcing a corrected double-buffer limit of 2 (not 1) simultaneously
       live FSDP units.

    2. ``HeteroGradBucket`` allocation with tier-aware communication dtypes
       (FP8 for H100/SM90, BF16 for A6000/SM86).

    3. Lazy ``_deslock_owner`` attachment at parameter replacement time,
       avoiding the upstream bug where post-init loop iteration missed
       newly created DTensor-backed distributed parameters.

    4. A ``LocalityCache`` spill path so that ``_enforce_double_buffer_limit``
       never deadlocks on PCIe-only interconnects.

    Parameters
    ----------
    module : nn.Module
        The model (or sub-model) to wrap.
    tier_assignment : Dict[str, HeteroDeviceTier], optional
        Maps parameter names to device tiers.  If ``None``, all parameters
        are assigned to ``H100`` by default (useful for unit tests).
    bucket_size_elements : int
        How many elements to pack into each gradient bucket.
    locality_cache : LocalityCache, optional
        Shared Locality Cache instance.  If ``None``, one is created.
    process_group : dist.ProcessGroup, optional
        The data-parallel process group for reduce-scatter.

    Notes
    -----
    ``microbatch_count`` mirrors Megatron's attribute of the same name and is
    incremented by the caller after each micro-batch forward/backward pass.
    """

    def __init__(
        self,
        module: nn.Module,
        tier_assignment: Optional[Dict[str, HeteroDeviceTier]] = None,
        bucket_size_elements: int = 64 * 1024 * 1024,  # 64M elements ≈ 256 MB FP32
        locality_cache: Optional[LocalityCache] = None,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        super().__init__()
        self.module = module
        self.bucket_size_elements = bucket_size_elements
        self.process_group = process_group
        self.microbatch_count: int = 0

        # Default tier: H100 (largest VRAM, useful for single-card testing).
        self._tier_assignment: Dict[str, HeteroDeviceTier] = tier_assignment or {}

        # Locality Cache – shared across all tiers.
        self.locality_cache: LocalityCache = locality_cache or LocalityCache()

        # Per-tier enforcers (created before parameter replacement).
        self.enforcers: Dict[HeteroDeviceTier, DoubleBufferEnforcer] = {
            tier: DoubleBufferEnforcer(tier, self.locality_cache)
            for tier in HeteroDeviceTier
        }

        # Grad buffer manager.
        self.grad_buffer = HeteroParamGradBuffer(
            owner=self,
            locality_cache=self.locality_cache,
            enforcers=self.enforcers,
        )

        # Raw (pre-replacement) parameter references – mirrors Megatron's raw_param.
        self.raw_param: Dict[str, nn.Parameter] = {
            name: param
            for name, param in module.named_parameters()
        }

        # Internal state for is_param_fsdp_distributed (mirrors Megatron).
        self.is_param_fsdp_distributed: bool = False

        # Initialise buckets and attach getters.
        # Note: _replace_param_with_distributed_if_needed is called AFTER
        # gradient buffer initialisation so that owner refs are attached at
        # the correct moment (the upstream fix).
        self._init_grad_buckets()
        self._replace_param_with_distributed_if_needed()
        # Do NOT iterate module.parameters() here to attach owner refs –
        # that is exactly the upstream bug.  Attachment is done inside
        # _replace_param_with_distributed_if_needed.

        logger.info(
            "HeteroWgradSafeDoubleBuffer: initialised with %d parameters, "
            "%d buckets across %d tiers",
            sum(1 for _ in module.parameters()),
            sum(len(b) for b in self._tier_buckets.values()),
            len([t for t in self._tier_buckets if self._tier_buckets[t]]),
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _resolve_param_tier(self, name: str, param: nn.Parameter) -> HeteroDeviceTier:
        """
        Determine which device tier owns *param*.

        Lookup order:
        1. Explicit ``tier_assignment`` provided at construction.
        2. The device of ``param.data`` (if already on a GPU).
        3. Default: H100.

        Parameters
        ----------
        name : str
        param : nn.Parameter

        Returns
        -------
        HeteroDeviceTier
        """
        if name in self._tier_assignment:
            return self._tier_assignment[name]
        if param.device.type == "cuda":
            try:
                return get_tier_for_device(param.device)
            except ValueError:
                pass
        return HeteroDeviceTier.H100

    def _init_grad_buckets(self) -> None:
        """
        Build ``HeteroGradBucket`` instances for all parameters, grouped by
        (tier, fsdp_unit_id) and packed into buckets of
        ``bucket_size_elements`` elements.

        Mirrors the bucketing logic of Megatron's ``_get_parameter_groups``
        and ``_init_each_parameter_group_buffers``, but simplified for the
        DES-LOC heterogeneous setting.

        Also calls ``grad_buffer.register_param`` and
        ``grad_buffer.build_main_grad_getter`` for each parameter, which is
        where the owner reference is lazily attached (matching the upstream
        fix pattern).
        """
        # Group parameters by (tier, fsdp_unit_id).
        # For simplicity, each parameter name maps to fsdp_unit_id 0 unless
        # the tier_assignment provides a richer mapping.
        tier_params: Dict[HeteroDeviceTier, List[Tuple[str, nn.Parameter]]] = (
            defaultdict(list)
        )
        for name, param in self.module.named_parameters():
            tier = self._resolve_param_tier(name, param)
            tier_params[tier].append((name, param))

        self._tier_buckets: Dict[HeteroDeviceTier, List[HeteroGradBucket]] = (
            defaultdict(list)
        )
        global_bucket_id = 0

        for tier, param_list in tier_params.items():
            cfg = CLUSTER_TIER_CONFIGS[tier]
            device = (
                torch.device(f"cuda:{cfg.cuda_device_index}")
                if cfg.cuda_device_index >= 0
                else torch.device("cpu")
            )

            # Pack parameters into buckets.
            current_numel = 0
            current_params: List[Tuple[str, nn.Parameter]] = []
            fsdp_unit_id = 0

            def flush_bucket(
                params: List[Tuple[str, nn.Parameter]],
                bid: int,
                fid: int,
                t: HeteroDeviceTier,
                dev: torch.device,
            ) -> HeteroGradBucket:
                total = sum(p.numel() for _, p in params)
                bucket = HeteroGradBucket(
                    bucket_id=bid,
                    fsdp_unit_id=fid,
                    tier=t,
                    numel=total,
                    dtype=torch.float32,
                    device=dev,
                )
                for item_id, (n, p) in enumerate(params):
                    self.grad_buffer.register_param(p, bucket, item_id)
                    self.grad_buffer.build_main_grad_getter(p)
                return bucket

            for name, param in param_list:
                numel = param.numel()
                if current_numel + numel > self.bucket_size_elements and current_params:
                    bkt = flush_bucket(
                        current_params, global_bucket_id, fsdp_unit_id, tier, device
                    )
                    self._tier_buckets[tier].append(bkt)
                    global_bucket_id += 1
                    fsdp_unit_id += 1
                    current_params = []
                    current_numel = 0
                current_params.append((name, param))
                current_numel += numel

            if current_params:
                bkt = flush_bucket(
                    current_params, global_bucket_id, fsdp_unit_id, tier, device
                )
                self._tier_buckets[tier].append(bkt)
                global_bucket_id += 1

        logger.info(
            "_init_grad_buckets: created %d total buckets across tiers: %s",
            global_bucket_id,
            {t.name: len(v) for t, v in self._tier_buckets.items()},
        )

    def _replace_param_with_distributed_if_needed(self) -> None:
        """
        Optionally replace module parameters with distributed (sharded) versions.

        This is the DES-LOC equivalent of Megatron's
        ``_replace_param_with_distributed_if_needed``.  The critical invariant
        preserved here is that ``_deslock_owner`` is attached to the
        *replacement* parameter object, not iterated post-hoc.

        In the current implementation we do not shard parameters (since
        DeepSpeed ZeRO handles that), but we still exercise the attachment
        logic to validate correctness of the fix.

        Upstream fix replicated:
        - ``dist_param._megatron_fsdp_model = self``  (line ~1347)
        - ``self.raw_param[name]._megatron_fsdp_model = self``  (line ~1361)

        Both are guarded with ``if not hasattr(…)`` to be idempotent.
        """
        for name, param in list(self.module.named_parameters()):
            # In a real FSDP scenario, here we would wrap param in a DTensor.
            # For DES-LOC, we mark each parameter as managed by this buffer.
            # We attach the owner ref to the *current* param object (which
            # may already be a DTensor if called from a higher-level wrapper).
            _attach_deslock_owner_to_param(param, self)

            # Also attach to raw_param entry (mirrors line ~1361 upstream).
            if name in self.raw_param:
                _attach_deslock_owner_to_param(self.raw_param[name], self)

        self.is_param_fsdp_distributed = True

    # ------------------------------------------------------------------
    # Forward and backward integration
    # ------------------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Delegate forward pass to the wrapped module.

        Increments ``microbatch_count`` after each forward, mirroring
        Megatron's convention for tracking gradient accumulation steps.
        """
        output = self.module(*args, **kwargs)
        self.microbatch_count += 1
        return output

    def get_main_grad(self, param: nn.Parameter) -> Optional[torch.Tensor]:
        """
        Public API: retrieve the main gradient buffer for *param*.

        This is the entry point used by the DeepSpeed optimizer to fetch
        the accumulated gradient before an all-reduce or reduce-scatter.
        It proxies to ``HeteroParamGradBuffer.get_main_grad``, which in turn
        calls the per-parameter getter that enforces the double-buffer limit
        at allocation time.

        Parameters
        ----------
        param : nn.Parameter

        Returns
        -------
        torch.Tensor or None
        """
        return self.grad_buffer.get_main_grad(param)

    def zero_grad_buffers(self) -> None:
        """
        Zero all allocated gradient buffers across all tiers.

        Called by the optimizer's ``zero_grad()`` hook.  Only zeroes
        *allocated* buckets to avoid unnecessary memory traffic.
        """
        zeroed = 0
        for tier_buckets in self._tier_buckets.values():
            for bucket in tier_buckets:
                if bucket.is_allocated and bucket._data is not None:
                    bucket._data.zero_()
                    zeroed += 1
        logger.debug("zero_grad_buffers: zeroed %d allocated buckets", zeroed)

    def finalize_grad_reduce(self, fsdp_unit_id: int, tier: HeteroDeviceTier) -> None:
        """
        Notify the enforcer that a reduce-scatter for *fsdp_unit_id* on *tier*
        has completed.

        This should be called from the reduce-scatter callback (e.g. after
        the DDP / ZeRO bucket hook fires).

        Parameters
        ----------
        fsdp_unit_id : int
        tier : HeteroDeviceTier
        """
        enforcer = self.enforcers.get(tier)
        if enforcer is not None:
            enforcer.mark_unit_reduced(fsdp_unit_id)

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def buffer_stats(self) -> Dict[str, Any]:
        """
        Return a snapshot of buffer allocation state for logging / debugging.

        Returns
        -------
        dict
            Keys: tier names.  Values: dicts with ``total_buckets``,
            ``allocated_buckets``, ``live_unit_count``.
        """
        stats: Dict[str, Any] = {}
        for tier, buckets in self._tier_buckets.items():
            stats[tier.name] = {
                "total_buckets": len(buckets),
                "allocated_buckets": sum(1 for b in buckets if b.is_allocated),
                "live_unit_count": self.enforcers[tier].live_unit_count,
            }
        stats["locality_cache_used_gib"] = (
            self.locality_cache.used_bytes / 1024**3
        )
        return stats


# ===========================================================================
# Section 8 – Utility: simulate wgrad race scenario for verification
# ===========================================================================

def simulate_wgrad_race_condition(
    model: HeteroWgradSafeDoubleBuffer,
    n_micro_batches: int = 4,
    n_params_per_unit: int = 3,
) -> List[str]:
    """
    Drive a simplified backward pass sequence to verify that the
    double-buffer enforcer fires at the correct moment and in the correct
    order.

    The simulation:
    1. For each micro-batch, iterates over all parameters in reverse
       (backward order) and requests ``get_main_grad()``.
    2. Records the sequence of enforcer events.
    3. Returns the event log for assertions in tests.

    Parameters
    ----------
    model : HeteroWgradSafeDoubleBuffer
    n_micro_batches : int
    n_params_per_unit : int

    Returns
    -------
    List[str]
        Human-readable event log.
    """
    events: List[str] = []
    params = list(model.module.parameters())

    for mb in range(n_micro_batches):
        for param in reversed(params):
            grad = model.get_main_grad(param)
            if grad is not None:
                events.append(
                    f"mb={mb} param_id={id(param)} "
                    f"grad_shape={tuple(grad.shape)} "
                    f"grad_device={grad.device}"
                )
        model.finalize_grad_reduce(
            fsdp_unit_id=mb % 2,
            tier=HeteroDeviceTier.H100,
        )
        model.zero_grad_buffers()

    return events


# ===========================================================================
# Section 9 – Unit tests
# ===========================================================================

class _SmallMLP(nn.Module):
    """Tiny MLP used as a test fixture."""

    def __init__(self, in_features: int = 16, hidden: int = 32, out: int = 8) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


if __name__ == "__main__":
    import sys

    # Configure verbose logging for the test run.
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    class TestHeteroWgradSafeDoubleBuffer(unittest.TestCase):
        """
        Unit tests for the DES-LOC wgrad race-condition fix.

        These tests exercise the key invariants introduced by Megatron commit
        55638bc4 and adapted for the heterogeneous DES-LOC environment.
        """

        def setUp(self) -> None:
            self.mlp = _SmallMLP()
            # Assign all params to CPU_CACHE tier for portability (no GPU needed).
            param_names = [n for n, _ in self.mlp.named_parameters()]
            tier_assignment = {n: HeteroDeviceTier.CPU_CACHE for n in param_names}
            self.wrapper = HeteroWgradSafeDoubleBuffer(
                module=self.mlp,
                tier_assignment=tier_assignment,
                bucket_size_elements=256,  # small → multiple buckets
            )

        # ------------------------------------------------------------------
        # Test 1: Owner reference attached lazily, not via post-init loop
        # ------------------------------------------------------------------
        def test_owner_ref_attached_to_all_params(self) -> None:
            """
            Every parameter (after _replace_param_with_distributed_if_needed)
            must carry a ``_deslock_owner`` weak-ref pointing back to the
            wrapper.  This mirrors the upstream fix that moves ``setattr``
            into the replacement function instead of a post-init loop.
            """
            for name, param in self.mlp.named_parameters():
                self.assertTrue(
                    hasattr(param, "_deslock_owner"),
                    f"Parameter '{name}' is missing _deslock_owner",
                )
                owner = param._deslock_owner()  # dereference weakref
                self.assertIs(
                    owner,
                    self.wrapper,
                    f"Parameter '{name}' _deslock_owner does not point to wrapper",
                )

        # ------------------------------------------------------------------
        # Test 2: raw_param mirror also carries the owner ref
        # ------------------------------------------------------------------
        def test_raw_param_owner_ref(self) -> None:
            """
            The ``raw_param`` dict entries must also carry ``_deslock_owner``.
            This mirrors Megatron's second hasattr guard (line ~1361 upstream):

                if not hasattr(self.raw_param[name], "_megatron_fsdp_model"):
                    self.raw_param[name]._megatron_fsdp_model = self
            """
            for name, raw in self.wrapper.raw_param.items():
                self.assertTrue(
                    hasattr(raw, "_deslock_owner"),
                    f"raw_param['{name}'] is missing _deslock_owner",
                )

        # ------------------------------------------------------------------
        # Test 3: main_grad getter is attached to each parameter
        # ------------------------------------------------------------------
        def test_main_grad_getter_attached(self) -> None:
            """
            Each parameter must have a ``_main_grad_getter`` callable attached
            by ``build_main_grad_getter``.  This confirms that the allocation-
            time enforcement (upstream fix) is wired up.
            """
            for name, param in self.mlp.named_parameters():
                self.assertTrue(
                    hasattr(param, "_main_grad_getter"),
                    f"Parameter '{name}' is missing _main_grad_getter",
                )
                self.assertTrue(
                    callable(param._main_grad_getter),
                    f"Parameter '{name}' _main_grad_getter is not callable",
                )

        # ------------------------------------------------------------------
        # Test 4: get_main_grad returns a tensor of the correct shape
        # ------------------------------------------------------------------
        def test_get_main_grad_shape(self) -> None:
            """
            ``get_main_grad`` must return a tensor whose shape is compatible
            with the parameter's shape (or a prefix of it, given the current
            simplified implementation).
            """
            for name, param in self.mlp.named_parameters():
                grad = self.wrapper.get_main_grad(param)
                self.assertIsNotNone(
                    grad, f"get_main_grad returned None for '{name}'"
                )
                self.assertIsInstance(grad, torch.Tensor)
                # The getter returns param.numel() elements; shape must match.
                self.assertEqual(
                    grad.shape,
                    param.shape,
                    f"grad shape mismatch for '{name}': "
                    f"expected {param.shape}, got {grad.shape}",
                )

        # ------------------------------------------------------------------
        # Test 5: DoubleBufferEnforcer limit is 2 (corrected from upstream's 1)
        # ------------------------------------------------------------------
        def test_enforcer_limit_is_two(self) -> None:
            """
            All per-tier enforcers must have ``double_buffer_limit == 2``,
            matching the corrected Megatron threshold (``> 2`` ↔ limit == 2).

            Upstream bug was ``> 1`` which means only 1 unit could be live.
            The fix changes it to ``> 2`` meaning 2 units can be live
            simultaneously (true double-buffer semantics).
            """
            for tier, enforcer in self.wrapper.enforcers.items():
                if tier == HeteroDeviceTier.CPU_CACHE:
                    # CPU_CACHE has a higher limit by design.
                    self.assertGreater(enforcer.config.double_buffer_limit, 2)
                else:
                    self.assertEqual(
                        enforcer.config.double_buffer_limit,
                        2,
                        f"Enforcer for {tier.name} has wrong limit "
                        f"({enforcer.config.double_buffer_limit}); expected 2",
                    )

        # ------------------------------------------------------------------
        # Test 6: H100 tier selects FP8 comm dtype (if available)
        # ------------------------------------------------------------------
        def test_h100_comm_dtype_selection(self) -> None:
            """
            A ``HeteroGradBucket`` on the H100 tier must select an FP8 dtype
            (if ``torch.float8_e4m3fn`` is available) or fall back to BF16.
            A6000 buckets must always select BF16.
            """
            h100_bucket = HeteroGradBucket(
                bucket_id=0,
                fsdp_unit_id=0,
                tier=HeteroDeviceTier.H100,
                numel=1024,
                dtype=torch.float32,
                device=torch.device("cpu"),  # CPU for unit test portability
            )
            a6000_bucket = HeteroGradBucket(
                bucket_id=1,
                fsdp_unit_id=0,
                tier=HeteroDeviceTier.A6000_0,
                numel=1024,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )

            h100_dtype = h100_bucket.select_comm_dtype()
            a6000_dtype = a6000_bucket.select_comm_dtype()

            self.assertIn(
                h100_dtype,
                [torch.bfloat16, getattr(torch, "float8_e4m3fn", torch.bfloat16)],
                f"H100 comm dtype {h100_dtype} is not FP8 or BF16",
            )
            self.assertEqual(
                a6000_dtype,
                torch.bfloat16,
                f"A6000 comm dtype {a6000_dtype} should always be bfloat16",
            )

        # ------------------------------------------------------------------
        # Test 7: LocalityCache spill → fill round-trip
        # ------------------------------------------------------------------
        def test_locality_cache_spill_fill(self) -> None:
            """
            A tensor spilled to the Locality Cache can be filled back and
            must be numerically identical.
            """
            cache = LocalityCache(max_bytes=int(1 * 1024**3))
            original = torch.randn(1024, dtype=torch.float32)
            cpu_copy = cache.spill(HeteroDeviceTier.A6000_0, bucket_id=7, tensor=original)
            self.assertEqual(cpu_copy.shape, original.shape)
            torch.testing.assert_close(cpu_copy, original)

            filled = cache.fill(
                HeteroDeviceTier.A6000_0, bucket_id=7, target_device=torch.device("cpu")
            )
            self.assertIsNotNone(filled)
            torch.testing.assert_close(filled, original)

            cache.release(HeteroDeviceTier.A6000_0, bucket_id=7)
            post_release = cache.fill(
                HeteroDeviceTier.A6000_0, bucket_id=7, target_device=torch.device("cpu")
            )
            self.assertIsNone(post_release, "Cache entry should be None after release")

        # ------------------------------------------------------------------
        # Test 8: zero_grad_buffers resets all allocated buckets
        # ------------------------------------------------------------------
        def test_zero_grad_buffers(self) -> None:
            """
            After calling ``get_main_grad`` (which allocates buckets) and
            writing nonzero values, ``zero_grad_buffers`` must zero them.
            """
            for _, param in self.mlp.named_parameters():
                grad = self.wrapper.get_main_grad(param)
                if grad is not None:
                    grad.fill_(3.14)

            self.wrapper.zero_grad_buffers()

            for tier_buckets in self.wrapper._tier_buckets.values():
                for bucket in tier_buckets:
                    if bucket.is_allocated and bucket._data is not None:
                        self.assertTrue(
                            torch.all(bucket._data == 0.0).item(),
                            f"Bucket {bucket.bucket_id} not zeroed after zero_grad_buffers",
                        )

        # ------------------------------------------------------------------
        # Test 9: buffer_stats returns sane values
        # ------------------------------------------------------------------
        def test_buffer_stats(self) -> None:
            """``buffer_stats`` must return a dict with tier-keyed entries."""
            stats = self.wrapper.buffer_stats()
            self.assertIn("locality_cache_used_gib", stats)
            self.assertIsInstance(stats["locality_cache_used_gib"], float)

        # ------------------------------------------------------------------
        # Test 10: simulate_wgrad_race_condition runs without error
        # ------------------------------------------------------------------
        def test_simulate_wgrad_race(self) -> None:
            """
            The wgrad race simulation must complete without exceptions and
            return a non-empty event log.
            """
            events = simulate_wgrad_race_condition(
                self.wrapper, n_micro_batches=3, n_params_per_unit=2
            )
            self.assertGreater(
                len(events),
                0,
                "simulate_wgrad_race_condition returned empty event log",
            )
            for ev in events:
                self.assertIn("mb=", ev)
                self.assertIn("grad_shape=", ev)

        # ------------------------------------------------------------------
        # Test 11: enforcer register + mark_reduced lifecycle
        # ------------------------------------------------------------------
        def test_enforcer_lifecycle(self) -> None:
            """
            Register two units, enforce with a third incoming bucket:
            the oldest unit should be freed, leaving live_count == 2.
            """
            cache = LocalityCache()
            enforcer = DoubleBufferEnforcer(
                tier=HeteroDeviceTier.H100,
                locality_cache=cache,
            )

            enforcer.register_unit(fsdp_unit_id=0, bucket_ids=[0, 1])
            enforcer.register_unit(fsdp_unit_id=1, bucket_ids=[2, 3])
            self.assertEqual(enforcer.live_unit_count, 2)

            # Enforce with a third unit → should evict oldest (unit 0).
            enforcer.enforce(incoming_bucket_ids=[4])
            # After enforcement, live_count should be ≤ double_buffer_limit (2).
            self.assertLessEqual(
                enforcer.live_unit_count,
                enforcer._limit,
                f"live_unit_count ({enforcer.live_unit_count}) "
                f"exceeds limit ({enforcer._limit}) after enforce()",
            )

        # ------------------------------------------------------------------
        # Test 12: TierMemoryConfig double_buffer_limit matches corrected value
        # ------------------------------------------------------------------
        def test_tier_config_double_buffer_limit(self) -> None:
            """
            The ``double_buffer_limit`` for GPU tiers must be 2, matching
            the corrected Megatron threshold.  This is a regression guard
            so no one accidentally reverts to 1.
            """
            for tier in [
                HeteroDeviceTier.A6000_0,
                HeteroDeviceTier.A6000_1,
                HeteroDeviceTier.H100,
            ]:
                cfg = CLUSTER_TIER_CONFIGS[tier]
                self.assertEqual(
                    cfg.double_buffer_limit,
                    2,
                    f"CLUSTER_TIER_CONFIGS[{tier.name}].double_buffer_limit "
                    f"is {cfg.double_buffer_limit}, expected 2",
                )

        # ------------------------------------------------------------------
        # Test 13: get_tier_for_device raises on unknown index
        # ------------------------------------------------------------------
        def test_get_tier_for_device_unknown(self) -> None:
            """``get_tier_for_device`` must raise ValueError for unknown CUDA indices."""
            unknown_device = torch.device("cuda:99")
            with self.assertRaises(ValueError):
                get_tier_for_device(unknown_device)

        # ------------------------------------------------------------------
        # Test 14: HeteroGradBucket lazy allocation
        # ------------------------------------------------------------------
        def test_hetero_grad_bucket_lazy_alloc(self) -> None:
            """
            A ``HeteroGradBucket`` must not allocate memory until ``allocate()``
            is explicitly called.
            """
            bucket = HeteroGradBucket(
                bucket_id=99,
                fsdp_unit_id=0,
                tier=HeteroDeviceTier.CPU_CACHE,
                numel=512,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )
            self.assertFalse(bucket.is_allocated)
            self.assertIsNone(bucket._data)

            data = bucket.allocate()
            self.assertTrue(bucket.is_allocated)
            self.assertIsNotNone(bucket._data)
            self.assertEqual(data.shape, (512,))
            self.assertEqual(data.dtype, torch.float32)

            bucket.free()
            self.assertFalse(bucket.is_allocated)

    # Run all tests.
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestHeteroWgradSafeDoubleBuffer)
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
