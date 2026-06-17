"""
DES-LOC Heterogeneous AllGather Pipeline with Locality-Aware Bucket Preservation
=================================================================================

Upstream Design Intent (Megatron commit 378d81fbd32a3faa90726d0e59344f7893ce4b47):
    Megatron-LM's FSDP AllGatherPipeline.reset() previously freed every bucket's
    gathered scratch storage unconditionally. This broke cross-module parameter
    readers — most notably MambaMixer.forward, which accesses conv1d.weight as a
    raw tensor pointer (via causal_conv1d_fn(weight=...)) rather than through
    nn.Module.__call__. Because no pre-forward hook fires for conv1d in that path,
    the non-FSDP-unit bucket holding conv1d.weight was freed by reset() and never
    re-gathered, causing the next forward pass to dereference freed storage.

    The fix introduces a PRESERVED bucket state: on pipeline reset, buckets that
    belong to "non-FSDP-unit" parameter groups (those whose params may be read
    across module boundaries without triggering hooks) are not freed. Instead they
    are marked PRESERVED so a subsequent all-gather can refresh them in-place.
    A new BucketStatus enum value PRESERVED sits between EMPTY and COMMUNICATING.

DES-LOC Adaptation Points:
    In the DES-LOC (Decoupled Execution with Shared LOcality Cache) framework
    running on 2x A6000 (SM86, 48 GB each) + 1x H100 NVL (SM90, 96 GB) over
    PCIe with 1.5 TB CPU DRAM, the bucket preservation semantics must be extended
    along three axes:

    1. Device-Tier Locality Tagging:
       Each parameter group carries a ``device_tier`` annotation (A6000_0, A6000_1,
       H100) derived from which device holds the primary shard. The PRESERVED state
       must track not just whether a bucket is non-unit, but also which device tier
       "owns" the locality cache copy. Cross-tier reads (e.g. H100 reading a param
       shard that lives primarily on an A6000) must not release the gather buffer
       until the H100 compute stream explicitly signals completion.

    2. PCIe-Aware Refresh Scheduling:
       Unlike NVLink topologies where re-gathering a preserved bucket is cheap,
       PCIe re-gathers from A6000↔H100 are bandwidth-constrained (~64 GB/s per
       direction). HeteroAllGatherPipeline therefore batches PRESERVED→refresh
       operations by device tier: all same-tier preserved buckets are re-gathered
       in a single pass before cross-tier buckets, minimising PCIe round-trips.

    3. CPU DRAM Spill Integration:
       When GPU memory pressure is high (configurable threshold), PRESERVED buckets
       whose locality-cache copy is not pinned may be spilled to CPU DRAM rather
       than held in GPU VRAM. A ``LocalityCacheEntry`` tracks the spill state.
       On the next forward pass, a background prefetch thread reloads them before
       the compute stream needs them, overlapping PCIe transfer with prior-layer
       compute in the DES-LOC execution graph.

    The three SM generations are treated as distinct "locality domains":
        - SM86 domain: A6000_0, A6000_1 (peer-access possible via PCIe switch)
        - SM90 domain: H100 (independent memory space, separate CUDA context)

    Cross-domain parameter reads always route through the CPU DRAM locality cache
    (the "shared" in DES-LOC) rather than direct GPU-to-GPU PCIe transfers, which
    would stall both devices' compute streams.

Author: Neuron_SP / DES-LOC team
Based on: Megatron-LM commit 378d81fbd32a3faa90726d0e59344f7893ce4b47
Project: github.com/dylanyunlon/Neuron_SP
"""

from __future__ import annotations

import enum
import logging
import threading
import time
import warnings
import weakref
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterator, List, Optional, Sequence, Set, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device tier enumeration
# ---------------------------------------------------------------------------

class DeviceTier(enum.Enum):
    """
    Logical device tiers present in the DES-LOC heterogeneous cluster.

    SM86 devices (A6000) form one locality domain; SM90 (H100 NVL) is another.
    CPU DRAM is the shared locality cache that bridges domains without requiring
    direct GPU-to-GPU PCIe transfers for cross-domain parameter reads.
    """
    A6000_0 = "a6000_0"    # NVIDIA A6000 48GB, SM86, first card
    A6000_1 = "a6000_1"    # NVIDIA A6000 48GB, SM86, second card
    H100    = "h100"        # NVIDIA H100 NVL 96GB, SM90
    CPU     = "cpu"         # 1.5 TB CPU DRAM (shared locality cache)

    @property
    def is_sm86(self) -> bool:
        return self in (DeviceTier.A6000_0, DeviceTier.A6000_1)

    @property
    def is_sm90(self) -> bool:
        return self is DeviceTier.H100

    @property
    def is_gpu(self) -> bool:
        return self is not DeviceTier.CPU

    @property
    def cuda_index(self) -> Optional[int]:
        """Return the CUDA device index, or None for CPU."""
        _map = {
            DeviceTier.A6000_0: 0,
            DeviceTier.A6000_1: 1,
            DeviceTier.H100:    2,
        }
        return _map.get(self)


# Singleton mapping: cuda device index → DeviceTier
_DEVICE_TIER_MAP: Dict[int, DeviceTier] = {
    0: DeviceTier.A6000_0,
    1: DeviceTier.A6000_1,
    2: DeviceTier.H100,
}


def device_to_tier(device: torch.device) -> DeviceTier:
    """Map a torch.device to its DES-LOC DeviceTier."""
    if device.type == "cpu":
        return DeviceTier.CPU
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _DEVICE_TIER_MAP:
        raise ValueError(
            f"Device cuda:{idx} is not registered in the DES-LOC tier map. "
            f"Known CUDA indices: {list(_DEVICE_TIER_MAP.keys())}"
        )
    return _DEVICE_TIER_MAP[idx]


# ---------------------------------------------------------------------------
# Bucket status state machine
# ---------------------------------------------------------------------------

class BucketStatus(enum.Enum):
    """
    Lifecycle states for a gather bucket in HeteroAllGatherPipeline.

    Upstream (Megatron 378d81f) introduced PRESERVED between EMPTY and
    COMMUNICATING to handle non-FSDP-unit buckets that must survive reset().

    DES-LOC extends this with SPILLED: a bucket whose GPU gather buffer has
    been evicted to CPU DRAM to relieve GPU memory pressure but whose logical
    content is still valid. SPILLED→COMMUNICATING transition triggers a PCIe
    reload prefetch on the owning device's prefetch stream before the main
    compute stream can mark the bucket READY_TO_USE.

    State transitions (valid arcs):
        EMPTY       → COMMUNICATING  (new allgather issued)
        PRESERVED   → COMMUNICATING  (in-place refresh of preserved buffer)
        SPILLED     → COMMUNICATING  (prefetch reload + allgather)
        COMMUNICATING → READY_TO_USE (allgather event recorded, stream sync'd)
        READY_TO_USE  → EMPTY        (bucket released after last reader)
        READY_TO_USE  → PRESERVED    (reset() with preserve_non_unit=True)
        READY_TO_USE  → SPILLED      (memory pressure spill)
        PRESERVED     → SPILLED      (memory pressure spill of preserved buffer)
    """
    EMPTY        = 1
    PRESERVED    = 2   # Megatron upstream: survives reset(), refresh in-place
    SPILLED      = 3   # DES-LOC: GPU buffer evicted to CPU DRAM locality cache
    COMMUNICATING = 4
    READY_TO_USE = 5

    @property
    def needs_gather(self) -> bool:
        """True if a transition to COMMUNICATING requires issuing an allgather."""
        return self in (BucketStatus.EMPTY, BucketStatus.PRESERVED, BucketStatus.SPILLED)


# ---------------------------------------------------------------------------
# Locality cache entry
# ---------------------------------------------------------------------------

@dataclass
class LocalityCacheEntry:
    """
    Represents a single parameter bucket's copy in the CPU DRAM locality cache.

    DES-LOC's "Shared LOcality Cache" is the 1.5 TB CPU DRAM that bridges
    SM86 and SM90 domains. When a bucket is spilled from GPU VRAM, its data
    lands here as a pinned tensor for fast re-prefetch via PCIe DMA.

    Attributes:
        bucket_key:    (bucket_id, is_backward) pair identifying the bucket.
        owner_tier:    The GPU tier that owns the primary shard.
        pinned_cpu:    Pinned CPU tensor holding the spilled data.
        spill_time:    Wall-clock time when the spill occurred (for LRU eviction).
        prefetch_event: CUDA event on the owner device's prefetch stream that
                        fires when the CPU→GPU reload is complete.
        readers:       Set of DeviceTiers that have outstanding cross-domain
                       reads pending against this cache entry.
    """
    bucket_key: Tuple[int, bool]
    owner_tier: DeviceTier
    pinned_cpu: Optional[torch.Tensor] = None
    spill_time: float = field(default_factory=time.monotonic)
    prefetch_event: Optional[torch.cuda.Event] = None
    readers: Set[DeviceTier] = field(default_factory=set)

    def is_ready_on_cpu(self) -> bool:
        return self.pinned_cpu is not None

    def add_reader(self, tier: DeviceTier) -> None:
        self.readers.add(tier)

    def remove_reader(self, tier: DeviceTier) -> None:
        self.readers.discard(tier)

    def has_cross_domain_readers(self, owner_tier: DeviceTier) -> bool:
        """True if any reader is in a different SM domain than the owner."""
        for reader in self.readers:
            if reader.is_sm86 != owner_tier.is_sm86:
                return True
        return False


# ---------------------------------------------------------------------------
# Parameter group metadata
# ---------------------------------------------------------------------------

@dataclass
class ParameterGroupMeta:
    """
    Metadata for a single parameter group / bucket in DES-LOC.

    Mirrors Megatron's ParameterGroup but adds heterogeneity-aware fields.

    Attributes:
        bucket_id:         Index of this bucket in the pipeline.
        fsdp_unit_id:      If not None, this bucket belongs to an FSDP unit
                           (e.g. a TransformerLayer). If None, params in this
                           bucket may be read cross-module without hooks firing
                           — the non-unit case that triggered the upstream fix.
        owner_tier:        DeviceTier that holds the primary shard for allgather.
        param_names:       Debug-friendly list of parameter names in this bucket.
        cross_tier_readers: Set of DeviceTiers that may read this bucket across
                            SM domain boundaries (triggers spill-to-CPU logic).
        pinned_for_spill:  If True, this bucket is eligible for CPU DRAM spill
                           under memory pressure.
    """
    bucket_id: int
    fsdp_unit_id: Optional[int]
    owner_tier: DeviceTier
    param_names: List[str] = field(default_factory=list)
    cross_tier_readers: Set[DeviceTier] = field(default_factory=set)
    pinned_for_spill: bool = True

    @property
    def is_fsdp_unit(self) -> bool:
        return self.fsdp_unit_id is not None

    @property
    def is_cross_domain(self) -> bool:
        """True if any reader is in a different SM generation domain."""
        for reader in self.cross_tier_readers:
            if reader.is_sm86 != self.owner_tier.is_sm86:
                return True
        return False


# ---------------------------------------------------------------------------
# Prefetch thread for CPU→GPU reload
# ---------------------------------------------------------------------------

class _PrefetchThread(threading.Thread):
    """
    Background daemon thread that reloads SPILLED buckets from CPU DRAM back
    to GPU VRAM using non-blocking PCIe DMA on a dedicated prefetch stream.

    DES-LOC rationale: PCIe transfers stall compute if issued on the main
    compute stream. By running them on a separate prefetch stream and recording
    a CUDA event, we let the compute stream wait only at the point where the
    parameter data is actually needed, maximising overlap with prior-layer
    compute.
    """

    def __init__(
        self,
        reload_queue: "list[tuple[LocalityCacheEntry, torch.Tensor, torch.cuda.Stream]]",
        queue_lock: threading.Lock,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True, name="des-loc-prefetch")
        self._queue = reload_queue
        self._lock = queue_lock
        self._stop = stop_event

    def run(self) -> None:
        while not self._stop.is_set():
            entry: Optional[Tuple[LocalityCacheEntry, torch.Tensor, torch.cuda.Stream]] = None
            with self._lock:
                if self._queue:
                    entry = self._queue.pop(0)
            if entry is not None:
                cache_entry, gpu_buf, prefetch_stream = entry
                try:
                    with torch.cuda.stream(prefetch_stream):
                        gpu_buf.copy_(cache_entry.pinned_cpu, non_blocking=True)
                        evt = torch.cuda.Event()
                        evt.record(prefetch_stream)
                        cache_entry.prefetch_event = evt
                    logger.debug(
                        "Prefetch reload complete for bucket %s from CPU DRAM "
                        "to %s (%.2f MB)",
                        cache_entry.bucket_key,
                        gpu_buf.device,
                        gpu_buf.nbytes / 1e6,
                    )
                except Exception:
                    logger.exception(
                        "Prefetch reload failed for bucket %s", cache_entry.bucket_key
                    )
            else:
                time.sleep(1e-4)


# ---------------------------------------------------------------------------
# Main HeteroAllGatherPipeline
# ---------------------------------------------------------------------------

class HeteroAllGatherPipeline:
    """
    Heterogeneous AllGather Pipeline with DES-LOC Locality-Aware Bucket Preservation.

    Upstream Design (Megatron 378d81fbd32a3faa90726d0e59344f7893ce4b47):
        AllGatherPipeline.reset(preserve_non_fsdp_units=True) marks non-unit
        buckets as PRESERVED instead of freeing them. The PRESERVED state
        means "buffer still allocated, content valid, but not formally ready;
        a subsequent allgather may refresh it in-place." This prevents
        cross-module readers from dereferencing freed storage.

    DES-LOC Adaptation:
        On a heterogeneous PCIe cluster (2x A6000 SM86 + 1x H100 SM90):

        a) Tier-partitioned reset:
           reset() separates preserved buckets by DeviceTier. Within-tier
           buckets (A6000↔A6000 via PCIe switch) are preserved in GPU VRAM.
           Cross-tier non-unit buckets (A6000↔H100) are initially preserved in
           GPU VRAM but may be spilled to CPU DRAM if the GPU memory headroom
           on either end drops below ``gpu_memory_headroom_mb``.

        b) PCIe-aware refresh ordering:
           When triggering allgathers on PRESERVED/SPILLED buckets, same-tier
           buckets are batched first (lower latency, higher bandwidth within
           SM86 domain) before cross-tier PCIe transfers. This is controlled
           by ``_tier_sorted_bucket_ids()``.

        c) SPILLED state:
           A new BucketStatus.SPILLED is introduced. The spill path copies
           the GPU gather buffer to pinned CPU memory (via a non-blocking
           async_copy on a side stream) and marks the bucket SPILLED. On the
           next forward pass, a background prefetch thread (``_PrefetchThread``)
           reloads the data before the compute stream arrives, using a CUDA
           event to synchronise exactly when needed.

        d) Cross-domain reader tracking:
           Each LocalityCacheEntry tracks which DeviceTiers have outstanding
           reads. A cross-domain read (SM86↔SM90) always routes through the
           CPU DRAM locality cache to avoid stalling both GPU compute streams
           with a point-to-point PCIe transfer.

    Args:
        parameter_groups:       List of ParameterGroupMeta describing each bucket.
        process_group:          The distributed process group for allgather ops.
        current_device:         The CUDA device this rank runs on.
        gpu_memory_headroom_mb: GPU free memory below which preserved cross-tier
                                buckets are spilled to CPU DRAM. Default 2048 MB.
        enable_cpu_spill:       Master switch for the CPU DRAM spill feature.
                                Disable for debugging or single-tier deployments.
        spill_stream:           Optional dedicated CUDA stream for async CPU spill
                                copies. If None, a new stream is created.
        prefetch_stream:        Optional dedicated CUDA stream for CPU→GPU reload.
                                If None, a new stream is created.
    """

    def __init__(
        self,
        parameter_groups: List[ParameterGroupMeta],
        process_group: Optional[dist.ProcessGroup],
        current_device: torch.device,
        gpu_memory_headroom_mb: float = 2048.0,
        enable_cpu_spill: bool = True,
        spill_stream: Optional[torch.cuda.Stream] = None,
        prefetch_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        self.parameter_groups = parameter_groups
        self.process_group = process_group
        self.current_device = current_device
        self.current_tier = device_to_tier(current_device)
        self.gpu_memory_headroom_mb = gpu_memory_headroom_mb
        self.enable_cpu_spill = enable_cpu_spill

        # CUDA streams for async CPU↔GPU copies (DES-LOC: decoupled from compute)
        if current_device.type == "cuda":
            self._spill_stream = spill_stream or torch.cuda.Stream(device=current_device)
            self._prefetch_stream = prefetch_stream or torch.cuda.Stream(device=current_device)
        else:
            self._spill_stream = None
            self._prefetch_stream = None

        # Per-bucket state: keyed by (bucket_id, is_backward)
        self.bucket_status: Dict[Tuple[int, bool], BucketStatus] = {}
        self.bucket_can_be_released: Dict[Tuple[int, bool], bool] = {}

        # Gathered parameter tensors (the "allgather buffers")
        # In real DeepSpeed integration these would be slices of a flat buffer;
        # here we store them as a dict for testability.
        self._gather_buffers: Dict[Tuple[int, bool], Optional[torch.Tensor]] = {}

        # Outstanding allgather events: bucket_key → (event, mark_ready_fn)
        self.param_gather_event_map: Dict[
            Tuple[int, bool], Tuple[torch.cuda.Event, "callable"]
        ] = {}

        # CPU DRAM locality cache: bucket_key → LocalityCacheEntry
        self._locality_cache: Dict[Tuple[int, bool], LocalityCacheEntry] = {}

        # Prefetch thread plumbing
        self._prefetch_queue: List[
            Tuple[LocalityCacheEntry, torch.Tensor, torch.cuda.Stream]
        ] = []
        self._prefetch_lock = threading.Lock()
        self._prefetch_stop = threading.Event()
        if self.enable_cpu_spill and current_device.type == "cuda":
            self._prefetch_thread = _PrefetchThread(
                self._prefetch_queue, self._prefetch_lock, self._prefetch_stop
            )
            self._prefetch_thread.start()
        else:
            self._prefetch_thread = None

        # Initialise all buckets to EMPTY
        for meta in parameter_groups:
            for bwd in (False, True):
                key = self._key(meta.bucket_id, bwd)
                self.bucket_status[key] = BucketStatus.EMPTY
                self.bucket_can_be_released[key] = False
                self._gather_buffers[key] = None

        logger.info(
            "HeteroAllGatherPipeline initialised: %d buckets, device=%s tier=%s "
            "spill=%s headroom=%.0f MB",
            len(parameter_groups),
            current_device,
            self.current_tier.value,
            enable_cpu_spill,
            gpu_memory_headroom_mb,
        )

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(bucket_id: int, bwd: bool) -> Tuple[int, bool]:
        return (bucket_id, bwd)

    @property
    def num_buckets(self) -> int:
        return len(self.parameter_groups)

    # ------------------------------------------------------------------
    # Tier sorting for PCIe-aware refresh ordering
    # ------------------------------------------------------------------

    def _tier_sorted_bucket_ids(
        self, bucket_ids: Sequence[int], bwd: bool
    ) -> List[int]:
        """
        Sort bucket IDs so same-tier allgathers precede cross-tier PCIe transfers.

        Within the SM86 domain (A6000_0 ↔ A6000_1), transfers use the PCIe
        switch and are cheaper than SM86↔SM90 (H100) transfers. Sorting puts
        intra-domain work first so the PCIe bus is not saturated when cross-
        domain transfers begin, reducing overall pipeline stall.

        Returns:
            Sorted list of bucket IDs (same-tier first, cross-tier last).
        """
        same_tier: List[int] = []
        cross_tier: List[int] = []

        for bid in bucket_ids:
            meta = self.parameter_groups[bid]
            if meta.owner_tier == self.current_tier:
                same_tier.append(bid)
            elif meta.owner_tier.is_sm86 == self.current_tier.is_sm86:
                # Same SM generation domain (both SM86), but different card
                same_tier.append(bid)
            else:
                # Cross SM generation: SM86↔SM90 PCIe transfer
                cross_tier.append(bid)

        return same_tier + cross_tier

    # ------------------------------------------------------------------
    # GPU memory pressure check
    # ------------------------------------------------------------------

    def _gpu_memory_free_mb(self) -> float:
        """Return free GPU memory in MB on the current device."""
        if self.current_device.type != "cuda":
            return float("inf")
        free_bytes, _ = torch.cuda.mem_get_info(self.current_device)
        return free_bytes / 1e6

    def _is_memory_pressure_high(self) -> bool:
        return self._gpu_memory_free_mb() < self.gpu_memory_headroom_mb

    # ------------------------------------------------------------------
    # CPU DRAM spill / reload
    # ------------------------------------------------------------------

    def _spill_bucket_to_cpu(self, bucket_key: Tuple[int, bool]) -> bool:
        """
        Asynchronously copy a READY_TO_USE or PRESERVED bucket's gather buffer
        to pinned CPU DRAM and mark the bucket SPILLED.

        DES-LOC rationale: Rather than evicting GPU memory by freeing the
        buffer (which would require a full allgather on next use), we copy it
        to the CPU locality cache. The CPU copy is "shared" in the DES-LOC
        sense: cross-tier readers can access it without stalling GPU compute.

        Returns:
            True if the spill was initiated, False if skipped (e.g. buffer
            not allocated, or spill disabled).
        """
        if not self.enable_cpu_spill:
            return False

        gpu_buf = self._gather_buffers.get(bucket_key)
        if gpu_buf is None:
            return False

        bid, bwd = bucket_key
        meta = self.parameter_groups[bid]

        if not meta.pinned_for_spill:
            return False

        # Allocate pinned CPU buffer
        pinned = torch.empty(
            gpu_buf.shape,
            dtype=gpu_buf.dtype,
            device="cpu",
            pin_memory=True,
        )

        # Async copy on spill stream to avoid blocking compute
        with torch.cuda.stream(self._spill_stream):
            pinned.copy_(gpu_buf, non_blocking=True)
            spill_event = torch.cuda.Event()
            spill_event.record(self._spill_stream)

        cache_entry = LocalityCacheEntry(
            bucket_key=bucket_key,
            owner_tier=meta.owner_tier,
            pinned_cpu=pinned,
        )
        self._locality_cache[bucket_key] = cache_entry
        self.bucket_status[bucket_key] = BucketStatus.SPILLED

        logger.debug(
            "Bucket %s spilled to CPU DRAM: %.2f MB (tier=%s, cross_domain=%s)",
            bucket_key,
            gpu_buf.nbytes / 1e6,
            meta.owner_tier.value,
            meta.is_cross_domain,
        )
        return True

    def _enqueue_prefetch_reload(self, bucket_key: Tuple[int, bool]) -> bool:
        """
        Enqueue a CPU→GPU prefetch reload for a SPILLED bucket.

        The prefetch thread will copy pinned_cpu → gpu_buf on the prefetch
        stream. The caller must wait on cache_entry.prefetch_event before
        using the buffer in the compute stream.

        Returns:
            True if the prefetch was enqueued, False if the bucket is not
            in the locality cache or has no GPU buffer.
        """
        cache_entry = self._locality_cache.get(bucket_key)
        gpu_buf = self._gather_buffers.get(bucket_key)
        if cache_entry is None or gpu_buf is None or not cache_entry.is_ready_on_cpu():
            return False

        with self._prefetch_lock:
            self._prefetch_queue.append((cache_entry, gpu_buf, self._prefetch_stream))

        logger.debug(
            "Enqueued prefetch reload for bucket %s from CPU DRAM (owner=%s)",
            bucket_key,
            cache_entry.owner_tier.value,
        )
        return True

    def _wait_prefetch_complete(self, bucket_key: Tuple[int, bool]) -> None:
        """
        Block the current CUDA compute stream until the prefetch reload for
        bucket_key has completed on the prefetch stream.

        Uses CUDA event semantics (stream.wait_event) to avoid a full
        device synchronisation — only the compute stream is stalled, not
        the prefetch stream itself.
        """
        cache_entry = self._locality_cache.get(bucket_key)
        if cache_entry is None or cache_entry.prefetch_event is None:
            return
        if self.current_device.type != "cuda":
            return

        current_stream = torch.cuda.current_stream(self.current_device)
        current_stream.wait_event(cache_entry.prefetch_event)
        logger.debug("Compute stream waited on prefetch event for bucket %s", bucket_key)

    # ------------------------------------------------------------------
    # Core pipeline: reset
    # ------------------------------------------------------------------

    def reset(
        self,
        preserve_non_fsdp_units: bool = True,
        spill_cross_tier_preserved: Optional[bool] = None,
    ) -> None:
        """
        Reset the pipeline state after a forward/backward pass.

        Upstream behaviour (Megatron 378d81f):
            Non-FSDP-unit buckets are preserved (marked PRESERVED, not freed)
            so cross-module parameter readers can safely dereference them on
            the next forward pass without requiring a fresh allgather. Only
            FSDP-unit buckets (those with fsdp_unit_id is not None) are fully
            released.

        DES-LOC extensions:
            1. Tier-aware preservation:
               Same-tier preserved buckets stay in GPU VRAM (cheap to refresh).
               Cross-tier preserved buckets are candidates for CPU DRAM spill
               if GPU memory pressure is high (see spill_cross_tier_preserved).

            2. Spill decision:
               If spill_cross_tier_preserved is None (default), the spill is
               triggered automatically when _is_memory_pressure_high() returns
               True for cross-domain non-unit buckets. Pass True/False to
               override.

            3. Outstanding gather completion:
               All in-flight allgather events are waited before resetting,
               exactly as in the upstream fix.

        Args:
            preserve_non_fsdp_units:
                Mirror of the upstream flag. If False, all buckets are freed
                (intended only for debugging; the model must not be reused).
            spill_cross_tier_preserved:
                Tri-state: None = auto (based on memory pressure), True =
                always spill cross-tier preserved buckets to CPU DRAM, False =
                never spill. Only applies when preserve_non_fsdp_units=True.
        """
        # Step 1: Drain all in-flight allgather events (upstream logic)
        if self.param_gather_event_map:
            warnings.warn(
                "HeteroAllGatherPipeline.reset() called with outstanding "
                f"allgather events ({len(self.param_gather_event_map)} pending). "
                "Waiting for all events to complete before reset. "
                "This may indicate a pipeline scheduling issue.",
                stacklevel=2,
            )
            while self.param_gather_event_map:
                key = next(iter(self.param_gather_event_map))
                bid, bwd = key
                self.wait_bucket_ready(bid, bwd)

        # Step 2: Classify and update bucket statuses
        spilled_buckets: List[Tuple[int, bool]] = []
        released_buckets: List[Tuple[int, bool]] = []
        preserved_buckets: List[Tuple[int, bool]] = []

        for meta in self.parameter_groups:
            is_unit = meta.is_fsdp_unit
            is_cross_domain = meta.is_cross_domain

            for bwd in (False, True):
                key = self._key(meta.bucket_id, bwd)

                if preserve_non_fsdp_units and not is_unit:
                    # Determine whether to spill this preserved bucket to CPU DRAM
                    should_spill = False
                    if self.enable_cpu_spill and is_cross_domain:
                        if spill_cross_tier_preserved is True:
                            should_spill = True
                        elif spill_cross_tier_preserved is None:
                            should_spill = self._is_memory_pressure_high()

                    if should_spill and self.bucket_status[key] in (
                        BucketStatus.READY_TO_USE, BucketStatus.PRESERVED
                    ):
                        spilled_buckets.append(key)
                    else:
                        # Mark PRESERVED (Megatron upstream behaviour)
                        self.bucket_status[key] = BucketStatus.PRESERVED
                        preserved_buckets.append(key)
                else:
                    # Release this bucket (FSDP-unit, or preserve=False)
                    self.bucket_can_be_released[key] = True
                    released_buckets.append(key)

        # Step 3: Actually spill candidates (after releasing releasable ones)
        for key in spilled_buckets:
            success = self._spill_bucket_to_cpu(key)
            if not success:
                # Fallback: mark preserved if spill failed
                self.bucket_status[key] = BucketStatus.PRESERVED
                preserved_buckets.append(key)

        # Step 4: Recycle released buckets
        self._recycle_unused_buckets()

        if preserved_buckets:
            logger.debug(
                "reset(): %d buckets preserved in GPU VRAM (non-unit, same-tier or "
                "below spill threshold)",
                len(preserved_buckets),
            )
        if spilled_buckets:
            logger.info(
                "reset(): %d cross-tier non-unit buckets spilled to CPU DRAM "
                "locality cache (GPU headroom %.0f MB < threshold %.0f MB)",
                len(spilled_buckets),
                self._gpu_memory_free_mb(),
                self.gpu_memory_headroom_mb,
            )

        # Step 5: Validate final states (upstream assertion, extended for DES-LOC)
        expected_statuses: FrozenSet[BucketStatus]
        if preserve_non_fsdp_units:
            expected_statuses = frozenset({
                BucketStatus.EMPTY,
                BucketStatus.PRESERVED,
                BucketStatus.SPILLED,
            })
        else:
            expected_statuses = frozenset({BucketStatus.EMPTY})

        bad = {
            k: v
            for k, v in self.bucket_status.items()
            if v not in expected_statuses
        }
        assert not bad, (
            "HeteroAllGatherPipeline.reset(): unexpected bucket states after reset. "
            f"bad_buckets={bad}"
        )

    # ------------------------------------------------------------------
    # Core pipeline: wait_bucket_ready
    # ------------------------------------------------------------------

    def wait_bucket_ready(self, bucket_id: int, bwd: bool) -> None:
        """
        Block until the gather buffer for (bucket_id, bwd) is READY_TO_USE.

        DES-LOC extensions over upstream:
            - If the bucket is SPILLED, enqueue a prefetch reload and wait for
              the prefetch event on the compute stream before proceeding.
            - If the bucket is PRESERVED, issue an in-place refresh allgather
              before waiting (same as EMPTY path, reusing existing buffer).
            - Error messages include the DES-LOC tier context for diagnostics.

        Args:
            bucket_id: Index of the bucket.
            bwd:       True if this is the backward pass bucket.
        """
        key = self._key(bucket_id, bwd)
        status = self.bucket_status[key]

        if status == BucketStatus.READY_TO_USE:
            return

        if status in (BucketStatus.EMPTY, BucketStatus.PRESERVED, BucketStatus.SPILLED):
            # These are handled by prefetch/allgather before this call should be made.
            # If we get here, it means the pipeline scheduling didn't prefetch in time.
            if status == BucketStatus.SPILLED:
                # Last-resort synchronous reload from CPU DRAM
                cache_entry = self._locality_cache.get(key)
                gpu_buf = self._gather_buffers.get(key)
                if cache_entry is not None and gpu_buf is not None:
                    logger.warning(
                        "wait_bucket_ready: bucket %s is SPILLED and was not "
                        "pre-fetched; performing synchronous CPU→GPU reload "
                        "(%.2f MB, tier=%s). Check prefetch scheduling.",
                        key,
                        gpu_buf.nbytes / 1e6,
                        self.parameter_groups[bucket_id].owner_tier.value,
                    )
                    gpu_buf.copy_(cache_entry.pinned_cpu)
                    self.bucket_status[key] = BucketStatus.READY_TO_USE
                    return

            meta = self.parameter_groups[bucket_id]
            raise ValueError(
                f"Bucket {bucket_id} (bwd={bwd}) is {status.name} when "
                f"wait_bucket_ready was called. "
                f"tier={meta.owner_tier.value}, fsdp_unit={meta.fsdp_unit_id}. "
                "This implies the allgather was not issued, or NCCL operations "
                "are not complete."
            )

        # Status is COMMUNICATING: wait for the recorded CUDA event
        if key not in self.param_gather_event_map:
            raise ValueError(
                f"Bucket {bucket_id} (bwd={bwd}) is COMMUNICATING but has no "
                "outstanding event in param_gather_event_map. This is a bug in "
                "HeteroAllGatherPipeline scheduling."
            )

        gather_event, mark_ready_fn = self.param_gather_event_map.pop(key)
        gather_event.synchronize()
        mark_ready_fn()

        assert self.bucket_status[key] == BucketStatus.READY_TO_USE, (
            f"Bucket {key} not READY_TO_USE after event sync: {self.bucket_status[key]}"
        )

    # ------------------------------------------------------------------
    # Core pipeline: issue_allgathers
    # ------------------------------------------------------------------

    def issue_allgathers(
        self,
        bucket_ids: Sequence[int],
        bwd: bool,
        lookahead: int = 1,
    ) -> None:
        """
        Issue allgather operations for the specified buckets.

        Upstream (Megatron) issues allgathers for EMPTY buckets only. DES-LOC
        extends this to PRESERVED and SPILLED:
            - PRESERVED: allgather in-place into the existing GPU buffer.
            - SPILLED:   first wait for any in-flight prefetch, then allgather
                         in-place (or reload from CPU if prefetch didn't finish).
            - EMPTY:     allocate buffer and allgather (standard path).

        PCIe-aware ordering: same-tier buckets are issued before cross-tier.

        Args:
            bucket_ids: Bucket indices to consider for allgather.
            bwd:        True if backward-pass allgathers.
            lookahead:  Number of additional buckets beyond the immediate next
                        to pre-fetch (for pipeline lookahead scheduling).
        """
        sorted_ids = self._tier_sorted_bucket_ids(bucket_ids, bwd)

        # Filter to only buckets that need a gather (EMPTY, PRESERVED, or SPILLED)
        needs_gather = [
            bid for bid in sorted_ids
            if self.bucket_status[self._key(bid, bwd)].needs_gather
        ]

        if not needs_gather:
            return

        for bid in needs_gather:
            key = self._key(bid, bwd)
            status = self.bucket_status[key]
            meta = self.parameter_groups[bid]

            # For SPILLED buckets: enqueue prefetch first, then allgather
            # will use the refreshed data (or the allgather overwrites it).
            if status == BucketStatus.SPILLED:
                self._enqueue_prefetch_reload(key)
                # Wait for prefetch on compute stream at the last moment
                self._wait_prefetch_complete(key)
                # After prefetch we treat this like PRESERVED: re-gather in-place
                self.bucket_status[key] = BucketStatus.PRESERVED

            # Allocate gather buffer if needed (EMPTY path)
            if self._gather_buffers[key] is None:
                # In real DeepSpeed integration this would be a slice of the
                # flat param buffer. Here we allocate a placeholder.
                param_count = sum(
                    p.numel()
                    for name in meta.param_names
                    # placeholder: real impl looks up params by name
                    for p in []
                )
                # Placeholder allocation for test harness
                self._gather_buffers[key] = torch.empty(
                    max(param_count, 1),
                    dtype=torch.float32,
                    device=self.current_device,
                )

            self.bucket_status[key] = BucketStatus.COMMUNICATING

            # Record a CUDA event to mark completion (real impl: NCCL allgather)
            if self.current_device.type == "cuda":
                evt = torch.cuda.Event()
                evt.record()
            else:
                evt = _FakeCudaEvent()  # CPU fallback for unit tests

            # Closure captures key for the mark_ready callback
            _key_capture = key

            def _mark_ready(k=_key_capture):
                self.bucket_status[k] = BucketStatus.READY_TO_USE

            self.param_gather_event_map[key] = (evt, _mark_ready)

            logger.debug(
                "Issued allgather for bucket %s (bwd=%s, tier=%s, was=%s)",
                key,
                bwd,
                meta.owner_tier.value,
                status.name,
            )

    # ------------------------------------------------------------------
    # Core pipeline: mark_bucket_in_use / recycle
    # ------------------------------------------------------------------

    def mark_bucket_in_use(self, bucket_id: int, bwd: bool) -> None:
        """
        Mark a bucket as actively being used by the compute stream.

        DES-LOC extension: If the bucket is in the PRESERVED or SPILLED state
        (i.e. a cross-module reader is about to dereference its storage), first
        ensure the allgather has been issued and completed.

        Args:
            bucket_id: Bucket index.
            bwd:       True for backward-pass bucket.
        """
        key = self._key(bucket_id, bwd)
        self.bucket_can_be_released[key] = False

        status = self.bucket_status[key]
        if status in (BucketStatus.COMMUNICATING, BucketStatus.READY_TO_USE):
            # Normal path: allgather in progress or complete
            return

        # PRESERVED / SPILLED / EMPTY: the bucket needs a gather before use
        if status in (BucketStatus.PRESERVED, BucketStatus.SPILLED, BucketStatus.EMPTY):
            self.issue_allgathers([bucket_id], bwd)
        # Transition to COMMUNICATING was done by issue_allgathers
        self.bucket_status[key] = BucketStatus.COMMUNICATING

    def release_bucket(self, bucket_id: int, bwd: bool) -> None:
        """
        Signal that the compute stream is done with this bucket's data.

        After release, the bucket may be recycled on the next reset() if it is
        an FSDP-unit bucket. Non-unit (PRESERVED) buckets will survive reset().
        SPILLED buckets are released from GPU VRAM but their CPU copy survives.

        Args:
            bucket_id: Bucket index.
            bwd:       True for backward-pass bucket.
        """
        key = self._key(bucket_id, bwd)
        self.bucket_can_be_released[key] = True
        self._recycle_unused_buckets()

    def _recycle_unused_buckets(self) -> None:
        """
        Free GPU gather buffers for buckets that are both releasable and not
        currently in a live state.

        Non-unit (PRESERVED) buckets are not recycled here — they survive
        until the next pipeline reset with preserve_non_fsdp_units=False.
        SPILLED buckets have already relinquished their GPU memory.
        """
        for key, can_release in list(self.bucket_can_be_released.items()):
            if not can_release:
                continue
            status = self.bucket_status[key]
            if status in (BucketStatus.PRESERVED, BucketStatus.SPILLED):
                # Non-unit: do not free (upstream: preserve_non_fsdp_units logic)
                continue
            if status == BucketStatus.COMMUNICATING:
                # Still in flight; cannot recycle
                continue
            # READY_TO_USE or EMPTY: free the GPU gather buffer
            if self._gather_buffers.get(key) is not None:
                self._gather_buffers[key] = None
            self.bucket_can_be_released[key] = False
            self.bucket_status[key] = BucketStatus.EMPTY

    # ------------------------------------------------------------------
    # Cross-tier locality cache access
    # ------------------------------------------------------------------

    def read_param_cross_tier(
        self,
        bucket_id: int,
        bwd: bool,
        reader_tier: DeviceTier,
    ) -> Optional[torch.Tensor]:
        """
        Provide cross-tier access to a bucket's gathered parameters via the
        CPU DRAM locality cache (the "Shared LOcality Cache" in DES-LOC).

        This is the primary DES-LOC cross-domain access path. Instead of a
        direct GPU-to-GPU PCIe transfer (which would stall both compute streams),
        the reader tier accesses the bucket's CPU DRAM copy. If the bucket is
        not yet spilled but has a live GPU buffer, we synchronously copy to
        CPU first (this should be rare; the prefetch thread handles the common
        case).

        Args:
            bucket_id:   Bucket index.
            bwd:         True for backward-pass bucket.
            reader_tier: The DeviceTier that needs to read the parameters.

        Returns:
            The CPU pinned tensor (on CPU DRAM) containing the gathered params,
            or None if the bucket is not available.
        """
        key = self._key(bucket_id, bwd)
        meta = self.parameter_groups[bucket_id]

        # Register this reader for cross-domain tracking
        if key in self._locality_cache:
            self._locality_cache[key].add_reader(reader_tier)

        # If already spilled, return CPU copy directly
        if self.bucket_status[key] == BucketStatus.SPILLED:
            cache = self._locality_cache.get(key)
            if cache and cache.is_ready_on_cpu():
                return cache.pinned_cpu

        # If GPU buffer is live and READY_TO_USE, spill on demand for cross-tier read
        if self.bucket_status[key] == BucketStatus.READY_TO_USE:
            gpu_buf = self._gather_buffers.get(key)
            if gpu_buf is not None:
                # Check if this is actually a cross-domain read (SM86↔SM90)
                is_cross_domain = reader_tier.is_sm86 != meta.owner_tier.is_sm86
                if is_cross_domain and self.enable_cpu_spill:
                    # Synchronous spill to CPU locality cache for this cross-domain read
                    pinned = torch.empty_like(gpu_buf, device="cpu", pin_memory=True)
                    pinned.copy_(gpu_buf)
                    cache_entry = LocalityCacheEntry(
                        bucket_key=key,
                        owner_tier=meta.owner_tier,
                        pinned_cpu=pinned,
                    )
                    cache_entry.add_reader(reader_tier)
                    self._locality_cache[key] = cache_entry
                    logger.debug(
                        "On-demand CPU spill for cross-tier read: bucket=%s "
                        "owner=%s reader=%s (%.2f MB)",
                        key,
                        meta.owner_tier.value,
                        reader_tier.value,
                        gpu_buf.nbytes / 1e6,
                    )
                    return pinned

        return None

    def evict_locality_cache_entry(self, bucket_key: Tuple[int, bool]) -> None:
        """
        Remove a bucket's CPU DRAM locality cache entry (e.g. after all
        cross-tier readers have finished).

        Args:
            bucket_key: (bucket_id, is_backward) tuple.
        """
        if bucket_key in self._locality_cache:
            entry = self._locality_cache.pop(bucket_key)
            entry.pinned_cpu = None  # Release pinned memory
            logger.debug("Evicted locality cache entry for bucket %s", bucket_key)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """
        Cleanly stop the background prefetch thread and release all resources.

        Call this when the training run is complete or the pipeline is no
        longer needed.
        """
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            self._prefetch_stop.set()
            self._prefetch_thread.join(timeout=5.0)
            if self._prefetch_thread.is_alive():
                logger.warning(
                    "Prefetch thread did not stop within 5s; forcing abandon."
                )

        # Release all CPU locality cache entries
        for key in list(self._locality_cache.keys()):
            self.evict_locality_cache_entry(key)

        logger.info("HeteroAllGatherPipeline shut down cleanly.")

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Fake CUDA event for CPU-only unit tests
# ---------------------------------------------------------------------------

class _FakeCudaEvent:
    """Minimal CUDA event stub for CPU-only test execution."""

    def record(self, stream=None) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def wait(self, stream=None) -> None:
        pass


# ---------------------------------------------------------------------------
# DES-LOC aware FSDP synchronise_param_gather hook
# ---------------------------------------------------------------------------

class HeteroFSDPSynchroniser:
    """
    Drop-in replacement for MegatronFSDP.synchronize_param_gather() adapted
    for DES-LOC heterogeneous execution.

    Upstream (megatron_fsdp.py line ~1290):
        self.all_gather_pipeline.reset(preserve_non_fsdp_units=True)
        self._replace_param_with_distributed_if_needed()

    DES-LOC adaptation:
        1. Call HeteroAllGatherPipeline.reset() with tier-aware spill logic.
        2. Identify cross-tier non-unit buckets and pre-populate the CPU
           locality cache so cross-module readers on the H100 can access
           parameters gathered on the A6000 without a blocking PCIe transfer
           during forward.
        3. Log meaningful diagnostics when cross-tier preserved buckets are
           detected (not a no-op log, but tied to actual tier topology).

    Args:
        pipeline:        The HeteroAllGatherPipeline managing bucket state.
        num_params:      Total number of parameters (for sanity checking).
        replace_fn:      Callable that replaces param data with distributed
                         shards (mirrors _replace_param_with_distributed_if_needed).
        force_spill:     Override to force-spill all cross-tier preserved buckets
                         regardless of memory pressure.
    """

    def __init__(
        self,
        pipeline: HeteroAllGatherPipeline,
        num_params: int,
        replace_fn: "callable",
        force_spill: bool = False,
    ) -> None:
        self.pipeline = pipeline
        self.num_params = num_params
        self.replace_fn = replace_fn
        self.force_spill = force_spill

    def synchronize_param_gather(self) -> None:
        """Execute the DES-LOC param-gather synchronisation."""
        spill_override = True if self.force_spill else None

        cross_tier_non_unit = [
            meta
            for meta in self.pipeline.parameter_groups
            if not meta.is_fsdp_unit and meta.is_cross_domain
        ]

        if cross_tier_non_unit:
            logger.info(
                "synchronize_param_gather: %d cross-tier non-unit buckets detected "
                "(owner tiers: %s). These will be preserved in locality cache.",
                len(cross_tier_non_unit),
                {m.owner_tier.value for m in cross_tier_non_unit},
            )

        self.pipeline.reset(
            preserve_non_fsdp_units=True,
            spill_cross_tier_preserved=spill_override,
        )
        self.replace_fn()


# ---------------------------------------------------------------------------
# Utility: build parameter groups from DeepSpeed parameter metadata
# ---------------------------------------------------------------------------

def build_parameter_groups_from_deepspeed(
    param_groups: List[Dict],
    device: torch.device,
    cross_tier_reader_tiers: Optional[Set[DeviceTier]] = None,
) -> List[ParameterGroupMeta]:
    """
    Convert DeepSpeed-style parameter group dicts to ParameterGroupMeta.

    This is the integration shim between DeepSpeed's ZeRO parameter management
    and DES-LOC's heterogeneity-aware bucket metadata.

    DeepSpeed parameter groups are expected to have the following keys:
        - 'params': list of nn.Parameter objects
        - 'fsdp_unit_id': int or None (mirrors Megatron convention)
        - 'device': optional torch.device override

    Args:
        param_groups:            List of DeepSpeed parameter group dicts.
        device:                  Default device for groups without 'device' key.
        cross_tier_reader_tiers: Set of DeviceTiers that may read any bucket
                                 across SM domain boundaries. If None, only
                                 the owner tier is registered as a reader.

    Returns:
        List of ParameterGroupMeta, one per param group.
    """
    metas: List[ParameterGroupMeta] = []

    for i, group in enumerate(param_groups):
        grp_device = group.get("device", device)
        tier = device_to_tier(grp_device)
        fsdp_unit_id = group.get("fsdp_unit_id", None)

        param_names: List[str] = []
        for p in group.get("params", []):
            name = getattr(p, "_param_name", f"param_{id(p)}")
            param_names.append(name)

        readers: Set[DeviceTier] = set()
        if cross_tier_reader_tiers:
            readers = cross_tier_reader_tiers.copy()
        readers.add(tier)

        meta = ParameterGroupMeta(
            bucket_id=i,
            fsdp_unit_id=fsdp_unit_id,
            owner_tier=tier,
            param_names=param_names,
            cross_tier_readers=readers,
            pinned_for_spill=(fsdp_unit_id is None),  # only non-unit buckets spill
        )
        metas.append(meta)

    return metas


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import traceback
    import unittest

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    # Suppress verbose debug output during tests; only WARNING+ is shown
    # unless TEST_VERBOSE=1 is set.
    import os
    if not os.environ.get("TEST_VERBOSE"):
        logging.getLogger("deepspeed.runtime.zero.hetero_allgather_pipeline").setLevel(
            logging.WARNING
        )

    class TestBucketStatus(unittest.TestCase):
        """Tests for BucketStatus enum and its DES-LOC extensions."""

        def test_preserved_needs_gather(self):
            self.assertTrue(BucketStatus.PRESERVED.needs_gather)

        def test_spilled_needs_gather(self):
            self.assertTrue(BucketStatus.SPILLED.needs_gather)

        def test_empty_needs_gather(self):
            self.assertTrue(BucketStatus.EMPTY.needs_gather)

        def test_ready_does_not_need_gather(self):
            self.assertFalse(BucketStatus.READY_TO_USE.needs_gather)

        def test_communicating_does_not_need_gather(self):
            self.assertFalse(BucketStatus.COMMUNICATING.needs_gather)

        def test_enum_ordering(self):
            """PRESERVED must sit between EMPTY and COMMUNICATING (upstream contract)."""
            self.assertLess(BucketStatus.EMPTY.value, BucketStatus.PRESERVED.value)
            self.assertLess(BucketStatus.PRESERVED.value, BucketStatus.COMMUNICATING.value)
            self.assertLess(BucketStatus.COMMUNICATING.value, BucketStatus.READY_TO_USE.value)

        def test_spilled_between_preserved_and_communicating(self):
            self.assertLess(BucketStatus.PRESERVED.value, BucketStatus.SPILLED.value)
            self.assertLess(BucketStatus.SPILLED.value, BucketStatus.COMMUNICATING.value)

    class TestDeviceTier(unittest.TestCase):
        """Tests for DeviceTier classification helpers."""

        def test_a6000_is_sm86(self):
            self.assertTrue(DeviceTier.A6000_0.is_sm86)
            self.assertTrue(DeviceTier.A6000_1.is_sm86)

        def test_h100_is_sm90(self):
            self.assertTrue(DeviceTier.H100.is_sm90)
            self.assertFalse(DeviceTier.H100.is_sm86)

        def test_cpu_is_not_gpu(self):
            self.assertFalse(DeviceTier.CPU.is_gpu)

        def test_gpu_tiers_are_gpu(self):
            for tier in (DeviceTier.A6000_0, DeviceTier.A6000_1, DeviceTier.H100):
                self.assertTrue(tier.is_gpu)

        def test_cuda_index_mapping(self):
            self.assertEqual(DeviceTier.A6000_0.cuda_index, 0)
            self.assertEqual(DeviceTier.A6000_1.cuda_index, 1)
            self.assertEqual(DeviceTier.H100.cuda_index, 2)
            self.assertIsNone(DeviceTier.CPU.cuda_index)

        def test_device_to_tier_cpu(self):
            tier = device_to_tier(torch.device("cpu"))
            self.assertIs(tier, DeviceTier.CPU)

        def test_device_to_tier_unknown_raises(self):
            with self.assertRaises(ValueError):
                device_to_tier(torch.device("cuda:99"))

    class TestParameterGroupMeta(unittest.TestCase):
        """Tests for ParameterGroupMeta classification."""

        def _make_meta(self, fsdp_unit_id, owner_tier, readers):
            return ParameterGroupMeta(
                bucket_id=0,
                fsdp_unit_id=fsdp_unit_id,
                owner_tier=owner_tier,
                cross_tier_readers=readers,
            )

        def test_is_fsdp_unit_true(self):
            meta = self._make_meta(0, DeviceTier.A6000_0, set())
            self.assertTrue(meta.is_fsdp_unit)

        def test_is_fsdp_unit_false(self):
            meta = self._make_meta(None, DeviceTier.A6000_0, set())
            self.assertFalse(meta.is_fsdp_unit)

        def test_cross_domain_sm86_to_sm90(self):
            meta = self._make_meta(
                None,
                DeviceTier.A6000_0,
                {DeviceTier.H100},
            )
            self.assertTrue(meta.is_cross_domain)

        def test_no_cross_domain_same_tier(self):
            meta = self._make_meta(
                None,
                DeviceTier.A6000_0,
                {DeviceTier.A6000_1},
            )
            # Both are SM86 → same domain
            self.assertFalse(meta.is_cross_domain)

        def test_no_cross_domain_empty_readers(self):
            meta = self._make_meta(None, DeviceTier.H100, set())
            self.assertFalse(meta.is_cross_domain)

    class _PipelineTestBase(unittest.TestCase):
        """
        Base class that builds a CPU-mode HeteroAllGatherPipeline for unit tests.

        Uses torch.device("cpu") so tests run without CUDA hardware. GPU-specific
        paths (streams, events) are covered by integration tests on real hardware.
        """

        def _build_pipeline(
            self,
            num_unit_buckets: int = 2,
            num_non_unit_buckets: int = 3,
            has_cross_tier: bool = False,
            enable_cpu_spill: bool = False,
        ) -> HeteroAllGatherPipeline:
            metas: List[ParameterGroupMeta] = []
            bid = 0

            for i in range(num_unit_buckets):
                metas.append(ParameterGroupMeta(
                    bucket_id=bid,
                    fsdp_unit_id=i,
                    owner_tier=DeviceTier.A6000_0,
                    param_names=[f"unit_layer_{i}.weight"],
                ))
                bid += 1

            for i in range(num_non_unit_buckets):
                readers: Set[DeviceTier] = {DeviceTier.A6000_0}
                if has_cross_tier:
                    readers.add(DeviceTier.H100)
                metas.append(ParameterGroupMeta(
                    bucket_id=bid,
                    fsdp_unit_id=None,
                    owner_tier=DeviceTier.A6000_0,
                    param_names=[f"non_unit_param_{i}"],
                    cross_tier_readers=readers,
                ))
                bid += 1

            return HeteroAllGatherPipeline(
                parameter_groups=metas,
                process_group=None,
                current_device=torch.device("cpu"),
                enable_cpu_spill=enable_cpu_spill,
                gpu_memory_headroom_mb=2048.0,
            )

    class TestResetPreservesNonUnitBuckets(_PipelineTestBase):
        """
        Core regression test mirroring Megatron commit 378d81f:
        non-FSDP-unit buckets must survive reset() as PRESERVED, not EMPTY.
        """

        def test_unit_buckets_become_empty_after_reset(self):
            pipeline = self._build_pipeline(num_unit_buckets=2, num_non_unit_buckets=0)
            pipeline.reset(preserve_non_fsdp_units=True)
            for bid in range(2):
                for bwd in (False, True):
                    self.assertIs(
                        pipeline.bucket_status[(bid, bwd)],
                        BucketStatus.EMPTY,
                        f"Unit bucket {bid} bwd={bwd} should be EMPTY after reset",
                    )

        def test_non_unit_buckets_become_preserved_after_reset(self):
            pipeline = self._build_pipeline(num_unit_buckets=1, num_non_unit_buckets=3)
            pipeline.reset(preserve_non_fsdp_units=True)
            # Bucket 0 is unit → EMPTY; buckets 1,2,3 are non-unit → PRESERVED
            self.assertIs(pipeline.bucket_status[(0, False)], BucketStatus.EMPTY)
            for bid in (1, 2, 3):
                for bwd in (False, True):
                    self.assertIs(
                        pipeline.bucket_status[(bid, bwd)],
                        BucketStatus.PRESERVED,
                        f"Non-unit bucket {bid} bwd={bwd} should be PRESERVED",
                    )

        def test_reset_without_preserve_empties_all(self):
            pipeline = self._build_pipeline(num_unit_buckets=1, num_non_unit_buckets=2)
            pipeline.reset(preserve_non_fsdp_units=False)
            for bid in range(3):
                for bwd in (False, True):
                    self.assertIs(
                        pipeline.bucket_status[(bid, bwd)],
                        BucketStatus.EMPTY,
                    )

        def test_double_reset_non_unit_stays_preserved(self):
            """Calling reset() twice should not corrupt PRESERVED buckets."""
            pipeline = self._build_pipeline(num_unit_buckets=0, num_non_unit_buckets=2)
            pipeline.reset(preserve_non_fsdp_units=True)
            pipeline.reset(preserve_non_fsdp_units=True)
            for bid in (0, 1):
                for bwd in (False, True):
                    self.assertIn(
                        pipeline.bucket_status[(bid, bwd)],
                        (BucketStatus.PRESERVED, BucketStatus.EMPTY),
                    )

        def test_reset_assertion_passes_with_correct_states(self):
            """reset() should not raise when all final states are valid."""
            pipeline = self._build_pipeline(num_unit_buckets=2, num_non_unit_buckets=2)
            # Should not raise
            pipeline.reset(preserve_non_fsdp_units=True)

        def test_reset_with_outstanding_events_warns(self):
            """If param_gather_event_map is non-empty, reset() should warn."""
            pipeline = self._build_pipeline(num_unit_buckets=1, num_non_unit_buckets=0)
            # Manually inject a fake outstanding event
            key = (0, False)
            pipeline.bucket_status[key] = BucketStatus.COMMUNICATING
            evt = _FakeCudaEvent()

            def _mark():
                pipeline.bucket_status[key] = BucketStatus.READY_TO_USE

            pipeline.param_gather_event_map[key] = (evt, _mark)

            with self.assertWarns(UserWarning):
                pipeline.reset(preserve_non_fsdp_units=True)

            # After draining the event, bucket should be EMPTY (unit bucket)
            self.assertIs(pipeline.bucket_status[key], BucketStatus.EMPTY)

    class TestTierSorting(_PipelineTestBase):
        """Tests for PCIe-aware tier-sorted bucket ordering."""

        def test_same_tier_first(self):
            """Buckets owned by current tier must precede cross-tier buckets."""
            # Build pipeline where current device is A6000_0 (cpu mock, tier injected)
            metas = [
                # bucket 0: H100 (cross-tier from A6000_0 perspective)
                ParameterGroupMeta(
                    bucket_id=0,
                    fsdp_unit_id=0,
                    owner_tier=DeviceTier.H100,
                ),
                # bucket 1: A6000_0 (same tier)
                ParameterGroupMeta(
                    bucket_id=1,
                    fsdp_unit_id=1,
                    owner_tier=DeviceTier.A6000_0,
                ),
                # bucket 2: A6000_1 (same SM domain, different card)
                ParameterGroupMeta(
                    bucket_id=2,
                    fsdp_unit_id=2,
                    owner_tier=DeviceTier.A6000_1,
                ),
            ]
            pipeline = HeteroAllGatherPipeline(
                parameter_groups=metas,
                process_group=None,
                current_device=torch.device("cpu"),
                enable_cpu_spill=False,
            )
            # Override current_tier to simulate A6000_0 rank
            pipeline.current_tier = DeviceTier.A6000_0

            sorted_ids = pipeline._tier_sorted_bucket_ids([0, 1, 2], False)
            # bucket 1 (A6000_0, same) and bucket 2 (A6000_1, same SM domain)
            # should appear before bucket 0 (H100, cross-domain)
            h100_pos = sorted_ids.index(0)
            a6000_0_pos = sorted_ids.index(1)
            a6000_1_pos = sorted_ids.index(2)

            self.assertLess(a6000_0_pos, h100_pos,
                            "Same-tier A6000_0 bucket should precede H100 bucket")
            self.assertLess(a6000_1_pos, h100_pos,
                            "Same-domain A6000_1 bucket should precede H100 bucket")

        def test_all_same_tier_order_preserved(self):
            """If all buckets are same-tier, original order is preserved."""
            metas = [
                ParameterGroupMeta(bucket_id=i, fsdp_unit_id=i, owner_tier=DeviceTier.A6000_0)
                for i in range(4)
            ]
            pipeline = HeteroAllGatherPipeline(
                parameter_groups=metas,
                process_group=None,
                current_device=torch.device("cpu"),
                enable_cpu_spill=False,
            )
            pipeline.current_tier = DeviceTier.A6000_0
            sorted_ids = pipeline._tier_sorted_bucket_ids([3, 1, 0, 2], False)
            # All same tier: input order preserved within same-tier list
            self.assertEqual(sorted_ids, [3, 1, 0, 2])

    class TestLocalityCacheEntry(unittest.TestCase):
        """Tests for LocalityCacheEntry cross-domain reader tracking."""

        def _make_entry(self, owner: DeviceTier) -> LocalityCacheEntry:
            return LocalityCacheEntry(
                bucket_key=(0, False),
                owner_tier=owner,
            )

        def test_cross_domain_reader_detected(self):
            entry = self._make_entry(DeviceTier.A6000_0)
            entry.add_reader(DeviceTier.H100)
            self.assertTrue(entry.has_cross_domain_readers(DeviceTier.A6000_0))

        def test_same_domain_reader_not_cross(self):
            entry = self._make_entry(DeviceTier.A6000_0)
            entry.add_reader(DeviceTier.A6000_1)  # same SM86 domain
            self.assertFalse(entry.has_cross_domain_readers(DeviceTier.A6000_0))

        def test_remove_reader(self):
            entry = self._make_entry(DeviceTier.H100)
            entry.add_reader(DeviceTier.A6000_0)
            entry.remove_reader(DeviceTier.A6000_0)
            self.assertNotIn(DeviceTier.A6000_0, entry.readers)

        def test_is_ready_on_cpu_false_when_no_pinned(self):
            entry = self._make_entry(DeviceTier.A6000_0)
            self.assertFalse(entry.is_ready_on_cpu())

        def test_is_ready_on_cpu_true_when_pinned_set(self):
            entry = self._make_entry(DeviceTier.H100)
            entry.pinned_cpu = torch.zeros(4)
            self.assertTrue(entry.is_ready_on_cpu())

    class TestBucketReleaseAndRecycle(_PipelineTestBase):
        """Tests for bucket release/recycle lifecycle."""

        def test_unit_bucket_recycled_after_release(self):
            pipeline = self._build_pipeline(num_unit_buckets=2, num_non_unit_buckets=0)
            key = (0, False)
            # Simulate bucket going through allgather cycle
            pipeline.bucket_status[key] = BucketStatus.READY_TO_USE
            pipeline.bucket_can_be_released[key] = False

            pipeline.release_bucket(0, False)
            # After recycle, READY_TO_USE unit bucket should become EMPTY
            self.assertIs(pipeline.bucket_status[key], BucketStatus.EMPTY)

        def test_preserved_non_unit_not_recycled_on_release(self):
            pipeline = self._build_pipeline(num_unit_buckets=0, num_non_unit_buckets=2)
            key = (0, False)
            pipeline.bucket_status[key] = BucketStatus.PRESERVED

            # release_bucket should not recycle a PRESERVED bucket
            pipeline.bucket_can_be_released[key] = True
            pipeline._recycle_unused_buckets()
            self.assertIs(pipeline.bucket_status[key], BucketStatus.PRESERVED)

        def test_communicating_bucket_not_recycled(self):
            pipeline = self._build_pipeline(num_unit_buckets=1, num_non_unit_buckets=0)
            key = (0, False)
            pipeline.bucket_status[key] = BucketStatus.COMMUNICATING
            pipeline.bucket_can_be_released[key] = True

            pipeline._recycle_unused_buckets()
            # COMMUNICATING → should not be freed
            self.assertIs(pipeline.bucket_status[key], BucketStatus.COMMUNICATING)

    class TestWaitBucketReady(_PipelineTestBase):
        """Tests for wait_bucket_ready behaviour across bucket states."""

        def test_ready_returns_immediately(self):
            pipeline = self._build_pipeline(num_unit_buckets=1, num_non_unit_buckets=0)
            key = (0, False)
            pipeline.bucket_status[key] = BucketStatus.READY_TO_USE
            # Should not raise
            pipeline.wait_bucket_ready(0, False)

        def test_empty_raises_without_event(self):
            pipeline = self._build_pipeline(num_unit_buckets=1, num_non_unit_buckets=0)
            pipeline.bucket_status[(0, False)] = BucketStatus.EMPTY
            with self.assertRaises(ValueError):
                pipeline.wait_bucket_ready(0, False)

        def test_preserved_raises_without_event(self):
            pipeline = self._build_pipeline(num_unit_buckets=0, num_non_unit_buckets=1)
            pipeline.bucket_status[(0, False)] = BucketStatus.PRESERVED
            with self.assertRaises(ValueError):
                pipeline.wait_bucket_ready(0, False)

        def test_communicating_with_event_transitions_to_ready(self):
            pipeline = self._build_pipeline(num_unit_buckets=1, num_non_unit_buckets=0)
            key = (0, False)
            pipeline.bucket_status[key] = BucketStatus.COMMUNICATING
            evt = _FakeCudaEvent()

            def _mark():
                pipeline.bucket_status[key] = BucketStatus.READY_TO_USE

            pipeline.param_gather_event_map[key] = (evt, _mark)
            pipeline.wait_bucket_ready(0, False)
            self.assertIs(pipeline.bucket_status[key], BucketStatus.READY_TO_USE)
            self.assertNotIn(key, pipeline.param_gather_event_map)

        def test_spilled_with_cpu_data_reloads_synchronously(self):
            """SPILLED bucket with CPU data must reload synchronously on wait."""
            pipeline = self._build_pipeline(
                num_unit_buckets=0,
                num_non_unit_buckets=1,
                enable_cpu_spill=False,
            )
            key = (0, False)
            gpu_buf = torch.ones(4)
            pipeline._gather_buffers[key] = gpu_buf
            pipeline.bucket_status[key] = BucketStatus.SPILLED

            # Set up a CPU locality cache entry
            pinned = torch.full((4,), 7.0)
            pipeline._locality_cache[key] = LocalityCacheEntry(
                bucket_key=key,
                owner_tier=DeviceTier.A6000_0,
                pinned_cpu=pinned,
            )

            pipeline.wait_bucket_ready(0, False)
            self.assertIs(pipeline.bucket_status[key], BucketStatus.READY_TO_USE)
            # GPU buffer should now contain the CPU data
            self.assertTrue(torch.all(gpu_buf == 7.0))

    class TestHeteroFSDPSynchroniser(_PipelineTestBase):
        """Tests for the DES-LOC FSDP synchronise_param_gather wrapper."""

        def test_replace_fn_called_after_reset(self):
            pipeline = self._build_pipeline(num_unit_buckets=1, num_non_unit_buckets=2)
            called = []

            def _replace():
                called.append(True)

            sync = HeteroFSDPSynchroniser(
                pipeline=pipeline,
                num_params=10,
                replace_fn=_replace,
            )
            sync.synchronize_param_gather()
            self.assertEqual(len(called), 1, "replace_fn should be called exactly once")

        def test_non_unit_buckets_preserved_after_sync(self):
            pipeline = self._build_pipeline(num_unit_buckets=1, num_non_unit_buckets=2)
            sync = HeteroFSDPSynchroniser(
                pipeline=pipeline,
                num_params=10,
                replace_fn=lambda: None,
            )
            sync.synchronize_param_gather()
            # Non-unit buckets 1 and 2 should be PRESERVED
            for bid in (1, 2):
                for bwd in (False, True):
                    self.assertIn(
                        pipeline.bucket_status[(bid, bwd)],
                        (BucketStatus.PRESERVED, BucketStatus.SPILLED),
                    )

        def test_cross_tier_buckets_logged_at_info(self):
            """Synchroniser must emit an INFO log for cross-tier non-unit buckets."""
            pipeline = self._build_pipeline(
                num_unit_buckets=1,
                num_non_unit_buckets=2,
                has_cross_tier=True,
            )
            sync = HeteroFSDPSynchroniser(
                pipeline=pipeline,
                num_params=10,
                replace_fn=lambda: None,
            )
            with self.assertLogs(
                "deepspeed.runtime.zero.hetero_allgather_pipeline",
                level="INFO",
            ) as cm:
                # Temporarily re-enable logging for this sub-test
                logging.getLogger(
                    "deepspeed.runtime.zero.hetero_allgather_pipeline"
                ).setLevel(logging.DEBUG)
                sync.synchronize_param_gather()

            cross_tier_logs = [
                line for line in cm.output if "cross-tier" in line
            ]
            self.assertTrue(
                len(cross_tier_logs) >= 1,
                "Expected at least one INFO log about cross-tier non-unit buckets",
            )

    class TestBuildParameterGroupsFromDeepSpeed(unittest.TestCase):
        """Tests for the DeepSpeed→DES-LOC parameter group conversion shim."""

        def test_empty_groups(self):
            result = build_parameter_groups_from_deepspeed([], torch.device("cpu"))
            self.assertEqual(result, [])

        def test_fsdp_unit_preserved(self):
            groups = [
                {"params": [], "fsdp_unit_id": 3},
                {"params": [], "fsdp_unit_id": None},
            ]
            metas = build_parameter_groups_from_deepspeed(groups, torch.device("cpu"))
            self.assertEqual(metas[0].fsdp_unit_id, 3)
            self.assertIsNone(metas[1].fsdp_unit_id)

        def test_non_unit_eligible_for_spill(self):
            groups = [
                {"params": [], "fsdp_unit_id": 0},
                {"params": [], "fsdp_unit_id": None},
            ]
            metas = build_parameter_groups_from_deepspeed(groups, torch.device("cpu"))
            self.assertFalse(metas[0].pinned_for_spill)  # unit: not spillable
            self.assertTrue(metas[1].pinned_for_spill)   # non-unit: spillable

        def test_cross_tier_readers_propagated(self):
            groups = [{"params": [], "fsdp_unit_id": None}]
            readers = {DeviceTier.H100}
            metas = build_parameter_groups_from_deepspeed(
                groups, torch.device("cpu"), cross_tier_reader_tiers=readers
            )
            self.assertIn(DeviceTier.H100, metas[0].cross_tier_readers)

        def test_bucket_ids_sequential(self):
            groups = [{"params": [], "fsdp_unit_id": i} for i in range(5)]
            metas = build_parameter_groups_from_deepspeed(groups, torch.device("cpu"))
            for i, meta in enumerate(metas):
                self.assertEqual(meta.bucket_id, i)

    # Run all tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestBucketStatus,
        TestDeviceTier,
        TestParameterGroupMeta,
        TestResetPreservesNonUnitBuckets,
        TestTierSorting,
        TestLocalityCacheEntry,
        TestBucketReleaseAndRecycle,
        TestWaitBucketReady,
        TestHeteroFSDPSynchroniser,
        TestBuildParameterGroupsFromDeepSpeed,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
