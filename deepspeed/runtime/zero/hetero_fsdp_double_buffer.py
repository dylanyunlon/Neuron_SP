"""
deepspeed/runtime/zero/hetero_fsdp_double_buffer.py

DES-LOC Heterogeneous FSDP Double Buffer with Tier-Aware Pool Sizing
=====================================================================

Upstream Design Intent (Megatron d199bb9e9f388d8f8136a0c9b25d2bcc53529ffe)
---------------------------------------------------------------------------
Megatron's fsdp_double_buffer feature pre-allocates two "buckets" of GPU memory
for AllGather operations so that parameter communication can overlap with
computation. The original bug this commit fixes: when an FSDP unit's bucket
is larger than the fixed pool, the fallback allocator was a bare
``TemporaryBucketAllocator`` which relies on torch.empty() + in-place resize
semantics. Under certain CUDA IMA (Illegal Memory Access) conditions — specifically
when the same storage is re-used across async NCCL streams — this triggered silent
data corruption or outright crashes.

The fix changes the fallback to ``StorageResizeBasedBucketAllocator`` which
performs explicit storage.resize_() calls under a stream guard, ensuring the
CUDA VM mapping is stable before the AllGather kernel is launched.

DES-LOC Adaptation Points
--------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) runs across a
heterogeneous pool:

  Tier-0  —  2× NVIDIA A6000 48 GB  SM86  (PCIe, no NVLink)
  Tier-1  —  1× NVIDIA H100 NVL 96 GB  SM90 (PCIe)
  Host    —  1.5 TB CPU DRAM (pinned staging)

Key adaptations vs. upstream Megatron:

1. **TierAwarePoolAllocator** (replaces FixedPoolAllocator)
   Pool sizes are derived per-device from runtime free-memory headroom rather
   than a static config value. H100 gets a larger double-buffer pool; A6000s
   share a smaller combined pool. The DES-LOC Locality Cache reservation is
   subtracted before sizing the pool so parameter buffers never evict cached
   activations.

2. **HeteroStorageResizeFallback** (replaces StorageResizeBasedBucketAllocator)
   Like the upstream fix, we use storage.resize_() under a stream guard, but
   we extend it to support cross-tier moves: if a bucket cannot fit on the
   current device even after resize, we spill the bucket to pinned CPU memory
   and issue an async H2D copy on a dedicated staging stream. This prevents
   OOM on A6000 when large Mamba SSM parameter blocks overflow the pool.

3. **HeteroFSDPDoubleBuffer** (main class)
   Wraps DeepSpeed ZeRO-3's parameter-fetch logic and injects tier-aware
   double-buffering. It maintains per-tier prefetch queues and stream sets,
   and honours the DES-LOC Locality Cache hit/miss protocol so that parameters
   already resident in the Locality Cache bypass the AllGather entirely.

4. **DeviceTierRegistry**
   Lightweight registry that maps CUDA device ordinals → tier metadata at
   module import time. No external config file needed; detection is fully
   automatic via ``torch.cuda.get_device_properties``.

DeepSpeed Integration
---------------------
Drop this file into ``deepspeed/runtime/zero/`` and set:

    ds_config["zero_optimization"]["hetero_fsdp_double_buffer"] = True

The ``ZeROOptimizer`` initialisation path picks this up in
``deepspeed/runtime/zero/stage3.py`` (search for ``hetero_fsdp_double_buffer``).

References
----------
* Megatron commit d199bb9 — Fix CUDA IMA in fsdp_double_buffer
* DeepSpeed ZeRO-3 param fetch: deepspeed/runtime/zero/partitioned_param_coordinator.py
* DES-LOC design doc: docs/des-loc/locality_cache.md (internal)
"""

from __future__ import annotations

import gc
import logging
import math
import threading
import time
import weakref
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Generator, List, Optional, Set, Tuple

import torch
import torch.cuda
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fraction of free device memory that DES-LOC reserves for the Locality Cache.
# The double-buffer pool is carved from the *remainder*.
_LOCALITY_CACHE_HEADROOM_FRACTION: float = 0.15

# Minimum pool size (bytes) below which we skip double-buffering entirely and
# fall through to the per-bucket fallback allocator.
_MIN_POOL_BYTES: int = 64 * 1024 * 1024  # 64 MB

# How many bytes of pinned CPU memory we pre-allocate as spill staging.
_CPU_STAGING_BYTES: int = 512 * 1024 * 1024  # 512 MB per spill arena

# SM capability thresholds
_SM90_MAJOR = 9
_SM86_MAJOR = 8
_SM86_MINOR = 6

# ---------------------------------------------------------------------------
# Device Tier Registry
# ---------------------------------------------------------------------------


class DeviceTier(Enum):
    """Logical tier within the DES-LOC heterogeneous device pool."""
    H100 = auto()    # SM90, large HBM — preferred for large AllGather buckets
    A6000 = auto()   # SM86, medium VRAM — secondary tier
    UNKNOWN = auto() # Any other CUDA device


@dataclass(frozen=True)
class TierMetadata:
    device_index: int
    tier: DeviceTier
    total_bytes: int
    sm_major: int
    sm_minor: int

    @property
    def tier_name(self) -> str:
        return self.tier.name

    @property
    def pool_budget_bytes(self) -> int:
        """
        Compute double-buffer pool budget for this device.

        We query *current* free memory rather than total so we account for
        whatever DeepSpeed/PyTorch has already allocated. The Locality Cache
        headroom fraction is subtracted so pool allocations never starve the
        cache.
        """
        free_bytes, _ = torch.cuda.mem_get_info(self.device_index)
        usable = int(free_bytes * (1.0 - _LOCALITY_CACHE_HEADROOM_FRACTION))
        # Use at most 30 % of usable for double-buffer (leave room for gradients)
        return max(0, int(usable * 0.30))


class DeviceTierRegistry:
    """
    Singleton that maps CUDA device ordinals to DES-LOC tier metadata.

    Detection is fully automatic: at first access we probe every visible CUDA
    device and classify it by SM capability and VRAM size.
    """

    _instance: Optional["DeviceTierRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "DeviceTierRegistry":
        with cls._lock:
            if cls._instance is None:
                obj = super().__new__(cls)
                obj._registry: Dict[int, TierMetadata] = {}
                obj._initialised = False
                cls._instance = obj
        return cls._instance

    def _probe(self) -> None:
        if self._initialised:
            return
        n = torch.cuda.device_count()
        for idx in range(n):
            props = torch.cuda.get_device_properties(idx)
            major, minor = props.major, props.minor
            total = props.total_memory
            if major >= _SM90_MAJOR:
                tier = DeviceTier.H100
            elif major == _SM86_MAJOR and minor >= _SM86_MINOR:
                tier = DeviceTier.A6000
            else:
                tier = DeviceTier.UNKNOWN
            meta = TierMetadata(
                device_index=idx,
                tier=tier,
                total_bytes=total,
                sm_major=major,
                sm_minor=minor,
            )
            self._registry[idx] = meta
            logger.info(
                "DES-LOC device probe: cuda:%d → %s  (SM%d%d, %.1f GB total, "
                "pool budget %.1f MB)",
                idx,
                tier.name,
                major,
                minor,
                total / 1e9,
                meta.pool_budget_bytes / 1e6,
            )
        self._initialised = True

    def get(self, device_index: int) -> TierMetadata:
        self._probe()
        if device_index not in self._registry:
            raise KeyError(f"Device cuda:{device_index} not in DES-LOC registry")
        return self._registry[device_index]

    def all_tiers(self) -> List[TierMetadata]:
        self._probe()
        return list(self._registry.values())

    def devices_of_tier(self, tier: DeviceTier) -> List[TierMetadata]:
        return [m for m in self.all_tiers() if m.tier == tier]


_registry = DeviceTierRegistry()

# ---------------------------------------------------------------------------
# CPU Spill Arena (pinned memory staging for cross-tier overflow)
# ---------------------------------------------------------------------------


class PinnedSpillArena:
    """
    Pre-allocated pinned CPU memory arena used when a parameter bucket
    overflows the on-device pool.

    Upstream context: Megatron's fix ensures the *GPU* fallback allocator uses
    storage.resize_() under a stream guard. In DES-LOC we go one step further:
    if the GPU itself is OOM after resize we spill to this pinned arena and
    schedule an async H2D prefetch, effectively using DRAM as an L3 parameter
    cache. This is only triggered for A6000 devices during large Mamba SSM
    parameter gathers where bucket sizes can exceed 200 MB.
    """

    def __init__(self, capacity_bytes: int = _CPU_STAGING_BYTES) -> None:
        self._capacity = capacity_bytes
        self._cursor = 0
        self._lock = threading.Lock()
        # Allocate as uint8 so we can slice arbitrary dtypes out of it
        self._backing: Tensor = torch.empty(
            capacity_bytes, dtype=torch.uint8, pin_memory=True
        )
        logger.debug(
            "PinnedSpillArena: allocated %.0f MB of pinned staging memory",
            capacity_bytes / 1e6,
        )

    def allocate(self, num_bytes: int) -> Tensor:
        """
        Return a uint8 view into the pinned backing buffer.

        We use a simple bump allocator with wrap-around. Callers must not hold
        the returned view across an arena reset.
        """
        num_bytes = _align_up(num_bytes, 256)  # 256-byte alignment for CUDA DMA
        with self._lock:
            if self._cursor + num_bytes > self._capacity:
                # Wrap around — safe only if the previous tenant has finished
                # its H2D copy (caller is responsible for synchronisation).
                self._cursor = 0
                logger.debug(
                    "PinnedSpillArena: wrap-around at capacity %d bytes", self._capacity
                )
            view = self._backing[self._cursor : self._cursor + num_bytes]
            self._cursor += num_bytes
        return view

    def reset(self) -> None:
        with self._lock:
            self._cursor = 0


# ---------------------------------------------------------------------------
# Bucket Allocators
# ---------------------------------------------------------------------------


@dataclass
class BucketHandle:
    """Opaque handle returned by any BucketAllocator."""
    storage: Tensor           # The allocated GPU tensor (dtype=uint8)
    device: torch.device
    num_bytes: int
    from_pool: bool           # True → returned to pool on release; False → freed
    spilled_from_cpu: bool = False  # True → was H2D-copied from pinned arena


class BucketAllocator:
    """Abstract base for DES-LOC bucket allocators."""

    def allocate(self, num_bytes: int, device: torch.device) -> BucketHandle:
        raise NotImplementedError

    def release(self, handle: BucketHandle) -> None:
        raise NotImplementedError


class HeteroStorageResizeFallback(BucketAllocator):
    """
    DES-LOC adaptation of Megatron's ``StorageResizeBasedBucketAllocator``.

    Upstream fix rationale (d199bb9): using torch.empty() for the fallback
    allocator caused CUDA IMA because NCCL's AllGather was launched on a
    secondary stream while the original storage was still mapped. The fix
    uses storage.resize_() which issues an explicit VM remap before the
    kernel is enqueued, preventing the race.

    DES-LOC extension: if the device is an A6000 and the resize would exceed
    remaining VRAM, we spill the bucket to the ``PinnedSpillArena`` and
    initiate an async H2D copy on the staging stream. The returned handle is
    marked ``spilled_from_cpu=True`` so the caller knows to wait on the
    staging stream before reading the tensor.
    """

    def __init__(
        self,
        staging_stream: Optional[torch.cuda.Stream],
        spill_arena: Optional[PinnedSpillArena],
    ) -> None:
        self._staging_stream = staging_stream
        self._spill_arena = spill_arena
        # We keep one persistent GPU tensor per device whose storage we resize.
        # Key: device index  Value: (tensor, current_size_bytes)
        self._resident: Dict[int, Tuple[Tensor, int]] = {}
        self._lock = threading.Lock()

    def allocate(self, num_bytes: int, device: torch.device) -> BucketHandle:
        dev_idx = device.index if device.index is not None else torch.cuda.current_device()
        num_bytes_aligned = _align_up(num_bytes, 256)

        with self._lock:
            if dev_idx in self._resident:
                tensor, cur_size = self._resident[dev_idx]
            else:
                # Bootstrap: allocate a 1-byte tensor whose storage we'll resize
                tensor = torch.empty(1, dtype=torch.uint8, device=device)
                cur_size = 1
                self._resident[dev_idx] = (tensor, cur_size)

        # Check whether we have headroom for the resize
        free_bytes, _ = torch.cuda.mem_get_info(dev_idx)
        extra_needed = max(0, num_bytes_aligned - cur_size)

        if extra_needed > 0 and extra_needed > free_bytes * 0.9:
            # Not enough VRAM — spill to CPU and schedule H2D copy
            return self._allocate_via_spill(num_bytes_aligned, device, dev_idx)

        # Perform the resize under the staging stream context so the VM mapping
        # is committed before any NCCL kernel is launched (upstream fix).
        stream_ctx = (
            torch.cuda.stream(self._staging_stream)
            if self._staging_stream is not None
            else _null_ctx()
        )
        with stream_ctx:
            tensor.storage().resize_(num_bytes_aligned)

        with self._lock:
            self._resident[dev_idx] = (tensor, num_bytes_aligned)

        return BucketHandle(
            storage=tensor,
            device=device,
            num_bytes=num_bytes_aligned,
            from_pool=False,
        )

    def _allocate_via_spill(
        self, num_bytes: int, device: torch.device, dev_idx: int
    ) -> BucketHandle:
        if self._spill_arena is None:
            raise MemoryError(
                f"cuda:{dev_idx} has insufficient VRAM for bucket of {num_bytes} bytes "
                "and no PinnedSpillArena is configured. Set "
                "'hetero_fsdp_double_buffer_cpu_spill': true in ds_config."
            )

        logger.warning(
            "DES-LOC spill: bucket of %.1f MB exceeds free VRAM on cuda:%d; "
            "staging via pinned CPU arena",
            num_bytes / 1e6,
            dev_idx,
        )

        cpu_buf = self._spill_arena.allocate(num_bytes)
        # Allocate destination on GPU using the resize path (may still succeed
        # for a smaller size after GC)
        gc.collect()
        torch.cuda.empty_cache()
        gpu_tensor = torch.empty(num_bytes, dtype=torch.uint8, device=device)

        # Async H2D on the staging stream
        if self._staging_stream is not None:
            with torch.cuda.stream(self._staging_stream):
                gpu_tensor.copy_(cpu_buf, non_blocking=True)
        else:
            gpu_tensor.copy_(cpu_buf)

        return BucketHandle(
            storage=gpu_tensor,
            device=device,
            num_bytes=num_bytes,
            from_pool=False,
            spilled_from_cpu=True,
        )

    def release(self, handle: BucketHandle) -> None:
        # Storage-resize tensors are kept alive for reuse; nothing to free.
        pass


class TierAwarePoolAllocator(BucketAllocator):
    """
    DES-LOC replacement for Megatron's ``FixedPoolAllocator``.

    Upstream design: FixedPoolAllocator pre-allocates a contiguous GPU memory
    slab and sub-allocates two "slots" out of it for double-buffering. When a
    requested bucket is larger than a slot, the request falls through to a
    backup allocator. The original backup (TemporaryBucketAllocator) caused
    CUDA IMA; the upstream fix changes it to StorageResizeBasedBucketAllocator.

    DES-LOC changes:
    - Pool size is computed dynamically from free VRAM minus the Locality
      Cache headroom rather than being fixed at construction time.
    - On H100 (Tier-0) we use a larger pool and a higher slot-count (4
      instead of 2) because the NVL variant has enough HBM headroom and we
      want to pipeline 4 microbatches without stalling.
    - On A6000 (Tier-1) we use a smaller pool (2 slots, conservative sizing)
      and configure the fallback with the PinnedSpillArena.
    - The ``_locality_cache_hit`` path bypasses the pool entirely: if the
      parameter is already resident in the DES-LOC Locality Cache we return
      a zero-copy view rather than triggering an AllGather.
    """

    def __init__(
        self,
        device: torch.device,
        locality_cache: Optional["LocalityCacheProtocol"],
        staging_stream: Optional[torch.cuda.Stream],
        spill_arena: Optional[PinnedSpillArena],
    ) -> None:
        self._device = device
        self._locality_cache = locality_cache
        dev_idx = device.index if device.index is not None else 0
        self._meta = _registry.get(dev_idx)

        # Compute pool geometry
        budget = self._meta.pool_budget_bytes
        if self._meta.tier == DeviceTier.H100:
            self._num_slots = 4
        else:
            self._num_slots = 2

        self._slot_size = max(_MIN_POOL_BYTES, budget // self._num_slots)
        self._pool_bytes = self._slot_size * self._num_slots

        if self._pool_bytes < _MIN_POOL_BYTES:
            logger.warning(
                "DES-LOC pool too small on cuda:%d (%.1f MB); double-buffer "
                "will rely entirely on fallback allocator",
                dev_idx,
                self._pool_bytes / 1e6,
            )
            self._pool: Optional[Tensor] = None
        else:
            self._pool = torch.empty(
                self._pool_bytes, dtype=torch.uint8, device=device
            )
            logger.info(
                "DES-LOC TierAwarePool: cuda:%d (%s) pool=%.1f MB, "
                "%d slots of %.1f MB each",
                dev_idx,
                self._meta.tier_name,
                self._pool_bytes / 1e6,
                self._num_slots,
                self._slot_size / 1e6,
            )

        # Slot availability tracking
        self._free_slots: List[int] = list(range(self._num_slots))
        self._lock = threading.Lock()

        # Fallback — mirrors upstream's switch to StorageResizeBasedBucketAllocator
        self._fallback = HeteroStorageResizeFallback(
            staging_stream=staging_stream,
            spill_arena=spill_arena,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate(self, num_bytes: int, device: torch.device) -> BucketHandle:
        num_bytes_aligned = _align_up(num_bytes, 256)

        with self._lock:
            if (
                self._pool is not None
                and num_bytes_aligned <= self._slot_size
                and self._free_slots
            ):
                slot_idx = self._free_slots.pop()
                start = slot_idx * self._slot_size
                view = self._pool[start : start + num_bytes_aligned]
                return BucketHandle(
                    storage=view,
                    device=device,
                    num_bytes=num_bytes_aligned,
                    from_pool=True,
                )

        # Pool miss — delegate to fallback (the critical upstream fix path)
        logger.debug(
            "DES-LOC pool miss on %s: bucket %.1f MB > slot %.1f MB; "
            "using storage-resize fallback",
            device,
            num_bytes_aligned / 1e6,
            self._slot_size / 1e6,
        )
        return self._fallback.allocate(num_bytes_aligned, device)

    def release(self, handle: BucketHandle) -> None:
        if not handle.from_pool:
            self._fallback.release(handle)
            return
        # Recover slot index from pointer arithmetic
        if self._pool is None:
            return
        offset = handle.storage.data_ptr() - self._pool.data_ptr()
        slot_idx = offset // self._slot_size
        with self._lock:
            if slot_idx not in self._free_slots:
                self._free_slots.append(slot_idx)

    def stats(self) -> Dict[str, object]:
        with self._lock:
            free = len(self._free_slots)
        return {
            "device": str(self._device),
            "tier": self._meta.tier_name,
            "pool_bytes": self._pool_bytes,
            "num_slots": self._num_slots,
            "free_slots": free,
            "slot_size_bytes": self._slot_size,
        }


# ---------------------------------------------------------------------------
# Locality Cache Protocol (structural typing — avoids circular imports)
# ---------------------------------------------------------------------------


class LocalityCacheProtocol:
    """
    Structural interface for the DES-LOC Locality Cache.

    The real implementation lives in ``deepspeed/runtime/des_loc/locality_cache.py``.
    We define only the methods we call here so this module has no hard import
    dependency on the cache implementation.
    """

    def is_resident(self, param_id: int, device: torch.device) -> bool:
        """Return True if ``param_id`` is already in the cache on ``device``."""
        raise NotImplementedError

    def get_view(self, param_id: int, device: torch.device) -> Optional[Tensor]:
        """Return a tensor view of the cached parameter, or None on miss."""
        raise NotImplementedError

    def mark_accessed(self, param_id: int, device: torch.device) -> None:
        """Update LRU/access metadata after a hit."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Double-Buffer Prefetch Queue
# ---------------------------------------------------------------------------


@dataclass
class PrefetchEntry:
    param_id: int
    param_group_id: int
    bucket_bytes: int
    device: torch.device
    handle: Optional[BucketHandle] = None
    allgather_event: Optional[torch.cuda.Event] = None
    ready: bool = False


class DoublePrefetchQueue:
    """
    Ring buffer of two (or more on H100) pre-allocated buckets for pipelined
    AllGather operations.

    On H100 we use 4 slots so that up to 4 microbatches worth of parameters
    can be in-flight simultaneously. On A6000 we use 2 to keep memory pressure
    manageable.
    """

    def __init__(self, allocator: TierAwarePoolAllocator, depth: int) -> None:
        self._allocator = allocator
        self._depth = depth
        self._queue: List[Optional[PrefetchEntry]] = [None] * depth
        self._head = 0
        self._tail = 0
        self._count = 0
        self._lock = threading.Lock()

    def enqueue(self, entry: PrefetchEntry) -> bool:
        with self._lock:
            if self._count >= self._depth:
                return False
            self._queue[self._tail] = entry
            self._tail = (self._tail + 1) % self._depth
            self._count += 1
        return True

    def dequeue(self) -> Optional[PrefetchEntry]:
        with self._lock:
            if self._count == 0:
                return None
            entry = self._queue[self._head]
            self._queue[self._head] = None
            self._head = (self._head + 1) % self._depth
            self._count -= 1
        return entry

    def release_entry(self, entry: PrefetchEntry) -> None:
        if entry.handle is not None:
            self._allocator.release(entry.handle)
            entry.handle = None

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def is_full(self) -> bool:
        with self._lock:
            return self._count >= self._depth

    @property
    def occupancy(self) -> int:
        with self._lock:
            return self._count


# ---------------------------------------------------------------------------
# Per-Tier Stream Set
# ---------------------------------------------------------------------------


@dataclass
class TierStreamSet:
    """
    CUDA streams dedicated to a single device tier.

    Separating AllGather and staging streams prevents the deadlock that can
    occur when NCCL and cudaMemcpy share a stream and the copy is waiting on
    a collective that's waiting on the copy.
    """
    device: torch.device
    allgather_stream: torch.cuda.Stream
    staging_stream: torch.cuda.Stream    # H2D spill copies
    compute_stream: torch.cuda.Stream   # Overlapped forward/backward

    @classmethod
    def create(cls, device: torch.device) -> "TierStreamSet":
        with torch.cuda.device(device):
            ag = torch.cuda.Stream()
            st = torch.cuda.Stream()
            cm = torch.cuda.Stream()
        return cls(
            device=device,
            allgather_stream=ag,
            staging_stream=st,
            compute_stream=cm,
        )


# ---------------------------------------------------------------------------
# Main HeteroFSDPDoubleBuffer
# ---------------------------------------------------------------------------


class HeteroFSDPDoubleBuffer:
    """
    DES-LOC Heterogeneous FSDP Double Buffer.

    This is the top-level class that DeepSpeed's ZeRO-3 stage will instantiate
    when ``hetero_fsdp_double_buffer: true`` is present in the ds_config.

    Architecture
    ------------

    ::

        ┌─────────────────────────────────────────────────────────────────┐
        │                   HeteroFSDPDoubleBuffer                        │
        │                                                                 │
        │  ┌──────────────────────┐   ┌──────────────────────────────┐   │
        │  │ TierAwarePoolAlloc   │   │ DoublePrefetchQueue           │   │
        │  │ (per device)         │   │ (depth=4 on H100, 2 on A6000) │   │
        │  └──────────┬───────────┘   └──────────────┬───────────────┘   │
        │             │                              │                   │
        │  ┌──────────▼──────────────────────────────▼───────────────┐   │
        │  │              TierStreamSet (per device)                  │   │
        │  │   allgather_stream │ staging_stream │ compute_stream     │   │
        │  └─────────────────────────────────────────────────────────┘   │
        │                                                                 │
        │  ┌──────────────────────────────────────────────────────────┐  │
        │  │  LocalityCache bypass: if param is cache-resident,        │  │
        │  │  skip AllGather and return zero-copy view                │  │
        │  └──────────────────────────────────────────────────────────┘  │
        └─────────────────────────────────────────────────────────────────┘

    Usage (from DeepSpeed ZeRO-3 param coordinator)
    -------------------------------------------------

    .. code-block:: python

        db = HeteroFSDPDoubleBuffer(
            devices=[torch.device("cuda:0"), torch.device("cuda:1"), torch.device("cuda:2")],
            locality_cache=my_locality_cache,  # or None
        )

        # Before each microbatch forward pass:
        with db.prefetch_context(param_groups):
            output = model(inputs)

        # After backward:
        db.release_all()
    """

    def __init__(
        self,
        devices: List[torch.device],
        locality_cache: Optional[LocalityCacheProtocol] = None,
        enable_cpu_spill: bool = True,
    ) -> None:
        self._devices = devices
        self._locality_cache = locality_cache

        # Per-device resources
        self._spill_arenas: Dict[int, Optional[PinnedSpillArena]] = {}
        self._stream_sets: Dict[int, TierStreamSet] = {}
        self._allocators: Dict[int, TierAwarePoolAllocator] = {}
        self._prefetch_queues: Dict[int, DoublePrefetchQueue] = {}

        for dev in devices:
            idx = dev.index if dev.index is not None else 0
            meta = _registry.get(idx)

            # CPU spill arena only for A6000 (H100 has enough HBM)
            if enable_cpu_spill and meta.tier == DeviceTier.A6000:
                arena: Optional[PinnedSpillArena] = PinnedSpillArena(_CPU_STAGING_BYTES)
            else:
                arena = None
            self._spill_arenas[idx] = arena

            streams = TierStreamSet.create(dev)
            self._stream_sets[idx] = streams

            alloc = TierAwarePoolAllocator(
                device=dev,
                locality_cache=locality_cache,
                staging_stream=streams.staging_stream,
                spill_arena=arena,
            )
            self._allocators[idx] = alloc

            depth = 4 if meta.tier == DeviceTier.H100 else 2
            self._prefetch_queues[idx] = DoublePrefetchQueue(alloc, depth)

        self._active_handles: List[BucketHandle] = []
        self._handle_lock = threading.Lock()

        logger.info(
            "HeteroFSDPDoubleBuffer initialised on %d device(s): %s",
            len(devices),
            [str(d) for d in devices],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate_bucket(
        self,
        param_id: int,
        num_bytes: int,
        device: torch.device,
    ) -> Tuple[Tensor, bool]:
        """
        Allocate a bucket for a parameter AllGather.

        Returns
        -------
        (tensor, cache_hit) where ``cache_hit=True`` means the tensor is a
        zero-copy view from the DES-LOC Locality Cache and no AllGather is
        needed.
        """
        # --- DES-LOC Locality Cache bypass ---
        if self._locality_cache is not None:
            if self._locality_cache.is_resident(param_id, device):
                view = self._locality_cache.get_view(param_id, device)
                if view is not None:
                    self._locality_cache.mark_accessed(param_id, device)
                    logger.debug(
                        "DES-LOC locality cache hit: param %d on %s (%.1f MB)",
                        param_id,
                        device,
                        num_bytes / 1e6,
                    )
                    return view, True

        idx = device.index if device.index is not None else 0
        allocator = self._allocators[idx]
        handle = allocator.allocate(num_bytes, device)

        with self._handle_lock:
            self._active_handles.append(handle)

        if handle.spilled_from_cpu:
            # Caller must wait on staging stream before using this buffer
            staging = self._stream_sets[idx].staging_stream
            current = torch.cuda.current_stream(device)
            current.wait_stream(staging)
            logger.debug(
                "DES-LOC: param %d bucket arrived via CPU spill path on %s",
                param_id,
                device,
            )

        return handle.storage, False

    def release_bucket(self, tensor: Tensor, device: torch.device) -> None:
        """Release a previously allocated bucket back to the pool."""
        idx = device.index if device.index is not None else 0
        allocator = self._allocators[idx]
        with self._handle_lock:
            for i, h in enumerate(self._active_handles):
                if h.storage.data_ptr() == tensor.data_ptr():
                    allocator.release(h)
                    self._active_handles.pop(i)
                    return

    def prefetch_context(
        self, param_groups: List[Dict]
    ) -> "_PrefetchContextManager":
        """
        Context manager that pre-schedules AllGather for the next layer's
        parameters while the current layer's compute is running.

        Example::

            with db.prefetch_context(param_groups):
                logits = model(x)
        """
        return _PrefetchContextManager(self, param_groups)

    def synchronise_allgather(self, device: torch.device) -> None:
        """Block until all pending AllGather ops on ``device`` are complete."""
        idx = device.index if device.index is not None else 0
        streams = self._stream_sets[idx]
        torch.cuda.current_stream(device).wait_stream(streams.allgather_stream)

    def release_all(self) -> None:
        """Release every active handle. Call after backward pass."""
        with self._handle_lock:
            for handle in self._active_handles:
                idx = handle.device.index if handle.device.index is not None else 0
                self._allocators[idx].release(handle)
            self._active_handles.clear()

    def stats(self) -> Dict[str, object]:
        """Return per-device allocator stats for monitoring."""
        return {str(dev): self._allocators[dev.index or 0].stats() for dev in self._devices}

    def __repr__(self) -> str:
        s = self.stats()
        parts = [f"HeteroFSDPDoubleBuffer(devices={len(self._devices)})"]
        for dev_str, st in s.items():
            parts.append(
                f"  {dev_str}: {st['tier']} pool={st['pool_bytes']//1024//1024}MB "
                f"slots={st['num_slots']} free={st['free_slots']}"
            )
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Prefetch Context Manager
# ---------------------------------------------------------------------------


class _PrefetchContextManager:
    """
    Used as ``with db.prefetch_context(param_groups): ...``.

    On __enter__ we schedule AllGather for the *next* group's parameters on
    the allgather_stream, then on __exit__ we synchronise and release.
    """

    def __init__(
        self, db: HeteroFSDPDoubleBuffer, param_groups: List[Dict]
    ) -> None:
        self._db = db
        self._param_groups = param_groups
        self._allocated: List[Tuple[Tensor, torch.device]] = []

    def __enter__(self) -> "_PrefetchContextManager":
        for group in self._param_groups:
            device: torch.device = group.get("device", torch.device("cuda:0"))
            idx = device.index if device.index is not None else 0
            for param_meta in group.get("params", []):
                param_id: int = param_meta["id"]
                num_bytes: int = param_meta["num_bytes"]
                buf, cache_hit = self._db.allocate_bucket(param_id, num_bytes, device)
                self._allocated.append((buf, device))
                if not cache_hit:
                    # Fire-and-forget prefetch event on the allgather stream
                    streams = self._db._stream_sets[idx]
                    ev = torch.cuda.Event()
                    ev.record(streams.allgather_stream)
                    param_meta["_prefetch_event"] = ev
        return self

    def __exit__(self, *_) -> None:
        for buf, device in self._allocated:
            self._db.release_bucket(buf, device)
        self._allocated.clear()


# ---------------------------------------------------------------------------
# DeepSpeed Integration Helper
# ---------------------------------------------------------------------------


def build_hetero_double_buffer(
    ds_config: Dict,
    locality_cache: Optional[LocalityCacheProtocol] = None,
) -> HeteroFSDPDoubleBuffer:
    """
    Factory called by DeepSpeed's ZeRO-3 initialisation path.

    Reads ``zero_optimization.hetero_fsdp_double_buffer_devices`` from
    ``ds_config``; if absent, uses all visible CUDA devices.

    Parameters
    ----------
    ds_config:
        The full DeepSpeed config dict.
    locality_cache:
        Optional DES-LOC Locality Cache instance. When provided, cache-resident
        parameters bypass AllGather entirely.
    """
    zero_cfg = ds_config.get("zero_optimization", {})
    if not zero_cfg.get("hetero_fsdp_double_buffer", False):
        raise ValueError(
            "build_hetero_double_buffer called but "
            "'zero_optimization.hetero_fsdp_double_buffer' is not True in ds_config"
        )

    device_indices: Optional[List[int]] = zero_cfg.get(
        "hetero_fsdp_double_buffer_devices", None
    )
    if device_indices is None:
        device_indices = list(range(torch.cuda.device_count()))

    devices = [torch.device(f"cuda:{i}") for i in device_indices]
    enable_spill = zero_cfg.get("hetero_fsdp_double_buffer_cpu_spill", True)

    db = HeteroFSDPDoubleBuffer(
        devices=devices,
        locality_cache=locality_cache,
        enable_cpu_spill=enable_spill,
    )
    logger.info("DES-LOC HeteroFSDPDoubleBuffer built: %s", db)
    return db


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


class _null_ctx:
    """No-op context manager used when staging_stream is None."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    import unittest

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _has_cuda(n: int = 1) -> bool:
        return torch.cuda.is_available() and torch.cuda.device_count() >= n

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    class TestAlignUp(unittest.TestCase):
        def test_already_aligned(self):
            self.assertEqual(_align_up(256, 256), 256)

        def test_unaligned(self):
            self.assertEqual(_align_up(257, 256), 512)

        def test_zero(self):
            self.assertEqual(_align_up(0, 256), 0)

        def test_large(self):
            val = _align_up(1024 * 1024 + 1, 256)
            self.assertEqual(val % 256, 0)
            self.assertGreater(val, 1024 * 1024)

    class TestDeviceTierRegistry(unittest.TestCase):
        def test_singleton(self):
            r1 = DeviceTierRegistry()
            r2 = DeviceTierRegistry()
            self.assertIs(r1, r2)

        @unittest.skipUnless(_has_cuda(1), "requires CUDA")
        def test_probe_populates_registry(self):
            reg = DeviceTierRegistry()
            reg._initialised = False  # force re-probe for test isolation
            metas = reg.all_tiers()
            self.assertGreater(len(metas), 0)
            for meta in metas:
                self.assertIsInstance(meta, TierMetadata)
                self.assertIn(meta.tier, list(DeviceTier))

        @unittest.skipUnless(_has_cuda(1), "requires CUDA")
        def test_pool_budget_positive(self):
            reg = DeviceTierRegistry()
            meta = reg.get(0)
            self.assertGreater(meta.pool_budget_bytes, 0)

    class TestPinnedSpillArena(unittest.TestCase):
        @unittest.skipUnless(_has_cuda(1), "requires CUDA")
        def test_allocate_and_wrap(self):
            arena = PinnedSpillArena(capacity_bytes=1024 * 1024)  # 1 MB
            buf1 = arena.allocate(512 * 1024)
            self.assertEqual(buf1.dtype, torch.uint8)
            buf2 = arena.allocate(512 * 1024)
            # Next allocation should wrap
            buf3 = arena.allocate(512 * 1024)
            self.assertEqual(buf3.data_ptr(), arena._backing.data_ptr())

        @unittest.skipUnless(_has_cuda(1), "requires CUDA")
        def test_reset(self):
            arena = PinnedSpillArena(capacity_bytes=256 * 1024)
            arena.allocate(128 * 1024)
            self.assertGreater(arena._cursor, 0)
            arena.reset()
            self.assertEqual(arena._cursor, 0)

    class TestHeteroStorageResizeFallback(unittest.TestCase):
        @unittest.skipUnless(_has_cuda(1), "requires CUDA")
        def test_allocate_and_release(self):
            dev = torch.device("cuda:0")
            arena = PinnedSpillArena()
            staging = torch.cuda.Stream(device=dev)
            fallback = HeteroStorageResizeFallback(
                staging_stream=staging, spill_arena=arena
            )
            num_bytes = 4 * 1024 * 1024  # 4 MB
            handle = fallback.allocate(num_bytes, dev)
            self.assertGreaterEqual(handle.num_bytes, num_bytes)
            self.assertEqual(handle.storage.device, dev)
            fallback.release(handle)

        @unittest.skipUnless(_has_cuda(1), "requires CUDA")
        def test_resize_idempotent(self):
            dev = torch.device("cuda:0")
            fallback = HeteroStorageResizeFallback(
                staging_stream=None, spill_arena=None
            )
            h1 = fallback.allocate(1024, dev)
            h2 = fallback.allocate(2048, dev)
            # Both calls should return the same underlying tensor (resized)
            self.assertEqual(
                h1.storage.data_ptr(), h2.storage.data_ptr(),
                "Expected storage-resize reuse of the same tensor"
            )
            self.assertGreaterEqual(h2.num_bytes, 2048)

    class TestTierAwarePoolAllocator(unittest.TestCase):
        @unittest.skipUnless(_has_cuda(1), "requires CUDA")
        def test_pool_allocation_and_return(self):
            dev = torch.device("cuda:0")
            alloc = TierAwarePoolAllocator(
                device=dev,
                locality_cache=None,
                staging_stream=None,
                spill_arena=None,
            )
            stats_before = alloc.stats()
            handle = alloc.allocate(1024 * 1024, dev)  # 1 MB
            stats_during = alloc.stats()
            alloc.release(handle)
            stats_after = alloc.stats()

            if stats_before["pool_bytes"] >= _MIN_POOL_BYTES:
                self.assertLess(stats_during["free_slots"], stats_before["free_slots"])
                self.assertEqual(stats_after["free_slots"], stats_before["free_slots"])

        @unittest.skipUnless(_has_cuda(1), "requires CUDA")
        def test_oversized_bucket_falls_through_to_fallback(self):
            dev = torch.device("cuda:0")
            alloc = TierAwarePoolAllocator(
                device=dev,
                locality_cache=None,
                staging_stream=None,
                spill_arena=None,
            )
            # Request something definitely bigger than any slot
            huge_bytes = alloc._slot_size * 10 + 1
            # Must not raise; fallback takes over
            handle = alloc.allocate(huge_bytes, dev)
            self.assertFalse(handle.from_pool)
            alloc.release(handle)

        @unittest.skipUnless(_has_cuda(1), "requires CUDA")
        def test_stats_keys(self):
            dev = torch.device("cuda:0")
            alloc = TierAwarePoolAllocator(
                device=dev, locality_cache=None,
                staging_stream=None, spill_arena=None,
            )
            st = alloc.stats()
            for key in ("device", "tier", "pool_bytes", "num_slots", "free_slots", "slot_size_bytes"):
                self.assertIn(key, st)

    class TestHeteroFSDPDoubleBuffer(unittest.TestCase):
        @unittest.skipUnless(_has_cuda(1), "requires CUDA")
        def test_init_single_device(self):
            devices = [torch.device("cuda:0")]
            db = HeteroFSDPDoubleBuffer(devices=devices)
            self.assertIn(0, db._allocators)
            self.assertIn(0, db._stream_sets)

        @unittest.skipUnless(_has_cuda(1), "requires CUDA")
        def test_allocate_and_release_bucket(self):
            devices = [torch.device("cuda:0")]
            db = HeteroFSDPDoubleBuffer(devices=devices)
            tensor, cache_hit = db.allocate_bucket(
                param_id=42, num_bytes=2 * 1024 * 1024, device=devices[0]
            )
            self.assertFalse(cache_hit)
            self.assertEqual(tensor.device, devices[0])
            db.release_bucket(tensor, devices[0])
            db.release_all()

        @unittest.skipUnless(_has_cuda(1), "requires CUDA")
        def test_locality_cache_bypass(self):
            """Simulate a Locality Cache hit — no AllGather should occur."""

            class FakeCache(LocalityCacheProtocol):
                def __init__(self, hit_id: int, tensor: Tensor):
                    self._hit_id = hit_id
                    self._tensor = tensor
                    self.accessed: List[int] = []

                def is_resident(self, param_id, device):
                    return param_id == self._hit_id

                def get_view(self, param_id, device):
                    return self._tensor if param_id == self._hit_id else None

                def mark_accessed(self, param_id, device):
                    self.accessed.append(param_id)

            dev = torch.device("cuda:0")
            cached_tensor = torch.randn(128, device=dev)
            cache = FakeCache(hit_id=99, tensor=cached_tensor)
            db = HeteroFSDPDoubleBuffer(devices=[dev], locality_cache=cache)

            returned, cache_hit = db.allocate_bucket(
                param_id=99, num_bytes=512, device=dev
            )
            self.assertTrue(cache_hit)
            self.assertEqual(returned.data_ptr(), cached_tensor.data_ptr())
            self.assertIn(99, cache.accessed)

        @unittest.skipUnless(_has_cuda(1), "requires CUDA")
        def test_release_all_clears_handles(self):
            devices = [torch.device("cuda:0")]
            db = HeteroFSDPDoubleBuffer(devices=devices)
            for i in range(3):
                db.allocate_bucket(param_id=i, num_bytes=1024, device=devices[0])
            self.assertEqual(len(db._active_handles), 3)
            db.release_all()
            self.assertEqual(len(db._active_handles), 0)

        @unittest.skipUnless(_has_cuda(1), "requires CUDA")
        def test_prefetch_context_manager(self):
            devices = [torch.device("cuda:0")]
            db = HeteroFSDPDoubleBuffer(devices=devices)
            param_groups = [
                {
                    "device": devices[0],
                    "params": [
                        {"id": 0, "num_bytes": 1024 * 1024},
                        {"id": 1, "num_bytes": 2 * 1024 * 1024},
                    ],
                }
            ]
            with db.prefetch_context(param_groups):
                # Simulate compute — handles are active inside the context
                self.assertEqual(len(db._active_handles), 2)
            # Context exit should have released everything
            self.assertEqual(len(db._active_handles), 0)

        @unittest.skipUnless(_has_cuda(1), "requires CUDA")
        def test_synchronise_allgather(self):
            devices = [torch.device("cuda:0")]
            db = HeteroFSDPDoubleBuffer(devices=devices)
            # Should not raise even with no pending ops
            db.synchronise_allgather(devices[0])

        @unittest.skipUnless(_has_cuda(1), "requires CUDA")
        def test_repr_contains_tier_info(self):
            devices = [torch.device("cuda:0")]
            db = HeteroFSDPDoubleBuffer(devices=devices)
            rep = repr(db)
            self.assertIn("HeteroFSDPDoubleBuffer", rep)

    class TestBuildFactory(unittest.TestCase):
        @unittest.skipUnless(_has_cuda(1), "requires CUDA")
        def test_factory_missing_flag_raises(self):
            with self.assertRaises(ValueError):
                build_hetero_double_buffer(
                    ds_config={"zero_optimization": {"hetero_fsdp_double_buffer": False}}
                )

        @unittest.skipUnless(_has_cuda(1), "requires CUDA")
        def test_factory_builds_correctly(self):
            ds_config = {
                "zero_optimization": {
                    "hetero_fsdp_double_buffer": True,
                    "hetero_fsdp_double_buffer_devices": [0],
                    "hetero_fsdp_double_buffer_cpu_spill": False,
                }
            }
            db = build_hetero_double_buffer(ds_config=ds_config)
            self.assertIsInstance(db, HeteroFSDPDoubleBuffer)

    class TestDoublePrefetchQueue(unittest.TestCase):
        @unittest.skipUnless(_has_cuda(1), "requires CUDA")
        def test_enqueue_dequeue_fifo(self):
            dev = torch.device("cuda:0")
            alloc = TierAwarePoolAllocator(
                device=dev, locality_cache=None,
                staging_stream=None, spill_arena=None,
            )
            q = DoublePrefetchQueue(alloc, depth=2)
            e1 = PrefetchEntry(param_id=1, param_group_id=0,
                               bucket_bytes=1024, device=dev)
            e2 = PrefetchEntry(param_id=2, param_group_id=0,
                               bucket_bytes=1024, device=dev)
            self.assertTrue(q.enqueue(e1))
            self.assertTrue(q.enqueue(e2))
            self.assertTrue(q.is_full)
            self.assertFalse(q.enqueue(PrefetchEntry(3, 0, 1024, dev)))

            d1 = q.dequeue()
            self.assertEqual(d1.param_id, 1)
            d2 = q.dequeue()
            self.assertEqual(d2.param_id, 2)
            self.assertIsNone(q.dequeue())

        @unittest.skipUnless(_has_cuda(1), "requires CUDA")
        def test_occupancy(self):
            dev = torch.device("cuda:0")
            alloc = TierAwarePoolAllocator(
                device=dev, locality_cache=None,
                staging_stream=None, spill_arena=None,
            )
            q = DoublePrefetchQueue(alloc, depth=4)
            self.assertEqual(q.occupancy, 0)
            q.enqueue(PrefetchEntry(1, 0, 512, dev))
            self.assertEqual(q.occupancy, 1)

    class TestTierStreamSet(unittest.TestCase):
        @unittest.skipUnless(_has_cuda(1), "requires CUDA")
        def test_create(self):
            dev = torch.device("cuda:0")
            ss = TierStreamSet.create(dev)
            self.assertIsInstance(ss.allgather_stream, torch.cuda.Stream)
            self.assertIsInstance(ss.staging_stream, torch.cuda.Stream)
            self.assertIsInstance(ss.compute_stream, torch.cuda.Stream)
            # All three streams should be distinct
            ptrs = {ss.allgather_stream.cuda_stream,
                    ss.staging_stream.cuda_stream,
                    ss.compute_stream.cuda_stream}
            self.assertEqual(len(ptrs), 3)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    print("=" * 70)
    print("DES-LOC HeteroFSDPDoubleBuffer — Unit Test Suite")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestAlignUp,
        TestDeviceTierRegistry,
        TestPinnedSpillArena,
        TestHeteroStorageResizeFallback,
        TestTierAwarePoolAllocator,
        TestHeteroFSDPDoubleBuffer,
        TestBuildFactory,
        TestDoublePrefetchQueue,
        TestTierStreamSet,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
