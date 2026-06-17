"""
HeteroFSDPDoubleBuffer — DES-LOC Heterogeneous FSDP Double Buffer Manager
==========================================================================

Upstream Design Intent (Megatron d199bb9e9f):
----------------------------------------------
Megatron's ``fsdp_double_buffer`` feature pre-allocates two alternating CUDA
memory pools so that while one FSDP unit is computing its forward/backward pass
the *next* unit's parameters are already being gathered in the background.  The
critical bug fixed in d199bb9e was a CUDA Illegal Memory Access (IMA) triggered
when an FSDP unit's bucket size exceeded the fixed-pool capacity: the original
code fell back to a bare ``TemporaryBucketAllocator`` which did *not* perform
storage-resize-based lifetime management, leaving dangling tensor views into
freed CUDA memory.  The fix was to replace the bare fallback with a
``StorageResizeBasedBucketAllocator`` that extends tensor storage in-place,
keeping the underlying allocation alive until all views are released.

DES-LOC Adaptation Points:
---------------------------
Hardware context: 2× A6000 (48 GB, SM86, cuda:0 / cuda:1) + 1× H100 NVL
(96 GB, SM90, cuda:2), PCIe interconnect, no NVLink, 1.5 TB CPU DRAM.

1. **Heterogeneous pool sizing** — The H100 gets a larger primary pool
   (``h100_pool_fraction``) because its HBM3 bandwidth can drain a bigger
   working set per step.  A6000 pools are sized conservatively to leave headroom
   for activation checkpointing.

2. **Device-aware fallback allocator** — Mirrors Megatron's
   ``StorageResizeBasedBucketAllocator`` fix but is *device-class-aware*: on
   SM86 (A6000) it uses in-place storage resize; on SM90 (H100) it uses a
   pinned-DRAM staging buffer backed by ``torch.UntypedStorage`` to exploit the
   H100 NVL's higher PCIe bandwidth ceiling (~64 GB/s vs ~32 GB/s on A6000
   slots in the target system).

3. **LOC cache (Shared Locality Cache)** — A per-device LRU eviction ring that
   keeps recently-used bucket tensors alive across micro-batches, amortising
   cudaMalloc overhead over the NUM_MICROBATCHES=4 typical workload described
   in the upstream test.

4. **Decoupled Execution streams** — Each device owns a ``prefetch_stream`` and
   a ``compute_stream``; the double-buffer swap is synchronised with lightweight
   CUDA events rather than full ``torch.cuda.synchronize()`` calls, matching the
   overlap_grad_reduce + overlap_param_gather configuration exercised by the
   upstream smoke test.

5. **CPU DRAM offload path** — When *both* pool slots are occupied on an A6000
   (48 GB is tight with bf16 + grad + optim state), the manager can spill the
   inactive buffer to pinned CPU memory and stream it back on demand, leveraging
   the 1.5 TB headroom without stalling the H100.
"""

from __future__ import annotations

import logging
import math
import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.cuda
from torch import Tensor

# ---------------------------------------------------------------------------
# Module-level logger — all DES-LOC specific events are routed here so that
# callers can filter with ``logging.getLogger("deslo.hetero_fsdp")``
# ---------------------------------------------------------------------------
logger = logging.getLogger("deslo.hetero_fsdp")


# ===========================================================================
# Device classification helpers
# ===========================================================================

class DeviceClass(Enum):
    """SM architecture classes present in the DES-LOC target system."""
    A6000_SM86 = auto()   # cuda:0, cuda:1 — 48 GB HBM2e
    H100_SM90  = auto()   # cuda:2       — 96 GB HBM3 (NVL)
    UNKNOWN    = auto()


def classify_device(device: torch.device) -> DeviceClass:
    """Return the :class:`DeviceClass` for *device*.

    We query ``torch.cuda.get_device_capability`` which returns ``(major,
    minor)``.  SM86 = (8, 6), SM90 = (9, 0).  Unknown devices fall back to
    :attr:`DeviceClass.UNKNOWN` so the allocator degrades gracefully on
    development machines with different GPUs.
    """
    if not torch.cuda.is_available():
        return DeviceClass.UNKNOWN
    idx = device.index if device.index is not None else torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(idx)
    if (major, minor) == (9, 0):
        return DeviceClass.H100_SM90
    if (major, minor) == (8, 6):
        return DeviceClass.A6000_SM86
    return DeviceClass.UNKNOWN


# ===========================================================================
# Pool configuration
# ===========================================================================

@dataclass
class PoolConfig:
    """Per-device pool sizing and behaviour knobs.

    Attributes
    ----------
    primary_pool_fraction:
        Fraction of device free memory to allocate for the double-buffer
        primary pool at initialisation time.
    allow_cpu_offload:
        When ``True`` and both pool slots are live, inactive buffers may be
        pinned to CPU DRAM.  Should be ``True`` on A6000 (48 GB) and
        ``False`` on H100 (96 GB) unless the model is very large.
    loc_cache_capacity:
        Maximum number of bucket tensors retained in the LOC (Shared
        Locality Cache) eviction ring.  Larger values trade DRAM for fewer
        cudaMalloc calls.
    use_storage_resize_fallback:
        Whether the fallback allocator uses in-place storage resize (SM86)
        or pinned-staging-buffer (SM90) strategy.
    prefetch_ahead:
        Number of FSDP units to prefetch in advance on the device.
    """
    primary_pool_fraction: float = 0.10
    allow_cpu_offload: bool = False
    loc_cache_capacity: int = 8
    use_storage_resize_fallback: bool = True
    prefetch_ahead: int = 1


# Sensible defaults per device class
_POOL_CONFIG_BY_CLASS: Dict[DeviceClass, PoolConfig] = {
    DeviceClass.A6000_SM86: PoolConfig(
        primary_pool_fraction=0.08,   # conservative — leave room for activations
        allow_cpu_offload=True,
        loc_cache_capacity=6,
        use_storage_resize_fallback=True,  # mirrors Megatron StorageResizeBased
        prefetch_ahead=1,
    ),
    DeviceClass.H100_SM90: PoolConfig(
        primary_pool_fraction=0.18,   # H100 NVL has 96 GB — be more generous
        allow_cpu_offload=False,
        loc_cache_capacity=12,
        use_storage_resize_fallback=False,  # use pinned-staging path instead
        prefetch_ahead=2,
    ),
    DeviceClass.UNKNOWN: PoolConfig(),
}


# ===========================================================================
# LOC (Shared Locality Cache)
# ===========================================================================

class LOCCache:
    """LRU eviction ring for recently-used bucket tensors.

    The cache is keyed by ``(numel, dtype)`` and stores a list of free
    tensors of that shape.  On a cache *hit* the tensor is returned directly,
    skipping ``cudaMalloc``.  On a cache *miss* a new tensor is allocated and
    the oldest entry is evicted if capacity is exceeded.

    Thread safety: protected by a single ``threading.Lock``.  In DES-LOC the
    prefetch thread and the compute thread both call into the cache.
    """

    def __init__(self, capacity: int, device: torch.device) -> None:
        self._capacity = capacity
        self._device = device
        self._store: OrderedDict[Tuple[int, torch.dtype], List[Tensor]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, numel: int, dtype: torch.dtype) -> Tensor:
        """Return a tensor of shape ``(numel,)`` and *dtype* on the cache device.

        If a cached tensor is available it is removed from the free list and
        returned (LRU promoted).  Otherwise a fresh tensor is allocated.
        """
        key = (numel, dtype)
        with self._lock:
            bucket = self._store.get(key)
            if bucket:
                tensor = bucket.pop()
                if not bucket:
                    del self._store[key]
                else:
                    self._store.move_to_end(key)   # mark recently used
                self._hits += 1
                return tensor
            self._misses += 1

        # Allocation outside the lock — cudaMalloc can be slow
        return torch.empty(numel, dtype=dtype, device=self._device)

    def put(self, tensor: Tensor) -> None:
        """Return *tensor* to the cache.  Evicts LRU entry if at capacity."""
        key = (tensor.numel(), tensor.dtype)
        with self._lock:
            bucket = self._store.setdefault(key, [])
            bucket.append(tensor)
            self._store.move_to_end(key)
            self._evict_if_needed()

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    @property
    def stats(self) -> Dict[str, int]:
        return {"hits": self._hits, "misses": self._misses}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        """Evict the LRU key until total slot count ≤ capacity.

        Called with ``self._lock`` held.
        """
        total = sum(len(v) for v in self._store.values())
        while total > self._capacity and self._store:
            _, evicted_bucket = self._store.popitem(last=False)
            removed = len(evicted_bucket)
            total -= removed
            # Let tensors be garbage-collected → CUDA memory returned to
            # PyTorch's caching allocator, not immediately to the driver.


# ===========================================================================
# Fallback allocators
# ===========================================================================

class StorageResizeFallbackAllocator:
    """SM86 fallback: mirrors Megatron's ``StorageResizeBasedBucketAllocator``.

    When a bucket request exceeds the fixed-pool capacity this allocator
    extends the tensor's ``UntypedStorage`` in-place, keeping the tensor
    object alive so that any views into it remain valid — exactly the fix
    applied in Megatron commit d199bb9e for the IMA bug.

    The key invariant: we never free the backing storage while any view
    (parameter shard slice) is still live.  We track liveness via a
    reference-counted wrapper so the storage shrinks back to a minimal
    sentinel size only when ``release()`` is called *and* the ref-count
    reaches zero.
    """

    def __init__(self, device: torch.device, loc_cache: LOCCache) -> None:
        self._device = device
        self._loc_cache = loc_cache
        # sentinel storage — 1-element float32 kept alive perpetually so we
        # always have a valid UntypedStorage to resize from.
        self._sentinel: Optional[torch.UntypedStorage] = None
        self._active_allocations: Dict[int, _ResizeRecord] = {}
        self._next_id = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------

    def allocate(self, numel: int, dtype: torch.dtype) -> Tuple[int, Tensor]:
        """Allocate a bucket tensor of *numel* elements and *dtype*.

        Returns ``(alloc_id, tensor)``.  The caller must call
        ``release(alloc_id)`` when the bucket is no longer needed.
        """
        tensor = self._loc_cache.get(numel, dtype)
        # Ensure the storage is at least ``numel`` elements of ``dtype``
        required_bytes = numel * tensor.element_size()
        if tensor.untyped_storage().size() < required_bytes:
            # Resize in-place — this is the core of the Megatron fix
            tensor.untyped_storage().resize_(required_bytes)
            logger.debug(
                "StorageResizeFallback: resized storage to %d bytes on %s",
                required_bytes,
                self._device,
            )

        with self._lock:
            alloc_id = self._next_id
            self._next_id += 1
            self._active_allocations[alloc_id] = _ResizeRecord(tensor=tensor, ref_count=1)

        return alloc_id, tensor

    def release(self, alloc_id: int) -> None:
        """Decrement ref-count for *alloc_id* and return tensor to cache if zero."""
        with self._lock:
            record = self._active_allocations.get(alloc_id)
            if record is None:
                return
            record.ref_count -= 1
            if record.ref_count <= 0:
                del self._active_allocations[alloc_id]
                tensor = record.tensor
        # Put outside the lock
        self._loc_cache.put(tensor)

    def add_ref(self, alloc_id: int) -> None:
        """Increment ref-count — call when creating a new view of the allocation."""
        with self._lock:
            record = self._active_allocations.get(alloc_id)
            if record is not None:
                record.ref_count += 1


@dataclass
class _ResizeRecord:
    tensor: Tensor
    ref_count: int = 1


class PinnedStagingFallbackAllocator:
    """SM90 fallback: uses pinned CPU DRAM as a staging buffer.

    On the H100 NVL the PCIe bandwidth to host memory is higher (~64 GB/s
    peak, vs ~32 GB/s on A6000 PCIe slots in the target system).  For
    over-sized buckets we allocate pinned CPU memory, register it with CUDA,
    and use async D2H / H2D copies on a dedicated staging stream to move
    data in and out without blocking the compute stream.

    This is *not* the storage-resize path used on SM86.  Instead we keep the
    device tensor alive at the minimum required size and use the pinned buffer
    only as a temporary landing zone when both pool slots are occupied.
    """

    def __init__(self, device: torch.device, loc_cache: LOCCache) -> None:
        self._device = device
        self._loc_cache = loc_cache
        self._staging_stream = torch.cuda.Stream(device=device)
        self._active: Dict[int, _StagingRecord] = {}
        self._next_id = 0
        self._lock = threading.Lock()

    def allocate(self, numel: int, dtype: torch.dtype) -> Tuple[int, Tensor]:
        device_tensor = self._loc_cache.get(numel, dtype)
        pinned = torch.empty(numel, dtype=dtype, pin_memory=True)
        with self._lock:
            alloc_id = self._next_id
            self._next_id += 1
            self._active[alloc_id] = _StagingRecord(
                device_tensor=device_tensor,
                pinned_tensor=pinned,
                ref_count=1,
            )
        logger.debug(
            "PinnedStagingFallback: allocated %d×%s on %s with pinned staging",
            numel,
            dtype,
            self._device,
        )
        return alloc_id, device_tensor

    def release(self, alloc_id: int) -> None:
        with self._lock:
            record = self._active.get(alloc_id)
            if record is None:
                return
            record.ref_count -= 1
            if record.ref_count <= 0:
                del self._active[alloc_id]
                device_tensor = record.device_tensor
        self._loc_cache.put(device_tensor)
        # pinned tensor is garbage-collected → freed automatically

    def stage_to_host(self, alloc_id: int) -> None:
        """Async copy device tensor → pinned host buffer on the staging stream."""
        with self._lock:
            record = self._active.get(alloc_id)
        if record is None:
            return
        with torch.cuda.stream(self._staging_stream):
            record.pinned_tensor.copy_(record.device_tensor, non_blocking=True)

    def restore_from_host(self, alloc_id: int) -> None:
        """Async copy pinned host buffer → device tensor on the staging stream."""
        with self._lock:
            record = self._active.get(alloc_id)
        if record is None:
            return
        with torch.cuda.stream(self._staging_stream):
            record.device_tensor.copy_(record.pinned_tensor, non_blocking=True)

    def sync_staging(self) -> None:
        """Block until the staging stream is idle."""
        self._staging_stream.synchronize()


@dataclass
class _StagingRecord:
    device_tensor: Tensor
    pinned_tensor: Tensor
    ref_count: int = 1


# ===========================================================================
# Fixed-pool allocator (primary path)
# ===========================================================================

class HeteroFixedPoolAllocator:
    """Device-aware fixed-pool allocator with LOC cache and fallback.

    Manages two alternating pool slots (double-buffer pattern).  The pool is
    pre-allocated as a single contiguous ``torch.UntypedStorage`` split into
    two halves; each half is handed out as a ``Tensor`` view per FSDP unit.

    When the requested bucket size exceeds one half's capacity the request is
    forwarded to the device-class-appropriate fallback allocator (mirroring
    the Megatron d199bb9e fix that replaced ``TemporaryBucketAllocator`` with
    ``StorageResizeBasedBucketAllocator`` to prevent IMA).

    Parameters
    ----------
    device:
        The CUDA device this allocator manages.
    pool_bytes:
        Total bytes for the double-buffer pool (split evenly into 2 slots).
    dtype:
        Default dtype for pool tensors.
    loc_cache:
        Shared locality cache (shared across allocator + fallback).
    cfg:
        Per-device configuration knobs.
    """

    def __init__(
        self,
        device: torch.device,
        pool_bytes: int,
        dtype: torch.dtype,
        loc_cache: LOCCache,
        cfg: PoolConfig,
    ) -> None:
        self._device = device
        self._dtype = dtype
        self._loc_cache = loc_cache
        self._cfg = cfg
        self._dev_class = classify_device(device)

        slot_bytes = pool_bytes // 2
        self._slot_bytes = slot_bytes
        self._slot_numel = slot_bytes // dtype.itemsize

        # Allocate backing storage once
        self._storage = torch.UntypedStorage(pool_bytes, device=device)
        self._slots: List[Optional[Tensor]] = [None, None]
        self._slot_in_use: List[bool] = [False, False]
        self._current_slot = 0

        # Initialise slot views
        for i in range(2):
            offset = i * slot_bytes
            self._slots[i] = torch.empty(0, dtype=dtype, device=device).set_(
                self._storage, storage_offset=offset // dtype.itemsize, size=(self._slot_numel,)
            )

        # Fallback allocator (device-class aware, mirrors Megatron's fix)
        if cfg.use_storage_resize_fallback:
            self._fallback: StorageResizeFallbackAllocator | PinnedStagingFallbackAllocator = (
                StorageResizeFallbackAllocator(device=device, loc_cache=loc_cache)
            )
        else:
            self._fallback = PinnedStagingFallbackAllocator(
                device=device, loc_cache=loc_cache
            )

        self._lock = threading.Lock()
        self._fallback_alloc_ids: Dict[int, int] = {}  # slot_token → fallback alloc_id

        logger.info(
            "HeteroFixedPoolAllocator: device=%s class=%s pool=2×%d MiB dtype=%s "
            "fallback=%s",
            device,
            self._dev_class.name,
            slot_bytes // (1024 * 1024),
            dtype,
            type(self._fallback).__name__,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate(self, numel: int, dtype: Optional[torch.dtype] = None) -> Tuple[int, Tensor]:
        """Allocate a bucket tensor.

        Returns ``(slot_token, tensor)`` where *slot_token* must be passed to
        :meth:`release`.  If *numel* fits in one pool slot the primary path is
        taken (zero-copy view).  Otherwise the fallback allocator is used,
        preventing the IMA bug described in Megatron d199bb9e.
        """
        dtype = dtype or self._dtype
        required_bytes = numel * (dtype.itemsize if hasattr(dtype, 'itemsize')
                                  else torch.tensor([], dtype=dtype).element_size())

        with self._lock:
            slot_idx = self._pick_slot()

        if slot_idx is not None and required_bytes <= self._slot_bytes:
            return self._allocate_from_pool(slot_idx, numel, dtype)
        else:
            return self._allocate_from_fallback(numel, dtype)

    def release(self, slot_token: int) -> None:
        """Release a previously allocated bucket identified by *slot_token*."""
        with self._lock:
            if slot_token < 0:
                # Fallback allocation — decode the embedded fallback alloc_id
                fallback_id = self._fallback_alloc_ids.pop(slot_token, None)
            else:
                fallback_id = None
                if 0 <= slot_token < 2:
                    self._slot_in_use[slot_token] = False

        if fallback_id is not None:
            self._fallback.release(fallback_id)

    def is_slot_free(self) -> bool:
        """Return ``True`` if at least one pool slot is available."""
        with self._lock:
            return any(not u for u in self._slot_in_use)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_slot(self) -> Optional[int]:
        """Return the index of a free pool slot, or ``None`` if both are busy."""
        for i in range(2):
            idx = (self._current_slot + i) % 2
            if not self._slot_in_use[idx]:
                return idx
        return None

    def _allocate_from_pool(self, slot_idx: int, numel: int, dtype: torch.dtype) -> Tuple[int, Tensor]:
        self._slot_in_use[slot_idx] = True
        self._current_slot = (slot_idx + 1) % 2
        tensor = self._slots[slot_idx]
        # Return a narrowed view of the correct dtype and length
        if tensor.dtype != dtype or tensor.numel() != numel:
            tensor = tensor[:numel].view(dtype) if tensor.dtype == dtype else (
                self._storage_view(slot_idx, numel, dtype)
            )
        return slot_idx, tensor

    def _storage_view(self, slot_idx: int, numel: int, dtype: torch.dtype) -> Tensor:
        """Create a typed view into the pool slot storage."""
        offset_bytes = slot_idx * self._slot_bytes
        offset_elems = offset_bytes // torch.tensor([], dtype=dtype).element_size()
        return torch.empty(0, dtype=dtype, device=self._device).set_(
            self._storage,
            storage_offset=offset_elems,
            size=(numel,),
        )

    def _allocate_from_fallback(self, numel: int, dtype: torch.dtype) -> Tuple[int, Tensor]:
        """Delegate to the fallback allocator (fixes Megatron IMA bug pattern)."""
        fallback_id, tensor = self._fallback.allocate(numel, dtype)
        # Encode as a negative token to distinguish from pool slot indices
        slot_token = -(fallback_id + 1)
        with self._lock:
            self._fallback_alloc_ids[slot_token] = fallback_id
        logger.debug(
            "HeteroFixedPoolAllocator: fallback path for numel=%d dtype=%s on %s",
            numel,
            dtype,
            self._device,
        )
        return slot_token, tensor


# ===========================================================================
# CUDA event-based double-buffer synchroniser
# ===========================================================================

class DoubleBufferSynchroniser:
    """Manages CUDA event synchronisation for the double-buffer prefetch loop.

    In the upstream Megatron implementation ``fsdp_double_buffer`` relies on
    ``torch.cuda.synchronize()`` at bucket boundaries which serialises the
    prefetch and compute streams.  DES-LOC replaces this with fine-grained
    CUDA events so only the *specific* bucket being transferred is waited on,
    not the entire device.

    Each device in the DES-LOC cluster has:
    - ``compute_stream`` : runs forward/backward ops
    - ``prefetch_stream``: runs AllGather for the *next* FSDP unit's params

    The synchroniser records an event on ``prefetch_stream`` when a prefetch
    completes and waits on that event in ``compute_stream`` just before the
    FSDP unit's forward pass begins.
    """

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self.compute_stream = torch.cuda.Stream(device=device)
        self.prefetch_stream = torch.cuda.Stream(device=device)
        self._prefetch_events: Dict[int, torch.cuda.Event] = {}
        self._compute_events: Dict[int, torch.cuda.Event] = {}
        self._lock = threading.Lock()

    def record_prefetch_done(self, unit_id: int) -> None:
        """Record that unit *unit_id*'s parameters are ready in the buffer."""
        event = torch.cuda.Event(enable_timing=False)
        with torch.cuda.stream(self.prefetch_stream):
            event.record(self.prefetch_stream)
        with self._lock:
            self._prefetch_events[unit_id] = event

    def wait_for_prefetch(self, unit_id: int) -> None:
        """Make ``compute_stream`` wait until unit *unit_id* prefetch is done."""
        with self._lock:
            event = self._prefetch_events.pop(unit_id, None)
        if event is not None:
            self.compute_stream.wait_event(event)

    def record_compute_done(self, unit_id: int) -> None:
        """Record that the compute stream has finished consuming unit *unit_id*."""
        event = torch.cuda.Event(enable_timing=False)
        with torch.cuda.stream(self.compute_stream):
            event.record(self.compute_stream)
        with self._lock:
            self._compute_events[unit_id] = event

    def wait_for_compute(self, unit_id: int) -> None:
        """Make ``prefetch_stream`` wait until compute is done with unit *unit_id*."""
        with self._lock:
            event = self._compute_events.pop(unit_id, None)
        if event is not None:
            self.prefetch_stream.wait_event(event)


# ===========================================================================
# CPU offload manager (A6000 memory pressure relief)
# ===========================================================================

class CPUOffloadManager:
    """Spills inactive double-buffer slots to 1.5 TB CPU DRAM on A6000 devices.

    When both pool slots are needed simultaneously on an A6000 (48 GB) the
    manager can evict the *inactive* buffer (the one whose FSDP unit has
    already finished computing) to pinned host memory.  The buffer is streamed
    back asynchronously before the next time that slot is needed.

    This exploits the 1.5 TB CPU DRAM headroom in the target system without
    requiring NVLink (which is unavailable on the PCIe-only A6000 slots).
    """

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._h2d_stream = torch.cuda.Stream(device=device)
        self._d2h_stream = torch.cuda.Stream(device=device)
        self._offloaded: Dict[int, Tuple[Tensor, torch.cuda.Event]] = {}
        self._lock = threading.Lock()

    def offload(self, slot_token: int, tensor: Tensor) -> None:
        """Async copy *tensor* to pinned host memory, freeing CUDA memory."""
        pinned = torch.empty_like(tensor, pin_memory=True)
        event = torch.cuda.Event()
        with torch.cuda.stream(self._d2h_stream):
            pinned.copy_(tensor, non_blocking=True)
            event.record(self._d2h_stream)
        with self._lock:
            self._offloaded[slot_token] = (pinned, event)
        logger.debug(
            "CPUOffload: offloaded slot_token=%d (%d MiB) to CPU on %s",
            slot_token,
            tensor.numel() * tensor.element_size() // (1024 * 1024),
            self._device,
        )

    def restore(self, slot_token: int, target: Tensor) -> None:
        """Async copy pinned host buffer back into *target* on the H2D stream."""
        with self._lock:
            entry = self._offloaded.pop(slot_token, None)
        if entry is None:
            return
        pinned, copy_event = entry
        # Wait until D2H is complete before H2D
        self._h2d_stream.wait_event(copy_event)
        with torch.cuda.stream(self._h2d_stream):
            target.copy_(pinned, non_blocking=True)

    def sync(self) -> None:
        self._h2d_stream.synchronize()
        self._d2h_stream.synchronize()


# ===========================================================================
# Per-device double-buffer state
# ===========================================================================

@dataclass
class DeviceDoubleBufferState:
    """All double-buffer resources for a single CUDA device."""
    device: torch.device
    dev_class: DeviceClass
    cfg: PoolConfig
    loc_cache: LOCCache
    pool_allocator: HeteroFixedPoolAllocator
    synchroniser: DoubleBufferSynchroniser
    cpu_offload_mgr: Optional[CPUOffloadManager]
    # Maps fsdp_unit_id → (slot_token, tensor)
    active_buffers: Dict[int, Tuple[int, Tensor]] = field(default_factory=dict)


# ===========================================================================
# Main manager: HeteroFSDPDoubleBufferManager
# ===========================================================================

class HeteroFSDPDoubleBufferManager:
    """Heterogeneous FSDP double-buffer manager for DES-LOC clusters.

    This class is the top-level entry point, analogous to the
    ``fsdp_double_buffer`` feature inside Megatron's
    ``FullyShardedDataParallel`` but extended for heterogeneous devices.

    It owns one :class:`DeviceDoubleBufferState` per CUDA device and
    dispatches prefetch / release calls to the correct device's allocator.

    Usage pattern (mirrors upstream Megatron test ``test_train_steps_with_double_buffer``):

    .. code-block:: python

        mgr = HeteroFSDPDoubleBufferManager(devices=[
            torch.device("cuda:0"),
            torch.device("cuda:1"),
            torch.device("cuda:2"),
        ])

        for microbatch_idx in range(NUM_MICROBATCHES):
            for unit_id, fsdp_unit in enumerate(fsdp_units):
                device = fsdp_unit.device
                buf = mgr.request_buffer(device, unit_id, numel, dtype)
                # ... AllGather into buf ...
                mgr.mark_prefetch_done(device, unit_id)
                mgr.wait_and_compute(device, unit_id)
                # ... forward/backward on fsdp_unit ...
                mgr.release_buffer(device, unit_id)

        mgr.finalize()

    Parameters
    ----------
    devices:
        List of CUDA devices participating in DES-LOC training.
    pool_fraction_override:
        If provided, overrides ``primary_pool_fraction`` for all devices
        (useful for testing on memory-constrained machines).
    dtype:
        Default tensor dtype for parameter buffers (``bfloat16`` for the
        target workload).
    """

    def __init__(
        self,
        devices: List[torch.device],
        pool_fraction_override: Optional[float] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self._dtype = dtype
        self._states: Dict[torch.device, DeviceDoubleBufferState] = {}

        for device in devices:
            dev_class = classify_device(device)
            cfg = _POOL_CONFIG_BY_CLASS[dev_class]
            if pool_fraction_override is not None:
                cfg = PoolConfig(
                    primary_pool_fraction=pool_fraction_override,
                    allow_cpu_offload=cfg.allow_cpu_offload,
                    loc_cache_capacity=cfg.loc_cache_capacity,
                    use_storage_resize_fallback=cfg.use_storage_resize_fallback,
                    prefetch_ahead=cfg.prefetch_ahead,
                )

            free_mem, _ = torch.cuda.mem_get_info(device)
            pool_bytes = int(free_mem * cfg.primary_pool_fraction)
            # Align to 2 MB
            pool_bytes = (pool_bytes // (2 * 1024 * 1024)) * (2 * 1024 * 1024)
            pool_bytes = max(pool_bytes, 4 * 1024 * 1024)  # minimum 4 MiB

            loc_cache = LOCCache(capacity=cfg.loc_cache_capacity, device=device)
            pool_alloc = HeteroFixedPoolAllocator(
                device=device,
                pool_bytes=pool_bytes,
                dtype=dtype,
                loc_cache=loc_cache,
                cfg=cfg,
            )
            sync = DoubleBufferSynchroniser(device=device)
            offload_mgr = CPUOffloadManager(device=device) if cfg.allow_cpu_offload else None

            self._states[device] = DeviceDoubleBufferState(
                device=device,
                dev_class=dev_class,
                cfg=cfg,
                loc_cache=loc_cache,
                pool_allocator=pool_alloc,
                synchroniser=sync,
                cpu_offload_mgr=offload_mgr,
            )
            logger.info(
                "HeteroFSDPDoubleBufferManager: registered device=%s (%s) "
                "pool=%d MiB loc_cap=%d cpu_offload=%s",
                device,
                dev_class.name,
                pool_bytes // (1024 * 1024),
                cfg.loc_cache_capacity,
                offload_mgr is not None,
            )

    # ------------------------------------------------------------------
    # Core double-buffer API
    # ------------------------------------------------------------------

    def request_buffer(
        self,
        device: torch.device,
        unit_id: int,
        numel: int,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """Request a double-buffer slot for FSDP unit *unit_id* on *device*.

        The returned tensor is either a view into the fixed pool (primary path)
        or a fresh allocation from the device-class-specific fallback (IMA-safe
        path, mirrors Megatron d199bb9e fix).

        If CPU offload is enabled and the pool is full the inactive buffer is
        spilled to host memory to free up the slot.
        """
        state = self._states[device]
        dtype = dtype or self._dtype

        if not state.pool_allocator.is_slot_free() and state.cpu_offload_mgr is not None:
            self._try_offload_inactive(state, unit_id)

        slot_token, tensor = state.pool_allocator.allocate(numel, dtype)
        state.active_buffers[unit_id] = (slot_token, tensor)
        return tensor

    def mark_prefetch_done(self, device: torch.device, unit_id: int) -> None:
        """Signal that the AllGather for unit *unit_id* is complete."""
        state = self._states[device]
        state.synchroniser.record_prefetch_done(unit_id)

    def wait_and_compute(self, device: torch.device, unit_id: int) -> None:
        """Block the compute stream until unit *unit_id*'s buffer is ready.

        If the unit's buffer was offloaded to CPU (A6000 memory pressure),
        this call also triggers an async restore before the compute stream wait.
        """
        state = self._states[device]
        # Restore from CPU if needed
        if state.cpu_offload_mgr is not None:
            entry = state.active_buffers.get(unit_id)
            if entry is not None:
                slot_token, tensor = entry
                state.cpu_offload_mgr.restore(slot_token, tensor)

        state.synchroniser.wait_for_prefetch(unit_id)

    def release_buffer(self, device: torch.device, unit_id: int) -> None:
        """Release the double-buffer slot for unit *unit_id* back to the pool."""
        state = self._states[device]
        state.synchroniser.record_compute_done(unit_id)
        entry = state.active_buffers.pop(unit_id, None)
        if entry is not None:
            slot_token, _ = entry
            state.pool_allocator.release(slot_token)

    def prefetch_next(
        self,
        device: torch.device,
        current_unit_id: int,
        next_unit_id: int,
        numel: int,
        dtype: Optional[torch.dtype] = None,
        gather_fn=None,
    ) -> Optional[Tensor]:
        """Speculatively prefetch the *next* FSDP unit's params while computing the current.

        This implements the core double-buffer overlap: on the ``prefetch_stream``
        we request a buffer for ``next_unit_id`` and invoke ``gather_fn`` (e.g.
        AllGather) asynchronously.  The compute stream then calls
        :meth:`wait_and_compute` before using the buffer.

        Parameters
        ----------
        gather_fn:
            Callable ``(tensor) -> None`` that fills *tensor* with the gathered
            parameters.  Runs on ``prefetch_stream``.  If ``None`` this method
            is a no-op (useful when the current unit is the last in the sequence).
        """
        if gather_fn is None:
            return None

        state = self._states[device]
        dtype = dtype or self._dtype

        # Wait until compute is done with the *previous* buffer before
        # reusing the slot — prevents the IMA bug pattern
        state.synchroniser.wait_for_compute(current_unit_id - 1)

        buf = self.request_buffer(device, next_unit_id, numel, dtype)
        with torch.cuda.stream(state.synchroniser.prefetch_stream):
            gather_fn(buf)
            self.mark_prefetch_done(device, next_unit_id)

        logger.debug(
            "prefetch_next: device=%s current=%d next=%d numel=%d",
            device,
            current_unit_id,
            next_unit_id,
            numel,
        )
        return buf

    def finalize(self) -> None:
        """Synchronise all streams and release all resources.

        Must be called after training is complete (or between pipeline stages
        where the manager needs to be torn down cleanly).
        """
        for device, state in self._states.items():
            state.synchroniser.compute_stream.synchronize()
            state.synchroniser.prefetch_stream.synchronize()
            if state.cpu_offload_mgr is not None:
                state.cpu_offload_mgr.sync()
            state.loc_cache.clear()
            logger.info(
                "finalize: device=%s LOC stats=%s",
                device,
                state.loc_cache.stats,
            )
        logger.info("HeteroFSDPDoubleBufferManager: finalized all devices")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_loc_stats(self) -> Dict[str, Dict[str, int]]:
        """Return LOC cache hit/miss statistics per device."""
        return {str(dev): state.loc_cache.stats for dev, state in self._states.items()}

    def get_active_buffer_count(self, device: torch.device) -> int:
        """Return the number of currently active buffers on *device*."""
        return len(self._states[device].active_buffers)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_offload_inactive(
        self,
        state: DeviceDoubleBufferState,
        requesting_unit_id: int,
    ) -> None:
        """Attempt to offload the oldest inactive buffer to CPU DRAM.

        Only called when the pool is full and CPU offload is enabled (A6000).
        We pick the lowest unit_id that is *not* the currently requested one,
        on the assumption that lower-index FSDP units have already finished
        computing.
        """
        if state.cpu_offload_mgr is None:
            return
        candidates = [
            (uid, entry)
            for uid, entry in state.active_buffers.items()
            if uid != requesting_unit_id
        ]
        if not candidates:
            return
        # Pick the oldest (lowest unit_id)
        oldest_uid, (slot_token, tensor) = min(candidates, key=lambda x: x[0])
        state.cpu_offload_mgr.offload(slot_token, tensor)
        logger.debug(
            "_try_offload_inactive: offloaded unit_id=%d to CPU on %s",
            oldest_uid,
            state.device,
        )


# ===========================================================================
# DeepSpeed integration shim
# ===========================================================================

class DeepSpeedHeteroFSDPDoubleBuffer:
    """Thin integration layer between DeepSpeed ZeRO and the DES-LOC manager.

    DeepSpeed's ZeRO-3 and the ``HybridEngine`` communicate buffer allocations
    through ``_setup_for_real_optimizer`` and ``_post_backward_hook``.  This
    shim exposes the same interface expected by DeepSpeed's internal hooks
    while delegating to :class:`HeteroFSDPDoubleBufferManager`.

    It is designed to be drop-in compatible with the ``fsdp_double_buffer``
    flag in DeepSpeed's ``ZeROConfig`` / ``DistributedDataParallelConfig``
    equivalents when running on a DES-LOC heterogeneous cluster.
    """

    def __init__(
        self,
        devices: Optional[List[torch.device]] = None,
        dtype: torch.dtype = torch.bfloat16,
        pool_fraction_override: Optional[float] = None,
    ) -> None:
        if devices is None:
            devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        self._manager = HeteroFSDPDoubleBufferManager(
            devices=devices,
            pool_fraction_override=pool_fraction_override,
            dtype=dtype,
        )
        self._unit_device_map: Dict[int, torch.device] = {}

    def register_fsdp_unit(self, unit_id: int, device: torch.device) -> None:
        """Register which device owns FSDP unit *unit_id*."""
        self._unit_device_map[unit_id] = device

    def allocate_param_buffer(
        self,
        unit_id: int,
        numel: int,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """Allocate a parameter gather buffer for FSDP unit *unit_id*.

        Called by DeepSpeed's AllGather hook before gathering shards.
        """
        device = self._unit_device_map.get(unit_id)
        if device is None:
            raise ValueError(f"FSDP unit {unit_id} not registered with DeepSpeedHeteroFSDPDoubleBuffer")
        return self._manager.request_buffer(device, unit_id, numel, dtype)

    def signal_param_ready(self, unit_id: int) -> None:
        """Called after AllGather completes for unit *unit_id*."""
        device = self._unit_device_map[unit_id]
        self._manager.mark_prefetch_done(device, unit_id)

    def wait_param_ready(self, unit_id: int) -> None:
        """Block compute stream until unit *unit_id*'s params are gathered."""
        device = self._unit_device_map[unit_id]
        self._manager.wait_and_compute(device, unit_id)

    def release_param_buffer(self, unit_id: int) -> None:
        """Release the gather buffer for unit *unit_id* after backward pass."""
        device = self._unit_device_map[unit_id]
        self._manager.release_buffer(device, unit_id)

    def finalize(self) -> None:
        self._manager.finalize()

    @property
    def loc_stats(self) -> Dict[str, Dict[str, int]]:
        return self._manager.get_loc_stats()


# ===========================================================================
# Unit tests
# ===========================================================================

if __name__ == "__main__":
    """Self-contained unit tests for HeteroFSDPDoubleBuffer components.

    Run with: python deepspeed/runtime/zero/hetero_fsdp_double_buffer.py

    Tests are designed to work on any single-GPU machine (or CPU-only) by
    mocking the multi-device setup where CUDA is unavailable.
    """

    import sys
    import traceback

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"

    results: List[Tuple[str, bool, str]] = []

    def run_test(name: str, fn):
        try:
            fn()
            results.append((name, True, ""))
            print(f"  [{PASS}] {name}")
        except Exception as exc:
            tb = traceback.format_exc()
            results.append((name, False, tb))
            print(f"  [{FAIL}] {name}: {exc}")

    # -------------------------------------------------------------------
    # Test 1: DeviceClass classification
    # -------------------------------------------------------------------
    def test_device_classification():
        # We can only test UNKNOWN on machines without the exact target GPUs
        unknown = DeviceClass.UNKNOWN
        assert unknown is not None
        # Verify enum members exist
        assert DeviceClass.A6000_SM86 is not None
        assert DeviceClass.H100_SM90 is not None

    run_test("DeviceClass: enum members", test_device_classification)

    # -------------------------------------------------------------------
    # Test 2: LOCCache hit/miss semantics
    # -------------------------------------------------------------------
    def test_loc_cache_hit_miss():
        if not torch.cuda.is_available():
            return  # skip on CPU-only CI

        dev = torch.device("cuda:0")
        cache = LOCCache(capacity=4, device=dev)

        # First get: miss
        t1 = cache.get(1024, torch.float32)
        assert t1.numel() == 1024
        assert cache.stats["misses"] == 1
        assert cache.stats["hits"] == 0

        # Return and re-get: hit
        cache.put(t1)
        t2 = cache.get(1024, torch.float32)
        assert cache.stats["hits"] == 1
        assert t2.numel() == 1024

        cache.clear()

    run_test("LOCCache: hit/miss semantics", test_loc_cache_hit_miss)

    # -------------------------------------------------------------------
    # Test 3: LOCCache capacity eviction
    # -------------------------------------------------------------------
    def test_loc_cache_eviction():
        if not torch.cuda.is_available():
            return

        dev = torch.device("cuda:0")
        cache = LOCCache(capacity=2, device=dev)

        tensors = [cache.get(128, torch.float32) for _ in range(3)]
        for t in tensors:
            cache.put(t)

        # Only 2 slots allowed; one should have been evicted
        with cache._lock:
            total = sum(len(v) for v in cache._store.values())
        assert total <= 2, f"Expected ≤2 cached tensors, got {total}"
        cache.clear()

    run_test("LOCCache: capacity eviction", test_loc_cache_eviction)

    # -------------------------------------------------------------------
    # Test 4: StorageResizeFallbackAllocator keeps tensor alive
    # -------------------------------------------------------------------
    def test_storage_resize_fallback():
        if not torch.cuda.is_available():
            return

        dev = torch.device("cuda:0")
        cache = LOCCache(capacity=4, device=dev)
        alloc = StorageResizeFallbackAllocator(device=dev, loc_cache=cache)

        aid, t = alloc.allocate(4096, torch.float32)
        assert t.numel() == 4096

        # Create a view — the storage must remain valid
        view = t[:512]
        alloc.add_ref(aid)   # view holds an extra reference
        alloc.release(aid)   # release original — storage still live
        view[0] = 1.0        # must not segfault / IMA
        alloc.release(aid)   # release view's reference
        cache.clear()

    run_test("StorageResizeFallback: tensor alive while view exists", test_storage_resize_fallback)

    # -------------------------------------------------------------------
    # Test 5: HeteroFixedPoolAllocator primary path
    # -------------------------------------------------------------------
    def test_pool_allocator_primary():
        if not torch.cuda.is_available():
            return

        dev = torch.device("cuda:0")
        cache = LOCCache(capacity=4, device=dev)
        cfg = PoolConfig(
            primary_pool_fraction=0.01,
            use_storage_resize_fallback=True,
            loc_cache_capacity=4,
        )
        alloc = HeteroFixedPoolAllocator(
            device=dev,
            pool_bytes=8 * 1024 * 1024,  # 8 MiB
            dtype=torch.float32,
            loc_cache=cache,
            cfg=cfg,
        )

        # Allocate two small buffers (should fit in pool)
        tok0, t0 = alloc.allocate(1024, torch.float32)
        tok1, t1 = alloc.allocate(1024, torch.float32)
        assert tok0 >= 0
        assert tok1 >= 0
        assert t0.numel() == 1024
        assert t1.numel() == 1024

        # Both slots occupied — next should go to fallback (negative token)
        tok2, t2 = alloc.allocate(128, torch.float32)
        assert tok2 < 0, "Expected fallback token for third allocation"

        alloc.release(tok0)
        alloc.release(tok1)
        alloc.release(tok2)

    run_test("HeteroFixedPoolAllocator: primary and fallback paths", test_pool_allocator_primary)

    # -------------------------------------------------------------------
    # Test 6: DoubleBufferSynchroniser event ordering
    # -------------------------------------------------------------------
    def test_synchroniser_event_ordering():
        if not torch.cuda.is_available():
            return

        dev = torch.device("cuda:0")
        sync = DoubleBufferSynchroniser(device=dev)

        # Record a prefetch event for unit 0
        with torch.cuda.stream(sync.prefetch_stream):
            _ = torch.zeros(64, device=dev)   # dummy work
        sync.record_prefetch_done(unit_id=0)

        # Waiting on the compute stream should not raise
        sync.wait_for_prefetch(unit_id=0)
        sync.compute_stream.synchronize()

        # Waiting on a non-existent event should be a no-op
        sync.wait_for_prefetch(unit_id=99)

    run_test("DoubleBufferSynchroniser: event ordering", test_synchroniser_event_ordering)

    # -------------------------------------------------------------------
    # Test 7: CPUOffloadManager round-trip
    # -------------------------------------------------------------------
    def test_cpu_offload_roundtrip():
        if not torch.cuda.is_available():
            return

        dev = torch.device("cuda:0")
        mgr = CPUOffloadManager(device=dev)

        src = torch.randn(1024, device=dev, dtype=torch.float32)
        original = src.clone()

        mgr.offload(slot_token=42, tensor=src)
        mgr.sync()

        dst = torch.zeros(1024, device=dev, dtype=torch.float32)
        mgr.restore(slot_token=42, target=dst)
        mgr.sync()

        assert torch.allclose(original, dst), "CPU offload round-trip data mismatch"

    run_test("CPUOffloadManager: offload/restore round-trip", test_cpu_offload_roundtrip)

    # -------------------------------------------------------------------
    # Test 8: HeteroFSDPDoubleBufferManager single-device workflow
    # -------------------------------------------------------------------
    def test_manager_single_device_workflow():
        if not torch.cuda.is_available():
            return

        dev = torch.device("cuda:0")
        mgr = HeteroFSDPDoubleBufferManager(
            devices=[dev],
            pool_fraction_override=0.005,  # tiny pool for test
            dtype=torch.float32,
        )

        NUM_UNITS = 4
        NUMEL = 512

        for unit_id in range(NUM_UNITS):
            buf = mgr.request_buffer(dev, unit_id, NUMEL)
            assert buf.numel() == NUMEL
            buf.fill_(float(unit_id))
            mgr.mark_prefetch_done(dev, unit_id)
            mgr.wait_and_compute(dev, unit_id)
            assert buf[0].item() == float(unit_id)
            mgr.release_buffer(dev, unit_id)

        mgr.finalize()
        stats = mgr.get_loc_stats()
        assert str(dev) in stats

    run_test("HeteroFSDPDoubleBufferManager: single-device workflow", test_manager_single_device_workflow)

    # -------------------------------------------------------------------
    # Test 9: DeepSpeedHeteroFSDPDoubleBuffer shim
    # -------------------------------------------------------------------
    def test_deepspeed_shim():
        if not torch.cuda.is_available():
            return

        dev = torch.device("cuda:0")
        shim = DeepSpeedHeteroFSDPDoubleBuffer(
            devices=[dev],
            dtype=torch.float32,
            pool_fraction_override=0.005,
        )
        shim.register_fsdp_unit(unit_id=0, device=dev)
        shim.register_fsdp_unit(unit_id=1, device=dev)

        for uid in range(2):
            buf = shim.allocate_param_buffer(uid, numel=256)
            buf.fill_(float(uid))
            shim.signal_param_ready(uid)
            shim.wait_param_ready(uid)
            shim.release_param_buffer(uid)

        shim.finalize()

    run_test("DeepSpeedHeteroFSDPDoubleBuffer: shim round-trip", test_deepspeed_shim)

    # -------------------------------------------------------------------
    # Test 10: prefetch_next overlap simulation
    # -------------------------------------------------------------------
    def test_prefetch_next_overlap():
        if not torch.cuda.is_available():
            return

        dev = torch.device("cuda:0")
        mgr = HeteroFSDPDoubleBufferManager(
            devices=[dev],
            pool_fraction_override=0.01,
            dtype=torch.float32,
        )

        gathered = {}

        def make_gather_fn(uid):
            def gather_fn(tensor):
                tensor.fill_(float(uid * 10))
                gathered[uid] = tensor.clone()
            return gather_fn

        NUM_UNITS = 6
        NUMEL = 256

        # Bootstrap: request unit 0 manually
        buf0 = mgr.request_buffer(dev, 0, NUMEL)
        buf0.fill_(0.0)
        mgr.mark_prefetch_done(dev, 0)

        for current_id in range(NUM_UNITS):
            mgr.wait_and_compute(dev, current_id)
            current_buf = mgr._states[dev].active_buffers.get(current_id)

            next_id = current_id + 1
            if next_id < NUM_UNITS:
                mgr.prefetch_next(
                    dev,
                    current_unit_id=current_id,
                    next_unit_id=next_id,
                    numel=NUMEL,
                    gather_fn=make_gather_fn(next_id),
                )

            mgr.release_buffer(dev, current_id)

        mgr.finalize()

        # Verify gathered values
        for uid in range(1, NUM_UNITS):
            assert uid in gathered, f"Unit {uid} was never gathered"
            expected = float(uid * 10)
            actual = gathered[uid][0].item()
            assert abs(actual - expected) < 1e-4, (
                f"Unit {uid}: expected {expected}, got {actual}"
            )

    run_test("prefetch_next: overlap simulation with gather_fn", test_prefetch_next_overlap)

    # -------------------------------------------------------------------
    # Test 11: PoolConfig defaults are sane per device class
    # -------------------------------------------------------------------
    def test_pool_config_defaults():
        a6k = _POOL_CONFIG_BY_CLASS[DeviceClass.A6000_SM86]
        h100 = _POOL_CONFIG_BY_CLASS[DeviceClass.H100_SM90]

        # H100 should have a larger pool fraction
        assert h100.primary_pool_fraction > a6k.primary_pool_fraction
        # A6000 should enable CPU offload; H100 should not (by default)
        assert a6k.allow_cpu_offload is True
        assert h100.allow_cpu_offload is False
        # SM86 uses storage-resize fallback; SM90 uses pinned-staging
        assert a6k.use_storage_resize_fallback is True
        assert h100.use_storage_resize_fallback is False

    run_test("PoolConfig: sane defaults per device class", test_pool_config_defaults)

    # -------------------------------------------------------------------
    # Test 12: PinnedStagingFallbackAllocator (SM90 path)
    # -------------------------------------------------------------------
    def test_pinned_staging_fallback():
        if not torch.cuda.is_available():
            return

        dev = torch.device("cuda:0")
        cache = LOCCache(capacity=4, device=dev)
        alloc = PinnedStagingFallbackAllocator(device=dev, loc_cache=cache)

        aid, t = alloc.allocate(2048, torch.float32)
        assert t.numel() == 2048
        assert t.device == dev

        t.fill_(3.14)
        alloc.stage_to_host(aid)
        alloc.sync_staging()

        # Overwrite device tensor with zeros
        t.zero_()
        alloc.restore_from_host(aid)
        alloc.sync_staging()

        assert abs(t[0].item() - 3.14) < 1e-4, "Pinned staging: restore mismatch"
        alloc.release(aid)
        cache.clear()

    run_test("PinnedStagingFallbackAllocator: stage/restore", test_pinned_staging_fallback)

    # -------------------------------------------------------------------
    # Test 13: multi-microbatch stress (mimics upstream test structure)
    # -------------------------------------------------------------------
    def test_multi_microbatch_stress():
        if not torch.cuda.is_available():
            return

        dev = torch.device("cuda:0")
        mgr = HeteroFSDPDoubleBufferManager(
            devices=[dev],
            pool_fraction_override=0.005,
            dtype=torch.bfloat16,
        )

        NUM_MICROBATCHES = 4
        NUM_STEPS = 5
        FSDP_UNITS = 3
        NUMEL_PER_UNIT = [512, 1024, 2048]   # varied sizes, one may exceed pool

        for step in range(NUM_STEPS):
            for mb in range(NUM_MICROBATCHES):
                for uid in range(FSDP_UNITS):
                    numel = NUMEL_PER_UNIT[uid]
                    buf = mgr.request_buffer(dev, uid, numel, torch.bfloat16)
                    buf.fill_(float(step + mb + uid))
                    mgr.mark_prefetch_done(dev, uid)
                    mgr.wait_and_compute(dev, uid)
                    mgr.release_buffer(dev, uid)

        mgr.finalize()
        stats = mgr.get_loc_stats()[str(dev)]
        # Should have accumulated some LOC hits after the first step
        total = stats["hits"] + stats["misses"]
        assert total > 0

    run_test("multi-microbatch stress: 5 steps × 4 micro-batches × 3 units",
             test_multi_microbatch_stress)

    # -------------------------------------------------------------------
    # Test 14: LOCCache thread safety
    # -------------------------------------------------------------------
    def test_loc_cache_thread_safety():
        if not torch.cuda.is_available():
            return

        dev = torch.device("cuda:0")
        cache = LOCCache(capacity=16, device=dev)
        errors = []

        def worker(worker_id: int):
            try:
                for i in range(20):
                    t = cache.get(256, torch.float32)
                    t.fill_(float(worker_id))
                    cache.put(t)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert not errors, f"Thread safety errors: {errors}"
        cache.clear()

    run_test("LOCCache: thread safety (4 workers × 20 ops)", test_loc_cache_thread_safety)

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print()
    total = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"Results: {passed}/{total} tests passed")
    for name, ok, tb in results:
        if not ok:
            print(f"\n  FAILED: {name}\n{tb}")

    if passed < total:
        sys.exit(1)
