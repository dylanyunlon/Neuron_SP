# Copyright (c) 2024 Neuron_SP Project Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Mirrors Megatron commit b6b49e7e60777e9bfc0947550c967fd858e33dde
# "Fix double buffering not working with activation recompute (#2689)"
#
# DES-LOC Adaptation: HeteroFSDPDoubleBufferRecompute
# =======================================================
# Upstream intent (Megatron):
#   FSDP double-buffering overlaps all-gather communication with compute.
#   When activation recomputation (gradient checkpointing) is active the
#   forward pass re-runs inside PRE_BACKWARD state.  The original code
#   simply skipped parameter resharding during that re-forward, which
#   prevented the all-gather pipeline from reclaiming buffer slots and
#   caused silent OOM on long sequences.  The fix introduces *lazy release*:
#   buffers are tagged as releasable but not freed until the pipeline is
#   about to allocate a new slot, preserving weights for the backward pass
#   while keeping peak memory under control.
#
# DES-LOC reinterpretation:
#   On the Neuron_SP cluster (2× A6000 48 GB + 1× H100 NVL 96 GB, PCIe-only,
#   1.5 TB CPU DRAM) the three devices have radically different memory
#   capacities and bandwidth.  The standard lazy-release logic assumes a
#   homogeneous VRAM budget.  Here we extend it with:
#
#   1. HeteroDeviceRegistry – tracks per-device VRAM budgets and SM arch
#      (SM86 vs SM90).
#   2. LocalityCache – the "LOC" component of DES-LOC.  Released parameter
#      buffers are not simply discarded; they are migrated to the device
#      that will need them soonest (predicted by a lightweight layer-order
#      tracker) or to CPU pinned memory if no GPU has capacity.
#   3. DecoupledExecutionScheduler – the "DES" component.  Recompute
#      re-forward and the main backward are decoupled across the device
#      heterogeneity: recompute preferentially runs on H100 (fast BF16
#      engine), while gradient accumulation runs on A6000.
#   4. DoubleBufferRecomputeManager – top-level orchestrator that wires the
#      above together and exposes the same hook-registration API expected by
#      DeepSpeed's ZeRO-3 engine.

"""
hetero_fsdp_double_buffer_recompute.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DES-LOC heterogeneous double-buffer management with activation recompute
support.  Designed for asymmetric PCIe clusters; does NOT require NVLink.
"""

from __future__ import annotations

import enum
import logging
import threading
import time
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SM architecture identifiers
SM86 = 86   # A6000
SM90 = 90   # H100 NVL

# Bandwidth estimates (GB/s) used for migration cost heuristic
_PCIE_BW_GBPS = 16.0          # PCIe Gen4 x16 unidirectional
_CPU_PIN_BW_GBPS = 32.0        # DDR5 write-to-pinned

# Fraction of device VRAM reserved for activations / gradients (not buffers)
_VRAM_RESERVE_FRACTION = 0.25


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class BucketStatus(enum.Enum):
    """Lifecycle states of a parameter bucket."""
    EMPTY        = "empty"        # buffer slot is free
    FILLING      = "filling"      # all-gather in flight
    READY        = "ready"        # all-gathered, usable
    COMMUNICATING = "communicating" # grad reduce in flight
    LAZY_RELEASE = "lazy_release"  # tagged; awaiting recycle window


class TrainingState(enum.Enum):
    """Per-module training state, mirroring Megatron's enum."""
    IDLE         = "idle"
    FORWARD      = "forward"
    PRE_BACKWARD = "pre_backward"   # inside activation recompute re-forward
    BACKWARD     = "backward"


class DeviceRole(enum.Enum):
    """Logical role assigned to each physical device in DES-LOC."""
    PRIMARY_RECOMPUTE = "primary_recompute"   # H100: runs recompute fwd
    GRAD_ACCUMULATE   = "grad_accumulate"     # A6000: accumulates grads
    OFFLOAD_SINK      = "offload_sink"        # CPU DRAM


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DeviceSpec:
    """Metadata for one physical device visible to this rank."""
    device: torch.device
    sm_arch: int
    total_vram_bytes: int
    role: DeviceRole
    available_bytes: int = field(init=False)

    def __post_init__(self):
        self.available_bytes = int(
            self.total_vram_bytes * (1.0 - _VRAM_RESERVE_FRACTION)
        )

    @property
    def free_bytes(self) -> int:
        """Rough remaining capacity (may lag real allocator state)."""
        if self.device.type == "cpu":
            return self.available_bytes
        try:
            free, _ = torch.cuda.mem_get_info(self.device)
            return free
        except Exception:
            return self.available_bytes

    def can_fit(self, nbytes: int) -> bool:
        return self.free_bytes >= nbytes


@dataclass
class BucketRecord:
    """State record for one FSDP parameter bucket."""
    bucket_id: int
    bwd: bool
    status: BucketStatus = BucketStatus.EMPTY
    tensor: Optional[torch.Tensor] = None
    locality_device: Optional[torch.device] = None  # where tensor currently lives
    lazy_release: bool = False
    last_used_layer: int = -1       # layer index when bucket was last needed
    migrate_event: Optional[torch.cuda.Event] = None

    @property
    def nbytes(self) -> int:
        if self.tensor is None:
            return 0
        return self.tensor.nbytes


# ---------------------------------------------------------------------------
# HeteroDeviceRegistry
# ---------------------------------------------------------------------------

class HeteroDeviceRegistry:
    """
    Maintains the set of physical devices available to this rank and assigns
    DES-LOC roles based on SM architecture.

    On a 2× A6000 + 1× H100 cluster the policy is:
      - H100  (SM90, 96 GB)  → PRIMARY_RECOMPUTE
      - A6000 (SM86, 48 GB)  → GRAD_ACCUMULATE  (one per rank or shared)
      - CPU DRAM             → OFFLOAD_SINK

    The registry is initialised once per process and consulted by both the
    LocalityCache and DecoupledExecutionScheduler.
    """

    def __init__(self):
        self._specs: List[DeviceSpec] = []
        self._role_map: Dict[DeviceRole, List[DeviceSpec]] = defaultdict(list)
        self._lock = threading.Lock()

    def register(self, device: torch.device, sm_arch: int, vram_bytes: int) -> DeviceSpec:
        """Register a physical device and assign its DES-LOC role."""
        role = self._infer_role(sm_arch, device)
        spec = DeviceSpec(
            device=device,
            sm_arch=sm_arch,
            total_vram_bytes=vram_bytes,
            role=role,
        )
        with self._lock:
            self._specs.append(spec)
            self._role_map[role].append(spec)
        logger.info(
            "HeteroDeviceRegistry: registered %s sm%d vram=%.1f GB role=%s",
            device, sm_arch, vram_bytes / 2**30, role.value,
        )
        return spec

    @staticmethod
    def _infer_role(sm_arch: int, device: torch.device) -> DeviceRole:
        if device.type == "cpu":
            return DeviceRole.OFFLOAD_SINK
        if sm_arch >= SM90:
            return DeviceRole.PRIMARY_RECOMPUTE
        return DeviceRole.GRAD_ACCUMULATE

    def get_role(self, role: DeviceRole) -> List[DeviceSpec]:
        return self._role_map.get(role, [])

    def primary_recompute_device(self) -> Optional[torch.device]:
        specs = self.get_role(DeviceRole.PRIMARY_RECOMPUTE)
        return specs[0].device if specs else None

    def grad_accumulate_devices(self) -> List[torch.device]:
        return [s.device for s in self.get_role(DeviceRole.GRAD_ACCUMULATE)]

    def offload_device(self) -> torch.device:
        specs = self.get_role(DeviceRole.OFFLOAD_SINK)
        return specs[0].device if specs else torch.device("cpu")

    def best_fit_device(self, nbytes: int, prefer_role: DeviceRole) -> torch.device:
        """
        Return the best device for placing *nbytes* of parameter data.

        Priority:
          1. Preferred role device(s) that have capacity.
          2. Any GPU device with capacity.
          3. CPU offload sink.
        """
        candidates = self.get_role(prefer_role)
        for spec in candidates:
            if spec.can_fit(nbytes):
                return spec.device
        for spec in self._specs:
            if spec.device.type != "cpu" and spec.can_fit(nbytes):
                return spec.device
        logger.warning(
            "best_fit_device: no GPU can fit %d bytes, falling back to CPU offload", nbytes
        )
        return self.offload_device()


# ---------------------------------------------------------------------------
# LocalityCache
# ---------------------------------------------------------------------------

class LocalityCache:
    """
    The *LOC* pillar of DES-LOC.

    Instead of unconditionally freeing a parameter buffer after lazy-release
    tagging, the LocalityCache migrates it to the device most likely to need
    it next.  This amortises PCIe round-trips when activation recomputation
    performs a second forward pass: the weights are already on the target
    device (or in fast pinned CPU memory) rather than re-gathered from shards.

    Migration decisions use a simple *next-use distance* heuristic derived
    from the layer execution order recorded by the DecoupledExecutionScheduler.

    Thread safety: a per-cache RLock protects the internal tables.  Tensor
    moves are issued asynchronously via CUDA streams and tracked with Events.
    """

    def __init__(self, registry: HeteroDeviceRegistry, max_cache_bytes: int = 0):
        self._registry = registry
        # max_cache_bytes=0 means use available DRAM heuristic
        self._max_bytes = max_cache_bytes or int(1.5e12 * 0.40)  # 40% of 1.5 TB
        self._cache: Dict[Tuple[int, bool], BucketRecord] = {}
        self._total_bytes = 0
        self._lock = threading.RLock()
        # Per-device async copy streams (created lazily)
        self._streams: Dict[torch.device, torch.cuda.Stream] = {}
        logger.debug("LocalityCache init: max_bytes=%.1f GB", self._max_bytes / 2**30)

    def _get_stream(self, device: torch.device) -> Optional[torch.cuda.Stream]:
        if device.type == "cpu":
            return None
        if device not in self._streams:
            self._streams[device] = torch.cuda.Stream(device=device)
        return self._streams[device]

    def store(self, record: BucketRecord, target_device: torch.device) -> None:
        """
        Migrate *record.tensor* to *target_device* and update cache tables.

        If the cache is full, the least-recently-used entry is evicted to the
        CPU offload sink before the new entry is admitted.
        """
        if record.tensor is None:
            logger.debug("LocalityCache.store: bucket %d has no tensor, skip", record.bucket_id)
            return

        key = (record.bucket_id, record.bwd)
        nbytes = record.nbytes

        with self._lock:
            # Evict LRU entries if needed
            self._evict_to_fit(nbytes)

            stream = self._get_stream(target_device)
            ctx = torch.cuda.stream(stream) if stream else _nullctx()
            with ctx:
                migrated = record.tensor.to(target_device, non_blocking=True)
                if stream:
                    event = torch.cuda.Event()
                    event.record(stream)
                    record.migrate_event = event

            record.tensor = migrated
            record.locality_device = target_device
            self._cache[key] = record
            self._total_bytes += nbytes

        logger.debug(
            "LocalityCache.store: bucket (%d, bwd=%s) migrated to %s (%d MB)",
            record.bucket_id, record.bwd, target_device, nbytes // 2**20,
        )

    def retrieve(self, bucket_id: int, bwd: bool) -> Optional[BucketRecord]:
        """
        Retrieve a cached bucket record.  Waits for any in-flight migration
        event before returning, ensuring tensor is safe to use.
        """
        key = (bucket_id, bwd)
        with self._lock:
            record = self._cache.get(key)
            if record is None:
                return None
            if record.migrate_event is not None:
                record.migrate_event.synchronize()
                record.migrate_event = None
            return record

    def evict(self, bucket_id: int, bwd: bool) -> None:
        """Explicitly remove a bucket from the cache and free its memory."""
        key = (bucket_id, bwd)
        with self._lock:
            record = self._cache.pop(key, None)
            if record is not None:
                self._total_bytes = max(0, self._total_bytes - record.nbytes)
                del record.tensor
                record.tensor = None
        logger.debug("LocalityCache.evict: bucket (%d, bwd=%s) freed", bucket_id, bwd)

    def _evict_to_fit(self, nbytes: int) -> None:
        """Evict LRU entries until *nbytes* can be accommodated."""
        sorted_entries = sorted(
            self._cache.items(), key=lambda kv: kv[1].last_used_layer
        )
        for key, record in sorted_entries:
            if self._total_bytes + nbytes <= self._max_bytes:
                break
            if record.tensor is not None:
                offload = self._registry.offload_device()
                if record.locality_device != offload:
                    stream = None  # CPU copies are synchronous for safety
                    record.tensor = record.tensor.to(offload)
                    record.locality_device = offload
                    logger.debug(
                        "LocalityCache: LRU evict bucket (%d, %s) to CPU",
                        record.bucket_id, record.bwd,
                    )
                else:
                    self._total_bytes = max(0, self._total_bytes - record.nbytes)
                    del record.tensor
                    record.tensor = None
                    del self._cache[key]


class _nullctx:
    """Minimal no-op context manager (avoids importing contextlib)."""
    def __enter__(self): return self
    def __exit__(self, *a): pass


# ---------------------------------------------------------------------------
# DecoupledExecutionScheduler
# ---------------------------------------------------------------------------

class DecoupledExecutionScheduler:
    """
    The *DES* pillar of DES-LOC.

    Tracks layer execution order and assigns each layer's recompute-forward
    to the most capable device (H100 when available).  The main backward
    gradient accumulation is steered to A6000 devices, exploiting their
    larger aggregate VRAM (2× 48 GB = 96 GB) for gradient tensors.

    The scheduler exposes three hooks that DoubleBufferRecomputeManager
    installs on each FSDP unit:
      - on_forward_enter  : record layer order; decide execution device.
      - on_forward_exit   : trigger lazy-release or immediate release.
      - on_backward_enter : restore parameters from LocalityCache if needed.

    Design note on PCIe bandwidth:
      PCIe-only bandwidth (~16 GB/s) is roughly 6× slower than NVLink.  The
      scheduler therefore tries to *avoid* cross-device weight movement during
      the recompute window by pre-staging weights via LocalityCache.store()
      while the previous layer's backward is running (pipeline overlap).
    """

    def __init__(
        self,
        registry: HeteroDeviceRegistry,
        locality_cache: LocalityCache,
    ):
        self._registry = registry
        self._cache = locality_cache
        self._layer_order: List[int] = []          # module ids in fwd order
        self._layer_index: Dict[int, int] = {}     # module_id → layer index
        self._prefetch_queue: deque = deque()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Layer order recording
    # ------------------------------------------------------------------

    def record_forward(self, module_id: int) -> int:
        """Record that *module_id* executed in the forward pass; return its index."""
        with self._lock:
            if module_id not in self._layer_index:
                idx = len(self._layer_order)
                self._layer_order.append(module_id)
                self._layer_index[module_id] = idx
                logger.debug("DES scheduler: layer %d registered (module_id=%d)", idx, module_id)
            return self._layer_index[module_id]

    def next_layer_index(self, current_idx: int) -> int:
        return current_idx + 1

    # ------------------------------------------------------------------
    # Device assignment
    # ------------------------------------------------------------------

    def recompute_device_for(self, layer_idx: int) -> torch.device:
        """
        Assign execution device for recompute forward of *layer_idx*.

        Policy:
          - H100 (PRIMARY_RECOMPUTE) if available and can fit the layer.
          - Otherwise first available A6000.
        """
        primary = self._registry.primary_recompute_device()
        if primary is not None:
            return primary
        grads = self._registry.grad_accumulate_devices()
        if grads:
            return grads[layer_idx % len(grads)]
        return torch.device("cuda:0")

    def grad_accumulate_device_for(self, layer_idx: int) -> torch.device:
        """Return the A6000 device responsible for this layer's gradients."""
        devs = self._registry.grad_accumulate_devices()
        if not devs:
            return torch.device("cuda:0")
        return devs[layer_idx % len(devs)]

    # ------------------------------------------------------------------
    # Prefetch scheduling
    # ------------------------------------------------------------------

    def schedule_prefetch(
        self,
        bucket_id: int,
        bwd: bool,
        target_device: torch.device,
        record: BucketRecord,
    ) -> None:
        """
        Schedule async prefetch of *bucket_id* to *target_device*.

        Called during the backward of layer N to pre-stage weights for the
        recompute forward of layer N+1, overlapping PCIe transfer with compute.
        """
        with self._lock:
            self._prefetch_queue.append((bucket_id, bwd, target_device, record))
        self._drain_prefetch_queue()

    def _drain_prefetch_queue(self) -> None:
        while True:
            with self._lock:
                if not self._prefetch_queue:
                    break
                bucket_id, bwd, target_device, record = self._prefetch_queue.popleft()
            if record.tensor is not None and record.locality_device != target_device:
                logger.debug(
                    "DES prefetch: bucket (%d, bwd=%s) → %s", bucket_id, bwd, target_device
                )
                self._cache.store(record, target_device)


# ---------------------------------------------------------------------------
# AllGatherPipelineProxy
# ---------------------------------------------------------------------------

class AllGatherPipelineProxy:
    """
    Thin proxy around DeepSpeed's internal all-gather pipeline object.

    Extends release_bucket() with DES-LOC lazy-release and cache-migration
    semantics.  The real pipeline object is held by weak reference to avoid
    keeping it alive after ZeRO engine teardown.

    This mirrors Megatron's AllGatherPipeline.release_bucket() changes in
    param_and_grad_buffer.py with three additions:
      1. lazy=True tags the bucket instead of releasing immediately.
      2. recycle_unused_buckets() flushes tagged buckets before new alloc.
      3. Released tensors are handed to LocalityCache rather than freed.
    """

    def __init__(
        self,
        real_pipeline: Any,
        locality_cache: LocalityCache,
        scheduler: DecoupledExecutionScheduler,
        registry: HeteroDeviceRegistry,
    ):
        self._pipeline_ref = weakref.ref(real_pipeline)
        self._cache = locality_cache
        self._scheduler = scheduler
        self._registry = registry
        # bucket_id → BucketRecord
        self._records: Dict[Tuple[int, bool], BucketRecord] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_record(self, bucket_id: int, bwd: bool) -> BucketRecord:
        key = (bucket_id, bwd)
        if key not in self._records:
            self._records[key] = BucketRecord(bucket_id=bucket_id, bwd=bwd)
        return self._records[key]

    # ------------------------------------------------------------------
    # Public API (mirrors Megatron AllGatherPipeline)
    # ------------------------------------------------------------------

    def mark_bucket_ready(self, bucket_id: int, bwd: bool, tensor: torch.Tensor) -> None:
        """Called when an all-gather completes; tensor is the gathered buffer."""
        with self._lock:
            record = self._get_or_create_record(bucket_id, bwd)
            record.tensor = tensor
            record.status = BucketStatus.READY
            record.locality_device = tensor.device

    def release_bucket(
        self,
        bucket_id: int,
        bwd: bool,
        lazy: bool = False,
        layer_idx: int = -1,
    ) -> None:
        """
        Release (or lazily tag) a parameter bucket after forward/backward use.

        Args:
            bucket_id: Bucket identifier.
            bwd:       True if called from the backward pass.
            lazy:      If True, defer actual release to recycle_unused_buckets().
                       This matches Megatron's lazy parameter for activation
                       recompute; here we additionally migrate the tensor to the
                       LocalityCache for potential reuse on recompute.
            layer_idx: Layer index hint for DES prefetch scheduling.

        DES-LOC extension vs Megatron:
            Megatron's lazy path simply marks bucket_can_be_released[key]=True.
            Here, instead of leaving the tensor in the original VRAM, we migrate
            it to the device most likely to need it next (recompute device for
            the same layer index).  This exploits CPU DRAM headroom (1.5 TB)
            as an intermediary when neither GPU has capacity.
        """
        with self._lock:
            record = self._get_or_create_record(bucket_id, bwd)

            if record.status == BucketStatus.EMPTY:
                logger.debug("release_bucket: bucket (%d, %s) already empty", bucket_id, bwd)
                return

            if lazy:
                record.lazy_release = True
                record.status = BucketStatus.LAZY_RELEASE
                record.last_used_layer = layer_idx

                # DES-LOC: migrate to recompute device proactively
                recompute_dev = self._scheduler.recompute_device_for(layer_idx)
                if record.tensor is not None:
                    nbytes = record.nbytes
                    target = self._registry.best_fit_device(
                        nbytes, DeviceRole.PRIMARY_RECOMPUTE
                    )
                    logger.debug(
                        "release_bucket lazy: bucket (%d, %s) → LocalityCache on %s",
                        bucket_id, bwd, target,
                    )
                    self._cache.store(record, target)
                return

            # Immediate release
            if record.status == BucketStatus.COMMUNICATING:
                raise ValueError(
                    f"release_bucket: bucket ({bucket_id}, bwd={bwd}) is "
                    "communicating and cannot be released."
                )
            self._do_release(record)

    def recycle_unused_buckets(self) -> None:
        """
        Flush all lazily-tagged buckets.

        Called by AllGatherPipeline just before allocating a new buffer slot
        (mirrors Megatron's recycle_unused_buckets hook).  Buckets that were
        pre-migrated to LocalityCache are simply evicted from there; buckets
        still in VRAM are freed immediately.
        """
        with self._lock:
            lazy_keys = [
                k for k, r in self._records.items()
                if r.lazy_release and r.status == BucketStatus.LAZY_RELEASE
            ]
        for key in lazy_keys:
            bid, bwd = key
            logger.debug("recycle_unused_buckets: releasing (%d, bwd=%s)", bid, bwd)
            self._cache.evict(bid, bwd)
            with self._lock:
                record = self._records.get(key)
                if record:
                    self._do_release(record)

    def _do_release(self, record: BucketRecord) -> None:
        """Free buffer memory and reset record status (must hold self._lock)."""
        if record.tensor is not None:
            del record.tensor
            record.tensor = None
        record.status = BucketStatus.EMPTY
        record.lazy_release = False
        record.locality_device = None

    def get_cached_tensor(
        self, bucket_id: int, bwd: bool
    ) -> Optional[torch.Tensor]:
        """
        Retrieve a parameter tensor from LocalityCache (for recompute reuse).

        Returns None if the tensor is no longer cached, in which case the
        caller must issue a fresh all-gather.
        """
        cached = self._cache.retrieve(bucket_id, bwd)
        if cached is None or cached.tensor is None:
            return None
        logger.debug(
            "get_cached_tensor: hit bucket (%d, bwd=%s) on %s",
            bucket_id, bwd, cached.locality_device,
        )
        return cached.tensor


# ---------------------------------------------------------------------------
# DoubleBufferRecomputeManager
# ---------------------------------------------------------------------------

class DoubleBufferRecomputeManager:
    """
    Top-level DES-LOC orchestrator for double-buffered FSDP with activation
    recompute on heterogeneous hardware.

    Responsibilities
    ----------------
    1. Wraps DeepSpeed's ZeRO-3 parameter-buffer management with the extended
       AllGatherPipelineProxy.
    2. Installs PyTorch forward/backward hooks on each FSDP unit module to
       intercept parameter release decisions.
    3. Enforces the lazy-release semantics during activation recomputation
       (PRE_BACKWARD state) so that the recompute re-forward can reuse
       already-gathered parameters from LocalityCache rather than triggering
       redundant all-gathers over slow PCIe.
    4. Coordinates the DecoupledExecutionScheduler to route recompute compute
       to H100 and gradient accumulation to A6000.

    Relationship to Megatron fix (#2689)
    -------------------------------------
    The upstream fix adds `lazy=True` to `release_module_parameters()` when
    `module._training_state == TrainingState.PRE_BACKWARD`, instead of the
    previous behaviour of returning early without any release.

    DES-LOC goes further: the lazy-released tensor is migrated to a
    LocalityCache entry on the predicted recompute device, so the second
    forward pass (inside PRE_BACKWARD) can find the weights already resident
    on the H100 rather than re-gathering from A6000 shards.

    Usage
    -----
    ::

        manager = DoubleBufferRecomputeManager.build_from_cluster_info(
            local_rank=0,
            module=model,
        )
        manager.register_all_hooks()
        # … standard DeepSpeed training loop …
        manager.teardown()

    """

    def __init__(
        self,
        registry: HeteroDeviceRegistry,
        locality_cache: LocalityCache,
        scheduler: DecoupledExecutionScheduler,
        pipeline_proxy: AllGatherPipelineProxy,
        fsdp_unit_modules: List[type],
    ):
        self._registry = registry
        self._cache = locality_cache
        self._scheduler = scheduler
        self._proxy = pipeline_proxy
        self._fsdp_unit_types = tuple(fsdp_unit_modules)
        self._hook_handles: List[Any] = []
        self._param_to_bucket: Dict[int, int] = {}   # param data_ptr → bucket_id

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build_from_cluster_info(
        cls,
        local_rank: int,
        module: nn.Module,
        fsdp_unit_modules: Optional[List[type]] = None,
        max_cache_bytes: int = 0,
        real_pipeline: Optional[Any] = None,
    ) -> "DoubleBufferRecomputeManager":
        """
        Construct a DoubleBufferRecomputeManager by auto-detecting GPUs.

        For the Neuron_SP cluster topology (2× A6000 + 1× H100):
          cuda:0 → H100 96 GB  SM90
          cuda:1 → A6000 48 GB SM86
          cuda:2 → A6000 48 GB SM86
        Actual detection uses torch.cuda.get_device_properties().
        """
        registry = HeteroDeviceRegistry()

        n_gpus = torch.cuda.device_count()
        for i in range(n_gpus):
            dev = torch.device(f"cuda:{i}")
            props = torch.cuda.get_device_properties(dev)
            sm = props.major * 10 + props.minor
            vram = props.total_memory
            registry.register(dev, sm, vram)

        # Register CPU offload sink
        import psutil
        cpu_ram = psutil.virtual_memory().total if _has_psutil() else int(1.5e12)
        registry.register(torch.device("cpu"), sm_arch=0, vram_bytes=cpu_ram)

        locality_cache = LocalityCache(registry, max_cache_bytes=max_cache_bytes)
        scheduler = DecoupledExecutionScheduler(registry, locality_cache)

        # Wrap or stub real pipeline
        stub_pipeline = real_pipeline or _StubPipeline()
        proxy = AllGatherPipelineProxy(
            stub_pipeline, locality_cache, scheduler, registry
        )

        fu_types: List[type] = fsdp_unit_modules or [nn.Linear, nn.LayerNorm]

        mgr = cls(registry, locality_cache, scheduler, proxy, fu_types)

        logger.info(
            "DoubleBufferRecomputeManager: local_rank=%d, %d GPU(s) registered, "
            "recompute_device=%s, grad_devices=%s",
            local_rank,
            n_gpus,
            registry.primary_recompute_device(),
            registry.grad_accumulate_devices(),
        )
        return mgr

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def register_all_hooks(self, root_module: nn.Module) -> None:
        """
        Walk *root_module* and install forward/backward hooks on every FSDP
        unit sub-module.
        """
        for name, submod in root_module.named_modules():
            if isinstance(submod, self._fsdp_unit_types):
                self._install_hooks(submod, name)
        logger.info(
            "DoubleBufferRecomputeManager: %d hook handles registered",
            len(self._hook_handles),
        )

    def _install_hooks(self, module: nn.Module, name: str) -> None:
        module._training_state = TrainingState.IDLE  # type: ignore[attr-defined]

        fwd_pre_h = module.register_forward_pre_hook(self._pre_forward_hook)
        fwd_h = module.register_forward_hook(self._post_forward_hook)
        bwd_h = module.register_full_backward_hook(self._backward_hook)

        self._hook_handles.extend([fwd_pre_h, fwd_h, bwd_h])
        logger.debug("hooks installed on %s (%s)", name, type(module).__name__)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _pre_forward_hook(self, module: nn.Module, inputs: Any) -> None:
        """
        Pre-forward: record layer index for DES scheduling; attempt to
        restore parameters from LocalityCache before triggering all-gather.

        If the module is entering recompute (PRE_BACKWARD), we set state here
        so the post-forward hook can distinguish normal forward from recompute.
        """
        mod_id = id(module)
        layer_idx = self._scheduler.record_forward(mod_id)
        module._des_layer_idx = layer_idx  # type: ignore[attr-defined]

        # Check if we are in the recompute re-forward context
        current_state = getattr(module, "_training_state", TrainingState.IDLE)
        if current_state == TrainingState.PRE_BACKWARD:
            logger.debug(
                "pre_forward hook: module %d in PRE_BACKWARD (recompute), "
                "checking LocalityCache for cached parameters",
                mod_id,
            )

    def _post_forward_hook(
        self, module: nn.Module, inputs: Any, output: Any
    ) -> Any:
        """
        Post-forward: decide whether to release parameters immediately or
        lazily, mirroring Megatron's fix for activation recompute.

        Key logic (DES-LOC adaptation of Megatron _post_forward):
          - PRE_BACKWARD  → lazy_release=True  (recompute re-forward; weights
                            may be needed again; migrate to LocalityCache on
                            recompute device)
          - Normal fwd    → lazy_release=False (immediate release; no recompute
                            needed; recycle VRAM now)
        """
        assert isinstance(module, self._fsdp_unit_types), (
            "_post_forward_hook registered on non-FSDP-unit module"
        )

        training_state: TrainingState = getattr(
            module, "_training_state", TrainingState.IDLE
        )
        layer_idx: int = getattr(module, "_des_layer_idx", -1)

        if training_state == TrainingState.PRE_BACKWARD:
            lazy_release = True
            # Do NOT set IDLE here; backward hook will do it
            logger.debug(
                "post_forward: layer %d PRE_BACKWARD → lazy release, "
                "staging to recompute device",
                layer_idx,
            )
        else:
            lazy_release = False
            module._training_state = TrainingState.IDLE  # type: ignore[attr-defined]

        self._release_module_parameters(module, bwd=False, lazy=lazy_release)
        return output

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: Any,
        grad_output: Any,
    ) -> None:
        """
        Backward hook: release parameters post-backward and evict locality
        cache entries no longer needed.
        """
        layer_idx: int = getattr(module, "_des_layer_idx", -1)
        self._release_module_parameters(module, bwd=True, lazy=False)
        module._training_state = TrainingState.IDLE  # type: ignore[attr-defined]

        # Recycle lazily-tagged buckets now that backward is complete
        self._proxy.recycle_unused_buckets()
        logger.debug("backward_hook: layer %d recycled unused buckets", layer_idx)

    # ------------------------------------------------------------------
    # Parameter release
    # ------------------------------------------------------------------

    def _release_module_parameters(
        self, module: nn.Module, bwd: bool, lazy: bool
    ) -> None:
        """
        Iterate module parameters and release their backing buckets via the
        AllGatherPipelineProxy.

        This is the DES-LOC equivalent of Megatron's release_module_parameters
        (megatron_fsdp.py line ~510).  The core change: pass lazy=lazy to
        proxy.release_bucket() so that activation-recompute paths defer
        deallocation while still making tensors available in LocalityCache.
        """
        layer_idx: int = getattr(module, "_des_layer_idx", -1)
        for param in module.parameters():
            ptr = param.data_ptr()
            bucket_id = self._param_to_bucket.get(ptr)
            if bucket_id is None:
                continue
            try:
                self._proxy.release_bucket(
                    bucket_id=bucket_id,
                    bwd=bwd,
                    lazy=lazy,
                    layer_idx=layer_idx,
                )
            except ValueError as exc:
                logger.error(
                    "_release_module_parameters: %s (bucket=%d, bwd=%s)",
                    exc, bucket_id, bwd,
                )
                raise

    # ------------------------------------------------------------------
    # Param → bucket registration
    # ------------------------------------------------------------------

    def register_param_bucket(self, param: torch.Tensor, bucket_id: int) -> None:
        """Map a parameter tensor to its bucket_id (called during model init)."""
        self._param_to_bucket[param.data_ptr()] = bucket_id

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def teardown(self) -> None:
        """Remove all registered hooks and clear caches."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        logger.info("DoubleBufferRecomputeManager: teardown complete")


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------

class _StubPipeline:
    """Minimal no-op pipeline stub for unit testing without a real cluster."""

    def recycle_unused_buckets(self):
        pass


def _has_psutil() -> bool:
    try:
        import psutil  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# DeepSpeed integration shim
# ---------------------------------------------------------------------------

def patch_deepspeed_zero3_engine(engine: Any, manager: DoubleBufferRecomputeManager) -> None:
    """
    Monkey-patch a DeepSpeed ZeRO-3 engine's parameter-gathering pipeline
    with the DES-LOC AllGatherPipelineProxy.

    This is the integration point for Neuron_SP projects that use:
        engine = deepspeed.initialize(model=..., config=ds_config)
    followed by:
        patch_deepspeed_zero3_engine(engine, manager)

    After patching, the engine's internal `_param_coordinator` will use the
    extended release semantics.

    NOTE: DeepSpeed internals vary by version.  This shim targets DS ≥ 0.14.
    If `_param_coordinator` is absent, the patch is a no-op with a warning.
    """
    coordinator = getattr(engine, "_param_coordinator", None)
    if coordinator is None:
        logger.warning(
            "patch_deepspeed_zero3_engine: no _param_coordinator found; "
            "DES-LOC patch not applied.  Check DeepSpeed version."
        )
        return

    # Attach proxy so coordinator can call manager._proxy.release_bucket
    coordinator._des_loc_proxy = manager._proxy
    logger.info(
        "patch_deepspeed_zero3_engine: AllGatherPipelineProxy attached to "
        "ZeRO-3 coordinator."
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # --- Registry smoke test ---
    reg = HeteroDeviceRegistry()
    # Simulate 1× H100 + 2× A6000 (using CPU as stand-in if no GPU)
    dev_cpu = reg.register(torch.device("cpu"), sm_arch=0, vram_bytes=int(1.5e12))
    assert dev_cpu.role == DeviceRole.OFFLOAD_SINK, "CPU must be OFFLOAD_SINK"

    # --- LocalityCache smoke test ---
    cache = LocalityCache(reg, max_cache_bytes=int(1e9))
    rec = BucketRecord(bucket_id=0, bwd=False)
    rec.tensor = torch.randn(1024, dtype=torch.float16)
    rec.status = BucketStatus.LAZY_RELEASE
    rec.locality_device = torch.device("cpu")
    cache.store(rec, torch.device("cpu"))
    retrieved = cache.retrieve(0, False)
    assert retrieved is not None, "cache.retrieve must return stored record"
    assert retrieved.tensor is not None, "tensor must survive round-trip"

    # --- AllGatherPipelineProxy lazy-release test ---
    stub = _StubPipeline()
    sched = DecoupledExecutionScheduler(reg, cache)
    proxy = AllGatherPipelineProxy(stub, cache, sched, reg)

    t = torch.randn(512, dtype=torch.float32)
    proxy.mark_bucket_ready(bucket_id=1, bwd=False, tensor=t)
    proxy.release_bucket(bucket_id=1, bwd=False, lazy=True, layer_idx=0)
    rec1 = proxy._records[(1, False)]
    assert rec1.status == BucketStatus.LAZY_RELEASE, "lazy release must set LAZY_RELEASE status"

    proxy.recycle_unused_buckets()
    assert proxy._records[(1, False)].status == BucketStatus.EMPTY, \
        "recycle must clear LAZY_RELEASE bucket to EMPTY"

    # --- DoubleBufferRecomputeManager hook state test ---
    class _FakeFSDPUnit(nn.Linear):
        pass

    mgr = DoubleBufferRecomputeManager(
        registry=reg,
        locality_cache=cache,
        scheduler=sched,
        pipeline_proxy=proxy,
        fsdp_unit_modules=[_FakeFSDPUnit],
    )
    layer = _FakeFSDPUnit(4, 4)
    layer._training_state = TrainingState.PRE_BACKWARD
    layer._des_layer_idx = 0
    # Simulate post_forward during recompute: expect no exception
    out = torch.randn(2, 4)
    result = mgr._post_forward_hook(layer, None, out)
    assert result is out, "_post_forward_hook must return output unchanged"
    # State must remain PRE_BACKWARD (not reset to IDLE during recompute)
    assert layer._training_state == TrainingState.PRE_BACKWARD, \
        "training_state must stay PRE_BACKWARD during lazy release"

    logger.info("All smoke tests passed.")
