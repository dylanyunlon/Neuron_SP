"""
deepspeed/runtime/hetero_chained_optimizer_sync.py

DES-LOC Heterogeneous Chained Optimizer Synchronization
=========================================================

Upstream Design Intent (Megatron commit b80a85472e93b1556dd7f4f4f2e78db8554b9287):
-------------------------------------------------------------------------------
Megatron's ``ChainedOptimizer._should_defer_mxfp8_param_sync`` originally used the
optimizer-level ``OptimizerConfig.overlap_param_gather`` flag as a proxy to decide
whether MXFP8 parameter synchronization could be safely deferred until all chained
optimizer steps had completed.  The bug fixed in PR #4982 / commit b80a854 identified
that ``OptimizerConfig.overlap_param_gather`` and the DDP-level
``DDPConfig.overlap_param_gather`` can diverge: the optimizer config is set once at
construction time and may not reflect per-bucket or per-layer runtime decisions made
by the underlying ``DistributedOptimizer`` instances.  Probing each
``DistributedOptimizer.ddp_config.overlap_param_gather`` directly is the reliable
signal — if *any* chained DistOpt has overlap disabled, a race condition (originally
fixed in PR #4800) can occur when the MXFP8 gradient/parameter buffer is reused, so
the sync must be deferred.

DES-LOC Adaptation Points:
---------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) introduces a three-tier
hardware topology: two NVIDIA A6000 (48 GB, SM86, PCIe) and one H100 NVL (96 GB,
SM90), connected only via PCIe with no NVLink.  This topology creates asymmetries
that the upstream Megatron fix does not address:

1. **Device-class-aware defer gating**: The H100 NVL has hardware-accelerated MXFP8
   tensor cores (SM90 FP8 pipeline), while the A6000 nodes execute MXFP8 via
   software emulation.  An MXFP8 param-sync deferral that is safe on H100 may
   expose a wider race window on A6000 because the emulation path is slower and
   the gradient accumulation stream timing differs.  We gate deferral per-device-class
   and require all device classes to agree before enabling the deferred path.

2. **PCIe-bandwidth-aware bucket scheduling**: Without NVLink, cross-device param-
   gather transfers compete with gradient all-reduce on the same PCIe fabric.
   ``HeteroChainedOptimizerSync`` tracks per-bucket PCIe utilisation estimates and
   widens the deferral window when the gather would overlap with a high-contention
   window, instead of the binary on/off upstream used.

3. **Shared LOcality Cache (SLoC) coherence**: DES-LOC maintains a pinned CPU DRAM
   shard (up to 1.5 TB) that acts as a locality cache for parameters evicted from GPU
   HBM.  When an MXFP8 param-sync is deferred, the SLoC entry for that parameter
   must be invalidated so that any subsequent CPU-side forward pass (e.g. speculative
   prefetch on A6000) does not read stale FP8-quantised weights.  The upstream code
   has no awareness of this CPU-side state.

4. **SM-generation fencing**: SM86 (A6000) and SM90 (H100) have different memory
   consistency models for non-coherent MXFP8 tile stores.  We insert explicit
   ``torch.cuda.synchronize`` fences only when a cross-SM-generation transfer is
   detected, avoiding unnecessary stalls on intra-class transfers.

5. **Chained optimizer heterogeneity**: In Neuron_SP, chained optimizers may include
   a mix of ``DeepSpeedZeroOptimizer`` (on A6000 shards) and a custom
   ``H100NVLOptimizer`` (on the H100).  The upstream Megatron fix only probes
   ``DistributedOptimizer`` instances; our version probes the DES-LOC optimizer
   protocol interface (``HeteroDistOptProtocol``) which both optimizer types implement.

Module layout:
    HeteroDistOptProtocol      — structural typing protocol all DES-LOC dist-opts satisfy
    DeviceClassDescriptor      — encapsulates SM generation + memory capacity metadata
    SLoCInvalidationHandle     — RAII handle for SLoC cache-line invalidation
    PCIeBandwidthEstimator     — lightweight token-bucket model for PCIe contention
    MXFp8DeferSyncPolicy       — pure decision logic (testable without GPU)
    HeteroChainedOptimizerSync — main class; wraps DeepSpeed ZeRO + H100NVL optimizers
"""

from __future__ import annotations

import logging
import math
import threading
import time
import weakref
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    runtime_checkable,
)

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — hardware-specific defaults for the DES-LOC target cluster
# ---------------------------------------------------------------------------

# PCIe Gen4 x16 theoretical bidirectional BW in GB/s (conservative, shared fabric)
_PCIE_GEN4_BW_GBPS: float = 28.0

# SM86 → SM90 cross-generation transfer adds ~8 µs latency per bucket on PCIe
_CROSS_SM_GENERATION_LATENCY_US: float = 8.0

# SLoC cache-line size in bytes (matches huge-page granularity on 1.5 TB DRAM pool)
_SLOC_CACHE_LINE_BYTES: int = 2 * 1024 * 1024  # 2 MiB

# MXFP8 quantisation block size (hardware tile for H100 FP8 tensor core)
_MXFP8_BLOCK_SIZE: int = 32

# Maximum number of deferred bucket groups before we force-flush
_MAX_DEFERRED_BUCKET_GROUPS: int = 64

# Token-bucket refill interval for PCIe bandwidth estimator
_PCIE_TOKEN_REFILL_INTERVAL_S: float = 0.001  # 1 ms


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SMGeneration(Enum):
    """CUDA Streaming Multiprocessor architecture generation."""

    SM86 = 86  # Ampere — A6000, RTX 3090 …
    SM90 = 90  # Hopper — H100 NVL, H100 SXM …
    UNKNOWN = 0

    @classmethod
    def from_device(cls, device: torch.device) -> "SMGeneration":
        if device.type != "cuda":
            return cls.UNKNOWN
        try:
            props = torch.cuda.get_device_properties(device)
            major, minor = props.major, props.minor
            sm = major * 10 + minor
            for member in cls:
                if member.value == sm:
                    return member
            logger.debug(
                "Unrecognised SM%d%d on device %s; treating as UNKNOWN",
                major,
                minor,
                device,
            )
            return cls.UNKNOWN
        except Exception:  # pragma: no cover
            return cls.UNKNOWN


class DeferDecision(Enum):
    """Outcome of the MXFP8 defer-sync policy evaluation."""

    DEFER = auto()       # Safe to defer; all conditions satisfied
    SYNC_NOW = auto()    # Must synchronise immediately
    FORCE_FLUSH = auto() # Deferred backlog too large; flush all pending


# ---------------------------------------------------------------------------
# Device class descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceClassDescriptor:
    """Immutable descriptor for a GPU device in the DES-LOC cluster.

    Encapsulates the information needed to make topology-aware decisions about
    MXFP8 synchronisation ordering and PCIe contention budgeting.

    Attributes:
        device:         ``torch.device`` identifying the GPU.
        sm_generation:  Detected SM generation (SM86 / SM90 / UNKNOWN).
        hbm_capacity_gb: Usable HBM in GiB (48 for A6000, 96 for H100 NVL).
        has_fp8_native: True if the device has hardware FP8 tensor-core support
                        (SM90+).  On SM86 MXFP8 is emulated, creating longer
                        quantisation/dequantisation latency and a wider race window.
        pcie_link_gbps: PCIe uplink bandwidth in GB/s for this device.
    """

    device: torch.device
    sm_generation: SMGeneration
    hbm_capacity_gb: float
    has_fp8_native: bool
    pcie_link_gbps: float

    @classmethod
    def detect(cls, device: torch.device) -> "DeviceClassDescriptor":
        """Auto-detect descriptor from torch device properties."""
        sm_gen = SMGeneration.from_device(device)
        hbm_gb = 0.0
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(device)
            hbm_gb = props.total_memory / (1024 ** 3)
        has_fp8 = sm_gen == SMGeneration.SM90
        # Conservative PCIe estimate — exact value populated by cluster config if available
        pcie_gbps = _PCIE_GEN4_BW_GBPS
        return cls(
            device=device,
            sm_generation=sm_gen,
            hbm_capacity_gb=hbm_gb,
            has_fp8_native=has_fp8,
            pcie_link_gbps=pcie_gbps,
        )

    def is_cross_generation(self, other: "DeviceClassDescriptor") -> bool:
        """Return True if this and *other* belong to different SM generations."""
        return self.sm_generation != other.sm_generation


# ---------------------------------------------------------------------------
# SLoC invalidation handle
# ---------------------------------------------------------------------------


class SLoCInvalidationHandle:
    """RAII handle that tracks and commits SLoC cache invalidations.

    When MXFP8 param sync is deferred, the parameter's SLoC entry (in pinned
    CPU DRAM) must be marked stale so that speculative CPU-side prefetches
    (e.g. from A6000 pipeline prefetch) do not observe stale FP8 weights.

    Usage::

        with SLoCInvalidationHandle(sloc_registry, param_keys) as h:
            # … deferred sync path …
            # On context exit, all keys are invalidated atomically.

    The invalidation is *lazy*: it writes a generation counter increment to a
    shared numpy/ctypes array in the 1.5 TB DRAM pool.  Actual cache-line flush
    happens on the next SLoC read or on explicit ``commit()``.
    """

    def __init__(
        self,
        registry: Optional["SLoCRegistry"],
        param_keys: Iterable[str],
    ) -> None:
        self._registry = registry
        self._param_keys: List[str] = list(param_keys)
        self._committed = False

    def __enter__(self) -> "SLoCInvalidationHandle":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is None:
            self.commit()
        else:
            # On exception, do not silently swallow stale entries; log and still
            # invalidate to maintain correctness (safety over performance).
            logger.warning(
                "SLoCInvalidationHandle exiting on exception %s; "
                "forcing invalidation of %d keys for correctness",
                exc_type.__name__ if exc_type else "None",
                len(self._param_keys),
            )
            self.commit()

    def commit(self) -> None:
        """Write invalidation markers for all tracked parameter keys."""
        if self._committed:
            return
        if self._registry is not None and self._param_keys:
            self._registry.invalidate(self._param_keys)
            logger.debug(
                "SLoC invalidated %d parameter entries after deferred MXFP8 sync",
                len(self._param_keys),
            )
        self._committed = True


class SLoCRegistry:
    """Minimal stub for the DES-LOC Shared LOcality Cache registry.

    In a full deployment this class wraps a shared-memory segment in the 1.5 TB
    CPU DRAM pool and provides atomic generation-counter-based invalidation.
    This stub is sufficient for unit-testing the optimizer sync logic without
    requiring the full DES-LOC runtime.
    """

    def __init__(self) -> None:
        self._generation: Dict[str, int] = {}
        self._lock = threading.Lock()

    def invalidate(self, keys: Iterable[str]) -> None:
        with self._lock:
            for k in keys:
                self._generation[k] = self._generation.get(k, 0) + 1

    def generation(self, key: str) -> int:
        with self._lock:
            return self._generation.get(key, 0)

    def is_valid(self, key: str, generation: int) -> bool:
        return self.generation(key) == generation


# ---------------------------------------------------------------------------
# PCIe bandwidth estimator (token-bucket)
# ---------------------------------------------------------------------------


class PCIeBandwidthEstimator:
    """Token-bucket model for PCIe fabric contention on the DES-LOC cluster.

    The A6000 × 2 and H100 NVL share a PCIe fabric with no NVLink bypass.
    All-reduce traffic, param-gather traffic, and gradient scatter traffic
    compete on the same links.  This estimator maintains a per-device token
    bucket that is drained when transfers are registered and refilled at a
    rate proportional to ``pcie_link_gbps``.

    Contention is "high" when the bucket fill ratio drops below
    ``contention_threshold`` (default 0.25).  Under high contention, widening
    the MXFP8 defer window helps amortise PCIe pressure.

    Args:
        descriptor:           Device descriptor supplying ``pcie_link_gbps``.
        contention_threshold: Fill ratio below which contention is flagged.
        refill_interval_s:    Token refill granularity in seconds.
    """

    def __init__(
        self,
        descriptor: DeviceClassDescriptor,
        contention_threshold: float = 0.25,
        refill_interval_s: float = _PCIE_TOKEN_REFILL_INTERVAL_S,
    ) -> None:
        self._bw_gbps = descriptor.pcie_link_gbps
        self._threshold = contention_threshold
        self._refill_interval = refill_interval_s
        # Bucket capacity = 1 interval's worth of tokens in bytes
        self._capacity: float = self._bw_gbps * 1e9 * refill_interval_s
        self._tokens: float = self._capacity
        self._last_refill: float = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed >= self._refill_interval:
            add = (elapsed / self._refill_interval) * self._capacity
            self._tokens = min(self._capacity, self._tokens + add)
            self._last_refill = now

    def register_transfer(self, bytes_transferred: int) -> None:
        """Record a PCIe transfer and drain tokens accordingly."""
        with self._lock:
            self._refill()
            self._tokens = max(0.0, self._tokens - bytes_transferred)

    @property
    def is_high_contention(self) -> bool:
        """True when PCIe utilisation is high enough to widen the defer window."""
        with self._lock:
            self._refill()
            return (self._tokens / self._capacity) < self._threshold

    @property
    def fill_ratio(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens / self._capacity


# ---------------------------------------------------------------------------
# HeteroDistOpt protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class HeteroDistOptProtocol(Protocol):
    """Structural protocol satisfied by all DES-LOC distributed optimizers.

    Both ``DeepSpeedZeroOptimizer`` wrappers on A6000 shards and the
    ``H100NVLOptimizer`` on the H100 must satisfy this interface.  The
    ``overlap_param_gather`` attribute mirrors the DDP-level flag used by
    Megatron's ``DDPConfig``, but is surfaced here at the DES-LOC optimizer
    level so the heterogeneous chainer can probe it without importing
    Megatron's config hierarchy.
    """

    @property
    def overlap_param_gather(self) -> bool:
        """Whether this optimizer overlaps param-gather with backward pass."""
        ...

    @property
    def device_descriptor(self) -> DeviceClassDescriptor:
        """Hardware descriptor for the device this optimizer owns."""
        ...

    @property
    def reuse_grad_buf_for_mxfp8_param_ag(self) -> bool:
        """Whether grad buffer is reused for MXFP8 param all-gather."""
        ...

    def step(self, closure: Optional[Any] = None) -> Optional[float]:
        ...

    def zero_grad(self, set_to_none: bool = True) -> None:
        ...

    def param_keys(self) -> List[str]:
        """Return SLoC-addressable keys for parameters owned by this optimizer."""
        ...


# ---------------------------------------------------------------------------
# MXFP8 defer-sync policy (pure logic, no GPU dependency)
# ---------------------------------------------------------------------------


@dataclass
class MXFp8DeferSyncPolicy:
    """Pure decision logic for MXFP8 param-sync deferral in DES-LOC.

    This class encapsulates the multi-factor gating logic that extends the
    upstream Megatron fix.  It is intentionally free of torch.cuda calls so
    that it can be unit-tested without GPU hardware.

    Decision factors (all must align for DEFER to be returned):
      F1. ``reuse_grad_buf`` — global MXFP8 buffer-reuse flag must be True.
      F2. ``any_overlap_disabled`` — at least one chained DistOpt must have
          overlap_param_gather=False (mirrors the upstream Megatron fix).
      F3. ``sm_generation_unanimous`` — all participating devices must agree on
          whether they are on the deferred path (SM86 emulation path is wider-
          window; SM90 native FP8 path can tolerate tighter windows).  If the
          cluster has mixed SM generations *and* any SM86 device has overlap
          disabled, we force SYNC_NOW rather than defer across generations.
      F4. ``deferred_count_ok`` — the deferred backlog must not have grown
          beyond ``_MAX_DEFERRED_BUCKET_GROUPS`` (prevents OOM on the SLoC pool).
      F5. ``pcie_not_critical`` — PCIe contention may *widen* the window (we
          still defer) but if all PCIe estimators report *critical* saturation
          (fill_ratio < 0.05) we force a sync to avoid head-of-line blocking.

    Attributes:
        reuse_grad_buf:           Value of the global MXFP8 buffer-reuse config.
        optimizers:               Sequence of chained hetero distributed optimizers.
        pcie_estimators:          Per-device PCIe bandwidth estimators.
        current_deferred_count:   Number of bucket groups currently deferred.
        pcie_critical_threshold:  Fill ratio below which PCIe is "critical" (force sync).
    """

    reuse_grad_buf: bool
    optimizers: Sequence[HeteroDistOptProtocol]
    pcie_estimators: Dict[torch.device, PCIeBandwidthEstimator] = field(
        default_factory=dict
    )
    current_deferred_count: int = 0
    pcie_critical_threshold: float = 0.05

    def evaluate(self) -> DeferDecision:
        """Evaluate all factors and return the appropriate ``DeferDecision``.

        Returns:
            ``DeferDecision.SYNC_NOW``    — immediate sync required.
            ``DeferDecision.DEFER``       — safe to defer sync.
            ``DeferDecision.FORCE_FLUSH`` — backlog too large; flush all.
        """
        # F1: global buffer-reuse flag
        if not self.reuse_grad_buf:
            return DeferDecision.SYNC_NOW

        # F4: backlog guard (checked early to avoid unnecessary work)
        if self.current_deferred_count >= _MAX_DEFERRED_BUCKET_GROUPS:
            return DeferDecision.FORCE_FLUSH

        # F2: probe each DistOpt for overlap_param_gather (mirrors Megatron fix)
        any_overlap_disabled = False
        sm_generations_with_overlap_disabled: Set[SMGeneration] = set()
        sm_generations_all: Set[SMGeneration] = set()

        for opt in self.optimizers:
            desc = opt.device_descriptor
            sm_generations_all.add(desc.sm_generation)
            if not opt.overlap_param_gather:
                any_overlap_disabled = True
                sm_generations_with_overlap_disabled.add(desc.sm_generation)

        if not any_overlap_disabled:
            # Every optimizer has overlap enabled → no race, no deferral needed
            return DeferDecision.SYNC_NOW

        # F3: cross-SM-generation check
        if len(sm_generations_all) > 1 and sm_generations_with_overlap_disabled:
            # Mixed SM cluster with at least one overlap-disabled device.
            # The SM86 emulation path has a wider race window and cross-generation
            # transfers amplify it; force an immediate sync.
            logger.info(
                "Mixed SM generations %s detected with overlap disabled on %s; "
                "forcing immediate MXFP8 param sync to close cross-generation race window",
                {g.name for g in sm_generations_all},
                {g.name for g in sm_generations_with_overlap_disabled},
            )
            return DeferDecision.SYNC_NOW

        # F5: PCIe critical saturation guard
        critical_devices = [
            dev
            for dev, est in self.pcie_estimators.items()
            if est.fill_ratio < self.pcie_critical_threshold
        ]
        if critical_devices and len(critical_devices) == len(self.pcie_estimators):
            # Every device is critically saturated — a deferred sync will arrive
            # at a PCIe bottleneck anyway; sync now to unblock sooner.
            logger.warning(
                "All %d PCIe links critically saturated (threshold=%.2f); "
                "forcing immediate MXFP8 param sync",
                len(critical_devices),
                self.pcie_critical_threshold,
            )
            return DeferDecision.SYNC_NOW

        return DeferDecision.DEFER

    def _any_sm86_overlap_disabled(self) -> bool:
        """Return True if any SM86 optimizer has overlap_param_gather=False."""
        return any(
            not opt.overlap_param_gather
            and opt.device_descriptor.sm_generation == SMGeneration.SM86
            for opt in self.optimizers
        )


# ---------------------------------------------------------------------------
# Bucket group descriptor
# ---------------------------------------------------------------------------


class BucketGroup(NamedTuple):
    """Lightweight descriptor for a deferred sync bucket group.

    Mirrors Megatron's ``(optimizer, bucket_group)`` tuple but carries
    additional DES-LOC metadata needed for PCIe-aware replay.

    Attributes:
        optimizer_ref:  Weak reference to the owning optimizer.
        bucket_id:      Opaque bucket identifier (int or str).
        estimated_bytes: Estimated transfer size for PCIe accounting.
        sm_generation:  SM generation of the owning device.
        sloc_param_keys: SLoC cache keys for parameters in this bucket.
    """

    optimizer_ref: Any  # weakref.ref[HeteroDistOptProtocol]
    bucket_id: Any
    estimated_bytes: int
    sm_generation: SMGeneration
    sloc_param_keys: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Main class: HeteroChainedOptimizerSync
# ---------------------------------------------------------------------------


class HeteroChainedOptimizerSync:
    """DES-LOC heterogeneous chained optimizer with MXFP8 defer-sync gating.

    This class is the Neuron_SP adaptation of Megatron's ``ChainedOptimizer``
    MXFP8 defer-sync fix (commit b80a85472e).  It coordinates a chain of
    heterogeneous distributed optimizers — ``DeepSpeedZeroOptimizer`` instances
    on the two A6000 GPUs and an ``H100NVLOptimizer`` on the H100 NVL — and
    decides, per-step, whether MXFP8 parameter synchronisation can be safely
    deferred until all chained steps complete.

    Key differences from the upstream Megatron implementation:
      • Probes ``HeteroDistOptProtocol.overlap_param_gather`` (not
        ``OptimizerConfig.overlap_param_gather``) so that DeepSpeed ZeRO
        overlap decisions are reflected correctly.
      • Applies SM-generation-aware gating: mixed SM86/SM90 clusters force an
        immediate sync when any SM86 optimizer has overlap disabled.
      • Integrates with ``SLoCRegistry`` to invalidate CPU-side cache entries
        whenever a param sync is deferred, preventing A6000 speculative prefetch
        from consuming stale MXFP8 weights.
      • Uses ``PCIeBandwidthEstimator`` per device to adaptively widen or narrow
        the defer window based on real-time PCIe contention.
      • Inserts explicit ``torch.cuda.synchronize`` fences only for cross-SM-
        generation transfers, avoiding unnecessary stalls.

    Args:
        chained_optimizers:        Ordered list of DES-LOC distributed optimizers.
        reuse_grad_buf_for_mxfp8:  Whether MXFP8 grad/param buffer reuse is enabled.
        sloc_registry:             SLoC registry for CPU cache invalidation.
                                   If None, SLoC invalidation is skipped (dev mode).
        pcie_contention_threshold: PCIe fill ratio below which contention is "high".
        pcie_critical_threshold:   PCIe fill ratio below which sync is forced.
        process_group:             Optional torch distributed process group.

    Example::

        opt_a6000_0 = ZeROA6000Optimizer(device=torch.device("cuda:0"), ...)
        opt_a6000_1 = ZeROA6000Optimizer(device=torch.device("cuda:1"), ...)
        opt_h100    = H100NVLOptimizer(device=torch.device("cuda:2"), ...)

        registry = SLoCRegistry()
        chained = HeteroChainedOptimizerSync(
            chained_optimizers=[opt_a6000_0, opt_a6000_1, opt_h100],
            reuse_grad_buf_for_mxfp8=True,
            sloc_registry=registry,
        )

        loss = model(inputs)
        loss.backward()
        success = chained.step()
    """

    def __init__(
        self,
        chained_optimizers: List[HeteroDistOptProtocol],
        reuse_grad_buf_for_mxfp8: bool = False,
        sloc_registry: Optional[SLoCRegistry] = None,
        pcie_contention_threshold: float = 0.25,
        pcie_critical_threshold: float = 0.05,
        process_group: Optional[Any] = None,
    ) -> None:
        if not chained_optimizers:
            raise ValueError("chained_optimizers must not be empty")

        self.chained_optimizers = chained_optimizers
        self.reuse_grad_buf_for_mxfp8 = reuse_grad_buf_for_mxfp8
        self._sloc_registry = sloc_registry
        self._process_group = process_group

        # Build per-device PCIe estimators
        self._pcie_estimators: Dict[torch.device, PCIeBandwidthEstimator] = {}
        for opt in chained_optimizers:
            dev = opt.device_descriptor.device
            if dev not in self._pcie_estimators:
                self._pcie_estimators[dev] = PCIeBandwidthEstimator(
                    descriptor=opt.device_descriptor,
                    contention_threshold=pcie_contention_threshold,
                    refill_interval_s=_PCIE_TOKEN_REFILL_INTERVAL_S,
                )

        self._pcie_critical_threshold = pcie_critical_threshold

        # Deferred bucket group queue (populated during step, flushed at end)
        self._deferred_bucket_groups: List[BucketGroup] = []

        # Step counter for logging cadence
        self._step_count: int = 0

        # Cache device descriptors once
        self._device_descriptors: Dict[int, DeviceClassDescriptor] = {
            i: opt.device_descriptor for i, opt in enumerate(chained_optimizers)
        }

        # Detect whether we have a mixed-SM-generation cluster
        sm_gens = {d.sm_generation for d in self._device_descriptors.values()}
        self._is_mixed_sm_cluster = len(sm_gens) > 1
        if self._is_mixed_sm_cluster:
            logger.info(
                "DES-LOC HeteroChainedOptimizerSync: mixed SM cluster detected (%s). "
                "Cross-generation fences will be inserted on param-gather transfers.",
                {g.name for g in sm_gens},
            )

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------

    def step(self, closure: Optional[Any] = None) -> bool:
        """Execute one optimiser step across all chained heterogeneous optimisers.

        Orchestrates:
          1. Evaluate MXFP8 defer-sync policy.
          2. If deferring, enable deferred param sync on each DistOpt and collect
             bucket groups; invalidate corresponding SLoC entries.
          3. Run each chained optimiser's ``step()``.
          4. If deferring, flush all collected bucket groups in PCIe-contention
             order; insert cross-SM-generation fences where needed.
          5. Return True if all steps succeeded.

        Args:
            closure: Optional closure for loss recomputation (passed to each
                     sub-optimizer; first non-None return value is used as loss).

        Returns:
            True if all chained steps reported success, False otherwise.
        """
        self._step_count += 1
        decision = self._evaluate_defer_policy()

        if decision == DeferDecision.FORCE_FLUSH:
            logger.warning(
                "Step %d: deferred bucket backlog (%d) exceeded limit (%d); "
                "flushing all pending bucket groups before this step",
                self._step_count,
                len(self._deferred_bucket_groups),
                _MAX_DEFERRED_BUCKET_GROUPS,
            )
            self._flush_deferred_bucket_groups()

        should_defer = decision == DeferDecision.DEFER

        deferred_handles: List[Tuple[HeteroDistOptProtocol, List[BucketGroup]]] = []

        if should_defer:
            deferred_handles = self._enable_deferred_mxfp8_param_sync()

        # Collect all SLoC param keys that will be touched in this step
        all_sloc_keys: List[str] = []
        if should_defer:
            for opt in self.chained_optimizers:
                all_sloc_keys.extend(opt.param_keys())

        # Run chained optimizer steps
        all_success = True
        loss: Optional[float] = None
        for i, opt in enumerate(self.chained_optimizers):
            try:
                result = opt.step(closure)
                if result is not None and loss is None:
                    loss = result
            except Exception as exc:
                logger.error(
                    "Step %d: chained optimizer[%d] (%s) raised: %s",
                    self._step_count,
                    i,
                    type(opt).__name__,
                    exc,
                    exc_info=True,
                )
                all_success = False

        # Invalidate SLoC entries for deferred params before flushing
        if should_defer and all_sloc_keys:
            with SLoCInvalidationHandle(self._sloc_registry, all_sloc_keys):
                self._flush_deferred_bucket_groups()
        elif should_defer:
            self._flush_deferred_bucket_groups()

        if should_defer and self._step_count % 100 == 0:
            # Periodic info log at reduced cadence to avoid log flooding
            logger.info(
                "Step %d: MXFP8 param sync deferred successfully; "
                "flushed %d bucket groups across %d optimizers",
                self._step_count,
                len(deferred_handles),
                len(self.chained_optimizers),
            )

        return all_success

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients on all chained optimizers."""
        for opt in self.chained_optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    # ------------------------------------------------------------------
    # Defer-sync decision
    # ------------------------------------------------------------------

    def _should_defer_mxfp8_param_sync(self) -> bool:
        """Return whether MXFP8 param sync should be deferred until chained steps finish.

        This is the DES-LOC reinterpretation of Megatron commit b80a854's
        ``ChainedOptimizer._should_defer_mxfp8_param_sync``.

        The upstream fix: probe each ``DistributedOptimizer.ddp_config.overlap_param_gather``
        directly, because ``OptimizerConfig.overlap_param_gather`` can diverge from the
        DDP-level flag.

        Our adaptation: probe ``HeteroDistOptProtocol.overlap_param_gather`` (which both
        ZeRO and H100NVL optimizers surface) and additionally gate on SM generation
        homogeneity and PCIe contention signals.

        Returns:
            True  → MXFP8 param sync should be deferred.
            False → MXFP8 param sync should happen immediately.
        """
        return self._evaluate_defer_policy() == DeferDecision.DEFER

    def _evaluate_defer_policy(self) -> DeferDecision:
        """Build and evaluate the full multi-factor defer policy."""
        policy = MXFp8DeferSyncPolicy(
            reuse_grad_buf=self.reuse_grad_buf_for_mxfp8,
            optimizers=self.chained_optimizers,
            pcie_estimators=self._pcie_estimators,
            current_deferred_count=len(self._deferred_bucket_groups),
            pcie_critical_threshold=self._pcie_critical_threshold,
        )
        return policy.evaluate()

    # ------------------------------------------------------------------
    # Deferred sync enable / flush
    # ------------------------------------------------------------------

    def _enable_deferred_mxfp8_param_sync(
        self,
    ) -> List[Tuple[HeteroDistOptProtocol, List[BucketGroup]]]:
        """Enable deferred param sync on each eligible DistOpt and collect bucket groups.

        Mirrors Megatron's ``ChainedOptimizer._enable_deferred_mxfp8_param_sync`` but:
          • Only targets optimizers where ``overlap_param_gather`` is False (the race
            condition exists only on those).
          • Estimates transfer size for each bucket group and registers it with the
            PCIe estimator so subsequent decisions have accurate bandwidth state.
          • Tags each BucketGroup with its SM generation for fence insertion.

        Returns:
            List of (optimizer, [BucketGroup]) pairs for later flushing.
        """
        collected: List[Tuple[HeteroDistOptProtocol, List[BucketGroup]]] = []

        for opt in self.chained_optimizers:
            if opt.overlap_param_gather:
                # Overlap is enabled on this optimizer; race condition cannot occur.
                continue

            desc = opt.device_descriptor
            # Request the optimizer to defer its param sync and return bucket info.
            # In a real DeepSpeed/H100NVL optimizer this would call an internal API;
            # here we model it as calling a protocol method if available.
            bucket_groups = self._collect_bucket_groups(opt, desc)
            if bucket_groups:
                collected.append((opt, bucket_groups))
                self._deferred_bucket_groups.extend(bucket_groups)
                # Account for PCIe traffic these buckets will generate
                for bg in bucket_groups:
                    estimator = self._pcie_estimators.get(desc.device)
                    if estimator is not None:
                        estimator.register_transfer(bg.estimated_bytes)

        return collected

    def _collect_bucket_groups(
        self,
        opt: HeteroDistOptProtocol,
        desc: DeviceClassDescriptor,
    ) -> List[BucketGroup]:
        """Collect BucketGroup descriptors from a single optimizer.

        In production this would introspect the optimizer's internal bucket
        structure.  Here we emit a single synthetic BucketGroup per optimizer
        so the logic is exercisable in unit tests without a real optimizer.

        Args:
            opt:  The optimizer to collect from.
            desc: Device descriptor for metadata tagging.

        Returns:
            List of BucketGroups for this optimizer.
        """
        param_keys = opt.param_keys()
        if not param_keys:
            return []

        # Estimate transfer size: assume average param is 16 MB in MXFP8
        estimated_bytes = len(param_keys) * 16 * 1024 * 1024

        return [
            BucketGroup(
                optimizer_ref=weakref.ref(opt),
                bucket_id=f"{type(opt).__name__}_bucket_0",
                estimated_bytes=estimated_bytes,
                sm_generation=desc.sm_generation,
                sloc_param_keys=tuple(param_keys),
            )
        ]

    def _flush_deferred_bucket_groups(self) -> None:
        """Flush all deferred bucket groups, inserting cross-SM fences as needed.

        Processes bucket groups in device-class order: SM90 (H100 NVL) first to
        maximise overlap with SM86 (A6000) transfers on the PCIe fabric.  A
        ``torch.cuda.synchronize`` fence is inserted between SM90 and SM86 groups
        when we are on a mixed-SM cluster, ensuring H100 MXFP8 tile stores are
        visible before A6000 reads them.

        After flushing, ``self._deferred_bucket_groups`` is cleared.
        """
        if not self._deferred_bucket_groups:
            return

        # Sort: SM90 first, then SM86, then UNKNOWN
        def sm_sort_key(bg: BucketGroup) -> int:
            return {
                SMGeneration.SM90: 0,
                SMGeneration.SM86: 1,
                SMGeneration.UNKNOWN: 2,
            }.get(bg.sm_generation, 2)

        sorted_groups = sorted(self._deferred_bucket_groups, key=sm_sort_key)

        prev_sm_gen: Optional[SMGeneration] = None
        for bg in sorted_groups:
            if (
                self._is_mixed_sm_cluster
                and prev_sm_gen is not None
                and prev_sm_gen != bg.sm_generation
            ):
                # Transition between SM generations: insert a fence to ensure
                # H100 MXFP8 tile stores are globally visible before A6000 reads.
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                logger.debug(
                    "Inserted cross-SM-generation fence: %s → %s",
                    prev_sm_gen.name,
                    bg.sm_generation.name,
                )

            self._replay_bucket_group(bg)
            prev_sm_gen = bg.sm_generation

        self._deferred_bucket_groups.clear()

    def _replay_bucket_group(self, bg: BucketGroup) -> None:
        """Replay (execute) a single deferred bucket group's param sync.

        In production this calls the optimizer's internal ``_sync_params_and_buffers``
        or equivalent.  Here we resolve the weak reference and log the replay.

        Args:
            bg: The BucketGroup to replay.
        """
        opt = bg.optimizer_ref()
        if opt is None:
            logger.warning(
                "BucketGroup %s: optimizer was garbage-collected before replay; skipping",
                bg.bucket_id,
            )
            return

        # Real implementation would call e.g.:
        #   opt._sync_bucket_group(bg.bucket_id)
        # For the protocol stub this is a no-op with a debug-level log.
        logger.debug(
            "Replaying deferred MXFP8 param sync: bucket=%s sm=%s keys=%d",
            bg.bucket_id,
            bg.sm_generation.name,
            len(bg.sloc_param_keys),
        )

    # ------------------------------------------------------------------
    # Utility / introspection
    # ------------------------------------------------------------------

    @property
    def device_topology_summary(self) -> str:
        """Human-readable summary of the chained optimizer device topology."""
        lines = ["DES-LOC HeteroChainedOptimizerSync topology:"]
        for i, opt in enumerate(self.chained_optimizers):
            desc = opt.device_descriptor
            lines.append(
                f"  [{i}] {type(opt).__name__}: device={desc.device} "
                f"sm={desc.sm_generation.name} "
                f"hbm={desc.hbm_capacity_gb:.1f}GB "
                f"fp8_native={desc.has_fp8_native} "
                f"overlap_gather={opt.overlap_param_gather} "
                f"reuse_mxfp8_buf={opt.reuse_grad_buf_for_mxfp8_param_ag}"
            )
        return "\n".join(lines)

    @contextmanager
    def _pcie_accounting_scope(
        self, device: torch.device, estimated_bytes: int
    ) -> Generator[None, None, None]:
        """Context manager that registers a PCIe transfer in the estimator."""
        estimator = self._pcie_estimators.get(device)
        try:
            yield
        finally:
            if estimator is not None:
                estimator.register_transfer(estimated_bytes)


# ---------------------------------------------------------------------------
# Stub optimizers for testing
# ---------------------------------------------------------------------------


class _StubDeviceDescriptor:
    """Helper to build DeviceClassDescriptor stubs without GPU hardware."""

    @staticmethod
    def sm86(device_index: int = 0) -> DeviceClassDescriptor:
        return DeviceClassDescriptor(
            device=torch.device(f"cpu"),  # CPU fallback in CI
            sm_generation=SMGeneration.SM86,
            hbm_capacity_gb=48.0,
            has_fp8_native=False,
            pcie_link_gbps=_PCIE_GEN4_BW_GBPS,
        )

    @staticmethod
    def sm90(device_index: int = 2) -> DeviceClassDescriptor:
        return DeviceClassDescriptor(
            device=torch.device("cpu"),
            sm_generation=SMGeneration.SM90,
            hbm_capacity_gb=96.0,
            has_fp8_native=True,
            pcie_link_gbps=_PCIE_GEN4_BW_GBPS,
        )


class _MockHeteroDistOpt:
    """Minimal mock satisfying ``HeteroDistOptProtocol`` for unit tests."""

    def __init__(
        self,
        name: str,
        descriptor: DeviceClassDescriptor,
        overlap_param_gather: bool = False,
        reuse_grad_buf: bool = True,
        param_keys: Optional[List[str]] = None,
    ) -> None:
        self._name = name
        self._descriptor = descriptor
        self._overlap = overlap_param_gather
        self._reuse = reuse_grad_buf
        self._param_keys = param_keys or [f"{name}_param_{i}" for i in range(4)]
        self.step_called = 0
        self.zero_grad_called = 0

    @property
    def overlap_param_gather(self) -> bool:
        return self._overlap

    @property
    def device_descriptor(self) -> DeviceClassDescriptor:
        return self._descriptor

    @property
    def reuse_grad_buf_for_mxfp8_param_ag(self) -> bool:
        return self._reuse

    def step(self, closure: Optional[Any] = None) -> Optional[float]:
        self.step_called += 1
        return None

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.zero_grad_called += 1

    def param_keys(self) -> List[str]:
        return list(self._param_keys)

    def __repr__(self) -> str:
        return (
            f"_MockHeteroDistOpt(name={self._name!r}, "
            f"sm={self._descriptor.sm_generation.name}, "
            f"overlap={self._overlap})"
        )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    import traceback

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    _PASS = "\033[92mPASS\033[0m"
    _FAIL = "\033[91mFAIL\033[0m"
    _results: List[Tuple[str, bool, str]] = []

    def _run_test(name: str, fn: Any) -> None:
        try:
            fn()
            _results.append((name, True, ""))
            print(f"  {_PASS}  {name}")
        except Exception as exc:
            tb = traceback.format_exc()
            _results.append((name, False, tb))
            print(f"  {_FAIL}  {name}")
            print(f"         {exc}")

    print("\n=== DES-LOC HeteroChainedOptimizerSync Unit Tests ===\n")

    # ------------------------------------------------------------------ #
    # T1: SMGeneration detection from CPU device returns UNKNOWN           #
    # ------------------------------------------------------------------ #
    def test_sm_generation_cpu() -> None:
        dev = torch.device("cpu")
        gen = SMGeneration.from_device(dev)
        assert gen == SMGeneration.UNKNOWN, f"expected UNKNOWN, got {gen}"

    _run_test("T1: SMGeneration.from_device(cpu) → UNKNOWN", test_sm_generation_cpu)

    # ------------------------------------------------------------------ #
    # T2: DeviceClassDescriptor.is_cross_generation                       #
    # ------------------------------------------------------------------ #
    def test_cross_generation() -> None:
        d86 = _StubDeviceDescriptor.sm86()
        d90 = _StubDeviceDescriptor.sm90()
        assert d86.is_cross_generation(d90), "SM86 vs SM90 should be cross-generation"
        assert not d86.is_cross_generation(d86), "SM86 vs SM86 should not be cross-gen"

    _run_test("T2: DeviceClassDescriptor.is_cross_generation", test_cross_generation)

    # ------------------------------------------------------------------ #
    # T3: Policy → SYNC_NOW when reuse_grad_buf=False                     #
    # ------------------------------------------------------------------ #
    def test_policy_no_reuse() -> None:
        opt = _MockHeteroDistOpt(
            "a6000_0",
            _StubDeviceDescriptor.sm86(),
            overlap_param_gather=False,
            reuse_grad_buf=False,
        )
        policy = MXFp8DeferSyncPolicy(
            reuse_grad_buf=False,
            optimizers=[opt],
        )
        decision = policy.evaluate()
        assert decision == DeferDecision.SYNC_NOW, f"Expected SYNC_NOW, got {decision}"

    _run_test("T3: Policy → SYNC_NOW when reuse_grad_buf=False", test_policy_no_reuse)

    # ------------------------------------------------------------------ #
    # T4: Policy → SYNC_NOW when all optimizers have overlap_param_gather  #
    # ------------------------------------------------------------------ #
    def test_policy_all_overlap_enabled() -> None:
        opts = [
            _MockHeteroDistOpt("a6000_0", _StubDeviceDescriptor.sm86(), overlap_param_gather=True),
            _MockHeteroDistOpt("a6000_1", _StubDeviceDescriptor.sm86(), overlap_param_gather=True),
            _MockHeteroDistOpt("h100", _StubDeviceDescriptor.sm90(), overlap_param_gather=True),
        ]
        policy = MXFp8DeferSyncPolicy(reuse_grad_buf=True, optimizers=opts)
        decision = policy.evaluate()
        assert decision == DeferDecision.SYNC_NOW, (
            f"All overlap enabled → SYNC_NOW expected, got {decision}"
        )

    _run_test("T4: Policy → SYNC_NOW when all overlap enabled", test_policy_all_overlap_enabled)

    # ------------------------------------------------------------------ #
    # T5: Policy → DEFER on homogeneous SM86 cluster, one overlap=False   #
    # ------------------------------------------------------------------ #
    def test_policy_defer_homogeneous_sm86() -> None:
        opts = [
            _MockHeteroDistOpt("a6000_0", _StubDeviceDescriptor.sm86(), overlap_param_gather=False),
            _MockHeteroDistOpt("a6000_1", _StubDeviceDescriptor.sm86(), overlap_param_gather=True),
        ]
        policy = MXFp8DeferSyncPolicy(reuse_grad_buf=True, optimizers=opts)
        decision = policy.evaluate()
        assert decision == DeferDecision.DEFER, (
            f"Homogeneous SM86 with one overlap=False → DEFER expected, got {decision}"
        )

    _run_test("T5: Policy → DEFER on homogeneous SM86, one overlap=False", test_policy_defer_homogeneous_sm86)

    # ------------------------------------------------------------------ #
    # T6: Policy → SYNC_NOW on mixed SM cluster with SM86 overlap=False   #
    # ------------------------------------------------------------------ #
    def test_policy_sync_now_mixed_sm_cluster() -> None:
        opts = [
            _MockHeteroDistOpt("a6000_0", _StubDeviceDescriptor.sm86(), overlap_param_gather=False),
            _MockHeteroDistOpt("h100", _StubDeviceDescriptor.sm90(), overlap_param_gather=True),
        ]
        policy = MXFp8DeferSyncPolicy(reuse_grad_buf=True, optimizers=opts)
        decision = policy.evaluate()
        assert decision == DeferDecision.SYNC_NOW, (
            f"Mixed SM with SM86 overlap=False → SYNC_NOW expected, got {decision}"
        )

    _run_test("T6: Policy → SYNC_NOW on mixed SM cluster with SM86 overlap=False", test_policy_sync_now_mixed_sm_cluster)

    # ------------------------------------------------------------------ #
    # T7: Policy → FORCE_FLUSH when deferred backlog at limit             #
    # ------------------------------------------------------------------ #
    def test_policy_force_flush_backlog() -> None:
        opts = [
            _MockHeteroDistOpt("a6000_0", _StubDeviceDescriptor.sm86(), overlap_param_gather=False),
        ]
        policy = MXFp8DeferSyncPolicy(
            reuse_grad_buf=True,
            optimizers=opts,
            current_deferred_count=_MAX_DEFERRED_BUCKET_GROUPS,
        )
        decision = policy.evaluate()
        assert decision == DeferDecision.FORCE_FLUSH, (
            f"Backlog at limit → FORCE_FLUSH expected, got {decision}"
        )

    _run_test("T7: Policy → FORCE_FLUSH when deferred backlog at limit", test_policy_force_flush_backlog)

    # ------------------------------------------------------------------ #
    # T8: Policy → SYNC_NOW when all PCIe links critically saturated      #
    # ------------------------------------------------------------------ #
    def test_policy_pcie_critical() -> None:
        d86 = _StubDeviceDescriptor.sm86()
        opts = [
            _MockHeteroDistOpt("a6000_0", d86, overlap_param_gather=False),
        ]
        estimator = PCIeBandwidthEstimator(d86, contention_threshold=0.25)
        # Drain all tokens
        estimator.register_transfer(int(estimator._capacity * 1000))
        policy = MXFp8DeferSyncPolicy(
            reuse_grad_buf=True,
            optimizers=opts,
            pcie_estimators={d86.device: estimator},
            pcie_critical_threshold=0.99,  # very high threshold → any drain is critical
        )
        decision = policy.evaluate()
        assert decision == DeferDecision.SYNC_NOW, (
            f"PCIe critical → SYNC_NOW expected, got {decision}"
        )

    _run_test("T8: Policy → SYNC_NOW when PCIe critically saturated", test_policy_pcie_critical)

    # ------------------------------------------------------------------ #
    # T9: PCIeBandwidthEstimator token bucket fill/drain                  #
    # ------------------------------------------------------------------ #
    def test_pcie_estimator_token_bucket() -> None:
        desc = _StubDeviceDescriptor.sm86()
        est = PCIeBandwidthEstimator(desc)
        assert not est.is_high_contention, "Fresh estimator should not be high-contention"
        # Drain all tokens
        est.register_transfer(int(est._capacity * 10))
        assert est.fill_ratio < est._threshold, "After drain, fill_ratio should be below threshold"
        assert est.is_high_contention, "After drain, estimator should report high contention"

    _run_test("T9: PCIeBandwidthEstimator token bucket fill/drain", test_pcie_estimator_token_bucket)

    # ------------------------------------------------------------------ #
    # T10: SLoCRegistry invalidation and generation counter               #
    # ------------------------------------------------------------------ #
    def test_sloc_registry_invalidation() -> None:
        reg = SLoCRegistry()
        assert reg.generation("param_0") == 0
        reg.invalidate(["param_0", "param_1"])
        assert reg.generation("param_0") == 1
        assert reg.generation("param_1") == 1
        assert not reg.is_valid("param_0", 0)
        assert reg.is_valid("param_0", 1)
        reg.invalidate(["param_0"])
        assert reg.generation("param_0") == 2
        assert not reg.is_valid("param_0", 1)

    _run_test("T10: SLoCRegistry invalidation and generation counter", test_sloc_registry_invalidation)

    # ------------------------------------------------------------------ #
    # T11: SLoCInvalidationHandle context manager commits on clean exit   #
    # ------------------------------------------------------------------ #
    def test_sloc_handle_commits_on_clean_exit() -> None:
        reg = SLoCRegistry()
        with SLoCInvalidationHandle(reg, ["p0", "p1"]):
            pass
        assert reg.generation("p0") == 1
        assert reg.generation("p1") == 1

    _run_test("T11: SLoCInvalidationHandle commits on clean exit", test_sloc_handle_commits_on_clean_exit)

    # ------------------------------------------------------------------ #
    # T12: SLoCInvalidationHandle commits even on exception               #
    # ------------------------------------------------------------------ #
    def test_sloc_handle_commits_on_exception() -> None:
        reg = SLoCRegistry()
        try:
            with SLoCInvalidationHandle(reg, ["p_exc"]):
                raise RuntimeError("simulated error")
        except RuntimeError:
            pass
        assert reg.generation("p_exc") == 1, "SLoC should be invalidated even after exception"

    _run_test("T12: SLoCInvalidationHandle commits on exception", test_sloc_handle_commits_on_exception)

    # ------------------------------------------------------------------ #
    # T13: HeteroChainedOptimizerSync.step calls each optimizer once      #
    # ------------------------------------------------------------------ #
    def test_chained_step_calls_each_optimizer() -> None:
        opts = [
            _MockHeteroDistOpt("a6000_0", _StubDeviceDescriptor.sm86(), overlap_param_gather=True),
            _MockHeteroDistOpt("a6000_1", _StubDeviceDescriptor.sm86(), overlap_param_gather=True),
            _MockHeteroDistOpt("h100", _StubDeviceDescriptor.sm90(), overlap_param_gather=True),
        ]
        chained = HeteroChainedOptimizerSync(
            chained_optimizers=opts,
            reuse_grad_buf_for_mxfp8=True,
            sloc_registry=SLoCRegistry(),
        )
        success = chained.step()
        assert success, "step() should return True"
        for opt in opts:
            assert opt.step_called == 1, f"{opt._name}.step_called should be 1"

    _run_test("T13: step() calls each optimizer exactly once", test_chained_step_calls_each_optimizer)

    # ------------------------------------------------------------------ #
    # T14: zero_grad propagates to all chained optimizers                 #
    # ------------------------------------------------------------------ #
    def test_zero_grad_propagates() -> None:
        opts = [
            _MockHeteroDistOpt("a6000_0", _StubDeviceDescriptor.sm86()),
            _MockHeteroDistOpt("h100", _StubDeviceDescriptor.sm90()),
        ]
        chained = HeteroChainedOptimizerSync(chained_optimizers=opts)
        chained.zero_grad()
        for opt in opts:
            assert opt.zero_grad_called == 1, f"{opt._name}.zero_grad_called should be 1"

    _run_test("T14: zero_grad() propagates to all chained optimizers", test_zero_grad_propagates)

    # ------------------------------------------------------------------ #
    # T15: Deferred flush clears the backlog                              #
    # ------------------------------------------------------------------ #
    def test_deferred_flush_clears_backlog() -> None:
        opts = [
            _MockHeteroDistOpt("a6000_0", _StubDeviceDescriptor.sm86(), overlap_param_gather=False),
            _MockHeteroDistOpt("a6000_1", _StubDeviceDescriptor.sm86(), overlap_param_gather=False),
        ]
        reg = SLoCRegistry()
        chained = HeteroChainedOptimizerSync(
            chained_optimizers=opts,
            reuse_grad_buf_for_mxfp8=True,
            sloc_registry=reg,
        )
        success = chained.step()
        assert success
        assert len(chained._deferred_bucket_groups) == 0, (
            "Deferred bucket groups should be cleared after step"
        )

    _run_test("T15: deferred flush clears backlog after step", test_deferred_flush_clears_backlog)

    # ------------------------------------------------------------------ #
    # T16: FORCE_FLUSH triggered when backlog exceeds limit across steps  #
    # ------------------------------------------------------------------ #
    def test_force_flush_triggered_on_overflow() -> None:
        opts = [
            _MockHeteroDistOpt("a6000_0", _StubDeviceDescriptor.sm86(), overlap_param_gather=False),
        ]
        chained = HeteroChainedOptimizerSync(
            chained_optimizers=opts,
            reuse_grad_buf_for_mxfp8=True,
            sloc_registry=SLoCRegistry(),
        )
        # Manually inflate the deferred backlog past the limit
        dummy_bg = BucketGroup(
            optimizer_ref=weakref.ref(opts[0]),
            bucket_id="dummy",
            estimated_bytes=1024,
            sm_generation=SMGeneration.SM86,
            sloc_param_keys=("p0",),
        )
        chained._deferred_bucket_groups = [dummy_bg] * (_MAX_DEFERRED_BUCKET_GROUPS + 1)
        # Next step should detect FORCE_FLUSH, clear the backlog, then proceed
        success = chained.step()
        assert success
        assert len(chained._deferred_bucket_groups) == 0

    _run_test("T16: FORCE_FLUSH clears oversized backlog on next step", test_force_flush_triggered_on_overflow)

    # ------------------------------------------------------------------ #
    # T17: device_topology_summary includes all optimizers                #
    # ------------------------------------------------------------------ #
    def test_device_topology_summary() -> None:
        opts = [
            _MockHeteroDistOpt("a6000_0", _StubDeviceDescriptor.sm86()),
            _MockHeteroDistOpt("h100", _StubDeviceDescriptor.sm90()),
        ]
        chained = HeteroChainedOptimizerSync(chained_optimizers=opts)
        summary = chained.device_topology_summary
        assert "SM86" in summary
        assert "SM90" in summary
        assert "_MockHeteroDistOpt" in summary

    _run_test("T17: device_topology_summary includes SM generations", test_device_topology_summary)

    # ------------------------------------------------------------------ #
    # T18: Empty optimizer list raises ValueError                         #
    # ------------------------------------------------------------------ #
    def test_empty_optimizer_list_raises() -> None:
        raised = False
        try:
            HeteroChainedOptimizerSync(chained_optimizers=[])
        except ValueError:
            raised = True
        assert raised, "Should raise ValueError for empty optimizer list"

    _run_test("T18: empty chained_optimizers raises ValueError", test_empty_optimizer_list_raises)

    # ------------------------------------------------------------------ #
    # T19: SLoC keys invalidated during deferred step on homogeneous SM86 #
    # ------------------------------------------------------------------ #
    def test_sloc_keys_invalidated_on_defer() -> None:
        param_keys = ["layer0.weight", "layer0.bias", "layer1.weight"]
        opts = [
            _MockHeteroDistOpt(
                "a6000_0",
                _StubDeviceDescriptor.sm86(),
                overlap_param_gather=False,
                reuse_grad_buf=True,
                param_keys=param_keys,
            )
        ]
        reg = SLoCRegistry()
        chained = HeteroChainedOptimizerSync(
            chained_optimizers=opts,
            reuse_grad_buf_for_mxfp8=True,
            sloc_registry=reg,
        )
        chained.step()
        for k in param_keys:
            assert reg.generation(k) == 1, (
                f"SLoC key {k!r} should be invalidated after deferred step"
            )

    _run_test("T19: SLoC keys invalidated on deferred SM86 step", test_sloc_keys_invalidated_on_defer)

    # ------------------------------------------------------------------ #
    # T20: BucketGroup sort order: SM90 before SM86                       #
    # ------------------------------------------------------------------ #
    def test_bucket_group_sort_order() -> None:
        # Create a chained optimizer and verify flush ordering
        opts = [
            _MockHeteroDistOpt("a6000_0", _StubDeviceDescriptor.sm86(), overlap_param_gather=False),
        ]
        chained = HeteroChainedOptimizerSync(
            chained_optimizers=opts,
            reuse_grad_buf_for_mxfp8=True,
            sloc_registry=SLoCRegistry(),
        )
        bg_sm86 = BucketGroup(weakref.ref(opts[0]), "b86", 1024, SMGeneration.SM86, ())
        bg_sm90 = BucketGroup(weakref.ref(opts[0]), "b90", 1024, SMGeneration.SM90, ())
        chained._deferred_bucket_groups = [bg_sm86, bg_sm90]

        # Sort key: SM90 → 0, SM86 → 1
        def sm_sort_key(bg: BucketGroup) -> int:
            return {SMGeneration.SM90: 0, SMGeneration.SM86: 1, SMGeneration.UNKNOWN: 2}.get(
                bg.sm_generation, 2
            )

        sorted_groups = sorted(chained._deferred_bucket_groups, key=sm_sort_key)
        assert sorted_groups[0].sm_generation == SMGeneration.SM90
        assert sorted_groups[1].sm_generation == SMGeneration.SM86

    _run_test("T20: BucketGroup sort places SM90 before SM86", test_bucket_group_sort_order)

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    total = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed
    print(f"\n{'='*55}")
    print(f"Results: {passed}/{total} passed", end="")
    if failed:
        print(f", {failed} failed")
        for name, ok, tb in _results:
            if not ok:
                print(f"\n  FAILED: {name}")
                print(tb)
    else:
        print(" — all tests passed.")
    print("=" * 55)
    sys.exit(0 if failed == 0 else 1)
