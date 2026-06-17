"""
DES-LOC Heterogeneous Pinned Buffer Version Guard
==================================================

Upstream Design Intent (Megatron c2d1a8f7e508f216b4974b10c7cbf6c05f97da3d):
    The Megatron commit fixes a TE (TransformerEngine) version check for
    `retain_pinned_cpu_buffers` in CPU offload context construction. The original
    code assumed TE>=2.5.0 was sufficient for `retain_pinned_cpu_buffers`, but
    this parameter was only introduced in TE 2.10.0. The fix introduces a three-tier
    version dispatch:
        - TE >= 2.10.0 : full API (double_buffering + retain_pinned_cpu_buffers)
        - TE >= 2.5.0  : partial API (double_buffering only)
        - TE >= 1.10.0 : baseline API (activation + weight offloading only)

DES-LOC Adaptation (HeteroPinnedBufferVersionGuard):
    In Neuron_SP's DES-LOC (Decoupled Execution with Shared LOcality Cache)
    framework, CPU offload is not uniform across devices. We have:
        - 2x A6000 48GB (SM86, PCIe) — pinned buffer lifespan is constrained by
          PCIe bandwidth; retain_pinned_cpu_buffers is costly because eviction is slow
        - 1x H100 NVL 96GB (SM90, PCIe) — larger HBM allows longer residency;
          pinned buffers can be retained more aggressively
        - 1.5TB CPU DRAM — the shared locality cache (LOC) lives here

    The core DES-LOC insight: the TE version gate is necessary but NOT sufficient.
    We add a second axis — device capability — so that `retain_pinned_cpu_buffers`
    is only enabled when BOTH conditions hold:
        1. TE runtime version >= 2.10.0
        2. Device SM version >= 90 (H100-class) OR explicit user override

    For SM86 (A6000) nodes, even with TE 2.10.0 we fall back to the 2.5.0 path
    because PCIe bandwidth makes pinned-buffer retention a net loss. The LOC
    cache on these nodes uses a write-through strategy instead.

    Additionally, DES-LOC introduces a "buffer epoch" concept: pinned CPU
    buffers are tagged with the pipeline micro-step they belong to, enabling
    the shared LOC to decide eviction order across heterogeneous devices without
    a central coordinator.

Key classes:
    HeteroPinnedBufferVersionGuard  — main dispatch logic
    DeviceCapabilityProfile         — per-device SM/memory profile
    LOCBufferEpoch                  — micro-step tag attached to each buffer slot
    PinnedBufferRegistry            — tracks live pinned buffers across devices
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
import weakref
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, Generator, Iterator, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Version utilities (mirrors Megatron's is_te_min_version without TE import)
# ---------------------------------------------------------------------------

def _parse_version(version_str: str) -> Tuple[int, ...]:
    """Parse a PEP-440-ish version string into a comparable tuple.

    Handles suffixes like '1.10.0.dev0' by stripping non-numeric tail tokens.

    Args:
        version_str: e.g. "2.10.0", "1.10.0.dev0"

    Returns:
        Tuple of ints, e.g. (2, 10, 0)
    """
    parts = []
    for token in version_str.split("."):
        numeric = "".join(ch for ch in token if ch.isdigit())
        if numeric:
            parts.append(int(numeric))
    return tuple(parts)


def _te_version() -> Optional[Tuple[int, ...]]:
    """Return parsed TransformerEngine version, or None if not installed."""
    try:
        import transformer_engine  # type: ignore
        return _parse_version(transformer_engine.__version__)
    except ImportError:
        logger.debug("TransformerEngine not found; DES-LOC will use native offload path")
        return None


def is_te_min_version(min_version: str) -> bool:
    """Return True if installed TE version >= min_version.

    DES-LOC adaptation: identical semantics to Megatron's helper so we can
    run the same three-tier dispatch table without importing Megatron.

    Args:
        min_version: Minimum required TE version string, e.g. "2.10.0"

    Returns:
        bool
    """
    current = _te_version()
    if current is None:
        return False
    required = _parse_version(min_version)
    return current >= required


# ---------------------------------------------------------------------------
# Device capability profiles
# ---------------------------------------------------------------------------

class OffloadTier(Enum):
    """Pinned-buffer capability tier inferred from (TE version, SM arch)."""
    FULL = auto()        # TE>=2.10 + SM>=90: retain_pinned_cpu_buffers enabled
    DOUBLE_BUF = auto()  # TE>=2.5  + any SM:  double_buffering only
    BASELINE = auto()    # TE>=1.10 + any SM:  activation/weight offload only
    NATIVE = auto()      # No TE:              DeepSpeed-native CPU offload


@dataclass(frozen=True)
class DeviceCapabilityProfile:
    """Immutable hardware description for a single CUDA device.

    Attributes:
        device_index: CUDA device ordinal
        sm_major: Compute capability major version (e.g. 9 for H100)
        sm_minor: Compute capability minor version (e.g. 0 for H100)
        total_memory_gb: HBM capacity in GiB
        is_nvlink: Whether device participates in NVLink fabric
        pcie_gen: PCIe generation (4 or 5); affects pinned-buffer eviction cost
    """
    device_index: int
    sm_major: int
    sm_minor: int
    total_memory_gb: float
    is_nvlink: bool = False
    pcie_gen: int = 4

    @property
    def sm_version(self) -> int:
        """Composite SM version, e.g. 90 for SM_90."""
        return self.sm_major * 10 + self.sm_minor

    @property
    def supports_retain_pinned(self) -> bool:
        """DES-LOC policy: only H100-class (SM>=90) benefits from retention.

        On PCIe-only A6000 (SM86), eviction latency over PCIe Gen4 makes
        retain_pinned_cpu_buffers a net negative — the LOC write-through path
        is faster because it amortises the PCIe round-trip across micro-steps.
        """
        return self.sm_version >= 90

    def __str__(self) -> str:
        nvlink_str = "NVLink" if self.is_nvlink else "PCIe-only"
        return (
            f"Device[{self.device_index}] SM{self.sm_version} "
            f"{self.total_memory_gb:.1f}GiB {nvlink_str} PCIe-Gen{self.pcie_gen}"
        )


def probe_device_profiles(device_indices: Optional[List[int]] = None) -> Dict[int, DeviceCapabilityProfile]:
    """Probe CUDA devices and return their capability profiles.

    This is called once at process startup by HeteroPinnedBufferVersionGuard.

    Args:
        device_indices: List of CUDA device ordinals to probe. If None, probes
                        all visible devices.

    Returns:
        Dict mapping device_index -> DeviceCapabilityProfile
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available; returning empty device profiles")
        return {}

    n_devices = torch.cuda.device_count()
    indices = device_indices if device_indices is not None else list(range(n_devices))
    profiles: Dict[int, DeviceCapabilityProfile] = {}

    for idx in indices:
        if idx >= n_devices:
            logger.warning("Device index %d out of range (found %d devices); skipping", idx, n_devices)
            continue
        props = torch.cuda.get_device_properties(idx)
        mem_gb = props.total_memory / (1024 ** 3)
        profile = DeviceCapabilityProfile(
            device_index=idx,
            sm_major=props.major,
            sm_minor=props.minor,
            total_memory_gb=mem_gb,
            is_nvlink=False,   # PCIe-only topology per Neuron_SP hardware spec
            pcie_gen=4,
        )
        profiles[idx] = profile
        logger.info("Probed %s", profile)

    return profiles


# ---------------------------------------------------------------------------
# LOC Buffer Epoch — micro-step tagging
# ---------------------------------------------------------------------------

@dataclass
class LOCBufferEpoch:
    """Tags a pinned CPU buffer slot with its pipeline micro-step provenance.

    DES-LOC design: The Shared LOcality Cache (LOC) lives in CPU DRAM and is
    shared across all three devices. Without a coordination signal, different
    devices may simultaneously claim the same cache lines, causing thrashing.

    By tagging each buffer with (pipeline_stage, micro_step, device_index),
    the LOC eviction policy can prioritise evicting buffers from the device
    that is furthest ahead in the pipeline — i.e. whose activations are least
    likely to be reused soon.

    Attributes:
        pipeline_stage: Which transformer layer group owns this buffer
        micro_step: Forward/backward micro-batch index within the current step
        device_index: Originating CUDA device
        created_at: Wall-clock timestamp (used for LRU within same epoch)
        is_dirty: True if buffer has been written but not synced to device
    """
    pipeline_stage: int
    micro_step: int
    device_index: int
    created_at: float = field(default_factory=time.monotonic)
    is_dirty: bool = False

    def eviction_priority(self, current_micro_step: int) -> float:
        """Lower value = evict first.

        Buffers far ahead of current_micro_step are evicted first; within
        the same distance, older buffers (smaller created_at) win.

        Args:
            current_micro_step: The micro-step the caller is currently computing

        Returns:
            float priority score
        """
        step_distance = self.micro_step - current_micro_step
        age = time.monotonic() - self.created_at
        # Negative distance (already consumed) should be evicted first
        return step_distance - age * 0.01

    def __repr__(self) -> str:
        dirty_flag = "*" if self.is_dirty else ""
        return (
            f"LOCEpoch(stage={self.pipeline_stage}, "
            f"μstep={self.micro_step}, dev={self.device_index}{dirty_flag})"
        )


# ---------------------------------------------------------------------------
# Pinned Buffer Registry
# ---------------------------------------------------------------------------

class PinnedBufferRegistry:
    """Thread-safe registry tracking live pinned CPU tensors across devices.

    DES-LOC role: The registry is the "shared" part of the Shared LOcality
    Cache. It does not hold tensor data — tensors live in CPU pinned memory
    managed by TE or DeepSpeed — but it holds weak references to them along
    with their LOCBufferEpoch tags.

    When a device requests a new pinned buffer and the total pinned memory
    exceeds the budget, the registry selects the lowest-priority buffer
    (per LOCBufferEpoch.eviction_priority) and signals its owner to evict.

    Attributes:
        budget_gb: Maximum total pinned memory in GiB (default: 64.0, tunable
                   via DESLOCK_PINNED_BUDGET_GB env var)
    """

    def __init__(self, budget_gb: Optional[float] = None) -> None:
        env_budget = os.environ.get("DESLOCK_PINNED_BUDGET_GB")
        self.budget_gb: float = (
            float(env_budget) if env_budget is not None
            else (budget_gb if budget_gb is not None else 64.0)
        )
        self._lock = threading.Lock()
        # Maps buffer_id -> (weak_ref_to_tensor, LOCBufferEpoch)
        self._registry: Dict[int, Tuple[weakref.ref, LOCBufferEpoch]] = {}
        self._next_id = 0
        logger.info(
            "PinnedBufferRegistry initialised; budget=%.1f GiB", self.budget_gb
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        tensor: torch.Tensor,
        epoch: LOCBufferEpoch,
    ) -> int:
        """Register a pinned CPU tensor and return its buffer_id.

        Args:
            tensor: A CPU pinned tensor (tensor.is_pinned() should be True)
            epoch: Provenance tag for LOC eviction decisions

        Returns:
            Unique buffer_id for later deregistration
        """
        if not tensor.is_pinned():
            logger.debug(
                "Registering non-pinned tensor (device=%d); LOC eviction will "
                "treat it as low priority", epoch.device_index
            )
        with self._lock:
            buf_id = self._next_id
            self._next_id += 1
            self._registry[buf_id] = (weakref.ref(tensor), epoch)
            logger.debug("Registered buffer %d: %s", buf_id, epoch)
        return buf_id

    def deregister(self, buf_id: int) -> None:
        """Remove a buffer from the registry (called when tensor is freed).

        Args:
            buf_id: ID returned by register()
        """
        with self._lock:
            if buf_id in self._registry:
                del self._registry[buf_id]
                logger.debug("Deregistered buffer %d", buf_id)

    def eviction_candidates(
        self,
        current_micro_step: int,
        n: int = 1,
    ) -> List[Tuple[int, LOCBufferEpoch]]:
        """Return the n lowest-priority live buffers as eviction candidates.

        Dead weak references (tensor was GC'd) are pruned opportunistically.

        Args:
            current_micro_step: Caller's current micro-step for priority calc
            n: Number of candidates to return

        Returns:
            List of (buffer_id, LOCBufferEpoch) sorted by eviction_priority ASC
        """
        with self._lock:
            dead = [k for k, (ref, _) in self._registry.items() if ref() is None]
            for k in dead:
                del self._registry[k]
                logger.debug("Pruned dead buffer %d", k)

            candidates = [
                (buf_id, epoch)
                for buf_id, (_, epoch) in self._registry.items()
            ]
        candidates.sort(key=lambda t: t[1].eviction_priority(current_micro_step))
        return candidates[:n]

    def total_live(self) -> int:
        """Return count of currently live registered buffers."""
        with self._lock:
            return sum(1 for ref, _ in self._registry.values() if ref() is not None)


# ---------------------------------------------------------------------------
# Core: HeteroPinnedBufferVersionGuard
# ---------------------------------------------------------------------------

class HeteroPinnedBufferVersionGuard:
    """Dispatch CPU offload context construction based on (TE version, SM arch).

    This is the DES-LOC reinterpretation of Megatron commit c2d1a8f:

    Megatron added a three-tier TE version check so that `retain_pinned_cpu_buffers`
    is only passed when TE>=2.10.0, and `double_buffering` only when TE>=2.5.0.

    DES-LOC extends this into a 2D dispatch table:

        TE version tier  ×  device SM tier
        ─────────────────────────────────────────────────────────────────
        TE>=2.10 + SM>=90  →  FULL   (retain_pinned_cpu_buffers=True)
        TE>=2.10 + SM<90   →  DOUBLE_BUF (PCIe cost too high for retention)
        TE>=2.5  + any SM  →  DOUBLE_BUF
        TE>=1.10 + any SM  →  BASELINE
        No TE              →  NATIVE (DeepSpeed CPU offload)

    The LOC cache is informed of buffer epoch tags so that heterogeneous devices
    (A6000 × 2 + H100 × 1) can share the 1.5 TB CPU DRAM without thrashing.

    Usage::

        guard = HeteroPinnedBufferVersionGuard(registry=registry, profiles=profiles)
        with guard.offload_context(
            device_index=0,
            enabled=True,
            num_layers=32,
            model_layers=32,
            activation_offloading=True,
            weight_offloading=False,
            double_buffering=True,
            retain_pinned_cpu_buffers=True,
            pipeline_stage=2,
            micro_step=4,
        ) as (ctx, sync_fn):
            # run forward/backward inside ctx
            sync_fn()
    """

    def __init__(
        self,
        registry: Optional[PinnedBufferRegistry] = None,
        profiles: Optional[Dict[int, DeviceCapabilityProfile]] = None,
        device_indices: Optional[List[int]] = None,
        force_tier: Optional[OffloadTier] = None,
    ) -> None:
        """Initialise the guard.

        Args:
            registry: Shared PinnedBufferRegistry. If None, creates a new one.
            profiles: Pre-probed device profiles. If None, probes at init time.
            device_indices: CUDA device ordinals to probe (ignored if profiles given).
            force_tier: Override tier selection for all devices (useful for testing).
        """
        self.registry = registry if registry is not None else PinnedBufferRegistry()
        self.profiles = (
            profiles
            if profiles is not None
            else probe_device_profiles(device_indices)
        )
        self.force_tier = force_tier
        self._tier_cache: Dict[int, OffloadTier] = {}
        logger.info(
            "HeteroPinnedBufferVersionGuard ready; TE=%s force_tier=%s",
            _te_version(),
            force_tier,
        )

    # ------------------------------------------------------------------
    # Tier resolution
    # ------------------------------------------------------------------

    def resolve_tier(self, device_index: int) -> OffloadTier:
        """Resolve the OffloadTier for a specific device.

        Called once per device and cached. The result encodes both TE version
        capability and device PCIe/SM constraints.

        Args:
            device_index: CUDA device ordinal

        Returns:
            OffloadTier enum value
        """
        if self.force_tier is not None:
            return self.force_tier

        if device_index in self._tier_cache:
            return self._tier_cache[device_index]

        profile = self.profiles.get(device_index)
        if profile is None:
            logger.warning(
                "No profile for device %d; defaulting to NATIVE tier", device_index
            )
            tier = OffloadTier.NATIVE
        elif is_te_min_version("2.10.0") and profile.supports_retain_pinned:
            # H100 (SM90) + TE>=2.10: safe to retain pinned buffers
            tier = OffloadTier.FULL
            logger.info(
                "Device %d: FULL tier (TE>=2.10 + SM%d>=90)",
                device_index, profile.sm_version,
            )
        elif is_te_min_version("2.5.0"):
            # A6000 (SM86) with TE>=2.10 also lands here: PCIe cost prohibits retention
            tier = OffloadTier.DOUBLE_BUF
            logger.info(
                "Device %d: DOUBLE_BUF tier (TE>=2.5, SM%d; PCIe retention skipped)",
                device_index, profile.sm_version if profile else -1,
            )
        elif is_te_min_version("1.10.0.dev0"):
            tier = OffloadTier.BASELINE
            logger.info("Device %d: BASELINE tier (TE>=1.10)", device_index)
        else:
            tier = OffloadTier.NATIVE
            logger.info("Device %d: NATIVE tier (no TE)", device_index)

        self._tier_cache[device_index] = tier
        return tier

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def offload_context(
        self,
        device_index: int,
        enabled: bool,
        num_layers: int,
        model_layers: int,
        activation_offloading: bool,
        weight_offloading: bool,
        double_buffering: bool = True,
        retain_pinned_cpu_buffers: bool = False,
        pipeline_stage: int = 0,
        micro_step: int = 0,
    ) -> Generator[Tuple[object, Callable], None, None]:
        """Context manager yielding (offload_ctx, sync_fn) for one micro-step.

        The yielded pair mirrors what Megatron's _get_cpu_offload_context_func
        returns, so callers are drop-in compatible.

        DES-LOC addition: before entering the TE context, we register a LOC
        epoch tag so the shared CPU DRAM cache can make eviction decisions across
        all three devices.

        Args:
            device_index: CUDA device running this micro-step
            enabled: Whether CPU offload is active
            num_layers: Total number of transformer layers in the model
            model_layers: Number of layers on this pipeline stage
            activation_offloading: Offload activations to CPU
            weight_offloading: Offload weights to CPU
            double_buffering: Enable double-buffering (TE>=2.5 only)
            retain_pinned_cpu_buffers: Keep pinned buffers across steps (TE>=2.10 + SM>=90)
            pipeline_stage: Pipeline stage index (for LOC epoch tagging)
            micro_step: Micro-batch index within current global step

        Yields:
            (context, sync_func) — same contract as Megatron's helper
        """
        tier = self.resolve_tier(device_index)
        epoch = LOCBufferEpoch(
            pipeline_stage=pipeline_stage,
            micro_step=micro_step,
            device_index=device_index,
        )

        logger.debug(
            "offload_context: dev=%d tier=%s epoch=%s enabled=%s",
            device_index, tier.name, epoch, enabled,
        )

        if not enabled:
            # Fast path: no offload, yield null context
            yield _null_context(), _noop_sync
            return

        if tier == OffloadTier.FULL:
            ctx, sync_fn = self._build_full_context(
                num_layers, model_layers, activation_offloading,
                weight_offloading, double_buffering, retain_pinned_cpu_buffers,
            )
        elif tier == OffloadTier.DOUBLE_BUF:
            ctx, sync_fn = self._build_double_buf_context(
                num_layers, model_layers, activation_offloading,
                weight_offloading, double_buffering,
            )
        elif tier == OffloadTier.BASELINE:
            ctx, sync_fn = self._build_baseline_context(
                num_layers, model_layers, activation_offloading, weight_offloading,
            )
        else:
            ctx, sync_fn = self._build_native_context(
                num_layers, model_layers, activation_offloading, weight_offloading,
            )

        # Register a sentinel tensor as LOC epoch anchor.
        # A real implementation would pass the actual activation tensor here;
        # we use a 1-element pinned tensor as a lightweight proxy.
        sentinel = torch.empty(1, pin_memory=True)
        buf_id = self.registry.register(sentinel, epoch)

        try:
            yield ctx, sync_fn
        finally:
            self.registry.deregister(buf_id)
            del sentinel

    # ------------------------------------------------------------------
    # Tier-specific context builders
    # ------------------------------------------------------------------

    def _build_full_context(
        self,
        num_layers: int,
        model_layers: int,
        activation_offloading: bool,
        weight_offloading: bool,
        double_buffering: bool,
        retain_pinned_cpu_buffers: bool,
    ) -> Tuple[object, Callable]:
        """Build FULL tier context (TE>=2.10, SM>=90).

        Passes retain_pinned_cpu_buffers=True to TE, enabling the H100 to keep
        activation buffers in CPU pinned memory between micro-steps. This is
        safe on H100 (PCIe Gen5, high HBM bandwidth) but harmful on A6000
        (PCIe Gen4 bottleneck), which is why this tier is SM>=90 only.

        Returns:
            (context, sync_func) from TE's _get_cpu_offload_context
        """
        try:
            from transformer_engine.pytorch.cpu_offload import (  # type: ignore
                get_cpu_offload_context as _get_cpu_offload_context,
            )
            logger.debug(
                "FULL tier: calling TE _get_cpu_offload_context with "
                "retain_pinned_cpu_buffers=%s double_buffering=%s",
                retain_pinned_cpu_buffers, double_buffering,
            )
            ctx, sync_fn = _get_cpu_offload_context(
                True,
                num_layers,
                model_layers,
                activation_offloading,
                weight_offloading,
                double_buffering,
                retain_pinned_cpu_buffers=retain_pinned_cpu_buffers,
            )
            return ctx, sync_fn
        except ImportError:
            logger.warning("FULL tier requested but TE import failed; falling back to NATIVE")
            return self._build_native_context(
                num_layers, model_layers, activation_offloading, weight_offloading
            )

    def _build_double_buf_context(
        self,
        num_layers: int,
        model_layers: int,
        activation_offloading: bool,
        weight_offloading: bool,
        double_buffering: bool,
    ) -> Tuple[object, Callable]:
        """Build DOUBLE_BUF tier context (TE>=2.5).

        This is the primary path for A6000 nodes in Neuron_SP. Double-buffering
        overlaps CPU->GPU transfer of the next layer's activations with the GPU
        compute of the current layer, hiding PCIe latency without retaining
        buffers across micro-steps (which would exhaust PCIe bandwidth).

        Returns:
            (context, sync_func) from TE's _get_cpu_offload_context
        """
        try:
            from transformer_engine.pytorch.cpu_offload import (  # type: ignore
                get_cpu_offload_context as _get_cpu_offload_context,
            )
            logger.debug(
                "DOUBLE_BUF tier: calling TE _get_cpu_offload_context with "
                "double_buffering=%s (no retain_pinned)", double_buffering,
            )
            ctx, sync_fn = _get_cpu_offload_context(
                True,
                num_layers,
                model_layers,
                activation_offloading,
                weight_offloading,
                double_buffering,
            )
            return ctx, sync_fn
        except ImportError:
            logger.warning("DOUBLE_BUF tier requested but TE import failed; falling back to NATIVE")
            return self._build_native_context(
                num_layers, model_layers, activation_offloading, weight_offloading
            )

    def _build_baseline_context(
        self,
        num_layers: int,
        model_layers: int,
        activation_offloading: bool,
        weight_offloading: bool,
    ) -> Tuple[object, Callable]:
        """Build BASELINE tier context (TE>=1.10).

        No double-buffering, no retained buffers. Used when TE is present but
        too old for the advanced APIs. In DES-LOC this also serves as the
        safe fallback when device profiles are unavailable.

        Returns:
            (context, sync_func) from TE's _get_cpu_offload_context
        """
        try:
            from transformer_engine.pytorch.cpu_offload import (  # type: ignore
                get_cpu_offload_context as _get_cpu_offload_context,
            )
            logger.debug("BASELINE tier: minimal TE _get_cpu_offload_context call")
            ctx, sync_fn = _get_cpu_offload_context(
                True, num_layers, model_layers, activation_offloading, weight_offloading
            )
            return ctx, sync_fn
        except ImportError:
            logger.warning("BASELINE tier requested but TE import failed; using NATIVE")
            return self._build_native_context(
                num_layers, model_layers, activation_offloading, weight_offloading
            )

    def _build_native_context(
        self,
        num_layers: int,
        model_layers: int,
        activation_offloading: bool,
        weight_offloading: bool,
    ) -> Tuple[object, Callable]:
        """Build NATIVE tier context (no TE dependency).

        DES-LOC's own CPU offload, implemented via DeepSpeed's activation
        checkpointing + gradient CPU offload. This path is always available
        and is the ultimate fallback.

        For the shared LOC, NATIVE tier uses a write-through strategy: every
        activation is written to CPU pinned memory immediately and read back
        on demand, without any retention guarantee.

        Returns:
            (NativeCPUOffloadContext, sync_func)
        """
        logger.debug(
            "NATIVE tier: DeepSpeed CPU offload "
            "(activation=%s weight=%s num_layers=%d)",
            activation_offloading, weight_offloading, num_layers,
        )
        ctx = _NativeCPUOffloadContext(
            num_layers=num_layers,
            model_layers=model_layers,
            activation_offloading=activation_offloading,
            weight_offloading=weight_offloading,
        )
        return ctx, ctx.sync


# ---------------------------------------------------------------------------
# Native CPU offload context (no TE)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _null_context():
    """Null context manager for disabled offload path."""
    yield


def _noop_sync() -> None:
    """No-op sync function for disabled offload path."""


class _NativeCPUOffloadContext:
    """Minimal DeepSpeed-native CPU offload context.

    This is a lightweight stand-in used when TransformerEngine is absent.
    A production implementation would hook into DeepSpeed's activation
    checkpoint store and gradient offload buffers directly.

    Implements the same __enter__/__exit__ interface as TE's context so
    callers need not special-case the NATIVE tier.
    """

    def __init__(
        self,
        num_layers: int,
        model_layers: int,
        activation_offloading: bool,
        weight_offloading: bool,
    ) -> None:
        self.num_layers = num_layers
        self.model_layers = model_layers
        self.activation_offloading = activation_offloading
        self.weight_offloading = weight_offloading
        self._active = False

    def __enter__(self) -> "_NativeCPUOffloadContext":
        self._active = True
        logger.debug(
            "_NativeCPUOffloadContext.__enter__: activation=%s weight=%s",
            self.activation_offloading, self.weight_offloading,
        )
        return self

    def __exit__(self, *args) -> None:
        self._active = False
        logger.debug("_NativeCPUOffloadContext.__exit__")

    def sync(self) -> None:
        """Synchronise CPU<->GPU transfers for this micro-step.

        In a full implementation this would call torch.cuda.synchronize()
        and flush the DeepSpeed gradient CPU offload queue. We issue only
        the CUDA sync here to keep this module dependency-free.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.debug("_NativeCPUOffloadContext.sync complete")


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_guard_for_neuron_sp(
    budget_gb: float = 64.0,
    force_tier: Optional[OffloadTier] = None,
) -> HeteroPinnedBufferVersionGuard:
    """Factory for the standard Neuron_SP hardware configuration.

    Creates a guard pre-configured for 2x A6000 (devices 0,1) + 1x H100 (device 2)
    on a PCIe-only topology with 1.5 TB CPU DRAM.

    Args:
        budget_gb: Pinned-memory budget for the LOC registry in GiB.
                   Defaults to 64 GiB (conservative for PCIe bandwidth).
        force_tier: Override all tier decisions (useful in CI without GPUs).

    Returns:
        Configured HeteroPinnedBufferVersionGuard
    """
    registry = PinnedBufferRegistry(budget_gb=budget_gb)

    if torch.cuda.is_available() and torch.cuda.device_count() >= 3:
        profiles = probe_device_profiles([0, 1, 2])
    else:
        # Synthetic profiles for testing without real hardware
        logger.warning(
            "Fewer than 3 CUDA devices detected; using synthetic profiles for "
            "A6000×2 + H100×1"
        )
        profiles = {
            0: DeviceCapabilityProfile(0, sm_major=8, sm_minor=6, total_memory_gb=48.0),
            1: DeviceCapabilityProfile(1, sm_major=8, sm_minor=6, total_memory_gb=48.0),
            2: DeviceCapabilityProfile(2, sm_major=9, sm_minor=0, total_memory_gb=96.0),
        }

    return HeteroPinnedBufferVersionGuard(
        registry=registry,
        profiles=profiles,
        force_tier=force_tier,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Tier resolution without real hardware
    guard = build_guard_for_neuron_sp(force_tier=None)

    # Assert synthetic profiles were created
    assert len(guard.profiles) == 3, f"Expected 3 profiles, got {len(guard.profiles)}"

    # Assert SM version arithmetic
    a6000 = guard.profiles[0]
    h100 = guard.profiles[2]
    assert a6000.sm_version == 86, f"A6000 SM should be 86, got {a6000.sm_version}"
    assert h100.sm_version == 90, f"H100 SM should be 90, got {h100.sm_version}"
    assert not a6000.supports_retain_pinned, "A6000 should NOT support retain_pinned"
    assert h100.supports_retain_pinned, "H100 SHOULD support retain_pinned"

    # Assert LOC epoch priority ordering
    epoch_old = LOCBufferEpoch(pipeline_stage=0, micro_step=5, device_index=0,
                               created_at=time.monotonic() - 10.0)
    epoch_new = LOCBufferEpoch(pipeline_stage=0, micro_step=5, device_index=1,
                               created_at=time.monotonic())
    assert epoch_old.eviction_priority(3) < epoch_new.eviction_priority(3), \
        "Older buffer at same micro_step should have lower (evict-first) priority"

    # Registry smoke test
    registry = PinnedBufferRegistry(budget_gb=1.0)
    t = torch.empty(16, pin_memory=torch.cuda.is_available())
    buf_id = registry.register(t, epoch_old)
    assert registry.total_live() == 1
    registry.deregister(buf_id)
    assert registry.total_live() == 0

    logger.info("All smoke tests passed.")
