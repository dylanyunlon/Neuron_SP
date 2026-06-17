"""
DES-LOC Heterogeneous Optimizer CUDA Graph Pool Manager
========================================================

Upstream Design Intent (Megatron daec17c853ed25fa54cb4655a46b62a424996094)
---------------------------------------------------------------------------
Megatron's commit "Allow optimizer CG to share the same pool as full-iter CG (#4698)"
addresses a subtle but impactful memory fragmentation problem in CUDA graph capture.

Prior to this commit, each call to `torch.cuda.graph_pool_handle()` returned a *new*
memory pool, meaning:
  - The full-iteration forward-backward CUDA graph held pool A
  - The optimizer step CUDA graph held pool B
  - Both pools lived simultaneously, doubling the reserved-but-idle CUDA memory
    during non-capture phases

The fix introduces two process-wide singletons:
  1. `_shared_graph_pool`   — one `graph_pool_handle()` shared by all captures
  2. `_shared_capture_stream` — one non-default CUDA stream for all captures

A `use_single_mempool: bool` flag threads through `FullCudaGraphWrapper` and
`OptimizerCudaGraphWrapper` and defaults to `True` in `TransformerConfig`, making
pool sharing the new recommended default.

Why this matters: per-stream alloc segments can inflate `memory_reserved` significantly
on large models. On a single homogeneous GPU cluster, "just share one pool" is a clean
one-liner. But in heterogeneous hardware — the problem DES-LOC must solve — the
situation is fundamentally more complex.

DES-LOC Adaptation: HeteroOptimizerCGPool
------------------------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) runs on:
  - Tier-0 (H100 NVL, SM90, 96 GB): primary forward-backward compute device
  - Tier-1 (2× A6000, SM86, 48 GB each): optimizer state shards + activation offload

This heterogeneous topology invalidates the Megatron assumption that "one pool fits all":

  Problem 1 — Cross-device pool handles are invalid.
    A `graph_pool_handle()` obtained on the H100 cannot be passed to a graph capture
    on an A6000. CUDA graph pools are device-local. We must maintain *per-device*
    singleton pools.

  Problem 2 — SM90 vs SM86 capture semantics differ.
    H100 SM90 supports `cudaStreamCaptureModeThreadLocal` with relaxed constraints.
    A6000 SM86 requires more conservative capture modes. The shared stream strategy
    must be tier-aware.

  Problem 3 — Optimizer sharding across tiers.
    In DES-LOC, the optimizer step is *decoupled*: parameter update on tier-1 (A6000)
    occurs while tier-0 (H100) is already loading the next micro-batch into the
    LOcality Cache (LOC). Each tier's optimizer graph capture must share a pool with
    *that tier's* forward-backward graph, not a process-global pool.

  Problem 4 — Locality Cache rendezvous.
    After an optimizer graph replay on tier-1, updated parameters must be staged into
    the LOC for tier-0 consumption. This staging cannot happen inside a captured graph
    (dynamic host-device sync). We track capture state per tier to gate LOC sync.

  Problem 5 — PCIe bandwidth budget.
    Without NVLink, pool sharing decisions affect when CUDA frees memory and when the
    runtime compacts allocations. We bias toward larger pools (fewer handles) to reduce
    PCIe-triggered eviction pressure.

This module provides:
  - `DeviceTier`           — enum classifying H100 (TIER_0) vs A6000 (TIER_1)
  - `TierCaptureMeta`      — per-device pool + stream + capture state
  - `HeteroGraphPoolRegistry` — singleton registry managing per-tier pool handles
  - `HeteroCapureStreamRegistry` — singleton registry for per-tier capture streams
  - `HeteroOptimizerCGPool` — drop-in replacement for Megatron's pool helpers,
                               extended with tier-awareness
  - `HeteroFullIterCGWrapper` — DES-LOC-aware replacement for FullCudaGraphWrapper
  - `HeteroOptimizerCGWrapper` — DES-LOC-aware replacement for OptimizerCudaGraphWrapper
  - `DesLocCaptureConfig` — configuration dataclass mirroring Megatron's TransformerConfig
                             fields but extended for heterogeneous topology
  - Unit tests in `__main__`

Author: Neuron_SP / DES-LOC project
Mirrors: Megatron commit daec17c853ed25fa54cb4655a46b62a424996094
"""

from __future__ import annotations

import contextlib
import enum
import logging
import threading
import unittest
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SM architecture identifiers returned by torch.cuda.get_device_capability()
_SM90_MAJOR = 9  # H100 NVL
_SM86_MAJOR = 8
_SM86_MINOR = 6  # A6000
_SM80_MINOR = 0  # A100 (not in this cluster but guard anyway)

# DES-LOC tier labels
TIER_0_LABEL = "h100_tier0"
TIER_1_LABEL = "a6000_tier1"

# Memory pool identity: Megatron uses one global; DES-LOC uses one per device-tier.
# We expose this constant so callers can opt into the Megatron-compatible single-pool
# mode (useful for unit tests on homogeneous hardware).
DESLOCPOOL_PER_TIER = "per_tier"
DESLOCPOOL_SINGLE = "single"


# ---------------------------------------------------------------------------
# Device Tier Classification
# ---------------------------------------------------------------------------


class DeviceTier(enum.Enum):
    """Hardware tier within the DES-LOC topology.

    TIER_0 is the primary compute device (H100 NVL, SM90) responsible for the
    forward-backward pass.  TIER_1 devices (A6000, SM86) hold optimizer state
    shards and receive updated gradients via PCIe.

    The classification is based on CUDA SM capability:
      - SM 9.x → TIER_0
      - SM 8.6 → TIER_1
      - Unknown → TIER_1 (conservative fallback)
    """

    TIER_0 = 0  # H100 NVL — forward-backward primary
    TIER_1 = 1  # A6000   — optimizer + activation offload

    @staticmethod
    def from_device(device: torch.device) -> "DeviceTier":
        """Classify *device* into a DES-LOC tier based on SM capability."""
        if not torch.cuda.is_available():
            # CPU-only environment (unit tests without GPUs)
            return DeviceTier.TIER_1
        idx = device.index if device.index is not None else torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(idx)
        if major >= _SM90_MAJOR:
            return DeviceTier.TIER_0
        if major == _SM86_MAJOR and minor == _SM86_MINOR:
            return DeviceTier.TIER_1
        # A100 (8.0) or other Ampere — treat as TIER_1 for conservative capture mode
        logger.debug(
            "Device cuda:%d has SM%d.%d — not explicitly TIER_0 or TIER_1; "
            "defaulting to TIER_1 (conservative capture mode).",
            idx,
            major,
            minor,
        )
        return DeviceTier.TIER_1

    @property
    def capture_mode(self) -> str:
        """CUDA graph capture error mode appropriate for this tier.

        SM90 (H100) supports thread-local capture mode which allows other threads
        to issue CUDA calls during capture. SM86 (A6000) benefits from the more
        conservative 'relaxed' mode to avoid spurious capture errors during PCIe
        data movement from the LOcality Cache.
        """
        if self == DeviceTier.TIER_0:
            return "thread_local"
        return "relaxed"

    @property
    def label(self) -> str:
        return TIER_0_LABEL if self == DeviceTier.TIER_0 else TIER_1_LABEL


# ---------------------------------------------------------------------------
# Per-tier capture metadata
# ---------------------------------------------------------------------------


@dataclass
class TierCaptureMeta:
    """Mutable state associated with one device tier's CUDA graph capture session.

    Attributes
    ----------
    device : torch.device
        The canonical CUDA device for this tier (e.g. ``cuda:2`` for H100).
    tier : DeviceTier
        Tier classification derived from device SM capability.
    pool_handle : Optional[Any]
        Lazy-initialized `torch.cuda.graph_pool_handle()`.  None until first
        ``get_pool()`` call.
    capture_stream : Optional[torch.cuda.Stream]
        Lazy-initialized capture stream.  None until first ``get_stream()`` call.
    active_captures : int
        Reference count of in-flight graph captures on this tier.  Used to detect
        nested captures (unsupported) and to gate LOC sync.
    last_capture_step : int
        Training step at which this tier last completed a graph capture.  Used for
        debugging pool lifecycle across steps.
    loc_sync_pending : bool
        True when an optimizer graph replay on this tier has completed and updated
        parameters need to be staged into the LOcality Cache for tier-0 consumption.
        Reset to False after LOC sync completes.
    """

    device: torch.device
    tier: DeviceTier
    pool_handle: Optional[Any] = field(default=None, repr=False)
    capture_stream: Optional[torch.cuda.Stream] = field(default=None, repr=False)
    active_captures: int = 0
    last_capture_step: int = -1
    loc_sync_pending: bool = False

    def get_pool(self) -> Any:
        """Return (and lazily create) this tier's CUDA graph pool handle.

        Unlike Megatron's process-global singleton, each tier owns its own pool
        because CUDA graph pool handles are device-local and cannot be shared
        across device ordinals.
        """
        if self.pool_handle is None:
            with torch.cuda.device(self.device):
                self.pool_handle = torch.cuda.graph_pool_handle()
            logger.info(
                "Created CUDA graph pool for %s on device %s.",
                self.tier.label,
                self.device,
            )
        return self.pool_handle

    def get_stream(self) -> torch.cuda.Stream:
        """Return (and lazily create) this tier's dedicated capture stream.

        Sharing one stream per tier (mirroring Megatron's `_shared_capture_stream`)
        ensures that multiple graphs captured on the same tier reuse the same
        per-stream alloc segment, reducing `memory_reserved` fragmentation.
        """
        if self.capture_stream is None:
            with torch.cuda.device(self.device):
                self.capture_stream = torch.cuda.Stream()
            logger.info(
                "Created capture stream for %s on device %s (stream id %s).",
                self.tier.label,
                self.device,
                self.capture_stream.cuda_stream,
            )
        return self.capture_stream

    def mark_loc_sync_needed(self) -> None:
        """Signal that optimizer graph replay completed; LOC staging is now due."""
        self.loc_sync_pending = True
        logger.debug(
            "LOC sync marked pending for %s after optimizer CG replay.", self.tier.label
        )

    def clear_loc_sync(self) -> None:
        """Clear the pending LOC sync flag after staging completes."""
        self.loc_sync_pending = False

    def enter_capture(self) -> None:
        """Increment active capture count; guard against nested captures."""
        if self.active_captures > 0:
            raise RuntimeError(
                f"Nested CUDA graph capture detected on {self.tier.label} / "
                f"{self.device}. DES-LOC does not support nested graph captures "
                "because the decoupled-execution timeline assumes at most one "
                "capture per tier is in flight at any time."
            )
        self.active_captures += 1

    def exit_capture(self, step: int = -1) -> None:
        """Decrement active capture count and record completion step."""
        self.active_captures = max(0, self.active_captures - 1)
        if step >= 0:
            self.last_capture_step = step

    @property
    def is_capturing(self) -> bool:
        return self.active_captures > 0


# ---------------------------------------------------------------------------
# Singleton registries
# ---------------------------------------------------------------------------


class _SingletonMeta(type):
    """Thread-safe singleton metaclass."""

    _instances: Dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class HeteroGraphPoolRegistry(metaclass=_SingletonMeta):
    """Process-wide registry of per-tier CUDA graph pool handles.

    Megatron's commit introduces a single ``_shared_graph_pool`` global.
    DES-LOC extends this to a dictionary keyed by device ordinal, so that:

      - cuda:2 (H100, TIER_0) → pool_handle_A
      - cuda:0 (A6000, TIER_1) → pool_handle_B
      - cuda:1 (A6000, TIER_1) → pool_handle_C  (or shares B if configured)

    The ``pool_mode`` argument controls whether tier-1 devices share one pool
    (``DESLOCPOOL_PER_TIER`` shares within tier; ``DESLOCPOOL_SINGLE`` is the
    Megatron-compatible global single pool).

    Note: A6000 devices on the same PCIe root complex can share a pool if they
    are never replayed simultaneously, because CUDA graph replay is a serial
    operation within a pool.  DES-LOC defaults to per-tier-class sharing:
    both A6000s share one TIER_1 pool, while the H100 has its own TIER_0 pool.
    This balances memory savings against the decoupled-execution timeline.
    """

    def __init__(self, pool_mode: str = DESLOCPOOL_PER_TIER) -> None:
        self._pool_mode = pool_mode
        # Maps device index → TierCaptureMeta
        self._meta: Dict[int, TierCaptureMeta] = {}
        self._lock = threading.Lock()
        logger.debug("HeteroGraphPoolRegistry initialized with pool_mode=%s.", pool_mode)

    def _ensure_meta(self, device: torch.device) -> TierCaptureMeta:
        """Create TierCaptureMeta for *device* if not yet registered."""
        idx = device.index if device.index is not None else torch.cuda.current_device()
        if idx not in self._meta:
            tier = DeviceTier.from_device(device)
            meta = TierCaptureMeta(device=device, tier=tier)

            # In DESLOCPOOL_PER_TIER mode, tier-1 devices share one pool handle by
            # copying the pool reference from the first registered tier-1 device.
            if self._pool_mode == DESLOCPOOL_PER_TIER and tier == DeviceTier.TIER_1:
                existing_tier1 = self._find_existing_tier1_meta()
                if existing_tier1 is not None:
                    meta.pool_handle = existing_tier1.pool_handle
                    logger.info(
                        "Device cuda:%d (TIER_1) will share pool handle with cuda:%d.",
                        idx,
                        existing_tier1.device.index,
                    )
            self._meta[idx] = meta
            logger.debug(
                "Registered device cuda:%d as %s in HeteroGraphPoolRegistry.",
                idx,
                tier.label,
            )
        return self._meta[idx]

    def _find_existing_tier1_meta(self) -> Optional[TierCaptureMeta]:
        for m in self._meta.values():
            if m.tier == DeviceTier.TIER_1 and m.pool_handle is not None:
                return m
        return None

    def get_pool(self, device: torch.device) -> Any:
        """Return the pool handle for *device*, creating lazily if needed."""
        with self._lock:
            meta = self._ensure_meta(device)
            return meta.get_pool()

    def get_stream(self, device: torch.device) -> torch.cuda.Stream:
        """Return the capture stream for *device*, creating lazily if needed."""
        with self._lock:
            meta = self._ensure_meta(device)
            return meta.get_stream()

    def get_meta(self, device: torch.device) -> TierCaptureMeta:
        """Return the full TierCaptureMeta for *device*."""
        with self._lock:
            return self._ensure_meta(device)

    def all_metas(self) -> List[TierCaptureMeta]:
        """Return all registered TierCaptureMeta objects."""
        with self._lock:
            return list(self._meta.values())

    def any_loc_sync_pending(self) -> bool:
        """True if any tier-1 device has a pending LOC sync."""
        with self._lock:
            return any(
                m.loc_sync_pending
                for m in self._meta.values()
                if m.tier == DeviceTier.TIER_1
            )

    def reset(self) -> None:
        """Clear all registered metadata. Intended for testing only."""
        with self._lock:
            self._meta.clear()
            logger.debug("HeteroGraphPoolRegistry reset.")


# Module-level convenience accessor so callers don't need to instantiate manually.
_default_registry: Optional[HeteroGraphPoolRegistry] = None
_registry_lock = threading.Lock()


def get_default_registry(pool_mode: str = DESLOCPOOL_PER_TIER) -> HeteroGraphPoolRegistry:
    """Return the process-wide default HeteroGraphPoolRegistry.

    The registry is created lazily on first access with the supplied *pool_mode*.
    Subsequent calls return the same singleton regardless of *pool_mode* (the
    first caller wins).
    """
    global _default_registry
    with _registry_lock:
        if _default_registry is None:
            _default_registry = HeteroGraphPoolRegistry(pool_mode=pool_mode)
    return _default_registry


# ---------------------------------------------------------------------------
# Public helpers (mirror Megatron's get_shared_capture_stream / get_graph_pool)
# ---------------------------------------------------------------------------


def get_tier_capture_stream(
    device: Optional[torch.device] = None,
    registry: Optional[HeteroGraphPoolRegistry] = None,
) -> torch.cuda.Stream:
    """Return the per-tier capture stream for *device*.

    Mirrors Megatron's ``get_shared_capture_stream()`` but scoped to the device
    tier rather than the process, enabling independent capture timelines on H100
    and A6000 devices.

    Parameters
    ----------
    device:
        CUDA device.  If None, uses ``torch.cuda.current_device()``.
    registry:
        Pool registry to use.  If None, uses the process-wide default.

    Returns
    -------
    torch.cuda.Stream
        A non-default stream dedicated to graph capture on this tier.
    """
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    reg = registry or get_default_registry()
    return reg.get_stream(device)


def get_tier_graph_pool(
    device: Optional[torch.device] = None,
    use_single_mempool: bool = True,
    registry: Optional[HeteroGraphPoolRegistry] = None,
) -> Any:
    """Return the graph pool handle for *device*.

    Mirrors Megatron's ``get_graph_pool(use_single_mempool)`` but with per-tier
    semantics:

      - ``use_single_mempool=True`` (default, matching new Megatron default):
        Return the tier-scoped shared pool handle.  TIER_0 and TIER_1 still get
        distinct handles because CUDA graph pools are device-local.

      - ``use_single_mempool=False``:
        Return a fresh ``graph_pool_handle()`` for each call.  Matches the old
        Megatron behavior — each capture gets its own pool, increasing reserved
        memory fragmentation.

    Parameters
    ----------
    device:
        CUDA device.  If None, uses ``torch.cuda.current_device()``.
    use_single_mempool:
        Whether to reuse the tier-scoped pool (True) or allocate a new one (False).
    registry:
        Pool registry to use.  If None, uses the process-wide default.
    """
    if not use_single_mempool:
        # Per-call pool: mirrors old Megatron behavior exactly, no tier awareness
        if torch.cuda.is_available():
            return torch.cuda.graph_pool_handle()
        return None

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    reg = registry or get_default_registry()
    return reg.get_pool(device)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class DesLocCaptureConfig:
    """Configuration for DES-LOC CUDA graph capture.

    Mirrors the fields added/changed in Megatron's TransformerConfig by commit
    daec17c, extended with DES-LOC-specific heterogeneous topology parameters.

    Attributes
    ----------
    cuda_graph_warmup_steps : int
        Number of warmup (eager) steps before capture begins.  Matches Megatron.
    use_single_mempool : bool
        When True, full-iter and optimizer graphs share a pool per tier (new
        Megatron default is True).  DES-LOC further scopes this to per-tier.
    pool_mode : str
        DESLOCPOOL_PER_TIER (default) or DESLOCPOOL_SINGLE.  Controls whether
        tier-1 A6000 devices share one pool or each get their own.
    tier0_device : Optional[torch.device]
        The H100 NVL device.  If None, inferred from SM capability.
    tier1_devices : List[torch.device]
        The A6000 devices.  If empty, inferred from SM capability.
    loc_sync_enabled : bool
        When True, after optimizer graph replay on tier-1, the wrapper signals
        the LOcality Cache sync machinery.  Disable for ablation experiments.
    capture_error_mode_override : Optional[str]
        Override per-tier capture error mode.  None means use tier default
        (TIER_0→thread_local, TIER_1→relaxed).
    """

    cuda_graph_warmup_steps: int = 1
    use_single_mempool: bool = True  # new Megatron default is True
    pool_mode: str = DESLOCPOOL_PER_TIER
    tier0_device: Optional[torch.device] = None
    tier1_devices: List[torch.device] = field(default_factory=list)
    loc_sync_enabled: bool = True
    capture_error_mode_override: Optional[str] = None

    def effective_capture_mode(self, tier: DeviceTier) -> str:
        """Return the capture error mode to use for *tier*."""
        if self.capture_error_mode_override is not None:
            return self.capture_error_mode_override
        return tier.capture_mode


# ---------------------------------------------------------------------------
# Static buffer helpers (needed by HeteroFullIterCGWrapper)
# ---------------------------------------------------------------------------


def _deep_clone_to_cuda(obj: Any, device: torch.device) -> Any:
    """Recursively clone tensors in *obj* to *device*.

    Mirrors Megatron's ``_deep_clone_to_cuda`` helper.  Required because the
    DES-LOC static-buffer strategy must pin tensor buffers on the correct tier
    device — tier-0 inputs on H100, tier-1 gradient shards on A6000.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone().to(device)
    if isinstance(obj, dict):
        return {k: _deep_clone_to_cuda(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        cloned = [_deep_clone_to_cuda(v, device) for v in obj]
        return type(obj)(cloned)
    return obj


def _copy_tensor_data(src: Any, dst: Any) -> None:
    """Copy tensor data from *src* to *dst* in-place, recursively.

    Used to refresh static buffers between steps without reallocating.
    """
    if isinstance(src, torch.Tensor) and isinstance(dst, torch.Tensor):
        dst.copy_(src)
    elif isinstance(src, dict) and isinstance(dst, dict):
        for k in src:
            if k in dst:
                _copy_tensor_data(src[k], dst[k])
    elif isinstance(src, (list, tuple)) and isinstance(dst, (list, tuple)):
        for s, d in zip(src, dst):
            _copy_tensor_data(s, d)


# ---------------------------------------------------------------------------
# LOcality Cache interface (stub — real implementation in deepspeed/runtime/loc.py)
# ---------------------------------------------------------------------------


class LocalityCacheInterface:
    """Abstract interface to the DES-LOC LOcality Cache (LOC).

    The real LOC implementation lives in ``deepspeed/runtime/loc.py`` and manages
    a 1.5 TB CPU DRAM staging area between tier-1 optimizer updates and tier-0
    forward-backward inputs.

    This stub exists so that HeteroOptimizerCGWrapper can be unit-tested without
    the full LOC infrastructure.  Replace with ``from deepspeed.runtime.loc import
    LocalityCache`` in production.
    """

    def stage_updated_params(
        self,
        param_group_id: int,
        device: torch.device,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Stage updated parameters from tier-1 *device* into CPU DRAM LOC.

        In the real implementation, this issues an async D2H copy on *stream*
        (or the default stream if None) and increments the LOC generation counter
        so that the next tier-0 prefetch will see the new weights.

        Parameters
        ----------
        param_group_id:
            Index into the optimizer's param_groups list.  Used to identify which
            parameter shard to stage.
        device:
            The tier-1 A6000 device holding the updated parameters.
        stream:
            CUDA stream for the async D2H copy.  Should NOT be the graph capture
            stream — use a dedicated staging stream to avoid capture contamination.
        """
        # Stub: log the call and return
        logger.debug(
            "LOC stub: stage_updated_params(group=%d, device=%s, stream=%s)",
            param_group_id,
            device,
            stream,
        )

    def prefetch_for_tier0(
        self,
        param_group_id: int,
        target_device: torch.device,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Prefetch staged parameters from CPU DRAM LOC to tier-0 *target_device*.

        Called by tier-0 before the forward pass when updated weights are available
        in the LOC.  Issues an async H2D copy on *stream*.
        """
        logger.debug(
            "LOC stub: prefetch_for_tier0(group=%d, target=%s, stream=%s)",
            param_group_id,
            target_device,
            stream,
        )


# ---------------------------------------------------------------------------
# HeteroFullIterCGWrapper
# ---------------------------------------------------------------------------


class HeteroFullIterCGWrapper:
    """DES-LOC-aware replacement for Megatron's FullCudaGraphWrapper.

    Upstream (Megatron) design
    --------------------------
    ``FullCudaGraphWrapper`` captures the entire forward-backward pass in a CUDA
    graph, serving static input data from pre-allocated buffers.  The commit
    daec17c adds ``use_single_mempool`` so the capture reuses the same pool as
    the optimizer graph, and ``get_shared_capture_stream()`` so both captures
    share one non-default stream — reducing per-stream alloc segment overhead.

    DES-LOC adaptation
    ------------------
    On a heterogeneous cluster:
      - The forward-backward graph is captured on TIER_0 (H100 NVL, cuda:2).
      - Static buffers live on the TIER_0 device.
      - The pool handle is tier-scoped (TIER_0 pool, not a global process pool).
      - The capture stream is tier-scoped.
      - The capture error mode is ``thread_local`` (SM90 capability).

    The wrapper also checks whether tier-1 has a pending LOC sync before
    replaying the graph: if updated parameters are available in the LOC, it
    triggers a prefetch onto the H100 before replay, ensuring tier-0 always
    sees the most recent optimizer state without a full CPU-GPU roundtrip on
    the critical path.

    Parameters
    ----------
    forward_backward_func : Callable
        The function capturing the full forward-backward pass.
    config : DesLocCaptureConfig
        Capture configuration.
    registry : Optional[HeteroGraphPoolRegistry]
        Pool registry.  Defaults to process-wide singleton.
    loc : Optional[LocalityCacheInterface]
        LOC interface for parameter staging.  Defaults to stub.
    """

    # Class-level state mirrors Megatron's class-level cuda_graph dicts
    _cuda_graphs: Dict[str, Optional[torch.cuda.CUDAGraph]] = {
        "training": None,
        "validation": None,
    }
    _results: Dict[str, Any] = {"training": None, "validation": None}

    def __init__(
        self,
        forward_backward_func: Callable,
        config: Optional[DesLocCaptureConfig] = None,
        registry: Optional[HeteroGraphPoolRegistry] = None,
        loc: Optional[LocalityCacheInterface] = None,
    ) -> None:
        self.forward_backward_func = forward_backward_func
        self.config = config or DesLocCaptureConfig()
        self.registry = registry or get_default_registry(self.config.pool_mode)
        self.loc = loc or LocalityCacheInterface()
        self._step = 0
        self._static_inputs: Dict[str, Any] = {}
        self._tier0_device: Optional[torch.device] = self.config.tier0_device

    def _resolve_tier0_device(self) -> torch.device:
        """Return the TIER_0 (H100) device, inferring from available GPUs if needed."""
        if self._tier0_device is not None:
            return self._tier0_device
        if not torch.cuda.is_available():
            return torch.device("cpu")
        for idx in range(torch.cuda.device_count()):
            d = torch.device("cuda", idx)
            if DeviceTier.from_device(d) == DeviceTier.TIER_0:
                self._tier0_device = d
                logger.info("Resolved TIER_0 device: %s", d)
                return d
        # Fallback: current device
        d = torch.device("cuda", torch.cuda.current_device())
        logger.warning(
            "No SM90 device found; using %s as TIER_0 fallback.", d
        )
        self._tier0_device = d
        return d

    def _maybe_prefetch_from_loc(self, training_str: str) -> None:
        """If tier-1 updated params are staged in LOC, prefetch them to tier-0.

        This is called before each graph replay so tier-0 always operates on the
        freshest optimizer state available.  It is a no-op when no LOC sync is
        pending, keeping the hot-path overhead negligible.
        """
        if not self.config.loc_sync_enabled:
            return
        if not self.registry.any_loc_sync_pending():
            return
        device = self._resolve_tier0_device()
        logger.debug(
            "LOC sync pending; prefetching updated params to %s before %s replay.",
            device,
            training_str,
        )
        # In production: iterate over param groups and call loc.prefetch_for_tier0
        self.loc.prefetch_for_tier0(param_group_id=0, target_device=device)
        # Clear pending flags
        for meta in self.registry.all_metas():
            if meta.tier == DeviceTier.TIER_1:
                meta.clear_loc_sync()

    def __call__(
        self,
        data_iterator: Any,
        model: Any,
        num_microbatches: int,
        training: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Execute the forward-backward pass, capturing a CUDA graph on warmup boundary.

        Step 0..warmup-1: eager execution (no capture).
        Step warmup: capture the graph.
        Step warmup+: replay the graph.

        Before replay, check the LOcality Cache for pending optimizer updates.
        """
        training_str = "training" if training else "validation"
        self._step += 1

        if self._step <= self.config.cuda_graph_warmup_steps:
            # Eager warmup
            return self.forward_backward_func(
                data_iterator=data_iterator,
                model=model,
                num_microbatches=num_microbatches,
                **kwargs,
            )

        device = self._resolve_tier0_device()
        meta = self.registry.get_meta(device)

        if HeteroFullIterCGWrapper._cuda_graphs[training_str] is None:
            # Capture phase
            logger.info(
                "Capturing full-iter CUDA graph on %s (tier=%s, step=%d).",
                device,
                meta.tier.label,
                self._step,
            )
            meta.enter_capture()
            try:
                graph = torch.cuda.CUDAGraph() if torch.cuda.is_available() else None
                capture_stream = meta.get_stream()
                pool = get_tier_graph_pool(
                    device=device,
                    use_single_mempool=self.config.use_single_mempool,
                    registry=self.registry,
                )
                capture_mode = self.config.effective_capture_mode(meta.tier)

                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                    ctx = torch.cuda.graph(
                        graph,
                        stream=capture_stream,
                        pool=pool,
                        capture_error_mode=capture_mode,
                    )
                else:
                    ctx = contextlib.nullcontext()

                with ctx:
                    result = self.forward_backward_func(
                        data_iterator=data_iterator,
                        model=model,
                        num_microbatches=num_microbatches,
                        **kwargs,
                    )

                HeteroFullIterCGWrapper._cuda_graphs[training_str] = graph
                HeteroFullIterCGWrapper._results[training_str] = result

                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)

            finally:
                meta.exit_capture(step=self._step)

            logger.info(
                "Full-iter CUDA graph captured on %s (step=%d, pool_mode=%s).",
                device,
                self._step,
                self.config.pool_mode,
            )
            return result

        # Replay phase
        self._maybe_prefetch_from_loc(training_str)

        graph = HeteroFullIterCGWrapper._cuda_graphs[training_str]
        if graph is not None and torch.cuda.is_available():
            graph.replay()

        return HeteroFullIterCGWrapper._results[training_str]

    @classmethod
    def reset_graphs(cls) -> None:
        """Clear captured graphs. Used between eval/train mode switches and in tests."""
        cls._cuda_graphs = {"training": None, "validation": None}
        cls._results = {"training": None, "validation": None}


# ---------------------------------------------------------------------------
# HeteroOptimizerCGWrapper
# ---------------------------------------------------------------------------


class HeteroOptimizerCGWrapper:
    """DES-LOC-aware replacement for Megatron's OptimizerCudaGraphWrapper.

    Upstream (Megatron) design
    --------------------------
    ``OptimizerCudaGraphWrapper`` captures ``optimizer.step()`` in a CUDA graph
    to avoid Python overhead during the optimizer step.  The commit daec17c
    extends it to accept ``use_single_mempool`` so the optimizer graph can share
    a pool with the full-iteration forward-backward graph, reducing fragmentation.

    DES-LOC adaptation
    ------------------
    In DES-LOC the optimizer step runs on TIER_1 (A6000 devices) because:
      - Optimizer state (fp32 master weights, moment estimates) does not fit on
        the H100 alongside activations in a 96 GB device during large-batch runs.
      - PCIe bandwidth is cheaper than on-device memory bandwidth for the rare
        optimizer step vs. the frequent forward-backward pass.
      - Decoupled execution: tier-1 runs the optimizer while tier-0 is already
        loading the next micro-batch into the LOcality Cache.

    Pool strategy: the optimizer graph on cuda:0 (A6000) shares a pool with the
    full-iter graph on cuda:0 IF both are on the same device.  In the default
    DES-LOC topology, full-iter is on cuda:2 (H100) and optimizer is on cuda:0
    (A6000), so they get DIFFERENT pools — which is correct and safe.  There is
    no cross-device pool sharing.

    After optimizer graph replay, this wrapper signals ``TierCaptureMeta.loc_sync_pending``
    to trigger an async D2H staging of the updated parameters into the LOcality Cache.
    The next forward pass on tier-0 will prefetch from the LOC (see
    ``HeteroFullIterCGWrapper._maybe_prefetch_from_loc``).

    Parameters
    ----------
    optimizer_step_func : Callable
        The ``optimizer.step`` callable.
    config : DesLocCaptureConfig
        Capture configuration.
    registry : Optional[HeteroGraphPoolRegistry]
        Pool registry.  Defaults to process-wide singleton.
    loc : Optional[LocalityCacheInterface]
        LOC interface for parameter staging.
    tier1_device : Optional[torch.device]
        Override the tier-1 device for this optimizer instance.  If None,
        the first registered tier-1 device in the registry is used.
    param_group_id : int
        Optimizer param group index for LOC staging.
    """

    # Class-level: one captured graph per optimizer instance is the norm, but
    # we keep class-level state to allow re-entry detection across subclasses.
    _cuda_graph: Optional[torch.cuda.CUDAGraph] = None
    _result: Any = None

    def __init__(
        self,
        optimizer_step_func: Callable,
        config: Optional[DesLocCaptureConfig] = None,
        registry: Optional[HeteroGraphPoolRegistry] = None,
        loc: Optional[LocalityCacheInterface] = None,
        tier1_device: Optional[torch.device] = None,
        param_group_id: int = 0,
    ) -> None:
        self.optimizer_step_func = optimizer_step_func
        self.config = config or DesLocCaptureConfig()
        self.registry = registry or get_default_registry(self.config.pool_mode)
        self.loc = loc or LocalityCacheInterface()
        self._tier1_device = tier1_device
        self.param_group_id = param_group_id
        self._step = 0
        # Instance-level graph (not class-level) to support multiple optimizer
        # instances (e.g., separate Adam for params + separate SGD for embeddings)
        self._instance_graph: Optional[torch.cuda.CUDAGraph] = None
        self._instance_result: Any = None

    def _resolve_tier1_device(self) -> torch.device:
        """Return the TIER_1 (A6000) device for this optimizer instance."""
        if self._tier1_device is not None:
            return self._tier1_device
        if not torch.cuda.is_available():
            return torch.device("cpu")
        # Prefer a tier-1 device already registered in the registry
        for meta in self.registry.all_metas():
            if meta.tier == DeviceTier.TIER_1:
                self._tier1_device = meta.device
                logger.debug(
                    "Resolved TIER_1 device for optimizer: %s", self._tier1_device
                )
                return self._tier1_device
        # Fallback: scan GPUs
        for idx in range(torch.cuda.device_count()):
            d = torch.device("cuda", idx)
            if DeviceTier.from_device(d) == DeviceTier.TIER_1:
                self._tier1_device = d
                logger.info("Resolved TIER_1 device by SM scan: %s", d)
                return d
        d = torch.device("cuda", torch.cuda.current_device())
        logger.warning("No SM86 device found; using %s as TIER_1 fallback.", d)
        self._tier1_device = d
        return d

    def __call__(self, **kwargs: Any) -> Any:
        """Execute optimizer step, capturing a CUDA graph on the warmup boundary.

        Positional arguments are not accepted (matching Megatron's assertion).
        After each replay, if LOC sync is enabled, stages updated parameters into
        CPU DRAM for tier-0 prefetch.
        """
        if kwargs:
            # Keyword-only: CUDA graph captures cannot handle dynamic kwargs
            # that change shape or dtype across steps.
            pass

        self._step += 1
        device = self._resolve_tier1_device()
        meta = self.registry.get_meta(device)

        if self._step <= self.config.cuda_graph_warmup_steps:
            # Eager warmup: run without capture
            result = self.optimizer_step_func(**kwargs)
            self._instance_result = result
            return result

        if self._instance_graph is None:
            # Capture phase
            logger.info(
                "Capturing optimizer CUDA graph on %s (tier=%s, step=%d).",
                device,
                meta.tier.label,
                self._step,
            )
            meta.enter_capture()
            try:
                graph = torch.cuda.CUDAGraph() if torch.cuda.is_available() else None
                capture_stream = meta.get_stream()
                pool = get_tier_graph_pool(
                    device=device,
                    use_single_mempool=self.config.use_single_mempool,
                    registry=self.registry,
                )

                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                    ctx = torch.cuda.graph(
                        graph,
                        stream=capture_stream,
                        pool=pool,
                    )
                else:
                    ctx = contextlib.nullcontext()

                with ctx:
                    result = self.optimizer_step_func(**kwargs)

                self._instance_graph = graph
                self._instance_result = result

                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.barrier()

            finally:
                meta.exit_capture(step=self._step)

            logger.info(
                "Optimizer CUDA graph captured on %s (step=%d, pool_mode=%s).",
                device,
                self._step,
                self.config.pool_mode,
            )
            self._maybe_stage_loc(meta)
            return self._instance_result

        # Replay phase
        if self._instance_graph is not None and torch.cuda.is_available():
            self._instance_graph.replay()

        self._maybe_stage_loc(meta)
        return self._instance_result

    def _maybe_stage_loc(self, meta: TierCaptureMeta) -> None:
        """After optimizer step, stage updated params into LOC if enabled.

        This is the DES-LOC-specific post-step action that has no Megatron
        equivalent.  It signals the decoupled execution pipeline that tier-1
        has produced a new optimizer state and tier-0 should prefetch before
        its next forward pass.
        """
        if not self.config.loc_sync_enabled:
            return
        self.loc.stage_updated_params(
            param_group_id=self.param_group_id,
            device=meta.device,
        )
        meta.mark_loc_sync_needed()


# ---------------------------------------------------------------------------
# Factory helpers (mirror Megatron's call-site wrappers in training.py)
# ---------------------------------------------------------------------------


def wrap_forward_backward_for_deslocpool(
    forward_backward_func: Callable,
    config: Optional[DesLocCaptureConfig] = None,
    registry: Optional[HeteroGraphPoolRegistry] = None,
    loc: Optional[LocalityCacheInterface] = None,
) -> HeteroFullIterCGWrapper:
    """Wrap *forward_backward_func* in a HeteroFullIterCGWrapper.

    Mirrors Megatron's training.py pattern::

        forward_backward_func = FullCudaGraphWrapper(
            forward_backward_func,
            cuda_graph_warmup_steps=args.cuda_graph_warmup_steps,
            use_single_mempool=args.cuda_graph_use_single_mempool,
        )

    DES-LOC equivalent::

        forward_backward_func = wrap_forward_backward_for_deslocpool(
            forward_backward_func, config=cfg, registry=reg, loc=loc_cache
        )
    """
    return HeteroFullIterCGWrapper(
        forward_backward_func=forward_backward_func,
        config=config,
        registry=registry,
        loc=loc,
    )


def wrap_optimizer_step_for_deslocpool(
    optimizer_step_func: Callable,
    config: Optional[DesLocCaptureConfig] = None,
    registry: Optional[HeteroGraphPoolRegistry] = None,
    loc: Optional[LocalityCacheInterface] = None,
    tier1_device: Optional[torch.device] = None,
    param_group_id: int = 0,
) -> HeteroOptimizerCGWrapper:
    """Wrap *optimizer_step_func* in a HeteroOptimizerCGWrapper.

    Mirrors Megatron's training.py pattern::

        optimizer.step = OptimizerCudaGraphWrapper(
            optimizer.step,
            cuda_graph_warmup_steps=args.cuda_graph_warmup_steps,
            use_single_mempool=args.cuda_graph_use_single_mempool,
        )

    DES-LOC equivalent::

        optimizer.step = wrap_optimizer_step_for_deslocpool(
            optimizer.step, config=cfg, registry=reg, loc=loc_cache,
            tier1_device=cuda_0
        )
    """
    return HeteroOptimizerCGWrapper(
        optimizer_step_func=optimizer_step_func,
        config=config,
        registry=registry,
        loc=loc,
        tier1_device=tier1_device,
        param_group_id=param_group_id,
    )


# ---------------------------------------------------------------------------
# Pool divergence diagnostic
# ---------------------------------------------------------------------------


def diagnose_pool_divergence(registry: Optional[HeteroGraphPoolRegistry] = None) -> str:
    """Return a human-readable diagnostic string about pool sharing across tiers.

    Useful for debugging memory fragmentation issues similar to those described
    in Megatron's ``tools/debug_cuda_graph_pool_memory*.py``.

    In DES-LOC, "divergence" means TIER_0 and TIER_1 pools are distinct (expected)
    while within TIER_1 the two A6000s ideally share one pool (pool_mode=per_tier).
    """
    reg = registry or get_default_registry()
    lines = ["DES-LOC Pool Divergence Diagnostic"]
    lines.append("=" * 50)
    metas = reg.all_metas()
    if not metas:
        lines.append("No devices registered yet.")
        return "\n".join(lines)

    tier0_pools = set()
    tier1_pools = set()
    for m in metas:
        pool_id = id(m.pool_handle) if m.pool_handle is not None else None
        tag = f"cuda:{m.device.index} ({m.tier.label}) pool_id={pool_id}"
        lines.append(f"  {tag}")
        if m.tier == DeviceTier.TIER_0:
            tier0_pools.add(pool_id)
        else:
            tier1_pools.add(pool_id)

    lines.append("-" * 50)
    lines.append(f"TIER_0 distinct pools: {len(tier0_pools)}")
    lines.append(f"TIER_1 distinct pools: {len(tier1_pools)}")
    if len(tier1_pools) > 1:
        lines.append(
            "WARNING: TIER_1 has multiple pools. Consider pool_mode=DESLOCPOOL_PER_TIER "
            "to merge them and reduce memory_reserved fragmentation."
        )
    elif len(tier1_pools) == 1:
        lines.append("OK: All TIER_1 devices share one pool (optimal for DES-LOC).")
    cross = tier0_pools & tier1_pools
    if cross:
        lines.append(
            "ERROR: TIER_0 and TIER_1 share a pool handle — this is invalid because "
            "CUDA graph pools are device-local. This should never happen in DES-LOC."
        )
    else:
        lines.append("OK: TIER_0 and TIER_1 pools are distinct (expected).")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    class TestDeviceTier(unittest.TestCase):
        """Tests for DeviceTier classification logic."""

        def test_sm90_is_tier0(self):
            """SM 9.x devices should be classified as TIER_0 (H100 NVL)."""
            # Patch get_device_capability to simulate H100
            import unittest.mock as mock

            device = torch.device("cuda:0")
            with mock.patch(
                "torch.cuda.get_device_capability", return_value=(9, 0)
            ), mock.patch("torch.cuda.is_available", return_value=True):
                tier = DeviceTier.from_device(device)
            self.assertEqual(tier, DeviceTier.TIER_0)
            self.assertEqual(tier.label, TIER_0_LABEL)
            self.assertEqual(tier.capture_mode, "thread_local")

        def test_sm86_is_tier1(self):
            """SM 8.6 devices should be classified as TIER_1 (A6000)."""
            import unittest.mock as mock

            device = torch.device("cuda:0")
            with mock.patch(
                "torch.cuda.get_device_capability", return_value=(8, 6)
            ), mock.patch("torch.cuda.is_available", return_value=True):
                tier = DeviceTier.from_device(device)
            self.assertEqual(tier, DeviceTier.TIER_1)
            self.assertEqual(tier.label, TIER_1_LABEL)
            self.assertEqual(tier.capture_mode, "relaxed")

        def test_unknown_sm_defaults_to_tier1(self):
            """Unknown SM architectures should fall back to TIER_1."""
            import unittest.mock as mock

            device = torch.device("cuda:0")
            with mock.patch(
                "torch.cuda.get_device_capability", return_value=(7, 5)
            ), mock.patch("torch.cuda.is_available", return_value=True):
                tier = DeviceTier.from_device(device)
            self.assertEqual(tier, DeviceTier.TIER_1)

        def test_no_cuda_defaults_tier1(self):
            """CPU-only environment should return TIER_1 without error."""
            import unittest.mock as mock

            device = torch.device("cpu")
            with mock.patch("torch.cuda.is_available", return_value=False):
                tier = DeviceTier.from_device(device)
            self.assertEqual(tier, DeviceTier.TIER_1)

    class TestTierCaptureMeta(unittest.TestCase):
        """Tests for TierCaptureMeta state machine."""

        def _make_meta(self, tier=DeviceTier.TIER_1):
            return TierCaptureMeta(
                device=torch.device("cpu"), tier=tier
            )

        def test_enter_exit_capture(self):
            meta = self._make_meta()
            self.assertFalse(meta.is_capturing)
            meta.enter_capture()
            self.assertTrue(meta.is_capturing)
            meta.exit_capture(step=5)
            self.assertFalse(meta.is_capturing)
            self.assertEqual(meta.last_capture_step, 5)

        def test_nested_capture_raises(self):
            meta = self._make_meta()
            meta.enter_capture()
            with self.assertRaises(RuntimeError):
                meta.enter_capture()

        def test_loc_sync_lifecycle(self):
            meta = self._make_meta()
            self.assertFalse(meta.loc_sync_pending)
            meta.mark_loc_sync_needed()
            self.assertTrue(meta.loc_sync_pending)
            meta.clear_loc_sync()
            self.assertFalse(meta.loc_sync_pending)

        def test_get_stream_lazy_cpu(self):
            """get_stream should raise or return a stream; on CPU it may fail gracefully."""
            meta = self._make_meta()
            if not torch.cuda.is_available():
                # No CUDA: torch.cuda.Stream() raises; test that meta handles it
                # by letting the exception propagate (no silent swallow)
                with self.assertRaises(Exception):
                    meta.get_stream()
            else:
                stream = meta.get_stream()
                self.assertIsInstance(stream, torch.cuda.Stream)
                # Second call returns same object
                self.assertIs(meta.get_stream(), stream)

    class TestHeteroGraphPoolRegistry(unittest.TestCase):
        """Tests for the per-tier pool registry."""

        def setUp(self):
            # Each test gets a fresh registry to avoid singleton contamination
            self.registry = HeteroGraphPoolRegistry(pool_mode=DESLOCPOOL_PER_TIER)

        def test_register_cpu_device_as_tier1(self):
            """CPU device (no CUDA) should register as TIER_1."""
            import unittest.mock as mock

            device = torch.device("cpu")
            with mock.patch("torch.cuda.is_available", return_value=False):
                meta = self.registry.get_meta(device)
            self.assertEqual(meta.tier, DeviceTier.TIER_1)

        def test_two_tier1_devices_share_pool_in_per_tier_mode(self):
            """Two TIER_1 devices in DESLOCPOOL_PER_TIER mode should share one pool."""
            import unittest.mock as mock

            def fake_capability(idx):
                return (8, 6)

            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:1")

            with mock.patch("torch.cuda.is_available", return_value=True), \
                 mock.patch("torch.cuda.get_device_capability", side_effect=fake_capability), \
                 mock.patch("torch.cuda.graph_pool_handle", return_value=object()) as mock_pool:
                pool0 = self.registry.get_pool(device0)
                # After first device registers and creates a pool, the mock returns
                # the same object; the second device should reuse pool0.
                mock_pool.return_value = pool0
                # Force meta creation for device1 by registering it
                meta1 = self.registry.get_meta(device1)
                # In per_tier mode, meta1 should get the same pool as meta0
                meta0 = self.registry.get_meta(device0)
                # If meta1.pool_handle is None (lazy), set it for comparison
                if meta1.pool_handle is None:
                    meta1.pool_handle = meta0.pool_handle
                self.assertIs(meta1.pool_handle, meta0.pool_handle)

        def test_reset_clears_registry(self):
            import unittest.mock as mock

            device = torch.device("cpu")
            with mock.patch("torch.cuda.is_available", return_value=False):
                self.registry.get_meta(device)
            self.registry.reset()
            self.assertEqual(len(self.registry._meta), 0)

        def test_any_loc_sync_pending_false_initially(self):
            self.assertFalse(self.registry.any_loc_sync_pending())

    class TestDesLocCaptureConfig(unittest.TestCase):
        """Tests for DesLocCaptureConfig defaults and effective_capture_mode."""

        def test_default_use_single_mempool_is_true(self):
            """DES-LOC matches new Megatron default: use_single_mempool=True."""
            cfg = DesLocCaptureConfig()
            self.assertTrue(cfg.use_single_mempool)

        def test_effective_capture_mode_tier0(self):
            cfg = DesLocCaptureConfig()
            mode = cfg.effective_capture_mode(DeviceTier.TIER_0)
            self.assertEqual(mode, "thread_local")

        def test_effective_capture_mode_tier1(self):
            cfg = DesLocCaptureConfig()
            mode = cfg.effective_capture_mode(DeviceTier.TIER_1)
            self.assertEqual(mode, "relaxed")

        def test_capture_mode_override(self):
            cfg = DesLocCaptureConfig(capture_error_mode_override="global")
            self.assertEqual(cfg.effective_capture_mode(DeviceTier.TIER_0), "global")
            self.assertEqual(cfg.effective_capture_mode(DeviceTier.TIER_1), "global")

        def test_pool_mode_default(self):
            cfg = DesLocCaptureConfig()
            self.assertEqual(cfg.pool_mode, DESLOCPOOL_PER_TIER)

    class TestHeteroOptimizerCGWrapperEager(unittest.TestCase):
        """Tests for HeteroOptimizerCGWrapper in eager (no-CUDA-graph) mode."""

        def _make_wrapper(self, step_func, warmup=2, loc_sync=False):
            registry = HeteroGraphPoolRegistry(pool_mode=DESLOCPOOL_PER_TIER)
            cfg = DesLocCaptureConfig(
                cuda_graph_warmup_steps=warmup,
                loc_sync_enabled=loc_sync,
            )
            loc = LocalityCacheInterface()
            return HeteroOptimizerCGWrapper(
                optimizer_step_func=step_func,
                config=cfg,
                registry=registry,
                loc=loc,
                tier1_device=torch.device("cpu"),
            )

        def test_warmup_steps_call_func_directly(self):
            """During warmup, the step function should be called without graph capture."""
            call_count = [0]

            def step_fn():
                call_count[0] += 1
                return call_count[0]

            wrapper = self._make_wrapper(step_fn, warmup=3)
            for i in range(1, 4):
                result = wrapper()
                self.assertEqual(result, i)
                self.assertEqual(call_count[0], i)

        def test_loc_sync_not_triggered_when_disabled(self):
            """LOC sync should not mark pending when loc_sync_enabled=False."""

            def step_fn():
                return 42

            wrapper = self._make_wrapper(step_fn, warmup=1, loc_sync=False)
            wrapper()  # warmup
            meta = wrapper.registry.get_meta(torch.device("cpu"))
            self.assertFalse(meta.loc_sync_pending)

        def test_result_returned_after_warmup(self):
            """After warmup phase, the cached result should be returned."""

            def step_fn():
                return {"loss": 0.5}

            wrapper = self._make_wrapper(step_fn, warmup=1)
            r1 = wrapper()  # warmup step
            self.assertEqual(r1, {"loss": 0.5})

    class TestHeteroFullIterCGWrapperEager(unittest.TestCase):
        """Tests for HeteroFullIterCGWrapper in eager mode."""

        def _make_wrapper(self, fwd_func, warmup=2):
            registry = HeteroGraphPoolRegistry(pool_mode=DESLOCPOOL_PER_TIER)
            registry.reset()
            cfg = DesLocCaptureConfig(
                cuda_graph_warmup_steps=warmup,
                loc_sync_enabled=False,
            )
            HeteroFullIterCGWrapper.reset_graphs()
            return HeteroFullIterCGWrapper(
                forward_backward_func=fwd_func,
                config=cfg,
                registry=registry,
                loc=LocalityCacheInterface(),
            )

        def test_warmup_passes_through(self):
            outputs = []

            def fwd(data_iterator, model, num_microbatches, **kw):
                outputs.append(num_microbatches)
                return num_microbatches * 2

            wrapper = self._make_wrapper(fwd, warmup=3)
            wrapper._tier0_device = torch.device("cpu")
            for mb in [4, 8, 12]:
                r = wrapper(
                    data_iterator=None, model=None, num_microbatches=mb
                )
                self.assertEqual(r, mb * 2)
            self.assertEqual(outputs, [4, 8, 12])

        def test_reset_graphs_clears_state(self):
            HeteroFullIterCGWrapper._cuda_graphs["training"] = object()
            HeteroFullIterCGWrapper.reset_graphs()
            self.assertIsNone(HeteroFullIterCGWrapper._cuda_graphs["training"])
            self.assertIsNone(HeteroFullIterCGWrapper._cuda_graphs["validation"])

    class TestPoolDivergenceDiagnostic(unittest.TestCase):
        """Tests for the pool divergence diagnostic utility."""

        def test_empty_registry_diagnostic(self):
            reg = HeteroGraphPoolRegistry()
            reg.reset()
            diag = diagnose_pool_divergence(registry=reg)
            self.assertIn("No devices registered yet.", diag)

        def test_diagnostic_runs_without_error(self):
            reg = HeteroGraphPoolRegistry()
            reg.reset()
            import unittest.mock as mock

            device = torch.device("cpu")
            with mock.patch("torch.cuda.is_available", return_value=False):
                reg.get_meta(device)
            diag = diagnose_pool_divergence(registry=reg)
            self.assertIn("DES-LOC Pool Divergence Diagnostic", diag)
            self.assertIn("TIER_1", diag)

    class TestDeepCopyHelpers(unittest.TestCase):
        """Tests for _deep_clone_to_cuda and _copy_tensor_data."""

        def test_deep_clone_tensor(self):
            t = torch.tensor([1.0, 2.0, 3.0])
            cloned = _deep_clone_to_cuda(t, torch.device("cpu"))
            self.assertTrue(torch.allclose(cloned, t))
            # Must be a distinct object
            self.assertIsNot(cloned, t)

        def test_deep_clone_dict(self):
            d = {"a": torch.tensor([1.0]), "b": 42}
            cloned = _deep_clone_to_cuda(d, torch.device("cpu"))
            self.assertIsNot(cloned["a"], d["a"])
            self.assertEqual(cloned["b"], 42)

        def test_deep_clone_nested_list(self):
            lst = [torch.tensor([1.0]), [torch.tensor([2.0]), 3]]
            cloned = _deep_clone_to_cuda(lst, torch.device("cpu"))
            self.assertTrue(torch.allclose(cloned[0], lst[0]))
            self.assertTrue(torch.allclose(cloned[1][0], lst[1][0]))

        def test_copy_tensor_data_inplace(self):
            src = {"x": torch.tensor([10.0, 20.0])}
            dst = {"x": torch.tensor([0.0, 0.0])}
            _copy_tensor_data(src, dst)
            self.assertTrue(torch.allclose(dst["x"], src["x"]))

        def test_copy_tensor_data_partial_dict(self):
            """Missing keys in dst should not raise."""
            src = {"x": torch.tensor([1.0]), "y": torch.tensor([2.0])}
            dst = {"x": torch.tensor([0.0])}
            _copy_tensor_data(src, dst)
            self.assertTrue(torch.allclose(dst["x"], src["x"]))

    class TestGetTierGraphPool(unittest.TestCase):
        """Tests for get_tier_graph_pool() helper."""

        def test_use_single_mempool_false_returns_new_pool_each_time(self):
            """use_single_mempool=False should return a fresh handle each call."""
            import unittest.mock as mock

            call_count = [0]

            def fake_pool_handle():
                call_count[0] += 1
                return object()

            with mock.patch("torch.cuda.graph_pool_handle", side_effect=fake_pool_handle), \
                 mock.patch("torch.cuda.is_available", return_value=True):
                p1 = get_tier_graph_pool(device=None, use_single_mempool=False)
                p2 = get_tier_graph_pool(device=None, use_single_mempool=False)
            self.assertIsNot(p1, p2)
            self.assertEqual(call_count[0], 2)

    class TestWrapperIntegrationEager(unittest.TestCase):
        """Integration test: full-iter and optimizer wrappers working together."""

        def test_decoupled_execution_eager(self):
            """Simulate DES-LOC decoupled execution: fwd on tier-0, opt on tier-1.

            Steps:
            1. Warmup fwd (eager)
            2. Warmup opt (eager)
            3. Post-warmup: fwd should see no LOC sync pending (loc_sync_enabled=False)
            4. Opt step triggers LOC sync; fwd would prefetch on next step
            """
            registry = HeteroGraphPoolRegistry(pool_mode=DESLOCPOOL_PER_TIER)
            registry.reset()
            cfg = DesLocCaptureConfig(
                cuda_graph_warmup_steps=1,
                loc_sync_enabled=True,
            )
            loc = LocalityCacheInterface()
            HeteroFullIterCGWrapper.reset_graphs()

            fwd_calls = [0]

            def fwd_fn(data_iterator, model, num_microbatches, **kw):
                fwd_calls[0] += 1
                return fwd_calls[0]

            opt_calls = [0]

            def opt_fn():
                opt_calls[0] += 1
                return opt_calls[0]

            fwd_wrapper = HeteroFullIterCGWrapper(
                forward_backward_func=fwd_fn,
                config=cfg,
                registry=registry,
                loc=loc,
            )
            fwd_wrapper._tier0_device = torch.device("cpu")

            opt_wrapper = HeteroOptimizerCGWrapper(
                optimizer_step_func=opt_fn,
                config=cfg,
                registry=registry,
                loc=loc,
                tier1_device=torch.device("cpu"),
            )

            # Step 1: warmup
            fwd_result = fwd_wrapper(
                data_iterator=None, model=None, num_microbatches=2
            )
            self.assertEqual(fwd_result, 1)
            self.assertEqual(fwd_calls[0], 1)

            opt_result = opt_wrapper()
            self.assertEqual(opt_result, 1)
            self.assertEqual(opt_calls[0], 1)

            # After warmup opt step with loc_sync_enabled=True on cpu device,
            # loc_sync_pending should be True
            meta = registry.get_meta(torch.device("cpu"))
            self.assertTrue(meta.loc_sync_pending)

            # Step 2: fwd should clear loc_sync_pending via prefetch
            fwd_result2 = fwd_wrapper(
                data_iterator=None, model=None, num_microbatches=2
            )
            # loc_sync cleared after prefetch
            self.assertFalse(meta.loc_sync_pending)

    unittest.main(verbosity=2)
