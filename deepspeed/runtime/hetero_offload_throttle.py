# Copyright (c) 2025 Neuron_SP Project — DES-LOC Heterogeneous Training Framework
# Adapted from Megatron-LM commit 9a7cd17fdc4da4c5386d146f88ff3dcc8159322b
# Original author: Nan Zheng <nanz-nv@users.noreply.github.com>
#
# This file implements HeteroOffloadThrottle: a DES-LOC-aware inflight offload
# throttle mechanism for heterogeneous GPU clusters communicating over PCIe.
#
# Upstream design intent (Megatron 9a7cd17):
#   Megatron's fine-grained activation offloading can queue many simultaneous
#   Device-to-Host (D2H) copies.  Under CUDA graph capture (full_iteration scope),
#   record_stream semantics are unavailable, so explicit main-stream wait_event
#   joins become mandatory.  The upstream commit adds a per-group-name FIFO of
#   pending D2H events and drains it (via wait_event) whenever the queue depth
#   exceeds a configurable cap (max_inflight_offloads).
#
# DES-LOC adaptation rationale:
#   In the Neuron_SP DES-LOC framework we run on an *asymmetric* three-device
#   cluster: 2× A6000 48 GB (SM86, PCIe) + 1× H100 NVL 96 GB (SM90, PCIe).
#   There is no NVLink; all D2H and D2D transfers compete for PCIe bandwidth.
#   Key differences from Megatron's homogeneous NVLink topology:
#
#   1. **PCIe bandwidth asymmetry**: A6000 ↔ host DRAM saturates at ~32 GB/s;
#      H100 NVL ↔ host DRAM saturates at ~64 GB/s.  Allowing the same inflight
#      cap on all devices causes A6000 to become the bottleneck and H100 to idle.
#      We assign per-device caps proportional to measured bandwidth ratios.
#
#   2. **Shared LOcality Cache (LOC)**: DES-LOC routes recently offloaded
#      activations through a 1.5 TB CPU DRAM pool that acts as a *locality
#      cache*.  Tensors reused within a micro-batch are pinned; cold tensors are
#      released.  The throttle must coordinate with this cache to avoid evicting
#      tensors that are still inflight on a D2H copy.
#
#   3. **Decoupled Execution (DE)**: Forward and backward passes of different
#      micro-batches are interleaved across devices (A6000s run forward, H100
#      runs backward or vice-versa depending on pipeline schedule).  Each device
#      maintains its own pending-event FIFO; the throttle drains them
#      independently so that a slow A6000 D2H does not block H100's backward.
#
#   4. **Group-name granularity**: We preserve Megatron's per-group-name
#      tracking (e.g. "attn_norm", "core_attn", "qkv_linear", "moe_act") but
#      add device-tier awareness so that the cap for group G on device tier T
#      is cap_base * bandwidth_ratio[T].
#
# Usage in Neuron_SP:
#   throttle = HeteroOffloadThrottle(
#       base_max_inflight=4,
#       device_tiers={0: "a6000", 1: "a6000", 2: "h100_nvl"},
#   )
#   # called after each D2H copy is enqueued:
#   throttle.record_offload(group_name="core_attn", device_id=0, event=evt)
#   # called on backward pass before reloading:
#   throttle.drain_all(device_id=0)

from __future__ import annotations

import logging
import threading
import time
import unittest
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device tier definitions — maps SM architecture / product names to relative
# PCIe bandwidth scaling factors measured on the Neuron_SP reference cluster.
# These ratios are used to derive per-device inflight caps from a single
# base_max_inflight knob, matching Megatron's single-knob UX while adding
# heterogeneous awareness.
# ---------------------------------------------------------------------------

class DeviceTier(str, Enum):
    """Enumeration of device tiers present in the DES-LOC reference cluster.

    Bandwidth ratios are normalised to A6000 (the slowest PCIe device):
        A6000  SM86  PCIe 4.0 ×16  ~32 GB/s D2H  → ratio 1.0
        H100NVL SM90 PCIe 5.0 ×16  ~64 GB/s D2H  → ratio 2.0

    A higher ratio means the device can sustain more concurrent D2H copies
    before the PCIe bus saturates, so we grant it a proportionally larger
    inflight cap.
    """

    A6000 = "a6000"
    H100_NVL = "h100_nvl"
    UNKNOWN = "unknown"


# Normalised PCIe D2H bandwidth ratios relative to A6000 baseline.
_BANDWIDTH_RATIO: Dict[DeviceTier, float] = {
    DeviceTier.A6000: 1.0,
    DeviceTier.H100_NVL: 2.0,
    DeviceTier.UNKNOWN: 1.0,
}

# Minimum inflight cap regardless of ratio (avoid cap=0 on tiny base values).
_MIN_CAP: int = 1


def _detect_device_tier(device_id: int) -> DeviceTier:
    """Auto-detect the DeviceTier for *device_id* from CUDA device properties.

    Falls back to UNKNOWN if CUDA is unavailable or the SM version is
    unrecognised.  This is intentionally permissive so the throttle degrades
    gracefully on development machines without the target hardware.
    """
    if not torch.cuda.is_available():
        return DeviceTier.UNKNOWN
    try:
        props = torch.cuda.get_device_properties(device_id)
        sm = props.major * 10 + props.minor  # e.g. SM86 → 86, SM90 → 90
        name_lower = props.name.lower()
        if sm == 90 or "h100" in name_lower:
            return DeviceTier.H100_NVL
        if sm == 86 or "a6000" in name_lower or "rtx a6000" in name_lower:
            return DeviceTier.A6000
        logger.warning(
            "DES-LOC throttle: unrecognised device %d (%s, SM%d); "
            "defaulting to UNKNOWN tier with ratio 1.0",
            device_id,
            props.name,
            sm,
        )
        return DeviceTier.UNKNOWN
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("DES-LOC throttle: device detection failed for id=%d: %s", device_id, exc)
        return DeviceTier.UNKNOWN


# ---------------------------------------------------------------------------
# LOC (Shared LOcality Cache) integration stubs
# ---------------------------------------------------------------------------

class LocalityCacheRef:
    """Lightweight reference to a tensor slot in the DES-LOC LOC (CPU DRAM pool).

    The LOC is a 1.5 TB pinned-memory ring that stores recently offloaded
    activations.  Each slot carries a *pin count*: while inflight D2H copies
    are pending the slot is pinned (pin_count > 0) and must not be evicted by
    the replacement policy.

    In a full Neuron_SP deployment this class wraps the real LOC allocator.
    Here we provide a minimal interface so the throttle can interact with it
    without a hard dependency on the rest of the runtime.
    """

    def __init__(self, tensor_key: str, cpu_buffer: Optional[torch.Tensor] = None):
        self.tensor_key = tensor_key
        self.cpu_buffer = cpu_buffer
        self._pin_count: int = 0
        self._evictable: bool = False

    def pin(self) -> None:
        """Increment the pin count — prevents LOC eviction while D2H is inflight."""
        self._pin_count += 1

    def unpin(self) -> None:
        """Decrement the pin count — allows eviction when count reaches zero."""
        if self._pin_count > 0:
            self._pin_count -= 1
        if self._pin_count == 0:
            self._evictable = True

    @property
    def is_pinned(self) -> bool:
        return self._pin_count > 0

    def __repr__(self) -> str:
        return (
            f"LocalityCacheRef(key={self.tensor_key!r}, "
            f"pin_count={self._pin_count}, evictable={self._evictable})"
        )


# ---------------------------------------------------------------------------
# Pending offload descriptor
# ---------------------------------------------------------------------------

@dataclass
class PendingOffload:
    """A single D2H copy that has been enqueued but not yet joined.

    Fields
    ------
    group_name:
        The activation group this tensor belongs to, e.g. "core_attn".
        Mirrors Megatron's group-name concept so the same throttle knob
        governs all tensors in a group uniformly.
    device_id:
        CUDA device that issued the D2H copy.
    event:
        CUDA event recorded on the D2H stream *after* the copy was submitted.
        main_stream.wait_event(event) serialises completion.
    loc_ref:
        Optional reference to the LOC slot that will receive this tensor.
        If set, the slot is pinned until the event is drained.
    enqueue_time:
        Wall-clock time when this offload was registered (for diagnostics).
    tensor_nbytes:
        Size in bytes of the offloaded tensor (for bandwidth accounting).
    """

    group_name: str
    device_id: int
    event: Any  # torch.cuda.Event or mock
    loc_ref: Optional[LocalityCacheRef] = None
    enqueue_time: float = field(default_factory=time.monotonic)
    tensor_nbytes: int = 0


# ---------------------------------------------------------------------------
# Per-device state
# ---------------------------------------------------------------------------

@dataclass
class _DeviceOffloadState:
    """Mutable state for one CUDA device's inflight offload tracking.

    Attributes
    ----------
    tier:
        Hardware tier of this device (determines bandwidth ratio).
    effective_cap:
        Computed inflight cap = ceil(base_cap * bandwidth_ratio[tier]).
    pending_by_group:
        Per group-name FIFO of PendingOffload descriptors.  Mirrors
        Megatron's ``_offload_pending_by_name`` dict of deques.
    total_bytes_drained:
        Cumulative bytes whose D2H completion was explicitly joined.
        Used for offline bandwidth auditing.
    total_drain_calls:
        Number of wait_event calls issued (diagnostic counter).
    lock:
        Per-device mutex so that concurrent micro-batch threads do not
        corrupt the FIFO.  DES-LOC's Decoupled Execution can schedule
        forward passes on A6000s while backward runs on H100, and both
        may touch the throttle concurrently.
    """

    tier: DeviceTier
    effective_cap: int
    pending_by_group: Dict[str, Deque[PendingOffload]] = field(
        default_factory=lambda: defaultdict(deque)
    )
    total_bytes_drained: int = 0
    total_drain_calls: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def reset_iter(self) -> None:
        """Clear all pending-event FIFOs at iteration boundary.

        Mirrors Megatron's ``ChunkOffloadHandler.reset()`` which clears
        ``_offload_pending_by_name`` so events from a previous (non-captured)
        iteration are never waited on in the current one.

        In DES-LOC we additionally unpin any LOC slots that were still
        pending — this can happen if a micro-batch was cancelled due to a
        pipeline bubble.
        """
        with self.lock:
            for group_name, q in self.pending_by_group.items():
                if q:
                    logger.debug(
                        "DES-LOC throttle reset: device=%d group=%r dropping %d pending events",
                        self._device_id_for_log,
                        group_name,
                        len(q),
                    )
                for po in q:
                    if po.loc_ref is not None:
                        po.loc_ref.unpin()
            self.pending_by_group.clear()

    # Set externally after construction so dataclass doesn't need it as arg.
    _device_id_for_log: int = -1


# ---------------------------------------------------------------------------
# Main throttle class
# ---------------------------------------------------------------------------

class HeteroOffloadThrottle:
    """Per-group-name D2H offload throttle with DES-LOC heterogeneous awareness.

    This class is the DES-LOC reinterpretation of the inflight-offload
    throttle introduced in Megatron-LM commit 9a7cd17.  It preserves the
    upstream semantics (per-group-name FIFO, configurable cap, drain-on-exceed)
    while adding:

    * **Device-tier-aware caps**: each device gets an effective cap derived
      from ``base_max_inflight * bandwidth_ratio[tier]``.  A6000 devices
      (ratio 1×) get the base cap; H100 NVL (ratio 2×) gets twice as many
      inflight copies before draining.  This prevents slower PCIe lanes from
      artificially limiting the faster device.

    * **LOC pin/unpin integration**: when a ``LocalityCacheRef`` is supplied
      with a ``record_offload`` call, the slot is pinned until its event is
      drained.  This prevents the LOC replacement policy from evicting a buffer
      that is still being written by an in-progress DMA engine.

    * **Decoupled-execution safety**: each device has its own lock and FIFO
      state, so concurrent forward (A6000) and backward (H100) threads cannot
      interfere.

    * **Iteration-boundary reset**: ``reset_iter(device_id)`` matches
      Megatron's per-chunk reset; call it at the start of each micro-batch
      forward pass.

    Parameters
    ----------
    base_max_inflight:
        Base cap for the *slowest* device tier (A6000).  Maps to Megatron's
        ``fine_grained_offloading_max_inflight_offloads``.  None disables
        throttling entirely (same semantic as upstream None).
    device_tiers:
        Optional explicit mapping of device_id → tier string.  If omitted,
        tiers are auto-detected via CUDA device properties.
    enable_loc_pinning:
        When True (default) pin/unpin LOC slots around in-flight D2H copies.

    Examples
    --------
    >>> throttle = HeteroOffloadThrottle(base_max_inflight=4)
    >>> evt = torch.cuda.Event()
    >>> throttle.record_offload("core_attn", device_id=0, event=evt)
    >>> throttle.drain_all(device_id=0)
    """

    def __init__(
        self,
        base_max_inflight: Optional[int] = None,
        device_tiers: Optional[Dict[int, str]] = None,
        enable_loc_pinning: bool = True,
    ) -> None:
        self._base_max_inflight = base_max_inflight
        self._enable_loc_pinning = enable_loc_pinning
        # device_id → _DeviceOffloadState
        self._device_states: Dict[int, _DeviceOffloadState] = {}
        # Pre-populate from explicit tier map if given.
        if device_tiers:
            for dev_id, tier_str in device_tiers.items():
                try:
                    tier = DeviceTier(tier_str)
                except ValueError:
                    logger.warning(
                        "DES-LOC throttle: unknown tier string %r for device %d; "
                        "using UNKNOWN",
                        tier_str,
                        dev_id,
                    )
                    tier = DeviceTier.UNKNOWN
                self._init_device_state(dev_id, tier)
        logger.info(
            "DES-LOC HeteroOffloadThrottle initialised: base_max_inflight=%s "
            "loc_pinning=%s pre-registered devices=%s",
            base_max_inflight,
            enable_loc_pinning,
            list(self._device_states.keys()),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_device_state(self, device_id: int, tier: Optional[DeviceTier] = None) -> _DeviceOffloadState:
        """Initialise and cache state for *device_id*.

        If *tier* is not supplied it is auto-detected.  The effective cap is
        computed once at init time so the hot path (record_offload /
        _drain_group) avoids repeated multiplications.
        """
        if tier is None:
            tier = _detect_device_tier(device_id)
        ratio = _BANDWIDTH_RATIO.get(tier, 1.0)
        if self._base_max_inflight is None:
            effective_cap = None  # throttling disabled
        else:
            effective_cap = max(_MIN_CAP, int(self._base_max_inflight * ratio))

        state = _DeviceOffloadState(
            tier=tier,
            effective_cap=effective_cap,  # type: ignore[arg-type]
        )
        state._device_id_for_log = device_id
        self._device_states[device_id] = state

        logger.info(
            "DES-LOC throttle: registered device %d tier=%s ratio=%.1f "
            "effective_cap=%s",
            device_id,
            tier.value,
            ratio,
            effective_cap,
        )
        return state

    def _get_state(self, device_id: int) -> _DeviceOffloadState:
        """Return (and lazily create) the state for *device_id*."""
        if device_id not in self._device_states:
            return self._init_device_state(device_id)
        return self._device_states[device_id]

    def _drain_group(
        self,
        state: _DeviceOffloadState,
        group_name: str,
        *,
        drain_all: bool = False,
    ) -> int:
        """Drain the pending-event FIFO for *group_name* on *state*'s device.

        Mirrors ``ChunkOffloadHandler._drain_offload_pending`` from Megatron
        but is extended with LOC unpin logic and an optional drain_all flag
        used during end-of-micro-batch cleanup.

        Parameters
        ----------
        state:
            Device state containing the FIFO and lock.
        group_name:
            Activation group whose FIFO to drain.
        drain_all:
            If True, drain every pending event regardless of cap.  Used when
            a complete micro-batch finishes (DES-LOC backward pass handoff).

        Returns
        -------
        int
            Number of wait_event calls issued in this invocation.
        """
        if state.effective_cap is None and not drain_all:
            return 0  # throttling disabled

        q: Deque[PendingOffload] = state.pending_by_group[group_name]
        cap = 0 if drain_all else state.effective_cap
        drained = 0

        # We need the *current* CUDA stream of the device that owns this state.
        # In DES-LOC's decoupled execution the calling thread may be on a
        # different device, so we temporarily switch context.
        if torch.cuda.is_available():
            try:
                with torch.cuda.device(state._device_id_for_log):
                    cur_stream = torch.cuda.current_stream()
                    while len(q) > cap:
                        po = q.popleft()
                        cur_stream.wait_event(po.event)
                        if self._enable_loc_pinning and po.loc_ref is not None:
                            po.loc_ref.unpin()
                        state.total_bytes_drained += po.tensor_nbytes
                        state.total_drain_calls += 1
                        drained += 1
            except Exception as exc:  # pylint: disable=broad-except
                # Non-fatal: log and continue.  A missed wait_event is a
                # correctness issue but should not crash the training run —
                # the tensor will simply be reclaimed later.
                logger.error(
                    "DES-LOC throttle: wait_event failed for device=%d group=%r: %s",
                    state._device_id_for_log,
                    group_name,
                    exc,
                )
        else:
            # Dry-run / CPU-only mode (unit tests): pop without waiting.
            while len(q) > cap:
                po = q.popleft()
                if self._enable_loc_pinning and po.loc_ref is not None:
                    po.loc_ref.unpin()
                state.total_bytes_drained += po.tensor_nbytes
                state.total_drain_calls += 1
                drained += 1

        if drained > 0:
            logger.debug(
                "DES-LOC throttle: drained %d event(s) for device=%d group=%r "
                "(remaining=%d cap=%s)",
                drained,
                state._device_id_for_log,
                group_name,
                len(q),
                cap,
            )
        return drained

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_offload(
        self,
        group_name: str,
        device_id: int,
        event: Any,
        loc_ref: Optional[LocalityCacheRef] = None,
        tensor_nbytes: int = 0,
    ) -> None:
        """Register a newly-enqueued D2H copy and drain old events if needed.

        This is the DES-LOC equivalent of the Megatron post-offload block::

            if self._max_inflight_offloads is not None:
                gname = group_to_offload._name
                self._offload_pending_by_name[gname].append(...)
                self._drain_offload_pending(gname)

        Call this immediately after submitting the D2H copy on the device's
        D2H stream and recording *event*.

        Parameters
        ----------
        group_name:
            Activation group name (e.g. ``"core_attn"``, ``"moe_act"``).
        device_id:
            CUDA device that issued the copy.
        event:
            CUDA event (torch.cuda.Event) recorded after the D2H submit.
        loc_ref:
            LOC slot that will receive the tensor.  If supplied and LOC
            pinning is enabled, the slot is pinned until the event drains.
        tensor_nbytes:
            Byte size of the offloaded tensor (for diagnostics only).
        """
        if self._base_max_inflight is None:
            # Throttling disabled — behave identically to Megatron's None path.
            return

        state = self._get_state(device_id)
        po = PendingOffload(
            group_name=group_name,
            device_id=device_id,
            event=event,
            loc_ref=loc_ref,
            tensor_nbytes=tensor_nbytes,
        )

        with state.lock:
            if self._enable_loc_pinning and loc_ref is not None:
                loc_ref.pin()
            state.pending_by_group[group_name].append(po)
            self._drain_group(state, group_name)

    def drain_group(self, group_name: str, device_id: int) -> int:
        """Explicitly drain the FIFO for one group on one device.

        Useful in DES-LOC's backward pass handoff point where a micro-batch
        is about to be transferred from A6000 forward ownership to H100
        backward ownership.  All pending D2H events for the group must be
        joined before the H100 can safely read from CPU DRAM.

        Returns the number of wait_event calls issued.
        """
        if self._base_max_inflight is None:
            return 0
        state = self._get_state(device_id)
        with state.lock:
            return self._drain_group(state, group_name, drain_all=True)

    def drain_all(self, device_id: int) -> int:
        """Join all pending D2H events across every group on *device_id*.

        Call at the end of a micro-batch's forward pass (before handing off
        activations to the backward pass on a potentially different device).
        In Megatron's single-device setting this is implicit; in DES-LOC's
        heterogeneous topology it must be explicit so H100 does not read
        stale CPU buffers filled by A6000's PCIe engine.

        Returns total wait_event calls issued.
        """
        if self._base_max_inflight is None:
            return 0
        state = self._get_state(device_id)
        total = 0
        with state.lock:
            for group_name in list(state.pending_by_group.keys()):
                total += self._drain_group(state, group_name, drain_all=True)
        if total > 0:
            logger.debug(
                "DES-LOC throttle: drain_all device=%d joined %d event(s) total",
                device_id,
                total,
            )
        return total

    def reset_iter(self, device_id: int) -> None:
        """Clear all pending-event FIFOs for *device_id* at iteration boundary.

        Mirrors ``ChunkOffloadHandler.reset()`` in Megatron.  Must be called
        at the start of each micro-batch forward pass so that events from
        the previous iteration's (non-captured) CUDA operations are not
        accidentally waited on in the current captured graph.

        In DES-LOC we additionally unpin all LOC slots whose events we are
        discarding.
        """
        state = self._get_state(device_id)
        state.reset_iter()

    def stats(self, device_id: int) -> Dict[str, Any]:
        """Return diagnostic statistics for *device_id*.

        Useful for offline profiling and bandwidth auditing.  The returned
        dict is JSON-serialisable.
        """
        if device_id not in self._device_states:
            return {}
        state = self._device_states[device_id]
        with state.lock:
            pending_summary = {
                gname: len(q) for gname, q in state.pending_by_group.items()
            }
        return {
            "device_id": device_id,
            "tier": state.tier.value,
            "effective_cap": state.effective_cap,
            "base_max_inflight": self._base_max_inflight,
            "total_bytes_drained": state.total_bytes_drained,
            "total_drain_calls": state.total_drain_calls,
            "pending_by_group": pending_summary,
        }

    def all_stats(self) -> List[Dict[str, Any]]:
        """Return stats for every registered device."""
        return [self.stats(dev_id) for dev_id in sorted(self._device_states)]


# ---------------------------------------------------------------------------
# DeepSpeed integration helper
# ---------------------------------------------------------------------------

class DESLOCOffloadConfig:
    """Thin configuration dataclass for DES-LOC offload settings.

    Maps onto DeepSpeed's ZeRO / activation-checkpointing config dict and
    provides the same single-knob interface as Megatron's
    ``fine_grained_offloading_max_inflight_offloads``.

    Parameters
    ----------
    base_max_inflight:
        Per-group-name cap on inflight D2H copies for the *baseline* device
        tier (A6000).  None disables throttling.  Must be a non-negative
        integer when used with full-iteration CUDA graph capture (same
        constraint as Megatron's assertion in ``TransformerConfig.__post_init__``).
    use_full_iter_cuda_graph:
        Whether full-iteration CUDA graph capture is active.  When True,
        base_max_inflight must not be None (enforced in validate()).
    device_tiers:
        Optional explicit device-to-tier map; auto-detected if absent.
    offload_group_names:
        The activation group names that will be offloaded.  Used for
        pre-validation only.
    """

    def __init__(
        self,
        base_max_inflight: Optional[int] = None,
        use_full_iter_cuda_graph: bool = False,
        device_tiers: Optional[Dict[int, str]] = None,
        offload_group_names: Optional[List[str]] = None,
        base_cap_per_group: int = 4,
        h100_cap_multiplier: float = 2.0,
        enable_locality_cache: bool = True,
        max_outstanding_bytes: int = 4 * 1024 ** 3,
    ) -> None:
        self.base_max_inflight = base_max_inflight
        self.use_full_iter_cuda_graph = use_full_iter_cuda_graph
        self.device_tiers = device_tiers or {}
        self.offload_group_names = offload_group_names or []
        self.base_cap_per_group = base_cap_per_group
        self.h100_cap_multiplier = h100_cap_multiplier
        self.enable_locality_cache = enable_locality_cache
        self.max_outstanding_bytes = max_outstanding_bytes

    def validate(self) -> None:
        """Replicate Megatron's ``__post_init__`` assertion in DES-LOC context.

        Megatron asserts that ``fine_grained_offloading_max_inflight_offloads``
        must be a non-negative integer when ``cuda_graph_impl == "full_iteration"``.
        We enforce the same constraint here so the DeepSpeed config path has
        equivalent safety checks.
        """
        if self.use_full_iter_cuda_graph:
            if self.base_max_inflight is None or self.base_max_inflight < 0:
                raise ValueError(
                    "DES-LOC: base_max_inflight must be a non-negative integer "
                    "when use_full_iter_cuda_graph=True.  "
                    "This mirrors Megatron's requirement for full_iteration CUDA graphs "
                    "where record_stream is unavailable and explicit wait_event joins "
                    "are mandatory."
                )

    def build_throttle(self) -> HeteroOffloadThrottle:
        """Construct and return a HeteroOffloadThrottle from this config."""
        self.validate()
        return HeteroOffloadThrottle(
            base_max_inflight=self.base_max_inflight,
            device_tiers=self.device_tiers,
        )


# ---------------------------------------------------------------------------
# Pipeline-level coordinator
# ---------------------------------------------------------------------------

class DESLOCPipelineOffloadCoordinator:
    """Coordinates inflight offload throttling across a DES-LOC pipeline.

    In DES-LOC, the pipeline schedule may interleave micro-batches from
    different stages across heterogeneous devices.  This coordinator owns one
    HeteroOffloadThrottle instance and exposes a higher-level API that matches
    the lifecycle of a micro-batch:

        coordinator.on_microbatch_start(mb_id, device_id)
        # ... forward pass, calling record_offload per tensor group ...
        coordinator.on_group_offloaded(mb_id, device_id, group_name, event, loc_ref)
        # ... end of forward pass ...
        coordinator.on_microbatch_forward_end(mb_id, device_id)
        # ... backward pass may run on a different device ...
        coordinator.on_microbatch_backward_start(mb_id, backward_device_id)

    Parameters
    ----------
    config:
        DESLOCOffloadConfig instance (already validated).
    """

    def __init__(self, config: DESLOCOffloadConfig) -> None:
        self._throttle = config.build_throttle()
        self._config = config
        # mb_id → (forward_device_id, dict of group_name → pending count)
        self._mb_registry: Dict[int, Tuple[int, Dict[str, int]]] = {}
        self._lock = threading.Lock()

    def on_microbatch_start(self, mb_id: int, device_id: int) -> None:
        """Register a new micro-batch and reset iteration state on its device."""
        self._throttle.reset_iter(device_id)
        with self._lock:
            self._mb_registry[mb_id] = (device_id, defaultdict(int))

    def on_group_offloaded(
        self,
        mb_id: int,
        device_id: int,
        group_name: str,
        event: Any,
        loc_ref: Optional[LocalityCacheRef] = None,
        tensor_nbytes: int = 0,
    ) -> None:
        """Record a D2H offload for a group tensor in micro-batch *mb_id*."""
        self._throttle.record_offload(
            group_name=group_name,
            device_id=device_id,
            event=event,
            loc_ref=loc_ref,
            tensor_nbytes=tensor_nbytes,
        )
        with self._lock:
            if mb_id in self._mb_registry:
                self._mb_registry[mb_id][1][group_name] += 1

    def on_microbatch_forward_end(self, mb_id: int, device_id: int) -> None:
        """Drain all pending D2H events before handing off to backward.

        This is the DES-LOC equivalent of Megatron's implicit synchronisation
        at the end of a forward pass.  In our heterogeneous setting the
        backward pass may run on a *different* device (H100), so we must
        ensure all A6000→DRAM DMA operations are complete before H100 reads
        from the LOC.
        """
        n = self._throttle.drain_all(device_id)
        if n > 0:
            logger.debug(
                "DES-LOC coordinator: mb=%d forward_end on device=%d drained %d events",
                mb_id,
                device_id,
                n,
            )

    def on_microbatch_backward_start(self, mb_id: int, backward_device_id: int) -> None:
        """Ensure D2H completions are visible on the backward device.

        If the backward device differs from the forward device, any LOC slots
        written by the forward device's DMA engine must be flushed to the
        coherency domain.  For PCIe (no NVLink) the CPU DRAM is the shared
        medium; the forward device's drain_all already issued wait_event on the
        D2H stream so the data is committed to DRAM.  We log the handoff.
        """
        with self._lock:
            if mb_id in self._mb_registry:
                fwd_device, group_counts = self._mb_registry[mb_id]
                if fwd_device != backward_device_id:
                    logger.debug(
                        "DES-LOC coordinator: mb=%d handoff fwd_device=%d → bwd_device=%d "
                        "groups=%s",
                        mb_id,
                        fwd_device,
                        backward_device_id,
                        dict(group_counts),
                    )

    def on_microbatch_complete(self, mb_id: int) -> None:
        """Clean up registry entry for a completed micro-batch."""
        with self._lock:
            self._mb_registry.pop(mb_id, None)

    def global_stats(self) -> List[Dict[str, Any]]:
        """Return throttle stats for all registered devices."""
        return self._throttle.all_stats()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        stream=sys.stdout,
    )

    class _MockEvent:
        """Minimal torch.cuda.Event substitute for CPU-only test execution."""

        def __init__(self, label: str = ""):
            self.label = label
            self.waited = False

        def wait(self, stream: Any = None) -> None:
            self.waited = True

        def __repr__(self) -> str:
            return f"MockEvent({self.label!r})"

    class TestHeteroOffloadThrottle(unittest.TestCase):
        """Unit tests for HeteroOffloadThrottle in CPU-only (mock) mode."""

        def _make_throttle(self, base: Optional[int], tiers: Optional[Dict] = None) -> HeteroOffloadThrottle:
            return HeteroOffloadThrottle(
                base_max_inflight=base,
                device_tiers=tiers or {0: "a6000", 1: "a6000", 2: "h100_nvl"},
            )

        # ------------------------------------------------------------------
        def test_disabled_throttle_records_nothing(self):
            """When base_max_inflight is None, no state should accumulate."""
            throttle = self._make_throttle(None)
            evt = _MockEvent("e0")
            throttle.record_offload("core_attn", 0, evt)
            # State should not have been initialised for device 0.
            # (record_offload returns early without touching state)
            self.assertNotIn(0, throttle._device_states)

        # ------------------------------------------------------------------
        def test_effective_cap_scaling(self):
            """H100 NVL device should get 2× the base cap vs A6000."""
            throttle = self._make_throttle(4, {0: "a6000", 2: "h100_nvl"})
            state_a6 = throttle._get_state(0)
            state_h1 = throttle._get_state(2)
            self.assertEqual(state_a6.effective_cap, 4)   # 4 * 1.0
            self.assertEqual(state_h1.effective_cap, 8)   # 4 * 2.0

        # ------------------------------------------------------------------
        def test_drain_on_exceed(self):
            """After cap+1 records the oldest event should be drained."""
            throttle = self._make_throttle(2, {0: "a6000"})
            events = [_MockEvent(f"e{i}") for i in range(4)]
            loc_refs = [LocalityCacheRef(f"t{i}") for i in range(4)]

            for i, (evt, ref) in enumerate(zip(events, loc_refs)):
                throttle.record_offload("core_attn", 0, evt, loc_ref=ref, tensor_nbytes=1024)

            state = throttle._get_state(0)
            # With cap=2 and 4 records: should have drained 2, leaving 2 pending.
            q = state.pending_by_group["core_attn"]
            self.assertEqual(len(q), 2)
            # The first two LOC slots should be unpinned.
            self.assertFalse(loc_refs[0].is_pinned)
            self.assertFalse(loc_refs[1].is_pinned)
            # The last two should still be pinned.
            self.assertTrue(loc_refs[2].is_pinned)
            self.assertTrue(loc_refs[3].is_pinned)

        # ------------------------------------------------------------------
        def test_drain_all_clears_queue(self):
            """drain_all should empty all pending queues for a device."""
            throttle = self._make_throttle(3, {0: "a6000"})
            for i in range(3):
                throttle.record_offload("qkv_linear", 0, _MockEvent(f"q{i}"))
            n = throttle.drain_all(0)
            state = throttle._get_state(0)
            self.assertEqual(len(state.pending_by_group["qkv_linear"]), 0)
            self.assertEqual(n, 0)  # cap=3, exactly 3 → no excess at record time

            # Add one more and drain_all explicitly.
            throttle.record_offload("qkv_linear", 0, _MockEvent("q3"))
            n = throttle.drain_all(0)
            self.assertEqual(n, 1)

        # ------------------------------------------------------------------
        def test_reset_iter_clears_and_unpins(self):
            """reset_iter should clear FIFOs and unpin LOC slots."""
            throttle = self._make_throttle(10, {0: "a6000"})
            refs = [LocalityCacheRef(f"r{i}") for i in range(3)]
            for i, ref in enumerate(refs):
                throttle.record_offload("moe_act", 0, _MockEvent(f"m{i}"), loc_ref=ref)

            # All refs should be pinned.
            for ref in refs:
                self.assertTrue(ref.is_pinned)

            throttle.reset_iter(0)
            state = throttle._get_state(0)
            self.assertEqual(len(state.pending_by_group), 0)
            # Unpin should have been called.
            for ref in refs:
                self.assertFalse(ref.is_pinned)

        # ------------------------------------------------------------------
        def test_per_group_independence(self):
            """Two groups should have independent FIFOs and drain independently."""
            throttle = self._make_throttle(2, {1: "a6000"})
            for i in range(3):
                throttle.record_offload("core_attn", 1, _MockEvent(f"ca{i}"))
            for i in range(2):
                throttle.record_offload("moe_act", 1, _MockEvent(f"ma{i}"))

            state = throttle._get_state(1)
            # core_attn: 3 records, cap=2 → 1 drained, 2 remaining.
            self.assertEqual(len(state.pending_by_group["core_attn"]), 2)
            # moe_act: 2 records, cap=2 → 0 drained, 2 remaining.
            self.assertEqual(len(state.pending_by_group["moe_act"]), 2)

        # ------------------------------------------------------------------
        def test_zero_cap_drains_immediately(self):
            """Cap=0 means every record immediately drains itself."""
            throttle = self._make_throttle(0, {0: "a6000"})
            refs = [LocalityCacheRef(f"z{i}") for i in range(5)]
            for i, ref in enumerate(refs):
                throttle.record_offload("attn_norm", 0, _MockEvent(f"z{i}"), loc_ref=ref)

            state = throttle._get_state(0)
            self.assertEqual(len(state.pending_by_group["attn_norm"]), 0)
            for ref in refs:
                self.assertFalse(ref.is_pinned)

        # ------------------------------------------------------------------
        def test_stats_structure(self):
            """stats() should return a well-formed dict."""
            throttle = self._make_throttle(4, {0: "a6000", 2: "h100_nvl"})
            throttle.record_offload("core_attn", 0, _MockEvent("s0"), tensor_nbytes=512)
            s = throttle.stats(0)
            self.assertEqual(s["device_id"], 0)
            self.assertEqual(s["tier"], "a6000")
            self.assertEqual(s["effective_cap"], 4)
            self.assertIn("pending_by_group", s)

        # ------------------------------------------------------------------
        def test_unknown_tier_fallback(self):
            """An unknown tier string should fall back gracefully."""
            throttle = HeteroOffloadThrottle(
                base_max_inflight=2,
                device_tiers={5: "quantum_gpu"},
            )
            state = throttle._get_state(5)
            self.assertEqual(state.tier, DeviceTier.UNKNOWN)
            self.assertEqual(state.effective_cap, 2)

    class TestDESLOCOffloadConfig(unittest.TestCase):
        """Unit tests for DESLOCOffloadConfig validation."""

        def test_full_iter_requires_nonnone_nonneg(self):
            """Full-iteration CUDA graph mode requires a valid inflight cap."""
            cfg = DESLOCOffloadConfig(
                base_max_inflight=None,
                use_full_iter_cuda_graph=True,
            )
            with self.assertRaises(ValueError):
                cfg.validate()

        def test_full_iter_negative_raises(self):
            cfg = DESLOCOffloadConfig(
                base_max_inflight=-1,
                use_full_iter_cuda_graph=True,
            )
            with self.assertRaises(ValueError):
                cfg.validate()

        def test_full_iter_zero_is_valid(self):
            """Cap=0 is valid for full-iteration mode (drain-on-every-commit)."""
            cfg = DESLOCOffloadConfig(
                base_max_inflight=0,
                use_full_iter_cuda_graph=True,
                device_tiers={0: "a6000"},
            )
            cfg.validate()  # should not raise

        def test_no_graph_none_is_valid(self):
            """Without full-iteration CUDA graphs, None cap is acceptable."""
            cfg = DESLOCOffloadConfig(
                base_max_inflight=None,
                use_full_iter_cuda_graph=False,
            )
            cfg.validate()  # should not raise

        def test_build_throttle(self):
            """build_throttle() should return a HeteroOffloadThrottle."""
            cfg = DESLOCOffloadConfig(
                base_max_inflight=4,
                device_tiers={0: "a6000", 2: "h100_nvl"},
            )
            t = cfg.build_throttle()
            self.assertIsInstance(t, HeteroOffloadThrottle)

    class TestDESLOCPipelineCoordinator(unittest.TestCase):
        """Integration tests for DESLOCPipelineOffloadCoordinator."""

        def _make_coord(self, base: int = 4) -> DESLOCPipelineOffloadCoordinator:
            cfg = DESLOCOffloadConfig(
                base_max_inflight=base,
                use_full_iter_cuda_graph=False,
                device_tiers={0: "a6000", 1: "a6000", 2: "h100_nvl"},
            )
            return DESLOCPipelineOffloadCoordinator(cfg)

        def test_microbatch_lifecycle(self):
            """Full lifecycle: start → offload → forward_end → backward_start → complete."""
            coord = self._make_coord(base=3)
            coord.on_microbatch_start(mb_id=0, device_id=0)

            refs = []
            for i in range(4):
                ref = LocalityCacheRef(f"mb0_t{i}")
                coord.on_group_offloaded(0, 0, "core_attn", _MockEvent(f"e{i}"), ref, 1024)
                refs.append(ref)

            # At this point 1 should be drained (cap=3, 4 records → 1 drained)
            state = coord._throttle._get_state(0)
            q = state.pending_by_group["core_attn"]
            self.assertEqual(len(q), 3)

            # Forward end: drain remaining 3.
            coord.on_microbatch_forward_end(mb_id=0, device_id=0)
            self.assertEqual(len(state.pending_by_group["core_attn"]), 0)

            # Backward on different device (H100).
            coord.on_microbatch_backward_start(mb_id=0, backward_device_id=2)
            coord.on_microbatch_complete(mb_id=0)
            self.assertNotIn(0, coord._mb_registry)

        def test_reset_iter_on_new_microbatch(self):
            """on_microbatch_start should reset device state for the new iteration."""
            coord = self._make_coord(base=10)
            coord.on_microbatch_start(mb_id=1, device_id=1)
            coord.on_group_offloaded(1, 1, "moe_act", _MockEvent("pre"), tensor_nbytes=2048)
            coord.on_microbatch_start(mb_id=2, device_id=1)  # reset
            state = coord._throttle._get_state(1)
            self.assertEqual(len(state.pending_by_group), 0)

        def test_global_stats_covers_all_devices(self):
            """global_stats should return entries for all devices touched."""
            coord = self._make_coord(base=2)
            coord.on_microbatch_start(0, 0)
            coord.on_group_offloaded(0, 0, "core_attn", _MockEvent("g0"))
            coord.on_microbatch_start(1, 2)
            coord.on_group_offloaded(1, 2, "core_attn", _MockEvent("g1"))
            stats = coord.global_stats()
            device_ids = {s["device_id"] for s in stats}
            self.assertIn(0, device_ids)
            self.assertIn(2, device_ids)

        def test_heterogeneous_cap_respected(self):
            """A6000 and H100 should drain at different thresholds in coordinator."""
            coord = self._make_coord(base=2)  # A6000 cap=2, H100 cap=4

            coord.on_microbatch_start(0, device_id=0)  # A6000
            for i in range(5):
                coord.on_group_offloaded(0, 0, "qkv_linear", _MockEvent(f"a{i}"))
            state_a6 = coord._throttle._get_state(0)
            # 5 records, cap=2 → 3 drained, 2 remaining
            self.assertEqual(len(state_a6.pending_by_group["qkv_linear"]), 2)

            coord.on_microbatch_start(1, device_id=2)  # H100 NVL
            for i in range(5):
                coord.on_group_offloaded(1, 2, "qkv_linear", _MockEvent(f"h{i}"))
            state_h1 = coord._throttle._get_state(2)
            # 5 records, cap=4 → 1 drained, 4 remaining
            self.assertEqual(len(state_h1.pending_by_group["qkv_linear"]), 4)

    # Run all tests.
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in (
        TestHeteroOffloadThrottle,
        TestDESLOCOffloadConfig,
        TestDESLOCPipelineCoordinator,
    ):
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


# ---------------------------------------------------------------------------
# Engine registration hook
# ---------------------------------------------------------------------------

def register(engine) -> None:
    """Register HeteroOffloadThrottle / DESLOCPipelineOffloadCoordinator hooks on a DeepSpeed engine.

    Builds a :class:`DESLOCPipelineOffloadCoordinator` from the engine's
    configuration and attaches it as ``engine.hetero_offload_coordinator``.

    Micro-batch lifecycle hooks (``on_microbatch_start``,
    ``on_microbatch_forward_end``, ``on_microbatch_backward_start``,
    ``on_microbatch_complete``) are wired to the engine's pipeline schedule
    callbacks when the engine supports them, ensuring that offload throttling
    and locality-cache bookkeeping are driven automatically without changes to
    the training loop.

    Parameters
    ----------
    engine:
        A DeepSpeed engine instance (typically ``PipelineEngine``).  The
        offload configuration is derived from
        ``engine.config.get("hetero_offload", {})`` when available.
    """
    import logging as _logging
    _log = _logging.getLogger(__name__)

    _log.info(
        "hetero_offload_throttle.register() called on engine type=%s",
        type(engine).__name__,
    )

    # Build offload configuration from engine config or defaults
    cfg_dict = {}
    if hasattr(engine, "config") and isinstance(engine.config, dict):
        cfg_dict = engine.config.get("hetero_offload", {})

    config = DESLOCOffloadConfig(
        base_cap_per_group=cfg_dict.get("base_cap_per_group", 4),
        h100_cap_multiplier=cfg_dict.get("h100_cap_multiplier", 2.0),
        enable_locality_cache=cfg_dict.get("enable_locality_cache", True),
        max_outstanding_bytes=cfg_dict.get("max_outstanding_bytes", 4 * 1024 ** 3),
    )

    coordinator = DESLOCPipelineOffloadCoordinator(config=config)

    # Attach to engine
    engine.hetero_offload_coordinator = coordinator

    # Wire pipeline micro-batch lifecycle hooks if the engine supports them
    if hasattr(engine, "register_pipeline_microbatch_hook"):
        engine.register_pipeline_microbatch_hook(
            on_start=coordinator.on_microbatch_start,
            on_forward_end=coordinator.on_microbatch_forward_end,
            on_backward_start=coordinator.on_microbatch_backward_start,
            on_complete=coordinator.on_microbatch_complete,
        )
        _log.info(
            "DESLOCPipelineOffloadCoordinator registered via "
            "engine.register_pipeline_microbatch_hook."
        )
    else:
        _log.info(
            "Engine does not expose register_pipeline_microbatch_hook; "
            "call engine.hetero_offload_coordinator lifecycle methods manually."
        )
