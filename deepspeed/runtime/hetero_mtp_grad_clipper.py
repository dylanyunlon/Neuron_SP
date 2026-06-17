"""
DES-LOC Heterogeneous MTP Gradient Clipper
===========================================

Upstream design intent (Megatron commit 5b16c99):
-------------------------------------------------
Megatron-LM introduced the ability to clip Multi-Token Prediction (MTP) head
gradients **separately** from the main model gradients when ``mtp_detach_heads=True``.
The core insight is that MTP heads, when detached, form a training sub-graph whose
gradient magnitudes can diverge wildly from the backbone — pooling them into a single
global norm causes the backbone gradients to be over-clipped (or under-clipped) based
on MTP dynamics that are structurally irrelevant to the main learning signal.

The upstream solution introduces:
1. A ``grad_norm_group`` attribute on parameters, acting as a routing tag.
2. ``SEPARATE_GRAD_NORM_GROUPS`` registry governing which groups get independent norms.
3. ``copy_optimizer_param_metadata`` — metadata propagation when creating param shards.
4. ``get_grads_for_grad_norm(group=...)`` replacing the old ``get_main_grads_for_grad_norm``.
5. ``has_grad_norm_group`` with a cached global all-reduce so all ranks agree on whether
   a collective is needed, avoiding asymmetric barrier deadlocks.
6. ``clip_grad_norm`` splitting params into ``main_params`` and per-group buckets, each
   clipped against its own norm.

DES-LOC adaptation points:
---------------------------
The DES-LOC (Decoupled Execution with Shared LOcality Cache) framework introduces
hardware heterogeneity that Megatron assumes away:

  Hardware topology (Neuron_SP target):
    - Device 0: NVIDIA A6000  48 GB  SM86  (PCIe)
    - Device 1: NVIDIA A6000  48 GB  SM86  (PCIe)
    - Device 2: NVIDIA H100 NVL  96 GB  SM90  (PCIe, no NVLink to A6000s)
    - Host:     1.5 TB CPU DRAM

  Key divergences from Megatron assumptions:
    1. **Device affinity**: MTP heads are assigned to SM86 devices (A6000) because their
       detached gradient graphs are self-contained and don't need H100 Tensor Core
       throughput.  Backbone layers live on the H100.  This means gradient all-reduces
       for different norm groups cross different PCIe roots — they must not be conflated
       into a single collective or the device-group topology becomes inconsistent.

    2. **Shared LOcality Cache (SLoC)**: DES-LOC maintains a CPU-resident parameter
       cache that mirrors recently-used parameter shards.  When a gradient norm group
       is computed, the SLoC must be invalidated for all params in that group before
       the optimizer step writes back updated values, otherwise stale cached grads
       corrupt the next forward pass's activation checkpointing.

    3. **PCIe bandwidth budgeting**: All-reduce latency on PCIe is ~10× worse than
       NVLink.  The ``has_grad_norm_group`` caching mechanism (upstream: cache after
       one all-reduce) is extended here to also record *which device process group*
       was used, so subsequent calls never re-enter the wrong group.

    4. **SM architecture divergence**: FP8 quantisation is only valid on SM90 (H100).
       MTP params on SM86 cannot use ``use_precision_aware_optimizer_no_fp8_or_ds_fp8``
       path; the ``use_decoupled_grad`` flag must be conditioned on both the optimizer
       config *and* the parameter's device capability.

    5. **DeepSpeed ZeRO integration**: Neuron_SP wraps the optimizer inside DeepSpeed
       ZeRO stage 1/2.  Parameter shards cross device boundaries.  ``copy_optimizer_param_metadata``
       must propagate not only ``shared`` and ``grad_norm_group`` but also DES-LOC's
       ``device_tier`` annotation (``'sm86'`` or ``'sm90'``), so shard views retain
       their hardware affinity throughout the ZeRO partitioning.

Module responsibilities
-----------------------
This file provides:
  - ``DESLOCK_MTP_GRAD_NORM_GROUP``  — canonical group name for MTP heads.
  - ``DESLOCK_SEPARATE_GRAD_NORM_GROUPS``  — registry (extensible).
  - ``DeviceTier``  — enum encoding SM-capability tiers present in the cluster.
  - ``HeteroParamMetadata``  — dataclass carrying all per-parameter routing metadata.
  - ``copy_hetero_param_metadata``  — metadata propagation for ZeRO shard creation
    (extends Megatron's ``copy_optimizer_param_metadata``).
  - ``HeteroGradNormGroupRouter``  — stateful router that partitions a flat parameter
    list into main vs. per-group buckets, respecting device affinity and SLoC state.
  - ``SLocInvalidator``  — lightweight wrapper around the DES-LOC locality cache that
    triggers targeted evictions when a grad-norm group is about to be updated.
  - ``HeteroMTPGradClipper``  — top-level class mirroring Megatron's
    ``clip_grad_norm`` / ``_compute_grad_norms_by_group`` logic, adapted for
    heterogeneous device topology and DeepSpeed integration.
  - ``HeteroChainedOptimizer``  — thin subclass of DeepSpeed's optimizer wrapper that
    overrides gradient-norm collection to use group-aware routing.

Author: Neuron_SP / DES-LOC project  (mirrors Megatron 5b16c99)
"""

from __future__ import annotations

import logging
import math
import weakref
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DESLOCK_MTP_GRAD_NORM_GROUP: str = "mtp"
"""Canonical grad-norm group name for Multi-Token Prediction heads.

Mirrors Megatron's ``MTP_GRAD_NORM_GROUP = 'mtp'``.  The string value is
intentionally identical so checkpoints and logging remain compatible with
upstream Megatron tooling.
"""

DESLOCK_GRAD_NORM_GROUP_ATTR: str = "grad_norm_group"
"""Attribute name written onto ``torch.nn.Parameter`` tensors to tag them."""

DESLOCK_DEVICE_TIER_ATTR: str = "device_tier"
"""DES-LOC extension: attribute recording the SM-capability tier of the device
that *owns* (or will own after ZeRO partitioning) this parameter."""

DESLOCK_SEPARATE_GRAD_NORM_GROUPS: Tuple[str, ...] = (DESLOCK_MTP_GRAD_NORM_GROUP,)
"""Registry of grad-norm groups that receive independent gradient clipping.

Extend this tuple (and add a corresponding ``DeviceTier`` mapping in
``GRAD_NORM_GROUP_DEVICE_TIER``) when new detached sub-graphs are introduced.
"""

GRAD_NORM_GROUP_DEVICE_TIER: Dict[str, "DeviceTier"] = {}
"""Populated after ``DeviceTier`` is defined; maps group name → preferred tier."""


# ---------------------------------------------------------------------------
# Hardware topology
# ---------------------------------------------------------------------------


class DeviceTier(Enum):
    """SM-capability tier encoding for Neuron_SP's heterogeneous cluster.

    SM86 covers the two A6000 GPUs (PCIe-attached, 48 GB each).
    SM90 covers the single H100 NVL (PCIe-attached, 96 GB).
    CPU represents the 1.5 TB host DRAM used by the SLoC.

    This enum is the DES-LOC counterpart of Megatron's implicit assumption that
    all devices are homogeneous SM90.  Having an explicit tier lets the gradient
    clipper decide which collective group to use and whether FP8 paths are safe.
    """

    SM86 = auto()  # A6000 — no FP8 native support
    SM90 = auto()  # H100 NVL — FP8 via Transformer Engine
    CPU = auto()   # Host DRAM / SLoC backing store


# Populate the group→tier mapping now that DeviceTier is defined.
GRAD_NORM_GROUP_DEVICE_TIER[DESLOCK_MTP_GRAD_NORM_GROUP] = DeviceTier.SM86
"""MTP heads are placed on A6000 (SM86) in the Neuron_SP topology."""


def _sm_capability_for_device(device: torch.device) -> DeviceTier:
    """Return the ``DeviceTier`` for a given torch device.

    This uses ``torch.cuda.get_device_properties`` to read the real SM version,
    so it works correctly even if the runtime device assignment changes.  The
    mapping is conservative: only confirmed SM90 hardware gets the SM90 tier.

    Args:
        device: A ``torch.device`` with type ``'cuda'``.

    Returns:
        The corresponding ``DeviceTier``.
    """
    if device.type != "cuda":
        return DeviceTier.CPU
    props = torch.cuda.get_device_properties(device)
    major = props.major
    if major >= 9:
        return DeviceTier.SM90
    return DeviceTier.SM86


def _fp8_safe_for_device(device: torch.device) -> bool:
    """Return whether FP8 precision-aware optimisation is safe on *device*.

    Upstream Megatron gates this on ``use_precision_aware_optimizer_no_fp8_or_ds_fp8``,
    which is a global optimizer config flag.  DES-LOC must also gate it
    per-device because A6000 (SM86) does not support FP8 natively, so the
    FP8 path must be disabled for params that live on those devices.

    Args:
        device: The device on which the parameter resides.

    Returns:
        ``True`` only if the device's SM capability supports FP8.
    """
    return _sm_capability_for_device(device) == DeviceTier.SM90


# ---------------------------------------------------------------------------
# Parameter metadata
# ---------------------------------------------------------------------------


@dataclass
class HeteroParamMetadata:
    """All DES-LOC metadata attached to a single parameter tensor.

    Upstream Megatron copies only ``shared`` and ``grad_norm_group``
    (via ``copy_optimizer_param_metadata``).  DES-LOC adds ``device_tier``
    so that ZeRO-sharded views of a parameter retain their hardware affinity
    even after the shard is moved across PCIe to a different device's memory.

    Attributes:
        shared: Whether the parameter is shared across model-parallel ranks
            (Megatron semantics — avoids double-counting in grad norms).
        grad_norm_group: The separate grad-norm group this parameter belongs
            to, or ``None`` for the main group.
        device_tier: The SM-capability tier of the device that owns this
            parameter in the DES-LOC topology.
        sloc_key: Optional string key used to look up this parameter's entry
            in the Shared LOcality Cache.  Set at model construction and
            propagated to shard views.
    """

    shared: bool = False
    grad_norm_group: Optional[str] = None
    device_tier: DeviceTier = DeviceTier.SM86
    sloc_key: Optional[str] = None


def _read_param_metadata(param: torch.Tensor) -> HeteroParamMetadata:
    """Read all DES-LOC metadata from a parameter tensor into a dataclass.

    Args:
        param: Source parameter tensor (may or may not have metadata attrs).

    Returns:
        A populated ``HeteroParamMetadata`` instance.
    """
    device_tier = DeviceTier.CPU
    if param.device.type == "cuda":
        device_tier = _sm_capability_for_device(param.device)
    # Override with explicitly stored tier if present (survives device moves).
    stored_tier = getattr(param, DESLOCK_DEVICE_TIER_ATTR, None)
    if stored_tier is not None and isinstance(stored_tier, DeviceTier):
        device_tier = stored_tier

    return HeteroParamMetadata(
        shared=getattr(param, "shared", False),
        grad_norm_group=getattr(param, DESLOCK_GRAD_NORM_GROUP_ATTR, None),
        device_tier=device_tier,
        sloc_key=getattr(param, "_sloc_key", None),
    )


def copy_hetero_param_metadata(
    destination: torch.Tensor,
    source: torch.Tensor,
) -> None:
    """Copy all optimizer-relevant and DES-LOC metadata from source to destination.

    This extends Megatron's ``copy_optimizer_param_metadata`` with the
    additional DES-LOC fields.  Called whenever ZeRO creates a shard view or
    a new precision cast of a parameter.

    Args:
        destination: Target tensor (shard view, fp32 copy, etc.).
        source: Original model parameter carrying authoritative metadata.
    """
    # --- Megatron-compatible fields ---
    if hasattr(source, "shared"):
        destination.shared = source.shared

    if hasattr(source, DESLOCK_GRAD_NORM_GROUP_ATTR):
        group_val = getattr(source, DESLOCK_GRAD_NORM_GROUP_ATTR)
        setattr(destination, DESLOCK_GRAD_NORM_GROUP_ATTR, group_val)

    # --- DES-LOC extension fields ---
    if hasattr(source, DESLOCK_DEVICE_TIER_ATTR):
        tier_val = getattr(source, DESLOCK_DEVICE_TIER_ATTR)
        setattr(destination, DESLOCK_DEVICE_TIER_ATTR, tier_val)

    if hasattr(source, "_sloc_key"):
        destination._sloc_key = source._sloc_key


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _get_param_grad_norm_group(param: torch.Tensor) -> Optional[str]:
    """Return the grad-norm group for *param*, or ``None`` for the main group.

    Direct port of Megatron's ``_get_param_grad_norm_group``.
    """
    return getattr(param, DESLOCK_GRAD_NORM_GROUP_ATTR, None)


def _validate_grad_norm_group(grad_norm_group: str) -> None:
    """Raise ``ValueError`` if *grad_norm_group* is not registered.

    Mirrors Megatron's ``_validate_grad_norm_group``.  Early failure is
    preferable to the group silently falling through to the main norm
    and corrupting gradient statistics.

    Args:
        grad_norm_group: The group name to validate.

    Raises:
        ValueError: If the group is not in ``DESLOCK_SEPARATE_GRAD_NORM_GROUPS``.
    """
    if grad_norm_group not in DESLOCK_SEPARATE_GRAD_NORM_GROUPS:
        raise ValueError(
            f"Unknown grad_norm_group '{grad_norm_group}'.  Register it in "
            "DESLOCK_SEPARATE_GRAD_NORM_GROUPS before tagging parameters with it."
        )


def _is_separate_grad_norm_group(grad_norm_group: Optional[str]) -> bool:
    """Return whether *grad_norm_group* is a registered separate-clip group.

    Args:
        grad_norm_group: Group name or ``None``.

    Returns:
        ``True`` only for registered non-None group names.
    """
    if grad_norm_group is None:
        return False
    _validate_grad_norm_group(grad_norm_group)
    return True


# ---------------------------------------------------------------------------
# Shared LOcality Cache invalidation
# ---------------------------------------------------------------------------


class SLocInvalidator:
    """Invalidates DES-LOC's Shared LOcality Cache for a set of parameters.

    The SLoC is a CPU-resident mirror of recently-used parameter shards that
    accelerates activation checkpointing recomputation by avoiding PCIe
    round-trips.  However, when an optimizer step modifies a parameter, any
    cached version becomes stale.  If the stale entry is read during the next
    forward pass's recomputation phase, it silently injects a wrong activation
    value into the backward graph — a silent correctness bug that is hard to
    detect.

    This class is injected into the gradient clipping pipeline so that before
    any parameter group is stepped, its SLoC entries are evicted.

    In production, ``cache_ref`` should be a ``weakref.ref`` to the actual
    SLoC object (a dict-like mapping ``sloc_key → tensor``).  In testing, a
    plain dict can be passed directly.

    Args:
        cache_ref: A callable returning the SLoC mapping, or the mapping
            itself.  Using a weak reference prevents the clipper from keeping
            the (potentially large) cache alive.
    """

    def __init__(self, cache_ref=None):
        self._cache_ref = cache_ref
        self._eviction_count: int = 0

    def _get_cache(self) -> Optional[Dict]:
        if self._cache_ref is None:
            return None
        if callable(self._cache_ref):
            return self._cache_ref()
        return self._cache_ref

    def invalidate_group(
        self,
        params: List[torch.Tensor],
        grad_norm_group: Optional[str],
    ) -> int:
        """Evict SLoC entries for all parameters in *params*.

        Args:
            params: Parameters about to be updated by the optimizer step.
            grad_norm_group: The group label, used only for logging context.

        Returns:
            Number of cache entries evicted.
        """
        cache = self._get_cache()
        if cache is None:
            return 0

        evicted = 0
        for param in params:
            key = getattr(param, "_sloc_key", None)
            if key is not None and key in cache:
                del cache[key]
                evicted += 1

        if evicted > 0:
            logger.debug(
                "SLoC: evicted %d entries for grad_norm_group=%r before optimizer step",
                evicted,
                grad_norm_group,
            )
        self._eviction_count += evicted
        return evicted

    @property
    def total_evictions(self) -> int:
        """Cumulative number of SLoC evictions since this object was created."""
        return self._eviction_count


# ---------------------------------------------------------------------------
# Gradient norm computation (FP32, heterogeneous-safe)
# ---------------------------------------------------------------------------


def _compute_grad_norm_fp32(
    grads: List[torch.Tensor],
    process_group: Optional[dist.ProcessGroup] = None,
) -> float:
    """Compute the global FP32 L2 norm of *grads* across *process_group*.

    This is a reimplementation of Megatron's ``get_grad_norm_fp32`` adapted
    for DES-LOC's heterogeneous device topology:

    1. All gradients are cast to FP32 before squaring to avoid overflow on
       FP16 accumulators — critical on A6000 (SM86) where FP16 range is the
       bottleneck.
    2. The local sum-of-squares is computed on the device that hosts the grad,
       then moved to CPU for the all-reduce sum, avoiding unnecessary PCIe
       traffic for the intermediate accumulation.
    3. The all-reduce uses ``ReduceOp.SUM`` over *process_group*.  If
       *process_group* is ``None``, the default group (all ranks) is used,
       matching Megatron's LayerWise global-reduction semantics.

    Args:
        grads: List of gradient tensors (may be on mixed devices).
        process_group: The distributed process group for the all-reduce.

    Returns:
        The global L2 norm as a Python float, or ``0.0`` if *grads* is empty.
    """
    if not grads:
        return 0.0

    # Accumulate sum-of-squares per device to minimize cross-device copies.
    device_sums: Dict[torch.device, torch.Tensor] = {}
    for grad in grads:
        dev = grad.device
        g_fp32 = grad.float()
        local_sq = (g_fp32 * g_fp32).sum()
        if dev not in device_sums:
            device_sums[dev] = local_sq
        else:
            device_sums[dev] = device_sums[dev] + local_sq

    # Sum across devices on CPU to avoid picking a "preferred" GPU.
    total_sq = sum(sq.cpu() for sq in device_sums.values())

    if dist.is_available() and dist.is_initialized():
        total_sq_tensor = total_sq.cuda()
        dist.all_reduce(total_sq_tensor, op=dist.ReduceOp.SUM, group=process_group)
        total_sq = total_sq_tensor.cpu()

    norm = math.sqrt(float(total_sq.item()))
    return norm


# ---------------------------------------------------------------------------
# Heterogeneous process-group registry
# ---------------------------------------------------------------------------


class HeteroProcessGroupRegistry:
    """Maps (DeviceTier, grad_norm_group) pairs to distributed process groups.

    In Megatron's homogeneous setting, a single ``grad_stats_parallel_group``
    suffices.  DES-LOC must maintain separate groups because:

    - The A6000 pair (SM86) cannot participate in H100-initiated NVLink
      collectives (there is no NVLink in this cluster, but the principle
      generalises).
    - PCIe bandwidth is the bottleneck; we want grad-norm all-reduces to
      stay within the smallest process group that spans all shards of the
      relevant parameter group.

    This registry is populated at training startup and consulted by
    ``HeteroMTPGradClipper`` when selecting the process group for a
    grad-norm all-reduce.

    Args:
        default_group: Fallback group used when no specific mapping exists.
    """

    def __init__(self, default_group: Optional[dist.ProcessGroup] = None):
        self._default_group = default_group
        self._registry: Dict[Tuple[DeviceTier, str], dist.ProcessGroup] = {}

    def register(
        self,
        tier: DeviceTier,
        grad_norm_group: str,
        process_group: dist.ProcessGroup,
    ) -> None:
        """Register *process_group* for a (tier, grad_norm_group) pair.

        Args:
            tier: The device tier that participates in this group.
            grad_norm_group: The grad-norm group label.
            process_group: The distributed process group.
        """
        _validate_grad_norm_group(grad_norm_group)
        self._registry[(tier, grad_norm_group)] = process_group
        logger.info(
            "HeteroProcessGroupRegistry: registered process group for "
            "tier=%s, grad_norm_group=%r",
            tier.name,
            grad_norm_group,
        )

    def get(
        self,
        tier: DeviceTier,
        grad_norm_group: Optional[str],
    ) -> Optional[dist.ProcessGroup]:
        """Return the process group for *(tier, grad_norm_group)*, or the default.

        Args:
            tier: Device tier.
            grad_norm_group: Group label, or ``None`` for main group.

        Returns:
            The registered process group, or ``self._default_group``.
        """
        if grad_norm_group is not None:
            key = (tier, grad_norm_group)
            if key in self._registry:
                return self._registry[key]
        return self._default_group


# ---------------------------------------------------------------------------
# Gradient router
# ---------------------------------------------------------------------------


class HeteroGradNormGroupRouter:
    """Partitions parameters into main vs. per-group buckets for independent clipping.

    This is the DES-LOC counterpart of the inline partitioning logic that
    Megatron introduced in ``clip_grad_norm`` and ``ChainedOptimizer.step``.
    By factoring it into a class we gain:

    1. Reusability across ``HeteroMTPGradClipper`` and ``HeteroChainedOptimizer``.
    2. A stable cache of the partition result — the partition does not change
       within a training step, so computing it once and reusing it for both
       the norm computation and the clipping is correct and saves iteration cost.
    3. Device-tier awareness: the router also groups params by ``DeviceTier``
       so the caller can choose the appropriate process group for each
       all-reduce.

    Args:
        params: Flat list of all parameters managed by the optimizer.
    """

    def __init__(self, params: List[torch.nn.Parameter]):
        self._params = params
        self._main_params: List[torch.nn.Parameter] = []
        self._by_group: Dict[str, List[torch.nn.Parameter]] = {}
        self._by_group_and_tier: Dict[str, Dict[DeviceTier, List[torch.nn.Parameter]]] = {}
        self._partitioned = False

    def _ensure_partitioned(self) -> None:
        if self._partitioned:
            return
        for p in self._params:
            group = _get_param_grad_norm_group(p)
            if _is_separate_grad_norm_group(group):
                self._by_group.setdefault(group, []).append(p)
                tier = getattr(p, DESLOCK_DEVICE_TIER_ATTR, None)
                if tier is None and p.device.type == "cuda":
                    tier = _sm_capability_for_device(p.device)
                elif tier is None:
                    tier = DeviceTier.CPU
                self._by_group_and_tier.setdefault(group, {}).setdefault(tier, []).append(p)
            else:
                self._main_params.append(p)
        self._partitioned = True

    @property
    def main_params(self) -> List[torch.nn.Parameter]:
        """Parameters that belong to the main (backbone) grad-norm group."""
        self._ensure_partitioned()
        return self._main_params

    @property
    def group_names(self) -> Set[str]:
        """Set of grad-norm group names that have at least one parameter."""
        self._ensure_partitioned()
        return set(self._by_group.keys())

    def params_for_group(self, grad_norm_group: str) -> List[torch.nn.Parameter]:
        """Return all parameters tagged with *grad_norm_group*.

        Args:
            grad_norm_group: The group label.

        Returns:
            List of parameters in that group (possibly empty).
        """
        self._ensure_partitioned()
        return self._by_group.get(grad_norm_group, [])

    def params_for_group_and_tier(
        self,
        grad_norm_group: str,
        tier: DeviceTier,
    ) -> List[torch.nn.Parameter]:
        """Return parameters in *grad_norm_group* that reside on *tier*.

        Used by the process-group registry lookup to ensure all-reduces stay
        within the correct hardware partition.

        Args:
            grad_norm_group: Group label.
            tier: Device tier to filter by.

        Returns:
            Possibly-empty list of parameters.
        """
        self._ensure_partitioned()
        return self._by_group_and_tier.get(grad_norm_group, {}).get(tier, [])

    def grads_for_group(
        self,
        grad_norm_group: Optional[str],
        param_filter: Optional[Callable[[torch.nn.Parameter], bool]] = None,
    ) -> List[torch.Tensor]:
        """Return non-None gradients for the specified group.

        When *grad_norm_group* is ``None``, returns main-group gradients
        (excluding shared and TP-duplicate params, mirroring Megatron's
        ``_filter_grads_for_norm``).

        Args:
            grad_norm_group: Group label, or ``None`` for main group.
            param_filter: Optional additional predicate applied after the
                group filter.  Useful for excluding TP-duplicate params.

        Returns:
            List of gradient tensors.
        """
        self._ensure_partitioned()
        if grad_norm_group is None:
            params = self._main_params
        else:
            _validate_grad_norm_group(grad_norm_group)
            params = self._by_group.get(grad_norm_group, [])

        grads = []
        for p in params:
            if param_filter is not None and not param_filter(p):
                continue
            # Exclude shared params (double-counting guard from Megatron).
            if getattr(p, "shared", False):
                continue
            # Exclude TP-duplicate params.  In DES-LOC, TP shards always live
            # on the same device tier, so this check is purely logical.
            if getattr(p, "tensor_model_parallel", False) and not getattr(
                p, "sequence_parallel", False
            ):
                if not getattr(p, "is_tensor_parallel_master", True):
                    continue
            # Collect the actual gradient tensor.
            grad = getattr(p, "main_grad", None) or p.grad
            if grad is not None:
                grads.append(grad)
        return grads


# ---------------------------------------------------------------------------
# Core clipper
# ---------------------------------------------------------------------------


class HeteroMTPGradClipper:
    """Heterogeneous MTP gradient clipper for DES-LOC / Neuron_SP.

    This class implements the grad-clipping logic introduced in Megatron
    commit 5b16c99, reinterpreted for a cluster with mixed SM86 (A6000) and
    SM90 (H100) devices connected over PCIe without NVLink.

    The key invariants upheld by this class:

    1. **Norm isolation**: MTP head gradients are never mixed with backbone
       gradients when computing the norm used for clipping.  This prevents
       MTP gradient spikes from triggering skip-threshold logic on backbone
       params (Megatron's primary motivation).

    2. **Device-group consistency**: All-reduces use the process group that
       spans exactly the devices holding the relevant parameter shards,
       preventing asymmetric collective hang from PCIe topology differences.

    3. **SLoC coherence**: Before each optimizer sub-step, the SLoC
       invalidator evicts entries for the about-to-be-updated parameter
       group, ensuring the locality cache never serves stale values.

    4. **FP8 gating per device tier**: The ``use_decoupled_grad`` flag is
       computed per-parameter based on device tier, not globally, because
       SM86 does not support FP8.

    5. **Caching of group membership**: ``has_grad_norm_group`` performs at
       most one all-reduce per group per training run and caches the result,
       matching Megatron's caching pattern but tracking which process group
       was used.

    Args:
        process_group_registry: Registry mapping (tier, group) → dist group.
        sloc_invalidator: SLoC eviction helper.  Pass ``None`` to disable.
        clip_grad: Maximum gradient norm (per group).
        grad_norm_skip_threshold: Skip the update if main norm exceeds this.
    """

    def __init__(
        self,
        process_group_registry: Optional[HeteroProcessGroupRegistry] = None,
        sloc_invalidator: Optional[SLocInvalidator] = None,
        clip_grad: float = 1.0,
        grad_norm_skip_threshold: float = float("inf"),
    ):
        self._pg_registry = process_group_registry or HeteroProcessGroupRegistry()
        self._sloc = sloc_invalidator or SLocInvalidator()
        self.clip_grad = clip_grad
        self.grad_norm_skip_threshold = grad_norm_skip_threshold

        # Cached results (reset each step).
        self.grad_norms_by_group: Dict[str, float] = {}
        self._has_group_cache: Dict[str, bool] = {}

    def reset_step_state(self) -> None:
        """Clear per-step cached gradient norms.  Call at the start of each step."""
        self.grad_norms_by_group = {}

    def has_grad_norm_group(
        self,
        grad_norm_group: str,
        params: List[torch.nn.Parameter],
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> bool:
        """Return whether any rank in *process_group* owns params in *grad_norm_group*.

        Implements the Megatron pattern: compute locally, all-reduce MAX,
        cache.  DES-LOC extension: the cache key includes the process group
        identity so the correct collective is used even if this method is
        called with different groups across ranks.

        Args:
            grad_norm_group: The group label to check.
            params: Parameters visible to this rank.
            process_group: The distributed group over which to check.
                Defaults to the default process group.

        Returns:
            ``True`` if any rank in *process_group* has params tagged with
            *grad_norm_group*.
        """
        _validate_grad_norm_group(grad_norm_group)
        cache_key = (
            grad_norm_group,
            id(process_group) if process_group is not None else None,
        )
        if cache_key in self._has_group_cache:
            return self._has_group_cache[cache_key]

        local_has = any(
            _get_param_grad_norm_group(p) == grad_norm_group
            for p in params
        )

        if dist.is_available() and dist.is_initialized():
            # Pick a CUDA device for the flag tensor; prefer the device of any
            # param in the group, fall back to cuda:0.
            flag_device = torch.device("cuda:0")
            for p in params:
                if p.device.type == "cuda":
                    flag_device = p.device
                    break
            flag = torch.tensor(
                [1 if local_has else 0], dtype=torch.int32, device=flag_device
            )
            dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=process_group)
            result = bool(flag.item() > 0)
        else:
            result = local_has

        self._has_group_cache[cache_key] = result
        return result

    def _use_decoupled_grad_for_params(
        self,
        params: List[torch.nn.Parameter],
        optimizer_config,
    ) -> bool:
        """Determine whether ``clip_grad_by_total_norm_fp32`` should use decoupled grad.

        Megatron's logic: ``use_precision_aware_optimizer_no_fp8_or_ds_fp8`` OR
        (``use_precision_aware_optimizer`` AND first param is an FSDP param).

        DES-LOC extension: additionally gate on device tier — SM86 devices
        (A6000) do not support FP8, so even if the global optimizer config
        enables the precision-aware path, we must disable it for SM86 params.

        Args:
            params: The parameter list whose decoupled-grad flag is needed.
            optimizer_config: DeepSpeed / Megatron optimizer config object.

        Returns:
            Boolean flag to pass to ``clip_grad_by_total_norm_fp32``.
        """
        if not params:
            return False

        # Check device tier of the first param.
        first_param = params[0]
        device_supports_fp8 = _fp8_safe_for_device(first_param.device)

        global_fp8_path = getattr(
            optimizer_config,
            "use_precision_aware_optimizer_no_fp8_or_ds_fp8",
            False,
        )
        fsdp_path = getattr(
            optimizer_config, "use_precision_aware_optimizer", False
        ) and getattr(first_param, "__fsdp_param__", False)

        # If global config requests FP8 path but device cannot support it,
        # log once and fall through to the non-FP8 path.
        if (global_fp8_path or fsdp_path) and not device_supports_fp8:
            tier = _sm_capability_for_device(first_param.device)
            logger.debug(
                "Disabling precision-aware (FP8) gradient path for params on "
                "%s (SM86 does not support FP8); using standard FP32 grad path.",
                tier.name,
            )
            return False

        return global_fp8_path or fsdp_path

    def clip_grad_by_total_norm_fp32_hetero(
        self,
        params: List[torch.nn.Parameter],
        max_norm: float,
        total_norm: float,
        use_decoupled_grad: bool,
    ) -> None:
        """Clip *params* gradients by *total_norm* to a maximum of *max_norm*.

        Pure-Python reimplementation that avoids importing Megatron's
        ``clip_grad_by_total_norm_fp32`` (which is not available in the
        DeepSpeed-only build of Neuron_SP).  Gradient tensors are modified
        in-place.

        If ``use_decoupled_grad`` is True, the ``main_grad`` attribute is
        used instead of ``.grad`` (Megatron FSDP / precision-aware path).

        Args:
            params: Parameters to clip.
            max_norm: Target maximum norm.
            total_norm: Precomputed L2 norm of the gradient group.
            use_decoupled_grad: Whether to clip ``param.main_grad`` instead
                of ``param.grad``.
        """
        if total_norm <= 0.0 or max_norm <= 0.0:
            return
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef >= 1.0:
            return  # Already within bounds; no scaling needed.

        for param in params:
            grad = (
                getattr(param, "main_grad", None)
                if use_decoupled_grad
                else param.grad
            )
            if grad is not None:
                grad.detach().mul_(clip_coef)

    def compute_grad_norms_by_group(
        self,
        router: HeteroGradNormGroupRouter,
        default_process_group: Optional[dist.ProcessGroup] = None,
    ) -> Dict[str, float]:
        """Compute gradient norms for all registered separate grad-norm groups.

        This is the DES-LOC counterpart of Megatron's
        ``_compute_grad_norms_by_group``, extended to use per-tier process
        groups from the registry.

        For each registered group (currently only ``'mtp'``):
          1. Check whether any rank has params in the group (cached).
          2. If yes, collect the gradients and compute the global FP32 norm
             using the process group corresponding to the group's device tier.
          3. Store the norm in ``self.grad_norms_by_group`` and return.

        Args:
            router: Partitioned parameter router for this optimizer.
            default_process_group: Fallback when no tier-specific group is
                registered.

        Returns:
            Dict mapping group name → computed norm (only groups present on
            any rank are included).
        """
        self.grad_norms_by_group = {}
        all_params = router.main_params + [
            p for g in DESLOCK_SEPARATE_GRAD_NORM_GROUPS for p in router.params_for_group(g)
        ]

        for group_name in DESLOCK_SEPARATE_GRAD_NORM_GROUPS:
            preferred_tier = GRAD_NORM_GROUP_DEVICE_TIER.get(group_name, DeviceTier.SM86)
            pg = self._pg_registry.get(preferred_tier, group_name) or default_process_group

            if not self.has_grad_norm_group(group_name, all_params, process_group=pg):
                continue

            grads = router.grads_for_group(group_name)
            norm = _compute_grad_norm_fp32(grads, process_group=pg)
            self.grad_norms_by_group[group_name] = norm

            logger.debug(
                "Computed grad norm for group=%r: %.6f (process_group=%s, tier=%s)",
                group_name,
                norm,
                pg,
                preferred_tier.name,
            )

        return self.grad_norms_by_group

    def clip_all_groups(
        self,
        router: HeteroGradNormGroupRouter,
        main_grad_norm: float,
        optimizer_config,
        default_process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        """Clip main params and all separate groups independently.

        Orchestrates the full clipping pipeline:
          1. Compute group norms for all separate groups.
          2. Evict SLoC entries for the about-to-be-stepped main params.
          3. Clip main params using *main_grad_norm*.
          4. For each registered separate group, evict SLoC entries and clip
             using the group-specific norm.

        The skip-threshold check (Megatron: ``grad_norm > skip_threshold``)
        applies **only** to the main norm, so MTP grad spikes never suppress
        backbone updates.

        Args:
            router: The parameter router for this step.
            main_grad_norm: Precomputed norm of the main (backbone) gradients.
            optimizer_config: Optimizer configuration (provides ``clip_grad``
                and precision flags).
            default_process_group: Fallback process group.
        """
        if self.clip_grad <= 0.0:
            return

        self.compute_grad_norms_by_group(router, default_process_group=default_process_group)

        # --- Main parameter group ---
        main_params = router.main_params
        if main_params:
            use_decoupled = self._use_decoupled_grad_for_params(main_params, optimizer_config)
            self._sloc.invalidate_group(main_params, grad_norm_group=None)
            self.clip_grad_by_total_norm_fp32_hetero(
                main_params,
                max_norm=self.clip_grad,
                total_norm=main_grad_norm,
                use_decoupled_grad=use_decoupled,
            )

        # --- Separate grad-norm groups ---
        for group_name in DESLOCK_SEPARATE_GRAD_NORM_GROUPS:
            group_params = router.params_for_group(group_name)
            if not group_params:
                continue
            group_norm = self.grad_norms_by_group.get(group_name)
            if group_norm is None:
                # No rank in the process group has params for this group;
                # skip to avoid a collective imbalance.
                continue
            use_decoupled = self._use_decoupled_grad_for_params(group_params, optimizer_config)
            self._sloc.invalidate_group(group_params, grad_norm_group=group_name)
            self.clip_grad_by_total_norm_fp32_hetero(
                group_params,
                max_norm=self.clip_grad,
                total_norm=group_norm,
                use_decoupled_grad=use_decoupled,
            )
            logger.info(
                "Clipped grad_norm_group=%r: pre-clip norm=%.6f → clip threshold=%.4f, "
                "use_decoupled_grad=%s",
                group_name,
                group_norm,
                self.clip_grad,
                use_decoupled,
            )


# ---------------------------------------------------------------------------
# DeepSpeed optimizer wrapper
# ---------------------------------------------------------------------------


class HeteroChainedOptimizer:
    """DeepSpeed-compatible chained optimizer with DES-LOC heterogeneous grad clipping.

    Mirrors Megatron's ``ChainedOptimizer.step`` logic (post-5b16c99) but
    operates within DeepSpeed's optimizer wrapper interface.  Key differences:

    1. Uses ``HeteroGradNormGroupRouter`` to partition parameters once per
       step instead of inlining the partition logic.
    2. Delegates all clipping to ``HeteroMTPGradClipper``, which handles SLoC
       eviction, device-tier-aware process group selection, and FP8 gating.
    3. The skip-threshold check is applied only to the main norm, matching
       the upstream intention.

    This class is *not* a subclass of any DeepSpeed class; it composes over
    a list of sub-optimizers to avoid tight coupling with DeepSpeed internals
    that may change.

    Args:
        sub_optimizers: List of DeepSpeed or Megatron sub-optimizers, each
            exposing ``get_parameters()``, ``step_with_ready_grads()``,
            ``get_grad_stats_parallel_group()``, and a ``config`` attribute.
        clipper: Pre-constructed ``HeteroMTPGradClipper`` instance.
        grad_norm_skip_threshold: Skip the update if main norm exceeds this.
    """

    def __init__(
        self,
        sub_optimizers: List,
        clipper: HeteroMTPGradClipper,
        grad_norm_skip_threshold: float = float("inf"),
    ):
        self.sub_optimizers = sub_optimizers
        self._clipper = clipper
        self.grad_norm_skip_threshold = grad_norm_skip_threshold

        # Exposed for logging / monitoring (matches Megatron's attribute name).
        self.grad_norms_by_group: Dict[str, float] = {}

    def _all_params(self) -> List[torch.nn.Parameter]:
        """Flatten parameters from all sub-optimizers."""
        params = []
        for opt in self.sub_optimizers:
            params.extend(opt.get_parameters())
        return params

    def _main_grad_norm(
        self,
        router: HeteroGradNormGroupRouter,
        process_group: Optional[dist.ProcessGroup],
    ) -> float:
        """Compute the main (backbone) gradient norm using the router.

        Args:
            router: Pre-partitioned parameter router.
            process_group: The global grad-stats process group.

        Returns:
            Main gradient norm as a float.
        """
        main_grads = router.grads_for_group(None)
        return _compute_grad_norm_fp32(main_grads, process_group=process_group)

    def _default_process_group(self) -> Optional[dist.ProcessGroup]:
        """Return the first available grad-stats group from sub-optimizers."""
        for opt in self.sub_optimizers:
            if hasattr(opt, "get_grad_stats_parallel_group"):
                pg = opt.get_grad_stats_parallel_group()
                if pg is not None:
                    return pg
        return None

    def prepare_grads(self) -> bool:
        """Run gradient preparation for all sub-optimizers.

        Returns:
            ``True`` if any sub-optimizer reports ``found_inf`` (overflow).
        """
        found_inf = False
        for opt in self.sub_optimizers:
            if hasattr(opt, "prepare_grads"):
                found_inf = opt.prepare_grads() or found_inf
        return found_inf

    def step(self) -> Tuple[bool, Optional[float], Optional[float]]:
        """Perform a full optimizer step with heterogeneous grad clipping.

        Step sequence:
          1. Prepare gradients (overflow check).
          2. Partition all parameters into main vs. group buckets.
          3. Compute the main gradient norm.
          4. Check the skip threshold (main norm only).
          5. Clip gradients for main and all registered separate groups.
          6. Step each sub-optimizer.

        Returns:
            Tuple of (update_successful, main_grad_norm, num_zeros_in_grad).
            ``num_zeros_in_grad`` is always ``None`` (not computed here; use
            a dedicated count-zeros pass if needed).
        """
        self._clipper.reset_step_state()

        found_inf = self.prepare_grads()
        if found_inf:
            logger.warning(
                "HeteroChainedOptimizer: gradient overflow detected, skipping step."
            )
            return False, None, None

        all_params = self._all_params()
        router = HeteroGradNormGroupRouter(all_params)
        default_pg = self._default_process_group()

        main_grad_norm = self._main_grad_norm(router, process_group=default_pg)

        # Skip-threshold check applies only to main norm (DES-LOC / Megatron intent).
        if main_grad_norm > self.grad_norm_skip_threshold:
            logger.info(
                "HeteroChainedOptimizer: skipping update — main grad norm %.6f "
                "exceeds skip threshold %.4f.",
                main_grad_norm,
                self.grad_norm_skip_threshold,
            )
            return False, main_grad_norm, None

        # Determine whether any sub-optimizer requests clipping.
        should_clip = any(
            not getattr(opt, "is_stub_optimizer", False)
            and getattr(getattr(opt, "config", None), "clip_grad", 0.0) > 0.0
            for opt in self.sub_optimizers
        )

        if should_clip:
            # Use the clip_grad from the first non-stub optimizer.
            for opt in self.sub_optimizers:
                if not getattr(opt, "is_stub_optimizer", False):
                    clip_grad = getattr(getattr(opt, "config", None), "clip_grad", 0.0)
                    if clip_grad > 0.0:
                        self._clipper.clip_grad = clip_grad
                        break

            self._clipper.clip_all_groups(
                router=router,
                main_grad_norm=main_grad_norm,
                optimizer_config=self.sub_optimizers[0].config
                if self.sub_optimizers
                else None,
                default_process_group=default_pg,
            )
            self.grad_norms_by_group = dict(self._clipper.grad_norms_by_group)

        # Step each sub-optimizer.
        update_successful = True
        for opt in self.sub_optimizers:
            if getattr(opt, "is_stub_optimizer", False):
                continue
            params = opt.get_parameters()
            if not params:
                continue
            ok = opt.step_with_ready_grads()
            update_successful = update_successful and ok

        return update_successful, main_grad_norm, None


# ---------------------------------------------------------------------------
# MTP parameter tagging utility
# ---------------------------------------------------------------------------


def tag_mtp_parameters(
    module: torch.nn.Module,
    sloc_key_prefix: Optional[str] = None,
) -> int:
    """Tag all parameters of *module* for separate MTP gradient clipping.

    This mirrors the tagging logic introduced in Megatron's
    ``MultiTokenPredictionBlock.__init__`` (commit 5b16c99):

    .. code-block:: python

        if self.config.mtp_detach_heads:
            for param in self.parameters():
                param.grad_norm_group = 'mtp'

    DES-LOC extension: also tags each parameter with:
      - ``device_tier``: the SM capability of the parameter's current device.
      - ``_sloc_key``: a unique key for SLoC eviction, derived from the
        parameter's name if *sloc_key_prefix* is provided.

    Args:
        module: The MTP head module whose parameters should be tagged.
        sloc_key_prefix: Optional prefix for SLoC cache keys.  When provided,
            each parameter receives a ``_sloc_key`` of the form
            ``"{sloc_key_prefix}:{param_name}"``.

    Returns:
        The number of parameters that were tagged.
    """
    tagged = 0
    for name, param in module.named_parameters():
        setattr(param, DESLOCK_GRAD_NORM_GROUP_ATTR, DESLOCK_MTP_GRAD_NORM_GROUP)

        # Device tier annotation.
        if param.device.type == "cuda":
            tier = _sm_capability_for_device(param.device)
        else:
            tier = DeviceTier.CPU
        setattr(param, DESLOCK_DEVICE_TIER_ATTR, tier)

        # SLoC key annotation.
        if sloc_key_prefix is not None:
            safe_name = name.replace(".", "_")
            param._sloc_key = f"{sloc_key_prefix}:{safe_name}"

        tagged += 1

    if tagged > 0:
        logger.debug(
            "tag_mtp_parameters: tagged %d parameters on module %s "
            "(sloc_key_prefix=%r, grad_norm_group=%r)",
            tagged,
            type(module).__name__,
            sloc_key_prefix,
            DESLOCK_MTP_GRAD_NORM_GROUP,
        )
    return tagged


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    import traceback

    # Minimal test harness — no pytest dependency required.
    _PASS = "\033[92mPASS\033[0m"
    _FAIL = "\033[91mFAIL\033[0m"
    _tests_run = 0
    _tests_failed = 0

    def _run_test(name: str, fn):
        global _tests_run, _tests_failed
        _tests_run += 1
        try:
            fn()
            print(f"  {_PASS}  {name}")
        except Exception:  # noqa: BLE001
            _tests_failed += 1
            print(f"  {_FAIL}  {name}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Test 1: copy_hetero_param_metadata propagates all fields
    # ------------------------------------------------------------------
    def _test_copy_metadata():
        src = torch.nn.Parameter(torch.randn(4, 4))
        src.shared = True
        setattr(src, DESLOCK_GRAD_NORM_GROUP_ATTR, DESLOCK_MTP_GRAD_NORM_GROUP)
        setattr(src, DESLOCK_DEVICE_TIER_ATTR, DeviceTier.SM86)
        src._sloc_key = "mtp:layer0_weight"

        dst = torch.nn.Parameter(torch.randn(4, 4))
        copy_hetero_param_metadata(dst, src)

        assert getattr(dst, "shared", False) is True
        assert getattr(dst, DESLOCK_GRAD_NORM_GROUP_ATTR, None) == DESLOCK_MTP_GRAD_NORM_GROUP
        assert getattr(dst, DESLOCK_DEVICE_TIER_ATTR, None) == DeviceTier.SM86
        assert getattr(dst, "_sloc_key", None) == "mtp:layer0_weight"

    _run_test("copy_hetero_param_metadata: all fields propagated", _test_copy_metadata)

    # ------------------------------------------------------------------
    # Test 2: copy_hetero_param_metadata with no metadata on source
    # ------------------------------------------------------------------
    def _test_copy_metadata_empty():
        src = torch.nn.Parameter(torch.randn(2, 2))
        dst = torch.nn.Parameter(torch.randn(2, 2))
        copy_hetero_param_metadata(dst, src)
        # No attributes should appear on dst.
        assert not hasattr(dst, "shared")
        assert not hasattr(dst, DESLOCK_GRAD_NORM_GROUP_ATTR)
        assert not hasattr(dst, DESLOCK_DEVICE_TIER_ATTR)

    _run_test(
        "copy_hetero_param_metadata: no metadata on source → dst stays clean",
        _test_copy_metadata_empty,
    )

    # ------------------------------------------------------------------
    # Test 3: _validate_grad_norm_group accepts known, rejects unknown
    # ------------------------------------------------------------------
    def _test_validate_group():
        _validate_grad_norm_group(DESLOCK_MTP_GRAD_NORM_GROUP)  # Must not raise.
        try:
            _validate_grad_norm_group("nonexistent_group")
            raise AssertionError("Expected ValueError was not raised")
        except ValueError as exc:
            assert "Unknown grad_norm_group" in str(exc)

    _run_test("_validate_grad_norm_group: known passes, unknown raises", _test_validate_group)

    # ------------------------------------------------------------------
    # Test 4: HeteroGradNormGroupRouter partitions params correctly
    # ------------------------------------------------------------------
    def _test_router_partition():
        main_params = [torch.nn.Parameter(torch.randn(3, 3)) for _ in range(3)]
        mtp_params = [torch.nn.Parameter(torch.randn(3, 3)) for _ in range(2)]
        for p in mtp_params:
            setattr(p, DESLOCK_GRAD_NORM_GROUP_ATTR, DESLOCK_MTP_GRAD_NORM_GROUP)

        router = HeteroGradNormGroupRouter(main_params + mtp_params)

        assert len(router.main_params) == 3
        assert len(router.params_for_group(DESLOCK_MTP_GRAD_NORM_GROUP)) == 2
        assert DESLOCK_MTP_GRAD_NORM_GROUP in router.group_names

    _run_test("HeteroGradNormGroupRouter: partition into main and mtp", _test_router_partition)

    # ------------------------------------------------------------------
    # Test 5: HeteroGradNormGroupRouter grads_for_group excludes None grads
    # ------------------------------------------------------------------
    def _test_router_grads_none():
        p_with_grad = torch.nn.Parameter(torch.randn(2, 2))
        p_with_grad.grad = torch.randn(2, 2)
        p_no_grad = torch.nn.Parameter(torch.randn(2, 2))
        # p_no_grad.grad is None

        router = HeteroGradNormGroupRouter([p_with_grad, p_no_grad])
        grads = router.grads_for_group(None)
        assert len(grads) == 1
        assert torch.equal(grads[0], p_with_grad.grad)

    _run_test(
        "HeteroGradNormGroupRouter: grads_for_group skips params with None grad",
        _test_router_grads_none,
    )

    # ------------------------------------------------------------------
    # Test 6: HeteroGradNormGroupRouter grads_for_group excludes shared params
    # ------------------------------------------------------------------
    def _test_router_grads_shared():
        p_normal = torch.nn.Parameter(torch.randn(2, 2))
        p_normal.grad = torch.ones(2, 2)
        p_shared = torch.nn.Parameter(torch.randn(2, 2))
        p_shared.grad = torch.ones(2, 2)
        p_shared.shared = True

        router = HeteroGradNormGroupRouter([p_normal, p_shared])
        grads = router.grads_for_group(None)
        assert len(grads) == 1
        assert torch.equal(grads[0], p_normal.grad)

    _run_test(
        "HeteroGradNormGroupRouter: grads_for_group excludes shared params",
        _test_router_grads_shared,
    )

    # ------------------------------------------------------------------
    # Test 7: _compute_grad_norm_fp32 correctness (single process)
    # ------------------------------------------------------------------
    def _test_compute_norm():
        # 4×4 tensor of all ones: norm = sqrt(16) = 4.0
        g = torch.ones(4, 4)
        norm = _compute_grad_norm_fp32([g])
        assert abs(norm - 4.0) < 1e-5, f"Expected 4.0, got {norm}"

        # Two tensors: sum-of-squares = 16 + 16 = 32; norm = sqrt(32)
        norm2 = _compute_grad_norm_fp32([g, g])
        assert abs(norm2 - math.sqrt(32.0)) < 1e-5, f"Expected sqrt(32), got {norm2}"

    _run_test("_compute_grad_norm_fp32: correctness on CPU tensors", _test_compute_norm)

    # ------------------------------------------------------------------
    # Test 8: _compute_grad_norm_fp32 returns 0.0 for empty list
    # ------------------------------------------------------------------
    def _test_compute_norm_empty():
        assert _compute_grad_norm_fp32([]) == 0.0

    _run_test("_compute_grad_norm_fp32: empty list returns 0.0", _test_compute_norm_empty)

    # ------------------------------------------------------------------
    # Test 9: HeteroMTPGradClipper.clip_grad_by_total_norm_fp32_hetero clips correctly
    # ------------------------------------------------------------------
    def _test_clipper_clips():
        # grad norm = sqrt(16 * 100) = 40.0; clip to 1.0 → scale = 1/40
        p = torch.nn.Parameter(torch.zeros(4, 4))
        p.grad = torch.full((4, 4), 10.0)  # norm = sqrt(16 * 100) = 40
        grad_norm = math.sqrt((p.grad ** 2).sum().item())

        clipper = HeteroMTPGradClipper(clip_grad=1.0)
        clipper.clip_grad_by_total_norm_fp32_hetero(
            [p], max_norm=1.0, total_norm=grad_norm, use_decoupled_grad=False
        )
        new_norm = math.sqrt((p.grad ** 2).sum().item())
        assert abs(new_norm - 1.0) < 1e-4, f"Expected norm ≈ 1.0 after clipping, got {new_norm}"

    _run_test(
        "HeteroMTPGradClipper.clip_grad_by_total_norm_fp32_hetero: clips to max_norm",
        _test_clipper_clips,
    )

    # ------------------------------------------------------------------
    # Test 10: clip_grad_by_total_norm_fp32_hetero is no-op when norm < max_norm
    # ------------------------------------------------------------------
    def _test_clipper_noop():
        p = torch.nn.Parameter(torch.zeros(2, 2))
        p.grad = torch.full((2, 2), 0.1)  # norm ≈ 0.2 < max_norm=1.0
        original_grad = p.grad.clone()
        grad_norm = math.sqrt((p.grad ** 2).sum().item())

        clipper = HeteroMTPGradClipper(clip_grad=1.0)
        clipper.clip_grad_by_total_norm_fp32_hetero(
            [p], max_norm=1.0, total_norm=grad_norm, use_decoupled_grad=False
        )
        assert torch.allclose(p.grad, original_grad), "Grad should be unchanged when norm < max"

    _run_test(
        "HeteroMTPGradClipper.clip_grad_by_total_norm_fp32_hetero: no-op when norm < max",
        _test_clipper_noop,
    )

    # ------------------------------------------------------------------
    # Test 11: SLocInvalidator evicts correct keys
    # ------------------------------------------------------------------
    def _test_sloc_eviction():
        cache: Dict[str, torch.Tensor] = {
            "mtp:layer0_w": torch.ones(4),
            "mtp:layer1_w": torch.ones(4),
            "main:embed_w": torch.ones(4),
        }
        invalidator = SLocInvalidator(cache_ref=cache)

        p0 = torch.nn.Parameter(torch.randn(4))
        p0._sloc_key = "mtp:layer0_w"
        p1 = torch.nn.Parameter(torch.randn(4))
        p1._sloc_key = "mtp:layer1_w"

        evicted = invalidator.invalidate_group([p0, p1], grad_norm_group="mtp")
        assert evicted == 2, f"Expected 2 evictions, got {evicted}"
        assert "mtp:layer0_w" not in cache
        assert "mtp:layer1_w" not in cache
        assert "main:embed_w" in cache  # Main cache entry untouched.
        assert invalidator.total_evictions == 2

    _run_test("SLocInvalidator: evicts only matching keys", _test_sloc_eviction)

    # ------------------------------------------------------------------
    # Test 12: SLocInvalidator is a no-op when cache is None
    # ------------------------------------------------------------------
    def _test_sloc_noop():
        invalidator = SLocInvalidator(cache_ref=None)
        p = torch.nn.Parameter(torch.randn(4))
        p._sloc_key = "mtp:layer0_w"
        evicted = invalidator.invalidate_group([p], grad_norm_group="mtp")
        assert evicted == 0
        assert invalidator.total_evictions == 0

    _run_test("SLocInvalidator: no-op when cache_ref is None", _test_sloc_noop)

    # ------------------------------------------------------------------
    # Test 13: tag_mtp_parameters tags all params correctly
    # ------------------------------------------------------------------
    def _test_tag_mtp():
        module = torch.nn.Linear(8, 8)
        n = tag_mtp_parameters(module, sloc_key_prefix="mtp_head_0")

        assert n == len(list(module.parameters()))
        for name, param in module.named_parameters():
            assert getattr(param, DESLOCK_GRAD_NORM_GROUP_ATTR, None) == DESLOCK_MTP_GRAD_NORM_GROUP
            assert hasattr(param, "_sloc_key")
            assert param._sloc_key.startswith("mtp_head_0:")

    _run_test("tag_mtp_parameters: tags all params with group and sloc_key", _test_tag_mtp)

    # ------------------------------------------------------------------
    # Test 14: tag_mtp_parameters without sloc_key_prefix leaves no _sloc_key
    # ------------------------------------------------------------------
    def _test_tag_mtp_no_sloc():
        module = torch.nn.Linear(4, 4)
        tag_mtp_parameters(module)
        for _, param in module.named_parameters():
            assert getattr(param, DESLOCK_GRAD_NORM_GROUP_ATTR, None) == DESLOCK_MTP_GRAD_NORM_GROUP
            assert not hasattr(param, "_sloc_key")

    _run_test(
        "tag_mtp_parameters: no sloc_key when prefix is None", _test_tag_mtp_no_sloc
    )

    # ------------------------------------------------------------------
    # Test 15: HeteroMTPGradClipper.compute_grad_norms_by_group (single process)
    # ------------------------------------------------------------------
    def _test_compute_group_norms():
        mtp_params = [torch.nn.Parameter(torch.randn(4, 4)) for _ in range(2)]
        for p in mtp_params:
            setattr(p, DESLOCK_GRAD_NORM_GROUP_ATTR, DESLOCK_MTP_GRAD_NORM_GROUP)
            p.grad = torch.full((4, 4), 1.0)

        main_params = [torch.nn.Parameter(torch.randn(4, 4)) for _ in range(3)]
        for p in main_params:
            p.grad = torch.full((4, 4), 0.5)

        router = HeteroGradNormGroupRouter(main_params + mtp_params)
        clipper = HeteroMTPGradClipper()

        # Manually populate has_group_cache to skip all-reduce in single-process test.
        cache_key = (DESLOCK_MTP_GRAD_NORM_GROUP, None)
        clipper._has_group_cache[cache_key] = True

        norms = clipper.compute_grad_norms_by_group(router)
        assert DESLOCK_MTP_GRAD_NORM_GROUP in norms
        # 2 params × 4×4 = 32 elements each = 1.0; total sq = 32 + 32 = 64; norm = 8.0
        expected_norm = math.sqrt(2 * 16 * 1.0)
        assert abs(norms[DESLOCK_MTP_GRAD_NORM_GROUP] - expected_norm) < 1e-4, (
            f"Expected {expected_norm}, got {norms[DESLOCK_MTP_GRAD_NORM_GROUP]}"
        )

    _run_test(
        "HeteroMTPGradClipper.compute_grad_norms_by_group: correct MTP norm",
        _test_compute_group_norms,
    )

    # ------------------------------------------------------------------
    # Test 16: HeteroMTPGradClipper.clip_all_groups clips only MTP, not main below threshold
    # ------------------------------------------------------------------
    def _test_clip_all_groups_mtp_only():
        # Main param: tiny grad (below clip threshold) — must not change.
        main_p = torch.nn.Parameter(torch.zeros(2, 2))
        main_p.grad = torch.full((2, 2), 0.1)
        main_grad_before = main_p.grad.clone()

        # MTP param: large grad (well above clip threshold=1.0).
        mtp_p = torch.nn.Parameter(torch.zeros(4, 4))
        setattr(mtp_p, DESLOCK_GRAD_NORM_GROUP_ATTR, DESLOCK_MTP_GRAD_NORM_GROUP)
        mtp_p.grad = torch.full((4, 4), 5.0)
        mtp_norm_before = math.sqrt((mtp_p.grad ** 2).sum().item())

        router = HeteroGradNormGroupRouter([main_p, mtp_p])

        clipper = HeteroMTPGradClipper(clip_grad=1.0)
        # Bypass all-reduce for single-process test.
        cache_key = (DESLOCK_MTP_GRAD_NORM_GROUP, None)
        clipper._has_group_cache[cache_key] = True

        # Dummy config.
        class _Config:
            use_precision_aware_optimizer_no_fp8_or_ds_fp8 = False
            use_precision_aware_optimizer = False

        main_norm = _compute_grad_norm_fp32(router.grads_for_group(None))
        clipper.clip_all_groups(
            router=router,
            main_grad_norm=main_norm,
            optimizer_config=_Config(),
        )

        # Main grad should be unchanged (its norm < clip threshold).
        assert torch.allclose(
            main_p.grad, main_grad_before
        ), "Main grad changed when it should not have been clipped"

        # MTP grad should be reduced.
        mtp_norm_after = math.sqrt((mtp_p.grad ** 2).sum().item())
        assert mtp_norm_after < mtp_norm_before, "MTP grad was not clipped"
        assert abs(mtp_norm_after - 1.0) < 1e-4, (
            f"MTP norm after clip should be ≈1.0, got {mtp_norm_after}"
        )
        assert DESLOCK_MTP_GRAD_NORM_GROUP in clipper.grad_norms_by_group

    _run_test(
        "HeteroMTPGradClipper.clip_all_groups: clips MTP independently, leaves small main alone",
        _test_clip_all_groups_mtp_only,
    )

    # ------------------------------------------------------------------
    # Test 17: HeteroChainedOptimizer skip-threshold uses main norm only
    # ------------------------------------------------------------------
    def _test_chained_skip_uses_main_norm():
        """Verifies that a large MTP grad norm does not trigger the skip logic."""

        class _MockSubOpt:
            is_stub_optimizer = False

            def __init__(self):
                self.config = type("C", (), {
                    "clip_grad": 0.0,
                    "use_precision_aware_optimizer_no_fp8_or_ds_fp8": False,
                    "use_precision_aware_optimizer": False,
                })()
                self.main_p = torch.nn.Parameter(torch.zeros(2, 2))
                self.main_p.grad = torch.full((2, 2), 0.1)  # tiny norm < skip_threshold
                self.mtp_p = torch.nn.Parameter(torch.zeros(4, 4))
                setattr(self.mtp_p, DESLOCK_GRAD_NORM_GROUP_ATTR, DESLOCK_MTP_GRAD_NORM_GROUP)
                self.mtp_p.grad = torch.full((4, 4), 50.0)  # huge norm >> skip_threshold
                self.step_called = False

            def prepare_grads(self):
                return False

            def get_parameters(self):
                return [self.main_p, self.mtp_p]

            def get_grad_stats_parallel_group(self):
                return None

            def step_with_ready_grads(self):
                self.step_called = True
                return True

        sub_opt = _MockSubOpt()
        clipper = HeteroMTPGradClipper(clip_grad=0.0, grad_norm_skip_threshold=1.0)
        chained = HeteroChainedOptimizer(
            sub_optimizers=[sub_opt],
            clipper=clipper,
            grad_norm_skip_threshold=1.0,
        )

        # Inject cached group membership to avoid all-reduce.
        cache_key = (DESLOCK_MTP_GRAD_NORM_GROUP, None)
        clipper._has_group_cache[cache_key] = True

        update_successful, main_norm, _ = chained.step()

        # Main norm is small → update should proceed despite huge MTP norm.
        assert update_successful is True, "Update was incorrectly skipped"
        assert sub_opt.step_called is True, "Sub-optimizer step was not called"
        assert main_norm < 1.0, f"Main norm {main_norm} should be < skip threshold 1.0"

    _run_test(
        "HeteroChainedOptimizer: skip-threshold uses main norm, ignores MTP norm",
        _test_chained_skip_uses_main_norm,
    )

    # ------------------------------------------------------------------
    # Test 18: _fp8_safe_for_device returns False for CPU
    # ------------------------------------------------------------------
    def _test_fp8_safe_cpu():
        cpu_dev = torch.device("cpu")
        assert _fp8_safe_for_device(cpu_dev) is False

    _run_test("_fp8_safe_for_device: CPU returns False", _test_fp8_safe_cpu)

    # ------------------------------------------------------------------
    # Test 19: HeteroGradNormGroupRouter params_for_group returns empty for absent group
    # ------------------------------------------------------------------
    def _test_router_absent_group():
        main_params = [torch.nn.Parameter(torch.randn(2, 2)) for _ in range(2)]
        router = HeteroGradNormGroupRouter(main_params)
        assert router.params_for_group(DESLOCK_MTP_GRAD_NORM_GROUP) == []
        assert router.group_names == set()

    _run_test(
        "HeteroGradNormGroupRouter: params_for_group empty when no MTP params",
        _test_router_absent_group,
    )

    # ------------------------------------------------------------------
    # Test 20: HeteroProcessGroupRegistry default fallback
    # ------------------------------------------------------------------
    def _test_pg_registry_fallback():
        sentinel = object()  # Pretend process group.
        registry = HeteroProcessGroupRegistry(default_group=sentinel)
        # No specific mapping → should return default.
        result = registry.get(DeviceTier.SM86, DESLOCK_MTP_GRAD_NORM_GROUP)
        assert result is sentinel

    _run_test(
        "HeteroProcessGroupRegistry: falls back to default when no entry registered",
        _test_pg_registry_fallback,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print(f"Results: {_tests_run - _tests_failed}/{_tests_run} tests passed.")
    if _tests_failed > 0:
        sys.exit(1)
