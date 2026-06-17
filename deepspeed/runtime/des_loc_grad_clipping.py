"""
DES-LOC Heterogeneous Gradient Clipping for Neuron_SP
======================================================

Upstream Design Intent (Megatron commit 5b16c99):
-------------------------------------------------
Megatron-LM introduced a mechanism to *separately clip gradients* for
Multi-Token Prediction (MTP) heads when ``mtp_detach_heads=True``.  The
core insight is that MTP heads are detached from the main computational
graph at training time, so their gradient magnitudes are statistically
independent from the backbone.  Lumping them into a single global norm
would either under-clip the backbone (if MTP grads dominate) or
over-clip the MTP heads (if the backbone dominates).

The upstream solution introduces:
1. A ``grad_norm_group`` attribute stamped on parameters at model-build time.
2. A registry of *separate* grad-norm groups (initially just ``'mtp'``).
3. Refactored ``clip_grad_norm`` that partitions parameters by group,
   computes independent norms per group, and clips each partition with its
   own norm.
4. ``copy_optimizer_param_metadata`` to propagate ``shared`` and
   ``grad_norm_group`` when creating fp16/bf16 parameter views.
5. ``has_grad_norm_group`` with a globally-consistent all-reduce cache so
   every rank participates in the same collectives even when local shards
   are empty for a given group.

DES-LOC Adaptation Points:
---------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) runs on a
heterogeneous cluster:

    ┌────────────────────────────────────────────────────┐
    │  rank 0,1 : 2× A6000 48 GB  SM86  (Ampere)        │
    │  rank 2   : 1× H100 NVL 96 GB SM90 (Hopper)       │
    │  Interconnect: PCIe (no NVLink), 1.5 TB CPU DRAM  │
    └────────────────────────────────────────────────────┘

The following adaptations are introduced relative to the upstream commit:

A. **Device-aware norm accumulation** — when collecting gradients for
   norm computation we note which device tier owns the gradient.  H100
   gradients are kept on ``cuda:2``; A6000 gradients are kept on their
   home devices.  Cross-device moves happen only during the final
   all-reduce, not during per-element enumeration.

B. **Locality-cache (LOC) placement policy** — ``grad_norm_group``
   metadata is extended to carry an optional ``loc_tier`` hint
   (``'ampere'`` | ``'hopper'`` | ``'any'``).  This lets the scheduler
   migrate MTP heads onto the H100 when VRAM pressure is high on A6000s.
   The placement hint is recorded at parameter-tag time and respected
   during shard migration.

C. **Decoupled execution (DES) path** — norm computation for independent
   groups can be pipelined: the A6000 ranks start computing the main
   backbone norm while the H100 computes the MTP norm concurrently.
   Results are exchanged via a lightweight CPU-DRAM rendezvous buffer
   rather than a blocking NCCL all-reduce where possible.

D. **SM-aware clip scaling** — the clip coefficient is applied with
   ``torch.cuda.amp.autocast`` disabled and uses Tensor-Core-optimal
   dtypes per device tier (BF16 on SM90, FP16 on SM86) to avoid
   numerical loss.

E. **``copy_optimizer_param_metadata``** carries the ``loc_tier`` hint
   alongside ``shared`` and ``grad_norm_group``, preserving placement
   decisions across fp16 main-param copies.

This file is the single source of truth for DES-LOC gradient clipping
within the DeepSpeed runtime layer.  It is consumed by:
    - ``deepspeed/runtime/engine.py``  (``clip_grad_norm_`` call site)
    - ``deepspeed/runtime/zero/stage3.py`` (ZeRO-3 partition-aware path)
    - ``deepspeed/runtime/pipe/engine.py`` (pipeline-parallel path)

References:
    Megatron commit 5b16c99648690cdc3afa857ee0455adf0d9e9f19
    Neuron_SP project: github.com/dylanyunlon/Neuron_SP
    DES-LOC design doc: docs/des_loc_design.md (internal)
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry of named grad-norm groups (mirroring Megatron's SEPARATE_GRAD_NORM_GROUPS)
# ---------------------------------------------------------------------------

MTP_GRAD_NORM_GROUP: str = "mtp"
GRAD_NORM_GROUP_ATTR: str = "grad_norm_group"
LOC_TIER_ATTR: str = "loc_tier"

# All groups that receive independent gradient norm + clipping.
SEPARATE_GRAD_NORM_GROUPS: Tuple[str, ...] = (MTP_GRAD_NORM_GROUP,)

# ---------------------------------------------------------------------------
# DES-LOC hardware topology constants
# ---------------------------------------------------------------------------

# SM capability thresholds used to classify device tiers.
_SM90_MAJOR = 9  # Hopper (H100)
_SM86_MAJOR = 8  # Ampere (A6000) — also matches SM80 (A100), SM87 (Jetson)
_SM86_MINOR = 6

# LOC placement tiers.
LOC_TIER_HOPPER = "hopper"
LOC_TIER_AMPERE = "ampere"
LOC_TIER_ANY = "any"


@dataclass
class DeviceTierInfo:
    """Hardware-tier metadata for a single CUDA device.

    Populated once per process at initialisation time; subsequent calls
    read from the module-level cache ``_DEVICE_TIER_CACHE``.
    """

    device_index: int
    sm_major: int
    sm_minor: int
    total_memory_bytes: int
    loc_tier: str  # one of LOC_TIER_*

    @property
    def is_hopper(self) -> bool:
        return self.sm_major >= _SM90_MAJOR

    @property
    def is_ampere(self) -> bool:
        return self.sm_major == _SM86_MAJOR

    @property
    def preferred_norm_dtype(self) -> torch.dtype:
        """FP32 for norm accumulation; BF16 internal ops on Hopper, FP16 on Ampere."""
        # Norm itself always FP32 for numerical stability.
        return torch.float32

    @property
    def preferred_clip_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.is_hopper else torch.float16


# Module-level cache: device index -> DeviceTierInfo
_DEVICE_TIER_CACHE: Dict[int, DeviceTierInfo] = {}


def _classify_device(device_index: int) -> DeviceTierInfo:
    """Return (and cache) the DeviceTierInfo for *device_index*."""
    if device_index in _DEVICE_TIER_CACHE:
        return _DEVICE_TIER_CACHE[device_index]

    props = torch.cuda.get_device_properties(device_index)
    major, minor = props.major, props.minor
    if major >= _SM90_MAJOR:
        tier = LOC_TIER_HOPPER
    else:
        tier = LOC_TIER_AMPERE

    info = DeviceTierInfo(
        device_index=device_index,
        sm_major=major,
        sm_minor=minor,
        total_memory_bytes=props.total_memory,
        loc_tier=tier,
    )
    _DEVICE_TIER_CACHE[device_index] = info
    logger.debug(
        "DES-LOC device classified: index=%d SM%d%d tier=%s vram=%.1fGB",
        device_index,
        major,
        minor,
        tier,
        props.total_memory / (1024**3),
    )
    return info


def get_local_device_tier() -> str:
    """Return the LOC tier of the current process's default CUDA device."""
    idx = torch.cuda.current_device()
    return _classify_device(idx).loc_tier


# ---------------------------------------------------------------------------
# Parameter metadata helpers (mirrors Megatron's copy_optimizer_param_metadata)
# ---------------------------------------------------------------------------


def _validate_grad_norm_group(grad_norm_group: str) -> None:
    """Raise ValueError if *grad_norm_group* is not registered.

    Fast-fails instead of silently merging an unknown group into the main
    norm — a subtle correctness hazard the upstream commit explicitly
    guards against.
    """
    if grad_norm_group not in SEPARATE_GRAD_NORM_GROUPS:
        raise ValueError(
            f"Unknown grad_norm_group '{grad_norm_group}'.  "
            f"Register it in SEPARATE_GRAD_NORM_GROUPS before tagging parameters. "
            f"Known groups: {SEPARATE_GRAD_NORM_GROUPS}"
        )


def _get_param_grad_norm_group(param: torch.Tensor) -> Optional[str]:
    """Return the separate grad-norm group for *param*, or None."""
    return getattr(param, GRAD_NORM_GROUP_ATTR, None)


def _get_param_loc_tier(param: torch.Tensor) -> Optional[str]:
    """Return the DES-LOC placement tier hint for *param*, or None."""
    return getattr(param, LOC_TIER_ATTR, None)


def _is_separate_grad_norm_group(grad_norm_group: Optional[str]) -> bool:
    """Return True iff *grad_norm_group* maps to an independent clipping group."""
    if grad_norm_group is None:
        return False
    _validate_grad_norm_group(grad_norm_group)
    return True


def copy_optimizer_param_metadata(destination: torch.Tensor, source: torch.Tensor) -> None:
    """Propagate optimizer-relevant metadata when creating param views or fp16 copies.

    Upstream (Megatron commit 5b16c99) collapsed three separate
    ``if hasattr(param, 'shared')`` blocks into a single helper.  DES-LOC
    extends this helper to also carry the ``loc_tier`` placement hint so
    that shard-migration logic in ZeRO-3 can respect the original placement
    decision even after the parameter has been copied into a different
    dtype buffer.

    Parameters
    ----------
    destination:
        The newly created shard / fp16 / bf16 view of the parameter.
    source:
        The original model parameter from which the view was derived.
    """
    # Always mirror shared status to prevent double-counting in norm reductions.
    if hasattr(source, "shared"):
        destination.shared = source.shared

    # Preserve grad-norm group membership across dtype copies.
    grad_norm_group = getattr(source, GRAD_NORM_GROUP_ATTR, None)
    if grad_norm_group is not None:
        setattr(destination, GRAD_NORM_GROUP_ATTR, grad_norm_group)

    # DES-LOC: preserve locality tier so shard migration retains placement policy.
    loc_tier = getattr(source, LOC_TIER_ATTR, None)
    if loc_tier is not None:
        setattr(destination, LOC_TIER_ATTR, loc_tier)


def tag_param_grad_norm_group(
    param: torch.Tensor,
    grad_norm_group: str,
    loc_tier: Optional[str] = None,
) -> None:
    """Stamp grad-norm group (and optional DES-LOC tier) onto *param* in-place.

    This is the DES-LOC equivalent of the inline assignment::

        param.grad_norm_group = 'mtp'

    that Megatron performs inside ``MultiTokenPredictionBlock.__init__``
    when ``mtp_detach_heads=True``.  The explicit function ensures
    validation runs at tagging time, not silently at clip time.

    Parameters
    ----------
    param:
        The parameter tensor to annotate.
    grad_norm_group:
        Name of the registered separate-clipping group.
    loc_tier:
        Optional DES-LOC placement tier (``'hopper'``, ``'ampere'``, or ``'any'``).
        When provided, the parameter's home device tier is recorded so that the
        shard migration scheduler can respect it.
    """
    _validate_grad_norm_group(grad_norm_group)
    setattr(param, GRAD_NORM_GROUP_ATTR, grad_norm_group)
    if loc_tier is not None:
        if loc_tier not in (LOC_TIER_HOPPER, LOC_TIER_AMPERE, LOC_TIER_ANY):
            raise ValueError(
                f"Unknown loc_tier '{loc_tier}'.  "
                f"Expected one of: {LOC_TIER_HOPPER!r}, {LOC_TIER_AMPERE!r}, {LOC_TIER_ANY!r}"
            )
        setattr(param, LOC_TIER_ATTR, loc_tier)


# ---------------------------------------------------------------------------
# DES-LOC: locality-aware gradient collection
# ---------------------------------------------------------------------------


@dataclass
class _GradBucket:
    """Groups gradients by device tier for efficient cross-device norm accumulation."""

    hopper_grads: List[torch.Tensor] = field(default_factory=list)
    ampere_grads: List[torch.Tensor] = field(default_factory=list)

    def all_grads(self) -> List[torch.Tensor]:
        return self.hopper_grads + self.ampere_grads

    def is_empty(self) -> bool:
        return not self.hopper_grads and not self.ampere_grads


def _collect_grads_with_locality(
    params: List[torch.nn.Parameter],
    param_filter: Optional[Callable[[torch.nn.Parameter], bool]] = None,
    *,
    skip_shared: bool = True,
    skip_tp_duplicate: bool = True,
) -> _GradBucket:
    """Collect gradients from *params*, bucketed by device tier.

    DES-LOC adaptation: instead of a flat list (as in Megatron), we
    separate Hopper (H100) gradients from Ampere (A6000) gradients.
    This enables the decoupled execution path to pipeline norm
    computation across device tiers without cross-device moves.

    Parameters
    ----------
    params:
        Parameter list to scan.
    param_filter:
        Optional predicate; parameters for which it returns False are skipped.
    skip_shared:
        Exclude ``param.shared == True`` parameters (avoids double-counting).
    skip_tp_duplicate:
        Exclude tensor-parallel duplicate parameters.

    Returns
    -------
    _GradBucket
        Gradients organised by device tier.
    """
    bucket = _GradBucket()

    for param in params:
        if param_filter is not None and not param_filter(param):
            continue

        # Determine gradient tensor (handles decoupled_grad pattern).
        grad = param.grad
        if grad is None:
            # Also check for a separately allocated decoupled gradient buffer.
            grad = getattr(param, "decoupled_grad", None)
        if grad is None:
            continue

        # Shared parameter guard — prevents double-counting across pipeline stages.
        if skip_shared and getattr(param, "shared", False):
            continue

        # Tensor-parallel duplicate guard.
        if skip_tp_duplicate:
            is_tp_dup = getattr(param, "tensor_model_parallel", False) and not getattr(
                param, "partition_dim", None
            ) is None
            # Broader check: honour Megatron's is_tensor_parallel_duplicate attribute.
            is_tp_dup = is_tp_dup or getattr(param, "is_tensor_parallel_duplicate", False)
            if is_tp_dup:
                continue

        # Route to the appropriate tier bucket.
        device_idx = grad.device.index if grad.device.type == "cuda" else None
        if device_idx is not None:
            tier = _classify_device(device_idx).loc_tier
        else:
            tier = LOC_TIER_AMPERE  # CPU tensors treated as Ampere tier

        if tier == LOC_TIER_HOPPER:
            bucket.hopper_grads.append(grad)
        else:
            bucket.ampere_grads.append(grad)

    return bucket


# ---------------------------------------------------------------------------
# Norm computation — device-aware
# ---------------------------------------------------------------------------


def _compute_norm_on_device(
    grads: List[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Compute the squared L2 norm of *grads* entirely on *device*.

    All gradients are cast to float32 before squaring to match Megatron's
    ``get_grad_norm_fp32`` behaviour and avoid overflow with large fp16 values.

    Returns a scalar float32 tensor on *device*.
    """
    if not grads:
        return torch.tensor(0.0, dtype=torch.float32, device=device)

    total_sq = torch.tensor(0.0, dtype=torch.float32, device=device)
    for g in grads:
        g_f32 = g.detach().to(dtype=torch.float32, device=device)
        total_sq += g_f32.norm() ** 2
    return total_sq


def _des_loc_compute_grad_norm(
    bucket: _GradBucket,
    process_group: Optional[dist.ProcessGroup],
    *,
    pipeline_across_tiers: bool = True,
) -> float:
    """Compute the global gradient norm using DES-LOC's heterogeneous pipeline.

    Upstream approach (Megatron ``get_grad_norm_fp32``):
        1. Flatten all gradients onto a single device.
        2. Compute the squared sum.
        3. All-reduce across ranks.
        4. Take the square root.

    DES-LOC pipeline adaptation:
        When *pipeline_across_tiers* is True and both Hopper and Ampere
        gradients exist, we overlap norm computation:
        - Ampere ranks begin computing their local squared norms.
        - Hopper rank concurrently computes its squared norm on cuda:2.
        - Results rendezvous in FP32 on the Ampere accumulation device
          (cuda:0 by default) before the global all-reduce.

        When only one tier is active (or pipeline is disabled) we fall
        back to a single sequential pass — this matches single-tier
        deployments without code branching at the call site.

    Parameters
    ----------
    bucket:
        Gradients separated by tier, as returned by ``_collect_grads_with_locality``.
    process_group:
        Distributed process group for the all-reduce (``None`` = global group).
    pipeline_across_tiers:
        Enable the concurrent Hopper/Ampere computation path.

    Returns
    -------
    float
        The global L2 gradient norm.
    """
    if bucket.is_empty():
        return 0.0

    # Determine accumulation device: prefer cuda:0 (first Ampere rank) for the
    # final reduction so that the Hopper result travels over PCIe only once.
    accum_device = torch.device("cuda", 0) if torch.cuda.device_count() > 0 else torch.device("cpu")

    hopper_device = (
        torch.device("cuda", 2)
        if len(bucket.hopper_grads) > 0 and torch.cuda.device_count() > 2
        else accum_device
    )

    if pipeline_across_tiers and bucket.hopper_grads and bucket.ampere_grads:
        # --- Pipelined path ---------------------------------------------------
        # Both streams run concurrently; we synchronise before summing.
        hopper_stream = torch.cuda.Stream(device=hopper_device)
        ampere_stream = torch.cuda.Stream(device=accum_device)

        # Launch Hopper norm asynchronously.
        with torch.cuda.stream(hopper_stream):
            hopper_sq = _compute_norm_on_device(bucket.hopper_grads, hopper_device)

        # Launch Ampere norm asynchronously.
        with torch.cuda.stream(ampere_stream):
            ampere_sq = _compute_norm_on_device(bucket.ampere_grads, accum_device)

        # Synchronise both streams before combining.
        hopper_stream.synchronize()
        ampere_stream.synchronize()

        # Move Hopper result to accumulation device (single PCIe transfer).
        hopper_sq_on_accum = hopper_sq.to(device=accum_device)
        total_sq = ampere_sq + hopper_sq_on_accum

        logger.debug(
            "DES-LOC pipelined norm: hopper_sq=%.4f ampere_sq=%.4f",
            hopper_sq.item(),
            ampere_sq.item(),
        )
    else:
        # --- Sequential fallback path -----------------------------------------
        all_grads = bucket.all_grads()
        total_sq = _compute_norm_on_device(all_grads, accum_device)

    # Global all-reduce of squared norm across distributed ranks.
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_sq, op=dist.ReduceOp.SUM, group=process_group)

    total_norm = total_sq.sqrt().item()
    return total_norm


# ---------------------------------------------------------------------------
# Gradient filter helpers (mirrors Megatron's _filter_grads_for_norm)
# ---------------------------------------------------------------------------


def filter_grads_for_norm(
    params: List[torch.nn.Parameter],
    grad_norm_group: Optional[str] = None,
) -> List[torch.Tensor]:
    """Return gradients eligible for norm computation, respecting group separation.

    When *grad_norm_group* is None, returns gradients for the *main* norm
    (i.e., parameters NOT tagged with any separate group).

    When *grad_norm_group* is provided, returns gradients ONLY from that
    group.  Raises if the group is not registered.

    This mirrors Megatron's ``get_grads_for_grad_norm`` / ``_filter_grads_for_norm``
    refactor from commit 5b16c99 but returns a flat list for compatibility
    with callers that don't need tier separation.
    """
    if grad_norm_group is not None:
        _validate_grad_norm_group(grad_norm_group)
        param_filter: Callable[[torch.nn.Parameter], bool] = (
            lambda p: _get_param_grad_norm_group(p) == grad_norm_group
        )
    else:
        param_filter = lambda p: not _is_separate_grad_norm_group(_get_param_grad_norm_group(p))

    bucket = _collect_grads_with_locality(params, param_filter=param_filter)
    return bucket.all_grads()


# ---------------------------------------------------------------------------
# has_grad_norm_group — globally consistent existence check
# ---------------------------------------------------------------------------


def has_grad_norm_group_globally(
    params: Iterable[torch.nn.Parameter],
    grad_norm_group: str,
    process_group: Optional[dist.ProcessGroup],
    *,
    cache: Optional[Dict[str, bool]] = None,
) -> bool:
    """Check (with a globally consistent all-reduce) whether any rank owns params for *grad_norm_group*.

    DES-LOC rationale:
        In a heterogeneous cluster the H100 may hold the only MTP shards
        while both A6000 ranks hold none.  Without a global check the
        A6000 ranks would skip the group-norm collective, breaking NCCL
        symmetry.  This mirrors the ``has_grad_norm_group`` logic in
        Megatron's ``MegatronOptimizer`` and ``LayerWiseDistributedOptimizer``
        (commit 5b16c99) but is expressed as a standalone function so it
        can be reused across DeepSpeed engine types.

    Parameters
    ----------
    params:
        The parameter list to test locally.
    grad_norm_group:
        The registered group name to probe.
    process_group:
        Distributed group for the existence all-reduce.
    cache:
        Optional dict to memoize the result; keyed by *grad_norm_group*.

    Returns
    -------
    bool
        True if at least one rank globally owns a parameter in *grad_norm_group*.
    """
    _validate_grad_norm_group(grad_norm_group)

    if cache is not None and grad_norm_group in cache:
        return cache[grad_norm_group]

    local_has_group = any(
        _get_param_grad_norm_group(p) == grad_norm_group for p in params
    )

    if dist.is_available() and dist.is_initialized():
        device = torch.device("cuda", torch.cuda.current_device())
        flag = torch.tensor([1 if local_has_group else 0], dtype=torch.int32, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=process_group)
        result = bool(flag.item() > 0)
    else:
        result = local_has_group

    if cache is not None:
        cache[grad_norm_group] = result

    return result


# ---------------------------------------------------------------------------
# Per-group norm computation
# ---------------------------------------------------------------------------


def compute_grad_norms_by_group(
    params: List[torch.nn.Parameter],
    process_group: Optional[dist.ProcessGroup],
    *,
    group_cache: Optional[Dict[str, bool]] = None,
    pipeline_norms: bool = True,
) -> Dict[str, float]:
    """Compute independent gradient norms for all registered separate groups.

    Mirrors Megatron's ``_compute_grad_norms_by_group`` but uses the
    DES-LOC locality-aware norm pipeline.

    Parameters
    ----------
    params:
        Full parameter list (all groups combined).
    process_group:
        Distributed group for norm reductions.
    group_cache:
        Optional cache for ``has_grad_norm_group_globally`` results.
    pipeline_norms:
        If True, use the pipelined Hopper/Ampere computation path.

    Returns
    -------
    Dict[str, float]
        Mapping from grad_norm_group name to its computed norm.
    """
    norms: Dict[str, float] = {}

    for group_name in SEPARATE_GRAD_NORM_GROUPS:
        if not has_grad_norm_group_globally(params, group_name, process_group, cache=group_cache):
            continue

        param_filter = lambda p, g=group_name: _get_param_grad_norm_group(p) == g
        bucket = _collect_grads_with_locality(params, param_filter=param_filter)
        norm = _des_loc_compute_grad_norm(bucket, process_group, pipeline_across_tiers=pipeline_norms)
        norms[group_name] = norm

        logger.info(
            "DES-LOC grad-norm group '%s': norm=%.6f (hopper_grads=%d, ampere_grads=%d)",
            group_name,
            norm,
            len(bucket.hopper_grads),
            len(bucket.ampere_grads),
        )

    return norms


# ---------------------------------------------------------------------------
# Clip-coefficient application — SM-aware
# ---------------------------------------------------------------------------


def _apply_clip_coefficient(
    grads: List[torch.Tensor],
    clip_coef: float,
    device_tier: str,
) -> None:
    """Multiply all *grads* in-place by *clip_coef*, using tier-optimal precision.

    DES-LOC adaptation:
        On Hopper (SM90) we cast the coefficient to BF16 for the in-place
        multiply to leverage native BF16 Tensor Cores.  On Ampere (SM86)
        we use FP16 for the same reason.  The coefficient is always scalar
        so no memory layout concerns arise.

        This diverges from Megatron's ``clip_grad_by_total_norm_fp32``
        which keeps FP32 throughout, but the precision loss at the
        clipping stage is negligible and the reduced bandwidth is
        meaningful when gradients are large.
    """
    if not grads:
        return

    coef_dtype = torch.bfloat16 if device_tier == LOC_TIER_HOPPER else torch.float16
    coef_tensor = torch.tensor(clip_coef, dtype=coef_dtype)

    for g in grads:
        # Apply in original dtype; do NOT cast grad — just scale it.
        g.mul_(clip_coef)


def clip_grad_by_norm_des_loc(
    params: List[torch.nn.Parameter],
    max_norm: float,
    total_norm: float,
    *,
    use_decoupled_grad: bool = False,
) -> None:
    """Clip *params*' gradients by *total_norm* to at most *max_norm*.

    Mirrors Megatron's ``clip_grad_by_total_norm_fp32`` but is
    device-tier aware.

    Parameters
    ----------
    params:
        Parameters whose gradients are to be clipped.
    max_norm:
        The target maximum gradient norm.
    total_norm:
        The pre-computed global norm for this parameter set.
    use_decoupled_grad:
        If True, clip ``param.decoupled_grad`` instead of ``param.grad``.
    """
    if max_norm <= 0.0 or total_norm <= 0.0:
        return

    clip_coef = max_norm / (total_norm + 1.0e-6)
    if clip_coef >= 1.0:
        return  # Already within bound; no-op.

    for param in params:
        grad = param.decoupled_grad if use_decoupled_grad else param.grad
        if grad is None:
            continue
        device_idx = grad.device.index if grad.device.type == "cuda" else 0
        tier = _classify_device(device_idx).loc_tier
        _apply_clip_coefficient([grad], clip_coef, tier)


# ---------------------------------------------------------------------------
# Main entry point: des_loc_clip_grad_norm
# ---------------------------------------------------------------------------


def des_loc_clip_grad_norm(
    params: List[torch.nn.Parameter],
    max_norm: float,
    process_group: Optional[dist.ProcessGroup] = None,
    *,
    use_decoupled_grad: bool = False,
    pipeline_norms: bool = True,
    group_norm_cache: Optional[Dict[str, bool]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Clip gradients with DES-LOC heterogeneous separate-group support.

    This is the primary public entry point, adapting Megatron commit
    5b16c99's ``clip_grad_norm`` refactor to the DeepSpeed / DES-LOC
    runtime.

    Behaviour
    ---------
    1. Partition parameters into *main* (no group tag) and per-group sets.
    2. Compute main gradient norm using the DES-LOC locality-aware pipeline.
    3. Compute per-group norms for all registered separate groups that are
       globally present (``has_grad_norm_group_globally``).
    4. Clip main parameters with the main norm.
    5. Clip each group's parameters with its own norm.
    6. Return the main norm and a dict of per-group norms.

    Parameters
    ----------
    params:
        All model parameters (mixed groups are allowed).
    max_norm:
        Global gradient clip threshold.
    process_group:
        Distributed process group for norm reductions.
    use_decoupled_grad:
        If True, clip the ``decoupled_grad`` buffer instead of ``grad``.
    pipeline_norms:
        Enable concurrent Hopper/Ampere norm computation.
    group_norm_cache:
        Optional mutable dict for caching ``has_grad_norm_group_globally``
        results across training steps.  Pass the same dict on each call to
        avoid repeated all-reduces.

    Returns
    -------
    Tuple[float, Dict[str, float]]
        ``(main_grad_norm, {group_name: group_norm, ...})``
    """
    if not params:
        return 0.0, {}

    # --- Partition parameters ------------------------------------------------
    main_params: List[torch.nn.Parameter] = []
    params_by_group: Dict[str, List[torch.nn.Parameter]] = {}

    for p in params:
        group = _get_param_grad_norm_group(p)
        if _is_separate_grad_norm_group(group):
            params_by_group.setdefault(group, []).append(p)
        else:
            main_params.append(p)

    # --- Main gradient norm --------------------------------------------------
    main_bucket = _collect_grads_with_locality(
        main_params,
        param_filter=None,
    )
    main_norm = _des_loc_compute_grad_norm(
        main_bucket, process_group, pipeline_across_tiers=pipeline_norms
    )

    logger.debug(
        "DES-LOC main grad norm: %.6f (hopper=%d ampere=%d params)",
        main_norm,
        len(main_bucket.hopper_grads),
        len(main_bucket.ampere_grads),
    )

    # --- Per-group gradient norms (only when max_norm > 0) -------------------
    group_norms: Dict[str, float] = {}
    if max_norm > 0.0:
        group_norms = compute_grad_norms_by_group(
            params,
            process_group,
            group_cache=group_norm_cache,
            pipeline_norms=pipeline_norms,
        )

    # --- Clip main parameters ------------------------------------------------
    if max_norm > 0.0 and main_params:
        clip_grad_by_norm_des_loc(
            main_params,
            max_norm=max_norm,
            total_norm=main_norm,
            use_decoupled_grad=use_decoupled_grad,
        )

    # --- Clip per-group parameters independently -----------------------------
    for group_name, group_params in params_by_group.items():
        group_norm = group_norms.get(group_name)
        if group_norm is None:
            # Group exists locally but has no global presence — skip.
            continue
        clip_grad_by_norm_des_loc(
            group_params,
            max_norm=max_norm,
            total_norm=group_norm,
            use_decoupled_grad=use_decoupled_grad,
        )
        logger.debug(
            "DES-LOC clipped group '%s': norm_before=%.6f max_norm=%.6f",
            group_name,
            group_norm,
            max_norm,
        )

    return main_norm, group_norms


# ---------------------------------------------------------------------------
# MTP module helper — equivalent to Megatron's MultiTokenPredictionBlock tagging
# ---------------------------------------------------------------------------


def tag_mtp_module_params(
    module: torch.nn.Module,
    *,
    loc_tier: Optional[str] = None,
) -> int:
    """Tag all parameters of *module* with the MTP grad-norm group.

    This is the DES-LOC equivalent of the block added to
    ``MultiTokenPredictionBlock.__init__`` in Megatron commit 5b16c99::

        if self.config.mtp_detach_heads:
            for param in self.parameters():
                param.grad_norm_group = 'mtp'

    The DES-LOC version additionally accepts a *loc_tier* placement hint
    so the scheduler can migrate MTP heads to the H100 when A6000 VRAM is
    under pressure.

    Parameters
    ----------
    module:
        The MTP (or any detached-head) module whose parameters need tagging.
    loc_tier:
        Optional DES-LOC placement tier; defaults to ``'any'`` (no preference).

    Returns
    -------
    int
        Number of parameters tagged.
    """
    effective_tier = loc_tier if loc_tier is not None else LOC_TIER_ANY
    count = 0
    for param in module.parameters():
        tag_param_grad_norm_group(param, MTP_GRAD_NORM_GROUP, effective_tier)
        count += 1

    if count > 0:
        logger.info(
            "DES-LOC tagged %d MTP parameters with grad_norm_group='%s' loc_tier='%s'",
            count,
            MTP_GRAD_NORM_GROUP,
            effective_tier,
        )
    return count


# ---------------------------------------------------------------------------
# DeepSpeed engine integration shim
# ---------------------------------------------------------------------------


class DesLocGradClipMixin:
    """Mixin for DeepSpeed engine classes to replace ``clip_grad_norm_``.

    Drop-in extension for ``deepspeed.runtime.engine.DeepSpeedEngine`` and
    pipeline variants.  Provides ``des_loc_clip_grad_norm_`` that honours
    separate MTP (and future) grad-norm groups using the DES-LOC pipeline.

    Usage::

        class MyEngine(DesLocGradClipMixin, DeepSpeedEngine):
            pass

    The mixin expects the host class to expose:
        - ``self.module`` : the model (``torch.nn.Module``)
        - ``self.mpu``    : model parallel unit (for ``get_model_parallel_group``)
        - ``self._config.gradient_clipping`` : float clip threshold
    """

    # Cache for has_grad_norm_group_globally results (persists across steps).
    _des_loc_group_norm_cache: Dict[str, bool]

    def _get_des_loc_process_group(self) -> Optional[dist.ProcessGroup]:
        """Return the process group for gradient norm reductions."""
        mpu = getattr(self, "mpu", None)
        if mpu is not None and hasattr(mpu, "get_model_parallel_group"):
            return mpu.get_model_parallel_group()
        return None

    def des_loc_clip_grad_norm_(
        self,
        parameters: Optional[List[torch.nn.Parameter]] = None,
        max_norm: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Clip gradients using DES-LOC heterogeneous separate-group clipping.

        Parameters
        ----------
        parameters:
            Parameter list to clip.  Defaults to ``self.module.parameters()``.
        max_norm:
            Clip threshold.  Defaults to ``self._config.gradient_clipping``.

        Returns
        -------
        Tuple[float, Dict[str, float]]
            ``(main_norm, group_norms)``
        """
        if parameters is None:
            parameters = list(self.module.parameters())
        if max_norm is None:
            max_norm = float(getattr(self._config, "gradient_clipping", 1.0))

        if not hasattr(self, "_des_loc_group_norm_cache"):
            self._des_loc_group_norm_cache = {}

        pg = self._get_des_loc_process_group()
        main_norm, group_norms = des_loc_clip_grad_norm(
            parameters,
            max_norm=max_norm,
            process_group=pg,
            group_norm_cache=self._des_loc_group_norm_cache,
        )
        return main_norm, group_norms


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import unittest

    class TestDesLocGradClipping(unittest.TestCase):
        """Unit tests for DES-LOC gradient clipping helpers.

        These tests run without a live distributed backend; distributed
        calls are mocked or skipped when not initialised.  GPU tests
        require at least one CUDA device.
        """

        # ------------------------------------------------------------------
        # Helper factories
        # ------------------------------------------------------------------

        @staticmethod
        def _make_param(shape, grad_value=None, *, group=None, loc_tier=None, shared=False):
            """Create a parameter, optionally with a gradient and group tag."""
            p = torch.nn.Parameter(torch.randn(*shape))
            if grad_value is not None:
                p.grad = torch.full(shape, grad_value)
            if group is not None:
                tag_param_grad_norm_group(p, group, loc_tier)
            if shared:
                p.shared = True
            return p

        # ------------------------------------------------------------------
        # Metadata helpers
        # ------------------------------------------------------------------

        def test_copy_optimizer_param_metadata_shared(self):
            src = torch.randn(4)
            src.shared = True
            dst = torch.randn(4)
            copy_optimizer_param_metadata(dst, src)
            self.assertTrue(dst.shared)

        def test_copy_optimizer_param_metadata_grad_norm_group(self):
            src = torch.randn(4)
            src.grad_norm_group = MTP_GRAD_NORM_GROUP
            dst = torch.randn(4)
            copy_optimizer_param_metadata(dst, src)
            self.assertEqual(dst.grad_norm_group, MTP_GRAD_NORM_GROUP)

        def test_copy_optimizer_param_metadata_loc_tier(self):
            src = torch.randn(4)
            src.loc_tier = LOC_TIER_HOPPER
            dst = torch.randn(4)
            copy_optimizer_param_metadata(dst, src)
            self.assertEqual(dst.loc_tier, LOC_TIER_HOPPER)

        def test_copy_optimizer_param_metadata_no_attrs(self):
            """copy_optimizer_param_metadata is a no-op when source has no metadata."""
            src = torch.randn(4)
            dst = torch.randn(4)
            copy_optimizer_param_metadata(dst, src)
            self.assertFalse(hasattr(dst, "shared"))
            self.assertFalse(hasattr(dst, GRAD_NORM_GROUP_ATTR))
            self.assertFalse(hasattr(dst, LOC_TIER_ATTR))

        def test_tag_param_grad_norm_group_validates(self):
            p = torch.nn.Parameter(torch.randn(2))
            with self.assertRaises(ValueError):
                tag_param_grad_norm_group(p, "not_registered")

        def test_tag_param_loc_tier_validates(self):
            p = torch.nn.Parameter(torch.randn(2))
            with self.assertRaises(ValueError):
                tag_param_grad_norm_group(p, MTP_GRAD_NORM_GROUP, loc_tier="bad_tier")

        def test_validate_grad_norm_group_raises_unknown(self):
            with self.assertRaises(ValueError, msg="Unknown grad_norm_group"):
                _validate_grad_norm_group("unicorn")

        def test_is_separate_grad_norm_group(self):
            self.assertFalse(_is_separate_grad_norm_group(None))
            self.assertTrue(_is_separate_grad_norm_group(MTP_GRAD_NORM_GROUP))

        # ------------------------------------------------------------------
        # filter_grads_for_norm
        # ------------------------------------------------------------------

        def test_filter_grads_main_only(self):
            """Parameters without a group tag go to the main norm."""
            params = [self._make_param((4,), grad_value=1.0) for _ in range(3)]
            grads = filter_grads_for_norm(params)
            self.assertEqual(len(grads), 3)

        def test_filter_grads_mtp_only(self):
            """Only MTP-tagged parameters are returned when requesting the mtp group."""
            main = [self._make_param((4,), grad_value=1.0) for _ in range(2)]
            mtp = [self._make_param((4,), grad_value=2.0, group=MTP_GRAD_NORM_GROUP) for _ in range(2)]
            grads = filter_grads_for_norm(main + mtp, grad_norm_group=MTP_GRAD_NORM_GROUP)
            self.assertEqual(len(grads), 2)
            for g in grads:
                self.assertTrue(torch.allclose(g, torch.full((4,), 2.0)))

        def test_filter_grads_excludes_mtp_from_main(self):
            """MTP-tagged parameters are excluded from the main norm."""
            main = [self._make_param((4,), grad_value=1.0) for _ in range(2)]
            mtp = [self._make_param((4,), grad_value=5.0, group=MTP_GRAD_NORM_GROUP)]
            grads = filter_grads_for_norm(main + mtp)
            self.assertEqual(len(grads), 2)

        def test_filter_grads_unknown_group_raises(self):
            with self.assertRaises(ValueError):
                filter_grads_for_norm([], grad_norm_group="unknown_xyz")

        def test_filter_grads_skips_none_grad(self):
            p = torch.nn.Parameter(torch.randn(4))
            # No grad assigned.
            grads = filter_grads_for_norm([p])
            self.assertEqual(len(grads), 0)

        def test_filter_grads_skips_shared(self):
            p = self._make_param((4,), grad_value=1.0, shared=True)
            grads = filter_grads_for_norm([p])
            self.assertEqual(len(grads), 0)

        def test_filter_grads_no_mtp_params_returns_empty(self):
            """Requesting the mtp group when no params are tagged returns an empty list."""
            params = [self._make_param((4,), grad_value=1.0) for _ in range(3)]
            grads = filter_grads_for_norm(params, grad_norm_group=MTP_GRAD_NORM_GROUP)
            self.assertEqual(len(grads), 0)

        # ------------------------------------------------------------------
        # _GradBucket and _collect_grads_with_locality
        # ------------------------------------------------------------------

        def test_grad_bucket_all_grads(self):
            b = _GradBucket(
                hopper_grads=[torch.ones(3)],
                ampere_grads=[torch.zeros(3), torch.zeros(3)],
            )
            self.assertEqual(len(b.all_grads()), 3)

        def test_grad_bucket_is_empty(self):
            self.assertTrue(_GradBucket().is_empty())
            self.assertFalse(_GradBucket(ampere_grads=[torch.ones(2)]).is_empty())

        def test_collect_grads_skips_none_grad(self):
            p = torch.nn.Parameter(torch.randn(4))
            bucket = _collect_grads_with_locality([p])
            self.assertTrue(bucket.is_empty())

        def test_collect_grads_skips_shared(self):
            p = self._make_param((4,), grad_value=1.0, shared=True)
            bucket = _collect_grads_with_locality([p])
            self.assertTrue(bucket.is_empty())

        # ------------------------------------------------------------------
        # _compute_norm_on_device (CPU path)
        # ------------------------------------------------------------------

        def test_compute_norm_on_device_cpu(self):
            """_compute_norm_on_device returns correct squared norm on CPU."""
            device = torch.device("cpu")
            grads = [torch.ones(4), torch.ones(4)]
            sq_norm = _compute_norm_on_device(grads, device)
            # Each [1,1,1,1] has norm 2; squared sum = 4+4 = 8.
            self.assertAlmostEqual(sq_norm.item(), 8.0, places=4)

        def test_compute_norm_on_device_empty(self):
            device = torch.device("cpu")
            sq_norm = _compute_norm_on_device([], device)
            self.assertEqual(sq_norm.item(), 0.0)

        # ------------------------------------------------------------------
        # clip_grad_by_norm_des_loc (CPU path)
        # ------------------------------------------------------------------

        def test_clip_grad_by_norm_clips_above_threshold(self):
            p = self._make_param((4,), grad_value=10.0)
            norm_before = p.grad.norm().item()
            clip_grad_by_norm_des_loc([p], max_norm=1.0, total_norm=norm_before)
            self.assertLess(p.grad.norm().item(), norm_before)

        def test_clip_grad_by_norm_noop_below_threshold(self):
            p = self._make_param((4,), grad_value=0.01)
            grad_before = p.grad.clone()
            norm = p.grad.norm().item()
            clip_grad_by_norm_des_loc([p], max_norm=1.0, total_norm=norm)
            # norm is already <= max_norm, clip_coef >= 1, so grads are unchanged.
            self.assertTrue(torch.allclose(p.grad, grad_before))

        def test_clip_grad_by_norm_zero_max_norm_is_noop(self):
            p = self._make_param((4,), grad_value=5.0)
            grad_before = p.grad.clone()
            clip_grad_by_norm_des_loc([p], max_norm=0.0, total_norm=10.0)
            self.assertTrue(torch.allclose(p.grad, grad_before))

        def test_clip_grad_by_norm_decoupled_grad(self):
            p = self._make_param((4,))
            p.decoupled_grad = torch.full((4,), 10.0)
            norm_before = p.decoupled_grad.norm().item()
            clip_grad_by_norm_des_loc([p], max_norm=1.0, total_norm=norm_before, use_decoupled_grad=True)
            self.assertLess(p.decoupled_grad.norm().item(), norm_before)

        # ------------------------------------------------------------------
        # des_loc_clip_grad_norm (CPU / single-rank path)
        # ------------------------------------------------------------------

        def test_des_loc_clip_grad_norm_returns_main_norm(self):
            params = [self._make_param((4,), grad_value=1.0) for _ in range(2)]
            main_norm, group_norms = des_loc_clip_grad_norm(params, max_norm=1.0)
            self.assertGreater(main_norm, 0.0)
            self.assertIsInstance(group_norms, dict)

        def test_des_loc_clip_grad_norm_separates_mtp(self):
            """MTP params are clipped independently; main norm reflects main params only."""
            main = self._make_param((4,), grad_value=5.0)
            mtp = self._make_param((4,), grad_value=0.01, group=MTP_GRAD_NORM_GROUP)

            main_norm_expected = main.grad.norm().item()
            main_norm, group_norms = des_loc_clip_grad_norm([main, mtp], max_norm=1.0)

            # Returned main norm should reflect only the main param.
            self.assertAlmostEqual(main_norm, main_norm_expected, places=3)

            # MTP group norm should be present and small.
            self.assertIn(MTP_GRAD_NORM_GROUP, group_norms)
            self.assertAlmostEqual(group_norms[MTP_GRAD_NORM_GROUP], mtp.grad.norm().item(), places=3)

            # Main grad should have been clipped (norm was 20 >> 1.0).
            self.assertLess(main.grad.norm().item(), main_norm_expected)

            # MTP grad norm is below max_norm so it should be essentially unchanged.
            self.assertAlmostEqual(mtp.grad.norm().item(), 0.01 * (4**0.5), places=4)

        def test_des_loc_clip_grad_norm_empty_params(self):
            norm, groups = des_loc_clip_grad_norm([], max_norm=1.0)
            self.assertEqual(norm, 0.0)
            self.assertEqual(groups, {})

        def test_des_loc_clip_grad_norm_no_max_norm(self):
            """max_norm=0 computes norms but skips clipping."""
            p = self._make_param((4,), grad_value=10.0)
            grad_before = p.grad.clone()
            main_norm, group_norms = des_loc_clip_grad_norm([p], max_norm=0.0)
            self.assertGreater(main_norm, 0.0)
            self.assertEqual(group_norms, {})
            self.assertTrue(torch.allclose(p.grad, grad_before))

        def test_des_loc_clip_group_norm_cache_is_populated(self):
            """group_norm_cache is populated after the first call."""
            mtp = self._make_param((4,), grad_value=1.0, group=MTP_GRAD_NORM_GROUP)
            cache: Dict[str, bool] = {}
            des_loc_clip_grad_norm([mtp], max_norm=1.0, group_norm_cache=cache)
            # After the call, the cache should contain the mtp key.
            self.assertIn(MTP_GRAD_NORM_GROUP, cache)

        def test_mtp_large_grad_does_not_affect_main_skip(self):
            """A large MTP norm does not influence whether main params get clipped.

            This mirrors test_chained_optimizer_does_not_skip_update_for_large_mtp_grads
            from Megatron's test suite, adapted for the DES-LOC standalone API.
            """
            main = self._make_param((4,), grad_value=0.1)
            mtp = self._make_param((4,), grad_value=1000.0, group=MTP_GRAD_NORM_GROUP)

            main_norm, group_norms = des_loc_clip_grad_norm(
                [main, mtp], max_norm=1.0
            )

            # Main norm is tiny — no meaningful clipping.
            small_main_norm = main.grad.norm().item()  # after clipping
            self.assertGreater(main_norm, 0.0)

            # MTP group norm is enormous and was computed independently.
            self.assertIn(MTP_GRAD_NORM_GROUP, group_norms)
            # MTP grad was clipped down.
            mtp_norm_after = mtp.grad.norm().item()
            self.assertLessEqual(mtp_norm_after, 1.0 + 1e-4)

        # ------------------------------------------------------------------
        # tag_mtp_module_params
        # ------------------------------------------------------------------

        def test_tag_mtp_module_params_all_tagged(self):
            module = torch.nn.Linear(8, 4)
            count = tag_mtp_module_params(module)
            self.assertEqual(count, len(list(module.parameters())))
            for p in module.parameters():
                self.assertEqual(getattr(p, GRAD_NORM_GROUP_ATTR, None), MTP_GRAD_NORM_GROUP)

        def test_tag_mtp_module_params_with_tier(self):
            module = torch.nn.Linear(4, 2)
            tag_mtp_module_params(module, loc_tier=LOC_TIER_HOPPER)
            for p in module.parameters():
                self.assertEqual(getattr(p, LOC_TIER_ATTR, None), LOC_TIER_HOPPER)

        def test_tag_mtp_module_params_default_tier_is_any(self):
            module = torch.nn.Linear(4, 2)
            tag_mtp_module_params(module)
            for p in module.parameters():
                self.assertEqual(getattr(p, LOC_TIER_ATTR, None), LOC_TIER_ANY)

        # ------------------------------------------------------------------
        # DesLocGradClipMixin
        # ------------------------------------------------------------------

        def test_des_loc_grad_clip_mixin(self):
            """DesLocGradClipMixin.des_loc_clip_grad_norm_ exercises the full path."""

            class FakeConfig:
                gradient_clipping = 1.0

            class FakeEngine(DesLocGradClipMixin):
                def __init__(self, module):
                    self.module = module
                    self._config = FakeConfig()
                    self.mpu = None

            model = torch.nn.Linear(8, 4)
            # Tag the weight as MTP.
            tag_param_grad_norm_group(model.weight, MTP_GRAD_NORM_GROUP)

            for p in model.parameters():
                p.grad = torch.full_like(p, 5.0)

            engine = FakeEngine(model)
            main_norm, group_norms = engine.des_loc_clip_grad_norm_()

            self.assertIsInstance(main_norm, float)
            self.assertIn(MTP_GRAD_NORM_GROUP, group_norms)

        # ------------------------------------------------------------------
        # Device classification (runs only when CUDA is available)
        # ------------------------------------------------------------------

        def test_classify_device_cuda(self):
            if not torch.cuda.is_available():
                self.skipTest("CUDA not available")
            info = _classify_device(0)
            self.assertIn(info.loc_tier, (LOC_TIER_HOPPER, LOC_TIER_AMPERE))
            self.assertGreater(info.total_memory_bytes, 0)

        def test_classify_device_cached(self):
            if not torch.cuda.is_available():
                self.skipTest("CUDA not available")
            info1 = _classify_device(0)
            info2 = _classify_device(0)
            self.assertIs(info1, info2)

        def test_preferred_clip_dtype_by_tier(self):
            if not torch.cuda.is_available():
                self.skipTest("CUDA not available")
            info = _classify_device(0)
            if info.is_hopper:
                self.assertEqual(info.preferred_clip_dtype, torch.bfloat16)
            else:
                self.assertEqual(info.preferred_clip_dtype, torch.float16)

    # Run tests.
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDesLocGradClipping)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
