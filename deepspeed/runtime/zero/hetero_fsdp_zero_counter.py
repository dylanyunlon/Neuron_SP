"""
DES-LOC Heterogeneous FSDP Zero-Gradient Counter
=================================================

Upstream Design Intent (Megatron commit 3e6e32b50d5cfe6e5ce12b829d2af9c203e89116):
    Megatron-LM's ``count_zeros_fp32`` function had a subtle ordering bug: the
    ``grad_attr`` and ``grad_not_none`` variables were computed *inside* the FSDP
    branch only after the non-FSDP path had already fallen through, meaning that
    for parameters managed by Megatron-FSDP the function always dereferenced
    ``param.grad`` instead of ``param.decoupled_grad``.  The fix moves the two
    variable assignments to the top of the per-parameter loop so that both the
    FSDP and non-FSDP code paths share a single, consistent view of "which
    gradient attribute should I be looking at?"  A clarifying comment was also
    added to document why Megatron-FSDP does *not* need a subsequent DP
    all-reduce: FSDP already performed the reduction during the backward pass.

DES-LOC Adaptation — HeteroFSDPZeroCounter:
    The Neuron_SP / DES-LOC environment differs from Megatron's assumed topology
    in three important ways that require the zero-counter logic to be re-thought
    rather than simply ported:

    1.  **Heterogeneous device fleet.**  We have two SM86 A6000 GPUs (48 GB each)
        and one SM90 H100 NVL (96 GB).  The H100 holds the master shards of FSDP
        parameters while the A6000s each hold replica shards.  The "local shard"
        of a DTensor therefore has *different* shapes and dtypes on each device,
        so the zero-count must be weighted by shard *size* rather than summed
        naively.

    2.  **PCIe-only interconnect, no NVLink.**  All-reduce across devices is
        expensive.  The upstream code simply raises an error when a data-parallel
        group is present alongside FSDP.  In DES-LOC we honour the same
        invariant (FSDP backward already handles DP reduction) but we emit a
        structured diagnostic rather than a hard crash, because during speculative
        execution a transient dp-group reference is valid.

    3.  **Decoupled-gradient (DES) execution.**  The Shared-LOcality Cache (LOC)
        stores a *stale* gradient copy from the previous micro-step in
        ``param.loc_cached_grad``.  When ``use_decoupled_grad=True`` we must
        prefer ``param.decoupled_grad`` over both ``param.grad`` and the LOC
        cache.  The priority chain is:
            decoupled_grad  >  grad  >  loc_cached_grad  (never counted)

    The class ``HeteroFSDPZeroCounter`` encapsulates all of this logic behind a
    clean interface that DeepSpeed's ZeRO optimizer can call at gradient-clip
    time without knowing about the underlying heterogeneity.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device-tier registry
# ---------------------------------------------------------------------------

# SM architecture → tier label.  Higher tier = more memory / compute.
_SM_TO_TIER: Dict[int, str] = {
    86: "a6000",   # NVIDIA A6000, 48 GB, SM86
    90: "h100",    # NVIDIA H100 NVL, 96 GB, SM90
}

_TIER_WEIGHT: Dict[str, float] = {
    "a6000": 1.0,
    "h100":  2.0,   # H100 shard is twice as large; weight zero-count accordingly
    "unknown": 1.0,
}


def _device_tier(device: torch.device) -> str:
    """Return the DES-LOC tier label for *device*.

    We query ``torch.cuda.get_device_capability`` which returns the SM version.
    Falls back to ``"unknown"`` for CPU or unrecognised architectures so that
    the counter degrades gracefully in non-GPU unit-test environments.
    """
    if device.type != "cuda":
        return "unknown"
    major, minor = torch.cuda.get_device_capability(device)
    sm = major * 10 + minor
    return _SM_TO_TIER.get(sm, "unknown")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ZeroCountResult:
    """Structured result returned by :class:`HeteroFSDPZeroCounter`.

    Attributes
    ----------
    total_zeros:
        Raw (unweighted) count of zero elements across all *counted* parameters.
    weighted_zeros:
        Zero count weighted by device tier.  Use this for gradient-norm
        diagnostics that must account for shard-size asymmetry.
    total_elements:
        Raw element count for all *counted* parameters.
    weighted_elements:
        Element count weighted by device tier.
    fsdp_param_count:
        Number of parameters handled by the Megatron/DES-LOC FSDP path.
    non_fsdp_param_count:
        Number of parameters handled by the standard DeepSpeed path.
    skipped_param_count:
        Parameters whose gradient was None or that were skipped for other
        reasons (TP duplicate, shared param, …).
    dp_group_collision:
        ``True`` if a data-parallel group was detected alongside FSDP params.
        This is not raised as an error in DES-LOC (see module docstring).
    tier_breakdown:
        Per-tier zero / element counts for debugging heterogeneous imbalance.
    """

    total_zeros: int = 0
    weighted_zeros: float = 0.0
    total_elements: int = 0
    weighted_elements: float = 0.0
    fsdp_param_count: int = 0
    non_fsdp_param_count: int = 0
    skipped_param_count: int = 0
    dp_group_collision: bool = False
    tier_breakdown: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def zero_fraction(self, weighted: bool = True) -> float:
        """Return the fraction of zero gradient elements.

        Parameters
        ----------
        weighted:
            If ``True`` (default) use the tier-weighted counts so that the H100
            shard's larger contribution is properly reflected.
        """
        if weighted:
            denom = self.weighted_elements
            return self.weighted_zeros / denom if denom > 0 else 0.0
        denom = self.total_elements
        return self.total_zeros / denom if denom > 0 else 0.0

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ZeroCountResult("
            f"zeros={self.total_zeros}/{self.total_elements}, "
            f"weighted_frac={self.zero_fraction():.4f}, "
            f"fsdp={self.fsdp_param_count}, "
            f"non_fsdp={self.non_fsdp_param_count}, "
            f"skipped={self.skipped_param_count})"
        )


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _param_is_not_shared(param: torch.nn.Parameter) -> bool:
    """Return ``True`` if *param* is not a shared (tied) parameter.

    DeepSpeed marks shared parameters with ``param.ds_status`` or the simpler
    ``param._is_param_native_zero_parallel``.  We also check the Megatron
    convention of ``param.shared`` for compatibility.
    """
    if getattr(param, "shared", False):
        return False
    if getattr(param, "_is_param_native_zero_parallel", False):
        return False
    return True


def _param_is_not_tp_duplicate(
    param: torch.nn.Parameter,
    tp_group: Optional[dist.ProcessGroup],
) -> bool:
    """Return ``True`` if this rank *owns* *param* in the tensor-parallel group.

    When tensor-parallelism is active, certain parameters (e.g. bias in a
    column-parallel linear) are replicated across all TP ranks.  Only the rank
    with ``tp_rank == 0`` should count their gradients to avoid double-counting.
    """
    if tp_group is None:
        return True
    tp_rank = dist.get_rank(tp_group)
    is_tp_duplicate = getattr(param, "tensor_model_parallel", False) is False and tp_rank != 0
    return not is_tp_duplicate


def _resolve_grad_tensor(
    param: torch.nn.Parameter,
    use_decoupled_grad: bool,
) -> Optional[torch.Tensor]:
    """Return the gradient tensor for *param* according to DES-LOC priority.

    Priority (highest → lowest):
        1. ``param.decoupled_grad``  — only when ``use_decoupled_grad=True``
        2. ``param.grad``            — standard PyTorch gradient
        3. *None*                    — ``param.loc_cached_grad`` is intentionally
                                       excluded; the LOC cache is stale and must
                                       never be counted as a live gradient.

    Returns
    -------
    torch.Tensor or None
    """
    if use_decoupled_grad:
        decoupled = getattr(param, "decoupled_grad", None)
        if decoupled is not None:
            return decoupled
        # Fall through: decoupled_grad not yet populated (e.g. first micro-step).
        # We intentionally do *not* fall back to loc_cached_grad here.

    return param.grad  # may be None


def _local_tensor(t: torch.Tensor) -> torch.Tensor:
    """Extract the local shard from a DTensor, or return *t* unchanged."""
    # DTensor (torch.distributed.tensor) exposes ``._local_tensor``.
    local = getattr(t, "_local_tensor", None)
    if local is not None:
        return local
    # DistributedTensor in older DeepSpeed/Megatron builds.
    local = getattr(t, "local_tensor", None)
    if local is not None and callable(local):
        return local()
    return t


# ---------------------------------------------------------------------------
# Core counter class
# ---------------------------------------------------------------------------

class HeteroFSDPZeroCounter:
    """Count zero-valued gradient elements across a heterogeneous FSDP setup.

    This class is the DES-LOC reinterpretation of Megatron's ``count_zeros_fp32``
    function.  Unlike the upstream function, it is stateful (accumulates across
    multiple calls), device-tier-aware, and separates the FSDP and non-FSDP
    counting paths *before* inspecting any gradient attribute — which is exactly
    the fix introduced in Megatron commit 3e6e32b.

    Parameters
    ----------
    use_decoupled_grad:
        When ``True``, prefer ``param.decoupled_grad`` over ``param.grad``.
        This mirrors the ``use_decoupled_grad`` flag in the upstream Megatron
        optimizer and is necessary for DES (Decoupled Execution) micro-steps.
    tp_group:
        Tensor-parallel process group.  Pass ``None`` if TP is not in use.
    model_parallel_group:
        Model-parallel process group used for non-FSDP all-reduce of the zero
        count.  Not used for FSDP parameters (FSDP handles its own DP reduction
        during backward).
    dtype:
        Cast gradients to this dtype before counting.  Default ``torch.float32``
        matches the Megatron upstream behaviour.
    log_tier_imbalance_threshold:
        If the zero-fraction difference between the H100 tier and the A6000
        tier exceeds this value, emit a warning.  Set to ``None`` to disable.
    """

    def __init__(
        self,
        use_decoupled_grad: bool = False,
        tp_group: Optional[dist.ProcessGroup] = None,
        model_parallel_group: Optional[dist.ProcessGroup] = None,
        dtype: torch.dtype = torch.float32,
        log_tier_imbalance_threshold: Optional[float] = 0.15,
    ) -> None:
        self.use_decoupled_grad = use_decoupled_grad
        self.tp_group = tp_group
        self.model_parallel_group = model_parallel_group
        self.dtype = dtype
        self.log_tier_imbalance_threshold = log_tier_imbalance_threshold

        # Running totals reset by each call to ``count``.
        self._result: Optional[ZeroCountResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def count(
        self,
        parameters: Iterable[torch.nn.Parameter],
    ) -> ZeroCountResult:
        """Count zero gradient elements across *parameters*.

        Parameters
        ----------
        parameters:
            Iterable of ``nn.Parameter`` objects.  Parameters without a
            gradient (or whose gradient resolves to ``None`` under the
            DES-LOC priority rules) are silently skipped.

        Returns
        -------
        ZeroCountResult
            Structured result containing raw and weighted counts.  Examine
            ``result.dp_group_collision`` instead of catching exceptions when
            running inside a speculative-execution context.
        """
        result = ZeroCountResult()
        use_fsdp = False
        dp_group: Optional[dist.ProcessGroup] = None

        for param in parameters:
            self._process_param(param, result, use_fsdp_ref=[use_fsdp], dp_group_ref=[dp_group])
            # Sync the mutable references back (Python lists used as ref cells).
            # The actual mutation happens inside _process_param.

        # Re-run with mutable ref cells so we can propagate FSDP flag.
        result, use_fsdp, dp_group = self._count_all(parameters)

        self._check_dp_group_collision(result, use_fsdp, dp_group)
        self._log_tier_imbalance(result)

        self._result = result
        return result

    @property
    def last_result(self) -> Optional[ZeroCountResult]:
        """Return the result from the most recent :meth:`count` call."""
        return self._result

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _count_all(
        self,
        parameters: Iterable[torch.nn.Parameter],
    ) -> Tuple[ZeroCountResult, bool, Optional[dist.ProcessGroup]]:
        """Full counting loop with mutable FSDP-flag and dp-group tracking."""
        result = ZeroCountResult()
        use_megatron_fsdp = False
        data_parallel_group: Optional[dist.ProcessGroup] = None

        for param in parameters:
            # ----------------------------------------------------------------
            # DES-LOC fix (mirrors Megatron 3e6e32b):
            #   Resolve the gradient attribute BEFORE branching on __fsdp_param__
            #   so that both paths use the same consistent gradient reference.
            # ----------------------------------------------------------------
            grad = _resolve_grad_tensor(param, self.use_decoupled_grad)
            grad_not_none = grad is not None

            if getattr(param, "__fsdp_param__", False) and grad_not_none:
                # ----------------------------------------------------------
                # FSDP path — local shard only.
                #
                # Upstream note: FSDP has already performed the DP all-reduce
                # during the backward pass, so we must NOT include this
                # parameter's zero-count in any subsequent all-reduce.
                #
                # DES-LOC extension: weight the count by the device tier so
                # that the H100's larger shard is proportionally reflected.
                # ----------------------------------------------------------
                use_megatron_fsdp = True
                local = _local_tensor(grad).to(self.dtype)
                tier = _device_tier(local.device)
                weight = _TIER_WEIGHT[tier]

                n_zeros = int((local.numel() - torch.count_nonzero(local)).item())
                n_elems = local.numel()

                result.total_zeros += n_zeros
                result.weighted_zeros += n_zeros * weight
                result.total_elements += n_elems
                result.weighted_elements += n_elems * weight
                result.fsdp_param_count += 1

                self._update_tier_breakdown(result, tier, n_zeros, n_elems)
                continue

            # ----------------------------------------------------------------
            # Non-FSDP path — standard DeepSpeed zero-counter logic.
            # ----------------------------------------------------------------
            is_not_shared = _param_is_not_shared(param)
            is_not_tp_dup = _param_is_not_tp_duplicate(param, self.tp_group)

            if grad_not_none and is_not_shared and is_not_tp_dup:
                grad_fp32 = grad.to(self.dtype)
                tier = _device_tier(grad_fp32.device)
                weight = _TIER_WEIGHT[tier]

                if self.model_parallel_group is not None:
                    # Collect across model-parallel ranks before counting.
                    # We all-gather to a flat tensor, count globally.
                    grad_fp32 = self._mp_gather_grad(grad_fp32)
                    # After gather, treat as a single flat tensor on current device.
                    data_parallel_group = self.model_parallel_group

                n_zeros = int((grad_fp32.numel() - torch.count_nonzero(grad_fp32)).item())
                n_elems = grad_fp32.numel()

                result.total_zeros += n_zeros
                result.weighted_zeros += n_zeros * weight
                result.total_elements += n_elems
                result.weighted_elements += n_elems * weight
                result.non_fsdp_param_count += 1

                self._update_tier_breakdown(result, tier, n_zeros, n_elems)
            else:
                result.skipped_param_count += 1

        return result, use_megatron_fsdp, data_parallel_group

    # ------------------------------------------------------------------

    @staticmethod
    def _update_tier_breakdown(
        result: ZeroCountResult,
        tier: str,
        n_zeros: int,
        n_elems: int,
    ) -> None:
        """Accumulate per-tier statistics into *result.tier_breakdown*."""
        if tier not in result.tier_breakdown:
            result.tier_breakdown[tier] = {"zeros": 0, "elements": 0}
        result.tier_breakdown[tier]["zeros"] += n_zeros
        result.tier_breakdown[tier]["elements"] += n_elems

    # ------------------------------------------------------------------

    def _mp_gather_grad(self, grad: torch.Tensor) -> torch.Tensor:
        """All-gather *grad* across the model-parallel group.

        Returns a single flat tensor containing the concatenated shards from
        all model-parallel ranks.  This is only called for non-FSDP parameters
        when ``self.model_parallel_group`` is set.
        """
        if not dist.is_available() or not dist.is_initialized():
            return grad

        world_size = dist.get_world_size(self.model_parallel_group)
        if world_size == 1:
            return grad

        flat = grad.contiguous().view(-1)
        gathered: List[torch.Tensor] = [torch.zeros_like(flat) for _ in range(world_size)]
        dist.all_gather(gathered, flat, group=self.model_parallel_group)
        return torch.cat(gathered)

    # ------------------------------------------------------------------

    def _check_dp_group_collision(
        self,
        result: ZeroCountResult,
        use_megatron_fsdp: bool,
        data_parallel_group: Optional[dist.ProcessGroup],
    ) -> None:
        """Handle the FSDP + DP-group collision case.

        Upstream Megatron raises ``ValueError`` unconditionally.  In DES-LOC
        we set a flag on the result and emit a structured log entry because
        speculative-execution passes may legitimately have a transient DP-group
        reference before FSDP teardown completes.  Callers that need strict
        semantics can inspect ``result.dp_group_collision`` and raise themselves.
        """
        if use_megatron_fsdp and data_parallel_group is not None:
            result.dp_group_collision = True
            logger.warning(
                "HeteroFSDPZeroCounter: detected FSDP parameters alongside an active "
                "data-parallel group (group rank=%d, world=%d).  "
                "FSDP already performed DP reduction during backward; "
                "the DP group reference is unexpected and may indicate a "
                "configuration error in the DES speculative-execution scheduler.  "
                "Proceeding without all-reduce for FSDP parameters.",
                dist.get_rank(data_parallel_group) if dist.is_initialized() else -1,
                dist.get_world_size(data_parallel_group) if dist.is_initialized() else -1,
            )

    # ------------------------------------------------------------------

    def _log_tier_imbalance(self, result: ZeroCountResult) -> None:
        """Warn if zero-fraction differs significantly between device tiers.

        A large imbalance suggests that the gradient shard assignment across
        the heterogeneous fleet is not load-balanced, which can cause the A6000
        GPUs to stall while the H100 finishes its (larger) backward pass.
        """
        if self.log_tier_imbalance_threshold is None:
            return
        breakdown = result.tier_breakdown
        if len(breakdown) < 2:
            return

        fracs: Dict[str, float] = {}
        for tier, counts in breakdown.items():
            elems = counts["elements"]
            fracs[tier] = counts["zeros"] / elems if elems > 0 else 0.0

        tiers = list(fracs.keys())
        for i in range(len(tiers)):
            for j in range(i + 1, len(tiers)):
                t_a, t_b = tiers[i], tiers[j]
                diff = abs(fracs[t_a] - fracs[t_b])
                if diff > self.log_tier_imbalance_threshold:
                    logger.warning(
                        "HeteroFSDPZeroCounter: tier zero-fraction imbalance detected "
                        "between '%s' (%.4f) and '%s' (%.4f), diff=%.4f > threshold=%.4f.  "
                        "Check DES-LOC shard assignment policy.",
                        t_a, fracs[t_a], t_b, fracs[t_b], diff,
                        self.log_tier_imbalance_threshold,
                    )


# ---------------------------------------------------------------------------
# Module-level convenience function (drop-in for Megatron count_zeros_fp32)
# ---------------------------------------------------------------------------

def count_zeros_fp32_hetero(
    parameters: Iterable[torch.nn.Parameter],
    use_decoupled_grad: bool = False,
    tp_group: Optional[dist.ProcessGroup] = None,
    model_parallel_group: Optional[dist.ProcessGroup] = None,
    log_tier_imbalance_threshold: Optional[float] = 0.15,
) -> ZeroCountResult:
    """Count zero-valued fp32 gradient elements with DES-LOC heterogeneity awareness.

    This is a drop-in replacement for Megatron's ``count_zeros_fp32`` that
    additionally handles:

    * Heterogeneous SM86 / SM90 device fleets (tier-weighted counts).
    * Decoupled-gradient (DES) execution (``use_decoupled_grad`` flag).
    * Non-fatal DP-group collision reporting (no hard crash inside speculative
      execution).

    Parameters
    ----------
    parameters:
        Iterable of parameters whose gradients should be counted.
    use_decoupled_grad:
        Prefer ``param.decoupled_grad`` over ``param.grad``.
    tp_group:
        Tensor-parallel process group (or ``None``).
    model_parallel_group:
        Model-parallel process group for non-FSDP all-reduce (or ``None``).
    log_tier_imbalance_threshold:
        Log a warning if per-tier zero fractions differ by more than this.

    Returns
    -------
    ZeroCountResult
    """
    counter = HeteroFSDPZeroCounter(
        use_decoupled_grad=use_decoupled_grad,
        tp_group=tp_group,
        model_parallel_group=model_parallel_group,
        log_tier_imbalance_threshold=log_tier_imbalance_threshold,
    )
    return counter.count(parameters)


# ---------------------------------------------------------------------------
# DeepSpeed optimizer hook integration
# ---------------------------------------------------------------------------

class HeteroZeroCounterOptimizerHook:
    """Mixin for DeepSpeed ZeRO optimizers that want hetero zero-counting.

    Usage
    -----
    Inherit from this alongside ``DeepSpeedZeroOptimizer`` (or any subclass)
    and call ``self.count_parameter_zeros()`` during gradient clipping.

    Example
    -------
    .. code-block:: python

        class MyHeteroOptimizer(HeteroZeroCounterOptimizerHook, DeepSpeedZeroOptimizer):
            def clip_gradients(self, max_norm):
                result = self.count_parameter_zeros()
                if result.zero_fraction() > 0.5:
                    logger.info("More than 50%% of gradient elements are zero; skipping clip.")
                    return
                super().clip_gradients(max_norm)
    """

    _hetero_zero_counter: Optional[HeteroFSDPZeroCounter] = None

    def _get_zero_counter(self) -> HeteroFSDPZeroCounter:
        if self._hetero_zero_counter is None:
            # Lazily construct so that subclass ``__init__`` can set attributes
            # like ``use_decoupled_grad`` before the counter is created.
            self._hetero_zero_counter = HeteroFSDPZeroCounter(
                use_decoupled_grad=getattr(self, "use_decoupled_grad", False),
                tp_group=getattr(self, "tensor_parallel_group", None),
                model_parallel_group=getattr(self, "model_parallel_group", None),
            )
        return self._hetero_zero_counter

    def count_parameter_zeros(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None,
    ) -> ZeroCountResult:
        """Count zero gradient elements for *parameters* (defaults to ``self.param_groups``)."""
        if parameters is None:
            parameters = [
                p
                for group in getattr(self, "param_groups", [])
                for p in group.get("params", [])
            ]
        return self._get_zero_counter().count(parameters)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import unittest

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    class _FakeParam(torch.nn.Parameter):
        """Minimal fake parameter for unit testing."""

        @staticmethod
        def _make(
            data: torch.Tensor,
            grad: Optional[torch.Tensor] = None,
            decoupled_grad: Optional[torch.Tensor] = None,
            fsdp: bool = False,
            shared: bool = False,
        ) -> "_FakeParam":
            p = _FakeParam(data, requires_grad=True)
            p.grad = grad
            if decoupled_grad is not None:
                p.decoupled_grad = decoupled_grad
            if fsdp:
                p.__fsdp_param__ = True
                if grad is not None:
                    p.grad = _FakeDTensor(grad)
                    if decoupled_grad is not None:
                        p.decoupled_grad = _FakeDTensor(decoupled_grad)
            p.shared = shared
            return p

    class _FakeDTensor:
        """Minimal DTensor stub with a ``_local_tensor`` attribute."""

        def __init__(self, local: torch.Tensor) -> None:
            self._local_tensor = local

        def __repr__(self) -> str:
            return f"FakeDTensor(shape={self._local_tensor.shape})"

    # ------------------------------------------------------------------

    class TestResolveGrad(unittest.TestCase):

        def test_prefers_decoupled_when_flag_set(self):
            g = torch.ones(4)
            dg = torch.zeros(4)
            p = torch.nn.Parameter(torch.empty(4))
            p.grad = g
            p.decoupled_grad = dg
            result = _resolve_grad_tensor(p, use_decoupled_grad=True)
            self.assertIs(result, dg)

        def test_falls_back_to_grad_when_no_decoupled(self):
            g = torch.ones(4)
            p = torch.nn.Parameter(torch.empty(4))
            p.grad = g
            result = _resolve_grad_tensor(p, use_decoupled_grad=True)
            self.assertIs(result, g)

        def test_uses_grad_when_flag_false(self):
            g = torch.ones(4)
            dg = torch.zeros(4)
            p = torch.nn.Parameter(torch.empty(4))
            p.grad = g
            p.decoupled_grad = dg
            result = _resolve_grad_tensor(p, use_decoupled_grad=False)
            self.assertIs(result, g)

        def test_returns_none_when_no_grad(self):
            p = torch.nn.Parameter(torch.empty(4))
            result = _resolve_grad_tensor(p, use_decoupled_grad=False)
            self.assertIsNone(result)

        def test_loc_cached_grad_never_returned(self):
            """LOC cache must NOT be counted as a live gradient."""
            p = torch.nn.Parameter(torch.empty(4))
            p.grad = None
            p.loc_cached_grad = torch.ones(4)
            result = _resolve_grad_tensor(p, use_decoupled_grad=True)
            self.assertIsNone(result)

    # ------------------------------------------------------------------

    class TestLocalTensor(unittest.TestCase):

        def test_plain_tensor_returned_unchanged(self):
            t = torch.randn(8)
            self.assertIs(_local_tensor(t), t)

        def test_dtensor_local_extracted(self):
            local = torch.randn(4)
            dt = _FakeDTensor(local)
            result = _local_tensor(dt)
            self.assertIs(result, local)

    # ------------------------------------------------------------------

    class TestZeroCountResultFraction(unittest.TestCase):

        def test_zero_fraction_no_elements(self):
            r = ZeroCountResult()
            self.assertEqual(r.zero_fraction(), 0.0)
            self.assertEqual(r.zero_fraction(weighted=False), 0.0)

        def test_unweighted_fraction(self):
            r = ZeroCountResult(
                total_zeros=3,
                total_elements=10,
                weighted_zeros=3.0,
                weighted_elements=10.0,
            )
            self.assertAlmostEqual(r.zero_fraction(weighted=False), 0.3)

        def test_weighted_fraction(self):
            r = ZeroCountResult(
                total_zeros=3,
                total_elements=10,
                weighted_zeros=6.0,    # H100 weight=2
                weighted_elements=20.0,
            )
            self.assertAlmostEqual(r.zero_fraction(weighted=True), 0.3)

    # ------------------------------------------------------------------

    class TestHeteroFSDPZeroCounter(unittest.TestCase):

        def _make_counter(self, use_decoupled: bool = False) -> HeteroFSDPZeroCounter:
            return HeteroFSDPZeroCounter(
                use_decoupled_grad=use_decoupled,
                tp_group=None,
                model_parallel_group=None,
                log_tier_imbalance_threshold=None,  # suppress imbalance logs in tests
            )

        # ------------------------------------------------------------------

        def test_all_grad_none_gives_empty_result(self):
            params = [torch.nn.Parameter(torch.empty(8)) for _ in range(3)]
            for p in params:
                p.grad = None
            counter = self._make_counter()
            result = counter.count(iter(params))
            self.assertEqual(result.total_zeros, 0)
            self.assertEqual(result.total_elements, 0)
            self.assertEqual(result.skipped_param_count, 3)

        def test_counts_zeros_in_plain_grad(self):
            p = torch.nn.Parameter(torch.empty(10))
            p.grad = torch.tensor([0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 4.0])
            counter = self._make_counter()
            result = counter.count([p])
            self.assertEqual(result.total_zeros, 6)
            self.assertEqual(result.total_elements, 10)
            self.assertEqual(result.non_fsdp_param_count, 1)
            self.assertEqual(result.fsdp_param_count, 0)

        def test_counts_zeros_in_decoupled_grad(self):
            p = torch.nn.Parameter(torch.empty(6))
            p.grad = torch.ones(6)          # all non-zero
            p.decoupled_grad = torch.zeros(6)  # all zero
            counter = self._make_counter(use_decoupled=True)
            result = counter.count([p])
            # Should count 6 zeros from decoupled_grad, not 0 from grad.
            self.assertEqual(result.total_zeros, 6)
            self.assertEqual(result.total_elements, 6)

        def test_fsdp_param_uses_local_tensor(self):
            local = torch.zeros(5)
            local[2] = 1.0  # one non-zero → 4 zeros
            param_data = torch.empty(5)
            p = torch.nn.Parameter(param_data)
            p.__fsdp_param__ = True
            fake_dt = _FakeDTensor(local)
            p.grad = fake_dt
            counter = self._make_counter()
            result = counter.count([p])
            self.assertEqual(result.total_zeros, 4)
            self.assertEqual(result.total_elements, 5)
            self.assertEqual(result.fsdp_param_count, 1)
            self.assertEqual(result.non_fsdp_param_count, 0)

        def test_fsdp_param_with_decoupled_grad(self):
            """Bug from Megatron 3e6e32b: decoupled_grad should be used for FSDP params."""
            regular_grad_local = torch.ones(4)       # 0 zeros
            decoupled_local = torch.zeros(4)         # 4 zeros

            p = torch.nn.Parameter(torch.empty(4))
            p.__fsdp_param__ = True
            p.grad = _FakeDTensor(regular_grad_local)
            p.decoupled_grad = _FakeDTensor(decoupled_local)

            counter = self._make_counter(use_decoupled=True)
            result = counter.count([p])
            # Must see 4 zeros (from decoupled_grad), not 0 (from grad).
            self.assertEqual(result.total_zeros, 4)

        def test_shared_param_is_skipped(self):
            p = torch.nn.Parameter(torch.empty(4))
            p.grad = torch.zeros(4)
            p.shared = True
            counter = self._make_counter()
            result = counter.count([p])
            self.assertEqual(result.total_zeros, 0)
            self.assertEqual(result.skipped_param_count, 1)

        def test_mixed_fsdp_and_plain_params(self):
            # Plain param: 3 zeros out of 5
            plain = torch.nn.Parameter(torch.empty(5))
            plain.grad = torch.tensor([0.0, 1.0, 0.0, 2.0, 0.0])

            # FSDP param: 2 zeros out of 4
            fsdp_local = torch.tensor([0.0, 1.0, 0.0, 2.0])
            fsdp_p = torch.nn.Parameter(torch.empty(4))
            fsdp_p.__fsdp_param__ = True
            fsdp_p.grad = _FakeDTensor(fsdp_local)

            counter = self._make_counter()
            result = counter.count([plain, fsdp_p])
            self.assertEqual(result.total_zeros, 5)
            self.assertEqual(result.total_elements, 9)
            self.assertEqual(result.fsdp_param_count, 1)
            self.assertEqual(result.non_fsdp_param_count, 1)

        def test_fsdp_grad_none_is_skipped(self):
            p = torch.nn.Parameter(torch.empty(4))
            p.__fsdp_param__ = True
            p.grad = None
            counter = self._make_counter()
            result = counter.count([p])
            self.assertEqual(result.fsdp_param_count, 0)
            self.assertEqual(result.skipped_param_count, 1)

        def test_tier_breakdown_populated(self):
            p = torch.nn.Parameter(torch.empty(6))
            # Force the tensor onto CPU so device_tier returns "unknown".
            p.grad = torch.zeros(6, device="cpu")
            counter = self._make_counter()
            result = counter.count([p])
            self.assertIn("unknown", result.tier_breakdown)
            self.assertEqual(result.tier_breakdown["unknown"]["zeros"], 6)

        def test_last_result_cached(self):
            p = torch.nn.Parameter(torch.empty(4))
            p.grad = torch.zeros(4)
            counter = self._make_counter()
            r1 = counter.count([p])
            self.assertIs(counter.last_result, r1)

        def test_dp_group_collision_flag_set(self):
            """Collision should set flag, not raise."""
            import unittest.mock as mock

            fsdp_local = torch.zeros(3)
            p = torch.nn.Parameter(torch.empty(3))
            p.__fsdp_param__ = True
            p.grad = _FakeDTensor(fsdp_local)

            # Simulate a model_parallel_group existing (causes dp_group to be set).
            # We inject a fake dp_group by patching the gather method to be a no-op
            # and directly calling _check_dp_group_collision.
            counter = self._make_counter()
            result = ZeroCountResult()
            result.fsdp_param_count = 1

            # Create a fake process group sentinel.
            fake_group = object()

            with mock.patch("torch.distributed.is_initialized", return_value=False):
                counter._check_dp_group_collision(
                    result, use_megatron_fsdp=True, data_parallel_group=fake_group
                )

            self.assertTrue(result.dp_group_collision)

        def test_multiple_params_accumulate(self):
            params = []
            for i in range(5):
                p = torch.nn.Parameter(torch.empty(4))
                # Alternate: half zeros, half ones.
                p.grad = torch.tensor([0.0, 0.0, 1.0, 1.0])
                params.append(p)

            counter = self._make_counter()
            result = counter.count(params)
            self.assertEqual(result.total_zeros, 10)
            self.assertEqual(result.total_elements, 20)
            self.assertEqual(result.non_fsdp_param_count, 5)

        def test_zero_fraction_all_zeros(self):
            p = torch.nn.Parameter(torch.empty(8))
            p.grad = torch.zeros(8)
            counter = self._make_counter()
            result = counter.count([p])
            self.assertAlmostEqual(result.zero_fraction(weighted=False), 1.0)

        def test_zero_fraction_no_zeros(self):
            p = torch.nn.Parameter(torch.empty(8))
            p.grad = torch.ones(8)
            counter = self._make_counter()
            result = counter.count([p])
            self.assertAlmostEqual(result.zero_fraction(weighted=False), 0.0)

    # ------------------------------------------------------------------

    class TestConvenienceFunction(unittest.TestCase):

        def test_count_zeros_fp32_hetero_basic(self):
            p = torch.nn.Parameter(torch.empty(6))
            p.grad = torch.tensor([0.0, 0.0, 1.0, 0.0, 2.0, 0.0])
            result = count_zeros_fp32_hetero([p])
            self.assertEqual(result.total_zeros, 4)
            self.assertIsInstance(result, ZeroCountResult)

    # ------------------------------------------------------------------

    class TestOptimizerHookMixin(unittest.TestCase):

        def test_mixin_lazy_counter_creation(self):
            class _FakeOpt(HeteroZeroCounterOptimizerHook):
                use_decoupled_grad = False
                tensor_parallel_group = None
                model_parallel_group = None
                param_groups = []

            opt = _FakeOpt()
            self.assertIsNone(opt._hetero_zero_counter)
            result = opt.count_parameter_zeros([])
            self.assertIsNotNone(opt._hetero_zero_counter)
            self.assertEqual(result.total_zeros, 0)

        def test_mixin_uses_param_groups_by_default(self):
            p = torch.nn.Parameter(torch.empty(4))
            p.grad = torch.zeros(4)

            class _FakeOpt(HeteroZeroCounterOptimizerHook):
                use_decoupled_grad = False
                tensor_parallel_group = None
                model_parallel_group = None
                param_groups = [{"params": [p]}]

            opt = _FakeOpt()
            result = opt.count_parameter_zeros()
            self.assertEqual(result.total_zeros, 4)

    # ------------------------------------------------------------------

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestResolveGrad,
        TestLocalTensor,
        TestZeroCountResultFraction,
        TestHeteroFSDPZeroCounter,
        TestConvenienceFunction,
        TestOptimizerHookMixin,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
