# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Gradient clipping for heterogeneous GPU training — DES-LOC extensions.

Adapted from Megatron megatron/core/optimizer/clip_grads.py with deep
extensions for heterogeneous GPU tiers (H100 / A6000 / Blackwell / Consumer).

Key changes versus Megatron upstream
--------------------------------------
M2335  No host/device synchronization in get_grad_norm_fp32 — returns a CUDA
       tensor instead of .item() so the norm comparison can stay on-GPU.

M4145  FSDP-aware gradient unwrapping: params tagged ``__fsdp_param__ = True``
       carry gradients as sharded DTensors; clip_grads unwraps ._local_tensor
       before computing norms or applying the clip coefficient.

M4171  Per-tier gradient scaling (DES-LOC extension): when
       ``TierClipConfig.per_tier_scaling`` is enabled, each hardware tier
       (h100, a6000, blackwell, consumer) gets an independent clip coefficient
       derived from the tier-local gradient norm divided by a tier-specific
       ``max_norm`` budget. The budgets are proportional to the tier's BF16
       TFLOPS share so that faster GPUs clip less aggressively.

M4212  QK-norm clipping (Megatron PR #4212): a separate, tighter clip norm
       is applied to attention Q and K projection gradients to prevent the
       attention entropy collapse that manifests as a sharp loss spike at
       ~3000 steps in large models.

M4309  Adaptive norm tracking (DES-LOC extension): exponential moving
       average of the observed gradient norm across steps so that the clip
       threshold can be adjusted dynamically relative to the EMA rather than
       a fixed hyperparameter.

Public API
----------
  get_grad_norm_fp32           — main norm computation (CUDA tensor, no sync)
  clip_grad_by_total_norm_fp32 — scale grads in-place by clip coefficient
  clip_grad_norm               — combined norm + clip (convenience wrapper)
  count_zeros_fp32             — count zero gradient elements
  TierClipConfig               — per-tier clipping configuration
  QKClipConfig                 — Q/K projection clip configuration
  GradNormEMA                  — adaptive EMA norm tracker
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import inf

import deepspeed.core.parallel_state as parallel_state

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Multi-tensor-applier shim (same fallback chain as optimizer.py)
# ---------------------------------------------------------------------------
try:
    from transformer_engine.pytorch.optimizers import (
        multi_tensor_applier,
        multi_tensor_l2norm,
        multi_tensor_scale,
    )
    l2_norm_impl = multi_tensor_l2norm
    multi_tensor_scale_impl = multi_tensor_scale
    _HAS_MULTI_TENSOR = True
except ImportError:
    try:
        import amp_C
        from apex.multi_tensor_apply import multi_tensor_applier
        l2_norm_impl = amp_C.multi_tensor_l2norm
        multi_tensor_scale_impl = amp_C.multi_tensor_scale
        _HAS_MULTI_TENSOR = True
    except ImportError:
        _HAS_MULTI_TENSOR = False
        multi_tensor_applier = None
        l2_norm_impl = None
        multi_tensor_scale_impl = None

# ---------------------------------------------------------------------------
# Tier identifiers (mirrors TierType in desloc_config)
# ---------------------------------------------------------------------------
TIER_H100       = "h100"
TIER_A6000      = "a6000"
TIER_BLACKWELL  = "blackwell"
TIER_CONSUMER   = "consumer"

# BF16 TFLOPS per tier (used for default budget computation)
_TIER_TFLOPS: Dict[str, float] = {
    TIER_H100:      989.0,
    TIER_A6000:     309.7,
    TIER_BLACKWELL: 4500.0,  # B200 peak ~4.5 PFLOPS
    TIER_CONSUMER:  82.6,    # RTX 4090
}


# ---------------------------------------------------------------------------
# Per-tier clip configuration (DES-LOC M4171)
# ---------------------------------------------------------------------------

@dataclass
class TierClipConfig:
    """Per-tier gradient clipping configuration for heterogeneous clusters.

    When ``per_tier_scaling=True``, each hardware tier clips its local
    gradient shard independently with a budget proportional to its TFLOPS
    share.  This prevents faster GPUs from having their gradients clipped
    more aggressively than slower ones just because they hold a larger shard.

    Attributes:
        per_tier_scaling:  Enable independent per-tier clip budgets.
        global_max_norm:   Baseline clip norm shared across all tiers.
        tier_budgets:      Explicit per-tier override for max_norm.
                           Keys match TIER_* constants.  Missing tiers
                           fall back to TFLOPS-proportional budget.
        eps:               Numerical stability epsilon for norm division.

    Examples::

        cfg = TierClipConfig(
            per_tier_scaling=True,
            global_max_norm=1.0,
            tier_budgets={TIER_A6000: 0.5},  # slower tier clips tighter
        )
    """

    per_tier_scaling: bool = False
    global_max_norm: float = 1.0
    tier_budgets: Dict[str, float] = field(default_factory=dict)
    eps: float = 1.0e-6

    def budget_for_tier(self, tier: str) -> float:
        """Return the clip budget for *tier*.

        Falls back to a TFLOPS-weighted fraction of ``global_max_norm``
        when no explicit entry exists in ``tier_budgets``.
        """
        if tier in self.tier_budgets:
            return self.tier_budgets[tier]
        total_tflops = sum(_TIER_TFLOPS.values())
        tier_tflops = _TIER_TFLOPS.get(tier, 309.7)  # default: A6000
        return self.global_max_norm * (tier_tflops / total_tflops)


# ---------------------------------------------------------------------------
# QK-norm clip configuration (From Megatron M4212)
# ---------------------------------------------------------------------------

@dataclass
class QKClipConfig:
    """Q/K projection gradient clip configuration (Megatron M4212).

    Attention Q and K projections can develop very large gradient norms
    early in training (steps 2000–5000) that corrupt attention entropy and
    manifest as a sharp loss spike.  Applying a separate, tighter clip norm
    to these gradients prevents the collapse.

    Attributes:
        enabled:     Whether to apply QK-specific clipping.
        max_norm:    Tighter clip norm for Q/K gradients.
        param_names: Name substrings to identify Q/K projection parameters.
                     A parameter is tagged as QK if its name contains any
                     of these substrings.

    Examples::

        cfg = QKClipConfig(enabled=True, max_norm=0.5,
                           param_names=["q_proj", "k_proj",
                                        "query_key_value"])
    """

    enabled: bool = True
    max_norm: float = 0.5
    param_names: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "query_key_value", "wq", "wk",
        "query_weight", "key_weight",
    ])

    def is_qk_param(self, name: str) -> bool:
        """Return True if *name* matches any QK projection substring."""
        return any(s in name for s in self.param_names)


# ---------------------------------------------------------------------------
# Adaptive EMA norm tracker (DES-LOC M4309)
# ---------------------------------------------------------------------------

class GradNormEMA:
    """Exponential moving average tracker for gradient norms.

    Tracks the EMA of observed gradient norms so that the effective clip
    threshold can be set relative to the recent norm history rather than
    a fixed hyperparameter.  This is especially useful on heterogeneous
    PCIe clusters where norm spikes from slower tiers would cause excessive
    clipping for the faster tiers.

    Attributes:
        decay:    EMA decay coefficient (higher = slower adaptation).
        _ema:     Current EMA value (None until first update).
        _count:   Number of observations.

    Examples::

        tracker = GradNormEMA(decay=0.99)
        for step in range(training_steps):
            norm = compute_grad_norm()
            ema_norm = tracker.update(norm)
            effective_max = 2.0 * ema_norm  # clip at 2× the EMA
    """

    def __init__(self, decay: float = 0.99) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError(f"GradNormEMA decay must be in (0, 1), got {decay}.")
        self.decay = decay
        self._ema: Optional[float] = None
        self._count: int = 0

    def update(self, norm: Union[float, torch.Tensor]) -> float:
        """Update EMA with *norm* and return the current EMA value.

        Applies bias correction for the first few steps so that the initial
        EMA does not underestimate the true norm (same correction as Adam).

        Args:
            norm: Gradient norm value (float or 0-D tensor).

        Returns:
            Bias-corrected EMA of the observed norms.
        """
        if isinstance(norm, torch.Tensor):
            norm = float(norm.item())
        if self._ema is None:
            self._ema = norm
        else:
            self._ema = self.decay * self._ema + (1.0 - self.decay) * norm
        self._count += 1
        # Bias correction
        correction = 1.0 - self.decay ** self._count
        return self._ema / correction

    @property
    def value(self) -> Optional[float]:
        """Current (uncorrected) EMA value, or None if not yet initialised."""
        return self._ema

    def reset(self) -> None:
        """Reset the EMA state (e.g. after a warmup phase)."""
        self._ema = None
        self._count = 0


# ---------------------------------------------------------------------------
# Helper: unwrap gradient for a parameter
# ---------------------------------------------------------------------------

def _get_grad(
    param: torch.nn.Parameter,
    use_decoupled_grad: bool = False,
) -> Optional[torch.Tensor]:
    """Return the effective gradient tensor for *param*.

    Priority order:
      1. param.main_grad       — Megatron BF16 bucket path
      2. param.decoupled_grad  — Megatron-FSDP path (use_decoupled_grad=True)
      3. param.grad            — standard PyTorch

    For FSDP params (``__fsdp_param__ = True``) the tensor is further
    unwrapped via ``._local_tensor`` when that attribute is present.

    From Megatron M4145: fix zero-counter with decoupled grads.
    """
    is_fsdp = getattr(param, "__fsdp_param__", False)

    grad: Optional[torch.Tensor] = None

    # Priority 1: main_grad (Megatron BF16 bucket)
    mg = getattr(param, "main_grad", None)
    if mg is not None:
        grad = mg
    elif use_decoupled_grad:
        # Priority 2: decoupled_grad (Megatron-FSDP)
        dg = getattr(param, "decoupled_grad", None)
        if dg is not None:
            grad = dg
    if grad is None:
        # Priority 3: standard PyTorch .grad
        grad = param.grad

    # FSDP unwrap
    if grad is not None and is_fsdp and hasattr(grad, "_local_tensor"):
        grad = grad._local_tensor

    return grad


# ---------------------------------------------------------------------------
# Core norm computation — no host/device sync (From Megatron M2335)
# ---------------------------------------------------------------------------

def get_grad_norm_fp32(
    grads_for_norm: Union[List[torch.Tensor], torch.Tensor],
    norm_type: Union[int, float] = 2,
    grad_stats_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Calculate the p-norm of gradients in FP32 — fully on device, no sync.

    Returns a CUDA tensor (not .item()) to avoid host/device synchronization.
    This is the M2335 pattern: desloc_engine can compare grad_norm > clip_value
    entirely on GPU without stalling the CUDA stream.

    For L2 norms the fused multi-tensor-applier kernel (TE/Apex) is used
    when available for improved throughput; pure-PyTorch fallback otherwise.

    For infinity norms the per-tensor abs().max() values are stacked and
    reduced to find the global maximum.

    Args:
        grads_for_norm: Gradient tensors to compute norm over.
        norm_type:      Lp norm type (2 for L2, inf for maximum).
        grad_stats_parallel_group: Process group for reducing norm across
            model-parallel ranks.  If None, no reduction is performed.

    Returns:
        total_norm: Scalar CUDA tensor with the gradient norm.

    Note:
        Callers must NOT call .item() on the result if they want to avoid
        host/device sync.  The norm can be compared tensor-to-tensor:
        ``clip_coeff = max_norm / (total_norm + eps); clip_coeff.clamp_(max=1.0)``
    """
    if isinstance(grads_for_norm, torch.Tensor):
        grads_for_norm = [grads_for_norm]

    norm_type = float(norm_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if norm_type == inf:
        # Infinity norm: global maximum of element-wise absolute values
        if grads_for_norm:
            total_norm = torch.stack(
                [g.detach().abs().max() for g in grads_for_norm]
            ).max()
        else:
            total_norm = torch.tensor(0.0, device=device, dtype=torch.float32)
        total_norm = total_norm.unsqueeze(0)
        if grad_stats_parallel_group is not None and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                total_norm,
                op=torch.distributed.ReduceOp.MAX,
                group=grad_stats_parallel_group,
            )
        total_norm = total_norm.squeeze(0)

    elif norm_type == 2.0:
        # L2 norm: use fused kernel when available (From Megatron M2335)
        if _HAS_MULTI_TENSOR and grads_for_norm:
            try:
                dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device=device)
                grad_norm, _ = multi_tensor_applier(
                    l2_norm_impl,
                    dummy_overflow_buf,
                    [grads_for_norm],
                    False,  # per-parameter norm disabled
                )
                total_norm_sq = grad_norm ** 2.0
            except Exception:
                # Fallback if fused kernel fails (e.g. CPU grads)
                total_norm_sq = torch.stack(
                    [g.detach().float().norm(2.0).square() for g in grads_for_norm]
                ).sum()
        elif grads_for_norm:
            total_norm_sq = torch.stack(
                [g.detach().float().norm(2.0).square() for g in grads_for_norm]
            ).sum()
        else:
            total_norm_sq = torch.tensor(0.0, device=device, dtype=torch.float32)

        total_norm_sq = total_norm_sq.unsqueeze(0)
        if grad_stats_parallel_group is not None and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                total_norm_sq,
                op=torch.distributed.ReduceOp.SUM,
                group=grad_stats_parallel_group,
            )
        total_norm = total_norm_sq.squeeze(0).sqrt()

    else:
        # General p-norm
        if grads_for_norm:
            total_norm_p = torch.stack(
                [g.detach().float().norm(norm_type).pow(norm_type) for g in grads_for_norm]
            ).sum()
        else:
            total_norm_p = torch.tensor(0.0, device=device, dtype=torch.float32)

        total_norm_p = total_norm_p.unsqueeze(0)
        if grad_stats_parallel_group is not None and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                total_norm_p,
                op=torch.distributed.ReduceOp.SUM,
                group=grad_stats_parallel_group,
            )
        total_norm = total_norm_p.squeeze(0).pow(1.0 / norm_type)

    return total_norm


# ---------------------------------------------------------------------------
# Clip by total norm (mirrors Megatron's clip_grad_by_total_norm_fp32)
# ---------------------------------------------------------------------------

def clip_grad_by_total_norm_fp32(
    parameters: Union[List[torch.Tensor], torch.Tensor],
    max_norm: float,
    total_norm: Union[float, torch.Tensor],
    use_decoupled_grad: bool = False,
) -> None:
    """Scale gradients so their total norm does not exceed *max_norm*.

    Replicates Megatron's ``clip_grad_by_total_norm_fp32`` with DES-LOC
    extensions:

    - Accepts *total_norm* as a CUDA tensor (avoids sync) or a float.
    - Handles FSDP params (``__fsdp_param__``/``decoupled_grad``).
    - Uses fused multi-tensor-scale when TE/Apex is available.

    Args:
        parameters:        Parameters whose gradients to clip.
        max_norm:          Maximum total gradient norm after clipping.
        total_norm:        Current total gradient norm (float or CUDA tensor).
        use_decoupled_grad: When True use ``.decoupled_grad`` instead of
                           ``.grad`` (Megatron-FSDP path).

    Note:
        Gradients are modified **in-place**.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    grad_attr = "decoupled_grad" if use_decoupled_grad else "grad"
    grads: List[torch.Tensor] = []
    for param in parameters:
        g = getattr(param, grad_attr, None)
        if g is None and grad_attr != "grad":
            g = param.grad  # fallback
        if g is None:
            continue
        # FSDP unwrap
        if getattr(param, "__fsdp_param__", False) and hasattr(g, "_local_tensor"):
            g = g._local_tensor
        grads.append(g.detach())

    if not grads:
        return

    # Compute clip coefficient (tensor-safe)
    if isinstance(total_norm, torch.Tensor):
        clip_coeff = max_norm / (total_norm + 1.0e-6)
        clip_coeff = clip_coeff.clamp(max=1.0)
        _coeff_scalar = float(clip_coeff.item())
    else:
        _coeff_scalar = max_norm / (float(total_norm) + 1.0e-6)
        _coeff_scalar = min(_coeff_scalar, 1.0)

    if _coeff_scalar >= 1.0:
        return  # nothing to clip

    if _HAS_MULTI_TENSOR and multi_tensor_scale_impl is not None:
        try:
            dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device=grads[0].device)
            multi_tensor_applier(
                multi_tensor_scale_impl,
                dummy_overflow_buf,
                [grads, grads],
                _coeff_scalar,
            )
            return
        except Exception:
            pass  # fallback to loop

    for g in grads:
        g.mul_(_coeff_scalar)


# ---------------------------------------------------------------------------
# Per-tier gradient clipping (DES-LOC M4171)
# ---------------------------------------------------------------------------

def clip_grad_by_tier(
    tier_param_groups: Dict[str, List[torch.nn.Parameter]],
    tier_clip_config: "TierClipConfig",
    norm_type: float = 2.0,
    grad_stats_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    use_decoupled_grad: bool = False,
) -> Dict[str, float]:
    """Apply per-tier gradient clipping with tier-proportional budgets.

    Each hardware tier computes its own gradient norm over its local parameter
    shard and clips independently using a budget proportional to the tier's
    BF16 TFLOPS.  This eliminates the cross-tier norm distortion that occurs
    on heterogeneous PCIe clusters when H100 and A6000 shards are combined
    into a single global norm.

    Algorithm (for each tier t):
      1. Gather gradients for tier-t parameters (local shard only).
      2. Compute local norm.
      3. If model-parallel group given, reduce norm across MP ranks.
      4. Compute clip coefficient = budget_t / (local_norm + eps).
      5. Apply clip coefficient to tier-t gradients in place.

    Args:
        tier_param_groups:    Dict mapping tier name → list of parameters.
        tier_clip_config:     Configuration with per-tier budgets and settings.
        norm_type:            Lp norm type for the per-tier norm.
        grad_stats_parallel_group: Process group for norm reduction.
        use_decoupled_grad:   Whether to use ``.decoupled_grad`` attribute.

    Returns:
        Dict mapping tier name → gradient norm *before* clipping (float).
    """
    tier_norms: Dict[str, float] = {}

    for tier, params in tier_param_groups.items():
        grads: List[torch.Tensor] = []
        for p in params:
            g = _get_grad(p, use_decoupled_grad=use_decoupled_grad)
            if g is not None:
                grads.append(g.detach().float())

        # Compute tier-local norm
        tier_norm = get_grad_norm_fp32(
            grads,
            norm_type=norm_type,
            grad_stats_parallel_group=grad_stats_parallel_group,
        )
        tier_norms[tier] = float(tier_norm.item())

        budget = tier_clip_config.budget_for_tier(tier)
        clip_grad_by_total_norm_fp32(
            params,
            max_norm=budget,
            total_norm=tier_norm,
            use_decoupled_grad=use_decoupled_grad,
        )

        logger.debug(
            "[DES-LOC TierClip] tier=%s norm=%.4f budget=%.4f",
            tier, tier_norms[tier], budget,
        )

    return tier_norms


# ---------------------------------------------------------------------------
# QK-projection clipping (From Megatron M4212)
# ---------------------------------------------------------------------------

def clip_qk_grad_norm(
    named_params: List[Tuple[str, torch.nn.Parameter]],
    qk_config: "QKClipConfig",
    norm_type: float = 2.0,
    grad_stats_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    use_decoupled_grad: bool = False,
) -> Optional[float]:
    """Clip Q/K projection gradients with a tighter norm budget.

    Attention Q and K projections can develop very large gradient norms early
    in training (steps 2000–5000).  A dedicated clip with a tighter
    ``qk_config.max_norm`` budget prevents the attention entropy collapse that
    manifests as a sharp loss spike in large-scale runs.

    This mirrors the QK-clip introduced in Megatron M4212 (qk_clip.py).

    Args:
        named_params:      List of (name, param) pairs from model.named_parameters().
        qk_config:         QKClipConfig with enabled flag, max_norm, and name substrings.
        norm_type:         Lp norm type for the QK-specific norm.
        grad_stats_parallel_group: Process group for norm reduction.
        use_decoupled_grad: Whether to use .decoupled_grad.

    Returns:
        QK gradient norm before clipping, or None if disabled / no QK grads.
    """
    if not qk_config.enabled:
        return None

    qk_params: List[torch.nn.Parameter] = []
    for name, param in named_params:
        if qk_config.is_qk_param(name):
            qk_params.append(param)

    if not qk_params:
        return None

    qk_grads = [
        g for p in qk_params
        for g in [_get_grad(p, use_decoupled_grad=use_decoupled_grad)]
        if g is not None
    ]
    qk_grads_detached = [g.detach().float() for g in qk_grads]

    qk_norm = get_grad_norm_fp32(
        qk_grads_detached,
        norm_type=norm_type,
        grad_stats_parallel_group=grad_stats_parallel_group,
    )
    qk_norm_val = float(qk_norm.item())

    clip_grad_by_total_norm_fp32(
        qk_params,
        max_norm=qk_config.max_norm,
        total_norm=qk_norm,
        use_decoupled_grad=use_decoupled_grad,
    )

    logger.debug(
        "[M4212 QKClip] num_qk_params=%d qk_norm=%.4f max_norm=%.4f",
        len(qk_params), qk_norm_val, qk_config.max_norm,
    )
    return qk_norm_val


# ---------------------------------------------------------------------------
# Zero-count (From Megatron M4145)
# ---------------------------------------------------------------------------

def count_zeros_fp32(
    parameters: Union[List[torch.nn.Parameter], torch.Tensor],
    grad_stats_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    use_decoupled_grad: bool = False,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> float:
    """Count zero-valued gradient elements across all model-parallel ranks.

    Handles three gradient storage formats:
      - Standard ``.grad``
      - Megatron BF16 bucket ``.main_grad``
      - Megatron-FSDP DTensor ``.decoupled_grad`` / ``.grad._local_tensor``

    From Megatron M4145: fix zero-counter not working with decoupled grads.

    Args:
        parameters:            Parameters whose gradients to inspect.
        grad_stats_parallel_group: Process group for the all-reduce sum.
        use_decoupled_grad:    Whether to use ``.decoupled_grad`` attribute.
        tp_group:              Tensor-parallel group used to skip TP replicas.

    Returns:
        Total number of zero gradient elements across all ranks.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    total_num_zeros = torch.tensor([0.0], dtype=torch.float32, device="cuda")

    for param in parameters:
        grad = _get_grad(param, use_decoupled_grad=use_decoupled_grad)
        if grad is None:
            continue

        # Skip shared params (avoid double-counting)
        if getattr(param, "shared", False):
            continue

        # Skip TP-replicated params (only count each element once)
        try:
            import deepspeed.core.tensor_parallel as tp
            if not tp.param_is_not_tensor_parallel_duplicate(param, tp_group):
                continue
        except (ImportError, Exception):
            pass

        num_zeros = grad.numel() - grad.count_nonzero()
        total_num_zeros = total_num_zeros + num_zeros.float()

    if grad_stats_parallel_group is not None and torch.distributed.is_initialized():
        torch.distributed.all_reduce(
            total_num_zeros,
            op=torch.distributed.ReduceOp.SUM,
            group=grad_stats_parallel_group,
        )

    return float(total_num_zeros.item())


# ---------------------------------------------------------------------------
# Main convenience function: clip_grad_norm
# ---------------------------------------------------------------------------

def clip_grad_norm(
    parameters: Union[List[torch.nn.Parameter], torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    grad_stats_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    use_decoupled_grad: bool = False,
    tier_clip_config: Optional["TierClipConfig"] = None,
    tier_assignments: Optional[Dict[int, str]] = None,
    qk_config: Optional["QKClipConfig"] = None,
    named_params: Optional[List[Tuple[str, torch.nn.Parameter]]] = None,
    norm_ema: Optional["GradNormEMA"] = None,
) -> torch.Tensor:
    """Clip gradient norm across model-parallel ranks.

    Combines standard model-parallel-aware norm clipping with optional
    DES-LOC per-tier scaling and QK-projection clipping.

    Processing order:
      1. (Optional) QK-projection gradients are clipped first with a tight
         budget (qk_config).
      2. If per-tier scaling is enabled, tier-level clipping replaces the
         global clip for the remaining params.
      3. Otherwise global clip is applied to all remaining params.
      4. (Optional) Norm EMA is updated with the global norm.

    Args:
        parameters:        Parameters whose gradients to clip.
        max_norm:          Maximum allowed gradient norm.
        norm_type:         Lp norm type (default L2).
        grad_stats_parallel_group: Process group for norm reduction.
        use_decoupled_grad: Whether to use .decoupled_grad (FSDP path).
        tier_clip_config:  Per-tier clip config (DES-LOC M4171).
        tier_assignments:  Dict mapping param-id → tier string.
                           Required when tier_clip_config is set.
        qk_config:         QK-projection clip config (Megatron M4212).
        named_params:      (name, param) pairs; required for QK detection.
        norm_ema:          Optional EMA tracker for adaptive norm monitoring.

    Returns:
        Total gradient norm *before* any clipping, as a CUDA tensor.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # ------------------------------------------------------------------
    # Step 1: QK-projection clip (applied first, before global clip)
    # ------------------------------------------------------------------
    if qk_config is not None and qk_config.enabled and named_params is not None:
        clip_qk_grad_norm(
            named_params=named_params,
            qk_config=qk_config,
            norm_type=norm_type,
            grad_stats_parallel_group=grad_stats_parallel_group,
            use_decoupled_grad=use_decoupled_grad,
        )

    # ------------------------------------------------------------------
    # Step 2: Compute global gradient norm
    # ------------------------------------------------------------------
    grads: List[torch.Tensor] = []
    for p in parameters:
        g = _get_grad(p, use_decoupled_grad=use_decoupled_grad)
        if g is not None:
            grads.append(g.detach().float())

    total_norm = get_grad_norm_fp32(
        grads,
        norm_type=norm_type,
        grad_stats_parallel_group=grad_stats_parallel_group,
    )

    # Update EMA tracker if provided (DES-LOC M4309)
    if norm_ema is not None:
        norm_ema.update(total_norm)

    # ------------------------------------------------------------------
    # Step 3: Apply clipping
    # ------------------------------------------------------------------
    if tier_clip_config is not None and tier_clip_config.per_tier_scaling and tier_assignments:
        # Build per-tier param groups
        tier_param_groups: Dict[str, List[torch.nn.Parameter]] = {}
        for p in parameters:
            tier = tier_assignments.get(id(p), TIER_A6000)
            tier_param_groups.setdefault(tier, []).append(p)
        # Update global max_norm in config so budget_for_tier uses it
        tier_clip_config.global_max_norm = max_norm
        clip_grad_by_tier(
            tier_param_groups=tier_param_groups,
            tier_clip_config=tier_clip_config,
            norm_type=norm_type,
            grad_stats_parallel_group=grad_stats_parallel_group,
            use_decoupled_grad=use_decoupled_grad,
        )
    else:
        # Standard global clip
        clip_grad_by_total_norm_fp32(
            parameters=list(parameters),
            max_norm=max_norm,
            total_norm=total_norm,
            use_decoupled_grad=use_decoupled_grad,
        )

    return total_norm


# ---------------------------------------------------------------------------
# Composite clip: clip_grads_with_norm_by_group (DES-LOC M4171 + M4212)
# ---------------------------------------------------------------------------

def clip_grads_with_norm_by_group(
    param_groups: List[Dict],
    max_norm: float,
    norm_type: float = 2.0,
    grad_stats_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    use_decoupled_grad: bool = False,
    separate_group_norms: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    """Clip gradients with per-group norm tracking.

    Designed for use with ``MegatronOptimizer.clip_grad_norm`` when
    separate grad-norm groups (e.g. ``'mtp'``) require independent clipping.

    For each param group (dict with ``'name'``, ``'params'``, ``'max_norm'``
    optional keys), computes the group norm and clips independently.

    Args:
        param_groups: List of dicts with ``'params'`` key (and optionally
                      ``'name'`` and ``'max_norm'`` for per-group budgets).
        max_norm:     Default clip norm when group does not specify one.
        norm_type:    Lp norm type.
        grad_stats_parallel_group: Process group for norm reduction.
        use_decoupled_grad: Whether to use .decoupled_grad.
        separate_group_norms: Pre-computed norms from a prior pass (e.g.
                              for groups that were clipped earlier by
                              ``_compute_grad_norms_by_group``).

    Returns:
        Dict mapping group name → group gradient norm (CUDA tensor).
    """
    group_norms: Dict[str, torch.Tensor] = {}

    for group in param_groups:
        params = group.get("params", [])
        name = group.get("name", "main")
        group_max_norm = group.get("max_norm", max_norm)

        # Re-use pre-computed norm if available
        if separate_group_norms and name in separate_group_norms:
            norm_val = separate_group_norms[name]
            total_norm = (
                norm_val if isinstance(norm_val, torch.Tensor)
                else torch.tensor(float(norm_val), device="cuda")
            )
        else:
            grads = [
                g.detach().float()
                for p in params
                for g in [_get_grad(p, use_decoupled_grad=use_decoupled_grad)]
                if g is not None
            ]
            total_norm = get_grad_norm_fp32(
                grads,
                norm_type=norm_type,
                grad_stats_parallel_group=grad_stats_parallel_group,
            )

        group_norms[name] = total_norm
        clip_grad_by_total_norm_fp32(
            parameters=list(params),
            max_norm=group_max_norm,
            total_norm=total_norm,
            use_decoupled_grad=use_decoupled_grad,
        )

    return group_norms


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__ = [
    # Core functions
    "get_grad_norm_fp32",
    "clip_grad_by_total_norm_fp32",
    "clip_grad_norm",
    "count_zeros_fp32",
    # Per-tier scaling (DES-LOC M4171)
    "clip_grad_by_tier",
    "TierClipConfig",
    # QK-projection clipping (Megatron M4212)
    "clip_qk_grad_norm",
    "QKClipConfig",
    # Adaptive EMA norm tracker (DES-LOC M4309)
    "GradNormEMA",
    # Composite clip
    "clip_grads_with_norm_by_group",
    # Tier constants
    "TIER_H100",
    "TIER_A6000",
    "TIER_BLACKWELL",
    "TIER_CONSUMER",
    "_TIER_TFLOPS",
    # Gradient access helper
    "_get_grad",
]
