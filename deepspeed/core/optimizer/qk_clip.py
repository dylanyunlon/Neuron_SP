# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""QK-logit and QK-gradient clipping for attention stability.

Ported from Megatron-LM/megatron/core/optimizer/qk_clip.py with DES-LOC
extensions for heterogeneous multi-tier training.

Problem statement
-----------------
In large language models trained at high learning rates or on noisy data,
attention Q and K projections can develop very large gradient norms during
the early to mid stages of training (typically steps 2000–5000).  These
gradient spikes cause the attention entropy to collapse — the softmax
becomes extremely peaked, effectively ignoring most tokens — which manifests
as a sharp spike in the training loss ("attention entropy collapse").

Two complementary mitigations are provided here:

1. **Attention logit clipping** (``clip_attention_logits``)
   Caps the raw QK attention scores before the softmax.  Applied in the
   forward pass by registering a forward hook on the attention module.
   Prevents the softmax from concentrating on a single position.

2. **QK gradient clipping** (``clip_qk_grad_norm``)
   Applies a tighter gradient norm budget to Q/K projection weight gradients
   (separate from the main gradient norm budget).  Applied in the backward
   pass by the optimizer's gradient clipping logic.

From Megatron M4212 / qk_clip.py; DES-LOC extension: cross-tier attention
logit all-reduce so that the global maximum logit is computed consistently
across H100, A6000, Blackwell, and Consumer GPU tiers connected via PCIe.

Public API
----------
  clip_attention_logits          — forward-pass logit capper
  register_qk_logit_hooks        — register forward hooks on all attention layers
  clip_qk_grad_norm              — backward-pass Q/K gradient clipper (convenience)
  QKLogitClipConfig              — configuration for logit clipping
  AttentionLogitMonitor          — tracks max attention logits for diagnostics
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

import deepspeed.core.parallel_state as parallel_state

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class QKLogitClipConfig:
    """Configuration for QK attention logit and gradient clipping.

    Attributes:
        logit_softcap:       If > 0, apply tanh soft-capping at this value
                             (``score = softcap * tanh(score / softcap)``).
                             Used by Gemma-2 and similar architectures.
        logit_hard_clip:     If > 0, apply hard clipping (``score.clamp(max=logit_hard_clip)``).
                             Simpler but less smooth than softcap.
        grad_max_norm:       Separate gradient clip norm for Q/K projections.
                             Applied by the optimizer before the main clip.
        qk_param_names:      Name substrings to identify Q/K projection params.
        log_max_only:        If True, compute and log the max logit without clipping.
        data_parallel_group: Process group for max-logit all-reduce (DES-LOC).

    Examples::

        cfg = QKLogitClipConfig(
            logit_softcap=50.0,   # Gemma-2 style
            grad_max_norm=0.5,
        )
    """
    logit_softcap: float = 0.0
    logit_hard_clip: float = 0.0
    grad_max_norm: float = 0.5
    qk_param_names: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "query_key_value", "wq", "wk",
        "query_weight", "key_weight", "linear_q", "linear_k",
    ])
    log_max_only: bool = False
    data_parallel_group: Optional[torch.distributed.ProcessGroup] = None


# ---------------------------------------------------------------------------
# Attention logit monitor
# ---------------------------------------------------------------------------

class AttentionLogitMonitor:
    """Track maximum attention logits across training steps.

    Maintains a running maximum of the attention logit values observed during
    the forward pass.  Used by the QK-clip logic to detect when logits are
    growing dangerously large and by diagnostics dashboards.

    Thread-safety: not thread-safe; designed for single-threaded training.

    Attributes:
        max_logit_history: List of (step, max_logit) pairs.
        _current_max:      Maximum logit observed in the current forward pass.
    """

    def __init__(self) -> None:
        self.max_logit_history: List[Tuple[int, float]] = []
        self._current_max: Optional[torch.Tensor] = None
        self._step: int = 0

    def update(self, logits: torch.Tensor) -> None:
        """Update the current-step maximum with a new attention logit tensor.

        Args:
            logits: Attention score tensor of any shape.  The global max is
                    extracted and merged with the running per-step maximum.
        """
        step_max = logits.detach().abs().max()
        if self._current_max is None:
            self._current_max = step_max
        else:
            self._current_max = torch.maximum(self._current_max, step_max)

    def step_done(
        self,
        data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> float:
        """Finalise the current step's max logit and record it.

        Performs an all-reduce(MAX) across data-parallel ranks so that the
        recorded maximum is globally consistent (required on PCIe clusters
        where different ranks may see different mini-batches).

        Args:
            data_parallel_group: Process group for the all-reduce.

        Returns:
            The globally-reduced maximum logit for this step, or 0.0 if
            no logits were observed (e.g., the forward pass was skipped).
        """
        if self._current_max is None:
            self.max_logit_history.append((self._step, 0.0))
            self._step += 1
            return 0.0

        max_val = self._current_max.unsqueeze(0)
        if data_parallel_group is not None and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                max_val,
                op=torch.distributed.ReduceOp.MAX,
                group=data_parallel_group,
            )
        val = float(max_val.item())
        self.max_logit_history.append((self._step, val))
        self._current_max = None
        self._step += 1
        return val

    def get_recent_max(self, n_steps: int = 10) -> float:
        """Return the maximum logit observed in the last *n_steps* steps."""
        if not self.max_logit_history:
            return 0.0
        recent = self.max_logit_history[-n_steps:]
        return max(v for _, v in recent)

    def is_dangerous(self, threshold: float = 100.0, n_steps: int = 10) -> bool:
        """Return True if recent max logit exceeds *threshold*.

        Args:
            threshold: Logit value above which attention entropy collapse
                       is considered imminent.
            n_steps:   Number of recent steps to inspect.
        """
        return self.get_recent_max(n_steps) > threshold


# Global monitor instance (shared across all attention modules in the process)
_GLOBAL_LOGIT_MONITOR: AttentionLogitMonitor = AttentionLogitMonitor()


# ---------------------------------------------------------------------------
# Forward-pass logit clipping
# ---------------------------------------------------------------------------

def clip_attention_logits(
    attention_scores: torch.Tensor,
    config: QKLogitClipConfig,
    monitor: Optional[AttentionLogitMonitor] = None,
) -> torch.Tensor:
    """Apply softcap or hard-clip to attention logit scores.

    This function is designed to be called inside the attention forward pass
    after computing ``QK^T / sqrt(d_k)`` but before the softmax.

    Two clipping modes:
      - **Softcap**: ``score = softcap * tanh(score / softcap)``
        Smooth, differentiable; used by Gemma-2.
      - **Hard clip**: ``score = score.clamp(max=clip_value)``
        Discontinuous gradient; simpler to implement.

    Only one of ``logit_softcap`` or ``logit_hard_clip`` should be set > 0
    (softcap takes precedence when both are positive).

    Args:
        attention_scores: Raw attention logit tensor, shape (..., seq_len, seq_len).
        config:           QKLogitClipConfig with clip parameters.
        monitor:          Optional logit monitor for diagnostics.

    Returns:
        Clipped attention logit tensor of the same shape.
    """
    if monitor is not None:
        monitor.update(attention_scores)

    if config.logit_softcap > 0.0:
        # Softcap: smooth tanh clipping (Gemma-2 style)
        s = config.logit_softcap
        attention_scores = s * torch.tanh(attention_scores / s)

    elif config.logit_hard_clip > 0.0:
        # Hard clip: simple clamping
        attention_scores = attention_scores.clamp(max=config.logit_hard_clip)

    return attention_scores


# ---------------------------------------------------------------------------
# Forward hook registration
# ---------------------------------------------------------------------------

class _QKLogitHook:
    """Forward hook that clips attention logit tensors in place.

    Registered on attention modules via ``register_qk_logit_hooks``.
    Intercepts the output of the attention score computation and applies
    the configured clipping before it is passed to softmax.

    The hook identifies the attention score tensor by finding the output
    or intermediate tensor with the expected (batch, heads, seq, seq) shape.
    """

    def __init__(
        self,
        config: QKLogitClipConfig,
        monitor: Optional[AttentionLogitMonitor] = None,
    ) -> None:
        self.config = config
        self.monitor = monitor

    def __call__(
        self,
        module: torch.nn.Module,
        inputs: Tuple,
        output: Any,
    ) -> Any:
        if isinstance(output, torch.Tensor) and output.dim() == 4:
            # Likely (batch, heads, seq_q, seq_k) attention score tensor
            return clip_attention_logits(output, self.config, self.monitor)
        elif isinstance(output, (tuple, list)):
            # Some attention implementations return (scores, weights, ...)
            result = list(output)
            for i, t in enumerate(result):
                if isinstance(t, torch.Tensor) and t.dim() == 4:
                    result[i] = clip_attention_logits(t, self.config, self.monitor)
                    break
            return type(output)(result)
        return output


def register_qk_logit_hooks(
    model_chunks: List[torch.nn.Module],
    config: QKLogitClipConfig,
    monitor: Optional[AttentionLogitMonitor] = None,
) -> List[torch.utils.hooks.RemovableHook]:
    """Register forward hooks on all attention modules in *model_chunks*.

    Searches for modules whose class name contains "Attention" (case-insensitive)
    or whose name contains "self_attention" / "cross_attention".  Registers a
    :class:`_QKLogitHook` that clips attention scores in the forward pass.

    Args:
        model_chunks: List of model chunks to hook.
        config:       QKLogitClipConfig controlling the clip parameters.
        monitor:      Optional logit monitor for diagnostics.

    Returns:
        List of ``RemovableHook`` handles.  Call ``handle.remove()`` on each
        to deregister the hooks (e.g., at the end of training).

    Examples::

        handles = register_qk_logit_hooks(
            [model],
            QKLogitClipConfig(logit_softcap=50.0),
        )
        # ... training ...
        for h in handles:
            h.remove()
    """
    if monitor is None:
        monitor = _GLOBAL_LOGIT_MONITOR

    hook = _QKLogitHook(config, monitor)
    handles: List[torch.utils.hooks.RemovableHook] = []

    for chunk in model_chunks:
        for name, module in chunk.named_modules():
            cls_name = type(module).__name__.lower()
            is_attention = (
                "attention" in cls_name
                or "self_attention" in name.lower()
                or "cross_attention" in name.lower()
            )
            if is_attention:
                handle = module.register_forward_hook(hook)
                handles.append(handle)
                logger.debug(
                    "[QKLogit] registered hook on %s (%s)",
                    name,
                    type(module).__name__,
                )

    logger.info(
        "register_qk_logit_hooks: registered %d hooks (softcap=%.1f, hard_clip=%.1f)",
        len(handles),
        config.logit_softcap,
        config.logit_hard_clip,
    )
    return handles


# ---------------------------------------------------------------------------
# Backward-pass QK gradient clipping
# ---------------------------------------------------------------------------

def clip_qk_grad_norm(
    model_chunks: List[torch.nn.Module],
    config: QKLogitClipConfig,
    norm_type: float = 2.0,
    use_decoupled_grad: bool = False,
) -> Optional[float]:
    """Clip Q/K projection gradients with a tighter norm budget.

    Identifies Q/K projection parameters by name (using
    ``config.qk_param_names``) and applies ``clip_grad_by_total_norm_fp32``
    with ``config.grad_max_norm``.

    This is the backward-pass complement to ``clip_attention_logits``.
    Applied after the backward pass, before the main gradient clip.

    From Megatron M4212: prevent attention entropy collapse via tighter
    QK gradient clipping.  On DES-LOC PCIe clusters the all-reduce for the
    QK grad norm uses ``config.data_parallel_group`` rather than the global
    group to avoid mixing heterogeneous tier gradients.

    Args:
        model_chunks:       List of model chunks.
        config:             QKLogitClipConfig with qk_param_names and grad_max_norm.
        norm_type:          Lp norm type (default L2).
        use_decoupled_grad: Whether to read from .decoupled_grad.

    Returns:
        QK gradient norm before clipping (float), or None if no QK params.
    """
    from deepspeed.core.optimizer.clip_grads import (
        get_grad_norm_fp32,
        clip_grad_by_total_norm_fp32,
        _get_grad,
    )

    if config.grad_max_norm <= 0.0:
        return None

    # Collect Q/K params by name
    qk_params: List[torch.nn.Parameter] = []
    for chunk in model_chunks:
        for name, param in chunk.named_parameters():
            if not param.requires_grad:
                continue
            if any(s in name for s in config.qk_param_names):
                qk_params.append(param)

    if not qk_params:
        return None

    # Compute QK grad norm
    qk_grads = [
        g.detach().float()
        for p in qk_params
        for g in [_get_grad(p, use_decoupled_grad=use_decoupled_grad)]
        if g is not None
    ]

    dp_group = config.data_parallel_group
    if dp_group is None and parallel_state.is_initialized():
        dp_group = parallel_state.get_data_parallel_group()

    qk_norm = get_grad_norm_fp32(
        qk_grads,
        norm_type=norm_type,
        grad_stats_parallel_group=dp_group,
    )
    qk_norm_val = float(qk_norm.item())

    if not config.log_max_only:
        clip_grad_by_total_norm_fp32(
            parameters=qk_params,
            max_norm=config.grad_max_norm,
            total_norm=qk_norm,
            use_decoupled_grad=use_decoupled_grad,
        )

    logger.debug(
        "[M4212 QKGradClip] num_params=%d qk_norm=%.4f max_norm=%.4f clipped=%s",
        len(qk_params),
        qk_norm_val,
        config.grad_max_norm,
        not config.log_max_only,
    )

    return qk_norm_val


# ---------------------------------------------------------------------------
# Model-chunk level clip_qk dispatcher (mirrors Megatron qk_clip.clip_qk)
# ---------------------------------------------------------------------------

def clip_qk(
    model_chunks: List[torch.nn.Module],
    config: Optional[QKLogitClipConfig] = None,
    log_max_only: bool = False,
    monitor: Optional[AttentionLogitMonitor] = None,
) -> float:
    """Clip QK attention logits in all attention layers of *model_chunks*.

    High-level dispatcher that iterates all attention layers and calls
    ``clip_attention_logits`` on any cached ``current_max_attn_logits``
    tensors.  Compatible with the Megatron attention module interface where
    ``core_attention.current_max_attn_logits`` holds the step's max logit.

    Performs an all-reduce(MAX) across the data-parallel group so that the
    reported maximum is consistent across all ranks.

    Args:
        model_chunks:  List of model chunks.
        config:        QKLogitClipConfig; if None, uses default hardcap of 50.
        log_max_only:  If True, only log the maximum logit without clipping.
        monitor:       Optional logit monitor for diagnostics.

    Returns:
        Maximum attention logit observed across all layers and all ranks.
    """
    if config is None:
        config = QKLogitClipConfig(logit_hard_clip=50.0)

    dp_group = config.data_parallel_group
    if dp_group is None and parallel_state.is_initialized():
        dp_group = parallel_state.get_data_parallel_group()

    global_max_logit: float = 0.0

    with torch.no_grad():
        for chunk in model_chunks:
            # Walk all named modules looking for attention layers
            for name, module in chunk.named_modules():
                # Try the Megatron API first: core_attention.current_max_attn_logits
                core_attn = getattr(module, "core_attention", None)
                if core_attn is not None:
                    max_logit_t = getattr(core_attn, "current_max_attn_logits", None)
                    if max_logit_t is None:
                        continue
                    # Reduce across DP ranks
                    max_logit_buf = max_logit_t.clone()
                    if dp_group is not None and torch.distributed.is_initialized():
                        torch.distributed.all_reduce(
                            max_logit_buf,
                            op=torch.distributed.ReduceOp.MAX,
                            group=dp_group,
                        )
                    local_max = float(max_logit_buf.max().item())
                    global_max_logit = max(global_max_logit, local_max)

                    if monitor is not None:
                        monitor.update(max_logit_buf)

                    if not log_max_only:
                        # Apply softcap or hard-clip to the module's stored logits
                        if hasattr(module, "clip_qk") and callable(module.clip_qk):
                            module.clip_qk()
                        else:
                            # Fallback: directly apply to the logit tensor
                            if config.logit_softcap > 0.0:
                                s = config.logit_softcap
                                core_attn.current_max_attn_logits = (
                                    s * torch.tanh(max_logit_t / s)
                                )
                            elif config.logit_hard_clip > 0.0:
                                core_attn.current_max_attn_logits = max_logit_t.clamp(
                                    max=config.logit_hard_clip
                                )

    logger.debug("[clip_qk] global_max_logit=%.4f log_max_only=%s", global_max_logit, log_max_only)
    return global_max_logit


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "QKLogitClipConfig",
    "AttentionLogitMonitor",
    "clip_attention_logits",
    "register_qk_logit_hooks",
    "clip_qk_grad_norm",
    "clip_qk",
    "_GLOBAL_LOGIT_MONITOR",
]
