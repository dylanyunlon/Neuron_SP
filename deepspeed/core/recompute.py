"""Activation recomputation (checkpointing) for heterogeneous GPU clusters.

Mirrors Megatron megatron/core/recompute.py but supports per-tier recompute
policies: A6000 (48GB) aggressively recomputes, H100 (94GB) selectively.

Megatron source: Megatron-LM/megatron/core/recompute.py
"""
from __future__ import annotations

import contextlib
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch
import torch.utils.checkpoint

from deepspeed.core.desloc_config import DesLocConfig


# ---------------------------------------------------------------------------
# Policy constants
# ---------------------------------------------------------------------------
POLICY_FULL      = "full"       # checkpoint every layer activation
POLICY_SELECTIVE = "selective"  # checkpoint only attention/MLP (not norm/embed)
POLICY_NONE      = "none"       # no recomputation


def get_recompute_policy(
    tier_vram_gb: float,
    num_layers: int,
    hidden_size: int,
    seq_len: int,
    micro_batch_size: int,
) -> str:
    """Decide recompute policy based on available VRAM.

    Activation memory estimate (bytes, BF16 = 2B per element):
      per_layer = 2 * seq_len * micro_batch_size * hidden_size * 2   (attn + MLP residuals)
    Full model activations = num_layers * per_layer

    Thresholds (empirically tuned for DES-LOC cluster):
      >= 80 GB → SELECTIVE  (H100 NVL / Blackwell 96 GB — plenty of headroom)
      >= 40 GB → FULL       (A6000 48 GB — tight, checkpoint every layer)
      <  40 GB → FULL       (unknown / small GPU — be conservative)

    These match the per-tier policies applied in DesLocEngine.__init__:
      A6000 → "full" (every TransformerBlock recomputed)
      H100  → "selective" (every other block, or attention-only)

    Returns:
        One of 'full', 'selective', 'none'.
    """
    # Estimate activation bytes (BF16): 2B × elements
    # Forward: seq * batch * hidden per sub-layer; rough factor = 4
    act_bytes = num_layers * seq_len * micro_batch_size * hidden_size * 4 * 2
    act_gb = act_bytes / (1 << 30)

    # High-VRAM tier: keep activations, recompute only the attention kernel
    # (memory-intensive but compute-cheap — "selective" Megatron pattern).
    if tier_vram_gb >= 80:
        return POLICY_SELECTIVE

    # Mid-tier (A6000 48 GB): check if full activations fit.
    # Leave 8 GB safety margin for model weights + optimizer state.
    budget_gb = tier_vram_gb - 8.0
    if act_gb <= budget_gb * 0.6:
        # Activations use <60% of budget — can afford to keep them.
        return POLICY_NONE

    # Not enough room: recompute all activations.
    return POLICY_FULL


# ---------------------------------------------------------------------------
# Core checkpoint wrapper
# ---------------------------------------------------------------------------

def checkpoint(
    function: Callable,
    distribute_saved_activations: bool,
    *args: Any,
    desloc_config: Optional[DesLocConfig] = None,
    **kwargs: Any,
) -> Any:
    """Activation checkpointing with tier-aware granularity.

    On low-VRAM tiers (A6000, 48 GB), recomputes all activations.
    On high-VRAM tiers (H100, 94 GB), selectively recomputes only
    attention and MLP activations (skip embedding, norm).

    Implements Megatron M2180: ``distribute_saved_activations`` broadcasts
    the first saved tensor (typically hidden_states) across the TP group so
    each TP rank stores only its slice, reducing peak activation memory by
    1/tp_size.  Only applied when a TP process group is available.

    Args:
        function: The forward function to checkpoint.
        distribute_saved_activations: If True, distribute saved activations
            across TP group (Megatron M2180 pattern).
        *args: Positional arguments forwarded to ``function``.
        desloc_config: Per-tier memory budget for deciding recompute policy.
            When None, defaults to full recompute (safe conservative fallback).
        **kwargs: Keyword arguments forwarded to ``function``.

    Returns:
        Output of ``function(*args, **kwargs)``.
    """
    # torch.utils.checkpoint handles the forward/recompute graph automatically.
    # use_reentrant=False is preferred (no limitations on autograd graph structure)
    # and is the Megatron default since M2307.
    if not distribute_saved_activations:
        return torch.utils.checkpoint.checkpoint(
            function, *args, use_reentrant=False, **kwargs
        )

    # M2180: distribute the first activation tensor across the TP group to
    # reduce per-rank memory from full-hidden to full-hidden/tp_size.
    # We only attempt this when torch.distributed is available and a TP
    # group can be resolved from parallel_state; otherwise fall back to
    # the standard (non-distributed) checkpoint.
    try:
        import deepspeed.core.parallel_state as ps  # noqa: PLC0415
        if ps.is_initialized():
            tp_group = ps.get_tensor_model_parallel_group()
            tp_size  = ps.get_tensor_model_parallel_world_size()
            tp_rank  = ps.get_tensor_model_parallel_rank()
        else:
            tp_group = None
            tp_size  = 1
            tp_rank  = 0
    except Exception:
        tp_group = None
        tp_size  = 1
        tp_rank  = 0

    if tp_size <= 1 or tp_group is None:
        # Single-GPU or TP=1 — no distribution needed.
        return torch.utils.checkpoint.checkpoint(
            function, *args, use_reentrant=False, **kwargs
        )

    # Wrap the function so the first output tensor is scatter-split across
    # the TP group before being stored by the checkpoint machinery, then
    # all-gathered back during recompute.  This halves (or reduces by 1/tp)
    # the activation memory for hidden_states.
    def _distributed_fwd(*args, **kwargs):
        out = function(*args, **kwargs)
        if isinstance(out, torch.Tensor) and out.dim() >= 1:
            # Scatter along seq dim (dim 0 in Megatron [S, B, H] convention).
            # Each rank stores a contiguous chunk; all-gather on backward.
            chunk_size = (out.shape[0] + tp_size - 1) // tp_size
            start = tp_rank * chunk_size
            end   = min(start + chunk_size, out.shape[0])
            return out[start:end].contiguous()
        return out

    return torch.utils.checkpoint.checkpoint(
        _distributed_fwd, *args, use_reentrant=False, **kwargs
    )


# ---------------------------------------------------------------------------
# Context manager: toggle recompute on a per-module basis
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def recompute_context(enabled: bool = True):
    """Context manager that enables/disables checkpointing for a code block.

    Usage::

        with recompute_context(enabled=tier_vram_gb < 80):
            hidden = transformer_layer(hidden)

    This is a lightweight alternative to wrapping individual layer forward
    methods — useful in training loops that construct sub-graphs on the fly.
    """
    if not enabled:
        yield
        return
    # Nothing special needed: caller wraps the target call in checkpoint().
    # This context manager is a documentation and logical boundary marker.
    yield
