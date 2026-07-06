"""Fused bias + dropout + residual add.

Mirrors Megatron-LM megatron/core/fusions/fused_bias_dropout.py.

Provides bias-dropout-add in three forms:
- ``bias_dropout_add_unfused``: plain PyTorch, no JIT.
- ``bias_dropout_add_fused_train`` / ``bias_dropout_add_fused_inference``:
  JIT-scripted (or torch.compile on PyTorch ≥ 2.2) for training and inference
  respectively — avoids separate Dropout and Add kernel launches.
- ``get_bias_dropout_add``: dispatch helper used by TransformerLayer.

The x_with_bias tuple matches Megatron's calling convention so callers can
import this module as a drop-in replacement.

Megatron source: Megatron-LM/megatron/core/fusions/fused_bias_dropout.py
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from deepspeed.core.jit import jit_fuser


# ---------------------------------------------------------------------------
# Core implementation — single function used by all three export paths.
# ---------------------------------------------------------------------------

def _bias_dropout_add_func(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]],
    residual: torch.Tensor,
    prob: float,
    training: bool,
) -> torch.Tensor:
    """Core bias-dropout-add operation.

    Implements: ``residual + dropout(x + bias)`` with dtype-promotion for
    fp32 residual connections and optional in-place execution at inference
    to minimise peak VRAM on A6000 devices.

    Args:
        x_with_bias: Tuple of ``(x, bias)`` where ``bias`` may be ``None``.
        residual: Residual tensor; may be fp32 even when ``x`` is bf16/fp16.
        prob: Dropout probability (0.0 at inference).
        training: Whether the model is in training mode.

    Returns:
        ``[s, b, h]`` tensor with same dtype as ``residual``.
    """
    x, bias = x_with_bias

    # In-place is safe at inference when no gradient computation is needed.
    inplace = (
        not training
        and not x.requires_grad
        and not residual.requires_grad
        and (bias is None or not bias.requires_grad)
    )

    # Fp32 residual connection: upcast x (and bias) so the residual stream
    # stays in fp32, matching Megatron's fp32_residual_connection behaviour.
    if x.dtype != residual.dtype:
        x = x.to(residual.dtype)
        if bias is not None:
            bias = bias.to(residual.dtype)

    if bias is not None:
        if inplace:
            x.add_(bias)
        else:
            x = x + bias
        out = F.dropout(x, p=prob, training=training, inplace=inplace)
        if inplace:
            out.add_(residual)
        else:
            out = residual + out
        return out
    else:
        out = F.dropout(x, p=prob, training=training, inplace=inplace)
        if inplace:
            out.add_(residual)
        else:
            out = residual + out
        return out


# ---------------------------------------------------------------------------
# Public API — mirrors Megatron exactly.
# ---------------------------------------------------------------------------

def bias_dropout_add_unfused(training: bool):
    """Return an unfused bias-dropout-add callable.

    The returned callable has signature ``(x_with_bias, residual, prob)``.

    Args:
        training: Whether to run in training mode (enables dropout).

    Returns:
        ``Callable[[Tuple[Tensor, Optional[Tensor]], Tensor, float], Tensor]``
    """
    def _bias_dropout_add(
        x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]],
        residual: torch.Tensor,
        prob: float,
    ) -> torch.Tensor:
        return _bias_dropout_add_func(x_with_bias, residual, prob, training)

    return _bias_dropout_add


@jit_fuser
def bias_dropout_add_fused_train(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]],
    residual: torch.Tensor,
    prob: float,
) -> torch.Tensor:
    """JIT-fused bias-dropout-add for training.

    Fuses bias add, dropout, and residual add into a single kernel launch.
    On PyTorch ≥ 2.2 this is compiled with ``torch.compile``; on older
    versions it falls back to ``torch.jit.script``.

    Args:
        x_with_bias: ``(x [s, b, h], bias [h] or None)``.
        residual: ``[s, b, h]`` residual tensor.
        prob: Dropout probability.

    Returns:
        ``residual + dropout(x + bias)``, shape ``[s, b, h]``.
    """
    return _bias_dropout_add_func(x_with_bias, residual, prob, True)


@jit_fuser
def bias_dropout_add_fused_inference(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]],
    residual: torch.Tensor,
    prob: float,
) -> torch.Tensor:
    """JIT-fused bias-dropout-add for inference (dropout disabled).

    Args:
        x_with_bias: ``(x [s, b, h], bias [h] or None)``.
        residual: ``[s, b, h]`` residual tensor.
        prob: Ignored; dropout is disabled at inference.

    Returns:
        ``residual + x + bias``, shape ``[s, b, h]``.
    """
    return _bias_dropout_add_func(x_with_bias, residual, prob, False)


def get_bias_dropout_add(training: bool, fused: bool):
    """Return the appropriate bias-dropout-add function.

    Mirrors Megatron's ``get_bias_dropout_add`` dispatch pattern.  The
    fused JIT path avoids separate kernel launches for bias-add, dropout,
    and residual-add.  It is gated on ``fused=True`` (controlled by
    ``config.bias_dropout_fusion``).

    On the DES-LOC cluster the H100 tier benefits from JIT fusion;
    the A6000 tier uses the unfused path transparently.

    Args:
        training: Whether the model is in training mode.
        fused: Whether ``config.bias_dropout_fusion`` is enabled.

    Returns:
        Callable ``(x_with_bias, residual, prob) -> Tensor``.
    """
    if fused:
        # JIT scripting for an nn.Module with Dropout does not trigger fusion.
        # Use two separate functions to capture the training state at
        # compile time — matching Megatron's approach.
        if training:
            return bias_dropout_add_fused_train
        else:
            return bias_dropout_add_fused_inference
    else:
        return bias_dropout_add_unfused(training)
