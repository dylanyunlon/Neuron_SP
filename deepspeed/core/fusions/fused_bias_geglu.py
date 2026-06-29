"""Fused bias + GEGLU/SwiGLU activation.

Mirrors Megatron megatron/core/fusions/fused_bias_geglu.py.

Pure-PyTorch implementation — no custom CUDA kernel required.  The
split-and-multiply pattern is equivalent to the Triton / apex fused kernels
but uses standard torch ops so it works on SM86 (A6000) without any extra
compilation step.

For the DES-LOC cluster (2×A6000 + 1×H100, PCIe-only) the bottleneck is
PCIe bandwidth and expert routing overhead, not the activation function
itself, so a pure-PyTorch fallback is production-grade here.

Megatron source: Megatron-LM/megatron/core/fusions/fused_bias_geglu.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def fused_bias_geglu(input: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Fused bias add + GEGLU activation.

    Adds ``bias`` to ``input``, splits along the last dimension into two
    equal halves (gate, up), applies GELU to the gate, then multiplies:
        output = GELU(gate + bias_gate) * (up + bias_up)

    Matches Megatron's ``bias_geglu_impl`` pattern (M2312 quick_geglu variant).

    Args:
        input: ``[..., 2 * ffn_hidden_size]``
        bias:  ``[2 * ffn_hidden_size]``  (or broadcastable)

    Returns:
        ``[..., ffn_hidden_size]``
    """
    assert input.shape[-1] % 2 == 0, (
        f"fused_bias_geglu: last dim must be even, got {input.shape[-1]}"
    )
    x = input + bias
    gate, up = x.chunk(2, dim=-1)
    return F.gelu(gate, approximate="tanh") * up


def fused_bias_swiglu(input: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Fused bias add + SwiGLU activation (used by LLaMA-style models).

    Adds ``bias`` to ``input``, splits into two halves (gate, up),
    applies SiLU to the gate, then multiplies:
        output = SiLU(gate + bias_gate) * (up + bias_up)

    Matches Megatron's ``bias_swiglu_impl`` (M2346: gated_linear_unit + SiLU).

    Args:
        input: ``[..., 2 * ffn_hidden_size]``
        bias:  ``[2 * ffn_hidden_size]``  (or broadcastable)

    Returns:
        ``[..., ffn_hidden_size]``
    """
    assert input.shape[-1] % 2 == 0, (
        f"fused_bias_swiglu: last dim must be even, got {input.shape[-1]}"
    )
    x = input + bias
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


# ---------------------------------------------------------------------------
# Weighted variants — used by MoE experts (experts.py lines 174, 179)
# ---------------------------------------------------------------------------

def weighted_bias_swiglu_impl(
    input: torch.Tensor,
    bias: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """SwiGLU with per-token routing weight scaling.

    Computes ``fused_bias_swiglu(input, bias) * weight``.  The ``weight``
    tensor carries the top-k router probability for each token so that the
    expert output is properly scaled before being summed in the MoE layer.

    Args:
        input:  ``[num_tokens, 2 * ffn_hidden_size]``
        bias:   ``[2 * ffn_hidden_size]``
        weight: ``[num_tokens, 1]`` or ``[num_tokens]``

    Returns:
        ``[num_tokens, ffn_hidden_size]``
    """
    out = fused_bias_swiglu(input, bias)
    if weight.dim() == 1:
        weight = weight.unsqueeze(-1)
    return out * weight


def weighted_bias_quick_geglu_impl(
    input: torch.Tensor,
    bias: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Quick-GeGLU (tanh GELU approximation) with per-token routing weight.

    Megatron M2312 introduced ``quick_geglu`` as a faster approximation
    using ``F.gelu(..., approximate='tanh')`` instead of the exact erf-GELU.

    Args:
        input:  ``[num_tokens, 2 * ffn_hidden_size]``
        bias:   ``[2 * ffn_hidden_size]``
        weight: ``[num_tokens, 1]`` or ``[num_tokens]``

    Returns:
        ``[num_tokens, ffn_hidden_size]``
    """
    out = fused_bias_geglu(input, bias)   # fused_bias_geglu already uses tanh approx
    if weight.dim() == 1:
        weight = weight.unsqueeze(-1)
    return out * weight
