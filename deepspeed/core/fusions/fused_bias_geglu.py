"""Fused bias + GEGLU/SwiGLU activation.

Mirrors Megatron megatron/core/fusions/fused_bias_geglu.py.

Megatron source: Megatron-LM/megatron/core/fusions/fused_bias_geglu.py
"""
from __future__ import annotations

import torch


def fused_bias_geglu(input: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Fused bias add + GEGLU activation.

    Splits input in half along last dim, applies GELU to first half,
    multiplies with second half. Fuses bias add into the kernel.
    """
    raise NotImplementedError("Task: implement fused bias + GEGLU")


def fused_bias_swiglu(input: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Fused bias add + SwiGLU activation (used by LLaMA-style models)."""
    raise NotImplementedError("Task: implement fused bias + SwiGLU")
