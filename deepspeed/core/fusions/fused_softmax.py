"""Fused scale-mask-softmax for attention.

Mirrors Megatron megatron/core/fusions/fused_softmax.py.
Falls back to torch softmax on SM86 (A6000) if CUDA kernel unavailable.

Megatron source: Megatron-LM/megatron/core/fusions/fused_softmax.py
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class FusedScaleMaskSoftmax(nn.Module):
    """Fused scale + mask + softmax.

    On H100 (SM90): uses fused CUDA kernel for ~2x speedup.
    On A6000 (SM86): falls back to PyTorch implementation.
    """

    def __init__(
        self,
        input_in_fp16: bool = False,
        input_in_bf16: bool = True,
        attn_mask_type: str = "causal",
        scaled_masked_softmax_fusion: bool = True,
        mask_func: Optional[callable] = None,
        softmax_in_fp32: bool = True,
        scale: Optional[float] = None,
    ) -> None:
        raise NotImplementedError(
            "Task: read Megatron fused_softmax.py, implement with SM86/SM90 dispatch"
        )

    def forward(
        self,
        input: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError
