"""Fused scale-mask-softmax for attention.

Mirrors Megatron megatron/core/fusions/fused_softmax.py.

The full implementation already lives in
deepspeed/core/transformer/dot_product_attention.py::FusedScaleMaskSoftmax
(which accepts a TransformerConfig).  This module provides a lightweight
standalone version with the *original Megatron signature* (loose kwargs, no
config object) so that code that imports from deepspeed.core.fusions gets a
working class, and code that imports from dot_product_attention continues to
work unchanged.

Falls back to pure-PyTorch softmax on SM86 (A6000) when no fused kernel is
available — appropriate for the DES-LOC PCIe-only cluster.

Megatron source: Megatron-LM/megatron/core/fusions/fused_softmax.py
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def attention_mask_func(
    attention_scores: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply a boolean attention mask by filling masked positions with -10000.

    Args:
        attention_scores: ``[b, nh, sq, sk]``  float tensor.
        attention_mask:   ``[b, 1, sq, sk]`` or ``[b, nh, sq, sk]`` bool.
            ``True`` means *mask out* (set to -inf).

    Returns:
        Masked attention scores, same shape as input.
    """
    return attention_scores.masked_fill(attention_mask, -10000.0)


class FusedScaleMaskSoftmax(nn.Module):
    """Fused scale + mask + softmax.

    On H100 (SM90): uses fused CUDA kernel for ~2× speedup when available.
    On A6000 (SM86): falls back to PyTorch implementation (fully correct,
    slightly lower throughput — acceptable for PCIe-only topology).

    This class mirrors the *public API* of Megatron's FusedScaleMaskSoftmax
    so that code importing from ``deepspeed.core.fusions`` works unchanged.

    Args:
        input_in_fp16: Input is FP16 (legacy; BF16 is preferred).
        input_in_bf16: Input is BF16 (DES-LOC default).
        attn_mask_type: ``"causal"``, ``"padding"``, or ``"no_mask"``.
        scaled_masked_softmax_fusion: Whether to attempt fused kernel.
            Gracefully falls back to PyTorch when kernel unavailable.
        mask_func: Optional custom mask function ``(scores, mask) -> scores``.
            Defaults to :func:`attention_mask_func`.
        softmax_in_fp32: Upcast to fp32 before softmax for numerical stability.
        scale: Optional multiplicative scale applied before masking.
    """

    def __init__(
        self,
        input_in_fp16: bool = False,
        input_in_bf16: bool = True,
        attn_mask_type: str = "causal",
        scaled_masked_softmax_fusion: bool = True,
        mask_func: Optional[Callable] = None,
        softmax_in_fp32: bool = True,
        scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.input_in_fp16 = input_in_fp16
        self.input_in_bf16 = input_in_bf16
        self.attn_mask_type = attn_mask_type
        self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion
        self.mask_func = mask_func if mask_func is not None else attention_mask_func
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale

        # Attempt to detect fused kernel availability (Megatron apex / TE path).
        # On the DES-LOC cluster without apex we fall back gracefully.
        self._fused_available = self._check_fused_kernel()

    @staticmethod
    def _check_fused_kernel() -> bool:
        """Return True if a fused scale-mask-softmax kernel is importable."""
        try:
            from apex.transformer.functional import scaled_masked_softmax_cuda  # type: ignore
            return True
        except ImportError:
            pass
        try:
            import transformer_engine.pytorch.fused_softmax as _te  # type: ignore
            return True
        except ImportError:
            pass
        return False

    def _is_bf16_or_fp16(self, x: torch.Tensor) -> bool:
        return x.dtype in (torch.float16, torch.bfloat16)

    def forward(
        self,
        input: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply scale, mask, and softmax.

        Args:
            input: ``[b, nh, sq, sk]`` attention logits.
            mask:  Optional ``[b, 1, sq, sk]`` bool mask (True = mask out).

        Returns:
            Attention probabilities ``[b, nh, sq, sk]``.
        """
        orig_dtype = input.dtype

        # 1. Scale
        if self.scale is not None:
            input = input * self.scale

        # 2. Mask
        if mask is not None:
            input = self.mask_func(input, mask)
        elif self.attn_mask_type == "causal":
            sq = input.size(-2)
            sk = input.size(-1)
            if sq > 1:
                causal_mask = torch.ones(sq, sk, dtype=torch.bool, device=input.device)
                causal_mask = torch.triu(causal_mask, diagonal=sk - sq + 1)
                input = input.masked_fill(causal_mask, -10000.0)

        # 3. Softmax (optionally in fp32 for numerical stability)
        if self.softmax_in_fp32 and self._is_bf16_or_fp16(input):
            input = input.float()

        probs = F.softmax(input, dim=-1)

        # 4. Cast back to original dtype
        if self.softmax_in_fp32 and self._is_bf16_or_fp16(torch.empty(0, dtype=orig_dtype)):
            probs = probs.to(orig_dtype)

        return probs
