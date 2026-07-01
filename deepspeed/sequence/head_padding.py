# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Dummy-head padding for non-divisible tensor/sequence parallelism.

When num_attention_heads % parallel_size != 0, standard TP/SP
frameworks (Megatron, DeepSpeed-Ulysses, vLLM) raise ValueError.

This module implements the "Dummy Head" approach from:
  - 360-LLaMA-Factory (arxiv:2505.22296) Section 4 "Dummy Head Ulysses"
  - sglang PR #6771: zero-padding for non-divisible heads
  - DistFlashAttn (arxiv:2310.03294): analysis of padding waste

The idea: pad num_heads to the next multiple of parallel_size by
adding zero-initialized "dummy" heads. After attention, strip the
dummy heads. The dummy heads contribute zero to the output because
their Q/K/V projections are initialized to zero.

For 32 heads ÷ 5 GPUs: pad to 35 heads (3 dummy), 8.6% compute waste.
For 32 heads ÷ 3 GPUs: pad to 33 heads (1 dummy), 3.1% compute waste.

References:
  - vLLM Issue #11797: https://github.com/vllm-project/vllm/issues/11797
  - vLLM Issue #596:   https://github.com/vllm-project/vllm/issues/596
  - sglang PR #6771:   https://github.com/sgl-project/sglang/pull/6771
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def compute_padded_heads(
    num_heads: int,
    num_kv_heads: int,
    parallel_size: int,
) -> tuple[int, int, int, int]:
    """Compute padded head counts for non-divisible TP/SP.

    Returns:
        (padded_heads, padded_kv_heads, pad_heads, pad_kv_heads)
    """
    if num_heads % parallel_size == 0 and num_kv_heads % parallel_size == 0:
        return num_heads, num_kv_heads, 0, 0

    padded_heads = math.ceil(num_heads / parallel_size) * parallel_size
    padded_kv_heads = math.ceil(num_kv_heads / parallel_size) * parallel_size
    return (
        padded_heads,
        padded_kv_heads,
        padded_heads - num_heads,
        padded_kv_heads - num_kv_heads,
    )


class PaddedQKVProjection(nn.Module):
    """Q/K/V projection with dummy-head zero padding.

    Wraps existing q_proj, k_proj, v_proj linear layers and pads
    their output with zeros for dummy heads. The padding is applied
    *after* the linear projection, so no extra parameters are added.

    Args:
        q_proj: Original query projection (dim → n_heads * head_dim)
        k_proj: Original key projection   (dim → n_kv_heads * head_dim)
        v_proj: Original value projection  (dim → n_kv_heads * head_dim)
        n_heads: Original number of attention heads
        n_kv_heads: Original number of KV heads (for GQA)
        padded_heads: Target padded head count
        padded_kv_heads: Target padded KV head count
        head_dim: Dimension per head
    """

    def __init__(
        self,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        n_heads: int,
        n_kv_heads: int,
        padded_heads: int,
        padded_kv_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.padded_heads = padded_heads
        self.padded_kv_heads = padded_kv_heads
        self.head_dim = head_dim
        self.pad_q = padded_heads - n_heads
        self.pad_kv = padded_kv_heads - n_kv_heads

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project and pad.

        Args:
            x: (B, T, dim)

        Returns:
            q: (B, T, padded_heads, head_dim)
            k: (B, T, padded_kv_heads, head_dim)
            v: (B, T, padded_kv_heads, head_dim)
        """
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        if self.pad_q > 0:
            q_pad = torch.zeros(
                B, T, self.pad_q, self.head_dim,
                dtype=q.dtype, device=q.device,
            )
            q = torch.cat([q, q_pad], dim=2)

        if self.pad_kv > 0:
            kv_pad = torch.zeros(
                B, T, self.pad_kv, self.head_dim,
                dtype=k.dtype, device=k.device,
            )
            k = torch.cat([k, kv_pad], dim=2)
            v = torch.cat([v, kv_pad], dim=2)

        return q, k, v


class PaddedOutputProjection(nn.Module):
    """Output projection that strips dummy heads before o_proj.

    Args:
        o_proj: Original output projection (n_heads * head_dim → dim)
        n_heads: Original (unpadded) number of heads
        head_dim: Dimension per head
    """

    def __init__(self, o_proj: nn.Linear, n_heads: int, head_dim: int):
        super().__init__()
        self.o_proj = o_proj
        self.n_heads = n_heads
        self.head_dim = head_dim

    def forward(self, attn_output: torch.Tensor) -> torch.Tensor:
        """Strip dummy heads and project.

        Args:
            attn_output: (B, T, padded_heads, head_dim) from attention

        Returns:
            (B, T, dim) after stripping padding and o_proj
        """
        # Strip dummy heads (keep only real heads)
        real = attn_output[:, :, : self.n_heads, :]
        B, T, _, _ = real.shape
        return self.o_proj(real.reshape(B, T, self.n_heads * self.head_dim))


def patch_attention_for_tp(
    model: nn.Module,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    tp_size: int,
) -> nn.Module:
    """Patch all attention layers in model with dummy-head padding.

    Walks the module tree, finds modules with q_proj/k_proj/v_proj/o_proj
    attributes, and wraps them with PaddedQKVProjection / PaddedOutputProjection.

    Args:
        model: The model to patch
        num_heads: Original number of attention heads
        num_kv_heads: Original number of KV heads
        head_dim: Head dimension
        tp_size: Tensor parallel size

    Returns:
        model with patched attention layers (in-place)
    """
    padded_heads, padded_kv_heads, pad_q, pad_kv = compute_padded_heads(
        num_heads, num_kv_heads, tp_size
    )

    if pad_q == 0 and pad_kv == 0:
        return model  # No padding needed

    for name, module in model.named_modules():
        if (
            hasattr(module, "q_proj")
            and hasattr(module, "k_proj")
            and hasattr(module, "v_proj")
            and hasattr(module, "o_proj")
        ):
            # Wrap QKV projections
            module._orig_q_proj = module.q_proj
            module._orig_k_proj = module.k_proj
            module._orig_v_proj = module.v_proj

            module.padded_qkv = PaddedQKVProjection(
                module.q_proj, module.k_proj, module.v_proj,
                num_heads, num_kv_heads,
                padded_heads, padded_kv_heads, head_dim,
            )

            # Wrap output projection
            module._orig_o_proj = module.o_proj
            module.padded_output = PaddedOutputProjection(
                module.o_proj, num_heads, head_dim,
            )

            # Update head counts on the module
            module.n_heads = padded_heads
            module.n_kv_heads = padded_kv_heads

    return model
