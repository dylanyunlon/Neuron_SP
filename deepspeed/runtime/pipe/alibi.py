# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M716: DES-LOC — BLOOM ALiBi Position Encoding for heterogeneous pipeline
#
# Migrates BLOOM/Megatron-DeepSpeed ALiBi (Attention with Linear Biases)
# into the DES-LOC pipeline. Key features:
#   - Fixed per-head slopes: 2^(-8/n_heads) series (no learnable params)
#   - Pipeline stage offset: ALiBi bias matrix shifted by position_offset so
#     each stage uses correct absolute token positions
#   - Heterogeneous dtype: bias computed in FP32, cast to device precision
#     (FP16 for A6000, BF16 for H100) before addition to attention scores
#   - Non-power-of-2 head count supported via recursive slope construction
#
# Usage in a stage's transformer block:
#
#   from deepspeed.runtime.pipe.alibi import ALiBiEmbedding, build_alibi_bias
#
#   self.alibi_emb = ALiBiEmbedding(
#       n_heads=num_attention_heads,
#       max_seq_len=max_seq_len,
#       pipe_dtype=torch.float16,   # FP16 for A6000; torch.bfloat16 for H100
#   )
#
#   # In forward (position_offset comes from pipeline activation):
#   alibi_bias = build_alibi_bias(self.alibi_emb, seq_len_q, seq_len_k,
#                                  position_offset, device, dtype)
#   # alibi_bias shape: (n_heads, seq_len_q, seq_len_k)
#   # Add to attention scores before softmax:
#   attention_scores = attention_scores + alibi_bias
#
# References:
#   - Press et al. 2022: https://arxiv.org/abs/2108.12409
#   - BLOOM Megatron-DeepSpeed: references/pretrain_frameworks/bloom_megatron_ds/transformer.py
#   - DES-LOC M686 (Rotary): deepspeed/runtime/pipe/rotary.py
# ---------------------------------------------------------------------------

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

__all__ = [
    "ALiBiEmbedding",
    "build_alibi_bias",
    "get_alibi_slopes",
    "DTYPE_A6000",
    "DTYPE_H100",
]

# DES-LOC heterogeneous dtype constants (mirrors rotary.py)
DTYPE_A6000 = torch.float16    # A6000 uses FP16
DTYPE_H100  = torch.bfloat16   # H100 uses BF16


# ---------------------------------------------------------------------------
# Slope computation (BLOOM formula)
# ---------------------------------------------------------------------------

def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """Compute per-head ALiBi slopes as defined in Press et al. 2022.

    For n_heads that is a power of 2:
        slopes_i = 2^(-(8 / n_heads) * (i+1))  for i in [0, n_heads)

    For non-power-of-2 n_heads, slopes are constructed recursively:
        use slopes for closest lower power-of-2, then fill the rest by
        taking every-other slope from the next power-of-2 set.

    This matches BLOOM's implementation in bloom_megatron_ds/transformer.py.

    Args:
        n_heads: Number of attention heads.

    Returns:
        Tensor of shape (n_heads,) with dtype float32.
    """
    def _slopes_power_of_2(n: int):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    def _slopes(n: int):
        if math.log2(n).is_integer():
            return _slopes_power_of_2(n)
        closest_pow2 = 2 ** math.floor(math.log2(n))
        return _slopes_power_of_2(closest_pow2) + _slopes(2 * closest_pow2)[0::2][:n - closest_pow2]

    return torch.tensor(_slopes(n_heads), dtype=torch.float32)


# ---------------------------------------------------------------------------
# ALiBiEmbedding module
# ---------------------------------------------------------------------------

class ALiBiEmbedding(nn.Module):
    """ALiBi (Attention with Linear Biases) position encoding for DES-LOC pipeline.

    Stores fixed per-head slopes and provides a method to compute the bias
    matrix for a given (seq_len_q, seq_len_k, position_offset) triple.

    No learnable parameters → zero VRAM overhead during pipeline partitioning.

    The bias for head h at query position i, key position j (absolute) is:
        bias[h, i, j] = slope[h] * (j - i)
    which penalises attending to distant tokens linearly.

    In pipeline-parallel inference the position_offset shifts both query and
    key positions by the cumulative token count of preceding stages, so
    relative distances remain correct across stage boundaries.

    Args:
        n_heads (int): Total number of attention heads.
        max_seq_len (int): Maximum sequence length; used to pre-compute and
            cache the relative-position index matrix.
        pipe_dtype (torch.dtype): Compute dtype — FP16 for A6000, BF16 for H100.
            The bias is always computed in FP32 internally and cast here.
    """

    def __init__(
        self,
        n_heads: int,
        max_seq_len: int = 2048,
        pipe_dtype: torch.dtype = torch.float32,
    ) -> None:
        # DES-LOC M716: BLOOM ALiBi for pipeline stages
        super().__init__()
        self.n_heads     = n_heads
        self.max_seq_len = max_seq_len
        self.pipe_dtype  = pipe_dtype

        # Slopes: (n_heads,) — not a trainable parameter
        slopes = get_alibi_slopes(n_heads)          # (n_heads,)
        self.register_buffer("slopes", slopes, persistent=False)

        print(
            f"[M716-ALiBiEmbedding] n_heads={n_heads} max_seq_len={max_seq_len} "
            f"pipe_dtype={pipe_dtype} "
            f"(DES-LOC heterogeneous pipeline ALiBi)"
        )

    # ------------------------------------------------------------------
    # Bias computation
    # ------------------------------------------------------------------

    def get_bias(
        self,
        seq_len_q: int,
        seq_len_k: int,
        position_offset: int = 0,
        device: torch.device | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Compute ALiBi bias tensor for the given sequence lengths.

        Absolute positions are computed using position_offset so that each
        pipeline stage applies correct relative distances even though it only
        sees a partial sequence window.

        The bias formula (for head h, query token at absolute position i,
        key token at absolute position j):
            bias[h, i, j] = slope[h] * (j - i)

        Since j >= 0 and i >= position_offset, this is always <= 0 for
        causal attention (where we mask future tokens), which is the
        desired behaviour: nearby keys receive a small negative bias and
        far keys receive a larger negative bias.

        Args:
            seq_len_q: Number of query tokens in this micro-batch.
            seq_len_k: Number of key tokens in this micro-batch.
            position_offset: Absolute position of the first query token
                             (cumulative tokens from preceding pipeline stages).
            device: Target device; defaults to slopes device.
            out_dtype: Output dtype; defaults to self.pipe_dtype.

        Returns:
            Bias tensor of shape (n_heads, seq_len_q, seq_len_k) in out_dtype.
        """
        # DES-LOC M716: compute ALiBi bias with pipeline position offset
        if device is None:
            device = self.slopes.device
        if out_dtype is None:
            out_dtype = self.pipe_dtype

        # Absolute positions for queries and keys
        # q_pos: absolute positions of query tokens [position_offset, ..., position_offset + seq_len_q)
        # k_pos: absolute positions of key tokens   [0, ..., seq_len_k)
        q_pos = torch.arange(
            position_offset, position_offset + seq_len_q,
            dtype=torch.float32, device=device,
        )  # (seq_len_q,)
        k_pos = torch.arange(
            seq_len_k,
            dtype=torch.float32, device=device,
        )  # (seq_len_k,)

        # Relative distance matrix: key_pos - query_pos (negative for causal)
        # Shape: (seq_len_q, seq_len_k)
        rel_pos = k_pos.unsqueeze(0) - q_pos.unsqueeze(1)   # (seq_len_q, seq_len_k)

        # Apply per-head slopes: (n_heads, 1, 1) * (1, seq_len_q, seq_len_k)
        # → (n_heads, seq_len_q, seq_len_k) in FP32
        bias = self.slopes.view(self.n_heads, 1, 1) * rel_pos.unsqueeze(0)

        # Cast to target dtype (FP16 for A6000, BF16 for H100)
        return bias.to(out_dtype)

    def forward(
        self,
        seq_len_q: int,
        seq_len_k: int,
        position_offset: int = 0,
        device: torch.device | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Alias of get_bias for nn.Module forward compatibility."""
        return self.get_bias(
            seq_len_q=seq_len_q,
            seq_len_k=seq_len_k,
            position_offset=position_offset,
            device=device,
            out_dtype=out_dtype,
        )


# ---------------------------------------------------------------------------
# Convenience wrapper for pipeline stages
# ---------------------------------------------------------------------------

def build_alibi_bias(
    alibi_emb: ALiBiEmbedding,
    seq_len_q: int,
    seq_len_k: int,
    position_offset: int = 0,
    device: torch.device | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build ALiBi bias for a pipeline stage, respecting the position offset.

    Drop-in helper for transformer blocks that have an ALiBiEmbedding attached.
    The position_offset is the cumulative token count of all previous pipeline
    stages, ensuring correct relative distances across stage boundaries even
    when each stage sees only a partial sequence window.

    Args:
        alibi_emb: An ALiBiEmbedding module attached to the stage.
        seq_len_q: Number of query tokens in this micro-batch.
        seq_len_k: Number of key tokens in this micro-batch (= seq_len_q for
                   self-attention without KV cache).
        position_offset: Absolute position of the first query token (from
                         pipeline activation, mirrors rotary position_offset).
        device: Target device for the bias tensor.
        out_dtype: Output dtype; defaults to alibi_emb.pipe_dtype.

    Returns:
        Bias tensor of shape (n_heads, seq_len_q, seq_len_k).
        Add directly to attention scores before softmax.
    """
    # DES-LOC M716: route to ALiBiEmbedding with pipeline position offset
    return alibi_emb.get_bias(
        seq_len_q=seq_len_q,
        seq_len_k=seq_len_k,
        position_offset=position_offset,
        device=device,
        out_dtype=out_dtype,
    )
