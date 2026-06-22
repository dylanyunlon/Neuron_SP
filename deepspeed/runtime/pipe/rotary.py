# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M686: DES-LOC — NeoX Rotary Position Embedding for heterogeneous pipeline
#
# Migrates EleutherAI GPT-NeoX RotaryEmbedding into the DES-LOC pipeline.
# Key features:
#   - Interleaved and non-interleaved rotation modes (matching NeoX)
#   - cos/sin cache with sequence position offset for pipeline stage correctness
#   - Heterogeneous dtype: A6000 → FP16, H100 → BF16, both cast-safe
#   - position_offset carried as pipeline activation so each stage knows its
#     absolute token positions (critical for multi-stage RoPE correctness)
#
# Usage in a stage's transformer block:
#
#   from deepspeed.runtime.pipe.rotary import NeoXRotaryEmbedding, apply_rotary_pipe
#
#   self.rotary_emb = NeoXRotaryEmbedding(
#       dim=head_dim,
#       max_seq_len=max_seq_len,
#       interleaved=False,          # set True to match NeoX interleaved mode
#       pipe_dtype=torch.float16,   # FP16 for A6000; torch.bfloat16 for H100
#   )
#
#   # In forward (position_offset comes from pipeline activation):
#   q, k = apply_rotary_pipe(self.rotary_emb, q, k, seq_len, position_offset)
#
# ---------------------------------------------------------------------------

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

__all__ = [
    "NeoXRotaryEmbedding",
    "apply_rotary_pipe",
    "DTYPE_A6000",
    "DTYPE_H100",
]

# DES-LOC heterogeneous dtype constants
DTYPE_A6000 = torch.float16    # A6000 uses FP16 rotary
DTYPE_H100  = torch.bfloat16   # H100 uses BF16 rotary


class NeoXRotaryEmbedding(nn.Module):
    """Rotary Position Embedding matching EleutherAI GPT-NeoX.

    Supports two rotation modes:
      - interleaved=False (default, "rotate-half"): rotate the last half of
        the head-dim features.  This matches NeoX's non-interleaved path and
        the majority of open-source RoPE implementations.
      - interleaved=True: interleave even/odd dimensions before rotation.
        Matches NeoX's neox_rotary_style=True path (used when
        rotary_interleaved=True in the config).

    The cos/sin cache is built lazily up to `max_seq_len` positions.  Passing
    `position_offset` (the absolute token offset of the first token in this
    stage's micro-batch) ensures correct positional encoding in pipeline
    parallel where each stage sees a different slice of the full sequence.

    Args:
        dim (int): Rotary dimension (typically head_dim or a fraction of it).
        max_seq_len (int): Maximum sequence length for cache pre-computation.
        base (float): Frequency base; NeoX default is 10000.
        interleaved (bool): Use interleaved rotation (NeoX rotary_interleaved).
        pipe_dtype (torch.dtype): Compute dtype — FP16 for A6000, BF16 for H100.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        interleaved: bool = False,
        pipe_dtype: torch.dtype = torch.float32,
    ) -> None:
        # DES-LOC M686: NeoX rotary for pipeline stages
        super().__init__()
        self.dim          = dim
        self.max_seq_len  = max_seq_len
        self.base         = base
        self.interleaved  = interleaved
        self.pipe_dtype   = pipe_dtype

        # Inverse frequencies — not a trainable parameter
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

        print(
            f"[M686-NeoXRotaryEmbedding] dim={dim} max_seq_len={max_seq_len} "
            f"base={base} interleaved={interleaved} pipe_dtype={pipe_dtype} "
            f"(DES-LOC heterogeneous pipeline rotary)"
        )

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _build_cache(self, seq_len: int) -> None:
        """Pre-compute cos/sin tables for positions 0..seq_len-1."""
        t     = torch.arange(seq_len, dtype=self.inv_freq.dtype,
                             device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)           # (seq_len, dim/2)

        if self.interleaved:
            # Interleaved: duplicate each frequency for paired dims (NeoX style)
            # emb[i, 2j]   = freqs[i, j]
            # emb[i, 2j+1] = freqs[i, j]
            emb = torch.stack([freqs, freqs], dim=-1).flatten(-2)  # (seq_len, dim)
        else:
            # Non-interleaved (rotate-half): cat freqs with itself
            emb = torch.cat([freqs, freqs], dim=-1)                # (seq_len, dim)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _ensure_cache(self, seq_len: int, offset: int) -> None:
        needed = seq_len + offset
        if needed > self.cos_cached.shape[0]:
            self._build_cache(needed)

    # ------------------------------------------------------------------
    # Rotation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """NeoX non-interleaved rotation: negate second half and swap."""
        h = x.shape[-1] // 2
        x1, x2 = x[..., :h], x[..., h:]
        return torch.cat([-x2, x1], dim=-1)

    @staticmethod
    def _rotate_interleaved(x: torch.Tensor) -> torch.Tensor:
        """NeoX interleaved rotation: negate odd-indexed features and shift."""
        x1 = x[..., ::2]   # even dims
        x2 = x[..., 1::2]  # odd dims
        # Interleave back: [-x2, x1] in paired layout
        out = torch.stack([-x2, x1], dim=-1).flatten(-2)
        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors.

        Args:
            q: Query tensor  (batch, n_heads, seq_len, head_dim).
            k: Key tensor    (batch, n_kv_heads, seq_len, head_dim).
            seq_len: Token count in this micro-batch.
            offset: Absolute position of the first token (pipeline stage offset).

        Returns:
            Rotated (q, k) in the original input dtype.
        """
        # DES-LOC M686: ensure cache covers offset+seq_len positions
        self._ensure_cache(seq_len, offset)

        # Slice cos/sin for this stage's positions
        cos = self.cos_cached[offset: offset + seq_len]   # (seq_len, dim)
        sin = self.sin_cached[offset: offset + seq_len]   # (seq_len, dim)

        # Broadcast over batch and head dims: (1, 1, seq_len, dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Cast to pipe_dtype for heterogeneous compute (FP16 / BF16)
        orig_dtype = q.dtype
        q_fp = q.to(self.pipe_dtype)
        k_fp = k.to(self.pipe_dtype)
        cos  = cos.to(self.pipe_dtype)
        sin  = sin.to(self.pipe_dtype)

        rotate_fn = self._rotate_interleaved if self.interleaved else self._rotate_half

        q_rot = q_fp * cos + rotate_fn(q_fp) * sin
        k_rot = k_fp * cos + rotate_fn(k_fp) * sin

        # Cast back to original dtype (preserves model weight dtype)
        return q_rot.to(orig_dtype), k_rot.to(orig_dtype)


# ---------------------------------------------------------------------------
# Convenience wrapper for pipeline stages
# ---------------------------------------------------------------------------

def apply_rotary_pipe(
    rotary_emb: NeoXRotaryEmbedding,
    q: torch.Tensor,
    k: torch.Tensor,
    seq_len: int,
    position_offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Thin wrapper that calls rotary_emb.forward with a pipeline position_offset.

    The position_offset is the cumulative token count of all previous pipeline
    stages, so each stage applies correct absolute positions even though it only
    sees a partial sequence.

    Args:
        rotary_emb: A NeoXRotaryEmbedding module attached to the stage.
        q: Query (batch, n_heads, seq_len, head_dim).
        k: Key   (batch, n_kv_heads, seq_len, head_dim).
        seq_len: Number of tokens in this micro-batch.
        position_offset: Absolute token index of the first token (from pipeline activation).

    Returns:
        Rotated (q, k).
    """
    # DES-LOC M686: route to NeoX rotary with pipeline position offset
    return rotary_emb(q, k, seq_len=seq_len, offset=position_offset)
