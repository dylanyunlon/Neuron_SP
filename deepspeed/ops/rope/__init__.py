# Copyright (c) 2026 Neuron_SP Project. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Addresses issue #23: fused RoPE for heterogeneous head counts

"""
deepspeed.ops.rope — Fused Rotary Position Embedding
=====================================================

Provides a single fused CUDA kernel for RoPE that:
  1. Takes Q [s, b, nh, d] and K [s, b, nkv, d] as inputs
  2. Handles GQA (different num_heads for Q vs K)
  3. Supports both interleaved (GPT-J) and non-interleaved (NeoX/Llama) styles
  4. Dispatches BF16/FP16/FP32 at runtime
  5. SM-specialised: SM 8.6 (A6000), SM 9.0 (H100), SM 12.0 (Blackwell)

Eliminates the 3 failed fix attempts (#11) by replacing the pure Python RoPE
with a fused kernel that has correct tensor semantics by construction.

Usage::

    from deepspeed.ops.rope import apply_fused_rope, apply_fused_rope_qk

    # Apply to Q and K separately:
    q_rot = apply_fused_rope(q, cos, sin, neox_style=True)
    k_rot = apply_fused_rope(k, cos, sin, neox_style=True)

    # Or apply to both Q and K in one call (handles GQA head counts):
    q_rot, k_rot = apply_fused_rope_qk(q, k, cos, sin, neox_style=True)

    # Cacheless mode (computes sin/cos on-the-fly, useful for very long seqs):
    q_rot = apply_fused_rope(q, base=10000.0, pos_offset=0, neox_style=True)
"""

from typing import Optional, Tuple

import torch


def _get_sm_version() -> int:
    """Return the SM version of the current CUDA device (e.g. 86, 90, 120)."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.major * 10 + props.minor


def _load_hetero_reduce():
    """Lazy-load the hetero_reduce C++ extension."""
    try:
        from deepspeed.ops.hetero_reduce import hetero_reduce_op
        return hetero_reduce_op
    except ImportError:
        # Fallback: try loading the extension directly.
        try:
            import hetero_reduce as ext
            return ext
        except ImportError:
            return None


def apply_fused_rope(
    input: torch.Tensor,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    neox_style: bool = True,
    base: float = 10000.0,
    pos_offset: int = 0,
) -> torch.Tensor:
    """
    Apply Rotary Position Embedding to a single tensor (Q or K).

    Supports two modes:
      1. Cached mode: cos and sin tensors provided (precomputed).
      2. Cacheless mode: cos=None, sin=None — computes sin/cos on-the-fly.

    Args:
        input: BF16 tensor [B, S, H, D] or contiguous view thereof.
        cos: FP32 tensor [S, D/2] (precomputed cosine cache), or None.
        sin: FP32 tensor [S, D/2] (precomputed sine cache), or None.
        neox_style: True for Llama/NeoX style, False for GPT-J interleaved.
        base: RoPE base frequency (only used in cacheless mode).
        pos_offset: Global position offset (only used in cacheless mode).

    Returns:
        BF16 tensor with same shape as input, with RoPE applied.
    """
    ext = _load_hetero_reduce()
    sm_version = _get_sm_version()

    # Validate input shape: [B, S, H, D]
    if input.dim() != 4:
        raise ValueError(
            f"apply_fused_rope: expected 4D input [B, S, H, D], got {input.dim()}D"
        )

    B, S, H, D = input.shape
    output = torch.empty_like(input)

    if ext is not None:
        # Use the fused CUDA kernel.
        if cos is not None and sin is not None:
            ext.fused_rope_hetero(
                output, input, cos, sin,
                B, S, H, D, neox_style, sm_version,
            )
        else:
            ext.fused_rope_cacheless(
                output, input,
                B, S, H, D, base, pos_offset, neox_style, sm_version,
            )
    else:
        # Pure Python fallback (correct reference implementation).
        output = _python_rope_fallback(input, cos, sin, neox_style, base, pos_offset)

    return output


def apply_fused_rope_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    neox_style: bool = True,
    base: float = 10000.0,
    pos_offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to both Q and K in one call.  Handles GQA where Q and K
    have different numbers of heads.

    Args:
        q: BF16 tensor [B, S, num_heads_q, D]
        k: BF16 tensor [B, S, num_heads_kv, D]
        cos: FP32 tensor [S, D/2] (or None for cacheless mode)
        sin: FP32 tensor [S, D/2] (or None for cacheless mode)
        neox_style: True for Llama/NeoX, False for GPT-J.
        base: RoPE base frequency (cacheless mode only).
        pos_offset: Global position offset (cacheless mode only).

    Returns:
        Tuple of (q_rotated, k_rotated), both BF16 with original shapes.
    """
    q_rot = apply_fused_rope(q, cos, sin, neox_style, base, pos_offset)
    k_rot = apply_fused_rope(k, cos, sin, neox_style, base, pos_offset)
    return q_rot, k_rot


def precompute_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    pos_offset: int = 0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos/sin cache tensors on GPU using the fused CUDA kernel.

    Args:
        seq_len: Number of sequence positions.
        head_dim: Full head dimension (cache has shape [seq_len, head_dim/2]).
        base: RoPE base frequency.
        pos_offset: Starting position index.
        device: CUDA device (defaults to current device).

    Returns:
        Tuple of (cos_cache, sin_cache), both FP32 [seq_len, head_dim/2].
    """
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())

    half_dim = head_dim // 2
    cos_cache = torch.empty(seq_len, half_dim, dtype=torch.float32, device=device)
    sin_cache = torch.empty(seq_len, half_dim, dtype=torch.float32, device=device)

    ext = _load_hetero_reduce()
    if ext is not None:
        sm_version = _get_sm_version()
        ext.rope_cache(cos_cache, sin_cache, seq_len, head_dim, base,
                       pos_offset, sm_version)
    else:
        # Python fallback.
        positions = torch.arange(pos_offset, pos_offset + seq_len,
                                 dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, dtype=torch.float32,
                                                  device=device) * 2.0 / head_dim))
        freqs = torch.outer(positions, inv_freq)
        cos_cache.copy_(freqs.cos())
        sin_cache.copy_(freqs.sin())

    return cos_cache, sin_cache


def _python_rope_fallback(
    x: torch.Tensor,
    cos: Optional[torch.Tensor],
    sin: Optional[torch.Tensor],
    neox_style: bool,
    base: float,
    pos_offset: int,
) -> torch.Tensor:
    """Pure Python RoPE — correct reference implementation for testing."""
    B, S, H, D = x.shape
    half_dim = D // 2

    # Compute cos/sin if not provided.
    if cos is None or sin is None:
        positions = torch.arange(pos_offset, pos_offset + S,
                                 dtype=torch.float32, device=x.device)
        inv_freq = 1.0 / (base ** (torch.arange(0, half_dim,
                                                  dtype=torch.float32,
                                                  device=x.device) * 2.0 / D))
        freqs = torch.outer(positions, inv_freq)
        cos = freqs.cos()  # [S, D/2]
        sin = freqs.sin()  # [S, D/2]

    x_float = x.float()
    output = torch.empty_like(x_float)

    # Reshape cos/sin for broadcasting: [1, S, 1, D/2]
    cos_r = cos.unsqueeze(0).unsqueeze(2)  # [1, S, 1, D/2]
    sin_r = sin.unsqueeze(0).unsqueeze(2)

    if neox_style:
        # NeoX/Llama: first half paired with second half
        x1 = x_float[..., :half_dim]
        x2 = x_float[..., half_dim:]
        output[..., :half_dim] = x1 * cos_r - x2 * sin_r
        output[..., half_dim:] = x1 * sin_r + x2 * cos_r
    else:
        # GPT-J: adjacent pairs [x0, y0, x1, y1, ...]
        x_pairs = x_float.view(B, S, H, half_dim, 2)
        x1 = x_pairs[..., 0]  # [B, S, H, D/2]
        x2 = x_pairs[..., 1]  # [B, S, H, D/2]
        r1 = x1 * cos_r - x2 * sin_r
        r2 = x1 * sin_r + x2 * cos_r
        output = torch.stack([r1, r2], dim=-1).view(B, S, H, D)

    return output.to(x.dtype)
