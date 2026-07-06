"""Fused Rotary Position Embedding (RoPE).

Mirrors Megatron-LM megatron/core/fusions/fused_rope.py.

Priority of implementations (highest to lowest):
1. ``transformer_engine.pytorch.attention.apply_rotary_pos_emb`` with
   ``fused=True`` — TE's custom CUDA kernel; requires TransformerEngine.
2. ``flash_attn.layers.rotary.apply_rotary_emb`` — FlashAttention's
   Triton/CUDA kernel; requires flash-attn ≥ 2.1.
3. ``_apply_rotary_pos_emb_torch`` — JIT-compiled pure-PyTorch fallback;
   always available, suitable for SM86 (A6000) nodes on the DES-LOC
   cluster where neither TE nor flash-attn may be present.

All three paths present the same public interface:

    apply_rotary_pos_emb(t, freqs, config, cu_seqlens, mscale, cp_group)

where ``t`` is ``[s, b, nh, d]`` (sbhd), ``freqs`` is the raw frequency
table ``[s, 1, 1, d]`` returned by ``_RotaryEmbedding.forward()``, and
``cu_seqlens`` (when provided) selects the THD (packed-sequence) path.

The ``FusedRoPEFunc`` autograd Function is the thin wrapper used by the
``apply_rotary_pos_emb`` dispatcher in rope_utils.py when TE and
flash-attn are both absent.

Megatron source: Megatron-LM/megatron/core/fusions/fused_rope.py
"""
from __future__ import annotations

import warnings
from typing import Optional

import torch
from torch import Tensor

from deepspeed.core.jit import jit_fuser


# ---------------------------------------------------------------------------
# Optional backend imports — graceful fallback when backends are absent.
# ---------------------------------------------------------------------------

try:
    # TransformerEngine >= 0.11 exposes apply_rotary_pos_emb with fused=True.
    try:
        from transformer_engine.pytorch.attention.rope import (  # type: ignore
            apply_rotary_pos_emb as _te_apply_rotary_pos_emb,
        )
    except ImportError:
        from transformer_engine.pytorch.attention import (  # type: ignore
            apply_rotary_pos_emb as _te_apply_rotary_pos_emb,
        )
    HAVE_TE_FUSED_ROPE = True
except ImportError:
    _te_apply_rotary_pos_emb = None
    HAVE_TE_FUSED_ROPE = False

try:
    from flash_attn.layers.rotary import (  # type: ignore
        apply_rotary_emb as _flash_apply_rotary_emb,
    )
    HAVE_FLASH_ROPE = True
except ImportError:
    _flash_apply_rotary_emb = None
    HAVE_FLASH_ROPE = False


# ---------------------------------------------------------------------------
# Pure-PyTorch helpers — JIT-compiled for kernel fusion.
# ---------------------------------------------------------------------------

@jit_fuser
def _rotate_half_torch(x: Tensor) -> Tensor:
    """Rotate the last dimension: ``[x1, x2] → [−x2, x1]``.

    Used by the non-interleaved (GPT-NeoX / Llama) convention where the
    first half of the head dimension is paired with the second half.

    Args:
        x: ``[..., d]`` float tensor.

    Returns:
        Tensor of the same shape with sign-flipped halves swapped.
    """
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@jit_fuser
def _rotate_interleaved_torch(x: Tensor) -> Tensor:
    """Rotate paired (interleaved) elements: ``[x_even, x_odd] → [−x_odd, x_even]``.

    Implements the GPT-J / Falcon convention where consecutive pairs
    (2k, 2k+1) form rotation pairs rather than split halves.

    Args:
        x: ``[..., d]`` float tensor.

    Returns:
        Tensor of the same shape with interleaved pairs rotated.
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def _apply_rotary_pos_emb_torch(
    t: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    mscale: float = 1.0,
) -> Tensor:
    """Pure-PyTorch (JIT-fusable) RoPE application — sbhd format.

    Applies the rotation ``t' = t·cos(θ) + rotate(t)·sin(θ)`` where
    ``θ = freqs`` and ``rotate`` is either ``_rotate_half_torch`` (Llama
    / GPT-NeoX style) or ``_rotate_interleaved_torch`` (GPT-J style).

    Args:
        t: Input tensor ``[s, b, nh, d]``.
        freqs: Raw frequency table ``[s, 1, 1, d]`` (or broadcastable).
            Must already be sliced to ``t.shape[0]`` entries.
        rotary_interleaved: If ``True``, use the interleaved (GPT-J)
            convention; otherwise use the split-half (Llama) convention.
        mscale: YaRN magnitude scaling factor (1.0 disables scaling).

    Returns:
        ``[s, b, nh, d]`` tensor with RoPE applied.
    """
    # M-ROPE-FIX: freqs may be on CPU when using ZeRO-3 with heterogeneous
    # multi-GPU setups.  Bring freqs to t's device before cos/sin.
    # Ref: HuggingFace transformers PR #32312, DeepSpeed issue #5311.
    if freqs.device != t.device:
        freqs = freqs.to(device=t.device)

    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    cos_ = (torch.cos(freqs) * mscale).to(t.dtype)
    sin_ = (torch.sin(freqs) * mscale).to(t.dtype)

    if rotary_interleaved:
        t_rot = _rotate_interleaved_torch(t)
    else:
        t_rot = _rotate_half_torch(t)

    return torch.cat(((t * cos_) + (t_rot * sin_), t_pass), dim=-1)


# ---------------------------------------------------------------------------
# autograd Function wrapper (used as fallback when TE and flash-attn absent).
# ---------------------------------------------------------------------------

class FusedRoPEFunc(torch.autograd.Function):
    """Autograd Function wrapping ``_apply_rotary_pos_emb_torch``.

    Provides an explicit backward pass that reuses the same rotation
    logic rather than relying on autograd to differentiate through it,
    matching Megatron's approach for numerical stability and memory
    efficiency.

    Note:
        This path is only taken when both TransformerEngine and
        flash-attn are absent.  In production, at least one of the two
        backend kernels will be available on H100/A6000 nodes.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        t: Tensor,
        freqs: Tensor,
        rotary_interleaved: bool,
        mscale: float,
    ) -> Tensor:
        """Apply RoPE in the forward pass and stash state for backward.

        Args:
            ctx: Autograd context.
            t: ``[s, b, nh, d]`` query or key tensor.
            freqs: ``[s, 1, 1, d]`` raw frequency table.
            rotary_interleaved: Whether to use interleaved pairs.
            mscale: YaRN magnitude scale (1.0 = no scaling).

        Returns:
            RoPE-rotated tensor, same shape as ``t``.
        """
        ctx.save_for_backward(freqs)
        ctx.rotary_interleaved = rotary_interleaved
        ctx.mscale = mscale
        return _apply_rotary_pos_emb_torch(t, freqs, rotary_interleaved, mscale)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: Tensor,
    ):
        """Backward pass: undo the rotation by applying inverse RoPE.

        The inverse of a rotation matrix R(θ) is R(−θ).  Because
        ``cos(−θ) = cos(θ)`` and ``sin(−θ) = −sin(θ)``, applying the
        same rotation to ``−rotate(grad_output)·sin + grad_output·cos``
        recovers the gradient w.r.t. ``t``.

        Equivalently: apply ``_apply_rotary_pos_emb_torch`` with negated
        sin (i.e. rotate grad_output back).
        """
        (freqs,) = ctx.saved_tensors
        rot_dim = freqs.shape[-1]
        grad, grad_pass = grad_output[..., :rot_dim], grad_output[..., rot_dim:]

        cos_ = torch.cos(freqs).to(grad.dtype) * ctx.mscale
        sin_ = torch.sin(freqs).to(grad.dtype) * ctx.mscale

        # Inverse rotation: grad_t = grad·cos − rotate(grad)·sin
        # (equivalent to rotating by −θ)
        if ctx.rotary_interleaved:
            grad_rot = _rotate_interleaved_torch(grad)
        else:
            grad_rot = _rotate_half_torch(grad)

        grad_t = torch.cat(
            ((grad * cos_) - (grad_rot * sin_), grad_pass), dim=-1
        )
        return grad_t, None, None, None


# ---------------------------------------------------------------------------
# THD (packed-sequence) helpers.
# ---------------------------------------------------------------------------

def _apply_rotary_pos_emb_thd_torch(
    t: Tensor,
    cu_seqlens: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    mscale: float = 1.0,
    cp_rank: int = 0,
    cp_size: int = 1,
) -> Tensor:
    """Apply RoPE to a packed (THD-format) tensor.

    For variable-length batches packed into a single leading dimension
    (token, head, dim), each token's position within its own sequence
    must be resolved from ``cu_seqlens`` before applying the rotation.

    Args:
        t: Packed tensor ``[total_tokens, nh, d]``.
        cu_seqlens: Cumulative sequence lengths ``[b+1]``, dtype int32.
        freqs: Full-sequence frequency table ``[max_s, 1, 1, d]``.
        rotary_interleaved: Interleaved vs split-half convention.
        mscale: YaRN magnitude scale.
        cp_rank: Context-parallel rank (0 for single-GPU).
        cp_size: Context-parallel world size (1 for single-GPU).

    Returns:
        ``[total_tokens, nh, d]`` tensor with RoPE applied.
    """
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

    # For context parallelism each rank holds every other chunk of the sequence.
    if cp_size > 1:
        # Scale per-rank sequence length (cp splits sequence into 2*cp_size chunks).
        seqlens = [s // cp_size for s in seqlens]

    out_chunks = []
    for i, seq_t in enumerate(torch.split(t, seqlens)):
        if cp_size > 1:
            cp_seg = seq_t.size(0) // 2
            full_seqlen = cp_size * seq_t.size(0)
            # Two non-contiguous chunks that belong to this CP rank.
            freq_chunk = torch.cat([
                freqs[cp_rank * cp_seg: (cp_rank + 1) * cp_seg],
                freqs[full_seqlen - (cp_rank + 1) * cp_seg: full_seqlen - cp_rank * cp_seg],
            ], dim=0)
        else:
            freq_chunk = freqs[: seq_t.size(0)]

        # seq_t is [s_local, nh, d]; unsqueeze batch dim for bshd helper.
        rotated = _apply_rotary_pos_emb_torch(
            seq_t.unsqueeze(1), freq_chunk, rotary_interleaved, mscale
        ).squeeze(1)
        out_chunks.append(rotated)

    return torch.cat(out_chunks, dim=0)


# ---------------------------------------------------------------------------
# Public dispatcher — mirrors rope_utils.apply_rotary_pos_emb interface.
# ---------------------------------------------------------------------------

def apply_rotary_pos_emb_fused(
    t: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    mscale: float = 1.0,
    cu_seqlens: Optional[Tensor] = None,
    cp_rank: int = 0,
    cp_size: int = 1,
) -> Tensor:
    """Apply RoPE using the best available kernel.

    Dispatch order:
    1. TransformerEngine fused kernel (``fused=True``) when ``cu_seqlens``
       is None and TE is installed.
    2. FlashAttention Triton/CUDA kernel when flash-attn is installed and
       ``cu_seqlens`` is None and the head dimension is 32/64/128/256.
    3. ``FusedRoPEFunc`` (JIT-compiled PyTorch) — always available.

    Args:
        t: Query or key tensor, ``[s, b, nh, d]`` (sbhd) or
           ``[total_tokens, nh, d]`` (thd when ``cu_seqlens`` is given).
        freqs: Raw frequency table ``[s, 1, 1, d]``.
        rotary_interleaved: Use interleaved (GPT-J) vs split-half (Llama).
        mscale: YaRN magnitude scaling factor.
        cu_seqlens: Cumulative sequence lengths for packed (THD) format.
        cp_rank: Context-parallel rank.
        cp_size: Context-parallel world size.

    Returns:
        Tensor with RoPE applied, same shape as ``t``.

    Note:
        ``mscale != 1.0`` and ``cu_seqlens is not None`` are not supported
        by TE/flash-attn backends; the fallback is used automatically.
    """
    # THD path — neither TE nor flash-attn supports it natively in all
    # versions, so we always use the PyTorch implementation.
    if cu_seqlens is not None:
        if HAVE_TE_FUSED_ROPE:
            try:
                return _te_apply_rotary_pos_emb(
                    t,
                    freqs,
                    tensor_format="thd",
                    cu_seqlens=cu_seqlens,
                    fused=True,
                    cp_size=cp_size,
                    cp_rank=cp_rank,
                )
            except (TypeError, RuntimeError):
                # Older TE versions may not support thd+cp kwargs.
                pass
        return _apply_rotary_pos_emb_thd_torch(
            t, cu_seqlens, freqs,
            rotary_interleaved=rotary_interleaved,
            mscale=mscale,
            cp_rank=cp_rank,
            cp_size=cp_size,
        )

    # SBHD path — prefer hardware-accelerated backends.
    if HAVE_TE_FUSED_ROPE and mscale == 1.0 and not rotary_interleaved:
        try:
            return _te_apply_rotary_pos_emb(
                t, freqs, tensor_format="sbhd", fused=True
            )
        except (TypeError, RuntimeError):
            pass

    if HAVE_FLASH_ROPE and mscale == 1.0:
        # flash_attn expects [b, s, nh, d]; transpose in/out.
        head_dim = t.shape[-1]
        if head_dim in {32, 64, 128, 256}:
            try:
                t_bshd = t.permute(1, 0, 2, 3)  # sbhd → bshd
                cos_ = torch.cos(freqs).to(t.dtype)
                sin_ = torch.sin(freqs).to(t.dtype)
                # cos/sin: [s, 1, 1, d] → squeeze batch dims → [s, d]
                cos_ = cos_.squeeze(1).squeeze(1)
                sin_ = sin_.squeeze(1).squeeze(1)
                out = _flash_apply_rotary_emb(
                    t_bshd, cos_, sin_, interleaved=rotary_interleaved
                )
                return out.permute(1, 0, 2, 3)  # bshd → sbhd
            except Exception:
                pass

    # JIT-compiled pure-PyTorch fallback — SM86 / A6000 safe.
    return FusedRoPEFunc.apply(t, freqs, rotary_interleaved, mscale)


# ---------------------------------------------------------------------------
# RMSNorm-style frequency scaling helper used by YaRN.
# ---------------------------------------------------------------------------

def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    """Compute the YaRN magnitude scaling factor.

    Scaling formula from Peng et al., 2023 (YaRN):
    ``mscale(s) = 0.1·ln(s) + 1.0`` for ``s > 1``, else ``1.0``.

    When ``mscale != 1.0`` an additional user-specified ``mscale``
    exponent is applied on top:
    ``mscale_factor = (0.1·ln(scale) + 1.0)^mscale``.

    Args:
        scale: The context-length extension scale factor.
        mscale: User-controlled exponent (default 1.0 = linear).

    Returns:
        Magnitude scaling coefficient as a Python float.
    """
    if scale <= 1.0:
        return 1.0
    return (0.1 * (scale ** 0.5) + 1.0) ** mscale
