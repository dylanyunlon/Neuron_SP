"""Fused scale + mask + softmax for attention.

Mirrors Megatron-LM megatron/core/fusions/fused_softmax.py.

Provides three custom autograd ``Function``s that wrap apex/CUDA fused
kernels when available, plus a full ``FusedScaleMaskSoftmax`` nn.Module
with the same public API as Megatron's:

- ``ScaledUpperTriangMaskedSoftmax``: causal (upper-triangular) mask + scale.
- ``ScaledMaskedSoftmax``: arbitrary additive mask + scale.
- ``ScaledSoftmax``: scale only (no mask).
- ``FusedScaleMaskSoftmax``: dispatch wrapper used by dot-product attention.

On clusters without apex (e.g. DES-LOC A6000 nodes), all paths fall back to
``forward_torch_softmax`` which is pure PyTorch and numerically identical.

Megatron source: Megatron-LM/megatron/core/fusions/fused_softmax.py
"""
from __future__ import annotations

from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepspeed.core.transformer.enums import AttnMaskType
from deepspeed.core.transformer.utils import (
    get_default_causal_mask,
    get_sliding_window_causal_mask,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def attention_mask_func(
    attention_scores: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply a boolean attention mask by filling masked positions with -10000.

    Args:
        attention_scores: ``[b, nh, sq, sk]`` float tensor.
        attention_mask: ``[b, 1, sq, sk]`` or broadcastable bool tensor.
            ``True`` means *mask out*.

    Returns:
        Masked scores with same shape as ``attention_scores``.
    """
    return attention_scores.masked_fill(attention_mask, -10000.0)


# ---------------------------------------------------------------------------
# Custom autograd Functions — wrap apex CUDA kernels when available,
# fall back gracefully when the extension is absent.
# ---------------------------------------------------------------------------

class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
    """Scale + causal upper-triangular mask + softmax.

    Dispatches to ``scaled_upper_triang_masked_softmax_cuda`` (apex) when
    available; otherwise raises ImportError (the module-level availability
    check in ``FusedScaleMaskSoftmax.is_kernel_available`` prevents this
    path being taken when apex is absent).
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, scale: float) -> torch.Tensor:
        """Forward: scale, apply causal mask, softmax.

        Args:
            ctx: Autograd context.
            inputs: ``[attn_batches, sq, sk]`` float16/bfloat16 tensor.
            scale: Multiplicative scale applied before masking.

        Returns:
            Softmax probabilities, same shape as ``inputs``.
        """
        import scaled_upper_triang_masked_softmax_cuda  # type: ignore

        scale_t = torch.tensor([scale])
        softmax_results = scaled_upper_triang_masked_softmax_cuda.forward(
            inputs, scale_t[0]
        )
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads: torch.Tensor):
        """Backward: gradient through causal-masked softmax.

        Returns:
            ``(input_grads, None)`` — None for the scale scalar.
        """
        import scaled_upper_triang_masked_softmax_cuda  # type: ignore

        softmax_results, scale_t = ctx.saved_tensors
        input_grads = scaled_upper_triang_masked_softmax_cuda.backward(
            output_grads, softmax_results, scale_t[0]
        )
        return input_grads, None


class ScaledMaskedSoftmax(torch.autograd.Function):
    """Scale + arbitrary mask + softmax.

    Dispatches to ``scaled_masked_softmax_cuda`` (apex) when available.
    """

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        mask: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Forward: scale, apply mask, softmax.

        Args:
            ctx: Autograd context.
            inputs: ``[b, np, sq, sk]`` float16/bfloat16 tensor.
            mask: Additive mask broadcastable to ``inputs``.
            scale: Multiplicative scale.

        Returns:
            Softmax probabilities, same shape as ``inputs``.
        """
        import scaled_masked_softmax_cuda  # type: ignore

        scale_t = torch.tensor([scale])
        softmax_results = scaled_masked_softmax_cuda.forward(inputs, mask, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads: torch.Tensor):
        """Backward.

        Returns:
            ``(input_grads, None, None)`` — None for mask and scale.
        """
        import scaled_masked_softmax_cuda  # type: ignore

        softmax_results, scale_t = ctx.saved_tensors
        input_grads = scaled_masked_softmax_cuda.backward(
            output_grads, softmax_results, scale_t[0]
        )
        return input_grads, None, None


class ScaledSoftmax(torch.autograd.Function):
    """Scale + softmax (no mask).

    Dispatches to ``scaled_softmax_cuda`` (apex) when available.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, scale: float) -> torch.Tensor:
        """Forward: scale then softmax.

        Args:
            ctx: Autograd context.
            inputs: ``[b, np, sq, sk]`` float16/bfloat16 tensor.
            scale: Multiplicative scale.

        Returns:
            Softmax probabilities, same shape as ``inputs``.
        """
        import scaled_softmax_cuda  # type: ignore

        scale_t = torch.tensor([scale])
        softmax_results = scaled_softmax_cuda.forward(inputs, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads: torch.Tensor):
        """Backward.

        Returns:
            ``(input_grads, None)`` — None for scale.
        """
        import scaled_softmax_cuda  # type: ignore

        softmax_results, scale_t = ctx.saved_tensors
        input_grads = scaled_softmax_cuda.backward(
            output_grads, softmax_results, scale_t[0]
        )
        return input_grads, None


# ---------------------------------------------------------------------------
# Softmax-off-by-one (learnable denominator offset)
# ---------------------------------------------------------------------------

class SoftmaxOne(nn.Module):
    r"""Softmax-off-by-one: adds a learnable or fixed denominator offset.

    Introduced in https://www.evanmiller.org/attention-is-off-by-one.html
    Appends a "sink" token to the key dimension so the denominator includes
    an extra constant, then discards the sink from the output.

    Args:
        dim: Dimension over which softmax is computed (default ``-1``).
        denominator_offset: Fixed float or ``[np]`` learnable offset tensor.
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        denominator_offset: Union[torch.Tensor, float] = 1.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.denominator_offset = denominator_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply softmax-off-by-one.

        Args:
            x: ``[b, np, sq, sk]`` attention logits.

        Returns:
            Attention probabilities ``[b, np, sq, sk]``.
        """
        # sink: [np] → [1, np, 1, 1] → [b, np, sq, 1]
        sink = self.denominator_offset.reshape(1, -1, 1, 1).expand(
            x.size(0), -1, x.size(2), -1
        )
        # append sink along key dimension, softmax, remove sink
        qk = torch.cat([x, sink], dim=-1)
        return torch.softmax(qk, dim=-1)[..., :-1]


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class FusedScaleMaskSoftmax(nn.Module):
    """Fused scale + mask + softmax for attention logits.

    Mirrors the Megatron ``FusedScaleMaskSoftmax`` API exactly so callers
    can switch between implementations without code changes.

    Dispatch order:
    1. Apex fused CUDA kernel (``is_kernel_available`` returns ``True``).
    2. PyTorch fallback via ``forward_torch_softmax``.

    On the DES-LOC cluster the H100 tier may benefit from the fused kernel
    when apex is compiled; A6000 nodes always take the PyTorch path.

    Args:
        input_in_fp16: Input is FP16.
        input_in_bf16: Input is BF16 (DES-LOC default).
        attn_mask_type: ``AttnMaskType.causal``, ``.padding``, or ``.no_mask``.
        scaled_masked_softmax_fusion: Enable fused kernel dispatch when the
            constraints in ``is_kernel_available`` are satisfied.
        mask_func: Custom mask function ``(scores, mask) -> scores``.
            Defaults to :func:`attention_mask_func`.
        softmax_in_fp32: Upcast to fp32 before softmax.
        scale: Multiplicative scale applied before masking.
        window_size: Sliding-window attention window ``(left, right)``
            or ``None`` for full attention / causal.
    """

    def __init__(
        self,
        input_in_fp16: bool = False,
        input_in_bf16: bool = True,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        scaled_masked_softmax_fusion: bool = True,
        mask_func: Optional[Callable] = None,
        softmax_in_fp32: bool = True,
        scale: Optional[float] = None,
        window_size: Optional[tuple] = None,
    ) -> None:
        super().__init__()
        self.input_in_fp16 = input_in_fp16
        self.input_in_bf16 = input_in_bf16
        assert not (self.input_in_fp16 and self.input_in_bf16), (
            "FusedScaleMaskSoftmax: both fp16 and bf16 flags cannot be active simultaneously."
        )
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16
        self.attn_mask_type = attn_mask_type
        self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion
        self.mask_func = mask_func if mask_func is not None else attention_mask_func
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale
        self.window_size = window_size
        assert self.scale is None or softmax_in_fp32, (
            "FusedScaleMaskSoftmax: softmax must be in fp32 when a scale is provided."
        )

    def forward(
        self,
        input: torch.Tensor,
        mask: Optional[torch.Tensor],
        softmax_offset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply scale, mask, and softmax.

        When ``attn_mask_type`` is causal, the mask is generated internally
        and ``mask=None`` is valid.  A user-provided mask overrides the
        auto-generated one for non-causal attention.

        Args:
            input: ``[b, np, sq, sk]`` attention logits.
            mask: Optional ``[b, 1, sq, sk]`` boolean mask (``True`` = mask out).
                Pass ``None`` for causal attention (mask generated automatically).
            softmax_offset: Optional learnable denominator offset for
                softmax-off-by-one; enables :class:`SoftmaxOne` path.

        Returns:
            Attention probabilities ``[b, np, sq, sk]``.
        """
        assert input.dim() == 4, (
            f"FusedScaleMaskSoftmax.forward: expected 4-D input, got {input.dim()}-D"
        )

        if self.is_kernel_available(mask, *input.size()) and softmax_offset is None:
            return self.forward_fused_softmax(input, mask)
        else:
            return self.forward_torch_softmax(input, mask, softmax_offset)

    def is_kernel_available(
        self,
        mask: Optional[torch.Tensor],
        b: int,
        np: int,
        sq: int,
        sk: int,
    ) -> bool:
        """Check whether the apex fused CUDA kernel can handle this request.

        The kernel imposes constraints on data type, sequence lengths, and
        batch-head product divisibility.  Returns ``False`` on any mismatch
        so that ``forward_torch_softmax`` is used as the fallback.

        Args:
            mask: Optional attention mask (may be ``None`` for causal).
            b: Batch size.
            np: Number of attention heads per TP partition.
            sq: Query sequence length.
            sk: Key sequence length.

        Returns:
            ``True`` if all constraints are satisfied and apex is available.
        """
        attn_batches = b * np

        if (
            self.scaled_masked_softmax_fusion      # user opt-in
            and self.input_in_float16              # kernel requires fp16/bf16
            and 16 < sk <= 4096                   # key-length range
            and sq % 4 == 0                        # sq divisibility
            and sk % 4 == 0                        # sk divisibility
            and attn_batches % 4 == 0              # b*np divisibility
        ):
            try:
                batch_per_block = self.get_batch_per_block(sq, sk, b, np)
            except Exception:
                return False

            if self.attn_mask_type == AttnMaskType.causal:
                if attn_batches % batch_per_block == 0:
                    return True
            else:
                if sq % batch_per_block == 0:
                    return True
        return False

    def forward_fused_softmax(
        self,
        input: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run apex fused softmax kernels.

        Selects among ``ScaledUpperTriangMaskedSoftmax`` (causal),
        ``ScaledMaskedSoftmax`` (masked non-causal), and ``ScaledSoftmax``
        (unmasked non-causal) based on ``attn_mask_type`` and ``mask``.

        Args:
            input: ``[b, np, sq, sk]`` float16/bfloat16 logits.
            mask: Optional mask for non-causal paths.

        Returns:
            Softmax probabilities ``[b, np, sq, sk]``.
        """
        b, np, sq, sk = input.size()
        scale = self.scale if self.scale is not None else 1.0

        if self.attn_mask_type == AttnMaskType.causal:
            assert sq == sk, (
                "forward_fused_softmax: causal mask requires sq == sk "
                f"(got sq={sq}, sk={sk})"
            )
            # Fused causal kernel expects 3-D input.
            input = input.view(-1, sq, sk)
            probs = ScaledUpperTriangMaskedSoftmax.apply(input, scale)
            return probs.view(b, np, sq, sk)
        else:
            if mask is not None:
                return ScaledMaskedSoftmax.apply(input, mask, scale)
            else:
                return ScaledSoftmax.apply(input, scale)

    def forward_torch_softmax(
        self,
        input: torch.Tensor,
        mask: Optional[torch.Tensor],
        softmax_offset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pure-PyTorch fallback: scale, build/apply mask, softmax.

        Used on A6000 nodes (no apex) and whenever the fused kernel
        constraints are not met.

        Args:
            input: ``[b, np, sq, sk]`` attention logits.
            mask: Optional boolean mask; auto-generated for causal attention.
            softmax_offset: Optional ``[np]`` learnable denominator offset;
                enables the :class:`SoftmaxOne` variant.

        Returns:
            Attention probabilities ``[b, np, sq, sk]``.
        """
        if self.input_in_float16 and self.softmax_in_fp32:
            input = input.float()

        if self.scale is not None:
            input = input * self.scale

        sq, sk = input.size(2), input.size(3)

        # Build mask if not supplied.
        if self.window_size is not None:
            mask = get_sliding_window_causal_mask(sq, sk, self.window_size)
        elif self.attn_mask_type == AttnMaskType.causal and mask is None and sq > 1:
            # sq == 1: KV-cache decode step — no mask needed.
            assert sq == sk, (
                "forward_torch_softmax: causal mask requires sq == sk "
                f"(got sq={sq}, sk={sk})"
            )
            mask = get_default_causal_mask(sq)

        mask_output = self.mask_func(input, mask) if mask is not None else input

        if softmax_offset is None:
            softmax_fn = torch.nn.Softmax(dim=-1)
        else:
            softmax_fn = SoftmaxOne(-1, softmax_offset.to(input.device))

        probs = softmax_fn(mask_output)

        # Cast back to original float16/bfloat16.
        if self.input_in_float16 and self.softmax_in_fp32:
            if self.input_in_fp16:
                probs = probs.half()
            else:
                probs = probs.bfloat16()

        return probs

    @staticmethod
    def get_batch_per_block(sq: int, sk: int, b: int, np: int) -> int:
        """Query apex's batch-per-block kernel parameter.

        Used by ``is_kernel_available`` to verify divisibility constraints.

        Args:
            sq: Query sequence length.
            sk: Key sequence length.
            b: Batch size.
            np: Heads per TP rank.

        Returns:
            Batch-per-block value from the CUDA extension.

        Raises:
            ImportError: If ``scaled_masked_softmax_cuda`` is not available.
        """
        import scaled_masked_softmax_cuda  # type: ignore

        return scaled_masked_softmax_cuda.get_batch_per_block(sq, sk, b, np)
