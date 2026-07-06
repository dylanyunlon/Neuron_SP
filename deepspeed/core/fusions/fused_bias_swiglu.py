"""Fused bias + SwiGLU activation with autograd support.

Mirrors Megatron-LM megatron/core/fusions/fused_bias_swiglu.py.

Provides:
- ``swiglu``: bare SwiGLU (no bias), JIT-fused.
- ``bias_swiglu``: SwiGLU with bias add, JIT-fused.
- ``swiglu_back`` / ``bias_swiglu_back``: analytic gradients, JIT-fused.
- ``BiasSwiGLUFunction`` / ``SwiGLUFunction``: custom autograd with optional
  FP8 activation store and CPU offload (activation memory optimisation for
  the DES-LOC A6000 tier).
- ``bias_swiglu_impl``: high-level entry point used by MLP layers.
- ``weighted_bias_swiglu_impl``: token-routing-weighted variant for MoE
  (imported by deepspeed/core/transformer/moe/experts.py).

The pure-PyTorch JIT path matches Megatron's kernel output exactly and runs
correctly on SM86 (A6000) without additional CUDA compilation.

Megatron source: Megatron-LM/megatron/core/fusions/fused_bias_swiglu.py
"""
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn.functional as F

from deepspeed.core.jit import jit_fuser
from deepspeed.core.utils import nvtx_decorator


# ---------------------------------------------------------------------------
# JIT-fused primitive ops
# ---------------------------------------------------------------------------

@jit_fuser
def swiglu(y: torch.Tensor) -> torch.Tensor:
    """SwiGLU activation: ``SiLU(y1) * y2``.

    Args:
        y: Input tensor; last dimension is split into two equal halves.

    Returns:
        Tensor with last dimension halved.
    """
    y_1, y_2 = torch.chunk(y, 2, -1)
    return F.silu(y_1) * y_2


@jit_fuser
def bias_swiglu(y: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Bias add followed by SwiGLU activation.

    Args:
        y: Input tensor ``[..., 2 * ffn_hidden_size]``.
        bias: Bias tensor ``[2 * ffn_hidden_size]``.

    Returns:
        ``swiglu(y + bias)`` with shape ``[..., ffn_hidden_size]``.
    """
    y = y + bias
    return swiglu(y)


@jit_fuser
def weighted_swiglu(y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """SwiGLU scaled by per-token routing weights.

    Args:
        y: Input ``[num_tokens, 2 * ffn_hidden_size]``.
        weights: Router probabilities ``[num_tokens, 1]``.

    Returns:
        ``swiglu(y) * weights`` in the original dtype of ``y``.
    """
    dtype = y.dtype
    res = swiglu(y) * weights
    return res.to(dtype)


# ---------------------------------------------------------------------------
# Analytic gradient kernels
# ---------------------------------------------------------------------------

@jit_fuser
def swiglu_back(g: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Gradient of SwiGLU with respect to its input.

    Uses the derivative of SiLU: ``sigmoid(x) * (1 + x * (1 - sigmoid(x)))``.

    Args:
        g: Upstream gradient ``[..., ffn_hidden_size]``.
        y: Input to the forward SwiGLU ``[..., 2 * ffn_hidden_size]``.

    Returns:
        Input gradient ``[..., 2 * ffn_hidden_size]``.
    """
    y_1, y_2 = torch.chunk(y, 2, -1)
    return torch.cat(
        (
            g * torch.sigmoid(y_1) * (1 + y_1 * (1 - torch.sigmoid(y_1))) * y_2,
            g * F.silu(y_1),
        ),
        -1,
    )


@jit_fuser
def bias_swiglu_back(
    g: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """Gradient of bias_swiglu with respect to the pre-bias input.

    Args:
        g: Upstream gradient ``[..., ffn_hidden_size]``.
        y: Pre-bias input ``[..., 2 * ffn_hidden_size]``.
        bias: Bias added in the forward pass ``[2 * ffn_hidden_size]``.

    Returns:
        Input gradient ``[..., 2 * ffn_hidden_size]``.
    """
    y = y + bias
    return swiglu_back(g, y)


@jit_fuser
def weighted_swiglu_back(
    g: torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradient of weighted_swiglu.

    Args:
        g: Upstream gradient ``[num_tokens, ffn_hidden_size]``.
        y: Input to forward pass ``[num_tokens, 2 * ffn_hidden_size]``.
        weights: Router probabilities ``[num_tokens, 1]``.

    Returns:
        Tuple ``(input_grad, weights_grad)``.
    """
    input_dtype = y.dtype
    w_dtype = weights.dtype
    input_grad = swiglu_back(g * weights, y)
    weights_grad = swiglu(y) * g.to(w_dtype)
    weights_grad = torch.sum(weights_grad, dim=-1, keepdim=True)
    return input_grad.to(input_dtype), weights_grad.to(w_dtype)


# ---------------------------------------------------------------------------
# Custom autograd Functions
# ---------------------------------------------------------------------------

class BiasSwiGLUFunction(torch.autograd.Function):
    """Custom autograd for SwiGLU with bias, FP8 store, and CPU offload.

    FP8 intermediate storage halves activation memory for long sequences on
    the DES-LOC A6000 tier (24 GB VRAM).  CPU offload is supported for the
    same reason via PyTorch's activation-offloading hooks.
    """

    @staticmethod
    @nvtx_decorator()
    def forward(
        ctx,
        input: torch.Tensor,
        bias: torch.Tensor,
        fp8_input_store: bool,
        cpu_offload_input: bool,
    ) -> torch.Tensor:
        """Forward pass: bias add then SwiGLU.

        Args:
            ctx: Autograd context.
            input: ``[..., 2 * ffn_hidden_size]``.
            bias: ``[2 * ffn_hidden_size]``.
            fp8_input_store: If ``True``, cast the saved activation to FP8
                (``torch.float8_e4m3fn``) to save memory.
            cpu_offload_input: If ``True``, mark tensors for CPU offload.

        Returns:
            ``[..., ffn_hidden_size]``.
        """
        input_for_backward = (
            input.to(torch.float8_e4m3fn) if fp8_input_store else input
        )
        if cpu_offload_input:
            input_for_backward.activation_offloading = True
            bias.activation_offloading = True
        ctx.save_for_backward(input_for_backward, bias)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return bias_swiglu(input, bias)

    @staticmethod
    @nvtx_decorator()
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass.

        Returns:
            ``(input_grad, bias_grad, None, None)``.
        """
        input, bias = ctx.saved_tensors
        if ctx.fp8_input_store:
            input = input.to(ctx.ori_input_dtype)
        tmp = bias_swiglu_back(grad_output, input, bias)
        # bias gradient = sum over all but the last dim
        return tmp, tmp, None, None


class SwiGLUFunction(torch.autograd.Function):
    """Custom autograd for SwiGLU without bias."""

    @staticmethod
    @nvtx_decorator()
    def forward(
        ctx,
        input: torch.Tensor,
        fp8_input_store: bool,
        cpu_offload_input: bool,
    ) -> torch.Tensor:
        """Forward pass: SwiGLU activation.

        Args:
            ctx: Autograd context.
            input: ``[..., 2 * ffn_hidden_size]``.
            fp8_input_store: Store activation in FP8 to save memory.
            cpu_offload_input: Mark activation for CPU offload.

        Returns:
            ``[..., ffn_hidden_size]``.
        """
        input_for_backward = (
            input.to(torch.float8_e4m3fn) if fp8_input_store else input
        )
        if cpu_offload_input:
            input_for_backward.activation_offloading = True
        ctx.save_for_backward(input_for_backward)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return swiglu(input)

    @staticmethod
    @nvtx_decorator()
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass.

        Returns:
            ``(input_grad, None, None)``.
        """
        (input,) = ctx.saved_tensors
        if ctx.fp8_input_store:
            input = input.to(ctx.ori_input_dtype)
        tmp = swiglu_back(grad_output, input)
        return tmp, None, None


class WeightedSwiGLUFunction(torch.autograd.Function):
    """Custom autograd for token-routing-weighted SwiGLU."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weights: torch.Tensor,
        fp8_input_store: bool,
    ) -> torch.Tensor:
        """Forward: ``swiglu(input) * weights``.

        Args:
            ctx: Autograd context.
            input: ``[num_tokens, 2 * ffn_hidden_size]``.
            weights: ``[num_tokens, 1]`` router probabilities.
            fp8_input_store: Store activation in FP8.

        Returns:
            ``[num_tokens, ffn_hidden_size]``.
        """
        input_for_backward = (
            input.to(torch.float8_e4m3fn) if fp8_input_store else input
        )
        ctx.save_for_backward(input_for_backward, weights)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return weighted_swiglu(input, weights)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass.

        Returns:
            ``(input_grad, weights_grad, None)``.
        """
        input, weights = ctx.saved_tensors
        if ctx.fp8_input_store:
            input = input.to(ctx.ori_input_dtype)
        tmp, wgrad = weighted_swiglu_back(grad_output, input, weights)
        return tmp, wgrad, None


# ---------------------------------------------------------------------------
# High-level entry points (used by MLP / MoE layers)
# ---------------------------------------------------------------------------

def bias_swiglu_impl(
    input: torch.Tensor,
    bias: torch.Tensor | None,
    fp8_input_store: bool = False,
    cpu_offload_input: bool = False,
) -> torch.Tensor:
    """Apply SwiGLU (with optional bias) using the custom autograd path.

    Handles 2-D ``[num_tokens, 2H]`` and 3-D ``[s, b, 2H]`` inputs by
    reshaping to 2-D, applying the activation, then restoring the shape.

    Args:
        input: ``[..., 2 * ffn_hidden_size]``, 2-D or 3-D.
        bias: ``[2 * ffn_hidden_size]`` or ``None``.
        fp8_input_store: Store the activation checkpoint in FP8.
        cpu_offload_input: Mark activation tensors for CPU offload.

    Returns:
        ``[..., ffn_hidden_size]``.

    Raises:
        AssertionError: If input is not 2-D or 3-D.
    """
    ori_shape = input.shape
    assert len(ori_shape) in (2, 3), (
        f"bias_swiglu_impl: expected 2-D or 3-D input, got {len(ori_shape)}-D"
    )
    input = input.view(-1, ori_shape[-1])
    if bias is not None:
        output = BiasSwiGLUFunction.apply(
            input, bias, fp8_input_store, cpu_offload_input
        )
    else:
        output = SwiGLUFunction.apply(input, fp8_input_store, cpu_offload_input)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


def weighted_bias_swiglu_impl(
    input: torch.Tensor,
    bias: torch.Tensor | None,
    weights: torch.Tensor,
    fp8_input_store: bool = False,
) -> torch.Tensor:
    """Token-routing-weighted SwiGLU for MoE expert aggregation.

    Imported by ``deepspeed.core.transformer.moe.experts`` to scale each
    token's expert output by its routing probability before summation.

    Args:
        input: ``[num_tokens, 2 * ffn_hidden_size]`` or 3-D equivalent.
        bias: Not supported; must be ``None`` (raises if provided).
        weights: ``[num_tokens, 1]`` or ``[num_tokens]`` router probabilities.
        fp8_input_store: Store activation in FP8 to save VRAM.

    Returns:
        ``[num_tokens, ffn_hidden_size]`` (or 3-D restored shape).

    Raises:
        NotImplementedError: If ``bias`` is not ``None``.
        AssertionError: If input is not 2-D or 3-D.
    """
    ori_shape = input.shape
    assert len(ori_shape) in (2, 3), (
        f"weighted_bias_swiglu_impl: expected 2-D or 3-D input, got {len(ori_shape)}-D"
    )
    input = input.view(-1, ori_shape[-1])
    if bias is not None:
        raise NotImplementedError(
            "weighted_bias_swiglu_impl: bias is not supported for weighted swiglu fusion"
        )
    output = WeightedSwiGLUFunction.apply(input, weights, fp8_input_store)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)
