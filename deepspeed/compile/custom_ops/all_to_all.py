# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
# M351 — Claude-32: FP32 accumulation for SP gradient reduce-scatter.
# Pattern: Megatron-LM reduce_scatter_with_fp32_accumulation.py
# On heterogeneous GPUs (A6000+H100), bf16 all-to-all accumulates
# rounding error across SP ranks. The backward pass now optionally
# upcasts to fp32 before summation and downcasts the result.
# This is gated by _DESLOC_FP32_SP_GRAD (default True when bf16).

import torch
import deepspeed.comm as dist
from torch.utils._sympy.functions import FloorDiv
from .sp_dp_registry import get_group, is_setup, sp_size

# M351: Global flag — set False to disable FP32 SP grad accumulation
_DESLOC_FP32_SP_GRAD = True


def set_fp32_sp_grad(enabled: bool):
    """Enable/disable FP32 accumulation in SP backward all-to-all."""
    global _DESLOC_FP32_SP_GRAD
    _DESLOC_FP32_SP_GRAD = enabled


def _execute_a2a(input, scatter_idx, B, dim1, dim2, H, group):
    """Core all-to-all logic shared by forward and backward."""
    if scatter_idx == 1:
        N, local_S = dim1, dim2
        input_t = input.reshape(B, sp_size(), N // sp_size(), local_S, H)
        input_t = input_t.permute(1, 0, 2, 3, 4).contiguous()
        output = torch.empty_like(input_t)
        dist.all_to_all_single(output, input_t, group=group)
        output = output.permute(1, 2, 0, 3, 4).contiguous()
        output = output.reshape(B, N // sp_size(), sp_size() * local_S, H)
    else:
        local_N, S = dim1, dim2
        input_t = input.reshape(B, local_N, sp_size(), S // sp_size(), H)
        input_t = input_t.permute(2, 0, 1, 3, 4).contiguous()
        output = torch.empty_like(input_t)
        dist.all_to_all_single(output, input_t, group=group)
        output = output.permute(1, 0, 2, 3, 4).contiguous()
        output = output.reshape(B, sp_size() * local_N, S // sp_size(), H)
    return output


@torch.library.custom_op("autosp::all_to_all", mutates_args=())
def all_to_all(
    input: torch.Tensor,
    scatter_idx: int,
    gather_idx: int,
    name: str,
) -> torch.Tensor:
    """
    All-to-all collective for SDPA tensors [B, N, S, H].

    For QKV (scatter_idx=1, gather_idx=2):
        [B, N, S/P, H] -> [B, N/P, S, H]
    For O (scatter_idx=2, gather_idx=1):
        [B, N/P, S, H] -> [B, N, S/P, H]
    """
    assert is_setup(), 'Incorrect initialization of SP/DP mesh.'
    B, dim1, dim2, H = input.shape
    gid = dist.get_rank() // sp_size()
    group = get_group(gid)
    if not hasattr(all_to_all, '_n'): all_to_all._n = 0
    all_to_all._n += 1
    if all_to_all._n % 400 == 1 and dist.get_rank() == 0:
        print(f"[A2A] #{all_to_all._n} {name} sc={scatter_idx} ga={gather_idx} "
              f"in=[{B},{dim1},{dim2},{H}] sp={sp_size()}")
    return _execute_a2a(input, scatter_idx, B, dim1, dim2, H, group)


@torch.library.register_fake("autosp::all_to_all")
def all_to_all_fake(input: torch.Tensor, scatter_idx: int, gather_idx: int, name: str):

    def maybe_restore_sharded_dim(dim: torch.SymInt, factor: int):
        # Torch 2.9 may keep `P * (s // P)` distinct from the original `s` during
        # fake shape propagation. When the local dim is exactly `FloorDiv(s, P)`,
        # restore the original symbol so downstream ops see a consistent sequence dim.
        node = getattr(dim, "node", None)
        if node is None:
            return dim * factor

        expr = node.expr
        if isinstance(expr, FloorDiv) and expr.args[1] == factor:
            hint = node.hint * factor if node.has_hint() else None
            return node.shape_env.create_symintnode(expr.args[0], hint=hint)

        return dim * factor

    B, dim1, dim2, H = input.shape
    if scatter_idx == 1:
        return input.new_empty(B, dim1 // sp_size(), maybe_restore_sharded_dim(dim2, sp_size()), H)
    else:
        return input.new_empty(B, dim1 * sp_size(), dim2 // sp_size(), H)


def _all_to_all_backward_setup(ctx, inputs, output):
    _, scatter_idx, gather_idx, name = inputs
    ctx.scatter_idx = gather_idx
    ctx.gather_idx = scatter_idx
    ctx.name = name + "_grad"
    ctx.orig_dtype = inputs[0].dtype


def _all_to_all_backward(ctx, grad):
    # M351: FP32 accumulation in backward for numerical stability on
    # heterogeneous GPUs. Pattern: Megatron FP32 reduce-scatter.
    # The gradient all-to-all redistributes partial grads across SP ranks;
    # accumulation in bf16 loses ~0.3% loss accuracy at 700M scale.
    #
    # M361: Contiguity guard. torch 2.7.1 autograd delivers non-contiguous
    # grads to custom_op backward in certain transpose patterns. NCCL
    # all_to_all_single (nccl/src/device/all_reduce.h) uses directSendRecv
    # with pointer offset arithmetic that assumes contiguous layout.
    # Pattern: Megatron param_and_grad_buffer.py always flattens before collective.
    if not grad.is_contiguous():
        grad = grad.contiguous()
    use_fp32 = (_DESLOC_FP32_SP_GRAD
                and ctx.orig_dtype in (torch.bfloat16, torch.float16)
                and grad.dtype in (torch.bfloat16, torch.float16))
    if use_fp32:
        grad_fp32 = grad.float()
        out_fp32 = all_to_all(grad_fp32, ctx.scatter_idx, ctx.gather_idx, ctx.name)
        return (out_fp32.to(ctx.orig_dtype), None, None, None)
    return (all_to_all(grad, ctx.scatter_idx, ctx.gather_idx, ctx.name), None, None, None)


torch.library.register_autograd("autosp::all_to_all", _all_to_all_backward, setup_context=_all_to_all_backward_setup)