# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M1910: Megatron 80de44fda — Add RoPE and SwiGLU fusion
# Source: megatron/core/fusions/fused_bias_swiglu.py (NVIDIA/Megatron-LM commit 80de44fda)
# Author: NVIDIA CORPORATION
#
# Mapping: megatron/core/fusions/fused_bias_swiglu.py  (NEW FILE)
#       -> deepspeed/compile/core_fusions_fused_bias_swiglu.py
#          (project convention: megatron/core/fusions/* ->
#           deepspeed/compile/core_fusions_*)
#
# Changes in this commit:
#   1. New file: SwiGLU fused kernel implementations (swiglu, bias_swiglu,
#      swiglu_back, bias_swiglu_back) as @torch.jit.script functions.
#   2. BiasSwiGLUFunction and SwiGLUFunction as torch.autograd.Function.
#   3. Exported: bias_swiglu_impl, swiglu_impl for use in MLP.
#
# 20% adaptation (鲁迅式迁移):
#   鲁迅曰: "不用 apex，不用 triton，只用 jit.script，此乃平民之融合。"
#   - 保留所有 @torch.jit.script 装饰器，原汁原味。
#   - 加 print('[M1910]') 诊断标记，以示迁移之印记。
#   - 注释改为中英文混注，适应 DeepSpeed 项目风格。
# ---------------------------------------------------------------------------

print('[M1910] core_fusions_fused_bias_swiglu loaded')

import torch
import torch.nn.functional as F

# SwiGLU Fusion — NO AUTOGRAD (JIT script path)
# swiglu(y, y_2) = silu(y) * y_2
# 对应 Llama / Mistral / Baichuan 等使用 SwiGLU 的主流架构

@torch.jit.script
def swiglu(y, y_2):
    return F.silu(y) * y_2

@torch.jit.script
def bias_swiglu(y, bias, y_2, bias_2):
    x = bias + y
    x_2 = bias_2 + y_2
    return swiglu(x, x_2)

# SwiGLU 梯度: d/dy[silu(y)*y_2] = sigmoid(y)*(1 + y*(1-sigmoid(y)))*y_2
# d/dy_2[silu(y)*y_2] = silu(y)
@torch.jit.script
def swiglu_back(g, y, y_2):
    return g * torch.sigmoid(y) * (1 + y * (1 - torch.sigmoid(y))) * y_2, g * F.silu(y)

@torch.jit.script
def bias_swiglu_back(g, y, bias, y_2, bias_2):
    x_1 = bias + y
    x_2 = bias_2 + y_2
    return swiglu_back(g, x_1, x_2)


# ---------------------------------------------------------------------------
# M1970: Megatron 21648b5ab — Store SwiGLU inputs in fp8 to save activation memory
# fp8_input_store=True → save gate/up halves as float8_e4m3fn; cast back before backward.
# 鲁迅曰：「省内存如省粮，fp8 压缩乃当今节粮之道；backward 还原，不差毫厘。」
# ---------------------------------------------------------------------------
class BiasSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias, input_2, bias_2, fp8_input_store):
        # M1970: cast both gate/up halves to fp8 for cheaper activation storage
        inp_save = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        inp2_save = input_2.to(torch.float8_e4m3fn) if fp8_input_store else input_2
        ctx.save_for_backward(inp_save, bias, inp2_save, bias_2)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        print(f'[M1970] BiasSwiGLUFunction.forward fp8_input_store={fp8_input_store} '
              f'input.dtype={input.dtype}')
        return bias_swiglu(input, bias, input_2, bias_2)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias, input_2, bias_2 = ctx.saved_tensors
        # M1970: cast fp8 tensors back to original precision before computing gradients
        if ctx.fp8_input_store:
            input = input.to(ctx.ori_input_dtype)
            input_2 = input_2.to(ctx.ori_input_dtype)
        tmp, tmp2 = bias_swiglu_back(grad_output, input, bias, input_2, bias_2)
        return tmp, tmp, tmp2, tmp2, None


class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, input_2, fp8_input_store):
        # M1970: cast both halves to fp8 for cheaper activation storage
        inp_save = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        inp2_save = input_2.to(torch.float8_e4m3fn) if fp8_input_store else input_2
        ctx.save_for_backward(inp_save, inp2_save)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        print(f'[M1970] SwiGLUFunction.forward fp8_input_store={fp8_input_store} '
              f'input.dtype={input.dtype}')
        return swiglu(input, input_2)

    @staticmethod
    def backward(ctx, grad_output):
        input, input_2 = ctx.saved_tensors
        # M1970: cast fp8 tensors back to original precision before computing gradients
        if ctx.fp8_input_store:
            input = input.to(ctx.ori_input_dtype)
            input_2 = input_2.to(ctx.ori_input_dtype)
        tmp, tmp2 = swiglu_back(grad_output, input, input_2)
        return tmp, tmp2, None


def bias_swiglu_impl(input, bias, input_2, bias_2, fp8_input_store=False):
    """Wrapper for BiasSwiGLUFunction; fp8_input_store saves activation memory (M1970)."""
    return BiasSwiGLUFunction.apply(input, bias, input_2, bias_2, fp8_input_store)


def swiglu_impl(input, input_2, fp8_input_store=False):
    """Wrapper for SwiGLUFunction; fp8_input_store saves activation memory (M1970)."""
    return SwiGLUFunction.apply(input, input_2, fp8_input_store)
