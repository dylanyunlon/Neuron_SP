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


class BiasSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias, input_2, bias_2):
        ctx.save_for_backward(input, bias, input_2, bias_2)
        return bias_swiglu(input, bias, input_2, bias_2)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias, input_2, bias_2 = ctx.saved_tensors
        tmp, tmp2 = bias_swiglu_back(grad_output, input, bias, input_2, bias_2)
        return tmp, tmp, tmp2, tmp2


class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, input_2):
        ctx.save_for_backward(input, input_2)
        return swiglu(input, input_2)

    @staticmethod
    def backward(ctx, grad_output):
        input, input_2 = ctx.saved_tensors
        tmp, tmp2 = swiglu_back(grad_output, input, input_2)
        return tmp, tmp2


bias_swiglu_impl = BiasSwiGLUFunction.apply
swiglu_impl = SwiGLUFunction.apply
