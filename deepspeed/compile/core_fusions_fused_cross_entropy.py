# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M1990: Megatron 38722c39b — Support jit.script with cross entropy fusion
# Source: megatron/core/fusions/fused_cross_entropy.py (NVIDIA/Megatron-LM commit 38722c39b)
# Author: NVIDIA CORPORATION
#
# Mapping: megatron/core/fusions/fused_cross_entropy.py  (NEW FILE)
#       -> deepspeed/compile/core_fusions_fused_cross_entropy.py
#          (project convention: megatron/core/fusions/* ->
#           deepspeed/compile/core_fusions_*)
#
# 核心问题与修复（commit 38722c39b）:
#   jit.script 要求被 script 化的函数内部不能有副作用（如 distributed state 查询）。
#   旧版 calculate_predicted_logits 在函数体内调用
#   get_tensor_model_parallel_rank / get_tensor_model_parallel_world_size，
#   导致 @jit_fuser（即 @torch.jit.script）编译失败。
#
#   修复：将 vocab_start_index/vocab_end_index 提升为显式 int 参数，
#   由 _VocabParallelCrossEntropy.forward()（非 JIT 环境）负责计算后传入。
#   @torch.jit.script 内部只做纯张量运算，distributed 调用全部移出。
#
# Changes vs prior commit:
#   1. fused_cross_entropy.py: import VocabUtility from utils.
#   2. calculate_predicted_logits: +vocab_start_index: int, +vocab_end_index: int.
#   3. _VocabParallelCrossEntropy.forward(): compute vocab range before fused call.
#   4. calculate_gradients: grad_input.bfloat16() → grad_input.to(torch.bfloat16)
#      (jit.script 兼容写法，.bfloat16() 可能不被所有 script 路径识别).
#
# 20% adaptation (鲁迅式迁移):
#   鲁迅曰："旧代码在 script 内问询分布式，如入无人之境却偏要拦门验证；
#            今将问询移出 script，令纯计算之函数得以自由驰骋。"
#   - @jit_fuser → @torch.jit.script（项目映射）。
#   - 所有 megatron.core.* 导入替换为 deepspeed.compile.* 等价模块。
#   - VocabParallelCrossEntropy 来自本项目的 core_tensor_parallel_cross_entropy。
#   - 加 print('[M1990]') 诊断标记。
# ---------------------------------------------------------------------------

print('[M1990] core_fusions_fused_cross_entropy: jit.script cross-entropy fusion loaded')

from typing import Tuple

import torch

from deepspeed.compile.core_parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from deepspeed.compile.core_tensor_parallel_cross_entropy import VocabParallelCrossEntropy
from deepspeed.compile.core_tensor_parallel_utils import VocabUtility


@torch.jit.script
def calculate_logits_max(
    vocab_parallel_logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 纯张量运算，无副作用，可安全 script。
    vocab_parallel_logits, logits_max = VocabParallelCrossEntropy.calculate_logits_max(
        vocab_parallel_logits
    )
    return vocab_parallel_logits, logits_max


@torch.jit.script
def calculate_predicted_logits(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    logits_max: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # M1990 关键：vocab_start_index/vocab_end_index 由调用方（非 JIT 环境）传入，
    # 此处不再调用任何 distributed helper，jit.script 编译得以通过。
    (
        target_mask,
        masked_target_1d,
        predicted_logits,
        sum_exp_logits,
        exp_logits,
    ) = VocabParallelCrossEntropy.calculate_predicted_logits(
        vocab_parallel_logits, target, logits_max, vocab_start_index, vocab_end_index
    )

    predicted_logits_sum_exp_logits = torch.cat((predicted_logits, sum_exp_logits))

    return target_mask, masked_target_1d, predicted_logits_sum_exp_logits, exp_logits


@torch.jit.script
def calculate_cross_entropy_loss(
    exp_logits: torch.Tensor,
    predicted_logits_sum_exp_logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    split_val = predicted_logits_sum_exp_logits.size()[0] // 2
    predicted_logits, sum_exp_logits = torch.split(predicted_logits_sum_exp_logits, split_val)

    exp_logits, loss = VocabParallelCrossEntropy.calculate_cross_entropy_loss(
        exp_logits, predicted_logits, sum_exp_logits
    )

    return exp_logits, loss


@torch.jit.script
def calculate_gradients(
    softmax: torch.Tensor,
    grad_output: torch.Tensor,
    target_mask: torch.Tensor,
    masked_target_1d: torch.Tensor,
) -> torch.Tensor:

    (
        grad_2d,
        arange_1d,
        softmax_update,
        grad_input,
    ) = VocabParallelCrossEntropy.prepare_gradient_calculation_operands(softmax, target_mask)

    grad_input = VocabParallelCrossEntropy.calculate_gradients(
        grad_2d, arange_1d, masked_target_1d, softmax_update, grad_input, grad_output
    )

    # M1990: .bfloat16() → .to(torch.bfloat16)，jit.script 兼容写法。
    grad_input = grad_input.to(torch.bfloat16)

    return grad_input


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target):

        vocab_parallel_logits, logits_max = calculate_logits_max(vocab_parallel_logits)
        torch.distributed.all_reduce(
            logits_max,
            op=torch.distributed.ReduceOp.MAX,
            group=get_tensor_model_parallel_group(),
        )

        # Get the partition's vocab indices
        # M1990: 在非 JIT 的 forward 上下文中计算 vocab 范围，以 int 传入 fused 函数。
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)
        print(f'[M1990] fused _VocabParallelCrossEntropy.forward vocab_start={vocab_start_index} vocab_end={vocab_end_index} partition_vocab_size={partition_vocab_size}')

        (
            target_mask,
            masked_target_1d,
            predicted_logits_sum_exp_logits,
            exp_logits,
        ) = calculate_predicted_logits(
            vocab_parallel_logits, target, logits_max, vocab_start_index, vocab_end_index
        )

        # All reduce is needed to get the chunks from other GPUs.
        # In the fused case, tensors are batched to invoke a single AllReduce call.
        torch.distributed.all_reduce(
            predicted_logits_sum_exp_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
        )

        exp_logits, loss = calculate_cross_entropy_loss(exp_logits, predicted_logits_sum_exp_logits)

        # Store softmax, target-mask and masked-target for backward pass.
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    def backward(ctx, grad_output):

        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        grad_input = calculate_gradients(softmax, grad_output, target_mask, masked_target_1d)

        return grad_input, None


def fused_vocab_parallel_cross_entropy(vocab_parallel_logits, target):
    """
    Performs cross entropy loss when logits are split across tensor parallel ranks.
    Uses jit.script-fused kernel functions for improved throughput.

    Args:
        vocab_parallel_logits: logits split across tensor parallel ranks
                               dimension is [sequence_length, batch_size, hidden_size]

        target: correct vocab ids of dimseion [sequence_length, micro_batch_size]
    """
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target)
