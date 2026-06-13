# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M1233: Megatron 2e6a46e45 — Start Megatron-Core with vocab parallel cross entropy
# Source: megatron/core/utils.py (NVIDIA/Megatron-LM commit 2e6a46e45)
# Author: Jared Casper <jcasper@nvidia.com>  Date: 2022-09-22
#
# Mapping: megatron/core/utils.py → deepspeed/compile/core_utils.py
#
# New file in this commit — utility functions for megatron.core:
#   1. ensure_divisibility(numerator, denominator)
#   2. divide(numerator, denominator)
#   3. split_tensor_into_1d_equal_chunks(tensor)
#   4. gather_split_1d_tensor(tensor)
#
# Imports parallel_state from megatron.core; adapted to use
# deepspeed.compile.core_parallel_state in our mapping.
#
# 20% adaptation: imports core_parallel_state from deepspeed.compile rather
# than megatron.core.parallel_state; uses torch.distributed._all_gather_base
# matching upstream; adds print('[M1233]') marker.
# ---------------------------------------------------------------------------

print('[M1233]')

"""Utility functions used through Megatron core"""
import torch

from deepspeed.compile import core_parallel_state as parallel_state


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_into_1d_equal_chunks(tensor):
    """Break a tensor into equal 1D chunks."""
    data = tensor.view(-1)
    partition_size = (
        torch.numel(data) // parallel_state.get_tensor_model_parallel_world_size()
    )
    start_index = partition_size * parallel_state.get_tensor_model_parallel_rank()
    end_index = start_index + partition_size
    return data[start_index:end_index]


def gather_split_1d_tensor(tensor):
    """Opposite of above function, gather values from model parallel ranks."""
    world_size = parallel_state.get_tensor_model_parallel_world_size()
    numel = torch.numel(tensor)
    numel_gathered = world_size * numel
    gathered = torch.empty(
        numel_gathered,
        dtype=tensor.dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
    torch.distributed._all_gather_base(
        gathered,
        tensor,
        group=parallel_state.get_tensor_model_parallel_group()
    )
    return gathered


# ---------------------------------------------------------------------------
# M2050: Megatron 40fb590e4 — Move get_batch_on_this_cp_rank to mcore utils
# Source: megatron/core/utils.py (NVIDIA/Megatron-LM commit 40fb590e4)
#
# Mapping: megatron/core/utils.py → deepspeed/compile/core_utils.py
#          megatron/training/utils.py (old home) → function deleted there
#
# Changes ported from upstream diff:
#   1. Added get_batch_on_this_cp_rank(batch) to megatron/core/utils.py
#      (previously lived in megatron/training/utils.py, used get_args()+mpu).
#   2. Removed get_batch_on_this_cp_rank from megatron/training/utils.py.
#   3. Updated megatron/training/utils.py import block to pull from mcore.
#   4. Updated megatron/core/models/multimodal/llava_model.py import + call site.
#
# 20% adaptation (鲁迅式迁移):
#   鲁迅云：「从来如此，便对么？」旧函数久居 training/utils，
#   依赖 get_args() 与 mpu，如寄人篱下；今迁 mcore utils，
#   改用 parallel_state API，方为正途。
#   get_args() 与 mpu 之依赖尽除，换 parallel_state.get_context_parallel_*，
#   使 mcore 模块自洽，不必仰赖 training 层之鼻息。
#   print('[M2050]') 诊断印记，见证此次搬迁。
# ---------------------------------------------------------------------------


########################
### context parallel ###
########################


def get_batch_on_this_cp_rank(batch):
    """Slice batch input along sequence dimension into multiple chunks,
    which are parallelized across GPUs in a context parallel group.

    Upstream (megatron/training/utils.py) used get_args().context_parallel_size
    and mpu.get_context_parallel_rank(); here we use parallel_state API directly
    so mcore utils stays self-contained without training-layer imports.
    """
    # With causal masking, each token only attends to its prior tokens. Simply split
    # sequence into CP chunks can result in severe load imbalance. That's to say, chunks
    # at the end of sequence have bigger workload than others. To address this issue,
    # we split sequence into 2*CP ranks. Assuming CP=2, we then get 4 chunks, chunk_0
    # and chunk_3 are assigned to GPU0, chunk_1 and chunk_2 are assigned to GPU1, so
    # that we can get balanced workload among GPUs in a context parallel group.
    cp_size = parallel_state.get_context_parallel_world_size()
    print('[M2050] get_batch_on_this_cp_rank: cp_size=%d keys=%s' % (cp_size, list(batch.keys())))
    if cp_size > 1:
        cp_rank = parallel_state.get_context_parallel_rank()
        print('[M2050] get_batch_on_this_cp_rank: cp_rank=%d, slicing sequence dim' % cp_rank)
        for key, val in batch.items():
            if val is not None:
                seq_dim = 1 if key != 'attention_mask' else 2
                val = val.view(
                    *val.shape[0:seq_dim],
                    2 * cp_size,
                    val.shape[seq_dim] // (2 * cp_size),
                    *val.shape[(seq_dim + 1):],
                )
                index = torch.tensor(
                    [cp_rank, (2 * cp_size - cp_rank - 1)], device='cpu', pin_memory=True
                ).cuda(non_blocking=True)
                val = val.index_select(seq_dim, index)
                val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2):])
                batch[key] = val
    return batch
