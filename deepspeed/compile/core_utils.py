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
