# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M1233: Megatron 2e6a46e45 — Start Megatron-Core with vocab parallel cross entropy
# Source: megatron/core/tensor_parallel/utils.py (NVIDIA/Megatron-LM commit 2e6a46e45)
# Author: Jared Casper <jcasper@nvidia.com>  Date: 2022-09-22
#
# Mapping: megatron/core/tensor_parallel/utils.py
#          → deepspeed/compile/core_tensor_parallel_utils.py
#
# This commit renames megatron/mpu/utils.py → megatron/core/tensor_parallel/utils.py
# Changes from mpu/utils.py:
#   1. Add `from typing import List, Sequence` imports.
#   2. Import divide from megatron.core.utils (new core utils module).
#   3. split_tensor_along_last_dim(): add type annotations on all params + return.
#   4. VocabUtility docstring: fix typos ("amd"→"and", "indecies"→"indices").
#   5. VocabUtility.vocab_range_from_per_partition_vocab_size(): add Sequence[int] return type.
#   6. VocabUtility.vocab_range_from_global_vocab_size(): add type annotations.
#   7. Style: reformat multi-line function calls.
#
# 20% adaptation: imports divide from deepspeed.compile.core_utils (our mapping
# of megatron.core.utils); adds print('[M1233]') marker.
# ---------------------------------------------------------------------------

print('[M1233]')

import torch
from typing import List, Sequence

from deepspeed.compile.core_utils import divide


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class VocabUtility:
    """Split the vocabulary into `world_size` chunks and return the
    first and last index of the vocabulary belonging to the `rank`
    partition: Note that indices in [fist, last)"""

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size: int, rank, world_size: int
    ) -> Sequence[int]:
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size: int, rank: int, world_size: int) -> Sequence[int]:
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size, rank, world_size
        )
