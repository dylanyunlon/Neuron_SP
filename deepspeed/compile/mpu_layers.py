# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M54: Megatron 57c2060fe — Model parallel merger
# Source: megatron/mpu/layers.py (NVIDIA/Megatron-LM commit 57c2060fe)
# Author: Mohammad Shoeybi <mshoeybi@nvidia.com>  Date: 2020-02-10
#
# Mapping: mpu/* → deepspeed/compile/  (project convention)
#
# Changes ported from mpu/layers.py:
#   1. _initialize_affine_weight: set weight.model_parallel = True,
#      weight.partition_dim = partition_dim, weight.stride = stride
#      at the top of the function (before the world_size==1 shortcut).
#      Previously these attributes were set individually in each layer class;
#      centralising them here ensures every layer that uses this helper
#      automatically gets the metadata needed by checkpoint-merge tools.
#
#   2. VocabParallelEmbedding.__init__: remove standalone
#      self.weight.model_parallel = True  (now set in _initialize_affine_weight).
#
#   3. ParallelEmbedding.__init__: same removal.
#
#   4. ColumnParallelLinear.__init__: remove self.weight.model_parallel = True;
#      add self.bias.partition_dim = 0 and self.bias.stride = stride so that
#      checkpoint-merge code can reconstruct the bias shard as well.
#
#   5. RowParallelLinear.__init__: remove self.weight.model_parallel = True.
#
# 20% adaptation: standalone helper mark_weight_parallel() rather than
# inline attribute assignment; imports from deepspeed.compile.mpu_initialize
# instead of the original mpu group helpers; adds print markers.
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn

from .mpu_initialize import get_model_parallel_world_size

print('[M54]')


def mark_weight_parallel(weight: nn.Parameter,
                         partition_dim: int,
                         stride: int = 1) -> None:
    """Tag a weight Parameter with model-parallel shard metadata.

    Megatron 57c2060fe mpu/layers.py _initialize_affine_weight:
      weight.model_parallel = True
      weight.partition_dim  = partition_dim
      weight.stride         = stride

    These three attributes let checkpoint-merge utilities (merge_mp_partitions)
    reconstruct the full weight from per-rank shards without extra config files.

    Called at the top of _initialize_affine_weight-equivalent helpers so that
    every parallel layer automatically inherits the metadata.
    """
    weight.model_parallel = True
    weight.partition_dim = partition_dim
    weight.stride = stride
    print(f'[M54-LAYERS] mark_weight_parallel: '
          f'shape={list(weight.shape)} '
          f'partition_dim={partition_dim} stride={stride}')


def maybe_mark_bias_parallel(bias: nn.Parameter,
                              partition_dim: int,
                              stride: int = 1) -> None:
    """Tag a bias Parameter with model-parallel shard metadata.

    Megatron 57c2060fe mpu/layers.py ColumnParallelLinear.__init__:
      self.bias.model_parallel = True   (unchanged)
      self.bias.partition_dim  = 0      (NEW in 57c2060fe)
      self.bias.stride         = stride (NEW in 57c2060fe)

    Also used by BertLMHead.bias (bert_model.py → engine.py mapping).
    """
    bias.model_parallel = True
    bias.partition_dim = partition_dim
    bias.stride = stride
    print(f'[M54-LAYERS] maybe_mark_bias_parallel: '
          f'shape={list(bias.shape)} '
          f'partition_dim={partition_dim} stride={stride}')
