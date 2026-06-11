# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M54: Megatron 57c2060fe — Model parallel merger
# Source: merge_mp_partitions.py (NVIDIA/Megatron-LM commit 57c2060fe)
# Author: Mohammad Shoeybi <mshoeybi@nvidia.com>  Date: 2020-02-10
#
# Mapping: merge_mp_partitions.py (new file) → deepspeed/compile/
#          (project convention: mpu/* → deepspeed/compile/)
#
# Utility to merge model-parallel checkpoint partitions into a single
# full-model checkpoint.  The key primitives:
#   split_into_partitions()  — split a full tensor into N shards (for testing)
#   merge_partitions()       — concatenate N shards back into a full tensor,
#                              handling stride > 1 (QKV interleaving)
#   get_parallel_checkpoint_name() — resolve checkpoint path from tracker file
#   test_split_merge()       — self-test for the split/merge roundtrip
#   main()                   — full merge workflow: load N rank checkpoints
#                              and save a single merged checkpoint
#
# 20% adaptation: imports use deepspeed.compile.mpu_initialize instead of
# the original megatron.mpu.initialize; model/checkpoint helpers reference
# deepspeed utilities; adds print('[M54]') marker.
# ---------------------------------------------------------------------------

import os

import torch

from .mpu_initialize import (
    set_model_parallel_world_size,
    set_model_parallel_rank,
    get_model_parallel_world_size,
)

print('[M54]')


# ---------------------------------------------------------------------------
# Partition splitting / merging helpers
# ---------------------------------------------------------------------------

def _divide(numerator, denominator):
    """Integer divide with assertion (mirrors mpu.utils.divide)."""
    assert numerator % denominator == 0, \
        f'{numerator} is not divisible by {denominator}'
    return numerator // denominator


def split_into_partitions(tensor, num_partitions, partition_dim, stride):
    """Split *tensor* into *num_partitions* shards along *partition_dim*.

    When stride > 1 the tensor is first split into stride-sized chunks and
    then interleaved, matching the ColumnParallelLinear weight layout used by
    QKV projections.

    Megatron 57c2060fe merge_mp_partitions.py::split_into_partitions.
    """
    per_partition_size = _divide(tensor.size(partition_dim), num_partitions)
    per_partition_per_stride_size = _divide(per_partition_size, stride)

    partitions_list = torch.split(tensor,
                                  per_partition_per_stride_size,
                                  dim=partition_dim)

    partitions = []
    for i in range(num_partitions):
        partition = torch.cat(partitions_list[i::num_partitions],
                              dim=partition_dim)
        partitions.append(partition)

    return partitions


def merge_partitions(merged, partitions, partition_dim, stride):
    """Merge *partitions* into *merged* along *partition_dim*.

    Handles stride > 1 by reversing the interleaving used during splitting.
    If the concatenated size exceeds *merged*'s size, the excess is trimmed
    (vocab-padding use case).

    Megatron 57c2060fe merge_mp_partitions.py::merge_partitions.
    """
    num_partitions = len(partitions)
    per_partition_size = None
    for partition in partitions:
        if per_partition_size is None:
            per_partition_size = partition.size(partition_dim)
        else:
            assert per_partition_size == partition.size(partition_dim)

    def concat_partitions(partitions_):
        with torch.no_grad():
            if (per_partition_size * num_partitions) == merged.size(partition_dim):
                torch.cat(partitions_, dim=partition_dim, out=merged)
            else:
                print('     ***WARNING*** sizes do not match. Will cut '
                      'the merged partitions by {} along dimension {} '
                      'to reduce the size from {} to {} ...'.format(
                          (per_partition_size * num_partitions) -
                          merged.size(partition_dim), partition_dim,
                          per_partition_size * num_partitions,
                          merged.size(partition_dim)))
                merged_ = torch.cat(partitions_, dim=partition_dim)
                merged_split = torch.split(merged_, merged.size(partition_dim),
                                           dim=partition_dim)
                merged_ = merged_split[0]
                assert merged_.size(partition_dim) == merged.size(partition_dim)
                merged.data.copy_(merged_.data)

    # Stride == 1: simple concatenation.
    if stride == 1:
        concat_partitions(partitions)
        return

    # Stride > 1: undo per-partition interleaving.
    per_partition_per_stride_size = _divide(per_partition_size, stride)
    chunks = None
    for i, partition in enumerate(partitions):
        chunk = torch.split(partition,
                            per_partition_per_stride_size,
                            dim=partition_dim)
        if chunks is None:
            chunks = [0] * (num_partitions * len(chunk))
        chunks[i::num_partitions] = chunk

    concat_partitions(chunks)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def test_split_merge():
    """Verify split→merge roundtrip returns the original tensor.

    Megatron 57c2060fe merge_mp_partitions.py::test_split_merge.
    """
    print('[M54] testing split and merge ...')

    # [QKV.ROW-COL] — 3 heads × 4 rows × 5 cols
    tensor = torch.FloatTensor([
        [1.11, 1.12, 1.13, 1.14, 1.15],
        [1.21, 1.22, 1.23, 1.24, 1.25],
        [1.31, 1.32, 1.33, 1.34, 1.35],
        [1.41, 1.42, 1.43, 1.44, 1.45],
        [2.11, 2.12, 2.13, 2.14, 2.15],
        [2.21, 2.22, 2.23, 2.24, 2.25],
        [2.31, 2.32, 2.33, 2.34, 2.35],
        [2.41, 2.42, 2.43, 2.44, 2.45],
        [3.11, 3.12, 3.13, 3.14, 3.15],
        [3.21, 3.22, 3.23, 3.24, 3.25],
        [3.31, 3.32, 3.33, 3.34, 3.35],
        [3.41, 3.42, 3.43, 3.44, 3.45],
    ])

    num_partitions = 2
    partition_dim = 0
    stride = 3
    partitions = split_into_partitions(tensor, num_partitions,
                                       partition_dim, stride)

    merged = torch.zeros_like(tensor)
    merge_partitions(merged, partitions, partition_dim, stride)

    max_error = (merged - tensor).abs().max()
    print(f'[M54]   > max error (should be zero): {max_error}')


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def get_parallel_checkpoint_name(path, get_checkpoint_tracker_filename,
                                  get_checkpoint_name):
    """Read the iteration tracker and return the checkpoint path.

    Megatron 57c2060fe merge_mp_partitions.py::get_parallel_checkpoint_name.

    Args:
        path: checkpoint root directory.
        get_checkpoint_tracker_filename: callable(path) → tracker filename.
        get_checkpoint_name: callable(path, iteration) → checkpoint filename.
    """
    tracker_filename = get_checkpoint_tracker_filename(path)
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        iteration = int(metastring)
    assert iteration > 0
    checkpoint_name = get_checkpoint_name(path, iteration)
    return checkpoint_name, iteration


# ---------------------------------------------------------------------------
# Main merge workflow
# ---------------------------------------------------------------------------

def merge_model_parallel_checkpoints(
    load_path,
    model_parallel_size,
    get_model_fn,
    get_checkpoint_tracker_filename,
    get_checkpoint_name,
    ensure_directory_exists,
):
    """Merge *model_parallel_size* checkpoint partitions into one.

    Megatron 57c2060fe merge_mp_partitions.py::main — adapted as a callable
    function rather than a script entry point so it can be invoked from
    DeepSpeed checkpoint utilities.

    Args:
        load_path: root directory containing rank checkpoints.
        model_parallel_size: number of model-parallel partitions.
        get_model_fn: callable() → nn.Module (full merged model, fp16).
        get_checkpoint_tracker_filename: callable(path) → str.
        get_checkpoint_name: callable(path, iteration) → str.
        ensure_directory_exists: callable(filename) — creates parent dirs.
    """
    print('[M54] merging model parallel partitions ...')
    print(f' > number of partitions: {model_parallel_size}')
    print(f' > checkpoint path: {load_path}')

    # Build full model at world_size=1 / rank=0.
    set_model_parallel_world_size(1)
    set_model_parallel_rank(0)
    merged_model = get_model_fn()

    # Build and load each partition.
    partitions = []
    iteration = 0
    set_model_parallel_world_size(model_parallel_size)
    for rank in range(model_parallel_size):
        set_model_parallel_rank(rank)
        checkpoint_name, iteration = get_parallel_checkpoint_name(
            load_path, get_checkpoint_tracker_filename, get_checkpoint_name)
        print(f'[M54] > loading {checkpoint_name} ...')
        model_ = get_model_fn()
        sd = torch.load(checkpoint_name, map_location='cpu')
        model_.load_state_dict(sd['model'])
        partitions.append(model_)

    # Iterate over named parameters and merge shards.
    merged_params_gen = merged_model.named_parameters()
    partitions_params_gen = [p.named_parameters() for p in partitions]
    while True:
        try:
            name, merged_param = next(merged_params_gen)
            print(f'[M54] > working on {name} ...')
            print(f'     merged         type: {merged_param.dtype}, '
                  f'size: {list(merged_param.size())}')
            partitions_param = []
            for rank, partition_params_gen in enumerate(partitions_params_gen):
                partition_name, partition_param = next(partition_params_gen)
                assert partition_name == name
                partitions_param.append(partition_param)
                print(f'     partition {rank}    type: {partition_param.dtype}, '
                      f'size: {list(partition_param.size())}')

            if not hasattr(merged_param, 'model_parallel'):
                print('     none-parallel parameter, simple copy from rank 0')
                with torch.no_grad():
                    merged_param.data.copy_(partitions_param[0].data)
            else:
                print('     parallel parameter merge with stride {} along '
                      'dimention {}'.format(merged_param.stride,
                                            merged_param.partition_dim))
                merge_partitions(merged_param,
                                 partitions_param,
                                 merged_param.partition_dim,
                                 merged_param.stride)
        except StopIteration:
            break

    # Save merged checkpoint.
    set_model_parallel_rank(0)
    sd = {}
    sd['model'] = merged_model.state_dict()
    sd['iteration'] = iteration
    merged_path = os.path.join(load_path, 'merged')
    checkpoint_name = get_checkpoint_name(merged_path, iteration)
    ensure_directory_exists(checkpoint_name)
    print(f'[M54] > saving merged model to {checkpoint_name}')
    torch.save(sd, checkpoint_name)
    print('[M54] done :-)')
