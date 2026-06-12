# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M54: Megatron 57c2060fe — Model parallel merger
# Source: megatron/mpu/initialize.py (NVIDIA/Megatron-LM commit 57c2060fe)
# Author: Mohammad Shoeybi <mshoeybi@nvidia.com>  Date: 2020-02-10
#
# Mapping: mpu/* → deepspeed/compile/  (project convention)
#
# Changes ported:
#   1. Add module-level _MPU_WORLD_SIZE / _MPU_RANK override globals.
#   2. Add set_model_parallel_world_size() setter.
#   3. Modify get_model_parallel_world_size() to honour override first.
#   4. Add set_model_parallel_rank() setter.
#   5. Modify get_model_parallel_rank() to honour override first.
#
# These overrides allow checkpoint-merge tooling to pretend the process
# group has world_size=1 / rank=0 without rebuilding NCCL groups — the
# primary use case in merge_mp_partitions.py.
#
# 20% adaptation: uses deepspeed.comm instead of torch.distributed directly;
# falls back to 1/0 when not initialised (test-safe); adds print markers.
# ---------------------------------------------------------------------------
# M345: Megatron 5c04ceb31 — Implementing lazy parallel initialization
# Source: megatron/mpu/__init__.py + megatron/mpu/initialize.py
#         (NVIDIA/Megatron-LM commit 5c04ceb31)
# Author: Boris Fomitchev <bfomitchev@nvidia.com>  Date: 2020-08-05
#
# Mapping: megatron/mpu/initialize.py → deepspeed/compile/mpu_initialize.py
#
# Changes ported:
#   1. megatron/mpu/__init__.py: export set_model_parallel_rank and
#      set_model_parallel_world_size (already present from M54 in DS).
#   2. megatron/mpu/initialize.py: remove set_model_parallel_group() and
#      set_data_parallel_group() helpers — these were not present in the DS
#      mapping (no _MODEL_PARALLEL_GROUP / _DATA_PARALLEL_GROUP globals here),
#      so no deletion needed; the export additions are the meaningful change.
#
# 20% adaptation: set_model_parallel_rank / set_model_parallel_world_size
# already exported; this entry records the upstream mpu/__init__.py change.
# ---------------------------------------------------------------------------

import deepspeed.comm as dist

print('[M345]')

# These values enable us to change the mpu sizes on the fly.
_MPU_WORLD_SIZE = None
_MPU_RANK = None


def set_model_parallel_world_size(world_size):
    """Set the model parallel size.

    Megatron 57c2060fe mpu/initialize.py — allows callers (e.g. checkpoint
    merge tools) to override the distributed world size without touching
    process groups.
    """
    global _MPU_WORLD_SIZE
    _MPU_WORLD_SIZE = world_size
    print(f'[M54-MPU] set_model_parallel_world_size({world_size})')


def get_model_parallel_world_size():
    """Return world size for the model parallel group.

    Megatron 57c2060fe mpu/initialize.py — returns override if set,
    otherwise queries the distributed group.
    """
    global _MPU_WORLD_SIZE
    if _MPU_WORLD_SIZE is not None:
        return _MPU_WORLD_SIZE
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def set_model_parallel_rank(rank):
    """Set model parallel rank.

    Megatron 57c2060fe mpu/initialize.py — allows callers to override
    the distributed rank without touching process groups.
    """
    global _MPU_RANK
    _MPU_RANK = rank
    print(f'[M54-MPU] set_model_parallel_rank({rank})')


def get_model_parallel_rank():
    """Return my rank for the model parallel group.

    Megatron 57c2060fe mpu/initialize.py — returns override if set,
    otherwise queries the distributed group.
    """
    global _MPU_RANK
    if _MPU_RANK is not None:
        return _MPU_RANK
    if dist.is_initialized():
        return dist.get_rank()
    return 0


# ===========================================================================
# M447: Megatron 5c45db4a7 — Initial implementation of pipelined text
#       generation
# ===========================================================================
#
# Upstream sources:
#   megatron/mpu/__init__.py    → deepspeed/compile/mpu_initialize.py
#   megatron/mpu/initialize.py  → deepspeed/compile/mpu_initialize.py
#
# Changes ported:
#
#   mpu/__init__.py:
#     - Replace 'from .initialize import get_pipeline_model_parallel_src_rank'
#       with two new exports:
#         'from .initialize import get_pipeline_model_parallel_first_rank'
#         'from .initialize import get_pipeline_model_parallel_last_rank'
#
#   mpu/initialize.py:
#     1. Add module-level _PIPELINE_GLOBAL_RANKS = None to track the ordered
#        list of global ranks for the current process's pipeline group.
#     2. In initialize_model_parallel(): declare global _PIPELINE_GLOBAL_RANKS;
#        when a process joins a pipeline group assign _PIPELINE_GLOBAL_RANKS = ranks.
#     3. Remove get_pipeline_model_parallel_src_rank() (old formula using
#        global_world_size // local_world_size, which was incorrect for
#        heterogeneous topologies).
#     4. Add get_pipeline_model_parallel_first_rank(): returns
#        _PIPELINE_GLOBAL_RANKS[0].
#     5. Add get_pipeline_model_parallel_last_rank(): returns
#        _PIPELINE_GLOBAL_RANKS[last_rank_local] where last_rank_local =
#        get_pipeline_model_parallel_world_size() - 1.
#     6. Update docstring of get_tensor_model_parallel_src_rank() to say
#        "first local rank" instead of "local rank".
#
# DeepSpeed adaptation:
#   Neuron_SP does not replicate the full process-group init (no NCCL groups
#   here), so _PIPELINE_GLOBAL_RANKS is managed as a module-level variable
#   that callers set via set_pipeline_global_ranks().  The two new accessor
#   functions are added verbatim from upstream.  The old
#   get_pipeline_model_parallel_src_rank() is retained as a deprecated shim
#   that logs a warning, preserving backward compatibility with any callers
#   in the wider DeepSpeed codebase that have not yet been updated.
# ===========================================================================

print('[M447]')

_PIPELINE_GLOBAL_RANKS = None


def set_pipeline_global_ranks(ranks):
    """Store the ordered pipeline-group global rank list for this process.

    Megatron 5c45db4a7 mpu/initialize.py — _PIPELINE_GLOBAL_RANKS is
    populated inside initialize_model_parallel() when the process joins
    its pipeline group.  In the DeepSpeed mapping callers invoke this
    setter after group construction.
    """
    global _PIPELINE_GLOBAL_RANKS
    _PIPELINE_GLOBAL_RANKS = list(ranks)
    print(f'[M447] set_pipeline_global_ranks: ranks={_PIPELINE_GLOBAL_RANKS}')


def get_pipeline_model_parallel_first_rank():
    """Return the global rank of the first stage in this pipeline group.

    Megatron 5c45db4a7 mpu/initialize.py — replaces the old
    get_pipeline_model_parallel_src_rank() which used a modular-arithmetic
    approximation.  The new implementation reads directly from the stored
    _PIPELINE_GLOBAL_RANKS list, which is accurate for all topologies.
    """
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "Pipeline parallel group is not initialized"
    return _PIPELINE_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_last_rank():
    """Return the global rank of the last stage in this pipeline group.

    Megatron 5c45db4a7 mpu/initialize.py — new function introduced
    alongside get_pipeline_model_parallel_first_rank() to support
    pipelined text generation where the last stage must broadcast newly
    sampled tokens back to the first stage via the embedding group.
    """
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "Pipeline parallel group is not initialized"
    # Downstream callers supply get_pipeline_model_parallel_world_size();
    # here we derive it from the stored rank list length.
    last_rank_local = len(_PIPELINE_GLOBAL_RANKS) - 1
    return _PIPELINE_GLOBAL_RANKS[last_rank_local]


def get_pipeline_model_parallel_src_rank():
    """[DEPRECATED] Use get_pipeline_model_parallel_first_rank() instead.

    Megatron 5c45db4a7 removed this function and replaced it with
    get_pipeline_model_parallel_first_rank() / _last_rank().  Retained
    here as a backward-compatibility shim so existing DeepSpeed callers
    are not broken before they migrate.
    """
    import warnings
    warnings.warn(
        "get_pipeline_model_parallel_src_rank() is deprecated as of M447 "
        "(Megatron 5c45db4a7).  Use get_pipeline_model_parallel_first_rank() "
        "instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if _PIPELINE_GLOBAL_RANKS is not None:
        return _PIPELINE_GLOBAL_RANKS[0]
    # Fallback: reproduce the old modular-arithmetic formula.
    if dist.is_initialized():
        global_rank = dist.get_rank()
        global_world_size = dist.get_world_size()
        local_world_size = get_model_parallel_world_size()
        return global_rank % (global_world_size // local_world_size)
    return 0
