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

# ===========================================================================
# M556: Megatron dd8890626 — Interleaved pipeline execution and code refactoring
# ===========================================================================
#
# Upstream source: megatron/mpu/initialize.py  → deepspeed/compile/mpu_initialize.py
#
# Changes ported:
#
#   1. Add module-level globals:
#        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
#        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
#
#   2. initialize_model_parallel(): accept virtual_pipeline_model_parallel_size_
#      parameter; when not None, set the two new globals to 0 and the given size.
#
#   3. is_pipeline_first_stage(ignore_virtual=False):
#        When ignore_virtual is False and WORLD_SIZE is set and RANK != 0:
#        return False before checking physical rank.
#
#   4. is_pipeline_last_stage(ignore_virtual=False):
#        When ignore_virtual is False and WORLD_SIZE is set and RANK != last:
#        return False before checking physical rank.
#
#   5. Add get_virtual_pipeline_model_parallel_rank(): returns the global.
#   6. Add set_virtual_pipeline_model_parallel_rank(rank): sets the global.
#
#   mpu/__init__.py:
#     from .initialize import get_virtual_pipeline_model_parallel_rank, \
#                             set_virtual_pipeline_model_parallel_rank
#
# 20% adaptation: DeepSpeed does not replicate full process-group init here;
# virtual rank management is standalone (no NCCL group construction).
# is_pipeline_first/last_stage stubs refer to the same physical-rank checks
# already present via get_pipeline_model_parallel_rank() / world_size.
# Adds print('[M556]').
# ===========================================================================

print('[M556]')

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None


def init_virtual_pipeline_model_parallel(virtual_pipeline_model_parallel_size):
    """Initialise virtual pipeline parallel globals.

    Megatron dd8890626 mpu/initialize.py — called inside
    initialize_model_parallel() when virtual_pipeline_model_parallel_size_ is
    not None; sets RANK to 0 and WORLD_SIZE to the given value.
    """
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size
    print(f'[M556] init_virtual_pipeline_model_parallel: '
          f'world_size={virtual_pipeline_model_parallel_size}')


def get_virtual_pipeline_model_parallel_rank():
    """Return the virtual pipeline-parallel rank.

    Megatron dd8890626 mpu/initialize.py — getter for
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK; returns None when virtual
    pipelining is not enabled.
    """
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK


def set_virtual_pipeline_model_parallel_rank(rank):
    """Set the virtual pipeline-parallel rank.

    Megatron dd8890626 mpu/initialize.py — called by the schedule helpers
    (schedules.py) before each virtual-stage forward/backward pass so that
    is_pipeline_first_stage() / is_pipeline_last_stage() return the correct
    answer for the current virtual chunk.
    """
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank


def is_pipeline_first_stage(ignore_virtual=False):
    """Return True if in the first pipeline model-parallel stage.

    Megatron dd8890626 mpu/initialize.py — extended with ignore_virtual:
      When ignore_virtual is False and virtual pipelining is active,
      also check that VIRTUAL_RANK == 0; otherwise return False early.
    """
    if not ignore_virtual:
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        if _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None and \
                _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK != 0:
            return False
    # Physical-rank check (DeepSpeed: use dist.get_rank() as proxy).
    if dist.is_initialized():
        from deepspeed.compile.mpu_initialize import get_model_parallel_rank
        return get_model_parallel_rank() == 0
    return True


def is_pipeline_last_stage(ignore_virtual=False):
    """Return True if in the last pipeline model-parallel stage.

    Megatron dd8890626 mpu/initialize.py — extended with ignore_virtual:
      When ignore_virtual is False and virtual pipelining is active,
      also check that VIRTUAL_RANK == (WORLD_SIZE - 1); otherwise return
      False early.
    """
    if not ignore_virtual:
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        if _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None and \
                _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK != (
                    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE - 1):
            return False
    # Physical-rank check.
    if dist.is_initialized():
        from deepspeed.compile.mpu_initialize import (
            get_model_parallel_rank, get_model_parallel_world_size)
        return get_model_parallel_rank() == (get_model_parallel_world_size() - 1)
    return True


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
