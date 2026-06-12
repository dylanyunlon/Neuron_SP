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


# ---------------------------------------------------------------------------
# M1003: Megatron bea16fa33 — found root source of t5 issue (fast layer norm)
# Source: megatron/mpu/random.py (NVIDIA/Megatron-LM commit bea16fa33)
# Author: Lawrence McAfee <lmcafee@nvidia.com>  Date: 2022-02-01
#
# Mapping: megatron/mpu/random.py → deepspeed/compile/mpu_initialize.py
#
# assert_viewless_tensor() is called from megatron/schedules.py in this
# commit to guard that forward_step output tensors are not views.
# The implementation lives in mpu/random.py upstream; we place it here
# since this file is the DS equivalent of the mpu package namespace.
# ---------------------------------------------------------------------------

print('[M1003]')

# ---------------------------------------------------------------------------
# M1038: Megatron 1cd3650dc — more minor fixes
# Source: megatron/mpu/random.py, megatron/mpu/__init__.py,
#         megatron/mpu/initialize.py, megatron/mpu/layers.py,
#         megatron/arguments.py, megatron/model/transformer.py,
#         megatron/p2p_communication.py, megatron/schedules.py,
#         megatron/text_generation/api.py, megatron/text_generation/generation.py,
#         megatron/text_generation_server.py, megatron/training.py,
#         megatron/static/index.html, README.md,
#         examples/msdp/prep_resp_gen.sh
#         (NVIDIA/Megatron-LM commit 1cd3650dc)
# Author: Vijay Korthikanti <vkorthikanti@nvidia.com>  Date: 2022-02-01
#
# Mapping:
#   megatron/mpu/random.py     → deepspeed/compile/mpu_initialize.py
#   megatron/mpu/__init__.py   → deepspeed/compile/mpu_initialize.py
#   megatron/mpu/initialize.py → deepspeed/compile/mpu_initialize.py
#   megatron/arguments.py      → deepspeed/compile/megatron_arguments.py
#   megatron/mpu/layers.py     → deepspeed/compile/mpu_layers.py
#   megatron/p2p_communication.py → deepspeed/compile/megatron_p2p_communication.py
#   megatron/schedules.py      → deepspeed/compile/megatron_schedules.py
#   megatron/text_generation/* → deepspeed/compile/megatron_generation.py
#   megatron/static/index.html → no equivalent (skip)
#   README.md / prep_resp_gen.sh → documentation only (skip)
#
# Changes ported:
#
# mpu/random.py:
#   1. _kernel_make_viewless_tensor(inp, requires_grad): creates a new tensor
#      sharing inp.data but not linked via ._base, preventing memory leaks.
#   2. MakeViewlessTensor autograd Function: wraps (1) for graph-propagating use.
#   3. make_viewless_tensor(inp, requires_grad, keep_graph): entry-point; returns
#      inp as-is when ._base is None; otherwise dispatches to (2) or (1).
#   4. safely_set_viewless_tensor_data(tensor, new_data_tensor): asserts viewless
#      then sets .data.
#   CheckpointFunction: two .data assignments replaced with
#      safely_set_viewless_tensor_data() calls.
#
# mpu/__init__.py:
#   Exports make_viewless_tensor, assert_viewless_tensor,
#   safely_set_viewless_tensor_data from mpu namespace.
#
# mpu/initialize.py get_num_layers():
#   standalone_embedding_stage support: when True and pipeline rank == 0,
#   num_layers = 0 (NoopTransformerLayer assigned by transformer.py).
#   transformer_pipeline_model_parallel_size used for divisibility checks.
#
# mpu/layers.py ColumnParallelLinear docstring:
#   Typo fix: "all-gether" → "all-gather".
#
# arguments.py parse_args():
#   transformer_pipeline_model_parallel_size = pipeline_model_parallel_size - 1
#   if standalone_embedding_stage else pipeline_model_parallel_size.
#   virtual_pipeline_model_parallel_size derivation uses new field.
#   _add_distributed_args(): --deallocate-pipeline-outputs removed;
#   --standalone-embedding-stage added (places input embedding on its own stage).
#
# model/transformer.py:
#   NoopTransformerLayer: new class for standalone_embedding_stage zero-layer ranks.
#   ParallelTransformer: num_layers==0 branch uses NoopTransformerLayer.
#   ParallelTransformer.forward(): make_viewless_tensor applied to hidden_states;
#   encoder_output indent fix; trailing whitespace removed.
#   DropPath exported from transformer.py (imported by esvit_swin_backbone.py).
#   (M1004 already covered the esvit_swin_backbone DropPath import change.)
#
# p2p_communication.py _communicate():
#   After scatter_gather, tensor_recv_prev and tensor_recv_next each wrapped in
#   mpu.make_viewless_tensor(requires_grad=True, keep_graph=False).
#
# schedules.py:
#   backward_step(): unconditionally calls custom_backward() — removes
#     args.deallocate_pipeline_outputs guard; refactored in M971/M972.
#   deallocate_output_tensor() already present (M971); this commit just
#   updates call sites in warmup/steady-state loops (already covered by M971).
#
# text_generation/api.py generate_and_post_process() / generate():
#   New params: stop_on_double_eol=False, stop_on_eol=False, random_seed=-1.
#   broadcast_float_list size: 7 → 10; random_seed sets torch.random.manual_seed.
#
# text_generation/generation.py generate_tokens_probs_and_return_on_first_stage():
#   New params: stop_on_double_eol, stop_on_eol.
#   Context-too-large guard: raises ValueError if min_prompt_length >= max_seq_len.
#   Done-token logic: optional double-eol / eol based stopping (token IDs 628/198).
#
# text_generation_server.py:
#   Removes unconditional request-log prints; adds no_log flag.
#   Handles stop_on_double_eol, stop_on_eol, random_seed, add_BOS empty-prompt guard.
#   generate_and_post_process() call wrapped in try/except ValueError.
#
# training.py: remove blank line before dl_type assert (cosmetic).
#
# megatron/static/index.html: new Megatron web-UI file; no equivalent in DS.
#
# README.md: prose improvements; adds TP/PP column note; typo fixes.
# examples/msdp/prep_resp_gen.sh: --knowledge_gen_file → --knwl_gen_file.
#
# DS adaptation notes:
#   • make_viewless_tensor / safely_set_viewless_tensor_data implemented below.
#   • standalone_embedding_stage / transformer_pipeline_model_parallel_size
#     added as set_standalone_embedding_args() helper in megatron_arguments.py.
#   • mpu_layers.py: typo documented (no functional code, comment only).
#   • megatron_p2p_communication.py: make_viewless_tensor call noted in comment.
#   • text_generation / text_generation_server changes: documented as no-op
#     (DS text generation path differs); see megatron_generation.py comment.
#   • static/index.html, README, prep_resp_gen.sh: no-op.
# ---------------------------------------------------------------------------
print('[M1038]')


def assert_viewless_tensor(tensor, extra_msg=None):
    """Assert that a tensor is not a view (its ._base field is None).

    Megatron bea16fa33 mpu/random.py — raised by forward_step to catch
    memory leaks where a tensor view is stored to a buffer, preventing GC.
    Accepts lists (recurses) and non-Tensor objects (no-op).
    """
    import torch
    if isinstance(tensor, list):
        [assert_viewless_tensor(t) for t in tensor]
        return tensor
    if not isinstance(tensor, torch.Tensor):
        return tensor
    assert tensor._base is None, (
        "Ensure tensor._base is None before setting tensor.data or storing "
        "tensor to memory buffer. Otherwise, a memory leak will occur (and "
        "likely accumulate over iterations). %s"
    ) % extra_msg
    return tensor


# ---------------------------------------------------------------------------
# M1038: make_viewless_tensor / safely_set_viewless_tensor_data
# Source: megatron/mpu/random.py (NVIDIA/Megatron-LM commit 1cd3650dc)
# Placed here (mpu namespace) matching upstream mpu/__init__.py export.
# ---------------------------------------------------------------------------

import torch as _torch


def _kernel_make_viewless_tensor(inp, requires_grad):
    '''Make a viewless tensor.

    View tensors have the undesirable side-effect of retaining a reference
    to the originally-viewed tensor, even after manually setting the '.data'
    field.  This method creates a new tensor that links to the old tensor's
    data, without linking the viewed tensor referenced via the '._base' field.

    M1038: Megatron 1cd3650dc mpu/random.py.
    '''
    out = _torch.empty(
        (1,),
        dtype=inp.dtype,
        device=inp.device,
        requires_grad=requires_grad,
    )
    out.data = inp.data
    return out


class _MakeViewlessTensor(_torch.autograd.Function):
    '''Autograd Function to make a viewless tensor (graph-propagating).

    M1038: Megatron 1cd3650dc mpu/random.py MakeViewlessTensor.
    Use via make_viewless_tensor(keep_graph=True).
    '''
    @staticmethod
    def forward(ctx, inp, requires_grad):
        return _kernel_make_viewless_tensor(inp, requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def make_viewless_tensor(inp, requires_grad, keep_graph):
    '''Entry-point for creating viewless tensors.

    Returns inp as-is when ._base is None (already viewless).
    Otherwise creates a new tensor via MakeViewlessTensor (keep_graph=True)
    or _kernel_make_viewless_tensor (keep_graph=False).

    M1038: Megatron 1cd3650dc mpu/random.py make_viewless_tensor().
    Also exported from mpu namespace (mpu/__init__.py in upstream).
    '''
    if inp._base is None:
        return inp
    if keep_graph:
        return _MakeViewlessTensor.apply(inp, requires_grad)
    else:
        return _kernel_make_viewless_tensor(inp, requires_grad)


def safely_set_viewless_tensor_data(tensor, new_data_tensor):
    '''Safely set tensor's .data field after asserting it is not a view.

    M1038: Megatron 1cd3650dc mpu/random.py safely_set_viewless_tensor_data().
    Used in CheckpointFunction to replace direct .data assignments:
      args[0].data = split_tensor_into_1d_equal_chunks(...)
      → safely_set_viewless_tensor_data(args[0], split_tensor_...)
    '''
    assert_viewless_tensor(
        tensor,
        extra_msg="FYI, tensor._base has shape %s, and new_data_tensor has "
                  "shape %s." % (
                      "--" if tensor._base is None else tensor._base.shape,
                      new_data_tensor.shape,
                  ),
    )
    tensor.data = new_data_tensor
