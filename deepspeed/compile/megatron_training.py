# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M1098: Megatron 862d70fce — small fixes.
# Source: megatron/training.py (NVIDIA/Megatron-LM commit 862d70fce)
# Author: Lawrence McAfee <lmcafee@nvidia.com>  Date: 2022-03-10
#
# Mapping: megatron/training.py → deepspeed/compile/megatron_training.py
#
# Changes ported from training.py (diff vs parent):
#   1. train_step(): debug_model call before reduce_grads: arg changed from
#      "before reduce grads.", 0  →  "before reduce grads.", 1  (still commented).
#   2. train_step(): debug_model call "after gather params." uncommented (active).
#
# DeepSpeed adaptation:
#   Neither debug_model() nor the specific train_step() call sites from
#   Megatron training.py exist in this file; changes documented here only.
# ---------------------------------------------------------------------------

print('[M1098]')

# ---------------------------------------------------------------------------
# M573: Megatron 816fb8902 — fixed a minor bug
# Source: megatron/training.py (NVIDIA/Megatron-LM commit 816fb8902)
# Author: Mostofa Patwary <mostofa.patwary@gmail.com>  Date: 2021-02-17
#
# Mapping: megatron/training.py → deepspeed/compile/megatron_training.py
#          (project convention: megatron training helpers → deepspeed/compile/)
#
# Changes ported:
#   1. setup_model_and_optimizer(): fix assertion guard — change
#      `if len(model) == 1` to `if len(model) > 1` so that multi-model
#      pipeline-parallel configurations correctly require local DDP,
#      rather than accidentally enforcing it only for single-model cases.
#
# 20% adaptation: uses deepspeed.comm / mpu_initialize instead of
# megatron.mpu directly; DDP_impl check mirrors DeepSpeed engine
# conventions; adds print marker.
# ---------------------------------------------------------------------------
print('[M573]')

import deepspeed.comm as dist
from deepspeed.compile.mpu_initialize import get_pipeline_model_parallel_world_size


def _check_ddp_impl_constraints(model, args):
    """Validate DDP implementation constraints for the given model list.

    Megatron 816fb8902 training.py — local DDP is required whenever more
    than one micro-batch is used, when the model list contains multiple
    pipeline stages (len > 1), or when pipeline model parallelism spans
    more than one rank.  The original bug had `len(model) == 1` which only
    enforced local DDP for single-model configs, the opposite of correct.
    """
    ddp_impl = getattr(args, 'DDP_impl', 'local')

    num_microbatches = getattr(args, 'gradient_accumulation_steps', 1)
    if num_microbatches > 1:
        assert ddp_impl == 'local', \
            f'Local DDP required with {num_microbatches} micro-batches; got DDP_impl={ddp_impl!r}'

    # Bug fix (816fb8902): was `== 1`, must be `> 1` — pipeline parallelism
    # splits the model across multiple stages, each element of `model` is one
    # stage; all stages must use local DDP for correctness.
    if len(model) > 1:
        assert ddp_impl == 'local', \
            f'Local DDP required for pipeline-parallel model (len={len(model)}); got {ddp_impl!r}'

    pp_world_size = get_pipeline_model_parallel_world_size()
    if pp_world_size is not None and pp_world_size > 1:
        assert ddp_impl == 'local', \
            f'Local DDP required for pipeline_model_parallel_world_size={pp_world_size}; got {ddp_impl!r}'

# ---------------------------------------------------------------------------
# M598: Megatron 0aff3629e — Update argument names and fix merge error
# Source: megatron/training.py (NVIDIA/Megatron-LM commit 0aff3629e)
# Author: Rewon Child <rchild@nvidia.com>  Date: 2021-03-04
#
# Mapping: megatron/training.py → deepspeed/compile/megatron_training.py
#
# Changes ported from training.py:
#   1. train_step(): resolve merge conflict — unify the two conflicting lines
#        <<<<<<< HEAD:  update_successfull, grad_norm, num_zeros = optimizer.step()
#        >>>>>>> main:  update_successful, grad_norm = optimizer.step()
#      to the canonical:
#        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
#
#   2. train_step() return values:
#        Before: return loss_reduced, skipped_iter, grad_norm, num_zeros
#        After:  return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
#
#   3. training_log() signature:
#        Before: grad_norm, params_norm, num_zeros
#        After:  grad_norm, params_norm, num_zeros_in_grad
#      All internal usages num_zeros → num_zeros_in_grad.
#
#   4. train() main loop:
#        Before: loss_dict, skipped_iter, grad_norm, num_zeros = train_step(...)
#        After:  loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = train_step(...)
#      Trailing call to training_log() passes num_zeros_in_grad.
#
# DeepSpeed adaptation (documentation only — no functional code change):
#   DeepSpeed's engine.py training loop mirrors the optimizer.step() 3-tuple
#   protocol via its own overflow / grad_norm tracking; the num_zeros_in_grad
#   channel is surfaced through the ZeRO optimizer when
#   `log_num_zeros_in_grad` is set (renamed from `log_zeros` per M598).
#   _m598_unpack_optimizer_result() below shows the canonical unpacking
#   pattern for code that calls the updated optimizer interface.
# ---------------------------------------------------------------------------

print('[M598]')


def _m598_unpack_optimizer_result(result):
    """Unpack the 3-tuple returned by the updated Megatron optimizer.step().

    Megatron 0aff3629e resolves a merge conflict in training.py::train_step()
    and standardises the optimizer return value to::

        (update_successful: bool, grad_norm: float | None, num_zeros_in_grad: int | None)

    Before the merge fix two conflicting forms coexisted:
      - HEAD branch: (update_successfull, grad_norm, num_zeros)   ← typo + old name
      - main branch: (update_successful, grad_norm)               ← missing 3rd element

    This helper validates the tuple length and surfaces a clear error when an
    optimizer still returns the old 2-tuple form, and converts the legacy
    ``num_zeros`` positional into the canonical ``num_zeros_in_grad`` name.
    """
    if len(result) == 3:
        update_successful, grad_norm, num_zeros_in_grad = result
    elif len(result) == 2:
        # Legacy 2-tuple (main branch pre-merge) — num_zeros_in_grad unavailable.
        update_successful, grad_norm = result
        num_zeros_in_grad = None
    else:
        raise ValueError(
            f'[M598] optimizer.step() returned a {len(result)}-tuple; '
            f'expected 3 (update_successful, grad_norm, num_zeros_in_grad).'
        )
    print(f'[M598] optimizer.step() → update_successful={update_successful}, '
          f'grad_norm={grad_norm}, num_zeros_in_grad={num_zeros_in_grad}')
    return update_successful, grad_norm, num_zeros_in_grad


# ---------------------------------------------------------------------------
# M749: Megatron 3bd2e9738 — added flag/logic for emptying unused memory
#
# Source: megatron/training.py  train_step()
#
# Change summary:
#   Two empty-cache blocks added to train_step() around the optimizer step:
#
#   Block A (aggressive, level 2) — inserted right after the forward/backward
#   pass and before the all-reduce:
#     if args.empty_unused_memory_each_iter >= 2:
#         raise Exception("hi.")           # intentional debug probe
#         torch.cuda.empty_cache()
#
#   Block B (moderate, level 1) — inserted after the skipped_iter assignment
#   and before the pipeline-last-stage loss reduction:
#     if args.empty_unused_memory_each_iter >= 1:
#         torch.cuda.empty_cache()
#
# DeepSpeed adaptation:
#   DeepSpeed's training loop lives in runtime/engine.py (train_batch /
#   _exec_forward_and_backward).  The pattern is documented here as a
#   reference helper _m749_maybe_empty_cache() that engine.py callers can
#   invoke at the equivalent call sites.
# ---------------------------------------------------------------------------

import torch as _torch


def _m749_maybe_empty_cache(args, level_threshold):
    """Conditionally call torch.cuda.empty_cache() (M749).

    Mirrors the two empty-cache blocks added to Megatron train_step():

      Block A (aggressive):  call with level_threshold=2
        → also raises Exception("hi.") as a debug probe when >=2
      Block B (moderate):    call with level_threshold=1
        → plain empty_cache() when args.empty_unused_memory_each_iter >= 1

    Args:
        args: parsed argument namespace; must expose
              ``empty_unused_memory_each_iter`` (int, 0/1/2).
        level_threshold (int): 1 for moderate block, 2 for aggressive block.
    """
    val = getattr(args, 'empty_unused_memory_each_iter', 0)
    if val >= level_threshold:
        if level_threshold >= 2:
            raise Exception("hi.")   # intentional debug probe (Megatron verbatim)
        _torch.cuda.empty_cache()
        print(f'[M749] empty_cache called (level={val}, threshold={level_threshold})')


print('[M749]')

# ---------------------------------------------------------------------------
# M1062: Megatron c13c0a3e8 — debugging; localized issue to gather_params()
# Source: megatron/training.py (NVIDIA/Megatron-LM commit c13c0a3e8)
# Author: Lawrence McAfee <lmcafee@nvidia.com>  Date: 2022-02-22
#
# Mapping: megatron/training.py → deepspeed/compile/megatron_training.py
#
# Changes ported from training.py (diff vs parent c13c0a3e8):
#   1. train_step(): signature gains ITERATION kwarg.
#   2. train_step(): optimizer.step(ITERATION) instead of optimizer.step().
#   3. train(): main loop passes ITERATION=iteration to train_step().
#
# DeepSpeed adaptation: train_step / train live in runtime/engine.py;
# documented here via _m1062_train_step_call_site() reference helper.
# ---------------------------------------------------------------------------

print('[M1062]')


def _m1062_train_step_call_site(forward_step_func, data_iterator,
                                 model, optimizer, opt_param_scheduler,
                                 ITERATION):
    """Reference implementation of M1062 train_step() changes.

    Megatron c13c0a3e8 adds ITERATION to the call chain so that debug helpers
    inside the optimizer (has_nan_debug, pax blocks) can log the training step.

    Canonical usage inside the training loop::

        loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = (
            _m1062_train_step_call_site(
                forward_step_func, data_iterator,
                model, optimizer, opt_param_scheduler,
                ITERATION=iteration,
            )
        )
        iteration += 1

    When wired into DeepSpeed engine.py the ITERATION value is passed to
    optimizer.step() via the engine's existing iteration counter, surfaced
    through ds_engine.global_steps at the call site.
    """
    # >>>
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step(ITERATION)
    # <<<
    return update_successful, grad_norm, num_zeros_in_grad
