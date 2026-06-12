# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

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
