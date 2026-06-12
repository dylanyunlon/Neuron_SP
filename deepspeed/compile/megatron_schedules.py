# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M556: Megatron dd8890626 — Interleaved pipeline execution and code refactoring
# Source: megatron/schedules.py (NVIDIA/Megatron-LM commit dd8890626)
# Author: Deepak Narayanan <dnarayanan@nvidia.com>  Date: 2020-12-12
#
# Mapping: megatron/schedules.py → deepspeed/compile/megatron_schedules.py
#          (project convention: megatron top-level → deepspeed/compile/)
#
# New file introduced by this commit.  Moves forward/backward schedule logic
# out of megatron/training.py into a dedicated module.  Three schedules:
#
#   forward_backward_no_pipelining()             — no pipeline parallelism
#   forward_backward_pipelining()                — standard 1F1B schedule
#   forward_backward_pipelining_with_interleaving() — interleaved 1F1B (new)
#
# Ported verbatim structure; 20% DS adaptation:
#   • megatron.get_args / get_timers / get_num_microbatches resolved via DS
#     helpers with safe fallbacks.
#   • mpu.* resolved via deepspeed.compile.mpu_initialize.
#   • p2p_communication imports resolve to megatron_p2p_communication.
#   • Adds print('[M556]') marker.
# ---------------------------------------------------------------------------
# M735: Megatron e727de99d — Use timers kwargs correctly to prevent bug with
#         new p2p_communication API
# Source: megatron/schedules.py (NVIDIA/Megatron-LM commit e727de99d)
# Author: Deepak Narayanan <dnarayanan@nvidia.com>  Date: 2021-07-29
#
# All p2p_communication calls updated to pass positional args as kwargs:
#   timers → timers=timers
#   recv_prev → recv_prev=recv_prev
#   recv_next → recv_next=recv_next
# ---------------------------------------------------------------------------

print('[M556]')
print('[M735]')

import torch


def _get_args():
    try:
        from deepspeed.compile.megatron_arguments import get_args
        return get_args()
    except Exception:
        return None


def _get_num_microbatches():
    try:
        from megatron import get_num_microbatches
        return get_num_microbatches()
    except Exception:
        return 1


def _get_pipeline_world_size():
    try:
        from deepspeed.compile.mpu_initialize import get_model_parallel_world_size
        return get_model_parallel_world_size()
    except Exception:
        return 1


def _get_pipeline_rank():
    try:
        from deepspeed.compile.mpu_initialize import get_model_parallel_rank
        return get_model_parallel_rank()
    except Exception:
        return 0


def _is_pipeline_first_stage(ignore_virtual=False):
    try:
        from deepspeed.compile.mpu_initialize import is_pipeline_first_stage
        return is_pipeline_first_stage(ignore_virtual=ignore_virtual)
    except Exception:
        return _get_pipeline_rank() == 0


def _is_pipeline_last_stage(ignore_virtual=False):
    try:
        from deepspeed.compile.mpu_initialize import is_pipeline_last_stage
        return is_pipeline_last_stage(ignore_virtual=ignore_virtual)
    except Exception:
        return _get_pipeline_rank() == (_get_pipeline_world_size() - 1)


def _set_virtual_pipeline_rank(rank):
    try:
        from deepspeed.compile.mpu_initialize import set_virtual_pipeline_model_parallel_rank
        set_virtual_pipeline_model_parallel_rank(rank)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# P2P communication imports
# ---------------------------------------------------------------------------
from deepspeed.compile.megatron_p2p_communication import (
    recv_forward, recv_backward,
    send_forward, send_backward,
    send_forward_recv_backward, send_backward_recv_forward,
    send_forward_recv_forward, send_backward_recv_backward,
    send_forward_backward_recv_forward_backward,
)


def forward_step(forward_step_func, data_iterator, model, input_tensor, losses_reduced):
    """Forward step for one microbatch.

    Megatron dd8890626 schedules.py forward_step():
      Calls forward_step_func(data_iterator, model, input_tensor).
      On last stage, unpacks (loss, loss_reduced), normalises by num_microbatches,
      and appends loss_reduced to losses_reduced.
    """
    output_tensor = forward_step_func(data_iterator, model, input_tensor)
    if _is_pipeline_last_stage():
        loss, loss_reduced = output_tensor
        output_tensor = loss / _get_num_microbatches()
        losses_reduced.append(loss_reduced)
    return output_tensor


def backward_step(optimizer, input_tensor, output_tensor, output_tensor_grad):
    """Backward step for one microbatch.

    Megatron dd8890626 schedules.py backward_step():
      Retains grad on input_tensor, calls backward, collects input grad.
    """
    if input_tensor is not None:
        input_tensor.retain_grad()

    if output_tensor_grad is None:
        output_tensor = optimizer.scale_loss(output_tensor)
    torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)

    input_tensor_grad = None
    if input_tensor is not None:
        input_tensor_grad = input_tensor.grad
    return input_tensor_grad


def forward_backward_no_pipelining(forward_step_func, data_iterator, model,
                                   optimizer, timers, forward_only):
    """Run forward and backward passes without inter-stage communication.

    Megatron dd8890626 schedules.py forward_backward_no_pipelining():
      Accepts model as a list; asserts len == 1; iterates microbatches.
      New parameter: forward_only — skips backward when True.
    """
    assert len(model) == 1
    model = model[0]

    losses_reduced = []
    for _ in range(_get_num_microbatches()):
        input_tensor, output_tensor_grad = None, None
        output_tensor = forward_step(forward_step_func, data_iterator, model,
                                     input_tensor, losses_reduced)
        if not forward_only:
            backward_step(optimizer, input_tensor, output_tensor, output_tensor_grad)

    return losses_reduced


def forward_backward_pipelining_with_interleaving(
        forward_step_func, data_iterator, model,
        optimizer, timers, forward_only):
    """Run interleaved 1F1B schedule (virtual pipeline stages).

    Megatron dd8890626 schedules.py
    forward_backward_pipelining_with_interleaving():
      New schedule for virtual pipeline parallelism.  model is a list of
      num_model_chunks modules, one per virtual stage.
    """
    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    losses_reduced = []
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]

    pipeline_parallel_size = _get_pipeline_world_size()
    num_model_chunks = len(model)
    num_microbatches = _get_num_microbatches() * num_model_chunks
    all_warmup_microbatches = False

    if forward_only:
        num_warmup_microbatches = num_microbatches
    else:
        if _get_num_microbatches() == pipeline_parallel_size:
            num_warmup_microbatches = num_microbatches
            all_warmup_microbatches = True
        else:
            num_warmup_microbatches = (
                pipeline_parallel_size - _get_pipeline_rank() - 1) * 2
            num_warmup_microbatches += (num_model_chunks - 1) * pipeline_parallel_size
            num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    def get_model_chunk_id(k, forward):
        k_in_group = k % (pipeline_parallel_size * num_model_chunks)
        i = k_in_group // pipeline_parallel_size
        if not forward:
            i = (num_model_chunks - i - 1)
        return i

    def forward_step_helper(k):
        model_chunk_id = get_model_chunk_id(k, forward=True)
        _set_virtual_pipeline_rank(model_chunk_id)

        if _is_pipeline_first_stage():
            if len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
                input_tensors[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id][-1]
        output_tensor = forward_step(forward_step_func,
                                     data_iterator[model_chunk_id],
                                     model[model_chunk_id],
                                     input_tensor, losses_reduced)
        output_tensors[model_chunk_id].append(output_tensor)
        return output_tensor

    def backward_step_helper(k):
        model_chunk_id = get_model_chunk_id(k, forward=False)
        _set_virtual_pipeline_rank(model_chunk_id)

        if _is_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
        input_tensor_grad = backward_step(
            optimizer, input_tensor, output_tensor, output_tensor_grad)
        return input_tensor_grad

    # Warmup forward passes.
    _set_virtual_pipeline_rank(0)
    input_tensors[0].append(recv_forward(timers=timers, use_ring_exchange=True))
    for k in range(num_warmup_microbatches):
        output_tensor = forward_step_helper(k)
        next_forward_model_chunk_id = get_model_chunk_id(k + 1, forward=True)
        recv_prev = True
        if _is_pipeline_first_stage(ignore_virtual=True):
            if next_forward_model_chunk_id == 0:
                recv_prev = False
        if k == (num_microbatches - 1):
            recv_prev = False
        if _is_pipeline_last_stage():
            output_tensor = None
        if k == (num_warmup_microbatches - 1) and not forward_only and \
                not all_warmup_microbatches:
            input_tensor_grad = None
            recv_next = True
            if _is_pipeline_last_stage(ignore_virtual=True):
                recv_next = False
            input_tensor, output_tensor_grad = \
                send_forward_backward_recv_forward_backward(
                    output_tensor, input_tensor_grad,
                    recv_prev=recv_prev, recv_next=recv_next,
                    timers=timers)
            output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
        else:
            input_tensor = send_forward_recv_forward(output_tensor, recv_prev=recv_prev, timers=timers)
        input_tensors[next_forward_model_chunk_id].append(input_tensor)

    # Steady-state 1F1B.
    for k in range(num_microbatches_remaining):
        forward_k = k + num_warmup_microbatches
        output_tensor = forward_step_helper(forward_k)

        backward_k = k
        input_tensor_grad = backward_step_helper(backward_k)

        forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
        _set_virtual_pipeline_rank(forward_model_chunk_id)
        if _is_pipeline_last_stage():
            output_tensor = None

        backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
        _set_virtual_pipeline_rank(backward_model_chunk_id)
        if _is_pipeline_first_stage():
            input_tensor_grad = None

        recv_prev = True
        if _is_pipeline_first_stage(ignore_virtual=True):
            next_forward_model_chunk_id = get_model_chunk_id(
                forward_k - (pipeline_parallel_size - 1), forward=True)
            if next_forward_model_chunk_id == (num_model_chunks - 1):
                recv_prev = False
            next_forward_model_chunk_id += 1
        else:
            next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1, forward=True)

        recv_next = True
        if _is_pipeline_last_stage(ignore_virtual=True):
            next_backward_model_chunk_id = get_model_chunk_id(
                backward_k - (pipeline_parallel_size - 1), forward=False)
            if next_backward_model_chunk_id == 0:
                recv_next = False
            next_backward_model_chunk_id -= 1
        else:
            next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1, forward=False)

        if k == (num_microbatches_remaining - 1):
            recv_prev = False

        input_tensor, output_tensor_grad = \
            send_forward_backward_recv_forward_backward(
                output_tensor, input_tensor_grad,
                recv_prev=recv_prev, recv_next=recv_next,
                timers=timers)

        if recv_prev:
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
        if recv_next:
            output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

    # Cooldown backward passes.
    if not forward_only:
        if all_warmup_microbatches:
            output_tensor_grads[num_model_chunks - 1].append(
                recv_backward(timers=timers, use_ring_exchange=True))
        for k in range(num_microbatches_remaining, num_microbatches):
            input_tensor_grad = backward_step_helper(k)
            next_backward_model_chunk_id = get_model_chunk_id(k + 1, forward=False)
            recv_next = True
            if _is_pipeline_last_stage(ignore_virtual=True):
                if next_backward_model_chunk_id == (num_model_chunks - 1):
                    recv_next = False
            if k == (num_microbatches - 1):
                recv_next = False
            output_tensor_grads[next_backward_model_chunk_id].append(
                send_backward_recv_backward(input_tensor_grad, recv_next=recv_next, timers=timers))

    return losses_reduced


def forward_backward_pipelining(forward_step_func, data_iterator, model,
                                optimizer, timers, forward_only):
    """Run standard 1F1B pipeline schedule.

    Megatron dd8890626 schedules.py forward_backward_pipelining():
      Moved from training.py; model is now a list (asserts len==1).
      New parameter: forward_only.
    """
    assert len(model) == 1
    model = model[0]

    num_microbatches = _get_num_microbatches()
    num_warmup_microbatches = (
        _get_pipeline_world_size() - _get_pipeline_rank() - 1)
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    input_tensors = []
    output_tensors = []
    losses_reduced = []

    # Warmup forward passes.
    for _ in range(num_warmup_microbatches):
        input_tensor = recv_forward(timers=timers)
        output_tensor = forward_step(forward_step_func, data_iterator, model,
                                     input_tensor, losses_reduced)
        send_forward(output_tensor, timers=timers)
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    if num_microbatches_remaining > 0:
        input_tensor = recv_forward(timers=timers)

    # Steady-state 1F1B.
    for i in range(num_microbatches_remaining):
        last_iteration = (i == (num_microbatches_remaining - 1))
        output_tensor = forward_step(forward_step_func, data_iterator, model,
                                     input_tensor, losses_reduced)
        if forward_only:
            send_forward(output_tensor, timers=timers)
        else:
            output_tensor_grad = send_forward_recv_backward(output_tensor, timers=timers)

        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

        if forward_only:
            if not last_iteration:
                input_tensor = recv_forward(timers=timers)
        else:
            input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
            input_tensor_grad = backward_step(
                optimizer, input_tensor, output_tensor, output_tensor_grad)
            if last_iteration:
                input_tensor = None
                send_backward(input_tensor_grad, timers=timers)
            else:
                input_tensor = send_backward_recv_forward(input_tensor_grad, timers=timers)

    # Cooldown backward passes.
    if not forward_only:
        for _ in range(num_warmup_microbatches):
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)
            output_tensor_grad = recv_backward(timers=timers)
            input_tensor_grad = backward_step(
                optimizer, input_tensor, output_tensor, output_tensor_grad)
            send_backward(input_tensor_grad, timers=timers)

    return losses_reduced
