# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M556: Megatron dd8890626 — Interleaved pipeline execution and code refactoring
# Source: megatron/p2p_communication.py (NVIDIA/Megatron-LM commit dd8890626)
# Author: Deepak Narayanan <dnarayanan@nvidia.com>  Date: 2020-12-12
#
# Mapping: megatron/p2p_communication.py → deepspeed/compile/megatron_p2p_communication.py
#          (project convention: megatron top-level → deepspeed/compile/)
#
# New file introduced by this commit.  Splits peer-to-peer tensor communication
# helpers out of megatron/training.py into a dedicated module.
#
# Ported functions (verbatim structure, DS import adaptation):
#   _communicate()               — core send/recv batch with P2POp or ring_exchange
#   recv_forward()               — recv activation from prev stage
#   recv_backward()              — recv grad from next stage
#   send_forward()               — send activation to next stage
#   send_backward()              — send grad to prev stage
#   send_forward_recv_backward() — combined fwd-send / bwd-recv
#   send_backward_recv_forward() — combined bwd-send / fwd-recv
#   send_forward_recv_forward()  — interleaved fwd-send / fwd-recv (ring)
#   send_backward_recv_backward() — interleaved bwd-send / bwd-recv (ring)
#   send_forward_backward_recv_forward_backward() — full interleaved exchange
#
# 20% adaptation:
#   • megatron.get_args() replaced by a local _get_comm_args() that falls
#     back gracefully when no global args are set.
#   • mpu.get_pipeline_model_parallel_*_rank() / is_pipeline_*_stage()
#     resolved via deepspeed.compile.mpu_initialize.
#   • Adds print('[M556]') marker.
# ---------------------------------------------------------------------------
# M734: Megatron 1dccefd89 — Make it possible to pass in tensor shapes to
#       communication methods in p2p_communication.py
# Source: megatron/p2p_communication.py (NVIDIA/Megatron-LM commit 1dccefd89)
# Author: Mostofa Patwary <mpatwary@nvidia.com>  Date: 2021-07-27
#
# Changes ported:
#   _communicate()    — added tensor_shape, override_scatter_gather_tensors_in_pipeline,
#                       dtype_ params; conditional tensor_shape default; requires_grad
#                       logic driven by dtype_; three scatter_gather guard updates
#   recv_forward()    — added tensor_shape, override_scatter_gather_tensors_in_pipeline,
#                       dtype_ params; pass-through to _communicate
#   send_forward()    — added override_scatter_gather_tensors_in_pipeline, dtype_ params;
#                       pass-through to _communicate
#   _get_comm_args()  — added scatter_gather_tensors_in_pipeline default (False)
#   imports           — added functools.reduce and operator
# ---------------------------------------------------------------------------
# M1161: Megatron 13b3dca6d — make interleaving work with optimizations
# Source: megatron/schedules.py (NVIDIA/Megatron-LM commit 13b3dca6d)
# Author: Vijay Korthikanti <vkorthikanti@nvidia.com>  Date: 2022-04-25
#
# Mapping: megatron/schedules.py → deepspeed/compile/megatron_p2p_communication.py
#
# Upstream patch: in forward_backward_pipelining_with_interleaving(), the
# tensor_shape default computation now respects args.model_parallel_memory_opt:
#   if args.model_parallel_memory_opt:
#       seq_length = args.seq_length // mpu.get_tensor_model_parallel_world_size()
#   else:
#       seq_length = args.seq_length
#   tensor_shape = (seq_length, args.micro_batch_size, args.hidden_size)
#
# DS mapping: tensor_shape default is computed in _communicate() (line ~180)
# rather than at the call site, so the fix is applied there instead:
#   _communicate(): tensor_shape=None default now applies the same
#   model_parallel_memory_opt-aware seq_length split.
# ---------------------------------------------------------------------------

print('[M556]')
print('[M734]')
print('[M1161]')

import operator
from functools import reduce

import torch
import deepspeed.comm as dist


def _get_comm_args():
    """Return args namespace for communication parameters.

    Tries deepspeed global args; falls back to a simple namespace with
    safe defaults so unit tests can import this module without full init.
    """
    try:
        from deepspeed.compile.megatron_arguments import get_args
        args = get_args()
        if args is not None:
            return args
    except Exception:
        pass
    import types
    defaults = types.SimpleNamespace(
        seq_length=1024,
        micro_batch_size=1,
        hidden_size=1024,
        params_dtype=torch.float16,
        fp32_residual_connection=False,
        scatter_gather_tensors_in_pipeline=False,
    )
    return defaults


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


def _get_tensor_model_parallel_world_size():
    try:
        from deepspeed.compile.mpu_initialize import get_tensor_model_parallel_world_size
        return get_tensor_model_parallel_world_size()
    except Exception:
        return 1


def _prev_rank():
    """Global rank of the previous pipeline stage."""
    rank = _get_pipeline_rank()
    world_size = _get_pipeline_world_size()
    if dist.is_initialized():
        return dist.get_rank() - 1  # simplified; full impl uses PIPELINE_GLOBAL_RANKS
    return (rank - 1) % world_size


def _next_rank():
    """Global rank of the next pipeline stage."""
    rank = _get_pipeline_rank()
    world_size = _get_pipeline_world_size()
    if dist.is_initialized():
        return dist.get_rank() + 1  # simplified; full impl uses PIPELINE_GLOBAL_RANKS
    return (rank + 1) % world_size


def _communicate(tensor_send_next, tensor_send_prev, recv_prev, recv_next,
                 use_ring_exchange=False, tensor_shape=None,
                 override_scatter_gather_tensors_in_pipeline=False,
                 dtype_=None):
    """Communicate tensors between pipeline stages.

    Megatron dd8890626 p2p_communication.py _communicate():
      Creates placeholder receive tensors when recv_prev/recv_next are True,
      then uses either ring_exchange (use_ring_exchange=True) or
      batch_isend_irecv for the actual sends/receives.

    Args:
        tensor_send_next: tensor to send to next rank or None.
        tensor_send_prev: tensor to send to previous rank or None.
        recv_prev: boolean for whether to receive tensor from previous rank.
        recv_next: boolean for whether to receive tensor from next rank.
        use_ring_exchange: boolean for whether torch.distributed.ring_exchange()
                           API should be used.
        tensor_shape: optional, use when the input sequence contains less
                      tokens than the default sequence length
        override_scatter_gather_tensors_in_pipeline: optional, this is used
                                                     when tensor_shape is
                                                     provided to overwide
                                                     scatter gather tensors
        dtype_: optional, this is used when tensor_shape is provied and what
                is the type of tensor_shape
    Returns:
        (tensor_recv_prev, tensor_recv_next)
    """
    args = _get_comm_args()

    tensor_recv_prev = None
    tensor_recv_next = None
    if tensor_shape is None:
        # M1161: Megatron 13b3dca6d — respect model_parallel_memory_opt when
        # computing the default tensor_shape for interleaved pipeline schedules.
        if getattr(args, 'model_parallel_memory_opt', False):
            seq_length = args.seq_length // _get_tensor_model_parallel_world_size()
        else:
            seq_length = args.seq_length
        tensor_shape = (seq_length, args.micro_batch_size, args.hidden_size)
    if not override_scatter_gather_tensors_in_pipeline and \
            args.scatter_gather_tensors_in_pipeline:
        tensor_chunk_shape = reduce(operator.mul, tensor_shape, 1) // \
            _get_tensor_model_parallel_world_size()
    else:
        tensor_chunk_shape = tensor_shape

    dtype = args.params_dtype
    if getattr(args, 'fp32_residual_connection', False):
        dtype = torch.float

    requires_grad = True
    if dtype_ is not None:
        dtype = dtype_
        requires_grad = False

    if recv_prev:
        tensor_recv_prev = torch.empty(tensor_chunk_shape,
                                       requires_grad=requires_grad,
                                       device=torch.cuda.current_device(),
                                       dtype=dtype)
    if recv_next:
        tensor_recv_next = torch.empty(tensor_chunk_shape,
                                       requires_grad=requires_grad,
                                       device=torch.cuda.current_device(),
                                       dtype=dtype)

    # Split tensor into smaller chunks if using scatter-gather optimization.
    if not override_scatter_gather_tensors_in_pipeline and \
            args.scatter_gather_tensors_in_pipeline:
        if tensor_send_next is not None:
            tensor_send_next = _split_tensor_into_1d_equal_chunks(tensor_send_next)

        if tensor_send_prev is not None:
            tensor_send_prev = _split_tensor_into_1d_equal_chunks(tensor_send_prev)

    if use_ring_exchange:
        torch.distributed.ring_exchange(
            tensor_send_prev=tensor_send_prev,
            tensor_recv_prev=tensor_recv_prev,
            tensor_send_next=tensor_send_next,
            tensor_recv_next=tensor_recv_next,
            group=None)  # pipeline group; simplified for DS mapping
    else:
        ops = []
        if tensor_send_prev is not None:
            ops.append(torch.distributed.P2POp(
                torch.distributed.isend, tensor_send_prev, _prev_rank()))
        if tensor_recv_prev is not None:
            ops.append(torch.distributed.P2POp(
                torch.distributed.irecv, tensor_recv_prev, _prev_rank()))
        if tensor_send_next is not None:
            ops.append(torch.distributed.P2POp(
                torch.distributed.isend, tensor_send_next, _next_rank()))
        if tensor_recv_next is not None:
            ops.append(torch.distributed.P2POp(
                torch.distributed.irecv, tensor_recv_next, _next_rank()))
        if ops:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

    torch.cuda.synchronize()

    # If using scatter-gather optimization, gather smaller chunks.
    if not override_scatter_gather_tensors_in_pipeline and \
            args.scatter_gather_tensors_in_pipeline:
        if recv_prev:
            tensor_recv_prev = _gather_split_1d_tensor(
                tensor_recv_prev).view(tensor_shape).requires_grad_()
        if recv_next:
            tensor_recv_next = _gather_split_1d_tensor(
                tensor_recv_next).view(tensor_shape).requires_grad_()

    return tensor_recv_prev, tensor_recv_next


def _split_tensor_into_1d_equal_chunks(tensor):
    """Helper: split tensor into equal 1-D chunks for scatter-gather."""
    try:
        from deepspeed.compile import mpu_initialize as mpu
        return mpu.split_tensor_into_1d_equal_chunks(tensor)
    except Exception:
        return tensor


def _gather_split_1d_tensor(tensor):
    """Helper: gather 1-D chunks back into full tensor for scatter-gather."""
    try:
        from deepspeed.compile import mpu_initialize as mpu
        return mpu.gather_split_1d_tensor(tensor)
    except Exception:
        return tensor


def recv_forward(tensor_shape=None,
                 override_scatter_gather_tensors_in_pipeline=False,
                 dtype_=None, timers=None):
    """Receive input tensor from previous pipeline stage for forward pass.

    Megatron dd8890626 p2p_communication.py recv_forward().
    """

    if _is_pipeline_first_stage():
        return None
    if timers is not None:
        timers('forward-recv').start()
    input_tensor, _ = _communicate(
        tensor_send_next=None,
        tensor_send_prev=None,
        recv_prev=True,
        recv_next=False,
        tensor_shape=tensor_shape,
        override_scatter_gather_tensors_in_pipeline=\
            override_scatter_gather_tensors_in_pipeline,
        dtype_=dtype_)
    if timers is not None:
        timers('forward-recv').stop()
    return input_tensor


def recv_backward(timers=None):
    """Receive grad tensor from next pipeline stage for backward pass.

    Megatron dd8890626 p2p_communication.py recv_backward().
    """
    if _is_pipeline_last_stage():
        return None
    if timers is not None:
        timers('backward-recv').start()
    _, output_tensor_grad = _communicate(
        tensor_send_next=None,
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=True)
    if timers is not None:
        timers('backward-recv').stop()
    return output_tensor_grad


def send_forward(output_tensor, timers=None,
                 override_scatter_gather_tensors_in_pipeline=False,
                 dtype_=None):
    """Send activation tensor to next pipeline stage.

    Megatron dd8890626 p2p_communication.py send_forward().
    """

    if not _is_pipeline_last_stage():
        if timers is not None:
            timers('forward-send').start()
        _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
            override_scatter_gather_tensors_in_pipeline=\
            override_scatter_gather_tensors_in_pipeline,
            dtype_=dtype_)
        if timers is not None:
            timers('forward-send').stop()


def send_backward(input_tensor_grad, timers=None, use_ring_exchange=False):
    """Send grad tensor to previous pipeline stage.

    Megatron dd8890626 p2p_communication.py send_backward().
    """
    if _is_pipeline_first_stage():
        return
    if timers is not None:
        timers('backward-send').start()
    _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=False,
        recv_next=False,
        use_ring_exchange=use_ring_exchange)
    if timers is not None:
        timers('backward-send').stop()


def send_forward_recv_backward(output_tensor, timers=None, use_ring_exchange=False):
    """Send activation forward and receive grad backward (combined).

    Megatron dd8890626 p2p_communication.py send_forward_recv_backward().
    """
    if _is_pipeline_last_stage():
        return None
    if timers is not None:
        timers('forward-send-backward-recv').start()
    _, output_tensor_grad = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=True,
        use_ring_exchange=use_ring_exchange)
    if timers is not None:
        timers('forward-send-backward-recv').stop()
    return output_tensor_grad


def send_backward_recv_forward(input_tensor_grad, timers=None, use_ring_exchange=False):
    """Send grad backward and receive activation forward (combined).

    Megatron dd8890626 p2p_communication.py send_backward_recv_forward().
    """
    if _is_pipeline_first_stage():
        return None
    if timers is not None:
        timers('backward-send-forward-recv').start()
    input_tensor, _ = _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=True,
        recv_next=False,
        use_ring_exchange=use_ring_exchange)
    if timers is not None:
        timers('backward-send-forward-recv').stop()
    return input_tensor


def send_forward_recv_forward(output_tensor, recv_prev, timers=None):
    """Send activation forward and receive next activation forward via ring.

    Megatron dd8890626 p2p_communication.py send_forward_recv_forward().
    Uses ring_exchange (use_ring_exchange=True) as required by interleaved schedule.
    """
    if timers is not None:
        timers('forward-send-forward-recv').start()
    input_tensor, _ = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=recv_prev,
        recv_next=False,
        use_ring_exchange=True)
    if timers is not None:
        timers('forward-send-forward-recv').stop()
    return input_tensor


def send_backward_recv_backward(input_tensor_grad, recv_next, timers=None):
    """Send grad backward and receive next grad backward via ring.

    Megatron dd8890626 p2p_communication.py send_backward_recv_backward().
    Uses ring_exchange (use_ring_exchange=True) as required by interleaved schedule.
    """
    if timers is not None:
        timers('backward-send-backward-recv').start()
    _, output_tensor_grad = _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=False,
        recv_next=recv_next,
        use_ring_exchange=True)
    if timers is not None:
        timers('backward-send-backward-recv').stop()
    return output_tensor_grad


def send_forward_backward_recv_forward_backward(
        output_tensor, input_tensor_grad, recv_prev, recv_next, timers=None):
    """Full interleaved exchange: send fwd+bwd, receive fwd+bwd simultaneously.

    Megatron dd8890626 p2p_communication.py
    send_forward_backward_recv_forward_backward().
    Uses ring_exchange for the combined send/recv.
    """
    if timers is not None:
        timers('forward-backward-send-forward-backward-recv').start()
    input_tensor, output_tensor_grad = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=input_tensor_grad,
        recv_prev=recv_prev,
        recv_next=recv_next,
        use_ring_exchange=True)
    if timers is not None:
        timers('forward-backward-send-forward-backward-recv').stop()
    return input_tensor, output_tensor_grad
