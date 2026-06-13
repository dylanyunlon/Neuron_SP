# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M1082: Megatron dd96d402a — bug fixes
# Source: megatron/mpu/mappings.py (NVIDIA/Megatron-LM commit dd96d402a)
# Author: Vijay Korthikanti <vkorthikanti@nvidia.com>  Date: 2022-03-08
#
# Mapping: megatron/mpu/mappings.py → deepspeed/compile/mpu_mappings.py
#          (project convention: mpu/* → deepspeed/compile/)
#
# Changes ported from upstream (mpu/mappings.py):
#
#   1. _split_along_first_dim() [line ~70]:
#      Add .contiguous() to the slice output:
#        output = input_[dim_offset:dim_offset+local_dim_size].contiguous()
#      Rationale: tensor slicing returns a non-contiguous view; subsequent
#      distributed ops (all_gather_base, reduce_scatter_base) require
#      contiguous memory.  Missing .contiguous() causes a RuntimeError at
#      runtime when sequence parallelism is active.
#
#   2. _gather_along_first_dim() [line ~109-112]:
#      a. Remove requires_grad=False from torch.empty() for the output buffer.
#         requires_grad=False is the default and setting it explicitly can
#         interfere with autograd graph construction in some PyTorch versions.
#      b. Add .contiguous() to the input argument of _all_gather_base:
#           torch.distributed._all_gather_base(output, input_.contiguous(), ...)
#      Same rationale as change 1 — non-contiguous input to collective ops
#      can silently produce wrong results or raise a RuntimeError.
#
#   3. _reduce_scatter_along_first_dim() [line ~120-138]:
#      a. Remove leading blank line after the docstring.
#      b. Fix spacing: get_tensor_model_parallel_world_size()==1
#                    → get_tensor_model_parallel_world_size() == 1
#      c. Fix spacing: dim_size[0]= dim_size[0] // world_size
#                    → dim_size[0] = dim_size[0] // world_size
#      d. Remove requires_grad=False from torch.empty() (same as 2a).
#      e. Add .contiguous() to input of _reduce_scatter_base:
#           torch.distributed._reduce_scatter_base(output, input_.contiguous(), ...)
#      f. Remove blank line between the _reduce_scatter_base call and
#         return output.
#      g. Remove trailing blank lines after return output.
#
# Summary: all three functions now ensure contiguous memory before passing
# tensors to distributed collectives, and requires_grad=False is dropped
# from intermediate output buffers (default behaviour is False anyway, and
# explicit False can block autograd in edge cases).
# ---------------------------------------------------------------------------

print('[M1082]')

# ---------------------------------------------------------------------------
# Reference implementations of the three patched functions.
# These are directly portable; import helpers from mpu_initialize when
# integrating into a live module.
# ---------------------------------------------------------------------------

import torch


def _split_along_first_dim(input_, get_tensor_model_parallel_rank,
                            get_tensor_model_parallel_world_size):
    """Split the tensor along its first dimension and keep the
    corresponding slice.

    Megatron dd96d402a fix: add .contiguous() so the slice is a contiguous
    tensor before being used in distributed collectives.
    """
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along first dimension.
    dim_size = input_.size()[0]
    assert dim_size % world_size == 0, \
        "First dimension of the tensor should be divisible by tensor parallel size"
    local_dim_size = dim_size // world_size

    rank = get_tensor_model_parallel_rank()
    dim_offset = rank * local_dim_size

    # M1082 (dd96d402a): .contiguous() ensures distributed collectives
    # receive a contiguous buffer instead of a non-contiguous view.
    output = input_[dim_offset:dim_offset + local_dim_size].contiguous()

    return output


def _gather_along_first_dim(input_, get_tensor_model_parallel_world_size,
                             get_tensor_model_parallel_group):
    """Gather tensors and concatenate along the first dimension.

    Megatron dd96d402a fixes:
      - requires_grad=False removed from output buffer (default; explicit
        False can block autograd).
      - input_.contiguous() passed to _all_gather_base.
    """
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Size and dimension.
    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    # M1082: removed requires_grad=False
    output = torch.empty(dim_size, dtype=input_.dtype,
                         device=torch.cuda.current_device())
    # M1082: input_.contiguous() ensures non-contiguous views don't break
    # the collective.
    torch.distributed._all_gather_base(output, input_.contiguous(),
                                       group=get_tensor_model_parallel_group())

    return output


def _reduce_scatter_along_first_dim(input_, get_tensor_model_parallel_world_size,
                                     get_tensor_model_parallel_group):
    """Reduce-scatter the input tensor across model parallel group.

    Megatron dd96d402a fixes:
      - Spacing cleanup (== instead of ==, = instead of =).
      - requires_grad=False removed from output buffer.
      - input_.contiguous() passed to _reduce_scatter_base.
      - Trailing blank lines removed.
    """
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:  # M1082: space around == (was ==1)
        return input_

    dim_size = list(input_.size())
    assert dim_size[0] % world_size == 0
    dim_size[0] = dim_size[0] // world_size  # M1082: space around = (was =)

    # M1082: removed requires_grad=False
    output = torch.empty(dim_size, dtype=input_.dtype,
                         device=torch.cuda.current_device())
    # M1082: input_.contiguous() to avoid non-contiguous tensor errors
    torch.distributed._reduce_scatter_base(output, input_.contiguous(),
                                           group=get_tensor_model_parallel_group())
    return output


# ---------------------------------------------------------------------------
# M1157: Megatron 86e1df4e2 — parallel MOE support
# Source: megatron/mpu/mappings.py (NVIDIA/Megatron-LM commit 86e1df4e2)
# Author: Vijay Korthikanti <vkorthikanti@nvidia.com>  Date: 2022-03-30
#
# Mapping: megatron/mpu/mappings.py → deepspeed/compile/mpu_mappings.py
#          (project convention: mpu/* → deepspeed/compile/)
#
# Changes ported from upstream (mpu/mappings.py):
#
#   1. _gather_along_first_dim_moe() [new function]:
#      All-gather along first dimension using the global process group
#      (torch.distributed.get_world_size() / _all_gather_base), NOT the
#      tensor-model-parallel group.  Used to gather hidden states across
#      data-parallel ranks before MOE expert dispatch so each rank sees
#      all tokens.
#
#   2. _reduce_scatter_along_first_dim_moe() [new function]:
#      Reduce-scatter along first dimension using the global process group.
#      Inverse of _gather_along_first_dim_moe; used after MOE expert
#      computation to scatter results back to data-parallel ranks.
#
#   3. _GatherFromSequenceParallelRegionToMOE [new autograd.Function]:
#      forward  → _gather_along_first_dim_moe
#      backward → _reduce_scatter_along_first_dim_moe
#      Includes symbolic() for torch.jit / ONNX tracing.
#
#   4. _ReduceScatterToSequenceParallelRegionFromMOE [new autograd.Function]:
#      forward  → _reduce_scatter_along_first_dim_moe
#      backward → _gather_along_first_dim_moe
#      Inverse of class 3.
#
#   5. gather_from_sequence_parallel_region_to_moe() [new public function]:
#      Thin wrapper: _GatherFromSequenceParallelRegionToMOE.apply(input_)
#
#   6. reduce_scatter_to_sequence_parallel_region_from_moe() [new public function]:
#      Thin wrapper: _ReduceScatterToSequenceParallelRegionFromMOE.apply(input_)
#
# Key design note: these MOE collectives use the *global* distributed group
# (not the tensor-model-parallel group) because MOE expert parallelism is
# orthogonal to tensor parallelism — experts are partitioned across
# data-parallel ranks, not tensor-parallel ranks.
#
# Exported from megatron/mpu/__init__.py (also M1157):
#   from .mappings import gather_from_sequence_parallel_region_to_moe
#   from .mappings import reduce_scatter_to_sequence_parallel_region_from_moe
# ---------------------------------------------------------------------------

print('[M1157]')


def _gather_along_first_dim_moe(input_):
    """Gather tensors and concatinate along the first dimension.

    M1157 (86e1df4e2): uses global dist group, not tensor-model-parallel group,
    so that hidden states are gathered across data-parallel ranks for MOE dispatch.
    """
    world_size = torch.distributed.get_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype,
                         device=torch.cuda.current_device())
    torch.distributed._all_gather_base(output, input_.contiguous())

    return output


def _reduce_scatter_along_first_dim_moe(input_):
    """Reduce-scatter the input tensor across model parallel group.

    M1157 (86e1df4e2): inverse of _gather_along_first_dim_moe; uses global
    dist group to scatter MOE expert outputs back to data-parallel ranks.
    """
    world_size = torch.distributed.get_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    assert dim_size[0] % world_size == 0
    dim_size[0] = dim_size[0] // world_size

    output = torch.empty(dim_size, dtype=input_.dtype,
                         device=torch.cuda.current_device())
    torch.distributed._reduce_scatter_base(output, input_.contiguous())
    return output


class _GatherFromSequenceParallelRegionToMOE(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""  # TODO

    @staticmethod
    def symbolic(graph, input_):
        return _gather_along_first_dim_moe(input_)

    @staticmethod
    def forward(ctx, input_):
        return _gather_along_first_dim_moe(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce_scatter_along_first_dim_moe(grad_output)


class _ReduceScatterToSequenceParallelRegionFromMOE(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce_scatter_along_first_dim_moe(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce_scatter_along_first_dim_moe(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim_moe(grad_output)


def gather_from_sequence_parallel_region_to_moe(input_):
    return _GatherFromSequenceParallelRegionToMOE.apply(input_)


def reduce_scatter_to_sequence_parallel_region_from_moe(input_):
    return _ReduceScatterToSequenceParallelRegionFromMOE.apply(input_)

# ---------------------------------------------------------------------------
# M1730: Megatron b3fac674f — Fix expert parallelism issues from merge
# Source: megatron/core/tensor_parallel/__init__.py
#         (NVIDIA/Megatron-LM commit b3fac674f)
#
# Mapping: megatron/core/tensor_parallel/__init__.py → deepspeed/compile/mpu_mappings.py
#
# Changes ported from upstream:
#   gather_from_sequence_parallel_region_to_moe and
#   reduce_scatter_to_sequence_parallel_region_from_moe are now explicitly
#   re-exported from megatron.core.tensor_parallel.__all__, so callers can
#   import directly instead of going through mpu.
#
#   This file (mpu_mappings.py) already implements both functions since M1157.
#   M1730 merely confirms they are canonical public API — no code change here.
# ---------------------------------------------------------------------------

print('[M1730] mpu_mappings: gather_from_sequence_parallel_region_to_moe and '
      'reduce_scatter_to_sequence_parallel_region_from_moe confirmed as public API')
