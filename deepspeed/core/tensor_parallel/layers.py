# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Tensor-parallel layer implementations.

Contains:
  * VocabParallelEmbedding
  * ColumnParallelLinear          (with heterogeneous head-count padding)
  * RowParallelLinear             (with heterogeneous head-count padding)
  * linear_with_grad_accumulation_and_async_allreduce
  * linear_with_frozen_weight
  * Supporting autograd functions and collective helpers
  * sharded_state_dict support for distributed checkpointing

Design notes
------------
**Heterogeneous head-count padding** (32 heads / 5 GPUs → non-divisible TP):
  Standard TP requires output_size % tp_world_size == 0.  Real MHA configs
  (e.g. 32 heads over 5 GPUs) violate this.  We solve it with
  *dummy-head padding*:

    padded_output_size = ceil(output_size / tp_world_size) * tp_world_size

  Every rank allocates ``padded_output_size // tp_world_size`` output
  columns.  The last rank's weight rows beyond ``output_size`` are
  *dummy rows*: initialised to zero and masked out before/after every
  communication collective.  This keeps all-gather / reduce-scatter
  shapes uniform while producing numerically identical results to an
  un-padded single-rank baseline.

  The same strategy applies to VocabParallelEmbedding (vocab size padding)
  and RowParallelLinear (input dimension padding).

**Expert-parallel support** (is_expert=True):
  When is_expert=True the layer belongs to an MoE expert shard.  The
  expert-parallel RNG state is used for weight initialisation and the
  ``allreduce`` attribute on weight/bias is set to False so that the
  distributed optimizer knows not to all-reduce these parameters.

**Strided linear layers**:
  stride > 1 is used to interleave Q, K, V weight rows so that a single
  fused weight matrix can be split into Q/K/V shards without a copy.

**sharded_state_dict**:
  Each layer exposes sharded_state_dict() that tags weight/bias tensors
  with their TP axis for use with the distributed checkpointing framework.

These follow the Megatron-LM design (megatron/core/tensor_parallel/layers.py).
When TP=1 all layers behave identically to their standard PyTorch counterparts,
so the code is correct on PCIe-only clusters with no NVLink.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from deepspeed.core.model_parallel_config import ModelParallelConfig
import copy


# ---------------------------------------------------------------------------
# Helpers to retrieve TP group info without hard-crashing when the
# process group has not been initialised (e.g. single-process unit tests).
# ---------------------------------------------------------------------------

def _get_tp_group() -> Optional[torch.distributed.ProcessGroup]:
    """Return the TP process group, or None when not initialised."""
    try:
        from deepspeed.core.parallel_state import get_tensor_model_parallel_group
        return get_tensor_model_parallel_group()
    except (ImportError, AssertionError):
        return None


def _get_tp_world_size() -> int:
    """Return TP world size (1 when not initialised)."""
    group = _get_tp_group()
    if group is None:
        return 1
    return torch.distributed.get_world_size(group=group)


def _get_tp_rank() -> int:
    """Return TP rank (0 when not initialised)."""
    group = _get_tp_group()
    if group is None:
        return 0
    return torch.distributed.get_rank(group=group)


def _get_expert_tp_group() -> Optional[torch.distributed.ProcessGroup]:
    """Return the expert TP process group, or fall back to TP group."""
    try:
        from deepspeed.core.parallel_state import get_expert_tensor_parallel_group
        return get_expert_tensor_parallel_group()
    except (ImportError, AssertionError):
        return _get_tp_group()


def _resolve_tp_group(
    tp_group: Optional[torch.distributed.ProcessGroup],
    is_expert: bool = False,
) -> Optional[torch.distributed.ProcessGroup]:
    """Resolve TP process group: explicit > expert > global TP."""
    if tp_group is not None:
        return tp_group
    if is_expert:
        return _get_expert_tp_group()
    return _get_tp_group()


def _pg_world_size(group: Optional[torch.distributed.ProcessGroup]) -> int:
    if group is None:
        return 1
    return torch.distributed.get_world_size(group=group)


def _pg_rank(group: Optional[torch.distributed.ProcessGroup]) -> int:
    if group is None:
        return 0
    return torch.distributed.get_rank(group=group)


# ---------------------------------------------------------------------------
# TP attribute helpers (mirrors Megatron's layers.py public API)
# ---------------------------------------------------------------------------

_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS: Dict[str, Any] = {
    "expert_tp": False,
    "is_qkv": False,
    "qkv_split_shapes": None,
    "tensor_model_parallel": False,
    "partition_dim": -1,
    "partition_stride": 1,
}


def set_tensor_model_parallel_attributes(
    tensor: torch.Tensor,
    is_parallel: bool,
    dim: int,
    stride: int,
) -> None:
    """Attach TP sharding metadata to a tensor/parameter.

    Args:
        tensor:      The parameter to annotate.
        is_parallel: Whether this tensor is sharded across TP ranks.
        dim:         The dimension along which it is sharded (0 or 1).
        stride:      Partition stride (usually 1, >1 for QKV fused layers).
    """
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute), (
            f"Attribute {attribute} already set on tensor; call "
            f"set_defaults_if_not_set_tensor_model_parallel_attributes first if needed."
        )
    tensor.tensor_model_parallel = is_parallel
    tensor.partition_dim = dim
    tensor.partition_stride = stride


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor: torch.Tensor) -> None:
    """Set TP metadata defaults on *tensor* if not already set."""
    def _set(key, val):
        if not hasattr(tensor, key):
            setattr(tensor, key, val)
    for attr, default in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS.items():
        _set(attr, default)


def copy_tensor_model_parallel_attributes(
    destination_tensor: torch.Tensor,
    source_tensor: torch.Tensor,
) -> None:
    """Copy TP sharding attributes from *source_tensor* to *destination_tensor*."""
    def _copy(attr):
        if hasattr(source_tensor, attr):
            setattr(destination_tensor, attr, getattr(source_tensor, attr))
    for attr in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        _copy(attr)


def param_is_not_tensor_parallel_duplicate(
    param: torch.Tensor,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> bool:
    """Return True if *param* is NOT a TP duplicate (i.e. it should be in grads).

    Replicated parameters (e.g. biases in RowParallelLinear) only need to have
    their gradients reduced once; this predicate identifies them.
    """
    if hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel:
        return True
    if tp_group is not None:
        return torch.distributed.get_rank(group=tp_group) == 0
    return _get_tp_rank() == 0


# ---------------------------------------------------------------------------
# Heterogeneous padding utilities
# ---------------------------------------------------------------------------

def _padded_partition_size(total_size: int, tp_world_size: int) -> int:
    """Return the per-rank partition size after ceiling-padding *total_size*.

    For total_size=32, tp_world_size=5  →  ceil(32/5)*5=35  →  per_rank=7.
    For total_size=32, tp_world_size=4  →  ceil(32/4)*4=32  →  per_rank=8.
    For total_size=32, tp_world_size=1  →  per_rank=32.

    This is intentionally lenient (no divisibility assert) so heterogeneous
    configs (e.g. 32 heads / 5 GPUs) work out of the box.
    """
    if tp_world_size == 1:
        return total_size
    padded = math.ceil(total_size / tp_world_size) * tp_world_size
    return padded // tp_world_size


def _rank_output_slice(
    total_size: int,
    tp_rank: int,
    tp_world_size: int,
) -> Tuple[int, int]:
    """Return [start, end) column range for *tp_rank* (unpadded real columns).

    Columns in [start, end) are real; columns in [end, per_rank_padded) on the
    last rank are dummies (zero-weighted).

    Example: total=32, tp=5 → per_rank_padded=7
      rank 0: [0,  7)   real=7, dummy=0
      rank 1: [7,  14)  real=7, dummy=0
      rank 2: [14, 21)  real=7, dummy=0
      rank 3: [21, 28)  real=7, dummy=0
      rank 4: [28, 35)  real=min(35,32)-28=4, dummy=3
    """
    per_rank = _padded_partition_size(total_size, tp_world_size)
    start = tp_rank * per_rank
    end = min(start + per_rank, total_size)
    return start, end


def _num_dummy_cols(total_size: int, tp_rank: int, tp_world_size: int) -> int:
    """Return the number of dummy (padding) columns on this rank."""
    per_rank = _padded_partition_size(total_size, tp_world_size)
    _, real_end = _rank_output_slice(total_size, tp_rank, tp_world_size)
    start = tp_rank * per_rank
    return per_rank - (real_end - start)


def _divide_strict(numerator: int, denominator: int) -> int:
    """Integer division with assertion that it divides evenly."""
    assert numerator % denominator == 0, (
        f"{numerator} is not divisible by {denominator}"
    )
    return numerator // denominator


def _vocab_range(
    num_embeddings: int,
    tp_rank: int,
    tp_world_size: int,
) -> Tuple[int, int]:
    """Return [start, end) vocab indices for this TP rank (with ceiling padding).

    We ceiling-pad num_embeddings to the next multiple of tp_world_size so
    every rank holds the same number of rows.  Rows beyond num_embeddings
    are padding and never looked up.
    """
    per_partition = _padded_partition_size(num_embeddings, tp_world_size)
    start = tp_rank * per_partition
    end = min(start + per_partition, num_embeddings)
    return start, end


# ---------------------------------------------------------------------------
# Weight initialisation helpers
# ---------------------------------------------------------------------------

def _initialize_affine_weight_cpu(
    weight: Parameter,
    output_size: int,
    input_size: int,
    per_partition_size: int,
    partition_dim: int,
    init_method: Callable,
    stride: int = 1,
    return_master_weight: bool = False,
    *,
    params_dtype: torch.dtype = torch.float32,
    tp_rank: Optional[int] = None,
    tp_world_size: Optional[int] = None,
    skip_set_tensor_parallel_attributes: bool = False,
) -> Optional[torch.Tensor]:
    """Initialize affine weight for model parallel on CPU.

    Builds the full master weight, splits it, and copies this rank's shard
    into *weight*.  Supports strided linear layers (stride > 1) for QKV fusion.

    Args:
        weight:                  Parameter to fill in-place.
        output_size:             Full output feature dimension.
        input_size:              Full input feature dimension.
        per_partition_size:      Per-rank partition size along partition_dim.
        partition_dim:           0 for column parallel, 1 for row parallel.
        init_method:             Weight initialiser callable.
        stride:                  Stride for strided (QKV) linear layers.
        return_master_weight:    If True, return the master weight tensor.
        params_dtype:            Parameter dtype.
        tp_rank:                 Override TP rank.
        tp_world_size:           Override TP world size.
        skip_set_tensor_parallel_attributes: Skip setting TP attrs (for callers
                                             that already set them).
    Returns:
        Master weight tensor if return_master_weight else None.
    """
    if not skip_set_tensor_parallel_attributes:
        set_tensor_model_parallel_attributes(
            tensor=weight, is_parallel=True, dim=partition_dim, stride=stride
        )

    if tp_rank is None:
        tp_rank = _get_tp_rank()
    if tp_world_size is None:
        tp_world_size = _get_tp_world_size()

    # Build full master weight in float32 for numerical stability
    master_weight = torch.empty(output_size, input_size, dtype=torch.float32, requires_grad=False)
    init_method(master_weight)
    master_weight = master_weight.to(dtype=params_dtype)

    # Handle padding for heterogeneous configs
    full_size = output_size if partition_dim == 0 else input_size
    padded = _padded_partition_size(full_size, tp_world_size) * tp_world_size
    if padded > full_size:
        pad_size = padded - full_size
        if partition_dim == 0:
            padding = torch.zeros(pad_size, input_size, dtype=params_dtype)
            master_weight = torch.cat([master_weight, padding], dim=0)
        else:
            padding = torch.zeros(output_size, pad_size, dtype=params_dtype)
            master_weight = torch.cat([master_weight, padding], dim=1)

    # Strided split: interleave chunks for QKV fusion
    per_partition_per_stride_size = per_partition_size // stride
    weight_list = torch.split(master_weight, per_partition_per_stride_size, dim=partition_dim)
    my_weight_list = weight_list[tp_rank::tp_world_size]

    with torch.no_grad():
        cpu_weight = torch.cat(my_weight_list, dim=partition_dim)
        weight.data.copy_(cpu_weight.contiguous())

    if return_master_weight:
        return master_weight
    return None


def _initialize_affine_weight_gpu(
    weight: Parameter,
    init_method: Callable,
    partition_dim: int,
    stride: int = 1,
    is_expert: bool = False,
) -> None:
    """Initialize affine weight for model parallel on GPU using RNG tracker.

    Args:
        weight:        Parameter to initialise in-place.
        init_method:   Weight initialiser callable.
        partition_dim: 0 for column parallel, 1 for row parallel.
        stride:        Stride for strided (QKV) linear layers.
        is_expert:     If True, use expert-parallel RNG tracker.
    """
    set_tensor_model_parallel_attributes(
        tensor=weight, is_parallel=True, dim=partition_dim, stride=stride
    )
    try:
        from deepspeed.core.tensor_parallel.random import (
            get_cuda_rng_tracker,
            get_expert_parallel_rng_tracker_name,
        )
        tracker = get_cuda_rng_tracker()
        if is_expert:
            with tracker.fork(get_expert_parallel_rng_tracker_name()):
                init_method(weight)
        else:
            with tracker.fork():
                init_method(weight)
    except (ImportError, AttributeError, Exception):
        # No RNG tracker – initialise directly (acceptable for TP=1 / tests)
        init_method(weight)


def _init_weight_cpu(
    weight: Parameter,
    full_shape: Tuple[int, int],
    partition_dim: int,
    init_method: Callable,
    params_dtype: torch.dtype,
    tp_rank: int,
    tp_world_size: int,
    stride: int = 1,
) -> None:
    """Thin wrapper around _initialize_affine_weight_cpu for backward compat."""
    output_size, input_size = full_shape
    per_partition_size = _padded_partition_size(
        full_shape[partition_dim], tp_world_size
    ) * stride // stride  # = _padded_partition_size
    _initialize_affine_weight_cpu(
        weight=weight,
        output_size=output_size,
        input_size=input_size,
        per_partition_size=per_partition_size,
        partition_dim=partition_dim,
        init_method=init_method,
        stride=stride,
        params_dtype=params_dtype,
        tp_rank=tp_rank,
        tp_world_size=tp_world_size,
    )


def _init_weight_gpu(weight: Parameter, init_method: Callable, is_expert: bool = False) -> None:
    """Initialise weight in-place on GPU using RNG tracker when available."""
    try:
        from deepspeed.core.tensor_parallel.random import (
            get_cuda_rng_tracker,
            get_expert_parallel_rng_tracker_name,
        )
        tracker = get_cuda_rng_tracker()
        if is_expert:
            with tracker.fork(get_expert_parallel_rng_tracker_name()):
                init_method(weight)
        else:
            with tracker.fork():
                init_method(weight)
    except (ImportError, AttributeError, Exception):
        init_method(weight)


# ---------------------------------------------------------------------------
# linear_with_grad_accumulation_and_async_allreduce
# ---------------------------------------------------------------------------

def linear_with_frozen_weight(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool = False,
    allreduce_dgrad: bool = False,
    sequence_parallel: bool = False,
    grad_output_buffer: Optional[List[torch.Tensor]] = None,
    wgrad_deferral_limit: Optional[int] = 0,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Linear layer with frozen (no-grad) weight.

    Uses standard F.linear since no gradient accumulation is needed.
    Handles sequence-parallel all-gather when sequence_parallel=True.
    """
    if sequence_parallel and tp_group is not None:
        world_size = torch.distributed.get_world_size(group=tp_group)
        if world_size > 1:
            gather_shape = list(input.shape)
            gather_shape[0] = gather_shape[0] * world_size
            gather_buffer = torch.empty(
                gather_shape, dtype=input.dtype, device=input.device
            )
            torch.distributed.all_gather_into_tensor(
                gather_buffer, input.contiguous(), group=tp_group
            )
            input = gather_buffer
    return F.linear(input, weight, bias)


class _LinearWithGradAccumulationAndAsyncAllReduce(torch.autograd.Function):
    """Custom autograd function for column-parallel linear.

    Forward:  Y = X W^T  (+ optional all-gather for sequence_parallel)
    Backward: computes grad_input and grad_weight, optionally fusing weight
              gradient accumulation into main_grad for optimizer efficiency.

    Key behaviours:
    * sequence_parallel=True: all-gather inputs before GEMM (forward),
      reduce-scatter gradients after GEMM (backward).
    * allreduce_dgrad=True: all-reduce input gradients across TP group.
    * gradient_accumulation_fusion=True: accumulate weight gradients directly
      into weight.main_grad when available (requires APEX fused kernel or
      falls back gracefully).
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        gradient_accumulation_fusion: bool,
        allreduce_dgrad: bool,
        sequence_parallel: bool,
        grad_output_buffer: Optional[List[torch.Tensor]],
        wgrad_deferral_limit: Optional[int],
        tp_group: Optional[torch.distributed.ProcessGroup],
    ) -> torch.Tensor:
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.sequence_parallel = sequence_parallel
        ctx.grad_output_buffer = grad_output_buffer
        ctx.wgrad_deferral_limit = wgrad_deferral_limit
        ctx.tp_group = tp_group

        if sequence_parallel and tp_group is not None:
            world_size = torch.distributed.get_world_size(group=tp_group)
            if world_size > 1:
                gather_shape = list(input.shape)
                gather_shape[0] = gather_shape[0] * world_size
                gather_buffer = torch.empty(
                    gather_shape, dtype=input.dtype, device=input.device
                )
                torch.distributed.all_gather_into_tensor(
                    gather_buffer, input.contiguous(), group=tp_group
                )
                total_input = gather_buffer
            else:
                total_input = input
        else:
            total_input = input

        output = torch.matmul(total_input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        tp_group = ctx.tp_group
        sequence_parallel = ctx.sequence_parallel
        grad_output_buffer = ctx.grad_output_buffer
        wgrad_deferral_limit = ctx.wgrad_deferral_limit

        # Reconstruct total_input for wgrad computation
        if sequence_parallel and tp_group is not None:
            world_size = torch.distributed.get_world_size(group=tp_group)
            if world_size > 1:
                gather_shape = list(input.shape)
                gather_shape[0] = gather_shape[0] * world_size
                gather_buffer = torch.empty(
                    gather_shape, dtype=input.dtype, device=input.device
                )
                handle = torch.distributed.all_gather_into_tensor(
                    gather_buffer, input.contiguous(), group=tp_group, async_op=True
                )
                total_input = gather_buffer
            else:
                total_input = input
                handle = None
        else:
            total_input = input
            handle = None

        # Compute grad_input
        grad_input = grad_output.matmul(weight)

        # All-reduce or reduce-scatter grad_input across TP ranks
        if ctx.allreduce_dgrad and tp_group is not None:
            world_size = torch.distributed.get_world_size(group=tp_group)
            if world_size > 1:
                torch.distributed.all_reduce(grad_input, group=tp_group)
        elif sequence_parallel and tp_group is not None:
            world_size = torch.distributed.get_world_size(group=tp_group)
            if world_size > 1:
                grad_input = grad_input.contiguous()
                out_shape = list(grad_input.shape)
                assert out_shape[0] % world_size == 0, (
                    f"Sequence dim {out_shape[0]} must be divisible by TP world size {world_size}"
                )
                out_shape[0] //= world_size
                local_grad = torch.empty(
                    out_shape, dtype=grad_input.dtype, device=grad_input.device
                )
                torch.distributed.reduce_scatter_tensor(local_grad, grad_input, group=tp_group)
                grad_input = local_grad

        # Wait for all-gather to complete before wgrad
        if handle is not None:
            handle.wait()

        # Compute weight gradient
        wgrad_compute = True
        if grad_output_buffer is not None and wgrad_deferral_limit is not None:
            if wgrad_deferral_limit == 0 or len(grad_output_buffer) < wgrad_deferral_limit:
                grad_output_buffer.append(grad_output)
                wgrad_compute = False

        if wgrad_compute:
            if ctx.gradient_accumulation_fusion and hasattr(weight, "main_grad"):
                # Fused gradient accumulation: accumulate directly into main_grad
                # This avoids a separate copy and reduces memory traffic.
                if weight.main_grad.dtype == torch.float32:
                    try:
                        import fused_weight_gradient_mlp_cuda
                        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                            total_input, grad_output, weight.main_grad
                        )
                        grad_weight = None
                    except (ImportError, RuntimeError):
                        # Fallback: manual accumulation
                        grad_weight = grad_output.t().matmul(total_input)
                        weight.main_grad.add_(grad_weight.to(weight.main_grad.dtype))
                        grad_weight = None
                elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                    try:
                        import fused_weight_gradient_mlp_cuda
                        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                            total_input, grad_output, weight.main_grad
                        )
                        grad_weight = None
                    except (ImportError, RuntimeError):
                        grad_weight = grad_output.t().matmul(total_input)
                        weight.main_grad.add_(grad_weight.to(weight.main_grad.dtype))
                        grad_weight = None
                else:
                    grad_weight = grad_output.t().matmul(total_input)
            else:
                grad_weight = grad_output.t().matmul(total_input)
        else:
            grad_weight = None

        grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1))) if use_bias else None

        return (
            grad_input,
            grad_weight,
            grad_bias,
            None,  # gradient_accumulation_fusion
            None,  # allreduce_dgrad
            None,  # sequence_parallel
            None,  # grad_output_buffer
            None,  # wgrad_deferral_limit
            None,  # tp_group
        )


def linear_with_grad_accumulation_and_async_allreduce(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool = False,
    allreduce_dgrad: bool = False,
    sequence_parallel: bool = False,
    grad_output_buffer: Optional[List[torch.Tensor]] = None,
    wgrad_deferral_limit: Optional[int] = 0,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    # Legacy compat: async_grad_allreduce maps to allreduce_dgrad
    async_grad_allreduce: Optional[bool] = None,
) -> torch.Tensor:
    """Linear layer with optional async all-reduce and gradient accumulation fusion.

    This is the central GEMM primitive for tensor-parallel layers.  It handles:

    * **sequence_parallel=True**: all-gather inputs across TP ranks before GEMM
      (column-parallel forward path).  In backward, reduce-scatter gradients.
    * **allreduce_dgrad=True**: all-reduce input gradients across TP group in
      backward (standard column-parallel path when SP is disabled).
    * **gradient_accumulation_fusion=True**: accumulate weight gradients
      directly into ``weight.main_grad`` using APEX fused kernel when available.
    * **wgrad_deferral_limit**: defer weight gradient computation for the
      embedding layer to overlap with pipeline communication.

    Args:
        input:                        Input tensor.
        weight:                       Weight tensor (shape: [out, in]).
        bias:                         Optional bias.
        gradient_accumulation_fusion: Fuse wgrad accumulation into main_grad.
        allreduce_dgrad:              All-reduce input gradients in backward.
        sequence_parallel:            All-gather inputs (sequence-parallel path).
        grad_output_buffer:           Buffer for deferred wgrad (embedding opt).
        wgrad_deferral_limit:         Max deferred wgrad entries.
        tp_group:                     Tensor-parallel process group.
        async_grad_allreduce:         Legacy alias for allreduce_dgrad.

    Returns:
        Output tensor of shape ``[*, output_size]``.
    """
    if async_grad_allreduce is not None:
        allreduce_dgrad = async_grad_allreduce

    return _LinearWithGradAccumulationAndAsyncAllReduce.apply(
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
        tp_group,
    )


# ---------------------------------------------------------------------------
# VocabParallelEmbedding
# ---------------------------------------------------------------------------

class VocabParallelEmbedding(nn.Module):
    """Embedding parallelized across TP ranks (with ceiling-padding for non-divisible vocab).

    The full vocabulary of size *num_embeddings* is ceiling-padded to the next
    multiple of *tp_world_size* and then split evenly.  Dummy rows (beyond
    num_embeddings) are zero-initialised and never looked up.

    When TP=1 this is equivalent to ``nn.Embedding``.

    Args:
        num_embeddings: Vocabulary size (may be non-divisible by tp_world_size).
        embedding_dim:  Hidden dimension.
        config:         ModelParallelConfig.
        init_method:    Weight initialiser (defaults to nn.init.normal_).
        tp_group:       Explicit TP process group (overrides parallel_state).
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        config: ModelParallelConfig,
        init_method: Optional[Callable] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.config = config
        self.tp_group = _resolve_tp_group(tp_group)

        if init_method is None:
            init_method = nn.init.normal_

        tp_world_size = _pg_world_size(self.tp_group)
        tp_rank = _pg_rank(self.tp_group)

        # Ceiling-padded partition: handles non-divisible vocab sizes
        self.vocab_start_index, self.vocab_end_index = _vocab_range(
            num_embeddings, tp_rank, tp_world_size
        )
        self.num_embeddings_per_partition = _padded_partition_size(num_embeddings, tp_world_size)
        self._real_rows = self.vocab_end_index - self.vocab_start_index

        # Allocate the weight shard
        if config.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _init_weight_cpu(
                    self.weight,
                    full_shape=(num_embeddings, embedding_dim),
                    partition_dim=0,
                    init_method=init_method,
                    params_dtype=config.params_dtype,
                    tp_rank=tp_rank,
                    tp_world_size=tp_world_size,
                )
                if self._real_rows < self.num_embeddings_per_partition:
                    with torch.no_grad():
                        self.weight.data[self._real_rows:].zero_()
            else:
                set_tensor_model_parallel_attributes(self.weight, True, 0, 1)
        else:
            device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    device=device,
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(self.weight, init_method, partition_dim=0)
                if self._real_rows < self.num_embeddings_per_partition:
                    with torch.no_grad():
                        self.weight.data[self._real_rows:].zero_()
            else:
                set_tensor_model_parallel_attributes(self.weight, True, 0, 1)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Embedding lookup with all-reduce for TP>1.

        Args:
            input_: Integer token index tensor of any shape.

        Returns:
            Float tensor of shape ``(*input_.shape, embedding_dim)``.
        """
        tp_world_size = _pg_world_size(self.tp_group)

        if tp_world_size > 1:
            input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
            local_input = input_.clone() - self.vocab_start_index
            local_input.clamp_(min=0, max=self._real_rows - 1)
            local_input[input_mask] = 0
        else:
            local_input = input_

        output_parallel = F.embedding(local_input, self.weight)

        if tp_world_size > 1:
            output_parallel[input_mask] = 0.0
            torch.distributed.all_reduce(output_parallel, group=self.tp_group)

        return output_parallel

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> dict:
        """Return sharded state dict for distributed checkpointing (axis 0)."""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        try:
            from deepspeed.core.utils import make_tp_sharded_tensor_for_checkpoint
            return {
                f"{prefix}weight": make_tp_sharded_tensor_for_checkpoint(
                    state_dict["weight"], f"{prefix}weight", tp_axis=0,
                    tp_group=self.tp_group,
                    **({"dp_cp_group": metadata["dp_cp_group"]} if metadata else {}),
                )
            }
        except (ImportError, Exception):
            return {f"{prefix}{k}": v for k, v in state_dict.items()}


# ---------------------------------------------------------------------------
# ColumnParallelLinear
# ---------------------------------------------------------------------------

class ColumnParallelLinear(nn.Module):
    """Linear layer with column parallelism + heterogeneous head-count padding.

    Splits weight matrix along the output dimension across TP ranks.
    Supports non-divisible output sizes (e.g. 32 heads / 5 GPUs) via
    ceiling-padding with dummy rows that are zero-initialised and masked
    away before returning results.

    Y = X A^T + b   where A is [output_size, input_size].
    A is partitioned column-wise (output dimension) so each rank holds
    A_i of shape [output_size_per_partition, input_size].

    Dummy-head padding mechanics
    ----------------------------
    1. Weight rows [real_output_size_this_rank, output_size_per_partition]
       are zero-initialised and their gradients are zeroed before each step.
    2. When gather_output=True, the all-gathered tensor is sliced [:output_size]
       to strip padding columns before returning.
    3. Bias (when used) is also truncated after gather.

    Expert-parallel support
    -----------------------
    When is_expert=True:
    * Expert-parallel RNG state is used for weight init.
    * ``weight.allreduce = False`` so the optimizer skips all-reduce for
      this parameter (experts are already reduced in expert-TP).
    * Explicit-expert-comm path skips the copy-to-TP-region and all-gather.

    Args:
        input_size:                 Input feature dimension.
        output_size:                Full output feature dimension (before TP split).
        config:                     ModelParallelConfig.
        bias:                       Whether to add a bias term.
        gather_output:              If True, all-gather the output so every rank sees
                                    the full [*, output_size] tensor.
        init_method:                Weight initialiser callable.
        stride:                     Partition stride (for strided QKV linear layers).
        keep_master_weight_for_test: Return master weight (only for testing).
        skip_bias_add:              If True, do not add bias inside forward(); instead
                                    return it as the second element of the output tuple.
        skip_weight_param_allocation: If True, weight is not allocated; caller must pass
                                    it as a keyword argument in forward().
        embedding_activation_buffer: Buffer for deferred embedding wgrad activations.
        grad_output_buffer:         Buffer for deferred embedding wgrad gradients.
        is_expert:                  Whether this layer is an MoE expert.
        disable_grad_reduce:        Skip all-reduce of input gradients (for LoRA etc.)
        tp_group:                   Explicit TP process group.
        num_heads:                  (Optional) total attention heads for documentation.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: ModelParallelConfig,
        bias: bool = True,
        gather_output: bool = True,
        init_method: Optional[Callable] = None,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        skip_bias_add: bool = False,
        skip_weight_param_allocation: bool = False,
        embedding_activation_buffer: Optional[List[torch.Tensor]] = None,
        grad_output_buffer: Optional[List[torch.Tensor]] = None,
        is_expert: bool = False,
        disable_grad_reduce: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        num_heads: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add
        self.is_expert = is_expert
        self.expert_parallel = config.expert_model_parallel_size > 1
        self.embedding_activation_buffer = embedding_activation_buffer
        self.grad_output_buffer = grad_output_buffer
        self.config = config
        self.disable_grad_reduce = disable_grad_reduce
        self.num_heads = num_heads

        if init_method is None:
            init_method = nn.init.xavier_normal_

        # Resolve TP group (expert-aware)
        self.tp_group = _resolve_tp_group(tp_group, is_expert=is_expert)
        world_size = _pg_world_size(self.tp_group)
        rank = _pg_rank(self.tp_group)

        # Expert-comm path: expert is in its own TP group but expert_parallel is True
        self.explicit_expert_comm = is_expert and (world_size > 1 or self.expert_parallel)

        # Ceiling-padded partition size (same on every rank for uniform collectives)
        self.output_size_per_partition = _padded_partition_size(output_size, world_size)

        # Real (non-dummy) output columns on this rank
        real_start, real_end = _rank_output_slice(output_size, rank, world_size)
        self._real_output_cols = real_end - real_start
        self._has_dummy = self._real_output_cols < self.output_size_per_partition

        # Sequence-parallel: warn and disable when world_size == 1
        self.sequence_parallel = getattr(config, "sequence_parallel", False)
        if self.sequence_parallel and world_size <= 1:
            warnings.warn(
                "`sequence_parallel` is set to `True`, but tensor model parallel size "
                f"is {world_size}. Disabling sequence parallel.",
                stacklevel=2,
            )
            self.sequence_parallel = False

        # allreduce_dgrad: all-reduce input grads in standard (non-SP) path
        self.allreduce_dgrad = (
            world_size > 1
            and not self.sequence_parallel
            and not self.disable_grad_reduce
        )

        # gradient_accumulation_fusion
        self.gradient_accumulation_fusion = getattr(
            config, "gradient_accumulation_fusion", False
        )

        if self.allreduce_dgrad and self.sequence_parallel:
            raise RuntimeError(
                "`allreduce_dgrad` and `sequence_parallel` cannot be enabled simultaneously."
            )

        # ----- Weight allocation -----
        if not skip_weight_param_allocation:
            if config.use_cpu_initialization:
                self.weight = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        input_size,
                        dtype=config.params_dtype,
                    )
                )
                if config.perform_initialization:
                    self.master_weight = _initialize_affine_weight_cpu(
                        self.weight,
                        output_size=output_size,
                        input_size=input_size,
                        per_partition_size=self.output_size_per_partition,
                        partition_dim=0,
                        init_method=init_method,
                        stride=stride,
                        return_master_weight=keep_master_weight_for_test,
                        params_dtype=config.params_dtype,
                        tp_rank=rank,
                        tp_world_size=world_size,
                    )
                    if self._has_dummy:
                        with torch.no_grad():
                            self.weight.data[self._real_output_cols:].zero_()
                else:
                    set_tensor_model_parallel_attributes(self.weight, True, 0, stride)
            else:
                device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
                self.weight = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        input_size,
                        device=device,
                        dtype=config.params_dtype,
                    )
                )
                if config.perform_initialization:
                    _initialize_affine_weight_gpu(
                        self.weight,
                        init_method,
                        partition_dim=0,
                        stride=stride,
                        is_expert=is_expert,
                    )
                    if self._has_dummy:
                        with torch.no_grad():
                            self.weight.data[self._real_output_cols:].zero_()
                else:
                    set_tensor_model_parallel_attributes(self.weight, True, 0, stride)

            # Mark for distributed optimizer: expert weights skip allreduce
            setattr(self.weight, "allreduce", not (is_expert and self.expert_parallel))

            # Register gradient hook to keep dummy rows zeroed
            if self._has_dummy and config.perform_initialization:
                self._register_dummy_weight_grad_hook()
        else:
            self.weight = None

        # ----- Bias allocation -----
        if bias:
            if config.use_cpu_initialization:
                self.bias = Parameter(
                    torch.empty(self.output_size_per_partition, dtype=config.params_dtype)
                )
            else:
                device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
                self.bias = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        device=device,
                        dtype=config.params_dtype,
                    )
                )
            if config.perform_initialization:
                with torch.no_grad():
                    self.bias.zero_()
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            setattr(self.bias, "allreduce", not (is_expert and self.expert_parallel))
            if self._has_dummy and config.perform_initialization:
                self._register_dummy_bias_grad_hook()
        else:
            self.register_parameter("bias", None)

        # Register load state dict pre-hook for TE compat
        self._register_load_state_dict_pre_hook(
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
                f"{prefix}_extra_state"
            )
        )

    def _register_dummy_weight_grad_hook(self) -> None:
        """Register gradient hook to zero dummy weight rows (keep them inert)."""
        real_cols = self._real_output_cols

        def _zero_dummy_grad(grad: torch.Tensor) -> torch.Tensor:
            if grad is not None and grad.shape[0] > real_cols:
                grad = grad.clone()
                grad[real_cols:].zero_()
            return grad

        self.weight.register_hook(_zero_dummy_grad)

    def _register_dummy_bias_grad_hook(self) -> None:
        """Register gradient hook to zero dummy bias elements."""
        real_cols = self._real_output_cols

        def _zero_dummy_bias_grad(grad: torch.Tensor) -> torch.Tensor:
            if grad is not None and grad.shape[0] > real_cols:
                grad = grad.clone()
                grad[real_cols:].zero_()
            return grad

        self.bias.register_hook(_zero_dummy_bias_grad)

    def _forward_impl(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        allreduce_dgrad: bool,
    ) -> torch.Tensor:
        """Dispatch to frozen-weight or trainable-weight linear."""
        if not weight.requires_grad:
            return linear_with_frozen_weight(
                input=input,
                weight=weight,
                bias=bias,
                gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                allreduce_dgrad=allreduce_dgrad,
                sequence_parallel=False if self.explicit_expert_comm else self.sequence_parallel,
                grad_output_buffer=(
                    self.grad_output_buffer
                    if getattr(self.config, "defer_embedding_wgrad_compute", False)
                    else None
                ),
                wgrad_deferral_limit=(
                    getattr(self.config, "wgrad_deferral_limit", 0)
                    if getattr(self.config, "defer_embedding_wgrad_compute", False)
                    else None
                ),
                tp_group=self.tp_group,
            )
        else:
            return linear_with_grad_accumulation_and_async_allreduce(
                input=input,
                weight=weight,
                bias=bias,
                gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                allreduce_dgrad=allreduce_dgrad,
                sequence_parallel=False if self.explicit_expert_comm else self.sequence_parallel,
                grad_output_buffer=(
                    self.grad_output_buffer
                    if getattr(self.config, "defer_embedding_wgrad_compute", False)
                    else None
                ),
                wgrad_deferral_limit=(
                    getattr(self.config, "wgrad_deferral_limit", 0)
                    if getattr(self.config, "defer_embedding_wgrad_compute", False)
                    else None
                ),
                tp_group=self.tp_group,
            )

    def forward(
        self,
        input_: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            input_:               Input tensor of shape ``[*, input_size]``.
            weight:               External weight (only when skip_weight_param_allocation=True).
            runtime_gather_output: Override gather_output at runtime.

        Returns:
            Tuple of:
            * output tensor:
              - gather_output=True:  shape ``[*, output_size]`` (padding stripped).
              - gather_output=False: shape ``[*, output_size_per_partition]``.
            * bias tensor when ``skip_bias_add=True``, else ``None``.
        """
        if weight is None:
            if self.weight is None:
                raise RuntimeError(
                    "weight was not supplied to ColumnParallelLinear.forward() "
                    "but skip_weight_param_allocation=True was set."
                )
            weight = self.weight
        else:
            expected = (self.output_size_per_partition, self.input_size)
            if weight.shape != expected:
                raise RuntimeError(
                    f"Supplied weight shape {tuple(weight.shape)} != expected {expected}"
                )

        bias = self.bias if not self.skip_bias_add else None
        world_size = _pg_world_size(self.tp_group)

        # Input handling: copy-to-region for standard path
        if (
            self.allreduce_dgrad
            or self.sequence_parallel
            or self.explicit_expert_comm
            or self.disable_grad_reduce
        ):
            input_parallel = input_
        else:
            input_parallel = _CopyToModelParallelRegion.apply(input_, self.tp_group)

        # Activation buffering for deferred embedding wgrad
        if getattr(self.config, "defer_embedding_wgrad_compute", False):
            wgrad_limit = getattr(self.config, "wgrad_deferral_limit", 0)
            if wgrad_limit == 0 or len(self.embedding_activation_buffer) < wgrad_limit:
                self.embedding_activation_buffer.append(input_parallel)

        # GEMM
        allreduce_dgrad = False if self.explicit_expert_comm else self.allreduce_dgrad
        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=weight,
            bias=bias,
            allreduce_dgrad=allreduce_dgrad,
        )

        # Gather output
        gather_output = self.gather_output
        if runtime_gather_output is not None:
            gather_output = runtime_gather_output

        if gather_output and world_size > 1:
            # All-gather along the last dimension → [*, out_per_rank * tp]
            gathered = _gather_along_last_dim(output_parallel, self.tp_group)
            # Strip padding columns to get the true [*, output_size]
            output = gathered[..., :self.output_size]
        else:
            # Without gather, strip dummy cols from this rank's partial result
            if self._has_dummy:
                output = output_parallel[..., :self._real_output_cols]
            else:
                output = output_parallel

        # Bias for skip_bias_add path
        output_bias: Optional[torch.Tensor] = None
        if self.skip_bias_add and self.bias is not None:
            if gather_output and world_size > 1:
                bias_gathered = _gather_along_last_dim(self.bias.unsqueeze(0), self.tp_group)
                output_bias = bias_gathered.squeeze(0)[..., :self.output_size]
            else:
                output_bias = (
                    self.bias[:self._real_output_cols] if self._has_dummy else self.bias
                )

        return output, output_bias

    def backward_dw(self) -> None:
        """Compute weight gradients if delay_wgrad_compute is enabled. (no-op here)"""
        pass

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> dict:
        """Return sharded state dict for distributed checkpointing (axis 0 for weight and bias)."""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        try:
            from deepspeed.core.utils import make_tp_sharded_tensor_for_checkpoint
            kwargs = {"tp_group": self.tp_group}
            if metadata:
                kwargs["dp_cp_group"] = metadata.get("dp_cp_group")
            result = {}
            if "weight" in state_dict:
                result[f"{prefix}weight"] = make_tp_sharded_tensor_for_checkpoint(
                    state_dict["weight"], f"{prefix}weight", tp_axis=0, **kwargs
                )
            if "bias" in state_dict:
                result[f"{prefix}bias"] = make_tp_sharded_tensor_for_checkpoint(
                    state_dict["bias"], f"{prefix}bias", tp_axis=0, **kwargs
                )
            return result
        except (ImportError, Exception):
            return {f"{prefix}{k}": v for k, v in state_dict.items()}

    def set_extra_state(self, state: Any) -> None:
        """Extra state is ignored (TE compatibility)."""

    def get_extra_state(self) -> None:
        """Keep compatibility with TransformerEngine state dict."""
        return None

    def extra_repr(self) -> str:
        tp = _pg_world_size(self.tp_group)
        use_bias = self.bias is not None
        padded = self.output_size_per_partition * tp
        return (
            f"in_features={self.input_size}, "
            f"out_features={self.output_size}, "
            f"out_padded={padded}, "
            f"bias={use_bias}, "
            f"TP={tp}, "
            f"is_expert={self.is_expert}, "
            f"dummy_cols_this_rank={self.output_size_per_partition - self._real_output_cols}"
        )


# ---------------------------------------------------------------------------
# RowParallelLinear
# ---------------------------------------------------------------------------

class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism + heterogeneous input-dim padding.

    Splits weight matrix along the input dimension across TP ranks.
    Supports non-divisible input sizes via ceiling-padding.

    Y = X A^T + b   where A is [output_size, input_size].
    A is partitioned row-wise (input dimension) so each rank holds
    A_i of shape [output_size, input_size_per_partition].

    Expert-parallel support
    -----------------------
    When is_expert=True:
    * Expert-parallel RNG state is used for weight init.
    * ``weight.allreduce = False``.
    * Explicit-expert-comm path skips all-reduce and returns output directly.

    Args:
        input_size:         Full input feature dimension (before TP split).
        output_size:        Output feature dimension (not split).
        config:             ModelParallelConfig.
        bias:               Whether to add a bias term (NOT split across TP).
        input_is_parallel:  If True, input is already scattered across TP ranks.
        init_method:        Weight initialiser callable.
        stride:             Partition stride for strided linear layers.
        keep_master_weight_for_test: Return master weight (only for testing).
        skip_bias_add:      If True, return bias as second element instead of
                            adding it in forward.
        is_expert:          Whether this layer is an MoE expert.
        tp_group:           Explicit TP process group.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: ModelParallelConfig,
        bias: bool = True,
        input_is_parallel: bool = False,
        init_method: Optional[Callable] = None,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        skip_bias_add: bool = False,
        is_expert: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add
        self.config = config
        self.is_expert = is_expert
        self.expert_parallel = config.expert_model_parallel_size > 1
        self.gradient_accumulation_fusion = getattr(
            config, "gradient_accumulation_fusion", False
        )

        if init_method is None:
            init_method = nn.init.xavier_normal_

        # Resolve TP group (expert-aware)
        self.tp_group = _resolve_tp_group(tp_group, is_expert=is_expert)
        world_size = _pg_world_size(self.tp_group)
        rank = _pg_rank(self.tp_group)

        # Expert-comm path
        self.explicit_expert_comm = is_expert and (world_size > 1 or self.expert_parallel)

        # Sequence-parallel
        self.sequence_parallel = getattr(config, "sequence_parallel", False)
        if self.sequence_parallel and not input_is_parallel:
            raise RuntimeError(
                "To enable `sequence_parallel` in RowParallelLinear, "
                "`input_is_parallel` must be `True`."
            )

        # Ceiling-padded partition size
        self.input_size_per_partition = _padded_partition_size(input_size, world_size)

        # Real (non-dummy) input columns on this rank
        real_start, real_end = _rank_output_slice(input_size, rank, world_size)
        self._real_input_cols = real_end - real_start
        self._has_dummy = self._real_input_cols < self.input_size_per_partition

        # ----- Weight allocation -----
        if config.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    output_size,
                    self.input_size_per_partition,
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                self.master_weight = _initialize_affine_weight_cpu(
                    self.weight,
                    output_size=output_size,
                    input_size=input_size,
                    per_partition_size=self.input_size_per_partition,
                    partition_dim=1,
                    init_method=init_method,
                    stride=stride,
                    return_master_weight=keep_master_weight_for_test,
                    params_dtype=config.params_dtype,
                    tp_rank=rank,
                    tp_world_size=world_size,
                )
                if self._has_dummy:
                    with torch.no_grad():
                        self.weight.data[:, self._real_input_cols:].zero_()
            else:
                set_tensor_model_parallel_attributes(self.weight, True, 1, stride)
        else:
            device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            self.weight = Parameter(
                torch.empty(
                    output_size,
                    self.input_size_per_partition,
                    device=device,
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight,
                    init_method,
                    partition_dim=1,
                    stride=stride,
                    is_expert=is_expert,
                )
                if self._has_dummy:
                    with torch.no_grad():
                        self.weight.data[:, self._real_input_cols:].zero_()
            else:
                set_tensor_model_parallel_attributes(self.weight, True, 1, stride)

        setattr(self.weight, "allreduce", not (is_expert and self.expert_parallel))

        # Register dummy-col gradient hook
        if self._has_dummy and config.perform_initialization:
            self._register_dummy_grad_hook()

        # ----- Bias allocation (not partitioned) -----
        if bias:
            if config.use_cpu_initialization:
                self.bias = Parameter(
                    torch.empty(output_size, dtype=config.params_dtype)
                )
            else:
                device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
                self.bias = Parameter(
                    torch.empty(output_size, device=device, dtype=config.params_dtype)
                )
            if config.perform_initialization:
                with torch.no_grad():
                    self.bias.zero_()
            # Bias is replicated, not TP-sharded
            self.bias.tensor_model_parallel = False
            setattr(self.bias, "allreduce", not (is_expert and self.expert_parallel))
            setattr(self.bias, "sequence_parallel", self.sequence_parallel)
        else:
            self.register_parameter("bias", None)

        # Register load state dict pre-hook for TE compat
        self._register_load_state_dict_pre_hook(
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
                f"{prefix}_extra_state"
            )
        )

    def _register_dummy_grad_hook(self) -> None:
        """Register gradient hook to zero dummy weight columns."""
        real_cols = self._real_input_cols

        def _zero_dummy_grad(grad: torch.Tensor) -> torch.Tensor:
            if grad is not None and grad.shape[1] > real_cols:
                grad = grad.clone()
                grad[:, real_cols:].zero_()
            return grad

        self.weight.register_hook(_zero_dummy_grad)

    def _forward_impl(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch to frozen-weight or trainable-weight linear (no bias; added separately)."""
        if not weight.requires_grad:
            return linear_with_frozen_weight(
                input=input,
                weight=weight,
                bias=None,
                gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                allreduce_dgrad=False,
                sequence_parallel=False,
                tp_group=None,
                grad_output_buffer=None,
                wgrad_deferral_limit=None,
            )
        else:
            return linear_with_grad_accumulation_and_async_allreduce(
                input=input,
                weight=weight,
                bias=None,
                gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                allreduce_dgrad=False,
                sequence_parallel=False,
                tp_group=None,
                grad_output_buffer=None,
                wgrad_deferral_limit=None,
            )

    def forward(
        self,
        input_: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            input_: Input tensor.
              * If ``input_is_parallel=True``: already scattered shard
                ``[seq, batch, input_size_per_partition]``.
              * Otherwise: full shape ``[seq, batch, input_size]``.

        Returns:
            Tuple of:
            * output tensor of shape ``[seq, batch, output_size]``.
            * bias tensor when ``skip_bias_add=True``, else ``None``.
        """
        world_size = _pg_world_size(self.tp_group)

        # Scatter or pass-through
        if self.input_is_parallel or world_size == 1:
            input_parallel = input_
        else:
            # Scatter along last dim (with padding for non-divisible input_size)
            input_parallel = _scatter_along_last_dim_padded(
                input_, self.tp_group, self.input_size_per_partition
            )

        # Local GEMM
        output_parallel = self._forward_impl(input_parallel, self.weight)

        # Reduce partial results
        if self.explicit_expert_comm:
            assert self.skip_bias_add, (
                "explicit_expert_comm requires skip_bias_add=True"
            )
            output_ = output_parallel
        elif self.sequence_parallel and world_size > 1:
            # reduce_scatter: sum across TP ranks, scatter along dim 0 (sequence)
            from deepspeed.core.tensor_parallel.mappings import (
                reduce_scatter_to_sequence_parallel_region,
            )
            output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
        elif world_size > 1:
            torch.distributed.all_reduce(output_parallel, group=self.tp_group)
            output_ = output_parallel
        else:
            output_ = output_parallel

        # Add non-parallelised bias
        if not self.skip_bias_add:
            output = (output_ + self.bias) if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias

        return output, output_bias

    def backward_dw(self) -> None:
        """Compute weight gradients if delay_wgrad_compute is enabled. (no-op here)"""
        pass

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> dict:
        """Return sharded state dict for distributed checkpointing (axis 1 for weight)."""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        try:
            from deepspeed.core.utils import make_tp_sharded_tensor_for_checkpoint
            kwargs = {"tp_group": self.tp_group}
            if metadata:
                kwargs["dp_cp_group"] = metadata.get("dp_cp_group")
            result = {}
            if "weight" in state_dict:
                result[f"{prefix}weight"] = make_tp_sharded_tensor_for_checkpoint(
                    state_dict["weight"], f"{prefix}weight", tp_axis=1, **kwargs
                )
            # Bias is replicated (not sharded)
            if "bias" in state_dict and self.bias is not None:
                result[f"{prefix}bias"] = state_dict["bias"]
            return result
        except (ImportError, Exception):
            return {f"{prefix}{k}": v for k, v in state_dict.items()}

    def set_extra_state(self, state: Any) -> None:
        """Extra state is ignored (TE compatibility)."""

    def get_extra_state(self) -> None:
        """Keep compatibility with TransformerEngine state dict."""
        return None

    def extra_repr(self) -> str:
        tp = _pg_world_size(self.tp_group)
        use_bias = self.bias is not None
        padded = self.input_size_per_partition * tp
        return (
            f"in_features={self.input_size}, "
            f"in_padded={padded}, "
            f"out_features={self.output_size}, "
            f"bias={use_bias}, "
            f"TP={tp}, "
            f"is_expert={self.is_expert}, "
            f"dummy_cols_this_rank={self.input_size_per_partition - self._real_input_cols}"
        )


# ---------------------------------------------------------------------------
# Collective communication helpers
# ---------------------------------------------------------------------------

def _gather_along_last_dim(
    tensor: torch.Tensor,
    group: Optional[torch.distributed.ProcessGroup],
) -> torch.Tensor:
    """All-gather *tensor* along its last dimension.

    Each rank contributes a shard of size ``last_dim``.  The output last
    dim is ``last_dim * world_size``.  This is the inverse of
    ``_scatter_along_last_dim``.

    Args:
        tensor: Local shard (any shape).
        group:  TP process group.
    Returns:
        Gathered tensor with last dim = ``last_dim * world_size``.
    """
    if group is None:
        return tensor
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return tensor

    tensor = tensor.contiguous()
    output_shape = list(tensor.shape)
    output_shape[-1] = tensor.shape[-1] * world_size
    output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
    torch.distributed.all_gather_into_tensor(output, tensor, group=group)
    return output


def _scatter_along_last_dim(
    tensor: torch.Tensor,
    group: Optional[torch.distributed.ProcessGroup],
) -> torch.Tensor:
    """Scatter *tensor* along its last dimension to the calling rank's shard.

    Requires last_dim % world_size == 0.  Use ``_scatter_along_last_dim_padded``
    for heterogeneous configs.

    Args:
        tensor: Full tensor (replicated on all ranks).
        group:  TP process group.
    Returns:
        This rank's shard (last_dim // world_size).
    """
    if group is None:
        return tensor
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return tensor

    rank = torch.distributed.get_rank(group=group)
    last_dim = tensor.shape[-1]
    assert last_dim % world_size == 0, (
        f"Last dim {last_dim} not divisible by tp_world_size {world_size}. "
        f"Use _scatter_along_last_dim_padded for non-divisible sizes."
    )
    per_rank = last_dim // world_size
    return tensor[..., rank * per_rank: (rank + 1) * per_rank].contiguous()


def _scatter_along_last_dim_padded(
    tensor: torch.Tensor,
    group: Optional[torch.distributed.ProcessGroup],
    per_rank_size: int,
) -> torch.Tensor:
    """Scatter *tensor* along last dim with ceiling-padding support.

    Pads *tensor* to ``per_rank_size * world_size`` with zeros if needed,
    then returns this rank's shard of size ``per_rank_size``.

    The dummy positions (beyond the true last-dim size) are zero so the
    corresponding dummy weight columns contribute zero to the GEMM output.

    Args:
        tensor:        Input tensor (full, replicated on all ranks).
        group:         TP process group.
        per_rank_size: Target per-rank size (ceiling-padded).
    Returns:
        This rank's shard of size ``per_rank_size``.
    """
    if group is None:
        return tensor
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return tensor

    rank = torch.distributed.get_rank(group=group)
    last_dim = tensor.shape[-1]
    total_padded = per_rank_size * world_size

    if last_dim < total_padded:
        pad_size = total_padded - last_dim
        tensor = F.pad(tensor, (0, pad_size))

    start = rank * per_rank_size
    return tensor[..., start: start + per_rank_size].contiguous()


# ---------------------------------------------------------------------------
# Autograd function: copy to TP region (identity forward, all-reduce backward)
# ---------------------------------------------------------------------------

class _CopyToModelParallelRegion(torch.autograd.Function):
    """Identity in forward; all-reduce across the TP group in backward.

    Used by ColumnParallelLinear to ensure that input-gradient tensors are
    properly reduced across TP ranks after the backward GEMM.
    """

    @staticmethod
    def forward(
        ctx,
        input_: torch.Tensor,
        group: Optional[torch.distributed.ProcessGroup],
    ) -> torch.Tensor:
        ctx.group = group
        return input_

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        group = ctx.group
        if group is not None and torch.distributed.get_world_size(group=group) > 1:
            torch.distributed.all_reduce(grad_output, group=group)
        return grad_output, None
