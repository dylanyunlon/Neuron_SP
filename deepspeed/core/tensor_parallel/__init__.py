# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Tensor parallelism layers.

Implements vocabulary-parallel embedding, column-parallel linear, and
row-parallel linear following the Megatron-LM design.  When TP=1 these
layers behave identically to their standard PyTorch counterparts so the
code is correct on PCIe-only clusters with no NVLink.

Communication primitives
------------------------
* VocabParallelEmbedding : all-reduce after local lookup (TP>1)
* ColumnParallelLinear   : optional all-gather on output   (TP>1, gather_output=True)
* RowParallelLinear      : all-reduce of partial outputs   (TP>1)

All collectives use the TP process group returned by parallel_state;
when parallel_state is not yet initialised we fall back to a safe
no-op path so that unit tests that skip distributed init still work.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from deepspeed.core.model_parallel_config import ModelParallelConfig


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


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _divide(numerator: int, denominator: int) -> int:
    """Integer division with assertion that it divides evenly."""
    assert numerator % denominator == 0, (
        f"{numerator} is not divisible by {denominator}"
    )
    return numerator // denominator


def _vocab_range(num_embeddings: int, tp_rank: int, tp_world_size: int) -> Tuple[int, int]:
    """Return [start, end) vocab indices for this TP rank.

    We round up num_embeddings to the next multiple of tp_world_size so
    every rank holds the same number of rows.  Rows beyond num_embeddings
    are padding and never looked up.
    """
    per_partition = _divide(
        math.ceil(num_embeddings / tp_world_size) * tp_world_size,
        tp_world_size,
    )
    start = tp_rank * per_partition
    end = min(start + per_partition, num_embeddings)
    return start, end


# ---------------------------------------------------------------------------
# Weight initialisation helpers
# ---------------------------------------------------------------------------

def _init_weight_cpu(
    weight: Parameter,
    full_shape: Tuple[int, int],  # (out, in) of the full (un-partitioned) matrix
    partition_dim: int,
    init_method,
    params_dtype: torch.dtype,
    tp_rank: int,
    tp_world_size: int,
) -> None:
    """Build full weight on CPU, slice the relevant chunk into *weight*."""
    master = torch.empty(full_shape, dtype=torch.float32)
    init_method(master)
    master = master.to(dtype=params_dtype)
    chunks = torch.chunk(master, tp_world_size, dim=partition_dim)
    with torch.no_grad():
        weight.data.copy_(chunks[tp_rank].contiguous())


def _init_weight_gpu(
    weight: Parameter,
    init_method,
) -> None:
    """Initialise weight in-place on GPU using RNG tracker when available."""
    try:
        from deepspeed.core.tensor_parallel.random import get_cuda_rng_tracker  # noqa: F401
        with get_cuda_rng_tracker().fork():
            init_method(weight)
    except (ImportError, AttributeError):
        # No RNG tracker – initialise directly (acceptable for TP=1)
        init_method(weight)


# ---------------------------------------------------------------------------
# VocabParallelEmbedding
# ---------------------------------------------------------------------------

class VocabParallelEmbedding(nn.Module):
    """Embedding parallelized across TP ranks.

    The full vocabulary of size *num_embeddings* is split evenly across the
    TP group.  Each rank stores ``num_embeddings // tp_world_size`` rows.
    Forward performs a local embedding lookup followed by an all-reduce so
    that every rank receives the full embedding vector.

    When TP=1 this is equivalent to ``nn.Embedding``.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        config: ModelParallelConfig,
        init_method: Optional[callable] = None,
    ) -> None:
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.config = config

        if init_method is None:
            init_method = nn.init.normal_

        tp_world_size = _get_tp_world_size()
        tp_rank = _get_tp_rank()

        self.vocab_start_index, self.vocab_end_index = _vocab_range(
            num_embeddings, tp_rank, tp_world_size
        )
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index

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
        else:
            device = (
                torch.cuda.current_device()
                if torch.cuda.is_available()
                else "cpu"
            )
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    device=device,
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _init_weight_gpu(self.weight, init_method)

        # Mark this parameter as TP-sharded (dimension 0) so the
        # distributed optimizer / checkpoint code can shard it correctly.
        self.weight.tensor_model_parallel = True
        self.weight.partition_dim = 0
        self.weight.partition_stride = 1

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Embedding lookup with all-reduce for TP>1.

        Args:
            input_: Integer token index tensor of any shape.

        Returns:
            Float tensor of shape ``(*input_.shape, embedding_dim)``.
        """
        tp_world_size = _get_tp_world_size()

        if tp_world_size > 1:
            # Tokens outside this rank's vocab range → look up index 0 and
            # zero out afterwards (the all-reduce will sum contributions from
            # the rank that actually owns the token).
            input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
            # Shift indices to be local; out-of-range ones become 0 (safe)
            local_input = input_.clone() - self.vocab_start_index
            local_input.clamp_(min=0)
            local_input[input_mask] = 0
        else:
            local_input = input_

        # Local embedding lookup
        output_parallel = F.embedding(local_input, self.weight)

        if tp_world_size > 1:
            # Zero out contributions for tokens not owned by this rank
            output_parallel[input_mask] = 0.0
            # Sum across all TP ranks so every rank has the full result
            tp_group = _get_tp_group()
            torch.distributed.all_reduce(output_parallel, group=tp_group)

        return output_parallel


# ---------------------------------------------------------------------------
# ColumnParallelLinear
# ---------------------------------------------------------------------------

class ColumnParallelLinear(nn.Module):
    """Linear layer with column parallelism.

    Splits weight matrix along the output dimension across TP ranks.
    Each rank computes a partial result, then all-gather if needed.

    Y = X A^T + b   where A is [output_size, input_size].
    A is partitioned column-wise (output dimension) so each rank holds
    A_i of shape [output_size // tp, input_size].

    Args:
        input_size: Input feature dimension.
        output_size: Full output feature dimension (before TP split).
        config: ModelParallelConfig.
        bias: Whether to add a bias term.
        gather_output: If True, all-gather the output so every rank sees
            the full [*, output_size] tensor.  Set False when the next
            layer is a RowParallelLinear that expects a partitioned input.
        init_method: Weight initialiser callable.
        skip_bias_add: If True, do not add bias inside forward(); instead
            return it as the second element of the output tuple so callers
            can fuse it with other element-wise ops.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: ModelParallelConfig,
        bias: bool = True,
        gather_output: bool = True,
        init_method: Optional[callable] = None,
        skip_bias_add: bool = False,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add
        self.config = config

        if init_method is None:
            init_method = nn.init.xavier_normal_

        tp_world_size = _get_tp_world_size()
        tp_rank = _get_tp_rank()

        self.output_size_per_partition = _divide(output_size, tp_world_size)

        # Weight: shape [output_size_per_partition, input_size]
        # (F.linear computes X @ weight.T, so we store the transposed layout)
        if config.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    input_size,
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _init_weight_cpu(
                    self.weight,
                    full_shape=(output_size, input_size),
                    partition_dim=0,
                    init_method=init_method,
                    params_dtype=config.params_dtype,
                    tp_rank=tp_rank,
                    tp_world_size=tp_world_size,
                )
        else:
            device = (
                torch.cuda.current_device()
                if torch.cuda.is_available()
                else "cpu"
            )
            self.weight = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    input_size,
                    device=device,
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _init_weight_gpu(self.weight, init_method)

        # Mark as TP-sharded on output dimension
        self.weight.tensor_model_parallel = True
        self.weight.partition_dim = 0
        self.weight.partition_stride = 1

        # Bias: shape [output_size_per_partition], also partitioned
        if bias:
            if config.use_cpu_initialization:
                self.bias = Parameter(
                    torch.empty(self.output_size_per_partition, dtype=config.params_dtype)
                )
            else:
                device = (
                    torch.cuda.current_device()
                    if torch.cuda.is_available()
                    else "cpu"
                )
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
            self.bias.tensor_model_parallel = True
            self.bias.partition_dim = 0
            self.bias.partition_stride = 1
        else:
            self.register_parameter("bias", None)

    def forward(
        self, input_: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            input_: Input tensor of shape ``[*, input_size]``.

        Returns:
            Tuple of:
            * output tensor – shape ``[*, output_size]`` when
              ``gather_output=True``, else ``[*, output_size_per_partition]``.
            * bias tensor when ``skip_bias_add=True``, else ``None``.
        """
        # When TP>1, copy input to all TP ranks (no-op for TP=1; the
        # autograd function handles the all-reduce on dgrad in backward).
        tp_world_size = _get_tp_world_size()
        tp_group = _get_tp_group()

        if tp_world_size > 1 and not self.config.sequence_parallel:
            # Identity in forward; all-reduce in backward so input grads
            # are correctly accumulated across TP ranks.
            input_parallel = _CopyToModelParallelRegion.apply(input_, tp_group)
        else:
            input_parallel = input_

        # Local GEMM: [*, input_size] x [input_size, out/tp]^T → [*, out/tp]
        bias = self.bias if not self.skip_bias_add else None
        output_parallel = F.linear(input_parallel, self.weight, bias)

        if self.gather_output and tp_world_size > 1:
            # All-gather along the last dimension to reconstruct full output
            output = _gather_along_last_dim(output_parallel, tp_group)
        else:
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


# ---------------------------------------------------------------------------
# RowParallelLinear
# ---------------------------------------------------------------------------

class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism.

    Splits weight matrix along the input dimension across TP ranks.
    Each rank computes a partial result, then all-reduce.

    Y = X A^T + b   where A is [output_size, input_size].
    A is partitioned row-wise (input dimension) so each rank holds
    A_i of shape [output_size, input_size // tp].

    Args:
        input_size: Full input feature dimension (before TP split).
        output_size: Output feature dimension (not split).
        config: ModelParallelConfig.
        bias: Whether to add a bias term.  Bias is NOT split across TP ranks;
            only rank 0 logically "owns" it but all ranks add it after the
            all-reduce.
        input_is_parallel: If True, the input has already been scattered
            across TP ranks (e.g. it comes from a ColumnParallelLinear with
            gather_output=False) and we skip the scatter step.
        init_method: Weight initialiser callable.
        skip_bias_add: If True, return bias as second element instead of
            adding it in forward.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: ModelParallelConfig,
        bias: bool = True,
        input_is_parallel: bool = False,
        init_method: Optional[callable] = None,
        skip_bias_add: bool = False,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add
        self.config = config

        if init_method is None:
            init_method = nn.init.xavier_normal_

        tp_world_size = _get_tp_world_size()
        tp_rank = _get_tp_rank()

        self.input_size_per_partition = _divide(input_size, tp_world_size)

        # Weight: shape [output_size, input_size_per_partition]
        if config.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    output_size,
                    self.input_size_per_partition,
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _init_weight_cpu(
                    self.weight,
                    full_shape=(output_size, input_size),
                    partition_dim=1,
                    init_method=init_method,
                    params_dtype=config.params_dtype,
                    tp_rank=tp_rank,
                    tp_world_size=tp_world_size,
                )
        else:
            device = (
                torch.cuda.current_device()
                if torch.cuda.is_available()
                else "cpu"
            )
            self.weight = Parameter(
                torch.empty(
                    output_size,
                    self.input_size_per_partition,
                    device=device,
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _init_weight_gpu(self.weight, init_method)

        # Mark as TP-sharded on input dimension
        self.weight.tensor_model_parallel = True
        self.weight.partition_dim = 1
        self.weight.partition_stride = 1

        # Bias: full shape [output_size], not split across TP ranks
        if bias:
            if config.use_cpu_initialization:
                self.bias = Parameter(
                    torch.empty(output_size, dtype=config.params_dtype)
                )
            else:
                device = (
                    torch.cuda.current_device()
                    if torch.cuda.is_available()
                    else "cpu"
                )
                self.bias = Parameter(
                    torch.empty(output_size, device=device, dtype=config.params_dtype)
                )
            if config.perform_initialization:
                with torch.no_grad():
                    self.bias.zero_()
            # Bias is replicated, not TP-sharded
            self.bias.tensor_model_parallel = False
        else:
            self.register_parameter("bias", None)

    def forward(
        self, input_: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            input_: Input tensor.
              * If ``input_is_parallel=True``: shape ``[*, input_size // tp]``
                (already scattered by the preceding ColumnParallelLinear).
              * Otherwise: shape ``[*, input_size]`` which will be scattered
                to ``[*, input_size // tp]`` locally.

        Returns:
            Tuple of:
            * output tensor of shape ``[*, output_size]``.
            * bias tensor when ``skip_bias_add=True``, else ``None``.
        """
        tp_world_size = _get_tp_world_size()
        tp_group = _get_tp_group()

        if self.input_is_parallel or tp_world_size == 1:
            input_parallel = input_
        else:
            # Scatter input along the last dimension across TP ranks
            input_parallel = _scatter_along_last_dim(input_, tp_group)

        # Local GEMM: [*, in/tp] x [in/tp, out]^T → [*, out]
        output_parallel = F.linear(input_parallel, self.weight)

        # All-reduce partial results across TP ranks
        if tp_world_size > 1:
            torch.distributed.all_reduce(output_parallel, group=tp_group)

        # Add non-parallelised bias
        if not self.skip_bias_add:
            output = (output_parallel + self.bias) if self.bias is not None else output_parallel
            output_bias = None
        else:
            output = output_parallel
            output_bias = self.bias

        return output, output_bias


# ---------------------------------------------------------------------------
# Collective communication helpers
# ---------------------------------------------------------------------------

def _gather_along_last_dim(
    tensor: torch.Tensor,
    group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """All-gather *tensor* along its last dimension.

    Each rank contributes a shard of size ``last_dim // tp``.  The output
    has the full ``last_dim``.  This is the inverse of
    ``_scatter_along_last_dim``.
    """
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return tensor

    # Contiguous memory is required for the collective
    tensor = tensor.contiguous()

    # Pre-allocate output buffer across all shards
    output_shape = list(tensor.shape)
    output_shape[-1] = tensor.shape[-1] * world_size
    output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)

    # all_gather_into_tensor is the modern API (torch ≥ 1.13)
    torch.distributed.all_gather_into_tensor(output, tensor, group=group)
    return output


def _scatter_along_last_dim(
    tensor: torch.Tensor,
    group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """Scatter *tensor* along its last dimension to the calling rank's shard.

    This is a local slice (no communication); the caller must ensure the
    full tensor is already replicated on all ranks before calling this.
    """
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return tensor

    rank = torch.distributed.get_rank(group=group)
    last_dim = tensor.shape[-1]
    assert last_dim % world_size == 0, (
        f"Last dim {last_dim} not divisible by tp_world_size {world_size}"
    )
    per_rank = last_dim // world_size
    return tensor[..., rank * per_rank : (rank + 1) * per_rank].contiguous()


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
        group: torch.distributed.ProcessGroup,
    ) -> torch.Tensor:
        ctx.group = group
        return input_

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        torch.distributed.all_reduce(grad_output, group=ctx.group)
        return grad_output, None
