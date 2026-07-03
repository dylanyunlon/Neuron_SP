# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Tensor-parallel layer implementations.

Contains:
  * VocabParallelEmbedding
  * ColumnParallelLinear          (with heterogeneous head-count padding)
  * RowParallelLinear             (with heterogeneous head-count padding)
  * Supporting autograd functions and collective helpers

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

These follow the Megatron-LM design (megatron/core/tensor_parallel/layers.py).
When TP=1 all layers behave identically to their standard PyTorch counterparts,
so the code is correct on PCIe-only clusters with no NVLink.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

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
# TP attribute helpers (mirrors Megatron's layers.py public API)
# ---------------------------------------------------------------------------

def set_tensor_model_parallel_attributes(tensor: torch.Tensor, is_parallel: bool,
                                          dim: int, stride: int) -> None:
    """Attach TP sharding metadata to a tensor/parameter.

    Args:
        tensor:      The parameter to annotate.
        is_parallel: Whether this tensor is sharded across TP ranks.
        dim:         The dimension along which it is sharded (0 or 1).
        stride:      Partition stride (usually 1).
    """
    tensor.tensor_model_parallel = is_parallel
    tensor.partition_dim = dim
    tensor.partition_stride = stride


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor: torch.Tensor) -> None:
    """Set TP metadata defaults on *tensor* if not already set."""
    def _set(key, val):
        if not hasattr(tensor, key):
            setattr(tensor, key, val)
    _set("tensor_model_parallel", False)
    _set("partition_dim", -1)
    _set("partition_stride", 1)


def copy_tensor_model_parallel_attributes(destination_tensor: torch.Tensor,
                                           source_tensor: torch.Tensor) -> None:
    """Copy TP sharding attributes from *source_tensor* to *destination_tensor*."""
    def _copy(attr):
        if hasattr(source_tensor, attr):
            setattr(destination_tensor, attr, getattr(source_tensor, attr))
    _copy("tensor_model_parallel")
    _copy("partition_dim")
    _copy("partition_stride")


def param_is_not_tensor_parallel_duplicate(param: torch.Tensor) -> bool:
    """Return True if *param* is NOT a TP duplicate (i.e. it should be in grads).

    Replicated parameters (e.g. biases in RowParallelLinear) only need to have
    their gradients reduced once; this predicate identifies them.
    """
    return (
        hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel
    ) or _get_tp_rank() == 0


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


def _rank_output_slice(total_size: int, tp_rank: int, tp_world_size: int) -> Tuple[int, int]:
    """Return [start, end) column range for *tp_rank* (unpadded).

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


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _divide_strict(numerator: int, denominator: int) -> int:
    """Integer division with assertion that it divides evenly (for internal use only)."""
    assert numerator % denominator == 0, (
        f"{numerator} is not divisible by {denominator}"
    )
    return numerator // denominator


def _vocab_range(num_embeddings: int, tp_rank: int, tp_world_size: int) -> Tuple[int, int]:
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

def _init_weight_cpu(
    weight: Parameter,
    full_shape: Tuple[int, int],   # (out, in) of the full (un-partitioned) matrix
    partition_dim: int,
    init_method: Callable,
    params_dtype: torch.dtype,
    tp_rank: int,
    tp_world_size: int,
    padded_total_size: Optional[int] = None,
) -> None:
    """Build full weight on CPU, slice the relevant chunk into *weight*.

    When *padded_total_size* is given the master is first padded to that size
    along *partition_dim* (with zeros) before slicing, so the slice shape
    matches *weight* exactly even for non-divisible configs.
    """
    master = torch.empty(full_shape, dtype=torch.float32)
    init_method(master)
    master = master.to(dtype=params_dtype)

    per_rank = _padded_partition_size(
        full_shape[partition_dim] if padded_total_size is None else full_shape[partition_dim],
        tp_world_size,
    )
    real_size = full_shape[partition_dim]
    padded = per_rank * tp_world_size

    if padded > real_size:
        # Pad master along partition_dim with zeros
        pad_size = padded - real_size
        pad_shape = list(full_shape)
        pad_shape[partition_dim] = pad_size
        padding = torch.zeros(pad_shape, dtype=params_dtype)
        master = torch.cat([master, padding], dim=partition_dim)

    chunks = torch.chunk(master, tp_world_size, dim=partition_dim)
    with torch.no_grad():
        weight.data.copy_(chunks[tp_rank].contiguous())


def _init_weight_gpu(weight: Parameter, init_method: Callable) -> None:
    """Initialise weight in-place on GPU using RNG tracker when available."""
    try:
        from deepspeed.core.tensor_parallel.random import get_cuda_rng_tracker
        with get_cuda_rng_tracker().fork():
            init_method(weight)
    except (ImportError, AttributeError):
        # No RNG tracker – initialise directly (acceptable for TP=1)
        init_method(weight)


# ---------------------------------------------------------------------------
# linear_with_grad_accumulation_and_async_allreduce
# ---------------------------------------------------------------------------

def linear_with_grad_accumulation_and_async_allreduce(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool = False,
    async_grad_allreduce: bool = False,
    sequence_parallel: bool = False,
    grad_output_buffer: Optional[torch.Tensor] = None,
    wgrad_deferral_limit: int = 0,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Linear layer with optional async all-reduce and gradient accumulation fusion.

    This is a simplified version of Megatron's implementation that provides
    the same interface but falls back to standard F.linear when advanced
    features are unavailable.

    When sequence_parallel=True the input is a local sequence shard;
    all-gather across TP ranks before the GEMM using all_gather_into_tensor.
    """
    # When sequence_parallel, all-gather input shards first (dim 0 = sequence)
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

    output = F.linear(input, weight, bias)

    if async_grad_allreduce and tp_group is not None:
        world_size = torch.distributed.get_world_size(group=tp_group)
        if world_size > 1:
            torch.distributed.all_reduce(output, group=tp_group)

    return output


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
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        config: ModelParallelConfig,
        init_method: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.config = config

        if init_method is None:
            init_method = nn.init.normal_

        tp_world_size = _get_tp_world_size()
        tp_rank = _get_tp_rank()

        # Ceiling-padded partition: handles non-divisible vocab sizes
        self.vocab_start_index, self.vocab_end_index = _vocab_range(
            num_embeddings, tp_rank, tp_world_size
        )
        self.num_embeddings_per_partition = _padded_partition_size(num_embeddings, tp_world_size)
        # Real rows (non-dummy) owned by this rank
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
                # Zero out dummy rows (if any)
                if self._real_rows < self.num_embeddings_per_partition:
                    with torch.no_grad():
                        self.weight.data[self._real_rows:].zero_()
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
                # Zero out dummy rows (if any)
                if self._real_rows < self.num_embeddings_per_partition:
                    with torch.no_grad():
                        self.weight.data[self._real_rows:].zero_()

        set_tensor_model_parallel_attributes(self.weight, True, 0, 1)

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
            local_input.clamp_(min=0, max=self._real_rows - 1)
            local_input[input_mask] = 0
        else:
            local_input = input_

        # Local embedding lookup (only real rows are ever looked up)
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
    """Linear layer with column parallelism + heterogeneous head-count padding.

    Splits weight matrix along the output dimension across TP ranks.
    Supports non-divisible output sizes (e.g. 32 heads / 5 GPUs) via
    ceiling-padding with dummy rows that are zero-initialised and masked
    away before returning results.

    Y = X A^T + b   where A is [output_size, input_size].
    A is partitioned column-wise (output dimension) so each rank holds
    A_i of shape [output_size_per_partition, input_size], where
    output_size_per_partition = ceil(output_size / tp) // tp (padded).

    Dummy-head padding mechanics
    ----------------------------
    1. Weight rows [real_output_size_this_rank, output_size_per_partition]
       are zero-initialised and their gradients are zeroed before each step.
    2. When gather_output=True, the all-gathered tensor has shape
       [*, output_size_per_partition * tp] and we slice [:output_size] to
       strip the padding columns before returning.
    3. Bias (when used) is also truncated after gather.

    Args:
        input_size:         Input feature dimension.
        output_size:        Full output feature dimension (before TP split).
        config:             ModelParallelConfig.
        bias:               Whether to add a bias term.
        gather_output:      If True, all-gather the output so every rank sees
                            the full [*, output_size] tensor.  Set False when
                            the next layer is a RowParallelLinear that expects
                            a partitioned input.
        init_method:        Weight initialiser callable.
        skip_bias_add:      If True, do not add bias inside forward(); instead
                            return it as the second element of the output tuple.
        num_heads:          (Optional) total number of attention heads. When set,
                            used only for documentation; the padding logic derives
                            automatically from output_size and tp_world_size.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: ModelParallelConfig,
        bias: bool = True,
        gather_output: bool = True,
        init_method: Optional[Callable] = None,
        skip_bias_add: bool = False,
        num_heads: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add
        self.config = config
        self.num_heads = num_heads

        if init_method is None:
            init_method = nn.init.xavier_normal_

        tp_world_size = _get_tp_world_size()
        tp_rank = _get_tp_rank()

        # Ceiling-padded partition size (same on every rank)
        self.output_size_per_partition = _padded_partition_size(output_size, tp_world_size)

        # Number of real (non-dummy) output columns on this rank
        real_start, real_end = _rank_output_slice(output_size, tp_rank, tp_world_size)
        self._real_output_cols = real_end - real_start
        self._has_dummy = self._real_output_cols < self.output_size_per_partition

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
                # Zero dummy rows so they never contribute to output
                if self._has_dummy:
                    with torch.no_grad():
                        self.weight.data[self._real_output_cols:].zero_()
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
                # Zero dummy rows so they never contribute to output
                if self._has_dummy:
                    with torch.no_grad():
                        self.weight.data[self._real_output_cols:].zero_()

        set_tensor_model_parallel_attributes(self.weight, True, 0, 1)

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
            set_tensor_model_parallel_attributes(self.bias, True, 0, 1)
        else:
            self.register_parameter("bias", None)

        # Register a hook to zero dummy-row gradients so they stay at zero
        if self._has_dummy and config.perform_initialization:
            self._register_dummy_grad_hook()

    def _register_dummy_grad_hook(self) -> None:
        """Register gradient hook to zero dummy rows so they stay inert."""
        real_cols = self._real_output_cols

        def _zero_dummy_grad(grad: torch.Tensor) -> torch.Tensor:
            if grad is not None and grad.shape[0] > real_cols:
                grad = grad.clone()
                grad[real_cols:].zero_()
            return grad

        self.weight.register_hook(_zero_dummy_grad)
        if self.bias is not None:
            bias_real = self._real_output_cols

            def _zero_dummy_bias_grad(grad: torch.Tensor) -> torch.Tensor:
                if grad is not None and grad.shape[0] > bias_real:
                    grad = grad.clone()
                    grad[bias_real:].zero_()
                return grad

            self.bias.register_hook(_zero_dummy_bias_grad)

    def forward(
        self, input_: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            input_: Input tensor of shape ``[*, input_size]``.

        Returns:
            Tuple of:
            * output tensor:
              - gather_output=True:  shape ``[*, output_size]`` (padding stripped).
              - gather_output=False: shape ``[*, output_size_per_partition]``
                (callers that know about padding should strip it themselves).
            * bias tensor when ``skip_bias_add=True``, else ``None``.
        """
        tp_world_size = _get_tp_world_size()
        tp_group = _get_tp_group()

        if tp_world_size > 1 and not getattr(self.config, 'sequence_parallel', False):
            # Identity in forward; all-reduce in backward so input grads
            # are correctly accumulated across TP ranks.
            input_parallel = _CopyToModelParallelRegion.apply(input_, tp_group)
        else:
            input_parallel = input_

        # Local GEMM: [*, input_size] x [input_size, out_per_rank]^T → [*, out_per_rank]
        bias = self.bias if not self.skip_bias_add else None
        output_parallel = F.linear(input_parallel, self.weight, bias)

        if self.gather_output and tp_world_size > 1:
            # All-gather along the last dimension → [*, out_per_rank * tp]
            gathered = _gather_along_last_dim(output_parallel, tp_group)
            # Strip padding columns to get the true [*, output_size]
            output = gathered[..., :self.output_size]
        else:
            # When not gathering, strip dummy cols from this rank's slice
            if self._has_dummy:
                output = output_parallel[..., :self._real_output_cols]
            else:
                output = output_parallel

        # Handle bias for skip_bias_add path
        output_bias: Optional[torch.Tensor] = None
        if self.skip_bias_add and self.bias is not None:
            if self.gather_output and tp_world_size > 1:
                # Gather bias across ranks and strip padding
                bias_gathered = _gather_along_last_dim(self.bias.unsqueeze(0), tp_group)
                output_bias = bias_gathered.squeeze(0)[..., :self.output_size]
            else:
                output_bias = self.bias[:self._real_output_cols] if self._has_dummy else self.bias

        return output, output_bias

    def extra_repr(self) -> str:
        tp = _get_tp_world_size()
        use_bias = self.bias is not None
        padded = self.output_size_per_partition * tp
        return (
            f"in_features={self.input_size}, "
            f"out_features={self.output_size}, "
            f"out_padded={padded}, "
            f"bias={use_bias}, "
            f"TP={tp}, "
            f"dummy_cols_this_rank={self.output_size_per_partition - self._real_output_cols}"
        )


# ---------------------------------------------------------------------------
# RowParallelLinear
# ---------------------------------------------------------------------------

class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism + heterogeneous input-dim padding.

    Splits weight matrix along the input dimension across TP ranks.
    Supports non-divisible input sizes via ceiling-padding with dummy
    columns in the weight matrix (initialised to zero; their contributions
    are never seen because the corresponding padded input positions are
    also zero).

    Y = X A^T + b   where A is [output_size, input_size].
    A is partitioned row-wise (input dimension) so each rank holds
    A_i of shape [output_size, input_size_per_partition], where
    input_size_per_partition = ceil(input_size / tp) // tp (padded).

    The paired ColumnParallelLinear (with gather_output=False) strips
    dummy output columns before producing the input to this layer, so
    this layer never actually receives non-zero values in dummy positions.

    Args:
        input_size:        Full input feature dimension (before TP split).
        output_size:       Output feature dimension (not split).
        config:            ModelParallelConfig.
        bias:              Whether to add a bias term.  Bias is NOT split
                           across TP ranks.
        input_is_parallel: If True, the input has already been scattered
                           across TP ranks (e.g. it comes from a
                           ColumnParallelLinear with gather_output=False)
                           and we skip the scatter step.
        init_method:       Weight initialiser callable.
        skip_bias_add:     If True, return bias as second element instead
                           of adding it in forward.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: ModelParallelConfig,
        bias: bool = True,
        input_is_parallel: bool = False,
        init_method: Optional[Callable] = None,
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

        # Ceiling-padded partition size (same on every rank)
        self.input_size_per_partition = _padded_partition_size(input_size, tp_world_size)

        # Number of real (non-dummy) input columns on this rank
        real_start, real_end = _rank_output_slice(input_size, tp_rank, tp_world_size)
        self._real_input_cols = real_end - real_start
        self._has_dummy = self._real_input_cols < self.input_size_per_partition

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
                # Zero dummy columns in the weight
                if self._has_dummy:
                    with torch.no_grad():
                        self.weight.data[:, self._real_input_cols:].zero_()
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
                # Zero dummy columns in the weight
                if self._has_dummy:
                    with torch.no_grad():
                        self.weight.data[:, self._real_input_cols:].zero_()

        set_tensor_model_parallel_attributes(self.weight, True, 1, 1)

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

        # Register dummy-col gradient hook
        if self._has_dummy and config.perform_initialization:
            self._register_dummy_grad_hook()

    def _register_dummy_grad_hook(self) -> None:
        """Register gradient hook to zero dummy weight columns."""
        real_cols = self._real_input_cols

        def _zero_dummy_grad(grad: torch.Tensor) -> torch.Tensor:
            if grad is not None and grad.shape[1] > real_cols:
                grad = grad.clone()
                grad[:, real_cols:].zero_()
            return grad

        self.weight.register_hook(_zero_dummy_grad)

    def forward(
        self, input_: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            input_: Input tensor.
              * If ``input_is_parallel=True``: already scattered shard of shape
                ``[seq, batch, input_size_per_partition]`` (may include dummy zeros
                for the last rank if input_size is non-divisible).
              * Otherwise: full shape ``[seq, batch, input_size]``.

        Returns:
            Tuple of:
            * output tensor:
              - ``sequence_parallel=False``: full shape ``[seq, batch, output_size]``.
              - ``sequence_parallel=True``:  sharded ``[seq/tp, batch, output_size]``.
            * bias tensor when ``skip_bias_add=True``, else ``None``.
        """
        tp_world_size = _get_tp_world_size()
        tp_group = _get_tp_group()

        if self.input_is_parallel or tp_world_size == 1:
            input_parallel = input_
        else:
            # Scatter input along the last dimension across TP ranks.
            # For non-divisible input_size, pad to the ceiling-padded size first.
            input_parallel = _scatter_along_last_dim_padded(
                input_, tp_group, self.input_size_per_partition
            )

        # Local GEMM: [*, in_per_rank] x [in_per_rank, out]^T → [*, out]
        output_parallel = F.linear(input_parallel, self.weight)

        # Reduce partial results across TP ranks
        if tp_world_size > 1:
            sequence_parallel = getattr(self.config, 'sequence_parallel', False)
            if sequence_parallel:
                # reduce_scatter: sum across TP ranks, scatter along dim 0 (sequence)
                output_parallel = output_parallel.contiguous()
                output_shape = list(output_parallel.shape)
                assert output_shape[0] % tp_world_size == 0, (
                    f"Sequence dim {output_shape[0]} not divisible by "
                    f"tp_world_size {tp_world_size}"
                )
                output_shape[0] = output_shape[0] // tp_world_size
                output = torch.empty(
                    output_shape,
                    dtype=output_parallel.dtype,
                    device=output_parallel.device,
                )
                torch.distributed.reduce_scatter_tensor(
                    output, output_parallel, group=tp_group
                )
            else:
                # Standard path: all_reduce replicates the full result on every TP rank
                torch.distributed.all_reduce(output_parallel, group=tp_group)
                output = output_parallel
        else:
            output = output_parallel

        # Add non-parallelised bias
        if not self.skip_bias_add:
            output = (output + self.bias) if self.bias is not None else output
            output_bias = None
        else:
            output_bias = self.bias

        return output, output_bias

    def extra_repr(self) -> str:
        tp = _get_tp_world_size()
        use_bias = self.bias is not None
        padded = self.input_size_per_partition * tp
        return (
            f"in_features={self.input_size}, "
            f"in_padded={padded}, "
            f"out_features={self.output_size}, "
            f"bias={use_bias}, "
            f"TP={tp}, "
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

    Requires last_dim % world_size == 0 (use _scatter_along_last_dim_padded
    for heterogeneous configs).
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
        # Pad with zeros along last dim
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
