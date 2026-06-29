# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Collective communication mappings for tensor parallelism.

Provides the scatter/gather/reduce helpers that move tensors between
TP ranks.  Referenced by both random.py (for distribute_saved_activations)
and layers.py (for ColumnParallelLinear / RowParallelLinear).
"""

from __future__ import annotations

from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_tp_group() -> Optional[torch.distributed.ProcessGroup]:
    try:
        from deepspeed.core.parallel_state import get_tensor_model_parallel_group
        return get_tensor_model_parallel_group()
    except (ImportError, AssertionError):
        return None


def _tp_world_size() -> int:
    g = _get_tp_group()
    return 1 if g is None else torch.distributed.get_world_size(group=g)


def _tp_rank() -> int:
    g = _get_tp_group()
    return 0 if g is None else torch.distributed.get_rank(group=g)


# ---------------------------------------------------------------------------
# 1-D tensor split / gather helpers (for distribute_saved_activations)
# ---------------------------------------------------------------------------

def split_tensor_into_1d_equal_chunks(
    tensor: torch.Tensor,
    new_buffer: bool = False,
) -> torch.Tensor:
    """Return this rank's contiguous 1-D chunk of *tensor*.

    The tensor is first flattened to 1-D, then split evenly across TP ranks.
    Padding zeros are appended if ``numel % tp_world_size != 0``.

    Args:
        tensor:     Input tensor (any shape).
        new_buffer: If True, allocate a new buffer instead of slicing.

    Returns:
        1-D tensor of size ``ceil(numel / tp_world_size)``.
    """
    world_size = _tp_world_size()
    rank = _tp_rank()
    flat = tensor.view(-1)
    numel = flat.numel()

    per_rank = (numel + world_size - 1) // world_size
    padded = per_rank * world_size

    if padded > numel:
        flat = torch.nn.functional.pad(flat, (0, padded - numel))

    start = rank * per_rank
    chunk = flat[start:start + per_rank]
    if new_buffer:
        chunk = chunk.clone()
    return chunk.contiguous()


def gather_split_1d_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather the 1-D chunks from all TP ranks into one tensor.

    Inverse of ``split_tensor_into_1d_equal_chunks``.

    Args:
        tensor: 1-D chunk owned by this rank.

    Returns:
        Concatenated 1-D tensor of size ``per_rank * tp_world_size``.
    """
    group = _get_tp_group()
    world_size = _tp_world_size()
    if world_size == 1:
        return tensor

    tensor = tensor.contiguous()
    chunks = [torch.empty_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(chunks, tensor, group=group)
    return torch.cat(chunks, dim=0)


# ---------------------------------------------------------------------------
# Tensor-parallel region functions (mirrors Megatron's mappings.py API)
# ---------------------------------------------------------------------------

def copy_to_tensor_model_parallel_region(
    input_: torch.Tensor,
    group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Identity in forward; all-reduce in backward (for column-parallel input).

    M3981: ``group`` overrides the global TP group so callers that carry an
    explicit ``pg_collection.tp`` reference no longer rely on parallel_state.
    """
    return _CopyToModelParallelRegion.apply(input_, group)


def reduce_from_tensor_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce in forward; identity in backward."""
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_tensor_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    """Scatter along last dim in forward; all-gather in backward."""
    return _ScatterToModelParallelRegion.apply(input_)


def gather_from_tensor_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    """All-gather along last dim in forward; scatter in backward."""
    return _GatherFromModelParallelRegion.apply(input_)


def scatter_to_sequence_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    """Scatter along sequence (dim 0) in forward; all-gather in backward."""
    return _ScatterToSequenceParallelRegion.apply(input_)


def gather_from_sequence_parallel_region(
    input_: torch.Tensor,
    tensor_parallel_output_grad: bool = True,
    group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """All-gather along sequence (dim 0) in forward.

    M3981: ``group`` overrides the global TP group so SharedExpertMLP can
    pass ``self.tp_group`` rather than relying on parallel_state.
    """
    return _GatherFromSequenceParallelRegion.apply(input_, tensor_parallel_output_grad, group)


def reduce_scatter_to_sequence_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    """Reduce-scatter along sequence dim in forward; all-gather in backward."""
    return _ReduceScatterToSequenceParallelRegion.apply(input_)


def all_gather_last_dim_from_tensor_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    """All-gather along last dim (no backward scatter — use for inference)."""
    group = _get_tp_group()
    world_size = _tp_world_size()
    if world_size == 1:
        return input_
    input_ = input_.contiguous()
    out_shape = list(input_.shape)
    out_shape[-1] *= world_size
    output = torch.empty(out_shape, dtype=input_.dtype, device=input_.device)
    torch.distributed.all_gather_into_tensor(output, input_, group=group)
    return output


def reduce_scatter_last_dim_to_tensor_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    """Reduce-scatter along last dim."""
    group = _get_tp_group()
    world_size = _tp_world_size()
    if world_size == 1:
        return input_
    input_ = input_.contiguous()
    out_shape = list(input_.shape)
    assert out_shape[-1] % world_size == 0
    out_shape[-1] //= world_size
    output = torch.empty(out_shape, dtype=input_.dtype, device=input_.device)
    torch.distributed.reduce_scatter_tensor(output, input_, group=group)
    return output


def all_to_all(
    input_: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
) -> torch.Tensor:
    """All-to-all: scatter along *scatter_dim*, gather along *gather_dim*."""
    group = _get_tp_group()
    world_size = _tp_world_size()
    if world_size == 1:
        return input_

    input_ = input_.contiguous()
    # Split along scatter_dim
    chunks = list(torch.chunk(input_, world_size, dim=scatter_dim))
    out_chunks = [torch.empty_like(c) for c in chunks]
    torch.distributed.all_to_all(out_chunks, chunks, group=group)
    return torch.cat(out_chunks, dim=gather_dim)


def all_to_all_hp2sp(input_: torch.Tensor) -> torch.Tensor:
    """All-to-all head-parallel → sequence-parallel (scatter heads, gather seq)."""
    return all_to_all(input_, scatter_dim=1, gather_dim=0)


def all_to_all_sp2hp(input_: torch.Tensor) -> torch.Tensor:
    """All-to-all sequence-parallel → head-parallel (scatter seq, gather heads)."""
    return all_to_all(input_, scatter_dim=0, gather_dim=1)


# ---------------------------------------------------------------------------
# Autograd function implementations
# ---------------------------------------------------------------------------

class _CopyToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, group=None):
        # M3981: store explicit group (None → fall back to _get_tp_group() in bwd)
        ctx.group = group
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group if ctx.group is not None else _get_tp_group()
        if group is not None and torch.distributed.get_world_size(group=group) > 1:
            torch.distributed.all_reduce(grad_output, group=group)
        return grad_output, None


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        group = _get_tp_group()
        if group is not None and _tp_world_size() > 1:
            torch.distributed.all_reduce(input_, group=group)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        world_size = _tp_world_size()
        if world_size == 1:
            return input_
        rank = _tp_rank()
        last_dim = input_.shape[-1]
        assert last_dim % world_size == 0
        per_rank = last_dim // world_size
        return input_[..., rank * per_rank:(rank + 1) * per_rank].contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        group = _get_tp_group()
        world_size = _tp_world_size()
        if world_size == 1:
            return grad_output
        grad_output = grad_output.contiguous()
        out_shape = list(grad_output.shape)
        out_shape[-1] *= world_size
        full_grad = torch.empty(out_shape, dtype=grad_output.dtype, device=grad_output.device)
        torch.distributed.all_gather_into_tensor(full_grad, grad_output, group=group)
        return full_grad


class _GatherFromModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        group = _get_tp_group()
        world_size = _tp_world_size()
        if world_size == 1:
            return input_
        input_ = input_.contiguous()
        out_shape = list(input_.shape)
        out_shape[-1] *= world_size
        output = torch.empty(out_shape, dtype=input_.dtype, device=input_.device)
        torch.distributed.all_gather_into_tensor(output, input_, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        world_size = _tp_world_size()
        if world_size == 1:
            return grad_output
        rank = _tp_rank()
        last_dim = grad_output.shape[-1]
        per_rank = last_dim // world_size
        return grad_output[..., rank * per_rank:(rank + 1) * per_rank].contiguous()


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        world_size = _tp_world_size()
        if world_size == 1:
            return input_
        rank = _tp_rank()
        seq_len = input_.shape[0]
        assert seq_len % world_size == 0
        per_rank = seq_len // world_size
        return input_[rank * per_rank:(rank + 1) * per_rank].contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        group = _get_tp_group()
        world_size = _tp_world_size()
        if world_size == 1:
            return grad_output
        grad_output = grad_output.contiguous()
        chunks = [torch.empty_like(grad_output) for _ in range(world_size)]
        torch.distributed.all_gather(chunks, grad_output, group=group)
        return torch.cat(chunks, dim=0)


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, tensor_parallel_output_grad, group=None):
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        # M3981: explicit group overrides parallel_state singleton
        ctx.group = group
        _group = group if group is not None else _get_tp_group()
        world_size = 1 if _group is None else torch.distributed.get_world_size(group=_group)
        if world_size == 1:
            return input_
        input_ = input_.contiguous()
        chunks = [torch.empty_like(input_) for _ in range(world_size)]
        torch.distributed.all_gather(chunks, input_, group=_group)
        return torch.cat(chunks, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        _group = ctx.group if ctx.group is not None else _get_tp_group()
        world_size = 1 if _group is None else torch.distributed.get_world_size(group=_group)
        rank = 0 if _group is None else torch.distributed.get_rank(group=_group)
        if world_size == 1:
            return grad_output, None, None
        seq_len = grad_output.shape[0]
        per_rank = seq_len // world_size
        if ctx.tensor_parallel_output_grad:
            local_grad = grad_output[rank * per_rank:(rank + 1) * per_rank].contiguous()
            torch.distributed.all_reduce(local_grad, group=_group)
            return local_grad, None, None
        return grad_output[rank * per_rank:(rank + 1) * per_rank].contiguous(), None, None


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        group = _get_tp_group()
        world_size = _tp_world_size()
        if world_size == 1:
            return input_
        input_ = input_.contiguous()
        seq_len = input_.shape[0]
        assert seq_len % world_size == 0
        per_rank = seq_len // world_size
        out_shape = list(input_.shape)
        out_shape[0] = per_rank
        output = torch.empty(out_shape, dtype=input_.dtype, device=input_.device)
        torch.distributed.reduce_scatter_tensor(output, input_, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        group = _get_tp_group()
        world_size = _tp_world_size()
        if world_size == 1:
            return grad_output
        grad_output = grad_output.contiguous()
        chunks = [torch.empty_like(grad_output) for _ in range(world_size)]
        torch.distributed.all_gather(chunks, grad_output, group=group)
        return torch.cat(chunks, dim=0)
