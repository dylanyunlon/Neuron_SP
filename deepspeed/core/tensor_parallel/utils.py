# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Tensor-parallel utility functions and classes.

Ported from Megatron-LM/megatron/core/tensor_parallel/utils.py with
DES-LOC extensions for heterogeneous tensor parallelism.

Contents
--------
* ``split_tensor_along_last_dim``  — partition a tensor across TP ranks
* ``split_tensor_into_1d_equal_chunks`` — 1-D flat split for activation checkpointing
* ``gather_split_1d_tensor``        — inverse of the above
* ``VocabUtility``                   — vocab-range helpers for sharded embeddings

DES-LOC extension — heterogeneous TP split
-------------------------------------------
When the hidden dimension is not evenly divisible by the TP world size
(e.g. 32 attention heads across 5 GPUs), the standard ``divide(dim, tp)``
would fail.  ``split_tensor_along_last_dim_hetero`` uses ceil-division and
pads the last rank's chunk with zeros so that all-gather / reduce-scatter
shapes are uniform.  The same strategy is used by
``layers.py::_padded_partition_size``.
"""

from __future__ import annotations

from math import ceil
from typing import List, Optional, Sequence, Tuple

import torch

# ---------------------------------------------------------------------------
# Import the all_gather_into_tensor (name changed across PyTorch versions)
# ---------------------------------------------------------------------------
try:
    dist_all_gather_func = torch.distributed.all_gather_into_tensor
except AttributeError:
    try:
        dist_all_gather_func = torch.distributed._all_gather_base
    except AttributeError:
        dist_all_gather_func = None  # will fail at call-site with a clear error


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _divide(numerator: int, denominator: int) -> int:
    """Integer division that asserts exact divisibility."""
    assert denominator != 0, "Division by zero"
    assert numerator % denominator == 0, (
        f"{numerator} is not divisible by {denominator}"
    )
    return numerator // denominator


def _get_tp_group() -> Optional[torch.distributed.ProcessGroup]:
    try:
        from deepspeed.core.parallel_state import get_tensor_model_parallel_group
        return get_tensor_model_parallel_group()
    except (ImportError, AssertionError):
        return None


def _tp_group_or_default(
    tp_group: Optional[torch.distributed.ProcessGroup],
) -> Optional[torch.distributed.ProcessGroup]:
    if tp_group is not None:
        return tp_group
    return _get_tp_group()


# ---------------------------------------------------------------------------
# split_tensor_along_last_dim
# ---------------------------------------------------------------------------

def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """Split *tensor* along its last dimension into *num_partitions* chunks.

    Args:
        tensor: Input tensor.
        num_partitions: Number of chunks.
        contiguous_split_chunks: If ``True``, make each chunk contiguous.

    Returns:
        List of tensor chunks.
    """
    last_dim = tensor.dim() - 1
    last_dim_size = _divide(tensor.size()[last_dim], num_partitions)
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    if contiguous_split_chunks:
        return [chunk.contiguous() for chunk in tensor_list]
    return list(tensor_list)


def split_tensor_along_last_dim_hetero(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """Like :func:`split_tensor_along_last_dim` but tolerates non-divisible sizes.

    The last chunk is zero-padded to match ``ceil(dim / num_partitions)`` so
    that all chunks have the same size (required by NCCL collectives).

    This is the DES-LOC extension for heterogeneous TP where the number of
    attention heads may not be divisible by the TP world size.
    """
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size(last_dim)
    chunk_size = int(ceil(last_dim_size / num_partitions))
    padded_size = chunk_size * num_partitions

    if padded_size != last_dim_size:
        pad_sizes = [0] * (2 * tensor.dim())
        pad_sizes[1] = padded_size - last_dim_size  # pad last dim at the end
        tensor = torch.nn.functional.pad(tensor, pad_sizes)

    tensor_list = torch.split(tensor, chunk_size, dim=last_dim)
    if contiguous_split_chunks:
        return [chunk.contiguous() for chunk in tensor_list]
    return list(tensor_list)


# ---------------------------------------------------------------------------
# 1-D flat tensor split / gather (for activation checkpointing)
# ---------------------------------------------------------------------------

def split_tensor_into_1d_equal_chunks(
    tensor: torch.Tensor,
    new_buffer: bool = False,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Break *tensor* into equal 1-D chunks across TP ranks.

    Returns a contiguous Tensor or a view with this rank's portion.
    """
    tp_group = _tp_group_or_default(tp_group)
    if tp_group is None:
        return tensor
    partition_size = torch.numel(tensor) // tp_group.size()
    start = partition_size * tp_group.rank()
    end = start + partition_size
    if new_buffer:
        data = torch.empty(
            partition_size,
            dtype=tensor.dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )
        data.copy_(tensor.view(-1)[start:end])
    else:
        data = tensor.view(-1)[start:end]
    return data


def gather_split_1d_tensor(
    tensor: torch.Tensor,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Inverse of :func:`split_tensor_into_1d_equal_chunks`.

    All-gathers the 1-D chunks back into a single contiguous tensor.
    """
    tp_group = _tp_group_or_default(tp_group)
    if tp_group is None:
        return tensor
    numel_gathered = torch.numel(tensor) * tp_group.size()
    gathered = torch.empty(
        numel_gathered,
        dtype=tensor.dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
    if dist_all_gather_func is not None:
        dist_all_gather_func(gathered, tensor, group=tp_group)
    else:
        raise RuntimeError(
            "torch.distributed.all_gather_into_tensor is unavailable — "
            "upgrade PyTorch to ≥ 1.13"
        )
    return gathered


# ---------------------------------------------------------------------------
# VocabUtility — vocabulary partitioning for VocabParallelEmbedding
# ---------------------------------------------------------------------------

class VocabUtility:
    """Compute vocabulary shard ranges for tensor-parallel embedding layers.

    Indices follow the convention ``[first, last)`` (last is exclusive).
    """

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size: int,
        rank: int,
        world_size: int,
    ) -> Sequence[int]:
        """Return ``(start, end)`` for the given rank."""
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(
        global_vocab_size: int,
        rank: int,
        world_size: int,
    ) -> Sequence[int]:
        """Return ``(start, end)`` computed from the global vocabulary size."""
        per_partition = _divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(
            per_partition, rank, world_size
        )

    @staticmethod
    def vocab_range_from_global_vocab_size_hetero(
        global_vocab_size: int,
        rank: int,
        world_size: int,
    ) -> Tuple[int, int]:
        """DES-LOC variant that uses ceil-division for non-divisible vocab sizes.

        The last rank owns fewer tokens (no padding needed at the vocab level
        because the embedding weight is padded in ``VocabParallelEmbedding``).
        """
        per_partition = int(ceil(global_vocab_size / world_size))
        start = rank * per_partition
        end = min(start + per_partition, global_vocab_size)
        return start, end
