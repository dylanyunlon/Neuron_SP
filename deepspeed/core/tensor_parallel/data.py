# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Tensor-parallel data broadcasting utilities.

Ported from Megatron-LM/megatron/core/tensor_parallel/data.py.

Provides :func:`broadcast_data` which broadcasts a dictionary of CPU tensors
from rank 0 of a tensor-model-parallel group to all other members.  This is
used during data loading to ensure every TP rank sees the same input tokens
without requiring each rank to read from the dataset independently.

DES-LOC note
------------
In heterogeneous clusters, TP groups may span GPUs with different PCIe
bandwidths.  The broadcast itself uses NCCL and is topology-aware, so no
special handling is needed here.  However, the caller should be aware that
rank 0 of the TP group may be on a different GPU tier than the other ranks,
which affects the CPU → GPU transfer speed for the initial ``flatten_data``
copy.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

_MAX_DATA_DIM = 5


# ---------------------------------------------------------------------------
# TP group helpers
# ---------------------------------------------------------------------------

def _get_tp_group() -> Optional[torch.distributed.ProcessGroup]:
    try:
        from deepspeed.core.parallel_state import get_tensor_model_parallel_group
        return get_tensor_model_parallel_group()
    except (ImportError, AssertionError):
        return None


def _resolve_tp_group(
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Optional[torch.distributed.ProcessGroup]:
    if tp_group is not None:
        return tp_group
    return _get_tp_group()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_data_types(
    keys: List[str],
    data: Dict[str, torch.Tensor],
    target_dtype: torch.dtype,
) -> None:
    """Assert that every key in *data* has *target_dtype*."""
    for key in keys:
        assert data[key].dtype == target_dtype, (
            f"{key} has dtype {data[key].dtype} which is different from {target_dtype}"
        )


def _build_key_size_numel_dictionaries(
    keys: List[str],
    data: Dict[str, torch.Tensor],
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
):
    """Broadcast tensor shapes from rank 0 and compute per-key sizes/numel.

    Returns:
        ``(key_size, key_numel, total_numel)``
    """
    tp_group = _resolve_tp_group(tp_group)
    max_dim = _MAX_DATA_DIM
    sizes = [0 for _ in range(max_dim) for _ in keys]

    # Pack sizes on rank 0
    if tp_group is not None and tp_group.rank() == 0:
        offset = 0
        for key in keys:
            assert data[key].dim() < max_dim, (
                f"Tensor for key '{key}' has {data[key].dim()} dims — "
                f"increase _MAX_DATA_DIM (currently {max_dim})"
            )
            for i, s in enumerate(data[key].size()):
                sizes[i + offset] = s
            offset += max_dim
    elif tp_group is None:
        # No TP — single-rank path
        offset = 0
        for key in keys:
            for i, s in enumerate(data[key].size()):
                sizes[i + offset] = s
            offset += max_dim

    sizes_cuda = torch.tensor(sizes, dtype=torch.long, device="cuda")

    if tp_group is not None:
        group_ranks = torch.distributed.get_process_group_ranks(group=tp_group)
        torch.distributed.broadcast(sizes_cuda, group_ranks[0], group=tp_group)

    sizes_cpu = sizes_cuda.cpu()
    key_size: Dict[str, List[int]] = {}
    key_numel: Dict[str, int] = {}
    total_numel = 0
    offset = 0
    for key in keys:
        i = 0
        size: List[int] = []
        numel = 1
        while sizes_cpu[offset + i] > 0:
            this_size = int(sizes_cpu[offset + i])
            size.append(this_size)
            numel *= this_size
            i += 1
        key_size[key] = size
        key_numel[key] = numel
        total_numel += numel
        offset += max_dim

    return key_size, key_numel, total_numel


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def broadcast_data(
    keys: List[str],
    data: Dict[str, torch.Tensor],
    datatype: torch.dtype,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Dict[str, torch.Tensor]:
    """Broadcast *data* from TP rank 0 to all TP members.

    Args:
        keys: Which keys in *data* to broadcast.
        data: ``{key: cpu_tensor}`` dictionary (only rank 0 needs valid values).
        datatype: Expected dtype of all tensors.
        tp_group: Tensor-parallel process group (uses default if ``None``).

    Returns:
        ``{key: cuda_tensor}`` — each rank gets the same data on its GPU.
    """
    key_size, key_numel, total_numel = _build_key_size_numel_dictionaries(
        keys, data, tp_group
    )
    tp_group = _resolve_tp_group(tp_group)

    if tp_group is not None and tp_group.rank() == 0:
        _check_data_types(keys, data, datatype)
        flatten_data = torch.cat(
            [data[key].cuda().contiguous().view(-1) for key in keys], dim=0
        )
    elif tp_group is None:
        # Single-rank: just move to GPU
        _check_data_types(keys, data, datatype)
        flatten_data = torch.cat(
            [data[key].cuda().contiguous().view(-1) for key in keys], dim=0
        )
    else:
        flatten_data = torch.empty(
            total_numel, device=torch.cuda.current_device(), dtype=datatype
        )

    if tp_group is not None:
        group_ranks = torch.distributed.get_process_group_ranks(group=tp_group)
        torch.distributed.broadcast(flatten_data, group_ranks[0], group=tp_group)

    # Unpack
    output: Dict[str, torch.Tensor] = {}
    offset = 0
    for key in keys:
        size = key_size[key]
        numel = key_numel[key]
        output[key] = flatten_data.narrow(0, offset, numel).view(size)
        offset += numel

    return output
