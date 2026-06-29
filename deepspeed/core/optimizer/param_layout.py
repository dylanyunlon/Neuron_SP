# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Parameter layout dataclasses for optimizer-driven buffer layout.

Ported from Megatron-LM M3811 (55b8111ad / c6a886ed).

These dataclasses describe how parameters are laid out in contiguous buffers.
Each distributed optimizer implementation (e.g., DistributedOptimizer) is
responsible for computing these layouts via _compute_per_buffer_param_layout,
applying its own padding, alignment, and bucket splitting rules.  DDP and
buffers consume the resulting layouts without any optimizer-specific knowledge.

Alignment rules (matching Megatron):
  - Each param start is rounded up to a 64-element boundary so that every
    parameter begins on a cache-line-friendly boundary inside the flat buffer.
  - Each bucket end is rounded up to lcm(dp_world_size, 128) elements so that
    (a) reduce-scatter slices divide evenly across DP ranks, and
    (b) the bucket is at minimum 128-element aligned for NCCL efficiency.
  - When ``pad_for_high_nccl_busbw`` is True (opt-in flag in DDP config),
    the divisor is further extended to lcm(dp_world_size, 128, 2^16) so that
    large-message NCCL collectives achieve peak bus bandwidth.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch


# ---------------------------------------------------------------------------
# Padding helpers
# ---------------------------------------------------------------------------

def pad_to_divisor(value: int, divisor: int) -> int:
    """Round up ``value`` to the nearest multiple of ``divisor``."""
    return int(math.ceil(value / divisor) * divisor)


def pad_param_start(param_start_index: int) -> int:
    """Align parameter start index to a 64-element boundary.

    This ensures every parameter in the flat grad buffer starts on a
    cache-line-friendly address (64 × 2 bytes = 128 bytes for BF16).
    """
    return pad_to_divisor(param_start_index, 64)


def bucket_end_divisor(data_parallel_world_size: int, pad_for_high_nccl_busbw: bool) -> int:
    """Return the divisor used to pad bucket ends.

    Args:
        data_parallel_world_size: DP world size (for even reduce-scatter slicing).
        pad_for_high_nccl_busbw: When True, extend divisor to 2^16 for peak
            NCCL bus-bandwidth on large-message collectives.

    Returns:
        lcm(dp_world_size, 128) or lcm(dp_world_size, 128, 2^16).
    """
    if pad_for_high_nccl_busbw:
        return math.lcm(data_parallel_world_size, 128, 2 ** 16)
    return math.lcm(data_parallel_world_size, 128)


def pad_bucket_end(
    bucket_end_index: int,
    data_parallel_world_size: int,
    pad_for_high_nccl_busbw: bool,
) -> int:
    """Pad bucket end for DP-divisibility and NCCL alignment.

    Args:
        bucket_end_index: Unpadded end index of the bucket.
        data_parallel_world_size: DP world size.
        pad_for_high_nccl_busbw: Enables the 2^16 extension.

    Returns:
        Padded end index.
    """
    return pad_to_divisor(
        bucket_end_index,
        bucket_end_divisor(data_parallel_world_size, pad_for_high_nccl_busbw),
    )


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BufferKey:
    """Identifies a distinct parameter buffer.

    Each unique combination of these fields corresponds to a separate
    contiguous buffer in DDP.  Parameters are grouped into buffers by
    these dimensions.

    Attributes:
        param_dtype: Storage dtype (torch.uint8 for FP8/NVFP4 parameters,
            else param.dtype).
        grad_dtype: Gradient reduction dtype.
        is_expert_parallel: Whether the buffer holds expert-parallel
            parameters, which use a separate data-parallel group.
        is_managed_by_layer_wise_optimizer: Whether parameters in this
            buffer are managed by LayerWiseDistributedOptimizer (shard-
            aligned layout so each whole param lives in one shard).
            Non-LayerWise params get DistributedOptimizer's byte-level
            layout in a separate buffer.  Added in M3998.
    """

    param_dtype: torch.dtype
    grad_dtype: torch.dtype
    is_expert_parallel: bool
    is_managed_by_layer_wise_optimizer: bool = False


@dataclass
class PerBufferParamLayout:
    """Layout for parameters within a single contiguous buffer.

    Describes how parameters are laid out in the contiguous flat buffer.

    Attributes:
        param_index_map: Mapping from parameter to
            (start_index, end_index, bucket_id) in the buffer.
        bucket_indices: List of (start_index, end_index) for each bucket
            (end_index is padded for DP divisibility).
        per_bucket_numel_unpadded: Number of unpadded elements per bucket.
        param_indices: The index of each param among same-dtype params
            (using the "fake" high-precision dtype for FP8/NVFP4 params).
            Needed for loading non-native-fp8 checkpoints in native-fp8
            mode.  Order matches param_index_map iteration order.
    """

    param_index_map: Dict[torch.nn.Parameter, Tuple[int, int, int]] = field(
        default_factory=dict
    )
    bucket_indices: List[Tuple[int, int]] = field(default_factory=list)
    per_bucket_numel_unpadded: List[int] = field(default_factory=list)
    param_indices: List[int] = field(default_factory=list)


@dataclass
class FullParamLayout:
    """Layout for all parameters across all buffer groups in a model chunk.

    Maps BufferKey to per-buffer PerBufferParamLayout objects.  Each
    PerBufferParamLayout has its own independent index space since different
    buffer groups are physically separate buffers.

    Attributes:
        layouts: Mapping from BufferKey to PerBufferParamLayout.
    """

    layouts: Dict[BufferKey, PerBufferParamLayout] = field(default_factory=dict)
