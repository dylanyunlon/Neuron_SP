# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Distributed data parallelism with DES-LOC support."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Union
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from deepspeed.core.model_parallel_config import ModelParallelConfig
from deepspeed.core.desloc_config import DesLocConfig


# ---- deepspeed/core/distributed/__init__.py ----
# Re-exports all public names from submodules.


# ===========================================================================
# distributed_data_parallel_config.py
# ===========================================================================

@dataclass
class DistributedDataParallelConfig:
    """Configuration for DDP wrapper.

    Mirrors Megatron's DistributedDataParallelConfig with DES-LOC extensions.
    """

    grad_reduce_in_fp32: bool = False
    overlap_grad_reduce: bool = False
    overlap_param_gather: bool = False
    align_param_gather: bool = False
    use_distributed_optimizer: bool = False
    num_distributed_optimizer_instances: int = 1
    check_for_nan_in_grad: bool = False
    bucket_size: Optional[int] = None

    # DES-LOC: allow skipping grad sync on non-Kx steps
    allow_skip_grad_sync: bool = False


# ===========================================================================
# param_and_grad_buffer.py
# ===========================================================================

class BufferType(Enum):
    PARAM = auto()
    GRAD = auto()


class ParamAndGradBucket:
    """A contiguous buffer holding a subset of model params and their grads."""

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        param_data: torch.Tensor,
        grad_data: torch.Tensor,
        offset: int,
        numel_unpadded: int,
        gradient_scaling_factor: float,
        communication_dtype: torch.dtype,
    ) -> None:
        raise NotImplementedError("Claude task: distributed/param_and_grad_buffer")


class ParamAndGradBucketGroup:
    """Manages multiple buckets and orchestrates grad sync / param gather.

    This is the core communication abstraction. DES-LOC hooks into
    `start_grad_sync` and `finish_grad_sync` to gate on Kx steps.
    """

    def __init__(
        self,
        buckets: List[ParamAndGradBucket],
        ddp_config: DistributedDataParallelConfig,
        data_parallel_group: torch.distributed.ProcessGroup,
        data_parallel_world_size: int,
    ) -> None:
        raise NotImplementedError("Claude task: distributed/param_and_grad_buffer")

    def reset(self) -> None:
        raise NotImplementedError("Claude task: distributed/param_and_grad_buffer")

    def start_param_sync(self, force_sync: bool = False) -> None:
        raise NotImplementedError("Claude task: distributed/param_and_grad_buffer")

    def finish_param_sync(self) -> None:
        raise NotImplementedError("Claude task: distributed/param_and_grad_buffer")

    def start_grad_sync(self, skip_sync: bool = False) -> None:
        """Start gradient synchronization.

        Args:
            skip_sync: If True (non-Kx step in DES-LOC), skip the all-reduce
                       but still accumulate locally.
        """
        raise NotImplementedError("Claude task: distributed/param_and_grad_buffer")

    def finish_grad_sync(self, force_all_reduce: bool = False) -> None:
        raise NotImplementedError("Claude task: distributed/param_and_grad_buffer")

    def register_grad_ready(self, param: torch.nn.Parameter) -> None:
        raise NotImplementedError("Claude task: distributed/param_and_grad_buffer")

    def scale_gradients(self, scaling_factor: float) -> None:
        raise NotImplementedError("Claude task: distributed/param_and_grad_buffer")


class ParamAndGradBuffer:
    """Top-level buffer manager. One per model chunk, holds all bucket groups."""

    def __init__(
        self,
        config: ModelParallelConfig,
        ddp_config: DistributedDataParallelConfig,
        param_to_name: Dict[torch.nn.Parameter, str],
        params: List[torch.nn.Parameter],
        data_parallel_group: torch.distributed.ProcessGroup,
        bucket_size: int,
        param_dtype: torch.dtype,
        grad_dtype: torch.dtype,
    ) -> None:
        raise NotImplementedError("Claude task: distributed/param_and_grad_buffer")

    def scale_gradients(self, scaling_factor: float) -> None:
        raise NotImplementedError("Claude task: distributed/param_and_grad_buffer")

    def reset(self) -> None:
        raise NotImplementedError("Claude task: distributed/param_and_grad_buffer")

    def zero_grad_buffer(self) -> None:
        raise NotImplementedError("Claude task: distributed/param_and_grad_buffer")


# ===========================================================================
# distributed_data_parallel.py
# ===========================================================================

class DistributedDataParallel(nn.Module):
    """DDP wrapper with DES-LOC Kx-gated gradient synchronization.

    Unlike torch.nn.parallel.DistributedDataParallel, this implementation:
    - Supports skipping gradient all-reduce on non-Kx steps
    - Supports per-shard param broadcast after optimizer step
    - Manages param_and_grad_buffers directly
    - Integrates with Megatron-style pipeline parallelism
    """

    def __init__(
        self,
        config: ModelParallelConfig,
        ddp_config: DistributedDataParallelConfig,
        module: nn.Module,
        data_parallel_group: torch.distributed.ProcessGroup,
        expert_data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        disable_bucketing: bool = False,
    ) -> None:
        raise NotImplementedError("Claude task: distributed/distributed_data_parallel")

    @property
    def module(self) -> nn.Module:
        raise NotImplementedError("Claude task: distributed/distributed_data_parallel")

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Claude task: distributed/distributed_data_parallel")

    def start_grad_sync(self, skip_sync: bool = False) -> None:
        """Start gradient synchronization.

        Args:
            skip_sync: On non-Kx DES-LOC steps, skip the collective but
                       keep local gradient accumulation.
        """
        raise NotImplementedError("Claude task: distributed/distributed_data_parallel")

    def finish_grad_sync(self, force_all_reduce: bool = False) -> None:
        raise NotImplementedError("Claude task: distributed/distributed_data_parallel")

    def start_param_sync(self, force_sync: bool = False) -> None:
        raise NotImplementedError("Claude task: distributed/distributed_data_parallel")

    def broadcast_params(self) -> None:
        """Broadcast all parameters from rank 0 of each shard to all ranks.

        Called every step in DES-LOC to prevent the Kx spike bug
        (ZeRO-3 shard inconsistency).
        """
        raise NotImplementedError("Claude task: distributed/distributed_data_parallel")

    def zero_grad_buffer(self) -> None:
        raise NotImplementedError("Claude task: distributed/distributed_data_parallel")

    def scale_gradients(self, scaling_factor: float) -> None:
        raise NotImplementedError("Claude task: distributed/distributed_data_parallel")

    def no_sync(self):
        """Context manager to disable gradient sync (used in gradient accumulation)."""
        raise NotImplementedError("Claude task: distributed/distributed_data_parallel")


# ===========================================================================
# finalize_model_grads.py
# ===========================================================================

def finalize_model_grads(
    model: List[nn.Module],
    config: ModelParallelConfig,
    num_tokens: Optional[torch.Tensor] = None,
    skip_grad_sync: bool = False,
) -> None:
    """Finalize gradients before optimizer step.

    Handles:
    - All-reduce of embedding grads across PP stages
    - All-reduce of expert router grads across DP
    - Scaling by number of tokens (per-token loss)
    - DES-LOC: skip grad all-reduce on non-Kx steps when skip_grad_sync=True

    Args:
        model: List of model chunks (for virtual pipeline parallelism).
        config: Model parallel config.
        num_tokens: Token count for per-token loss scaling.
        skip_grad_sync: Skip the main gradient all-reduce (DES-LOC non-Kx step).
    """
    raise NotImplementedError("Claude task: distributed/finalize_model_grads")
