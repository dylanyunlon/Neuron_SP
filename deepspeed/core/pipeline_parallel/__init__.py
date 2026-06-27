# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Pipeline parallelism with heterogeneous stage scheduling."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from deepspeed.core.model_parallel_config import ModelParallelConfig


# ===========================================================================
# p2p_communication.py
# ===========================================================================

class P2PCommunicator:
    """Point-to-point communication for pipeline parallelism.

    Handles send/recv of activations and gradients between adjacent PP stages.
    DES-LOC extension: PCIe-aware message sizing — cross-NUMA transfers
    use smaller chunks to avoid congestion.
    """

    def __init__(
        self,
        config: ModelParallelConfig,
        pg: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        raise NotImplementedError("Claude task: pipeline_parallel/p2p_communication")

    def send_forward(
        self, tensor: torch.Tensor, override_comm_dtype: Optional[torch.dtype] = None,
    ) -> None:
        raise NotImplementedError("Claude task: pipeline_parallel/p2p_communication")

    def recv_forward(
        self, tensor_shape: torch.Size, dtype: torch.dtype,
    ) -> torch.Tensor:
        raise NotImplementedError("Claude task: pipeline_parallel/p2p_communication")

    def send_backward(
        self, tensor: torch.Tensor, override_comm_dtype: Optional[torch.dtype] = None,
    ) -> None:
        raise NotImplementedError("Claude task: pipeline_parallel/p2p_communication")

    def recv_backward(
        self, tensor_shape: torch.Size, dtype: torch.dtype,
    ) -> torch.Tensor:
        raise NotImplementedError("Claude task: pipeline_parallel/p2p_communication")

    def send_forward_recv_backward(
        self,
        send_tensor: torch.Tensor,
        recv_shape: torch.Size,
        recv_dtype: torch.dtype,
    ) -> torch.Tensor:
        raise NotImplementedError("Claude task: pipeline_parallel/p2p_communication")

    def send_backward_recv_forward(
        self,
        send_tensor: torch.Tensor,
        recv_shape: torch.Size,
        recv_dtype: torch.dtype,
    ) -> torch.Tensor:
        raise NotImplementedError("Claude task: pipeline_parallel/p2p_communication")


# ===========================================================================
# schedules.py
# ===========================================================================

def get_forward_backward_func(
    virtual_pipeline_model_parallel_size: Optional[int],
    pipeline_model_parallel_size: int,
    *,
    forward_only: bool = False,
) -> Callable:
    """Return the appropriate pipeline schedule function.

    Returns:
        forward_backward_no_pipelining: PP=1
        forward_backward_pipelining_with_interleaving: VPP > 1
        forward_backward_pipelining_without_interleaving: standard 1F1B
    """
    raise NotImplementedError("Claude task: pipeline_parallel/schedules")


def forward_step(
    forward_step_func: Callable,
    data_iterator: object,
    model: nn.Module,
    num_microbatches: int,
    input_tensor: Optional[torch.Tensor],
    forward_data_store: list,
    config: ModelParallelConfig,
    collect_non_loss_data: bool = False,
    is_first_microbatch: bool = False,
) -> torch.Tensor:
    """Execute one forward microbatch."""
    raise NotImplementedError("Claude task: pipeline_parallel/schedules")


def backward_step(
    input_tensor: Optional[torch.Tensor],
    output_tensor: torch.Tensor,
    output_tensor_grad: Optional[torch.Tensor],
    config: ModelParallelConfig,
) -> Optional[torch.Tensor]:
    """Execute one backward microbatch."""
    raise NotImplementedError("Claude task: pipeline_parallel/schedules")


def forward_backward_no_pipelining(
    forward_step_func: Callable,
    data_iterator: object,
    model: Union[nn.Module, List[nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    *,
    config: ModelParallelConfig,
    p2p_communicator: Optional[P2PCommunicator] = None,
) -> dict:
    """No pipeline parallelism — simple fwd+bwd loop with gradient accumulation.

    DES-LOC extension: num_microbatches can differ per rank (heterogeneous
    micro-batch sizes). Each rank accumulates its own count.
    """
    raise NotImplementedError("Claude task: pipeline_parallel/schedules")


def forward_backward_pipelining_without_interleaving(
    forward_step_func: Callable,
    data_iterator: object,
    model: Union[nn.Module, List[nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    *,
    config: ModelParallelConfig,
    p2p_communicator: Optional[P2PCommunicator] = None,
) -> dict:
    """Standard 1F1B pipeline schedule.

    DES-LOC extension: supports unequal layer counts per stage
    (pipeline_layer_split in config). Stages with fewer layers finish
    faster — schedule handles the asymmetry in warmup/cooldown phases.
    """
    raise NotImplementedError("Claude task: pipeline_parallel/schedules")


def forward_backward_pipelining_with_interleaving(
    forward_step_func: Callable,
    data_iterator: object,
    model: List[nn.Module],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    *,
    config: ModelParallelConfig,
    p2p_communicator: Optional[P2PCommunicator] = None,
) -> dict:
    """Interleaved 1F1B for virtual pipeline parallelism."""
    raise NotImplementedError("Claude task: pipeline_parallel/schedules")


# ===========================================================================
# utils.py
# ===========================================================================

def get_num_microbatches() -> int:
    raise NotImplementedError("Claude task: pipeline_parallel/utils")


def get_pipeline_model_parallel_rank_for_layer(layer_number: int) -> int:
    """Given a global layer number, return which PP rank owns it.

    Uses pipeline_layer_split from config for heterogeneous splits.
    """
    raise NotImplementedError("Claude task: pipeline_parallel/utils")
