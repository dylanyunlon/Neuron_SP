# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Distributed optimizer with DES-LOC Ku/Kv decomposed moment sync."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from deepspeed.core.model_parallel_config import ModelParallelConfig
from deepspeed.core.desloc_config import DesLocConfig
from deepspeed.core.distributed import ParamAndGradBuffer


# ===========================================================================
# optimizer_config.py
# ===========================================================================

@dataclass
class OptimizerConfig:
    """Optimizer configuration.

    Mirrors Megatron's OptimizerConfig with DES-LOC moment sync extensions.
    """
    lr: Optional[float] = None
    min_lr: Optional[float] = None
    weight_decay: float = 0.01
    params_dtype: torch.dtype = torch.float32
    loss_scale: Optional[float] = None
    initial_loss_scale: float = 2**32
    min_loss_scale: float = 1.0
    loss_scale_window: int = 1000
    hysteresis: int = 2
    fp16: bool = False
    bf16: bool = True
    clip_grad: float = 1.0
    use_distributed_optimizer: bool = True
    overlap_param_gather: bool = False
    overlap_grad_reduce: bool = False

    # Adam hyperparams
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8


# ===========================================================================
# clip_grads.py
# ===========================================================================

def clip_grad_norm(
    parameters: Union[torch.Tensor, List[torch.Tensor]],
    max_norm: float,
    norm_type: float = 2.0,
    model_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
) -> float:
    """Clip gradient norm across model-parallel ranks.

    Args:
        parameters: Model parameters.
        max_norm: Max gradient norm.
        norm_type: Norm type (default L2).
        model_parallel_group: Process group for norm all-reduce.

    Returns:
        Total gradient norm before clipping.
    """
    raise NotImplementedError("Claude task: optimizer/clip_grads")


# ===========================================================================
# optimizer.py — base class
# ===========================================================================

class MegatronOptimizer(ABC):
    """Base optimizer class. All Neuron_SP optimizers inherit from this.

    Provides the interface for gradient preparation, stepping, state
    management, and DES-LOC moment synchronization.
    """

    def __init__(
        self,
        config: OptimizerConfig,
        optimizer: torch.optim.Optimizer,
        params: List[torch.nn.Parameter],
        model_parallel_config: ModelParallelConfig,
    ) -> None:
        raise NotImplementedError("Claude task: optimizer/optimizer")

    @abstractmethod
    def prepare_grads(self) -> bool:
        """Prepare gradients for optimizer step. Returns False if step should be skipped."""
        ...

    @abstractmethod
    def step(self) -> bool:
        """Execute optimizer step. Returns True if step was taken."""
        ...

    @abstractmethod
    def zero_grad(self, set_to_none: bool = True) -> None:
        ...

    def clip_grad_norm(self, clip_grad: float) -> float:
        raise NotImplementedError("Claude task: optimizer/optimizer")

    def get_loss_scale(self) -> torch.Tensor:
        raise NotImplementedError("Claude task: optimizer/optimizer")

    def state_dict(self) -> dict:
        raise NotImplementedError("Claude task: optimizer/optimizer")

    def load_state_dict(self, state_dict: dict) -> None:
        raise NotImplementedError("Claude task: optimizer/optimizer")

    def reload_model_params(self) -> None:
        """Reload model params from optimizer's FP32 master copy."""
        raise NotImplementedError("Claude task: optimizer/optimizer")


class MixedPrecisionOptimizer(MegatronOptimizer):
    """Optimizer with mixed-precision (FP32 master weights, BF16 model weights)."""

    def __init__(
        self,
        config: OptimizerConfig,
        optimizer: torch.optim.Optimizer,
        params: List[torch.nn.Parameter],
        model_parallel_config: ModelParallelConfig,
        grad_scaler: Optional[Any] = None,
    ) -> None:
        raise NotImplementedError("Claude task: optimizer/optimizer")

    def prepare_grads(self) -> bool:
        raise NotImplementedError("Claude task: optimizer/optimizer")

    def step(self) -> bool:
        raise NotImplementedError("Claude task: optimizer/optimizer")

    def zero_grad(self, set_to_none: bool = True) -> None:
        raise NotImplementedError("Claude task: optimizer/optimizer")


# ===========================================================================
# distrib_optimizer.py
# ===========================================================================

class DistributedOptimizer(MixedPrecisionOptimizer):
    """ZeRO-style distributed optimizer with DES-LOC Ku/Kv decomposed sync.

    Each rank holds 1/N of the FP32 master params + optimizer states.
    On optimizer step:
      1. Reduce-scatter gradients (or skip if non-Kx step)
      2. Update local FP32 shard
      3. Write updated shard back to BF16 model
      4. Broadcast so all ranks have consistent model (every step, not just Kx)

    DES-LOC extension: first moments (exp_avg) sync every Ku steps,
    second moments (exp_avg_sq) sync every Kv steps. This reduces
    optimizer state communication by (1 - 1/Ku) + (1 - 1/Kv) volume.
    """

    def __init__(
        self,
        config: OptimizerConfig,
        optimizer: torch.optim.Optimizer,
        params: List[torch.nn.Parameter],
        model_parallel_config: ModelParallelConfig,
        param_and_grad_buffers: List[ParamAndGradBuffer],
        data_parallel_group: torch.distributed.ProcessGroup,
        data_parallel_group_gloo: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        raise NotImplementedError("Claude task: optimizer/distrib_optimizer")

    def step(self) -> bool:
        """Execute optimizer step with DES-LOC moment sync.

        After base Adam step on local shard:
        - Always: write FP32 shard → BF16 model + broadcast (prevent Kx spike)
        - Every Ku steps: all-reduce exp_avg across ranks
        - Every Kv steps: all-reduce exp_avg_sq across ranks
        """
        raise NotImplementedError("Claude task: optimizer/distrib_optimizer")

    def sync_moments(self, sync_first: bool = False, sync_second: bool = False) -> None:
        """Synchronize optimizer moments across DES-LOC tiers.

        Args:
            sync_first: All-reduce first moment (exp_avg) — Ku step.
            sync_second: All-reduce second moment (exp_avg_sq) — Kv step.
        """
        raise NotImplementedError("Claude task: optimizer/distrib_optimizer")

    def shard_to_model_broadcast(self) -> None:
        """Write local FP32 shard to BF16 model and broadcast to all ranks.

        This is called EVERY step (not just Kx steps) to prevent the
        Frankenstein model bug where each rank's model has only 1/N
        params updated.
        """
        raise NotImplementedError("Claude task: optimizer/distrib_optimizer")

    def state_dict(self) -> dict:
        raise NotImplementedError("Claude task: optimizer/distrib_optimizer")

    def load_state_dict(self, state_dict: dict) -> None:
        raise NotImplementedError("Claude task: optimizer/distrib_optimizer")

    def save_parameter_state(self, filename: str) -> None:
        raise NotImplementedError("Claude task: optimizer/distrib_optimizer")
