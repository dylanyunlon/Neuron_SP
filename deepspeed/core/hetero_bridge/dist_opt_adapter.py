# SPDX-License-Identifier: Apache-2.0
"""dist_opt_adapter.py — adapt core.optimizer.DistributedOptimizer to the hetero ShardPlan.

Per HEAD commit a52efee1: A6000 ranks -> DeepSpeedCPUAdam (CPU-resident state);
H100/Blackwell ranks -> fused AdamW on GPU. Phase 1 skeleton, bodies raise.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from .shard_planner import ShardPlan
    from .tier_map import TierMap


class DistOptAdapter:
    def __init__(self, model: "nn.Module", shard_plan: "ShardPlan",
                 tier_map: "TierMap", lr: float,
                 betas: "tuple[float, float]", weight_decay: float) -> None:
        self.model = model
        self.shard_plan = shard_plan
        self.tier_map = tier_map
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self._opt: "torch.optim.Optimizer | None" = None

    def build(self) -> "torch.optim.Optimizer":
        """Construct DistributedOptimizer with per-rank optimizer type (CPUAdam vs fused AdamW)."""
        raise NotImplementedError("DistOptAdapter.build")

    def step(self) -> None:
        raise NotImplementedError("DistOptAdapter.step")

    def zero_grad(self, set_to_none: bool = True) -> None:
        raise NotImplementedError("DistOptAdapter.zero_grad")

    def reduce_scatter_grads(self) -> None:
        """PCIe-aware reduce-scatter of grads to owning rank's fp32 shard."""
        raise NotImplementedError("DistOptAdapter.reduce_scatter_grads")

    def all_gather_params(self) -> None:
        """Gather updated bf16 params back to all ranks after step."""
        raise NotImplementedError("DistOptAdapter.all_gather_params")
