# SPDX-License-Identifier: Apache-2.0
"""shard_planner.py — heterogeneous fp32 optimizer-shard assignment.

Phase 1 skeleton. Signatures frozen. Bodies raise NotImplementedError.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from .tier_map import TierMap


@dataclass
class ShardPlan:
    rank_to_param_ids: dict[int, list[int]] = field(default_factory=dict)
    rank_to_bytes: dict[int, int] = field(default_factory=dict)
    rationale: str = ""


class HeteroShardPlanner:
    """Assigns fp32 optimizer shards so higher-VRAM tiers own more parameters."""

    def __init__(self, tier_map: "TierMap") -> None:
        self.tier_map = tier_map

    def plan(self, named_params: "list[tuple[str, torch.Tensor]]") -> "ShardPlan":
        """Deterministic shard assignment respecting TierMap.mem_budget.
        Same inputs must yield the same plan on every rank."""
        raise NotImplementedError("HeteroShardPlanner.plan")
