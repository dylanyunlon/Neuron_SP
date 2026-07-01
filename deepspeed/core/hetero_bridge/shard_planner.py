# SPDX-License-Identifier: Apache-2.0
"""shard_planner.py — heterogeneous fp32 optimizer-shard assignment.

Assigns fp32 optimizer shards proportionally to each rank's available VRAM
budget, as returned by TierMap.mem_budget().  The proportional-split math
mirrors zero3_hetero_shard.ShardState.build() (lines 154-163) so the two
subsystems stay in lock-step without duplicating the ShardState object itself
(the engine already builds one in Phase 4b; we don't need a second).

ShardPlan.rank_to_param_ids maps each rank to the *set of param indices*
whose flat-buffer range overlaps that rank's shard window.  A param that
straddles a rank boundary appears in *both* neighbouring ranks' lists — this
is intentional: each rank that touches any bytes of a param must know about
it for gradient accumulation purposes.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch
    from .tier_map import TierMap

logger = logging.getLogger(__name__)


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

        Same inputs must yield the same plan on every rank.  The algorithm:

        1. Collect vram_weights = [mem_budget(r) for r in range(world_size)].
        2. Compute shard_sizes proportionally (mirrors ShardState.build math).
        3. Build shard_offsets (cumulative prefix sums).
        4. For each rank r, collect param indices whose flat-buffer range
           overlaps [shard_offsets[r], shard_offsets[r+1]).

        Params that straddle a boundary are listed in both neighbouring ranks
        so neither rank is missing gradient information for its owned bytes.

        Falls back to a full-replica single-rank plan when world_size <= 1
        or when named_params is empty.
        """
        import torch
        import torch.distributed as dist

        world_size = self.tier_map.world_size
        try:
            local_rank = dist.get_rank() if dist.is_initialized() else 0
        except Exception:
            local_rank = int(os.environ.get("RANK", "0"))

        plan = ShardPlan()

        # ── Edge cases ────────────────────────────────────────────────
        if not named_params:
            plan.rationale = "no trainable parameters"
            return plan

        if world_size <= 1:
            plan.rank_to_param_ids = {0: list(range(len(named_params)))}
            plan.rank_to_bytes = {
                0: sum(p.numel() * 4 for _, p in named_params)
            }
            plan.rationale = "single-rank: full replica"
            return plan

        # ── Step 1: VRAM weights from TierMap.mem_budget() ───────────
        # mem_budget() already applies tier-specific reserve fractions
        # (A6000: 35%, H100: 25%, Blackwell: 20%).
        vram_weights: List[int] = [
            self.tier_map.mem_budget(r) for r in range(world_size)
        ]
        # Guard: all weights must be positive for proportional split.
        if any(w <= 0 for w in vram_weights):
            logger.warning(
                "HeteroShardPlanner: non-positive mem_budget detected "
                "(%s); falling back to uniform split.", vram_weights
            )
            vram_weights = [1] * world_size

        # ── Step 2: Shard sizes (mirrors ShardState.build lines 154-163) ─
        raw_total: int = sum(p.numel() for _, p in named_params)
        wsum: float = float(sum(vram_weights))

        shard_sizes: List[int] = []
        assigned = 0
        for r in range(world_size - 1):
            size_r = int(vram_weights[r] / wsum * raw_total)
            shard_sizes.append(size_r)
            assigned += size_r
        shard_sizes.append(max(0, raw_total - assigned))

        # ── Step 3: Cumulative offsets ────────────────────────────────
        shard_offsets: List[int] = [0]
        for s in shard_sizes:
            shard_offsets.append(shard_offsets[-1] + s)

        # ── Step 4: Param flat-buffer positions ───────────────────────
        # Walk params in order and track each param's [global_start, global_end).
        param_global_starts: List[int] = []
        cursor = 0
        for _, p in named_params:
            param_global_starts.append(cursor)
            cursor += p.numel()
        # cursor == raw_total here

        # ── Step 5: Map params → ranks (overlap test) ─────────────────
        for r in range(world_size):
            plan.rank_to_param_ids[r] = []
            plan.rank_to_bytes[r] = shard_sizes[r] * 4  # fp32 bytes

        for idx, (_, p) in enumerate(named_params):
            g_start = param_global_starts[idx]
            g_end = g_start + p.numel()
            for r in range(world_size):
                lo = shard_offsets[r]
                hi = shard_offsets[r + 1]
                # Overlap: param range [g_start, g_end) ∩ shard [lo, hi) ≠ ∅
                if g_end > lo and g_start < hi:
                    plan.rank_to_param_ids[r].append(idx)

        plan.rationale = (
            f"VRAM-weighted hetero split across {world_size} ranks: "
            f"shard_sizes={shard_sizes}, "
            f"vram_weights_bytes={vram_weights}"
        )
        logger.info(
            "[HeteroShardPlanner] plan: %d params, %d total fp32 elems, "
            "shard_sizes=%s (local rank %d owns %d params)",
            len(named_params),
            raw_total,
            shard_sizes,
            local_rank,
            len(plan.rank_to_param_ids.get(local_rank, [])),
        )
        return plan
