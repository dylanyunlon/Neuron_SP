# SPDX-License-Identifier: Apache-2.0
"""tier_map.py — GPU tier discovery for the heterogeneous cluster.

Phase 1 skeleton. Signatures frozen (see ARCHITECTURE.md). Bodies raise NotImplementedError.
"""
from __future__ import annotations
import enum
from dataclasses import dataclass


class GPUTier(enum.Enum):
    A6000 = "a6000"          # SM8.6, ~48GB
    H100 = "h100"            # SM9.0, ~96GB
    BLACKWELL = "blackwell"  # SM12.0, ~96GB
    UNKNOWN = "unknown"


@dataclass
class TierInfo:
    rank: int
    tier: GPUTier
    total_vram_bytes: int
    numa_node: int
    peak_bf16_tflops: float


class TierMap:
    """Maps distributed rank -> GPUTier + memory budget + NUMA affinity."""

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self._by_rank: dict[int, TierInfo] = {}

    @classmethod
    def discover(cls) -> "TierMap":
        """Enumerate ranks via torch.cuda + NVML; fill TierInfo per rank."""
        raise NotImplementedError("TierMap.discover: enumerate ranks -> TierInfo")

    def tier_of(self, rank: int) -> GPUTier:
        raise NotImplementedError("TierMap.tier_of")

    def mem_budget(self, rank: int) -> int:
        """Usable optimizer-state bytes after model + activation reserve. Tier-specific."""
        raise NotImplementedError("TierMap.mem_budget")

    def ranks_of_tier(self, tier: GPUTier) -> list[int]:
        raise NotImplementedError("TierMap.ranks_of_tier")
