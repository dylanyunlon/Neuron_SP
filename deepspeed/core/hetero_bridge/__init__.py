# SPDX-License-Identifier: Apache-2.0
"""hetero_bridge — makes the migrated Megatron distributed optimizer + pipeline
schedules drive DES-LOC / AutoSP training on the heterogeneous PCIe cluster.

Phase 1: skeleton with frozen interfaces. See ARCHITECTURE.md.
"""
from .tier_map import TierMap, TierInfo, GPUTier
from .shard_planner import HeteroShardPlanner, ShardPlan
from .dist_opt_adapter import DistOptAdapter
from .desloc_sync_policy import DesLocSyncPolicy, SyncPeriods
from .pp_schedule_adapter import PPScheduleAdapter
from .autosp_hook import AutoSPHook
from .engine_integration import install
from .mem_probe import (
    probe_free_vram,
    budget_from_probe,
    validate_tier_info_vram,
    VRAM_SAFETY_MARGIN,
)

__all__ = [
    "TierMap", "TierInfo", "GPUTier",
    "HeteroShardPlanner", "ShardPlan",
    "DistOptAdapter",
    "DesLocSyncPolicy", "SyncPeriods",
    "PPScheduleAdapter",
    "AutoSPHook",
    "install",
    "probe_free_vram",
    "budget_from_probe",
    "validate_tier_info_vram",
    "VRAM_SAFETY_MARGIN",
]
