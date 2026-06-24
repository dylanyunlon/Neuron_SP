import os
import math
import time
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import torch
import deepspeed.comm as dist

from . import sp_dp_registry

logger = logging.getLogger(__name__)

@dataclass
class GPUTierInfo:
    rank: int
    device_name: str = ""
    compute_capability: Tuple[int, int] = (0, 0)
    memory_total_gb: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    nvlink_available: bool = False
    pcie_bandwidth_gbps: float = 0.0
    tier: int = 0

    def compute_score(self) -> float:
        mem_bw_score = self.memory_bandwidth_gbps / 100.0
        compute_score = (self.compute_capability[0] * 10
                         + self.compute_capability[1])
        link_bonus = 2.0 if self.nvlink_available else 0.0
        return 0.6 * mem_bw_score + 0.3 * compute_score + 0.1 * link_bonus

def _probe_local_gpu() -> GPUTierInfo:
    info = GPUTierInfo(rank=dist.get_rank() if dist.is_initialized() else 0)

    if not torch.cuda.is_available():
        info.device_name = "cpu"
        return info

    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    info.device_name = props.name
    info.compute_capability = (props.major, props.minor)
    info.memory_total_gb = props.total_memory / (1024 ** 3)

    _BW_TABLE = {
        "H100": 3350, "H200": 4800, "B100": 8000, "B200": 8000,
        "A100": 2039, "A6000": 768,
        "L40": 864, "L40S": 864, "A40": 696, "V100": 900,
        "RTX 4090": 1008, "RTX 3090": 936, "RTX 6000": 960,
    }
    for name_fragment, bw in _BW_TABLE.items():
        if name_fragment.lower() in props.name.lower():
            info.memory_bandwidth_gbps = bw
            break

    if props.major >= 9:
        info.tier = 3
    elif props.major >= 8:
        info.tier = 2
    else:
        info.tier = 1

    info.nvlink_available = (torch.cuda.device_count() > 1
                             and info.tier >= 2)

    return info

@dataclass
class HeteroMeshPlan:
    sp_size: int
    dp_size: int
    sp_groups: List[List[int]]
    rank_to_sp_group: Dict[int, int] = field(default_factory=dict)
    tier_infos: Dict[int, GPUTierInfo] = field(default_factory=dict)
    group_chunk_weights: Dict[int, List[float]] = field(default_factory=dict)

def plan_hetero_mesh(
    world_size: int,
    sp_size: int,
    tier_infos: Optional[Dict[int, GPUTierInfo]] = None,
    strategy: str = "capability_sort",
) -> HeteroMeshPlan:
    assert world_size % sp_size == 0, (
        f"world_size={world_size} must be divisible by sp_size={sp_size}")
    dp_size = world_size // sp_size

    if tier_infos is None:
        tier_infos = {r: GPUTierInfo(rank=r) for r in range(world_size)}

    if len(tier_infos) != world_size and strategy != "contiguous":
        logger.warning(
            f"[HeteroMesh] tier_infos has {len(tier_infos)} entries but "
            f"world_size={world_size}. Falling back to contiguous strategy.")
        strategy = "contiguous"

    if strategy == "contiguous":
        groups = []
        for g in range(dp_size):
            groups.append(list(range(g * sp_size, (g + 1) * sp_size)))
    elif strategy == "capability_sort":
        sorted_ranks = sorted(
            tier_infos.keys(),
            key=lambda r: tier_infos[r].compute_score(),
            reverse=True,
        )
        groups = [[] for _ in range(dp_size)]
        for i, rank in enumerate(sorted_ranks):
            gid = i % dp_size
            groups[gid].append(rank)
    elif strategy == "bandwidth_aware":
        nvlink_ranks = [r for r, t in tier_infos.items() if t.nvlink_available]
        other_ranks = [r for r, t in tier_infos.items() if not t.nvlink_available]
        all_sorted = nvlink_ranks + other_ranks
        groups = [[] for _ in range(dp_size)]
        for i, rank in enumerate(all_sorted):
            gid = i // sp_size
            if gid < dp_size:
                groups[gid].append(rank)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    for g in groups:
        assert len(g) == sp_size, (
            f"Group {g} has {len(g)} members, expected {sp_size}")

    for g in groups:
        g.sort()

    if strategy != "contiguous":
        _is_non_contiguous = any(
            g != list(range(g[0], g[0] + sp_size)) for g in groups
        )
        if _is_non_contiguous:
            try:
                nccl_ver = torch.cuda.nccl.version() if hasattr(torch.cuda, 'nccl') else (0, 0, 0)
                nccl_ver_int = nccl_ver[0] * 10000 + nccl_ver[1] * 100 + nccl_ver[2]
                if nccl_ver_int < 21900:
                    logger.warning(
                        f"[HeteroMesh] Non-contiguous SP groups detected with "
                        f"NCCL {nccl_ver}. NCCL >= 2.19.0 recommended for "
                        f"non-contiguous process groups. Falling back to "
                        f"contiguous strategy.")
                    groups = []
                    for g_idx in range(dp_size):
                        groups.append(list(range(g_idx * sp_size, (g_idx + 1) * sp_size)))
            except Exception:
                pass

    group_chunk_weights = {}
    for gid, ranks in enumerate(groups):
        scores = [tier_infos[r].compute_score() for r in ranks]
        total = sum(scores) or 1.0
        group_chunk_weights[gid] = [s / total for s in scores]

    rank_to_sp_group = {}
    for gid, ranks in enumerate(groups):
        for r in ranks:
            rank_to_sp_group[r] = gid

    return HeteroMeshPlan(
        sp_size=sp_size,
        dp_size=dp_size,
        sp_groups=groups,
        rank_to_sp_group=rank_to_sp_group,
        tier_infos=tier_infos,
        group_chunk_weights=group_chunk_weights,
    )

def gather_all_tier_infos() -> Dict[int, GPUTierInfo]:
    if not dist.is_initialized():
        local = _probe_local_gpu()
        return {local.rank: local}

    local = _probe_local_gpu()
    world_size = dist.get_world_size()

    _backend = dist.get_backend()
    if _backend == "nccl" and torch.cuda.is_available():
        _device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        _device = torch.device("cpu")

    info_tensor = torch.tensor([
        local.rank,
        local.compute_capability[0],
        local.compute_capability[1],
        round(local.memory_total_gb * 10),
        round(local.memory_bandwidth_gbps * 10),
        local.tier,
        1 if local.nvlink_available else 0,
    ], dtype=torch.long, device=_device)

    gathered = [torch.zeros_like(info_tensor) for _ in range(world_size)]
    dist.all_gather(gathered, info_tensor)

    result = {}
    for t in gathered:
        vals = t.tolist()
        rank = int(vals[0])
        result[rank] = GPUTierInfo(
            rank=rank,
            compute_capability=(int(vals[1]), int(vals[2])),
            memory_total_gb=vals[3] / 10.0,
            memory_bandwidth_gbps=vals[4] / 10.0,
            tier=int(vals[5]),
            nvlink_available=bool(vals[6]),
        )
    return result

_HETERO_PLAN: Optional[HeteroMeshPlan] = None
_HETERO_LOCK = threading.Lock()

def populate_hetero_registry(
    sp_size: int,
    dp_size: int,
    strategy: str = "contiguous",
) -> HeteroMeshPlan:
    global _HETERO_PLAN

    with _HETERO_LOCK:
        if sp_dp_registry.is_setup():
            return _HETERO_PLAN

        world_size = dist.get_world_size()

        if strategy == "contiguous":
            sp_dp_registry.populate_registry(sp_size, dp_size)
            tier_infos = {r: GPUTierInfo(rank=r) for r in range(world_size)}
            _HETERO_PLAN = HeteroMeshPlan(
                sp_size=sp_size,
                dp_size=dp_size,
                sp_groups=[[g * sp_size + i for i in range(sp_size)]
                           for g in range(dp_size)],
                rank_to_sp_group={r: r // sp_size for r in range(world_size)},
                tier_infos=tier_infos,
            )
            return _HETERO_PLAN

        tier_infos = gather_all_tier_infos()
        plan = plan_hetero_mesh(world_size, sp_size, tier_infos, strategy)

        sp_dp_registry.register_groups(plan.sp_groups)
        sp_dp_registry._MESH_META["sp_size"] = sp_size
        sp_dp_registry._MESH_META["dp_size"] = dp_size
        sp_dp_registry._MESH_META["is_registered"] = True

        _HETERO_PLAN = plan

        import torch as _torch
        _train_dtype = _torch.bfloat16
        dtype_warnings = validate_sp_group_dtype_consistency(plan, _train_dtype)
        for w in dtype_warnings:
            logger.warning(w)

        if dist.get_rank() == 0:
            logger.info(f"[HeteroMesh] strategy={strategy} SP={sp_size} DP={dp_size}")
            for gid, ranks in enumerate(plan.sp_groups):
                tiers = [tier_infos[r].tier for r in ranks]
                scores = [f"{tier_infos[r].compute_score():.1f}" for r in ranks]
                logger.info(f"  group[{gid}]: ranks={ranks} tiers={tiers} scores={scores}")

        return plan

def get_hetero_plan() -> Optional[HeteroMeshPlan]:
    return _HETERO_PLAN


def reset_hetero_plan():
    global _HETERO_PLAN
    with _HETERO_LOCK:
        _HETERO_PLAN = None

def get_local_chunk_weight() -> float:
    plan = _HETERO_PLAN
    if plan is None:
        return 1.0 / max(sp_dp_registry.sp_size(), 1)

    rank = dist.get_rank()
    gid = plan.rank_to_sp_group.get(rank, 0)
    weights = plan.group_chunk_weights.get(gid)
    if weights is None:
        return 1.0 / plan.sp_size

    local_idx = plan.sp_groups[gid].index(rank)
    return weights[local_idx]

def validate_sp_group_dtype_consistency(
    plan: HeteroMeshPlan,
    dtype: 'torch.dtype',
) -> List[str]:
    warnings = []
    _DTYPE_MIN_CC = {
        'torch.bfloat16': (8, 0),
        'torch.float16': (7, 0),
        'torch.float32': (0, 0),
    }

    dtype_str = str(dtype)
    min_cc = _DTYPE_MIN_CC.get(dtype_str, (0, 0))

    for gid, ranks in enumerate(plan.sp_groups):
        for rank in ranks:
            tier = plan.tier_infos.get(rank)
            if tier is None:
                continue
            cc = tier.compute_capability
            if cc[0] < min_cc[0] or (cc[0] == min_cc[0] and cc[1] < min_cc[1]):
                warnings.append(
                    f"[HeteroMesh] SP group {gid}: rank {rank} "
                    f"({tier.device_name}, cc={cc}) does not support "
                    f"{dtype_str} (requires cc>={min_cc}). "
                    f"Mixed-dtype A2A will fail at runtime.")
    return warnings

A2A_HANDLE_HIGH_WATER_MARK = 64

def enforce_handle_high_water_mark():
    pending = sp_dp_registry.pending_handle_count()
    if pending > A2A_HANDLE_HIGH_WATER_MARK:
        fenced = sp_dp_registry.fence_all_sp_handles()
        if fenced > 0:
            logger.debug(
                f"[HeteroMesh] Force-fenced {fenced} A2A handles "
                f"(high-water mark={A2A_HANDLE_HIGH_WATER_MARK})")
    return pending

def compute_tier_aware_grid_sizes(
    plan: HeteroMeshPlan,
    seq_len: int,
) -> Dict[int, int]:
    from .occupancy_grid import compute_a2a_grid_for_tier, get_cached_capability

    result = {}
    for gid, ranks in enumerate(plan.sp_groups):
        tiers = [plan.tier_infos.get(r, GPUTierInfo(rank=r)) for r in ranks]
        min_tier = min(t.tier for t in tiers) if tiers else 1
        cap = get_cached_capability()
        grid = compute_a2a_grid_for_tier(seq_len, min_tier, cap)
        result[gid] = grid
    return result
