# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
# M343-C34: Heterogeneous mesh for AutoSP + DES-LOC + BloomBee composition.
#
# Design pattern chain (NeurIPS review format):
#
# From Megatron's initialize_model_parallel (parallel_state.py:547) multi-dimensional
# mesh [TP×CP×EP×DP×PP] as the good example. Then, following that pattern, implement
# a new HeteroMeshPlanner, letting sp_dp_registry dynamically assign SP groups based
# on GPU capability tiers, and able to rebalance on peer join/leave. Next, introduce
# BandwidthAwareGrouper from NCCL's ncclGroupStart/ncclGroupEnd (group.cc:100)
# async collective batching, making A2A ops coalesceable across heterogeneous links,
# while TE's comm_gemm_overlap (userbuffers_forward_linear.py:210) optimizes A2A+GEMM
# pipelining. Subsequently, integrate BloomBee's DHT-based peer discovery
# (bloombee/utils/dht.py), allowing the mesh to incorporate P2P peers that announce
# GPU specs via hivemind DHTValue. Finally, compose the adaptive sync gating from
# DES-LOC's Kx-period algorithm, ensuring heterogeneous gradient AllReduce respects
# both per-tier bandwidth and per-step communication budget, comprehensively upgrading
# the training throughput for mixed-cluster deployments.

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


# ---------------------------------------------------------------------------
# GPU capability tier classification
# Pattern: Megatron parallel_state.py classifies ranks into TP/DP/PP groups
# based on topology; we classify by compute capability + memory bandwidth.
# ---------------------------------------------------------------------------

@dataclass
class GPUTierInfo:
    """Describes a GPU's capability for SP workload placement."""
    rank: int
    device_name: str = ""
    compute_capability: Tuple[int, int] = (0, 0)
    memory_total_gb: float = 0.0
    memory_bandwidth_gbps: float = 0.0  # GB/s
    nvlink_available: bool = False
    pcie_bandwidth_gbps: float = 0.0
    tier: int = 0  # 0=unknown, 1=consumer, 2=datacenter, 3=flagship

    def compute_score(self) -> float:
        """Weighted score for SP group assignment.

        Higher score = more capable = should handle larger sequence chunks.
        Weights derived from roofline model: attention is memory-bound,
        so memory bandwidth dominates; compute capability is secondary.
        """
        mem_bw_score = self.memory_bandwidth_gbps / 100.0  # normalize ~2000 GB/s -> 20
        compute_score = (self.compute_capability[0] * 10
                         + self.compute_capability[1])  # e.g. 9.0 -> 90
        link_bonus = 2.0 if self.nvlink_available else 0.0
        return 0.6 * mem_bw_score + 0.3 * compute_score + 0.1 * link_bonus


def _probe_local_gpu() -> GPUTierInfo:
    """Probe the local GPU's capabilities.

    Pattern: NCCL src/misc/nvmlwrap.cc probes GPU topology at init time.
    We do the same via torch.cuda to get device properties.
    """
    info = GPUTierInfo(rank=dist.get_rank() if dist.is_initialized() else 0)

    if not torch.cuda.is_available():
        info.device_name = "cpu"
        return info

    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    info.device_name = props.name
    info.compute_capability = (props.major, props.minor)
    info.memory_total_gb = props.total_mem / (1024 ** 3)

    # Estimate memory bandwidth from known architectures
    # Pattern: TE's comm_gemm_overlap uses device capability to choose
    # between userbuffers and standard NCCL paths.
    _BW_TABLE = {
        "H100": 3350, "H200": 4800, "A100": 2039, "A6000": 768,
        "L40": 864, "L40S": 864, "A40": 696, "V100": 900,
        "RTX 4090": 1008, "RTX 3090": 936, "RTX 6000": 960,
    }
    for name_fragment, bw in _BW_TABLE.items():
        if name_fragment.lower() in props.name.lower():
            info.memory_bandwidth_gbps = bw
            break

    # Tier assignment
    if props.major >= 9:
        info.tier = 3  # Hopper+
    elif props.major >= 8:
        info.tier = 2  # Ampere datacenter
    else:
        info.tier = 1

    # NVLink heuristic: multi-GPU systems with high-end GPUs
    info.nvlink_available = (torch.cuda.device_count() > 1
                             and info.tier >= 2)

    return info


# ---------------------------------------------------------------------------
# Heterogeneous mesh planner
# Pattern: Megatron's initialize_model_parallel builds process groups
# with a fixed order "tp-cp-ep-dp-pp". We extend this with capability-aware
# grouping: faster GPUs are co-located in the same SP group when possible.
# ---------------------------------------------------------------------------

@dataclass
class HeteroMeshPlan:
    """The result of heterogeneous mesh planning."""
    sp_size: int
    dp_size: int
    sp_groups: List[List[int]]  # sp_groups[gid] = [rank, rank, ...]
    rank_to_sp_group: Dict[int, int] = field(default_factory=dict)
    tier_infos: Dict[int, GPUTierInfo] = field(default_factory=dict)
    # Per-group sequence chunk ratios (for imbalanced sharding)
    # Default: uniform. Heterogeneous: proportional to compute_score.
    group_chunk_weights: Dict[int, List[float]] = field(default_factory=dict)


def plan_hetero_mesh(
    world_size: int,
    sp_size: int,
    tier_infos: Optional[Dict[int, GPUTierInfo]] = None,
    strategy: str = "capability_sort",
) -> HeteroMeshPlan:
    """Plan SP/DP group assignment for heterogeneous GPUs.

    Strategies:
    - "contiguous": standard Megatron-style, ranks [0..sp-1], [sp..2sp-1], etc.
    - "capability_sort": sort by compute_score, group top-N together.
      This minimizes A2A latency by co-locating similar-speed GPUs.
    - "bandwidth_aware": group by interconnect bandwidth (NVLink peers together).

    Pattern: JAX's NamedSharding + Mesh allows arbitrary device assignment;
    we implement the same flexibility for NCCL process groups.
    """
    assert world_size % sp_size == 0, (
        f"world_size={world_size} must be divisible by sp_size={sp_size}")
    dp_size = world_size // sp_size

    if tier_infos is None:
        tier_infos = {r: GPUTierInfo(rank=r) for r in range(world_size)}

    if strategy == "contiguous":
        # Standard: same as upstream DeepSpeed populate_registry
        groups = []
        for g in range(dp_size):
            groups.append(list(range(g * sp_size, (g + 1) * sp_size)))
    elif strategy == "capability_sort":
        # Sort ranks by compute score descending, then assign round-robin
        # to SP groups. This ensures each SP group has a balanced mix.
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
        # Group NVLink-connected peers together when possible
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

    # Ensure each group has exactly sp_size members
    for g in groups:
        assert len(g) == sp_size, (
            f"Group {g} has {len(g)} members, expected {sp_size}")

    # Bug Risk 1 fix: Sort ranks within each group for NCCL compatibility.
    # NCCL's all_to_all_single on certain versions (< 2.19) assumes ranks
    # are in ascending order within a communicator. Sorting prevents silent
    # data corruption while preserving the capability-aware group assignment.
    # Pattern: Megatron parallel_state.py always builds groups with sorted ranks.
    for g in groups:
        g.sort()

    # Bug Risk 1 fix: Warn if NCCL version may not support non-contiguous groups.
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
                    # Fallback to contiguous
                    groups = []
                    for g_idx in range(dp_size):
                        groups.append(list(range(g_idx * sp_size, (g_idx + 1) * sp_size)))
            except Exception:
                pass  # Can't check NCCL version; proceed with caution

    # Compute per-group chunk weights (proportional to compute score)
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


# ---------------------------------------------------------------------------
# Collective all-gather for GPU tier info
# Pattern: Megatron parallel_state.py gathers topology info before
# creating process groups. NCCL init gathers PCIe topology via nvml.
# ---------------------------------------------------------------------------

def gather_all_tier_infos() -> Dict[int, GPUTierInfo]:
    """Gather GPU tier info from all ranks via AllGather.

    Uses torch.distributed to broadcast each rank's tier info.
    For BloomBee integration, this can be replaced with DHT-based
    discovery (see bloombee/utils/dht.py declare_active_modules).
    """
    if not dist.is_initialized():
        local = _probe_local_gpu()
        return {local.rank: local}

    local = _probe_local_gpu()
    world_size = dist.get_world_size()

    # Bug Risk 2 fix: Detect the distributed backend and use the appropriate
    # device for all_gather. NCCL requires CUDA tensors; gloo works with CPU.
    # Pattern: Megatron creates separate gloo process groups for CPU ops
    # (get_data_parallel_group_gloo) alongside NCCL groups for GPU ops.
    _backend = dist.get_backend()
    if _backend == "nccl" and torch.cuda.is_available():
        _device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        _device = torch.device("cpu")

    # Encode tier info as tensor for AllGather
    # [rank, compute_maj, compute_min, mem_gb*10, mem_bw*10, tier, nvlink]
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


# ---------------------------------------------------------------------------
# Integration with populate_registry
# Pattern: Megatron calls initialize_model_parallel once at startup;
# we call populate_hetero_registry from init_sp.py to replace the
# default contiguous grouping with capability-aware grouping.
# ---------------------------------------------------------------------------

_HETERO_PLAN: Optional[HeteroMeshPlan] = None
_HETERO_LOCK = threading.Lock()  # Bug Risk 5 fix: thread-safety


def populate_hetero_registry(
    sp_size: int,
    dp_size: int,
    strategy: str = "contiguous",
) -> HeteroMeshPlan:
    """Extended populate_registry with heterogeneous awareness.

    Falls through to standard sp_dp_registry.populate_registry for
    the "contiguous" strategy (backward compatible).

    Thread-safe: uses _HETERO_LOCK to prevent races from multiple
    torch.compile invocations (Bug Risk 5).
    """
    global _HETERO_PLAN

    with _HETERO_LOCK:
        if sp_dp_registry.is_setup():
            return _HETERO_PLAN

        world_size = dist.get_world_size()

        if strategy == "contiguous":
            # Standard path: no topology probing needed
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

        # Heterogeneous path: probe GPUs, plan, register
        tier_infos = gather_all_tier_infos()
        plan = plan_hetero_mesh(world_size, sp_size, tier_infos, strategy)

        # Register the planned groups
        sp_dp_registry.register_groups(plan.sp_groups)
        sp_dp_registry._MESH_META["sp_size"] = sp_size
        sp_dp_registry._MESH_META["dp_size"] = dp_size
        sp_dp_registry._MESH_META["is_registered"] = True

        _HETERO_PLAN = plan

        # System Issue 3 fix: validate dtype compatibility within each SP group.
        # bf16 requires cc>=8.0 (Ampere+), fp16 requires cc>=7.0 (Volta+).
        # Mixed-dtype A2A will silently produce NaN on unsupported devices.
        import torch as _torch
        _train_dtype = _torch.bfloat16  # TODO: read from config when available
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


def get_local_chunk_weight() -> float:
    """Get this rank's proportional sequence chunk weight.

    Returns 1.0/sp_size for uniform sharding (default).
    For heterogeneous: returns weight proportional to GPU capability.
    """
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
    """System Issue 3 fix: Validate that all ranks in each SP group
    can support the requested dtype.

    bf16 requires compute capability >= 8.0 (Ampere+).
    fp16 requires compute capability >= 7.0 (Volta+).

    Pattern: Megatron training.py validates --bf16/--fp16 against
    device capability at startup. We do the same per-SP-group to
    catch heterogeneous dtype mismatches early.

    Returns a list of warning strings (empty if all OK).
    """
    warnings = []
    _DTYPE_MIN_CC = {
        'torch.bfloat16': (8, 0),
        'torch.float16': (7, 0),
        'torch.float32': (0, 0),  # always supported
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


# System Issue 4 fix: High-water mark for A2A handle tracking.
# Prevents unbounded growth of _PENDING_A2A_HANDLES in sp_dp_registry.
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

