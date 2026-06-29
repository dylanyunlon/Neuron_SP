# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# Ported from Megatron-LM/megatron/core/parallel_state.py (2238 lines)
# with DES-LOC tier-group extensions for heterogeneous GPU clusters.
"""Parallel state management for DeepSpeed Neuron_SP.

Manages TP / PP / DP / CP / EP process groups and optionally DES-LOC tier
groups (one group per GPU-tier, e.g. 'datacenter' / 'professional').

All collective operations in deepspeed/core/ should route through the
accessors defined here rather than calling torch.distributed directly.

Typical usage::

    import torch.distributed as dist
    from deepspeed.core.parallel_state import (
        initialize_model_parallel,
        get_tensor_model_parallel_group,
        get_data_parallel_group,
    )

    dist.init_process_group(backend="nccl")
    initialize_model_parallel(
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=2,
    )
"""

from __future__ import annotations

import logging
import os
import warnings
from datetime import timedelta
from math import log2
from typing import Callable, Dict, List, Optional, Tuple

import torch

from deepspeed.core.desloc_config import DesLocConfig, TierSpec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level process-group state
# (mirrors Megatron parallel_state variable layout exactly)
# ---------------------------------------------------------------------------

# Intra-layer model parallel (tensor parallel)
_TENSOR_MODEL_PARALLEL_GROUP: Optional[torch.distributed.ProcessGroup] = None
_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS: Optional[List[int]] = None

# Inter-layer model parallel (pipeline parallel)
_PIPELINE_MODEL_PARALLEL_GROUP = None   # may become list[group] for VPP
_PIPELINE_GLOBAL_RANKS = None           # may become list[list[int]] for VPP

# Combined tensor + pipeline (model parallel)
_MODEL_PARALLEL_GROUP: Optional[torch.distributed.ProcessGroup] = None
_MODEL_PARALLEL_GLOBAL_RANKS: Optional[List[int]] = None

# Data parallel
_DATA_PARALLEL_GROUP: Optional[torch.distributed.ProcessGroup] = None
_DATA_PARALLEL_GROUP_GLOO: Optional[torch.distributed.ProcessGroup] = None
_DATA_PARALLEL_GLOBAL_RANKS: Optional[List[int]] = None

# DP + CP combined (for weight-grad all-reduce when CP > 1)
_DATA_PARALLEL_GROUP_WITH_CP: Optional[torch.distributed.ProcessGroup] = None
_DATA_PARALLEL_GROUP_WITH_CP_GLOO: Optional[torch.distributed.ProcessGroup] = None
_DATA_PARALLEL_GLOBAL_RANKS_WITH_CP: Optional[List[int]] = None

# Intra-partial DP (for distributed optimizer sharding)
_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP: Optional[torch.distributed.ProcessGroup] = None
_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO: Optional[torch.distributed.ProcessGroup] = None

# Insight I2: independent AG/RS groups (Megatron M3102)
# Separate NCCL communicators for forward all-gather and backward reduce-scatter.
# On PCIe topologies (no NVLink) a single DP communicator serialises AG and RS;
# two independent communicators allow the NCCL scheduler to overlap them truly
# concurrently across the PCIe fabric, improving utilisation by up to 40%.
_DATA_PARALLEL_GROUP_INDEPENDENT_AG: Optional[torch.distributed.ProcessGroup] = None
_DATA_PARALLEL_GROUP_INDEPENDENT_RS: Optional[torch.distributed.ProcessGroup] = None

# Context parallel
_CONTEXT_PARALLEL_GROUP: Optional[torch.distributed.ProcessGroup] = None
_CONTEXT_PARALLEL_GLOBAL_RANKS: Optional[List[int]] = None
_HIERARCHICAL_CONTEXT_PARALLEL_GROUPS: Optional[List[torch.distributed.ProcessGroup]] = None
_HYBRID_DP_CP_GROUPS: Dict[int, torch.distributed.ProcessGroup] = {}

# Tensor + Data parallel (for FP8 amax reduction)
_TENSOR_AND_DATA_PARALLEL_GROUP: Optional[torch.distributed.ProcessGroup] = None
_TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP: Optional[torch.distributed.ProcessGroup] = None

# Tensor + Context parallel
_TENSOR_AND_CONTEXT_PARALLEL_GROUP: Optional[torch.distributed.ProcessGroup] = None

# Embedding groups (pipeline stages that hold embedding weights)
_EMBEDDING_GROUP: Optional[torch.distributed.ProcessGroup] = None
_EMBEDDING_GLOBAL_RANKS: Optional[List[int]] = None
_POSITION_EMBEDDING_GROUP: Optional[torch.distributed.ProcessGroup] = None
_POSITION_EMBEDDING_GLOBAL_RANKS: Optional[List[int]] = None

# Expert model parallel
_EXPERT_MODEL_PARALLEL_GROUP: Optional[torch.distributed.ProcessGroup] = None
_EXPERT_MODEL_PARALLEL_RANKS: Optional[List[int]] = None

# Expert tensor parallel
_EXPERT_TENSOR_PARALLEL_GROUP: Optional[torch.distributed.ProcessGroup] = None

# Expert tensor + model combined
_EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP: Optional[torch.distributed.ProcessGroup] = None

# Expert tensor + model + pipeline combined
_EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP: Optional[torch.distributed.ProcessGroup] = None

# Expert data parallel
_EXPERT_DATA_PARALLEL_GROUP: Optional[torch.distributed.ProcessGroup] = None
_EXPERT_DATA_PARALLEL_GROUP_GLOO: Optional[torch.distributed.ProcessGroup] = None
_INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP: Optional[torch.distributed.ProcessGroup] = None
_INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO: Optional[torch.distributed.ProcessGroup] = None
_INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP: Optional[torch.distributed.ProcessGroup] = None

# Distributed optimizer instance groups
_INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP: Optional[torch.distributed.ProcessGroup] = None

# Virtual pipeline parallel
_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK: Optional[int] = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE: Optional[int] = None

# Overrideable world-size / rank values (set_xxx calls)
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE: Optional[int] = None
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE: Optional[int] = None
_MPU_DATA_PARALLEL_WORLD_SIZE: Optional[int] = None
_MPU_DATA_PARALLEL_RANK: Optional[int] = None
_MPU_TENSOR_MODEL_PARALLEL_RANK: Optional[int] = None
_MPU_PIPELINE_MODEL_PARALLEL_RANK: Optional[int] = None
_MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE: Optional[int] = None
_MPU_EXPERT_MODEL_PARALLEL_RANK: Optional[int] = None
_MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE: Optional[int] = None
_MPU_EXPERT_TENSOR_PARALLEL_RANK: Optional[int] = None

# Track all created groups for timeout updates (None = default PG)
_global_process_group_list: Optional[List] = None

# ---------------------------------------------------------------------------
# DES-LOC tier groups  (one per TierType, keyed by tier_type.name.lower())
# ---------------------------------------------------------------------------
_TIER_GROUPS: Dict[str, torch.distributed.ProcessGroup] = {}
_LOCAL_TIER: Optional[TierSpec] = None


# ---------------------------------------------------------------------------
# NCCL flight recorder auto-configuration  (Megatron M3499 / PR #3806)
# ---------------------------------------------------------------------------

def _configure_nccl_flight_recorder(config) -> None:
    """Set NCCL flight recorder env vars from ModelParallelConfig.

    From Megatron M3499 (PR #3806): configure via training config rather
    than manual os.environ before launch. Uses setdefault — never overwrites
    pre-existing env vars.

    Critical for PCIe topologies (DES-LOC: A6000+H100+Blackwell no NVLink)
    where higher latency makes NCCL hangs more frequent.
    """
    if config is None:
        return
    dump_path = getattr(config, 'flight_recorder_dump_path', None)
    if dump_path is not None:
        os.environ.setdefault('TORCH_FR_DUMP_TEMP_FILE', str(dump_path))
        os.environ.setdefault('TORCH_NCCL_DEBUG_INFO_TEMP_FILE', str(dump_path))
    buf = getattr(config, 'flight_recorder_trace_buffer_size', 36864)
    os.environ.setdefault('TORCH_NCCL_TRACE_BUFFER_SIZE', str(buf))
    dot = getattr(config, 'flight_recorder_dump_on_timeout', True)
    os.environ.setdefault('TORCH_NCCL_DUMP_ON_TIMEOUT', '1' if dot else '0')
    ist = getattr(config, 'flight_recorder_include_stack_trace', True)
    os.environ.setdefault('TORCH_INCLUDE_STACK_TRACE', '1' if ist else '0')
    oa = getattr(config, 'flight_recorder_include_only_active', False)
    os.environ.setdefault('TORCH_INCLUDE_ONLY_ACTIVE', '1' if oa else '0')
    ed = getattr(config, 'flight_recorder_extra_dump_on_exec', False)
    os.environ.setdefault('TORCH_NCCL_EXTRA_DUMP_ON_EXEC', '1' if ed else '0')


# ---------------------------------------------------------------------------
# Utility: GlobalMemoryBuffer (lightweight replacement for Megatron's version)
# ---------------------------------------------------------------------------

class GlobalMemoryBuffer:
    """A simple memory buffer that re-uses the same tensor across calls."""

    def __init__(self) -> None:
        self.buffer: Dict[Tuple, torch.Tensor] = {}

    def get_tensor(self, tensor_shape: List[int], dtype: torch.dtype, name: str) -> torch.Tensor:
        required_len = 1
        for d in tensor_shape:
            required_len *= d
        key = (name, dtype)
        if key not in self.buffer or self.buffer[key].numel() < required_len:
            device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            self.buffer[key] = torch.empty(
                required_len, dtype=dtype, device=device, requires_grad=False
            )
        return self.buffer[key][0:required_len].view(*tensor_shape)


_GLOBAL_MEMORY_BUFFER: Optional[GlobalMemoryBuffer] = None


# ---------------------------------------------------------------------------
# NCCL options helper
# ---------------------------------------------------------------------------

def get_nccl_options(pg_name: str, nccl_comm_cfgs: dict):
    """Return ProcessGroupNCCL.Options for pg_name if configured, else None."""
    if pg_name in nccl_comm_cfgs:
        nccl_options = torch.distributed.ProcessGroupNCCL.Options(
            is_high_priority_stream=nccl_comm_cfgs[pg_name].get("is_high_priority_stream", False)
        )
        cfg = nccl_comm_cfgs[pg_name]
        if "cga_cluster_size" in cfg:
            nccl_options.config.cga_cluster_size = cfg["cga_cluster_size"]
        if "max_ctas" in cfg:
            nccl_options.config.max_ctas = cfg["max_ctas"]
        if "min_ctas" in cfg:
            nccl_options.config.min_ctas = cfg["min_ctas"]
        if "net_name" in cfg:
            nccl_options.config.net_name = cfg["net_name"]
            if nccl_options.config.net_name.lower() not in ["ib", "socket"]:
                raise RuntimeError(
                    f"net_name ({nccl_options.config.net_name}) is not supported. "
                    "Accepted values: 'IB' or 'socket'."
                )
        return nccl_options
    return None


def update_pg_timeout(
    timeout: timedelta,
    pg: Optional[torch.distributed.ProcessGroup] = None,
) -> None:
    """Update timeout on one or all process groups.

    Fix from Megatron M3728: log the failing pg, timeout value, and global
    process-group list before re-raising so timeouts are diagnosable.
    """
    if hasattr(torch.distributed.distributed_c10d, "_set_pg_timeout"):
        torch.distributed.barrier(pg)
        torch.cuda.synchronize()
        try:
            if pg is None:
                global _global_process_group_list
                for grp in (_global_process_group_list or []):
                    torch.distributed.distributed_c10d._set_pg_timeout(timeout, grp)
            else:
                torch.distributed.distributed_c10d._set_pg_timeout(timeout, pg)
        except Exception as exc:
            logger.error("Error updating pg timeout: %s", exc)
            logger.error("Process group: %s", pg)
            logger.error("Timeout: %s", timeout)
            logger.error("Global process group list: %s", _global_process_group_list)
            raise


def update_pg_timeout_per_tier(
    intra_tier_timeout: timedelta,
    inter_tier_timeout: timedelta,
) -> None:
    """Update NCCL timeouts with per-tier granularity for DES-LOC topologies.

    In heterogeneous topologies (2xA6000 + 1xH100 + 2xBlackwell PCIe),
    the same collective completes in very different times depending on whether
    it crosses tier boundaries (slow PCIe) or stays within a tier (fast NVLink
    or local bus). A single global timeout causes false positives on slow paths
    or misses real hangs on fast paths.

    - intra_tier_timeout: for PGs whose members all fall within one hardware tier
    - inter_tier_timeout: for PGs spanning multiple tiers (DP, PP cross-tier).
      Should be 3-5x intra_tier_timeout.

    From Megatron M2674: EP group was missing timeout; we generalise the fix
    to make timeout a per-tier first-class concept rather than a global setting.

    Args:
        intra_tier_timeout: timeout for same-tier process groups.
        inter_tier_timeout: timeout for cross-tier process groups.
    """
    global _global_process_group_list, _TIER_GROUPS
    if _global_process_group_list is None:
        logger.warning("update_pg_timeout_per_tier: called before initialize_model_parallel()")
        return

    import torch.distributed as _dist

    # Build set of ranks per tier from _TIER_GROUPS
    tier_rank_sets = []
    if _TIER_GROUPS:
        for tier_name, pg in _TIER_GROUPS.items():
            try:
                tier_rank_sets.append(set(_dist.get_process_group_ranks(pg)))
            except Exception:
                pass

    updated_intra = updated_inter = 0
    for pg in _global_process_group_list:
        if pg is None:
            continue
        try:
            pg_ranks = set(_dist.get_process_group_ranks(pg))
            is_intra = (
                any(pg_ranks.issubset(ts) for ts in tier_rank_sets)
                if tier_rank_sets else False
            )
            timeout = intra_tier_timeout if is_intra else inter_tier_timeout
            if hasattr(torch.distributed.distributed_c10d, "_set_pg_timeout"):
                torch.distributed.distributed_c10d._set_pg_timeout(timeout, pg)
            if is_intra:
                updated_intra += 1
            else:
                updated_inter += 1
        except Exception as exc:
            logger.warning("update_pg_timeout_per_tier: skipping pg: %s", exc)

    logger.info(
        "update_pg_timeout_per_tier: intra=%s (%d PGs), inter=%s (%d PGs)",
        intra_tier_timeout, updated_intra, inter_tier_timeout, updated_inter,
    )
    # From Megatron M2674: generalise timeout fix to per-tier for DES-LOC


def _torch_version_ge(major: int, minor: int) -> bool:
    """Return True if torch.__version__ >= major.minor (ignores patch/pre-release)."""
    try:
        v = torch.__version__.split(".")
        return (int(v[0]), int(v[1].split("+")[0].split("a")[0].split("b")[0])) >= (major, minor)
    except Exception:
        return False


def create_group(
    ranks: Optional[List[int]] = None,
    timeout: Optional[timedelta] = None,
    backend: Optional[str] = None,
    pg_options=None,
    use_local_synchronization: bool = False,
    group_desc: Optional[str] = None,
) -> torch.distributed.ProcessGroup:
    """Create a ProcessGroup and register it for timeout updates.

    Fix from Megatron M3728: use explicit version check instead of
    try/except TypeError to strip unsupported kwargs. The try/except approach
    can swallow real TypeErrors from within new_group, masking bugs. We also
    strip timeout=None on old PyTorch (< 2.4) because older versions expect a
    timedelta default rather than accepting None.
    """
    kwargs: dict = {
        "ranks": ranks,
        "timeout": timeout,
        "backend": backend,
        "pg_options": pg_options,
        "use_local_synchronization": use_local_synchronization,
        "group_desc": group_desc,
    }
    # group_desc and None-timeout accepted only in torch >= 2.4
    if not _torch_version_ge(2, 4):
        kwargs.pop("group_desc", None)
        if kwargs.get("timeout") is None:
            kwargs.pop("timeout", None)
    group = torch.distributed.new_group(**kwargs)

    global _global_process_group_list
    if _global_process_group_list is None:
        _global_process_group_list = [None]  # None = default PG
    if ranks is not None and torch.distributed.get_rank() in ranks:
        _global_process_group_list.append(group)
    return group


# ---------------------------------------------------------------------------
# Rank-group generation helpers (ported from Megatron)
# ---------------------------------------------------------------------------

def generate_masked_orthogonal_rank_groups(
    world_size: int,
    parallel_size: List[int],
    mask: List[bool],
) -> List[List[int]]:
    """Generate orthogonal parallel groups based on parallel_size and mask.

    Ported verbatim from Megatron-LM/megatron/core/parallel_state.py.

    Given a layout where::

        global_rank = sum(idx[i] * stride[i])  (stride = prefix_product of parallel_size)

    and mask[i] selects which dimensions to *include* in each group,
    returns a list of rank-lists, one per group.
    """

    def prefix_product(a: List[int], init: int = 1) -> List[int]:
        r = [init]
        for v in a:
            init = init * v
            r.append(init)
        return r

    def inner_product(a: List[int], b: List[int]) -> int:
        return sum(x * y for x, y in zip(a, b))

    def decompose(index: int, shape: List[int], stride: Optional[List[int]] = None) -> List[int]:
        """Inverse of inner_product(idx, stride) == index, given shape."""
        if stride is None:
            stride = prefix_product(shape)
        idx = [(index // d) % s for s, d in zip(shape, stride)]
        # stride[-1] is unused (it equals world_size); check against stride[:-1]
        assert sum(x * y for x, y in zip(idx, stride[:-1])) == index, (
            f"idx {idx} with shape {shape} does not reconstruct index {index}"
        )
        return idx

    masked_shape = [s for s, m in zip(parallel_size, mask) if m]
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]

    global_stride = prefix_product(parallel_size)
    masked_stride = [d for d, m in zip(global_stride, mask) if m]
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]

    group_size = prefix_product(masked_shape)[-1]
    num_of_group = world_size // group_size

    ranks = []
    for group_index in range(num_of_group):
        # decompose group_index using the UNMASKED shape (no stride override → uses prefix_product)
        decomposed_group_idx = decompose(group_index, unmasked_shape)
        rank = []
        for rank_in_group in range(group_size):
            # decompose rank_in_group using the MASKED shape
            decomposed_rank_idx = decompose(rank_in_group, masked_shape)
            rank.append(
                inner_product(decomposed_rank_idx, masked_stride)
                + inner_product(decomposed_group_idx, unmasked_stride)
            )
        ranks.append(rank)
    return ranks


def create_hierarchical_groups(
    rank: int,
    ranks: List[int],
    hierarchical_group_sizes: List[int],
    create_gloo_process_groups: bool = False,
    pg_options=None,
    timeout: Optional[timedelta] = None,
    group_desc: Optional[str] = None,
) -> Tuple[List, List]:
    """Create hierarchical sub-groups for context parallelism."""
    try:
        import einops
        import numpy as np
    except ImportError:
        raise ImportError(
            "einops and numpy are required for hierarchical context parallel groups. "
            "Install with: pip install einops numpy"
        )

    hierarchical_groups: List = []
    hierarchical_groups_gloo: List = []
    if not isinstance(pg_options, list):
        pg_options = [pg_options] * len(hierarchical_group_sizes)

    for level in range(len(hierarchical_group_sizes)):
        u = int(np.prod(hierarchical_group_sizes[:level]))
        s = hierarchical_group_sizes[level]
        l = int(np.prod(hierarchical_group_sizes[level + 1:]))
        rearranged = einops.rearrange(
            np.array(ranks), "(l s u) -> (l u) s", u=u, s=s, l=l
        ).tolist()
        for sub_ranks in rearranged:
            sub_group = create_group(
                sub_ranks,
                timeout=timeout,
                pg_options=pg_options[level],
                group_desc=f"HIERARCHICAL_{group_desc}_L{level}",
            )
            if create_gloo_process_groups:
                sub_group_gloo = create_group(
                    sub_ranks,
                    timeout=timeout,
                    backend="gloo",
                    group_desc=f"HIERARCHICAL_{group_desc}_GLOO_L{level}",
                )
            else:
                sub_group_gloo = None
            if rank in sub_ranks:
                hierarchical_groups.append(sub_group)
                hierarchical_groups_gloo.append(sub_group_gloo)

    assert rank not in ranks or len(hierarchical_groups) == len(hierarchical_group_sizes)
    return hierarchical_groups, hierarchical_groups_gloo


def create_hybrid_dp_cp_groups(
    rank: int,
    ranks: List[int],
    pg_options,
) -> Dict[int, torch.distributed.ProcessGroup]:
    """Create power-of-2-sized sub-groups within a DP×CP group."""
    hybrid: Dict[int, torch.distributed.ProcessGroup] = {}
    group_sizes = [2 ** i for i in range(1, int(log2(len(ranks))))]
    for gs in group_sizes:
        for i in range(0, len(ranks), gs):
            grp = create_group(
                ranks[i: i + gs],
                pg_options=pg_options,
                group_desc=f"HYBRID_DP_CP_GROUP_{gs}",
            )
            if rank in ranks[i: i + gs]:
                assert gs not in hybrid, (
                    f"Rank {rank} appears in multiple Hybrid DP CP groups of size {gs}"
                )
                hybrid[gs] = grp
    return hybrid


# ---------------------------------------------------------------------------
# RankGenerator  (ported from Megatron)
# ---------------------------------------------------------------------------

class RankGenerator:
    """Generate process-group rank lists for TP / PP / DP / CP / EP."""

    def __init__(
        self,
        tp: int,
        ep: int,
        dp: int,
        pp: int,
        cp: int,
        order: str,
        rank_offset: int = 0,
    ) -> None:
        assert ep == 1 or cp == 1, (
            "Both EP and CP > 1 is not allowed in one RankGenerator. "
            "CP is only in default generator; EP only in expert generator."
        )
        self.tp = tp
        self.ep = ep
        self.dp = dp
        self.pp = pp
        self.cp = cp
        self.rank_offset = rank_offset
        self.world_size = tp * dp * pp * cp * ep

        self.name_to_size = {"tp": tp, "pp": pp, "dp": dp, "ep": ep, "cp": cp}
        order = order.lower()
        for name, size in self.name_to_size.items():
            if name not in order and size != 1:
                raise RuntimeError(
                    f"The size of ({name}) is ({size}), but it is not specified in order ({order})."
                )
            elif name not in order:
                order = order + "-" + name
        self.order = order
        self.ordered_size = [self.name_to_size[t] for t in order.split("-")]

    def get_mask(self, order: str, token: str) -> List[bool]:
        ordered_token = order.split("-")
        token_list = token.split("-")
        mask = [False] * len(ordered_token)
        for t in token_list:
            mask[ordered_token.index(t)] = True
        return mask

    def get_ranks(self, token: str) -> List[List[int]]:
        """Return rank groups for the given parallelism token (e.g. 'tp', 'dp-cp')."""
        mask = self.get_mask(self.order, token)
        ranks = generate_masked_orthogonal_rank_groups(
            self.world_size, self.ordered_size, mask
        )
        if self.rank_offset > 0:
            for rg in ranks:
                for i in range(len(rg)):
                    rg[i] += self.rank_offset
        return ranks


# ---------------------------------------------------------------------------
# Embedding rank helpers
# ---------------------------------------------------------------------------

def default_embedding_ranks(pp_ranks: List[int], vp_stage: Optional[int] = None) -> List[int]:
    """Return ranks that hold embedding weights (first + last PP stage)."""
    if len(pp_ranks) == 1:
        return [pp_ranks[0]]
    return [pp_ranks[0], pp_ranks[-1]]


def default_position_embedding_ranks(
    pp_ranks: List[int], vp_stage: Optional[int] = None
) -> List[int]:
    """Return ranks that hold position embeddings (first PP stage only)."""
    return [pp_ranks[0]]


def overwrite_nccl_comm_cfgs(nccl_comm_cfgs: dict, pg_name: str, key_value_pair: tuple) -> None:
    if pg_name not in nccl_comm_cfgs:
        nccl_comm_cfgs[pg_name] = {}
    nccl_comm_cfgs[pg_name][key_value_pair[0]] = key_value_pair[1]


# ---------------------------------------------------------------------------
# initialize_model_parallel  (core entry point)
# ---------------------------------------------------------------------------

# From Megatron M2821: NEVER use LOCAL_RANK as global rank in init_process_group.
# LOCAL_RANK is node-local (0..N-1). Global rank must be os.environ["RANK"].
# Using LOCAL_RANK causes world_size mismatch and deadlock in multi-node training.
# Only use LOCAL_RANK for: torch.cuda.set_device(local_rank)
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_comm_backend: Optional[str] = None,
    use_sharp: bool = False,
    context_parallel_size: int = 1,
    hierarchical_context_parallel_sizes: Optional[List[int]] = None,
    hybrid_context_parallel: bool = False,
    expert_model_parallel_size: int = 1,
    num_distributed_optimizer_instances: int = 1,
    expert_tensor_parallel_size: Optional[int] = None,
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: int = 30,
    order: str = "tp-cp-ep-dp-pp",
    get_embedding_ranks: Optional[Callable[[List[int], Optional[int]], List[int]]] = None,
    get_position_embedding_ranks: Optional[Callable[[List[int], Optional[int]], List[int]]] = None,
    create_gloo_process_groups: bool = True,
    high_priority_stream_groups: Optional[List[str]] = None,
    sharp_enabled_group: Optional[str] = None,
    rank_offset: int = 0,
    local_world_size: Optional[int] = None,
    # DES-LOC extension
    desloc_config: Optional[DesLocConfig] = None,
    config=None,
) -> None:
    """Initialize all model-parallel process groups.

    Must be called after ``torch.distributed.init_process_group()``.

    Creates TP, PP, DP, CP, EP groups following Megatron's rank layout::

        global_rank = tp_rank
                    + cp_rank * tp
                    + ep_rank * tp * cp
                    + dp_rank * tp * cp * ep
                    + pp_rank * tp * cp * ep * dp

    (exact ordering controlled by the ``order`` parameter).

    Additionally creates DES-LOC tier groups when ``desloc_config`` is
    provided, grouping GPUs by their :class:`~deepspeed.core.desloc_config.TierType`.

    Args:
        tensor_model_parallel_size: Number of GPUs for intra-layer (tensor) parallelism.
        pipeline_model_parallel_size: Number of pipeline stages.
        virtual_pipeline_model_parallel_size: Stages per GPU for interleaved 1F1B.
        pipeline_model_parallel_comm_backend: Backend for PP P2P comms ('nccl' or 'ucc').
        use_sharp: Enable SHARP acceleration for DP all-reduce.
        context_parallel_size: Number of ranks for context (sequence chunk) parallelism.
        hierarchical_context_parallel_sizes: Sizes for hierarchical CP sub-groups.
        hybrid_context_parallel: Create power-of-2 CP hybrid groups.
        expert_model_parallel_size: Number of ranks for expert-model parallelism (MoE).
        num_distributed_optimizer_instances: Partial distributed optimizer sharding factor.
        expert_tensor_parallel_size: TP degree for expert layers (defaults to tp_size).
        nccl_communicator_config_path: Path to YAML NCCL config.
        distributed_timeout_minutes: Timeout for distributed collectives.
        order: Rank layout order string, e.g. 'tp-cp-ep-dp-pp'.
        get_embedding_ranks: Callable that selects embedding ranks from a PP group.
        get_position_embedding_ranks: Callable that selects position-embedding ranks.
        create_gloo_process_groups: Also create Gloo groups for CPU fallback.
        high_priority_stream_groups: PG names that should use high-priority NCCL streams.
        sharp_enabled_group: Which group to enable SHARP on ('dp' or 'dp_replica').
        rank_offset: Offset added to all ranks (for multi-node sub-worlds).
        local_world_size: Override torch.distributed world_size (for sub-worlds).
        desloc_config: DES-LOC configuration with tier specs for heterogeneous GPU groups.
    """
    # -- SHARP env-var cleanup (must precede any new_group calls) --
    if "NCCL_COLLNET_ENABLE" in os.environ:
        del os.environ["NCCL_COLLNET_ENABLE"]

    if use_sharp:
        if sharp_enabled_group is None:
            sharp_enabled_group = "dp"
        else:
            assert sharp_enabled_group in ["dp", "dp_replica"], (
                f"Invalid sharp_enabled_group: {sharp_enabled_group}. "
                "Must be 'dp' or 'dp_replica'."
            )
            if sharp_enabled_group == "dp_replica":
                assert num_distributed_optimizer_instances > 1, (
                    "dp_replica group requires num_distributed_optimizer_instances > 1"
                )
    else:
        assert sharp_enabled_group is None, (
            "sharp_enabled_group is only valid when use_sharp=True"
        )

    if get_embedding_ranks is None:
        get_embedding_ranks = default_embedding_ranks
    if get_position_embedding_ranks is None:
        get_position_embedding_ranks = default_position_embedding_ranks

    assert torch.distributed.is_initialized(), (
        "torch.distributed must be initialized before initialize_model_parallel()"
    )

    world_size: int = (
        local_world_size if local_world_size is not None
        else torch.distributed.get_world_size()
    )
    rank: int = torch.distributed.get_rank()

    model_size = (
        tensor_model_parallel_size
        * pipeline_model_parallel_size
        * context_parallel_size
    )
    if world_size % model_size != 0:
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by "
            f"tp({tensor_model_parallel_size}) * pp({pipeline_model_parallel_size}) "
            f"* cp({context_parallel_size}) = {model_size}"
        )

    data_parallel_size: int = world_size // model_size

    # -- Virtual pipeline setup --
    if virtual_pipeline_model_parallel_size is not None:
        if pipeline_model_parallel_size <= 1:
            raise RuntimeError(
                "pipeline_model_parallel_size must be > 1 for interleaved (VPP) schedule"
            )
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

    # -- NCCL communicator config --
    nccl_comm_cfgs: dict = {}
    if nccl_communicator_config_path is not None:
        try:
            import yaml
        except ImportError:
            raise RuntimeError(
                "Cannot import `yaml`. Setting custom NCCL communicator configs "
                "requires the yaml package: pip install pyyaml"
            )
        with open(nccl_communicator_config_path, "r") as fh:
            nccl_comm_cfgs = yaml.safe_load(fh)

    high_priority_stream_groups = high_priority_stream_groups or []
    for pg_name in high_priority_stream_groups:
        overwrite_nccl_comm_cfgs(nccl_comm_cfgs, pg_name, ("is_high_priority_stream", True))

    timeout = timedelta(minutes=distributed_timeout_minutes)

    # -- Build rank generators --
    decoder_rank_generator = RankGenerator(
        tp=tensor_model_parallel_size,
        ep=1,
        dp=data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=context_parallel_size,
        order=order,
        rank_offset=rank_offset,
    )

    if expert_tensor_parallel_size is None:
        expert_tensor_parallel_size = tensor_model_parallel_size
    expert_tmp_size = (
        expert_tensor_parallel_size
        * expert_model_parallel_size
        * pipeline_model_parallel_size
    )
    expert_data_parallel_size = world_size // expert_tmp_size
    if world_size % expert_tmp_size != 0:
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by expert TP×EP×PP = {expert_tmp_size}"
        )

    expert_decoder_rank_generator = RankGenerator(
        tp=expert_tensor_parallel_size,
        ep=expert_model_parallel_size,
        dp=expert_data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=1,
        order=order,
        rank_offset=rank_offset,
    )

    assert (
        order.endswith("pp")
        or pipeline_model_parallel_size == 1
        or expert_data_parallel_size == data_parallel_size
    ), (
        "When not using pp-last ordering, the data-parallel size of attention and MoE "
        "layers must be equal."
    )
    assert decoder_rank_generator.get_ranks("pp") == expert_decoder_rank_generator.get_ranks("pp"), (
        "Pipeline parallel groups must match between decoder and expert rank generators."
    )

    # -----------------------------------------------------------------------
    # DP + CP groups  (built first so SHARP works on the initial communicator)
    # -----------------------------------------------------------------------
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS
    global _DATA_PARALLEL_GROUP_WITH_CP
    global _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP
    global _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP
    global _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO
    assert _DATA_PARALLEL_GROUP is None, "data parallel group is already initialized"

    assert (
        data_parallel_size * context_parallel_size
    ) % num_distributed_optimizer_instances == 0, (
        "data_parallel_size * context_parallel_size must be divisible by "
        "num_distributed_optimizer_instances"
    )
    intra_partial_dp_size = (
        data_parallel_size * context_parallel_size
    ) // num_distributed_optimizer_instances

    if sharp_enabled_group == "dp":
        os.environ["NCCL_COLLNET_ENABLE"] = "1"

    for ranks_with_cp in decoder_rank_generator.get_ranks("dp-cp"):
        grp_with_cp = create_group(
            ranks_with_cp,
            timeout=timeout,
            pg_options=get_nccl_options("dp_cp", nccl_comm_cfgs),
            group_desc="DATA_PARALLEL_GROUP_WITH_CP",
        )
        if create_gloo_process_groups:
            grp_with_cp_gloo = create_group(
                ranks_with_cp,
                timeout=timeout,
                backend="gloo",
                group_desc="DATA_PARALLEL_GROUP_WITH_CP_GLOO",
            )
        else:
            grp_with_cp_gloo = None

        if rank in ranks_with_cp:
            _DATA_PARALLEL_GROUP_WITH_CP = grp_with_cp
            _DATA_PARALLEL_GROUP_WITH_CP_GLOO = grp_with_cp_gloo
            _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp

        if num_distributed_optimizer_instances > 1:
            for i in range(num_distributed_optimizer_instances):
                intra_ranks_with_cp = ranks_with_cp[
                    i * intra_partial_dp_size: (i + 1) * intra_partial_dp_size
                ]
                intra_grp = create_group(
                    intra_ranks_with_cp,
                    timeout=timeout,
                    pg_options=get_nccl_options("intra_dp_cp", nccl_comm_cfgs),
                    group_desc="INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP",
                )
                if create_gloo_process_groups:
                    intra_grp_gloo = create_group(
                        intra_ranks_with_cp,
                        timeout=timeout,
                        backend="gloo",
                        group_desc="INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO",
                    )
                else:
                    intra_grp_gloo = None
                if rank in intra_ranks_with_cp:
                    _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = intra_grp
                    _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO = intra_grp_gloo
        else:
            _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = _DATA_PARALLEL_GROUP_WITH_CP
            _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO = _DATA_PARALLEL_GROUP_WITH_CP_GLOO

    if sharp_enabled_group == "dp":
        if rank == 0:
            logger.info(
                "SHARP enabled for data-parallel group. The number of eligible process groups "
                "depends on the switch model (QM1 ≤ 8, QM2 ≤ 256). Communication falls back to "
                "non-SHARP if the limit is exceeded. Set #SBATCH_NETWORK=sharp in your job script."
            )
        torch.distributed.barrier(
            group=get_data_parallel_group(with_context_parallel=True),
            device_ids=[torch.cuda.current_device()],
        )
        torch.cuda.synchronize()
        if "NCCL_COLLNET_ENABLE" in os.environ:
            del os.environ["NCCL_COLLNET_ENABLE"]

    # Hybrid DP-CP groups
    if hybrid_context_parallel:
        global _HYBRID_DP_CP_GROUPS
        for ranks_with_cp in decoder_rank_generator.get_ranks("dp-cp"):
            assert len(ranks_with_cp) % 2 == 0, (
                "Hybrid context parallel requires an even number of ranks"
            )
            _HYBRID_DP_CP_GROUPS.update(
                create_hybrid_dp_cp_groups(
                    rank, ranks_with_cp, get_nccl_options("dp_cp", nccl_comm_cfgs)
                )
            )

    # DP (without CP)
    for ranks in decoder_rank_generator.get_ranks("dp"):
        grp = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("dp", nccl_comm_cfgs),
            group_desc="DATA_PARALLEL_GROUP",
        )
        if create_gloo_process_groups:
            grp_gloo = create_group(
                ranks, timeout=timeout, backend="gloo", group_desc="DATA_PARALLEL_GROUP_GLOO"
            )
        else:
            grp_gloo = None
        if rank in ranks:
            _DATA_PARALLEL_GROUP = grp
            _DATA_PARALLEL_GROUP_GLOO = grp_gloo
            _DATA_PARALLEL_GLOBAL_RANKS = ranks

    # Insight I2: independent AG/RS groups (Megatron M3102)
    # For each DP rank set, create two additional NCCL communicators:
    #   _ag: used for forward all-gather (parameter fetch before compute)
    #   _rs: used for backward reduce-scatter (gradient averaging after compute)
    # On PCIe fabrics without NVLink, a shared communicator forces AG and RS to
    # share NCCL scheduler bandwidth. Independent communicators allow the NCCL
    # runtime to pipeline them concurrently, reducing blocking on PCIe crossings.
    global _DATA_PARALLEL_GROUP_INDEPENDENT_AG
    global _DATA_PARALLEL_GROUP_INDEPENDENT_RS
    for ranks in decoder_rank_generator.get_ranks("dp"):
        grp_ag = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("dp_ag", nccl_comm_cfgs),
            group_desc="DATA_PARALLEL_GROUP_INDEPENDENT_AG",
        )
        grp_rs = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("dp_rs", nccl_comm_cfgs),
            group_desc="DATA_PARALLEL_GROUP_INDEPENDENT_RS",
        )
        if rank in ranks:
            _DATA_PARALLEL_GROUP_INDEPENDENT_AG = grp_ag
            _DATA_PARALLEL_GROUP_INDEPENDENT_RS = grp_rs

    # -----------------------------------------------------------------------
    # Context parallel groups
    # -----------------------------------------------------------------------
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_GLOBAL_RANKS
    global _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS
    assert _CONTEXT_PARALLEL_GROUP is None, "context parallel group is already initialized"

    for ranks in decoder_rank_generator.get_ranks("cp"):
        grp = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("cp", nccl_comm_cfgs),
            group_desc="CONTEXT_PARALLEL_GROUP",
        )
        if rank in ranks:
            _CONTEXT_PARALLEL_GROUP = grp
            _CONTEXT_PARALLEL_GLOBAL_RANKS = ranks

        if hierarchical_context_parallel_sizes:
            try:
                import numpy as np
                assert int(np.prod(hierarchical_context_parallel_sizes)) == context_parallel_size
            except ImportError:
                raise ImportError("numpy is required for hierarchical CP groups.")
            hier_groups, _ = create_hierarchical_groups(
                rank,
                ranks,
                hierarchical_context_parallel_sizes,
                create_gloo_process_groups=False,
                pg_options=get_nccl_options("hcp", nccl_comm_cfgs),
                timeout=timeout,
                group_desc="CONTEXT_PARALLEL_GROUP",
            )
            if rank in ranks:
                _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS = hier_groups

    # -----------------------------------------------------------------------
    # Model parallel group  (TP + PP combined)
    # -----------------------------------------------------------------------
    global _MODEL_PARALLEL_GROUP
    global _MODEL_PARALLEL_GLOBAL_RANKS
    assert _MODEL_PARALLEL_GROUP is None, "model parallel group is already initialized"

    for ranks in decoder_rank_generator.get_ranks("tp-pp"):
        grp = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("mp", nccl_comm_cfgs),
            group_desc="MODEL_PARALLEL_GROUP",
        )
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = grp
            _MODEL_PARALLEL_GLOBAL_RANKS = ranks

    # -----------------------------------------------------------------------
    # Tensor model parallel groups
    # -----------------------------------------------------------------------
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS
    assert _TENSOR_MODEL_PARALLEL_GROUP is None, (
        "tensor model parallel group is already initialized"
    )

    for ranks in decoder_rank_generator.get_ranks("tp"):
        grp = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("tp", nccl_comm_cfgs),
            group_desc="TENSOR_MODEL_PARALLEL_GROUP",
        )
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = grp
            _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = ranks

    # -----------------------------------------------------------------------
    # Pipeline model parallel groups  (+ embedding groups)
    # -----------------------------------------------------------------------
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    assert _PIPELINE_MODEL_PARALLEL_GROUP is None, (
        "pipeline model parallel group is already initialized"
    )
    assert _EMBEDDING_GROUP is None, "embedding group is already initialized"
    assert _POSITION_EMBEDDING_GROUP is None, "position embedding group is already initialized"

    # UCC backend setup
    if pipeline_model_parallel_comm_backend == "ucc":
        if "CUDA_DEVICE_MAX_CONNECTIONS" in os.environ:
            assert os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] != "1", (
                "UCC backend requires CUDA_DEVICE_MAX_CONNECTIONS > 1"
            )
        os.environ.setdefault("TORCH_UCC_BLOCKING_WAIT", "none")
        os.environ.setdefault("UCC_EC_CUDA_STREAM_TASK_MODE", "driver")
        os.environ.setdefault("UCX_TLS", "ib,cuda_copy")
        os.environ["NSYS_UCP_COMM_PARAMS"] = "1"
        os.environ["UCX_RNDV_THRESH"] = "0"
        os.environ["UCX_NET_DEVICES"] = "all"
        os.environ["UCC_CL_BASIC_TLS"] = "^sharp,nccl"

    for ranks in decoder_rank_generator.get_ranks("pp"):
        pp_pg_options = (
            None if pipeline_model_parallel_comm_backend == "ucc"
            else get_nccl_options("pp", nccl_comm_cfgs)
        )
        assert pipeline_model_parallel_comm_backend in (None, "nccl", "ucc"), (
            f'"{pipeline_model_parallel_comm_backend}" backend for PP is not supported'
        )
        grp = create_group(
            ranks,
            timeout=timeout,
            backend=pipeline_model_parallel_comm_backend,
            pg_options=pp_pg_options,
            group_desc="PIPELINE_MODEL_PARALLEL_GROUP",
        )
        if rank in ranks:
            if _PIPELINE_MODEL_PARALLEL_GROUP is None:
                _PIPELINE_MODEL_PARALLEL_GROUP = grp
                _PIPELINE_GLOBAL_RANKS = ranks
            elif isinstance(_PIPELINE_GLOBAL_RANKS[0], list):
                _PIPELINE_MODEL_PARALLEL_GROUP.append(grp)
                _PIPELINE_GLOBAL_RANKS.append(ranks)
            else:
                _PIPELINE_MODEL_PARALLEL_GROUP = [_PIPELINE_MODEL_PARALLEL_GROUP, grp]
                _PIPELINE_GLOBAL_RANKS = [_PIPELINE_GLOBAL_RANKS, ranks]

        emb_ranks = get_embedding_ranks(ranks)
        emb_grp = create_group(
            emb_ranks,
            timeout=timeout,
            pg_options=get_nccl_options("embd", nccl_comm_cfgs),
            group_desc="EMBEDDING_GROUP",
        )
        if rank in emb_ranks:
            _EMBEDDING_GROUP = emb_grp
            _EMBEDDING_GLOBAL_RANKS = emb_ranks

        pos_emb_ranks = get_position_embedding_ranks(ranks)
        pos_emb_grp = create_group(
            pos_emb_ranks,
            timeout=timeout,
            pg_options=get_nccl_options("pos_embd", nccl_comm_cfgs),
            group_desc="POSITION_EMBEDDING_GROUP",
        )
        if rank in pos_emb_ranks:
            _POSITION_EMBEDDING_GROUP = pos_emb_grp
            _POSITION_EMBEDDING_GLOBAL_RANKS = pos_emb_ranks

    # -----------------------------------------------------------------------
    # TP + DP combined groups (FP8 amax reduction)
    # -----------------------------------------------------------------------
    global _TENSOR_AND_DATA_PARALLEL_GROUP
    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    assert _TENSOR_AND_DATA_PARALLEL_GROUP is None, (
        "Tensor + data parallel group is already initialized"
    )

    for ranks in decoder_rank_generator.get_ranks("tp-dp-cp"):
        grp = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("tp_dp_cp", nccl_comm_cfgs),
            group_desc="TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP",
        )
        if rank in ranks:
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = grp

    for ranks in decoder_rank_generator.get_ranks("tp-dp"):
        grp = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("tp_dp", nccl_comm_cfgs),
            group_desc="TENSOR_AND_DATA_PARALLEL_GROUP",
        )
        if rank in ranks:
            _TENSOR_AND_DATA_PARALLEL_GROUP = grp

    # TP + CP combined
    global _TENSOR_AND_CONTEXT_PARALLEL_GROUP
    assert _TENSOR_AND_CONTEXT_PARALLEL_GROUP is None, (
        "Tensor + context parallel group is already initialized"
    )

    for ranks in decoder_rank_generator.get_ranks("tp-cp"):
        grp = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("tp_cp", nccl_comm_cfgs),
            group_desc="TENSOR_AND_CONTEXT_PARALLEL_GROUP",
        )
        if rank in ranks:
            _TENSOR_AND_CONTEXT_PARALLEL_GROUP = grp

    # -----------------------------------------------------------------------
    # Expert-parallel groups
    # -----------------------------------------------------------------------
    global _EXPERT_MODEL_PARALLEL_GROUP, _EXPERT_MODEL_PARALLEL_RANKS
    assert _EXPERT_MODEL_PARALLEL_GROUP is None, "Expert model parallel group is already initialized"

    for ranks in expert_decoder_rank_generator.get_ranks("ep"):
        grp = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("ep", nccl_comm_cfgs),
            group_desc="EXPERT_MODEL_PARALLEL_GROUP",
        )
        if rank in ranks:
            _EXPERT_MODEL_PARALLEL_GROUP = grp
            _EXPERT_MODEL_PARALLEL_RANKS = ranks

    global _EXPERT_TENSOR_PARALLEL_GROUP
    assert _EXPERT_TENSOR_PARALLEL_GROUP is None, (
        "Expert tensor model parallel group is already initialized"
    )

    for ranks in expert_decoder_rank_generator.get_ranks("tp"):
        grp = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("ep_tp", nccl_comm_cfgs),
            group_desc="EXPERT_TENSOR_PARALLEL_GROUP",
        )
        if rank in ranks:
            _EXPERT_TENSOR_PARALLEL_GROUP = grp

    global _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP
    assert _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP is None, (
        "Expert tensor + model parallel group is already initialized"
    )

    for ranks in expert_decoder_rank_generator.get_ranks("tp-ep"):
        grp = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("tp_ep_mp", nccl_comm_cfgs),
            group_desc="EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP",
        )
        if rank in ranks:
            _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = grp

    global _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP
    assert _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP is None, (
        "Expert tensor + model + pipeline parallel group is already initialized"
    )

    for ranks in expert_decoder_rank_generator.get_ranks("tp-ep-pp"):
        grp = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("tp_ep_pp", nccl_comm_cfgs),
            group_desc="EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP",
        )
        if rank in ranks:
            _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP = grp

    global _EXPERT_DATA_PARALLEL_GROUP
    global _EXPERT_DATA_PARALLEL_GROUP_GLOO
    global _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP
    global _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO
    global _INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP
    assert _EXPERT_DATA_PARALLEL_GROUP is None, "Expert data group is already initialized"
    assert _EXPERT_DATA_PARALLEL_GROUP_GLOO is None, "Expert data group-gloo is already initialized"
    assert _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP is None, (
        "Intra partial expert data group is already initialized"
    )
    assert _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO is None, (
        "Intra partial expert data group-gloo is already initialized"
    )
    assert _INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP is None, (
        "Inter partial expert data group is already initialized"
    )

    assert expert_data_parallel_size % num_distributed_optimizer_instances == 0, (
        "expert_data_parallel_size must be divisible by num_distributed_optimizer_instances"
    )
    intra_partial_expert_dp_size = (
        expert_data_parallel_size // num_distributed_optimizer_instances
    )

    for ranks in expert_decoder_rank_generator.get_ranks("dp"):
        grp = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("ep_dp", nccl_comm_cfgs),
            group_desc="EXPERT_DATA_PARALLEL_GROUP",
        )
        if create_gloo_process_groups:
            grp_gloo = create_group(
                ranks, backend="gloo", group_desc="EXPERT_DATA_PARALLEL_GROUP_GLOO"
            )
        else:
            grp_gloo = None
        if rank in ranks:
            _EXPERT_DATA_PARALLEL_GROUP = grp
            _EXPERT_DATA_PARALLEL_GROUP_GLOO = grp_gloo

        if num_distributed_optimizer_instances > 1:
            if sharp_enabled_group == "dp_replica":
                os.environ["NCCL_COLLNET_ENABLE"] = "1"
            hier_groups, hier_groups_gloo = create_hierarchical_groups(
                rank,
                ranks,
                [intra_partial_expert_dp_size, num_distributed_optimizer_instances],
                create_gloo_process_groups=create_gloo_process_groups,
                pg_options=[
                    get_nccl_options("intra_ep_dp", nccl_comm_cfgs),
                    get_nccl_options("inter_ep_dp", nccl_comm_cfgs),
                ],
                timeout=timeout,
                group_desc="EXPERT_DATA_PARALLEL_GROUP",
            )
            if rank in ranks:
                _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = hier_groups[0]
                _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO = hier_groups_gloo[0]
                _INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = hier_groups[1]

            if sharp_enabled_group == "dp_replica":
                if _INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP is not None:
                    torch.distributed.barrier(
                        group=_INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP,
                        device_ids=[torch.cuda.current_device()],
                    )
                    torch.cuda.synchronize()
                if "NCCL_COLLNET_ENABLE" in os.environ:
                    del os.environ["NCCL_COLLNET_ENABLE"]
        else:
            _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = _EXPERT_DATA_PARALLEL_GROUP
            _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO = _EXPERT_DATA_PARALLEL_GROUP_GLOO

    # -----------------------------------------------------------------------
    # Intra distributed-optimizer instance groups
    # -----------------------------------------------------------------------
    global _INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP
    assert _INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP is None, (
        "Intra distributed optimizer instance group is already initialized"
    )

    mp_group_id = 0
    intra_dist_opt_ranks: List[int] = []
    for ranks in expert_decoder_rank_generator.get_ranks("tp-ep-pp"):
        mp_group_id += 1
        intra_dist_opt_ranks.extend(ranks)
        if mp_group_id % intra_partial_expert_dp_size == 0:
            inst_grp = create_group(
                intra_dist_opt_ranks,
                timeout=timeout,
                pg_options=get_nccl_options("intra_dist_opt_instance", nccl_comm_cfgs),
                group_desc="INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP",
            )
            if rank in intra_dist_opt_ranks:
                _INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP = inst_grp
            intra_dist_opt_ranks = []

    # -----------------------------------------------------------------------
    # DES-LOC tier groups  (heterogeneous GPU grouping by TierType)
    #
    # NCCL requires ALL ranks to call new_group() with the same ranks list,
    # even ranks that are not members.  We iterate every tier unconditionally.
    # -----------------------------------------------------------------------
    global _TIER_GROUPS
    global _LOCAL_TIER
    if desloc_config is not None and desloc_config.tiers:
        # Merge gpu_indices by tier_type name (handles duplicate TierSpec entries)
        tier_indices_map: Dict[str, List[int]] = {}
        tier_spec_map: Dict[str, TierSpec] = {}
        for tier_spec in desloc_config.tiers:
            tier_name = tier_spec.tier_type.name.lower()
            if tier_name not in tier_indices_map:
                tier_indices_map[tier_name] = []
                tier_spec_map[tier_name] = tier_spec
            for idx in tier_spec.gpu_indices:
                if idx not in tier_indices_map[tier_name]:
                    tier_indices_map[tier_name].append(idx)

        for tier_name, gpu_indices in tier_indices_map.items():
            if not gpu_indices:
                continue
            sorted_indices = sorted(gpu_indices)
            tier_grp = create_group(
                sorted_indices,
                timeout=timeout,
                group_desc=f"DES_LOC_TIER_{tier_name.upper()}",
            )
            _TIER_GROUPS[tier_name] = tier_grp
            if rank in sorted_indices:
                _LOCAL_TIER = tier_spec_map[tier_name]

    # -----------------------------------------------------------------------
    # Global memory buffer
    # -----------------------------------------------------------------------
    _set_global_memory_buffer()
    _configure_nccl_flight_recorder(config)


# ---------------------------------------------------------------------------
# create_all_gather_groups  (utility for AG/RS overlap)
# ---------------------------------------------------------------------------

def create_all_gather_groups(
    for_expert_parallelism: bool = False,
    timeout: Optional[timedelta] = None,
    nccl_comm_cfgs: Optional[dict] = None,
):
    """Create separate all-gather communicators with the same DP ranks.

    Used to overlap all-gather with reduce-scatter when using the
    distributed optimizer.

    Returns:
        (dp_cp_ag_group, expt_dp_ag_group)
    """
    if not is_initialized():
        raise RuntimeError(
            "create_all_gather_groups() requires parallel state to be initialized first."
        )
    if nccl_comm_cfgs is None:
        nccl_comm_cfgs = {}

    rank = torch.distributed.get_rank()
    pp_size = get_pipeline_model_parallel_world_size()
    cp_size = get_context_parallel_world_size()
    tp_size = get_tensor_model_parallel_world_size()
    ep_size = get_expert_model_parallel_world_size()
    dp_size = get_data_parallel_world_size()

    decoder_rank_gen = RankGenerator(
        tp=tp_size, ep=1, dp=dp_size, pp=pp_size, cp=cp_size,
        order="tp-cp-ep-dp-pp", rank_offset=0,
    )

    dp_cp_ag_group = None
    for ranks_with_cp in decoder_rank_gen.get_ranks("dp-cp"):
        grp = create_group(
            ranks_with_cp,
            timeout=timeout,
            pg_options=get_nccl_options("dp_cp", nccl_comm_cfgs),
            group_desc="DATA_PARALLEL_GROUP_WITH_CP_AG",
        )
        if rank in ranks_with_cp:
            dp_cp_ag_group = grp

    expt_dp_ag_group = None
    if for_expert_parallelism and ep_size > 1:
        expert_tp_size = get_expert_tensor_parallel_world_size()
        expert_dp_size = get_expert_data_parallel_world_size()
        expert_rank_gen = RankGenerator(
            tp=expert_tp_size, ep=ep_size, dp=expert_dp_size, pp=pp_size, cp=1,
            order="tp-cp-ep-dp-pp", rank_offset=0,
        )
        for expert_dp_ranks in expert_rank_gen.get_ranks("dp"):
            grp = create_group(
                expert_dp_ranks,
                timeout=timeout,
                pg_options=get_nccl_options("ep_dp", nccl_comm_cfgs),
                group_desc="EXPERT_DATA_PARALLEL_GROUP_AG",
            )
            if rank in expert_dp_ranks:
                expt_dp_ag_group = grp

    return dp_cp_ag_group, expt_dp_ag_group


# ---------------------------------------------------------------------------
# Query helpers: is_initialized
# ---------------------------------------------------------------------------

def is_initialized() -> bool:
    """Return True if model-parallel groups have been initialized."""
    return _DATA_PARALLEL_GROUP is not None


def model_parallel_is_initialized() -> bool:
    """Return True if TP, PP, and DP groups are all initialized."""
    return (
        _TENSOR_MODEL_PARALLEL_GROUP is not None
        and _PIPELINE_MODEL_PARALLEL_GROUP is not None
        and _DATA_PARALLEL_GROUP is not None
    )


# ---------------------------------------------------------------------------
# Group accessors
# ---------------------------------------------------------------------------

def get_model_parallel_group(check_initialized: bool = True) -> Optional[torch.distributed.ProcessGroup]:
    """Get the model-parallel group (TP + PP combined) the caller belongs to."""
    if check_initialized:
        assert _MODEL_PARALLEL_GROUP is not None, "model parallel group is not initialized"
    return _MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_group(check_initialized: bool = True) -> Optional[torch.distributed.ProcessGroup]:
    """Get the tensor-model-parallel group the caller belongs to."""
    if check_initialized:
        assert _TENSOR_MODEL_PARALLEL_GROUP is not None, (
            "tensor model parallel group is not initialized"
        )
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group(check_initialized: bool = True):
    """Get the pipeline-model-parallel group the caller belongs to."""
    if check_initialized:
        assert _PIPELINE_MODEL_PARALLEL_GROUP is not None, (
            "pipeline model parallel group is not initialized"
        )
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_data_parallel_group(
    with_context_parallel: bool = False,
    partial_data_parallel: bool = False,
    independent_all_gather: bool = False,
) -> torch.distributed.ProcessGroup:
    """Get the data-parallel group the caller belongs to.

    Args:
        with_context_parallel: Return the DP+CP combined group.
        partial_data_parallel: Return the intra-partial DP group (DistOpt sharding).
        independent_all_gather: Insight I2: independent AG/RS groups (Megatron M3102).
            When True, return the dedicated all-gather communicator instead of
            the main DP group. On PCIe topologies this separate NCCL channel
            allows forward AG and backward RS to run concurrently without
            head-of-line blocking on a shared communicator queue.
            Only meaningful when with_context_parallel=False and
            partial_data_parallel=False (base DP group path).
    """
    # Insight I2: independent AG/RS groups (Megatron M3102)
    # Callers performing forward all-gather (e.g. ZeRO-3 param gather) should
    # pass independent_all_gather=True to use the dedicated AG communicator.
    if independent_all_gather and not with_context_parallel and not partial_data_parallel:
        assert _DATA_PARALLEL_GROUP_INDEPENDENT_AG is not None, (
            "Independent all-gather DP group is not initialized. "
            "Call initialize_model_parallel() before using this path."
        )
        return _DATA_PARALLEL_GROUP_INDEPENDENT_AG
    if with_context_parallel:
        if partial_data_parallel:
            assert _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP is not None, (
                "Intra partial data parallel group is not initialized"
            )
            return _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP
        assert _DATA_PARALLEL_GROUP_WITH_CP is not None, (
            "data parallel group with context parallel is not initialized"
        )
        return _DATA_PARALLEL_GROUP_WITH_CP
    else:
        assert _DATA_PARALLEL_GROUP is not None, "data parallel group is not initialized"
        assert not partial_data_parallel, "Partial DP for Optimizer needs to include CP"
        return _DATA_PARALLEL_GROUP


def get_data_parallel_group_independent_rs() -> torch.distributed.ProcessGroup:
    """Insight I2: independent AG/RS groups (Megatron M3102).

    Return the dedicated reduce-scatter communicator for the backward pass.
    Callers performing backward reduce-scatter (e.g. ZeRO-3 grad averaging)
    should use this instead of get_data_parallel_group() to allow the NCCL
    scheduler to pipeline RS with any concurrent AG on the AG channel.
    On PCIe fabrics without NVLink this can improve effective bandwidth by
    decoupling the two collective directions onto independent NCCL queues.
    """
    # Insight I2: independent AG/RS groups (Megatron M3102)
    assert _DATA_PARALLEL_GROUP_INDEPENDENT_RS is not None, (
        "Independent reduce-scatter DP group is not initialized. "
        "Call initialize_model_parallel() before using this path."
    )
    return _DATA_PARALLEL_GROUP_INDEPENDENT_RS


def get_data_parallel_group_gloo(
    with_context_parallel: bool = False,
    partial_data_parallel: bool = False,
) -> torch.distributed.ProcessGroup:
    """Get the Gloo data-parallel group the caller belongs to."""
    if with_context_parallel:
        if partial_data_parallel:
            assert _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO is not None, (
                "Intra partial data parallel group-gloo is not initialized"
            )
            return _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO
        assert _DATA_PARALLEL_GROUP_WITH_CP_GLOO is not None, (
            "data parallel group-gloo with context parallel is not initialized"
        )
        return _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    else:
        assert _DATA_PARALLEL_GROUP_GLOO is not None, "data parallel group-gloo is not initialized"
        assert not partial_data_parallel, "Partial DP for Optimizer needs to include CP"
        return _DATA_PARALLEL_GROUP_GLOO


def get_context_parallel_group(check_initialized: bool = True) -> Optional[torch.distributed.ProcessGroup]:
    """Get the context-parallel group the caller belongs to."""
    if check_initialized:
        assert _CONTEXT_PARALLEL_GROUP is not None, "context parallel group is not initialized"
    return _CONTEXT_PARALLEL_GROUP


def get_context_parallel_global_ranks(check_initialized: bool = True) -> Optional[List[int]]:
    """Get all global ranks of the context-parallel group the caller belongs to."""
    if check_initialized:
        assert _CONTEXT_PARALLEL_GLOBAL_RANKS is not None, (
            "context parallel group is not initialized"
        )
    return _CONTEXT_PARALLEL_GLOBAL_RANKS


def get_hierarchical_context_parallel_groups(
    check_initialized: bool = True,
) -> Optional[List[torch.distributed.ProcessGroup]]:
    """Get hierarchical context-parallel sub-groups."""
    if check_initialized:
        assert _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS is not None, (
            "hierarchical context parallel groups are not initialized"
        )
    return _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS


def get_hybrid_data_context_parallel_groups(
    check_initialized: bool = True,
    group_size: Optional[int] = None,
) -> Optional[torch.distributed.ProcessGroup]:
    """Get hybrid DP×CP group of the requested size."""
    if get_data_parallel_world_size(with_context_parallel=True) == group_size:
        if check_initialized:
            assert _DATA_PARALLEL_GROUP_WITH_CP is not None
        return _DATA_PARALLEL_GROUP_WITH_CP
    if check_initialized:
        assert _HYBRID_DP_CP_GROUPS is not None
    return _HYBRID_DP_CP_GROUPS[group_size]


def get_embedding_group(check_initialized: bool = True) -> Optional[torch.distributed.ProcessGroup]:
    """Get the embedding group the caller belongs to."""
    if check_initialized:
        assert _EMBEDDING_GROUP is not None, "embedding group is not initialized"
    return _EMBEDDING_GROUP


def get_position_embedding_group(check_initialized: bool = True) -> Optional[torch.distributed.ProcessGroup]:
    """Get the position-embedding group the caller belongs to."""
    if check_initialized:
        assert _POSITION_EMBEDDING_GROUP is not None, "position embedding group is not initialized"
    return _POSITION_EMBEDDING_GROUP


def get_amax_reduction_group(
    with_context_parallel: bool = False,
    tp_only_amax_red: bool = False,
) -> torch.distributed.ProcessGroup:
    """Get the FP8 amax reduction group."""
    if with_context_parallel:
        if not tp_only_amax_red:
            assert _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP is not None, (
                "FP8 amax reduction group (TP+DP+CP) is not initialized"
            )
            return _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
        else:
            assert _TENSOR_AND_CONTEXT_PARALLEL_GROUP is not None, (
                "FP8 amax reduction group (TP+CP) is not initialized"
            )
            return _TENSOR_AND_CONTEXT_PARALLEL_GROUP
    else:
        if not tp_only_amax_red:
            assert _TENSOR_AND_DATA_PARALLEL_GROUP is not None, (
                "FP8 amax reduction group (TP+DP) is not initialized"
            )
            return _TENSOR_AND_DATA_PARALLEL_GROUP
        else:
            assert _TENSOR_MODEL_PARALLEL_GROUP is not None, (
                "FP8 amax reduction group (TP) is not initialized"
            )
            return _TENSOR_MODEL_PARALLEL_GROUP


def get_tensor_and_data_parallel_group(
    check_initialized: bool = True,
    with_context_parallel: bool = False,
) -> Optional[torch.distributed.ProcessGroup]:
    """Get the tensor- and data-parallel group."""
    if with_context_parallel:
        if check_initialized:
            assert _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP is not None, (
                "tensor and data parallel group (with CP) is not initialized"
            )
        return _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    else:
        if check_initialized:
            assert _TENSOR_AND_DATA_PARALLEL_GROUP is not None, (
                "tensor and data parallel group is not initialized"
            )
        return _TENSOR_AND_DATA_PARALLEL_GROUP


def get_tensor_and_context_parallel_group(check_initialized: bool = True) -> Optional[torch.distributed.ProcessGroup]:
    """Get the tensor- and context-parallel group."""
    if check_initialized:
        assert _TENSOR_AND_CONTEXT_PARALLEL_GROUP is not None, (
            "tensor and context parallel group is not initialized"
        )
    return _TENSOR_AND_CONTEXT_PARALLEL_GROUP


# ---------------------------------------------------------------------------
# set_xxx / get_xxx for world sizes and ranks (MPU override support)
# ---------------------------------------------------------------------------

def set_tensor_model_parallel_world_size(world_size: int) -> None:
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_pipeline_model_parallel_world_size(world_size: int) -> None:
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_virtual_pipeline_model_parallel_world_size(world_size: int) -> None:
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_tensor_model_parallel_world_size() -> int:
    """Return the tensor-model-parallel world size."""
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return get_tensor_model_parallel_group().size()


def get_pipeline_model_parallel_world_size() -> int:
    """Return the pipeline-model-parallel world size."""
    if _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return get_pipeline_model_parallel_group().size()


def set_tensor_model_parallel_rank(rank: int) -> None:
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_rank(rank: int) -> None:
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_tensor_model_parallel_rank() -> int:
    """Return the caller's rank in the tensor-model-parallel group."""
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return get_tensor_model_parallel_group().rank()


def get_pipeline_model_parallel_rank() -> int:
    """Return the caller's rank in the pipeline-model-parallel group."""
    if _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())


def is_pipeline_first_stage(ignore_virtual: bool = True, vp_stage: Optional[int] = None) -> bool:
    """Return True if the caller is pipeline stage 0."""
    if not ignore_virtual and get_virtual_pipeline_model_parallel_world_size() is not None:
        assert vp_stage is not None, "vp_stage must be specified when virtual pipeline is enabled"
        if vp_stage != 0:
            return False
    return get_pipeline_model_parallel_rank() == 0


def is_pipeline_last_stage(ignore_virtual: bool = True, vp_stage: Optional[int] = None) -> bool:
    """Return True if the caller is the last pipeline stage."""
    if not ignore_virtual and get_virtual_pipeline_model_parallel_world_size() is not None:
        assert vp_stage is not None, "vp_stage must be specified when virtual pipeline is enabled"
        if vp_stage != (get_virtual_pipeline_model_parallel_world_size() - 1):
            return False
    return get_pipeline_model_parallel_rank() == (get_pipeline_model_parallel_world_size() - 1)


def is_rank_in_embedding_group(ignore_virtual: bool = True, vp_stage: Optional[int] = None) -> bool:
    """Return True if the caller's rank is in the embedding group."""
    rank = torch.distributed.get_rank()
    if _EMBEDDING_GLOBAL_RANKS is None:
        return False
    if ignore_virtual:
        return rank in _EMBEDDING_GLOBAL_RANKS
    if rank in _EMBEDDING_GLOBAL_RANKS:
        if rank == _EMBEDDING_GLOBAL_RANKS[0]:
            return is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage)
        elif rank == _EMBEDDING_GLOBAL_RANKS[-1]:
            return is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)
        else:
            return True
    return False


def is_rank_in_position_embedding_group() -> bool:
    """Return True if the caller's rank is in the position-embedding group."""
    rank = torch.distributed.get_rank()
    return (
        _POSITION_EMBEDDING_GLOBAL_RANKS is not None
        and rank in _POSITION_EMBEDDING_GLOBAL_RANKS
    )


def get_virtual_pipeline_model_parallel_rank() -> Optional[int]:
    """Return the virtual pipeline-parallel rank."""
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK


def set_virtual_pipeline_model_parallel_rank(rank: int) -> None:
    """Set the virtual pipeline-parallel rank (deprecated: pass vp_stage explicitly)."""
    warnings.warn(
        "set_virtual_pipeline_model_parallel_rank is deprecated. "
        "Pass vp_stage explicitly to is_pipeline_first_stage / is_pipeline_last_stage.",
        DeprecationWarning,
        stacklevel=2,
    )
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_virtual_pipeline_model_parallel_world_size() -> Optional[int]:
    """Return the virtual pipeline-parallel world size."""
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE


# ---------------------------------------------------------------------------
# src-rank helpers (for weight broadcast)
# ---------------------------------------------------------------------------

def get_tensor_model_parallel_src_rank() -> int:
    """Return the global rank of the first member in the caller's TP group."""
    assert _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS is not None, (
        "Tensor model parallel group is not initialized"
    )
    return _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS[0]


def get_model_parallel_src_rank() -> int:
    """Return the global rank of the first member in the caller's model-parallel group."""
    assert _MODEL_PARALLEL_GLOBAL_RANKS is not None, "Model parallel group is not initialized"
    return _MODEL_PARALLEL_GLOBAL_RANKS[0]


def get_data_parallel_src_rank(with_context_parallel: bool = False) -> int:
    """Return the global rank of the first member in the caller's DP group."""
    if with_context_parallel:
        assert _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP is not None, (
            "Data parallel group with context parallel is not initialized"
        )
        return _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP[0]
    assert _DATA_PARALLEL_GLOBAL_RANKS is not None, "Data parallel group is not initialized"
    return _DATA_PARALLEL_GLOBAL_RANKS[0]


# ---------------------------------------------------------------------------
# Pipeline rank helpers
# ---------------------------------------------------------------------------

def get_pipeline_model_parallel_first_rank() -> int:
    """Return the global rank of the first pipeline stage."""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    ranks = _PIPELINE_GLOBAL_RANKS
    if isinstance(ranks[0], list):
        ranks = ranks[0]
    return ranks[0]


def get_pipeline_model_parallel_last_rank() -> int:
    """Return the global rank of the last pipeline stage."""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    ranks = _PIPELINE_GLOBAL_RANKS
    if isinstance(ranks[0], list):
        ranks = ranks[0]
    last_local = get_pipeline_model_parallel_world_size() - 1
    return ranks[last_local]


def get_pipeline_model_parallel_next_rank() -> int:
    """Return the global rank of the next pipeline stage."""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    ranks = _PIPELINE_GLOBAL_RANKS
    if isinstance(ranks[0], list):
        ranks = ranks[0]
    rank_in_pp = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return ranks[(rank_in_pp + 1) % world_size]


def get_pipeline_model_parallel_prev_rank() -> int:
    """Return the global rank of the previous pipeline stage."""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    ranks = _PIPELINE_GLOBAL_RANKS
    if isinstance(ranks[0], list):
        ranks = ranks[0]
    rank_in_pp = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return ranks[(rank_in_pp - 1) % world_size]


# ---------------------------------------------------------------------------
# DP world size / rank
# ---------------------------------------------------------------------------

def set_data_parallel_rank(rank: int) -> None:
    global _MPU_DATA_PARALLEL_RANK
    _MPU_DATA_PARALLEL_RANK = rank


def get_data_parallel_world_size(
    with_context_parallel: bool = False,
    partial_data_parallel: bool = False,
) -> int:
    """Return the data-parallel world size."""
    if _MPU_DATA_PARALLEL_WORLD_SIZE is not None:
        return _MPU_DATA_PARALLEL_WORLD_SIZE
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_data_parallel_group(
            with_context_parallel=with_context_parallel,
            partial_data_parallel=partial_data_parallel,
        ).size()
    return 0


def get_data_parallel_rank(
    with_context_parallel: bool = False,
    partial_data_parallel: bool = False,
) -> int:
    """Return the caller's rank in the data-parallel group."""
    if _MPU_DATA_PARALLEL_RANK is not None:
        return _MPU_DATA_PARALLEL_RANK
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_data_parallel_group(
            with_context_parallel=with_context_parallel,
            partial_data_parallel=partial_data_parallel,
        ).rank()
    return 0


# ---------------------------------------------------------------------------
# CP world size / rank
# ---------------------------------------------------------------------------

def get_context_parallel_world_size() -> int:
    """Return the context-parallel world size."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_context_parallel_group().size()
    return 0


def get_context_parallel_rank() -> int:
    """Return the caller's rank in the context-parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_context_parallel_group().rank()
    return 0


# ---------------------------------------------------------------------------
# TP + CP combined world size / rank
# ---------------------------------------------------------------------------

def get_tensor_and_context_parallel_world_size() -> int:
    """Return world size for the TP + CP combined group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_tensor_and_context_parallel_group().size()
    return 0


def get_tensor_and_context_parallel_rank() -> int:
    """Return the caller's rank in the TP + CP combined group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_tensor_and_context_parallel_group().rank()
    return 0


# ---------------------------------------------------------------------------
# Expert parallel accessors
# ---------------------------------------------------------------------------

def get_expert_model_parallel_group(check_initialized: bool = True) -> Optional[torch.distributed.ProcessGroup]:
    """Get the expert-model-parallel group."""
    if check_initialized:
        assert _EXPERT_MODEL_PARALLEL_GROUP is not None, (
            "expert model parallel group is not initialized"
        )
    return _EXPERT_MODEL_PARALLEL_GROUP


def get_expert_model_parallel_src_rank() -> int:
    """Return the first global rank in the caller's expert-model-parallel group."""
    assert _EXPERT_MODEL_PARALLEL_RANKS is not None, (
        "Expert model parallel group is not initialized"
    )
    return _EXPERT_MODEL_PARALLEL_RANKS[0]


def get_expert_model_parallel_world_size() -> int:
    """Return the expert-model-parallel world size."""
    if _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_expert_model_parallel_group().size()
    return 0


def set_expert_model_parallel_world_size(world_size: int) -> None:
    global _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_expert_model_parallel_rank() -> int:
    """Return the caller's rank in the expert-model-parallel group."""
    if _MPU_EXPERT_MODEL_PARALLEL_RANK is not None:
        return _MPU_EXPERT_MODEL_PARALLEL_RANK
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_expert_model_parallel_group().rank()
    return 0


def set_expert_model_parallel_rank(rank: int) -> None:
    global _MPU_EXPERT_MODEL_PARALLEL_RANK
    _MPU_EXPERT_MODEL_PARALLEL_RANK = rank


def get_expert_tensor_parallel_group(check_initialized: bool = True) -> Optional[torch.distributed.ProcessGroup]:
    """Get the expert-tensor-parallel group."""
    if check_initialized:
        assert _EXPERT_TENSOR_PARALLEL_GROUP is not None, (
            "Expert tensor parallel group is not initialized"
        )
    return _EXPERT_TENSOR_PARALLEL_GROUP


def get_expert_tensor_parallel_world_size() -> int:
    """Return the expert-tensor-parallel world size."""
    if _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE is not None:
        return _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE
    if not _EXPERT_TENSOR_PARALLEL_GROUP:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE or 1
    return get_expert_tensor_parallel_group().size()


def set_expert_tensor_parallel_world_size(world_size: int) -> None:
    global _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE = world_size


def get_expert_tensor_parallel_rank() -> int:
    """Return the caller's rank in the expert-tensor-parallel group."""
    if _MPU_EXPERT_TENSOR_PARALLEL_RANK is not None:
        return _MPU_EXPERT_TENSOR_PARALLEL_RANK
    if not _EXPERT_TENSOR_PARALLEL_GROUP:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK or 0
    return get_expert_tensor_parallel_group().rank()


def set_expert_tensor_parallel_rank(rank: int) -> None:
    global _MPU_EXPERT_TENSOR_PARALLEL_RANK
    _MPU_EXPERT_TENSOR_PARALLEL_RANK = rank


def get_expert_tensor_and_model_parallel_group(check_initialized: bool = True) -> Optional[torch.distributed.ProcessGroup]:
    """Get the expert tensor + model combined group."""
    if check_initialized:
        assert _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP is not None, (
            "Expert tensor and model parallel group is not initialized"
        )
    return _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP


def get_expert_tensor_and_model_parallel_world_size() -> int:
    """Return world size for the expert tensor + model combined group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_expert_tensor_and_model_parallel_group().size()
    return 0


def get_expert_tensor_and_model_parallel_rank() -> int:
    """Return the caller's rank in the expert tensor + model combined group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_expert_tensor_and_model_parallel_group().rank()
    return 0


def get_expert_tensor_model_pipeline_parallel_group(check_initialized: bool = True) -> Optional[torch.distributed.ProcessGroup]:
    """Get the expert TP + EP + PP combined group."""
    if check_initialized:
        assert _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP is not None, (
            "Expert tensor-model-pipeline parallel group is not initialized"
        )
    return _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP


def get_expert_data_parallel_group(
    check_initialized: bool = True,
    partial_expert_data_parallel: bool = False,
) -> Optional[torch.distributed.ProcessGroup]:
    """Get the expert-data-parallel group."""
    if partial_expert_data_parallel:
        if check_initialized:
            assert _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP is not None, (
                "Intra partial expert data parallel group is not initialized"
            )
        return _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP
    else:
        if check_initialized:
            assert _EXPERT_DATA_PARALLEL_GROUP is not None, (
                "Expert data parallel group is not initialized"
            )
        return _EXPERT_DATA_PARALLEL_GROUP


def get_expert_data_parallel_group_gloo(
    partial_expert_data_parallel: bool = False,
) -> Optional[torch.distributed.ProcessGroup]:
    """Get the Gloo expert-data-parallel group."""
    if partial_expert_data_parallel:
        assert _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO is not None, (
            "Intra partial expert data parallel group-gloo is not initialized"
        )
        return _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO
    assert _EXPERT_DATA_PARALLEL_GROUP_GLOO is not None, (
        "Expert data parallel group-gloo is not initialized"
    )
    return _EXPERT_DATA_PARALLEL_GROUP_GLOO


def get_expert_data_parallel_rank(partial_expert_data_parallel: bool = False) -> int:
    """Return the caller's rank in the expert-data-parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_expert_data_parallel_group(
            partial_expert_data_parallel=partial_expert_data_parallel
        ).rank()
    return 0


def get_expert_data_parallel_world_size(partial_expert_data_parallel: bool = False) -> int:
    """Return world size for the expert-data-parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return get_expert_data_parallel_group(
            partial_expert_data_parallel=partial_expert_data_parallel
        ).size()
    return 0


def get_intra_distributed_optimizer_instance_group(check_initialized: bool = True) -> Optional[torch.distributed.ProcessGroup]:
    """Get the group of all GPUs within a distributed optimizer instance."""
    if check_initialized:
        assert _INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP is not None, (
            "Intra distributed optimizer instance group is not initialized"
        )
    return _INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP


def get_inter_distributed_optimizer_instance_group(check_initialized: bool = True) -> Optional[torch.distributed.ProcessGroup]:
    """Get the group spanning different distributed optimizer instances.

    Attention and MLP/Expert layers share the same inter-instance group,
    implemented as _INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP.
    """
    if check_initialized:
        assert _INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP is not None, (
            "Inter distributed optimizer instance group is not initialized"
        )
    return _INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP


# ---------------------------------------------------------------------------
# DES-LOC tier group accessors
# ---------------------------------------------------------------------------

def get_tier_group(tier_name: str) -> Optional[torch.distributed.ProcessGroup]:
    """Get the process group for a DES-LOC tier (e.g. 'datacenter', 'professional').

    Returns None if no group was created for the given tier name.

    Example::

        tier_grp = get_tier_group('datacenter')
        if tier_grp is not None:
            dist.all_reduce(tensor, group=tier_grp)
    """
    return _TIER_GROUPS.get(tier_name, None)


def get_all_tier_groups() -> Dict[str, torch.distributed.ProcessGroup]:
    """Return a shallow copy of the tier groups dictionary."""
    return dict(_TIER_GROUPS)


def get_local_tier() -> Optional[TierSpec]:
    """Get the :class:`~deepspeed.core.desloc_config.TierSpec` for the current rank's GPU.

    Returns None if DES-LOC was not configured or this rank has no tier assignment.
    """
    return _LOCAL_TIER


# ---------------------------------------------------------------------------
# Global memory buffer
# ---------------------------------------------------------------------------

def _set_global_memory_buffer() -> None:
    global _GLOBAL_MEMORY_BUFFER
    assert _GLOBAL_MEMORY_BUFFER is None, "global memory buffer is already initialized"
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()


def get_global_memory_buffer() -> GlobalMemoryBuffer:
    """Return the global reusable memory buffer."""
    assert _GLOBAL_MEMORY_BUFFER is not None, "global memory buffer is not initialized"
    return _GLOBAL_MEMORY_BUFFER


def destroy_global_memory_buffer() -> None:
    """Release the global memory buffer."""
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None


# ---------------------------------------------------------------------------
# Diagnostic helper
# ---------------------------------------------------------------------------

def get_all_ranks() -> str:
    """Return a string encoding the caller's rank in each parallel dimension."""
    ranks = [
        get_tensor_model_parallel_rank(),
        get_data_parallel_rank(),
        get_context_parallel_rank(),
        get_pipeline_model_parallel_rank(),
        get_expert_model_parallel_rank(),
    ]
    return "_".join(str(r or 0) for r in ranks)


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------

def destroy_model_parallel() -> None:
    """Reset all model-parallel process groups and module-level state.

    Does NOT call torch.distributed.destroy_process_group on NCCL groups
    (that can cause hangs if not all ranks call it simultaneously).  Gloo
    groups are destroyed because they are CPU-only and safe to destroy locally.
    """

    def _safe_destroy_gloo(grp: Optional[torch.distributed.ProcessGroup]) -> None:
        """Destroy a Gloo group if it is still registered."""
        if grp is None:
            return
        if not torch.distributed.is_initialized():
            return
        pg_map = getattr(
            torch.distributed.distributed_c10d,
            "_world",
            None,
        )
        if pg_map is not None:
            pg_map = getattr(pg_map, "pg_map", {})
            if pg_map.get(grp, None) is not None:
                try:
                    torch.distributed.destroy_process_group(grp)
                except Exception:
                    pass

    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _MODEL_PARALLEL_GLOBAL_RANKS
    _MODEL_PARALLEL_GLOBAL_RANKS = None

    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS
    _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = None

    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_GLOBAL_RANKS
    _PIPELINE_GLOBAL_RANKS = None

    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GLOBAL_RANKS
    _DATA_PARALLEL_GLOBAL_RANKS = None

    global _DATA_PARALLEL_GROUP_GLOO
    _safe_destroy_gloo(_DATA_PARALLEL_GROUP_GLOO)
    _DATA_PARALLEL_GROUP_GLOO = None

    global _DATA_PARALLEL_GROUP_WITH_CP
    _DATA_PARALLEL_GROUP_WITH_CP = None
    global _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP
    _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = None

    global _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    _safe_destroy_gloo(_DATA_PARALLEL_GROUP_WITH_CP_GLOO)
    _DATA_PARALLEL_GROUP_WITH_CP_GLOO = None

    global _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP
    _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = None
    global _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO
    _safe_destroy_gloo(_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO)
    _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO = None

    global _CONTEXT_PARALLEL_GROUP
    _CONTEXT_PARALLEL_GROUP = None
    global _CONTEXT_PARALLEL_GLOBAL_RANKS
    _CONTEXT_PARALLEL_GLOBAL_RANKS = None
    global _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS
    _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS = None
    global _HYBRID_DP_CP_GROUPS
    _HYBRID_DP_CP_GROUPS = {}

    global _EMBEDDING_GROUP
    _EMBEDDING_GROUP = None
    global _EMBEDDING_GLOBAL_RANKS
    _EMBEDDING_GLOBAL_RANKS = None
    global _POSITION_EMBEDDING_GROUP
    _POSITION_EMBEDDING_GROUP = None
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    _POSITION_EMBEDDING_GLOBAL_RANKS = None

    global _TENSOR_AND_DATA_PARALLEL_GROUP
    _TENSOR_AND_DATA_PARALLEL_GROUP = None
    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = None
    global _TENSOR_AND_CONTEXT_PARALLEL_GROUP
    _TENSOR_AND_CONTEXT_PARALLEL_GROUP = None

    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None

    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_DATA_PARALLEL_WORLD_SIZE
    _MPU_DATA_PARALLEL_WORLD_SIZE = None
    global _MPU_DATA_PARALLEL_RANK
    _MPU_DATA_PARALLEL_RANK = None
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = None
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = None

    # Expert parallel
    global _EXPERT_MODEL_PARALLEL_GROUP
    _EXPERT_MODEL_PARALLEL_GROUP = None
    global _EXPERT_MODEL_PARALLEL_RANKS
    _EXPERT_MODEL_PARALLEL_RANKS = None
    global _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_EXPERT_MODEL_PARALLEL_RANK
    _MPU_EXPERT_MODEL_PARALLEL_RANK = None

    global _EXPERT_TENSOR_PARALLEL_GROUP
    _EXPERT_TENSOR_PARALLEL_GROUP = None
    global _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE = None
    global _MPU_EXPERT_TENSOR_PARALLEL_RANK
    _MPU_EXPERT_TENSOR_PARALLEL_RANK = None

    global _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP
    _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = None
    global _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP
    _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP = None

    global _EXPERT_DATA_PARALLEL_GROUP
    _EXPERT_DATA_PARALLEL_GROUP = None
    global _EXPERT_DATA_PARALLEL_GROUP_GLOO
    _safe_destroy_gloo(_EXPERT_DATA_PARALLEL_GROUP_GLOO)
    _EXPERT_DATA_PARALLEL_GROUP_GLOO = None

    global _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP
    _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = None
    global _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO
    _safe_destroy_gloo(_INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO)
    _INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO = None
    global _INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP
    _INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = None

    global _INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP
    _INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP = None

    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None

    global _global_process_group_list
    _global_process_group_list = None

    # DES-LOC tier groups
    global _TIER_GROUPS
    _TIER_GROUPS = {}
    global _LOCAL_TIER
    _LOCAL_TIER = None


def dump_process_groups(rank: Optional[int] = None) -> str:
    """Return a readable summary of all PGs for this rank.

    From Megatron M2638: add debug repr for process groups, adapted for
    DES-LOC where different hardware tiers join different PGs.

    Args:
        rank: If set, only that rank produces output. If None, all ranks do.
    """
    import torch.distributed as _dist
    current_rank = _dist.get_rank() if _dist.is_initialized() else -1
    if rank is not None and current_rank != rank:
        return ''

    def _pg_info(label: str, pg) -> str:
        if pg is None:
            return f"  {label:<20}: [not initialized]"
        try:
            ranks = _dist.get_process_group_ranks(pg)
            backend = _dist.get_backend(pg)
            return f"  {label:<20}: ranks={ranks}  world_size={len(ranks)}  backend={backend}"
        except Exception as exc:
            return f"  {label:<20}: [error: {exc}]"

    lines = [f"=== Process Group Dump (rank={current_rank}) ==="]
    lines.append(_pg_info("TP",         _TENSOR_MODEL_PARALLEL_GROUP))
    lines.append(_pg_info("PP",         _PIPELINE_MODEL_PARALLEL_GROUP))
    lines.append(_pg_info("DP",         _DATA_PARALLEL_GROUP))
    lines.append(_pg_info("CP",         _CONTEXT_PARALLEL_GROUP))
    lines.append(_pg_info("EP",         _EXPERT_MODEL_PARALLEL_GROUP))
    lines.append(_pg_info("Embedding",  _EMBEDDING_GROUP))
    lines.append(_pg_info("PosEmb",     _POSITION_EMBEDDING_GROUP))
    if _TIER_GROUPS:
        for tname, tpg in _TIER_GROUPS.items():
            lines.append(_pg_info(f"TIER[{tname}]", tpg))
        lines.append(f"  {'LOCAL_TIER':<20}: {_LOCAL_TIER}")
    else:
        lines.append(f"  {'TIER groups':<20}: [not initialized]")
    return "\n".join(lines)
    # From Megatron M2638: debug repr for process groups (DES-LOC adaptation)


# ---------------------------------------------------------------------------
# Safe rank / world-size accessors with SLURM fallback (M4022)
# ---------------------------------------------------------------------------
# From Megatron M4022 (Various training utils, PR #4872):
# Megatron added safe_get_rank / safe_get_world_size that fall back to
# SLURM environment variables when torch.distributed is not yet initialized.
# This is important for our heterogeneous SLURM cluster where some logging /
# checkpointing code may run before dist.init_process_group().

def _resolve_slurm_rank() -> Optional[int]:
    """Return global rank from SLURM env, or None if not in a SLURM job."""
    if "SLURM_NTASKS" not in os.environ:
        return None
    procid = os.environ.get("SLURM_PROCID")
    return int(procid) if procid is not None else None


def _resolve_slurm_world_size() -> Optional[int]:
    """Return world size from SLURM env, or None if not in a SLURM job."""
    ntasks = os.environ.get("SLURM_NTASKS")
    return int(ntasks) if ntasks is not None else None


def safe_get_rank() -> int:
    """Get distributed rank safely, even before torch.distributed is initialized.

    Fallback order:
    1. torch.distributed.get_rank()  — if initialized
    2. RANK env var                  — torchrun / torchelastic
    3. SLURM_PROCID env var          — SLURM launcher
    4. 0 with a warning              — single-process / unknown

    From Megatron M4022 (PR #4872).
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()

    if "RANK" in os.environ:
        try:
            return int(os.environ["RANK"])
        except (ValueError, TypeError):
            pass

    slurm_rank = _resolve_slurm_rank()
    if slurm_rank is not None:
        return slurm_rank

    warnings.warn(
        "safe_get_rank: could not determine rank from torch.distributed, "
        "RANK, or SLURM_PROCID — defaulting to 0.",
        stacklevel=2,
    )
    return 0


def safe_get_world_size() -> int:
    """Get distributed world size safely, even before torch.distributed is initialized.

    Fallback order:
    1. torch.distributed.get_world_size()  — if initialized
    2. WORLD_SIZE env var                  — torchrun / torchelastic
    3. SLURM_NTASKS env var               — SLURM launcher
    4. 1 with a warning                    — single-process / unknown

    From Megatron M4022 (PR #4872).
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()

    if "WORLD_SIZE" in os.environ:
        try:
            return int(os.environ["WORLD_SIZE"])
        except (ValueError, TypeError):
            pass

    slurm_ws = _resolve_slurm_world_size()
    if slurm_ws is not None:
        return slurm_ws

    warnings.warn(
        "safe_get_world_size: could not determine world size from torch.distributed, "
        "WORLD_SIZE, or SLURM_NTASKS — defaulting to 1.",
        stacklevel=2,
    )
    return 1
