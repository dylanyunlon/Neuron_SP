# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Parallel state management for heterogeneous GPU clusters.

Manages process groups for TP, PP, DP, SP, and DES-LOC tier groups.
All collective operations in deepspeed/core/ route through these groups.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from deepspeed.core.desloc_config import DesLocConfig, TierSpec


# ---------------------------------------------------------------------------
# Module-level state (initialized by initialize_model_parallel)
# ---------------------------------------------------------------------------

_TENSOR_MODEL_PARALLEL_GROUP: Optional[torch.distributed.ProcessGroup] = None
_PIPELINE_MODEL_PARALLEL_GROUP: Optional[torch.distributed.ProcessGroup] = None
_DATA_PARALLEL_GROUP: Optional[torch.distributed.ProcessGroup] = None
_DATA_PARALLEL_GROUP_WITH_CP: Optional[torch.distributed.ProcessGroup] = None
_SEQUENCE_PARALLEL_GROUP: Optional[torch.distributed.ProcessGroup] = None
_CONTEXT_PARALLEL_GROUP: Optional[torch.distributed.ProcessGroup] = None

# DES-LOC tier groups: GPUs in the same tier
_TIER_GROUPS: dict[str, torch.distributed.ProcessGroup] = {}

# Rank info cache
_TENSOR_MODEL_PARALLEL_RANK: Optional[int] = None
_PIPELINE_MODEL_PARALLEL_RANK: Optional[int] = None
_DATA_PARALLEL_RANK: Optional[int] = None

# Pipeline stage boundary ranks (global ranks)
_PIPELINE_GLOBAL_RANKS: Optional[List[int]] = None

# Local tier for the current rank (set during initialize_model_parallel)
_LOCAL_TIER: Optional[TierSpec] = None


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    sequence_parallel_size: int = 1,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    desloc_config: Optional[DesLocConfig] = None,
) -> None:
    """Initialize all model-parallel process groups.

    Must be called after torch.distributed.init_process_group().
    Creates TP, PP, DP, SP, CP groups and optionally DES-LOC tier groups.

    Args:
        tensor_model_parallel_size: TP degree.
        pipeline_model_parallel_size: PP degree.
        virtual_pipeline_model_parallel_size: VPP degree (interleaved 1F1B).
        sequence_parallel_size: SP degree for AutoSP.
        context_parallel_size: CP degree.
        expert_model_parallel_size: EP degree for MoE.
        desloc_config: DES-LOC config with tier specs for heterogeneous groups.
    """
    assert torch.distributed.is_initialized(), \
        "torch.distributed must be initialized before calling initialize_model_parallel"

    world_size: int = torch.distributed.get_world_size()
    rank: int = torch.distributed.get_rank()

    # Validate sizes
    model_parallel_size = tensor_model_parallel_size * pipeline_model_parallel_size
    if world_size % model_parallel_size != 0:
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by "
            f"tensor_model_parallel_size ({tensor_model_parallel_size}) * "
            f"pipeline_model_parallel_size ({pipeline_model_parallel_size})"
        )

    data_parallel_size: int = world_size // model_parallel_size

    # -----------------------------------------------------------------------
    # Build TP groups
    # Layout: rank = tp_rank + dp_rank * tp_size + pp_rank * tp_size * dp_size
    # TP group: all ranks with same dp_rank and pp_rank
    # -----------------------------------------------------------------------
    global _TENSOR_MODEL_PARALLEL_GROUP
    assert _TENSOR_MODEL_PARALLEL_GROUP is None, \
        "tensor model parallel group is already initialized"

    tp = tensor_model_parallel_size
    dp = data_parallel_size
    pp = pipeline_model_parallel_size

    # For each (dp_rank, pp_rank) combination, collect tp_size ranks
    for pp_rank in range(pp):
        for dp_rank in range(dp):
            tp_ranks = [
                tp_rank + dp_rank * tp + pp_rank * tp * dp
                for tp_rank in range(tp)
            ]
            group = torch.distributed.new_group(ranks=tp_ranks)
            if rank in tp_ranks:
                _TENSOR_MODEL_PARALLEL_GROUP = group
                _TENSOR_MODEL_PARALLEL_RANK = tp_ranks.index(rank)

    # -----------------------------------------------------------------------
    # Build PP groups
    # PP group: all ranks with same tp_rank and dp_rank, varying pp_rank
    # -----------------------------------------------------------------------
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert _PIPELINE_MODEL_PARALLEL_GROUP is None, \
        "pipeline model parallel group is already initialized"

    for dp_rank in range(dp):
        for tp_rank in range(tp):
            pp_ranks = [
                tp_rank + dp_rank * tp + pp_rank * tp * dp
                for pp_rank in range(pp)
            ]
            group = torch.distributed.new_group(ranks=pp_ranks)
            if rank in pp_ranks:
                _PIPELINE_MODEL_PARALLEL_GROUP = group
                _PIPELINE_MODEL_PARALLEL_RANK = pp_ranks.index(rank)
                _PIPELINE_GLOBAL_RANKS = pp_ranks

    # -----------------------------------------------------------------------
    # Build DP groups
    # DP group: all ranks with same tp_rank and pp_rank, varying dp_rank
    # -----------------------------------------------------------------------
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, \
        "data parallel group is already initialized"

    for pp_rank in range(pp):
        for tp_rank in range(tp):
            dp_ranks = [
                tp_rank + dp_rank * tp + pp_rank * tp * dp
                for dp_rank in range(dp)
            ]
            group = torch.distributed.new_group(ranks=dp_ranks)
            if rank in dp_ranks:
                _DATA_PARALLEL_GROUP = group
                _DATA_PARALLEL_RANK = dp_ranks.index(rank)

    # -----------------------------------------------------------------------
    # Build SP (Sequence Parallel / AutoSP) groups
    # SP shares the same group topology as TP when sequence_parallel_size == tp.
    # When sequence_parallel_size > 1 and differs from tp, we create separate SP groups.
    # -----------------------------------------------------------------------
    global _SEQUENCE_PARALLEL_GROUP
    if sequence_parallel_size > 1:
        sp = sequence_parallel_size
        # SP groups are formed within each PP stage, tiling over DP
        # SP size must divide world_size within a PP stage
        ranks_per_pp_stage = tp * dp
        if ranks_per_pp_stage % sp != 0:
            raise RuntimeError(
                f"ranks_per_pp_stage ({ranks_per_pp_stage}) is not divisible by "
                f"sequence_parallel_size ({sp})"
            )
        sp_groups_per_pp = ranks_per_pp_stage // sp
        for pp_rank in range(pp):
            for g in range(sp_groups_per_pp):
                sp_ranks = [
                    pp_rank * ranks_per_pp_stage + g * sp + i
                    for i in range(sp)
                ]
                group = torch.distributed.new_group(ranks=sp_ranks)
                if rank in sp_ranks:
                    _SEQUENCE_PARALLEL_GROUP = group

    # -----------------------------------------------------------------------
    # Build CP (Context Parallel) groups
    # CP is orthogonal to TP; within each (pp_rank, dp_rank), CP tiles over tp_rank
    # -----------------------------------------------------------------------
    global _CONTEXT_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_WITH_CP
    if context_parallel_size > 1:
        cp = context_parallel_size
        if dp % cp != 0:
            raise RuntimeError(
                f"data_parallel_size ({dp}) is not divisible by "
                f"context_parallel_size ({cp})"
            )
        dp_without_cp = dp // cp

        # CP groups: for each (pp_rank, dp_without_cp_rank, tp_rank), group cp ranks
        for pp_rank in range(pp):
            for dp_base_rank in range(dp_without_cp):
                for tp_rank in range(tp):
                    cp_ranks = [
                        tp_rank + (dp_base_rank * cp + cp_rank) * tp + pp_rank * tp * dp
                        for cp_rank in range(cp)
                    ]
                    group = torch.distributed.new_group(ranks=cp_ranks)
                    if rank in cp_ranks:
                        _CONTEXT_PARALLEL_GROUP = group

        # DP-with-CP groups: for each (pp_rank, tp_rank), group dp*cp ranks
        for pp_rank in range(pp):
            for tp_rank in range(tp):
                dp_cp_ranks = [
                    tp_rank + dp_rank * tp + pp_rank * tp * dp
                    for dp_rank in range(dp)
                ]
                group = torch.distributed.new_group(ranks=dp_cp_ranks)
                if rank in dp_cp_ranks:
                    _DATA_PARALLEL_GROUP_WITH_CP = group
    else:
        # No CP; DP-with-CP equals DP
        _DATA_PARALLEL_GROUP_WITH_CP = _DATA_PARALLEL_GROUP

    # -----------------------------------------------------------------------
    # Build DES-LOC tier groups
    # Each tier in desloc_config.tiers contains a list of gpu_indices (global
    # ranks).  We create one process group per distinct TierType.  Multiple
    # TierSpec entries sharing the same TierType are merged into a single group.
    # NCCL requires ALL ranks to call new_group() with the same ranks list,
    # even non-members; we therefore iterate every tier unconditionally.
    # -----------------------------------------------------------------------
    global _TIER_GROUPS
    global _LOCAL_TIER
    if desloc_config is not None and desloc_config.tiers:
        # Merge gpu_indices by tier_type name (handles duplicate TierType entries)
        tier_indices_map: dict[str, List[int]] = {}
        tier_spec_map: dict[str, TierSpec] = {}
        for tier_spec in desloc_config.tiers:
            tier_name = tier_spec.tier_type.name.lower()
            if tier_name not in tier_indices_map:
                tier_indices_map[tier_name] = []
                tier_spec_map[tier_name] = tier_spec
            for idx in tier_spec.gpu_indices:
                if idx not in tier_indices_map[tier_name]:
                    tier_indices_map[tier_name].append(idx)

        # All ranks must participate in every new_group call (NCCL collective)
        for tier_name, gpu_indices in tier_indices_map.items():
            if not gpu_indices:
                continue
            sorted_indices = sorted(gpu_indices)
            group = torch.distributed.new_group(ranks=sorted_indices)
            _TIER_GROUPS[tier_name] = group
            if rank in sorted_indices:
                _LOCAL_TIER = tier_spec_map[tier_name]


def is_initialized() -> bool:
    """Return True if model parallel groups have been initialized."""
    return _DATA_PARALLEL_GROUP is not None


# --- TP ---
def get_tensor_model_parallel_group() -> torch.distributed.ProcessGroup:
    """Get the tensor-model-parallel group the caller rank belongs to."""
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, \
        "tensor model parallel group is not initialized"
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_world_size() -> int:
    """Return the world size of the tensor model parallel group."""
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())


def get_tensor_model_parallel_rank() -> int:
    """Return the rank of the current process in the tensor model parallel group."""
    if _TENSOR_MODEL_PARALLEL_RANK is not None:
        return _TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


# --- PP ---
def get_pipeline_model_parallel_group() -> torch.distributed.ProcessGroup:
    """Get the pipeline-model-parallel group the caller rank belongs to."""
    assert _PIPELINE_MODEL_PARALLEL_GROUP is not None, \
        "pipeline model parallel group is not initialized"
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_world_size() -> int:
    """Return the world size of the pipeline model parallel group."""
    return torch.distributed.get_world_size(group=get_pipeline_model_parallel_group())


def get_pipeline_model_parallel_rank() -> int:
    """Return the rank of the current process in the pipeline model parallel group."""
    if _PIPELINE_MODEL_PARALLEL_RANK is not None:
        return _PIPELINE_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())


def is_pipeline_first_stage() -> bool:
    """Return True if the caller is the first pipeline stage (rank 0 in the PP group)."""
    return get_pipeline_model_parallel_rank() == 0


def is_pipeline_last_stage() -> bool:
    """Return True if the caller is the last pipeline stage."""
    return get_pipeline_model_parallel_rank() == (get_pipeline_model_parallel_world_size() - 1)


# --- DP ---
def get_data_parallel_group(with_context_parallel: bool = False) -> torch.distributed.ProcessGroup:
    """Get the data-parallel group the caller rank belongs to.

    Args:
        with_context_parallel: If True, return the DP group that includes CP ranks.
    """
    if with_context_parallel:
        assert _DATA_PARALLEL_GROUP_WITH_CP is not None, \
            "data parallel group with context parallel is not initialized"
        return _DATA_PARALLEL_GROUP_WITH_CP
    assert _DATA_PARALLEL_GROUP is not None, \
        "data parallel group is not initialized"
    return _DATA_PARALLEL_GROUP


def get_data_parallel_world_size(with_context_parallel: bool = False) -> int:
    """Return the world size of the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group(with_context_parallel))


def get_data_parallel_rank(with_context_parallel: bool = False) -> int:
    """Return the rank of the current process in the data parallel group."""
    if not with_context_parallel and _DATA_PARALLEL_RANK is not None:
        return _DATA_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_data_parallel_group(with_context_parallel))


# --- SP (AutoSP) ---
def get_sequence_parallel_group() -> Optional[torch.distributed.ProcessGroup]:
    """Get the sequence-parallel group the caller rank belongs to, or None."""
    return _SEQUENCE_PARALLEL_GROUP


def get_sequence_parallel_world_size() -> int:
    """Return the world size of the sequence parallel group (1 if not initialized)."""
    if _SEQUENCE_PARALLEL_GROUP is None:
        return 1
    return torch.distributed.get_world_size(group=_SEQUENCE_PARALLEL_GROUP)


def get_sequence_parallel_rank() -> int:
    """Return the rank of the current process in the sequence parallel group (0 if not initialized)."""
    if _SEQUENCE_PARALLEL_GROUP is None:
        return 0
    return torch.distributed.get_rank(group=_SEQUENCE_PARALLEL_GROUP)


# --- CP ---
def get_context_parallel_group() -> Optional[torch.distributed.ProcessGroup]:
    """Get the context-parallel group the caller rank belongs to, or None."""
    return _CONTEXT_PARALLEL_GROUP


def get_context_parallel_world_size() -> int:
    """Return the world size of the context parallel group (1 if not initialized)."""
    if _CONTEXT_PARALLEL_GROUP is None:
        return 1
    return torch.distributed.get_world_size(group=_CONTEXT_PARALLEL_GROUP)


def get_context_parallel_rank() -> int:
    """Return the rank of the current process in the context parallel group (0 if not initialized)."""
    if _CONTEXT_PARALLEL_GROUP is None:
        return 0
    return torch.distributed.get_rank(group=_CONTEXT_PARALLEL_GROUP)


# --- PP helpers ---
def get_pipeline_model_parallel_first_rank() -> int:
    """Return the global rank of the first stage in the current rank's pipeline group."""
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "pipeline parallel group is not initialized"
    return _PIPELINE_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_last_rank() -> int:
    """Return the global rank of the last stage in the current rank's pipeline group."""
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "pipeline parallel group is not initialized"
    last_rank_local = get_pipeline_model_parallel_world_size() - 1
    return _PIPELINE_GLOBAL_RANKS[last_rank_local]


def get_pipeline_model_parallel_next_rank() -> int:
    """Return the global rank of the next stage in the current rank's pipeline group."""
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def get_pipeline_model_parallel_prev_rank() -> int:
    """Return the global rank of the previous stage in the current rank's pipeline group."""
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


# --- TP / DP src-rank helpers (used for weight broadcast) ---
def get_tensor_model_parallel_src_rank() -> int:
    """Return the global rank of the first member in the caller's TP group.

    Used to broadcast RNG states and model weights from rank-0 of the TP
    group to all other TP ranks.  In the canonical rank layout::

        global_rank = tp_rank + dp_rank * tp_size + pp_rank * tp_size * dp_size

    the first TP rank in each group shares the same integer-division bucket
    when dividing by tp_size.
    """
    global_rank = torch.distributed.get_rank()
    tp_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // tp_world_size) * tp_world_size


def get_data_parallel_src_rank(with_context_parallel: bool = False) -> int:
    """Return the global rank of the first member in the caller's DP group.

    Used to broadcast model weights from the canonical source replica.
    """
    assert _DATA_PARALLEL_GROUP is not None, \
        "data parallel group is not initialized"
    dp_group = get_data_parallel_group(with_context_parallel=with_context_parallel)
    dp_ranks = torch.distributed.get_process_group_ranks(dp_group)
    return min(dp_ranks)


# --- DES-LOC tier groups ---
def get_tier_group(tier_name: str) -> Optional[torch.distributed.ProcessGroup]:
    """Get process group for a DES-LOC tier (e.g. 'datacenter', 'professional')."""
    return _TIER_GROUPS.get(tier_name, None)


def get_all_tier_groups() -> dict[str, torch.distributed.ProcessGroup]:
    """Return a copy of the tier groups dictionary."""
    return dict(_TIER_GROUPS)


def get_local_tier() -> Optional[TierSpec]:
    """Get the tier spec for the current rank's GPU."""
    return _LOCAL_TIER


# --- Cleanup ---
def destroy_model_parallel() -> None:
    """Destroy all model-parallel process groups and reset module state.

    Calls ``torch.distributed.destroy_process_group`` on every group that
    was created by ``initialize_model_parallel`` before clearing the
    module-level references, so that NCCL communicators are released
    immediately rather than waiting for garbage collection.
    """
    def _destroy(group: Optional[torch.distributed.ProcessGroup]) -> None:
        if group is not None and torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group(group)
            except Exception:
                pass

    global _TENSOR_MODEL_PARALLEL_GROUP
    _destroy(_TENSOR_MODEL_PARALLEL_GROUP)
    _TENSOR_MODEL_PARALLEL_GROUP = None

    global _PIPELINE_MODEL_PARALLEL_GROUP
    _destroy(_PIPELINE_MODEL_PARALLEL_GROUP)
    _PIPELINE_MODEL_PARALLEL_GROUP = None

    global _DATA_PARALLEL_GROUP
    _destroy(_DATA_PARALLEL_GROUP)
    _DATA_PARALLEL_GROUP = None

    global _DATA_PARALLEL_GROUP_WITH_CP
    # Only destroy if it is a distinct group (CP==1 reuses _DATA_PARALLEL_GROUP)
    if _DATA_PARALLEL_GROUP_WITH_CP is not _DATA_PARALLEL_GROUP:
        _destroy(_DATA_PARALLEL_GROUP_WITH_CP)
    _DATA_PARALLEL_GROUP_WITH_CP = None

    global _SEQUENCE_PARALLEL_GROUP
    _destroy(_SEQUENCE_PARALLEL_GROUP)
    _SEQUENCE_PARALLEL_GROUP = None

    global _CONTEXT_PARALLEL_GROUP
    _destroy(_CONTEXT_PARALLEL_GROUP)
    _CONTEXT_PARALLEL_GROUP = None

    global _TIER_GROUPS
    for _tier_group in _TIER_GROUPS.values():
        _destroy(_tier_group)
    _TIER_GROUPS = {}

    global _TENSOR_MODEL_PARALLEL_RANK
    _TENSOR_MODEL_PARALLEL_RANK = None

    global _PIPELINE_MODEL_PARALLEL_RANK
    _PIPELINE_MODEL_PARALLEL_RANK = None

    global _DATA_PARALLEL_RANK
    _DATA_PARALLEL_RANK = None

    global _PIPELINE_GLOBAL_RANKS
    _PIPELINE_GLOBAL_RANKS = None

    global _LOCAL_TIER
    _LOCAL_TIER = None
