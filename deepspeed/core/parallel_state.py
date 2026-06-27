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
_SEQUENCE_PARALLEL_GROUP: Optional[torch.distributed.ProcessGroup] = None
_CONTEXT_PARALLEL_GROUP: Optional[torch.distributed.ProcessGroup] = None

# DES-LOC tier groups: GPUs in the same tier
_TIER_GROUPS: dict[str, torch.distributed.ProcessGroup] = {}

# Rank info
_TENSOR_MODEL_PARALLEL_RANK: Optional[int] = None
_PIPELINE_MODEL_PARALLEL_RANK: Optional[int] = None
_DATA_PARALLEL_RANK: Optional[int] = None


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
    raise NotImplementedError("Claude task: parallel_state")


def is_initialized() -> bool:
    raise NotImplementedError("Claude task: parallel_state")


# --- TP ---
def get_tensor_model_parallel_group() -> torch.distributed.ProcessGroup:
    raise NotImplementedError("Claude task: parallel_state")


def get_tensor_model_parallel_world_size() -> int:
    raise NotImplementedError("Claude task: parallel_state")


def get_tensor_model_parallel_rank() -> int:
    raise NotImplementedError("Claude task: parallel_state")


# --- PP ---
def get_pipeline_model_parallel_group() -> torch.distributed.ProcessGroup:
    raise NotImplementedError("Claude task: parallel_state")


def get_pipeline_model_parallel_world_size() -> int:
    raise NotImplementedError("Claude task: parallel_state")


def get_pipeline_model_parallel_rank() -> int:
    raise NotImplementedError("Claude task: parallel_state")


def is_pipeline_first_stage() -> bool:
    raise NotImplementedError("Claude task: parallel_state")


def is_pipeline_last_stage() -> bool:
    raise NotImplementedError("Claude task: parallel_state")


# --- DP ---
def get_data_parallel_group(with_context_parallel: bool = False) -> torch.distributed.ProcessGroup:
    raise NotImplementedError("Claude task: parallel_state")


def get_data_parallel_world_size(with_context_parallel: bool = False) -> int:
    raise NotImplementedError("Claude task: parallel_state")


def get_data_parallel_rank(with_context_parallel: bool = False) -> int:
    raise NotImplementedError("Claude task: parallel_state")


# --- SP (AutoSP) ---
def get_sequence_parallel_group() -> Optional[torch.distributed.ProcessGroup]:
    raise NotImplementedError("Claude task: parallel_state")


def get_sequence_parallel_world_size() -> int:
    raise NotImplementedError("Claude task: parallel_state")


def get_sequence_parallel_rank() -> int:
    raise NotImplementedError("Claude task: parallel_state")


# --- CP ---
def get_context_parallel_group() -> Optional[torch.distributed.ProcessGroup]:
    raise NotImplementedError("Claude task: parallel_state")


# --- DES-LOC tier groups ---
def get_tier_group(tier_name: str) -> Optional[torch.distributed.ProcessGroup]:
    """Get process group for a DES-LOC tier (e.g. 'datacenter', 'professional')."""
    raise NotImplementedError("Claude task: parallel_state")


def get_all_tier_groups() -> dict[str, torch.distributed.ProcessGroup]:
    raise NotImplementedError("Claude task: parallel_state")


def get_local_tier() -> Optional[TierSpec]:
    """Get the tier spec for the current rank's GPU."""
    raise NotImplementedError("Claude task: parallel_state")


# --- Cleanup ---
def destroy_model_parallel() -> None:
    raise NotImplementedError("Claude task: parallel_state")
