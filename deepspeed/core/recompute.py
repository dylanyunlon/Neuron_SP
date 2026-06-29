"""Activation recomputation (checkpointing) for heterogeneous GPU clusters.

Mirrors Megatron megatron/core/recompute.py but supports per-tier recompute
policies: A6000 (48GB) aggressively recomputes, H100 (94GB) selectively.

Megatron source: Megatron-LM/megatron/core/recompute.py
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch

from deepspeed.core.desloc_config import DesLocConfig


def checkpoint(
    function: Callable,
    distribute_saved_activations: bool,
    *args: Any,
    desloc_config: Optional[DesLocConfig] = None,
    **kwargs: Any,
) -> Any:
    """Activation checkpointing with tier-aware granularity.

    On low-VRAM tiers (A6000, 48GB), recomputes all activations.
    On high-VRAM tiers (H100, 94GB), selectively recomputes only
    attention and MLP activations (skip embedding, norm).

    Args:
        function: The forward function to checkpoint.
        distribute_saved_activations: If True, distribute saved activations
            across TP group (Megatron M2180 pattern).
        desloc_config: Per-tier memory budget for deciding recompute policy.
    """
    raise NotImplementedError("Task: read Megatron recompute.py + M2180 commit, implement")


def get_recompute_policy(
    tier_vram_gb: float,
    num_layers: int,
    hidden_size: int,
    seq_len: int,
    micro_batch_size: int,
) -> str:
    """Decide recompute policy based on available VRAM.

    Returns:
        One of 'full', 'selective', 'none'.
    """
    raise NotImplementedError("Task: implement tier-aware policy selector")
