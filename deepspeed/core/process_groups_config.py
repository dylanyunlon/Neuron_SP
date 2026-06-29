"""Declarative process group configuration for heterogeneous training.

Adapted from Megatron megatron/core/process_groups_config.py.

ProcessGroupCollection bundles all process groups needed by model components
(transformer layers, optimizer, DDP, finalize_model_grads) into a single
object, eliminating scattered parallel_state.get_*_group() calls.

For heterogeneous clusters: allows per-tier process group overrides,
e.g. A6000 tier may have different TP/DP groups than H100 tier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.distributed as dist

import deepspeed.core.parallel_state as parallel_state


@dataclass
class ProcessGroupCollection:
    """Unified process group collection for transformer model parallelism.

    All fields use init=False — create an instance, then set the groups you need.

    Example:
        pgs = ProcessGroupCollection()
        pgs.tp = parallel_state.get_tensor_model_parallel_group()
        pgs.dp = parallel_state.get_data_parallel_group()
        pgs.pp = parallel_state.get_pipeline_model_parallel_group()

        model = TransformerLayer(config, pg_collection=pgs)
    """

    # --- Model Parallelism Groups ---
    tp: Optional[dist.ProcessGroup] = field(init=False, default=None)
    """Tensor parallel group."""

    pp: Optional[dist.ProcessGroup] = field(init=False, default=None)
    """Pipeline parallel group."""

    mp: Optional[dist.ProcessGroup] = field(init=False, default=None)
    """Model parallel group (tensor + pipeline)."""

    cp: Optional[dist.ProcessGroup] = field(init=False, default=None)
    """Context parallel group."""

    tp_cp: Optional[dist.ProcessGroup] = field(init=False, default=None)
    """Tensor and context parallel group."""

    ep: Optional[dist.ProcessGroup] = field(init=False, default=None)
    """Expert parallel group (for MoE)."""

    expt_tp: Optional[dist.ProcessGroup] = field(init=False, default=None)
    """Expert tensor parallel group."""

    # --- Data Parallelism Groups ---
    dp: Optional[dist.ProcessGroup] = field(init=False, default=None)
    """Data parallel group."""

    dp_cp: Optional[dist.ProcessGroup] = field(init=False, default=None)
    """Data and context parallel group."""

    expt_dp: Optional[dist.ProcessGroup] = field(init=False, default=None)
    """Expert data parallel group."""

    # --- Combined Groups ---
    tp_dp_cp: Optional[dist.ProcessGroup] = field(init=False, default=None)
    """Tensor + data + context parallel group (for MoE aux loss reduction)."""

    embd: Optional[dist.ProcessGroup] = field(init=False, default=None)
    """Embedding group (first and last PP stages share embeddings)."""

    @classmethod
    def from_parallel_state(cls) -> "ProcessGroupCollection":
        """Build from the global parallel_state module.

        Convenience method that pulls all groups from parallel_state.*().
        """
        pgs = cls()
        if not parallel_state.is_initialized():
            return pgs

        # Safe getattr pattern — not all groups may be initialized
        _get = lambda name: getattr(parallel_state, f"get_{name}", lambda: None)()

        pgs.tp = _get("tensor_model_parallel_group")
        pgs.pp = _get("pipeline_model_parallel_group")
        pgs.dp = _get("data_parallel_group")
        pgs.mp = _get("model_parallel_group")
        pgs.cp = _get("context_parallel_group")
        pgs.ep = _get("expert_model_parallel_group")

        return pgs

    def get_group(self, name: str) -> Optional[dist.ProcessGroup]:
        """Get a process group by name string."""
        return getattr(self, name, None)


def get_default_pg_collection() -> ProcessGroupCollection:
    """Return a ProcessGroupCollection populated from global parallel_state.

    This is the default used by model components when no explicit
    pg_collection is passed.
    """
    return ProcessGroupCollection.from_parallel_state()
