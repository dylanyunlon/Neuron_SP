# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Distributed checkpointing with sharded save/load."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


@dataclass
class ShardedTensor:
    """Tensor with sharding metadata for distributed checkpointing."""
    key: str
    data: torch.Tensor
    global_shape: Tuple[int, ...]
    global_offset: Tuple[int, ...]
    axis_fragmentations: Tuple[int, ...]
    replica_id: int = 0

    def without_data(self) -> "ShardedTensor":
        raise NotImplementedError("Claude task: dist_checkpointing")


def save(
    sharded_state_dict: Dict[str, Any],
    checkpoint_dir: str,
    *,
    async_save: bool = False,
) -> Optional[Any]:
    """Save a sharded state dict to checkpoint_dir."""
    raise NotImplementedError("Claude task: dist_checkpointing")


def load(
    sharded_state_dict: Dict[str, Any],
    checkpoint_dir: str,
) -> Dict[str, Any]:
    """Load a sharded state dict, handling resharding if parallelism changed."""
    raise NotImplementedError("Claude task: dist_checkpointing")
