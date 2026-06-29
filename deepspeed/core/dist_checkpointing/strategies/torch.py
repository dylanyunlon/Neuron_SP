# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Strategies using plain torch.save / torch.load as an underlying format."""

import os
from pathlib import Path
from typing import List

import torch

from ..core import CheckpointingException
from ..mapping import ShardedTensor, ShardedStateDict, StateDict, is_main_replica
from ..dict_utils import dict_list_map_inplace
from .base import (
    default_strategies,
    StrategyAction,
    LoadShardedStrategy,
    SaveShardedStrategy,
    LoadCommonStrategy,
    SaveCommonStrategy,
)

# File name used by the common (non-sharded) torch strategy.
COMMON_STATE_FNAME = 'common.pt'
# Sub-directory that holds per-key shard files.
SHARDS_DIR = 'shards'


# ---------------------------------------------------------------------------
# Common (non-sharded) strategies
# ---------------------------------------------------------------------------

class TorchSaveCommonStrategy(SaveCommonStrategy):
    """Save common (non-sharded) state dict with torch.save on rank 0."""

    def save(self, common_state_dict: StateDict, checkpoint_dir: Path) -> None:
        """Save common state dict.

        Only rank 0 writes; all other ranks skip so there is no race.

        Args:
            common_state_dict: state dict containing plain (non-sharded) tensors
                and scalar values.
            checkpoint_dir: directory where the checkpoint is being written.
        """
        checkpoint_dir = Path(checkpoint_dir)
        if torch.distributed.get_rank() == 0:
            torch.save(common_state_dict, checkpoint_dir / COMMON_STATE_FNAME)

    def check_backend_compatibility(self, loaded_version: int) -> None:
        pass  # torch backend is always compatible with itself

    def check_version_compatibility(self, loaded_version: int) -> None:
        pass  # no versioned format changes yet


class TorchLoadCommonStrategy(LoadCommonStrategy):
    """Load common (non-sharded) state dict with torch.load."""

    def load(self, checkpoint_dir: Path) -> StateDict:
        """Load and return the common state dict.

        Args:
            checkpoint_dir: directory containing the checkpoint.

        Returns:
            state dict loaded from *common.pt*.

        Raises:
            CheckpointingException: if the common state file does not exist.
        """
        checkpoint_dir = Path(checkpoint_dir)
        common_path = checkpoint_dir / COMMON_STATE_FNAME
        if not common_path.exists():
            raise CheckpointingException(
                f'Common state file not found: {common_path}'
            )
        return torch.load(common_path, map_location='cpu', weights_only=False)

    def check_backend_compatibility(self, loaded_version: int) -> None:
        pass

    def check_version_compatibility(self, loaded_version: int) -> None:
        pass


# ---------------------------------------------------------------------------
# Sharded strategies
# ---------------------------------------------------------------------------

class TorchSaveShardedStrategy(SaveShardedStrategy):
    """Save sharded tensors using torch.save, one file per (key, global_offset).

    Each shard is stored as::

        <checkpoint_dir>/shards/<key>/<offset_tuple>.pt

    Only main-replica shards are persisted; replica shards are silently skipped.
    A distributed barrier is performed after all ranks finish writing so that
    subsequent reads are safe.
    """

    def save(self, sharded_tensors: List[ShardedTensor], checkpoint_dir: Path) -> None:
        """Save all sharded tensors.

        Args:
            sharded_tensors: list of :class:`ShardedTensor` objects whose
                ``data`` attribute holds the local tensor slice to save.
            checkpoint_dir: root directory for the checkpoint.
        """
        checkpoint_dir = Path(checkpoint_dir)
        shards_root = checkpoint_dir / SHARDS_DIR

        for sh_ten in sharded_tensors:
            if not is_main_replica(sh_ten.replica_id):
                continue

            # Build a stable, filesystem-safe path from the key + offset.
            shard_dir = shards_root / sh_ten.key
            shard_dir.mkdir(parents=True, exist_ok=True)

            offset_str = '_'.join(str(o) for o in sh_ten.global_offset)
            shard_path = shard_dir / f'{offset_str}.pt'

            payload = {
                'data': sh_ten.data.detach().cpu(),
                'global_shape': sh_ten.global_shape,
                'local_shape': sh_ten.local_shape,
                'global_offset': sh_ten.global_offset,
                'dtype': sh_ten.dtype,
                'flattened_range': sh_ten.flattened_range,
            }
            torch.save(payload, shard_path)

        torch.distributed.barrier()

    def check_backend_compatibility(self, loaded_version: int) -> None:
        pass

    def check_version_compatibility(self, loaded_version: int) -> None:
        pass


class TorchLoadShardedStrategy(LoadShardedStrategy):
    """Load sharded tensors that were saved by :class:`TorchSaveShardedStrategy`.

    For each :class:`ShardedTensor` in the state dict the corresponding shard
    file ``<checkpoint_dir>/shards/<key>/<offset_tuple>.pt`` is located and
    the local tensor slice is loaded and placed back into the state dict.
    """

    def load(
        self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path
    ) -> ShardedStateDict:
        """Load sharded tensors from the checkpoint directory.

        Args:
            sharded_state_dict: mapping of state-dict keys to
                :class:`ShardedTensor` objects describing the expected sharding.
            checkpoint_dir: directory containing the checkpoint.

        Returns:
            The same mapping but with :class:`ShardedTensor` objects replaced
            by plain :class:`torch.Tensor` values.

        Raises:
            CheckpointingException: if a required shard file is missing or the
                stored shape does not match the expected shape.
        """
        checkpoint_dir = Path(checkpoint_dir)
        shards_root = checkpoint_dir / SHARDS_DIR

        def _load_shard(sh_ten: ShardedTensor) -> torch.Tensor:
            if not isinstance(sh_ten, ShardedTensor):
                return sh_ten

            offset_str = '_'.join(str(o) for o in sh_ten.global_offset)
            shard_path = shards_root / sh_ten.key / f'{offset_str}.pt'

            if not shard_path.exists():
                raise CheckpointingException(
                    f'Shard file not found for key "{sh_ten.key}" '
                    f'at offset {sh_ten.global_offset}: {shard_path}'
                )

            payload = torch.load(shard_path, map_location='cpu', weights_only=False)
            tensor: torch.Tensor = payload['data']

            # Shape sanity check (honour allow_shape_mismatch flag).
            if (
                not sh_ten.allow_shape_mismatch
                and tuple(tensor.shape) != tuple(sh_ten.local_shape)
            ):
                raise CheckpointingException(
                    f'Local shape mismatch for key "{sh_ten.key}": '
                    f'loaded {tuple(tensor.shape)}, '
                    f'expected {sh_ten.local_shape}'
                )

            # Restore flattened-range slice if present.
            if sh_ten.flattened_range is not None:
                tensor = tensor.flatten()[sh_ten.flattened_range]

            return tensor

        dict_list_map_inplace(_load_shard, sharded_state_dict)
        return sharded_state_dict

    def check_backend_compatibility(self, loaded_version: int) -> None:
        pass

    def check_version_compatibility(self, loaded_version: int) -> None:
        pass


# ---------------------------------------------------------------------------
# Register as defaults
# ---------------------------------------------------------------------------

default_strategies[StrategyAction.SAVE_SHARDED.value][('torch', 1)] = (
    TorchSaveShardedStrategy('torch', 1)
)
default_strategies[StrategyAction.LOAD_SHARDED.value][('torch', 1)] = (
    TorchLoadShardedStrategy('torch', 1)
)
default_strategies[StrategyAction.SAVE_COMMON.value][('torch', 1)] = (
    TorchSaveCommonStrategy('torch', 1)
)
default_strategies[StrategyAction.LOAD_COMMON.value][('torch', 1)] = (
    TorchLoadCommonStrategy('torch', 1)
)
