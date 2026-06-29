# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict

from ..mapping import ShardedStateDict, ShardedTensor, CheckpointingException, \
    StateDict


class StrategyAction(Enum):
    LOAD_COMMON = 'load_common'
    LOAD_SHARDED = 'load_sharded'
    SAVE_COMMON = 'save_common'
    SAVE_SHARDED = 'save_sharded'


default_strategies = defaultdict(dict)


def get_default_strategy(action: StrategyAction, backend: str, version: int):
    try:
        return default_strategies[action.value][(backend, version)]
    except KeyError as e:
        raise CheckpointingException(
            f'Cannot find default strategy for: {(action, backend, version)}'
        ) from e


class LoadStrategyBase(ABC):
    @abstractmethod
    def check_backend_compatibility(self, loaded_version):
        """Verify that this strategy is compatible with *loaded_version*.

        Concrete implementations should raise :class:`CheckpointingException`
        when the loaded backend is incompatible.  The default torch
        implementation treats all versions as compatible.
        """
        # Implemented by concrete sub-classes (e.g. TorchLoadShardedStrategy).
        # The abstract body is intentionally a no-op so that callers who
        # invoke super() get a safe default.
        pass

    @abstractmethod
    def check_version_compatibility(self, loaded_version):
        """Verify that this strategy can read a checkpoint at *loaded_version*.

        Concrete implementations should raise :class:`CheckpointingException`
        when the version is too old or too new.  The default torch
        implementation treats all versions as compatible.
        """
        pass


class SaveStrategyBase(ABC):
    def __init__(self, backend: str, version: int):
        self.backend = backend
        self.version = version


class LoadCommonStrategy(LoadStrategyBase):
    @abstractmethod
    def load(self, checkpoint_dir: Path):
        """Load and return the common (non-sharded) state dict.

        Uses :func:`torch.load` to deserialise the common state file that was
        written by :class:`SaveCommonStrategy`.

        Args:
            checkpoint_dir: directory that contains the checkpoint.

        Returns:
            Plain Python / PyTorch state dict.

        Raises:
            CheckpointingException: if the expected common state file is absent.
        """
        # Concrete torch implementation lives in
        # .torch.TorchLoadCommonStrategy; this body is never called directly
        # because the class is abstract.
        import torch
        from pathlib import Path as _Path
        from ..mapping import CheckpointingException as _CE

        _checkpoint_dir = _Path(checkpoint_dir)
        _common_path = _checkpoint_dir / 'common.pt'
        if not _common_path.exists():
            raise _CE(f'Common state file not found: {_common_path}')
        return torch.load(_common_path, map_location='cpu', weights_only=False)


class LoadShardedStrategy(LoadStrategyBase):
    @abstractmethod
    def load(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path):
        """Load sharded tensors from *checkpoint_dir* into *sharded_state_dict*.

        Each :class:`~..mapping.ShardedTensor` in *sharded_state_dict* is
        replaced in-place with the corresponding local :class:`torch.Tensor`
        slice loaded from disk via :func:`torch.load`.

        Args:
            sharded_state_dict: mapping of state-dict keys to
                :class:`~..mapping.ShardedTensor` objects describing the
                expected sharding.
            checkpoint_dir: root checkpoint directory produced by
                :class:`SaveShardedStrategy`.

        Returns:
            The updated *sharded_state_dict* with tensors substituted for
            :class:`~..mapping.ShardedTensor` placeholders.

        Raises:
            CheckpointingException: if a required shard file is missing or the
                shape of the loaded tensor does not match expectations.
        """
        # Concrete torch implementation lives in
        # .torch.TorchLoadShardedStrategy; this body is never called directly.
        import torch as _torch
        from pathlib import Path as _Path
        from ..mapping import CheckpointingException as _CE, is_main_replica
        from ..dict_utils import dict_list_map_inplace

        _checkpoint_dir = _Path(checkpoint_dir)
        _shards_root = _checkpoint_dir / 'shards'

        def _load_shard(sh_ten):
            if not isinstance(sh_ten, ShardedTensor):
                return sh_ten
            _offset_str = '_'.join(str(o) for o in sh_ten.global_offset)
            _shard_path = _shards_root / sh_ten.key / f'{_offset_str}.pt'
            if not _shard_path.exists():
                raise _CE(
                    f'Shard file not found for key "{sh_ten.key}" '
                    f'at offset {sh_ten.global_offset}: {_shard_path}'
                )
            _payload = _torch.load(_shard_path, map_location='cpu', weights_only=False)
            _tensor = _payload['data']
            if (not sh_ten.allow_shape_mismatch
                    and tuple(_tensor.shape) != tuple(sh_ten.local_shape)):
                raise _CE(
                    f'Local shape mismatch for key "{sh_ten.key}": '
                    f'loaded {tuple(_tensor.shape)}, expected {sh_ten.local_shape}'
                )
            if sh_ten.flattened_range is not None:
                _tensor = _tensor.flatten()[sh_ten.flattened_range]
            return _tensor

        dict_list_map_inplace(_load_shard, sharded_state_dict)
        return sharded_state_dict


class SaveCommonStrategy(SaveStrategyBase):
    @abstractmethod
    def save(self, common_state_dict: StateDict, checkpoint_dir: Path):
        """Save the common (non-sharded) portion of the state dict.

        Rank 0 writes *common_state_dict* to ``common.pt`` inside
        *checkpoint_dir* using :func:`torch.save`.  All other ranks skip the
        write so there is no race condition.

        Args:
            common_state_dict: plain Python / PyTorch state dict containing
                non-sharded tensors and scalar values.
            checkpoint_dir: destination directory (must already exist).
        """
        import torch as _torch
        from pathlib import Path as _Path

        _checkpoint_dir = _Path(checkpoint_dir)
        if _torch.distributed.get_rank() == 0:
            _torch.save(common_state_dict, _checkpoint_dir / 'common.pt')


class SaveShardedStrategy(SaveStrategyBase):
    @abstractmethod
    def save(self, sharded_tensors: List[ShardedTensor], checkpoint_dir: Path):
        """Save sharded tensors to *checkpoint_dir*.

        Each main-replica :class:`~..mapping.ShardedTensor` in
        *sharded_tensors* is persisted as an individual file under
        ``<checkpoint_dir>/shards/<key>/<global_offset>.pt`` using
        :func:`torch.save`.  Replica tensors (``replica_id != 0``) are
        silently skipped.  A distributed barrier synchronises all ranks after
        the writes complete.

        Args:
            sharded_tensors: flat list of :class:`~..mapping.ShardedTensor`
                objects whose ``data`` attribute contains the local slice.
            checkpoint_dir: root directory for the checkpoint.
        """
        import torch as _torch
        from pathlib import Path as _Path
        from ..mapping import is_main_replica

        _checkpoint_dir = _Path(checkpoint_dir)
        _shards_root = _checkpoint_dir / 'shards'

        for _sh_ten in sharded_tensors:
            if not is_main_replica(_sh_ten.replica_id):
                continue
            _shard_dir = _shards_root / _sh_ten.key
            _shard_dir.mkdir(parents=True, exist_ok=True)
            _offset_str = '_'.join(str(o) for o in _sh_ten.global_offset)
            _shard_path = _shard_dir / f'{_offset_str}.pt'
            _payload = {
                'data': _sh_ten.data.detach().cpu(),
                'global_shape': _sh_ten.global_shape,
                'local_shape': _sh_ten.local_shape,
                'global_offset': _sh_ten.global_offset,
                'dtype': _sh_ten.dtype,
                'flattened_range': _sh_ten.flattened_range,
            }
            _torch.save(_payload, _shard_path)

        _torch.distributed.barrier()


print('[M1437]')
