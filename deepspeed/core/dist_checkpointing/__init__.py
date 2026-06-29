# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Distributed checkpointing with sharded save/load."""

from __future__ import annotations

import io
import os
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

# From Megatron M2620: delegate to central safe_globals registry.
from deepspeed.core.safe_globals import register_safe_globals as _rsg
_rsg()

# Insight I4: versioned checkpoint schema (Megatron ab-3.3)
# Re-export CheckpointManifest and its I/O helpers so callers can do:
#   from deepspeed.core.dist_checkpointing import CheckpointManifest, load_manifest, save_manifest
from deepspeed.core.dist_checkpointing.core import (  # noqa: F401
    CheckpointManifest,
    CheckpointingConfig,
    CheckpointingException,
    load_manifest,
    save_manifest,
    maybe_load_config,
    save_config,
    check_is_distributed_checkpoint,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ShardedTensor
# ---------------------------------------------------------------------------

@dataclass
class ShardedTensor:
    """Tensor with sharding metadata for distributed checkpointing.

    Attributes:
        key:                unique string identifier for the global tensor.
        data:               the local shard's tensor data (may be None for
                            metadata-only operations).
        global_shape:       shape of the full (unsharded) parameter tensor.
        global_offset:      per-dimension offset of this shard inside the
                            global tensor, in number of elements.
        axis_fragmentations: number of shards along each dimension.
        replica_id:         0 for the primary replica; non-zero for copies
                            (e.g. DP replicas).  Only replica 0 is saved.
    """
    key: str
    data: torch.Tensor
    global_shape: Tuple[int, ...]
    global_offset: Tuple[int, ...]
    axis_fragmentations: Tuple[int, ...]
    replica_id: int = 0

    def without_data(self) -> "ShardedTensor":
        """Return a metadata-only copy with data set to None.

        Used for broadcasting sharding descriptors across ranks during
        validation and re-sharding without transmitting tensor payloads.
        """
        return replace(self, data=None)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rank() -> int:
    """Global rank of the current process (0 if not distributed)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _world_size() -> int:
    """Total number of processes (1 if not distributed)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _barrier() -> None:
    """Cross-rank synchronisation barrier (no-op if not distributed)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _shard_filename(rank: int) -> str:
    """Canonical filename for a rank's shard file."""
    return f"shard_{rank:05d}.pt"


def _is_main_replica(replica_id: Union[int, Tuple[int, ...]]) -> bool:
    """Return True only for the primary (replica_id == 0) copy."""
    if isinstance(replica_id, int):
        return replica_id == 0
    return all(r == 0 for r in replica_id)


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------

def save(
    sharded_state_dict: Dict[str, Any],
    checkpoint_dir: str,
    *,
    async_save: bool = False,
) -> Optional[Any]:
    """Save a sharded state dict to checkpoint_dir.

    Each rank serialises its own ShardedTensor shards into a per-rank file
    inside *checkpoint_dir*.  Non-ShardedTensor values (plain tensors,
    scalars, etc.) are saved by rank 0 into a shared ``common.pt`` file so
    that they are available to all ranks on load regardless of topology
    changes.

    Directory layout after save::

        checkpoint_dir/
            common.pt           # non-sharded values, written by rank 0
            shard_00000.pt      # rank 0's ShardedTensor shards
            shard_00001.pt      # rank 1's ShardedTensor shards
            ...
            metadata.json       # topology / format version metadata

    Args:
        sharded_state_dict: state dict whose leaf values may be
            :class:`ShardedTensor` instances or plain Python objects.
        checkpoint_dir:     directory to save into.  Must already exist and
                            be empty (enforced by rank 0).
        async_save:         reserved for future async-IO support; currently
                            ignored (save is always synchronous).

    Returns:
        None (``async_save`` future handle when async is implemented).
    """
    rank = _rank()
    ckpt_path = Path(checkpoint_dir)

    # ------------------------------------------------------------------ #
    # Rank 0 validates the target directory.
    # ------------------------------------------------------------------ #
    if rank == 0:
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint directory does not exist: {ckpt_path}"
            )
        if any(ckpt_path.iterdir()):
            raise RuntimeError(
                f"Checkpoint directory is not empty: {ckpt_path}"
            )

    # All ranks wait until rank 0 has finished the directory check so
    # that subsequent writes from non-zero ranks don't race.
    _barrier()

    # ------------------------------------------------------------------ #
    # Separate ShardedTensors from plain (common) values.
    # ------------------------------------------------------------------ #
    shard_entries: Dict[str, ShardedTensor] = {}
    common_entries: Dict[str, Any] = {}

    def _collect(d: Dict[str, Any], prefix: str = "") -> None:
        for k, v in d.items():
            full_key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            if isinstance(v, dict):
                _collect(v, full_key)
            elif isinstance(v, ShardedTensor):
                shard_entries[full_key] = v
            else:
                common_entries[full_key] = v

    _collect(sharded_state_dict)

    # ------------------------------------------------------------------ #
    # Rank 0 saves the common (non-sharded) state dict.
    # ------------------------------------------------------------------ #
    if rank == 0:
        common_path = ckpt_path / "common.pt"
        torch.save(common_entries, str(common_path))
        logger.debug("rank 0: saved common.pt (%d entries)", len(common_entries))

    # ------------------------------------------------------------------ #
    # Every rank saves its own shards (only for the primary replica).
    # ------------------------------------------------------------------ #
    rank_shards: Dict[str, Dict[str, Any]] = {}
    for key, st in shard_entries.items():
        if not _is_main_replica(st.replica_id):
            continue  # skip non-primary replicas
        rank_shards[key] = {
            "data": st.data,
            "global_shape": st.global_shape,
            "global_offset": st.global_offset,
            "axis_fragmentations": st.axis_fragmentations,
            "replica_id": st.replica_id,
        }

    shard_path = ckpt_path / _shard_filename(rank)
    torch.save(rank_shards, str(shard_path))
    logger.debug(
        "rank %d: saved %s (%d shards)", rank, shard_path.name, len(rank_shards)
    )

    # ------------------------------------------------------------------ #
    # Rank 0 writes the topology / version metadata.
    # ------------------------------------------------------------------ #
    _barrier()  # wait for all shard files to be flushed

    if rank == 0:
        import json
        meta = {
            "format": "neuron_sp_dist_ckpt",
            "version": 1,
            "world_size": _world_size(),
        }
        meta_path = ckpt_path / "metadata.json"
        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2)
        logger.debug("rank 0: saved metadata.json")

    _barrier()  # ensure metadata is visible before returning
    return None


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------

def load(
    sharded_state_dict: Dict[str, Any],
    checkpoint_dir: str,
) -> Dict[str, Any]:
    """Load a sharded state dict, handling resharding if parallelism changed.

    The function reconstructs each global parameter tensor from all the rank
    shard files present in *checkpoint_dir* and then extracts the slice that
    belongs to the *current* rank according to the sharding metadata provided
    in *sharded_state_dict*.  This means the checkpoint can be loaded into a
    run with a different TP / PP topology than the one used to save it.

    Args:
        sharded_state_dict: state dict whose :class:`ShardedTensor` entries
            describe how the current run expects the parameters to be sharded.
            Used as a *map* — only keys present here are loaded.
        checkpoint_dir:     directory written by :func:`save`.

    Returns:
        A plain state dict (no :class:`ShardedTensor` wrappers) ready to be
        passed to ``model.load_state_dict()``.
    """
    rank = _rank()
    ckpt_path = Path(checkpoint_dir)

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {ckpt_path}"
        )

    # ------------------------------------------------------------------ #
    # Load the common (non-sharded) state dict on every rank.
    # ------------------------------------------------------------------ #
    common_path = ckpt_path / "common.pt"
    common_state: Dict[str, Any] = {}
    if common_path.exists():
        common_state = torch.load(str(common_path), map_location="cpu")
        logger.debug("rank %d: loaded common.pt", rank)

    # ------------------------------------------------------------------ #
    # Collect ShardedTensor descriptors from the target state dict.
    # ------------------------------------------------------------------ #
    target_shards: Dict[str, ShardedTensor] = {}

    def _collect_shards(d: Dict[str, Any], prefix: str = "") -> None:
        for k, v in d.items():
            full_key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            if isinstance(v, dict):
                _collect_shards(v, full_key)
            elif isinstance(v, ShardedTensor):
                target_shards[full_key] = v

    _collect_shards(sharded_state_dict)

    # ------------------------------------------------------------------ #
    # Discover all shard files in the checkpoint directory.
    # ------------------------------------------------------------------ #
    shard_files = sorted(ckpt_path.glob("shard_*.pt"))
    if not shard_files:
        raise RuntimeError(
            f"No shard files found in {ckpt_path}. "
            "Is this a distributed checkpoint saved by save()?"
        )

    # ------------------------------------------------------------------ #
    # Reconstruct each global tensor from all saved shards, then extract
    # the slice expected by the current rank.
    # ------------------------------------------------------------------ #
    loaded: Dict[str, torch.Tensor] = {}

    # Build an index: key → list of (global_offset, data) from all saved shards.
    global_index: Dict[str, List[Dict[str, Any]]] = {}
    for sf in shard_files:
        saved_rank_data: Dict[str, Dict[str, Any]] = torch.load(
            str(sf), map_location="cpu"
        )
        for key, entry in saved_rank_data.items():
            if key not in global_index:
                global_index[key] = []
            global_index[key].append(entry)

    for key, target_st in target_shards.items():
        if key not in global_index:
            logger.warning(
                "rank %d: key '%s' not found in checkpoint — skipping", rank, key
            )
            continue

        saved_entries = global_index[key]
        # Infer global shape from the first saved entry's metadata.
        ref_entry = saved_entries[0]
        global_shape: Tuple[int, ...] = tuple(ref_entry["global_shape"])

        # Allocate the full global tensor.
        ref_data: torch.Tensor = ref_entry["data"]
        global_tensor = torch.zeros(global_shape, dtype=ref_data.dtype)

        # Fill in each saved shard.
        for entry in saved_entries:
            shard_data: torch.Tensor = entry["data"]
            offset: Tuple[int, ...] = tuple(entry["global_offset"])
            # Build a multi-dimensional index slice for insertion.
            slices = tuple(
                slice(off, off + sh)
                for off, sh in zip(offset, shard_data.shape)
            )
            global_tensor[slices] = shard_data

        # Extract the slice that this rank needs according to target_st.
        my_slices = tuple(
            slice(off, off + sh)
            for off, sh in zip(target_st.global_offset, global_tensor.shape)
            if sh > 0
        )
        # Handle case where target global_offset / shape differ from saved.
        # Use target_st.global_offset and infer local shape from axis_fragmentations.
        local_slices = []
        for dim_idx, (g_off, g_sh, frag) in enumerate(
            zip(target_st.global_offset, target_st.global_shape, target_st.axis_fragmentations)
        ):
            local_dim_size = g_sh // frag if frag > 0 else g_sh
            local_slices.append(slice(g_off, g_off + local_dim_size))

        # Resize global_tensor to target global_shape if resharding occurred.
        if global_tensor.shape != target_st.global_shape:
            # Pad or crop along each dimension to match expected global shape.
            new_global = torch.zeros(target_st.global_shape, dtype=global_tensor.dtype)
            crop = tuple(
                slice(0, min(a, b))
                for a, b in zip(global_tensor.shape, target_st.global_shape)
            )
            new_global[crop] = global_tensor[crop]
            global_tensor = new_global

        local_tensor = global_tensor[tuple(local_slices)].contiguous()

        # Move to the same device as the target ShardedTensor's data (if available).
        if target_st.data is not None:
            local_tensor = local_tensor.to(device=target_st.data.device,
                                           dtype=target_st.data.dtype)

        loaded[key] = local_tensor
        logger.debug(
            "rank %d: loaded '%s' local_shape=%s from global_shape=%s",
            rank, key, list(local_tensor.shape), list(global_shape),
        )

    # ------------------------------------------------------------------ #
    # Merge common + sharded results.
    # ------------------------------------------------------------------ #
    result: Dict[str, Any] = {}
    result.update(common_state)
    result.update(loaded)
    return result
