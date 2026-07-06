# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Save / load DistributedDataParallel model state with ShardedTensor.

Addresses issue #121: implement save/load with ShardedTensor.

Design
------
Each rank saves only the slice of every parameter that it *owns* — for
tensor-parallel (TP) axes the local shard differs across TP ranks; for
data-parallel (DP) replicas every rank holds the same values, so only
replica-0 writes (``replica_id > 0`` shards are skipped by the storage
layer).

The ``dist_checkpointing`` infrastructure already provides:
- ``ShardedTensor``  — wraps a local tensor with full sharding metadata.
- ``save()``         — each rank writes its own ``shard_NNNNN.pt`` file.
- ``load()``         — reconstructs the global tensor from all shard files
                        and slices out the local share for the current rank.

This module adds the *bridge*:
1. ``sharded_state_dict(ddp)``  — iterate DDP model params → ShardedTensor.
2. ``save_checkpoint(ddp, …)``  — call dist_checkpointing.save.
3. ``load_checkpoint(ddp, …)``  — call dist_checkpointing.load, then
                                    ``load_state_dict`` back into the model.

ShardedTensor construction follows Megatron's convention:
- TP-parallel params (``param.tensor_model_parallel == True``) are sharded
  along ``param.partition_dim`` with ``tp_size`` fragments.
- All other params are *replicated* across TP ranks; only DP-rank 0 writes
  (``replica_id = (0, tp_rank, dp_rank)``).

PP (pipeline-parallel) offsets are *prepended* as extra dimensions when a
``sharded_offsets`` tuple is supplied by the pipeline schedule (same as
Megatron ``ShardedTensor.from_rank_offsets`` ``prepend_axis_num``).

Megatron references
-------------------
- megatron.core.transformer.utils.make_tp_sharded_tensor_for_checkpoint
- megatron.core.transformer.utils.make_sharded_tensor_for_checkpoint
- megatron.core.dist_checkpointing.serialization.save / load
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy parallel_state helpers (guards against import-before-init)
# ---------------------------------------------------------------------------

def _ps_get(fn_name: str, *args, default=None, **kwargs):
    """Call parallel_state.fn_name(*args, **kwargs) or return *default*."""
    try:
        import deepspeed.core.parallel_state as _ps
        if not _ps.is_initialized():
            return default
        fn = getattr(_ps, fn_name, None)
        if fn is not None:
            return fn(*args, **kwargs)
    except Exception:
        pass
    return default


def _tp_rank() -> int:
    return _ps_get("get_tensor_model_parallel_rank", default=0) or 0


def _tp_size() -> int:
    return _ps_get("get_tensor_model_parallel_world_size", default=1) or 1


def _dp_rank() -> int:
    return _ps_get("get_data_parallel_rank", default=0) or 0


def _dp_size() -> int:
    return _ps_get("get_data_parallel_world_size", default=1) or 1


def _pp_rank() -> int:
    return _ps_get("get_pipeline_model_parallel_rank", default=0) or 0


def _pp_size() -> int:
    return _ps_get("get_pipeline_model_parallel_world_size", default=1) or 1


# ---------------------------------------------------------------------------
# ShardedTensor import — from the deepspeed.core.dist_checkpointing package
# ---------------------------------------------------------------------------

try:
    from deepspeed.core.dist_checkpointing.mapping import ShardedTensor
except ImportError:
    # Fallback: use the ShardedTensor defined in dist_checkpointing/__init__.py
    from deepspeed.core.dist_checkpointing import ShardedTensor  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Core: build a sharded state dict for one module
# ---------------------------------------------------------------------------

def _make_sharded_tensor(
    param: torch.Tensor,
    key: str,
    sharded_offsets: Tuple[Tuple[int, int, int], ...] = (),
) -> ShardedTensor:
    """Wrap *param* as a ShardedTensor encoding its TP/DP sharding.

    Args:
        param:           the parameter tensor (already the *local* shard on
                         this rank for TP parameters).
        key:             unique string key for the global parameter.
        sharded_offsets: PP-level ``(axis, rank_offset, num_ranks)`` tuples
                         prepended as extra dimensions (Megatron convention).

    Returns:
        A :class:`ShardedTensor` ready for use with
        ``dist_checkpointing.save`` / ``dist_checkpointing.load``.
    """
    tp_rank = _tp_rank()
    tp_size = _tp_size()
    dp_rank = _dp_rank()

    # DP replicas: only replica 0 writes — signal with replica_id.
    # (0, tp_rank, dp_rank) is Megatron's convention: first 0 = PP replica 0.
    replica_id: Union[int, Tuple[int, ...]] = (0, tp_rank, dp_rank)

    # Build rank offsets (axis, rank_offset, num_ranks) for ShardedTensor.
    rank_offsets = list(sharded_offsets)  # copy PP offsets first

    prepend_axis_num = len(sharded_offsets)

    is_tp_param = getattr(param, "tensor_model_parallel", False) and tp_size > 1
    if is_tp_param:
        tp_axis = getattr(param, "partition_dim", 0)
        tp_stride = getattr(param, "partition_stride", 1)
        # Axis in the combined (prepended+local) space.
        combined_axis = tp_axis + prepend_axis_num
        rank_offsets.append((combined_axis, tp_rank, tp_size))

    return ShardedTensor.from_rank_offsets(
        key,
        param,
        *rank_offsets,
        replica_id=replica_id,
        prepend_axis_num=prepend_axis_num,
    )


def sharded_state_dict(
    model: nn.Module,
    prefix: str = "",
    sharded_offsets: Tuple[Tuple[int, int, int], ...] = (),
    *,
    keep_vars: bool = True,
) -> Dict[str, ShardedTensor]:
    """Return a state dict whose values are :class:`ShardedTensor` objects.

    Iterates over all named parameters (and buffers) of *model* and wraps
    each one with a :class:`ShardedTensor` that encodes its TP/DP sharding
    metadata.  TP-sharded parameters (``param.tensor_model_parallel == True``)
    are saved as proper partial shards; all other parameters are replicated
    across TP ranks and only the primary DP replica writes.

    This mirrors Megatron's
    ``MegatronModule.sharded_state_dict`` /
    ``make_tp_sharded_tensor_for_checkpoint`` pattern but operates directly
    on an arbitrary ``nn.Module`` (no MegatronModule base required).

    Args:
        model:           the model (or DDP wrapper's ``.module``) to snapshot.
        prefix:          prepended to every key in the result dict.
        sharded_offsets: pipeline-parallel ``(axis, rank_offset, num_ranks)``
                         offsets, passed along to every ShardedTensor.
        keep_vars:       if ``True``, tensors keep ``requires_grad``; this
                         matches torch's ``state_dict(keep_vars=True)``
                         convention used for checkpointing.

    Returns:
        Dict mapping qualified parameter names → :class:`ShardedTensor`.
    """
    sharded_sd: Dict[str, ShardedTensor] = {}

    for name, param in model.named_parameters():
        if not keep_vars:
            param = param.detach()
        key = f"{prefix}{name}" if prefix else name
        sharded_sd[key] = _make_sharded_tensor(param, key, sharded_offsets)

    for name, buf in model.named_buffers():
        key = f"{prefix}{name}" if prefix else name
        if key in sharded_sd:
            continue  # named_parameters already covered this (e.g. bias as param)
        sharded_sd[key] = _make_sharded_tensor(buf, key, sharded_offsets)

    return sharded_sd


# ---------------------------------------------------------------------------
# DDP-level helpers
# ---------------------------------------------------------------------------

def _unwrap_module(model: nn.Module) -> nn.Module:
    """Return the unwrapped inner module from a DDP wrapper if applicable."""
    # DistributedDataParallel stores its inner module as ._module.
    if hasattr(model, "_module"):
        return model._module  # type: ignore[attr-defined]
    # torch.nn.parallel.DistributedDataParallel
    if hasattr(model, "module"):
        return model.module  # type: ignore[attr-defined]
    return model


# ---------------------------------------------------------------------------
# save_checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    checkpoint_dir: str,
    *,
    prefix: str = "",
    sharded_offsets: Tuple[Tuple[int, int, int], ...] = (),
    extra_state: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model parameters as a distributed checkpoint using ShardedTensor.

    Each rank writes its own shard file; rank 0 additionally writes
    ``common.pt`` for non-sharded scalars and ``metadata.json`` for topology
    information.

    Directory layout produced::

        checkpoint_dir/
            shard_00000.pt      ← rank 0 shards
            shard_00001.pt      ← rank 1 shards
            ...
            common.pt           ← non-sharded data (written by rank 0)
            metadata.json       ← topology / format version

    Args:
        model:            the model (or DDP wrapper) to checkpoint.
        checkpoint_dir:   target directory (must already exist and be empty).
        prefix:           optional key prefix inserted into all state-dict
                          keys (e.g. ``"model."``).
        sharded_offsets:  pipeline-parallel ``(axis, rank_offset, num_ranks)``
                          tuples prepended to every ShardedTensor.
        extra_state:      additional plain-Python / non-sharded data to save
                          alongside the model parameters (e.g. iteration
                          count, RNG state).  Merged into the common.pt dict.

    Raises:
        FileNotFoundError: if *checkpoint_dir* does not exist (rank 0 check).
        RuntimeError:      if *checkpoint_dir* is non-empty (rank 0 check).
    """
    from deepspeed.core.dist_checkpointing import save as _dc_save

    inner = _unwrap_module(model)
    sharded_sd = sharded_state_dict(inner, prefix=prefix, sharded_offsets=sharded_offsets)

    # Inject extra_state as plain (non-sharded) entries so they land in common.pt.
    state_dict_to_save: Dict[str, Any] = {}
    state_dict_to_save.update(sharded_sd)  # ShardedTensor values
    if extra_state:
        for k, v in extra_state.items():
            flat_key = f"__extra__.{k}"
            state_dict_to_save[flat_key] = v  # plain value → goes to common.pt

    pp_rank = _pp_rank()
    tp_rank = _tp_rank()
    dp_rank = _dp_rank()
    logger.info(
        "save_checkpoint: dir=%s PP=%d TP=%d DP=%d shards=%d extra=%d",
        checkpoint_dir,
        pp_rank,
        tp_rank,
        dp_rank,
        len(sharded_sd),
        len(extra_state) if extra_state else 0,
    )

    _dc_save(state_dict_to_save, checkpoint_dir)
    logger.info("save_checkpoint: complete (rank %d)", _dp_rank())


# ---------------------------------------------------------------------------
# load_checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(
    model: nn.Module,
    checkpoint_dir: str,
    *,
    prefix: str = "",
    sharded_offsets: Tuple[Tuple[int, int, int], ...] = (),
    strict: bool = True,
) -> Dict[str, Any]:
    """Load a distributed checkpoint into *model* using ShardedTensor resharding.

    Reconstructs the full global parameter tensor from all rank shard files,
    then extracts the local slice appropriate for the current rank's TP/PP/DP
    position.  This means the checkpoint can be loaded into a run with a
    **different** TP / PP topology than the one used to save it (resharding
    is handled transparently by :func:`~deepspeed.core.dist_checkpointing.load`).

    Args:
        model:            the model (or DDP wrapper) to load parameters into.
        checkpoint_dir:   directory written by :func:`save_checkpoint`.
        prefix:           the same key prefix used when saving.
        sharded_offsets:  the same pipeline-parallel offsets used when saving.
        strict:           if ``True``, ``model.load_state_dict`` is called with
                          ``strict=True`` (default); set ``False`` to allow
                          partial loads (e.g. pretrain → finetune with extra
                          heads).

    Returns:
        A plain dict containing any ``extra_state`` entries that were saved
        alongside the model parameters.

    Raises:
        FileNotFoundError: if *checkpoint_dir* does not exist.
        RuntimeError:      if no shard files are found in *checkpoint_dir*.
    """
    from deepspeed.core.dist_checkpointing import load as _dc_load

    inner = _unwrap_module(model)

    # Build the target sharded_state_dict that describes *this* run's sharding.
    # dist_checkpointing.load will use it to slice the correct local portion
    # from the globally-reconstructed tensor.
    target_sharded_sd = sharded_state_dict(inner, prefix=prefix, sharded_offsets=sharded_offsets)

    logger.info(
        "load_checkpoint: dir=%s PP=%d TP=%d DP=%d target_keys=%d",
        checkpoint_dir,
        _pp_rank(),
        _tp_rank(),
        _dp_rank(),
        len(target_sharded_sd),
    )

    # dc_load returns a plain state dict with ShardedTensor entries replaced
    # by concrete tensors sliced for this rank, plus any common.pt entries.
    loaded_sd = _dc_load(target_sharded_sd, checkpoint_dir)

    # Separate model-param entries from extra_state entries.
    model_sd: Dict[str, torch.Tensor] = {}
    extra_state: Dict[str, Any] = {}
    for k, v in loaded_sd.items():
        if k.startswith("__extra__."):
            extra_state[k[len("__extra__."):]] = v
        else:
            # Strip the prefix to get the bare parameter name for load_state_dict.
            bare_key = k[len(prefix):] if prefix and k.startswith(prefix) else k
            model_sd[bare_key] = v

    # Load tensors into the model — move to the same device as current params.
    device = next(inner.parameters(), torch.tensor(0)).device
    model_sd_on_device: Dict[str, torch.Tensor] = {}
    for k, v in model_sd.items():
        if isinstance(v, torch.Tensor):
            model_sd_on_device[k] = v.to(device=device)
        else:
            model_sd_on_device[k] = v

    missing, unexpected = inner.load_state_dict(model_sd_on_device, strict=strict)

    if missing:
        logger.warning(
            "load_checkpoint: %d missing keys: %s",
            len(missing),
            missing[:10],
        )
    if unexpected:
        logger.warning(
            "load_checkpoint: %d unexpected keys: %s",
            len(unexpected),
            unexpected[:10],
        )

    logger.info(
        "load_checkpoint: loaded %d params (rank %d)",
        len(model_sd_on_device),
        _dp_rank(),
    )
    return extra_state
