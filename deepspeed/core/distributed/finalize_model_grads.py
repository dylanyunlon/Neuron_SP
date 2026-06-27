# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""finalize_model_grads — gradient finalization before optimizer step.

Handles:
  1. DP all-reduce / reduce-scatter of main grads across data-parallel ranks.
  2. All-reduce of word-embedding grads across the first and last PP stages.
  3. All-reduce of sequence-parallel (TP) grads across tensor-parallel ranks.
  4. Per-token loss normalization: scale grads by 1 / num_tokens.

DES-LOC extensions (Algorithm 1, Kx/Ku/Kv):
  - On non-Kx steps ``skip_grad_sync=True`` skips the DP all-reduce but
    still runs embedding and SP collectives so that PP and TP stay in sync.
  - ``desloc_sync_optimizer_state`` triggers Ku/Kv syncs: on a Ku step the
    caller passes ``sync_m1=True`` and on a Kv step ``sync_m2=True``.
    Both flags are forwarded to the optimizer via the config hooks.

Public API:
  finalize_model_grads(model, config, num_tokens, skip_grad_sync,
                       desloc_step, desloc_config)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

import deepspeed.core.parallel_state as parallel_state
from deepspeed.core.model_parallel_config import ModelParallelConfig
from deepspeed.core.desloc_config import DesLocConfig
from deepspeed.core.distributed.distributed_data_parallel import DistributedDataParallel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_main_grad_attr(param: torch.nn.Parameter) -> str:
    """Return ``'main_grad'`` if present, else ``'grad'``."""
    if hasattr(param, 'main_grad'):
        return 'main_grad'
    return 'grad'


def _allreduce_word_embedding_grads(
    model: List[nn.Module],
    config: ModelParallelConfig,
) -> None:
    """All-reduce word-embedding gradients across the first and last PP stages.

    When the embedding table is shared between the input embedding and the
    output linear (lm_head), its gradient must be summed across the first
    and last pipeline stages so that both copies stay in sync.
    """
    if not parallel_state.is_initialized():
        return

    pp_world_size = parallel_state.get_pipeline_model_parallel_world_size()
    if pp_world_size <= 1:
        return

    is_first = parallel_state.is_pipeline_first_stage()
    is_last = parallel_state.is_pipeline_last_stage()

    if not (is_first or is_last):
        return

    # Collect embedding params shared across PP stages.
    grads: List[torch.Tensor] = []
    for model_chunk in model:
        shared_emb_weight = None
        for attr in ('word_embeddings', 'shared_embedding', 'embed_tokens'):
            module = getattr(model_chunk, attr, None)
            if module is None:
                try:
                    for sub in model_chunk.modules():
                        candidate = getattr(sub, attr, None)
                        if candidate is not None and hasattr(candidate, 'weight'):
                            module = candidate
                            break
                except Exception:
                    pass
            if module is not None and hasattr(module, 'weight'):
                shared_emb_weight = module.weight
                break

        if shared_emb_weight is None:
            continue
        if not shared_emb_weight.requires_grad:
            continue
        grad_attr = _get_main_grad_attr(shared_emb_weight)
        grad = getattr(shared_emb_weight, grad_attr, None)
        if grad is not None:
            grads.append(grad)

    if not grads:
        return

    pp_group = parallel_state.get_pipeline_model_parallel_group()
    coalesced = _flatten_dense_tensors(grads)
    torch.distributed.all_reduce(coalesced, group=pp_group)
    for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
        buf.copy_(synced)


def _allreduce_sequence_parallel_grads(
    model: List[nn.Module],
    config: ModelParallelConfig,
) -> None:
    """All-reduce grads for params that use sequence parallelism across TP ranks.

    Params with ``param.sequence_parallel = True`` receive partial gradients
    from each TP rank (because activations are sharded across the sequence
    dimension).  We must sum them across the TP group before the DP all-reduce
    so that every replica sees the correct full-sequence gradient.
    """
    if not parallel_state.is_initialized():
        return
    if not config.sequence_parallel:
        return

    tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
    if tp_world_size <= 1:
        return

    tp_group = parallel_state.get_tensor_model_parallel_group()
    grads: List[torch.Tensor] = []

    for model_chunk in model:
        inner = getattr(model_chunk, '_module', model_chunk)
        for param in inner.parameters():
            if not param.requires_grad:
                continue
            if not getattr(param, 'sequence_parallel', False):
                continue
            grad_attr = _get_main_grad_attr(param)
            grad = getattr(param, grad_attr, None)
            if grad is not None:
                grads.append(grad.data)

    if not grads:
        return

    coalesced = _flatten_dense_tensors(grads)
    torch.distributed.all_reduce(coalesced, group=tp_group)
    for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
        buf.copy_(synced)


def _direct_allreduce_grads(
    model_chunk: nn.Module,
    config: ModelParallelConfig,
) -> None:
    """Fallback: directly all-reduce grads on a raw nn.Module (no DDP buffer)."""
    if not parallel_state.is_initialized():
        return
    dp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
    if torch.distributed.get_world_size(group=dp_group) <= 1:
        return

    inner = getattr(model_chunk, '_module', model_chunk)
    grads: List[torch.Tensor] = []
    for param in inner.parameters():
        if not param.requires_grad:
            continue
        grad_attr = _get_main_grad_attr(param)
        grad = getattr(param, grad_attr, None)
        if grad is not None:
            grads.append(grad.data)

    if not grads:
        return

    dp_world_size = torch.distributed.get_world_size(group=dp_group)
    coalesced = _flatten_dense_tensors(grads)
    torch.distributed.all_reduce(coalesced, group=dp_group)
    coalesced.div_(dp_world_size)
    for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
        buf.copy_(synced)


# ---------------------------------------------------------------------------
# DES-LOC Kx/Ku/Kv conditional sync helpers
# ---------------------------------------------------------------------------

def _desloc_should_sync_grads(
    step: int,
    desloc_config: DesLocConfig,
) -> bool:
    """Return True if this step is a Kx synchronization step.

    DES-LOC Algorithm 1: gradient all-reduce is performed only every Kx steps.
    On intermediate steps, each rank accumulates gradients locally, deferring
    the expensive cross-rank communication to the Kx boundary.

    Args:
        step: Current training step (0-indexed).
        desloc_config: DES-LOC configuration carrying Kx/Ku/Kv periods.

    Returns:
        True on Kx steps (all-reduce should proceed), False otherwise.
    """
    return desloc_config.is_kx_step(step + 1)


def _desloc_log_sync_state(
    step: int,
    is_kx: bool,
    is_ku: bool,
    is_kv: bool,
    skip_grad: bool,
) -> None:
    """Log DES-LOC sync decisions for debugging."""
    if step < 10 or is_kx or is_ku or is_kv:
        logger.info(
            "[DES-LOC finalize_model_grads] step=%d "
            "Kx=%s Ku=%s Kv=%s grad_sync=%s",
            step + 1,
            "SYNC" if is_kx else "skip",
            "SYNC" if is_ku else "skip",
            "SYNC" if is_kv else "skip",
            "SKIP" if skip_grad else "RUN",
        )


def _desloc_sync_optimizer_moments(
    model: List[nn.Module],
    config: ModelParallelConfig,
    is_ku: bool,
    is_kv: bool,
) -> None:
    """Synchronize optimizer first/second moments across DP ranks on Ku/Kv steps.

    On a Ku step: broadcast m1 (exp_avg) from rank-0 to all DP ranks.
    On a Kv step: broadcast m2 (exp_avg_sq) from rank-0 to all DP ranks.

    The optimizer is accessed through ``config.optimizer`` if that attribute
    is present (set by the training engine after wrapping the model).  If the
    attribute is absent (unit-test / non-DES-LOC path), this function is a
    no-op and the log warning is emitted once.

    Args:
        model: List of model chunks.
        config: Model parallel config — may carry a ``desloc_optimizer`` ref.
        is_ku: True on Ku synchronization steps.
        is_kv: True on Kv synchronization steps.
    """
    if not (is_ku or is_kv):
        return

    if not parallel_state.is_initialized():
        return

    optimizer = getattr(config, 'desloc_optimizer', None)
    if optimizer is None:
        logger.debug(
            "[DES-LOC] config.desloc_optimizer not set; "
            "skipping Ku/Kv moment synchronization."
        )
        return

    try:
        dp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
    except AssertionError:
        dp_group = parallel_state.get_data_parallel_group(with_context_parallel=False)

    src_rank = torch.distributed.get_global_rank(dp_group, 0)

    for group in getattr(optimizer, 'param_groups', []):
        for param in group.get('params', []):
            state = optimizer.state.get(param)
            if state is None:
                continue

            if is_ku:
                m1 = state.get('exp_avg')
                if m1 is not None:
                    torch.distributed.broadcast(m1, src=src_rank, group=dp_group)
                    logger.debug("[DES-LOC] Ku: broadcast exp_avg for param shape=%s", list(m1.shape))

            if is_kv:
                m2 = state.get('exp_avg_sq')
                if m2 is not None:
                    torch.distributed.broadcast(m2, src=src_rank, group=dp_group)
                    logger.debug("[DES-LOC] Kv: broadcast exp_avg_sq for param shape=%s", list(m2.shape))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def finalize_model_grads(
    model: List[nn.Module],
    config: ModelParallelConfig,
    num_tokens: Optional[torch.Tensor] = None,
    skip_grad_sync: bool = False,
    *,
    desloc_step: Optional[int] = None,
    desloc_config: Optional[DesLocConfig] = None,
) -> None:
    """Finalize gradients before optimizer step.

    Handles:

    1. **DP all-reduce** of main grads across data-parallel ranks.
       - DES-LOC: skipped on non-Kx steps (``skip_grad_sync=True``).
    2. **Embedding grad all-reduce** across first/last PP stages.
       - Always performed — embedding must stay in sync regardless of Kx.
    3. **Sequence-parallel grad all-reduce** across TP ranks.
    4. **Per-token loss normalization**: scale grads by ``1 / num_tokens``.
    5. **DES-LOC Ku/Kv moment sync**: if ``desloc_config`` is provided,
       broadcast optimizer moments on Ku and Kv steps respectively.

    Args:
        model: List of model chunks (for virtual pipeline parallelism).
        config: Model parallel config.
        num_tokens: Token count tensor for per-token loss scaling.
        skip_grad_sync: If True, skip the main DP gradient all-reduce.
                        Typically set by the training loop on non-Kx steps.
        desloc_step: Current training step (0-indexed) for DES-LOC logging.
                     If None, DES-LOC moment sync is skipped.
        desloc_config: DES-LOC configuration.  When provided together with
                       ``desloc_step``, triggers Ku/Kv moment synchronization
                       and detailed sync-state logging.
    """
    # ------------------------------------------------------------------
    # Resolve DES-LOC step flags
    # ------------------------------------------------------------------
    _is_kx = False
    _is_ku = False
    _is_kv = False

    if desloc_step is not None and desloc_config is not None and desloc_config.enabled:
        _is_kx = desloc_config.is_kx_step(desloc_step + 1)
        _is_ku = desloc_config.is_ku_step(desloc_step + 1)
        _is_kv = desloc_config.is_kv_step(desloc_step + 1)
        # Reconcile: if caller did not pass skip_grad_sync but DES-LOC says
        # this is not a Kx step, honour DES-LOC and set the skip flag.
        if not skip_grad_sync and not _is_kx:
            skip_grad_sync = True

        _desloc_log_sync_state(desloc_step, _is_kx, _is_ku, _is_kv, skip_grad_sync)

    # ------------------------------------------------------------------
    # 1. Main DP all-reduce / reduce-scatter across data-parallel ranks.
    #    On non-Kx steps (skip_grad_sync=True) we skip the collective but
    #    still call finish_grad_sync so that each DDP module can clean
    #    up its bookkeeping state.
    # ------------------------------------------------------------------
    if config.timers is not None:
        config.timers('all-grads-sync', log_level=1).start(
            barrier=getattr(config, 'barrier_with_L1_time', False)
        )

    for model_chunk in model:
        if isinstance(model_chunk, DistributedDataParallel):
            if skip_grad_sync:
                # Tell each bucket group to skip the collective.
                for bg in (model_chunk.bucket_groups +
                           model_chunk.expert_parallel_bucket_groups):
                    bg._skip_sync = True
            # finish_grad_sync respects _skip_sync internally.
            model_chunk.finish_grad_sync(force_all_reduce=False)
        else:
            # Raw nn.Module or wrapped model without DDP — fall back to
            # direct parameter iteration (no-op when skip requested).
            if not skip_grad_sync:
                _direct_allreduce_grads(model_chunk, config)

    if config.timers is not None:
        config.timers('all-grads-sync').stop()

    # ------------------------------------------------------------------
    # 2. All-reduce embedding grads across PP stages.
    #    This is always required regardless of skip_grad_sync because the
    #    embedding params on the first and last PP stages must stay in sync.
    # ------------------------------------------------------------------
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce', log_level=1).start(
            barrier=getattr(config, 'barrier_with_L1_time', False)
        )
    _allreduce_word_embedding_grads(model, config)
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce').stop()

    # ------------------------------------------------------------------
    # 3. All-reduce sequence-parallel grads across TP ranks.
    # ------------------------------------------------------------------
    if config.timers is not None:
        config.timers('non-tensor-parallel-grads-all-reduce', log_level=1).start(
            barrier=getattr(config, 'barrier_with_L1_time', False)
        )
    _allreduce_sequence_parallel_grads(model, config)
    if config.timers is not None:
        config.timers('non-tensor-parallel-grads-all-reduce').stop()

    # ------------------------------------------------------------------
    # 4. Per-token loss normalization: scale gradients by 1/num_tokens.
    #    num_tokens lives on the last PP stage; broadcast to all stages,
    #    then all-reduce across DP ranks so every rank uses the global count.
    # ------------------------------------------------------------------
    if num_tokens is not None:
        if parallel_state.is_initialized():
            pp_group = parallel_state.get_pipeline_model_parallel_group()
            last_rank = parallel_state.get_pipeline_model_parallel_last_rank()
            torch.distributed.broadcast(num_tokens, src=last_rank, group=pp_group)

            # Prefer DP-with-CP group; fall back to plain DP when CP is absent.
            try:
                dp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
            except AssertionError:
                dp_group = parallel_state.get_data_parallel_group(with_context_parallel=False)
            torch.distributed.all_reduce(num_tokens, group=dp_group)

        safe_num_tokens = torch.clamp(num_tokens, min=1)
        scaling = 1.0 / safe_num_tokens

        for model_chunk in model:
            if isinstance(model_chunk, DistributedDataParallel):
                model_chunk.scale_gradients(scaling)
            else:
                for param in model_chunk.parameters():
                    if not param.requires_grad:
                        continue
                    grad_attr = _get_main_grad_attr(param)
                    grad = getattr(param, grad_attr, None)
                    if grad is not None:
                        grad.mul_(scaling)

    # ------------------------------------------------------------------
    # 5. DES-LOC Ku/Kv: synchronize optimizer first/second moments.
    #    These expensive but infrequent syncs keep Adam state consistent
    #    across heterogeneous DP replicas.
    # ------------------------------------------------------------------
    if desloc_step is not None and desloc_config is not None and desloc_config.enabled:
        _desloc_sync_optimizer_moments(model, config, _is_ku, _is_kv)
