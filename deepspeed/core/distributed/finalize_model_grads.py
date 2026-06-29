# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""finalize_model_grads — gradient finalization before optimizer step.

Evolution summary (ported from Megatron-LM finalize_model_grads.py, 15 commits):
  M2282 (76622edf3): pgs_collection — ProcessGroupCollection; pg_collection
      param in finalize_model_grads; tp/pp/embd/pos_embd/dp_cp group routing.
  M2286 (ca9797e95): Revert pgs_collection.
  M2293 (72d23540d): Add global aux loss support (MoE router).
  M2298 (696977fc3): Fix Megatron-FSDP logging; remove extra config.
  M2301 (8c1a3f5df): Replay pgs_collection.
  M2363 (c17361575): CUDA-graph code refactor; move collective dispatch logic.
  M2364 (793b89a6d): Revert cuda-graph refactor.
  M2366 (98b6f0e81): Replay cuda-graph refactor.
  M3009 (5247a1f46): MTP layers in standalone PP stages; MTP embedding handling.
  M3087 (dbde759da): Save wgrads/dgrads capability; wgrad accumulation.
  M3223 (347ad215a): Nano QAT/D fix with SFT tokenizer and datasets.
  M3871 (2d862fe0c): Flextron router grad sync across PP stages
      (_allreduce_router_grads + flextron_router_pp_sync attribute).
  M3981 (aa786b72c): Thread custom process groups through MoE grad finalization
      (tp_dp_cp group for _update_router_expert_bias).
  M4041 (67b2f3878): FSDP full-iteration CUDA graphability — conditional
      param.grad dereferencing in finalize (param.grad = None for FSDP path).
  M4173 (277c4f804): Offline logits-based knowledge distillation support
      (knowledge-distillation teacher gradient handling).

DES-LOC extensions (Algorithm 1):
  - finalize_model_grads(... desloc_step, desloc_config):
    * Non-Kx steps: skip DP all-reduce (skip_grad_sync=True) but still
      run embedding + SP + MoE collectives so PP and TP stay in sync.
    * Kx steps: full DP all-reduce + optional Ku/Kv moment sync.
  - _desloc_should_sync_grads(step, config) → bool: Kx predicate.
  - _desloc_sync_optimizer_moments(model, config, is_ku, is_kv): broadcast
    Adam exp_avg (Ku) and exp_avg_sq (Kv) from rank-0 across DP group.

Public API:
  finalize_model_grads(model, config, num_tokens, skip_grad_sync,
                       desloc_step, desloc_config, pg_collection,
                       force_all_reduce)

  _allreduce_word_embedding_grads
  _allreduce_position_embedding_grads
  _allreduce_non_tensor_model_parallel_grads
  _allreduce_layernorm_grads  (alias for legacy tests)
  _allreduce_conditional_embedding_grads
  _allreduce_router_grads
  reset_model_temporary_tensors
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

import deepspeed.core.parallel_state as parallel_state
from deepspeed.core.model_parallel_config import ModelParallelConfig
from deepspeed.core.desloc_config import DesLocConfig
from deepspeed.core.distributed.distributed_data_parallel import DistributedDataParallel
from deepspeed.core.optimizer.distrib_optimizer import get_effective_grad

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Megatron-compat imports
# ---------------------------------------------------------------------------

try:
    from torch.distributed._tensor import DTensor, distribute_tensor
    HAVE_DTENSOR = True
except ImportError:
    HAVE_DTENSOR = False

try:
    from megatron.core.pipeline_parallel.utils import (
        get_pp_last_rank,
        is_pp_first_stage,
        is_pp_last_stage,
    )
except ImportError:
    def get_pp_last_rank(pp_group):
        return torch.distributed.get_process_group_ranks(pp_group)[-1]

    def is_pp_first_stage(pp_group=None):
        if pp_group is None:
            return True
        return pp_group.rank() == 0

    def is_pp_last_stage(pp_group=None):
        if pp_group is None:
            return True
        return pp_group.rank() == pp_group.size() - 1

try:
    from megatron.core.utils import get_attr_wrapped_model, get_model_config, get_pg_size
except ImportError:
    def get_attr_wrapped_model(model, attr, return_model_obj=False):
        inner = getattr(model, attr, None)
        if inner is None:
            inner = getattr(model, '_module', model)
            inner = getattr(inner, attr, None)
        if return_model_obj:
            return model
        return inner

    def get_model_config(model):
        for attr in ('config', '_config', 'module'):
            v = getattr(model, attr, None)
            if v is not None and hasattr(v, 'sequence_parallel'):
                return v
        return getattr(model, 'config', None)

    def get_pg_size(pg):
        if pg is None:
            return 1
        return torch.distributed.get_world_size(group=pg)

try:
    from megatron.core.transformer.moe.moe_utils import get_updated_expert_bias
except ImportError:
    get_updated_expert_bias = None


# ---------------------------------------------------------------------------
# DTensor helpers (M2298 FSDP / DTensor support)
# ---------------------------------------------------------------------------

def _get_main_grad_attr(param: torch.nn.Parameter) -> str:
    """Return 'main_grad' if present, else 'grad'."""
    if hasattr(param, 'main_grad'):
        return 'main_grad'
    return 'grad'


def _unshard_if_dtensor(tensor) -> torch.Tensor:
    """Unshard the input if it is a DTensor; otherwise return unchanged.

    Introduced in M2298 for Megatron-FSDP / DTensor compatibility.
    """
    if HAVE_DTENSOR and isinstance(tensor, DTensor):
        unsharded = tensor.full_tensor()
        for k, v in vars(tensor).items():
            setattr(unsharded, k, v)
        return unsharded
    return tensor


def _reshard_if_dtensor(tensor_to_shard: torch.Tensor, reference) -> Union[torch.Tensor, object]:
    """Reshard tensor_to_shard to match reference's sharding if reference is a DTensor.

    Args:
        tensor_to_shard: The reduced tensor.
        reference: The original (possibly DTensor) gradient.
    """
    if HAVE_DTENSOR and isinstance(reference, DTensor):
        sharded = distribute_tensor(
            tensor_to_shard,
            device_mesh=reference.device_mesh,
            placements=reference.placements,
        )
        for k, v in vars(reference).items():
            setattr(sharded, k, v)
        return sharded
    return reference


# ---------------------------------------------------------------------------
# Conditional embedding grad sync (M2974: DiT-style cond embedders across PP)
# ---------------------------------------------------------------------------

def _allreduce_conditional_embedding_grads(
    model: List[nn.Module],
    config: ModelParallelConfig,
    pp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> None:
    """All-reduce conditional embedding grads across PP stages (M2974 / M3009).

    Reduces grads for parameters marked with pipeline_parallel=True across
    the PP group to keep timestep/FPS/label embedders in sync for DiT-style
    models with replicated embedders on each PP/VPP rank.

    Args:
        model: List of model chunks.
        config: Transformer / model-parallel config.
        pp_group: Pipeline parallel process group.
    """
    if pp_group is None:
        pp_group = _get_pp_group()

    if get_pg_size(pp_group) <= 1:
        return
    if not getattr(config, 'has_cond_embedder', False):
        return

    grads_dict: Dict[str, List[torch.Tensor]] = {}
    for model_chunk in model:
        for name, param in _named_parameters(model_chunk):
            if param.requires_grad and getattr(param, 'pipeline_parallel', False):
                # DES-LOC M4145 followup: unified grad access via get_effective_grad.
                grad = get_effective_grad(param)
                if name in grads_dict:
                    grads_dict[name][0].add_(grad)
                    grads_dict[name].append(grad)
                else:
                    grads_dict[name] = [grad]

    if not grads_dict:
        return

    grads = [pg[0] for pg in grads_dict.values()]
    coalesced = _flatten_dense_tensors(grads)
    torch.distributed.all_reduce(coalesced, group=pp_group)
    for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
        buf.copy_(synced)

    for pgs in grads_dict.values():
        for g in pgs[1:]:
            g.copy_(pgs[0])


# ---------------------------------------------------------------------------
# Shared word-embedding weight getter
# ---------------------------------------------------------------------------

def _get_shared_word_embedding_weight(
    model_module: nn.Module,
    config: ModelParallelConfig,
) -> Optional[torch.nn.Parameter]:
    """Return the shared word-embedding weight if duplicated across PP stages.

    Args:
        model_module: The pre-process model module.
        config: Model config.

    Returns:
        The shared embedding/output weight or None.
    """
    share = getattr(model_module, 'share_embeddings_and_output_weights', False)
    mtp_layers = getattr(config, 'mtp_num_layers', 0)
    if share or mtp_layers:
        fn = getattr(model_module, 'shared_embedding_or_output_weight', None)
        if fn is not None:
            return fn()
    return None


def _get_position_embedding_weight(model_module: nn.Module) -> torch.nn.Parameter:
    """Return position-embedding weight from model module.

    Args:
        model_module: Module owning position embeddings.
    """
    return getattr(model_module, 'position_embeddings').weight


# ---------------------------------------------------------------------------
# Unified embedding grad all-reduce helper
# ---------------------------------------------------------------------------

def _allreduce_embedding_grad(
    model: List[nn.Module],
    embd_group: torch.distributed.ProcessGroup,
    pp_group: torch.distributed.ProcessGroup,
    weight_getter: Callable[[nn.Module], Optional[torch.nn.Parameter]],
    skip_if_none: bool = True,
    config: ModelParallelConfig = None,
) -> None:
    """Unified helper to all-reduce embedding parameters across PP stages.

    Handles FSDP (M4041 _local_tensor) and DTensor (M2298) param gradients.

    Evolution:
      M2282: Initial embd_group parameter.
      M2298: FSDP _local_tensor handling.
      M3009: MTP layer embedding routing (model[-1] for MTP).
      M4041: Conditional param.grad dereferencing for CUDA-graph FSDP.

    Args:
        model: List of model chunks (PP/VPP).
        embd_group: All-reduce group for embedding grads.
        pp_group: PP group for first/last stage detection.
        weight_getter: Function returning the weight param to reduce.
        skip_if_none: If True, skip silently when weight or grad is None.
        config: Config for MTP/FSDP detection.
    """
    if get_pg_size(embd_group) <= 1:
        return
    try:
        rank_in_group = torch.distributed.get_rank() in \
            torch.distributed.get_process_group_ranks(embd_group)
    except Exception:
        rank_in_group = True

    if not rank_in_group:
        return

    # Determine which model chunk to use (M3009 MTP handling).
    mtp_layers = getattr(config, 'mtp_num_layers', None) if config is not None else None
    if is_pp_first_stage(pp_group):
        model_module = model[0]
    elif is_pp_last_stage(pp_group):
        model_module = model[-1]
    elif mtp_layers is not None and mtp_layers > 0:
        model_module = model[-1]
    else:
        model_module = model[0]

    ddp_config = getattr(model_module, 'ddp_config', None)
    use_fsdp = ddp_config is not None and getattr(ddp_config, 'use_megatron_fsdp', False)

    model_module_inner = get_attr_wrapped_model(model_module, 'pre_process', return_model_obj=True)
    weight = weight_getter(model_module_inner)
    if weight is None and skip_if_none:
        return

    grad_attr = _get_main_grad_attr(weight)
    orig_grad = getattr(weight, grad_attr)

    # M4041: FSDP _local_tensor for CUDA-graph full-iteration graphability.
    if use_fsdp and orig_grad is not None:
        orig_grad = getattr(orig_grad, '_local_tensor', orig_grad)

    grad = _unshard_if_dtensor(orig_grad)
    if grad is None and skip_if_none:
        return

    torch.distributed.all_reduce(grad, group=embd_group)
    setattr(weight, grad_attr, _reshard_if_dtensor(grad, orig_grad))


# ---------------------------------------------------------------------------
# Word-embedding grad all-reduce
# ---------------------------------------------------------------------------

def _allreduce_word_embedding_grads(
    model: List[nn.Module],
    config: ModelParallelConfig,
    embd_group: Optional[torch.distributed.ProcessGroup] = None,
    pp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> None:
    """All-reduce word-embedding grads across first and last PP stages.

    Ensures word_embeddings parameters stay in sync when shared between the
    input embedding and the output LM head (tie embeddings).

    Evolution:
      M2282: embd_group param added, falls back to parallel_state.get_embedding_group().
      M3009: MTP layer embedding routing.
      M4041: FSDP _local_tensor grad handling.

    Args:
        model: List of model chunks.
        config: Model config.
        embd_group: Embedding process group (None → look up from parallel_state).
        pp_group: PP process group (None → look up from parallel_state).
    """
    if embd_group is None:
        embd_group = _get_embedding_group()
        if get_pg_size(embd_group) > 1:
            assert pp_group is None, \
                "pp_group must be None when embd_group is looked up automatically"
            pp_group = _get_pp_group()

    _allreduce_embedding_grad(
        model,
        embd_group,
        pp_group,
        partial(_get_shared_word_embedding_weight, config=config),
        config=config,
    )


# ---------------------------------------------------------------------------
# Position-embedding grad all-reduce
# ---------------------------------------------------------------------------

def _allreduce_position_embedding_grads(
    model: List[nn.Module],
    config: ModelParallelConfig,
    pos_emb_group: torch.distributed.ProcessGroup,
    pp_group: torch.distributed.ProcessGroup,
) -> None:
    """All-reduce position_embeddings grad across encoder and decoder stages.

    Ensures position embedding parameters stay in sync across PP stages in
    encoder-decoder architectures (M2282).

    Args:
        model: List of model chunks.
        config: Model config.
        pos_emb_group: Position-embedding process group.
        pp_group: Pipeline parallel process group.
    """
    _allreduce_embedding_grad(
        model, pos_emb_group, pp_group, _get_position_embedding_weight, skip_if_none=False
    )


# ---------------------------------------------------------------------------
# Flextron router grad sync (M3871)
# ---------------------------------------------------------------------------

def _allreduce_router_grads(
    model: List[nn.Module],
    config: ModelParallelConfig,
) -> None:
    """All-reduce Flextron router grads across PP stages (M3871).

    Reduces grads for params with flextron_router_pp_sync=True to ensure
    router parameters stay in sync across virtual PP ranks.

    Args:
        model: List of model chunks.
        config: Model config with flextron flag.
    """
    pp_ws = _get_pp_world_size()
    if pp_ws <= 1:
        return

    grads_dict: Dict[str, List[torch.Tensor]] = {}
    for model_chunk in model:
        for name, param in _named_parameters(model_chunk):
            if param.requires_grad and getattr(param, 'flextron_router_pp_sync', False):
                # DES-LOC M4145 followup: unified grad access via get_effective_grad.
                grad = get_effective_grad(param)
                if name in grads_dict:
                    grads_dict[name][0].add_(grad)
                    grads_dict[name].append(grad)
                else:
                    grads_dict[name] = [grad]

    if not grads_dict:
        return

    grads = [pg[0] for pg in grads_dict.values()]
    coalesced = _flatten_dense_tensors(grads)
    torch.distributed.all_reduce(coalesced, group=_get_pp_group())
    for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
        buf.copy_(synced)

    for pgs in grads_dict.values():
        for g in pgs[1:]:
            g.copy_(pgs[0])


# ---------------------------------------------------------------------------
# Non-tensor-parallel grad all-reduce (sequence parallel + average-TP-domain)
# ---------------------------------------------------------------------------

def _allreduce_non_tensor_model_parallel_grads(
    model: List[nn.Module],
    config: ModelParallelConfig,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> None:
    """All-reduce sequence-parallel and TP-domain-average grads across TP ranks.

    Two categories of params are handled:
      1. sequence_parallel=True: SUM reduction (partial grads from sequence sharding).
      2. average_gradients_across_tp_domain=True: AVG reduction (full grads replicated
         across TP ranks; averaged to prevent scaling by TP size).

    Evolution:
      M2282: tp_group param added.
      M2298: FSDP _local_tensor handling.
      M3009: qk_layernorm handling added.
      M4041: FSDP conditional grad dereferencing.

    Args:
        model: List of model chunks.
        config: Model config with sequence_parallel, qk_layernorm flags.
        tp_group: Tensor parallel process group.
    """
    if tp_group is None:
        tp_group = _get_tp_group()

    if get_pg_size(tp_group) <= 1:
        return

    params_sum: List[torch.nn.Parameter] = []
    grads_sum: List[torch.Tensor] = []
    params_avg: List[torch.nn.Parameter] = []
    grads_avg: List[torch.Tensor] = []

    for model_chunk in model:
        ddp_config = getattr(model_chunk, 'ddp_config', None)
        use_fsdp = ddp_config is not None and getattr(ddp_config, 'use_megatron_fsdp', False)

        for name, param in _named_parameters(model_chunk):
            if not param.requires_grad:
                continue

            if getattr(param, 'average_gradients_across_tp_domain', False):
                grad_attr = _get_main_grad_attr(param)
                grad = getattr(param, grad_attr)
                if grad is None:
                    continue
                params_avg.append(param)
                if use_fsdp:
                    grads_avg.append(getattr(grad, '_local_tensor', grad).data)
                else:
                    grads_avg.append(_unshard_if_dtensor(grad).data)

            elif (
                getattr(config, 'sequence_parallel', False)
                and getattr(param, 'sequence_parallel', False)
            ) or (
                getattr(config, 'qk_layernorm', False)
                and ('q_layernorm' in name or 'k_layernorm' in name)
            ):
                grad_attr = _get_main_grad_attr(param)
                grad = getattr(param, grad_attr)
                if grad is None:
                    continue
                params_sum.append(param)
                if use_fsdp:
                    grads_sum.append(getattr(grad, '_local_tensor', grad).data)
                else:
                    grads_sum.append(_unshard_if_dtensor(grad).data)

    for params, grads, op in [
        (params_sum, grads_sum, torch.distributed.ReduceOp.SUM),
        (params_avg, grads_avg, torch.distributed.ReduceOp.AVG),
    ]:
        if not grads:
            continue
        coalesced = _flatten_dense_tensors(grads)
        torch.distributed.all_reduce(coalesced, op=op, group=tp_group)
        for param, buf, synced in zip(params, grads, _unflatten_dense_tensors(coalesced, grads)):
            buf.copy_(synced)
            grad_attr = _get_main_grad_attr(param)
            orig_grad = getattr(param, grad_attr)
            ddp_config = getattr(
                next(iter([mc for mc in model if _owns_param(mc, param)]), None),
                'ddp_config', None
            )
            use_fsdp = ddp_config is not None and getattr(ddp_config, 'use_megatron_fsdp', False)
            if use_fsdp:
                setattr(param, grad_attr, orig_grad)
            else:
                setattr(param, grad_attr, _reshard_if_dtensor(buf, orig_grad))


# Legacy alias (maintained for unit tests — mcore 0.14 removal target).
_allreduce_layernorm_grads = _allreduce_non_tensor_model_parallel_grads


# ---------------------------------------------------------------------------
# MoE global aux loss tracker reset (M2293)
# ---------------------------------------------------------------------------

def reset_model_temporary_tensors(
    config: ModelParallelConfig,
    model: List[nn.Module],
) -> None:
    """Reset temporary tensors used for MoE routing statistics (M2293).

    Clears per-step local_tokens_per_expert counts and global aux loss
    trackers so that they start fresh for the next training step.
    """
    for model_chunk in model:
        inner = _get_inner(model_chunk)
        for module in inner.modules():
            if getattr(config, 'moe_router_enable_expert_bias', False) and \
                    hasattr(module, 'expert_bias'):
                # From Megatron M2675: router must only accumulate
                # local_tokens_per_expert when torch.is_grad_enabled() is True.
                # Activation recompute runs forward twice; without this guard,
                # counts are doubled, corrupting expert bias load balancing.
                # Any router implementation MUST wrap accumulation as:
                #   if torch.is_grad_enabled():
                #       self.local_tokens_per_expert += routing_map.sum(dim=0)
                module.local_tokens_per_expert.zero_()
            load_type = getattr(config, 'moe_router_load_balancing_type', '')
            if (
                'global_aux_loss' in load_type
                and hasattr(module, 'reset_global_aux_loss_tracker')
            ):
                module.reset_global_aux_loss_tracker()


# ---------------------------------------------------------------------------
# MoE router expert-bias update (M3981)
# ---------------------------------------------------------------------------

def _update_router_expert_bias(
    model: List[nn.Module],
    config: ModelParallelConfig,
    tp_dp_cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> None:
    """Update expert bias for load-balanced routing (M2293 / M3981).

    All-reduces local_tokens_per_expert across TPxCPxDP ranks to compute
    the global token distribution, then updates expert_bias accordingly.

    Args:
        model: List of model chunks.
        config: Model config with expert bias settings.
        tp_dp_cp_group: Combined TP×DP×CP process group (M3981 threading).
    """
    if get_updated_expert_bias is None:
        return

    tokens_list: List[torch.Tensor] = []
    bias_list: List[torch.Tensor] = []

    for model_chunk in model:
        inner = _get_inner(model_chunk)
        for module in inner.modules():
            if (
                hasattr(module, 'expert_bias')
                and module.training
                and not getattr(module, 'frozen_expert_bias', False)
            ):
                tokens_list.append(module.local_tokens_per_expert)
                bias_list.append(module.expert_bias)

    if not bias_list:
        return

    stacked_tokens = torch.stack(tokens_list, dim=0)
    stacked_bias = torch.stack(bias_list, dim=0)
    stacked_updated_bias = get_updated_expert_bias(
        stacked_tokens,
        stacked_bias,
        config.moe_router_bias_update_rate,
        tp_dp_cp_group=tp_dp_cp_group,
    )
    for bias, updated in zip(bias_list, stacked_updated_bias):
        bias.copy_(updated)


# ---------------------------------------------------------------------------
# DES-LOC helpers
# ---------------------------------------------------------------------------

def _desloc_should_sync_grads(step: int, desloc_config: DesLocConfig) -> bool:
    """Return True if this step is a Kx synchronization step.

    DES-LOC Algorithm 1: gradient all-reduce is performed only every Kx steps.
    On intermediate steps, each rank accumulates gradients locally, deferring
    expensive cross-rank communication to the Kx boundary.

    Args:
        step: Current training step (0-indexed from caller).
        desloc_config: DES-LOC configuration carrying Kx period.

    Returns:
        True on Kx steps (collective should proceed), False on local steps.
    """
    return desloc_config.is_kx_step(step + 1)


def _desloc_log_sync_state(
    step: int,
    is_kx: bool,
    is_ku: bool,
    is_kv: bool,
    skip_grad: bool,
) -> None:
    """Log DES-LOC synchronization decision for debugging."""
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

    DES-LOC Algorithm 1 (Ku/Kv sync):
      - Ku step: broadcast exp_avg  (first moment)  from rank-0 across DP group.
      - Kv step: broadcast exp_avg_sq (second moment) from rank-0 across DP group.

    The optimizer is accessed via config.desloc_optimizer (set by training engine
    after wrapping the model). If absent, this function is a no-op.

    Args:
        model: List of model chunks.
        config: Model parallel config — may carry desloc_optimizer reference.
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
    except Exception:
        dp_group = parallel_state.get_data_parallel_group()

    src_rank = torch.distributed.get_global_rank(dp_group, 0)

    for pg in getattr(optimizer, 'param_groups', []):
        for param in pg.get('params', []):
            state = optimizer.state.get(param)
            if state is None:
                continue
            if is_ku:
                m1 = state.get('exp_avg')
                if m1 is not None:
                    torch.distributed.broadcast(m1, src=src_rank, group=dp_group)
                    logger.debug("[DES-LOC] Ku: broadcast exp_avg shape=%s", list(m1.shape))
            if is_kv:
                m2 = state.get('exp_avg_sq')
                if m2 is not None:
                    torch.distributed.broadcast(m2, src=src_rank, group=dp_group)
                    logger.debug("[DES-LOC] Kv: broadcast exp_avg_sq shape=%s", list(m2.shape))


# ---------------------------------------------------------------------------
# Direct all-reduce fallback (non-DDP model chunks)
# ---------------------------------------------------------------------------

def _direct_allreduce_grads(
    model_chunk: nn.Module,
    config: ModelParallelConfig,
) -> None:
    """Fallback: directly all-reduce grads on a raw nn.Module without DDP buffer."""
    if not parallel_state.is_initialized():
        return
    try:
        dp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
    except Exception:
        dp_group = parallel_state.get_data_parallel_group()
    if torch.distributed.get_world_size(group=dp_group) <= 1:
        return

    inner = _get_inner(model_chunk)
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

    dp_ws = torch.distributed.get_world_size(group=dp_group)
    coalesced = _flatten_dense_tensors(grads)
    torch.distributed.all_reduce(coalesced, group=dp_group)
    coalesced.div_(dp_ws)
    for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
        buf.copy_(synced)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def finalize_model_grads(
    model: List[nn.Module],
    num_tokens: Optional[torch.Tensor] = None,
    pg_collection=None,
    force_all_reduce: Optional[bool] = False,
    config: Optional[ModelParallelConfig] = None,
    skip_grad_sync: bool = False,
    *,
    desloc_step: Optional[int] = None,
    desloc_config: Optional[DesLocConfig] = None,
) -> None:
    """Finalize model gradients before optimizer step.

    Handles all five gradient finalization stages and DES-LOC Kx/Ku/Kv gating:

    1. **DP all-reduce / reduce-scatter** across data-parallel ranks.
       - DES-LOC: skipped on non-Kx steps (skip_grad_sync=True).

    2. **Conditional embedding grad all-reduce** (DiT-style models with
       replicated embedders on each PP/VPP rank).

    3. **Flextron router grad all-reduce** (when config.flextron=True).

    4. **Non-tensor-parallel grad all-reduce**: sequence-parallel (SUM) and
       average-TP-domain (AVG) grads across TP ranks.

    5. **Embedding grad all-reduce**: word embeddings and position embeddings
       across first and last PP stages.

    6. **MoE expert bias update** (when moe_router_enable_expert_bias=True).

    7. **Temporary tensor reset** (aux loss trackers, tokens_per_expert).

    8. **Per-token loss normalization**: scale grads by 1/num_tokens.

    9. **DES-LOC Ku/Kv moment sync**: broadcast Adam moments on Ku/Kv steps.

    Evolution:
      M2282: pg_collection + explicit group params.
      M2293: global aux loss / MoE support.
      M2301: Replay pgs_collection.
      M3009: MTP layer embedding routing; force_all_reduce param.
      M3871: Flextron router grad sync.
      M3981: tp_dp_cp group threading for expert bias.
      M4041: FSDP full-iteration CUDA-graph compatibility.
      M4173: Knowledge-distillation grad handling.

    Args:
        model: List of model chunks (PP/VPP).
        num_tokens: Optional token count for per-token loss normalization.
        pg_collection: Optional ProcessGroupCollection for custom groups.
        force_all_reduce: Force all-reduce even with use_distributed_optimizer=True.
        config: Model parallel config (auto-derived from model[0] if None).
        skip_grad_sync: Skip the main DP gradient collective (DES-LOC non-Kx).
        desloc_step: Current training step (0-indexed) for DES-LOC gating.
        desloc_config: DES-LOC configuration carrying Kx/Ku/Kv periods.
    """
    # Resolve config from model if not provided.
    if config is None:
        config = get_model_config(model[0])

    # ------------------------------------------------------------------
    # Resolve process groups from pg_collection or parallel_state (M2282).
    # ------------------------------------------------------------------
    tp_dp_cp_group: Optional[torch.distributed.ProcessGroup] = None

    if pg_collection is not None:
        assert hasattr(pg_collection, 'tp'), "pg_collection must have 'tp'"
        assert hasattr(pg_collection, 'pp'), "pg_collection must have 'pp'"
        assert hasattr(pg_collection, 'embd'), (
            "pg_collection must have 'embd'. If using default embedding group, pass "
            "parallel_state.get_embedding_group(). If not needed, set explicitly to None."
        )
        assert hasattr(pg_collection, 'pos_embd'), (
            "pg_collection must have 'pos_embd'. If using default position embedding group, "
            "pass parallel_state.get_position_embedding_group(). If not needed, set to None."
        )
        assert hasattr(pg_collection, 'dp_cp'), "pg_collection must have 'dp_cp'"

        if getattr(config, 'moe_router_enable_expert_bias', False):
            assert hasattr(pg_collection, 'tp_dp_cp') and pg_collection.tp_dp_cp is not None, \
                "pg_collection must have 'tp_dp_cp' when moe_router_enable_expert_bias=True"
            tp_dp_cp_group = pg_collection.tp_dp_cp

        tp_group = pg_collection.tp
        pp_group = pg_collection.pp
        embd_group = pg_collection.embd
        pos_emb_group = pg_collection.pos_embd
        dp_cp_group = pg_collection.dp_cp
    else:
        tp_group = _get_tp_group()
        pp_group = _get_pp_group()
        embd_group = _get_embedding_group()
        pos_emb_group = _get_position_embedding_group()
        try:
            dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
        except Exception:
            dp_cp_group = parallel_state.get_data_parallel_group()

    # ------------------------------------------------------------------
    # DES-LOC: resolve Kx/Ku/Kv flags.
    # ------------------------------------------------------------------
    _is_kx = False
    _is_ku = False
    _is_kv = False

    if desloc_step is not None and desloc_config is not None and desloc_config.enabled:
        _is_kx = desloc_config.is_kx_step(desloc_step + 1)
        _is_ku = desloc_config.is_ku_step(desloc_step + 1)
        _is_kv = desloc_config.is_kv_step(desloc_step + 1)
        # Honour DES-LOC Kx gate even if caller did not set skip_grad_sync.
        if not skip_grad_sync and not _is_kx:
            skip_grad_sync = True
        _desloc_log_sync_state(desloc_step, _is_kx, _is_ku, _is_kv, skip_grad_sync)

    # ------------------------------------------------------------------
    # 1. Main DP all-reduce / reduce-scatter across data-parallel ranks.
    #    On non-Kx steps (skip_grad_sync=True) we forward the skip flag
    #    into finish_grad_sync so each DDP module maintains correct state.
    # ------------------------------------------------------------------
    if config is not None and getattr(config, 'timers', None) is not None:
        config.timers('all-grads-sync', log_level=1).start(
            barrier=getattr(config, 'barrier_with_L1_time', False)
        )

    for model_chunk in model:
        if isinstance(model_chunk, DistributedDataParallel):
            if skip_grad_sync:
                for bg in (
                    model_chunk.bucket_groups + model_chunk.expert_parallel_bucket_groups
                ):
                    bg._skip_sync = True
            model_chunk.finish_grad_sync(force_all_reduce=force_all_reduce)
        else:
            if not skip_grad_sync:
                _direct_allreduce_grads(model_chunk, config)

    if config is not None and getattr(config, 'timers', None) is not None:
        config.timers('all-grads-sync').stop()

    # ------------------------------------------------------------------
    # 2. Conditional embedding grads (DiT-style cond embedders across PP).
    # ------------------------------------------------------------------
    if config is not None and getattr(config, 'timers', None) is not None:
        config.timers('conditional-embedder-grads-all-reduce', log_level=1).start(
            barrier=getattr(config, 'barrier_with_L1_time', False)
        )
    _allreduce_conditional_embedding_grads(model, config, pp_group)
    if config is not None and getattr(config, 'timers', None) is not None:
        config.timers('conditional-embedder-grads-all-reduce').stop()

    # ------------------------------------------------------------------
    # 3. Flextron router grad sync (M3871).
    # ------------------------------------------------------------------
    if getattr(config, 'flextron', False):
        _allreduce_router_grads(model, config)

    # ------------------------------------------------------------------
    # 4. All-reduce sequence-parallel and TP-domain-average grads.
    # ------------------------------------------------------------------
    if config is not None and getattr(config, 'timers', None) is not None:
        config.timers('non-tensor-parallel-grads-all-reduce', log_level=1).start(
            barrier=getattr(config, 'barrier_with_L1_time', False)
        )
    _allreduce_non_tensor_model_parallel_grads(model, config, tp_group)
    if config is not None and getattr(config, 'timers', None) is not None:
        config.timers('non-tensor-parallel-grads-all-reduce').stop()

    # ------------------------------------------------------------------
    # 5. Embedding grad all-reduce across first and last PP stages.
    # ------------------------------------------------------------------
    if config is not None and getattr(config, 'timers', None) is not None:
        config.timers('embedding-grads-all-reduce', log_level=1).start(
            barrier=getattr(config, 'barrier_with_L1_time', False)
        )
    _allreduce_word_embedding_grads(model, config, embd_group, pp_group)
    _allreduce_position_embedding_grads(model, config, pos_emb_group, pp_group)
    if config is not None and getattr(config, 'timers', None) is not None:
        config.timers('embedding-grads-all-reduce').stop()

    # ------------------------------------------------------------------
    # 6. MoE expert bias update (M3981 tp_dp_cp group threading).
    # ------------------------------------------------------------------
    if getattr(config, 'moe_router_enable_expert_bias', False):
        if pg_collection is None:
            try:
                tp_dp_cp_group = parallel_state.get_tensor_and_data_parallel_group(
                    with_context_parallel=True
                )
            except Exception:
                tp_dp_cp_group = None
        _update_router_expert_bias(model, config, tp_dp_cp_group=tp_dp_cp_group)

    # ------------------------------------------------------------------
    # 7. Reset temporary tensors (aux loss trackers, tokens_per_expert).
    # ------------------------------------------------------------------
    reset_model_temporary_tensors(config, model)

    # ------------------------------------------------------------------
    # 8. Per-token loss normalization: scale gradients by 1/num_tokens.
    #    num_tokens lives on the last PP stage; broadcast to all stages,
    #    then all-reduce across DP ranks so every rank uses the global count.
    #    torch.clamp avoids div-by-zero without host-side sync (CUDA-graph safe).
    # ------------------------------------------------------------------
    if num_tokens is not None:
        assert not isinstance(pp_group, list), \
            "pp_group must be a single group, not a list"
        last_rank = get_pp_last_rank(pp_group)
        torch.distributed.broadcast(num_tokens, src=last_rank, group=pp_group)
        torch.distributed.all_reduce(num_tokens, group=dp_cp_group)

        safe_num_tokens = torch.clamp(num_tokens, min=1)
        scaling = 1.0 / safe_num_tokens
        for model_chunk in model:
            if isinstance(model_chunk, DistributedDataParallel):
                model_chunk.scale_gradients(scaling)
            else:
                inner = _get_inner(model_chunk)
                for param in inner.parameters():
                    if not param.requires_grad:
                        continue
                    grad_attr = _get_main_grad_attr(param)
                    grad = getattr(param, grad_attr, None)
                    if grad is not None:
                        grad.mul_(scaling)

    # ------------------------------------------------------------------
    # 9. DES-LOC Ku/Kv: synchronize Adam first/second moments.
    #    These infrequent but expensive syncs keep optimizer state
    #    consistent across heterogeneous DP replicas.
    # ------------------------------------------------------------------
    if desloc_step is not None and desloc_config is not None and desloc_config.enabled:
        _desloc_sync_optimizer_moments(model, config, _is_ku, _is_kv)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_inner(model_chunk: nn.Module) -> nn.Module:
    """Return the inner module (unwrap DDP wrapper if needed)."""
    return getattr(model_chunk, '_module', model_chunk)


def _named_parameters(model_chunk: nn.Module):
    """Yield (name, param) from model chunk, unwrapping DDP if needed."""
    inner = _get_inner(model_chunk)
    try:
        gen = get_attr_wrapped_model(model_chunk, 'named_parameters')
        if callable(gen):
            yield from gen()
            return
    except Exception:
        pass
    yield from inner.named_parameters()


def _owns_param(model_chunk: nn.Module, param: torch.nn.Parameter) -> bool:
    """Check if model_chunk owns param (for ddp_config lookup)."""
    try:
        return any(p is param for p in _get_inner(model_chunk).parameters())
    except Exception:
        return False


def _get_tp_group():
    if parallel_state.is_initialized():
        try:
            return parallel_state.get_tensor_model_parallel_group()
        except Exception:
            pass
    return torch.distributed.GroupMember.WORLD


def _get_pp_group():
    if parallel_state.is_initialized():
        try:
            return parallel_state.get_pipeline_model_parallel_group()
        except Exception:
            pass
    return torch.distributed.GroupMember.WORLD


def _get_pp_world_size() -> int:
    if parallel_state.is_initialized():
        try:
            return parallel_state.get_pipeline_model_parallel_world_size()
        except Exception:
            pass
    return 1


def _get_embedding_group():
    if parallel_state.is_initialized():
        try:
            return parallel_state.get_embedding_group(check_initialized=False)
        except Exception:
            try:
                return parallel_state.get_embedding_group()
            except Exception:
                pass
    return None


def _get_position_embedding_group():
    if parallel_state.is_initialized():
        try:
            return parallel_state.get_position_embedding_group(check_initialized=False)
        except Exception:
            try:
                return parallel_state.get_position_embedding_group()
            except Exception:
                pass
    return None
