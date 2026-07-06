# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""deepspeed.core.optimizer — public API re-exports.

This package implements a ZeRO-3 style distributed optimizer with DES-LOC
heterogeneous shard sizing.  The implementation is split across three modules:

  optimizer_config.py   — :class:`OptimizerConfig` dataclass
  distrib_optimizer.py  — :class:`DistributedOptimizer` and base classes
  __init__.py           — re-exports for ``from deepspeed.core.optimizer import X``

Guaranteed import paths
-----------------------
::

    from deepspeed.core.optimizer import DistributedOptimizer
    from deepspeed.core.optimizer import OptimizerConfig
    from deepspeed.core.optimizer import MegatronOptimizer
    from deepspeed.core.optimizer import MixedPrecisionOptimizer
    from deepspeed.core.optimizer import clip_grad_norm
    from deepspeed.core.optimizer import Range

    from deepspeed.core.optimizer.optimizer_config import OptimizerConfig, ParamKey
    from deepspeed.core.optimizer.distrib_optimizer import DistributedOptimizer
"""

from deepspeed.core.optimizer.optimizer_config import (
    OptimizerConfig,
    ParamKey,
    # From Megatron M2933: flexible optimizer/scheduler override system
    ParamPredicate,
    ParamGroupOverride,
    combine_param_group_overrides,
    param_group_override_to_tuple,
    get_standard_config_overrides,
)
from deepspeed.core.optimizer.distrib_optimizer import (
    # Core optimizer hierarchy
    MegatronOptimizer,
    MixedPrecisionOptimizer,
    DistributedOptimizer,
    # Utilities
    clip_grad_norm,
    Range,
    # Shard sizing helper (useful for external callers)
    _compute_hetero_shard_boundaries,
    _round_up,
    # From Megatron M4171: separate grad-norm groups for MTP detach heads
    copy_optimizer_param_metadata,
    MTP_GRAD_NORM_GROUP,
    GRAD_NORM_GROUP_ATTR,
    SEPARATE_GRAD_NORM_GROUPS,
    _get_param_grad_norm_group,
    _is_separate_grad_norm_group,
    # LR logging helper (From Megatron M3286)
    get_canonical_lr_for_logging,
)
# From Megatron M3811: parameter layout dataclasses and padding helpers.
# These are consumed by DDP buffer construction and the NVFP4 packed layout
# path in param_and_grad_buffer.py, which imports them directly from
# deepspeed.core.optimizer.param_layout (or megatron.core.optimizer.param_layout
# with a no-op ImportError fallback).
from deepspeed.core.optimizer.param_layout import (
    BufferKey,
    PerBufferParamLayout,
    FullParamLayout,
    pad_param_start,
    pad_bucket_end,
    pad_to_divisor,
    bucket_end_divisor,
)

# Megatron optimizer wrappers ported to deepspeed.core (optimizer.py)
from deepspeed.core.optimizer.optimizer import (
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    ChainedOptimizer,
    ProxyDict,
    clip_grad_by_total_norm_fp32,
    count_zeros_fp32,
    param_is_not_shared,
    param_group_identifier_keys,
    _zero_grad_group_helper,
    _multi_tensor_copy_this_to_that,
)


def build_optimizer(params, config: OptimizerConfig):
    """Construct an optimizer from *config*.

    From Megatron M3543 (PR #3813): dispatches to Lion when
    ``config.optimizer_type == 'lion'``, falling back to AdamW if
    lion-pytorch is not installed.  All other values of optimizer_type
    route to AdamW (default).

    Args:
        params: Iterable of parameters or param-groups passed to the optimizer.
        config: :class:`OptimizerConfig` instance.

    Returns:
        A :class:`torch.optim.Optimizer`.
    """
    import logging
    import torch

    if getattr(config, 'optimizer_type', 'adamw') == 'lion':
        try:
            from lion_pytorch import Lion
            return Lion(
                params,
                lr=config.lr,
                betas=(getattr(config, 'lion_beta1', 0.9), getattr(config, 'lion_beta2', 0.99)),
                weight_decay=getattr(config, 'weight_decay', 0.1),
            )
        except ImportError:
            logging.getLogger(__name__).warning(
                'Lion optimizer (M3543) requested but lion-pytorch not installed. '
                'Falling back to AdamW. Install: pip install lion-pytorch'
            )

    return torch.optim.AdamW(
        params,
        lr=config.lr,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )

__all__ = [
    "OptimizerConfig",
    "MegatronOptimizer",
    "MixedPrecisionOptimizer",
    "DistributedOptimizer",
    "clip_grad_norm",
    "Range",
    "_compute_hetero_shard_boundaries",
    "_round_up",
    # From Megatron M4171
    "copy_optimizer_param_metadata",
    "MTP_GRAD_NORM_GROUP",
    "GRAD_NORM_GROUP_ATTR",
    "SEPARATE_GRAD_NORM_GROUPS",
    "_get_param_grad_norm_group",
    "_is_separate_grad_norm_group",
    "get_canonical_lr_for_logging",
    # From Megatron M3543: Lion optimizer factory
    "build_optimizer",
    # From optimizer.py port
    "Float16OptimizerWithFloat16Params",
    "FP32Optimizer",
    "ChainedOptimizer",
    "ProxyDict",
    "clip_grad_by_total_norm_fp32",
    "count_zeros_fp32",
    "param_is_not_shared",
    "param_group_identifier_keys",
    "_zero_grad_group_helper",
    "_multi_tensor_copy_this_to_that",
    # From Megatron M3811: param layout dataclasses + padding helpers
    "BufferKey",
    "PerBufferParamLayout",
    "FullParamLayout",
    "pad_param_start",
    "pad_bucket_end",
    "pad_to_divisor",
    "bucket_end_divisor",
]

# ---------------------------------------------------------------------------
# ZeRO-2 + DES-LOC additional exports (from distrib_optimizer.py appendix)
# ---------------------------------------------------------------------------
from deepspeed.core.optimizer.distrib_optimizer import (
    ZeROStage2Optimizer,
    build_distributed_optimizer,
    sync_desloc_moments,
    desync_aware_step,
    detect_grad_overflow,
    skip_step_on_overflow,
)

# ---------------------------------------------------------------------------
# Additional optimizer utilities (from optimizer.py appendix)
# ---------------------------------------------------------------------------
from deepspeed.core.optimizer.optimizer import (
    GradNormSkipScheduler,
    StubOptimizer,
    OptimizerGroupBuilder,
    get_optimizer_lr,
    _safe_get_rank,
    _safe_get_world_size,
)

# ---------------------------------------------------------------------------
# Per-tier and QK clipping (from clip_grads.py)
# ---------------------------------------------------------------------------
from deepspeed.core.optimizer.clip_grads import (
    get_grad_norm_fp32,
    clip_grad_by_total_norm_fp32,
    clip_grad_norm,
    count_zeros_fp32,
    clip_grad_by_tier,
    clip_qk_grad_norm,
    clip_grads_with_norm_by_group,
    TierClipConfig,
    QKClipConfig,
    GradNormEMA,
    TIER_H100,
    TIER_A6000,
    TIER_BLACKWELL,
    TIER_CONSUMER,
)

__all__ += [
    # ZeRO-2 + DES-LOC
    "ZeROStage2Optimizer",
    "build_distributed_optimizer",
    "sync_desloc_moments",
    "desync_aware_step",
    "detect_grad_overflow",
    "skip_step_on_overflow",
    # Optimizer utilities
    "GradNormSkipScheduler",
    "StubOptimizer",
    "OptimizerGroupBuilder",
    "get_optimizer_lr",
    "_safe_get_rank",
    "_safe_get_world_size",
    # Clip grads
    "get_grad_norm_fp32",
    "clip_grad_by_total_norm_fp32",
    "clip_grad_norm",
    "count_zeros_fp32",
    "clip_grad_by_tier",
    "clip_qk_grad_norm",
    "clip_grads_with_norm_by_group",
    "TierClipConfig",
    "QKClipConfig",
    "GradNormEMA",
    "TIER_H100",
    "TIER_A6000",
    "TIER_BLACKWELL",
    "TIER_CONSUMER",
]


# ===========================================================================
# Section: get_megatron_optimizer — top-level optimizer factory
# Ported from Megatron-LM/megatron/core/optimizer/__init__.py
# ===========================================================================

def _get_param_groups(
    model_chunks: List,
    config: "OptimizerConfig",
    config_overrides: Optional[Dict] = None,
) -> List[Dict]:
    """Create parameter groups from model chunks and optional overrides.

    Groups parameters by their combined override tuple so that parameters
    sharing identical hyper-parameter overrides land in the same optimizer
    param group.  Cross-rank alignment is performed via all_gather_object so
    that all ranks always have the same number of param groups (required for
    distributed checkpoint consistency).

    From Megatron M2654 / M2933: per-param optimizer config overrides.

    Args:
        model_chunks:     List of model modules (or MegatronModule subclasses).
        config:           Base optimizer configuration.
        config_overrides: Dict mapping ParamKey → ParamGroupOverride.

    Returns:
        List of param group dicts with ``'params'``, ``'lr'``,
        ``'weight_decay'``, ``'default_config'``, etc.
    """
    from deepspeed.core.optimizer.optimizer_config import (
        ParamKey,
        combine_param_group_overrides,
        param_group_override_to_tuple,
    )

    params_map: Dict[tuple, List[torch.nn.Parameter]] = {}

    for model_chunk in model_chunks:
        named_params = (
            model_chunk.named_parameters()
            if hasattr(model_chunk, "named_parameters")
            else []
        )
        for name, param in named_params:
            if not param.requires_grad:
                continue

            is_expert_parallel = not getattr(param, "allreduce", True)

            param_overrides_list: List[dict] = []
            if config_overrides is not None:
                for param_key, param_override in config_overrides.items():
                    if isinstance(param_key, ParamKey) and param_key.matches(param, name):
                        param_overrides_list.append(param_override)

            if param_overrides_list:
                merged_override = combine_param_group_overrides(param_overrides_list)
                override_tuple = param_group_override_to_tuple(merged_override)
            else:
                merged_override = {}
                override_tuple = None

            key = (override_tuple, is_expert_parallel)
            if key not in params_map:
                params_map[key] = []
            params_map[key].append(param)

    # Cross-rank alignment: gather all keys and fill missing ones with empty param lists.
    params_key = list(params_map.keys())
    if torch.distributed.is_initialized():
        gathered_params_key = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_params_key, params_key)
        for keys in gathered_params_key:
            if keys is None:
                continue
            for key in keys:
                if key not in params_key:
                    params_key.append(key)

    param_groups: List[Dict] = []
    for key in sorted(params_key, key=lambda x: (x[0] is not None, x[0])):
        override_tuple, is_expert_parallel = key
        params = params_map.get(key, [])

        if override_tuple is None:
            param_override: Dict = {}
        else:
            param_override = {k: v for (k, v) in override_tuple}

        uses_default_lr_schedule = (not bool(override_tuple)) or not any(
            "lr" in k for k in param_override
        )

        default_fields: Dict = {
            "wd_mult": 1.0,
            "lr_mult": 1.0,
            "is_decoupled_lr": False,
            "is_expert_parallel": is_expert_parallel,
            "max_lr": config.lr,
            "min_lr": getattr(config, "min_lr", 0.0),
        }
        param_group = {
            "params": params,
            "default_config": uses_default_lr_schedule,
            **default_fields,
            **param_override,
        }
        param_groups.append(param_group)

    return param_groups


def _get_param_groups_and_buffers(
    model_chunks: List,
    model_chunk_offset: int,
    config: "OptimizerConfig",
    config_overrides: Optional[Dict],
    filter_fn: Callable,
    buffer_name: str,
) -> Tuple[List[Dict], Dict]:
    """Return param groups and grad buffers for the given model chunks.

    Wraps ``_get_param_groups`` and additionally collects the named buffer
    from each model chunk into a dict keyed by chunk index.

    Args:
        model_chunks:       Model chunks to extract from.
        model_chunk_offset: Global index offset for multi-optimizer setups.
        config:             Optimizer config.
        config_overrides:   Per-param overrides.
        filter_fn:          Predicate to filter param groups (e.g. expert vs non-expert).
        buffer_name:        Attribute name of the grad buffer on model chunk.

    Returns:
        (param_groups, buffers_dict) pair.
    """
    param_groups = _get_param_groups(model_chunks, config, config_overrides)
    param_groups = [g for g in param_groups if filter_fn(g)]
    buffers: Dict[int, list] = {}
    for idx, model_chunk in enumerate(model_chunks):
        if hasattr(model_chunk, buffer_name):
            buffers[idx + model_chunk_offset] = getattr(model_chunk, buffer_name)
    return param_groups, buffers


def _build_inner_optimizer(
    param_groups: List[Dict],
    config: "OptimizerConfig",
) -> Tuple[torch.optim.Optimizer, Optional[Callable]]:
    """Construct the raw PyTorch optimizer from config.

    Supports Adam (AdamW) and SGD.  TE/Apex FusedAdam is used when
    available; falls back to PyTorch AdamW.

    Returns:
        (optimizer, init_state_fn) pair; init_state_fn is None for AdamW/SGD.
    """
    optimizer_name = getattr(config, "optimizer", "adam").lower()
    init_state_fn: Optional[Callable] = None

    # Flatten param_groups: optimizer needs actual lr / weight_decay scalars.
    flat_groups: List[Dict] = []
    for g in param_groups:
        lr_mult = g.get("lr_mult", 1.0)
        wd_mult = g.get("wd_mult", 1.0)
        flat_g = {
            "params": g["params"],
            "lr": (config.lr or 1e-4) * lr_mult,
            "weight_decay": config.weight_decay * wd_mult,
        }
        flat_groups.append(flat_g)

    if optimizer_name in ("adam", "adamw"):
        try:
            from transformer_engine.pytorch.optimizers import FusedAdam as _Adam
        except ImportError:
            try:
                from apex.optimizers import FusedAdam as _Adam
            except ImportError:
                from torch.optim import AdamW as _Adam

        optimizer = _Adam(
            flat_groups,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
        )

    elif optimizer_name == "sgd":
        from torch.optim import SGD
        optimizer = SGD(
            flat_groups,
            momentum=getattr(config, "sgd_momentum", 0.9),
        )

    else:
        raise ValueError(
            f"_build_inner_optimizer: unsupported optimizer '{optimizer_name}'. "
            "Supported: 'adam', 'adamw', 'sgd'."
        )

    return optimizer, init_state_fn


def _wrap_optimizer(
    optimizer: torch.optim.Optimizer,
    config: "OptimizerConfig",
    model_chunks: List,
    param_groups: List[Dict],
    per_model_buffers: Optional[Dict],
    data_parallel_group: Optional[torch.distributed.ProcessGroup],
    data_parallel_group_gloo: Optional[torch.distributed.ProcessGroup],
    model_parallel_group: Optional[torch.distributed.ProcessGroup],
    tier_assignments: Optional[List] = None,
    init_state_fn: Optional[Callable] = None,
    intra_dist_opt_group: Optional[torch.distributed.ProcessGroup] = None,
) -> "MegatronOptimizer":
    """Wrap *optimizer* in the appropriate Megatron optimizer class.

    Dispatches to:
      - :class:`DistributedOptimizer`        when ``config.use_distributed_optimizer``
      - :class:`Float16OptimizerWithFloat16Params` for bf16/fp16 non-distributed
      - :class:`FP32Optimizer`               otherwise

    Args:
        optimizer:              Raw PyTorch optimizer.
        config:                 Optimizer configuration.
        model_chunks:           Model chunks (used to extract all params).
        param_groups:           Param groups with metadata.
        per_model_buffers:      Grad buffers per model chunk (for DistOpt).
        data_parallel_group:    DP process group.
        data_parallel_group_gloo: Gloo DP group for checkpoint I/O.
        model_parallel_group:   MP process group (for grad stats).
        tier_assignments:       DES-LOC tier list for heterogeneous sizing.
        init_state_fn:          Optional state initialiser (Lion, etc.).
        intra_dist_opt_group:   Intra-instance DP group for multi-instance setups.

    Returns:
        Wrapped Megatron optimizer instance.
    """
    from deepspeed.core.model_parallel_config import ModelParallelConfig
    from deepspeed.core.optimizer.distrib_optimizer import DistributedOptimizer
    from deepspeed.core.optimizer.optimizer import (
        Float16OptimizerWithFloat16Params,
        FP32Optimizer,
    )

    # Collect all params from model chunks
    all_params: List[torch.nn.Parameter] = []
    for g in param_groups:
        all_params.extend(g["params"])

    # Build a minimal ModelParallelConfig (enough for DistributedOptimizer)
    mp_config = ModelParallelConfig()

    if config.use_distributed_optimizer:
        if per_model_buffers is None:
            per_model_buffers = {}
        # Flatten per_model_buffers to a single list
        all_buffers = []
        for chunk_idx in sorted(per_model_buffers.keys()):
            bufs = per_model_buffers[chunk_idx]
            if isinstance(bufs, list):
                all_buffers.extend(bufs)
            else:
                all_buffers.append(bufs)

        wrapped = DistributedOptimizer(
            config=config,
            optimizer=optimizer,
            params=all_params,
            model_parallel_config=mp_config,
            param_and_grad_buffers=all_buffers,
            data_parallel_group=data_parallel_group,
            data_parallel_group_gloo=data_parallel_group_gloo,
            tier_assignments=tier_assignments,
            intra_dist_opt_group=intra_dist_opt_group,
        )

    elif config.fp16 or config.bf16:
        grad_scaler = None
        if config.fp16 and getattr(config, "loss_scale", None):
            # Constant scaler
            try:
                from deepspeed.core.optimizer.grad_scaler import ConstantGradScaler
                grad_scaler = ConstantGradScaler(config.loss_scale)
            except ImportError:
                pass

        wrapped = Float16OptimizerWithFloat16Params(
            optimizer=optimizer,
            config=config,
            grad_scaler=grad_scaler,
            init_state_fn=init_state_fn or (lambda opt, cfg=None: None),
        )

    else:
        wrapped = FP32Optimizer(
            optimizer=optimizer,
            config=config,
            init_state_fn=init_state_fn or (lambda opt, cfg=None: None),
        )

    # Attach process group attributes
    setattr(wrapped, "grad_stats_parallel_group", model_parallel_group)
    if intra_dist_opt_group is not None:
        setattr(wrapped, "grad_stats_parallel_group", intra_dist_opt_group)

    return wrapped


def get_megatron_optimizer(
    config: "OptimizerConfig",
    model_chunks: List,
    config_overrides: Optional[Dict] = None,
    data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group_gloo: Optional[torch.distributed.ProcessGroup] = None,
    model_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    tier_assignments: Optional[List] = None,
    intra_dist_opt_group: Optional[torch.distributed.ProcessGroup] = None,
    zero_stage: int = 3,
) -> "MegatronOptimizer":
    """Top-level Megatron/DeepSpeed optimizer factory — ported from Megatron __init__.py.

    Handles parameter grouping (with per-param overrides), inner optimizer
    construction, and wrapping in the appropriate :class:`MegatronOptimizer`
    subclass.  When MoE expert-parallel parameters are present they are built
    into a separate optimizer and the two are combined via :class:`ChainedOptimizer`.

    DES-LOC integration
    ~~~~~~~~~~~~~~~~~~~
    When ``config.use_distributed_optimizer=True`` and *tier_assignments* is
    provided, the :class:`DistributedOptimizer` uses TFLOPS-weighted heterogeneous
    shard boundaries (via ``_compute_hetero_shard_boundaries``) so each GPU
    tier receives an optimizer state shard proportional to its compute capacity.

    Standard optimizer selection rules (from Megatron M2307 / M3543)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - ``config.use_distributed_optimizer=True`` → :class:`DistributedOptimizer` (ZeRO-3)
    - ``zero_stage=2`` → :class:`ZeROStage2Optimizer`
    - ``config.bf16 or config.fp16`` → :class:`Float16OptimizerWithFloat16Params`
    - Otherwise → :class:`FP32Optimizer`

    Expert-parallel parameters (``param.allreduce=False``) are separated into a
    dedicated optimizer group so that expert and dense params can use different
    data-parallel communication groups.

    Args:
        config:                   Optimizer hyper-parameters and flags.
        model_chunks:             Model chunks (list of ``nn.Module`` or
                                  ``MegatronModule``).
        config_overrides:         Dict mapping :class:`ParamKey` →
                                  :class:`ParamGroupOverride` for per-param
                                  lr/wd overrides.  ``None`` uses standard
                                  defaults (no bias/length-1 wd zeroing).
        data_parallel_group:      NCCL process group for gradient communication.
        data_parallel_group_gloo: Gloo process group for checkpoint I/O.
        model_parallel_group:     Process group for gradient norm reduction.
        tier_assignments:         Per-rank :class:`TierType` list for DES-LOC
                                  heterogeneous shard sizing.
        intra_dist_opt_group:     Intra-instance DP group for multi-instance
                                  DistOpt setups (From Megatron M2456).
        zero_stage:               ZeRO stage override (2 or 3).

    Returns:
        A :class:`MegatronOptimizer` instance (or :class:`ChainedOptimizer`
        if MoE expert params are present).

    Raises:
        ValueError: If zero_stage is not 2 or 3.

    Examples::

        # Standard BF16 distributed optimizer
        optimizer = get_megatron_optimizer(
            config=OptimizerConfig(bf16=True, use_distributed_optimizer=True, lr=3e-4),
            model_chunks=[model],
        )

        # Heterogeneous PCIe cluster with H100 + A6000
        optimizer = get_megatron_optimizer(
            config=OptimizerConfig(bf16=True, use_distributed_optimizer=True,
                                   heterogeneous_shard_sizing=True, lr=3e-4),
            model_chunks=[model],
            tier_assignments=[TierType.DATACENTER, TierType.PROFESSIONAL, ...],
        )
    """
    import logging as _logging
    from deepspeed.core.optimizer.distrib_optimizer import ZeROStage2Optimizer
    from deepspeed.core.optimizer.optimizer import ChainedOptimizer

    _logger = _logging.getLogger(__name__)

    if zero_stage not in (2, 3):
        raise ValueError(
            f"get_megatron_optimizer: zero_stage must be 2 or 3, got {zero_stage}."
        )

    # Resolve process groups from parallel_state when not explicitly given.
    if data_parallel_group is None and parallel_state.is_initialized():
        data_parallel_group = parallel_state.get_data_parallel_group()
    if model_parallel_group is None and parallel_state.is_initialized():
        model_parallel_group = parallel_state.get_model_parallel_group()

    _logger.info(
        "get_megatron_optimizer: optimizer=%s bf16=%s fp16=%s use_distopt=%s zero_stage=%d "
        "hetero=%s chunks=%d",
        getattr(config, "optimizer", "adam"),
        config.bf16,
        config.fp16,
        config.use_distributed_optimizer,
        zero_stage,
        config.heterogeneous_shard_sizing,
        len(model_chunks),
    )

    # -----------------------------------------------------------------------
    # ZeRO-2 fast path: no grad buffers needed
    # -----------------------------------------------------------------------
    if zero_stage == 2:
        from deepspeed.core.model_parallel_config import ModelParallelConfig
        all_params = [
            p for chunk in model_chunks
            for _, p in (chunk.named_parameters() if hasattr(chunk, "named_parameters") else [])
            if p.requires_grad
        ]
        inner_optimizer, init_fn = _build_inner_optimizer(
            [{"params": all_params}], config
        )
        mp_config = ModelParallelConfig()
        z2 = ZeROStage2Optimizer(
            config=config,
            optimizer=inner_optimizer,
            params=all_params,
            model_parallel_config=mp_config,
            data_parallel_group=data_parallel_group,
            tier_assignments=tier_assignments,
        )
        setattr(z2, "grad_stats_parallel_group", model_parallel_group)
        return z2

    # -----------------------------------------------------------------------
    # Collect dense (non-expert) param groups + buffers
    # -----------------------------------------------------------------------
    dense_param_groups, dense_buffers = _get_param_groups_and_buffers(
        model_chunks=model_chunks,
        model_chunk_offset=0,
        config=config,
        config_overrides=config_overrides,
        filter_fn=lambda g: not g.get("is_expert_parallel", False),
        buffer_name="buffers",
    )

    # Collect expert-parallel param groups + buffers
    moe_param_groups, moe_buffers = _get_param_groups_and_buffers(
        model_chunks=model_chunks,
        model_chunk_offset=0,
        config=config,
        config_overrides=config_overrides,
        filter_fn=lambda g: g.get("is_expert_parallel", False),
        buffer_name="expert_parallel_buffers",
    )

    optimizers: List["MegatronOptimizer"] = []

    # -----------------------------------------------------------------------
    # Dense optimizer
    # -----------------------------------------------------------------------
    if dense_param_groups and any(g["params"] for g in dense_param_groups):
        dense_inner, dense_init_fn = _build_inner_optimizer(dense_param_groups, config)
        dense_opt = _wrap_optimizer(
            optimizer=dense_inner,
            config=config,
            model_chunks=model_chunks,
            param_groups=dense_param_groups,
            per_model_buffers=dense_buffers if dense_buffers else None,
            data_parallel_group=data_parallel_group,
            data_parallel_group_gloo=data_parallel_group_gloo,
            model_parallel_group=model_parallel_group,
            tier_assignments=tier_assignments,
            init_state_fn=dense_init_fn,
            intra_dist_opt_group=intra_dist_opt_group,
        )
        optimizers.append(dense_opt)

    # -----------------------------------------------------------------------
    # Expert-parallel optimizer (MoE)
    # -----------------------------------------------------------------------
    if moe_param_groups and any(g["params"] for g in moe_param_groups):
        moe_inner, moe_init_fn = _build_inner_optimizer(moe_param_groups, config)
        # Expert params use the expert-parallel data group if available
        expt_dp_group = (
            parallel_state.get_expert_data_parallel_group()
            if parallel_state.is_initialized() and hasattr(
                parallel_state, "get_expert_data_parallel_group"
            )
            else data_parallel_group
        )
        expt_mp_group = (
            parallel_state.get_expert_model_parallel_group()
            if parallel_state.is_initialized() and hasattr(
                parallel_state, "get_expert_model_parallel_group"
            )
            else model_parallel_group
        )
        moe_opt = _wrap_optimizer(
            optimizer=moe_inner,
            config=config,
            model_chunks=model_chunks,
            param_groups=moe_param_groups,
            per_model_buffers=moe_buffers if moe_buffers else None,
            data_parallel_group=expt_dp_group,
            data_parallel_group_gloo=data_parallel_group_gloo,
            model_parallel_group=expt_mp_group,
            tier_assignments=tier_assignments,
            init_state_fn=moe_init_fn,
            intra_dist_opt_group=intra_dist_opt_group,
        )
        optimizers.append(moe_opt)

    if not optimizers:
        # No trainable parameters — return a stub
        from deepspeed.core.optimizer.optimizer import StubOptimizer
        _logger.warning(
            "get_megatron_optimizer: no trainable parameters found; returning StubOptimizer."
        )
        return StubOptimizer(config)

    if len(optimizers) == 1:
        return optimizers[0]

    return ChainedOptimizer(optimizers)


# ---------------------------------------------------------------------------
# Layer-wise optimizer (Muon) exports
# ---------------------------------------------------------------------------
from deepspeed.core.optimizer.layer_wise_optimizer import (
    LayerWiseDistributedOptimizer,
    MuonOptimizer,
    is_managed_by_layer_wise_optimizer,
    tag_params_for_buffer_routing,
    _zeropower_via_newtonschulz5,
)

# ---------------------------------------------------------------------------
# QK-clip exports
# ---------------------------------------------------------------------------
from deepspeed.core.optimizer.qk_clip import (
    QKLogitClipConfig,
    AttentionLogitMonitor,
    clip_attention_logits,
    register_qk_logit_hooks,
    clip_qk_grad_norm,
    clip_qk,
)

# ---------------------------------------------------------------------------
# get_megatron_optimizer and helpers (defined above in this file)
# ---------------------------------------------------------------------------

__all__ += [
    # Factory
    "get_megatron_optimizer",
    "_get_param_groups",
    "_get_param_groups_and_buffers",
    "_build_inner_optimizer",
    "_wrap_optimizer",
    # Layer-wise / Muon
    "LayerWiseDistributedOptimizer",
    "MuonOptimizer",
    "is_managed_by_layer_wise_optimizer",
    "tag_params_for_buffer_routing",
    "_zeropower_via_newtonschulz5",
    # QK clip
    "QKLogitClipConfig",
    "AttentionLogitMonitor",
    "clip_attention_logits",
    "register_qk_logit_hooks",
    "clip_qk_grad_norm",
    "clip_qk",
]

# ---------------------------------------------------------------------------
# CUDA graph optimizer wrapper (Megatron optimizer_cuda_graph.py port)
# ---------------------------------------------------------------------------
from deepspeed.core.optimizer.optimizer_cuda_graph import (
    OptimizerCudaGraphWrapper,
    wrap_optimizer_step,
)

# ---------------------------------------------------------------------------
# Emerging optimizer registry (Muon, Lion, SOAP)
# ---------------------------------------------------------------------------
from deepspeed.core.optimizer.emerging_optimizers import (
    EmergingOptimizerEntry,
    EMERGING_OPTIMIZER_REGISTRY,
    register_emerging_optimizer,
    get_emerging_optimizer,
    list_emerging_optimizers,
    build_emerging_optimizer,
    route_params_by_tier,
)

__all__ += [
    # CUDA graph wrapper
    "OptimizerCudaGraphWrapper",
    "wrap_optimizer_step",
    # Emerging optimizers
    "EmergingOptimizerEntry",
    "EMERGING_OPTIMIZER_REGISTRY",
    "register_emerging_optimizer",
    "get_emerging_optimizer",
    "list_emerging_optimizers",
    "build_emerging_optimizer",
    "route_params_by_tier",
    # get_megatron_optimizer factory (defined inline above)
    "get_megatron_optimizer",
    "_get_param_groups",
    "_get_param_groups_and_buffers",
    "_build_inner_optimizer",
    "_wrap_optimizer",
]
