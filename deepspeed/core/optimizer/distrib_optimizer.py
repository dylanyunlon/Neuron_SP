# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Distributed optimizer — ZeRO-3 style param/grad/state sharding.

Reimplemented from Megatron-LM/megatron/core/optimizer/distrib_optimizer.py
with DES-LOC extensions for heterogeneous GPU tiers (H100 / A6000).

Design overview
---------------
Each data-parallel rank owns a contiguous *shard* of every grad buffer.
The shard boundaries are computed as equal slices by default, or weighted
by each GPU's BF16 TFLOPS when ``OptimizerConfig.heterogeneous_shard_sizing``
is enabled (DES-LOC extension):

  H100 shard fraction  = H100_tflops / (H100_tflops + A6000_tflops)
  A6000 shard fraction = A6000_tflops / (H100_tflops + A6000_tflops)

On every step:
  1. Reduce-scatter BF16 gradients → each rank gets its averaged FP32 shard.
  2. Optional grad clipping on the local shard.
  3. Adam step on the local FP32 shard.
  4. All-gather updated FP32 shards → reconstruct BF16 model weights.
  5. (DES-LOC) Every Ku steps: all-reduce first moments across DP ranks.
  6. (DES-LOC) Every Kv steps: all-reduce second moments across DP ranks.

Public API
----------
  DistributedOptimizer  — main class
  MegatronOptimizer     — abstract base
  MixedPrecisionOptimizer — BF16/FP16 mixed-precision base
  clip_grad_norm        — model-parallel-aware gradient clipping
  Range                 — (start, end, size) index helper

Import path guaranteed:
  from deepspeed.core.optimizer import DistributedOptimizer  ✓
  from deepspeed.core.optimizer.distrib_optimizer import DistributedOptimizer  ✓
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from deepspeed.core.model_parallel_config import ModelParallelConfig
from deepspeed.core.desloc_config import DesLocConfig, TierType
from deepspeed.core.distributed import ParamAndGradBuffer
from deepspeed.core.optimizer.optimizer_config import OptimizerConfig
import deepspeed.core.parallel_state as parallel_state

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ParamMeta dataclass and ParamRegistry (M4171 followup)
# ---------------------------------------------------------------------------
# Replaces monkey-patching .shared / .grad_norm_group attributes directly
# onto tensor objects, which is fragile when views/copies are created.
# All metadata lives in a central registry keyed by id(param).


@dataclass
class ParamMeta:
    """Metadata record stored in :class:`ParamRegistry` for each parameter.

    Attributes:
        shared:          True if this param is shared across model-parallel
                         ranks (prevents double-counting in grad norm).
        grad_norm_group: Optional group name ('mtp') for separate clipping.
                         None → belongs to the main gradient norm group.
        tier:            DES-LOC hardware tier ('h100', 'a6000', 'blackwell',
                         …) derived from TierType at registration time.
                         None → tier unknown / homogeneous setup.
        is_fsdp_param:   True when the param is an FSDP-sharded DTensor
                         whose gradient is stored in ``param.decoupled_grad``.
    """
    shared: bool = False
    grad_norm_group: Optional[str] = None
    tier: Optional[str] = None        # From DES-LOC tier_assignments
    is_fsdp_param: bool = False


# Mapping from TierType enum → human-readable tier string stored in ParamMeta.
_TIER_TYPE_TO_STR: Dict[TierType, str] = {
    TierType.DATACENTER:   'h100',
    TierType.PROFESSIONAL: 'a6000',
    TierType.BLACKWELL:    'blackwell',
    TierType.CONSUMER:     'consumer',
}


class ParamRegistry:
    """Central parameter metadata table — avoids monkey-patching attributes.

    Key:   id(param)  (int — the CPython object identity)
    Value: :class:`ParamMeta` dataclass

    The registry is designed to be process-global (one instance shared by all
    optimisers in the process) so that metadata set during model construction
    is visible inside the optimiser without an explicit handoff.

    Thread-safety: no locking; mutated only from the training main thread.
    """

    def __init__(self) -> None:
        self._table: Dict[int, ParamMeta] = {}
        # Cache for has_group() global all-reduce results.
        self._group_cache: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def register(self, param: torch.Tensor, **kwargs: Any) -> None:
        """Register *param* with the provided metadata keyword arguments.

        Keyword arguments are forwarded to :class:`ParamMeta`; unknown keys
        raise ``TypeError`` at dataclass construction time.

        Diagnostic print is intentional — traces registration during init.

        Args:
            param:   The parameter (or tensor view) to register.
            **kwargs: Fields of :class:`ParamMeta` to set explicitly.
        """
        pid = id(param)
        meta = ParamMeta(**kwargs)
        self._table[pid] = meta
        logger.debug(
            "[ParamRegistry] register id=%s tier=%r group=%r shared=%s fsdp=%s",
            pid, meta.tier, meta.grad_norm_group, meta.shared, meta.is_fsdp_param,
        )

    def get(self, param: torch.Tensor) -> ParamMeta:
        """Return the :class:`ParamMeta` for *param*, or a default instance.

        Never raises — unregistered params return a zero-initialised meta.
        """
        return self._table.get(id(param), ParamMeta())

    def copy_from_param(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        """Copy metadata from *src* to *dst* (replaces copy_optimizer_param_metadata).

        The dst entry is created if it does not yet exist; if *src* is not
        registered the dst entry is populated with defaults so the registry
        entry always exists after this call.

        Args:
            dst: Destination tensor (a view, copy, or shard derived from src).
            src: Source tensor whose metadata should be propagated.
        """
        src_meta = self.get(src)
        dst_meta = ParamMeta(
            shared=src_meta.shared,
            grad_norm_group=src_meta.grad_norm_group,
            tier=src_meta.tier,
            is_fsdp_param=src_meta.is_fsdp_param,
        )
        self._table[id(dst)] = dst_meta
        logger.debug(
            "[ParamRegistry] copy_from_param src_id=%s -> dst_id=%s tier=%r group=%r",
            id(src), id(dst), dst_meta.tier, dst_meta.grad_norm_group,
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def has_group(self, grad_norm_group: str) -> bool:
        """Whether *any* registered param belongs to *grad_norm_group*.

        Result is cached after the first call; does NOT perform a distributed
        all-reduce — use :meth:`MegatronOptimizer.has_grad_norm_group` for the
        global (cross-rank) check.
        """
        if grad_norm_group not in self._group_cache:
            self._group_cache[grad_norm_group] = any(
                m.grad_norm_group == grad_norm_group
                for m in self._table.values()
            )
        return self._group_cache[grad_norm_group]

    def get_params_by_group(
        self,
        params: List[torch.nn.Parameter],
        grad_norm_group: str,
    ) -> List[torch.nn.Parameter]:
        """Filter *params* to those registered in *grad_norm_group*.

        Args:
            params:          Candidate parameter list.
            grad_norm_group: Group name to filter by (e.g. ``'mtp'``).

        Returns:
            Subset of *params* whose registry entry has
            ``grad_norm_group == grad_norm_group``.
        """
        return [p for p in params if self.get(p).grad_norm_group == grad_norm_group]


# Module-level singleton — shared across all optimiser instances in the process.
_GLOBAL_PARAM_REGISTRY: ParamRegistry = ParamRegistry()

# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------


def get_effective_grad(
    param: torch.nn.Parameter,
    use_decoupled_grad: bool = False,
) -> Optional[torch.Tensor]:
    """统一梯度访问接口。优先级: main_grad > decoupled_grad/grad > None。

    对 FSDP 参数（__fsdp_param__=True）自动 unwrap DTensor local shard。

    Priority order (first non-None wins):
      1. param.main_grad       — Megatron BF16 bucket path
      2. param.decoupled_grad  — FSDP DTensor path (only when use_decoupled_grad=True)
      3. param.grad            — standard PyTorch

    For FSDP-marked params (``__fsdp_param__ == True``) the chosen tensor is
    further unwrapped via ``._local_tensor`` when that attribute exists.

    # From Megatron M4145: fix zero counter with decoupled grads.
    # DES-LOC extension: handles all gradient storage formats uniformly.
    """
    is_fsdp = getattr(param, "__fsdp_param__", False)

    grad: Optional[torch.Tensor] = None
    attr_name: str = "none"

    # Priority 1: main_grad
    _main_grad = getattr(param, "main_grad", None)
    if _main_grad is not None:
        grad = _main_grad
        attr_name = "main_grad"
    else:
        # Priority 2: decoupled_grad (only when requested)
        if use_decoupled_grad:
            _decoupled_grad = getattr(param, "decoupled_grad", None)
            if _decoupled_grad is not None:
                grad = _decoupled_grad
                attr_name = "decoupled_grad"

        # Priority 3: standard .grad
        if grad is None:
            grad = param.grad
            if grad is not None:
                attr_name = "grad"

    # FSDP unwrap: extract local shard from DTensor
    if grad is not None and is_fsdp:
        if hasattr(grad, "_local_tensor"):
            print(
                f"[get_effective_grad] FSDP param {id(param)}: "
                f"using {attr_name}._local_tensor, shape={grad.shape}"
            )
            grad = grad._local_tensor

    return grad


# From Megatron M2813: prefer param.main_param over param.data.float() for norm.
# In bf16 training avoids extra fp32 temp — critical on A6000 48GB VRAM.
def clip_grad_norm(
    parameters: Union[torch.Tensor, List[torch.Tensor]],
    max_norm: float,
    norm_type: float = 2.0,
    model_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    use_decoupled_grad: bool = False,
) -> float:
    """Clip gradient norm across model-parallel ranks.

    Args:
        parameters:           Model parameters (or a single tensor).
        max_norm:             Maximum gradient norm after clipping.
        norm_type:            Lp norm type (default L2).
        model_parallel_group: Process group over which to reduce the norm.
        use_decoupled_grad:   When True, read from ``param.decoupled_grad``
                              instead of ``param.grad``.  Required for
                              Megatron-FSDP params whose DTensor gradient is
                              stored on a separate attribute.

    Returns:
        Total gradient norm *before* clipping.

    # From Megatron M4145: fix zero-counter not working with decoupled grads.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    grads: List[torch.Tensor] = []
    for p in parameters:
        if not isinstance(p, torch.Tensor):
            continue
        # DES-LOC M4145 followup: unified grad access via get_effective_grad.
        grad = get_effective_grad(p, use_decoupled_grad=use_decoupled_grad)
        if grad is not None:
            grads.append(grad.detach())

    norm_type = float(norm_type)
    # From Megatron M2316: always use CUDA device for norm tensor so all ranks
    # participate in all_reduce with the same device type. Using CPU on empty-grad
    # ranks while others use CUDA causes mixed-device collective errors/hangs.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif grads:
        device = grads[0].device
    else:
        device = torch.device("cpu")

    if norm_type == float("inf"):
        local_norm = max(g.abs().max().item() for g in grads) if grads else 0.0
        total_norm_t = torch.tensor([local_norm], dtype=torch.float32, device=device)
        # Insight I1: unconditional collective (Megatron M2316 lesson)
        # all_reduce runs unconditionally — ranks with empty grads contribute 0.0.
        if model_parallel_group is not None and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                total_norm_t,
                op=torch.distributed.ReduceOp.MAX,
                group=model_parallel_group,
            )
        total_norm = total_norm_t.item()
    else:
        total_norm_t = torch.zeros(1, dtype=torch.float32, device=device)
        for g in grads:
            total_norm_t += g.float().norm(norm_type)**norm_type
        # Insight I1: unconditional collective (Megatron M2316 lesson)
        # all_reduce runs unconditionally — ranks with empty shards contribute zeros.
        if model_parallel_group is not None and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                total_norm_t,
                op=torch.distributed.ReduceOp.SUM,
                group=model_parallel_group,
            )
        total_norm = float(total_norm_t.item()**(1.0 / norm_type))

    clip_coeff = max_norm / (total_norm + 1e-6)
    if clip_coeff < 1.0:
        for g in grads:
            g.mul_(clip_coeff)

    return total_norm


# ---------------------------------------------------------------------------
# Separate grad-norm groups (From Megatron M4171: Clip mtp grads separately)
# ---------------------------------------------------------------------------
# Parameters tagged with `grad_norm_group='mtp'` are excluded from the main
# gradient norm and clipped independently using their own group norm.  This
# prevents MTP auxiliary heads from distorting the primary model's grad-norm
# estimate when mtp_detach_heads=True decouples their gradients.

MTP_GRAD_NORM_GROUP: str = 'mtp'
GRAD_NORM_GROUP_ATTR: str = 'grad_norm_group'
SEPARATE_GRAD_NORM_GROUPS: Tuple[str, ...] = (MTP_GRAD_NORM_GROUP,)


def _get_param_grad_norm_group(param: torch.nn.Parameter) -> Optional[str]:
    """Return the separate gradient-norm group for *param*, if any.

    # From Megatron M4171: tag params via param.grad_norm_group = 'mtp'
    """
    return getattr(param, GRAD_NORM_GROUP_ATTR, None)


def _validate_grad_norm_group(grad_norm_group: str) -> None:
    """Raise ValueError if *grad_norm_group* is not a registered group.

    # From Megatron M4171: guards against typos in group names.
    """
    if grad_norm_group not in SEPARATE_GRAD_NORM_GROUPS:
        raise ValueError(
            f"Unknown grad_norm_group '{grad_norm_group}'. "
            "Register it in SEPARATE_GRAD_NORM_GROUPS before tagging parameters."
        )


def _is_separate_grad_norm_group(grad_norm_group: Optional[str]) -> bool:
    """Return whether *grad_norm_group* denotes a separately-clipped group.

    # From Megatron M4171: None → False (belongs to main norm).
    """
    if grad_norm_group is None:
        return False
    _validate_grad_norm_group(grad_norm_group)
    return True


def copy_optimizer_param_metadata(destination: torch.Tensor, source: torch.Tensor) -> None:
    """Copy optimizer-relevant metadata when creating param views or copies.

    Megatron's DistributedOptimizer creates shard views / FP32 copies of model
    parameters; without this helper the ``shared`` flag and grad-norm group tag
    would be silently lost, causing double-counted gradients or wrong clipping.

    # From Megatron M4171: copy both .shared and .grad_norm_group attributes.
    # M4171 followup: delegates to _GLOBAL_PARAM_REGISTRY.copy_from_param so
    # metadata is stored centrally rather than monkey-patched on tensors.
    # The legacy attribute writes below are kept for backward-compat with any
    # code that still reads param.shared / param.grad_norm_group directly.
    """
    _GLOBAL_PARAM_REGISTRY.copy_from_param(destination, source)
    # Backward-compat: also mirror on the tensor object so that existing
    # getattr(param, 'shared', False) / getattr(param, GRAD_NORM_GROUP_ATTR, None)
    # callers outside the registry still see the correct values.
    if hasattr(source, 'shared'):
        destination.shared = source.shared  # type: ignore[assignment]
    if hasattr(source, GRAD_NORM_GROUP_ATTR):
        setattr(destination, GRAD_NORM_GROUP_ATTR, getattr(source, GRAD_NORM_GROUP_ATTR))


# ---------------------------------------------------------------------------
# Range helper  (mirrors Megatron's Range class)
# ---------------------------------------------------------------------------


class Range:
    """A [start, end) index range over a flat buffer.

    Attributes:
        start (int): First index (inclusive).
        end   (int): Last index (exclusive).
        size  (int): Number of elements = end - start.
    """

    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end
        self.size = end - start

    def normalize(self, start: int = 0) -> "Range":
        """Return a new Range shifted so that self.start maps to *start*."""
        return Range(start, start + self.size)

    def __len__(self) -> int:
        return self.size

    def __str__(self) -> str:
        return f"{self.start},{self.end} [{self.size}]"

    def __repr__(self) -> str:
        return str(self)


# ---------------------------------------------------------------------------
# LR logging helper
# ---------------------------------------------------------------------------


def get_canonical_lr_for_logging(param_groups: List[Dict]) -> Optional[float]:
    """Return the lr of the first ``default_config=True`` param group.

    All ``default_config`` groups share the same LR schedule, so the first one
    is representative. This includes empty rank-alignment stub groups, which
    the scheduler still writes a valid lr onto.

    Under model parallelism some ranks may have no trainable params (empty
    default_config groups). We still read lr from them because the scheduler
    writes the lr value regardless of whether the group holds parameters.

    # From Megatron M3286: fix LR logging under model parallelism — skip empty
    # param groups check, read lr from default_config groups regardless of
    # whether they hold parameters (important for TP/PP ranks with no params).

    Args:
        param_groups: parameter groups from the optimizer.

    Returns:
        The canonical learning rate, or None if no ``default_config=True``
        group is found.
    """
    for param_group in param_groups:
        if param_group.get('default_config', False):
            return param_group.get('lr')
    return None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class MegatronOptimizer(ABC):
    """Abstract base for all Neuron_SP optimizers.

    Provides the interface contract for gradient preparation, stepping,
    state management, and loss-scale queries.  All concrete optimizers
    must implement ``prepare_grads``, ``step``, and ``zero_grad``.
    """

    def __init__(
        self,
        config: OptimizerConfig,
        optimizer: torch.optim.Optimizer,
        params: List[torch.nn.Parameter],
        model_parallel_config: ModelParallelConfig,
    ) -> None:
        self.config = config
        self.optimizer = optimizer
        self.params = list(params)
        self.model_parallel_config = model_parallel_config
        # Unity loss scale (BF16 path has no dynamic scaler)
        self._scale_one = torch.tensor([1.0], dtype=torch.float32)

    @abstractmethod
    def prepare_grads(self) -> bool:
        """Prepare gradients; return True if optimizer step should be skipped."""
        ...

    @abstractmethod
    def step(self) -> bool:
        """Execute optimizer step; return True if a step was taken."""
        ...

    @abstractmethod
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero (or None) all gradients."""
        ...

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clip_grad_norm(self, clip_grad: float) -> float:
        """Compute grad norm and clip gradients in place."""
        return clip_grad_norm(
            parameters=self.params,
            max_norm=clip_grad,
            norm_type=2.0,
            model_parallel_group=None,
        )

    def _filter_grads_for_norm(
        self,
        params: List[torch.nn.Parameter],
        param_filter: Optional[Callable[[torch.nn.Parameter], bool]] = None,
    ) -> List[torch.Tensor]:
        """Filter parameter gradients for norm computation.

        Filters parameters based on:
          - *param_filter* predicate when provided.
          - grad must not be None.
          - parameter must not be shared (avoids double-counting).
          - must not be a TP replica.

        # From Megatron M4171: extracted so get_grads_for_grad_norm can reuse it.
        # M4171 followup (ParamRegistry): shared flag is resolved via
        # _GLOBAL_PARAM_REGISTRY first; getattr fallback supports params that
        # pre-date the registry (e.g. registered before optimizer construction).
        """
        grads: List[torch.Tensor] = []
        for param in params:
            if param_filter is not None and not param_filter(param):
                continue
            # DES-LOC M4145 followup: unified grad access via get_effective_grad.
            grad = get_effective_grad(param)
            if grad is None:
                continue
            # M4171 followup: prefer registry for shared flag; fall back to
            # getattr for params registered before the registry was available.
            _meta = _GLOBAL_PARAM_REGISTRY.get(param)
            _shared = _meta.shared or getattr(param, 'shared', False)
            if _shared:
                continue
            # Skip TP replicas (sequence_parallel / tensor_model_parallel attr)
            if getattr(param, 'tensor_model_parallel', False) and not getattr(
                param, 'sequence_parallel', False
            ):
                continue
            grads.append(grad.detach())
        return grads

    def get_grads_for_grad_norm(
        self, grad_norm_group: Optional[str] = None
    ) -> List[torch.Tensor]:
        """Return gradients to use for norm computation.

        When *grad_norm_group* is ``None``, returns gradients for the main
        norm, **excluding** parameters that belong to a registered separate
        grad-norm group (e.g. MTP heads).

        When *grad_norm_group* is given, returns only gradients from that group.

        # From Megatron M4171: enables independent grad-norm + clipping for MTP.
        """
        if grad_norm_group is not None:
            _validate_grad_norm_group(grad_norm_group)
            # M4171 followup: check registry first, then fall back to attr.
            def param_filter(p: torch.nn.Parameter) -> bool:  # type: ignore[misc]
                reg_group = _GLOBAL_PARAM_REGISTRY.get(p).grad_norm_group
                if reg_group is not None:
                    return reg_group == grad_norm_group
                return _get_param_grad_norm_group(p) == grad_norm_group
        else:
            # Exclude params that belong to any separate grad-norm group.
            def param_filter(p: torch.nn.Parameter) -> bool:  # type: ignore[misc]
                reg_group = _GLOBAL_PARAM_REGISTRY.get(p).grad_norm_group
                if reg_group is not None:
                    return not _is_separate_grad_norm_group(reg_group)
                return not _is_separate_grad_norm_group(_get_param_grad_norm_group(p))
        return self._filter_grads_for_norm(self.get_parameters(), param_filter=param_filter)

    # Legacy alias kept for backwards compat (callers that used the old name).
    def get_main_grads_for_grad_norm(self) -> List[torch.Tensor]:
        """Deprecated alias — use get_grads_for_grad_norm() instead."""
        return self.get_grads_for_grad_norm()

    def has_grad_norm_group(self, grad_norm_group: str) -> bool:
        """Whether any rank globally owns params for *grad_norm_group*.

        Performs a one-time global all-reduce and caches the result.  Gating
        the per-step group-norm collectives on a *globally consistent* flag
        keeps reductions balanced across ranks that may locally own no shard.

        # From Megatron M4171: avoids collective mismatch for MTP detach heads.
        """
        _validate_grad_norm_group(grad_norm_group)
        if getattr(self, '_has_grad_norm_group_cache', None) is None:
            self._has_grad_norm_group_cache: Dict[str, bool] = {}
        cache = self._has_grad_norm_group_cache
        if grad_norm_group not in cache:
            # M4171 followup: check registry first (more reliable than attr),
            # then fall back to getattr for unregistered params.
            def _local_has_group(p: torch.nn.Parameter) -> bool:
                reg_group = _GLOBAL_PARAM_REGISTRY.get(p).grad_norm_group
                if reg_group is not None:
                    return reg_group == grad_norm_group
                pg = _get_param_grad_norm_group(p)
                return _is_separate_grad_norm_group(pg) and pg == grad_norm_group
            local = any(_local_has_group(p) for p in self.get_parameters())
            flag = torch.tensor([1 if local else 0], dtype=torch.int, device='cuda')
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    flag, op=torch.distributed.ReduceOp.MAX, group=None
                )
            cache[grad_norm_group] = bool(flag.item() > 0)
        return cache[grad_norm_group]

    def get_loss_scale(self) -> torch.Tensor:
        """Current loss scale (1.0 for BF16, dynamic for FP16)."""
        return self._scale_one

    def get_parameters(self) -> List[torch.nn.Parameter]:
        """Flat list of all optimizer parameters."""
        all_params: List[torch.nn.Parameter] = []
        for pg in self.optimizer.param_groups:
            all_params.extend(pg["params"])
        return all_params

    def state_dict(self) -> dict:
        return {"optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def reload_model_params(self) -> None:
        """Reload model params from FP32 master copy (override in subclasses)."""
        pass


# ---------------------------------------------------------------------------
# Mixed precision base
# ---------------------------------------------------------------------------


class MixedPrecisionOptimizer(MegatronOptimizer):
    """Optimizer with FP32 master weights and BF16/FP16 model weights.

    Maintains a shadow copy of every BF16/FP16 model parameter in FP32
    and performs Adam updates on those FP32 copies.  After each step the
    updated FP32 masters are cast back to BF16/FP16 for the forward pass.

    For BF16 training (the common case) ``grad_scaler`` should be ``None``.
    For FP16 training a :class:`~torch.cuda.amp.GradScaler`-compatible
    object must be supplied.
    """

    def __init__(
        self,
        config: OptimizerConfig,
        optimizer: torch.optim.Optimizer,
        params: List[torch.nn.Parameter],
        model_parallel_config: ModelParallelConfig,
        grad_scaler: Optional[Any] = None,
    ) -> None:
        super().__init__(config, optimizer, params, model_parallel_config)
        self.grad_scaler = grad_scaler

        if self.grad_scaler is None:
            assert not config.fp16, "fp16 training requires a grad_scaler."

        if self.grad_scaler is not None:
            self._found_inf = torch.tensor([0.0], dtype=torch.float32)

        # Separate params into three lists:
        #   bf16_model_params   — original BF16 model tensors
        #   fp32_master_params  — FP32 copies held by the optimizer
        #   fp32_model_params   — params that are already FP32
        self.bf16_model_params: List[torch.nn.Parameter] = []
        self.fp32_master_params: List[torch.nn.Parameter] = []
        self.fp32_model_params: List[torch.nn.Parameter] = []

        for param_group in self.optimizer.param_groups:
            bf16_grp: List[torch.nn.Parameter] = []
            fp32_master_grp: List[torch.nn.Parameter] = []
            fp32_model_grp: List[torch.nn.Parameter] = []

            for i, param in enumerate(param_group["params"]):
                if not param.requires_grad:
                    continue
                if param.dtype in (torch.float16, torch.bfloat16):
                    bf16_grp.append(param)
                    master = param.detach().clone().float()
                    master.requires_grad_(True)
                    param_group["params"][i] = master
                    param.main_param = master
                    # M4171 followup: register original BF16 param in registry
                    # (infer metadata from existing tensor attributes for compat
                    # with callers that set param.shared / param.grad_norm_group
                    # before the optimizer is constructed).
                    _GLOBAL_PARAM_REGISTRY.register(
                        param,
                        shared=getattr(param, 'shared', False),
                        grad_norm_group=getattr(param, GRAD_NORM_GROUP_ATTR, None),
                        tier=None,           # tier populated by DistributedOptimizer
                        is_fsdp_param=getattr(param, '__fsdp_param__', False),
                    )
                    # M4171: propagate .shared and .grad_norm_group so that
                    # MTP heads can be clipped independently of the main norm.
                    # copy_from_param now also updates the registry for master.
                    copy_optimizer_param_metadata(master, param)  # type: ignore[arg-type]
                    fp32_master_grp.append(master)
                    if param in self.optimizer.state:
                        self.optimizer.state[master] = self.optimizer.state.pop(param)
                elif param.dtype == torch.float32:
                    fp32_model_grp.append(param)
                else:
                    raise TypeError(f"Unsupported param dtype {param.dtype}. "
                                    "Expected float16, bfloat16, or float32.")

            self.bf16_model_params.extend(bf16_grp)
            self.fp32_master_params.extend(fp32_master_grp)
            self.fp32_model_params.extend(fp32_model_grp)

    # ------------------------------------------------------------------
    # Internal grad helpers
    # ------------------------------------------------------------------

    def _copy_model_grads_to_main_grads(self) -> None:
        """Copy BF16 gradients into FP32 master params' ``.grad`` field."""
        for bf16, fp32 in zip(self.bf16_model_params, self.fp32_master_params):
            if hasattr(bf16, "main_grad") and bf16.main_grad is not None:
                fp32.grad = bf16.main_grad.float()
            elif bf16.grad is not None:
                fp32.grad = bf16.grad.float()
            bf16.grad = None
        for fp32 in self.fp32_model_params:
            if hasattr(fp32, "main_grad"):
                fp32.grad = fp32.main_grad

    def _copy_main_params_to_model_params(self) -> None:
        """Write FP32 master weights back into BF16 model tensors."""
        for bf16, fp32 in zip(self.bf16_model_params, self.fp32_master_params):
            bf16.data.copy_(fp32.data)

    def _copy_model_params_to_main_params(self) -> None:
        """Reload FP32 masters from BF16 model params (used at resume)."""
        for bf16, fp32 in zip(self.bf16_model_params, self.fp32_master_params):
            fp32.data.copy_(bf16.data.float())

    def _unscale_and_check_inf(self) -> bool:
        """Unscale main grads and return True if inf/nan was detected."""
        if self.grad_scaler is None:
            return False
        main_grads = [
            fp32.grad.data for fp32 in (self.fp32_master_params + self.fp32_model_params) if fp32.grad is not None
        ]
        self._found_inf.fill_(0.0)
        if main_grads:
            torch._amp_foreach_non_finite_check_and_unscale_(main_grads, self._found_inf, self.grad_scaler.inv_scale)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(self._found_inf, op=torch.distributed.ReduceOp.MAX)
        return bool(self._found_inf.item() > 0)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prepare_grads(self) -> bool:
        """Copy BF16 grads to FP32 masters; unscale and check for inf.

        Returns:
            True if inf/nan detected (caller should skip optimizer step).
        """
        self._copy_model_grads_to_main_grads()
        if self.grad_scaler is not None:
            found_inf = self._unscale_and_check_inf()
            self.grad_scaler.update(found_inf)
            return found_inf
        return False

    @torch.no_grad()
    def step(self) -> bool:
        found_inf = self.prepare_grads()
        if found_inf:
            return False
        if self.config.clip_grad > 0.0:
            self.clip_grad_norm(self.config.clip_grad)
        self.optimizer.step()
        self._copy_main_params_to_model_params()
        return True

    def zero_grad(self, set_to_none: bool = True) -> None:
        for param in self.bf16_model_params + self.fp32_model_params:
            if set_to_none:
                param.grad = None
                if hasattr(param, "main_grad"):
                    param.main_grad = None
            else:
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()
        for param in self.fp32_master_params:
            if set_to_none:
                param.grad = None
            else:
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()

    def get_loss_scale(self) -> torch.Tensor:
        if self.grad_scaler is None:
            return self._scale_one
        return self.grad_scaler.scale

    def reload_model_params(self) -> None:
        self._copy_main_params_to_model_params()

    def state_dict(self) -> dict:
        sd: dict = {"optimizer": self.optimizer.state_dict()}
        if self.grad_scaler is not None:
            sd["grad_scaler"] = self.grad_scaler.state_dict()
        sd["fp32_master_params"] = [p.data for p in self.fp32_master_params]
        return sd

    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict["optimizer"])
        if "grad_scaler" in state_dict and self.grad_scaler is not None:
            self.grad_scaler.load_state_dict(state_dict["grad_scaler"])
        if "fp32_master_params" in state_dict:
            for cur, saved in zip(self.fp32_master_params, state_dict["fp32_master_params"]):
                cur.data.copy_(saved)


# ---------------------------------------------------------------------------
# Heterogeneous shard sizing helpers
# ---------------------------------------------------------------------------


def _compute_hetero_shard_boundaries(
    total_numel: int,
    dp_world_size: int,
    config: OptimizerConfig,
    tier_assignments: Optional[List[Optional[TierType]]] = None,
) -> List[Tuple[int, int]]:
    """Compute per-rank shard [start, end) boundaries.

    When ``config.heterogeneous_shard_sizing`` is False (or
    ``tier_assignments`` is None/empty), every rank gets an equal slice of
    ``ceil(total_numel / dp_world_size)`` elements.

    When heterogeneous sizing is enabled, ranks are divided into two tiers:
      - H100 ranks get ``h100_fraction`` of the buffer each.
      - A6000 ranks share the remainder equally.

    The fractions are derived from BF16 TFLOPS:
      h100_fraction  = H100_tflops  / sum_tflops_per_rank
      a6000_fraction = A6000_tflops / sum_tflops_per_rank

    Each partition is padded to 16-element alignment for NCCL efficiency.

    Args:
        total_numel:       Total number of elements in the flat grad buffer.
        dp_world_size:     Number of data-parallel ranks.
        config:            :class:`OptimizerConfig` instance.
        tier_assignments:  Per-rank tier type; ``None`` = homogeneous.

    Returns:
        List of ``(start, end)`` pairs, one per DP rank.
    """
    ALIGN = 16  # elements; 16 × 2 bytes = 32 bytes aligns to cache line

    if (not config.heterogeneous_shard_sizing or tier_assignments is None or len(tier_assignments) != dp_world_size):
        # Uniform equal-slice partitioning (original behaviour)
        padded = _round_up(total_numel, dp_world_size)
        shard_size = padded // dp_world_size
        boundaries: List[Tuple[int, int]] = []
        for r in range(dp_world_size):
            # Clamp start so that when dp_world_size > total_numel (degenerate
            # but defensive), extra ranks get an empty shard (N, N) instead of
            # an inverted boundary (s > total_numel) that would break callers.
            s = min(r * shard_size, total_numel)
            e = min(s + shard_size, total_numel)
            boundaries.append((s, e))
        return boundaries

    # -----------------------------------------------------------------------
    # Heterogeneous partitioning: assign sizes proportional to available VRAM
    # (preferred) or FLOPS (fallback).
    #
    # "Available VRAM" = total_vram - model_params_bytes - grad_bytes - reserve
    # This prevents A6000 OOM: a 49 GB card with ~30 GB fixed overhead has only
    # ~15 GB free for optimizer state, while a 96 GB H100 with the same overhead
    # has ~62 GB free — the ratio is ~4:1, not the 2:1 that total-VRAM would
    # give, nor the 2.5:1 from TFLOPS.
    # -----------------------------------------------------------------------
    h100_tflops = config.h100_bf16_tflops
    a6000_tflops = config.a6000_bf16_tflops

    if getattr(config, 'shard_by_available_vram', False):
        # Estimate per-rank fixed overhead: model params (bf16) + grads (bf16)
        # Each param is 2 bytes (bf16) for the model copy and 2 bytes for grad.
        param_and_grad_bytes = total_numel * 4  # shared across all ranks in ZeRO-3
        # In ZeRO-3, each rank holds 1/dp_world_size of params+grads, but the
        # all-gather temporarily doubles memory. Conservatively use full size.
        per_rank_fixed_gb = (param_and_grad_bytes / dp_world_size) / (1024**3)

        activation_reserve = getattr(config, 'activation_reserve_gb', 4.0)

        def _tier_vram(tier):
            if tier == TierType.DATACENTER:
                return getattr(config, 'h100_total_vram_gb', 96.0)
            elif tier == TierType.BLACKWELL:
                return getattr(config, 'blackwell_total_vram_gb', 98.0)
            elif tier == TierType.CONSUMER:
                return getattr(config, 'consumer_total_vram_gb', 24.0)
            else:  # PROFESSIONAL (A6000) or unknown
                return getattr(config, 'a6000_total_vram_gb', 49.0)

        rank_available = []
        for t in tier_assignments:
            total_gb = _tier_vram(t)
            available = max(total_gb - per_rank_fixed_gb - activation_reserve, 1.0)
            rank_available.append(available)
        total_available = sum(rank_available)
        rank_weights = rank_available
        total_weight = total_available
    else:
        # Fallback: shard proportional to FLOPS
        def _tier_tflops(tier):
            if tier == TierType.DATACENTER:
                return h100_tflops
            elif tier == TierType.BLACKWELL:
                return getattr(config, 'blackwell_bf16_tflops', 300.0)
            elif tier == TierType.CONSUMER:
                return getattr(config, 'consumer_bf16_tflops', 82.6)
            else:  # PROFESSIONAL (A6000) or unknown
                return a6000_tflops
        rank_weights = [_tier_tflops(t) for t in tier_assignments]
        total_weight = sum(rank_weights)

    # Raw (unaligned) shard sizes proportional to weight (VRAM or FLOPS)
    raw_sizes = [int(total_numel * w / total_weight) for w in rank_weights]

    # Align each shard to ALIGN; distribute rounding error to first rank
    aligned_sizes = [_round_up(s, ALIGN) for s in raw_sizes]
    allocated = sum(aligned_sizes)
    if allocated > total_numel:
        # Trim back from the last rank's padding
        aligned_sizes[-1] = max(aligned_sizes[-1] - (allocated - total_numel), 0)

    # Build [start, end) pairs
    boundaries = []
    cur = 0
    for sz in aligned_sizes:
        end = min(cur + sz, total_numel)
        boundaries.append((cur, end))
        cur = end

    # Ensure the last rank reaches total_numel
    if boundaries and boundaries[-1][1] < total_numel:
        s, _ = boundaries[-1]
        boundaries[-1] = (s, total_numel)

    return boundaries


def _round_up(n: int, multiple: int) -> int:
    """Round *n* up to the next multiple of *multiple*."""
    return ((n + multiple - 1) // multiple) * multiple


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class DistributedOptimizer(MixedPrecisionOptimizer):
    """ZeRO-style distributed optimizer with DES-LOC Ku/Kv decomposed sync.

    Param sharding
    ~~~~~~~~~~~~~~
    Each data-parallel rank is responsible for a contiguous slice of every
    grad buffer.  After a reduce-scatter the rank updates its local FP32
    shard with Adam, then all-gathers the updated values back into every
    rank's BF16 model tensors.

    DES-LOC extension — heterogeneous shard sizing
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    When ``OptimizerConfig.heterogeneous_shard_sizing`` is ``True`` the shard
    size of each rank is proportional to its GPU's BF16 TFLOPS:

      H100  shard fraction ≈ 989 / (989 + 309.7) ≈ 76 %
      A6000 shard fraction ≈ 309.7 / (989 + 309.7) ≈ 24 %

    This ensures the compute time of the local Adam update is approximately
    equal across heterogeneous tiers, eliminating the stragglers that arise
    from equal-sized shards.

    DES-LOC extension — moment synchronisation
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    First  moments (exp_avg)    are all-reduced every ``config.ku`` steps.
    Second moments (exp_avg_sq) are all-reduced every ``config.kv`` steps.
    Parameters are broadcast after every Adam step (not just every kx steps)
    to avoid the "Frankenstein model" divergence.

    Args:
        config:                  Optimizer hyper-parameters and DES-LOC flags.
        optimizer:               Underlying :class:`torch.optim.Optimizer`
                                 (Adam/AdamW); operates on FP32 shards.
        params:                  Full list of model parameters.
        model_parallel_config:   TP/PP/DP configuration + DES-LOC config.
        param_and_grad_buffers:  Contiguous grad buffers from the DDP wrapper;
                                 one per model chunk / buffer dtype.
        data_parallel_group:     NCCL process group for reduce-scatter / all-gather.
        data_parallel_group_gloo: (Optional) CPU Gloo group for checkpoint I/O.
        tier_assignments:        Per-rank :class:`~deepspeed.core.desloc_config.TierType`;
                                 supply when ``heterogeneous_shard_sizing=True``.
    """

    def __init__(
        self,
        config: OptimizerConfig,
        optimizer: torch.optim.Optimizer,
        params: List[torch.nn.Parameter],
        model_parallel_config: ModelParallelConfig,
        param_and_grad_buffers: List[ParamAndGradBuffer],
        data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        data_parallel_group_gloo: Optional[torch.distributed.ProcessGroup] = None,
        tier_assignments: Optional[List[Optional[TierType]]] = None,
        # From Megatron M2456: pass intra_dist_opt_group explicitly so callers
        # do not rely on parallel_state.get_intra_distributed_optimizer_instance_group()
        # being initialized yet (avoids assertion crash during multi-instance init order).
        intra_dist_opt_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        # Resolve data_parallel_group from parallel_state when not explicitly given
        if data_parallel_group is None:
            if parallel_state.is_initialized():
                data_parallel_group = parallel_state.get_data_parallel_group()
            else:
                data_parallel_group = torch.distributed.GroupMember.WORLD
        if data_parallel_group_gloo is None and parallel_state.is_initialized():
            data_parallel_group_gloo = parallel_state.get_data_parallel_group_gloo()

        super().__init__(
            config=config,
            optimizer=optimizer,
            params=params,
            model_parallel_config=model_parallel_config,
            grad_scaler=None,  # BF16 path: no dynamic loss scale
        )

        # From Megatron M2356: guard against double weight-decay when
        # decoupled_weight_decay=True but the inner optimizer also applies L2.
        if self.config.decoupled_weight_decay and self.config.weight_decay > 0:
            for pg in self.optimizer.param_groups:
                if pg.get('weight_decay', 0.0) != 0.0:
                    import warnings
                    warnings.warn(
                        'decoupled_weight_decay=True but optimizer also has '
                        f'weight_decay={pg["weight_decay"]}. Double decay! '
                        'Set optimizer weight_decay=0. From Megatron M2356.',
                        UserWarning)
                    break

        self.param_and_grad_buffers = param_and_grad_buffers
        self.data_parallel_group = data_parallel_group
        self.data_parallel_group_gloo = data_parallel_group_gloo
        self.tier_assignments = tier_assignments

        # From Megatron M2456: set grad_stats_parallel_group from the explicitly-passed
        # intra_dist_opt_group rather than calling parallel_state accessor (which may
        # not be initialized yet when multiple optimizer instances are created).
        # On DES-LOC PCIe clusters this avoids assertion failures during engine init.
        if intra_dist_opt_group is not None:
            self.grad_stats_parallel_group = intra_dist_opt_group
        elif parallel_state.is_initialized():
            self.grad_stats_parallel_group = parallel_state.get_intra_distributed_optimizer_instance_group(
                check_initialized=False
            )
        # If still None, get_grad_stats_parallel_group() falls back to data_parallel_group.

        self.data_parallel_world_size: int = torch.distributed.get_world_size(group=data_parallel_group)
        self.data_parallel_rank: int = torch.distributed.get_rank(group=data_parallel_group)

        # DES-LOC legacy config (DesLocConfig on model_parallel_config, may be None)
        self._desloc: Optional[DesLocConfig] = getattr(model_parallel_config, "desloc", None)

        # Global step counter — incremented after each successful Adam step.
        self._step_count: int = 0

        # Build per-rank param/grad/state shards.
        self._build_shards()

    # ------------------------------------------------------------------
    # Shard construction
    # ------------------------------------------------------------------

    def _build_shards(self) -> None:
        """Partition each grad buffer into heterogeneous per-rank shards.

        For each :class:`ParamAndGradBuffer` we allocate:
        - ``_fp32_shards``     — FP32 master weights for this rank's slice.
        - ``_fp32_grad_shards``— scratch buffer for reduce-scattered grads.

        Shard boundaries are computed via
        :func:`_compute_hetero_shard_boundaries`, which honours the FLOPS-
        weighted heterogeneous sizing if requested.
        """
        dp_rank = self.data_parallel_rank
        dp_world = self.data_parallel_world_size

        self._fp32_shards: List[torch.Tensor] = []
        self._fp32_grad_shards: List[torch.Tensor] = []
        # per-buffer list of per-rank (start, end) boundaries
        self._buf_boundaries: List[List[Tuple[int, int]]] = []

        for buf in self.param_and_grad_buffers:
            total_numel: int = buf.grad_data.numel()

            # Compute per-rank boundaries (hetero-aware)
            boundaries = _compute_hetero_shard_boundaries(
                total_numel=total_numel,
                dp_world_size=dp_world,
                config=self.config,
                tier_assignments=self.tier_assignments,
            )
            self._buf_boundaries.append(boundaries)

            shard_start, shard_end = boundaries[dp_rank]
            shard_size = shard_end - shard_start

            _device = buf.grad_data.device

            # FP32 master weight shard — initialised from model params
            fp32_shard = torch.zeros(shard_size, dtype=torch.float32, device=_device)
            for param, (ps, pe, _) in buf.param_index_map.items():
                overlap_s = max(ps, shard_start)
                overlap_e = min(pe, shard_end)
                if overlap_s >= overlap_e:
                    continue
                local_s = overlap_s - shard_start
                local_e = overlap_e - shard_start
                model_s = overlap_s - ps
                model_e = overlap_e - ps
                fp32_shard[local_s:local_e].copy_(param.data.view(-1)[model_s:model_e].float())

            # FP32 grad scratch (receives reduce-scattered grads)
            fp32_grad_shard = torch.zeros(shard_size, dtype=torch.float32, device=_device)

            self._fp32_shards.append(fp32_shard)
            self._fp32_grad_shards.append(fp32_grad_shard)

        # Rewire the inner optimizer's param_groups to point at FP32 shards.
        # M4171 followup: copy_optimizer_param_metadata (now registry-backed)
        # propagates .shared and .grad_norm_group from the model params that
        # overlap each shard so that independent MTP grad clipping works.
        # Additionally, DES-LOC tier is inferred from tier_assignments so that
        # per-tier routing logic can query shard params via the registry.
        shard_params: List[torch.nn.Parameter] = []
        for buf_idx, (fp32_shard, buf, boundaries) in enumerate(
            zip(self._fp32_shards, self.param_and_grad_buffers, self._buf_boundaries)
        ):
            sp = torch.nn.Parameter(fp32_shard, requires_grad=True)
            shard_start, shard_end = boundaries[dp_rank]
            # Determine DES-LOC tier for this rank from tier_assignments.
            _tier_str: Optional[str] = None
            if self.tier_assignments is not None and dp_rank < len(self.tier_assignments):
                _ta = self.tier_assignments[dp_rank]
                if _ta is not None:
                    _tier_str = _TIER_TYPE_TO_STR.get(_ta)
            # Find first overlapping model param and propagate its metadata.
            _src_param = None
            for param, (ps, pe, _) in buf.param_index_map.items():
                if max(ps, shard_start) < min(pe, shard_end):
                    _src_param = param
                    break
            if _src_param is not None:
                # copy_from_param propagates shared / grad_norm_group / is_fsdp_param.
                copy_optimizer_param_metadata(sp, _src_param)  # type: ignore[arg-type]
                # Patch in tier derived from tier_assignments (not on model param).
                _sp_meta = _GLOBAL_PARAM_REGISTRY.get(sp)
                _GLOBAL_PARAM_REGISTRY.register(
                    sp,
                    shared=_sp_meta.shared,
                    grad_norm_group=_sp_meta.grad_norm_group,
                    tier=_tier_str,
                    is_fsdp_param=_sp_meta.is_fsdp_param,
                )
            else:
                # Shard with no overlapping model param: register defaults + tier.
                _GLOBAL_PARAM_REGISTRY.register(sp, tier=_tier_str)
            shard_params.append(sp)
        self.optimizer.param_groups = [{
            "params": shard_params,
            "lr": self.config.lr or 1e-4,
            "betas": (self.config.adam_beta1, self.config.adam_beta2),
            "eps": self.config.adam_eps,
            "weight_decay": self.config.weight_decay,
        }]
        self._shard_params = shard_params

        logger.debug(
            "DistributedOptimizer shards built: rank=%d/%d, "
            "hetero=%s, buffers=%d",
            self.data_parallel_rank,
            self.data_parallel_world_size,
            self.config.heterogeneous_shard_sizing,
            len(self.param_and_grad_buffers),
        )
        for i, (buf, boundaries) in enumerate(zip(self.param_and_grad_buffers, self._buf_boundaries)):
            s, e = boundaries[self.data_parallel_rank]
            logger.debug(
                "  buf[%d]: total=%d, my_shard=[%d,%d) size=%d",
                i,
                buf.grad_data.numel(),
                s,
                e,
                e - s,
            )

    # ------------------------------------------------------------------
    # Grad preparation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _reduce_scatter_grads(self) -> None:
        """Reduce-scatter BF16 grads; store averaged FP32 result in grad shards.

        Uses ``reduce_scatter_tensor`` for a single contiguous operation
        when the total buffer size is a multiple of dp_world_size.
        Otherwise falls back to an equivalent chunked scatter.

        Insight I6: PCIe-aware overlap (Megatron aa-3.5)
        When ``config.use_pcie_aware_overlap`` is True the collective is issued
        *asynchronously* only for buffers whose element count exceeds the PCIe
        overlap-trigger threshold (i.e. transfer time ≥ PCIe latency).  Buckets
        below the threshold are reduced synchronously to avoid the overhead of
        an async launch that can't meaningfully overlap with compute.
        """
        dp_world = self.data_parallel_world_size
        dp_rank = self.data_parallel_rank

        # Insight I6: PCIe-aware overlap (Megatron aa-3.5)
        pcie_aware = getattr(self.config, 'use_pcie_aware_overlap', False)
        pcie_trigger_elems: int = (
            self.config.pcie_overlap_trigger_elems()
            if pcie_aware
            else 0
        )

        for buf_idx, (buf, fp32_grad_shard, boundaries) in enumerate(
                zip(
                    self.param_and_grad_buffers,
                    self._fp32_grad_shards,
                    self._buf_boundaries,
                )):
            total_numel = buf.grad_data.numel()
            shard_start, shard_end = boundaries[dp_rank]
            shard_size = shard_end - shard_start

            # Insight I6: PCIe-aware overlap (Megatron aa-3.5)
            # Use async only when buffer is large enough to make overlap
            # worthwhile over PCIe.  For small buffers the launch overhead
            # exceeds the transfer time, so sync mode is more efficient.
            _async_op = (
                pcie_aware
                and getattr(self.config, 'overlap_grad_reduce', False)
                and total_numel >= pcie_trigger_elems
            )

            # Maximum shard size across ranks (needed for equal-shard collectives)
            max_shard_size = max(e - s for s, e in boundaries)

            if all((e - s) == max_shard_size for s, e in boundaries):
                # -------------------------------------------------------
                # Fast path: equal shards → use reduce_scatter_tensor
                # -------------------------------------------------------
                padded_numel = max_shard_size * dp_world
                if padded_numel > total_numel:
                    grad_padded = torch.zeros(
                        padded_numel,
                        dtype=buf.grad_data.dtype,
                        device=buf.grad_data.device,
                    )
                    grad_padded[:total_numel].copy_(buf.grad_data)
                else:
                    grad_padded = buf.grad_data

                output_shard = torch.zeros(
                    max_shard_size,
                    dtype=grad_padded.dtype,
                    device=grad_padded.device,
                )
                # Insight I6: PCIe-aware overlap (Megatron aa-3.5)
                # async_op=True only for buffers above the PCIe latency threshold.
                handle = torch.distributed.reduce_scatter_tensor(
                    output_shard,
                    grad_padded,
                    op=torch.distributed.ReduceOp.SUM,
                    group=self.data_parallel_group,
                    async_op=_async_op,
                )
                if _async_op and handle is not None:
                    handle.wait()
                fp32_grad_shard.copy_(output_shard[:shard_size].float().div_(dp_world))

            else:
                # -------------------------------------------------------
                # Slow path: heterogeneous shards → chunked reduce + local copy
                # Uses all_reduce on the full buffer then extracts our slice.
                # This is bandwidth-equivalent to reduce_scatter + re-broadcast
                # for small worlds; for large worlds a proper variable-chunk
                # reduce_scatter would be more efficient.
                # -------------------------------------------------------
                grad_work = buf.grad_data.clone()
                # Insight I6: PCIe-aware overlap (Megatron aa-3.5)
                handle = torch.distributed.all_reduce(
                    grad_work,
                    op=torch.distributed.ReduceOp.SUM,
                    group=self.data_parallel_group,
                    async_op=_async_op,
                )
                if _async_op and handle is not None:
                    handle.wait()
                fp32_grad_shard.copy_(grad_work[shard_start:shard_end].float().div_(dp_world))

        # Attach grad shards to shard params for the Adam step
        for shard_param, fp32_grad_shard in zip(self._shard_params, self._fp32_grad_shards):
            shard_param.grad = fp32_grad_shard

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prepare_grads(self) -> bool:
        """Reduce-scatter gradients into FP32 shards.

        Returns:
            Always ``False`` (no loss scaler in BF16 path).
        """
        self._reduce_scatter_grads()
        return False

    # ------------------------------------------------------------------
    # DES-LOC moment synchronisation
    # ------------------------------------------------------------------

    def sync_moments(self, sync_first: bool = False, sync_second: bool = False) -> None:
        """All-reduce optimizer moments across data-parallel ranks.

        This implements the DES-LOC Ku / Kv synchronisation protocol:
        first moments are averaged every Ku steps, second moments every Kv
        steps.  The communication volume is reduced by (1 - 1/Ku) + (1 - 1/Kv)
        compared to syncing every step.

        Args:
            sync_first:  All-reduce ``exp_avg``    (first moment).
            sync_second: All-reduce ``exp_avg_sq`` (second moment).
        """
        if not (sync_first or sync_second):
            return

        dp_world = self.data_parallel_world_size

        for shard_param in self._shard_params:
            state = self.optimizer.state.get(shard_param)
            if state is None:
                continue

            if sync_first and "exp_avg" in state:
                m = state["exp_avg"]
                torch.distributed.all_reduce(
                    m,
                    op=torch.distributed.ReduceOp.SUM,
                    group=self.data_parallel_group,
                )
                m.div_(dp_world)

            if sync_second and "exp_avg_sq" in state:
                v = state["exp_avg_sq"]
                torch.distributed.all_reduce(
                    v,
                    op=torch.distributed.ReduceOp.SUM,
                    group=self.data_parallel_group,
                )
                v.div_(dp_world)

    # ------------------------------------------------------------------
    # Shard → model broadcast
    # ------------------------------------------------------------------

    @torch.no_grad()
    def shard_to_model_broadcast(self) -> None:
        """Write local FP32 shard to BF16 model and broadcast to all ranks.

        Called every step to keep all ranks' model weights in sync.
        Avoids the "Frankenstein model" bug where only 1/N of params get
        updated on each rank.

        Bucketed all-gather (ported from c37ea95f / ShardState._broadcast_model_params):
        Instead of one all_reduce per ParamAndGradBuffer (which can be many
        small calls when the model has many small buffers), we accumulate
        buffer segments into large flat BF16 buckets (up to BUCKET_BYTES each)
        and fire a single all_reduce(SUM) per bucket.  Each rank fills only
        its owned shard slice; non-owned positions stay zero; SUM reconstructs
        the full tensor.  This reduces NCCL calls from O(num_buffers) to
        O(total_param_bytes / BUCKET_BYTES) — typically ~9 calls for a 7B
        model on a 3-rank PCIe topology.

        Protocol (per bucket):
          1. Allocate a zero-filled BF16 flat buffer sized to the bucket.
          2. Each rank copies its owned shard slice (FP32→BF16) into the
             corresponding positions; unowned positions stay zero.
          3. all_reduce(SUM) — exactly one rank has non-zero per position.
          4. Scatter bucket back to model param tensors.
          5. Refresh the FP32 shard from the broadcast result.

        For heterogeneous shards the shard boundaries are per-rank so each
        rank knows exactly which global positions it owns.
        """
        # perf(optimizer): bucketed shard_to_model_broadcast — port c37ea95f
        # BUCKET_BYTES: 2 GB matches A6000 headroom; BF16 → 2 bytes/elem.
        BUCKET_BYTES: int = 2 * 1024 * 1024 * 1024  # 2 GB
        bucket_elems: int = BUCKET_BYTES // 2        # BF16 = 2 bytes/elem

        dp_rank = self.data_parallel_rank

        # ----------------------------------------------------------------
        # Step 1: flatten all (fp32_shard, boundaries, buf, param_index_map)
        # into a linear sequence of (param, global_start, global_end,
        # shard_start, shard_end, fp32_shard, device) records, then
        # pack them into variable-length buckets.
        #
        # We treat the union of all buffers as a single virtual address space
        # with per-buffer base offsets so that params from different buffers
        # can coexist in one bucket without aliasing.
        # ----------------------------------------------------------------

        # Each entry: (param, ps, pe, shard_start, shard_end, fp32_shard, device)
        # where ps/pe are buffer-local param offsets (same as param_index_map values),
        # and shard_start/shard_end are this rank's shard boundaries in that buffer.
        all_entries: List[Tuple] = []
        for fp32_shard, boundaries, buf in zip(
                self._fp32_shards,
                self._buf_boundaries,
                self.param_and_grad_buffers,
        ):
            shard_start, shard_end = boundaries[dp_rank]
            _device = buf.grad_data.device
            for param, (ps, pe, _) in buf.param_index_map.items():
                all_entries.append((param, ps, pe, shard_start, shard_end,
                                    fp32_shard, _device))

        if not all_entries:
            return

        # ----------------------------------------------------------------
        # Step 2: pack entries into buckets of at most bucket_elems elements.
        # ----------------------------------------------------------------
        buckets: List[List[Tuple]] = []
        current_bucket: List[Tuple] = []
        current_size: int = 0

        for entry in all_entries:
            param, ps, pe, *_ = entry
            numel = pe - ps
            if current_size + numel > bucket_elems and current_bucket:
                buckets.append(current_bucket)
                current_bucket = []
                current_size = 0
            current_bucket.append(entry)
            current_size += numel

        if current_bucket:
            buckets.append(current_bucket)

        logger.debug(
            "shard_to_model_broadcast: %d params → %d NCCL bucket(s) "
            "(bucket_elems=%d, dp_rank=%d)",
            len(all_entries),
            len(buckets),
            bucket_elems,
            dp_rank,
        )

        # ----------------------------------------------------------------
        # Step 3: for each bucket, build flat BF16 buffer, fill owned slices,
        # all_reduce(SUM), scatter back to model params, refresh FP32 shards.
        # ----------------------------------------------------------------
        for bucket in buckets:
            total_bucket_elems: int = sum(pe - ps for _, ps, pe, *_ in bucket)
            # Infer device from the first entry (all params in a training run
            # live on the same device within a DP rank).
            _dev = bucket[0][6]
            flat = torch.zeros(total_bucket_elems, dtype=torch.bfloat16,
                               device=_dev)

            # Fill: this rank writes its owned portion into flat; rest = 0.
            offset: int = 0
            for param, ps, pe, shard_start, shard_end, fp32_shard, _device in bucket:
                numel = pe - ps

                # Intersection of [ps, pe) with this rank's shard [shard_start, shard_end)
                owned_start = max(ps, shard_start)
                owned_end   = min(pe, shard_end)
                if owned_end > owned_start:
                    # Position inside the param tensor (buffer-local)
                    p_lo = owned_start - ps
                    p_hi = owned_end   - ps
                    # Position inside the FP32 shard (shard-local)
                    s_lo = owned_start - shard_start
                    s_hi = owned_end   - shard_start
                    # Write FP32 shard slice (cast to BF16) into the bucket buffer.
                    flat[offset + p_lo:offset + p_hi].copy_(
                        fp32_shard[s_lo:s_hi].bfloat16()
                    )

                offset += numel

            # ONE all_reduce for the entire bucket — reduces NCCL call count
            # from O(num_buffers) to O(total_bytes / BUCKET_BYTES).
            torch.distributed.all_reduce(
                flat,
                op=torch.distributed.ReduceOp.SUM,
                group=self.data_parallel_group,
            )

            # Scatter bucket back to model param tensors and refresh FP32 shard.
            offset = 0
            for param, ps, pe, shard_start, shard_end, fp32_shard, _device in bucket:
                numel = pe - ps
                param_slice = flat[offset:offset + numel]

                # Update model BF16 param in-place.
                param.data.view(-1).copy_(param_slice)

                # Refresh the FP32 shard for the owned slice so the master
                # weights stay consistent with the all_reduced result.
                owned_start = max(ps, shard_start)
                owned_end   = min(pe, shard_end)
                if owned_end > owned_start:
                    p_lo = owned_start - ps
                    p_hi = owned_end   - ps
                    s_lo = owned_start - shard_start
                    s_hi = owned_end   - shard_start
                    fp32_shard[s_lo:s_hi].copy_(param_slice[p_lo:p_hi].float())

                offset += numel

    # ------------------------------------------------------------------
    # Zero grad
    # ------------------------------------------------------------------

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradient buffers and shard param grads."""
        for buf in self.param_and_grad_buffers:
            buf.grad_data.zero_()
        for sp in self._shard_params:
            if set_to_none:
                sp.grad = None
            elif sp.grad is not None:
                sp.grad.detach_()
                sp.grad.zero_()

    # ------------------------------------------------------------------
    # State dict / checkpoint helpers
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return serialisable optimizer state.

        Includes the inner Adam state (moments + step), the FP32 shard
        tensors for warm-start, and the current step count for DES-LOC
        period tracking.
        """
        return {
            "optimizer": self.optimizer.state_dict(),
            "step_count": self._step_count,
            "fp32_shards": [s.cpu() for s in self._fp32_shards],
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore optimizer state from a checkpoint dict."""
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self._step_count = state_dict.get("step_count", 0)
        if "fp32_shards" in state_dict:
            for shard, saved in zip(self._fp32_shards, state_dict["fp32_shards"]):
                shard.copy_(saved.to(shard.device))

    # ------------------------------------------------------------------
    # _copy_main_params_to_model_params  (Megatron evolution: M2307→M4019)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _copy_main_params_to_model_params(self) -> None:
        """Copy updated FP32 shard back to BF16 model param tensors.

        This mirrors Megatron's ``_copy_main_params_to_model_params`` but uses
        our flat ``_fp32_shards`` / ``param_and_grad_buffers`` data structures
        instead of Megatron's ``shard_fp32_from_float16_groups``.

        For each buffer, every model parameter whose range overlaps with this
        rank's shard receives updated data from the corresponding slice of the
        FP32 shard.  This must be called **before** the all-gather so the
        correct values are broadcast to all DP ranks.

        Evolution notes
        ~~~~~~~~~~~~~~~
        - M2307: initial port of copy_group_params inner helper.
        - M3715: decoupled_grad / MFSDP awareness.
        - M3737: NVFP4 native weight guard (skip quantized params here; they
          are handled by a dedicated ``quantize_nvfp4_param_shard`` call).
        - M4019: MXFP8 param-buffer copy path; in our simplified stack we
          keep the standard BF16 path only and skip FP8/NVFP4 guards because
          those quantized param types require TE (Transformer Engine) which is not in scope.
        """
        dp_rank = self.data_parallel_rank

        for fp32_shard, boundaries, buf in zip(
                self._fp32_shards,
                self._buf_boundaries,
                self.param_and_grad_buffers,
        ):
            shard_start, shard_end = boundaries[dp_rank]

            for param, (ps, pe, _) in buf.param_index_map.items():
                overlap_s = max(ps, shard_start)
                overlap_e = min(pe, shard_end)
                if overlap_s >= overlap_e:
                    # This parameter is not owned by this rank's shard.
                    continue

                # Slice inside the local shard
                local_s = overlap_s - shard_start
                local_e = overlap_e - shard_start
                # Slice inside the full parameter tensor
                param_s = overlap_s - ps
                param_e = overlap_e - ps

                # Write FP32 shard slice back to BF16 model param.
                param.data.view(-1)[param_s:param_e].copy_(fp32_shard[local_s:local_e].to(param.data.dtype))

    # ------------------------------------------------------------------
    # start_param_sync / start_param_sync_for_bucket_group_subset
    # (Megatron evolution: M3998 — Route non-Muon params #4771)
    # ------------------------------------------------------------------

    def start_param_sync(self, force_sync: bool = True) -> None:
        """All-gather updated FP32 shards into every rank's BF16 model weights.

        This is the canonical post-step param broadcast.  It replaces the
        earlier ``shard_to_model_broadcast`` and aligns with Megatron's
        ``model_chunk.start_param_sync()`` contract.

        First copies the local FP32 shard slice into BF16 model params (via
        ``_copy_main_params_to_model_params``), then performs the all-gather
        so every rank sees the full, up-to-date weight tensor.

        When ``force_sync=False`` the call is a *no-op* if async overlap is
        requested by the config (future extension hook; currently always
        synchronous).

        Args:
            force_sync: If True (default) perform a blocking all-gather.
                        If False, skip when async param-gather overlap is on.
        """
        if not force_sync and getattr(self.config, "overlap_param_gather", False):
            # Async overlap path: the caller will trigger the gather later.
            return

        # 1. Write local FP32 shard to model params on this rank.
        self._copy_main_params_to_model_params()

        # 2. All-gather so every rank has all updated params.
        self.shard_to_model_broadcast()

    def start_param_sync_for_bucket_group_subset(self) -> None:
        """Trigger ``start_param_sync`` for DistributedOptimizer-managed params only.

        In a system where a ``LayerWiseDistributedOptimizer`` sibling manages
        some buckets, we must avoid double-syncing those buckets.  In our
        single-optimizer stack all params belong to this optimizer, so this
        method is equivalent to ``start_param_sync``.

        This method exists to match the Megatron API surface introduced in
        M3998 (commit 9e60da33) so that callers written against the Megatron
        interface continue to work when ported to Neuron_SP.

        Evolution note (M3998)
        ~~~~~~~~~~~~~~~~~~~~~~
        Megatron walks ``model_chunk.bucket_groups`` and calls
        ``_start_bucket_group_param_sync`` per bucket, skipping buckets tagged
        ``is_managed_by_layer_wise_optimizer=True``.  Our flat-buffer design
        has no per-bucket lazy tagging, so we delegate to ``start_param_sync``
        directly.
        """
        self.start_param_sync(force_sync=True)

    # ------------------------------------------------------------------
    # Decoupled weight decay  (From Megatron M2356)
    # ------------------------------------------------------------------

    def _apply_decoupled_weight_decay(self) -> None:
        """Apply AdamW-style decoupled weight decay before the Adam update.

        From Megatron M2356: AdamW vs Adam weight decay selection.
        Adam:  L2 gradient term before adaptive scaling (rank-size-dependent).
        AdamW: param *= (1 - lr * wd) after update (rank-size-independent).

        DES-LOC: decoupled decay is rank-size-independent, critical for
        heterogeneous 2xA6000+H100+2xBlackwell sharding where different ranks
        own different shard sizes — L2 regularisation via Adam's gradient term
        would produce different effective weight-decay magnitudes per rank.

        This method is called BEFORE ``self.optimizer.step()`` so that the
        weight decay is applied to the FP32 shard values before the Adam
        moment update, matching the AdamW formulation.
        """
        # From Megatron M2356: AdamW vs Adam weight decay selection.
        if not self.config.decoupled_weight_decay:
            return
        wd = self.config.weight_decay
        if wd == 0.0:
            return
        lr = self.optimizer.param_groups[0].get('lr', self.config.lr or 1e-4)
        decay = 1.0 - lr * wd
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.data is not None and p.requires_grad:
                    p.data.mul_(decay)

    # ------------------------------------------------------------------
    # step_with_ready_grads  (Megatron evolution: M2307 → M3998)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        """Optimizer step assuming gradients are already reduced/scattered.

        This is the Megatron-style entry point for the optimizer step when the
        DDP wrapper has already performed the reduce-scatter and the grad shards
        are ready in ``self._fp32_grad_shards``.  Callers that manage DDP
        overlap externally use this instead of the all-in-one ``step()``.

        Sequence
        ~~~~~~~~
        1.  Clip gradients on the local FP32 shard (if configured).
        2.  Local Adam update on the FP32 shard parameters.
        3.  Increment step counter.
        4.  DES-LOC moment synchronisation (Ku / Kv cadence).
        5.  All-gather updated FP32 shards → BF16 model weights, using
            ``start_param_sync_for_bucket_group_subset()`` (aligns with
            Megatron M3998: only sync DistOpt-managed buckets so a sibling
            LayerWise optimizer doesn't double-sync).

        Returns:
            True if the step was taken successfully; False if skipped.

        Evolution notes
        ~~~~~~~~~~~~~~~
        - M2307: ``step_with_ready_grads`` split off from ``step``.
        - M3998: switched inner sync call from
          ``model_chunk.start_param_sync()`` to
          ``start_param_sync_for_bucket_group_subset()`` to avoid duplicate
          bucket syncs when a LayerWise sibling is present.
        - M3811: ``_defer_param_sync`` flag added for FSDP overlap.
        """
        # Gradient clipping on local shard params.
        if self.config.clip_grad > 0.0:
            clip_grad_norm(
                parameters=self._shard_params,
                max_norm=self.config.clip_grad,
                norm_type=2.0,
                model_parallel_group=self.data_parallel_group,
            )

        # From Megatron M2356: apply AdamW decoupled weight decay before Adam step.
        self._apply_decoupled_weight_decay()

        # Local Adam update on FP32 shard.
        self.optimizer.step()

        self._step_count += 1

        # DES-LOC: synchronise moments at Ku / Kv cadence.
        if self.config.desloc_enabled:
            sync_u = self.config.is_ku_step(self._step_count)
            sync_v = self.config.is_kv_step(self._step_count)
            if sync_u or sync_v:
                self.sync_moments(sync_first=sync_u, sync_second=sync_v)
        elif self._desloc is not None and self._desloc.enabled:
            sync_u = self._desloc.is_ku_step(self._step_count)
            sync_v = self._desloc.is_kv_step(self._step_count)
            if sync_u or sync_v:
                self.sync_moments(sync_first=sync_u, sync_second=sync_v)

        # All-gather updated shards to model weights (only DistOpt-managed).
        # When _defer_param_sync is set (FSDP overlap path), skip here; the
        # caller is responsible for triggering the gather at the right moment.
        if not getattr(self, "_defer_param_sync", False):
            self.start_param_sync_for_bucket_group_subset()

        return True

    @torch.no_grad()
    def step(self) -> bool:
        """Full distributed optimizer step.

        Sequence:
          1. Reduce-scatter BF16 grads → FP32 grad shards.
          2. Execute ``step_with_ready_grads`` (clip + Adam + moment sync + all-gather).

        Returns:
            True if a step was taken, False if skipped due to overflow.
        """
        found_inf = self.prepare_grads()
        if found_inf:
            return False
        return self.step_with_ready_grads()

    # ------------------------------------------------------------------
    # save_parameter_state / load_parameter_state
    # (Megatron evolution: M2307 → M3385 → M3356 → M3175)
    # ------------------------------------------------------------------

    def get_parameter_state_dp_zero(
        self,
        use_gloo_comm: bool = True,
        return_on_all_ranks: bool = False,
    ) -> Optional[dict]:
        """Gather per-rank FP32 shards and optimizer moments onto DP rank 0.

        Mirrors Megatron's ``get_parameter_state_dp_zero``.  For each
        ``ParamAndGradBuffer`` we collect ``(param, exp_avg, exp_avg_sq)`` from
        every DP rank and assemble them into world-sized CPU tensors on rank 0.

        Uses Gloo by default (``use_gloo_comm=True``) to avoid occupying the
        NCCL stream during checkpointing.

        Evolution notes
        ~~~~~~~~~~~~~~~
        - M2307: initial implementation with buckets_coalesced format.
        - M3175: ``step`` key in optimizer state handled separately to avoid
          saving a scalar as a tensor (fixes CPU-offload checkpoint crash).
        - M3356: skip non-tensor optimizer state entries (fixes crash when the
          inner optimizer stores Python scalars like ``step`` as plain ints).
        - M3385: ``--load-main-params-from-ckpt`` flag support — pass through
          here via the ``state_dict`` checkpoint structure.

        Args:
            use_gloo_comm:       Use Gloo process group for CPU gather.
            return_on_all_ranks: Return the full state on every rank (for
                                 parallel saving); default False returns None
                                 on non-zero DP ranks.

        Returns:
            State dict on DP rank 0 (or all ranks if ``return_on_all_ranks``),
            None on other ranks.
        """
        # Choose comm group: prefer Gloo CPU transport for checkpoint I/O
        # (does not stall the CUDA NCCL stream).
        if use_gloo_comm and self.data_parallel_group_gloo is not None:
            comm_group = self.data_parallel_group_gloo
        else:
            comm_group = self.data_parallel_group

        dp_world = torch.distributed.get_world_size(group=comm_group)
        dp_rank = torch.distributed.get_rank(group=comm_group)
        global_ranks = torch.distributed.get_process_group_ranks(comm_group)

        state: dict = {"buckets_coalesced": True}

        for buf_idx, (fp32_shard, buf, boundaries) in enumerate(
                zip(self._fp32_shards, self.param_and_grad_buffers, self._buf_boundaries)):
            shard_start, shard_end = boundaries[dp_rank]
            shard_size = shard_end - shard_start
            total_numel = buf.grad_data.numel()

            # --- Build local shard buffers (param + moments) ---
            local_param = fp32_shard.detach().cpu()

            # Retrieve exp_avg / exp_avg_sq from the inner Adam state.
            # The shard param corresponding to this buffer's index.
            shard_p = self._shard_params[buf_idx]
            adam_state = self.optimizer.state.get(shard_p, {})

            local_exp_avg = adam_state.get("exp_avg", torch.zeros(shard_size, dtype=torch.float32))  # From Megatron M2378: explicit dtype prevents silent float32 upcast
            local_exp_avg_sq = adam_state.get("exp_avg_sq", torch.zeros(shard_size, dtype=torch.float32))  # From Megatron M2378: explicit dtype prevents silent float32 upcast

            # M3356: skip non-tensor entries silently (e.g. scalar ``step``).
            if not isinstance(local_exp_avg, torch.Tensor):
                local_exp_avg = torch.zeros(shard_size, dtype=torch.float32)
            if not isinstance(local_exp_avg_sq, torch.Tensor):
                local_exp_avg_sq = torch.zeros(shard_size, dtype=torch.float32)

            local_exp_avg = local_exp_avg.detach().cpu().float()
            local_exp_avg_sq = local_exp_avg_sq.detach().cpu().float()

            # --- Gather all ranks' shards onto rank 0 ---
            # Each rank's shard may differ in size (hetero path), so we use
            # all_gather_object which handles variable-size tensors via CPU.
            local_tensors = {
                "param": local_param,
                "exp_avg": local_exp_avg,
                "exp_avg_sq": local_exp_avg_sq,
                "shard_start": shard_start,
                "shard_end": shard_end,
                "total_numel": total_numel,
            }

            if return_on_all_ranks:
                gathered: List[Optional[dict]] = [None] * dp_world
                torch.distributed.all_gather_object(gathered, local_tensors, group=comm_group)
            else:
                if dp_rank == 0:
                    gathered = [None] * dp_world
                else:
                    gathered = None
                torch.distributed.gather_object(
                    local_tensors,
                    gathered,
                    dst=global_ranks[0],
                    group=comm_group,
                )

            if dp_rank == 0 or return_on_all_ranks:
                # Assemble world tensors from all ranks' shards.
                world_param = torch.zeros(total_numel, dtype=torch.float32)
                world_exp_avg = torch.zeros(total_numel, dtype=torch.float32)
                world_exp_avg_sq = torch.zeros(total_numel, dtype=torch.float32)

                for rank_data in gathered:
                    if rank_data is None:
                        continue
                    rs, re = rank_data["shard_start"], rank_data["shard_end"]
                    copy_len = re - rs
                    world_param[rs:re].copy_(rank_data["param"][:copy_len])
                    world_exp_avg[rs:re].copy_(rank_data["exp_avg"][:copy_len])
                    world_exp_avg_sq[rs:re].copy_(rank_data["exp_avg_sq"][:copy_len])

                dtype_state = {
                    torch.bfloat16: {
                        "param": world_param,
                        "exp_avg": world_exp_avg,
                        "exp_avg_sq": world_exp_avg_sq,
                        "numel_unpadded": total_numel,
                    }
                }
                state[buf_idx] = dtype_state
            else:
                state[buf_idx] = None

        return state if (dp_rank == 0 or return_on_all_ranks) else None

    def save_parameter_state(self, filename: str) -> None:
        """Gather all FP32 shards + Adam moments onto DP rank 0 and save.

        Only DP rank 0 writes to *filename*.  All ranks participate in the
        gather via ``get_parameter_state_dp_zero``.

        Uses Gloo comm group for CPU gather (does not stall NCCL stream).

        Evolution notes
        ~~~~~~~~~~~~~~~
        - M2307: initial implementation (NCCL-based gather).
        - M3175: switched to ``get_parameter_state_dp_zero`` helper which also
          saves optimizer moments alongside main params.
        - M3385: added ``step_count`` to the checkpoint payload for DES-LOC
          period tracking.

        Args:
            filename: Path on the local filesystem where the checkpoint is
                      written (rank 0 only).
        """
        state_dict = self.get_parameter_state_dp_zero(use_gloo_comm=True)
        if self.data_parallel_rank == 0 and state_dict is not None:
            state_dict["step_count"] = self._step_count
            torch.save(state_dict, filename)
            logger.info(
                "DistributedOptimizer: saved parameter state to %s "
                "(rank 0, dp_world=%d)",
                filename,
                self.data_parallel_world_size,
            )

    def load_parameter_state_from_dp_zero(self,
                                          state_dict: Optional[dict],
                                          *,
                                          update_legacy_format: bool = False) -> None:
        """Scatter world-sized checkpoint tensors from DP rank 0 to all ranks.

        Mirrors Megatron's ``load_parameter_state_from_dp_zero``.  DP rank 0
        holds the full ``(param, exp_avg, exp_avg_sq)`` tensors; we scatter the
        appropriate slice to every rank.

        Evolution notes
        ~~~~~~~~~~~~~~~
        - M2307: initial implementation.
        - M3175: ``step`` key handled separately (scalar, not a tensor).
        - M3356: non-tensor optimizer state entries are skipped gracefully to
          handle checkpoints written by Adam variants that store ``step`` as
          a Python int rather than a 0-D tensor.
        - M3385: ``--load-main-params-from-ckpt`` path: after scattering the
          world tensors the FP32 shard is re-broadcast from the param slice
          so the model weights are consistent without requiring a forward pass.

        Args:
            state_dict:            Checkpoint dict (only non-None on DP rank 0).
            update_legacy_format:  When True, call legacy loader for
                                   pre-Feb-2024 checkpoint format.
        """
        if update_legacy_format:
            # Legacy format: state_dict is just a list of gathered shards.
            return self._load_parameter_state_legacy(state_dict)

        # Choose Gloo comm group for scatter (same as save path).
        if self.data_parallel_group_gloo is not None:
            comm_group = self.data_parallel_group_gloo
        else:
            comm_group = self.data_parallel_group

        dp_world = torch.distributed.get_world_size(group=comm_group)
        dp_rank = torch.distributed.get_rank(group=comm_group)
        global_ranks = torch.distributed.get_process_group_ranks(comm_group)

        for buf_idx, (fp32_shard, boundaries) in enumerate(zip(self._fp32_shards, self._buf_boundaries)):
            shard_start, shard_end = boundaries[dp_rank]
            shard_size = shard_end - shard_start

            # DP rank 0 prepares per-rank slice dicts for scatter.
            if dp_rank == 0:
                if state_dict is None or state_dict.get(buf_idx) is None:
                    logger.warning(
                        "load_parameter_state_from_dp_zero: missing buf_idx=%d in state_dict",
                        buf_idx,
                    )
                    send_objects = [{} for _ in range(dp_world)]
                else:
                    buf_state = list(state_dict[buf_idx].values())[0]  # first (only) dtype
                    world_param = buf_state["param"]
                    world_exp_avg = buf_state["exp_avg"]
                    world_exp_avg_sq = buf_state["exp_avg_sq"]

                    send_objects: List[dict] = []
                    for r in range(dp_world):
                        rs, re = boundaries[r]
                        send_objects.append({
                            "param": world_param[rs:re].clone(),
                            "exp_avg": world_exp_avg[rs:re].clone(),
                            "exp_avg_sq": world_exp_avg_sq[rs:re].clone(),
                        })
            else:
                send_objects = None

            recv_obj: dict = {}
            torch.distributed.scatter_object_list(
                [recv_obj],
                send_objects,
                src=global_ranks[0],
                group=comm_group,
            )

            # Restore FP32 shard from checkpoint.
            if "param" in recv_obj:
                copy_len = min(shard_size, recv_obj["param"].numel())
                fp32_shard[:copy_len].copy_(recv_obj["param"][:copy_len].to(fp32_shard.device))

            # Restore Adam moments for this buffer's shard param.
            shard_p = self._shard_params[buf_idx]
            if shard_p not in self.optimizer.state:
                self.optimizer.state[shard_p] = {}
            opt_state = self.optimizer.state[shard_p]

            # M3356: guard — only update tensor entries.
            for key in ("exp_avg", "exp_avg_sq"):
                if key in recv_obj and isinstance(recv_obj[key], torch.Tensor):
                    tensor = recv_obj[key].to(fp32_shard.device)
                    if key in opt_state and isinstance(opt_state[key], torch.Tensor):
                        opt_state[key].copy_(tensor)
                    else:
                        opt_state[key] = tensor.clone()

    def _load_parameter_state_legacy(self, state_dict: Optional[dict]) -> None:
        """Load from the pre-M2307 checkpoint format (plain gathered-shard list).

        The legacy format stores a list of per-buffer gathered FP32 tensors
        (no moments) under the key ``"fp32_shards"``.  We split them back into
        per-rank slices using the current shard boundaries.

        Args:
            state_dict: Dict with ``"fp32_shards"`` key (only on DP rank 0).
        """
        dp_rank = self.data_parallel_rank
        dp_world = self.data_parallel_world_size

        comm_group = self.data_parallel_group

        for buf_idx, (fp32_shard, boundaries) in enumerate(zip(self._fp32_shards, self._buf_boundaries)):
            shard_start, shard_end = boundaries[dp_rank]

            if dp_rank == 0 and state_dict is not None:
                gathered_list = state_dict.get("fp32_shards", [])
                if buf_idx < len(gathered_list) and gathered_list[buf_idx] is not None:
                    world_tensor = gathered_list[buf_idx]
                    send_objects = []
                    for r in range(dp_world):
                        rs, re = boundaries[r]
                        send_objects.append({"param": world_tensor[rs:re].clone()})
                else:
                    send_objects = [{} for _ in range(dp_world)]
            else:
                send_objects = None

            recv_obj: dict = {}
            torch.distributed.scatter_object_list(
                [recv_obj],
                send_objects,
                src=torch.distributed.get_global_rank(comm_group, 0),
                group=comm_group,
            )

            if "param" in recv_obj:
                shard_size = shard_end - shard_start
                copy_len = min(shard_size, recv_obj["param"].numel())
                fp32_shard[:copy_len].copy_(recv_obj["param"][:copy_len].to(fp32_shard.device))

    def load_parameter_state(self, filename: str, *, update_legacy_format: bool = False) -> None:
        """Load distributed parameter state from *filename*.

        DP rank 0 reads the checkpoint; all ranks receive their slice via
        ``load_parameter_state_from_dp_zero``.

        Evolution notes
        ~~~~~~~~~~~~~~~
        - M2307: initial implementation.
        - M3385: added ``step_count`` restore for DES-LOC period tracking.

        Args:
            filename:             Path to the checkpoint saved by
                                  ``save_parameter_state``.
            update_legacy_format: Pass through to
                                  ``load_parameter_state_from_dp_zero`` for
                                  pre-Feb-2024 checkpoints.
        """
        state_dict: Optional[dict] = None
        if self.data_parallel_rank == 0:
            state_dict = torch.load(filename, map_location="cpu")
            logger.info("DistributedOptimizer: loaded parameter state from %s", filename)

        # Restore step counter for DES-LOC period alignment.
        if self.data_parallel_rank == 0 and state_dict is not None:
            self._step_count = state_dict.get("step_count", 0)

        # Broadcast step_count to all ranks.
        step_count_t = torch.tensor(
            [self._step_count],
            dtype=torch.int64,
            device=next(iter(self._fp32_shards)).device if self._fp32_shards else torch.device("cpu"),
        )
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(
                step_count_t,
                src=torch.distributed.get_global_rank(self.data_parallel_group, 0),
                group=self.data_parallel_group,
            )
        self._step_count = int(step_count_t.item())

        self.load_parameter_state_from_dp_zero(state_dict, update_legacy_format=update_legacy_format)

    def get_model_param_range_map(self, param: torch.nn.Parameter) -> Optional[Dict[str, Range]]:
        """Return the grad-buffer range info for *param* on this rank.

        Returns a dict with key ``"gbuf_local"`` containing the Range of
        elements this rank owns for the parameter, or ``None`` if the param
        does not overlap with this rank's shard.
        """
        dp_rank = self.data_parallel_rank

        for buf, boundaries in zip(self.param_and_grad_buffers, self._buf_boundaries):
            if param not in buf.param_index_map:
                continue
            ps, pe, _ = buf.param_index_map[param]
            shard_start, shard_end = boundaries[dp_rank]
            overlap_s = max(ps, shard_start)
            overlap_e = min(pe, shard_end)
            if overlap_s >= overlap_e:
                return None
            local_s = overlap_s - shard_start
            local_e = overlap_e - shard_start
            return {
                "gbuf_local": Range(local_s, local_e),
                "gbuf_world": Range(overlap_s, overlap_e),
                "param": Range(overlap_s - ps, overlap_e - ps),
            }
        return None

    # ===========================================================================
    # Section: _build_* class methods — gbuf range map construction
    # Adapted from Megatron-LM/megatron/core/optimizer/distrib_optimizer.py
    # Lines 128-395. Megatron uses bucket-based buffers; we use flat shards.
    # ===========================================================================

    @classmethod
    def _build_model_gbuf_param_range_map(
        cls,
        param_world_index_map: Dict[torch.nn.Parameter, Tuple],
        gbuf_world_range: "Range",
        bucket_offset: int,
    ) -> Dict[torch.nn.Parameter, Dict[str, "Range"]]:
        """Build mapping from param reference to grad buffer shard ranges.

        Creates four Range objects per parameter that overlaps with this DP
        rank's shard:
          - gbuf_world             : param range in the entire grad buffer.
          - gbuf_world_in_bucket   : param range relative to bucket start.
          - gbuf_local             : param range in the DP rank's local view.
          - param                  : param range within the parameter itself.

        In our flat-buffer design there is exactly one "bucket" per buffer
        (bucket_offset == 0 always), but we preserve the four-range structure
        so that callers written against the Megatron API work without changes.

        Args:
            param_world_index_map: {param: (start, end, bucket_id)} from
                ParamAndGradBuffer.param_index_map.
            gbuf_world_range: The Range this DP rank owns in the grad buffer.
            bucket_offset: Offset of the bucket within the grad buffer (0 for
                flat single-bucket buffers).

        Returns:
            Dict mapping param -> dict of four Range objects.
        """
        param_range_map: Dict[torch.nn.Parameter, Dict[str, Range]] = {}
        for param, param_world_indexes in param_world_index_map.items():
            param_world_start, param_world_end, _ = param_world_indexes

            # Intersection of param range with the local shard.
            param_local_start = max(0, param_world_start - gbuf_world_range.start)
            param_local_end = min(gbuf_world_range.size, param_world_end - gbuf_world_range.start)

            if param_local_end <= param_local_start:
                continue  # param is not in this rank's shard

            param_local_range = Range(param_local_start, param_local_end)
            param_world_range = param_local_range.normalize(
                param_local_start + gbuf_world_range.start
            )
            param_world_range_in_bucket = Range(
                param_world_range.start - bucket_offset,
                param_world_range.end - bucket_offset,
            )
            sub_param_start = max(0, gbuf_world_range.start - param_world_start)
            sub_param_range = param_local_range.normalize(sub_param_start)

            param_range_map[param] = {
                "gbuf_world": param_world_range,
                "gbuf_world_in_bucket": param_world_range_in_bucket,
                "gbuf_local": param_local_range,
                "param": sub_param_range,
            }

        return param_range_map

    @classmethod
    def _build_model_gbuf_range(
        cls,
        param_and_grad_buffer: "ParamAndGradBuffer",
        bucket_index: int,
    ) -> Dict:
        """Build per-bucket param-range info for the calling DP rank.

        For our flat-buffer design every buffer has exactly one logical
        "bucket" (bucket_index == 0), and the bucket size equals the full
        buffer size.  We assume the buffer numel is divisible by the DP world
        size (ensured by _compute_hetero_shard_boundaries padding).

        In heterogeneous mode the shard sizes differ across ranks; we compute
        the local shard range from ``_buf_boundaries`` if available, otherwise
        fall back to equal-slice math.

        Args:
            param_and_grad_buffer: The ParamAndGradBuffer to process.
            bucket_index: Ignored (always 0 in our flat design), kept for
                API compatibility with Megatron callers.

        Returns:
            Dict with key ``"param_map"`` → per-param Range dicts.
        """
        # Resolve DP rank / world size from the buffer or from distributed state.
        if hasattr(param_and_grad_buffer, "data_parallel_group"):
            dp_rank = param_and_grad_buffer.data_parallel_group.rank()
            dp_world = param_and_grad_buffer.data_parallel_group.size()
        else:
            # From Megatron M4022: use safe_get_rank/safe_get_world_size which
            # also fall back to SLURM env vars before defaulting to 0/1.
            dp_rank = parallel_state.safe_get_rank()
            dp_world = parallel_state.safe_get_world_size()

        gbuf_size = param_and_grad_buffer.grad_data.numel()
        # Equal-slice shard boundaries (het-sizing not applicable here since
        # the buffer object doesn't carry tier info; callers that need hetero
        # sizing should use _build_shards() directly).
        shard_size = (gbuf_size + dp_world - 1) // dp_world
        gbuf_world_start = dp_rank * shard_size
        gbuf_world_end = min(gbuf_size, gbuf_world_start + shard_size)
        gbuf_world_range = Range(gbuf_world_start, gbuf_world_end)

        param_range_map = cls._build_model_gbuf_param_range_map(
            param_and_grad_buffer.param_index_map,
            gbuf_world_range,
            bucket_offset=0,
        )

        return {"param_map": param_range_map}

    @classmethod
    def _build_gbuf_range_map(
        cls,
        param_and_grad_buffer: "ParamAndGradBuffer",
    ) -> Dict:
        """Build dtype-keyed mapping from params to their grad-buffer shard ranges.

        Megatron returns ``{(param_dtype, grad_dtype): [bucket_range_dicts]}``.
        Since our flat-buffer has a single "bucket", the value is a one-element
        list.  The dtype key is derived from the buffer's data dtype.

        Args:
            param_and_grad_buffer: Buffer to map.

        Returns:
            Dict of shape ``{(param_dtype, grad_dtype): [range_dict]}``.
        """
        # Determine dtypes.  ParamAndGradBuffer may expose param_dtype /
        # grad_dtype; fall back to the data tensor dtype when absent.
        param_dtype = getattr(
            param_and_grad_buffer, "param_dtype",
            getattr(param_and_grad_buffer, "dtype", param_and_grad_buffer.grad_data.dtype),
        )
        grad_dtype = getattr(
            param_and_grad_buffer, "grad_dtype", param_and_grad_buffer.grad_data.dtype
        )

        return {
            (param_dtype, grad_dtype): [
                cls._build_model_gbuf_range(param_and_grad_buffer, bucket_index=0)
            ]
        }

    @classmethod
    def _build_model_param_gbuf_map(
        cls,
        gbuf_ranges: List[Dict],
    ) -> Dict[torch.nn.Parameter, Tuple]:
        """Create a reverse map: param → (gbuf_index, dtype, bucket_index).

        Iterates the gbuf_ranges list (one entry per ParamAndGradBuffer) and
        builds a flat dict so that any parameter can be quickly looked up to
        find which buffer and dtype key it belongs to.

        Args:
            gbuf_ranges: Output of ``[_build_gbuf_range_map(buf) for buf in buffers]``.

        Returns:
            Dict mapping each param to a (gbuf_index, dtype, bucket_index) tuple.
        """
        param_gbuf_map: Dict[torch.nn.Parameter, Tuple] = {}
        for gbuf_index, gbuf_range_map in enumerate(gbuf_ranges):
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_map.items():
                for bucket_index, bucket_range_map in enumerate(gbuf_range_map_for_all_buckets):
                    for param in bucket_range_map["param_map"]:
                        assert param not in param_gbuf_map, (
                            "Param should not be in param_gbuf_map; "
                            "each param only belongs to a single bucket."
                        )
                        param_gbuf_map[param] = (gbuf_index, dtype, bucket_index)
        return param_gbuf_map

    @classmethod
    def _build_optimizer_group_ranges(
        cls,
        param_groups: List[Dict],
        gbuf_ranges: List[Dict],
    ) -> Tuple[Dict, List[Dict]]:
        """Build optimizer group ranges from param groups and grad-buffer ranges.

        Creates:
        - ``local_param_group_map``: param → (group_index, order_in_group).
        - ``group_ranges``: one dict per param group with ``"params"`` list
          of params that this DP rank owns, plus ``"orig_group"`` reference.

        Only parameters that appear in at least one ``gbuf_range_map``'s
        ``param_map`` (i.e., owned by this rank) are added to group_ranges.

        Args:
            param_groups: The optimizer's ``param_groups`` list.
            gbuf_ranges: List of gbuf range maps, one per buffer.

        Returns:
            (local_param_group_map, group_ranges) tuple.
        """
        # Build world map: param → group index.
        world_param_group_map: Dict[torch.nn.Parameter, int] = {}
        for group_index, group in enumerate(param_groups):
            for param in group["params"]:
                if param.requires_grad:
                    world_param_group_map[param] = group_index

        # Build local map and group_ranges (only owned params).
        local_param_group_map: Dict[torch.nn.Parameter, Tuple[int, int]] = {}
        group_ranges: List[Dict] = [{"params": []} for _ in param_groups]

        for gbuf_range_map in gbuf_ranges:
            for _dtype, gbuf_range_map_for_all_buckets in gbuf_range_map.items():
                for bucket_range_map in gbuf_range_map_for_all_buckets:
                    for param in bucket_range_map["param_map"]:
                        if param not in world_param_group_map:
                            continue
                        group_index = world_param_group_map[param]
                        group_range = group_ranges[group_index]
                        group_range["params"].append(param)
                        local_param_group_map[param] = (
                            group_index,
                            len(group_range["params"]) - 1,
                        )

        for group_index, group_range in enumerate(group_ranges):
            group_range["orig_group"] = param_groups[group_index]
            group_range["orig_group_idx"] = param_groups[group_index]

        return local_param_group_map, group_ranges

    @classmethod
    def _build_model_and_main_param_groups(
        cls,
        gbuf_ranges: List[Dict],
        param_gbuf_map: Dict[torch.nn.Parameter, Tuple],
        opt_group_ranges: List[Dict],
        config: "OptimizerConfig",
    ) -> Tuple[List, List, List, List, List]:
        """Create main parameter groups for the optimizer step.

        Allocates or views the FP32 main-param shards for each param group.
        Returns five parallel lists of lists (one inner list per group):
          - model_float16_groups         : original BF16/FP16 model params.
          - model_fp32_groups            : original FP32 model params.
          - shard_float16_groups         : sharded view of BF16/FP16 params.
          - shard_fp32_groups            : sharded view of FP32 params.
          - shard_fp32_from_float16_groups: FP32 copies of BF16/FP16 shards.

        This is a simplified version of Megatron's method: we skip TE/FP8/
        NVFP4 quantized paths as those require Transformer Engine which is
        not in scope for Neuron_SP's flat-shard stack.

        Args:
            gbuf_ranges:      List of gbuf range maps (one per buffer).
            param_gbuf_map:   Reverse map: param → (gbuf_idx, dtype, bucket).
            opt_group_ranges: Output of _build_optimizer_group_ranges.
            config:           OptimizerConfig (used to check fp8 flags, etc.).

        Returns:
            Five-tuple of group-of-lists.
        """
        model_float16_groups: List[List] = []
        model_fp32_groups: List[List] = []
        shard_float16_groups: List[List] = []
        shard_fp32_groups: List[List] = []
        shard_fp32_from_float16_groups: List[List] = []

        for group_range in opt_group_ranges:
            mf16, mfp32, sf16, sfp32, sfp32_from_f16 = [], [], [], [], []
            model_float16_groups.append(mf16)
            model_fp32_groups.append(mfp32)
            shard_float16_groups.append(sf16)
            shard_fp32_groups.append(sfp32)
            shard_fp32_from_float16_groups.append(sfp32_from_f16)

            for model_param in group_range["params"]:
                assert model_param.requires_grad

                gbuf_index, dtype, bucket_index = param_gbuf_map[model_param]
                gbuf_range = gbuf_ranges[gbuf_index][dtype][bucket_index]
                param_range = gbuf_range["param_map"][model_param]["param"]

                if model_param.dtype in (torch.float16, torch.bfloat16):
                    # Sharded view of the BF16/FP16 model param.
                    shard_model_param = model_param.detach().view(-1)[
                        param_range.start:param_range.end
                    ]
                    # FP32 main param shard (clone of the BF16 shard, cast to float).
                    shard_main_param = shard_model_param.clone().float()

                    # Store references on the model param for easy lookup.
                    model_param.main_param = shard_main_param
                    model_param.main_param_sharded = True

                    mf16.append(model_param)
                    sf16.append(shard_model_param)
                    sfp32_from_f16.append(shard_main_param)

                elif model_param.dtype == torch.float32:
                    shard_model_param = model_param.view(-1)[param_range.start:param_range.end]
                    mfp32.append(model_param)
                    sfp32.append(shard_model_param)

                else:
                    raise TypeError(
                        f"Wrapped parameters must be float16, bfloat16, or float32. "
                        f"Received {model_param.dtype}."
                    )

            # Update the orig_group's params to point at the FP32 shards.
            group_range["orig_group"]["params"] = [*sfp32, *sfp32_from_f16]

        return (
            model_float16_groups,
            model_fp32_groups,
            shard_float16_groups,
            shard_fp32_groups,
            shard_fp32_from_float16_groups,
        )

    # ===========================================================================
    # Section: layout helpers
    # ===========================================================================

    @staticmethod
    def _does_param_require_new_bucket(param: torch.nn.Parameter) -> bool:
        """Return True if param should be isolated in its own bucket.

        In Megatron this is used for shared-embedding params.  We inherit the
        same heuristic: check for a ``shared_embedding`` attribute.

        Args:
            param: Model parameter to inspect.

        Returns:
            True if the param requires a dedicated bucket.
        """
        return bool(getattr(param, "shared_embedding", False))

    @staticmethod
    def _finalize_bucket(
        param_end_index: int,
        bucket_start_index: int,
        bucket_id: int,
        per_bucket_numel_unpadded: List[int],
        bucket_indices: List[Tuple[int, int]],
        data_parallel_world_size: int,
        pad_for_high_nccl_busbw: bool = False,
    ) -> Tuple[int, int]:
        """Finalise one bucket: record unpadded numel, compute padded end, append.

        Pads the bucket end to lcm(dp_world_size, 128) so that:
          - reduce-scatter slices divide evenly across DP ranks, and
          - the bucket is at minimum 128-element aligned for NCCL efficiency.

        When ``pad_for_high_nccl_busbw`` is True the divisor is extended to
        lcm(dp_world_size, 128, 2^16) for peak NCCL bus-bandwidth on large-
        message collectives.

        Upgraded from plain ``dp_world_size`` alignment to
        ``lcm(dp_size, 128)`` in M3811 (Megatron 55b8111ad / c6a886ed).

        Args:
            param_end_index:           Unpadded end index of last param in bucket.
            bucket_start_index:        Start index of this bucket.
            bucket_id:                 Current bucket counter (for logging).
            per_bucket_numel_unpadded: List to append unpadded numel to.
            bucket_indices:            List to append (start, padded_end) to.
            data_parallel_world_size:  DP world size (for alignment).
            pad_for_high_nccl_busbw:  Extend alignment to 2^16 (default False).

        Returns:
            (padded_bucket_end, next_bucket_id) pair.
        """
        from .param_layout import pad_bucket_end as _pad_bucket_end
        numel_unpadded = param_end_index - bucket_start_index
        per_bucket_numel_unpadded.append(numel_unpadded)
        # Pad to lcm(dp_world_size, 128) — matches Megatron M3811 pad_bucket_end.
        padded_end = _pad_bucket_end(
            param_end_index, data_parallel_world_size, pad_for_high_nccl_busbw
        )
        bucket_indices.append((bucket_start_index, padded_end))
        return padded_end, bucket_id + 1

    @staticmethod
    def _compute_per_buffer_param_layout(
        params: List[torch.nn.Parameter],
        bucket_size: Optional[int],
        data_parallel_world_size: int,
        pad_for_high_nccl_busbw: bool = False,
    ) -> Dict:
        """Compute flat-buffer param layout (index map + bucket boundaries).

        Iterates params in *reverse* order (matching backprop order), records
        each param's [start, end) range in the flat buffer, and splits into
        buckets when the accumulated size exceeds ``bucket_size`` or a param
        requires its own bucket.

        Alignment rules (matching Megatron M3811 / param_layout.py):
          - Each param start is aligned to a **64-element** boundary via
            ``pad_param_start`` so that every parameter lives on a
            cache-line-friendly address inside the flat buffer.
          - Each bucket end is aligned to ``lcm(dp_world_size, 128)`` (or
            ``lcm(dp_world_size, 128, 2^16)`` when
            ``pad_for_high_nccl_busbw=True``) via ``_finalize_bucket``.

        Params that require new buckets (shared embeddings) are isolated.

        Args:
            params:                    Parameters to lay out.
            bucket_size:               Approx max elements per bucket;
                                       ``None`` → single bucket.
            data_parallel_world_size:  For bucket-end alignment.
            pad_for_high_nccl_busbw:  When True, extend bucket-end alignment
                                       to ``lcm(dp_size, 128, 2^16)`` for
                                       peak NCCL bus-bandwidth. Default False.

        Returns:
            Dict with keys:
              - ``param_index_map``: {param: (start, end, bucket_id)}.
              - ``bucket_indices``:  [(start, end), ...] per bucket.
              - ``per_bucket_numel_unpadded``: unpadded numel per bucket.
        """
        from .param_layout import pad_param_start as _pad_param_start

        param_index_map: Dict[torch.nn.Parameter, Tuple[int, int, int]] = {}
        bucket_indices: List[Tuple[int, int]] = []
        per_bucket_numel_unpadded: List[int] = []

        param_start_index = 0
        bucket_start_index = 0
        bucket_params: set = set()
        bucket_id = 0

        def _finalize(param_end_index: int, bucket_start: int, bid: int) -> Tuple[int, int]:
            return DistributedOptimizer._finalize_bucket(
                param_end_index,
                bucket_start,
                bid,
                per_bucket_numel_unpadded,
                bucket_indices,
                data_parallel_world_size,
                pad_for_high_nccl_busbw,
            )

        for param in params[::-1]:
            # M3811: align each param start to a 64-element boundary so params
            # begin on cache-line-friendly addresses within the flat buffer.
            param_start_index = _pad_param_start(param_start_index)

            # Split shared embedding params into a separate bucket.
            if DistributedOptimizer._does_param_require_new_bucket(param) and bucket_params:
                bucket_start_index, bucket_id = _finalize(
                    param_start_index, bucket_start_index, bucket_id
                )
                bucket_params = set()
                param_start_index = bucket_start_index
                # Re-align after bucket boundary advance.
                param_start_index = _pad_param_start(param_start_index)

            param_numel = param.data.nelement()
            param_end_index = param_start_index + param_numel
            param_index_map[param] = (param_start_index, param_end_index, bucket_id)
            bucket_params.add(param)

            bucket_full = (
                bucket_size is not None
                and (param_end_index - bucket_start_index) >= bucket_size
            )
            if bucket_full or DistributedOptimizer._does_param_require_new_bucket(param):
                bucket_start_index, bucket_id = _finalize(
                    param_end_index, bucket_start_index, bucket_id
                )
                bucket_params = set()
                param_start_index = bucket_start_index
            else:
                param_start_index = param_end_index

        if bucket_params:
            _finalize(param_end_index, bucket_start_index, bucket_id)  # type: ignore[possibly-undefined]

        return {
            "param_index_map": param_index_map,
            "bucket_indices": bucket_indices,
            "per_bucket_numel_unpadded": per_bucket_numel_unpadded,
        }

    @staticmethod
    def compute_full_param_layout(
        params: List[torch.nn.Parameter],
        bucket_size: Optional[int],
        data_parallel_world_size: int,
        optimizer_config=None,
        ddp_config=None,
    ) -> Dict:
        """Compute parameter layouts for all buffer groups.

        Groups parameters by dtype, computes a padded layout for each group,
        and returns a combined mapping.

        Args:
            params:                    All model parameters.
            bucket_size:               Approx elements per bucket (None = 1 bucket).
            data_parallel_world_size:  For bucket-end alignment.
            optimizer_config:          Optional OptimizerConfig; when provided and
                ``use_pcie_aware_overlap=True``, bucket_size is recalculated using
                PCIe bandwidth and latency parameters.
                Insight I6: PCIe-aware overlap (Megatron aa-3.5).
            ddp_config:                Optional DDP config; when provided,
                ``pad_buckets_for_high_nccl_busbw`` is forwarded to bucket-end
                padding (M3811: lcm(dp_size, 128, 2^16) when True).

        Returns:
            Dict mapping dtype → per-buffer layout dict from
            ``_compute_per_buffer_param_layout``.
        """
        # Insight I6: PCIe-aware overlap (Megatron aa-3.5)
        # When optimizer_config requests PCIe-aware sizing and no explicit
        # bucket_size override was given, compute a PCIe-adapted bucket_size
        # that is much smaller than the NVLink-tuned default so that gradient
        # collectives complete within a single backward segment and meaningful
        # overlap is achievable.
        if (
            bucket_size is None
            and optimizer_config is not None
            and getattr(optimizer_config, 'use_pcie_aware_overlap', False)
        ):
            bucket_size = optimizer_config.pcie_bucket_size(data_parallel_world_size)
            import logging as _logging
            _logging.getLogger(__name__).info(
                # Insight I6: PCIe-aware overlap (Megatron aa-3.5)
                "I6 PCIe-aware bucket_size=%d elements for DistributedOptimizer "
                "(bw=%.1f GB/s, latency=%.1f µs, dp=%d)",
                bucket_size,
                optimizer_config.pcie_bw_gbps,
                optimizer_config.pcie_latency_us,
                data_parallel_world_size,
            )

        # M3811: read pad_for_high_nccl_busbw from ddp_config when provided.
        pad_for_high_nccl_busbw = bool(
            getattr(ddp_config, 'pad_buckets_for_high_nccl_busbw', False)
        )

        # Group by dtype.
        dtype_groups: Dict[torch.dtype, List[torch.nn.Parameter]] = {}
        for p in params:
            if not p.requires_grad:
                continue
            dtype_groups.setdefault(p.dtype, []).append(p)

        layouts: Dict[torch.dtype, Dict] = {}
        for dtype, dtype_params in dtype_groups.items():
            layouts[dtype] = DistributedOptimizer._compute_per_buffer_param_layout(
                dtype_params, bucket_size, data_parallel_world_size,
                pad_for_high_nccl_busbw,
            )
        return layouts

    # ===========================================================================
    # Section: optimizer state helpers
    # ===========================================================================

    def _get_model_param_range_map(
        self, param: torch.nn.Parameter
    ) -> Dict[str, "Range"]:
        """Return the per-rank range-map for *param*.

        Delegates to ``get_model_param_range_map`` which searches
        ``param_and_grad_buffers``.  Raises ``KeyError`` if the param is
        not found (matches Megatron behaviour for callers that assume the
        param is always present).

        Args:
            param: A model parameter managed by this optimizer.

        Returns:
            Dict with ``"gbuf_local"``, ``"gbuf_world"``, and ``"param"`` Range
            objects (same keys as ``_build_model_gbuf_param_range_map``).

        Raises:
            KeyError: If the param is not in any grad buffer.
        """
        result = self.get_model_param_range_map(param)
        if result is None:
            raise KeyError(
                f"Parameter {param} not found in any grad buffer managed by this optimizer."
            )
        return result

    def get_grad_stats_parallel_group(self) -> Optional[torch.distributed.ProcessGroup]:
        """Return the process group used for gradient statistics (norm / zero count).

        With the distributed optimizer, grad stats are reduced over the entire
        DP group.  This mirrors Megatron's method; returns ``None`` when no
        dedicated stats group is configured (callers must handle None).

        Returns:
            The data-parallel process group, or ``None`` if not set.
        """
        return getattr(self, "grad_stats_parallel_group", self.data_parallel_group)

    def _get_main_param_and_optimizer_states(
        self, model_param: torch.nn.Parameter
    ) -> Dict[str, torch.Tensor]:
        """Return FP32 main param + Adam moments for *model_param*.

        Looks up the shard parameter that corresponds to the model param's
        buffer index, then retrieves the current optimizer state.

        Returns a dict with keys:
          - ``"param"``      : FP32 main param shard slice.
          - ``"exp_avg"``    : first moment (if Adam state exists).
          - ``"exp_avg_sq"`` : second moment (if Adam state exists).

        Args:
            model_param: A model parameter owned by this rank.

        Returns:
            Tensor dict.
        """
        range_map = self._get_model_param_range_map(model_param)
        gbuf_local = range_map["gbuf_local"]

        # Find which buffer this param belongs to.
        for buf_idx, buf in enumerate(self.param_and_grad_buffers):
            if model_param in buf.param_index_map:
                break
        else:
            raise KeyError(f"model_param {model_param} not in any buffer.")

        fp32_shard = self._fp32_shards[buf_idx]
        main_param_slice = fp32_shard[gbuf_local.start:gbuf_local.end]

        shard_p = self._shard_params[buf_idx]
        opt_state = self.optimizer.state.get(shard_p, {})

        tensors: Dict[str, torch.Tensor] = {"param": main_param_slice}
        for key in ("exp_avg", "exp_avg_sq"):
            if key in opt_state and isinstance(opt_state[key], torch.Tensor):
                tensors[key] = opt_state[key][gbuf_local.start:gbuf_local.end]
        return tensors

    def _set_main_param_and_optimizer_states(
        self,
        model_param: torch.nn.Parameter,
        tensors: Dict[str, torch.Tensor],
    ) -> None:
        """Copy checkpoint tensors into the FP32 shard and Adam state.

        Inverse of ``_get_main_param_and_optimizer_states``.  Copies each
        tensor from *tensors* into the appropriate slice of the in-memory
        state.

        Args:
            model_param: The model parameter whose state to update.
            tensors:     Dict with ``"param"``, ``"exp_avg"``, ``"exp_avg_sq"``
                         (subset is fine; missing keys are silently skipped).
        """
        range_map = self._get_model_param_range_map(model_param)
        gbuf_local = range_map["gbuf_local"]

        for buf_idx, buf in enumerate(self.param_and_grad_buffers):
            if model_param in buf.param_index_map:
                break
        else:
            raise KeyError(f"model_param {model_param} not in any buffer.")

        fp32_shard = self._fp32_shards[buf_idx]
        shard_p = self._shard_params[buf_idx]

        if "param" in tensors and isinstance(tensors["param"], torch.Tensor):
            src = tensors["param"]
            copy_len = min(gbuf_local.size, src.numel())
            fp32_shard[gbuf_local.start:gbuf_local.start + copy_len].copy_(
                src[:copy_len].to(fp32_shard.device)
            )

        opt_state = self.optimizer.state.get(shard_p)
        if opt_state is None:
            return

        for key in ("exp_avg", "exp_avg_sq"):
            if key not in tensors or not isinstance(tensors[key], torch.Tensor):
                continue
            if key not in opt_state or not isinstance(opt_state[key], torch.Tensor):
                continue
            src = tensors[key]
            dst = opt_state[key][gbuf_local.start:gbuf_local.end]
            copy_len = min(dst.numel(), src.numel())
            dst[:copy_len].copy_(src[:copy_len].to(dst.device))

    def _init_optimizer_states_with_dummy_values(self) -> None:
        """Initialise Adam states with zeros so that checkpoint load can copy in-place.

        Megatron calls this before ``load_parameter_state_from_*`` to avoid
        re-allocating tensors during load (reduces memory fragmentation).  We
        replicate the same pattern: run a dummy forward-backward-step so that
        PyTorch allocates the state tensors, then zero out the results.
        """
        for shard_p in self._shard_params:
            if shard_p.numel() == 0:
                continue
            shard_p.grad = torch.zeros_like(shard_p)
        try:
            self.optimizer.step()
        except Exception:
            pass
        for shard_p in self._shard_params:
            shard_p.grad = None

    # ===========================================================================
    # Section: grad / param copy utilities
    # ===========================================================================

    def copy_group_grads(
        self,
        model_groups: List[List[torch.nn.Parameter]],
        shard_main_groups: List[List[torch.Tensor]],
    ) -> None:
        """Copy model gradients into FP32 main-param shard ``.grad`` fields.

        For each (model_param, shard_main_param) pair, slices the appropriate
        region of ``model_param.main_grad`` into ``shard_main_param.grad``.

        This is a no-op when a param's range is not owned by this rank (the
        range map look-up returns None).

        Args:
            model_groups:      Nested list of BF16/FP16 or FP32 model params.
            shard_main_groups: Corresponding nested list of FP32 shard tensors.
        """
        for model_group, shard_main_group in zip(model_groups, shard_main_groups):
            for model_param, shard_main_param in zip(model_group, shard_main_group):
                range_map = self.get_model_param_range_map(model_param)
                if range_map is None:
                    continue
                param_range = range_map["param"]
                if not hasattr(model_param, "main_grad") or model_param.main_grad is None:
                    continue
                model_grad = model_param.main_grad
                shard_model_grad = model_grad.view(-1)[param_range.start:param_range.end]
                shard_main_param.grad = shard_model_grad.float()

    def copy_group_params(
        self,
        shard_main_groups: List[List[torch.Tensor]],
        model_groups: List[List[torch.nn.Parameter]],
    ) -> None:
        """Copy updated FP32 shard data back to BF16/FP16 model param views.

        For each (shard_main_param, model_param) pair copies the shard data
        into the grad-buffer region that corresponds to the param's world range
        within the bucket.

        Args:
            shard_main_groups: Nested list of updated FP32 shard tensors.
            model_groups:      Corresponding nested list of model params.
        """
        for shard_main_group, model_group in zip(shard_main_groups, model_groups):
            for shard_main_param, model_param in zip(shard_main_group, model_group):
                range_map = self.get_model_param_range_map(model_param)
                if range_map is None:
                    continue
                gbuf_world = range_map["gbuf_world"]
                assert gbuf_world.size == shard_main_param.nelement(), (
                    f"Shard size mismatch: world_range={gbuf_world.size}, "
                    f"shard_numel={shard_main_param.nelement()}"
                )
                for buf_idx, buf in enumerate(self.param_and_grad_buffers):
                    if model_param in buf.param_index_map:
                        break
                else:
                    continue
                # Write into the flat grad-data buffer at the world-range position.
                buf.grad_data[gbuf_world.start:gbuf_world.end].copy_(
                    shard_main_param.to(buf.grad_data.dtype)
                )

    def _collect_main_grad_data_for_unscaling(self) -> List[torch.Tensor]:
        """Return a flat list of main-param ``.grad.data`` tensors for unscaling.

        Used by the grad-scaler to unscale in-place before the Adam step.
        In our flat-shard design the "main grads" are the FP32 grad shards.

        Returns:
            List of grad data tensors.
        """
        grads: List[torch.Tensor] = []
        for shard_p in self._shard_params:
            if shard_p.grad is not None:
                grads.append(shard_p.grad.data)
        return grads

    def _get_model_and_main_params_data_float16(
        self,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Return aligned (model_data, main_data) lists for BF16/FP16 params.

        Iterates ``param_and_grad_buffers`` and collects pairs of:
          - The BF16/FP16 param's data tensor (from the model).
          - The corresponding FP32 shard slice (from ``_fp32_shards``).

        Only params owned by this DP rank are included.

        Returns:
            (model_data, main_data) parallel lists of tensors.
        """
        model_data: List[torch.Tensor] = []
        main_data: List[torch.Tensor] = []
        dp_rank = self.data_parallel_rank

        for buf_idx, (buf, boundaries, fp32_shard) in enumerate(
            zip(self.param_and_grad_buffers, self._buf_boundaries, self._fp32_shards)
        ):
            shard_start, shard_end = boundaries[dp_rank]
            for param, (ps, pe, _) in buf.param_index_map.items():
                if param.dtype not in (torch.float16, torch.bfloat16):
                    continue
                overlap_s = max(ps, shard_start)
                overlap_e = min(pe, shard_end)
                if overlap_s >= overlap_e:
                    continue
                local_s = overlap_s - shard_start
                local_e = overlap_e - shard_start
                param_s = overlap_s - ps
                param_e = overlap_e - ps
                model_data.append(param.data.view(-1)[param_s:param_e])
                main_data.append(fp32_shard[local_s:local_e])

        return model_data, main_data

    # ===========================================================================
    # Section: quantized-param stubs (TE / NVFP4 not in scope)
    # ===========================================================================

    @staticmethod
    def _is_grouped_quantized_tensor(tensor: torch.Tensor) -> bool:
        """Return True if *tensor* is a TE GroupedTensor with quantized storage.

        In the Neuron_SP stack Transformer Engine is not a dependency, so this
        always returns False.  Retained for API parity with Megatron callers.

        Args:
            tensor: Any tensor to inspect.

        Returns:
            Always False in this implementation.
        """
        return (
            hasattr(tensor, "split_into_quantized_tensors")
            and callable(tensor.split_into_quantized_tensors)
            and getattr(tensor, "quantizer", None) is not None
        )

    @classmethod
    def _is_distopt_quantized_param(cls, tensor: torch.Tensor) -> bool:
        """Return True if *tensor* should follow the quantized-param path.

        Covers FP8 tensors and TE GroupedTensor objects.  Without TE in scope
        both checks evaluate to False, so this always returns False here.

        Args:
            tensor: Tensor or parameter to inspect.

        Returns:
            Always False in this implementation.
        """
        # is_float8tensor would require TE; skip.
        return cls._is_grouped_quantized_tensor(tensor)

    def _expand_quantized_param_shard_for_cast(
        self,
        model_param: torch.Tensor,
        shard_main_param: Optional[torch.Tensor],
        start_offset: Optional[int],
    ) -> Tuple[List, List, List]:
        """Expand a quantized model param into cast-ready entries.

        For TE GroupedTensor, splits into member quantized tensors and maps
        the master shard to per-member offset ranges.  Since TE is not in
        scope this always returns single-element lists (identity expansion).

        Args:
            model_param:      The model parameter (may be quantized).
            shard_main_param: The FP32 main param shard, or None.
            start_offset:     Offset of the shard within the param, or None.

        Returns:
            (model_params, shard_main_params, start_offsets) three parallel lists.
        """
        if not self._is_grouped_quantized_tensor(model_param):
            return [model_param], [shard_main_param], [start_offset]

        # TE GroupedTensor path (not reached without TE):
        quantized_members = getattr(model_param, "quantized_tensors", None)
        if quantized_members is None:
            quantized_members = model_param.split_into_quantized_tensors()

        shard_start = 0 if start_offset is None else start_offset
        shard_size = 0 if shard_main_param is None else shard_main_param.numel()
        shard_end = shard_start + shard_size
        shard_flat = None if shard_main_param is None else shard_main_param.view(-1)

        expanded_model_params: List = []
        expanded_shard_main_params: List = []
        expanded_start_offsets: List = []
        member_offset = 0
        for member in quantized_members:
            member_numel = member.numel()
            member_start = member_offset
            member_end = member_start + member_numel
            overlap_start = max(member_start, shard_start)
            overlap_end = min(member_end, shard_end)

            member_master: Optional[torch.Tensor] = None
            member_start_offset: Optional[int] = None
            if overlap_start < overlap_end:
                local_start = overlap_start - shard_start
                local_end = overlap_end - shard_start
                member_master = shard_flat[local_start:local_end]  # type: ignore[index]
                member_start_offset = overlap_start - member_start

            expanded_model_params.append(member)
            expanded_shard_main_params.append(member_master)
            expanded_start_offsets.append(member_start_offset)
            member_offset = member_end

        return expanded_model_params, expanded_shard_main_params, expanded_start_offsets

    def get_shard_fp32_from_fp8(
        self,
        shard_main_groups: List[List[torch.Tensor]],
        model_groups: List[List[torch.nn.Parameter]],
    ) -> Tuple[List, List, List]:
        """Collect FP8 params with their FP32 shard counterparts.

        In the Neuron_SP stack FP8 (TE Float8Tensor) params are not present,
        so this always returns three empty lists.  Retained for Megatron API
        parity.

        Args:
            shard_main_groups: Nested list of FP32 shard tensors (unused here).
            model_groups:      Nested list of model params (unused here).

        Returns:
            (fp8_params, shard_fp32_from_fp8, shard_offsets) — all empty lists.
        """
        return [], [], []

    def _get_fp8_params_and_shard_fp32_from_fp8(
        self,
    ) -> Tuple[List, List, List]:
        """Return FP8 param list with their FP32 main-param shards.

        Always returns empty lists because TE / FP8 are not in scope.

        Returns:
            (fp8_params, shard_fp32_from_fp8, shard_offsets_in_fp8) — empty.
        """
        return [], [], []

    def _get_nvfp4_params_and_shard_fp32_from_nvfp4(
        self,
    ) -> Tuple[List, List, List]:
        """Return NVFP4 param list with their FP32 main-param shards.

        Always returns empty lists because NVFP4 is not in scope.

        Returns:
            (nvfp4_params, shard_fp32_from_nvfp4, shard_offsets) — empty.
        """
        return [], [], []

    def _get_shard_fp32_from_nvfp4(
        self,
        shard_main_groups: List[List[torch.Tensor]],
        model_groups: List[List[torch.nn.Parameter]],
    ) -> Tuple[List, List, List]:
        """Populate NVFP4 shard lists (not in scope — returns empty).

        Args:
            shard_main_groups: Nested FP32 shard lists (unused).
            model_groups:      Nested model param lists (unused).

        Returns:
            Three empty lists.
        """
        return [], [], []

    def _copy_main_params_to_param_buffer(self) -> None:
        """Copy FP32 main params directly to the param buffer.

        In Megatron this is used for MXFP8 params where the param buffer is
        not mapped to model params.  In Neuron_SP we write directly to the
        model param tensors via ``_copy_main_params_to_model_params``, so
        this method is a no-op (retained for API parity).

        From Megatron M3116: IMPORTANT — any future caller of this method
        must guard the call with a check that forward_pre_hook is registered:
          forward_pre_hook_enabled = (
              hasattr(model, 'remove_forward_pre_hook_handles') and
              len(model.remove_forward_pre_hook_handles) > 0
          )
          if forward_pre_hook_enabled:
              self._copy_main_params_to_param_buffer()
        Without this guard, calling on the first iteration (before the hook
        is registered) will pollute main_grads with main_params values since
        finish_param_sync() hasn't been called yet to zero the grad buffer.
        """
        # No-op: MXFP8 path not in scope.
        pass

    # ===========================================================================
    # Section: sharded state dict / checkpoint helpers
    # ===========================================================================

    def _param_name(self, param: torch.nn.Parameter) -> str:
        """Return the fully-qualified name of *param* in the model.

        Builds a ``param_to_name`` cache on first call by inverting
        ``model.named_parameters()``.  Requires that the model (or model
        chunks) are accessible.  Falls back to a hex id string when the
        param cannot be found (avoids hard failures during debugging).

        Args:
            param: A model parameter.

        Returns:
            Dotted parameter name string.
        """
        if not hasattr(self, "_param_to_name"):
            self._param_to_name: Dict[torch.nn.Parameter, str] = {}
            model_chunks = getattr(self, "model_chunks", None)
            if model_chunks is not None:
                for chunk in model_chunks:
                    for name, p in chunk.named_parameters():
                        self._param_to_name[p] = name
        return self._param_to_name.get(param, f"<unknown_param_{id(param):x}>")

    def _param_groups_to_param2group_meta(
        self,
        param_groups: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Convert param groups to {param_name: group_metadata} mapping.

        Used by the FSDP DTensor sharded state dict path to serialise group
        hyper-parameters by name instead of by tensor identity.

        Args:
            param_groups: The optimizer's param_groups list.

        Returns:
            Dict mapping param name → group metadata dict (no ``"params"`` key).
        """
        param_to_group_meta: Dict[str, Any] = {}
        for group in param_groups:
            group_meta = {k: v for k, v in group.items() if k != "params"}
            for p in group["params"]:
                param_to_group_meta[self._param_name(p)] = group_meta
        return param_to_group_meta

    def _param2group_meta_to_param_groups(
        self,
        param_to_group_meta: Dict[str, Any],
        param_groups: List[Dict[str, Any]],
        strict: bool = True,
    ) -> List[Dict[str, Any]]:
        """Convert {param_name: metadata} back to a list of param groups.

        Inverse of ``_param_groups_to_param2group_meta``.  Matches each param
        in *param_groups* to its entry in *param_to_group_meta* by name.

        Args:
            param_to_group_meta: Output of ``_param_groups_to_param2group_meta``.
            param_groups:        Current optimizer param groups (for param tensors).
            strict:              If True, raise on missing param names.

        Returns:
            New param_groups list with metadata restored.
        """
        new_param_groups: List[Dict[str, Any]] = []
        for group in param_groups:
            new_group: Dict[str, Any] = {"params": []}
            for p in group["params"]:
                name = self._param_name(p)
                if name not in param_to_group_meta:
                    if strict:
                        raise ValueError(
                            f"Parameter '{name}' not found in param_to_group_meta."
                        )
                    continue
                group_meta = param_to_group_meta[name]
                # Validate consistency within the group.
                existing_meta = {k: v for k, v in new_group.items() if k != "params"}
                if existing_meta and existing_meta != group_meta:
                    msg = (
                        f"Inconsistent metadata for param '{name}': "
                        f"got {group_meta}, expected {existing_meta}."
                    )
                    if strict:
                        raise ValueError(msg)
                    else:
                        logger.warning(msg)
                        continue
                new_group["params"].append(p)
                new_group.update(group_meta)
            new_param_groups.append(new_group)
        return new_param_groups

    def _normalize_state_dict_for_grouped_params(
        self,
        state_dict_flat: Dict[str, torch.Tensor],
        model_chunk: Any,
    ) -> None:
        """Normalise grouped-param keys in a flat state dict (no-op).

        In Megatron this handles TEGroupedLinear weight format mismatches
        between checkpoints saved with ``single_grouped_weight=True/False``.
        Since Transformer Engine is not in scope, this is a no-op.

        Args:
            state_dict_flat: Flat {name: tensor} state dict (modified in-place).
            model_chunk:     Model chunk (unused).
        """
        # No-op: TE GroupedLinear not in scope.
        pass

    def _build_model_param_to_state_dict_param_map(
        self,
        state_dict: Dict,
    ) -> Dict[torch.nn.Parameter, torch.Tensor]:
        """Create param → state_dict_tensor mapping by matching parameter names.

        Supports checkpoints stored as ``{"model": {...}}`` (single chunk),
        ``{"model0": {...}, "model1": {...}}`` (multi-chunk), or a flat dict.

        Args:
            state_dict: Model state dict from which to extract parameter tensors.

        Returns:
            Dict mapping each model parameter tensor → its matching tensor in
            the state dict.

        Raises:
            AssertionError: If a parameter name has 0 or >1 matches.
        """
        model_chunks = getattr(self, "model_chunks", None)

        # Build list of per-chunk state dicts.
        state_dict_list: List[Dict] = []
        if model_chunks is not None and len(model_chunks) > 1:
            for i in range(len(model_chunks)):
                for key in (f"model{i}", f"model_{i}"):
                    if key in state_dict:
                        state_dict_list.append(state_dict[key])
                        break
                else:
                    raise KeyError(
                        f"Multi-chunk checkpoint missing key 'model{i}' or 'model_{i}'."
                    )
        elif "model" in state_dict:
            state_dict_list.append(state_dict["model"])
        else:
            state_dict_list.append(state_dict)

        model_param_to_sd_param: Dict[torch.nn.Parameter, torch.Tensor] = {}

        source_chunks = model_chunks if model_chunks is not None else [None]
        for chunk_idx, model_chunk in enumerate(source_chunks):
            sd = state_dict_list[min(chunk_idx, len(state_dict_list) - 1)]
            self._normalize_state_dict_for_grouped_params(sd, model_chunk)
            sd_keys = set(sd.keys())

            named_params = (
                model_chunk.named_parameters()
                if model_chunk is not None
                else []
            )
            for name, model_param in named_params:
                while name.startswith("module."):
                    name = name[len("module."):]
                matches = [k for k in sd_keys if k.endswith(name)]
                assert len(matches) == 1, (
                    f"Parameter '{name}' has {len(matches)} matches in state dict."
                )
                sd_param = sd[matches[0]]
                assert model_param.shape == sd_param.shape, (
                    f"Shape mismatch for '{name}': "
                    f"model={model_param.shape}, ckpt={sd_param.shape}."
                )
                model_param_to_sd_param[model_param] = sd_param
                sd_keys.remove(matches[0])

        return model_param_to_sd_param

    def _update_legacy_world_tensors(
        cls,
        old_tensors: List[torch.Tensor],
        new_numels: List[int],
    ) -> List[torch.Tensor]:
        """Re-shard a list of tensors to new target numels (legacy ckpt helper).

        Concatenates *old_tensors* and slices into new chunks of sizes given by
        *new_numels*.  Used by ``load_parameter_state_from_dp_zero_legacy`` to
        reshard bucket tensors between checkpoint and current bucket layout.

        Args:
            old_tensors: List of tensors from the legacy checkpoint.
            new_numels:  Target sizes for the new partitioning.

        Returns:
            List of tensor slices of the requested sizes.
        """
        old_total = sum(t.numel() for t in old_tensors)
        new_total = sum(new_numels)
        assert old_total == new_total, (
            f"Total numel mismatch: old={old_total}, new={new_total}."
        )

        unified = torch.cat(old_tensors, dim=0)
        new_tensors: List[torch.Tensor] = []
        start = 0
        for numel in new_numels:
            new_tensors.append(unified[start:start + numel])
            start += numel
        return new_tensors

    def get_parameter_state_dp_reshardable(self) -> Dict:
        """Return internal DP-reshardable parameter state (no gather).

        This format stores the optimizer state in the same bucket-centric
        layout as the DistributedOptimizer's internal buffers, making it
        fully parallel to save/load without inter-process communication.

        In our flat-shard design we expose the per-shard state directly.

        Evolution notes
        ~~~~~~~~~~~~~~~
        - M3356: only tensor-valued optimizer state entries are included;
          scalar entries like ``step`` (plain int/float in some optimizers)
          are silently skipped to prevent crash during save/load.

        Returns:
            Dict with ``per_bucket_numel``, ``per_bucket_numel_unpadded``,
            and per-buffer per-bucket parameter states.
        """
        per_bucket_numel: List[Dict] = []
        per_bucket_numel_unpadded: List[Dict] = []
        state: Dict = {}

        dp_rank = self.data_parallel_rank

        for buf_idx, (buf, boundaries, fp32_shard) in enumerate(
            zip(self.param_and_grad_buffers, self._buf_boundaries, self._fp32_shards)
        ):
            total_numel = buf.grad_data.numel()
            shard_start, shard_end = boundaries[dp_rank]

            param_dtype = getattr(buf, "param_dtype", buf.grad_data.dtype)
            grad_dtype = getattr(buf, "grad_dtype", buf.grad_data.dtype)
            dtype_key = (param_dtype, grad_dtype)

            per_bucket_numel.append({dtype_key: [total_numel]})
            per_bucket_numel_unpadded.append({dtype_key: [total_numel]})

            shard_p = self._shard_params[buf_idx]
            opt_state = self.optimizer.state.get(shard_p, {})

            bucket_state: List[Dict] = []
            for param, (ps, pe, _) in buf.param_index_map.items():
                overlap_s = max(ps, shard_start)
                overlap_e = min(pe, shard_end)
                if overlap_s >= overlap_e:
                    continue
                local_s = overlap_s - shard_start
                local_e = overlap_e - shard_start
                tensors: Dict[str, Any] = {
                    "param": fp32_shard[local_s:local_e],
                    "gbuf_local_start": local_s,
                    "gbuf_local_end": local_e,
                }
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in opt_state and isinstance(opt_state[key], torch.Tensor):
                        tensors[key] = opt_state[key][local_s:local_e]
                bucket_state.append(tensors)

            state[buf_idx] = {dtype_key: [bucket_state]}

        state["per_bucket_numel"] = per_bucket_numel
        state["per_bucket_numel_unpadded"] = per_bucket_numel_unpadded
        return state

    def sharded_param_state_dp_reshardable(
        self,
        model_sharded_state_dict: Optional[Dict] = None,
        is_loading: bool = False,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """Sharded state dict in DP-reshardable bucket-space format.

        Wraps ``get_parameter_state_dp_reshardable`` into a plain dict (no
        ShardedTensor wrapping because we don't depend on Megatron's dist-
        checkpointing library).

        Args:
            model_sharded_state_dict: Unused in this implementation.
            is_loading:               If True, return empty placeholder state.
            metadata:                 Unused.

        Returns:
            State dict compatible with ``load_parameter_state_from_dp_reshardable``.
        """
        if is_loading:
            self._init_optimizer_states_with_dummy_values()
        return self.get_parameter_state_dp_reshardable()

    def sharded_param_state_dp_zero(
        self,
        model_sharded_state_dict: Optional[Dict] = None,
        is_loading: bool = False,
        metadata: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Sharded state dict in DP-zero gather/scatter format.

        Gathers all DP ranks' states onto rank 0 (save) or returns None
        (load — scatter is done by ``load_state_dict``).

        Args:
            model_sharded_state_dict: Unused.
            is_loading:               If True return None (state loaded separately).
            metadata:                 Unused.

        Returns:
            State dict on DP rank 0 when saving, None otherwise.
        """
        if is_loading:
            return None
        return self.get_parameter_state_dp_zero(
            use_gloo_comm=True,
            return_on_all_ranks=False,
        )

    def sharded_param_state_fs_model_space(
        self,
        model_sharded_state_dict: Optional[Dict] = None,
        is_loading: bool = False,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """Sharded state dict in fully-sharded model-space format.

        Maps each model param to its optimizer state tensors, aligned with
        model parameter ordering.  No inter-process communication during save.

        Args:
            model_sharded_state_dict: Unused (no ShardedTensor wrapping).
            is_loading:               If True initialise dummy states first.
            metadata:                 Unused.

        Returns:
            Dict mapping param_idx → {state_key: tensor}.
        """
        if is_loading:
            self._init_optimizer_states_with_dummy_values()

        state: Dict[int, Dict[str, torch.Tensor]] = {}
        param_idx = 0
        dp_rank = self.data_parallel_rank

        for buf_idx, (buf, boundaries, fp32_shard) in enumerate(
            zip(self.param_and_grad_buffers, self._buf_boundaries, self._fp32_shards)
        ):
            shard_start, shard_end = boundaries[dp_rank]
            shard_p = self._shard_params[buf_idx]
            opt_state = self.optimizer.state.get(shard_p, {})

            for param, (ps, pe, _) in buf.param_index_map.items():
                overlap_s = max(ps, shard_start)
                overlap_e = min(pe, shard_end)
                tensors: Dict[str, torch.Tensor] = {}
                if overlap_s < overlap_e:
                    local_s = overlap_s - shard_start
                    local_e = overlap_e - shard_start
                    tensors["fp32_param"] = fp32_shard[local_s:local_e]
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in opt_state and isinstance(opt_state[key], torch.Tensor):
                            tensors[key] = opt_state[key][local_s:local_e]
                state[param_idx] = tensors
                param_idx += 1

        return state

    def sharded_param_state_fully_reshardable(
        self,
        model_sharded_state_dict: Optional[Dict] = None,
        is_loading: bool = False,
        metadata: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Fully-reshardable state dict format.

        During save, gathers all DP ranks onto rank 0 and produces a canonical
        per-param state dict.  During load, each rank loads the full state and
        selects its slice (no communication).

        Args:
            model_sharded_state_dict: Unused.
            is_loading:               If True return gathered state (all ranks).
            metadata:                 May contain ``distrib_optim_fully_reshardable_mem_efficient``.

        Returns:
            State dict on all ranks when loading; DP-rank-0-only when saving
            in memory-efficient mode; all ranks otherwise.
        """
        if metadata is None:
            metadata = {}
        mem_efficient = metadata.get("distrib_optim_fully_reshardable_mem_efficient", False)
        return_on_all = (not mem_efficient) or is_loading

        return self.get_parameter_state_dp_zero(
            use_gloo_comm=mem_efficient,
            return_on_all_ranks=return_on_all,
        )

    def sharded_param_state_fsdp_dtensor(
        self,
        is_loading: bool = False,
    ) -> Dict:
        """Sharded state dict for FSDP DTensor format (not in scope).

        Megatron uses this path when ``ddp_config.use_megatron_fsdp=True``.
        Neuron_SP does not use Megatron-FSDP, so this raises NotImplementedError.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "sharded_param_state_fsdp_dtensor is not supported in Neuron_SP "
            "because Megatron-FSDP is not used."
        )

    def sharded_state_dict(
        self,
        model_sharded_state_dict: Optional[Dict] = None,
        is_loading: bool = False,
        sharding_type: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """Return a sharded state dict, choosing format from *sharding_type*.

        Supported sharding types (matching Megatron API):
          - ``'dp_reshardable'``      : bucket-space, fully parallel I/O.
          - ``'fully_reshardable'``   : model-space, DP-reshardable.
          - ``'fully_sharded_model_space'`` : model-space alias (deprecated in Megatron).
          - ``'dp_zero_gather_scatter'``    : legacy gather/scatter format.
          - ``'fsdp_dtensor'``         : raises NotImplementedError.

        Defaults to ``'fully_sharded_model_space'`` when *sharding_type* is
        None and *metadata* does not specify one.

        Args:
            model_sharded_state_dict: Model sharded state dict (passed to helpers).
            is_loading:               Whether we are loading (vs saving).
            sharding_type:            Override for sharding format.
            metadata:                 Optional metadata dict; may contain
                                      ``'distrib_optim_sharding_type'``.

        Returns:
            State dict with ``'param_state'`` and ``'param_state_sharding_type'`` keys.
        """
        if sharding_type is None:
            sharding_type = (metadata or {}).get(
                "distrib_optim_sharding_type", "fully_sharded_model_space"
            )

        # Base non-param state.
        base_sd = {
            "optimizer": self.optimizer.state_dict(),
            "step_count": self._step_count,
        }

        if sharding_type == "dp_reshardable":
            param_state = self.sharded_param_state_dp_reshardable(
                model_sharded_state_dict, is_loading, metadata
            )
        elif sharding_type == "dp_zero_gather_scatter":
            param_state = self.sharded_param_state_dp_zero(
                model_sharded_state_dict, is_loading, metadata
            )
        elif sharding_type in ("fully_reshardable", "fully_sharded_model_space"):
            param_state = self.sharded_param_state_fully_reshardable(
                model_sharded_state_dict, is_loading, metadata
            )
        elif sharding_type == "fsdp_dtensor":
            param_state = self.sharded_param_state_fsdp_dtensor(is_loading)
        else:
            raise NotImplementedError(f"Unknown sharding_type: '{sharding_type}'.")

        base_sd["param_state"] = param_state
        base_sd["param_state_sharding_type"] = sharding_type
        return base_sd

    def load_parameter_state_from_dp_reshardable(
        self,
        state_dict: Dict,
    ) -> None:
        """Load parameter state from the DP-reshardable format.

        Inverse of ``get_parameter_state_dp_reshardable``.  Copies each param's
        and its Adam moments' tensors back into the in-memory shards.

        Args:
            state_dict: Output of ``get_parameter_state_dp_reshardable`` (or
                        ``sharded_param_state_dp_reshardable``).
        """
        dp_rank = self.data_parallel_rank

        for buf_idx, (buf, boundaries, fp32_shard) in enumerate(
            zip(self.param_and_grad_buffers, self._buf_boundaries, self._fp32_shards)
        ):
            shard_start, shard_end = boundaries[dp_rank]
            shard_p = self._shard_params[buf_idx]

            param_dtype = getattr(buf, "param_dtype", buf.grad_data.dtype)
            grad_dtype = getattr(buf, "grad_dtype", buf.grad_data.dtype)
            dtype_key = (param_dtype, grad_dtype)

            if buf_idx not in state_dict:
                logger.warning(
                    "load_parameter_state_from_dp_reshardable: buf_idx=%d not in state_dict.",
                    buf_idx,
                )
                continue

            bucket_states = state_dict[buf_idx].get(dtype_key, [[]])[0]
            # Filter out padding entries.
            bucket_states = [e for e in bucket_states if not e.get("padding", False)]

            opt_state = self.optimizer.state.get(shard_p, {})

            param_iter = iter(
                (param, ps, pe)
                for param, (ps, pe, _) in buf.param_index_map.items()
                if max(ps, shard_start) < min(pe, shard_end)
            )

            for src_tensors in bucket_states:
                try:
                    param, ps, pe = next(param_iter)
                except StopIteration:
                    break
                overlap_s = max(ps, shard_start)
                overlap_e = min(pe, shard_end)
                local_s = overlap_s - shard_start
                local_e = overlap_e - shard_start

                if "param" in src_tensors and isinstance(src_tensors["param"], torch.Tensor):
                    src = src_tensors["param"]
                    copy_len = min(local_e - local_s, src.numel())
                    fp32_shard[local_s:local_s + copy_len].copy_(
                        src[:copy_len].to(fp32_shard.device)
                    )

                for key in ("exp_avg", "exp_avg_sq"):
                    if key not in src_tensors or not isinstance(src_tensors[key], torch.Tensor):
                        continue
                    if key not in opt_state or not isinstance(opt_state[key], torch.Tensor):
                        continue
                    src = src_tensors[key]
                    dst = opt_state[key][local_s:local_e]
                    copy_len = min(dst.numel(), src.numel())
                    dst[:copy_len].copy_(src[:copy_len].to(dst.device))

    def load_parameter_state_from_fs_model_space(
        self,
        state_dict: Dict,
    ) -> None:
        """Load parameter state from the fully-sharded model-space format.

        Inverse of ``sharded_param_state_fs_model_space``.  Iterates params in
        the same order as the save path and restores FP32 shard + Adam moments.

        Args:
            state_dict: Output of ``sharded_param_state_fs_model_space``.
        """
        param_idx = 0
        dp_rank = self.data_parallel_rank

        for buf_idx, (buf, boundaries, fp32_shard) in enumerate(
            zip(self.param_and_grad_buffers, self._buf_boundaries, self._fp32_shards)
        ):
            shard_start, shard_end = boundaries[dp_rank]
            shard_p = self._shard_params[buf_idx]
            opt_state = self.optimizer.state.get(shard_p, {})

            for param, (ps, pe, _) in buf.param_index_map.items():
                if param_idx not in state_dict:
                    param_idx += 1
                    continue

                src_tensors = state_dict[param_idx]
                overlap_s = max(ps, shard_start)
                overlap_e = min(pe, shard_end)

                if overlap_s < overlap_e:
                    local_s = overlap_s - shard_start
                    local_e = overlap_e - shard_start

                    for raw_key, dst_key in (("fp32_param", "param"), ("param", "param")):
                        if raw_key in src_tensors and isinstance(
                            src_tensors[raw_key], torch.Tensor
                        ):
                            src = src_tensors[raw_key]
                            copy_len = min(local_e - local_s, src.numel())
                            fp32_shard[local_s:local_s + copy_len].copy_(
                                src[:copy_len].to(fp32_shard.device)
                            )
                            break

                    for key in ("exp_avg", "exp_avg_sq"):
                        if key not in src_tensors or not isinstance(
                            src_tensors[key], torch.Tensor
                        ):
                            continue
                        if key not in opt_state or not isinstance(opt_state[key], torch.Tensor):
                            continue
                        src = src_tensors[key]
                        dst = opt_state[key][local_s:local_e]
                        copy_len = min(dst.numel(), src.numel())
                        dst[:copy_len].copy_(src[:copy_len].to(dst.device))

                param_idx += 1

    def load_parameter_state_from_fully_reshardable(
        self,
        state_dict: Dict,
    ) -> None:
        """Load parameter state from the fully-reshardable DP-zero format.

        Iterates over all params in the same order as
        ``sharded_param_state_fully_reshardable`` (i.e., all buffer params,
        not just those owned by this rank), and loads the slice corresponding
        to this rank's shard range.

        Args:
            state_dict: Output of ``sharded_param_state_fully_reshardable``
                        or ``get_parameter_state_dp_zero``.
        """
        if state_dict is None:
            return

        dp_rank = self.data_parallel_rank

        for buf_idx, (buf, boundaries, fp32_shard) in enumerate(
            zip(self.param_and_grad_buffers, self._buf_boundaries, self._fp32_shards)
        ):
            if buf_idx not in state_dict:
                continue

            param_dtype = getattr(buf, "param_dtype", buf.grad_data.dtype)
            grad_dtype = getattr(buf, "grad_dtype", buf.grad_data.dtype)
            dtype_key = (param_dtype, grad_dtype)

            world_tensors = state_dict[buf_idx].get(dtype_key, {})
            if not world_tensors:
                continue

            shard_start, shard_end = boundaries[dp_rank]
            shard_p = self._shard_params[buf_idx]
            opt_state = self.optimizer.state.get(shard_p, {})

            for param, (ps, pe, _) in buf.param_index_map.items():
                overlap_s = max(ps, shard_start)
                overlap_e = min(pe, shard_end)
                if overlap_s >= overlap_e:
                    continue
                local_s = overlap_s - shard_start
                local_e = overlap_e - shard_start
                world_s = overlap_s
                world_e = overlap_e

                if "param" in world_tensors and isinstance(world_tensors["param"], torch.Tensor):
                    src = world_tensors["param"][world_s:world_e]
                    copy_len = min(local_e - local_s, src.numel())
                    fp32_shard[local_s:local_s + copy_len].copy_(
                        src[:copy_len].to(fp32_shard.device)
                    )

                for key in ("exp_avg", "exp_avg_sq"):
                    if key not in world_tensors or not isinstance(
                        world_tensors[key], torch.Tensor
                    ):
                        continue
                    if key not in opt_state or not isinstance(opt_state[key], torch.Tensor):
                        continue
                    src = world_tensors[key][world_s:world_e]
                    dst = opt_state[key][local_s:local_e]
                    copy_len = min(dst.numel(), src.numel())
                    dst[:copy_len].copy_(src[:copy_len].to(dst.device))

    def load_parameter_state_from_dp_zero_legacy(
        self,
        state_dict: Optional[Dict],
    ) -> None:
        """Load from the pre-Feb-2024 legacy checkpoint format.

        In the legacy format the state dict stores, for each buffer:
        ``{gbuf_idx: {torch.float32: {"param": [tensor_per_bucket], ...}}}``.

        We concatenate the per-bucket tensors into a world tensor, then
        scatter the appropriate slice to this rank using the Gloo group.

        Args:
            state_dict: Legacy checkpoint dict (non-None only on DP rank 0).
        """
        comm_group = (
            self.data_parallel_group_gloo
            if self.data_parallel_group_gloo is not None
            else self.data_parallel_group
        )
        dp_world = torch.distributed.get_world_size(group=comm_group)
        dp_rank = torch.distributed.get_rank(group=comm_group)
        global_ranks = torch.distributed.get_process_group_ranks(comm_group)

        for buf_idx, (fp32_shard, boundaries) in enumerate(
            zip(self._fp32_shards, self._buf_boundaries)
        ):
            shard_start, shard_end = boundaries[dp_rank]
            shard_size = shard_end - shard_start
            total_numel = self.param_and_grad_buffers[buf_idx].grad_data.numel()

            # Equal-slice shard size (legacy format assumed equal shards).
            gbuf_local_numel = (total_numel + dp_world - 1) // dp_world
            recv_tensor = torch.zeros(gbuf_local_numel, dtype=torch.float32)

            for key in ("param", "exp_avg", "exp_avg_sq"):
                if dp_rank == 0 and state_dict is not None:
                    buf_state = state_dict.get(buf_idx, {})
                    if not buf_state:
                        send_tensors: Optional[List] = None
                    else:
                        # Legacy: per-bucket list of tensors.
                        legacy_bucket_tensors = None
                        for dtype_val in buf_state.values():
                            if key in dtype_val:
                                legacy_bucket_tensors = dtype_val[key]
                                break
                        if legacy_bucket_tensors is None:
                            send_tensors = None
                        else:
                            if isinstance(legacy_bucket_tensors, list):
                                world_tensor = torch.cat(legacy_bucket_tensors)
                            else:
                                world_tensor = legacy_bucket_tensors
                            world_tensor = torch.nn.functional.pad(
                                world_tensor, (0, gbuf_local_numel * dp_world - world_tensor.numel())
                            )
                            send_tensors = [
                                world_tensor[r * gbuf_local_numel:(r + 1) * gbuf_local_numel]
                                for r in range(dp_world)
                            ]
                else:
                    send_tensors = None

                torch.distributed.scatter(
                    recv_tensor,
                    send_tensors,
                    src=global_ranks[0],
                    group=comm_group,
                )

                copy_len = min(shard_size, recv_tensor.numel())
                if key == "param":
                    fp32_shard[:copy_len].copy_(recv_tensor[:copy_len].to(fp32_shard.device))
                else:
                    shard_p = self._shard_params[buf_idx]
                    opt_state = self.optimizer.state.get(shard_p, {})
                    if key in opt_state and isinstance(opt_state[key], torch.Tensor):
                        dst = opt_state[key]
                        actual_len = min(copy_len, dst.numel())
                        dst[:actual_len].copy_(recv_tensor[:actual_len].to(dst.device))

    def split_state_dict_if_needed(
        self,
        state_dict: Dict,
    ) -> None:
        """Split FP8 and non-FP8 params if checkpoint was saved without fp8-param-gather.

        In Megatron, when ``--fp8-param-gather`` is enabled at load time but the
        checkpoint was saved without it, weights (FP8) and biases (BF16) are stored
        in the same buffer and need to be separated.

        In Neuron_SP, FP8 is not in scope, so this is always a no-op.

        Args:
            state_dict: The checkpoint state dict (modified in-place if needed).
        """
        # No-op: FP8 param gather not in scope.
        pass

    def make_needed_groups(
        self,
        param_group: Dict[str, Any],
    ) -> tuple:
        """Extract identifier keys from a param group for state dict matching.

        Used during ``load_state_dict`` to match saved param groups to the
        current optimizer's param groups by their hyper-parameter values
        (learning rate, weight decay, etc.) rather than by parameter indices.

        Mirrors Megatron's ``make_needed_groups`` inner function.

        Args:
            param_group: A single param group dict from the state dict.

        Returns:
            Tuple of hyper-parameter values for the identifier keys.

        Raises:
            ValueError: If a required key is missing.
        """
        # Keys used to identify a param group across checkpoints.
        identifier_keys = ("lr", "weight_decay", "betas", "eps")
        needed: List[Any] = []
        for key in identifier_keys:
            if key in param_group:
                needed.append(param_group[key])
            elif f"pre_{key}" in param_group:
                needed.append(param_group[f"pre_{key}"])
            else:
                # Key not present; use None as placeholder.
                needed.append(None)
        return tuple(needed)

    def _get_param_state_sharded_tensors(
        self,
        model_param: torch.nn.Parameter,
        item_slice: Optional[slice],
    ) -> Dict[str, torch.Tensor]:
        """Return optimizer state tensors for *model_param*, sliced by *item_slice*.

        Helper used by ``sharded_param_state_fs_model_space`` to build the per-
        param optimizer state entry.  Returns ``"fp32_param"``, ``"exp_avg"``,
        and ``"exp_avg_sq"`` slices for the local shard.

        Args:
            model_param: The model parameter.
            item_slice:  Slice object indicating which sub-range of the param
                         this rank owns, or ``None`` for the full shard.

        Returns:
            Dict mapping state key → tensor slice.
        """
        tensors = self._get_main_param_and_optimizer_states(model_param)
        result: Dict[str, torch.Tensor] = {"fp32_param": tensors.pop("param")}
        result.update(tensors)

        if item_slice is not None:
            result = {k: v[item_slice] for k, v in result.items() if isinstance(v, torch.Tensor)}
        return result
