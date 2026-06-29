# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""DeepSpeed optimizer wrappers — ported from Megatron-LM.

Provides the MegatronOptimizer hierarchy (MegatronOptimizer,
MixedPrecisionOptimizer, Float16OptimizerWithFloat16Params, FP32Optimizer,
ChainedOptimizer) with all megatron.core imports rewritten to their
deepspeed.core equivalents.

Original source:
  Megatron-LM/megatron/core/optimizer/optimizer.py
  Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

Import mapping
--------------
  megatron.core.parallel_state              → deepspeed.core.parallel_state
  megatron.core.tensor_parallel             → deepspeed.core.tensor_parallel
  megatron.core.config_logger               → (inlined: no-op stubs)
  megatron.core.dist_checkpointing.*        → deepspeed.core.dist_checkpointing.*
  megatron.core.transformer.module          → param_is_not_shared inlined below
  megatron.core.utils.log_single_rank       → _log_single_rank (local helper)
  megatron.core.optimizer.clip_grads        → deepspeed.core.optimizer.clip_grads
  megatron.core.optimizer.grad_scaler       → (any GradScaler-compatible object)
  megatron.core.optimizer.optimizer_config  → deepspeed.core.optimizer.optimizer_config
  megatron.core.utils.local_multi_tensor_*  → local implementations below
"""

from __future__ import annotations

import copy
import logging
import math
import warnings
from abc import ABC, abstractmethod
from itertools import chain
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from typing_extensions import override

# ---------------------------------------------------------------------------
# multi_tensor_applier / multi_tensor_scale — same fallback chain as Megatron
# ---------------------------------------------------------------------------
try:
    from transformer_engine.pytorch.optimizers import multi_tensor_applier, multi_tensor_scale

    multi_tensor_scale_impl = multi_tensor_scale
except ImportError:
    try:
        import amp_C
        from apex.multi_tensor_apply import multi_tensor_applier

        multi_tensor_scale_impl = amp_C.multi_tensor_scale
    except ImportError:
        warnings.warn(
            'Transformer Engine and Apex are not installed. '
            'Falling back to local implementations of '
            'multi_tensor_applier and multi_tensor_scale'
        )

        def _local_multi_tensor_scale(
            scale: float,
            input_list: List[torch.Tensor],
            output_list: List[torch.Tensor],
        ) -> None:
            """Scale each tensor in *input_list* by *scale*, writing into *output_list*."""
            for src, dst in zip(input_list, output_list):
                dst.copy_(src.float() * scale)

        def _local_multi_tensor_applier(op, noop_flag, tensor_lists, *args):
            """Minimal multi-tensor-applier shim that just calls *op* directly."""
            op(*args, *tensor_lists)

        multi_tensor_applier = _local_multi_tensor_applier
        multi_tensor_scale_impl = _local_multi_tensor_scale

# ---------------------------------------------------------------------------
# DeepSpeed equivalents for megatron.core.* utilities
# ---------------------------------------------------------------------------
import deepspeed.core.parallel_state as parallel_state
import deepspeed.core.tensor_parallel as tensor_parallel

from deepspeed.core.optimizer.clip_grads import get_grad_norm_fp32
from deepspeed.core.optimizer.optimizer_config import OptimizerConfig

# dist_checkpointing — import the same names Megatron uses; fall back gracefully
try:
    from deepspeed.core.dist_checkpointing.mapping import ShardedStateDict
    from deepspeed.core.dist_checkpointing.optimizer import (
        get_param_id_to_sharded_param_map,
        make_sharded_optimizer_tensor,
        optim_state_to_sharding_state,
    )
    from deepspeed.core.dist_checkpointing.utils import add_prefix_for_sharding
    _HAS_DIST_CKPT = True
except ImportError:  # pragma: no cover
    _HAS_DIST_CKPT = False
    ShardedStateDict = dict  # type: ignore[misc,assignment]

    def get_param_id_to_sharded_param_map(*a, **kw):  # type: ignore[misc]
        raise NotImplementedError("deepspeed.core.dist_checkpointing not available")

    def make_sharded_optimizer_tensor(*a, **kw):  # type: ignore[misc]
        raise NotImplementedError("deepspeed.core.dist_checkpointing not available")

    def optim_state_to_sharding_state(*a, **kw):  # type: ignore[misc]
        raise NotImplementedError("deepspeed.core.dist_checkpointing not available")

    def add_prefix_for_sharding(*a, **kw):  # type: ignore[misc]
        raise NotImplementedError("deepspeed.core.dist_checkpointing not available")


# ---------------------------------------------------------------------------
# Megatron clip_grads helpers not present in deepspeed's slim clip_grads.py
# ---------------------------------------------------------------------------
def clip_grad_by_total_norm_fp32(
    params: List[torch.nn.Parameter],
    max_norm: float,
    total_norm: float,
    use_decoupled_grad: bool = False,
) -> None:
    """Scale gradients so their total norm equals *max_norm*.

    Replicates Megatron's clip_grads.clip_grad_by_total_norm_fp32.
    Works for both `.grad` and `.decoupled_grad` depending on
    *use_decoupled_grad*.
    """
    grad_attr = "decoupled_grad" if use_decoupled_grad else "grad"
    clip_coef = max_norm / (total_norm + 1.0e-6)
    if clip_coef < 1.0:
        for p in params:
            g = getattr(p, grad_attr, None)
            if g is not None:
                g.detach().mul_(clip_coef)


def count_zeros_fp32(
    params: List[torch.nn.Parameter],
    grad_stats_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    use_decoupled_grad: bool = False,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> float:
    """Count zero-valued gradient elements across all model-parallel ranks."""
    grad_attr = "decoupled_grad" if use_decoupled_grad else "grad"
    total_num_zeros = torch.tensor([0.0], dtype=torch.float, device="cuda")
    for param in params:
        grad = getattr(param, grad_attr, None)
        if grad is not None:
            grad_not_none = True
            is_not_shared = param_is_not_shared(param)
            is_not_tp_dup = tensor_parallel.param_is_not_tensor_parallel_duplicate(
                param, tp_group
            )
            if grad_not_none and is_not_shared and is_not_tp_dup:
                num_zeros = grad.numel() - grad.count_nonzero()
                total_num_zeros += num_zeros.float()
    if grad_stats_parallel_group is not None:
        torch.distributed.all_reduce(
            total_num_zeros,
            op=torch.distributed.ReduceOp.SUM,
            group=grad_stats_parallel_group,
        )
    return total_num_zeros.item()


# ---------------------------------------------------------------------------
# param_is_not_shared — inlined from megatron.core.transformer.module
# ---------------------------------------------------------------------------
def param_is_not_shared(param: torch.nn.Parameter) -> bool:
    """Return True unless the parameter is tagged as shared."""
    return not getattr(param, "shared", False)


# ---------------------------------------------------------------------------
# log_single_rank — lightweight stand-in for megatron.core.utils.log_single_rank
# ---------------------------------------------------------------------------
def _log_single_rank(lg: logging.Logger, level: int, msg: str, *args) -> None:
    """Log *msg* only on rank 0 (or when distributed is not initialised)."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        lg.log(level, msg, *args)
    elif torch.distributed.get_rank() == 0:
        lg.log(level, msg, *args)


# ---------------------------------------------------------------------------
# config_logger stubs — Megatron uses these for structured logging to disk;
# not needed in DeepSpeed; replaced with no-ops.
# ---------------------------------------------------------------------------
def _has_config_logger_enabled(_config) -> bool:  # noqa: ANN001
    return False


def _log_config_to_disk(_config, _locals, prefix: str = "") -> None:
    pass


logger = getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers (identical to Megatron originals)
# ---------------------------------------------------------------------------

def _zero_grad_group_helper(
    group: List[torch.nn.Parameter], set_to_none: bool, use_decoupled_grad: bool = False
):
    """Zero out the gradient for a group of parameters."""
    for param in group:
        grad_attr = "decoupled_grad" if use_decoupled_grad else "grad"
        if hasattr(param, grad_attr) and getattr(param, grad_attr) is not None:
            if set_to_none:
                setattr(param, grad_attr, None)
            else:
                grad_obj = getattr(param, grad_attr)
                if grad_obj.grad_fn is not None:
                    grad_obj.detach_()
                else:
                    grad_obj.requires_grad_(False)
                grad_obj.zero_()


def _multi_tensor_copy_this_to_that(
    this: List[torch.Tensor],
    that: List[torch.Tensor],
    overflow_buf: Optional[torch.Tensor] = None,
):
    """Use multi-tensor-applier to copy values from one list to another."""
    if overflow_buf is not None:
        overflow_buf.fill_(0)
        multi_tensor_applier(multi_tensor_scale_impl, overflow_buf, [this, that], 1.0)
    else:
        for this_, that_ in zip(this, that):
            that_.copy_(this_)


param_group_identifier_keys = ('wd_mult', 'lr_mult', 'is_expert_parallel', 'is_decoupled_lr')
MTP_GRAD_NORM_GROUP = 'mtp'
GRAD_NORM_GROUP_ATTR = 'grad_norm_group'
SEPARATE_GRAD_NORM_GROUPS = (MTP_GRAD_NORM_GROUP,)


def _get_param_grad_norm_group(param: torch.nn.Parameter) -> Optional[str]:
    """Return the separate gradient-norm group for a parameter, if any."""
    return getattr(param, GRAD_NORM_GROUP_ATTR, None)


def _validate_grad_norm_group(grad_norm_group: str) -> None:
    """Raise if the grad-norm group is not registered for separate clipping."""
    if grad_norm_group not in SEPARATE_GRAD_NORM_GROUPS:
        raise ValueError(
            f"Unknown grad_norm_group '{grad_norm_group}'. Register it in "
            "SEPARATE_GRAD_NORM_GROUPS before tagging parameters with it."
        )


def _is_separate_grad_norm_group(grad_norm_group: Optional[str]) -> bool:
    """Return whether the optimizer computes a separate norm for this group."""
    if grad_norm_group is None:
        return False
    _validate_grad_norm_group(grad_norm_group)
    return True


def copy_optimizer_param_metadata(destination: torch.Tensor, source: torch.Tensor) -> None:
    """Copy optimizer-relevant metadata when creating param views/copies."""
    if hasattr(source, 'shared'):
        destination.shared = source.shared
    if hasattr(source, GRAD_NORM_GROUP_ATTR):
        setattr(destination, GRAD_NORM_GROUP_ATTR, getattr(source, GRAD_NORM_GROUP_ATTR))


# ---------------------------------------------------------------------------
# MegatronOptimizer — abstract base
# ---------------------------------------------------------------------------

class MegatronOptimizer(ABC):
    """Base class for all Megatron/DeepSpeed optimizers.

    Provides a consistent interface for gradient management, parameter
    access, and state-dict handling across different optimization types.

    Args:
        optimizer (torch.optim.Optimizer): The base PyTorch optimizer.
        config (OptimizerConfig): The optimizer configuration.
        init_state_fn (Callable, optional): Function to initialize optimizer state.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizerConfig,
        init_state_fn: Callable = lambda x: None,
    ):
        self.optimizer = optimizer
        if self.optimizer is None:
            warnings.warn(
                f"WARNING: there is no optimizer on RANK {torch.distributed.get_rank()}. "
                "This may be expected if you have frozen sub-models."
            )
        self.config = config
        self.init_state_fn = init_state_fn

    def get_parameters(self) -> List[torch.nn.Parameter]:
        """Get list of parameters wrapped in optimizer."""
        params = []
        if hasattr(self.optimizer, 'param_groups'):
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    params.append(param)
        return params

    def _filter_grads_for_norm(
        self,
        params: List[torch.nn.Parameter],
        param_filter: Optional[Callable[[torch.nn.Parameter], bool]] = None,
    ) -> List[torch.Tensor]:
        """Filter parameter gradients for norm computation."""
        grads_for_norm = []
        for param in params:
            if param_filter is not None and not param_filter(param):
                continue
            if self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8 or (
                self.config.use_precision_aware_optimizer
                and getattr(param, "__fsdp_param__", False)
            ):
                grad = param.decoupled_grad if hasattr(param, "decoupled_grad") else None
                if (
                    getattr(param, "__fsdp_param__", False)
                    and grad is not None
                    and hasattr(grad, "_local_tensor")
                ):
                    grad = grad._local_tensor
            elif getattr(param, "__fsdp_param__", False):
                grad = param.grad._local_tensor if param.grad is not None else None
            else:
                grad = param.grad
            grad_not_none = grad is not None
            is_not_shared = param_is_not_shared(param)
            is_not_tp_duplicate = tensor_parallel.param_is_not_tensor_parallel_duplicate(
                param, getattr(self, 'tp_group', None)
            )
            if grad_not_none and is_not_shared and is_not_tp_duplicate:
                grads_for_norm.append(grad)
        return grads_for_norm

    def get_grads_for_grad_norm(self, grad_norm_group: Optional[str] = None) -> List[torch.Tensor]:
        """Get gradients for norm computation."""
        if grad_norm_group is not None:
            _validate_grad_norm_group(grad_norm_group)
            param_filter = lambda p: _get_param_grad_norm_group(p) == grad_norm_group
        else:
            param_filter = lambda p: not _is_separate_grad_norm_group(_get_param_grad_norm_group(p))
        return self._filter_grads_for_norm(self.get_parameters(), param_filter=param_filter)

    def has_grad_norm_group(self, grad_norm_group: str) -> bool:
        """Whether any rank in this optimizer's grad-stats group owns grouped params."""
        _validate_grad_norm_group(grad_norm_group)
        if getattr(self, '_has_grad_norm_group_cache', None) is None:
            self._has_grad_norm_group_cache = {}
        cache = self._has_grad_norm_group_cache
        if grad_norm_group not in cache:
            local = False
            for param in self.get_parameters():
                param_grad_norm_group = _get_param_grad_norm_group(param)
                if _is_separate_grad_norm_group(param_grad_norm_group):
                    local = local or param_grad_norm_group == grad_norm_group
            flag = torch.tensor([1 if local else 0], dtype=torch.int, device='cuda')
            torch.distributed.all_reduce(
                flag, op=torch.distributed.ReduceOp.MAX, group=self.get_grad_stats_parallel_group()
            )
            cache[grad_norm_group] = bool(flag.item() > 0)
        return cache[grad_norm_group]

    def get_grad_stats_parallel_group(self) -> torch.distributed.ProcessGroup:
        """Process group for reducing gradient statistics (num_zeros & norm)."""
        if hasattr(self, 'model_parallel_group'):
            warnings.warn(
                "WARNING: `optimizer.model_parallel_group` deprecated and renamed to "
                "`optimizer.grad_stats_parallel_group`. The previous name will be "
                "removed in a future release."
            )
            self.grad_stats_parallel_group = self.model_parallel_group
            delattr(self, "model_parallel_group")
            return self.grad_stats_parallel_group
        if hasattr(self, 'grad_stats_parallel_group'):
            return self.grad_stats_parallel_group
        return parallel_state.get_model_parallel_group()

    @abstractmethod
    def prepare_grads(self) -> bool:
        """Pre-processing gradients before the optimizer step, returns whether inf/nan is found."""
        return False

    @abstractmethod
    def step_with_ready_grads(self) -> bool:
        """Step the optimizer with ready gradients, return successful."""
        return True

    @torch.no_grad()
    def get_grad_norm(self):
        """Compute and return grad norm."""
        grads_for_norm = self.get_grads_for_grad_norm()
        total_norm = get_grad_norm_fp32(
            grads_for_norm, grad_stats_parallel_group=self.get_grad_stats_parallel_group()
        )
        return total_norm

    @torch.no_grad()
    def _compute_grad_norms_by_group(self) -> Dict[str, float]:
        """Compute gradient norms for registered separate grad-norm groups."""
        self.grad_norms_by_group = {}
        for grad_norm_group in SEPARATE_GRAD_NORM_GROUPS:
            if self.has_grad_norm_group(grad_norm_group):
                grouped_grads = self.get_grads_for_grad_norm(grad_norm_group)
                group_grad_norm = get_grad_norm_fp32(
                    grouped_grads, grad_stats_parallel_group=self.get_grad_stats_parallel_group()
                )
                self.grad_norms_by_group[grad_norm_group] = group_grad_norm
        return self.grad_norms_by_group

    def clip_grad_norm(self, clip_grad: float) -> float:
        """Compute and return grad norm, also clip grads."""
        self.grad_norms_by_group = {}
        params = self.get_parameters()
        if params:
            grads_for_norm = self.get_grads_for_grad_norm()
        else:
            grads_for_norm = []
        grad_norm = get_grad_norm_fp32(
            grads_for_norm, grad_stats_parallel_group=self.get_grad_stats_parallel_group()
        )

        if clip_grad > 0.0 and params:
            self._compute_grad_norms_by_group()

            def use_decoupled_grad(param_list):
                return self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8 or (
                    self.config.use_precision_aware_optimizer
                    and getattr(param_list[0], "__fsdp_param__", False)
                )

            main_params = []
            params_by_grad_norm_group = {}
            for p in params:
                grad_norm_group = _get_param_grad_norm_group(p)
                if _is_separate_grad_norm_group(grad_norm_group):
                    params_by_grad_norm_group.setdefault(grad_norm_group, []).append(p)
                else:
                    main_params.append(p)
            if main_params:
                clip_grad_by_total_norm_fp32(
                    main_params,
                    clip_grad,
                    grad_norm,
                    use_decoupled_grad=use_decoupled_grad(main_params),
                )
            for grad_norm_group, grouped_params in params_by_grad_norm_group.items():
                group_grad_norm = self.grad_norms_by_group.get(grad_norm_group)
                if group_grad_norm is None:
                    continue
                clip_grad_by_total_norm_fp32(
                    grouped_params,
                    clip_grad,
                    group_grad_norm,
                    use_decoupled_grad=use_decoupled_grad(grouped_params),
                )
        return grad_norm

    def count_zeros(self) -> float:
        """Count number of zeros in model's gradients."""
        params = self.get_parameters()
        return count_zeros_fp32(
            params,
            grad_stats_parallel_group=self.get_grad_stats_parallel_group(),
            use_decoupled_grad=self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
            or (
                self.config.use_precision_aware_optimizer
                and getattr(params[0], "__fsdp_param__", False)
            ),
            tp_group=getattr(self, 'tp_group', None),
        )

    @abstractmethod
    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients and prepare for next forward pass."""
        pass

    @abstractmethod
    def get_loss_scale(self) -> torch.Tensor:
        """Get current loss scale factor. Returns a CUDA tensor of size 1."""
        pass

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Simple scaling."""
        return self.get_loss_scale() * loss

    @abstractmethod
    def reload_model_params(self, state_dict=None):
        """Refreshes any internal state from the current model parameters."""
        pass

    @abstractmethod
    def state_dict(self):
        """Return state_dict."""
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        """Load pass-in `state_dict`."""
        pass

    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value

    state = property(_get_state, _set_state)

    def _get_param_groups(self):
        if self.is_stub_optimizer:
            return []
        else:
            return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)

    @abstractmethod
    def step(self):
        """Step the optimizer."""
        pass

    @abstractmethod
    def sharded_state_dict(
        self,
        model_sharded_state_dict: ShardedStateDict,
        is_loading: bool = False,
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Build sharded state dict for the optimizer."""

    @staticmethod
    def _extract_common_per_param_step(state_dict) -> Union[int, torch.Tensor, None]:
        common_step = None
        for param_idx, param_state in state_dict['state'].items():
            param_step = param_state.get('step', None)
            if param_step is not None:
                if common_step is None:
                    common_step = param_step
                elif common_step != param_step:
                    raise ValueError(
                        "The optimizer step differs per parameter. Mcore only supports "
                        "optimizers whose step is shared across all parameters."
                    )
        return common_step

    @staticmethod
    def _restore_common_per_param_step(state_dict: Dict, step: Union[int, torch.Tensor]):
        for param_idx, param_state in state_dict['state'].items():
            param_state['step'] = copy.deepcopy(step)

    def offload_to_cpu(self):
        """Move optimizer state tensors to CPU to free GPU memory during inference."""
        if getattr(self, 'optimizer', None) is not None and not getattr(
            self, 'is_stub_optimizer', False
        ):
            _log_single_rank(logger, logging.INFO, '[OFFLOAD] moving optimizer state to CPU')
            for param_group in self.optimizer.param_groups:
                for p in param_group['params']:
                    if isinstance(p, torch.Tensor) and p.is_cuda:
                        p.data = p.data.cpu()
            for state_dict in self.optimizer.state.values():
                for k, v in state_dict.items():
                    if isinstance(v, torch.Tensor) and v.is_cuda:
                        state_dict[k] = v.cpu()
            torch.cuda.empty_cache()

    def restore_from_cpu(self):
        """Restore optimizer state tensors from CPU back to GPU for training."""
        if getattr(self, 'optimizer', None) is not None and not getattr(
            self, 'is_stub_optimizer', False
        ):
            _log_single_rank(logger, logging.INFO, '[RESTORE] moving optimizer state back to GPU')
            for param_group in self.optimizer.param_groups:
                for p in param_group['params']:
                    if isinstance(p, torch.Tensor) and not p.is_cuda:
                        p.data = p.data.cuda()
            for state_dict in self.optimizer.state.values():
                for k, v in state_dict.items():
                    if isinstance(v, torch.Tensor) and not v.is_cuda:
                        state_dict[k] = v.cuda()

    @staticmethod
    def _filter_and_reorder_param_groups(
        current_groups: List[Dict], state_dict_groups: List[Dict]
    ) -> List[Dict]:
        """Filter and reorder state_dict parameter groups to match current optimizer groups."""
        needed_groups = [
            tuple(g[key] if key in g else g[f"pre_{key}"] for key in param_group_identifier_keys)
            for g in current_groups
        ]
        params_in_state_dict_order = [g['params'] for g in state_dict_groups]
        loaded_groups_map = {
            tuple(
                group[key] if key in group else group[f"pre_{key}"]
                for key in param_group_identifier_keys
            ): group
            for group in state_dict_groups
        }
        final_groups = []
        for key, params in zip(needed_groups, params_in_state_dict_order):
            if key not in loaded_groups_map:
                available_keys = '\n'.join(str(k) for k in loaded_groups_map.keys())
                raise ValueError(
                    f"Could not find parameter group with key {key} in loaded checkpoint.\n"
                    f"Available keys:\n{available_keys}\n"
                    f"Parameter group key definition: {param_group_identifier_keys}"
                )
            group = loaded_groups_map[key]
            group['params'] = params
            final_groups.append(group)
        return final_groups


# ---------------------------------------------------------------------------
# MixedPrecisionOptimizer
# ---------------------------------------------------------------------------

class MixedPrecisionOptimizer(MegatronOptimizer):
    """Base class for both the float-16 and the distributed optimizer.

    Args:
        optimizer (torch.optim.Optimizer): base optimizer such as Adam or SGD.
        config (OptimizerConfig): configuration object for optimizer.
        grad_scaler: GradScaler-compatible object (or None for BF16).
        init_state_fn (Callable, optional): function to initialize state in the optimizer.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizerConfig,
        grad_scaler,
        init_state_fn: Callable,
    ):
        if _has_config_logger_enabled(config):
            _log_config_to_disk(config, locals(), prefix=type(self).__name__)

        super().__init__(optimizer, config, init_state_fn)
        self.grad_scaler = grad_scaler

        if self.grad_scaler is None:
            assert not self.config.fp16, 'fp16 expects a grad scaler.'

        if self.grad_scaler:
            self.found_inf = torch.tensor([0.0], dtype=torch.float, device='cuda')

        if self.config.bf16:
            self._dummy_overflow_buf = None
        else:
            self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')

        if self.grad_scaler is None:
            self._scale_one = torch.tensor([1.0], dtype=torch.float, device='cuda')

    def get_loss_scale(self):
        if self.grad_scaler is None:
            return self._scale_one
        return self.grad_scaler.scale

    def reload_model_params(self, state_dict=None):
        if self.param_groups:
            self._copy_model_params_to_main_params(state_dict=state_dict)

    def _unscale_main_grads_and_check_for_nan(self):
        if not self.is_stub_optimizer:
            main_grads = self._collect_main_grad_data_for_unscaling()

        self.found_inf.fill_(0.0)

        if not self.is_stub_optimizer:
            torch._amp_foreach_non_finite_check_and_unscale_(
                main_grads, self.found_inf, self.grad_scaler.inv_scale
            )

        torch.distributed.all_reduce(
            self.found_inf,
            op=torch.distributed.ReduceOp.MAX,
            group=self.get_grad_stats_parallel_group(),
        )

        found_inf_flag = self.found_inf.item() > 0
        return found_inf_flag

    @torch.no_grad()
    def prepare_grads(self) -> bool:
        """Pre-processing gradients before the optimizer step, returns whether inf/nan is found."""
        timers = self.config.timers

        if timers is not None:
            timers('optimizer-copy-to-main-grad', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        if not self.is_stub_optimizer:
            self._copy_model_grads_to_main_grads()
        if timers is not None:
            timers('optimizer-copy-to-main-grad').stop()

        if self.grad_scaler:
            if timers is not None:
                timers('optimizer-unscale-and-check-inf', log_level=1).start(
                    barrier=self.config.barrier_with_L1_time
                )
            found_inf_flag = self._unscale_main_grads_and_check_for_nan()
            if timers is not None:
                timers('optimizer-unscale-and-check-inf').stop()

            self.grad_scaler.update(found_inf_flag)
            return found_inf_flag

        return False

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        """Step the optimizer with ready gradients, return successful."""
        timers = self.config.timers
        if timers is not None:
            timers('optimizer-inner-step', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        if not self.is_stub_optimizer:
            self.optimizer.step()
        if timers is not None:
            timers('optimizer-inner-step').stop()

        if timers is not None:
            timers('optimizer-copy-main-to-model-params', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        if not self.is_stub_optimizer:
            if self.config.reuse_grad_buf_for_mxfp8_param_ag:
                if not self.config.overlap_param_gather:
                    self._copy_main_params_to_param_buffer()
            else:
                self._copy_main_params_to_model_params()

        if timers is not None:
            timers('optimizer-copy-main-to-model-params').stop()

        return True

    @torch.no_grad()
    def step(self):
        timers = self.config.timers
        self.grad_norms_by_group = {}

        found_inf_flag = self.prepare_grads()
        if found_inf_flag:
            return False, None, None

        if timers is not None:
            timers('optimizer-clip-main-grad', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        grad_norm = 0.0
        if self.config.clip_grad > 0.0:
            grad_norm = self.clip_grad_norm(self.config.clip_grad)
        if timers is not None:
            timers('optimizer-clip-main-grad').stop()

        if timers is not None:
            timers('optimizer-count-zeros', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        num_zeros_in_grad = self.count_zeros() if self.config.log_num_zeros_in_grad else 0
        if timers is not None:
            timers('optimizer-count-zeros').stop()

        success = self.step_with_ready_grads()
        return success, grad_norm, num_zeros_in_grad


# ---------------------------------------------------------------------------
# Float16OptimizerWithFloat16Params
# ---------------------------------------------------------------------------

class Float16OptimizerWithFloat16Params(MixedPrecisionOptimizer):
    """Float16 optimizer for fp16 and bf16 data types.

    Args:
        optimizer (torch.optim.Optimizer): base optimizer such as Adam or SGD.
        config (OptimizerConfig): configuration object for optimizer.
        grad_scaler: GradScaler-compatible object (or None for bf16).
        init_state_fn (Callable, optional): function to initialize state in the optimizer.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizerConfig,
        grad_scaler,
        init_state_fn: Callable,
    ):
        super().__init__(optimizer, config, grad_scaler, init_state_fn)

        if optimizer:
            self.float16_groups = []
            self.fp32_from_float16_groups = []
            self.fp32_from_fp32_groups = []

            for param_group in self.optimizer.param_groups:
                float16_params_this_group = []
                fp32_params_this_group = []
                fp32_from_float16_params_this_group = []
                for i, param in enumerate(param_group['params']):
                    if param.requires_grad:
                        if param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:
                            float16_params_this_group.append(param)
                            main_param = param.detach().clone().float()
                            tensor_parallel.copy_tensor_model_parallel_attributes(main_param, param)
                            copy_optimizer_param_metadata(main_param, param)
                            param_group['params'][i] = main_param
                            param.main_param = main_param
                            fp32_from_float16_params_this_group.append(main_param)
                            if param in self.optimizer.state:
                                self.optimizer.state[main_param] = self.optimizer.state.pop(param)
                        elif param.type() == 'torch.cuda.FloatTensor':
                            fp32_params_this_group.append(param)
                            param_group['params'][i] = param
                        else:
                            raise TypeError(
                                'Wrapped parameters must be one of '
                                'torch.cuda.FloatTensor,  '
                                'torch.cuda.HalfTensor, or '
                                'torch.cuda.BFloat16Tensor. '
                                'Received {}'.format(param.type())
                            )

                self.float16_groups.append(float16_params_this_group)
                self.fp32_from_float16_groups.append(fp32_from_float16_params_this_group)
                self.fp32_from_fp32_groups.append(fp32_params_this_group)
            self.is_stub_optimizer = False
        else:
            self.is_stub_optimizer = True

    def zero_grad(self, set_to_none=True):
        """Zero gradients for float16 and fp32 parameter groups."""
        if self.is_stub_optimizer:
            return
        for group in self.float16_groups:
            _zero_grad_group_helper(group, set_to_none)
        for group in self.fp32_from_float16_groups:
            _zero_grad_group_helper(group, set_to_none)
        for group in self.fp32_from_fp32_groups:
            _zero_grad_group_helper(group, set_to_none)

    def _collect_main_grad_data_for_unscaling(self):
        if self.is_stub_optimizer:
            return
        main_grads = []
        for main_group in self.fp32_from_float16_groups:
            for main_param in main_group:
                if main_param.grad is not None:
                    main_grads.append(main_param.grad.data)
        for main_group in self.fp32_from_fp32_groups:
            for main_param in main_group:
                if main_param.grad is not None:
                    main_grads.append(main_param.grad.data)
        return main_grads

    def _get_model_and_main_params_data_float16(self):
        model_data = []
        main_data = []
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                model_data.append(model_param.data)
                main_data.append(main_param.data)
        return model_data, main_data

    def _copy_model_grads_to_main_grads(self):
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                if hasattr(model_param, 'main_grad'):
                    main_param.grad = model_param.main_grad.float()
                else:
                    if model_param.grad is not None:
                        main_param.grad = model_param.grad.float()
                model_param.grad = None
        for model_group in self.fp32_from_fp32_groups:
            for model_param in model_group:
                model_param.grad = model_param.main_grad

    def _copy_main_params_to_model_params(self):
        model_data, main_data = self._get_model_and_main_params_data_float16()
        _multi_tensor_copy_this_to_that(
            this=main_data, that=model_data, overflow_buf=self._dummy_overflow_buf
        )

    def _copy_model_params_to_main_params(self, state_dict=None):
        assert state_dict is None, "Initialize main params from state dict is not supported"
        model_data, main_data = self._get_model_and_main_params_data_float16()
        _multi_tensor_copy_this_to_that(
            this=model_data, that=main_data, overflow_buf=self._dummy_overflow_buf
        )

    def state_dict(self, is_loading: bool = False):
        if is_loading:
            self.init_state_fn(self.optimizer, self.config)
        state_dict = {}
        state_dict['optimizer'] = self.optimizer.state_dict()
        if self.grad_scaler:
            state_dict['grad_scaler'] = self.grad_scaler.state_dict()
        state_dict['fp32_from_fp16_params'] = self.fp32_from_float16_groups
        return state_dict

    def sharded_state_dict(
        self,
        model_sharded_state_dict: ShardedStateDict,
        is_loading: bool = False,
        metadata: Optional[dict] = None,
    ):
        if is_loading:
            self.init_state_fn(self.optimizer, self.config)

        state_dict = self.state_dict()

        id_to_sharded_param_map = get_param_id_to_sharded_param_map(
            model_sharded_state_dict, chain.from_iterable(g for g in self.float16_groups)
        )

        assert len(state_dict['fp32_from_fp16_params']) == len(
            state_dict['optimizer']['param_groups']
        )
        state_dict['fp32_from_fp16_params'] = [
            [
                make_sharded_optimizer_tensor(
                    id_to_sharded_param_map[param_id],
                    fp32_param,
                    prefix='optimizer.state.fp32_param',
                )
                for param_id, fp32_param in zip(state_group['params'], fp32_group)
            ]
            for fp32_group, state_group in zip(
                state_dict['fp32_from_fp16_params'], state_dict['optimizer']['param_groups']
            )
        ]

        step = self._extract_common_per_param_step(state_dict['optimizer'])
        optim_state_to_sharding_state(
            state_dict['optimizer'], id_to_sharded_param_map, exclude_keys="step"
        )
        if step:
            state_dict['optimizer']['state']['common_step'] = step
        return state_dict

    def load_state_dict(self, state_dict):
        optimizer_key = 'optimizer'
        if optimizer_key not in state_dict:
            optimizer_key = 'optimizer_state_dict'
            logger.info('***WARNING*** loading optimizer from an old checkpoint ...')
        if 'common_step' in state_dict[optimizer_key]['state']:
            common_step = state_dict[optimizer_key]['state'].pop('common_step')
            self._restore_common_per_param_step(state_dict[optimizer_key], common_step)

        state_dict[optimizer_key]['param_groups'] = self._filter_and_reorder_param_groups(
            self.optimizer.param_groups, state_dict[optimizer_key]['param_groups']
        )
        self.optimizer.load_state_dict(state_dict[optimizer_key])

        if 'grad_scaler' not in state_dict:
            if self.config.fp16:
                logger.info('***WARNING*** found an old checkpoint, will not load grad scaler ...')
        else:
            if self.grad_scaler:
                self.grad_scaler.load_state_dict(state_dict['grad_scaler'])
            else:
                logger.info(
                    '***WARNING*** fould the grad scaler in the '
                    'checkpoint but it is None in the class. '
                    'Skipping loading grad scaler ...'
                )

        fp32_from_float16_params_key = 'fp32_from_fp16_params'
        if fp32_from_float16_params_key not in state_dict:
            fp32_from_float16_params_key = 'fp32_from_fp16'
        for current_group, saved_group in zip(
            self.fp32_from_float16_groups, state_dict[fp32_from_float16_params_key]
        ):
            for current_param, saved_param in zip(current_group, saved_group):
                current_param.data.copy_(saved_param.data)


# ---------------------------------------------------------------------------
# FP32Optimizer
# ---------------------------------------------------------------------------

class FP32Optimizer(MegatronOptimizer):
    """Float32 optimizer.

    Args:
        optimizer (torch.optim.Optimizer): base optimizer such as Adam or SGD.
        config (OptimizerConfig): configuration object for optimizer.
        init_state_fn (Callable, optional): function to initialize state in the optimizer.
    """

    def __init__(
        self, optimizer: torch.optim.Optimizer, config: OptimizerConfig, init_state_fn: Callable
    ):
        if _has_config_logger_enabled(config):
            _log_config_to_disk(config, locals(), prefix=type(self).__name__)

        super(FP32Optimizer, self).__init__(optimizer, config, init_state_fn)
        self._scale = torch.tensor([1.0], dtype=torch.float, device='cuda')
        self.is_stub_optimizer = True if optimizer is None else False

    def zero_grad(self, set_to_none=True):
        """Copied from torch.optim.optimizer."""
        if self.is_stub_optimizer:
            return
        for group in self.optimizer.param_groups:
            _zero_grad_group_helper(group['params'], set_to_none)

    def get_loss_scale(self):
        """FP32 optimizer does not do any scaling."""
        return self._scale

    @torch.no_grad()
    def prepare_grads(self) -> bool:
        """Pre-processing gradients before the optimizer step, returns whether inf/nan is found."""
        if self.is_stub_optimizer:
            return False
        timers = self.config.timers

        if timers is not None:
            timers('optimizer-copy-to-main-grad', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                if hasattr(param, 'main_grad'):
                    param.grad = param.main_grad
        if timers is not None:
            timers('optimizer-copy-to-main-grad').stop()

        return False

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        """Step the optimizer with ready gradients, return successful."""
        if self.is_stub_optimizer:
            return True
        timers = self.config.timers

        if timers is not None:
            timers('optimizer-inner-step', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        self.optimizer.step()
        if timers is not None:
            timers('optimizer-inner-step').stop()

        return True

    @torch.no_grad()
    def step(self):
        """Clip gradients (if needed) and step the base optimizer."""
        timers = self.config.timers
        self.grad_norms_by_group = {}

        found_inf_flag = self.prepare_grads()
        if found_inf_flag:
            return False, None, None

        if timers is not None:
            timers('optimizer-clip-main-grad', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        grad_norm = None
        if self.config.clip_grad > 0.0:
            grad_norm = self.clip_grad_norm(self.config.clip_grad)
        if timers is not None:
            timers('optimizer-clip-main-grad').stop()

        if timers is not None:
            timers('optimizer-count-zeros', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        num_zeros_in_grad = self.count_zeros() if self.config.log_num_zeros_in_grad else None
        if timers is not None:
            timers('optimizer-count-zeros').stop()

        success = self.step_with_ready_grads()
        return success, grad_norm, num_zeros_in_grad

    def reload_model_params(self, state_dict=None):
        pass

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        if 'common_step' in state_dict['state']:
            common_step = state_dict['state'].pop('common_step')
            self._restore_common_per_param_step(state_dict, common_step)
        state_dict['param_groups'] = self._filter_and_reorder_param_groups(
            self.optimizer.param_groups, state_dict['param_groups']
        )
        self.optimizer.load_state_dict(state_dict)

    def sharded_state_dict(
        self,
        model_sharded_state_dict: ShardedStateDict,
        is_loading: bool = False,
        metadata: Optional[dict] = None,
    ):
        if is_loading:
            self.init_state_fn(self.optimizer, self.config)

        state_dict = self.state_dict()
        id_to_sharded_param_map = get_param_id_to_sharded_param_map(
            model_sharded_state_dict, self.get_parameters()
        )
        step = self._extract_common_per_param_step(state_dict)
        optim_state_to_sharding_state(state_dict, id_to_sharded_param_map, exclude_keys="step")
        if step:
            state_dict['state']['common_step'] = step
        return state_dict


# ---------------------------------------------------------------------------
# ProxyDict (unchanged from Megatron)
# ---------------------------------------------------------------------------

class ProxyDict:
    """A dictionary-like object that proxies to a list of dictionaries.

    e.g., ProxyDict([{'a': 1}, {'b': 2}]) behaves like::

        {(0, 'a'): 1, (1, 'b'): 2}
    """

    def __init__(self, inner_dicts: List[dict]):
        self._inner_dicts = inner_dicts

    def __getitem__(self, key: Tuple[int, str]):
        idx, inner_key = key
        return self._inner_dicts[idx].get(inner_key)

    def __setitem__(self, key: Tuple[int, str], value: Any):
        idx, inner_key = key
        self._inner_dicts[idx][inner_key] = value

    def __len__(self) -> int:
        return sum([len(d) for d in self._inner_dicts])

    def __iter__(self):
        for idx, inner_dict in enumerate(self._inner_dicts):
            for inner_key in inner_dict:
                yield (idx, inner_key)

    def items(self):
        """Return generator over underlying items."""
        for idx, inner_dict in enumerate(self._inner_dicts):
            for inner_key, value in inner_dict.items():
                yield (idx, inner_key), value


# ---------------------------------------------------------------------------
# ChainedOptimizer
# ---------------------------------------------------------------------------

class ChainedOptimizer(MegatronOptimizer):
    """ChainedOptimizer is designed for a collection of optimizers.

    These optimizers are responsible for different parts of multiple models for
    a training task and will be executed one-by-one when the model is updated.

    Args:
        chained_optimizers: a list of optimizers.
    """

    def __init__(self, chained_optimizers: List[MegatronOptimizer]):
        self.model_chunks = []
        if chained_optimizers:
            self.config = getattr(chained_optimizers[0], 'config', None)
            for optimizer in chained_optimizers:
                if hasattr(optimizer, 'model_chunks'):
                    for model_chunk in optimizer.model_chunks:
                        if model_chunk not in self.model_chunks:
                            self.model_chunks.append(model_chunk)
                assert self.config == getattr(optimizer, 'config', None)
            self.is_stub_optimizer = all(
                getattr(optimizer, 'is_stub_optimizer', False) for optimizer in chained_optimizers
            )
        else:
            self.is_stub_optimizer = True
        self.chained_optimizers = chained_optimizers

    @property
    def optimizer(self):
        """Access underlying optimizer when only one optimizer included."""
        assert (
            len(self.chained_optimizers) == 1
        ), "ChainedOptimizer has more than one optimizer when accessing self.optimizer"
        return self.chained_optimizers[0].optimizer

    @property
    def param_groups(self) -> List[dict]:
        """Get param_groups aggregated over underlying optimizers."""
        param_groups = []
        for optimizer in self.chained_optimizers:
            param_groups += optimizer.param_groups
        return param_groups

    @override
    def get_parameters(self) -> List[torch.nn.Parameter]:
        """Get list of parameters wrapped in all chained optimizers."""
        params = []
        for optimizer in self.chained_optimizers:
            params.extend(optimizer.get_parameters())
        return params

    @property
    def state(self) -> ProxyDict:
        """Return optimizer state with tuple keys."""
        return ProxyDict([opt.state for opt in self.chained_optimizers])

    def zero_grad(self, set_to_none=True):
        for optimizer in self.chained_optimizers:
            optimizer.zero_grad(set_to_none)

    def get_loss_scale(self):
        if self.chained_optimizers:
            return self.chained_optimizers[0].get_loss_scale()
        else:
            return torch.tensor([1.0], dtype=torch.float32, device=torch.cuda.current_device())

    def _split_state_dict(self, state_dict):
        """Split the state dict into sub-state dicts per sub-optimizer."""
        state_dicts = [None] * len(self.chained_optimizers)
        if state_dict is not None:
            if len(self.model_chunks) == 1:
                state_dicts = [state_dict] * len(self.chained_optimizers)
            else:
                prefix = "model" if "model0" in state_dict.keys() else "model_"
                chunk_to_global_idx = {chunk: idx for idx, chunk in enumerate(self.model_chunks)}
                for optimizer_idx, optimizer in enumerate(self.chained_optimizers):
                    if hasattr(optimizer, "model_chunks"):
                        d = {}
                        for chunk_idx, model_chunk in enumerate(optimizer.model_chunks):
                            assert model_chunk in chunk_to_global_idx
                            global_idx = chunk_to_global_idx[model_chunk]
                            assert f"{prefix}{global_idx}" in state_dict
                            d[f"{prefix}{chunk_idx}"] = state_dict[f"{prefix}{global_idx}"]
                        if len(d) > 0:
                            state_dicts[optimizer_idx] = d
        return state_dicts

    def reload_model_params(self, state_dict=None):
        state_dicts = self._split_state_dict(state_dict)
        for idx, optimizer in enumerate(self.chained_optimizers):
            optimizer.reload_model_params(state_dict=state_dicts[idx])

    def state_dict(self):
        if len(self.chained_optimizers) == 1:
            return self.chained_optimizers[0].state_dict()
        else:
            return [optimizer.state_dict() for optimizer in self.chained_optimizers]

    def sharded_state_dict(
        self, model_sharded_state_dict: ShardedStateDict, is_loading: bool = False, **kwargs
    ):
        metadata = kwargs.get('metadata') or {}
        from deepspeed.core.optimizer.distrib_optimizer import DistributedOptimizer

        should_add_prefix = (
            "distrib_optim_sharding_type" in metadata
            and metadata["distrib_optim_sharding_type"]
            not in DistributedOptimizer.checkpoint_fully_reshardable_formats
        ) or not metadata.get('chained_optim_avoid_prefix', False)

        if len(self.chained_optimizers) == 1:
            return self.chained_optimizers[0].sharded_state_dict(
                model_sharded_state_dict, is_loading, **kwargs
            )
        else:
            self._synchronize_steps()
            sharded_state_dict = {}
            for optimizer_idx, optimizer in enumerate(self.chained_optimizers):
                optim_state_dict = optimizer.sharded_state_dict(
                    model_sharded_state_dict, is_loading, **kwargs
                )
                if should_add_prefix:
                    add_prefix_for_sharding(optim_state_dict, f'chained_{optimizer_idx}.')
                sharded_state_dict[optimizer_idx] = optim_state_dict
            return sharded_state_dict

    def load_state_dict(self, state_dict):
        if len(self.chained_optimizers) == 1:
            self.chained_optimizers[0].load_state_dict(state_dict)
            return
        if len(self.chained_optimizers) != len(state_dict):
            raise RuntimeError(
                f'Expected {len(self.chained_optimizers)} entries'
                f' in state dict, but got {len(state_dict)}.'
            )
        if isinstance(state_dict, dict):
            state_dict = (v for k, v in sorted(state_dict.items()))
        for optimizer, state in zip(self.chained_optimizers, state_dict):
            optimizer.load_state_dict(state)
        self._synchronize_steps()

    @torch.no_grad()
    def prepare_grads(self) -> bool:
        """Pre-processing gradients before the optimizer step, returns whether inf/nan is found."""
        found_inf_flag = False
        for optimizer in self.chained_optimizers:
            found_inf_flag |= optimizer.prepare_grads()
        return found_inf_flag

    def _step(self) -> bool:
        """Step all optimizers in this chain."""
        success = True
        for optimizer_idx, optimizer in enumerate(self.chained_optimizers):
            success &= optimizer.step_with_ready_grads()
            if self.config.overlap_param_gather_with_optimizer_step and optimizer_idx == 0:
                assert success
                assert len(optimizer.model_chunks) == 1
                optimizer.model_chunks[0].start_param_sync(force_dispatch=True)
        return success

    def _should_defer_mxfp8_param_sync(self) -> bool:
        """Return whether MXFP8 param sync should be deferred until chained steps finish."""
        if not self.config.reuse_grad_buf_for_mxfp8_param_ag:
            return False
        from deepspeed.core.optimizer.distrib_optimizer import DistributedOptimizer
        for optimizer in self.chained_optimizers:
            if not isinstance(optimizer, DistributedOptimizer):
                continue
            if not optimizer.ddp_config.overlap_param_gather:
                return True
        return False

    def _enable_deferred_mxfp8_param_sync(self) -> List[Tuple[Any, Any]]:
        """Enable deferred DistOpt param sync and collect bucket groups to sync later."""
        from deepspeed.core.optimizer.distrib_optimizer import DistributedOptimizer

        deferred_bucket_groups = []
        deferred_bucket_group_ids = set()

        for optimizer in self.chained_optimizers:
            if not isinstance(optimizer, DistributedOptimizer):
                continue
            optimizer._defer_param_sync = True
            for model_chunk in optimizer.model_chunks:
                for bucket_group in (
                    model_chunk.bucket_groups + model_chunk.expert_parallel_bucket_groups
                ):
                    if not bucket_group.buckets:
                        continue
                    bucket_group_id = id(bucket_group)
                    if bucket_group_id in deferred_bucket_group_ids:
                        continue
                    deferred_bucket_group_ids.add(bucket_group_id)
                    deferred_bucket_groups.append((model_chunk, bucket_group))

        return deferred_bucket_groups

    def _disable_deferred_mxfp8_param_sync(self) -> None:
        """Disable deferred DistOpt param sync."""
        for optimizer in self.chained_optimizers:
            if hasattr(optimizer, '_defer_param_sync'):
                optimizer._defer_param_sync = False

    def _start_deferred_mxfp8_param_sync(
        self, deferred_bucket_groups: List[Tuple[Any, Any]]
    ) -> None:
        """Start param sync for deferred bucket groups."""
        timers = self.config.timers
        if timers is not None:
            timers('params-all-gather', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        for model_chunk, bucket_group in deferred_bucket_groups:
            model_chunk._start_bucket_group_param_sync(bucket_group, force_sync=False)
        if timers is not None:
            timers('params-all-gather').stop()

    def _step_with_deferred_mxfp8_param_sync(self) -> bool:
        """Step optimizers with MXFP8 param sync deferred until all steps finish."""
        deferred_bucket_groups = self._enable_deferred_mxfp8_param_sync()
        try:
            success = self._step()
        finally:
            self._disable_deferred_mxfp8_param_sync()
        if success and deferred_bucket_groups:
            self._start_deferred_mxfp8_param_sync(deferred_bucket_groups)
        return success

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        """Step the optimizer with ready gradients, return successful."""
        if self._should_defer_mxfp8_param_sync():
            return self._step_with_deferred_mxfp8_param_sync()
        return self._step()

    def grads_states_parallel_group_is_shared(self):
        """Check if all optimizers share the same gradient statistics parallel group."""
        reference_group = self.chained_optimizers[0].get_grad_stats_parallel_group()
        return all(
            optimizer.get_grad_stats_parallel_group() == reference_group
            for optimizer in self.chained_optimizers
        )

    def get_grad_stats_parallel_group(self) -> torch.distributed.ProcessGroup:
        assert self.grads_states_parallel_group_is_shared(), (
            "Can't use get_grad_stats_parallel_group() for ChainedOptimizer, "
            "since grads states parallel group are not shared across all optimizers"
        )
        return self.chained_optimizers[0].get_grad_stats_parallel_group()

    @torch.no_grad()
    def get_grad_norm(self):
        if len(self.chained_optimizers) == 1:
            return self.chained_optimizers[0].get_grad_norm()
        if self.grads_states_parallel_group_is_shared():
            grads_for_norm = []
            for optimizer in self.chained_optimizers:
                grads_for_norm += optimizer.get_grads_for_grad_norm()
            grad_norm = get_grad_norm_fp32(
                grads_for_norm, grad_stats_parallel_group=self.get_grad_stats_parallel_group()
            )
        else:
            grad_norms = []
            for optimizer in self.chained_optimizers:
                _grad_norm = optimizer.get_grad_norm()
                grad_norms += [_grad_norm if _grad_norm else 0.0]
            grad_norm = math.sqrt(sum([x**2 for x in grad_norms]))
        return grad_norm

    @torch.no_grad()
    def count_zeros(self):
        if self.grads_states_parallel_group_is_shared():
            params = []
            for optimizer in self.chained_optimizers:
                params += optimizer.get_parameters()
            return count_zeros_fp32(
                params,
                grad_stats_parallel_group=self.get_grad_stats_parallel_group(),
                use_decoupled_grad=self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
                or (
                    self.config.use_precision_aware_optimizer
                    and getattr(params[0], "__fsdp_param__", False)
                ),
            )
        else:
            num_zeros_in_grad = 0
            for optimizer in self.chained_optimizers:
                num_zeros_in_grad += (
                    optimizer.count_zeros() if optimizer.config.log_num_zeros_in_grad else 0
                )
            return num_zeros_in_grad

    def has_grad_norm_group(self, grad_norm_group: str) -> bool:
        """Whether any chained optimizer owns params for a gradient-norm group."""
        _validate_grad_norm_group(grad_norm_group)
        if getattr(self, '_has_grad_norm_group_cache', None) is None:
            self._has_grad_norm_group_cache = {}
        cache = self._has_grad_norm_group_cache
        if grad_norm_group not in cache:
            cache[grad_norm_group] = any(
                optimizer.has_grad_norm_group(grad_norm_group)
                for optimizer in self.chained_optimizers
            )
        return cache[grad_norm_group]

    @torch.no_grad()
    def _get_grad_norm_for_group(self, grad_norm_group: str):
        """Compute gradient norm for a named parameter group."""
        _validate_grad_norm_group(grad_norm_group)
        if self.grads_states_parallel_group_is_shared():
            grouped_grads = []
            for optimizer in self.chained_optimizers:
                grouped_grads += optimizer.get_grads_for_grad_norm(grad_norm_group)
            return get_grad_norm_fp32(
                grouped_grads, grad_stats_parallel_group=self.get_grad_stats_parallel_group()
            )
        else:
            group_norms = []
            for optimizer in self.chained_optimizers:
                grouped_grads = optimizer.get_grads_for_grad_norm(grad_norm_group)
                norm = get_grad_norm_fp32(
                    grouped_grads,
                    grad_stats_parallel_group=optimizer.get_grad_stats_parallel_group(),
                )
                group_norms.append(norm if norm else 0.0)
            return math.sqrt(sum([x**2 for x in group_norms]))

    @torch.no_grad()
    def _compute_grad_norms_by_group(self) -> Dict[str, float]:
        """Compute gradient norms for registered separate grad-norm groups."""
        self.grad_norms_by_group = {}
        for grad_norm_group in SEPARATE_GRAD_NORM_GROUPS:
            if self.has_grad_norm_group(grad_norm_group):
                group_grad_norm = self._get_grad_norm_for_group(grad_norm_group)
                self.grad_norms_by_group[grad_norm_group] = group_grad_norm
        return self.grad_norms_by_group

    @torch.no_grad()
    def step(self):
        """ChainedOptimizer will step all optimizers one by one."""
        self.grad_norms_by_group = {}
        found_inf_flag = self.prepare_grads()
        if found_inf_flag:
            return False, None, None

        grad_norm = self.get_grad_norm()
        should_skip_update = False

        should_clip = any(
            not (hasattr(optimizer, 'is_stub_optimizer') and optimizer.is_stub_optimizer)
            and optimizer.config.clip_grad > 0.0
            for optimizer in self.chained_optimizers
        )
        if should_clip:
            self._compute_grad_norms_by_group()

        for optimizer in self.chained_optimizers:
            if hasattr(optimizer, 'is_stub_optimizer') and optimizer.is_stub_optimizer:
                continue
            parameters = optimizer.get_parameters()
            if len(parameters) == 0:
                continue

            use_fsdp_decoupled_grad = (
                optimizer.config.use_precision_aware_optimizer
                and getattr(parameters[0], "__fsdp_param__", False)
            )
            use_decoupled_grad = (
                optimizer.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
                or use_fsdp_decoupled_grad
            )

            main_params = []
            params_by_grad_norm_group = {}
            for p in parameters:
                grad_norm_group = _get_param_grad_norm_group(p)
                if _is_separate_grad_norm_group(grad_norm_group):
                    params_by_grad_norm_group.setdefault(grad_norm_group, []).append(p)
                else:
                    main_params.append(p)

            if optimizer.config.clip_grad > 0.0:
                if main_params:
                    clip_grad_by_total_norm_fp32(
                        main_params,
                        max_norm=optimizer.config.clip_grad,
                        total_norm=grad_norm,
                        use_decoupled_grad=use_decoupled_grad,
                    )
                for grad_norm_group, grouped_params in params_by_grad_norm_group.items():
                    group_grad_norm = self.grad_norms_by_group.get(grad_norm_group)
                    if group_grad_norm is None:
                        continue
                    clip_grad_by_total_norm_fp32(
                        grouped_params,
                        max_norm=optimizer.config.clip_grad,
                        total_norm=group_grad_norm,
                        use_decoupled_grad=use_decoupled_grad,
                    )

            if grad_norm > optimizer.config.grad_norm_skip_threshold and main_params:
                _log_single_rank(
                    logger, logging.INFO, "skipping grad norm because it's too large %s", grad_norm
                )
                should_skip_update = True

        num_zeros_in_grad = self.count_zeros() if self.config.log_num_zeros_in_grad else None
        update_successful = False if should_skip_update else self.step_with_ready_grads()

        return update_successful, grad_norm, num_zeros_in_grad

    def save_parameter_state(self, filename: str):
        """Save the distributed parameter states of all optimizers to a file."""
        if len(self.chained_optimizers) == 1:
            self.chained_optimizers[0].save_parameter_state(filename)
            return
        save_states = False
        states = []
        for optimizer in self.chained_optimizers:
            if hasattr(optimizer, 'get_parameter_state_dp_zero'):
                state_dict = optimizer.get_parameter_state_dp_zero()
                if optimizer.data_parallel_group.rank() == 0:
                    states.append(state_dict)
                    save_states = True
                else:
                    assert state_dict is None
                    states.append(None)
        if save_states:
            torch.save(states, filename)

    def load_parameter_state(self, filename: str, *, update_legacy_format: bool = False):
        """Load the distributed parameter states of all optimizers from a file."""
        if len(self.chained_optimizers) == 1:
            self.chained_optimizers[0].load_parameter_state(
                filename, update_legacy_format=update_legacy_format
            )
            return
        states = None
        for idx, optimizer in enumerate(self.chained_optimizers):
            if not hasattr(optimizer, 'load_parameter_state_from_dp_zero'):
                continue
            if optimizer.data_parallel_group.rank() == 0 and states is None:
                states = torch.load(filename)
            state_dict = states[idx] if states else None
            optimizer.load_parameter_state_from_dp_zero(
                state_dict, update_legacy_format=update_legacy_format
            )

    def _synchronize_steps(self):
        """Synchronize the step of all optimizers."""
        steps = []
        for optimizer in self.chained_optimizers:
            for param_group in optimizer.optimizer.param_groups:
                if len(param_group['params']) > 0 and 'step' in param_group:
                    steps.append(param_group['step'])
        steps = list(set(steps))
        assert len(steps) <= 1, f"steps: {steps}"
        step = steps[0] if len(steps) == 1 else None
        for optimizer in self.chained_optimizers:
            for param_group in optimizer.optimizer.param_groups:
                if len(param_group['params']) > 0 and 'step' in param_group:
                    param_group['step'] = step
        return step

    def offload_to_cpu(self):
        """Move optimizer state to CPU to free GPU memory during inference."""
        for optimizer in self.chained_optimizers:
            optimizer.offload_to_cpu()

    def restore_from_cpu(self):
        """Restore optimizer state from CPU back to GPU for training."""
        for optimizer in self.chained_optimizers:
            optimizer.restore_from_cpu()


# ---------------------------------------------------------------------------
# Public re-exports (mirrors what __init__.py expects from this module)
# ---------------------------------------------------------------------------
__all__ = [
    # Base classes
    "MegatronOptimizer",
    "MixedPrecisionOptimizer",
    "Float16OptimizerWithFloat16Params",
    "FP32Optimizer",
    "ChainedOptimizer",
    "ProxyDict",
    # Metadata helpers
    "copy_optimizer_param_metadata",
    "param_is_not_shared",
    # Grad-norm group constants / helpers
    "MTP_GRAD_NORM_GROUP",
    "GRAD_NORM_GROUP_ATTR",
    "SEPARATE_GRAD_NORM_GROUPS",
    "_get_param_grad_norm_group",
    "_is_separate_grad_norm_group",
    "_validate_grad_norm_group",
    # Grad utilities
    "clip_grad_by_total_norm_fp32",
    "count_zeros_fp32",
    "_zero_grad_group_helper",
    "_multi_tensor_copy_this_to_that",
    "param_group_identifier_keys",
]
