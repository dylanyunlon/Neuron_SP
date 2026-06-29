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

from deepspeed.core.optimizer.optimizer_config import OptimizerConfig, ParamKey
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
]
