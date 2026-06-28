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

    from deepspeed.core.optimizer.optimizer_config import OptimizerConfig
    from deepspeed.core.optimizer.distrib_optimizer import DistributedOptimizer
"""

from deepspeed.core.optimizer.optimizer_config import OptimizerConfig
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
]
