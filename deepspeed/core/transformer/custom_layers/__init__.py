# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""deepspeed.core.transformer.custom_layers — custom kernel implementations."""

from deepspeed.core.transformer.custom_layers.batch_invariant_kernels import (
    set_batch_invariant_mode,
    is_batch_invariant_mode_enabled,
    disable_batch_invariant_mode,
    enable_batch_invariant_mode,
)

__all__ = [
    "set_batch_invariant_mode",
    "is_batch_invariant_mode_enabled",
    "disable_batch_invariant_mode",
    "enable_batch_invariant_mode",
]
