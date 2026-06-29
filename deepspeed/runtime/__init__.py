# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# M4189: SSM recurrent-state offload for Mamba/Hybrid models (mirrors Megatron 0dc36dfc6)
# Lazy import: ssm_state_manager uses torch which requires CUDA .so files.
# We expose the names at package level but don't crash if torch isn't loadable.
try:
    from .ssm_state_manager import (
        SSMStateManager,
        SSMStateConfig,
        wrap_mamba_model_for_offload,
        attach_ssm_state_manager_to_engine,
    )
except (ImportError, OSError):
    SSMStateManager = None          # type: ignore[assignment,misc]
    SSMStateConfig = None           # type: ignore[assignment,misc]
    wrap_mamba_model_for_offload = None     # type: ignore[assignment]
    attach_ssm_state_manager_to_engine = None  # type: ignore[assignment]
