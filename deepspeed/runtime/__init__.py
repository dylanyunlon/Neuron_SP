# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# M4189: SSM recurrent-state offload for Mamba/Hybrid models (mirrors Megatron 0dc36dfc6)
from .ssm_state_manager import (
    SSMStateManager,
    SSMStateConfig,
    wrap_mamba_model_for_offload,
    attach_ssm_state_manager_to_engine,
)
