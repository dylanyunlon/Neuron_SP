# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
'''Copyright The Microsoft DeepSpeed Team'''

# mirrors Megatron bb979dd64 — Add vLLM grouped gemm backend for MoE inference,
# reinterpreted as arch-routed GroupedGemmBackendSelector for DES-LOC SM86/SM90.
from .grouped_gemm_backend import (
    GroupedGemmBackend,
    GroupedGemmBackendSelector,
    moe_grouped_gemm_forward,
)
