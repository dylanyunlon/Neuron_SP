# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from .v2 import RaggedInferenceEngineConfig, DeepSpeedTPConfig
from .v2.engine_v2 import InferenceEngineV2
from .v2 import build_hf_engine, build_engine_from_ds_checkpoint

# DES-LOC: heterogeneous-device routing layer (mirrors Megatron 3c39d98b5)
from .heterogeneous_engine import (
    HeterogeneousInferenceEngine,
    DeviceTier,
    TierConfig,
    TierSaturatedError,
    _DeviceRouter,
    _TierHandle,
)
