# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .elasticity import compute_elastic_config, elasticity_enabled, ensure_immutable_elastic_config
from .utils import is_torch_elastic_compatible
from .constants import ENABLED, ENABLED_DEFAULT, ELASTICITY
if is_torch_elastic_compatible():
    from .elastic_agent import DSElasticAgent

# M4188: Flextron hetero-GPU subnet selection (mirrors Megatron 2d862fe0c)
try:
    from .hetero_flextron_config import (
        HeteroFlextronConfig,
        build_hetero_flextron_config,
    )
except ImportError:
    pass
from .flextron_budget import (
    DeslocMemoryProfile,
    count_subnet_params,
    estimate_subnet_memory_gb,
    scan_budget_list_memory,
)
