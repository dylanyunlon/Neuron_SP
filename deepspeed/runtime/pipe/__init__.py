# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .module import PipelineModule, LayerSpec, TiedLayerSpec
from .topology import ProcessTopology
# DES-LOC M716: BLOOM ALiBi position encoding
from .alibi import ALiBiEmbedding, build_alibi_bias, get_alibi_slopes
# M592: centralized pipeline stall guard (deadlock prevention)
from .pipeline_stall_guard import (
    BLOCKED_BARRIER_CONDITIONS,
    PipelineStallGuard,
    ScheduleParams,
    sanitize_schedule_params,
    sanitize_timer_name,
    should_measure_pipeline_stall,
    validate_schedule_no_deadlock,
    warn_microbatch_underflow,
)
