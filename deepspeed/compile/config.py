# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import List, Optional, Literal
from deepspeed.runtime.config_utils import DeepSpeedConfigModel

PassName = Literal["z1", "z3", "autosp"]


class CompileConfig(DeepSpeedConfigModel):

    deepcompile: bool = False
    free_activation: bool = False
    free_activation_threshold: int = 10 * 1024 * 1024
    offload_activation: bool = False
    offload_opt_states: bool = False
    double_buffer: bool = True
    symmetric_memory: bool = False
    debug_log: bool = False
    offload_parameters: bool = False
    sync_before_reduce: bool = False
    sync_after_reduce: bool = False
    sync_before_allgather: bool = False
    sync_after_allgather: bool = False
    keep_int_input_tensors: bool = True
    keep_all_input_tensors: bool = False
    passes: Optional[List[PassName]] = None
