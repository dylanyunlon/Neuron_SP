# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
from typing import Union
from enum import Enum

import torch
import json
import hjson
import copy
import base64

from .constants import *

# DES-LOC configuration keys — Algorithm 1 sync periods
DESLOC = 'desloc'
DESLOC_ENABLED = 'enabled'
DESLOC_ENABLED_DEFAULT = False
DESLOC_KX = 'Kx'
DESLOC_KX_DEFAULT = 1           # Kx=1 = standard DDP (every step sync)
DESLOC_KU = 'Ku'
DESLOC_KU_DEFAULT = 3           # First momentum sync period
DESLOC_KV = 'Kv'
DESLOC_KV_DEFAULT = 6           # Second momentum sync period
DESLOC_CLIP_RHO = 'clip_rho'
DESLOC_CLIP_RHO_DEFAULT = 1.0   # Per-coordinate clipping bound
DESLOC_OUTER_OPT = 'outer_optimizer'
DESLOC_OUTER_OPT_DEFAULT = 'average'  # 'average' or 'nesterov'
DESLOC_INNER_OPT = 'inner_optimizer'
DESLOC_INNER_OPT_DEFAULT = 'adam'     # 'adam', 'adopt', 'muon'
DESLOC_WARMUP = 'warmup_steps'
DESLOC_WARMUP_DEFAULT = 512     # Ref: Section A.1 — TWARM=512
from .config_utils import (
    get_scalar_param,
    dict_raise_error_on_duplicate_keys,
    ScientificNotationEncoder,
)
from .zero.config import get_zero_config, ZeroStageEnum
from .activation_checkpointing.config import DeepSpeedActivationCheckpointingConfig
from ..comm.config import DeepSpeedCommsConfig
from ..monitor.config import get_monitor_config
from ..inference.config import WeightQuantConfig
from .precision_config import get_bfloat16_config, get_float16_config
from ..compile.config import CompileConfig

from deepspeed import comm as dist
from deepspeed.runtime.config_utils import DeepSpeedConfigModel

from ..git_version_info import version as __version__
from ..utils import logger

from ..elasticity import (
    elasticity_enabled,
    compute_elastic_config,
    ensure_immutable_elastic_config,
)
from ..elasticity.config import ElasticityConfigError
from ..elasticity.constants import (
    ELASTICITY,
    IGNORE_NON_ELASTIC_BATCH_INFO,
    IGNORE_NON_ELASTIC_BATCH_INFO_DEFAULT,
    MODEL_PARALLEL_SIZE,
    MODEL_PARALLEL_SIZE_DEFAULT,
    NUM_GPUS_PER_NODE,
    NUM_GPUS_PER_NODE_DEFAULT,
)

from ..profiling.config import DeepSpeedFlopsProfilerConfig
from ..autotuning.config import DeepSpeedAutotuningConfig
from ..nebula.config import DeepSpeedNebulaConfig
from ..datastates.config import DeepSpeedDataStatesConfig

from ..compression.config import get_compression_config, get_quantize_enabled
from ..compression.constants import *
from .swap_tensor.aio_config import get_aio_config
from .model_checkpointing.config import get_checkpoint_config

from .tensor_parallel import get_tensor_parallel_config
from .data_pipeline.config import get_data_efficiency_enabled, get_data_efficiency_config, get_curriculum_enabled_legacy, get_curriculum_params_legacy
from .data_pipeline.constants import *

from ..utils.config import get_timers_config

TENSOR_CORE_ALIGN_SIZE = 8

ADAGRAD_OPTIMIZER = 'adagrad'
ADAM_OPTIMIZER = 'adam'
ADAMW_OPTIMIZER = 'adamw'
LAMB_OPTIMIZER = 'lamb'
ONEBIT_ADAM_OPTIMIZER = 'onebitadam'
ZERO_ONE_ADAM_OPTIMIZER = 'zerooneadam'
ONEBIT_LAMB_OPTIMIZER = 'onebitlamb'
MUADAM_OPTIMIZER = 'muadam'
MUADAMW_OPTIMIZER = 'muadamw'
MUSGD_OPTIMIZER = 'musgd'
LION_OPTIMIZER = 'lion'
MUON_OPTIMIZER = 'muon'

DEEPSPEED_OPTIMIZERS = [
    ADAGRAD_OPTIMIZER, ADAM_OPTIMIZER, ADAMW_OPTIMIZER, LAMB_OPTIMIZER, ONEBIT_ADAM_OPTIMIZER, ONEBIT_LAMB_OPTIMIZER,
    ZERO_ONE_ADAM_OPTIMIZER, MUADAM_OPTIMIZER, MUADAMW_OPTIMIZER, MUSGD_OPTIMIZER, LION_OPTIMIZER, MUON_OPTIMIZER
]

# extra optimizer parameters for adam/adamw
TORCH_ADAM_PARAM = "torch_adam"

# default to adamw logic for adam/adamw optimizers unless user explicitly opts out
ADAM_W_MODE = "adam_w_mode"
ADAM_W_MODE_DEFAULT = True


class DeepSpeedConfigError(Exception):
    pass


class DtypeEnum(Enum):
    # The torch dtype must always be the first value (so we return torch.dtype)
    fp16 = torch.float16, "torch.float16", "fp16", "float16", "half"
    fp32 = torch.float32, "torch.float32", "fp32", "float32", "float"
    int8 = torch.int8, "torch.int8", "int8"
    bf16 = torch.bfloat16, "torch.bfloat16", "bf16", "bfloat16"

    # Copied from https://stackoverflow.com/a/43210118
    # Allows us to use multiple values for each Enum index and returns first
    # listed value when Enum is called
    def __new__(cls, *values):
        obj = object.__new__(cls)
        # first value is canonical value
        obj._value_ = values[0]
        for other_value in values[1:]:
            cls._value2member_map_[other_value] = obj
        obj._all_values = values
        return obj

    def __repr__(self):
        return "<%s.%s: %s>" % (
            self.__class__.__name__,
            self._name_,
            ", ".join([repr(v) for v in self._all_values]),
        )


def get_pld_enabled(param_dict):
    if PROGRESSIVE_LAYER_DROP in param_dict.keys():
        return get_scalar_param(param_dict[PROGRESSIVE_LAYER_DROP], PLD_ENABLED, PLD_ENABLED_DEFAULT)
    else:
        return False


def get_pld_params(param_dict):
    if PROGRESSIVE_LAYER_DROP in param_dict.keys():
        pld_params = copy.copy(param_dict[PROGRESSIVE_LAYER_DROP])
        pld_params.pop(PLD_ENABLED)
        return pld_params
    else:
        return False


def get_amp_enabled(param_dict):
    if AMP in param_dict.keys():
        return get_scalar_param(param_dict[AMP], AMP_ENABLED, AMP_ENABLED_DEFAULT)
    else:
        return False


def get_amp_params(param_dict):
    if AMP in param_dict.keys():
        amp_params = copy.copy(param_dict[AMP])
        amp_params.pop(AMP_ENABLED)
        return amp_params
    else:
        return False


def get_torch_autocast_enabled(param_dict):
    if TORCH_AUTOCAST in param_dict.keys():
        return get_scalar_param(param_dict[TORCH_AUTOCAST], TORCH_AUTOCAST_ENABLED, TORCH_AUTOCAST_ENABLED_DEFAULT)
    else:
        return False


def get_torch_autocast_dtype(param_dict):
    if TORCH_AUTOCAST in param_dict:
        if TORCH_AUTOCAST_DTYPE in param_dict[TORCH_AUTOCAST]:
            try:
                return DtypeEnum(param_dict[TORCH_AUTOCAST][TORCH_AUTOCAST_DTYPE]).value
            except KeyError:
                raise ValueError(
                    f"Invalid dtype for torch autocast: {param_dict[TORCH_AUTOCAST][TORCH_AUTOCAST_DTYPE]}")
    return None


def get_lower_precision_safe_modules(param_dict):
    if TORCH_AUTOCAST in param_dict:
        if TORCH_AUTOCAST_LOWER_PRECISION_SAFE_MODULES in param_dict[TORCH_AUTOCAST]:
            module_names_with_package = param_dict[TORCH_AUTOCAST][TORCH_AUTOCAST_LOWER_PRECISION_SAFE_MODULES]
            if not all(isinstance(module_name, str) for module_name in module_names_with_package):
                raise ValueError(
                    f"Invalid module names for torch autocast: {module_names_with_package}. Expected list of strings.")
            return module_names_with_package
    return None


def get_gradient_accumulation_steps(param_dict):
    return get_scalar_param(param_dict, GRADIENT_ACCUMULATION_STEPS, GRADIENT_ACCUMULATION_STEPS_DEFAULT)


def get_sparse_gradients_enabled(param_dict):
    return get_scalar_param(param_dict, SPARSE_GRADIENTS, SPARSE_GRADIENTS_DEFAULT)


def get_communication_data_type(param_dict,
                                comm_type=COMMUNICATION_DATA_TYPE,
                                comm_data_type_default=COMMUNICATION_DATA_TYPE_DEFAULT):
    val = get_scalar_param(param_dict, comm_type, comm_data_type_default)
    val = val.lower() if val is not None else val
    if val is None:
        return val  # we must determine it by other parameters
    elif val == "fp32":
        return torch.float32
    elif val == "fp16":
        return torch.float16
    elif val == "bf16":
        return torch.bfloat16

    raise ValueError(f"Invalid communication_data_type. Supported data types: ['fp16', 'bf16', 'fp32']. Got: {val}")


def get_prescale_gradients(param_dict):
    return get_scalar_param(param_dict, PRESCALE_GRADIENTS, PRESCALE_GRADIENTS_DEFAULT)


def get_gradient_predivide_factor(param_dict):
    return get_scalar_param(param_dict, GRADIENT_PREDIVIDE_FACTOR, GRADIENT_PREDIVIDE_FACTOR_DEFAULT)


def get_steps_per_print(param_dict):
    return get_scalar_param(param_dict, STEPS_PER_PRINT, STEPS_PER_PRINT_DEFAULT)


def get_disable_allgather(param_dict):
    return get_scalar_param(param_dict, DISABLE_ALLGATHER, DISABLE_ALLGATHER_DEFAULT)


def get_dump_state(param_dict):
    return get_scalar_param(param_dict, DUMP_STATE, DUMP_STATE_DEFAULT)


def get_gradient_clipping(param_dict):
    return get_scalar_param(param_dict, GRADIENT_CLIPPING, GRADIENT_CLIPPING_DEFAULT)


def get_graph_harvesting(param_dict):
    return get_scalar_param(param_dict, GRAPH_HARVESTING, GRAPH_HARVESTING_DEFAULT)


def get_sparse_attention(param_dict):
    if SPARSE_ATTENTION in param_dict.keys():
        sparsity = param_dict[SPARSE_ATTENTION]
        mode = get_sparse_attention_mode(sparsity)

        if mode == SPARSE_DENSE_MODE:
            return get_sparse_dense_config(sparsity)
        elif mode == SPARSE_FIXED_MODE:
            return get_sparse_fixed_config(sparsity)
        elif mode == SPARSE_VARIABLE_MODE:
            return get_sparse_variable_config(sparsity)
        elif mode == SPARSE_BIGBIRD_MODE:
            return get_sparse_bigbird_config(sparsity)
        elif mode == SPARSE_BSLONGFORMER_MODE:
            return get_sparse_bslongformer_config(sparsity)
        else:
            raise NotImplementedError(f"Given sparsity mode, {mode}, has not been implemented yet!")

    else:
        return None


def get_sparse_dense_config(sparsity):
    block = get_scalar_param(sparsity, SPARSE_BLOCK, SPARSE_BLOCK_DEFAULT)
    return {SPARSE_MODE: SPARSE_DENSE_MODE, SPARSE_BLOCK: block}


def get_sparse_fixed_config(sparsity):
    block = get_scalar_param(sparsity, SPARSE_BLOCK, SPARSE_BLOCK_DEFAULT)
    different_layout_per_head = get_scalar_param(
        sparsity,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD_DEFAULT,
    )
    num_local_blocks = get_scalar_param(sparsity, SPARSE_NUM_LOCAL_BLOCKS, SPARSE_NUM_LOCAL_BLOCKS_DEFAULT)
    num_global_blocks = get_scalar_param(sparsity, SPARSE_NUM_GLOBAL_BLOCKS, SPARSE_NUM_GLOBAL_BLOCKS_DEFAULT)
    attention = get_scalar_param(sparsity, SPARSE_ATTENTION_TYPE, SPARSE_ATTENTION_TYPE_DEFAULT)
    horizontal_global_attention = get_scalar_param(
        sparsity,
        SPARSE_HORIZONTAL_GLOBAL_ATTENTION,
        SPARSE_HORIZONTAL_GLOBAL_ATTENTION_DEFAULT,
    )
    num_different_global_patterns = get_scalar_param(
        sparsity,
        SPARSE_NUM_DIFFERENT_GLOBAL_PATTERNS,
        SPARSE_NUM_DIFFERENT_GLOBAL_PATTERNS_DEFAULT,
    )

    return {
        SPARSE_MODE: SPARSE_FIXED_MODE,
        SPARSE_BLOCK: block,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD: different_layout_per_head,
        SPARSE_NUM_LOCAL_BLOCKS: num_local_blocks,
        SPARSE_NUM_GLOBAL_BLOCKS: num_global_blocks,
        SPARSE_ATTENTION_TYPE: attention,
        SPARSE_HORIZONTAL_GLOBAL_ATTENTION: horizontal_global_attention,
        SPARSE_NUM_DIFFERENT_GLOBAL_PATTERNS: num_different_global_patterns,
    }


def get_sparse_variable_config(sparsity):
    block = get_scalar_param(sparsity, SPARSE_BLOCK, SPARSE_BLOCK_DEFAULT)
    different_layout_per_head = get_scalar_param(
        sparsity,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD_DEFAULT,
    )
    num_random_blocks = get_scalar_param(sparsity, SPARSE_NUM_RANDOM_BLOCKS, SPARSE_NUM_RANDOM_BLOCKS_DEFAULT)
    local_window_blocks = get_scalar_param(sparsity, SPARSE_LOCAL_WINDOW_BLOCKS, SPARSE_LOCAL_WINDOW_BLOCKS_DEFAULT)
    global_block_indices = get_scalar_param(sparsity, SPARSE_GLOBAL_BLOCK_INDICES, SPARSE_GLOBAL_BLOCK_INDICES_DEFAULT)
    global_block_end_indices = get_scalar_param(
        sparsity,
        SPARSE_GLOBAL_BLOCK_END_INDICES,
        SPARSE_GLOBAL_BLOCK_END_INDICES_DEFAULT,
    )
    attention = get_scalar_param(sparsity, SPARSE_ATTENTION_TYPE, SPARSE_ATTENTION_TYPE_DEFAULT)
    horizontal_global_attention = get_scalar_param(
        sparsity,
        SPARSE_HORIZONTAL_GLOBAL_ATTENTION,
        SPARSE_HORIZONTAL_GLOBAL_ATTENTION_DEFAULT,
    )

    return {
        SPARSE_MODE: SPARSE_VARIABLE_MODE,
        SPARSE_BLOCK: block,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD: different_layout_per_head,
        SPARSE_NUM_RANDOM_BLOCKS: num_random_blocks,
        SPARSE_LOCAL_WINDOW_BLOCKS: local_window_blocks,
        SPARSE_GLOBAL_BLOCK_INDICES: global_block_indices,
        SPARSE_GLOBAL_BLOCK_END_INDICES: global_block_end_indices,
        SPARSE_ATTENTION_TYPE: attention,
        SPARSE_HORIZONTAL_GLOBAL_ATTENTION: horizontal_global_attention,
    }


def get_sparse_bigbird_config(sparsity):
    block = get_scalar_param(sparsity, SPARSE_BLOCK, SPARSE_BLOCK_DEFAULT)
    different_layout_per_head = get_scalar_param(
        sparsity,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD_DEFAULT,
    )
    num_random_blocks = get_scalar_param(sparsity, SPARSE_NUM_RANDOM_BLOCKS, SPARSE_NUM_RANDOM_BLOCKS_DEFAULT)
    num_sliding_window_blocks = get_scalar_param(
        sparsity,
        SPARSE_NUM_SLIDING_WINDOW_BLOCKS,
        SPARSE_NUM_SLIDING_WINDOW_BLOCKS_DEFAULT,
    )
    num_global_blocks = get_scalar_param(sparsity, SPARSE_NUM_GLOBAL_BLOCKS, SPARSE_NUM_GLOBAL_BLOCKS_DEFAULT)

    return {
        SPARSE_MODE: SPARSE_BIGBIRD_MODE,
        SPARSE_BLOCK: block,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD: different_layout_per_head,
        SPARSE_NUM_RANDOM_BLOCKS: num_random_blocks,
        SPARSE_NUM_SLIDING_WINDOW_BLOCKS: num_sliding_window_blocks,
        SPARSE_NUM_GLOBAL_BLOCKS: num_global_blocks,
    }


def get_sparse_bslongformer_config(sparsity):
    block = get_scalar_param(sparsity, SPARSE_BLOCK, SPARSE_BLOCK_DEFAULT)
    different_layout_per_head = get_scalar_param(
        sparsity,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD_DEFAULT,
    )
    num_sliding_window_blocks = get_scalar_param(
        sparsity,
        SPARSE_NUM_SLIDING_WINDOW_BLOCKS,
        SPARSE_NUM_SLIDING_WINDOW_BLOCKS_DEFAULT,
    )
    global_block_indices = get_scalar_param(sparsity, SPARSE_GLOBAL_BLOCK_INDICES, SPARSE_GLOBAL_BLOCK_INDICES_DEFAULT)
    global_block_end_indices = get_scalar_param(
        sparsity,
        SPARSE_GLOBAL_BLOCK_END_INDICES,
        SPARSE_GLOBAL_BLOCK_END_INDICES_DEFAULT,
    )

    return {
        SPARSE_MODE: SPARSE_BSLONGFORMER_MODE,
        SPARSE_BLOCK: block,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD: different_layout_per_head,
        SPARSE_NUM_SLIDING_WINDOW_BLOCKS: num_sliding_window_blocks,
        SPARSE_GLOBAL_BLOCK_INDICES: global_block_indices,
        SPARSE_GLOBAL_BLOCK_END_INDICES: global_block_end_indices,
    }


def get_sparse_attention_mode(param_dict):
    if SPARSE_MODE in param_dict.keys():
        return param_dict[SPARSE_MODE]
    else:
        return SPARSE_MODE_DEFAULT


def get_sparse_attention_type(param_dict):
    if SPARSE_ATTENTION_TYPE in param_dict.keys():
        return param_dict[SPARSE_ATTENTION_TYPE]
    else:
        return SPARSE_ATTENTION_TYPE_DEFAULT


def get_pipeline_config(param_dict):
    """Parses pipeline engine configuration. """
    default_pipeline = {
        "stages": "auto",
        "partition": "best",
        "seed_layers": False,
        "activation_checkpoint_interval": 0,
        "pipe_partitioned": True,
        "grad_partitioned": True,
    }
    config = default_pipeline
    for key, val in param_dict.get("pipeline", {}).items():
        config[key] = val
    return config


def get_optimizer_name(param_dict):
    if OPTIMIZER in param_dict.keys() and TYPE in param_dict[OPTIMIZER].keys():
        return param_dict[OPTIMIZER][TYPE]
    else:
        return OPTIMIZER_TYPE_DEFAULT


def get_optimizer_params(param_dict):
    if (get_optimizer_name(param_dict) is not None and OPTIMIZER_PARAMS in param_dict[OPTIMIZER].keys()):
        return param_dict[OPTIMIZER][OPTIMIZER_PARAMS]
    else:
        return None


def get_optimizer_gradient_clipping(param_dict):
    optimizer_params = get_optimizer_params(param_dict)
    if optimizer_params is not None and MAX_GRAD_NORM in optimizer_params.keys():
        return optimizer_params[MAX_GRAD_NORM]
    else:
        return None


def get_optimizer_legacy_fusion(param_dict):
    if OPTIMIZER in param_dict.keys() and LEGACY_FUSION in param_dict[OPTIMIZER].keys():
        return param_dict[OPTIMIZER][LEGACY_FUSION]
    else:
        return LEGACY_FUSION_DEFAULT


def get_zero_allow_untested_optimizer(param_dict):
    return get_scalar_param(param_dict, ZERO_ALLOW_UNTESTED_OPTIMIZER, ZERO_ALLOW_UNTESTED_OPTIMIZER_DEFAULT)


def get_zero_force_ds_cpu_optimizer(param_dict):
    return get_scalar_param(param_dict, ZERO_FORCE_DS_CPU_OPTIMIZER, ZERO_FORCE_DS_CPU_OPTIMIZER_DEFAULT)


def get_scheduler_name(param_dict):
    if SCHEDULER in param_dict.keys() and TYPE in param_dict[SCHEDULER].keys():
        return param_dict[SCHEDULER][TYPE]
    else:
        return SCHEDULER_TYPE_DEFAULT


def get_scheduler_params(param_dict):
    if (get_scheduler_name(param_dict) is not None and SCHEDULER_PARAMS in param_dict[SCHEDULER].keys()):
        return param_dict[SCHEDULER][SCHEDULER_PARAMS]
    else:
        return None


def get_train_batch_size(param_dict):
    return get_scalar_param(param_dict, TRAIN_BATCH_SIZE, TRAIN_BATCH_SIZE_DEFAULT)


def get_train_micro_batch_size_per_gpu(param_dict):
    return get_scalar_param(
        param_dict,
        TRAIN_MICRO_BATCH_SIZE_PER_GPU,
        TRAIN_MICRO_BATCH_SIZE_PER_GPU_DEFAULT,
    )


def get_wall_clock_breakdown(param_dict):
    return get_scalar_param(param_dict, WALL_CLOCK_BREAKDOWN, WALL_CLOCK_BREAKDOWN_DEFAULT)


def get_memory_breakdown(param_dict):
    return get_scalar_param(param_dict, MEMORY_BREAKDOWN, MEMORY_BREAKDOWN_DEFAULT)


class HybridEngineConfig(DeepSpeedConfigModel):
    enabled: bool = False
    max_out_tokens: int = 512
    inference_tp_size: int = 1
    release_inference_cache: bool = False
    pin_parameters: bool = True
    tp_gather_partition_size: int = 8


def get_hybrid_engine_config(param_dict):
    hybrid_engine_config_dict = param_dict.get("hybrid_engine", {})
    hybrid_engine_config = HybridEngineConfig(**hybrid_engine_config_dict)
    return hybrid_engine_config


def get_expert_data_topo_config(param_dict):
    return get_scalar_param(param_dict, USE_DATA_BEFORE_EXPERT_PARALLEL, USE_DATA_BEFORE_EXPERT_PARALLEL_DEFAULT)


def get_eigenvalue_config(param_dict):
    if get_quantize_enabled(param_dict):
        param_dict = param_dict[QUANTIZE_TRAINING]
        assert not get_eigenvalue_enabled(param_dict), "Eigenvalue based MoQ is temporarily disabled"
        return (
            get_eigenvalue_enabled(param_dict),
            get_eigenvalue_verbose(param_dict),
            get_eigenvalue_max_iter(param_dict),
            get_eigenvalue_tol(param_dict),
            get_eigenvalue_stability(param_dict),
            get_eigenvalue_gas_boundary_resolution(param_dict),
            get_eigenvalue_layer_name(param_dict),
            get_eigenvalue_layer_num(param_dict),
        )
    else:
        return (
            EIGENVALUE_ENABLED_DEFAULT,
            EIGENVALUE_VERBOSE_DEFAULT,
            EIGENVALUE_MAX_ITER_DEFAULT,
            EIGENVALUE_TOL_DEFAULT,
            EIGENVALUE_STABILITY_DEFAULT,
            EIGENVALUE_GAS_BOUNDARY_RESOLUTION_DEFAULT,
            EIGENVALUE_LAYER_NAME_DEFAULT,
            EIGENVALUE_LAYER_NUM_DEFAULT,
        )


def get_eigenvalue_enabled(param_dict):
    if EIGENVALUE in param_dict.keys():
        return get_scalar_param(param_dict[EIGENVALUE], EIGENVALUE_ENABLED, EIGENVALUE_ENABLED_DEFAULT)
    else:
        return EIGENVALUE_ENABLED_DEFAULT


def get_eigenvalue_verbose(param_dict):
    if EIGENVALUE in param_dict.keys():
        return get_scalar_param(param_dict[EIGENVALUE], EIGENVALUE_VERBOSE, EIGENVALUE_VERBOSE_DEFAULT)
    else:
        return EIGENVALUE_VERBOSE_DEFAULT


def get_eigenvalue_max_iter(param_dict):
    if EIGENVALUE in param_dict.keys():
        return get_scalar_param(param_dict[EIGENVALUE], EIGENVALUE_MAX_ITER, EIGENVALUE_MAX_ITER_DEFAULT)
    else:
        return EIGENVALUE_MAX_ITER_DEFAULT


def get_eigenvalue_tol(param_dict):
    if EIGENVALUE in param_dict.keys():
        return get_scalar_param(param_dict[EIGENVALUE], EIGENVALUE_TOL, EIGENVALUE_TOL_DEFAULT)
    else:
        return EIGENVALUE_TOL_DEFAULT


def get_eigenvalue_stability(param_dict):
    if EIGENVALUE in param_dict.keys():
        return get_scalar_param(param_dict[EIGENVALUE], EIGENVALUE_STABILITY, EIGENVALUE_STABILITY_DEFAULT)
    else:
        return EIGENVALUE_STABILITY_DEFAULT


def get_eigenvalue_gas_boundary_resolution(param_dict):
    if EIGENVALUE in param_dict.keys():
        return get_scalar_param(
            param_dict[EIGENVALUE],
            EIGENVALUE_GAS_BOUNDARY_RESOLUTION,
            EIGENVALUE_GAS_BOUNDARY_RESOLUTION_DEFAULT,
        )
    else:
        return EIGENVALUE_GAS_BOUNDARY_RESOLUTION_DEFAULT


def get_eigenvalue_layer_name(param_dict):
    if EIGENVALUE in param_dict.keys():
        return get_scalar_param(param_dict[EIGENVALUE], EIGENVALUE_LAYER_NAME, EIGENVALUE_LAYER_NAME_DEFAULT)
    else:
        return EIGENVALUE_LAYER_NAME_DEFAULT


def get_eigenvalue_layer_num(param_dict):
    if EIGENVALUE in param_dict.keys():
        return get_scalar_param(param_dict[EIGENVALUE], EIGENVALUE_LAYER_NUM, EIGENVALUE_LAYER_NUM_DEFAULT)
    else:
        return EIGENVALUE_LAYER_NUM_DEFAULT


def get_checkpoint_params(param_dict):
    return param_dict.get(CHECKPOINT, {})


def get_data_types_params(param_dict):
    return param_dict.get(DATA_TYPES, {})


def get_checkpoint_tag_validation_mode(checkpoint_params):
    tag_validation_mode = checkpoint_params.get(CHECKPOINT_TAG_VALIDATION, CHECKPOINT_TAG_VALIDATION_DEFAULT)
    tag_validation_mode = tag_validation_mode.upper()
    if tag_validation_mode in CHECKPOINT_TAG_VALIDATION_MODES:
        return tag_validation_mode
    else:
        raise DeepSpeedConfigError(
            "Checkpoint config contains invalid tag_validation "
            f"value of {tag_validation_mode}, expecting one of {CHECKPOINT_TAG_VALIDATION_MODES}")


def get_checkpoint_parallel_write_pipeline(checkpoint_params):
    par_write_params = checkpoint_params.get(CHECKPOINT_PARALLEL_WRITE, {})
    par_write_pipeline = par_write_params.get(CHECKPOINT_PARALLEL_WRITE_PIPELINE_STAGE,
                                              CHECKPOINT_PARALLEL_WRITE_PIPELINE_STAGE_DEFAULT)
    if par_write_pipeline in [True, False]:
        return par_write_pipeline
    else:
        raise DeepSpeedConfigError("checkpoint::parallel_write::pipeline_stage "
                                   f"value of '{par_write_pipeline}' is invalid, expecting: true or false")


def get_dataloader_drop_last(param_dict):
    return get_scalar_param(param_dict, DATALOADER_DROP_LAST, DATALOADER_DROP_LAST_DEFAULT)


'''Write deepspeed config files by modifying basic templates.
Can be used for quickly changing parameters via command line parameters.'''


class DeepSpeedConfigWriter:

    def __init__(self, data=None):
        self.data = data if data is not None else {}

    def add_config(self, key, value):
        self.data[key] = value

    def load_config(self, filename):
        self.data = json.load(open(filename, "r"), object_pairs_hook=dict_raise_error_on_duplicate_keys)

    def write_config(self, filename):
        with open(filename, "w") as outfile:
            json.dump(self.data, outfile)


class DeepSpeedConfig(object):

    def __init__(self, config: Union[str, dict], mpu=None, mesh_device=None):
        super(DeepSpeedConfig, self).__init__()
        if isinstance(config, dict):
            self._param_dict = config
        elif os.path.exists(config):
            self._param_dict = hjson.load(open(config, "r"), object_pairs_hook=dict_raise_error_on_duplicate_keys)
        else:
            try:
                config_decoded = base64.urlsafe_b64decode(config).decode('utf-8')
                self._param_dict = hjson.loads(config_decoded)
            except (UnicodeDecodeError, AttributeError):
                raise ValueError(
                    f"Expected a string path to an existing deepspeed config, or a dictionary or a valid base64. Received: {config}"
                )

        try:
            self.global_rank = dist.get_rank()
            if mpu is not None:
                # Ulysses SP
                if not hasattr(mpu, "get_data_parallel_world_size"):
                    self.world_size = dist.get_world_size() / mpu.get_sequence_parallel_world_size()
                else:
                    self.world_size = mpu.get_data_parallel_world_size()
            elif mesh_device is not None:
                self.world_size = dist.get_world_size(mesh_device.get_group(mesh_dim="data_parallel"))
            else:
                # HF zero.init case where there is no mpu
                if "sequence_parallel_size" in config:
                    self.world_size = dist.get_world_size() / config["sequence_parallel_size"]
                else:
                    self.world_size = dist.get_world_size()
        except Exception:
            self.global_rank = 0
            self.world_size = 1
        logger.info(f"Config mesh_device {mesh_device} world_size = {self.world_size}")
        # If elastic-mode enabled, update compute + update _param_dict
        self.elasticity_enabled = elasticity_enabled(self._param_dict)
        if self.elasticity_enabled:
            logger.info("DeepSpeed elasticity support enabled")
            final_batch_size, valid_gpus, micro_batch_size = compute_elastic_config(
                ds_config=self._param_dict,
                target_deepspeed_version=__version__,
                world_size=self.world_size,
            )

            elastic_dict = self._param_dict[ELASTICITY]

            # Ensure the resource scheduler saw the same elastic config we are using at runtime
            ensure_immutable_elastic_config(runtime_elastic_config_dict=elastic_dict)

            self.elastic_model_parallel_size = elastic_dict.get(MODEL_PARALLEL_SIZE, MODEL_PARALLEL_SIZE_DEFAULT)
            if self.elastic_model_parallel_size < 1:
                raise ElasticityConfigError("Model-Parallel size cannot be less than 1, "
                                            f"given model-parallel size: {self.elastic_model_parallel_size}")

            self.num_gpus_per_node = elastic_dict.get(NUM_GPUS_PER_NODE, NUM_GPUS_PER_NODE_DEFAULT)
            if self.num_gpus_per_node < 1:
                raise ElasticityConfigError("NUmber of GPUs per node cannot be less than 1, "
                                            f"given number of GPUs per node: {self.num_gpus_per_node}")

            ignore_non_elastic_batch_info = elastic_dict.get(IGNORE_NON_ELASTIC_BATCH_INFO,
                                                             IGNORE_NON_ELASTIC_BATCH_INFO_DEFAULT)

            if not ignore_non_elastic_batch_info:
                batch_params = [
                    TRAIN_BATCH_SIZE,
                    TRAIN_MICRO_BATCH_SIZE_PER_GPU,
                    GRADIENT_ACCUMULATION_STEPS,
                ]
                if any(map(lambda t: t in self._param_dict, batch_params)):
                    raise ElasticityConfigError("One or more batch related parameters were found in your " \
                        f"ds_config ({TRAIN_BATCH_SIZE}, {TRAIN_MICRO_BATCH_SIZE_PER_GPU}, and/or " \
                        f"{GRADIENT_ACCUMULATION_STEPS}). These parameters *will not be used* since " \
                        "elastic training is enabled, which takes control of these parameters. " \
                        "If you want to suppress this error (the parameters will be silently ignored) " \
                        f"please set {IGNORE_NON_ELASTIC_BATCH_INFO}':true in your elasticity config.")

            # micro_bsz * world_size * gas = total_batch_size
            # gas = total_batch_size // (micro_bsz * world_size)
            gradient_accu_steps = final_batch_size // (micro_batch_size * self.world_size)

            if TRAIN_BATCH_SIZE in self._param_dict:
                logger.warning("[Elasticity] overriding training_batch_size: "
                               f"{self._param_dict[TRAIN_BATCH_SIZE]} -> {final_batch_size}")
            if TRAIN_MICRO_BATCH_SIZE_PER_GPU in self._param_dict:
                logger.warning("[Elasticity] overriding train_micro_batch_size_per_gpu: "
                               f"{self._param_dict[TRAIN_MICRO_BATCH_SIZE_PER_GPU]} -> {micro_batch_size}")
            if GRADIENT_ACCUMULATION_STEPS in self._param_dict:
                logger.warning("[Elasticity] overriding gradient_accumulation_steps: "
                               f"{self._param_dict[GRADIENT_ACCUMULATION_STEPS]} -> {gradient_accu_steps}")

            logger.info(f"[Elasticity] valid GPU counts: {valid_gpus}")

            self._param_dict[TRAIN_BATCH_SIZE] = final_batch_size
            self._param_dict[TRAIN_MICRO_BATCH_SIZE_PER_GPU] = micro_batch_size
            self._param_dict[GRADIENT_ACCUMULATION_STEPS] = gradient_accu_steps

        # Pass a copy so that user json is unmodified, e.g. for logging
        self._initialize_params(copy.copy(self._param_dict))
        self._configure_train_batch_size()
        self._do_sanity_check()

    def _initialize_params(self, param_dict):
        self.train_batch_size = get_train_batch_size(param_dict)
        self.train_micro_batch_size_per_gpu = get_train_micro_batch_size_per_gpu(param_dict)
        self.gradient_accumulation_steps = get_gradient_accumulation_steps(param_dict)
        self.steps_per_print = get_steps_per_print(param_dict)
        self.dump_state = get_dump_state(param_dict)

        self.disable_allgather = get_disable_allgather(param_dict)
        self.communication_data_type = get_communication_data_type(param_dict)
        self.seq_parallel_communication_data_type = get_communication_data_type(
            param_dict, SEQ_PARALLEL_COMMUNICATION_DATA_TYPE, SEQ_PARALLEL_COMMUNICATION_DATA_TYPE_DEFAULT)
        self.prescale_gradients = get_prescale_gradients(param_dict)
        self.gradient_predivide_factor = get_gradient_predivide_factor(param_dict)
        self.sparse_gradients_enabled = get_sparse_gradients_enabled(param_dict)

        self.zero_config = get_zero_config(param_dict)
        self.mics_shard_size = self.zero_config.mics_shard_size
        self.mics_hierarchial_params_gather = self.zero_config.mics_hierarchical_params_gather
        self.zero_optimization_stage = self.zero_config.stage
        self.zero_enabled = self.zero_optimization_stage > 0

        self.activation_checkpointing_config = DeepSpeedActivationCheckpointingConfig(param_dict)

        self.comms_config = DeepSpeedCommsConfig(param_dict)
        self.monitor_config = get_monitor_config(param_dict)

        # DES-LOC: parse independent sync period configuration
        # Ref: Algorithm 1 line 7 — Kx for params, Ku for m1, Kv for m2
        desloc_dict = param_dict.get(DESLOC, {})
        self.desloc_enabled = desloc_dict.get(DESLOC_ENABLED, DESLOC_ENABLED_DEFAULT)
        self.desloc_Kx = desloc_dict.get(DESLOC_KX, DESLOC_KX_DEFAULT)
        self.desloc_Ku = desloc_dict.get(DESLOC_KU, DESLOC_KU_DEFAULT)
        self.desloc_Kv = desloc_dict.get(DESLOC_KV, DESLOC_KV_DEFAULT)
        self.desloc_clip_rho = desloc_dict.get(DESLOC_CLIP_RHO, DESLOC_CLIP_RHO_DEFAULT)
        self.desloc_outer_opt = desloc_dict.get(DESLOC_OUTER_OPT, DESLOC_OUTER_OPT_DEFAULT)
        self.desloc_inner_opt = desloc_dict.get(DESLOC_INNER_OPT, DESLOC_INNER_OPT_DEFAULT)
        self.desloc_warmup = desloc_dict.get(DESLOC_WARMUP, DESLOC_WARMUP_DEFAULT)

        self.gradient_clipping = get_gradient_clipping(param_dict)
        self.float16_config = get_float16_config(param_dict)
        self.bfloat16_config = get_bfloat16_config(param_dict)
        assert not (self.float16_config.enabled
                    and self.bfloat16_config.enabled), 'bfloat16 and fp16 modes cannot be simultaneously enabled'

        self.amp_enabled = get_amp_enabled(param_dict)
        self.amp_params = get_amp_params(param_dict)

        self.torch_autocast_enabled = get_torch_autocast_enabled(param_dict)
        self.torch_autocast_dtype = get_torch_autocast_dtype(param_dict)
        self.torch_autocast_lower_precision_safe_modules = get_lower_precision_safe_modules(param_dict)

        self.compression_config = get_compression_config(param_dict)
        self.graph_harvesting = get_graph_harvesting(param_dict)

        self.optimizer_name = get_optimizer_name(param_dict)
        if (self.optimizer_name is not None and self.optimizer_name.lower() in DEEPSPEED_OPTIMIZERS):
            self.optimizer_name = self.optimizer_name.lower()

        self.optimizer_params = get_optimizer_params(param_dict)
        self.optimizer_legacy_fusion = get_optimizer_legacy_fusion(param_dict)

        self.zero_allow_untested_optimizer = get_zero_allow_untested_optimizer(param_dict)

        self.zero_force_ds_cpu_optimizer = get_zero_force_ds_cpu_optimizer(param_dict)

        self.scheduler_name = get_scheduler_name(param_dict)
        self.scheduler_params = get_scheduler_params(param_dict)

        self.flops_profiler_config = DeepSpeedFlopsProfilerConfig(param_dict)
        self.wall_clock_breakdown = (get_wall_clock_breakdown(param_dict) | self.flops_profiler_config.enabled)
        self.memory_breakdown = get_memory_breakdown(param_dict)
        self.autotuning_config = DeepSpeedAutotuningConfig(param_dict)

        (
            self.eigenvalue_enabled,
            self.eigenvalue_verbose,
            self.eigenvalue_max_iter,
            self.eigenvalue_tol,
            self.eigenvalue_stability,
            self.eigenvalue_gas_boundary_resolution,
            self.eigenvalue_layer_name,
            self.eigenvalue_layer_num,
        ) = get_eigenvalue_config(param_dict)

        self.use_data_before_expert_parallel_ = get_expert_data_topo_config(param_dict)
        self.hybrid_engine = get_hybrid_engine_config(param_dict)

        self.sparse_attention = get_sparse_attention(param_dict)
        self.pipeline = get_pipeline_config(param_dict)

        self.pld_enabled = get_pld_enabled(param_dict)
        self.pld_params = get_pld_params(param_dict)

        self.curriculum_enabled_legacy = get_curriculum_enabled_legacy(param_dict)
        self.curriculum_params_legacy = get_curriculum_params_legacy(param_dict)

        self.data_efficiency_enabled = get_data_efficiency_enabled(param_dict)
        self.data_efficiency_config = get_data_efficiency_config(param_dict)

        checkpoint_params = get_checkpoint_params(param_dict)
        validation_mode = get_checkpoint_tag_validation_mode(checkpoint_params)
        self.checkpoint_tag_validation_enabled = (validation_mode != ValidationMode.IGNORE)
        self.checkpoint_tag_validation_fail = validation_mode == ValidationMode.FAIL
        self.load_universal_checkpoint = checkpoint_params.get(LOAD_UNIVERSAL_CHECKPOINT,
                                                               LOAD_UNIVERSAL_CHECKPOINT_DEFAULT)

        self.use_node_local_storage = checkpoint_params.get(USE_NODE_LOCAL_STORAGE_CHECKPOINT,
                                                            USE_NODE_LOCAL_STORAGE_CHECKPOINT_DEFAULT)

        data_types_params = get_data_types_params(param_dict)
        self.grad_accum_dtype = data_types_params.get(GRAD_ACCUM_DTYPE, GRAD_ACCUM_DTYPE_DEFAULT)

        par_write_pipe = get_checkpoint_parallel_write_pipeline(checkpoint_params)
        self.checkpoint_parallel_write_pipeline = par_write_pipe

        self.aio_config = get_aio_config(param_dict)

        self.dataloader_drop_last = get_dataloader_drop_last(param_dict)

        self.nebula_config = DeepSpeedNebulaConfig(param_dict)
        self.datastates_config = DeepSpeedDataStatesConfig(param_dict)
        self.checkpoint_config = get_checkpoint_config(param_dict)

        self.weight_quantization_config = WeightQuantConfig(
            **param_dict['weight_quantization']) if 'weight_quantization' in param_dict else None

        self.compile_config = CompileConfig(**param_dict.get('compile', {}))

        self.timers_config = get_timers_config(param_dict)
        self.tensor_parallel_config = get_tensor_parallel_config(param_dict)

    def _batch_assertion(self):

        train_batch = self.train_batch_size
        micro_batch = self.train_micro_batch_size_per_gpu
        grad_acc = self.gradient_accumulation_steps

        assert (train_batch > 0), f"Train batch size: {train_batch} has to be greater than 0"

        assert (micro_batch > 0), f"Micro batch size per gpu: {micro_batch} has to be greater than 0"

        assert (grad_acc > 0), f"Gradient accumulation steps: {grad_acc} has to be greater than 0"

        assert train_batch == micro_batch * grad_acc * self.world_size, (
            f"Check batch related parameters. train_batch_size is not equal "
            "to micro_batch_per_gpu * gradient_acc_step * world_size "
            f"{train_batch} != {micro_batch} * {grad_acc} * {self.world_size}")

    def _set_batch_related_parameters(self):

        train_batch = self.train_batch_size
        micro_batch = self.train_micro_batch_size_per_gpu
        grad_acc = self.gradient_accumulation_steps

        #print(f"in: train_batch = {train_batch}, micro_batch={micro_batch}")

        # all values are provided nothing needs to be set
        if train_batch is not None and micro_batch is not None and grad_acc is not None:
            return

        # global_accumulation_steps needs to be set
        elif train_batch is not None and micro_batch is not None:
            grad_acc = train_batch // micro_batch
            grad_acc //= self.world_size
            self.gradient_accumulation_steps = grad_acc

        # micro_batch_per_gpu needs to be set
        elif train_batch is not None and grad_acc is not None:
            micro_batch = train_batch // self.world_size
            micro_batch //= grad_acc
            self.train_micro_batch_size_per_gpu = micro_batch

        # train_batch_size needs to be set
        elif micro_batch is not None and grad_acc is not None:
            train_batch_size = micro_batch * grad_acc
            train_batch_size *= self.world_size
            self.train_batch_size = train_batch_size

        # gradient_accumulation_steps and micro_batch_per_gpus is set
        elif train_batch is not None:
            self.gradient_accumulation_steps = 1
            self.train_micro_batch_size_per_gpu = train_batch // self.world_size

        # train_batch_size and gradient_accumulation_step is set
        elif micro_batch is not None:
            self.train_batch_size = micro_batch * self.world_size
            self.gradient_accumulation_steps = 1

        # either none of the three parameters are provided or just gradient_accumulation_step is provided
        else:
            assert False, \
                'Either train_batch_size or train_micro_batch_size_per_gpu needs to be provided'

        #print(f"final: {self.train_batch_size=} {self.train_micro_batch_size_per_gpu=} {self.gradient_accumulation_steps=}")

    def _configure_train_batch_size(self):
        self._set_batch_related_parameters()
        self._batch_assertion()

    def _do_sanity_check(self):
        self._do_error_check()

        self._do_warning_check()

    def print_user_config(self):
        logger.info("  json = {}".format(
            json.dumps(
                self._param_dict,
                sort_keys=True,
                indent=4,
                cls=ScientificNotationEncoder,
                separators=(",", ":"),
            )))

    def print(self, name):
        logger.info("{}:".format(name))
        for arg in sorted(vars(self)):
            if arg != "_param_dict":
                dots = "." * (29 - len(arg))
                logger.info("  {} {} {}".format(arg, dots, getattr(self, arg)))

        self.print_user_config()

    def _do_error_check(self):
        assert (self.train_micro_batch_size_per_gpu
                ), "DeepSpeedConfig: {} is not defined".format(TRAIN_MICRO_BATCH_SIZE_PER_GPU)

        assert (
            self.gradient_accumulation_steps), "DeepSpeedConfig: {} is not defined".format(GRADIENT_ACCUMULATION_STEPS)

        if self.zero_enabled:
            assert (self.zero_optimization_stage
                    <= ZeroStageEnum.max_stage), "DeepSpeedConfig: Maximum supported ZeRO stage is {}".format(
                        ZeroStageEnum.max_stage)

        if self.float16_config.fp16_master_weights_and_grads:
            assert self.zero_enabled and self.zero_optimization_stage in (
                ZeroStageEnum.optimizer_states, ZeroStageEnum.gradients,
                ZeroStageEnum.weights), "Fp16_master_weights_and_grads is only supported with ZeRO Stage 1, 2, or 3."
        if self.bfloat16_config.bf16_master_weights_and_grads:
            assert self.zero_enabled and self.zero_optimization_stage in (
                ZeroStageEnum.optimizer_states, ZeroStageEnum.gradients,
                ZeroStageEnum.weights), "Bf16_master_weights_and_grads is only supported with ZeRO Stage 1, 2, or 3."
        if self.bfloat16_config.bf16_optimizer_states:
            assert self.zero_enabled and self.zero_optimization_stage in (
                ZeroStageEnum.optimizer_states, ZeroStageEnum.gradients,
                ZeroStageEnum.weights), "bf16_optimizer_states is only supported with ZeRO Stage 1, 2, or 3."
            assert self.bfloat16_config.bf16_master_weights_and_grads, "bf16_optimizer_states requires bf16_master_weights_and_grads to be enabled."

    def _do_warning_check(self):
        fp16_enabled = self.float16_config.enabled

        vocabulary_size = self._param_dict.get(VOCABULARY_SIZE, VOCABULARY_SIZE_DEFAULT)
        if vocabulary_size and vocabulary_size % TENSOR_CORE_ALIGN_SIZE != 0:
            logger.warning(
                "DeepSpeedConfig: vocabulary size {} is not aligned to {}, may import tensor core utilization.".format(
                    vocabulary_size, TENSOR_CORE_ALIGN_SIZE))

        if (self.optimizer_params is not None and MAX_GRAD_NORM in self.optimizer_params.keys()
                and self.optimizer_params[MAX_GRAD_NORM] > 0):
            if fp16_enabled:
                if self.global_rank == 0:
                    logger.warning("DeepSpeedConfig: In FP16 mode, DeepSpeed will pass {}:{} to FP16 wrapper".format(
                        MAX_GRAD_NORM, self.optimizer_params[MAX_GRAD_NORM]))
            else:
                if self.global_rank == 0:
                    logger.warning(
                        "DeepSpeedConfig: In FP32 mode, DeepSpeed does not permit MAX_GRAD_NORM ({}) > 0, setting to zero"
                        .format(self.optimizer_params[MAX_GRAD_NORM]))
                self.optimizer_params[MAX_GRAD_NORM] = 0.0


# =========================================================================
# DES-LOC Configuration Class
# Ref: Algorithm 1 + Section 4.1 + Theorem 1
# =========================================================================

class DeslocConfig:
    """DES-LOC configuration with validation and diagnostics.

    Parameters:
        Kx: Parameter sync period (must be >= 1). Kx=1 = DDP baseline.
        Ku: First momentum sync period (should be >= Kx).
        Kv: Second momentum sync period (should be >= Ku).
        clip_rho: Per-coordinate gradient clipping bound (> 0).
        outer_optimizer: 'average' or 'nesterov' (Section 5.5).
        inner_optimizer: 'adam', 'adopt', or 'muon' (Section 5.6).
        warmup_steps: Steps with Kx=1 before DES-LOC activates.

    Ref: Section 5.3 — Ku=3*Kx, Kv=6*Kx as default heuristic.
    """

    VALID_OUTER = ('average', 'nesterov')
    VALID_INNER = ('adam', 'adopt', 'muon', 'sgdm')

    def __init__(self, enabled=False, Kx=1, Ku=3, Kv=6, clip_rho=1.0,
                 outer_optimizer='average', inner_optimizer='adam',
                 warmup_steps=512, beta1=0.9, beta2=0.999):
        self.enabled = enabled
        self.Kx = max(1, int(Kx))
        self.Ku = max(1, int(Ku))
        self.Kv = max(1, int(Kv))
        self.clip_rho = float(clip_rho)
        self.outer_optimizer = outer_optimizer
        self.inner_optimizer = inner_optimizer
        self.warmup_steps = max(0, int(warmup_steps))
        self.beta1 = beta1
        self.beta2 = beta2

    def validate(self):
        """Validate configuration. Returns list of issue strings."""
        import math
        issues = []
        if self.Kx < 1:
            issues.append(f'ERROR: Kx={self.Kx} must be >= 1')
        if self.Ku < self.Kx:
            issues.append(f'WARNING: Ku={self.Ku} < Kx={self.Kx}')
        if self.Kv < self.Ku:
            issues.append(f'WARNING: Kv={self.Kv} < Ku={self.Ku}')
        if self.clip_rho <= 0:
            issues.append(f'ERROR: clip_rho={self.clip_rho} must be > 0')
        if self.outer_optimizer not in self.VALID_OUTER:
            issues.append(f'ERROR: outer_optimizer={self.outer_optimizer}')
        if self.inner_optimizer not in self.VALID_INNER:
            issues.append(f'ERROR: inner_optimizer={self.inner_optimizer}')
        # Half-life alignment check
        if 0 < self.beta1 < 1 and 0 < self.beta2 < 1:
            tau1 = -1.0 / math.log(self.beta1)
            tau2 = -1.0 / math.log(self.beta2)
            ratio = tau2 / tau1
            if self.Kv / max(1, self.Kx) < ratio * 0.5:
                issues.append(
                    f'WARNING: Kv/Kx={self.Kv/self.Kx:.1f} << '
                    f'tau2/tau1={ratio:.1f}; consider Kv={int(self.Kx*ratio)}')
        if self.warmup_steps > 0 and self.warmup_steps % self.Kx != 0:
            issues.append(f'WARNING: warmup_steps={self.warmup_steps} '
                         f'not multiple of Kx={self.Kx}')
        return issues

    def to_dict(self):
        return {
            DESLOC_ENABLED: self.enabled,
            DESLOC_KX: self.Kx, DESLOC_KU: self.Ku, DESLOC_KV: self.Kv,
            DESLOC_CLIP_RHO: self.clip_rho,
            DESLOC_OUTER_OPT: self.outer_optimizer,
            DESLOC_INNER_OPT: self.inner_optimizer,
            DESLOC_WARMUP: self.warmup_steps,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            enabled=d.get(DESLOC_ENABLED, False),
            Kx=d.get(DESLOC_KX, 1), Ku=d.get(DESLOC_KU, 3), Kv=d.get(DESLOC_KV, 6),
            clip_rho=d.get(DESLOC_CLIP_RHO, 1.0),
            outer_optimizer=d.get(DESLOC_OUTER_OPT, 'average'),
            inner_optimizer=d.get(DESLOC_INNER_OPT, 'adam'),
            warmup_steps=d.get(DESLOC_WARMUP, 512),
        )

    def summary(self):
        import math
        if not self.enabled:
            return 'DES-LOC: disabled (standard DDP)'
        tau1 = -1.0/math.log(self.beta1) if 0<self.beta1<1 else 0
        tau2 = -1.0/math.log(self.beta2) if 0<self.beta2<1 else 0
        return (f'DES-LOC: Kx={self.Kx} Ku={self.Ku} Kv={self.Kv} '
                f'rho={self.clip_rho} outer={self.outer_optimizer} '
                f'inner={self.inner_optimizer} warmup={self.warmup_steps} '
                f'tau1={tau1:.1f} tau2={tau2:.1f}')


# Experiment matrix for NeurIPS ablation
DESLOC_ABLATION = {
    'kx_sweep': [1, 2, 4, 8, 16, 32, 64, 128],
    'beta2_sweep': [0.9, 0.95, 0.99, 0.999],
    'model_sweep': ['125M', '350M', '1B'],
    'outer_sweep': ['average', 'nesterov'],
    'inner_sweep': ['adam', 'adopt'],
}


def desloc_generate_ablation(seeds=(42, 137, 2024)):
    """Generate complete experiment list. Ref: Section 5 RQ1-RQ6.
    Total: 7 Kx x 3 beta2 x 2 models x 3 seeds = 126 base.
    Plus: 2 outer x 3 seeds + 2 inner x 3 seeds = 12.
    Grand total: 138 experiments."""
    exps = []
    for kx in [1, 4, 8, 16, 32, 64, 128]:
        for b2 in [0.95, 0.99, 0.999]:
            for model in ['125M', '350M']:
                for seed in seeds:
                    ku, kv = max(1, kx*3), max(1, kx*6)
                    exps.append({'Kx': kx, 'Ku': ku, 'Kv': kv,
                                'beta2': b2, 'model': model, 'seed': seed,
                                'rq': 'RQ1-4'})
    for outer in ['average', 'nesterov']:
        for seed in seeds:
            exps.append({'Kx': 32, 'Ku': 96, 'Kv': 192,
                        'outer': outer, 'model': '125M', 'seed': seed, 'rq': 'RQ5'})
    for inner in ['adam', 'adopt']:
        for seed in seeds:
            exps.append({'Kx': 32, 'Ku': 96, 'Kv': 192,
                        'inner': inner, 'model': '125M', 'seed': seed, 'rq': 'RQ6'})
    return exps


def desloc_validate_precision(results):
    """Check all numeric values have >= 4 significant digits.
    Ref: NeurIPS standard — reviewers reject 1, 11, 0.9."""
    violations = []
    def walk(d, p=''):
        if isinstance(d, dict):
            for k, v in d.items():
                walk(v, f'{p}.{k}' if p else k)
        elif isinstance(d, (list, tuple)):
            for i, v in enumerate(d):
                walk(v, f'{p}[{i}]')
        elif isinstance(d, float) and d != 0:
            s = f'{d:.10g}'.lstrip('-0').replace('.', '')
            if len(s.rstrip('0')) < 4:
                violations.append(p)
    walk(results)
    return violations


DESLOC_FIGURE_SPECS = {
    'fig1': {'title': 'RQ1: Half-Life Validation', 'x': 'beta2', 'y': 'change_rate_ratio'},
    'fig2': {'title': 'RQ2: Independent Sync Periods', 'x': 'Kx', 'y': 'final_loss'},
    'fig3': {'title': 'RQ3: Comm Reduction', 'x': 'method', 'y': 'comm_bytes'},
    'fig4': {'title': 'RQ4: Large-Scale Loss Curves', 'x': 'step', 'y': 'loss'},
    'fig5': {'title': 'RQ5: Nesterov Outer Optimizer', 'x': 'step', 'y': 'loss'},
    'fig6': {'title': 'RQ6: Muon Compatibility', 'x': 'model', 'y': 'final_loss'},
}


class DeslocOuterOptimizerConfig:
    """Configuration for the DES-LOC outer optimizer.
    Ref: Section 5.5 — Nesterov outer optimizer improves over averaging.

    At each Kx boundary, workers average parameters. The outer optimizer
    controls HOW: simple averaging or Nesterov momentum on the delta.

    Nesterov update at sync boundary:
      v_{t+1} = beta_outer * v_t + (x_avg - x_local)
      x_{t+1} = x_avg + beta_outer * v_{t+1}

    This reduces noise from local optimization between sync points.
    """

    def __init__(self, mode='average', beta_outer=0.9, outer_lr=1.0):
        self.mode = mode
        self.beta_outer = beta_outer
        self.outer_lr = outer_lr

    def validate(self):
        issues = []
        if self.mode not in ('average', 'nesterov'):
            issues.append(f'ERROR: mode={self.mode} not in (average, nesterov)')
        if not (0 < self.beta_outer < 1):
            issues.append(f'ERROR: beta_outer={self.beta_outer} not in (0,1)')
        if self.outer_lr <= 0:
            issues.append(f'ERROR: outer_lr={self.outer_lr} must be > 0')
        return issues

    def to_dict(self):
        return {'mode': self.mode, 'beta_outer': self.beta_outer,
                'outer_lr': self.outer_lr}

    @classmethod
    def from_dict(cls, d):
        return cls(mode=d.get('mode', 'average'),
                   beta_outer=d.get('beta_outer', 0.9),
                   outer_lr=d.get('outer_lr', 1.0))


class DeslocInnerOptimizerConfig:
    """Configuration for DES-LOC inner optimizer.
    Ref: Section 5.6 — ADOPT modifies Adam update to guarantee
    convergence for any beta2, removing beta1 < sqrt(beta2) constraint.

    ADOPT update: v_t uses g_{t-1} instead of g_t.
    Muon: uses Nesterov SGD on steepest-descent directions."""

    def __init__(self, optimizer_type='adam', beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.0):
        self.optimizer_type = optimizer_type
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

    def validate(self):
        issues = []
        if self.optimizer_type not in ('adam', 'adopt', 'muon', 'sgdm'):
            issues.append(f'ERROR: type={self.optimizer_type}')
        if not (0 <= self.beta1 < 1):
            issues.append(f'ERROR: beta1={self.beta1}')
        if not (0 <= self.beta2 < 1):
            issues.append(f'ERROR: beta2={self.beta2}')
        # ADOPT removes this constraint, but standard Adam needs it
        if self.optimizer_type == 'adam' and self.beta1 >= self.beta2 ** 0.5:
            issues.append(f'WARNING: beta1={self.beta1} >= sqrt(beta2)='
                         f'{self.beta2**0.5:.4f}; consider ADOPT')
        return issues

    def to_dict(self):
        return {'type': self.optimizer_type, 'beta1': self.beta1,
                'beta2': self.beta2, 'eps': self.eps, 'wd': self.weight_decay}

    @classmethod
    def from_dict(cls, d):
        return cls(optimizer_type=d.get('type', 'adam'),
                   beta1=d.get('beta1', 0.9), beta2=d.get('beta2', 0.999),
                   eps=d.get('eps', 1e-8), weight_decay=d.get('wd', 0.0))


def desloc_format_table(results, metrics, group_key='Kx'):
    """Format results as comparison table for logging.
    Ref: Section 5 tables — NeurIPS comparison format."""
    import math
    groups = {}
    for r in results:
        g = r.get(group_key, r.get('config', {}).get(group_key))
        groups.setdefault(g, []).append(r)
    header = [group_key] + metrics
    rows = [header]
    for gval in sorted(groups.keys()):
        row = [str(gval)]
        for mk in metrics:
            vals = [e.get(mk) or e.get('metrics', {}).get(mk) for e in groups[gval]]
            vals = [v for v in vals if v is not None]
            if vals:
                m = sum(vals) / len(vals)
                if len(vals) > 1:
                    s = math.sqrt(sum((x-m)**2 for x in vals) / (len(vals)-1))
                    row.append(f'{m:.4f}+/-{s:.4f}')
                else:
                    row.append(f'{m:.4f}')
            else:
                row.append('-')
        rows.append(row)
    return rows


def desloc_compute_optimal_config(model_params, num_workers,
                                   peak_tflops, net_bw_gbps,
                                   beta1=0.9, beta2=0.999):
    """Compute optimal DES-LOC config for given hardware setup.
    Finds smallest Kx where training becomes compute-bound.

    Returns DeslocConfig instance with recommended parameters.
    Ref: Section 5.4 — 'setting Kx for sufficient throughput
    based on bandwidth, then Ku, Kv as constant multiples.'"""
    import math
    param_bytes = model_params * 2  # BF16
    # Estimate compute time: 6*N*512*4 / (peak * 1e12)
    flops = 6 * model_params * 512 * 4
    compute_s = flops / (peak_tflops * 1e12) if peak_tflops > 0 else 1
    # Allreduce time
    if num_workers > 1 and net_bw_gbps > 0:
        ring = 2.0 * (num_workers - 1) / num_workers
        ar_s = ring * param_bytes / (net_bw_gbps * 1e9)
    else:
        ar_s = 0
    # Find smallest Kx where ar/Kx < compute
    if ar_s <= compute_s:
        Kx = 1
    else:
        Kx = 2 ** int(math.ceil(math.log2(ar_s / compute_s)))
    Kx = min(Kx, 256)
    # Recommend Ku, Kv from half-life
    tau1 = -1.0 / math.log(beta1) if 0 < beta1 < 1 else 1
    tau2 = -1.0 / math.log(beta2) if 0 < beta2 < 1 else 1
    ratio = min(10, tau2 / tau1) if tau1 > 0 else 3
    Ku = max(1, int(round(Kx * 3)))
    Kv = max(Ku, int(round(Kx * ratio)))
    return DeslocConfig(enabled=True, Kx=Kx, Ku=Ku, Kv=Kv,
                        beta1=beta1, beta2=beta2)


def desloc_config_from_json(json_path):
    """Load DES-LOC config from standalone JSON file.
    Format: {"desloc": {"enabled": true, "Kx": 32, ...}}"""
    import json
    with open(json_path) as f:
        data = json.load(f)
    desloc_dict = data.get('desloc', data)
    return DeslocConfig.from_dict(desloc_dict)


def desloc_config_to_deepspeed_json(desloc_config, base_config=None):
    """Merge DES-LOC config into a DeepSpeed JSON config dict.
    If base_config is None, creates minimal config."""
    if base_config is None:
        base_config = {
            'train_batch_size': 32,
            'fp16': {'enabled': True},
        }
    base_config[DESLOC] = desloc_config.to_dict()
    return base_config


def desloc_print_config_summary(config):
    """Print DES-LOC configuration summary to console.
    Includes validation warnings."""
    if isinstance(config, DeslocConfig):
        print(config.summary())
        issues = config.validate()
        for iss in issues:
            print(f'  {iss}')
    elif hasattr(config, 'desloc_enabled'):
        # DeepSpeedConfig object
        print(f'DES-LOC: enabled={config.desloc_enabled} '
              f'Kx={config.desloc_Kx} Ku={config.desloc_Ku} Kv={config.desloc_Kv}')


DESLOC_DEFAULT_CONFIGS = {
    'ddp_baseline': DeslocConfig(enabled=False),
    'conservative': DeslocConfig(enabled=True, Kx=4, Ku=12, Kv=24),
    'moderate': DeslocConfig(enabled=True, Kx=32, Ku=96, Kv=192),
    'aggressive': DeslocConfig(enabled=True, Kx=128, Ku=384, Kv=768),
}


def desloc_config_summary_string(cfg):
    """One-line DES-LOC config summary for logging."""
    if not cfg.get('enabled', False):
        return 'DES-LOC: disabled'
    return (f"DES-LOC: Kx={cfg.get('Kx',1)} Ku={cfg.get('Ku',3)} "
            f"Kv={cfg.get('Kv',6)} rho={cfg.get('clip_rho',1.0)}")
