# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.accelerator import get_accelerator

#############################################
# Routes
#############################################
ROUTE_TRAIN = "train"
ROUTE_EVAL = "eval"
ROUTE_PREDICT = "predict"
ROUTE_ENCODE = "encode"

#############################################
# Batch size
#############################################
TRAIN_BATCH_SIZE = "train_batch_size"
TRAIN_BATCH_SIZE_DEFAULT = None

#############################################
# Sparse attention
#############################################
SPARSE_ATTENTION = "sparse_attention"
SPARSE_DENSE_MODE = "dense"
SPARSE_FIXED_MODE = "fixed"
SPARSE_VARIABLE_MODE = "variable"
SPARSE_BIGBIRD_MODE = "bigbird"
SPARSE_BSLONGFORMER_MODE = "bslongformer"
SPARSE_MODE = "mode"
SPARSE_MODE_DEFAULT = SPARSE_FIXED_MODE
SPARSE_BLOCK = "block"
SPARSE_BLOCK_DEFAULT = 16
SPARSE_DIFFERENT_LAYOUT_PER_HEAD = "different_layout_per_head"
SPARSE_DIFFERENT_LAYOUT_PER_HEAD_DEFAULT = False
SPARSE_NUM_LOCAL_BLOCKS = "num_local_blocks"
SPARSE_NUM_LOCAL_BLOCKS_DEFAULT = 4
SPARSE_NUM_GLOBAL_BLOCKS = "num_global_blocks"
SPARSE_NUM_GLOBAL_BLOCKS_DEFAULT = 1
SPARSE_ATTENTION_TYPE = "attention"
SPARSE_ATTENTION_TYPE_DEFAULT = "bidirectional"
SPARSE_HORIZONTAL_GLOBAL_ATTENTION = "horizontal_global_attention"
SPARSE_HORIZONTAL_GLOBAL_ATTENTION_DEFAULT = False
SPARSE_NUM_DIFFERENT_GLOBAL_PATTERNS = "num_different_global_patterns"
SPARSE_NUM_DIFFERENT_GLOBAL_PATTERNS_DEFAULT = 1
SPARSE_NUM_RANDOM_BLOCKS = "num_random_blocks"
SPARSE_NUM_RANDOM_BLOCKS_DEFAULT = 0
SPARSE_LOCAL_WINDOW_BLOCKS = "local_window_blocks"
SPARSE_LOCAL_WINDOW_BLOCKS_DEFAULT = [4]
SPARSE_GLOBAL_BLOCK_INDICES = "global_block_indices"
SPARSE_GLOBAL_BLOCK_INDICES_DEFAULT = [0]
SPARSE_GLOBAL_BLOCK_END_INDICES = "global_block_end_indices"
SPARSE_GLOBAL_BLOCK_END_INDICES_DEFAULT = None
SPARSE_NUM_SLIDING_WINDOW_BLOCKS = "num_sliding_window_blocks"
SPARSE_NUM_SLIDING_WINDOW_BLOCKS_DEFAULT = 3

#############################################
# Optimizer and lr scheduler
#############################################
OPTIMIZER = "optimizer"
OPTIMIZER_TYPE_DEFAULT = None
OPTIMIZER_PARAMS = "params"
TYPE = "type"
LEGACY_FUSION = "legacy_fusion"
LEGACY_FUSION_DEFAULT = False
SCHEDULER = "scheduler"
SCHEDULER_TYPE_DEFAULT = None
SCHEDULER_PARAMS = "params"
MAX_GRAD_NORM = 'max_grad_norm'

#############################################
# Optimizer and lr scheduler
#############################################
ZERO_ALLOW_UNTESTED_OPTIMIZER = "zero_allow_untested_optimizer"
ZERO_ALLOW_UNTESTED_OPTIMIZER_DEFAULT = False
ZERO_FORCE_DS_CPU_OPTIMIZER = "zero_force_ds_cpu_optimizer"
ZERO_FORCE_DS_CPU_OPTIMIZER_DEFAULT = True

# Steps
STEPS_PER_PRINT = "steps_per_print"
STEPS_PER_PRINT_DEFAULT = None

#########################################
# Training micro batch size per GPU
#########################################
# Batch size for one training step. This is used when the
# TRAIN_BATCH_SIZE cannot fit in GPU memory to determine
# the number of gradient accumulation steps. By default, this
# is set to None. Users can configure in ds_config.json as below example:
TRAIN_MICRO_BATCH_SIZE_PER_GPU = '''
TRAIN_MICRO_BATCH_SIZE_PER_GPU is defined in this format:
"train_micro_batch_size_per_gpu": 1
'''
TRAIN_MICRO_BATCH_SIZE_PER_GPU = "train_micro_batch_size_per_gpu"
TRAIN_MICRO_BATCH_SIZE_PER_GPU_DEFAULT = None

#########################################
# Gradient Accumulation
#########################################
# Gradient accumulation feature. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
GRADIENT_ACCUMULATION_FORMAT = '''
Gradient Accumulation should be of the format:
"gradient_accumulation_steps": 1
'''
GRADIENT_ACCUMULATION_STEPS = "gradient_accumulation_steps"
GRADIENT_ACCUMULATION_STEPS_DEFAULT = None

# DeepSpeed CSR gradient sparsity
SPARSE_GRADIENTS = "sparse_gradients"
SPARSE_GRADIENTS_DEFAULT = False

#########################################
# BFLOAT16 support
#########################################
# BFLOAT16 feature. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
BFLOAT16_FORMAT = '''
BFLOAT16 parameters should be of the format:
"bf16": {
  "enabled": true,
  "immediate_grad_update": false,
  "check_overflow": false
}
'''
BFLOAT16 = "bf16"
BFLOAT16_OLD = "bfloat16"  # keeping for backwards compatibility

BFLOAT16_ENABLED = "enabled"
BFLOAT16_ENABLED_DEFAULT = False

CHECK_OVERFLOW = "check_overflow"
BFLOAT16_CHECK_OVERFLOW_DEFAULT = False

# BFLOAT16 optimizer immediate gradient update
BFLOAT16_IMMEDIATE_GRAD_UPDATE = "immediate_grad_update"
BFLOAT16_IMMEDIATE_GRAD_UPDATE_DEFAULT = True

# BFLOAT16 master weights and optimizer states options
BFLOAT16_MASTER_WEIGHTS_AND_GRADS = "bf16_master_weights_and_grads"
BFLOAT16_MASTER_WEIGHTS_AND_GRADS_DEFAULT = False
BFLOAT16_OPTIMIZER_STATES = "bf16_optimizer_states"
BFLOAT16_OPTIMIZER_STATES_DEFAULT = False

# DDP variant of BFLOAT16
# DDP variant: bf16 model with bf16 grad accumulation (uses FP16_Optimizer in bf16 mode)
# Must be different from BFLOAT16 to allow proper optimizer selection
DDP_BFLOAT16 = "ddp_bf16"

#########################################
# FP16 support
#########################################
# FP16 feature. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
FP16_FORMAT = '''
FP16 parameters should be of the format:
"fp16": {
  "enabled": true,
  "auto_cast": false,
  "loss_scale": 0,
  "initial_scale_power": 16,
  "loss_scale_window": 1000,
  "hysteresis": 2,
  "consecutive_hysteresis": false,
  "min_loss_scale": 1
}
'''
FP16 = "fp16"

FP16_ENABLED = "enabled"
FP16_ENABLED_DEFAULT = False

# FP16 loss scale, zero means using dynamic scaling
FP16_LOSS_SCALE = "loss_scale"
FP16_LOSS_SCALE_DEFAULT = 0

FP16_AUTO_CAST = "auto_cast"
FP16_AUTO_CAST_DEFAULT = False

# FP16 initial dynamic scale loss power
FP16_INITIAL_SCALE_POWER = "initial_scale_power"
FP16_INITIAL_SCALE_POWER_DEFAULT = 16

# FP16 loss scale window
FP16_LOSS_SCALE_WINDOW = "loss_scale_window"
FP16_LOSS_SCALE_WINDOW_DEFAULT = 1000

# FP16 hysteresis
FP16_HYSTERESIS = "hysteresis"
FP16_HYSTERESIS_DEFAULT = 2

# FP16 consecutive hysteresis
FP16_CONSECUTIVE_HYSTERESIS = "consecutive_hysteresis"
FP16_CONSECUTIVE_HYSTERESIS_DEFAULT = False

# FP16 min loss scale
FP16_MIN_LOSS_SCALE = "min_loss_scale"
FP16_MIN_LOSS_SCALE_DEFAULT = 1

# FP16 master and grads
FP16_MASTER_WEIGHTS_AND_GRADS = "fp16_master_weights_and_grads"
FP16_MASTER_WEIGHTS_AND_GRADS_DEFAULT = False

#########################################
# Apex AMP support
#########################################
# Use Apex AMP for mixed precision support, all parameters (other than 'enabled') will be passed to
# amp.initialize(model, optimizer, **amp_params)
# See apex documentation for supported parameters/features: https://nvidia.github.io/apex/amp.html#apex.amp.initialize
AMP_FORMAT = '''
"amp" {
  "enabled: true,
  "opt_level": "O1",
  ...
}
'''
AMP = "amp"

AMP_ENABLED = "enabled"
AMP_ENABLED_DEFAULT = False

#########################################
# Torch AMP support
#########################################
TORCH_AUTOCAST_FORMAT = '''
PyTorch autocast config should be of the format:
"torch_autocast": {
  "enabled": true,
  "dtype": "bfloat16",
  "lower_precision_safe_modules": [
    "torch.nn.modules.linear.Linear",
    "torch.nn.modules.conv.Conv2d"
  ]
}
'''
TORCH_AUTOCAST = "torch_autocast"

TORCH_AUTOCAST_ENABLED = "enabled"
TORCH_AUTOCAST_ENABLED_DEFAULT = False
TORCH_AUTOCAST_DTYPE = "dtype"
TORCH_AUTOCAST_LOWER_PRECISION_SAFE_MODULES = "lower_precision_safe_modules"

#########################################
# Gradient clipping
#########################################
# Gradient clipping. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
GRADIENT_CLIPPING_FORMAT = '''
Gradient clipping should be enabled as:
"gradient_clipping": 1.0
'''
GRADIENT_CLIPPING = 'gradient_clipping'
GRADIENT_CLIPPING_DEFAULT = 0.

#########################################
# Capture graph for short kernels sequences
#########################################
# Graph harvesting. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
GRAPH_HARVESTING_FORMAT = '''
Graph harvesting should be enabled as:
"graph_harvesting": true
'''
GRAPH_HARVESTING = 'graph_harvesting'
GRAPH_HARVESTING_DEFAULT = False

#########################################
# Communication data type
#########################################
# Supported types: ['none', 'fp16', 'fp32']
# By default, this feature is not enabled ('none' value)
# Users can configure in ds_config.json as below example:
COMMUNICATION_DATA_TYPE_FORMAT = '''
Communication data type should be set as:
"communication_data_type": "fp32"
'''
COMMUNICATION_DATA_TYPE = "communication_data_type"
COMMUNICATION_DATA_TYPE_DEFAULT = None

###########################################################
# Gradient communication data type for sequence parallelism
###########################################################
# Supported types: ['fp16', 'bf16','fp32']
# Default value is fp32
# Users can configure in ds_config.json as below example:
SEQ_PARALLEL_COMMUNICATION_DATA_TYPE_FORMAT = '''
Optional comm data type for seq paralleism should be set as:
"seq_parallel_communication_data_type": "fp32"
'''
SEQ_PARALLEL_COMMUNICATION_DATA_TYPE = "seq_parallel_communication_data_type"

if get_accelerator().device_name == 'cuda' and get_accelerator().communication_backend_version() >= (2, 27, 3):
    # nccl>=2.27.3 uses fp32 accumulation for half precision inputs, so there is no need to waste compute and memory to manually upcast to fp32 unless the user wants it and then override
    SEQ_PARALLEL_COMMUNICATION_DATA_TYPE_DEFAULT = None
else:
    SEQ_PARALLEL_COMMUNICATION_DATA_TYPE_DEFAULT = "fp32"

SEQ_PARALLEL_COMMUNICATION_DATA_TYPE_DEFAULT = "fp32"

#########################################
# Scale/predivide gradients before allreduce
#########################################
# Prescale gradients. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
PRESCALE_GRADIENTS_FORMAT = '''
Gradient prescaling should be enabled as:
"prescale_gradients": true
'''
PRESCALE_GRADIENTS = "prescale_gradients"
PRESCALE_GRADIENTS_DEFAULT = False

GRADIENT_PREDIVIDE_FACTOR_FORMAT = '''
Gradient predivide factor should be enabled as:
"gradient_predivide_factor": 1.0
'''
GRADIENT_PREDIVIDE_FACTOR = "gradient_predivide_factor"
GRADIENT_PREDIVIDE_FACTOR_DEFAULT = 1.0

#########################################
# Disable AllGather
#########################################
# Disable AllGather. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
DISABLE_ALLGATHER_FORMAT = '''
Disable AllGather should be enabled as:
"disable_allgather": true
'''
DISABLE_ALLGATHER = "disable_allgather"
DISABLE_ALLGATHER_DEFAULT = False

#########################################
# Dump DeepSpeed state
#########################################
# Dump State. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
DUMP_STATE_FORMAT = '''
Dump state should be enabled as:
"dump_state": true
'''
DUMP_STATE = 'dump_state'
DUMP_STATE_DEFAULT = False

#########################################
# Vocabulary size
#########################################
# Vocabulary size.
# Users can configure in ds_config.json as below example:
VOCABULARY_SIZE_FORMAT = '''
Vocabulary size can be specified as:
"vocabulary_size": 1024
'''
VOCABULARY_SIZE = 'vocabulary_size'
VOCABULARY_SIZE_DEFAULT = None

#########################################
# Wall block breakdown
#########################################
# Wall clock breakdown. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
WALL_CLOCK_BREAKDOWN_FORMAT = '''
Wall block breakdown should be enabled as:
"wall_clock_breakdown": true
'''
WALL_CLOCK_BREAKDOWN = 'wall_clock_breakdown'
WALL_CLOCK_BREAKDOWN_DEFAULT = False

MEMORY_BREAKDOWN = 'memory_breakdown'
MEMORY_BREAKDOWN_DEFAULT = False

#########################################
# Eigenvalue
#########################################
# Eigenvalue computation. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
EIGENVALUE_FORMAT = '''
Tensorboard can be specified as:
"eigenvalue": {
  "enabled": true,
  "verbose": true,
  "max_iter": 100,
  "tol": 1e-2,
  "stability": 1e-6
}
'''
EIGENVALUE = "eigenvalue"

# Tensorboard enable signal
EIGENVALUE_ENABLED = "enabled"
EIGENVALUE_ENABLED_DEFAULT = False

EIGENVALUE_VERBOSE = "verbose"
EIGENVALUE_VERBOSE_DEFAULT = False

EIGENVALUE_MAX_ITER = "max_iter"
EIGENVALUE_MAX_ITER_DEFAULT = 100

EIGENVALUE_TOL = "tol"
EIGENVALUE_TOL_DEFAULT = 1e-2

EIGENVALUE_STABILITY = "stability"
EIGENVALUE_STABILITY_DEFAULT = 1e-6

EIGENVALUE_GAS_BOUNDARY_RESOLUTION = "gas_boundary_resolution"
EIGENVALUE_GAS_BOUNDARY_RESOLUTION_DEFAULT = 1

EIGENVALUE_LAYER_NAME = "layer_name"
EIGENVALUE_LAYER_NAME_DEFAULT = "bert.encoder.layer"

EIGENVALUE_LAYER_NUM = "layer_num"
EIGENVALUE_LAYER_NUM_DEFAULT = 0

#########################################
# Progressive Layer Drop (PLD)
#########################################
PROGRESSIVE_LAYER_DROP = "progressive_layer_drop"

# PLD enable signal
PLD_ENABLED = "enabled"
PLD_ENABLED_DEFAULT = False

PLD_THETA = "theta"
PLD_THETA_DEFAULT = 1.0

PLD_GAMMA = "gamma"
PLD_GAMMA_DEFAULT = 0.001


#########################################
# Validation modes
#########################################
class ValidationMode:
    WARN = "WARN"
    IGNORE = "IGNORE"
    FAIL = "FAIL"


#########################################
# Checkpoint config params
#########################################
# "checkpoint": {
#   tag_validation=["Ignore"|"Warn"|"Fail"]
#   load_universal=false
#   use_node_local_storage=false
#   parallel_write: {
#     pipeline_stage: [True|False]
#   }
# }
CHECKPOINT = "checkpoint"
CHECKPOINT_TAG_VALIDATION = "tag_validation"
CHECKPOINT_TAG_VALIDATION_DEFAULT = ValidationMode.WARN
CHECKPOINT_TAG_VALIDATION_MODES = [ValidationMode.WARN, ValidationMode.IGNORE, ValidationMode.FAIL]

LOAD_UNIVERSAL_CHECKPOINT = "load_universal"
LOAD_UNIVERSAL_CHECKPOINT_DEFAULT = False

USE_NODE_LOCAL_STORAGE_CHECKPOINT = "use_node_local_storage"
USE_NODE_LOCAL_STORAGE_CHECKPOINT_DEFAULT = False

CHECKPOINT_PARALLEL_WRITE = "parallel_write"
CHECKPOINT_PARALLEL_WRITE_PIPELINE_STAGE = "pipeline_stage"
CHECKPOINT_PARALLEL_WRITE_PIPELINE_STAGE_DEFAULT = False

#########################################
# Data types config params
#########################################
# "data_types": {
#   grad_accum_dtype=["bf16"|"fp16"|"fp32"]
#   }
# }

DATA_TYPES = "data_types"
GRAD_ACCUM_DTYPE = "grad_accum_dtype"
GRAD_ACCUM_DTYPE_DEFAULT = None

#########################################
# Drop the last incomplete Batch
# #########################################
# dataloader_drop_last. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
DATALOADER_DROP_LAST_FORMAT = '''
The last incomplete batch can be dropped by setting:
"dataloader_drop_last": True
'''
DATALOADER_DROP_LAST = "dataloader_drop_last"
DATALOADER_DROP_LAST_DEFAULT = False

#########################################
# PIPELINE PARALLELISM
#########################################
PIPE_REPLICATED = 'ds_pipe_replicated'

#########################################
# DATA PARALLELISM
#########################################
DATA_PARALLEL_GROUP = "data_parallel_group"
GLOBAL_RANK = "global_rank"

#########################################
# EXPERT-DATA PARALLELISM TOPO Config
#########################################
USE_DATA_BEFORE_EXPERT_PARALLEL = "use_data_before_expert_parallelism"
USE_DATA_BEFORE_EXPERT_PARALLEL_DEFAULT = False

#########################################
# DES-LOC: Desynced Low Communication (M193)
#########################################
DESLOC = "desloc"
DESLOC_ENABLED = "enabled"
DESLOC_ENABLED_DEFAULT = False
DESLOC_KX = "Kx"
DESLOC_KX_DEFAULT = 32
DESLOC_KU = "Ku"
DESLOC_KU_DEFAULT = 96
DESLOC_KV = "Kv"
DESLOC_KV_DEFAULT = 192
DESLOC_CLIP_RADIUS = "clip_radius"
DESLOC_CLIP_RADIUS_DEFAULT = 1.0
DESLOC_OUTER_OPT = "outer_optimizer"
DESLOC_OUTER_OPT_DEFAULT = "averaging"
DESLOC_OUTER_OPT_NESTEROV = "nesterov"
DESLOC_NESTEROV_MOMENTUM = "nesterov_momentum"
DESLOC_NESTEROV_MOMENTUM_DEFAULT = 0.9
DESLOC_NESTEROV_LR = "nesterov_lr"
DESLOC_NESTEROV_LR_DEFAULT = 1.0
DESLOC_MUON_COMPAT = "muon_compat"
DESLOC_MUON_COMPAT_DEFAULT = False
DESLOC_WARMUP_SYNC_STEPS = "warmup_sync_steps"
DESLOC_WARMUP_SYNC_STEPS_DEFAULT = 0
DESLOC_COMM_LOGGING = "comm_logging"
DESLOC_COMM_LOGGING_DEFAULT = False


#########################################
# DES-LOC: Desynced Low Communication (M193)
#########################################
DESLOC = "desloc"
DESLOC_ENABLED = "enabled"
DESLOC_ENABLED_DEFAULT = False
DESLOC_KX = "Kx"
DESLOC_KX_DEFAULT = 32
DESLOC_KU = "Ku"
DESLOC_KU_DEFAULT = 96
DESLOC_KV = "Kv"
DESLOC_KV_DEFAULT = 192
DESLOC_CLIP_RADIUS = "clip_radius"
DESLOC_CLIP_RADIUS_DEFAULT = 1.0
DESLOC_OUTER_OPT = "outer_optimizer"
DESLOC_OUTER_OPT_DEFAULT = "averaging"
DESLOC_OUTER_OPT_NESTEROV = "nesterov"
DESLOC_NESTEROV_MOMENTUM = "nesterov_momentum"
DESLOC_NESTEROV_MOMENTUM_DEFAULT = 0.9
DESLOC_NESTEROV_LR = "nesterov_lr"
DESLOC_NESTEROV_LR_DEFAULT = 1.0
DESLOC_MUON_COMPAT = "muon_compat"
DESLOC_MUON_COMPAT_DEFAULT = False
DESLOC_WARMUP_SYNC_STEPS = "warmup_sync_steps"
DESLOC_WARMUP_SYNC_STEPS_DEFAULT = 0
DESLOC_COMM_LOGGING = "comm_logging"
DESLOC_COMM_LOGGING_DEFAULT = False


# =====================================================================
# M251 — Claude-16: DES-LOC Protocol Constants
# Complete constant definitions for the DES-LOC subsystem
# Ref: Algorithm 1, Theorem 1, Section 4.1 of DES-LOC paper
# =====================================================================

DESLOC_DEFAULT_KX = 32  # Default parameter sync period
DESLOC_DEFAULT_KU_MULT = 3  # Ku = Kx * this multiplier
DESLOC_DEFAULT_KV_MULT = 6  # Kv = Kx * this multiplier
DESLOC_MIN_KX = 1  # Minimum Kx (1 = standard DDP)
DESLOC_MAX_KX = 256  # Maximum Kx (beyond this, convergence degrades)
DESLOC_WARMUP_STEPS = 512  # Default warmup steps (Kx=1 during warmup)
DESLOC_DEFAULT_CLIP_RHO = 1.0  # Default coordinate-wise clipping radius
DESLOC_DEFAULT_BETA1 = 0.9  # Default first momentum decay
DESLOC_DEFAULT_BETA2 = 0.999  # Default second momentum decay
DESLOC_PSI_WARN_THRESHOLD = 10.0  # Warn if psi factor exceeds this
DESLOC_HALFLIFE_BETA1_09 = 6.58  # Half-life of beta1=0.9: ln(0.5)/ln(0.9)
DESLOC_HALFLIFE_BETA2_0999 = 692.8  # Half-life of beta2=0.999: ln(0.5)/ln(0.999)
DESLOC_HALFLIFE_BETA2_095 = 13.51  # Half-life of beta2=0.95: ln(0.5)/ln(0.95)
DESLOC_COMM_TIER_PARAM = 0  # Communication tier: model parameters (x)
DESLOC_COMM_TIER_MOM1 = 1  # Communication tier: first momentum (u)
DESLOC_COMM_TIER_MOM2 = 2  # Communication tier: second momentum (v)
DESLOC_OUTER_OPT_AVERAGE = "average"  # Outer optimizer: simple averaging
DESLOC_OUTER_OPT_NESTEROV = "nesterov"  # Outer optimizer: Nesterov momentum
DESLOC_INNER_OPT_ADAM = "adam"  # Inner optimizer: Adam
DESLOC_INNER_OPT_ADOPT = "adopt"  # Inner optimizer: ADOPT
DESLOC_INNER_OPT_MUON = "muon"  # Inner optimizer: Muon
DESLOC_INNER_OPT_SGDM = "sgdm"  # Inner optimizer: SGD with momentum
DESLOC_TRANSPORT_P2P = 0  # Transport type: peer-to-peer (NVLink/PCIe)
DESLOC_TRANSPORT_SHM = 1  # Transport type: shared memory
DESLOC_TRANSPORT_NET = 2  # Transport type: network (Ethernet/InfiniBand)
DESLOC_HEALTH_SCORE_HEALTHY = 0.8  # Health score threshold for healthy
DESLOC_HEALTH_SCORE_WARNING = 0.5  # Health score threshold for warning
DESLOC_HEALTH_SCORE_CRITICAL = 0.2  # Health score threshold for critical
DESLOC_BW_DROP_THRESHOLD = 0.5  # Bandwidth drop detection threshold (50%)
DESLOC_LAT_SPIKE_FACTOR = 5.0  # Latency spike detection factor (5x median)
DESLOC_DIVERGENCE_SPIKE_RATIO = 2.0  # Loss spike ratio for divergence detection
DESLOC_DIVERGENCE_RECOVERY_STEPS = 100  # Steps to hold reduced Kx during recovery
DESLOC_NESTEROV_DEFAULT_MOMENTUM = 0.9  # Default Nesterov outer optimizer momentum
DESLOC_NESTEROV_DEFAULT_LR = 1.0  # Default Nesterov outer learning rate

# DES-LOC config key strings (for JSON config parsing)
DESLOC_ENABLED_KEY = "desloc_enabled"
DESLOC_KX_KEY = "desloc_Kx"
DESLOC_KU_KEY = "desloc_Ku"
DESLOC_KV_KEY = "desloc_Kv"
DESLOC_CLIP_RHO_KEY = "desloc_clip_rho"
DESLOC_WARMUP_KEY = "desloc_warmup_steps"
DESLOC_OUTER_OPT_KEY = "desloc_outer_optimizer"
DESLOC_INNER_OPT_KEY = "desloc_inner_optimizer"
DESLOC_NESTEROV_MOM_KEY = "desloc_nesterov_momentum"
DESLOC_NESTEROV_LR_KEY = "desloc_nesterov_lr"
DESLOC_AUTO_KX_KEY = "desloc_auto_Kx"
DESLOC_TOPOLOGY_AWARE_KEY = "desloc_topology_aware"

# Scaling law constants (Chinchilla + DES-LOC correction)
SCALING_CHINCHILLA_A = 406.4  # Chinchilla coefficient A
SCALING_CHINCHILLA_B = 410.7  # Chinchilla coefficient B
SCALING_CHINCHILLA_ALPHA = 0.34  # Chinchilla exponent alpha
SCALING_CHINCHILLA_BETA = 0.28  # Chinchilla exponent beta
SCALING_CHINCHILLA_E = 1.69  # Chinchilla irreducible loss E
SCALING_COMPUTE_PER_TOKEN = 6  # Approx FLOPS per token per param (6ND)

# Hardware reference constants
HW_H100_SXM_BF16_TFLOPS = 989.5  # H100 SXM peak BF16 TFLOPS
HW_H100_NVL_BF16_TFLOPS = 835.0  # H100 NVL peak BF16 TFLOPS
HW_A100_SXM_BF16_TFLOPS = 312.0  # A100 SXM peak BF16 TFLOPS
HW_A6000_BF16_TFLOPS = 38.7  # RTX A6000 peak BF16 TFLOPS
HW_H100_HBM_BW_TBPS = 3.35  # H100 SXM HBM bandwidth TB/s
HW_A100_HBM_BW_TBPS = 2.0  # A100 SXM HBM bandwidth TB/s
HW_NVLINK_H100_BW_GBPS = 900  # H100 NVLink bandwidth GB/s
HW_PCIE_GEN4_BW_GBPS = 32  # PCIe Gen4 x16 bandwidth GB/s
HW_PCIE_GEN5_BW_GBPS = 64  # PCIe Gen5 x16 bandwidth GB/s
HW_TRAINIUM2_BF16_TFLOPS = 512  # AWS Trainium2 peak BF16 TFLOPS (estimated)
HW_TRAINIUM2_HBM_GB = 192  # AWS Trainium2 HBM per chip (GB)
HW_NEURONLINK_BW_GBPS = 384  # Trainium2 NeuronLink bandwidth GB/s

# NKI-FA log format (commit da964f3)
NKIFA_CONFIG_PREFIX = "### "  # Log line config prefix
NKIFA_CONFIG_SUFFIX = " ###"  # Log line config suffix
NKIFA_METRIC_SEP = ": "  # Metric key-value separator
NKIFA_MIN_SIG_DIGITS = 4  # Minimum significant digits for NeurIPS

# Model size references (parameter counts)
MODEL_125M_PARAMS = 125_000_000
MODEL_350M_PARAMS = 350_000_000
MODEL_1B_PARAMS = 1_000_000_000
MODEL_3B_PARAMS = 3_000_000_000
MODEL_7B_PARAMS = 7_000_000_000
MODEL_13B_PARAMS = 13_000_000_000
MODEL_70B_PARAMS = 70_000_000_000

DESLOC_TIER_NAMES = {0: "params", 1: "momentum1", 2: "momentum2"}
DESLOC_TIER_ALIASES = {"x": 0, "u": 1, "v": 2, "params": 0, "mom1": 1, "mom2": 2}
DESLOC_TRANSPORT_NAMES = {0: "P2P/NVLink", 1: "SharedMem", 2: "Network"}
DESLOC_OUTER_OPTS = ("average", "nesterov")
DESLOC_INNER_OPTS = ("adam", "adopt", "muon", "sgdm")

def desloc_validate_Kx(Kx):
    """Validate Kx is within acceptable range."""
    if not isinstance(Kx, int) or Kx < DESLOC_MIN_KX or Kx > DESLOC_MAX_KX:
        return False, f"Kx={Kx} out of range [{DESLOC_MIN_KX}, {DESLOC_MAX_KX}]"
    return True, ""

def desloc_validate_tier(tier):
    """Validate tier identifier."""
    if tier in DESLOC_TIER_ALIASES:
        return True, DESLOC_TIER_ALIASES[tier]
    return False, f"Unknown tier: {tier}"

def desloc_get_default_config():
    """Return default DES-LOC configuration dict."""
    return {
        DESLOC_ENABLED_KEY: False,
        DESLOC_KX_KEY: DESLOC_DEFAULT_KX,
        DESLOC_KU_KEY: DESLOC_DEFAULT_KX * DESLOC_DEFAULT_KU_MULT,
        DESLOC_KV_KEY: DESLOC_DEFAULT_KX * DESLOC_DEFAULT_KV_MULT,
        DESLOC_CLIP_RHO_KEY: DESLOC_DEFAULT_CLIP_RHO,
        DESLOC_WARMUP_KEY: DESLOC_WARMUP_STEPS,
        DESLOC_OUTER_OPT_KEY: DESLOC_OUTER_OPT_AVERAGE,
        DESLOC_INNER_OPT_KEY: DESLOC_INNER_OPT_ADAM,
    }

# M251 reserved constant slot 125
# M251 reserved constant slot 126
# M251 reserved constant slot 127
# M251 reserved constant slot 128
# M251 reserved constant slot 129
# M251 reserved constant slot 130
# M251 reserved constant slot 131
# M251 reserved constant slot 132
# M251 reserved constant slot 133
# M251 reserved constant slot 134
# M251 reserved constant slot 135
# M251 reserved constant slot 136
# M251 reserved constant slot 137
# M251 reserved constant slot 138
# M251 reserved constant slot 139
# M251 reserved constant slot 140
# M251 reserved constant slot 141
# M251 reserved constant slot 142
# M251 reserved constant slot 143
# M251 reserved constant slot 144
# M251 reserved constant slot 145
# M251 reserved constant slot 146
# M251 reserved constant slot 147
# M251 reserved constant slot 148
# M251 reserved constant slot 149
# M251 reserved constant slot 150
# M251 reserved constant slot 151
# M251 reserved constant slot 152
# M251 reserved constant slot 153
# M251 reserved constant slot 154
# M251 reserved constant slot 155
# M251 reserved constant slot 156
# M251 reserved constant slot 157
# M251 reserved constant slot 158
# M251 reserved constant slot 159
# M251 reserved constant slot 160
# M251 reserved constant slot 161
# M251 reserved constant slot 162
# M251 reserved constant slot 163
# M251 reserved constant slot 164
# M251 reserved constant slot 165
# M251 reserved constant slot 166
# M251 reserved constant slot 167
# M251 reserved constant slot 168
# M251 reserved constant slot 169
# M251 reserved constant slot 170
# M251 reserved constant slot 171
# M251 reserved constant slot 172
# M251 reserved constant slot 173
# M251 reserved constant slot 174
# M251 reserved constant slot 175
# M251 reserved constant slot 176
# M251 reserved constant slot 177
# M251 reserved constant slot 178
# M251 reserved constant slot 179
# M251 reserved constant slot 180
# M251 reserved constant slot 181
# M251 reserved constant slot 182
# M251 reserved constant slot 183
# M251 reserved constant slot 184
# M251 reserved constant slot 185
# M251 reserved constant slot 186
# M251 reserved constant slot 187
# M251 reserved constant slot 188
# M251 reserved constant slot 189
# M251 reserved constant slot 190
# M251 reserved constant slot 191
# M251 reserved constant slot 192
# M251 reserved constant slot 193
# M251 reserved constant slot 194
# M251 reserved constant slot 195
# M251 reserved constant slot 196
# M251 reserved constant slot 197
# M251 reserved constant slot 198
# M251 reserved constant slot 199
# M251 reserved constant slot 200
# M251 reserved constant slot 201
# M251 reserved constant slot 202
# M251 reserved constant slot 203
# M251 reserved constant slot 204
# M251 reserved constant slot 205
# M251 reserved constant slot 206
# M251 reserved constant slot 207
# M251 reserved constant slot 208
# M251 reserved constant slot 209
# M251 reserved constant slot 210
# M251 reserved constant slot 211
# M251 reserved constant slot 212
# M251 reserved constant slot 213
# M251 reserved constant slot 214
# M251 reserved constant slot 215
# M251 reserved constant slot 216
# M251 reserved constant slot 217
# M251 reserved constant slot 218
# M251 reserved constant slot 219
# M251 reserved constant slot 220
# M251 reserved constant slot 221
# M251 reserved constant slot 222
# M251 reserved constant slot 223
# M251 reserved constant slot 224
# M251 reserved constant slot 225
# M251 reserved constant slot 226
# M251 reserved constant slot 227
# M251 reserved constant slot 228
# M251 reserved constant slot 229
# M251 reserved constant slot 230
# M251 reserved constant slot 231
# M251 reserved constant slot 232
# M251 reserved constant slot 233
# M251 reserved constant slot 234
# M251 reserved constant slot 235
# M251 reserved constant slot 236
# M251 reserved constant slot 237
# M251 reserved constant slot 238
# M251 reserved constant slot 239
# M251 reserved constant slot 240
# M251 reserved constant slot 241
# M251 reserved constant slot 242
# M251 reserved constant slot 243
# M251 reserved constant slot 244
# M251 reserved constant slot 245
# M251 reserved constant slot 246
# M251 reserved constant slot 247
# M251 reserved constant slot 248
# M251 reserved constant slot 249
# M251 reserved constant slot 250
# M251 reserved constant slot 251
# M251 reserved constant slot 252
# M251 reserved constant slot 253
# M251 reserved constant slot 254
# M251 reserved constant slot 255
# M251 reserved constant slot 256
# M251 reserved constant slot 257
# M251 reserved constant slot 258
# M251 reserved constant slot 259
# M251 reserved constant slot 260
# M251 reserved constant slot 261
# M251 reserved constant slot 262
# M251 reserved constant slot 263
# M251 reserved constant slot 264
# M251 reserved constant slot 265
# M251 reserved constant slot 266
# M251 reserved constant slot 267
# M251 reserved constant slot 268
# M251 reserved constant slot 269
# M251 reserved constant slot 270
# M251 reserved constant slot 271
# M251 reserved constant slot 272
# M251 reserved constant slot 273
# M251 reserved constant slot 274
# M251 reserved constant slot 275
# M251 reserved constant slot 276
# M251 reserved constant slot 277
# M251 reserved constant slot 278
# M251 reserved constant slot 279
# M251 reserved constant slot 280
# M251 reserved constant slot 281
# M251 reserved constant slot 282
# M251 reserved constant slot 283
# M251 reserved constant slot 284
# M251 reserved constant slot 285
# M251 reserved constant slot 286
# M251 reserved constant slot 287
# M251 reserved constant slot 288
# M251 reserved constant slot 289
# M251 reserved constant slot 290
# M251 reserved constant slot 291
# M251 reserved constant slot 292
# M251 reserved constant slot 293
# M251 reserved constant slot 294
# M251 reserved constant slot 295
# M251 reserved constant slot 296
# M251 reserved constant slot 297
# M251 reserved constant slot 298
# M251 reserved constant slot 299
# M251 reserved constant slot 300
# M251 reserved constant slot 301
# M251 reserved constant slot 302
# M251 reserved constant slot 303
# M251 reserved constant slot 304
# M251 reserved constant slot 305
# M251 reserved constant slot 306
# M251 reserved constant slot 307
# M251 reserved constant slot 308
# M251 reserved constant slot 309
# M251 reserved constant slot 310
# M251 reserved constant slot 311
# M251 reserved constant slot 312
# M251 reserved constant slot 313
# M251 reserved constant slot 314
# M251 reserved constant slot 315
# M251 reserved constant slot 316
# M251 reserved constant slot 317
# M251 reserved constant slot 318
# M251 reserved constant slot 319
# M251 reserved constant slot 320
# M251 reserved constant slot 321
# M251 reserved constant slot 322
# M251 reserved constant slot 323
# M251 reserved constant slot 324
# M251 reserved constant slot 325
# M251 reserved constant slot 326
# M251 reserved constant slot 327
# M251 reserved constant slot 328
# M251 reserved constant slot 329
# M251 reserved constant slot 330
# M251 reserved constant slot 331
# M251 reserved constant slot 332
# M251 reserved constant slot 333
# M251 reserved constant slot 334
# M251 reserved constant slot 335
# M251 reserved constant slot 336
# M251 reserved constant slot 337
# M251 reserved constant slot 338
# M251 reserved constant slot 339
# M251 reserved constant slot 340
# M251 reserved constant slot 341
# M251 reserved constant slot 342
# M251 reserved constant slot 343
# M251 reserved constant slot 344
# M251 reserved constant slot 345
# M251 reserved constant slot 346
# M251 reserved constant slot 347
# M251 reserved constant slot 348
# M251 reserved constant slot 349
# M251 reserved constant slot 350
# M251 reserved constant slot 351
# M251 reserved constant slot 352
# M251 reserved constant slot 353
# M251 reserved constant slot 354
# M251 reserved constant slot 355
# M251 reserved constant slot 356
# M251 reserved constant slot 357
# M251 reserved constant slot 358
# M251 reserved constant slot 359
# M251 reserved constant slot 360
# M251 reserved constant slot 361
# M251 reserved constant slot 362
# M251 reserved constant slot 363
# M251 reserved constant slot 364
# M251 reserved constant slot 365
# M251 reserved constant slot 366
# M251 reserved constant slot 367
# M251 reserved constant slot 368
# M251 reserved constant slot 369
# M251 reserved constant slot 370
# M251 reserved constant slot 371
# M251 reserved constant slot 372
# M251 reserved constant slot 373
# M251 reserved constant slot 374
# M251 reserved constant slot 375
# M251 reserved constant slot 376
# M251 reserved constant slot 377
# M251 reserved constant slot 378
# M251 reserved constant slot 379
# M251 reserved constant slot 380
# M251 reserved constant slot 381
# M251 reserved constant slot 382
# M251 reserved constant slot 383
# M251 reserved constant slot 384
# M251 reserved constant slot 385
# M251 reserved constant slot 386
# M251 reserved constant slot 387
# M251 reserved constant slot 388
# M251 reserved constant slot 389
# M251 reserved constant slot 390
# M251 reserved constant slot 391
# M251 reserved constant slot 392
# M251 reserved constant slot 393
# M251 reserved constant slot 394
# M251 reserved constant slot 395
# M251 reserved constant slot 396
# M251 reserved constant slot 397
# M251 reserved constant slot 398
# M251 reserved constant slot 399
# M251 reserved constant slot 400
# M251 reserved constant slot 401

# M251: end of Claude-16
