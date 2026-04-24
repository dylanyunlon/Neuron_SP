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



# M286: Figure 3-7 specs
DESLOC_FIG3_SPEC = {"title":"Half-life vs Change Rate","type":"scatter"}
DESLOC_FIG4_SPEC = {"title":"Sync Sensitivity","type":"grouped_bar"}
DESLOC_FIG5_SPEC = {"title":"Scaling to Large Models","type":"grouped_bar"}
DESLOC_FIG6_SPEC = {"title":"Nesterov vs Averaging","type":"bar"}
DESLOC_FIG7_SPEC = {"title":"Adam vs Muon","type":"bar"}
DESLOC_ALL_FIGS = ["figure1_loss_curve","figure2_comm_cumulative",
    "figure3_halflife","figure4_sync_sensitivity","figure5_scaling",
    "figure6_nesterov","figure7_optimizer"]

# M300 — Claude-19: Protocol v2 + Trainium2/NKI + Megatron map
DESLOC_V2=2;DESLOC_MN_KX=1;DESLOC_MX_KX=4096;DESLOC_DKX=32;DESLOC_DKU_R=3;DESLOC_DKV_R=6
DESLOC_DRHO=1.;DESLOC_DWU=512;DESLOC_DOUTER='average';DESLOC_DNM=.9
DESLOC_PSI_S='4(1-px)/px^2*(1-b)(1-pu)/(6(1-(1-pu)*b))'
DESLOC_HL={'b1_09':6.579,'b1_095':13.513,'b2_095':13.513,'b2_099':68.968,'b2_0999':692.802,'b2_09999':6931.472}
DESLOC_OPTS=['Adam','AdamW','ADOPT','SGDM','Muon','LAMB']
DESLOC_INNER=['Adam','AdamW','ADOPT','Muon']
DESLOC_OUTER=['average','nesterov']
DESLOC_COMM={'param':'allreduce','momentum':'allreduce','variance':'allreduce','gradient':'reduce_scatter','reconstruct':'allgather'}
DESLOC_WSD={'warmup':.05,'stable':.8,'decay':.15,'type':'cosine'}
DESLOC_BENCH={'models':['125M','350M','1.3B','2.7B'],'Kx':[1,2,4,8,16,32,64,128,256],'seeds':[42,137,2024],'steps':5000}
DESLOC_ACC={'cuda':{'cc':(7,0),'dt':['bf16','fp16','fp32'],'be':'nccl'},'neuron':{'dt':['bf16','fp32'],'be':'xla','nki':True},'tpu':{'dt':['bf16','fp32'],'be':'xla'}}
DESLOC_NEURON={'tile':128,'pdim':0,'pipe':4,'efa_bw':400,'trn2_hbm':96,'trn2_tf':380,'cc':['cc.allreduce','cc.reduce_scatter','cc.allgather']}
DESLOC_MEGA={'tp':{'rel':'none'},'pp':{'rel':'partial','note':'Kx=N*micro_batches'},'dp':{'rel':'full'},'cp':{'rel':'none'},'ep':{'rel':'partial','note':'MoE capacity NOT gated'}}
DESLOC_FIG={'dpi':300,'font':'serif','fs':11,'w':6.5,'h':4.}
DESLOC_PAL={'DDP':'#2196F3','LocalAdam':'#FF9800','DESLOC':'#4CAF50','DESLOC_nesterov':'#9C27B0','Muon':'#F44336'}
DESLOC_CK='desloc';DESLOC_KXK='Kx';DESLOC_KUK='Ku';DESLOC_KVK='Kv';DESLOC_CLK='clip_rho';DESLOC_WUK='warmup_steps';DESLOC_OUK='outer_optimizer';DESLOC_ENK='enabled'
# M300: end


# =====================================================================
# M336 — Claude-30: NCCL Protocol Constants
# Source: nccl/src/device/all_reduce.h (RunWorkColl), nccl/src/include/collectives.h
# Ref: NCCL Ring AllReduce chunk/slice step protocol for DES-LOC gated comm
# =====================================================================

# NCCL Ring AllReduce protocol parameters (from nccl/src/include/collectives.h:19-20)
# NCCL_STEPS = 8 (default pipeline depth in NCCL ring protocol)
DESLOC_NCCL_STEPS = 8
DESLOC_ALLREDUCE_SLICESTEPS = DESLOC_NCCL_STEPS // 4  # = 2, matches NCCL
DESLOC_ALLREDUCE_CHUNKSTEPS = DESLOC_NCCL_STEPS // 2  # = 4, matches NCCL
DESLOC_ALLREDUCE_SLICE_PER_CHUNK = DESLOC_ALLREDUCE_CHUNKSTEPS // DESLOC_ALLREDUCE_SLICESTEPS

# NCCL algorithm selection (from nccl/src/device/all_reduce.h:234-777)
# Maps to RunWorkColl<ncclFuncAllReduce, T, RedOp, ALGO, PROTO> specializations
DESLOC_NCCL_ALGO_RING = 0       # Ring AllReduce — best for small clusters, NVLink
DESLOC_NCCL_ALGO_TREE = 1       # Tree AllReduce — best for large clusters, IB
DESLOC_NCCL_ALGO_COLLNET = 2    # CollNet Direct — SHARP-capable networks
DESLOC_NCCL_ALGO_NVLS = 3       # NVLink SHARP — intra-node NVLink multicast
DESLOC_NCCL_ALGO_NVLS_TREE = 4  # NVLS + Tree — hybrid intra/inter-node
DESLOC_NCCL_ALGO_COLLNET_CHAIN = 5  # CollNet Chain — chained SHARP

# NCCL protocol selection (from nccl/src/device/all_reduce.h:756-777)
DESLOC_NCCL_PROTO_SIMPLE = 0    # Simple protocol — high throughput
DESLOC_NCCL_PROTO_LL = 1        # Low-Latency protocol — sub-microsecond
DESLOC_NCCL_PROTO_LL128 = 2     # LL128 — 128-byte low-latency for small msgs

# NCCL ncclInfo fields (from nccl/src/collectives.cc:111-120)
# struct ncclInfo for AllReduce dispatch
DESLOC_COMM_INFO_FIELDS = {
    'coll': 'ncclFuncAllReduce',
    'name': 'AllReduce',
    'sendbuff': 'const void*',
    'recvbuff': 'void*',
    'count': 'size_t',
    'datatype': 'ncclDataType_t',
    'op': 'ncclRedOp_t',
    'root': 'int',
    'comm': 'ncclComm*',
    'stream': 'cudaStream_t',
}

# DES-LOC tier→algorithm mapping for gated AllReduce
# Ring for intra-node (param tier), Tree for inter-node (momentum tiers)
DESLOC_TIER_ALGO_MAP = {
    0: DESLOC_NCCL_ALGO_RING,    # param sync: Ring (low latency)
    1: DESLOC_NCCL_ALGO_TREE,    # momentum1: Tree (bandwidth-optimal)
    2: DESLOC_NCCL_ALGO_TREE,    # momentum2: Tree (infrequent, bulk)
}
DESLOC_TIER_PROTO_MAP = {
    0: DESLOC_NCCL_PROTO_SIMPLE,   # param: throughput matters
    1: DESLOC_NCCL_PROTO_SIMPLE,   # momentum1: large tensors
    2: DESLOC_NCCL_PROTO_LL128,    # momentum2: rare, can use LL
}

# =====================================================================
# M336 — Claude-30: Megatron Bucket Sizing Constants
# Source: Megatron-LM/megatron/core/distributed/distributed_data_parallel_config.py:46-51
# Ref: _ParamAndGradBuffer bucket configuration for DES-LOC overlap
# =====================================================================

# Bucket sizing (from Megatron DistributedDataParallelConfig.bucket_size)
# Megatron default: max(40_000_000, 1_000_000 * dp_size) parameters per bucket
DESLOC_DEFAULT_BUCKET_SIZE = 40_000_000  # 40M params minimum bucket
DESLOC_BUCKET_SIZE_PER_DP_RANK = 1_000_000  # 1M params per DP rank scaling
DESLOC_MIN_BUCKET_SIZE = 1_000_000  # 1M minimum to avoid latency-bound comms

# Bucket padding for NCCL bus bandwidth (from Megatron config:51)
# pad_buckets_for_high_nccl_busbw: divisible by 2^16 = 65536
DESLOC_BUCKET_PAD_ALIGNMENT = 65536  # 2^16, ensures NCCL high busbw
DESLOC_BUCKET_PAD_ENABLED_DEFAULT = False

# DES-LOC per-tier bucket sizing strategy
# Different tiers have different communication patterns:
# - Param tier (Kx): frequent small syncs → smaller buckets for overlap
# - Mom1 tier (Ku): medium frequency → standard buckets
# - Mom2 tier (Kv): infrequent large syncs → larger buckets for throughput
DESLOC_TIER_BUCKET_SCALE = {
    0: 0.5,   # param: half standard size for better overlap
    1: 1.0,   # mom1: standard bucket size
    2: 2.0,   # mom2: double for throughput (rare sync)
}

# Bucket gradient reduce precision (from Megatron config:13-14)
DESLOC_GRAD_REDUCE_FP32 = "fp32"
DESLOC_GRAD_REDUCE_BF16 = "bf16"
DESLOC_GRAD_REDUCE_FP16 = "fp16"
DESLOC_GRAD_REDUCE_DEFAULT = DESLOC_GRAD_REDUCE_BF16

# Overlap modes (from Megatron config:16-19)
DESLOC_OVERLAP_GRAD_REDUCE = True   # overlap grad reduce with backward
DESLOC_OVERLAP_PARAM_GATHER = False  # param gather overlap (ZeRO-3)
DESLOC_ALIGN_PARAM_GATHER = False    # align across PP stages

# =====================================================================
# M336 — Claude-30: AutoSP + DES-LOC Integration Constants
# Source: neuronx-distributed/src/.../optimizer/zero_redundancy_optimizer.py:96
#         Megatron-LM/megatron/core/optimizer/emerging_optimizers.py:154
# Ref: AutoSP (sequence parallel) works in ZeRO stage 0, DES-LOC also stage 0/1
#      AutoSP cuts along sequence dim; DES-LOC gates along worker dim → orthogonal
# =====================================================================

# AutoSP compatibility flags
DESLOC_AUTOSP_COMPAT = "autosp_compat"
DESLOC_AUTOSP_COMPAT_DEFAULT = True  # DES-LOC compatible with AutoSP by default

# AutoSP operates in ZeRO stage 0, DES-LOC in stage 0/1 — confirmed compatible
DESLOC_AUTOSP_ZERO_STAGE = 0
DESLOC_ZERO_COMPAT_STAGES = (0, 1)  # DES-LOC works with ZeRO stage 0 and 1

# Sequence parallel dimension config
DESLOC_AUTOSP_SEQ_DIM = 1  # AutoSP splits along sequence (dim=1 in [B, S, H])
DESLOC_WORKER_DIM = 0      # DES-LOC gates along worker/data-parallel dim (dim=0)

# AutoSP + DES-LOC combined: each worker does Kx independent SP forward/backward,
# then AllReduces at the Kx boundary
DESLOC_AUTOSP_SYNC_MODE = "boundary"  # sync only at Kx boundaries
DESLOC_AUTOSP_OVERLAP_SP_COMM = True  # overlap SP comm with DES-LOC local steps

# Muon optimizer integration (from Megatron emerging_optimizers.py:154)
# TensorParallelMuon uses Newton-Schulz orthogonalization — DES-LOC ρ-clipping
# must apply BEFORE orthogonalization to ensure bounded divergence
DESLOC_MUON_CLIP_BEFORE_ORTH = True   # clip before Newton-Schulz
DESLOC_MUON_NS_STEPS = 5              # Newton-Schulz iterations (Megatron default)
DESLOC_MUON_COEFF_TYPE = "quintic"    # coefficient type (Megatron default)
DESLOC_MUON_SCALE_MODE = "spectral"   # scaling mode for orthogonalized momentum
DESLOC_MUON_MOMENTUM = 0.95           # Muon momentum (Megatron default)
DESLOC_MUON_NESTEROV = True           # Muon uses Nesterov by default

# =====================================================================
# M336 — Claude-30: Distributed Experiment Scheduling Constants
# Source: Megatron-LM/megatron/core/pipeline_parallel/schedules.py
# Ref: 流浪地球计划 — maximize GPU utilization across heterogeneous clusters
# =====================================================================

# GPU instance types for distributed scheduling
DESLOC_GPU_INSTANCES = {
    'sgn8ia_m2_xlarge': {
        'vcpu': 4, 'mem_gib': 16, 'vgpu_mem_gb': 2,
        'cost_per_hour': 1.09727, 'gpu_type': 'vGPU8-2G'
    },
    'sgn8ia_m4_2xlarge': {
        'vcpu': 8, 'mem_gib': 32, 'vgpu_mem_gb': 48,
        'cost_per_hour': 24.398771, 'gpu_type': 'vGPU'
    },
    'gn8v_4xlarge': {
        'vcpu': 16, 'mem_gib': 96, 'gpu_mem_gb': 96,
        'cost_per_hour': 36.3852, 'gpu_type': 'H20', 'n_gpu': 1
    },
    'gn8v_6xlarge': {
        'vcpu': 24, 'mem_gib': 128, 'gpu_mem_gb': 96,
        'cost_per_hour': 38.4891, 'gpu_type': 'H20', 'n_gpu': 1
    },
    'gn8v_2x_8xlarge': {
        'vcpu': 32, 'mem_gib': 192, 'gpu_mem_gb': 192,
        'cost_per_hour': 71.5803, 'gpu_type': 'H20', 'n_gpu': 2
    },
    'gn8v_2x_12xlarge': {
        'vcpu': 48, 'mem_gib': 256, 'gpu_mem_gb': 192,
        'cost_per_hour': 76.9781, 'gpu_type': 'H20', 'n_gpu': 2
    },
}
DESLOC_SELECTED_INSTANCE = 'gn8v_2x_8xlarge'  # Current: 32vCPU, 192GiB, 2xGPU_H

# Experiment matrix for distributed scheduling
# 流浪地球计划: maximize experiments per hour across cluster
DESLOC_EXP_MATRIX_MODELS = ['125M', '350M', '700M', '1.3B']
DESLOC_EXP_MATRIX_KX = [1, 2, 4, 8, 16, 32, 64]
DESLOC_EXP_MATRIX_SEEDS = [42, 137, 2024]
DESLOC_EXP_MATRIX_METHODS = ['DDP', 'LocalAdam', 'DESLOC', 'DESLOC_nesterov']
DESLOC_EXP_TOTAL_CONFIGS = (
    len(DESLOC_EXP_MATRIX_MODELS) * len(DESLOC_EXP_MATRIX_KX) *
    len(DESLOC_EXP_MATRIX_SEEDS) * len(DESLOC_EXP_MATRIX_METHODS)
)

# Experiment scheduling priorities (higher = run first)
DESLOC_EXP_PRIORITY = {
    'DDP_125M': 10,       # baseline, fast
    'DESLOC_125M': 9,     # main comparison
    'DDP_350M': 8,        # medium baseline
    'DESLOC_350M': 7,     # medium comparison
    'DDP_700M': 6,        # large baseline
    'DESLOC_700M': 5,     # large comparison — M335 results available
    'DDP_1.3B': 4,        # largest baseline
    'DESLOC_1.3B': 3,     # largest comparison
}

# Time budget per experiment (seconds) — from M335 empirical data
DESLOC_EXP_TIME_BUDGET = {
    '125M': 300,    # ~5 min per seed
    '350M': 600,    # ~10 min per seed
    '700M': 1200,   # ~20 min per seed
    '1.3B': 3600,   # ~60 min per seed
}

# Convergence validation thresholds (from M335 results)
# DESLOC 7.31±0.49 vs DDP 7.67±0.05 at 700M
DESLOC_CONVERGENCE_LOSS_THRESHOLD = 10.0  # loss above this = diverged
DESLOC_CONVERGENCE_MIN_IMPROVEMENT = 0.01  # min loss decrease over 100 steps
DESLOC_CONVERGENCE_PARITY_TOLERANCE = 0.1  # DESLOC loss within this of DDP = parity

# Warmup u-sync fix constants (from M335: warmup u-sync 100→5)
DESLOC_WARMUP_U_SYNC_STEPS = 5  # was 100, reduced to save 95 AllReduces
DESLOC_WARMUP_U_SYNC_OLD = 100  # old value for reference
DESLOC_WARMUP_ALLREDUCE_SAVINGS = 95  # AllReduces saved by this fix

# =====================================================================
# M336 — Claude-30: Gradient Clipping Protocol Constants
# Source: Megatron-LM/megatron/core/optimizer/clip_grads.py:57
#         neuronx-distributed/src/.../optimizer/zero_redundancy_optimizer.py:96
# Ref: Per-bucket gradient norm + per-tier clipping for DES-LOC
# =====================================================================

# Gradient norm computation (from Megatron clip_grads.py:57 get_grad_norm_fp32)
DESLOC_GRAD_NORM_TYPE = 2.0          # L2 norm (default, matches Megatron)
DESLOC_GRAD_NORM_INF = float('inf')  # Infinity norm option
DESLOC_GRAD_NORM_CLIP_EPS = 1.0e-6   # Epsilon for clip coefficient (Megatron uses same)

# Per-tier gradient clipping (adapted from clip_grad_by_total_norm_fp32)
# Different tiers need different clipping thresholds because:
# - Param tier: gradients change fast → tighter clip
# - Mom1 tier: momentum smooths gradients → medium clip
# - Mom2 tier: variance very stable → loose clip
DESLOC_TIER_CLIP_MAX_NORM = {
    0: 1.0,    # param tier: standard gradient clipping
    1: 2.0,    # mom1 tier: relaxed (momentum dampens spikes)
    2: 5.0,    # mom2 tier: very relaxed (variance is stable)
}

# ZeRO-1 + DES-LOC grad clip gate (from neuronx zero_redundancy_optimizer.py:96)
# _clip_grad_norm() is called per-step; with DES-LOC Kx gating,
# we only clip at Kx boundaries to avoid clipping stale gradients
DESLOC_CLIP_AT_KX_BOUNDARY_ONLY = True
DESLOC_CLIP_ACCUMULATE_BETWEEN = True  # accumulate norms between Kx boundaries

# =====================================================================
# M336 — Claude-30: Comm/Compute Overlap Constants
# Source: TransformerEngine/.../userbuffers_forward_linear.py:175
#         Megatron-LM/megatron/core/distributed/distributed_data_parallel.py
# Ref: Overlap communication with computation for DES-LOC gated sync
# =====================================================================

# Userbuffers overlap (from TransformerEngine userbuffers_forward_linear.py)
DESLOC_UB_OVERLAP_RS = True    # overlap reduce-scatter with forward
DESLOC_UB_OVERLAP_AG = True    # overlap all-gather with forward
DESLOC_UB_COMM_SM = 16         # SMs reserved for communication (default)
DESLOC_UB_COMM_SM_SHARP = 6   # SMs with SHARP enabled

# DES-LOC overlap strategy: during Kx local steps, overlap current AllReduce
# with next local step's forward pass
DESLOC_OVERLAP_AR_WITH_FORWARD = True
DESLOC_OVERLAP_STREAM_PRIORITY = -1  # high priority for comm stream

# Async AllReduce enqueue (adapted from nccl/src/enqueue.cc ncclEnqueueCheck)
DESLOC_ASYNC_AR_ENABLED = True
DESLOC_ASYNC_AR_MAX_PENDING = 4  # max pending async AllReduces before sync

# =====================================================================
# M336 — Claude-30: Roofline Model Constants
# Source: TransformerEngine/benchmarks/attention/benchmark_dot_product_attention.py
#         flash-attention/flash_attn/utils/benchmark.py:8,30,258
# Ref: MFU calculation for DES-LOC efficiency reporting
# =====================================================================

# Roofline model parameters
DESLOC_ROOFLINE_OPERATIONAL_INTENSITY_THRESHOLD = 64.0  # ops/byte, GEMM crossover
DESLOC_ROOFLINE_MFU_TARGET_TRAIN = 0.35   # target MFU for training (35%)
DESLOC_ROOFLINE_MFU_TARGET_DESLOC = 0.40  # DES-LOC should improve MFU (+5%)
DESLOC_ROOFLINE_WARMUP_ITERS = 10  # warmup iterations before timing (FA benchmark)
DESLOC_ROOFLINE_BENCH_ITERS = 50   # benchmark iterations (FA benchmark)

# MFU computation: TFLOPS_achieved / TFLOPS_peak
# 6ND rule: compute per token ≈ 6 * N_params * D_tokens
DESLOC_MFU_COMPUTE_FACTOR = 6  # 6ND approximation
DESLOC_MFU_BACKWARD_FACTOR = 2  # backward is ~2x forward FLOPS

# Per-GPU peak TFLOPS for MFU denominator (extended from M300)
DESLOC_GPU_PEAK_TFLOPS = {
    'H100_SXM': {'bf16': 989.5, 'fp16': 989.5, 'fp32': 67.0, 'tf32': 495.0},
    'H100_NVL': {'bf16': 835.0, 'fp16': 835.0, 'fp32': 56.0, 'tf32': 417.5},
    'A100_SXM': {'bf16': 312.0, 'fp16': 312.0, 'fp32': 19.5, 'tf32': 156.0},
    'A100_PCIe': {'bf16': 312.0, 'fp16': 312.0, 'fp32': 19.5, 'tf32': 156.0},
    'A6000': {'bf16': 38.7, 'fp16': 38.7, 'fp32': 38.7, 'tf32': 19.35},
    'L40': {'bf16': 181.0, 'fp16': 181.0, 'fp32': 90.5, 'tf32': 90.5},
    'RTX_4090': {'bf16': 165.2, 'fp16': 165.2, 'fp32': 82.6, 'tf32': 82.6},
    'H20': {'bf16': 148.0, 'fp16': 148.0, 'fp32': 39.5, 'tf32': 74.0, 'fp8': 296.0},
    'Trainium2': {'bf16': 380.0, 'fp32': 95.0},
}

# =====================================================================
# M336 — Claude-30: NKI-FA Extended Log Format Constants
# Source: NKI-FA/exp_utils/draw_plot.py (commit da964f3)
# Ref: Extended log format for DES-LOC experiments beyond NKI-FA base format
# =====================================================================

# Extended NKI-FA log keys for DES-LOC experiments
NKIFA_DESLOC_KEYS = {
    'method': 'method',          # DDP/LocalAdam/DESLOC/DESLOC_nesterov
    'model_size': 'model_size',  # 125M/350M/700M/1.3B
    'kx': 'Kx',
    'ku': 'Ku',
    'kv': 'Kv',
    'seed': 'seed',
    'final_loss': 'final_loss',
    'best_loss': 'best_loss',
    'speedup': 'speedup',
    'mfu': 'mfu',
    'comm_reduction': 'comm_reduction',
    'total_allreduce': 'total_allreduce',
    'total_tokens': 'total_tokens',
    'wall_time_s': 'wall_time_s',
    'peak_mem_gb': 'peak_mem_gb',
}

# NKI-FA figure specifications for DES-LOC paper (NeurIPS format)
NKIFA_FIGURE_SPECS = {
    'figure1_loss_curve': {
        'type': 'line', 'x': 'step', 'y': 'loss',
        'hue': 'method', 'style': 'Kx',
        'width': 6.5, 'height': 4.0, 'dpi': 300,
    },
    'figure2_comm_cumulative': {
        'type': 'bar', 'x': 'method', 'y': 'total_allreduce',
        'hue': 'model_size',
        'width': 6.5, 'height': 3.5, 'dpi': 300,
    },
    'figure3_halflife': {
        'type': 'scatter', 'x': 'beta2', 'y': 'Kv',
        'annotate': True,
        'width': 5.0, 'height': 4.0, 'dpi': 300,
    },
    'figure4_sync_sensitivity': {
        'type': 'grouped_bar', 'x': 'Kx', 'y': 'final_loss',
        'hue': 'model_size',
        'width': 6.5, 'height': 4.0, 'dpi': 300,
    },
    'figure5_scaling': {
        'type': 'grouped_bar', 'x': 'model_size', 'y': 'final_loss',
        'hue': 'method', 'error_bar': 'std',
        'width': 6.5, 'height': 4.0, 'dpi': 300,
    },
    'figure6_nesterov': {
        'type': 'bar', 'x': 'model_size', 'y': 'final_loss',
        'hue': 'outer_opt',
        'width': 5.5, 'height': 4.0, 'dpi': 300,
    },
    'figure7_optimizer': {
        'type': 'bar', 'x': 'model_size', 'y': 'final_loss',
        'hue': 'inner_opt',
        'width': 5.5, 'height': 4.0, 'dpi': 300,
    },
}

# =====================================================================
# M336 — Claude-30: FP32 Accumulation Constants
# Source: Megatron-LM/megatron/core/distributed/reduce_scatter_with_fp32_accumulation.py
# Ref: DES-LOC uses FP32 accumulation at Kx boundaries for numerical stability
# =====================================================================

# FP32 accumulation for reduce-scatter (from Megatron reduce_scatter_with_fp32_accumulation.py:9)
DESLOC_RS_FP32_ACCUM_ENABLED = True  # use FP32 accumulation at sync boundaries
DESLOC_RS_FP32_ACCUM_THRESHOLD = 1000  # only for tensors > 1000 elements

# Reduce-scatter work handle tracking
DESLOC_RS_MAX_INFLIGHT = 4  # max in-flight reduce-scatter handles
DESLOC_RS_WAIT_TIMEOUT_MS = 30000  # 30s timeout for reduce-scatter completion

# =====================================================================
# M336 — Claude-30: Adaptive Kx Selection Constants
# Source: Megatron-LM/megatron/core/optimizer/emerging_optimizers.py:154 (TensorParallelMuon)
# Ref: Adaptive Kx based on gradient variance trend, analogous to Muon's
#      adaptive scale_factor = get_muon_scale_factor(size[0], size[1], mode)
# =====================================================================

# Adaptive Kx parameters
DESLOC_ADAPTIVE_KX_ENABLED = "adaptive_Kx"
DESLOC_ADAPTIVE_KX_ENABLED_DEFAULT = False
DESLOC_ADAPTIVE_KX_WINDOW = 50     # gradient variance window (steps)
DESLOC_ADAPTIVE_KX_INCREASE_THRESHOLD = 0.9  # variance ratio to increase Kx
DESLOC_ADAPTIVE_KX_DECREASE_THRESHOLD = 1.5  # variance ratio to decrease Kx
DESLOC_ADAPTIVE_KX_MIN = 2   # minimum adaptive Kx
DESLOC_ADAPTIVE_KX_MAX = 128  # maximum adaptive Kx
DESLOC_ADAPTIVE_KX_STEP_UP = 2    # multiply Kx by 2 when increasing
DESLOC_ADAPTIVE_KX_STEP_DOWN = 2  # divide Kx by 2 when decreasing

# Loss trend detection for Kx adjustment
DESLOC_LOSS_TREND_WINDOW = 100  # steps to compute loss trend
DESLOC_LOSS_PLATEAU_THRESHOLD = 0.001  # loss change below this = plateau
DESLOC_LOSS_SPIKE_THRESHOLD = 2.0  # loss increase ratio = spike

# =====================================================================
# M336 — Claude-30: veScale DTensor Integration Constants
# Source: veScale/vescale/dtensor/_collective_utils.py:66 mesh_scatter_ragged
# Ref: DTensor redistribute for heterogeneous DES-LOC workers
# =====================================================================

# DTensor redistribution for DES-LOC (from veScale _collective_utils.py)
DESLOC_DTENSOR_REDISTRIBUTE = True   # enable DTensor-style redistribute
DESLOC_DTENSOR_MESH_DIM = 0          # device mesh dimension for DP

# Ragged scatter support for heterogeneous worker loads
DESLOC_RAGGED_SCATTER_ENABLED = False  # enable ragged scatter (experimental)
DESLOC_RAGGED_SCATTER_ASYNC = True     # async ragged scatter

# veScale-style placement types for DES-LOC
DESLOC_PLACEMENT_REPLICATE = "replicate"   # all workers get full copy
DESLOC_PLACEMENT_SHARD = "shard"           # shard across workers
DESLOC_PLACEMENT_PARTIAL = "partial"       # partial reduction (DES-LOC default)

# =====================================================================
# M336 — Claude-30: Validation Functions (extending M251 validators)
# =====================================================================

def desloc_validate_bucket_size(bucket_size, dp_size):
    """Validate bucket size against dp_size scaling rules (Megatron pattern)."""
    min_size = max(DESLOC_MIN_BUCKET_SIZE, DESLOC_BUCKET_SIZE_PER_DP_RANK * dp_size)
    if bucket_size is not None and bucket_size < min_size:
        return False, (
            f"bucket_size={bucket_size} too small for dp_size={dp_size}, "
            f"minimum={min_size}"
        )
    return True, ""

def desloc_validate_autosp_compat(zero_stage, autosp_enabled):
    """Validate AutoSP + DES-LOC compatibility (both need ZeRO stage 0/1)."""
    if autosp_enabled and zero_stage not in DESLOC_ZERO_COMPAT_STAGES:
        return False, (
            f"AutoSP requires ZeRO stage {DESLOC_AUTOSP_ZERO_STAGE}, "
            f"DES-LOC supports stages {DESLOC_ZERO_COMPAT_STAGES}, "
            f"but got stage {zero_stage}"
        )
    return True, ""

def desloc_compute_bucket_size(n_params, dp_size, tier=0):
    """Compute tier-aware bucket size using Megatron scaling + DES-LOC tier factor."""
    base = max(DESLOC_DEFAULT_BUCKET_SIZE, DESLOC_BUCKET_SIZE_PER_DP_RANK * dp_size)
    scale = DESLOC_TIER_BUCKET_SCALE.get(tier, 1.0)
    raw = int(base * scale)
    if DESLOC_BUCKET_PAD_ENABLED_DEFAULT:
        raw = ((raw + DESLOC_BUCKET_PAD_ALIGNMENT - 1)
               // DESLOC_BUCKET_PAD_ALIGNMENT * DESLOC_BUCKET_PAD_ALIGNMENT)
    return min(raw, n_params)

def desloc_estimate_exp_cost(model_size, n_seeds, n_kx_values, instance_key=None):
    """Estimate total GPU-hours and cost for an experiment sweep."""
    if instance_key is None:
        instance_key = DESLOC_SELECTED_INSTANCE
    inst = DESLOC_GPU_INSTANCES.get(instance_key)
    if inst is None:
        return 0.0, 0.0
    time_per_seed = DESLOC_EXP_TIME_BUDGET.get(model_size, 1800)
    total_seconds = time_per_seed * n_seeds * n_kx_values * len(DESLOC_EXP_MATRIX_METHODS)
    total_hours = total_seconds / 3600.0
    total_cost = total_hours * inst['cost_per_hour']
    return total_hours, total_cost

def desloc_select_tier_algo(tier, n_workers):
    """Select NCCL algorithm per tier based on worker count."""
    if n_workers <= 8:
        return DESLOC_NCCL_ALGO_RING   # Ring for small clusters
    return DESLOC_TIER_ALGO_MAP.get(tier, DESLOC_NCCL_ALGO_TREE)

def desloc_compute_mfu(achieved_tflops, gpu_name, dtype='bf16'):
    """Compute Model FLOPS Utilization given achieved TFLOPS."""
    peak = DESLOC_GPU_PEAK_TFLOPS.get(gpu_name, {}).get(dtype, 312.0)
    if peak <= 0:
        return 0.0
    return achieved_tflops / peak

def desloc_grad_norm_fp32(grads, norm_type=2.0):
    """Pure-Python gradient norm in FP32 (Megatron clip_grads.py:57 pattern).

    For use in environments without apex/TE multi_tensor_applier.
    In production, the CUDA fused version should be used instead.
    """
    import torch as _torch
    if norm_type == float('inf'):
        return max(g.abs().max().item() for g in grads) if grads else 0.0
    total = _torch.zeros(1, dtype=_torch.float32,
                         device=grads[0].device if grads else 'cpu')
    for g in grads:
        gf = g.float() if g.dtype != _torch.float32 else g
        total += gf.norm(norm_type) ** norm_type
    return total.item() ** (1.0 / norm_type)

def desloc_clip_grad_per_tier(params, max_norm, tier, total_norm=None):
    """Per-tier gradient clipping (adapted from Megatron clip_grad_by_total_norm_fp32).

    Each tier has its own max_norm threshold from DESLOC_TIER_CLIP_MAX_NORM.
    """
    import torch as _torch
    tier_max = DESLOC_TIER_CLIP_MAX_NORM.get(tier, max_norm)
    if total_norm is None:
        grads = [p.grad for p in params if p.grad is not None]
        total_norm = desloc_grad_norm_fp32(grads) if grads else 0.0
    clip_coeff = tier_max / (total_norm + DESLOC_GRAD_NORM_CLIP_EPS)
    if clip_coeff < 1.0:
        for p in params:
            if p.grad is not None:
                p.grad.mul_(clip_coeff)
    return total_norm
