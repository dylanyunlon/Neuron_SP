# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import sys
from typing import Optional, Dict, Any
from enum import Enum
from pydantic import Field, model_validator
from deepspeed.runtime.config_utils import get_scalar_param, pp_int, DeepSpeedConfigModel
from deepspeed.utils import logger
from .offload_config import DeepSpeedZeroOffloadParamConfig, DeepSpeedZeroOffloadOptimizerConfig, OffloadDeviceEnum
from deepspeed.runtime.zenflow.zenflow_config import ZenFlowConfig
from .leaf_module_config import DeepSpeedZeroLeafModuleConfig

# ZeRO optimization. By default, this optimization is not enabled.
# Users have to configure the desired optimization (0 means disabled) in params.json as below example:
ZERO_FORMAT = """
ZeRO optimization should be enabled as:
"session_params": {
  "zero_optimization": {
    "stage": [0|1|2],
    "stage3_max_live_parameters" : 1000000000,
    "stage3_max_reuse_distance" : 1000000000,
    "stage3_use_all_reduce_for_fetch_params": [true|false],
    "stage3_module_granularity_threshold": 0,
    "allgather_partitions": [true|false],
    "use_multi_rank_bucket_allreduce": [true|false],
    "stage3_allgather_sequential": [true|false],
    "allgather_bucket_size": 500000000,
    "reduce_scatter": [true|false],
    "contiguous_gradients" : [true|false]
    "overlap_comm": [true|false],
    "reduce_bucket_size": 500000000,
    "load_from_fp32_weights": [true|false],
    "cpu_offload": [true|false] (deprecated),
    "cpu_offload_param" : [true|false] (deprecated),
    "cpu_offload_use_pin_memory": [true|false] (deprecated),
    "sub_group_size" : 1000000000000,
    "offload_param": {...},
    "offload_optimizer": {...},
    "ignore_unused_parameters": [true|false],
    "round_robin_gradients": [true|false],
    "zero_hpz_partition_size": 1,
    "zero_quantized_weights": [true|false],
    "zero_quantized_nontrainable_weights": [true|false],
    "zero_quantized_gradients": [true|false],
    "memory_efficient_linear": [true|false],
    "override_module_apply": [true|false],
    "zeropp_loco_param": {...},
    "log_trace_cache_warnings" : [true|false],
    "enable_sanity_checks": [true|false],
    }
}
"""

ZERO_OPTIMIZATION = "zero_optimization"


def read_zero_config_deprecated(param_dict):
    zero_config_dict = {}
    zero_config_dict["stage"] = 1 if param_dict[ZERO_OPTIMIZATION] else 0
    if zero_config_dict["stage"] > 0:
        zero_config_dict["allgather_bucket_size"] = get_scalar_param(param_dict, "allgather_size", 5e8)
    logger.warning(
        "DeepSpeedConfig: this format of ZeRO optimization setup is deprecated. Please use the following format: {}".
        format(ZERO_FORMAT))
    return zero_config_dict


def get_zero_config(param_dict):
    if ZERO_OPTIMIZATION in param_dict:
        zero_config_dict = param_dict[ZERO_OPTIMIZATION]
        if isinstance(zero_config_dict, bool):
            zero_config_dict = read_zero_config_deprecated(param_dict)
    else:
        zero_config_dict = {}
    return DeepSpeedZeroConfig(**zero_config_dict)


class ZeroStageEnum(int, Enum):
    """ Enum class for possible zero stages """
    disabled = 0
    optimizer_states = 1
    gradients = 2
    weights = 3
    max_stage = 3


class DeepSpeedZeroConfig(DeepSpeedConfigModel):
    """
    Sets parameters for ZeRO optimizations.
    """

    stage: ZeroStageEnum = 0
    """
    Chooses different stages of ZeRO Optimizer. Stage 0, 1, 2, and 3 refer
    to disabled, optimizer state partitioning, and optimizer+gradient state
    partitioning, and optimizer+gradient+parameter partitioning, respectively.
    """

    contiguous_gradients: bool = True
    """
    Copies the gradients to a contiguous buffer as they are produced. Avoids
    memory fragmentation during backward pass.
    """

    reduce_scatter: bool = True
    """
    Uses reduce or reduce scatter instead of allreduce to average gradients
    """

    reduce_bucket_size: int = Field(pp_int(5e8), ge=0)
    """
    Number of elements reduced/allreduced at a time. Limits the memory required
    for the allgather for large model sizes
    """

    use_multi_rank_bucket_allreduce: bool = True
    """
    Combine the reduce buckets of the different ranks and do an All-Reduce instead of multiple Reduce ops.
    This feature is useful when the model is small and we want to scale it on too many GPUs which therefore
    reduces the message sizes of each packet.
    """

    allgather_partitions: bool = True
    """
    Chooses between allgather collective or a series of broadcast collectives
    to gather updated parameters from all the GPUs at the end of each step
    """

    allgather_bucket_size: int = Field(pp_int(5e8), ge=0)
    """
    Number of elements allgathered at a time. Limits the memory required for
    the allgather for large model sizes
    """

    overlap_comm: Optional[bool] = None  # None for dynamic default value (see validator `overlap_comm_valid` below)
    """
    Attempts to overlap the reduction of the gradients with backward computation
    """

    load_from_fp32_weights: bool = True
    """
    Boolean indicating whether to initialize fp32 master weights from fp32
    copies in checkpoint (no precision loss) or from model's fp16 copies (with
    precision loss). This can be used to initialize optimizer state even when
    checkpoint is missing optimizer state.
    """

    elastic_checkpoint: bool = False
    """
    Enable loading checkpoint that was saved by job with different GPU count.
    No longer supported.
    """

    offload_param: Optional[DeepSpeedZeroOffloadParamConfig] = None
    """
    Enable offloading of model parameters to CPU or NVMe. This frees up GPU
    memory for larger models or batch sizes. Valid only with stage 3. Expects a
    dictionary containing values for :any:`DeepSpeedZeroOffloadParamConfig`.
    """

    offload_optimizer: Optional[DeepSpeedZeroOffloadOptimizerConfig] = None
    """
    Enable offloading of optimizer state to CPU or NVMe, and optimizer
    computation to CPU. This frees up GPU memory for larger models or batch
    sizes. Valid for ZeRO stage 1, 2, 3. Expects a dictionary containing values
    for :any:`DeepSpeedZeroOffloadOptimizerConfig`.
    """

    zenflow: Optional[ZenFlowConfig] = None
    """Enable ZenFlow"""

    sub_group_size: int = Field(pp_int(1e9), ge=0)
    """
    Tile size for parameter processing to fit massive models (with trillions of
    parameters). Used by ZeRO3-Offload and ZeRO-Infinity
    """

    cpu_offload_param: Optional[bool] = Field(
        None,
        json_schema_extra={
            "deprecated": True,
            "new_param": "offload_param",
            "new_param_fn": (lambda val: DeepSpeedZeroOffloadParamConfig(device=OffloadDeviceEnum.cpu)
                             if val else None)
        },
    )
    """ Deprecated, please use ``offload_param`` """

    cpu_offload_use_pin_memory: Optional[bool] = Field(
        None,
        json_schema_extra={
            "deprecated": True,
            "new_param": "offload_param or offload_optimizer",
            "set_new_param": False
        },
    )
    """ Deprecated, please use ``offload_param`` or ``offload_optimizer`` """

    cpu_offload: Optional[bool] = Field(
        None,
        json_schema_extra={
            "deprecated":
            True,
            "new_param":
            "offload_optimizer",
            "new_param_fn": (lambda val: DeepSpeedZeroOffloadOptimizerConfig(device=OffloadDeviceEnum.cpu)
                             if val else None)
        },
    )
    """ Deprecated, please use ``offload_optimizer`` """

    prefetch_bucket_size: int = Field(pp_int(5e7), ge=0, alias="stage3_prefetch_bucket_size")
    """
    Maximum number of parameter elements to fetch ahead of use. Used by ZeRO3,
    ZeRO3-Offload, ZeRO-Infinity, and ZeRO-Inference.
    """

    param_persistence_threshold: int = Field(pp_int(1e5), ge=0, alias="stage3_param_persistence_threshold")
    """
    Do not partition parameters smaller than this threshold. Smaller values use
    less memory, but can greatly increase communication (especially
    latency-bound messages).
    """

    model_persistence_threshold: int = Field(pp_int(sys.maxsize, "sys.maxsize"),
                                             ge=0,
                                             alias="stage3_model_persistence_threshold")
    """
    Maximum number of parameter elements that can be persisted in GPU and not
    partitioned. This imposes an upper bound on the number of unpartitioned
    parameters resulting from param_persistence_threshold setting. Used by
    ZeRO3-Offload, ZeRO-Infinity and ZeRO-Inference.
    """

    max_live_parameters: int = Field(pp_int(1e9), ge=0, alias="stage3_max_live_parameters")
    """
    The maximum number of parameters resident per GPU before releasing. Smaller
    values use less memory, but perform more communication.
    """

    max_reuse_distance: int = Field(pp_int(1e9), ge=0, alias="stage3_max_reuse_distance")
    """
    Do not release a parameter if it will be reused within this threshold of
    parameters. Smaller values use less memory, but perform more communication.
    """

    gather_16bit_weights_on_model_save: bool = Field(False, alias="stage3_gather_16bit_weights_on_model_save")
    """
    Consolidate the weights before saving the model by ``save_16bit_model()``.
    Since the weights are partitioned across GPUs, they aren’t part of
    ``state_dict``, so this function automatically gathers the weights when
    this option is enabled and then saves the fp16 model weights.
    """

    module_granularity_threshold: int = Field(pp_int(0), alias="stage3_module_granularity_threshold")
    """
    The granularity of a module is determined by the ratio of "parameter_count / (1 + descendant count)".
    ZeRO3 classifies modules with a granularity below the threshold as fine-grained,
    which are treated as integral units during parameter fetching. This reduces host overhead
    and the separate allgather overhead introduced by hooks for fine-grained layers when fetching parameters.
    """

    use_all_reduce_for_fetch_params: bool = Field(False, alias="stage3_use_all_reduce_for_fetch_params")
    """
    Use all_reduce op when fetching module parameters at stage3. This improves performance by reducing
    the overhead of concatenation and slicing on the host.
    """

    allgather_sequential: bool = Field(default=False, alias="stage3_allgather_sequential")
    """
    Performs allgather on individual parameters sequentially, bypassing the standard parameter bucketing
    mechanism in stage3. This significantly reduces data copy overhead (eliminating copy-to-bucket operations)
    and lowers peak memory usage by avoiding the allocation of large temporary flattening buffers.
    Recommended for scenarios with high memory pressure.
    """

    stage3_gather_fp16_weights_on_model_save: bool = Field(False,
                                                           json_schema_extra={
                                                               "deprecated": True,
                                                               "new_param": "gather_16bit_weights_on_model_save"
                                                           })
    """ Deprecated, please use ``gather_16bit_weights_on_model_save`` """

    ignore_unused_parameters: bool = True
    """
    Unused parameters in modules may be unexpected in static networks, but
    could be normal in dynamic networks. This controls whether or not training
    should terminate with an error message when unused parameters are detected.
    This is set to ``True`` by default, which means unused parameters are
    ignored and training continues. Now is just used in stage 2.
    """

    legacy_stage1: bool = False
    """
    For backward-compatibility enable old ZeRO stage 1 implementation. Use at
    your own risk, will be deprecated soon.
    """

    round_robin_gradients: bool = False
    """
    Stage 1 and 2 optimization for CPU offloading that parallelizes gradient
    copying to CPU memory among ranks by fine-grained gradient partitioning.
    Performance benefit grows with gradient accumulation steps (more copying
    between optimizer steps) or GPU count (increased parallelism).
    """
    zero_hpz_partition_size: int = Field(1, ge=0)
    """
    Number of ranks in zero parameters partitioning secondary group
    """
    zero_quantized_weights: bool = False
    """
    Boolean indicating whether to quantize zero parameters (weights)
    for efficient all_gather comm
    """
    zero_quantized_nontrainable_weights: bool = False
    """
    Boolean indicating whether to quantize non-trainable zero parameters (weights)
    for efficient memory usage and communication. Different from zero_quantized_weights
    that stores the weights in original precision and only perform quantization during communication,
    this flag will store the weights in quantized precision. This is useful for LoRA training.
    """
    zero_quantized_gradients: bool = False
    """
    Boolean indicating whether to use quantized zero gradients
    for efficient all_2_all_reduce comm
    """
    zeropp_loco_param: Optional[Dict[str, Any]] = None
    """
    This dictionary contains parameters for using LoCo-Zero++, with two key parameters:
    - `err_beta`: A coefficient for the moving average of quantization errors before and after gradient computation.
    It ranges between 0 and 1, with a default value of 0.8.
    - `reset_T`: The number of steps after which the moving-average error buffer is cleared. The default value is 1024.
    These parameters can be adjusted based on performance needs. Example configuration in ds config:
    "zeropp_loco_param": { "err_beta": 0.8, "reset_T": 1024 }.
    See LoCo paper for more details: (https://arxiv.org/abs/2407.04480).
    """

    mics_shard_size: int = Field(-1, json_schema_extra={"new_param": "mics_shard_size"})

    mics_hierarchical_params_gather: bool = False

    memory_efficient_linear: bool = True
    """
    Use memory efficient linear implementation, for Stage 3.
    """
    """
    Whether force load checkpoint in pipeline mode, current only for Stage 3.
    """
    pipeline_loading_checkpoint: bool = False

    override_module_apply: bool = True
    """
    Override nn.Module apply function, for Stage 3.
    """

    log_trace_cache_warnings: bool = False
    """
    Whether to log warnings from trace cache, such as invalidation events.
    """

    enable_sanity_checks: bool = False
    """
    Enable internal sanity checks, which could be useful for debugging
    """

    save_muon_momentum_buffer_in_memory: bool = False
    """
    When using the Muon optimizer with ZeRO Stage 3, keeps the Muon momentum
    buffer in GPU/CPU memory instead of swapping to NVMe with other optimizer
    states. Only relevant when using NVMe offloading.
    """

    leaf_module: DeepSpeedZeroLeafModuleConfig = Field(default_factory=DeepSpeedZeroLeafModuleConfig)
    """
    Configuration for modules that should be treated as ZeRO3 leaf modules.
    """

    # Validators
    @model_validator(mode="after")
    def overlap_comm_valid(self):
        if self.overlap_comm is None:
            self.overlap_comm = self.stage == ZeroStageEnum.weights
        return self

    @model_validator(mode="after")
    def offload_ratio_check(self):
        offload_config = self.offload_optimizer
        if offload_config and offload_config.ratio < 1.0:
            assert self.stage == ZeroStageEnum.weights, "Partial offloading only supported for ZeRO Stage 3."
        return self


# ═══════════════════════════════════════════════════════════════
# DES-LOC ZeRO Integration Config (M194)
# ═══════════════════════════════════════════════════════════════


DESLOC_ZERO_DEFAULTS = {
    "desloc_enabled": False,
    "desloc_Kx": 32,
    "desloc_Ku": 96,
    "desloc_Kv": 192,
    "desloc_clip_rho": 1.0,
    "desloc_outer_optimizer": "averaging",
    "desloc_nesterov_momentum": 0.9,
    "desloc_nesterov_lr": 1.0,
    "desloc_muon_compat": False,
    "desloc_variant": "adam",
    "desloc_warmup_sync_steps": 0,
}


class DESLOCZeroConfig:
    """Configuration for DES-LOC + ZeRO integration.

    DES-LOC operates at the optimizer level by gating allreduce.
    ZeRO partitions optimizer states across workers. The interaction:

    ZeRO-1 (optimizer state partitioning):
      - DES-LOC Kx gates the gradient allreduce
      - Ku/Kv gate optimizer state synchronization
      - Compatible: partitioned states sync at different rates

    ZeRO-2 (gradient partitioning):
      - DES-LOC Kx gates reduce_scatter of gradients
      - Must ensure gradient buckets align with sync boundaries
      - Compatible with caveats on bucket ordering

    ZeRO-3 (parameter partitioning):
      - DES-LOC Kx would gate the all_gather of parameters
      - Complex interaction: params already distributed
      - Partial compatibility — requires careful Kx selection

    Knuth critique (user perspective):
      Bug risk: if ZeRO-2 bucket boundaries don't align with
      DES-LOC Kx boundaries, some buckets sync while others don't,
      leading to inconsistent parameter states across workers.
      Mitigation: force bucket flush at every Kx boundary.

    Knuth critique (system perspective):
      ZeRO-3's all_gather is fundamentally different from DES-LOC's
      allreduce gating. ZeRO-3 gathers parameters for forward pass
      (required for correctness), not for synchronization. Gating
      ZeRO-3 all_gather by Kx would break forward pass. Solution:
      only gate the optimizer state sync, not param gathering.
    """

    def __init__(self, zero_config=None, desloc_dict=None):
        self.zero_stage = 0
        self.desloc_enabled = False
        self.desloc_Kx = 32
        self.desloc_Ku = 96
        self.desloc_Kv = 192
        self.desloc_clip_rho = 1.0
        self.desloc_outer_optimizer = 'averaging'
        self.desloc_muon_compat = False
        self.desloc_variant = 'adam'
        self.compatible = True
        self.warnings = []

        if zero_config is not None:
            if hasattr(zero_config, 'stage'):
                self.zero_stage = zero_config.stage
            elif isinstance(zero_config, dict):
                self.zero_stage = zero_config.get('stage', 0)

        if desloc_dict is not None and isinstance(desloc_dict, dict):
            self.desloc_enabled = desloc_dict.get('enabled', False)
            self.desloc_Kx = desloc_dict.get('Kx', 32)
            self.desloc_Ku = desloc_dict.get('Ku', 96)
            self.desloc_Kv = desloc_dict.get('Kv', 192)
            self.desloc_clip_rho = desloc_dict.get('clip_radius', 1.0)
            self.desloc_outer_optimizer = desloc_dict.get('outer_optimizer', 'averaging')
            self.desloc_muon_compat = desloc_dict.get('muon_compat', False)
            self.desloc_variant = desloc_dict.get('variant', 'adam')

        if self.desloc_enabled:
            self._validate_compatibility()

    def _validate_compatibility(self):
        """Validate DES-LOC + ZeRO compatibility."""
        self.compatible = True
        self.warnings = []

        if self.zero_stage == 0:
            # No ZeRO — fully compatible
            pass

        elif self.zero_stage == 1:
            # ZeRO-1: optimizer state partitioning — compatible
            if self.desloc_Kx > 128:
                self.warnings.append(
                    f"ZeRO-1 + DES-LOC Kx={self.desloc_Kx}: very infrequent "
                    f"sync may cause optimizer state drift between partitions")

        elif self.zero_stage == 2:
            # ZeRO-2: gradient partitioning — compatible with caveats
            self.warnings.append(
                "ZeRO-2 + DES-LOC: gradient reduce_scatter gated by Kx. "
                "Ensure gradient accumulation steps align with Kx.")
            if self.desloc_Kx > 64:
                self.warnings.append(
                    f"ZeRO-2 + Kx={self.desloc_Kx}: high Kx with gradient "
                    f"partitioning may cause memory pressure from "
                    f"accumulated ungathered gradients")

        elif self.zero_stage == 3:
            # ZeRO-3: parameter partitioning — partial compatibility
            self.warnings.append(
                "ZeRO-3 + DES-LOC: only optimizer state sync is gated by "
                "Kx/Ku/Kv. Parameter all_gather for forward pass is NOT "
                "gated (required for correctness).")
            if self.desloc_Kx < 4:
                self.warnings.append(
                    "ZeRO-3 + low Kx: minimal benefit since params are "
                    "already gathered every forward pass")

        # Muon + ZeRO interaction
        if self.desloc_muon_compat and self.zero_stage >= 2:
            self.warnings.append(
                "Muon + ZeRO-2/3: Muon's per-worker Newton-Schulz "
                "orthogonalization operates on local slices. Ensure "
                "the orthogonalization is applied before any gather.")

        return self.compatible

    def get_effective_config(self):
        """Return the effective DES-LOC+ZeRO configuration."""
        return {
            'zero_stage': self.zero_stage,
            'desloc_enabled': self.desloc_enabled,
            'Kx': self.desloc_Kx,
            'Ku': self.desloc_Ku,
            'Kv': self.desloc_Kv,
            'outer_optimizer': self.desloc_outer_optimizer,
            'muon_compat': self.desloc_muon_compat,
            'compatible': self.compatible,
            'warnings': self.warnings,
        }

    def format_report(self):
        """Format a human-readable compatibility report."""
        lines = [f"DES-LOC + ZeRO-{self.zero_stage} Compatibility Report:"]
        lines.append(f"  DES-LOC: {'enabled' if self.desloc_enabled else 'disabled'}")
        if self.desloc_enabled:
            lines.append(f"  Kx={self.desloc_Kx}, Ku={self.desloc_Ku}, Kv={self.desloc_Kv}")
            lines.append(f"  Outer optimizer: {self.desloc_outer_optimizer}")
            lines.append(f"  Muon compat: {self.desloc_muon_compat}")
            lines.append(f"  Compatible: {'YES' if self.compatible else 'NO'}")
            if self.warnings:
                lines.append("  Warnings:")
                for w in self.warnings:
                    lines.append(f"    - {w}")
        return "\n".join(lines)


class DESLOCZeROBucketAligner:
    """Align ZeRO gradient buckets with DES-LOC sync boundaries.

    When using ZeRO-2 + DES-LOC, gradient reduce_scatter must
    be gated by Kx. This aligner ensures that all buckets in a
    gradient accumulation boundary are either all synced or all
    skipped — no partial sync.

    Implementation: at Kx boundaries, flush all pending buckets.
    Between Kx boundaries, accumulate locally without communication.
    """

    def __init__(self, Kx=32):
        self.Kx = Kx
        self.step = 0
        self.pending_buckets = []
        self.flushed_count = 0
        self.skipped_count = 0

    def should_flush(self, step=None):
        """Check if buckets should be flushed (synced) at this step."""
        s = step if step is not None else self.step
        if self.Kx <= 1:
            return True
        return s % self.Kx == 0

    def register_bucket(self, bucket_id, bucket_bytes):
        """Register a gradient bucket for tracking."""
        self.pending_buckets.append({
            'id': bucket_id,
            'bytes': bucket_bytes,
            'step': self.step,
        })

    def advance(self):
        """Advance step and flush if at Kx boundary."""
        self.step += 1
        if self.should_flush():
            self.flushed_count += len(self.pending_buckets)
            self.pending_buckets = []
        else:
            self.skipped_count += len(self.pending_buckets)
            self.pending_buckets = []

    def get_stats(self):
        return {
            'step': self.step,
            'flushed': self.flushed_count,
            'skipped': self.skipped_count,
            'Kx': self.Kx,
        }


# End M194
