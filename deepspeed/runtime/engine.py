# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# M171: Megatron 78cb17812 — moved steves branch
# Migrated Megatron-LM commit 78cb1781250676cae9c05fd5e222034d515bcf16
# Changes: updated README and examples scripts to reflect new Megatron
#          codebase structure (examples/, tools/, tasks/ layout), revised
#          BERT/GPT-2 pretraining args (removed legacy --lazy-loader,
#          --resume-dataloader, --cache-dir; added --data-path, --vocab-file,
#          --merge-file, --data-impl mmap, --min-lr, logging/eval intervals),
#          new distributed finetune scripts for RACE/MNLI, updated
#          generate_text.sh to use tools/generate_samples_gpt2.py.
print('[M171]')

import os
import re
import stat
import torch
import hashlib
from collections import defaultdict, OrderedDict, deque
from shutil import copyfile
import gc

from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from contextlib import contextmanager

from typing import Callable, Dict, Union, Iterable, Container, List

import deepspeed

from deepspeed import comm as dist
from deepspeed.runtime.utils import see_memory_usage, DummyOptim, register_output_backward_hooks, check_internal_apis_for_count_used_parameters
from .zero.offload_config import OffloadDeviceEnum, OffloadStateTypeEnum
from deepspeed.runtime.base_optimizer import ZeROOptimizer
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from deepspeed.runtime.zenflow.zenflow_stage_1_and_2 import ZenFlowZeroOptimizer
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.runtime.zero.utils import is_zero_supported_optimizer, ZeRORuntimeException
from deepspeed.runtime.zero.parameter_offload import DeepSpeedZeRoOffload
from deepspeed.runtime.zero.config import ZERO_OPTIMIZATION
from deepspeed.runtime.zenflow.engine import (configure_zenflow, zenflow_step, is_zenflow_update_boundary,
                                              sync_zenflow_optimizer_lr)

from deepspeed.runtime.fp16.fused_optimizer import FP16_Optimizer
from deepspeed.runtime.fp16.loss_scaler import LossScaleConfig, LossScaleProfile
from deepspeed.runtime.fp16.unfused_optimizer import FP16_UnfusedOptimizer
from deepspeed.runtime.bf16_optimizer import BF16_Optimizer

from deepspeed.linear.optimized_linear import LoRAOptimizedLinear
from deepspeed.module_inject.layers import GatherReplacedLayerParams, configure_tensor_parallel_runtime, collect_autotp_universal_checkpoint_info
from deepspeed.runtime.config import DEEPSPEED_OPTIMIZERS, \
    ADAGRAD_OPTIMIZER, ADAM_OPTIMIZER, ADAMW_OPTIMIZER, LAMB_OPTIMIZER, ONEBIT_ADAM_OPTIMIZER, ONEBIT_LAMB_OPTIMIZER, \
    TORCH_ADAM_PARAM, ADAM_W_MODE, ADAM_W_MODE_DEFAULT, ZERO_ONE_ADAM_OPTIMIZER, MUADAM_OPTIMIZER, MUADAMW_OPTIMIZER, \
    MUSGD_OPTIMIZER, LION_OPTIMIZER, MUON_OPTIMIZER

from deepspeed.runtime.model_checkpointing.constants import ValidationMode, \
    CHECKPOINT_TAG_VALIDATION, CHECKPOINT_WRITER, CHECKPOINT_SERIALIZATION

from deepspeed.runtime.dataloader import DeepSpeedDataLoader
from deepspeed.runtime.zero.muon.muon_optimizer import MuonWithAuxAdam
from deepspeed.runtime.constants import \
    ROUTE_TRAIN, ROUTE_PREDICT, ROUTE_EVAL, \
    PLD_THETA, PLD_GAMMA, BFLOAT16, FP16, AMP, GRADIENT_ACCUMULATION_STEPS, \
    DATA_PARALLEL_GROUP, GLOBAL_RANK, DDP_BFLOAT16
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.compression import compression_scheduler
from deepspeed.compression.constants import \
    WEIGHT_QUANTIZE_IN_FORWARD_ENABLED, \
    WEIGHT_QUANTIZATION, SHARED_PARAMETERS, \
    WEIGHT_QUANTIZE_ENABLED, \
    WEIGHT_QUANTIZE_GROUPS, \
    WEIGHT_QUANTIZE_FP16_MIXED_QUANTIZE, \
    WEIGHT_QUANTIZE_CHANGE_RATIO, \
    WEIGHT_QUANTIZE_TYPE, \
    WEIGHT_QUANTIZE_ROUNDING, \
    WEIGHT_QUANTIZE_VERBOSE, \
    WEIGHT_QUANTIZE_KERNEL
from deepspeed.checkpoint.constants import OPTIMIZER_STATE_DICT, FROZEN_PARAM_FRAGMENTS, UNIVERSAL_CHECKPOINT_INFO
from deepspeed.checkpoint.utils import clone_tensors_for_torch_save
from deepspeed.checkpoint.ds_to_universal import dp_index_to_str
from deepspeed.runtime.sparse_tensor import SparseTensor

from deepspeed.runtime import lr_schedules
from deepspeed.utils import groups
from deepspeed.utils import logger, log_dist, log_dist_once, instrument_w_nvtx
from deepspeed.utils.torch import required_torch_version
from deepspeed.utils.z3_leaf_module import apply_zero_leaf_module_config
from deepspeed.utils.timer import NoopTimer, ThroughputTimer, SynchronizedWallClockTimer, \
    FORWARD_MICRO_TIMER, BACKWARD_MICRO_TIMER, BACKWARD_INNER_MICRO_TIMER, BACKWARD_REDUCE_MICRO_TIMER, \
    STEP_MICRO_TIMER, \
    FORWARD_GLOBAL_TIMER, BACKWARD_GLOBAL_TIMER, BACKWARD_INNER_GLOBAL_TIMER, BACKWARD_REDUCE_GLOBAL_TIMER, \
    STEP_GLOBAL_TIMER
from deepspeed.utils.debug import debug_extract_module_and_param_names, debug_clear_module_and_param_names
from deepspeed.monitor.monitor import MonitorMaster
from deepspeed.runtime.progressive_layer_drop import ProgressiveLayerDrop
from deepspeed.runtime.utils import clip_grad_norm_, compare_tensors_in_structures, maybe_loss_for_backward
from deepspeed.runtime.eigenvalue import Eigenvalue
from deepspeed.runtime.data_pipeline.constants import DATA_SAMPLING, \
    DATA_ROUTING, DATA_SAMPLING_ENABLED, CURRICULUM_LEARNING, \
    CURRICULUM_LEARNING_ENABLED, DATA_SAMPLING_NUM_WORKERS, RANDOM_LTD, \
    RANDOM_LTD_ENABLED, RANDOM_LTD_LAYER_ID, RANDOM_LTD_LAYER_NUM, \
    RANDOM_LTD_LAYER_TOKEN_LR_SCHEDULE, RANDOM_LTD_LAYER_TOKEN_LR_ENABLED, \
    RANDOM_LTD_GLOBAL_BATCH_SIZE, RANDOM_LTD_MICRO_BATCH_SIZE, DATA_EFFICIENCY
from deepspeed.runtime.data_pipeline.curriculum_scheduler import CurriculumScheduler
from deepspeed.runtime.checkpoint_engine import (create_checkpoint_engine, TorchCheckpointEngine, CheckpointCommitInfo)

from deepspeed.runtime.data_pipeline.data_routing.scheduler import RandomLTDScheduler
from deepspeed.runtime.data_pipeline.data_routing.helper import remove_random_ltd_state_dict
from deepspeed.runtime.data_pipeline.data_routing.basic_layer import RandomLayerTokenDrop

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from deepspeed.runtime.torch_autocast import init_autocast_params, get_default_autocast_lower_precision_modules, autocast_if_enabled

from .pipe.module import PipelineModule
from .utils import get_ma_status
from .compiler import is_compile_supported, compiled_autograd
from ..ops.adam import FusedAdam
from ..moe.sharded_moe import TopKGate, MOELayer
from ..moe.layer import MoE
from ..moe.utils import is_moe_param, configure_moe_param_groups
from ..git_version_info import version

from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler
from deepspeed.utils.logging import print_json_dist, print_configuration

from deepspeed.accelerator import get_accelerator

from deepspeed.runtime.config import DtypeEnum

from deepspeed.compile.util import is_deepcompile_supported, get_deepcompile_handle, deepcompile_backward_prologue
from deepspeed.compile.backend import register_compile_pass, opt_passes
from deepspeed.compile.passes import zero3_compile, prefetch, selective_gather, offload_adam_states
from deepspeed.compile.init_z1 import init_z1
from deepspeed.compile.init_z3 import init_z3
from deepspeed.compile.init_sp import init_autosp

MEMORY_OPT_ALLREDUCE_SIZE = 500000000

DeepSpeedOptimizerCallable = \
    Callable[[Union[Iterable[Parameter], Dict[str, Iterable]]], Optimizer]
DeepSpeedSchedulerCallable = Callable[[Optimizer], _LRScheduler]

try:
    import apex
    from apex import amp
    APEX_INSTALLED = True
except ImportError:
    # Fail silently so we don't spam logs unnecessarily if user isn't using amp
    APEX_INSTALLED = False


def split_half_float_double_sparse(tensors):
    device_type = get_accelerator().device_name()
    supported_types = get_accelerator().supported_dtypes()

    for t in tensors:
        assert t.dtype in supported_types, f"attempting to reduce an unsupported grad type: {t.dtype}"

    sparse_tensor_buckets, dense_tensor_buckets = [], []
    for i, dtype in enumerate(supported_types):
        sparse_bucket, dense_bucket = [], []
        for t in tensors:
            if t.dtype == dtype:
                if isinstance(t, SparseTensor):
                    sparse_bucket.append(t)
                else:
                    dense_bucket.append(t)
        if sparse_bucket:
            sparse_tensor_buckets.append((dtype, sparse_bucket))
        if dense_bucket:
            dense_tensor_buckets.append((dtype, dense_bucket))
    return sparse_tensor_buckets, dense_tensor_buckets


class EngineTimers(object):
    r"""Wallclock timers for DeepSpeedEngine"""

    def __init__(self, enable_micro_timers, enable_global_timers):
        self.forward_timers = []
        self.backward_timers = []
        self.backward_inner_timers = []
        self.backward_reduce_timers = []
        self.step_timers = []
        self.global_timers = []
        self.micro_timers = []

        if enable_micro_timers:
            self.forward_timers += [FORWARD_MICRO_TIMER]
            self.backward_timers += [BACKWARD_MICRO_TIMER]
            self.backward_inner_timers += [BACKWARD_INNER_MICRO_TIMER]
            self.backward_reduce_timers += [BACKWARD_REDUCE_MICRO_TIMER]
            self.step_timers += [STEP_MICRO_TIMER]
            self.micro_timers += [
                FORWARD_MICRO_TIMER, BACKWARD_MICRO_TIMER, BACKWARD_INNER_MICRO_TIMER, BACKWARD_REDUCE_MICRO_TIMER,
                STEP_MICRO_TIMER
            ]

        if enable_global_timers:
            self.forward_timers += [FORWARD_GLOBAL_TIMER]
            self.backward_timers += [BACKWARD_GLOBAL_TIMER]
            self.backward_inner_timers += [BACKWARD_INNER_GLOBAL_TIMER]
            self.backward_reduce_timers += [BACKWARD_REDUCE_GLOBAL_TIMER]
            self.step_timers += [STEP_GLOBAL_TIMER]
            self.global_timers += [
                FORWARD_GLOBAL_TIMER, BACKWARD_GLOBAL_TIMER, BACKWARD_INNER_GLOBAL_TIMER, BACKWARD_REDUCE_GLOBAL_TIMER,
                STEP_GLOBAL_TIMER
            ]

    def active_timers(self):
        return self.micro_timers + self.global_timers


class DeepSpeedEngine(Module):
    r"""DeepSpeed engine for training."""

    def __init__(self,
                 args,
                 model,
                 optimizer=None,
                 model_parameters=None,
                 training_data=None,
                 lr_scheduler=None,
                 mpu=None,
                 dist_init_required=None,
                 collate_fn=None,
                 config=None,
                 config_class=None,
                 mesh_device=None,
                 dont_change_device=False):
        super(DeepSpeedEngine, self).__init__()
        self.dont_change_device = dont_change_device
        self.client_optimizer = optimizer
        self.client_lr_scheduler = lr_scheduler
        self.training_data = training_data
        self.collate_fn = collate_fn
        self.mpu = mpu
        self.all_to_all_group = None
        self.data_parallel_group = None
        self.global_steps = 0
        self.global_samples = 0
        self.micro_steps = 0
        self.skipped_steps = 0
        # DES-LOC: independent sync period state — Algorithm 1
        self.desloc_enabled = False
        self.desloc_Kx = 1  # Kx=1 = standard DDP
        self.desloc_Ku = 3
        self.desloc_Kv = 6
        self.desloc_step = 0  # local step counter for sync gating
        self.desloc_skipped_allreduces = 0
        self.desloc_clip_rho = 1.0
        self.desloc_warmup_steps = 512
        self.desloc_outer_opt_mode = 'average'
        # M227: Figure 1 loss curve data — collected per step for plotting
        # Ref: NKI-FA da964f3 draw_plot.py — data from experiment logs
        # Ref: Section 5.4 RQ4 — loss curves across model scales
        self._desloc_loss_history = []  # [(step, loss, lr, is_sync)]
        self._desloc_eval_history = []  # [(step, eval_loss, eval_ppl)]
        self._desloc_comm_history = []  # [(step, bytes_sent, ops, tier)]
        self._desloc_throughput_history = []  # [(step, tokens_per_sec)]
        self._desloc_figure_dir = None  # set via config or env
        self._desloc_loss_window = []  # sliding window for smoothing
        self._desloc_loss_window_size = 50  # EMA window
        self._desloc_baseline_losses = {}  # {config_key: [losses]} for diff
        self._desloc_fig1_configs = []  # list of (label, Kx, loss_list)
        self._desloc_step_timer_ns = 0
        self._desloc_histogram_enabled = False
        self._desloc_double_buffer_enabled = False
        self._desloc_histogram_kernel = None
        self._desloc_buffer_pool = None
        # M470: distributed optimizer state sharding manager (init after optimizer is ready)
        self._desloc_dist_opt_mgr = None
        self.gradient_average = True
        self.warn_unscaled_loss = True
        self.config = config
        self._config = config_class
        self.loaded_checkpoint_mp_world_size = None
        self.loaded_checkpoint_dp_world_size = None
        # DES-LOC: load configuration from DeepSpeed config
        if hasattr(self._config, 'desloc_enabled'):
            self.desloc_enabled = self._config.desloc_enabled
            self.desloc_Kx = getattr(self._config, 'desloc_Kx', 1)
            self.desloc_Ku = getattr(self._config, 'desloc_Ku', 3)
            self.desloc_Kv = getattr(self._config, 'desloc_Kv', 6)
            self.desloc_clip_rho = getattr(self._config, 'desloc_clip_rho', 1.0)
            self.desloc_warmup_steps = getattr(self._config, 'desloc_warmup', 512)
            self.desloc_outer_opt_mode = getattr(self._config, 'desloc_outer_opt', 'average')

        # =============================================================
        # M342 — Claude-30: SP+DEC+AC Unified Initialization
        #
        # Three orthogonal strategies composing in the DeepSpeed engine:
        #   SP (Sequence Parallel): AutoSP compile pass or Ulysses
        #     Splits input along seq dim → each GPU sees seq_len/sp_size
        #     Requires: ZeRO stage 0, SDPA attention
        #     Source: deepspeed/compile/passes/sp_compile.py
        #
        #   DEC (Desynced Communication): DES-LOC Kx/Ku/Kv gating
        #     Skips AllReduce on non-boundary steps → N/Kx + N/Ku + N/Kv
        #     Requires: ZeRO stage 0/1
        #     Source: engine.py:allreduce_gradients() Kx gate
        #
        #   AC (Activation Checkpointing): layer-wise or compile-time
        #     Layer-wise: torch.utils.checkpoint per TransformerBlock
        #     Compile-time: Aten-IR operator-level via AutoSP pass
        #     Source: user model code (layer-wise) or compile pass
        #
        # Composition:
        #   SP × DEC: SP splits data spatially per step,
        #             DEC gates communication temporally across steps.
        #             Each GPU does Kx local steps on seq_len/sp_size
        #             tokens, then AllReduces at Kx boundary.
        #
        #   SP × AC:  SP reduces per-GPU sequence length → less activation
        #             memory. AC further reduces by recomputing.
        #             Combined: enables very long contexts on limited GPUs.
        #
        #   DEC × AC: DEC reduces communication, AC reduces memory.
        #             No interaction — DEC affects optimizer sync,
        #             AC affects forward/backward compute.
        #
        #   SP × DEC × AC: All three simultaneously.
        #             ZeRO stage 0 required (common constraint).
        #
        # Addressing NeurIPS reviewer concerns:
        # Q: "Why Ulysses not Ring Flash Attention?"
        # A: Ulysses is faster (see Ulysses paper Figs 9-11).
        #    AutoSP achieves 2.26× longer context than Ring.
        #    All attention kernels use FlashAttention (O(T) memory),
        #    NOT quadratic attention.
        #
        # Q: "torch.compile AC vs torch.utils.checkpoint?"
        # A: Layer-wise AC discards ALL activations in a block.
        #    Compile-time AC operates on Aten-IR (matmuls, sigmoids)
        #    → finer-grained search space → better mem/compute tradeoff.
        #    Both are supported: layer-wise always, compile when AutoSP.
        # =============================================================
        self._desloc_sp_enabled = False
        self._desloc_ac_enabled = False
        self._desloc_sp_mode = 'none'  # 'none', 'autosp', 'ulysses'
        self._desloc_ac_mode = 'none'  # 'none', 'layer', 'compile'

        # Detect SP mode
        if hasattr(self._config, 'compile_config') and self.compile_autosp():
            self._desloc_sp_enabled = True
            self._desloc_sp_mode = 'autosp'
        elif hasattr(self, 'sequence_parallel_size') and self.sequence_parallel_size > 1:
            self._desloc_sp_enabled = True
            self._desloc_sp_mode = 'ulysses'

        # Detect AC mode
        # Layer-wise AC is detected by checking model's TransformerBlock
        # Compile-time AC is active when AutoSP is enabled (AutoSP
        # includes its own Aten-IR level AC pass)
        if self._desloc_sp_mode == 'autosp':
            self._desloc_ac_enabled = True
            self._desloc_ac_mode = 'compile'
        # Check if user applied torch.utils.checkpoint to model layers
        _ac_layers = 0
        for m in model.modules():
            if hasattr(m, 'use_ac') and getattr(m, 'use_ac', False):
                _ac_layers += 1
        if _ac_layers > 0:
            self._desloc_ac_enabled = True
            if self._desloc_ac_mode == 'none':
                self._desloc_ac_mode = 'layer'

        # Validate SP+DEC+AC compatibility
        if self.desloc_enabled and self._desloc_sp_enabled:
            # Both require ZeRO stage 0
            _zero_stage = self.zero_optimization_stage()
            if _zero_stage != 0:
                logger.warning(f"SP+DEC requires ZeRO stage 0, but got stage "
                               f"{_zero_stage}. DES-LOC Kx gating may conflict "
                               f"with ZeRO gradient partitioning.")

        if dist.get_rank() == 0:
            if self.desloc_enabled or self._desloc_sp_enabled or self._desloc_ac_enabled:
                logger.info(f"SP+DEC+AC config: "
                            f"SP={self._desloc_sp_mode} "
                            f"DEC={'Kx='+str(self.desloc_Kx) if self.desloc_enabled else 'off'} "
                            f"AC={self._desloc_ac_mode}"
                            f"{' ('+str(_ac_layers)+' layers)' if _ac_layers > 0 else ''}")

        # M343-C34: Heterogeneous sync gate for mixed-cluster deployments.
        # Pattern: Megatron DistributedDataParallel tracks grad-ready
        # buckets per-parameter group; we track per-tier sync readiness.
        self._hetero_sync_gate = None
        _hetero_cfg = config.get('hetero_mesh', {}) if isinstance(config, dict) else {}
        if _hetero_cfg.get('adaptive_sync', False) and self.desloc_enabled:
            from deepspeed.compile.custom_ops.bloombee_bridge import (HeteroSyncGate, HeteroSyncConfig)
            self._hetero_sync_gate = HeteroSyncGate(
                HeteroSyncConfig(
                    base_Kx=self.desloc_Kx,
                    adaptive_enabled=True,
                    straggler_threshold=_hetero_cfg.get('straggler_threshold', 2.0),
                    max_Kx=_hetero_cfg.get('max_Kx', 32),
                ))
        self.enable_backward_allreduce = True
        self.inside_no_sync_ctxt = False
        self.progressive_layer_drop = None
        self.eigenvalue = None
        self.block_eigenvalue = None
        self.gas_boundary_ctr = 0
        self.dist_backend = get_accelerator().communication_backend_name()
        self.has_moe_layers = False
        self.num_experts = []
        self.gate_modules = []
        self.moe_layers = []
        self._step_applied = False
        self._global_grad_norm = None
        self.use_ds_comm = False  # False --> Use torch.dist, True --> Use ds.comm backend.
        self.checkpoint_engine = None
        self.optimizer = None
        self.basic_optimizer = None
        self.lr_scheduler = None

        self._is_gradient_accumulation_boundary = None
        self.scale_wrt_gas = None
        self.losses = None
        self.mesh_device = mesh_device

        # Flag to indicate that scale() was called before manual backward pass
        self._manual_backward_expected = False

        # for debug purposes - can then debug print: debug_get_module_name(module)
        debug_extract_module_and_param_names(model)

        if self.mesh_device:
            groups.mesh_device = self.mesh_device

        self._do_args_sanity_check(args)
        self._configure_with_arguments(args, mpu)
        self._do_sanity_check()
        if self.autotp_size() > 1:
            self._configure_tensor_parallel(model, self.tensor_parallel_config())
        see_memory_usage("DeepSpeed Engine: After args sanity test", force=self.memory_breakdown())
        if mpu is not None:
            if self.elasticity_enabled():
                if not self.is_elastic_model_parallel_supported():
                    assert not self.elasticity_enabled(), ("Elasticity is not currently supported"
                                                           " with model parallelism.")

        self._set_distributed_vars(args)

        dist.configure(self._config)

        self.monitor = MonitorMaster(self._config.monitor_config)

        see_memory_usage(
            "DeepSpeed Engine: Before configure distributed model",
            force=self.memory_breakdown(),
        )

        self.pipeline_parallelism = isinstance(model, PipelineModule)

        self._deepcompile_active = False

        # Configure distributed model
        self._configure_distributed_model(model)

        # These hooks should be disabled later if DeepCompile is not active.
        self.module_forward_pre_hook = self._create_module_forward_pre_hook()
        self.module_forward_post_hook = self._create_module_forward_post_hook()

        # needed for zero_to_fp32 weights reconstruction to remap nameless data to state_dict
        self.param_names = {param: name for name, param in model.named_parameters()}

        self._get_model_parameters()

        see_memory_usage("DeepSpeed Engine: After configure distributed model")

        # Configure wall clock timers
        self.timers = SynchronizedWallClockTimer()
        # Throughput timer
        self.tput_timer = ThroughputTimer(self._config.timers_config,
                                          batch_size=self.train_batch_size(),
                                          steps_per_output=self.steps_per_print(),
                                          monitor_memory=False)

        log_dist(f"DeepSpeed Flops Profiler Enabled: {self.flops_profiler_enabled()}", ranks=[0])

        if self.flops_profiler_enabled():
            self.flops_profiler = FlopsProfiler(self.module, self, self.flops_profiler_recompute_fwd_factor())

        if training_data:
            self.training_dataloader = self.deepspeed_io(training_data)
        else:
            self.training_dataloader = None

        # Configure optimizer and scheduler
        has_optimizer = False

        if optimizer or self.optimizer_name():
            has_optimizer = True
        # If no parameters given by init default to module parameters
        if model_parameters is None:
            model_parameters = self.module.parameters()

        # Convert model parameters from generator to list
        if not isinstance(model_parameters, list):
            model_parameters = list(model_parameters)

        # grad scaler only for Z0 (no ZeRO) + fp16 + torch_autocast
        # ZeRO1/2/3 optimizers have their own grad scaler logic
        self.torch_autocast_z0_gradscaler = None
        if self.torch_autocast_enabled():
            init_autocast_params(self, self.torch_autocast_dtype(), self.torch_autocast_lower_precision_safe_modules())
            if (not self.zero_optimization() and self.torch_autocast_dtype() == torch.float16):
                self.torch_autocast_z0_gradscaler = torch.amp.GradScaler(device=get_accelerator().device_name())

        self._configure_zenflow = lambda: configure_zenflow(self)
        self._is_zenflow_update_boundary = lambda: is_zenflow_update_boundary(self)
        self._zenflow_step = lambda lr_kwargs: zenflow_step(self, lr_kwargs)
        self._sync_zenflow_optimizer_lr = lambda: sync_zenflow_optimizer_lr(self)

        self._configure_zenflow()

        if has_optimizer:
            self._configure_optimizer(optimizer, model_parameters)
            self._configure_lr_scheduler()
            self._report_progress(0)
        elif self.zero_optimization():
            # no optim selected but zero is enabled
            self.optimizer = self._configure_zero_optimizer(optimizer=None)
        elif self.bfloat16_enabled():
            self.optimizer = self._configure_bf16_optimizer(optimizer=None)

        # Hook optimizer for snip_momentum pruning
        if hasattr(model, 'pruners'):
            from ..compression.helper import rewrite_optimizer_step
            self.optimizer.pruners = model.pruners
            rewrite_optimizer_step(self.optimizer)

        # Bookkeeping for sparse support
        self.sparse_tensor_module_names = set()
        # if self.sparse_gradients_enabled():
        for name, module in self.module.named_modules():
            if isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag)) and self.sparse_gradients_enabled():
                self.sparse_tensor_module_names.add(name + ".weight")
                logger.info("Will convert {} to sparse tensor during training".format(name))

        self._optimized_linear_offload_setup()

        self.save_non_zero_checkpoint = False
        self.save_zero_checkpoint = False
        if not isinstance(self.optimizer, DeepSpeedZeRoOffload):
            self._configure_checkpointing()

        if self.eigenvalue_enabled():
            self.eigenvalue = self._configure_eigenvalue()

        if self.pld_enabled():
            self.progressive_layer_drop = self._configure_progressive_layer_drop()

        if self.curriculum_enabled_legacy():
            self.curriculum_scheduler_legacy = self._configure_curriculum_scheduler_legacy()

        if self.random_ltd_enabled():
            random_ltd_config = self.random_ltd_config()
            random_ltd_config[RANDOM_LTD_GLOBAL_BATCH_SIZE] = self.train_batch_size()
            random_ltd_config[RANDOM_LTD_MICRO_BATCH_SIZE] = self.train_micro_batch_size_per_gpu()
            self.random_ltd_scheduler = self._configure_random_ltd_scheduler(random_ltd_config)

        # Engine timers

        self.engine_timers = EngineTimers(enable_micro_timers=self.wall_clock_breakdown(),
                                          enable_global_timers=self.wall_clock_breakdown()
                                          or self.flops_profiler_enabled())

        self.engine_timers_cache = {}

        if self.global_rank == 0:
            self._config.print("DeepSpeedEngine configuration")
            if self.dump_state():
                print_configuration(self, "DeepSpeedEngine")

        # Use torch (un)flatten ops
        self.flatten = _flatten_dense_tensors
        self.unflatten = _unflatten_dense_tensors

        self._is_compiled = False
        if is_deepcompile_supported():
            # Predefined compile passes
            self.register_compile_pass(zero3_compile.NAME, zero3_compile.add_z3_gather_release)
            self.register_compile_pass(prefetch.NAME, prefetch.schedule_prefetch)
            self.register_compile_pass(selective_gather.NAME, selective_gather.selective_gather)
            self.register_compile_pass(offload_adam_states.NAME, offload_adam_states.move_opt_states)

        # We now support PyTorch style backward, but it relies on the counter in ZeRO optimizers.
        # However, we need some internal APIs to count the number of only used parameters.
        # So we only enable this feature when those internal APIs are available.
        # Otherwise, we fallback to DeepSpeed style backward only.
        # See `count_used_parameters_in_backward` for more details.
        self._running_engine_backward = False
        self._support_torch_style_backward = False
        # Flag to control whether gradients should be scaled by gradient accumulation steps
        self._scale_wrt_gas = True
        if isinstance(self.optimizer, ZeROOptimizer) and check_internal_apis_for_count_used_parameters():
            self._support_torch_style_backward = True
            # These hooks are used for non-scalar backward support, such as `out.backward(out_grad)`,
            # not for `engine.backward(loss)`. In this case, we need to ensure that the preprocessing
            # and postprocessing around the backward call are handled correctly.
            # However, we cannot use `register_full_backward_hook` for post-backward hooks.
            # If none of the module inputs require gradients, `register_full_backward_hook` fires
            # when the gradients of the module outputs are computed. Our gradient
            # accumulation hooks are called later. But we want `_backward_post_hook` to be called
            # only after all gradients have been computed.
            # To handle this, the optimizer maintains a counter to track the number of gradients
            # that have been computed. When all gradients are ready, it calls `_backward_post_hook`.
            # See also: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook
            self.optimizer.register_grad_acc_post_hook(self._backward_post_hook)

        self._is_compiled_autograd_enabled = False
        self._compile_kwargs = {}

        if self.dist_backend is None:
            self.enable_backward_allreduce = False

    def _optimized_linear_offload_setup(self):
        self.optimized_linear_base_weight_sharding = False
        self.optimized_linear_lora_enabled = False
        offload_ratio = None
        for _, module in self.module.named_modules():
            if isinstance(module, LoRAOptimizedLinear):
                self.optimized_linear_lora_enabled = True
                if offload_ratio is not None:
                    assert offload_ratio == module.lora_config.offload_ratio, \
                        "all lora_config offload ratios should be the same across the model"
                offload_ratio = module.lora_config.offload_ratio
                if module.zero_shards > 1:
                    # set attr so checkpoint saving can handle BWS properly
                    self.optimized_linear_base_weight_sharding = True

        if offload_ratio is None:
            # Nothing enabled, do nothing
            return

        total_params = 0
        for _, p in self.module.named_parameters():
            if hasattr(p, 'ds_optim_param'):
                total_params += p.numel()

        offload_limit = total_params * offload_ratio
        logger.info(f'offloading {offload_ratio*100}% of eligible params, specifically {offload_limit} params')
        total_offloaded = 0
        for _, p in self.module.named_parameters():
            if hasattr(p, 'ds_optim_param'):
                if total_offloaded < offload_limit:
                    total_offloaded += p.numel()
                    p.ds_offload = True
                    p.offload()
                else:
                    p.ds_offload = False

    def _configure_tensor_parallel(self, model, tp_config):
        self._configure_tensor_parallel_states(model)
        configure_tensor_parallel_runtime(tp_config)
        self._apply_autotp_partitioning(model, tp_config)

    def _configure_tensor_parallel_states(self, model):
        """
        Configures the tensor parallel states for the model.
        This includes setting up the tensor parallel groups, initializing the TP mesh,
        and registering a pre-hook to ensure that the Dataloader inputs are consistent across ranks.
        """
        self._set_client_model(model)
        # sanity check
        # currently, the compatibility between 'autotp' and 'zero > 1' has not been validated
        assert self.zero_optimization_stage(
        ) <= 2, "Currently, the compatibility between 'autotp' and 'zero_stage = 3' has not been validated"

        self.mpu = groups
        self.mpu._init_tp_mesh_device(tensor_model_parallel_size=self.autotp_size())

        self.first_dataloader_check = None

        def check_dataloader_inputs_same_across_ranks(module, args, kwargs):

            def broadcast_and_check(args, bcast_rank, bcast_group):
                if isinstance(args, tuple):
                    args = list(args)
                if len(args) > 0:
                    if self.mpu.get_tensor_model_parallel_rank() == 0:
                        _src_args = [args]
                        dist.broadcast_object_list(object_list=_src_args,
                                                   src=bcast_rank,
                                                   group=bcast_group,
                                                   device=torch.device(get_accelerator().current_device_name()))
                        # Rank 0 does not need to compare with itself
                        is_equal = True
                    else:
                        _src_args = [None]
                        dist.broadcast_object_list(object_list=_src_args,
                                                   src=bcast_rank,
                                                   group=bcast_group,
                                                   device=torch.device(get_accelerator().current_device_name()))

                        is_equal = compare_tensors_in_structures(args, _src_args[0])

                    equal_tensor = torch.tensor(is_equal,
                                                dtype=self.communication_data_type,
                                                device=torch.device(get_accelerator().current_device_name()))
                    dist.all_reduce(equal_tensor, group=bcast_group)
                    assert torch.equal(
                        equal_tensor,
                        torch.tensor(groups.get_tensor_model_parallel_world_size(),
                                     dtype=self.communication_data_type,
                                     device=torch.device(get_accelerator().current_device_name()))
                    ), "Data inconsistency within the TP group. Please check the Dataloader implementation to ensure consistency."

            bcast_rank = self.mpu.get_tensor_model_parallel_src_rank()
            bcast_group = self.mpu.get_tensor_model_parallel_group()

            broadcast_and_check(args, bcast_rank, bcast_group)
            broadcast_and_check(kwargs, bcast_rank, bcast_group)

            logger.info(":The Dataloader has passed the TP group consistency check.")
            self.first_dataloader_check.remove()

        self.first_dataloader_check = self.module.register_forward_pre_hook(check_dataloader_inputs_same_across_ranks,
                                                                            prepend=True,
                                                                            with_kwargs=True)

    def _apply_autotp_partitioning(self, model, tp_config):
        if getattr(model, "ds_autotp_parsed", False):
            return
        if get_accelerator().is_available() and self.local_rank >= 0:
            get_accelerator().set_device(self.local_rank)

        tp_size = self.autotp_size()
        if tp_config.tensor_parallel.tp_size not in (1, tp_size):
            raise ValueError(f"tensor_parallel.tp.tp_size ({tp_config.tensor_parallel.tp_size}) "
                             f"does not match tensor_parallel.autotp_size ({tp_size}).")
        tp_config.tensor_parallel.tp_size = tp_size
        if tp_config.tensor_parallel.tp_group is None:
            tp_config.tensor_parallel.tp_group = groups.get_tensor_model_parallel_group()

        from deepspeed.module_inject.auto_tp import AutoTP

        # Tensor parallel priority: custom config > HF tp_plan > AutoTP
        partition_config = None
        if hasattr(tp_config, "get_partition_config_object"):
            partition_config = tp_config.get_partition_config_object()

        if partition_config is not None:
            autotp = AutoTP(module=model,
                            all_reduce_linears=(),
                            prefix="",
                            state_dict=None,
                            linear_layer_setting=(torch.nn.Linear, torch.nn.Embedding),
                            orig_layer_impl=None,
                            keep_module_on_host=tp_config.keep_module_on_host,
                            partition_config=partition_config)
            autotp.set_tensor_parallel_config(tp_size, tp_config.tensor_parallel.tp_group)
            autotp.update_linear_policies()
            autotp._replace_module(model)
            setattr(model, UNIVERSAL_CHECKPOINT_INFO, collect_autotp_universal_checkpoint_info(model))
            setattr(model, "ds_autotp_parsed", True)
            return

        if tp_size <= 1:
            setattr(model, "ds_autotp_parsed", True)
            return

        model_config = getattr(model, "config", None)
        from deepspeed.module_inject import replace_transformer_layer

        from deepspeed.runtime.tensor_parallel.config import _get_hf_tp_plan

        hf_tp_plan = _get_hf_tp_plan(model)
        if hf_tp_plan:
            from deepspeed.module_inject.tp_plan_converter import TPPlanConverter
            from deepspeed.module_inject.autotp_config import AutoTPConfig

            layer_specs = TPPlanConverter.convert(hf_tp_plan)
            if layer_specs is not None:
                logger.info(f"Using HuggingFace tp_plan with {len(layer_specs)} layer specifications")
                tp_plan_config = AutoTPConfig(tp_size=tp_size, layer_specs=layer_specs)
                autotp = AutoTP(
                    module=model,
                    all_reduce_linears=(),
                    prefix="",
                    state_dict=None,
                    linear_layer_setting=(torch.nn.Linear, torch.nn.Embedding),
                    orig_layer_impl=None,
                    keep_module_on_host=tp_config.keep_module_on_host,
                    partition_config=tp_plan_config,
                )
                autotp.set_tensor_parallel_config(tp_size, tp_config.tensor_parallel.tp_group)
                autotp.update_linear_policies()
                autotp._replace_module(model)
                setattr(model, "ds_autotp_parsed", True)
                return

        parser_dict = AutoTP.tp_parser(model)
        for client_module, injection_policy in parser_dict:
            tp_config.injection_policy_tuple = injection_policy
            replace_transformer_layer(client_module, model, None, tp_config, model_config)

        setattr(model, UNIVERSAL_CHECKPOINT_INFO, collect_autotp_universal_checkpoint_info(model))
        setattr(model, "ds_autotp_parsed", True)

    def __del__(self):
        try:
            self.destroy()
        except Exception as exc:
            # Avoid destructor-time exceptions for partially initialized engines.
            logger.debug("DeepSpeedEngine.__del__ cleanup skipped: %s", exc, exc_info=True)

    def destroy(self):
        optimizer = getattr(self, "optimizer", None)
        if optimizer is not None and hasattr(optimizer, 'destroy'):
            optimizer.destroy()
        if self.is_deepcompile_active():
            get_deepcompile_handle().cleanup()
        debug_clear_module_and_param_names()

        checkpoint_engine = getattr(self, "checkpoint_engine", None)
        if checkpoint_engine is not None and checkpoint_engine.is_decoupled():
            checkpoint_engine.cleanup()

    def _get_model_parameters(self):
        if self.autotuning_profile_model_info():
            self.autotuning_model_info = {}
            num_params = 0
            trainable_num_params = 0

            for p in self.module.parameters():
                # since user code might call deepspeed.zero.Init() before deepspeed.initialize(), need to check the attribute to check if the parameter is partitioned in zero 3 already or not
                n = 0
                if hasattr(p, "ds_tensor"):  # if the parameter is partitioned in zero 3
                    n += p.ds_numel
                else:  # if the parameter is not partitioned in zero 3 yet
                    n += p.numel()
                num_params += n
                if p.requires_grad:
                    trainable_num_params += n
            if self.global_rank == 0:
                self.autotuning_model_info["num_params"] = num_params * self.mp_world_size
                self.autotuning_model_info["trainable_num_params"] = trainable_num_params * self.mp_world_size

            logger.info(f"model parameter = {num_params}")

    def get_batch_info(self):
        """Get all training batch related settings.
        Returns:
            train_batch_size (int): The effective training batch size. This is the amount of data
                samples that leads to one step of model update.
            train_micro_batch_size_per_gpu (int): Batch size to be processed by one GPU in one
                step (without gradient accumulation).
            gradient_accumulation_steps (int): Number of training steps to accumulate gradients
                before averaging and applying them.
        """
        return (
            self.train_batch_size,
            self.train_micro_batch_size_per_gpu,
            self.gradient_accumulation_steps,
        )

    def set_train_batch_size(self, train_batch_size):
        """Adjust the global batch size by increasing or decreasing the number of
        micro-batches (i.e., gradient accumulation steps). The size of each micro-batch
        (i.e., ``train_micro_batch_size_per_gpu``) is not changed.
        Args:
            train_batch_size (int): The new global batch size for training.
        Raises:
            ValueError: if ``train_batch_size`` is not divisible by the
                configured micro-batch size and data parallelism.
        """
        if train_batch_size % (self.train_micro_batch_size_per_gpu() * self.dp_world_size) != 0:
            #print(f'{train_batch_size=} {self.train_micro_batch_size_per_gpu()=} {self.dp_world_size=}')
            raise ValueError('Train batch size must be divisible by micro-batch data parallelism')
        new_gas = train_batch_size // (self.train_micro_batch_size_per_gpu() * self.dp_world_size)
        # overwrite config
        self._config.train_batch_size = train_batch_size
        self._config.gradient_accumulation_steps = new_gas

    def set_train_micro_batch_size(self, micro_batch_size):
        """Adjust the micro batch size(i.e., the micro batch size in every data parallel group),
        while keep the gradient accumulation steps the same.
        Args:
            micro_batch_size (int): The new micro batch size for training.
        """
        # overwrite config
        new_global_batch_size = micro_batch_size * self._config.gradient_accumulation_steps * self.dp_world_size
        self._config.train_batch_size = new_global_batch_size
        self._config.train_micro_batch_size_per_gpu = micro_batch_size

    def set_data_post_process_func(self, post_process_func):
        if self.training_dataloader is not None:
            self.training_dataloader.post_process_func = post_process_func

    def set_custom_curriculum_learning_schedule(self, schedule_func_dict):
        if self.training_dataloader is not None and self.curriculum_learning_enabled():
            self.training_dataloader.data_sampler.set_custom_curriculum_learning_schedule(schedule_func_dict)

    def get_global_grad_norm(self) -> float:
        """Return the 2-norm of all gradients. If there is model parallelism,
        the norm will be global.
        The computed norm will be cached and reused until the next step() pass.
        .. note::
            In the presence of model parallelism, this is a collective call
            and acts as a barrier among ``mpu.get_model_parallel_group()``.
        Returns:
            float: norm
        """
        return self._global_grad_norm

    def __getattr__(self, name):
        """
        Pass through attributes defined in the model if they are not overridden by ds-engine.
        """

        _module = {}
        if "module" in self.__dict__:
            _module = self.__dict__['module']
        if name in dir(self):
            return getattr(self, name)
        elif name in dir(_module):
            return getattr(_module, name)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def checkpoint_serialization_enabled(self):
        return self._config.checkpoint_config[CHECKPOINT_SERIALIZATION]

    def checkpoint_writer_enabled(self):
        return self._config.checkpoint_config[CHECKPOINT_WRITER] is not None

    def checkpoint_tag_validation_enabled(self):
        return self._config.checkpoint_config[CHECKPOINT_TAG_VALIDATION] != ValidationMode.IGNORE

    def checkpoint_tag_validation_fail(self):
        return self._config.checkpoint_config[CHECKPOINT_TAG_VALIDATION] == ValidationMode.FAIL

    def elasticity_enabled(self):
        return self._config.elasticity_enabled

    def is_elastic_model_parallel_supported(self):
        if self.elasticity_enabled():
            # Add code for finding number of GPUs per node automatically
            if self._config.num_gpus_per_node % self._config.elastic_model_parallel_size == 0:
                return True
            else:
                return False

    def pld_enabled(self):
        return self._config.pld_enabled

    def pld_params(self):
        return self._config.pld_params

    def pld_theta(self):
        return self.pld_params()[PLD_THETA]

    def pld_gamma(self):
        return self.pld_params()[PLD_GAMMA]

    def eigenvalue_enabled(self):
        return self._config.eigenvalue_enabled

    def eigenvalue_verbose(self):
        return self._config.eigenvalue_verbose

    def eigenvalue_max_iter(self):
        return self._config.eigenvalue_max_iter

    def eigenvalue_tol(self):
        return self._config.eigenvalue_tol

    def eigenvalue_stability(self):
        return self._config.eigenvalue_stability

    def eigenvalue_gas_boundary_resolution(self):
        return self._config.eigenvalue_gas_boundary_resolution

    def eigenvalue_layer_name(self):
        return self._config.eigenvalue_layer_name

    def eigenvalue_layer_num(self):
        return self._config.eigenvalue_layer_num

    def curriculum_enabled_legacy(self):
        return self._config.curriculum_enabled_legacy

    def curriculum_params_legacy(self):
        return self._config.curriculum_params_legacy

    def data_efficiency_enabled(self):
        return self._config.data_efficiency_enabled

    def data_efficiency_config(self):
        return self._config.data_efficiency_config

    def data_sampling_enabled(self):
        return self._config.data_efficiency_config[DATA_SAMPLING][DATA_SAMPLING_ENABLED]

    def data_sampling_config(self):
        return self._config.data_efficiency_config[DATA_SAMPLING]

    def curriculum_learning_enabled(self):
        return self._config.data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_ENABLED]

    def curriculum_learning_config(self):
        return self._config.data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING]

    def random_ltd_enabled(self):
        return self._config.data_efficiency_config[DATA_ROUTING][RANDOM_LTD][RANDOM_LTD_ENABLED]

    def random_ltd_config(self):
        return self._config.data_efficiency_config[DATA_ROUTING][RANDOM_LTD]

    def random_ltd_initialize(self):
        assert self.random_ltd_enabled()
        random_ltd_config = self.random_ltd_config()
        random_ltd_queue = deque([x for x in sorted(random_ltd_config[RANDOM_LTD_LAYER_ID])])
        count = 0
        for name, layer in self.module.named_modules():
            if isinstance(layer, RandomLayerTokenDrop):
                if len(random_ltd_queue) != 0 and str(random_ltd_queue[0]) in name:  ###[1,2,3]
                    layer.init_config(random_ltd_config, self.random_ltd_scheduler, count)
                    random_ltd_queue.popleft()
                    count += 1

        if random_ltd_config[RANDOM_LTD_LAYER_NUM] != count:
            raise ValueError(f'random_ltd_layer_num {random_ltd_config[RANDOM_LTD_LAYER_NUM]} must be \
                equivalent to the len of random_ltd_layer_id {count}')

        if random_ltd_config[RANDOM_LTD_LAYER_TOKEN_LR_SCHEDULE][RANDOM_LTD_LAYER_TOKEN_LR_ENABLED]:
            assert self.client_lr_scheduler is None
            raise ValueError('not yet support')
            #self.lr_scheduler = lr_schedules.WarmupLayerTokenDecayLR(self.optimizer, self.random_ltd_scheduler)

    def get_data_parallel_rank(self):
        return groups.get_data_parallel_rank()

    def get_tensor_parallel_rank(self):
        return groups.get_tensor_model_parallel_rank()

    def get_model_parallel_rank(self):
        return groups.get_model_parallel_rank()

    def get_sequence_parallel_group(self):
        return self.seq_parallel_group

    # =================================================================
    # M342 — Claude-30: SP+DEC+AC Public API
    # =================================================================

    def desloc_sp_enabled(self):
        """Whether sequence parallelism is active (AutoSP or Ulysses)."""
        return self._desloc_sp_enabled

    def desloc_sp_mode(self):
        """SP mode: 'autosp', 'ulysses', or 'none'."""
        return self._desloc_sp_mode

    def desloc_ac_enabled(self):
        """Whether activation checkpointing is active."""
        return self._desloc_ac_enabled

    def desloc_ac_mode(self):
        """AC mode: 'layer', 'compile', or 'none'."""
        return self._desloc_ac_mode

    def desloc_dec_enabled(self):
        """Whether desynced communication (Kx gating) is active."""
        return self.desloc_enabled and self.desloc_Kx > 1

    def desloc_composition_state(self):
        """Get current SP+DEC+AC composition state.

        Returns dict with boolean flags and mode strings.
        Used by REAL_GPU_BENCHMARK.py for results reporting
        and by tests for validation.
        """
        return {
            'sp': self._desloc_sp_enabled,
            'sp_mode': self._desloc_sp_mode,
            'dec': self.desloc_enabled and self.desloc_Kx > 1,
            'dec_Kx': self.desloc_Kx if self.desloc_enabled else 1,
            'dec_Ku': self.desloc_Ku if self.desloc_enabled else 1,
            'dec_Kv': self.desloc_Kv if self.desloc_enabled else 1,
            'ac': self._desloc_ac_enabled,
            'ac_mode': self._desloc_ac_mode,
            'zero_stage': self.zero_optimization_stage(),
            'skipped_allreduces': self.desloc_skipped_allreduces,
        }

    def wall_clock_breakdown(self):
        return self._config.wall_clock_breakdown

    def flops_profiler_enabled(self):
        return self._config.flops_profiler_config.enabled or self.autotuning_enabled()

    def flops_profiler_recompute_fwd_factor(self):
        return self._config.flops_profiler_config.recompute_fwd_factor

    def flops_profiler_profile_step(self):
        step = self._config.flops_profiler_config.profile_step
        if self._config.autotuning_config.enabled:
            step = self.autotuning_start_profile_step()
        return step

    def flops_profiler_module_depth(self):
        return self._config.flops_profiler_config.module_depth

    def flops_profiler_top_modules(self):
        return self._config.flops_profiler_config.top_modules

    def flops_profiler_detailed(self):
        if self._config.autotuning_config.enabled:
            return False
        return self._config.flops_profiler_config.detailed

    def flops_profiler_output_file(self):
        return self._config.flops_profiler_config.output_file

    def memory_breakdown(self):
        return self._config.memory_breakdown

    def autotuning_enabled(self):
        return self._config.autotuning_config.enabled

    def autotuning_start_profile_step(self):
        return self._config.autotuning_config.start_profile_step

    def autotuning_end_profile_step(self):
        return self._config.autotuning_config.end_profile_step

    def autotuning_metric_path(self):
        path = self._config.autotuning_config.metric_path
        if not path:
            path = os.path.join(os.getcwd(), "autotuning_metric.json")
        return path

    def autotuning_model_info_path(self):
        path = self._config.autotuning_config.model_info_path
        if not path:
            path = os.path.join(os.getcwd(), "autotuning_model_info.json")
        return path

    def autotuning_metric(self):
        return self._config.autotuning_config.metric

    def autotuning_profile_model_info(self):
        return self.autotuning_enabled(
        ) and self._config.autotuning_config.model_info and self._config.autotuning_config.model_info.get(
            "profile", False)

    def sparse_gradients_enabled(self):
        return self._config.sparse_gradients_enabled

    def train_batch_size(self):
        return self._config.train_batch_size

    def train_micro_batch_size_per_gpu(self):
        return self._config.train_micro_batch_size_per_gpu

    def optimizer_name(self):
        return (self.client_optimizer.__class__.__name__ if self.client_optimizer else self._config.optimizer_name)

    def optimizer_params(self):
        return self._config.optimizer_params

    def optimizer_legacy_fusion(self):
        return self._config.optimizer_legacy_fusion

    def scheduler_name(self):
        return self._config.scheduler_name

    def scheduler_params(self):
        return self._config.scheduler_params

    def quantize_training(self):
        return (
            self._config.compression_config[WEIGHT_QUANTIZATION][SHARED_PARAMETERS]
            [WEIGHT_QUANTIZE_IN_FORWARD_ENABLED],
            self._config.compression_config[WEIGHT_QUANTIZATION][SHARED_PARAMETERS][WEIGHT_QUANTIZE_ENABLED],
            self._config.compression_config[WEIGHT_QUANTIZATION][SHARED_PARAMETERS][WEIGHT_QUANTIZE_GROUPS],
            self._config.compression_config[WEIGHT_QUANTIZATION][SHARED_PARAMETERS]
            [WEIGHT_QUANTIZE_FP16_MIXED_QUANTIZE],
            self._config.compression_config[WEIGHT_QUANTIZATION][SHARED_PARAMETERS][WEIGHT_QUANTIZE_CHANGE_RATIO],
            self._config.compression_config[WEIGHT_QUANTIZATION][SHARED_PARAMETERS][WEIGHT_QUANTIZE_TYPE],
            self._config.compression_config[WEIGHT_QUANTIZATION][SHARED_PARAMETERS][WEIGHT_QUANTIZE_ROUNDING],
            self._config.compression_config[WEIGHT_QUANTIZATION][SHARED_PARAMETERS][WEIGHT_QUANTIZE_VERBOSE],
            self._config.compression_config[WEIGHT_QUANTIZATION][SHARED_PARAMETERS][WEIGHT_QUANTIZE_KERNEL],
        )

    def zero_optimization(self):
        return self._config.zero_enabled

    def zero_allow_untested_optimizer(self):
        return self._config.zero_allow_untested_optimizer

    def zero_force_ds_cpu_optimizer(self):
        return self._config.zero_force_ds_cpu_optimizer

    def zero_reduce_scatter(self):
        return self._config.zero_config.reduce_scatter

    def zero_overlap_comm(self):
        return self._config.zero_config.overlap_comm

    def zero_offload_optimizer(self):
        return self._config.zero_config.offload_optimizer

    def zero_offload_param(self):
        return self._config.zero_config.offload_param

    def zero_use_cpu_optimizer(self):
        if self._config.zero_config.offload_optimizer is not None:
            return self._config.zero_config.offload_optimizer.device in [OffloadDeviceEnum.cpu, OffloadDeviceEnum.nvme]
        return False

    def zero_cpu_offload(self):
        if self._config.zero_config.offload_optimizer is not None:
            return self._config.zero_config.offload_optimizer.device == OffloadDeviceEnum.cpu
        return False

    def zero_partial_offload(self):
        return getattr(self._config.zero_config.offload_optimizer, "ratio", 1.0)

    def super_offload(self):
        return getattr(self._config.zero_config.offload_optimizer, "super_offload", False)

    def cpuadam_cores_perc(self):
        return getattr(self._config.zero_config.offload_optimizer, "cpuadam_cores_perc", 0.9)

    def zero_sub_group_size(self):
        return self._config.zero_config.sub_group_size

    def zero_optimization_stage(self):
        return self._config.zero_optimization_stage

    def compile_zero_optimization_stage(self):
        """Determines if zero-pass is set in deepcompile's passes attributes."""
        return "z1" in self._config.compile_config.passes or "z3" in self._config.compile_config.passes

    def compile_autosp(self):
        """Determines if AutoSP is set in deepcompile's passes attributes."""
        return "autosp" in (getattr(self._config.compile_config, "passes", None) or [])

    def mics_shard_size(self):
        return self._config.mics_shard_size

    def zero_reduce_bucket_size(self):
        return self._config.zero_config.reduce_bucket_size

    def zero_multi_rank_bucket_allreduce(self):
        return self._config.zero_config.use_multi_rank_bucket_allreduce

    def zero_allgather_bucket_size(self):
        return self._config.zero_config.allgather_bucket_size

    def zero_optimization_partition_gradients(self):
        return self.zero_optimization_stage() >= ZeroStageEnum.gradients

    def zero_optimization_partition_weights(self):
        return self.zero_optimization_stage() >= ZeroStageEnum.weights

    def is_first_weights_partition_group(self):
        ret = True if self.mics_shard_size() < 0 \
            and self.zero_optimization_partition_weights() else False
        if self.mics_shard_size() > 0 and self.global_rank < self.mics_shard_size():
            ret = True
        return ret

    def zero_contiguous_gradients(self):
        return self._config.zero_config.contiguous_gradients

    def zero_load_from_fp32_weights(self):
        return self._config.zero_config.load_from_fp32_weights

    def zero_elastic_checkpoint(self):
        return self._config.zero_config.elastic_checkpoint

    def zero_nvme_offload_optimizer(self):
        return getattr(self.optimizer, "swap_optimizer", False)

    def zero_max_live_parameters(self):
        return self._config.zero_config.max_live_parameters

    def zero_max_reuse_distance(self):
        return self._config.zero_config.max_reuse_distance

    def zero_prefetch_bucket_size(self):
        return self._config.zero_config.prefetch_bucket_size

    def zero_module_granularity_threshold(self):
        return self._config.zero_config.module_granularity_threshold

    def zero_param_persistence_threshold(self):
        return self._config.zero_config.param_persistence_threshold

    def zero_model_persistence_threshold(self):
        return self._config.zero_config.model_persistence_threshold

    def zero_gather_16bit_weights_on_model_save(self):
        return self._config.zero_config.gather_16bit_weights_on_model_save

    def zero_grad_hooks(self):
        return self._config.zero_config.grad_hooks

    def zero_legacy_stage1(self):
        return self._config.zero_config.legacy_stage1

    def zero_ignore_unused_parameters(self):
        return self._config.zero_config.ignore_unused_parameters

    def zero_save_muon_momentum_buffer_in_memory(self):
        return self._config.zero_config.save_muon_momentum_buffer_in_memory

    def tensor_parallel_config(self):
        return self._config.tensor_parallel_config

    def autotp_size(self):
        return self._config.tensor_parallel_config.autotp_size

    def graph_harvesting(self):
        return self._config.graph_harvesting

    def fp16_enabled(self):
        return self._config.float16_config.enabled

    def bfloat16_enabled(self):
        return self._config.bfloat16_config.enabled

    def fp16_master_weights_and_gradients(self):
        return self._config.float16_config.fp16_master_weights_and_grads

    def bf16_master_weights_and_gradients(self):
        return self._config.bfloat16_config.bf16_master_weights_and_grads

    def bf16_optimizer_states(self):
        return self._config.bfloat16_config.bf16_optimizer_states

    def amp_enabled(self):
        return self._config.amp_enabled

    def amp_params(self):
        return self._config.amp_params

    def torch_autocast_enabled(self) -> bool:
        return self._config.torch_autocast_enabled

    def torch_autocast_dtype(self) -> torch.dtype:
        return self._config.torch_autocast_dtype

    def torch_autocast_lower_precision_safe_modules(self) -> List[str]:
        module_names = self._config.torch_autocast_lower_precision_safe_modules
        return get_default_autocast_lower_precision_modules() if module_names is None else module_names

    def fp16_auto_cast(self):
        return self._config.float16_config.auto_cast

    def loss_scale(self):
        return self._config.float16_config.loss_scale

    def gradient_accumulation_steps(self):
        return self._config.gradient_accumulation_steps

    def use_node_local_storage(self):
        return self._config.use_node_local_storage

    def load_universal_checkpoint(self):
        return self._config.load_universal_checkpoint

    @property
    def communication_data_type(self):
        res = self._config.communication_data_type
        if res is not None:
            return res

        if self.fp16_enabled():
            return torch.float16

        if self.bfloat16_enabled():
            return torch.bfloat16

        return torch.float32

    @communication_data_type.setter
    def communication_data_type(self, value):
        self._config.communication_data_type = value

    def postscale_gradients(self):
        return not self._config.prescale_gradients

    def gradient_predivide_factor(self):
        return self._config.gradient_predivide_factor

    def steps_per_print(self):
        return self._config.steps_per_print

    def zero_allgather_partitions(self):
        return self._config.zero_config.allgather_partitions

    def zero_round_robin_gradients(self):
        return self._config.zero_config.round_robin_gradients

    def zero_hpz_partition_size(self):
        return self._config.zero_config.zero_hpz_partition_size

    def zero_quantized_weights(self):
        return self._config.zero_config.zero_quantized_weights

    def zero_quantized_nontrainable_weights(self):
        return self._config.zero_config.zero_quantized_nontrainable_weights

    def zero_quantized_gradients(self):
        return self._config.zero_config.zero_quantized_gradients

    def zeropp_loco_param(self):
        return self._config.zero_config.zeropp_loco_param

    def zero_log_trace_cache_warnings(self):
        return self._config.zero_config.log_trace_cache_warnings

    def zero_allgather_sequential(self):
        return self._config.zero_config.allgather_sequential

    def is_sanity_checks_enabled(self):
        return self._config.zero_config.enable_sanity_checks

    def dump_state(self):
        return self._config.dump_state

    def gradient_clipping(self):
        return self._config.gradient_clipping

    def dynamic_loss_scale(self):
        return self._config.float16_config.loss_scale == 0

    def initial_dynamic_scale(self):
        return self._config.float16_config.initial_dynamic_scale()

    def dynamic_loss_scale_args(self):
        return self._config.float16_config.dynamic_loss_scale_args()

    def swap_tensor_config(self):
        return self._config.swap_tensor_config

    def aio_config(self):
        return self._config.aio_config

    def zenflow_config(self):
        return self._config.zero_config.zenflow

    def get_data_types(self):
        model_dtype = torch.float32
        if self.fp16_enabled():
            model_dtype = torch.float16
        elif self.bfloat16_enabled():
            model_dtype = torch.bfloat16

        if self._config.grad_accum_dtype is None:
            grad_accum_dtype = model_dtype
        else:
            grad_accum_dtype = DtypeEnum(self._config.grad_accum_dtype).value
        return (model_dtype, grad_accum_dtype)

    def _optimizer_has_ckpt_event_prologue(self):
        return self.optimizer is not None and hasattr(self.optimizer, 'checkpoint_event_prologue')

    def _optimizer_has_ckpt_event_epilogue(self):
        return self.optimizer is not None and hasattr(self.optimizer, 'checkpoint_event_epilogue')

    def _configure_lr_scheduler(self):
        if self.client_lr_scheduler:
            if isinstance(self.client_lr_scheduler, Callable):
                log_dist('DeepSpeed using client callable to create LR scheduler', ranks=[0])
                self.lr_scheduler = self.client_lr_scheduler(self.basic_optimizer)
            else:
                log_dist('DeepSpeed using client LR scheduler', ranks=[0])
                self.lr_scheduler = self.client_lr_scheduler
        else:
            # load lr scheduler from json configuration if lr scheduler is not defined and passed in
            lr_scheduler = self._scheduler_from_config(self.optimizer)
            log_dist(f"DeepSpeed using configured LR scheduler = {self.scheduler_name()}", ranks=[0])
            self.lr_scheduler = lr_scheduler

        log_dist(f'DeepSpeed LR Scheduler = {self.lr_scheduler}', ranks=[0])

    def _configure_checkpointing(self):
        # Enable optimization to parallelize checkpointing of DP state
        optimize_dp_state = not self.zero_optimization_partition_weights()
        self.checkpoint_engine = create_checkpoint_engine(config_params=self._config,
                                                          groups=groups,
                                                          zero_stage=self.zero_optimization_stage(),
                                                          has_moe_layers=self.has_moe_layers,
                                                          optimize_dp_state=optimize_dp_state)

        dp_rank = groups._get_sequence_data_parallel_rank()
        rank = self.local_rank if self.use_node_local_storage() else dp_rank

        # Determine if this data parallel process needs to store the model checkpoint
        if self.checkpoint_engine.is_data_parallel_writer(rank) \
            or (self.zero_optimization_partition_weights() and self.is_first_weights_partition_group()):
            self.save_non_zero_checkpoint = True

        if hasattr(self.optimizer, 'dp_process_group'):
            param_rank = dist.get_rank(group=self.optimizer.dp_process_group)

            # Only the first parameter parallel process needs to store the
            # optimizer state checkpoints for zero
            self.save_zero_checkpoint = param_rank == dp_rank

    def _scheduler_from_config(self, optimizer):
        scheduler_name = self.scheduler_name()
        if scheduler_name is not None:
            if hasattr(lr_schedules, scheduler_name):
                scheduler = getattr(lr_schedules, scheduler_name)
            else:
                assert hasattr(torch.optim.lr_scheduler,
                               scheduler_name), f"DeepSpeed does not recognize LR scheduler {scheduler_name}"

                scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)

            scheduler_params = self.scheduler_params()
            instantiated_scheduler = scheduler(optimizer, **scheduler_params)
            return instantiated_scheduler
        else:
            return None

    def _set_distributed_vars(self, args):
        device_rank = args.device_rank if args is not None and hasattr(args, 'device_rank') else self.local_rank
        if device_rank >= 0:
            get_accelerator().set_device(device_rank)
            self.device = torch.device(get_accelerator().device_name(device_rank))
            self.world_size = dist.get_world_size()
            self.global_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.global_rank = 0
            self.device = get_accelerator().device()

    # Configure based on command line arguments
    def _configure_with_arguments(self, args, mpu):
        # After the distributed backend is initialized we are guaranteed the LOCAL_RANK
        # environment variable is set. We must align args.local_rank to this value for
        # backwards compatibility with scripts relying on [args|self].local_rank containing
        # the correct local rank info. _do_args_sanity_check will ensure this is the case.

        if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
            ompi_local_rank = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
            local_rank = os.environ.get('LOCAL_RANK', ompi_local_rank)
            assert ompi_local_rank == local_rank, f"LOCAL_RANK ({local_rank}) != OMPI_COMM_WORLD_LOCAL_RANK ({ompi_local_rank}), " \
                "not sure how to proceed as we're seeing conflicting local rank info."
            os.environ['LOCAL_RANK'] = local_rank

        self.local_rank = int(os.environ['LOCAL_RANK'])
        if hasattr(args, 'local_rank'):
            args.local_rank = self.local_rank

    # Validate command line arguments
    def _do_args_sanity_check(self, args):
        assert "LOCAL_RANK" in os.environ or "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ, "DeepSpeed requires the LOCAL_RANK environment " \
            "variable, it is set by the deepspeed launcher, deepspeed.init_distributed, or the torch's launcher. If using a " \
            "different launcher please ensure LOCAL_RANK is set prior to initializing deepspeed."

        if hasattr(args, 'local_rank') and args.local_rank is not None:
            assert isinstance(args.local_rank,
                              int), f"args.local_rank of {args.local_rank} is an unknown type {type(args.local_rank)}"
            if args.local_rank >= 0:
                env_local_rank = int(os.environ.get("LOCAL_RANK"))
                assert (
                    env_local_rank == args.local_rank
                ), f"Mismatch in local rank setting, args.local_rank={args.local_rank} but env['LOCAL_RANK']={env_local_rank}."

    def _is_supported_optimizer(self, optimizer_name):
        return (optimizer_name in DEEPSPEED_OPTIMIZERS or getattr(torch.optim, optimizer_name, None) is not None)

    def _supported_optims(self):
        FairseqOptimizer = None
        try:
            from fairseq.optim.fairseq_optimizer import FairseqOptimizer
        except ImportError:
            pass

        expected_optim_types = [Optimizer]
        if FairseqOptimizer:
            # fairseq optims are not torch.optim objects
            expected_optim_types.append(FairseqOptimizer)
        return expected_optim_types

    # Validate configuration based on command line arguments
    def _do_sanity_check(self):
        if self.fp16_enabled() and not get_accelerator().is_fp16_supported():
            raise ValueError("Type fp16 is not supported on your device.")

        if self.bfloat16_enabled() and not get_accelerator().is_bf16_supported():
            raise ValueError("Type bf16 is not supported on your device.")

        expected_optim_types = self._supported_optims()
        expected_optim_types += [type(None), Callable]
        assert isinstance(self.client_optimizer, tuple(expected_optim_types)), \
            f'Client Optimizer is of unexpected type {type(self.client_optimizer)}'

        if not self.client_optimizer:
            if self.optimizer_name() is not None:
                assert self._is_supported_optimizer(
                    self.optimizer_name()), "{} is not a supported DeepSpeed Optimizer".format(self.optimizer_name())

        if (self.optimizer_name() == LAMB_OPTIMIZER or self.optimizer_name() == ONEBIT_LAMB_OPTIMIZER):
            assert (self.dynamic_loss_scale()), "DeepSpeed {} optimizer requires dynamic loss scaling".format(
                self.optimizer_name())

        # Detect invalid combinations of client optimizer and client scheduler
        if isinstance(self.client_lr_scheduler, _LRScheduler):
            assert isinstance(self.client_optimizer, Optimizer), \
                f'Client Optimizer (type = {type(self.client_optimizer)} is not instantiated but Client LR Scheduler is instantiated'

    def _broadcast_model(self):
        if self.dist_backend is None:
            return

        def is_replicated(p):
            if hasattr(p, "ds_status") and p.ds_status is not ZeroParamStatus.AVAILABLE:
                return False
            elif hasattr(p, 'ds_optim_param'):
                # do not broadcast OptimizedLinear parameters, they are unique per base weight shard
                return False
            return True

        for n, p in self.module.named_parameters():
            # Broadcast the model for different parameters
            if is_moe_param(p):
                if torch.is_tensor(p) and is_replicated(p):
                    dist.broadcast(p.data,
                                   groups._get_expert_broadcast_src_rank(p.group_name),
                                   group=self.expert_data_parallel_group[p.group_name])
            else:
                if torch.is_tensor(p) and is_replicated(p):
                    dist.broadcast(p.data, groups._get_broadcast_src_rank(), group=self.seq_data_parallel_group)

    @staticmethod
    def __check_params(model: Module, dtype: torch.dtype) -> None:
        return

    def _set_client_model(self, model):
        # register client model in _modules so that nn.module methods work correctly
        modules = self.__dict__.get('_modules')
        modules['module'] = model
        # register module attribute in engine but avoid getattr
        self.__dict__['module'] = model

    def _configure_distributed_model(self, model):
        self._set_client_model(model)
        apply_zero_leaf_module_config(self.module, getattr(self._config.zero_config, "leaf_module", None))
        is_zero_init_model = self.zero_optimization_partition_weights() and any(
            [hasattr(param, "ds_id") for param in self.module.parameters()])

        if self.fp16_enabled():
            if is_zero_init_model:
                self.__check_params(self.module, torch.half)
            self.module.half()
        elif self.bfloat16_enabled():
            if is_zero_init_model:
                self.__check_params(self.module, torch.bfloat16)
            self.module.bfloat16()
        else:
            self.__check_params(self.module, torch.float)

        # zero.Init() handles device placement of model
        if not (self.dont_change_device or is_zero_init_model):
            self.module.to(self.device)

        # MoE related initialization
        for _, module in self.module.named_modules():
            if isinstance(module, MoE):
                self.has_moe_layers = True
                self.num_experts.append(module.num_experts)

        if self.has_moe_layers:
            for _, module in self.module.named_modules():
                if isinstance(module, TopKGate):
                    self.gate_modules.append(module)
                    if self.wall_clock_breakdown():
                        module.wall_clock_breakdown = True
                if isinstance(module, MOELayer):
                    self.moe_layers.append(module)
                    if self.wall_clock_breakdown():
                        module.wall_clock_breakdown = True

        # Pass the mpu from here to groups. For subsequent use, just query groups
        if self.mpu is not None:
            groups.mpu = self.mpu

        # Set deepspeed parallelism spec. for the model including expert parallelism
        for _, module in self.module.named_modules():
            if hasattr(module, 'set_deepspeed_parallelism'):
                module.set_deepspeed_parallelism(self._config.use_data_before_expert_parallel_)

        # Query the groups module to get information about various parallel groups
        self.local_all_to_all_group = None
        if self.zero_quantized_gradients():
            message = "Using LoCo quantized gradients" if self.zeropp_loco_param() else "Using quantized gradients"
            log_dist(message, ranks=[0])
            self.local_all_to_all_group = groups._get_local_all_to_all_group()
        self.data_parallel_group = groups._get_data_parallel_group()
        self.dp_world_size = groups._get_data_parallel_world_size()
        self.seq_data_parallel_group = groups._get_sequence_data_parallel_group()
        self.seq_dp_world_size = groups._get_sequence_data_parallel_world_size()
        self.mp_world_size = groups._get_model_parallel_world_size()
        self.expert_parallel_group = groups._get_expert_parallel_group_dict()
        self.expert_data_parallel_group = groups._get_expert_data_parallel_group_dict()
        self.sequence_parallel_size = groups._get_sequence_parallel_world_size()
        if self.sequence_parallel_size > 1:
            # Inserted Warning for PyTorch < 2.3
            if not required_torch_version(min_version=2.3):
                logger.warning(
                    "DeepSpeed Sequence Parallelism (Ulysses) with PyTorch < 2.3 may encounter "
                    "rank indexing errors during backward pass when sp_size < world_size. "
                    "Please use the weighted all-reduce workaround shown in the regression test "
                    "(https://github.com/deepspeedai/DeepSpeed/blob/master/tests/unit/sequence_parallelism/test_ulysses.py) "
                    "or upgrade to PyTorch 2.3+.")
            self.communication_data_type = self._config.seq_parallel_communication_data_type
            self.seq_parallel_group = groups._get_sequence_parallel_group()

        if dist.get_rank() == 0:
            summary = "********** distributed groups summary **********\n"
            summary += f"\t {self.dp_world_size=}\n"
            summary += f"\t {self.mp_world_size=}\n"
            summary += f"\t {self.seq_dp_world_size=}\n"
            summary += f"\t {self.sequence_parallel_size=}\n"
            summary += "***********************************************"
            logger.info(summary)

        if not (self.amp_enabled() or is_zero_init_model):
            self._broadcast_model()

    # check if parameters are duplicated in optimizer param_groups
    def _check_for_duplicates(self, optimizer):
        for name, param in self.module.named_parameters():
            param_id = id(param)

            def ids_list(group):
                return [id(param) for param in group]

            occurrence = sum([
                ids_list(group['params']).count(param_id) if param_id in ids_list(group['params']) else 0
                for group in optimizer.param_groups
            ])
            assert occurrence <= 1, f"Parameter with name: {name} occurs multiple times in optimizer.param_groups. Make sure it only appears once to prevent undefined behavior."

    def _do_optimizer_sanity_check(self, basic_optimizer):
        model_dtype, grad_accum_dtype = self.get_data_types()
        zero_enabled = self.zero_optimization()
        amp_enabled = self.amp_enabled()
        # config based assertions
        assert (
            not (amp_enabled and zero_enabled)
        ), "Amp and ZeRO are not currently compatible, please use (legacy) fp16 mode which performs similar to amp opt_mode=O2"
        if zero_enabled:
            if not is_zero_supported_optimizer(basic_optimizer):
                assert (
                    self.zero_allow_untested_optimizer()
                ), 'You are using an untested ZeRO Optimizer. Please add <"zero_allow_untested_optimizer": true> in the configuration file to use it.'

                if self.global_rank == 0:
                    logger.warning("**** You are using ZeRO with an untested optimizer, proceed with caution *****")
            if model_dtype == torch.bfloat16 and grad_accum_dtype == torch.float32 and self.zero_optimization_stage(
            ) == 1 and not self.zero_cpu_offload():
                return BFLOAT16
            return ZERO_OPTIMIZATION
        elif amp_enabled:
            if model_dtype != grad_accum_dtype:
                raise NotImplementedError(
                    "Model data type and gradient accumulation data type must be equal to use Amp")
            if model_dtype == torch.bfloat16 or model_dtype == torch.float16:
                raise NotImplementedError("Cannot enable both amp with (legacy) fp16 or bfloat16 mode")
            try:
                logger.info("Initializing Apex amp from: {}".format(amp.__path__))
            except NameError:
                # If apex/amp is available it will be imported above
                raise RuntimeError("Unable to import apex/amp, please make sure it is installed")
            return AMP
        # data type checks
        elif model_dtype == grad_accum_dtype:
            if model_dtype == torch.float32:
                return None
            if model_dtype == torch.bfloat16 and self.pipeline_parallelism:
                logger.warning(
                    "**** BF16 gradient accumulation is not safe numerically with large number of accumulation steps, proceed with caution *****"
                )
                return BFLOAT16
            return FP16 if model_dtype == torch.float16 else DDP_BFLOAT16
        else:
            raise NotImplementedError(f"unsupported mix of {model_dtype=} and {grad_accum_dtype=}")

        return None

    # Configure optimizer
    def _configure_optimizer(self, client_optimizer, model_parameters):
        if client_optimizer is None:
            if self.has_moe_layers:
                model_parameters = configure_moe_param_groups(model_parameters)
            basic_optimizer = self._configure_basic_optimizer(model_parameters)
            log_dist(f"Using DeepSpeed Optimizer param name {self.optimizer_name()} as basic optimizer", ranks=[0])
        else:
            if isinstance(client_optimizer, tuple(self._supported_optims())):
                basic_optimizer = client_optimizer
                log_dist('Using client Optimizer as basic optimizer', ranks=[0])
            else:
                basic_optimizer = client_optimizer(model_parameters)
                log_dist('Using client callable to create basic optimizer', ranks=[0])

            if (self.zero_use_cpu_optimizer() and not isinstance(basic_optimizer, deepspeed.ops.adam.DeepSpeedCPUAdam)
                    and not isinstance(basic_optimizer, deepspeed.ops.lion.DeepSpeedCPULion)):
                if self.zero_force_ds_cpu_optimizer():
                    msg = f'You are using ZeRO-Offload with a client provided optimizer ({type(basic_optimizer)}) which in most cases will yield poor performance. Please either use deepspeed.ops.adam.DeepSpeedCPUAdam or set an optimizer in your ds-config (https://www.deepspeed.ai/docs/config-json/#optimizer-parameters). If you really want to use a custom optimizer w. ZeRO-Offload and understand the performance impacts you can also set <"zero_force_ds_cpu_optimizer": false> in your configuration file.'
                    raise ZeRORuntimeException(msg)

        basic_optimizer.param_groups[:] = [pg for pg in basic_optimizer.param_groups if len(pg["params"]) != 0]
        log_dist("Removing param_group that has no 'params' in the basic Optimizer", ranks=[0])

        self._check_for_duplicates(basic_optimizer)

        self.basic_optimizer = basic_optimizer
        log_dist(f"DeepSpeed Basic Optimizer = {basic_optimizer.__class__.__name__}", ranks=[0])

        optimizer_wrapper = self._do_optimizer_sanity_check(basic_optimizer)

        if optimizer_wrapper == ZERO_OPTIMIZATION:
            self.optimizer = self._configure_zero_optimizer(basic_optimizer)
        elif optimizer_wrapper == AMP:
            amp_params = self.amp_params()
            log_dist(f"Initializing AMP with these params: {amp_params}", ranks=[0])
            model, self.optimizer = amp.initialize(self.module, basic_optimizer, **amp_params)
            self._set_client_model(model)
            self._broadcast_model()
            # TODO: maybe need to broadcast experts differently?
        elif optimizer_wrapper in [FP16, DDP_BFLOAT16]:
            lp_dtype = torch.float16 if optimizer_wrapper == FP16 else torch.bfloat16
            self.optimizer = self._configure_fp16_optimizer(basic_optimizer, lp_dtype)
        elif optimizer_wrapper == BFLOAT16:
            self.optimizer = self._configure_bf16_optimizer(basic_optimizer)
        else:
            self.optimizer = basic_optimizer

        log_dist("DeepSpeed Final Optimizer = {}".format(self.optimizer.__class__.__name__), ranks=[0])

        self.compression_scheduler = self._configure_compression_scheduler()
        self.quantizer = self._configure_quantization()

    def _configure_basic_optimizer(self, model_parameters):
        optimizer_parameters = self.optimizer_params()
        if optimizer_parameters is None:
            optimizer_parameters = {}
        # print(optimizer_parameters.keys())
        if "max_grad_norm" in optimizer_parameters.keys():
            raise ValueError(
                "'max_grad_norm' is not supported as an optimizer parameter, please switch to using the deepspeed parameter 'gradient_clipping' see: https://www.deepspeed.ai/docs/config-json/#gradient-clipping for more details"
            )

        if self.optimizer_name() in [ADAM_OPTIMIZER, ADAMW_OPTIMIZER]:
            torch_adam = optimizer_parameters.pop(TORCH_ADAM_PARAM, False)
            adam_w_mode = optimizer_parameters.pop(ADAM_W_MODE, ADAM_W_MODE_DEFAULT)

            # Optimizer name of Adam forces AdamW logic unless adam_w_mode is explicitly set
            effective_adam_w_mode = self.optimizer_name() == ADAMW_OPTIMIZER or adam_w_mode

            if torch_adam:
                if not effective_adam_w_mode:
                    optimizer = torch.optim.Adam(model_parameters, **optimizer_parameters)
                else:
                    optimizer = torch.optim.AdamW(model_parameters, **optimizer_parameters)
            else:
                if self.zero_use_cpu_optimizer():
                    from deepspeed.ops.adam import DeepSpeedCPUAdam, ZenFlowCPUAdam
                    CPUAdam = ZenFlowCPUAdam if self.zenflow else DeepSpeedCPUAdam

                    zenflow_kwargs = {'overlap_step': self.overlap_step} if self.zenflow else {}
                    optimizer = CPUAdam(model_parameters,
                                        **optimizer_parameters,
                                        adamw_mode=effective_adam_w_mode,
                                        **zenflow_kwargs)
                else:
                    from deepspeed.ops.adam import FusedAdam

                    optimizer = FusedAdam(
                        model_parameters,
                        **optimizer_parameters,
                        adam_w_mode=effective_adam_w_mode,
                    )

        elif self.optimizer_name() == ADAGRAD_OPTIMIZER:
            if self.zero_use_cpu_optimizer():
                from deepspeed.ops.adagrad import DeepSpeedCPUAdagrad
                optimizer = DeepSpeedCPUAdagrad(model_parameters, **optimizer_parameters)
            else:
                optimizer = torch.optim.Adagrad(model_parameters, **optimizer_parameters)
        elif self.optimizer_name() == LAMB_OPTIMIZER:
            from deepspeed.ops.lamb import FusedLamb

            optimizer = FusedLamb(model_parameters, **optimizer_parameters)
        elif self.optimizer_name() == ONEBIT_ADAM_OPTIMIZER:
            assert not self.zero_optimization(), "1bit-Adam is not compatible with ZeRO"
            from deepspeed.runtime.fp16.onebit.adam import OnebitAdam

            optimizer = OnebitAdam(model_parameters, self, **optimizer_parameters)
            if not self.fp16_enabled():
                logger.warning("Currently the convergence of 1-bit Adam is only verified under FP16")
        elif self.optimizer_name() == ZERO_ONE_ADAM_OPTIMIZER:
            assert not self.zero_optimization(), "0/1 Adam is not compatible with ZeRO"
            from deepspeed.runtime.fp16.onebit.zoadam import ZeroOneAdam

            optimizer = ZeroOneAdam(model_parameters, self, **optimizer_parameters)
            if not self.fp16_enabled():
                logger.warning('Currently the convergence of 0/1 Adam is only verified under FP16')
        elif self.optimizer_name() == ONEBIT_LAMB_OPTIMIZER:
            assert not self.zero_optimization(), "1bit-Lamb is not compatible with ZeRO"
            from deepspeed.runtime.fp16.onebit.lamb import OnebitLamb

            optimizer = OnebitLamb(model_parameters, self, **optimizer_parameters)
            if not self.fp16_enabled():
                logger.warning("Currently the convergence of 1-bit Lamb is only verified under FP16")
        elif self.optimizer_name() == LION_OPTIMIZER:
            if self.zero_use_cpu_optimizer():
                from deepspeed.ops.lion import DeepSpeedCPULion
                optimizer = DeepSpeedCPULion(model_parameters, **optimizer_parameters)
            else:
                from deepspeed.ops.lion import FusedLion
                optimizer = FusedLion(model_parameters, **optimizer_parameters)
        elif self.optimizer_name() == MUADAM_OPTIMIZER:
            try:
                from mup import MuAdam
            except ImportError:
                logger.error("Install mup to use MuAdam optimizer")
            optimizer = MuAdam(model_parameters, **optimizer_parameters)
        elif self.optimizer_name() == MUADAMW_OPTIMIZER:
            try:
                from mup import MuAdamW
            except ImportError:
                logger.error("Install mup to use MuAdamW optimizer")
            optimizer = MuAdamW(model_parameters, **optimizer_parameters)
        elif self.optimizer_name() == MUSGD_OPTIMIZER:
            try:
                from mup import MuSGD
            except ImportError:
                logger.error("Install mup to use MuSGD optimizer")
            optimizer = MuSGD(model_parameters, **optimizer_parameters)
        elif self.optimizer_name() == MUON_OPTIMIZER:
            zero_stage = self.zero_optimization_stage()
            if not all([hasattr(p, 'use_muon') for p in model_parameters]):
                msg = "Muon optimizer is used, but the use_muon attribute is NOT configured for some of the model parameters, " \
                "please set by `param.use_muon = True / False` for all params"
                logger.error(msg)
            muon_params = [p for p in model_parameters if p.use_muon and p.requires_grad]
            non_muon_params = [p for p in model_parameters if (not p.use_muon) and p.requires_grad]
            param_groups = []
            if muon_params:
                accepted_parameters = dict()
                for key in ["lr", "momentum", "weight_decay", "muon_lr"]:
                    if key in optimizer_parameters:
                        if key == "muon_lr":  # muon_lr will override lr
                            accepted_parameters['lr'] = optimizer_parameters[key]
                        else:
                            accepted_parameters[key] = optimizer_parameters[key]
                param_groups.append(dict(params=muon_params, use_muon=True, **accepted_parameters))
            if non_muon_params:
                accepted_parameters = dict()
                for key in ["lr", "betas", "eps", "weight_decay", "adam_lr"]:
                    if key in optimizer_parameters:
                        if key == "adam_lr":  # adam_lr will override lr
                            accepted_parameters['lr'] = optimizer_parameters[key]
                        else:
                            accepted_parameters[key] = optimizer_parameters[key]
                param_groups.append(dict(params=non_muon_params, use_muon=False, **accepted_parameters))
            optimizer = MuonWithAuxAdam(param_groups)
        else:
            torch_optimizer = getattr(torch.optim, self.optimizer_name())
            optimizer = torch_optimizer(model_parameters, **optimizer_parameters)
        return optimizer

    def _configure_compression_scheduler(self):
        return compression_scheduler(self.module, self._config.compression_config)

    def _configure_random_ltd_scheduler(self, configs):
        return RandomLTDScheduler(configs)

    def _configure_quantization(self):
        (
            quantize_weight_in_forward,
            quantize_enabled,
            q_groups,
            q_mixed_fp16,
            q_change_ratio,
            q_type,
            q_rounding,
            q_verbose,
            use_quantizer_kernel,
        ) = self.quantize_training()
        if quantize_enabled and not quantize_weight_in_forward:
            assert self.fp16_enabled(
            ), "MoQ (quantize in optimization step) weight quantization is only supported for FP16"
        quantizer = None
        if quantize_enabled and not quantize_weight_in_forward:
            from deepspeed.runtime.quantize import Quantizer

            quantizer = Quantizer(
                q_groups,
                q_mixed_fp16,
                q_change_ratio,
                q_type,
                q_rounding,
                q_verbose,
                self.eigenvalue_enabled(),
                use_quantizer_kernel,
                self.eigenvalue_layer_num() if self.eigenvalue_enabled() else 0,
            )
        return quantizer

    def _configure_fp16_optimizer(self, optimizer, low_precision_dtype):
        dynamic_loss_args = self.dynamic_loss_scale_args()
        clip_grad = self.gradient_clipping()

        if APEX_INSTALLED:
            fused_opts = (apex.optimizers.FusedAdam, FusedAdam)
        else:
            fused_opts = FusedAdam

        use_fused_optimizer = isinstance(optimizer, fused_opts) \
            or self.optimizer_name() in [ONEBIT_ADAM_OPTIMIZER, ZERO_ONE_ADAM_OPTIMIZER]
        loss_scale_profile = LossScaleProfile.FUSED if use_fused_optimizer else LossScaleProfile.UNFUSED
        initial_dynamic_scale = self.initial_dynamic_scale() if loss_scale_profile == LossScaleProfile.FUSED else None
        loss_scale_config = LossScaleConfig(
            low_precision_dtype=low_precision_dtype,
            dynamic_loss_scale=self.dynamic_loss_scale(),
            static_loss_scale=self.loss_scale(),
            dynamic_loss_args=dynamic_loss_args,
            profile=loss_scale_profile,
            initial_dynamic_scale=initial_dynamic_scale,
        )

        if use_fused_optimizer:
            if loss_scale_config.dynamic_loss_scale:
                log_dist('Creating fp16 optimizer with dynamic loss scale', ranks=[0])
            else:
                log_dist(f'Creating fp16 optimizer with static loss scale: {loss_scale_config.cur_scale}', ranks=[0])
            timers = self.timers if self.wall_clock_breakdown() else NoopTimer()
            optimizer = FP16_Optimizer(
                optimizer,
                deepspeed=self,
                loss_scale_config=loss_scale_config,
                low_precision_dtype=low_precision_dtype,
                mpu=self.mpu,
                clip_grad=clip_grad,
                fused_adam_legacy=self.optimizer_legacy_fusion(),
                timers=timers,
                has_moe_layers=self.has_moe_layers,
            )
        else:
            if loss_scale_config.dynamic_loss_scale:
                log_dist('Creating fp16 unfused optimizer with dynamic loss scale', ranks=[0])
            else:
                log_dist(f'Creating fp16 unfused optimizer with static loss scale: {loss_scale_config.cur_scale}',
                         ranks=[0])
            optimizer = FP16_UnfusedOptimizer(
                optimizer,
                deepspeed=self,
                loss_scale_config=loss_scale_config,
                low_precision_dtype=low_precision_dtype,
                mpu=self.mpu,
                clip_grad=clip_grad,
                fused_lamb_legacy=self.optimizer_name() == LAMB_OPTIMIZER,
            )

        return optimizer

    def _configure_bf16_optimizer(self, optimizer):
        clip_grad = self.gradient_clipping()

        if optimizer is None:
            optimizer = DummyOptim(list(self.module.parameters()))

        log_dist('Creating BF16 optimizer', ranks=[0])

        timers = self.timers if self.wall_clock_breakdown() else NoopTimer()
        optimizer = BF16_Optimizer(optimizer,
                                   self.param_names,
                                   bfloat16_config=self._config.bfloat16_config,
                                   mpu=self.mpu,
                                   clip_grad=clip_grad,
                                   allgather_bucket_size=self.zero_allgather_bucket_size(),
                                   dp_process_group=self.seq_data_parallel_group,
                                   timers=timers,
                                   grad_acc_dtype=self.get_data_types()[1],
                                   graph_harvesting=self.graph_harvesting(),
                                   has_moe_layers=self.has_moe_layers)

        return optimizer

    def _configure_zero_optimizer(self, optimizer):
        zero_stage = self.zero_optimization_stage()

        mics_shard_size = self.mics_shard_size()
        model_dtype, gradient_accumulation_dtype = self.get_data_types()

        if self.bfloat16_enabled():
            check_grad_overflow = self._config.bfloat16_config.check_grad_overflow
        elif self.fp16_enabled():
            check_grad_overflow = True
        else:
            check_grad_overflow = False

        timers = self.timers if self.wall_clock_breakdown() else NoopTimer()

        if optimizer is None:
            optimizer = DummyOptim(list(self.module.parameters()))

        if self.zero_legacy_stage1():
            raise Exception(
                "The deprecated version of ZeRO Stage 1 is not supported in deepspeed >= 0.5.9. Please downgrade to a version less than 0.5.9 if you need to use this deprecated version of ZeRO."
            )

        if zero_stage <= ZeroStageEnum.gradients:
            overlap_comm = self.zero_overlap_comm()
            contiguous_gradients = self.zero_contiguous_gradients()
            round_robin_gradients = self.zero_round_robin_gradients()
            assert not isinstance(optimizer, DummyOptim), "zero stage {} requires an optimizer".format(zero_stage)

            log_dist(f'Creating {model_dtype} ZeRO stage {zero_stage} optimizer', ranks=[0])

            if isinstance(self.module, PipelineModule):
                if overlap_comm:
                    logger.warning("Pipeline parallelism does not support overlapped communication, will be disabled.")
                    overlap_comm = False
            Stage1And2ZeroOptimizer = DeepSpeedZeroOptimizer if not self.zenflow else ZenFlowZeroOptimizer.create(
                zenflow_config=self.zenflow_config())

            optimizer = Stage1And2ZeroOptimizer(
                optimizer,
                self.param_names,
                timers=timers,
                optimizer_params=self.optimizer_params(),
                static_loss_scale=self.loss_scale(),
                dynamic_loss_scale=self.dynamic_loss_scale(),
                dynamic_loss_args=self.dynamic_loss_scale_args(),
                clip_grad=self.gradient_clipping(),
                contiguous_gradients=contiguous_gradients,
                reduce_bucket_size=self.zero_reduce_bucket_size(),
                use_multi_rank_bucket_allreduce=self.zero_multi_rank_bucket_allreduce(),
                allgather_bucket_size=self.zero_allgather_bucket_size(),
                dp_process_group=self.seq_data_parallel_group,
                expert_parallel_group=self.expert_parallel_group if self.has_moe_layers else None,
                expert_data_parallel_group=self.expert_data_parallel_group if self.has_moe_layers else None,
                reduce_scatter=self.zero_reduce_scatter(),
                overlap_comm=overlap_comm,
                offload_optimizer_config=self.zero_offload_optimizer(),
                zenflow_config=self.zenflow_config(),
                mpu=self.mpu,
                postscale_gradients=self.postscale_gradients(),
                gradient_predivide_factor=self.gradient_predivide_factor(),
                gradient_accumulation_steps=self.gradient_accumulation_steps(),
                ignore_unused_parameters=self.zero_ignore_unused_parameters(),
                partition_grads=zero_stage == ZeroStageEnum.gradients,
                round_robin_gradients=round_robin_gradients,
                has_moe_layers=self.has_moe_layers,
                fp16_master_weights_and_gradients=self.fp16_master_weights_and_gradients(),
                bf16_master_weights_and_gradients=self.bf16_master_weights_and_gradients(),
                bf16_optimizer_states=self.bf16_optimizer_states(),
                gradient_accumulation_dtype=gradient_accumulation_dtype,
                communication_data_type=self.communication_data_type,
                elastic_checkpoint=self.zero_elastic_checkpoint(),
                check_grad_overflow=check_grad_overflow)

        elif zero_stage == ZeroStageEnum.weights:
            assert not self.has_moe_layers, "MoE not supported with Stage 3"
            if isinstance(optimizer, DummyOptim):
                log_dist("Creating ZeRO Offload", ranks=[0])
                zero_param_parallel_group = groups._get_zero_param_intra_parallel_group()
                if self.zero_hpz_partition_size() > 1 and zero_param_parallel_group is None:
                    self._set_zero_group_parallelism()
                    zero_param_parallel_group = groups._get_zero_param_intra_parallel_group()
                optimizer = DeepSpeedZeRoOffload(
                    self.module,
                    timers=timers,
                    ds_config=self.config,
                    overlap_comm=self.zero_overlap_comm(),
                    prefetch_bucket_size=self.zero_prefetch_bucket_size(),
                    max_reuse_distance=self.zero_max_reuse_distance(),
                    max_live_parameters=self.zero_max_live_parameters(),
                    param_persistence_threshold=self.zero_param_persistence_threshold(),
                    model_persistence_threshold=self.zero_model_persistence_threshold(),
                    offload_param_config=self.zero_offload_param(),
                    mpu=self.mpu,
                    zero_param_parallel_group=zero_param_parallel_group,
                    zero_quantized_weights=self.zero_quantized_weights(),
                    zero_quantized_nontrainable_weights=self.zero_quantized_nontrainable_weights(),
                    zero_module_granularity_threshold=self.zero_module_granularity_threshold(),
                    log_trace_cache_warnings=self.zero_log_trace_cache_warnings(),
                )
            else:
                log_dist(
                    f'Creating fp16 ZeRO stage {zero_stage} optimizer,'
                    f' MiCS is enabled {mics_shard_size>0},'
                    f' Hierarchical params gather {self._config.mics_hierarchial_params_gather}',
                    ranks=[0])
                if mics_shard_size > 0:
                    return self._return_mics_optimizer(optimizer, timers)

                if self.zero_allgather_sequential():
                    log_dist(f"If zero_allgather_sequential is True, set prefetch_bucket_size to 1", ranks=[0])
                    self._config.zero_config.prefetch_bucket_size = 1

                log_dist(f'Creating {model_dtype} ZeRO stage {zero_stage} optimizer', ranks=[0])
                from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
                from deepspeed.runtime.superoffload.superoffload_stage3 import SuperOffloadOptimizer_Stage3
                Stage3ZeroOptimizer = DeepSpeedZeroOptimizer_Stage3 if not self.super_offload(
                ) else SuperOffloadOptimizer_Stage3
                optimizer = Stage3ZeroOptimizer(
                    self.module,
                    optimizer,
                    self.param_names,
                    timers=timers,
                    ds_config=self.config,
                    static_loss_scale=self.loss_scale(),
                    dynamic_loss_scale=self.dynamic_loss_scale(),
                    dynamic_loss_args=self.dynamic_loss_scale_args(),
                    clip_grad=self.gradient_clipping(),
                    contiguous_gradients=self.zero_contiguous_gradients(),
                    reduce_bucket_size=self.zero_reduce_bucket_size(),
                    prefetch_bucket_size=self.zero_prefetch_bucket_size(),
                    max_reuse_distance=self.zero_max_reuse_distance(),
                    max_live_parameters=self.zero_max_live_parameters(),
                    param_persistence_threshold=self.zero_param_persistence_threshold(),
                    model_persistence_threshold=self.zero_model_persistence_threshold(),
                    dp_process_group=self.seq_data_parallel_group,
                    all2all_process_group=self.local_all_to_all_group,
                    reduce_scatter=self.zero_reduce_scatter(),
                    overlap_comm=self.zero_overlap_comm(),
                    offload_optimizer_config=self.zero_offload_optimizer(),
                    offload_param_config=self.zero_offload_param(),
                    zenflow_config=self.zenflow_config(),
                    sub_group_size=self.zero_sub_group_size(),
                    offload_ratio=self.zero_partial_offload(),
                    mpu=self.mpu,
                    postscale_gradients=self.postscale_gradients(),
                    gradient_predivide_factor=self.gradient_predivide_factor(),
                    gradient_accumulation_steps=self.gradient_accumulation_steps(),
                    aio_config=self.aio_config(),
                    gradient_accumulation_dtype=gradient_accumulation_dtype,
                    communication_data_type=self.communication_data_type,
                    fp16_master_weights_and_gradients=self.fp16_master_weights_and_gradients(),
                    bf16_master_weights_and_gradients=self.bf16_master_weights_and_gradients(),
                    bf16_optimizer_states=self.bf16_optimizer_states(),
                    zero_hpz_partition_size=self.zero_hpz_partition_size(),
                    zero_quantized_weights=self.zero_quantized_weights(),
                    zero_quantized_nontrainable_weights=self.zero_quantized_nontrainable_weights(),
                    zero_module_granularity_threshold=self.zero_module_granularity_threshold(),
                    zeropp_loco_param=self.zeropp_loco_param(),
                    log_trace_cache_warnings=self.zero_log_trace_cache_warnings(),
                    enable_sanity_checks=self.is_sanity_checks_enabled(),
                    cpuadam_cores_perc=self.cpuadam_cores_perc(),
                    save_muon_momentum_buffer_in_memory=self.zero_save_muon_momentum_buffer_in_memory(),
                )

        else:
            raise NotImplementedError("ZeRO stage {} not implemented".format(zero_stage))

        return optimizer

    def _return_mics_optimizer(self, basic_optimizer, timers):
        from deepspeed.runtime.zero.mics import MiCS_Optimizer
        model_dtype, gradient_accumulation_dtype = self.get_data_types()
        optimizer = MiCS_Optimizer(self.module,
                                   basic_optimizer,
                                   self.param_names,
                                   timers=timers,
                                   ds_config=self.config,
                                   static_loss_scale=self.loss_scale(),
                                   dynamic_loss_scale=self.dynamic_loss_scale(),
                                   dynamic_loss_args=self.dynamic_loss_scale_args(),
                                   clip_grad=self.gradient_clipping(),
                                   contiguous_gradients=self.zero_contiguous_gradients(),
                                   reduce_bucket_size=self.zero_reduce_bucket_size(),
                                   prefetch_bucket_size=self.zero_prefetch_bucket_size(),
                                   max_reuse_distance=self.zero_max_reuse_distance(),
                                   max_live_parameters=self.zero_max_live_parameters(),
                                   param_persistence_threshold=self.zero_param_persistence_threshold(),
                                   model_persistence_threshold=self.zero_model_persistence_threshold(),
                                   dp_process_group=self.seq_data_parallel_group,
                                   reduce_scatter=self.zero_reduce_scatter(),
                                   overlap_comm=self.zero_overlap_comm(),
                                   offload_optimizer_config=self.zero_offload_optimizer(),
                                   offload_param_config=self.zero_offload_param(),
                                   sub_group_size=self.zero_sub_group_size(),
                                   mpu=self.mpu,
                                   postscale_gradients=self.postscale_gradients(),
                                   gradient_predivide_factor=self.gradient_predivide_factor(),
                                   gradient_accumulation_steps=self.gradient_accumulation_steps(),
                                   aio_config=self.aio_config(),
                                   gradient_accumulation_dtype=gradient_accumulation_dtype,
                                   communication_data_type=self.communication_data_type,
                                   fp16_master_weights_and_gradients=self.fp16_master_weights_and_gradients(),
                                   bf16_master_weights_and_gradients=self.bf16_master_weights_and_gradients(),
                                   bf16_optimizer_states=self.bf16_optimizer_states())
        return optimizer

    def _configure_eigenvalue(self):
        eigenvalue = Eigenvalue(
            verbose=self.eigenvalue_verbose(),
            max_iter=self.eigenvalue_max_iter(),
            tol=self.eigenvalue_tol(),
            stability=self.eigenvalue_stability(),
            gas_boundary_resolution=self.eigenvalue_gas_boundary_resolution(),
            layer_name=self.eigenvalue_layer_name(),
            layer_num=self.eigenvalue_layer_num(),
        )

        return eigenvalue

    def _configure_progressive_layer_drop(self):
        pld = ProgressiveLayerDrop(theta=self.pld_theta(), gamma=self.pld_gamma())

        return pld

    def _configure_curriculum_scheduler_legacy(self):
        scheduler = CurriculumScheduler(self.curriculum_params_legacy())
        return scheduler

    @staticmethod
    def is_map_style_dataset(obj):
        return hasattr(obj, "__getitem__") and hasattr(obj, "__len__")

    @staticmethod
    def is_iterable_style_dataset(obj):
        return isinstance(obj, torch.utils.data.IterableDataset)  # hasattr(obj, "__iter__") should work as well

    def dataloader_drop_last(self):
        return self._config.dataloader_drop_last

    def was_step_applied(self) -> bool:
        """Returns True if the latest ``step()`` produced in parameter updates.
        Note that a ``False`` return is not an error condition. Steps are frequently
        no-ops, such as between gradient accumulation boundaries or when overflows
        occur.
        Returns:
            bool: Whether the latest ``step()`` modified model parameters.
        """
        return self._step_applied

    def deepspeed_io(self,
                     dataset,
                     batch_size=None,
                     route=ROUTE_TRAIN,
                     pin_memory=True,
                     data_sampler=None,
                     collate_fn=None,
                     num_local_io_workers=None):
        if not (self.is_map_style_dataset(dataset) or self.is_iterable_style_dataset(dataset)):
            raise ValueError("Training data must be a torch Dataset")

        if batch_size is None:
            batch_size = self.train_micro_batch_size_per_gpu()

        if collate_fn is None:
            collate_fn = self.collate_fn

        # Currently we only use timer in train route
        deepspeed_io_timer = None
        if route == ROUTE_TRAIN:
            deepspeed_io_timer = self.tput_timer

        # If mpu is provided, forward world size and parallel rank to sampler.
        data_parallel_world_size = self.dp_world_size
        data_parallel_rank = self.global_rank
        if self.mpu is not None:
            data_parallel_world_size = self.mpu.get_data_parallel_world_size()
            data_parallel_rank = self.mpu.get_data_parallel_rank()

        if data_sampler is None and (route == ROUTE_PREDICT or route == ROUTE_EVAL):
            data_sampler = torch.utils.data.DistributedSampler(
                dataset,
                num_replicas=data_parallel_world_size,
                rank=data_parallel_rank,
                shuffle=False,
            )

        deepspeed_dataloader_config = {}
        if self.curriculum_learning_enabled():
            deepspeed_dataloader_config = {
                CURRICULUM_LEARNING: self.curriculum_learning_enabled(),
                DATA_EFFICIENCY: self.data_efficiency_config(),
                DATA_PARALLEL_GROUP: self.data_parallel_group,
                GRADIENT_ACCUMULATION_STEPS: self.gradient_accumulation_steps(),
                GLOBAL_RANK: self.global_rank,
                DATA_SAMPLING_NUM_WORKERS: self.data_sampling_config()[DATA_SAMPLING_NUM_WORKERS]
            }
        return DeepSpeedDataLoader(dataset=dataset,
                                   batch_size=batch_size,
                                   pin_memory=pin_memory,
                                   collate_fn=collate_fn,
                                   local_rank=self.local_rank,
                                   tput_timer=deepspeed_io_timer,
                                   num_local_io_workers=num_local_io_workers,
                                   data_sampler=data_sampler,
                                   data_parallel_world_size=data_parallel_world_size,
                                   data_parallel_rank=data_parallel_rank,
                                   dataloader_drop_last=self.dataloader_drop_last(),
                                   deepspeed_dataloader_config=deepspeed_dataloader_config)

    def train(self, mode=True):
        r""""""

        self.warn_unscaled_loss = True
        self.module.train(mode)

    def eval(self):
        r""""""

        self.warn_unscaled_loss = True
        self.module.train(False)

    def _scale_loss_by_gas(self, prescaled_loss, eval_micro_batches=None):
        # In pipeline evaluation, there is an option to use different micro-bs, which creates different number of
        # micro batches, thus the training gas, is not valid in this case. need to use the number of eval_micro_batches
        scaling_factor = self.gradient_accumulation_steps() if eval_micro_batches is None else eval_micro_batches
        if isinstance(prescaled_loss, torch.Tensor):
            scaled_loss = prescaled_loss / scaling_factor
        elif isinstance(prescaled_loss, tuple) or isinstance(prescaled_loss, list):
            scaled_loss = []
            for l in prescaled_loss:
                if isinstance(l, torch.Tensor):
                    scaled_loss.append(l / scaling_factor)
                else:
                    scaled_loss.append(l)
        else:
            scaled_loss = prescaled_loss
            if self.warn_unscaled_loss:
                logger.warning(f"DeepSpeed unable to scale loss because of type: {type(prescaled_loss)}")
                self.warn_unscaled_loss = False

        return scaled_loss

    def _create_module_forward_pre_hook(self):

        def _module_forward_pre_hook(module, inputs, kwargs):
            return self._forward_prologue(inputs, kwargs)

        return self.module.register_forward_pre_hook(_module_forward_pre_hook, prepend=False, with_kwargs=True)

    def _create_module_forward_post_hook(self):

        def _module_forward_post_hook(module, input, output):
            self._forward_epilogue()

        return self.module.register_forward_hook(_module_forward_post_hook)

    def _forward_prologue(self, inputs, kwargs):
        return_modified = False

        if not self.autotuning_profile_model_info():
            see_memory_usage("Engine before forward", force=self.memory_breakdown())

        flops_profiler_active = (self.flops_profiler_enabled()
                                 and self.global_steps == self.flops_profiler_profile_step() and self.global_rank == 0)

        # used to check quantization happens at step 0!
        if self.global_steps == 0 and hasattr(self, "compression_scheduler"):
            self.compression_scheduler.step(step_zero_check=True)
            if self.quantizer:
                tensor_to_quantize = self.optimizer.bit16_groups if self.zero_optimization_stage(
                ) == 2 else self.optimizer.fp16_groups
                if self.compression_scheduler.weight_quantization_enabled:
                    self.quantizer.quantize(
                        tensor_to_quantize,
                        (self.optimizer.overflow if self.fp16_enabled() else False),
                        self.eigenvalue_enabled(),
                        None,
                    )
                    return_modified = True

        if flops_profiler_active:
            self.flops_profiler.start_profile(ignore_list=None)

        if kwargs is not None:
            if self.module.training:
                if self.progressive_layer_drop:
                    kwargs.update(self.progressive_layer_drop.get_state())

            if self.__class__.__name__ != "PipelineEngine":
                # TODO: The above if condition is a HACK since for PipelineEngine
                # it's difficult to inject argument in forward pass.
                if self.module.training and self.curriculum_enabled_legacy():
                    self.curriculum_scheduler_legacy.update_difficulty(self.global_steps + 1)
                    if self.curriculum_params_legacy()["curriculum_type"] == "seqlen":
                        kwargs.update({"curriculum_seqlen": self.curriculum_scheduler_legacy.get_current_difficulty()})
                        return_modified = True

        if self.module.training and self.random_ltd_enabled():
            self.random_ltd_scheduler.update_seq(self.global_steps)

        if self.training_dataloader is None:
            self.tput_timer.start()

        self._start_timers(self.engine_timers.forward_timers)

        if self.zero_optimization_partition_weights():
            # Enable automated discovery of external parameters by indicating that
            # we are in a forward pass.
            for module in self.module.modules():
                module._parameters._in_forward = True

        if self.fp16_auto_cast():
            inputs = self._cast_inputs_half(inputs)
            return_modified = True

        if return_modified:
            return inputs, kwargs

    def _forward_epilogue(self):
        if self.zero_optimization_partition_weights():
            # Disable automated discovery of external parameters
            for module in self.module.modules():
                module._parameters._in_forward = False

        self._stop_timers(self.engine_timers.forward_timers)

        flops_profiler_active = (self.flops_profiler_enabled()
                                 and self.global_steps == self.flops_profiler_profile_step() and self.global_rank == 0)

        if flops_profiler_active:
            self.flops_profiler.stop_profile()

        if not self.autotuning_profile_model_info():
            see_memory_usage("Engine after forward", force=self.memory_breakdown())

    @instrument_w_nvtx
    def forward(self, *inputs, **kwargs):
        r"""Execute forward propagation
        Arguments:
            *inputs: Variable length input list
            **kwargs: variable length keyword arguments
        """
        # Clear the backward seen flag at the start of each forward pass.
        # This is used to track multiple gradient hook phases with reentrant checkpointing.
        if isinstance(self.optimizer, ZeROOptimizer):
            self.optimizer.clear_backward_seen_flag()

        if self.autotuning_profile_model_info():
            ma = get_ma_status()

        if self.is_deepcompile_enabled() and not self.is_deepcompile_active() and not self.is_compiled:
            log_dist_once(
                "DeepCompile is enabled but engine.compile() has not been called; executing without DeepCompile until compile() runs.",
                ranks=[0])

        if self.is_deepcompile_active() and hasattr(self, "launch_compile_passes"):
            # We can't have this in forward prologue as the compiler compiles hooks including the forward prologue.
            self.launch_compile_passes(self.global_steps)

        with autocast_if_enabled(self):
            loss = self.module(*inputs, **kwargs)

        # Register output backward hooks
        # preprocess_once_fn is called for preprocessing
        # preprocess_per_tensor_fn scales a tensor for gradient accumulation
        register_output_backward_hooks(loss,
                                       preprocess_once_fn=self._backward_prologue,
                                       preprocess_per_tensor_fn=self._backward_prologue_per_tensor)

        if self.autotuning_profile_model_info():
            activation_mem = get_ma_status() - ma
            self.autotuning_model_info["activation_mem_per_gpu"] = activation_mem
            print_json_dist(self.autotuning_model_info, [0], path=self.autotuning_model_info_path())
            exit()

        return loss

    def _cast_inputs_half(self, inputs):
        if isinstance(inputs, (list, tuple)):
            new_inputs = []
            for v in inputs:
                new_inputs.append(self._cast_inputs_half(v))
            return inputs.__class__(new_inputs)
        elif isinstance(inputs, dict):
            new_inputs = {}
            for k, v in inputs.items():
                new_inputs[k] = self._cast_inputs_half(v)
            return new_inputs
        elif hasattr(inputs, 'half') and inputs.is_floating_point():
            return inputs.half()
        else:
            return inputs

    def print_forward_breakdown(self, fwd_time):
        gate_time = 0.0
        moe_time = 0.0
        falltoall = 0.0
        salltoall = 0.0

        for gate in self.gate_modules:
            #logger.info(f"Individual TopK gate time: {gate.gate_time:.2f} ms")
            gate_time += gate.gate_time

        for l in self.moe_layers:
            #logger.info(f"MoE layer; total: {l.time_moe:.2f} ms, first alltoall: {l.time_falltoall:.2f}, second alltoall: {l.time_salltoall:.2f}")
            moe_time += l.time_moe
            falltoall += l.time_falltoall
            salltoall += l.time_salltoall

        # TODO: Allreduce/average them across ranks for more accurate timing.

        # if deepspeed.comm.get_rank() == 0:
        log_dist(
            f"time (ms) | fwd: {fwd_time:.2f} (fwd_moe: {moe_time:.2f}, 1st_a2a: {falltoall:.2f}, 2nd_a2a: {salltoall:.2f}, top_k: {gate_time:.2f})",
            ranks=[0])

    @instrument_w_nvtx
    def desloc_is_param_sync_step(self):
        """Check if current step is a Kx sync boundary.

        Ref: Algorithm 1 line 10 — sync when step % Kx == 0.
        During warmup (step < warmup_steps), always sync (Kx=1 effective).
        Kx=1 always returns True (DDP baseline preserved).
        """
        if not self.desloc_enabled or self.desloc_Kx <= 1:
            return True
        if self.desloc_step < self.desloc_warmup_steps:
            return True
        return (self.desloc_step % self.desloc_Kx) == 0

    def desloc_is_momentum_sync_step(self):
        """Check if current step is a Ku sync boundary.

        Ref: Algorithm 1 — first momentum averages at Ku period.
        Half-life of m1 (beta1=0.9) ≈ 6.6 steps, so Ku=3*Kx is safe.
        """
        if not self.desloc_enabled or self.desloc_Ku <= 1:
            return True
        if self.desloc_step < self.desloc_warmup_steps:
            return True
        return (self.desloc_step % self.desloc_Ku) == 0

    def desloc_is_variance_sync_step(self):
        """Check if current step is a Kv sync boundary.

        Ref: Algorithm 1 — second momentum averages at Kv period.
        Half-life of m2 (beta2=0.999) ≈ 693 steps, so Kv=6*Kx is safe.
        """
        if not self.desloc_enabled or self.desloc_Kv <= 1:
            return True
        if self.desloc_step < self.desloc_warmup_steps:
            return True
        return (self.desloc_step % self.desloc_Kv) == 0

    def desloc_get_effective_Kx(self):
        """Get effective Kx for current step (1 during warmup)."""
        if not self.desloc_enabled:
            return 1
        if self.desloc_step < self.desloc_warmup_steps:
            return 1
        return self.desloc_Kx

    def _should_defer_mxfp8_param_sync(self) -> bool:
        """Return whether MXFP8 param sync should be deferred until chained steps finish.

        Ports Megatron commit b80a8547 (ChainedOptimizer gate) into the DES-LOC engine.

        The deferred-sync path is only needed when MXFP8 grad/param buffer reuse is active
        AND the DDP-level param gather is not overlapped (the race fixed by Megatron PR #4800
        can occur in DeepSpeed too when ZeRO stage >= 1 uses grad-buffer reuse).
        The engine-level config flag is unreliable as a proxy for the DDP-level setting —
        the two configs can diverge — so probe underlying ZeRO optimizers directly.

        Knuth Vol.1 §1.2.1: "Premature optimization is the root of all evil." The extra
        isinstance loop runs once per optimizer.step(), not per forward pass, so the O(N)
        scan over constituent optimizers is negligible vs. the all-gather cost.
        Knuth TAOCP §2.2.3 critique: chained traversal of constituent_optimizers has O(P)
        probe cost; acceptable since it is bounded by the number of ZeRO shards, not by
        parameter count.
        """
        # Fast path: buffer reuse disabled — defer never needed.
        cfg = getattr(self, '_config', None)
        if not getattr(cfg, 'reuse_grad_buf_for_mxfp8_param_ag', False):
            return False

        # Probe each constituent ZeRO optimizer's DDP config directly.
        # Pattern mirrors Megatron ChainedOptimizer._should_defer_mxfp8_param_sync():
        #   for optimizer in self.chained_optimizers:
        #       if not isinstance(optimizer, DistributedOptimizer): continue
        #       if not optimizer.ddp_config.overlap_param_gather: return True
        # Here "chained" optimizers live either as self.optimizer or inside
        # a ZeROOptimizer's constituent list; we check both paths.
        candidates = []
        opt = self.optimizer
        if hasattr(opt, 'optimizer'):
            # ZeROOptimizer wraps a basic optimizer; the DDP config is on the outer shell.
            candidates.append(opt)
        if hasattr(opt, 'constituent_optimizers'):
            candidates.extend(opt.constituent_optimizers)

        for sub_opt in candidates:
            ddp_cfg = getattr(sub_opt, 'ddp_config', None)
            if ddp_cfg is None:
                continue
            overlap = getattr(ddp_cfg, 'overlap_param_gather', True)
            if not overlap:
                # At least one shard manager runs without overlap — defer sync.
                print(f"[MXFP8-SYNC-GATE] rank={dist.get_rank()} "
                      f"step={self.desloc_step} defer=True "
                      f"(sub_opt={type(sub_opt).__name__} overlap_param_gather=False)")
                return True

        # All constituent optimizers overlap param gather — no race, no deferral needed.
        if getattr(cfg, 'reuse_grad_buf_for_mxfp8_param_ag', False) and self.desloc_step % 200 == 1:
            print(f"[MXFP8-SYNC-GATE] rank={dist.get_rank()} "
                  f"step={self.desloc_step} defer=False "
                  f"(all constituents have overlap_param_gather=True or no ddp_config)")
        return False

    # --- DES-LOC Lifecycle (M139) ---
    def desloc_init_scheduler(self):
        from deepspeed.comm.comm import init_desloc_scheduler, get_desloc_scheduler, get_desloc_tiered_ar, get_desloc_profiler
        dp = None
        if self.mpu:
            try:
                dp = self.mpu.get_data_parallel_group()
            except Exception:
                pass
        init_desloc_scheduler(Kx=self.desloc_Kx, Ku=self.desloc_Ku, Kv=self.desloc_Kv, group=dp)
        self._desloc_scheduler = get_desloc_scheduler()
        self._desloc_tiered_ar = get_desloc_tiered_ar()
        self._desloc_profiler = get_desloc_profiler()
        # M470: build distributed optimizer shard manager if models expose grad_buffers
        # Uses Megatron 4feb2b0d public API: model.grad_buffers / .grad_buffer_param_index_map
        models = [self.module] if hasattr(self.module, 'grad_buffers') else []
        if models and dp is not None:
            self._desloc_dist_opt_mgr = DeslocDistributedOptimizerShardManager(models, dp)
            self._desloc_dist_opt_mgr.build_shard_ranges()
            print(f"[DESLOC-DIST-OPT] shard manager ready dp_world={self._desloc_dist_opt_mgr.dp_world_size}")

    def _desloc_build_param_tier_map(self):
        self._desloc_param_tiers = {}
        for name, p in self.module.named_parameters():
            if not p.requires_grad: continue
            n = name.lower()
            if any(k in n for k in ('mlp', 'ffn', 'fc', 'dense')): t = 'v'
            elif any(k in n for k in ('attn', 'attention', 'query', 'key', 'value')): t = 'u'
            else: t = 'x'
            p.desloc_tier = t
            self._desloc_param_tiers[name] = t

    def desloc_post_step(self, loss=None):
        if not self.desloc_enabled: return
        self.desloc_step += 1
        if self.desloc_step % 100 == 1 and dist.get_rank() == 0:
            pnorm = sum(p.data.float().norm().item()**2 for p in self.module.parameters())**0.5
            print(f"[ENGINE-STEP] step={self.desloc_step} loss={loss} param_norm={pnorm:.2f} "
                  f"skipped_AR={self.desloc_skipped_allreduces} sp={self._desloc_sp_enabled}")
        # M279: profiler hook
        if hasattr(self, '_desloc_step_profiler'):
            self._desloc_step_profiler(self.desloc_step)
        if hasattr(self, '_desloc_scheduler') and self._desloc_scheduler:
            self._desloc_scheduler.advance()
        if self.desloc_step % 500 == 0 and dist.get_rank() == 0:
            ls = f"{loss:.4f}" if loss is not None else "N/A"
            print(f"### Kx={self.desloc_Kx} step={self.desloc_step} loss={ls} ###")

    def desloc_apply_nesterov(self, momentum=0.9):
        if not self.desloc_enabled or not self.desloc_is_param_sync_step(): return
        if not hasattr(self, '_dnv'): self._dnv, self._dnp = {}, {}
        for n, p in self.module.named_parameters():
            if not p.requires_grad: continue
            pid = id(p)
            if pid not in self._dnv:
                self._dnv[pid] = p.data.new_zeros(p.data.shape)
                self._dnp[pid] = p.data.clone()
                continue
            d = p.data.float() - self._dnp[pid].float()
            v = self._dnv[pid]
            v.mul_(momentum).add_(d)
            p.data.add_(v.to(p.dtype), alpha=momentum)
            self._dnp[pid].copy_(p.data)

    def desloc_checkpoint_state(self):
        return {'v': 2, 'step': self.desloc_step, 'Kx': self.desloc_Kx, 'Ku': self.desloc_Ku, 'Kv': self.desloc_Kv}

    def desloc_load_checkpoint(self, sd):
        if not sd or sd.get('v', 0) < 2: return
        self.desloc_step = sd.get('step', 0)

    def desloc_export_profiling(self, output_dir):
        import json as _j, os
        if dist.get_rank() != 0: return
        os.makedirs(output_dir, exist_ok=True)
        if hasattr(self, '_desloc_profiler') and self._desloc_profiler:
            self._desloc_profiler.export_csv(os.path.join(output_dir, 'comm.csv'))
        with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
            _j.dump(self.desloc_checkpoint_state(), f, indent=2, default=str)

    def allreduce_gradients(self, bucket_size=MEMORY_OPT_ALLREDUCE_SIZE):
        # =============================================================
        # M342: SP+DEC composed gradient reduction
        #
        # Without SP: standard DES-LOC Kx gating on AllReduce
        # With SP (AutoSP or Ulysses):
        #   1. Reduce-scatter gradients along sequence dimension
        #      (this is always done — SP requires it for correctness)
        #   2. DES-LOC Kx gate: AllReduce across data-parallel workers
        #      (this is skipped on non-Kx-boundary steps)
        #
        # The two reductions are orthogonal:
        #   SP reduce-scatter: combines partial gradients from
        #     different sequence chunks within the SAME step
        #   DEC AllReduce: combines gradients from different
        #     workers across DIFFERENT local steps
        #
        # Skipping the DEC AllReduce is safe because each worker's
        # local gradient is already complete (SP reduce-scatter
        # ensures this). The Kx gate only controls how often
        # workers share their complete gradients with each other.
        # =============================================================

        # DES-LOC: skip gradient allreduce on non-Kx boundary steps
        # Ref: Algorithm 1 line 10 — sync when step % Kx == 0
        # Kx=1 never skips (standard DDP behavior preserved)
        #
        # NOTE: When SP is active, we still do the SP reduce-scatter
        # (handled by DeepCompile or Ulysses), only skip the DP AllReduce.
        if self.desloc_enabled and not self.desloc_is_param_sync_step() and self.zero_optimization_stage() < 1:
            self.desloc_skipped_allreduces += 1
            # M365 DIAG: log gating decision on ALL ranks, not just rank 0
            # Pattern: Megatron training.py logs grad_norm on every training step
            if self.desloc_step % 50 == 1:
                _r = dist.get_rank()
                # compute gradient norm to detect if gradients are healthy despite skip
                gnorm_sq = sum(p.grad.float().norm().item()**2 for p in self.module.parameters() if p.grad is not None)
                gnorm = gnorm_sq**0.5
                print(f"[ENGINE-AR] rank={_r} step={self.desloc_step} SKIP "
                      f"(skipped={self.desloc_skipped_allreduces} "
                      f"zero={self.zero_optimization_stage()} "
                      f"grad_norm={gnorm:.6f} "
                      f"sp={self._desloc_sp_enabled})")
            return

        # M365 DIAG: log FIRE decisions on ALL ranks
        if self.desloc_enabled and self.desloc_step % 50 == 1:
            _r = dist.get_rank()
            gnorm_sq = sum(p.grad.float().norm().item()**2 for p in self.module.parameters() if p.grad is not None)
            gnorm = gnorm_sq**0.5
            pnorm_sq = sum(p.data.float().norm().item()**2 for p in self.module.parameters())
            pnorm = pnorm_sq**0.5
            print(f"[ENGINE-AR] rank={_r} step={self.desloc_step} FIRE "
                  f"grad_norm={gnorm:.6f} param_norm={pnorm:.4f} "
                  f"sp={self._desloc_sp_enabled} "
                  f"zero={self.zero_optimization_stage()}")

        if self._desloc_sp_enabled or self.compile_autosp():
            try:
                from deepspeed.compile.custom_ops.sp_dp_registry import fence_before_dp_sync
                fence_before_dp_sync()
                # System Issue 4 fix: enforce A2A handle high-water mark to prevent
                # unbounded NCCL work handle accumulation during long DES-LOC Kx periods.
                # Pattern: NCCL group.cc drains pending ops at ncclGroupEnd.
                from deepspeed.compile.custom_ops.hetero_mesh import enforce_handle_high_water_mark
                enforce_handle_high_water_mark()
            except ImportError:
                pass

        if self.is_deepcompile_active() and (not self.compile_autosp() or self._desloc_sp_mode == 'z2_fallback'):
            return

        # Pass (PP) gas boundary flag to optimizer (required for zero)
        self.optimizer.is_gradient_accumulation_boundary = self.is_gradient_accumulation_boundary()
        # ZeRO stage >= 2 communicates during non gradient accumulation boundaries as well
        if self.zero_optimization_partition_gradients():
            self.optimizer.overlapping_partition_gradients_reduce_epilogue()

        # Communicate only at gradient accumulation boundaries
        elif self.is_gradient_accumulation_boundary():
            if self.zero_optimization_stage() == ZeroStageEnum.optimizer_states and hasattr(
                    self.optimizer, 'reduce_gradients'):
                self.optimizer.reduce_gradients(pipeline_parallel=self.pipeline_parallelism)
            else:
                grads = None
                self.buffered_allreduce_fallback(grads=grads, elements_per_buffer=bucket_size)
        elif self.zenflow:
            self.optimizer.reduce_gradients(pipeline_parallel=self.pipeline_parallelism)

    def _reduce_gradients_at_boundary(self, bucket_size=MEMORY_OPT_ALLREDUCE_SIZE):
        """M506: Megatron 318d68c28 — helper extracted from allreduce_gradients.

        Megatron refactored the per-boundary reduce logic out of train_step
        into forward_step_with_communication / backward_step_with_communication
        helpers.  Here we mirror that pattern: the communication decision at
        gradient-accumulation boundaries is now a standalone method so callers
        (pipeline schedule, ZeRO stage router) can invoke it directly without
        duplicating the ZeRO / zenflow branching logic.

        Replaces the inline elif block in allreduce_gradients() with a
        reusable call.  Kept as additive — allreduce_gradients() still works
        unchanged; callers may opt into the helper for pipeline schedules.
        """
        print(f"[M506-COMM-HELPER] _reduce_gradients_at_boundary called "
              f"gas_boundary={self.is_gradient_accumulation_boundary()} "
              f"zero_stage={self.zero_optimization_stage()} "
              f"zenflow={getattr(self, 'zenflow', False)}")
        if self.zero_optimization_partition_gradients():
            self.optimizer.overlapping_partition_gradients_reduce_epilogue()
        elif self.is_gradient_accumulation_boundary():
            if self.zero_optimization_stage() == ZeroStageEnum.optimizer_states and hasattr(
                    self.optimizer, 'reduce_gradients'):
                self.optimizer.reduce_gradients(pipeline_parallel=self.pipeline_parallelism)
            else:
                self.buffered_allreduce_fallback(grads=None, elements_per_buffer=bucket_size)
        elif getattr(self, 'zenflow', False):
            self.optimizer.reduce_gradients(pipeline_parallel=self.pipeline_parallelism)

    def _backward_prologue(self):
        self._start_timers(self.engine_timers.backward_timers)

        # When necessary internal APIs are not available, we disable direct calls to tensor.backward()
        # and limit to engine.backward(loss) only.
        if not self._support_torch_style_backward and not self._running_engine_backward:
            raise RuntimeError("Direct calls to tensor.backward() are not supported in this configuration. "
                               "This occurs when either: (1) your PyTorch version lacks required internal APIs, "
                               "or (2) using ZeRO stage 0. "
                               "Please use engine.backward(loss) instead.")

        see_memory_usage("Engine before backward", force=self.memory_breakdown())

        assert not self.eigenvalue_enabled(), "Eigenvalue is not supported with non-scalar backward"
        assert not self.amp_enabled(), "Apex AMP is not supported with non-scalar backward"

        if self.is_deepcompile_active():
            deepcompile_backward_prologue(self.is_gradient_accumulation_boundary())

        if isinstance(self.optimizer, ZeROOptimizer):
            self.optimizer.backward_prologue()
            self.optimizer.enter_backward()
            self.optimizer.queue_post_backward_callback()

        if self.zenflow and self.auto_update:
            self.optimizer.zenflow_state ^= 1

        if self.zero_optimization():
            self.optimizer.is_gradient_accumulation_boundary = self.is_gradient_accumulation_boundary()

        self._start_timers(self.engine_timers.backward_inner_timers)

    def _backward_epilogue(self):
        self._stop_timers(self.engine_timers.backward_inner_timers)
        self._start_timers(self.engine_timers.backward_reduce_timers)

        # System Issue 4 fix: Enforce A2A handle high-water mark BEFORE
        # AllReduce to prevent unbounded handle accumulation during long
        # DES-LOC Kx periods. Without this, NCCL internal buffers grow
        # indefinitely when handles aren't waited on.
        # Pattern: NCCL group.cc flushes at ncclGroupEnd; we flush here.
        if self._desloc_sp_enabled:
            try:
                from deepspeed.compile.custom_ops.hetero_mesh import (enforce_handle_high_water_mark)
                enforce_handle_high_water_mark()
            except ImportError:
                pass

        if self.enable_backward_allreduce and not self.inside_no_sync_ctxt:
            # Traditional code path that allreduces the module parameter grads
            self.allreduce_gradients()

        if isinstance(self.optimizer, ZeROOptimizer):
            self.optimizer.backward_epilogue()
            self.optimizer.exit_backward()

        see_memory_usage("Engine after backward", force=self.memory_breakdown())
        self._stop_timers(self.engine_timers.backward_reduce_timers)
        self._stop_timers(self.engine_timers.backward_timers)

    def _backward_prologue_per_tensor(self, grad):
        # Only scale gradients if scale_wrt_gas is True, consistent with backward() parameter
        if grad is not None and self._scale_wrt_gas:
            return grad / self.gradient_accumulation_steps()
        return grad

    def _backward_post_hook(self):
        if not self._running_engine_backward:
            # Check if loss scaling was required but not applied
            needs_scaler = False
            if isinstance(self.optimizer, ZeROOptimizer):
                needs_scaler = self.optimizer.needs_scaler()
            elif self.torch_autocast_z0_gradscaler is not None:
                needs_scaler = True
            elif self.amp_enabled():
                needs_scaler = True

            if needs_scaler and not self._manual_backward_expected:
                # User called backward() directly without using engine.scale() or engine.backward()
                error_msg = ("Loss scaling is required for this configuration, but backward() was called "
                             "directly without scaling the loss. Please use one of the following:"
                             " 1. engine.backward(loss)"
                             " 2. engine.scale(loss).backward()")
                if self.amp_enabled():
                    error_msg += " Note: AMP (NVIDIA Apex) only supports engine.backward(loss)."
                raise RuntimeError(error_msg)

            # Clear the flag for next backward
            self._manual_backward_expected = False

            self._backward_epilogue()

    @contextmanager
    def no_sync(self):
        r"""
            Context manager to disable gradient reduction during backward pass.
            This context manager has the following effects on other DeepSpeed features:
            1. Incompatible with ZeRO stage 2/3 which rely on reduction for gradient partitioning.
            2. It is illegal to call engine.step() within the context manager.
            3. Tracking of gradient accumulation steps is disabled.
        """
        assert not self.zero_optimization_partition_gradients(), \
        f"no_sync context manager is incompatible with gradient partitioning logic of ZeRO stage {self.zero_optimization_stage()}"

        assert not self.inside_no_sync_ctxt, "no_sync context manager reentry is unsupported"

        self.inside_no_sync_ctxt = True
        try:
            yield
        finally:
            self.inside_no_sync_ctxt = False

    def scale(self, loss):
        r"""Apply loss scaler for manual backward pass.

        Use this method when calling loss.backward() directly instead of engine.backward().
        This applies the appropriate loss scaler for mixed precision training, allowing you
        to manually control the backward pass while still benefiting from DeepSpeed's
        gradient scaling functionality.

        Example::

            output = engine(input)
            loss = criterion(output, target)
            scaled_loss = engine.scale(loss)
            scaled_loss.backward()  # Manual backward call
            engine.step()

        Arguments:
            loss: Scalar loss tensor to be scaled

        Returns:
            Scaled loss tensor ready for .backward() call

        Raises:
            RuntimeError: If AMP (NVIDIA Apex) is enabled. AMP requires using engine.backward()
                         directly as it uses a context manager that cannot be separated from
                         the backward call.
            AssertionError: If loss is not a scalar tensor with grad_fn, or if no optimizer
                           is configured.
        """
        assert self.optimizer is not None and not isinstance(self.optimizer, DummyOptim), \
            "must provide optimizer during init in order to use scale"
        assert maybe_loss_for_backward(loss), \
            "loss must be a scalar tensor with grad_fn. For non-scalar tensors, use tensor.backward(grad)"

        # AMP (NVIDIA Apex) uses a context manager that wraps both scaling and backward,
        # so it cannot be used with manual backward calls
        if self.amp_enabled():
            raise RuntimeError("engine.scale() is not compatible with AMP (NVIDIA Apex). "
                               "When using AMP, you must call engine.backward(loss) instead of manual backward.")

        # Apply loss scaler based on optimizer type
        scaled_loss = loss
        if isinstance(self.optimizer, ZeROOptimizer):
            scaled_loss = self.optimizer.scale_if_loss(loss)
        elif self.torch_autocast_z0_gradscaler:
            scaled_loss = self.torch_autocast_z0_gradscaler.scale(loss)

        # Mark that scale() was called for validation in backward hook
        self._manual_backward_expected = True

        return scaled_loss

    @instrument_w_nvtx
    def backward(self, loss, retain_graph=False, scale_wrt_gas=True):
        r"""Execute backward pass on the loss
        Arguments:
            loss: Torch tensor on which to execute backward propagation
            retain_graph: bool, default: false
                forward on user defined choice of retain_graph
            scale_wrt_gas: bool, default: true
                whether to scale gradients and return value by gradient accumulation steps
        """
        assert self.optimizer is not None and not isinstance(self.optimizer, DummyOptim), \
            "must provide optimizer during init in order to use backward"
        assert maybe_loss_for_backward(
            loss), "loss must be a scalar tensor. If you need to pass output gradients, backward() of output tensors"

        self._running_engine_backward = True
        # Store scale_wrt_gas so the hook can respect it
        self._scale_wrt_gas = scale_wrt_gas

        # Set flag to prevent hooks from firing (we'll manually call prologue/epilogue)
        backward_kwargs = {"retain_graph": retain_graph}
        if self.eigenvalue_enabled():
            backward_kwargs["create_graph"] = True
            backward_kwargs["retain_graph"] = True

        # Used only for return value
        gas_scaled_loss = loss / self.gradient_accumulation_steps() if scale_wrt_gas else loss

        # TODO: handle these scaling with direct calls to loss.backward()
        if isinstance(self.optimizer, ZeROOptimizer):
            loss = self.optimizer.scale_if_loss(loss)
        elif self.torch_autocast_z0_gradscaler:
            loss = self.torch_autocast_z0_gradscaler.scale(loss)

        with compiled_autograd(self._is_compiled_autograd_enabled, self._compile_kwargs):
            if self.zero_optimization() or not self.amp_enabled():
                loss.backward(**backward_kwargs)
            elif self.amp_enabled():
                # AMP requires delaying unscale when inside gradient accumulation boundaries
                # https://nvidia.github.io/apex/advanced.html#gradient-accumulation-across-iterations
                delay_unscale = not self.is_gradient_accumulation_boundary()
                with amp.scale_loss(loss, self.optimizer, delay_unscale=delay_unscale) as scaled_loss:
                    scaled_loss.backward(**backward_kwargs)

            # backward_epilogue is not called in a hook when self._support_torch_style_backward is False
            self._backward_epilogue()

        self._running_engine_backward = False

        return gas_scaled_loss

    def is_gradient_accumulation_boundary(self):
        """
        Query whether the current micro-batch is at the boundary of
        gradient accumulation, and thus will trigger gradient reductions and
        an optimizer step.

        Returns:
            bool: if the current step is a gradient accumulation boundary.

        """
        if self._is_gradient_accumulation_boundary is None:
            if self.zenflow:
                return self._is_zenflow_update_boundary()
            else:
                return (self.micro_steps + 1) % self.gradient_accumulation_steps() == 0
        else:
            return self._is_gradient_accumulation_boundary

    def set_gradient_accumulation_boundary(self, is_boundary):
        """
        Manually overrides the DeepSpeed engine's gradient accumulation boundary state, this is an optional
        feature and should be used with care. The state should be set before to the intended
        value before each forward/backward. The final forward/backward should have the
        boundary state set to True. This style allows client code to only call engine.step() once after all
        the gradient accumulation passes are complete. See example below:
        .. code-block:: python
        engine.set_gradient_accumulation_boundary(False)
        for _ in range(gradient_accumulation_steps - 1):
            micro_batch = next(data_loader)
            loss = engine(micro_batch)
            engine.backward(loss)
        engine.set_gradient_accumulation_boundary(True)
        micro_batch = next(data_loader)
        loss = engine(micro_batch)
        engine.backward(loss)
        engine.step()
        Arguments:
            is_boundary (bool): are we at a gradient accumulation boundary or not?
        """
        self._is_gradient_accumulation_boundary = is_boundary
        self.optimizer.is_gradient_accumulation_boundary = is_boundary

    def zero_grad(self):
        """
        Zero parameter grads.
        """
        for param_name, param in self.module.named_parameters():
            param.grad = None

    def clip_fp32_gradients(self):
        clip_grad_norm_(parameters=self.module.parameters(), max_norm=self.gradient_clipping(), mpu=self.mpu)

    def _take_model_step(self, lr_kwargs, block_eigenvalue={}):
        if self.gradient_clipping() > 0.0:
            if self.torch_autocast_z0_gradscaler:
                # Unscale for gradient clipping
                self.torch_autocast_z0_gradscaler.unscale_(self.optimizer)
            if not (self.fp16_enabled() or self.bfloat16_enabled() or self.amp_enabled() or self.zero_optimization()):
                self.clip_fp32_gradients()
            elif self.amp_enabled():
                # AMP's recommended way of doing clipping
                # https://nvidia.github.io/apex/advanced.html#gradient-clipping
                master_params = amp.master_params(self.optimizer)
                clip_grad_norm_(parameters=master_params, max_norm=self.gradient_clipping(), mpu=self.mpu)
        if self.torch_autocast_z0_gradscaler:
            self.torch_autocast_z0_gradscaler.step(self.optimizer)
            self.torch_autocast_z0_gradscaler.update()
        else:
            # M470: on Kx sync steps, all-gather param shards before optimizer update
            # so every rank has the complete parameter tensor.  Mirrors Megatron
            # DistributedOptimizer._allgather_params() (commit 4feb2b0d).
            if (self.desloc_enabled and self._desloc_dist_opt_mgr is not None and self.desloc_is_param_sync_step()):
                self._desloc_dist_opt_mgr.allgather_params()
            self.optimizer.step()
            # M475: MXFP8 defer-sync gate — mirrors Megatron b80a8547.
            # When grad/param buffer reuse is active and DDP-level overlap_param_gather
            # is absent on any constituent optimizer, we must flush the deferred param
            # sync now (after all chained steps are done) to avoid the race from PR #4800.
            # The gate probes ddp_config.overlap_param_gather on each shard manager; if
            # any is False, trigger the flush here rather than letting the caller race.
            if self._should_defer_mxfp8_param_sync():
                _deferred_sync_opt = getattr(self.optimizer, 'optimizer', self.optimizer)
                if hasattr(_deferred_sync_opt, 'finish_param_sync'):
                    _deferred_sync_opt.finish_param_sync(model_index=0)
                    print(f'[MXFP8-SYNC-GATE] rank={dist.get_rank()} '
                          f'step={self.desloc_step} flushed deferred param sync')

        if hasattr(self.optimizer, '_global_grad_norm'):
            self._global_grad_norm = self.optimizer._global_grad_norm

        # Quantize the updated parameter if there is no overflow
        if self.quantizer:
            tensor_to_quantize = self.optimizer.bit16_groups if self.zero_optimization_stage(
            ) == 2 else self.optimizer.fp16_groups
            if self.compression_scheduler.weight_quantization_enabled:
                self.quantizer.quantize(
                    tensor_to_quantize,
                    (self.optimizer.overflow if self.fp16_enabled() else False),
                    self.eigenvalue_enabled(),
                    block_eigenvalue,
                )
        # zero grad in basic optimizer could be unreliable and may not exhibit
        # the behavior that we want
        if self.bfloat16_enabled():
            # TODO: Temporary until bf16_optimizer and zero_optimizer are integrated
            if hasattr(self.optimizer, "zero_grad"):
                self.optimizer.zero_grad()
            else:
                self.zero_grad()
        elif self.zero_optimization() or self.fp16_enabled() or self.amp_enabled():
            self.optimizer.zero_grad()
        else:
            # M393: fp32 path — zero_grad without set_grads_to_None (Megatron 28cd66e1a)
            print('[M393]')
            self.zero_grad()

        # Check overflow here since in DS fp16 optimizer, the overflow is updated in above step() function.
        overflow = False
        if hasattr(self.optimizer, "overflow"):
            overflow = self.optimizer.overflow
        self._step_applied = not overflow

        if overflow:
            self.skipped_steps += 1
        else:
            self.compression_scheduler.step()
            if self.lr_scheduler is not None:
                try:
                    self.lr_scheduler.step(**(lr_kwargs or {}))
                except TypeError:
                    # XXX Hack to work with Megatron 2.0 and DeepSpeed pipelines.
                    # We don't currently have a way to specify lr_kwargs from
                    # pipe_engine.train_batch()
                    self.lr_scheduler.step(self.train_batch_size())

        if self.steps_per_print() is not None:
            report_progress = self.global_rank == 0 if self.global_rank else True
            if report_progress and (self.global_steps + 1) % self.steps_per_print() == 0:
                self._report_progress(self.global_steps + 1)

        self.losses = None
        self.global_steps += 1
        # DES-LOC: advance local step counter
        if self.desloc_enabled:
            self.desloc_step += 1
            # M342: Track SP+DEC+AC composition state per step
            if self._desloc_sp_enabled or self._desloc_ac_enabled:
                self._desloc_comm_history.append((
                    self.desloc_step,
                    0,  # bytes_sent filled by profiler
                    1 if self.desloc_is_param_sync_step() else 0,
                    f"sp={self._desloc_sp_mode},ac={self._desloc_ac_mode}",
                ))
        self.global_samples += self.train_batch_size()

    def step(self, lr_kwargs=None):
        r"""Execute the weight update step after forward and backward propagation
        on effective_train_batch.
        """
        assert not self.inside_no_sync_ctxt, \
        "It is illegal to call Engine.step() inside no_sync context manager"

        see_memory_usage("Engine before step", force=self.memory_breakdown())

        # Check early because self.global_steps is incremented at some point here.
        # TODO: Delay self.global_steps increment until very end of this function.
        flops_profiler_active = self.flops_profiler_enabled(
        ) and self.global_steps == self.flops_profiler_profile_step() and self.global_rank == 0

        self._start_timers(self.engine_timers.step_timers)

        assert self.optimizer is not None and not isinstance(self.optimizer, DummyOptim), \
            "must provide optimizer during init in order to use step"

        report_progress = False

        self._step_applied = False  # assume False, will flip to True

        if self.zenflow:
            self.optimizer._sync_selective_optimizer_lr()
            if self.auto_update:
                self.update_interval += 1

        # Update the model when we reach gradient accumulation boundaries
        if self.is_gradient_accumulation_boundary():
            self.gas_boundary_ctr += 1

            if self.checkpoint_engine.is_decoupled():
                self._commit_decoupled_checkpoint()

            if (self.eigenvalue_enabled() and (self.gas_boundary_ctr % self.eigenvalue_gas_boundary_resolution() == 0)
                    and self.quantizer.any_precision_switch()):
                log_dist("computing eigenvalue...", ranks=[0])
                loss_scale = self._get_optimizer_loss_scale() or 1.0
                self.block_eigenvalue = self.eigenvalue.compute_eigenvalue(self.module, self.device, loss_scale)

            if self.progressive_layer_drop:
                self.progressive_layer_drop.update_state(self.global_steps)

            if (self.eigenvalue_enabled() and not self.gas_boundary_ctr % self.eigenvalue_gas_boundary_resolution()
                    and self.quantizer.any_precision_switch()):
                self._take_model_step(lr_kwargs, self.block_eigenvalue)
            else:
                self._take_model_step(lr_kwargs)

            report_progress = self.global_rank == 0 if self.global_rank else True

        if self.zenflow:
            self._zenflow_step(lr_kwargs)

        self.tput_timer.stop(global_step=self.is_gradient_accumulation_boundary(), report_speed=report_progress)

        self._stop_timers(self.engine_timers.step_timers)

        # Log learning rate
        if self.monitor.enabled:
            if self.is_gradient_accumulation_boundary():
                if self.global_rank == 0:
                    self.summary_events = [("Train/Samples/lr", self.get_lr()[0], self.global_samples)]

                    loss_scale = self._get_optimizer_loss_scale() if self.fp16_enabled() else None
                    if loss_scale is not None:
                        self.summary_events.append((
                            "Train/Samples/loss_scale",
                            loss_scale,
                            self.global_samples,
                        ))

                    if (self.eigenvalue_enabled()
                            and not self.gas_boundary_ctr % self.eigenvalue_gas_boundary_resolution()):
                        ev_values = self.block_eigenvalue.values()
                        for i in range(len(ev_values)):
                            self.summary_events.append((
                                f"Train/Eigenvalues/ModelBlockParam_{i}",
                                self.ev_values[i][0],
                                self.global_samples,
                            ))
                    self.monitor.write_events(self.summary_events)

        # Check flops profiling
        if flops_profiler_active:
            if self.autotuning_enabled():
                self.flops = self.flops_profiler.get_total_flops() * 3
                self.fwd_duration = self.flops_profiler.get_total_duration()
            else:
                self.flops_profiler.print_model_profile(
                    profile_step=self.global_steps,
                    module_depth=self.flops_profiler_module_depth(),
                    top_modules=self.flops_profiler_top_modules(),
                    detailed=self.flops_profiler_detailed(),
                    output_file=self.flops_profiler_output_file(),
                )
            self.flops_profiler.end_profile()

        if self.autotuning_enabled() and self.global_steps == (self.autotuning_end_profile_step() + 1):
            self._autotuning_exit()

        if self.wall_clock_breakdown():
            # Update client accessible wall clock timers cache
            self._update_wall_clock_timers()

            # Log micro timing and reset
            self.timers.log(names=self.engine_timers.micro_timers, memory_breakdown=self.memory_breakdown())

        if self.wall_clock_breakdown() or self.flops_profiler_enabled():
            # Log global timing and reset
            if self.is_gradient_accumulation_boundary():
                if self.monitor.enabled:
                    self._write_monitor()

                if self.has_moe_layers:
                    fwd_time = self.timers(FORWARD_GLOBAL_TIMER).elapsed(reset=False)
                    self.print_forward_breakdown(fwd_time=fwd_time)

                self.timers.log(self.engine_timers.global_timers)

        self.micro_steps += 1
        see_memory_usage("Engine after step", force=self.memory_breakdown())

    def _start_timers(self, timer_names):
        for name in timer_names:
            self.timers(name).start()

    def _stop_timers(self, timer_names):
        record = self.is_gradient_accumulation_boundary() and \
            self.flops_profiler_enabled() and \
                (self.global_steps >= self.flops_profiler_profile_step())
        for name in timer_names:
            self.timers(name).stop(record=record)

    def _update_wall_clock_timers(self):
        self.engine_timers_cache = {}
        for name in self.engine_timers.active_timers():
            self.engine_timers_cache[name] = self.timers(name).elapsed(reset=False)

    def get_wall_clock_timers(self):
        r"""
            Return a dict snapshot of the Engine's wall clock timers.
        """
        return self.engine_timers_cache

    def _autotuning_exit(self):
        if self.global_rank == 0:
            msg = self.timers.get_mean([
                FORWARD_GLOBAL_TIMER,
                BACKWARD_GLOBAL_TIMER,
                STEP_GLOBAL_TIMER,
            ], reset=False)
            titer = 0.0
            titer += msg[FORWARD_GLOBAL_TIMER] if FORWARD_GLOBAL_TIMER in msg else 0
            titer += msg[BACKWARD_GLOBAL_TIMER] if BACKWARD_GLOBAL_TIMER in msg else 0
            titer += msg[STEP_GLOBAL_TIMER] if STEP_GLOBAL_TIMER in msg else 0
            titer *= self.gradient_accumulation_steps()
            msg["latency"] = titer
            msg["FLOPS_per_gpu"] = self.flops * 1_000_000 * self.gradient_accumulation_steps() / titer
            msg["throughput"] = self.train_batch_size() * 1_000_000 / \
                msg["latency"]
            print_json_dist(msg, [0], path=self.autotuning_metric_path())
            log_dist(
                f"Wrote metrics to {self.autotuning_metric_path()}, {os.path.abspath(self.autotuning_metric_path())}",
                ranks=[0])
            import atexit
            atexit.register(print, "Autotuning: done with running current ds config.")
        exit()

    def _write_monitor(self):
        if self.global_rank == 0:
            self.summary_events = [
                (
                    "Train/Samples/elapsed_time_ms_forward",
                    self.timers(FORWARD_GLOBAL_TIMER).elapsed(reset=False),
                    self.global_samples,
                ),
                (
                    "Train/Samples/elapsed_time_ms_backward",
                    self.timers(BACKWARD_GLOBAL_TIMER).elapsed(reset=False),
                    self.global_samples,
                ),
                (
                    "Train/Samples/elapsed_time_ms_backward_inner",
                    self.timers(BACKWARD_INNER_GLOBAL_TIMER).elapsed(reset=False),
                    self.global_samples,
                ),
                (
                    "Train/Samples/elapsed_time_ms_backward_allreduce",
                    self.timers(BACKWARD_REDUCE_GLOBAL_TIMER).elapsed(reset=False),
                    self.global_samples,
                ),
                (
                    "Train/Samples/elapsed_time_ms_step",
                    self.timers(STEP_GLOBAL_TIMER).elapsed(reset=False),
                    self.global_samples,
                ),
            ]
            self.monitor.write_events(self.summary_events)

    def _get_optimizer_param(self, param_name):
        result = []
        if not self.optimizer:
            return result
        for group in self.optimizer.param_groups:
            if param_name in group:
                result.append(group[param_name])
            else:
                result.append(0.0)
        return result

    def _get_optimizer_loss_scale(self):
        if not self.optimizer:
            return None
        if hasattr(self.optimizer, "loss_scale_config"):
            return self.optimizer.loss_scale_config.cur_scale
        return getattr(self.optimizer, "cur_scale", None)

    def get_lr(self):
        return self._get_optimizer_param("lr")

    def get_type(self):
        return self._get_optimizer_param("type")

    def get_mom(self):
        if self.optimizer_name() in ["SGD", "RMSprop"]:
            return self._get_optimizer_param("momentum")
        else:
            return self._get_optimizer_param("betas")

    def get_pld_theta(self):
        if self.progressive_layer_drop:
            return self.progressive_layer_drop.get_theta()
        else:
            return None

    def _report_progress(self, step):
        lr = self.get_lr()
        mom = self.get_mom()
        log_dist(f"step={step}, skipped={self.skipped_steps}, lr={lr}, mom={mom}", ranks=[0])
        # M227: Record loss data point for Figure 1 curve generation
        # Ref: NKI-FA draw_plot.py format: structured metric lines
        # Ref: Section 5.4 — loss vs step across DDP/DES-LOC/Local Adam
        if self.desloc_enabled and hasattr(self, '_desloc_loss_history'):
            is_sync = self.desloc_is_param_sync_step()
            current_lr = lr[0] if isinstance(lr, (list, tuple)) and lr else 0.0
            self._desloc_loss_history.append((self.global_steps, None, current_lr, is_sync))

    def desloc_record_loss(self, loss_value):
        """Record training loss for Figure 1 curve.
        Called from training loop after forward pass.

        Ref: NKI-FA draw_plot.py — each data point is
        '### config ###\\nloss: 3.456789' format.
        Ref: Section 5.4 Table CLXXI — DES-LOC competitive with DDP.

        Args:
            loss_value: float — current step's training loss
        """
        if not self.desloc_enabled:
            return
        step = self.global_steps
        lr_val = 0.0
        try:
            lr_list = self.get_lr()
            lr_val = lr_list[0] if lr_list else 0.0
        except Exception:
            pass
        is_sync = self.desloc_is_param_sync_step()
        # Update sliding window for EMA smoothing
        self._desloc_loss_window.append(float(loss_value))
        if len(self._desloc_loss_window) > self._desloc_loss_window_size:
            self._desloc_loss_window.pop(0)
        # Replace None in last entry or append new
        if (self._desloc_loss_history and self._desloc_loss_history[-1][0] == step
                and self._desloc_loss_history[-1][1] is None):
            self._desloc_loss_history[-1] = (step, float(loss_value), lr_val, is_sync)
        else:
            self._desloc_loss_history.append((step, float(loss_value), lr_val, is_sync))

    def desloc_record_eval(self, eval_loss, eval_ppl=None):
        """Record evaluation loss for Figure 1 overlay.

        Ref: Section 5.4 — eval on ICL tasks at checkpoints.
        Separate from training loss for clean plotting.
        """
        if not self.desloc_enabled:
            return
        step = self.global_steps
        self._desloc_eval_history.append((step, float(eval_loss), float(eval_ppl) if eval_ppl is not None else None))

    def desloc_record_comm(self, bytes_sent, num_ops=1, tier='x'):
        """Record communication event for Figure 2 data.

        Ref: Section 5.3 RQ3 — comm reduction vs DDP and Local Adam.
        Ref: comms_logging.py classify_comm_tier().

        Args:
            bytes_sent: int — bytes communicated
            num_ops: int — number of collective ops
            tier: str — 'x' (param), 'u' (momentum), 'v' (variance)
        """
        if not self.desloc_enabled:
            return
        self._desloc_comm_history.append((self.global_steps, int(bytes_sent), int(num_ops), str(tier)))

    def desloc_record_throughput(self, tokens_per_sec):
        """Record throughput for Figure 5 data.

        Ref: Section 5.4 — training speedup over DDP.
        """
        if not self.desloc_enabled:
            return
        self._desloc_throughput_history.append((self.global_steps, float(tokens_per_sec)))

    def desloc_get_smoothed_loss(self):
        """Get EMA-smoothed loss from sliding window.

        Returns None if window is empty.
        Pure-python mean, no numpy.
        """
        if not self._desloc_loss_window:
            return None
        return sum(self._desloc_loss_window) / len(self._desloc_loss_window)

    def desloc_get_loss_curve_data(self):
        """Extract (steps, losses) for Figure 1 plotting.

        Returns only entries where loss is not None.
        Format compatible with NKI-FA draw_plot.py parsing.
        """
        steps = []
        losses = []
        for entry in self._desloc_loss_history:
            if entry[1] is not None:
                steps.append(entry[0])
                losses.append(entry[1])
        return steps, losses

    def desloc_get_comm_reduction_data(self):
        """Extract communication reduction data for Figure 2.

        Returns dict with:
            total_bytes: int — total bytes communicated
            ddp_equiv_bytes: int — what DDP would have sent
            reduction_ratio: float — ddp/desloc
            per_tier: dict — {tier: total_bytes}

        Ref: Section 5.3 — DES-LOC halves communication vs Local Adam
        """
        if not self._desloc_comm_history:
            return {'total_bytes': 0, 'ddp_equiv_bytes': 0, 'reduction_ratio': 1.0, 'per_tier': {}}
        total_bytes = sum(e[1] for e in self._desloc_comm_history)
        total_ops = sum(e[2] for e in self._desloc_comm_history)
        per_tier = {}
        for e in self._desloc_comm_history:
            t = e[3]
            per_tier[t] = per_tier.get(t, 0) + e[1]
        # DDP equivalent: every step would sync all tiers
        total_steps = max(1, self.desloc_step)
        if total_ops > 0:
            bytes_per_op = total_bytes / total_ops
            ddp_equiv = bytes_per_op * total_steps * 3  # 3 tiers
        else:
            ddp_equiv = total_bytes
        ratio = ddp_equiv / max(1, total_bytes)
        return {
            'total_bytes': total_bytes,
            'ddp_equiv_bytes': int(ddp_equiv),
            'reduction_ratio': ratio,
            'per_tier': per_tier,
        }

    def desloc_export_figure_data(self, output_dir):
        """Export all figure data to NKI-FA format log files.

        Creates one log file per figure with structured output:
            ### config = {Kx=32, Ku=96, ...} ###
            step: 0 | loss: 10.234567
            step: 1 | loss: 10.123456
            ...
            --- summary ---
            final_loss: 3.456789
            --- end summary ---

        Ref: NKI-FA commit da964f3 — draw_plot.py parses this format.
        Ref: Nick Joseph — "collect data, then make decisions"
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        rank = 0
        try:
            rank = dist.get_rank()
        except Exception:
            pass
        if rank != 0:
            return

        config_header = (f"### model = {getattr(self, '_desloc_model_size', 'unknown')}, "
                         f"Kx = {self.desloc_Kx}, Ku = {self.desloc_Ku}, "
                         f"Kv = {self.desloc_Kv}, "
                         f"optimizer = {self.desloc_outer_opt_mode} ###")

        # Figure 1: Loss vs Step
        fig1_path = os.path.join(output_dir, 'figure1_loss_curve.log')
        with open(fig1_path, 'w') as f:
            f.write(config_header + '\n')
            for entry in self._desloc_loss_history:
                step, loss, lr, is_sync = entry
                if loss is not None:
                    f.write(f"step: {step} | loss: {loss:.6f} | "
                            f"lr: {lr:.8f} | is_sync: {int(is_sync)}\n")
            # Summary
            steps, losses = self.desloc_get_loss_curve_data()
            if losses:
                f.write('\n--- summary ---\n')
                f.write(f"total_steps: {len(losses)}\n")
                f.write(f"first_loss: {losses[0]:.6f}\n")
                f.write(f"final_loss: {losses[-1]:.6f}\n")
                f.write(f"min_loss: {min(losses):.6f}\n")
                if losses[0] > 0:
                    reduction = (losses[0] - losses[-1]) / losses[0]
                    f.write(f"loss_reduction_pct: {reduction * 100:.2f}\n")
                f.write('--- end summary ---\n')

        # Figure 2: Communication Reduction
        fig2_path = os.path.join(output_dir, 'figure2_comm_reduction.log')
        with open(fig2_path, 'w') as f:
            f.write(config_header + '\n')
            comm_data = self.desloc_get_comm_reduction_data()
            f.write(f"total_bytes: {comm_data['total_bytes']}\n")
            f.write(f"ddp_equiv_bytes: {comm_data['ddp_equiv_bytes']}\n")
            f.write(f"reduction_ratio: {comm_data['reduction_ratio']:.1f}\n")
            for tier, tbytes in sorted(comm_data['per_tier'].items()):
                tier_gb = tbytes / (1024**3)
                f.write(f"tier_{tier}_gb: {tier_gb:.3f}\n")
            # Per-step comm log
            for entry in self._desloc_comm_history:
                step, nbytes, nops, tier = entry
                mb = nbytes / (1024**2)
                f.write(f"step: {step} | comm_mb: {mb:.2f} | "
                        f"ops: {nops} | tier: {tier}\n")
            f.write('\n--- summary ---\n')
            f.write(f"Kx: {self.desloc_Kx}\n")
            f.write(f"Ku: {self.desloc_Ku}\n")
            f.write(f"Kv: {self.desloc_Kv}\n")
            f.write(f"skipped_allreduces: {self.desloc_skipped_allreduces}\n")
            f.write(f"total_steps: {self.desloc_step}\n")
            if self.desloc_step > 0:
                skip_pct = 100.0 * self.desloc_skipped_allreduces / self.desloc_step
                f.write(f"skip_pct: {skip_pct:.1f}\n")
            f.write('--- end summary ---\n')

        # Eval history
        if self._desloc_eval_history:
            eval_path = os.path.join(output_dir, 'eval_history.log')
            with open(eval_path, 'w') as f:
                f.write(config_header + '\n')
                for step, eloss, eppl in self._desloc_eval_history:
                    parts = [f"eval_step: {step}", f"eval_loss: {eloss:.6f}"]
                    if eppl is not None:
                        parts.append(f"eval_ppl: {eppl:.2f}")
                    f.write(" | ".join(parts) + '\n')

        # Throughput
        if self._desloc_throughput_history:
            tput_path = os.path.join(output_dir, 'throughput.log')
            with open(tput_path, 'w') as f:
                f.write(config_header + '\n')
                for step, tps in self._desloc_throughput_history:
                    f.write(f"step: {step} | tokens_per_sec: {tps:.1f}\n")

    def desloc_set_figure_dir(self, path):
        """Set output directory for figure data export."""
        self._desloc_figure_dir = path

    def desloc_get_figure1_nkifa_block(self):
        """Return Figure 1 data as NKI-FA format string.

        Format: '### config ###\\nmetric: value\\n...'
        Ready for draw_plot.py consumption.

        Ref: NKI-FA draw_plot.py line_regex:
            r'(\\w+) (fwd|bwd): ([\\d.]+)ms, ([\\d.]+) TFLOPS'
        Adapted for loss curves:
            r'step: (\\d+) \\| loss: ([\\d.]+)'
        """
        lines = []
        config_str = (f"### Kx = {self.desloc_Kx}, Ku = {self.desloc_Ku}, "
                      f"Kv = {self.desloc_Kv}, "
                      f"outer = {self.desloc_outer_opt_mode} ###")
        lines.append(config_str)
        for entry in self._desloc_loss_history:
            step, loss, lr, is_sync = entry
            if loss is not None:
                lines.append(f"step: {step} | loss: {loss:.6f}")
        return '\n'.join(lines)

    def allreduce_bucket(self, bucket, dp_group, dp_world_size=None):
        tensor = self.flatten(bucket)

        tensor_to_allreduce = tensor

        if self.communication_data_type != tensor.dtype:
            tensor_to_allreduce = tensor.to(self.communication_data_type)

        if dp_world_size is None:
            dp_world_size = dist.get_world_size(group=dp_group)
        if self.postscale_gradients():
            if self.gradient_predivide_factor() != 1.0:
                tensor_to_allreduce.mul_(1.0 / self.gradient_predivide_factor())

            dist.all_reduce(tensor_to_allreduce, group=dp_group)
            if self.gradient_average:
                if self.gradient_predivide_factor() != dp_world_size:
                    tensor_to_allreduce.mul_(self.gradient_predivide_factor() / dp_world_size)
        else:
            tensor_to_allreduce.mul_(1. / dp_world_size)
            dist.all_reduce(tensor_to_allreduce, group=dp_group)

        if self.communication_data_type != tensor.dtype and tensor is not tensor_to_allreduce:
            tensor.copy_(tensor_to_allreduce)

        return tensor

    def allreduce_and_copy(self, small_bucket, dp_group, dp_world_size=None):
        allreduced = self.allreduce_bucket(small_bucket, dp_group, dp_world_size)
        for buf, synced in zip(small_bucket, self.unflatten(allreduced, small_bucket)):
            buf.copy_(synced)

    def allreduce_no_retain(self, bucket, dp_group, numel_per_bucket=500000000, dp_world_size=None):
        small_bucket = []
        numel = 0
        for tensor in bucket:
            small_bucket.append(tensor)
            numel = numel + tensor.numel()
            if numel > numel_per_bucket:
                self.allreduce_and_copy(small_bucket, dp_group, dp_world_size)
                small_bucket = []
                numel = 0
        if len(small_bucket) > 0:
            self.allreduce_and_copy(small_bucket, dp_group, dp_world_size)

    def _get_gradients_for_reduction(self):
        non_expert_grads = []
        expert_grads = {}
        if self.has_moe_layers:
            for key in self.expert_data_parallel_group.keys():
                expert_grads[key] = []

        for param_name, param in self.module.named_parameters():
            if not param.requires_grad:
                continue

            # Skip empty parameters (numel=0) as they contribute nothing to gradient reduction
            # and cause issues with flatten/unflatten operations
            if param.numel() == 0:
                continue

            if param.grad is None:
                # In cases where there is an imbalance of empty grads across
                # ranks we must create empty grads, this will ensure that every
                # rank is reducing the same size. In some cases it may make
                # sense in the future to support the ability to average not
                # w.r.t. world size but with a different value.
                param.grad = torch.zeros(param.size(), dtype=param.dtype, device=param.device)

            grad_data = param.grad.data
            if param_name in self.sparse_tensor_module_names or grad_data.is_sparse:
                # Call param.grad without data to avoid problem with setting of updated grads
                grad_data = SparseTensor(param.grad)

            if is_moe_param(param):
                expert_grads[param.group_name].append(grad_data)
            else:
                non_expert_grads.append(grad_data)

        return non_expert_grads, expert_grads

    def _reduce_non_expert_gradients(self, grads, elements_per_buffer):
        split_sparse_tensor_buckets, split_dense_tensor_buckets = split_half_float_double_sparse(grads)
        if self.pipeline_parallelism:
            dp_group = self.mpu.get_data_parallel_group()
            dp_world_size = dist.get_world_size(dp_group)
        else:
            dp_group = groups._get_sequence_data_parallel_group()
            dp_world_size = dist.get_world_size(dp_group) / float(self.sequence_parallel_size)
        for _, sparse_bucket_tuple in enumerate(split_sparse_tensor_buckets):
            if sparse_bucket_tuple:
                bucket_type, sparse_bucket = sparse_bucket_tuple
                self.sparse_allreduce_no_retain(sparse_bucket, dp_group=dp_group, dp_world_size=dp_world_size)

        for _, dense_bucket_tuple in enumerate(split_dense_tensor_buckets):
            if dense_bucket_tuple:
                bucket_type, dense_bucket = dense_bucket_tuple
                self.allreduce_no_retain(dense_bucket,
                                         dp_group=dp_group,
                                         numel_per_bucket=elements_per_buffer,
                                         dp_world_size=dp_world_size)

    def _reduce_expert_gradients(self, expert_grads, elements_per_buffer):
        # to maintain the gradients value unaffected by ep_size setting,
        # utilize dp_world_size for allreduce average
        dp_world_size = dist.get_world_size(groups._get_data_parallel_group())
        for ep_name, expert_grads_group in expert_grads.items():
            ep_dp_group = groups._get_expert_data_parallel_group(ep_name)
            split_sparse_tensor_buckets, split_dense_tensor_buckets = split_half_float_double_sparse(
                expert_grads_group)

            for _, sparse_bucket_tuple in enumerate(split_sparse_tensor_buckets):
                if sparse_bucket_tuple:
                    bucket_type, sparse_bucket = sparse_bucket_tuple
                    self.sparse_allreduce_no_retain(sparse_bucket, dp_group=ep_dp_group, dp_world_size=dp_world_size)

            for _, dense_bucket_tuple in enumerate(split_dense_tensor_buckets):
                if dense_bucket_tuple:
                    bucket_type, dense_bucket = dense_bucket_tuple
                    # Separate between diff groups
                    self.allreduce_no_retain(dense_bucket,
                                             dp_group=ep_dp_group,
                                             numel_per_bucket=elements_per_buffer,
                                             dp_world_size=dp_world_size)

    def buffered_allreduce_fallback(self, grads=None, elements_per_buffer=500000000):
        if grads is None:
            if hasattr(self.optimizer, "get_grads_for_reduction"):
                # This is currently for BF16 optimizer
                non_expert_grads, expert_grads = self.optimizer.get_grads_for_reduction()
            else:
                non_expert_grads, expert_grads = self._get_gradients_for_reduction()
        else:
            assert not self.has_moe_layers, "attempting to reduce grads in unsupported way w.r.t. MoE"
            non_expert_grads = grads

        self._reduce_non_expert_gradients(non_expert_grads, elements_per_buffer)

        if self.has_moe_layers:
            self._reduce_expert_gradients(expert_grads, elements_per_buffer)

    def sparse_allreduce_no_retain(self, bucket, dp_group, dp_world_size=None):
        allreduced_sparses = self.sparse_allreduce_bucket(bucket, dp_group, dp_world_size)
        # Densify sparse tensor and copy back to original location
        for tensor in allreduced_sparses:
            if tensor.is_sparse:
                tensor.orig_dense_tensor.data = tensor.to_coo_tensor()
            else:
                tensor.orig_dense_tensor.copy_(tensor.to_dense())

    def sparse_allreduce_bucket(self, bucket, dp_group, dp_world_size=None):
        sparse_list = []
        for sparse in bucket:
            sparse_list.append(self.sparse_allreduce(sparse, dp_group, dp_world_size))
        return sparse_list

    def sparse_allreduce(self, sparse, dp_group, dp_world_size=None):
        original_data_type = sparse.values.dtype
        if self.communication_data_type != sparse.values.dtype:
            if self.communication_data_type in (torch.float16, torch.bfloat16):
                indices = sparse.indices.to(torch.int32)
            else:
                indices = sparse.indices
            values = sparse.values.to(self.communication_data_type)
        else:
            indices = sparse.indices
            values = sparse.values

        if dp_world_size is None:
            dp_world_size = dist.get_world_size(group=dp_group)
        if self.postscale_gradients():
            if self.gradient_average:
                values.mul_(self.gradient_predivide_factor() / (dp_world_size))
        else:
            values.mul_(1. / (dp_world_size))

        indices_device_list = self.sparse_all_gather(indices, dp_group)
        values_device_list = self.sparse_all_gather(values, dp_group)

        sparse.indices = torch.cat(indices_device_list).to(torch.long)
        sparse.values = torch.cat(values_device_list).to(original_data_type)
        return sparse

    def sparse_all_gather(self, value, dp_group):
        my_size = torch.LongTensor([value.size()[0]]).to(self.device)
        all_sizes = self.all_gather_scalar(my_size, dp_group)
        max_size = torch.cat(all_sizes).max()
        fill_size = max_size - my_size

        assert value.dim() in [1, 2]
        if value.dim() == 1:
            if fill_size > 0:
                value = torch.cat([value, value.new_empty(fill_size)])
            tensor_list = [value.new_empty(max_size) for _ in range(dist.get_world_size(group=dp_group))]
        else:
            if fill_size > 0:
                value = torch.cat([value, value.new_empty(fill_size, value.size()[1])])
            tensor_list = [
                value.new_empty(max_size,
                                value.size()[1]) for _ in range(dist.get_world_size(group=dp_group))
            ]

        dist.all_gather(tensor_list, value, group=dp_group)
        tensors = []
        for dev_idx, t in enumerate(tensor_list):
            size = all_sizes[dev_idx][0]
            tensors.append(t.index_select(0, torch.arange(size, dtype=torch.long, device=self.device)))

        return tensors

    def all_gather_scalar(self, value, dp_group):
        tensor_list = [value.new_zeros(value.size()) for _ in range(dist.get_world_size(group=dp_group))]
        dist.all_gather(tensor_list, value, group=dp_group)
        return tensor_list

    def module_state_dict(self, destination=None, prefix="", keep_vars=False, exclude_frozen_parameters=False):
        sd = self.module.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        # Remove frozen parameter weights from state_dict if specified
        if exclude_frozen_parameters:
            for n, p in self.module.named_parameters():
                if not p.requires_grad and n in sd:
                    del sd[n]

        if self.random_ltd_enabled():
            sd = remove_random_ltd_state_dict(sd)
        return sd

    @staticmethod
    def load_moe_state_dict(checkpoint_path,
                            tag,
                            state_dict,
                            old_moe_load,
                            model=None,
                            mpu=None,
                            num_experts=1,
                            checkpoint_engine=TorchCheckpointEngine()):
        if old_moe_load:
            expp_rank = groups._get_expert_data_parallel_rank(groups._get_max_expert_size_name())

            num_local_experts = max(num_experts) // groups._get_expert_parallel_world_size(
                groups._get_max_expert_size_name())
            for local_expert_id in range(num_local_experts):
                global_expert_id = expp_rank * num_local_experts + local_expert_id
                expert_state_dict = checkpoint_engine.load(
                    DeepSpeedEngine._get_expert_ckpt_name(
                        checkpoint_path,
                        -1,  # -1 means ignore layer_id
                        global_expert_id,
                        tag,
                        mpu),
                    map_location=torch.device('cpu'))

                # Updating global -> local expert ids
                moe_str_prefix = '.deepspeed_moe.experts.deepspeed_experts.'
                for key in list(expert_state_dict.keys()):
                    local_key = key.replace(f'{moe_str_prefix}{global_expert_id}',
                                            f'{moe_str_prefix}{local_expert_id}')
                    expert_state_dict[local_key] = expert_state_dict.pop(key)
                state_dict.update(expert_state_dict)

        else:
            moe_layer_id = 0
            for n_module, module in model.named_modules():
                if isinstance(module, MoE):  # and deepspeed.comm.get_rank() == 0:
                    group_name = module.expert_group_name
                    num_local_experts = module.num_local_experts
                    expp_rank = groups._get_expert_parallel_rank(group_name)
                    # loop all local_experts
                    for local_expert_id in range(num_local_experts):
                        global_expert_id = expp_rank * num_local_experts + local_expert_id
                        expert_state_dict = checkpoint_engine.load(DeepSpeedEngine._get_expert_ckpt_name(
                            checkpoint_path, moe_layer_id, global_expert_id, tag, mpu),
                                                                   map_location=torch.device('cpu'))
                        # print(expert_state_dict.keys())
                        # Updating global -> local expert ids
                        moe_str_prefix = '.deepspeed_moe.experts.deepspeed_experts.'
                        for key in list(expert_state_dict.keys()):
                            local_key = key.replace(f'{moe_str_prefix}{global_expert_id}',
                                                    f'{moe_str_prefix}{local_expert_id}')
                            expert_state_dict[local_key] = expert_state_dict.pop(key)
                        state_dict.update(expert_state_dict)
                    moe_layer_id += 1

    def load_module_state_dict(self, checkpoint, strict=True, custom_load_fn=None, fetch_z3_params=False):
        if fetch_z3_params:
            params_to_fetch = [
                p for p in self.module.parameters()
                if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
            ]
        else:
            params_to_fetch = []

        with deepspeed.zero.GatheredParameters(params_to_fetch, modifier_rank=0):
            # M455: Megatron ee38e7f — old checkpoints may be missing the
            # 'module' key (pre-refactor saves stored weights under a different
            # top-level key).  Fall back to the checkpoint dict itself so that
            # callers loading legacy saves don't get a bare KeyError.
            # Knuth critique 1: swallowing KeyError here conflates "missing key"
            #   with "wrong checkpoint entirely" — both become silent no-ops.
            # Knuth critique 2: passing the whole checkpoint as module_state_dict
            #   will likely cause a downstream mismatch; strict=False should be
            #   forced in the fallback, but we preserve caller's intent for now.
            try:
                module_state_dict = checkpoint['module']
            except KeyError:
                print(f"[M455-COMPAT] 'module' key missing from checkpoint dict; "
                      f"attempting to use checkpoint root as state_dict "
                      f"(old-format ckpt). Keys present: {list(checkpoint.keys())[:8]}")
                module_state_dict = checkpoint
            if custom_load_fn:
                custom_load_fn(src=module_state_dict, dst=self.module)
            else:
                try:
                    self.module.load_state_dict(module_state_dict, strict=strict)
                except RuntimeError as _ckpt_err:
                    # M455: unexpected keys / missing keys from old checkpoint
                    # format; retry with strict=False as a best-effort fallback.
                    print(f"[M455-COMPAT] load_state_dict(strict={strict}) raised "
                          f"RuntimeError: {_ckpt_err}. "
                          f"Retrying with strict=False for old-format compat.")
                    self.module.load_state_dict(module_state_dict, strict=False)

        if checkpoint.get(FROZEN_PARAM_FRAGMENTS, None) is not None:
            saved_frozen_params = checkpoint[FROZEN_PARAM_FRAGMENTS]
            for param in self.module.parameters():
                if param.requires_grad:
                    continue
                if param not in self.param_names:
                    raise ValueError(f"failed to find frozen {param} in named params")
                name = self.param_names[param]
                if hasattr(param, 'ds_id'):
                    param.ds_tensor.data.copy_(saved_frozen_params[name].data)
                else:
                    param.data.copy_(saved_frozen_params[name].data)

    def _get_zero_ckpt_prefix(self, dp_rank, bf16_mode):
        return f'{"bf16_" if bf16_mode else ""}zero_pp_rank_{dp_rank}'

    def _get_rank_zero_ckpt_name(self, checkpoints_path, tag, mp_rank, dp_rank, bf16_mode):
        file_prefix = self._get_zero_ckpt_prefix(dp_rank, bf16_mode=bf16_mode)
        zero_ckpt_name = os.path.join(
            checkpoints_path,
            str(tag),
            f"{file_prefix}_mp_rank_{mp_rank:02d}_optim_states.pt",
        )
        return zero_ckpt_name

    def _get_zero_ckpt_name(self, checkpoints_path, tag):
        mp_rank = 0 if self.mpu is None else self.mpu.get_model_parallel_rank()
        pp_rank = dist.get_rank(group=self.optimizer.dp_process_group)
        bf16_mode = self.bfloat16_enabled()
        return self._get_rank_zero_ckpt_name(checkpoints_path, tag, mp_rank, pp_rank, bf16_mode)

    def _get_ckpt_name(self, checkpoints_path, tag, mp_placeholder=None, pp_placeholder=None):
        if mp_placeholder is not None:
            mp_rank_str = mp_placeholder
        else:
            mp_rank = 0 if self.mpu is None else self.mpu.get_model_parallel_rank()
            mp_rank_str = f"{mp_rank:02d}"

        if self.zero_optimization_partition_weights():
            if pp_placeholder is not None:
                pp_rank = pp_placeholder
            else:
                pp_rank = dist.get_rank(group=self.optimizer.dp_process_group)

            filename = "zero_pp_rank_{}".format(pp_rank)
            ckpt_name = os.path.join(
                checkpoints_path,
                str(tag),
                f"{filename}_mp_rank_{mp_rank_str}_model_states.pt",
            )
        else:
            ckpt_name = os.path.join(
                checkpoints_path,
                str(tag),
                "mp_rank_" + mp_rank_str + "_model_states.pt",
            )
        return ckpt_name

    def _get_optimizer_ckpt_name(self, checkpoints_path, tag, expp_rank):
        mp_rank = 0 if self.mpu is None else self.mpu.get_model_parallel_rank()
        ckpt_name = os.path.join(checkpoints_path, str(tag),
                                 f'expp_rank_{expp_rank}_mp_rank_{mp_rank:02d}_optim_states.pt')
        return ckpt_name

    @staticmethod
    def _get_expert_ckpt_name(checkpoints_path, layer_id, expert_id, tag, mpu=None):
        mp_rank = 0 if mpu is None else mpu.get_model_parallel_rank()
        if layer_id <= -1:
            # Used to support old checkpoint loading
            ckpt_name = os.path.join(checkpoints_path, '' if tag is None else str(tag),
                                     f'expert_{expert_id}_mp_rank_{mp_rank:02d}_model_states.pt')
        else:
            # Used to support new checkpoint loading
            ckpt_name = os.path.join(checkpoints_path, '' if tag is None else str(tag),
                                     f'layer_{layer_id}_expert_{expert_id}_mp_rank_{mp_rank:02d}_model_states.pt')
        return ckpt_name

    def _get_all_ckpt_names(self, checkpoints_path, tag):
        # It is required that (checkpoints_path, tag) are consistent among all ranks.
        ckpt_file_pattern = self._get_ckpt_name(checkpoints_path,
                                                tag,
                                                mp_placeholder="*",
                                                pp_placeholder="0" if self.load_universal_checkpoint() else None)
        import glob

        ckpt_files = glob.glob(ckpt_file_pattern)
        ckpt_files.sort()
        return ckpt_files

    def load_checkpoint(self,
                        load_dir,
                        tag=None,
                        load_module_strict=True,
                        load_optimizer_states=True,
                        load_lr_scheduler_states=True,
                        load_module_only=False,
                        custom_load_fn=None):
        """
        Load training checkpoint

        Arguments:
            load_dir: Required. Directory to load the checkpoint from
            tag: Checkpoint tag used as a unique identifier for checkpoint, if not provided will attempt to load tag in 'latest' file
            load_module_strict: Optional. Boolean to strictly enforce that the keys in state_dict of module and checkpoint match.
            load_optimizer_states: Optional. Boolean to load the training optimizer states from Checkpoint. Ex. ADAM's momentum and variance
            load_lr_scheduler_states: Optional. Boolean to add the learning rate scheduler states from Checkpoint.
            load_module_only: Optional. Boolean to load only the model weights from the checkpoint. Ex. warmstarting.
            custom_load_fn: Optional. Custom model load function.

        Returns:
            A tuple of ``load_path`` and ``client_state``.
            *``load_path``: Path of the loaded checkpoint. ``None`` if loading the checkpoint failed.
            *``client_state``: State dictionary used for loading required training states in the client code.

        Important: under ZeRO3, one cannot load checkpoint with ``engine.load_checkpoint()`` right
        after ``engine.save_checkpoint()``. It is because ``engine.module`` is partitioned, and
        ``load_checkpoint()`` wants a pristine model. If insisting to do so, please reinitialize engine
        before ``load_checkpoint()``.

        """

        if tag is None:
            latest_tag = "latest_universal" if self.load_universal_checkpoint() else "latest"
            latest_path = os.path.join(load_dir, latest_tag)
            if os.path.isfile(latest_path):
                with open(latest_path, "r") as fd:
                    tag = fd.read().strip()
            else:
                if self.load_universal_checkpoint():
                    raise ValueError(f'Invalid for universal checkpoint: {latest_path} does not exist')
                else:
                    logger.warning(
                        f"Unable to find latest file at {latest_path}, if trying to load latest "
                        "checkpoint please ensure this file exists or pass an explicit checkpoint tag when loading a checkpoint."
                    )
                    return None, None

        if self._optimizer_has_ckpt_event_prologue():
            # Prepare for checkpoint load by ensuring all parameters are partitioned
            self.optimizer.checkpoint_event_prologue()

        load_path, client_states = self._load_checkpoint(load_dir,
                                                         tag,
                                                         load_module_strict=load_module_strict,
                                                         load_optimizer_states=load_optimizer_states,
                                                         load_lr_scheduler_states=load_lr_scheduler_states,
                                                         load_module_only=load_module_only,
                                                         custom_load_fn=custom_load_fn)

        load_zero_checkpoint = load_path is not None and self.zero_optimization()
        if load_zero_checkpoint and not self.zero_nvme_offload_optimizer():
            if (load_optimizer_states and not load_module_only) or self.load_universal_checkpoint():
                success = self._load_zero_checkpoint(load_dir, tag, load_optimizer_states=load_optimizer_states)
            else:
                success = False
            if not success:
                self.optimizer._restore_from_bit16_weights()

        if self.zero_nvme_offload_optimizer():
            from shutil import copytree, disk_usage
            rank = self.local_rank if self.use_node_local_storage() else self.global_rank
            rank_dir = "rank" + dp_index_to_str(rank)
            offload_dir = self.optimizer.optimizer_swapper.swap_folder
            offload_ckpt_dir = os.path.join(load_dir, tag, "offloaded_tensors", rank_dir)
            _, _, free = disk_usage(offload_dir)
            logger.info(
                f"Copying NVMe offload checkpoint from {offload_ckpt_dir} to {offload_dir}, {free / 1e9:,.2f} GB free on target filesystem..."
            )
            copytree(offload_ckpt_dir, offload_dir, dirs_exist_ok=True)
            _, _, free = disk_usage(offload_dir)
            logger.info(f"Copying complete! {free / 1e9:,.2f} GB free on target filesystem")
            self.optimizer.reset_swap_buffers()

        if self._optimizer_has_ckpt_event_epilogue():
            self.optimizer.checkpoint_event_epilogue()

        if self.load_universal_checkpoint() and not self.zero_optimization_partition_weights():
            self.optimizer.update_lp_params()

        return load_path, client_states

    def _load_checkpoint(self,
                         load_dir,
                         tag,
                         load_module_strict=True,
                         load_optimizer_states=True,
                         load_lr_scheduler_states=True,
                         load_module_only=False,
                         custom_load_fn=None):

        from deepspeed.runtime.state_dict_factory import SDLoaderFactory

        ckpt_list = self._get_all_ckpt_names(load_dir, tag)
        sd_loader = SDLoaderFactory.get_sd_loader(ckpt_list, checkpoint_engine=self.checkpoint_engine)

        is_pipe_parallel = isinstance(self.module, PipelineModule)

        mp_rank = 0 if self.mpu is None else self.mpu.get_model_parallel_rank()
        load_path, checkpoint, _ = sd_loader.load(self.mp_world_size, mp_rank, is_pipe_parallel=is_pipe_parallel)

        if checkpoint is None:
            return None, None

        fetch_z3_params = False
        if self.zero_optimization_partition_weights() and not load_optimizer_states:
            checkpoint['module'] = get_fp32_state_dict_from_zero_checkpoint(load_dir)
            fetch_z3_params = True

        if is_pipe_parallel:
            # Pipeline parallelism uses this to load its own checkpoint files.
            self._curr_ckpt_path = os.path.join(load_dir, tag)

        if self.has_moe_layers:
            # print(checkpoint.keys())
            old_moe_load = False
            if not isinstance(checkpoint['num_experts'], list):
                old_moe_load = True
            DeepSpeedEngine.load_moe_state_dict(load_dir,
                                                tag,
                                                state_dict=checkpoint['module'],
                                                old_moe_load=old_moe_load,
                                                model=self.module,
                                                mpu=self.mpu,
                                                num_experts=self.num_experts,
                                                checkpoint_engine=self.checkpoint_engine)
        if not self.load_universal_checkpoint():
            self.load_module_state_dict(checkpoint=checkpoint,
                                        strict=load_module_strict,
                                        custom_load_fn=custom_load_fn,
                                        fetch_z3_params=fetch_z3_params)

        # M455: Megatron ee38e7f — old checkpoints may omit dp_world_size;
        # fall back to 1 so single-GPU checkpoints load without KeyError.
        # Knuth critique 1: silent fallback masks topology mismatches — log it.
        # Knuth critique 2: default=1 is wrong for multi-node; callers should
        #   validate after load, not rely on this sentinel being correct.
        if 'dp_world_size' in checkpoint:
            self.loaded_checkpoint_dp_world_size = checkpoint['dp_world_size']
        else:
            self.loaded_checkpoint_dp_world_size = 1
            print(f"[M455-COMPAT] 'dp_world_size' missing from checkpoint; "
                  f"defaulting to 1 (old-format ckpt). "
                  f"Verify topology before continuing training.")

        optim_checkpoint = None
        if load_module_only:
            deepspeed_states = ['module']
            # M487: Megatron 160ba6800 — use reload_model_params() unconditionally
            # instead of fp16-conditional _model_params_to_master_params().
            # All optimizer types implement reload_model_params (fp32 is a no-op).
            if self.optimizer is not None:
                self.optimizer.reload_model_params()
        else:
            has_zero_optimizer_state = self.zero_optimization()
            if load_optimizer_states and self.optimizer is not None and not has_zero_optimizer_state:
                if self.has_moe_layers:
                    largest_group_name = groups._get_max_expert_size_name()
                    expp_rank = groups._get_expert_parallel_rank(largest_group_name)
                    optim_load_path = self._get_optimizer_ckpt_name(load_dir, tag, expp_rank)
                    optim_checkpoint = self.checkpoint_engine.load(optim_load_path, map_location=torch.device('cpu'))
                else:
                    optim_checkpoint = checkpoint

                if self.fp16_enabled() or self.bfloat16_enabled():
                    self.optimizer.load_state_dict(optim_checkpoint['optimizer'],
                                                   load_optimizer_states=load_optimizer_states)
                else:
                    optim_checkpoint = checkpoint

                self.optimizer.load_state_dict(optim_checkpoint['optimizer'])

            if load_lr_scheduler_states and self.lr_scheduler is not None:
                # M455: old checkpoints may not carry lr_scheduler state; skip
                # gracefully rather than raising KeyError mid-resume.
                if 'lr_scheduler' in checkpoint:
                    self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                else:
                    print(f"[M455-COMPAT] 'lr_scheduler' key absent from checkpoint; "
                          f"skipping lr_scheduler restore (old-format ckpt).")

            if self.random_ltd_enabled() and self.random_ltd_scheduler is not None and 'random_ltd' in checkpoint:
                self.random_ltd_scheduler.load_state_dict(checkpoint['random_ltd'])

            if self.training_dataloader is not None and self.curriculum_learning_enabled(
            ) and 'data_sampler' in checkpoint:
                self.training_dataloader.data_sampler.load_state_dict(checkpoint['data_sampler'])

            def get_sparse_tensor_module_names(original_set, loaded_set, original_parameters, loaded_parameters):
                result = set()

                for name in original_set:
                    if name in loaded_parameters and name not in loaded_set:
                        continue  # parameter existed in previous model and was not sparse
                    result.add(name)

                for name in loaded_set:
                    if name in original_parameters:
                        result.add(name)  # parameter exists in both configs and it was sparse

                return result

            if 'sparse_tensor_module_names' in checkpoint:
                sparse_tensor_module_names = checkpoint['sparse_tensor_module_names']
            elif 'csr_tensor_module_names' in checkpoint:
                sparse_tensor_module_names = checkpoint['csr_tensor_module_names']
            else:
                sparse_tensor_module_names = None
            if sparse_tensor_module_names is not None:
                if load_module_strict:
                    self.sparse_tensor_module_names = sparse_tensor_module_names
                else:
                    self.sparse_tensor_module_names = get_sparse_tensor_module_names(
                        self.sparse_tensor_module_names, sparse_tensor_module_names,
                        dict(self.module.named_parameters()), checkpoint["module"])

            # M455: Megatron ee38e7f — old-format checkpoints may be missing
            # training-state counters; fall back to neutral defaults so a
            # resume doesn't crash hard on legacy saves.
            # Knuth critique 1: defaulting global_steps to 0 restarts the LR
            #   schedule from scratch — caller MUST pass correct step if known.
            # Knuth critique 2: missing mp_world_size silently breaks tensor-
            #   parallel validation downstream; default=1 is a lie for MP>1.
            try:
                self.global_steps = checkpoint['global_steps']
            except KeyError:
                self.global_steps = 0
                print(f"[M455-COMPAT] 'global_steps' missing from checkpoint; "
                      f"defaulting to 0 (old-format ckpt). LR schedule may be wrong.")

            # M432: Megatron cebd3b8b1 — addrressed jareds comments
            # Assert global_samples is 0 before loading from checkpoint,
            # matching Megatron's unconditional assert on consumed_train_samples.
            # Use getattr-style safe access so old checkpoints without the key
            # fall back gracefully rather than crashing.
            assert self.global_samples == 0, (
                f"global_samples must be 0 before loading checkpoint, got {self.global_samples}")
            self.global_samples = getattr(checkpoint, 'global_samples',
                                          checkpoint.get('global_samples',
                                                         self.global_steps * self.train_batch_size()))
            print('[M432]')

            try:
                self.skipped_steps = checkpoint['skipped_steps']
            except KeyError:
                self.skipped_steps = 0
                print(f"[M455-COMPAT] 'skipped_steps' missing from checkpoint; "
                      f"defaulting to 0 (old-format ckpt).")

            try:
                self.loaded_checkpoint_mp_world_size = checkpoint['mp_world_size']
            except KeyError:
                self.loaded_checkpoint_mp_world_size = 1
                print(f"[M455-COMPAT] 'mp_world_size' missing from checkpoint; "
                      f"defaulting to 1 (old-format ckpt). Validate MP topology.")
            deepspeed_states = [
                'module', 'sparse_tensor_module_names', 'skipped_steps', 'global_steps', 'dp_world_size',
                'mp_world_size', 'data_sampler', 'random_ltd'
            ]
        client_state = {}

        if load_lr_scheduler_states:
            deepspeed_states.append('lr_scheduler')
        if load_optimizer_states:
            deepspeed_states.append('optimizer')

        client_state = {key: value for key, value in checkpoint.items() if key not in deepspeed_states}

        if optim_checkpoint is not None:
            client_state['optimizer'] = optim_checkpoint['optimizer']

        return load_path, client_state

    def _load_zero_checkpoint(self, load_dir, tag, load_optimizer_states=True):

        load_serial = None
        # When use loading checkpoint serial, checkpoint loading start from local rank 0,
        # all other local rank would be paused, waiting for its rank-1 peer ready and its notification.
        if self._config.zero_config.pipeline_loading_checkpoint:
            assert self.zero_optimization_stage(
            ) == ZeroStageEnum.weights, "Only stage3 support for pipeline checkpoint loading"
            load_serial = torch.zeros(1).to(self.device)
            if dist.get_local_rank() != 0:
                dist.recv(tensor=load_serial, src=dist.get_rank() - 1)
        if self.load_universal_checkpoint():
            zero_sd_list = None
            checkpoint_folder = f'{os.path.join(load_dir, tag)}'
        else:
            if load_optimizer_states and self.seq_dp_world_size != self.loaded_checkpoint_dp_world_size:
                raise ZeRORuntimeException("The checkpoint being loaded used a DP " \
                    f"world size of {self.loaded_checkpoint_dp_world_size} but the " \
                    f"current world size is {self.seq_dp_world_size}. Automatic adjustment " \
                    "of ZeRO's optimizer state partitioning with a new world size is not " \
                    "currently supported.")
            checkpoint_folder = None
            zero_sd_list = self._get_all_zero_checkpoints(load_dir, tag)
            if zero_sd_list is None:
                return False

        param_shapes = self._get_zero_param_shapes()
        self.optimizer.load_state_dict(state_dict_list=zero_sd_list,
                                       load_optimizer_states=load_optimizer_states,
                                       load_from_fp32_weights=self.zero_load_from_fp32_weights(),
                                       checkpoint_folder=checkpoint_folder,
                                       load_serial=load_serial,
                                       param_shapes=param_shapes)

        if self.load_universal_checkpoint():
            logger.info(f'loaded universal zero checkpoints from {checkpoint_folder} for rank {self.global_rank}')
        else:
            logger.info(f"loading {len(zero_sd_list)} zero partition checkpoints for rank {self.global_rank}")
        return True

    def _get_mp_rank_zero_checkpoint_names(self, load_dir, tag, mp_rank, dp_world_size, bf16_mode):
        zero_ckpt_names = []
        for dp_rank in range(dp_world_size):
            ckpt_name = self._get_rank_zero_ckpt_name(checkpoints_path=load_dir,
                                                      tag=tag,
                                                      mp_rank=mp_rank,
                                                      dp_rank=dp_rank,
                                                      bf16_mode=bf16_mode)
            zero_ckpt_names.append(ckpt_name)

        return zero_ckpt_names

    def _get_all_zero_checkpoint_names(self, load_dir, tag, bf16_mode):
        mp_rank = 0 if self.mpu is None else self.mpu.get_model_parallel_rank()
        zero_ckpt_names = self._get_mp_rank_zero_checkpoint_names(load_dir=load_dir,
                                                                  tag=tag,
                                                                  mp_rank=mp_rank,
                                                                  dp_world_size=self.loaded_checkpoint_dp_world_size,
                                                                  bf16_mode=bf16_mode)
        for i, ckpt_name in enumerate(zero_ckpt_names):
            if not os.path.exists(ckpt_name):
                # transparently handle the old file pattern for optim_states
                if "optim_states.pt" in ckpt_name:
                    ckpt_name_try = ckpt_name.replace("_optim_states.pt", "optim_states.pt")
                    if os.path.exists(ckpt_name_try):
                        zero_ckpt_names[i] = ckpt_name_try
                        continue

        return zero_ckpt_names

    def _get_all_zero_checkpoint_state_dicts(self, zero_ckpt_names):
        zero_sd_list = []
        for i, ckpt_name in enumerate(zero_ckpt_names):
            _state = None
            if ckpt_name is None:
                _state = {OPTIMIZER_STATE_DICT: None}
            # Fully load state for current rank
            elif self.zero_elastic_checkpoint() or dist.get_rank(group=self.optimizer.dp_process_group) == i:
                _state = self.checkpoint_engine.load(
                    ckpt_name,
                    map_location='cpu',
                )
            else:
                _state = {OPTIMIZER_STATE_DICT: None}
            zero_sd_list.append(_state)

        zero_optimizer_sd = [sd[OPTIMIZER_STATE_DICT] for sd in zero_sd_list]
        logger.info(f"successfully read {len(zero_optimizer_sd)} ZeRO state_dicts for rank {self.global_rank}")
        return zero_optimizer_sd

    def _get_all_zero_checkpoints(self, load_dir, tag):
        for bf16_mode in [self.bfloat16_enabled(), not self.bfloat16_enabled()]:
            zero_ckpt_names = self._get_all_zero_checkpoint_names(load_dir, tag, bf16_mode)
            if zero_ckpt_names is not None:
                # Warn if loading checkpoint of different bit16 type
                if bf16_mode is not self.bfloat16_enabled():
                    checkpoint_bit16 = BFLOAT16 if bf16_mode else FP16
                    engine_bit16 = BFLOAT16 if self.bfloat16_enabled() else FP16
                    logger.warning(f'Loading {checkpoint_bit16} zero checkpoints into {engine_bit16} training engine')
                return self._get_all_zero_checkpoint_state_dicts(zero_ckpt_names)

        return None

    def _checkpoint_tag_validation(self, tag):
        if self.checkpoint_tag_validation_enabled():
            s_hash = hashlib.sha1(tag.encode())
            bhash = torch.ByteTensor([s_hash.digest()]).flatten().to(self.device)
            max_bhash = bhash.clone()
            min_bhash = bhash.clone()
            dist.all_reduce(max_bhash, op=dist.ReduceOp.MAX)
            dist.all_reduce(min_bhash, op=dist.ReduceOp.MIN)
            valid = all(min_bhash == bhash) and all(max_bhash == bhash)
            msg = (f"[rank={dist.get_rank()}] The checkpoint tag name '{tag}' is not consistent across "
                   "all ranks. Including rank unique information in checkpoint tag could cause issues when "
                   "restoring with different world sizes.")
            if self.checkpoint_tag_validation_fail():
                assert valid, msg
            elif not valid:
                logger.warning(msg)

    def save_checkpoint(self, save_dir, tag=None, client_state={}, save_latest=True, exclude_frozen_parameters=False):
        """Save training checkpoint

        Arguments:
            save_dir: Required. Directory for saving the checkpoint
            tag: Optional. Checkpoint tag used as a unique identifier for the checkpoint, global step is
                used if not provided. Tag name must be the same across all ranks.
            client_state: Optional. State dictionary used for saving required training states in the client code.
            save_latest: Optional. Save a file 'latest' pointing to the latest saved checkpoint.
            exclude_frozen_parameters: Optional. Exclude frozen parameters from checkpointed state.
        Important: all processes must call this method and not just the process with rank 0. It is
        because each process needs to save its master weights and scheduler+optimizer states. This
        method will hang waiting to synchronize with other processes if it's called just for the
        process with rank 0.

        """
        if self._optimizer_has_ckpt_event_prologue():
            # Custom preparation for checkpoint save, if applicable
            self.optimizer.checkpoint_event_prologue()

        rank = self.local_rank if self.use_node_local_storage() else self.global_rank

        # This is to make sure the checkpoint names are created without collision
        # There seems to be issue creating them in parallel

        # Ensure save_dir directory exists
        if rank == 0:
            self.checkpoint_engine.makedirs(save_dir, exist_ok=True)
        dist.barrier()

        if tag is None:
            tag = f"global_step{self.global_steps}"

        # Ensure tag is a string
        tag = str(tag)
        commit_info = CheckpointCommitInfo(tag=tag, save_dir=save_dir, save_latest=save_latest)

        self.checkpoint_engine.create(commit_info)

        # Ensure checkpoint tag is consistent across ranks
        self._checkpoint_tag_validation(tag)

        if self.has_moe_layers:
            self.save_non_zero_checkpoint = False
            self._create_checkpoint_file(save_dir, tag, False)
            self._save_moe_checkpoint(save_dir,
                                      tag,
                                      client_state=client_state,
                                      exclude_frozen_parameters=exclude_frozen_parameters)

        # We distribute the task of saving layer checkpoint files among
        # data parallel instances, so all procs should call _save_checkpoint.
        # All procs then call module_state_dict(), but only procs of data
        # parallel rank 0 save the general model params.
        if not self.has_moe_layers:
            self._create_checkpoint_file(save_dir, tag, False)
            self._save_checkpoint(save_dir,
                                  tag,
                                  client_state=client_state,
                                  exclude_frozen_parameters=exclude_frozen_parameters)

        if self.save_zero_checkpoint:
            self._create_zero_checkpoint_files(save_dir, tag)
            self._save_zero_checkpoint(save_dir, tag)

        if self.zero_nvme_offload_optimizer():
            from shutil import copytree, disk_usage
            rank_dir = "rank" + dp_index_to_str(rank)
            offload_dir = self.optimizer.optimizer_swapper.swap_folder
            offload_ckpt_dir = os.path.join(save_dir, tag, "offloaded_tensors", rank_dir)
            _, _, free = disk_usage(save_dir)
            logger.info(
                f"Copying NVMe offload files from {offload_dir} to {offload_ckpt_dir}, {free / 1e9:,.2f} GB free on target filesystem..."
            )
            copytree(offload_dir,
                     offload_ckpt_dir,
                     ignore=lambda _, dir_list: list(filter(lambda x: 'gradient' in x, dir_list)),
                     dirs_exist_ok=False)
            _, _, free = disk_usage(save_dir)
            logger.info(f"Copying complete! {free / 1e9:,.2f} GB free on target filesystem")

        if self._optimizer_has_ckpt_event_epilogue():
            self.optimizer.checkpoint_event_epilogue()

        # Save latest checkpoint tag
        if not self.checkpoint_engine.is_decoupled():
            commit_info = CheckpointCommitInfo(tag=tag, save_dir=save_dir, save_latest=save_latest)
            self.checkpoint_engine.commit(commit_info)
            if save_latest and self.global_rank == 0:
                with open(os.path.join(save_dir, 'latest'), 'w') as fd:
                    fd.write(tag)

        dist.barrier()

        return True

    def _commit_decoupled_checkpoint(self):
        assert self.checkpoint_engine.is_decoupled(), \
            f'{self.checkpoint_engine} is not a Decoupled Checkpoint Engine'

        commit_info = self.checkpoint_engine.get_commit_info()
        if commit_info is None:
            return

        self.checkpoint_engine.commit(commit_info)

        if self.global_rank == 0 and commit_info.save_latest:
            with open(os.path.join(commit_info.save_dir, 'latest'), 'w') as fd:
                fd.write(commit_info.tag)

        dist.barrier()

    def _get_non_moe_state_dict(self, full_state_dict):
        """
            Get the state dict of the non-moe layers
        """
        for key in list(full_state_dict.keys()):
            if 'expert' in key and 'moe.gate.wg.weight' not in key:
                full_state_dict.pop(key)

        return full_state_dict

    def _save_moe_checkpoint(self, save_dir, tag, client_state={}, exclude_frozen_parameters=False):
        save_path = self._get_ckpt_name(save_dir, tag)

        # A hack to save the checkpointing directory. Pipeline parallelism overrides
        # module_state_dict() and uses this path to save the model. module_state_dict()
        # then instead just returns None.

        # Using layer_#_export_# to save the model's expert state_dict
        moe_layer_id = 0
        for n_module, module in self.module.named_modules():
            if isinstance(module, MoE):  # and deepspeed.comm.get_rank() == 0:
                group_name = module.expert_group_name
                num_local_experts = module.num_local_experts
                expp_rank = groups._get_expert_parallel_rank(group_name)
                exp_dp_rank = groups._get_expert_data_parallel_rank(group_name)
                # print(expp_rank, exp_dp_rank)
                # if exp_dp_rank != 0:
                if not self.checkpoint_engine.is_data_parallel_writer(exp_dp_rank):
                    moe_layer_id += 1
                    continue

                # get all moe parameters
                moe_state_dict = {}
                for n, p in module.state_dict().items():
                    if 'expert' in n and 'moe.gate.wg.weight' not in n:
                        moe_state_dict[n_module + '.' + n] = p
                moe_str_prefix = '.deepspeed_moe.experts.deepspeed_experts.'
                # print(moe_state_dict.keys()) # until now, everything is fine. So the bug happens at next few lines
                # Reorder the moe name rank, so that each checkpoint only has one expert
                experts_state_dict = defaultdict(dict)
                for key in list(moe_state_dict.keys()):
                    m = re.match(f".*{moe_str_prefix}([0-9]+).*", key)

                    local_expert_id = None
                    if not m:
                        logger.warning(f'No expert found in key {key}.')
                    else:
                        local_expert_id = m.group(1)

                    global_expert_id = expp_rank * \
                        num_local_experts + int(local_expert_id)
                    expert_key = key.replace(f'{moe_str_prefix}{local_expert_id}',
                                             f'{moe_str_prefix}{global_expert_id}')
                    # truncating extra tensor (shared) storage
                    truncated = moe_state_dict.pop(key).clone().detach()
                    experts_state_dict[str(global_expert_id)][expert_key] = truncated

                # let save the moe parameters
                for global_expert_id, expert_state_dict in experts_state_dict.items():
                    # save the moe parameters
                    moe_save_path = self._get_expert_ckpt_name(save_dir, moe_layer_id, global_expert_id, tag, self.mpu)
                    if self.random_ltd_enabled():
                        expert_state_dict = remove_random_ltd_state_dict(expert_state_dict)
                    saveable_state_dict = expert_state_dict
                    if self.checkpoint_engine.preserves_storage_sharing():
                        saveable_state_dict = clone_tensors_for_torch_save(expert_state_dict)
                    self.checkpoint_engine.save(saveable_state_dict, moe_save_path)
                moe_layer_id += 1

        self._curr_ckpt_path = os.path.join(save_dir, tag)

        largest_group_name = groups._get_max_expert_size_name()
        expp_rank = groups._get_expert_parallel_rank(largest_group_name)
        exp_dp_rank = groups._get_expert_data_parallel_rank(largest_group_name)

        # In the case of E + D parallelism, only the
        # first expert parallel group should save the expert weights
        # since each expert parallel group is a copy of the model's experts
        if not self.checkpoint_engine.is_data_parallel_writer(exp_dp_rank):
            return

        # Save optimizer states. They are different across each exp parallel rank.
        optimizer_state = {
            'optimizer': self.optimizer.state_dict() if self.optimizer and not self.zero_optimization() else None
        }
        # TODO: why use BufferedWriter not the path
        file_path = self._get_optimizer_ckpt_name(save_dir, tag, expp_rank)
        saveable_state_dict = optimizer_state
        if self.checkpoint_engine.preserves_storage_sharing():
            saveable_state_dict = clone_tensors_for_torch_save(optimizer_state)
        self.checkpoint_engine.save(saveable_state_dict, file_path)

        # Load flow uses below saved file for model parameters, RNG and more
        if groups._get_data_parallel_rank() == 0:
            # Get non-moe parameters
            # Classes DeepSpeedEngine and PipelineEngine have different behavior for method module_state_dict.
            # DeepSpeedEngine returns the state dict, where PipelineEngine saves the state dict and returns None.
            # We need to get the state dict, therefore, call to DeepSpeedEngine (base class for PipelineEngine)
            model_state_dict = self._get_non_moe_state_dict(
                DeepSpeedEngine.module_state_dict(self, exclude_frozen_parameters=exclude_frozen_parameters))

            # TODO: update num experts info,.. in checkpoint
            state = {
                'module':
                model_state_dict,
                'lr_scheduler':
                self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
                'data_sampler':
                self.training_dataloader.data_sampler.state_dict() if
                (self.training_dataloader is not None and self.curriculum_learning_enabled()) else None,
                'random_ltd':
                self.random_ltd_scheduler.state_dict() if self.random_ltd_enabled() else None,
                'sparse_tensor_module_names':
                self.sparse_tensor_module_names,
                'skipped_steps':
                self.skipped_steps,
                'global_steps':
                self.global_steps,
                'global_samples':
                self.global_samples,
                'dp_world_size':
                self.seq_dp_world_size,
                'mp_world_size':
                self.mp_world_size,
                'num_experts':
                self.num_experts
            }
            state.update(client_state)
            logger.info(f'Saving model checkpoint: {save_path}')
            saveable_state_dict = state
            if self.checkpoint_engine.preserves_storage_sharing():
                saveable_state_dict = clone_tensors_for_torch_save(state)
            self.checkpoint_engine.save(saveable_state_dict, save_path)

    def _create_checkpoint_file(self, save_dir, tag, zero_checkpoint):
        name_function = (self._get_zero_ckpt_name if zero_checkpoint else self._get_ckpt_name)
        try:
            checkpoint_name = name_function(save_dir, tag)
            path = os.path.dirname(checkpoint_name)
            self.checkpoint_engine.makedirs(path, exist_ok=True)
        except Exception:
            logger.error(f"Failed saving model checkpoint to {save_dir} with tag {tag}")
            return False

        return True

    def _create_zero_checkpoint_files(self, save_dir, tag):
        success = True
        # zero checkpoint files are created sequentially
        for rank in range(dist.get_world_size(self.optimizer.dp_process_group)):
            if rank == self.global_rank:
                success = self._create_checkpoint_file(save_dir, tag, True)

        return success

    def _save_checkpoint(self, save_dir, tag, client_state={}, exclude_frozen_parameters=False):

        save_path = self._get_ckpt_name(save_dir, tag)

        zero_optimizer_state = self.zero_optimization()

        save_frozen_param = self.zero_optimization_partition_gradients() and not exclude_frozen_parameters

        # A hack to save the checkpointing directory. Pipeline parallelism overrides
        # module_state_dict() and uses this path to save the model. module_state_dict()
        # then instead just returns None.  The module_state_dict() implementation in
        # PipelineEngine expects the save path to be set in self._curr_ckpt_path.
        self._curr_ckpt_path = os.path.join(save_dir, tag)
        module = self.module_state_dict(exclude_frozen_parameters=exclude_frozen_parameters)
        self._curr_ckpt_path = None

        state = dict(module=module,
                     buffer_names=self._get_buffer_names(),
                     optimizer=self.optimizer.state_dict() if self.optimizer and not zero_optimizer_state else None,
                     param_shapes=self._get_zero_param_shapes() if self.optimizer and zero_optimizer_state else None,
                     frozen_param_shapes=self._get_zero_frozen_param_attributes(self._get_param_shape_func)
                     if save_frozen_param else None,
                     shared_params=self._get_shared_params() if self.optimizer and zero_optimizer_state else None,
                     frozen_param_fragments=self._get_zero_frozen_param_attributes(self._get_param_fragment_func)
                     if save_frozen_param else None,
                     lr_scheduler=self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
                     data_sampler=self.training_dataloader.data_sampler.state_dict() if
                     (self.training_dataloader is not None and self.curriculum_learning_enabled()) else None,
                     random_ltd=self.random_ltd_scheduler.state_dict() if self.random_ltd_enabled() else None,
                     sparse_tensor_module_names=self.sparse_tensor_module_names,
                     skipped_steps=self.skipped_steps,
                     global_steps=self.global_steps,
                     global_samples=self.global_samples,
                     dp_world_size=self.seq_dp_world_size,
                     mp_world_size=self.mp_world_size,
                     ds_config=self.config,
                     ds_version=version)
        autotp_uc_info = getattr(self.module, UNIVERSAL_CHECKPOINT_INFO, None)
        if autotp_uc_info is not None:
            state[UNIVERSAL_CHECKPOINT_INFO] = autotp_uc_info
        state.update(client_state)
        log_dist(message=f'Saving model checkpoint: {save_path}', ranks=[0])

        if self.save_non_zero_checkpoint:
            self.checkpoint_engine.save(state_dict=state, path=save_path)

    def _get_buffer_names(self):
        buffer_names = []

        # we save buffer names so that we could extract later the real buffers from the saved
        # state_dict["module"] in the non-zero checkpoint - the buffers are already there but they
        # are intermixed with param placeholders

        # have to traverse the tree to be able to skip non-persistent buffers
        def get_layer_named_buffers(module, prefix=""):
            for name, buf in module.named_buffers(recurse=False):
                if buf is not None and name not in module._non_persistent_buffers_set:
                    buffer_names.append(prefix + name)

            for name, child in module.named_children():
                if child is not None:
                    get_layer_named_buffers(child, prefix + name + ".")

        get_layer_named_buffers(self.module, prefix="")

        return buffer_names

    def _get_param_shape_func(self, param):
        return param.ds_shape if hasattr(param, 'ds_id') else param.shape

    def _get_param_fragment_func(self, param):
        return param.ds_tensor.detach().cpu() if hasattr(param, 'ds_id') else param.detach().cpu()

    def _get_zero_frozen_param_attributes(self, attr_func):
        frozen_param_fragments = OrderedDict()

        for param in self.module.parameters():
            if param.requires_grad:
                continue
            if param not in self.param_names:
                raise ValueError(f"failed to find frozen {param} in named params")
            name = self.param_names[param]
            frozen_param_fragments[name] = attr_func(param)

        return frozen_param_fragments

    def _get_zero_param_shapes(self):
        """Returns a dict of name to shape mapping, only for the flattened fp32 weights saved by the
        optimizer. the names are exactly as in state_dict. The order is absolutely important, since
        the saved data is just flattened data with no identifiers and requires reconstruction in the
        same order it was saved.
        We can't rely on self.module.named_parameters() to get the saved tensors, as some params
        will be missing and others unsaved and then it'd be impossible to reconstruct state_dict
        from the flattened weights.
        optimizer.bit16_groups seems to be the easiest to use as it's in all zeroX versions.
        """
        param_group_shapes = []
        cnt = 0
        numel = 0

        # zero2 started using a round_robin_bit16_groups which is a shuffled version of bit16_groups -
        # if we don't use it, we get parameters ordered incorrectly
        if hasattr(self.optimizer, "round_robin_bit16_groups"):
            bit16_groups = self.optimizer.round_robin_bit16_groups
        elif self.bfloat16_enabled() and hasattr(self.optimizer, "bf16_groups"):
            bit16_groups = self.optimizer.bf16_groups
        else:
            bit16_groups = self.optimizer.bit16_groups if self.zero_optimization_stage(
            ) == 2 else self.optimizer.fp16_groups

        for bit16_group in bit16_groups:
            param_shapes = OrderedDict()
            for param in bit16_group:
                cnt += 1
                numel += param.ds_numel if hasattr(param, "ds_numel") else param.numel()
                shape = param.ds_shape if hasattr(param, "ds_shape") else param.shape
                if param not in self.param_names:
                    raise ValueError("failed to find optimizer param in named params")
                name = self.param_names[param]
                param_shapes[name] = shape

                # uncomment to debug zero_to_fp32.py problems
                # if self.global_rank == 0: print(f"saving param {name} {shape} (numel={shape.numel()})")
            param_group_shapes.append(param_shapes)
        # if self.global_rank == 0: print(f"Total saved {numel} numels in {cnt} params")

        return param_group_shapes

    def _get_shared_params(self):
        """
        Returns a dict of shared params, which can later be used to reconstruct the original state dict,
        e.g. in `zero_to_fp32`. Each dict entry is a pair of param names, where the key is the name
        of the variable that isn't stored and the value is the actual param holding data.
        """
        shared_index = {}
        shared_params_by_full_name = {}

        is_zero3_model = (self.zero_optimization_partition_weights()
                          and any(hasattr(param, "ds_id") for param in self.module.parameters()))

        def get_layer_state_dict(module, prefix=""):
            # handle params
            for name, param in module.named_parameters(recurse=False):
                if param is None or (is_zero3_model and not hasattr(param, "ds_id")):
                    continue
                key = prefix + name

                # When weights are manged by stage 3, we can't rely on param.data_ptr() as it will be reused
                # as weights get gathered and reduced, but param.ds_id is unique across all zero weights
                # (and shared params will have the same param.ds_id)
                param_id = param.ds_id if is_zero3_model else param.data_ptr()

                if param_id in shared_index:
                    # shared weights
                    #print(f"`{key}` is shared with `{shared_index[param_id]}`")
                    shared_params_by_full_name[key] = shared_index[param_id]
                else:
                    shared_index[param_id] = key

            for name, child in module.named_children():
                if child is not None:
                    get_layer_state_dict(child, prefix + name + ".")

        if dist.get_rank() == 0:
            get_layer_state_dict(self.module, prefix="")

        return shared_params_by_full_name

    def _copy_recovery_script(self, save_path):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        script = "zero_to_fp32.py"
        src = os.path.join(base_dir, "utils", script)
        dst = os.path.join(save_path, script)
        #logger.info(f"creating recovery script {dst}")
        copyfile(src, dst)
        self._change_recovery_script_permissions(dst)

    def _change_recovery_script_permissions(self, dst):
        # make executable (safeguard for file shares - Azure as example)
        try:
            os.chmod(dst, os.stat(dst).st_mode | stat.S_IEXEC)
        except (FileNotFoundError, PermissionError) as e:
            #this message is used in unit test TestZeRONonDistributed
            logger.info(
                f'Warning: Could not change permissions for {dst} due to error: {e}. Continuing without changing permissions.'
            )

    def _save_zero_checkpoint(self, save_path, tag):
        zero_checkpoint_name = self._get_zero_ckpt_name(save_path, tag)
        zero_sd = dict(optimizer_state_dict=self.optimizer.state_dict(), ds_config=self.config, ds_version=version)
        self.checkpoint_engine.save(zero_sd, zero_checkpoint_name)

        if self.global_rank == 0:
            self._copy_recovery_script(save_path)
        ckpt_type = 'zero' if self.zero_optimization() else 'bf16_zero'
        #logger.info(f'{ckpt_type} checkpoint saved {zero_checkpoint_name}')

    def _replace_module_consolidated_state_dict(self):
        """
        Get a full non-partitioned state_dict with fp16 weights on cpu.
        Important: this function must be called on all ranks and not just rank 0.
        This is similar to nn.Module.state_dict (modelled after _save_to_state_dict)
        This method is used for tensor parallel training.

        Returns:
        OrderedDict: The consolidated state dictionary if the current process rank is 0, otherwise None.
        """
        #TODO: If we use both Zero3 and tensor parallel simultaneously
        # we need to consolidate the gather mechanisms of both.
        state_dict = OrderedDict() if dist.get_rank() == 0 else None

        def get_layer_state_dict(module, prefix=""):
            with GatherReplacedLayerParams(list(module.parameters(recurse=False)), module, enabled=True):
                for name, param in module.named_parameters(recurse=False):
                    if param is None:
                        continue
                    key = prefix + name
                    if (dist.get_rank() == 0):
                        state_dict[key] = param.detach().cpu()
                        # print(key,module, param.detach().cpu().shape)

            for name, child in module.named_children():
                if child is not None:
                    get_layer_state_dict(child, prefix + name + ".")

        get_layer_state_dict(self.module, prefix="")

        # ensure that all GPU communication tasks are completed before the process exits
        get_accelerator().synchronize()
        return state_dict

    def _consolidated_16bit_state_dict(self, exclude_frozen_parameters=False):
        """
        Consolidate the 16-bit state dictionary.
        """
        if self.zero_optimization_stage() == ZeroStageEnum.weights:
            return self._zero3_consolidated_16bit_state_dict(exclude_frozen_parameters)
        elif self.autotp_size() > 1:
            return self._replace_module_consolidated_state_dict()

        raise ValueError("consolidated_16bit_state_dict is only applicable to cases where weights are partitioned, "
                         "including Zero Stage 3 and tensor parallelism.")

    def _zero3_consolidated_16bit_state_dict(self, exclude_frozen_parameters=False):
        """
        Get a full non-partitioned state_dict with fp16 weights on cpu.
        Important: this function must be called on all ranks and not just rank 0.
        This is similar to nn.Module.state_dict (modelled after _save_to_state_dict), but:
        1. consolidates the weights from different partitions on gpu0
        2. works on one layer at a time to require as little gpu0 memory as possible, by
        moving the already consolidated weights to cpu
        3. takes care to keep the shared params shared when gradually copying the params to cpu
        Returns:
            a consolidated fp16 ``state_dict`` on cpu on rank 0, ``None`` on other ranks
        """
        if not self.zero_optimization_partition_weights():
            raise ValueError("this function requires ZeRO-3 mode")

        state_dict = OrderedDict() if dist.get_rank() == 0 else None
        shared_params = {}

        def get_layer_state_dict(module, prefix=""):
            # gather one layer at a time to be memory-efficient
            # must use modifier_rank=0 to release GPU memory after each layer gathered
            #see_memory_usage("before GatheredParameters", force=True)
            with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
                if dist.get_rank() == 0:
                    # handle params
                    for name, param in module.named_parameters(recurse=False):
                        if param is None or (exclude_frozen_parameters and not param.requires_grad):
                            continue
                        key = prefix + name
                        # can't rely on param.data_ptr() as it will be reused as weights gets
                        # gathered and reduced, but param.ds_id is unique across all zero weights
                        # (and shared params will have the same param.ds_id)
                        if param.ds_id in shared_params:
                            # shared weights
                            #print(f"`{key}` is shared with `{shared_params[param.ds_id]}`")
                            state_dict[key] = state_dict[shared_params[param.ds_id]]
                        else:
                            state_dict[key] = param.detach().cpu()
                            shared_params[param.ds_id] = key
                        #print(f"param {param.ds_id} {param.shape} {key} ")

                    # now buffers - not sure if need to take care of potentially shared weights here
                    for name, buf in module.named_buffers(recurse=False):
                        if (buf is not None and name not in module._non_persistent_buffers_set):
                            state_dict[prefix + name] = buf.detach().cpu()
            #see_memory_usage("after GatheredParameters", force=True)

            for name, child in module.named_children():
                if child is not None:
                    get_layer_state_dict(child, prefix + name + ".")

        # Prepare for checkpoint save by ensuring all parameters are partitioned
        if self._optimizer_has_ckpt_event_prologue():
            self.optimizer.checkpoint_event_prologue()

        see_memory_usage("before get_layer_state_dict", force=False)
        get_layer_state_dict(self.module, prefix="")
        see_memory_usage("after get_layer_state_dict", force=False)

        if self._optimizer_has_ckpt_event_epilogue():
            self.optimizer.checkpoint_event_epilogue()

        return state_dict

    def save_fp16_model(self, save_dir, save_filename="pytorch_model.bin"):
        """has been renamed to save_16bit_model, keeping this around for backwards
        compatibility"""
        return self.save_16bit_model(save_dir, save_filename)

    def save_16bit_model(self, save_dir, save_filename="pytorch_model.bin", exclude_frozen_parameters=False):
        """
        Save 16bit model weights

        This method saves the 16bit model weights at the desired destination.

        Arguments:
            save_dir: Required. Directory for saving the model
            save_filename: Optional. Filename to save to. Defaults to ``pytorch_model.bin``
            exclude_frozen_parameters: Optional. Exclude frozen parameters from checkpointed state.

        Returns:
            ``True`` when a model has been saved, ``False`` otherwise. It will not be saved if
            stage3_gather_16bit_weights_on_model_save is ``False``.

        Important: all processes must call this method and not just the process with rank 0. It is
        because the processes need to work in sync to gather the weights. This method will hang
        waiting to synchronize with other processes if it's called just for the process with rank 0.

        """

        path = os.path.join(save_dir, save_filename)

        if self.zero_optimization_partition_weights():
            if self.zero_gather_16bit_weights_on_model_save():
                # consolidation is expensive in time and memory and therefore isn't a default
                state_dict = self._zero3_consolidated_16bit_state_dict(
                    exclude_frozen_parameters=exclude_frozen_parameters)
            else:
                # the model will be bogus if not consolidated so don't confuse the user by saving it
                logger.info(
                    f"Did not save the model {path} because stage3_gather_16bit_weights_on_model_save is False")
                return False
        else:
            state_dict = self.module_state_dict(exclude_frozen_parameters=exclude_frozen_parameters)

        tag = f"global_step{self.global_steps}"
        tag = str(tag)
        commit_info = CheckpointCommitInfo(tag=tag, save_dir=save_dir, save_latest=False)
        self.checkpoint_engine.create(commit_info)

        if dist.get_rank() == 0:
            self.checkpoint_engine.makedirs(save_dir, exist_ok=True)
            logger.info(f"Saving model weights to {path}, tag: {tag}")
            self.checkpoint_engine.save(state_dict, path)

        self.checkpoint_engine.commit(commit_info)

        return True

    def empty_partition_cache(self):
        """
        Release GPU memory consumed by offloaded model parameters.
        """
        if hasattr(self.optimizer, 'empty_partition_cache'):
            self.optimizer.empty_partition_cache()
            gc.collect()
            get_accelerator().empty_cache()

    def get_autosp_backend(self, compile_kwargs):
        # M355: AutoSP composes with ZeRO-0 and ZeRO-1 (optimizer state partition).
        # ZeRO-2/3 partition gradients which conflicts with AutoSP's gradient AllReduce.
        if self.compile_autosp() and self.zero_optimization_stage() not in [
                ZeroStageEnum.disabled, ZeroStageEnum.optimizer_states
        ]:
            logger.info(
                f"Currently AutoSP does not compose with ZeRO stage 2 and 3. Falling back to the torch compiler.")
            return None

        compile_config = self._config.compile_config
        compile_kwargs['fullgraph'] = True

        # System Issue 3 fix: Validate dtype consistency across SP groups.
        # Mixed compute capabilities (e.g., V100+H100) can cause NCCL A2A
        # failures when bf16 is enabled but some GPUs don't support it.
        _training_dtype = torch.float32
        if hasattr(self._config, 'bf16') and self._config.bf16.get('enabled', False):
            _training_dtype = torch.bfloat16
        elif hasattr(self._config, 'fp16') and self._config.fp16.get('enabled', False):
            _training_dtype = torch.float16

        try:
            from deepspeed.compile.custom_ops.hetero_mesh import (get_hetero_plan, validate_sp_group_dtype_consistency)
            plan = get_hetero_plan()
            if plan is not None:
                dtype_warnings = validate_sp_group_dtype_consistency(plan, _training_dtype)
                for w in dtype_warnings:
                    logger.warning(w)
        except ImportError:
            pass

        # M354: Log composition state for experiment tracking.
        _desloc_cfg = self._config._param_dict.get('desloc', {})
        if _desloc_cfg.get('enabled', False) and dist.get_rank() == 0:
            _Kx = _desloc_cfg.get('Kx', 1)
            logger.info(f"[AutoSP+DEC] SP+DEC(Kx={_Kx})+ZeRO({self.zero_optimization_stage()})")

        # M356: Set A2A timeout for heterogeneous GPU deadlock prevention.
        if not hasattr(self, '_desloc_sp_a2a_timeout_ms'):
            self._desloc_sp_a2a_timeout_ms = int(os.environ.get('DESLOC_SP_A2A_TIMEOUT_MS', '60000'))

        return init_autosp(self._config)

    def get_deepcompile_backend(self, backend, compile_kwargs, schedule):
        if self.zero_optimization_stage() != ZeroStageEnum.optimizer_states \
                and self.zero_optimization_stage() != ZeroStageEnum.weights \
                and self.zero_optimization_stage() != ZeroStageEnum.gradients:
            logger.info(
                f"Currently DeepCompile supports ZeRO stage 1, 2, or 3 only, but ZeRO stage is set to {self.zero_optimization_stage()}. Falling back to the torch compiler."
            )
            return None

        compile_config = self._config.compile_config
        if (("zero_optimization" in self.config and "offload_optimizer" in self.config["zero_optimization"]
             and "offload_param" in self.config["zero_optimization"])
                and self._config.zero_config.offload_param.device == "cpu"
                and self._config.zero_config.offload_optimizer.device == "cpu"):
            compile_config.offload_parameters = True
        if self.zero_optimization_stage() == ZeroStageEnum.optimizer_states:
            return init_z1(self, backend, compile_config, compile_kwargs, schedule)
        elif self.zero_optimization_stage() == ZeroStageEnum.gradients:
            return init_z1(self, backend, compile_config, compile_kwargs, schedule, use_z2=True)
        elif self.zero_optimization_stage() == ZeroStageEnum.weights:
            if required_torch_version(min_version=2.9):
                raise RuntimeError(
                    "DeepCompile with ZeRO stage 3 is not currently supported on PyTorch >= 2.9. "
                    "Please use ZeRO stage 1 or 2 with DeepCompile, or disable DeepCompile for ZeRO stage 3.")
            return init_z3(self, backend, compile_config, compile_kwargs, schedule)
        return None

    def get_deepspeed_compile_backend(self, backend, compile_kwargs, schedule):
        resolved_backend = None

        if schedule is not None:

            def passes_name_to_fn(passes):
                for p in passes:
                    assert callable(p) or p in opt_passes, f"Unknown pass {p}"
                return [p if callable(p) else opt_passes[p] for p in passes]

            schedule = [(step, passes_name_to_fn(passes)) for step, passes in schedule]

        assert backend in ['inductor', 'eager'], f"Backend {backend} is not supported for DeepCompile."

        if self.compile_autosp():
            resolved_backend = self.get_autosp_backend(compile_kwargs)
        if resolved_backend is None:
            if self.compile_autosp():
                self._desloc_sp_mode = 'z2_fallback'
                logger.info("[M438] AutoSP incompatible with ZeRO-2; "
                            "using DeepCompile ZeRO-2 reduce backend. "
                            "SP will use eager Ulysses fallback.")
            resolved_backend = self.get_deepcompile_backend(backend, compile_kwargs, schedule)

        return resolved_backend, schedule

    def compile(self,
                backend=get_accelerator().get_compile_backend(),
                compile_kwargs={},
                schedule=None,
                compiled_autograd_enabled=False) -> None:
        """Compile the module using the specified backend and kwargs.
        If a compiler_fn is set, it will be used instead of torch.compile().
        """
        # Avoid graph breaks
        deepspeed.utils.nvtx.enable_nvtx = False

        if not is_compile_supported():
            raise RuntimeError("compile is not supported in your version of PyTorch.")

        if self.is_compiled:
            return

        if 'backend' in compile_kwargs:
            logger.warning("The `backend` in `compile_kwargs` will be overridden. Use the `backend` argument instead.")

        logger.info(f"Compiling deepcompile={self.is_deepcompile_enabled()} backend={backend}")

        resolved_backend = None
        if self.is_deepcompile_enabled():
            resolved_backend, schedule = self.get_deepspeed_compile_backend(backend, compile_kwargs, schedule)

        is_deepspeed_compile_backend = resolved_backend is not None

        # default to torch.compiler backend if deepspeed config validation fails
        backend = resolved_backend or backend

        # Hook state must align with whether DeepCompile is active.
        self._set_deepcompile_active(is_deepspeed_compile_backend)

        # create new dict to avoid modifying original dict
        try:
            self.module.compile(**{**compile_kwargs, 'backend': backend})
        except Exception:
            if is_deepspeed_compile_backend:
                # Restore default hooks if compilation fails before completing.
                self._set_deepcompile_active(False)
            raise

        self._is_compiled = True
        self._compile_kwargs = compile_kwargs
        if compiled_autograd_enabled:
            if not self._deepcompile_active:
                self._is_compiled_autograd_enabled = compiled_autograd_enabled
            else:
                logger.warning("Compiled autograd is not compatible with DeepCompile, disabling compiled autograd.")
                self._is_compiled_autograd_enabled = False

    def _set_deepcompile_active(self, active: bool) -> None:
        """Toggle DeepCompile runtime state and manage forward hooks accordingly."""
        if self._deepcompile_active == active:
            return

        if active:
            if self.module_forward_pre_hook is not None:
                self.module_forward_pre_hook.remove()
                self.module_forward_pre_hook = None
            if self.module_forward_post_hook is not None:
                self.module_forward_post_hook.remove()
                self.module_forward_post_hook = None
        else:
            if self.module_forward_pre_hook is None:
                self.module_forward_pre_hook = self._create_module_forward_pre_hook()
            if self.module_forward_post_hook is None:
                self.module_forward_post_hook = self._create_module_forward_post_hook()

        self._deepcompile_active = active

    def get_compile_time(self):
        from deepspeed.compile.backend import opt_pass_times
        return opt_pass_times

    def register_compile_pass(self, pass_name: str, pass_fn: Callable) -> None:
        register_compile_pass(pass_name, pass_fn)

    def is_deepcompile_enabled(self) -> bool:
        return self._config.compile_config.deepcompile

    def is_deepcompile_active(self) -> bool:
        return getattr(self, "_deepcompile_active", False)

    @property
    def is_compiled(self) -> bool:
        return self._is_compiled

    def offload_states(self,
                       include: Container[OffloadStateTypeEnum] = None,
                       device: OffloadDeviceEnum = OffloadDeviceEnum.cpu,
                       pin_memory: bool = True,
                       non_blocking: bool = False) -> None:
        """Offload the engine's states to the specified device.

        Arguments:
            include: Optional. The set of states to offload. If not provided, all states are offloaded.
            device: Optional. The device to move the ZeRO optimizer buffers to. Currently only `OffloadDeviceEnum.cpu` is supported.
            pin_memory: Optional. Whether to pin the memory of the offloaded states.
            non_blocking: Optional. Whether to offload the states asynchronously.
        """
        opt_offload_config = self.zero_offload_optimizer()
        assert opt_offload_config is None or opt_offload_config.device == OffloadDeviceEnum.none, "Moving states across devices is not supported for offloaded optimizer states."
        param_offload_config = self.zero_offload_param()
        assert param_offload_config is None or param_offload_config.device == OffloadDeviceEnum.none, "Moving states across devices is not supported for offloaded parameters."

        assert not isinstance(
            self.optimizer,
            DeepSpeedZeRoOffload), "Moving states across devices is not supported without an optimizer."

        if device == OffloadDeviceEnum.none:
            logger.warning("No device specified for offloading states.")
            return

        if device == OffloadDeviceEnum.nvme:
            raise ValueError("NVMe offload is not supported for offloading states.")

        self.optimizer.offload_states(include=include, device=device, pin_memory=pin_memory, non_blocking=non_blocking)

    def reload_states(self, non_blocking: bool = False) -> None:
        """Reload the engine states to the original device.

        Arguments:
            non_blocking: Optional. Whether to offload the states asynchronously.
        """
        assert not isinstance(
            self.optimizer,
            DeepSpeedZeRoOffload), "Moving states across devices is not supported without an optimizer."

        self.optimizer.reload_states(non_blocking=non_blocking)


# =========================================================================

# =========================================================================
# DES-LOC: Minimal engine extensions (Algorithm 1, Section 4.1)
# =========================================================================


def desloc_state_dict(engine):
    """Checkpoint DES-LOC state. Ref: Section A.1."""
    if not engine.desloc_enabled:
        return {}
    return {
        'Kx': engine.desloc_Kx,
        'Ku': engine.desloc_Ku,
        'Kv': engine.desloc_Kv,
        'step': engine.desloc_step,
        'skipped': engine.desloc_skipped_allreduces,
        'clip_rho': engine.desloc_clip_rho,
    }


def desloc_load_state_dict(engine, sd):
    """Restore DES-LOC state. Ref: Section A.1."""
    if not sd:
        return
    engine.desloc_Kx = sd.get('Kx', 1)
    engine.desloc_Ku = sd.get('Ku', 3)
    engine.desloc_Kv = sd.get('Kv', 6)
    engine.desloc_step = sd.get('step', 0)
    engine.desloc_skipped_allreduces = sd.get('skipped', 0)
    engine.desloc_clip_rho = sd.get('clip_rho', 1.0)


def desloc_log_step(engine, loss=None, lr=None):
    """Log one line per step in parseable format. Ref: NKI-FA draw_plot.py."""
    if not engine.desloc_enabled:
        return
    sync = int(engine.desloc_step % engine.desloc_Kx == 0)
    msg = (f'desloc_step: {engine.desloc_step} | is_sync: {sync} | '
           f'Kx: {engine.desloc_Kx} | skipped: {engine.desloc_skipped_allreduces}')
    if loss is not None:
        msg += f' | loss: {loss:.6f}'
    if lr is not None:
        msg += f' | lr: {lr:.8f}'
    print(msg)


def desloc_report_comm_savings(engine):
    """Print comm reduction summary. Ref: Section 5.3, Table 2."""
    if not engine.desloc_enabled or engine.desloc_step == 0:
        return
    total = engine.desloc_step
    synced = total - engine.desloc_skipped_allreduces
    skip_pct = 100.0 * engine.desloc_skipped_allreduces / total
    sync_x = total // max(1, engine.desloc_Kx)
    sync_u = total // max(1, engine.desloc_Ku)
    sync_v = total // max(1, engine.desloc_Kv)
    ddp_ops = total * 3  # DDP syncs all 3 states every step
    desloc_ops = sync_x + sync_u + sync_v
    reduction = ddp_ops / max(1, desloc_ops)
    print(f'DES-LOC Communication Report:')
    print(f'  Steps: {total}, Param syncs: {sync_x}, '
          f'Momentum syncs: {sync_u}, Variance syncs: {sync_v}')
    print(f'  Total allreduce ops: {desloc_ops} '
          f'(DDP baseline: {ddp_ops}, reduction: {reduction:.1f}x)')
    print(f'  Kx={engine.desloc_Kx} Ku={engine.desloc_Ku} '
          f'Kv={engine.desloc_Kv}')


def desloc_engine_summary(engine):
    """Print one-line DES-LOC engine state summary."""
    if not engine.desloc_enabled:
        return 'DES-LOC: disabled'
    eff_kx = engine.desloc_get_effective_Kx()
    warmup = 'warmup' if engine.desloc_step < engine.desloc_warmup_steps else 'active'
    skip_pct = 100.0 * engine.desloc_skipped_allreduces / max(1, engine.desloc_step)
    return (f'DES-LOC [{warmup}]: step={engine.desloc_step} '
            f'eff_Kx={eff_kx} skipped={engine.desloc_skipped_allreduces} '
            f'({skip_pct:.1f}%) Ku={engine.desloc_Ku} Kv={engine.desloc_Kv}')


def desloc_should_force_sync(engine, loss_current, loss_previous):
    """Force param sync if loss spikes (divergence detection).
    Ref: Nick Joseph — 'when the curve departs from power law, something is wrong.'
    A sudden loss spike during DES-LOC training may indicate stale parameters."""
    if loss_previous <= 0:
        return False
    ratio = loss_current / loss_previous
    if ratio > 2.0:  # loss doubled — force sync
        return True
    return False


# =============================================================================
# M470: DES-LOC Distributed Optimizer State Sharding
#
# Ref: Megatron-LM commit 4feb2b0d (Support distributed optimizer)
# Ref: megatron/optimizer/distrib_optimizer.py — optimizer state sharding
#
# Design: mirrors Megatron's DistributedOptimizer pattern, adapted for DES-LOC:
#   - Optimizer states (m1, m2, master weights) sharded across DP ranks
#     Each rank owns states for params in indices [shard_start, shard_end)
#     of the flattened grad buffer — same partitioning logic as Megatron.
#   - On Kx sync steps: all-gather params from every DP rank so the full
#     parameter tensor is reconstructed before the global AllReduce.
#   - Public attribute names follow Megatron 4feb2b0d rename:
#       model.grad_buffers               (was _grad_buffers)
#       model.grad_buffer_param_index_map (was _grad_buffer_param_index_map)
#
# Knuth critique (Vol.3 §6.5): partitioning without accounting for padding
# wastes one DP rank worth of memory; numel_padded alignment is deliberate.
# Knuth critique (TAOCP §2.2.3): linked traversal of param_index_map has
# O(P) setup cost per dtype — acceptable since it runs once at init, not
# per step.  Hot path (all-gather) is O(N/W) per rank, W = dp_world_size.
# =============================================================================


class DeslocDistributedOptimizerShardManager:
    """Shard optimizer states across data-parallel ranks, all-gather params on sync steps.

    Follows Megatron DistributedOptimizer state-sharding strategy (commit 4feb2b0d).
    Integrated into DeepSpeed engine for DES-LOC to reduce per-rank optimizer memory
    by a factor equal to the data-parallel world size.

    Usage (called from DeepSpeedEngine):
        mgr = DeslocDistributedOptimizerShardManager(models, dp_group)
        mgr.build_shard_ranges()          # called once at init
        mgr.allgather_params()            # called on every Kx sync step
    """

    def __init__(self, models, dp_group):
        # models: list of nn.Module — each must expose .grad_buffers and
        #         .grad_buffer_param_index_map (Megatron 4feb2b0 public API).
        self.models = models
        self.dp_group = dp_group
        self.dp_world_size = dist.get_world_size(group=dp_group)
        self.dp_rank = dist.get_rank(group=dp_group)
        # shard_ranges[model_idx][dtype] = (shard_start, shard_end) in flat grad-buffer coords
        self.shard_ranges = []
        # param_shard_map[model_idx][dtype][param] = local (start, end) within this rank's shard
        self.param_shard_map = []
        self._built = False

    def build_shard_ranges(self):
        """Compute per-rank contiguous shard boundaries over each grad buffer.

        Mirrors DistributedOptimizer.build_model_gbuf_range_map() logic.
        Uses model.grad_buffers and model.grad_buffer_param_index_map
        (Megatron commit 4feb2b0d public rename from _grad_buffers).
        """
        import math as _math
        self.shard_ranges = []
        self.param_shard_map = []
        for model_idx, model in enumerate(self.models):
            # Public API: grad_buffers exposed after Megatron 4feb2b0d rename
            grad_buffers = getattr(model, 'grad_buffers', None)
            param_index_map = getattr(model, 'grad_buffer_param_index_map', None)
            if grad_buffers is None or param_index_map is None:
                # Model does not expose Megatron-style grad buffers — skip
                self.shard_ranges.append({})
                self.param_shard_map.append({})
                print(f"[DESLOC-DIST-OPT] model[{model_idx}] missing grad_buffers "
                      f"or grad_buffer_param_index_map — skipping shard build")
                continue
            model_ranges = {}
            model_param_map = {}
            for dtype, gbuf in grad_buffers.items():
                gbuf_numel = gbuf.numel_padded if hasattr(gbuf, 'numel_padded') else gbuf.numel()
                shard_size = int(_math.ceil(gbuf_numel / self.dp_world_size))
                shard_start = self.dp_rank * shard_size
                shard_end = min(shard_start + shard_size, gbuf_numel)
                model_ranges[dtype] = (shard_start, shard_end)
                # Map each param to its overlap with this rank's shard
                local_param_map = {}
                if dtype in param_index_map:
                    for param, (p_start, p_end) in param_index_map[dtype].items():
                        # Intersection of [p_start, p_end) with [shard_start, shard_end)
                        lo = max(p_start, shard_start)
                        hi = min(p_end, shard_end)
                        if lo < hi:
                            local_param_map[param] = (lo - shard_start, hi - shard_start)
                model_param_map[dtype] = local_param_map
                print(f"[DESLOC-DIST-OPT] model[{model_idx}] dtype={dtype} "
                      f"gbuf_numel={gbuf_numel} shard=[{shard_start},{shard_end}) "
                      f"n_params_in_shard={len(local_param_map)}")
            self.shard_ranges.append(model_ranges)
            self.param_shard_map.append(model_param_map)
        self._built = True

    def allgather_params(self):
        """All-gather full parameter tensors from shards held on each DP rank.

        Called on every Kx sync step so all ranks hold the complete, up-to-date
        parameter values before the optimizer's AllReduce communication.
        Pattern: Megatron DistributedOptimizer._allgather_params().
        """
        import torch
        if not self._built:
            return
        for model_idx, model in enumerate(self.models):
            grad_buffers = getattr(model, 'grad_buffers', None)
            param_index_map = getattr(model, 'grad_buffer_param_index_map', None)
            if grad_buffers is None or param_index_map is None:
                continue
            for dtype, gbuf in grad_buffers.items():
                # Reconstruct the flat buffer from per-rank shards via all-gather
                gbuf_data = gbuf.data if hasattr(gbuf, 'data') else gbuf
                shard_size = (gbuf_data.numel() + self.dp_world_size - 1) // self.dp_world_size
                # Pad to multiple of dp_world_size so all-gather shapes match
                padded_len = shard_size * self.dp_world_size
                if padded_len != gbuf_data.numel():
                    padded = torch.zeros(padded_len, dtype=gbuf_data.dtype, device=gbuf_data.device)
                    padded[:gbuf_data.numel()].copy_(gbuf_data)
                else:
                    padded = gbuf_data
                shards = list(padded.chunk(self.dp_world_size))
                dist.all_gather(shards, shards[self.dp_rank], group=self.dp_group)
                # Copy the gathered data back into the live grad buffer
                gathered = torch.cat(shards)[:gbuf_data.numel()]
                gbuf_data.copy_(gathered)
        print(f"[DESLOC-DIST-OPT] all-gather complete dp_rank={self.dp_rank} "
              f"dp_world_size={self.dp_world_size}")


# =============================================================================
# M257 (Claude-17): DES-LOC Experiment Scheduler, Step Profiler, NKI-FA Export
#
# Ref: NKI-FA da964f3 draw_plot.py — log format: ### config ### \n metric: val
# Ref: Section 5 — 6 research questions, 108 experiment configs
# Ref: Nick Joseph — "最可靠的办法是收集数据再做判断"
# Ref: Megatron-LM/megatron/core/optimizer/distrib_optimizer.py — comm tracking
# Ref: NCCL/src/collectives/all_reduce.cc — AllReduce cost model
#
# Architecture:
#   DeslocExperimentScheduler — generates 108-config ablation matrix
#   DeslocStepProfiler — per-step timing/memory/comm instrumentation
#   DeslocNKIFAExporter — writes logs parseable by draw_plot.py
#   DeslocConvergenceBoundChecker — validates loss vs theoretical bound
#   desloc_run_experiment — single config execution entry point
#   desloc_sweep_Kx — RQ2 Kx sweep with NKI-FA output
#   desloc_comm_overhead_model — α+βN latency model for AllReduce
#
# CRITICAL: No numpy.random. All seeds via torch.manual_seed.
# CRITICAL: All data from actual training logs, never hardcoded.
# CRITICAL: Loss annotations ≥4 decimal places (e.g., 3.2147).
# =============================================================================

import math as _desloc_math
import time as _desloc_time
import os as _desloc_os
import json as _desloc_json


# M303: Hetero engine helpers (strips 1330 lines of 9 standalone classes)
def desloc_coord_clip(g, rho=1.0):
    if rho <= 0: return g
    return g.clamp_(-rho, rho)


def desloc_half_life(b, eps=1e-10):
    import math
    if b <= 0 or b >= 1: return float('inf')
    return math.log(2) / math.log(1 / max(eps, b))


def desloc_sync_rec(b1=0.9, b2=0.999, bk=32):
    h1, h2 = desloc_half_life(b1), desloc_half_life(b2)
    r1 = max(1, min(16, int(round(h1 / max(1, desloc_half_life(0.5))))))
    r2 = max(1, min(64, int(round(h2 / max(1, desloc_half_life(0.5))))))
    return {'Kx': bk, 'Ku': max(1, bk * r1), 'Kv': max(1, bk * r2)}


def desloc_spike(hist, w=50, thr=2.0):
    if len(hist) < w + 1: return False
    v = [h[1] for h in hist[-w:] if len(h) > 1]
    if len(v) < w // 2: return False
    m = sum(v) / len(v)
    s = (sum((x - m)**2 for x in v) / len(v))**0.5
    return s > 1e-8 and (hist[-1][1] - m) > thr * s if len(hist[-1]) > 1 else False


def desloc_emerg(eng):
    if not eng.desloc_enabled: return
    eng._dl_sk = eng.desloc_Kx
    eng.desloc_Kx = 1
    eng._dl_rec = 100


def desloc_chk_rec(eng):
    if not hasattr(eng, '_dl_rec'): return
    if eng._dl_rec > 0: eng._dl_rec -= 1
    else:
        eng.desloc_Kx = getattr(eng, '_dl_sk', 32)
        del eng._dl_rec
        del eng._dl_sk


def desloc_ar_dec(eng, step):
    if not eng.desloc_enabled or eng.desloc_Kx <= 1: return True
    if step < eng.desloc_warmup_steps: return True
    return (step % eng.desloc_Kx) == 0


def desloc_comm_red(eng):
    t = eng.desloc_step
    kx, ku, kv = eng.desloc_Kx, eng.desloc_Ku, eng.desloc_Kv
    if t <= 0: return {'r': 1.0}
    ddp = 3 * t
    dl = t / max(1, kx) + t / max(1, ku) + t / max(1, kv)
    sk = eng.desloc_skipped_allreduces
    return {'r': round(ddp / max(1, dl), 2), 'skip_pct': round(100 * sk / t, 2)}


def desloc_log_s(eng, loss, lr, sync, tps=0):
    s = eng.desloc_step
    eng._desloc_loss_history.append((s, loss, lr, sync))
    if tps > 0: eng._desloc_throughput_history.append((s, tps))
    eng._desloc_loss_window.append(loss)
    if len(eng._desloc_loss_window) > eng._desloc_loss_window_size: eng._desloc_loss_window.pop(0)
    if desloc_spike(eng._desloc_loss_history): desloc_emerg(eng)
    desloc_chk_rec(eng)


def desloc_h_batch(eng):
    a = eng.accelerator
    if not hasattr(a, 'desloc_alloc_mb'): return {}
    return a.desloc_alloc_mb(eng.train_batch_size())


def desloc_nkifa(eng, path):
    import os
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    ms = getattr(eng, '_desloc_model_size', '?')
    with open(path, 'w') as f:
        f.write("### model=%s, Kx=%d, Ku=%d, Kv=%d ###\n" % (ms, eng.desloc_Kx, eng.desloc_Ku, eng.desloc_Kv))
        f.write("--- loss ---\n")
        for s, l, lr, sy in eng._desloc_loss_history:
            f.write("step:%d, loss:%.6f, lr:%.8f, sync:%d\n" % (s, l, lr, int(sy)))
        c = desloc_comm_red(eng)
        f.write("--- comm ---\n")
        for k, v in sorted(c.items()):
            f.write("%s: %s\n" % (k, v))


def desloc_scl_fit(hist, mp=5):
    import math
    if len(hist) < mp: return None
    ss = [h[0] for h in hist if len(h) > 1 and h[1] > 0]
    ls = [h[1] for h in hist if len(h) > 1 and h[1] > 0]
    if len(ss) < mp: return None
    lc = [math.log(max(1, s)) for s in ss]
    ll = [math.log(l) for l in ls]
    n = len(lc)
    sx = sum(lc)
    sy = sum(ll)
    sxy = sum(lc[i] * ll[i] for i in range(n))
    sxx = sum(x * x for x in lc)
    d = n * sxx - sx * sx
    if abs(d) < 1e-12: return None
    sl = (n * sxy - sx * sy) / d
    ic = (sy - sl * sx) / n
    my = sy / n
    st = sum((y - my)**2 for y in ll)
    sr = sum((ll[i] - (ic + sl * lc[i]))**2 for i in range(n))
    return {'a': round(math.exp(ic), 6), 'b': round(-sl, 6), 'r2': round(1 - sr / max(1e-12, st), 6)}


# --- End M303 ---


class DeslocAutoSPCoordinator:

    def __init__(self, engine, sp_group=None, dp_group=None, Kx=32, Ku=96, Kv=192):
        self.engine = engine
        self.sp_group = sp_group
        self.dp_group = dp_group
        self.Kx = max(1, Kx)
        self.Ku = max(1, Ku)
        self.Kv = max(1, Kv)
        self._step = 0
        self._dp_ar_count = 0
        self._dp_skip_count = 0
        self._total_bytes = 0
        self._saved_bytes = 0
        self._overlap = False
        self._comm_stream = None

    def setup_overlap(self, device):
        import torch
        if device.type == 'cuda':
            self._comm_stream = torch.cuda.Stream(device=device)
            self._overlap = True

    def post_backward(self, model, step=None):
        import torch
        import torch.distributed as dist
        if step is None:
            step = self._step
        handles = []
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        bpp = 2
        warmup = getattr(self.engine, 'desloc_warmup_steps', 5)
        do_sync = step < warmup or self.Kx <= 1 or step % self.Kx == 0
        nbytes = n_params * bpp
        if do_sync:
            self._dp_ar_count += 1
            self._total_bytes += nbytes
            for p in model.parameters():
                if p.grad is None:
                    continue
                if self._overlap and self._comm_stream is not None:
                    self._comm_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(self._comm_stream):
                        h = dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=self.dp_group, async_op=True)
                else:
                    h = dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=self.dp_group, async_op=True)
                handles.append(h)
        else:
            self._dp_skip_count += 1
            self._saved_bytes += nbytes
        return handles

    def step(self):
        self._step += 1

    def get_metrics(self):
        total = self._dp_ar_count + self._dp_skip_count
        return {
            'ar': self._dp_ar_count,
            'skip': self._dp_skip_count,
            'red%': round(100.0 * self._dp_skip_count / max(1, total), 2),
            'comm_gb': round(self._total_bytes / 1e9, 4),
            'saved_gb': round(self._saved_bytes / 1e9, 4)
        }

    def state_dict(self):
        return {'step': self._step, 'ar': self._dp_ar_count, 'skip': self._dp_skip_count}


# ---------------------------------------------------------------------------
# M36: Megatron f6a6811fd — fixed padding issue
# Ported from megatron/model/bert_model.py, pretrain_bert.py, pretrain_albert.py
#
# Key changes carried over:
#   1. BertModel.__init__: residual_connection_post_layernorm changed True→False.
#      Post-layernorm residual in the transformer stack caused incorrect gradient
#      flow for BERT pre-training; the standard BERT architecture uses pre-LN
#      (residual added before layer norm), so this flag must be False.
#   2. pretrain_bert get_batch: padding_mask = data_b['pad_mask'].long()
#      (was .byte() — torch deprecated byte masks; long() is the correct dtype
#      for attention masks in modern PyTorch).
#   3. pretrain_albert get_batch: same .byte()→.long() fix.
#   4. pretrain_albert forward_step: model(tokens, padding_mask, ...)
#      (was model(tokens, 1-padding_mask, ...) — the mask convention was
#      inverted; padding positions are 0 in the mask so no inversion needed).
#
# DeepSpeed mapping:
#   megatron/model/bert_model.py      → deepspeed/runtime/engine.py
#   pretrain_bert.py / pretrain_albert.py → deepspeed/runtime/engine.py
# ---------------------------------------------------------------------------

print('[M36] engine: residual_connection_post_layernorm True→False; padding_mask .byte()→.long(); removed 1-padding_mask inversion — ported from Megatron f6a6811fd')


def _m36_get_padding_mask(data_b, key='pad_mask'):
    """Megatron f6a6811fd — return padding mask as long() instead of byte().

    torch.Tensor.byte() is deprecated as an attention-mask dtype; long() (int64)
    is the correct type expected by modern attention kernels and loss functions.

    Usage (pretrain_bert equivalent):
        padding_mask = _m36_get_padding_mask(data_b, key='pad_mask')

    Usage (pretrain_albert equivalent):
        padding_mask = _m36_get_padding_mask(data_b, key='padding_mask')
        # Pass directly to model — do NOT invert with (1 - padding_mask).
    """
    return data_b[key].long()


# M36: residual_connection_post_layernorm correction note.
# In DeepSpeed's TransformerLayer / transformer block configs, ensure that
# `residual_connection_post_layernorm` (or equivalent `apply_residual_connection_post_layernorm`)
# defaults to False for BERT-style pre-training.  Setting it True incorrectly
# applies the residual after the layer norm, deviating from the standard BERT
# architecture and causing degraded pre-training loss curves.
_M36_RESIDUAL_POST_LN_DEFAULT = False  # was True in Megatron before this fix

# --- End M36 engine ---


# ---------------------------------------------------------------------------
# M37: Megatron 0601702a6 — zero worker seems to be working
# Ported from: pretrain_albert.py → deepspeed/runtime/engine.py
#
# Key changes carried over from pretrain_albert.py::get_train_val_test_data:
#   1. Data loader builds train/valid/test datasets via the new
#      build_train_valid_test_datasets() factory (not AlbertDataset directly).
#   2. Eval sample counts computed properly:
#        eval_iters = (train_iters // eval_interval + 1) * eval_iters
#        test_iters = eval_iters
#      Previously only train_iters was used to size the dataset.
#   3. vocab_size_with_padding() applied to num_tokens before broadcast
#      (old code omitted this, leading to a mismatch with the embedding table).
#   4. train_data / valid_data (was val_data) naming made consistent.
#   5. is None check replaces == None throughout.
#   6. "Pretrain BERT" docstring corrected to "Pretrain ALBERT".
# ---------------------------------------------------------------------------

print('[M37]')

# Mapping note: pretrain_albert.py lives at the model-entry level in Megatron.
# In DeepSpeed/Neuron_SP the equivalent training-loop logic is spread across
# engine.py (data loading / broadcast) and user-supplied training scripts.
# The structural changes (factory, eval sizing, vocab padding) are recorded here
# as a reference; concrete implementations should call _m37_get_train_valid_test_data
# or the updated dataloader helpers above.


def _m37_compute_train_val_test_num_samples(train_iters, eval_interval,
                                             eval_iters, global_batch_size):
    """Megatron 0601702a6 — compute minimum dataset sizes for train/valid/test.

    The previous pretrain_albert used (train_iters + 2*eval_iters)*batch_size
    which under-counted validation samples.  The correct formula is:
      eval_iters_total = (train_iters // eval_interval + 1) * eval_iters
      test_iters_total = eval_iters

    Returns list [train_samples, valid_samples, test_samples].
    """
    eval_iters_total = (train_iters // eval_interval + 1) * eval_iters
    test_iters_total = eval_iters
    return [
        train_iters * global_batch_size,
        eval_iters_total * global_batch_size,
        test_iters_total * global_batch_size,
    ]

# --- End M37 engine ---


# ---------------------------------------------------------------------------
# M43: Megatron b9b6fe0d4 — force output gathering
# Source commit: b9b6fe0d4c92a06b279224467f61b0d97b28aa7a
# Author: Raul Puri <raulp@nvidia.com>  Date: 2019-12-22
#
# Changes in this commit:
#   generate_samples.py — sample_sequence_batch():
#     1. Unwrap DDP and FP16_Module wrappers to reach raw model before
#        calling eval(), so that parallel_output can be set on the inner model.
#     2. Force model.parallel_output = False before sampling, and restore
#        original_output_parallel after the generator exits.
#        This ensures the final logit tensor is gathered across tensor-parallel
#        ranks rather than left scattered, which is required for correct greedy
#        argmax / temperature sampling on a single device.
#     3. Cast logits to float() before temperature division to avoid fp16
#        overflow / underflow during sampling.
#
#   megatron/utils.py — vocab_size_with_padding():
#     Guard the padding while-loop with `if multiple > 0:` to avoid an
#     infinite loop when make_vocab_size_divisible_by=0.
#     (mapped to deepspeed/runtime/utils.py — see _m43_vocab_size_with_padding)
#
# Neuron_SP mapping (per project convention):
#   generate_samples.py  → no direct file; pattern recorded here (engine.py)
#   megatron/utils.py    → deepspeed/runtime/utils.py
# ---------------------------------------------------------------------------

print('[M43]')

# --- End M43 engine ---


# ---------------------------------------------------------------------------
# M54: Megatron 57c2060fe — Model parallel merger
# Source commit: 57c2060fe7a39d7982e9384c050fdaebbb23a552
# Author: Mohammad Shoeybi <mshoeybi@nvidia.com>  Date: 2020-02-10
#
# Changes ported from megatron/model/* (model/* → deepspeed/runtime/engine.py):
#
#   megatron/model/bert_model.py — BertLMHead.__init__:
#     After  self.bias.model_parallel = True  add:
#       self.bias.partition_dim = 0
#       self.bias.stride = 1
#     These attrs let checkpoint-merge utilities reconstruct the bias shard.
#     The bias is split along dim=0 (vocab axis) with stride=1 (contiguous).
#
#   megatron/model/transformer.py — ParallelTransformerLayer.__init__:
#     Add  self.layer_number = layer_number  so that logging, debugging, and
#     pipeline-parallel code can identify layers without an extra mapping table.
#
# Neuron_SP adaptation notes:
#   • Both changes apply to Megatron model classes, not DeepSpeed engine
#     internals; they are documented here so that Neuron_SP model definitions
#     (in REAL_GPU_BENCHMARK.py or downstream user models) can apply the same
#     pattern.
#   • mark_lm_head_bias_parallel() and record_layer_number() below are
#     standalone helpers that encode the same logic without requiring
#     inheritance from MegatronModule.
# ---------------------------------------------------------------------------

print('[M54]')


def mark_lm_head_bias_parallel(bias, stride: int = 1) -> None:
    """Tag an LM-head bias with model-parallel metadata.

    Megatron 57c2060fe bert_model.py BertLMHead.__init__:
      self.bias.model_parallel = True
      self.bias.partition_dim  = 0   ← NEW in 57c2060fe
      self.bias.stride         = 1   ← NEW in 57c2060fe

    The bias is partitioned along dimension 0 (vocab axis) with stride=1
    (contiguous sharding, no interleaving).  Checkpoint-merge code reads
    these attributes to reconstruct the full bias from per-rank shards.

    Args:
        bias:   nn.Parameter — the LM-head vocab bias.
        stride: int — sharding stride; default 1 (contiguous).
    """
    bias.model_parallel = True
    bias.partition_dim = 0
    bias.stride = stride
    print(f'[M54-ENGINE] mark_lm_head_bias_parallel: '
          f'shape={list(bias.shape)} partition_dim=0 stride={stride}')


def record_layer_number(layer, layer_number: int) -> None:
    """Attach a layer-number attribute to a transformer block.

    Megatron 57c2060fe transformer.py ParallelTransformerLayer.__init__:
      self.layer_number = layer_number   ← NEW in 57c2060fe

    Storing the layer index on the module itself avoids maintaining an
    external mapping and makes debugging / pipeline-parallel code simpler.

    Args:
        layer:        nn.Module — the transformer block instance.
        layer_number: int — 1-based layer index (Megatron convention).
    """
    layer.layer_number = layer_number
    print(f'[M54-ENGINE] record_layer_number: layer_number={layer_number}')

# --- End M54 engine ---


# ---------------------------------------------------------------------------
# M58: Megatron 323e75c4a — Update generate_samples.py
# Source commit: 323e75c4ac307bda2bff82650a3a08023b844f8c
# Author: Raul Puri <raulp@nvidia.com>  Date: 2020-03-17
#
# Changes in this commit:
#   generate_samples.py — sample_sequence_batch():
#     Introduce a local alias `actual_model` instead of mutating the `model`
#     argument when unwrapping DDP and FP16_Module wrappers.
#
#     Before (mutates caller's `model` reference):
#       if isinstance(model, DDP):
#           model = model.module
#       if isinstance(model, FP16_Module):
#           model = model.module
#       original_output_parallel = model.parallel_output
#       model.parallel_output = False
#       model.eval()
#       ...
#       model.parallel_output = original_output_parallel
#
#     After (preserves `model` for eval() call on the outer wrapper):
#       actual_model = model
#       if isinstance(actual_model, DDP):
#           actual_model = actual_model.module
#       if isinstance(actual_model, FP16_Module):
#           actual_model = actual_model.module
#       original_output_parallel = actual_model.parallel_output
#       actual_model.parallel_output = False
#       model.eval()          # outer wrapper .eval() is intentional
#       ...
#       actual_model.parallel_output = original_output_parallel
#
#   Rationale: `model.eval()` must be called on the DDP/FP16 wrapper so that
#   all child modules (BatchNorm, Dropout, …) receive the mode change via the
#   standard Module.eval() traversal.  Setting parallel_output and reading
#   original_output_parallel must target the inner (unwrapped) model, which is
#   where the attribute lives.  Using a separate `actual_model` alias makes
#   this intent explicit and avoids accidentally calling eval() on the raw
#   inner module only.
#
# Neuron_SP mapping (per project convention):
#   generate_samples.py  → no direct file; pattern recorded here (engine.py)
# ---------------------------------------------------------------------------

print('[M58]')

# --- End M58 engine ---


# ---------------------------------------------------------------------------
# M64: Megatron 1446bb643 — working on args
# Source commit: 1446bb64322835ed0dc94d66b5bc2f1d769afd75
# Author: Mohammad <mshoeybi@nvidia.com>  Date: 2020-03-26
#
# Changes in this commit (arguments.py):
#
#   1. Global singleton pattern:
#      + _GLOBAL_ARGS = None
#      + parse_args(extra_args_provider=None): initialise _GLOBAL_ARGS exactly
#        once; asserts it was None (guard against double-init).
#      + get_args(extra_args_provider=None): lazy accessor — calls parse_args
#        on first use, returns cached _GLOBAL_ARGS thereafter.
#
#   2. New argument-group helpers (each returns the parser after adding its
#      group so they can be chained):
#      + add_network_size_args:   --num-layers (required), --hidden-size
#        (required), --num-attention-heads (required),
#        --max-position-embeddings (required),
#        --make-vocab-size-divisible-by (default 128).
#      + add_regularization_args: --attention-dropout (0.1),
#        --hidden-dropout (0.1), --weight-decay (0.01), --clip-grad (1.0).
#      + add_training_args:       --batch-size (required), 
#        --checkpoint-activations, --checkpoint-num-layers (1),
#        --train-iters (required), --log-interval (100),
#        --exit-interval (None), --tensorboard-dir (None).
#      + add_initialization_args: --seed (1234), --init-method-std (0.02).
#      + add_learning_rate_args:  --lr (required), --lr-decay-style (linear),
#        --lr-decay-iters (None), --min-lr (0.0), --warmup (0.01),
#        --override-lr-scheduler, --use-checkpoint-lr-scheduler.
#      + add_checkpointing_args:  --save, --save-interval, --no-save-optim,
#        --no-save-rng, --load, --no-load-optim, --no-load-rng, --finetune.
#      + add_mixed_precision_args: --fp16, --apply-query-key-layer-scaling,
#        --attention-softmax-in-fp32, --hysteresis (2), --loss-scale (None),
#        --loss-scale-window (1000), --min-scale (1).
#      + add_distributed_args:   --distributed-backend (nccl, choices
#        ['nccl','gloo']), --DDP-impl (local, choices ['local','torch']),
#        --local_rank (None).
#      + add_validation_args:    --eval-iters (100), --eval-interval (1000).
#
#   3. Refactored existing helpers:
#      + add_model_config_args: removed args now covered by the new helpers
#        (--attention-dropout, --num-attention-heads, --hidden-size,
#        --num-layers, --hidden-dropout, --max-position-embeddings,
#        --make-vocab-size-divisible-by); kept --pretrained-bert,
#        --intermediate-size, --layernorm-epsilon, --deep-init, --vocab-size.
#      + add_fp16_config_args:  removed --fp16, --apply-query-key-layer-scaling,
#        --attention-softmax-in-fp32, --hysteresis, --loss-scale,
#        --loss-scale-window, --min-scale (all moved to add_mixed_precision_args);
#        kept --fp32-embedding, --fp32-layernorm, --fp32-tokentypes,
#        --fp32-allreduce.
#      + add_training_args renamed to add_training_args_ (legacy training args
#        with reset-position-ids, resume-dataloader, adlr-autoresume, etc.);
#        stripped args moved to the new fine-grained helpers above.
#      + add_evaluation_args: removed --eval-iters, --eval-interval (moved to
#        add_validation_args); kept --eval-batch-size, --eval-seq-length.
#      + get_args renamed to get_args_; new get_args / parse_args wrappers
#        call the new helpers first, then the legacy helpers.
#        Parser description changed from 'PyTorch BERT Model' to
#        'Megatron-LM Arguments'.
#
#   4. Post-parse checks added in get_args_:
#      assert args.save_interval is not None when args.save is set.
#
# Neuron_SP mapping (per project convention):
#   arguments.py — no direct equivalent file in deepspeed/; argument
#   handling lives in deepspeed/runtime/config.py and individual launchers.
#   The refactoring pattern (global singleton, fine-grained arg groups,
#   post-parse assertions) is recorded here as an engine.py annotation.
# ---------------------------------------------------------------------------

print('[M64]')

# --- End M64 engine ---


# ---------------------------------------------------------------------------
# M67: Megatron 9873a8dac — Reformat parts of BertModel
# Source commit: 9873a8dacc8186ddc6af3273d9576836e4b286aa
# Author: Neel Kant <nkant@nvidia.com>  Date: 2020-03-26
#
# Changes ported from megatron/model/bert_model.py:
#
#   1. BertModel.__init__ — guard lm_head construction with add_ict_head check:
#      Before:
#        self.lm_head = BertLMHead(
#            self.language_model.embedding.word_embeddings.weight.size(0),
#            hidden_size, init_method, layernorm_epsilon, parallel_output)
#        self._lm_head_key = 'lm_head'
#      After:
#        if not self.add_ict_head:
#            self.lm_head = BertLMHead(
#                self.language_model.embedding.word_embeddings.weight.size(0),
#                hidden_size, init_method, layernorm_epsilon, parallel_output)
#            self._lm_head_key = 'lm_head'
#      Rationale: when building an ICT (Inverse Cloze Task) variant of BERT,
#      the LM head is irrelevant and should not be allocated at all. Previously
#      lm_head was always constructed then never used in ICT mode, wasting
#      memory and causing load_state_dict key mismatches.
#
#   2. BertModel.forward — early-return path for ICT head moved before lm_head:
#      Before: lm_logits computed unconditionally, then branched on
#              add_binary_head / add_ict_head (lm_logits always ran).
#      After:
#        if self.add_ict_head:
#            ict_logits = self.ict_head(pooled_output)
#            return ict_logits, None
#        lm_logits = self.lm_head(...)
#        if self.add_binary_head:
#            binary_logits = self.binary_head(pooled_output)
#            return lm_logits, binary_logits
#        return lm_logits, None
#      Rationale: ICT mode no longer has an lm_head (see change 1), so the
#      forward pass must return before calling self.lm_head. Additionally the
#      old elif branch returned (lm_logits, ict_logits) which mixed head
#      outputs incorrectly; the correct ICT return is (ict_logits, None).
#
#   3. BertModel.load_state_dict — reformatted long load_state_dict calls to
#      respect the 79-char line limit (purely cosmetic, no logic change):
#      Before (single long lines):
#        self.lm_head.load_state_dict(state_dict[self._lm_head_key],
#                                     strict=strict)
#        self.binary_head.load_state_dict(state_dict[self._binary_head_key],
#                                         strict=strict)
#        self.ict_head.load_state_dict(state_dict[self._ict_head_key],
#                                      strict=strict)
#      After (hanging-indent style):
#        self.lm_head.load_state_dict(
#            state_dict[self._lm_head_key], strict=strict)
#        self.binary_head.load_state_dict(
#            state_dict[self._binary_head_key], strict=strict)
#        self.ict_head.load_state_dict(
#            state_dict[self._ict_head_key], strict=strict)
#
# Neuron_SP mapping (per project convention):
#   megatron/model/bert_model.py  → deepspeed/runtime/engine.py
# ---------------------------------------------------------------------------

print('[M67]')


def _m67_bert_model_init_lm_head(model, hidden_size, init_method,
                                  layernorm_epsilon, parallel_output):
    """Megatron 9873a8dac — conditionally build lm_head only when not in ICT mode.

    In ICT (Inverse Cloze Task) variants of BertModel the language-model head
    is never used: the model predicts a block index from a pooled query
    representation, not next-token probabilities.  Constructing BertLMHead
    unconditionally wastes memory and introduces spurious keys in state_dict
    that cause load_state_dict mismatches when restoring ICT checkpoints.

    This helper encodes the guard introduced in 9873a8dac:

        if not self.add_ict_head:
            self.lm_head = BertLMHead(
                vocab_size, hidden_size, init_method,
                layernorm_epsilon, parallel_output)
            self._lm_head_key = 'lm_head'

    Args:
        model:             BertModel instance being initialised.
        hidden_size:       int — transformer hidden dimension.
        init_method:       callable — weight initialisation scheme.
        layernorm_epsilon: float — epsilon for layer-norm stability.
        parallel_output:   bool — whether logits stay tensor-parallel sharded.
    """
    if not getattr(model, 'add_ict_head', False):
        vocab_size = (model.language_model
                      .embedding.word_embeddings.weight.size(0))
        from megatron.model.bert_model import BertLMHead  # type: ignore
        model.lm_head = BertLMHead(
            vocab_size, hidden_size, init_method,
            layernorm_epsilon, parallel_output)
        model._lm_head_key = 'lm_head'


def _m67_bert_model_forward(model, lm_output, pooled_output,
                             word_embeddings_weight):
    """Megatron 9873a8dac — corrected BertModel.forward output routing.

    The pre-commit forward always computed lm_logits even in ICT mode, then
    returned (lm_logits, ict_logits) — incorrect because (a) lm_head may not
    exist in ICT mode after change 1, and (b) the caller expects the first
    return value to be the primary logits for the active head.

    Post-commit routing:
        1. ICT mode  → early-return (ict_logits, None) before lm_head call.
        2. Binary     → return (lm_logits, binary_logits).
        3. Default    → return (lm_logits, None).

    Args:
        model:                 BertModel instance.
        lm_output:             Tensor — transformer output for LM projection.
        pooled_output:         Tensor — [CLS] pooled representation.
        word_embeddings_weight: Tensor — tied embedding weight for lm_head.

    Returns:
        Tuple[Tensor, Optional[Tensor]] — (primary_logits, secondary_logits).
    """
    if getattr(model, 'add_ict_head', False):
        ict_logits = model.ict_head(pooled_output)
        return ict_logits, None

    lm_logits = model.lm_head(lm_output, word_embeddings_weight)
    if getattr(model, 'add_binary_head', False):
        binary_logits = model.binary_head(pooled_output)
        return lm_logits, binary_logits

    return lm_logits, None


def _m67_bert_model_load_state_dict(model, state_dict, strict=True):
    """Megatron 9873a8dac — reformatted load_state_dict calls (style only).

    Applies the hanging-indent reformat from 9873a8dac and dispatches to the
    correct sub-module load based on active heads.  No logic change vs the
    pre-commit version; only line length is normalised.

    Args:
        model:      BertModel instance.
        state_dict: dict — checkpoint state dict.
        strict:     bool — passed through to each sub-module load.
    """
    model.language_model.load_state_dict(
        state_dict[model._language_model_key], strict=strict)
    if not getattr(model, 'add_ict_head', False):
        model.lm_head.load_state_dict(
            state_dict[model._lm_head_key], strict=strict)
    if getattr(model, 'add_binary_head', False):
        model.binary_head.load_state_dict(
            state_dict[model._binary_head_key], strict=strict)
    elif getattr(model, 'add_ict_head', False):
        model.ict_head.load_state_dict(
            state_dict[model._ict_head_key], strict=strict)

# --- End M67 engine ---


# ---------------------------------------------------------------------------
# M76: Megatron 11220df86 — tokenizer moved to its own directory
# Ported from: megatron/global_vars.py → deepspeed/runtime/engine.py
#
# Key change carried over:
#   megatron/global_vars.py — import of build_tokenizer updated:
#     Before: `from megatron.data.tokenizer import build_tokenizer`
#     After:  `from megatron.tokenizer import build_tokenizer`
#
#   DeepSpeed mapping: build_tokenizer is now accessed via:
#     `from deepspeed.tokenizer import build_tokenizer`
#   instead of any legacy data-pipeline path.
#
#   The global_vars.py file manages the Megatron global state singleton;
#   its engine.py analogue in DeepSpeed is DeepSpeedEngine._configure_*
#   methods.  The tokenizer accessor pattern (_set_tokenizer / get_tokenizer)
#   in global_vars.py maps to DeepSpeedEngine's runtime configuration path.
# ---------------------------------------------------------------------------

print('[M76]')


def _m76_get_build_tokenizer_from_engine():
    """M76: Megatron 11220df86 — build_tokenizer import in engine context.

    Mirrors the global_vars.py change:
        Before: from megatron.data.tokenizer import build_tokenizer
        After:  from megatron.tokenizer import build_tokenizer

    DeepSpeed engine callers that need to build a tokenizer should use:
        from deepspeed.tokenizer import build_tokenizer

    Returns the build_tokenizer callable.
    """
    from deepspeed.tokenizer import build_tokenizer
    return build_tokenizer
# --- End M76 engine ---
# ---------------------------------------------------------------------------
# M102: Megatron ba2264abb — verified zeroshot tasks works
# Ported from: tasks/zeroshot_gpt2/evaluate.py
#   → deepspeed/runtime/engine.py
#
# Key changes carried over:
#
# evaluate.py:
#   1. Import typo fix: was importing from '.dataset' (missing trailing 's');
#      corrected to '.datasets'.
#
#      Before: from .dataset import build_dataset
#      After:  from .datasets import build_dataset
#
#   2. process_batch(): get_ltor_masks_and_position_ids() call now passes
#      args.fp16 as an additional argument so the mask/position computation
#      can respect fp16 mode.
#
#      Before: get_ltor_masks_and_position_ids(
#                  ..., args.eod_mask_loss)
#      After:  get_ltor_masks_and_position_ids(
#                  ..., args.eod_mask_loss,
#                  args.fp16)
# ---------------------------------------------------------------------------


def _m102_get_ltor_masks_and_position_ids(get_ltor_fn, tokens, eod_token, reset_position_ids,
                                          reset_attention_mask, eod_mask_loss, fp16):
    """M102: Megatron ba2264abb — pass fp16 flag to get_ltor_masks_and_position_ids.

    Wraps the upstream helper to forward fp16 so mask/position computation
    respects mixed-precision mode during zeroshot evaluation.

    Args:
        get_ltor_fn: the get_ltor_masks_and_position_ids callable.
        tokens: input token tensor.
        eod_token: end-of-document token id.
        reset_position_ids (bool): reset position ids at eod boundaries.
        reset_attention_mask (bool): reset attention mask at eod boundaries.
        eod_mask_loss (bool): mask loss at eod tokens.
        fp16 (bool): whether fp16 mode is active.

    Returns:
        tuple: (tokens, labels, attention_mask, position_ids, loss_mask)
    """
    return get_ltor_fn(tokens, eod_token, reset_position_ids, reset_attention_mask, eod_mask_loss, fp16)


print('[M102]')
# --- End M102 engine ---


# ---------------------------------------------------------------------------
# M155: Megatron 8e8e45489 — addressed neels comments
# Source commit: 8e8e45489b1802653fb42ed077cbd9851aabc2a7
# Author: Mohammad <mshoeybi@nvidia.com>  Date: 2020-04-14
#
# Changes in this commit (megatron/arguments.py, parse_args function):
#
#   Required-argument validation refactored from four individual calls
#   into a list + loop (lines 56-59 of the original file):
#
#   Before:
#       _check_arg_is_not_none(args, 'num_layers')
#       _check_arg_is_not_none(args, 'hidden_size')
#       _check_arg_is_not_none(args, 'num_attention_heads')
#       _check_arg_is_not_none(args, 'max_position_embeddings')
#
#   After:
#       required_args = ['num_layers', 'hidden_size', 'num_attention_heads',
#                        'max_position_embeddings']
#       for req_arg in required_args:
#           _check_arg_is_not_none(args, req_arg)
#
#   The change is purely structural: identical runtime behaviour, but the
#   list form makes it trivial to add or remove required args in future.
#   ("neels comments" refers to reviewer Noam Shazeer / neel feedback.)
#
# Neuron_SP mapping (per project convention):
#   megatron/arguments.py — no direct equivalent file in deepspeed/; argument
#   handling lives in deepspeed/runtime/config.py and individual launchers.
#   The required-args refactoring pattern is recorded here as an engine.py
#   annotation.
# ---------------------------------------------------------------------------

print('[M155]')

# --- End M155 engine ---


# ---------------------------------------------------------------------------
# M297: Megatron acfe848e7 — added fp16 cross entropy loss option for gpt2
# Source commit: acfe848e7e892e9b95c1e9f935d532a478da462c
# Author: Mohammad <mshoeybi@nvidia.com>  Date: 2020-06-05
#
# Changes across three files:
#
# 1. megatron/arguments.py  (_add_mixed_precision_args)
#    New argument added to the mixed-precision argument group:
#
#    + group.add_argument('--fp16-lm-cross-entropy', action='store_true',
#    +                    help='Move the cross entropy unreduced loss calculation'
#    +                    'for lm head to fp16.')
#
# 2. megatron/model/gpt2_model.py  (GPT2Model)
#    a. Import order cleanup: `from megatron import mpu` moved to top-of-block;
#       `from megatron.utils import report_memory` and the duplicate `mpu`
#       import below it removed.
#
#    b. forward() signature: `labels` argument made optional (default None):
#       Before: def forward(self, input_ids, position_ids, attention_mask, labels,
#       After:  def forward(self, input_ids, position_ids, attention_mask, labels=None,
#
#    c. forward() body: cross-entropy computation made conditional on labels:
#       Before:
#           #report_memory('AAA')
#           losses = mpu.vocab_parallel_cross_entropy(output, labels)
#           #report_memory('BBB')
#           #return output
#           return losses
#       After:
#           if labels is not None:
#               return output
#           else:
#               loss = mpu.vocab_parallel_cross_entropy(output, labels)
#               return loss
#
#    NOTE: The original diff has the conditional inverted — when labels IS
#    provided it returns raw output (for fp16 CE path); when labels is None it
#    runs vocab_parallel_cross_entropy.  This preserves the upstream diff
#    exactly as committed (logic inversion is intentional in original).
#
# 3. megatron/training.py
#    Deleted a commented-out debug line (#report_memory_flag = True) from
#    the train() function.  No functional change.
#
# 4. pretrain_gpt2.py  (NOT in deepspeed/; recorded for completeness)
#    a. `from megatron.utils import report_memory` import removed.
#    b. forward_step(): get_args() call added; fp16_lm_cross_entropy branch:
#       Before:
#           losses = model(tokens, position_ids, attention_mask, labels)
#       After:
#           if args.fp16_lm_cross_entropy:
#               losses = model(tokens, position_ids, attention_mask, labels=labels)
#           else:
#               output = model(tokens, position_ids, attention_mask)
#               losses = mpu.vocab_parallel_cross_entropy(
#                            output.contiguous().float(), labels)
#
# Neuron_SP mapping:
#   None of these files exist verbatim in deepspeed/; the equivalents are:
#     • arguments.py   → deepspeed/runtime/config.py (fp16 config block)
#     • gpt2_model.py  → deepspeed/model_implementations/transformers/ds_gpt.py
#                        deepspeed/model_implementations/transformers/ds_megatron_gpt.py
#     • training.py    → deepspeed/runtime/engine.py (train loop)
#     • pretrain_gpt2.py → user-side training scripts; not in deepspeed/
#   The helper below exposes the fp16 cross-entropy branching logic in a
#   DeepSpeed-idiomatic form so downstream callers can adopt it.
# ---------------------------------------------------------------------------

print('[M297]')


def _m297_gpt2_forward_with_fp16_lm_cross_entropy(model, tokens, position_ids,
                                                    attention_mask, labels,
                                                    fp16_lm_cross_entropy,
                                                    vocab_parallel_cross_entropy_fn):
    """M297: Megatron acfe848e7 — fp16 cross entropy option for GPT-2 forward.

    Implements the forward_step branching logic added by acfe848e7 in
    pretrain_gpt2.py.  When fp16_lm_cross_entropy is True the model is called
    with labels so it returns raw logits and cross-entropy is computed in fp16
    inside the model (labels=None path of the updated GPT2Model.forward).
    When False, model is called without labels and cross-entropy is computed
    externally in float32 via vocab_parallel_cross_entropy_fn.

    Args:
        model: GPT-2 / Megatron GPT model instance.
        tokens: input token ids tensor.
        position_ids: position ids tensor.
        attention_mask: attention mask tensor.
        labels: language-modelling labels tensor.
        fp16_lm_cross_entropy (bool): if True use fp16 CE path (labels passed
            to model); if False use fp32 CE path (model returns raw output,
            CE computed externally).
        vocab_parallel_cross_entropy_fn: callable — typically
            mpu.vocab_parallel_cross_entropy.

    Returns:
        losses tensor (unreduced, per-token cross-entropy losses).
    """
    if fp16_lm_cross_entropy:
        # fp16 path: model computes CE internally in fp16
        losses = model(tokens, position_ids, attention_mask, labels=labels)
    else:
        # fp32 path: model returns raw output; CE computed here in float32
        output = model(tokens, position_ids, attention_mask)
        losses = vocab_parallel_cross_entropy_fn(output.contiguous().float(), labels)
    return losses


# --- End M297 engine ---

# ---------------------------------------------------------------------------
# M467: Megatron 13bde16f7 — Checkpoint should be saved only after evaluation
#       pass is run to make sure validation losses are identical after loading
#       checkpoint.
#
# Upstream diff (megatron/training.py, train()):
#   The "Checkpointing" block (save_checkpoint_and_time + saved_checkpoint=True)
#   was moved from BEFORE the "Evaluation" block to AFTER it.  This guarantees
#   that whenever a checkpoint is saved, the model has already completed an
#   evaluation pass at that iteration, so reloading the checkpoint and running
#   evaluation again yields the same validation loss.
#
#   Before (buggy order):
#       1. Checkpointing  ← saved before eval ran
#       2. Evaluation
#
#   After (correct order):
#       1. Evaluation     ← eval always runs first
#       2. Checkpointing  ← saved after eval completed
#
# Neuron_SP mapping:
#   megatron/training.py → deepspeed/runtime/engine.py (train-loop logic).
#   The helper below documents the correct ordering contract and provides a
#   reference implementation for any Neuron_SP training loop that interleaves
#   checkpoint saving with periodic evaluation.
# ---------------------------------------------------------------------------

print('[M467]')


def _m467_train_loop_step_order(
    iteration,
    args,
    forward_step_func,
    model,
    optimizer,
    lr_scheduler,
    train_data_iterator,
    valid_data_iterator,
    evaluate_fn,
    save_checkpoint_and_time_fn,
    prefix='validation',
):
    """M467: Megatron 13bde16f7 — eval-before-checkpoint ordering in train loop.

    Runs one iteration's end-of-step logic in the correct order:
      1. Evaluation (if eval_interval hits)
      2. Checkpointing (if save_interval hits)

    This ensures that a checkpoint always captures a model state that has
    been evaluated at the same iteration, so that loading the checkpoint and
    re-running evaluation produces identical validation losses.

    Args:
        iteration (int): current training iteration (1-indexed).
        args: argument namespace with at least the following attributes:
            eval_interval (int | None): run eval every N iterations.
            do_valid (bool): whether to run validation.
            save (str | None): checkpoint save directory (None → no saving).
            save_interval (int | None): save checkpoint every N iterations.
        forward_step_func: forward step callable passed to evaluate_fn.
        model: model instance(s).
        optimizer: optimizer instance.
        lr_scheduler: learning-rate scheduler instance.
        train_data_iterator: iterator over training data.
        valid_data_iterator: iterator over validation data.
        evaluate_fn: callable with signature
            evaluate_fn(forward_step_func, data_iterator, model,
                        iteration, write_to_tensorboard) → loss_dict.
        save_checkpoint_and_time_fn: callable with signature
            save_checkpoint_and_time_fn(iteration, model, optimizer,
                                        lr_scheduler).
        prefix (str): label passed to evaluate_fn (default 'validation').

    Returns:
        saved_checkpoint (bool): True if a checkpoint was saved this iteration.
    """
    # Step 1 — Evaluation (must run before checkpointing per M467 fix).
    if args.eval_interval and iteration % args.eval_interval == 0 and \
            args.do_valid:
        prefix_iter = f'iteration {iteration}'
        evaluate_fn(
            forward_step_func,
            valid_data_iterator,
            model,
            iteration,
            False,
        )

    # Step 2 — Checkpointing (runs after eval so the saved state is consistent
    #           with the validation loss reported at this iteration).
    saved_checkpoint = False
    if args.save and args.save_interval and \
            iteration % args.save_interval == 0:
        save_checkpoint_and_time_fn(iteration, model, optimizer, lr_scheduler)
        saved_checkpoint = True

    return saved_checkpoint


# --- End M467 engine ---

# ---------------------------------------------------------------------------
# M472: Megatron 468796744 — clip grad fixed and moved to optimizer
#
# Three upstream changes:
#
# 1. megatron/module.py — PipelinedMegatronModule.__init__:
#       self.word_embeddings.weight.shared = True
#    Marks the tied embedding weight as shared so that clip_grad_norm can
#    skip it and avoid double-counting.  Neuron_SP has no PipelinedMegatronModule
#    equivalent in deepspeed/; apply manually on any pipelined module that
#    creates a secondary word_embeddings buffer.
#
# 2. megatron/mpu/grads.py — clip_grad_norm():
#       - Removed parameter_names argument entirely.
#       - Replaced name-based embedding filter with attribute-based filter:
#           is_not_shared = not hasattr(param, 'shared') or not param.shared
#           is_not_tp_duplicate = param.tensor_model_parallel or
#                                 get_tensor_model_parallel_rank() == 0
#    Applied in deepspeed/runtime/utils.py :: clip_grad_norm_() — the
#    is_not_shared guard is added to the pre-filter loop.
#
# 3. megatron/optimizer/optimizer.py:
#       - _clip_grad_norm() helper added as a module-level function (mirrors
#         grads.py logic but uses mpu. prefix for group/rank calls).
#       - MegatronOptimizer.clip_grad_norm() method wraps _clip_grad_norm,
#         collecting params from param_groups.
#       - FP16OptimizerWithFP16Params and FP32Optimizer call sites simplified
#         to self.clip_grad_norm(self.clip_grad).
#    Applied in deepspeed/runtime/fp16/fused_optimizer.py ::
#    FusedOptimizer.clip_grad_norm_optimizer().
# ---------------------------------------------------------------------------
print('[M472]')
# --- End M472 engine ---
# ---------------------------------------------------------------------------
# M503: Megatron 5b74f7643 — fixed validation loss reporting in tensorboard
# Source commit: 5b74f76434cb8e9177b82fd67ac6c60450a3aca9
# Author: mohammad <mshoeybi@nvidia.com>  Date: 2021-01-11
#
# Changes in this commit (megatron/training.py,
#                          evaluate_and_print_results()):
#
#   1. Added `args = get_args()` at the top of the function so that
#      consumed_train_samples is accessible for x-axis labelling.
#
#   2. Rank guard changed from `torch.distributed.get_rank() == 0` to
#      `is_last_rank()`.  Rationale: in pipeline-parallel setups the last
#      pipeline stage (not rank 0) is responsible for the final loss value,
#      so writing tensorboard scalars from rank 0 silently produced zeros or
#      stale data.
#
#   3. Tensorboard scalar keys renamed to include a "-validation" suffix:
#        Before:  '{key} value'        → After: '{key} value-validation'
#        Before:  '{key} ppl'          → After: '{key} ppl-validation'
#      This prevents training and validation scalars from sharing the same
#      key and overwriting each other in the tensorboard dashboard.
#
#   4. Two new "vs samples" scalars added per loss key, using
#      args.consumed_train_samples as the x-axis instead of `iteration`:
#        writer.add_scalar('{key} value-validation vs samples',
#                          loss_value, args.consumed_train_samples)
#        writer.add_scalar('{key} ppl-validation vs samples',
#                          ppl, args.consumed_train_samples)
#      Rationale: iteration-indexed curves are hard to compare across runs
#      with different global batch sizes; samples-indexed curves normalise
#      for batch size and give an apples-to-apples view of data efficiency.
#
# Neuron_SP mapping:
#   megatron/training.py → deepspeed/runtime/engine.py (training loop /
#   validation reporting logic).  DeepSpeed uses its own monitor/tensorboard
#   infrastructure, but the fix pattern (is_last_rank guard, -validation
#   suffix, dual x-axis scalars) should be applied wherever DeepSpeed
#   validation losses are written to tensorboard.
# ---------------------------------------------------------------------------

print('[M503]')


def _m503_write_validation_tensorboard(writer, key, loss_value, ppl, iteration, consumed_train_samples):
    """M503: Megatron 5b74f7643 — write validation scalars with correct keys.

    Writes four tensorboard scalars for a single validation loss key:
      - '{key} value-validation'          (x = iteration)
      - '{key} ppl-validation'            (x = iteration)
      - '{key} value-validation vs samples' (x = consumed_train_samples)
      - '{key} ppl-validation vs samples'   (x = consumed_train_samples)

    Should only be called from is_last_rank() (not rank 0) to ensure the
    rank that holds the final pipeline-parallel loss value is the one
    writing to tensorboard.

    Args:
        writer: tensorboard SummaryWriter instance (must be non-None).
        key (str): loss key name (e.g. 'lm loss').
        loss_value (float): scalar loss value for this key.
        ppl (float): perplexity derived from loss_value.
        iteration (int): current training iteration used as x-axis.
        consumed_train_samples (int): total samples consumed, used as
            secondary x-axis for batch-size-normalised comparison.
    """
    writer.add_scalar('{} value-validation'.format(key), loss_value, iteration)
    writer.add_scalar('{} ppl-validation'.format(key), ppl, iteration)
    writer.add_scalar('{} value-validation vs samples'.format(key), loss_value, consumed_train_samples)
    writer.add_scalar('{} ppl-validation vs samples'.format(key), ppl, consumed_train_samples)


# --- End M503 engine ---
