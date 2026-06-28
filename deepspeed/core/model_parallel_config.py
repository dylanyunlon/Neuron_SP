# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Model parallelism configuration — base config for all core modules."""

from __future__ import annotations

import warnings  # From Megatron M2576: add missing warnings import
from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional

import torch

from deepspeed.core.desloc_config import DesLocConfig


@dataclass
class ModelParallelConfig:
    """Configuration for all forms of model parallelism.

    Mirrors Megatron's ModelParallelConfig but adds DES-LOC fields.
    Every module in deepspeed/core/ receives this config.
    """

    # --- Parallelism dimensions ---
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: Optional[int] = None
    sequence_parallel: bool = False
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1

    # --- Pipeline ---
    pipeline_model_parallel_comm_backend: Optional[Literal["nccl", "ucc"]] = None
    # Layer split across PP stages (for heterogeneous PP)
    pipeline_layer_split: Optional[List[int]] = None

    # --- Precision ---
    params_dtype: torch.dtype = torch.bfloat16
    fp32_residuals: bool = False

    # --- Initialization ---
    perform_initialization: bool = True
    use_cpu_initialization: bool = False

    # --- Gradient handling hooks (set by training loop) ---
    finalize_model_grads_func: Optional[Callable] = None
    grad_scale_func: Optional[Callable] = None
    no_sync_func: Optional[Callable] = None
    grad_sync_func: Optional[Callable] = None
    param_sync_func: Optional[Callable] = None

    # --- Communication overlap ---
    overlap_grad_reduce: bool = False
    overlap_param_gather: bool = False
    gradient_accumulation_fusion: bool = False
    tp_comm_overlap: bool = False

    # --- Activation checkpointing ---
    recompute_method: Optional[Literal["uniform", "block"]] = None
    recompute_granularity: Optional[Literal["full", "selective"]] = None
    recompute_num_layers: Optional[int] = None
    distribute_saved_activations: bool = False

    # --- Determinism ---
    deterministic_mode: bool = False

    # --- DES-LOC (heterogeneous training) ---
    desloc: Optional[DesLocConfig] = None

    # --- Timers ---
    timers: Optional[Callable] = None

    # --- NCCL flight recorder (From Megatron M3499) ---
    # Set these to enable NCCL flight recorder for debugging hangs/timeouts.
    # Priority: pre-existing env vars > these fields.
    flight_recorder_dump_path: Optional[str] = None
    """Path prefix for NCCL flight recorder dumps. If set, enables flight
    recorder and sets TORCH_FR_DUMP_TEMP_FILE / TORCH_NCCL_DEBUG_INFO_TEMP_FILE."""
    flight_recorder_trace_buffer_size: int = 36864
    """NCCL trace buffer size (TORCH_NCCL_TRACE_BUFFER_SIZE)."""
    flight_recorder_dump_on_timeout: bool = True
    """Dump flight recorder on NCCL timeout (TORCH_NCCL_DUMP_ON_TIMEOUT)."""
    flight_recorder_include_stack_trace: bool = True
    """Include stack traces in flight recorder (TORCH_INCLUDE_STACK_TRACE)."""
    flight_recorder_include_only_active: bool = False
    """Only include active ops in flight recorder (TORCH_INCLUDE_ONLY_ACTIVE)."""
    flight_recorder_extra_dump_on_exec: bool = False
    """Extra dump on exec in flight recorder (TORCH_NCCL_EXTRA_DUMP_ON_EXEC)."""

    @property
    def desloc_enabled(self) -> bool:
        return self.desloc is not None and self.desloc.enabled
