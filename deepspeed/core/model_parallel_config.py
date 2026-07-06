# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Model parallelism configuration — base config for all core modules."""

from __future__ import annotations

import warnings  # From Megatron M2576: add missing warnings import
from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional

try:
    import torch as _torch
    _DTYPE_BFLOAT16 = _torch.bfloat16
    _DTYPE_TYPE = _torch.dtype
except (ImportError, OSError):
    # Fallback when torch CUDA libraries are unavailable (dry-run / CI).
    _torch = None  # type: ignore[assignment]
    _DTYPE_BFLOAT16 = None  # type: ignore[assignment]
    _DTYPE_TYPE = type(None)

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
    params_dtype: "torch.dtype" = _DTYPE_BFLOAT16  # type: ignore[assignment]
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

    # ---------------------------------------------------------------------------
    # Pipeline-parallel communication fields (required by pipeline_parallel/)
    # ---------------------------------------------------------------------------

    # Dtype used for pipeline activation tensors (None → use params_dtype)
    pipeline_dtype: "Optional[torch.dtype]" = None

    # Model hidden dimension (needed by get_tensor_shapes)
    hidden_size: int = 4096

    # Enable/disable batched P2P (NCCL batch_isend_irecv) vs sequential
    batch_p2p_comm: bool = True
    batch_p2p_sync: bool = True

    # Ring-exchange P2P (alternative to batch_isend_irecv for some backends)
    use_ring_exchange_p2p: bool = False

    # Variable sequence lengths — enables dynamic shape negotiation in P2P
    variable_seq_lengths: bool = False

    # Overlap P2P comm with compute in VPP schedule
    overlap_p2p_comm: bool = False
    overlap_p2p_comm_warmup_flush: bool = False

    # Deallocate pipeline output tensors after sending (save activation memory)
    deallocate_pipeline_outputs: bool = False

    # Defer embedding weight-gradient compute to after pipeline cooldown
    defer_embedding_wgrad_compute: bool = False

    # MoE expert-parallel A2A overlap with compute
    overlap_moe_expert_parallel_comm: bool = False

    # Number of microbatches with partial activation checkpointing
    num_microbatches_with_partial_activation_checkpoints: Optional[int] = None

    # Microbatch group size per virtual pipeline stage (VPP tunable schedule)
    microbatch_group_size_per_vp_stage: int = 1

    # Per-token loss normalisation (instead of dividing by num_microbatches)
    calculate_per_token_loss: bool = False

    # MoE experts (used to gate MoE aux-loss scaling)
    num_moe_experts: Optional[int] = None

    # Multi-token prediction layers (MTP)
    mtp_num_layers: Optional[int] = None

    # MTP standalone mode — shape negotiation on standalone MTP stage (M3009)
    mtp_standalone: bool = False

    # CUDA graph implementation ("local" | None)
    cuda_graph_impl: Optional[str] = None

    # Autocast settings
    enable_autocast: bool = False
    autocast_dtype: "Optional[torch.dtype]" = None

    # Barrier before timing L1 timers
    barrier_with_L1_time: bool = True

    # Paged activation stashing for MoE (M4012)
    moe_paged_stash: bool = False

    # Hybrid context parallel
    hybrid_context_parallel: bool = False

    # Grad/MoE/MTP scale functions (set by training loop)
    moe_grad_scale_func: Optional[Callable] = None
    mtp_grad_scale_func: Optional[Callable] = None

    # Experimental attention variant (e.g. 'dsa')
    experimental_attention_variant: Optional[str] = None

    # Fine-grained activation offloading (M3018)
    fine_grained_activation_offloading: bool = False

    # ---------------------------------------------------------------------------
    # Heterogeneous pipeline (DES-LOC PP=5) — per-stage micro_batch_size
    # ---------------------------------------------------------------------------
    # When set, each pipeline stage i uses hetero_micro_batch_sizes[i] instead
    # of the global micro_batch_size.  This allows fast stages (H100) to process
    # larger micro-batches while slow stages (A6000) use smaller ones, reducing
    # the pipeline bubble imposed by the slowest stage.
    #
    # Example for PP=5 with H100 at stages 0,2,4 and A6000 at stages 1,3:
    #   hetero_micro_batch_sizes = [4, 2, 4, 2, 4]
    #
    # The P2P communicator handles the shape mismatch by enabling
    # variable_seq_lengths when this list is non-empty.
    hetero_micro_batch_sizes: Optional[List[int]] = None

    @property
    def desloc_enabled(self) -> bool:
        return self.desloc is not None and self.desloc.enabled

    @property
    def effective_pipeline_dtype(self) -> torch.dtype:
        """Return the dtype to use for pipeline activation tensors."""
        if self.pipeline_dtype is not None:
            return self.pipeline_dtype
        return self.params_dtype

    def get_stage_micro_batch_size(self, stage: int, default: int) -> int:
        """Return micro_batch_size for a given pipeline stage.

        Used by the 1F1B schedule to support heterogeneous pipelines (PP=5)
        where different stages run different micro-batch sizes.
        """
        if self.hetero_micro_batch_sizes is not None:
            if stage < len(self.hetero_micro_batch_sizes):
                return self.hetero_micro_batch_sizes[stage]
        return default
