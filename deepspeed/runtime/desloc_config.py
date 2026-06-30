"""
Configuration dataclasses and enums for the DES-LOC Heterogeneous Training Engine.

Extracted from deepspeed/runtime/desloc_engine.py (see docs/ISSUE5_architecture_refactor.md)
to isolate pure data/config types from the engine's runtime logic, mirroring the
Megatron-LM `megatron/training/training_config.py` pattern of keeping configuration
dataclasses in their own module separate from the training loop and engine code.

Contents:
    PartitionStrategy  - Enum of supported partition strategies (ZeRO-3 hetero / Pipeline 1F1B)
    TierClass          - Enum of GPU tier classifications (H100 / A6000 / RTX_PRO_6000_BW / UNKNOWN)
    TierSpec           - Dataclass describing a single discovered GPU tier
    PartitionPlan      - Dataclass describing the result of PartitionSolver
    TrainingConfig     - Full training configuration dataclass for the DES-LOC engine

These types have no dependency on deepspeed.runtime.desloc_engine itself, so they can be
imported independently (e.g. by tests, CLI tooling, or other engine modules) without pulling
in torch.distributed / CUDA-dependent engine machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CHECKPOINT_DIR = Path("checkpoints")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class PartitionStrategy(Enum):
    """Supported partition strategies for heterogeneous training."""
    ZERO3_HETERO = auto()   # ZeRO-3 + heterogeneous gradient accumulation
    PIPELINE_1F1B = auto()  # Pipeline parallelism with 1F1B schedule


class TierClass(Enum):
    """GPU tier classification based on SM version and memory."""
    H100 = "H100"
    A6000 = "A6000"
    RTX_PRO_6000_BW = "RTX_PRO_6000_BW"  # Blackwell SM12.0, 96GB
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class TierSpec:
    """
    Specification of a single GPU tier discovered at runtime.

    Attributes:
        device_index: CUDA device index.
        tier: Classification of the GPU tier.
        total_mem_gb: Total GPU memory in GB.
        free_mem_gb: Free GPU memory in GB at discovery time.
        sm_major: CUDA SM major version.
        sm_minor: CUDA SM minor version.
        bf16_tflops: BF16 theoretical peak TFLOPs.
        pcie_bw_gbs: PCIe bandwidth in GB/s.
        numa_node: NUMA node affinity (-1 if unknown).
        name: Human-readable GPU name.
    """
    device_index: int
    tier: TierClass
    total_mem_gb: float
    free_mem_gb: float
    sm_major: int
    sm_minor: int
    bf16_tflops: float
    pcie_bw_gbs: float
    numa_node: int
    name: str

    @property
    def device(self) -> torch.device:
        """Return the torch.device for this tier."""
        return torch.device(f"cuda:{self.device_index}")

    def __repr__(self) -> str:
        return (
            f"TierSpec(idx={self.device_index}, tier={self.tier.value}, "
            f"mem={self.total_mem_gb:.0f}GB, SM={self.sm_major}.{self.sm_minor}, "
            f"BF16={self.bf16_tflops}TFLOPS, name='{self.name}')"
        )


@dataclass
class PartitionPlan:
    """
    Result of PartitionSolver: describes how model layers are assigned to tiers.

    Attributes:
        strategy: The chosen partition strategy.
        tier_layer_map: Maps device_index -> list of layer indices assigned.
        grad_accum_steps: Per-device gradient accumulation steps dict.
        micro_batch_sizes: Per-device micro-batch sizes.
        estimated_throughput: Estimated tokens/s for this plan.
        notes: Human-readable notes about why this plan was chosen.
    """
    strategy: PartitionStrategy
    tier_layer_map: Dict[int, List[int]]
    grad_accum_steps: Dict[int, int]
    micro_batch_sizes: Dict[int, int]
    estimated_throughput: float
    notes: str = ""


@dataclass
class TrainingConfig:
    """
    Full training configuration for the DES-LOC engine.

    All fields have sensible defaults tuned for the 2xA6000+1xH100 target cluster.
    """
    # Model dimensions
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    seq_len: int = 2048

    # Training hyperparameters
    total_steps: int = 100_000
    global_batch_size: int = 64
    micro_batch_size: int = 2
    # Per-GPU micro-batch sizes for heterogeneous clusters.
    # Indexed by CUDA device order matching `tiers` discovery order.
    # When set, _plan_zero3() and the Phase-7 DeviceProfile builder use these
    # values directly instead of computing from multipliers.  The fallback
    # `micro_batch_size` (above) is still used as the base for any device
    # not listed here and for consistency_check arithmetic.
    # Example (5-GPU ags1): [2, 8, 16, 2, 8]  → A6000:2, BW:8, H100:16, A6000:2, BW:8
    micro_batch_size_per_gpu: Optional[List[int]] = None
    grad_accum_steps: int = 8
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    # Checkpointing
    save_every: int = 1000
    checkpoint_dir: Path = _CHECKPOINT_DIR
    resume_from: Optional[Path] = None

    # Logging
    log_every: int = 10

    # Eval hook: run eval/run_eval.py every this many steps (0 = disabled)
    eval_every: int = 0
    # Path to a saved model checkpoint dir for evaluation (None = skip model load)
    eval_model_path: Optional[str] = None
    # Output directory for eval result JSON files
    eval_output_dir: str = "desloc_results/eval_runs"

    # Strategy override (None = auto-select)
    strategy_override: Optional[PartitionStrategy] = None

    # Heterogeneous checkpoint config.  When None the engine will auto-build
    # one via build_config_for_cluster() at init time.
    hetero_checkpoint_config: Optional[Any] = None

    # HeteroStepBatchScheduler config
    # Format: "0:32 90B:64 180B:128" (THRESHOLD:BATCH_SIZE, token or sample units)
    # If None, scheduler uses a single constant entry based on global_batch_size
    batch_schedule: Optional[str] = None
    # If provided, schedule thresholds are interpreted as token counts (÷ seq_len → samples)
    batch_schedule_seq_length: Optional[int] = None

    # Activation checkpointing config.
    # activation_checkpointing: master on/off switch.
    #   Defaults True — required for seq_len≥4096 on A6000 (48 GB VRAM).
    #   Set False explicitly only on H100-only clusters where VRAM is plentiful.
    # checkpoint_activations_granularity: "full" (every layer) or "selective" (every other layer).
    # Per-tier policy applied in DesLocEngine.__init__ (overrides granularity):
    #   A6000 (48 GB, SM 8.6)  → FULL checkpoint   (wrap every TransformerBlock)
    #   H100  (96 GB, SM 9.x)  → SELECTIVE ckpt    (wrap every other TransformerBlock)
    #   RTX_PRO_6000_BW (96 GB)→ SELECTIVE ckpt    (same as H100)
    # References: PipeDream runtime.py enable_recompute per-stage flag;
    #             HetSeq controller.py OOM guard at line 282.
    activation_checkpointing: bool = True
    checkpoint_activations_granularity: str = "full"  # "full" | "selective"

    # Fine-grained CPU activation offload (A6000 only).
    # Wired via core_adapters.maybe_enable_activation_offload() → PipelineOffloadManager.
    # On A6000 PCIe (32 GB/s BW) offload is only worth it for cheap-to-transfer
    # tensors (embeddings, small residuals); attention scores are faster to recompute.
    # use_activation_offload: opt-in master switch (default False; safe on H100 clusters).
    # activation_offload_min_size: minimum tensor element count to offload to pinned CPU.
    #   Default 1 M elements ≈ 4 MB at fp32 / 2 MB at bf16 — skips small weight-grad tensors.
    # activation_offload_max_inflight: max concurrent D2H transfers per group name.
    #   None = unlimited; 4 is a safe cap for A6000 PCIe to avoid BW saturation.
    use_activation_offload: bool = False
    activation_offload_min_size: int = 1_048_576   # 1 M elements
    activation_offload_max_inflight: Optional[int] = 4

    # Logging backends (rank 0 only)
    # wandb_project: W&B project name; None = disabled.  Requires wandb installed.
    # tensorboard_dir: directory for SummaryWriter; None = disabled.  Requires tensorboard.
    wandb_project: Optional[str] = None
    tensorboard_dir: Optional[str] = None

    # From Megatron M2492: cpu-offloading-num-layers interface.
    # Setting cpu_offloading_num_layers > 0 enables layer-level CPU offloading
    # (moves activations/weights of that many Transformer layers to host RAM).
    # On DES-LOC A6000×2 (48 GB each, PCIe, no NVLink) this is the primary knob
    # for fitting larger models: offload the bottom N layers to the 1.5 TB DDR5
    # pool when VRAM is exhausted.  Setting this also forces cpu_offloading=True.
    cpu_offloading_num_layers: int = 0
    """Number of Transformer layers to offload to CPU. 0 = disabled.
    When > 0, cpu_offloading is automatically enabled (Megatron M2492)."""

    @property
    def cpu_offloading(self) -> bool:
        """Derived flag: True when any layers are scheduled for CPU offload."""
        return self.cpu_offloading_num_layers > 0

    # Insight I8: default flight_recorder for PCIe (Megatron M3499)
    # Megatron M3499 introduced flight_recorder_dump_on_timeout to capture NCCL
    # collective traces when a hang is detected.  In PCIe topologies (our cluster:
    # 2×A6000 + 1×H100 + 2×Blackwell, no NVLink) hangs are far more frequent than
    # on NVLink clusters due to longer all-reduce latency and noisier fabric.
    # We therefore default flight_recorder on here rather than requiring users to
    # opt in.  Buffer size 65536 > Megatron's default 36864 to capture longer traces
    # across the slower PCIe all-reduce steps before the timeout fires.
    flight_recorder_dump_on_timeout: bool = True
    """Dump NCCL flight-recorder traces when a collective timeout is detected.
    Defaults True for PCIe topologies where hangs are more likely than on NVLink."""

    flight_recorder_trace_buffer_size: int = 65536
    """NCCL flight-recorder ring-buffer size (entries). 65536 > Megatron default 36864
    to capture enough history across slow PCIe all-reduce steps before timeout."""

    # From Megatron M2833 (PR #2306): TrainingConfig dataclass fields ported
    # to DES-LOC. These are the fields most relevant to heterogeneous training.

    empty_unused_memory_level: int = 0
    """Call torch.cuda.empty_cache() each iteration to reduce fragmentation.
    0=off, 1=moderate (every step boundary), 2=aggressive (every micro-step).
    From Megatron M2833 (PR #2306).
    On DES-LOC A6000×2 (48 GB, PCIe) memory fragmentation is a key failure
    mode under long training runs — level 1 or 2 is recommended when OOM
    errors occur in the middle of a run rather than at startup."""

    decrease_batch_size_if_needed: bool = False
    """If True, reduce global_batch_size when micro_batch_size * dp_size does
    not evenly divide global_batch_size.  Original batch size is restored if
    training is restarted with a dp_size that does divide it.
    From Megatron M2833 (PR #2306).
    Relevant for DES-LOC when tier failures reduce effective dp_size mid-run."""

    # --- Core module adapter switches ---
    # Each switch gates a deepspeed/core/ module that replaces the inline
    # implementation in desloc_engine. Adapters in core_adapters.py handle
    # the wiring; all fallback gracefully when the module isn't ready.

    use_core_scheduler: bool = False
    """Replace torch LambdaLR with OptimizerParamScheduler from
    deepspeed/core/optimizer_param_scheduler.py. Enables WSD decay,
    per-tier LR multipliers, and weight decay scheduling."""

    lr_decay_style: str = "cosine"
    """LR decay style when use_core_scheduler=True. 'cosine', 'linear', 'WSD'."""

    tier_lr_multiplier: float = 1.0
    """Per-tier LR scaling factor when use_core_scheduler=True.
    A6000 tiers should use ~0.8, H100 tiers use 1.0."""

    use_pipeline_schedule: bool = True
    """Replace inline forward_backward_func closure with Megatron-style
    pipeline schedule from deepspeed/core/pipeline_parallel/schedules.py.
    Enables combined_1f1b, interleaved 1F1B, A2A overlap.
    Enabled by default; set False only for debugging single-GPU runs."""

    virtual_pipeline_model_parallel_size: Optional[int] = None
    """Virtual pipeline model parallel size for interleaved 1F1B.
    Only used when use_pipeline_schedule=True."""

    pipeline_parallel_size: int = 1
    """Number of pipeline stages (PP world size).
    Must be consistent with the parallel_state PP group initialised before
    DesLocEngine.__init__.  When use_pipeline_schedule=True and this is > 1,
    the engine initialises a P2PCommunicator and ProcessGroupCollection for
    the 1F1B schedule and replaces the per-micro-batch serial loop."""

    pipeline_layer_split: List[int] = field(default_factory=list)
    """Per-stage layer counts for heterogeneous (DES-LOC) pipelines.
    Example for 5-stage split across NUMA0 (GPU0-2) and NUMA1 (GPU3-4):
        [4, 8, 8, 4, 8]   (total 32 layers)
    When non-empty, registered via set_pipeline_layer_split() so that
    get_pipeline_model_parallel_rank_for_layer() resolves correctly.
    Leave empty for uniform splits."""

    use_dist_checkpointing: bool = False
    """Replace torch.save/load with async sharded checkpointing from
    deepspeed/core/dist_checkpointing/. Enables non-blocking saves."""

    use_bridge_communicator: bool = False
    """Replace PCIeP2PCommunicator with BridgeCommunicator from
    deepspeed/core/pipeline_parallel/p2p_communication.py for
    cross-grid activation transfer when PP > 1."""

    use_context_parallel: bool = False
    """Replace the default micro-batch forward/backward loop with
    hybrid_context_parallel_forward_backward from
    deepspeed/core/pipeline_parallel/hybrid_cp_schedule.py.
    Enables variable-length sub-sample packing across the DPxCP domain
    with balanced workload scheduling. Requires PP=1."""

    # --- MoE (Mixture-of-Experts) subsystem ---
    # Wired via deepspeed/runtime/core_adapters.py::build_moe_adapter().
    # When use_moe=False (default) the MoE path is completely skipped and
    # all models remain dense — no overhead.

    use_moe: bool = False
    """Replace every moe_layer_freq-th TransformerBlock MLP with a MoELayer
    from deepspeed/core/transformer/moe/. Disabled by default so existing
    dense-model training runs are completely unaffected."""

    num_moe_experts: int = 8
    """Number of expert MLPs per MoE layer. Effective for both A6000 and H100
    tiers. Production Mixtral-style: 8 experts, top-2 routing."""

    moe_router_topk: int = 2
    """Number of experts each token is routed to (top-k routing)."""

    moe_aux_loss_coeff: float = 0.01
    """Auxiliary load-balancing loss coefficient (Switch Transformer / Megatron
    convention). Added to the main cross-entropy loss at each micro-batch."""

    moe_z_loss_coeff: float = 0.0
    """Router z-loss weight (ST-MoE). Penalises large router logits to improve
    training stability. 0.0 = disabled."""

    moe_token_capacity_factor: Optional[float] = None
    """Expert capacity factor (tokens per expert = capacity_factor * num_tokens /
    num_experts). None = no capacity cap (drop no tokens)."""

    moe_layer_freq: int = 1
    """Replace every N-th TransformerBlock MLP with a MoELayer. 1 = every layer,
    2 = every other layer, etc. Useful for hybrid dense-MoE architectures."""

    moe_on_all_tiers: bool = True
    """When True, apply MoE to layers on A6000 tiers as well as H100 tiers.
    When False, only H100 (high-VRAM) layers get MoE; A6000 layers stay dense.
    Set False when VRAM on A6000 (48 GB) is the binding constraint."""

    moe_num_shared_experts: int = 0
    """DeepSeek-style shared (always-active) expert count per MoE layer. 0 = none."""

    moe_log_every: int = 100
    """Log per-expert token-utilisation statistics every this many steps."""

    ffn_hidden_size: Optional[int] = None
    """Expert intermediate (FFN) hidden size. None → defaults to hidden_size * 4."""

    activation_func_type: str = "swiglu"
    """Expert activation function: 'swiglu' (default, matches LLaMA/Mixtral) or 'gelu'."""
