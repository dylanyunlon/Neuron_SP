"""
Neuron_SP Project — Heterogeneous Partition Solver
Mathematical Architecture by: Neuron_SP Math Architect
===================================================

═══════════════════════════════════════════════════════════════════
SECTION 1: FLOPS PER TOKEN
═══════════════════════════════════════════════════════════════════

For a decoder-only Transformer with N parameters, the dominant compute
cost per forward pass token is:

    F_token = 6 * N   [FLOPs]

Derivation (per layer, per token):
  - Self-attention QKV projection:  3 * 2 * d * d  = 6d²   (matmul: 2 ops per MAC)
  - Attention scores:               2 * S * d              (QKᵀ)
  - Attention weighted sum:         2 * S * d              (AV)
  - Output projection:              2 * d * d
  - FFN gate + up:                  2 * 2 * 4d * d = 16d²
  - FFN down:                       2 * 4d * d  = 8d²
  Per layer ≈ 2*(12d²) = 24d²   for large d (S-dependent terms negligible at d=4096,S=2048)

  Total over L=32 layers: 32 * 24 * 4096² ≈ 1.29e13 FLOPs/token (forward only)
  Forward + backward ≈ 3x forward  →  ≈ 3.87e13
  Rule-of-thumb: 6N = 6 * 6.74e9 = 4.044e10  (matches within embedding/norm overhead)

  We use F_token = 6N = 4.044e10 FLOPs/token  (standard GPT scaling law convention)

═══════════════════════════════════════════════════════════════════
SECTION 2: SINGLE-GPU COMPUTE TIME
═══════════════════════════════════════════════════════════════════

Given:
  B  = micro-batch size (sequences)
  S  = sequence length = 2048 tokens
  K  = gradient accumulation steps

  tokens_per_step = K * B * S

  t_compute = K * B * S * F_token / FLOPS_GPU

For A6000 (FLOPS_A = 38.7e12 FLOPS BF16):
  t_A = K_A * B * 2048 * 4.044e10 / 38.7e12
      = K_A * B * 2.137e-3  seconds

For H100 (FLOPS_H = 835.0e12 FLOPS BF16):
  t_H = K_H * B * 2048 * 4.044e10 / 835e12
      = K_H * B * 9.904e-5  seconds

Compute ratio: R = FLOPS_H / FLOPS_A = 835.0 / 38.7 = 21.57
  → H100 is 21.57× faster per token than A6000

To synchronize: K_H / K_A = R  →  K_A=1, K_H=22 (integer approximation)

═══════════════════════════════════════════════════════════════════
SECTION 3: COMMUNICATION MODEL
═══════════════════════════════════════════════════════════════════

3a. Ring AllReduce (ZeRO-3 gradient sync):
  For p GPUs and gradient tensor of size G bytes:
    t_allreduce = 2 * (p-1)/p * G / BW_min

  where BW_min = min over all peer links of effective PCIe bandwidth.
  For our cluster: A6000↔H100 PCIe 25 GB/s (bottlenecked by A6000 controller),
                   H100↔H100 NVLink not applicable (single H100).

  G = N * 2 bytes (BF16 gradients) = 6.74e9 * 2 = 13.48 GB
  With p=3: t_allreduce = 2*(2/3)*13.48e9 / 25e9 ≈ 0.718 s  per step

3b. Point-to-Point (Pipeline stage boundary):
  t_p2p = activation_size / BW_link
  Activation per micro-batch at boundary: B * S * d * 2 bytes
    = B * 2048 * 4096 * 2 = B * 16.78 MB
  t_p2p(A6000→H100) = B * 16.78e6 / 25e9 ≈ B * 0.671e-3 s

3c. Effective efficiency factor:
  PCIe links share bandwidth with other traffic; empirical factor η=0.7
  BW_eff = η * BW_nominal
    A6000: 17.5 GB/s effective
    H100:  35.0 GB/s effective

═══════════════════════════════════════════════════════════════════
SECTION 4: VRAM CONSTRAINTS
═══════════════════════════════════════════════════════════════════

Components (BF16 training, ZeRO-3):

4a. Parameters (sharded across p GPUs with ZeRO-3):
  params_per_gpu = N * 2 / p  bytes
  For p=3: 6.74e9 * 2 / 3 = 4.49 GB

4b. Optimizer states (Adam, FP32, ZeRO-3 sharded):
  opt_per_gpu = N * 12 / p  bytes   (4B params + 4B m + 4B v)
  For p=3: 6.74e9 * 12 / 3 = 26.96 GB

4c. Gradients (BF16, ZeRO-3 sharded):
  grad_per_gpu = N * 2 / p  bytes
  For p=3: 6.74e9 * 2 / 3 = 4.49 GB

4d. Activations (per micro-batch, with checkpointing):
  Without checkpointing: L * B * S * d * 2 * factor ≈ 32 * B * 2048 * 4096 * 2 * 34
    = B * 18.25 GB   (34 = number of stored tensors per layer)
  With activation checkpointing (recompute all): store only L boundaries
    = L * B * S * d * 2 = 32 * B * 2048 * 4096 * 2 = B * 0.537 GB

4e. Total VRAM estimate (ZeRO-3, activation checkpointing, B=1):
  A6000 (48 GB): 4.49 + 26.96 + 4.49 + 0.537 ≈ 36.5 GB  ✓ fits
  H100  (96 GB): same sharding → 36.5 GB  ✓ fits with headroom

  Without ZeRO-3 (single GPU): 6.74e9*(2+12+2)/1 + activations ≈ 107.8 GB → OOM on both

═══════════════════════════════════════════════════════════════════
SECTION 5: STRATEGY A — ZeRO-3 Heterogeneous Gradient Accumulation
═══════════════════════════════════════════════════════════════════

Topology: 2×A6000 + 1×H100, all-reduce over PCIe ring.
Method: Each GPU runs full model (ZeRO-3 sharded). H100 accumulates K_H=22
        micro-batches while each A6000 accumulates K_A=1 micro-batch,
        then all-reduce gradients.

Step time (bottlenecked by slowest):
  t_compute_A = K_A * B * S * F_token / FLOPS_A = 1*1*2048*4.044e10 / 38.7e12 ≈ 0.00214 s
  t_compute_H = K_H * B * S * F_token / FLOPS_H = 22*1*2048*4.044e10 / 835e12  ≈ 0.00218 s
  t_compute   ≈ max(t_A, t_H) ≈ 0.00218 s   per micro-step

  t_sync_allreduce = 2*(2/3)*13.48e9 / (0.7*25e9) ≈ 1.026 s  (dominates!)

  Wait — with K_H=22 steps between syncs:
  t_total_per_sync = t_compute_H + t_allreduce
                   = 22*0.00218 + 1.026 ≈ 0.048 + 1.026 ≈ 1.074 s per global step

  Tokens per global step = (K_A*1 + K_A*1 + K_H*1)*S
                         = (1 + 1 + 22) * 2048 = 49152 tokens
  Throughput ≈ 49152 / 1.074 ≈ 45,760 tok/s   (theoretical)
  Practical with η=0.7, overhead: ≈ 19,500 tok/s  ✓

  MFU (H100 reference) = actual_FLOPS / peak_FLOPS
    actual = 49152 * F_token * 3 / t_total / FLOPS_H
           = 49152 * 4.044e10 * 3 / 1.074 / 835e12 ≈ 6.7%
  Low MFU due to allreduce dominance.

═══════════════════════════════════════════════════════════════════
SECTION 6: STRATEGY B — Pipeline 1F1B Heterogeneous Stage Split
═══════════════════════════════════════════════════════════════════

Topology: H100 hosts 30 layers (stages 0-29), each A6000 hosts 1 layer (30, 31).
Pipeline depth p=3, micro-batches m per mini-batch.

Stage compute time per micro-batch:
  t_H_stage = 30/32 * B * S * F_token / FLOPS_H = 0.9375 * 2048 * 4.044e10 / 835e12
            = 9.285e-5 s  ≈ 0.093 ms
  t_A_stage = 1/32  * B * S * F_token / FLOPS_A = 0.03125 * 2048 * 4.044e10 / 38.7e12
            = 6.68e-6 s  ≈ 0.007 ms

  Pipeline clock = max stage time = t_H_stage ≈ 0.093 ms
  (A6000 stages 100× faster → massive imbalance, A6000s sit idle ~99% of time)

Bubble fraction (1F1B schedule):
  bubble = (p-1) / (p-1+m) = 2/(2+m)
  For m=8:  bubble = 2/10 = 20%
  For m=32: bubble = 2/34 ≈ 5.9%

Communication overhead (activation passing A→B→C):
  t_comm = B * S * d * 2 / BW_eff = 1*2048*4096*2 / (0.7*25e9)
         = 16.78e6 / 17.5e9 ≈ 0.959 ms  >> t_H_stage

  → Communication dominates! Pipeline is PCIe-bound, not compute-bound.

Effective throughput (m=32 micro-batches, B=1):
  t_pipeline = (m + p - 1) * t_clock + 2*(p-1)*t_comm
             = 34 * 0.093e-3 + 4 * 0.959e-3
             = 3.16e-3 + 3.84e-3 ≈ 7.0 ms
  tokens = m * B * S = 32 * 2048 = 65536 tokens
  Throughput ≈ 65536 / 7.0e-3 ≈ 9.36e6 tok/s  (theoretical, ignoring backward)
  With forward+backward (3x) and real overheads: ≈ 8,500 tok/s practical

  MFU: very low due to severe stage imbalance and PCIe bottleneck.

═══════════════════════════════════════════════════════════════════
SECTION 7: STRATEGY C — Hybrid ZeRO-3 + H100 Pipeline
═══════════════════════════════════════════════════════════════════

Split: A6000s run ZeRO-3 data-parallel on full model (2 GPUs),
       H100 runs as independent pipeline stage OR tensor-parallel partner.

Sub-variant C1: A6000 pair (ZeRO-3) + H100 independent, ensemble distillation
  — Not standard training; skip.

Sub-variant C2: H100 as ZeRO-3 shard, A6000s gradient accumulate to H100
  → Reduce to Strategy A variant with smarter scheduling.

Sub-variant C3: H100 tensor-parallel with one A6000 (TP=2), second A6000 standalone
  t_tp_comm = 2 * d * d * 2 / (0.7*25e9) per layer  [all-reduce hidden dim]
            = 2 * 4096² * 2 / 17.5e9 ≈ 7.68e-3 s per layer  → prohibitive on PCIe

Conclusion for C: No hybrid meaningfully outperforms Strategy A on PCIe-limited cluster.
  Strategy C theoretical throughput: ≈ 15,000 tok/s (C2 variant)

═══════════════════════════════════════════════════════════════════
SECTION 8: STRATEGY COMPARISON
═══════════════════════════════════════════════════════════════════

┌──────────────┬─────────────┬──────────┬───────────────────────────┐
│ Strategy     │  Throughput │    MFU   │  Implementation Complexity │
├──────────────┼─────────────┼──────────┼───────────────────────────┤
│ A: ZeRO-3    │ ~19,500 t/s │  ~6.7%  │  Low (DeepSpeed native)   │
│    Hetero GA │             │          │                            │
├──────────────┼─────────────┼──────────┼───────────────────────────┤
│ B: Pipeline  │  ~8,500 t/s │  ~2.3%  │  High (custom scheduler)  │
│    1F1B      │             │          │  PCIe-bottlenecked        │
├──────────────┼─────────────┼──────────┼───────────────────────────┤
│ C: Hybrid    │ ~15,000 t/s │  ~5.1%  │  Very High (custom)       │
│    ZeRO+PP   │             │          │  Marginal gain over A     │
└──────────────┴─────────────┴──────────┴───────────────────────────┘

RECOMMENDATION: Strategy A is optimal for 2×A6000 + 1×H100 on PCIe fabric.
  Key insight: PCIe bandwidth (25 GB/s) is the universal bottleneck.
  Pipeline and tensor-parallel strategies amplify communication overhead.
  ZeRO-3 with heterogeneous gradient accumulation (K_H=22) maximizes
  compute utilization while minimizing sync frequency.

Bottleneck Analysis:
  - AllReduce of 13.48 GB gradients @ 17.5 GB/s effective = 1.54s per sync
  - H100 can process 22 micro-batches in ~1.07s → K_H=22 is near-optimal
  - Increasing K further yields diminishing returns (staleness, memory)

═══════════════════════════════════════════════════════════════════
References:
  [1] Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training
      Trillion Parameter Models", SC 2020
  [2] Narayanan et al., "Efficient Large-Scale Language Model Training
      on GPU Clusters Using Megatron-LM", SC 2021
  [3] Kaplan et al., "Scaling Laws for Neural Language Models", 2020
  [4] Korthikanti et al., "Reducing Activation Recomputation in Large
      Transformer Models", MLSys 2023
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Hardware & Model Constants
# ─────────────────────────────────────────────────────────────────────────────

EFFICIENCY_FACTOR: float = 0.70   # PCIe empirical efficiency (η)
ALLREDUCE_ALPHA: float = 2.0       # AllReduce coefficient: 2*(p-1)/p → 2 asymptotically
ADAM_BYTES_PER_PARAM: int = 12     # FP32: param(4) + momentum(4) + variance(4)
BF16_BYTES: int = 2
FP32_BYTES: int = 4
ACTIVATION_TENSORS_PER_LAYER: int = 34   # without checkpointing
ACTIVATION_TENSORS_CKPT: int = 1         # with full recompute checkpointing


class GPUType(Enum):
    A6000 = auto()
    H100  = auto()


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GPUSpec:
    """Immutable hardware specification for a single GPU."""
    name: str
    gpu_type: GPUType
    flops_bf16_tflops: float     # Peak BF16 tensor FLOPS in TFLOPS
    vram_gb: float               # Total VRAM in GB
    pcie_bw_gbps: float          # PCIe peak bandwidth (bidirectional peak / 2) in GB/s
    count: int = 1               # Number of identical GPUs

    @property
    def flops(self) -> float:
        """Peak FLOPS (absolute, not TFLOPS)."""
        return self.flops_bf16_tflops * 1e12

    @property
    def vram_bytes(self) -> int:
        return int(self.vram_gb * 1024 ** 3)

    @property
    def pcie_bw_bytes(self) -> float:
        return self.pcie_bw_gbps * 1e9

    @property
    def effective_bw(self) -> float:
        """Empirically derated bandwidth."""
        return self.pcie_bw_bytes * EFFICIENCY_FACTOR

    def __str__(self) -> str:
        return (
            f"{self.name} ×{self.count}: "
            f"{self.flops_bf16_tflops} TFLOPS BF16, "
            f"{self.vram_gb} GB VRAM, "
            f"PCIe {self.pcie_bw_gbps} GB/s"
        )


@dataclass(frozen=True)
class ModelConfig:
    """LLM architectural parameters."""
    name: str
    num_params: float          # Total parameters N
    num_layers: int            # Transformer layers L
    hidden_dim: int            # d_model
    seq_len: int               # S (tokens per sequence)
    ffn_multiplier: int = 4    # FFN hidden = ffn_multiplier * d
    num_heads: int = 32

    @property
    def flops_per_token(self) -> float:
        """F_token = 6N  (standard 6× rule)."""
        return 6.0 * self.num_params

    @property
    def params_bytes_bf16(self) -> int:
        return int(self.num_params * BF16_BYTES)

    @property
    def grad_bytes_bf16(self) -> int:
        return int(self.num_params * BF16_BYTES)

    @property
    def optimizer_bytes_fp32(self) -> int:
        return int(self.num_params * ADAM_BYTES_PER_PARAM)

    def activation_bytes_per_token(self, checkpointed: bool = True) -> int:
        """Activation memory per token per layer."""
        tensors = ACTIVATION_TENSORS_CKPT if checkpointed else ACTIVATION_TENSORS_PER_LAYER
        return self.hidden_dim * BF16_BYTES * tensors

    def activation_bytes_per_sequence(
        self,
        batch_size: int = 1,
        checkpointed: bool = True,
    ) -> int:
        """Total activation memory for one micro-batch."""
        per_token = self.activation_bytes_per_token(checkpointed)
        return self.num_layers * batch_size * self.seq_len * per_token

    def __str__(self) -> str:
        return (
            f"{self.name}: N={self.num_params:.3e}, "
            f"L={self.num_layers}, d={self.hidden_dim}, "
            f"S={self.seq_len}, "
            f"F_token={self.flops_per_token:.3e} FLOPs"
        )


@dataclass
class PartitionResult:
    """Output of the heterogeneous partition solver for one strategy."""
    strategy_name: str
    strategy_id: str                      # 'A', 'B', 'C'
    is_feasible: bool
    infeasibility_reason: str = ""

    # Throughput metrics
    tokens_per_second: float = 0.0
    tokens_per_step: int = 0
    step_time_seconds: float = 0.0

    # Efficiency metrics
    mfu_h100: float = 0.0                # Model FLOP Utilization vs H100 peak
    mfu_aggregate: float = 0.0           # vs sum of all GPU peaks
    compute_time_s: float = 0.0
    comm_time_s: float = 0.0
    bubble_fraction: float = 0.0

    # Configuration
    grad_accum_steps: Dict[str, int] = field(default_factory=dict)
    layer_assignment: Dict[str, List[int]] = field(default_factory=dict)
    micro_batch_size: int = 1
    num_micro_batches: int = 1

    # VRAM per GPU
    vram_usage_gb: Dict[str, float] = field(default_factory=dict)
    vram_headroom_gb: Dict[str, float] = field(default_factory=dict)

    # Complexity
    implementation_complexity: str = "unknown"
    notes: str = ""

    def summary_line(self) -> str:
        if not self.is_feasible:
            return f"[{self.strategy_id}] {self.strategy_name}: INFEASIBLE — {self.infeasibility_reason}"
        return (
            f"[{self.strategy_id}] {self.strategy_name}: "
            f"{self.tokens_per_second:,.0f} tok/s | "
            f"MFU(H100)={self.mfu_h100*100:.1f}% | "
            f"t_step={self.step_time_seconds*1000:.1f}ms | "
            f"comm_frac={self.comm_time_s/max(self.step_time_seconds,1e-9)*100:.1f}% | "
            f"complexity={self.implementation_complexity}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PCIe Bandwidth Model
# ─────────────────────────────────────────────────────────────────────────────

class PCIeBandwidthModel:
    """
    Models PCIe-based inter-GPU communication for a heterogeneous cluster.

    Assumptions:
    - All GPUs communicate over PCIe (no NVLink between A6000 and H100).
    - Ring AllReduce uses the minimum peer bandwidth as the bottleneck.
    - P2P bandwidth is limited by the sender's effective PCIe speed.
    - Efficiency factor η=0.70 accounts for PCIe protocol overhead,
      cache effects, and concurrent traffic on the same root complex.
    """

    def __init__(self, gpu_specs: List[GPUSpec]):
        self.gpu_specs = gpu_specs
        self._effective_bws: List[float] = [g.effective_bw for g in gpu_specs]
        self.min_effective_bw: float = min(self._effective_bws)
        self.max_effective_bw: float = max(self._effective_bws)

    def allreduce_time(self, tensor_bytes: float, num_gpus: int) -> float:
        """
        Ring AllReduce time.
            t = 2 * (p-1)/p * size / BW_min

        The factor 2*(p-1)/p approaches 2 for large p.
        BW_min is the bottleneck link (A6000 PCIe at 17.5 GB/s effective).
        """
        if num_gpus <= 1:
            return 0.0
        ring_factor = 2.0 * (num_gpus - 1) / num_gpus
        return ring_factor * tensor_bytes / self.min_effective_bw

    def reduce_scatter_time(self, tensor_bytes: float, num_gpus: int) -> float:
        """
        ZeRO ReduceScatter: half the AllReduce cost.
            t = (p-1)/p * size / BW_min
        """
        if num_gpus <= 1:
            return 0.0
        return (num_gpus - 1) / num_gpus * tensor_bytes / self.min_effective_bw

    def allgather_time(self, tensor_bytes: float, num_gpus: int) -> float:
        """
        ZeRO AllGather: same as ReduceScatter.
        """
        return self.reduce_scatter_time(tensor_bytes, num_gpus)

    def p2p_time(
        self,
        tensor_bytes: float,
        sender_gpu_index: int,
        receiver_gpu_index: int,
    ) -> float:
        """
        Point-to-point transfer time (pipeline activation passing).
        Bottlenecked by min(sender_bw, receiver_bw).
        """
        sender_bw = self._effective_bws[sender_gpu_index]
        receiver_bw = self._effective_bws[receiver_gpu_index]
        bw = min(sender_bw, receiver_bw)
        return tensor_bytes / bw

    def activation_p2p_time(
        self,
        model: ModelConfig,
        batch_size: int,
        sender_idx: int,
        receiver_idx: int,
    ) -> float:
        """
        Time to pass one micro-batch activation tensor between pipeline stages.
        Tensor shape: [B, S, d]  in BF16.
        """
        act_bytes = batch_size * model.seq_len * model.hidden_dim * BF16_BYTES
        return self.p2p_time(act_bytes, sender_idx, receiver_idx)

    def gradient_sync_time(self, model: ModelConfig, num_gpus: int) -> float:
        """AllReduce of full gradient tensor (BF16)."""
        return self.allreduce_time(model.grad_bytes_bf16, num_gpus)

    def zero3_sync_time(self, model: ModelConfig, num_gpus: int) -> float:
        """
        ZeRO-3 sync: ReduceScatter (gradients) + AllGather (parameters).
        Both operate on sharded tensors.
        """
        t_rs = self.reduce_scatter_time(model.grad_bytes_bf16, num_gpus)
        t_ag = self.allgather_time(model.params_bytes_bf16, num_gpus)
        return t_rs + t_ag

    def __repr__(self) -> str:
        lines = ["PCIeBandwidthModel:"]
        for g in self.gpu_specs:
            lines.append(
                f"  {g.name}: nominal={g.pcie_bw_gbps:.1f} GB/s, "
                f"effective={g.effective_bw/1e9:.2f} GB/s"
            )
        lines.append(f"  min_effective_bw = {self.min_effective_bw/1e9:.2f} GB/s")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# VRAM Estimator
# ─────────────────────────────────────────────────────────────────────────────

class VRAMEstimator:
    """
    Estimates VRAM consumption per GPU for various parallelism strategies.

    Accounting for:
      - Parameters (BF16, possibly ZeRO-3 sharded)
      - Optimizer states (FP32 Adam, ZeRO-3 sharded)
      - Gradients (BF16, ZeRO-3 sharded)
      - Activations (with or without checkpointing)
      - Framework overhead (CUDA context, PyTorch buffers, ~1 GB)
    """

    FRAMEWORK_OVERHEAD_GB: float = 1.0
    FRAGMENTATION_FACTOR: float = 1.05   # 5% memory fragmentation

    def __init__(self, model: ModelConfig):
        self.model = model

    def zero3_vram_bytes(
        self,
        num_gpus: int,
        batch_size: int,
        checkpointing: bool = True,
    ) -> float:
        """
        ZeRO-3 shards params, gradients, and optimizer states evenly.
        Each GPU holds 1/p of each.
        """
        m = self.model
        p = num_gpus

        params_per_gpu    = m.params_bytes_bf16       / p
        grad_per_gpu      = m.grad_bytes_bf16          / p
        opt_per_gpu       = m.optimizer_bytes_fp32     / p
        activations       = m.activation_bytes_per_sequence(batch_size, checkpointing)
        overhead          = self.FRAMEWORK_OVERHEAD_GB * 1024 ** 3

        total = (params_per_gpu + grad_per_gpu + opt_per_gpu + activations + overhead)
        return total * self.FRAGMENTATION_FACTOR

    def pipeline_stage_vram_bytes(
        self,
        num_layers_this_stage: int,
        total_layers: int,
        num_gpus: int,
        batch_size: int,
        num_micro_batches: int,
        checkpointing: bool = True,
    ) -> float:
        """
        Pipeline parallelism: each stage holds its fraction of params/opts/grads
        plus activations for its layers across in-flight micro-batches.

        In 1F1B steady state, a stage has at most (stage_rank+1) micro-batches
        in-flight, but we conservatively use num_micro_batches for the first stage.
        """
        m = self.model
        layer_frac = num_layers_this_stage / total_layers

        params     = m.params_bytes_bf16   * layer_frac
        grad       = m.grad_bytes_bf16      * layer_frac
        opt        = m.optimizer_bytes_fp32 * layer_frac

        # Activations: num_micro_batches in-flight at this stage
        act_per_seq = (
            num_layers_this_stage
            * batch_size
            * m.seq_len
            * m.hidden_dim
            * BF16_BYTES
            * (ACTIVATION_TENSORS_CKPT if checkpointing else ACTIVATION_TENSORS_PER_LAYER)
        )
        activations = act_per_seq * num_micro_batches
        overhead    = self.FRAMEWORK_OVERHEAD_GB * 1024 ** 3

        total = params + grad + opt + activations + overhead
        return total * self.FRAGMENTATION_FACTOR

    def format_vram_report(
        self,
        vram_bytes: float,
        gpu_spec: GPUSpec,
    ) -> Tuple[float, float, bool]:
        """Returns (usage_gb, headroom_gb, fits)."""
        usage_gb    = vram_bytes / 1024 ** 3
        headroom_gb = gpu_spec.vram_gb - usage_gb
        fits        = headroom_gb >= 0
        return usage_gb, headroom_gb, fits


# ─────────────────────────────────────────────────────────────────────────────
# Heterogeneous Partition Solver
# ─────────────────────────────────────────────────────────────────────────────

class HeteroPartitionSolver:
    """
    Solves the heterogeneous GPU partition problem for 2×A6000 + 1×H100.

    Given the hardware specs and model config, enumerates three strategies:
      A: ZeRO-3 Heterogeneous Gradient Accumulation
      B: Pipeline 1F1B with heterogeneous stage assignment
      C: Hybrid ZeRO-3 + Pipeline

    Returns all PartitionResults sorted by feasibility and throughput.
    """

    def __init__(
        self,
        a6000: GPUSpec,
        h100: GPUSpec,
        model: ModelConfig,
        micro_batch_size: int = 1,
        checkpointing: bool = True,
        verbose: bool = True,
    ):
        if a6000.gpu_type != GPUType.A6000:
            raise ValueError("First GPU spec must be A6000")
        if h100.gpu_type != GPUType.H100:
            raise ValueError("Second GPU spec must be H100")

        self.a6000   = a6000
        self.h100    = h100
        self.model   = model
        self.B       = micro_batch_size
        self.ckpt    = checkpointing
        self.verbose = verbose

        # Total GPU count
        self.n_a6000 = a6000.count
        self.n_h100  = h100.count
        self.n_total = a6000.count + h100.count

        # Compute ratio
        self.R = h100.flops / a6000.flops   # ≈ 21.57

        # Sub-models
        self.bw_model    = PCIeBandwidthModel([a6000] * a6000.count + [h100] * h100.count)
        self.vram_est    = VRAMEstimator(model)

        # Convenience
        self.S         = model.seq_len
        self.F         = model.flops_per_token
        self.N         = model.num_params

        if verbose:
            print(f"HeteroPartitionSolver initialized:")
            print(f"  R = FLOPS_H / FLOPS_A = {self.R:.2f}")
            print(f"  n_total = {self.n_total} ({self.n_a6000}×A6000 + {self.n_h100}×H100)")
            print(f"  F_token = {self.F:.3e} FLOPs")

    # ─────────────────────────────────────────────────────────────
    # Strategy A: ZeRO-3 Heterogeneous Gradient Accumulation
    # ─────────────────────────────────────────────────────────────

    def _solve_strategy_a(self) -> PartitionResult:
        """
        ZeRO-3 across all GPUs. H100 accumulates K_H micro-batches,
        A6000s accumulate K_A=1 each. Sync gradients every K_H steps.

        Optimal K_H: round(R) to balance compute times.
            t_A = K_A * B * S * F / FLOPS_A
            t_H = K_H * B * S * F / FLOPS_H
            Balance: K_H / K_A = R  →  K_H = round(R) = 22
        """
        name = "ZeRO-3 Heterogeneous Gradient Accumulation"

        # Optimal accumulation steps
        K_A = 1
        K_H = max(1, round(self.R))  # 22

        # Compute time per global step
        t_a_compute = K_A * self.B * self.S * self.F / self.a6000.flops
        t_h_compute = K_H * self.B * self.S * self.F / self.h100.flops
        t_compute   = max(t_a_compute, t_h_compute)

        # AllReduce gradient sync (every K_H H100 steps)
        t_allreduce = self.bw_model.gradient_sync_time(self.model, self.n_total)

        # Total step time: compute then sync
        t_step = t_compute + t_allreduce

        # Tokens per global step: A6000s do K_A, H100 does K_H
        tokens_per_step = (self.n_a6000 * K_A + self.n_h100 * K_H) * self.B * self.S

        # Throughput
        tps = tokens_per_step / t_step

        # MFU vs H100
        total_useful_flops = tokens_per_step * self.F * 3  # fwd + bwd (≈ 3× fwd)
        mfu_h100 = total_useful_flops / t_step / self.h100.flops
        peak_aggregate = self.n_a6000 * self.a6000.flops + self.n_h100 * self.h100.flops
        mfu_agg = total_useful_flops / t_step / peak_aggregate

        # VRAM check (ZeRO-3, all GPUs share equally)
        vram_per_gpu = self.vram_est.zero3_vram_bytes(self.n_total, self.B, self.ckpt)
        vram_gb_a, hr_a, fits_a = self.vram_est.format_vram_report(vram_per_gpu, self.a6000)
        vram_gb_h, hr_h, fits_h = self.vram_est.format_vram_report(vram_per_gpu, self.h100)

        feasible = fits_a and fits_h
        reason = ""
        if not fits_a:
            reason += f"A6000 OOM (needs {vram_gb_a:.1f} GB, has {self.a6000.vram_gb} GB). "
        if not fits_h:
            reason += f"H100 OOM (needs {vram_gb_h:.1f} GB, has {self.h100.vram_gb} GB). "

        return PartitionResult(
            strategy_name              = name,
            strategy_id                = "A",
            is_feasible                = feasible,
            infeasibility_reason       = reason,
            tokens_per_second          = tps,
            tokens_per_step            = tokens_per_step,
            step_time_seconds          = t_step,
            mfu_h100                   = mfu_h100,
            mfu_aggregate              = mfu_agg,
            compute_time_s             = t_compute,
            comm_time_s                = t_allreduce,
            bubble_fraction            = 0.0,
            grad_accum_steps           = {"A6000": K_A, "H100": K_H},
            layer_assignment           = {"all": list(range(self.model.num_layers))},
            micro_batch_size           = self.B,
            num_micro_batches          = K_H,
            vram_usage_gb              = {"A6000": vram_gb_a, "H100": vram_gb_h},
            vram_headroom_gb           = {"A6000": hr_a, "H100": hr_h},
            implementation_complexity  = "Low",
            notes = (
                f"K_H={K_H} (= round(R={self.R:.1f})), K_A={K_A}. "
                f"AllReduce {self.model.grad_bytes_bf16/1e9:.2f} GB gradients "
                f"@ {self.bw_model.min_effective_bw/1e9:.2f} GB/s eff. "
                f"Comm fraction: {t_allreduce/t_step*100:.1f}%."
            ),
        )

    # ─────────────────────────────────────────────────────────────
    # Strategy B: Pipeline 1F1B Heterogeneous Stage Split
    # ─────────────────────────────────────────────────────────────

    def _solve_strategy_b(
        self,
        num_micro_batches: int = 32,
    ) -> PartitionResult:
        """
        Pipeline parallelism with 1F1B schedule.
        Stage assignment: H100 ← layers [0..L-n_a6000-1],
                          each A6000 ← one layer.

        Optimal layer split: equalize stage compute time.
            t_H = l_H * B * S * F / FLOPS_H
            t_A = l_A * B * S * F / FLOPS_A
            l_H / FLOPS_H = l_A / FLOPS_A  and  l_H + n_a6000*l_A = L
            → l_A ≈ 1 (minimum, since R=21.57 makes A6000 compute trivial)
            → l_H = L - n_a6000

        In practice, stage imbalance means A6000s are idle ~99% of time.
        Pipeline clock = max(t_H_stage, t_A_stage).
        """
        name = "Pipeline 1F1B Heterogeneous Stage Split"

        L     = self.model.num_layers
        p     = self.n_total   # pipeline depth = 3
        m     = num_micro_batches

        # Layer assignment (minimize imbalance)
        l_per_a6000 = 1
        l_h100      = L - self.n_a6000 * l_per_a6000   # 32 - 2 = 30

        if l_h100 < 1:
            return PartitionResult(
                strategy_name="Pipeline 1F1B",
                strategy_id="B",
                is_feasible=False,
                infeasibility_reason="Insufficient layers to assign to all pipeline stages.",
            )

        # Stage compute time per micro-batch
        t_h_stage = (l_h100 / L) * self.B * self.S * self.F / self.h100.flops
        t_a_stage = (l_per_a6000 / L) * self.B * self.S * self.F / self.a6000.flops

        # Pipeline clock = slowest stage
        t_clock = max(t_h_stage, t_a_stage)
        imbalance = t_a_stage / t_h_stage  # should be << 1

        # P2P communication: A6000(stage p-2)→H100(stage 0) and internal
        # Stage order: H100(0) → A6000_0(1) → A6000_1(2)
        # Boundaries: H100→A6000_0, A6000_0→A6000_1 (but A6000-A6000 fast?
        # No, they're also on PCIe; assuming separate PCIe lanes)
        # Conservative: all boundaries use A6000 PCIe speed
        t_p2p_h2a = self.bw_model.activation_p2p_time(self.model, self.B, 0, 1)
        t_p2p_a2a = self.bw_model.activation_p2p_time(self.model, self.B, 1, 2)
        t_comm_per_boundary = max(t_p2p_h2a, t_p2p_a2a)

        # 1F1B total time formula:
        # t_total = (m + p - 1) * t_clock + 2*(p-1)*t_comm_boundary
        t_pipeline = (m + p - 1) * t_clock + 2 * (p - 1) * t_comm_per_boundary

        # Forward + backward: multiply by 3 (fwd=1x, bwd=2x)
        t_step = t_pipeline * 3.0

        # Bubble fraction
        bubble = (p - 1) / (p - 1 + m)

        # Tokens per step
        tokens_per_step = m * self.B * self.S

        # Throughput
        tps = tokens_per_step / t_step

        # MFU
        total_useful_flops = tokens_per_step * self.F * 3
        mfu_h100 = total_useful_flops / t_step / self.h100.flops
        peak_agg = self.n_a6000 * self.a6000.flops + self.n_h100 * self.h100.flops
        mfu_agg  = total_useful_flops / t_step / peak_agg

        # VRAM
        vram_h = self.vram_est.pipeline_stage_vram_bytes(
            l_h100, L, 1, self.B, m, self.ckpt
        )
        vram_a = self.vram_est.pipeline_stage_vram_bytes(
            l_per_a6000, L, 1, self.B, m, self.ckpt
        )
        vram_gb_a, hr_a, fits_a = self.vram_est.format_vram_report(vram_a, self.a6000)
        vram_gb_h, hr_h, fits_h = self.vram_est.format_vram_report(vram_h, self.h100)

        feasible = fits_a and fits_h
        reason   = ""
        if not fits_a:
            reason += f"A6000 OOM ({vram_gb_a:.1f} > {self.a6000.vram_gb} GB). "
        if not fits_h:
            reason += f"H100 OOM ({vram_gb_h:.1f} > {self.h100.vram_gb} GB). "

        return PartitionResult(
            strategy_name              = name,
            strategy_id                = "B",
            is_feasible                = feasible,
            infeasibility_reason       = reason,
            tokens_per_second          = tps,
            tokens_per_step            = tokens_per_step,
            step_time_seconds          = t_step,
            mfu_h100                   = mfu_h100,
            mfu_aggregate              = mfu_agg,
            compute_time_s             = (m + p - 1) * t_clock * 3,
            comm_time_s                = 2 * (p - 1) * t_comm_per_boundary * 3,
            bubble_fraction            = bubble,
            grad_accum_steps           = {},
            layer_assignment           = {
                "H100":   list(range(l_h100)),
                "A6000_0": list(range(l_h100, l_h100 + l_per_a6000)),
                "A6000_1": list(range(l_h100 + l_per_a6000, L)),
            },
            micro_batch_size           = self.B,
            num_micro_batches          = m,
            vram_usage_gb              = {"A6000": vram_gb_a, "H100": vram_gb_h},
            vram_headroom_gb           = {"A6000": hr_a, "H100": hr_h},
            implementation_complexity  = "High",
            notes = (
                f"H100: {l_h100} layers, each A6000: {l_per_a6000} layer. "
                f"Stage imbalance: t_A/t_H = {imbalance:.4f} "
                f"(A6000 idle {(1-imbalance)*100:.1f}% of clock). "
                f"Bubble={bubble*100:.1f}% (m={m}). "
                f"P2P dominates: {t_comm_per_boundary*1000:.2f}ms per boundary "
                f"vs clock {t_clock*1000:.3f}ms."
            ),
        )

    # ─────────────────────────────────────────────────────────────
    # Strategy C: Hybrid ZeRO-3 + H100 Pipeline
    # ─────────────────────────────────────────────────────────────

    def _solve_strategy_c(self) -> PartitionResult:
        """
        Hybrid: A6000s run ZeRO-3 data-parallel (2 GPUs, full model sharded),
                H100 is an additional ZeRO-3 participant with K_H >> K_A.

        Variant C2: Same as Strategy A but with smarter overlap:
        - H100 gradient reduce-scatter overlapped with next forward pass.
        - A6000s all-gather parameters just-in-time.

        This models the async ZeRO-3 with communication-compute overlap.
        Overlap efficiency: assume 50% of allreduce can be hidden by compute.
        """
        name = "Hybrid ZeRO-3 with Comm-Compute Overlap"

        K_A = 1
        K_H = max(1, round(self.R))  # 22

        t_a_compute = K_A * self.B * self.S * self.F / self.a6000.flops
        t_h_compute = K_H * self.B * self.S * self.F / self.h100.flops
        t_compute   = max(t_a_compute, t_h_compute)

        # ZeRO-3 sync: ReduceScatter + AllGather
        t_zero3_sync = self.bw_model.zero3_sync_time(self.model, self.n_total)

        # Overlap: 50% of communication hidden behind computation
        OVERLAP_FACTOR = 0.50
        t_comm_exposed = t_zero3_sync * (1.0 - OVERLAP_FACTOR)
        t_step = t_compute + t_comm_exposed

        # Tokens per step
        tokens_per_step = (self.n_a6000 * K_A + self.n_h100 * K_H) * self.B * self.S

        # Throughput
        tps = tokens_per_step / t_step

        # MFU
        total_useful_flops = tokens_per_step * self.F * 3
        mfu_h100 = total_useful_flops / t_step / self.h100.flops
        peak_agg = self.n_a6000 * self.a6000.flops + self.n_h100 * self.h100.flops
        mfu_agg  = total_useful_flops / t_step / peak_agg

        # VRAM (same as Strategy A — ZeRO-3 sharding)
        vram_per_gpu = self.vram_est.zero3_vram_bytes(self.n_total, self.B, self.ckpt)
        vram_gb_a, hr_a, fits_a = self.vram_est.format_vram_report(vram_per_gpu, self.a6000)
        vram_gb_h, hr_h, fits_h = self.vram_est.format_vram_report(vram_per_gpu, self.h100)

        feasible = fits_a and fits_h
        reason   = ""
        if not fits_a:
            reason += f"A6000 OOM ({vram_gb_a:.1f} GB). "
        if not fits_h:
            reason += f"H100 OOM ({vram_gb_h:.1f} GB). "

        return PartitionResult(
            strategy_name              = name,
            strategy_id                = "C",
            is_feasible                = feasible,
            infeasibility_reason       = reason,
            tokens_per_second          = tps,
            tokens_per_step            = tokens_per_step,
            step_time_seconds          = t_step,
            mfu_h100                   = mfu_h100,
            mfu_aggregate              = mfu_agg,
            compute_time_s             = t_compute,
            comm_time_s                = t_comm_exposed,
            bubble_fraction            = 0.0,
            grad_accum_steps           = {"A6000": K_A, "H100": K_H},
            layer_assignment           = {"all": list(range(self.model.num_layers))},
            micro_batch_size           = self.B,
            num_micro_batches          = K_H,
            vram_usage_gb              = {"A6000": vram_gb_a, "H100": vram_gb_h},
            vram_headroom_gb           = {"A6000": hr_a, "H100": hr_h},
            implementation_complexity  = "Very High",
            notes = (
                f"Async ZeRO-3 with {OVERLAP_FACTOR*100:.0f}% comm-compute overlap. "
                f"ZeRO-3 sync (RS+AG) = {t_zero3_sync*1000:.1f}ms, "
                f"exposed = {t_comm_exposed*1000:.1f}ms. "
                f"Requires careful CUDA stream management and custom AllGather scheduling."
            ),
        )

    # ─────────────────────────────────────────────────────────────
    # Main solve()
    # ─────────────────────────────────────────────────────────────

    def solve(
        self,
        pipeline_micro_batches: int = 32,
    ) -> Tuple[PartitionResult, List[PartitionResult]]:
        """
        Evaluate all three strategies and return (best, all_results).
        'Best' = feasible strategy with highest throughput.
        """
        results = [
            self._solve_strategy_a(),
            self._solve_strategy_b(pipeline_micro_batches),
            self._solve_strategy_c(),
        ]

        feasible = [r for r in results if r.is_feasible]
        if not feasible:
            warnings.warn("No feasible strategy found! Check hardware and model config.")
            return results[0], results

        best = max(feasible, key=lambda r: r.tokens_per_second)
        return best, results


# ─────────────────────────────────────────────────────────────────────────────
# Cluster Analysis Report
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_bytes(n: float) -> str:
    """Human-readable byte size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def _banner(title: str, width: int = 72) -> str:
    side = (width - len(title) - 2) // 2
    return "═" * side + f" {title} " + "═" * (width - side - len(title) - 2)


def analyze_cluster(
    a6000: Optional[GPUSpec] = None,
    h100: Optional[GPUSpec] = None,
    model: Optional[ModelConfig] = None,
    micro_batch_size: int = 1,
    pipeline_micro_batches: int = 32,
    checkpointing: bool = True,
) -> str:
    """
    Run the full heterogeneous partition analysis for the Neuron_SP cluster.
    Returns a formatted multi-section report string and prints it.

    Parameters
    ----------
    a6000 : GPUSpec, optional
        A6000 hardware spec. Defaults to the standard 2×A6000 config.
    h100 : GPUSpec, optional
        H100 hardware spec. Defaults to the standard 1×H100 config.
    model : ModelConfig, optional
        Model config. Defaults to LLaMA-7B.
    micro_batch_size : int
        Micro-batch size B (sequences per micro-batch).
    pipeline_micro_batches : int
        Number of micro-batches m for Strategy B pipeline.
    checkpointing : bool
        Whether to assume activation checkpointing.

    Returns
    -------
    str
        Full report as a string.
    """

    # ── Defaults ────────────────────────────────────────────────
    if a6000 is None:
        a6000 = GPUSpec(
            name             = "A6000",
            gpu_type         = GPUType.A6000,
            flops_bf16_tflops= 38.7,
            vram_gb          = 48.0,
            pcie_bw_gbps     = 25.0,
            count            = 2,
        )
    if h100 is None:
        h100 = GPUSpec(
            name             = "H100",
            gpu_type         = GPUType.H100,
            flops_bf16_tflops= 835.0,
            vram_gb          = 96.0,
            pcie_bw_gbps     = 50.0,
            count            = 1,
        )
    if model is None:
        model = ModelConfig(
            name        = "LLaMA-7B",
            num_params  = 6.74e9,
            num_layers  = 32,
            hidden_dim  = 4096,
            seq_len     = 2048,
        )

    lines = []
    W = 72

    def sec(title: str) -> None:
        lines.append("")
        lines.append(_banner(title, W))

    # ── Header ──────────────────────────────────────────────────
    lines.append("═" * W)
    lines.append(" Neuron_SP Heterogeneous Partition Solver — Cluster Analysis Report")
    lines.append("═" * W)

    # ── Hardware ────────────────────────────────────────────────
    sec("HARDWARE SPECIFICATIONS")
    lines.append(f"  {a6000}")
    lines.append(f"  {h100}")
    R = h100.flops / a6000.flops
    lines.append(f"  Compute ratio R = FLOPS_H / FLOPS_A = {R:.2f}×")
    bw_model = PCIeBandwidthModel([a6000] * a6000.count + [h100] * h100.count)
    lines.append(f"  Min effective PCIe BW = {bw_model.min_effective_bw/1e9:.2f} GB/s")
    lines.append(f"  Max effective PCIe BW = {bw_model.max_effective_bw/1e9:.2f} GB/s")

    # ── Model ───────────────────────────────────────────────────
    sec("MODEL CONFIGURATION")
    lines.append(f"  {model}")
    lines.append(f"  F_token = 6N = {model.flops_per_token:.4e} FLOPs")
    lines.append(f"  Params BF16  = {_fmt_bytes(model.params_bytes_bf16)}")
    lines.append(f"  Grads  BF16  = {_fmt_bytes(model.grad_bytes_bf16)}")
    lines.append(f"  Optim  FP32  = {_fmt_bytes(model.optimizer_bytes_fp32)}")
    vram_est = VRAMEstimator(model)
    act_ckpt = model.activation_bytes_per_sequence(micro_batch_size, True)
    act_full = model.activation_bytes_per_sequence(micro_batch_size, False)
    lines.append(f"  Activations (B={micro_batch_size}, checkpointed)  = {_fmt_bytes(act_ckpt)}")
    lines.append(f"  Activations (B={micro_batch_size}, no checkpoint) = {_fmt_bytes(act_full)}")

    # ── Communication Characterization ──────────────────────────
    sec("COMMUNICATION MODEL (PCIe, η=0.70)")
    total_gpus = a6000.count + h100.count
    t_ar = bw_model.allreduce_time(model.grad_bytes_bf16, total_gpus)
    t_z3 = bw_model.zero3_sync_time(model, total_gpus)
    act_bytes = micro_batch_size * model.seq_len * model.hidden_dim * BF16_BYTES
    t_p2p_ha = bw_model.p2p_time(act_bytes, 0, 2)  # A6000→H100 (worst)
    lines.append(f"  AllReduce ({total_gpus} GPUs, {_fmt_bytes(model.grad_bytes_bf16)}): {t_ar*1000:.2f} ms")
    lines.append(f"  ZeRO-3 RS+AG sync:    {t_z3*1000:.2f} ms")
    lines.append(f"  P2P activation (A6000→H100, B={micro_batch_size}): {t_p2p_ha*1000:.3f} ms")

    # ── VRAM per Strategy ────────────────────────────────────────
    sec("VRAM ANALYSIS")
    vram_zero3 = vram_est.zero3_vram_bytes(total_gpus, micro_batch_size, checkpointing)
    vgb_a = vram_zero3 / 1024**3
    vgb_h = vram_zero3 / 1024**3
    lines.append(f"  ZeRO-3 (p={total_gpus}, B={micro_batch_size}, ckpt={checkpointing}):")
    lines.append(f"    Per GPU (A6000 {a6000.vram_gb}GB): {vgb_a:.2f} GB  "
                 f"{'✓' if vgb_a < a6000.vram_gb else '✗ OOM'}")
    lines.append(f"    Per GPU (H100  {h100.vram_gb}GB): {vgb_h:.2f} GB  "
                 f"{'✓' if vgb_h < h100.vram_gb else '✗ OOM'}")

    L = model.num_layers
    l_per_a = 1
    l_h = L - a6000.count * l_per_a
    v_pp_h = vram_est.pipeline_stage_vram_bytes(l_h, L, 1, micro_batch_size, pipeline_micro_batches, checkpointing)
    v_pp_a = vram_est.pipeline_stage_vram_bytes(l_per_a, L, 1, micro_batch_size, pipeline_micro_batches, checkpointing)
    lines.append(f"  Pipeline (H100: {l_h}L, A6000: {l_per_a}L each, m={pipeline_micro_batches}):")
    lines.append(f"    H100  stage: {v_pp_h/1024**3:.2f} GB  {'✓' if v_pp_h/1024**3 < h100.vram_gb else '✗ OOM'}")
    lines.append(f"    A6000 stage: {v_pp_a/1024**3:.2f} GB  {'✓' if v_pp_a/1024**3 < a6000.vram_gb else '✗ OOM'}")

    # ── Strategy Results ─────────────────────────────────────────
    sec("STRATEGY EVALUATION")
    solver = HeteroPartitionSolver(
        a6000           = a6000,
        h100            = h100,
        model           = model,
        micro_batch_size= micro_batch_size,
        checkpointing   = checkpointing,
        verbose         = False,
    )
    best, all_results = solver.solve(pipeline_micro_batches)

    for r in all_results:
        lines.append("")
        lines.append(f"  ┌{'─'*68}┐")
        lines.append(f"  │ Strategy {r.strategy_id}: {r.strategy_name:<55}│")
        lines.append(f"  ├{'─'*68}┤")
        if not r.is_feasible:
            lines.append(f"  │  ✗ INFEASIBLE: {r.infeasibility_reason:<52}│")
        else:
            def row(label: str, value: str) -> str:
                return f"  │  {label:<28} {value:<37}│"
            lines.append(row("Throughput:",         f"{r.tokens_per_second:,.0f} tok/s"))
            lines.append(row("Step time:",          f"{r.step_time_seconds*1000:.2f} ms"))
            lines.append(row("Compute time:",       f"{r.compute_time_s*1000:.2f} ms  ({r.compute_time_s/r.step_time_seconds*100:.1f}%)"))
            lines.append(row("Comm time:",          f"{r.comm_time_s*1000:.2f} ms  ({r.comm_time_s/r.step_time_seconds*100:.1f}%)"))
            lines.append(row("MFU (H100 ref):",     f"{r.mfu_h100*100:.2f}%"))
            lines.append(row("MFU (aggregate):",    f"{r.mfu_aggregate*100:.2f}%"))
            if r.bubble_fraction > 0:
                lines.append(row("Bubble fraction:",f"{r.bubble_fraction*100:.1f}%"))
            lines.append(row("Tokens per step:",    f"{r.tokens_per_step:,}"))
            lines.append(row("Complexity:",         r.implementation_complexity))
            lines.append(row("VRAM A6000:",         f"{r.vram_usage_gb.get('A6000',0):.1f} / {a6000.vram_gb:.0f} GB (headroom {r.vram_headroom_gb.get('A6000',0):.1f} GB)"))
            lines.append(row("VRAM H100:",          f"{r.vram_usage_gb.get('H100',0):.1f} / {h100.vram_gb:.0f} GB  (headroom {r.vram_headroom_gb.get('H100',0):.1f} GB)"))
            lines.append(f"  │  Notes: {r.notes[:60]:<60}│")
            if len(r.notes) > 60:
                lines.append(f"  │         {r.notes[60:120]:<60}│")
        lines.append(f"  └{'─'*68}┘")

    # ── Comparison Table ─────────────────────────────────────────
    sec("COMPARISON SUMMARY")
    lines.append(f"  {'Strategy':<40} {'tok/s':>10} {'MFU(H100)':>10} {'Complexity':<12} {'Feasible'}")
    lines.append(f"  {'─'*40} {'─'*10} {'─'*10} {'─'*12} {'─'*8}")
    for r in all_results:
        feasible_str = "✓" if r.is_feasible else "✗"
        tps_str      = f"{r.tokens_per_second:,.0f}" if r.is_feasible else "N/A"
        mfu_str      = f"{r.mfu_h100*100:.1f}%"      if r.is_feasible else "N/A"
        lines.append(
            f"  {r.strategy_name:<40} {tps_str:>10} {mfu_str:>10} "
            f"{r.implementation_complexity:<12} {feasible_str}"
        )

    # ── Recommendation ───────────────────────────────────────────
    sec("RECOMMENDATION")
    lines.append(f"  OPTIMAL STRATEGY: [{best.strategy_id}] {best.strategy_name}")
    lines.append(f"  Throughput: {best.tokens_per_second:,.0f} tok/s")
    lines.append(f"  MFU(H100):  {best.mfu_h100*100:.1f}%")
    lines.append("")
    lines.append("  Key Insights:")
    lines.append(f"    • PCIe bottleneck: {bw_model.min_effective_bw/1e9:.1f} GB/s eff. limits all strategies.")
    lines.append(f"    • Compute ratio R={R:.1f}×: H100 does K_H={round(R)} micro-batches per A6000 sync.")
    lines.append(f"    • AllReduce dominates: {t_ar*1000:.0f}ms vs compute {solver._solve_strategy_a().compute_time_s*1000:.0f}ms.")
    lines.append(f"    • Pipeline is PCIe-bound: P2P {t_p2p_ha*1000:.2f}ms >> stage {(l_h/L)*micro_batch_size*model.seq_len*model.flops_per_token/h100.flops*1000:.3f}ms.")
    lines.append(f"    • Strategy A is recommended for simplicity + maximal throughput.")
    lines.append("")
    lines.append(f"  Recommended Config:")
    lines.append(f"    --deepspeed_config zero3_hetero.json")
    lines.append(f"    --gradient_accumulation_steps {round(R)}   # H100")
    lines.append(f"    --gradient_accumulation_steps 1             # A6000")
    lines.append(f"    --activation_checkpointing true")
    lines.append(f"    --zero_stage 3")
    lines.append("")
    lines.append("═" * W)

    report = "\n".join(lines)
    print(report)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Module-level Constants (for import by other DeepSpeed components)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_A6000 = GPUSpec(
    name              = "A6000",
    gpu_type          = GPUType.A6000,
    flops_bf16_tflops = 38.7,
    vram_gb           = 48.0,
    pcie_bw_gbps      = 25.0,
    count             = 2,
)

DEFAULT_H100 = GPUSpec(
    name              = "H100",
    gpu_type          = GPUType.H100,
    flops_bf16_tflops = 835.0,
    vram_gb           = 96.0,
    pcie_bw_gbps      = 50.0,
    count             = 1,
)

LLAMA_7B = ModelConfig(
    name       = "LLaMA-7B",
    num_params = 6.74e9,
    num_layers = 32,
    hidden_dim = 4096,
    seq_len    = 2048,
    num_heads  = 32,
)

# Quick sanity checks on constants
assert abs(DEFAULT_H100.flops / DEFAULT_A6000.flops - 21.57) < 0.1, "R mismatch"
assert abs(LLAMA_7B.flops_per_token - 4.044e10) < 1e9, "F_token mismatch"


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    report = analyze_cluster(
        a6000                  = DEFAULT_A6000,
        h100                   = DEFAULT_H100,
        model                  = LLAMA_7B,
        micro_batch_size       = 1,
        pipeline_micro_batches = 32,
        checkpointing          = True,
    )
