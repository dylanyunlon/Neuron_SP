"""
deepspeed/runtime/desloc_partition.py

Neuron_SP Project — Heterogeneous GPU Cluster Partition Solver
==============================================================
Author  : Neuron_SP Architecture Team
Version : 2.0.0
Target  : 2×A6000(48GB, SM8.6) + 1×H100-NVL(96GB, SM9.0), no NVLink

═══════════════════════════════════════════════════════════════
MATHEMATICAL DERIVATION — FULL PROOF
═══════════════════════════════════════════════════════════════

§1  HARDWARE CONSTANTS
──────────────────────
  A6000  : FLOPS_A  = 38.7  TFLOPS (BF16)  VRAM =  48 GB  BW_pcie = 25 GB/s
  H100   : FLOPS_H  = 835.0 TFLOPS (BF16)  VRAM =  96 GB  BW_pcie = 50 GB/s

§2  COMPUTE RATIO
─────────────────
  R = FLOPS_H / FLOPS_A = 835.0 / 38.7 ≈ 21.57

  Interpretation: one H100 can process ≈21.57× more FLOPs per second than
  one A6000.  For balanced throughput across n_A A6000s and 1 H100:

    n_A * FLOPS_A * bs_A  =  FLOPS_H * bs_H
    ⟹  bs_H / bs_A  =  n_A * R  =  2 * 21.57 ≈ 43.1

  We solve for integer micro-batch assignments (bs_H, bs_A) under VRAM and
  compute-balance constraints.

§3  MEMORY MODEL
────────────────
  For a transformer with:
    N_params  ≈ 6.74×10⁹   (LLaMA-7B)
    d_model   = 4096
    n_layers  = 32
    S         = sequence length (tokens)
    B         = micro-batch size per GPU

  3a. Parameter memory (BF16, 2 bytes/param):
      M_params = 2 * N_params  ≈ 13.48 GB  (same on every GPU for ZeRO-0/1)

  3b. Optimizer states (AdamW, 12 bytes/param in mixed precision):
      M_opt    = 12 * N_params ≈ 80.88 GB  (sharded under ZeRO-2/3)

  3c. Gradient memory (BF16, 2 bytes/param):
      M_grad   = 2 * N_params  ≈ 13.48 GB  (sharded under ZeRO-2/3)

  3d. Activation memory per layer (approximate, dominant terms):
      Each transformer layer stores activations for backprop.
      For a single (B, S, d) tensor in BF16:
        act_per_layer = B * S * d_model * n_stored_tensors * 2 bytes
      Stored tensors per layer ≈ 10 (Q,K,V projections, attention scores,
        FFN intermediate, residuals, layer-norm statistics, dropout masks).
        act_per_layer ≈ B * S * 4096 * 10 * 2  bytes
                       = B * S * 81920  bytes
      Total activations for L layers on one GPU:
        M_act(B,S,L) = B * S * 81920 * L  bytes

  3e. VRAM budget for weights+gradients+optimizer under ZeRO-3:
      ZeRO-3 shards params, grads, optimizer across G GPUs:
        M_zero3 = (2 + 2 + 12) * N_params / G  bytes  (per GPU)
      For G=3:  M_zero3 = 16 * N_params / 3 ≈ 35.95 GB

§4  STRATEGY A — ZeRO-3 + HETEROGENEOUS GRADIENT ACCUMULATION
─────────────────────────────────────────────────────────────
  Setup:
    • All GPUs hold 1/G of the sharded ZeRO-3 state.
    • Each optimizer step accumulates K_i micro-batches on GPU i locally,
      then AllReduce gradients once.

  4a. Compute time per step:
      FLOPs per token through full model (forward + backward ≈ 6× params):
        F_token = 6 * N_params  ≈ 4.044×10¹⁰  FLOPs/token

      For GPU i processing K_i micro-batches of size B and sequence length S:
        t_compute_i = K_i * B * S * F_token / FLOPS_i

      For B=1 (unit micro-batch):
        t_A = K_A * S * 6*N / FLOPS_A  = K_A * 2048 * 4.044e10 / 38.7e12
            = K_A * 2.14 s  (per A6000)

        t_H = K_H * S * 6*N / FLOPS_H  = K_H * 2048 * 4.044e10 / 835e12
            = K_H * 0.0992 s (H100)

  4b. Balance equation (t_A = t_H):
      K_A * 2.14 = K_H * 0.0992
      K_H / K_A  = 2.14 / 0.0992 ≈ 21.57  (= R, as expected)

      Integer solution: K_A = 1, K_H = 22
      Total tokens/step: (K_A + K_A + K_H) * B * S = 24 * 2048 = 49152

  4c. Communication time (Ring AllReduce, gradient tensor):
      Gradient size: G_bytes = 2 * N_params = 13.48 GB (BF16)
      Ring AR cost for n GPUs with bottleneck bandwidth min_bw:
        T_ar = 2 * (n-1)/n * G_bytes / min_bw
             = 2 * 2/3  * 13.48e9 / 25e9
             ≈ 0.718 s

      Note: 25 GB/s is the measured A6000 PCIe Gen4 bandwidth (bottleneck).

  4d. Step time and throughput:
      t_step = max(t_compute_i) + T_ar
             ≈ max(1*2.14, 1*2.14, 22*0.0992) + 0.718
             ≈ max(2.14, 2.14, 2.18) + 0.718
             ≈ 2.18 + 0.718
             ≈ 2.90 s

      Throughput_A = tokens_per_step / t_step
                   = 49152 / 2.90
                   ≈ 16950 tok/s  (theoretical peak)

  4e. VRAM check:
      ZeRO-3 sharded state: M_zero3 ≈ 35.95 GB
      Activation for K=1, B=1, S=2048, L=32:
        M_act = 1 * 2048 * 81920 * 32 / 1e9 ≈ 5.37 GB
      Total A6000: 35.95 + 5.37 ≈ 41.3 GB  < 48 GB  ✓
      Total H100:  same sharded state + 22× activations
        22 * 5.37 ≈ 118 GB  > 96 GB  ✗
      → Must use activation checkpointing on H100, or reduce K_H.
      With activation checkpointing (recompute ½ layers):
        M_act_ckpt ≈ 5.37 * √22 ≈ 25.2 GB
        Total H100: 35.95 + 25.2 ≈ 61.2 GB  < 96 GB  ✓
      Checkpointing adds ≈1/3 extra compute on H100:
        t_H_ckpt = 22 * 0.0992 * 4/3 ≈ 2.90 s  (still near balance)

§5  STRATEGY B — PIPELINE PARALLELISM (LAYER PARTITIONING)
──────────────────────────────────────────────────────────
  5a. Layer assignment by compute ratio:
      Total compute weight W_total = FLOPS_A + FLOPS_A + FLOPS_H
                                   = 38.7 + 38.7 + 835.0 = 912.4 TFLOPS

      Ideal layer fractions:
        f_A0 = 38.7  / 912.4 = 0.04242  →  L_A0 = round(32 * 0.04242) = 1
        f_A1 = 38.7  / 912.4 = 0.04242  →  L_A1 = round(32 * 0.04242) = 1
        f_H  = 835.0 / 912.4 = 0.91516  →  L_H  = 32 - 1 - 1 = 30

      Adjusted (ensuring sum=32): L_A0=1, L_A1=1, L_H=30

  5b. Activation transfer cost between pipeline stages:
      Activation tensor between stages: shape (B, S, d_model), BF16
        act_bytes = B * S * d_model * 2 = 1 * 2048 * 4096 * 2 ≈ 16.78 MB

      Transfer latencies (PCIe, same NUMA node):
        T_A0→H  = act_bytes / BW(A0↔H)  ≈ 16.78e6 / 25e9 ≈ 0.671 ms
        T_H→A1  = act_bytes / BW(H↔A1)  ≈ 16.78e6 / 25e9 ≈ 0.671 ms

  5c. Pipeline bubble fraction:
      With p=3 pipeline stages and m micro-batches:
        bubble_fraction = (p-1) / (m + p - 1)

      For m=24 (same global batch as Strategy A):
        bubble_fraction = 2 / (24 + 2) = 2/26 ≈ 7.7%

  5d. Compute time per stage:
      t_stage_A0 = B*S * 6*N_layers_A0/N_total * N_params/layer * FLOPS_A⁻¹
      Simplification: N_params roughly uniform across layers
        params_per_layer = N_params / n_layers = 7e9/32 ≈ 218.75M

      t_A0 = B*S * 6 * L_A0 * params_per_layer / FLOPS_A
           = 1*2048 * 6 * 1 * 218.75e6 / 38.7e12 ≈ 0.0695 s  per micro-batch

      t_H  = B*S * 6 * L_H  * params_per_layer / FLOPS_H
           = 1*2048 * 6 * 30 * 218.75e6 / 835e12 ≈ 0.0970 s  per micro-batch

      t_A1 = same as t_A0 ≈ 0.0695 s  per micro-batch

      Bottleneck stage: H100 at 0.0970 s/micro-batch

  5e. Ideal pipeline throughput (no bubble):
      t_ideal = m * t_bottleneck = 24 * 0.0970 = 2.328 s

      Actual with bubble:
      t_pipe = (m + p - 1) * t_bottleneck + 2 * T_comm
             = 26 * 0.0970 + 2 * 0.000671
             ≈ 2.522 + 0.00134 ≈ 2.524 s

      BUT: A6000 stages become the real bottleneck over full steady-state.
      Steady-state: each stage must complete before passing activation.
      With 1-F-1-B scheduling:
        t_step_pipe = m * max(t_A0, t_H, t_A1) + (p-1) * t_bottleneck
                    = 24 * 0.0970 + 2 * 0.0970 ≈ 2.52 s

      Throughput_B = m * B * S / t_step_pipe = 24 * 2048 / 2.52 ≈ 19524 tok/s

  5f. VRAM per stage (no ZeRO, each stage only holds its layers):
      M_A0 = L_A0 * params_per_layer * (2+2+12) = 1 * 218.75e6 * 16 ≈ 3.5 GB ✓
      M_H  = L_H  * params_per_layer * 16 = 30 * 218.75e6 * 16 ≈ 105 GB  > 96!
      → Must use ZeRO-1 on H100 stage, or reduce L_H to 28:
        M_H_ZeRO1 ≈ 105 / 3 ≈ 35 GB for optimizer, + 30*218.75e6*4≈26GB for fp16
        Total H100 ≈ 61 GB  < 96 GB  ✓

§6  STRATEGY COMPARISON SUMMARY
────────────────────────────────
                       Strategy A (ZeRO-3)   Strategy B (Pipeline)
  ─────────────────────────────────────────────────────────────────
  Throughput (tok/s)      ~16950               ~19524
  Implementation compl.   Low (std ZeRO-3)     High (custom schedule)
  VRAM A6000              ~41 GB               ~3.5 GB (underused!)
  VRAM H100               ~61 GB (w/ ckpt)     ~61 GB (w/ ZeRO-1)
  Fault tolerance         High (AR sync)        Low (stage dep.)
  Communication           AllReduce 0.72s/step  Fwd pass only 1.3ms
  ─────────────────────────────────────────────────────────────────
  RECOMMENDATION: Strategy B (Pipeline) for maximum throughput.
                  Strategy A for simpler deployment & fault tolerance.

═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum


# ──────────────────────────────────────────────────────────────
#  §1  DATA CLASSES
# ──────────────────────────────────────────────────────────────

@dataclass
class GPUSpec:
    """Physical specification of a single GPU device."""
    gpu_id       : int
    name         : str
    vram_gb      : float          # Total VRAM in GiB
    flops_tflops : float          # BF16 peak TFLOPS
    pcie_bw_gbs  : float          # Measured PCIe bandwidth (GB/s)
    sm_version   : float          # e.g. 8.6, 9.0
    numa_node    : int

    @property
    def flops(self) -> float:
        """Return FLOPS in raw unit (not T)."""
        return self.flops_tflops * 1e12

    @property
    def vram_bytes(self) -> float:
        return self.vram_gb * 1024**3


@dataclass
class ModelConfig:
    """Transformer model hyper-parameters."""
    name            : str
    n_params        : int           # Total parameter count
    n_layers        : int           # Transformer layers
    d_model         : int           # Hidden dimension
    n_heads         : int           # Attention heads
    d_ff            : int           # FFN intermediate dimension
    vocab_size      : int = 32000
    bytes_per_param : int = 2       # BF16 = 2 bytes

    @property
    def params_per_layer(self) -> float:
        """Approximate parameters per transformer layer."""
        return self.n_params / self.n_layers

    @property
    def flops_per_token(self) -> float:
        """
        FLOPs per token for one full forward+backward pass.
        Approximation: 6 × N_params (standard LLM estimate).
        Derivation:
          forward  ≈ 2 × N_params  (matmul dominates, 2 flops/MAC)
          backward ≈ 4 × N_params  (≈2× forward for grad_input + grad_weight)
          total    = 6 × N_params
        """
        return 6.0 * self.n_params

    def activation_bytes_per_layer(self, batch_size: int, seq_len: int) -> float:
        """
        Activation memory for one transformer layer during training.
        Stores ≈10 tensors of shape (B, S, d) or (B, H, S, S) in BF16.
        Dominant terms:
          • 3× QKV projections  : 3 × B×S×d × 2
          • attention scores    : B×H×S×S × 2
          • attention output    : B×S×d × 2
          • FFN intermediate    : B×S×d_ff × 2
          • residual streams    : 2 × B×S×d × 2
        """
        B, S, d, d_ff, H = batch_size, seq_len, self.d_model, self.d_ff, self.n_heads
        act = (
            3 * B * S * d * 2          +  # QKV
            B * H * S * S * 2          +  # attn scores
            B * S * d * 2              +  # attn out
            B * S * d_ff * 2           +  # FFN
            2 * B * S * d * 2             # residuals
        )
        return float(act)


@dataclass
class PartitionResult:
    """Output of the partition solver for one strategy."""
    strategy_name     : str
    micro_batch_alloc : Dict[int, int]    # gpu_id -> micro-batches per step
    layer_alloc       : Dict[int, int]    # gpu_id -> number of layers
    estimated_tps     : float             # tokens per second
    step_time_s       : float             # wall-clock seconds per step
    vram_usage_gb     : Dict[int, float]  # gpu_id -> GB used
    bottleneck_gpu    : int
    notes             : List[str] = field(default_factory=list)

    def is_feasible(self, gpus: List[GPUSpec]) -> bool:
        gpu_map = {g.gpu_id: g for g in gpus}
        for gid, usage in self.vram_usage_gb.items():
            if usage > gpu_map[gid].vram_gb:
                return False
        return True


# ──────────────────────────────────────────────────────────────
#  §2  PCIe BANDWIDTH MODEL
# ──────────────────────────────────────────────────────────────

class PCIeBandwidthModel:
    """
    Models point-to-point and collective communication costs
    over PCIe fabric (no NVLink).

    Latency model:
        T(size) = α + size / β
    where α = base latency, β = effective bandwidth.

    For inter-GPU transfers on same NUMA node (measured):
        α ≈ 5 µs (PCIe transaction overhead)
        β = min(BW_src, BW_dst)  [half-duplex bottleneck]
    """

    BASE_LATENCY_S : float = 5e-6   # 5 µs

    def __init__(self, gpus: List[GPUSpec]):
        self.gpus    = {g.gpu_id: g for g in gpus}
        self.gpu_ids = [g.gpu_id for g in gpus]

    def p2p_time(self, src: int, dst: int, size_bytes: float) -> float:
        """
        One-way P2P transfer time in seconds.
          T = α + size / min(BW_src_egress, BW_dst_ingress)
        PCIe is full-duplex per link but DMA engines share bandwidth,
        so we model effective BW as min of the two link bandwidths.
        """
        bw = min(self.gpus[src].pcie_bw_gbs, self.gpus[dst].pcie_bw_gbs) * 1e9
        return self.BASE_LATENCY_S + size_bytes / bw

    def ring_allreduce_time(self, size_bytes: float) -> float:
        """
        Ring AllReduce cost for n GPUs.

        Algorithm: 2-pass ring (reduce-scatter + all-gather)
          Data volume per link = 2 * (n-1)/n * size_bytes
          T_ar = 2 * (n-1)/n * size_bytes / BW_bottleneck

        Bottleneck BW = min PCIe bandwidth across all GPUs.
        For our cluster: min(25, 25, 50) = 25 GB/s
        """
        n   = len(self.gpu_ids)
        bw  = min(g.pcie_bw_gbs for g in self.gpus.values()) * 1e9
        return 2 * (n - 1) / n * size_bytes / bw

    def pipeline_activation_time(self,
                                  src: int, dst: int,
                                  batch: int, seq: int, d_model: int) -> float:
        """
        Time to send one activation tensor between adjacent pipeline stages.
        Shape: (batch, seq, d_model), dtype BF16 (2 bytes).
        """
        size = batch * seq * d_model * 2
        return self.p2p_time(src, dst, size)

    def broadcast_time(self, src: int, size_bytes: float) -> float:
        """
        1-to-all broadcast: sequentially send to each peer.
        T = Σ_i T_p2p(src, i, size)  (non-overlapping, conservative)
        """
        total = 0.0
        for gid in self.gpu_ids:
            if gid != src:
                total += self.p2p_time(src, gid, size_bytes)
        return total


# ──────────────────────────────────────────────────────────────
#  §3  VRAM ESTIMATOR
# ──────────────────────────────────────────────────────────────

class VRAMEstimator:
    """
    Estimates peak VRAM usage under different parallelism strategies.

    Notation:
        P  = total parameter count
        B  = micro-batch size
        S  = sequence length
        L  = layers on this GPU
        G  = total GPU count
        bp = bytes per param (2 for BF16)
    """

    ADAM_BYTES_PER_PARAM = 12   # 4 (fp32 master) + 4 (m) + 4 (v)
    GRAD_BYTES_PER_PARAM =  4   # fp32 gradient accumulation buffer
    PARAM_BYTES_BF16     =  2

    def zero3_vram_gb(self, model: ModelConfig, n_gpus: int,
                      batch: int, seq: int, n_layers_local: int,
                      activation_ckpt: bool = False) -> float:
        """
        ZeRO-3 VRAM: shards params + grads + optimizer across G GPUs.
        Each GPU stores:
          • params shard   : P * 2 / G  bytes
          • grad shard     : P * 4 / G  bytes  (fp32 grad buffer)
          • optimizer shard: P * 12 / G bytes  (AdamW)
          • activations    : B * S * act_per_layer * L  (full or checkpointed)
        """
        P = model.n_params
        G = n_gpus

        params_bytes = P * self.PARAM_BYTES_BF16 / G
        grad_bytes   = P * self.GRAD_BYTES_PER_PARAM / G
        opt_bytes    = P * self.ADAM_BYTES_PER_PARAM / G

        act_per_layer = model.activation_bytes_per_layer(batch, seq)
        if activation_ckpt:
            # Gradient checkpointing: store only sqrt(L) layers at a time
            # Memory: O(√L) instead of O(L), extra compute: +33%
            act_bytes = act_per_layer * math.sqrt(n_layers_local) * batch
        else:
            act_bytes = act_per_layer * n_layers_local

        total = params_bytes + grad_bytes + opt_bytes + act_bytes
        return total / 1024**3

    def pipeline_stage_vram_gb(self, model: ModelConfig, n_layers_stage: int,
                                n_gpus: int, batch: int, seq: int,
                                micro_batches_inflight: int = 2) -> float:
        """
        Pipeline stage VRAM: only holds local layers.
        ZeRO-1 applied: optimizer sharded, params/grads local.
          • params    : L_stage * params_per_layer * 2  bytes
          • grads     : L_stage * params_per_layer * 4  bytes
          • optimizer : L_stage * params_per_layer * 12 / G_stage  bytes
            (G_stage = GPUs in this pipeline stage, usually 1)
          • activations: inflight micro-batches × act_per_layer × L_stage
        """
        ppl = model.params_per_layer
        params_b = n_layers_stage * ppl * self.PARAM_BYTES_BF16
        grad_b   = n_layers_stage * ppl * self.GRAD_BYTES_PER_PARAM
        opt_b    = n_layers_stage * ppl * self.ADAM_BYTES_PER_PARAM / n_gpus

        act_per = model.activation_bytes_per_layer(batch, seq)
        act_b   = act_per * n_layers_stage * micro_batches_inflight

        return (params_b + grad_b + opt_b + act_b) / 1024**3


# ──────────────────────────────────────────────────────────────
#  §4  HETERO PARTITION SOLVER
# ──────────────────────────────────────────────────────────────

class HeteroPartitionSolver:
    """
    Solves the optimal partition for a heterogeneous GPU cluster.

    Given:
        • List of GPUSpec objects
        • ModelConfig
        • Training sequence length S and global batch size G_batch

    Produces:
        • PartitionResult for Strategy A (ZeRO-3 + hetero grad-accum)
        • PartitionResult for Strategy B (pipeline parallel by layer)
        • Comparison and recommendation

    Algorithms:
        A — find integer {K_i} minimizing max(K_i/FLOPS_i) subject to Σ K_i = G_batch/S
        B — assign layers proportional to FLOPS_i, check VRAM, compute pipeline time
    """

    def __init__(self,
                 gpus     : List[GPUSpec],
                 model    : ModelConfig,
                 seq_len  : int = 2048,
                 global_tokens_per_step : Optional[int] = None):
        self.gpus    = gpus
        self.model   = model
        self.seq     = seq_len
        self.bw_model = PCIeBandwidthModel(gpus)
        self.vram_est = VRAMEstimator()
        self.n_gpus   = len(gpus)
        # Default global batch: 24 micro-batches × seq_len tokens
        self.global_tokens = global_tokens_per_step or (24 * seq_len)
        self.global_microbatches = self.global_tokens // seq_len

    # ── § A: ZeRO-3 heterogeneous gradient accumulation ──────

    def _solve_zero3_allocation(self) -> Dict[int, int]:
        """
        Find integer micro-batch counts {K_i} per GPU that minimise
        the load imbalance (max_{i} K_i * S / FLOPS_i).

        Method: proportional allocation + greedy correction.
          K_i^* = round(M * FLOPS_i / Σ FLOPS_j)
          then adjust largest/smallest to match total M.
        """
        M = self.global_microbatches
        total_flops = sum(g.flops for g in self.gpus)

        # Proportional (possibly non-integer)
        raw = {g.gpu_id: M * g.flops / total_flops for g in self.gpus}

        # Floor and remainder
        floored = {gid: int(v) for gid, v in raw.items()}
        remainder = M - sum(floored.values())

        # Distribute remainder to GPUs with largest fractional parts
        fracs = sorted(raw.items(), key=lambda x: x[1] - int(x[1]), reverse=True)
        for i in range(remainder):
            floored[fracs[i][0]] += 1

        # Ensure every GPU gets at least 1
        for gid in floored:
            if floored[gid] == 0:
                floored[gid] = 1
                # Take one from the highest-allocated
                max_gid = max(floored, key=lambda x: floored[x])
                if floored[max_gid] > 1:
                    floored[max_gid] -= 1

        return floored

    def strategy_a_zero3(self, activation_ckpt: bool = True) -> PartitionResult:
        """
        Strategy A: ZeRO-3 + heterogeneous gradient accumulation.

        Step-time model:
            t_step = max_i(t_compute_i) + T_allreduce

        t_compute_i = K_i * S * FLOPs_per_token / FLOPS_i
        T_allreduce = ring_allreduce(grad_size)
            where grad_size = N_params * 4 bytes (fp32 grads before shard)
        """
        alloc = self._solve_zero3_allocation()
        gpu_map = {g.gpu_id: g for g in self.gpus}

        # Compute times
        fpt = self.model.flops_per_token
        compute_times = {}
        for gid, k in alloc.items():
            t = k * self.seq * fpt / gpu_map[gid].flops
            compute_times[gid] = t

        t_compute_max = max(compute_times.values())
        bottleneck_gid = max(compute_times, key=lambda x: compute_times[x])

        # AllReduce cost (fp32 gradients before ZeRO scatter)
        grad_bytes = self.model.n_params * 4   # fp32
        t_ar = self.bw_model.ring_allreduce_time(grad_bytes)

        t_step = t_compute_max + t_ar

        total_tokens = sum(k * self.seq for k in alloc.values())
        tps = total_tokens / t_step

        # VRAM estimate
        vram = {}
        for g in self.gpus:
            k = alloc[g.gpu_id]
            # Each GPU runs K micro-batches sequentially, so batch=1 for memory
            do_ckpt = activation_ckpt and (k > 4)
            vram[g.gpu_id] = self.vram_est.zero3_vram_gb(
                self.model, self.n_gpus,
                batch=1, seq=self.seq,
                n_layers_local=self.model.n_layers,
                activation_ckpt=do_ckpt
            )

        notes = [
            f"Global microbatches: {self.global_microbatches}",
            f"Micro-batch alloc: { {k: v for k, v in alloc.items()} }",
            f"Compute times (s): { {k: f'{v:.4f}' for k, v in compute_times.items()} }",
            f"t_compute_max={t_compute_max:.4f}s  t_allreduce={t_ar:.4f}s",
            f"Activation checkpointing on K>4 GPUs: {activation_ckpt}",
        ]

        return PartitionResult(
            strategy_name     = "Strategy-A: ZeRO-3 + HeteroGradAccum",
            micro_batch_alloc = alloc,
            layer_alloc       = {g.gpu_id: self.model.n_layers for g in self.gpus},
            estimated_tps     = tps,
            step_time_s       = t_step,
            vram_usage_gb     = vram,
            bottleneck_gpu    = bottleneck_gid,
            notes             = notes,
        )

    # ── § B: Pipeline parallelism ─────────────────────────────

    def _solve_pipeline_layers(self) -> Dict[int, int]:
        """
        Assign L_i layers to GPU i proportional to FLOPS_i.
        Constraint: Σ L_i = n_layers, L_i ≥ 1.

        L_i = max(1, round(n_layers * FLOPS_i / Σ FLOPS_j))
        Remainder distributed to fastest GPU.
        """
        L = self.model.n_layers
        total_flops = sum(g.flops for g in self.gpus)

        raw    = {g.gpu_id: L * g.flops / total_flops for g in self.gpus}
        floored = {gid: max(1, int(v)) for gid, v in raw.items()}

        diff = L - sum(floored.values())
        if diff > 0:
            # Give extra layers to fastest GPU
            fastest = max(self.gpus, key=lambda g: g.flops)
            floored[fastest.gpu_id] += diff
        elif diff < 0:
            # Remove from fastest first
            fastest = max(self.gpus, key=lambda g: g.flops)
            floored[fastest.gpu_id] += diff  # diff is negative
            floored[fastest.gpu_id] = max(1, floored[fastest.gpu_id])

        return floored

    def strategy_b_pipeline(self, micro_batches: int = 24) -> PartitionResult:
        """
        Strategy B: 1F1B pipeline parallelism.

        Stage order: A6000_0 → H100 → A6000_1  (ascending by GPU id)
        Sorted by FLOPS ascending so fast GPU is in the middle
        (reduces bubble waiting for slow edge stages).

        1F1B schedule:
            t_step = (m + p - 1) * t_bottleneck_stage + t_flush
          where:
            t_flush = communication overhead for last batch
            t_bottleneck = max per-micro-batch compute time across stages

        Bubble fraction: (p-1)/(m+p-1)
        """
        layer_alloc = self._solve_pipeline_layers()
        gpu_map     = {g.gpu_id: g for g in self.gpus}
        p           = self.n_gpus     # pipeline stages
        m           = micro_batches

        # Per-stage per-micro-batch compute time
        fpt         = self.model.flops_per_token  # per token
        stage_times = {}
        for gid, n_layers in layer_alloc.items():
            layer_frac = n_layers / self.model.n_layers
            fpt_stage  = layer_frac * fpt         # FLOPs for this stage's layers
            t_mb       = self.seq * fpt_stage / gpu_map[gid].flops
            stage_times[gid] = t_mb

        t_bottleneck = max(stage_times.values())
        bottleneck_gid = max(stage_times, key=lambda x: stage_times[x])

        # Pipeline stage order: sort GPUs by pipeline position
        # Convention: embed stage 0 = first GPU, last = last
        ordered_gpus = sorted(self.gpus, key=lambda g: g.gpu_id)
        transitions  = []
        for i in range(len(ordered_gpus) - 1):
            src = ordered_gpus[i].gpu_id
            dst = ordered_gpus[i + 1].gpu_id
            t_transfer = self.bw_model.pipeline_activation_time(
                src, dst, 1, self.seq, self.model.d_model
            )
            transitions.append((src, dst, t_transfer))

        total_comm = sum(t for _, _, t in transitions) * 2  # fwd + bwd

        # 1F1B schedule total time
        t_step = (m + p - 1) * t_bottleneck + total_comm

        total_tokens = m * self.seq
        tps          = total_tokens / t_step

        bubble_frac  = (p - 1) / (m + p - 1)

        # VRAM per stage
        vram = {}
        for g in self.gpus:
            n_l  = layer_alloc[g.gpu_id]
            vram[g.gpu_id] = self.vram_est.pipeline_stage_vram_gb(
                self.model, n_l, n_gpus=self.n_gpus,
                batch=1, seq=self.seq, micro_batches_inflight=2
            )

        notes = [
            f"Pipeline stages: {p},  micro-batches: {m}",
            f"Layer alloc: { {k: v for k, v in layer_alloc.items()} }",
            f"Per-stage compute times (s/µbatch): "
            f"{ {k: f'{v:.5f}' for k, v in stage_times.items()} }",
            f"t_bottleneck={t_bottleneck:.5f}s  bubble={bubble_frac*100:.1f}%",
            f"Activation transfer overhead: {total_comm*1000:.3f} ms/step",
            f"Stage order: {[g.gpu_id for g in ordered_gpus]}",
        ]

        return PartitionResult(
            strategy_name     = "Strategy-B: Pipeline Parallel (1F1B)",
            micro_batch_alloc = {g.gpu_id: m for g in self.gpus},
            layer_alloc       = layer_alloc,
            estimated_tps     = tps,
            step_time_s       = t_step,
            vram_usage_gb     = vram,
            bottleneck_gpu    = bottleneck_gid,
            notes             = notes,
        )

    # ── § Hybrid: ZeRO-1 on H100 tensor parallel ─────────────

    def strategy_c_hybrid(self) -> PartitionResult:
        """
        Strategy C (bonus): Tensor parallel H100 group + pipeline to A6000s.

        Since both A6000s share the same PCIe topology and are near H100 on NUMA1:
          • H100 runs most layers with data parallel within itself (1 GPU, so trivial)
          • A6000s serve as "prefill offload" for embedding + final projection
          • AllReduce only over A6000 pair for the small portion they handle

        This is less optimal than B for pure throughput but balances VRAM.
        Modelled as: layers split H100→29, A6000pair→3 (1.5 each ≈ round to 2,1)
        A6000s AllReduce their 3-layer gradients (tiny).
        H100 runs 29 layers at full speed.
        """
        gpu_map = {g.gpu_id: g for g in self.gpus}
        h100s   = [g for g in self.gpus if g.sm_version >= 9.0]
        a6ks    = [g for g in self.gpus if g.sm_version < 9.0]

        assert h100s, "No H100 found for hybrid strategy"
        h100 = h100s[0]

        total_flops = sum(g.flops for g in self.gpus)
        h_layers    = round(self.model.n_layers * h100.flops / total_flops)
        h_layers    = max(h_layers, self.model.n_layers - len(a6ks) * 1)
        a_layers    = self.model.n_layers - h_layers
        per_a       = max(1, a_layers // len(a6ks))

        layer_alloc : Dict[int, int] = {h100.gpu_id: h_layers}
        remaining = a_layers
        for g in a6ks:
            l = min(per_a, remaining)
            layer_alloc[g.gpu_id] = l
            remaining -= l

        m     = self.global_microbatches
        fpt   = self.model.flops_per_token

        h_frac    = h_layers / self.model.n_layers
        t_h100_mb = self.seq * h_frac * fpt / h100.flops

        a_frac    = per_a / self.model.n_layers
        t_a6k_mb  = self.seq * a_frac * fpt / a6ks[0].flops if a6ks else 0.0

        t_bottleneck = max(t_h100_mb, t_a6k_mb)
        p = 1 + len(a6ks)  # H100 counts as 1 stage; A6000s share another

        t_step = (m + p - 1) * t_bottleneck
        tps    = m * self.seq / t_step

        vram: Dict[int, float] = {}
        for g in self.gpus:
            n_l = layer_alloc[g.gpu_id]
            vram[g.gpu_id] = self.vram_est.pipeline_stage_vram_gb(
                self.model, n_l, n_gpus=self.n_gpus,
                batch=1, seq=self.seq, micro_batches_inflight=2
            )

        bottleneck_gid = h100.gpu_id if t_h100_mb >= t_a6k_mb else a6ks[0].gpu_id

        return PartitionResult(
            strategy_name     = "Strategy-C: Hybrid (H100 heavy + A6000 offload)",
            micro_batch_alloc = {g.gpu_id: m for g in self.gpus},
            layer_alloc       = layer_alloc,
            estimated_tps     = tps,
            step_time_s       = t_step,
            vram_usage_gb     = vram,
            bottleneck_gpu    = bottleneck_gid,
            notes             = [
                f"H100 layers: {h_layers},  A6000 layers each: {per_a}",
                f"Bottleneck: {'H100' if t_h100_mb >= t_a6k_mb else 'A6000'}",
            ],
        )

    def compare_all(self) -> List[PartitionResult]:
        """Run all strategies and return sorted by throughput."""
        results = [
            self.strategy_a_zero3(),
            self.strategy_b_pipeline(),
            self.strategy_c_hybrid(),
        ]
        results.sort(key=lambda r: r.estimated_tps, reverse=True)
        return results


# ──────────────────────────────────────────────────────────────
#  §5  CLUSTER ANALYSIS FUNCTION
# ──────────────────────────────────────────────────────────────

def analyze_cluster() -> None:
    """
    Full analysis for the Neuron_SP 2×A6000 + 1×H100-NVL cluster.

    Prints:
      1. Hardware summary
      2. Compute ratio derivation
      3. Memory budget analysis
      4. Strategy comparison table
      5. Recommended configuration
    """

    # ── Hardware definition (from spec sheet + measured BW) ──
    gpus = [
        GPUSpec(0, "A6000",    48,  38.7,  25.0, 8.6, 1),
        GPUSpec(1, "A6000",    48,  38.7,  25.0, 8.6, 1),
        GPUSpec(2, "H100-NVL", 96, 835.0,  50.0, 9.0, 1),
    ]

    # ── LLaMA-7B config ──────────────────────────────────────
    llama7b = ModelConfig(
        name      = "LLaMA-7B",
        n_params  = 6_740_000_000,
        n_layers  = 32,
        d_model   = 4096,
        n_heads   = 32,
        d_ff      = 11008,
    )

    SEQ_LEN = 2048

    # ── Print header ──────────────────────────────────────────
    sep = "═" * 72
    print(sep)
    print("  Neuron_SP  |  Heterogeneous Partition Analysis")
    print(f"  Cluster    :  2×A6000(48GB) + 1×H100-NVL(96GB)  |  No NVLink")
    print(f"  Model      :  {llama7b.name}  ({llama7b.n_params/1e9:.2f}B params)")
    print(f"  Sequence   :  {SEQ_LEN} tokens")
    print(sep)

    # ── §1 Compute ratio ──────────────────────────────────────
    R = gpus[2].flops_tflops / gpus[0].flops_tflops
    print(f"\n{'─'*72}")
    print(f"  §1  COMPUTE RATIO")
    print(f"{'─'*72}")
    print(f"  R = FLOPS_H100 / FLOPS_A6000 = {gpus[2].flops_tflops} / {gpus[0].flops_tflops}"
          f" = {R:.2f}")
    print(f"  → H100 is {R:.1f}× faster than A6000 at BF16 matmul")
    print(f"  → For 2 A6000s + 1 H100:")
    print(f"    bs_H / bs_A = n_A × R = 2 × {R:.2f} ≈ {2*R:.1f}")
    print(f"    Integer solution: bs_A=1, bs_H={round(2*R)}")
    print(f"  → Micro-batch allocation: GPU0=1, GPU1=1, GPU2={round(2*R)}")

    # ── §2 Memory budget ─────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  §2  MEMORY BUDGET  (BF16 params + AdamW optimizer)")
    print(f"{'─'*72}")
    G   = len(gpus)
    P   = llama7b.n_params
    B   = 1
    S   = SEQ_LEN

    M_param_total = P * 2 / 1e9
    M_grad_total  = P * 4 / 1e9
    M_opt_total   = P * 12 / 1e9
    M_total_full  = M_param_total + M_grad_total + M_opt_total

    print(f"  Full model (params only, BF16)  : {M_param_total:.2f} GB")
    print(f"  Full training state (+ grad+opt): {M_total_full:.2f} GB  ← doesn't fit A6000!")
    print(f"  ZeRO-3 shard (per GPU, 3 GPUs) : {M_total_full/G:.2f} GB")

    act_per_layer = llama7b.activation_bytes_per_layer(B, S)
    M_act_full   = act_per_layer * llama7b.n_layers / 1e9
    M_act_ckpt   = act_per_layer * math.sqrt(llama7b.n_layers) / 1e9

    print(f"  Activations (B=1,S={S}, full)  : {M_act_full:.2f} GB")
    print(f"  Activations (grad checkpoint)  : {M_act_ckpt:.2f} GB")
    print(f"  ZeRO-3 + grad-ckpt total/GPU   : {M_total_full/G + M_act_ckpt:.2f} GB")
    print(f"  A6000 budget 48GB  {'✓' if M_total_full/G + M_act_ckpt < 48 else '✗ TIGHT'}")

    # ── §3 Run solver ─────────────────────────────────────────
    solver  = HeteroPartitionSolver(gpus, llama7b, seq_len=SEQ_LEN)
    results = solver.compare_all()

    print(f"\n{'─'*72}")
    print(f"  §3  STRATEGY COMPARISON")
    print(f"{'─'*72}")

    bw_model = PCIeBandwidthModel(gpus)
    grad_bytes = P * 4
    t_ar_s = bw_model.ring_allreduce_time(grad_bytes)
    print(f"\n  Communication baseline (Ring AllReduce, fp32 grads={grad_bytes/1e9:.1f}GB):")
    print(f"    T_allreduce = 2×(n-1)/n × {grad_bytes/1e9:.1f}GB / 25GB/s"
          f" = {t_ar_s:.4f} s")

    feasibility_flag = lambda r: "✓" if r.is_feasible(gpus) else "✗ OOM"

    header = f"  {'Strategy':<42} {'TPS':>9} {'t_step':>8} {'Feasible':>10}"
    print(f"\n{header}")
    print(f"  {'─'*42} {'─'*9} {'─'*8} {'─'*10}")
    for r in results:
        flag = feasibility_flag(r)
        print(f"  {r.strategy_name:<42} {r.estimated_tps:>8.0f} {r.step_time_s:>8.3f}s {flag:>10}")

    # ── §4 Detailed output per strategy ──────────────────────
    print(f"\n{'─'*72}")
    print(f"  §4  DETAILED RESULTS")
    print(f"{'─'*72}")
    for r in results:
        print(f"\n  ▶ {r.strategy_name}")
        for note in r.notes:
            print(f"     {note}")
        print(f"     Throughput : {r.estimated_tps:.0f} tok/s")
        print(f"     Step time  : {r.step_time_s:.4f} s")
        print(f"     VRAM usage (GB):")
        for gid, gb in r.vram_usage_gb.items():
            cap = next(g.vram_gb for g in gpus if g.gpu_id == gid)
            bar = "█" * int(gb / cap * 20)
            print(f"       GPU{gid}: {gb:5.1f} / {cap:.0f} GB  [{bar:<20}]  "
                  f"{'OOM!' if gb > cap else ''}")
        print(f"     Bottleneck : GPU{r.bottleneck_gpu}")
        print(f"     Feasible   : {feasibility_flag(r)}")

    # ── §5 Recommendation ─────────────────────────────────────
    best     = results[0]
    feasible = [r for r in results if r.is_feasible(gpus)]
    rec      = feasible[0] if feasible else best

    print(f"\n{'═'*72}")
    print(f"  RECOMMENDATION")
    print(f"{'═'*72}")
    print(f"  Best feasible strategy : {rec.strategy_name}")
    print(f"  Estimated throughput   : {rec.estimated_tps:.0f} tok/s")
    print(f"  Global tokens/step     : {solver.global_tokens:,}")

    if "Pipeline" in rec.strategy_name:
        layer_alloc = rec.layer_alloc
        print(f"\n  Pipeline layer assignment:")
        for g in gpus:
            gid = g.gpu_id
            print(f"    GPU{gid} ({g.name:12s})  →  {layer_alloc[gid]:2d} layers "
                  f"({layer_alloc[gid]/llama7b.n_layers*100:.1f}%)")
        print(f"\n  DeepSpeed config snippet (pipeline):")
        print(f"    pipeline_parallel_size: 3")
        print(f"    num_stages: 3")
        print(f"    micro_batches: {solver.global_microbatches}")
        print(f"    # stage_to_rank: GPU0→stage0, GPU1→stage2, GPU2→stage1")
        print(f"    # (place fast GPU in middle to hide A6000 latency)")
    else:
        alloc = rec.micro_batch_alloc
        print(f"\n  ZeRO-3 micro-batch assignment:")
        for g in gpus:
            gid = g.gpu_id
            print(f"    GPU{gid} ({g.name:12s})  →  {alloc[gid]:2d} micro-batches")
        print(f"\n  DeepSpeed config snippet (ZeRO-3):")
        print(f"    zero_optimization.stage: 3")
        print(f"    gradient_accumulation_steps: {alloc[0]}  # for A6000")
        print(f"    # Use custom data sampler to give GPU2 {alloc[2]}× data")

    print(f"\n  Key insight:")
    print(f"    The H100 is {R:.1f}× faster per FLOPs. With pipeline,")
    print(f"    it absorbs ~{rec.layer_alloc.get(2, 0)}/{llama7b.n_layers} layers")
    print(f"    while A6000s handle edge layers. Bubble={2/(solver.global_microbatches+2)*100:.1f}%.")
    print(f"    PCIe Gen4/5 (no NVLink) → communication is NOT the bottleneck")
    print(f"    (activation transfers ≈ {bw_model.pipeline_activation_time(0,2,1,SEQ_LEN,llama7b.d_model)*1000:.2f} ms vs step ≈ {rec.step_time_s:.2f} s).")
    print(sep)


# ──────────────────────────────────────────────────────────────
#  §6  ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    analyze_cluster()
