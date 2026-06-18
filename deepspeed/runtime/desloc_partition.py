"""
DES-LOC Heterogeneous Partition Solver
=======================================

Mathematical framework for optimal model partitioning across heterogeneous GPUs.

Hardware target:
    GPU0: A6000 48GB, SM8.6, PCIe Gen4 x16, BF16 38.7 TFLOPS
    GPU1: A6000 48GB, SM8.6, PCIe Gen4 x16, BF16 38.7 TFLOPS
    GPU2: H100 NVL 96GB, SM9.0, PCIe Gen5 x16, BF16 835 TFLOPS

Key insight: The FLOPS ratio R = 835/38.7 ≈ 21.6x makes naive data-parallel
impossible (A6000 becomes a 21x bottleneck). Instead we use:

    Strategy: ZeRO-3 + asymmetric gradient accumulation
    - All 3 GPUs hold the full model (ZeRO-3 shards optimizer state)
    - H100 processes more micro-batches per gradient accumulation step
    - A6000s process fewer micro-batches but contribute to optimizer state

Mathematical derivation:
    Let B_i = micro-batches processed by GPU i per global step
    Total tokens per step: T = sum(B_i) * seq_len
    Time per step: max_i(B_i * t_fwd_i + B_i * t_bwd_i + t_allreduce)

    For load balance: B_i * t_compute_i = constant for all i
    → B_i ∝ FLOPS_i
    → B_h100 / B_a6000 = 835 / 38.7 ≈ 21.6

    With seq_len=2048, micro_batch=1:
      t_fwd_a6000 ≈ 6*N*S / FLOPS = 6*7e9*2048 / 38.7e12 ≈ 2.22s
      t_fwd_h100  ≈ 6*7e9*2048 / 835e12 ≈ 0.103s

    If A6000 does 1 micro-batch, H100 should do ~22 micro-batches
    → Total effective batch = (1 + 1 + 22) * 2048 = 49,152 tokens

    Allreduce cost (ring, 3 GPUs):
      grad_size = 7e9 * 2 bytes = 14 GB
      ZeRO-3 reduces this to ~14/3 ≈ 4.7 GB per allreduce
      PCIe Gen4: 2*(3-1)/3 * 4.7e9 / 25e9 ≈ 0.25s
      → Communication is ~11% of A6000 compute time (acceptable)

    Estimated throughput:
      Tokens/step = 49,152
      Time/step ≈ 2.22s (bottleneck: A6000 doing 1 forward+backward)
      → ~22,100 tokens/second
      → MFU = 22100 * 6 * 7e9 / (835+38.7+38.7)*1e12 ≈ 10.2%

    This MFU is low due to the massive FLOPS imbalance. To improve:
    - Increase A6000 micro-batches (requires fitting in 48GB)
    - Use CPU offload to free A6000 VRAM for larger batches
    - Pipeline parallelism (H100 handles more layers)
"""

import math
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class GPUSpec:
    """Specification of one GPU."""
    index: int
    name: str
    vram_gb: float
    bf16_tflops: float
    pcie_bw_gbps: float


@dataclass
class PartitionResult:
    """Output of the partition solver."""
    strategy: str
    micro_batches_per_gpu: Dict[int, int]
    layers_per_gpu: Dict[int, List[int]]
    effective_batch_tokens: int
    estimated_step_time_s: float
    estimated_tokens_per_sec: float
    estimated_mfu: float
    allreduce_time_s: float
    compute_time_s: float
    communication_fraction: float
    analysis: str


class HeteroPartitionSolver:
    """
    Solve the heterogeneous partition problem for a given model and GPU set.

    Explores two strategies and picks the better one:
    1. ZeRO-3 + Asymmetric Grad Accum (all GPUs hold full model)
    2. Pipeline Parallel (layers split by FLOPS ratio)
    """

    def __init__(self, gpus: List[GPUSpec], model_params: float = 7e9,
                 num_layers: int = 32, seq_len: int = 2048):
        self.gpus = gpus
        self.N = model_params
        self.L = num_layers
        self.S = seq_len
        self.n_gpu = len(gpus)

    def solve(self) -> PartitionResult:
        """Run both strategies and return the better one."""
        r1 = self._solve_zero3_asymmetric()
        r2 = self._solve_pipeline()

        if r1.estimated_tokens_per_sec >= r2.estimated_tokens_per_sec:
            logger.info("Selected strategy: %s (%.0f tok/s > %.0f tok/s)",
                        r1.strategy, r1.estimated_tokens_per_sec,
                        r2.estimated_tokens_per_sec)
            return r1
        else:
            logger.info("Selected strategy: %s (%.0f tok/s > %.0f tok/s)",
                        r2.strategy, r2.estimated_tokens_per_sec,
                        r1.estimated_tokens_per_sec)
            return r2

    def _compute_time(self, gpu: GPUSpec, n_micro: int) -> float:
        """Time for n_micro forward+backward passes on one GPU (seconds)."""
        flops_per_token = 6 * self.N  # 2N fwd + 4N bwd
        total_flops = flops_per_token * self.S * n_micro
        return total_flops / (gpu.bf16_tflops * 1e12)

    def _allreduce_time(self, grad_bytes: float) -> float:
        """Ring allreduce time across all GPUs."""
        min_bw = min(g.pcie_bw_gbps for g in self.gpus) * 1e9
        return 2 * (self.n_gpu - 1) / self.n_gpu * grad_bytes / min_bw

    def _solve_zero3_asymmetric(self) -> PartitionResult:
        """
        Strategy 1: All GPUs hold the full model (ZeRO-3 shards optimizer state).
        Each GPU processes micro-batches proportional to its FLOPS.
        """
        total_flops = sum(g.bf16_tflops for g in self.gpus)
        min_flops = min(g.bf16_tflops for g in self.gpus)

        # Micro-batches proportional to FLOPS
        micro = {}
        for g in self.gpus:
            mb = max(1, round(g.bf16_tflops / min_flops))
            # Check VRAM constraint
            param_bytes = self.N * 2  # BF16
            optimizer_bytes = self.N * 12 / self.n_gpu  # ZeRO-3 sharded
            activation_bytes = mb * self.S * 4096 * 2 * self.L * 2  # rough estimate
            total_mem = (param_bytes + optimizer_bytes + activation_bytes) / 1e9
            while total_mem > g.vram_gb * 0.85 and mb > 1:
                mb -= 1
                activation_bytes = mb * self.S * 4096 * 2 * self.L * 2
                total_mem = (param_bytes + optimizer_bytes + activation_bytes) / 1e9
            micro[g.index] = mb

        # Time analysis
        compute_times = {g.index: self._compute_time(g, micro[g.index]) for g in self.gpus}
        bottleneck_time = max(compute_times.values())

        grad_bytes = self.N * 2 / self.n_gpu  # ZeRO-3 reduce-scatter
        ar_time = self._allreduce_time(grad_bytes)

        step_time = bottleneck_time + ar_time
        total_tokens = sum(micro.values()) * self.S
        tps = total_tokens / step_time

        theoretical_peak_flops = total_flops * 1e12
        actual_flops = 6 * self.N * total_tokens / step_time
        mfu = actual_flops / theoretical_peak_flops

        layers = {g.index: list(range(self.L)) for g in self.gpus}

        analysis = (
            f"ZeRO-3 Asymmetric Gradient Accumulation:\n"
            f"  Micro-batches: {micro}\n"
            f"  Per-GPU compute: {', '.join(f'GPU{k}={v:.3f}s' for k,v in compute_times.items())}\n"
            f"  Bottleneck: {bottleneck_time:.3f}s | Allreduce: {ar_time:.3f}s\n"
            f"  Tokens/step: {total_tokens} | Step time: {step_time:.3f}s\n"
            f"  Throughput: {tps:.0f} tok/s | MFU: {mfu*100:.1f}%\n"
            f"  Comm fraction: {ar_time/step_time*100:.1f}%"
        )

        return PartitionResult(
            strategy="zero3_asymmetric",
            micro_batches_per_gpu=micro,
            layers_per_gpu=layers,
            effective_batch_tokens=total_tokens,
            estimated_step_time_s=step_time,
            estimated_tokens_per_sec=tps,
            estimated_mfu=mfu,
            allreduce_time_s=ar_time,
            compute_time_s=bottleneck_time,
            communication_fraction=ar_time / step_time,
            analysis=analysis,
        )

    def _solve_pipeline(self) -> PartitionResult:
        """
        Strategy 2: Pipeline parallelism — split layers by FLOPS ratio.
        """
        total_flops = sum(g.bf16_tflops for g in self.gpus)

        # Assign layers proportional to FLOPS
        layers_per_gpu = {}
        remaining = self.L
        sorted_gpus = sorted(self.gpus, key=lambda g: g.bf16_tflops, reverse=True)

        for i, g in enumerate(sorted_gpus):
            if i == len(sorted_gpus) - 1:
                n_layers = remaining
            else:
                n_layers = max(1, round(self.L * g.bf16_tflops / total_flops))
                n_layers = min(n_layers, remaining - (len(sorted_gpus) - i - 1))
            layers_per_gpu[g.index] = n_layers
            remaining -= n_layers

        # Assign actual layer indices
        layer_assignments = {}
        offset = 0
        for g in sorted_gpus:
            n = layers_per_gpu[g.index]
            layer_assignments[g.index] = list(range(offset, offset + n))
            offset += n

        # Time analysis
        n_microbatch = 4  # pipeline microbatches
        flops_per_layer = 6 * (self.N / self.L) * self.S
        compute_times = {}
        for g in self.gpus:
            n = layers_per_gpu[g.index]
            t = n * flops_per_layer / (g.bf16_tflops * 1e12)
            compute_times[g.index] = t

        bottleneck = max(compute_times.values())
        # Pipeline bubble: (p-1)/m fraction
        p = self.n_gpu
        bubble = (p - 1) / n_microbatch
        effective_time = bottleneck * n_microbatch * (1 + bubble)

        # Inter-stage communication
        activation_bytes = self.S * 4096 * 2  # one activation tensor
        comm_time = activation_bytes / (min(g.pcie_bw_gbps for g in self.gpus) * 1e9) * (p - 1)

        step_time = effective_time + comm_time
        total_tokens = n_microbatch * self.S
        tps = total_tokens / step_time
        actual_flops = 6 * self.N * total_tokens / step_time
        mfu = actual_flops / (total_flops * 1e12)

        analysis = (
            f"Pipeline Parallel (by FLOPS ratio):\n"
            f"  Layers/GPU: {layers_per_gpu}\n"
            f"  Per-stage compute: {', '.join(f'GPU{k}={v:.4f}s' for k,v in compute_times.items())}\n"
            f"  Microbatches: {n_microbatch} | Bubble: {bubble*100:.0f}%\n"
            f"  Step time: {step_time:.3f}s | Comm: {comm_time:.4f}s\n"
            f"  Throughput: {tps:.0f} tok/s | MFU: {mfu*100:.1f}%"
        )

        return PartitionResult(
            strategy="pipeline_parallel",
            micro_batches_per_gpu={g.index: n_microbatch for g in self.gpus},
            layers_per_gpu=layer_assignments,
            effective_batch_tokens=total_tokens,
            estimated_step_time_s=step_time,
            estimated_tokens_per_sec=tps,
            estimated_mfu=mfu,
            allreduce_time_s=comm_time,
            compute_time_s=bottleneck,
            communication_fraction=comm_time / step_time,
            analysis=analysis,
        )


def analyze_cluster():
    """Analyze the actual 2×A6000 + 1×H100 cluster."""
    gpus = [
        GPUSpec(0, "A6000", 48.0, 38.7, 25.0),
        GPUSpec(1, "A6000", 48.0, 38.7, 25.0),
        GPUSpec(2, "H100 NVL", 96.0, 835.0, 50.0),
    ]
    solver = HeteroPartitionSolver(gpus)
    result = solver.solve()
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    result = analyze_cluster()
    print("\n" + "=" * 60)
    print("DES-LOC PARTITION ANALYSIS")
    print("=" * 60)
    print(result.analysis)
    print(f"\nRecommended strategy: {result.strategy}")
    print(f"Expected throughput: {result.estimated_tokens_per_sec:.0f} tokens/sec")
    print(f"Expected MFU: {result.estimated_mfu*100:.1f}%")
