# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
scaling_autoconfig.py — Auto-configuration for Chinchilla Scaling Law experiments
==================================================================================
Given a target model parameter count, automatically selects:
  - Pipeline stages count
  - ZeRO stage
  - Micro-batch size / gradient accumulation steps

Then prints diagnostic output:
  - Theoretical tokens_per_second per device and cluster-wide
  - MFU (Model FLOP Utilization) per GPU tier
  - Pipeline bubble ratio (if pipeline > 1 stage)
  - Cross-NUMA allreduce latency estimate

Designed for the 2×A6000 (48GB) + 1×H100-NVL (96GB) heterogeneous cluster.

Usage (standalone):
    python deepspeed/scripts/scaling_autoconfig.py --params 70e6
    python deepspeed/scripts/scaling_autoconfig.py --params 410e6 --seq_len 2048

Usage (import):
    from deepspeed.scripts.scaling_autoconfig import HeteroAutoConfig
    cfg = HeteroAutoConfig(params=410e6)
    plan = cfg.solve()
    print(plan)
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Cluster hardware constants — 2×A6000 + 1×H100
# ---------------------------------------------------------------------------
_A6000_MEM_GB = 48.0
_H100_MEM_GB = 96.0
_A6000_BF16_TFLOPS = 38.7
_H100_BF16_TFLOPS = 835.0
_PCIE4_BW_GBS = 25.0  # A6000 PCIe-4
_PCIE5_BW_GBS = 50.0  # H100 PCIe-5
_MFU_ESTIMATE = 0.35  # Typical achieved MFU on PCIe-only heterogeneous cluster

# Memory overhead per parameter in BF16 with optimizer state (AdamW ZeRO-3 sharded)
_BYTES_PER_PARAM_ZERO3 = 2.0  # BF16 weights only on device (optimizer state sharded)
_BYTES_PER_PARAM_ZERO2 = 16.0  # BF16 weights + fp32 master + Adam m/v
_ACTIVATION_MEM_FACTOR = 12.0  # bytes per token per layer (approx, with recompute=False)

# Cross-NUMA allreduce baseline latency (measured on this cluster, μs per MB)
_CROSS_NUMA_LATENCY_US_PER_MB = 28.0
# PCIe P2P activation transfer latency (μs per MB) — pipeline stage boundary
_P2P_LATENCY_US_PER_MB = 18.0


@dataclass
class GpuSpec:
    index: int
    name: str
    mem_gb: float
    bf16_tflops: float
    pcie_bw_gbs: float


@dataclass
class ScalingAutoConfigResult:
    # Model
    params: int
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    seq_len: int

    # Chinchilla budget
    chinchilla_optimal_tokens: int
    chinchilla_total_steps: int

    # Partition
    zero_stage: int
    pipeline_stages: int
    placement_strategy: str  # "data_parallel" or "pipeline_hetero"
    layer_assignment: Dict[int, List[int]]  # gpu_index -> layer list

    # Batch sizing
    micro_batch_size: int
    gradient_accumulation_steps: int
    global_batch_tokens: int

    # Per-GPU throughput diagnostics
    tokens_per_second_per_gpu: Dict[int, float]  # gpu_index -> tok/s
    mfu_per_gpu: Dict[int, float]  # gpu_index -> MFU fraction
    cluster_tokens_per_second: float
    cluster_mfu: float

    # Pipeline overhead (None if pipeline_stages == 1)
    pipeline_bubble_ratio: Optional[float]
    # Cross-NUMA allreduce latency per step (ms)
    cross_numa_allreduce_ms: float
    # P2P activation transfer per step (ms); None if no pipeline
    p2p_activation_transfer_ms: Optional[float]

    # Memory estimates
    peak_mem_gb_per_gpu: Dict[int, float]  # gpu_index -> GB

    def __str__(self) -> str:
        lines = [
            "=" * 72,
            f"  HeteroAutoConfig — {self.params / 1e6:.0f}M params",
            "=" * 72,
            f"  Architecture:  hidden={self.hidden_size}  layers={self.num_layers}"
            f"  heads={self.num_heads}  ffn={self.intermediate_size}",
            f"  Seq len:       {self.seq_len}",
            f"  Chinchilla:    {self.chinchilla_optimal_tokens / 1e9:.1f}B tokens"
            f"  ({self.chinchilla_total_steps:,} steps)",
            "",
            f"  ZeRO stage:    {self.zero_stage}",
            f"  Pipeline:      {self.pipeline_stages} stage(s)  [{self.placement_strategy}]",
            f"  Micro-batch:   {self.micro_batch_size}",
            f"  Grad accum:    {self.gradient_accumulation_steps}",
            f"  Global tokens: {self.global_batch_tokens:,} / step",
            "",
            "  --- Throughput diagnostics ---",
        ]
        gpu_names = {0: "A6000#0", 1: "A6000#1", 2: "H100"}
        for idx in sorted(self.tokens_per_second_per_gpu):
            name = gpu_names.get(idx, f"GPU{idx}")
            tok_s = self.tokens_per_second_per_gpu[idx]
            mfu = self.mfu_per_gpu[idx]
            mem = self.peak_mem_gb_per_gpu.get(idx, 0.0)
            lines.append(f"  {name:<10} actual={tok_s:>8,.0f} tok/s  MFU={mfu:.3f}"
                         f"  peak_mem={mem:.1f}GB")
        lines += [
            f"  CLUSTER        actual={self.cluster_tokens_per_second:>8,.0f} tok/s"
            f"  MFU={self.cluster_mfu:.3f}",
            "",
            "  --- Pipeline / communication overhead ---",
        ]
        if self.pipeline_bubble_ratio is not None:
            lines.append(f"  Pipeline bubble ratio:      {self.pipeline_bubble_ratio:.4f}"
                         f"  ({self.pipeline_bubble_ratio * 100:.1f}% of step wasted)")
        else:
            lines.append("  Pipeline bubble ratio:      N/A (data-parallel)")
        lines.append(f"  Cross-NUMA allreduce:       {self.cross_numa_allreduce_ms:.2f} ms/step")
        if self.p2p_activation_transfer_ms is not None:
            lines.append(f"  P2P activation transfer:    {self.p2p_activation_transfer_ms:.2f} ms/step")
        lines.append("=" * 72)
        return "\n".join(lines)


class HeteroAutoConfig:
    """
    Automatically derives optimal training config for a given parameter count
    on the 2×A6000 + 1×H100 heterogeneous cluster.

    Decision logic:
      - params <= 200M:  ZeRO-2, 1 pipeline stage, data parallel
      - 200M < params <= 600M:  ZeRO-3, 2 pipeline stages (A6000 pair / H100)
      - params > 600M:  ZeRO-3, 3 pipeline stages (one per GPU)

    Batch size is chosen to fill ~128K tokens per global step (matches existing
    pretrain_7b.py defaults) while respecting per-GPU memory limits.
    """

    # Standard cluster GPUs (indices match CUDA_VISIBLE_DEVICES)
    GPUS: List[GpuSpec] = [
        GpuSpec(0, "A6000#0", _A6000_MEM_GB, _A6000_BF16_TFLOPS, _PCIE4_BW_GBS),
        GpuSpec(1, "A6000#1", _A6000_MEM_GB, _A6000_BF16_TFLOPS, _PCIE4_BW_GBS),
        GpuSpec(2, "H100", _H100_MEM_GB, _H100_BF16_TFLOPS, _PCIE5_BW_GBS),
    ]

    TARGET_GLOBAL_TOKENS = 131072  # ~128K tokens/step

    def __init__(
        self,
        params: float,
        seq_len: int = 2048,
        chinchilla_multiplier: float = 20.0,
        mfu_override: Optional[float] = None,
    ) -> None:
        self.params = int(params)
        self.seq_len = seq_len
        self.chinchilla_multiplier = chinchilla_multiplier
        self.mfu = mfu_override if mfu_override is not None else _MFU_ESTIMATE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def solve(self) -> ScalingAutoConfigResult:
        """Solve for the full config and return diagnostic result."""
        arch = self._derive_arch()
        partition = self._derive_partition(arch)
        batch = self._derive_batch(arch, partition)
        throughput = self._estimate_throughput(arch, partition, batch)
        overhead = self._estimate_overhead(arch, partition, batch)
        memory = self._estimate_memory(arch, partition, batch)

        chinchilla_tokens = int(self.params * self.chinchilla_multiplier)
        total_steps = math.ceil(chinchilla_tokens / batch["global_batch_tokens"])

        return ScalingAutoConfigResult(
            params=self.params,
            hidden_size=arch["hidden_size"],
            num_layers=arch["num_layers"],
            num_heads=arch["num_heads"],
            intermediate_size=arch["intermediate_size"],
            seq_len=self.seq_len,
            chinchilla_optimal_tokens=chinchilla_tokens,
            chinchilla_total_steps=total_steps,
            zero_stage=partition["zero_stage"],
            pipeline_stages=partition["pipeline_stages"],
            placement_strategy=partition["placement_strategy"],
            layer_assignment=partition["layer_assignment"],
            micro_batch_size=batch["micro_batch_size"],
            gradient_accumulation_steps=batch["gradient_accumulation_steps"],
            global_batch_tokens=batch["global_batch_tokens"],
            tokens_per_second_per_gpu=throughput["per_gpu"],
            mfu_per_gpu=throughput["mfu_per_gpu"],
            cluster_tokens_per_second=throughput["cluster"],
            cluster_mfu=throughput["cluster_mfu"],
            pipeline_bubble_ratio=overhead["pipeline_bubble_ratio"],
            cross_numa_allreduce_ms=overhead["cross_numa_allreduce_ms"],
            p2p_activation_transfer_ms=overhead["p2p_activation_transfer_ms"],
            peak_mem_gb_per_gpu=memory,
        )

    # ------------------------------------------------------------------
    # Architecture derivation
    # ------------------------------------------------------------------
    def _derive_arch(self) -> dict:
        """
        Map a parameter count to a concrete architecture.
        Uses the standard GPT-style hidden_size/num_layers scaling.
        """
        # Reference points: (params, hidden, layers, heads, ffn_mult)
        _ARCH_TABLE = [
            (70e6, 512, 8, 8, 2.688),  # ~70M
            (160e6, 768, 12, 12, 2.667),  # ~160M (GPT-2 medium)
            (410e6, 1024, 24, 16, 2.667),  # ~410M
            (1e9, 2048, 22, 16, 2.688),  # ~1B (Pythia-1B style)
        ]
        # Find closest by log-scale
        best = min(
            _ARCH_TABLE,
            key=lambda row: abs(math.log(row[0]) - math.log(self.params)),
        )
        _, hidden, layers, heads, ffn_mult = best
        intermediate = int(hidden * ffn_mult)
        return {
            "hidden_size": hidden,
            "num_layers": layers,
            "num_heads": heads,
            "intermediate_size": intermediate,
        }

    # ------------------------------------------------------------------
    # Partition strategy selection
    # ------------------------------------------------------------------
    def _derive_partition(self, arch: dict) -> dict:
        """
        Select ZeRO stage and pipeline stages based on model size and memory budget.
        """
        num_layers = arch["num_layers"]

        if self.params <= 200e6:
            # Small model: no pipeline needed, ZeRO-2 (replicate weights, shard optimizer)
            zero_stage = 2
            pipeline_stages = 1
            placement = "data_parallel"
            layer_assignment = {
                0: list(range(num_layers)),
                1: list(range(num_layers)),
                2: list(range(num_layers)),
            }
        elif self.params <= 600e6:
            # Medium: 2-stage pipeline — A6000 pair handles first half, H100 second half
            zero_stage = 3
            pipeline_stages = 2
            placement = "pipeline_hetero"
            split = num_layers // 2
            # Both A6000s share stage 0 (micro-batch alternation handled by ZeRO-3)
            layer_assignment = {
                0: list(range(split)),
                1: list(range(split)),
                2: list(range(split, num_layers)),
            }
        else:
            # Large: 3-stage pipeline — one stage per physical GPU
            zero_stage = 3
            pipeline_stages = 3
            placement = "pipeline_hetero"
            # Distribute more layers to H100 (it's ~21× faster in BF16)
            # A6000 compute ratio: 38.7 / (38.7 + 38.7 + 835) ≈ 0.043 each
            # Allocate layers proportional to compute: A6000#0 ≈ A6000#1 ≈ 0.043 × L
            # H100 ≈ 0.913 × L  →  clamp to at least 1 layer per A6000
            a_layers = max(1, int(num_layers * _A6000_BF16_TFLOPS / (2 * _A6000_BF16_TFLOPS + _H100_BF16_TFLOPS)))
            h_layers = num_layers - 2 * a_layers
            layer_assignment = {
                0: list(range(a_layers)),
                1: list(range(a_layers, 2 * a_layers)),
                2: list(range(2 * a_layers, num_layers)),
            }
            print(f"[AutoConfig] Layer split: A6000#0={a_layers}, A6000#1={a_layers},"
                  f" H100={h_layers}")

        print(f"[AutoConfig] ZeRO-{zero_stage}, pipeline_stages={pipeline_stages},"
              f" strategy={placement}")
        return {
            "zero_stage": zero_stage,
            "pipeline_stages": pipeline_stages,
            "placement_strategy": placement,
            "layer_assignment": layer_assignment,
        }

    # ------------------------------------------------------------------
    # Batch sizing
    # ------------------------------------------------------------------
    def _derive_batch(self, arch: dict, partition: dict) -> dict:
        """
        Choose micro-batch size to stay within memory limits and hit ~128K tokens/step.
        """
        target = self.TARGET_GLOBAL_TOKENS
        seq = self.seq_len
        n_data_parallel = 3 if partition["pipeline_stages"] == 1 else 1

        # Try micro-batch sizes from large to small; pick largest that fits
        # Bottleneck device: A6000 with 48GB
        for mbs in [8, 4, 2, 1]:
            # Rough activation memory check (ignores gradient checkpointing)
            act_mem = (mbs * seq * arch["num_layers"] * _ACTIVATION_MEM_FACTOR / 1e9)
            if act_mem < _A6000_MEM_GB * 0.6:  # leave 40% headroom for weights
                break

        # Gradient accumulation to reach target token count
        grad_accum = max(
            1,
            round(target / (mbs * seq * n_data_parallel)),
        )
        # Round to next power of 2 for cleaner step counts
        grad_accum = 2**round(math.log2(max(grad_accum, 1)))
        global_batch_tokens = mbs * seq * grad_accum * n_data_parallel

        print(f"[AutoConfig] micro_batch={mbs}, grad_accum={grad_accum},"
              f" global_tokens={global_batch_tokens:,}")
        return {
            "micro_batch_size": mbs,
            "gradient_accumulation_steps": grad_accum,
            "global_batch_tokens": global_batch_tokens,
        }

    # ------------------------------------------------------------------
    # Throughput estimation
    # ------------------------------------------------------------------
    def _flops_per_token(self) -> float:
        """Standard 6N FLOPs approximation (forward + backward = 6 × params)."""
        return 6.0 * self.params

    def _estimate_throughput(
        self,
        arch: dict,
        partition: dict,
        batch: dict,
    ) -> dict:
        """
        Estimate tokens/s per GPU and cluster-wide.

        For pipeline: throughput is limited by slowest stage (theoretical max).
        For data-parallel: throughput sums across GPUs.
        Print actual vs. theoretical throughput for each GPU.
        """
        fpt = self._flops_per_token()
        mbs = batch["micro_batch_size"]
        n_stages = partition["pipeline_stages"]
        seq = self.seq_len

        per_gpu: Dict[int, float] = {}
        mfu_per_gpu: Dict[int, float] = {}

        for gpu in self.GPUS:
            n_layers_on_gpu = len(partition["layer_assignment"].get(gpu.index, []))
            if n_layers_on_gpu == 0:
                per_gpu[gpu.index] = 0.0
                mfu_per_gpu[gpu.index] = 0.0
                continue

            # Layer fraction this GPU handles
            layer_frac = n_layers_on_gpu / arch["num_layers"]
            flops_per_token_this_gpu = fpt * layer_frac

            # Theoretical token rate for this GPU
            peak_flops_per_s = gpu.bf16_tflops * 1e12
            # tokens_per_sec = achieved_flops / flops_per_token
            tokens_per_sec_theory = peak_flops_per_s * self.mfu / flops_per_token_this_gpu

            per_gpu[gpu.index] = tokens_per_sec_theory
            # MFU = actual_flops / peak_flops = (tokens_s × fpt_this_gpu) / peak_flops
            mfu_val = (tokens_per_sec_theory * flops_per_token_this_gpu) / peak_flops_per_s
            mfu_per_gpu[gpu.index] = mfu_val

            print(f"[AutoConfig][{gpu.name:<8}] "
                  f"theoretical={tokens_per_sec_theory:>9,.0f} tok/s  "
                  f"actual≈{tokens_per_sec_theory:>9,.0f} tok/s  "
                  f"MFU={mfu_val:.4f}  "
                  f"(layers {n_layers_on_gpu}/{arch['num_layers']})")

        # Cluster-level throughput
        if n_stages == 1:
            # Data parallel: sum contributions
            cluster_tps = sum(per_gpu.values())
        else:
            # Pipeline: bottlenecked by slowest stage
            nonzero = [v for v in per_gpu.values() if v > 0]
            bottleneck = min(nonzero) if nonzero else 0.0
            # Apply pipeline bubble penalty
            bubble = (n_stages - 1) / n_stages
            cluster_tps = bottleneck * n_stages * (1 - bubble)

        total_peak = sum(g.bf16_tflops * 1e12 for g in self.GPUS)
        cluster_mfu = (cluster_tps * fpt) / total_peak if total_peak > 0 else 0.0

        print(f"[AutoConfig][CLUSTER  ] "
              f"cluster={cluster_tps:>9,.0f} tok/s  cluster_MFU={cluster_mfu:.4f}")

        return {
            "per_gpu": per_gpu,
            "mfu_per_gpu": mfu_per_gpu,
            "cluster": cluster_tps,
            "cluster_mfu": cluster_mfu,
        }

    # ------------------------------------------------------------------
    # Overhead estimation
    # ------------------------------------------------------------------
    def _estimate_overhead(
        self,
        arch: dict,
        partition: dict,
        batch: dict,
    ) -> dict:
        """
        Estimate pipeline bubble ratio and communication overhead.
        """
        n_stages = partition["pipeline_stages"]
        n_micro = batch["gradient_accumulation_steps"]

        # Pipeline bubble ratio: fraction of compute time wasted in bubble
        # 1F1B formula: bubble = (n_stages - 1) / (n_stages + n_microbatches - 1)
        if n_stages > 1:
            bubble = (n_stages - 1) / (n_stages + n_micro - 1)
        else:
            bubble = None

        # Cross-NUMA allreduce: gradient size = params × 2 bytes (BF16)
        # All-reduce across 3 GPUs on PCIe ≈ 2× ring passes
        grad_size_mb = (self.params * 2) / 1e6  # MB
        cross_numa_ms = (grad_size_mb * _CROSS_NUMA_LATENCY_US_PER_MB * 2 / 1000.0)

        # P2P activation transfer between pipeline stages
        if n_stages > 1:
            # Activation tensor at stage boundary: mbs × seq × hidden × 2 bytes (BF16)
            act_mb = (batch["micro_batch_size"] * self.seq_len * arch["hidden_size"] * 2) / 1e6
            # (n_stages - 1) boundaries, using slower of the two connected GPUs
            p2p_ms = act_mb * _P2P_LATENCY_US_PER_MB * (n_stages - 1) / 1000.0
        else:
            p2p_ms = None

        if bubble is not None:
            print(f"[AutoConfig] pipeline_bubble_ratio={bubble:.4f}"
                  f"  cross_numa_allreduce={cross_numa_ms:.2f}ms"
                  f"  p2p_transfer={p2p_ms:.2f}ms")
        else:
            print(f"[AutoConfig] no pipeline  cross_numa_allreduce={cross_numa_ms:.2f}ms")

        return {
            "pipeline_bubble_ratio": bubble,
            "cross_numa_allreduce_ms": cross_numa_ms,
            "p2p_activation_transfer_ms": p2p_ms,
        }

    # ------------------------------------------------------------------
    # Memory estimation
    # ------------------------------------------------------------------
    def _estimate_memory(self, arch: dict, partition: dict, batch: dict) -> Dict[int, float]:
        """
        Estimate peak GPU memory (GB) per device.
        Includes: model weights (BF16) + optimizer state (sharded) + activations.
        """
        zero_stage = partition["zero_stage"]
        mbs = batch["micro_batch_size"]
        num_layers = arch["num_layers"]
        hidden = arch["hidden_size"]
        seq = self.seq_len

        if zero_stage == 3:
            bytes_per_param = _BYTES_PER_PARAM_ZERO3
        else:
            bytes_per_param = _BYTES_PER_PARAM_ZERO2

        mem_per_gpu: Dict[int, float] = {}
        for gpu in self.GPUS:
            n_layers_on_gpu = len(partition["layer_assignment"].get(gpu.index, []))
            # Weight memory: fraction of total params (approximate)
            layer_frac = n_layers_on_gpu / num_layers
            weight_gb = (self.params * bytes_per_param * layer_frac) / 1e9
            # Optimizer state sharded across 3 GPUs in ZeRO-3
            opt_gb = (self.params * 12 / 3) / 1e9 if zero_stage == 3 else 0.0
            # Activation memory (no gradient checkpointing assumed)
            act_gb = (mbs * seq * n_layers_on_gpu * _ACTIVATION_MEM_FACTOR) / 1e9
            total_gb = weight_gb + opt_gb + act_gb
            mem_per_gpu[gpu.index] = total_gb

        return mem_per_gpu


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DES-LOC Heterogeneous Auto-Config")
    p.add_argument(
        "--params",
        type=float,
        required=True,
        help="Model parameter count, e.g. 70e6, 160e6, 410e6, 1e9",
    )
    p.add_argument(
        "--seq_len",
        type=int,
        default=2048,
        help="Sequence length (default: 2048)",
    )
    p.add_argument(
        "--chinchilla_multiplier",
        type=float,
        default=20.0,
        help="Chinchilla token multiplier C_opt = multiplier × N (default: 20)",
    )
    p.add_argument(
        "--mfu",
        type=float,
        default=None,
        help="Override MFU estimate (default: 0.35)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = HeteroAutoConfig(
        params=args.params,
        seq_len=args.seq_len,
        chinchilla_multiplier=args.chinchilla_multiplier,
        mfu_override=args.mfu,
    )
    result = cfg.solve()
    print(result)


if __name__ == "__main__":
    main()
