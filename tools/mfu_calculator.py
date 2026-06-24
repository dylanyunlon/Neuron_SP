# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
mfu_calculator.py — Model FLOP Utilization calculator for heterogeneous GPU clusters.

Reads a DES-LOC experiment config YAML to discover the cluster GPU layout, then
computes per-device and aggregate MFU given an observed training throughput
(tokens/sec).

FLOPs per token (forward + backward) follow the standard convention:
    flops_per_token = 6 * N
where N is the number of model parameters and the factor 6 accounts for
multiply-add (×2) and the backward-pass cost (×3 of forward).

Per-GPU achieved FLOPS are derived by distributing the total throughput across
devices proportionally to their theoretical peak, which matches the DES-LOC
heterogeneous workload balancing strategy.

Usage:
    python tools/mfu_calculator.py --config experiments/configs/7b_ags1_desloc.yaml --tokens-per-sec 1200
    python tools/mfu_calculator.py --config experiments/configs/7b_ags1_desloc.yaml --tokens-per-sec 1200 --json
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import List

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required.  pip install pyyaml", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GpuInfo:
    """Single GPU entry parsed from the cluster config."""
    index: int
    name: str
    mem_gb: float
    bf16_tflops: float
    pcie_gen: int = 4
    pcie_bw_gbs: float = 25.0
    sm_version: str = ""
    numa_node: int = 0


@dataclass
class GpuMfuResult:
    """MFU breakdown for one GPU."""
    index: int
    name: str
    peak_tflops: float
    achieved_tflops: float
    mfu: float  # 0.0–1.0


@dataclass
class ClusterMfuResult:
    """Aggregate MFU result for the whole cluster."""
    model_name: str
    model_params: float
    flops_per_token: float
    tokens_per_sec: float
    total_achieved_tflops: float
    total_peak_tflops: float
    aggregate_mfu: float  # 0.0–1.0
    per_gpu: List[GpuMfuResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# GPU hardware registry (BF16 peak TFLOPS)
# ---------------------------------------------------------------------------

# Known BF16 TFLOPS for devices that appear in ags1 or similar clusters.
# Blackwell RTX PRO 6000: ~100 TFLOPS is the conservative community estimate
# for the desktop/workstation SKU (not the datacenter GB200); NVIDIA has not
# published a final datasheet figure as of mid-2025, so this value is marked
# as an approximation and should be updated once official specs are released.
GPU_BF16_TFLOPS_REGISTRY: dict = {
    # Ampere (SM 8.x)
    "A6000":                       38.7,   # NVIDIA RTX A6000, official
    "A100-SXM4-80GB":             312.0,   # A100 SXM4 80 GB, official
    "A100-PCIe-40GB":             312.0,   # A100 PCIe 40 GB, official
    # Hopper (SM 9.x)
    "H100-SXM5":                  989.5,   # H100 SXM5, official (w/ sparsity: 1979)
    "H100-NVL":                   835.0,   # H100 NVL 94 GB, official
    "H100-PCIe":                  756.0,   # H100 PCIe 80 GB, official
    # Blackwell (SM 12.x) — workstation/desktop SKUs
    "RTX-PRO-6000-Blackwell":     100.0,   # ~100 TFLOPS, community estimate (≈2025)
    "RTX-5090":                   209.5,   # RTX 5090, NVIDIA official
}

# ---------------------------------------------------------------------------
# ags1 cluster definition (canonical 5-GPU heterogeneous workstation)
# ---------------------------------------------------------------------------

#: Fixed GPU topology for the ags1 machine:
#:   GPU0, GPU1 — NVIDIA RTX A6000 (48 GB, NUMA 0 / 1)
#:   GPU2       — H100-NVL 96 GB (NUMA 0)
#:   GPU3, GPU4 — RTX PRO 6000 Blackwell (96 GB each, NUMA 1)
AGS1_GPUS: list = [
    GpuInfo(index=0, name="A6000",                   mem_gb=48,  bf16_tflops=38.7,  pcie_gen=4, pcie_bw_gbs=25.0, sm_version="8.6",  numa_node=0),
    GpuInfo(index=1, name="A6000",                   mem_gb=48,  bf16_tflops=38.7,  pcie_gen=4, pcie_bw_gbs=25.0, sm_version="8.6",  numa_node=1),
    GpuInfo(index=2, name="H100-NVL",                mem_gb=96,  bf16_tflops=835.0, pcie_gen=5, pcie_bw_gbs=50.0, sm_version="9.0",  numa_node=0),
    GpuInfo(index=3, name="RTX-PRO-6000-Blackwell",  mem_gb=96,  bf16_tflops=100.0, pcie_gen=4, pcie_bw_gbs=25.0, sm_version="12.0", numa_node=1),
    GpuInfo(index=4, name="RTX-PRO-6000-Blackwell",  mem_gb=96,  bf16_tflops=100.0, pcie_gen=4, pcie_bw_gbs=25.0, sm_version="12.0", numa_node=1),
]

# ---------------------------------------------------------------------------
# Convenience function: ags1 5-GPU aggregate MFU
# ---------------------------------------------------------------------------


def calculate_5gpu_mfu(
    throughput_tokens_per_sec: float,
    model_params: float,
    model_name: str = "unknown",
) -> dict:
    """Calculate aggregate peak TFLOPS and MFU for the ags1 5-GPU cluster.

    Uses the canonical AGS1_GPUS topology (2×A6000 + 1×H100-NVL +
    2×RTX-PRO-6000-Blackwell) with the BF16 TFLOPS values defined in
    GPU_BF16_TFLOPS_REGISTRY.  The Blackwell RTX PRO 6000 figure (~100 TFLOPS)
    is a community estimate; results should be revisited once NVIDIA publishes
    official specs.

    FLOPs convention:
        flops_per_token = 6 × model_params
    (multiply-add ×2, backward-pass ×3 relative to forward)

    Args:
        throughput_tokens_per_sec: Observed aggregate training throughput
            (tokens/sec) across all 5 GPUs.
        model_params: Total number of trainable model parameters (float).
            E.g. 6.74e9 for a 7B model.
        model_name: Optional string label used in the returned dict.

    Returns:
        A dict with the following keys:

        ``model_name``                – echo of the *model_name* argument
        ``model_params``              – echo of *model_params*
        ``throughput_tokens_per_sec`` – echo of input throughput
        ``flops_per_token``           – 6 × model_params
        ``total_peak_tflops``         – sum of BF16 TFLOPS for all 5 GPUs
        ``total_achieved_tflops``     – TFLOPS implied by observed throughput
        ``aggregate_mfu``             – achieved / peak (0.0–1.0)
        ``aggregate_mfu_pct``         – aggregate_mfu × 100 (%)
        ``per_gpu``                   – list of per-device breakdowns (dicts)

    Example::

        >>> result = calculate_5gpu_mfu(throughput_tokens_per_sec=1200, model_params=6.74e9)
        >>> print(f"Aggregate MFU: {result['aggregate_mfu_pct']:.2f}%")
        >>> print(f"Peak TFLOPS:   {result['total_peak_tflops']:.1f}")
    """
    result = compute_mfu(
        gpus=AGS1_GPUS,
        model_params=model_params,
        tokens_per_sec=throughput_tokens_per_sec,
        model_name=model_name,
    )
    out = to_json(result)
    # Propagate the throughput input explicitly for caller convenience
    out["throughput_tokens_per_sec"] = throughput_tokens_per_sec
    return out


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load and return the experiment YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def extract_gpus(config: dict) -> List[GpuInfo]:
    """Extract GPU list from the cluster section of the config."""
    cluster = config.get("cluster", {})
    raw_gpus = cluster.get("gpus", [])
    if not raw_gpus:
        raise ValueError("No GPUs found in config under cluster.gpus")

    gpus = []
    for entry in raw_gpus:
        gpus.append(GpuInfo(
            index=entry["index"],
            name=entry["name"],
            mem_gb=entry.get("mem_gb", 0),
            bf16_tflops=entry["bf16_tflops"],
            pcie_gen=entry.get("pcie_gen", 4),
            pcie_bw_gbs=entry.get("pcie_bw_gbs", 25.0),
            sm_version=entry.get("sm_version", ""),
            numa_node=entry.get("numa_node", 0),
        ))
    return gpus


def extract_model_params(config: dict) -> float:
    """Extract approximate model parameter count from config."""
    model = config.get("model", {})
    # Prefer explicit approximate count if available
    if "params_approx" in model:
        return float(model["params_approx"])

    # Fall back to computing from architecture dimensions
    V = model.get("vocab_size", 32000)
    d = model.get("hidden_size", 4096)
    L = model.get("num_layers", 32)
    h = model.get("num_heads", 32)
    ffn = model.get("intermediate_size", 11008)
    # LLaMA-style: Q,K,V,O attention + SwiGLU FFN (gate, up, down) + embed + lm_head
    attn_per_layer = 4 * d * d  # Q, K, V, O (assuming GQA head_dim matches)
    ffn_per_layer = 3 * d * ffn  # gate + up + down
    embed = 2 * V * d  # token embed + lm_head
    return float(L * (attn_per_layer + ffn_per_layer) + embed)


# ---------------------------------------------------------------------------
# MFU computation
# ---------------------------------------------------------------------------

def compute_mfu(
    gpus: List[GpuInfo],
    model_params: float,
    tokens_per_sec: float,
    model_name: str = "",
) -> ClusterMfuResult:
    """Compute per-GPU and aggregate MFU.

    Total achieved FLOPS = 6 * N * tokens_per_sec.  This is distributed
    across GPUs proportionally to their peak throughput, which models the
    DES-LOC heterogeneous load balancing where faster GPUs process more
    micro-batches per step.
    """
    flops_per_token = 6.0 * model_params
    total_achieved_flops = flops_per_token * tokens_per_sec  # FLOP/s
    total_achieved_tflops = total_achieved_flops / 1e12

    total_peak_tflops = sum(g.bf16_tflops for g in gpus)
    aggregate_mfu = total_achieved_tflops / total_peak_tflops if total_peak_tflops > 0 else 0.0

    # Distribute achieved throughput proportionally to each GPU's peak share
    per_gpu = []
    for g in gpus:
        frac = g.bf16_tflops / total_peak_tflops if total_peak_tflops > 0 else 0.0
        gpu_achieved = total_achieved_tflops * frac
        gpu_mfu = gpu_achieved / g.bf16_tflops if g.bf16_tflops > 0 else 0.0
        per_gpu.append(GpuMfuResult(
            index=g.index,
            name=g.name,
            peak_tflops=g.bf16_tflops,
            achieved_tflops=round(gpu_achieved, 4),
            mfu=round(gpu_mfu, 6),
        ))

    return ClusterMfuResult(
        model_name=model_name,
        model_params=model_params,
        flops_per_token=flops_per_token,
        tokens_per_sec=tokens_per_sec,
        total_achieved_tflops=round(total_achieved_tflops, 4),
        total_peak_tflops=round(total_peak_tflops, 4),
        aggregate_mfu=round(aggregate_mfu, 6),
        per_gpu=per_gpu,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_table(result: ClusterMfuResult) -> None:
    """Print a human-readable per-device breakdown table."""
    print("=" * 80)
    print(f"  MFU Calculator — Heterogeneous Cluster")
    print(f"  Model: {result.model_name}  ({result.model_params:.3e} params)")
    print(f"  Observed throughput: {result.tokens_per_sec:.1f} tokens/sec")
    print(f"  FLOPs/token (6N): {result.flops_per_token:.3e}")
    print("=" * 80)
    print()

    # Per-GPU table
    hdr = f"  {'GPU':>4}  {'Device':<28} {'Peak TFLOPS':>12} {'Achieved':>12} {'MFU':>8}"
    sep = f"  {'─' * 4}  {'─' * 28} {'─' * 12} {'─' * 12} {'─' * 8}"
    print(hdr)
    print(sep)
    for g in result.per_gpu:
        mfu_pct = g.mfu * 100
        print(f"  {g.index:>4}  {g.name:<28} {g.peak_tflops:>10.1f}   {g.achieved_tflops:>10.4f}   {mfu_pct:>6.2f}%")
    print(sep)

    # Aggregate
    agg_pct = result.aggregate_mfu * 100
    print(f"  {'SUM':>4}  {'(cluster aggregate)':<28} "
          f"{result.total_peak_tflops:>10.1f}   {result.total_achieved_tflops:>10.4f}   {agg_pct:>6.2f}%")
    print()


def to_json(result: ClusterMfuResult) -> dict:
    """Convert result to a JSON-serializable dict."""
    return {
        "model_name": result.model_name,
        "model_params": result.model_params,
        "flops_per_token": result.flops_per_token,
        "tokens_per_sec": result.tokens_per_sec,
        "total_achieved_tflops": result.total_achieved_tflops,
        "total_peak_tflops": result.total_peak_tflops,
        "aggregate_mfu": result.aggregate_mfu,
        "aggregate_mfu_pct": round(result.aggregate_mfu * 100, 4),
        "per_gpu": [
            {
                "index": g.index,
                "name": g.name,
                "peak_tflops": g.peak_tflops,
                "achieved_tflops": g.achieved_tflops,
                "mfu": g.mfu,
                "mfu_pct": round(g.mfu * 100, 4),
            }
            for g in result.per_gpu
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MFU calculator for heterogeneous GPU clusters (reads DES-LOC YAML config).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, required=True,
                   help="Path to experiment YAML config (e.g. experiments/configs/7b_ags1_desloc.yaml)")
    p.add_argument("--tokens-per-sec", type=float, required=True,
                   help="Observed aggregate training throughput in tokens/sec")
    p.add_argument("--json", action="store_true",
                   help="Output results as JSON instead of a table")
    p.add_argument("--output", type=str, default=None,
                   help="Write JSON results to this file (implies --json)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    gpus = extract_gpus(config)
    model_params = extract_model_params(config)
    model_name = config.get("model", {}).get("name", "unknown")

    result = compute_mfu(gpus, model_params, args.tokens_per_sec, model_name=model_name)

    if args.output or args.json:
        out = to_json(result)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Results written to {args.output}")
        else:
            print(json.dumps(out, indent=2))
    else:
        print_table(result)


if __name__ == "__main__":
    main()
