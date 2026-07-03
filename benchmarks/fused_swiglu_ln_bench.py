# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team / Neuron_SP
"""
benchmarks/fused_swiglu_ln_bench.py — SwiGLU + RMSNorm fusion benchmark

Benchmarks the fused SwiGLU + RMSLayerNorm CUDA kernel
(csrc/hetero_reduce/fused_swiglu_ln.cu) against an unfused PyTorch baseline.

Kernel computes in a single pass:
    out = RMSNorm(SwiGLU(gate, up))
    SwiGLU(g, u)[i] = g[i] * sigmoid(g[i]) * u[i]   (SiLU × up-projection)
    RMSNorm(x)[i]   = x[i] / rms(x) * weight[i]

Memory model:
    Reads  : gate (BF16) + up (BF16) + ln_weight (FP32) = 2 × B×H×2 + H×4
    Writes : output (BF16)                                = B×H×2
    Effective bytes ≈ 3 × B × H × 2  (ignoring small weight term)

Single-pass regime (hidden ≤ kRegBudget × kVecWidth per SM):
    - H100 SM90: single-pass for hidden ≤ 262,144
    - A6000 SM86: single-pass for hidden ≤ 131,072

Configurations swept:
    batch sizes  : 1, 4, 16, 64, 256, 1024
    hidden dims  : 1024, 2048, 4096, 8192, 14336, 16384

Baselines compared:
    A) naive_unfused   — three separate PyTorch kernels (SiLU, mul, rms_norm)
    B) torch_fused     — torch.compile of the same three ops
    C) flash_swiglu_ln — custom compiled kernel (if available)

Launch:
    python benchmarks/fused_swiglu_ln_bench.py [--device 0] [--iters 50]
        [--warmup 10] [--batches 1 64 256] [--hiddens 4096 8192]
        [--eps 1e-6] [--json results_swiglu.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

import torch
import torch.nn.functional as F

try:
    import deepspeed.ops.fused_swiglu_ln as _ext  # type: ignore[import]
    _HAS_EXT = True
except ImportError:
    _HAS_EXT = False

from kernel_bench import (
    BenchmarkHarness,
    BenchResult,
    print_device_header,
    print_results_table,
    get_device_info,
    print_json,
)

# ---------------------------------------------------------------------------
# Kernel implementations
# ---------------------------------------------------------------------------


def _naive_swiglu_ln(
    gate: torch.Tensor,
    up: torch.Tensor,
    ln_weight: torch.Tensor,
    eps: float,
    out: torch.Tensor,
) -> None:
    """
    Baseline A: naive three-kernel unfused path.

    gate  : (batch, hidden) BF16
    up    : (batch, hidden) BF16
    ln_weight : (hidden,) FP32
    out   : (batch, hidden) BF16 (written in-place)
    """
    # Kernel 1: SwiGLU (SiLU gate × up)
    x = F.silu(gate.float()) * up.float()          # (B, H) FP32

    # Kernel 2+3: RMSNorm (custom loop — torch.nn.RMSNorm requires PyTorch ≥ 2.4)
    #   rms = sqrt(mean(x^2) + eps)
    rms = x.pow(2).mean(dim=-1, keepdim=True).add_(eps).sqrt_()
    x_norm = x / rms * ln_weight.float()

    out.copy_(x_norm.to(torch.bfloat16))


def _torch_native_swiglu_ln(
    gate: torch.Tensor,
    up: torch.Tensor,
    ln_weight: torch.Tensor,
    eps: float,
    out: torch.Tensor,
) -> None:
    """
    Baseline B: torch.nn.functional ops — lets torch dispatch to optimised
    cuBLAS / cuDNN kernels where available.
    """
    x = torch.nn.functional.silu(gate) * up              # BF16 intermediate
    # RMSNorm — torch built-in (PyTorch 2.4+); fall back to manual
    try:
        normed = torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=ln_weight.to(x.dtype), eps=eps)
    except AttributeError:
        rms = x.float().pow(2).mean(-1, keepdim=True).add_(eps).sqrt_()
        normed = (x.float() / rms * ln_weight.float()).to(torch.bfloat16)
    out.copy_(normed)


def _custom_swiglu_ln(
    gate: torch.Tensor,
    up: torch.Tensor,
    ln_weight: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    sm_version: int,
) -> None:
    """
    Custom fused CUDA kernel wrapper.
    Falls back to torch.nn.functional path when extension not compiled.
    """
    if _HAS_EXT:
        _ext.fused_swiglu_ln(out, gate, up, ln_weight, eps, sm_version)
    else:
        _torch_native_swiglu_ln(gate, up, ln_weight, eps, out)


# ---------------------------------------------------------------------------
# Numerical correctness check
# ---------------------------------------------------------------------------

def _check_correctness(device: int, sm_version: int, eps: float = 1e-6) -> bool:
    """
    Verify custom kernel matches naive baseline to within BF16 precision.
    Returns True if OK, False if mismatch.
    """
    batch, hidden = 8, 4096
    dev = f"cuda:{device}"
    gate = torch.randn(batch, hidden, dtype=torch.bfloat16, device=dev)
    up = torch.randn(batch, hidden, dtype=torch.bfloat16, device=dev)
    ln_w = torch.ones(hidden, dtype=torch.float32, device=dev)
    out_naive = torch.empty(batch, hidden, dtype=torch.bfloat16, device=dev)
    out_custom = torch.empty(batch, hidden, dtype=torch.bfloat16, device=dev)

    _naive_swiglu_ln(gate, up, ln_w, eps, out_naive)
    _custom_swiglu_ln(gate, up, ln_w, eps, out_custom, sm_version)
    torch.cuda.synchronize()

    diff = (out_naive.float() - out_custom.float()).abs()
    max_diff = diff.max().item()
    # BF16 relative tolerance ~0.4% (7-bit mantissa)
    return max_diff < 0.02


# ---------------------------------------------------------------------------
# Single-config benchmark
# ---------------------------------------------------------------------------

def _bench_config(
    harness: BenchmarkHarness,
    batch: int,
    hidden: int,
    device: int,
    sm_version: int,
    eps: float,
) -> List[BenchResult]:
    dev = f"cuda:{device}"
    gate = torch.randn(batch, hidden, dtype=torch.bfloat16, device=dev)
    up = torch.randn(batch, hidden, dtype=torch.bfloat16, device=dev)
    ln_w = torch.ones(hidden, dtype=torch.float32, device=dev)
    out = torch.empty(batch, hidden, dtype=torch.bfloat16, device=dev)

    # Memory: read gate + up (2B each BF16) + write out (2B BF16)
    # Ignoring ln_weight (small, stays in L1/L2 after first access)
    bytes_accessed = batch * hidden * 2 * 3   # 3 × (B × H × 2B)
    # FLOPs: SiLU(gate) ≈ 4 FLOP/elem, mul=1, RMSNorm ≈ 4 (sq+acc+sqrt+div+mul)
    flops = batch * hidden * (4 + 1 + 4)

    tag = f"batch={batch:<5d} hidden={hidden}"

    # Baseline A: naive three kernels
    base_naive = harness.run(
        label=f"naive unfused (3 kernels) | {tag}",
        fn=lambda: _naive_swiglu_ln(gate, up, ln_w, eps, out),
        bytes_accessed=bytes_accessed,
        flops=flops,
    )

    # Baseline B: torch.nn.functional
    base_torch = harness.run(
        label=f"torch.nn.functional       | {tag}",
        fn=lambda: _torch_native_swiglu_ln(gate, up, ln_w, eps, out),
        bytes_accessed=bytes_accessed,
        flops=flops,
    )

    # Custom fused kernel
    kernel_label = "fused_swiglu_ln kernel" if _HAS_EXT else "fused_swiglu_ln (sim)"
    custom = harness.run(
        label=f"{kernel_label:<26s} | {tag}",
        fn=lambda: _custom_swiglu_ln(gate, up, ln_w, eps, out, sm_version),
        bytes_accessed=bytes_accessed,
        flops=flops,
        baseline_label=base_naive.label,
    )
    harness.compare_to_baseline(custom, base_naive)

    return [base_naive, base_torch, custom]


# ---------------------------------------------------------------------------
# Roofline helper
# ---------------------------------------------------------------------------

def _print_roofline(device_info: dict, batch: int, hidden: int) -> None:
    """Print the arithmetic intensity and roofline-model prediction."""
    bytes_accessed = batch * hidden * 2 * 3
    flops = batch * hidden * 9
    ai = flops / bytes_accessed   # FLOP/byte

    peak_bw = device_info["peak_hbm_bandwidth_gbs"]  # GB/s
    peak_tf = device_info["peak_bf16_tflops_approx"]  # TFLOPS

    # BW-bound limit: bytes_accessed / peak_bw
    bw_bound_us = (bytes_accessed / (peak_bw * 1e9)) * 1e6
    # Compute-bound limit: flops / peak_flops
    c_bound_us = (flops / (peak_tf * 1e12)) * 1e6
    roofline_us = max(bw_bound_us, c_bound_us)
    bound = "BW-bound" if bw_bound_us > c_bound_us else "compute-bound"

    print(
        f"  Roofline [{batch}×{hidden}]: AI={ai:.2f} FLOP/B  "
        f"lower-bound={roofline_us:.2f} µs  ({bound})"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SwiGLU+RMSNorm fusion benchmark: custom kernel vs PyTorch baselines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--batches", nargs="+", type=int, default=[1, 16, 64, 256],
        help="Batch sizes to sweep",
    )
    parser.add_argument(
        "--hiddens", nargs="+", type=int, default=[4096, 8192, 16384],
        help="Hidden dimensions to sweep",
    )
    parser.add_argument("--eps", type=float, default=1e-6, help="RMSNorm epsilon")
    parser.add_argument("--json", metavar="PATH", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA device available.")
        sys.exit(1)

    device_info = get_device_info(args.device)
    sm_version = device_info["sm_version"]

    print_device_header(args.device)

    if not _HAS_EXT:
        print(
            "WARNING: deepspeed.ops.fused_swiglu_ln not found.\n"
            "         Custom kernel results use torch.nn.functional simulation.\n"
            "         Build with: cd csrc/hetero_reduce && python setup.py install\n"
        )
    else:
        print("Correctness check: ", end="")
        ok = _check_correctness(args.device, sm_version, args.eps)
        print("PASS ✓" if ok else "FAIL ✗ — results may be incorrect")
        print()

    harness = BenchmarkHarness(warmup=args.warmup, iters=args.iters, device=args.device)
    all_results: List[BenchResult] = []

    for hidden in args.hiddens:
        print(f"\n{'─'*70}")
        print(f"  Hidden dim: {hidden}")
        print(f"{'─'*70}")

        group: List[BenchResult] = []
        for batch in args.batches:
            _print_roofline(device_info, batch, hidden)
            results = _bench_config(harness, batch, hidden, args.device, sm_version, args.eps)
            group.extend(results)

        print_results_table(group)
        all_results.extend(group)

    # Summary: speedup of custom kernel vs naive unfused
    print("\n" + "="*70)
    print("  SUMMARY — fused kernel speedup vs naive three-kernel path")
    print("="*70)
    custom_results = [r for r in all_results if "fused_swiglu_ln" in r.label]
    if custom_results:
        print_results_table(custom_results)

    if args.json:
        with open(args.json, "w") as f:
            f.write(print_json(all_results))
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
