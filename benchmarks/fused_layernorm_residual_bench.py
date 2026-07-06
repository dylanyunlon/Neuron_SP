# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team / Neuron_SP
"""
benchmarks/fused_layernorm_residual_bench.py
    — Fused residual add + RMSNorm benchmark

Benchmarks the fused_layernorm_residual CUDA kernel
(csrc/hetero_reduce/fused_layernorm_residual.cu) against an unfused
PyTorch baseline.

Kernel computes in a single pass:
    residual_i += input_i
    output_i   = RMSNorm(residual_i) * weight

Memory model (single-pass path):
    Reads  : input (BF16) + residual_in (BF16) + ln_weight (FP32 ≈ 0)
    Writes : residual_out (BF16) + output (BF16)
    Effective bytes ≈ 8 × batch × hidden bytes

Unfused path adds an extra read of residual_out in the norm kernel:
    Effective bytes ≈ 10 × batch × hidden bytes

Configurations swept:
    batch sizes  : 1, 8, 64, 512, 2048
    hidden dims  : 1024, 4096, 8192, 14336, 16384

Baselines compared:
    A) naive_unfused   — two separate PyTorch kernels (add, rms_norm)
    B) torch_compiled  — torch.compile of the same two ops (if available)
    C) fused kernel    — custom CUDA kernel (via hetero_reduce extension)

Launch:
    python benchmarks/fused_layernorm_residual_bench.py [--device 0] [--iters 50]
        [--warmup 10] [--batches 1 64 2048] [--hiddens 4096 8192]
        [--eps 1e-6] [--json results_ln_residual.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

import torch
import torch.nn.functional as F

try:
    import deepspeed.ops.hetero_reduce as _ext  # type: ignore[import]
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


def _naive_ln_residual(
    output:   torch.Tensor,
    residual: torch.Tensor,
    inp:      torch.Tensor,
    weight:   torch.Tensor,
    eps:      float,
) -> None:
    """
    Baseline A: naive two-kernel unfused path.

    residual : (batch, hidden) BF16  — updated in-place
    inp      : (batch, hidden) BF16  — new sub-layer contribution
    output   : (batch, hidden) BF16  — LN normalised result
    weight   : (hidden,)       FP32  — RMSNorm scale
    """
    # Kernel 1: residual add
    residual.add_(inp)

    # Kernel 2: RMSNorm
    x = residual.float()
    rms = x.pow(2).mean(dim=-1, keepdim=True).add_(eps).sqrt_()
    normed = x / rms * weight.float()
    output.copy_(normed.to(torch.bfloat16))


def _torch_compile_ln_residual(
    output:   torch.Tensor,
    residual: torch.Tensor,
    inp:      torch.Tensor,
    weight:   torch.Tensor,
    eps:      float,
    _fn,
) -> None:
    """Baseline B: torch.compile of the same two operations."""
    _fn(output, residual, inp, weight, eps)


def _custom_ln_residual(
    output:   torch.Tensor,
    residual: torch.Tensor,
    inp:      torch.Tensor,
    weight:   torch.Tensor,
    eps:      float,
    sm_version: int,
) -> None:
    """Custom fused CUDA kernel wrapper."""
    if _HAS_EXT:
        _ext.fused_layernorm_residual(output, residual, inp, weight, eps, sm_version)
    else:
        # Simulation: use naive path (does NOT exercise the custom kernel)
        _naive_ln_residual(output, residual, inp, weight, eps)


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------


def _check_correctness(device: int, sm_version: int, eps: float = 1e-6) -> bool:
    """Compare custom kernel output to naive reference within BF16 tolerance."""
    batch, hidden = 8, 4096
    dev = f"cuda:{device}"
    inp = torch.randn(batch, hidden, dtype=torch.bfloat16, device=dev) * 0.3
    res_naive  = torch.randn(batch, hidden, dtype=torch.bfloat16, device=dev)
    res_custom = res_naive.clone()
    out_naive  = torch.empty(batch, hidden, dtype=torch.bfloat16, device=dev)
    out_custom = torch.empty(batch, hidden, dtype=torch.bfloat16, device=dev)
    weight = torch.ones(hidden, dtype=torch.float32, device=dev)

    _naive_ln_residual(out_naive, res_naive, inp, weight, eps)
    _custom_ln_residual(out_custom, res_custom, inp, weight, eps, sm_version)
    torch.cuda.synchronize()

    out_diff = (out_naive.float() - out_custom.float()).abs().max().item()
    res_diff = (res_naive.float() - res_custom.float()).abs().max().item()
    return out_diff < 0.02 and res_diff < 0.005


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
    compiled_fn=None,
) -> List[BenchResult]:
    dev = f"cuda:{device}"

    def _make_tensors():
        inp     = torch.randn(batch, hidden, dtype=torch.bfloat16, device=dev) * 0.3
        residual = torch.randn(batch, hidden, dtype=torch.bfloat16, device=dev)
        weight  = torch.ones(hidden, dtype=torch.float32, device=dev)
        output  = torch.empty(batch, hidden, dtype=torch.bfloat16, device=dev)
        return inp, residual, weight, output

    # Effective BW: reads (input + residual_in) + writes (residual_out + output)
    #   = 4 × batch × hidden × 2 bytes  (all BF16)
    bytes_accessed = 4 * batch * hidden * 2
    # FLOPs per element: add=1, sq=1, acc=1 (reduce), rsqrt≈4, mul+scale=2 → ~10
    flops = batch * hidden * 10

    tag = f"batch={batch:<5d} hidden={hidden}"

    # ── Baseline A: naive unfused ────────────────────────────────────────
    inp, residual, weight, output = _make_tensors()
    base_naive = harness.run(
        label=f"naive unfused (add+rms)    | {tag}",
        fn=lambda: _naive_ln_residual(output, residual, inp, weight, eps),
        bytes_accessed=bytes_accessed,
        flops=flops,
    )

    # ── Baseline B: torch.compile ────────────────────────────────────────
    inp, residual, weight, output = _make_tensors()
    if compiled_fn is not None:
        base_compile = harness.run(
            label=f"torch.compile (add+rms)   | {tag}",
            fn=lambda: _torch_compile_ln_residual(
                output, residual, inp, weight, eps, compiled_fn),
            bytes_accessed=bytes_accessed,
            flops=flops,
        )
    else:
        base_compile = base_naive  # skip if torch.compile unavailable

    # ── Custom fused kernel ───────────────────────────────────────────────
    inp, residual, weight, output = _make_tensors()
    kernel_label = "fused_ln_residual kernel" if _HAS_EXT else "fused_ln_residual (sim)"
    custom = harness.run(
        label=f"{kernel_label:<26s} | {tag}",
        fn=lambda: _custom_ln_residual(output, residual, inp, weight, eps, sm_version),
        bytes_accessed=bytes_accessed,
        flops=flops,
        baseline_label=base_naive.label,
    )
    harness.compare_to_baseline(custom, base_naive)

    return [base_naive, base_compile, custom] if compiled_fn else [base_naive, custom]


# ---------------------------------------------------------------------------
# Roofline helper
# ---------------------------------------------------------------------------


def _print_roofline(device_info: dict, batch: int, hidden: int) -> None:
    bytes_accessed = 4 * batch * hidden * 2   # 4 BF16 tensors
    flops          = batch * hidden * 10
    ai = flops / bytes_accessed

    peak_bw = device_info["peak_hbm_bandwidth_gbs"]
    peak_tf = device_info["peak_bf16_tflops_approx"]

    bw_bound_us = (bytes_accessed / (peak_bw * 1e9)) * 1e6
    c_bound_us  = (flops / (peak_tf * 1e12)) * 1e6
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
        description="Fused residual add + RMSNorm benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device",  type=int,   default=0)
    parser.add_argument("--warmup",  type=int,   default=10)
    parser.add_argument("--iters",   type=int,   default=50)
    parser.add_argument(
        "--batches", nargs="+", type=int,
        default=[1, 8, 64, 512, 2048],
    )
    parser.add_argument(
        "--hiddens", nargs="+", type=int,
        default=[1024, 4096, 8192, 16384],
    )
    parser.add_argument("--eps",  type=float, default=1e-6)
    parser.add_argument("--json", metavar="PATH", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA device available.")
        sys.exit(1)

    device_info = get_device_info(args.device)
    sm_version  = device_info["sm_version"]

    print_device_header(args.device)

    # Try to set up torch.compile baseline (PyTorch ≥ 2.0)
    compiled_fn = None
    try:
        def _naive_raw(output, residual, inp, weight, eps):
            residual.add_(inp)
            x = residual.float()
            rms = x.pow(2).mean(-1, keepdim=True).add_(eps).sqrt_()
            output.copy_((x / rms * weight.float()).to(torch.bfloat16))

        compiled_fn = torch.compile(_naive_raw)
    except Exception:
        pass

    if not _HAS_EXT:
        print(
            "WARNING: deepspeed.ops.hetero_reduce not found.\n"
            "         Custom kernel results use naive simulation.\n"
            "         Build with: cd csrc/hetero_reduce && python setup.py install\n"
        )
    else:
        print("Correctness check: ", end="", flush=True)
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
            results = _bench_config(
                harness, batch, hidden, args.device, sm_version,
                args.eps, compiled_fn)
            group.extend(results)

        print_results_table(group)
        all_results.extend(group)

    # Single-pass threshold summary
    reg_budget = 128 if sm_version >= 90 else 64
    block_size = 512 if sm_version >= 120 else 256
    max_sp = block_size * 8 * reg_budget
    print("\n" + "="*70)
    print(f"  SM{sm_version}: single-pass threshold = hidden ≤ {max_sp:,} elements")
    print(f"  Llama-7B (4096), Llama-70B (8192): single-pass → zero extra DRAM reads")
    print("="*70)

    # Summary table: fused kernel only
    fused_results = [r for r in all_results if "fused_ln_residual" in r.label]
    if fused_results:
        print("\n  SUMMARY — fused_layernorm_residual speedup vs naive two-kernel path")
        print_results_table(fused_results)

    if args.json:
        with open(args.json, "w") as f:
            f.write(print_json(all_results))
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
