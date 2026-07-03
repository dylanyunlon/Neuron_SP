# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team / Neuron_SP
"""
benchmarks/hetero_reduce_bench.py — Reduce-scatter benchmark

Benchmarks the fused BF16→FP32 reduce-scatter custom CUDA kernel
(csrc/hetero_reduce/hetero_reduce.cu) against PyTorch-native equivalents.

The kernel performs:
    output[i] = sum(input_k[i] for k in 0..num_tensors)
in a single fused pass using FP32 accumulation and a BF16 output write-back.

Configurations swept:
    tensor_sizes    : 1M, 4M, 16M, 64M, 128M, 256M elements (BF16)
    num_tensors     : 1, 2, 4, 8, 16, 32
    baselines       : (a) loop of torch.add()  (b) torch.stack().sum()

Metrics reported per config:
    - latency (min / median / p95) in µs
    - effective read bandwidth GB/s
        = num_tensors × n_elems × 2 bytes (BF16) / latency
    - speedup vs best PyTorch baseline

Launch (single GPU):
    python benchmarks/hetero_reduce_bench.py [--device 0] [--iters 50]
        [--warmup 10] [--sizes 1M 16M 128M] [--num-tensors 1 4 8]
        [--json results_hetero.json]

Note: If the hetero_reduce C extension is not compiled, the script falls back
to a pure-PyTorch simulation that exercises the same data-movement pattern.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

import torch

# Try to import the compiled extension; gracefully fall back if absent.
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
# Size parsing helpers
# ---------------------------------------------------------------------------

def _parse_size(s: str) -> int:
    """Parse e.g. '16M', '128M', '1024K', '262144' → integer element count."""
    s = s.strip().upper()
    if s.endswith("M"):
        return int(float(s[:-1]) * 1024 * 1024)
    if s.endswith("K"):
        return int(float(s[:-1]) * 1024)
    return int(s)


# ---------------------------------------------------------------------------
# PyTorch baseline implementations
# ---------------------------------------------------------------------------

def _torch_loop_reduce(inputs: List[torch.Tensor], output: torch.Tensor) -> None:
    """Baseline A: sequential in-place add.  Mimics what a naive DDP loop does."""
    output.copy_(inputs[0])
    for inp in inputs[1:]:
        output.add_(inp)


def _torch_stack_reduce(inputs: List[torch.Tensor], output: torch.Tensor) -> None:
    """Baseline B: torch.stack + sum — generates a temporary but exposes parallelism."""
    stacked = torch.stack(inputs, dim=0)         # (num_tensors, n_elems)
    reduced = stacked.sum(dim=0, dtype=torch.float32).to(torch.bfloat16)
    output.copy_(reduced)


def _torch_fp32_accum_reduce(inputs: List[torch.Tensor], output: torch.Tensor) -> None:
    """
    Baseline C: manual FP32 accumulation loop — closest to what the custom
    kernel does in accuracy, so the speedup vs this is the 'true' kernel gain.
    """
    acc = inputs[0].float()
    for inp in inputs[1:]:
        acc.add_(inp.float())
    output.copy_(acc.to(torch.bfloat16))


# ---------------------------------------------------------------------------
# Custom kernel wrapper
# ---------------------------------------------------------------------------

def _custom_reduce(inputs: List[torch.Tensor], output: torch.Tensor, sm_version: int) -> None:
    """
    Calls the compiled hetero_reduce CUDA extension.

    Falls back to _torch_fp32_accum_reduce when the extension is unavailable,
    clearly annotated in the benchmark label.
    """
    if _HAS_EXT:
        _ext.fused_bf16_reduce(output, inputs, sm_version)
    else:
        _torch_fp32_accum_reduce(inputs, output)


# ---------------------------------------------------------------------------
# Single-config benchmark
# ---------------------------------------------------------------------------

def _bench_config(
    harness: BenchmarkHarness,
    n_elems: int,
    num_tensors: int,
    device: int,
    sm_version: int,
) -> List[BenchResult]:
    """
    Run all baselines + custom kernel for one (n_elems, num_tensors) config.
    Returns list of BenchResult in order: [loop, stack, fp32_loop, custom].
    """
    dev = f"cuda:{device}"
    inputs = [torch.randn(n_elems, dtype=torch.bfloat16, device=dev) for _ in range(num_tensors)]
    output = torch.empty(n_elems, dtype=torch.bfloat16, device=dev)

    # bytes read = num_tensors × n_elems × 2B  |  bytes write = n_elems × 2B
    bytes_accessed = (num_tensors + 1) * n_elems * 2
    # FP32 accumulate: num_tensors adds per element
    flops = num_tensors * n_elems

    size_tag = f"n={n_elems//1024//1024}M" if n_elems >= 1024*1024 else f"n={n_elems//1024}K"
    tag = f"{size_tag} / nT={num_tensors}"

    # Baseline A: torch loop add
    base_loop = harness.run(
        label=f"torch.add loop           | {tag}",
        fn=lambda: _torch_loop_reduce(inputs, output),
        bytes_accessed=bytes_accessed,
        flops=flops,
    )

    # Baseline B: stack + sum
    base_stack = harness.run(
        label=f"torch.stack.sum          | {tag}",
        fn=lambda: _torch_stack_reduce(inputs, output),
        bytes_accessed=bytes_accessed,
        flops=flops,
    )

    # Baseline C: manual FP32 accum (fairest comparison for custom kernel)
    base_fp32 = harness.run(
        label=f"torch fp32 accum loop    | {tag}",
        fn=lambda: _torch_fp32_accum_reduce(inputs, output),
        bytes_accessed=bytes_accessed,
        flops=flops,
    )

    # Custom kernel
    ext_label = "hetero_reduce CUDA ext" if _HAS_EXT else "hetero_reduce (PyTorch sim)"
    custom = harness.run(
        label=f"{ext_label:<24s} | {tag}",
        fn=lambda: _custom_reduce(inputs, output, sm_version),
        bytes_accessed=bytes_accessed,
        flops=flops,
        baseline_label=base_fp32.label,
    )
    harness.compare_to_baseline(custom, base_fp32)

    return [base_loop, base_stack, base_fp32, custom]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hetero reduce-scatter benchmark: custom kernel vs PyTorch baselines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=50, help="Measurement iterations")
    parser.add_argument(
        "--sizes", nargs="+", default=["1M", "4M", "16M", "64M", "128M"],
        help="Tensor sizes to sweep, e.g. 1M 16M 128M",
    )
    parser.add_argument(
        "--num-tensors", nargs="+", type=int, default=[1, 2, 4, 8],
        help="Number of input tensors to reduce",
    )
    parser.add_argument(
        "--json", metavar="PATH", default=None,
        help="Optional path to write JSON results",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA device available.")
        sys.exit(1)

    device_info = get_device_info(args.device)
    sm_version = device_info["sm_version"]

    print_device_header(args.device)

    if not _HAS_EXT:
        print(
            "WARNING: deepspeed.ops.hetero_reduce not found.\n"
            "         Custom kernel results use PyTorch FP32 simulation.\n"
            "         Build with: cd csrc/hetero_reduce && python setup.py install\n"
        )

    harness = BenchmarkHarness(warmup=args.warmup, iters=args.iters, device=args.device)

    all_results: List[BenchResult] = []

    for size_str in args.sizes:
        n_elems = _parse_size(size_str)
        size_bytes_mb = n_elems * 2 / 1e6

        print(f"\n{'─'*70}")
        print(f"  Tensor size: {size_str}  ({size_bytes_mb:.0f} MB BF16 per tensor)")
        print(f"{'─'*70}")

        group: List[BenchResult] = []
        for num_tensors in args.num_tensors:
            results = _bench_config(harness, n_elems, num_tensors, args.device, sm_version)
            group.extend(results)

        print_results_table(group)
        all_results.extend(group)

    # Summary: custom kernel speedup across all configs
    print("\n" + "="*70)
    print("  SUMMARY — custom kernel speedup vs torch fp32 accum loop")
    print("="*70)
    custom_results = [r for r in all_results if "hetero_reduce" in r.label]
    summary_rows: List[BenchResult] = []
    for r in custom_results:
        summary_rows.append(r)
    if summary_rows:
        print_results_table(summary_rows)

    # JSON output
    if args.json:
        with open(args.json, "w") as f:
            f.write(print_json(all_results))
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
