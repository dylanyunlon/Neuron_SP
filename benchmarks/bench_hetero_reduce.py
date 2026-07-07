#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
#
# DeepSpeed Team / Neuron_SP — issue #143
#
# benchmarks/bench_hetero_reduce.py
# ==================================
# Allreduce throughput benchmark for csrc/hetero_reduce production kernels.
#
# Measures effective bandwidth (GB/s) for:
#   1. hetero_ring_reduce_step   — single ring reduce-scatter step (C++ kernel)
#   2. hetero_ring_gather_step   — single ring all-gather step (C++ kernel)
#   3. pcie_ring_reduce_step     — double-buffered PCIe ring reduce step (C++ kernel)
#   4. torch.add_ baseline       — naive element-wise add (PyTorch)
#   5. torch.stack().sum()       — vectorised sum baseline (PyTorch)
#
# Metrics reported per (tensor_size, world_size) config:
#   - latency  : min / median / p95 in µs
#   - bandwidth: effective GB/s  = bytes_touched / latency
#   - speedup  : vs torch.add_ baseline
#
# Usage (single GPU, no inter-process comm — measures pure kernel throughput)
# --------------------------------------------------------------------------
#   python benchmarks/bench_hetero_reduce.py
#   python benchmarks/bench_hetero_reduce.py --device 0 --iters 100 --warmup 20
#   python benchmarks/bench_hetero_reduce.py --sizes 1M 16M 128M 256M
#   python benchmarks/bench_hetero_reduce.py --json results.json
#
# Note: If the hetero_reduce C extension is not compiled, all hetero kernel
#       results are replaced by their PyTorch-simulation equivalents, clearly
#       labelled.  Build with:
#           pip install -e . --no-build-isolation
#       or:
#           cd csrc/hetero_reduce && python setup.py install

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from typing import Callable, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Extension import — graceful fallback
# ---------------------------------------------------------------------------
try:
    from deepspeed.ops.hetero_reduce import HeteroReduceOp as _HRed
    _op = _HRed()
    _HAS_EXT = True
except Exception:
    _op = None
    _HAS_EXT = False


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class BenchResult:
    label: str
    n_elems: int
    dtype_bytes: int
    latency_us_min: float
    latency_us_med: float
    latency_us_p95: float
    bw_gbps: float
    speedup: float = 1.0  # relative to baseline (filled in after run)

    @property
    def size_mb(self) -> float:
        return self.n_elems * self.dtype_bytes / 1e6


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------
def _cuda_time_us(fn: Callable, warmup: int, iters: int, device: int) -> List[float]:
    """Return list of per-iteration latencies in µs."""
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize(device)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    times = []
    for _ in range(iters):
        start_ev.record()
        fn()
        end_ev.record()
        torch.cuda.synchronize(device)
        times.append(start_ev.elapsed_time(end_ev) * 1e3)  # ms → µs
    return times


def _stats(times: List[float]) -> Tuple[float, float, float]:
    s = sorted(times)
    n = len(s)
    p95_idx = max(0, int(math.ceil(0.95 * n)) - 1)
    return s[0], s[n // 2], s[p95_idx]


# ---------------------------------------------------------------------------
# Size parsing
# ---------------------------------------------------------------------------
def _parse_size(s: str) -> int:
    s = s.strip().upper()
    if s.endswith("M"):
        return int(float(s[:-1]) * 1024 * 1024)
    if s.endswith("K"):
        return int(float(s[:-1]) * 1024)
    return int(s)


def _size_tag(n_elems: int) -> str:
    if n_elems >= 1024 * 1024:
        return f"{n_elems // (1024*1024)}M"
    if n_elems >= 1024:
        return f"{n_elems // 1024}K"
    return str(n_elems)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def _run_config(
    n_elems: int,
    world_size: int,
    warmup: int,
    iters: int,
    device: int,
    sm_version: int,
) -> List[BenchResult]:
    """
    Benchmark a single (tensor_size, world_size) configuration.

    We allocate one BF16 tensor per simulated "peer" and measure the time
    for a single reduce step that processes all of them.  This exercises the
    pure kernel throughput on a single GPU — real multi-GPU comm latency
    involves cudaMemcpyPeerAsync and is measured separately by the C-level
    bench programs in benchmarks/*.cu.

    Bytes touched per step:
      reduce_step : 2 reads (accum + recv) + 1 write  = 3 × n_elems × 2 bytes
      gather_step : 1 read  + 1 write                  = 2 × n_elems × 2 bytes
    """
    dev = f"cuda:{device}"
    bf16 = torch.bfloat16

    # Two BF16 tensors: accumulator (a) and receive buffer (b).
    a = torch.randn(n_elems, dtype=bf16, device=dev)
    b = torch.randn(n_elems, dtype=bf16, device=dev)

    # Additional tensors for multi-input baselines.
    inputs = [torch.randn(n_elems, dtype=bf16, device=dev) for _ in range(world_size)]
    output = torch.empty(n_elems, dtype=bf16, device=dev)

    stag = _size_tag(n_elems)
    tag  = f"n={stag} / ws={world_size}"

    results: List[BenchResult] = []

    # ------------------------------------------------------------------
    # Baseline A: torch.add_ (sequential in-place add across world_size
    #             tensors; mirrors what a naive DDP reduction loop does)
    # ------------------------------------------------------------------
    def _torch_add_loop():
        output.copy_(inputs[0])
        for inp in inputs[1:]:
            output.add_(inp)

    # bytes: world_size reads + 1 write
    bw_bytes_loop = (world_size + 1) * n_elems * 2

    t_loop = _cuda_time_us(lambda: _torch_add_loop(), warmup, iters, device)
    mn, med, p95 = _stats(t_loop)
    bw_base = bw_bytes_loop / (mn * 1e-6) / 1e9
    r_base = BenchResult(
        label=f"torch.add_ loop            | {tag}",
        n_elems=n_elems, dtype_bytes=2,
        latency_us_min=mn, latency_us_med=med, latency_us_p95=p95,
        bw_gbps=bw_base,
    )
    results.append(r_base)

    # ------------------------------------------------------------------
    # Baseline B: torch.stack().sum() — exposes parallelism, one alloc
    # ------------------------------------------------------------------
    def _torch_stack_sum():
        output.copy_(torch.stack(inputs, dim=0).sum(dim=0, dtype=torch.float32).to(bf16))

    t_stk = _cuda_time_us(lambda: _torch_stack_sum(), warmup, iters, device)
    mn, med, p95 = _stats(t_stk)
    bw_stk = bw_bytes_loop / (mn * 1e-6) / 1e9
    results.append(BenchResult(
        label=f"torch.stack.sum            | {tag}",
        n_elems=n_elems, dtype_bytes=2,
        latency_us_min=mn, latency_us_med=med, latency_us_p95=p95,
        bw_gbps=bw_stk,
    ))

    # ------------------------------------------------------------------
    # hetero_ring_reduce_step  (accum += recv — one ring step, BF16→FP32→BF16)
    # bytes: 2 reads + 1 write = 3 × n × 2
    # ------------------------------------------------------------------
    bw_bytes_step = 3 * n_elems * 2

    if _HAS_EXT:
        def _hetero_ring_reduce():
            _op.hetero_ring_reduce_step(a, b, sm_version)
        fn_rr = _hetero_ring_reduce
        lbl_rr = "hetero_ring_reduce_step    | " + tag
    else:
        def _sim_ring_reduce():
            a.add_(b.float().to(bf16))
        fn_rr = _sim_ring_reduce
        lbl_rr = "hetero_ring_reduce (sim)   | " + tag

    t_rr = _cuda_time_us(fn_rr, warmup, iters, device)
    mn, med, p95 = _stats(t_rr)
    bw_rr = bw_bytes_step / (mn * 1e-6) / 1e9
    speedup_rr = r_base.bw_gbps / bw_rr if bw_rr > 0 else 0.0
    r_rr = BenchResult(
        label=lbl_rr,
        n_elems=n_elems, dtype_bytes=2,
        latency_us_min=mn, latency_us_med=med, latency_us_p95=p95,
        bw_gbps=bw_rr, speedup=speedup_rr,
    )
    results.append(r_rr)

    # ------------------------------------------------------------------
    # hetero_ring_gather_step  (output = recv — vectorised copy)
    # bytes: 1 read + 1 write = 2 × n × 2
    # ------------------------------------------------------------------
    bw_bytes_gather = 2 * n_elems * 2

    if _HAS_EXT:
        def _hetero_ring_gather():
            _op.hetero_ring_gather_step(a, b, sm_version)
        fn_rg = _hetero_ring_gather
        lbl_rg = "hetero_ring_gather_step    | " + tag
    else:
        def _sim_ring_gather():
            a.copy_(b)
        fn_rg = _sim_ring_gather
        lbl_rg = "hetero_ring_gather (sim)   | " + tag

    t_rg = _cuda_time_us(fn_rg, warmup, iters, device)
    mn, med, p95 = _stats(t_rg)
    bw_rg = bw_bytes_gather / (mn * 1e-6) / 1e9
    results.append(BenchResult(
        label=lbl_rg,
        n_elems=n_elems, dtype_bytes=2,
        latency_us_min=mn, latency_us_med=med, latency_us_p95=p95,
        bw_gbps=bw_rg,
    ))

    # ------------------------------------------------------------------
    # pcie_ring_reduce_step  (double-buffered PCIe ring reduce step)
    # Same byte pattern as hetero_ring_reduce_step.
    # ------------------------------------------------------------------
    if _HAS_EXT:
        def _pcie_ring_reduce():
            _op.pcie_ring_reduce_step(a, b, sm_version)
        fn_pr = _pcie_ring_reduce
        lbl_pr = "pcie_ring_reduce_step      | " + tag
    else:
        def _sim_pcie_ring_reduce():
            a.add_(b)
        fn_pr = _sim_pcie_ring_reduce
        lbl_pr = "pcie_ring_reduce (sim)     | " + tag

    t_pr = _cuda_time_us(fn_pr, warmup, iters, device)
    mn, med, p95 = _stats(t_pr)
    bw_pr = bw_bytes_step / (mn * 1e-6) / 1e9
    results.append(BenchResult(
        label=lbl_pr,
        n_elems=n_elems, dtype_bytes=2,
        latency_us_min=mn, latency_us_med=med, latency_us_p95=p95,
        bw_gbps=bw_pr,
    ))

    # Fill in speedup vs torch.add_ baseline for hetero ring reduce step
    for r in results:
        if r.speedup == 1.0 and r is not r_base:
            r.speedup = r_base.latency_us_min / r.latency_us_min if r.latency_us_min > 0 else 0.0

    return results


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------
_HDR = (
    f"{'Kernel':<42s}  {'min µs':>8s}  {'med µs':>8s}  {'p95 µs':>8s}  "
    f"{'BW GB/s':>9s}  {'speedup':>7s}"
)
_SEP = "─" * len(_HDR)


def _print_results(results: List[BenchResult]) -> None:
    print(_HDR)
    print(_SEP)
    for r in results:
        print(
            f"{r.label:<42s}  "
            f"{r.latency_us_min:>8.2f}  "
            f"{r.latency_us_med:>8.2f}  "
            f"{r.latency_us_p95:>8.2f}  "
            f"{r.bw_gbps:>9.2f}  "
            f"{r.speedup:>6.2f}x"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "hetero_reduce allreduce throughput benchmark (issue #143).\n"
            "Measures effective bandwidth of ring reduce-scatter and gather\n"
            "kernels in csrc/hetero_reduce/ vs PyTorch baselines."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device",  type=int, default=0,
                        help="CUDA device index (default: 0)")
    parser.add_argument("--warmup",  type=int, default=20,
                        help="Warmup iterations (default: 20)")
    parser.add_argument("--iters",   type=int, default=100,
                        help="Measurement iterations (default: 100)")
    parser.add_argument(
        "--sizes", nargs="+",
        default=["1M", "4M", "16M", "64M", "128M", "256M"],
        help="BF16 tensor sizes to sweep (default: 1M 4M 16M 64M 128M 256M)",
    )
    parser.add_argument(
        "--world-sizes", nargs="+", type=int,
        default=[2, 4, 5, 8],
        help="Simulated world sizes for multi-tensor baselines (default: 2 4 5 8)",
    )
    parser.add_argument("--json", metavar="PATH", default=None,
                        help="Write JSON results to PATH")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no CUDA device found.", file=sys.stderr)
        sys.exit(1)

    torch.cuda.set_device(args.device)
    props = torch.cuda.get_device_properties(args.device)
    sm_version = props.major * 10 + props.minor

    print(f"\n{'='*70}")
    print(f"  hetero_reduce allreduce throughput benchmark  (issue #143)")
    print(f"{'='*70}")
    print(f"  Device : {props.name}  (SM {props.major}.{props.minor}  →  sm_version={sm_version})")
    print(f"  VRAM   : {props.total_memory / 1e9:.1f} GB")
    print(f"  Warmup : {args.warmup}  iters  |  Measure: {args.iters} iters")
    if not _HAS_EXT:
        print(
            "\n  WARNING: deepspeed.ops.hetero_reduce not found.\n"
            "  Kernel results show PyTorch simulation only.\n"
            "  Build with:  pip install -e . --no-build-isolation\n"
        )
    print()

    all_results: List[BenchResult] = []

    for size_str in args.sizes:
        n_elems = _parse_size(size_str)
        size_mb = n_elems * 2 / 1e6
        print(f"\n{'─'*70}")
        print(f"  Tensor: {size_str}  ({size_mb:.0f} MB BF16 per buffer)")
        print(f"{'─'*70}")

        group: List[BenchResult] = []
        for ws in args.world_sizes:
            sub = _run_config(
                n_elems=n_elems,
                world_size=ws,
                warmup=args.warmup,
                iters=args.iters,
                device=args.device,
                sm_version=sm_version,
            )
            group.extend(sub)

        _print_results(group)
        all_results.extend(group)

    # Summary: hetero ring reduce step only
    print(f"\n{'='*70}")
    print("  SUMMARY — hetero_ring_reduce_step effective bandwidth")
    print(f"{'='*70}")
    key_results = [r for r in all_results if "ring_reduce_step" in r.label or "ring_reduce (sim)" in r.label]
    if key_results:
        _print_results(key_results)

    if args.json:
        with open(args.json, "w") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        print(f"\nResults written to {args.json}")


if __name__ == "__main__":
    main()
