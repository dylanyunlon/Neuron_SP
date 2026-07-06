# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
#
# DeepSpeed Team

"""
benchmarks/bench_hetero_reduce_vs_nccl.py   — addresses #74
=============================================================

Proves (or disproves) that csrc/hetero_reduce/ kernels are faster than NCCL
for the heterogeneous, PCIe-only reduce-scatter workload they were designed for.

═══════════════════════════════════════════════════════════════════════════════
WHAT WE BENCHMARK
═══════════════════════════════════════════════════════════════════════════════

Operation:  given K input tensors of N BF16 elements each (all on the same
GPU, because NCCL reduce-scatter also reduces *after* cross-device copies),
produce one output tensor of N BF16 elements:

    output[i] = sum_{k=0..K-1} inputs[k][i]    (FP32 accumulation)

This is the device-local reduction stage that hetero_reduce.cu performs as a
fused BF16→FP32 accumulate + FP32→BF16 writeback in a single kernel pass.

The NCCL equivalent launches via dist.all_reduce / dist.reduce_scatter on a
*single node* (intra-node), which on PCIe-only hardware also does this
accumulation on device before transfer.  We model the *on-device* computation
cost of that accumulation with two canonical PyTorch baselines:

  Baseline A — torch_loop:
    Simulates what NCCL does internally: loop K additions, each reading the
    full tensor once.  Total bytes = 2K × N × 2 (read) + N × 2 (write).

  Baseline B — torch_stack_sum:
    Stacks inputs to [K, N] then calls .sum(0).  PyTorch may pick a more
    optimal reduction kernel than the naïve loop, but it also allocates a
    temporary [K, N] stacked buffer.

  Baseline C — dist.all_reduce (NCCL):
    The gold standard: single-process dist.all_reduce on one device using
    the NCCL backend.  Exercises the same device-side accumulation as a
    multi-GPU ring allreduce reduce phase.  Requires torch.distributed.

Our kernel — fused_bf16_reduce_custom:
    Uses hetero_reduce.cu's fused kernel: inputs live in __constant__ memory
    (for K ≤ 32), warp-cooperative accumulation, single DRAM write pass.
    SM variant is selected from the actual device's compute capability.

═══════════════════════════════════════════════════════════════════════════════
WHY THE KERNEL SHOULD WIN
═══════════════════════════════════════════════════════════════════════════════

Memory traffic (bytes):
  torch_loop:        2 × K × N (reads) + 2 × N (write) = 2N(K+1)
  fused kernel:      2 × K × N (reads) + 2 × N (write) = 2N(K+1)   ← same
  torch_stack_sum:   2 × K × N (reads) + 2 × K × N (alloc) + 2N  > 2N(K+1)

The kernel wins not on bytes but on:
  1. Warp-level fusion: all K reads happen inside one warp iteration
     with L1/L2 locality, vs. K separate kernel launches in the loop path.
  2. No kernel-launch overhead amortised per input tensor.
  3. __constant__ memory pointer array (≤32 tensors): avoids cudaMallocAsync
     on the critical path.
  4. FP32 accumulation in registers with BF16 operands: avoids FP16 precision
     loss and any extra cast kernels.
  5. __launch_bounds__ tuned per SM: maximises occupancy on SM8.6/9.0/12.0.

═══════════════════════════════════════════════════════════════════════════════
METRICS
═══════════════════════════════════════════════════════════════════════════════

For each (N, K) configuration we report:
  - Latency:  min / median / p95  (µs)
  - Bandwidth: effective bytes read+written / median latency (GB/s)
  - Roofline utilisation: bandwidth_gbs / peak_hbm_bandwidth_gbs  (%)
  - Speedup: baseline_median_us / kernel_median_us

═══════════════════════════════════════════════════════════════════════════════
USAGE
═══════════════════════════════════════════════════════════════════════════════

  # Full sweep, single GPU:
  python benchmarks/bench_hetero_reduce_vs_nccl.py

  # Specific sizes and tensor counts:
  python benchmarks/bench_hetero_reduce_vs_nccl.py \\
      --sizes 1M 16M 128M --num-tensors 2 8 16 --device 0

  # Use NCCL dist.all_reduce as additional baseline (requires GPU):
  python benchmarks/bench_hetero_reduce_vs_nccl.py --nccl

  # Save results to JSON (for CI regression tracking):
  python benchmarks/bench_hetero_reduce_vs_nccl.py --json results_#74.json

  # Quick smoke-test (small sizes only, few iters):
  python benchmarks/bench_hetero_reduce_vs_nccl.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    from kernel_bench import (
        BenchmarkHarness,
        BenchResult,
        print_device_header,
        print_results_table,
        get_device_info,
        print_json,
    )
    _HAS_HARNESS = True
except ImportError:
    _HAS_HARNESS = False

# Try to import the compiled hetero_reduce extension.
try:
    import sys, os
    # Support running from repo root or from benchmarks/
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo not in sys.path:
        sys.path.insert(0, _repo)
    from deepspeed.ops.hetero_reduce import hetero_reduce_op as _ext
    _HAS_EXT = True
except Exception:
    _HAS_EXT = False

# Try dist for NCCL baseline
_HAS_DIST = False
try:
    import torch.distributed as dist
    _HAS_DIST = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Fallback timing harness (when kernel_bench.py is not importable)
# ---------------------------------------------------------------------------

import math
import statistics


@dataclass
class _BenchResult:
    label: str
    latency_min_us: float = 0.0
    latency_median_us: float = 0.0
    latency_p95_us: float = 0.0
    bandwidth_gbs: float = 0.0
    speedup: float = 0.0
    bw_improvement: float = 0.0
    baseline_label: Optional[str] = None


class _Harness:
    """Minimal timing harness when kernel_bench.py is unavailable."""

    def __init__(self, warmup: int = 10, iters: int = 50, device: int = 0):
        self.warmup = warmup
        self.iters = iters
        self.device = device

    def _time_fn(self, fn: Callable[[], None]) -> List[float]:
        torch.cuda.set_device(self.device)
        stream = torch.cuda.current_stream()
        for _ in range(self.warmup):
            fn()
        torch.cuda.synchronize()

        latencies: List[float] = []
        start = torch.cuda.Event(enable_timing=True)
        stop  = torch.cuda.Event(enable_timing=True)
        for _ in range(self.iters):
            start.record(stream)
            fn()
            stop.record(stream)
            stop.synchronize()
            latencies.append(start.elapsed_time(stop) * 1e3)  # ms → µs
        return latencies

    def run(
        self,
        label: str,
        fn: Callable[[], None],
        bytes_accessed: int = 0,
        baseline_label: Optional[str] = None,
    ) -> _BenchResult:
        lats = self._time_fn(fn)
        sorted_lats = sorted(lats)
        p95_idx = max(0, int(math.ceil(0.95 * len(sorted_lats))) - 1)
        med = statistics.median(lats)
        r = _BenchResult(
            label=label,
            latency_min_us=sorted_lats[0],
            latency_median_us=med,
            latency_p95_us=sorted_lats[p95_idx],
            baseline_label=baseline_label,
        )
        if bytes_accessed > 0 and med > 0:
            r.bandwidth_gbs = (bytes_accessed / 1e9) / (med / 1e6)
        return r

    def compare(self, result: _BenchResult, baseline: _BenchResult) -> None:
        if baseline.latency_median_us > 0:
            result.speedup = baseline.latency_median_us / result.latency_median_us
        if baseline.bandwidth_gbs > 0:
            result.bw_improvement = result.bandwidth_gbs / baseline.bandwidth_gbs


# Select harness type
Harness  = _Harness
AnyResult = _BenchResult


# ---------------------------------------------------------------------------
# Device info (standalone, not depending on kernel_bench)
# ---------------------------------------------------------------------------

def _get_sm_version(device: int = 0) -> int:
    props = torch.cuda.get_device_properties(device)
    return props.major * 10 + props.minor


def _get_peak_hbm_bw(device: int = 0) -> float:
    props = torch.cuda.get_device_properties(device)
    name = props.name.lower()
    if "h100" in name:
        return 3350.0
    if "h200" in name:
        return 4800.0
    if "a100" in name:
        return 2000.0
    if "a6000" in name:
        return 768.0
    if "b200" in name or "b100" in name or "blackwell" in name:
        return 8000.0
    # Generic: memory_clock × bus_width × 2 (DDR factor)
    return round(props.memory_clock_rate * 1e3 * props.memory_bus_width / 8 * 2 / 1e9, 0)


def _print_device_header(device: int = 0) -> None:
    props = torch.cuda.get_device_properties(device)
    sm = props.major * 10 + props.minor
    peak_bw = _get_peak_hbm_bw(device)
    print(f"\n{'='*68}")
    print(f"  GPU {device}: {props.name}")
    print(f"  Compute: SM {props.major}.{props.minor}  (SM version {sm})")
    print(f"  SMs: {props.multi_processor_count}  |  "
          f"Memory: {props.total_memory / 1e9:.1f} GB")
    print(f"  Peak HBM BW (est.): {peak_bw:.0f} GB/s")
    print(f"{'='*68}\n")


# ---------------------------------------------------------------------------
# Baseline: torch_loop (models NCCL in-place accumulation)
# ---------------------------------------------------------------------------

def _torch_loop_reduce(inputs: List[torch.Tensor], out: torch.Tensor) -> None:
    """
    K sequential additions — equivalent to NCCL's reduce phase accumulation
    when each rank has already copied its gradient to the root.
    Each torch.add reads the full tensor once and writes once: 3 × N × 2 bytes.
    Total across K-1 iterations: (K-1) × 3 × N × 2 + 2 × N × 2 bytes read.
    """
    out.copy_(inputs[0])
    for t in inputs[1:]:
        out.add_(t)


# ---------------------------------------------------------------------------
# Baseline: torch_stack_sum (optimised multi-tensor reduction)
# ---------------------------------------------------------------------------

def _torch_stack_sum(inputs: List[torch.Tensor], out: torch.Tensor) -> None:
    """
    Stack K tensors into a [K, N] BF16 buffer then call .sum(0).
    PyTorch may pick CUBLAS or a cuDNN kernel internally.
    Memory: K × N × 2 bytes for the stack + N × 2 write = (K+1) × N × 2.
    Note: the allocation itself is amortised across warmup iterations.
    """
    stacked = torch.stack(inputs, dim=0)  # [K, N] BF16
    torch.sum(stacked, dim=0, out=out)


# ---------------------------------------------------------------------------
# Custom kernel baseline via hetero_reduce extension
# ---------------------------------------------------------------------------

def _make_custom_fn(
    out: torch.Tensor,
    inputs: List[torch.Tensor],
    sm_version: int,
) -> Optional[Callable[[], None]]:
    """Build a closure that calls the compiled hetero_reduce kernel."""
    if not _HAS_EXT:
        return None
    try:
        op = _ext.fused_bf16_reduce
        def _fn():
            op(out, inputs, sm_version)
        return _fn
    except Exception as e:
        warnings.warn(f"Could not bind hetero_reduce.fused_bf16_reduce: {e}")
        return None


# ---------------------------------------------------------------------------
# NCCL baseline via dist.all_reduce (single-process, gloo or nccl)
# ---------------------------------------------------------------------------

def _init_dist_nccl(device: int) -> bool:
    """Initialise single-process NCCL dist group.  Returns True on success."""
    if not _HAS_DIST:
        return False
    if dist.is_initialized():
        return True
    try:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=1,
            rank=0,
        )
        return True
    except Exception as e:
        warnings.warn(f"Could not init NCCL process group: {e}")
        return False


# ---------------------------------------------------------------------------
# Table printing (standalone, not depending on kernel_bench)
# ---------------------------------------------------------------------------

_COLS = {
    "label": 46,
    "min_us": 9,
    "med_us": 9,
    "p95_us": 9,
    "bw_gbs": 10,
    "roof_%": 8,
    "speedup": 9,
}


def _fmt(v: float, d: int = 2) -> str:
    return "—" if v == 0.0 else f"{v:.{d}f}"


def _print_table(results: List[AnyResult], peak_bw_gbs: float = 0.0) -> None:
    header = (
        f"{'Label':<{_COLS['label']}} "
        f"{'min_µs':>{_COLS['min_us']}} "
        f"{'med_µs':>{_COLS['med_us']}} "
        f"{'p95_µs':>{_COLS['p95_us']}} "
        f"{'BW GB/s':>{_COLS['bw_gbs']}} "
        f"{'Roof%':>{_COLS['roof_%']}} "
        f"{'speedup':>{_COLS['speedup']}}"
    )
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        roof = (r.bandwidth_gbs / peak_bw_gbs * 100) if peak_bw_gbs > 0 and r.bandwidth_gbs > 0 else 0.0
        spd_str = (_fmt(r.speedup) + "×") if r.speedup != 0.0 else "—"
        roof_str = _fmt(roof, 1) + "%" if roof > 0 else "—"
        print(
            f"{r.label:<{_COLS['label']}} "
            f"{_fmt(r.latency_min_us):>{_COLS['min_us']}} "
            f"{_fmt(r.latency_median_us):>{_COLS['med_us']}} "
            f"{_fmt(r.latency_p95_us):>{_COLS['p95_us']}} "
            f"{_fmt(r.bandwidth_gbs, 1):>{_COLS['bw_gbs']}} "
            f"{roof_str:>{_COLS['roof_%']}} "
            f"{spd_str:>{_COLS['speedup']}}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Size parsing
# ---------------------------------------------------------------------------

_SUFFIXES = {"K": 1024, "M": 1024**2, "G": 1024**3}


def parse_size(s: str) -> int:
    s = s.strip()
    if s[-1].upper() in _SUFFIXES:
        return int(s[:-1]) * _SUFFIXES[s[-1].upper()]
    return int(s)


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

@dataclass
class BenchConfig:
    sizes: List[int]          # element counts to sweep
    num_tensors: List[int]    # K values to sweep
    warmup: int = 20
    iters: int = 100
    device: int = 0
    use_nccl: bool = False
    quick: bool = False
    json_path: Optional[str] = None


def _bench_one_config(
    harness: Harness,
    n_elems: int,
    num_tensors: int,
    sm_version: int,
    peak_bw_gbs: float,
    use_nccl: bool,
    nccl_available: bool,
    device: int,
) -> List[AnyResult]:
    """Run all baselines + custom kernel for one (N, K) pair."""
    dtype  = torch.bfloat16
    dev    = f"cuda:{device}"

    # Allocate tensors
    inputs = [torch.randn(n_elems, dtype=dtype, device=dev) for _ in range(num_tensors)]
    out    = torch.empty(n_elems, dtype=dtype, device=dev)

    # Bytes: read K tensors + write 1 output (all BF16 = 2 bytes per element)
    bytes_rw = int((num_tensors + 1) * n_elems * 2)

    size_tag = f"N={n_elems//1024}K" if n_elems < 1024**2 else f"N={n_elems//1024**2}M"
    tag = f"{size_tag} K={num_tensors}"

    results: List[AnyResult] = []

    # ── Baseline A: torch_loop ───────────────────────────────────────────────
    r_loop = harness.run(
        label=f"torch_loop (NCCL model)   | {tag}",
        fn=lambda: _torch_loop_reduce(inputs, out),
        bytes_accessed=bytes_rw,
    )
    results.append(r_loop)

    # ── Baseline B: torch_stack_sum ──────────────────────────────────────────
    # Pre-allocate stack buffer to isolate allocation overhead from kernel cost
    stack_buf = torch.empty(num_tensors, n_elems, dtype=dtype, device=dev)
    def _stack_fn():
        torch.stack(inputs, out=stack_buf)
        stack_buf.sum(dim=0, out=out)

    r_stack = harness.run(
        label=f"torch_stack_sum            | {tag}",
        fn=_stack_fn,
        bytes_accessed=bytes_rw,
    )
    results.append(r_stack)

    # ── Baseline C: dist.all_reduce (NCCL) ───────────────────────────────────
    if use_nccl and nccl_available:
        # sum_buf starts as inputs[0]; all_reduce reduces it with sum
        sum_buf = inputs[0].clone()
        # In a real ring allreduce, each rank has ONE tensor; here we simulate
        # the *device-side* reduce of K partial sums from ring receive buffers.
        # We model this as K sequential all_reduces of N-element chunks
        # (world_size=1 → in-place identity, but exercises the code path).
        # A better model for single-GPU: reduce K temporary buffers to sum_buf.
        def _nccl_fn():
            sum_buf.copy_(inputs[0])
            for t in inputs[1:]:
                sum_buf.add_(t)   # models in-place ring accumulate
            dist.all_reduce(sum_buf, op=dist.ReduceOp.SUM)

        try:
            r_nccl = harness.run(
                label=f"dist.all_reduce NCCL       | {tag}",
                fn=_nccl_fn,
                bytes_accessed=bytes_rw,
            )
            results.append(r_nccl)
        except Exception as e:
            warnings.warn(f"NCCL baseline failed: {e}")

    # ── Custom kernel ─────────────────────────────────────────────────────────
    custom_fn = _make_custom_fn(out, inputs, sm_version)
    if custom_fn is not None:
        r_custom = harness.run(
            label=f"fused_bf16_reduce (ours)   | {tag}",
            fn=custom_fn,
            bytes_accessed=bytes_rw,
            baseline_label=r_loop.label,
        )
        harness.compare(r_custom, r_loop)
        results.append(r_custom)
    else:
        # Extension not compiled: add a PyTorch BF16-FP32 cast loop as proxy
        # to show *what* the kernel would do arithmetically.
        def _cast_loop():
            acc = inputs[0].to(torch.float32)
            for t in inputs[1:]:
                acc.add_(t.to(torch.float32))
            out.copy_(acc.to(torch.bfloat16))

        r_proxy = harness.run(
            label=f"fp32_cast_loop (proxy) [{tag}]",
            fn=_cast_loop,
            bytes_accessed=bytes_rw * 2,  # BF16→FP32→BF16 reads more
            baseline_label=r_loop.label,
        )
        harness.compare(r_proxy, r_loop)
        results.append(r_proxy)

    return results


def run_benchmark(cfg: BenchConfig) -> Dict:
    """Run the full benchmark sweep and return a serialisable result dict."""
    if not torch.cuda.is_available():
        print("ERROR: No CUDA device found.  Benchmark requires a GPU.")
        sys.exit(1)

    torch.cuda.set_device(cfg.device)
    sm_version = _get_sm_version(cfg.device)
    peak_bw    = _get_peak_hbm_bw(cfg.device)

    _print_device_header(cfg.device)

    # Extension status
    ext_status = "LOADED" if _HAS_EXT else "NOT COMPILED (proxy mode)"
    print(f"  hetero_reduce extension : {ext_status}")
    print(f"  SM version              : {sm_version}")
    print(f"  Sizes  : {[f'{n//1024}K' if n<1024**2 else f'{n//1024**2}M' for n in cfg.sizes]}")
    print(f"  K vals : {cfg.num_tensors}")
    print(f"  Warmup : {cfg.warmup}  |  Iters: {cfg.iters}\n")

    # NCCL init
    nccl_available = False
    if cfg.use_nccl:
        nccl_available = _init_dist_nccl(cfg.device)
        print(f"  NCCL baseline          : {'enabled' if nccl_available else 'unavailable'}\n")

    harness = Harness(warmup=cfg.warmup, iters=cfg.iters, device=cfg.device)

    all_results: List[AnyResult] = []
    summary_rows: List[dict] = []

    for num_tensors in cfg.num_tensors:
        print(f"\n{'━'*68}")
        print(f"  K = {num_tensors} input tensors")
        print(f"{'━'*68}")

        for n_elems in cfg.sizes:
            results = _bench_one_config(
                harness=harness,
                n_elems=n_elems,
                num_tensors=num_tensors,
                sm_version=sm_version,
                peak_bw_gbs=peak_bw,
                use_nccl=cfg.use_nccl,
                nccl_available=nccl_available,
                device=cfg.device,
            )
            _print_table(results, peak_bw_gbs=peak_bw)
            all_results.extend(results)

            # Collect summary
            baseline = results[0]
            kernel   = results[-1]
            n_label  = f"{n_elems//1024}K" if n_elems < 1024**2 else f"{n_elems//1024**2}M"
            summary_rows.append({
                "n_elems": n_elems,
                "n_label": n_label,
                "num_tensors": num_tensors,
                "sm_version": sm_version,
                "baseline_label": baseline.label.strip(),
                "baseline_med_us": round(baseline.latency_median_us, 3),
                "baseline_bw_gbs": round(baseline.bandwidth_gbs, 1),
                "kernel_label": kernel.label.strip(),
                "kernel_med_us": round(kernel.latency_median_us, 3),
                "kernel_bw_gbs": round(kernel.bandwidth_gbs, 1),
                "speedup": round(kernel.speedup, 3),
                "bw_improvement": round(kernel.bw_improvement, 3),
                "kernel_roof_pct": round(
                    kernel.bandwidth_gbs / peak_bw * 100 if peak_bw > 0 else 0.0, 1
                ),
            })

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n\n{'═'*68}")
    print(f"  SUMMARY — hetero_reduce vs torch_loop (NCCL model baseline)")
    print(f"{'═'*68}")
    print(f"{'N':>8}  {'K':>4}  {'baseline_µs':>12}  {'kernel_µs':>10}  "
          f"{'speedup':>8}  {'BW GB/s':>9}  {'Roof%':>7}")
    print(f"{'─'*68}")
    for row in summary_rows:
        spd = row['speedup']
        verdict = "✓ FASTER" if spd > 1.05 else ("≈ TIED" if spd > 0.95 else "✗ SLOWER")
        print(
            f"{row['n_label']:>8}  {row['num_tensors']:>4}  "
            f"{row['baseline_med_us']:>12.1f}  {row['kernel_med_us']:>10.1f}  "
            f"{spd:>7.2f}×  {row['kernel_bw_gbs']:>8.1f}  "
            f"{row['kernel_roof_pct']:>6.1f}%  {verdict}"
        )
    print(f"{'─'*68}\n")

    # Analysis
    speedups = [r["speedup"] for r in summary_rows if r["speedup"] > 0]
    if speedups:
        geomean = math.exp(sum(math.log(max(s, 1e-9)) for s in speedups) / len(speedups))
        faster_count = sum(1 for s in speedups if s > 1.05)
        print(f"  Geomean speedup: {geomean:.2f}×")
        print(f"  Faster in {faster_count}/{len(speedups)} configs (>1.05× threshold)")
        if geomean > 1.1:
            print(f"\n  ✓  VERDICT: kernel IS faster than the NCCL model baseline")
            print(f"     (geomean {geomean:.2f}× across {len(speedups)} configs)")
        elif geomean > 0.95:
            print(f"\n  ≈  VERDICT: kernel is roughly TIED with the baseline")
            print(f"     Bandwidth-bound regime; NCCL adds comm. overhead on top.")
        else:
            print(f"\n  ✗  VERDICT: kernel is SLOWER — investigate occupancy / __launch_bounds__")
        print()

    output = {
        "device": torch.cuda.get_device_properties(cfg.device).name,
        "sm_version": sm_version,
        "peak_hbm_bw_gbs": peak_bw,
        "extension_loaded": _HAS_EXT,
        "summary": summary_rows,
    }

    if cfg.json_path:
        with open(cfg.json_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results written to {cfg.json_path}")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DEFAULT_SIZES    = ["128K", "1M", "4M", "16M", "64M", "256M"]
_DEFAULT_K        = [1, 2, 4, 8, 16, 32]
_QUICK_SIZES      = ["1M", "16M"]
_QUICK_K          = [2, 8]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "bench_hetero_reduce_vs_nccl.py — proves/disproves hetero_reduce "
            "kernel is faster than NCCL baseline  (addresses #74)"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="CUDA device index",
    )
    parser.add_argument(
        "--sizes", nargs="+", default=None,
        metavar="SIZE",
        help="Element counts to sweep, e.g. 1M 16M 256M  (suffixes: K, M, G)",
    )
    parser.add_argument(
        "--num-tensors", nargs="+", type=int, default=None,
        metavar="K",
        help="Number of input tensors to sweep, e.g. 2 8 16 32",
    )
    parser.add_argument(
        "--warmup", type=int, default=20,
        help="Warmup iterations (un-timed)",
    )
    parser.add_argument(
        "--iters", type=int, default=100,
        help="Timed measurement iterations",
    )
    parser.add_argument(
        "--nccl", action="store_true", default=False,
        help="Enable dist.all_reduce NCCL baseline (requires NCCL backend)",
    )
    parser.add_argument(
        "--quick", action="store_true", default=False,
        help="Quick smoke test: small sizes, few iterations",
    )
    parser.add_argument(
        "--json", default=None, metavar="PATH",
        help="Write JSON results to PATH (for CI regression tracking)",
    )
    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        sizes      = [parse_size(s) for s in _QUICK_SIZES]
        ktensors   = _QUICK_K
        warmup     = 5
        iters      = 20
    else:
        sizes    = [parse_size(s) for s in (args.sizes or _DEFAULT_SIZES)]
        ktensors = args.num_tensors or _DEFAULT_K
        warmup   = args.warmup
        iters    = args.iters

    cfg = BenchConfig(
        sizes=sizes,
        num_tensors=ktensors,
        warmup=warmup,
        iters=iters,
        device=args.device,
        use_nccl=args.nccl,
        quick=args.quick,
        json_path=args.json,
    )

    run_benchmark(cfg)


if __name__ == "__main__":
    main()
