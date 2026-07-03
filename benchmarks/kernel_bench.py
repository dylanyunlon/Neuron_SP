# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team / Neuron_SP
"""
benchmarks/kernel_bench.py — Universal CUDA kernel benchmark harness

Provides a reusable BenchmarkHarness class used by all per-kernel benchmark
scripts in this directory.  It wraps torch.cuda.Event for sub-microsecond GPU
timing, executes configurable warmup + measurement iterations, and computes
bandwidth (GB/s), latency (µs), and FLOPs from caller-supplied callbacks.

Usage (standalone):
    python benchmarks/kernel_bench.py            # smoke-test on GPU 0
    python benchmarks/kernel_bench.py --device 1 # specific GPU

Typical integration in a per-kernel benchmark::

    from kernel_bench import BenchmarkHarness, BenchResult, print_results_table

    harness = BenchmarkHarness(warmup=10, iters=50, device=0)

    def fn():
        my_kernel(...)

    result = harness.run(
        label="my_kernel",
        fn=fn,
        bytes_accessed=n_bytes,   # for bandwidth calculation
        flops=n_flops,            # optional — set to 0 if unknown
    )
    print_results_table([result])
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import torch

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    """Holds all measurements for a single kernel configuration."""

    label: str                        # human-readable name, e.g. "fused_swiglu_ln / batch=64 / hidden=8192"
    baseline_label: Optional[str]     # label of the baseline to compare against, or None
    latency_us_all: List[float]       # per-iteration latency in microseconds (post-warmup)

    # Derived metrics — computed by BenchmarkHarness.run()
    latency_min_us: float = 0.0
    latency_median_us: float = 0.0
    latency_mean_us: float = 0.0
    latency_p95_us: float = 0.0

    bandwidth_gbs: float = 0.0        # effective memory bandwidth  (GB/s, 10^9 B/s)
    tflops: float = 0.0               # arithmetic throughput (TFLOPS)

    # Fields populated by compare_to_baseline()
    speedup: float = 0.0              # latency_baseline / latency_this  (>1 means faster)
    bw_improvement: float = 0.0       # bw_this / bw_baseline


@dataclass
class BenchConfig:
    """Configures a BenchmarkHarness run."""
    warmup: int = 10
    iters: int = 50
    device: int = 0
    sync_every_iter: bool = True      # cudaDeviceSynchronize after each iter (accurate but slow)
    use_median: bool = True           # if False, use mean latency for derived metrics


# ---------------------------------------------------------------------------
# Core harness
# ---------------------------------------------------------------------------


class BenchmarkHarness:
    """
    Wraps torch.cuda.Event timing + statistics for GPU kernel benchmarking.

    Parameters
    ----------
    warmup : int
        Number of un-timed warmup iterations (clears cold-start artefacts and
        populates HW prefetchers / instruction caches).
    iters : int
        Number of timed measurement iterations.
    device : int
        CUDA device index to benchmark on.
    sync_every_iter : bool
        If True, records a stop event and calls cudaEventSynchronize() after
        every iteration — gives accurate per-iteration times at the cost of
        slightly higher host overhead.  For kernels with very short duration
        (< 5 µs) consider setting False and using the aggregate timing path.
    """

    def __init__(
        self,
        warmup: int = 10,
        iters: int = 50,
        device: int = 0,
        sync_every_iter: bool = True,
    ) -> None:
        self.warmup = warmup
        self.iters = iters
        self.device = device
        self.sync_every_iter = sync_every_iter
        self._stream = torch.cuda.Stream(device=device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        label: str,
        fn: Callable[[], None],
        bytes_accessed: int = 0,
        flops: int = 0,
        baseline_label: Optional[str] = None,
    ) -> BenchResult:
        """
        Time *fn* over warmup + measurement iterations and return a BenchResult.

        Parameters
        ----------
        label : str
            Human-readable name for the kernel variant being benchmarked.
        fn : Callable
            Zero-argument callable that launches the kernel.  The callable
            should not call any synchronisation itself — the harness manages
            stream synchronisation.
        bytes_accessed : int
            Total bytes read + written by the kernel (used for bandwidth).
            Pass 0 to skip bandwidth calculation.
        flops : int
            Total floating-point operations (FLOPs, not FLOP/s) executed by
            the kernel.  Pass 0 to skip TFLOPS calculation.
        baseline_label : str, optional
            Label of another BenchResult to compare against (speedup is
            computed later via compare_to_baseline).

        Returns
        -------
        BenchResult with all latency statistics populated.
        """
        torch.cuda.set_device(self.device)

        # ----- warmup --------------------------------------------------------
        with torch.cuda.stream(self._stream):
            for _ in range(self.warmup):
                fn()
        self._stream.synchronize()

        # ----- timed measurement ---------------------------------------------
        latencies_us: List[float] = []

        if self.sync_every_iter:
            start_ev = torch.cuda.Event(enable_timing=True)
            stop_ev = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(self._stream):
                for _ in range(self.iters):
                    start_ev.record(self._stream)
                    fn()
                    stop_ev.record(self._stream)
                    stop_ev.synchronize()
                    latencies_us.append(start_ev.elapsed_time(stop_ev) * 1e3)  # ms → µs
        else:
            # Aggregate timing: one pair of events around the full batch.
            # Latency per iteration = total / iters.
            start_ev = torch.cuda.Event(enable_timing=True)
            stop_ev = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(self._stream):
                start_ev.record(self._stream)
                for _ in range(self.iters):
                    fn()
                stop_ev.record(self._stream)
            stop_ev.synchronize()
            avg_us = (start_ev.elapsed_time(stop_ev) * 1e3) / self.iters
            latencies_us = [avg_us] * self.iters  # fill list so statistics still work

        # ----- statistics ----------------------------------------------------
        sorted_us = sorted(latencies_us)
        p95_idx = max(0, int(math.ceil(0.95 * len(sorted_us))) - 1)

        result = BenchResult(
            label=label,
            baseline_label=baseline_label,
            latency_us_all=latencies_us,
            latency_min_us=sorted_us[0],
            latency_median_us=statistics.median(latencies_us),
            latency_mean_us=statistics.mean(latencies_us),
            latency_p95_us=sorted_us[p95_idx],
        )

        # representative latency for derived metrics
        rep_us = result.latency_median_us

        if bytes_accessed > 0 and rep_us > 0:
            # bandwidth in GB/s (10^9 bytes / second)
            result.bandwidth_gbs = (bytes_accessed / 1e9) / (rep_us / 1e6)

        if flops > 0 and rep_us > 0:
            result.tflops = (flops / 1e12) / (rep_us / 1e6)

        return result

    def compare_to_baseline(
        self,
        result: BenchResult,
        baseline: BenchResult,
    ) -> None:
        """
        Populate *result.speedup* and *result.bw_improvement* relative to
        *baseline*.  Modifies *result* in-place.
        """
        if baseline.latency_median_us > 0:
            result.speedup = baseline.latency_median_us / result.latency_median_us
        if baseline.bandwidth_gbs > 0:
            result.bw_improvement = result.bandwidth_gbs / baseline.bandwidth_gbs

    def run_comparison(
        self,
        baseline_label: str,
        baseline_fn: Callable[[], None],
        custom_label: str,
        custom_fn: Callable[[], None],
        bytes_accessed: int = 0,
        flops: int = 0,
    ) -> tuple[BenchResult, BenchResult]:
        """
        Convenience wrapper: benchmark baseline + custom kernel and compute
        relative speedup in one call.

        Returns
        -------
        (baseline_result, custom_result)
        """
        base = self.run(
            label=baseline_label,
            fn=baseline_fn,
            bytes_accessed=bytes_accessed,
            flops=flops,
        )
        custom = self.run(
            label=custom_label,
            fn=custom_fn,
            bytes_accessed=bytes_accessed,
            flops=flops,
            baseline_label=baseline_label,
        )
        self.compare_to_baseline(custom, base)
        return base, custom


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

_COL_WIDTHS = {
    "label": 44,
    "min_us": 9,
    "med_us": 9,
    "p95_us": 9,
    "bw_gbs": 10,
    "tflops": 8,
    "speedup": 8,
}


def _fmt(value: float, digits: int = 2) -> str:
    if value == 0.0:
        return "—"
    return f"{value:.{digits}f}"


def print_results_table(results: Sequence[BenchResult]) -> None:
    """Print a human-readable markdown-style table to stdout."""
    header = (
        f"{'Label':<{_COL_WIDTHS['label']}} "
        f"{'min_µs':>{_COL_WIDTHS['min_us']}} "
        f"{'med_µs':>{_COL_WIDTHS['med_us']}} "
        f"{'p95_µs':>{_COL_WIDTHS['p95_us']}} "
        f"{'BW GB/s':>{_COL_WIDTHS['bw_gbs']}} "
        f"{'TFLOPS':>{_COL_WIDTHS['tflops']}} "
        f"{'speedup':>{_COL_WIDTHS['speedup']}}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        speedup_str = _fmt(r.speedup) + "×" if r.speedup != 0.0 else "—"
        row = (
            f"{r.label:<{_COL_WIDTHS['label']}} "
            f"{_fmt(r.latency_min_us):>{_COL_WIDTHS['min_us']}} "
            f"{_fmt(r.latency_median_us):>{_COL_WIDTHS['med_us']}} "
            f"{_fmt(r.latency_p95_us):>{_COL_WIDTHS['p95_us']}} "
            f"{_fmt(r.bandwidth_gbs):>{_COL_WIDTHS['bw_gbs']}} "
            f"{_fmt(r.tflops, 3):>{_COL_WIDTHS['tflops']}} "
            f"{speedup_str:>{_COL_WIDTHS['speedup']}}"
        )
        print(row)
    print(sep)


def print_json(results: Sequence[BenchResult]) -> str:
    """Serialise results to a JSON string."""
    import json

    def _to_dict(r: BenchResult) -> dict:
        return {
            "label": r.label,
            "baseline_label": r.baseline_label,
            "latency_min_us": r.latency_min_us,
            "latency_median_us": r.latency_median_us,
            "latency_mean_us": r.latency_mean_us,
            "latency_p95_us": r.latency_p95_us,
            "bandwidth_gbs": r.bandwidth_gbs,
            "tflops": r.tflops,
            "speedup": r.speedup,
            "bw_improvement": r.bw_improvement,
        }

    return json.dumps([_to_dict(r) for r in results], indent=2)


# ---------------------------------------------------------------------------
# GPU info helpers
# ---------------------------------------------------------------------------


def get_device_info(device: int = 0) -> dict:
    """Return a dict with useful GPU properties."""
    props = torch.cuda.get_device_properties(device)
    return {
        "name": props.name,
        "device": device,
        "sm_count": props.multi_processor_count,
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_version": props.major * 10 + props.minor,
        "total_memory_gb": props.total_memory / 1e9,
        # theoretical peak BF16 TFLOPS (approximate, no Tensor Core factor)
        "peak_bf16_tflops_approx": _peak_bf16_tflops(props),
        # theoretical peak HBM bandwidth from known GPU database
        "peak_hbm_bandwidth_gbs": _peak_hbm_bandwidth(props),
    }


def _peak_bf16_tflops(props) -> float:
    """Rough BF16 TFLOPS from SM count × clock × ops-per-SM (Tensor Cores)."""
    sm_version = props.major * 10 + props.minor
    # ops/SM/cycle for BF16 Tensor Cores (INT matrix units)
    ops_per_sm = {90: 512, 86: 256, 80: 512, 89: 256, 120: 512}.get(sm_version, 128)
    peak = props.multi_processor_count * props.clock_rate * 1e3 * ops_per_sm * 2 / 1e12
    return round(peak, 1)


def _peak_hbm_bandwidth(props) -> float:
    """Approximate HBM peak bandwidth from known GPU types."""
    name = props.name.lower()
    if "h100" in name:
        return 3350.0   # H100 SXM5
    if "h200" in name:
        return 4800.0
    if "a100" in name:
        return 2000.0
    if "a6000" in name:
        return 768.0
    if "b200" in name or "blackwell" in name:
        return 8000.0
    # fallback: memory_clock × bus_width × 2 (DDR)
    return round(props.memory_clock_rate * 1e3 * props.memory_bus_width / 8 * 2 / 1e9, 0)


def print_device_header(device: int = 0) -> None:
    info = get_device_info(device)
    print(f"\n{'='*60}")
    print(f"  GPU {info['device']}: {info['name']}")
    print(f"  Compute capability: SM {info['compute_capability']}  |  "
          f"SMs: {info['sm_count']}  |  "
          f"Memory: {info['total_memory_gb']:.1f} GB")
    print(f"  Peak BF16 TFLOPS (est.): {info['peak_bf16_tflops_approx']}  |  "
          f"Peak HBM BW: {info['peak_hbm_bandwidth_gbs']} GB/s")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Smoke-test / standalone entry point
# ---------------------------------------------------------------------------

def _smoke_test(device: int) -> None:
    """
    Validates the harness on a trivial BF16 elementwise kernel (torch.add)
    and prints results to confirm timing infrastructure works.
    """
    print_device_header(device)

    N = 16 * 1024 * 1024  # 16M elements BF16 = 32 MB
    a = torch.randn(N, dtype=torch.bfloat16, device=f"cuda:{device}")
    b = torch.randn(N, dtype=torch.bfloat16, device=f"cuda:{device}")
    out = torch.empty_like(a)
    bytes_rw = N * 2 * 3  # read a, read b, write out  (2 bytes each BF16)

    harness = BenchmarkHarness(warmup=10, iters=50, device=device)

    # Baseline: torch.add (single kernel, no copy)
    base, custom = harness.run_comparison(
        baseline_label="torch.add (baseline)",
        baseline_fn=lambda: torch.add(a, b, out=out),
        custom_label="torch.add (custom — same fn, verify 1.00×)",
        custom_fn=lambda: torch.add(a, b, out=out),
        bytes_accessed=bytes_rw,
        flops=N,  # one FP add per element
    )

    print("Smoke-test: elementwise BF16 add (16M elements, 32 MB)")
    print_results_table([base, custom])

    # Additional: vector norm (compute-bound for comparison)
    norm_base = harness.run(
        label="torch.linalg.norm (compute ref)",
        fn=lambda: torch.linalg.norm(a),
        bytes_accessed=N * 2,
        flops=N * 2,  # mul + accumulate
    )
    print_results_table([norm_base])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Neuron_SP kernel benchmark harness smoke-test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA device found.  Run on a GPU node.")
        raise SystemExit(1)

    _smoke_test(args.device)


if __name__ == "__main__":
    main()
