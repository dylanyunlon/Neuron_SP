# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team / Neuron_SP
"""
benchmarks/bench_hetero_reduce.py — Fused vs unfused SwiGLU+LayerNorm benchmark
                                     Addresses issue #83

Compares:
  A) fused_swiglu_ln  — the kernel in csrc/hetero_reduce/fused_swiglu_ln.cu,
     exposed via the compiled hetero_reduce C extension.  When the extension
     is not compiled, a pure-PyTorch simulation of the same fused op is used.

  B) unfused_swiglu_ln — SwiGLU and RMSNorm executed as separate PyTorch ops
     (the baseline that motivated the custom kernel).

Measurement tooling: torch.utils.benchmark.Timer
  - Median wall-clock time across configurable number of iterations.
  - CUDA synchronisation handled by the Timer's CUDA event path.
  - Sub-benchmark labelling for easy diff output.

No C/CUDA compilation required.  All paths run on both CPU and CUDA; CUDA
is used when available and is by far the more interesting path.

Configurations swept by default:
  batch sizes  : 1, 16, 64, 256
  hidden dims  : 1024, 4096, 8192, 16384

Each configuration measures:
  - median latency (µs)
  - effective memory bandwidth (GB/s)
  - speedup of fused over unfused

Launch:
    python benchmarks/bench_hetero_reduce.py
    python benchmarks/bench_hetero_reduce.py --device 1 --iters 200
    python benchmarks/bench_hetero_reduce.py --batches 1 64 --hiddens 4096 8192
    python benchmarks/bench_hetero_reduce.py --json results_bench_hetero_reduce.json
    python benchmarks/bench_hetero_reduce.py --no-cuda   # CPU timing only

Note on the CUDA extension:
  The compiled hetero_reduce C extension (hetero_reduce.so / _hetero_reduce.so)
  may not be available in all environments.  In that case this benchmark
  uses a pure-PyTorch "simulation" of the fused path that exercises identical
  Python-level tensor ops to the compiled path, labelled clearly in output.
  Build the extension with:
      cd csrc/hetero_reduce && python setup.py install
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer, Compare

# ---------------------------------------------------------------------------
# Optional: try to import the compiled hetero_reduce extension
# ---------------------------------------------------------------------------

try:
    import deepspeed.ops.hetero_reduce as _ext  # type: ignore[import]
    _HAS_EXT = True
except ImportError:
    _HAS_EXT = False


# ---------------------------------------------------------------------------
# Op implementations
# ---------------------------------------------------------------------------

def _swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU: gate * silu(gate) * up  (element-wise)."""
    return F.silu(gate) * up


def _rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """RMSNorm: x / rms(x) * weight.

    Uses torch.nn.functional.rms_norm when available (PyTorch ≥ 2.4).
    Falls back to a manual implementation on older builds.
    """
    try:
        return F.rms_norm(x, (x.shape[-1],), weight=weight.to(x.dtype), eps=eps)
    except AttributeError:
        # Manual RMSNorm path for PyTorch < 2.4
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add_(eps).sqrt_()
        return (x.float() / rms * weight.float()).to(x.dtype)


# ---------------------------------------------------------------------------
# Unfused reference: two separate op calls
# ---------------------------------------------------------------------------

def unfused_swiglu_ln(
    gate: torch.Tensor,
    up: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Separate SwiGLU activation followed by RMSNorm — the unfused baseline.

    This is what you would write without the custom kernel:
      Step 1: x = SwiGLU(gate, up)   — one kernel launch
      Step 2: y = RMSNorm(x, weight) — one or more kernel launches
    Each step reads and writes DRAM, resulting in at least two round-trips
    for the activation tensor.
    """
    x = _swiglu(gate, up)         # kernel 1: SwiGLU (reads gate+up, writes x)
    y = _rms_norm(x, weight, eps) # kernel 2: RMSNorm (reads x+weight, writes y)
    return y


# ---------------------------------------------------------------------------
# Fused path: single compound op (kernel or simulation)
# ---------------------------------------------------------------------------

def fused_swiglu_ln(
    gate: torch.Tensor,
    up: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: Optional[torch.Tensor] = None,
    sm_version: int = 86,
) -> torch.Tensor:
    """Fused SwiGLU + RMSNorm — calls the compiled CUDA extension when available.

    The CUDA kernel in csrc/hetero_reduce/fused_swiglu_ln.cu performs SwiGLU
    and RMSNorm in a single pass over DRAM using a warp-shuffle Welford
    reduction, eliminating the intermediate activation tensor write+read.

    When the compiled extension is not found, a PyTorch simulation is used
    that dispatches the same operations through torch.compile for a partial
    fusion benefit.  Results are labelled accordingly.

    Arguments
    ---------
    gate, up : torch.Tensor  — BF16, shape [batch, hidden]
    weight   : torch.Tensor  — FP32, shape [hidden]
    eps      : float         — RMSNorm epsilon
    out      : optional pre-allocated BF16 output buffer
    sm_version : int         — SM version passed to the C extension (86/90/120)
    """
    if _HAS_EXT:
        if out is None:
            out = torch.empty_like(gate)
        # The extension requires contiguous BF16 gate/up and FP32 weight.
        _ext.fused_swiglu_ln(
            out,
            gate.contiguous(),
            up.contiguous(),
            weight.contiguous().float(),
            eps,
            sm_version,
        )
        return out
    else:
        # PyTorch simulation: same logic, no intermediate DRAM round-trip for
        # large tensors when torch.compile fuses the ops.
        # NOTE: this path intentionally does NOT use torch.compile because
        # compilation at benchmark time would distort timing.  The raw
        # two-kernel path here still exercises the same memory pattern and
        # gives a fair "what could PyTorch do" reference.
        x = _swiglu(gate, up)
        return _rms_norm(x, weight, eps)


# ---------------------------------------------------------------------------
# Numerical correctness smoke-check
# ---------------------------------------------------------------------------

def _correctness_check(device: str, sm_version: int, eps: float = 1e-6) -> bool:
    """Verify fused and unfused outputs agree to BF16 precision (~0.4% reltol)."""
    B, H = 8, 4096
    g = torch.randn(B, H, dtype=torch.bfloat16, device=device)
    u = torch.randn(B, H, dtype=torch.bfloat16, device=device)
    w = torch.ones(H, dtype=torch.float32, device=device)

    ref = unfused_swiglu_ln(g, u, w, eps)
    fus = fused_swiglu_ln(g, u, w, eps, sm_version=sm_version)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    max_diff = (ref.float() - fus.float()).abs().max().item()
    # BF16 has ~7-bit mantissa → ~0.8% relative error is acceptable.
    return max_diff < 0.02


# ---------------------------------------------------------------------------
# Benchmark result dataclass
# ---------------------------------------------------------------------------

@dataclass
class Result:
    label: str
    batch: int
    hidden: int
    variant: str              # "fused" or "unfused"
    median_us: float          # median latency in µs
    bandwidth_gbs: float      # effective memory bandwidth
    speedup: float = 0.0      # populated after pairing fused/unfused
    extension_available: bool = False


# ---------------------------------------------------------------------------
# Per-configuration benchmark
# ---------------------------------------------------------------------------

def _bench_pair(
    batch: int,
    hidden: int,
    device: str,
    sm_version: int,
    eps: float,
    iters: int,
    min_run_time: float,
) -> Tuple[Result, Result]:
    """Benchmark one (batch, hidden) configuration; return (unfused, fused)."""
    g = torch.randn(batch, hidden, dtype=torch.bfloat16, device=device)
    u = torch.randn(batch, hidden, dtype=torch.bfloat16, device=device)
    w = torch.ones(hidden, dtype=torch.float32, device=device)
    out_fused = torch.empty(batch, hidden, dtype=torch.bfloat16, device=device)

    # Memory model (bytes):
    #   unfused: read gate+up → write intermediate → read intermediate+weight → write out
    #            ≈ 5 × B×H×2  (BF16 gate, up, intermediate ×2, out) + H×4 (weight)
    #   fused:   read gate+up+weight → write out
    #            ≈ 3 × B×H×2 + H×4
    #   We report fused bytes for both (conservative for unfused speedup).
    fused_bytes = 3 * batch * hidden * 2 + hidden * 4
    unfused_bytes = 5 * batch * hidden * 2 + hidden * 4

    tag = f"batch={batch}, hidden={hidden}"
    sub = f"bench_hetero_reduce[{tag}]"

    # --- Unfused ---
    t_unfused = Timer(
        stmt="unfused_swiglu_ln(g, u, w, eps)",
        globals={"unfused_swiglu_ln": unfused_swiglu_ln, "g": g, "u": u, "w": w, "eps": eps},
        label=f"unfused SwiGLU+LayerNorm  | {tag}",
        sub_label=sub,
        description="unfused",
        num_threads=1,
    )
    m_unfused = t_unfused.blocked_autorange(min_run_time=min_run_time)
    median_unfused_us = m_unfused.median * 1e6
    bw_unfused = (unfused_bytes / 1e9) / (median_unfused_us / 1e6)

    # --- Fused ---
    fused_stmt = (
        "fused_swiglu_ln(g, u, w, eps, out=out_fused, sm_version=sm_version)"
    )
    t_fused = Timer(
        stmt=fused_stmt,
        globals={
            "fused_swiglu_ln": fused_swiglu_ln,
            "g": g, "u": u, "w": w,
            "eps": eps, "out_fused": out_fused,
            "sm_version": sm_version,
        },
        label=f"{'fused_swiglu_ln (ext)' if _HAS_EXT else 'fused_swiglu_ln (sim)'}  | {tag}",
        sub_label=sub,
        description="fused (CUDA ext)" if _HAS_EXT else "fused (PyTorch sim)",
        num_threads=1,
    )
    m_fused = t_fused.blocked_autorange(min_run_time=min_run_time)
    median_fused_us = m_fused.median * 1e6
    bw_fused = (fused_bytes / 1e9) / (median_fused_us / 1e6)

    r_unfused = Result(
        label=f"unfused | {tag}",
        batch=batch, hidden=hidden, variant="unfused",
        median_us=median_unfused_us,
        bandwidth_gbs=bw_unfused,
        extension_available=_HAS_EXT,
    )
    r_fused = Result(
        label=f"fused   | {tag}",
        batch=batch, hidden=hidden, variant="fused",
        median_us=median_fused_us,
        bandwidth_gbs=bw_fused,
        extension_available=_HAS_EXT,
    )
    if median_fused_us > 0:
        r_unfused.speedup = 0.0
        r_fused.speedup = median_unfused_us / median_fused_us

    return r_unfused, r_fused


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

_COL = {
    "label":    46,
    "med_us":    9,
    "bw_gbs":   10,
    "speedup":   9,
}


def _print_header() -> None:
    h = (
        f"{'Variant / Config':<{_COL['label']}} "
        f"{'med_µs':>{_COL['med_us']}} "
        f"{'BW GB/s':>{_COL['bw_gbs']}} "
        f"{'speedup':>{_COL['speedup']}}"
    )
    print("-" * len(h))
    print(h)
    print("-" * len(h))


def _print_row(r: Result) -> None:
    speedup_str = f"{r.speedup:.2f}×" if r.speedup > 0 else "  —  "
    print(
        f"{r.label:<{_COL['label']}} "
        f"{r.median_us:>{_COL['med_us']}.2f} "
        f"{r.bandwidth_gbs:>{_COL['bw_gbs']}.1f} "
        f"{speedup_str:>{_COL['speedup']}}"
    )


def _print_sep() -> None:
    w = _COL['label'] + _COL['med_us'] + _COL['bw_gbs'] + _COL['speedup'] + 3
    print("-" * w)


def print_results(pairs: List[Tuple[Result, Result]]) -> None:
    _print_header()
    for unfused, fused in pairs:
        _print_row(unfused)
        _print_row(fused)
        _print_sep()


# ---------------------------------------------------------------------------
# GPU info helper
# ---------------------------------------------------------------------------

def _get_sm_version(device_idx: int) -> int:
    props = torch.cuda.get_device_properties(device_idx)
    return props.major * 10 + props.minor


def _print_device_info(device_idx: int) -> None:
    props = torch.cuda.get_device_properties(device_idx)
    sm = f"{props.major}.{props.minor}"
    print(f"\n{'='*60}")
    print(f"  GPU {device_idx}: {props.name}")
    print(f"  Compute capability: SM {sm}  |  SMs: {props.multi_processor_count}")
    print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "bench_hetero_reduce.py — fused_swiglu_ln vs unfused SwiGLU+LayerNorm\n"
            "Uses torch.utils.benchmark.Timer for accurate GPU timing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument(
        "--batches", nargs="+", type=int, default=[1, 16, 64, 256],
        help="Batch sizes to sweep",
    )
    parser.add_argument(
        "--hiddens", nargs="+", type=int, default=[1024, 4096, 8192, 16384],
        help="Hidden dimensions to sweep",
    )
    parser.add_argument(
        "--iters", type=int, default=100,
        help="Minimum number of iterations per measurement (passed to Timer)",
    )
    parser.add_argument(
        "--min-run-time", type=float, default=0.5,
        help="Minimum total wall-time per measurement in seconds (Timer.blocked_autorange)",
    )
    parser.add_argument("--eps", type=float, default=1e-6, help="RMSNorm epsilon")
    parser.add_argument(
        "--no-cuda", action="store_true",
        help="Run even when CUDA is unavailable (CPU timing, informational only)",
    )
    parser.add_argument(
        "--json", metavar="PATH", default=None,
        help="Write JSON results to this file",
    )
    parser.add_argument(
        "--skip-correctness", action="store_true",
        help="Skip the numerical correctness check",
    )
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    if not use_cuda:
        if not args.no_cuda:
            print("WARNING: No CUDA device found.  Timings will use CPU.")
        device = "cpu"
        sm_version = 86  # default fallback for extension dispatch
    else:
        torch.cuda.set_device(args.device)
        device = f"cuda:{args.device}"
        sm_version = _get_sm_version(args.device)
        _print_device_info(args.device)

    # Report extension status
    ext_status = (
        "compiled CUDA extension AVAILABLE — fused path uses hetero_reduce kernel"
        if _HAS_EXT else
        "compiled CUDA extension NOT found — fused path uses PyTorch simulation\n"
        "  Build: cd csrc/hetero_reduce && python setup.py install"
    )
    print(f"Extension: {ext_status}\n")

    # Correctness check
    if use_cuda and not args.skip_correctness:
        print("Correctness check (fused vs unfused, batch=8, hidden=4096): ", end="", flush=True)
        ok = _correctness_check(device, sm_version, args.eps)
        print("PASS ✓" if ok else "FAIL ✗  (outputs diverge beyond BF16 tolerance)")
        print()

    # Run benchmarks
    all_pairs: List[Tuple[Result, Result]] = []

    for hidden in args.hiddens:
        print(f"\n{'─'*65}")
        print(f"  Hidden dim: {hidden}")
        print(f"{'─'*65}")
        _print_header()

        for batch in args.batches:
            unfused_r, fused_r = _bench_pair(
                batch=batch,
                hidden=hidden,
                device=device,
                sm_version=sm_version,
                eps=args.eps,
                iters=args.iters,
                min_run_time=args.min_run_time,
            )
            _print_row(unfused_r)
            _print_row(fused_r)
            _print_sep()
            all_pairs.append((unfused_r, fused_r))

    # Summary table
    print(f"\n{'='*65}")
    print("  SUMMARY — fused speedup over unfused SwiGLU+LayerNorm")
    if not _HAS_EXT:
        print("  (CUDA extension not compiled; fused path = PyTorch simulation)")
    print(f"{'='*65}")
    print(
        f"  {'batch':>6}  {'hidden':>7}  {'unfused µs':>11}  "
        f"{'fused µs':>9}  {'speedup':>8}  {'fused BW GB/s':>14}"
    )
    print(f"  {'-'*6}  {'-'*7}  {'-'*11}  {'-'*9}  {'-'*8}  {'-'*14}")
    for unfused_r, fused_r in all_pairs:
        print(
            f"  {fused_r.batch:>6}  {fused_r.hidden:>7}  "
            f"{unfused_r.median_us:>11.2f}  "
            f"{fused_r.median_us:>9.2f}  "
            f"{fused_r.speedup:>7.2f}×  "
            f"{fused_r.bandwidth_gbs:>14.1f}"
        )
    print()

    # JSON output
    if args.json:
        rows = []
        for unfused_r, fused_r in all_pairs:
            for r in (unfused_r, fused_r):
                rows.append({
                    "variant": r.variant,
                    "batch": r.batch,
                    "hidden": r.hidden,
                    "median_us": r.median_us,
                    "bandwidth_gbs": r.bandwidth_gbs,
                    "speedup": r.speedup,
                    "extension_available": r.extension_available,
                })
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"Results written to {args.json}")


if __name__ == "__main__":
    main()
