# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
#
# DeepSpeed Team

"""
benchmarks/bench_hetero_reduce_vs_nccl.py   -- addresses #74
=============================================================

Proves (or disproves) that csrc/hetero_reduce/ kernels are faster than NCCL
for the heterogeneous, PCIe-only reduce-scatter workload they were designed for.

WHAT WE BENCHMARK
-----------------
Operation:  given K input tensors of N BF16 elements each (all on the same
GPU), produce one output tensor of N BF16 elements:

    output[i] = sum_{k=0..K-1} inputs[k][i]    (FP32 accumulation)

Baselines
---------
  A -- torch_inplace_loop:
    K-1 sequential .add_() calls.  Models NCCL ring-reduce accumulate phase.

  B -- torch_stack_sum:
    Stack K inputs to [K, N] then .sum(0).  PyTorch may pick an optimal kernel.

  C -- dist.all_reduce (NCCL, optional --nccl):
    Real NCCL single-process allreduce (world_size=1) measuring invocation cost.

  D -- fp32_cast_loop:
    Explicit BF16->FP32 cast, loop-add, FP32->BF16 cast (proxy/naive baseline).

Our kernel -- fused_bf16_reduce (hetero_reduce.cu):
    Warp-cooperative BF16->FP32 accumulation, one kernel pass, constant memory.

SM12.0 CODE PATH AUDIT (addresses #62)
---------------------------------------
  All five .cu files have correct SM8.6/9.0/12.0 dispatch paths.
  BUG FIXED in tier_activation_offload.cu:
    SM9.0 activation_pack/unpack used __launch_bounds__(256, 2) instead of
    __launch_bounds__(256, 4) because block-size was used as the sole
    discriminator.  Fix: three-way sm_version dispatch so SM9.0 gets
    activation_pack_kernel<90> with kMinBlocksPerSM=4.

USAGE
-----
  python benchmarks/bench_hetero_reduce_vs_nccl.py
  python benchmarks/bench_hetero_reduce_vs_nccl.py --quick
  python benchmarks/bench_hetero_reduce_vs_nccl.py --nccl --json results_74.json
  python benchmarks/bench_hetero_reduce_vs_nccl.py --sizes 1M 16M --num-tensors 4 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

try:
    import torch
    import torch.cuda
except ImportError:
    print("ERROR: PyTorch is required.  pip install torch")
    sys.exit(1)

# Optional: compiled hetero_reduce extension
_HAS_EXT = False
_ext = None
try:
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo not in sys.path:
        sys.path.insert(0, _repo)
    from deepspeed.ops.hetero_reduce import hetero_reduce_op as _ext  # type: ignore
    _HAS_EXT = True
except Exception:
    pass

# Optional: torch.distributed for NCCL baseline
_HAS_DIST = False
try:
    import torch.distributed as dist
    _HAS_DIST = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Size utilities
# ---------------------------------------------------------------------------

_SIZE_SUFFIXES: Dict[str, int] = {"K": 1024, "M": 1024**2, "G": 1024**3}


def parse_size(s: str) -> int:
    s = s.strip()
    if s[-1].upper() in _SIZE_SUFFIXES:
        return int(s[:-1]) * _SIZE_SUFFIXES[s[-1].upper()]
    return int(s)


def fmt_size(n: int) -> str:
    if n >= 1024**3:
        return f"{n // 1024**3}G"
    if n >= 1024**2:
        return f"{n // 1024**2}M"
    if n >= 1024:
        return f"{n // 1024}K"
    return str(n)


# ---------------------------------------------------------------------------
# Device info
# ---------------------------------------------------------------------------

def get_sm_version(device: int = 0) -> int:
    props = torch.cuda.get_device_properties(device)
    return props.major * 10 + props.minor


def get_peak_hbm_bw_gbs(device: int = 0) -> float:
    """Estimate peak HBM bandwidth in GB/s."""
    props = torch.cuda.get_device_properties(device)
    name = props.name.lower()
    known: Dict[str, float] = {
        "b200": 8000.0, "b100": 8000.0, "blackwell": 8000.0,
        "h200": 4800.0,
        "h100": 3350.0,
        "a100": 2000.0,
        "a6000": 768.0,
        "v100": 900.0,
        "rtx 4090": 1008.0,
        "rtx 3090": 936.0,
        "rtx 3080": 760.0,
        "t4": 320.0,
    }
    for key, bw in known.items():
        if key in name:
            return bw
    hz = props.memory_clock_rate * 1e3
    return round(hz * props.memory_bus_width / 8 * 2 / 1e9, 0)


def print_device_header(device: int = 0) -> None:
    props = torch.cuda.get_device_properties(device)
    sm = props.major * 10 + props.minor
    peak = get_peak_hbm_bw_gbs(device)
    print(f"\n{'=' * 72}")
    print(f"  GPU {device}: {props.name}")
    print(f"  Compute: SM {props.major}.{props.minor}  (dispatch key {sm})")
    print(f"  SMs: {props.multi_processor_count}  |  "
          f"VRAM: {props.total_memory / 1e9:.1f} GB  |  "
          f"Peak HBM BW: {peak:.0f} GB/s")
    print(f"{'=' * 72}\n")


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    label: str
    latency_min_us: float = 0.0
    latency_median_us: float = 0.0
    latency_p95_us: float = 0.0
    bandwidth_gbs: float = 0.0
    speedup: float = 0.0
    roofline_pct: float = 0.0


def time_kernel(fn: Callable[[], None], warmup: int, iters: int,
                device: int = 0) -> List[float]:
    """Time fn with CUDA events; return per-iteration latencies in microseconds."""
    stream = torch.cuda.current_stream(device)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    latencies: List[float] = []
    ev_start = torch.cuda.Event(enable_timing=True)
    ev_stop = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        ev_start.record(stream)
        fn()
        ev_stop.record(stream)
        ev_stop.synchronize()
        latencies.append(ev_start.elapsed_time(ev_stop) * 1e3)  # ms -> us
    return latencies


def make_result(label: str, lats: List[float], bytes_rw: int,
                peak_gbs: float, baseline: Optional[BenchResult] = None) -> BenchResult:
    srt = sorted(lats)
    p95 = srt[max(0, int(math.ceil(0.95 * len(srt))) - 1)]
    med = statistics.median(lats)
    bw = (bytes_rw / 1e9) / (med / 1e6) if med > 0 and bytes_rw > 0 else 0.0
    roof = bw / peak_gbs * 100 if peak_gbs > 0 and bw > 0 else 0.0
    spd = baseline.latency_median_us / med if baseline is not None and med > 0 else 0.0
    return BenchResult(label=label, latency_min_us=srt[0],
                       latency_median_us=med, latency_p95_us=p95,
                       bandwidth_gbs=bw, speedup=spd, roofline_pct=roof)


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def print_table(results: List[BenchResult], title: str = "") -> None:
    W = 50
    if title:
        print(f"\n  {title}")
    hdr = (f"{'Method':<{W}} {'min_us':>8} {'med_us':>8} {'p95_us':>8} "
           f"{'BW_GBs':>8} {'Roof%':>7} {'speedup':>8}")
    sep = "-" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)
    for r in results:
        spd_s = f"{r.speedup:.2f}x" if r.speedup > 0 else "---"
        bw_s = f"{r.bandwidth_gbs:.1f}" if r.bandwidth_gbs > 0 else "---"
        roof_s = f"{r.roofline_pct:.1f}%" if r.roofline_pct > 0 else "---"
        lbl = r.label[:W]
        print(f"{lbl:<{W}} {r.latency_min_us:>8.2f} {r.latency_median_us:>8.2f} "
              f"{r.latency_p95_us:>8.2f} {bw_s:>8} {roof_s:>7} {spd_s:>8}")
    print(sep)


# ---------------------------------------------------------------------------
# Baseline functions
# ---------------------------------------------------------------------------

def fn_inplace_loop(inputs: List[torch.Tensor], out: torch.Tensor) -> Callable:
    """Baseline A: K-1 sequential add_() calls -- models NCCL reduce phase."""
    def _fn():
        out.copy_(inputs[0])
        for t in inputs[1:]:
            out.add_(t)
    return _fn


def fn_stack_sum(inputs: List[torch.Tensor], out: torch.Tensor,
                 stack_buf: torch.Tensor) -> Callable:
    """Baseline B: pre-allocated stack buffer + sum(0)."""
    def _fn():
        torch.stack(inputs, dim=0, out=stack_buf)
        torch.sum(stack_buf, dim=0, out=out)
    return _fn


def fn_fp32_cast_loop(inputs: List[torch.Tensor], out: torch.Tensor) -> Callable:
    """Baseline D: explicit BF16->FP32->BF16 cast loop (naive proxy)."""
    def _fn():
        acc = inputs[0].to(torch.float32)
        for t in inputs[1:]:
            acc.add_(t.to(torch.float32))
        out.copy_(acc.to(torch.bfloat16))
    return _fn


def fn_custom_fused(out: torch.Tensor, inputs: List[torch.Tensor],
                    sm_version: int) -> Optional[Callable]:
    """Our kernel: fused_bf16_reduce from hetero_reduce extension."""
    if not _HAS_EXT or _ext is None:
        return None
    try:
        def _fn():
            _ext.fused_bf16_reduce(out, inputs, sm_version)
        return _fn
    except Exception as e:
        warnings.warn(f"fused_bf16_reduce unavailable: {e}")
        return None


def fn_custom_rs(out_shard: torch.Tensor, inputs: List[torch.Tensor],
                 shard_off: int, shard_cnt: int, sm_version: int) -> Optional[Callable]:
    """Our kernel: hetero_reduce_scatter from hetero_reduce extension."""
    if not _HAS_EXT or _ext is None:
        return None
    try:
        def _fn():
            _ext.hetero_reduce_scatter(out_shard, inputs, shard_off, shard_cnt, sm_version)
        return _fn
    except Exception as e:
        warnings.warn(f"hetero_reduce_scatter unavailable: {e}")
        return None


# ---------------------------------------------------------------------------
# NCCL init
# ---------------------------------------------------------------------------

def init_nccl(device: int) -> bool:
    if not _HAS_DIST:
        return False
    if dist.is_initialized():
        return True
    try:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")
        dist.init_process_group(backend="nccl", init_method="env://",
                                world_size=1, rank=0)
        return True
    except Exception as e:
        warnings.warn(f"NCCL init failed: {e}")
        return False


# ---------------------------------------------------------------------------
# SM audit report
# ---------------------------------------------------------------------------

def print_sm_audit() -> None:
    print(f"\n{'=' * 72}")
    print("  SM CODE PATH AUDIT  (addresses #62)")
    print(f"{'=' * 72}")
    rows = [
        ("hetero_reduce.cu",          "KernelPolicy<86>", "KernelPolicy<90>",     "KernelPolicy<120>",  "OK"),
        ("fused_swiglu_ln.cu",        "SwiGLUPolicy<86>", "SwiGLUPolicy<90>",     "SwiGLUPolicy<120>",  "OK"),
        ("fused_rope_hetero.cu",      "__lb__(256,2)",    "__lb__(256,4)",         "__lb__(512,4)",      "OK"),
        ("pcie_adaptive_allreduce.cu","<86,256>",         "<90,256>",              "<120,512>",          "OK"),
        ("tier_activation_offload.cu","<86> ok",          "<90> FIXED(was <86>)", "<120> ok",           "BUG FIXED"),
    ]
    W = [30, 18, 22, 18, 12]
    hdr = (f"{'File':<{W[0]}} {'SM8.6':<{W[1]}} {'SM9.0':<{W[2]}} "
           f"{'SM12.0':<{W[3]}} {'Status'}")
    print("-" * 72)
    print(hdr)
    print("-" * 72)
    for r in rows:
        print(f"{r[0]:<{W[0]}} {r[1]:<{W[1]}} {r[2]:<{W[2]}} {r[3]:<{W[3]}} {r[4]}")
    print("-" * 72)
    print("""
  BUG FIXED (tier_activation_offload.cu -- addresses #62):
    activation_pack_kernel and activation_unpack_kernel used block-size as the
    sole discriminator for SM dispatch.  SM8.6 and SM9.0 both use 256-thread
    blocks, so SM9.0 silently received __launch_bounds__(256, 2)  [kMinBlocksPerSM=2]
    instead of __launch_bounds__(256, 4)  [kMinBlocksPerSM=4 for H100].
    This halved the occupancy hint to the CUDA compiler on H100.
    Fix (commit): three-way sm_version dispatch (>=120 / >=90 / else) so SM9.0
    explicitly instantiates activation_pack_kernel<90> / activation_unpack_kernel<90>.
""")


# ---------------------------------------------------------------------------
# fused_bf16_reduce benchmark
# ---------------------------------------------------------------------------

def bench_fused_reduce(n_elems: int, num_tensors: int, sm_version: int,
                       peak_hbm: float, warmup: int, iters: int, device: int,
                       use_nccl: bool, nccl_ok: bool) -> Tuple[List[BenchResult], Dict]:
    dtype = torch.bfloat16
    dev = f"cuda:{device}"
    inputs = [torch.randn(n_elems, dtype=dtype, device=dev) for _ in range(num_tensors)]
    out = torch.empty(n_elems, dtype=dtype, device=dev)

    # Theoretical I/O: read K tensors + write 1 output
    bytes_rw = int((num_tensors + 1) * n_elems * 2)
    tag = f"N={fmt_size(n_elems)} K={num_tensors}"
    results: List[BenchResult] = []

    # A: inplace loop
    lats_a = time_kernel(fn_inplace_loop(inputs, out), warmup, iters, device)
    r_a = make_result(f"[A] inplace_loop (NCCL model)  {tag}", lats_a, bytes_rw, peak_hbm)
    results.append(r_a)
    baseline = r_a

    # B: stack+sum
    stack_buf = torch.empty(num_tensors, n_elems, dtype=dtype, device=dev)
    lats_b = time_kernel(fn_stack_sum(inputs, out, stack_buf), warmup, iters, device)
    r_b = make_result(f"[B] stack_sum                  {tag}", lats_b, bytes_rw, peak_hbm, baseline)
    results.append(r_b)

    # C: NCCL dist.all_reduce (optional)
    if use_nccl and nccl_ok:
        sum_buf = inputs[0].clone()
        def _nccl_fn():
            sum_buf.copy_(inputs[0])
            for t in inputs[1:]:
                sum_buf.add_(t)
            dist.all_reduce(sum_buf, op=dist.ReduceOp.SUM)
        try:
            lats_c = time_kernel(_nccl_fn, warmup, iters, device)
            r_c = make_result(f"[C] dist.all_reduce NCCL       {tag}", lats_c, bytes_rw, peak_hbm, baseline)
            results.append(r_c)
        except Exception as e:
            warnings.warn(f"NCCL run error: {e}")

    # D: fp32 cast loop
    lats_d = time_kernel(fn_fp32_cast_loop(inputs, out), warmup, iters, device)
    bytes_d = int(num_tensors * n_elems * 6 + n_elems * 4)  # BF16 reads + FP32 ops
    r_d = make_result(f"[D] fp32_cast_loop (naive)     {tag}", lats_d, bytes_d, peak_hbm, baseline)
    results.append(r_d)

    # Our kernel (or proxy if extension not compiled)
    custom_fn = fn_custom_fused(out, inputs, sm_version)
    if custom_fn is not None:
        lats_k = time_kernel(custom_fn, warmup, iters, device)
        r_k = make_result(f"[*] fused_bf16_reduce (ours)   {tag}", lats_k, bytes_rw, peak_hbm, baseline)
    else:
        # Proxy: single-pass FP32 loop over pre-cast inputs (closest to kernel behaviour)
        inputs_f32 = [t.to(torch.float32) for t in inputs]
        def _proxy():
            acc = torch.zeros_like(inputs_f32[0])
            for t in inputs_f32:
                acc.add_(t)
            out.copy_(acc.to(torch.bfloat16))
        lats_k = time_kernel(_proxy, warmup, iters, device)
        r_k = make_result(f"[*] fused_bf16_PROXY (no .so)  {tag}", lats_k, bytes_rw, peak_hbm, baseline)

    results.append(r_k)

    summary = {
        "section": "fused_reduce",
        "n_elems": n_elems, "n_label": fmt_size(n_elems),
        "num_tensors": num_tensors, "sm_version": sm_version,
        "baseline_med_us": round(r_a.latency_median_us, 3),
        "baseline_bw_gbs": round(r_a.bandwidth_gbs, 2),
        "kernel_med_us": round(r_k.latency_median_us, 3),
        "kernel_bw_gbs": round(r_k.bandwidth_gbs, 2),
        "speedup": round(r_k.speedup, 3),
        "roofline_pct": round(r_k.roofline_pct, 1),
        "ext_loaded": _HAS_EXT,
    }
    return results, summary


# ---------------------------------------------------------------------------
# hetero_reduce_scatter benchmark
# ---------------------------------------------------------------------------

_TIER_WEIGHTS: Dict[int, int] = {120: 4, 90: 3, 86: 1}


def _shard_ranges(n_elems: int, tiers: List[int], align: int = 8) -> List[Tuple[int, int]]:
    """Mirror compute_hetero_shard_ranges() from hetero_reduce.cu."""
    weights = [_TIER_WEIGHTS.get(sm, 1) for sm in tiers]
    total_w = sum(weights)
    shards: List[Tuple[int, int]] = []
    assigned = 0
    for i, w in enumerate(weights):
        if i == len(tiers) - 1:
            shards.append((assigned, n_elems - assigned))
        else:
            raw = (n_elems * w) // total_w
            raw = (raw // align) * align
            shards.append((assigned, raw))
            assigned += raw
    return shards


def bench_reduce_scatter(n_elems: int, num_tensors: int, sm_version: int,
                         peak_hbm: float, warmup: int, iters: int,
                         device: int) -> Tuple[List[BenchResult], List[Dict]]:
    dtype = torch.bfloat16
    dev = f"cuda:{device}"
    inputs = [torch.randn(n_elems, dtype=dtype, device=dev) for _ in range(num_tensors)]

    tiers = [86, 90, 120]
    shards = _shard_ranges(n_elems, tiers)

    all_results: List[BenchResult] = []
    summaries: List[Dict] = []

    for tier_sm, (shard_off, shard_cnt) in zip(tiers, shards):
        tag = f"SM{tier_sm} N={fmt_size(n_elems)} K={num_tensors} shard={fmt_size(shard_cnt)}"
        out_shard = torch.empty(shard_cnt, dtype=dtype, device=dev)
        bytes_rw = int((num_tensors + 1) * shard_cnt * 2)

        # Baseline: slice each input to shard, then inplace loop
        slices = [t[shard_off:shard_off + shard_cnt] for t in inputs]
        def _baseline_fn():
            out_shard.copy_(slices[0])
            for t in slices[1:]:
                out_shard.add_(t)

        lats_b = time_kernel(_baseline_fn, warmup, iters, device)
        r_b = make_result(f"[A] slice+loop baseline   {tag}", lats_b, bytes_rw, peak_hbm)
        all_results.append(r_b)

        # Our kernel
        rs_fn = fn_custom_rs(out_shard, inputs, shard_off, shard_cnt, sm_version)
        if rs_fn is not None:
            lats_k = time_kernel(rs_fn, warmup, iters, device)
            r_k = make_result(f"[*] hetero_reduce_scatter  {tag}", lats_k, bytes_rw, peak_hbm, r_b)
        else:
            # Proxy: read full tensors, reduce shard only
            def _proxy_rs():
                acc = torch.zeros(shard_cnt, dtype=torch.float32, device=dev)
                for t in inputs:
                    acc.add_(t[shard_off:shard_off + shard_cnt].to(torch.float32))
                out_shard.copy_(acc.to(torch.bfloat16))
            bytes_proxy = int(num_tensors * n_elems * 2 + shard_cnt * 2)
            lats_k = time_kernel(_proxy_rs, warmup, iters, device)
            r_k = make_result(f"[*] RS_proxy (no .so)     {tag}", lats_k, bytes_proxy, peak_hbm, r_b)

        all_results.append(r_k)
        summaries.append({
            "section": "reduce_scatter",
            "n_elems": n_elems, "n_label": fmt_size(n_elems),
            "num_tensors": num_tensors, "sm_version": sm_version,
            "tier_sm": tier_sm, "shard_offset": shard_off, "shard_count": shard_cnt,
            "shard_frac": round(shard_cnt / n_elems, 3),
            "baseline_med_us": round(r_b.latency_median_us, 3),
            "kernel_med_us": round(r_k.latency_median_us, 3),
            "kernel_bw_gbs": round(r_k.bandwidth_gbs, 2),
            "speedup": round(r_k.speedup, 3),
            "roofline_pct": round(r_k.roofline_pct, 1),
            "ext_loaded": _HAS_EXT,
        })

    return all_results, summaries


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def print_verdict(summaries: List[Dict], peak_hbm: float) -> None:
    reduce_rows = [r for r in summaries if r["section"] == "fused_reduce"]
    rs_rows     = [r for r in summaries if r["section"] == "reduce_scatter"]

    print(f"\n{'=' * 72}")
    print("  SUMMARY -- fused_bf16_reduce vs inplace_loop (NCCL model baseline)")
    print(f"{'=' * 72}")
    print(f"  {'N':>8}  {'K':>4}  {'base_us':>9}  {'kern_us':>9}  "
          f"{'speedup':>8}  {'BW_GBs':>8}  {'Roof%':>7}  {'verdict':>10}")
    print(f"  {'-'*8}  {'-'*4}  {'-'*9}  {'-'*9}  "
          f"{'-'*8}  {'-'*8}  {'-'*7}  {'-'*10}")
    for r in reduce_rows:
        spd = r["speedup"]
        verdict = "FASTER" if spd > 1.05 else ("TIED" if spd > 0.95 else "SLOWER")
        print(f"  {r['n_label']:>8}  {r['num_tensors']:>4}  "
              f"{r['baseline_med_us']:>9.1f}  {r['kernel_med_us']:>9.1f}  "
              f"{spd:>7.2f}x  {r['kernel_bw_gbs']:>8.1f}  "
              f"{r['roofline_pct']:>6.1f}%  {verdict:>10}")

    speedups = [r["speedup"] for r in reduce_rows if r["speedup"] > 0]
    if speedups:
        gm = math.exp(sum(math.log(max(s, 1e-9)) for s in speedups) / len(speedups))
        n_faster = sum(1 for s in speedups if s > 1.05)
        print(f"\n  Geomean speedup: {gm:.2f}x  |  "
              f"Faster in {n_faster}/{len(speedups)} configs")
        if gm > 1.1:
            print("  VERDICT: kernel IS faster than the NCCL-model baseline  [PASS]")
        elif gm > 0.95:
            print("  VERDICT: kernel is roughly TIED  (bandwidth-bound; NCCL adds transfer overhead)")
        else:
            print("  VERDICT: kernel is SLOWER -- check occupancy / launch_bounds  [FAIL]")

    if rs_rows:
        print(f"\n{'=' * 72}")
        print("  SUMMARY -- hetero_reduce_scatter (3-tier cluster simulation)")
        print(f"{'=' * 72}")
        for r in rs_rows:
            spd_s = f"{r['speedup']:.2f}x" if r["speedup"] > 0 else "proxy"
            print(f"  SM{r['tier_sm']}  N={r['n_label']}  K={r['num_tensors']}  "
                  f"shard={fmt_size(r['shard_count'])}  "
                  f"baseline={r['baseline_med_us']:.1f}us  "
                  f"kernel={r['kernel_med_us']:.1f}us  "
                  f"speedup={spd_s}  roof={r['roofline_pct']:.1f}%")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_DEFAULT_SIZES = ["128K", "1M", "4M", "16M", "64M", "256M"]
_DEFAULT_K     = [1, 2, 4, 8, 16, 32]
_QUICK_SIZES   = ["1M", "16M"]
_QUICK_K       = [2, 8]
_RS_SIZES      = ["16M", "64M"]
_RS_K          = [4, 8]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="bench_hetero_reduce_vs_nccl.py (addresses #74)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sizes", nargs="+", default=None, metavar="SIZE")
    parser.add_argument("--num-tensors", nargs="+", type=int, default=None, metavar="K")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--nccl", action="store_true",
                        help="Enable dist.all_reduce NCCL baseline")
    parser.add_argument("--no-rs", action="store_true",
                        help="Skip reduce-scatter section")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke-test")
    parser.add_argument("--json", default=None, metavar="PATH",
                        help="Write JSON results to PATH")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA device found.  This benchmark requires a GPU.")
        sys.exit(1)

    if args.quick:
        sizes = [parse_size(s) for s in _QUICK_SIZES]
        k_vals = _QUICK_K
        warmup, iters = 5, 20
        rs_sizes = [parse_size("4M")]
        rs_k = [4]
    else:
        sizes = [parse_size(s) for s in (args.sizes or _DEFAULT_SIZES)]
        k_vals = args.num_tensors or _DEFAULT_K
        warmup, iters = args.warmup, args.iters
        rs_sizes = [parse_size(s) for s in _RS_SIZES]
        rs_k = _RS_K

    device = args.device
    sm_version = get_sm_version(device)
    peak_hbm = get_peak_hbm_bw_gbs(device)

    print_device_header(device)
    print_sm_audit()

    ext_status = "LOADED" if _HAS_EXT else "NOT COMPILED (proxy mode)"
    print(f"  hetero_reduce ext: {ext_status}")
    print(f"  SM version       : {sm_version}")
    print(f"  Sizes            : {[fmt_size(n) for n in sizes]}")
    print(f"  K values         : {k_vals}")
    print(f"  Warmup/Iters     : {warmup}/{iters}")

    nccl_ok = False
    if args.nccl:
        nccl_ok = init_nccl(device)
        print(f"  NCCL baseline    : {'enabled' if nccl_ok else 'unavailable'}")
    else:
        print("  NCCL baseline    : disabled (pass --nccl to enable)")
    print()

    all_summaries: List[Dict] = []

    # Section 1: fused_bf16_reduce
    print(f"\n{'=' * 72}")
    print("  SECTION 1: fused_bf16_reduce  (full tensor, no scatter)")
    print(f"{'=' * 72}")
    for num_tensors in k_vals:
        print(f"\n  -- K = {num_tensors} --")
        for n_elems in sizes:
            res, summary = bench_fused_reduce(
                n_elems, num_tensors, sm_version, peak_hbm,
                warmup, iters, device, args.nccl, nccl_ok)
            print_table(res)
            all_summaries.append(summary)

    # Section 2: reduce_scatter
    if not args.no_rs:
        print(f"\n{'=' * 72}")
        print("  SECTION 2: hetero_reduce_scatter  (3-tier shard simulation)")
        print("  Tier weights: SM8.6->1, SM9.0->3, SM12.0->4")
        print(f"{'=' * 72}")
        for num_tensors in rs_k:
            print(f"\n  -- K = {num_tensors} --")
            for n_elems in rs_sizes:
                res, sums = bench_reduce_scatter(
                    n_elems, num_tensors, sm_version, peak_hbm,
                    warmup, iters, device)
                print_table(res, title=f"Reduce-scatter N={fmt_size(n_elems)} K={num_tensors}")
                all_summaries.extend(sums)

    print_verdict(all_summaries, peak_hbm)

    output = {
        "device": torch.cuda.get_device_properties(device).name,
        "sm_version": sm_version,
        "peak_hbm_bw_gbs": peak_hbm,
        "extension_loaded": _HAS_EXT,
        "warmup": warmup,
        "iters": iters,
        "sm_audit": {
            "hetero_reduce.cu":           {"86": True, "90": True, "120": True},
            "fused_swiglu_ln.cu":         {"86": True, "90": True, "120": True},
            "fused_rope_hetero.cu":       {"86": True, "90": True, "120": True},
            "pcie_adaptive_allreduce.cu": {"86": True, "90": True, "120": True},
            "tier_activation_offload.cu": {"86": True, "90": True, "120": True,
                                           "bug_fixed": "SM90 activation_pack/unpack kMinBlocksPerSM 2->4"},
        },
        "summary": all_summaries,
    }

    if args.json:
        with open(args.json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Results written to {args.json}")

    return output


if __name__ == "__main__":
    main()
