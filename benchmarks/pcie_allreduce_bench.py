# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team / Neuron_SP
"""
benchmarks/pcie_allreduce_bench.py — PCIe adaptive ring-allreduce benchmark

Benchmarks the PCIe adaptive ring-allreduce pipeline
(csrc/hetero_reduce/pcie_adaptive_allreduce.cu) against:
    A) torch.distributed.all_reduce (NCCL over NVLink when available)
    B) manual ring-allreduce with cudaMemcpyPeerAsync (simulate NCCL over PCIe)
    C) chunk-at-a-time ring with PyTorch all_reduce fallback

The custom kernel implements:
    1. Runtime bandwidth probe via a small ping-pong transfer
    2. Adaptive chunk sizing: chunk = max(2 MB, 5 × bandwidth_GBs × latency_target)
    3. Double-buffered ring pipeline: compute ↔ transfer overlap

Modes:
    --mode single   : single-GPU micro-kernel throughput only
    --mode multi    : torchrun multi-GPU allreduce comparison (default)

Single-GPU launch:
    python benchmarks/pcie_allreduce_bench.py --mode single --device 0

Multi-GPU launch (4 GPUs, simulates 4-node PCIe cluster):
    torchrun --nproc_per_node=4 benchmarks/pcie_allreduce_bench.py --mode multi

Tensor sizes swept:
    8 MB, 32 MB, 128 MB, 512 MB, 2048 MB  (BF16 gradient tensors)

Metrics:
    - Ring-reduce kernel throughput (GB/s)
    - Effective allreduce throughput (GB/s) = tensor_bytes / latency
    - Overlap efficiency (%) = 1 - idle_fraction (double-buffer model)
    - vs NCCL speedup / ratio
"""

from __future__ import annotations

import argparse
import os
import sys
import math
from typing import List, Optional, Tuple

import torch

try:
    import deepspeed.ops.pcie_allreduce as _ext  # type: ignore[import]
    _HAS_EXT = True
except ImportError:
    _HAS_EXT = False

_HAS_DIST = False
try:
    import torch.distributed as dist
    _HAS_DIST = True
except ImportError:
    pass

from kernel_bench import (
    BenchmarkHarness,
    BenchResult,
    print_device_header,
    print_results_table,
    get_device_info,
    print_json,
)

# ---------------------------------------------------------------------------
# Size helpers
# ---------------------------------------------------------------------------

def _parse_mb(s: str) -> int:
    """Parse '32MB', '128' (interpreted as MB), '1GB' → bytes."""
    s = s.strip().upper()
    if s.endswith("GB"):
        return int(float(s[:-2]) * 1024 * 1024 * 1024)
    if s.endswith("MB"):
        return int(float(s[:-2]) * 1024 * 1024)
    # bare number treated as MB
    return int(float(s) * 1024 * 1024)


def _bytes_to_bf16_elems(n_bytes: int) -> int:
    return n_bytes // 2


# ---------------------------------------------------------------------------
# Baseline implementations
# ---------------------------------------------------------------------------

def _nccl_allreduce(tensor: torch.Tensor) -> None:
    """Baseline A: NCCL all_reduce (NVLink-optimised when available)."""
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)


def _manual_ring_allreduce_pytorch(
    tensor: torch.Tensor,
    world_size: int,
    rank: int,
    group,
) -> None:
    """
    Baseline B: Manual ring-allreduce using dist.send / dist.recv.
    Models the same algorithm as the custom CUDA kernel to isolate
    the kernel optimisation benefit from the algorithmic difference.

    reduce-scatter phase: world_size-1 rounds
    all-gather phase:     world_size-1 rounds
    """
    n = tensor.numel()
    chunk = math.ceil(n / world_size)
    buf = tensor.view(-1)

    left = (rank - 1) % world_size
    right = (rank + 1) % world_size

    # reduce-scatter
    for step in range(world_size - 1):
        send_chunk_idx = (rank - step) % world_size
        recv_chunk_idx = (rank - step - 1) % world_size
        s = send_chunk_idx * chunk
        e = min(s + chunk, n)
        r_s = recv_chunk_idx * chunk
        r_e = min(r_s + chunk, n)
        send_req = dist.isend(buf[s:e].contiguous(), dst=right, group=group)
        tmp = torch.empty(r_e - r_s, dtype=tensor.dtype, device=tensor.device)
        dist.recv(tmp, src=left, group=group)
        buf[r_s:r_e].add_(tmp)
        send_req.wait()

    # all-gather
    for step in range(world_size - 1):
        send_chunk_idx = (rank - step + 1) % world_size
        recv_chunk_idx = (rank - step) % world_size
        s = send_chunk_idx * chunk
        e = min(s + chunk, n)
        r_s = recv_chunk_idx * chunk
        r_e = min(r_s + chunk, n)
        send_req = dist.isend(buf[s:e].contiguous(), dst=right, group=group)
        dist.recv(buf[r_s:r_e], src=left, group=group)
        send_req.wait()


def _adaptive_chunk_bytes(bandwidth_gbs: float, latency_target_ms: float = 2.0) -> int:
    """
    Mirror the adaptive chunk-sizing formula from pcie_adaptive_allreduce.cu.
    chunk = max(2 MB, 5 × bandwidth_GBs × latency_target_s)
    """
    floor_bytes = 2 * 1024 * 1024   # 2 MB minimum
    optimal = int(5 * bandwidth_gbs * 1e9 * (latency_target_ms / 1e3))
    return max(floor_bytes, optimal)


def _probe_pcie_bandwidth_pytorch(src_dev: int, dst_dev: int, size_mb: float = 64.0) -> float:
    """
    Probe GPU-to-GPU PCIe bandwidth using torch.cuda.Event timing.
    Returns bandwidth in GB/s.
    """
    n_bytes = int(size_mb * 1024 * 1024)
    n_floats = n_bytes // 4
    warmup, iters = 3, 10

    src_tensor = torch.ones(n_floats, dtype=torch.float32, device=f"cuda:{src_dev}")
    dst_tensor = torch.empty(n_floats, dtype=torch.float32, device=f"cuda:{dst_dev}")

    for _ in range(warmup):
        dst_tensor.copy_(src_tensor)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record(torch.cuda.current_stream(src_dev))
    for _ in range(iters):
        dst_tensor.copy_(src_tensor)
    stop.record(torch.cuda.current_stream(src_dev))
    stop.synchronize()

    elapsed_s = start.elapsed_time(stop) / 1e3 / iters
    return (n_bytes / elapsed_s) / 1e9


# ---------------------------------------------------------------------------
# Single-GPU micro-kernel throughput benchmark
# ---------------------------------------------------------------------------

def _bench_single_gpu(
    harness: BenchmarkHarness,
    tensor_bytes: int,
    device: int,
    sm_version: int,
    simulated_bw_gbs: Optional[float],
) -> List[BenchResult]:
    """
    Benchmark ring-reduce kernel throughput on a single GPU.
    Simulates the ring-reduce kernel operating on one chunk at a time.
    """
    dev = f"cuda:{device}"
    n_elems = _bytes_to_bf16_elems(tensor_bytes)
    # Simulate world_size=4 → chunk = tensor_bytes / 4
    chunk_elems = max(n_elems // 4, 1)
    chunk_bytes = chunk_elems * 2  # BF16

    dst = torch.zeros(chunk_elems, dtype=torch.bfloat16, device=dev)
    src = torch.ones(chunk_elems, dtype=torch.bfloat16, device=dev)

    mb = tensor_bytes / 1e6
    tag = f"tensor={mb:.0f}MB"

    def _torch_reduce_add():
        dst.add_(src)

    def _custom_ring_reduce():
        if _HAS_EXT:
            _ext.launch_pcie_ring_reduce(dst, src, chunk_elems, sm_version)
        else:
            dst.add_(src)

    base = harness.run(
        label=f"torch.add (chunk reduce) | {tag}",
        fn=_torch_reduce_add,
        bytes_accessed=chunk_bytes * 3,  # read dst + src, write dst
        flops=chunk_elems,
    )

    kernel_label = "pcie_ring_reduce kernel" if _HAS_EXT else "pcie_ring_reduce (sim)"
    custom = harness.run(
        label=f"{kernel_label:<25s} | {tag}",
        fn=_custom_ring_reduce,
        bytes_accessed=chunk_bytes * 3,
        flops=chunk_elems,
        baseline_label=base.label,
    )
    harness.compare_to_baseline(custom, base)

    # Also benchmark finalisation kernel (scale by 1/world_size)
    world_size = 4

    def _torch_finalise():
        dst.div_(world_size)

    def _custom_finalise():
        if _HAS_EXT:
            _ext.launch_pcie_ring_finalise(dst, chunk_elems, world_size, sm_version)
        else:
            dst.div_(world_size)

    base_fin = harness.run(
        label=f"torch.div (finalise)     | {tag}",
        fn=_torch_finalise,
        bytes_accessed=chunk_bytes * 2,
        flops=chunk_elems,
    )
    custom_fin = harness.run(
        label=f"pcie_finalise kernel     | {tag}",
        fn=_custom_finalise,
        bytes_accessed=chunk_bytes * 2,
        flops=chunk_elems,
    )
    harness.compare_to_baseline(custom_fin, base_fin)

    # Adaptive chunk-size analysis (printed inline)
    bw = simulated_bw_gbs or 16.0
    chunk_size = _adaptive_chunk_bytes(bw)
    print(
        f"  [adaptive] BW={bw:.1f} GB/s → optimal_chunk={chunk_size//1024//1024:.0f} MB  "
        f"({'fits' if chunk_size <= tensor_bytes else 'whole tensor'})"
    )

    return [base, custom, base_fin, custom_fin]


# ---------------------------------------------------------------------------
# Multi-GPU allreduce benchmark (torchrun)
# ---------------------------------------------------------------------------

def _bench_multi_gpu(
    harness: BenchmarkHarness,
    tensor_bytes: int,
    rank: int,
    world_size: int,
    group,
) -> List[BenchResult]:
    device = rank  # one GPU per rank
    dev = f"cuda:{device}"
    n_elems = _bytes_to_bf16_elems(tensor_bytes)

    # Each rank owns a gradient tensor that needs allreduction
    tensor = torch.randn(n_elems, dtype=torch.bfloat16, device=dev)
    # effective bytes for allreduce: 2 × (world_size-1)/world_size × N (optimal ring)
    effective_bytes = 2 * (world_size - 1) / world_size * tensor_bytes

    mb = tensor_bytes / 1e6
    tag = f"tensor={mb:.0f}MB ws={world_size}"

    def _nccl_fn():
        t = tensor.clone()
        _nccl_allreduce(t)

    def _manual_ring_fn():
        t = tensor.clone()
        _manual_ring_allreduce_pytorch(t, world_size, rank, group)

    def _custom_fn():
        t = tensor.clone()
        if _HAS_EXT:
            _ext.pcie_adaptive_allreduce(t, world_size, rank)
        else:
            _nccl_allreduce(t)

    base_nccl = harness.run(
        label=f"NCCL all_reduce          | {tag}",
        fn=_nccl_fn,
        bytes_accessed=int(effective_bytes),
    )

    base_ring = harness.run(
        label=f"manual ring (PyTorch)    | {tag}",
        fn=_manual_ring_fn,
        bytes_accessed=int(effective_bytes),
    )

    kernel_label = "pcie_adaptive_allreduce" if _HAS_EXT else "pcie_allreduce (sim)"
    custom = harness.run(
        label=f"{kernel_label:<25s} | {tag}",
        fn=_custom_fn,
        bytes_accessed=int(effective_bytes),
        baseline_label=base_nccl.label,
    )
    harness.compare_to_baseline(custom, base_nccl)

    return [base_nccl, base_ring, custom]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PCIe adaptive ring-allreduce benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["single", "multi"], default="single",
        help="'single' = single-GPU micro-kernel bench; 'multi' = torchrun multi-GPU",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device (single mode)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--sizes", nargs="+", default=["8MB", "32MB", "128MB", "512MB"],
        help="Tensor sizes to sweep, e.g. 8MB 128MB 512MB",
    )
    parser.add_argument(
        "--simulated-bw", type=float, default=None,
        help="Simulated PCIe bandwidth in GB/s for chunk-size analysis (single mode)",
    )
    parser.add_argument("--json", metavar="PATH", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA device available.")
        sys.exit(1)

    # ── Multi-GPU mode ───────────────────────────────────────────────────────
    if args.mode == "multi":
        if not _HAS_DIST:
            print("ERROR: torch.distributed not available.")
            sys.exit(1)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        group = dist.new_group(ranks=list(range(world_size)))
        device = rank
        torch.cuda.set_device(device)

        if rank == 0:
            print_device_header(device)

        harness = BenchmarkHarness(warmup=args.warmup, iters=args.iters, device=device)
        all_results: List[BenchResult] = []

        for size_str in args.sizes:
            tensor_bytes = _parse_mb(size_str)
            results = _bench_multi_gpu(harness, tensor_bytes, rank, world_size, group)
            all_results.extend(results)
            dist.barrier()

        if rank == 0:
            print_results_table(all_results)
            if args.json:
                with open(args.json, "w") as f:
                    f.write(print_json(all_results))
                print(f"\nResults saved to {args.json}")

        dist.destroy_process_group()
        return

    # ── Single-GPU mode ──────────────────────────────────────────────────────
    device_info = get_device_info(args.device)
    sm_version = device_info["sm_version"]
    print_device_header(args.device)

    if not _HAS_EXT:
        print(
            "WARNING: deepspeed.ops.pcie_allreduce not found.\n"
            "         Custom kernel results use PyTorch simulation.\n"
            "         Build with: cd csrc/hetero_reduce && python setup.py install\n"
        )

    # Probe PCIe bandwidth if multiple GPUs present
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1 and args.simulated_bw is None:
        print(f"Probing GPU 0 → GPU 1 PCIe bandwidth (64 MB)...")
        probed_bw = _probe_pcie_bandwidth_pytorch(0, 1, size_mb=64.0)
        print(f"  Measured: {probed_bw:.2f} GB/s\n")
        simulated_bw = probed_bw
    else:
        simulated_bw = args.simulated_bw or 16.0
        print(f"Using simulated PCIe bandwidth: {simulated_bw:.1f} GB/s\n")

    harness = BenchmarkHarness(warmup=args.warmup, iters=args.iters, device=args.device)
    all_results: List[BenchResult] = []

    for size_str in args.sizes:
        tensor_bytes = _parse_mb(size_str)
        mb = tensor_bytes / 1e6
        print(f"\n{'─'*70}")
        print(f"  Tensor: {mb:.0f} MB  ({_bytes_to_bf16_elems(tensor_bytes) // 1024 // 1024}M BF16 elements)")
        print(f"{'─'*70}")
        results = _bench_single_gpu(
            harness, tensor_bytes, args.device, sm_version, simulated_bw
        )
        all_results.extend(results)
        print_results_table(results)

    # Summary of custom kernel speedup
    print("\n" + "="*70)
    print("  SUMMARY — ring-reduce kernel vs torch.add baseline")
    print("="*70)
    custom = [r for r in all_results if "kernel" in r.label or "sim" in r.label]
    if custom:
        print_results_table(custom)

    if args.json:
        with open(args.json, "w") as f:
            f.write(print_json(all_results))
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
