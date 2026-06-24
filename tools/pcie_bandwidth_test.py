#!/usr/bin/env python3
"""
PCIe Bandwidth Test — GPU-to-GPU D2D Copy Bandwidth Matrix
===========================================================
Measures device-to-device copy bandwidth between every pair of GPUs using a
512 MB tensor transfer, timed with torch.cuda.Event for accurate GPU-side
measurement.  The resulting matrix helps pinpoint PCIe topology bottlenecks
(e.g., cross-socket hops, shared PCIe switches, missing NVLink paths).

Usage
-----
    python tools/pcie_bandwidth_test.py [--size-mb 512] [--warmup 3] [--iters 10]

Output
------
    Bandwidth matrix (GB/s) — rows = src GPU, cols = dst GPU
    Diagonal entries are omitted (shown as "  —  ").
"""

import argparse
import sys

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bytes_for_mb(mb: float) -> int:
    return int(mb * 1024 * 1024)


def _measure_d2d_bandwidth_gbps(
    src: int,
    dst: int,
    n_bytes: int,
    warmup: int,
    iters: int,
) -> float:
    """Return average D2D copy bandwidth in GB/s from *src* to *dst*."""
    n_floats = n_bytes // 4  # float32

    # Allocate source tensor on src GPU
    with torch.cuda.device(src):
        src_tensor = torch.ones(n_floats, dtype=torch.float32, device=f"cuda:{src}")

    # Allocate destination tensor on dst GPU
    with torch.cuda.device(dst):
        dst_tensor = torch.empty(n_floats, dtype=torch.float32, device=f"cuda:{dst}")

    # Use the src-side stream for timing; the copy is initiated from src
    src_stream = torch.cuda.Stream(device=src)

    # Warm-up passes (not timed)
    for _ in range(warmup):
        with torch.cuda.stream(src_stream):
            dst_tensor.copy_(src_tensor, non_blocking=True)
        src_stream.synchronize()

    # Timed passes with cuda.Event
    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)

    total_ms = 0.0
    for _ in range(iters):
        with torch.cuda.stream(src_stream):
            start_event.record(src_stream)
            dst_tensor.copy_(src_tensor, non_blocking=True)
            end_event.record(src_stream)
        src_stream.synchronize()
        total_ms += start_event.elapsed_time(end_event)  # milliseconds

    avg_ms  = total_ms / iters
    gbps    = (n_bytes / 1e9) / (avg_ms / 1e3)
    return gbps


# ---------------------------------------------------------------------------
# Matrix builder
# ---------------------------------------------------------------------------

def run_bandwidth_matrix(
    n_gpus: int,
    size_mb: float,
    warmup: int,
    iters: int,
) -> list[list[float | None]]:
    """Run D2D bandwidth measurement for every ordered (src, dst) pair."""
    n_bytes = _bytes_for_mb(size_mb)
    matrix: list[list[float | None]] = [
        [None] * n_gpus for _ in range(n_gpus)
    ]

    total_pairs = n_gpus * (n_gpus - 1)
    done = 0

    for src in range(n_gpus):
        for dst in range(n_gpus):
            if src == dst:
                continue
            done += 1
            print(
                f"  [{done:3d}/{total_pairs}] GPU {src} → GPU {dst} ...",
                end="",
                flush=True,
            )
            gbps = _measure_d2d_bandwidth_gbps(src, dst, n_bytes, warmup, iters)
            matrix[src][dst] = gbps
            print(f"  {gbps:6.2f} GB/s")

    return matrix


# ---------------------------------------------------------------------------
# Pretty-print the matrix
# ---------------------------------------------------------------------------

def print_matrix(matrix: list[list[float | None]], n_gpus: int, size_mb: float) -> None:
    col_w = 9  # width per cell

    header_parts = [f"{'src \\ dst':>10}"]
    for dst in range(n_gpus):
        header_parts.append(f"{'GPU ' + str(dst):>{col_w}}")
    print("\n" + "  ".join(header_parts))
    print("  " + "-" * (10 + (col_w + 2) * n_gpus))

    for src in range(n_gpus):
        row_parts = [f"{'GPU ' + str(src):>10}"]
        for dst in range(n_gpus):
            if src == dst:
                row_parts.append(f"{'—':>{col_w}}")
            else:
                val = matrix[src][dst]
                row_parts.append(f"{val:>{col_w}.2f}" if val is not None else f"{'N/A':>{col_w}}")
        print("  ".join(row_parts))

    print(
        f"\n  Units: GB/s  |  Tensor size: {size_mb:.0f} MB (float32)  "
        f"|  Rows = source GPU, Columns = destination GPU"
    )


# ---------------------------------------------------------------------------
# Bottleneck analysis
# ---------------------------------------------------------------------------

def print_analysis(matrix: list[list[float | None]], n_gpus: int) -> None:
    pairs = [
        (matrix[s][d], s, d)
        for s in range(n_gpus)
        for d in range(n_gpus)
        if s != d and matrix[s][d] is not None
    ]
    if not pairs:
        return

    pairs.sort()
    slowest_bw, slow_s, slow_d = pairs[0]
    fastest_bw, fast_s, fast_d = pairs[-1]

    # Asymmetry check: compare (s→d) vs (d→s)
    asymmetric = []
    checked = set()
    for bw, s, d in pairs:
        key = (min(s, d), max(s, d))
        if key in checked:
            continue
        checked.add(key)
        rev = matrix[d][s]
        if rev is not None:
            ratio = max(bw, rev) / (min(bw, rev) + 1e-9)
            if ratio > 1.25:  # >25 % asymmetry
                asymmetric.append((s, d, bw, rev, ratio))

    print("\n── Bottleneck Analysis ─────────────────────────────────────────────")
    print(f"  Fastest link : GPU {fast_s} → GPU {fast_d}  ({fastest_bw:.2f} GB/s)")
    print(f"  Slowest link : GPU {slow_s} → GPU {slow_d}  ({slowest_bw:.2f} GB/s)")
    print(f"  Speed ratio  : {fastest_bw / (slowest_bw + 1e-9):.2f}×")

    if asymmetric:
        print("\n  ⚠  Asymmetric links detected (>25 % difference):")
        for s, d, fwd, rev, ratio in asymmetric:
            print(f"     GPU {s}↔GPU {d}  fwd={fwd:.2f}  rev={rev:.2f}  ratio={ratio:.2f}×")
    else:
        print("\n  ✓  No significant link asymmetry detected.")

    # Flag links below 5 GB/s as likely PCIe bottlenecks
    bottlenecks = [(s, d, bw) for bw, s, d in pairs if bw < 5.0]
    if bottlenecks:
        print("\n  ⚠  Potential PCIe bottlenecks (< 5 GB/s):")
        for s, d, bw in bottlenecks:
            print(f"     GPU {s} → GPU {d}  {bw:.2f} GB/s")
    print("────────────────────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# NVLink / P2P access map
# ---------------------------------------------------------------------------

def print_p2p_access_map(n_gpus: int) -> None:
    print("\n── Peer-to-Peer Access Map ──────────────────────────────────────────")
    header = f"  {'src \\ dst':>10}"
    for dst in range(n_gpus):
        header += f"  {'GPU ' + str(dst):>6}"
    print(header)
    for src in range(n_gpus):
        row = f"  {'GPU ' + str(src):>10}"
        for dst in range(n_gpus):
            if src == dst:
                row += f"  {'self':>6}"
            else:
                can = torch.cuda.can_device_access_peer(src, dst)
                row += f"  {'yes' if can else 'no':>6}"
        print(row)
    print("  (yes = NVLink or P2P-capable PCIe path present)")
    print("────────────────────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure GPU-to-GPU D2D copy bandwidth for all pairs."
    )
    parser.add_argument(
        "--size-mb",
        type=float,
        default=512.0,
        help="Transfer size in MB (default: 512)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warm-up iterations (default: 3)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        help="Number of timed iterations per pair (default: 10)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA-capable GPUs found.", file=sys.stderr)
        sys.exit(1)

    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        print(
            f"WARNING: Only {n_gpus} GPU(s) found.  "
            "At least 2 GPUs are required for a D2D bandwidth test.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("═" * 68)
    print("  PCIe / NVLink D2D Bandwidth Test")
    print("═" * 68)
    print(f"  GPUs detected : {n_gpus}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"    GPU {i} : {props.name}  ({props.total_memory // (1024**3)} GiB)")
    print(f"  Transfer size : {args.size_mb:.0f} MB")
    print(f"  Warm-up iters : {args.warmup}")
    print(f"  Timed iters   : {args.iters}")
    print("═" * 68)

    # Enable P2P access where possible
    for src in range(n_gpus):
        for dst in range(n_gpus):
            if src != dst and torch.cuda.can_device_access_peer(src, dst):
                torch.cuda.device(src)
    # torch.cuda.enable_peer_access is per-context; copy_ handles it automatically

    print_p2p_access_map(n_gpus)

    print(f"\nRunning {n_gpus * (n_gpus - 1)} directional transfers …\n")
    matrix = run_bandwidth_matrix(n_gpus, args.size_mb, args.warmup, args.iters)

    print("\n" + "═" * 68)
    print("  D2D Bandwidth Matrix (GB/s)")
    print("═" * 68)
    print_matrix(matrix, n_gpus, args.size_mb)
    print_analysis(matrix, n_gpus)


if __name__ == "__main__":
    main()
