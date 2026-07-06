# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team / Neuron_SP
"""
benchmarks/cross_entropy_tp_bench.py
    — Tensor-parallel cross-entropy loss benchmark

Benchmarks the cross_entropy_tp CUDA kernel
(csrc/hetero_reduce/cross_entropy_tp.cu) against unfused PyTorch baselines.

Kernel computes (per TP rank, for this rank's vocab shard of size V/tp_size):
    Phase 1 (forward):
        local_max      = max_j logit[row, j]
        local_sum_exp  = Σ_j exp(logit[row, j] - local_max)
        local_logit    = logit[row, label - shard_offset]  (or 0 if not in shard)
    Phase 2 (after AllReduce across TP ranks):
        loss[row] = log(global_sum_exp[row]) + global_max[row] - global_logit[row]
    Backward:
        d_logit[row, j] = (exp(logit[row,j] - max - log_sum) -
                           1{j + shard_offset == label[row]}) / batch_size

Memory model (forward phase, single TP rank):
    Reads  : logits [batch × v_local × 2 B]  (BF16)
    Writes : local_max, local_sum_exp, local_logit [3 × batch × 4 B]  (FP32)
    Effective bytes ≈ 2 × batch × v_local bytes  (output is negligible)

Configurations swept:
    batch sizes   : 1, 8, 64, 512
    total vocab   : 32K, 65K, 128K (Llama 3), 256K
    TP sizes      : 1, 2, 4, 8

Baselines compared:
    A) torch_naive  — torch.max + torch.exp + torch.sum (3 kernel passes)
    B) torch_logsumexp — single torch.logsumexp pass (2 passes internally)
    C) fused kernel — custom CUDA kernel (via hetero_reduce extension)

Launch:
    python benchmarks/cross_entropy_tp_bench.py [--device 0] [--iters 50]
        [--warmup 10] [--tp 8] [--batches 1 64 512]
        [--vocab 32768 65536 128000] [--json results_ce_tp.json]
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Tuple

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
# Forward implementations
# ---------------------------------------------------------------------------


def _torch_naive_ce_forward(
    logits: torch.Tensor,   # (batch, v_local) BF16
    labels: torch.Tensor,   # (batch,) int32, global indices
    shard_offset: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Baseline A: three separate PyTorch ops (max → exp → sum).

    Returns (local_max, local_sum_exp, local_logit) each shape (batch,) FP32.
    """
    lf = logits.float()
    lmax = lf.max(dim=-1).values                         # (batch,)
    se   = (lf - lmax.unsqueeze(-1)).exp_().sum(dim=-1)  # (batch,)

    # Label logit extraction (host-side gather)
    local_labels = labels - shard_offset
    in_shard     = (local_labels >= 0) & (local_labels < logits.shape[1])
    safe_labels  = local_labels.clamp(0, logits.shape[1] - 1)
    ll = torch.where(
        in_shard,
        lf[torch.arange(lf.shape[0], device=lf.device), safe_labels],
        torch.zeros_like(lmax),
    )
    return lmax, se, ll


def _torch_logsumexp_ce_forward(
    logits: torch.Tensor,
    labels: torch.Tensor,
    shard_offset: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Baseline B: torch.logsumexp — internally still two passes but fused in C++.
    Returns (local_max, local_sum_exp, local_logit).
    """
    lf   = logits.float()
    lse  = torch.logsumexp(lf, dim=-1)   # (batch,) = log(Σ exp(x))
    lmax = lf.max(dim=-1).values
    se   = (lse - lmax).exp()             # recover sum_exp from log representation

    local_labels = labels - shard_offset
    in_shard     = (local_labels >= 0) & (local_labels < logits.shape[1])
    safe_labels  = local_labels.clamp(0, logits.shape[1] - 1)
    ll = torch.where(
        in_shard,
        lf[torch.arange(lf.shape[0], device=lf.device), safe_labels],
        torch.zeros_like(lmax),
    )
    return lmax, se, ll


def _custom_ce_forward(
    logits: torch.Tensor,
    labels: torch.Tensor,
    shard_offset: int,
    sm_version: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Custom fused CUDA kernel wrapper."""
    if _HAS_EXT:
        return _ext.cross_entropy_tp_forward(logits, labels, shard_offset, sm_version)
    else:
        return _torch_logsumexp_ce_forward(logits, labels, shard_offset)


# ---------------------------------------------------------------------------
# Correctness check (single TP rank, tp_size=1)
# ---------------------------------------------------------------------------


def _check_correctness(device: int, sm_version: int) -> bool:
    """Compare fused kernel loss to torch.nn.functional.cross_entropy reference."""
    batch, v_local = 16, 512
    shard_offset   = 0
    dev = f"cuda:{device}"

    logits_bf = torch.randn(batch, v_local, dtype=torch.bfloat16, device=dev)
    labels    = torch.randint(0, v_local, (batch,), dtype=torch.int32, device=dev)

    # PyTorch reference (FP32 precision)
    loss_ref = F.cross_entropy(
        logits_bf.float(),
        labels.long(),
        reduction="none",
    )

    # Custom forward + loss
    lmax, lsum, ll = _custom_ce_forward(logits_bf, labels, shard_offset, sm_version)
    loss_custom = torch.log(lsum) + lmax - ll   # Phase-2 formula (tp_size=1)

    torch.cuda.synchronize()
    diff = (loss_ref - loss_custom).abs().max().item()
    if diff > 0.05:
        print(f"\n  max loss diff = {diff:.4f} (ref={loss_ref[:4]}, got={loss_custom[:4]})")
    return diff < 0.05


# ---------------------------------------------------------------------------
# Single-config benchmark
# ---------------------------------------------------------------------------


def _bench_config(
    harness:      BenchmarkHarness,
    batch:        int,
    v_local:      int,
    v_total:      int,
    shard_offset: int,
    device:       int,
    sm_version:   int,
) -> List[BenchResult]:
    dev = f"cuda:{device}"
    logits = torch.randn(batch, v_local, dtype=torch.bfloat16, device=dev)
    labels = torch.randint(0, v_total, (batch,), dtype=torch.int32, device=dev)

    # BW: read logits (BF16) only; output scalars are negligible
    bytes_accessed = batch * v_local * 2
    # FLOPs: exp per element + one add (online accumulator)
    flops = batch * v_local * 6   # exp≈4, add=1, compare=1

    tag = f"batch={batch:<5d} V_local={v_local:<8d} V_total={v_total}"

    # ── Baseline A: naive three ops ────────────────────────────────────────
    base_naive = harness.run(
        label=f"torch naive (max+exp+sum) | {tag}",
        fn=lambda: _torch_naive_ce_forward(logits, labels, shard_offset),
        bytes_accessed=bytes_accessed,
        flops=flops,
    )

    # ── Baseline B: torch.logsumexp ────────────────────────────────────────
    base_lse = harness.run(
        label=f"torch.logsumexp           | {tag}",
        fn=lambda: _torch_logsumexp_ce_forward(logits, labels, shard_offset),
        bytes_accessed=bytes_accessed,
        flops=flops,
    )

    # ── Custom fused kernel ─────────────────────────────────────────────────
    kernel_label = "cross_entropy_tp kernel" if _HAS_EXT else "cross_entropy_tp (sim)"
    custom = harness.run(
        label=f"{kernel_label:<26s} | {tag}",
        fn=lambda: _custom_ce_forward(logits, labels, shard_offset, sm_version),
        bytes_accessed=bytes_accessed,
        flops=flops,
        baseline_label=base_naive.label,
    )
    harness.compare_to_baseline(custom, base_naive)

    return [base_naive, base_lse, custom]


# ---------------------------------------------------------------------------
# Backward benchmark helper
# ---------------------------------------------------------------------------


def _bench_backward(
    harness:    BenchmarkHarness,
    batch:      int,
    v_local:    int,
    v_total:    int,
    device:     int,
    sm_version: int,
) -> BenchResult:
    dev = f"cuda:{device}"
    logits   = torch.randn(batch, v_local, dtype=torch.bfloat16, device=dev)
    d_logits = torch.empty_like(logits)
    labels   = torch.randint(0, v_total, (batch,), dtype=torch.int32, device=dev)
    gmax     = torch.zeros(batch, dtype=torch.float32, device=dev)
    lse      = torch.full((batch,), 7.0, dtype=torch.float32, device=dev)

    bytes_accessed = batch * v_local * 4   # read logits (2B) + write d_logits (2B)
    flops          = batch * v_local * 6

    tag = f"batch={batch:<5d} V_local={v_local}"

    if _HAS_EXT:
        return harness.run(
            label=f"cross_entropy_tp_backward | {tag}",
            fn=lambda: _ext.cross_entropy_tp_backward(
                d_logits, logits, labels, gmax, lse, 0, 1.0 / batch, sm_version),
            bytes_accessed=bytes_accessed,
            flops=flops,
        )
    else:
        # Simulate with autograd
        lf = logits.float().requires_grad_(True)
        def _sim_bwd():
            F.cross_entropy(lf, labels.long(), reduction="sum").backward()
        return harness.run(
            label=f"cross_entropy_tp_backward (sim) | {tag}",
            fn=_sim_bwd,
            bytes_accessed=bytes_accessed,
            flops=flops,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tensor-parallel cross-entropy benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device",  type=int,   default=0)
    parser.add_argument("--warmup",  type=int,   default=10)
    parser.add_argument("--iters",   type=int,   default=50)
    parser.add_argument("--tp",      type=int,   default=8,
                        help="Simulated TP size (v_local = vocab / tp)")
    parser.add_argument(
        "--batches", nargs="+", type=int,
        default=[1, 8, 64, 512],
    )
    parser.add_argument(
        "--vocab", nargs="+", type=int,
        default=[32768, 65536, 128000, 256000],
        help="Full vocabulary sizes (v_local = vocab / tp)",
    )
    parser.add_argument("--json", metavar="PATH", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA device available.")
        sys.exit(1)

    device_info = get_device_info(args.device)
    sm_version  = device_info["sm_version"]

    print_device_header(args.device)
    print(f"Simulated TP size: {args.tp}\n")

    if not _HAS_EXT:
        print(
            "WARNING: deepspeed.ops.hetero_reduce not found.\n"
            "         Custom kernel results use simulation.\n"
            "         Build with: cd csrc/hetero_reduce && python setup.py install\n"
        )
    else:
        print("Forward correctness check: ", end="", flush=True)
        ok = _check_correctness(args.device, sm_version)
        print("PASS ✓" if ok else "FAIL ✗ — results may be incorrect")
        print()

    harness = BenchmarkHarness(warmup=args.warmup, iters=args.iters, device=args.device)
    fwd_results:  List[BenchResult] = []
    bwd_results:  List[BenchResult] = []

    # ── Forward sweep ───────────────────────────────────────────────────────
    print(f"{'─'*70}")
    print("  FORWARD: cross_entropy_tp_forward")
    print(f"{'─'*70}")

    for v_total in args.vocab:
        v_local = v_total // args.tp
        if v_local < 8:
            continue

        print(f"\n  V_total={v_total:>7d}  V_local={v_local:>6d}")
        group: List[BenchResult] = []
        for batch in args.batches:
            shard_offset = 0  # Rank 0 for benchmark purposes
            results = _bench_config(
                harness, batch, v_local, v_total, shard_offset,
                args.device, sm_version)
            group.extend(results)

        print_results_table(group)
        fwd_results.extend(group)

    # ── Backward sweep ──────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  BACKWARD: cross_entropy_tp_backward")
    print(f"{'─'*70}")

    for v_total in args.vocab:
        v_local = v_total // args.tp
        if v_local < 8:
            continue
        print(f"\n  V_total={v_total:>7d}  V_local={v_local:>6d}")
        bwd_group: List[BenchResult] = []
        for batch in args.batches:
            r = _bench_backward(harness, batch, v_local, v_total,
                                args.device, sm_version)
            bwd_group.append(r)
        print_results_table(bwd_group)
        bwd_results.extend(bwd_group)

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  SUMMARY — cross_entropy_tp_forward speedup vs torch naive")
    print("="*70)
    custom_fwd = [r for r in fwd_results if "cross_entropy_tp kernel" in r.label
                  or "cross_entropy_tp (sim)" in r.label]
    if custom_fwd:
        print_results_table(custom_fwd)

    all_results = fwd_results + bwd_results
    if args.json:
        with open(args.json, "w") as f:
            f.write(print_json(all_results))
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
