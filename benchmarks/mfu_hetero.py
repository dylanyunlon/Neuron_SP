# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
benchmarks/mfu_hetero.py — MFU benchmark for heterogeneous GPU cluster

Measures Model FLOPs Utilization (MFU) for a 3B Llama-style model across
four configurations:
  1. H100 only        (1 GPU, single-process)
  2. A6000 only       (1 GPU, single-process)
  3. DES-LOC hetero   (H100 + 2×A6000, 3-process torchrun)
  4. Naive DDP        (H100 + 2×A6000, 3-process torchrun)

MFU formula (Chinchilla / PaLM convention):
    MFU = (6 * N_params * T_tokens_per_step * world_size) / (step_latency * peak_flops)
    where 6 accounts for fwd + bwd multiply-adds (×2 per fwd FLOP)

Theoretical peak BF16 TFLOPS:
    H100 NVL:  835 TFLOPS
    A6000:      38.7 TFLOPS

Usage (single-process per-GPU benchmarks):
    python benchmarks/mfu_hetero.py --gpu h100 --steps 50
    python benchmarks/mfu_hetero.py --gpu a6000 --steps 50

Usage (multi-GPU, launched via torchrun):
    torchrun --nproc_per_node=3 benchmarks/mfu_hetero.py --mode ddp --steps 50
    torchrun --nproc_per_node=3 benchmarks/mfu_hetero.py --mode desloc --steps 50

Output: markdown table printed to stdout + JSON saved to benchmarks/mfu_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

# ---------------------------------------------------------------------------
# 3B Llama-style model (identical to run_pretrain.py's LlamaModel)
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class _SwiGLUMLP(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        intermediate = int(hidden * 8 / 3)
        intermediate = (intermediate + 63) // 64 * 64
        self.gate = nn.Linear(hidden, intermediate, bias=False)
        self.up   = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class _CausalAttn(nn.Module):
    def __init__(self, hidden: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.qkv  = nn.Linear(hidden, 3 * hidden, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).contiguous().reshape(B, T, C))


class _TransformerBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int) -> None:
        super().__init__()
        self.norm1 = _RMSNorm(hidden)
        self.attn  = _CausalAttn(hidden, n_heads)
        self.norm2 = _RMSNorm(hidden)
        self.mlp   = _SwiGLUMLP(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LlamaModel3B(nn.Module):
    """3B Llama-style causal LM (hidden=3200, layers=26, heads=32)."""
    HIDDEN   = 3200
    N_LAYERS = 26
    N_HEADS  = 32
    VOCAB    = 32000
    SEQ_LEN  = 1024  # reduced from 2048 for A6000 VRAM budget

    def __init__(self) -> None:
        super().__init__()
        H = self.HIDDEN
        self.embedding     = nn.Embedding(self.VOCAB, H)
        self.pos_embedding = nn.Embedding(self.SEQ_LEN, H)
        self.layers = nn.ModuleList(
            [_TransformerBlock(H, self.N_HEADS) for _ in range(self.N_LAYERS)]
        )
        self.norm    = _RMSNorm(H)
        self.lm_head = nn.Linear(H, self.VOCAB, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x    = self.embedding(input_ids) + self.pos_embedding(pos)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Theoretical peak FLOPS per GPU tier
# ---------------------------------------------------------------------------
# Source: NVIDIA datasheets — BF16 tensor-core TFLOPS (no sparsity)
_PEAK_TFLOPS: Dict[str, float] = {
    "h100":   835.0,
    "a6000":   38.7,
    "blackwell": 1800.0,  # RTX PRO 6000 Blackwell estimate; not yet published
    # Heterogeneous effective peak = weighted harmonic mean of per-tier peaks.
    # For H100 + 2×A6000: harmonic mean weighted by grad_accum ratio.
    # H100 handles 22× more micro-batches → contributes ~73% of throughput.
    "h100_2xa6000": 835.0 * 0.73 + 38.7 * 0.27,  # ≈ 620 TFLOPS effective
}


def effective_peak_tflops(gpu_label: str) -> float:
    """Return theoretical peak BF16 TFLOPS for the given GPU label."""
    return _PEAK_TFLOPS.get(gpu_label, 100.0)


# ---------------------------------------------------------------------------
# MFU calculation
# ---------------------------------------------------------------------------

def compute_mfu(
    n_params: int,
    tokens_per_step: int,
    world_size: int,
    step_latency_s: float,
    peak_tflops: float,
) -> float:
    """
    MFU = (6 * N * T_global) / (latency * peak_flops)

    Args:
        n_params: model parameter count
        tokens_per_step: tokens processed per step per rank (batch * seq_len)
        world_size: number of GPUs (data-parallel factor)
        step_latency_s: wall-clock seconds per training step
        peak_tflops: theoretical peak FLOPs/s for the cluster
    """
    total_tokens = tokens_per_step * world_size
    achieved_flops = 6 * n_params * total_tokens  # multiply-add counts 2 FLOPs
    peak_flops = peak_tflops * 1e12 * step_latency_s
    return achieved_flops / max(peak_flops, 1.0)


# ---------------------------------------------------------------------------
# Single-GPU benchmark (H100 or A6000)
# ---------------------------------------------------------------------------

def bench_single_gpu(
    gpu_label: str,
    device_idx: int,
    batch_size: int,
    n_steps: int,
) -> Dict:
    """Run n_steps of 3B model on a single GPU, return metrics dict."""
    device = torch.device(f"cuda:{device_idx}")
    dtype  = torch.bfloat16

    model = LlamaModel3B().to(dtype=dtype, device=device)
    n_params = model.num_parameters
    optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    seq_len    = LlamaModel3B.SEQ_LEN
    vocab_size = LlamaModel3B.VOCAB

    # Warmup — 3 steps to prime CUDA caches
    for _ in range(3):
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len + 1), device=device)
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype):
            logits = model(inp)
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, V),
                tgt[:, :T - 1].reshape(-1),
            )
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    latencies: List[float] = []
    for _ in range(n_steps):
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len + 1), device=device)
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=dtype):
            logits = model(inp)
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, V),
                tgt[:, :T - 1].reshape(-1),
            )
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize(device)
        latencies.append(time.perf_counter() - t0)

    median_lat = sorted(latencies)[len(latencies) // 2]
    tokens_per_step = batch_size * seq_len
    tok_per_sec = tokens_per_step / median_lat
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1 << 30)
    peak_tflops = effective_peak_tflops(gpu_label)
    mfu = compute_mfu(n_params, tokens_per_step, 1, median_lat, peak_tflops)

    return {
        "config":          f"{gpu_label.upper()} (1 GPU, single-process)",
        "gpu_label":       gpu_label,
        "world_size":      1,
        "batch_size":      batch_size,
        "seq_len":         seq_len,
        "n_params_M":      n_params / 1e6,
        "median_step_ms":  median_lat * 1000,
        "tokens_per_sec":  tok_per_sec,
        "mfu_pct":         mfu * 100,
        "peak_mem_gb":     peak_mem_gb,
        "peak_tflops_ref": peak_tflops,
    }


# ---------------------------------------------------------------------------
# Multi-GPU benchmark (DDP or DES-LOC stub)
# ---------------------------------------------------------------------------

def bench_multi_gpu(
    mode: str,
    batch_size: int,
    n_steps: int,
) -> Dict:
    """
    Run n_steps of 3B model in multi-GPU mode (DDP or DES-LOC).

    Called via torchrun; each rank runs independently.
    DES-LOC mode uses proportional gradient accumulation:
      H100 (rank 2): 22 micro-batches per step
      A6000 (rank 0,1): 1 micro-batch per step
    """
    rank       = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if not dist.is_initialized():
        import datetime as _dt  # noqa: PLC0415
        dist.init_process_group(
            "nccl", init_method="env://",
            timeout=_dt.timedelta(minutes=10),
        )

    device = torch.device(f"cuda:{local_rank}")
    dtype  = torch.bfloat16
    torch.cuda.set_device(local_rank)

    model = LlamaModel3B().to(dtype=dtype, device=device)
    n_params = model.num_parameters

    if mode == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: PLC0415
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        raw_model = model.module
        grad_accum = 1
    else:
        # DES-LOC: H100 gets 22×, A6000 gets 1× (proportional to BF16 throughput)
        raw_model = model
        # Approximate H100 as rank 2 (highest VRAM) for this benchmark
        _props = torch.cuda.get_device_properties(local_rank)
        if _props.total_memory > 80 * (1 << 30):
            grad_accum = 22  # H100 NVL tier
        else:
            grad_accum = 1   # A6000 tier

    optimizer = AdamW(raw_model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    seq_len    = LlamaModel3B.SEQ_LEN
    vocab_size = LlamaModel3B.VOCAB

    # Warmup
    for _ in range(3):
        for _ in range(grad_accum):
            tokens = torch.randint(0, vocab_size, (batch_size, seq_len + 1), device=device)
            inp, tgt = tokens[:, :-1], tokens[:, 1:]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype):
                logits = raw_model(inp)
                B, T, V = logits.shape
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, V),
                    tgt[:, :T - 1].reshape(-1),
                ) / grad_accum
            loss.backward()
        clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()

    dist.barrier()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    latencies: List[float] = []
    for _ in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        for _mb in range(grad_accum):
            tokens = torch.randint(0, vocab_size, (batch_size, seq_len + 1), device=device)
            inp, tgt = tokens[:, :-1], tokens[:, 1:]
            with torch.autocast("cuda", dtype=dtype):
                logits = raw_model(inp)
                B, T, V = logits.shape
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, V),
                    tgt[:, :T - 1].reshape(-1),
                ) / grad_accum
            loss.backward()
        clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()
        dist.barrier()
        torch.cuda.synchronize(device)
        latencies.append(time.perf_counter() - t0)

    # Gather median latency across ranks — report worst (bottleneck)
    median_lat_local = sorted(latencies)[len(latencies) // 2]
    lat_t = torch.tensor(median_lat_local, device=device)
    dist.all_reduce(lat_t, op=dist.ReduceOp.MAX)
    median_lat = lat_t.item()

    tokens_per_step = batch_size * seq_len * grad_accum
    tok_per_sec = tokens_per_step * world_size / median_lat
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1 << 30)

    # Gather max peak memory
    mem_t = torch.tensor(peak_mem_gb, device=device)
    dist.all_reduce(mem_t, op=dist.ReduceOp.MAX)
    max_peak_mem_gb = mem_t.item()

    peak_tflops = effective_peak_tflops("h100_2xa6000")
    mfu = compute_mfu(
        n_params,
        batch_size * seq_len * grad_accum,
        world_size,
        median_lat,
        peak_tflops,
    )

    mode_label = "DES-LOC (proportional grad_accum)" if mode == "desloc" else "Naive DDP (equal batch)"
    result = {
        "config":          f"H100 + 2×A6000  ({mode_label})",
        "mode":            mode,
        "world_size":      world_size,
        "batch_size":      batch_size,
        "grad_accum":      grad_accum,
        "seq_len":         seq_len,
        "n_params_M":      n_params / 1e6,
        "median_step_ms":  median_lat * 1000,
        "tokens_per_sec":  tok_per_sec,
        "mfu_pct":         mfu * 100,
        "peak_mem_max_gb": max_peak_mem_gb,
        "peak_tflops_ref": peak_tflops,
    }

    if dist.get_rank() == 0:
        return result
    return {}


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _fmt_table(results: List[Dict]) -> str:
    """Render a markdown table from a list of result dicts."""
    rows = []
    for r in results:
        rows.append({
            "Config":         r.get("config", "—"),
            "Params (M)":     f"{r.get('n_params_M', 0):.0f}",
            "Step (ms)":      f"{r.get('median_step_ms', 0):.1f}",
            "Tok/s":          f"{r.get('tokens_per_sec', 0):,.0f}",
            "MFU (%)":        f"{r.get('mfu_pct', 0):.2f}",
            "Peak Mem (GB)":  f"{r.get('peak_mem_gb') or r.get('peak_mem_max_gb') or 0:.1f}",
        })

    if not rows:
        return "(no results)"

    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(r[c]) for r in rows)) for c in cols}
    sep  = "| " + " | ".join("-" * widths[c] for c in cols) + " |"
    hdr  = "| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |"
    lines = [hdr, sep]
    for r in rows:
        lines.append("| " + " | ".join(r[c].ljust(widths[c]) for c in cols) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MFU benchmark for heterogeneous GPU cluster (3B Llama)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--gpu",
        choices=["h100", "a6000"],
        default=None,
        help="Single-GPU benchmark target (single-process mode).",
    )
    p.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index for single-GPU benchmark.",
    )
    p.add_argument(
        "--mode",
        choices=["ddp", "desloc"],
        default=None,
        help="Multi-GPU benchmark mode (requires torchrun --nproc_per_node=3).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Micro-batch size per GPU per step.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of benchmark steps (after 3-step warmup).",
    )
    p.add_argument(
        "--output",
        default="benchmarks/mfu_results.json",
        help="Path to write JSON results.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    results: List[Dict] = []

    if args.gpu is not None:
        # Single-GPU benchmark
        print(f"Benchmarking {args.gpu.upper()} (device={args.device}, steps={args.steps})…")
        r = bench_single_gpu(
            gpu_label  = args.gpu,
            device_idx = args.device,
            batch_size = args.batch_size,
            n_steps    = args.steps,
        )
        results.append(r)

    elif args.mode is not None:
        # Multi-GPU benchmark — torchrun context
        r = bench_multi_gpu(
            mode       = args.mode,
            batch_size = args.batch_size,
            n_steps    = args.steps,
        )
        if r:  # rank 0 returns non-empty dict
            results.append(r)
        if dist.is_initialized():
            dist.destroy_process_group()

    else:
        print("Specify --gpu {h100,a6000} for single-GPU or --mode {ddp,desloc} for multi-GPU.")
        print("Example: python benchmarks/mfu_hetero.py --gpu h100 --steps 50")
        sys.exit(1)

    if not results:
        return

    # Print table
    print("\n## MFU Benchmark Results — 3B Llama-style model\n")
    print(_fmt_table(results))
    print()
    for r in results:
        print(f"  {r['config']}")
        print(f"    Params:     {r.get('n_params_M', 0):.0f}M")
        print(f"    Latency:    {r.get('median_step_ms', 0):.1f} ms/step")
        print(f"    Throughput: {r.get('tokens_per_sec', 0):,.0f} tok/s")
        print(f"    MFU:        {r.get('mfu_pct', 0):.2f}%  (ref={r.get('peak_tflops_ref', 0):.0f} TFLOPS)")
        mem = r.get("peak_mem_gb") or r.get("peak_mem_max_gb") or 0
        print(f"    Peak mem:   {mem:.1f} GB")
        print()

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
