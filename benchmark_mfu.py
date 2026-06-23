# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
benchmark_mfu.py — Pure-PyTorch MFU benchmark (no DeepSpeed dependency).

Builds a ~70 M-parameter GPT-style transformer, runs 100 training steps, and
reports Model FLOP Utilisation (MFU) relative to the theoretical peak TFLOPS
of the detected GPU.

Supported GPU peak TFLOPS (BF16/FP16 Tensor Core):
    A6000  —  38.7 TFLOPS
    H100   — 835.0 TFLOPS

TFLOPs per forward+backward step are estimated with the standard formula:
    FLOPs ≈ 6 * N * T
where N = number of model parameters and T = number of tokens per batch.
(The factor 6 accounts for multiply-add in forward pass × 2 and the × 3
forward/backward ratio under the PaLM / Chinchilla convention.)

Usage:
    python benchmark_mfu.py [--steps 100] [--batch-size 8] [--seq-len 1024] \
                            [--warmup 10] [--output results_mfu.json]

Output: JSON written to --output (default: results_mfu.json).
"""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# GPU peak TFLOPS registry  (BF16 / FP16 Tensor Core, SXM/PCIe variants)
# ---------------------------------------------------------------------------

_PEAK_TFLOPS: dict = {
    # key substring matched against torch.cuda.get_device_name()
    "H100": 835.0,
    "H800": 835.0,
    "A100": 312.0,
    "A6000": 38.7,
    "A40": 37.4,
    "A30": 165.0,
    "3090": 35.6,
    "4090": 82.6,
    "V100": 112.0,
}

_FALLBACK_PEAK_TFLOPS = 38.7  # conservative fallback (A6000)


def _get_peak_tflops(device: torch.device) -> tuple[float, str]:
    """Return (peak_tflops, gpu_name) for the given CUDA device."""
    name = torch.cuda.get_device_name(device)
    for key, peak in _PEAK_TFLOPS.items():
        if key in name:
            return peak, name
    # Unknown GPU — return fallback but flag it
    return _FALLBACK_PEAK_TFLOPS, name


# ---------------------------------------------------------------------------
# 70 M-parameter GPT-style model
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    # ffn hidden = 4 * n_embd  (standard GPT-2 ratio)
    dropout: float = 0.0
    bias: bool = True
    seq_len: int = 1024


class CausalSelfAttention(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        # fused QKV projection
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape to (B, n_head, T, head_dim)
        def _split_heads(t):
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k, v = _split_heads(q), _split_heads(k), _split_heads(v)

        # Scaled dot-product attention (uses Flash Attention when available)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0,
                                           is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = 4 * cfg.n_embd
        self.c_fc = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.c_proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))


class Block(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT70M(nn.Module):
    """GPT-style transformer targeting ~70 M parameters.

    Default config (n_layer=12, n_head=12, n_embd=768) produces:
        embedding:  50257 * 768 ≈ 38.6 M
        blocks:     12 * (4*768^2 + 2*768*3072) ≈ 28.3 M
        total:                                   ≈ 66.9 M  (~70 M)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
                wpe=nn.Embedding(cfg.seq_len, cfg.n_embd),
                drop=nn.Dropout(cfg.dropout),
                h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
                ln_f=nn.LayerNorm(cfg.n_embd),
            ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# FLOPs estimation
# ---------------------------------------------------------------------------

def estimate_flops_per_step(num_params: int, tokens_per_step: int) -> float:
    """Estimate FLOPs for one forward+backward step.

    Formula: FLOPs ≈ 6 * N * T
        - factor 2: multiply-add counts as 2 FLOPs
        - factor 3: backward ≈ 2× forward  → total ≈ 3× forward
        → combined factor = 6
    This matches the PaLM / Chinchilla / nanoGPT convention.
    """
    return 6 * num_params * tokens_per_step


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    steps: int = 100,
    batch_size: int = 8,
    seq_len: int = 1024,
    warmup: int = 10,
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Run the MFU benchmark and return a result dictionary."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MFU benchmarking.")

    device = torch.device("cuda", 0)
    peak_tflops, gpu_name = _get_peak_tflops(device)

    cfg = ModelConfig(seq_len=seq_len)
    model = GPT70M(cfg).to(device=device, dtype=dtype)
    num_params = model.num_parameters()

    tokens_per_step = batch_size * seq_len
    flops_per_step = estimate_flops_per_step(num_params, tokens_per_step)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Warm-up — not timed
    model.train()
    for _ in range(warmup):
        idx = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)
        with torch.amp.autocast("cuda", dtype=dtype):
            logits = model(idx)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize(device)

    # Timed measurement
    step_times: list[float] = []
    t_total_start = time.perf_counter()

    for _ in range(steps):
        idx = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)

        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=dtype):
            logits = model(idx)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - t0)

    t_total = time.perf_counter() - t_total_start

    avg_step_s = sum(step_times) / len(step_times)
    min_step_s = min(step_times)
    max_step_s = max(step_times)

    # Actual TFLOPS = FLOPs per step / time per step
    actual_tflops = (flops_per_step / avg_step_s) / 1e12
    mfu = actual_tflops / peak_tflops

    # Tokens per second
    tokens_per_sec = tokens_per_step / avg_step_s

    result = {
        "gpu_name": gpu_name,
        "peak_tflops": peak_tflops,
        "dtype": str(dtype).replace("torch.", ""),
        "model": {
            "type": "GPT-70M",
            "num_parameters": num_params,
            "num_layers": cfg.n_layer,
            "n_head": cfg.n_head,
            "n_embd": cfg.n_embd,
            "vocab_size": cfg.vocab_size,
        },
        "run": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "tokens_per_step": tokens_per_step,
            "warmup_steps": warmup,
            "measured_steps": steps,
        },
        "results": {
            "actual_tflops": round(actual_tflops, 4),
            "mfu": round(mfu, 6),
            "mfu_pct": round(mfu * 100, 4),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "avg_step_ms": round(avg_step_s * 1e3, 3),
            "min_step_ms": round(min_step_s * 1e3, 3),
            "max_step_ms": round(max_step_s * 1e3, 3),
            "total_time_s": round(t_total, 3),
        },
    }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pure-PyTorch MFU benchmark (70 M GPT).")
    p.add_argument("--steps", type=int, default=100, help="Number of timed training steps (default: 100)")
    p.add_argument("--warmup", type=int, default=10, help="Number of warm-up steps before timing (default: 10)")
    p.add_argument("--batch-size", type=int, default=8, help="Micro-batch size (default: 8)")
    p.add_argument("--seq-len", type=int, default=1024, help="Sequence length (default: 1024)")
    p.add_argument("--output", type=str, default="results_mfu.json",
                   help="Path for JSON output (default: results_mfu.json)")
    p.add_argument("--fp16", action="store_true", help="Use FP16 instead of BF16")
    return p.parse_args()


def main():
    args = _parse_args()
    dtype = torch.float16 if args.fp16 else torch.bfloat16

    print(f"[benchmark_mfu] steps={args.steps}  warmup={args.warmup}  "
          f"batch={args.batch_size}  seq_len={args.seq_len}  dtype={dtype}")

    result = run_benchmark(
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        warmup=args.warmup,
        dtype=dtype,
    )

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\n[benchmark_mfu] Results written to {args.output}")
    print(f"[benchmark_mfu] MFU = {result['results']['mfu_pct']:.2f}%  "
          f"({result['results']['actual_tflops']:.2f} / {result['peak_tflops']} TFLOPS  "
          f"on {result['gpu_name']})")


if __name__ == "__main__":
    main()
