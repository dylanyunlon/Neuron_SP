# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
profile_memory.py — GPU memory estimator for LLM training (no DeepSpeed required).

Estimates per-GPU VRAM breakdown for a given model config:
  - Parameters        (bf16, 2 bytes/param)
  - Optimizer states  (AdamW fp32: 8 bytes/param — m + v)
  - Gradients         (fp32, 4 bytes/param)
  - Activations       (per layer, per token, scales with batch × seq_len)

Then sweeps batch sizes to find the maximum that fits each GPU.

Usage:
    python tools/profile_memory.py                        # default: 7b, A100-40GB
    python tools/profile_memory.py --model 1b
    python tools/profile_memory.py --model 7b --gpu-mem 80
    python tools/profile_memory.py --model 7b --gpu-mem 80 --tp 2
    python tools/profile_memory.py --model 7b --seq-len 4096 --gpu-mem 40
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Model configs  (must stay in sync with models/llama_pretrain.py)
# ---------------------------------------------------------------------------

@dataclass
class LlamaConfig:
    name: str
    vocab_size: int
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    ffn_hidden: int
    max_seq_len: int


CONFIGS = {
    "70m": LlamaConfig(
        name="70m", vocab_size=32000, dim=512, n_layers=8,
        n_heads=8, n_kv_heads=4, ffn_hidden=1376, max_seq_len=2048,
    ),
    "1b": LlamaConfig(
        name="1b", vocab_size=32000, dim=2048, n_layers=16,
        n_heads=32, n_kv_heads=8, ffn_hidden=8192, max_seq_len=2048,
    ),
    "7b": LlamaConfig(
        name="7b", vocab_size=32000, dim=4096, n_layers=32,
        n_heads=32, n_kv_heads=8, ffn_hidden=11008, max_seq_len=4096,
    ),
}

# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_params(cfg: LlamaConfig) -> dict:
    """
    Break down parameter count by component.
    Returns a dict with component → param count (integers).
    """
    d = cfg.dim
    h = cfg.n_heads
    kv = cfg.n_kv_heads
    head_dim = d // h
    ffn = cfg.ffn_hidden
    V = cfg.vocab_size
    L = cfg.n_layers

    # Per-layer counts
    # Attention: Q, K, V, O projections
    q_proj   = d * d                    # (dim, dim)
    k_proj   = d * (kv * head_dim)      # (dim, kv_heads * head_dim)
    v_proj   = d * (kv * head_dim)
    o_proj   = d * d
    attn     = q_proj + k_proj + v_proj + o_proj

    # FFN (SwiGLU): gate, up, down — three matrices
    gate_proj = d * ffn
    up_proj   = d * ffn
    down_proj = ffn * d
    mlp       = gate_proj + up_proj + down_proj

    # RMSNorm: one weight vector per norm, 2 norms per layer + 1 final
    norm_per_layer = d * 2   # pre-attn norm + pre-ffn norm
    norm_final     = d

    per_layer = attn + mlp + norm_per_layer
    total_layers = per_layer * L

    # Embeddings + LM head (weight-tied in Llama-2, but count once each)
    embed  = V * d
    lm_head = V * d   # separate if not tied; conservative estimate

    total = total_layers + norm_final + embed + lm_head

    return {
        "attention_per_layer": attn,
        "mlp_per_layer": mlp,
        "norm_per_layer": norm_per_layer,
        "layers": L,
        "embed + lm_head": embed + lm_head,
        "norm_final": norm_final,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Memory estimation helpers (returns bytes)
# ---------------------------------------------------------------------------

BYTES = {
    "bf16": 2,
    "fp16": 2,
    "fp32": 4,
}


def params_memory(n_params: int) -> int:
    """bf16 parameter storage."""
    return n_params * BYTES["bf16"]


def optimizer_memory(n_params: int) -> int:
    """
    AdamW fp32 optimizer states: master weights + first moment + second moment.
    = 3 × fp32 × n_params  (master copy counts as an optimizer state)
    Some implementations store only m+v (2×) if master weights live elsewhere;
    we use the conservative 3× figure that matches mixed-precision training.
    """
    return n_params * BYTES["fp32"] * 3


def gradient_memory(n_params: int) -> int:
    """fp32 gradient accumulation buffer."""
    return n_params * BYTES["fp32"]


def activation_memory_per_layer(cfg: LlamaConfig, batch_size: int, seq_len: int) -> int:
    """
    Activation memory for ONE transformer layer without activation checkpointing.

    Stored tensors (needed for backward pass):
      - Input to layer          : B × T × d  (fp32)
      - Attention scores QK^T  : B × H × T × T (fp32)
      - Softmax output          : B × H × T × T (fp32)
      - V output                : B × H × T × head_dim (fp32)
      - Post-attention residual : B × T × d
      - FFN gate/up activations : B × T × ffn_hidden × 2 (SwiGLU stores both)

    Uses fp32 for all activations (conservative; bf16 would halve this).
    """
    B, T, d = batch_size, seq_len, cfg.dim
    H = cfg.n_heads
    head_dim = d // H
    ffn = cfg.ffn_hidden
    fp = BYTES["fp32"]

    layer_input     = B * T * d * fp
    attn_scores_qk  = B * H * T * T * fp       # QK^T before softmax
    attn_weights    = B * H * T * T * fp       # softmax output
    attn_v_out      = B * H * T * head_dim * fp
    post_attn       = B * T * d * fp
    ffn_gate_up     = B * T * ffn * 2 * fp     # gate + up (SwiGLU)

    return layer_input + attn_scores_qk + attn_weights + attn_v_out + post_attn + ffn_gate_up


def total_activation_memory(cfg: LlamaConfig, batch_size: int, seq_len: int) -> int:
    """Total activation memory across all layers."""
    per_layer = activation_memory_per_layer(cfg, batch_size, seq_len)
    # Also count embedding output and final norm output
    extra = batch_size * seq_len * cfg.dim * BYTES["fp32"] * 2
    return per_layer * cfg.n_layers + extra


# ---------------------------------------------------------------------------
# CUDA context overhead (empirically observed, rough constant)
# ---------------------------------------------------------------------------
CUDA_OVERHEAD_BYTES = 800 * 1024 * 1024   # ~800 MB


def estimate_total_memory(
    cfg: LlamaConfig,
    batch_size: int,
    seq_len: int,
    tp_degree: int = 1,
) -> dict:
    """
    Estimate total GPU memory for ONE GPU when using tensor-parallelism of degree tp_degree.

    With TP, parameters and optimizer states are sharded across tp_degree GPUs.
    Activations are replicated (each GPU processes the full batch per layer shard).
    """
    param_info = count_params(cfg)
    n_params = param_info["total"]

    # Sharded by TP
    n_params_local = n_params // tp_degree

    mem_params     = params_memory(n_params_local)
    mem_optim      = optimizer_memory(n_params_local)
    mem_grads      = gradient_memory(n_params_local)
    mem_activations = total_activation_memory(cfg, batch_size, seq_len)
    mem_cuda       = CUDA_OVERHEAD_BYTES

    total = mem_params + mem_optim + mem_grads + mem_activations + mem_cuda

    return {
        "n_params_total": n_params,
        "n_params_local": n_params_local,
        "params_bf16_GiB":    mem_params      / 2**30,
        "optimizer_fp32_GiB": mem_optim       / 2**30,
        "gradients_fp32_GiB": mem_grads       / 2**30,
        "activations_GiB":    mem_activations / 2**30,
        "cuda_overhead_GiB":  mem_cuda        / 2**30,
        "total_GiB":          total           / 2**30,
        "total_bytes":        total,
    }


# ---------------------------------------------------------------------------
# Max batch size search
# ---------------------------------------------------------------------------

def find_max_batch_size(
    cfg: LlamaConfig,
    gpu_mem_gib: float,
    seq_len: int,
    tp_degree: int = 1,
) -> Optional[int]:
    """
    Binary-search the largest batch_size that fits within gpu_mem_gib.
    Returns None if even batch_size=1 doesn't fit.
    """
    gpu_bytes = gpu_mem_gib * 2**30

    # Quick check: does batch=1 fit?
    if estimate_total_memory(cfg, 1, seq_len, tp_degree)["total_bytes"] > gpu_bytes:
        return None

    lo, hi = 1, 256
    # Expand upper bound until it doesn't fit
    while estimate_total_memory(cfg, hi, seq_len, tp_degree)["total_bytes"] <= gpu_bytes:
        hi *= 2
        if hi > 4096:
            break

    # Binary search
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if estimate_total_memory(cfg, mid, seq_len, tp_degree)["total_bytes"] <= gpu_bytes:
            lo = mid
        else:
            hi = mid

    return lo


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

GIB = 2**30


def fmt_gib(b: float) -> str:
    return f"{b:.2f} GiB"


def fmt_params(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.0f}M"
    return str(n)


def print_summary(
    cfg: LlamaConfig,
    gpu_mem_gib: float,
    seq_len: int,
    tp_degree: int,
    batch_size: int,
) -> None:
    est = estimate_total_memory(cfg, batch_size, seq_len, tp_degree)
    fits = est["total_bytes"] <= gpu_mem_gib * GIB

    print("=" * 60)
    print(f"  Model      : Llama-{cfg.name.upper()}")
    print(f"  Params     : {fmt_params(est['n_params_total'])} total  "
          f"({fmt_params(est['n_params_local'])} / GPU with TP={tp_degree})")
    print(f"  GPU memory : {gpu_mem_gib} GiB")
    print(f"  Seq len    : {seq_len}")
    print(f"  Batch size : {batch_size}")
    print("=" * 60)
    print(f"  {'Component':<28} {'Memory':>10}")
    print(f"  {'-'*28} {'-'*10}")
    print(f"  {'Params (bf16)':<28} {fmt_gib(est['params_bf16_GiB']):>10}")
    print(f"  {'Optimizer states (fp32)':<28} {fmt_gib(est['optimizer_fp32_GiB']):>10}")
    print(f"  {'Gradients (fp32)':<28} {fmt_gib(est['gradients_fp32_GiB']):>10}")
    print(f"  {'Activations (fp32)':<28} {fmt_gib(est['activations_GiB']):>10}")
    print(f"  {'CUDA overhead':<28} {fmt_gib(est['cuda_overhead_GiB']):>10}")
    print(f"  {'─'*28} {'─'*10}")
    print(f"  {'TOTAL':<28} {fmt_gib(est['total_GiB']):>10}  "
          f"{'✓ fits' if fits else '✗ OOM'}")
    print(f"  {'Available':<28} {fmt_gib(gpu_mem_gib):>10}")
    print(f"  {'Headroom':<28} {fmt_gib(gpu_mem_gib - est['total_GiB']):>10}")
    print()


def print_batch_sweep(
    cfg: LlamaConfig,
    gpu_mem_gib: float,
    seq_len: int,
    tp_degree: int,
) -> None:
    print(f"  Batch size sweep  (GPU={gpu_mem_gib} GiB, seq_len={seq_len}, TP={tp_degree})")
    print(f"  {'Batch':>6}  {'Activations':>12}  {'Total':>10}  {'Status':>8}")
    print(f"  {'─'*6}  {'─'*12}  {'─'*10}  {'─'*8}")

    for bs in [1, 2, 4, 8, 16, 32, 64]:
        est = estimate_total_memory(cfg, bs, seq_len, tp_degree)
        fits = est["total_bytes"] <= gpu_mem_gib * GIB
        status = "✓" if fits else "✗ OOM"
        print(f"  {bs:>6}  {fmt_gib(est['activations_GiB']):>12}  "
              f"{fmt_gib(est['total_GiB']):>10}  {status:>8}")

    max_bs = find_max_batch_size(cfg, gpu_mem_gib, seq_len, tp_degree)
    print()
    if max_bs is None:
        print(f"  ➜  Even batch_size=1 does NOT fit on {gpu_mem_gib} GiB GPU.")
    else:
        print(f"  ➜  Max batch_size per GPU: {max_bs}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate GPU memory for LLM training without DeepSpeed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", choices=list(CONFIGS.keys()), default="7b",
                   help="Model size to profile.")
    p.add_argument("--gpu-mem", type=float, default=40.0,
                   help="GPU memory in GiB (e.g. 40 for A100-40GB, 80 for A100-80GB).")
    p.add_argument("--seq-len", type=int, default=2048,
                   help="Sequence length.")
    p.add_argument("--tp", type=int, default=1,
                   help="Tensor parallelism degree (parameters sharded across TP GPUs).")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Batch size to estimate for (default: sweep 1..64).")
    p.add_argument("--all-models", action="store_true",
                   help="Profile all model sizes.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    models_to_run = list(CONFIGS.keys()) if args.all_models else [args.model]

    for model_key in models_to_run:
        cfg = CONFIGS[model_key]

        if args.batch_size is not None:
            print_summary(cfg, args.gpu_mem, args.seq_len, args.tp, args.batch_size)
        else:
            # Default: show batch=1 summary + sweep
            print_summary(cfg, args.gpu_mem, args.seq_len, args.tp, batch_size=1)
            print_batch_sweep(cfg, args.gpu_mem, args.seq_len, args.tp)


if __name__ == "__main__":
    main()
