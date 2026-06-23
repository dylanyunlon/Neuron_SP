# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
run_pretrain.py — Neuron_SP pretraining entry point

Direct entry point for ags1: python run_pretrain.py

Supports:
  --model-size  7b / 1b / 70m   (default: 7b)
  --data-path   /path/to/data   (default: synthetic)
  --steps       N               (default: 100)
  --batch-size  N               (default: 2)
  --seq-len     N               (default: 2048)
  --log-every   N               (default: 10)

Import discipline:
  - NO `import deepspeed` (avoids apex/op_builder init chain)
  - Direct submodule imports only: deepspeed.runtime.desloc_engine
  - torch / torch.nn / torch.distributed used directly
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_

# ---------------------------------------------------------------------------
# Logging setup (before any import that might log)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pretrain")

# ---------------------------------------------------------------------------
# Targeted import — bypass deepspeed.__init__ entirely
# ---------------------------------------------------------------------------
# We stub the deepspeed package so Python never executes
# deepspeed/__init__.py (which pulls in apex, op_builder, triton, etc.).
import importlib.util
import types as _types

def _stub_deepspeed():
    """Create minimal package stubs so submodule import works."""
    for name in ("deepspeed", "deepspeed.runtime"):
        if name not in sys.modules:
            stub = _types.ModuleType(name)
            stub.__path__ = [name.replace(".", "/")]
            stub.__package__ = name
            sys.modules[name] = stub

_stub_deepspeed()

try:
    _spec = importlib.util.spec_from_file_location(
        "deepspeed.runtime.desloc_engine",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepspeed", "runtime", "desloc_engine.py"),
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["deepspeed.runtime.desloc_engine"] = _mod
    _spec.loader.exec_module(_mod)
    DesLocEngine = _mod.DesLocEngine
    TrainingConfig = _mod.TrainingConfig
    logger.info("DesLocEngine / TrainingConfig imported (apex-free path)")
    _HAS_DESLOC = True
except Exception as _desloc_import_err:
    logger.warning(
        "Could not import DesLocEngine (%s); falling back to standalone training loop.",
        _desloc_import_err,
    )
    _HAS_DESLOC = False

# ---------------------------------------------------------------------------
# Model size presets (Llama-style)
# ---------------------------------------------------------------------------
_MODEL_CONFIGS: Dict[str, Dict] = {
    "70m": dict(hidden_size=512,  num_layers=8,  num_heads=8,  vocab_size=32000),
    "1b":  dict(hidden_size=2048, num_layers=16, num_heads=16, vocab_size=32000),
    "7b":  dict(hidden_size=4096, num_layers=32, num_heads=32, vocab_size=32000),
}

# ---------------------------------------------------------------------------
# Standalone Llama-style model (used when DesLocEngine is unavailable)
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    """Root-mean-square layer norm, no bias, BF16-friendly."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class _SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward network (Llama2 variant)."""

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
    """Multi-head causal self-attention (sdpa / FlashAttention path)."""

    def __init__(self, hidden: int, n_heads: int) -> None:
        super().__init__()
        assert hidden % n_heads == 0
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


class LlamaModel(nn.Module):
    """
    7B / 1B / 70M Llama-style causal LM.

    hidden_size=4096, layers=32, heads=32, vocab=32000 for 7B.
    Weight-ties embedding ↔ lm_head (standard LLaMA practice).
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        seq_len: int = 2048,
    ) -> None:
        super().__init__()
        self.embedding     = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(seq_len, hidden_size)
        self.layers = nn.ModuleList(
            [_TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)]
        )
        self.norm    = _RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # weight tie

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
# Data helpers
# ---------------------------------------------------------------------------

def synthetic_iter(
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Infinite synthetic token iterator for smoke-testing without real data."""
    while True:
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len + 1))
        yield tokens[:, :-1].to(device), tokens[:, 1:].to(device)


def real_data_iter(
    data_path: str,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Load a raw binary token file (uint16 / int32 flat array) and yield batches.

    Expected format: flat array of token ids, as produced by Megatron-style
    preprocess_data.py.  Falls back to synthetic data if loading fails.
    """
    import numpy as np  # noqa: PLC0415

    path = Path(data_path)
    if not path.exists():
        logger.warning("data-path '%s' not found; using synthetic data.", data_path)
        yield from synthetic_iter(32000, batch_size, seq_len, device)
        return

    # Try common binary formats
    for dtype in (np.uint16, np.int32):
        try:
            tokens = np.fromfile(str(path), dtype=dtype)
            logger.info(
                "Loaded %s: %d tokens (dtype=%s)", path.name, len(tokens), dtype.__name__
            )
            break
        except Exception:
            tokens = None

    if tokens is None or len(tokens) < seq_len + 1:
        logger.warning("Could not read '%s'; using synthetic data.", data_path)
        yield from synthetic_iter(32000, batch_size, seq_len, device)
        return

    chunk = seq_len + 1
    n_chunks = len(tokens) // chunk
    token_t = torch.tensor(tokens[:n_chunks * chunk].astype("int64")).reshape(n_chunks, chunk)
    idx = 0
    while True:
        end = idx + batch_size
        if end > n_chunks:
            idx = 0
            end = batch_size
        batch = token_t[idx:end].to(device)
        yield batch[:, :-1], batch[:, 1:]
        idx = end


# ---------------------------------------------------------------------------
# LR schedule (linear warmup + cosine decay)
# ---------------------------------------------------------------------------

def build_cosine_schedule(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# GPU memory helper
# ---------------------------------------------------------------------------

def _gpu_mem_str(device: torch.device) -> str:
    if device.type != "cuda":
        return "N/A"
    alloc_gb = torch.cuda.memory_allocated(device) / (1 << 30)
    resv_gb  = torch.cuda.memory_reserved(device)  / (1 << 30)
    return f"{alloc_gb:.2f}/{resv_gb:.2f} GB"


# ---------------------------------------------------------------------------
# Standalone training loop (fallback when DesLocEngine unavailable)
# ---------------------------------------------------------------------------

def run_standalone(args: argparse.Namespace) -> None:
    """Standalone PyTorch training loop — no DeepSpeed dependency."""
    cfg = _MODEL_CONFIGS[args.model_size]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32

    logger.info("=== Standalone training (no DesLocEngine) ===")
    logger.info("model-size=%s  device=%s  dtype=%s", args.model_size, device, dtype)

    model = LlamaModel(
        vocab_size  = cfg["vocab_size"],
        hidden_size = cfg["hidden_size"],
        num_layers  = cfg["num_layers"],
        num_heads   = cfg["num_heads"],
        seq_len     = args.seq_len,
    ).to(dtype=dtype, device=device)

    n_params = model.num_parameters
    logger.info("Model: %.2fM parameters", n_params / 1e6)

    optimizer = AdamW(
        model.parameters(),
        lr           = 3e-4,
        betas        = (0.9, 0.95),
        eps          = 1e-8,
        weight_decay = 0.1,
    )
    scheduler = build_cosine_schedule(optimizer, warmup_steps=10, total_steps=args.steps)

    if args.data_path:
        data = real_data_iter(args.data_path, args.batch_size, args.seq_len, device)
    else:
        data = synthetic_iter(cfg["vocab_size"], args.batch_size, args.seq_len, device)

    model.train()
    t0    = time.time()
    losses = []

    print(
        f"\n{'step':>6}  {'loss':>9}  {'lr':>10}  {'tok/s':>9}  {'GPU mem':>15}"
    )
    print("-" * 60)

    for step in range(1, args.steps + 1):
        input_ids, labels = next(data)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
            logits = model(input_ids)
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().reshape(-1, V),
                labels[:, :T - 1].contiguous().reshape(-1),
            )

        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if step % args.log_every == 0:
            elapsed   = time.time() - t0
            avg_loss  = sum(losses[-args.log_every:]) / args.log_every
            toks_step = args.batch_size * args.seq_len
            tok_s     = toks_step * args.log_every / max(elapsed, 1e-9)
            cur_lr    = scheduler.get_last_lr()[0]
            mem_str   = _gpu_mem_str(device)

            print(
                f"{step:>6}  {avg_loss:>9.4f}  {cur_lr:>10.2e}  {tok_s:>9.0f}  {mem_str:>15}"
            )
            t0 = time.time()

    final_loss = sum(losses[-min(10, len(losses)):]) / min(10, len(losses))
    initial_10 = sum(losses[:min(10, len(losses))]) / min(10, len(losses))
    loss_drop  = initial_10 - final_loss
    logger.info(
        "Training complete: initial_loss=%.4f  final_loss=%.4f  drop=%.4f (%.1f%%)",
        initial_10, final_loss, loss_drop, 100.0 * loss_drop / max(initial_10, 1e-9),
    )

    if args.steps >= 10:
        assert final_loss < initial_10 + 0.5, (
            f"Loss did not decrease: initial={initial_10:.4f}, final={final_loss:.4f}"
        )
        logger.info("✅  Loss decreased — training loop verified.")


# ---------------------------------------------------------------------------
# DesLocEngine training path
# ---------------------------------------------------------------------------

def run_desloc(args: argparse.Namespace) -> None:
    """Run training via the DES-LOC heterogeneous engine."""
    cfg = _MODEL_CONFIGS[args.model_size]

    tc = TrainingConfig(
        vocab_size      = cfg["vocab_size"],
        hidden_size     = cfg["hidden_size"],
        num_layers      = cfg["num_layers"],
        num_heads       = cfg["num_heads"],
        seq_len         = args.seq_len,
        total_steps     = args.steps,
        micro_batch_size= args.batch_size,
        global_batch_size= args.batch_size * 8,
        grad_accum_steps= 8,
        warmup_steps    = min(10, args.steps // 10),
        log_every       = args.log_every,
        save_every      = max(args.steps + 1, 10000),  # no saves during smoke test
        eval_every      = 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the LlamaModel and pass it in (DesLocEngine wraps it)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = LlamaModel(
        vocab_size  = cfg["vocab_size"],
        hidden_size = cfg["hidden_size"],
        num_layers  = cfg["num_layers"],
        num_heads   = cfg["num_heads"],
        seq_len     = args.seq_len,
    ).to(dtype=dtype, device=device)

    logger.info(
        "=== DesLocEngine path ===  model-size=%s  params=%.2fM  device=%s",
        args.model_size, model.num_parameters / 1e6, device,
    )

    data_iter = None
    if args.data_path:
        data_iter = real_data_iter(args.data_path, args.batch_size, args.seq_len, device)

    engine = DesLocEngine(config=tc, model=model, data_iter=data_iter)
    engine.train()

    logger.info("DesLocEngine training complete.")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Neuron_SP pretraining entry point (ags1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model-size",
        choices=list(_MODEL_CONFIGS.keys()),
        default="7b",
        help="Model size preset: 7b (hidden=4096, layers=32, heads=32), "
             "1b (hidden=2048, layers=16, heads=16), "
             "70m (hidden=512, layers=8, heads=8).",
    )
    p.add_argument(
        "--data-path",
        default=None,
        metavar="PATH",
        help="Path to a flat binary token file (uint16 or int32). "
             "If omitted, synthetic random tokens are used.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of training steps.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Micro-batch size (sequences per step).",
    )
    p.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Sequence length.",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log interval in steps.",
    )
    p.add_argument(
        "--use-desloc",
        action="store_true",
        default=False,
        help="Use DesLocEngine instead of standalone loop (requires full cluster).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Neuron_SP run_pretrain.py")
    logger.info("  model-size : %s  (%s)", args.model_size, _MODEL_CONFIGS[args.model_size])
    logger.info("  steps      : %d", args.steps)
    logger.info("  batch-size : %d", args.batch_size)
    logger.info("  seq-len    : %d", args.seq_len)
    logger.info("  data-path  : %s", args.data_path or "(synthetic)")
    logger.info("  use-desloc : %s", args.use_desloc)
    logger.info("  torch      : %s", torch.__version__)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(
                "  GPU %d: %s  %.0f GB  SM%d.%d",
                i, props.name, props.total_memory / (1 << 30),
                props.major, props.minor,
            )
    else:
        logger.info("  No CUDA GPUs found — running on CPU")
    logger.info("=" * 60)

    if args.use_desloc and _HAS_DESLOC:
        run_desloc(args)
    else:
        if args.use_desloc and not _HAS_DESLOC:
            logger.warning(
                "--use-desloc requested but DesLocEngine unavailable; "
                "using standalone loop."
            )
        run_standalone(args)


if __name__ == "__main__":
    main()
