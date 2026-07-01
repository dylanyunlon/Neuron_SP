"""
Minimal BF16-native transformer building blocks for DES-LOC pretraining smoke tests.
"""
from __future__ import annotations

import math
from typing import Dict, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from deepspeed.runtime.desloc_config import TrainingConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (BF16-friendly, no mean subtraction)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, BF16-compatible."""

    def __init__(self, hidden: int, n_heads: int) -> None:
        super().__init__()
        assert hidden % n_heads == 0, "hidden must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, n_heads, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    """Position-wise feed-forward network (SwiGLU variant)."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        intermediate = int(hidden * 8 / 3)
        intermediate = (intermediate + 63) // 64 * 64  # round to multiple of 64
        self.gate = nn.Linear(hidden, intermediate, bias=False)
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    """Single transformer decoder block with pre-norm."""

    def __init__(self, hidden: int, n_heads: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden)
        self.attn = CausalSelfAttention(hidden, n_heads)
        self.norm2 = RMSNorm(hidden)
        self.mlp = MLP(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MiniTransformer(nn.Module):
    """
    Minimal causal language model for DES-LOC pretraining smoke tests.

    In production this would be replaced by the full model passed into
    DesLocEngine, but it serves as the default when no model is provided.
    """

    def __init__(self, cfg: TrainingConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_embedding = nn.Embedding(cfg.seq_len, cfg.hidden_size)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.hidden_size, cfg.num_heads)
             for _ in range(cfg.num_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Long tensor of shape (B, T).

        Returns:
            Logits tensor of shape (B, T, vocab_size).
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)


def build_warmup_cosine_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Build a combined linear-warmup + cosine-decay LR schedule.

    Args:
        optimizer: The AdamW optimizer.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total training steps.
        min_lr_ratio: Ratio of min_lr to max_lr for cosine floor.

    Returns:
        LambdaLR scheduler.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def infinite_data_iter(
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> Iterator[Dict[str, torch.Tensor]]:
    """
    Infinite iterator of random token batches for smoke testing.

    Yields dicts with "tokens" and "labels" keys matching the BATCH_KEYS
    contract expected by HeteroElasticBatch._fetch_from_iterator().

    Args:
        vocab_size: Vocabulary size for random token sampling.
        batch_size: Number of sequences per batch.
        seq_len: Sequence length.
        device: Target device (tokens are on CPU, moved to GPU in the loop).

    Yields:
        Dict with "tokens" and "labels" each of shape (batch_size, seq_len).
    """
    while True:
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len + 1))
        yield {"tokens": tokens[:, :-1], "labels": tokens[:, 1:]}
