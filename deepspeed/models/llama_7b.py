"""
Pure-PyTorch LLaMA-7B model definition for Neuron_SP / DES-LOC heterogeneous pretraining.
Zero HuggingFace dependencies.

Config:
    vocab_size    = 32000
    d_model       = 4096
    n_heads       = 32
    n_layers      = 32
    intermediate  = 11008
    max_seq_len   = 2048
    n_kv_heads    = 32   (GQA-ready; set < n_heads to enable grouped-query attention)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LLaMAConfig:
    vocab_size:     int   = 32000
    d_model:        int   = 4096
    n_heads:        int   = 32
    n_kv_heads:     int   = 32        # set < n_heads for GQA
    n_layers:       int   = 32
    intermediate:   int   = 11008
    max_seq_len:    int   = 2048
    rms_norm_eps:   float = 1e-6
    rope_theta:     float = 10000.0
    init_std:       float = 0.02
    tie_embeddings: bool  = False      # LLaMA-7B does NOT tie by default


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.weight.shape[0]}, eps={self.eps}"


# ---------------------------------------------------------------------------
# Rotary Positional Embedding (complex-number rotation)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """
    RoPE via complex-number rotation (Su et al., 2021).
    Frequencies are cached up to max_seq_len and moved to the correct device
    lazily on first forward call.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.head_dim   = head_dim
        self.max_seq_len = max_seq_len
        self.theta      = theta

        # Pre-compute inverse frequencies — not a parameter, registered as buffer
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)          # (seq_len, head_dim/2)
        emb   = torch.cat([freqs, freqs], dim=-1)      # (seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q, k: (batch, n_heads, seq_len, head_dim)
        Returns rotated q, k with the same shape.
        """
        if seq_len + offset > self.cos_cached.shape[0]:
            self._build_cache(seq_len + offset)

        cos = self.cos_cached[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot.to(q.dtype), k_rot.to(k.dtype)


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """
    SwiGLU FFN: FFN(x) = (Swish(W_gate · x) ⊙ (W_up · x)) · W_down
    intermediate_size should be pre-scaled to 2/3 * 4 * d_model ≈ 11008 for d=4096.
    """

    def __init__(self, d_model: int, intermediate: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, intermediate, bias=False)
        self.w_up   = nn.Linear(d_model, intermediate, bias=False)
        self.w_down = nn.Linear(intermediate, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ---------------------------------------------------------------------------
# GQA-ready Multi-Head (Grouped-Query) Attention
# ---------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    """
    Attention with Grouped-Query support (Ainslie et al., 2023).
    When n_kv_heads == n_heads this is standard MHA.
    Uses F.scaled_dot_product_attention for fused Flash-Attention / memory-efficient kernels.
    """

    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        assert config.d_model % config.n_heads == 0, \
            "d_model must be divisible by n_heads"
        assert config.n_heads % config.n_kv_heads == 0, \
            "n_heads must be divisible by n_kv_heads"

        self.n_heads    = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep      = config.n_heads // config.n_kv_heads   # repetition factor for GQA
        self.head_dim   = config.d_model // config.n_heads

        self.q_proj  = nn.Linear(config.d_model, config.n_heads    * self.head_dim, bias=False)
        self.k_proj  = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj  = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj  = nn.Linear(config.n_heads * self.head_dim, config.d_model,    bias=False)

        self.rope = RotaryEmbedding(
            head_dim    = self.head_dim,
            max_seq_len = config.max_seq_len,
            theta       = config.rope_theta,
        )

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Expand KV heads to match Q heads for GQA."""
        if n_rep == 1:
            return x
        b, nkv, s, hd = x.shape
        return x[:, :, None, :, :].expand(b, nkv, n_rep, s, hd).reshape(b, nkv * n_rep, s, hd)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """
        x              : (batch, seq_len, d_model)
        attention_mask : (batch, 1, seq_len, seq_len) additive mask, or None
                         Pass a causal mask for autoregressive decoding.
        """
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = self.rope(q, k, seq_len=S, offset=position_offset)

        # Expand KV for GQA
        k = self._repeat_kv(k, self.n_rep)
        v = self._repeat_kv(v, self.n_rep)

        # Fused scaled dot-product attention (Flash-Attention when available)
        # is_causal=True generates the causal mask internally; pass attn_mask=None
        # when relying on the built-in causal mask to avoid redundancy.
        if attention_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)

        out = out.transpose(1, 2).contiguous().view(B, S, self.n_heads * self.head_dim)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Transformer Decoder Layer
# ---------------------------------------------------------------------------

class LLaMADecoderLayer(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        self.self_attn  = GroupedQueryAttention(config)
        self.ffn        = SwiGLU(config.d_model, config.intermediate)
        self.input_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.post_norm  = RMSNorm(config.d_model, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        # Pre-norm architecture (same as Meta's LLaMA)
        x = x + self.self_attn(self.input_norm(x), attention_mask, position_offset)
        x = x + self.ffn(self.post_norm(x))
        return x


# ---------------------------------------------------------------------------
# Full LLaMA-7B Model
# ---------------------------------------------------------------------------

class LLaMA7B(nn.Module):
    """
    LLaMA-7B transformer for Neuron_SP / DES-LOC heterogeneous pretraining.

    No HuggingFace / transformers dependencies.
    Supports pipeline-parallel layer splitting via get_layer_groups().
    """

    def __init__(self, config: Optional[LLaMAConfig] = None) -> None:
        super().__init__()
        self.config = config or LLaMAConfig()
        cfg = self.config

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers       = nn.ModuleList(
            [LLaMADecoderLayer(cfg) for _ in range(cfg.n_layers)]
        )
        self.norm         = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.lm_head      = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Normal(0, 0.02) initialisation for all linear / embedding weights."""
        std = self.config.init_std
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_offset: int = 0,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        input_ids  : (batch, seq_len)  — token indices
        labels     : (batch, seq_len)  — shifted targets; if provided, cross-entropy loss
                                         is returned as second element
        Returns    : (logits, loss | None)
        """
        x = self.embed_tokens(input_ids)                    # (B, S, d_model)

        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask, position_offset=position_offset)

        x      = self.norm(x)
        logits = self.lm_head(x)                            # (B, S, vocab_size)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            # Shift so that token t predicts token t+1
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    # ------------------------------------------------------------------
    # Pipeline / tensor-parallel helpers
    # ------------------------------------------------------------------

    def get_layer_groups(self) -> List[nn.Module]:
        """
        Returns a flat, ordered list of sliceable module groups for
        pipeline-parallel stage assignment in DES-LOC:

            [embed_tokens, layer_0, layer_1, ..., layer_31, norm + lm_head]

        Each group can be assigned to a different device / rank.
        """
        groups: List[nn.Module] = [self.embed_tokens]
        groups.extend(self.layers)
        groups.append(nn.Sequential(self.norm, self.lm_head))
        return groups

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Returns total (trainable) parameter count.
        Expected ~6.74 B for the default 7B config.

        Breakdown (approx):
            embed_tokens       :  32000 × 4096       ≈  131 M
            attention (×32)    :  4 × 4096² (MHA)    ≈ 2147 M
            ffn (×32)          :  3 × 4096 × 11008   ≈ 4076 M
            norms              :  negligible
            lm_head            :  4096 × 32000        ≈  131 M  (not tied)
            Total                                      ≈ 6585 M  ≈ 6.74 B
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def print_parameter_summary(self) -> None:
        total   = self.count_parameters(trainable_only=False)
        trainable = self.count_parameters(trainable_only=True)
        print(f"{'='*55}")
        print(f"  LLaMA-7B Parameter Summary")
        print(f"{'='*55}")
        print(f"  Total parameters      : {total:>15,}  ({total/1e9:.3f} B)")
        print(f"  Trainable parameters  : {trainable:>15,}  ({trainable/1e9:.3f} B)")
        print(f"  Non-trainable params  : {total-trainable:>15,}")
        print(f"{'='*55}")
        for name, module in self.named_children():
            p = sum(x.numel() for x in module.parameters())
            print(f"  {name:<20s}  {p:>15,}  ({p/1e9:.4f} B)")
        print(f"{'='*55}")


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_llama_7b(device: str = "cpu", dtype: torch.dtype = torch.float32) -> LLaMA7B:
    """Convenience factory used by DES-LOC training scripts."""
    model = LLaMA7B(LLaMAConfig()).to(device=device, dtype=dtype)
    return model


# ---------------------------------------------------------------------------
# Smoke-test (python -m deepspeed.models.llama_7b)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    model = LLaMA7B()
    model.print_parameter_summary()

    # Forward pass
    B, S = 2, 128
    ids  = torch.randint(0, 32000, (B, S))
    lbl  = torch.randint(0, 32000, (B, S))

    with torch.no_grad():
        logits, loss = model(ids, labels=lbl)

    print(f"\nForward pass OK  —  logits: {tuple(logits.shape)},  loss: {loss.item():.4f}")
    print(f"Layer groups     : {len(model.get_layer_groups())} slices")
    assert logits.shape == (B, S, 32000), "Unexpected logits shape"
    assert loss is not None and loss.item() > 0, "Loss sanity check failed"
    print("\nAll checks passed.")
