"""
Neuron_SP Project — LLaMA-7B Architecture
Pure PyTorch, zero HuggingFace dependency.
Supports GQA-ready attention, RoPE, SwiGLU, RMSNorm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LLaMAConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32          # set < num_attention_heads for GQA
    intermediate_size: int = 11008
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    pad_token_id: int = 0
    # derived
    head_dim: int = field(init=False)

    def __post_init__(self):
        assert self.hidden_size % self.num_attention_heads == 0, \
            "hidden_size must be divisible by num_attention_heads"
        assert self.num_attention_heads % self.num_key_value_heads == 0, \
            "num_attention_heads must be divisible by num_key_value_heads"
        object.__setattr__(self, "head_dim", self.hidden_size // self.num_attention_heads)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no mean subtraction, no bias)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # cast to float32 for numerical stability, then back
        orig_dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms).to(orig_dtype) * self.weight


# ---------------------------------------------------------------------------
# Rotary Positional Embedding (RoPE)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """
    RoPE with cached inverse-frequency table.
    Supports arbitrary sequence lengths (cache is extended on demand).
    """

    def __init__(self, dim: int, max_position_embeddings: int = 4096,
                 base: float = 10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_position_embeddings, device=device)

    def _build_cache(self, seq_len: int, device=None, dtype=torch.float32):
        self.max_seq_cached = seq_len
        t = torch.arange(seq_len, device=device or self.inv_freq.device, dtype=dtype)
        freqs = torch.outer(t, self.inv_freq)          # [seq, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)        # [seq, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        if seq_len > self.max_seq_cached:
            self._build_cache(seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor,
                          cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # cos/sin: [seq_len, head_dim] → unsqueeze for broadcast over batch/heads
    cos = cos.unsqueeze(0).unsqueeze(0)   # [1, 1, seq, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


# ---------------------------------------------------------------------------
# Attention (GQA-ready, uses F.scaled_dot_product_attention → FlashAttn)
# ---------------------------------------------------------------------------

class LLaMAAttention(nn.Module):

    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Expand KV heads to match Q heads for GQA."""
        if self.num_kv_groups == 1:
            return x
        bsz, num_kv_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(bsz, num_kv_heads, self.num_kv_groups, seq_len, head_dim)
        return x.reshape(bsz, num_kv_heads * self.num_kv_groups, seq_len, head_dim)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(q, seq_len=seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # F.scaled_dot_product_attention dispatches to FlashAttention when available
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=(attention_mask is None),
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.o_proj(attn_output)


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward
# ---------------------------------------------------------------------------

class LLaMAMLP(nn.Module):
    """SwiGLU: gate_proj + up_proj → SiLU(gate) * up → down_proj."""

    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class LLaMADecoderLayer(nn.Module):

    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.input_layernorm    = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn          = LLaMAAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp               = LLaMAMLP(config)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        # Pre-norm MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Full LLaMA Model
# ---------------------------------------------------------------------------

class LLaMAModel(nn.Module):
    """Decoder-only transformer trunk (no LM head)."""

    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                         padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([LLaMADecoderLayer(config)
                                      for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._init_weights()

    def _init_weights(self):
        std = self.config.initializer_range
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        return self.norm(hidden_states)


class LLaMAForCausalLM(nn.Module):
    """LLaMA with language-model head for causal pretraining."""

    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        self.model  = LLaMAModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        hidden_states = self.model(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)               # [B, T, vocab]

        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return loss, logits

    # ------------------------------------------------------------------
    # Pipeline-parallel helpers
    # ------------------------------------------------------------------

    def get_layer_groups(self) -> List[nn.Module]:
        """
        Return a flat list of pipeline-splittable segments.
        DeepSpeed Pipeline Engine assigns each rank a contiguous slice.

        Segments:
          [0]       embedding
          [1..32]   decoder layers 0-31
          [33]      final norm + lm_head
        """
        groups: List[nn.Module] = [self.model.embed_tokens]
        groups.extend(self.model.layers)
        groups.append(nn.Sequential(self.model.norm, self.lm_head))
        return groups

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_llama_7b(config: Optional[LLaMAConfig] = None) -> LLaMAForCausalLM:
    if config is None:
        config = LLaMAConfig()
    return LLaMAForCausalLM(config)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = LLaMAConfig()
    model = build_llama_7b(cfg)
    total = model.count_parameters()
    print(f"Total parameters : {total:,}  ({total/1e9:.3f}B)")
    assert 6.5e9 < total < 7.0e9, f"Unexpected param count: {total}"

    # Forward pass smoke test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    ids = torch.randint(0, cfg.vocab_size, (2, 128), device=device)
    labels = ids.clone()
    loss, logits = model(ids, labels=labels)
    print(f"Loss  : {loss.item():.4f}")
    print(f"Logits: {logits.shape}")   # [2, 128, 32000]
    print("llama_7b.py — OK")
