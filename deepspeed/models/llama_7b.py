"""
LLaMA-7B Model Definition for DES-LOC Heterogeneous Training
=============================================================

Pure PyTorch implementation of LLaMA-7B (6.74B params):
- 32 TransformerBlocks, d_model=4096, 32 heads, intermediate=11008
- RMSNorm, SwiGLU activation, Rotary Position Embeddings (RoPE)
- No dependency on HuggingFace transformers

Supports pipeline-parallel slicing via get_layer_groups().
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class LLaMA7BConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 32       # GQA: set < num_heads for grouped query
    max_seq_len: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    dtype: torch.dtype = torch.bfloat16


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._max_seq = max_seq_len

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(self, config: LLaMA7BConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.num_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(q, S)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA: expand kv heads
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        # Scaled dot-product attention (uses FlashAttention when available)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=(mask is None))
        attn = attn.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(attn)


class TransformerBlock(nn.Module):
    def __init__(self, config: LLaMA7BConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LLaMA7B(nn.Module):
    """
    Full LLaMA-7B model.

    Parameter count breakdown:
      - embed_tokens:  32000 * 4096       =  131M
      - 32 x TransformerBlock:
          - q/k/v/o proj: 4 * 4096^2      =   67M  (x32 = 2.15B)
          - SwiGLU: 3 * 4096 * 11008      =  135M  (x32 = 4.33B)
          - norms: negligible
      - lm_head:       4096 * 32000       =  131M
      Total: ~6.74B parameters
    """

    def __init__(self, config: Optional[LLaMA7BConfig] = None):
        super().__init__()
        if config is None:
            config = LLaMA7BConfig()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.lm_head(x)

    def get_layer_groups(self) -> List[nn.Module]:
        """Return sliceable layer groups for pipeline parallelism."""
        groups = [self.embed_tokens]
        groups.extend(self.layers)
        groups.append(nn.Sequential(self.norm, self.lm_head))
        return groups

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    config = LLaMA7BConfig()
    model = LLaMA7B(config)
    n_params = model.count_parameters()
    print(f"LLaMA-7B: {n_params:,} parameters ({n_params/1e9:.2f}B)")
    assert 6.5e9 < n_params < 7.5e9, f"Expected ~6.7B, got {n_params/1e9:.2f}B"

    # Smoke test forward pass (CPU, small input)
    x = torch.randint(0, config.vocab_size, (1, 16))
    with torch.no_grad():
        logits = model(x)
    print(f"Forward OK: input {x.shape} → logits {logits.shape}")
    assert logits.shape == (1, 16, config.vocab_size)
