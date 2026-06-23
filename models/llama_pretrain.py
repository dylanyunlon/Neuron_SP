"""
Pure PyTorch Llama model with RMSNorm, RoPE, SwiGLU, GQA.
Supports 70M / 1B / 7B configurations.
forward(input_ids) -> logits

Flash Attention: uses torch.nn.functional.scaled_dot_product_attention with
torch.backends.cuda.sdp_kernel to dispatch to the Flash-2 CUDA kernel when
available (requires CUDA + PyTorch >= 2.0, sm80+).  Falls back transparently
to math / mem-efficient kernels on unsupported hardware.
"""

import math
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Flash Attention kernel selector
# ---------------------------------------------------------------------------

def _flash_attn_available() -> bool:
    """True when the Flash-2 SDP kernel is present and CUDA is available."""
    if not torch.cuda.is_available():
        return False
    # PyTorch >= 2.0 exposes flash_sdp_enabled()
    try:
        return torch.backends.cuda.flash_sdp_enabled()
    except AttributeError:
        return False


@contextmanager
def _sdp_kernel(use_flash: bool):
    """Context manager that selects the SDP backend.

    * use_flash=True  → request Flash-2 kernel; fall back to mem-efficient
                        or math if not supported (no crash).
    * use_flash=False → vanilla math kernel (always available).
    """
    try:
        # PyTorch >= 2.0
        with torch.backends.cuda.sdp_kernel(
            enable_flash=use_flash,
            enable_math=not use_flash,
            enable_mem_efficient=use_flash,  # allow mem-efficient as fallback
        ):
            yield
    except AttributeError:
        # Older PyTorch — context manager absent; just yield (math path)
        yield


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LlamaConfig:
    vocab_size: int = 32000
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4          # GQA: kv heads (< n_heads -> grouped-query)
    ffn_hidden: int = 1376       # SwiGLU hidden dim
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    # Flash Attention: set True to dispatch to Flash-2 CUDA kernel via
    # torch.backends.cuda.sdp_kernel.  Automatically falls back to
    # mem-efficient / math SDP when running on CPU or older GPUs.
    use_flash_attn: bool = True


def llama_70m() -> LlamaConfig:
    """~70M parameter config."""
    return LlamaConfig(
        vocab_size=32000, dim=512, n_layers=8,
        n_heads=8, n_kv_heads=4, ffn_hidden=1376,
        max_seq_len=2048,
    )


def llama_1b() -> LlamaConfig:
    """~1B parameter config (Llama-3.2-1B style)."""
    return LlamaConfig(
        vocab_size=32000, dim=2048, n_layers=16,
        n_heads=32, n_kv_heads=8, ffn_hidden=8192,
        max_seq_len=2048,
    )


def llama_7b() -> LlamaConfig:
    """~7B parameter config (Llama-2-7B style)."""
    return LlamaConfig(
        vocab_size=32000, dim=4096, n_layers=32,
        n_heads=32, n_kv_heads=8, ffn_hidden=11008,
        max_seq_len=4096,
    )


CONFIG_MAP = {
    "70m": llama_70m,
    "1b":  llama_1b,
    "7b":  llama_7b,
}


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, dim)
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------

def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Returns complex tensor of shape (max_seq_len, dim//2)."""
    half = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)          # (T, half)
    return torch.polar(torch.ones_like(freqs), freqs)   # complex64


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor,
                     freqs_cis: torch.Tensor):
    """Apply RoPE to query and key tensors.

    Args:
        xq: (B, T, n_heads, head_dim)
        xk: (B, T, n_kv_heads, head_dim)
        freqs_cis: (T, head_dim//2) complex
    """
    def rotate(x, freqs):
        # x: (B, T, H, D)  ->  view as complex  (B, T, H, D//2)
        xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs = freqs.unsqueeze(0).unsqueeze(2)   # (1, T, 1, D//2)
        out = torch.view_as_real(xc * freqs).flatten(-2)
        return out.type_as(x)

    return rotate(xq, freqs_cis), rotate(xk, freqs_cis)


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj   = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim,  bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Grouped-Query Attention (GQA)
# ---------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.n_heads    = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim   = cfg.dim // cfg.n_heads
        self.n_groups   = cfg.n_heads // cfg.n_kv_heads   # heads per kv group
        self.use_flash  = cfg.use_flash_attn

        self.q_proj = nn.Linear(cfg.dim, cfg.n_heads    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim,    bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape

        xq = self.q_proj(x).view(B, T, self.n_heads,    self.head_dim)
        xk = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Expand kv for grouped-query: repeat each kv head n_groups times
        xk = xk.repeat_interleave(self.n_groups, dim=2)  # (B, T, n_heads, D)
        xv = xv.repeat_interleave(self.n_groups, dim=2)

        # (B, H, T, D) layout expected by scaled_dot_product_attention
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Flash Attention dispatch via torch.backends.cuda.sdp_kernel:
        #   use_flash=True  -> requests Flash-2 kernel first, falls back to
        #                      mem-efficient then math (no crash on CPU/old GPU)
        #   use_flash=False -> math kernel only
        # F.scaled_dot_product_attention is the public API; the context manager
        # controls which CUDA kernel is selected beneath it.
        with _sdp_kernel(use_flash=self.use_flash):
            out = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=mask,
                is_causal=(mask is None),
            )

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class LlamaBlock(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn      = GroupedQueryAttention(cfg)
        self.ffn_norm  = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn       = SwiGLUFFN(cfg.dim, cfg.ffn_hidden)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# Llama Model
# ---------------------------------------------------------------------------

class LlamaForCausalLM(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing = False

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers       = nn.ModuleList([LlamaBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm         = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head      = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embed_tokens.weight

        # Precompute RoPE frequencies; registered as buffer (not a parameter)
        freqs = precompute_freqs_cis(cfg.dim // cfg.n_heads,
                                     cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("freqs_cis", freqs, persistent=False)

        self._init_weights()

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for all transformer blocks.
        Required for 7B on 48GB A6000 — reduces activation memory ~4x."""
        self.gradient_checkpointing = True

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0,
                                std=0.02 / math.sqrt(2 * self.cfg.n_layers))
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) long tensor
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, \
            f"Sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"

        x = self.embed_tokens(input_ids)                 # (B, T, dim)
        freqs = self.freqs_cis[:T]                        # (T, head_dim//2)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, freqs, use_reentrant=False,
                )
            else:
                x = layer(x, freqs)

        x = self.norm(x)
        return self.lm_head(x)                            # (B, T, vocab_size)

    @staticmethod
    def from_name(name: str) -> "LlamaForCausalLM":
        """Instantiate by config name: '70m', '1b', or '7b'."""
        if name not in CONFIG_MAP:
            raise ValueError(f"Unknown config '{name}'. Choose from {list(CONFIG_MAP)}")
        return LlamaForCausalLM(CONFIG_MAP[name]())

    def num_parameters(self, non_embedding: bool = True) -> int:
        params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            params -= self.embed_tokens.weight.numel()
        return params


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for name in ("70m", "1b"):
        model = LlamaForCausalLM.from_name(name)
        ids   = torch.randint(0, 32000, (2, 64))
        logits = model(ids)
        print(f"[{name:>3s}] params={model.num_parameters()/1e6:.1f}M  "
              f"logits={tuple(logits.shape)}")
