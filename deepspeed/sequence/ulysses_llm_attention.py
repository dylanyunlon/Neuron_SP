# SPDX-License-Identifier: Apache-2.0
"""Ulysses SP wrapper for LLM attention (GroupedQueryAttention interface).

Handles non-divisible head counts by padding. When n_heads % sp_size != 0,
pads Q heads to the next multiple, scatters, computes, gathers, and strips.

K/V heads are NOT scattered — they are replicated per rank (GQA already
means n_kv_heads << n_heads, so replication is cheap).

Ref: DeepSpeed Ulysses (Jacobs et al. 2023), layer.py _SeqAllToAll
"""
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


class UlyssesSPLLMAttention(nn.Module):
    """Drop-in replacement for GroupedQueryAttention with Ulysses SP.

    forward(x, freqs_cis, mask) → output  (same interface)
    """

    def __init__(self, original_attn: nn.Module, sp_group: dist.ProcessGroup):
        super().__init__()
        self.attn = original_attn
        self.sp_group = sp_group
        self.sp_size = dist.get_world_size(sp_group)
        self.sp_rank = dist.get_rank(sp_group)

        self.n_heads = original_attn.n_heads
        self.n_kv_heads = original_attn.n_kv_heads
        self.head_dim = original_attn.head_dim
        self.n_groups = original_attn.n_groups  # GQA group count

        # Compute per-rank head assignment (may be unequal)
        self.heads_per_rank = _ceil_div(self.n_heads, self.sp_size)
        self.padded_n_heads = self.heads_per_rank * self.sp_size
        self._head_pad = self.padded_n_heads - self.n_heads

        # This rank's head range
        self.local_head_start = self.sp_rank * self.heads_per_rank
        self.local_head_end = min(self.local_head_start + self.heads_per_rank, self.n_heads)
        self.local_n_heads = self.local_head_end - self.local_head_start

        logger.info(
            "UlyssesSPLLMAttention: sp_size=%d, total_heads=%d, "
            "rank=%d heads=[%d:%d] (%d heads, padded=%d)",
            self.sp_size, self.n_heads, self.sp_rank,
            self.local_head_start, self.local_head_end,
            self.local_n_heads, self._head_pad,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape

        # 1. QKV projection (all heads, local sequence T)
        xq = self.attn.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        xk = self.attn.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        xv = self.attn.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        # 2. Apply RoPE BEFORE scatter (positions are local, correct for this rank's tokens)
        from models.llama_pretrain import apply_rotary_emb
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # 3. Expand KV for GQA (replicate kv heads to match q heads)
        xk = xk.repeat_interleave(self.n_groups, dim=2)  # (B, T, n_heads, D)
        xv = xv.repeat_interleave(self.n_groups, dim=2)

        # 4. Pad heads if n_heads % sp_size != 0
        if self._head_pad > 0:
            pad = torch.zeros(B, T, self._head_pad, self.head_dim,
                            device=x.device, dtype=xq.dtype)
            xq = torch.cat([xq, pad], dim=2)
            xk = torch.cat([xk, pad], dim=2)
            xv = torch.cat([xv, pad], dim=2)

        # 5. All-to-all: scatter head dim, gather seq dim
        # Each rank sends its chunk of heads to all ranks, receives all seq chunks
        # Before: (B, T, padded_H, D) on each rank
        # After:  (B, T*P, heads_per_rank, D) on each rank
        xq = self._head_scatter_seq_gather(xq)
        xk = self._head_scatter_seq_gather(xk)
        xv = self._head_scatter_seq_gather(xv)

        T_full = xq.shape[1]  # T * sp_size

        # 6. Attention on local heads, full sequence
        # (B, heads_per_rank, T_full, D)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Mask real vs padded heads
        if self.local_n_heads < self.heads_per_rank:
            # Zero out padded heads so they don't contribute
            xq[:, self.local_n_heads:] = 0
            xk[:, self.local_n_heads:] = 0
            xv[:, self.local_n_heads:] = 0

        out = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=None,
            is_causal=True,
        )
        # (B, heads_per_rank, T_full, D) → (B, T_full, heads_per_rank, D)
        out = out.transpose(1, 2).contiguous()

        # 7. All-to-all back: scatter seq dim, gather head dim
        # Before: (B, T_full, heads_per_rank, D)
        # After:  (B, T, padded_H, D)
        out = self._seq_scatter_head_gather(out)

        # 8. Strip padded heads
        if self._head_pad > 0:
            out = out[:, :, :self.n_heads, :]

        out = out.contiguous().view(B, T, -1)
        return self.attn.o_proj(out)

    def _head_scatter_seq_gather(self, x: torch.Tensor) -> torch.Tensor:
        """All-to-all: scatter head dim (2), gather seq dim (1).
        Input:  (B, T, padded_H, D)
        Output: (B, T*P, heads_per_rank, D)
        """
        B, T, H, D = x.shape
        P = self.sp_size
        HPR = H // P  # heads_per_rank
        # Reshape: (B, T, P, HPR, D) → (P, B, T, HPR, D)
        x = x.view(B, T, P, HPR, D).permute(2, 0, 1, 3, 4).contiguous()
        output = torch.empty_like(x)
        dist.all_to_all_single(output, x, group=self.sp_group)
        # (P, B, T, HPR, D) → (B, P*T, HPR, D)
        output = output.permute(1, 0, 2, 3, 4).contiguous()
        return output.view(B, T * P, HPR, D)

    def _seq_scatter_head_gather(self, x: torch.Tensor) -> torch.Tensor:
        """All-to-all: scatter seq dim (1), gather head dim (2). (reverse)
        Input:  (B, T*P, HPR, D)
        Output: (B, T, padded_H, D)
        """
        B, TP, HPR, D = x.shape
        P = self.sp_size
        T = TP // P
        # (B, P, T, HPR, D) → (P, B, T, HPR, D)
        x = x.view(B, P, T, HPR, D).permute(1, 0, 2, 3, 4).contiguous()
        output = torch.empty_like(x)
        dist.all_to_all_single(output, x, group=self.sp_group)
        # (P, B, T, HPR, D) → (B, T, P*HPR, D)
        output = output.permute(1, 2, 0, 3, 4).contiguous()
        return output.view(B, T, P * HPR, D)
