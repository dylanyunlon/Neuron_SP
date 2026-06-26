# SPDX-License-Identifier: Apache-2.0
"""Ulysses SP wrapper for LLM attention (GroupedQueryAttention interface).

Adapts the (x, freqs_cis, mask) → output interface used by LLaMA-style
attention to Ulysses all-to-all sequence parallelism from DeepSpeed's
_SeqAllToAll (deepspeed/sequence/layer.py).

Ulysses SP flow for LLM:
  Input x: (B, S, D) — full sequence, replicated across ranks
  1. QKV projection: x → Q(B,S,H,D_h), K(B,S,Hkv,D_h), V(B,S,Hkv,D_h)
  2. All-to-all: scatter seq dim, gather head dim
     Q: (B, S, H, D_h) → (B, S*P, H/P, D_h)   [each rank: full seq, subset heads]
  3. Apply RoPE on full-length positions
  4. GQA head expansion + attention (on subset of heads)
  5. All-to-all back: scatter head dim, gather seq dim
  6. Output projection

Reference: DeepSpeed Ulysses (Jacobs et al. 2023), layer.py _SeqAllToAll
"""
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist

logger = logging.getLogger(__name__)


class UlyssesSPLLMAttention(nn.Module):
    """Drop-in replacement for GroupedQueryAttention with Ulysses SP.

    Keeps the same forward signature: forward(x, freqs_cis, mask) → output
    but internally scatters sequence and gathers heads via all-to-all.
    """

    def __init__(self, original_attn: nn.Module, sp_group: dist.ProcessGroup):
        super().__init__()
        self.attn = original_attn
        self.sp_group = sp_group
        self.sp_size = dist.get_world_size(sp_group)
        self.sp_rank = dist.get_rank(sp_group)

        # Validate head count divisibility
        n_heads = original_attn.n_heads
        n_kv_heads = original_attn.n_kv_heads
        if n_heads % self.sp_size != 0:
            raise ValueError(
                f"UlyssesSP: n_heads={n_heads} not divisible by sp_size={self.sp_size}"
            )
        self.local_n_heads = n_heads // self.sp_size
        self.local_n_kv_heads = n_kv_heads // self.sp_size if n_kv_heads % self.sp_size == 0 else n_kv_heads
        self.head_dim = original_attn.head_dim

        logger.info(
            "UlyssesSPLLMAttention: sp_size=%d, heads %d→%d/rank, kv_heads %d→%d/rank",
            self.sp_size, n_heads, self.local_n_heads,
            n_kv_heads, self.local_n_kv_heads,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape

        # 1. QKV projection (full heads, local sequence)
        xq = self.attn.q_proj(x).view(B, T, self.attn.n_heads, self.head_dim)
        xk = self.attn.k_proj(x).view(B, T, self.attn.n_kv_heads, self.head_dim)
        xv = self.attn.v_proj(x).view(B, T, self.attn.n_kv_heads, self.head_dim)

        # 2. All-to-all: scatter seq dim (1), gather head dim (2)
        # Before: (B, T, H, D) — each rank has T tokens, all H heads
        # After:  (B, T*P, H/P, D) — each rank has T*P tokens, H/P heads
        xq = self._seq_scatter_head_gather(xq)
        xk = self._seq_scatter_head_gather(xk)
        xv = self._seq_scatter_head_gather(xv)

        # Now each rank has full sequence (T*P) for its subset of heads
        T_full = xq.shape[1]

        # 3. Apply RoPE on full-length positions
        # freqs_cis was computed for T tokens; need T_full positions
        if freqs_cis.shape[0] < T_full:
            # Recompute for full length — use the model's precomputed table
            # freqs_cis is typically sliced from a larger table
            freqs_cis = freqs_cis.repeat(self.sp_size, 1) if freqs_cis.shape[0] * self.sp_size == T_full else freqs_cis

        from models.llama_pretrain import apply_rotary_emb
        xq_local = xq[:, :, :self.local_n_heads, :]
        xk_local = xk[:, :, :self.local_n_kv_heads, :]
        xq_local, xk_local = apply_rotary_emb(xq_local, xk_local, freqs_cis[:T_full])

        # 4. GQA head expansion for local heads
        n_groups = self.local_n_heads // self.local_n_kv_heads if self.local_n_kv_heads > 0 else 1
        if n_groups > 1:
            xk_local = xk_local.repeat_interleave(n_groups, dim=2)
            xv_local = xv[:, :, :self.local_n_kv_heads, :].repeat_interleave(n_groups, dim=2)
        else:
            xv_local = xv[:, :, :self.local_n_kv_heads, :]

        # 5. Attention on (B, H_local, T_full, D)
        xq_local = xq_local.transpose(1, 2)
        xk_local = xk_local.transpose(1, 2)
        xv_local = xv_local.transpose(1, 2)

        import torch.nn.functional as F
        out = F.scaled_dot_product_attention(
            xq_local, xk_local, xv_local,
            attn_mask=None,
            is_causal=True,
        )
        # out: (B, H_local, T_full, D) → (B, T_full, H_local, D)
        out = out.transpose(1, 2).contiguous()

        # 6. All-to-all back: scatter head dim, gather seq dim
        # Before: (B, T_full, H/P, D) — full seq, local heads
        # After:  (B, T, H, D) — local seq, all heads
        out = self._head_scatter_seq_gather(out)
        # out: (B, T, H_local*P, D) = (B, T, H, D)

        out = out.contiguous().view(B, T, -1)
        return self.attn.o_proj(out)

    def _seq_scatter_head_gather(self, x: torch.Tensor) -> torch.Tensor:
        """All-to-all: scatter sequence dim, gather head dim.
        Input:  (B, T, H, D) — each rank has T tokens, H heads
        Output: (B, T*P, H/P, D) — each rank has T*P tokens, H/P heads
        """
        B, T, H, D = x.shape
        P = self.sp_size
        # Reshape to (B, T, P, H//P, D) then permute to (P, B, T, H//P, D)
        # and all_to_all along dim 0
        x = x.view(B, T, P, H // P, D)
        x = x.permute(2, 0, 1, 3, 4).contiguous()  # (P, B, T, H//P, D)
        output = torch.empty_like(x)
        dist.all_to_all_single(output, x, group=self.sp_group)
        # output: (P, B, T, H//P, D) → (B, T*P, H//P, D)
        output = output.permute(1, 0, 2, 3, 4).contiguous()  # (B, P, T, H//P, D)
        return output.view(B, T * P, H // P, D)

    def _head_scatter_seq_gather(self, x: torch.Tensor) -> torch.Tensor:
        """All-to-all: scatter head dim, gather sequence dim (reverse of above).
        Input:  (B, T*P, H/P, D) — full seq, local heads
        Output: (B, T, H, D) — local seq, all heads
        """
        B, TP, HP, D = x.shape
        P = self.sp_size
        T = TP // P
        # (B, T*P, H//P, D) → (B, P, T, H//P, D) → (P, B, T, H//P, D)
        x = x.view(B, P, T, HP, D)
        x = x.permute(1, 0, 2, 3, 4).contiguous()  # (P, B, T, H//P, D)
        output = torch.empty_like(x)
        dist.all_to_all_single(output, x, group=self.sp_group)
        # (P, B, T, H//P, D) → (B, T, P*H//P, D) = (B, T, H, D)
        output = output.permute(1, 2, 0, 3, 4).contiguous()
        return output.view(B, T, P * HP, D)
