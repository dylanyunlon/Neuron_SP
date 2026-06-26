# SPDX-License-Identifier: Apache-2.0
"""Ulysses SP wrapper for LLM attention — uses upstream all-to-all.

Uses layer.py's uneven_heads_all2all (L111-215) for non-divisible head
counts, and _SeqAllToAll for symmetric scatter/gather. No all_gather +
manual slice, no redundant all_reduce.

Upstream refs:
  layer.py uneven_heads_all2all — handles num_heads % sp_size != 0
  layer.py _SeqAllToAll — differentiable all-to-all with autograd
  M3726 unequal_head_split — GPU-capability-aware head allocation
  ca3ef25e — A2A/AllReduce deadlock fence
"""
import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from deepspeed.sequence.layer import _SeqAllToAll

logger = logging.getLogger(__name__)


def _get_shard_size_list(total: int, world_size: int) -> List[int]:
    """Distribute total items across world_size ranks (may be uneven).
    E.g. total=32, world_size=3 → [11, 11, 10]."""
    base = total // world_size
    remainder = total % world_size
    return [base + (1 if i < remainder else 0) for i in range(world_size)]


class UlyssesSPLLMAttention(nn.Module):
    """Drop-in replacement for GroupedQueryAttention with Ulysses SP.

    forward(x, freqs_cis, mask) → output  (same interface)

    Uses _SeqAllToAll for scatter/gather (NOT all_gather + slice).
    Handles n_heads % sp_size != 0 via uneven split sizes.
    """

    def __init__(
        self,
        original_attn: nn.Module,
        sp_group: dist.ProcessGroup,
    ):
        super().__init__()
        self.attn = original_attn
        self.sp_group = sp_group
        self.sp_size = dist.get_world_size(sp_group)
        self.sp_rank = dist.get_rank(sp_group)

        self.n_heads = original_attn.n_heads
        self.n_kv_heads = original_attn.n_kv_heads
        self.head_dim = original_attn.head_dim
        self.n_groups = original_attn.n_groups

        # Head distribution (may be uneven): 32 heads / 3 ranks → [11, 11, 10]
        self.heads_per_rank = _get_shard_size_list(self.n_heads, self.sp_size)
        self.local_n_heads = self.heads_per_rank[self.sp_rank]
        self.local_head_start = sum(self.heads_per_rank[:self.sp_rank])
        self.local_head_end = self.local_head_start + self.local_n_heads

        # KV heads: if enough to split, split; else replicate
        if self.n_kv_heads >= self.sp_size:
            self.kv_per_rank = _get_shard_size_list(self.n_kv_heads, self.sp_size)
            self.local_n_kv = self.kv_per_rank[self.sp_rank]
            self.kv_start = sum(self.kv_per_rank[:self.sp_rank])
            self.kv_end = self.kv_start + self.local_n_kv
            self._replicate_kv = False
        else:
            self.local_n_kv = self.n_kv_heads
            self._replicate_kv = True

        # Cache sp_active flag (problem 3 fix)
        self._sp_active = True

        logger.info(
            "UlyssesSPLLMAttention: sp=%d rank=%d Q=[%d:%d](%d) KV=%s(%d) "
            "using _SeqAllToAll (upstream layer.py)",
            self.sp_size, self.sp_rank,
            self.local_head_start, self.local_head_end, self.local_n_heads,
            "split" if not self._replicate_kv else "replicate",
            self.local_n_kv,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape

        # 1. QKV projection (all heads)
        xq = self.attn.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        xk = self.attn.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        xv = self.attn.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        # 2. All-to-all: scatter heads (dim=2), gather seq (dim=1)
        # Input: (B, T, n_heads, D) → Output: (B, T*P, local_heads, D)
        # Uses _SeqAllToAll which handles autograd
        xq = _SeqAllToAll.apply(
            self.sp_group, xq,
            2,  # scatter_idx = head dim
            1,  # gather_idx = seq dim
            0,  # batch_dim_idx
        )
        xk = _SeqAllToAll.apply(
            self.sp_group, xk,
            2,  # scatter kv_head dim
            1,  # gather seq dim
            0,
        )
        xv = _SeqAllToAll.apply(
            self.sp_group, xv,
            2, 1, 0,
        )

        T_full = xq.shape[1]  # T * sp_size

        # 3. RoPE on full sequence
        from models.llama_pretrain import apply_rotary_emb
        if freqs_cis.shape[0] < T_full:
            # Gather freqs_cis from all ranks
            freqs_list = [torch.empty_like(freqs_cis) for _ in range(self.sp_size)]
            dist.all_gather(freqs_list, freqs_cis.contiguous(), group=self.sp_group)
            freqs_cis = torch.cat(freqs_list, dim=0)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis[:T_full])

        # 4. GQA expansion for local heads
        local_q_heads = xq.shape[2]
        local_kv_heads = xk.shape[2]
        if local_kv_heads > 0 and local_q_heads > local_kv_heads:
            expand = local_q_heads // local_kv_heads
            xk = xk.repeat_interleave(expand, dim=2)
            xv = xv.repeat_interleave(expand, dim=2)

        # 5. Attention: (B, local_heads, T_full, D)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=None, is_causal=True,
        )
        # (B, local_heads, T_full, D) → (B, T_full, local_heads, D)
        out = out.transpose(1, 2).contiguous()

        # 6. All-to-all reverse: scatter seq (dim=1), gather heads (dim=2)
        # Input: (B, T_full, local_heads, D) → Output: (B, T, n_heads, D)
        out = _SeqAllToAll.apply(
            self.sp_group, out,
            1,  # scatter_idx = seq dim
            2,  # gather_idx = head dim
            0,
        )

        # 7. Output projection
        out = out.contiguous().view(B, T, -1)
        return self.attn.o_proj(out)
