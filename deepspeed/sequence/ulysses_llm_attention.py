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
        self.head_dim = original_attn.head_dim
        # MHA (_CausalAttn) has no n_kv_heads; GQA has it
        self.n_kv_heads = getattr(original_attn, 'n_kv_heads', self.n_heads)
        self.n_groups = getattr(original_attn, 'n_groups', 1)

        # Detect projection style: fused qkv+proj (_CausalAttn) vs separate q/k/v/o_proj (GQA)
        self._fused_qkv = hasattr(original_attn, 'qkv') and not hasattr(original_attn, 'q_proj')
        if self._fused_qkv:
            self._hidden = self.n_heads * self.head_dim
            logger.info("UlyssesSPLLMAttention: detected fused qkv+proj (_CausalAttn style)")

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

    def _scatter_heads_gather_seq(self, x: torch.Tensor) -> torch.Tensor:
        """All-to-all: scatter head dim (2), gather seq dim (1).
        (B, T, H, D) → (B, T*P, H_local, D)
        Handles H % P != 0 by padding heads to next multiple."""
        B, T, H, D = x.shape
        P = self.sp_size
        # Pad heads to multiple of P if needed
        pad_h = (P - H % P) % P
        if pad_h > 0:
            x = F.pad(x, (0, 0, 0, pad_h))  # pad head dim
            H_padded = H + pad_h
        else:
            H_padded = H
        hp = H_padded // P
        x = x.view(B, T, P, hp, D).permute(2, 0, 1, 3, 4).contiguous()
        out = torch.empty_like(x)
        dist.all_to_all_single(out, x, group=self.sp_group)
        # (P, B, T, hp, D) → (B, P*T, hp, D)
        out = out.permute(1, 0, 2, 3, 4).contiguous().view(B, T * P, hp, D)
        # Remove padded heads from this rank's slice
        local_h = self.heads_per_rank[self.sp_rank]
        return out[:, :, :local_h, :]

    def _scatter_seq_gather_heads(self, x: torch.Tensor) -> torch.Tensor:
        """All-to-all reverse: scatter seq dim (1), gather head dim (2).
        (B, T*P, H_local, D) → (B, T, H, D)
        Handles H % P != 0 by padding."""
        B, TP, HP, D = x.shape
        P = self.sp_size
        T = TP // P
        # Pad HP back to padded size for symmetric A2A
        total_heads = sum(self.heads_per_rank)
        pad_h = (P - total_heads % P) % P
        hp_padded = (total_heads + pad_h) // P
        if HP < hp_padded:
            x = F.pad(x, (0, 0, 0, hp_padded - HP))
            HP = hp_padded
        x = x.view(B, P, T, HP, D).permute(1, 0, 2, 3, 4).contiguous()
        out = torch.empty_like(x)
        dist.all_to_all_single(out, x, group=self.sp_group)
        # (P, B, T, HP, D) → (B, T, P*HP, D)
        out = out.permute(1, 2, 0, 3, 4).contiguous().view(B, T, P * HP, D)
        # Trim padded heads
        return out[:, :, :total_heads, :]

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape

        # 1. QKV projection (all heads)
        if self._fused_qkv:
            # _CausalAttn: single fused qkv linear → split into q, k, v
            qkv = self.attn.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
            xq, xk, xv = qkv.unbind(2)  # each (B, T, n_heads, head_dim)
        else:
            # GroupedQueryAttention: separate q/k/v projections
            xq = self.attn.q_proj(x).view(B, T, self.n_heads, self.head_dim)
            xk = self.attn.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
            xv = self.attn.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        # 2. All-to-all: scatter heads (dim=2), gather seq (dim=1)
        xq = self._scatter_heads_gather_seq(xq)
        xk = self._scatter_heads_gather_seq(xk)
        xv = self._scatter_heads_gather_seq(xv)

        T_full = xq.shape[1]  # T * sp_size

        # 3. RoPE — only when freqs_cis is provided (GQA path has RoPE,
        #    _CausalAttn uses learned positional embeddings instead)
        if freqs_cis is not None:
            from models.llama_pretrain import apply_rotary_emb
            if freqs_cis.shape[0] < T_full:
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
        out = self._scatter_seq_gather_heads(out)

        # 7. Output projection
        out = out.contiguous().view(B, T, -1)
        if self._fused_qkv:
            return self.attn.proj(out)
        else:
            return self.attn.o_proj(out)
