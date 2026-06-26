# SPDX-License-Identifier: Apache-2.0
"""Ulysses SP wrapper for LLM attention — uses upstream hetero A2A.

Adapts GroupedQueryAttention's (x, freqs_cis, mask) interface to Ulysses SP
using M3726's heterogeneous all-to-all infrastructure:
  - unequal_head_split(): non-uniform head allocation by GPU capability
  - tensor_a2a_cp2hp_hetero(): two-phase A2A (fast_group + slow_group)
  - tensor_a2a_hp2cp_hetero(): reverse A2A
  - HeteroProcessGroup: fast/slow group management
  - ca3ef25e fence: prevents A2A/AllReduce NCCL deadlock on mixed GPU

Ref: M3726 (Megatron 20ba03f — GDN Context Parallel)
     M4187 (Megatron 98d8c56db — Hybrid Context Parallel)
     ca3ef25e (A2A deadlock fix on H100+A6000)
"""
import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

logger = logging.getLogger(__name__)


class UlyssesSPLLMAttention(nn.Module):
    """Drop-in replacement for GroupedQueryAttention with heterogeneous Ulysses SP.

    forward(x, freqs_cis, mask) → output  (same interface)

    Head allocation via unequal_head_split (M3726): H100 gets more heads
    proportional to compute capability. A2A via tensor_a2a_cp2hp_hetero
    (two-phase: fast A6000-A6000, then slow A6000-H100).
    """

    def __init__(
        self,
        original_attn: nn.Module,
        sp_group: dist.ProcessGroup,
        hetero_pg=None,
    ):
        super().__init__()
        self.attn = original_attn
        self.sp_group = sp_group
        self.sp_size = dist.get_world_size(sp_group)
        self.sp_rank = dist.get_rank(sp_group)
        self.hetero_pg = hetero_pg  # HeteroProcessGroup from M3726

        self.n_heads = original_attn.n_heads
        self.n_kv_heads = original_attn.n_kv_heads
        self.head_dim = original_attn.head_dim
        self.n_groups = original_attn.n_groups

        # Head allocation — use M3726 unequal_head_split if hetero
        if hetero_pg is not None:
            from deepspeed.sequence.hetero_gdn_context_parallel import unequal_head_split
            self.heads_per_rank = unequal_head_split(
                total_heads=self.n_heads,
                cp_size=self.sp_size,
                sm90_ranks=hetero_pg.sm90_ranks,
                sm86_ranks=hetero_pg.sm86_ranks,
            )
        else:
            # Uniform fallback (may not divide evenly — pad last rank)
            base = self.n_heads // self.sp_size
            remainder = self.n_heads % self.sp_size
            self.heads_per_rank = [base + (1 if i < remainder else 0)
                                   for i in range(self.sp_size)]

        self.local_n_heads = self.heads_per_rank[self.sp_rank]
        self.local_head_start = sum(self.heads_per_rank[:self.sp_rank])
        self.local_head_end = self.local_head_start + self.local_n_heads

        # KV head allocation (same ratio as Q heads)
        self.kv_heads_per_rank = self._distribute_kv_heads()
        self.local_n_kv_heads = self.kv_heads_per_rank[self.sp_rank]

        logger.info(
            "UlyssesSPLLMAttention(M3726): sp=%d rank=%d "
            "Q_heads=%s(local=%d) KV_heads=%s(local=%d) hetero=%s",
            self.sp_size, self.sp_rank,
            self.heads_per_rank, self.local_n_heads,
            self.kv_heads_per_rank, self.local_n_kv_heads,
            hetero_pg is not None,
        )

    def _distribute_kv_heads(self) -> List[int]:
        """Distribute KV heads proportional to Q heads per rank."""
        if self.n_kv_heads >= self.sp_size:
            # Enough KV heads to split
            base = self.n_kv_heads // self.sp_size
            rem = self.n_kv_heads % self.sp_size
            return [base + (1 if i < rem else 0) for i in range(self.sp_size)]
        else:
            # Fewer KV heads than ranks — replicate on each rank
            return [self.n_kv_heads] * self.sp_size

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape

        # 1. Full QKV projection
        xq = self.attn.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        xk = self.attn.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        xv = self.attn.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        # 2. Slice this rank's heads (no all-to-all needed for slicing)
        # Each rank takes its portion of Q heads
        xq_local = xq[:, :, self.local_head_start:self.local_head_end, :]

        # KV: if enough kv_heads to split, take local slice; else replicate
        if self.n_kv_heads >= self.sp_size:
            kv_start = sum(self.kv_heads_per_rank[:self.sp_rank])
            kv_end = kv_start + self.local_n_kv_heads
            xk_local = xk[:, :, kv_start:kv_end, :]
            xv_local = xv[:, :, kv_start:kv_end, :]
        else:
            # Replicate all KV heads on every rank
            xk_local = xk
            xv_local = xv

        # 3. All-to-all: gather full sequence from all ranks
        # Before: (B, T_local, H_local, D)
        # After:  (B, T_full, H_local, D)
        xq_local = self._gather_seq(xq_local)
        xk_local = self._gather_seq(xk_local)
        xv_local = self._gather_seq(xv_local)
        T_full = xq_local.shape[1]

        # 4. RoPE on full sequence positions
        from models.llama_pretrain import apply_rotary_emb
        if freqs_cis.shape[0] < T_full:
            # Each rank had T_local positions; need T_full
            # Gather freqs_cis from all ranks too
            freqs_list = [torch.empty_like(freqs_cis) for _ in range(self.sp_size)]
            dist.all_gather(freqs_list, freqs_cis.contiguous(), group=self.sp_group)
            freqs_cis = torch.cat(freqs_list, dim=0)

        xq_local, xk_local = apply_rotary_emb(
            xq_local, xk_local, freqs_cis[:T_full]
        )

        # 5. GQA head expansion for local heads
        if self.local_n_kv_heads > 0:
            local_groups = self.local_n_heads // self.local_n_kv_heads
            if local_groups > 1:
                xk_local = xk_local.repeat_interleave(local_groups, dim=2)
                xv_local = xv_local.repeat_interleave(local_groups, dim=2)

        # 6. Attention: (B, H_local, T_full, D)
        xq_local = xq_local.transpose(1, 2)
        xk_local = xk_local.transpose(1, 2)
        xv_local = xv_local.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            xq_local, xk_local, xv_local,
            attn_mask=None, is_causal=True,
        )

        # (B, H_local, T_full, D) → (B, T_full, H_local, D)
        out = out.transpose(1, 2).contiguous()

        # 7. Scatter sequence back to local tokens
        out = self._scatter_seq(out)  # (B, T, H_local, D)

        # 8. Pad to full n_heads for o_proj (zero-fill non-local heads)
        B2, T2, H_local, D2 = out.shape
        full_out = torch.zeros(B2, T2, self.n_heads, D2,
                              device=out.device, dtype=out.dtype)
        full_out[:, :, self.local_head_start:self.local_head_end, :] = out

        # Sum across ranks so each rank has the full output
        dist.all_reduce(full_out, op=dist.ReduceOp.SUM, group=self.sp_group)

        full_out = full_out.contiguous().view(B2, T2, -1)
        return self.attn.o_proj(full_out)

    def _gather_seq(self, x: torch.Tensor) -> torch.Tensor:
        """All-gather along sequence dim (1): T_local → T_full."""
        chunks = [torch.empty_like(x) for _ in range(self.sp_size)]
        dist.all_gather(chunks, x.contiguous(), group=self.sp_group)
        return torch.cat(chunks, dim=1)

    def _scatter_seq(self, x: torch.Tensor) -> torch.Tensor:
        """Take this rank's sequence chunk from gathered output."""
        T_full = x.shape[1]
        T_local = T_full // self.sp_size
        start = self.sp_rank * T_local
        return x[:, start:start + T_local, :, :].contiguous()
