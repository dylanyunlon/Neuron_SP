# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Attention modules ported from Megatron-LM with AutoSP and DES-LOC support.

DES-LOC integration
-------------------
Each SelfAttention layer carries its ``layer_number`` (1-based global index).
At construction time it queries ``TransformerConfig.get_layer_tier()`` and logs
which GPU tier it belongs to.  The tier does not alter forward-pass logic — that
routing is managed externally by the DES-LOC engine — but the annotation is
available for profiling and dynamic scheduling.

AutoSP (Sequence Parallelism)
------------------------------
When ``config.sequence_parallel`` is True the input hidden_states arrive as
shards of shape ``[seq/sp_size, batch, hidden]``.  SelfAttention uses
all-to-all communication to scatter sequence → head dimension before attention
and gather back afterwards.  See :meth:`SelfAttention._sp_all_to_all_scatter`
and :meth:`SelfAttention._sp_all_to_all_gather`.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_config import TransformerConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy parallel-state helpers (safe when dist not initialised)
# ---------------------------------------------------------------------------

def _get_sp_world_size() -> int:
    try:
        from deepspeed.core.parallel_state import get_sequence_parallel_world_size
        return get_sequence_parallel_world_size()
    except Exception:
        return 1


def _get_sp_rank() -> int:
    try:
        from deepspeed.core.parallel_state import get_sequence_parallel_rank
        return get_sequence_parallel_rank()
    except Exception:
        return 0


def _get_sp_group():
    try:
        from deepspeed.core.parallel_state import get_sequence_parallel_group
        return get_sequence_parallel_group()
    except Exception:
        return None


def _get_tp_world_size() -> int:
    try:
        from deepspeed.core.parallel_state import get_tensor_model_parallel_world_size
        return get_tensor_model_parallel_world_size()
    except Exception:
        return 1


def _get_tp_group():
    try:
        from deepspeed.core.parallel_state import get_tensor_model_parallel_group
        return get_tensor_model_parallel_group()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Base MegatronModule (imported here to avoid circular deps)
# ---------------------------------------------------------------------------

from deepspeed.core.transformer.module import MegatronModule  # noqa: E402


# ---------------------------------------------------------------------------
# Abstract base attention class
# ---------------------------------------------------------------------------

class Attention(MegatronModule, ABC):
    """Base attention class.

    AutoSP integration: when sequence_parallel is enabled, the input
    sequence is already partitioned across SP ranks. The attention
    computation handles the local chunk and uses A2A for KV exchange.

    DES-LOC integration: each concrete subclass logs its tier at init time.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: str = "causal",
    ) -> None:
        super().__init__(config)
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type

        assert config.kv_channels is not None, "kv_channels must be set in TransformerConfig"
        assert config.num_attention_heads is not None
        assert config.num_query_groups is not None

        self.hidden_size_per_attention_head: int = config.kv_channels
        self.num_attention_heads: int = config.num_attention_heads
        self.num_query_groups: int = config.num_query_groups

        # Full (un-TP-split) projection sizes
        self.query_projection_size: int = config.kv_channels * config.num_attention_heads
        self.kv_projection_size: int = config.kv_channels * config.num_query_groups

        # DES-LOC: log which tier this attention head lives on
        tier = config.get_layer_tier(self.layer_number - 1)  # 0-based
        if tier is not None:
            logger.debug(
                "Attention layer %d assigned to tier: %s",
                self.layer_number,
                tier.upper(),
            )

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        inference_params: Optional[object] = None,
    ) -> torch.Tensor:
        """Compute attention and return output of the same shape as input."""
        ...


# ---------------------------------------------------------------------------
# Scaled dot-product attention kernel
# ---------------------------------------------------------------------------

class DotProductAttention(MegatronModule):
    """Scaled dot-product attention with FlashAttention-2 dispatch.

    Uses :func:`torch.nn.functional.scaled_dot_product_attention` (SDPA)
    which dispatches to FlashAttention when hardware / software supports it.

    Tensor layout throughout: ``[seq, batch, num_heads, head_dim]``.

    Supports GQA via head expansion (repeat_interleave) for backends that
    don't natively handle grouped keys/values.
    """

    def __init__(self, config: TransformerConfig, layer_number: int) -> None:
        super().__init__(config)
        self.layer_number = max(1, layer_number)
        self.softmax_scale: float = 1.0 / math.sqrt(config.kv_channels)
        self.attn_dropout_p: float = config.attention_dropout
        self.num_attention_heads: int = config.num_attention_heads
        self.num_query_groups: int = config.num_query_groups

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute scaled dot-product attention.

        Args:
            query: ``[seq_q, batch, num_heads, head_dim]``
            key:   ``[seq_k, batch, num_kv_heads, head_dim]``
            value: ``[seq_k, batch, num_kv_heads, head_dim]``
            attention_mask: Optional additive or boolean mask
                ``[batch, 1, seq_q, seq_k]``.  Boolean True = keep (SDPA convention).

        Returns:
            context: ``[seq_q, batch, num_heads * head_dim]``
        """
        seq_q, batch, num_heads, head_dim = query.shape
        num_kv_heads = key.shape[2]

        # GQA: expand KV heads to match Q heads
        if num_kv_heads != num_heads:
            expand_factor = num_heads // num_kv_heads
            key = key.repeat_interleave(expand_factor, dim=2)
            value = value.repeat_interleave(expand_factor, dim=2)

        # [sq, b, nh, hd] → [b, nh, sq, hd]
        q = query.permute(1, 2, 0, 3).contiguous()
        k = key.permute(1, 2, 0, 3).contiguous()
        v = value.permute(1, 2, 0, 3).contiguous()

        is_causal = attention_mask is None

        attn_mask_sdpa: Optional[torch.Tensor] = None
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                # Megatron convention: True = mask out; SDPA: True = keep → invert
                attn_mask_sdpa = ~attention_mask
            else:
                attn_mask_sdpa = attention_mask

        dropout_p = self.attn_dropout_p if self.training else 0.0

        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask_sdpa,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=self.softmax_scale,
        )
        # context: [b, nh, sq, hd] → [sq, b, nh*hd]
        context = context.permute(2, 0, 1, 3).contiguous()
        context = context.view(seq_q, batch, num_heads * head_dim)
        return context


# ---------------------------------------------------------------------------
# Standard multi-head self-attention
# ---------------------------------------------------------------------------

class SelfAttention(Attention):
    """Multi-head self-attention with QKV projection.

    Layout: ``[seq, batch, hidden]`` throughout (Megatron convention).

    AutoSP integration:
      When config.sequence_parallel is True the *input* hidden_states
      arrive as a sequence-parallel shard of shape
      ``[seq/sp_size, batch, hidden]``.  After the output projection the
      result is gathered back to the full sequence before returning.
      The all-to-all (A2A) pattern redistributes tokens so that each
      SP rank owns a contiguous block during attention computation.

    DES-LOC integration:
      The tier annotation from the base :class:`Attention` is preserved.
      The layer tag can be used by DES-LOC engine to route the module to
      the correct device pool.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
    ) -> None:
        super().__init__(config, layer_number, attn_mask_type="causal")

        tp_size = _get_tp_world_size()

        # --- QKV projection -------------------------------------------
        qkv_out_dim = self.query_projection_size + 2 * self.kv_projection_size
        self.qkv_out_dim_per_tp = qkv_out_dim // tp_size
        self.q_size_per_tp = self.query_projection_size // tp_size
        self.kv_size_per_tp = self.kv_projection_size // tp_size

        self.num_heads_per_tp = config.num_attention_heads // tp_size
        self.num_kv_heads_per_tp = max(1, config.num_query_groups // tp_size)

        self.qkv_proj = nn.Linear(
            config.hidden_size,
            self.qkv_out_dim_per_tp,
            bias=config.add_bias_linear,
        )
        # Mark TP-sharded so checkpoint code knows the partition dim
        self.qkv_proj.weight.tensor_model_parallel = True
        self.qkv_proj.weight.partition_dim = 0

        # --- Core attention -------------------------------------------
        self.core_attention = DotProductAttention(config, layer_number)

        # --- Output projection -----------------------------------------
        self.out_proj = nn.Linear(
            self.query_projection_size // tp_size,
            config.hidden_size,
            bias=config.add_bias_linear,
        )
        self.out_proj.weight.tensor_model_parallel = True
        self.out_proj.weight.partition_dim = 1

        self.attn_dropout = nn.Dropout(p=config.attention_dropout)

    # ------------------------------------------------------------------
    # AutoSP all-to-all helpers
    # ------------------------------------------------------------------

    def _sp_all_to_all_scatter(
        self, hidden_states: torch.Tensor, sp_group
    ) -> torch.Tensor:
        """Scatter sequence dim → head dim for SP before attention.

        Input:  ``[seq/sp,  batch, hidden]``
        Output: ``[seq,     batch, hidden/sp]``

        Uses all-to-all: each rank sends ``seq/sp`` tokens, receives from
        all ranks, then transposes so local heads are contiguous.
        """
        sp_size = _get_sp_world_size()
        if sp_size == 1 or sp_group is None:
            return hidden_states

        seq_chunk, batch, hidden = hidden_states.shape
        assert hidden % sp_size == 0, (
            f"hidden {hidden} not divisible by sp_size {sp_size}"
        )
        hidden_per_rank = hidden // sp_size

        inp = hidden_states.view(seq_chunk, batch, sp_size, hidden_per_rank)
        inp = inp.permute(2, 0, 1, 3).contiguous()  # [sp, sc, b, h/sp]

        out_list = [torch.empty_like(inp[0]) for _ in range(sp_size)]
        in_list = list(inp.unbind(0))
        torch.distributed.all_to_all(out_list, in_list, group=sp_group)

        out = torch.stack(out_list, dim=0)           # [sp, sc, b, h/sp]
        out = out.permute(1, 0, 2, 3).reshape(
            seq_chunk * sp_size, batch, hidden_per_rank
        )
        return out

    def _sp_all_to_all_gather(
        self, context: torch.Tensor, sp_group
    ) -> torch.Tensor:
        """Gather head dim → sequence dim for SP after attention.

        Inverse of :meth:`_sp_all_to_all_scatter`.

        Input:  ``[seq,     batch, hidden/sp]``
        Output: ``[seq/sp,  batch, hidden]``
        """
        sp_size = _get_sp_world_size()
        if sp_size == 1 or sp_group is None:
            return context

        total_seq, batch, hidden_per_rank = context.shape
        seq_chunk = total_seq // sp_size

        ctx = context.reshape(sp_size, seq_chunk, batch, hidden_per_rank)
        ctx = ctx.permute(1, 0, 2, 3).contiguous()

        inp_list = list(ctx.unbind(1))
        out_list = [torch.empty_like(inp_list[0]) for _ in range(sp_size)]
        torch.distributed.all_to_all(out_list, inp_list, group=sp_group)

        out = torch.stack(out_list, dim=1)
        out = out.permute(0, 2, 1, 3).reshape(
            seq_chunk, batch, hidden_per_rank * sp_size
        )
        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        inference_params: Optional[object] = None,
    ) -> torch.Tensor:
        """Self-attention forward pass.

        Args:
            hidden_states: ``[seq, batch, hidden]``
                (or ``[seq/sp, batch, hidden]`` in SP mode).
            attention_mask: Optional boolean mask ``[batch, 1, seq, seq]``.
            rotary_pos_emb: Optional rotary position embeddings.
            inference_params: Inference state (currently unused).

        Returns:
            output: same shape as *hidden_states*.
        """
        sp_group = _get_sp_group()
        use_sp = (
            getattr(self.config, "sequence_parallel", False)
            and _get_sp_world_size() > 1
        )

        # --- AutoSP A2A scatter: seq/sp → seq, hidden → hidden/sp --------
        if use_sp:
            hidden_states = self._sp_all_to_all_scatter(hidden_states, sp_group)

        # --- QKV projection ----------------------------------------------
        mixed_qkv = self.qkv_proj(hidden_states)  # [seq, batch, q+2kv per tp]
        seq_len, batch, _ = mixed_qkv.shape
        head_dim = self.hidden_size_per_attention_head

        q, k, v = torch.split(
            mixed_qkv,
            [self.q_size_per_tp, self.kv_size_per_tp, self.kv_size_per_tp],
            dim=-1,
        )

        q = q.view(seq_len, batch, self.num_heads_per_tp, head_dim)
        k = k.view(seq_len, batch, self.num_kv_heads_per_tp, head_dim)
        v = v.view(seq_len, batch, self.num_kv_heads_per_tp, head_dim)

        # --- Rotary position embeddings ----------------------------------
        if rotary_pos_emb is not None:
            q, k = self._apply_rotary_emb(q, k, rotary_pos_emb)

        # --- Core attention ----------------------------------------------
        context = self.core_attention(q, k, v, attention_mask)

        # --- Output projection -------------------------------------------
        output = self.out_proj(context)  # [seq, batch, hidden]

        # All-reduce across TP ranks (RowParallelLinear pattern)
        tp_size = _get_tp_world_size()
        if tp_size > 1:
            tp_group = _get_tp_group()
            if tp_group is not None:
                torch.distributed.all_reduce(output, group=tp_group)

        # --- AutoSP A2A gather: seq → seq/sp, hidden/sp → hidden ---------
        if use_sp:
            output = self._sp_all_to_all_gather(output, sp_group)

        return output

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_rotary_emb(
        q: torch.Tensor, k: torch.Tensor, rotary_pos_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to Q and K.

        Args:
            q: ``[seq, batch, num_heads, head_dim]``
            k: ``[seq, batch, num_kv_heads, head_dim]``
            rotary_pos_emb: ``[seq, 1, 1, head_dim]``

        Returns:
            Rotated q and k tensors with the same shapes.
        """
        dim = q.shape[-1]
        half = dim // 2
        cos_emb = rotary_pos_emb[..., :half]
        sin_emb = rotary_pos_emb[..., half:]

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x[..., :half], x[..., half:]
            return torch.cat((-x2, x1), dim=-1)

        q_rot = q * cos_emb + rotate_half(q) * sin_emb
        k_rot = (
            k * cos_emb[: k.shape[0]] + rotate_half(k) * sin_emb[: k.shape[0]]
        )
        return q_rot, k_rot
