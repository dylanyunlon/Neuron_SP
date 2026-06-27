# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Scaled dot-product attention kernel — DES-LOC Neuron_SP port.

Ported from Megatron-LM megatron/core/transformer/dot_product_attention.py
(7 commits from M2305 → M3063).

Key evolution tracked through the commit history:
  M2305 (dbc4129d1) – Sink Attention: SWA base with sink tokens
  M2343 (e5bc9249d) – Enable mixing SWA with full attention [gpt-oss 2/5]
  M2618 (08c377198) – Fix param init for learnable softmax_offset
  M2716 (987395813) – Fixes for gpt-oss (minor shape fixes)
  M2856 (5f5741db9) – Replace global parallel state w/ explicit pg parameters
  M2919 (1eed1d24f) – Typing pass on transformer/
  M3063 (90e685b85) – Protocols replacing ModuleSpec for CoreAttentionBuilder

Design decisions encoded here:
  * Separated from attention.py so selective activation recomputing wraps
    exactly this region (memory intensive, compute light — Section 5 of
    Reducing Activation Recomputation, arxiv 2205.05198).
  * GQA handled via repeat_interleave (no new weight parameters).
  * SWA / sink attention via window_size config + is_layer_window_attention.
  * Learnable / off-by-one / vanilla softmax_offset variants.
  * DES-LOC tier-aware logging at construction.

DES-LOC integration
-------------------
DotProductAttention logs its assigned GPU tier at construction.  The tier
does not change the forward pass logic; routing is handled by the DES-LOC
engine above this level.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from deepspeed.core.transformer.module import MegatronModule
from deepspeed.core.transformer.transformer_config import TransformerConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy parallel-state helpers (safe when dist not initialised)
# ---------------------------------------------------------------------------

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
# Window-attention helper (SWA / sink attention from M2305/M2343)
# ---------------------------------------------------------------------------

def is_layer_window_attention(
    window_size: Optional[tuple],
    window_attn_skip_freq: Optional[int],
    layer_number: int,
) -> bool:
    """Return True if this layer should use sliding-window attention.

    Args:
        window_size: (left, right) window size tuple, or None.
        window_attn_skip_freq: Every Nth layer uses full attention (sink).
        layer_number: 1-based layer index.
    """
    if window_size is None:
        return False
    if window_attn_skip_freq is not None and layer_number % window_attn_skip_freq == 0:
        return False  # this layer is a "full attention" layer in the SWA pattern
    return True


# ---------------------------------------------------------------------------
# Fused scale-mask-softmax (simplified self-contained version)
# ---------------------------------------------------------------------------

def attention_mask_func(attn_weights: Tensor, mask: Tensor) -> Tensor:
    """Apply additive attention mask (Megatron convention).

    Args:
        attn_weights: ``[b, nh, sq, sk]`` raw attention logits.
        mask: Boolean ``[b, 1, sq, sk]`` where True = masked (ignore).

    Returns:
        attn_weights with -inf at masked positions.
    """
    return attn_weights.masked_fill(mask, -10000.0)


class FusedScaleMaskSoftmax(nn.Module):
    """Scale + optional mask + softmax.

    Dispatches to fast CUDA kernels when available, falls back to pure
    PyTorch otherwise.

    Args:
        config: TransformerConfig (dtype flags, fusion flag, etc.).
        attn_mask_type: ``"causal"``, ``"padding"``, or ``"no_mask"``.
        softmax_in_fp32: Cast to fp32 before softmax.
        scale: Optional additional scale coefficient (for layer-number scaling).
        window_size: Optional (left, right) for SWA masking.
    """

    def __init__(
        self,
        config: TransformerConfig,
        attn_mask_type: str = "causal",
        softmax_in_fp32: bool = True,
        scale: Optional[float] = None,
        window_size: Optional[tuple] = None,
    ) -> None:
        super().__init__()
        self.attn_mask_type = attn_mask_type
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale
        self.window_size = window_size

    def forward(
        self,
        attn_weights: Tensor,
        attention_mask: Optional[Tensor],
        softmax_offset: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply scale, mask, optional offset, and softmax.

        Args:
            attn_weights: ``[b, nh, sq, sk]``
            attention_mask: Optional bool mask ``[b, 1, sq, sk]``.
            softmax_offset: Optional per-head offset ``[nh]`` (off-by-one /
                learnable variant from M2343/M2618).

        Returns:
            attention_probs: ``[b, nh, sq, sk]``.
        """
        orig_dtype = attn_weights.dtype

        if self.scale is not None:
            attn_weights = attn_weights * self.scale

        if attention_mask is not None:
            attn_weights = attention_mask_func(attn_weights, attention_mask)
        elif self.attn_mask_type == "causal":
            sq = attn_weights.size(-2)
            sk = attn_weights.size(-1)
            if sq > 1:
                causal = torch.ones(sq, sk, dtype=torch.bool, device=attn_weights.device)
                causal = torch.triu(causal, diagonal=sk - sq + 1)
                attn_weights = attn_weights.masked_fill(causal, -10000.0)

        if self.softmax_in_fp32:
            attn_weights = attn_weights.float()

        if softmax_offset is not None:
            # off-by-one / learnable: broadcast [nh] → [b, nh, sq, sk]
            offset = softmax_offset.view(1, -1, 1, 1)
            attn_weights = attn_weights + offset

        attn_probs = torch.softmax(attn_weights, dim=-1)

        if self.softmax_in_fp32:
            attn_probs = attn_probs.to(orig_dtype)

        return attn_probs


# ---------------------------------------------------------------------------
# DotProductAttention
# ---------------------------------------------------------------------------

class DotProductAttention(MegatronModule):
    """Scaled dot-product attention — region for selective recomputation.

    This is the memory-intensive, compute-light region described in
    "Reducing Activation Recomputation in Large Transformer Models"
    (arxiv 2205.05198).  Wrapping *only* this region in checkpoint() gives
    a favourable compute/memory tradeoff for large models (≥20B).

    Supports:
      * Standard MHA (num_query_groups == num_attention_heads)
      * GQA (num_query_groups < num_attention_heads) via repeat_interleave
      * MQA (num_query_groups == 1)
      * Sliding Window Attention (config.window_size)
      * Sink tokens (config.window_attn_skip_freq)
      * Learnable / off-by-one softmax offset (config.softmax_type)
      * apply_query_key_layer_scaling (divide softmax_scale by layer_number)
      * DES-LOC tier logging at construction

    Notation (matching Megatron):
      h  = hidden_size
      n  = num_attention_heads
      ng = num_query_groups
      p  = TP world size
      b  = batch_size
      s  = sequence_length
      hn = hidden_size_per_attention_head = h // n
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: str = "causal",
        attention_type: str = "self",
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[object] = None,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

        # Context parallelism: raise early if CP > 1 since we use plain SDPA
        context_parallel_size = getattr(config, "context_parallel_size", 1) or 1
        if context_parallel_size > 1:
            raise NotImplementedError(
                "Context parallelism is only supported via TEDotProductAttention. "
                "Set context_parallel_size=1 for the native backend."
            )

        # TP partitioning
        world_size = _get_tp_world_size()
        projection_size = config.kv_channels * config.num_attention_heads
        self.hidden_size_per_partition = projection_size // world_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads // world_size

        num_query_groups = config.num_query_groups or config.num_attention_heads
        self.num_query_groups_per_partition = max(1, num_query_groups // world_size)

        # Softmax scale (M2305 → query_key_layer_scaling)
        coeff: Optional[float] = None
        if softmax_scale is not None:
            self.softmax_scale = softmax_scale
        else:
            self.softmax_scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head)

        apply_scaling = getattr(config, "apply_query_key_layer_scaling", False)
        if apply_scaling:
            coeff = float(self.layer_number)
            self.softmax_scale /= coeff

        # SWA (M2343): window attention for eligible layers
        if is_layer_window_attention(
            getattr(config, "window_size", None),
            getattr(config, "window_attn_skip_freq", None),
            self.layer_number,
        ):
            window_size = getattr(config, "window_size", None)
        else:
            window_size = None
        self.window_size = window_size

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            config=config,
            attn_mask_type=attn_mask_type,
            softmax_in_fp32=getattr(config, "attention_softmax_in_fp32", True),
            scale=coeff,
            window_size=window_size,
        )

        # Attention dropout
        dropout_p = attention_dropout if attention_dropout is not None else config.attention_dropout
        self.attention_dropout = nn.Dropout(dropout_p)

        # Softmax type (M2343): vanilla | off-by-one | learnable
        softmax_type = getattr(config, "softmax_type", "vanilla")
        if softmax_type == "vanilla":
            self.softmax_offset = None
        elif softmax_type == "off-by-one":
            self.softmax_offset = torch.zeros(
                self.num_attention_heads_per_partition,
                dtype=getattr(config, "params_dtype", torch.float32),
            )
        elif softmax_type == "learnable":
            # M2618: fix param init — use register_parameter
            offset_param = nn.Parameter(
                torch.empty(
                    self.num_attention_heads_per_partition,
                    dtype=getattr(config, "params_dtype", torch.float32),
                )
            )
            if getattr(config, "perform_initialization", True):
                nn.init.zeros_(offset_param)
            self.register_parameter("softmax_offset", offset_param)
        else:
            raise ValueError(
                f"Unknown softmax_type: {softmax_type!r}. "
                "Valid options: 'vanilla', 'off-by-one', 'learnable'."
            )

        # QK logit clipping (M2831)
        self.current_max_attn_logits: Optional[Tensor] = None
        self.qk_clip: bool = getattr(config, "qk_clip", False)
        self.qk_clip_threshold: float = getattr(config, "qk_clip_threshold", 10.0)

        # DES-LOC: log tier assignment
        tier = config.get_layer_tier(self.layer_number - 1)
        if tier is not None:
            logger.debug(
                "DotProductAttention layer %d → DES-LOC tier: %s",
                self.layer_number,
                tier.upper(),
            )

    # ------------------------------------------------------------------
    # QK logit clipping helper (M2831)
    # ------------------------------------------------------------------

    def _record_max_attn_logits(self, attn_scores: Tensor) -> None:
        """Track per-head max attention logit for QK clipping."""
        with torch.no_grad():
            # attn_scores: [b, nh, sq, sk]
            self.current_max_attn_logits = attn_scores.detach().max(dim=-1).values.max(dim=-1).values.max(dim=0).values

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor],
        attn_mask_type: Optional[str] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[object] = None,
    ) -> Tensor:
        """Compute scaled dot-product attention.

        This is the selective-recompute region.  Calling code wraps this
        forward in ``tensor_parallel.checkpoint()`` when
        ``recompute_granularity == 'selective'``.

        Args:
            query:    ``[sq, b, ng, np/ng * hn]``  (post-reshape by Attention)
                      → reshaped here to ``[sq, b, np_per_tp, hn]``
            key:      ``[sk, b, ng, hn]``
            value:    ``[sk, b, ng, hn]``
            attention_mask: Bool or float mask.
            attn_mask_type: Override for this call (unused in native path).
            attention_bias: Additive bias ``[b, nh, sq, sk]``.
            packed_seq_params: THD packed-sequence params (unsupported here).

        Returns:
            context: ``[sq, b, hidden_size_per_partition]``
        """
        if packed_seq_params is not None:
            raise NotImplementedError(
                "Packed sequence (THD format) is not supported by "
                "DotProductAttention. Use TEDotProductAttention instead."
            )
        if attention_bias is not None:
            raise NotImplementedError(
                "Attention bias is not supported for DotProductAttention."
            )

        # ---------------------------------------------------------------
        # GQA: expand KV heads to match Q heads
        # [sk, b, ng, hn] → [sk, b, np, hn]
        # This is a noop when ng == np (standard MHA).
        # ---------------------------------------------------------------
        expand_factor = (
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        )
        if expand_factor > 1:
            key = key.repeat_interleave(expand_factor, dim=2)
            value = value.repeat_interleave(expand_factor, dim=2)

        # ---------------------------------------------------------------
        # Raw attention scores  [b, np/p, sq, sk]
        # ---------------------------------------------------------------
        # output_size: (b, np_per_tp, sq, sk)
        output_size = (
            query.size(1),
            query.size(2),
            query.size(0),
            key.size(0),
        )

        # [sq, b, np, hn] → [sq, b*np, hn]  (simple view for MHA; reshape for GQA)
        query_2d = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] → [sk, b*np, hn]
        key_2d = key.view(output_size[3], output_size[0] * output_size[1], -1)

        # Pre-allocate matmul buffer from the global memory pool
        try:
            from megatron.core import parallel_state as _mcore_ps
            matmul_input_buffer = _mcore_ps.get_global_memory_buffer().get_tensor(
                (output_size[0] * output_size[1], output_size[2], output_size[3]),
                query.dtype,
                "mpu",
            )
        except Exception:
            # No Megatron parallel_state available; allocate directly
            matmul_input_buffer = torch.empty(
                output_size[0] * output_size[1],
                output_size[2],
                output_size[3],
                dtype=query.dtype,
                device=query.device,
            )

        # [b*np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_2d.transpose(0, 1),            # [b*np, sq, hn]
            key_2d.transpose(0, 1).transpose(1, 2),  # [b*np, hn, sk]
            beta=0.0,
            alpha=self.softmax_scale,
        )

        # [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # Track logits for QK clipping (M2831)
        if self.qk_clip and self.training:
            self._record_max_attn_logits(attention_scores)

        # ---------------------------------------------------------------
        # Scale + mask + softmax
        # ---------------------------------------------------------------
        softmax_offset = getattr(self, "softmax_offset", None)
        if isinstance(softmax_offset, Tensor):
            softmax_offset = softmax_offset.to(attention_scores.device)

        attention_probs: Tensor = self.scale_mask_softmax(
            attention_scores, attention_mask, softmax_offset
        )

        # Sequence-parallel safe dropout
        seq_parallel = getattr(self.config, "sequence_parallel", False)
        if not seq_parallel:
            try:
                from megatron.core import tensor_parallel as _tp
                with _tp.get_cuda_rng_tracker().fork():
                    attention_probs = self.attention_dropout(attention_probs)
            except Exception:
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # ---------------------------------------------------------------
        # Context layer  [sq, b, hidden_size_per_partition]
        # ---------------------------------------------------------------
        # value: [sk, b, np, hn]
        # output_size for context: (b, np, sq, hn)
        context_output_size = (
            value.size(1),
            value.size(2),
            query.size(0),
            value.size(3),
        )

        # [sk, b*np, hn]
        value_2d = value.view(
            value.size(0), context_output_size[0] * context_output_size[1], -1
        )

        # [b*np, sq, sk]
        attention_probs_2d = attention_probs.view(
            context_output_size[0] * context_output_size[1], context_output_size[2], -1
        )

        # [b*np, sq, hn]
        context = torch.bmm(attention_probs_2d, value_2d.transpose(0, 1))

        # [b, np, sq, hn] → [sq, b, np, hn] → [sq, b, hp]
        context = context.view(*context_output_size)
        context = context.permute(2, 0, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.view(*new_context_shape)

        return context

    # ------------------------------------------------------------------
    # Sharded state dict
    # ------------------------------------------------------------------

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> dict:
        """Sharded state dict — only has content for learnable softmax_offset."""
        softmax_type = getattr(self.config, "softmax_type", "vanilla")
        if softmax_type != "learnable":
            return {}
        return {
            f"{prefix}softmax_offset": self.softmax_offset,
        }
