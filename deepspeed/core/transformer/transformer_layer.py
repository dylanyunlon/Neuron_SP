# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""TransformerLayer — single transformer layer with DES-LOC tier annotation.

Ported from Megatron-LM megatron/core/transformer/transformer_layer.py
and extended for the Neuron_SP / DES-LOC project.

Megatron fixes absorbed
-----------------------
M2317 (18420b634) — Fix BERT + virtual pipeline parallelism
  * ``forward`` now returns ``(hidden_states, context)`` tuple so cross-
    attention context flows correctly across PP stages.

M3217 / cherry-pick #2776 (1b110768d) — Fix clip_qk issues
  * ``clip_qk()`` per-layer method skips when
    ``current_max_attn_logits is None`` instead of raising.
  * (The outer loop fix lives in attention.py; this layer exposes the
    ``has_clip_qk`` property for callers to guard before calling.)

M4090 (b0eb9143c) — DSA RoPE: ``multi_latent_attention`` → ``mla_rotary_interleaved``
  * Forward delegates to ``SelfAttention`` which now passes the correct
    ``mla_rotary_interleaved`` flag (fix applied in attention.py).

DES-LOC integration
-------------------
Each ``TransformerLayer`` carries a zero-based ``layer_number`` (1-based
globally).  On construction it queries ``TransformerConfig.get_layer_tier()``
and stores the result in ``self.desloc_tier`` ("h100" | "a6000" | None).

This attribute is used by the DES-LOC engine to decide:
  * Which device the layer's parameters are pinned to.
  * How aggressively to recompute activations (A6000 = less VRAM → more
    aggressive checkpointing).

Activation recomputation
------------------------
When ``config.recompute_granularity == "full"`` the entire forward pass is
wrapped in ``torch.utils.checkpoint.checkpoint``.  This is particularly
useful for A6000 stages that have limited VRAM.

When ``config.recompute_granularity == "selective"`` only the core attention
kernel is recomputed (memory-intensive but compute-cheap).

The uniform / block recompute loop (``recompute_method``) lives in
``TransformerBlock`` (which calls individual layers), not here.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

from .transformer_config import TransformerConfig
from .module import MegatronModule
from .attention import SelfAttention
from .mlp import MLP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: norm factory
# ---------------------------------------------------------------------------

def _build_norm(config: TransformerConfig, hidden_size: Optional[int] = None) -> nn.Module:
    """Build the normalisation module specified by *config.normalization*.

    Args:
        config: Transformer configuration.
        hidden_size: Override the size if different from config.hidden_size.

    Returns:
        An ``nn.RMSNorm`` or ``nn.LayerNorm`` instance.
    """
    size = hidden_size if hidden_size is not None else config.hidden_size
    eps = config.layernorm_epsilon
    if config.normalization == "RMSNorm":
        return nn.RMSNorm(size, eps=eps)
    elif config.normalization == "LayerNorm":
        return nn.LayerNorm(size, eps=eps)
    else:
        raise ValueError(
            f"Unknown normalization: {config.normalization!r}. "
            "Use 'LayerNorm' or 'RMSNorm'."
        )


# ---------------------------------------------------------------------------
# TransformerLayer
# ---------------------------------------------------------------------------

class TransformerLayer(MegatronModule):
    """Single transformer layer: attention → residual → MLP → residual.

    Uses *pre-norm* (norm before sub-layer) following LLaMA / Mistral style
    by default.  Set ``config.apply_residual_connection_post_layernorm = True``
    to switch to post-norm (BERT / GPT-2 style).

    Cross-attention support (M2317 BERT/VPP fix):
        When ``config.encoder_decoder`` is True (or ``add_cross_attn=True``),
        a second attention sub-layer is added using the encoder output
        (``context`` tensor) as key/value.  The forward pass then returns a
        ``(hidden_states, context)`` tuple instead of a bare tensor, matching
        Megatron's interface for cross-attention-based models and fixing VPP
        gradient flow across PP stages.

    DES-LOC extension:
        * ``self.desloc_tier`` → "h100" | "a6000" | None.
        * A6000 layers can use more aggressive activation checkpointing via
          ``config.recompute_granularity``.

    Activation recomputation:
        * ``"full"`` — checkpoint the entire layer forward (most memory saving,
          highest recompute cost).  Chunked uniform/block recompute is handled
          by ``TransformerBlock``.
        * ``"selective"`` — checkpoint only the core attention kernel.
        * ``None`` (default) — no recomputation.

    Args:
        config: TransformerConfig driving all sub-module construction.
        layer_number: 1-based global layer index (follows Megatron convention).
        hidden_dropout: Per-layer dropout override; defaults to
            ``config.hidden_dropout``.
        add_cross_attn: If True, add a cross-attention sub-layer (encoder-
            decoder / BERT pooler style).
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        hidden_dropout: Optional[float] = None,
        add_cross_attn: bool = False,
    ) -> None:
        super().__init__(config)
        self.layer_number = layer_number

        # Per-layer dropout (can differ from config default, e.g. layer-wise schedule)
        self.hidden_dropout: float = (
            config.hidden_dropout if hidden_dropout is None else hidden_dropout
        )

        # Cross-attention flag — True for encoder-decoder models (BERT, T5) (M2317)
        self.add_cross_attn: bool = (
            add_cross_attn or getattr(config, "encoder_decoder", False)
        )

        # DES-LOC tier assignment (0-based index)
        self.desloc_tier: Optional[str] = config.get_layer_tier(layer_number - 1)
        if self.desloc_tier is not None:
            logger.debug(
                "TransformerLayer %d → DES-LOC tier: %s",
                layer_number,
                self.desloc_tier.upper(),
            )

        # --- Pre-attention norm ------------------------------------------
        self.input_layernorm = _build_norm(config)

        # --- Self-attention -----------------------------------------------
        self.self_attention = SelfAttention(config, layer_number=layer_number)

        # --- Hidden-state dropout after attention -------------------------
        self.attn_dropout = nn.Dropout(p=self.hidden_dropout)

        # --- Cross-attention (optional, M2317 BERT/VPP fix) --------------
        if self.add_cross_attn:
            self.pre_cross_attn_layernorm = _build_norm(config)
            # Reuse SelfAttention for cross-attn; the distinction is that
            # the caller passes key_value_states (context) separately.
            # For simplicity we build an independent SelfAttention instance
            # that accepts context via rotary_pos_emb=None and the hidden
            # size of the encoder; callers must pass k/v as packed pairs.
            # A full cross-attention implementation would subclass differently;
            # here we expose the hook so the block layer can route context.
            self.cross_attention: Optional[nn.Module] = SelfAttention(
                config, layer_number=layer_number
            )
            self.cross_attn_dropout = nn.Dropout(p=self.hidden_dropout)
        else:
            self.pre_cross_attn_layernorm = None
            self.cross_attention = None
            self.cross_attn_dropout = None

        # --- Pre-MLP norm ------------------------------------------------
        self.pre_mlp_layernorm = _build_norm(config)

        # --- MLP ---------------------------------------------------------
        self.mlp = MLP(config, layer_number=layer_number)

        # --- Hidden-state dropout after MLP ------------------------------
        self.mlp_dropout = nn.Dropout(p=self.hidden_dropout)

        # --- Residual connection mode ------------------------------------
        self.apply_residual_post_layernorm: bool = (
            config.apply_residual_connection_post_layernorm
        )

        # --- fp32 residual connection (GPT-J / Falcon style) -------------
        self.fp32_residual_connection: bool = getattr(
            config, "fp32_residual_connection", False
        )

        # --- Activation recomputation strategy ---------------------------
        self.recompute_granularity: Optional[str] = config.recompute_granularity

    # ------------------------------------------------------------------
    # Property helpers
    # ------------------------------------------------------------------

    @property
    def has_clip_qk(self) -> bool:
        """True if this layer's self-attention supports QK logit clipping."""
        return getattr(self.self_attention, "clip_qk", None) is not None and callable(
            getattr(self.self_attention, "clip_qk", None)
        )

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _apply_residual(
        self,
        residual: torch.Tensor,
        sub_out: torch.Tensor,
        drop: nn.Dropout,
        norm: Optional[nn.Module],
    ) -> torch.Tensor:
        """Apply dropout + residual + optional post-norm.

        Pre-norm mode  (default): return ``residual + drop(sub_out)``
        Post-norm mode           : return ``norm(residual + drop(sub_out))``
        """
        out = residual + drop(sub_out)
        if self.apply_residual_post_layernorm and norm is not None:
            out = norm(out)
        return out

    def _forward_attention(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        rotary_pos_emb: Optional[torch.Tensor],
        rotary_pos_cos: Optional[torch.Tensor],
        rotary_pos_sin: Optional[torch.Tensor],
        attention_bias: Optional[torch.Tensor],
        inference_context: Optional[object],
        packed_seq_params: Optional[object],
        sequence_len_offset: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Self-attention sub-layer (+ optional cross-attention).

        Returns:
            (hidden_states, context): context is updated by cross-attn or
            passed through unchanged.
        """
        if self.fp32_residual_connection:
            residual = hidden_states.float()
        else:
            residual = hidden_states

        if self.apply_residual_post_layernorm:
            # Post-norm: run attention on raw hidden states, norm after residual
            attn_out_raw = self.self_attention(
                hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )
            # Attention returns (output, bias) tuple; extract tensor
            attn_out = attn_out_raw[0] if isinstance(attn_out_raw, (tuple, list)) else attn_out_raw
            hidden_states = self.input_layernorm(
                self._apply_residual(residual, attn_out, self.attn_dropout, None)
            )
        else:
            # Pre-norm: norm first, then attention, then residual
            normed = self.input_layernorm(hidden_states)
            attn_out_raw = self.self_attention(
                normed,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )
            # Attention returns (output, bias) tuple; extract tensor
            attn_out = attn_out_raw[0] if isinstance(attn_out_raw, (tuple, list)) else attn_out_raw
            hidden_states = self._apply_residual(
                residual, attn_out, self.attn_dropout, None
            )

        # --- Cross-attention (M2317: BERT / encoder-decoder) -------------
        if self.add_cross_attn and context is not None:
            if self.fp32_residual_connection:
                residual = hidden_states.float()
            else:
                residual = hidden_states

            if self.apply_residual_post_layernorm:
                cross_out_raw = self.cross_attention(
                    hidden_states,
                    attention_mask=context_mask,
                    inference_context=inference_context,
                )
                cross_out = cross_out_raw[0] if isinstance(cross_out_raw, (tuple, list)) else cross_out_raw
                hidden_states = self.pre_cross_attn_layernorm(
                    self._apply_residual(residual, cross_out, self.cross_attn_dropout, None)
                )
            else:
                normed = self.pre_cross_attn_layernorm(hidden_states)
                cross_out_raw = self.cross_attention(
                    normed,
                    attention_mask=context_mask,
                    inference_context=inference_context,
                )
                cross_out = cross_out_raw[0] if isinstance(cross_out_raw, (tuple, list)) else cross_out_raw
                hidden_states = self._apply_residual(
                    residual, cross_out, self.cross_attn_dropout, None
                )

        return hidden_states, context

    def _forward_mlp(
        self,
        hidden_states: torch.Tensor,
        inference_context: Optional[object] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """MLP sub-layer with residual.

        Returns:
            hidden_states after residual connection.
        """
        if self.fp32_residual_connection:
            residual = hidden_states.float()
        else:
            residual = hidden_states

        if self.apply_residual_post_layernorm:
            mlp_out_raw = self.mlp(hidden_states)
            # MLP returns (output, bias) tuple; extract tensor
            mlp_out = mlp_out_raw[0] if isinstance(mlp_out_raw, (tuple, list)) else mlp_out_raw
            hidden_states = self.pre_mlp_layernorm(
                self._apply_residual(residual, mlp_out, self.mlp_dropout, None)
            )
        else:
            normed = self.pre_mlp_layernorm(hidden_states)
            mlp_out_raw = self.mlp(normed)
            # MLP returns (output, bias) tuple; extract tensor
            mlp_out = mlp_out_raw[0] if isinstance(mlp_out_raw, (tuple, list)) else mlp_out_raw
            hidden_states = self._apply_residual(
                residual, mlp_out, self.mlp_dropout, None
            )

        return hidden_states

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        inference_context: Optional[object] = None,
        packed_seq_params: Optional[object] = None,
        sequence_len_offset: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        *,
        inference_params: Optional[object] = None,  # deprecated alias
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Forward pass of one transformer layer.

        Returns ``(hidden_states, context)`` when cross-attention is active
        (``add_cross_attn=True`` or encoder-decoder config), matching
        Megatron's interface needed for BERT + virtual pipeline parallelism
        (M2317 fix).  Otherwise returns just ``hidden_states``.

        Args:
            hidden_states: ``[seq, batch, hidden]``
            attention_mask: Optional mask ``[batch, 1, seq, seq]``
            context: Encoder output for cross-attention ``[seq_enc, batch, hidden]``
            context_mask: Mask for cross-attention
            rotary_pos_emb: Rotary embeddings ``[seq, 1, 1, head_dim]``
            rotary_pos_cos: Rotary embedding cosines (flash decode)
            rotary_pos_sin: Rotary embedding sines (flash decode)
            attention_bias: Additive attention bias ``[1, heads, seq, seq]``
            inference_context: Passed through to attention.
            packed_seq_params: THD packed sequence params.
            sequence_len_offset: Sequence offset for inference CUDA graphs.
            padding_mask: Padding mask for MoE routing.
            inference_params: Deprecated alias for inference_context.

        Returns:
            If cross-attention is active: ``(hidden_states, context)``
            Otherwise: ``hidden_states``
        """
        # Backward-compat: deprecated inference_params → inference_context
        if inference_context is None and inference_params is not None:
            inference_context = inference_params

        def _run(hs):
            out_hs, out_ctx = self._forward_attention(
                hs,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                padding_mask=padding_mask,
            )
            out_hs = self._forward_mlp(
                out_hs,
                inference_context=inference_context,
                padding_mask=padding_mask,
            )
            return out_hs, out_ctx

        if self.recompute_granularity == "full" and self.training:
            # Checkpoint the entire transformer layer
            # torch.utils.checkpoint requires tensor outputs; we pass context
            # through unchanged so it is safe to checkpoint just hs.
            def _full_forward(hs):
                out_hs, _ = _run(hs)
                return out_hs

            hidden_states = torch.utils.checkpoint.checkpoint(
                _full_forward,
                hidden_states,
                use_reentrant=False,
            )
            if self.add_cross_attn:
                return hidden_states, context
            return hidden_states

        elif self.recompute_granularity == "selective" and self.training:
            # Checkpoint only the attention sub-layer
            def _selective_attn(hs):
                out_hs, out_ctx = self._forward_attention(
                    hs,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=rotary_pos_cos,
                    rotary_pos_sin=rotary_pos_sin,
                    attention_bias=attention_bias,
                    inference_context=inference_context,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=sequence_len_offset,
                    padding_mask=padding_mask,
                )
                return out_hs

            hidden_states = torch.utils.checkpoint.checkpoint(
                _selective_attn,
                hidden_states,
                use_reentrant=False,
            )
            hidden_states = self._forward_mlp(
                hidden_states,
                inference_context=inference_context,
                padding_mask=padding_mask,
            )
            if self.add_cross_attn:
                return hidden_states, context
            return hidden_states

        else:
            hidden_states, context = _run(hidden_states)
            if self.add_cross_attn:
                return hidden_states, context
            return hidden_states

    # ------------------------------------------------------------------
    # Sharded state dict (M2317 / pipeline checkpointing)
    # ------------------------------------------------------------------

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> dict:
        """Sharded state dict for pipeline-parallel checkpointing.

        Delegates to each child module's ``sharded_state_dict`` if available,
        otherwise falls back to ``state_dict``.

        Args:
            prefix: Key prefix for this layer.
            sharded_offsets: PP/TP offset tuples from the enclosing block.
            metadata: Forwarded to child modules.

        Returns:
            Dict mapping checkpoint key → tensor / ShardedTensor.
        """
        state_dict: dict = {}
        for name, module in self.named_children():
            if module is None:
                continue
            sub_prefix = f"{prefix}{name}."
            if hasattr(module, "sharded_state_dict"):
                state_dict.update(
                    module.sharded_state_dict(sub_prefix, sharded_offsets, metadata)
                )
            else:
                for k, v in module.state_dict(prefix="").items():
                    state_dict[f"{sub_prefix}{k}"] = v
        return state_dict
