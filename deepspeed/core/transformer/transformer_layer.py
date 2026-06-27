# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""TransformerLayer — single transformer layer with DES-LOC tier annotation.

DES-LOC integration
-------------------
Each ``TransformerLayer`` carries a zero-based ``layer_number`` (1-based
globally).  On construction it queries ``TransformerConfig.get_layer_tier()``
and stores the result in ``self.desloc_tier`` ("h100" | "a6000" | None).

This attribute is used by the DES-LOC engine to decide:
  * Which device the layer's parameters are pinned to.
  * How aggressively to recompute activations (A6000 = less VRAM → more
    aggressive checkpointing).

Usage example::

    layer = TransformerLayer(config, layer_number=5)
    print(layer.desloc_tier)   # "h100" or "a6000"

Activation recomputation
------------------------
When ``config.recompute_granularity == "full"`` the entire forward pass is
wrapped in ``torch.utils.checkpoint.checkpoint``.  This is particularly
useful for A6000 stages that have limited VRAM.

When ``config.recompute_granularity == "selective"`` only the core attention
kernel is recomputed (memory-intensive but compute-cheap).
"""

from __future__ import annotations

import logging
from typing import Optional

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

    DES-LOC extension:
        * ``self.desloc_tier`` → "h100" | "a6000" | None.
        * A6000 layers can use more aggressive activation checkpointing via
          ``config.recompute_granularity``.

    Activation recomputation:
        * ``"full"`` — checkpoint the entire layer forward (most memory saving,
          highest recompute cost).
        * ``"selective"`` — checkpoint only the core attention kernel.
        * ``None`` (default) — no recomputation.

    Args:
        config: TransformerConfig driving all sub-module construction.
        layer_number: 1-based global layer index (follows Megatron convention).
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
    ) -> None:
        super().__init__(config)
        self.layer_number = layer_number

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
        self.self_attention = SelfAttention(config, layer_number)

        # --- Hidden-state dropout after attention -------------------------
        self.attn_dropout = nn.Dropout(p=config.hidden_dropout)

        # --- Pre-MLP norm ------------------------------------------------
        self.pre_mlp_layernorm = _build_norm(config)

        # --- MLP ---------------------------------------------------------
        self.mlp = MLP(config, layer_number=layer_number)

        # --- Hidden-state dropout after MLP ------------------------------
        self.mlp_dropout = nn.Dropout(p=config.hidden_dropout)

        # --- Residual connection mode ------------------------------------
        self.apply_residual_post_layernorm: bool = (
            config.apply_residual_connection_post_layernorm
        )

        # --- Activation recomputation strategy ---------------------------
        self.recompute_granularity: Optional[str] = config.recompute_granularity

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _forward_attention(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        rotary_pos_emb: Optional[torch.Tensor],
        inference_params: Optional[object],
    ) -> torch.Tensor:
        """Run the attention sub-layer and apply the residual.

        Supports both pre-norm and post-norm modes.

        Returns:
            hidden_states after residual connection.
        """
        residual = hidden_states

        if self.apply_residual_post_layernorm:
            attn_out = self.self_attention(
                hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                inference_params=inference_params,
            )
            hidden_states = self.input_layernorm(
                residual + self.attn_dropout(attn_out)
            )
        else:
            normed = self.input_layernorm(hidden_states)
            attn_out = self.self_attention(
                normed,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                inference_params=inference_params,
            )
            hidden_states = residual + self.attn_dropout(attn_out)

        return hidden_states

    def _forward_mlp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the MLP sub-layer and apply the residual.

        Returns:
            hidden_states after residual connection.
        """
        residual = hidden_states

        if self.apply_residual_post_layernorm:
            mlp_out = self.mlp(hidden_states)
            hidden_states = self.pre_mlp_layernorm(
                residual + self.mlp_dropout(mlp_out)
            )
        else:
            normed = self.pre_mlp_layernorm(hidden_states)
            mlp_out = self.mlp(normed)
            hidden_states = residual + self.mlp_dropout(mlp_out)

        return hidden_states

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
        """Forward pass of one transformer layer.

        Args:
            hidden_states: ``[seq, batch, hidden]``
            attention_mask: Optional mask ``[batch, 1, seq, seq]``
            rotary_pos_emb: Optional rotary embeddings ``[seq, 1, 1, head_dim]``
            inference_params: Passed through to attention (currently unused).

        Returns:
            output: ``[seq, batch, hidden]``
        """
        if self.recompute_granularity == "full":
            # Checkpoint the entire transformer layer
            def _full_forward(hs, mask, rope, ip):
                hs = self._forward_attention(hs, mask, rope, ip)
                hs = self._forward_mlp(hs)
                return hs

            # torch.utils.checkpoint doesn't handle None well in some versions;
            # pass through a dummy tensor if needed
            return torch.utils.checkpoint.checkpoint(
                _full_forward,
                hidden_states,
                attention_mask,
                rotary_pos_emb,
                inference_params,
                use_reentrant=False,
            )

        elif self.recompute_granularity == "selective":
            # Checkpoint only the core attention kernel
            # (handled inside DotProductAttention in a future extension;
            #  for now we checkpoint the full attention sub-layer)
            hidden_states = torch.utils.checkpoint.checkpoint(
                self._forward_attention,
                hidden_states,
                attention_mask,
                rotary_pos_emb,
                inference_params,
                use_reentrant=False,
            )
            hidden_states = self._forward_mlp(hidden_states)
            return hidden_states

        else:
            # No recomputation
            hidden_states = self._forward_attention(
                hidden_states, attention_mask, rotary_pos_emb, inference_params
            )
            hidden_states = self._forward_mlp(hidden_states)
            return hidden_states
