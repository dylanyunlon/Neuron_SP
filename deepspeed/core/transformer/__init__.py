# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""deepspeed.core.transformer — re-export facade.

This package provides the canonical import surface for the DeepSpeed transformer
stack.  All implementation lives in the individual sub-modules; this file only
re-exports the public API so that consumers can write::

    from deepspeed.core.transformer import TransformerConfig, TransformerBlock

Public API
----------
TransformerConfig       — configuration dataclass with DES-LOC tier fields
TransformerBlock        — stack of transformer layers with PP support
TransformerLayer        — single attention + MLP layer
SelfAttention           — multi-head self-attention with AutoSP
DotProductAttention     — scaled dot-product attention kernel
MLP                     — SwiGLU MLP with TP sharding
MegatronModule          — base module class (sharded_state_dict support)

DES-LOC tier assignment
-----------------------
TransformerConfig exposes:
  * desloc_h100_layers  / desloc_a6000_layers  — layer index lists
  * desloc_tier_strategy                        — "front_heavy" | "back_heavy" |
                                                  "interleave" | "manual"
  * desloc_h100_layer_fraction                  — fraction for front/back strategies
  * get_layer_tier(layer_idx) → "h100" | "a6000" | None
  * is_h100_layer / is_a6000_layer              — convenience predicates

TransformerBlock exposes:
  * get_desloc_tier_map() → {local_layer_idx: tier_str}
"""

from deepspeed.core.transformer.transformer_config import TransformerConfig
from deepspeed.core.transformer.transformer_block import TransformerBlock
from deepspeed.core.transformer.transformer_layer import TransformerLayer
from deepspeed.core.transformer.attention import (
    Attention,
    SelfAttention,
    DotProductAttention,
)
from deepspeed.core.transformer.mlp import MLP
from deepspeed.core.transformer.module import MegatronModule
from deepspeed.core.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
    MultiTokenPredictionLayerSubmodules,
    MultiTokenPredictionBlock,
    MultiTokenPredictionBlockSubmodules,
    MTPLossAutoScaler,
    MTPLossLoggingHelper,
    process_mtp_loss,
    roll_tensor,
    get_mtp_layer_spec,
    get_mtp_layer_spec_for_backend,
    get_mtp_layer_offset,
    get_mtp_num_layers_to_build,
    get_mtp_ranks,
    mtp_on_this_rank,
    tie_word_embeddings_state_dict,
    tie_output_layer_state_dict,
)

__all__ = [
    # Configuration
    "TransformerConfig",
    # Block / layer
    "TransformerBlock",
    "TransformerLayer",
    # Attention
    "Attention",
    "SelfAttention",
    "DotProductAttention",
    # MLP
    "MLP",
    # Base
    "MegatronModule",
    # Multi-Token Prediction (MTP)
    "MultiTokenPredictionLayer",
    "MultiTokenPredictionLayerSubmodules",
    "MultiTokenPredictionBlock",
    "MultiTokenPredictionBlockSubmodules",
    "MTPLossAutoScaler",
    "MTPLossLoggingHelper",
    "process_mtp_loss",
    "roll_tensor",
    "get_mtp_layer_spec",
    "get_mtp_layer_spec_for_backend",
    "get_mtp_layer_offset",
    "get_mtp_num_layers_to_build",
    "get_mtp_ranks",
    "mtp_on_this_rank",
    "tie_word_embeddings_state_dict",
    "tie_output_layer_state_dict",
]
