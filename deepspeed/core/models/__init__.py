# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Model definitions — GPT and hybrid architectures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from deepspeed.core.transformer import TransformerConfig, TransformerBlock, MegatronModule


# ===========================================================================
# common/language_module.py
# ===========================================================================

class LanguageModule(MegatronModule, ABC):
    """Base class for language models (GPT, T5, hybrid).

    Manages embedding layers, output projection, and loss computation.
    """

    def __init__(self, config: TransformerConfig) -> None:
        raise NotImplementedError("Claude task: models/common")

    def setup_embeddings_and_output_layer(self) -> None:
        """Initialize input embeddings and output projection.

        Handles weight tying between input embeddings and output layer
        when share_embeddings_and_output_weights is True.
        """
        raise NotImplementedError("Claude task: models/common")

    def compute_language_model_loss(
        self, labels: torch.Tensor, logits: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy loss with optional per-token scaling."""
        raise NotImplementedError("Claude task: models/common")

    def shared_embedding_or_output_weight(self) -> torch.Tensor:
        raise NotImplementedError("Claude task: models/common")


# ===========================================================================
# gpt/gpt_model.py
# ===========================================================================

class GPTModel(LanguageModule):
    """GPT-style autoregressive language model.

    Architecture: embedding → TransformerBlock → output_layer → loss
    Supports: GQA, RoPE, SwiGLU, RMSNorm, per-token loss.

    DES-LOC integration:
    - Pipeline splits determined by config.pipeline_layer_split
    - Activation checkpointing granularity varies per tier
    - AutoSP shards sequence dim in attention
    """

    def __init__(
        self,
        config: TransformerConfig,
        pre_process: bool = True,   # True for first PP stage (has embeddings)
        post_process: bool = True,  # True for last PP stage (has output layer)
        share_embeddings_and_output_weights: bool = True,
    ) -> None:
        raise NotImplementedError("Claude task: models/gpt")

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Set input tensor received from previous PP stage."""
        raise NotImplementedError("Claude task: models/gpt")

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inference_params: Optional[object] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Returns loss if labels provided (training), logits otherwise (inference).
        """
        raise NotImplementedError("Claude task: models/gpt")

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None,
    ) -> dict:
        raise NotImplementedError("Claude task: models/gpt")


# ===========================================================================
# gpt/gpt_layer_specs.py
# ===========================================================================

def get_gpt_layer_spec(config: TransformerConfig) -> dict:
    """Return the layer specification for a GPT model.

    The spec defines which submodule classes to use for attention,
    MLP, norms, etc. Allows swapping components (e.g. standard attention
    vs MLA, standard MLP vs MoE).
    """
    raise NotImplementedError("Claude task: models/gpt")
