# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Transformer layers with AutoSP sequence parallel and DES-LOC support."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepspeed.core.model_parallel_config import ModelParallelConfig


# ===========================================================================
# transformer_config.py
# ===========================================================================

@dataclass
class TransformerConfig(ModelParallelConfig):
    """Full transformer configuration. Extends ModelParallelConfig."""

    num_layers: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_query_groups: Optional[int] = None  # GQA
    ffn_hidden_size: Optional[int] = None
    kv_channels: Optional[int] = None

    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    layernorm_epsilon: float = 1e-5
    add_bias_linear: bool = False
    gated_linear_unit: bool = False
    activation_func: Callable = F.silu  # SwiGLU default

    normalization: Literal["LayerNorm", "RMSNorm"] = "RMSNorm"
    apply_residual_connection_post_layernorm: bool = False

    # Rotary embeddings
    rotary_interleaved: bool = False
    window_size: Optional[Tuple[int, int]] = None

    # MoE
    num_moe_experts: Optional[int] = None

    # MTP (Multi-Token Prediction)
    mtp_num_layers: Optional[int] = None
    mtp_loss_scaling_factor: float = 0.1

    # MLA (Multi-Latent Attention)
    multi_latent_attention: bool = False

    # Per-token loss
    calculate_per_token_loss: bool = False

    def __post_init__(self):
        if self.ffn_hidden_size is None:
            # SwiGLU default: 8/3 * hidden, rounded to 64
            self.ffn_hidden_size = int(self.hidden_size * 8 / 3)
            self.ffn_hidden_size = ((self.ffn_hidden_size + 63) // 64) * 64
        if self.kv_channels is None:
            self.kv_channels = self.hidden_size // self.num_attention_heads
        if self.num_query_groups is None:
            self.num_query_groups = self.num_attention_heads


# ===========================================================================
# module.py — base class for all transformer modules
# ===========================================================================

class MegatronModule(nn.Module, ABC):
    """Base module class. All transformer components inherit from this."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None,
    ) -> dict:
        """Return state dict with sharding metadata for distributed checkpointing."""
        raise NotImplementedError("Claude task: transformer/module")


# ===========================================================================
# attention.py
# ===========================================================================

class Attention(MegatronModule, ABC):
    """Base attention class.

    AutoSP integration: when sequence_parallel is enabled, the input
    sequence is already partitioned across SP ranks. The attention
    computation handles the local chunk and uses A2A for KV exchange.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: str = "causal",
    ) -> None:
        raise NotImplementedError("Claude task: transformer/attention")

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        inference_params: Optional[object] = None,
    ) -> torch.Tensor:
        ...


class SelfAttention(Attention):
    """Standard multi-head self-attention with QKV projection."""

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
    ) -> None:
        raise NotImplementedError("Claude task: transformer/attention")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        inference_params: Optional[object] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Claude task: transformer/attention")


class DotProductAttention(MegatronModule):
    """Scaled dot-product attention with flash attention support."""

    def __init__(self, config: TransformerConfig, layer_number: int) -> None:
        raise NotImplementedError("Claude task: transformer/dot_product_attention")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Claude task: transformer/dot_product_attention")


# ===========================================================================
# mlp.py
# ===========================================================================

class MLP(MegatronModule):
    """MLP with SwiGLU activation.

    Structure: gate_proj + up_proj → SiLU(gate) * up → down_proj
    """

    def __init__(self, config: TransformerConfig) -> None:
        raise NotImplementedError("Claude task: transformer/mlp")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Claude task: transformer/mlp")


# ===========================================================================
# transformer_layer.py
# ===========================================================================

class TransformerLayer(MegatronModule):
    """Single transformer layer: attention → residual → MLP → residual.

    DES-LOC extension: supports selective activation checkpointing
    controlled per-tier (A6000 stages checkpoint more aggressively
    than H100 stages due to smaller VRAM).
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
    ) -> None:
        raise NotImplementedError("Claude task: transformer/transformer_layer")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        inference_params: Optional[object] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Claude task: transformer/transformer_layer")


# ===========================================================================
# transformer_block.py
# ===========================================================================

class TransformerBlock(MegatronModule):
    """Stack of TransformerLayers with optional final layer norm.

    In PP mode, each rank holds a contiguous subset of layers determined
    by pipeline_layer_split. Supports heterogeneous splits where
    high-VRAM stages hold more layers.
    """

    def __init__(self, config: TransformerConfig) -> None:
        raise NotImplementedError("Claude task: transformer/transformer_block")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        inference_params: Optional[object] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Claude task: transformer/transformer_block")

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Set input tensor for PP receive."""
        raise NotImplementedError("Claude task: transformer/transformer_block")

    def _build_layers(self) -> None:
        """Build TransformerLayer stack for this PP stage."""
        raise NotImplementedError("Claude task: transformer/transformer_block")


# ===========================================================================
# moe/moe_layer.py (stub)
# ===========================================================================

class MoELayer(MegatronModule):
    """Mixture of Experts layer with router and token dispatcher.

    DES-LOC extension: expert placement strategy considers per-GPU VRAM.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
    ) -> None:
        raise NotImplementedError("Claude task: transformer/moe")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Claude task: transformer/moe")


class Router(MegatronModule, ABC):
    """Base router for MoE."""

    def __init__(self, config: TransformerConfig) -> None:
        raise NotImplementedError("Claude task: transformer/moe")

    @abstractmethod
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route tokens to experts. Returns (dispatch_weights, dispatch_indices)."""
        ...


class TopKRouter(Router):
    """Top-K token routing with auxiliary load-balancing loss."""

    def __init__(self, config: TransformerConfig) -> None:
        raise NotImplementedError("Claude task: transformer/moe")

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Claude task: transformer/moe")


class MoETokenDispatcher(ABC):
    """Dispatches tokens to experts and combines results."""

    def __init__(self, config: TransformerConfig) -> None:
        raise NotImplementedError("Claude task: transformer/moe")

    @abstractmethod
    def token_dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def token_combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
