"""MoE Layer adapted from Megatron megatron/core/transformer/moe/moe_layer.py.

Combines TopKRouter + expert MLPs + token dispatcher into a single module
that replaces the standard MLP in TransformerLayer.

Pure PyTorch. PCIe-optimized: uses permute/unpermute instead of all-to-all
for single-machine heterogeneous topologies.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from deepspeed.core.transformer.moe.moe_utils import permute_tokens, unpermute_tokens
from deepspeed.core.transformer.moe.router import TopKRouter


class ExpertMLP(nn.Module):
    """Single expert MLP (SwiGLU or GELU).

    Same architecture as the standard MLP but instantiated per-expert.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ffn_hidden_size = getattr(config, "ffn_hidden_size", config.hidden_size * 4)
        use_swiglu = getattr(config, "activation_func_type", "gelu") == "swiglu"

        if use_swiglu:
            # SwiGLU: gate + up projection, then down
            self.gate_proj = nn.Linear(self.hidden_size, self.ffn_hidden_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.ffn_hidden_size, bias=False)
            self.down_proj = nn.Linear(self.ffn_hidden_size, self.hidden_size, bias=False)
            self.use_swiglu = True
        else:
            self.up_proj = nn.Linear(self.hidden_size, self.ffn_hidden_size, bias=True)
            self.down_proj = nn.Linear(self.ffn_hidden_size, self.hidden_size, bias=True)
            self.use_swiglu = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
        else:
            return self.down_proj(nn.functional.gelu(self.up_proj(x)))


class MoELayer(nn.Module):
    """Mixture-of-Experts layer.

    Replaces the standard MLP in TransformerLayer when num_moe_experts > 0.
    Token routing via TopKRouter, expert processing via per-expert MLPs.

    For PCIe-only heterogeneous topologies (A6000 + H100), uses permute-based
    token dispatch instead of all-to-all to avoid PCIe bandwidth bottleneck.

    Args:
        config: TransformerConfig with MoE fields.
        layer_number: Layer index (for logging/debugging).
    """

    def __init__(self, config, layer_number: int = 0) -> None:
        super().__init__()
        self.config = config
        self.num_experts = config.num_moe_experts
        self.topk = config.moe_router_topk
        self.layer_number = layer_number

        # Router
        self.router = TopKRouter(config)

        # Expert MLPs
        self.experts = nn.ModuleList([ExpertMLP(config) for _ in range(self.num_experts)])

        # Optional shared expert (DeepSeek-style)
        self.shared_expert = None
        num_shared = getattr(config, "moe_num_shared_experts", 0)
        if num_shared > 0:
            self.shared_expert = ExpertMLP(config)

        # For tracking
        self._expert_utilization = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Process tokens through MoE.

        Args:
            hidden_states: [seq_len, batch_size, hidden_size] or [num_tokens, hidden_size].

        Returns:
            output: Same shape as input.
        """
        # Reshape to 2D if needed
        original_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            seq_len, batch_size, hidden_size = hidden_states.shape
            hidden_states_2d = hidden_states.reshape(-1, hidden_size)
        else:
            hidden_states_2d = hidden_states

        num_tokens = hidden_states_2d.shape[0]

        # Route tokens
        scores, indices, probs = self.router(hidden_states_2d)

        # Permute tokens by expert assignment
        permuted_tokens, sorted_indices = permute_tokens(
            hidden_states_2d, indices, self.num_experts, self.topk
        )

        # Process tokens through experts
        # Group tokens by expert for efficient batched computation
        flat_indices = indices.reshape(-1)
        sorted_flat = flat_indices[sorted_indices]

        expert_outputs = torch.zeros_like(permuted_tokens)
        offset = 0
        for expert_idx in range(self.num_experts):
            mask = sorted_flat == expert_idx
            num_expert_tokens = mask.sum().item()
            if num_expert_tokens > 0:
                expert_input = permuted_tokens[mask]
                expert_output = self.experts[expert_idx](expert_input)
                expert_outputs[mask] = expert_output

        # Handle dropped tokens (index == -1)
        dropped_mask = sorted_flat < 0
        expert_outputs[dropped_mask] = 0.0

        # Unpermute and weight by routing scores
        output = unpermute_tokens(expert_outputs, sorted_indices, scores, self.topk)

        # Add shared expert output if present
        if self.shared_expert is not None:
            shared_out = self.shared_expert(hidden_states_2d)
            output = output + shared_out

        # Reshape back
        output = output.reshape(original_shape)
        return output

    def get_aux_loss(self) -> torch.Tensor:
        """Return accumulated auxiliary loss from router."""
        return self.router.aux_loss
