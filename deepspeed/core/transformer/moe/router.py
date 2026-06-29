"""MoE Router adapted from Megatron megatron/core/transformer/moe/router.py.

Supports top-1 and top-2 routing with load balancing auxiliary loss.
Pure PyTorch, no TE dependency. Works on SM86 (A6000) and SM90 (H100).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

import deepspeed.core.parallel_state as parallel_state
from deepspeed.core.transformer.moe.moe_utils import (
    switch_load_balancing_loss_func,
    topk_softmax_with_capacity,
    z_loss_func,
)


class TopKRouter(nn.Module):
    """Top-K expert router with load balancing.

    Routes each token to the top-k experts based on a learned linear gate.
    Computes auxiliary load balancing loss to encourage uniform utilization.

    Args:
        config: TransformerConfig with MoE fields:
            num_moe_experts, moe_router_topk, moe_aux_loss_coeff,
            hidden_size, moe_z_loss_coeff, moe_token_capacity_factor.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.num_experts = config.num_moe_experts
        self.topk = config.moe_router_topk
        self.aux_loss_coeff = getattr(config, "moe_aux_loss_coeff", 0.01)
        self.z_loss_coeff = getattr(config, "moe_z_loss_coeff", 0.0)
        self.capacity_factor = getattr(config, "moe_token_capacity_factor", None)

        # Gate projection: hidden_size -> num_experts
        self.weight = nn.Parameter(
            torch.empty(self.num_experts, config.hidden_size, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.weight)

        # Accumulated auxiliary loss for the current forward pass
        self.aux_loss = torch.tensor(0.0)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        Args:
            hidden_states: [num_tokens, hidden_size] (already reshaped from [S, B, H]).

        Returns:
            scores: Routing scores [num_tokens, topk].
            indices: Expert indices [num_tokens, topk].
            probs: Full softmax probabilities [num_tokens, num_experts] (for aux loss).
        """
        # Compute router logits
        logits = torch.matmul(
            hidden_states.float(), self.weight.t()
        )  # [num_tokens, num_experts]

        # Top-k selection with optional capacity
        probs, indices, scores = topk_softmax_with_capacity(
            logits, self.topk, self.capacity_factor
        )

        # Compute auxiliary losses
        tokens_per_expert = torch.zeros(
            self.num_experts, device=hidden_states.device, dtype=torch.long
        )
        for k_idx in range(self.topk):
            valid_mask = indices[:, k_idx] >= 0
            if valid_mask.any():
                tokens_per_expert.scatter_add_(
                    0,
                    indices[:, k_idx][valid_mask],
                    torch.ones_like(indices[:, k_idx][valid_mask], dtype=torch.long),
                )

        total_tokens = hidden_states.shape[0]
        # Reduce across DP group for global load balancing
        dp_group = parallel_state.get_data_parallel_group() if parallel_state.is_initialized() else None
        if dp_group is not None:
            torch.distributed.all_reduce(tokens_per_expert, group=dp_group)
            total_tokens_tensor = torch.tensor(
                total_tokens, device=hidden_states.device, dtype=torch.long
            )
            torch.distributed.all_reduce(total_tokens_tensor, group=dp_group)
            total_tokens = total_tokens_tensor.item()

        self.aux_loss = switch_load_balancing_loss_func(
            probs, tokens_per_expert, total_tokens, self.topk, self.aux_loss_coeff
        )
        if self.z_loss_coeff > 0:
            self.aux_loss = self.aux_loss + z_loss_func(logits, self.z_loss_coeff)

        return scores, indices, probs
