"""MoE utility functions adapted from Megatron megatron/core/transformer/moe/moe_utils.py.

Stripped of TE/flashinfer/triton dependencies. Pure PyTorch implementation
for heterogeneous GPU clusters (A6000 SM86 + H100 SM90).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist

import deepspeed.core.parallel_state as parallel_state


def switch_load_balancing_loss_func(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    total_num_tokens: int,
    topk: int,
    moe_aux_loss_coeff: float,
) -> torch.Tensor:
    """Switch Transformer load balancing loss (Fedus et al., 2021).

    Encourages uniform expert utilization across the batch.

    Args:
        probs: Router probabilities [num_tokens, num_experts].
        tokens_per_expert: Count of tokens routed to each expert [num_experts].
        total_num_tokens: Total tokens in the batch (across DP).
        topk: Number of experts each token is routed to.
        moe_aux_loss_coeff: Loss coefficient.
    """
    num_experts = probs.shape[1]
    # f_i = fraction of tokens routed to expert i
    f = tokens_per_expert.float() / (total_num_tokens * topk + 1e-8)
    # P_i = mean probability assigned to expert i
    p = probs.mean(dim=0)
    aux_loss = moe_aux_loss_coeff * num_experts * (f * p).sum()
    return aux_loss


def z_loss_func(logits: torch.Tensor, z_loss_coeff: float) -> torch.Tensor:
    """Router z-loss to encourage small logit magnitudes (ST-MoE).

    Args:
        logits: Router logits [num_tokens, num_experts].
        z_loss_coeff: Loss coefficient.
    """
    z = logits.float().logsumexp(dim=-1).square().mean()
    return z_loss_coeff * z


def topk_softmax_with_capacity(
    logits: torch.Tensor,
    topk: int,
    capacity_factor: Optional[float] = None,
    drop_policy: str = "probs",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top-k routing with optional capacity-based token dropping.

    Args:
        logits: Router logits [num_tokens, num_experts].
        topk: Number of experts per token.
        capacity_factor: If set, limit tokens per expert. None = no limit.
        drop_policy: 'probs' (drop lowest-prob) or 'position' (drop last tokens).

    Returns:
        probs: Routing probabilities [num_tokens, num_experts] (full softmax).
        indices: Selected expert indices [num_tokens, topk].
        scores: Selected expert scores [num_tokens, topk].
    """
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    scores, indices = torch.topk(probs, k=topk, dim=-1)
    # Normalize scores so they sum to 1 per token
    scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-8)

    if capacity_factor is not None:
        num_tokens, num_experts = logits.shape
        capacity = int(capacity_factor * num_tokens * topk / num_experts)
        # Count tokens per expert, drop excess
        for k_idx in range(topk):
            expert_counts = torch.zeros(num_experts, device=logits.device, dtype=torch.long)
            for e in range(num_experts):
                mask = indices[:, k_idx] == e
                count = mask.sum()
                if count > capacity:
                    # Drop excess tokens (set score to 0)
                    excess_positions = mask.nonzero(as_tuple=True)[0][capacity:]
                    scores[excess_positions, k_idx] = 0.0
                    indices[excess_positions, k_idx] = -1

    return probs, indices, scores


def permute_tokens(
    hidden_states: torch.Tensor,
    indices: torch.Tensor,
    num_experts: int,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Permute tokens by expert assignment for grouped processing.

    Args:
        hidden_states: Input [num_tokens, hidden_size].
        indices: Expert indices [num_tokens, topk].
        num_experts: Total number of experts.
        topk: Experts per token.

    Returns:
        permuted_tokens: Reordered [num_tokens * topk, hidden_size].
        sorted_indices: Index mapping for unpermute.
    """
    num_tokens, hidden_size = hidden_states.shape
    # Expand tokens for topk
    expanded = hidden_states.unsqueeze(1).expand(-1, topk, -1).reshape(-1, hidden_size)
    flat_indices = indices.reshape(-1)
    # Sort by expert index
    sorted_indices = flat_indices.argsort(stable=True)
    permuted_tokens = expanded[sorted_indices]
    return permuted_tokens, sorted_indices


def unpermute_tokens(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Reverse permute and apply routing scores.

    Args:
        permuted_tokens: Expert outputs [num_tokens * topk, hidden_size].
        sorted_indices: From permute_tokens.
        scores: Routing scores [num_tokens, topk].
        topk: Experts per token.

    Returns:
        output: Weighted combination [num_tokens, hidden_size].
    """
    num_total = permuted_tokens.shape[0]
    hidden_size = permuted_tokens.shape[1]
    num_tokens = num_total // topk

    # Reverse the permutation
    unpermuted = torch.zeros_like(permuted_tokens)
    unpermuted[sorted_indices] = permuted_tokens

    # Reshape and apply scores
    unpermuted = unpermuted.reshape(num_tokens, topk, hidden_size)
    scores_expanded = scores.unsqueeze(-1)  # [num_tokens, topk, 1]
    output = (unpermuted * scores_expanded).sum(dim=1)
    return output
