"""Mixture-of-Experts subsystem for heterogeneous GPU training.

Adapted from Megatron megatron/core/transformer/moe/.
"""
from deepspeed.core.transformer.moe.moe_layer import ExpertMLP, MoELayer
from deepspeed.core.transformer.moe.moe_utils import (
    permute_tokens,
    switch_load_balancing_loss_func,
    topk_softmax_with_capacity,
    unpermute_tokens,
    z_loss_func,
)
from deepspeed.core.transformer.moe.router import TopKRouter

__all__ = [
    "MoELayer",
    "TopKRouter",
    "ExpertMLP",
    "switch_load_balancing_loss_func",
    "z_loss_func",
    "topk_softmax_with_capacity",
    "permute_tokens",
    "unpermute_tokens",
]
