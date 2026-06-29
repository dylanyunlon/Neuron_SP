"""Mixture-of-Experts subsystem for heterogeneous GPU training.

Adapted from Megatron megatron/core/transformer/moe/.
"""
from deepspeed.core.transformer.moe.experts import (
    GroupedMLPSubmodules,
    TEGroupedMLP,
    InferenceGroupedMLP,
    SequentialMLP,
)
from deepspeed.core.transformer.moe.moe_layer import ExpertMLP, MoELayer
from deepspeed.core.transformer.moe.shared_experts import SharedExpertMLP, SharedExpertState
from deepspeed.core.transformer.moe.moe_utils import (
    permute_tokens,
    switch_load_balancing_loss_func,
    topk_softmax_with_capacity,
    unpermute_tokens,
    z_loss_func,
)
from deepspeed.core.transformer.moe.router import TopKRouter
from deepspeed.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
    MoETokenDispatcher,
    PCIeAwareAlltoAll,
)

__all__ = [
    # Grouped / sequential expert implementations (ported from Megatron experts.py)
    "GroupedMLPSubmodules",
    "TEGroupedMLP",
    "InferenceGroupedMLP",
    "SequentialMLP",
    # Legacy simple expert + MoE layer
    "MoELayer",
    "TopKRouter",
    "ExpertMLP",
    "SharedExpertMLP",
    "SharedExpertState",
    "switch_load_balancing_loss_func",
    "z_loss_func",
    "topk_softmax_with_capacity",
    "permute_tokens",
    "unpermute_tokens",
    # Token dispatchers (ported from Megatron + PCIe-aware A2A)
    "MoETokenDispatcher",
    "MoEAllGatherTokenDispatcher",
    "MoEAlltoAllTokenDispatcher",
    "MoEFlexTokenDispatcher",
    "PCIeAwareAlltoAll",
]
