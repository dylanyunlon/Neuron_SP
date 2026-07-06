"""Fused kernels for heterogeneous GPU clusters.

Mirrors Megatron megatron/core/fusions/ but dispatches different kernel
implementations per compute capability (SM86 for A6000, SM90 for H100).

Key fusions:
- fused_bias_dropout:  bias + dropout + residual add (JIT-fused)
- fused_bias_geglu / fused_bias_swiglu: bias + activation in one kernel
- fused_layer_norm:  apex persistent / fused LayerNorm with PyTorch fallback
- fused_softmax:     memory-efficient softmax (critical for A6000 VRAM)
"""
from deepspeed.core.fusions.fused_bias_dropout import (
    bias_dropout_add_fused_inference,
    bias_dropout_add_fused_train,
    bias_dropout_add_unfused,
    get_bias_dropout_add,
)
from deepspeed.core.fusions.fused_bias_geglu import (
    fused_bias_geglu,
    fused_bias_swiglu,
    weighted_bias_quick_geglu_impl,
    weighted_bias_swiglu_impl as weighted_bias_swiglu_impl_geglu,
)
from deepspeed.core.fusions.fused_bias_swiglu import (
    BiasSwiGLUFunction,
    SwiGLUFunction,
    WeightedSwiGLUFunction,
    bias_swiglu,
    bias_swiglu_impl,
    swiglu,
    weighted_bias_swiglu_impl,
    weighted_swiglu,
)
from deepspeed.core.fusions.fused_layer_norm import FusedLayerNorm
from deepspeed.core.fusions.fused_softmax import (
    FusedScaleMaskSoftmax,
    ScaledMaskedSoftmax,
    ScaledSoftmax,
    ScaledUpperTriangMaskedSoftmax,
    SoftmaxOne,
    attention_mask_func,
)

__all__ = [
    # fused_bias_dropout
    "bias_dropout_add_fused_inference",
    "bias_dropout_add_fused_train",
    "bias_dropout_add_unfused",
    "get_bias_dropout_add",
    # fused_bias_geglu (existing)
    "fused_bias_geglu",
    "fused_bias_swiglu",
    "weighted_bias_quick_geglu_impl",
    # fused_bias_swiglu (new — full autograd)
    "BiasSwiGLUFunction",
    "SwiGLUFunction",
    "WeightedSwiGLUFunction",
    "bias_swiglu",
    "bias_swiglu_impl",
    "swiglu",
    "weighted_bias_swiglu_impl",
    "weighted_swiglu",
    # fused_layer_norm
    "FusedLayerNorm",
    # fused_softmax
    "FusedScaleMaskSoftmax",
    "ScaledMaskedSoftmax",
    "ScaledSoftmax",
    "ScaledUpperTriangMaskedSoftmax",
    "SoftmaxOne",
    "attention_mask_func",
]
