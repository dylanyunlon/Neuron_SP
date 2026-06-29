"""Fused kernels for heterogeneous GPU clusters.

Mirrors Megatron megatron/core/fusions/ but dispatches different kernel
implementations per compute capability (SM86 for A6000, SM90 for H100).

Key fusions:
- fused_bias_geglu / fused_bias_swiglu: bias + activation in one kernel
- fused_softmax: memory-efficient softmax (critical for A6000 VRAM)
"""
from deepspeed.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from deepspeed.core.fusions.fused_bias_geglu import fused_bias_geglu, fused_bias_swiglu

__all__ = ["FusedScaleMaskSoftmax", "fused_bias_geglu", "fused_bias_swiglu"]
