"""Shared enumerations for model parallelism and training modes.

Mirrors Megatron megatron/core/enums.py.
"""
from __future__ import annotations

import enum


class ModelType(enum.Enum):
    """Type of model for pipeline schedule selection."""
    encoder_or_decoder = 1
    encoder_and_decoder = 2
    retro_encoder = 3
    retro_decoder = 4


class AttnMaskType(enum.Enum):
    """Attention mask type for transformer layers."""
    padding = 1
    causal = 2
    no_mask = 3
    arbitrary = 4


class AttnBackend(enum.Enum):
    """Attention computation backend."""
    flash = "flash"
    fused = "fused"
    unfused = "unfused"
    local = "local"


class LayerType(enum.Enum):
    """Transformer layer type for heterogeneous stacks."""
    encoder = 1
    decoder = 2
    retro_encoder = 3
    retro_decoder = 4
    retro_decoder_with_retriever = 5
