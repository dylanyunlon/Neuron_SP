# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""TransformerConfig — ported from Megatron-LM with DES-LOC per-layer tier assignment.

DES-LOC extension
-----------------
Two new fields control heterogeneous GPU placement:

  desloc_h100_layers : Optional[List[int]]
      Zero-based global layer indices that should reside on H100 GPUs.
      When None (default), no assignment is enforced.

  desloc_a6000_layers : Optional[List[int]]
      Zero-based global layer indices that should reside on A6000 GPUs.
      When None (default), no assignment is enforced.

  desloc_tier_strategy : Literal["front_heavy", "back_heavy", "interleave", "manual"]
      Automatic tier assignment strategy used when desloc_h100_layers /
      desloc_a6000_layers are not set manually:
        "front_heavy" — first N layers go to H100 (faster, more VRAM).
        "back_heavy"  — last N layers go to H100.
        "interleave"  — every other layer alternates H100/A6000.
        "manual"      — no automatic assignment; caller fills the lists.

Usage example::

    cfg = TransformerConfig(
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        desloc_tier_strategy="front_heavy",
        desloc_h100_layer_fraction=0.5,   # put first 50% on H100
    )
    # TransformerConfig.__post_init__ calls _resolve_desloc_tiers()
    h100 = cfg.desloc_h100_layers   # [0..15]
    a6000 = cfg.desloc_a6000_layers # [16..31]
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from deepspeed.core.model_parallel_config import ModelParallelConfig

logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig(ModelParallelConfig):
    """Full transformer configuration extending ModelParallelConfig.

    Closely mirrors Megatron-LM's TransformerConfig with the following
    DeepSpeed / DES-LOC additions:

    * ``desloc_h100_layers`` / ``desloc_a6000_layers`` — explicit per-layer
      tier assignment lists (zero-based global indices).
    * ``desloc_tier_strategy`` — automatic assignment strategy.
    * ``desloc_h100_layer_fraction`` — fraction of layers placed on H100 when
      using "front_heavy" or "back_heavy" strategies.
    * ``pipeline_layer_split`` — heterogeneous PP split retained for backward
      compatibility with the existing TransformerBlock.
    """

    # ------------------------------------------------------------------
    # Core model architecture
    # ------------------------------------------------------------------

    num_layers: int = 0
    """Number of transformer layers in a transformer block."""

    hidden_size: int = 0
    """Transformer hidden size."""

    num_attention_heads: int = 0
    """Number of transformer attention heads."""

    num_query_groups: Optional[int] = None
    """Number of query groups for group query attention (GQA/MQA).
    If None, falls back to num_attention_heads (standard MHA)."""

    ffn_hidden_size: Optional[int] = None
    """Feed-Forward Network hidden size. Defaults to 4*hidden_size for
    standard transformers; SwiGLU default is 8/3*hidden_size rounded up."""

    kv_channels: Optional[int] = None
    """Projection weights dimension in multi-head attention.
    Defaults to hidden_size // num_attention_heads."""

    # ------------------------------------------------------------------
    # Regularisation
    # ------------------------------------------------------------------

    hidden_dropout: float = 0.0
    """Dropout probability for transformer hidden state."""

    attention_dropout: float = 0.0
    """Post-attention dropout probability."""

    layernorm_epsilon: float = 1e-5
    """Epsilon value for any LayerNorm / RMSNorm operations."""

    # ------------------------------------------------------------------
    # Linear layers
    # ------------------------------------------------------------------

    add_bias_linear: bool = False
    """Include a bias term in all linear layers."""

    # ------------------------------------------------------------------
    # Activation / gating
    # ------------------------------------------------------------------

    gated_linear_unit: bool = False
    """Use a gated linear unit (SwiGLU / GeGLU) for the first MLP linear."""

    activation_func: Callable = F.silu
    """Activation function for the MLP non-linearity (SiLU / SwiGLU default)."""

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    normalization: Literal["LayerNorm", "RMSNorm"] = "RMSNorm"
    """Which norm to use, valid options are LayerNorm and RMSNorm."""

    apply_residual_connection_post_layernorm: bool = False
    """If True, applies residual connection after the layer norm (post-norm).
    Default is pre-norm (residual before norm)."""

    # ------------------------------------------------------------------
    # Positional embeddings
    # ------------------------------------------------------------------

    rotary_interleaved: bool = False
    """True: rotate pairs of even/odd dims (RoFormer style).
    False: rotate first-half/second-half (LLaMA style)."""

    window_size: Optional[Tuple[int, int]] = None
    """Sliding window attention size. None = full attention."""

    # ------------------------------------------------------------------
    # Mixture-of-Experts
    # ------------------------------------------------------------------

    num_moe_experts: Optional[int] = None
    """Number of MoE experts. When set, MLP is replaced by MoELayer."""

    # ------------------------------------------------------------------
    # Multi-Token Prediction (MTP)
    # ------------------------------------------------------------------

    mtp_num_layers: Optional[int] = None
    """Number of Multi-Token Prediction (MTP) layers."""

    mtp_loss_scaling_factor: float = 0.1
    """Scaling factor applied to the MTP auxiliary loss."""

    # ------------------------------------------------------------------
    # Multi-Latent Attention (MLA / DeepSeek style)
    # ------------------------------------------------------------------

    multi_latent_attention: bool = False
    """Whether to use Multi-Latent Attention."""

    # ------------------------------------------------------------------
    # Per-token loss
    # ------------------------------------------------------------------

    calculate_per_token_loss: bool = False
    """Whether cross-entropy loss is calculated over non-padded tokens only."""

    # ------------------------------------------------------------------
    # Precision / mixed-precision
    # ------------------------------------------------------------------

    fp32_residual_connection: bool = False
    """If True, move residual connections to fp32."""

    attention_softmax_in_fp32: bool = True
    """If True, run attention softmax in fp32."""

    # ------------------------------------------------------------------
    # Activation recomputation
    # ------------------------------------------------------------------

    recompute_granularity: Optional[Literal["full", "selective"]] = None
    """Activation recomputation granularity. None = no recompute."""

    recompute_method: Optional[Literal["uniform", "block"]] = None
    """Which layers to recompute. None = all layers."""

    recompute_num_layers: Optional[int] = None
    """Number of transformer layers per recompute unit."""

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    init_method_std: float = 0.02
    """Standard deviation of the zero-mean normal for default initialisation."""

    init_method: Optional[Callable] = None
    """Weight initialisation callable. Defaults to normal(init_method_std)."""

    output_layer_init_method: Optional[Callable] = None
    """Initialisation for output layers (attention + MLP output projections)."""

    # ------------------------------------------------------------------
    # PP layer split (backwards compat with existing TransformerBlock)
    # ------------------------------------------------------------------

    pipeline_layer_split: Optional[List[int]] = None
    """Explicit heterogeneous PP split: list of per-stage layer counts.
    len must equal pipeline_model_parallel_size and sum must equal num_layers."""

    # ------------------------------------------------------------------
    # DES-LOC: per-layer GPU tier assignment
    # ------------------------------------------------------------------

    desloc_h100_layers: Optional[List[int]] = field(default=None, repr=False)
    """Zero-based global layer indices assigned to H100 GPUs.
    Populated automatically by _resolve_desloc_tiers() unless set manually."""

    desloc_a6000_layers: Optional[List[int]] = field(default=None, repr=False)
    """Zero-based global layer indices assigned to A6000 GPUs.
    Populated automatically by _resolve_desloc_tiers() unless set manually."""

    desloc_tier_strategy: Literal[
        "front_heavy", "back_heavy", "interleave", "manual"
    ] = "front_heavy"
    """Automatic tier-assignment strategy.

    front_heavy : First fraction of layers → H100, rest → A6000.
    back_heavy  : Last fraction of layers → H100, rest → A6000.
    interleave  : Even-indexed layers → H100, odd-indexed → A6000.
    manual      : No automatic assignment (caller fills the lists).
    """

    desloc_h100_layer_fraction: float = 0.5
    """Fraction [0, 1] of total layers placed on H100 GPUs when using
    front_heavy or back_heavy strategies. Ignored for interleave/manual."""

    desloc_tier_map: Optional[Dict[int, str]] = field(default=None, repr=False)
    """Read-only dict built by _resolve_desloc_tiers(): layer_idx → "h100" | "a6000".
    Useful for quick per-layer look-ups at run time."""

    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate and derive derived fields."""
        # Call parent __post_init__ only if the parent defines it.
        parent_post = getattr(super(), '__post_init__', None)
        if callable(parent_post):
            parent_post()

        # --- Derived defaults ------------------------------------------
        if self.ffn_hidden_size is None:
            # SwiGLU default: 8/3 * hidden, rounded up to multiple of 64
            raw = int(self.hidden_size * 8 / 3)
            self.ffn_hidden_size = ((raw + 63) // 64) * 64

        if self.kv_channels is None:
            if self.num_attention_heads > 0:
                self.kv_channels = self.hidden_size // self.num_attention_heads

        if self.num_query_groups is None:
            self.num_query_groups = self.num_attention_heads

        # --- Initialisation methods ------------------------------------
        if self.init_method is None and self.init_method_std > 0:
            self._set_default_init_methods()

        # --- DES-LOC tier resolution -----------------------------------
        self._resolve_desloc_tiers()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _set_default_init_methods(self) -> None:
        """Set default weight initialisation callables."""
        import torch.nn.init as init

        std = self.init_method_std

        def _normal_init(tensor: torch.Tensor) -> None:
            init.normal_(tensor, mean=0.0, std=std)

        def _scaled_init(tensor: torch.Tensor) -> None:
            scaled_std = std / math.sqrt(2.0 * max(self.num_layers, 1))
            init.normal_(tensor, mean=0.0, std=scaled_std)

        self.init_method = _normal_init
        self.output_layer_init_method = _scaled_init

    def _resolve_desloc_tiers(self) -> None:
        """Populate desloc_h100_layers / desloc_a6000_layers / desloc_tier_map.

        Skipped when num_layers == 0 (config not yet fully specified) or
        when strategy is "manual" and both lists are already provided.
        """
        if self.num_layers <= 0:
            return  # config not ready yet

        # If both lists already set (manual or pre-filled), just build the map
        if self.desloc_h100_layers is not None and self.desloc_a6000_layers is not None:
            self._build_tier_map()
            return

        if self.desloc_tier_strategy == "manual":
            # Caller promised to fill in manually; nothing to do here
            return

        n = self.num_layers
        strategy = self.desloc_tier_strategy
        frac = max(0.0, min(1.0, self.desloc_h100_layer_fraction))
        n_h100 = int(round(n * frac))

        if strategy == "front_heavy":
            h100 = list(range(n_h100))
            a6000 = list(range(n_h100, n))

        elif strategy == "back_heavy":
            n_a6000 = n - n_h100
            a6000 = list(range(n_a6000))
            h100 = list(range(n_a6000, n))

        elif strategy == "interleave":
            h100 = list(range(0, n, 2))    # even layers
            a6000 = list(range(1, n, 2))   # odd layers

        else:
            raise ValueError(
                f"Unknown desloc_tier_strategy: {self.desloc_tier_strategy!r}. "
                "Valid values: 'front_heavy', 'back_heavy', 'interleave', 'manual'."
            )

        self.desloc_h100_layers = h100
        self.desloc_a6000_layers = a6000
        self._build_tier_map()

        logger.debug(
            "DES-LOC tier assignment (%s): %d layers → H100, %d layers → A6000",
            strategy,
            len(h100),
            len(a6000),
        )

    def _build_tier_map(self) -> None:
        """Build the layer_idx → tier string look-up dict."""
        tier_map: Dict[int, str] = {}
        if self.desloc_h100_layers:
            for idx in self.desloc_h100_layers:
                tier_map[idx] = "h100"
        if self.desloc_a6000_layers:
            for idx in self.desloc_a6000_layers:
                tier_map[idx] = "a6000"
        self.desloc_tier_map = tier_map

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_layer_tier(self, layer_idx: int) -> Optional[str]:
        """Return the tier string ("h100" | "a6000") for a global layer index.

        Args:
            layer_idx: Zero-based global layer index.

        Returns:
            "h100", "a6000", or None if the layer has no tier assignment.
        """
        if self.desloc_tier_map is None:
            return None
        return self.desloc_tier_map.get(layer_idx)

    def is_h100_layer(self, layer_idx: int) -> bool:
        """Return True if *layer_idx* is assigned to an H100 GPU."""
        return self.get_layer_tier(layer_idx) == "h100"

    def is_a6000_layer(self, layer_idx: int) -> bool:
        """Return True if *layer_idx* is assigned to an A6000 GPU."""
        return self.get_layer_tier(layer_idx) == "a6000"
