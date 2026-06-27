# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""TransformerBlock — stack of TransformerLayers with DES-LOC tier assignment.

DES-LOC integration
-------------------
``TransformerBlock`` is the central place where DES-LOC tier assignment
is *displayed* (via logging) and *summarised* at model build time.  Each
:class:`TransformerLayer` already stores ``desloc_tier``; ``TransformerBlock``
collects these into a structured per-stage summary.

Pipeline-parallel support
-------------------------
In PP mode each rank holds a contiguous subset of global layers.  The subset
is determined by either:

1. ``config.pipeline_layer_split`` — explicit heterogeneous split.
   ``len(pipeline_layer_split)`` must equal ``pipeline_model_parallel_size``
   and their sum must equal ``num_layers``.
2. Even split — ``num_layers // pipeline_model_parallel_size`` layers per stage.

The first stage receives input activations from the embedding layer; every
other stage receives its input via ``set_input_tensor()``.  The last stage
applies a final layer norm before returning.

Usage example::

    block = TransformerBlock(config)
    # During PP receive (called by pipeline schedule):
    block.set_input_tensor(received_tensor)
    output = block(hidden_states)   # hidden_states ignored when input_tensor set
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .transformer_config import TransformerConfig
from .module import MegatronModule
from .transformer_layer import TransformerLayer, _build_norm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_pp_info(config: TransformerConfig) -> Tuple[int, int]:
    """Return ``(pp_rank, pp_size)`` safely.

    Tries the live distributed process group first, then falls back to the
    config for the size (rank → 0).  This allows unit tests that skip
    ``torch.distributed.init_process_group`` to still work.
    """
    pp_size_from_cfg: int = getattr(
        config, "pipeline_model_parallel_size", 1
    ) or 1
    try:
        from deepspeed.core.parallel_state import (
            get_pipeline_model_parallel_rank,
            get_pipeline_model_parallel_world_size,
        )
        rank = get_pipeline_model_parallel_rank()
        size = get_pipeline_model_parallel_world_size()
        return rank, size
    except Exception:
        return 0, pp_size_from_cfg


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

class TransformerBlock(MegatronModule):
    """Stack of :class:`TransformerLayer`s with optional final layer norm.

    In PP mode each rank owns a contiguous subset of layers; which subset is
    determined by :meth:`_build_layers`.  The last PP stage applies the final
    layer norm.

    DES-LOC extension:
        At construction time the block logs a per-stage tier summary showing
        how many layers are assigned to H100 vs A6000.  Individual
        ``TransformerLayer`` objects carry their own ``desloc_tier`` attribute.

    Args:
        config: TransformerConfig driving all sub-module construction.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)

        # PP receive buffer
        self.input_tensor: Optional[torch.Tensor] = None

        # Build local layers
        self._build_layers()

        # Final layer norm (last PP stage only)
        self.final_layernorm: Optional[nn.Module] = None
        pp_rank, pp_size = _get_pp_info(config)
        if pp_rank == pp_size - 1:
            self.final_layernorm = _build_norm(config)

        # DES-LOC: log stage tier summary
        self._log_desloc_summary()

    # ------------------------------------------------------------------
    # Layer construction
    # ------------------------------------------------------------------

    def _build_layers(self) -> None:
        """Build the TransformerLayer stack for this PP stage.

        Determines which global layer indices (1-based, Megatron convention)
        belong to this PP rank, then instantiates one TransformerLayer per
        index.

        Sets:
            * ``self.layers`` — the :class:`nn.ModuleList` of local layers.
            * ``self._layer_offset`` — global index of the first local layer.
            * ``self._num_local_layers`` — number of layers on this stage.
        """
        config = self.config
        pp_rank, pp_size = _get_pp_info(config)
        total_layers: int = config.num_layers

        pipeline_layer_split: Optional[List[int]] = getattr(
            config, "pipeline_layer_split", None
        )

        if pipeline_layer_split is not None:
            # Explicit heterogeneous split
            if len(pipeline_layer_split) != pp_size:
                raise ValueError(
                    f"pipeline_layer_split length {len(pipeline_layer_split)} "
                    f"must equal pipeline_model_parallel_size {pp_size}"
                )
            if sum(pipeline_layer_split) != total_layers:
                raise ValueError(
                    f"pipeline_layer_split sum {sum(pipeline_layer_split)} "
                    f"must equal num_layers {total_layers}"
                )
            offset = sum(pipeline_layer_split[:pp_rank])
            num_local = pipeline_layer_split[pp_rank]
        else:
            # Even split
            if total_layers % pp_size != 0:
                raise ValueError(
                    f"num_layers {total_layers} must be divisible by "
                    f"pipeline_model_parallel_size {pp_size} when "
                    "pipeline_layer_split is None"
                )
            layers_per_rank = total_layers // pp_size
            offset = pp_rank * layers_per_rank
            num_local = layers_per_rank

        self._layer_offset: int = offset
        self._num_local_layers: int = num_local

        # Instantiate (layer_number is 1-based globally)
        self.layers = nn.ModuleList(
            [
                TransformerLayer(config, layer_number=offset + i + 1)
                for i in range(num_local)
            ]
        )

    # ------------------------------------------------------------------
    # DES-LOC helpers
    # ------------------------------------------------------------------

    def _log_desloc_summary(self) -> None:
        """Log the DES-LOC tier breakdown for this PP stage."""
        tier_counts: Dict[str, int] = {"h100": 0, "a6000": 0, "unassigned": 0}
        for layer in self.layers:
            tier = getattr(layer, "desloc_tier", None)
            if tier == "h100":
                tier_counts["h100"] += 1
            elif tier == "a6000":
                tier_counts["a6000"] += 1
            else:
                tier_counts["unassigned"] += 1

        pp_rank, pp_size = _get_pp_info(self.config)
        logger.info(
            "DES-LOC | PP stage %d/%d | layers [%d..%d] → "
            "H100: %d, A6000: %d, unassigned: %d",
            pp_rank,
            pp_size,
            self._layer_offset + 1,
            self._layer_offset + self._num_local_layers,
            tier_counts["h100"],
            tier_counts["a6000"],
            tier_counts["unassigned"],
        )

    def get_desloc_tier_map(self) -> Dict[int, str]:
        """Return a mapping of local-layer index (0-based) → tier string.

        Returns:
            Dict with keys in ``range(self._num_local_layers)`` and values
            "h100", "a6000", or "unassigned".
        """
        result: Dict[int, str] = {}
        for i, layer in enumerate(self.layers):
            tier = getattr(layer, "desloc_tier", None)
            result[i] = tier if tier is not None else "unassigned"
        return result

    # ------------------------------------------------------------------
    # Pipeline-parallel helpers
    # ------------------------------------------------------------------

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Set input tensor for PP receive.

        Called by the pipeline schedule to inject the activation received
        from the previous PP stage via P2P communication.

        Args:
            input_tensor: ``[seq, batch, hidden]`` from the previous stage.
        """
        self.input_tensor = input_tensor

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        inference_params: Optional[object] = None,
    ) -> torch.Tensor:
        """Forward pass through all local layers on this PP stage.

        If ``self.input_tensor`` is set (PP receive), it overrides
        *hidden_states*.

        Args:
            hidden_states: ``[seq, batch, hidden]`` (ignored when input_tensor
                is set by a prior :meth:`set_input_tensor` call).
            attention_mask: Optional mask ``[batch, 1, seq, seq]``.
            rotary_pos_emb: Optional rotary embeddings.
            inference_params: Passed through to each layer.

        Returns:
            output: ``[seq, batch, hidden]``
        """
        if self.input_tensor is not None:
            hidden_states = self.input_tensor
            self.input_tensor = None

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                inference_params=inference_params,
            )

        # Apply final layer norm on the last PP stage
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states
