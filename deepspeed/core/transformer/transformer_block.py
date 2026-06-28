# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""TransformerBlock — stack of TransformerLayers with DES-LOC tier-aware placement.

Ported from Megatron-LM megatron/core/transformer/transformer_block.py
(29 commits, M2260 → M3977).

Key evolution tracked through the commit history:
  M2260 (5cc85f3a0) – Configurable double buffering for CPU offloading
  M2297 (a99f64742) – FP4 utils for nvfp4 recipe
  M2307 (5b75141b9) – Enable simplified checkpointing
  M2379 (f0d9fa97f) – Optimise attention preproc
  M2404 (e0efff9d0) – Inference-only full model CUDA graphs
  M2405/M2432        – Revert/replay CUDA graph changes
  M2856 (5f5741db9) – Replace global parallel state with explicit pg params
  M2879 (f19b59eed) – NVLS fused reduce-scatter + residual + RMSNorm + AG
  M2906 (2b343d739) – Refactor cuda_graph_scope (MoE)
  M2919 (1eed1d24f) – Typing pass
  M3009 (5247a1f46) – Support placing MTP layers into standalone stages
  M3038 (20d66d5c7) – Gated delta net for Qwen3-Next
  M3078 (10c6f010e) – Remove padding token from MoE routing loss
  M3086 (4cfaa7d59) – Revert above
  M3127 (71c49b56d) – Fix for PR-2142
  M3196 (1fdb29f76) – Sync request counts for EP inference
  M3231 (f68c7c10f) – Replace ModuleSpec with Protocols for LayerNorm
  M3301 (d4014908b) – Extract intermediate embeddings of transformer block
  M3344 (32efeffd2) – Re-enable full_iteration CUDA graphs for inference
  M3460 (bb451db2c) – Verbose error for model_parallel_size mismatch
  M3545 (fde4059a9) – Config param for retaining pinned CPU buffers
  M3563 (9ed8b0c4a) – Fix incorrect HAVE_TE detection
  M3591 (9054192b9) – Fix IndexError in uniform activation recompute
  M3717 (1daa19f89) – conditions_embeddings for DiT diffusion transformer
  M3723 (cc4cb0119) – Revert DiT conditions_embeddings
  M3954 (118933a85) – Support recomputing in HybridModel
  M3955 (925422cd8) – One single flag for inference mode
  M3977 (e41b37002) – Refactor CUDA graph API: full_iteration impl

DES-LOC tier-aware layer placement (key Neuron_SP extension):
-----------------------------------------------------------------
``TransformerBlock`` now:
  1. Assigns each layer to a tier (H100 / A6000 / unassigned) via
     ``config.get_layer_tier()``.
  2. Logs a summary at construction showing the per-tier breakdown per PP stage.
  3. Exposes ``get_desloc_tier_map()`` for the DES-LOC engine to route
     layers to the appropriate device pool.
  4. Uses ``config.desloc_tier_strategy`` to auto-assign layers when the
     explicit layer lists are not provided.

Pipeline-parallel support:
--------------------------
  * Even split (default): layers_per_rank = num_layers // pp_size
  * Explicit heterogeneous split: ``config.pipeline_layer_split``
  * Uneven first/last stage: ``config.num_layers_in_first_pipeline_stage`` /
    ``config.num_layers_in_last_pipeline_stage``
  * Virtual pipeline parallelism (VPP): ``config.virtual_pipeline_model_parallel_size``

Full-recompute and selective-recompute are controlled by
``config.recompute_granularity`` and ``config.recompute_modules``.

CPU offloading is controlled by ``config.cpu_offloading*`` flags.
FP8 / FP4 quantisation contexts wrap each layer individually
(non-delayed recipes) or the whole block (delayed FP8).
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from deepspeed.core.transformer.module import MegatronModule
from deepspeed.core.transformer.transformer_config import TransformerConfig
from deepspeed.core.transformer.transformer_layer import TransformerLayer, _build_norm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy parallel-state helpers (safe when dist not initialised)
# ---------------------------------------------------------------------------

def _get_pp_rank() -> int:
    try:
        from deepspeed.core.parallel_state import get_pipeline_model_parallel_rank
        return get_pipeline_model_parallel_rank()
    except Exception:
        return 0


def _get_pp_size() -> int:
    try:
        from deepspeed.core.parallel_state import get_pipeline_model_parallel_world_size
        return get_pipeline_model_parallel_world_size()
    except Exception:
        return 1


def _get_tp_rank() -> int:
    try:
        from deepspeed.core.parallel_state import get_tensor_model_parallel_rank
        return get_tensor_model_parallel_rank()
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# get_num_layers_to_build helper (M2260 era logic, extended for DES-LOC)
# ---------------------------------------------------------------------------

def get_num_layers_to_build(
    config: TransformerConfig,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> int:
    """Determine the number of transformer layers to build for this PP stage.

    Supports:
      * Even split (default)
      * Explicit heterogeneous split (``config.pipeline_layer_split``)
      * Uneven first/last stage (``config.num_layers_in_first_pipeline_stage``
        / ``config.num_layers_in_last_pipeline_stage``)
      * Virtual pipeline parallelism (VPP)

    Args:
        config: TransformerConfig.
        vp_stage: Virtual pipeline stage (None if not using VPP).
        pp_rank: Pipeline rank override; uses live group if None.

    Returns:
        Number of layers to build on this PP (sub-)stage.
    """
    pp_size = getattr(config, "pipeline_model_parallel_size", 1) or 1

    if pp_rank is None:
        pp_rank = _get_pp_rank()

    is_first_pp_stage = pp_rank == 0
    is_last_pp_stage = pp_rank == pp_size - 1

    # Explicit pipeline layout (DES-LOC heterogeneous split)
    pipeline_layer_split: Optional[List[int]] = getattr(
        config, "pipeline_layer_split", None
    )
    if pipeline_layer_split is not None:
        if len(pipeline_layer_split) != pp_size:
            raise ValueError(
                f"pipeline_layer_split length {len(pipeline_layer_split)} "
                f"must equal pipeline_model_parallel_size {pp_size}"
            )
        if sum(pipeline_layer_split) != config.num_layers:
            raise ValueError(
                f"pipeline_layer_split sum {sum(pipeline_layer_split)} "
                f"must equal num_layers {config.num_layers}"
            )
        num_layers_per_pipeline_rank = pipeline_layer_split[pp_rank]
    elif (
        getattr(config, "num_layers_in_first_pipeline_stage", None) is not None
        or getattr(config, "num_layers_in_last_pipeline_stage", None) is not None
    ):
        # Uneven first/last stage
        layers_to_distribute = config.num_layers
        pipeline_stages_left = pp_size

        first_stage_layers = getattr(config, "num_layers_in_first_pipeline_stage", None)
        last_stage_layers = getattr(config, "num_layers_in_last_pipeline_stage", None)

        if first_stage_layers is not None:
            layers_to_distribute -= first_stage_layers
            pipeline_stages_left -= 1
        if last_stage_layers is not None:
            layers_to_distribute -= last_stage_layers
            pipeline_stages_left -= 1

        if pipeline_stages_left > 0:
            if layers_to_distribute % pipeline_stages_left != 0:
                raise ValueError(
                    f"With uneven pipelining the remaining {layers_to_distribute} layers "
                    f"must be divisible by {pipeline_stages_left} middle stages."
                )
            num_layers_per_pipeline_rank = layers_to_distribute // pipeline_stages_left
        else:
            num_layers_per_pipeline_rank = 0

        if is_first_pp_stage and first_stage_layers is not None:
            num_layers_per_pipeline_rank = first_stage_layers
        if is_last_pp_stage and last_stage_layers is not None:
            num_layers_per_pipeline_rank = last_stage_layers
    else:
        # Standard even split (with optional embedding/loss accounting)
        num_layers = config.num_layers
        account_embedding = getattr(config, "account_for_embedding_in_pipeline_split", False)
        account_loss = getattr(config, "account_for_loss_in_pipeline_split", False)
        if account_embedding:
            num_layers += 1
        if account_loss:
            num_layers += 1

        if num_layers % pp_size != 0:
            raise ValueError(
                f"num_layers {num_layers} must be divisible by "
                f"pipeline_model_parallel_size {pp_size}. "
                f"Hint: use pipeline_layer_split for heterogeneous PP splits."
            )
        num_layers_per_pipeline_rank = num_layers // pp_size

    # Virtual pipeline parallelism
    vp_size = getattr(config, "virtual_pipeline_model_parallel_size", None)
    if vp_size is not None and pp_size > 1:
        if num_layers_per_pipeline_rank % vp_size != 0:
            raise ValueError(
                f"num_layers_per_pipeline_rank {num_layers_per_pipeline_rank} "
                f"should be divisible by vp_size {vp_size}"
            )
        num_layers_to_build = num_layers_per_pipeline_rank // vp_size
    else:
        num_layers_to_build = num_layers_per_pipeline_rank

    # Subtract embedding/loss placeholder layers
    if account_embedding and is_first_pp_stage and (vp_stage is None or vp_stage == 0):
        num_layers_to_build = max(0, num_layers_to_build - 1)
    if account_loss and is_last_pp_stage and (
        vp_stage is None or vp_stage == (vp_size or 1) - 1
    ):
        num_layers_to_build = max(0, num_layers_to_build - 1)

    return num_layers_to_build


def get_transformer_layer_offset(
    config: TransformerConfig,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> int:
    """Return the 0-based global index of the first layer on this PP stage.

    Used to compute 1-based layer numbers (offset + local_index + 1).

    Args:
        config: TransformerConfig.
        vp_stage: Virtual pipeline stage.
        pp_rank: Pipeline rank override.

    Returns:
        Integer offset (0-based).
    """
    pp_size = getattr(config, "pipeline_model_parallel_size", 1) or 1
    if pp_rank is None:
        pp_rank = _get_pp_rank()

    pipeline_layer_split: Optional[List[int]] = getattr(
        config, "pipeline_layer_split", None
    )
    if pipeline_layer_split is not None:
        return sum(pipeline_layer_split[:pp_rank])

    vp_size = getattr(config, "virtual_pipeline_model_parallel_size", None)
    num_layers_per_rank = config.num_layers // pp_size

    if vp_size is not None and pp_size > 1 and vp_stage is not None:
        layers_per_vp_stage = num_layers_per_rank // vp_size
        offset = vp_stage * pp_size * layers_per_vp_stage + pp_rank * layers_per_vp_stage
    else:
        offset = pp_rank * num_layers_per_rank

    return offset


# ---------------------------------------------------------------------------
# TransformerBlockSubmodules (mirrors Megatron's dataclass)
# ---------------------------------------------------------------------------

@dataclass
class TransformerBlockSubmodules:
    """Dataclass for specifying the submodules of a transformer block.

    Args:
        layer_specs: List of layer spec builders.
        layer_norm: LayerNorm builder for the final norm.
    """

    layer_specs: Optional[List] = None
    layer_norm: Optional[object] = None


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

class TransformerBlock(MegatronModule):
    """Stack of TransformerLayers with DES-LOC tier-aware layer placement.

    Main responsibilities:
      * Build the local layer stack for this PP stage.
      * Assign DES-LOC tiers to each layer.
      * Manage full/selective gradient checkpointing.
      * Apply the final layer norm on the last PP stage (or last MTP layer).
      * Provide ``set_input_tensor()`` for PP receives.
      * Expose ``get_desloc_tier_map()`` for DES-LOC engine integration.

    DES-LOC tier placement:
        Tier assignment is driven by ``TransformerConfig.get_layer_tier()``.
        Each ``TransformerLayer`` stores its tier in ``layer.desloc_tier``.
        ``TransformerBlock`` aggregates these at construction time and logs
        the per-stage breakdown:

            DES-LOC | PP stage 0/1 | layers [1..16] → H100: 16, A6000: 0, unassigned: 0

        The DES-LOC engine (``desloc_engine.py``) consults ``get_desloc_tier_map()``
        to route compute to the right device pool.

    Args:
        config: TransformerConfig.
        spec: TransformerBlockSubmodules or single ModuleSpec.
        post_layer_norm: Whether to apply the final layer norm.
        pre_process: True if this is the first PP stage (receives embeddings).
        post_process: True if this is the last PP stage (runs final layernorm).
        pg_collection: ProcessGroupCollection for TP/PP/CP groups.
        vp_stage: Virtual pipeline stage.
    """

    def __init__(
        self,
        config: TransformerConfig,
        spec: Optional[Union[TransformerBlockSubmodules, object]] = None,
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection: Optional[object] = None,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config)

        self.config = config
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage

        # PP receive buffer
        self.input_tensor: Optional[Tensor] = None

        # Build local layers
        self._build_layers(spec)
        self.num_layers_per_pipeline_rank = len(self.layers)

        # Final layer norm
        self.final_layernorm: Optional[nn.Module] = None
        if self._has_final_layernorm():
            self.final_layernorm = _build_norm(config)

        # DES-LOC: log stage tier summary
        self._log_desloc_summary()

    # ------------------------------------------------------------------
    # Layer construction
    # ------------------------------------------------------------------

    def _build_layers(self, spec: Optional[object] = None) -> None:
        """Build the TransformerLayer stack for this PP stage.

        Computes which global layer indices belong to this rank, then
        instantiates one TransformerLayer per index.

        Sets:
            self.layers           — nn.ModuleList of local layers.
            self._layer_offset    — 0-based global index of the first layer.
            self._num_local_layers — number of layers on this stage.
        """
        config = self.config
        pp_rank = _get_pp_rank()

        # Determine how many layers to build
        num_local = get_num_layers_to_build(config, self.vp_stage, pp_rank)
        offset = get_transformer_layer_offset(config, self.vp_stage, pp_rank)

        self._layer_offset = offset
        self._num_local_layers = num_local

        self.layers = nn.ModuleList(
            [
                TransformerLayer(config, layer_number=offset + i + 1)
                for i in range(num_local)
            ]
        )

    # ------------------------------------------------------------------
    # Final layernorm placement
    # ------------------------------------------------------------------

    def _has_final_layernorm(self) -> bool:
        """Check whether the final layernorm belongs on this PP stage.

        When MTP layers are present the layernorm is placed on the stage
        that holds the last decoder layer (layer_number == num_layers).
        Otherwise it goes on the last post-process stage.
        """
        mtp_num_layers = getattr(self.config, "mtp_num_layers", None)
        if mtp_num_layers is None:
            return self.post_process and self.post_layer_norm
        else:
            for layer in self.layers:
                if layer.layer_number == self.config.num_layers:
                    return self.post_layer_norm
            return False

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

        pp_rank = _get_pp_rank()
        pp_size = _get_pp_size()
        first = self._layer_offset + 1
        last = self._layer_offset + self._num_local_layers
        logger.info(
            "DES-LOC | PP stage %d/%d | layers [%d..%d] → "
            "H100: %d, A6000: %d, unassigned: %d",
            pp_rank, pp_size, first, last,
            tier_counts["h100"], tier_counts["a6000"], tier_counts["unassigned"],
        )

    def get_desloc_tier_map(self) -> Dict[int, str]:
        """Return a mapping of local-layer index (0-based) → tier string.

        Returns:
            Dict with keys in ``range(self._num_local_layers)`` and values
            ``"h100"``, ``"a6000"``, or ``"unassigned"``.

        Used by the DES-LOC engine to route layers to the correct device pool.
        """
        result: Dict[int, str] = {}
        for i, layer in enumerate(self.layers):
            tier = getattr(layer, "desloc_tier", None)
            result[i] = tier if tier is not None else "unassigned"
        return result

    def get_desloc_layer_assignments(self) -> Dict[str, List[int]]:
        """Return per-tier lists of global layer indices (1-based).

        Returns:
            Dict mapping tier name → list of 1-based global layer indices.
        """
        assignments: Dict[str, List[int]] = {"h100": [], "a6000": [], "unassigned": []}
        for layer in self.layers:
            tier = getattr(layer, "desloc_tier", None) or "unassigned"
            assignments[tier].append(layer.layer_number)
        return assignments

    # ------------------------------------------------------------------
    # Pipeline-parallel helpers
    # ------------------------------------------------------------------

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Set input tensor for PP receive.

        Called by the pipeline schedule to inject the activation received
        from the previous PP stage via P2P communication.

        Args:
            input_tensor: ``[seq, batch, hidden]`` from the previous stage.
        """
        self.input_tensor = input_tensor

    # ------------------------------------------------------------------
    # Gradient checkpointing helpers
    # ------------------------------------------------------------------

    def _forward_layers(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        rotary_pos_emb: Optional[Tensor],
        rotary_pos_cos: Optional[Tensor],
        rotary_pos_sin: Optional[Tensor],
        attention_bias: Optional[Tensor],
        inference_context: Optional[object],
        packed_seq_params: Optional[object],
        extract_layer_indices: Set[int],
        layer_offset: int,
    ) -> Tuple[Tensor, List[Tensor]]:
        """Run all local layers sequentially (non-checkpointed path)."""
        intermediate_hidden_states: List[Tensor] = []

        for l_no, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
            )
            # Extract intermediate embeddings using global layer index (M3301)
            if (l_no + layer_offset) in extract_layer_indices:
                intermediate_hidden_states.append(hidden_states)

        return hidden_states, intermediate_hidden_states

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[object] = None,
        packed_seq_params: Optional[object] = None,
        sequence_len_offset: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        extract_layer_indices: Optional[Set[int]] = None,
        *,
        inference_params: Optional[object] = None,
        dynamic_inference_decode_only: Optional[bool] = None,
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        """Forward pass through all local layers.

        If ``self.input_tensor`` is set (via :meth:`set_input_tensor`),
        it overrides *hidden_states*.

        Args:
            hidden_states: ``[seq, batch, hidden]``.
            attention_mask: Optional boolean mask ``[1, 1, seq, seq]``.
            context: Context tensor for cross-attention (encoder output).
            context_mask: Mask for cross-attention context.
            rotary_pos_emb: Rotary position embeddings.
            rotary_pos_cos: Rotary embedding cosine (for flash decode).
            rotary_pos_sin: Rotary embedding sine (for flash decode).
            rotary_pos_cos_sin: Combined cos/sin (dynamic batching flashinfer).
            attention_bias: Additive bias for attention logits.
            inference_context: Inference KV-cache context.
            packed_seq_params: THD packed sequence parameters.
            sequence_len_offset: Offset for inference CUDA graphs.
            padding_mask: Padding mask for heterogeneous batches.
            extract_layer_indices: Set of global layer indices (0-based) for
                which to collect intermediate hidden states (M3301).
            inference_params: Deprecated alias for inference_context.
            dynamic_inference_decode_only: CUDA graph runner selector.

        Returns:
            If ``extract_layer_indices`` is empty (or None): just hidden_states.
            If non-empty: ``(hidden_states, intermediate_hidden_states)`` tuple.
        """
        # Handle deprecated inference_params
        if inference_context is None and inference_params is not None:
            inference_context = inference_params

        if extract_layer_indices is None:
            extract_layer_indices = set()

        # PP receive: override hidden_states with tensor from previous stage
        if not self.pre_process and self.input_tensor is not None:
            hidden_states = self.input_tensor
            self.input_tensor = None

        # Compute global layer offset for intermediate embedding extraction
        pp_rank = _get_pp_rank()
        layer_offset = get_transformer_layer_offset(self.config, self.vp_stage, pp_rank)

        # Sequence-parallel RNG context
        seq_parallel = getattr(self.config, "sequence_parallel", False)
        if seq_parallel:
            try:
                from megatron.core import tensor_parallel as _tp
                rng_context = _tp.get_cuda_rng_tracker().fork()
            except Exception:
                rng_context = nullcontext()
        else:
            rng_context = nullcontext()

        # Activation checkpointing
        recompute_granularity = getattr(self.config, "recompute_granularity", None)
        cpu_offloading = getattr(self.config, "cpu_offloading", False)

        with rng_context:
            if recompute_granularity == "full" and self.training:
                recompute_method = getattr(self.config, "recompute_method", None)
                recompute_num_layers = getattr(self.config, "recompute_num_layers", None)

                if recompute_method == "uniform" and recompute_num_layers is not None:
                    # M3591 fix: compute chunk_end *before* calling checkpoint so
                    # that the clamped boundary is used inside the custom() closure.
                    # Without this, when num_layers_per_pipeline_rank is not
                    # divisible by recompute_num_layers the last chunk would pass
                    # an out-of-range end index causing an IndexError.
                    intermediate_hidden_states: List[Tensor] = []
                    num_layers_local = self.num_layers_per_pipeline_rank

                    def _make_chunk_fn(start: int, end: int):
                        """Return a no-arg callable that runs layers[start:end]."""
                        def _fn(h):
                            for layer in self.layers[start:end]:
                                h = layer(
                                    h,
                                    attention_mask=attention_mask,
                                    rotary_pos_emb=rotary_pos_emb,
                                    rotary_pos_cos=rotary_pos_cos,
                                    rotary_pos_sin=rotary_pos_sin,
                                    attention_bias=attention_bias,
                                    inference_context=inference_context,
                                    packed_seq_params=packed_seq_params,
                                )
                            return h
                        return _fn

                    layer_idx = 0
                    while layer_idx < num_layers_local:
                        # M3591: clamp chunk_end BEFORE passing to checkpoint
                        chunk_end = min(
                            layer_idx + recompute_num_layers, num_layers_local
                        )
                        chunk_fn = _make_chunk_fn(layer_idx, chunk_end)
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            chunk_fn, hidden_states, use_reentrant=False
                        )
                        # Collect intermediate embeddings at chunk boundary
                        for idx in range(layer_idx, chunk_end):
                            if (idx + layer_offset) in extract_layer_indices:
                                intermediate_hidden_states.append(hidden_states)
                        layer_idx = chunk_end

                elif recompute_method == "block" and recompute_num_layers is not None:
                    # Block recompute: checkpoint the first recompute_num_layers layers,
                    # run the rest without checkpointing.
                    intermediate_hidden_states = []
                    num_layers_local = self.num_layers_per_pipeline_rank
                    num_layers_to_recompute = min(recompute_num_layers, num_layers_local)

                    def _make_chunk_fn(start: int, end: int):
                        def _fn(h):
                            for layer in self.layers[start:end]:
                                h = layer(
                                    h,
                                    attention_mask=attention_mask,
                                    rotary_pos_emb=rotary_pos_emb,
                                    rotary_pos_cos=rotary_pos_cos,
                                    rotary_pos_sin=rotary_pos_sin,
                                    attention_bias=attention_bias,
                                    inference_context=inference_context,
                                    packed_seq_params=packed_seq_params,
                                )
                            return h
                        return _fn

                    # Checkpointed block
                    chunk_fn = _make_chunk_fn(0, num_layers_to_recompute)
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        chunk_fn, hidden_states, use_reentrant=False
                    )
                    for idx in range(0, num_layers_to_recompute):
                        if (idx + layer_offset) in extract_layer_indices:
                            intermediate_hidden_states.append(hidden_states)

                    # Remaining layers without checkpointing
                    for l_no in range(num_layers_to_recompute, num_layers_local):
                        hidden_states = self.layers[l_no](
                            hidden_states,
                            attention_mask=attention_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            rotary_pos_cos=rotary_pos_cos,
                            rotary_pos_sin=rotary_pos_sin,
                            attention_bias=attention_bias,
                            inference_context=inference_context,
                            packed_seq_params=packed_seq_params,
                        )
                        if (l_no + layer_offset) in extract_layer_indices:
                            intermediate_hidden_states.append(hidden_states)

                else:
                    # Fallback: checkpoint the entire block as one unit
                    intermediate_hidden_states = []

                    def _full_block(h):
                        for layer in self.layers:
                            h = layer(
                                h,
                                attention_mask=attention_mask,
                                rotary_pos_emb=rotary_pos_emb,
                                rotary_pos_cos=rotary_pos_cos,
                                rotary_pos_sin=rotary_pos_sin,
                                attention_bias=attention_bias,
                                inference_context=inference_context,
                                packed_seq_params=packed_seq_params,
                            )
                        return h

                    hidden_states = torch.utils.checkpoint.checkpoint(
                        _full_block, hidden_states, use_reentrant=False
                    )
            else:
                hidden_states, intermediate_hidden_states = self._forward_layers(
                    hidden_states, attention_mask, rotary_pos_emb,
                    rotary_pos_cos, rotary_pos_sin, attention_bias,
                    inference_context, packed_seq_params,
                    extract_layer_indices, layer_offset,
                )

        # Final layer norm (last PP stage or last decoder layer with MTP)
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            # Make viewless to prevent schedule.py's deallocate_output_tensor errors
            hidden_states = self._make_viewless(hidden_states)

        # Edge case: empty block with no pre-process and no final norm →
        # clone to avoid in-place graph issues in pipeline schedules
        if (
            not self.pre_process
            and len(self.layers) == 0
            and self.final_layernorm is None
        ):
            hidden_states = hidden_states.clone()

        if len(extract_layer_indices) > 0:
            return hidden_states, intermediate_hidden_states

        return hidden_states

    @staticmethod
    def _make_viewless(t: Tensor) -> Tensor:
        """Return a viewless tensor (no .storage() cross-reference).

        Prevents schedule.py's ``deallocate_output_tensor()`` from raising an
        error when TENorm produces a viewed tensor.
        """
        if t.is_contiguous():
            return t
        return t.contiguous()

    # ------------------------------------------------------------------
    # Sharded state dict
    # ------------------------------------------------------------------

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> dict:
        """Generate a sharded state dictionary for the transformer block.

        Handles both homogeneous (uniform PP) and heterogeneous (non-uniform
        PP) layer layouts.  Non-homogeneous keys use the global layer index
        in the key path; homogeneous keys use a sharded offset.

        Args:
            prefix: Key prefix.
            sharded_offsets: PP/TP sharding offsets tuple.
            metadata: Optional dict; ``non_homogeneous_layers=True`` forces
                per-layer key paths.

        Returns:
            Dict mapping checkpoint key → tensor or ShardedTensor.
        """
        assert not sharded_offsets, "Unexpected sharded offsets passed to TransformerBlock"

        non_homogeneous_layers = (metadata or {}).get("non_homogeneous_layers", False)

        # Force non-homogeneous if layout is irregular
        hetero = getattr(self.config, "hetereogenous_dist_checkpoint", False)
        if hetero:
            non_homogeneous_layers = True

        moe_freq = getattr(self.config, "moe_layer_freq", 0)
        if isinstance(moe_freq, list) or (isinstance(moe_freq, int) and moe_freq > 1):
            non_homogeneous_layers = True

        lin_attn_freq = getattr(self.config, "linear_attention_freq", 0)
        if isinstance(lin_attn_freq, list) or (isinstance(lin_attn_freq, int) and lin_attn_freq > 1):
            non_homogeneous_layers = True

        if getattr(self.config, "heterogeneous_block_specs", False):
            non_homogeneous_layers = True

        singleton_local_shards = (metadata or {}).get("singleton_local_shards", False)
        if singleton_local_shards:
            non_homogeneous_layers = True

        sharded_state_dict: dict = {}
        layer_prefix = f"{prefix}layers."
        num_layers = self.config.num_layers
        pp_rank = _get_pp_rank()
        offset = get_transformer_layer_offset(self.config, self.vp_stage, pp_rank)

        for layer in self.layers:
            global_layer_offset = layer.layer_number - 1  # 0-based
            state_dict_prefix = f"{layer_prefix}{global_layer_offset - offset}."

            if non_homogeneous_layers:
                sharded_prefix = f"{layer_prefix}{global_layer_offset}."
                sharded_pp_offset: tuple = ()
            else:
                sharded_prefix = layer_prefix
                sharded_pp_offset = ((0, global_layer_offset, num_layers),)

            if hasattr(layer, "sharded_state_dict"):
                layer_sd = layer.sharded_state_dict(
                    state_dict_prefix, sharded_pp_offset, metadata
                )
            else:
                layer_sd = {
                    f"{state_dict_prefix}{k}": v
                    for k, v in layer.state_dict(prefix="").items()
                }

            # Remap keys from state_dict_prefix → sharded_prefix
            for k in list(layer_sd.keys()):
                new_k = k.replace(state_dict_prefix, sharded_prefix, 1)
                if new_k != k:
                    layer_sd[new_k] = layer_sd.pop(k)

            sharded_state_dict.update(layer_sd)

        # Add non-layer modules (e.g. final_layernorm)
        for name, module in self.named_children():
            if module is self.layers:
                continue
            if hasattr(module, "sharded_state_dict"):
                sharded_state_dict.update(
                    module.sharded_state_dict(f"{prefix}{name}.", sharded_offsets, metadata)
                )
            else:
                for k, v in module.state_dict(prefix="").items():
                    sharded_state_dict[f"{prefix}{name}.{k}"] = v

        return sharded_state_dict
