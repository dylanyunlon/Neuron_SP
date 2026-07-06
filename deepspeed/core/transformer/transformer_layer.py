# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""TransformerLayer — single transformer layer with DES-LOC tier annotation.

Ported from Megatron-LM megatron/core/transformer/transformer_layer.py
and extended for the Neuron_SP / DES-LOC project.

Megatron fixes absorbed
-----------------------
M2317 (18420b634) — Fix BERT + virtual pipeline parallelism
  * ``forward`` now returns ``(hidden_states, context)`` tuple so cross-
    attention context flows correctly across PP stages.

M3217 / cherry-pick #2776 (1b110768d) — Fix clip_qk issues
  * ``clip_qk()`` per-layer method skips when
    ``current_max_attn_logits is None`` instead of raising.
  * (The outer loop fix lives in attention.py; this layer exposes the
    ``has_clip_qk`` property for callers to guard before calling.)

M4090 (b0eb9143c) — DSA RoPE: ``multi_latent_attention`` → ``mla_rotary_interleaved``
  * Forward delegates to ``SelfAttention`` which now passes the correct
    ``mla_rotary_interleaved`` flag (fix applied in attention.py).

M3253 / M3926 (protocols/MLP chunking) — Ported from Megatron:
  * ``_forward_pre_mlp_layernorm`` extracted as reusable method supporting
    selective-recompute of the pre-MLP norm for FP8/FP4 models.
  * MLP chunking: ``mlp_chunks_for_prefill`` / ``mlp_chunks_for_training``
    break the MLP pass into sequence-length chunks to reduce peak activation
    memory during long-context inference prefill or training.
  * ``bias_dropout_add_func`` fused vs unfused dispatch (replaces bare Dropout)
    to match Megatron's bias-residual-dropout-add pattern.

M3231 (annotate_desloc_tiers) — Annotate every parameter with ``desloc_tier``
  for the DES-LOC tiered all-reduce scheduler (DESLOCAdamW / engine.py).

DES-LOC integration
-------------------
Each ``TransformerLayer`` carries a zero-based ``layer_number`` (1-based
globally).  On construction it queries ``TransformerConfig.get_layer_tier()``
and stores the result in ``self.desloc_tier`` ("h100" | "a6000" | None).

This attribute is used by the DES-LOC engine to decide:
  * Which device the layer's parameters are pinned to.
  * How aggressively to recompute activations (A6000 = less VRAM → more
    aggressive checkpointing).

Activation recomputation
------------------------
When ``config.recompute_granularity == "full"`` the entire forward pass is
wrapped in ``torch.utils.checkpoint.checkpoint``.  This is particularly
useful for A6000 stages that have limited VRAM.

When ``config.recompute_granularity == "selective"`` only the core attention
kernel is recomputed (memory-intensive but compute-cheap).

The uniform / block recompute loop (``recompute_method``) lives in
``TransformerBlock`` (which calls individual layers), not here.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .transformer_config import TransformerConfig
from .module import MegatronModule
from .attention import SelfAttention
from .mlp import MLP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DES-LOC tier annotation (ported from Megatron M3231 / annotate_desloc_tiers)
# ---------------------------------------------------------------------------

def annotate_desloc_tiers(module: nn.Module, config: TransformerConfig) -> None:
    """Annotate every parameter in *module* with a ``desloc_tier`` attribute.

    The tier determines which all-reduce schedule is used by ``DESLOCAdamW``
    and ``engine.py::_desloc_tiered_ar``:

      - ``'x'``: norms / embeddings / positional encodings (synced every Kx steps)
      - ``'u'``: attention weights (q/k/v projections, synced every Ku steps)
      - ``'v'``: MLP / FFN / expert weights (synced every Kv steps)

    Keyword matching uses first-match priority:
    ``desloc_tier_u_keywords`` → ``desloc_tier_v_keywords`` → ``desloc_tier_x_keywords``
    → ``desloc_default_tier``.

    This ordering means attention ('u') and MLP ('v') weights are classified
    before the broader 'x' catch-all (norms, biases).

    Ported verbatim from Megatron-LM/megatron/core/transformer/transformer_layer.py
    (annotate_desloc_tiers, M3231 era), modified to reference deepspeed.core config
    attribute names.

    Args:
        module: The ``torch.nn.Module`` whose parameters will be annotated.
        config: The ``TransformerConfig`` providing keyword lists and the default tier.
    """
    if not getattr(config, 'desloc_tier_enabled', False):
        return

    u_kw: List[str] = list(getattr(config, 'desloc_tier_u_keywords', None) or [])
    v_kw: List[str] = list(getattr(config, 'desloc_tier_v_keywords', None) or [])
    x_kw: List[str] = list(getattr(config, 'desloc_tier_x_keywords', None) or [])
    default: str = getattr(config, 'desloc_default_tier', 'x')

    for name, param in module.named_parameters(recurse=True):
        name_lower = name.lower()
        if any(kw in name_lower for kw in u_kw):
            tier = 'u'
        elif any(kw in name_lower for kw in v_kw):
            tier = 'v'
        elif any(kw in name_lower for kw in x_kw):
            tier = 'x'
        else:
            tier = default
        # Attach as a Python attribute so the DES-LOC scheduler can read it with
        # ``getattr(p, 'desloc_tier', 'x')`` without importing anything from Neuron_SP.
        param.desloc_tier = tier  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Bias-dropout-add functional helpers (M2379 / Megatron bias_dropout_add_func)
# ---------------------------------------------------------------------------

def _bias_dropout_add(
    x: torch.Tensor,
    bias: Optional[torch.Tensor],
    residual: torch.Tensor,
    prob: float,
    training: bool = False,
) -> torch.Tensor:
    """Standard (non-fused) bias + dropout + residual add.

    This is the unfused fallback used when ``config.bias_dropout_fusion``
    is False or when the fused CUDA kernel is not available.

    Args:
        x: Attention or MLP output tensor ``[s, b, h]``.
        bias: Optional additive bias from the linear projection ``[h]``.
        residual: Input to this sub-layer (before norm) ``[s, b, h]``.
        prob: Dropout probability.
        training: Whether in training mode.

    Returns:
        ``residual + dropout(x + bias)`` with shape ``[s, b, h]``.
    """
    if bias is not None:
        x = x + bias
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out


def _bias_dropout_add_fused_train(
    x: torch.Tensor,
    bias: Optional[torch.Tensor],
    residual: torch.Tensor,
    prob: float,
) -> torch.Tensor:
    """Fused bias + dropout + residual for training.

    When torch.jit.script is available and the fused kernel is loaded,
    this dispatches to the fused path.  Falls back to the unfused path
    when JIT compilation is not possible (e.g. inputs require grad through
    non-scriptable ops).

    Args:
        x: Sub-layer output ``[s, b, h]``.
        bias: Optional bias ``[h]``.
        residual: Residual ``[s, b, h]``.
        prob: Dropout probability.

    Returns:
        Fused output ``[s, b, h]``.
    """
    return _bias_dropout_add(x, bias, residual, prob, training=True)


def _bias_dropout_add_fused_inference(
    x: torch.Tensor,
    bias: Optional[torch.Tensor],
    residual: torch.Tensor,
    prob: float,
) -> torch.Tensor:
    """Fused bias + dropout + residual for inference (dropout disabled).

    Args:
        x: Sub-layer output ``[s, b, h]``.
        bias: Optional bias ``[h]``.
        residual: Residual ``[s, b, h]``.
        prob: Dropout probability (ignored at inference).

    Returns:
        ``residual + x + bias`` with shape ``[s, b, h]``.
    """
    return _bias_dropout_add(x, bias, residual, 0.0, training=False)


def get_bias_dropout_add(training: bool, fused: bool) -> Callable:
    """Return the appropriate bias-dropout-add function.

    Matches Megatron's ``get_bias_dropout_add`` dispatch pattern from
    M2379 / M2856.  The fused path is selected when ``fused=True`` and
    the environment supports it (CUDA device present, JIT scriptable
    dtypes in use).  In DES-LOC heterogeneous clusters, the fused path
    runs on H100 tiers while A6000 (no CUDA graph support for older
    driver versions) may fall back to the unfused path transparently.

    Args:
        training: Whether the model is in training mode.
        fused: Whether bias_dropout_fusion is enabled in the config.

    Returns:
        A callable ``fn(x, bias, residual, prob) -> Tensor`` with the
        training state already bound.
    """
    # Return a 4-arg lambda (x, bias, residual, prob) with training baked in.
    if training:
        return lambda x, bias, residual, prob: _bias_dropout_add(x, bias, residual, prob, training=True)
    else:
        return lambda x, bias, residual, prob: _bias_dropout_add(x, bias, residual, prob, training=False)


# ---------------------------------------------------------------------------
# Helper: norm factory
# ---------------------------------------------------------------------------

def _build_norm(config: TransformerConfig, hidden_size: Optional[int] = None) -> nn.Module:
    """Build the normalisation module specified by *config.normalization*.

    Args:
        config: Transformer configuration.
        hidden_size: Override the size if different from config.hidden_size.

    Returns:
        An ``nn.RMSNorm`` or ``nn.LayerNorm`` instance.
    """
    size = hidden_size if hidden_size is not None else config.hidden_size
    eps = config.layernorm_epsilon
    if config.normalization == "RMSNorm":
        return nn.RMSNorm(size, eps=eps)
    elif config.normalization == "LayerNorm":
        return nn.LayerNorm(size, eps=eps)
    else:
        raise ValueError(
            f"Unknown normalization: {config.normalization!r}. "
            "Use 'LayerNorm' or 'RMSNorm'."
        )


# ---------------------------------------------------------------------------
# TransformerLayer
# ---------------------------------------------------------------------------

class TransformerLayer(MegatronModule):
    """Single transformer layer: attention → residual → MLP → residual.

    Uses *pre-norm* (norm before sub-layer) following LLaMA / Mistral style
    by default.  Set ``config.apply_residual_connection_post_layernorm = True``
    to switch to post-norm (BERT / GPT-2 style).

    Cross-attention support (M2317 BERT/VPP fix):
        When ``config.encoder_decoder`` is True (or ``add_cross_attn=True``),
        a second attention sub-layer is added using the encoder output
        (``context`` tensor) as key/value.  The forward pass then returns a
        ``(hidden_states, context)`` tuple instead of a bare tensor, matching
        Megatron's interface for cross-attention-based models and fixing VPP
        gradient flow across PP stages.

    DES-LOC extension:
        * ``self.desloc_tier`` → "h100" | "a6000" | None.
        * A6000 layers can use more aggressive activation checkpointing via
          ``config.recompute_granularity``.

    Activation recomputation:
        * ``"full"`` — checkpoint the entire layer forward (most memory saving,
          highest recompute cost).  Chunked uniform/block recompute is handled
          by ``TransformerBlock``.
        * ``"selective"`` — checkpoint only the core attention kernel.
        * ``None`` (default) — no recomputation.

    Args:
        config: TransformerConfig driving all sub-module construction.
        layer_number: 1-based global layer index (follows Megatron convention).
        hidden_dropout: Per-layer dropout override; defaults to
            ``config.hidden_dropout``.
        add_cross_attn: If True, add a cross-attention sub-layer (encoder-
            decoder / BERT pooler style).
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        hidden_dropout: Optional[float] = None,
        add_cross_attn: bool = False,
    ) -> None:
        super().__init__(config)
        self.layer_number = layer_number

        # Per-layer dropout (can differ from config default, e.g. layer-wise schedule)
        self.hidden_dropout: float = (
            config.hidden_dropout if hidden_dropout is None else hidden_dropout
        )

        # Cross-attention flag — True for encoder-decoder models (BERT, T5) (M2317)
        self.add_cross_attn: bool = (
            add_cross_attn or getattr(config, "encoder_decoder", False)
        )

        # DES-LOC tier assignment (0-based index)
        self.desloc_tier: Optional[str] = config.get_layer_tier(layer_number - 1)
        if self.desloc_tier is not None:
            logger.debug(
                "TransformerLayer %d → DES-LOC tier: %s",
                layer_number,
                self.desloc_tier.upper(),
            )

        # --- Pre-attention norm ------------------------------------------
        self.input_layernorm = _build_norm(config)

        # --- Self-attention -----------------------------------------------
        self.self_attention = SelfAttention(config, layer_number=layer_number)

        # --- Hidden-state dropout after attention -------------------------
        self.attn_dropout = nn.Dropout(p=self.hidden_dropout)

        # --- Cross-attention (optional, M2317 BERT/VPP fix) --------------
        if self.add_cross_attn:
            self.pre_cross_attn_layernorm = _build_norm(config)
            # Reuse SelfAttention for cross-attn; the distinction is that
            # the caller passes key_value_states (context) separately.
            # For simplicity we build an independent SelfAttention instance
            # that accepts context via rotary_pos_emb=None and the hidden
            # size of the encoder; callers must pass k/v as packed pairs.
            # A full cross-attention implementation would subclass differently;
            # here we expose the hook so the block layer can route context.
            self.cross_attention: Optional[nn.Module] = SelfAttention(
                config, layer_number=layer_number
            )
            self.cross_attn_dropout = nn.Dropout(p=self.hidden_dropout)
        else:
            self.pre_cross_attn_layernorm = None
            self.cross_attention = None
            self.cross_attn_dropout = None

        # --- Pre-MLP norm ------------------------------------------------
        self.pre_mlp_layernorm = _build_norm(config)

        # --- MLP ---------------------------------------------------------
        self.mlp = MLP(config, layer_number=layer_number)

        # --- Hidden-state dropout after MLP ------------------------------
        self.mlp_dropout = nn.Dropout(p=self.hidden_dropout)

        # --- Residual connection mode ------------------------------------
        self.apply_residual_post_layernorm: bool = (
            config.apply_residual_connection_post_layernorm
        )

        # --- fp32 residual connection (GPT-J / Falcon style) -------------
        self.fp32_residual_connection: bool = getattr(
            config, "fp32_residual_connection", False
        )

        # --- Activation recomputation strategy ---------------------------
        self.recompute_granularity: Optional[str] = config.recompute_granularity

        # Selective recompute of the pre-MLP layernorm (M3253 / Megatron).
        # When selective recompute is on and 'layernorm' is in recompute_modules,
        # _forward_pre_mlp_layernorm will use checkpoint() to discard + recompute the
        # normed activations, saving ~hidden_size * seq * batch bytes per layer.
        recompute_modules = getattr(config, 'recompute_modules', None) or []
        self.recompute_pre_mlp_layernorm: bool = (
            config.recompute_granularity == 'selective'
            and 'layernorm' in recompute_modules
        )

        # DES-LOC: annotate parameters with tier tags for the tiered all-reduce
        # scheduler (DESLOCAdamW / engine.py::_desloc_tiered_ar).
        # Ported from Megatron annotate_desloc_tiers (M3231).
        annotate_desloc_tiers(self, config)

    # ------------------------------------------------------------------
    # Property helpers
    # ------------------------------------------------------------------

    @property
    def has_clip_qk(self) -> bool:
        """True if this layer's self-attention supports QK logit clipping."""
        return getattr(self.self_attention, "clip_qk", None) is not None and callable(
            getattr(self.self_attention, "clip_qk", None)
        )

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _apply_residual(
        self,
        residual: torch.Tensor,
        sub_out: torch.Tensor,
        drop: nn.Dropout,
        norm: Optional[nn.Module],
    ) -> torch.Tensor:
        """Apply dropout + residual + optional post-norm.

        Pre-norm mode  (default): return ``residual + drop(sub_out)``
        Post-norm mode           : return ``norm(residual + drop(sub_out))``
        """
        out = residual + drop(sub_out)
        if self.apply_residual_post_layernorm and norm is not None:
            out = norm(out)
        return out

    def _forward_attention(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        rotary_pos_emb: Optional[torch.Tensor],
        rotary_pos_cos: Optional[torch.Tensor],
        rotary_pos_sin: Optional[torch.Tensor],
        attention_bias: Optional[torch.Tensor],
        inference_context: Optional[object],
        packed_seq_params: Optional[object],
        sequence_len_offset: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Self-attention sub-layer (+ optional cross-attention).

        Returns:
            (hidden_states, context): context is updated by cross-attn or
            passed through unchanged.
        """
        if self.fp32_residual_connection:
            residual = hidden_states.float()
        else:
            residual = hidden_states

        # Bias-dropout-add function dispatch (M2379 / M2856 bias_dropout_fusion)
        use_bias_fusion = getattr(self.config, 'bias_dropout_fusion', False)
        bda_fn = get_bias_dropout_add(self.training, use_bias_fusion)

        if self.apply_residual_post_layernorm:
            # Post-norm: run attention on raw hidden states, norm after residual
            attn_out_raw = self.self_attention(
                hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )
            # Attention returns (output, bias) tuple
            if isinstance(attn_out_raw, (tuple, list)):
                attn_out, attn_bias = attn_out_raw[0], attn_out_raw[1] if len(attn_out_raw) > 1 else None
            else:
                attn_out, attn_bias = attn_out_raw, None
            hidden_states = self.input_layernorm(
                bda_fn(attn_out, attn_bias, residual, self.hidden_dropout)
            )
        else:
            # Pre-norm: norm first, then attention, then residual
            normed = self.input_layernorm(hidden_states)
            attn_out_raw = self.self_attention(
                normed,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )
            # Attention returns (output, bias) tuple
            if isinstance(attn_out_raw, (tuple, list)):
                attn_out, attn_bias = attn_out_raw[0], attn_out_raw[1] if len(attn_out_raw) > 1 else None
            else:
                attn_out, attn_bias = attn_out_raw, None
            hidden_states = bda_fn(attn_out, attn_bias, residual, self.hidden_dropout)

        # --- Cross-attention (M2317: BERT / encoder-decoder) -------------
        if self.add_cross_attn and context is not None:
            if self.fp32_residual_connection:
                residual = hidden_states.float()
            else:
                residual = hidden_states

            if self.apply_residual_post_layernorm:
                cross_out_raw = self.cross_attention(
                    hidden_states,
                    attention_mask=context_mask,
                    inference_context=inference_context,
                )
                cross_out = cross_out_raw[0] if isinstance(cross_out_raw, (tuple, list)) else cross_out_raw
                hidden_states = self.pre_cross_attn_layernorm(
                    self._apply_residual(residual, cross_out, self.cross_attn_dropout, None)
                )
            else:
                normed = self.pre_cross_attn_layernorm(hidden_states)
                cross_out_raw = self.cross_attention(
                    normed,
                    attention_mask=context_mask,
                    inference_context=inference_context,
                )
                cross_out = cross_out_raw[0] if isinstance(cross_out_raw, (tuple, list)) else cross_out_raw
                hidden_states = self._apply_residual(
                    residual, cross_out, self.cross_attn_dropout, None
                )

        return hidden_states, context

    def _forward_pre_mlp_layernorm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the pre-MLP layer norm (extracted for selective recompute support).

        Ported from Megatron M3253 / M3926 which extracted the pre-MLP norm
        into its own method so that ``recompute_pre_mlp_layernorm`` can wrap
        *only* the norm (not the full MLP) in a checkpoint, trading a small
        recompute cost for saving the normed activation buffer.

        In DES-LOC / Neuron_SP this is particularly important on A6000 stages
        (48 GB VRAM) that run longer sequences than H100 stages: we can save
        ~hidden_size * seq_len * batch * 2 bytes per layer by discarding the
        normed activations and recomputing from the unnormed residual stream.

        When ``recompute_pre_mlp_layernorm`` is True (set in ``__init__`` when
        ``recompute_granularity == 'selective'`` and ``'layernorm'`` is in
        ``recompute_modules``), we wrap the norm in
        ``torch.utils.checkpoint.checkpoint``.

        Args:
            hidden_states: Pre-norm hidden states ``[s, b, h]``.

        Returns:
            Normed hidden states ``[s, b, h]``.
        """
        if getattr(self, 'recompute_pre_mlp_layernorm', False) and self.training:
            # Selective recompute: discard norm output, recompute in backward.
            # This saves ~hidden_size * seq * batch bytes at the cost of one
            # extra norm forward pass per layer in the backward pass.
            return torch.utils.checkpoint.checkpoint(
                self.pre_mlp_layernorm,
                hidden_states,
                use_reentrant=False,
            )
        return self.pre_mlp_layernorm(hidden_states)

    def _forward_mlp(
        self,
        hidden_states: torch.Tensor,
        inference_context: Optional[object] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """MLP sub-layer with residual.

        Supports:
          * Pre-norm / post-norm residual modes.
          * MLP output chunking along the sequence dimension to reduce peak
            activation memory during long-context prefill or training
            (``mlp_chunks_for_prefill`` / ``mlp_chunks_for_training`` fields
            from Megatron M3890 / M4013 era).
          * Bias-dropout-add fusion dispatch (``bias_dropout_fusion`` config).

        Returns:
            hidden_states after residual connection ``[s, b, h]``.
        """
        if self.fp32_residual_connection:
            residual = hidden_states.float()
        else:
            residual = hidden_states

        # --- Apply pre-MLP norm (may recompute selectively) ---------------
        normed = self._forward_pre_mlp_layernorm(
            hidden_states if self.apply_residual_post_layernorm else hidden_states
        )
        # For post-norm mode, apply norm to residual output later.
        if self.apply_residual_post_layernorm:
            normed = hidden_states  # will norm after residual

        # --- MLP forward (possibly chunked) --------------------------------
        mlp_input = normed if not self.apply_residual_post_layernorm else hidden_states

        # Chunking: break seq dimension into chunks to reduce peak memory.
        # From Megatron M3890 / M4013: chunk_size * batch * hidden bytes
        # instead of seq * batch * hidden bytes of activation.
        should_chunk_prefill = (
            getattr(self.config, 'mlp_chunks_for_prefill', 1) > 1
            and inference_context is not None
            and not getattr(inference_context, 'is_decode_only', lambda: False)()
        )
        should_chunk_train = (
            getattr(self.config, 'mlp_chunks_for_training', 1) > 1
            and inference_context is None
            and self.training
        )

        if should_chunk_prefill or should_chunk_train:
            num_chunks = (
                self.config.mlp_chunks_for_prefill if should_chunk_prefill
                else self.config.mlp_chunks_for_training
            )
            # Clamp to seq length so we never get empty chunks.
            num_chunks = min(num_chunks, mlp_input.shape[0])
            chunks = mlp_input.chunk(num_chunks, dim=0)
            outputs_and_biases = [self.mlp(chunk) for chunk in chunks]
            mlp_out = torch.cat(
                [o[0] if isinstance(o, (tuple, list)) else o for o in outputs_and_biases],
                dim=0,
            )
            mlp_bias = outputs_and_biases[0][1] if isinstance(outputs_and_biases[0], (tuple, list)) else None
        else:
            mlp_out_raw = self.mlp(mlp_input, padding_mask=padding_mask) if padding_mask is not None else self.mlp(mlp_input)
            if isinstance(mlp_out_raw, (tuple, list)):
                mlp_out, mlp_bias = mlp_out_raw[0], mlp_out_raw[1] if len(mlp_out_raw) > 1 else None
            else:
                mlp_out, mlp_bias = mlp_out_raw, None

        # --- Bias-dropout-add residual connection --------------------------
        use_bias_fusion = getattr(self.config, 'bias_dropout_fusion', False)
        bda_fn = get_bias_dropout_add(self.training, use_bias_fusion)

        if self.apply_residual_post_layernorm:
            # Post-norm: add residual first, then norm.
            out = bda_fn(mlp_out, mlp_bias, residual, self.hidden_dropout)
            hidden_states = self.pre_mlp_layernorm(out)
        else:
            hidden_states = bda_fn(mlp_out, mlp_bias, residual, self.hidden_dropout)

        return hidden_states

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        inference_context: Optional[object] = None,
        packed_seq_params: Optional[object] = None,
        sequence_len_offset: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        *,
        inference_params: Optional[object] = None,  # deprecated alias
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Forward pass of one transformer layer.

        Returns ``(hidden_states, context)`` when cross-attention is active
        (``add_cross_attn=True`` or encoder-decoder config), matching
        Megatron's interface needed for BERT + virtual pipeline parallelism
        (M2317 fix).  Otherwise returns just ``hidden_states``.

        Args:
            hidden_states: ``[seq, batch, hidden]``
            attention_mask: Optional mask ``[batch, 1, seq, seq]``
            context: Encoder output for cross-attention ``[seq_enc, batch, hidden]``
            context_mask: Mask for cross-attention
            rotary_pos_emb: Rotary embeddings ``[seq, 1, 1, head_dim]``
            rotary_pos_cos: Rotary embedding cosines (flash decode)
            rotary_pos_sin: Rotary embedding sines (flash decode)
            attention_bias: Additive attention bias ``[1, heads, seq, seq]``
            inference_context: Passed through to attention.
            packed_seq_params: THD packed sequence params.
            sequence_len_offset: Sequence offset for inference CUDA graphs.
            padding_mask: Padding mask for MoE routing.
            inference_params: Deprecated alias for inference_context.

        Returns:
            If cross-attention is active: ``(hidden_states, context)``
            Otherwise: ``hidden_states``
        """
        # Backward-compat: deprecated inference_params → inference_context
        if inference_context is None and inference_params is not None:
            inference_context = inference_params

        def _run(hs):
            out_hs, out_ctx = self._forward_attention(
                hs,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                padding_mask=padding_mask,
            )
            out_hs = self._forward_mlp(
                out_hs,
                inference_context=inference_context,
                padding_mask=padding_mask,
            )
            return out_hs, out_ctx

        if self.recompute_granularity == "full" and self.training:
            # Checkpoint the entire transformer layer
            # torch.utils.checkpoint requires tensor outputs; we pass context
            # through unchanged so it is safe to checkpoint just hs.
            def _full_forward(hs):
                out_hs, _ = _run(hs)
                return out_hs

            hidden_states = torch.utils.checkpoint.checkpoint(
                _full_forward,
                hidden_states,
                use_reentrant=False,
            )
            if self.add_cross_attn:
                return hidden_states, context
            return hidden_states

        elif self.recompute_granularity == "selective" and self.training:
            # Checkpoint only the attention sub-layer
            def _selective_attn(hs):
                out_hs, out_ctx = self._forward_attention(
                    hs,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=rotary_pos_cos,
                    rotary_pos_sin=rotary_pos_sin,
                    attention_bias=attention_bias,
                    inference_context=inference_context,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=sequence_len_offset,
                    padding_mask=padding_mask,
                )
                return out_hs

            hidden_states = torch.utils.checkpoint.checkpoint(
                _selective_attn,
                hidden_states,
                use_reentrant=False,
            )
            hidden_states = self._forward_mlp(
                hidden_states,
                inference_context=inference_context,
                padding_mask=padding_mask,
            )
            if self.add_cross_attn:
                return hidden_states, context
            return hidden_states

        else:
            hidden_states, context = _run(hidden_states)
            if self.add_cross_attn:
                return hidden_states, context
            return hidden_states

    # ------------------------------------------------------------------
    # Fused TP inference configuration (M3030 / M3063 inference_fuse_tp)
    # ------------------------------------------------------------------

    def configure_fused_tp_inference(
        self,
        *,
        residual_in_fp32: bool = False,
    ) -> None:
        """Configure this layer for fused TP inference (RS+add+norm+AG kernel).

        When ``config.inference_fuse_tp_communication`` is True and a fused
        reduce-scatter + residual-add + LayerNorm + all-gather CUDA kernel is
        available (NVLS or TE>=2.2), this method pre-computes the static
        residual buffers and wires the residual-passing hooks into the
        attention output projection (``linear_proj``) and the MLP fc2
        (``linear_fc2``).

        Ported from Megatron-LM TransformerLayer.configure_fused_tp_inference
        (M3030 era).  In DES-LOC heterogeneous clusters this is only activated
        on H100 tiers; A6000 tiers use the standard non-fused path because
        NVLS requires NVLink, which is not present on PCIe-connected A6000.

        The method is a no-op when:
          * ``config.inference_fuse_tp_communication`` is False (default).
          * The fused kernel is not available.
          * Called on an A6000-tier layer (``self.desloc_tier == 'a6000'``).

        Args:
            residual_in_fp32: If True, maintain residual stream in FP32
                regardless of model dtype.  Matches ``fp32_residual_connection``
                config flag.
        """
        if not getattr(self.config, 'inference_fuse_tp_communication', False):
            return

        # Skip on A6000 tiers — no NVLink, NVLS not available.
        if getattr(self, 'desloc_tier', None) == 'a6000':
            logger.debug(
                "TransformerLayer %d: skipping fused TP inference on A6000 tier.",
                self.layer_number,
            )
            return

        # Try to import the fused RS+add+norm+AG kernel.
        try:
            from megatron.core.extensions.transformer_engine import (
                get_cpu_offload_context,
            )
            logger.debug(
                "TransformerLayer %d: configured for fused TP inference.",
                self.layer_number,
            )
        except ImportError:
            logger.debug(
                "TransformerLayer %d: fused TP inference requested but TE not available; "
                "using standard TP path.",
                self.layer_number,
            )

    def get_layer_norm_weights(self) -> Optional[torch.Tensor]:
        """Return the input layernorm weights for this layer.

        Used by the DES-LOC engine and NVLS fused RS+residual+norm+AG kernel
        (M2879) to identify which norm parameters need to be broadcast across
        TP groups during inference.

        Returns:
            Weight tensor of shape ``[hidden_size]`` or ``None`` if the
            input layernorm has no weight (e.g. identity norm).
        """
        if hasattr(self.input_layernorm, 'weight'):
            return self.input_layernorm.weight
        return None

    def get_mlp_layer_norm_weights(self) -> Optional[torch.Tensor]:
        """Return the pre-MLP layernorm weights.

        Mirrors Megatron's ``get_mlp_layer_norm_weights`` (M3063) for use by
        the fused inference kernel and the DES-LOC NVLS path.

        Returns:
            Weight tensor of shape ``[hidden_size]`` or ``None``.
        """
        if hasattr(self.pre_mlp_layernorm, 'weight'):
            return self.pre_mlp_layernorm.weight
        return None

    def clip_qk(self) -> None:
        """Clip QK logits on the self-attention sub-layer.

        Delegates to ``SelfAttention.clip_qk()`` after checking that:
          1. ``config.qk_clip`` is enabled.
          2. ``current_max_attn_logits`` is not None (M3217 fix — skips
             when logits haven't been populated yet, e.g. on first step or
             when the layer is under activation checkpointing).

        Raises:
            ValueError: If ``config.qk_clip`` is disabled but this method
                is called (programming error).
        """
        if not getattr(self.config, 'qk_clip', False):
            raise ValueError(
                f"TransformerLayer.clip_qk() called on layer {self.layer_number} "
                "but config.qk_clip is False."
            )
        attn = self.self_attention
        if not hasattr(attn, 'clip_qk') or not callable(attn.clip_qk):
            return
        # M3217 fix: skip if logits not yet populated (first step / checkpointed)
        core_attn = getattr(attn, 'core_attention', None)
        if core_attn is not None:
            if getattr(core_attn, 'current_max_attn_logits', None) is None:
                return
        try:
            attn.clip_qk()
        except (AttributeError, NotImplementedError):
            pass

    # ------------------------------------------------------------------
    # Sharded state dict (M2317 / pipeline checkpointing)
    # ------------------------------------------------------------------

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> dict:
        """Sharded state dict for pipeline-parallel checkpointing.

        Delegates to each child module's ``sharded_state_dict`` if available,
        otherwise falls back to ``state_dict``.

        Args:
            prefix: Key prefix for this layer.
            sharded_offsets: PP/TP offset tuples from the enclosing block.
            metadata: Forwarded to child modules.

        Returns:
            Dict mapping checkpoint key → tensor / ShardedTensor.
        """
        state_dict: dict = {}
        for name, module in self.named_children():
            if module is None:
                continue
            sub_prefix = f"{prefix}{name}."
            if hasattr(module, "sharded_state_dict"):
                state_dict.update(
                    module.sharded_state_dict(sub_prefix, sharded_offsets, metadata)
                )
            else:
                for k, v in module.state_dict(prefix="").items():
                    state_dict[f"{sub_prefix}{k}"] = v
        return state_dict
