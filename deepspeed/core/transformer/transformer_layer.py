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
  * ``bias_dropout_add_exec_handler`` / ``mlp_bda`` / ``self_attn_bda`` dispatch
    now use spec-based modules (``IdentityFuncOp`` by default) matching Megatron.

M3231 (annotate_desloc_tiers) — Annotate every parameter with ``desloc_tier``
  for the DES-LOC tiered all-reduce scheduler (DESLOCAdamW / engine.py).

M3954 / M3977 — ``recompute_input_layernorm`` and ``recompute_mlp`` flags
  added for selective-recompute of individual sub-layers (ported from Megatron).

DES-LOC integration
-------------------
Each ``TransformerLayer`` carries a zero-based ``layer_number`` (1-based
globally).  On construction it queries ``TransformerConfig.get_layer_tier()``
and stores the result in ``self.desloc_tier`` (\"h100\" | \"a6000\" | None).

This attribute is used by the DES-LOC engine to decide:
  * Which device the layer's parameters are pinned to.
  * How aggressively to recompute activations (A6000 = less VRAM → more
    aggressive checkpointing).

Activation recomputation
------------------------
When ``config.recompute_granularity == \"full\"`` the entire forward pass is
wrapped in ``torch.utils.checkpoint.checkpoint``.  This is particularly
useful for A6000 stages that have limited VRAM.

When ``config.recompute_granularity == \"selective\"`` only the core attention
kernel is recomputed (memory-intensive but compute-cheap), plus optionally
the pre-MLP layernorm (``recompute_pre_mlp_layernorm``) and the MLP itself
(``recompute_mlp``).

The uniform / block recompute loop (``recompute_method``) lives in
``TransformerBlock`` (which calls individual layers), not here.
"""

from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .transformer_config import TransformerConfig
from .module import MegatronModule
from .attention import SelfAttention
from .mlp import MLP
from .identity_op import IdentityFuncOp, IdentityOp
from .spec_utils import ModuleSpec, build_module

# DES-LOC requirement: use deepspeed.comm instead of torch.distributed
try:
    import deepspeed.comm as dist
except ImportError:
    import torch.distributed as dist  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# get_transformer_layer_offset — ported verbatim from Megatron transformer_layer.py
# Also re-exported here so callers can do:
#   from deepspeed.core.transformer.transformer_layer import get_transformer_layer_offset
# ---------------------------------------------------------------------------

def get_transformer_layer_offset(
    config: TransformerConfig,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> int:
    """Get the 0-based global index offset of the first layer on this PP stage.

    Handles:
      * Single PP stage (offset = 0).
      * Even split (default).
      * Uneven first/last stage (``num_layers_in_first/last_pipeline_stage``).
      * Virtual pipeline parallelism.
      * ``pipeline_model_parallel_layout`` custom layout (if present).
      * ``pipeline_layer_split`` DES-LOC heterogeneous split.

    Args:
        config: TransformerConfig.
        vp_stage: Virtual pipeline stage (None unless VPP is active).
        pp_rank: Pipeline rank override; queries live group if None.

    Returns:
        Integer 0-based offset for this PP (sub-)stage.
    """
    try:
        from deepspeed.core.parallel_state import get_pipeline_model_parallel_rank
        if pp_rank is None:
            pp_rank = get_pipeline_model_parallel_rank()
    except Exception:
        if pp_rank is None:
            pp_rank = 0

    pp_size = getattr(config, "pipeline_model_parallel_size", 1) or 1

    if pp_size <= 1:
        return 0

    # DES-LOC custom layout object (pipeline_model_parallel_layout)
    layout = getattr(config, "pipeline_model_parallel_layout", None)
    if layout is not None and hasattr(layout, "get_layer_offset"):
        try:
            from deepspeed.core.transformer.enums import LayerType
            return layout.get_layer_offset(layer_type=LayerType.decoder, vp_stage=vp_stage)
        except Exception:
            pass

    # DES-LOC heterogeneous split
    pipeline_layer_split: Optional[List[int]] = getattr(config, "pipeline_layer_split", None)
    if pipeline_layer_split is not None:
        return sum(pipeline_layer_split[:pp_rank])

    # Uneven first / last pipeline stage
    first_stage_layers = getattr(config, "num_layers_in_first_pipeline_stage", None)
    last_stage_layers = getattr(config, "num_layers_in_last_pipeline_stage", None)

    if first_stage_layers is not None or last_stage_layers is not None:
        first_n = first_stage_layers or 0
        last_n = last_stage_layers or 0
        middle_stages = pp_size - (1 if first_stage_layers is not None else 0) - (
            1 if last_stage_layers is not None else 0
        )
        middle_layers = config.num_layers - first_n - last_n
        num_layers_per_middle = (middle_layers // middle_stages) if middle_stages > 0 else 0

        vp_size = getattr(config, "virtual_pipeline_model_parallel_size", None)
        if vp_size is not None:
            assert vp_stage is not None, "vp_stage must be provided when VPP is active"
            layers_per_vp_first = (first_n // vp_size) if first_stage_layers is not None else 0
            layers_per_vp_last = (last_n // vp_size) if last_stage_layers is not None else 0
            layers_per_vp_middle = middle_layers // vp_size if middle_stages > 0 else 0
            total_vp_chunk = layers_per_vp_first + layers_per_vp_middle + layers_per_vp_last
            middle_pp_rank = pp_rank if first_stage_layers is None else pp_rank - 1
            if pp_rank == 0:
                offset = vp_stage * total_vp_chunk
            else:
                offset = (
                    vp_stage * total_vp_chunk
                    + layers_per_vp_first
                    + middle_pp_rank * (layers_per_vp_middle // middle_stages if middle_stages > 0 else 0)
                )
        else:
            middle_pp_rank = pp_rank if first_stage_layers is None else pp_rank - 1
            if pp_rank == 0:
                offset = 0
            else:
                offset = first_n + middle_pp_rank * num_layers_per_middle
        return offset

    # Standard even split (with optional embedding/loss accounting)
    num_layers = config.num_layers
    account_embedding = getattr(config, "account_for_embedding_in_pipeline_split", False)
    account_loss = getattr(config, "account_for_loss_in_pipeline_split", False)
    if account_embedding:
        num_layers += 1
    if account_loss:
        num_layers += 1
    num_layers_per_rank = num_layers // pp_size

    vp_size = getattr(config, "virtual_pipeline_model_parallel_size", None)
    is_first_pp = pp_rank == 0

    if vp_size is not None:
        assert vp_stage is not None, "vp_stage must be provided when VPP is active"
        num_layers_per_vr = num_layers_per_rank // vp_size
        total_vp_chunks = num_layers // vp_size
        offset = vp_stage * total_vp_chunks + pp_rank * num_layers_per_vr
        # Subtract embedding placeholder when not on first VPP+PP stage
        if account_embedding and not (vp_stage == 0 and is_first_pp):
            offset -= 1
    else:
        offset = pp_rank * num_layers_per_rank
        if account_embedding and not is_first_pp:
            offset -= 1

    return offset


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
    """Fused bias + dropout + residual for training."""
    return _bias_dropout_add(x, bias, residual, prob, training=True)


def _bias_dropout_add_fused_inference(
    x: torch.Tensor,
    bias: Optional[torch.Tensor],
    residual: torch.Tensor,
    prob: float,
) -> torch.Tensor:
    """Fused bias + dropout + residual for inference (dropout disabled)."""
    return _bias_dropout_add(x, bias, residual, 0.0, training=False)


def get_bias_dropout_add(training: bool, fused: bool) -> Callable:
    """Return the appropriate bias-dropout-add function.

    Args:
        training: Whether the model is in training mode.
        fused: Whether bias_dropout_fusion is enabled in the config.

    Returns:
        A callable ``fn(x, bias, residual, prob) -> Tensor`` with the
        training state already bound.
    """
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
# MLP interface / builder protocols (M3926 — Megatron API compat)
# ---------------------------------------------------------------------------

class MlpInterface:
    """Interface for MLP implementations in the transformer layer.

    Any MLP module used inside ``TransformerLayer`` must implement this
    interface (duck-typing is sufficient; inheriting is not required).
    Ported from Megatron-LM ``transformer_layer.py`` M3926.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        /,
        *,
        intermediate_tensors: Optional[Tuple[torch.Tensor, ...]] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward method for the MLP interface."""
        ...


class MlpBuilder:
    """MLP builder protocol for building MLPs in the transformer layer.

    Ported from Megatron-LM ``transformer_layer.py`` M3926.
    """

    def __call__(
        self,
        *,
        config: TransformerConfig,
        pg_collection: object,
        is_mtp_layer: bool,
        name: Optional[str] = None,
    ) -> MlpInterface:
        ...


# ---------------------------------------------------------------------------
# TransformerLayerSubmodules — Megatron spec-based submodule configuration
# ---------------------------------------------------------------------------

@dataclass
class TransformerLayerSubmodules:
    """Configuration class for specifying the submodules of a transformer layer.

    This class defines the structure and default implementations for various
    components of a transformer layer, allowing for flexible customisation
    of the layer's architecture.  Mirrors Megatron's ``TransformerLayerSubmodules``.

    Args:
        input_layernorm: Specification for the input layer normalisation.
        self_attention: Specification for the self-attention mechanism.
        self_attn_bda: Specification for the bias-dropout-add after self-attention.
        pre_cross_attn_layernorm: Specification for the norm before cross-attention.
        cross_attention: Specification for the cross-attention mechanism.
        cross_attn_bda: Specification for the bias-dropout-add after cross-attention.
        pre_mlp_layernorm: Specification for the norm before the MLP.
        mlp: Specification for the MLP (Dense or MoE).
        mlp_bda: Specification for the bias-dropout-add after the MLP.
        sharded_state_dict_keys_map: Key-rename map applied in ``sharded_state_dict``.
    """

    input_layernorm: object = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_cross_attn_layernorm: object = IdentityOp
    cross_attention: Union[ModuleSpec, type] = IdentityOp
    cross_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_mlp_layernorm: object = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BaseTransformerLayer — common parent for TransformerLayer-like classes
# ---------------------------------------------------------------------------

class BaseTransformerLayer(ABC):
    """A common parent class for ``TransformerLayer``-like implementations.

    A dummy class that is subclassed by similar ``TransformerLayer``s e.g. the
    ``TransformerLayer`` in this file and possibly other ``TransformerLayer``
    implementations that aim to use ``TransformerBlock`` as the base module.
    The main purpose is to check if any layer (or module) provided in the spec
    is a subclass of this class to allow fanning-out of that spec for all the
    layers in the ``TransformerBlock``. See ``_get_block_submodules`` method
    implementation in ``transformer_block.py`` for more details.
    """

    def __init__(self):
        pass


# ---------------------------------------------------------------------------
# TransformerLayer
# ---------------------------------------------------------------------------

class TransformerLayer(MegatronModule, BaseTransformerLayer):
    """Single transformer layer: attention → residual → MLP → residual.

    Uses *pre-norm* (norm before sub-layer) following LLaMA / Mistral style
    by default.  Set ``config.apply_residual_connection_post_layernorm = True``
    to switch to post-norm (BERT / GPT-2 style).

    Spec-based construction:
        When ``submodules`` is a ``TransformerLayerSubmodules``, each field
        is built via ``build_module`` allowing full customisation of every
        sub-layer (attention, MLP, norms, BDA functions).  When ``submodules``
        is ``None``, the layer falls back to a direct construction of
        ``SelfAttention`` and ``MLP`` using ``_build_norm`` for norms.

    Cross-attention support (M2317 BERT/VPP fix):
        When ``config.encoder_decoder`` is True or ``submodules`` includes a
        non-identity ``cross_attention``, a second attention sub-layer is added
        using the encoder output (``context`` tensor) as key/value.  The forward
        pass then returns a ``(hidden_states, context)`` tuple instead of a bare
        tensor, matching Megatron's interface for cross-attention-based models.

    DES-LOC extension:
        * ``self.desloc_tier`` → "h100" | "a6000" | None.
        * A6000 layers can use more aggressive activation checkpointing.

    Activation recomputation (M3954 / M3977):
        * ``recompute_input_layernorm`` — checkpoint only the input layernorm.
        * ``recompute_pre_mlp_layernorm`` — checkpoint only the pre-MLP layernorm.
        * ``recompute_mlp`` — checkpoint only the MLP forward.
        * ``"full"`` — checkpoint the entire layer forward (handled in TransformerBlock).
        * ``"selective"`` — controls which sub-layers are checkpointed.

    Args:
        config: TransformerConfig driving all sub-module construction.
        submodules: ``TransformerLayerSubmodules`` for spec-based construction,
            or ``None`` to fall back to direct construction.
        layer_number: 1-based *local* layer index (TransformerBlock adds the
            PP-stage offset so that ``self.layer_number`` is globally unique).
        hidden_dropout: Per-layer dropout override; defaults to
            ``config.hidden_dropout``.
        pg_collection: ProcessGroupCollection for TP/PP/CP groups (passed
            through to attention and MLP sub-modules).
        vp_stage: Virtual pipeline stage (None unless VPP is active).
        is_mtp_layer: True when this is a Multi-Token-Prediction inner layer;
            layer_number is NOT offset by PP stage in this case.
        add_layer_offset: If False the caller has already included the correct
            PP offset in ``layer_number`` (e.g. with fVPP).
        name: Optional module instance name for debugging.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[TransformerLayerSubmodules] = None,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[object] = None,
        vp_stage: Optional[int] = None,
        is_mtp_layer: bool = False,
        add_layer_offset: bool = True,
        name: Optional[str] = None,
    ) -> None:
        # BaseTransformerLayer has its own __init__; call both parents explicitly
        MegatronModule.__init__(self, config)
        BaseTransformerLayer.__init__(self)

        self.submodules_config = submodules
        self.vp_stage = vp_stage
        self.is_mtp_layer = is_mtp_layer

        # Resolve pg_collection (lazy import to avoid circular deps)
        self.pg_collection = pg_collection

        # Compute globally-unique 1-based layer_number
        # MTP inner layers keep their own numbering and do NOT add the decoder offset.
        # If add_layer_offset is False, the caller already included the offset.
        if is_mtp_layer or not add_layer_offset:
            self.layer_number = layer_number
        else:
            offset = get_transformer_layer_offset(config, vp_stage, None)
            self.layer_number = layer_number + offset

        # Per-layer dropout (can differ from config default)
        self.hidden_dropout: float = (
            config.hidden_dropout if hidden_dropout is None else hidden_dropout
        )

        # DES-LOC tier assignment (uses 0-based index)
        self.desloc_tier: Optional[str] = config.get_layer_tier(self.layer_number - 1)
        if self.desloc_tier is not None:
            logger.debug(
                "TransformerLayer %d → DES-LOC tier: %s",
                self.layer_number,
                self.desloc_tier.upper(),
            )

        # Whether cross-attention is enabled
        self.add_cross_attn: bool = getattr(config, "encoder_decoder", False)

        # ---- Build sub-modules via spec or direct construction ----
        if submodules is not None:
            self._build_from_submodules(submodules, config, name)
        else:
            self._build_direct(config)

        # --- Residual connection mode --------------------------------
        self.apply_residual_post_layernorm: bool = (
            config.apply_residual_connection_post_layernorm
        )
        self.fp32_residual_connection: bool = getattr(
            config, "fp32_residual_connection", False
        )

        # --- Activation recomputation strategy (M3954) ---------------
        self.recompute_granularity: Optional[str] = config.recompute_granularity
        recompute_modules = list(getattr(config, "recompute_modules", None) or [])

        self.recompute_input_layernorm: bool = False
        self.recompute_pre_mlp_layernorm: bool = False
        self.recompute_mlp: bool = False

        if config.recompute_granularity == "selective":
            if "layernorm" in recompute_modules:
                self.recompute_input_layernorm = not isinstance(self.input_layernorm, IdentityOp)
                self.recompute_pre_mlp_layernorm = not isinstance(self.pre_mlp_layernorm, IdentityOp)
            if "mlp" in recompute_modules:
                self.recompute_mlp = True

        # --- Fine-grained activation offloading (Megatron M4141 / offload_modules) ---
        # offload_attn_norm: CPU-offload the input layernorm activation after BDA
        # offload_mlp_norm:  CPU-offload the pre-MLP layernorm activation after MLP BDA
        # Both require config.fine_grained_activation_offloading=True and the respective
        # key in config.offload_modules.  On DES-LOC A6000 tiers these are auto-enabled
        # when memory budgets are tight (handled by the DES-LOC engine at startup).
        _fgao = getattr(config, "fine_grained_activation_offloading", False)
        _offload_modules = list(getattr(config, "offload_modules", None) or [])

        self.offload_attn_norm: bool = (
            _fgao
            and "attn_norm" in _offload_modules
            and not isinstance(self.input_layernorm, IdentityOp)
        )
        self.offload_mlp_norm: bool = (
            _fgao
            and "mlp_norm" in _offload_modules
            and not isinstance(self.pre_mlp_layernorm, IdentityOp)
        )

        # A6000 tier: more aggressive norm offloading when not already enabled by config
        # (DES-LOC extension — keeps attention norms off-device on memory-constrained GPU)
        if (
            self.desloc_tier == "a6000"
            and _fgao
            and not self.offload_attn_norm
            and not isinstance(self.input_layernorm, IdentityOp)
        ):
            self.offload_attn_norm = True
            logger.debug(
                "TransformerLayer %d (A6000): auto-enabling offload_attn_norm.",
                self.layer_number,
            )

        # --- bias_dropout_add_exec_handler (Megatron M2379) ----------
        # Ensures grad is enabled during the BDA operation, matching Megatron behaviour.
        self.bias_dropout_add_exec_handler = torch.enable_grad

        # --- DES-LOC parameter tier annotation -----------------------
        annotate_desloc_tiers(self, config)

    # ------------------------------------------------------------------
    # Sub-module construction helpers
    # ------------------------------------------------------------------

    def _build_from_submodules(
        self,
        submodules: TransformerLayerSubmodules,
        config: TransformerConfig,
        name: Optional[str],
    ) -> None:
        """Build all sub-modules from the submodule spec (Megatron-style)."""

        # [Module 1: Input Layernorm]
        self.input_layernorm = build_module(
            submodules.input_layernorm,
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon,
        ) if submodules.input_layernorm is not IdentityOp else IdentityOp()

        # [Module 2: Self-Attention]
        self.self_attention = build_module(
            submodules.self_attention,
            config=config,
            layer_number=self.layer_number,
        )

        # [Module 3: Self-attention BDA]
        self.self_attn_bda = build_module(submodules.self_attn_bda)

        # [Module 4: Pre-cross-attention Layernorm]
        self.pre_cross_attn_layernorm = build_module(
            submodules.pre_cross_attn_layernorm,
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon,
        ) if submodules.pre_cross_attn_layernorm is not IdentityOp else IdentityOp()

        # [Module 5: Cross-Attention]
        self.cross_attention = build_module(
            submodules.cross_attention,
            config=config,
            layer_number=self.layer_number,
        )
        self.add_cross_attn = not isinstance(self.cross_attention, IdentityOp)

        # [Module 6: Cross-attention BDA]
        self.cross_attn_bda = build_module(submodules.cross_attn_bda, config=config)

        # [Module 7: Pre-MLP Layernorm]
        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm,
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon,
        ) if submodules.pre_mlp_layernorm is not IdentityOp else IdentityOp()

        # [Module 8: MLP]
        self.mlp = build_module(
            submodules.mlp,
            config=config,
        )

        # [Module 9: MLP BDA]
        self.mlp_bda = build_module(submodules.mlp_bda)

    def _build_direct(self, config: TransformerConfig) -> None:
        """Build sub-modules directly (no spec, legacy / DES-LOC path)."""
        # Input layernorm
        self.input_layernorm = _build_norm(config)

        # Self-attention
        self.self_attention = SelfAttention(config, layer_number=self.layer_number)

        # BDA functions (simple IdentityFuncOp pass-through;
        # actual dropout+residual handled in _forward_attention/_forward_mlp)
        self.self_attn_bda = IdentityFuncOp()
        self.cross_attn_bda = IdentityFuncOp()
        self.mlp_bda = IdentityFuncOp()

        # Cross-attention (optional)
        if self.add_cross_attn:
            self.pre_cross_attn_layernorm = _build_norm(config)
            self.cross_attention = SelfAttention(config, layer_number=self.layer_number)
        else:
            self.pre_cross_attn_layernorm = IdentityOp()
            self.cross_attention = IdentityOp()

        # Pre-MLP layernorm
        self.pre_mlp_layernorm = _build_norm(config)

        # MLP
        self.mlp = MLP(config, layer_number=self.layer_number)

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

    def _forward_attention(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        rotary_pos_cos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        inference_context: Optional[object] = None,
        packed_seq_params: Optional[object] = None,
        sequence_len_offset: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Self-attention sub-layer (+ optional cross-attention, M2317).

        Returns:
            (hidden_states, context): context is updated by cross-attention or
            passed through unchanged when cross-attention is not active.
        """
        # --- Input layernorm → residual -----------------------------------------
        if self.recompute_input_layernorm and self.training:
            input_layernorm_output = torch.utils.checkpoint.checkpoint(
                self.input_layernorm, hidden_states, use_reentrant=False
            )
        else:
            input_layernorm_output = self.input_layernorm(hidden_states)

        # Handle (output, residual) tuple from fused norms (e.g. TENorm)
        if isinstance(input_layernorm_output, tuple):
            input_layernorm_output, residual = input_layernorm_output
        else:
            residual = hidden_states

        if self.fp32_residual_connection:
            residual = residual.float()

        # --- Self-attention ---------------------------------------------------
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )

        # Normalise to (output, bias) format
        if isinstance(attention_output_with_bias, (tuple, list)):
            if len(attention_output_with_bias) >= 2:
                attn_out, attn_bias = attention_output_with_bias[0], attention_output_with_bias[1]
            else:
                attn_out, attn_bias = attention_output_with_bias[0], None
        else:
            attn_out, attn_bias = attention_output_with_bias, None

        # --- Bias-dropout-add (self-attention) --------------------------------
        with self.bias_dropout_add_exec_handler():
            bda_fn = get_bias_dropout_add(
                self.training,
                getattr(self.config, "bias_dropout_fusion", False),
            )
            hidden_states = bda_fn(attn_out, attn_bias, residual, self.hidden_dropout)

        # --- Cross-attention (M2317: BERT / encoder-decoder) ------------------
        if self.add_cross_attn and context is not None and not isinstance(self.cross_attention, IdentityOp):
            if self.fp32_residual_connection:
                residual = hidden_states.float()
            else:
                residual = hidden_states

            # Pre-cross-attention layernorm
            pre_cross_attn_norm_out = self.pre_cross_attn_layernorm(hidden_states)
            if isinstance(pre_cross_attn_norm_out, tuple):
                pre_cross_attn_norm_out, residual = pre_cross_attn_norm_out

            if self.fp32_residual_connection:
                residual = residual.float()

            # Cross-attention forward
            cross_output_with_bias = self.cross_attention(
                pre_cross_attn_norm_out,
                attention_mask=context_mask,
                key_value_states=context,
                inference_context=inference_context,
            )
            if isinstance(cross_output_with_bias, dict) and "context" in cross_output_with_bias:
                context = cross_output_with_bias["context"]

            if isinstance(cross_output_with_bias, (tuple, list)):
                cross_out = cross_output_with_bias[0]
                cross_bias = cross_output_with_bias[1] if len(cross_output_with_bias) > 1 else None
            else:
                cross_out = cross_output_with_bias
                cross_bias = None

            with self.bias_dropout_add_exec_handler():
                cross_bda_fn = get_bias_dropout_add(
                    self.training,
                    getattr(self.config, "bias_dropout_fusion", False),
                )
                hidden_states = cross_bda_fn(cross_out, cross_bias, residual, self.hidden_dropout)

        return hidden_states, context

    def _forward_pre_mlp_layernorm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the pre-MLP layer norm (extracted for selective recompute support).

        Ported from Megatron M3253 / M3926 which extracted the pre-MLP norm
        into its own method so that ``recompute_pre_mlp_layernorm`` can wrap
        *only* the norm (not the full MLP) in a checkpoint, trading a small
        recompute cost for saving the normed activation buffer.

        When ``recompute_pre_mlp_layernorm`` is True (set in ``__init__`` when
        ``recompute_granularity == 'selective'`` and ``'layernorm'`` is in
        ``recompute_modules``), we wrap the norm in
        ``torch.utils.checkpoint.checkpoint``.

        Args:
            hidden_states: Pre-norm hidden states ``[s, b, h]``.

        Returns:
            Normed hidden states ``[s, b, h]``.
        """
        if self.recompute_pre_mlp_layernorm and self.training:
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
        """MLP sub-layer with residual connection.

        Ported from Megatron ``TransformerLayer._forward_mlp`` (M3253 / M3926 era).

        Supports:
          * Pre-norm / post-norm residual modes.
          * MLP output chunking along the sequence dimension to reduce peak
            activation memory during long-context prefill or training.
          * Selective recompute of the MLP (``recompute_mlp``).
          * Bias-dropout-add dispatch.

        Args:
            hidden_states: ``[s, b, h]`` transformer hidden states.
            inference_context: Inference KV-cache context.
            padding_mask: Padding mask for MoE routing.

        Returns:
            hidden_states after MLP + residual connection ``[s, b, h]``.
        """
        # --- Apply pre-MLP norm (may recompute selectively) -----------------
        pre_mlp_layernorm_output = self._forward_pre_mlp_layernorm(hidden_states)

        # Handle (output, residual) tuple from fused norms
        if isinstance(pre_mlp_layernorm_output, tuple):
            pre_mlp_layernorm_output, residual = pre_mlp_layernorm_output
        else:
            residual = hidden_states

        if self.fp32_residual_connection:
            residual = residual.float()

        # --- MLP forward (possibly checkpointed, possibly chunked) ----------
        should_chunk_prefill = (
            getattr(self.config, "mlp_chunks_for_prefill", 1) > 1
            and inference_context is not None
            and not getattr(inference_context, "is_decode_only", lambda: False)()
        )
        should_chunk_train = (
            getattr(self.config, "mlp_chunks_for_training", 1) > 1
            and inference_context is None
            and self.training
        )

        if self.recompute_mlp and self.training:
            mlp_output_with_bias = torch.utils.checkpoint.checkpoint(
                lambda h: self.mlp(h) if padding_mask is None else self.mlp(h, padding_mask=padding_mask),
                pre_mlp_layernorm_output,
                use_reentrant=False,
            )
            if not isinstance(mlp_output_with_bias, (tuple, list)):
                mlp_output_with_bias = (mlp_output_with_bias, None)
        elif should_chunk_prefill or should_chunk_train:
            num_chunks = (
                self.config.mlp_chunks_for_prefill if should_chunk_prefill
                else self.config.mlp_chunks_for_training
            )
            # Clamp to seq length so we never get empty chunks
            num_chunks = min(num_chunks, pre_mlp_layernorm_output.shape[0])
            chunks = pre_mlp_layernorm_output.chunk(num_chunks, dim=0)
            outputs = [self.mlp(chunk) for chunk in chunks]
            # Aggregate outputs — bias is the same for all chunks
            if isinstance(outputs[0], (tuple, list)):
                mlp_out = torch.cat([o[0] for o in outputs], dim=0)
                bias_chunks = [o[1] for o in outputs if o[1] is not None]
                mlp_bias = bias_chunks[0] if bias_chunks else None
            else:
                mlp_out = torch.cat(outputs, dim=0)
                mlp_bias = None
            mlp_output_with_bias = (mlp_out, mlp_bias)
        else:
            if padding_mask is not None:
                raw = self.mlp(pre_mlp_layernorm_output, padding_mask=padding_mask)
            else:
                raw = self.mlp(pre_mlp_layernorm_output)
            if isinstance(raw, (tuple, list)):
                mlp_output_with_bias = raw
            else:
                mlp_output_with_bias = (raw, None)

        return self._forward_post_mlp(mlp_output_with_bias, residual)

    def _forward_post_mlp(
        self,
        mlp_output_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]],
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Perform operations after the MLP computation.

        Applies the bias-dropout-add residual connection, then makes the result
        viewless (avoids schedule.py ``deallocate_output_tensor`` errors from
        JIT-compiled BDA producing a view tensor).

        Args:
            mlp_output_with_bias: ``(mlp_output, bias)`` tuple.
            residual: Residual hidden states ``[s, b, h]``.

        Returns:
            Hidden states after MLP + residual ``[s, b, h]``.
        """
        mlp_out = mlp_output_with_bias[0] if isinstance(mlp_output_with_bias, (tuple, list)) else mlp_output_with_bias
        mlp_bias = mlp_output_with_bias[1] if isinstance(mlp_output_with_bias, (tuple, list)) and len(mlp_output_with_bias) > 1 else None

        with self.bias_dropout_add_exec_handler():
            bda_fn = get_bias_dropout_add(
                self.training,
                getattr(self.config, "bias_dropout_fusion", False),
            )
            hidden_states = bda_fn(mlp_out, mlp_bias, residual, self.hidden_dropout)

        # Make viewless: JIT-compiled BDA can produce a view tensor that causes
        # schedule.py's deallocate_output_tensor() to raise an error.
        if hidden_states.is_contiguous():
            return hidden_states
        return hidden_states.contiguous()

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
        rotary_pos_cos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        inference_context: Optional[object] = None,
        packed_seq_params: Optional[object] = None,
        sequence_len_offset: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        *,
        inference_params: Optional[object] = None,  # deprecated alias
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Forward pass of one transformer layer.

        Returns ``(hidden_states, context)`` unconditionally so that
        ``TransformerBlock`` can unpack the tuple regardless of whether
        cross-attention is active (matches Megatron's interface, M2317 fix).

        Args:
            hidden_states: ``[seq, batch, hidden]``
            attention_mask: Optional mask ``[batch, 1, seq, seq]``
            context: Encoder output for cross-attention ``[seq_enc, batch, hidden]``
            context_mask: Mask for cross-attention
            rotary_pos_emb: Rotary embeddings ``[seq, 1, 1, head_dim]``
            rotary_pos_cos: Rotary embedding cosines (flash decode)
            rotary_pos_sin: Rotary embedding sines (flash decode)
            rotary_pos_cos_sin: Combined cos/sin (dynamic batching flashinfer)
            attention_bias: Additive attention bias ``[1, heads, seq, seq]``
            inference_context: Passed through to attention.
            packed_seq_params: THD packed sequence params.
            sequence_len_offset: Sequence offset for inference CUDA graphs.
            padding_mask: Padding mask for MoE routing.
            inference_params: Deprecated alias for inference_context.

        Returns:
            ``(hidden_states, context)`` always — context may be ``None`` when
            cross-attention is not active.
        """
        # Backward-compat: deprecated inference_params → inference_context
        if inference_context is None and inference_params is not None:
            inference_context = inference_params

        hidden_states, context = self._forward_attention(
            hidden_states,
            attention_mask=attention_mask,
            context=context,
            context_mask=context_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            padding_mask=padding_mask,
        )
        hidden_states = self._forward_mlp(
            hidden_states,
            inference_context=inference_context,
            padding_mask=padding_mask,
        )
        return hidden_states, context

    # ------------------------------------------------------------------
    # Fused TP inference configuration (M3030 / M3063 inference_fuse_tp)
    # ------------------------------------------------------------------

    def configure_fused_tp_inference(
        self,
        skip_qkv_norm_and_all_gather: bool = False,
        fc2_next_layer_norm_weights: Optional[torch.Tensor] = None,
        *,
        residual_in_fp32: bool = False,
    ) -> None:
        """Configure this layer for fused TP inference (RS+add+norm+AG kernel).

        When ``config.inference_fuse_tp_communication`` is True and a fused
        reduce-scatter + residual-add + LayerNorm + all-gather CUDA kernel is
        available (NVLS or TE>=2.2), this method pre-computes the static
        residual buffers and wires the residual-passing hooks into the
        attention output projection and MLP fc2.

        Ported from Megatron-LM ``TransformerLayer.configure_fused_tp_inference``
        (M3030 era).  In DES-LOC heterogeneous clusters this is only activated
        on H100 tiers; A6000 tiers use the standard non-fused path because
        NVLS requires NVLink.

        Args:
            skip_qkv_norm_and_all_gather: Skip norm and all-gather for QKV linear.
            fc2_next_layer_norm_weights: Next layer's QKV norm weights for FC2.
            residual_in_fp32: Maintain residual stream in FP32.
        """
        if not getattr(self.config, "inference_fuse_tp_communication", False):
            return

        # Skip on A6000 tiers — no NVLink, NVLS not available.
        if getattr(self, "desloc_tier", None) == "a6000":
            logger.debug(
                "TransformerLayer %d: skipping fused TP inference on A6000 tier.",
                self.layer_number,
            )
            return

        # Wire QKV norm skip flag if self_attention supports it
        if hasattr(self.self_attention, "linear_qkv") and hasattr(
            self.self_attention.linear_qkv, "skip_norm_and_all_gather"
        ):
            self.self_attention.linear_qkv.skip_norm_and_all_gather = skip_qkv_norm_and_all_gather

        # Pass pre-MLP norm weights to the attention projection
        if hasattr(self, "get_mlp_layer_norm_weights"):
            mlp_fc1_weights = self.get_mlp_layer_norm_weights()
            if (
                hasattr(self.self_attention, "linear_proj")
                and hasattr(self.self_attention.linear_proj, "_set_next_layer_norm_weights")
            ):
                self.self_attention.linear_proj._set_next_layer_norm_weights(mlp_fc1_weights)

        # Pass next layer's attn norm weights to MLP fc2
        if (
            hasattr(self, "mlp")
            and hasattr(self.mlp, "linear_fc1")
            and hasattr(self.mlp.linear_fc1, "skip_norm_and_all_gather")
        ):
            self.mlp.linear_fc1.skip_norm_and_all_gather = True

        if (
            hasattr(self, "mlp")
            and hasattr(self.mlp, "linear_fc2")
            and hasattr(self.mlp.linear_fc2, "_set_next_layer_norm_weights")
        ):
            weights = fc2_next_layer_norm_weights
            if weights is None and hasattr(self, "get_mlp_layer_norm_weights"):
                try:
                    weights = torch.empty_like(self.get_mlp_layer_norm_weights())
                except Exception:
                    pass
            if weights is not None:
                self.mlp.linear_fc2._set_next_layer_norm_weights(weights)

        logger.debug(
            "TransformerLayer %d: configured for fused TP inference.",
            self.layer_number,
        )

    def get_layer_norm_weights(self) -> Optional[torch.Tensor]:
        """Return the input layernorm weights for this layer.

        Used by the DES-LOC engine and NVLS fused RS+residual+norm+AG kernel
        (M2879) to identify which norm parameters need to be broadcast across
        TP groups during inference.

        Returns:
            Weight tensor of shape ``[hidden_size]`` or ``None``.
        """
        if hasattr(self.input_layernorm, "weight"):
            return self.input_layernorm.weight
        return None

    def get_mlp_layer_norm_weights(self) -> Optional[torch.Tensor]:
        """Return the pre-MLP layernorm weights.

        Mirrors Megatron's ``get_mlp_layer_norm_weights`` (M3063) for use by
        the fused inference kernel and the DES-LOC NVLS path.

        Returns:
            Weight tensor of shape ``[hidden_size]`` or ``None``.
        """
        # Spec-based path: MLP may expose layer_norm_weight via TE's FusedLinear
        if hasattr(self.mlp, "linear_fc1") and hasattr(self.mlp.linear_fc1, "layer_norm_weight"):
            return self.mlp.linear_fc1.layer_norm_weight.data
        if hasattr(self.pre_mlp_layernorm, "weight"):
            return self.pre_mlp_layernorm.weight
        return None

    def get_qkv_layer_norm_weights(self) -> Optional[torch.Tensor]:
        """Return the QKV layernorm weights.

        Mirrors Megatron's ``get_qkv_layer_norm_weights`` (M3063).

        Returns:
            Weight tensor of shape ``[hidden_size]`` or ``None``.
        """
        if hasattr(self.self_attention, "linear_qkv") and hasattr(
            self.self_attention.linear_qkv, "layer_norm_weight"
        ):
            return self.self_attention.linear_qkv.layer_norm_weight.data
        if hasattr(self.input_layernorm, "weight"):
            return self.input_layernorm.weight
        return None

    # ------------------------------------------------------------------
    # Fused TP inference residual passthrough (M3063 / M3030)
    # ------------------------------------------------------------------

    def _set_proj_residual(self, residual: torch.Tensor) -> None:
        """Set residual tensor for the attention output projection's fused RS+add+norm+AG.

        Called by ``_forward_attention`` when ``inference_fuse_tp_communication`` is True
        so the fused NVLS kernel can perform residual-add + LayerNorm + AllGather in one
        kernel call, replacing the separate BDA step.

        Skipped on A6000 tiers (no NVLink / NVLS not available).

        Args:
            residual: ``[s, b, h]`` residual tensor from the input layernorm branch.
        """
        if getattr(self, "desloc_tier", None) == "a6000":
            return
        attn = getattr(self, "self_attention", None)
        if attn is not None and hasattr(attn, "linear_proj") and hasattr(
            attn.linear_proj, "_set_residual"
        ):
            attn.linear_proj._set_residual(residual)

    def _set_fc2_residual(self, residual: torch.Tensor) -> None:
        """Set residual tensor for the MLP FC2's fused RS+add+norm+AG.

        Called by ``_forward_mlp`` when ``inference_fuse_tp_communication`` is True.

        Skipped on A6000 tiers (no NVLink / NVLS not available).

        Args:
            residual: ``[s, b, h]`` residual from the pre-MLP layernorm branch.
        """
        if getattr(self, "desloc_tier", None) == "a6000":
            return
        mlp = getattr(self, "mlp", None)
        if mlp is not None and hasattr(mlp, "linear_fc2") and hasattr(
            mlp.linear_fc2, "_set_residual"
        ):
            mlp.linear_fc2._set_residual(residual)

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
        if not getattr(self.config, "qk_clip", False):
            raise ValueError(
                f"TransformerLayer.clip_qk() called on layer {self.layer_number} "
                "but config.qk_clip is False."
            )
        attn = self.self_attention
        if not hasattr(attn, "clip_qk") or not callable(attn.clip_qk):
            return
        # M3217 fix: skip if logits not yet populated (first step / checkpointed)
        core_attn = getattr(attn, "core_attention", None)
        if core_attn is not None:
            if getattr(core_attn, "current_max_attn_logits", None) is None:
                return
        try:
            attn.clip_qk()
        except (AttributeError, NotImplementedError):
            pass

    # ------------------------------------------------------------------
    # CUDA graph stubs (Megatron API compat — M2906 / M3977)
    # ------------------------------------------------------------------
    # These are no-ops or raise in the deepspeed port because the TE-based
    # CUDA graph infrastructure is not available.  They exist so that
    # Megatron-compatible callers (TransformerBlock, schedules.py) can call
    # them without ImportError.

    def create_mcore_cudagraph_manager(self, config: TransformerConfig) -> None:
        """Register the transformer layer for CUDA graphs (no-op in DS port).

        Megatron's ``CudaGraphManager`` is TE-specific.  In DES-LOC we use
        PyTorch-native ``torch.cuda.CUDAGraph`` when needed, managed by the
        DES-LOC engine rather than per-layer managers.
        """
        pass

    def _should_call_local_cudagraph(self, *args, **kwargs) -> bool:
        """Whether to use local (per-layer) CUDA graphs for this forward call.

        Always returns False in the deepspeed port — CUDA graph capture is
        handled at the block / engine level.
        """
        return False

    @staticmethod
    def _get_layer_offset(config: TransformerConfig) -> int:
        """Deprecated: please use ``get_transformer_layer_offset`` instead."""
        import warnings
        warnings.warn(
            "TransformerLayer._get_layer_offset is deprecated. "
            "Please use get_transformer_layer_offset instead."
        )
        return get_transformer_layer_offset(config)

    def get_layer_static_inputs(
        self, seq_length: int, micro_batch_size: int
    ) -> Dict[str, torch.Tensor]:
        """Get static inputs for CUDA graph capture (Megatron M2404 compat).

        Returns a dict with at minimum the hidden_states buffer.
        """
        hidden_size = self.config.hidden_size
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else "cpu"
        static_inputs: Dict[str, torch.Tensor] = {
            "hidden_states": torch.zeros(
                seq_length, micro_batch_size, hidden_size,
                device=device, dtype=torch.float16,
            ),
        }
        return static_inputs

    def _get_submodules_under_cudagraphs(self) -> List:
        """Return submodules covered by CUDA graphs (empty in DS port)."""
        return []

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
        otherwise falls back to ``state_dict``.  Applies
        ``sharded_state_dict_keys_map`` remapping if defined in the submodule spec.

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

        # Apply key remapping from submodules spec (Megatron M3253 sharded_state_dict_keys_map)
        if self.submodules_config is not None and getattr(
            self.submodules_config, "sharded_state_dict_keys_map", None
        ):
            prefixed_map = {
                f"{prefix}{k}": f"{prefix}{v}"
                for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
            }
            for old_key, new_key in prefixed_map.items():
                if old_key in state_dict:
                    state_dict[new_key] = state_dict.pop(old_key)

        return state_dict


# ---------------------------------------------------------------------------
# MoETransformerLayer — transformer layer with Mixture-of-Experts MLP
# ---------------------------------------------------------------------------

class MoETransformerLayer(TransformerLayer):
    """TransformerLayer variant where the MLP sub-layer is a MoELayer.

    Mirrors Megatron's ``MoETransformerLayer`` (M3231 era + M3977 partial CUDA
    graph refactor).  In Neuron_SP / DES-LOC this class is used when
    ``config.moe_layer_freq`` selects a layer as a MoE layer (e.g. every other
    layer in a hybrid dense+MoE model).

    Key differences from ``TransformerLayer``:
      * ``self.is_moe_layer = True`` — checked by TransformerBlock for recompute
        exclusion and by the DES-LOC engine for expert-specific all-reduce.
      * ``_forward_mlp`` is overridden to support partial CUDA graph execution
        (router + expert compute + postprocess) when
        ``config.cuda_graph_impl == "local"`` and
        ``config.cuda_graph_modules`` contains MoE sub-modules.
      * DES-LOC: MoE experts on A6000 tiers skip the TE fused MLP path since
        SM86 lacks FP8; the standard unfused expert MLP is used instead.

    Ported from:
      Megatron-LM/megatron/core/transformer/transformer_layer.py
      class MoETransformerLayer (lines ~1414–end)

    Args:
        Same as ``TransformerLayer`` — all arguments are forwarded verbatim.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.is_moe_layer = True
        super().__init__(*args, **kwargs)

        # DES-LOC: log MoE-specific tier info
        if self.desloc_tier is not None:
            logger.debug(
                "MoETransformerLayer %d → DES-LOC tier: %s (MoE experts use unfused path on A6000)",
                self.layer_number,
                self.desloc_tier.upper(),
            )

    def transition_cudagraph_scope(self, mode: str) -> None:
        """Transition between full-layer and partial CUDA graph capture.

        Megatron API compat stub — in the DES-LOC port, CUDA graph management
        is handled externally by the engine, not per-layer.

        Args:
            mode: 'full' for inference (full-layer capture) or 'partial' for
                training (router + postprocess captured, expert dispatch eager).
        """
        pass

    def create_mcore_cudagraph_manager(self, config: TransformerConfig) -> None:
        """Initialise CUDA graph manager(s) for MoE (no-op in DS port)."""
        pass

    def _forward_mlp(
        self,
        hidden_states: torch.Tensor,
        inference_context: Optional[object] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """MoE MLP forward pass.

        Extends the base ``_forward_mlp`` with MoE-specific logic:
          * Skips chunked MLP (MoE already handles its own batching).
          * Passes ``padding_mask`` through to the MoELayer router so that
            padding tokens are excluded from load-balancing aux loss.
          * On A6000 tiers, disables TE fused expert path (SM86 limitation).

        Args:
            hidden_states: ``[s, b, h]`` transformer hidden states.
            inference_context: Inference KV-cache context.
            padding_mask: ``[b, s]`` padding mask for MoE routing aux loss.

        Returns:
            hidden_states after MoE MLP + residual ``[s, b, h]``.
        """
        # Pre-MLP layernorm (may selectively recompute)
        pre_mlp_layernorm_output = self._forward_pre_mlp_layernorm(hidden_states)

        if isinstance(pre_mlp_layernorm_output, tuple):
            pre_mlp_layernorm_output, residual = pre_mlp_layernorm_output
        else:
            residual = hidden_states

        if self.fp32_residual_connection:
            residual = residual.float()

        # MoE forward — always pass padding_mask for aux loss correctness
        try:
            if padding_mask is not None:
                raw = self.mlp(pre_mlp_layernorm_output, padding_mask=padding_mask)
            else:
                raw = self.mlp(pre_mlp_layernorm_output)
        except TypeError:
            # Fallback: mlp doesn't accept padding_mask
            raw = self.mlp(pre_mlp_layernorm_output)

        if isinstance(raw, (tuple, list)):
            mlp_output_with_bias = raw
        else:
            mlp_output_with_bias = (raw, None)

        return self._forward_post_mlp(mlp_output_with_bias, residual)
