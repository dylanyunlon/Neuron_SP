# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
#
# Ported from Megatron-LM megatron/core/transformer/multi_latent_attention.py
# (Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.)
#
# Changes vs. Megatron-LM source
# --------------------------------
# * All `megatron.core.*` imports replaced with `deepspeed.core.*` equivalents.
# * `megatron.core.models.common.embeddings` → lazy-imported from Megatron at
#   call time with a graceful fallback shim so the module is usable without a
#   Megatron installation (training path still requires Megatron for RoPE).
# * `megatron.core.pipeline_parallel.fine_grained_activation_offload` →
#   replaced by a lightweight no-op shim (`_OffloadCtx`) so the interface is
#   preserved without the Megatron dependency.
# * `megatron.core.typed_torch.apply_module` / `not_none` → inline shims.
# * `megatron.core.dist_checkpointing.mapping.ShardedObject` and
#   `megatron.core.utils.make_tp_sharded_tensor_for_checkpoint` →
#   lazy-imported from Megatron; stubs raised NotImplementedError when absent.
# * `megatron.core.extensions.transformer_engine.*` → imported from Megatron
#   when available, else set to None (matching existing deepspeed/core pattern).
# * `ProcessGroupCollection` → `deepspeed.core.process_groups_config`.
# * `ColumnParallelLinear` → `deepspeed.core.tensor_parallel.layers`.
# * `gather_from_sequence_parallel_region`, `gather_from_tensor_model_parallel_region`,
#   `scatter_to_sequence_parallel_region` → `deepspeed.core.tensor_parallel.mappings`.
# * `Attention`, `LinearProjBuilder`, `AttnMaskType` → `deepspeed.core.transformer.*`.
# * `tensor_parallel.CheckpointWithoutOutput` → `deepspeed.core.tensor_parallel.random`.
# * `MLATransformerConfig` → `deepspeed.core.transformer.transformer_config`.
# * `ModuleSpec` / `build_module` / `LayerNormBuilder` → kept as thin wrappers
#   matching the Megatron interface (see _build_module / type aliases below).
from __future__ import annotations

import math
import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, NoReturn, Optional, Union

import torch
import torch.nn.functional as F

try:
    from einops import rearrange

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False

# ---------------------------------------------------------------------------
# DeepSpeed-native replacements for Megatron imports
# ---------------------------------------------------------------------------

from deepspeed.core.process_groups_config import ProcessGroupCollection
from deepspeed.core.tensor_parallel.layers import ColumnParallelLinear
from deepspeed.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from deepspeed.core.tensor_parallel.random import CheckpointWithoutOutput
from deepspeed.core.transformer.attention import Attention, LinearProjBuilder
from deepspeed.core.transformer.transformer_config import MLATransformerConfig

# AttnMaskType lives in deepspeed.core.enums (string-based in this port)
try:
    from deepspeed.core.enums import AttnMaskType
except ImportError:
    class AttnMaskType:  # type: ignore[no-redef]
        padding = "padding"
        causal = "causal"
        no_mask = "no_mask"

# ---------------------------------------------------------------------------
# ModuleSpec / build_module / LayerNormBuilder  (Megatron spec-utils shim)
# ---------------------------------------------------------------------------
# In Megatron these are spec dataclasses + factory functions.  DeepSpeed uses
# callables directly.  We expose the same surface so existing spec dictionaries
# work unchanged.

class ModuleSpec:
    """Thin wrapper: either a (module_class, kwargs) pair or a plain callable."""

    def __init__(self, module: type, params: dict | None = None):
        self.module = module
        self.params: dict = params or {}


def build_module(spec, *args, **kwargs):
    """Instantiate a module from a ModuleSpec, a type, or a callable builder.

    Mirrors ``megatron.core.transformer.spec_utils.build_module``:
    - If *spec* is a ``ModuleSpec`` instance, merge ``spec.params`` into
      *kwargs* and call ``spec.module(*args, **merged_kwargs)``.
    - If *spec* is a plain ``type`` or callable, call it directly.
    """
    if isinstance(spec, ModuleSpec):
        merged = {**spec.params, **kwargs}
        return spec.module(*args, **merged)
    elif callable(spec):
        return spec(*args, **kwargs)
    else:
        raise TypeError(f"build_module: unsupported spec type {type(spec)}")


# LayerNormBuilder is just a callable protocol; expose the name for type hints.
LayerNormBuilder = Callable  # type: ignore[type-arg]

# ---------------------------------------------------------------------------
# Fine-grained activation offload shim
# ---------------------------------------------------------------------------
# Megatron's `off_interface` is a context manager that optionally offloads
# intermediate tensors to CPU.  We provide a transparent no-op so the forward
# pass compiles without the Megatron pipeline-parallel dependency.

class _OffloadCtx:
    """No-op context manager matching Megatron's FineGrainedActivationOffloadingInterface."""

    def __init__(self, enabled: bool, tensor, name: str):
        self._enabled = enabled
        self._tensor = tensor

    def __enter__(self):
        return self._tensor

    def __exit__(self, *_):
        pass

    @staticmethod
    def group_commit(tensor, *, name: str, forced_released_tensors=None):
        return tensor


off_interface = _OffloadCtx

# ---------------------------------------------------------------------------
# apply_module / not_none shims  (megatron.core.typed_torch)
# ---------------------------------------------------------------------------

def apply_module(module):
    """Return a callable that forwards to *module*.  Mirrors typed_torch.apply_module."""
    return module


def not_none(value):
    """Assert value is not None and return it.  Mirrors typed_torch.not_none."""
    assert value is not None
    return value

# ---------------------------------------------------------------------------
# get_pg_size helper
# ---------------------------------------------------------------------------

def get_pg_size(pg) -> int:
    """Return the world size of *pg*, or 1 if pg is None."""
    if pg is None:
        return 1
    return torch.distributed.get_world_size(group=pg)

# ---------------------------------------------------------------------------
# deprecate_inference_params shim
# ---------------------------------------------------------------------------

def deprecate_inference_params(inference_context, inference_params):
    """Prefer inference_context; fall back to inference_params."""
    if inference_context is not None:
        return inference_context
    return inference_params

# ---------------------------------------------------------------------------
# is_te_min_version shim
# ---------------------------------------------------------------------------

def is_te_min_version(version_str: str) -> bool:
    """Check transformer-engine version.  Returns False when TE is absent."""
    try:
        import transformer_engine  # noqa: F401
        from packaging.version import Version
        te_ver = Version(transformer_engine.__version__)
        return te_ver >= Version(version_str)
    except Exception:
        return False

# ---------------------------------------------------------------------------
# ShardedObject / make_tp_sharded_tensor_for_checkpoint
# (lazy-import from Megatron; stubs when unavailable)
# ---------------------------------------------------------------------------

try:
    from megatron.core.dist_checkpointing.mapping import ShardedObject
except ImportError:
    @dataclass  # type: ignore[no-redef]
    class ShardedObject:  # type: ignore[no-redef]
        key: str
        data: object
        global_shape: tuple
        global_offset: tuple
        replica_id: object = None


try:
    from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint
except ImportError:
    def make_tp_sharded_tensor_for_checkpoint(tensor, key, tp_axis=0, prepend_offsets=()):  # type: ignore[misc]
        raise NotImplementedError(
            "make_tp_sharded_tensor_for_checkpoint requires megatron.core; "
            "install Megatron-LM or implement a DeepSpeed equivalent."
        )

# ---------------------------------------------------------------------------
# Transformer Engine optional imports  (matching deepspeed/core/transformer/attention.py)
# ---------------------------------------------------------------------------

try:
    import transformer_engine  # noqa: F401
    HAVE_TE = True
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TELayerNormColumnParallelLinear,
        TELinear,
        set_save_original_input,
        split_te_layernorm_column_parallel_linear,
    )
    try:
        from megatron.core.post_training.modelopt.layers import Linear as _ModelOptLinear
        Linear = _ModelOptLinear
    except ImportError:
        Linear = None  # type: ignore[assignment]
except ImportError:
    HAVE_TE = False
    (
        TEColumnParallelLinear,
        TELayerNormColumnParallelLinear,
        TELinear,
        Linear,
        set_save_original_input,
        split_te_layernorm_column_parallel_linear,
    ) = (None, None, None, None, None, None)

# ---------------------------------------------------------------------------
# RoPE imports  (lazy from Megatron; fallback stubs for import-time safety)
# ---------------------------------------------------------------------------

try:
    from megatron.core.models.common.embeddings import (
        RotaryEmbedding,
        YarnRotaryEmbedding,
        _yarn_get_mscale,
        apply_rotary_pos_emb,
    )
    _HAVE_MEGATRON_ROPE = True
except ImportError:
    _HAVE_MEGATRON_ROPE = False
    RotaryEmbedding = None          # type: ignore[assignment]
    YarnRotaryEmbedding = None      # type: ignore[assignment]

    def _yarn_get_mscale(scaling_factor, mscale_all_dim):  # type: ignore[misc]
        return 1.0

    def apply_rotary_pos_emb(*args, **kwargs):  # type: ignore[misc]
        raise NotImplementedError("Megatron-LM is required for MLA RoPE application.")

# ---------------------------------------------------------------------------
# Fused MLA RoPE kernels  (optional Megatron fusions)
# ---------------------------------------------------------------------------

try:
    from megatron.core.fusions.fused_mla_yarn_rope_apply import (
        fused_apply_mla_rope_for_kv,
        fused_apply_mla_rope_for_q,
    )
except Exception:
    fused_apply_mla_rope_for_kv = None  # type: ignore[assignment]
    fused_apply_mla_rope_for_q = None   # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Megatron tensor_parallel (for CheckpointWithoutOutput fp8 path)
# ---------------------------------------------------------------------------

try:
    from megatron.core import tensor_parallel as _megatron_tensor_parallel
except ImportError:
    _megatron_tensor_parallel = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from megatron.core.inference.contexts import BaseInferenceContext
    from megatron.core.packed_seq_params import PackedSeqParams


# ===========================================================================
# Helper functions
# ===========================================================================

def _prepare_mla_core_attention_value(parallel_attention, query, value, packed_seq_params):
    """Prepare value tensor for MLA core attention THD execution."""
    orig_v_dim = value.shape[-1] if value is not None else None
    padded_v_dim = orig_v_dim
    need_v_pad = (
        packed_seq_params is not None
        and packed_seq_params.qkv_format == "thd"
        and parallel_attention.config.experimental_attention_variant is None
        and value is not None
        and query.shape[-1] != orig_v_dim
    )
    if need_v_pad:
        value = F.pad(value, [0, query.shape[-1] - orig_v_dim])
        padded_v_dim = value.shape[-1]
    return value, need_v_pad, orig_v_dim, padded_v_dim


def _trim_mla_core_attention_output(core_attn_out, need_v_pad, orig_v_dim, padded_v_dim):
    """Trim THD MLA core attention output back to the original V dimension."""
    if need_v_pad:
        if core_attn_out.ndim == 2:
            core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-1], -1, padded_v_dim)
        core_attn_out = core_attn_out[..., :orig_v_dim]
    return core_attn_out


# ===========================================================================
# Submodule spec dataclass
# ===========================================================================

@dataclass
class MLASelfAttentionSubmodules:
    """Submodules for the MLA self-attention layer."""

    linear_proj: LinearProjBuilder

    # TODO(nschank): Move layernorms back to the bottom once all other layers have defaults removed.
    q_layernorm: LayerNormBuilder
    kv_layernorm: LayerNormBuilder

    linear_q_proj: Union[ModuleSpec, type] = None
    linear_q_down_proj: Union[ModuleSpec, type] = None
    linear_q_up_proj: Union[ModuleSpec, type] = None
    linear_kv_down_proj: Union[ModuleSpec, type] = None
    linear_kv_up_proj: Union[ModuleSpec, type] = None
    linear_qkv_down_proj: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None


# ===========================================================================
# MultiLatentAttention  (abstract base)
# ===========================================================================

class MultiLatentAttention(Attention):
    """Multi-Latent Attention layer abstract class.

    This layer only contains common modules required for the "self attn" and
    "cross attn" specializations.

    Ported from Megatron-LM with DeepSpeed-compatible imports.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: MLASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        pp_layer_offset: Optional[int] = None,
        name: str | None = None,
    ) -> None:
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attention_type=attention_type,
            attn_mask_type=attn_mask_type,
            pp_layer_offset=pp_layer_offset,
        )
        self.config: MLATransformerConfig

        # Attach process-group collection (use global parallel state when absent)
        if pg_collection is None:
            from deepspeed.core.process_groups_config import get_default_pg_collection
            pg_collection = get_default_pg_collection()
        self.pg_collection = pg_collection

        # Convenience alias used throughout (matches Megatron's self.tp_group)
        self.tp_group = pg_collection.tp

        self.query_projection_size = self.config.v_head_dim * self.config.num_attention_heads

        self.q_head_dim = self.config.qk_head_dim + self.config.qk_pos_emb_head_dim

        # Overwrite the base class kv shape to support MLA inference
        self.key_hidden_size = self.q_head_dim
        self.val_hidden_size = self.config.v_head_dim

        self.recompute_up_proj = (
            getattr(self.config, 'recompute_granularity', None) == 'selective'
            and "mla_up_proj" in getattr(self.config, 'recompute_modules', [])
        )
        self.qkv_up_checkpoint = None

        mscale = _yarn_get_mscale(self.config.rotary_scaling_factor, self.config.mscale_all_dim)
        self.softmax_scale = mscale * mscale / math.sqrt(self.q_head_dim)
        self.cache_mla_latents = self.config.cache_mla_latents

        if not _HAVE_MEGATRON_ROPE:
            raise ImportError(
                "MultiLatentAttention requires Megatron-LM for RoPE utilities. "
                "Please install Megatron-LM or provide an alternative RoPE implementation."
            )

        if self.config.rope_type == "rope":
            self.rotary_pos_emb = RotaryEmbedding(
                self.config.qk_pos_emb_head_dim,
                rotary_percent=self.config.rotary_percent,
                rotary_base=self.config.rotary_base,
                cp_group=self.pg_collection.cp,
            )
        elif self.config.rope_type == "yarn":
            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.config.qk_pos_emb_head_dim,
                rotary_base=self.config.rotary_base,
                scaling_factor=self.config.rotary_scaling_factor,
                original_max_position_embeddings=self.config.original_max_position_embeddings,
                beta_fast=self.config.beta_fast,
                beta_slow=self.config.beta_slow,
                mscale=self.config.mscale,
                mscale_all_dim=self.config.mscale_all_dim,
                cp_group=self.pg_collection.cp,
            )
        else:
            raise ValueError(
                f"Unsupported RoPE type: {self.config.rope_type}, supported types are "
                "'rope' and 'yarn'"
            )

        self.core_attention = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            softmax_scale=self.softmax_scale,
            k_channels=self.q_head_dim,
            v_channels=self.config.v_head_dim,
            cp_comm_type=cp_comm_type,
            pg_collection=self.pg_collection,
        )

        # Output.
        self.linear_proj = submodules.linear_proj(
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=not_none(self.config.output_layer_init_method),
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='proj',
            tp_group=self.pg_collection.tp,
            name=(name + ".linear_proj") if name is not None else None,
        )

        if (
            HAVE_TE
            and isinstance(self.linear_proj, TELinear)
            and (
                (
                    self.config.fp8
                    and self.config.fp8_recipe != 'delayed'
                    and is_te_min_version("2.6.0dev0")
                )
                or (self.config.fp4 and is_te_min_version("2.7.0.dev0"))
            )
        ):
            set_save_original_input(self.linear_proj)

    def _run_core_attention(
        self,
        query,
        key,
        value,
        attention_mask,
        packed_seq_params=None,
        attn_mask_type=None,
        **extra_kwargs,
    ):
        """Run MLA core attention with the THD value pad/trim workaround."""
        value, need_v_pad, orig_v_dim, padded_v_dim = _prepare_mla_core_attention_value(
            self, query, value, packed_seq_params
        )
        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type

        orig_v_head_dim = None
        if (
            need_v_pad
            and value is not None
            and hasattr(self.core_attention, 'hidden_size_per_attention_head_v')
        ):
            orig_v_head_dim = self.core_attention.hidden_size_per_attention_head_v
            if value.shape[-1] == orig_v_head_dim:
                orig_v_head_dim = None
            else:
                self.core_attention.hidden_size_per_attention_head_v = value.shape[-1]

        try:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                packed_seq_params=packed_seq_params,
                attn_mask_type=attn_mask_type,
                **extra_kwargs,
            )
        finally:
            if orig_v_head_dim is not None:
                self.core_attention.hidden_size_per_attention_head_v = orig_v_head_dim

        return _trim_mla_core_attention_output(core_attn_out, need_v_pad, orig_v_dim, padded_v_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        inference_context=None,
        rotary_pos_emb: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        rotary_pos_cos: torch.Tensor | None = None,
        rotary_pos_sin: torch.Tensor | None = None,
        rotary_pos_cos_sin: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        packed_seq_params=None,
        position_ids: torch.Tensor | None = None,
        sequence_len_offset: int | None = None,
        *,
        inference_params=None,
    ):
        """Forward pass for multi-latent attention."""
        assert rotary_pos_emb is None, "Rotary position embeddings should not be passed into MLA."
        assert attention_bias is None, "Attention bias should not be passed into MLA."
        assert (
            rotary_pos_cos is None and rotary_pos_sin is None
        ), "MLA does not support Flash Decoding"
        assert not rotary_pos_cos_sin, "Flash-infer rope has not been tested with MLA."
        assert not (
            self.training and self.cache_mla_latents
        ), "cache_mla_latents conflicts with training."

        # hidden_states: [sq, b, h]
        inference_context = deprecate_inference_params(inference_context, inference_params)
        if inference_context and not inference_context.is_static_batching():
            assert (
                self.config.cache_mla_latents
            ), "currently to use dynamic backend for MLA cache mla latents must be true"

        if self.config.cache_mla_latents:
            self.prepare_for_absorption()

        # =====================
        # Query, Key, and Value
        # =====================
        with off_interface(self.offload_qkv_linear, hidden_states, "qkv_linear") as hidden_states:
            query, key, value, q_compressed, kv_compressed = self.get_query_key_value_tensors(
                hidden_states,
                key_value_states,
                position_ids,
                packed_seq_params,
                inference_context=inference_context,
            )
        if self.offload_qkv_linear:
            query = off_interface.group_commit(
                query, name="qkv_linear", forced_released_tensors=[hidden_states]
            )

        # ===================================================
        # Adjust key, value for inference
        # ===================================================
        query, key, value, _, attn_mask_type, block_table = self._adjust_key_value_for_inference(
            inference_context, query, key, value, rotary_pos_emb=None
        )

        query = query.contiguous()
        key = key.contiguous()

        if value is not None:
            value = value.contiguous()

        thd_packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"

        # ==================================
        # core attention computation
        # ==================================
        needs_output_trim = False
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query, key, value, attention_mask, packed_seq_params=packed_seq_params
            )
        else:
            if inference_context is None or inference_context.is_static_batching():
                extra_kwargs = {}
                if self.config.experimental_attention_variant == "dsa":
                    extra_kwargs["x"] = hidden_states
                    extra_kwargs["qr"] = q_compressed
                with off_interface(
                    self.offload_core_attention and self.training, query, "core_attn"
                ) as query:
                    core_attn_out = self._run_core_attention(
                        query,
                        key,
                        value,
                        attention_mask,
                        packed_seq_params=packed_seq_params,
                        attn_mask_type=attn_mask_type,
                        **extra_kwargs,
                    )
            elif self.cache_mla_latents:
                value, need_v_pad, orig_v_dim, padded_v_dim = _prepare_mla_core_attention_value(
                    self, query, value, packed_seq_params
                )
                q, k, v = (query, key, value)
                cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
                cu_kv_lengths, kv_lengths, max_seqlen_k = inference_context.cu_kv_lengths()

                core_attn_out = self.flash_decode_and_prefill(
                    q,
                    k,
                    v,
                    max_seqlen_q,
                    max_seqlen_k,
                    cu_query_lengths,
                    cu_kv_lengths,
                    kv_lengths,
                    block_table,
                )
                if not inference_context.is_decode_only():
                    core_attn_out = rearrange(core_attn_out, 's b h d -> s b (h d)')
                needs_output_trim = need_v_pad
            if self.offload_core_attention and self.training:
                core_attn_out = off_interface.group_commit(
                    core_attn_out, name="core_attn", forced_released_tensors=[query, key, value]
                )

        # Absorption with cache_mla_latents in decode mode.
        if self.cache_mla_latents and inference_context.is_decode_only():
            core_attn_out = torch.einsum("sbhc,hdc->sbhd", core_attn_out, self.up_v_weight)
            core_attn_out = core_attn_out.contiguous()
            core_attn_out = core_attn_out.view(core_attn_out.size(0), core_attn_out.size(1), -1)

        if needs_output_trim:
            core_attn_out = _trim_mla_core_attention_output(
                core_attn_out, need_v_pad, orig_v_dim, padded_v_dim
            )

        if thd_packed_seq:
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        if self.recompute_up_proj:
            assert self.qkv_up_checkpoint is not None
            self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
            self.qkv_up_checkpoint = None

        # =================
        # Output. [sq, b, h]
        # =================
        with off_interface(self.offload_attn_proj, core_attn_out, "attn_proj") as core_attn_out:
            output, bias = apply_module(self.linear_proj)(core_attn_out)
        if self.offload_attn_proj:
            output = off_interface.group_commit(
                output, name="attn_proj", forced_released_tensors=[core_attn_out]
            )

        return output, bias


# ===========================================================================
# MLASelfAttention
# ===========================================================================

class MLASelfAttention(MultiLatentAttention):
    """MLA Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: MLASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        pp_layer_offset: Optional[int] = None,
        name: str | None = None,
    ):
        if pg_collection is None:
            from deepspeed.core.process_groups_config import get_default_pg_collection
            pg_collection = get_default_pg_collection()

        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            name=name,
        )

        if self.config.q_lora_rank is None:
            # Not projecting query
            self.linear_q_proj = build_module(
                submodules.linear_q_proj,
                self.config.hidden_size,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='q_proj',
                name=(name + ".linear_q_proj") if name is not None else None,
            )

        else:
            q_down_proj_kwargs = {}
            if submodules.linear_q_down_proj in [TELinear]:
                q_down_proj_kwargs['parallel_mode'] = 'duplicated'
            elif submodules.linear_q_down_proj in [
                Linear,
                TEColumnParallelLinear,
                ColumnParallelLinear,
            ]:
                q_down_proj_kwargs['gather_output'] = False
            else:
                raise ValueError(f"Unsupported linear_q_down_proj: {submodules.linear_q_down_proj}")

            self.linear_q_down_proj = build_module(
                submodules.linear_q_down_proj,
                self.config.hidden_size,
                self.config.q_lora_rank,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='q_down_proj',
                skip_weight_param_allocation=False,
                tp_group=(
                    pg_collection.tp
                    if q_down_proj_kwargs.get('parallel_mode') != 'duplicated'
                    else None
                ),
                name=(name + ".linear_q_down_proj") if name is not None else None,
                **q_down_proj_kwargs,
            )

            self.linear_q_up_proj = build_module(
                submodules.linear_q_up_proj,
                self.config.q_lora_rank,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='q_up_proj',
                tp_group=pg_collection.tp,
                name=(name + ".linear_q_up_proj") if name is not None else None,
            )

        kv_down_proj_kwargs = {}
        if submodules.linear_kv_down_proj in [TELinear]:
            kv_down_proj_kwargs['parallel_mode'] = 'duplicated'
        elif submodules.linear_kv_down_proj in [
            Linear,
            TEColumnParallelLinear,
            ColumnParallelLinear,
        ]:
            kv_down_proj_kwargs['gather_output'] = False
        else:
            raise ValueError(f"Unsupported linear_kv_down_proj: {submodules.linear_kv_down_proj}")

        self.linear_kv_down_proj = build_module(
            submodules.linear_kv_down_proj,
            self.config.hidden_size,
            self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='kv_down_proj',
            skip_weight_param_allocation=False,
            tp_group=(
                pg_collection.tp
                if kv_down_proj_kwargs.get('parallel_mode') != 'duplicated'
                else None
            ),
            name=(name + ".linear_kv_down_proj") if name is not None else None,
            **kv_down_proj_kwargs,
        )

        self.linear_kv_up_proj = build_module(
            submodules.linear_kv_up_proj,
            self.config.kv_lora_rank,
            self.config.num_attention_heads * (self.config.qk_head_dim + self.config.v_head_dim),
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='kv_up_proj',
            tp_group=pg_collection.tp,
            name=(name + ".linear_kv_up_proj") if name is not None else None,
        )

        if self.config.q_lora_rank is not None:
            self.q_layernorm = submodules.q_layernorm(
                hidden_size=self.config.q_lora_rank,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )

        self.kv_layernorm = submodules.kv_layernorm(
            hidden_size=self.config.kv_lora_rank,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

    def _qkv_down_projection(self, hidden_states):
        """Unfused q/kv down projection path."""
        if self.config.q_lora_rank is not None:
            q_compressed, _ = self.linear_q_down_proj(hidden_states)
            if q_compressed.size(-1) != self.config.q_lora_rank:
                q_compressed = gather_from_tensor_model_parallel_region(q_compressed)
                if self.config.sequence_parallel:
                    q_compressed = scatter_to_sequence_parallel_region(q_compressed)
        else:
            q_compressed = hidden_states

        kv_combined, _ = self.linear_kv_down_proj(hidden_states)
        return q_compressed, kv_combined

    def get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
        inference_context=None,
        *,
        inference_params=None,
    ):
        """Derives `query`, `key` and `value` tensors from `hidden_states`."""
        assert (
            hidden_states.ndim == 3
        ), f"hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D"
        if packed_seq_params is not None:
            assert (
                packed_seq_params.local_cp_size is None
            ), "hybrid_context_parallel is not supported with MLA yet and is planned for future. \
            Please disable hybrid_context_parallel."

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # =========================================
        # Prepare RoPE and seqlen related params
        # =========================================
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            inference_context, None, hidden_states, self.config, packed_seq_params
        )

        mscale = 1.0
        rotary_pos_cos = None
        rotary_pos_sin = None
        thd_packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=thd_packed_seq)
        else:
            if self.config.apply_rope_fusion:
                rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb.get_cached_cos_sin(
                    rotary_seq_len, dtype=hidden_states.dtype, packed_seq=thd_packed_seq
                )
                rotary_pos_emb = None
                assert inference_context is None, "Inference with MLA RoPE fusion is not supported"
                assert (
                    fused_apply_mla_rope_for_q is not None
                    and fused_apply_mla_rope_for_kv is not None
                ), "Fused MLA RoPE apply is not imported successfully"
            else:
                rotary_pos_emb, mscale = self.rotary_pos_emb(
                    rotary_seq_len, packed_seq=thd_packed_seq
                )

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            if packed_seq_params.cu_seqlens_q_padded is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
            else:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
            if packed_seq_params.cu_seqlens_kv_padded is not None:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
            else:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        # =========================================
        # QKV down projection and layernorm
        # =========================================
        q_compressed, kv_combined = self._qkv_down_projection(hidden_states)
        if kv_combined.size(-1) != self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim:
            kv_combined = gather_from_tensor_model_parallel_region(kv_combined)
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
            )
            if self.config.sequence_parallel:
                kv_compressed = scatter_to_sequence_parallel_region(kv_compressed)
        else:
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
            )
            if get_pg_size(self.tp_group) > 1 and self.config.sequence_parallel:
                k_pos_emb = gather_from_sequence_parallel_region(k_pos_emb, group=self.tp_group)

        if packed_seq_params is not None:
            q_compressed = q_compressed.squeeze(1)
            kv_compressed = kv_compressed.squeeze(1)
            k_pos_emb = k_pos_emb.squeeze(1)

        # =========================================
        # Apply norm
        # =========================================
        if self.config.q_lora_rank is not None:
            q_compressed = apply_module(self.q_layernorm)(q_compressed)

        kv_compressed = apply_module(self.kv_layernorm)(kv_compressed)

        # =========================================
        # QKV up projection and RoPE apply
        # =========================================

        def qkv_up_proj_and_rope_apply_for_cached_latent_kv(
            q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
        ):
            if self.config.q_lora_rank is not None:
                q, _ = self.linear_q_up_proj(q_compressed)
            else:
                q, _ = self.linear_q_proj(q_compressed)

            q = q.view(*q.size()[:-1], self.num_attention_heads_per_partition, self.q_head_dim)

            k_pos_emb = torch.unsqueeze(k_pos_emb, -2)

            q_no_pe, q_pos_emb = torch.split(
                q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1
            )

            q_pos_emb = inference_context.apply_rotary_emb_query(
                q_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cu_seqlens_q=cu_seqlens_q,
                cp_group=self.pg_collection.cp,
                mscale=mscale,
            )
            k_pos_emb = inference_context.apply_rotary_emb_key(
                k_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cp_group=self.pg_collection.cp,
                mscale=mscale,
            )

            k_pos_emb_squeezed = k_pos_emb.squeeze(1)
            kv_cached = torch.cat([kv_compressed, k_pos_emb_squeezed], dim=-1)

            use_absorption = (
                self.config.cache_mla_latents
                and inference_context
                and inference_context.is_decode_only()
            )
            q_content = (
                torch.einsum("sbhd,hdk->sbhk", q_no_pe, self.up_k_weight)
                if use_absorption
                else q_no_pe
            )
            query = torch.cat([q_content, q_pos_emb], dim=-1)

            key = kv_cached
            value = None

            query = query.contiguous()
            key = key.contiguous()

            return query, key, value

        def qkv_up_proj_and_rope_apply(q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb):
            """Apply the up projection and RoPE to the query and key."""
            if self.config.q_lora_rank is not None:
                q, _ = self.linear_q_up_proj(q_compressed)
            else:
                q, _ = self.linear_q_proj(q_compressed)

            q = q.view(*q.size()[:-1], self.num_attention_heads_per_partition, self.q_head_dim)

            kv, _ = self.linear_kv_up_proj(kv_compressed)

            kv = kv.view(
                *kv.size()[:-1],
                self.num_attention_heads_per_partition,
                self.config.qk_head_dim + self.config.v_head_dim,
            )

            k_pos_emb = torch.unsqueeze(k_pos_emb, -2)

            if self.config.apply_rope_fusion:
                cp_rank = self.pg_collection.cp.rank()
                cp_size = self.pg_collection.cp.size()
                query = fused_apply_mla_rope_for_q(
                    q,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    self.config.qk_head_dim,
                    self.config.qk_pos_emb_head_dim,
                    cu_seqlens_q,
                    cp_rank,
                    cp_size,
                )
                key, value = fused_apply_mla_rope_for_kv(
                    kv,
                    k_pos_emb,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    self.config.qk_pos_emb_head_dim,
                    self.config.qk_head_dim,
                    self.config.v_head_dim,
                    cu_seqlens_kv,
                    cp_rank,
                    cp_size,
                )
            else:
                q_len = q.size()[0]
                if inference_context is not None:
                    sequence_start = inference_context.sequence_len_offset
                    sequence_end = sequence_start + q_len
                    rotary_pos_emb = rotary_pos_emb[sequence_start:sequence_end]
                elif packed_seq_params is None or self.config.context_parallel_size == 1:
                    rotary_pos_emb = rotary_pos_emb[0:q_len]

                q_no_pe, q_pos_emb = torch.split(
                    q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1
                )

                k_no_pe, value = torch.split(
                    kv, [self.config.qk_head_dim, self.config.v_head_dim], dim=-1
                )

                q_pos_emb = apply_rotary_pos_emb(
                    q_pos_emb,
                    rotary_pos_emb,
                    config=self.config,
                    cu_seqlens=cu_seqlens_q,
                    mscale=mscale,
                    cp_group=self.pg_collection.cp,
                    mla_rotary_interleaved=True,
                )
                k_pos_emb = apply_rotary_pos_emb(
                    k_pos_emb,
                    rotary_pos_emb,
                    config=self.config,
                    cu_seqlens=cu_seqlens_kv,
                    mscale=mscale,
                    cp_group=self.pg_collection.cp,
                    mla_rotary_interleaved=True,
                )

                query = torch.cat([q_no_pe, q_pos_emb], dim=-1)

                if k_pos_emb.ndim == 4:
                    k_pos_emb = k_pos_emb.expand(-1, -1, self.num_attention_heads_per_partition, -1)
                else:
                    assert k_pos_emb.ndim == 3
                    k_pos_emb = k_pos_emb.expand(-1, self.num_attention_heads_per_partition, -1)
                key = torch.cat([k_no_pe, k_pos_emb], dim=-1)

            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

            return query, key, value

        if self.recompute_up_proj:
            quantization = self.config.fp8 or self.config.fp4
            # Prefer deepspeed CheckpointWithoutOutput; fall back to Megatron's if needed
            self.qkv_up_checkpoint = CheckpointWithoutOutput(fp8=quantization)
            query, key, value = self.qkv_up_checkpoint.checkpoint(
                qkv_up_proj_and_rope_apply, q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
            )
        else:
            if self.cache_mla_latents:
                assert (
                    inference_context and not inference_context.is_static_batching()
                ), "Caching MLA latents only works with dynamic backend inference"
                query, key, value = qkv_up_proj_and_rope_apply_for_cached_latent_kv(
                    q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
                )
            else:
                query, key, value = qkv_up_proj_and_rope_apply(
                    q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
                )

        return query, key, value, q_compressed, kv_compressed

    def uncompress_kv_from_cache(self, kv_cached):
        """Take a compressed kv and uncompress them."""
        kv_compressed, k_pos_emb = torch.split(
            kv_cached, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
        )

        kv, _ = self.linear_kv_up_proj_linear(kv_compressed)

        kv = kv.view(
            *kv.size()[:-1],
            self.num_attention_heads_per_partition,
            self.config.qk_head_dim + self.config.v_head_dim,
        )

        k_no_pe, value = torch.split(kv, [self.config.qk_head_dim, self.config.v_head_dim], dim=-1)

        k_pos_emb = k_pos_emb.unsqueeze(-2)
        k_pos_emb = k_pos_emb.expand(-1, -1, self.num_attention_heads_per_partition, -1)

        key = torch.cat([k_no_pe, k_pos_emb], dim=-1)
        return key, value

    def prepare_for_absorption(self):
        """Prepare the model for absorption optimization in MLA.

        Splits the fused layernorm + linear (linear_kv_up_proj) into separate
        components and extracts K/V up-projection weights for the absorption path.
        """
        if not hasattr(self, "up_k_weight"):
            with torch.no_grad():
                linear_kv_up_proj_norm, linear_kv_up_proj_linear = (
                    split_te_layernorm_column_parallel_linear(
                        self.linear_kv_up_proj, self.config, None, self.linear_kv_up_proj.tp_group
                    )
                )

                self.kv_layernorm = linear_kv_up_proj_norm
                self.linear_kv_up_proj_linear = linear_kv_up_proj_linear

                kv_up_weight = self.linear_kv_up_proj.weight
                kv_up_weight = kv_up_weight.view(
                    self.num_attention_heads_per_partition,
                    self.config.qk_head_dim + self.config.v_head_dim,
                    self.config.kv_lora_rank,
                )
                self.up_k_weight = kv_up_weight[
                    :, : self.config.qk_head_dim, :
                ]
                self.up_v_weight = kv_up_weight[
                    :, self.config.qk_head_dim :, :
                ]

                del self.linear_kv_up_proj

    def backward_dw(self) -> NoReturn:
        """Execute weight gradient computation."""
        self._backward_kv_proj()
        self._backward_q_proj()
        self._backward_output_proj()

    def _backward_kv_proj(self):
        """Computes weight gradients of KV projection layers."""
        self.linear_kv_up_proj.backward_dw()
        self.linear_kv_down_proj.backward_dw()

    def _backward_q_proj(self):
        """Computes weight gradients of Q projection layers."""
        if self.config.q_lora_rank is None:
            self.linear_q_proj.backward_dw()
        else:
            self.linear_q_down_proj.backward_dw()
            self.linear_q_up_proj.backward_dw()

    def _backward_output_proj(self):
        """Computes weight gradients of output projection layer."""
        self.linear_proj.backward_dw()

    def set_for_recompute_input_layernorm(self):
        """Set the attention layer for recompute input_layernorm. Only needed for fp8/fp4."""
        if self.config.q_lora_rank is not None:
            set_save_original_input(self.linear_q_down_proj)
        set_save_original_input(self.linear_kv_down_proj)

    def clip_qk(self):
        """QK Clipping to prevent attention logit explosion (MuonClip).

        Called after Muon optimizer step to clamp q/k weights.
        """
        if not self.config.qk_clip:
            raise ValueError("qk_clip option needs to be enabled")

        if self.core_attention.current_max_attn_logits is None:
            raise ValueError("current_max_attn_logits is None")

        if self.cache_mla_latents and not hasattr(self, 'linear_kv_up_proj'):
            raise ValueError(
                "qk_clip is not supported when cache_mla_latents is enabled and absorption is "
                "active. The linear_kv_up_proj layer has been deleted during absorption "
                "preparation."
            )

        assert self.core_attention.current_max_attn_logits.shape == (
            self.num_attention_heads_per_partition,
        ), (
            f"current_max_attn_logits shape is not ({self.num_attention_heads_per_partition}, ) "
            f"but {self.core_attention.current_max_attn_logits.shape}"
        )

        if torch.any(self.core_attention.current_max_attn_logits > self.config.qk_clip_threshold):
            assert self.core_attention.current_max_attn_logits.shape == (
                self.num_attention_heads_per_partition,
            ), (
                f"current_max_attn_logits shape is not ({self.num_attention_heads_per_partition},) "
                f"but {self.core_attention.current_max_attn_logits.shape}"
            )
            self.qk_clip_balancing_eta = torch.clamp(
                self.config.qk_clip_threshold / self.core_attention.current_max_attn_logits, max=1.0
            ).view(self.num_attention_heads_per_partition, 1, 1)
            assert torch.all(self.qk_clip_balancing_eta <= 1.0)

            if self.config.q_lora_rank is None:
                q_proj_weight = self.linear_q_proj.weight
            else:
                q_proj_weight = self.linear_q_up_proj.weight

            if hasattr(q_proj_weight, 'main_param'):
                q_proj_weight.main_param.data.copy_(
                    self._clip_q_proj_weight(q_proj_weight.main_param.data)
                )
            q_proj_weight.data.copy_(self._clip_q_proj_weight(q_proj_weight.data))

            kv_proj_weight = self.linear_kv_up_proj.weight

            if hasattr(kv_proj_weight, 'main_param'):
                kv_proj_weight.main_param.data.copy_(
                    self._clip_kv_proj_weight(kv_proj_weight.main_param.data)
                )
            kv_proj_weight.data.copy_(self._clip_kv_proj_weight(kv_proj_weight.data))

        self.core_attention.current_max_attn_logits = None

    def _clip_q_proj_weight(self, weight):
        """Clip q_proj_weight."""
        weight_reshaped = weight.view(
            self.num_attention_heads_per_partition,
            self.config.qk_head_dim + self.config.qk_pos_emb_head_dim,
            -1,
        )

        weight_q_nope = weight_reshaped[:, : self.config.qk_head_dim, :]
        weight_q_pe = weight_reshaped[:, self.config.qk_head_dim :, :]

        weight_q_nope.mul_(torch.pow(self.qk_clip_balancing_eta, self.config.qk_clip_alpha))
        weight_q_pe.mul_(self.qk_clip_balancing_eta)

        weight_q_updated = torch.cat([weight_q_nope, weight_q_pe], dim=1)
        weight_q_updated = weight_q_updated.view(
            self.num_attention_heads_per_partition
            * (self.config.qk_head_dim + self.config.qk_pos_emb_head_dim),
            -1,
        )

        return weight_q_updated

    def _clip_kv_proj_weight(self, weight):
        """Clip kv_proj_weight."""
        weight_reshaped = weight.view(
            self.num_attention_heads_per_partition,
            self.config.qk_head_dim + self.config.v_head_dim,
            -1,
        )

        weight_k = weight_reshaped[:, : self.config.qk_head_dim, :]
        weight_v = weight_reshaped[:, self.config.qk_head_dim :, :]

        weight_k.mul_(torch.pow(self.qk_clip_balancing_eta, 1 - self.config.qk_clip_alpha))

        weight_kv_updated = torch.cat([weight_k, weight_v], dim=1)
        weight_kv_updated = weight_kv_updated.view(
            self.num_attention_heads_per_partition
            * (self.config.qk_head_dim + self.config.v_head_dim),
            -1,
        )

        return weight_kv_updated


# ===========================================================================
# FusedMLASelfAttention
# ===========================================================================

class FusedMLASelfAttention(MLASelfAttention):
    """MLA self-attention with fused q/kv down projection."""

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: MLASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        pp_layer_offset: Optional[int] = None,
        name: str | None = None,
    ):
        if pg_collection is None:
            from deepspeed.core.process_groups_config import get_default_pg_collection
            pg_collection = get_default_pg_collection()

        MultiLatentAttention.__init__(
            self,
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            name=name,
        )

        assert self.config.q_lora_rank is not None, (
            "FusedMLASelfAttention requires q_lora_rank to be set; "
            "fallback to MLASelfAttention for q_lora_rank=None."
        )

        qkv_down_proj_kwargs = {}
        if submodules.linear_qkv_down_proj in [TELinear]:
            qkv_down_proj_kwargs['parallel_mode'] = 'duplicated'
        elif submodules.linear_qkv_down_proj in [
            Linear,
            TEColumnParallelLinear,
            ColumnParallelLinear,
            TELayerNormColumnParallelLinear,
        ]:
            qkv_down_proj_kwargs['gather_output'] = False
        else:
            raise ValueError(f"Unsupported linear_qkv_down_proj: {submodules.linear_qkv_down_proj}")

        self.linear_qkv_down_proj = build_module(
            submodules.linear_qkv_down_proj,
            self.config.hidden_size,
            self.config.q_lora_rank + self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv_down_proj',
            skip_weight_param_allocation=False,
            tp_group=(
                pg_collection.tp
                if qkv_down_proj_kwargs.get('parallel_mode') != 'duplicated'
                else None
            ),
            name=(name + ".linear_qkv_down_proj") if name is not None else None,
            **qkv_down_proj_kwargs,
        )

        self.linear_q_up_proj = build_module(
            submodules.linear_q_up_proj,
            self.config.q_lora_rank,
            self.config.num_attention_heads * self.q_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='q_up_proj',
            tp_group=pg_collection.tp,
            name=(name + ".linear_q_up_proj") if name is not None else None,
        )

        self.linear_kv_up_proj = build_module(
            submodules.linear_kv_up_proj,
            self.config.kv_lora_rank,
            self.config.num_attention_heads * (self.config.qk_head_dim + self.config.v_head_dim),
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='kv_up_proj',
            tp_group=pg_collection.tp,
            name=(name + ".linear_kv_up_proj") if name is not None else None,
        )

        self.q_layernorm = submodules.q_layernorm(
            hidden_size=self.config.q_lora_rank,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )
        self.kv_layernorm = submodules.kv_layernorm(
            hidden_size=self.config.kv_lora_rank,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

    def _qkv_down_projection(self, hidden_states):
        """Fused q/kv down projection path."""
        qkv, _ = self.linear_qkv_down_proj(hidden_states)
        q_compressed, kv_combined = torch.split(
            qkv,
            [self.config.q_lora_rank, self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim],
            dim=-1,
        )
        return q_compressed, kv_combined

    def backward_dw(self) -> NoReturn:
        """Execute weight gradient computation."""
        self.linear_kv_up_proj.backward_dw()
        self.linear_qkv_down_proj.backward_dw()
        self.linear_q_up_proj.backward_dw()
        self._backward_output_proj()

    def set_for_recompute_input_layernorm(self):
        """Set the attention layer for recompute input_layernorm. Only needed for fp8/fp4."""
        set_save_original_input(self.linear_qkv_down_proj)

    def sharded_state_dict(self, prefix: str = "", sharded_offsets: tuple = (), metadata=None):
        """Return a sharded state dict compatible with pre-fusion checkpoints."""
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)

        def _clone_sharded_object_with_key(obj: ShardedObject, new_key: str) -> ShardedObject:
            return ShardedObject(
                key=new_key,
                data=obj.data,
                global_shape=obj.global_shape,
                global_offset=obj.global_offset,
                replica_id=obj.replica_id,
            )

        fused_prefix = f"{prefix}linear_qkv_down_proj."

        fused_extra_keys = [
            k
            for k in sharded_state_dict.keys()
            if k.startswith(fused_prefix) and "_extra_state" in k
        ]
        for fused_extra_key in fused_extra_keys:
            suffix = fused_extra_key[len(fused_prefix):]
            q_extra_key = f"{prefix}linear_q_down_proj.{suffix}"
            kv_extra_key = f"{prefix}linear_kv_down_proj.{suffix}"
            fused_obj = sharded_state_dict.get(fused_extra_key)
            if isinstance(fused_obj, ShardedObject):
                sharded_state_dict[q_extra_key] = _clone_sharded_object_with_key(
                    fused_obj, q_extra_key
                )
                sharded_state_dict[kv_extra_key] = _clone_sharded_object_with_key(
                    fused_obj, kv_extra_key
                )
            elif fused_obj is not None:
                sharded_state_dict[q_extra_key] = fused_obj
                sharded_state_dict[kv_extra_key] = fused_obj

        for key in list(sharded_state_dict.keys()):
            if key.startswith(fused_prefix):
                del sharded_state_dict[key]

        fused_weight = self.linear_qkv_down_proj.weight
        total_out = (
            self.config.q_lora_rank + self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim
        )
        tp_size = get_pg_size(self.tp_group)

        if fused_weight.size(0) == total_out:
            q_split = self.config.q_lora_rank
            kv_split = self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim
        else:
            assert (
                self.config.q_lora_rank % tp_size == 0
            ), "q_lora_rank must be divisible by tensor-parallel size"
            assert (
                self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim
            ) % tp_size == 0, (
                "kv_lora_rank + qk_pos_emb_head_dim must be divisible by tensor-parallel size"
            )
            q_split = self.config.q_lora_rank // tp_size
            kv_split = (self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim) // tp_size

        if q_split + kv_split != fused_weight.size(0):
            raise ValueError(
                "Unexpected fused qkv-down weight shape: "
                f"got {tuple(fused_weight.size())}, expected dim0 {q_split + kv_split}"
            )

        q_weight, kv_weight = torch.split(fused_weight, [q_split, kv_split], dim=0)

        q_key = f"{prefix}linear_q_down_proj.weight"
        kv_key = f"{prefix}linear_kv_down_proj.weight"

        sharded_state_dict[q_key] = make_tp_sharded_tensor_for_checkpoint(
            tensor=q_weight, key=q_key, tp_axis=0, prepend_offsets=sharded_offsets
        )
        sharded_state_dict[kv_key] = make_tp_sharded_tensor_for_checkpoint(
            tensor=kv_weight, key=kv_key, tp_axis=0, prepend_offsets=sharded_offsets
        )

        return sharded_state_dict

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Load state dict with automatic unfused->fused conversion."""
        q_key = f"{prefix}linear_q_down_proj.weight"
        kv_key = f"{prefix}linear_kv_down_proj.weight"
        fused_key = f"{prefix}linear_qkv_down_proj.weight"

        def _as_tensor(x):
            return x.data if hasattr(x, 'data') else x

        if fused_key not in state_dict and q_key in state_dict and kv_key in state_dict:
            q_weight = _as_tensor(state_dict[q_key])
            kv_weight = _as_tensor(state_dict[kv_key])
            state_dict[fused_key] = torch.cat([q_weight, kv_weight], dim=0)
            del state_dict[q_key]
            del state_dict[kv_key]
            state_dict.pop(f"{prefix}linear_q_down_proj.bias", None)
            state_dict.pop(f"{prefix}linear_kv_down_proj.bias", None)

        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
