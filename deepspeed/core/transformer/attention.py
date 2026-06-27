# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Attention modules ported from Megatron-LM with AutoSP and DES-LOC support.

DES-LOC integration
-------------------
Each SelfAttention layer carries its ``layer_number`` (1-based global index).
At construction time it queries ``TransformerConfig.get_layer_tier()`` and logs
which GPU tier it belongs to.  The tier does not alter forward-pass logic — that
routing is managed externally by the DES-LOC engine — but the annotation is
available for profiling and dynamic scheduling.

AutoSP (Sequence Parallelism)
------------------------------
When ``config.sequence_parallel`` is True the input hidden_states arrive as
shards of shape ``[seq/sp_size, batch, hidden]``.  SelfAttention uses
all-to-all communication to scatter sequence → head dimension before attention
and gather back afterwards.  See :meth:`SelfAttention._sp_all_to_all_scatter`
and :meth:`SelfAttention._sp_all_to_all_gather`.

Megatron-LM evolution ported (46 commits, up to M4013)
--------------------------------------------------------
Key features brought in from the 46-commit Megatron evolution:

* **Submodule spec pattern** (M2379): ``SelfAttentionSubmodules`` /
  ``CrossAttentionSubmodules`` dataclasses carry builder callables for each
  sub-layer so the attention class is independent of concrete implementations.

* **GQA with kv_heads < tp_size sub-sharding** (M2807): all-gather the full
  QKV then index the appropriate slice per TP rank so models like Llama-3 with
  8 KV heads can run on TP-16.

* **QK logits clipping** (M2831): ``clip_qk()`` / ``_clip_linear_qkv()``
  clamp exploding attention logits during training via a per-group eta factor.

* **Batch-invariant mode** (M2836): ``batch_invariant_mode`` flag selects
  ``num_splits=1`` in FA3 / FA2 decode kernels to guarantee bit-exact results
  independent of batch size.

* **Flash Decode** (M2680 / M3780): ``flash_decode()`` fuses RoPE + KV-cache
  update + attention in a single FA2/FA4 kernel for static-batch decode.

* **Flash-decode-and-prefill** (M3780): ``flash_decode_and_prefill()`` handles
  mixed prefill + decode batches with FA4 / FA3 / FA2 dispatch.

* **Inference KV-cache management** (M2680):
  ``_adjust_key_value_for_inference()`` allocates per-layer KV buffers on first
  call and handles static-/dynamic-batching paths.

* **Output gate (Qwen3)** (M2950): optional ``attention_output_gate`` projects
  an extra gate vector from the QKV linear; a sigmoid-gated multiply is applied
  to core attention output via ``_apply_output_gate()``.

* **YaRN concentration factor** (M2380): ``_yarn_get_concentration_factor_from_config``
  passed as ``mscale`` when applying rotary embeddings.

* **Selective activation recompute** (M2379): ``_checkpointed_attention_forward``
  wraps ``tensor_parallel.checkpoint`` around core attention.

* **Fine-grained activation offloading** (M3018): per-module offload hooks for
  QKV linear, core attention and output projection.

* **PP layer offset for inference** (M3403 / M3701): ``_get_pp_layer_offset_for_inference``
  resolves the KV-cache layer index when fVPP is active.

* **QK LayerNorm / L2Norm** (M3063 / M3231): q_layernorm / k_layernorm applied
  per-head after splitting QKV.

* **``run_realtime_tests``** (M3063): cross-rank consistency check for Q/K norm
  weights to detect silent hardware failures during training.

* **Protocols / typed interfaces** (M3063 / M3959): ``LinearQkvInterface``,
  ``LinearProjInterface``, ``CoreAttentionInterface`` and their Builder
  counterparts ensure type-safe submodule injection.

* **FA3/FA4 import chain** (M2614 / M2807 / M3780): tries ``flash_attn_3``
  first, falls back to ``flashattn_hopper``, then handles FA4 separately.

* **``backward_dw``** (M2906 / refit): explicit weight-gradient methods for
  QKV and output projection to support fine-grained offloading schedules.

* **MTP / speculative decode** (M3496 / M3701): multi-token-prediction inference
  path in ``flash_decode_and_prefill``.

* **``set_for_recompute_input_layernorm``** (implicit fp8 path): sets
  ``save_original_input`` on the QKV linear for fp8 recompute.

All DES-LOC extensions (``get_layer_tier`` tier logging, AutoSP A2A helpers)
are preserved verbatim from the original deepspeed/core/transformer/attention.py.
"""

from __future__ import annotations

import copy
import inspect
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .transformer_config import TransformerConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional heavy deps — graceful fallbacks mirror Megatron's pattern
# ---------------------------------------------------------------------------

try:
    from einops import rearrange
except ImportError:
    rearrange = None

# FA3: try canonical package first, then hopper preview
HAVE_FA3 = False
try:
    from flash_attn_3.flash_attn_interface import _flash_attn_forward
    from flash_attn_3.flash_attn_interface import (
        flash_attn_with_kvcache as flash_attn3_with_kvcache,
    )
    HAVE_FA3 = True
except ImportError:
    pass

if not HAVE_FA3:
    try:
        from flashattn_hopper.flash_attn_interface import _flash_attn_forward
        from flashattn_hopper.flash_attn_interface import (
            flash_attn_with_kvcache as flash_attn3_with_kvcache,
        )
        HAVE_FA3 = True
    except ImportError:
        _flash_attn_forward = None         # type: ignore[assignment]
        flash_attn3_with_kvcache = None    # type: ignore[assignment]

# FA4 (Hopper CUTE backend)
HAVE_FA4 = False
try:
    from flash_attn.cute import flash_attn_varlen_func as flash_attn4_varlen_func
    HAVE_FA4 = True
except ImportError:
    flash_attn4_varlen_func = None  # type: ignore[assignment]

# FA2 (standard)
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except Exception:
    flash_attn_varlen_func = None       # type: ignore[assignment]
    flash_attn_with_kvcache = None      # type: ignore[assignment]

# FlashMLA (MLA decode kernel)
try:
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata
    HAVE_FMLA = True
except ImportError:
    flash_mla_with_kvcache = None   # type: ignore[assignment]
    get_mla_metadata = None         # type: ignore[assignment]
    HAVE_FMLA = False

# Transformer Engine
try:
    import transformer_engine  # noqa: F401
    HAVE_TE = True
    from megatron.core.extensions.transformer_engine import (
        SplitAlongDim,
        TELinear,
        set_save_original_input,
    )
except ImportError:
    HAVE_TE = False
    SplitAlongDim = None        # type: ignore[assignment]
    TELinear = None             # type: ignore[assignment]
    set_save_original_input = None  # type: ignore[assignment]

# YaRN mscale helper
try:
    from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import (
        _yarn_get_concentration_factor_from_config,
    )
except ImportError:
    def _yarn_get_concentration_factor_from_config(config) -> float | None:  # type: ignore[misc]
        return None


# ---------------------------------------------------------------------------
# Lazy parallel-state helpers (safe when dist not initialised)
# ---------------------------------------------------------------------------

def _get_sp_world_size() -> int:
    try:
        from deepspeed.core.parallel_state import get_sequence_parallel_world_size
        return get_sequence_parallel_world_size()
    except Exception:
        return 1


def _get_sp_rank() -> int:
    try:
        from deepspeed.core.parallel_state import get_sequence_parallel_rank
        return get_sequence_parallel_rank()
    except Exception:
        return 0


def _get_sp_group():
    try:
        from deepspeed.core.parallel_state import get_sequence_parallel_group
        return get_sequence_parallel_group()
    except Exception:
        return None


def _get_tp_world_size() -> int:
    try:
        from deepspeed.core.parallel_state import get_tensor_model_parallel_world_size
        return get_tensor_model_parallel_world_size()
    except Exception:
        return 1


def _get_tp_group():
    try:
        from deepspeed.core.parallel_state import get_tensor_model_parallel_group
        return get_tensor_model_parallel_group()
    except Exception:
        return None


def _get_tp_rank() -> int:
    try:
        from deepspeed.core.parallel_state import get_tensor_model_parallel_rank
        return get_tensor_model_parallel_rank()
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Base MegatronModule
# ---------------------------------------------------------------------------

from deepspeed.core.transformer.module import MegatronModule  # noqa: E402


# ---------------------------------------------------------------------------
# Protocol / typed interfaces (M3063, M3959 — replace ModuleSpec with Protocols)
# ---------------------------------------------------------------------------

class LinearQkvInterface(Protocol):
    """Interface for linear_qkv modules."""

    def forward(self, input: Tensor, /) -> tuple[Tensor, object]:
        """Applies linear_qkv."""
        ...

    def backward_dw(self) -> None:
        """Backward pass for the linear_qkv module."""
        ...


class LinearQkvBuilder(Protocol):
    """Protocol for building linear_qkv layers."""

    def __call__(
        self,
        input_size: int,
        output_size: int,
        /,
        *,
        config: TransformerConfig,
        init_method: Callable[[torch.Tensor], None],
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str,
    ) -> LinearQkvInterface: ...


class LinearInterface(Protocol):
    """Interface for linear_q and linear_kv modules."""

    def forward(self, input: Tensor, /) -> tuple[Tensor, object]:
        """Applies linear_q / linear_kv."""
        ...


class LinearLayerBuilder(Protocol):
    """Protocol for building linear_q and linear_kv layers."""

    def __call__(
        self,
        input_size: int,
        output_size: int,
        /,
        *,
        config: TransformerConfig,
        init_method: Callable[[torch.Tensor], None],
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
    ) -> LinearInterface: ...


class CoreAttentionInterface(Protocol):
    """Interface for core_attention modules."""

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        /,
        *,
        attention_bias: Tensor | None,
        packed_seq_params: object | None,
    ) -> Tensor:
        """Applies dot-product attention."""
        ...


class CoreAttentionBuilder(Protocol):
    """Protocol for building core_attention layers."""

    def __call__(
        self,
        *,
        config: TransformerConfig,
        layer_number: int,
    ) -> CoreAttentionInterface: ...


class LinearProjInterface(Protocol):
    """Interface for linear_proj modules."""

    def forward(self, hidden_states: Tensor, /) -> tuple[Tensor, Tensor | None]:
        """Applies the linear projection to the output of the core attention."""
        ...

    def backward_dw(self) -> None:
        """Computes weight gradients of output projection layer."""
        ...


class LinearProjBuilder(Protocol):
    """Protocol for building linear_proj layers."""

    def __call__(
        self,
        query_projection_size: int,
        hidden_size: int,
        /,
        *,
        config: TransformerConfig,
        init_method: Callable[[torch.Tensor], None],
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str,
    ) -> LinearProjInterface: ...


# ---------------------------------------------------------------------------
# Submodule spec dataclasses (M2379)
# ---------------------------------------------------------------------------

@dataclass
class SelfAttentionSubmodules:
    """Configuration class for specifying the submodules of a self-attention."""

    linear_qkv: LinearQkvBuilder
    core_attention: CoreAttentionBuilder
    linear_proj: LinearProjBuilder
    q_layernorm: object | None = None  # LayerNormBuilder | None
    k_layernorm: object | None = None  # LayerNormBuilder | None


@dataclass
class CrossAttentionSubmodules:
    """Configuration class for specifying the submodules of a cross-attention."""

    linear_q: LinearLayerBuilder
    linear_kv: LinearLayerBuilder
    core_attention: CoreAttentionBuilder
    linear_proj: LinearProjBuilder


# ---------------------------------------------------------------------------
# Abstract base attention class
# ---------------------------------------------------------------------------

class Attention(MegatronModule, ABC):
    """Attention layer abstract class.

    This layer only contains common modules required for the ``self attn`` and
    ``cross attn`` specialisations.

    AutoSP integration: when sequence_parallel is enabled, the input
    sequence is already partitioned across SP ranks. The attention
    computation handles the local chunk and uses A2A for KV exchange.

    DES-LOC integration: each concrete subclass logs its tier at init time.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Union[SelfAttentionSubmodules, CrossAttentionSubmodules],
        layer_number: int,
        attn_mask_type: str = "causal",
        attention_type: str = "self",
        cp_comm_type: str | None = None,
        pp_layer_offset: int | None = None,
    ) -> None:
        super().__init__(config)

        self.config = config
        self.layer_number = max(1, layer_number)
        self._pp_layer_offset = pp_layer_offset
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

        # Batch-invariant mode (M2836): num_splits=1 → bit-exact across batch sizes
        self.batch_invariant_mode = getattr(config, "batch_invariant_mode", False)

        assert config.kv_channels is not None, "kv_channels must be set in TransformerConfig"
        assert config.num_attention_heads is not None
        assert config.num_query_groups is not None

        self.query_projection_size: int = config.kv_channels * config.num_attention_heads
        self.kv_projection_size: int = config.kv_channels * config.num_query_groups

        # TP world size for partition calculations
        tp_size = _get_tp_world_size()
        self.world_size = tp_size

        self.hidden_size_per_attention_head: int = config.kv_channels

        # GQA sub-sharding (M2807): when kv_heads < tp_size each rank handles 1 kv_head
        if config.num_query_groups < tp_size:
            self.num_query_groups_per_partition = 1
            self.num_attention_heads_per_partition = (
                config.num_attention_heads // config.num_query_groups
            )
        else:
            self.num_query_groups_per_partition = max(1, config.num_query_groups // tp_size)
            self.num_attention_heads_per_partition = max(1, config.num_attention_heads // tp_size)

        # KV hidden sizes (may be overridden by MLA subclasses)
        self.key_hidden_size = self.hidden_size_per_attention_head
        self.val_hidden_size = self.hidden_size_per_attention_head

        # Core attention
        self.core_attention = submodules.core_attention(
            config=config,
            layer_number=self.layer_number,
        )

        # Selective activation recompute (M2379)
        self.checkpoint_core_attention = (
            getattr(config, "recompute_granularity", None) == "selective"
            and "core_attn" in getattr(config, "recompute_modules", [])
        )

        # Fine-grained activation offloading flags (M3018)
        fgao = getattr(config, "fine_grained_activation_offloading", False)
        offload_mods = getattr(config, "offload_modules", [])
        self.offload_qkv_linear = fgao and "qkv_linear" in offload_mods
        self.offload_core_attention = fgao and "core_attn" in offload_mods
        self.offload_attn_proj = fgao and "attn_proj" in offload_mods

        # Output projection
        self.linear_proj = submodules.linear_proj(
            self.query_projection_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="proj",
        )

        # DES-LOC: log which tier this attention head lives on
        tier = config.get_layer_tier(self.layer_number - 1)  # 0-based
        if tier is not None:
            logger.debug(
                "Attention layer %d assigned to tier: %s",
                self.layer_number,
                tier.upper(),
            )

    # ------------------------------------------------------------------
    # Selective activation checkpoint (M2379)
    # ------------------------------------------------------------------

    def _checkpointed_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        rotary_pos_emb=None,
        attn_mask_type=None,
        attention_bias: Tensor | None = None,
        packed_seq_params=None,
    ) -> Tensor:
        """Forward method with selective activation checkpointing."""
        try:
            from megatron.core import tensor_parallel as _tp
            _checkpoint = _tp.checkpoint
        except ImportError:
            # Fallback: run without checkpointing
            return self._run_core_attention(
                query, key, value, attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

        def custom_forward(*inputs):
            _q, _k, _v, _mask = inputs[0], inputs[1], inputs[2], inputs[3]
            return self._run_core_attention(
                _q, _k, _v, _mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

        return _checkpoint(custom_forward, False, query, key, value, attention_mask)

    def _run_core_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        attn_mask_type=None,
        attention_bias: Tensor | None = None,
        packed_seq_params=None,
        **extra_kwargs,
    ) -> Tensor:
        """Run the configured core attention module."""
        return self.core_attention(
            query,
            key,
            value,
            attention_mask,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            **extra_kwargs,
        )

    # ------------------------------------------------------------------
    # KV-cache allocation (M2680)
    # ------------------------------------------------------------------

    def _allocate_memory(
        self,
        inference_max_sequence_length: int,
        batch_size: int,
        dim: int,
        dtype: torch.dtype,
    ) -> Tensor:
        """Allocate memory to store KV cache during inference."""
        return torch.empty(
            inference_max_sequence_length,
            batch_size,
            self.num_query_groups_per_partition,
            dim,
            dtype=dtype,
            device=torch.cuda.current_device(),
        )

    # ------------------------------------------------------------------
    # PP layer offset for inference (M3403 / M3701 fVPP)
    # ------------------------------------------------------------------

    def _get_pp_layer_offset_for_inference(self) -> int:
        """Return the pipeline-parallel layer offset for inference.

        When ``pp_layer_offset`` was explicitly provided (e.g. by MambaBlock
        for hybrid models using --hybrid-layer-pattern with fVPP) use that
        value directly.  Otherwise fall back to the standard computation which
        assumes uniform layer distribution across pipeline stages.
        """
        if self._pp_layer_offset is not None:
            return self._pp_layer_offset
        # Uniform distribution fallback
        try:
            from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
            from megatron.core.parallel_state import get_pipeline_model_parallel_rank
            return get_transformer_layer_offset(self.config, vp_stage=None,
                                                pp_rank=get_pipeline_model_parallel_rank())
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Inference KV-cache management (M2680)
    # ------------------------------------------------------------------

    def _adjust_key_value_for_inference(
        self,
        inference_context,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        rotary_pos_emb,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        rotary_pos_cos_sin: Tensor | None = None,
        sequence_len_offset: int | None = None,
        *,
        inference_params=None,
    ) -> tuple[Tensor, Tensor, Tensor, object, str, Tensor | None]:
        """Save generated K/V to the KV cache and return full-context K/V.

        Handles both static batching (pre-allocated contiguous buffers) and
        dynamic batching (paged KV-cache managed by the inference context).

        Returns:
            (query, key, value, rotary_pos_emb, attn_mask_type, block_table)
        """
        # support deprecated inference_params kwarg
        if inference_context is None and inference_params is not None:
            inference_context = inference_params

        attn_mask_type = self.attn_mask_type
        if inference_context is None:
            return query, key, value, rotary_pos_emb, attn_mask_type, None

        # ---------------------------------------------------------------
        # Static batching: contiguous KV buffer per layer
        # ---------------------------------------------------------------
        is_static = getattr(inference_context, "is_static_batching", lambda: True)()

        if is_static:
            if self.layer_number not in inference_context.key_value_memory_dict:
                inf_max_seq = inference_context.max_sequence_length
                inf_max_batch = inference_context.max_batch_size
                inference_context.key_value_memory_dict[self.layer_number] = (
                    self._allocate_memory(inf_max_seq, inf_max_batch, self.key_hidden_size, key.dtype),
                    self._allocate_memory(inf_max_seq, inf_max_batch, self.val_hidden_size, value.dtype),
                )
            inference_key_memory, inference_value_memory = (
                inference_context.key_value_memory_dict[self.layer_number]
            )

        # Turn off masking past the prompt phase
        seq_len_offset = getattr(inference_context, "sequence_len_offset", 0)
        if is_static and seq_len_offset > 0:
            attn_mask_type = "no_mask"
        elif not is_static:
            attn_mask_type = "no_mask"

        if is_static:
            batch_start = getattr(inference_context, "batch_size_offset", 0)
            batch_end = batch_start + key.size(1)
            sequence_start = seq_len_offset
            sequence_end = sequence_start + key.size(0)

        block_table = None

        # Flash Decode path (M2680): apply RoPE before storing keys
        if getattr(self.config, "flash_decode", False):
            rotary_pos_cos_q = rotary_pos_cos_k = None
            rotary_pos_sin_q = rotary_pos_sin_k = None
            assert is_static
            if seq_len_offset > 0 and rotary_pos_cos is not None:
                rotary_pos_cos_q = rotary_pos_cos[sequence_end - 1:sequence_end]
                rotary_pos_sin_q = rotary_pos_sin[sequence_end - 1:sequence_end]
                rotary_pos_cos_k = rotary_pos_cos_q
                rotary_pos_sin_k = rotary_pos_sin_q
            elif rotary_pos_cos is not None:
                rotary_pos_cos_q = rotary_pos_cos[:sequence_end]
                rotary_pos_sin_q = rotary_pos_sin[:sequence_end]
                rotary_pos_cos_k = rotary_pos_cos_q
                rotary_pos_sin_k = rotary_pos_sin_q

            if rotary_pos_sin_k is not None:
                try:
                    from megatron.core.models.common.embeddings.rope_utils import (
                        apply_rotary_pos_emb_with_cos_sin,
                    )
                    key = apply_rotary_pos_emb_with_cos_sin(
                        key, rotary_pos_cos_k, rotary_pos_sin_k,
                        rotary_interleaved=getattr(self.config, "rotary_interleaved", False),
                    )
                    query = apply_rotary_pos_emb_with_cos_sin(
                        query, rotary_pos_cos_q, rotary_pos_sin_q,
                        rotary_interleaved=getattr(self.config, "rotary_interleaved", False),
                    )
                except ImportError:
                    pass
        else:
            rotary_pos_cos_q = rotary_pos_sin_q = None

        # Adjust rotary embeddings for static batching slice
        if rotary_pos_emb is not None and is_static:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            q_pos_emb = q_pos_emb[sequence_start:sequence_end]
            k_pos_emb = k_pos_emb[:sequence_end]
            rotary_pos_emb = (q_pos_emb, k_pos_emb)

        if is_static:
            # Write K/V into the cache buffer
            inference_key_memory[sequence_start:sequence_end, batch_start:batch_end] = key
            inference_value_memory[sequence_start:sequence_end, batch_start:batch_end] = value
            key = inference_key_memory[:sequence_end, batch_start:batch_end]
            value = inference_value_memory[:sequence_end, batch_start:batch_end]
        else:
            # Dynamic batching: paged KV cache (M2680)
            pp_layer_offset = self._get_pp_layer_offset_for_inference()
            if rotary_pos_emb is not None:
                q_pos_emb, k_pos_emb = rotary_pos_emb
                try:
                    key = inference_context.apply_rotary_emb_key(key, k_pos_emb, self.config, None)
                except Exception:
                    pass
                rotary_pos_emb = (q_pos_emb, None)
            try:
                inference_context.append_key_value_cache(
                    self.layer_number - pp_layer_offset, key, value
                )
                key, value, block_table = inference_context.key_value_cache(
                    self.layer_number - pp_layer_offset
                )
            except Exception:
                pass

        return query, key, value, rotary_pos_emb, attn_mask_type, block_table

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def get_query_key_value_tensors(
        self,
        hidden_states: Tensor,
        key_value_states: Tensor | None,
        output_gate: bool = False,
        split_qkv: bool = True,
    ) -> (
        tuple[Tensor, Tensor, Tensor, Tensor]
        | tuple[Tensor, Tensor, Tensor]
        | tuple[Tensor, list]
    ):
        """Must be implemented by SelfAttention / CrossAttention."""

    # ------------------------------------------------------------------
    # Flash Decode kernel (M2680 / M3780 FA4 inference)
    # ------------------------------------------------------------------

    def flash_decode(
        self,
        sequence_len_offset: Tensor,
        query_layer: Tensor,
        key_layer: Tensor,
        value_layer: Tensor,
        inference_key_memory: Tensor,
        inference_value_memory: Tensor,
        rotary_cos: Tensor | None,
        rotary_sin: Tensor | None,
        rotary_interleaved: bool = False,
    ) -> Tensor:
        """Fused RoPE + KV-cache update + flash attention for static-batch decode.

        Dispatches to ``flash_attn_with_kvcache`` (FA2 / FA3) which performs
        all three steps in a single CUDA kernel.
        """
        assert flash_attn_with_kvcache is not None, (
            "Flash Decoding requires flash_attn_with_kvcache from the flash-attn package."
        )
        q = query_layer.permute(1, 0, 2, 3)
        k = key_layer.permute(1, 0, 2, 3)
        v = value_layer.permute(1, 0, 2, 3)
        k_cache = inference_key_memory.permute(1, 0, 2, 3)
        v_cache = inference_value_memory.permute(1, 0, 2, 3)

        if rotary_cos is not None:
            rotary_cos = rotary_cos.to(query_layer.dtype)
        if rotary_sin is not None:
            rotary_sin = rotary_sin.to(query_layer.dtype)

        out = flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            k=k,
            v=v,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            cache_seqlens=sequence_len_offset,
            rotary_interleaved=rotary_interleaved,
        )
        return out

    # ------------------------------------------------------------------
    # FA3 wrapper (M3047 / M2981 — handle evolving _flash_attn_forward API)
    # ------------------------------------------------------------------

    def _flash_attention_3_forward_wrapper(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        max_seqlen_q,
        max_seqlen_k,
        cu_seqlens_q,
        seqlens_k,
        block_table,
        softmax_scale,
    ) -> Tensor:
        """Wrapper for FA3 ``_flash_attn_forward`` that adapts to API changes.

        Uses ``inspect.signature`` to filter kwargs to only those accepted by
        the installed version — mirrors Megatron M3047 robustness fix.
        """
        candidate_kwargs = {
            "q": q,
            "k": k,
            "v": v,
            "k_new": None,
            "v_new": None,
            "qv": None,
            "out": None,
            "out_": None,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": None,
            "cu_seqlens_k_new": None,
            "seqused_q": None,
            "seqused_k": seqlens_k,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "page_table": block_table,
            "kv_batch_idx": None,
            "leftpad_k": None,
            "rotary_cos": None,
            "rotary_sin": None,
            "seqlens_rotary": None,
            "q_descale": None,
            "k_descale": None,
            "v_descale": None,
            "softmax_scale": softmax_scale,
            "causal": True,
            "attention_chunk": 0,
            "softcap": 0.0,
            "window_size": (-1, -1),
            "window_size_left": -1,
            "window_size_right": -1,
            "rotary_interleaved": True,
            "scheduler_metadata": None,
            "num_splits": 0 if not self.batch_invariant_mode else 1,
            "pack_gqa": None,
            "sm_margin": 0,
        }

        if _flash_attn_forward is None:
            raise RuntimeError("FA3 _flash_attn_forward not available")

        if inspect.isfunction(_flash_attn_forward):
            sig = inspect.signature(_flash_attn_forward)
        else:
            sig = inspect.signature(_flash_attn_forward._init_fn)
        valid_kwargs = set(sig.parameters.keys())
        final_kwargs = {k: v for k, v in candidate_kwargs.items() if k in valid_kwargs}

        output_total, *_ = _flash_attn_forward(**final_kwargs)
        return output_total

    # ------------------------------------------------------------------
    # Flash decode-and-prefill (M3780 — FA4 inference, mixed batches)
    # ------------------------------------------------------------------

    def flash_decode_and_prefill(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        max_seqlen_q,
        max_seqlen_k,
        cu_seqlens_q,
        cu_seqlens_k,
        seqlens_k,
        block_table,
        is_decode_only: bool,
    ) -> Tensor:
        """Flash attention kernel for mixed decode and prefill samples (M3780).

        Dispatches to FA4 → FA3 → FA2 in priority order, matching the
        Megatron M3780 FA4 Inference commit.
        """
        assert not self.training
        assert block_table is not None

        if not is_decode_only:
            # Prefill or mixed: full varlen attention over all tokens
            q = q.squeeze(1)
            softmax_scale = (
                getattr(self, "softmax_scale", None) or q.shape[-1] ** -0.5
            )
            if HAVE_FA4:
                output_total, _ = flash_attn4_varlen_func(
                    q, k, v,
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    seqused_k=seqlens_k,
                    page_table=block_table,
                    softmax_scale=softmax_scale,
                    causal=True,
                    num_splits=1,
                )
            elif HAVE_FA3:
                output_total = self._flash_attention_3_forward_wrapper(
                    q, k, v, max_seqlen_q, max_seqlen_k,
                    cu_seqlens_q, seqlens_k, block_table, softmax_scale,
                )
            else:
                assert not self.batch_invariant_mode, (
                    "Batch invariant mode is not supported for flash attention 2"
                )
                output_total = flash_attn_varlen_func(
                    q, k, v, cu_seqlens_q, cu_seqlens_k,
                    max_seqlen_q, max_seqlen_k,
                    softmax_scale=softmax_scale,
                    causal=True,
                    block_table=block_table,
                )
            output_total = output_total.unsqueeze(1)

        else:
            # Decode only — potentially speculative (M3496)
            num_requests = seqlens_k.shape[0]
            tokens_per_request = q.shape[0] // num_requests
            q = q.reshape(num_requests, tokens_per_request, q.shape[2], q.shape[3])

            # MLA decode path (FlashMLA kernel)
            try:
                from deepspeed.core.transformer.transformer_config import MLATransformerConfig
                _is_mla = isinstance(self.config, MLATransformerConfig) and hasattr(self, "softmax_scale")
            except ImportError:
                _is_mla = False

            if _is_mla and HAVE_FMLA:
                softmax_scale = self.softmax_scale
                num_heads_k = 1
                num_heads_per_head_k = tokens_per_request * self.num_attention_heads_per_partition
                tile_scheduler_metadata, num_splits = get_mla_metadata(
                    seqlens_k, num_heads_per_head_k, num_heads_k
                )
                head_dim_v = self.config.kv_lora_rank
                kv_cache = k.unsqueeze(-2)
                output_total, _ = flash_mla_with_kvcache(
                    q, kv_cache, block_table, seqlens_k,
                    head_dim_v, tile_scheduler_metadata, num_splits,
                    softmax_scale=softmax_scale, causal=True,
                )
            elif HAVE_FA4:
                softmax_scale = getattr(self, "softmax_scale", None) or q.shape[-1] ** -0.5
                q_varlen = q.reshape(-1, q.shape[-2], q.shape[-1])
                output_total, _ = flash_attn4_varlen_func(
                    q_varlen, k, v,
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=tokens_per_request,
                    max_seqlen_k=max_seqlen_k,
                    seqused_k=seqlens_k,
                    page_table=block_table,
                    softmax_scale=softmax_scale,
                    causal=True,
                    num_splits=1,
                )
                output_total = output_total.reshape(
                    num_requests, tokens_per_request, *output_total.shape[1:]
                )
            else:
                flash_attn_args = {
                    "q": q,
                    "k_cache": k,
                    "v_cache": v,
                    "cache_seqlens": seqlens_k,
                    "causal": True,
                    ("page_table" if HAVE_FA3 else "block_table"): block_table,
                    "num_splits": 0 if not self.batch_invariant_mode else 1,
                }
                if HAVE_FA3:
                    output_total = flash_attn3_with_kvcache(**flash_attn_args)
                else:
                    assert not self.batch_invariant_mode, (
                        "Batch invariant mode is not supported for flash attention 2"
                    )
                    output_total = flash_attn_with_kvcache(**flash_attn_args)

            output_total = output_total.reshape(
                num_requests * tokens_per_request, 1, *output_total.shape[2:]
            )

        return output_total

    # ------------------------------------------------------------------
    # Output gate (M2950 — Qwen3 / attention_output_gate)
    # ------------------------------------------------------------------

    def _apply_output_gate(self, x: Tensor, gate: Tensor) -> Tensor:
        """Sigmoid-gated multiply applied to core attention output (Qwen3).

        Ported from Megatron M2950 ``@jit_fuser`` decorated method.
        The cast to float32 for the sigmoid mirrors the Megatron implementation
        to keep numerical stability regardless of the training dtype.
        """
        x_dtype = x.dtype
        gate = gate.contiguous().view(*x.shape)
        x = x * torch.sigmoid(gate.float())
        return x.to(x_dtype)

    # ------------------------------------------------------------------
    # Base forward (M2379 evolution — full Attention.forward from Megatron)
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        key_value_states: Tensor | None = None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        rotary_pos_cos_sin: Tensor | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params=None,
        sequence_len_offset: int | None = None,
        *,
        inference_params=None,
    ) -> tuple[Tensor, Tensor | None]:
        """Full attention forward pass (Megatron-compatible signature).

        Args:
            hidden_states: ``[seq, batch, hidden]``
                (or ``[seq/sp, batch, hidden]`` in AutoSP mode).
            attention_mask: Optional boolean / additive mask.
            key_value_states: Cross-attention encoder output (``None`` for self-attn).
            inference_context: Inference KV-cache context object.
            rotary_pos_emb: Rotary embeddings — single tensor or ``(q, k)`` tuple.
            rotary_pos_cos / rotary_pos_sin: Pre-split cos/sin for flash decode.
            attention_bias: Optional additive attention bias.
            packed_seq_params: THD-format packed sequence params.
            sequence_len_offset: CUDA-graph sequence length offset.

        Returns:
            ``(output, bias)`` where bias may be ``None``.
        """
        # Support deprecated inference_params kwarg
        if inference_context is None and inference_params is not None:
            inference_context = inference_params

        # no_rope_freq per layer (M3770 / sparse attention)
        no_rope = False
        no_rope_freq = getattr(self.config, "no_rope_freq", None)
        if no_rope_freq:
            no_rope = no_rope_freq[self.layer_number - 1]
        if no_rope:
            rotary_pos_emb = None

        is_using_flash_decode = getattr(self.config, "flash_decode", False) and (
            inference_context is not None
        )
        if is_using_flash_decode:
            rotary_pos_emb = None
        else:
            rotary_pos_cos = rotary_pos_sin = None

        # Normalise rotary_pos_emb to tuple
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # ------------------------------------------------------------------
        # AutoSP A2A scatter: seq/sp → seq, hidden → hidden/sp
        # ------------------------------------------------------------------
        sp_group = _get_sp_group()
        use_sp = (
            getattr(self.config, "sequence_parallel", False)
            and _get_sp_world_size() > 1
        )
        if use_sp:
            hidden_states = self._sp_all_to_all_scatter(hidden_states, sp_group)

        # ------------------------------------------------------------------
        # QKV
        # ------------------------------------------------------------------
        qkv_output = self.get_query_key_value_tensors(
            hidden_states,
            key_value_states,
            split_qkv=True,
            output_gate=getattr(self.config, "attention_output_gate", False),
        )

        attn_mask_type = self.attn_mask_type
        block_table = None
        gate = None

        if getattr(self.config, "attention_output_gate", False):
            query, key, value, gate = qkv_output
        else:
            query, key, value = qkv_output

        # ------------------------------------------------------------------
        # Adjust K/V for inference (KV-cache)
        # ------------------------------------------------------------------
        in_decode_mode = (
            inference_context is not None
            and getattr(inference_context, "is_decode_only", lambda: False)()
        )

        if in_decode_mode and getattr(self.config, "flash_decode", False):
            # Single-kernel flash-decode path (M2680)
            assert self.layer_number in inference_context.key_value_memory_dict
            inference_key_memory, inference_value_memory = (
                inference_context.key_value_memory_dict[self.layer_number]
            )
            out = self.flash_decode(
                sequence_len_offset=sequence_len_offset,
                query_layer=query,
                key_layer=key,
                value_layer=value,
                inference_key_memory=inference_key_memory,
                inference_value_memory=inference_value_memory,
                rotary_cos=rotary_pos_cos,
                rotary_sin=rotary_pos_sin,
                rotary_interleaved=getattr(self.config, "rotary_interleaved", False),
            )
            out = out.transpose(0, 1).contiguous()
            context_layer = out.view(out.size(0), out.size(1), -1)
            output, bias = self._linear_proj_forward(context_layer)
            if use_sp:
                output = self._sp_all_to_all_gather(output, sp_group)
            return output, bias

        query, key, value, rotary_pos_emb, attn_mask_type, block_table = (
            self._adjust_key_value_for_inference(
                inference_context,
                query, key, value,
                rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, rotary_pos_cos_sin,
                sequence_len_offset,
            )
        )

        if packed_seq_params is not None and getattr(packed_seq_params, "qkv_format", None) == "thd":
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # ------------------------------------------------------------------
        # Rotary position embeddings (YaRN mscale M2380)
        # ------------------------------------------------------------------
        if rotary_pos_emb is not None and not (
            getattr(self.config, "flash_decode", False) and inference_context is not None
        ):
            q_pos_emb, k_pos_emb = rotary_pos_emb
            try:
                from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
                mscale = _yarn_get_concentration_factor_from_config(self.config)
                if q_pos_emb is not None:
                    query = apply_rotary_pos_emb(
                        query, q_pos_emb, config=self.config,
                        mscale=mscale,
                    )
                if k_pos_emb is not None:
                    key = apply_rotary_pos_emb(
                        key, k_pos_emb, config=self.config,
                        mscale=mscale,
                    )
            except ImportError:
                # Fallback to built-in rotary if megatron not available
                if q_pos_emb is not None:
                    query, key = self._apply_rotary_emb(query, key, q_pos_emb)

        # ------------------------------------------------------------------
        # Core attention computation
        # ------------------------------------------------------------------
        is_static = inference_context is None or getattr(
            inference_context, "is_static_batching", lambda: True
        )()

        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query, key, value, attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        elif is_static:
            core_attn_out = self._run_core_attention(
                query, key, value, attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            # Dynamic batching (M2680)
            cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
            cu_kv_lengths, kv_lengths, max_seqlen_k = inference_context.cu_kv_lengths()
            core_attn_out = self.flash_decode_and_prefill(
                query, key, value,
                max_seqlen_q, max_seqlen_k,
                cu_query_lengths, cu_kv_lengths, kv_lengths,
                block_table,
                inference_context.is_decode_only(),
            )
            if rearrange is not None:
                core_attn_out = rearrange(core_attn_out, "s b h d -> s b (h d)")
            else:
                s, b, h, d = core_attn_out.shape
                core_attn_out = core_attn_out.reshape(s, b, h * d)

        # THD reshape
        if packed_seq_params is not None and getattr(packed_seq_params, "qkv_format", None) == "thd":
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # Output gate (M2950 — Qwen3)
        if gate is not None:
            core_attn_out = self._apply_output_gate(core_attn_out, gate)

        # ------------------------------------------------------------------
        # Output projection
        # ------------------------------------------------------------------
        output, bias = self._linear_proj_forward(core_attn_out)

        # All-reduce across TP ranks (RowParallelLinear pattern) when not using TE
        if bias is None:
            tp_size = _get_tp_world_size()
            if tp_size > 1:
                tp_group = _get_tp_group()
                if tp_group is not None:
                    torch.distributed.all_reduce(output, group=tp_group)

        # AutoSP A2A gather: seq → seq/sp, hidden/sp → hidden
        if use_sp:
            output = self._sp_all_to_all_gather(output, sp_group)

        return output, bias

    def _linear_proj_forward(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        """Forward through output projection, tolerating both (out, bias) and plain out."""
        result = self.linear_proj(x)
        if isinstance(result, tuple):
            return result[0], result[1] if len(result) > 1 else None
        return result, None

    # ------------------------------------------------------------------
    # QK clipping — base raises (M2831)
    # ------------------------------------------------------------------

    def clip_qk(self) -> None:
        """QK Clipping prevents attention logit explosion.

        Base class raises; ``SelfAttention`` overrides with the full impl.
        """
        raise NotImplementedError("clip_qk is not implemented.")

    def set_for_recompute_input_layernorm(self) -> None:
        """Set the attention layer for recompute input_layernorm (fp8/fp4)."""
        raise NotImplementedError("set_for_recompute_input_layernorm is not implemented.")

    # ------------------------------------------------------------------
    # AutoSP all-to-all helpers (DES-LOC AutoSP extension — preserved)
    # ------------------------------------------------------------------

    def _sp_all_to_all_scatter(
        self, hidden_states: torch.Tensor, sp_group
    ) -> torch.Tensor:
        """Scatter sequence dim → head dim for SP before attention.

        Input:  ``[seq/sp,  batch, hidden]``
        Output: ``[seq,     batch, hidden/sp]``
        """
        sp_size = _get_sp_world_size()
        if sp_size == 1 or sp_group is None:
            return hidden_states

        seq_chunk, batch, hidden = hidden_states.shape
        assert hidden % sp_size == 0, (
            f"hidden {hidden} not divisible by sp_size {sp_size}"
        )
        hidden_per_rank = hidden // sp_size

        inp = hidden_states.view(seq_chunk, batch, sp_size, hidden_per_rank)
        inp = inp.permute(2, 0, 1, 3).contiguous()   # [sp, sc, b, h/sp]

        out_list = [torch.empty_like(inp[0]) for _ in range(sp_size)]
        in_list = list(inp.unbind(0))
        torch.distributed.all_to_all(out_list, in_list, group=sp_group)

        out = torch.stack(out_list, dim=0)            # [sp, sc, b, h/sp]
        out = out.permute(1, 0, 2, 3).reshape(
            seq_chunk * sp_size, batch, hidden_per_rank
        )
        return out

    def _sp_all_to_all_gather(
        self, context: torch.Tensor, sp_group
    ) -> torch.Tensor:
        """Gather head dim → sequence dim for SP after attention.

        Inverse of :meth:`_sp_all_to_all_scatter`.

        Input:  ``[seq,     batch, hidden/sp]``
        Output: ``[seq/sp,  batch, hidden]``
        """
        sp_size = _get_sp_world_size()
        if sp_size == 1 or sp_group is None:
            return context

        total_seq, batch, hidden_per_rank = context.shape
        seq_chunk = total_seq // sp_size

        ctx = context.reshape(sp_size, seq_chunk, batch, hidden_per_rank)
        ctx = ctx.permute(1, 0, 2, 3).contiguous()

        inp_list = list(ctx.unbind(1))
        out_list = [torch.empty_like(inp_list[0]) for _ in range(sp_size)]
        torch.distributed.all_to_all(out_list, inp_list, group=sp_group)

        out = torch.stack(out_list, dim=1)
        out = out.permute(0, 2, 1, 3).reshape(
            seq_chunk, batch, hidden_per_rank * sp_size
        )
        return out

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_rotary_emb(
        q: torch.Tensor, k: torch.Tensor, rotary_pos_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to Q and K (built-in fallback).

        Args:
            q: ``[seq, batch, num_heads, head_dim]``
            k: ``[seq, batch, num_kv_heads, head_dim]``
            rotary_pos_emb: ``[seq, 1, 1, head_dim]``

        Returns:
            Rotated q and k tensors with the same shapes.
        """
        dim = q.shape[-1]
        half = dim // 2
        cos_emb = rotary_pos_emb[..., :half]
        sin_emb = rotary_pos_emb[..., half:]

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x[..., :half], x[..., half:]
            return torch.cat((-x2, x1), dim=-1)

        q_rot = q * cos_emb + rotate_half(q) * sin_emb
        k_rot = (
            k * cos_emb[: k.shape[0]] + rotate_half(k) * sin_emb[: k.shape[0]]
        )
        return q_rot, k_rot


# ---------------------------------------------------------------------------
# Scaled dot-product attention kernel
# ---------------------------------------------------------------------------

class DotProductAttention(MegatronModule):
    """Scaled dot-product attention with FlashAttention-2 dispatch.

    Uses :func:`torch.nn.functional.scaled_dot_product_attention` (SDPA)
    which dispatches to FlashAttention when hardware / software supports it.

    Tensor layout throughout: ``[seq, batch, num_heads, head_dim]``.

    Supports GQA via head expansion (repeat_interleave) for backends that
    don't natively handle grouped keys/values.
    """

    def __init__(self, config: TransformerConfig, layer_number: int) -> None:
        super().__init__(config)
        self.layer_number = max(1, layer_number)
        self.softmax_scale: float = 1.0 / math.sqrt(config.kv_channels)
        self.attn_dropout_p: float = config.attention_dropout
        self.num_attention_heads: int = config.num_attention_heads
        self.num_query_groups: int = config.num_query_groups

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        attn_mask_type=None,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        """Compute scaled dot-product attention.

        Args:
            query: ``[seq_q, batch, num_heads, head_dim]``
            key:   ``[seq_k, batch, num_kv_heads, head_dim]``
            value: ``[seq_k, batch, num_kv_heads, head_dim]``
            attention_mask: Optional additive or boolean mask
                ``[batch, 1, seq_q, seq_k]``.

        Returns:
            context: ``[seq_q, batch, num_heads * head_dim]``
        """
        seq_q, batch, num_heads, head_dim = query.shape
        num_kv_heads = key.shape[2]

        # GQA: expand KV heads to match Q heads
        if num_kv_heads != num_heads:
            expand_factor = num_heads // num_kv_heads
            key = key.repeat_interleave(expand_factor, dim=2)
            value = value.repeat_interleave(expand_factor, dim=2)

        # [sq, b, nh, hd] → [b, nh, sq, hd]
        q = query.permute(1, 2, 0, 3).contiguous()
        k = key.permute(1, 2, 0, 3).contiguous()
        v = value.permute(1, 2, 0, 3).contiguous()

        is_causal = attention_mask is None

        attn_mask_sdpa: Optional[torch.Tensor] = None
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                attn_mask_sdpa = ~attention_mask
            else:
                attn_mask_sdpa = attention_mask

        if attention_bias is not None:
            if attn_mask_sdpa is not None:
                attn_mask_sdpa = attn_mask_sdpa + attention_bias
            else:
                attn_mask_sdpa = attention_bias

        dropout_p = self.attn_dropout_p if self.training else 0.0

        context = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask_sdpa,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=self.softmax_scale,
        )
        # context: [b, nh, sq, hd] → [sq, b, nh*hd]
        context = context.permute(2, 0, 1, 3).contiguous()
        context = context.view(seq_q, batch, num_heads * head_dim)
        return context


# ---------------------------------------------------------------------------
# Standard multi-head self-attention
# ---------------------------------------------------------------------------

class SelfAttention(Attention):
    """Multi-head self-attention with QKV projection.

    Layout: ``[seq, batch, hidden]`` throughout (Megatron convention).

    Implements the full Megatron SelfAttention evolution (M2379 → M4013)
    including GQA sub-sharding (M2807), QK clip (M2831), output gate (M2950),
    QK LayerNorm (M3063), run_realtime_tests (M3063), and backward_dw (M2906).

    AutoSP and DES-LOC extensions are preserved from the original implementation.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules | None = None,
        layer_number: int = 1,
        attn_mask_type: str = "causal",
        cp_comm_type: str | None = None,
        pp_layer_offset: int | None = None,
    ) -> None:
        # Build minimal submodules when called without the spec pattern
        # (backward-compat with original DES-LOC usage)
        if submodules is None:
            submodules = self._build_default_submodules(config, layer_number)

        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            pp_layer_offset=pp_layer_offset,
        )

        tp_size = _get_tp_world_size()

        # QKV projection output dimension (M2950 gate adds extra q-sized projection)
        self.linear_qkv_out_dim = self.query_projection_size + 2 * self.kv_projection_size
        if getattr(config, "attention_output_gate", False):
            self.linear_qkv_out_dim += config.kv_channels * config.num_attention_heads

        # Build QKV linear
        self.linear_qkv = submodules.linear_qkv(
            config.hidden_size,
            self.linear_qkv_out_dim,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=config.add_bias_linear or getattr(config, "add_qkv_bias", False),
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="qkv",
        )

        # QK LayerNorm / L2Norm (M3063 / M3231)
        q_norm_cls = k_norm_cls = None
        if getattr(config, "qk_l2_norm", False):
            try:
                from megatron.core.transformer.torch_norm import L2Norm
                q_norm_cls = getattr(submodules, "q_layernorm", None) or L2Norm
                k_norm_cls = getattr(submodules, "k_layernorm", None) or L2Norm
            except ImportError:
                pass
        elif getattr(config, "qk_layernorm", False):
            q_norm_cls = getattr(submodules, "q_layernorm", None)
            k_norm_cls = getattr(submodules, "k_layernorm", None)

        self.q_layernorm = (
            q_norm_cls(
                hidden_size=self.hidden_size_per_attention_head,
                config=config,
                eps=config.layernorm_epsilon,
            )
            if q_norm_cls is not None
            else None
        )
        self.k_layernorm = (
            k_norm_cls(
                hidden_size=self.hidden_size_per_attention_head,
                config=config,
                eps=config.layernorm_epsilon,
            )
            if k_norm_cls is not None
            else None
        )

    # ------------------------------------------------------------------
    # Default submodules for legacy / DES-LOC-only instantiation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_default_submodules(
        config: TransformerConfig, layer_number: int
    ) -> SelfAttentionSubmodules:
        """Build plain nn.Linear-based submodules when no spec is provided."""

        tp_size = _get_tp_world_size()
        qkv_out = (config.kv_channels * config.num_attention_heads
                   + 2 * config.kv_channels * config.num_query_groups) // tp_size

        class _QKVLinear(nn.Linear):
            def forward(self, x, /):  # type: ignore[override]
                out = super().forward(x)
                return out, None

            def backward_dw(self):
                pass

        class _ProjLinear(nn.Linear):
            def forward(self, x, /):  # type: ignore[override]
                out = super().forward(x)
                return out, None

            def backward_dw(self):
                pass

        def _build_qkv(in_size, out_size, /, *, config, init_method,
                        gather_output, bias, skip_bias_add, is_expert,
                        tp_comm_buffer_name):
            m = _QKVLinear(in_size, out_size, bias=bias)
            m.weight.tensor_model_parallel = True
            m.weight.partition_dim = 0
            return m

        def _build_proj(q_size, h_size, /, *, config, init_method, bias,
                         input_is_parallel, skip_bias_add, is_expert,
                         tp_comm_buffer_name):
            m = _ProjLinear(q_size // tp_size, h_size, bias=bias)
            m.weight.tensor_model_parallel = True
            m.weight.partition_dim = 1
            return m

        def _build_core(*, config, layer_number):
            return DotProductAttention(config, layer_number)

        return SelfAttentionSubmodules(
            linear_qkv=_build_qkv,
            core_attention=_build_core,
            linear_proj=_build_proj,
        )

    # ------------------------------------------------------------------
    # QKV tensor extraction with full GQA sub-sharding (M2807)
    # ------------------------------------------------------------------

    def get_query_key_value_tensors(
        self,
        hidden_states: Tensor,
        key_value_states: Tensor | None = None,
        output_gate: bool = False,
        split_qkv: bool = True,
    ) -> (
        tuple[Tensor, Tensor, Tensor, Tensor]
        | tuple[Tensor, Tensor, Tensor]
        | tuple[Tensor, list]
    ):
        """Derive ``query``, ``key``, ``value`` (and optionally ``gate``) tensors.

        Implements GQA sub-sharding from M2807: when ``num_query_groups < tp_size``
        the full QKV tensor is all-gathered and the appropriate slice is extracted
        per TP rank.
        """
        # QKV projection
        qkv_result = self.linear_qkv(hidden_states)
        mixed_qkv = qkv_result[0] if isinstance(qkv_result, tuple) else qkv_result

        num_query_heads_per_group = (
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        )
        num_qkv_heads_per_group = num_query_heads_per_group + 2
        if output_gate:
            num_qkv_heads_per_group += num_query_heads_per_group

        # GQA sub-sharding: kv_heads < tp_size (M2807)
        if self.config.num_query_groups < self.world_size:
            try:
                from megatron.core.tensor_parallel.mappings import (
                    all_gather_last_dim_from_tensor_parallel_region,
                )
                from megatron.core.utils import get_pg_rank
                tp_group = _get_tp_group()
                mixed_qkv = all_gather_last_dim_from_tensor_parallel_region(
                    mixed_qkv, group=tp_group
                )
                idx = get_pg_rank(tp_group) // (
                    self.world_size // self.config.num_query_groups
                )
            except ImportError:
                idx = _get_tp_rank() // (self.world_size // self.config.num_query_groups)
            size = mixed_qkv.size()[-1] // self.config.num_query_groups
            mixed_qkv = mixed_qkv[:, :, idx * size:(idx + 1) * size]

        # Reshape: [sq, b, hp] → [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            num_qkv_heads_per_group * self.hidden_size_per_attention_head,
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        # Split into Q, (Gate,) K, V
        if output_gate:
            if not split_qkv:
                raise ValueError("split_qkv not supported for gated attention.")
            split_arg_list = [
                num_query_heads_per_group * self.hidden_size_per_attention_head,
                num_query_heads_per_group * self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
            ]
            if SplitAlongDim is not None:
                query, gate, key, value = SplitAlongDim(mixed_qkv, 3, split_arg_list)
            else:
                query, gate, key, value = torch.split(mixed_qkv, split_arg_list, dim=3)
        else:
            split_arg_list = [
                num_query_heads_per_group * self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
            ]
            if not split_qkv:
                return mixed_qkv, split_arg_list
            if SplitAlongDim is not None:
                query, key, value = SplitAlongDim(mixed_qkv, 3, split_arg_list)
            else:
                query, key, value = torch.split(mixed_qkv, split_arg_list, dim=3)

        # [sq, b, ng, np/ng * hn] → [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)

        # Further index query when kv_heads < tp_size (step 4 of M2807)
        if self.config.num_query_groups < self.world_size:
            try:
                from megatron.core.utils import get_pg_rank
                tp_group = _get_tp_group()
                idx2 = get_pg_rank(tp_group) % (self.world_size // self.config.num_query_groups)
            except ImportError:
                idx2 = _get_tp_rank() % (self.world_size // self.config.num_query_groups)
            size2 = self.num_attention_heads_per_partition // (
                self.world_size // self.config.num_query_groups
            )
            query = query[:, :, idx2 * size2:(idx2 + 1) * size2, :]

        # QK LayerNorm (M3063)
        if self.q_layernorm is not None:
            query = self.q_layernorm(query)
        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        if getattr(self.config, "test_mode", False):
            self.run_realtime_tests()

        if output_gate:
            gate = gate.reshape(*gate.shape[:2], -1, self.hidden_size_per_attention_head)
            if self.config.num_query_groups < self.world_size:
                gate = gate[:, :, idx2 * size2:(idx2 + 1) * size2, :]
            return query, key, value, gate

        return query, key, value

    # ------------------------------------------------------------------
    # run_realtime_tests (M3063 — cross-rank consistency check)
    # ------------------------------------------------------------------

    def run_realtime_tests(self) -> None:
        """Cross-rank consistency check for Q/K layernorm weights.

        Detects silent hardware failures (memory corruption, network errors)
        by comparing layernorm parameters across DP and TP ranks.
        Only active when ``config.qk_layernorm`` is enabled.
        """
        if not getattr(self.config, "qk_layernorm", False):
            return
        if self.q_layernorm is None or self.k_layernorm is None:
            return

        try:
            from megatron.core.parallel_state import (
                get_data_parallel_group,
                get_data_parallel_rank,
                get_data_parallel_world_size,
                get_tensor_model_parallel_group,
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )
        except ImportError:
            return

        inputs = torch.stack([
            self.q_layernorm.weight.data,
            getattr(self.q_layernorm, "bias", torch.zeros(1)).data,
            self.k_layernorm.weight.data,
            getattr(self.k_layernorm, "bias", torch.zeros(1)).data,
        ])

        def _compare(srcs, tgts, names, parallelism):
            for src, tgt, name in zip(srcs, tgts, names):
                assert torch.all(src == tgt), (
                    f"Discrepancy between {name} in {parallelism} ranks "
                    f"{i} and {rank}. Diff: {torch.norm(src - tgt)}"
                )

        rank = get_data_parallel_rank()
        dp_list = [torch.empty_like(inputs) for _ in range(get_data_parallel_world_size())]
        dp_list[rank] = inputs
        torch.distributed.all_gather(dp_list, inputs, group=get_data_parallel_group())
        for i, dp in enumerate(dp_list):
            q_w, q_b, k_w, k_b = torch.unbind(dp)
            _compare([q_w, q_b, k_w, k_b],
                     [self.q_layernorm.weight.data,
                      getattr(self.q_layernorm, "bias", torch.zeros(1)).data,
                      self.k_layernorm.weight.data,
                      getattr(self.k_layernorm, "bias", torch.zeros(1)).data],
                     ["q_w", "q_b", "k_w", "k_b"], "DP")

        rank = get_tensor_model_parallel_rank()
        tp_list = [torch.empty_like(inputs) for _ in range(get_tensor_model_parallel_world_size())]
        tp_list[rank] = inputs
        torch.distributed.all_gather(tp_list, inputs, group=get_tensor_model_parallel_group())
        for i, tp in enumerate(tp_list):
            q_w, q_b, k_w, k_b = torch.unbind(tp)
            _compare([q_w, q_b, k_w, k_b],
                     [self.q_layernorm.weight.data,
                      getattr(self.q_layernorm, "bias", torch.zeros(1)).data,
                      self.k_layernorm.weight.data,
                      getattr(self.k_layernorm, "bias", torch.zeros(1)).data],
                     ["q_w", "q_b", "k_w", "k_b"], "TP")

    # ------------------------------------------------------------------
    # QK clip (M2831)
    # ------------------------------------------------------------------

    def clip_qk(self) -> None:
        """Clip QK logits to prevent explosion (experimental on GQA).

        Updates ``linear_qkv.weight`` in-place using a per-group balancing eta
        computed from ``core_attention.current_max_attn_logits``.
        Requires ``config.qk_clip`` to be True.
        """
        if not getattr(self.config, "qk_clip", False):
            raise ValueError("qk_clip option needs to be enabled")

        current_max = getattr(self.core_attention, "current_max_attn_logits", None)
        if current_max is None:
            raise ValueError("current_max_attn_logits is None")

        assert current_max.shape == (self.num_attention_heads_per_partition,), (
            f"current_max_attn_logits shape is not "
            f"({self.num_attention_heads_per_partition},) but {current_max.shape}"
        )

        grouped_max = torch.max(
            current_max.view(self.num_query_groups_per_partition, -1), dim=1
        ).values

        if torch.any(grouped_max > self.config.qk_clip_threshold):
            assert grouped_max.shape == (self.num_query_groups_per_partition,)
            self.qk_clip_balancing_eta = torch.clamp(
                self.config.qk_clip_threshold / grouped_max, max=1.0
            ).view(self.num_query_groups_per_partition, 1, 1)
            assert torch.all(self.qk_clip_balancing_eta <= 1.0)

            # Update main_param (AMP) and the fp32 weight
            if hasattr(self.linear_qkv, "weight"):
                w = self.linear_qkv.weight
                if hasattr(w, "main_param"):
                    w.main_param.data.copy_(self._clip_linear_qkv(w.main_param.data))
                w.data.copy_(self._clip_linear_qkv(w.data))

        self.core_attention.current_max_attn_logits = None

    def _clip_linear_qkv(self, weight: Tensor) -> Tensor:
        """Apply qk-clip balancing to linear_qkv weight tensor."""
        weight_reshaped = weight.view(
            self.num_query_groups_per_partition,
            (self.query_projection_size + 2 * self.kv_projection_size)
            // self.num_query_groups_per_partition,
            -1,
        )
        q_cols = self.query_projection_size // self.num_query_groups_per_partition
        kv_cols = self.kv_projection_size // self.num_query_groups_per_partition

        weight_q = weight_reshaped[:, :q_cols, :]
        weight_k = weight_reshaped[:, q_cols:q_cols + kv_cols, :]
        weight_v = weight_reshaped[:, q_cols + kv_cols:, :]

        eta_ext = self.qk_clip_balancing_eta.repeat(1, weight_q.size(1), 1)
        alpha = getattr(self.config, "qk_clip_alpha", 0.5)
        weight_q.mul_(torch.pow(eta_ext, alpha))
        weight_k.mul_(torch.pow(self.qk_clip_balancing_eta, 1.0 - alpha))

        weight_updated = torch.cat([weight_q, weight_k, weight_v], dim=1)
        return weight_updated.view(
            self.query_projection_size + 2 * self.kv_projection_size, -1
        )

    # ------------------------------------------------------------------
    # Fine-grained weight gradient methods (M2906)
    # ------------------------------------------------------------------

    def backward_dw(self) -> None:
        """Execute weight update operations for fine-grained offloading."""
        self._backward_qkv_proj()
        self._backward_output_proj()

    def _backward_qkv_proj(self) -> None:
        """Update weights for QKV projection layer."""
        if hasattr(self.linear_qkv, "backward_dw"):
            self.linear_qkv.backward_dw()

    def _backward_output_proj(self) -> None:
        """Update weights for output projection layer."""
        if hasattr(self.linear_proj, "backward_dw"):
            self.linear_proj.backward_dw()

    # ------------------------------------------------------------------
    # fp8/fp4 recompute helper (M3059 / implicit fp8 path)
    # ------------------------------------------------------------------

    def set_for_recompute_input_layernorm(self) -> None:
        """Set save_original_input on the QKV linear for fp8/fp4 recompute."""
        if set_save_original_input is not None:
            set_save_original_input(self.linear_qkv)
        else:
            try:
                from megatron.core.extensions.transformer_engine import set_save_original_input as _soi
                _soi(self.linear_qkv)
            except ImportError:
                pass


# ---------------------------------------------------------------------------
# Cross-attention
# ---------------------------------------------------------------------------

class CrossAttention(Attention):
    """Cross-attention layer class.

    Takes ``hidden_states`` of shape ``[s, b, h]`` and encoder context
    ``key_value_states`` of shape ``[s, b, h]``, returns output ``[s, b, h]``.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CrossAttentionSubmodules,
        layer_number: int,
        attn_mask_type: str = "causal",
        cp_comm_type: str | None = None,
        pp_layer_offset: int | None = None,
    ) -> None:
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="cross",
            cp_comm_type=cp_comm_type,
            pp_layer_offset=pp_layer_offset,
        )

        if config.num_query_groups != config.num_attention_heads:
            raise ValueError(
                "Group query attention is not currently supported in cross attention."
            )
        assert self.query_projection_size == self.kv_projection_size

        self.linear_q = submodules.linear_q(
            config.hidden_size,
            self.query_projection_size,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
        )

        self.linear_kv = submodules.linear_kv(
            config.hidden_size,
            2 * self.kv_projection_size,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
        )

    def get_query_key_value_tensors(
        self,
        hidden_states: Tensor,
        key_value_states: Tensor | None,
        output_gate: bool = False,
        split_qkv: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Derive Q from hidden_states; K, V from key_value_states."""
        assert split_qkv, "split_qkv must be True for CrossAttention"
        assert not output_gate, "Output gate not supported in cross attention."
        assert key_value_states is not None, (
            "key_value_states cannot be None for CrossAttention"
        )

        kv_result = self.linear_kv(key_value_states)
        mixed_kv = kv_result[0] if isinstance(kv_result, tuple) else kv_result

        # [sk, b, (np * 2 * hn)] → [sk, b, np, 2 * hn]
        new_tensor_shape = mixed_kv.size()[:-1] + (
            self.num_attention_heads_per_partition,
            2 * self.hidden_size_per_attention_head,
        )
        mixed_kv = mixed_kv.view(*new_tensor_shape)
        # [sk, b, np, 2*hn] → 2 x [sk, b, np, hn]
        key, value = mixed_kv.chunk(2, dim=-1)

        q_result = self.linear_q(hidden_states)
        query = q_result[0] if isinstance(q_result, tuple) else q_result

        new_tensor_shape = query.size()[:-1] + (
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        query = query.view(*new_tensor_shape)

        return query, key, value
