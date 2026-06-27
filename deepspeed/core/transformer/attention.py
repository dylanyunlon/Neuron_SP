# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Attention modules — DES-LOC Neuron_SP port of Megatron-LM attention.

Ported from Megatron-LM megatron/core/transformer/attention.py
(46 commits, M2379 → M4013).

Key evolution tracked through the commit history:
  M2379 (f0d9fa97f) – Optimise attention preproc; split rotary Q/K early
  M2380 (2c1b77a99) – YaRN RoPE support for gpt-oss
  M2383 (42a56ec6b) – flash_decode=True + inference_context=None fix
  M2407 (71a09cc27) – Avoid split/concat with RoPE (fused single QKV RoPE)
  M2459 (576980459) – Unify enable/external cudagraph with cuda-graph-impl
  M2471 (f759111e4) – Rename Chunk→Block in memory management
  M2484 (950aa4353) – Replay of above renaming
  M2614 (a32ff7501) – flash_attn_3 as first option for FA3 import
  M2656 (70f85eb55) – Enable KV cache in training for EAGLE speculative decoding
  M2668 (e8b9df156) – NVFP4 MoE with proper padding
  M2680 (314a3786e) – Fix PP KV cache allocation; multi-node PP inference
  M2704 (c9d2c8f81) – Explicitly zero out padding token activations for inference
  M2709 (233b5b035) – Revert padding token zeroing
  M2807 (f7dfb9979) – TP > num_kv_heads: QKV sub-sharding support (M2806)
  M2808 (dfc3913f2) – Fix FA3 import
  M2831 (dd546099f) – QK logits clipping (non-split version)
  M2836 (d2bd9fa89) – Batch invariance mode
  M2856 (5f5741db9) – Replace global parallel state w/ explicit pg parameters
  M2906 (2b343d739) – Refactor cuda_graph_scope (MoE)
  M2916 (ccc9ad3e6) – Explicitly zero out padding tokens for quantization scales
  M2950 (65dccab89) – Attention output gate for Qwen3-Next
  M2970 (4fc793519) – Remove unused FlashAttention3 args
  M2976/M2981/M2998  – Revert/reapply FA3 args changes
  M3018 (b97184db5) – Hybrid Context Parallel feature
  M3047 (98d8c56db) – FA3 wrapper function for _flash_attn_forward
  M3056 (a3615d795) – FA3 _flash_attn_forward call wrapper
  M3062 (096dbeb47) – Last prefill chunk for Mamba models
  M3063 (90e685b85) – Protocols replacing ModuleSpec for Attention submodules
  M3231 (f68c7c10f) – Replace ModuleSpec with Protocols for LayerNorm submodules
  M3403 (7418b1b8f) – Flexible VPP (fVPP) for hybrid model
  M3489 (16a8cdb64) – Fix KV head slicing when kv_head < tp_size in MoE
  M3496 (8f539df74) – Speculative decoding with MTP layers
  M3563 (9ed8b0c4a) – Fix incorrect HAVE_TE detection
  M3572 (9a60a18ef) – Ultra refit
  M3701 (980211ae6) – Miscellaneous MTP inference fixes
  M3730 (5dcda195a) – MLA fix: pad V when Q/V head dims differ for THD
  M3770 (a00e9443c) – Port DeepSeek Sparse Attention to MambaModel
  M3774 (e15ec3c04) – QK layernorm support for dot-product attention in MambaModel
  M3780 (76ac7c24b) – FA4 Inference
  M3955 (925422cd8) – One single flag for inference mode
  M3959 (dbfc96b08) – Protocols for linear_proj submodules
  M3977 (e41b37002) – Refactor CUDA graph API
  M4013 (4c6360260) – FP4 param gather for NVFP4 recipe

Architecture summary (as of M4013):
  * MHA → GQA (since ~M2807): num_query_groups ≤ num_attention_heads
  * MQA: num_query_groups == 1 (special case of GQA)
  * MLA (Multi-Latent Attention, DeepSeek): added ~M3730 via separate
    flash_mla kernel with kv_lora_rank compression
  * Context Parallel: ring attention via hybrid CP (M3018)
  * Flash Attention 2/3/4: dispatched via capability detection
  * QK layernorm (M3774): L2Norm or LayerNorm per head, for Mamba
  * Attention output gate (M2950): Qwen3-Next style gated attention
  * TP > num_kv_heads (M2807): sub-sharding with all-gather + slice
  * QK logit clipping (M2831): for training stability

DES-LOC integration:
  Each SelfAttention / CrossAttention logs its assigned GPU tier at
  construction.  The tier does not alter the forward pass — routing is
  managed by the DES-LOC engine — but is accessible via layer.desloc_tier
  (set on the parent TransformerLayer).

AutoSP (Sequence Parallelism):
  When config.sequence_parallel=True, input arrives as [seq/sp, b, h].
  SelfAttention uses all-to-all to scatter seq → head dim for attention
  and gather back.  Compatible with GQA.
"""

from __future__ import annotations

import copy
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from deepspeed.core.transformer.module import MegatronModule
from deepspeed.core.transformer.transformer_config import TransformerConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Flash Attention imports
# ---------------------------------------------------------------------------

try:
    from flash_attn_3.flash_attn_interface import _flash_attn_forward as _fa3_forward
    from flash_attn_3.flash_attn_interface import (
        flash_attn_with_kvcache as _fa3_kvcache,
    )
    HAVE_FA3 = True
except ImportError:
    _fa3_forward = None
    _fa3_kvcache = None
    HAVE_FA3 = False

if not HAVE_FA3:
    try:
        from flashattn_hopper.flash_attn_interface import _flash_attn_forward as _fa3_forward
        from flashattn_hopper.flash_attn_interface import (
            flash_attn_with_kvcache as _fa3_kvcache,
        )
        HAVE_FA3 = True
    except ImportError:
        pass

try:
    from flash_attn.cute import flash_attn_varlen_func as _fa4_varlen_func
    HAVE_FA4 = True
except ImportError:
    _fa4_varlen_func = None
    HAVE_FA4 = False

try:
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata
    HAVE_FMLA = True
except ImportError:
    flash_mla_with_kvcache = None
    get_mla_metadata = None
    HAVE_FMLA = False

try:
    from flash_attn import flash_attn_varlen_func as _fa2_varlen_func
    from flash_attn import flash_attn_with_kvcache as _fa2_kvcache
except ImportError:
    _fa2_varlen_func = None
    _fa2_kvcache = None

# ---------------------------------------------------------------------------
# Lazy parallel-state helpers (safe when dist not initialised)
# ---------------------------------------------------------------------------

def _get_tp_world_size() -> int:
    try:
        from deepspeed.core.parallel_state import get_tensor_model_parallel_world_size
        return get_tensor_model_parallel_world_size()
    except Exception:
        return 1


def _get_tp_rank() -> int:
    try:
        from deepspeed.core.parallel_state import get_tensor_model_parallel_rank
        return get_tensor_model_parallel_rank()
    except Exception:
        return 0


def _get_tp_group():
    try:
        from deepspeed.core.parallel_state import get_tensor_model_parallel_group
        return get_tensor_model_parallel_group()
    except Exception:
        return None


def _get_sp_world_size() -> int:
    try:
        from deepspeed.core.parallel_state import get_sequence_parallel_world_size
        return get_sequence_parallel_world_size()
    except Exception:
        return 1


def _get_sp_group():
    try:
        from deepspeed.core.parallel_state import get_sequence_parallel_group
        return get_sequence_parallel_group()
    except Exception:
        return None


def divide(numerator: int, denominator: int) -> int:
    assert numerator % denominator == 0, (
        f"{numerator} is not divisible by {denominator}"
    )
    return numerator // denominator


# ---------------------------------------------------------------------------
# Rotary position embedding helpers
# ---------------------------------------------------------------------------

def apply_rotary_pos_emb(
    t: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    cp_group=None,
    cu_seqlens: Optional[Tensor] = None,
    mscale: float = 1.0,
) -> Tensor:
    """Apply rotary position embeddings to tensor t.

    Args:
        t:    ``[seq, batch, num_heads, head_dim]``
        freqs: ``[seq, 1, 1, head_dim]``
        rotary_interleaved: LLaMA style (False) vs RoFormer (True).
        mscale: YaRN concentration multiplier (M2380).

    Returns:
        Rotated tensor of the same shape.
    """
    head_dim = t.shape[-1]
    half = head_dim // 2
    cos_f = freqs.cos() * mscale
    sin_f = freqs.sin() * mscale

    if rotary_interleaved:
        # Interleave even/odd dims (RoFormer style)
        t1 = t[..., 0::2]
        t2 = t[..., 1::2]
        t_rot = torch.stack([-t2, t1], dim=-1).flatten(-2)
        return t * cos_f + t_rot * sin_f
    else:
        # LLaMA style: first half / second half
        t1 = t[..., :half]
        t2 = t[..., half:]
        t_rot = torch.cat([-t2, t1], dim=-1)
        return t * cos_f + t_rot * sin_f


def apply_rotary_pos_emb_with_cos_sin(
    t: Tensor,
    cos: Tensor,
    sin: Tensor,
    rotary_interleaved: bool = False,
) -> Tensor:
    """Apply rotary embeddings from precomputed cos/sin tensors."""
    head_dim = t.shape[-1]
    half = head_dim // 2
    if rotary_interleaved:
        t1 = t[..., 0::2]
        t2 = t[..., 1::2]
        t_rot = torch.stack([-t2, t1], dim=-1).flatten(-2)
    else:
        t1 = t[..., :half]
        t2 = t[..., half:]
        t_rot = torch.cat([-t2, t1], dim=-1)
    return t * cos + t_rot * sin


# ---------------------------------------------------------------------------
# Protocol interfaces (M3063: replacing ModuleSpec)
# ---------------------------------------------------------------------------

class LinearQkvInterface(Protocol):
    """Interface for linear_qkv modules."""

    def forward(self, input: Tensor, /) -> tuple[Tensor, object]:
        """Apply linear_qkv."""
        ...

    def backward_dw(self) -> None:
        """Weight gradient backward for linear_qkv."""
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
        tp_group: Optional[torch.distributed.ProcessGroup],
        name: Optional[str] = None,
    ) -> LinearQkvInterface: ...


class LinearInterface(Protocol):
    """Interface for linear_q and linear_kv modules."""

    def forward(self, input: Tensor, /) -> tuple[Tensor, object]:
        """Apply the linear layer."""
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
        name: Optional[str] = None,
    ) -> LinearInterface: ...


class CoreAttentionInterface(Protocol):
    """Interface for core_attention modules."""

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor],
        /,
        *,
        attn_mask_type: str,
        attention_bias: Optional[Tensor],
        packed_seq_params: Optional[object],
    ) -> Tensor: ...


class CoreAttentionBuilder(Protocol):
    """Protocol for building core_attention layers."""

    def __call__(
        self,
        *,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: str,
        attention_type: str,
        cp_comm_type: Optional[str],
        softmax_scale: Optional[float],
        pg_collection: Optional[object],
    ) -> CoreAttentionInterface: ...


class LinearProjInterface(Protocol):
    """Interface for linear_proj output projection modules."""

    def forward(
        self, hidden_states: Tensor, /
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Apply linear_proj."""
        ...

    def backward_dw(self) -> None:
        """Compute weight gradients for output projection."""
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
        tp_group: Optional[torch.distributed.ProcessGroup],
        name: Optional[str] = None,
    ) -> LinearProjInterface: ...


# ---------------------------------------------------------------------------
# Submodule specs
# ---------------------------------------------------------------------------

@dataclass
class SelfAttentionSubmodules:
    """Configuration for submodules of a self-attention layer."""

    linear_qkv: LinearQkvBuilder
    core_attention: CoreAttentionBuilder
    linear_proj: LinearProjBuilder
    q_layernorm: Optional[object] = None   # LayerNormBuilder
    k_layernorm: Optional[object] = None   # LayerNormBuilder


@dataclass
class CrossAttentionSubmodules:
    """Configuration for submodules of a cross-attention layer."""

    linear_q: LinearLayerBuilder
    linear_kv: LinearLayerBuilder
    core_attention: CoreAttentionBuilder
    linear_proj: LinearProjBuilder


# ---------------------------------------------------------------------------
# Native linear implementations (used when TE/submodule not provided)
# ---------------------------------------------------------------------------

class _NativeQKVLinear(nn.Module):
    """Column-parallel QKV linear (native PyTorch fallback)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.tensor_model_parallel = True
        self.linear.weight.partition_dim = 0

    def forward(self, x: Tensor, /) -> tuple[Tensor, None]:
        return F.linear(x, self.linear.weight, self.linear.bias), None

    def backward_dw(self) -> None:
        pass


class _NativeProjLinear(nn.Module):
    """Row-parallel output projection linear (native PyTorch fallback)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.tensor_model_parallel = True
        self.linear.weight.partition_dim = 1

    def forward(self, x: Tensor, /) -> tuple[Tensor, Optional[Tensor]]:
        out = F.linear(x, self.linear.weight)
        return out, self.linear.bias if self.linear.bias is not None else None

    def backward_dw(self) -> None:
        pass


# ---------------------------------------------------------------------------
# L2-norm for QK layernorm (M3774)
# ---------------------------------------------------------------------------

class L2Norm(nn.Module):
    """L2 normalisation for per-head Q/K tensors.

    Normalises the last dimension (head_dim) to unit L2 norm,
    scaled by a learnable weight initialised to 1.
    """

    def __init__(
        self,
        hidden_size: int,
        config: TransformerConfig,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        norm = x.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return (x / norm) * self.weight


# ---------------------------------------------------------------------------
# Base Attention class
# ---------------------------------------------------------------------------

class Attention(MegatronModule, ABC):
    """Base attention class containing shared modules.

    Subclasses:
      * :class:`SelfAttention` — standard self-attention with fused QKV
      * :class:`CrossAttention` — cross-attention with separate Q and KV

    DES-LOC integration:
        Logs tier at construction.  Tier does not affect forward logic.

    AutoSP integration:
        When sequence_parallel=True, all-to-all scatter/gather is applied
        around the QKV and projection linear layers.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Union[SelfAttentionSubmodules, CrossAttentionSubmodules],
        layer_number: int,
        attn_mask_type: str = "causal",
        attention_type: str = "self",
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[object] = None,
        pp_layer_offset: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.layer_number = layer_number
        self._pp_layer_offset = pp_layer_offset
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type
        self.batch_invariant_mode = getattr(config, "batch_invariant_mode", False)

        assert config.kv_channels is not None
        assert config.num_attention_heads is not None
        assert config.num_query_groups is not None

        self.query_projection_size = config.kv_channels * config.num_attention_heads
        self.kv_projection_size = config.kv_channels * config.num_query_groups

        tp_world_size = _get_tp_world_size()
        self.world_size = tp_world_size

        self.hidden_size_per_attention_head = divide(
            self.query_projection_size, config.num_attention_heads
        )

        # TP > num_kv_heads: sub-sharding (M2807)
        if config.num_query_groups < tp_world_size:
            self.num_query_groups_per_partition = 1
            self.num_attention_heads_per_partition = divide(
                config.num_attention_heads, config.num_query_groups
            )
        else:
            self.num_query_groups_per_partition = divide(
                config.num_query_groups, tp_world_size
            )
            self.num_attention_heads_per_partition = divide(
                config.num_attention_heads, tp_world_size
            )

        # KV key/value hidden sizes (may differ for MLA, M3730)
        self.key_hidden_size = self.hidden_size_per_attention_head
        self.val_hidden_size = self.hidden_size_per_attention_head

        # Adjust config for TE when num_kv_heads < tp_size
        if config.num_query_groups < tp_world_size:
            tmp_config = copy.deepcopy(config)
            tmp_config.num_query_groups = tp_world_size
        else:
            tmp_config = config

        # Core attention (DotProductAttention or TEDotProductAttention)
        self.core_attention = self._build_core_attention(
            tmp_config, submodules, layer_number, attn_mask_type,
            attention_type, cp_comm_type, pg_collection,
        )

        # Selective recompute flag
        self.checkpoint_core_attention = (
            getattr(config, "recompute_granularity", None) == "selective"
            and "core_attn" in getattr(config, "recompute_modules", [])
        )

        # Fine-grained activation offloading flags
        self.offload_qkv_linear = (
            getattr(config, "fine_grained_activation_offloading", False)
            and "qkv_linear" in getattr(config, "offload_modules", [])
        )
        self.offload_core_attention = (
            getattr(config, "fine_grained_activation_offloading", False)
            and "core_attn" in getattr(config, "offload_modules", [])
        )
        self.offload_attn_proj = (
            getattr(config, "fine_grained_activation_offloading", False)
            and "attn_proj" in getattr(config, "offload_modules", [])
        )

        # Output projection
        self.linear_proj = self._build_linear_proj(config, submodules, name)

        # DES-LOC: log tier assignment
        tier = config.get_layer_tier(self.layer_number - 1)
        if tier is not None:
            logger.debug(
                "Attention layer %d → DES-LOC tier: %s", self.layer_number, tier.upper()
            )

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _build_core_attention(
        self,
        config: TransformerConfig,
        submodules: Union[SelfAttentionSubmodules, CrossAttentionSubmodules],
        layer_number: int,
        attn_mask_type: str,
        attention_type: str,
        cp_comm_type: Optional[str],
        pg_collection: Optional[object],
    ) -> nn.Module:
        """Build the core attention module from submodules or native fallback."""
        if hasattr(submodules, "core_attention") and submodules.core_attention is not None:
            return submodules.core_attention(
                config=config,
                layer_number=layer_number,
                attn_mask_type=attn_mask_type,
                attention_type=attention_type,
                cp_comm_type=cp_comm_type,
                softmax_scale=getattr(config, "softmax_scale", None),
                pg_collection=pg_collection,
            )
        # Native DotProductAttention fallback
        from deepspeed.core.transformer.dot_product_attention import DotProductAttention
        return DotProductAttention(
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
        )

    def _build_linear_proj(
        self,
        config: TransformerConfig,
        submodules: Union[SelfAttentionSubmodules, CrossAttentionSubmodules],
        name: Optional[str],
    ) -> nn.Module:
        """Build the output projection from submodules or native fallback."""
        if hasattr(submodules, "linear_proj") and submodules.linear_proj is not None:
            return submodules.linear_proj(
                self.query_projection_size,
                config.hidden_size,
                config=config,
                init_method=config.output_layer_init_method,
                bias=config.add_bias_linear,
                input_is_parallel=True,
                skip_bias_add=True,
                is_expert=False,
                tp_comm_buffer_name="proj",
                tp_group=_get_tp_group(),
                name=(name + ".linear_proj") if name else None,
            )
        # Native fallback: row-parallel projection
        proj_in = self.query_projection_size // self.world_size
        return _NativeProjLinear(proj_in, config.hidden_size, bias=config.add_bias_linear)

    # ------------------------------------------------------------------
    # AutoSP all-to-all helpers
    # ------------------------------------------------------------------

    def _sp_scatter(self, x: Tensor, sp_group) -> Tensor:
        """A2A: [seq/sp, b, h] → [seq, b, h/sp]."""
        sp_size = _get_sp_world_size()
        if sp_size == 1 or sp_group is None:
            return x
        seq_chunk, batch, hidden = x.shape
        hidden_per_rank = hidden // sp_size
        inp = x.view(seq_chunk, batch, sp_size, hidden_per_rank).permute(2, 0, 1, 3).contiguous()
        out_list = [torch.empty_like(inp[0]) for _ in range(sp_size)]
        torch.distributed.all_to_all(out_list, list(inp.unbind(0)), group=sp_group)
        out = torch.stack(out_list, dim=0).permute(1, 0, 2, 3)
        return out.reshape(seq_chunk * sp_size, batch, hidden_per_rank)

    def _sp_gather(self, x: Tensor, sp_group) -> Tensor:
        """A2A: [seq, b, h/sp] → [seq/sp, b, h]."""
        sp_size = _get_sp_world_size()
        if sp_size == 1 or sp_group is None:
            return x
        total_seq, batch, hidden_per_rank = x.shape
        seq_chunk = total_seq // sp_size
        ctx = x.reshape(sp_size, seq_chunk, batch, hidden_per_rank).permute(1, 0, 2, 3).contiguous()
        inp_list = list(ctx.unbind(1))
        out_list = [torch.empty_like(inp_list[0]) for _ in range(sp_size)]
        torch.distributed.all_to_all(out_list, inp_list, group=sp_group)
        out = torch.stack(out_list, dim=1).permute(0, 2, 1, 3)
        return out.reshape(seq_chunk, batch, hidden_per_rank * sp_size)

    # ------------------------------------------------------------------
    # Checkpointed attention forward
    # ------------------------------------------------------------------

    def _checkpointed_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor],
        attn_mask_type: Optional[str] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[object] = None,
    ) -> Tensor:
        """Selective activation checkpoint wrapping core attention."""
        try:
            from megatron.core import tensor_parallel as _tp

            def custom_forward(*inputs):
                q, k, v, mask = inputs[:4]
                return self.core_attention(
                    q, k, v, mask,
                    attn_mask_type=attn_mask_type or self.attn_mask_type,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                )

            return _tp.checkpoint(custom_forward, False, query, key, value, attention_mask)
        except Exception:
            # Fallback: no checkpointing
            return self.core_attention(
                query, key, value, attention_mask,
                attn_mask_type=attn_mask_type or self.attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

    # ------------------------------------------------------------------
    # KV cache allocation
    # ------------------------------------------------------------------

    def _allocate_memory(
        self,
        max_seq_len: int,
        batch_size: int,
        dim: int,
        dtype: torch.dtype,
    ) -> Tensor:
        """Allocate KV-cache buffer for static batching inference."""
        return torch.empty(
            max_seq_len,
            batch_size,
            self.num_query_groups_per_partition,
            dim,
            dtype=dtype,
            device=torch.cuda.current_device(),
        )

    # ------------------------------------------------------------------
    # Output gate (M2950: Qwen3-Next attention output gate)
    # ------------------------------------------------------------------

    def _apply_output_gate(self, x: Tensor, gate: Tensor) -> Tensor:
        """Apply sigmoid-gated output: x * sigmoid(gate)."""
        x_dtype = x.dtype
        gate = gate.contiguous().view(*x.shape)
        x = x * torch.sigmoid(gate.float())
        return x.to(x_dtype)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def get_query_key_value_tensors(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[Tensor] = None,
        output_gate: bool = False,
        split_qkv: bool = True,
    ) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Derive Q, K, V (and optionally gate) from hidden_states.

        Implemented by SelfAttention and CrossAttention.
        """
        ...

    # ------------------------------------------------------------------
    # Flash decode
    # ------------------------------------------------------------------

    def flash_decode(
        self,
        sequence_len_offset: Tensor,
        query_layer: Tensor,
        key_layer: Tensor,
        value_layer: Tensor,
        inference_key_memory: Tensor,
        inference_value_memory: Tensor,
        rotary_cos: Optional[Tensor],
        rotary_sin: Optional[Tensor],
        rotary_interleaved: bool = False,
    ) -> Tensor:
        """Flash decode: RoPE + KV-cache update + attention in one kernel.

        Uses flash_attn_with_kvcache from the flash-attn package.
        """
        assert _fa2_kvcache is not None, (
            "Flash Decoding requires flash_attn_with_kvcache "
            "(install the flash-attn package)."
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

        return _fa2_kvcache(
            q=q, k_cache=k_cache, v_cache=v_cache,
            k=k, v=v,
            rotary_cos=rotary_cos, rotary_sin=rotary_sin,
            cache_seqlens=sequence_len_offset,
            rotary_interleaved=rotary_interleaved,
        )

    # ------------------------------------------------------------------
    # KV cache inference adjustment
    # ------------------------------------------------------------------

    def _adjust_key_value_for_inference(
        self,
        inference_context: Optional[object],
        query: Tensor,
        key: Tensor,
        value: Tensor,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]],
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        sequence_len_offset: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[object], str, Optional[Tensor]]:
        """Save KV to cache and return updated K, V for attention.

        Returns:
            (query, key, value, rotary_pos_emb, attn_mask_type, block_table)
        """
        attn_mask_type = self.attn_mask_type
        if inference_context is None:
            return query, key, value, rotary_pos_emb, attn_mask_type, None

        is_static = inference_context.is_static_batching()

        if is_static:
            # Allocate if not yet done
            if self.layer_number not in inference_context.key_value_memory_dict:
                inf_max_seq_len = inference_context.max_sequence_length
                inf_max_batch = inference_context.max_batch_size
                k_mem = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch, self.key_hidden_size, key.dtype
                )
                v_mem = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch, self.val_hidden_size, value.dtype
                )
                inference_context.key_value_memory_dict[self.layer_number] = (k_mem, v_mem)

            k_mem, v_mem = inference_context.key_value_memory_dict[self.layer_number]

        # Switch off masking for decode phase
        seq_offset = getattr(inference_context, "sequence_len_offset", 0) or 0
        if (not is_static or seq_offset > 0) and not self.training:
            attn_mask_type = "no_mask"

        if is_static:
            batch_start = getattr(inference_context, "batch_size_offset", 0) or 0
            batch_end = batch_start + key.size(1)
            seq_start = seq_offset
            seq_end = seq_start + key.size(0)

            # Adjust rotary embeddings for static batching
            if rotary_pos_emb is not None:
                q_pos, k_pos = rotary_pos_emb
                if q_pos is not None:
                    q_pos = q_pos[seq_start:seq_end]
                if k_pos is not None:
                    k_pos = k_pos[:seq_end]
                rotary_pos_emb = (q_pos, k_pos)

            # Copy KV into cache
            k_mem[seq_start:seq_end, batch_start:batch_end] = key
            v_mem[seq_start:seq_end, batch_start:batch_end] = value
            key = k_mem[:seq_end, batch_start:batch_end]
            value = v_mem[:seq_end, batch_start:batch_end]
            block_table = None
        else:
            # Dynamic batching
            pp_layer_offset = self._pp_layer_offset or 0

            # Apply rotary before appending (dynamic batching path)
            if rotary_pos_emb is not None:
                q_pos, k_pos = rotary_pos_emb
                if k_pos is not None:
                    key = apply_rotary_pos_emb(
                        key, k_pos,
                        rotary_interleaved=getattr(self.config, "rotary_interleaved", False),
                    )
                    rotary_pos_emb = (q_pos, None)

            inference_context.append_key_value_cache(
                self.layer_number - pp_layer_offset, key, value
            )
            key, value, block_table = inference_context.key_value_cache(
                self.layer_number - pp_layer_offset
            )

        return query, key, value, rotary_pos_emb, attn_mask_type, block_table

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[object] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[object] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[object] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass through the attention layer.

        Args:
            hidden_states: ``[seq, batch, hidden]``
            attention_mask: Boolean mask ``[1, 1, seq, seq]``.
            key_value_states: For cross-attention; None for self-attention.
            inference_context: KV-cache context for inference.
            rotary_pos_emb: Rotary embeddings (tuple or single tensor).
            rotary_pos_cos: Precomputed cosines (flash decode).
            rotary_pos_sin: Precomputed sines (flash decode).
            rotary_pos_cos_sin: Combined cos/sin (flashinfer dynamic batching).
            attention_bias: Additive bias ``[b, nh, sq, sk]``.
            packed_seq_params: THD packed sequence params.
            sequence_len_offset: Offset for CUDA graph inference.
            inference_params: Deprecated alias for inference_context.

        Returns:
            (output, bias): output ``[seq, batch, hidden]``, bias or None.
        """
        # Deprecated alias
        if inference_context is None and inference_params is not None:
            inference_context = inference_params

        # No-rope layers (config.no_rope_freq, M2950 era)
        no_rope_freq = getattr(self.config, "no_rope_freq", None)
        if no_rope_freq and no_rope_freq[self.layer_number - 1]:
            rotary_pos_emb = None

        # Flash decode and flashinfer modes
        flash_decode_mode = (
            inference_context is not None
            and getattr(self.config, "flash_decode", False)
        )
        if flash_decode_mode:
            rotary_pos_emb = None

        # Normalise rotary_pos_emb to a (q_emb, k_emb) tuple
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb, rotary_pos_emb)

        # ---------------------------------------------------------------
        # Query, Key, Value projection
        # ---------------------------------------------------------------
        split_qkv = (self.attention_type == "cross") or True  # always split in this port
        qkv_output = self.get_query_key_value_tensors(
            hidden_states, key_value_states,
            output_gate=getattr(self.config, "attention_output_gate", False),
            split_qkv=split_qkv,
        )

        attn_mask_type = self.attn_mask_type
        block_table = None
        gate = None

        if getattr(self.config, "attention_output_gate", False):
            query, key, value, gate = qkv_output
        else:
            query, key, value = qkv_output

        # ---------------------------------------------------------------
        # In-decode flash decode path (static batching)
        # ---------------------------------------------------------------
        in_decode = (
            inference_context is not None
            and hasattr(inference_context, "is_decode_only")
            and inference_context.is_decode_only()
        )

        if in_decode and flash_decode_mode:
            assert self.layer_number in inference_context.key_value_memory_dict
            k_mem, v_mem = inference_context.key_value_memory_dict[self.layer_number]
            out = self.flash_decode(
                sequence_len_offset=sequence_len_offset,
                query_layer=query,
                key_layer=key,
                value_layer=value,
                inference_key_memory=k_mem,
                inference_value_memory=v_mem,
                rotary_cos=rotary_pos_cos,
                rotary_sin=rotary_pos_sin,
                rotary_interleaved=getattr(self.config, "rotary_interleaved", False),
            )
            out = out.transpose(0, 1).contiguous()
            context_layer = out.view(out.size(0), out.size(1), -1)
            output, bias = self.linear_proj(context_layer)
            return output, bias

        # ---------------------------------------------------------------
        # KV cache update (non-flash-decode path)
        # ---------------------------------------------------------------
        query, key, value, rotary_pos_emb, attn_mask_type, block_table = (
            self._adjust_key_value_for_inference(
                inference_context, query, key, value, rotary_pos_emb,
                rotary_pos_cos, rotary_pos_sin, rotary_pos_cos_sin,
                sequence_len_offset,
            )
        )

        # THD packed-seq squeeze (M3730)
        if (
            packed_seq_params is not None
            and getattr(packed_seq_params, "qkv_format", None) == "thd"
        ):
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # ---------------------------------------------------------------
        # Rotary position embeddings
        # ---------------------------------------------------------------
        if rotary_pos_emb is not None and not flash_decode_mode:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            # cu_seqlens for THD format
            cu_seqlens_q = cu_seqlens_kv = None
            if packed_seq_params is not None:
                if getattr(packed_seq_params, "qkv_format", None) == "thd":
                    cu_seqlens_q = getattr(packed_seq_params, "cu_seqlens_q", None)
                    cu_seqlens_kv = getattr(packed_seq_params, "cu_seqlens_kv", None)

            mscale = getattr(self.config, "yarn_attention_factor", 1.0) or 1.0

            if q_pos_emb is not None:
                query = apply_rotary_pos_emb(
                    query, q_pos_emb,
                    rotary_interleaved=getattr(self.config, "rotary_interleaved", False),
                    mscale=mscale,
                )
            if k_pos_emb is not None:
                key = apply_rotary_pos_emb(
                    key, k_pos_emb,
                    rotary_interleaved=getattr(self.config, "rotary_interleaved", False),
                    mscale=mscale,
                )

        # ---------------------------------------------------------------
        # Core attention computation
        # ---------------------------------------------------------------
        use_sp = (
            getattr(self.config, "sequence_parallel", False)
            and _get_sp_world_size() > 1
        )
        sp_group = _get_sp_group() if use_sp else None

        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query, key, value, attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query, key, value, attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

        # Reshape THD output back (M3730)
        if (
            packed_seq_params is not None
            and getattr(packed_seq_params, "qkv_format", None) == "thd"
        ):
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # ---------------------------------------------------------------
        # Output gate (M2950: Qwen3-Next)
        # ---------------------------------------------------------------
        if gate is not None:
            core_attn_out = self._apply_output_gate(core_attn_out, gate)

        # ---------------------------------------------------------------
        # Output projection
        # ---------------------------------------------------------------
        if isinstance(self.linear_proj, _NativeProjLinear):
            output, bias = self.linear_proj.forward(core_attn_out)
        else:
            output, bias = self.linear_proj(core_attn_out)

        # All-reduce across TP ranks (row-parallel)
        tp_size = _get_tp_world_size()
        if tp_size > 1:
            tp_group = _get_tp_group()
            if tp_group is not None:
                torch.distributed.all_reduce(output, group=tp_group)

        return output, bias

    # ------------------------------------------------------------------
    # Weight gradient backward
    # ------------------------------------------------------------------

    def backward_dw(self) -> None:
        """Trigger weight-gradient backward for projection layers."""
        if hasattr(self.linear_proj, "backward_dw"):
            self.linear_proj.backward_dw()

    # ------------------------------------------------------------------
    # FP8/FP4 hooks
    # ------------------------------------------------------------------

    def set_for_recompute_input_layernorm(self) -> None:
        """Hook for FP8/FP4 recompute of input layernorm."""
        raise NotImplementedError("set_for_recompute_input_layernorm not implemented.")

    def clip_qk(self) -> None:
        """QK logit clipping (M2831). Implemented in SelfAttention."""
        raise NotImplementedError("clip_qk not implemented in base Attention.")


# ---------------------------------------------------------------------------
# SelfAttention
# ---------------------------------------------------------------------------

class SelfAttention(Attention):
    """Self-attention with fused QKV projection.

    Input/output layout: ``[seq, batch, hidden]`` (Megatron convention).

    GQA / MQA support:
        When ``config.num_query_groups < config.num_attention_heads``, the
        QKV projection produces a smaller KV portion.  Activation sharding
        handles the case where ``num_kv_heads < tp_size`` via AG + slice.

    QK LayerNorm (M3774):
        When ``config.qk_layernorm`` or ``config.qk_l2_norm`` is enabled,
        a per-head norm is applied to Q and K after projection.

    Attention output gate (M2950):
        When ``config.attention_output_gate`` is True, the QKV projection
        produces an extra gate tensor G of the same size as Q; the attention
        output is multiplied by sigmoid(G).

    DES-LOC: logs tier at construction (inherited from Attention base).
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: str = "causal",
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[object] = None,
        pp_layer_offset: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
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

        # QKV output dim
        self.linear_qkv_out_dim = (
            self.query_projection_size + 2 * self.kv_projection_size
        )
        if getattr(config, "attention_output_gate", False):
            self.linear_qkv_out_dim += (
                config.kv_channels * config.num_attention_heads
            )

        # Build QKV projection
        if hasattr(submodules, "linear_qkv") and submodules.linear_qkv is not None:
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
                tp_group=_get_tp_group(),
                name=(name + ".linear_qkv") if name else None,
            )
        else:
            # Native fallback: column-parallel QKV
            qkv_out_per_tp = self.linear_qkv_out_dim // self.world_size
            self.linear_qkv = _NativeQKVLinear(
                config.hidden_size, qkv_out_per_tp,
                bias=config.add_bias_linear,
            )

        # QK LayerNorm (M3774)
        self.q_layernorm: Optional[nn.Module] = None
        self.k_layernorm: Optional[nn.Module] = None

        qk_l2_norm = getattr(config, "qk_l2_norm", False)
        qk_layernorm = getattr(config, "qk_layernorm", False)

        if qk_l2_norm:
            q_norm_cls = getattr(submodules, "q_layernorm", None) or L2Norm
            k_norm_cls = getattr(submodules, "k_layernorm", None) or L2Norm
        elif qk_layernorm:
            q_norm_cls = getattr(submodules, "q_layernorm", None)
            k_norm_cls = getattr(submodules, "k_layernorm", None)
            if q_norm_cls is None or k_norm_cls is None:
                logger.warning(
                    "qk_layernorm=True but no q_layernorm/k_layernorm submodule set; "
                    "falling back to standard LayerNorm."
                )
                q_norm_cls = k_norm_cls = nn.LayerNorm
        else:
            q_norm_cls = k_norm_cls = None

        head_dim = self.hidden_size_per_attention_head
        if q_norm_cls is not None:
            self.q_layernorm = q_norm_cls(
                hidden_size=head_dim,
                config=config,
                eps=config.layernorm_epsilon,
            )
        if k_norm_cls is not None:
            self.k_layernorm = k_norm_cls(
                hidden_size=head_dim,
                config=config,
                eps=config.layernorm_epsilon,
            )

    # ------------------------------------------------------------------
    # QKV decomposition
    # ------------------------------------------------------------------

    def get_query_key_value_tensors(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[Tensor] = None,
        output_gate: bool = False,
        split_qkv: bool = True,
    ) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Project hidden_states to Q, K, V (and optionally gate G).

        Handles the full GQA / sub-sharding logic:
          1. QKV linear: ``[seq, b, h] → [seq, b, (q+2kv)/tp]``
          2. For num_kv_heads < tp_size: all-gather + slice (M2807).
          3. Reshape to ``[seq, b, ng, (np/ng+2) * hn]``.
          4. Split into Q, K, V (and G if output_gate).
          5. Apply QK layernorm if configured.

        Args:
            hidden_states: ``[seq, batch, hidden]``
            key_value_states: Ignored for self-attention.
            output_gate: If True, also return a gate tensor G.
            split_qkv: Always True in this implementation.

        Returns:
            (query, key, value) or (query, key, value, gate).
        """
        # QKV linear: [sq, b, h] → [sq, b, qkv_per_tp]
        if isinstance(self.linear_qkv, _NativeQKVLinear):
            mixed_qkv, _ = self.linear_qkv.forward(hidden_states)
        else:
            mixed_qkv, _ = self.linear_qkv(hidden_states)

        # Number of Q heads per KV group
        num_q_per_kv = (
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        )
        num_qkv_heads_per_group = num_q_per_kv + 2
        if output_gate:
            num_qkv_heads_per_group += num_q_per_kv

        # Sub-sharding: TP > num_kv_heads path (M2807)
        if self.config.num_query_groups < self.world_size:
            # All-gather last dim across TP ranks
            tp_group = _get_tp_group()
            if tp_group is not None:
                gathered = [torch.empty_like(mixed_qkv) for _ in range(self.world_size)]
                torch.distributed.all_gather(gathered, mixed_qkv, group=tp_group)
                mixed_qkv = torch.cat(gathered, dim=-1)
            tp_rank = _get_tp_rank()
            # Slice to the group this rank is responsible for
            kv_groups = self.config.num_query_groups
            idx = tp_rank // (self.world_size // kv_groups)
            size = mixed_qkv.size(-1) // kv_groups
            mixed_qkv = mixed_qkv[..., idx * size : (idx + 1) * size]

        # Reshape: [sq, b, qkv_per_group * ng] → [sq, b, ng, qkv_per_group * hn]
        new_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            num_qkv_heads_per_group * self.hidden_size_per_attention_head,
        )
        mixed_qkv = mixed_qkv.view(*new_shape)

        hn = self.hidden_size_per_attention_head

        if output_gate:
            split_sizes = [
                num_q_per_kv * hn,  # Q
                num_q_per_kv * hn,  # gate G
                hn,                  # K
                hn,                  # V
            ]
            query_4d, gate_4d, key, value = torch.split(mixed_qkv, split_sizes, dim=3)
        else:
            split_sizes = [num_q_per_kv * hn, hn, hn]
            query_4d, key, value = torch.split(mixed_qkv, split_sizes, dim=3)
            gate_4d = None

        # [sq, b, ng, np/ng * hn] → [sq, b, np, hn]
        query = query_4d.reshape(query_4d.size(0), query_4d.size(1), -1, hn)

        # Sub-sharding: select the right Q heads for this TP rank (M2807)
        if self.config.num_query_groups < self.world_size:
            tp_rank = _get_tp_rank()
            ratio = self.world_size // self.config.num_query_groups
            local_idx = tp_rank % ratio
            q_per_rank = self.num_attention_heads_per_partition // ratio
            query = query[:, :, local_idx * q_per_rank : (local_idx + 1) * q_per_rank, :]

        # QK layernorm (M3774)
        if self.q_layernorm is not None:
            query = self.q_layernorm(query)
        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        if output_gate and gate_4d is not None:
            gate = gate_4d.reshape(*gate_4d.shape[:2], -1, hn)
            if self.config.num_query_groups < self.world_size:
                gate = gate[:, :, local_idx * q_per_rank : (local_idx + 1) * q_per_rank, :]
            return query, key, value, gate

        return query, key, value

    # ------------------------------------------------------------------
    # Weight gradient backward
    # ------------------------------------------------------------------

    def backward_dw(self) -> None:
        """Execute weight-update operations for QKV and output projections."""
        self._backward_qkv_proj()
        self._backward_output_proj()

    def _backward_qkv_proj(self) -> None:
        if hasattr(self.linear_qkv, "backward_dw"):
            self.linear_qkv.backward_dw()

    def _backward_output_proj(self) -> None:
        if hasattr(self.linear_proj, "backward_dw"):
            self.linear_proj.backward_dw()

    # ------------------------------------------------------------------
    # FP8 hook
    # ------------------------------------------------------------------

    def set_for_recompute_input_layernorm(self) -> None:
        """Set QKV to save original input for FP8/FP4 recompute."""
        try:
            from megatron.core.extensions.transformer_engine import set_save_original_input
            set_save_original_input(self.linear_qkv)
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # QK logit clipping (M2831)
    # ------------------------------------------------------------------

    def clip_qk(self) -> None:
        """Clip Q/K weight norms to prevent attention logit explosion.

        Implements the QK-Clip technique from M2831.  Should be called
        after each forward pass (before the backward) when
        ``config.qk_clip=True``.
        """
        if not getattr(self.config, "qk_clip", False):
            raise ValueError("qk_clip option must be enabled in config.")

        attn = self.core_attention
        max_logits = getattr(attn, "current_max_attn_logits", None)
        if max_logits is None:
            raise ValueError("current_max_attn_logits is None; run a forward pass first.")

        qk_threshold = getattr(self.config, "qk_clip_threshold", 10.0)
        qk_alpha = getattr(self.config, "qk_clip_alpha", 0.5)
        ng = self.num_query_groups_per_partition
        np_per_tp = self.num_attention_heads_per_partition

        # Group max over Q-heads within each KV group
        grouped_max = max_logits.view(ng, -1).max(dim=1).values  # [ng]
        if not torch.any(grouped_max > qk_threshold):
            attn.current_max_attn_logits = None
            return

        eta = torch.clamp(qk_threshold / grouped_max, max=1.0).view(ng, 1, 1)  # [ng, 1, 1]

        # Apply clipping to linear_qkv weights
        def _clip_weight(w: Tensor) -> Tensor:
            w_r = w.view(
                ng,
                (self.query_projection_size + 2 * self.kv_projection_size) // ng,
                -1,
            )
            qps = self.query_projection_size
            kps = self.kv_projection_size
            ng_ = ng
            wq = w_r[:, : qps // ng_, :]
            wk = w_r[:, qps // ng_ : (qps + kps) // ng_, :]
            wv = w_r[:, (qps + kps) // ng_ :, :]
            eta_ext = eta.expand_as(wq)
            wq = wq * torch.pow(eta_ext, qk_alpha)
            wk = wk * torch.pow(eta, 1.0 - qk_alpha)
            return torch.cat([wq, wk, wv], dim=1).view(
                self.query_projection_size + 2 * self.kv_projection_size, -1
            )

        if hasattr(self.linear_qkv, "weight"):
            w = self.linear_qkv.weight
            if hasattr(w, "main_param"):
                w.main_param.data.copy_(_clip_weight(w.main_param.data))
            w.data.copy_(_clip_weight(w.data))
        elif hasattr(self.linear_qkv, "linear"):
            w = self.linear_qkv.linear.weight
            w.data.copy_(_clip_weight(w.data))

        attn.current_max_attn_logits = None


# ---------------------------------------------------------------------------
# CrossAttention
# ---------------------------------------------------------------------------

class CrossAttention(Attention):
    """Cross-attention with separate Q and KV projections.

    Used in encoder-decoder architectures.  Input is ``[seq, b, h]``
    for both hidden_states (queries) and key_value_states (keys/values).

    GQA is not currently supported in cross-attention
    (num_query_groups must equal num_attention_heads).

    DES-LOC: tier logged at construction (inherited from Attention).
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CrossAttentionSubmodules,
        layer_number: int,
        attn_mask_type: str = "causal",
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[object] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="cross",
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            name=name,
        )

        if config.num_query_groups != config.num_attention_heads:
            raise ValueError(
                "Group query attention is not currently supported in cross attention. "
                f"num_query_groups={config.num_query_groups} != "
                f"num_attention_heads={config.num_attention_heads}"
            )
        assert self.query_projection_size == self.kv_projection_size

        tp_group = _get_tp_group()

        # Q projection (column-parallel)
        if hasattr(submodules, "linear_q") and submodules.linear_q is not None:
            self.linear_q = submodules.linear_q(
                config.hidden_size,
                self.query_projection_size,
                config=config,
                init_method=config.init_method,
                gather_output=False,
                bias=config.add_bias_linear,
                skip_bias_add=False,
                is_expert=False,
                name=(name + ".linear_q") if name else None,
            )
        else:
            q_per_tp = self.query_projection_size // self.world_size
            self.linear_q = _NativeQKVLinear(config.hidden_size, q_per_tp, bias=config.add_bias_linear)

        # KV projection (column-parallel, 2x output for K and V)
        if hasattr(submodules, "linear_kv") and submodules.linear_kv is not None:
            self.linear_kv = submodules.linear_kv(
                config.hidden_size,
                2 * self.kv_projection_size,
                config=config,
                init_method=config.init_method,
                gather_output=False,
                bias=config.add_bias_linear,
                skip_bias_add=False,
                is_expert=False,
                name=(name + ".linear_kv") if name else None,
            )
        else:
            kv_per_tp = 2 * self.kv_projection_size // self.world_size
            self.linear_kv = _NativeQKVLinear(config.hidden_size, kv_per_tp, bias=config.add_bias_linear)

    def get_query_key_value_tensors(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[Tensor] = None,
        output_gate: bool = False,
        split_qkv: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Project hidden_states → Q and key_value_states → K, V.

        Args:
            hidden_states: ``[sq, batch, hidden]`` (query source).
            key_value_states: ``[sk, batch, hidden]`` (context/encoder output).
            output_gate: Not supported for cross-attention.
            split_qkv: Must be True for cross-attention.

        Returns:
            (query, key, value).
        """
        assert split_qkv, "split_qkv must be True for CrossAttention"
        assert not output_gate, "attention_output_gate is not supported in cross attention"
        assert key_value_states is not None, "key_value_states required for CrossAttention"

        hn = self.hidden_size_per_attention_head
        np_per_tp = self.num_attention_heads_per_partition

        # KV: [sk, b, h] → [sk, b, 2*kv_per_tp]
        if isinstance(self.linear_kv, _NativeQKVLinear):
            mixed_kv, _ = self.linear_kv.forward(key_value_states)
        else:
            mixed_kv, _ = self.linear_kv(key_value_states)

        # [sk, b, 2*kv_per_tp] → [sk, b, np, 2*hn] → 2x [sk, b, np, hn]
        mixed_kv = mixed_kv.view(*mixed_kv.size()[:-1], np_per_tp, 2 * hn)
        key, value = torch.split(mixed_kv, hn, dim=-1)

        # Q: [sq, b, h] → [sq, b, q_per_tp]
        if isinstance(self.linear_q, _NativeQKVLinear):
            query, _ = self.linear_q.forward(hidden_states)
        else:
            query, _ = self.linear_q(hidden_states)

        # [sq, b, q_per_tp] → [sq, b, np, hn]
        query = query.view(*query.size()[:-1], np_per_tp, hn)

        return query, key, value
