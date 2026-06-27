# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Transformer layers with AutoSP sequence parallel and DES-LOC support."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepspeed.core.model_parallel_config import ModelParallelConfig


# ===========================================================================
# transformer_config.py
# ===========================================================================

@dataclass
class TransformerConfig(ModelParallelConfig):
    """Full transformer configuration. Extends ModelParallelConfig."""

    num_layers: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_query_groups: Optional[int] = None  # GQA
    ffn_hidden_size: Optional[int] = None
    kv_channels: Optional[int] = None

    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    layernorm_epsilon: float = 1e-5
    add_bias_linear: bool = False
    gated_linear_unit: bool = False
    activation_func: Callable = F.silu  # SwiGLU default

    normalization: Literal["LayerNorm", "RMSNorm"] = "RMSNorm"
    apply_residual_connection_post_layernorm: bool = False

    # Rotary embeddings
    rotary_interleaved: bool = False
    window_size: Optional[Tuple[int, int]] = None

    # MoE
    num_moe_experts: Optional[int] = None

    # MTP (Multi-Token Prediction)
    mtp_num_layers: Optional[int] = None
    mtp_loss_scaling_factor: float = 0.1

    # MLA (Multi-Latent Attention)
    multi_latent_attention: bool = False

    # Per-token loss
    calculate_per_token_loss: bool = False

    def __post_init__(self):
        if self.ffn_hidden_size is None:
            # SwiGLU default: 8/3 * hidden, rounded to 64
            self.ffn_hidden_size = int(self.hidden_size * 8 / 3)
            self.ffn_hidden_size = ((self.ffn_hidden_size + 63) // 64) * 64
        if self.kv_channels is None:
            self.kv_channels = self.hidden_size // self.num_attention_heads
        if self.num_query_groups is None:
            self.num_query_groups = self.num_attention_heads


# ===========================================================================
# module.py — base class for all transformer modules
# ===========================================================================

class MegatronModule(nn.Module, ABC):
    """Base module class. All transformer components inherit from this."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None,
    ) -> dict:
        """Return state dict with sharding metadata for distributed checkpointing.

        Iterates over all parameters and buffers, attaching TP sharding metadata
        when the parameter is marked as tensor_model_parallel, then recurses into
        all child submodules.  The result is a flat dict keyed by qualified name,
        where each value carries an optional ``tp_shard`` attribute indicating
        which dimension is sharded and on which TP rank.
        """
        sharded_sd: dict = {}

        # ---- Parameters and buffers registered directly on this module ----
        for name, param in self.named_parameters(recurse=False):
            key = f"{prefix}{name}"
            entry: dict = {"param": param, "shape": tuple(param.shape)}
            if getattr(param, "tensor_model_parallel", False):
                entry["tp_shard"] = {
                    "dim": getattr(param, "partition_dim", 0),
                    "stride": getattr(param, "partition_stride", 1),
                }
            if sharded_offsets:
                entry["sharded_offsets"] = sharded_offsets
            sharded_sd[key] = entry

        for name, buf in self.named_buffers(recurse=False):
            key = f"{prefix}{name}"
            sharded_sd[key] = {"param": buf, "shape": tuple(buf.shape)}

        # ---- Recurse into child submodules --------------------------------
        for child_name, child_module in self.named_children():
            child_prefix = f"{prefix}{child_name}."
            if isinstance(child_module, MegatronModule):
                sharded_sd.update(
                    child_module.sharded_state_dict(
                        prefix=child_prefix,
                        sharded_offsets=sharded_offsets,
                        metadata=metadata,
                    )
                )
            else:
                # Plain nn.Module – collect its state dict with prefix
                for pname, param in child_module.named_parameters():
                    key = f"{child_prefix}{pname}"
                    sharded_sd[key] = {"param": param, "shape": tuple(param.shape)}

        return sharded_sd


# ===========================================================================
# Helper: RMSNorm / LayerNorm factory
# ===========================================================================

def _build_norm(config: TransformerConfig, hidden_size: Optional[int] = None) -> nn.Module:
    """Build the norm module specified by config.normalization."""
    size = hidden_size if hidden_size is not None else config.hidden_size
    eps = config.layernorm_epsilon
    if config.normalization == "RMSNorm":
        return nn.RMSNorm(size, eps=eps)
    elif config.normalization == "LayerNorm":
        return nn.LayerNorm(size, eps=eps)
    else:
        raise ValueError(f"Unknown normalization: {config.normalization!r}")


# ===========================================================================
# Helper: SP / TP world-size queries (safe when dist not initialised)
# ===========================================================================

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


# ===========================================================================
# attention.py
# ===========================================================================

class Attention(MegatronModule, ABC):
    """Base attention class.

    AutoSP integration: when sequence_parallel is enabled, the input
    sequence is already partitioned across SP ranks. The attention
    computation handles the local chunk and uses A2A for KV exchange.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: str = "causal",
    ) -> None:
        super().__init__(config)
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type

        assert config.kv_channels is not None
        assert config.num_attention_heads is not None
        assert config.num_query_groups is not None

        self.hidden_size_per_attention_head: int = config.kv_channels
        self.num_attention_heads: int = config.num_attention_heads
        self.num_query_groups: int = config.num_query_groups

        # Sizes for the full (un-TP-split) projections
        self.query_projection_size: int = config.kv_channels * config.num_attention_heads
        self.kv_projection_size: int = config.kv_channels * config.num_query_groups

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        inference_params: Optional[object] = None,
    ) -> torch.Tensor:
        ...


class SelfAttention(Attention):
    """Standard multi-head self-attention with QKV projection.

    Layout: [seq, batch, hidden] throughout (Megatron convention).

    AutoSP integration:
      When config.sequence_parallel is True the *input* hidden_states
      arrive as a sequence-parallel shard of shape
      [seq/sp_size, batch, hidden].  After the output projection the
      result is gathered back to the full sequence before returning.
      The all-to-all (A2A) pattern redistributes tokens so that each
      SP rank owns a contiguous block of the sequence during attention.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
    ) -> None:
        super().__init__(config, layer_number, attn_mask_type="causal")

        tp_size = _get_tp_world_size()

        # ---- QKV projection ------------------------------------------------
        # Combined Q+K+V linear: output dim = num_heads*head_dim + 2*num_kv_heads*head_dim
        qkv_out_dim = self.query_projection_size + 2 * self.kv_projection_size
        # Per-TP-rank slice of the output dimension
        self.qkv_out_dim_per_tp = qkv_out_dim // tp_size
        self.q_size_per_tp = self.query_projection_size // tp_size
        self.kv_size_per_tp = self.kv_projection_size // tp_size

        # Number of heads per TP partition
        self.num_heads_per_tp = config.num_attention_heads // tp_size
        self.num_kv_heads_per_tp = max(1, config.num_query_groups // tp_size)

        self.qkv_proj = nn.Linear(
            config.hidden_size,
            self.qkv_out_dim_per_tp,
            bias=config.add_bias_linear,
        )
        # Mark TP-sharded so checkpointing code knows the partition dim
        self.qkv_proj.weight.tensor_model_parallel = True
        self.qkv_proj.weight.partition_dim = 0

        # ---- Core attention -----------------------------------------------
        self.core_attention = DotProductAttention(config, layer_number)

        # ---- Output projection ---------------------------------------------
        # Input: num_heads_per_tp * head_dim; Output: hidden_size
        self.out_proj = nn.Linear(
            self.query_projection_size // tp_size,
            config.hidden_size,
            bias=config.add_bias_linear,
        )
        self.out_proj.weight.tensor_model_parallel = True
        self.out_proj.weight.partition_dim = 1

        # Attention dropout
        self.attn_dropout = nn.Dropout(p=config.attention_dropout)

    # ------------------------------------------------------------------
    # AutoSP helpers
    # ------------------------------------------------------------------

    def _sp_all_to_all_scatter(
        self, hidden_states: torch.Tensor, sp_group
    ) -> torch.Tensor:
        """Scatter sequence dim → head dim for SP before attention.

        Input:  [seq/sp,  batch, hidden]
        Output: [seq,     batch, hidden/sp]

        Uses all-to-all: each rank sends seq/sp tokens, receives from
        all ranks, then transposes so local heads are contiguous.
        """
        sp_size = _get_sp_world_size()
        if sp_size == 1 or sp_group is None:
            return hidden_states

        # [seq/sp, batch, hidden] → [sp, seq/sp/sp, batch, hidden/sp]
        # For simplicity we use a standard all_to_all pattern:
        # scatter along hidden dim, gather along seq dim.
        seq_chunk, batch, hidden = hidden_states.shape
        assert hidden % sp_size == 0, f"hidden {hidden} not divisible by sp_size {sp_size}"
        hidden_per_rank = hidden // sp_size

        # Reshape to [sp, seq_chunk, batch, hidden_per_rank]
        inp = hidden_states.view(seq_chunk, batch, sp_size, hidden_per_rank)
        inp = inp.permute(2, 0, 1, 3).contiguous()  # [sp, seq_chunk, batch, h/sp]

        out_list = [torch.empty_like(inp[0]) for _ in range(sp_size)]
        in_list = list(inp.unbind(0))
        torch.distributed.all_to_all(out_list, in_list, group=sp_group)
        # out_list[i] shape: [seq_chunk, batch, hidden_per_rank]
        out = torch.stack(out_list, dim=0)  # [sp, seq_chunk, batch, h/sp]
        # merge seq dimension: [sp*seq_chunk, batch, h/sp]
        out = out.permute(1, 0, 2, 3).reshape(seq_chunk * sp_size, batch, hidden_per_rank)
        return out

    def _sp_all_to_all_gather(
        self, context: torch.Tensor, sp_group
    ) -> torch.Tensor:
        """Gather head dim → sequence dim for SP after attention.

        Inverse of _sp_all_to_all_scatter.
        Input:  [seq,     batch, hidden/sp]
        Output: [seq/sp,  batch, hidden]
        """
        sp_size = _get_sp_world_size()
        if sp_size == 1 or sp_group is None:
            return context

        total_seq, batch, hidden_per_rank = context.shape
        seq_chunk = total_seq // sp_size

        # [seq, batch, h/sp] → [sp, seq_chunk, batch, h/sp]
        ctx = context.reshape(sp_size, seq_chunk, batch, hidden_per_rank)
        # swap back: [seq_chunk, sp, batch, h/sp]
        ctx = ctx.permute(1, 0, 2, 3).contiguous()

        # Each rank gets seq_chunk tokens, broadcasts its h/sp slice
        inp_list = list(ctx.unbind(1))   # sp tensors of [seq_chunk, batch, h/sp]
        out_list = [torch.empty_like(inp_list[0]) for _ in range(sp_size)]
        torch.distributed.all_to_all(out_list, inp_list, group=sp_group)

        # Reassemble hidden: [seq_chunk, sp, batch, h/sp] → [seq_chunk, batch, hidden]
        out = torch.stack(out_list, dim=1)  # [seq_chunk, sp, batch, h/sp]
        out = out.permute(0, 2, 1, 3).reshape(seq_chunk, batch, hidden_per_rank * sp_size)
        return out

    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        inference_params: Optional[object] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: [seq, batch, hidden]  (or [seq/sp, batch, hidden] in SP mode)
            attention_mask: Optional boolean mask [batch, 1, seq, seq]
            rotary_pos_emb: Optional rotary position embeddings
            inference_params: Optional inference state (unused currently)

        Returns:
            output: same shape as hidden_states
        """
        sp_group = _get_sp_group()
        use_sp = self.config.sequence_parallel and _get_sp_world_size() > 1

        # --- AutoSP A2A scatter: seq/sp → seq, hidden → hidden/sp ----------
        if use_sp:
            hidden_states = self._sp_all_to_all_scatter(hidden_states, sp_group)

        # --- QKV projection -------------------------------------------------
        # hidden_states: [seq, batch, hidden] → mixed_qkv: [seq, batch, qkv_out/tp]
        mixed_qkv = self.qkv_proj(hidden_states)  # [seq, batch, q+2kv per tp]

        seq_len, batch, _ = mixed_qkv.shape
        head_dim = self.hidden_size_per_attention_head

        # Split into Q, K, V
        q, k, v = torch.split(
            mixed_qkv,
            [self.q_size_per_tp, self.kv_size_per_tp, self.kv_size_per_tp],
            dim=-1,
        )

        # Reshape to [seq, batch, num_heads, head_dim]
        q = q.view(seq_len, batch, self.num_heads_per_tp, head_dim)
        k = k.view(seq_len, batch, self.num_kv_heads_per_tp, head_dim)
        v = v.view(seq_len, batch, self.num_kv_heads_per_tp, head_dim)

        # --- Rotary position embeddings (optional) --------------------------
        if rotary_pos_emb is not None:
            q, k = self._apply_rotary_emb(q, k, rotary_pos_emb)

        # --- Core attention -------------------------------------------------
        # DotProductAttention expects [seq, batch, num_heads, head_dim]
        context = self.core_attention(q, k, v, attention_mask)
        # context: [seq, batch, num_heads_per_tp * head_dim]

        # --- Output projection ----------------------------------------------
        output = self.out_proj(context)  # [seq, batch, hidden]

        # All-reduce across TP ranks (RowParallelLinear pattern)
        tp_size = _get_tp_world_size()
        if tp_size > 1:
            try:
                from deepspeed.core.parallel_state import get_tensor_model_parallel_group
                tp_group = get_tensor_model_parallel_group()
                torch.distributed.all_reduce(output, group=tp_group)
            except Exception:
                pass

        # --- AutoSP A2A gather: seq → seq/sp, hidden/sp → hidden -----------
        if use_sp:
            output = self._sp_all_to_all_gather(output, sp_group)

        return output

    @staticmethod
    def _apply_rotary_emb(
        q: torch.Tensor, k: torch.Tensor, rotary_pos_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to Q and K.

        Args:
            q: [seq, batch, num_heads, head_dim]
            k: [seq, batch, num_kv_heads, head_dim]
            rotary_pos_emb: [seq, 1, 1, head_dim] or [seq, batch, 1, head_dim]

        Returns:
            Rotated q and k tensors of the same shapes.
        """
        # rotary_pos_emb shape: [seq, 1, 1, head_dim]
        # Split head_dim into sin/cos halves
        dim = q.shape[-1]
        half = dim // 2
        cos_emb = rotary_pos_emb[..., :half]
        sin_emb = rotary_pos_emb[..., half:]

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x[..., :half], x[..., half:]
            return torch.cat((-x2, x1), dim=-1)

        q_rot = q * cos_emb + rotate_half(q) * sin_emb
        k_rot = k * cos_emb[: k.shape[0]] + rotate_half(k) * sin_emb[: k.shape[0]]
        return q_rot, k_rot


class DotProductAttention(MegatronModule):
    """Scaled dot-product attention with flash attention support.

    Uses torch.nn.functional.scaled_dot_product_attention (SDPA) which
    dispatches to FlashAttention-2 / math attention depending on hardware.

    Tensor layout throughout: [seq, batch, num_heads, head_dim].
    """

    def __init__(self, config: TransformerConfig, layer_number: int) -> None:
        super().__init__(config)
        self.layer_number = max(1, layer_number)
        self.softmax_scale: float = 1.0 / math.sqrt(config.kv_channels)
        self.attn_dropout_p: float = config.attention_dropout
        # For GQA: ratio of query heads to kv heads
        self.num_attention_heads = config.num_attention_heads
        self.num_query_groups = config.num_query_groups

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute scaled dot-product attention.

        Args:
            query: [seq_q, batch, num_heads, head_dim]
            key:   [seq_k, batch, num_kv_heads, head_dim]
            value: [seq_k, batch, num_kv_heads, head_dim]
            attention_mask: Optional additive mask [batch, 1, seq_q, seq_k]
                            or boolean mask (True = mask out).

        Returns:
            context: [seq_q, batch, num_heads * head_dim]
        """
        seq_q, batch, num_heads, head_dim = query.shape
        seq_k = key.shape[0]
        num_kv_heads = key.shape[2]

        # GQA: expand KV heads to match Q heads if needed
        if num_kv_heads != num_heads:
            # Each KV head serves (num_heads // num_kv_heads) query heads
            expand_factor = num_heads // num_kv_heads
            key = key.repeat_interleave(expand_factor, dim=2)    # [sk, b, nh, hd]
            value = value.repeat_interleave(expand_factor, dim=2)

        # Rearrange to [batch, num_heads, seq, head_dim] for SDPA
        # query: [sq, b, nh, hd] → [b, nh, sq, hd]
        q = query.permute(1, 2, 0, 3).contiguous()
        k = key.permute(1, 2, 0, 3).contiguous()
        v = value.permute(1, 2, 0, 3).contiguous()

        # Build causal mask if no mask provided
        is_causal = attention_mask is None

        # Convert additive mask / bool mask for SDPA
        attn_mask_sdpa: Optional[torch.Tensor] = None
        if attention_mask is not None:
            # attention_mask: [batch, 1, sq, sk] (additive float mask or bool)
            if attention_mask.dtype == torch.bool:
                # SDPA expects True = keep (opposite of Megatron convention where
                # True = mask out). Invert here.
                attn_mask_sdpa = ~attention_mask
            else:
                attn_mask_sdpa = attention_mask

        dropout_p = self.attn_dropout_p if self.training else 0.0

        # torch SDPA – dispatches to FlashAttention when available
        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask_sdpa,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=self.softmax_scale,
        )
        # context: [batch, num_heads, seq_q, head_dim]

        # Rearrange back to [seq_q, batch, num_heads * head_dim]
        context = context.permute(2, 0, 1, 3).contiguous()
        context = context.view(seq_q, batch, num_heads * head_dim)
        return context


# ===========================================================================
# mlp.py
# ===========================================================================

class MLP(MegatronModule):
    """MLP with SwiGLU activation.

    Structure: gate_proj + up_proj → SiLU(gate) * up → down_proj

    Both gate_proj and up_proj take hidden_states as input and produce
    ffn_hidden_size outputs each.  down_proj maps ffn_hidden_size back
    to hidden_size.

    When tensor_parallel_size > 1 the up/gate projections are column-
    parallel (output split across TP ranks) and the down projection is
    row-parallel (input split, all-reduce on output).
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)

        tp_size = _get_tp_world_size()
        ffn_per_tp = config.ffn_hidden_size // tp_size

        # Gate and up projections (column-parallel: split output across TP)
        self.gate_proj = nn.Linear(config.hidden_size, ffn_per_tp, bias=config.add_bias_linear)
        self.up_proj = nn.Linear(config.hidden_size, ffn_per_tp, bias=config.add_bias_linear)

        # Down projection (row-parallel: input already split, all-reduce output)
        self.down_proj = nn.Linear(ffn_per_tp, config.hidden_size, bias=config.add_bias_linear)

        # Mark TP-sharded parameters
        for proj in (self.gate_proj, self.up_proj):
            proj.weight.tensor_model_parallel = True
            proj.weight.partition_dim = 0   # output dimension sharded

        self.down_proj.weight.tensor_model_parallel = True
        self.down_proj.weight.partition_dim = 1  # input dimension sharded

        self.activation_func: Callable = config.activation_func

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """SwiGLU forward pass.

        Args:
            hidden_states: [..., hidden_size]

        Returns:
            output: [..., hidden_size]
        """
        # SwiGLU: element-wise product of activated gate and up-projection
        gate = self.gate_proj(hidden_states)          # [..., ffn/tp]
        up = self.up_proj(hidden_states)              # [..., ffn/tp]
        activated = self.activation_func(gate) * up   # SiLU(gate) * up

        # Down projection
        output = self.down_proj(activated)            # [..., hidden]

        # All-reduce across TP ranks (RowParallelLinear pattern)
        tp_size = _get_tp_world_size()
        if tp_size > 1:
            try:
                from deepspeed.core.parallel_state import get_tensor_model_parallel_group
                tp_group = get_tensor_model_parallel_group()
                torch.distributed.all_reduce(output, group=tp_group)
            except Exception:
                pass

        return output


# ===========================================================================
# transformer_layer.py
# ===========================================================================

class TransformerLayer(MegatronModule):
    """Single transformer layer: attention → residual → MLP → residual.

    Uses pre-norm (norm before sub-layer) following LLaMA / Mistral style.

    DES-LOC extension: supports selective activation checkpointing
    controlled per-tier (A6000 stages checkpoint more aggressively
    than H100 stages due to smaller VRAM).
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
    ) -> None:
        super().__init__(config)
        self.layer_number = layer_number

        # Pre-attention normalisation
        self.input_layernorm = _build_norm(config)

        # Self-attention
        self.self_attention = SelfAttention(config, layer_number)

        # Hidden-state dropout after attention
        self.attn_dropout = nn.Dropout(p=config.hidden_dropout)

        # Pre-MLP normalisation
        self.pre_mlp_layernorm = _build_norm(config)

        # MLP
        self.mlp = MLP(config)

        # Hidden-state dropout after MLP
        self.mlp_dropout = nn.Dropout(p=config.hidden_dropout)

        # Whether to apply layernorm after the residual (post-norm) or before
        # (pre-norm, the default). Megatron calls this
        # apply_residual_connection_post_layernorm.
        self.apply_residual_post_layernorm: bool = (
            config.apply_residual_connection_post_layernorm
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        inference_params: Optional[object] = None,
    ) -> torch.Tensor:
        """Forward pass of one transformer layer.

        Args:
            hidden_states: [seq, batch, hidden]
            attention_mask: Optional mask [batch, 1, seq, seq]
            rotary_pos_emb: Optional [seq, 1, 1, head_dim]
            inference_params: Passed through to attention (unused here)

        Returns:
            output: [seq, batch, hidden]
        """
        # ---- Self-attention with pre-norm --------------------------------
        residual = hidden_states

        if self.apply_residual_post_layernorm:
            # post-norm: norm applied to result of residual add
            attn_out = self.self_attention(
                hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                inference_params=inference_params,
            )
            hidden_states = self.input_layernorm(residual + self.attn_dropout(attn_out))
        else:
            # pre-norm (default)
            normed = self.input_layernorm(hidden_states)
            attn_out = self.self_attention(
                normed,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                inference_params=inference_params,
            )
            hidden_states = residual + self.attn_dropout(attn_out)

        # ---- MLP with pre-norm ------------------------------------------
        residual = hidden_states

        if self.apply_residual_post_layernorm:
            mlp_out = self.mlp(hidden_states)
            hidden_states = self.pre_mlp_layernorm(residual + self.mlp_dropout(mlp_out))
        else:
            normed = self.pre_mlp_layernorm(hidden_states)
            mlp_out = self.mlp(normed)
            hidden_states = residual + self.mlp_dropout(mlp_out)

        return hidden_states


# ===========================================================================
# transformer_block.py
# ===========================================================================

class TransformerBlock(MegatronModule):
    """Stack of TransformerLayers with optional final layer norm.

    In PP mode, each rank holds a contiguous subset of layers determined
    by pipeline_layer_split. Supports heterogeneous splits where
    high-VRAM stages hold more layers.

    pipeline_layer_split (List[int] in ModelParallelConfig) specifies
    how many layers each PP stage owns in order. When absent, layers are
    distributed evenly across stages.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)

        # input_tensor is used by PP receive buffer
        self.input_tensor: Optional[torch.Tensor] = None

        # Determine the global layer indices owned by this PP rank
        self._build_layers()

        # Final layer norm (only on the last PP stage)
        self.final_layernorm: Optional[nn.Module] = None
        if self._is_last_pp_stage():
            self.final_layernorm = _build_norm(config)

    # ------------------------------------------------------------------

    def _get_pp_info(self) -> Tuple[int, int]:
        """Return (pp_rank, pp_size) safely.

        Prefers the live distributed process group when initialized.
        Falls back to config values so that unit tests that skip
        ``torch.distributed.init_process_group`` still work correctly
        (rank defaults to 0, size from config).
        """
        pp_size_from_cfg = self.config.pipeline_model_parallel_size
        try:
            from deepspeed.core.parallel_state import (
                get_pipeline_model_parallel_rank,
                get_pipeline_model_parallel_world_size,
            )
            rank = get_pipeline_model_parallel_rank()
            size = get_pipeline_model_parallel_world_size()
            return rank, size
        except Exception:
            # Distributed not initialised – use config for size, rank=0
            return 0, pp_size_from_cfg

    def _is_last_pp_stage(self) -> bool:
        pp_rank, pp_size = self._get_pp_info()
        return pp_rank == pp_size - 1

    def _build_layers(self) -> None:
        """Build TransformerLayer stack for this PP stage.

        Determines which global layer indices (1-based, following Megatron
        convention) belong to this PP rank, then instantiates
        TransformerLayer objects for each.

        Stores:
            self.layers (nn.ModuleList): the layers for this PP stage.
            self._layer_offset (int): global index of the first local layer.
            self._num_local_layers (int): number of layers on this stage.
        """
        config = self.config
        pp_rank, pp_size = self._get_pp_info()
        total_layers = config.num_layers

        # --- Compute layer distribution across PP stages ------------------
        pipeline_layer_split: Optional[List[int]] = config.pipeline_layer_split

        if pipeline_layer_split is not None:
            # Explicit heterogeneous split
            assert len(pipeline_layer_split) == pp_size, (
                f"pipeline_layer_split length {len(pipeline_layer_split)} "
                f"must equal pipeline_parallel_size {pp_size}"
            )
            assert sum(pipeline_layer_split) == total_layers, (
                f"pipeline_layer_split sum {sum(pipeline_layer_split)} "
                f"must equal num_layers {total_layers}"
            )
            offset = sum(pipeline_layer_split[:pp_rank])
            num_local = pipeline_layer_split[pp_rank]
        else:
            # Even split
            assert total_layers % pp_size == 0, (
                f"num_layers {total_layers} must be divisible by "
                f"pipeline_model_parallel_size {pp_size} when pipeline_layer_split is None"
            )
            layers_per_rank = total_layers // pp_size
            offset = pp_rank * layers_per_rank
            num_local = layers_per_rank

        self._layer_offset: int = offset
        self._num_local_layers: int = num_local

        # --- Instantiate layers (layer_number is 1-based globally) --------
        self.layers = nn.ModuleList(
            [
                TransformerLayer(config, layer_number=offset + i + 1)
                for i in range(num_local)
            ]
        )

    # ------------------------------------------------------------------

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Set input tensor for PP receive.

        Called by the pipeline schedule to inject the activation received
        from the previous PP stage via P2P communication.

        Args:
            input_tensor: [seq, batch, hidden] tensor from the previous stage.
        """
        self.input_tensor = input_tensor

    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        inference_params: Optional[object] = None,
    ) -> torch.Tensor:
        """Forward pass through all local layers.

        If self.input_tensor is set (PP receive), it overrides hidden_states.

        Args:
            hidden_states: [seq, batch, hidden]  (ignored when input_tensor set)
            attention_mask: Optional mask
            rotary_pos_emb: Optional rotary embeddings
            inference_params: Passed through to each layer

        Returns:
            output: [seq, batch, hidden]
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


# ===========================================================================
# moe/moe_layer.py (stub)
# ===========================================================================

class MoELayer(MegatronModule):
    """Mixture of Experts layer with router and token dispatcher.

    DES-LOC extension: expert placement strategy considers per-GPU VRAM.

    This implementation provides a functional stub: tokens are routed via
    TopKRouter, dispatched and combined by MoETokenDispatcher, then passed
    through a set of nn.Linear expert sub-networks.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
    ) -> None:
        super().__init__(config)
        self.layer_number = layer_number

        num_experts = config.num_moe_experts
        assert num_experts is not None and num_experts > 0, (
            "TransformerConfig.num_moe_experts must be set for MoELayer"
        )
        self.num_experts: int = num_experts

        # Router (top-2 by default)
        self.router = TopKRouter(config)

        # Token dispatcher
        self.dispatcher = _AllToAllMoETokenDispatcher(config, num_experts=num_experts)

        # Expert feed-forward networks (each is a small MLP)
        # Using SwiGLU structure matching the dense MLP
        ffn_hidden = config.ffn_hidden_size
        hidden = config.hidden_size
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    _SwiGLUExpertLayer(hidden, ffn_hidden, config.add_bias_linear),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """MoE forward pass.

        Args:
            hidden_states: [seq, batch, hidden]

        Returns:
            output: [seq, batch, hidden]
        """
        orig_shape = hidden_states.shape
        # Flatten to [tokens, hidden] for dispatch
        tokens = hidden_states.view(-1, hidden_states.shape[-1])

        # Route tokens
        probs, indices = self.router(tokens)  # [tokens, top_k], [tokens, top_k]

        # Dispatch tokens to experts (sorted by top-1 expert)
        dispatched, token_counts = self.dispatcher._dispatch_with_indices(tokens, probs, indices)

        # Run each expert on its contiguous slice of tokens
        expert_outputs = []
        offset = 0
        for i, expert in enumerate(self.experts):
            count = int(token_counts[i].item())
            expert_in = dispatched[offset: offset + count]
            if count > 0:
                expert_out = expert(expert_in)
            else:
                expert_out = torch.zeros(
                    0, dispatched.shape[-1],
                    dtype=dispatched.dtype, device=dispatched.device,
                )
            expert_outputs.append(expert_out)
            offset += count

        expert_out_all = torch.cat(expert_outputs, dim=0)  # [tokens, hidden]

        # Combine: unsort and weight by router probabilities
        output_flat = self.dispatcher.token_combine(expert_out_all)
        return output_flat.view(orig_shape)


class _SwiGLUExpertLayer(nn.Module):
    """Single expert: SwiGLU MLP (no TP sharding for expert layers)."""

    def __init__(self, hidden_size: int, ffn_hidden_size: int, bias: bool) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_size, ffn_hidden_size, bias=bias)
        self.up = nn.Linear(hidden_size, ffn_hidden_size, bias=bias)
        self.down = nn.Linear(ffn_hidden_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Router(MegatronModule, ABC):
    """Base router for MoE."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)

    @abstractmethod
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route tokens to experts. Returns (dispatch_weights, dispatch_indices)."""
        ...


class TopKRouter(Router):
    """Top-K token routing with auxiliary load-balancing loss.

    Computes per-token expert scores via a linear gate, selects the top-k
    experts, and returns normalised routing weights together with expert
    indices.  An auxiliary load-balancing loss (Switch Transformer style)
    is accumulated in ``self.aux_loss`` for the caller to add to the
    training loss.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        num_experts = config.num_moe_experts
        assert num_experts is not None
        self.num_experts: int = num_experts
        self.top_k: int = 2  # Standard MoE uses top-2 routing
        self.aux_loss_coeff: float = 0.01

        # Gating linear layer: hidden → num_experts (no bias, following Switch)
        self.gate = nn.Linear(config.hidden_size, num_experts, bias=False)

        # Accumulated auxiliary loss (reset each forward pass)
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        Args:
            input: [num_tokens, hidden_size]

        Returns:
            probs:   [num_tokens, top_k]  normalised routing weights
            indices: [num_tokens, top_k]  expert indices (int64)
        """
        # Compute router logits
        logits = self.gate(input)             # [tokens, num_experts]
        scores = F.softmax(logits, dim=-1)   # [tokens, num_experts]

        # Select top-k experts per token
        probs, indices = torch.topk(scores, self.top_k, dim=-1)  # [tokens, top_k]

        # Re-normalise weights so they sum to 1 over selected experts
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-9)

        # Load-balancing auxiliary loss (Switch Transformer, Eq. 4 & 5)
        # f_i = fraction of tokens routed to expert i
        # p_i = mean router probability for expert i
        # L_aux = num_experts * sum_i(f_i * p_i)
        expert_mask = F.one_hot(indices[:, 0], num_classes=self.num_experts).float()
        f = expert_mask.mean(dim=0)           # [num_experts]
        p = scores.mean(dim=0)                # [num_experts]
        self.aux_loss = (self.aux_loss_coeff * self.num_experts * (f * p).sum())

        return probs, indices


class MoETokenDispatcher(ABC):
    """Dispatches tokens to experts and combines results."""

    def __init__(self, config: TransformerConfig) -> None:
        self.config = config

    @abstractmethod
    def token_dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def token_combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...


class _AllToAllMoETokenDispatcher(MoETokenDispatcher):
    """Concrete token dispatcher using local sort-and-gather (no inter-rank comm).

    For a single-node / EP=1 setup, each token is sent to its chosen
    experts on the same rank.  Tokens are sorted by their top-1 expert
    assignment so that each expert's slice is contiguous in memory.

    This stub is intentionally simple and single-rank; a production
    implementation would use all-to-all for expert parallelism.

    Implements the ``MoETokenDispatcher`` abstract interface where both
    ``token_dispatch`` and ``token_combine`` take (hidden_states, probs).
    Routing indices are stored internally between the two calls.
    """

    def __init__(self, config: TransformerConfig, num_experts: int) -> None:
        super().__init__(config)
        self.num_experts = num_experts
        # State filled by token_dispatch and consumed by token_combine
        self._sorted_order: Optional[torch.Tensor] = None
        self._dispatch_probs: Optional[torch.Tensor] = None
        self._dispatch_indices: Optional[torch.Tensor] = None
        self._token_counts: Optional[torch.Tensor] = None

    def _dispatch_with_indices(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Internal dispatch that also accepts routing indices.

        Args:
            hidden_states: [num_tokens, hidden]
            probs:   [num_tokens, top_k]
            indices: [num_tokens, top_k]  int64 expert indices

        Returns:
            dispatched:    [num_tokens, hidden]  tokens sorted by expert
            token_counts:  [num_experts]         number of tokens per expert
        """
        self._dispatch_indices = indices
        self._dispatch_probs = probs

        # Sort by top-1 expert index so each expert's slice is contiguous
        top1 = indices[:, 0]
        sorted_order = torch.argsort(top1, stable=True)
        self._sorted_order = sorted_order

        dispatched = hidden_states[sorted_order]

        token_counts = torch.zeros(
            self.num_experts, dtype=torch.long, device=hidden_states.device
        )
        for e in range(self.num_experts):
            token_counts[e] = (top1 == e).sum()
        self._token_counts = token_counts

        return dispatched, token_counts

    # ------------------------------------------------------------------
    # Abstract interface implementations (MoETokenDispatcher API)
    # ------------------------------------------------------------------

    def token_dispatch(
        self, hidden_states: torch.Tensor, probs: torch.Tensor
    ) -> torch.Tensor:
        """Dispatch tokens using stored routing indices.

        The routing indices must have been set via
        ``_dispatch_with_indices`` (called by MoELayer.forward) before
        this method is invoked through the public interface.

        Args:
            hidden_states: [num_tokens, hidden]
            probs: [num_tokens, top_k] router weights

        Returns:
            dispatched: [num_tokens, hidden] tokens sorted by expert
        """
        assert self._dispatch_indices is not None, (
            "Routing indices not set; call _dispatch_with_indices first."
        )
        dispatched, _ = self._dispatch_with_indices(
            hidden_states, probs, self._dispatch_indices
        )
        return dispatched

    def token_combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Combine (unsort + weight) expert outputs back to original order.

        Args:
            hidden_states: [num_tokens, hidden] sorted expert outputs

        Returns:
            output: [num_tokens, hidden] in original token order, weighted
                    by top-1 router probability
        """
        assert self._sorted_order is not None, "Must call token_dispatch first."
        assert self._dispatch_probs is not None

        num_tokens = hidden_states.shape[0]
        hidden_sz = hidden_states.shape[-1]

        output = torch.zeros(
            num_tokens, hidden_sz,
            dtype=hidden_states.dtype, device=hidden_states.device,
        )
        output[self._sorted_order] = hidden_states

        # Weight by top-1 router probability
        top1_probs = self._dispatch_probs[:, 0:1]  # [tokens, 1]
        output = output * top1_probs

        # Reset state
        self._sorted_order = None
        self._dispatch_probs = None
        self._dispatch_indices = None
        self._token_counts = None

        return output
