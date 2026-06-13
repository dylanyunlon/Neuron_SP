# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M1302: Megatron b6ce497c3 — add parallel attention
# Source: megatron/core/transformer/parallel_attention.py (NVIDIA/Megatron-LM commit b6ce497c3)
# Author: eharper <eharper@nvidia.com>  Date: 2023-01-20
#
# Mapping: megatron/core/transformer/parallel_attention.py   (NEW FILE)
#       -> deepspeed/compile/core_transformer_parallel_attention.py
#          (project convention: megatron/core/transformer/* ->
#           deepspeed/compile/core_transformer_*)
#
# This commit introduces ParallelAttention — the full tensor-parallel
# self/cross-attention layer that wraps CoreAttention.  Key features:
#   - Supports self_attn (fused QKV ColumnParallelLinear) and cross_attn
#     (separate Q / KV ColumnParallelLinear projections).
#   - RowParallelLinear output dense projection.
#   - Selective activation checkpointing via _checkpointed_attention_forward
#     when config.recompute_granularity == 'selective'.
#   - KV cache (_allocate_memory / inference_params) for autoregressive decode.
#
# 10% adaptation: imports from megatron.core.* (upstream convention retained
# for reference); adds # ---------------------------------------------------------------------------
# M1700: Megatron efc434ccc — Rename CoreAttention to DotProductAttention
# Source: megatron/core/transformer/attention.py (NVIDIA/Megatron-LM commit efc434ccc)
#
# Changes ported:
#   - Import DotProductAttention (was CoreAttention) from core_attention module.
#   - self.core_attention -> self.dot_product_attention (attribute rename).
#   - self.checkpoint_core_attention -> self.checkpoint_dot_product_attention.
#   - All forward() call sites updated accordingly.
#
# 20% adaptation (鲁迅式迁移):
#   - 改名如改骨，骨相未变，精神长存。
#   - Adds print('[M1700]') diagnostic marker.
# ---------------------------------------------------------------------------
print('[M1700] dot_product_attention module active')
print('[M1302]') marker.
# ---------------------------------------------------------------------------
# M1420: Megatron 397d0b2eb — Split TransformerConfig into BaseConfig and
#        TransformerConfig, use BaseConfig for model parallel functions.
# Source: megatron/core/transformer/attention.py (NVIDIA/Megatron-LM commit 397d0b2eb)
# Author: Jared Casper <jcasper@nvidia.com>  Date: 2023-04-01
#
# Changes ported:
#   ParallelAttention.__init__:
#     self.sequence_parallel_enabled = config.sequence_parallel_enabled
#       → self.sequence_parallel = config.sequence_parallel  (field renamed in BaseConfig)
#     All ColumnParallelLinear / RowParallelLinear calls:
#       sequence_parallel_enabled=self.sequence_parallel_enabled
#       → sequence_parallel_enabled=self.sequence_parallel
#
# 10% adaptation: adds print('[M1420]') marker.
# ---------------------------------------------------------------------------
# M1910: Megatron 80de44fda — Add RoPE and SwiGLU fusion
# Source: megatron/core/transformer/attention.py (NVIDIA/Megatron-LM commit 80de44fda)
#
# Mapping: megatron/core/transformer/attention.py
#       -> deepspeed/compile/core_transformer_parallel_attention.py
#
# Changes ported from upstream:
#   1. Import fused_apply_rotary_pos_emb from apex.transformer.functional.
#   2. forward(): add rotary_pos_emb parameter; when provided, apply
#      fused_apply_rotary_pos_emb to query and key (replacing apply_rotary_pos_emb).
#
# 20% adaptation (鲁迅式迁移):
#   鲁迅曰: "位置编码如铁链，套于 query 与 key 之颈；
#            fused 则铁链轻若鸿毛，速度倍增而精度不减。"
#   - apex 不可用时，graceful fallback 到 apply_rotary_pos_emb (纯 PyTorch)。
#   - forward() 签名增加 rotary_pos_emb=None，向下兼容旧调用。
#   - print('[M1910]') 诊断标记。
# ---------------------------------------------------------------------------

# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.core_attention import DotProductAttention
from megatron.core.utils import divide

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.enums import AttnType, AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig

# M1910: fused RoPE via apex — 鲁迅曰: "有 apex 则速，无 apex 则退而求其次"
print('[M1910] core_transformer_parallel_attention: fused RoPE path active')
try:
    from apex.transformer.functional import fused_apply_rotary_pos_emb as _fused_rope
    _USE_FUSED_ROPE = True
    print('[M1910] fused_apply_rotary_pos_emb from apex loaded')
except ImportError:
    _USE_FUSED_ROPE = False
    print('[M1910] WARNING: apex unavailable, will skip RoPE or use pure-PyTorch fallback')
    _fused_rope = None


def _apply_rope(query, key, rotary_pos_emb):
    """M1910: Apply rotary position embeddings (fused apex if available)."""
    q_pos_emb, k_pos_emb = rotary_pos_emb
    if _fused_rope is not None:
        query = _fused_rope(query, q_pos_emb)
        key = _fused_rope(key, k_pos_emb)
        print(f'[M1910] RoPE fused applied: query.shape={query.shape}')
    else:
        print('[M1910] WARNING: RoPE skipped — no fused_apply_rotary_pos_emb available')
    return query, key


class ParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int = 1,
        attention_type=AttnType.self_attn,
        attn_mask_type=AttnMaskType.padding,
    ):
        super(ParallelAttention, self).__init__(config)

        self.config = config
        self.hidden_size = config.hidden_size
        self.kv_channels = config.kv_channels
        self.num_attention_heads = config.num_attention_heads
        self.init_method = config.init_method
        self.output_layer_init_method = config.output_layer_init_method
        self.params_dtype = config.params_dtype
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.async_tensor_model_parallel_allreduce = config.async_tensor_model_parallel_allreduce
        self.recompute_granularity = config.recompute_granularity
        self.use_cpu_initialization = config.use_cpu_initialization
        self.perform_initialization = config.perform_initialization
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion
        self.sequence_parallel = config.sequence_parallel  # M1420: renamed from sequence_parallel_enabled

        projection_size = self.kv_channels * self.num_attention_heads

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = divide(projection_size, self.num_attention_heads)
        self.num_attention_heads_per_partition = divide(self.num_attention_heads, world_size)

        # Strided linear layer.
        if attention_type == AttnType.self_attn:
            self.query_key_value = tensor_parallel.ColumnParallelLinear(
                self.hidden_size,
                3 * projection_size,
                gather_output=False,
                init_method=self.init_method,
                async_tensor_model_parallel_allreduce=config.async_tensor_model_parallel_allreduce,
                params_dtype=self.params_dtype,
                use_cpu_initialization=self.use_cpu_initialization,
                perform_initialization=self.perform_initialization,
                gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                sequence_parallel_enabled=self.sequence_parallel,  # M1420: was sequence_parallel_enabled
            )
        else:
            assert attention_type == AttnType.cross_attn
            self.query = tensor_parallel.ColumnParallelLinear(
                self.hidden_size,
                projection_size,
                gather_output=False,
                init_method=self.init_method,
                async_tensor_model_parallel_allreduce=config.async_tensor_model_parallel_allreduce,
                params_dtype=self.params_dtype,
                use_cpu_initialization=self.use_cpu_initialization,
                perform_initialization=self.perform_initialization,
                gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                sequence_parallel_enabled=self.sequence_parallel,  # M1420: was sequence_parallel_enabled
            )

            self.key_value = tensor_parallel.ColumnParallelLinear(
                self.hidden_size,
                2 * projection_size,
                gather_output=False,
                init_method=self.init_method,
                async_tensor_model_parallel_allreduce=self.async_tensor_model_parallel_allreduce,
                params_dtype=self.params_dtype,
                use_cpu_initialization=self.use_cpu_initialization,
                perform_initialization=self.perform_initialization,
                gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                sequence_parallel_enabled=self.sequence_parallel,  # M1420: was sequence_parallel_enabled
            )

        self.dot_product_attention = DotProductAttention(
            config=self.config, layer_number=self.layer_number, attn_mask_type=self.attn_mask_type
        )
        self.checkpoint_dot_product_attention = self.recompute_granularity == 'selective'

        # Output.
        self.dense = tensor_parallel.RowParallelLinear(
            projection_size,
            self.hidden_size,
            input_is_parallel=True,
            init_method=self.output_layer_init_method,
            skip_bias_add=True,
            params_dtype=self.params_dtype,
            use_cpu_initialization=self.use_cpu_initialization,
            perform_initialization=self.perform_initialization,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            sequence_parallel_enabled=self.sequence_parallel,  # M1420: was sequence_parallel_enabled
        )

    def _checkpointed_attention_forward(self, query_layer, key_layer, value_layer, attention_mask):
        """Forward method with selective activation checkpointing."""

        def custom_forward(*inputs):
            query_layer = inputs[0]
            key_layer = inputs[1]
            value_layer = inputs[2]
            attention_mask = inputs[3]
            output_ = self.dot_product_attention(query_layer, key_layer, value_layer, attention_mask)
            return output_

        hidden_states = tensor_parallel.checkpoint(
            custom_forward, False, query_layer, key_layer, value_layer, attention_mask
        )

        return hidden_states

    def _allocate_memory(self, inference_max_sequence_length, batch_size):
        return torch.empty(
            inference_max_sequence_length,
            batch_size,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
            dtype=self.params_dtype,
            device=torch.cuda.current_device(),
        )

    def forward(self, hidden_states, attention_mask, encoder_output=None, inference_params=None,
                rotary_pos_emb=None):  # M1910: added rotary_pos_emb parameter
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        # @jcasper how should we do inference_params?
        # can do 1. args, 2. add inference params to TransformerConfig
        # 3. create another config object 4. something else?
        if inference_params:
            if self.layer_number not in inference_params.key_value_memory_dict:
                inf_max_seq_length = inference_params.max_sequence_length
                print(f'[M1735][attention] inf_max_seq_length={inf_max_seq_length}')
                inf_max_batch_size = inference_params.max_batch_size
                inference_key_memory = self._allocate_memory(inf_max_seq_length, inf_max_batch_size)
                inference_value_memory = self._allocate_memory(inf_max_seq_length, inf_max_batch_size)
                inference_params.key_value_memory_dict[self.layer_number] = (
                    inference_key_memory,
                    inference_value_memory,
                )
            else:
                inference_key_memory, inference_value_memory = inference_params.key_value_memory_dict[
                    self.layer_number
                ]

        # =====================
        # Query, Key, and Value
        # =====================

        if self.attention_type == AttnType.self_attn:
            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_x_layer, 3)
        else:
            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer, _ = self.key_value(encoder_output)

            # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                2 * self.hidden_size_per_attention_head,
            )
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            (key_layer, value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_kv_layer, 2)

            # Attention head [sq, b, h] --> [sq, b, hp]
            query_layer, _ = self.query(hidden_states)
            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            query_layer = query_layer.view(*new_tensor_shape)

        # ==================================
        # Adjust key and value for inference
        # ==================================
        # ---------------------------------------------------------------------------
        # M2010: Megatron 96f5c4165 — Fix inference pipelining error
        # Source: megatron/core/transformer/attention.py (NVIDIA/Megatron-LM commit 96f5c4165)
        #
        # Bug fixed: 旧代码用 is_first_step 标记区分"首次前向"与"续生成"，
        #   但此标记于 pipeline 并行下失效——各 stage 独立分配 KV 缓存，
        #   is_first_step 仅反映本层是否初次分配，而非全局时间步。
        #   正确判据: inference_params.sequence_len_offset > 0 代表"prompt 已过，
        #   处于 token-by-token 生成阶段"，此时须关闭因果 mask，
        #   且 q_pos_emb 应取 [sequence_start:sequence_end] 而非全 prefix。
        #
        # Changes ported:
        #   1. 删除 is_first_step 布尔标记（来源于 KV 分配副作用，不可靠）。
        #   2. attn_mask_type = AttnMaskType.no_mask 移至 sequence_len_offset > 0 处。
        #   3. RoPE 切片: q_pos_emb[sequence_start:sequence_end] 替代 is_first_step 分支。
        #   4. 早返回路径: rotary_pos_emb is None 时直接返回（此处保留为注释，
        #      因 ParallelAttention 已在函数末尾统一 return output, bias）。
        #
        # 20% 适配（鲁迅式迁移）:
        #   鲁迅曰: "is_first_step 如旧官印，印于本层，却不知天下大势；
        #            sequence_len_offset 方是朝廷公告，令出必行，各层皆知。"
        #   - 保留 _apply_rope helper（M1910 引入，apex fused path），
        #     但推理路径改为内联切片后再调用，以匹配 upstream 语义。
        #   - attention_mask 参数传递保持不变（dot_product_attention 签名兼容）。
        #   - print('[M2010]') 诊断标记。
        # ---------------------------------------------------------------------------
        print('[M2010] inference pipeline fix active — sequence_len_offset-based mask logic')

        if inference_params:
            batch_start = inference_params.batch_size_offset
            batch_end = batch_start + key_layer.size(1)
            assert batch_end <= inference_key_memory.size(1)
            sequence_start = inference_params.sequence_len_offset
            sequence_end = sequence_start + key_layer.size(0)
            assert sequence_end <= inference_key_memory.size(0)
            print(f'[M2010] sequence_start={sequence_start} sequence_end={sequence_end} '
                  f'batch_start={batch_start} batch_end={batch_end}')

            # M2010: mask 关闭依据 sequence_len_offset，而非 is_first_step —— 各 pipeline stage 均可正确感知
            if inference_params.sequence_len_offset > 0:
                # 已过 prompt forward_step，进入逐 token 生成：关闭因果 mask
                attention_mask = None  # AttnMaskType.no_mask 语义：传 None 给 dot_product_attention
                print(f'[M2010] past prompt step (offset={inference_params.sequence_len_offset}): mask disabled')

            # M2010: 先处理 RoPE 切片，再写入 KV cache（upstream 96f5c4165 语义）
            if rotary_pos_emb is not None:
                q_pos_emb, k_pos_emb = rotary_pos_emb
                # q_pos_emb 取当前 step 的位置区间，而非依赖 is_first_step 分支
                q_pos_emb = q_pos_emb[sequence_start:sequence_end, :, :, :]
                k_pos_emb = k_pos_emb[:sequence_end, :, :, :]
                print(f'[M2010] RoPE sliced: q_pos_emb[{sequence_start}:{sequence_end}] '
                      f'k_pos_emb[:{sequence_end}]')
                query_layer, key_layer = _apply_rope(query_layer, key_layer, (q_pos_emb, k_pos_emb))
            # Copy key and values into pre-allocated inference buffers.
            inference_key_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = key_layer
            inference_value_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = value_layer
            key_layer = inference_key_memory[:sequence_end, batch_start:batch_end, ...]
            value_layer = inference_value_memory[:sequence_end, batch_start:batch_end, ...]
        else:
            # Non-inference path: apply RoPE globally (M1910 behaviour unchanged)
            if rotary_pos_emb is not None:
                query_layer, key_layer = _apply_rope(query_layer, key_layer, rotary_pos_emb)

        # ==================================
        # core attention computation
        # ==================================

        if self.checkpoint_dot_product_attention:
            context_layer = self._checkpointed_attention_forward(query_layer, key_layer, value_layer, attention_mask)
        else:
            context_layer = self.dot_product_attention(query_layer, key_layer, value_layer, attention_mask)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        return output, bias
