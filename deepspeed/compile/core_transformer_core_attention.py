# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M1302: Megatron b6ce497c3 — add parallel attention
# Source: megatron/core/transformer/core_attention.py (NVIDIA/Megatron-LM commit b6ce497c3)
# Author: eharper <eharper@nvidia.com>  Date: 2023-01-20
#
# Mapping: megatron/core/transformer/core_attention.py
#       -> deepspeed/compile/core_transformer_core_attention.py
#          (project convention: megatron/core/transformer/* ->
#           deepspeed/compile/core_transformer_*)
#
# Changes in this commit:
#   1. __init__: add blank line + `self.config = config` before self.fp16,
#      so that the config object is accessible on the instance (needed by
#      ParallelAttention which passes config=self.config to CoreAttention).
#
# 10% adaptation: imports from deepspeed.compile.* where megatron.core.* would
# be used; adds print('[M1302]') marker.
# ---------------------------------------------------------------------------
# M1420: Megatron 397d0b2eb — Split TransformerConfig into BaseConfig and
#        TransformerConfig, use BaseConfig for model parallel functions.
# Source: megatron/core/transformer/core_attention.py (NVIDIA/Megatron-LM commit 397d0b2eb)
# Author: Jared Casper <jcasper@nvidia.com>  Date: 2023-04-01
#
# Changes ported:
#   CoreAttention.__init__: config.sequence_parallel_enabled → config.sequence_parallel
#     (field was renamed to sequence_parallel in BaseConfig; TransformerConfig now
#      exposes sequence_parallel_enabled as a backward-compat property alias but
#      forward code should use the canonical name).
#
# 10% adaptation: adds print('[M1420]') marker.
# ---------------------------------------------------------------------------

print('[M1302]')
print('[M1420]')
# ---------------------------------------------------------------------------
# M1556: Megatron 8360677cc — Add GroupQueryCoreAttention class
# Source: megatron/model/transformer.py (NVIDIA/Megatron-LM commit 8360677cc)
# Author: Megatron-LM team  Date: 2023
#
# Mapping: megatron/model/transformer.py  GroupQueryCoreAttention
#       -> deepspeed/compile/core_transformer_core_attention.py
#          GroupQueryCoreAttention
#          (project convention: megatron/model/* ->
#           deepspeed/compile/core_transformer_*)
#
# Changes in this commit (upstream):
#   1. Refactored GQA logic out of CoreAttention.forward() conditional branches
#      into a standalone GroupQueryCoreAttention subclass.
#   2. CoreAttention.forward() now handles MHA only (no group_query_attention flag).
#   3. GroupQueryCoreAttention.__init__ computes num_query_groups_per_partition
#      (handles world_size >= num_query_groups edge case → 1 partition).
#   4. GroupQueryCoreAttention.forward reshapes Q as [b*ng, np/ng*sq, hn] and
#      K/V as [b*ng, sk, hn] for grouped batched-matmul via torch.baddbmm/bmm.
#
# 20% adaptation (鲁迅式迁移):
#   - Upstream uses megatron.model imports (mpu, get_args); here we use
#     megatron.core (parallel_state, divide) to match Neuron_SP conventions.
#   - config.num_query_groups replaces args.num_query_groups.
#   - parallel_state.get_global_memory_buffer() replaces mpu.get_global_memory_buffer().
#   - Adds print('[M1556]') diagnostic marker (鲁迅: 沉默是金，但诊断是命).
# ---------------------------------------------------------------------------
print('[M1556]')

import math

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.utils import divide
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.utils import attention_mask_func
from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax


class CoreAttention(MegatronModule):
    """ 
    Region where selective activation recomputation is applied.
    This region is memory intensive but less compute intensive which
    makes activation checkpointing more efficient for LLMs (20B+).
    See Reducing Activation Recomputation in Large Transformer Models: https://arxiv.org/abs/2205.05198 for more details.

    We use the following notation: 
     h: hidden size
     n: number of attention heads
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(self, config: TransformerConfig, layer_number: int = 1, attn_mask_type=AttnMaskType.padding):
        super(CoreAttention, self).__init__(config)

        self.config = config
        self.fp16 = config.fp16
        self.bf16 = config.bf16
        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        self.sequence_parallel = config.sequence_parallel  # M1420: renamed from sequence_parallel_enabled
        self.masked_softmax_fusion = config.masked_softmax_fusion
        self.attention_dropout = config.attention_dropout

        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(config.num_attention_heads, world_size)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            input_in_fp16=self.fp16,
            input_in_bf16=self.bf16,
            attn_mask_type=self.attn_mask_type,
            scaled_masked_softmax_fusion=self.masked_softmax_fusion,
            mask_func=attention_mask_func,
            softmax_in_fp32=self.attention_softmax_in_fp32,
            scale=coeff,
        )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(self.attention_dropout)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):

        # ===================================
        # Raw attention scores. [b, n/p, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0), key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(
            (output_size[0] * output_size[1], output_size[2], output_size[3]), query_layer.dtype, "mpu"
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3))

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class GroupQueryCoreAttention(CoreAttention):
    """GQA core attention: Groups of query heads share key/value heads.

    Ported from Megatron-LM commit 8360677cc (megatron/model/transformer.py).
    Adapts the upstream mpu / get_args pattern to Neuron_SP's megatron.core /
    TransformerConfig conventions.

    Upstream shape notation (retained verbatim for traceability):
      b  = batch size
      np = num_attention_heads_per_partition
      ng = num_query_groups_per_partition
      sq = source sequence length
      sk = target sequence length
      hn = hidden size per attention head
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # M1556: num_query_groups_per_partition — mirror of upstream logic.
        # When num_query_groups < world_size every partition holds 1 group.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        num_query_groups = self.config.num_query_groups
        print(f'[M1556] GroupQueryCoreAttention.__init__: '
              f'num_query_groups={num_query_groups}, world_size={world_size}')
        if num_query_groups >= world_size:
            self.num_query_groups_per_partition = divide(num_query_groups, world_size)
        else:
            self.num_query_groups_per_partition = 1
        print(f'[M1556] num_query_groups_per_partition={self.num_query_groups_per_partition}')

    def forward(self, query_layer, key_layer, value_layer, attention_mask):

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))

        print(f'[M1556] GroupQueryCoreAttention.forward: output_size={output_size}, '
              f'num_query_groups_per_partition={self.num_query_groups_per_partition}')

        # [sq, b, np, hn] -> [b * ng, np/ng * sq, hn]
        query_layer = query_layer.permute([1, 2, 0, 3]).reshape(
            output_size[0] * self.num_query_groups_per_partition,
            int(output_size[1] / self.num_query_groups_per_partition) * output_size[2],
            -1)

        # [sk, b, 1*ng, hn] -> [b * ng, sk, hn]
        key_layer = key_layer.permute([1, 2, 0, 3]).reshape(
            output_size[0] * self.num_query_groups_per_partition,
            output_size[3],
            -1)

        # preallocate input tensor: [b * ng, np/ng * sq, sk]
        matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(
            (output_size[0] * self.num_query_groups_per_partition,
             int(output_size[1] / self.num_query_groups_per_partition) * output_size[2],
             output_size[3]),
            query_layer.dtype, "mpu")

        # Raw attention scores. [b * ng, np/ng * sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer,                    # [b * ng, np/ng * sq, hn]
            key_layer.transpose(1, 2),      # [b * ng, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

        # Drop entire tokens (standard Transformer paper behaviour).
        if not self.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # context layer shape: [b, np, sq, hn]
        context_output_size = (value_layer.size(1), value_layer.size(2),
                               query_layer.size(0), value_layer.size(3))

        # change view [sk, b, ng, hn] --> [sk, b * ng, hn]
        value_layer = value_layer.view(
            value_layer.size(0), context_output_size[0] * context_output_size[1], -1)

        # change view from [b, np, sq, sk] --> [b * ng, np/ng * sq, sk]
        attention_probs = attention_probs.view(
            output_size[0] * self.num_query_groups_per_partition,
            int(output_size[1] / self.num_query_groups_per_partition) * output_size[2],
            -1)

        # matmul: [b * ng, np/ng * sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(
            output_size[0], output_size[1], output_size[2], -1)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        print(f'[M1556] GroupQueryCoreAttention.forward done: context_layer.shape={context_layer.shape}')
        return context_layer
