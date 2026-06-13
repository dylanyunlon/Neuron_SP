# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M1302: Megatron b6ce497c3 — add parallel attention
# Source: megatron/core/transformer/mlp.py (NVIDIA/Megatron-LM commit b6ce497c3)
# Author: eharper <eharper@nvidia.com>  Date: 2023-01-20
#
# Mapping: megatron/core/transformer/mlp.py
#       -> deepspeed/compile/core_transformer_mlp.py
#          (project convention: megatron/core/transformer/* ->
#           deepspeed/compile/core_transformer_*)
#
# Changes in this commit:
#   1. __init__: add blank line + `self.config = config` before self.hidden_size,
#      so that the config object is accessible on the instance for downstream use.
#
# 10% adaptation: imports from megatron.core.* (upstream convention retained
# for reference); adds print('[M1302]') marker.
# ---------------------------------------------------------------------------
# M1420: Megatron 397d0b2eb — Split TransformerConfig into BaseConfig and
#        TransformerConfig, use BaseConfig for model parallel functions.
# Source: megatron/core/transformer/mlp.py (NVIDIA/Megatron-LM commit 397d0b2eb)
# Author: Jared Casper <jcasper@nvidia.com>  Date: 2023-04-01
#
# Changes ported:
#   ParallelMLP.__init__:
#     self.sequence_parallel_enabled = config.sequence_parallel_enabled
#       → self.sequence_parallel = config.sequence_parallel  (field renamed in BaseConfig)
#     ColumnParallelLinear / RowParallelLinear kwarg:
#       sequence_parallel_enabled=self.sequence_parallel_enabled
#       → sequence_parallel_enabled=self.sequence_parallel  (kwarg name unchanged in
#         mpu_layers; only the config attribute was renamed)
#
# 10% adaptation: adds print('[M1420]') marker.
# ---------------------------------------------------------------------------
# M1910: Megatron 80de44fda — Add RoPE and SwiGLU fusion
# Source: megatron/core/transformer/mlp.py (NVIDIA/Megatron-LM commit 80de44fda)
#
# Mapping: megatron/core/transformer/mlp.py
#       -> deepspeed/compile/core_transformer_mlp.py
#
# Changes ported from upstream:
#   1. Import bias_swiglu_impl, swiglu_impl from fused_bias_swiglu.
#   2. forward(): replace bias_gelu_fusion branch with bias_activation_fusion,
#      supporting both gelu (existing) and swiglu (new glu-family activation).
#      SwiGLU path: chunk intermediate_parallel in half on last dim, then
#      call bias_swiglu_impl (with bias) or swiglu_impl (without).
#
# 20% adaptation (鲁迅式迁移):
#   鲁迅曰: "改名如改骨，旧字虽亡，新义长存。bias_gelu_fusion 已死，
#            bias_activation_fusion 当立，方能容纳 SwiGLU 之新秀。"
#   - bias_gelu_fusion → bias_activation_fusion 字段名跟随上游。
#   - 增加 try/except 保护 SwiGLU import，防止无 apex 环境崩溃。
#   - print('[M1910]') 诊断标记。
# ---------------------------------------------------------------------------

print('[M1302]')
print('[M1420]')
print('[M1910] core_transformer_mlp: SwiGLU bias_activation_fusion active')

import torch
import torch.nn.functional as F

from megatron.core import tensor_parallel
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

# M1910: SwiGLU fused impl — 鲁迅曰: "融合之道，在乎一 chunk，分而治之。"
try:
    from .core_fusions_fused_bias_swiglu import bias_swiglu_impl, swiglu_impl
    print('[M1910] bias_swiglu_impl and swiglu_impl imported successfully')
except ImportError as _e:
    print(f'[M1910] WARNING: SwiGLU fusion unavailable: {_e}')
    bias_swiglu_impl = None
    swiglu_impl = None


class ParallelMLP(MegatronModule):
    """
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.

    We use the following notation: 
     h: hidden size
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(self, config: TransformerConfig):
        super(ParallelMLP, self).__init__(config)

        self.config = config
        self.hidden_size = config.hidden_size
        self.ffn_hidden_size = config.ffn_hidden_size
        self.init_method = config.init_method
        self.output_layer_init_method = config.output_layer_init_method
        self.use_cpu_initialization = config.use_cpu_initialization
        self.perform_initialization = config.perform_initialization
        # M1910: bias_gelu_fusion → bias_activation_fusion (supports SwiGLU too)
        self.bias_activation_fusion = config.bias_activation_fusion
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion
        self.sequence_parallel = config.sequence_parallel  # M1420: renamed from sequence_parallel_enabled
        self.params_dtype = config.params_dtype
        self.async_tensor_model_parallel_allreduce = config.async_tensor_model_parallel_allreduce

        # Project to 4h.
        # @jcasper should we change the name dense_h_to_4h here?
        self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
            self.hidden_size,
            self.ffn_hidden_size,
            gather_output=False,
            init_method=self.init_method,
            skip_bias_add=True,
            async_tensor_model_parallel_allreduce=self.async_tensor_model_parallel_allreduce,
            params_dtype=self.params_dtype,
            use_cpu_initialization=self.use_cpu_initialization,
            perform_initialization=self.perform_initialization,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            sequence_parallel_enabled=self.sequence_parallel,  # M1420: was self.sequence_parallel_enabled
        )

        self.activation_func = F.gelu

        # @jcasper should we remove openai_gelu?
        # if args.openai_gelu:
        #     self.activation_func = openai_gelu
        # @jcasper should we remove onnx_safe?
        # elif args.onnx_safe:
        #     self.activation_func = erf_gelu

        # Project back to h.
        # @jcasper should we change the name here?
        self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
            self.ffn_hidden_size,
            self.hidden_size,
            input_is_parallel=True,
            init_method=self.output_layer_init_method,
            skip_bias_add=True,
            params_dtype=self.params_dtype,
            use_cpu_initialization=self.use_cpu_initialization,
            perform_initialization=self.perform_initialization,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            sequence_parallel_enabled=self.sequence_parallel,  # M1420: was self.sequence_parallel_enabled
        )

    def forward(self, hidden_states):

        # [s, b, 4 * h/p]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if self.bias_activation_fusion:
            # M1910: bias_activation_fusion 支持 gelu 和 swiglu 两种激活
            if self.activation_func == F.gelu:
                intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
            else:
                # SwiGLU: chunk intermediate 和 bias 各一半，再 fused apply
                x = torch.chunk(intermediate_parallel, 2, dim=-1)
                print(f'[M1910] SwiGLU fusion path: x[0].shape={x[0].shape}')
                if bias_parallel is not None:
                    bias = torch.chunk(bias_parallel, 2, dim=-1)
                    intermediate_parallel = bias_swiglu_impl(x[0], bias[0], x[1], bias[1])
                else:
                    intermediate_parallel = swiglu_impl(x[0], x[1])
        else:
            intermediate_parallel = self.activation_func(intermediate_parallel + bias_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias
