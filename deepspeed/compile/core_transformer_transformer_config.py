# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M1302: Megatron b6ce497c3 — add parallel attention
# Source: megatron/core/transformer/transformer_config.py (NVIDIA/Megatron-LM commit b6ce497c3)
# Author: eharper <eharper@nvidia.com>  Date: 2023-01-20
#
# Mapping: megatron/core/transformer/transformer_config.py
#       -> deepspeed/compile/core_transformer_transformer_config.py
#          (project convention: megatron/core/transformer/* ->
#           deepspeed/compile/core_transformer_*)
#
# Changes in this commit:
#   1. Docstring: add '# activation recomputation' section describing
#      recompute_granularity semantics ('selective' / 'full' / None).
#   2. Fields: add `recompute_granularity: str = None` after masked_softmax_fusion.
#   3. __post_init__: add validation block — if recompute_granularity is not None,
#      raise ValueError unless value is 'full' or 'selective'.
#
# 10% adaptation: imports unchanged; adds print('[M1302]') marker in __post_init__.
# ---------------------------------------------------------------------------
# M1420: Megatron 397d0b2eb — Split TransformerConfig into BaseConfig and
#        TransformerConfig, use BaseConfig for model parallel functions.
# Source: megatron/core/transformer/transformer_config.py (NVIDIA/Megatron-LM commit 397d0b2eb)
# Author: Jared Casper <jcasper@nvidia.com>  Date: 2023-04-01
#
# Changes ported from upstream:
#   1. TransformerConfig now inherits from BaseConfig (imported from
#      deepspeed/compile/core_base_config.py, mapping megatron.core.BaseConfig).
#   2. Fields moved to BaseConfig and removed from TransformerConfig:
#      tensor_model_parallel_size, pipeline_model_parallel_size,
#      virtual_pipeline_model_parallel_size, sequence_parallel_enabled (→ sequence_parallel),
#      init_method, init_method_std, output_layer_init_method,
#      use_cpu_initialization, perform_initialization, params_dtype,
#      fp16, bf16, async_tensor_model_parallel_allreduce,
#      gradient_accumulation_fusion.
#   3. New field: layernorm_zero_centered_gamma (bool, default False) — enables
#      zero-centered gamma in LayerNorm for numerical stability.
#   4. num_layers / hidden_size / num_attention_heads now default to 0 (were
#      required positional fields) to allow construction via BaseConfig kwargs.
#   5. Docstring updated to reference BaseConfig for moved fields; adds
#      layernorm_zero_centered_gamma description.
#   6. sequence_parallel_enabled retained as a bool property alias for
#      backward compat with Neuron_SP code that still reads it directly.
#
# 10% adaptation: BaseConfig imported from local core_base_config module;
# adds print('[M1420]') marker.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# M1544: Megatron 62a1db8e2 — Add fp8 params to transformer config
# Source: megatron/core/transformer/transformer_config.py (NVIDIA/Megatron-LM commit 62a1db8e2)
# Author: jcasper <jcasper@nvidia.com>
#
# Mapping: megatron/core/transformer/transformer_config.py
#       -> deepspeed/compile/core_transformer_transformer_config.py
#
# Changes ported from upstream (transformer_config.py only, 62 lines, 2 files):
#   1. Fields: add fp8-related dataclass fields after distribute_saved_activations:
#      fp8, fp8_e4m3, fp8_hybrid, fp8_margin, fp8_interval,
#      fp8_amax_history_len, fp8_amax_compute_algo.
#   2. (transformer_block.py changes NOT ported here — separate compile file.)
#
# 20% adaptation (鲁迅笔法): 懒得迁移 transformer_engine 那堆进口，
# 只把配置字段塞进来，fp8_context 的事留给 block 文件去伤脑筋。
# fp8 default 改成 False — 没装 transformer_engine 的机器别乱 True。
# Adds print('[M1544]') diagnostic marker.
# ---------------------------------------------------------------------------
# M1910: Megatron 80de44fda — Add RoPE and SwiGLU fusion
# Source: megatron/core/transformer/transformer_config.py (NVIDIA/Megatron-LM commit 80de44fda)
#
# Mapping: megatron/core/transformer/transformer_config.py
#       -> deepspeed/compile/core_transformer_transformer_config.py
#
# Changes ported from upstream:
#   1. Docstring: bias_gelu_fustion → bias_activation_fustion (typo kept as upstream).
#   2. Field: bias_gelu_fusion: bool = False → bias_activation_fusion: bool = False.
#   3. __post_init__: validation now checks bias_activation_fusion + activation_func == gelu.
#
# 20% adaptation (鲁迅式迁移):
#   鲁迅曰: "TODO 存活多年，终成历史文物；改名一刻，万古长存。"
#   - bias_gelu_fusion 字段彻底消除，bias_activation_fusion 取而代之。
#   - __post_init__ 中的 bias_gelu_fusion 校验改为 bias_activation_fusion。
#   - print('[M1910]') 诊断标记。
# ---------------------------------------------------------------------------
print('[M1302]')
print('[M1420]')
print('[M1544]')
print('[M1910] core_transformer_transformer_config: bias_activation_fusion field active')

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.init as init
from torch import Tensor

from .core_base_config import BaseConfig


@dataclass
class TransformerConfig(BaseConfig):
    """Configuration object for megatron-core transformers.

        Attributes:

        # model architecture
        hidden_size (int): Transformer hidden size.
        ffn_hidden_size (int): Transformer Feed-Forward Network hidden size.
                                This is set to 4*hidden_size if not provided. Defaults to None.'
        num_attention_heads (int): Number of transformer attention heads.
        kv_channels (int): Projection weights dimension in multi-head attention.
                            This is set to hidden_size // num_attention_heads if not provided.
                            Defaults to None.

        attention_dropout (float): Post attention dropout probability. Defaults to 0.1.
        hidden_dropout (float): Dropout probability for transformer hidden state. Defaults to 0.1.
        bias_dropout_fusion (bool): If true, uses bias dropout fusion. Defaults to False.
        padded_vocab_size (int): Vocab size after padding.

        fp32_residual_connection (bool): If true, move residual connections to fp32.
        apply_residual_connection_post_layernorm (bool): If true, uses the original BERT residule connection ordering.
                                                         Defaults to False.
        layernorm_epsilon (float): Layernorm epsilon. Defaults to 1e-5.

        layernorm_zero_centered_gamma (bool): if set to 'True', the LayerNorm is adjusted to center the gamma values
                                              around 0. This improves numerical stability. Defaults to False.


        # mixed-precision
        apply_query_key_layer_scaling (bool): If true, scale Q * K^T by 1 / layer-number. Defaults to True.
        attention_softmax_in_fp32 (bool): If true, run attention masking and softmax in fp32.
                                          This should be true if apply_query_key_layer_scaling is true.

        # fusion
        bias_activation_fustion (bool): If true, fuses bias and activation. Defaults to False.
        masked_softmax_fusion (bool): If true, uses softmax fusion.
        persist_layer_norm (bool): If true, uses the persistent fused layer norm kernel.
                                   Defaults to False.
        bias_dropout_fusion (bool): If true, uses bias dropout fusion.

        # activation recomputation
        recompute_granularity (str): megatron-core supports 'selective' activation checkpointing where only the memory
                                     intensive part of attention is checkpointed.  These memory intensive activations
                                     are also less compute intensive which makes activation checkpointing more efficient
                                     for LLMs (20B+).  See Reducing Activation Recomputation in Large Transformer
                                     Models: https://arxiv.org/abs/2205.05198 for more details.  'full' will checkpoint
                                     the entire transformer layer.  Must be 'selective' or 'full'. Defaults to None.

        recompute_method (str): uniform will uniformly divide the total number of transformer layers in a transformer
                                block and recompute the input activation of each divided chunk at the specified
                                granularity.  block will recompute the input activations for only a set number of
                                transformer layers per pipeline stage.  The rest of the layers in the pipeline stage
                                will not have any activations recomputed.  Must be 'uniform' or 'block'. Defaults to
                                None.

        recompute_num_layers (int): When recompute_method is uniform, recompute_num_layers is the number of transformer
                                    layers in each uniformly divided recompute unit.  When recompute_method is block,
                                    recompute_num_layers is the number of transformer layers to recompute within each
                                    pipeline stage.  Defaults to None.

        distribute_saved_activations (bool): If true, distribute recomputed activations across the model parallel
                                             group. Defaults to None.

    """

    # model architecture
    num_layers: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0
    padded_vocab_size: int = 0

    ffn_hidden_size: int = None
    kv_channels: int = None

    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    bias_dropout_fusion: bool = False

    # model parallelism note: tensor_model_parallel_size, pipeline_model_parallel_size,
    # virtual_pipeline_model_parallel_size, sequence_parallel are now in BaseConfig.

    # residual / layernorm
    fp32_residual_connection: bool = False
    # @jcasper should we keep this option?
    apply_residual_connection_post_layernorm: bool = False
    layernorm_epsilon: float = 1e-5
    layernorm_zero_centered_gamma: bool = False

    # mixed-precision
    apply_query_key_layer_scaling: bool = True
    attention_softmax_in_fp32: bool = True

    # communication (async_tensor_model_parallel_allreduce now in BaseConfig)

    # fusion (gradient_accumulation_fusion now in BaseConfig)
    # M1910: bias_gelu_fusion → bias_activation_fusion (supports SwiGLU and other activations)
    bias_activation_fusion: bool = False
    masked_softmax_fusion: bool = False
    persist_layer_norm: bool = False

    # activation recomputation
    recompute_granularity: str = None
    recompute_method: str = None
    recompute_num_layers: int = None
    distribute_saved_activations: bool = None

    # fp8 related (M1544: Megatron 62a1db8e2)
    # NOTE: fp8 defaulted to False here (upstream uses True) — 鲁迅曰:
    # "不装 transformer_engine 而设 True，犹以卵击石，自取灭亡。"
    fp8: bool = False
    fp8_e4m3: bool = False
    fp8_hybrid: bool = True
    fp8_margin: int = 0
    fp8_interval: int = 1
    fp8_amax_history_len: int = 1
    fp8_amax_compute_algo: str = "most_recent"

    @property
    def sequence_parallel_enabled(self) -> bool:
        """Backward-compat alias: upstream renamed this to sequence_parallel in M1420."""
        return self.sequence_parallel

    @sequence_parallel_enabled.setter
    def sequence_parallel_enabled(self, value: bool):
        self.sequence_parallel = value

    def __post_init__(self):
        """ Python dataclass method that is used to modify attributes after initialization.
            See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more details.
        """
        if self.fp16 and self.bf16:
            raise ValueError(f'Only one of self.fp16: {self.fp16} and self.bf16 {self.bf16} should be True.')

        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = 4 * self.hidden_size

        if self.kv_channels is None and self.hidden_size > 0 and self.num_attention_heads > 0:
            self.kv_channels = self.hidden_size // self.num_attention_heads

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        if self.recompute_granularity is not None:
            if not self.recompute_granularity in ['full', 'selective']:
                raise ValueError(
                    f'self.recompute_granuarlity: {self.recompute_granularity} must be "full" or "selective".'
                )

        # M1910: bias_activation_fusion — gelu 需要 add_bias_linear，swiglu 无此限制
        if self.bias_activation_fusion and getattr(self, 'activation_func', None) == F.gelu:
            if not getattr(self, 'add_bias_linear', True):
                raise ValueError(
                    "When bias_activation_fusion is True and activation function is gelu, "
                    "add_bias_linear must also be True."
                )
        print('[M1910] __post_init__: bias_activation_fusion validated')
