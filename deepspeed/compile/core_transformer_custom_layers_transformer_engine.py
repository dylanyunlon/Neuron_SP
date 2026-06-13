# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M1525: Megatron ea97be889 — Always return two values from linear layer
# Source: megatron/core/transformer/custom_layers/transformer_engine.py
#         megatron/core/transformer/attention.py
#
# Mapping:
#   megatron/core/transformer/custom_layers/transformer_engine.py
#     -> deepspeed/compile/core_transformer_custom_layers_transformer_engine.py
#
# Changes ported (M1525):
#   TELinear.__init__:
#     - Added bias/skip_bias_add params; self.te_return_bias = skip_bias_add and bias
#     - Passes bias and return_bias=self.te_return_bias to TE super().__init__
#   TELinear.forward (new):
#     - Always returns (output, bias_or_None) regardless of skip_bias_add arg
#     - If te_return_bias: TE already returns tuple, pass through
#     - Else: return out, None
#
# ---------------------------------------------------------------------------
# M1527: Megatron 5b6fb1ecd — Rename return_bias back to skip_bias_add
# Source: megatron/core/transformer/custom_layers/transformer_engine.py
#
# Changes ported:
#   TELinear.__init__: param return_bias -> skip_bias_add
#   self.te_return_bias = skip_bias_add and bias  (logic unchanged)
#
# 20% adaptation: adds print('[M1527]') diagnostic markers and 鲁迅式 docstring.
# ---------------------------------------------------------------------------

print('[M1425]')
print('[M1525]')
print('[M1527]')
import torch
import transformer_engine as te

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.parallel_state import get_tensor_model_parallel_group
from megatron.core.tensor_parallel import get_cuda_rng_tracker


class TELayerNorm(te.pytorch.module.LayerNorm):
    """
    Wrapper for the Transformer-Engine's `LayerNorm`.
    """
    def __init__(self,
                 hidden_size: int,
                 eps: float = 1e-5,
                 sequence_parallel: bool = False,
                 **kwargs):
        super().__init__(
            hidden_size=hidden_size,
            eps=eps,
            sequence_parallel=sequence_parallel
        )


class TELinear(te.pytorch.module.Linear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `RowParallelLinear` layer.

    M1525: forward() 始终返回 (output, bias_or_None) 两个值，
    不管 skip_bias_add 参数如何设置 —— 学鲁迅先生所言：
    "横眉冷对千夫指，俯首甘为孺子牛"，接口一律平等，
    调用方无需猜测返回几个值。

    M1527: 参数由 return_bias 正名为 skip_bias_add，与 Megatron 上游保持一致。
    鲁迅又云：名不正则言不顺，言不顺则事不成——参数命名亦然。
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 config: TransformerConfig,
                 parallel_mode: str,
                 bias: bool = True,
                 skip_bias_add: bool = False,
                 **kwargs):
        self.config = config

        # M1527: skip_bias_add replaces return_bias (rename only, logic identical).
        # TE 在 bias=False、skip_bias_add=True 时返回零长 Tensor，
        # 我们更喜欢 None。所以此时让 TE 不返回 bias，
        # 由我们自己补 None——这样 forward 始终两个返回值，
        # 省得上层代码还要猜。
        self.te_return_bias = skip_bias_add and bias
        print(f'[M1527] TELinear.__init__ bias={bias} skip_bias_add={skip_bias_add} '
              f'te_return_bias={self.te_return_bias}')

        super().__init__(
            in_features=input_size,
            out_features=output_size,
            sequence_parallel=self.config.sequence_parallel,
            fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
            tp_group=get_tensor_model_parallel_group(),
            tp_size=self.config.tensor_model_parallel_size,
            get_rng_state_tracker=get_cuda_rng_tracker,
            init_method=self.config.init_method,
            params_dtype=self.config.params_dtype,
            parallel_mode=parallel_mode,
            bias=bias,
            return_bias=self.te_return_bias,
            **kwargs
        )

    def forward(self, x):
        out = super().forward(x)

        # TE 只在 return_bias=True 时返回元组，否则返回单个 Tensor；
        # 我们无论如何都返回两个值，调用方一律用 output, _ = layer(x)。
        if self.te_return_bias:
            print(f'[M1527] TELinear.forward te_return_bias=True, returning tuple as-is')
            return out
        print(f'[M1527] TELinear.forward te_return_bias=False, appending None bias')
        return out, None


class TEColumnParallelLinear(TELinear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `ColumnParallelLinear` layer.
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 config: TransformerConfig,
                 **kwargs):
        self.config = config
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            config=self.config,
            parallel_mode="column",
            **kwargs
        )


class TERowParallelLinear(TELinear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `RowParallelLinear` layer.
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 config: TransformerConfig,
                 **kwargs):
        self.config = config
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            config=self.config,
            parallel_mode="row",
            **kwargs
        )


# ---------------------------------------------------------------------------
# M1700: Megatron efc434ccc — Rename TECoreAttention to TEDotProductAttention
# Source: megatron/core/transformer/custom_layers/transformer_engine.py (NVIDIA/Megatron-LM commit efc434ccc)
#
# Changes ported:
#   - class TECoreAttention -> class TEDotProductAttention.
#   - Wrapper name now matches underlying te.pytorch.transformer.DotProductAttention.
#
# 20% adaptation (鲁迅式迁移):
#   - 名不正则言不顺，TECoreAttention 终归其正名 TEDotProductAttention。
#   - Adds print('[M1700]') diagnostic marker.
# ---------------------------------------------------------------------------
class TEDotProductAttention(te.pytorch.transformer.DotProductAttention):
    """
    Wrapper for the Transformer-Engine's `DotProductAttention` layer that also
    has "flash attention" enabled.
    [M1700] Renamed from TECoreAttention to TEDotProductAttention.
    """
    def __init__(self,
                 config: TransformerConfig,
                 layer_number: int = 1,
                 attn_mask_type: AttnMaskType = AttnMaskType.padding,
                 **kwargs):
        self.config = config
        super().__init__(
            num_attention_heads=self.config.num_attention_heads,
            kv_channels=self.config.kv_channels,
            attention_dropout=self.config.attention_dropout,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type.name,
            sequence_parallel=self.config.sequence_parallel,
            tp_size=self.config.tensor_model_parallel_size,
            get_rng_state_tracker=get_cuda_rng_tracker,
            tp_group=get_tensor_model_parallel_group(),
            **kwargs
        )
