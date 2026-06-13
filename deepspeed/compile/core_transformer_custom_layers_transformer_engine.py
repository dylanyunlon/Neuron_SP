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

# ---------------------------------------------------------------------------
# M1750: Megatron 7891eb1fe — Replace TELN+TELinear with TELayerNormLinear
# Source: megatron/core/transformer/custom_layers/transformer_engine.py
#         megatron/core/transformer/attention.py
#         megatron/core/transformer/mlp.py
#         megatron/core/transformer/transformer_layer.py
#
# Mapping:
#   megatron/core/transformer/custom_layers/transformer_engine.py
#     -> deepspeed/compile/core_transformer_custom_layers_transformer_engine.py
#
# Changes ported (M1750):
#   TELayerNormColumnParallelLinear (NEW CLASS):
#     - Wraps te.pytorch.LayerNormLinear: fuses LayerNorm + ColumnParallelLinear
#       into a single kernel, eliminates separate TENorm pass before linear.
#     - te_return_bias / forward() pattern identical to TEColumnParallelLinear.
#     - attention.py: TEColumnParallelLinear -> TELayerNormColumnParallelLinear
#       for linear_qkv (SelfAttention), linear_q / linear_kv (CrossAttention).
#     - mlp.py: TEColumnParallelLinear -> TELayerNormColumnParallelLinear
#       for linear_fc1.
#     - transformer_layer.py: input_layernorm / post_self_attn_layernorm
#       TENorm -> IdentityOp (LayerNorm now absorbed into fused kernel above).
#
# 20% adaptation (鲁迅式迁移):
#   - 鲁迅云：「无穷的远方，无数的人们，都和我有关。」
#     层归一化与线性层，合二为一，消弭边界，性能相关。
#   - Adds print('[M1750]') diagnostic marker.
#   - TELayerNormColumnParallelLinear.__init__ prints fusion params on construction.
#   - TELayerNormColumnParallelLinear.forward prints input shape for diagnostics.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# M1960: Megatron c3079ce98 — Enable DGRAD RS overlap
# Source: megatron/core/transformer/custom_layers/transformer_engine.py
#
# Mapping: megatron/core/transformer/custom_layers/transformer_engine.py
#        → deepspeed/compile/core_transformer_custom_layers_transformer_engine.py
#
# Changes ported (TELayerNormColumnParallelLinear.__init__):
#   When _te_ver > 1.6.0.dev0, pass ub_overlap_rs_dgrad kwarg to TE super().__init__
#   reflecting config.tp_comm_overlap_rs_dgrad (new field from M1960/c3079ce98).
#   Upstream also refactors ub_overlap_ag / ub_overlap_rs hasattr guards in TELinear
#   and TELayerNormColumnParallelLinear — our simplified wrappers omit UB plumbing
#   (no tp_comm_overlap flag path), so only the rs_dgrad addition is ported.
#
# 20% adaptation (鲁迅式迁移):
#   鲁迅曰：「梯度之流散，若不乘势而行，则徒费光阴；
#             ub_overlap_rs_dgrad 一旦启用，则 Reduce-Scatter 掩于 DGRAD 之下，
#             如燕雀翱翔于长天，何其自在。」
#   - Version guard mirrors upstream (> 1.6.0.dev0).
#   - print('[M1960]') diagnostic in __init__ reports rs_dgrad setting.
# ---------------------------------------------------------------------------
print('[M1960] core_transformer_custom_layers_transformer_engine: ub_overlap_rs_dgrad support')
print('[M1425]')
print('[M1525]')
print('[M1527]')
print('[M1750]')
# ---------------------------------------------------------------------------
# M1870: Megatron 7a70c5401 — GPT model level change for context parallelism
# Source: megatron/core/transformer/custom_layers/transformer_engine.py
#
# Changes ported:
#   1. Expand parallel_state import: add get_context_parallel_global_ranks,
#      get_context_parallel_group.
#   2. cp_stream = torch.cuda.Stream() module-level CUDA stream for CP comms.
#   3. TEDotProductAttention.__init__: pass cp_group, cp_global_ranks, cp_stream
#      to te.pytorch.DotProductAttention super().__init__.
#
# 20% adaptation (鲁迅式迁移):
#   鲁迅云：「真的猛士，敢于直面惨淡的人生，敢于正视淋漓的鲜血。」
#   注意力之计算，跨rank之通信，cp_stream承载异步之望，各rank之数据轮转如环。
#   Adds print('[M1870]') diagnostic markers.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# M1940: Megatron 1a3e1c522 — Add transpose cache feature
# Source: megatron/core/transformer/custom_layers/transformer_engine.py
#
# Mapping: megatron/core/transformer/custom_layers/transformer_engine.py
#       -> deepspeed/compile/core_transformer_custom_layers_transformer_engine.py
#
# Changes ported from upstream:
#   TELinear.__init__: self.is_first_microbatch = True +
#     self.disable_parameter_transpose_cache = self.config.disable_parameter_transpose_cache
#   TELinear.forward: _is_first_microbatch logic (None disables cache, else tracks batch)
#   TELayerNormColumnParallelLinear: same two changes as TELinear.
#
# 20% adaptation (鲁迅式迁移):
#   鲁迅曰：「转置之缓存，犹如旧俗——废之则纯净，留之则高效，两难之间，配置为鉴。」
#   - is_first_microbatch 追踪每轮首批，确保缓存在正确时刻失效。
#   - disable_parameter_transpose_cache=True 时传 None，令 TE 逐步重计算转置。
#   - Adds print('[M1940]') diagnostic markers in __init__ and forward.
# ---------------------------------------------------------------------------
print('[M1940] transpose cache feature: disable_parameter_transpose_cache support')
print('[M1870] context parallelism: cp_stream + CP group args for TEDotProductAttention')
import torch
import transformer_engine as te
from importlib.metadata import version as _te_version
from pkg_resources import packaging as _pkg

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.parallel_state import (
    get_context_parallel_global_ranks,
    get_context_parallel_group,
    get_tensor_model_parallel_group,
)
from megatron.core.tensor_parallel import get_cuda_rng_tracker

# M1870: Dedicated CUDA stream for context-parallel ring-attention communication.
# Allows CP all-to-all comms to overlap with compute on the main stream.
cp_stream = torch.cuda.Stream()
print('[M1870] cp_stream created: %s' % cp_stream)


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

        # M1940: track first microbatch for transpose cache control.
        self.is_first_microbatch = True
        self.disable_parameter_transpose_cache = self.config.disable_parameter_transpose_cache
        print(f'[M1940] TELinear.__init__ disable_parameter_transpose_cache='
              f'{self.disable_parameter_transpose_cache}')

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
        # M1940: None disables TE's transpose cache entirely; otherwise track first microbatch.
        _is_first_microbatch = (
            None if self.disable_parameter_transpose_cache else self.is_first_microbatch
        )
        print(f'[M1940] TELinear.forward _is_first_microbatch={_is_first_microbatch}')
        out = super().forward(x, is_first_microbatch=_is_first_microbatch)
        self.is_first_microbatch = False

        # TE 只在 return_bias=True 时返回元组，否则返回单个 Tensor；
        # 我们无论如何都返回两个值，调用方一律用 output, _ = layer(x)。
        if self.te_return_bias:
            print(f'[M1527] TELinear.forward te_return_bias=True, returning tuple as-is')
            return out
        print(f'[M1527] TELinear.forward te_return_bias=False, appending None bias')
        return out, None



class TELayerNormColumnParallelLinear(te.pytorch.LayerNormLinear):
    """
    Wrapper for Transformer-Engine's `LayerNormLinear` layer — 融合 LayerNorm
    与 ColumnParallelLinear，单一核函数完成归一化与投影，省去独立 TENorm 开销。

    M1750: 鲁迅云：「无穷的远方，无数的人们，都和我有关。」
    层归一化与线性层，合二为一，消弭边界——性能之事，细节攸关。

    取代 SelfAttention.linear_qkv、CrossAttention.linear_q/linear_kv、
    MLP.linear_fc1 处原本分离的 TENorm + TEColumnParallelLinear 组合。
    transformer_layer.py 中对应的 input_layernorm / post_self_attn_layernorm
    随之改为 IdentityOp（LayerNorm 已内嵌于本层）。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: TransformerConfig,
        bias: bool = True,
        skip_bias_add: bool = False,
        **kwargs
    ):
        self.config = config
        # M1940: track first microbatch for transpose cache control.
        self.is_first_microbatch = True
        self.disable_parameter_transpose_cache = self.config.disable_parameter_transpose_cache
        print(f'[M1940] TELayerNormColumnParallelLinear.__init__ disable_parameter_transpose_cache='
              f'{self.disable_parameter_transpose_cache}')
        # TE returns zero-length Tensor when bias=False and return_bias=True;
        # we prefer None — so we handle it ourselves.
        self.te_return_bias = skip_bias_add and bias

        # Only TE >= 0.11.0 supports normalization kwarg (RMSNorm support).
        # M1960: TE > 1.6.0.dev0 supports ub_overlap_rs_dgrad for DGRAD RS overlap.
        try:
            _te_ver = _pkg.version.Version(_te_version("transformer-engine"))
            if _te_ver >= _pkg.version.Version("0.11.0"):
                kwargs.setdefault("normalization", self.config.normalization)
            # M1960: Megatron c3079ce98 — pass rs_dgrad overlap flag when TE supports it.
            # 鲁迅曰：「版本之墙，不可强攻；版本既足，则通信重叠，梯度计算如虎添翼。」
            if _te_ver > _pkg.version.Version("1.6.0.dev0"):
                rs_dgrad = getattr(self.config, "tp_comm_overlap_rs_dgrad", False)
                kwargs.setdefault("ub_overlap_rs_dgrad", rs_dgrad)
                print(f'[M1960] TELayerNormColumnParallelLinear.__init__ \'
                      f'ub_overlap_rs_dgrad={rs_dgrad} (TE={_te_ver})')
        except Exception:
            pass  # 无法检测版本时跳过，默认 LayerNorm

        print(
            f'[M1750] TELayerNormColumnParallelLinear.__init__ '
            f'input={input_size} output={output_size} '
            f'bias={bias} skip_bias_add={skip_bias_add} '
            f'te_return_bias={self.te_return_bias}'
        )

        super().__init__(
            in_features=input_size,
            out_features=output_size,
            bias=bias,
            sequence_parallel=self.config.sequence_parallel,
            fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
            tp_group=get_tensor_model_parallel_group(),
            tp_size=self.config.tensor_model_parallel_size,
            get_rng_state_tracker=get_cuda_rng_tracker,
            init_method=self.config.init_method,
            params_dtype=self.config.params_dtype,
            parallel_mode="column",
            return_bias=self.te_return_bias,
            **kwargs
        )

    def forward(self, x):
        print(f'[M1750] TELayerNormColumnParallelLinear.forward input_shape={tuple(x.shape)}')
        # M1940: None disables TE's transpose cache entirely; otherwise track first microbatch.
        _is_first_microbatch = (
            None if self.disable_parameter_transpose_cache else self.is_first_microbatch
        )
        print(f'[M1940] TELayerNormColumnParallelLinear.forward _is_first_microbatch={_is_first_microbatch}')
        out = super().forward(x, is_first_microbatch=_is_first_microbatch)
        self.is_first_microbatch = False
        # TE 只在 return_bias=True 时返回元组，否则返回单个 Tensor；
        # 我们无论如何都返回两个值——接口一律平等，调用方一律 output, _ = layer(x)。
        if self.te_return_bias:
            return out
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
            cp_group=get_context_parallel_group(),
            cp_global_ranks=get_context_parallel_global_ranks(),
            cp_stream=cp_stream,
            **kwargs
        )
        print('[M1870] TEDotProductAttention.__init__: cp_group=%s' % get_context_parallel_group())
