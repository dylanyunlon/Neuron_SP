# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M1890: Megatron e9bce9db4 — Transformer block checkpointed_forward handles context
# Source: megatron/core/transformer/transformer_block.py (NVIDIA/Megatron-LM commit e9bce9db4)
# Author: Lawrence McAfee <lmcafee@nvidia.com>  Date: 2023-10-10
#
# Mapping: megatron/core/transformer/transformer_block.py
#       -> deepspeed/compile/core_transformer_transformer_block.py
#          (project convention: megatron/core/transformer/* ->
#           deepspeed/compile/core_transformer_*)
#
# Changes ported from upstream (TransformerBlock._checkpointed_forward):
#
#   1. custom_forward signature — replaces the old *args/**kwargs catch-all with
#      explicit positional parameters:
#        (hidden_states, attention_mask, context, context_mask, rotary_pos_emb,
#         *args, **kwargs)
#      This makes context visible to the checkpoint machinery as a proper tensor
#      argument rather than a packed positional, enabling correct gradient tracking.
#
#   2. layer() call inside custom_forward — switches from the old dict-expansion
#      pattern (layer(x_, *args, **{**kwargs, "context": context_})) to explicit
#      keyword arguments:
#        layer(hidden_states=..., attention_mask=..., context=...,
#              context_mask=..., rotary_pos_emb=...)
#      Return is now unpacked as (hidden_states, context) consistently.
#
#   3. tensor_parallel.checkpoint() call sites (both 'uniform' and 'block' paths)
#      — argument order corrected to match the new custom_forward signature:
#        hidden_states, attention_mask, context, context_mask, rotary_pos_emb
#      (previously context and attention_mask were swapped).
#
#   4. Retained the commented-out old implementation wrapped in # >>> / # <<<
#      as the upstream commit does, for archaeological traceability.
#
# 20% 鲁迅式 adaptation:
#   - Neuron_SP does not import tensor_parallel from megatron.core; checkpoint
#     calls are stubbed with a local _checkpoint helper that falls back to direct
#     execution (no gradient checkpointing) with a print warning — sufficient for
#     benchmarking contexts where memory is not the constraint.
#   - distribute_saved_activations is hardcoded to False (Neuron_SP runs SP not TP).
#   - Added print('[M1890]') boot marker and per-call diagnostics so grep can
#     confirm the new forward path is active in benchmark logs.
# ---------------------------------------------------------------------------

print('[M1890] core_transformer_transformer_block loaded — checkpointed_forward handles context')


def _checkpoint(func, distribute_saved_activations, *args):
    """Minimal checkpoint shim for Neuron_SP benchmark context.

    In full Megatron this would be tensor_parallel.checkpoint().
    Here we run the function directly (no recompute) since Neuron_SP
    benchmarks prioritise throughput measurement over memory saving.
    A warning is printed so benchmark logs remain auditable.
    """
    print(
        '[M1890][_checkpoint] WARNING: activation recompute disabled '
        f'(distribute_saved_activations={distribute_saved_activations}); '
        f'calling func directly with {len(args)} tensor args'
    )
    return func(*args)


class CheckpointedForwardMixin:
    """Mixin that grafts the M1890 _checkpointed_forward onto a TransformerBlock.

    Usage in Neuron_SP model code::

        class MyTransformerBlock(CheckpointedForwardMixin, ...):
            pass

    The mixin expects self._get_layer(i) and self.config to exist,
    matching the Megatron TransformerBlock interface.
    """

    def _checkpointed_forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        context=None,
        context_mask=None,
    ):
        """Forward with activation checkpointing — M1890 context-aware version.

        鲁迅曰：以往之 *args 传参，如旧式大家庭之不言家规，人皆知其存，
        却无人能道其详；今以显名参数，开门见山，方知 context 安身何处。
        """
        print(
            f'[M1890][_checkpointed_forward] '
            f'hidden_states.shape={getattr(hidden_states, "shape", "?")} '
            f'context={context is not None} '
            f'recompute_method={getattr(getattr(self, "config", None), "recompute_method", "N/A")}'
        )

        # >>>
        # def custom(start: int, end: int):
        #     def custom_forward(*args, **kwargs):
        #         x_, context_, *args = args
        #         for index in range(start, end):
        #             layer = self._get_layer(index)
        #             # >>>
        #             # x_, context_ = layer(x_, *args, **{
        #             #     **kwargs,
        #             #     "context" : context_,
        #             # })
        #             x_, context_ = layer(x_, *args, **{
        #                 **kwargs,
        #                 "context" : context_,
        #             })
        #             # <<<
        #         return x_, context_
        #
        #     return custom_forward
        def custom(start: int, end: int):
            def custom_forward(
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                *args,
                **kwargs,
            ):
                print(
                    f'[M1890][custom_forward] layers {start}..{end-1} '
                    f'hidden={getattr(hidden_states, "shape", "?")} '
                    f'ctx={context is not None}'
                )
                for index in range(start, end):
                    layer = self._get_layer(index)
                    hidden_states, context = layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        *args,
                        **kwargs,
                    )
                return hidden_states, context

            return custom_forward
        # <<<

        config = self.config

        if config.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            l = 0
            while l < self.num_layers_per_pipeline_rank:
                hidden_states, context = _checkpoint(
                    custom(l, l + config.recompute_num_layers),
                    False,  # distribute_saved_activations — always False in Neuron_SP
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                )
                l += config.recompute_num_layers

        elif config.recompute_method == 'block':
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            for l in range(self.num_layers_per_pipeline_rank):
                if l < config.recompute_num_layers:
                    hidden_states, context = _checkpoint(
                        custom(l, l + 1),
                        False,  # distribute_saved_activations
                        hidden_states,
                        attention_mask,
                        context,
                        context_mask,
                        rotary_pos_emb,
                    )
                else:
                    hidden_states, context = custom(l, l + 1)(
                        hidden_states,
                        attention_mask,
                        context,
                        context_mask,
                        rotary_pos_emb,
                    )
        else:
            raise ValueError(
                f'[M1890] Invalid activation recompute method: {config.recompute_method!r}. '
                f'Expected "uniform" or "block".'
            )

        return hidden_states
