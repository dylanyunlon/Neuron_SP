# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M1312: Megatron 85a3a6d72 — consolidate gpt model
# Source: megatron/core/models/gpt/gpt_model.py (NVIDIA/Megatron-LM commit 85a3a6d72)
# Author: eharper <eharper@nvidia.com>  Date: 2023-02-14
#
# Mapping: megatron/core/models/gpt/gpt_language_model.py  → (deleted)
#          megatron/core/models/gpt/gpt_model.py            → deepspeed/compile/gpt_model.py
#          (project convention: megatron/core/models/gpt/* → deepspeed/compile/)
#
# Changes ported from upstream:
#
#   1. gpt_language_model.py DELETED; GPTLanguageModel class removed entirely.
#      The stripped-down language model (embedding + encoder only) is superseded
#      by the consolidated GPTModel below.
#
#   2. gpt_model.py CREATED; GPTModel replaces GPTLanguageModel with:
#
#      a) New constructor args:
#           fp_16_lm_cross_entropy: bool = False  — use fp16 cross-entropy loss
#           parallel_output: bool = True           — return parallel (sharded) logits
#
#      b) self.encoder renamed → self.transformer_block
#         (ParallelTransformerBlock instance; key still stored as _encoder_key='encoder'
#          for checkpoint backward-compatibility)
#
#      c) self.initialize_word_embeddings() called at end of __init__
#         to tie/sync embedding weights across pipeline stages.
#
#      d) set_input_tensor: delegates to self.transformer_block (was self.encoder)
#
#      e) forward() gains:
#           labels: Tensor = None  — when provided, returns cross-entropy loss
#                                    instead of raw logits
#           post-process branch:  calls post_language_model_processing()
#           returns hidden_states when not post_process
#
#      f) NEW parallel_lm_logits():
#         Computes vocabulary logits using tensor-parallel linear with optional
#         async_grad_allreduce / sequence_parallel paths; gathers if not
#         parallel_output.
#
#      g) NEW post_language_model_processing():
#         Calls parallel_lm_logits; if labels is None returns transposed logits
#         [b s h]; otherwise computes vocab_parallel_cross_entropy loss [b s].
#
#      h) NEW initialize_word_embeddings():
#         Ties word-embedding weights between first and last pipeline stages
#         via all_reduce over the embedding group.  Early-returns when
#         pipeline_model_parallel_size == 1 (no pipeline, nothing to tie).
#
#      i) NEW word_embeddings_weight():
#         Returns embedding.word_embeddings.weight for pre_process stages,
#         or self.word_embeddings.weight for the last stage head copy.
#
#      j) state_dict_for_save_checkpoint / load_state_dict stubbed out with
#         pass and TODO comments (distributed checkpointing deferred).
#
# 20% adaptation: deepspeed/compile/ targets the DS runtime; imports from
# megatron.core.* are preserved so this file is usable with the Megatron-Core
# backend bundled inside Neuron_SP.  Adds print('[M1312]') marker.
# ---------------------------------------------------------------------------

print('[M1312]')

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.parallel_transformer_block import ParallelTransformerBlock
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.models.gpt.gpt_embedding import GPTEmbedding


class GPTModel(MegatronModule):
    """Transformer language model.

    Arguments:
        config: TransformerConfig — transformer hyperparameters
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence; used for positional embedding
        pre_process: whether this rank handles the embedding stage
        post_process: whether this rank handles the output / loss stage
        fp_16_lm_cross_entropy: compute cross-entropy loss in fp16 (requires half output)
        parallel_output: return parallel (sharded) logits instead of gathered logits
    """

    def __init__(
        self,
        config: TransformerConfig,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp_16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
    ):
        super(GPTModel, self).__init__(config=config)

        self.config: TransformerConfig = config
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp_16_lm_cross_entropy = fp_16_lm_cross_entropy
        self.parallel_output = parallel_output

        # Embeddings.
        if self.pre_process:
            self.embedding = GPTEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
            )
            self._embedding_key = 'embedding'

        # Transformer.
        # Renamed from self.encoder → self.transformer_block (85a3a6d72).
        # _encoder_key kept as 'encoder' for checkpoint backward-compat.
        self.transformer_block = ParallelTransformerBlock(
            config=self.config,
            self_attn_mask_type=AttnMaskType.causal,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )
        self._encoder_key = 'encoder'

        self.initialize_word_embeddings()
        print('[M1312] GPTModel.__init__ complete')

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""

        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt'
        self.transformer_block.set_input_tensor(input_tensor[0])

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor = None,
        inference_params=None,
    ):

        # Encoder embedding.
        if self.pre_process:
            encoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # transformer_block will get hidden_states from input_tensor
            encoder_input = None

        # Run transformer.
        hidden_states = self.transformer_block(
            hidden_states=encoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
        )

        if self.post_process:
            logits = self.post_language_model_processing(
                hidden_states=hidden_states,
                labels=labels,
                logit_weights=self.word_embeddings_weight(),
            )
            return logits

        return hidden_states

    def parallel_lm_logits(
        self,
        input_: Tensor,
        word_embeddings_weight: Tensor,
        bias: Tensor = None,
    ):
        """LM logits using word embedding weights.

        Megatron 85a3a6d72: sequence_parallel_enabled / async_tensor_model_parallel_allreduce
        paths select whether to copy input to the TP region first.
        """
        # Parallel logits.
        if (
            self.config.async_tensor_model_parallel_allreduce
            or self.config.sequence_parallel_enabled
        ):
            input_parallel = input_
            model_parallel = parallel_state.get_tensor_model_parallel_world_size() > 1
            async_grad_allreduce = (
                self.config.async_tensor_model_parallel_allreduce
                and model_parallel
                and not self.config.sequence_parallel_enabled
            )
        else:
            input_parallel = tensor_parallel.copy_to_tensor_model_parallel_region(input_)
            async_grad_allreduce = False

        # Matrix multiply.
        logits_parallel = tensor_parallel.linear_with_grad_accumulation_and_async_allreduce(
            input=input_parallel,
            weight=word_embeddings_weight,
            bias=bias,
            gradient_accumulation_fusion=self.config.gradient_accumulation_fusion,
            async_grad_allreduce=async_grad_allreduce,
            sequence_parallel_enabled=self.config.sequence_parallel_enabled,
        )

        # Gather if needed.
        if self.parallel_output:
            return logits_parallel
        else:
            logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits_parallel)

        return logits

    def post_language_model_processing(
        self,
        hidden_states: Tensor,
        labels: Tensor,
        logit_weights: Tensor,
    ):
        # Output. Format [s b h]
        output = self.parallel_lm_logits(hidden_states, logit_weights)

        if labels is None:
            # [s b h] => [b s h]
            return output.transpose(0, 1).contiguous()
        else:
            # [b s] => [s b]
            labels = labels.transpose(0, 1).contiguous()
            if self.fp16_lm_cross_entropy:
                assert output.dtype == torch.half
                loss = tensor_parallel.vocab_parallel_cross_entropy(output, labels)
            else:
                loss = tensor_parallel.vocab_parallel_cross_entropy(output.float(), labels)

            # [s b] => [b, s]
            loss = loss.transpose(0, 1).contiguous()
            return loss

    def initialize_word_embeddings(self):
        """Tie word-embedding weights across the first and last pipeline stages.

        Megatron 85a3a6d72: early-return when pipeline_model_parallel_size == 1
        so that single-stage models don't attempt to all_reduce over a
        non-existent pipeline embedding group.
        """
        if self.config.pipeline_model_parallel_size == 1:
            return

        # Parameters are shared between the word embeddings layers, and the
        # heads at the end of the model. In a pipelined setup with more than
        # one stage, the initial embedding layer and the head are on different
        # workers, so we do the following:
        # 1. Create a second copy of word_embeddings on the last stage, with
        #    initial parameters of 0.0.
        # 2. Do an all-reduce between the first and last stage to ensure that
        #    the two copies of word_embeddings start off with the same
        #    parameter values.
        # 3. In the training loop, before an all-reduce between the grads of
        #    the two word_embeddings layers to ensure that every applied weight
        #    update is the same on both stages.
        if parallel_state.is_pipeline_last_stage() and not self.pre_process:
            assert not parallel_state.is_pipeline_first_stage()
            self._word_embeddings_for_head_key = 'word_embeddings_for_head'
            # set word_embeddings weights to 0 here, then copy first
            # stage's weights using all_reduce below.
            self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
                self.vocab_size,
                self.config.hidden_size,
                init_method=self.config.init_method(self.config.init_method_std),
                params_dtype=self.config.params_dtype,
                use_cpu_initialization=self.config.use_cpu_initialization,
                perform_initialization=self.config.perform_initialization,
            )
            self.word_embeddings.weight.data.fill_(0)
            self.word_embeddings.weight.shared = True

        # Zero out initial weights for decoder embedding.
        # NOTE: We don't currently support T5 with the interleaved schedule.
        if not parallel_state.is_pipeline_first_stage(ignore_virtual=True) and self.pre_process:
            self.transformer_block.embedding.zero_parameters()

        if not torch.distributed.is_initialized():
            # TODO: this should be log not print
            if not getattr(MegatronModule, "embedding_warning_printed", False):
                print(
                    "WARNING! Distributed processes aren't initialized, so "
                    "word embeddings in the last layer are not initialized. "
                    "If you are just manipulating a model this is fine, but "
                    "this needs to be handled manually. If you are training "
                    "something is definitely wrong."
                )
                MegatronModule.embedding_warning_printed = True
            return

        # Ensure that first and last stages have the same initial parameter
        # values.
        if parallel_state.is_rank_in_embedding_group():
            torch.distributed.all_reduce(
                self.word_embeddings_weight().data,
                group=parallel_state.get_embedding_group(),
            )

    def word_embeddings_weight(self):
        if self.pre_process:
            return self.embedding.word_embeddings.weight
        else:
            if not self.share_word_embeddings:
                raise Exception(
                    'word_embeddings_weight() called for last '
                    'stage, but share_word_embeddings is false'
                )
            return self.word_embeddings.weight

    # TODO: add distributed checkpointing
    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        pass

    # TODO: add distributed checkpointing
    def load_state_dict(self, state_dict, strict=True):
        pass
