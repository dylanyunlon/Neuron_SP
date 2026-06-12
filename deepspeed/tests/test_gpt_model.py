# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M1312: Megatron 85a3a6d72 — consolidate gpt model
# Source: tests/models/test_gpt_model.py (NVIDIA/Megatron-LM commit 85a3a6d72)
#
# Mapping: tests/models/test_gpt_language_model.py → (deleted)
#          tests/models/test_gpt_model.py           → deepspeed/tests/test_gpt_model.py
#
# Changes ported from upstream:
#
#   1. test_gpt_language_model.py DELETED; TestGPTLanguageModel removed.
#
#   2. test_gpt_model.py CREATED; TestGPTModel replaces TestGPTLanguageModel:
#      - Fixture gpt_language_model → gpt_model; class GPTLanguageModel → GPTModel
#      - test_constructor: same weight count assertion (5040)
#      - test_set_input_tensor: asserts on gpt_model.transformer_block.input_tensor
#        (was gpt_language_model.encoder.input_tensor)
#      - test_gpu_forward → test_post_process_forward:
#          now asserts logits shape [b s vocab_size] instead of hidden_states [s b h]
#      - test_no_post_process_forward / test_no_preprocess_forward: stubs (pass)
#      - test_state_dict_for_save_checkpoint / test_load_state_dict: stubs (pass)
#
# 20% adaptation: imports from deepspeed/compile/gpt_model.py.
# Adds print('[M1312]') marker.
# ---------------------------------------------------------------------------

print('[M1312]')

import pytest
import torch

from megatron.core.transformer.transformer_config import TransformerConfig
from deepspeed.compile.gpt_model import GPTModel


@pytest.fixture
def gpt_model(transformer_config):
    language_model = GPTModel(config=transformer_config, vocab_size=100, max_sequence_length=4)
    return language_model


class TestGPTModel:
    def test_constructor(self, gpt_model: GPTModel):
        assert isinstance(gpt_model, GPTModel)

        assert gpt_model.max_sequence_length == 4

        num_weights = sum([p.numel() for p in gpt_model.parameters()])
        assert num_weights == 5040

    def test_set_input_tensor(self, gpt_model: GPTModel):
        config: TransformerConfig = gpt_model.config
        sequence_length = gpt_model.max_sequence_length
        micro_batch_size = 2

        # [sequence length, batch size, hidden size]
        input_tensor = torch.ones((sequence_length, micro_batch_size, config.hidden_size))

        gpt_model.set_input_tensor(input_tensor)

        assert gpt_model.transformer_block.input_tensor.shape[0] == sequence_length
        assert gpt_model.transformer_block.input_tensor.shape[1] == micro_batch_size
        assert gpt_model.transformer_block.input_tensor.shape[2] == config.hidden_size

    def test_post_process_forward(self, gpt_model: GPTModel):
        config: TransformerConfig = gpt_model.config
        sequence_length = gpt_model.max_sequence_length
        micro_batch_size = 2

        gpt_model.cuda()

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (1, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        logits = gpt_model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == gpt_model.vocab_size

    def test_no_post_process_forward(self, gpt_model: GPTModel):
        pass

    def test_no_preprocess_forward(self, gpt_model: GPTModel):
        pass

    def test_state_dict_for_save_checkpoint(self, gpt_model: GPTModel):
        pass

    def test_load_state_dict(self, gpt_model: GPTModel):
        pass
