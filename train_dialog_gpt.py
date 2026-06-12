# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

# ---------------------------------------------------------------------------
# M702: Megatron 6f72a2851 — add dialog dataset and special tokens in tokenizer
# Source: train_dialog_gpt.py (NVIDIA/Megatron-LM commit 6f72a2851)
# Author: zihanl <zihanl@nvidia.com>  Date: 2021-06-28
#
# Mapping: train_dialog_gpt.py (Megatron root)
#          → train_dialog_gpt.py (Neuron_SP root)
#
# Changes ported:
#   1. model_provider(): builds GPTModel.
#   2. get_batch(): reads text/loss_mask tensors, builds tokens, labels,
#      attention_masks, position_ids via dialog_utils.
#   3. loss_func(): masked cross-entropy over output positions only.
#   4. forward_step(): wires get_batch → model → loss_func.
#   5. train_valid_test_datasets_provider(): delegates to dialog_dataset.
#   6. __main__: calls pretrain() with GPT2BPETokenizer default.
#
# DeepSpeed adaptation:
#   - Megatron imports replaced with deepspeed.compile equivalents:
#       from megatron import get_args      → from deepspeed.compile.megatron_initialize import get_args
#       from megatron import print_rank_0  → from deepspeed.compile.megatron_training import print_rank_0
#       from megatron import get_timers    → from deepspeed.compile.megatron_training import get_timers
#       from megatron import get_tokenizer → from deepspeed.compile.megatron_initialize import get_tokenizer
#       from megatron import mpu           → import deepspeed.comm as dist (check-torchdist compliance)
#       from megatron.model import GPTModel→ from deepspeed.compile.megatron_module import GPTModel
#       from megatron.training import pretrain → from deepspeed.compile.megatron_training import pretrain
#       from megatron.utils import average_losses_across_data_parallel_group
#                                          → from deepspeed.compile.megatron_training import ...
#   - dialog_ctrl.* → deepspeed.compile.dialog_*
#   - upstream typo `attention_mask` vs `attention_masks` preserved as-is
#     (faithfully mirrors upstream).
# ---------------------------------------------------------------------------

"""Train dialogue model based on GPT (M702)."""

from functools import partial

import torch

print('[M702]')

from deepspeed.compile.megatron_initialize import get_args, get_tokenizer
from deepspeed.compile.megatron_training import (
    get_timers,
    pretrain,
    print_rank_0,
    average_losses_across_data_parallel_group,
)
from deepspeed.compile.dialog_dataset import build_train_valid_test_datasets
from deepspeed.compile.dialog_utils import get_ltor_attention_masks_and_position_ids

try:
    from deepspeed.compile.megatron_module import GPTModel
except ImportError:
    # Fallback: GPTModel may live elsewhere depending on active M-series state.
    from deepspeed.compile.megatron_initialize import GPTModel  # noqa: F401

try:
    import deepspeed.comm as dist
    _mpu = dist
except ImportError:
    import deepspeed.comm as _mpu


def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    print_rank_0('building GPT model ...')
    model = GPTModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process,
    )
    return model


def get_batch(data_iterator):
    """Generate a batch."""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text', 'loss_mask']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = _mpu.broadcast_data(keys, data, datatype)

    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    loss_mask = data_b['loss_mask'].float()

    # Get the attention_mask and position ids.
    attention_masks, position_ids = get_ltor_attention_masks_and_position_ids(tokens, tokenizer.eod_id)

    return tokens, labels, loss_mask, attention_masks, position_ids


def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider():
    """Build train, valid, and test datasets for control module."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets for %s module ...' % args.train_module)

    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_folder=args.data_folder,
        dataset_name=args.dataset_name,
        train_module=args.train_module,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )
    print_rank_0('> finished creating datasets for %s module ...' % args.train_module)


if __name__ == '__main__':
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
