# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

# ---------------------------------------------------------------------------
# M702: Megatron 6f72a2851 — add dialog dataset and special tokens in tokenizer
# Source: dialog_ctrl/utils.py (NVIDIA/Megatron-LM commit 6f72a2851)
# Author: zihanl <zihanl@nvidia.com>  Date: 2021-06-28
#
# Mapping: dialog_ctrl/utils.py
#          → deepspeed/compile/dialog_utils.py
#
# Changes ported:
#   1. get_ltor_attention_masks_and_position_ids(): builds causal attention
#      masks and position ids for left-to-right model; resets attention mask
#      and positions at EOD token boundaries.
#
# DeepSpeed adaptation:
#   - No imports from megatron; pure torch utility.
#   - No other logic changes.
# ---------------------------------------------------------------------------

"""Dialog attention mask and position id utilities (M702)."""

import torch

print('[M702]')


def get_ltor_attention_masks_and_position_ids(data, eod_token_id):
    """Build attention masks and position ids for left to right model."""
    micro_batch_size, seq_length = data.size()

    # Attention mask.
    attention_mask = torch.tril(
        torch.ones((micro_batch_size, seq_length, seq_length), device=data.device)
    ).view(micro_batch_size, 1, seq_length, seq_length)

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)

    # Reset attention mask and position ids at EOD token boundaries.
    for b in range(micro_batch_size):
        # Find indices where EOD token is.
        eod_index = position_ids[b, data[b] == eod_token_id]
        eod_index = eod_index.clone()

        # Loop through EOD indices.
        prev_index = 0
        for j in range(eod_index.size()[0]):
            i = eod_index[j]
            # Mask attention loss.
            attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
            # Reset positions.
            position_ids[b, (i + 1):] -= (i + 1 - prev_index)
            prev_index = i + 1

    # Convert attention mask to binary.
    attention_mask = (attention_mask < 0.5)

    return attention_mask, position_ids
