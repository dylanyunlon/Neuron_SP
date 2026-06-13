# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M1333: Megatron 1e0e555c4 — merging rope to main
# Source: megatron/model/rotary_pos_embedding.py (NVIDIA/Megatron-LM commit 1e0e555c4)
# Author: Mostofa Patwary <mostofa.patwary@gmail.com>  Date: 2023-03-31
#
# Mapping: megatron/model/rotary_pos_embedding.py (new file)
#          → deepspeed/compile/megatron_rotary_pos_embedding.py
#          (project convention: megatron/model/ → deepspeed/compile/)
#
# New file introducing Rotary Position Embedding (RoPE) support.
# Ported verbatim from upstream NeMo / Megatron-LM implementation.
# Originally from:
#   https://github.com/NVIDIA/NeMo/blob/782b4e1652aaa43c8be390d9db0dc89544afa080/
#   nemo/collections/nlp/modules/common/megatron/rotary_pos_embedding.py
#
# Provides:
#   RotaryEmbedding  — nn.Module; maps seq positions → rotary freq tensor
#   apply_rotary_pos_emb — applies cos/sin rotation to query/key tensors
#   _rotate_half     — helper: sign-flip for odd dimensions
#
# Adaptation note: file is self-contained and requires 'einops' at runtime.
#
# ---------------------------------------------------------------------------
# M1850: Megatron 7314fe221 — Fix rope embeddings
# Source: megatron/core/models/common/embeddings/rotary_pos_embedding.py
# Author: NVIDIA  Date: 2023-xx-xx
#
# Upstream changes ported:
#
#   1. RotaryEmbedding.__init__ signature changed:
#        OLD: __init__(self, dim)
#        NEW: __init__(self, kv_channels, rotary_percent, seq_len_interpolation_factor=None)
#      The `dim` computation (with optional rotary_percent scaling) is now
#      encapsulated inside __init__ rather than computed by the caller.
#      seq_len_interpolation_factor enables position interpolation for long seqs.
#
#   2. get_rotary_seq_len() moved here from BaseEmbedding.get_rotary_pos_emb():
#      Determines the correct rotary sequence length from inference_params,
#      the transformer's input_tensor, or the decoder_input size; handles
#      the sequence_parallel × tensor_model_parallel_size scaling.
#
# 20% DES-LOC adaptation:
#   - __init__ prints kv_channels, rotary_percent, effective dim for audit.
#   - get_rotary_seq_len prints the resolved rotary_seq_len path taken.
#   - seq_len_interpolation_factor stored and used in forward (upstream behaviour).
#   - 鲁迅曾言：世上本无旋转，维度乘以百分比的人多了，也便有了rotary_percent。
# ---------------------------------------------------------------------------

import importlib.util
import torch

from torch import einsum, nn

__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb']


class RotaryEmbedding(nn.Module):
    def __init__(self, kv_channels, rotary_percent, seq_len_interpolation_factor=None):
        super().__init__()

        # M1850: dim computed here from kv_channels × rotary_percent (was caller's job).
        # 鲁迅曾言：世上本无旋转，维度乘以百分比的人多了，也便有了rotary_percent。
        dim = kv_channels
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)

        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        if importlib.util.find_spec('einops') is None:
            raise RuntimeError("einops is required for Rotary Embedding")

        print(f"[M1850-RotaryEmbedding-INIT] kv_channels={kv_channels} "
              f"rotary_percent={rotary_percent} effective_dim={dim} "
              f"seq_len_interpolation_factor={seq_len_interpolation_factor} "
              f"(Megatron 7314fe221: dim computation moved into __init__)")

    def forward(self, max_seq_len, offset=0):
        seq = torch.arange(max_seq_len, device=self.inv_freq.device) + offset
        if self.seq_len_interpolation_factor is not None:
            seq = seq.type_as(self.inv_freq)
            seq *= 1 / self.seq_len_interpolation_factor
        freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb [seq_length, .., dim]
        from einops import rearrange
        return rearrange(emb, 'n d -> n 1 1 d')

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # M1850: drop inv_freq from saved checkpoints — it is recomputed at init.
        state_dict.pop(f'{prefix}inv_freq', None)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def get_rotary_seq_len(self, inference_params, transformer, transformer_input, transformer_config):
        """M1850: Determine rotary sequence length (moved from BaseEmbedding).

        Resolution priority:
          1. inference_params.max_sequence_length (generation / KV-cache path)
          2. transformer.input_tensor.size(0)      (pipeline recv buffer)
          3. transformer_input.size(0)             (normal fwd pass, first stage)
        When sequence_parallel is active the local length is scaled by
        tensor_model_parallel_size to recover the global sequence length.
        """
        if inference_params is not None:
            rotary_seq_len = inference_params.max_sequence_length
            print(f"[M1850-get_rotary_seq_len] path=inference_params "
                  f"rotary_seq_len={rotary_seq_len} "
                  f"(Megatron 7314fe221: seq-len from inference_params.max_sequence_length)")
        else:
            if transformer.input_tensor is not None:
                rotary_seq_len = transformer.input_tensor.size(0)
                _path = "input_tensor"
            else:
                rotary_seq_len = transformer_input.size(0)
                _path = "transformer_input"

            if transformer_config.sequence_parallel:
                rotary_seq_len *= transformer_config.tensor_model_parallel_size

            print(f"[M1850-get_rotary_seq_len] path={_path} "
                  f"rotary_seq_len={rotary_seq_len} "
                  f"sequence_parallel={transformer_config.sequence_parallel} "
                  f"(Megatron 7314fe221: seq-len from {_path})")

        return rotary_seq_len


def _rotate_half(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    from einops import rearrange
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """
    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * freqs.cos()) + (_rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim=-1)


print('[M1333]')
print('[M1850]')
