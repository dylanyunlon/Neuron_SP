# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Ported from Megatron-LM commit fabd3e4eac16433c8b3253011d0e06444771275d
# ViT Backbone Tensor Shape Fix — M1274

"""Vision Transformer(VIT) model."""

import math
import einops
import torch
import torch.nn.functional as F

print('[M1274]')

CLASS_TOKEN_LENGTH = 8


class VitMlpHead(torch.nn.Module):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(self, hidden_size, num_classes):
        super(VitMlpHead, self).__init__()
        self.dense_in = torch.nn.Linear(hidden_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.dense_out = torch.nn.Linear(hidden_size, num_classes)
        torch.nn.init.constant_(self.dense_out.bias, -10)

    def forward(self, hidden_states):
        # hidden_states: [b, 1, h]
        dense_in_result = self.dense_in(hidden_states)
        tanh_result = torch.tanh(dense_in_result)
        dense_out_result = self.dense_out(tanh_result)
        return dense_out_result


def isPerfectSquare(x):
    if x >= 0:
        sr = math.sqrt(x)
        return int(sr) * int(sr) == x
    return False


def twod_interpolate_position_embeddings_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
    num_patches_per_dim_h,
    num_patches_per_dim_w,
    num_patches,
    hidden_size,
    class_token_present,
):
    key = prefix + "weight"

    assert key in state_dict
    if key in state_dict:
        input_param = state_dict[key]

        input_seq_len = input_param.shape[0]
        assert isPerfectSquare(input_seq_len) or isPerfectSquare(input_seq_len - CLASS_TOKEN_LENGTH)
        input_has_class_token = not isPerfectSquare(input_seq_len)
        num_tok_input = input_seq_len - CLASS_TOKEN_LENGTH if input_has_class_token else input_seq_len
        num_tok_output = num_patches
        output_has_class_token = class_token_present

        if input_has_class_token:
            input_param_tok = input_param[:CLASS_TOKEN_LENGTH, :]
            input_param_grid = input_param[CLASS_TOKEN_LENGTH:, :]
        else:
            input_param_tok = torch.zeros(CLASS_TOKEN_LENGTH, hidden_size)
            input_param_grid = input_param

        assert input_param.shape[1] == hidden_size

        if num_tok_input != num_tok_output:
            gs_input = int(math.sqrt(num_tok_input))
            gs_new = (num_patches_per_dim_h, num_patches_per_dim_w)

            input_param_grid = input_param_grid.transpose(0, 1).contiguous()
            input_param_grid = input_param_grid.reshape((1, -1, gs_input, gs_input))
            input_param_grid = input_param_grid.float()
            scale_factor = (gs_new[0] / gs_input, gs_new[1] / gs_input)

            input_param_grid = F.interpolate(
                input_param_grid, scale_factor=scale_factor, mode="bilinear"
            )

            input_param_grid = input_param_grid.half()
            input_param_grid = input_param_grid.reshape((-1, num_tok_output))
            input_param_grid = input_param_grid.transpose(0, 1).contiguous()

            assert input_param_grid.shape[1] == hidden_size

        input_param = input_param_grid
        assert (
            input_param.shape[0] == num_tok_output
            and input_param.shape[1] == hidden_size
        )

        if output_has_class_token:
            input_param = torch.cat((input_param_tok, input_param), dim=0)

        state_dict[key] = input_param


class VitBackbone(torch.nn.Module):
    """Vision Transformer Model.

    Ported from Megatron-LM fabd3e4ea — ViT Backbone Tensor Shape Fix.
    Key changes vs prior version:
      - token_embeddings transposed [b,s,h] -> [s,b,h] before dropout (pre_process path)
      - post_process guard added around output reshape
      - single_token_output: hidden_states[0] instead of hidden_states[:,0,:]
      - else branch: transpose back [s,b,h] -> [b,s,h]
    """

    def __init__(
        self,
        hidden_size,
        num_layers,
        num_attention_heads,
        patch_dim,
        img_h,
        img_w,
        num_channels=3,
        hidden_dropout=0.0,
        init_method_std=0.02,
        pre_process=True,
        post_process=True,
        class_token=True,
        single_token_output=False,
        post_layer_norm=True,
        drop_path_rate=0.0,
        micro_batch_size=1,
        transformer_cls=None,
    ):
        super(VitBackbone, self).__init__()

        self.pre_process = pre_process
        self.post_process = post_process
        self.class_token = class_token
        self.post_layer_norm = post_layer_norm
        self.hidden_size = hidden_size
        self.patch_dim = patch_dim
        self.img_h = img_h
        self.img_w = img_w
        self.micro_batch_size = micro_batch_size
        self.single_token_output = single_token_output
        self.drop_path_rate = drop_path_rate

        assert self.img_h % self.patch_dim == 0
        assert self.img_w % self.patch_dim == 0
        self.num_patches_per_dim_h = self.img_h // self.patch_dim
        self.num_patches_per_dim_w = self.img_w // self.patch_dim
        self.num_patches = self.num_patches_per_dim_h * self.num_patches_per_dim_w
        self.seq_length = self.num_patches + (CLASS_TOKEN_LENGTH if self.class_token else 0)
        self.flatten_dim = self.patch_dim * self.patch_dim * num_channels
        self.input_tensor = None
        self.position_ids = None

        if self.pre_process:
            # cls_token
            if self.class_token:
                self.cls_token = torch.nn.Parameter(
                    torch.randn(1, CLASS_TOKEN_LENGTH, self.hidden_size)
                )
                torch.nn.init.zeros_(self.cls_token)
            self.position_ids = torch.arange(self.seq_length).expand(1, -1).cuda()

            # Linear encoder
            self.linear_encoder = torch.nn.Linear(self.flatten_dim, self.hidden_size)

            # embedding
            self.position_embeddings = torch.nn.Embedding(self.seq_length, self.hidden_size)
            torch.nn.init.normal_(self.position_embeddings.weight, std=init_method_std)

            self._class_token_present = self.class_token
            self.position_embeddings._register_load_state_dict_pre_hook(
                lambda sd, pfx, lmd, strict, mk, uk, em: twod_interpolate_position_embeddings_hook(
                    sd, pfx, lmd, strict, mk, uk, em,
                    num_patches_per_dim_h=self.num_patches_per_dim_h,
                    num_patches_per_dim_w=self.num_patches_per_dim_w,
                    num_patches=self.num_patches,
                    hidden_size=self.hidden_size,
                    class_token_present=self._class_token_present,
                )
            )

            self.embedding_dropout = torch.nn.Dropout(hidden_dropout)

        # Transformer — injected externally to stay DeepSpeed-agnostic
        assert transformer_cls is not None, (
            "VitBackbone requires a transformer_cls argument "
            "(e.g. deepspeed.model_implementations.transformers.ds_transformer.DeepSpeedTransformerLayer)"
        )
        self.transformer = transformer_cls

    def set_input_tensor(self, input_tensor):
        """Relay to underlying transformer's set_input_tensor if supported."""
        if hasattr(self.transformer, "set_input_tensor"):
            self.transformer.set_input_tensor(input_tensor)

    def forward(self, input):

        if self.pre_process:
            rearranged_input = einops.rearrange(
                input,
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=self.patch_dim,
                p2=self.patch_dim,
            )

            assert rearranged_input.dtype == torch.half
            encoder_output = self.linear_encoder(rearranged_input)

            concatenated_tokens = encoder_output
            if self.class_token:
                cls_tokens = self.cls_token.expand(encoder_output.shape[0], -1, -1)
                concatenated_tokens = torch.cat((cls_tokens, encoder_output), dim=1)

            token_embeddings = concatenated_tokens + \
                    self.position_embeddings(self.position_ids[:, :concatenated_tokens.shape[1]])
            # [b, s, h] => [s, b, h]
            token_embeddings = token_embeddings.transpose(0, 1).contiguous()
            hidden_states = self.embedding_dropout(token_embeddings)
        else:
            hidden_states = input

        hidden_states = self.transformer(hidden_states, None)

        if self.post_process:
            # [s b h] => [b s h]
            if self.single_token_output:
                hidden_states = hidden_states[0]
            else:
                hidden_states = hidden_states.transpose(0, 1).contiguous()

        return hidden_states
