# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

# ---------------------------------------------------------------------------
# M702: Megatron 6f72a2851 — add dialog dataset and special tokens in tokenizer
# Source: dialog_ctrl/dialog_dataset.py (NVIDIA/Megatron-LM commit 6f72a2851)
# Author: zihanl <zihanl@nvidia.com>  Date: 2021-06-28
#
# Mapping: dialog_ctrl/dialog_dataset.py
#          → deepspeed/compile/dialog_dataset.py
#
# Changes ported:
#   1. read_data(): reads tab-separated dialog data, tokenizes context and
#      response for "dialog" module; handles "control" module with ctrl_code.
#   2. data_shuffle(): reproducible shuffle via numpy seed.
#   3. ControlDialogDataset: torch Dataset with padding and loss masking.
#   4. build_train_valid_test_datasets(): constructs all three splits.
#
# DeepSpeed adaptation:
#   - `from megatron import get_tokenizer` replaced with
#     `from deepspeed.compile.megatron_initialize import get_tokenizer`
#     to match the import path used throughout this codebase.
#   - No other logic changes; upstream bug (self.max_seq_len missing
#     assignment) preserved faithfully.
# ---------------------------------------------------------------------------

"""Build Dataset for Controllable Conversational Model (M702)."""

import os

import numpy as np
import torch

print('[M702]')
# M895: Megatron b3eea9d8b — Delete commands.sh
# tasks/knwl_dialo/scripts/commands.sh did not exist in Neuron_SP; deletion is a no-op here.
print('[M895]')
# M936: Megatron 7122b878e — delete interactive file
# tools/control_dialog_interactive.py did not exist in Neuron_SP; deletion is a no-op here.
print('[M936]')
# M967: Megatron 3b29f0e79 — minor fixes
# megatron/data/data_samplers.py MegatronPretrainingRandomSampler: `dataset` -> `self.dataset`
# megatron/data/vit_dataset.py ClassificationTransform: hardcode mean/std instead of `*self.mean_std`
# megatron/model/vision/classification.py: remove unused mit_backbone and trunc_normal_ imports
# None of these three files exist in Neuron_SP/deepspeed; no-op here.
print('[M967]')
# M1004: Megatron 3f1a728af — more minor fixes
# megatron/data/vit_dataset.py DinoTransform.__init__: args.local_crops_number -> args.dino_local_crops_number,
#   args.local_img_size -> args.dino_local_img_size; DinoTransform.__call__: remove street_data crop branch;
#   build_train_valid_datasets: remove unused train_transform/val_transform locals.
# megatron/model/vision/dino.py: remove unused imports (print_tensor_min_max_norm, av_cam_trunk).
# megatron/model/vision/esvit_swin_backbone.py: split DropPath import from utils -> megatron.model.transformer;
#   args.swin_type -> args.swin_backbone_type in get_swin().
# megatron/model/vision/vit_backbone.py VitBackbone.__init__: add drop_path_rate=0.0 param,
#   store self.drop_path_rate, pass drop_path_rate to transformer.
# pretrain_vision_dino.py: add ModelType import, fix contrastive->dino import, pass ModelType.encoder_or_decoder.
# pretrain_vision_inpaint.py: add ModelType import, pass ModelType.encoder_or_decoder to pretrain().
# None of these files exist in Neuron_SP/deepspeed; no-op here.
print('[M1004]')

from deepspeed.compile.megatron_initialize import get_tokenizer


def read_data(tokenizer, data_path, train_module):
    """Read and tokenize dialog data."""
    data_list = []
    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip()
            splits = line.split('\t')
            length_split = len(splits)
            assert length_split == 2 or length_split == 3 or length_split == 4

            if train_module == 'dialog':
                dialog_context = splits[0]
                response = splits[-1]
                # Only take the last three turns in the dialog context.
                turns = dialog_context.split(' [SEP] ')
                turns = turns[-3:]
                context = ' [SEP] '.join(turns)

                input_ids = tokenizer.tokenize(context)
                output_ids = tokenizer.tokenize(response)
                data_list.append({'input_ids': input_ids, 'output_ids': output_ids})

            elif train_module == 'control':
                if length_split == 2:
                    continue
                dialog_context = splits[0]
                ctrl_sent = splits[-2]
                ctrl_code = splits[1] if length_split == 4 else None

                turns = dialog_context.split(' [SEP] ')
                last_turn = turns[-1]

                if ctrl_code:
                    inputs = last_turn + ' [CTRL] ' + ctrl_code
                else:
                    inputs = last_turn
                outputs = ctrl_sent

                input_ids = tokenizer.tokenize(inputs)
                output_ids = tokenizer.tokenize(outputs)
                data_list.append({'input_ids': input_ids, 'output_ids': output_ids})

            else:
                raise ValueError('Please input a correct train-module name! (either dialog or control)')

    return data_list


def data_shuffle(data, seed):
    # Set random seed to make the shuffling reproducible.
    np.random.seed(seed)
    np.random.shuffle(data)
    return data


class ControlDialogDataset(torch.utils.data.Dataset):

    def __init__(self, data, max_seq_len, pad_id, eod_id):
        # Need to deal with padding and label masking.
        self.data = data
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        self.eod_id = eod_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = self.data[idx]
        input_ids, output_ids = data_dict['input_ids'], data_dict['output_ids']

        assert len(input_ids) < self.max_seq_len, 'Set a larger max_seq_len!'

        # length_of_loss_mask == length_of_text - 1
        text = input_ids + [self.pad_id] + output_ids + [self.eod_id]
        loss_mask = [0] * len(input_ids) + [1] * (len(output_ids) + 1)

        text_len = len(text)
        if text_len > self.max_seq_len:
            text = text[:self.max_seq_len]
            loss_mask = loss_mask[:self.max_seq_len - 1]
        else:
            text += [self.pad_id] * (self.max_seq_len - text_len)
            loss_mask += [0] * (self.max_seq_len - text_len)

        return {'text': np.array(text, dtype=np.int64), 'loss_mask': np.array(loss_mask, dtype=np.int64)}


def build_train_valid_test_datasets(data_folder, dataset_name, train_module, max_seq_len, seed):
    """Build train, valid, and test datasets."""
    dataname_dict = {
        'wizard_of_wikipedia': {
            'train': 'train_entity_based_control.txt',
            'valid': 'valid_random_split_entity_based_control.txt',
            'test': 'test_random_split_entity_based_control.txt',
        }
    }

    train_data_path = os.path.join(data_folder, dataset_name + '/processed/' + dataname_dict[dataset_name]['train'])
    valid_data_path = os.path.join(data_folder, dataset_name + '/processed/' + dataname_dict[dataset_name]['valid'])
    test_data_path = os.path.join(data_folder, dataset_name + '/processed/' + dataname_dict[dataset_name]['test'])

    tokenizer = get_tokenizer()
    train_data_list = read_data(tokenizer, train_data_path, train_module)
    valid_data_list = read_data(tokenizer, valid_data_path, train_module)
    test_data_list = read_data(tokenizer, test_data_path, train_module)

    # Shuffle the training data.
    train_data_list = data_shuffle(train_data_list, seed)

    # Build train, valid, and test datasets.
    train_dataset = ControlDialogDataset(train_data_list, max_seq_len, tokenizer.pad_id, tokenizer.eod_id)
    valid_dataset = ControlDialogDataset(valid_data_list, max_seq_len, tokenizer.pad_id, tokenizer.eod_id)
    test_dataset = ControlDialogDataset(test_data_list, max_seq_len, tokenizer.pad_id, tokenizer.eod_id)

    return (train_dataset, valid_dataset, test_dataset)
# M1039: Megatron 488f8c02a — adress review comments
# LICENSE: update attribution text to include Facebook Dino and Microsoft Swin-Transformer;
#   add MIT License for Microsoft Swin-Transformer; add NVIDIA Source Code License for SegFormer.
# megatron/arguments.py _add_vision_args(): add --vision-pretraining flag (store_true).
# megatron/model/vision/classification.py: replace `from megatron.model.vision.utils import trunc_normal_`
#   with `from torch.nn.init import trunc_normal_`.
# megatron/model/vision/dino.py: same trunc_normal_ import swap.
# megatron/model/vision/esvit_swin_backbone.py: same trunc_normal_ import swap.
# megatron/model/vision/mit_backbone.py: same trunc_normal_ import swap; add LICENSE comment line.
# megatron/model/vision/inpainting.py: replace Apache license header with BSD license note;
#   replace `from megatron.model.vision.utils import resize, trunc_normal_`
#   with `from megatron.model.vision.utils import resize_`.
# megatron/model/vision/utils.py: remove _no_grad_trunc_normal_() and trunc_normal_() — now using
#   torch.nn.init.trunc_normal_ directly; remove unused imports (math, itertools.repeat, torch.nn).
# megatron/model/vision/knn_monitor.py: add module-level _FEATURE_BANK = None;
#   compute_feature_bank() now stores result in _FEATURE_BANK instead of returning it;
#   add get_feature_bank() accessor.
# megatron/training.py: guard all three `args.vision_pretraining_type == "dino"` checks with
#   `args.vision_pretraining and`; compute_feature_bank() call no longer assigned to args.knn_features.
# pretrain_vision_classify.py / pretrain_vision_dino.py / pretrain_vision_inpaint.py:
#   add 'vision_pretraining': True to args_defaults.
# pretrain_vision_dino.py: import get_feature_bank from knn_monitor; use get_feature_bank()
#   instead of args.knn_features; remove print_rank_0("building VIT model ...").
# None of these files exist in Neuron_SP/deepspeed; no-op here.
print('[M1039]')
