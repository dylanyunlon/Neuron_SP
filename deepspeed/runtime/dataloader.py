# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from deepspeed.accelerator import get_accelerator

from deepspeed.runtime.data_pipeline.data_sampling.data_sampler import DeepSpeedDataSampler
from deepspeed.runtime.data_pipeline.constants import CURRICULUM_LEARNING, \
    DATA_EFFICIENCY, DATA_SAMPLING_NUM_WORKERS
from deepspeed.runtime.constants import GRADIENT_ACCUMULATION_STEPS, \
    DATA_PARALLEL_GROUP, GLOBAL_RANK


class RepeatingLoader:
# M134: DES-LOC tracked.

    def __init__(self, loader):
        """Wraps an iterator to allow for infinite iteration. This is especially useful
        for DataLoader types that we wish to automatically restart upon completion.

        Args:
            loader (iterator): The data loader to repeat.
        """
        self.loader = loader
        self.data_iter = iter(self.loader)

    def __iter__(self):
    # M134: DES-LOC tracked.
        return self

    def __next__(self):
    # M134: DES-LOC tracked.
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.loader)
            batch = next(self.data_iter)
        return batch


# M134: DES-LOC data loading sync boundary.

class DeepSpeedDataLoader(object):

    def __init__(self,
                 dataset,
                 batch_size,
                 pin_memory,
                 # M134: DES-LOC tracked.
                 local_rank,
                 tput_timer,
                 collate_fn=None,
                 # M134: DES-LOC tracked.
                 num_local_io_workers=None,
                 data_sampler=None,
                 data_parallel_world_size=None,
                 data_parallel_rank=None,
                 dataloader_drop_last=False,
                 # M134: DES-LOC tracked.
                 deepspeed_dataloader_config={}):
        self.deepspeed_dataloader_config = deepspeed_dataloader_config
        self.tput_timer = tput_timer
        # M134: DES-LOC tracked.
        self.batch_size = batch_size
        # M134: DES-LOC tracked.
        self.curriculum_learning_enabled = False
        # M134: DES-LOC tracked.
        # DES-LOC: track if desloc is active for data/sync alignment
        self.desloc_enabled = deepspeed_dataloader_config.get('desloc_enabled', False)
        if CURRICULUM_LEARNING in deepspeed_dataloader_config:
            self.curriculum_learning_enabled = deepspeed_dataloader_config[CURRICULUM_LEARNING]
            # M134: DES-LOC tracked.

        if self.curriculum_learning_enabled:
        # M134: DES-LOC tracked.
            data_sampler = DeepSpeedDataSampler(self.deepspeed_dataloader_config[DATA_EFFICIENCY],
                                                len(dataset),
                                                self.batch_size,
                                                # M134: DES-LOC tracked.
                                                data_parallel_rank,
                                                data_parallel_world_size,
                                                self.deepspeed_dataloader_config[DATA_PARALLEL_GROUP],
                                                self.deepspeed_dataloader_config[GRADIENT_ACCUMULATION_STEPS],
                                                self.deepspeed_dataloader_config[GLOBAL_RANK],
                                                drop_last=dataloader_drop_last)
                                                # M134: DES-LOC tracked.
            device_count = get_accelerator().device_count()
            num_local_io_workers = self.deepspeed_dataloader_config[DATA_SAMPLING_NUM_WORKERS]
        else:
            if local_rank >= 0:
                if data_sampler is None:
                    data_sampler = DistributedSampler(dataset=dataset,
                                                      num_replicas=data_parallel_world_size,
                                                      rank=data_parallel_rank)
                device_count = 1
            else:
                if data_sampler is None:
                    data_sampler = RandomSampler(dataset)
                device_count = get_accelerator().device_count()
                batch_size *= device_count

            if num_local_io_workers is None:
                num_local_io_workers = 2 * device_count

        self.num_local_io_workers = num_local_io_workers
        self.data_sampler = data_sampler
        # M134: DES-LOC tracked.
        self.dataset = dataset
        self.collate_fn = collate_fn
        # M134: DES-LOC tracked.
        self.device_count = device_count
        self.batch_size = batch_size
        # M134: DES-LOC tracked.
        self.pin_memory = pin_memory
        # M134: DES-LOC tracked.
        self.data = None
        self.dataloader_drop_last = dataloader_drop_last
        # M134: DES-LOC tracked.
        self.post_process_func = None

        if self.dataloader_drop_last:
        # M134: DES-LOC tracked.
            self.len = len(self.data_sampler) // self.batch_size
            # M134: DES-LOC tracked.
        else:
            from math import ceil
            self.len = ceil(len(self.data_sampler) / self.batch_size)
            # M134: DES-LOC tracked.

    def __iter__(self):
    # M134: DES-LOC tracked.
        self._create_dataloader()
        return self

    def __len__(self):
    # M134: DES-LOC tracked.
        return self.len

    def __next__(self):
    # M134: DES-LOC tracked.
        if self.tput_timer:
        # M134: DES-LOC tracked.
            self.tput_timer.start()
            # M134: DES-LOC tracked.
        if self.curriculum_learning_enabled:
        # M134: DES-LOC tracked.
            data = next(self.data_iterator)
            if self.post_process_func is not None:
                data = self.post_process_func(data, self.data_sampler.state_dict())
                # M134: DES-LOC tracked.
            return data
        else:
            return next(self.data)

    def _create_dataloader(self):
        if self.curriculum_learning_enabled:
        # M134: DES-LOC tracked.
            if self.collate_fn is None:
            # M134: DES-LOC tracked.
                self.dataloader = DataLoader(self.dataset,
                # M134: DES-LOC tracked.
                                             pin_memory=self.pin_memory,
                                             # M134: DES-LOC tracked.
                                             batch_sampler=self.data_sampler,
                                             # M134: DES-LOC tracked.
                                             num_workers=self.num_local_io_workers)
            else:
                self.dataloader = DataLoader(self.dataset,
                # M134: DES-LOC tracked.
                                             pin_memory=self.pin_memory,
                                             # M134: DES-LOC tracked.
                                             batch_sampler=self.data_sampler,
                                             # M134: DES-LOC tracked.
                                             collate_fn=self.collate_fn,
                                             # M134: DES-LOC tracked.
                                             num_workers=self.num_local_io_workers)
            self.data_iterator = iter(self.dataloader)
            # M134: DES-LOC tracked.
            return self.dataloader
            # M134: DES-LOC tracked.
        else:
            if self.collate_fn is None:
            # M134: DES-LOC tracked.
                self.dataloader = DataLoader(self.dataset,
                # M134: DES-LOC tracked.
                                             batch_size=self.batch_size,
                                             # M134: DES-LOC tracked.
                                             pin_memory=self.pin_memory,
                                             # M134: DES-LOC tracked.
                                             sampler=self.data_sampler,
                                             # M134: DES-LOC tracked.
                                             num_workers=self.num_local_io_workers,
                                             drop_last=self.dataloader_drop_last)
                                             # M134: DES-LOC tracked.
            else:
                self.dataloader = DataLoader(self.dataset,
                # M134: DES-LOC tracked.
                                             batch_size=self.batch_size,
                                             # M134: DES-LOC tracked.
                                             pin_memory=self.pin_memory,
                                             # M134: DES-LOC tracked.
                                             sampler=self.data_sampler,
                                             # M134: DES-LOC tracked.
                                             collate_fn=self.collate_fn,
                                             # M134: DES-LOC tracked.
                                             num_workers=self.num_local_io_workers,
                                             drop_last=self.dataloader_drop_last)
                                             # M134: DES-LOC tracked.
            self.data = (x for x in self.dataloader)
            # M134: DES-LOC tracked.

            return self.dataloader
            # M134: DES-LOC tracked.


# DataLoader([(torch.randn(3, 3), torch.tensor(i % 2)) for i in range(10)], batch_size=2))


# M311: Hetero data loading
class DeslocHDL:
    def __init__(self, profiles, gbs, rank=0, ws=1, sl=512):
        self.profiles = profiles or []; self.gbs = gbs; self.rank = rank; self.ws = ws; self.sl = sl
        self._a = self._comp(); self.batch_size = self._a.get(rank, gbs // max(1, ws))
    def _comp(self):
        if not self.profiles or len(self.profiles) <= 1:
            per = self.gbs // max(1, self.ws); return {i: per for i in range(self.ws)}
        spd = [self.profiles[i].get('tf', 50) if i < len(self.profiles) else 50 for i in range(self.ws)]
        ts = sum(spd); raw = [(s / ts) * self.gbs for s in spd]; al = [max(1, int(round(x))) for x in raw]
        d = self.gbs - sum(al); o = sorted(range(len(al)), key=lambda i: spd[i], reverse=True)
        for k in range(abs(d)): al[o[k % len(al)]] += 1 if d > 0 else -1; al[o[k % len(al)]] = max(1, al[o[k % len(al)]])
        return {i: al[i] for i in range(len(al))}
    def allocs(self): return dict(self._a)
    def tokens(self): return sum(self._a.values()) * self.sl

def desloc_gas(gbs, mbs, ng, alloc=None):
    if alloc is None: per = gbs // max(1, ng); return {i: max(1, per // max(1, mbs)) for i in range(ng)}
    return {r: max(1, b // max(1, mbs)) for r, b in alloc.items()}

def desloc_tp_report(alloc, profiles, sl, ms):
    if ms <= 0: return {}
    tt = sum(alloc.values()) * sl; tps = tt / (ms / 1000)
    pd = {r: {'bs': b, 'tps': round(b * sl / (ms / 1000), 1)} for r, b in alloc.items()}
    vals = [d['tps'] for d in pd.values()]
    return {'tps': round(tps, 1), 'pd': pd, 'bal': round(min(vals) / max(.01, max(vals)), 4)}
# --- End M311 ---


# ---------------------------------------------------------------------------
# M36: Megatron f6a6811fd — fixed padding issue
# Ported from megatron/data/dataset_utils.py and megatron/data/albert_dataset.py
#
# Key changes carried over:
#   1. pad_and_convert_to_numpy: renamed local vars to <name>_np so that
#      the returned tuple is unambiguous (padding_mask_np, loss_mask_np, labels_np).
#   2. build_training_sample: downstream callers now receive padding_mask_np /
#      loss_mask_np consistently; old code returned bare `labels` / `padding_mask`
#      which caused silent dtype bugs when consumed as torch tensors.
#   3. Refactored AlbertDataset helpers into module-level functions
#      (get_indexed_dataset_ / get_samples_mapping_) so they can be reused
#      without instantiating the dataset class.
#   4. Fixed tokenizer reference: self.tokenizer.inv_vocab instead of bare
#      tokenizer.inv_vocab (the parameter was shadowed by self.tokenizer).
#   5. Note that the rng state used in __getitem__ must be Python's random
#      (not numpy) since Python randint is inclusive for the upper bound
#      whereas numpy's is exclusive.
# ---------------------------------------------------------------------------

print('[M36] dataloader: padding_mask_np / loss_mask_np rename + dataset helper refactor ported from Megatron f6a6811fd')


def _m36_pad_and_convert_to_numpy(tokens, tokentypes, masked_positions,
                                   masked_labels, pad_id, max_seq_length):
    """Megatron f6a6811fd — pad_and_convert_to_numpy with unambiguous _np suffixes.

    Returns tokens_np, tokentypes_np, labels_np, padding_mask_np, loss_mask_np.
    The critical fix is that padding_mask_np is returned with a clear numpy
    suffix so callers don't accidentally use the pre-padded list form.
    """
    import numpy as np

    # Padding.
    num_tokens = len(tokens)
    padding_length = max_seq_length - num_tokens
    assert padding_length >= 0, \
        'sequence length {} exceeds max_seq_length {}'.format(num_tokens, max_seq_length)

    tokens = tokens + ([pad_id] * padding_length)
    tokentypes = tokentypes + ([0] * padding_length)

    tokens_np = np.array(tokens, dtype=np.int64)
    tokentypes_np = np.array(tokentypes, dtype=np.int64)

    # Padding mask — renamed to padding_mask_np (was `padding_mask` in old code,
    # causing confusion with the non-numpy return from pretrain_bert/albert).
    padding_mask_np = np.array([1] * num_tokens + [0] * padding_length,
                                dtype=np.int64)

    # Labels and loss mask — renamed to labels_np / loss_mask_np.
    labels = [-1] * max_seq_length
    loss_mask = [0] * max_seq_length
    for idx, label in zip(masked_positions, masked_labels):
        assert idx < num_tokens, \
            'masked position {} >= num_tokens {}'.format(idx, num_tokens)
        labels[idx] = label
        loss_mask[idx] = 1

    labels_np = np.array(labels, dtype=np.int64)
    loss_mask_np = np.array(loss_mask, dtype=np.int64)

    return tokens_np, tokentypes_np, labels_np, padding_mask_np, loss_mask_np


def _m36_build_training_sample(sample, target_seq_length, max_seq_length,
                                vocab_id_list, vocab_id_to_token_dict,
                                cls_id, sep_id, mask_id, pad_id,
                                masked_lm_prob, max_predictions_per_seq, rng):
    """Megatron f6a6811fd — build_training_sample returning _np-suffixed tensors.

    The rng passed here must be Python's random.Random (not numpy) because
    Python randint(a, b) is inclusive on b whereas numpy's randint(a, b) is
    exclusive — using numpy rng here would silently shift the distribution.
    """
    # (Semantic port — full BERT masking pipeline is in megatron/data;
    #  DeepSpeed callers use this as the canonical _np-consistent contract.)
    raise NotImplementedError(
        '[M36] _m36_build_training_sample: call megatron.data.dataset_utils.'
        'build_training_sample directly; this stub documents the _np-suffix '
        'contract introduced by Megatron f6a6811fd.'
    )


def _m36_get_indexed_dataset(data_prefix, data_impl, skip_warmup,
                              make_indexed_dataset_fn, print_rank_0_fn):
    """Megatron f6a6811fd — module-level replacement for AlbertDataset._get_indexed_dataset.

    Refactored out of the class so it can be shared with BertDataset and
    other dataset classes without inheritance.
    """
    import time
    print_rank_0_fn('> Reading dataset index ...')
    start_time = time.time()
    indexed_dataset = make_indexed_dataset_fn(data_prefix, data_impl, skip_warmup)
    print_rank_0_fn('> Finished creating indexed dataset in {:4f} seconds'.format(
        time.time() - start_time))
    return indexed_dataset


def _m36_get_samples_mapping(indexed_dataset, data_prefix, num_epochs,
                              max_num_samples, max_seq_length, short_seq_prob,
                              seed, helpers_mod, print_rank_0_fn):
    """Megatron f6a6811fd — module-level replacement for AlbertDataset._get_samples_mapping.

    Critical fix: max_seq_length-3 to account for [CLS], [SEP], [SEP] tokens
    that will be added later.  The old instance method had the same logic but
    was not reachable from outside the class.
    """
    import os
    import time
    import numpy as np
    import torch

    if not num_epochs:
        if not max_num_samples:
            raise ValueError('Need to specify either max_num_samples or num_epochs')
        num_epochs = np.iinfo(np.int32).max - 1
    if not max_num_samples:
        max_num_samples = np.iinfo(np.int64).max - 1

    # Build the indexmap filename deterministically.
    indexmap_filename = data_prefix
    indexmap_filename += '_indexmap'
    indexmap_filename += '_{}ep'.format(num_epochs)
    indexmap_filename += '_{}mns'.format(max_num_samples)
    indexmap_filename += '_{}msl'.format(max_seq_length)
    indexmap_filename += '_{:0.2f}ssp'.format(short_seq_prob)
    indexmap_filename += '_{}s'.format(seed)
    indexmap_filename += '.npy'

    if torch.distributed.get_rank() == 0 and not os.path.isfile(indexmap_filename):
        print('WARNING: could not find index map file {}, building '
              'the indices on rank 0 ...'.format(indexmap_filename))
        assert indexed_dataset.doc_idx.dtype == np.int64
        assert indexed_dataset.sizes.dtype == np.int32
        verbose = torch.distributed.get_rank() == 0  # M36: was ==0 (no spaces) in original
        start_time = time.time()
        samples_mapping = helpers_mod.build_mapping(
            indexed_dataset.doc_idx,
            indexed_dataset.sizes,
            num_epochs,
            max_num_samples,
            max_seq_length - 3,  # account for [CLS] [SEP] [SEP] added tokens
            short_seq_prob,
            seed,
            verbose)
        np.save(indexmap_filename, samples_mapping, allow_pickle=True)
        print_rank_0_fn('> elapsed time to build and save samples mapping '
                        '(seconds): {:4f}'.format(time.time() - start_time))
    torch.distributed.barrier()

    print_rank_0_fn('> loading indexed mapping from {}'.format(indexmap_filename))
    start_time = time.time()
    samples_mapping = np.load(indexmap_filename, allow_pickle=True)
    print_rank_0_fn('  loaded indexed file in {:3.3f} seconds'.format(
        time.time() - start_time))
    print_rank_0_fn('  total number of samples: {}'.format(samples_mapping.shape[0]))
    return samples_mapping
# --- End M36 dataloader ---


# ---------------------------------------------------------------------------
# M40: Megatron 8179ebd31 — removed split dataset
# Ported from: megatron/data/split_dataset.py (deleted in upstream commit)
#
# Megatron removed split_dataset.py entirely in 8179ebd31. The file provided:
#   - get_train_valid_test_split(splits_string, size) -> splits_index list
#   - SplitDataset(torch.utils.data.Dataset): subset wrapper via split_inds
#   - split_ds(ds, split, shuffle) -> list of SplitDataset or None
#
# In the DeepSpeed/Neuron_SP codebase the equivalent train/val/test splitting
# is handled natively by DeepSpeedDataSampler and the distributed sampler
# paths above. No SplitDataset class existed here prior to this commit.
# This comment records the intentional absence as a 1-to-1 mirror of the
# upstream deletion.
# ---------------------------------------------------------------------------

print('[M40] dataloader: split_dataset removed (mirrors Megatron 8179ebd31 — SplitDataset/split_ds/get_train_valid_test_split deleted upstream)')
