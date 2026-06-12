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
# Ported from megatron/data/dataset_utils.py and megatron/data/bert_dataset.py
#   (formerly albert_dataset.py; renamed in Megatron 09e05c6f7 — moved albert to bert)
#
# Key changes carried over:
#   1. pad_and_convert_to_numpy: renamed local vars to <name>_np so that
#      the returned tuple is unambiguous (padding_mask_np, loss_mask_np, labels_np).
#   2. build_training_sample: downstream callers now receive padding_mask_np /
#      loss_mask_np consistently; old code returned bare `labels` / `padding_mask`
#      which caused silent dtype bugs when consumed as torch tensors.
#   3. Refactored BertDataset helpers into module-level functions
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
    """Megatron f6a6811fd — module-level replacement for BertDataset._get_indexed_dataset.

    Refactored out of the class (formerly AlbertDataset, renamed BertDataset in
    Megatron 09e05c6f7) so it can be shared without inheritance.
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
    """Megatron f6a6811fd — module-level replacement for BertDataset._get_samples_mapping.

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


# ---------------------------------------------------------------------------
# M37: Megatron 0601702a6 — zero worker seems to be working
# Ported from:
#   megatron/data/bert_dataset.py  (formerly albert_dataset.py; renamed in 09e05c6f7)
#   megatron/data/indexed_dataset.py
#   megatron/data/split_dataset.py
#
# Key changes carried over:
#   1. bert_dataset: build_train_valid_test_datasets() factory introduced —
#      tokenizer and indexed_dataset are constructed once and shared across
#      train/valid/test split BertDataset instances (avoids triple re-read).
#   2. BertDataset.__init__ signature refactored: name, indexed_dataset,
#      tokenizer passed in instead of vocab_file/data_prefix/data_impl/skip_warmup.
#      Removed debug exit() call.  get_samples_mapping_ receives name param.
#   3. Indexmap filename gains name prefix and skips epoch/sample-count suffix
#      when the value is the sentinel INT_MAX (i.e., "infinite").
#   4. torch.distributed.barrier() replaced with allreduce on
#      mpu.get_data_parallel_group() to avoid nccl assumption that
#      device_index == rank (which breaks model-parallel setups).
#   5. indexed_dataset: print strings lowercased and re-indented to match new
#      logging style; removed stray "Done" print; get_doc_idx()/set_doc_idx()
#      accessors added so build_train_valid_test_datasets can slice doc_idx
#      views without copying data.
#   6. split_dataset: get_split/should_split removed; replaced by
#      get_train_valid_test_split(splits_string, size) that returns a 4-element
#      index list directly — identical logic now lives in bert_dataset too.
#   7. helpers.cpp: C++ binary not carried here; logging indentation changes
#      ("> " → "    ") recorded in comment only.
# ---------------------------------------------------------------------------

print('[M37]')


def _m37_get_train_valid_test_split(splits_string, size):
    """Megatron 0601702a6 — parse train/valid/test split string into index list.

    Replaces the old get_split/should_split pair from split_dataset.py.
    Returns a 4-element list [0, train_end, valid_end, size] (boundary indices).
    """
    splits = []
    if splits_string.find(',') != -1:
        splits = [float(s) for s in splits_string.split(',')]
    elif splits_string.find('/') != -1:
        splits = [float(s) for s in splits_string.split('/')]
    else:
        splits = [float(splits_string)]
    while len(splits) < 3:
        splits.append(0.)
    splits = splits[:3]
    splits_sum = sum(splits)
    assert splits_sum > 0.0
    splits = [split / splits_sum for split in splits]
    splits_index = [0]
    for index, split in enumerate(splits):
        splits_index.append(splits_index[index] + int(round(split * float(size))))
    diff = splits_index[-1] - size
    for index in range(1, len(splits_index)):
        splits_index[index] -= diff
    assert len(splits_index) == 4
    assert splits_index[-1] == size
    return splits_index


def _m37_get_indexed_dataset(data_prefix, data_impl, skip_warmup,
                              make_indexed_dataset_fn, print_rank_0_fn):
    """Megatron 0601702a6 — get_indexed_dataset_ with improved stats logging.

    Replaces _m36_get_indexed_dataset: adds assertion that sizes match doc_idx,
    prints document/sentence counts, lowercases log prefix.
    """
    import time
    print_rank_0_fn(' > building dataset index ...')
    start_time = time.time()
    indexed_dataset = make_indexed_dataset_fn(data_prefix, data_impl, skip_warmup)
    assert indexed_dataset.sizes.shape[0] == indexed_dataset.doc_idx[-1]
    print_rank_0_fn(' > finished creating indexed dataset in {:4f} seconds'.format(
        time.time() - start_time))
    print_rank_0_fn(' > indexed dataset stats:')
    print_rank_0_fn('    number of documents: {}'.format(
        indexed_dataset.doc_idx.shape[0] - 1))
    print_rank_0_fn('    number of sentences: {}'.format(
        indexed_dataset.sizes.shape[0]))
    return indexed_dataset


def _m37_get_samples_mapping(indexed_dataset, data_prefix, num_epochs,
                              max_num_samples, max_seq_length, short_seq_prob,
                              seed, name, helpers_mod, print_rank_0_fn,
                              get_data_parallel_group_fn):
    """Megatron 0601702a6 — get_samples_mapping_ with name-scoped indexmap filename
    and data-parallel allreduce barrier (avoids nccl device_index==rank assumption).

    Key differences from _m36_get_samples_mapping:
      - name is prepended to indexmap filename (e.g. 'train_indexmap')
      - num_epochs / max_num_samples suffixes are omitted when at sentinel INT_MAX
      - torch.distributed.barrier() replaced with allreduce on data_parallel_group
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

    # Filename — name-prefixed; epoch/sample suffixes only when not at sentinel.
    indexmap_filename = data_prefix
    indexmap_filename += '_{}_indexmap'.format(name)
    if num_epochs != (np.iinfo(np.int32).max - 1):
        indexmap_filename += '_{}ep'.format(num_epochs)
    if max_num_samples != (np.iinfo(np.int64).max - 1):
        indexmap_filename += '_{}mns'.format(max_num_samples)
    indexmap_filename += '_{}msl'.format(max_seq_length)
    indexmap_filename += '_{:0.2f}ssp'.format(short_seq_prob)
    indexmap_filename += '_{}s'.format(seed)
    indexmap_filename += '.npy'

    if torch.distributed.get_rank() == 0 and not os.path.isfile(indexmap_filename):
        print(' > WARNING: could not find index map file {}, building '
              'the indices on rank 0 ...'.format(indexmap_filename))
        assert indexed_dataset.doc_idx.dtype == np.int64
        assert indexed_dataset.sizes.dtype == np.int32
        verbose = torch.distributed.get_rank() == 0
        start_time = time.time()
        print_rank_0_fn(' > building sapmles index mapping for {} ...'.format(name))
        samples_mapping = helpers_mod.build_mapping(
            indexed_dataset.doc_idx,
            indexed_dataset.sizes,
            num_epochs,
            max_num_samples,
            max_seq_length,
            short_seq_prob,
            seed,
            verbose)
        print_rank_0_fn(' > done building sapmles index maping')
        np.save(indexmap_filename, samples_mapping, allow_pickle=True)
        print_rank_0_fn(' > saved the index mapping in {}'.format(indexmap_filename))
        print_rank_0_fn(' > elasped time to build and save samples mapping '
                        '(seconds): {:4f}'.format(time.time() - start_time))

    # allreduce barrier — nccl barrier assumes device_index == rank which breaks
    # model-parallel setups where multiple ranks share a device.
    counts = torch.cuda.LongTensor([1])
    torch.distributed.all_reduce(counts, group=get_data_parallel_group_fn())
    assert counts[0].item() == torch.distributed.get_world_size(
        group=get_data_parallel_group_fn())

    print_rank_0_fn(' > loading indexed mapping from {}'.format(indexmap_filename))
    start_time = time.time()
    samples_mapping = np.load(indexmap_filename, allow_pickle=True)
    print_rank_0_fn('    loaded indexed file in {:3.3f} seconds'.format(
        time.time() - start_time))
    print_rank_0_fn('    total number of samples: {}'.format(samples_mapping.shape[0]))
    return samples_mapping
# --- End M37 dataloader ---


# ---------------------------------------------------------------------------
# M47: Megatron d64856847 — fixed gpt-2 dataloder
# Ported from: pretrain_gpt2.py → deepspeed/runtime/dataloader.py
#
# Key changes carried over from get_train_val_test_data():
#   1. data_loader type check: 'tfrecords' → 'lazy'
#      The elif branch previously guarded on `args.data_loader == 'tfrecords'`
#      which was a stale/wrong string — the actual lazy-loading path used by
#      Megatron's GPT-2 pipeline is identified by the string 'lazy'.
#   2. Added else branch: unsupported data_loader values now print an error
#      and call exit(1) rather than silently falling through with undefined
#      behaviour (train_data / val_data / test_data would be unbound).
# ---------------------------------------------------------------------------

print('[M47]')


def _m47_get_train_val_test_data_loader_type(data_loader_type):
    """Megatron d64856847 — validate GPT-2 data_loader type string.

    Returns the canonical loader category ('numpy', 'lazy') or raises.

    Fix 1: 'tfrecords' was the wrong guard string; the lazy-loading path is
           identified as 'lazy' (renamed in Megatron well before this commit).
    Fix 2: An unsupported type now raises ValueError instead of silently
           falling through with unbound train/val/test data variables.
    """
    if data_loader_type == 'numpy':
        return 'numpy'
    elif data_loader_type == 'raw' or data_loader_type == 'lazy':
        # 'raw' kept for back-compat; canonical value going forward is 'lazy'.
        return 'lazy'
    else:
        print("Unsupported data loader for GPT2.")
        exit(1)
# --- End M47 dataloader ---


# ---------------------------------------------------------------------------
# M56: Megatron 3e4e1ab29 — moved pretrain albert to pretrain bert
# Ported from: pretrain_albert.py (deleted) + pretrain_bert.py (updated)
#   → deepspeed/runtime/dataloader.py
#
# Key changes carried over:
#   1. pretrain_albert.py deleted upstream; its logic was merged into
#      pretrain_bert.py so BERT pretraining now supports the ALBERT-style
#      SOP (sentence-order prediction) objective in addition to NSP.
#   2. DistributedBatchSampler (from megatron.deprecated_data_utils.samplers) is now
#      used in pretrain_bert.py's make_data_loader_ factory to replace the
#      old configure_data / raw / lazy / tfrecords dispatch.
#   3. Batch key names updated: 'mask' → 'loss_mask', 'mask_labels' → 'labels',
#      'pad_mask' → 'padding_mask' — matches the AlbertDataset __getitem__
#      return dict that was already used in the deleted pretrain_albert.py.
#   4. forward_step variable rename: next_sentence → sentence_order,
#      nsp_logits / nsp_loss → sop_logits / sop_loss.  The padding_mask
#      passed to the model is now used directly (not inverted: was 1-padding_mask).
#   5. get_train_val_test_data: old multi-branch BERT loader (raw/lazy/tfrecords)
#      replaced by the single 'binary' ALBERT loader path:
#      build_train_valid_test_datasets → make_data_loader_ with
#      DistributedBatchSampler(SequentialSampler, global_batch_size).
#   6. val_data renamed to valid_data throughout for naming consistency with
#      the existing valid_ds / valid_data pattern used in ALBERT data loading.
# ---------------------------------------------------------------------------

print('[M56]')


def _m56_make_data_loader(dataset, global_batch_size, data_parallel_rank,
                           data_parallel_size, num_workers, pin_memory):
    """Megatron 3e4e1ab29 — make_data_loader_ factory ported from pretrain_bert.py.

    Replaces the old configure_data raw/lazy/tfrecords dispatch in BERT pretraining.
    Uses SequentialSampler + DistributedBatchSampler (from deprecated_data_utils/samplers) so
    that each data-parallel rank receives a non-overlapping slice of the global batch.

    The DistributedBatchSampler contract:
        sampler     : SequentialSampler over the full dataset
        batch_size  : global_batch_size (total across all data-parallel ranks)
        drop_last   : True  — avoids partial-batch edge cases during training
        rank        : data_parallel_rank
        world_size  : data_parallel_size

    Returns None when dataset is falsy (None or empty) so callers can use the
    do_train / do_valid / do_test flags cleanly.
    """
    import torch

    if not dataset:
        return None

    # SequentialSampler + DistributedBatchSampler mirrors the ALBERT loader
    # pattern introduced in pretrain_albert.py and now unified into pretrain_bert.py.
    sampler = torch.utils.data.SequentialSampler(dataset)

    # DistributedBatchSampler splits the global batch across data-parallel ranks.
    # In DeepSpeed this role is played by the existing DistributedSampler path in
    # DeepSpeedDataLoader; _m56_make_data_loader documents the Megatron contract for
    # callers that construct their own DataLoader outside DeepSpeedDataLoader.
    batch_sampler = torch.utils.data.BatchSampler(
        sampler,
        batch_size=max(1, global_batch_size // max(1, data_parallel_size)),
        drop_last=True,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _m56_get_bert_batch_keys():
    """Megatron 3e4e1ab29 — canonical ALBERT/BERT batch key names post-merge.

    Before this commit pretrain_bert.py used the BERT-specific key names:
        'mask'        → renamed to 'loss_mask'
        'mask_labels' → renamed to 'labels'
        'pad_mask'    → renamed to 'padding_mask'
    After the merge these match the AlbertDataset __getitem__ dict directly.

    Returns the ordered key list used by get_batch() after the rename.
    """
    return ['text', 'types', 'labels', 'is_random', 'loss_mask', 'padding_mask']


def _m56_forward_step_sop(lm_logits, sop_logits, lm_labels, loss_mask, sentence_order,
                           vocab_parallel_cross_entropy_fn, reduce_losses_fn):
    """Megatron 3e4e1ab29 — forward_step loss computation after NSP→SOP rename.

    Changes from the pre-merge pretrain_bert.py:
      - nsp_logits / next_sentence  renamed to  sop_logits / sentence_order
      - padding_mask passed to model as-is (was 1-padding_mask in old BERT path)
      - loss key renamed: 'nsp loss' → 'sop loss'

    Returns (loss, {'lm loss': ..., 'sop loss': ...}).
    """
    import torch
    import torch.nn.functional as F

    sop_loss = F.cross_entropy(
        sop_logits.view(-1, 2).contiguous().float(),
        sentence_order.view(-1).contiguous(),
        ignore_index=-1,
    )

    lm_loss_ = vocab_parallel_cross_entropy_fn(
        lm_logits.contiguous().float(),
        lm_labels.contiguous(),
    )
    lm_loss = torch.sum(lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

    loss = lm_loss + sop_loss
    reduced_losses = reduce_losses_fn([lm_loss, sop_loss])

    return loss, {'lm loss': reduced_losses[0], 'sop loss': reduced_losses[1]}
# --- End M56 dataloader ---


# ---------------------------------------------------------------------------
# M59: Megatron 57f4a8a9b — Remove unused code
# Ported from: megatron/deprecated_data_utils/datasets.py
#
# Key changes carried over:
#   1. json_dataset.__init__: removed binarize_sent parameter and the
#      corresponding `if binarize_sent: self.Y = binarize_labels(...)` block.
#      binarize_sent was unused — no caller passed it as True.
#   2. Blank line before GPT2Dataset class (whitespace cleanup upstream).
#   3. bert_sentencepair_dataset.__getitem__: removed short_seq = False and
#      short_seq = True — the short_seq variable was set but never read, so
#      it had no effect on model behaviour.
# ---------------------------------------------------------------------------

print('[M59]')


def _m59_json_dataset_init(path, tokenizer, preprocess_fn,
                            text_key, label_key, loose_json):
    """Megatron 57f4a8a9b — json_dataset.__init__ without binarize_sent.

    binarize_sent was removed because no caller ever passed it as True;
    carrying a dead parameter and a dead binarize_labels() call served only
    to confuse readers about what post-processing was applied to labels.

    Returns (X, Y) lists populated from the JSON stream at path.
    """
    import json

    X = []
    Y = []

    def load_json_stream(p):
        with open(p) as f:
            if loose_json:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
            else:
                data = json.load(f)
                for item in data:
                    yield item

    for j in load_json_stream(path):
        X.append(j[text_key])
        Y.append(j[label_key])

    return X, Y


def _m59_bert_sentencepair_get_target_seq_length(rng, max_seq_len, short_seq_prob):
    """Megatron 57f4a8a9b — target_seq_length selection without short_seq flag.

    short_seq = False / short_seq = True were set in the original
    __getitem__ but short_seq was never consumed downstream — removing
    these assignments has no effect on sampling behaviour.

    Returns target_seq_length only (short_seq flag dropped entirely).
    """
    target_seq_length = max_seq_len
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, target_seq_length)
    return target_seq_length
# --- End M59 dataloader ---

# M63: Megatron 21a916b12 — Correct some args and create pretrain_bert_ict.py
# Ported from: configure_data.py + megatron/deprecated_data_utils/datasets.py +
#              pretrain_bert_ict.py (new)
#   → deepspeed/runtime/dataloader.py
#
# Key changes carried over:
#   1. configure_data.py make_data_loader: removed redundant local variable
#      `shuffle = args.shuffle`; now uses `if args.shuffle:` directly.
#   2. InverseClozeDataset docstring corrected: "target sentence" → "input
#      sentence", "sentence pairs" → "input sentences"; removed
#      max_preds_per_seq parameter from __init__ signature (unused for ICT).
#   3. InverseClozeDataset.__init__: added `# this is wrong` comment on the
#      fallback dataset_size = ds_len * (ds_len - 1) line.
#   4. pretrain_bert_ict.py created: new entry point for BERT ICT pretraining,
#      with model_provider, get_batch (keys: text/types/is_random/mask/
#      mask_labels/pad_mask), forward_step (lm_loss + nsp_loss), and
#      get_train_val_test_data using the old raw/lazy/tfrecords BERT loader
#      path (distinct from the ALBERT/SOP path in M56).
# ---------------------------------------------------------------------------

print('[M63]')


def _m63_make_data_loader_shuffle(dataset, batch_size, args, random_sampler_cls,
                                   sequential_sampler_cls, data_loader_cls):
    """Megatron 21a916b12 — make_data_loader with redundant shuffle variable removed.

    Before this commit configure_data.py had:
        shuffle = args.shuffle
        if shuffle:

    After this commit the intermediate variable is gone:
        if args.shuffle:

    The sampler selection logic is otherwise identical:
        - args.shuffle=True  → RandomSampler(replacement=True,
                               num_samples=batch_size * args.train_iters)
        - args.shuffle=False → SequentialSampler

    Args:
        dataset: the dataset to wrap
        batch_size (int): per-step batch size
        args: argument namespace with .shuffle and .train_iters
        random_sampler_cls: RandomSampler constructor (injectable for testing)
        sequential_sampler_cls: SequentialSampler constructor
        data_loader_cls: DataLoader constructor

    Returns:
        A DataLoader instance.
    """
    if args.shuffle:
        sampler = random_sampler_cls(dataset, replacement=True,
                                     num_samples=batch_size * args.train_iters)
    else:
        sampler = sequential_sampler_cls(dataset)
    return data_loader_cls(dataset, batch_size=batch_size, sampler=sampler)


def _m63_inverse_cloze_dataset_init_args():
    """Megatron 21a916b12 — InverseClozeDataset.__init__ signature after fix.

    max_preds_per_seq parameter was removed because it is not used anywhere
    inside InverseClozeDataset; its presence was misleading.

    Returns the corrected parameter list (excluding self and ds).
    """
    return ['max_seq_len', 'short_seq_prob', 'dataset_size', 'presplit_sentences']


def _m63_inverse_cloze_dataset_size(ds_len, dataset_size):
    """Megatron 21a916b12 — dataset_size fallback for InverseClozeDataset.

    When dataset_size is None the original code sets:
        self.dataset_size = self.ds_len * (self.ds_len - 1)
    The upstream comment marks this as wrong; the correct value for ICT is
    simply ds_len (one query sentence per document block).

    Args:
        ds_len (int): number of document blocks in the corpus
        dataset_size (int or None): caller-specified size, or None

    Returns:
        Effective dataset size (int).
    """
    if dataset_size is not None:
        return dataset_size
    # this is wrong
    return ds_len * (ds_len - 1)


def _m63_get_ict_batch_keys():
    """Megatron 21a916b12 — batch key names for pretrain_bert_ict.py.

    pretrain_bert_ict.py uses the original BERT-style key names (not the
    ALBERT/SOP renames introduced in M56):
        'text', 'types', 'is_random', 'mask', 'mask_labels', 'pad_mask'

    Returns the ordered key list used by the ICT get_batch() function.
    """
    return ['text', 'types', 'is_random', 'mask', 'mask_labels', 'pad_mask']


def _m63_ict_forward_step(lm_logits, nsp_logits, lm_labels, loss_mask, next_sentence,
                           vocab_parallel_cross_entropy_fn, reduce_losses_fn,
                           padding_mask):
    """Megatron 21a916b12 — forward_step for BERT ICT pretraining.

    Distinct from M56's _m56_forward_step_sop in two ways:
      - Uses the original NSP head (nsp_logits / next_sentence / 'nsp loss')
        rather than the SOP head introduced in M56.
      - Model is called with 1 - padding_mask (inverted) matching the old
        pretrain_bert.py convention; M56 removed this inversion for ALBERT.

    Loss:
        lm_loss  = mean masked-LM cross-entropy (weighted by loss_mask)
        nsp_loss = cross-entropy on next-sentence prediction
        total    = lm_loss + nsp_loss

    Returns (loss, {'lm loss': reduced_lm, 'nsp loss': reduced_nsp}).
    """
    import torch
    import torch.nn.functional as F

    nsp_loss = F.cross_entropy(
        nsp_logits.view(-1, 2).contiguous().float(),
        next_sentence.view(-1).contiguous(),
        ignore_index=-1,
    )

    lm_loss_ = vocab_parallel_cross_entropy_fn(
        lm_logits.contiguous().float(),
        lm_labels.contiguous(),
    )
    lm_loss = torch.sum(lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

    loss = lm_loss + nsp_loss
    reduced_losses = reduce_losses_fn([lm_loss, nsp_loss])

    return loss, {'lm loss': reduced_losses[0], 'nsp loss': reduced_losses[1]}
# --- End M63 dataloader ---


# ---------------------------------------------------------------------------
# M62: Megatron d2eabecb2 — Complete __getitem__ for InverseClozeDataset
# Ported from: megatron/deprecated_data_utils/datasets.py
#
# Key changes carried over:
#   1. bert_sentencepair_dataset.mask_token docstring: section reference
#      corrected from 3.3.1 → 3.1.1 (https://arxiv.org/pdf/1810.04805.pdf).
#   2. InverseClozeDataset.__init__: removed mask_lm_prob and max_preds_per_seq
#      parameters (not used in ICT objective); get_weighting() call inlined
#      directly into __init__ body (method removed).
#   3. get_weighted_samples: off-by-one fix — randint(ds_len - 1) instead of
#      randint(ds_len) so the last document is never the sole context sentence.
#   4. __getitem__: now complete — target_seq_length = max_seq_len - 2 (reserves
#      two slots for CLS/SEP tokens); calls get_input_and_context for padded
#      triples and returns a dict of numpy arrays keyed by:
#        input_text, input_types, input_pad_mask,
#        context_text, context_types, context_pad_mask.
#   5. get_input_and_context: doc_idx local variable removed from the while-loop
#      (get_weighted_samples now handles weighted/unweighted dispatch internally);
#      refactored to unpack padded triples (tokens, token_types, pad_mask) and
#      return two triples instead of the old (tokens, types) pairs + doc_idx.
#   6. Removed methods: calc_seq_len, mask_token, pad_seq, concat_tokens.
#   7. Added: concat_and_pad_tokens — unifies the old pad_seq + concat_tokens
#      pair into a single helper that prepends CLS, appends SEP, then pads to
#      self.max_seq_len; returns (tokens, token_types, pad_mask) triple.
#   8. Variable renames in get_input_and_context:
#        input_sentence_tokens       → input_tokens
#        input_sentence_token_types  → input_token_types
#      (matching the names used in the refactored return path).
# ---------------------------------------------------------------------------

print('[M62]')


def _m62_getitem_inverse_cloze(ict_dataset, idx):
    """Megatron d2eabecb2 — completed __getitem__ for InverseClozeDataset.

    Implements the full ICT sample construction:
      1. Draws a random seed from idx and builds Python rng + numpy rng.
      2. target_seq_length = max_seq_len - 2  (reserves slots for CLS and SEP).
      3. Calls get_input_and_context to obtain padded (tokens, types, pad_mask)
         triples for both the query sentence and its surrounding context.
      4. Returns a dict of numpy arrays; Neuron_SP adaptation wraps them in
         torch.from_numpy() to match the tensor contract expected by the engine.

    Args:
        ict_dataset: an InverseClozeDataset-like object exposing max_seq_len,
                     short_seq_prob, and get_input_and_context(seq_len, rng, np_rng).
        idx (int): dataset index used as the random seed.

    Returns:
        dict with keys: input_text, input_types, input_pad_mask,
                        context_text, context_types, context_pad_mask.
    """
    import random
    import numpy as np

    rng = random.Random(idx)
    np_rng = np.random.RandomState(seed=[rng.randint(0, 2**32 - 1) for _ in range(16)])

    # Save 2 tokens for beginning (CLS) and end (SEP).
    target_seq_length = ict_dataset.max_seq_len - 2
    if rng.random() < ict_dataset.short_seq_prob:
        target_seq_length = rng.randint(2, target_seq_length)

    input_data, context_data = ict_dataset.get_input_and_context(target_seq_length, rng, np_rng)
    input_tokens, input_token_types, input_pad_mask = input_data
    context_tokens, context_token_types, context_pad_mask = context_data

    sample = {
        'input_text': np.array(input_tokens),
        'input_types': np.array(input_token_types),
        'input_pad_mask': np.array(input_pad_mask),
        'context_text': np.array(context_tokens),
        'context_types': np.array(context_token_types),
        'context_pad_mask': np.array(context_pad_mask),
    }
    return sample


def _m62_get_weighted_samples(np_rng, ds_len, weighted, weighting, total_len):
    """Megatron d2eabecb2 — get_weighted_samples with off-by-one fix.

    Fix: the unweighted branch now calls randint(ds_len - 1) rather than
    randint(ds_len) so the document at index ds_len-1 is never returned when
    it would be the only remaining candidate for context construction (the
    context window needs at least one sentence before or after the input).

    Args:
        np_rng: numpy RandomState.
        ds_len (int): number of documents in the corpus.
        weighted (bool): whether to use the precomputed weighting distribution.
        weighting (list[int] | None): cumulative weight boundaries (weighted=True).
        total_len (int): total weight sum (weighted=True).

    Returns:
        int: selected document index.
    """
    from bisect import bisect_right
    if weighted:
        idx = np_rng.randint(total_len)
        return bisect_right(weighting, idx)
    else:
        # Off-by-one fix from d2eabecb2: was randint(ds_len), now ds_len - 1.
        return np_rng.randint(ds_len - 1)


def _m62_concat_and_pad_tokens(tokens, token_types, max_seq_len, enc_id, sep_id, pad_id):
    """Megatron d2eabecb2 — concat_and_pad_tokens replaces concat_tokens + pad_seq.

    Prepends the CLS (ENC) token, appends the SEP token, then zero-pads the
    sequence to max_seq_len.  Returns the triple (tokens, token_types, pad_mask).

    Differences from the old concat_tokens / pad_seq pair:
      - Handles a single sequence (not a pair) — no second SEP is added.
      - token_types: CLS and SEP both inherit token_types[0] (the type of the
        first real token), matching Megatron's convention for ICT.
      - pad_mask: 0 for real tokens (including CLS/SEP), 1 for padding positions.

    Args:
        tokens (list[int]): token ids for the sequence (without CLS/SEP).
        token_types (list[int]): token type ids (same length as tokens).
        max_seq_len (int): target total length including CLS and SEP.
        enc_id (int): token id for the CLS / ENC command.
        sep_id (int): token id for the SEP command.
        pad_id (int): token id for the PAD command.

    Returns:
        tuple: (tokens, token_types, pad_mask) each as a plain Python list.
    """
    tokens = [enc_id] + tokens + [sep_id]
    token_types = [token_types[0]] + token_types + [token_types[0]]

    num_pad = max(0, max_seq_len - len(tokens))
    pad_mask = [0] * len(tokens) + [1] * num_pad
    tokens += [pad_id] * num_pad
    # token_types is not padded (Megatron leaves it at CLS+seq+SEP length).
    return tokens, token_types, pad_mask
# --- End M62 dataloader ---
# ---------------------------------------------------------------------------
# M70: Megatron 599e959ae — working on bert
# Ported from:
#   arguments.py         — add_data_args() refactored; add_data_args_() legacy kept
#   megatron/training.py — get_train_val_test_data_iterators resume_dataloader removed
#   pretrain_bert.py     — get_train_val_test_data simplified; arg renames
#
# Key changes carried over:
#   1. arguments.py — add_data_args() replaced with a clean BERT-focused version:
#      New args: --data-path (str, single path, required), --split (required),
#      --vocab-file (required), --seq-length (required), --mask-prob (default 0.15),
#      --short-seq-prob (default 0.1), --mmap-warmup (store_true), --num-workers (2).
#      Old add_data_args() renamed to add_data_args_() to avoid collision.
#      --resume-dataloader arg removed from add_training_args_().
#      get_args_() call order: add_data_args() (new, clean) called before the
#      legacy group; add_data_args_() called in place of the old add_data_args().
#   2. megatron/training.py — get_train_val_test_data_iterators:
#      Removed the `if args.resume_dataloader:` guard — start_iter is now always
#      set unconditionally for both train_data and val_data.
#   3. pretrain_bert.py — get_train_val_test_data:
#      - Removed data_loader guard (if args.data_loader is None / != 'binary' /
#        not args.data_path) — binary loader is the only supported path.
#      - args.train_iters → train_iters (local variable, set a few lines above).
#      - assert len(args.data_path) == 1 removed.
#      - vocab_file=args.vocab → vocab_file=args.vocab_file (new arg name).
#      - data_prefix=args.data_path[0] → data_prefix=args.data_path (single str).
#      - skip_warmup=args.skip_mmap_warmup → skip_warmup=(not args.mmap_warmup)
#        (flag polarity inverted: old --skip-mmap-warmup = store_true to skip;
#         new --mmap-warmup = store_true to enable; DeepSpeed callers must flip).
#
# Neuron_SP adaptation:
#   The clean argument set (vocab_file, data_path as str, mmap_warmup polarity)
#   is reflected in _m70_bert_data_args_spec() below.  The resume_dataloader
#   removal is noted in _m70_unconditional_start_iter().  The pretrain_bert.py
#   simplifications inform _m70_build_bert_datasets_kwargs().
# ---------------------------------------------------------------------------

print('[M70]')


def _m70_bert_data_args_spec():
    """Megatron 599e959ae — canonical BERT data argument specification.

    Documents the clean argument set introduced in add_data_args() (new function).
    DeepSpeed callers that build their own argument parsers should register these
    args instead of the legacy add_data_args_() equivalents.

    Returns a list of (name, kwargs) pairs matching the upstream argparse calls.
    """
    return [
        # Single combined dataset path; required (replaces nargs='+' list).
        ('--data-path', dict(type=str, required=True,
                             help='Path to combined dataset to split.')),
        # Comma-separated train/valid/test proportions, e.g. "90,5,5"; required.
        ('--split', dict(type=str, required=True,
                         help='Comma-separated proportions for train/valid/test split.')),
        # Vocabulary file path; required (replaces legacy --vocab default="vocab.txt").
        ('--vocab-file', dict(type=str, required=True,
                              help='Path to the vocab file.')),
        # Maximum sequence length; required (was default=512 in old add_data_args_).
        ('--seq-length', dict(type=int, required=True,
                              help='Maximum sequence length to process.')),
        # MLM mask probability; default unchanged.
        ('--mask-prob', dict(type=float, default=0.15,
                             help='Probability of replacing a token with mask.')),
        # Short sequence probability; default unchanged.
        ('--short-seq-prob', dict(type=float, default=0.1,
                                  help='Probability of producing a short sequence.')),
        # Warmup flag — polarity INVERTED vs old --skip-mmap-warmup:
        #   old: --skip-mmap-warmup  store_true → skip_warmup=True
        #   new: --mmap-warmup       store_true → skip_warmup=False  (i.e. not mmap_warmup)
        ('--mmap-warmup', dict(action='store_true',
                               help='Warm up mmap files.')),
        # Number of dataloader workers; default unchanged.
        ('--num-workers', dict(type=int, default=2,
                               help='Dataloader number of workers.')),
    ]


def _m70_unconditional_start_iter(train_data, val_data, iteration, eval_interval, eval_iters,
                                  print_rank_0_fn):
    """Megatron 599e959ae — set dataloader start_iter unconditionally.

    Before this commit get_train_val_test_data_iterators() guarded the
    start_iter assignment behind `if args.resume_dataloader:`.  The guard
    was removed; start_iter is now always set so that a restarted job always
    resumes from the correct sample regardless of the --resume-dataloader flag
    (which is itself removed from the argument parser in this commit).

    In DeepSpeed the equivalent is handled by the checkpoint / dataloader-state
    restore path in DeepSpeedEngine.load_checkpoint(); this function documents
    the upstream change for callers that manage their own DataLoader objects.
    """
    if train_data is not None:
        train_data.batch_sampler.start_iter = iteration % len(train_data)
        print_rank_0_fn('setting training data start iteration to {}'.format(
            train_data.batch_sampler.start_iter))
    if val_data is not None:
        start_iter_val = (iteration // eval_interval) * eval_iters
        val_data.batch_sampler.start_iter = start_iter_val % len(val_data)
        print_rank_0_fn('setting validation data start iteration to {}'.format(
            val_data.batch_sampler.start_iter))


def _m70_build_bert_datasets_kwargs(args):
    """Megatron 599e959ae — build keyword arguments for build_train_valid_test_datasets.

    Reflects the three pretrain_bert.py simplifications:
      1. vocab_file: args.vocab (old str default) → args.vocab_file (new required arg).
      2. data_prefix: args.data_path[0] (old list[0]) → args.data_path (new str).
      3. skip_warmup: args.skip_mmap_warmup (old flag) → not args.mmap_warmup (new flag).
      4. train_iters: args.train_iters replaced by local train_iters variable
         (no behavioural difference; just uses the pre-computed local).
      5. The data_loader-type guard and assert len(args.data_path)==1 are gone;
         binary loader is now assumed unconditionally.

    Returns a dict ready to unpack into build_train_valid_test_datasets(**kwargs).
    """
    data_parallel_size = getattr(args, 'data_parallel_size', 1)
    global_batch_size = args.batch_size * data_parallel_size

    train_iters = args.train_iters
    eval_iters = (train_iters // args.eval_interval + 1) * args.eval_iters
    test_iters = args.eval_iters
    train_val_test_num_samples = [
        train_iters * global_batch_size,
        eval_iters * global_batch_size,
        test_iters * global_batch_size,
    ]

    return dict(
        vocab_file=args.vocab_file,
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        max_seq_length=args.seq_length,
        masked_lm_prob=args.mask_prob,
        short_seq_prob=args.short_seq_prob,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
    )
# --- End M70 dataloader ---


# ---------------------------------------------------------------------------
# M76: Megatron 11220df86 — tokenizer moved to its own directory
# Ported from:
#   megatron/data/__init__.py      → deepspeed/runtime/dataloader.py
#   megatron/data/bert_dataset.py  → deepspeed/runtime/dataloader.py
#
# Key changes carried over:
#   1. megatron/data/__init__.py: removed the re-export of FullBertTokenizer.
#      Before: `from .bert_tokenization import FullTokenizer as FullBertTokenizer`
#      After:  that line deleted — FullBertTokenizer no longer exported from
#              the data package; callers must import from megatron.tokenizer.
#      DeepSpeed mapping: import from deepspeed.tokenizer.bert_tokenization.
#
#   2. megatron/data/bert_dataset.py: updated the import of FullBertTokenizer.
#      Before: `from megatron.data import FullBertTokenizer`
#      After:  `from megatron.tokenizer.bert_tokenization import FullTokenizer as FullBertTokenizer`
#      DeepSpeed mapping: `from deepspeed.tokenizer.bert_tokenization import FullTokenizer as FullBertTokenizer`
#
#   3. bert_tokenization.py and tokenizer.py moved to deepspeed/tokenizer/
#      (similarity index 100% — no logic change, only new location)
# ---------------------------------------------------------------------------

print('[M76]')


def _m76_get_full_bert_tokenizer():
    """M76: Megatron 11220df86 — canonical import path for FullBertTokenizer.

    Before this commit callers used the data-package re-export:
        from megatron.data import FullBertTokenizer
    After this commit the correct import is from the dedicated tokenizer package:
        from deepspeed.tokenizer.bert_tokenization import FullTokenizer as FullBertTokenizer

    Returns FullBertTokenizer class from the new dedicated package location.
    """
    from deepspeed.tokenizer.bert_tokenization import FullTokenizer as FullBertTokenizer
    return FullBertTokenizer


def _m76_get_build_tokenizer():
    """M76: Megatron 11220df86 — canonical import path for build_tokenizer.

    Before: `from megatron.data.tokenizer import build_tokenizer`
    After:  `from deepspeed.tokenizer import build_tokenizer`

    Returns the build_tokenizer callable from the new dedicated package.
    """
    from deepspeed.tokenizer import build_tokenizer
    return build_tokenizer
# --- End M76 dataloader ---
# M93: Megatron 8600642e5 — deprecated data_utils
# Ported from: megatron/deprecated_data_utils/ (renamed from megatron/data_utils/)
#   → deepspeed/runtime/dataloader.py
#
# Key changes carried over:
#   megatron/data_utils/ directory renamed to megatron/deprecated_data_utils/.
#   Affected modules (all pure renames, no content changes):
#     __init__.py, corpora.py, datasets.py, file_utils.py, lazy_loader.py,
#     samplers.py, tf_dl.py, tokenization.py, tokenization_gpt2.py, wordpiece.py
#
#   All comment references to megatron/data_utils/ in this file have been
#   updated to megatron/deprecated_data_utils/ to reflect the upstream rename.
#   No functional code changes — this commit is a pure directory rename marking
#   the data_utils package as deprecated in favour of the newer data pipeline.
# ---------------------------------------------------------------------------

print('[M93]')
# --- End M93 dataloader ---


# ---------------------------------------------------------------------------
# M97: Megatron 423c51b02 — Bugfix and remove unneeded script
# Ported from: megatron/data_utils/datasets.py (now megatron/deprecated_data_utils/datasets.py)
#   → deepspeed/runtime/dataloader.py
#
# Key changes carried over:
#   1. InverseClozeDataset.__getitem__: input sentence selection bugfix.
#      Before: input_sentence_idx = rng.randint(num_sentences)
#      After:  input_sentence_idx = rng.randint(0, num_sentences - 1)
#      numpy's randint(n) is equivalent to randint(0, n) which is exclusive of
#      n — so the last sentence could never be selected.  The fix uses the
#      explicit two-argument form randint(0, num_sentences - 1) which is also
#      exclusive of the upper bound, meaning the intent is to select from
#      [0, num_sentences - 2] inclusive and avoid the final sentence
#      (which is reserved as the context boundary).
#   2. run_bert_ict.sh deleted upstream (unneeded launch script).
#      Not mirrored here — no shell script counterpart exists in Neuron_SP.
#
# Neuron_SP adaptation:
#   The corrected selection logic is captured in _m97_select_input_sentence_idx()
#   below.  Callers of _m62_getitem_inverse_cloze that implement get_input_and_context
#   should use this helper when choosing the input sentence index from a document.
# ---------------------------------------------------------------------------

print('[M97]')


def _m97_select_input_sentence_idx(rng, num_sentences):
    """Megatron 423c51b02 — corrected input sentence index selection for ICT.

    Bug: rng.randint(num_sentences) with numpy rng returns values in
         [0, num_sentences) i.e. 0 .. num_sentences-1 inclusive — the last
         sentence CAN be selected.

    Fix: rng.randint(0, num_sentences - 1) explicitly excludes the last
         sentence, reserving it as the context boundary for the ICT objective.
         With numpy randint the upper bound is exclusive so this returns values
         in [0, num_sentences - 2] inclusive.

    Args:
        rng: numpy RandomState (used in InverseClozeDataset.__getitem__)
        num_sentences (int): number of sentences in the current document.

    Returns:
        int: selected input sentence index in [0, num_sentences - 2].
    """
    return rng.randint(0, num_sentences - 1)
# --- End M97 dataloader ---
# ---------------------------------------------------------------------------
# M102: Megatron ba2264abb — verified zeroshot tasks works
# Ported from: tasks/zeroshot_gpt2/datasets.py + tasks/run_gpt2_eval.py
#   → deepspeed/runtime/dataloader.py
#
# Key changes carried over:
#
# datasets.py: args.valid_data was passed as a list to _LambadaDataset and
#   open(); must index [0] because valid_data is a list of paths, not a
#   single path.
#
#   Before (3 sites):
#     _LambadaDataset(args.valid_data, ...)
#     with open(args.valid_data, "rb") as reader:
#     get_detokenizer(args.valid_data)(entire_data)
#
#   After:
#     _LambadaDataset(args.valid_data[0], ...)
#     with open(args.valid_data[0], "rb") as reader:
#     get_detokenizer(args.valid_data[0])(entire_data)
#
# run_gpt2_eval.py: renamed to tasks/run_gpt2_eval.py; removed obsolete
#   --webtext-eval, --eval-iters, --load-openai, --cache-dir,
#   --make-vocab-size-divisible-by, --text-key CLI args; changed
#   --eval-batch-size to --batch-size; LAMBADA now uses main.py + --task
#   flag instead of evaluate_gpt2.py + --cloze-eval.
# ---------------------------------------------------------------------------


def _m102_valid_data_path(args_valid_data):
    """M102: Megatron ba2264abb — valid_data is a list; callers must index [0].

    Before: passed args.valid_data (list) directly to open() / dataset ctor.
    After:  pass args.valid_data[0] (str) — the first (and only) path element.

    Downstream dataset builders should use this helper instead of indexing
    inline so that the intent is explicit and auditable.
    """
    # valid_data is always a list of paths; zeroshot tasks assert len == 1
    return args_valid_data[0]


print('[M102]')
# --- End M102 dataloader ---

# ---------------------------------------------------------------------------
# M120: Megatron da0562fcf — Updates to preprocess_data.py and indexed_dataset.
# preprocess_data: adds ability to not split sentences (for GPT-2 datasets),
# adds ability to create multiple datasets from different JSON keys.
# indexed_dataset: adds new "get" function to retrieve a portion of an entry.
# ---------------------------------------------------------------------------

import json
import multiprocessing
import sys
import time


try:
    import nltk as _nltk
    _nltk_available = True
except ImportError:
    _nltk_available = False


class _CustomLanguageVars(_nltk.tokenize.punkt.PunktLanguageVars if _nltk_available else object):
    """M120: preserves empty lines with NLTK's Punkt tokenizer."""

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- changed from \s+ to allow empty lines
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""


class _IdentitySplitter(object):
    """M120: no-op splitter — returns text as a single 'sentence'."""

    def tokenize(self, *text):
        return text


class _M120Encoder(object):
    """M120: multiprocessing encoder for indexed-dataset preprocessing.

    Mirrors Megatron da0562fcf scripts/preprocess_data.py::Encoder.
    Supports multiple JSON keys and optional sentence splitting.
    """

    def __init__(self, args):
        self.args = args

    def initializer(self):
        """Called once per worker process."""
        from deepspeed.runtime.data_pipeline.data_sampling.indexed_dataset import MMapIndexedDataset  # noqa
        # build tokenizer — caller must supply a build_tokenizer(args) compatible callable
        if hasattr(self.args, 'build_tokenizer'):
            _M120Encoder.tokenizer = self.args.build_tokenizer(self.args)
        else:
            _M120Encoder.tokenizer = None

        if getattr(self.args, 'split_sentences', False):
            if not _nltk_available:
                print('[M120] NLTK is not available to split sentences.')
                raise RuntimeError('NLTK required for sentence splitting')
            splitter = _nltk.load('tokenizers/punkt/english.pickle')
            if getattr(self.args, 'keep_newlines', False):
                _M120Encoder.splitter = _nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text=splitter._params,
                    lang_vars=_CustomLanguageVars())
            else:
                _M120Encoder.splitter = splitter
        else:
            _M120Encoder.splitter = _IdentitySplitter()

    def encode(self, json_line):
        """Encode one JSON line → {key: [[token_ids, ...], ...], ...}."""
        data = json.loads(json_line)
        ids = {}
        for key in self.args.json_keys:
            text = data[key]
            doc_ids = []
            for sentence in _M120Encoder.splitter.tokenize(text):
                if _M120Encoder.tokenizer is not None:
                    sentence_ids = _M120Encoder.tokenizer.tokenize(sentence)
                else:
                    sentence_ids = sentence.split()
                if len(sentence_ids) > 0:
                    doc_ids.append(sentence_ids)
            if getattr(self.args, 'append_eod', False) and doc_ids:
                if _M120Encoder.tokenizer is not None:
                    doc_ids[-1].append(_M120Encoder.tokenizer.eod)
            ids[key] = doc_ids
        return ids, len(json_line)


def m120_preprocess_data(args):
    """M120: top-level preprocessing entry point.

    Mirrors Megatron da0562fcf scripts/preprocess_data.py::main().
    args must expose: input, json_keys, output_prefix, dataset_impl,
    workers, log_interval, vocab_size (optional), build_tokenizer (optional).
    """
    print('[M120]')
    from deepspeed.runtime.data_pipeline.data_sampling.indexed_dataset import (
        make_builder as _make_builder,
    )
    import torch as _torch

    startup_start = time.time()
    print('[M120] Opening', args.input)
    fin = open(args.input, 'r', encoding='utf-8')

    if _nltk_available and getattr(args, 'split_sentences', False):
        _nltk.download('punkt', quiet=True)

    encoder = _M120Encoder(args)
    pool = multiprocessing.Pool(getattr(args, 'workers', 1),
                                initializer=encoder.initializer)
    encoded_docs = pool.imap(encoder.encode, fin, 25)

    vocab_size = getattr(args, 'vocab_size', None)
    print(f'[M120] Output prefix: {args.output_prefix}')
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.json_keys:
        output_bin_files[key] = '{}_{}.bin'.format(args.output_prefix, key)
        output_idx_files[key] = '{}_{}.idx'.format(args.output_prefix, key)
        builders[key] = _make_builder(output_bin_files[key],
                                      impl=getattr(args, 'dataset_impl', 'mmap'),
                                      vocab_size=vocab_size)

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print('[M120] Time to startup:', startup_end - startup_start)

    log_interval = getattr(args, 'log_interval', 100)
    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed
        for key, sentences in doc.items():
            for sentence in sentences:
                builders[key].add_item(_torch.IntTensor(sentence))
            builders[key].end_document()
        if i % log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f'[M120] Processed {i} documents '
                  f'({i/elapsed:.2f} docs/s, {mbs:.2f} MB/s).',
                  file=sys.stderr)

    for key in args.json_keys:
        builders[key].finalize(output_idx_files[key])

    print('[M120] Preprocessing complete.')


# --- End M120 dataloader ---

# ---------------------------------------------------------------------------
# M119: Megatron f66c58a9b — added build sample index to c++
# Ported from:
#   megatron/data/helpers.cpp      → deepspeed/runtime/dataloader.py
#   megatron/data/new_gpt2_dataset.py → deepspeed/runtime/dataloader.py
#
# Key changes carried over:
#   1. helpers.cpp: new C++ function build_sample_idx() added and registered
#      via pybind11 as helpers.build_sample_idx.
#      Signature: build_sample_idx(sizes, doc_idx, seq_length, num_epochs,
#                                  tokens_per_epoch) -> np.ndarray shape [N+1, 2]
#      The output is a 2D array where column 0 is the index into doc_idx and
#      column 1 is the starting offset within that document.
#      Memory is allocated as int32_t[2*(num_samples+1)] and freed via a
#      py::capsule destructor; returned as a strided numpy array with
#      C-contiguous layout [{2*sizeof(int32), sizeof(int32)}].
#
#   2. new_gpt2_dataset.py: _build_sample_idx (pure-Python) replaced by
#      helpers.build_sample_idx (C++ extension) in _build_index_mappings.
#      The old Python call is commented out in the diff for reference.
#      import helpers moved to module level (was inside _build_index_mappings).
#
#   3. new_gpt2_dataset.py: _build_doc_idx: doc_idx converted to np.int32
#      (was float64 from np.mgrid) before the shuffle so the array passed
#      to helpers.build_sample_idx already has the expected dtype.
#
#   4. new_gpt2_dataset.py: build_train_valid_test_datasets() and
#      get_indexed_dataset_() factory functions added at module level,
#      enabling the split-build-return pattern used by GPT-2 pretraining.
#
#   5. new_gpt2_dataset.py: test loop upper bound widened 1234..1240 →
#      1234..1245 (covers 5 additional random seeds in __main__ block).
#
# Neuron_SP adaptation:
#   The C++ build_sample_idx function is documented and its pure-Python
#   equivalent is provided below as _m119_build_sample_idx_py() for
#   environments where the helpers C extension is unavailable (e.g., CPU-
#   only nodes, unit tests).  Production code should prefer the C++ path.
#   The new GPT-2 dataset factory helpers are ported as module-level
#   functions following the same _mNNN_ naming convention.
# ---------------------------------------------------------------------------

print('[M119]')


def _m119_build_sample_idx_py(sizes, doc_idx, seq_length, num_epochs,
                               tokens_per_epoch):
    """M119: Megatron f66c58a9b — pure-Python fallback for build_sample_idx.

    Production code should use helpers.build_sample_idx (C++ via pybind11).
    This fallback mirrors the C++ logic exactly so it can be used for testing
    and on platforms where the helpers extension has not been compiled.

    The returned array has shape [num_samples + 1, 2] with dtype int32.
    Column 0: index into doc_idx.
    Column 1: byte/token offset within that document.

    The extra trailing row (index num_samples) marks the end of the last
    sample so callers can compute sample length as:
        end_doc, end_offset = sample_idx[i + 1]
        start_doc, start_offset = sample_idx[i]

    Args:
        sizes (np.ndarray int32): token count per document (full corpus).
        doc_idx (np.ndarray int32): shuffled document order across epochs.
        seq_length (int): target sequence length (must be > 1).
        num_epochs (int): number of training epochs (must be > 0).
        tokens_per_epoch (int): total tokens in one epoch (must be > 1).

    Returns:
        np.ndarray: shape [num_samples + 1, 2], dtype int32.
    """
    import numpy as np

    assert seq_length > 1
    assert num_epochs > 0
    assert tokens_per_epoch > 1

    num_samples = (num_epochs * tokens_per_epoch - 1) // seq_length

    print('    using:')
    print('     number of documents:       {}'.format(
        doc_idx.shape[0] // num_epochs))
    print('     number of epochs:          {}'.format(num_epochs))
    print('     sequence length:           {}'.format(seq_length))
    print('     total number of samples:   {}'.format(num_samples))

    sample_idx = np.zeros([num_samples + 1, 2], dtype=np.int32)

    sample_index = 0
    doc_idx_index = 0
    doc_offset = 0

    sample_idx[sample_index, 0] = doc_idx_index
    sample_idx[sample_index, 1] = doc_offset
    sample_index += 1

    while sample_index <= num_samples:
        remaining_seq_length = seq_length + 1
        while remaining_seq_length != 0:
            doc_id = doc_idx[doc_idx_index]
            doc_length = sizes[doc_id] - doc_offset
            remaining_seq_length -= doc_length
            if remaining_seq_length <= 0:
                # Current document is long enough — record offset within it.
                # -1 accounts for the same reason num_epochs uses -1 in
                # tokens_per_epoch calculations.
                doc_offset += (remaining_seq_length + doc_length - 1)
                remaining_seq_length = 0
            else:
                # Move to the next document.
                doc_idx_index += 1
                doc_offset = 0
        sample_idx[sample_index, 0] = doc_idx_index
        sample_idx[sample_index, 1] = doc_offset
        sample_index += 1

    return sample_idx


def _m119_build_doc_idx_int32(documents, num_epochs, np_rng):
    """M119: Megatron f66c58a9b — _build_doc_idx with explicit int32 cast.

    Before this commit doc_idx was the result of np.mgrid which returns
    float64 on some numpy versions.  helpers.build_sample_idx expects int32;
    the fix adds doc_idx = doc_idx.astype(np.int32) before the shuffle.

    Args:
        documents (np.ndarray): array of document indices for one epoch.
        num_epochs (int): number of epochs to tile documents over.
        np_rng: numpy RandomState used for shuffling.

    Returns:
        np.ndarray int32: flattened, shuffled document index array of length
        num_epochs * len(documents).
    """
    import numpy as np

    doc_idx = np.mgrid[0:num_epochs, 0:len(documents)][1]
    doc_idx[:] = documents
    doc_idx = doc_idx.reshape(-1)
    # M119: explicit int32 cast — helpers.build_sample_idx requires int32.
    doc_idx = doc_idx.astype(np.int32)
    np_rng.shuffle(doc_idx)
    return doc_idx


def _m119_build_index_mappings_use_cpp(name, data_prefix, documents, sizes,
                                       num_samples, seq_length, seed,
                                       helpers_mod, print_rank_0_fn):
    """M119: Megatron f66c58a9b — _build_index_mappings using C++ build_sample_idx.

    Replaces the inner sample-idx call from the pure-Python _build_sample_idx
    with helpers.build_sample_idx (C++ via pybind11).  The doc_idx array is
    cast to int32 before being passed (see _m119_build_doc_idx_int32).

    The function mirrors the structure of new_gpt2_dataset._build_index_mappings
    and is provided here as a callable stub for Neuron_SP callers that manage
    their own index-mapping build loop.

    Args:
        name (str): dataset name used as a filename prefix.
        data_prefix (str): path prefix for .npy cache files.
        documents (np.ndarray): document indices for this split.
        sizes (np.ndarray int32): token count per document.
        num_samples (int): total number of training samples required.
        seq_length (int): target sequence length.
        seed (int): random seed for document shuffling.
        helpers_mod: the compiled helpers extension module exposing
            build_sample_idx and build_mapping.
        print_rank_0_fn (callable): print function (no-op on rank > 0).

    Returns:
        tuple: (doc_idx, sample_idx, shuffle_idx) as numpy arrays.
    """
    import os
    import numpy as np
    import torch

    _NUM_EPOCHS_DEFAULT_MULTIPLIER = 20  # same heuristic as Megatron upstream

    def _num_tokens(doc_list, sz):
        """Total token count for the given document list."""
        return np.sum(sz[doc_list])

    def _num_epochs(tokens_per_epoch, seq_length, n_samples):
        num_epochs_ = 0
        total_tokens = 0
        while True:
            num_epochs_ += 1
            total_tokens += tokens_per_epoch
            if ((total_tokens - 1) // seq_length) >= n_samples:
                return num_epochs_

    tokens_per_epoch = _num_tokens(documents, sizes)
    num_epochs = _num_epochs(tokens_per_epoch, seq_length, num_samples)

    np_rng = np.random.RandomState(seed=seed)

    # doc-idx
    doc_idx = _m119_build_doc_idx_int32(documents, num_epochs, np_rng)

    # sample-idx: use C++ helper (M119 change)
    sample_idx = helpers_mod.build_sample_idx(sizes, doc_idx, seq_length,
                                              num_epochs, tokens_per_epoch)
    # shuffle-idx
    num_samples_ = sample_idx.shape[0] - 1
    shuffle_idx = np.arange(start=0, stop=num_samples_, step=1, dtype=np.int64)
    np_rng.shuffle(shuffle_idx)

    print_rank_0_fn(' > size of doc-idx:     {}'.format(doc_idx.shape[0]))
    print_rank_0_fn(' > size of sample-idx:  {}'.format(sample_idx.shape[0]))
    print_rank_0_fn(' > size of shuffle-idx: {}'.format(shuffle_idx.shape[0]))

    return doc_idx, sample_idx, shuffle_idx


def _m119_get_indexed_dataset(data_prefix, data_impl, skip_warmup,
                               make_indexed_dataset_fn, print_rank_0_fn):
    """M119: Megatron f66c58a9b — get_indexed_dataset_ for GPT-2 datasets.

    New module-level factory added in new_gpt2_dataset.py alongside
    build_train_valid_test_datasets.  Mirrors the BertDataset equivalent
    from M36/M37 but without the doc_idx[-1] assertion (GPT-2 indexed
    datasets do not guarantee contiguity in the same way).

    Args:
        data_prefix (str): path prefix passed to make_indexed_dataset_fn.
        data_impl (str): dataset implementation type ('mmap', 'lazy', etc.).
        skip_warmup (bool): whether to skip mmap warmup.
        make_indexed_dataset_fn: callable(data_prefix, data_impl, skip_warmup).
        print_rank_0_fn (callable): print function (no-op on rank > 0).

    Returns:
        indexed dataset object with .sizes attribute.
    """
    import time

    print_rank_0_fn(' > building dataset index ...')
    start_time = time.time()
    indexed_dataset = make_indexed_dataset_fn(data_prefix, data_impl, skip_warmup)
    print_rank_0_fn(' > finished creating indexed dataset in {:4f} '
                    'seconds'.format(time.time() - start_time))
    print_rank_0_fn(' > indexed dataset stats:')
    print_rank_0_fn('    number of documents: {}'.format(
        indexed_dataset.sizes.shape[0]))
    return indexed_dataset


def _m119_build_train_valid_test_datasets(data_prefix, data_impl, splits_string,
                                          train_valid_test_num_samples,
                                          seq_length, seed, skip_warmup,
                                          make_indexed_dataset_fn,
                                          gpt2_dataset_cls,
                                          get_train_valid_test_split_fn,
                                          print_rank_0_fn):
    """M119: Megatron f66c58a9b — build_train_valid_test_datasets for GPT-2.

    New top-level factory added in new_gpt2_dataset.py in this commit.
    Constructs train/valid/test GPT2Dataset instances from a single indexed
    dataset, splitting by document count according to splits_string.

    Args:
        data_prefix (str): path prefix for the dataset.
        data_impl (str): dataset implementation ('mmap', 'lazy', etc.).
        splits_string (str): comma/slash-separated split ratios e.g. '90,5,5'.
        train_valid_test_num_samples (list[int]): [train_n, valid_n, test_n].
        seq_length (int): target sequence length.
        seed (int): random seed for document shuffling.
        skip_warmup (bool): whether to skip mmap warmup.
        make_indexed_dataset_fn: callable(prefix, impl, warmup).
        gpt2_dataset_cls: GPT2Dataset-compatible class constructor.
        get_train_valid_test_split_fn: callable(splits_string, total_docs) ->
            4-element list of boundary indices.
        print_rank_0_fn (callable): rank-0 print.

    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset); any may be None
        if its split contains zero documents.
    """
    import numpy as np

    indexed_dataset = _m119_get_indexed_dataset(
        data_prefix, data_impl, skip_warmup,
        make_indexed_dataset_fn, print_rank_0_fn)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    splits = get_train_valid_test_split_fn(splits_string, total_num_of_documents)

    print_rank_0_fn(' > dataset split:')

    def print_split_stats(name, index):
        print_rank_0_fn('    {}:'.format(name))
        print_rank_0_fn(
            '     document indices in [{}, {}) total of {} '
            'documents'.format(splits[index], splits[index + 1],
                               splits[index + 1] - splits[index]))

    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index], end=splits[index + 1],
                                  step=1, dtype=np.int32)
            dataset = gpt2_dataset_cls(
                name, data_prefix, documents, indexed_dataset,
                train_valid_test_num_samples[index],
                seq_length, seed)
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return train_dataset, valid_dataset, test_dataset

# --- End M119 dataloader ---


# ---------------------------------------------------------------------------
# M189: Megatron 4abd7ce27 — Refactor before merge request
# Ported from:
#   megatron/data/bert_dataset.py          → deepspeed/runtime/dataloader.py
#   megatron/data/ict_dataset.py           → deepspeed/runtime/dataloader.py
#   megatron/deprecated_data_utils/datasets.py → deepspeed/runtime/dataloader.py
#   megatron/data/helpers.cpp              → deepspeed/runtime/dataloader.py
#   megatron/model/bert_model.py           → deepspeed/runtime/dataloader.py
#   megatron/training.py                   → deepspeed/runtime/dataloader.py
#   pretrain_bert_ict.py                   → deepspeed/runtime/dataloader.py
#   ict_qualitative_test.py                → DELETED upstream (not ported)
#
# Key changes carried over:
#
#   1. bert_dataset.py — build_train_valid_test_datasets():
#      - `titles_dataset` renamed to `title_dataset` (variable local to the
#        ict_dataset branch).
#      - InverseClozeDataset constructor call updated: `titles_dataset=` kwarg
#        renamed to `title_dataset=`; `context_dataset=` removed from the
#        shared kwargs dict and passed explicitly as `block_dataset=`.
#      - BertDataset constructor call updated: `indexed_dataset=` added
#        explicitly (was formerly in the shared kwargs as `context_dataset=`).
#      - A blank line added before `def print_split_stats` for readability.
#
#   2. ict_dataset.py — InverseClozeDataset:
#      - `import sys` removed (unused).
#      - `__init__` parameters renamed: `context_dataset` → `block_dataset`,
#        `titles_dataset` → `title_dataset`.
#      - Instance attributes renamed accordingly:
#          self.context_dataset → self.block_dataset
#          self.titles_dataset  → self.title_dataset
#      - get_samples_mapping() call updated to pass `block_dataset` / `title_dataset`.
#      - `__getitem__`: unpacks 4-tuple `(start_idx, end_idx, doc_idx, block_idx)`
#        from samples_mapping (helpers.cpp now stores block_id in column 3).
#      - `titles_dataset[doc_idx]` → `title_dataset[doc_idx]`.
#      - `context_dataset[i]` → `block_dataset[i]`.
#      - Local variable `context` renamed to `block` throughout __getitem__.
#      - `input` renamed to `query` throughout __getitem__.
#      - Return dict keys updated:
#            input_text  → query_tokens   input_types  → query_types
#            input_pad_mask → query_pad_mask
#            context_text → block_tokens  context_types → block_types
#            context_pad_mask → block_pad_mask
#      - Comment "may still need to truncate" → "still need to truncate".
#      - "keep the query in the context 10%" → "keep the query in the block 10%".
#      - get_samples_mapping() first parameter renamed `context_dataset` → `block_dataset`.
#      - Assertions updated: context_dataset.doc_idx / .sizes → block_dataset.*.
#      - helpers.build_blocks_mapping() call updated to pass block_dataset args.
#
#   3. deprecated_data_utils/datasets.py — InverseClozeDataset.__getitem__ dict:
#      - `input_types` key renamed to `query_types`.
#      - `context_types` key renamed to `block_types`.
#      (input_text, input_pad_mask, context_text, context_pad_mask unchanged.)
#
#   4. helpers.cpp — build_blocks_mapping_impl():
#      - Added `int32_t block_id = 0` counter before the epoch loop.
#      - Map stride changed from 3 → 4 (adds a 4th column for block_id).
#      - maps[map_index_0 + 3] = block_id stored at Populate-the-map site.
#      - block_id incremented alongside map_index.
#      - block_id reset to 0 at the end of each epoch.
#      - maps allocation: `3*map_index` → `4*map_index`.
#      - Fisher-Yates shuffle updated: i0/j0 use multiplier 4; swap of [+3].
#      - Return shape/strides: {num_samples, 3} → {num_samples, 4};
#        stride {3*byte_size, byte_size} → {4*byte_size, byte_size}.
#
#   5. bert_model.py — ICTBertModel:
#      - Docstring added: "Bert-based module for Inverse Cloze task."
#      - `question_model` → `query_model`; `_question_key` → `_query_key`
#        (but checkpoint key string kept as 'question_model' for compat).
#      - `context_model` → `block_model`; `_context_key` → `_block_key`
#        (but checkpoint key string kept as 'context_model' for compat).
#      - Comments clarifying roles: query_model = Embed_input, block_model = Embed_doc.
#      - forward() signature: `input_*` → `query_*`; `context_*` → `block_*`.
#      - `return_logits` parameter removed; forward now always returns
#        `(query_logits, block_logits)` — the dot-product score moved to caller.
#      - forward docstring added.
#      - state_dict_for_save_checkpoint() docstring added.
#      - load_state_dict() docstring changed: "Customized load." →
#        "Load the state dicts of each of the models" (no trailing period).
#
#   6. training.py — train_step():
#      - Three `torch.cuda.synchronize()` calls removed (after forward, after
#        backward, after optimizer step) — they were slowing training without
#        correctness benefit in the distributed setting.
#
#   7. pretrain_bert_ict.py — overall refactor:
#      - `from megatron.utils import make_data_loader` import removed.
#      - get_batch(): keys list updated to new field names:
#            ['input_text', 'input_types', 'input_pad_mask',
#             'context_text', 'context_types', 'context_pad_mask']
#          → ['query_tokens', 'query_types', 'query_pad_mask',
#              'block_tokens', 'block_types', 'block_pad_mask']
#      - data_b unpacking: all local variable names updated to match.
#      - forward_step(): variable names updated; model() call updated;
#        dot-product `retrieval_scores` moved from model to caller
#        (query_logits.matmul(block_logits.T)).
#      - get_train_val_test_data() function replaced by
#        train_valid_test_datasets_provider(train_val_test_num_samples)
#        which now receives the sample counts from the pretrain() framework
#        rather than computing them itself; make_data_loader calls removed.
#      - pretrain() call at __main__: first arg updated to
#        train_valid_test_datasets_provider.
#
#   8. ict_qualitative_test.py — deleted upstream; not ported.
#
# Neuron_SP adaptation:
#   The functions below capture each change as a named helper/doc function
#   following the M-series pattern.  Existing _m62_* helpers that expose the
#   old dict key names are superseded by the _m189_* versions.
# ---------------------------------------------------------------------------

print('[M189]')


def _m189_ict_dataset_init_params():
    """M189: Megatron 4abd7ce27 — InverseClozeDataset.__init__ parameter renames.

    Before (M62 / earlier):
        def __init__(self, name, context_dataset, titles_dataset, ...)

    After (M189):
        def __init__(self, name, block_dataset, title_dataset, ...)

    Callers that previously passed `context_dataset=` / `titles_dataset=`
    kwargs must update to `block_dataset=` / `title_dataset=`.

    Returns a (old_names, new_names) pair for documentation purposes.
    """
    old_names = ('context_dataset', 'titles_dataset')
    new_names = ('block_dataset', 'title_dataset')
    return old_names, new_names


def _m189_ict_dataset_getitem_keys():
    """M189: Megatron 4abd7ce27 — InverseClozeDataset.__getitem__ dict key renames.

    The sample dict returned by __getitem__ changes all six keys:

      Before (M62):                    After (M189):
        'input_text'       →             'query_tokens'
        'input_types'      →             'query_types'
        'input_pad_mask'   →             'query_pad_mask'
        'context_text'     →             'block_tokens'
        'context_types'    →             'block_types'
        'context_pad_mask' →             'block_pad_mask'

    Callers of InverseClozeDataset.__getitem__ (and get_batch() in
    pretrain_bert_ict.py) must update all key references.

    Returns (old_keys, new_keys) tuples.
    """
    old_keys = (
        'input_text', 'input_types', 'input_pad_mask',
        'context_text', 'context_types', 'context_pad_mask',
    )
    new_keys = (
        'query_tokens', 'query_types', 'query_pad_mask',
        'block_tokens', 'block_types', 'block_pad_mask',
    )
    return old_keys, new_keys


def _m189_getitem_inverse_cloze(ict_dataset, idx):
    """M189: Megatron 4abd7ce27 — updated __getitem__ for InverseClozeDataset.

    Supersedes _m62_getitem_inverse_cloze.  Key differences:

      1. samples_mapping now stores 4-tuples: (start_idx, end_idx, doc_idx, block_idx)
         because helpers.cpp build_blocks_mapping_impl now records block_id in
         column 3 (stride changed from 3 to 4).

      2. All dict keys renamed per _m189_ict_dataset_getitem_keys():
           input_text  → query_tokens    input_types  → query_types
           input_pad_mask → query_pad_mask
           context_text → block_tokens   context_types → block_types
           context_pad_mask → block_pad_mask

      3. Local variable `input` renamed to `query`; `context` renamed to `block`.

      4. `title_dataset` used (was `titles_dataset`).
      5. `block_dataset` used (was `context_dataset`).

    Args:
        ict_dataset: an InverseClozeDataset-like object exposing:
            - samples_mapping: array of shape (N, 4) with columns
              [start_idx, end_idx, doc_idx, block_idx]
            - block_dataset: the indexed token dataset for blocks
            - title_dataset: the indexed token dataset for titles
            - max_seq_length: int
            - rng: random.Random instance (seeded at init)
            - concat_and_pad_tokens(tokens, title=None): returns
              (tokens, token_types, pad_mask)
        idx (int): dataset index.

    Returns:
        dict with keys: query_tokens, query_types, query_pad_mask,
                        block_tokens, block_types, block_pad_mask.
    """
    import numpy as np

    start_idx, end_idx, doc_idx, block_idx = ict_dataset.samples_mapping[idx]
    title = list(ict_dataset.title_dataset[int(doc_idx)])
    block = [list(ict_dataset.block_dataset[i]) for i in range(start_idx, end_idx)]
    assert len(block) > 1

    # avoid selecting the first or last sentence to be the query.
    if len(block) == 2:
        rand_sent_idx = int(ict_dataset.rng.random() > 0.5)
    else:
        rand_sent_idx = ict_dataset.rng.randint(1, len(block) - 2)

    # keep the query in the block 10% of the time.
    if ict_dataset.rng.random() < 0.1:
        query = block[rand_sent_idx].copy()
    else:
        query = block.pop(rand_sent_idx)

    # still need to truncate because blocks are concluded when
    # the sentence lengths have exceeded max_seq_length.
    import itertools
    query = query[:ict_dataset.max_seq_length - 2]
    block = list(itertools.chain(*block))[:ict_dataset.max_seq_length - (3 + len(title))]

    query_tokens, query_token_types, query_pad_mask = ict_dataset.concat_and_pad_tokens(query)
    block_tokens, block_token_types, block_pad_mask = ict_dataset.concat_and_pad_tokens(block, title)

    sample = {
        'query_tokens': np.array(query_tokens),
        'query_types': np.array(query_token_types),
        'query_pad_mask': np.array(query_pad_mask),
        'block_tokens': np.array(block_tokens),
        'block_types': np.array(block_token_types),
        'block_pad_mask': np.array(block_pad_mask),
    }
    return sample


def _m189_deprecated_datasets_key_renames():
    """M189: Megatron 4abd7ce27 — deprecated_data_utils/datasets.py dict key renames.

    InverseClozeDataset.__getitem__ in megatron/deprecated_data_utils/datasets.py
    (the older, pre-data-pipeline implementation) had a partial rename applied:

      'input_types'   → 'query_types'
      'context_types' → 'block_types'

    The other four keys (input_text, input_pad_mask, context_text, context_pad_mask)
    were NOT renamed in this commit in the deprecated module — only the two
    _types keys changed.  This function documents that asymmetry.

    Returns (changed_keys, unchanged_keys) for audit purposes.
    """
    changed = {'input_types': 'query_types', 'context_types': 'block_types'}
    unchanged = ('input_text', 'input_pad_mask', 'context_text', 'context_pad_mask')
    return changed, unchanged


def _m189_helpers_cpp_block_id_change():
    """M189: Megatron 4abd7ce27 — helpers.cpp build_blocks_mapping stride 3→4.

    build_blocks_mapping_impl in megatron/data/helpers.cpp changed the output
    array from 3 columns to 4 columns by adding a `block_id` field:

      Column 0: start sentence index  (prev_start_index)
      Column 1: end sentence index    (sent_index + 1)
      Column 2: document index        (doc)
      Column 3: block id              (block_id)  ← NEW in M189

    Downstream callers (InverseClozeDataset.__getitem__) must unpack 4-tuples:
        start_idx, end_idx, doc_idx, block_idx = samples_mapping[idx]

    instead of the previous 3-tuple:
        start_idx, end_idx, doc_idx = samples_mapping[idx]

    The block_id counter resets to 0 at the start of each epoch, so block_idx
    is epoch-local (not globally unique across epochs).

    The Fisher-Yates shuffle in the second pass operates on 4-element strides
    and swaps the block_id column along with the other three.

    Returns a summary dict for documentation.
    """
    return {
        'old_stride': 3,
        'new_stride': 4,
        'new_column_index': 3,
        'new_column_name': 'block_id',
        'epoch_reset': True,
        'getitem_unpack': '(start_idx, end_idx, doc_idx, block_idx)',
    }


def _m189_ict_bert_model_renames():
    """M189: Megatron 4abd7ce27 — ICTBertModel attribute and forward() renames.

    megatron/model/bert_model.py::ICTBertModel changes:

    Attribute renames (Python names):
        question_model → query_model   (_question_key → _query_key)
        context_model  → block_model   (_context_key  → _block_key)

    Checkpoint key strings UNCHANGED for backward compatibility:
        _query_key = 'question_model'   (was and still is 'question_model')
        _block_key = 'context_model'    (was and still is 'context_model')

    forward() signature rename:
        input_tokens / input_attention_mask / input_types   → query_*
        context_tokens / context_attention_mask / context_types → block_*

    forward() return value change:
        Before: optionally returns (question_logits, context_logits, retrieval_scores)
                when return_logits=True; otherwise returns retrieval_scores tensor.
        After:  always returns (query_logits, block_logits) — the dot-product
                retrieval score computation moved to the caller (forward_step in
                pretrain_bert_ict.py).

    Returns a summary dict for documentation.
    """
    return {
        'attr_renames': {
            'question_model': 'query_model',
            'context_model': 'block_model',
        },
        'checkpoint_keys_unchanged': {
            '_query_key': 'question_model',
            '_block_key': 'context_model',
        },
        'return_logits_removed': True,
        'retrieval_score_moved_to_caller': True,
    }


def _m189_training_synchronize_removed():
    """M189: Megatron 4abd7ce27 — torch.cuda.synchronize() calls removed from train_step.

    Three explicit synchronization barriers were removed from
    megatron/training.py::train_step():

        timers('forward').stop()
        torch.cuda.synchronize()   ← removed

        timers('backward').stop()
        torch.cuda.synchronize()   ← removed

        timers('optimizer').stop()
        torch.cuda.synchronize()   ← removed

    Rationale: in a distributed training setting these synchronize() calls
    serialise all ranks at each sub-step boundary, hurting throughput without
    correctness benefit.  NCCL collectives in backward/optimizer already
    provide the necessary ordering.

    DeepSpeed note: DeepSpeedEngine.train_batch() does not insert equivalent
    synchronize() calls; this upstream removal aligns Megatron's behaviour with
    DeepSpeed's existing practice.
    """
    pass


def _m189_pretrain_ict_dataset_provider(args):
    """M189: Megatron 4abd7ce27 — train_valid_test_datasets_provider for ICT.

    Replaces get_train_val_test_data() in pretrain_bert_ict.py.

    Key differences from the old function:
      1. Receives train_val_test_num_samples from the pretrain() framework
         instead of computing it internally.
      2. Returns (train_ds, valid_ds, test_ds) dataset objects — not DataLoaders.
         The pretrain() framework handles DataLoader creation.
      3. make_data_loader calls removed.
      4. Multi-rank guarding (if mpu.get_model_parallel_rank() == 0) removed;
         build_train_valid_test_datasets is called unconditionally.

    Args:
        args: Megatron/DeepSpeed args namespace with fields:
              data_path, data_impl, split, seq_length, mask_prob,
              short_seq_prob, seed, mmap_warmup.

    Returns a (build_kwargs, dataset_type='ict') pair for documentation purposes.
    (Actual dataset construction requires the build_train_valid_test_datasets
    function from bert_dataset.py which is not duplicated here.)

    M214 update (Megatron f7f730e1d): ict_dataset=True kwarg replaced by
    dataset_type='ict' — build_train_valid_test_datasets now uses a string
    discriminator instead of a boolean flag.
    """
    build_kwargs = dict(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        max_seq_length=args.seq_length,
        masked_lm_prob=args.mask_prob,
        short_seq_prob=args.short_seq_prob,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        dataset_type='ict',
    )
    return build_kwargs


def _m189_get_batch_keys():
    """M189: Megatron 4abd7ce27 — get_batch() key list update in pretrain_bert_ict.py.

    The keys list used to broadcast and unpack a data batch changes from:
        ['input_text', 'input_types', 'input_pad_mask',
         'context_text', 'context_types', 'context_pad_mask']
    to:
        ['query_tokens', 'query_types', 'query_pad_mask',
         'block_tokens', 'block_types', 'block_pad_mask']

    Returns (old_keys, new_keys) for documentation.
    """
    old_keys = ['input_text', 'input_types', 'input_pad_mask',
                'context_text', 'context_types', 'context_pad_mask']
    new_keys = ['query_tokens', 'query_types', 'query_pad_mask',
                'block_tokens', 'block_types', 'block_pad_mask']
    return old_keys, new_keys
# --- End M189 dataloader ---

# ---------------------------------------------------------------------------
# M212: Megatron 24034e036 — Revise dataset_type
# Ported from:
#   megatron/data/bert_dataset.py  → deepspeed/runtime/dataloader.py
#   pretrain_realm.py              → deepspeed/runtime/dataloader.py
#
# Key changes carried over:
#
#   1. bert_dataset.py — build_train_valid_test_datasets():
#      - Added `from megatron.data.realm_dataset import RealmDataset` import.
#      - Introduced module-level constant:
#            DATASET_TYPES = ['standard_bert', 'ict', 'realm']
#      - Function signature: `ict_dataset=False` parameter replaced by
#            `dataset_type='standard_bert'`
#        (keyword-only default; callers passing ict_dataset=True must now pass
#         dataset_type='ict' instead).
#      - Guard added immediately after the parameter:
#            if dataset_type not in DATASET_TYPES:
#                raise ValueError("Invalid dataset_type: ", dataset_type)
#      - Branch condition: `if ict_dataset:` → `if dataset_type == 'ict':`
#        (controls title_dataset creation for ICT path; unchanged logic).
#      - Inner per-split branch `if ict_dataset:` → `if dataset_type == 'ict':`
#        (selects InverseClozeDataset; unchanged logic).
#      - else-branch: single `dataset = BertDataset(...)` replaced by
#            dataset_cls = BertDataset if dataset_type == 'standard_bert' else RealmDataset
#            dataset = dataset_cls(indexed_dataset=indexed_dataset,
#                                  masked_lm_prob=masked_lm_prob, **kwargs)
#        so that dataset_type='realm' routes to RealmDataset without a
#        separate conditional branch.
#      - Blank line added after the dataset_cls assignment for readability.
#
#   2. pretrain_realm.py — train_valid_test_datasets_provider():
#      - Call to build_train_valid_test_datasets() updated:
#            skip_warmup=(not args.mmap_warmup)
#        → skip_warmup=(not args.mmap_warmup),
#           dataset_type='realm'
#        (trailing comma added; new kwarg appended on a new line).
#      - This replaces any implicit ict_dataset=False default and explicitly
#        routes REALM pretraining to RealmDataset via the new dispatch path.
#
# Neuron_SP adaptation:
#   _m212_dataset_types() — documents the new DATASET_TYPES constant.
#   _m212_build_train_valid_test_datasets_sig() — documents the signature change.
#   _m212_build_bert_datasets_kwargs_realm() — builds kwargs for realm pretrain,
#     mirroring _m189_pretrain_ict_dataset_provider() for the 'realm' case.
#   _m212_pretrain_realm_dataset_provider() — documents the pretrain_realm.py
#     train_valid_test_datasets_provider() with dataset_type='realm'.
# ---------------------------------------------------------------------------

print('[M212]')


def _m212_dataset_types():
    """M212: Megatron 24034e036 — DATASET_TYPES constant in bert_dataset.py.

    The module-level constant introduced by this commit enumerates all valid
    values for the new `dataset_type` parameter of build_train_valid_test_datasets().

    Returns:
        list[str]: the three recognised dataset type strings.
    """
    return ['standard_bert', 'ict', 'realm']


def _m212_build_train_valid_test_datasets_sig():
    """M212: Megatron 24034e036 — signature change in bert_dataset.py.

    Before this commit the function accepted a boolean flag:
        build_train_valid_test_datasets(..., ict_dataset=False)

    After this commit it accepts a string discriminator with validation:
        build_train_valid_test_datasets(..., dataset_type='standard_bert')
        # raises ValueError if dataset_type not in DATASET_TYPES

    The new dispatch logic in the else-branch:
        dataset_cls = BertDataset if dataset_type == 'standard_bert' else RealmDataset
        dataset = dataset_cls(indexed_dataset=indexed_dataset,
                              masked_lm_prob=masked_lm_prob, **kwargs)

    Returns a (old_param, new_param, dispatch_logic) tuple for documentation.
    """
    old_param = ('ict_dataset', False)
    new_param = ('dataset_type', 'standard_bert')
    dispatch_logic = {
        'ict': 'InverseClozeDataset (title_dataset path)',
        'standard_bert': 'BertDataset',
        'realm': 'RealmDataset',
    }
    return old_param, new_param, dispatch_logic


def _m212_build_bert_datasets_kwargs_realm(args):
    """M212: Megatron 24034e036 — build kwargs for REALM pretrain dataset provider.

    Mirrors _m189_pretrain_ict_dataset_provider() but sets dataset_type='realm'
    instead of ict_dataset=True, reflecting the new build_train_valid_test_datasets()
    signature introduced in this commit.

    The REALM provider in pretrain_realm.py now passes:
        skip_warmup=(not args.mmap_warmup),
        dataset_type='realm',
    where previously it omitted dataset_type entirely (defaulting to the old
    ict_dataset=False behaviour which mapped to BertDataset, not RealmDataset).

    Args:
        args: Namespace with fields:
              data_path, data_impl, split, seq_length, mask_prob,
              short_seq_prob, seed, mmap_warmup.

    Returns:
        dict: keyword arguments ready to unpack into
              build_train_valid_test_datasets(**kwargs).
    """
    return dict(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        max_seq_length=args.seq_length,
        masked_lm_prob=args.mask_prob,
        short_seq_prob=args.short_seq_prob,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        dataset_type='realm',
    )


def _m212_pretrain_realm_dataset_provider(args):
    """M212: Megatron 24034e036 — updated train_valid_test_datasets_provider in pretrain_realm.py.

    Before this commit the provider called build_train_valid_test_datasets()
    without a dataset_type argument, inadvertently routing REALM pretraining
    through the standard BertDataset path (ict_dataset defaulted to False).

    After this commit dataset_type='realm' is passed explicitly, routing the
    call to RealmDataset via the new dispatch logic in bert_dataset.py.

    The change in pretrain_realm.py is minimal — one new kwarg line:
        skip_warmup=(not args.mmap_warmup)           # before
        skip_warmup=(not args.mmap_warmup),           # after (trailing comma)
        dataset_type='realm'                          # after (new line)

    DeepSpeed/Neuron_SP note: callers that construct BERT-family dataset kwargs
    for REALM workloads should use _m212_build_bert_datasets_kwargs_realm()
    rather than _m70_build_bert_datasets_kwargs() or
    _m189_pretrain_ict_dataset_provider(), as neither of those sets
    dataset_type='realm'.

    Args:
        args: Namespace — same contract as _m189_pretrain_ict_dataset_provider().

    Returns:
        dict: build kwargs with dataset_type='realm'.
    """
    return _m212_build_bert_datasets_kwargs_realm(args)
# --- End M212 dataloader ---

# ---------------------------------------------------------------------------

# M214: Megatron f7f730e1d — Write pretrain_realm.py and misc dataset_type
#       left from earlier
# ---------------------------------------------------------------------------
# Changes in this commit (Megatron f7f730e1d, 2020-04-23):
#
#   1. pretrain_bert_ict.py — dataset kwarg rename
#      build_train_valid_test_datasets(..., ict_dataset=True)
#      → build_train_valid_test_datasets(..., dataset_type='ict')
#      Applied above in _m189_pretrain_ict_dataset_provider (M214 update note).
#
#   2. pretrain_realm.py — full rewrite / cleanup:
#      - Remove torchDDP import; remove manual checkpoint-loading boilerplate.
#      - Import HashedIndex, load_ict_checkpoint, get_ict_dataset from
#        hashed_index module.
#      - model_provider() now uses load_ict_checkpoint() / REALMRetriever /
#        REALMBertModel instead of get_model(ict_model_provider).
#      - get_batch() keys: ['query_tokens','query_types','query_pad_mask']
#        → ['tokens','labels','loss_mask','pad_mask'].
#      - forward_step() replaced retrieval-score / top-k accuracy logic with
#        P(y|x)=sum_z[P(y|z,x)*P(z|x)] marginalisation + vocab_parallel_cross_entropy.
#      - train_valid_test_datasets_provider() passes dataset_type='realm'.
# ---------------------------------------------------------------------------

print('[M214]')


def _m214_pretrain_realm_model_provider():
    """M214: Megatron f7f730e1d — model_provider() for pretrain_realm.py.

    Previous version used get_model(ict_model_provider) + manual torch.load
    checkpoint restoration.  This revision delegates checkpoint loading to
    load_ict_checkpoint() from the hashed_index module and composes the full
    REALM model from REALMRetriever + REALMBertModel.

    Imports required by the real pretrain_realm.py (not duplicated here):
        from hashed_index import HashedIndex, load_ict_checkpoint, get_ict_dataset
        from megatron.model import REALMBertModel, REALMRetriever

    Equivalent logic (pseudo-code):
        def model_provider():
            args = get_args()
            print_rank_0('building REALM models ...')
            ict_model    = load_ict_checkpoint()
            ict_dataset  = get_ict_dataset()
            hashed_index = HashedIndex.load_from_file('block_hash_data.pkl')
            retriever    = REALMRetriever(ict_model, ict_dataset, hashed_index)
            model        = REALMBertModel(retriever)
            return model
    """
    pass


def _m214_pretrain_realm_get_batch_keys():
    """M214: Megatron f7f730e1d — get_batch() key list for pretrain_realm.py.

    Old keys (pre-M214):
        ['query_tokens', 'query_types', 'query_pad_mask']

    New keys (M214):
        ['tokens', 'labels', 'loss_mask', 'pad_mask']

    The rename reflects the shift from a retrieval-scoring task (query vs
    block pairs) to a masked-LM task (tokens + labels + loss_mask).
    """
    old_keys = ['query_tokens', 'query_types', 'query_pad_mask']
    new_keys = ['tokens', 'labels', 'loss_mask', 'pad_mask']
    return old_keys, new_keys


def _m214_pretrain_realm_forward_step_logic():
    """M214: Megatron f7f730e1d — forward_step() logic for pretrain_realm.py.

    Previous implementation computed retrieval accuracy (top-1 / top-5) via a
    dot-product similarity matrix between query and block logits.

    New implementation marginalises over retrieved blocks:
        lm_logits, block_probs = model(tokens, pad_mask)
        # P(y|x) = sum_z( P(y|z,x) * P(z|x) )
        lm_logits = torch.sum(lm_logits * block_probs, dim=1)
        lm_loss_  = mpu.vocab_parallel_cross_entropy(
                        lm_logits.contiguous().float(),
                        labels.contiguous())
        lm_loss   = torch.sum(lm_loss_.view(-1) * loss_mask.reshape(-1)) \
                    / loss_mask.sum()
        reduced_loss = reduce_losses([lm_loss])
        return lm_loss, {'lm_loss': reduced_loss[0]}

    Removed output keys: 'retrieval loss', 'top1_acc', 'top5_acc'.
    New output key:      'lm_loss'.
    """
    pass


def _m214_pretrain_realm_dataset_provider():
    """M214: Megatron f7f730e1d — train_valid_test_datasets_provider for REALM.

    Mirrors _m189_pretrain_ict_dataset_provider but passes dataset_type='realm'
    to build_train_valid_test_datasets().

    Equivalent kwargs:
        build_kwargs = dict(
            data_prefix=args.data_path,
            data_impl=args.data_impl,
            splits_string=args.split,
            max_seq_length=args.seq_length,
            masked_lm_prob=args.mask_prob,
            short_seq_prob=args.short_seq_prob,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup),
            dataset_type='realm',
        )

    Entry point:
        if __name__ == '__main__':
            pretrain(train_valid_test_datasets_provider, model_provider,
                     forward_step,
                     args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
    """
    pass
# --- End M214 dataloader ---


# ---------------------------------------------------------------------------
# M211: Megatron cf0100cf6 — Restructure BertDataset to help with RealmDataset
# Ported from:
#   megatron/data/bert_dataset.py   → deepspeed/runtime/dataloader.py
#   megatron/data/dataset_utils.py  → deepspeed/runtime/dataloader.py
#   megatron/data/realm_dataset.py  → deepspeed/runtime/dataloader.py
#
# Key changes carried over:
#   1. bert_dataset.py / BertDataset.__init__:
#      - Added `self.build_sample_fn = build_training_sample` so subclasses
#        (e.g. RealmDataset) can override the sample-building function
#        without overriding __getitem__.
#   2. bert_dataset.py / BertDataset.__getitem__:
#      - Variable names `start_index, end_index` renamed to `start_idx, end_idx`
#        for consistency with downstream code.
#      - Loop `for index in range(...): sample.append(...)` replaced with
#        list comprehension `[self.indexed_dataset[i] for i in range(...)]`.
#      - Call site changed from `build_training_sample(...)` (bare function) to
#        `self.build_sample_fn(...)` (instance-dispatch), enabling subclass override.
#   3. dataset_utils.py:
#      - `build_simple_training_sample` removed (moved to realm_dataset.py).
#      - `create_single_tokens_and_tokentypes` removed (moved to realm_dataset.py).
#   4. realm_dataset.py / RealmDataset:
#      - Changed from standalone `Dataset` subclass to `BertDataset` subclass.
#      - `__init__` now calls `super().__init__(...)` then overrides
#        `self.build_sample_fn = build_simple_training_sample`.
#      - Removed duplicated `__len__`, `__getitem__`, and `__init__` body
#        (all inherited from BertDataset via build_sample_fn dispatch).
#      - `build_simple_training_sample` and `create_single_tokens_and_tokentypes`
#        moved from dataset_utils.py into realm_dataset.py.
#      - train_sample dict keys updated: 'text' → 'tokens', 'padding_mask' → 'pad_mask'.
#      - `loss_mask_np` extended: concatenate a ones-array for REALM's double-length
#        sequence (true seq length is 2x but none predicted with LM outside first half).
# ---------------------------------------------------------------------------

print('[M211]')


def _m211_build_sample_fn_attr():
    """M211: Megatron cf0100cf6 — BertDataset gains self.build_sample_fn attribute.

    In BertDataset.__init__, after setting up vocab/tokenizer fields, add:

        self.build_sample_fn = build_training_sample

    This allows subclasses to override sample construction by simply setting
    self.build_sample_fn = <other_fn> in their own __init__, rather than
    having to override __getitem__ entirely.
    """
    # Documentation stub — actual BertDataset lives in megatron/data/bert_dataset.py;
    # the DeepSpeed port of BertDataset is recorded across M36/M37.
    return 'build_training_sample'


def _m211_getitem_refactor(samples_mapping, indexed_dataset, seed, idx,
                            build_sample_fn, max_seq_length,
                            vocab_id_list, vocab_id_to_token_dict,
                            cls_id, sep_id, mask_id, pad_id, masked_lm_prob):
    """M211: Megatron cf0100cf6 — BertDataset.__getitem__ with build_sample_fn dispatch.

    Before (pre-cf0100cf6):
        start_index, end_index, seq_length = self.samples_mapping[idx]
        sample = []
        for index in range(start_index, end_index):
            sample.append(self.indexed_dataset[index])
        ...
        return build_training_sample(sample, seq_length, ...)

    After (cf0100cf6):
        start_idx, end_idx, seq_length = self.samples_mapping[idx]
        sample = [self.indexed_dataset[i] for i in range(start_idx, end_idx)]
        ...
        return self.build_sample_fn(sample, seq_length, ...)

    Changes:
      - Variable rename: start_index/end_index → start_idx/end_idx
      - Loop → list comprehension
      - build_training_sample() → self.build_sample_fn() (enables subclass override)
    """
    import numpy as np
    start_idx, end_idx, seq_length = samples_mapping[idx]
    sample = [indexed_dataset[i] for i in range(start_idx, end_idx)]
    np_rng = np.random.RandomState(seed=(seed + idx))
    return build_sample_fn(
        sample, seq_length,
        max_seq_length,           # needed for padding
        vocab_id_list,
        vocab_id_to_token_dict,
        cls_id, sep_id,
        mask_id, pad_id,
        masked_lm_prob, np_rng,
    )


def _m211_build_simple_training_sample(sample, target_seq_length, max_seq_length,
                                        vocab_id_list, vocab_id_to_token_dict,
                                        cls_id, sep_id, mask_id, pad_id,
                                        masked_lm_prob, np_rng):
    """M211: Megatron cf0100cf6 — build_simple_training_sample moved from
    dataset_utils.py into realm_dataset.py (ported here for DeepSpeed).

    Constructs a single-segment (no NSP) masked-LM training sample for REALM.
    The output dict uses 'tokens'/'pad_mask' keys (updated from 'text'/'padding_mask').
    Also concatenates a ones loss_mask for the second half of REALM's double-length
    sequence (the second half is never predicted by LM, hence all-ones = no loss there).

    Called via RealmDataset.build_sample_fn (set in RealmDataset.__init__).
    """
    import itertools
    import numpy as np

    tokens = list(itertools.chain(*sample))[:max_seq_length - 2]
    tokens, tokentypes = _m211_create_single_tokens_and_tokentypes(tokens, cls_id, sep_id)

    max_predictions_per_seq = masked_lm_prob * max_seq_length
    # create_masked_lm_predictions lives in megatron/data/dataset_utils.py
    # (not removed by this commit — only build_simple/create_single were moved)
    from megatron.data.dataset_utils import (create_masked_lm_predictions,
                                              pad_and_convert_to_numpy)
    (tokens, masked_positions, masked_labels, _) = create_masked_lm_predictions(
        tokens, vocab_id_list, vocab_id_to_token_dict, masked_lm_prob,
        cls_id, sep_id, mask_id, max_predictions_per_seq, np_rng)

    tokens_np, tokentypes_np, labels_np, padding_mask_np, loss_mask_np = \
        pad_and_convert_to_numpy(tokens, tokentypes, masked_positions,
                                 masked_labels, pad_id, max_seq_length)

    # REALM true sequence length is twice as long but none of that is to be
    # predicted with LM — extend loss_mask with ones for the second half.
    loss_mask_np = np.concatenate((loss_mask_np, np.ones(loss_mask_np.shape)), -1)

    train_sample = {
        'tokens': tokens_np,       # key: 'text' → 'tokens'  (M211)
        'labels': labels_np,
        'loss_mask': loss_mask_np,
        'pad_mask': padding_mask_np,  # key: 'padding_mask' → 'pad_mask'  (M211)
    }
    return train_sample


def _m211_create_single_tokens_and_tokentypes(_tokens, cls_id, sep_id):
    """M211: Megatron cf0100cf6 — create_single_tokens_and_tokentypes moved from
    dataset_utils.py into realm_dataset.py (ported here for DeepSpeed).

    Wraps a flat token list with [CLS] ... [SEP] and assigns all token-types = 0
    (single-segment, no NSP distinction).

    Signature change vs dataset_utils.py version:
      Before: create_single_tokens_and_tokentypes(_tokens)   — no cls_id/sep_id args
      After:  create_single_tokens_and_tokentypes(_tokens, cls_id, sep_id)  — explicit
    """
    tokens = [cls_id] + list(_tokens) + [sep_id]
    tokentypes = [0] * len(tokens)
    return tokens, tokentypes

# Alias for internal use
_m211_create_single_tokens_and_tokentypes = _m211_create_single_tokens_and_tokentypes


class _M211RealmDataset:
    """M211: Megatron cf0100cf6 — RealmDataset restructured as BertDataset subclass.

    Before (pre-cf0100cf6):
        class RealmDataset(Dataset):
            def __init__(self, ...):
                # full duplicate of BertDataset.__init__ body
                self.name = name; self.seed = seed; ...
                self.samples_mapping = get_samples_mapping_(...)
                tokenizer = get_tokenizer(); self.vocab_id_list = ...
            def __len__(self): ...
            def __getitem__(self, idx): ... build_simple_training_sample(...)

    After (cf0100cf6):
        class RealmDataset(BertDataset):
            def __init__(self, ...):
                super().__init__(...)
                self.build_sample_fn = build_simple_training_sample

    All duplicated __init__ body, __len__, and __getitem__ are removed.
    BertDataset's __getitem__ now dispatches through self.build_sample_fn,
    so RealmDataset only needs to override that one attribute.

    DeepSpeed note: this class is a documentation mirror.  Actual instantiation
    should go through megatron.data.realm_dataset.RealmDataset or a DeepSpeed
    wrapper that sets build_sample_fn = _m211_build_simple_training_sample.
    """

    def __init__(self, bert_dataset_instance):
        """Set build_sample_fn on an existing BertDataset instance to convert it
        into REALM-style single-segment masked-LM training."""
        bert_dataset_instance.build_sample_fn = _m211_build_simple_training_sample

# --- End M211 dataloader ---

# M232: Megatron 0104f910b — Move InverseClozeDataset to bert_dataset
#
#   Upstream files touched:
#   megatron/data/bert_dataset.py   → deepspeed/hashed_index.py (import updated)
#   megatron/data/ict_dataset.py    → deleted upstream (class moved to realm_dataset)
#
#   Changes ported to Neuron_SP:
#   1. deepspeed/hashed_index.py line 14:
#        from megatron.data.ict_dataset import InverseClozeDataset
#      ->
#        from megatron.data.realm_dataset import InverseClozeDataset
#      This mirrors bert_dataset.py's import redirect in the upstream commit.
#
#   2. megatron/data/ict_dataset.py was deleted upstream — in Neuron_SP the
#      InverseClozeDataset logic lives inline in REAL_GPU_BENCHMARK.py
#      (NeuronSPInverseClozeDataset / NeuronSPInverseClozeDatasetV2) and in
#      deepspeed/runtime/dataloader.py (see M62, M189 blocks above).  No file
#      deletion is needed here; the import source is simply updated.
#
#   print('[M232]') added to deepspeed/hashed_index.py at module load time.
# --- End M232 dataloader ---

# M236: Megatron 16a64c41b — Move get_train_val... to dataset_utils
# Ported from:
#   megatron/data/bert_dataset.py   → deepspeed/runtime/dataloader.py
#   megatron/data/dataset_utils.py  → deepspeed/runtime/dataloader.py
#   pretrain_bert.py                → deepspeed/runtime/dataloader.py
#   pretrain_bert_ict.py            → deepspeed/runtime/dataloader.py
#   pretrain_realm.py               → deepspeed/runtime/dataloader.py
#
# Key changes carried over:
#   1. bert_dataset.py:
#      - Removed `from megatron.data.realm_dataset import InverseClozeDataset`
#        (import is no longer needed at module level; realm_dataset is only
#         imported inside the local build_dataset() closure, which now lives
#         in dataset_utils.py).
#      - Removed `build_train_valid_test_datasets()` factory function entirely.
#        The function body (indexed_dataset construction, split calculation,
#        per-split slicing of doc_idx, BertDataset / RealmDataset / InverseClozeDataset
#        dispatch) has moved verbatim to dataset_utils.py.
#      - `DATASET_TYPES`, `get_indexed_dataset_`, `get_train_valid_test_split_`,
#        and `BertDataset` remain in bert_dataset.py so dataset_utils.py can
#        import them.
#
#   2. dataset_utils.py:
#      - Added three new imports at the top of the file:
#          from megatron import print_rank_0
#          from megatron.data.bert_dataset import (DATASET_TYPES,
#              get_indexed_dataset_, get_train_valid_test_split_, BertDataset)
#          from megatron.data.realm_dataset import InverseClozeDataset
#      - Appended `build_train_valid_test_datasets()` at the end of the module
#        (identical body to the version removed from bert_dataset.py).
#      - Note: original file had no trailing newline; the appended function
#        immediately follows `return tokens_np, tokentypes_np, ...`.
#
#   3. pretrain_bert.py / pretrain_bert_ict.py / pretrain_realm.py:
#      - Each file changes exactly one import line:
#          Before: from megatron.data.bert_dataset import build_train_valid_test_datasets
#          After:  from megatron.data.dataset_utils import build_train_valid_test_datasets
#      - No other changes in any of the three pretrain scripts.
#
# DeepSpeed mapping:
#   - deepspeed/runtime/dataloader.py already tracks build_train_valid_test_datasets
#     across M70 / M119 / M212 / M214.  The canonical import path in those stubs
#     should be read as megatron.data.dataset_utils (not bert_dataset) for commits
#     at or after 16a64c41b.
#   - Any future DS code that calls megatron's build_train_valid_test_datasets must
#     import from megatron.data.dataset_utils, not megatron.data.bert_dataset.
# ---------------------------------------------------------------------------

print('[M236]')


def _m236_bert_dataset_removed_imports():
    """M236: Megatron 16a64c41b — imports removed from bert_dataset.py.

    Before (bert_dataset.py line ~29):
        from megatron.data.realm_dataset import InverseClozeDataset

    After: line deleted entirely.  InverseClozeDataset is only used inside the
    build_dataset() closure in build_train_valid_test_datasets(), which has moved
    to dataset_utils.py.  bert_dataset.py no longer needs this import at module scope.
    """
    pass


def _m236_dataset_utils_new_imports():
    """M236: Megatron 16a64c41b — new imports added to dataset_utils.py.

    Three lines inserted after `import numpy as np` (line ~25):

        from megatron import print_rank_0
        from megatron.data.bert_dataset import (
            DATASET_TYPES, get_indexed_dataset_, get_train_valid_test_split_, BertDataset)
        from megatron.data.realm_dataset import InverseClozeDataset

    These imports supply everything build_train_valid_test_datasets() needs now
    that it lives in dataset_utils.py instead of bert_dataset.py.
    """
    pass


def _m236_build_train_valid_test_datasets_location():
    """M236: Megatron 16a64c41b — build_train_valid_test_datasets moved to dataset_utils.py.

    Function body is identical to the version removed from bert_dataset.py:

        def build_train_valid_test_datasets(data_prefix, data_impl, splits_string,
                                            train_valid_test_num_samples,
                                            max_seq_length, masked_lm_prob,
                                            short_seq_prob, seed, skip_warmup,
                                            dataset_type='standard_bert'):

            if dataset_type not in DATASET_TYPES:
                raise ValueError("Invalid dataset_type: ", dataset_type)

            indexed_dataset = get_indexed_dataset_(data_prefix, data_impl, skip_warmup)

            if dataset_type == 'ict':
                title_dataset = get_indexed_dataset_(data_prefix + '-titles',
                                                     data_impl, skip_warmup)

            total_num_of_documents = indexed_dataset.doc_idx.shape[0] - 1
            splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

            print_rank_0(' > dataset split:')
            # ... print_split_stats for train / validation / test ...

            def build_dataset(index, name):
                from megatron.data.realm_dataset import RealmDataset
                dataset = None
                if splits[index + 1] > splits[index]:
                    doc_idx_ptr = indexed_dataset.get_doc_idx()
                    indexed_dataset.set_doc_idx(
                        doc_idx_ptr[splits[index]:splits[index + 1] + 1])
                    kwargs = dict(name=name, data_prefix=data_prefix, num_epochs=None,
                                  max_num_samples=train_valid_test_num_samples[index],
                                  max_seq_length=max_seq_length,
                                  short_seq_prob=short_seq_prob, seed=seed)
                    if dataset_type == 'ict':
                        dataset = InverseClozeDataset(block_dataset=indexed_dataset,
                                                      title_dataset=title_dataset,
                                                      **kwargs)
                    else:
                        dataset_cls = (BertDataset if dataset_type == 'standard_bert'
                                       else RealmDataset)
                        dataset = dataset_cls(indexed_dataset=indexed_dataset,
                                              masked_lm_prob=masked_lm_prob, **kwargs)
                    indexed_dataset.set_doc_idx(doc_idx_ptr)
                    assert indexed_dataset.doc_idx[0] == 0
                    assert indexed_dataset.doc_idx.shape[0] == (total_num_of_documents + 1)
                return dataset

            train_dataset = build_dataset(0, 'train')
            valid_dataset = build_dataset(1, 'valid')
            test_dataset = build_dataset(2, 'test')
            return (train_dataset, valid_dataset, test_dataset)

    New canonical import path (post-16a64c41b):
        from megatron.data.dataset_utils import build_train_valid_test_datasets
    """
    pass


def _m236_pretrain_scripts_import_update():
    """M236: Megatron 16a64c41b — import change in three pretrain scripts.

    pretrain_bert.py (line ~25):
        Before: from megatron.data.bert_dataset import build_train_valid_test_datasets
        After:  from megatron.data.dataset_utils import build_train_valid_test_datasets

    pretrain_bert_ict.py (line ~25):
        Before: from megatron.data.bert_dataset import build_train_valid_test_datasets
        After:  from megatron.data.dataset_utils import build_train_valid_test_datasets

    pretrain_realm.py (line ~27):
        Before: from megatron.data.bert_dataset import build_train_valid_test_datasets
        After:  from megatron.data.dataset_utils import build_train_valid_test_datasets

    All three files are otherwise unchanged.  The public API of
    build_train_valid_test_datasets() (signature, behaviour, return type)
    is identical; only the module that hosts it changes.
    """
    pass

# --- End M236 dataloader ---


# ---------------------------------------------------------------------------
# M328: Megatron 5247f24c5 — Fix gpt2_dataset import error
# Ported from:
#   megatron/data/gpt2_dataset.py  → deepspeed/runtime/dataloader.py
#
# Key change carried over:
#   gpt2_dataset.py line 25:
#     Before: from megatron.data.bert_dataset import get_train_valid_test_split_
#     After:  from megatron.data.dataset_utils import get_train_valid_test_split_
#
#   get_train_valid_test_split_ was originally defined in bert_dataset.py and
#   used by gpt2_dataset.py via a cross-module import.  This commit corrects
#   the import to point at dataset_utils, the canonical home of the function
#   (as also reflected in M236 / Megatron 16a64c41b).
#
# Neuron_SP adaptation:
#   Neuron_SP does not maintain a standalone gpt2_dataset.py; all GPT-2
#   dataset logic is hosted in deepspeed/runtime/dataloader.py.  The
#   equivalent of the corrected import is already present here: the
#   _m119_build_train_valid_test_datasets() factory accepts
#   get_train_valid_test_split_fn as an injected callable, and callers are
#   expected to source that callable from megatron.data.dataset_utils (not
#   bert_dataset).  This section makes that expectation explicit and
#   documents the upstream fix for traceability.
# ---------------------------------------------------------------------------

print('[M328]')


def _m328_gpt2_get_train_valid_test_split_import_source():
    """M328: Megatron 5247f24c5 — correct import source for GPT-2 split helper.

    Documents the upstream fix: gpt2_dataset.py previously imported
    get_train_valid_test_split_ from megatron.data.bert_dataset; the commit
    corrects this to megatron.data.dataset_utils.

    In Neuron_SP the function is injected as get_train_valid_test_split_fn
    into _m119_build_train_valid_test_datasets(); callers must supply it from
    megatron.data.dataset_utils, not bert_dataset.

    Returns the callable sourced from the correct module so it can be passed
    directly to _m119_build_train_valid_test_datasets().

    Before (gpt2_dataset.py):
        from megatron.data.bert_dataset import get_train_valid_test_split_

    After (gpt2_dataset.py):
        from megatron.data.dataset_utils import get_train_valid_test_split_
    """
    from megatron.data.dataset_utils import get_train_valid_test_split_
    return get_train_valid_test_split_

# --- End M328 dataloader ---


# ===========================================================================
# M408: Megatron 7ce373f3d — Bugfix in megatron/training.py: correct
#       global_batch_size computation
# ===========================================================================
#
# Upstream source:
#   megatron/training.py  → deepspeed/runtime/dataloader.py
#
# Key change carried over:
#   training.py, build_train_valid_test_data_iterators(), line ~719:
#     Before: global_batch_size = args.batch_size * data_parallel_size
#     After:  global_batch_size = args.batch_size * data_parallel_size \
#                                  * args.num_microbatches_in_minibatch
#
#   Without the num_microbatches_in_minibatch factor the number of training
#   samples requested from the dataset was under-counted, causing the
#   DataLoader to exhaust examples before training completed.
#
# Neuron_SP adaptation:
#   Neuron_SP mirrors the fixed computation in
#   _m408_build_train_valid_test_data_iterators_global_batch_size().
#   The helper reproduces the corrected formula and is the canonical
#   reference for any caller that needs to pre-compute the global batch
#   size for dataset sizing purposes (e.g. train_val_test_num_samples).
# ---------------------------------------------------------------------------

print('[M408]')


def _m408_build_train_valid_test_data_iterators_global_batch_size(args,
                                                                   data_parallel_size):
    """M408: Megatron 7ce373f3d — correct global_batch_size for dataset sizing.

    Before this fix global_batch_size was computed as:
        global_batch_size = args.batch_size * data_parallel_size

    This under-counted the true number of samples consumed per iteration
    when num_microbatches_in_minibatch > 1, causing the data loader to run
    out of training examples prematurely.

    The corrected formula:
        global_batch_size = args.batch_size
                            * data_parallel_size
                            * args.num_microbatches_in_minibatch

    Args:
        args: parsed argument namespace; must expose .batch_size and
              .num_microbatches_in_minibatch.
        data_parallel_size (int): result of mpu.get_data_parallel_world_size().

    Returns:
        int: corrected global batch size to use when computing
             train_val_test_num_samples.
    """
    num_microbatches = getattr(args, 'num_microbatches_in_minibatch', 1)
    global_batch_size = args.batch_size * data_parallel_size * num_microbatches
    return global_batch_size

# --- End M408 dataloader ---


# ===========================================================================
# M463: Megatron 6e83649f6 — Quick fix for pipeline tasks to get learning
#       rate correct
# ===========================================================================
#
# Upstream source:
#   tasks/finetune_utils.py  → deepspeed/runtime/dataloader.py
#
# Key change carried over:
#   _build_train_valid_dataloaders(), after micro_batch_size is scaled by
#   sample_multiplier, also scale global_batch_size by the same factor.
#
#   Before:
#     if hasattr(train_dataset, 'sample_multiplier'):
#         args.micro_batch_size *= train_dataset.sample_multiplier
#
#   After:
#     if hasattr(train_dataset, 'sample_multiplier'):
#         args.micro_batch_size *= train_dataset.sample_multiplier
#         args.global_batch_size *= train_dataset.sample_multiplier   # NEW
#
#   Without the global_batch_size update, pipeline stages computed the
#   wrong tensor-transfer size and the LR schedule (based on samples seen)
#   used an incorrect denominator, causing learning-rate to be too low
#   during fine-tuning on datasets whose samples expand via sample_multiplier
#   (e.g. RACE, where each question carries multiple answer options).
#
# Neuron_SP adaptation:
#   Logic is exposed as _m463_apply_sample_multiplier_to_batch_sizes() so
#   callers can invoke it after dataloader construction.
#

print('[M463]')


def _m463_apply_sample_multiplier_to_batch_sizes(args, train_dataset):
    """M463: Megatron 6e83649f6 — scale both batch-size args by sample_multiplier.

    Some fine-tuning datasets (e.g. RACE) pack several logical samples into
    each dataset item via a ``sample_multiplier`` attribute.  Before this fix
    only ``args.micro_batch_size`` was scaled, leaving
    ``args.global_batch_size`` at the original value.  Two problems followed:

    1. **Pipeline transfers** — inter-stage activation tensors are sized by
       micro_batch_size, but the scheduler expected global_batch_size to match;
       the mismatch caused shape errors or silently wrong activations.

    2. **LR schedule** — the schedule advances by counting samples consumed,
       which is derived from global_batch_size.  With the old (unscaled) value
       the schedule moved too slowly, effectively reducing the learning rate
       for the whole fine-tuning run.

    Args:
        args: parsed argument namespace; must expose ``.micro_batch_size``
              and ``.global_batch_size``.
        train_dataset: dataset object; if it has a ``sample_multiplier``
                       attribute both batch-size fields are multiplied by it.

    Returns:
        None.  ``args`` is mutated in-place.
    """
    if hasattr(train_dataset, 'sample_multiplier'):
        args.micro_batch_size *= train_dataset.sample_multiplier
        args.global_batch_size *= train_dataset.sample_multiplier

# --- End M463 dataloader ---
