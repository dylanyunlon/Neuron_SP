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
#   2. DistributedBatchSampler (from megatron.data_utils.samplers) is now
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
    Uses SequentialSampler + DistributedBatchSampler (from data_utils/samplers) so
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
# Ported from: megatron/data_utils/datasets.py
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
# Ported from: configure_data.py + megatron/data_utils/datasets.py +
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
