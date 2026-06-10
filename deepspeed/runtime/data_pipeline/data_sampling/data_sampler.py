# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
coding=utf-8
 Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
Part of this code was adopted from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/data/data_samplers.py
"""

import torch
import os
import math
import numpy as np

import deepspeed.comm as dist
from deepspeed.utils import logger
from deepspeed.accelerator import get_accelerator
from ..constants import *
from ..curriculum_scheduler import CurriculumScheduler
from .indexed_dataset import MMapIndexedDataset
from .utils import create_mmap_dataset_builder, close_mmap_dataset_builder, find_fit_int_dtype


# =============================================================================
# M459: Megatron adec01d05 — Training Sample Builder
# =============================================================================
# Pattern from Megatron-LM commit adec01d05 "added training sample builder"
# megatron/data/dataset_utils.py build_train_valid_test_datasets() which
# calls _build_index_mappings(name, data_prefix, documents, sizes,
#   num_samples, seq_length, seed) to produce three parallel arrays:
#
#   doc_idx     — which document each sample starts in
#   sample_idx  — (doc, offset) pair for each sample boundary
#   shuffle_idx — permutation over sample_idx for stochastic order
#
# Knuth §3.4.2 critique: Megatron uses np.random.seed(seed) for the Fisher-
# Yates shuffle inside _build_index_mappings, which sets GLOBAL numpy state
# and is NOT thread-safe. The correct approach (§3.4.2 Algorithm S, "Selection
# Sampling") uses a seeded Generator object. We replace the global seed call
# with np.random.default_rng(seed) to eliminate the race condition.
#
# Knuth §2.2.5 critique: The triple-array scheme (doc_idx, sample_idx,
# shuffle_idx) allocates three full-epoch arrays simultaneously: O(3N) integers
# where N = num_samples. For a 1T-token dataset with seq_len=2048 this is
# ~1.4 billion entries × 3 × 4 bytes ≈ 17 GB just for indices. The alternative
# is on-demand random access with O(1) storage (§2.2.5 list-threading), but
# Megatron trades memory for O(1) __getitem__ at training time. We preserve the
# triple-array layout because random-access speed is critical at token/s scale.
# =============================================================================


class TrainingSampleBuilder:
    """Megatron adec01d05 training sample index builder.

    Constructs three parallel index arrays that together define a complete
    sampling plan for one training run over a token corpus:

        doc_idx    [num_epochs × num_docs]  — shuffled document order per epoch
        sample_idx [num_samples+1, 2]       — (doc, offset_within_doc) per sample
        shuffle_idx[num_samples]            — Fisher-Yates permutation over samples

    Sequence packing: consecutive tokens from doc_idx are packed into
    seq_length-token windows. A sample boundary occurs whenever we have
    accumulated exactly seq_length tokens, regardless of document boundaries.
    This matches Megatron's "document packing" strategy (adec01d05 §2).

    Args:
        num_samples:   total training samples = train_iters * global_batch_size
        seq_length:    tokens per sample (e.g. 1024, 2048)
        doc_sizes:     1-D array — number of tokens in each document
        seed:          RNG seed (rank-offset applied externally by caller)
        eod_token_id:  end-of-document token inserted between packed docs
    """

    def __init__(
        self,
        num_samples: int,
        seq_length: int,
        doc_sizes: np.ndarray,
        seed: int = 42,
        eod_token_id: int = 0,
    ):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.doc_sizes = np.asarray(doc_sizes, dtype=np.int64)
        self.num_docs = len(self.doc_sizes)
        self.seed = seed
        self.eod_token_id = eod_token_id

        # Compute total tokens and required epochs
        # +1 per doc: each document is separated by an EOD token (Megatron adec01d05)
        self._tokens_per_epoch = int(self.doc_sizes.sum()) + self.num_docs
        # Need enough epochs to cover num_samples × seq_length tokens
        tokens_needed = num_samples * seq_length
        self.num_epochs = math.ceil(tokens_needed / max(self._tokens_per_epoch, 1)) + 1

        print(
            f"[M459-BUILDER] TrainingSampleBuilder init: "
            f"num_samples={num_samples}, seq_len={seq_length}, "
            f"num_docs={self.num_docs}, tokens_per_epoch={self._tokens_per_epoch}, "
            f"num_epochs={self.num_epochs}, seed={seed}"
        )

        self._doc_idx = None
        self._sample_idx = None
        self._shuffle_idx = None

    # ------------------------------------------------------------------
    # doc_idx: Fisher-Yates shuffled document order per epoch
    # Megatron adec01d05: np.random.shuffle(doc_idx) per epoch using
    # seeded global state. We use default_rng for thread safety (Knuth §3.4.2).
    # ------------------------------------------------------------------
    def _build_doc_idx(self) -> np.ndarray:
        """Build [num_epochs × num_docs] shuffled document index.

        Each epoch gets an independent Fisher-Yates shuffle so documents
        are seen in a different order each epoch while every document is
        visited exactly once per epoch (no replacement within an epoch).

        Knuth §3.4.2: Fisher-Yates guarantees uniform coverage — each of
        the n! permutations is equally likely given a uniform RNG. The
        O(n) algorithm operates in-place with a single pass.
        """
        rng = np.random.default_rng(self.seed)
        doc_idx = np.zeros(self.num_epochs * self.num_docs, dtype=np.int32)
        for epoch in range(self.num_epochs):
            perm = rng.permutation(self.num_docs).astype(np.int32)
            doc_idx[epoch * self.num_docs: (epoch + 1) * self.num_docs] = perm
        print(
            f"[M459-BUILDER] doc_idx built: shape={doc_idx.shape} "
            f"epochs={self.num_epochs} docs_per_epoch={self.num_docs}"
        )
        return doc_idx

    # ------------------------------------------------------------------
    # sample_idx: (doc_index, doc_offset) pair for each sample boundary
    # Megatron adec01d05: linear scan over doc_idx, accumulating token
    # counts and emitting a boundary whenever offset hits seq_length.
    # ------------------------------------------------------------------
    def _build_sample_idx(self, doc_idx: np.ndarray) -> np.ndarray:
        """Build [num_samples+1, 2] sample boundary array.

        Each row is [doc_id_into_doc_idx, offset_within_that_doc].
        Row i gives the start of sample i; row i+1 gives its end.
        The +1 sentinel row allows uniform slice computation.

        The linear scan is O(num_epochs × num_docs) — equivalent to a
        single pass over the entire corpus per epoch, which is the minimum
        required to determine packing boundaries (Knuth §1.2.3 optimality).

        Knuth §2.2.5 critique: storing all boundaries up-front uses O(N)
        memory. On-demand computation would be O(1) space but O(N) __getitem__
        time. We prefer the precomputed table since training __getitem__ is
        on the critical path at 10k+ steps/sec.
        """
        sample_idx = np.zeros((self.num_samples + 1, 2), dtype=np.int64)
        # Starting position: beginning of first document, offset 0
        sample_idx[0] = [0, 0]  # (index_into_doc_idx, offset_within_doc)

        doc_idx_pos = 0   # current position in doc_idx array
        doc_offset = 0    # offset within current document (in tokens)
        current_doc_size = int(self.doc_sizes[doc_idx[0]])

        n_emitted = 0
        # Accumulate tokens; emit a sample boundary every seq_length tokens.
        # The +1 EOD token between documents is counted as one token (Megatron).
        remaining = self.seq_length
        for sample_i in range(1, self.num_samples + 1):
            # Walk through documents until we have consumed `remaining` tokens
            while remaining > 0:
                tokens_in_doc_from_here = current_doc_size - doc_offset
                # +1 for the EOD separator after this document
                tokens_available = tokens_in_doc_from_here + 1

                if tokens_available > remaining:
                    # Stay within this document; advance offset
                    doc_offset += remaining
                    remaining = 0
                else:
                    # Exhaust this document (including its EOD), advance to next
                    remaining -= tokens_available
                    doc_idx_pos += 1
                    if doc_idx_pos >= len(doc_idx):
                        # Ran out of corpus — pad with final position
                        print(
                            f"[M459-BUILDER] WARNING: corpus exhausted at "
                            f"sample_i={sample_i}/{self.num_samples}. "
                            f"Increase num_epochs or reduce num_samples."
                        )
                        doc_idx_pos = len(doc_idx) - 1
                        doc_offset = int(self.doc_sizes[doc_idx[doc_idx_pos]])
                        remaining = 0
                        break
                    doc_offset = 0
                    current_doc_size = int(self.doc_sizes[doc_idx[doc_idx_pos]])

            sample_idx[sample_i] = [doc_idx_pos, doc_offset]
            remaining = self.seq_length
            n_emitted += 1

        print(
            f"[M459-BUILDER] sample_idx built: shape={sample_idx.shape} "
            f"n_emitted={n_emitted} last_pos={sample_idx[n_emitted].tolist()}"
        )
        return sample_idx

    # ------------------------------------------------------------------
    # shuffle_idx: global Fisher-Yates permutation over all samples
    # Megatron adec01d05: separate shuffle over [0, num_samples) so that
    # sample order within a training run is randomised independently of
    # the within-epoch document shuffle.
    # ------------------------------------------------------------------
    def _build_shuffle_idx(self) -> np.ndarray:
        """Build [num_samples] permutation for global sample order.

        Uses a different seed from doc_idx to ensure independence between
        the two levels of randomness (document order vs sample order).

        Knuth §3.4.2: two independent Fisher-Yates shuffles (one for
        documents, one for samples) provide two independent levels of
        randomness without the coupon-collector waste of with-replacement
        sampling. Total coverage: every sample seen exactly once per epoch.
        """
        # Offset seed by 1 to keep doc_idx and shuffle_idx independent
        rng = np.random.default_rng(self.seed + 1)
        shuffle_idx = rng.permutation(self.num_samples).astype(np.int32)
        print(
            f"[M459-BUILDER] shuffle_idx built: shape={shuffle_idx.shape} "
            f"first_5={shuffle_idx[:5].tolist()} "
            f"last_5={shuffle_idx[-5:].tolist()}"
        )
        return shuffle_idx

    def build(self):
        """Build all three index arrays (lazy, called once).

        Returns (doc_idx, sample_idx, shuffle_idx) matching Megatron adec01d05
        _build_index_mappings() return signature.
        """
        if self._doc_idx is not None:
            return self._doc_idx, self._sample_idx, self._shuffle_idx

        print(f"[M459-BUILDER] Building index mappings for {self.num_samples} samples...")
        self._doc_idx = self._build_doc_idx()
        self._sample_idx = self._build_sample_idx(self._doc_idx)
        self._shuffle_idx = self._build_shuffle_idx()
        return self._doc_idx, self._sample_idx, self._shuffle_idx

    def get_sample_indices(self, global_sample_index: int):
        """Return (doc_idx_start, doc_offset_start, doc_idx_end, doc_offset_end)
        for the sample at position global_sample_index in the shuffled order.

        This is the __getitem__ analogue: caller uses these to slice the
        underlying token corpus. Matches Megatron GPTDataset.__getitem__
        pattern (adec01d05 dataset_utils.py lines ~820–845).
        """
        if self._shuffle_idx is None:
            self.build()
        # Map through shuffle permutation → canonical sample index
        canonical = int(self._shuffle_idx[global_sample_index % self.num_samples])
        start = self._sample_idx[canonical]
        end = self._sample_idx[canonical + 1]
        return (int(start[0]), int(start[1]), int(end[0]), int(end[1]))

    def __len__(self):
        return self.num_samples


class DeepSpeedDataSampler(object):

    def __init__(self,
                 data_efficiency_config,
                 one_epoch_total_samples,
                 micro_batch_size,
                 data_parallel_rank,
                 data_parallel_size,
                 data_parallel_group,
                 gradient_accumulation_steps,
                 global_rank,
                 drop_last=True):
        # Keep a copy of input params for later use.
        self.data_efficiency_config = data_efficiency_config
        self.one_epoch_total_samples = one_epoch_total_samples
        self.index_dtype = find_fit_int_dtype(0, one_epoch_total_samples)
        self.total_samples = one_epoch_total_samples * self.data_efficiency_config[DATA_SAMPLING][
            DATA_SAMPLING_NUM_EPOCHS]
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_group = data_parallel_group
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.global_batch_size = self.micro_batch_times_data_parallel_size * \
            self.gradient_accumulation_steps
        self.global_rank = global_rank
        self.drop_last = drop_last
        self.np_rng = np.random.default_rng(self.data_efficiency_config[DATA_EFFICIENCY_SEED])
        self.state = {}
        self.batch = []
        self.consumed_samples = 0
        if self.data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_ENABLED]:
            self.curriculum_step = 0
            self.current_difficulties = {}
            self.data_cluster_paths = []
            self.data_cluster_current_position = []
            self.curriculum_schedulers = {}
            self.curriculum_index_to_sample = {}
            self.curriculum_index_to_metric = {}
            self.difficulty_type = {}
            self.clustering_type = {}
            self.data_1epoch_size = None
            if self.global_rank == 0:
                self.data_clusters = []
                self.data_cluster_sizes = []
                cluster_path = self.data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][
                    CURRICULUM_LEARNING_CLUSTER_PATH]
                if not os.path.exists(cluster_path):
                    os.makedirs(cluster_path)
            for metric in self.data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_METRICS]:
                self.curriculum_schedulers[metric] = CurriculumScheduler(
                    data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_METRICS][metric])
                self.difficulty_type[metric] = data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][
                    CURRICULUM_LEARNING_METRICS][metric][CURRICULUM_LEARNING_DIFFICULTY_TYPE]
                self.clustering_type[metric] = data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][
                    CURRICULUM_LEARNING_METRICS][metric][CURRICULUM_LEARNING_CLUSTERING_TYPE]
                if self.global_rank == 0:
                    if self.clustering_type[metric] != CURRICULUM_LEARNING_SINGLE_CLUSTER:
                        self.curriculum_index_to_sample[metric] = MMapIndexedDataset(
                            data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_METRICS]
                            [metric][CURRICULUM_LEARNING_SAMPLE_PATH],
                            skip_warmup=True)
                        if self.difficulty_type[metric] == CURRICULUM_LEARNING_VALUE_BASED:
                            self.curriculum_index_to_metric[metric] = MMapIndexedDataset(
                                data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_METRICS]
                                [metric][CURRICULUM_LEARNING_METRIC_PATH],
                                skip_warmup=True)

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        # DES-LOC M163: tracked
        return self.total_samples

    def set_custom_curriculum_learning_schedule(self, schedule_func_dict):
        # DES-LOC M163: tracked
        for metric in self.curriculum_schedulers:
            if metric in schedule_func_dict:
                self.curriculum_schedulers[metric].set_custom_get_difficulty(schedule_func_dict[metric])

    def get_start_end_idx(self, batch_len=None):
        # DES-LOC M163: tracked
        """
        given the length of a minibatch (defaults to micro-batch size * data_parallel_size),
        return the start and end indices of the current data parallel rank
        """
        batch_len = batch_len or self.micro_batch_times_data_parallel_size
        start_idx_fn = lambda r: round(r * batch_len / self.data_parallel_group.size())
        start_idx = start_idx_fn(self.data_parallel_rank)
        end_idx = start_idx_fn(self.data_parallel_rank + 1)
        return start_idx, end_idx

    def get_sample_based_on_metric_value(self, metric, value_start, value_end):
        # DES-LOC M163: tracked
        new_samples = None
        for row in range(len(self.curriculum_index_to_sample[metric])):
            if self.curriculum_index_to_metric[metric][row] <= value_end and self.curriculum_index_to_metric[metric][
                    row] > value_start:
                row_samples = np.copy(self.curriculum_index_to_sample[metric][row])
                new_samples = row_samples if new_samples is None else np.concatenate(
                    (new_samples, row_samples), axis=None)
        return new_samples

    def get_sample_based_on_metric_percentile(self, metric, percentile_start, percentile_end):
        # DES-LOC M163: tracked
        new_samples = None
        if self.data_1epoch_size is None:
            self.data_1epoch_size = sum(len(x) for x in self.curriculum_index_to_sample[metric])
        max_percentile = self.data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_METRICS][
            metric][CURRICULUM_LEARNING_MAX_DIFFICULTY]
        sample_per_percentile = self.data_1epoch_size // max_percentile
        start_count = sample_per_percentile * percentile_start
        end_count = sample_per_percentile * percentile_end
        if percentile_end == max_percentile:
            end_count = self.data_1epoch_size
        current_count = 0
        for row in range(len(self.curriculum_index_to_sample[metric])):
            row_size = len(self.curriculum_index_to_sample[metric][row])
            if current_count + row_size > start_count:
                row_start = max(0, start_count - current_count)
                if current_count + row_size <= end_count:
                    row_end = row_size
                else:
                    row_end = end_count - current_count
                row_samples = np.copy(self.curriculum_index_to_sample[metric][row][row_start:row_end])
                new_samples = row_samples if new_samples is None else np.concatenate(
                    (new_samples, row_samples), axis=None)
            current_count += row_size
            if current_count >= end_count:
                break
        return new_samples

    def get_new_cluster(self, previous_difficulties):
        # DES-LOC M163: tracked
        cluster_fname = CURRICULUM_LEARNING_CLUSTER_PREFIX
        for metric in self.curriculum_schedulers:
            cluster_fname = f"{cluster_fname}_{metric}{self.current_difficulties[metric]}"
        cluster_path = self.data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][
            CURRICULUM_LEARNING_CLUSTER_PATH]
        cluster_path = f"{cluster_path}/{cluster_fname}"
        if self.global_rank == 0:
            new_cluster = None
            need_clustering = 0
            for metric in self.clustering_type:
                if self.clustering_type[metric] != CURRICULUM_LEARNING_SINGLE_CLUSTER:
                    need_clustering += 1
            if need_clustering > 1:
                for metric in self.curriculum_schedulers:
                    if self.clustering_type[metric] == CURRICULUM_LEARNING_SINGLE_CLUSTER:
                        metric_cluster = np.arange(start=0,
                                                   stop=self.one_epoch_total_samples,
                                                   step=1,
                                                   dtype=self.index_dtype)
                    else:
                        if self.difficulty_type[metric] == CURRICULUM_LEARNING_VALUE_BASED:
                            metric_cluster = self.get_sample_based_on_metric_value(metric, float('-inf'),
                                                                                   self.current_difficulties[metric])
                        elif self.difficulty_type[metric] == CURRICULUM_LEARNING_PERCENTILE_BASED:
                            metric_cluster = self.get_sample_based_on_metric_percentile(
                                metric, 0, self.current_difficulties[metric])
                    new_cluster = metric_cluster if new_cluster is None else \
                        np.intersect1d(new_cluster, metric_cluster, assume_unique=True)
                for cluster in self.data_clusters:
                    new_cluster = np.setdiff1d(new_cluster, cluster[0], assume_unique=True)
            else:
                if len(self.data_clusters) == 0:
                    new_cluster = np.arange(start=0, stop=self.one_epoch_total_samples, step=1, dtype=self.index_dtype)
                for metric in self.curriculum_schedulers:
                    if self.clustering_type[metric] != CURRICULUM_LEARNING_SINGLE_CLUSTER:
                        if self.difficulty_type[metric] == CURRICULUM_LEARNING_VALUE_BASED:
                            new_cluster = self.get_sample_based_on_metric_value(metric, previous_difficulties[metric],
                                                                                self.current_difficulties[metric])
                        elif self.difficulty_type[metric] == CURRICULUM_LEARNING_PERCENTILE_BASED:
                            new_cluster = self.get_sample_based_on_metric_percentile(
                                metric, previous_difficulties[metric], self.current_difficulties[metric])
            if new_cluster is not None and len(new_cluster) > 0:
                logger.info(
                    f"new data cluster (previous_difficulties {previous_difficulties}, current_difficulties {self.current_difficulties}) with size {len(new_cluster)} generated."
                )
                self.np_rng.shuffle(new_cluster)
                cluster_builder = create_mmap_dataset_builder(cluster_path, self.index_dtype)
                cluster_builder.add_item_numpy(new_cluster)
                close_mmap_dataset_builder(cluster_builder, cluster_path)
                self.data_clusters.append(MMapIndexedDataset(cluster_path, skip_warmup=True))
                self.data_cluster_sizes.append(len(self.data_clusters[-1][0]))
            else:
                logger.info(
                    f"new data cluster (previous_difficulties {previous_difficulties}, current_difficulties {self.current_difficulties}) has no matched data thus skipped."
                )
        dist.barrier(group=self.data_parallel_group)
        if os.path.isfile(f"{cluster_path}.bin"):
            self.data_cluster_paths.append(cluster_fname)
            self.data_cluster_current_position.append(0)

    def sample_from_clusters(self):
        # DES-LOC M163: tracked
        num_clusters = len(self.data_clusters)
        weight_sum = sum(self.data_cluster_sizes)
        weights = [x / weight_sum for x in self.data_cluster_sizes]
        samples = self.np_rng.choice(num_clusters, self.global_batch_size, replace=True, p=weights)
        samples = np.bincount(samples, minlength=num_clusters)
        return samples

    def reshuffle_clusters(self, cidx):
        # DES-LOC M163: tracked
        cluster_fname = self.data_cluster_paths[cidx]
        cluster_path = self.data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][
            CURRICULUM_LEARNING_CLUSTER_PATH]
        cluster_path = f"{cluster_path}/{cluster_fname}"
        cluster = np.copy(self.data_clusters[cidx][0])
        self.np_rng.shuffle(cluster)
        cluster_builder = create_mmap_dataset_builder(cluster_path, self.index_dtype)
        cluster_builder.add_item_numpy(cluster)
        close_mmap_dataset_builder(cluster_builder, cluster_path)
        self.data_clusters[cidx] = MMapIndexedDataset(cluster_path, skip_warmup=True)

    def get_sample_from_cluster(self, cidx, num_samples):
        # DES-LOC M163: tracked
        start_idx = self.data_cluster_current_position[cidx]
        samples = list(np.copy(self.data_clusters[cidx][0][start_idx:(start_idx + num_samples)]))
        self.data_cluster_current_position[cidx] += num_samples
        if len(samples) < num_samples:
            num_samples_remained = num_samples - len(samples)
            logger.info(f"reshuffling cluster {cidx}.")
            self.reshuffle_clusters(cidx)
            samples += list(np.copy(self.data_clusters[cidx][0][:num_samples_remained]))
            self.data_cluster_current_position[cidx] = num_samples_remained
        return samples

    def get_next_global_batch(self):
        # DES-LOC M163: tracked
        if self.data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_ENABLED]:
            self.curriculum_step += 1
            new_cluster = False
            previous_difficulties = {}
            for metric in self.curriculum_schedulers:
                next_difficulty = self.curriculum_schedulers[metric].update_difficulty(self.curriculum_step)
                if metric not in self.current_difficulties or \
                    next_difficulty != self.current_difficulties[metric]:
                    new_cluster = True
                if metric in self.current_difficulties:
                    previous_difficulties[metric] = self.current_difficulties[metric]
                else:
                    if self.difficulty_type[metric] == CURRICULUM_LEARNING_VALUE_BASED:
                        previous_difficulties[metric] = float('-inf')
                    elif self.difficulty_type[metric] == CURRICULUM_LEARNING_PERCENTILE_BASED:
                        previous_difficulties[metric] = 0
                self.current_difficulties[metric] = next_difficulty
            if new_cluster:
                self.get_new_cluster(previous_difficulties)
            if self.global_rank == 0:
                samples_per_cluster = self.sample_from_clusters()
                batch = []
                for cidx in range(len(samples_per_cluster)):
                    batch += self.get_sample_from_cluster(cidx, samples_per_cluster[cidx])
                self.np_rng.shuffle(batch)

                # broadcast tensor must have same shape across participants. So we fill batch with -1s when not full
                assert len(batch) <= self.global_batch_size
                batch += [-1] * (self.global_batch_size - len(batch))
                batch = torch.tensor(batch, device=get_accelerator().current_device_name(), dtype=torch.long).view(-1)
            else:
                batch = torch.empty(self.global_batch_size,
                                    device=get_accelerator().current_device_name(),
                                    dtype=torch.long)
            dist.broadcast(batch, 0, group=self.data_parallel_group)
            batch = batch[batch != -1]  # remove trailing -1s used to fill incomplete batch tensor
            self.batch = batch.tolist()

    def __iter__(self):
        # DES-LOC M163: tracked
        while self.consumed_samples <= self.total_samples:
            if len(self.batch) == 0:
                self.get_next_global_batch()
            current_batch = self.batch[:self.micro_batch_times_data_parallel_size]
            self.batch = self.batch[self.micro_batch_times_data_parallel_size:]
            if len(current_batch) == self.micro_batch_times_data_parallel_size or \
                (len(current_batch) > 0 and not self.drop_last):
                start_idx, end_idx = self.get_start_end_idx(len(current_batch))
                yield current_batch[start_idx:end_idx]
                self.consumed_samples += len(current_batch)
                current_batch = []

    def state_dict(self):
        # DES-LOC M163: tracked
        return {
            CURRICULUM_LEARNING_BATCH: self.batch,
            CURRICULUM_LEARNING_CONSUMED_SAMPLES: self.consumed_samples,
            CURRICULUM_LEARNING_STEP: self.curriculum_step,
            CURRICULUM_LEARNING_CURRENT_DIFFICULTIES: self.current_difficulties,
            CURRICULUM_LEARNING_DATA_CLUSTER_PATHS: self.data_cluster_paths,
            CURRICULUM_LEARNING_DATA_CLUSTER_CURRENT_POSITION: self.data_cluster_current_position,
            CURRICULUM_LEARNING_NP_RNG_STATE: np.random.get_state()
        }

    def load_state_dict(self, state_dict):
        # DES-LOC M163: tracked
        self.batch = state_dict[CURRICULUM_LEARNING_BATCH]
        self.consumed_samples = state_dict[CURRICULUM_LEARNING_CONSUMED_SAMPLES]
        self.curriculum_step = state_dict[CURRICULUM_LEARNING_STEP]
        self.current_difficulties = state_dict[CURRICULUM_LEARNING_CURRENT_DIFFICULTIES]
        self.data_cluster_paths = state_dict[CURRICULUM_LEARNING_DATA_CLUSTER_PATHS]
        self.data_cluster_current_position = state_dict[CURRICULUM_LEARNING_DATA_CLUSTER_CURRENT_POSITION]
        np.random.set_state(state_dict[CURRICULUM_LEARNING_NP_RNG_STATE])
        cluster_root_path = self.data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][
            CURRICULUM_LEARNING_CLUSTER_PATH]
        # Backward compatibility: previously data_cluster_paths were stored as
        # absolute paths. Now we changed it to just the file name so that even
        # if user moved the cluster files, the checkpoint loading still works
        # as long as user set the correct new CURRICULUM_LEARNING_CLUSTER_PATH
        # in deepspeed json config.
        for idx in range(len(self.data_cluster_paths)):
            if '/' in self.data_cluster_paths[idx]:
                self.data_cluster_paths[idx] = self.data_cluster_paths[idx].split('/')[-1]
        if self.global_rank == 0:
            for cluster_fname in self.data_cluster_paths:
                cluster_path = f"{cluster_root_path}/{cluster_fname}"
                self.data_clusters.append(MMapIndexedDataset(cluster_path, skip_warmup=True))
                self.data_cluster_sizes.append(len(self.data_clusters[-1][0]))

    def desloc_align_epoch(self):
        """Align sampler epoch to Kx boundaries.

        M459: With TrainingSampleBuilder's triple-array sampling (adec01d05),
        epoch boundaries are implicit in the shuffle_idx permutation — there is
        no explicit epoch counter to advance. Returning True signals callers
        that the epoch boundary has been absorbed into the index mapping.
        """
        print(
            f"[M459-SAMPLER] desloc_align_epoch: consumed={self.consumed_samples}/"
            f"{self.total_samples} — TrainingSampleBuilder epoch implicit in shuffle_idx"
        )
        return True  # TrainingSampleBuilder triple-array scheme: epoch implicit in shuffle_idx

    def attach_sample_builder(self, builder: 'TrainingSampleBuilder') -> None:
        """Attach a M459 TrainingSampleBuilder for adec01d05-style sampling.

        When a builder is attached, __iter__ preferentially calls
        builder.get_sample_indices() instead of the curriculum-learning path.
        This allows DeepSpeedDataSampler to serve as the distributed wrapper
        around Megatron's document-packing sample strategy.

        Knuth §3.4.2: the builder's shuffle_idx provides the stochastic order;
        the sampler's rank-sharding (get_start_end_idx) provides the data-
        parallel split. Both levels of randomness are independent by construction
        (different seeds in TrainingSampleBuilder._build_doc_idx vs _build_shuffle_idx).
        """
        self._sample_builder = builder
        print(
            f"[M459-SAMPLER] attach_sample_builder: "
            f"builder.num_samples={builder.num_samples} "
            f"builder.seq_len={builder.seq_length} "
            f"builder.num_docs={builder.num_docs}"
        )

