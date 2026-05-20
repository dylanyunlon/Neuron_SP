import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import deepspeed.comm as dist
from .sp_dp_registry import finalize_a2a_pass, fence_all_sp_handles


@dataclass
class SequenceHistogram:
    bin_counts: torch.Tensor = None
    total_tokens: int = 0
    num_bins: int = 256
    pass_index: int = 0
    timestamp_ns: int = 0
    rank: int = 0
    device_name: str = ""


@dataclass
class HistogramAccumulator:
    global_histogram: torch.Tensor = None
    per_rank_histograms: Dict[int, torch.Tensor] = field(default_factory=dict)
    num_passes: int = 0
    candidate_count: int = 0
    filter_count: int = 0


@dataclass
class PassCounter:
    previous_len: int = 0
    current_len: int = 0
    filter_cnt: int = 0
    k: int = 0
    finished_block_cnt: int = 0


class SPHistogramKernel:

    def __init__(self, num_bins=256, device=None, model_scale=1.0):
        self._num_bins = max(64, int(num_bins * max(1.0, model_scale)))
        self._device = device or (torch.device(f"cuda:{torch.cuda.current_device()}")
                                  if torch.cuda.is_available() else torch.device("cpu"))
        self._accumulator = HistogramAccumulator()
        self._pass_count = 0
        self._counter = PassCounter()
        self._model_scale = model_scale

    def invoke_histogram_only(self, input_ids, seq_dim=1):
        seq_len = input_ids.shape[seq_dim]
        bin_edges = torch.linspace(0, seq_len, self._num_bins + 1,
                                   device=self._device, dtype=torch.float32)

        if input_ids.dtype in (torch.long, torch.int):
            values = input_ids.float()
        else:
            values = input_ids

        flat = values.reshape(-1)
        bin_indices = torch.bucketize(flat, bin_edges[1:-1])
        counts = torch.zeros(self._num_bins, device=self._device, dtype=torch.long)
        counts.scatter_add_(0, bin_indices.long(), torch.ones_like(bin_indices, dtype=torch.long))

        result = SequenceHistogram(
            bin_counts=counts,
            total_tokens=flat.numel(),
            num_bins=self._num_bins,
            pass_index=self._pass_count,
            timestamp_ns=time.time_ns(),
            rank=dist.get_rank() if dist.is_initialized() else 0,
        )
        self._pass_count += 1
        return result

    def invoke_filter_and_histogram(self, input_ids, candidate_mask, seq_dim=1):
        seq_len = input_ids.shape[seq_dim]
        bin_edges = torch.linspace(0, seq_len, self._num_bins + 1,
                                   device=self._device, dtype=torch.float32)

        if input_ids.dtype in (torch.long, torch.int):
            values = input_ids.float()
        else:
            values = input_ids

        flat = values.reshape(-1)
        if candidate_mask is not None:
            mask_flat = candidate_mask.reshape(-1)
            flat = flat[mask_flat]

        bin_indices = torch.bucketize(flat, bin_edges[1:-1])
        counts = torch.zeros(self._num_bins, device=self._device, dtype=torch.long)
        counts.scatter_add_(0, bin_indices.long(), torch.ones_like(bin_indices, dtype=torch.long))

        self._counter.filter_cnt = int(flat.numel())

        result = SequenceHistogram(
            bin_counts=counts,
            total_tokens=flat.numel(),
            num_bins=self._num_bins,
            pass_index=self._pass_count,
            timestamp_ns=time.time_ns(),
            rank=dist.get_rank() if dist.is_initialized() else 0,
        )
        self._pass_count += 1
        return result

    def finalize_pass(self, histogram, current_k, pass_idx, is_last_pass, counter_update_fn):
        global_hist = self.reduce_histograms(histogram)

        counter_update_fn(global_hist)

        bin_offsets = self._compute_bin_offsets(global_hist)

        bucket_idx = self._choose_bucket(bin_offsets, current_k)

        if not is_last_pass:
            self._counter.finished_block_cnt = 0

        return global_hist, bucket_idx, bin_offsets

    def _compute_bin_offsets(self, histogram):
        if histogram.bin_counts is None:
            return torch.zeros(self._num_bins, device=self._device, dtype=torch.long)
        return torch.cumsum(histogram.bin_counts, dim=0)

    def _choose_bucket(self, bin_offsets, current_k):
        if bin_offsets is None or current_k <= 0:
            return 0
        idx = torch.searchsorted(bin_offsets, torch.tensor(
            [current_k], device=self._device, dtype=bin_offsets.dtype))
        return int(idx.item())

    def reduce_histograms(self, local_hist):
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return local_hist

        global_counts = local_hist.bin_counts.clone()
        dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)

        return SequenceHistogram(
            bin_counts=global_counts,
            total_tokens=local_hist.total_tokens * dist.get_world_size(),
            num_bins=local_hist.num_bins,
            pass_index=local_hist.pass_index,
            timestamp_ns=time.time_ns(),
            rank=local_hist.rank,
        )

    def compute_optimal_chunk_ratios(self, histogram, sp_size):
        counts = histogram.bin_counts.float()
        total = counts.sum()
        if total == 0:
            return [1.0 / sp_size] * sp_size

        cumulative = torch.cumsum(counts, dim=0)
        target_per_rank = total / sp_size

        boundaries = []
        for i in range(1, sp_size):
            target = target_per_rank * i
            idx = torch.searchsorted(cumulative, target).item()
            boundaries.append(idx)

        ratios = []
        prev = 0
        for b in boundaries:
            segment = counts[prev:b].sum().item()
            ratios.append(segment / total.item())
            prev = b
        ratios.append(counts[prev:].sum().item() / total.item())

        min_ratio = 0.05
        ratios = [max(r, min_ratio) for r in ratios]
        ratio_sum = sum(ratios)
        ratios = [r / ratio_sum for r in ratios]

        return ratios

    def extract_candidate_bins(self, histogram, threshold_ratio=0.01):
        counts = histogram.bin_counts.float()
        total = counts.sum()
        if total == 0:
            return torch.arange(histogram.num_bins, device=self._device)

        ratios = counts / total
        mask = ratios >= threshold_ratio
        return torch.where(mask)[0]

    def early_stop_check(self, histogram, sp_size):
        counts = histogram.bin_counts
        nonzero_bins = (counts > 0).sum().item()
        return nonzero_bins <= sp_size

    def accumulate(self, histogram):
        if self._accumulator.global_histogram is None:
            self._accumulator.global_histogram = histogram.bin_counts.clone()
        else:
            self._accumulator.global_histogram += histogram.bin_counts
        self._accumulator.per_rank_histograms[histogram.rank] = histogram.bin_counts.clone()
        self._accumulator.num_passes += 1

    def get_accumulator(self):
        return self._accumulator

    def get_counter(self):
        return self._counter

    def reset(self):
        self._accumulator = HistogramAccumulator()
        self._counter = PassCounter()
        self._pass_count = 0


_GLOBAL_HISTOGRAM_KERNEL = None


def get_histogram_kernel(num_bins=256):
    global _GLOBAL_HISTOGRAM_KERNEL
    if _GLOBAL_HISTOGRAM_KERNEL is None:
        _GLOBAL_HISTOGRAM_KERNEL = SPHistogramKernel(num_bins=num_bins)
    return _GLOBAL_HISTOGRAM_KERNEL


def run_first_pass_histogram(input_ids, sp_size, seq_dim=1, is_last_pass=False):
    kernel = get_histogram_kernel()
    local_hist = kernel.invoke_histogram_only(input_ids, seq_dim=seq_dim)

    num_items = input_ids.numel()

    def _first_pass_counter_update(global_hist):
        kernel.accumulate(global_hist)
        kernel._counter.previous_len = num_items
        kernel._counter.filter_cnt = 0

    global_hist, bucket_idx, bin_offsets = kernel.finalize_pass(
        local_hist, sp_size, 0, is_last_pass, _first_pass_counter_update)

    early_stop = kernel.early_stop_check(global_hist, sp_size)
    if early_stop:
        kernel._counter.previous_len = 0
        kernel._counter.current_len = 0
        return [1.0 / sp_size] * sp_size, True

    ratios = kernel.compute_optimal_chunk_ratios(global_hist, sp_size)
    return ratios, False


def run_filter_and_histogram_pass(
    input_ids, candidate_mask, sp_size, pass_idx,
    is_last_pass=False, seq_dim=1,
):
    kernel = get_histogram_kernel()
    counter = kernel.get_counter()

    current_k = counter.k if counter.k > 0 else sp_size
    current_len = counter.current_len
    previous_len = counter.previous_len

    if current_len == 0:
        return [1.0 / sp_size] * sp_size, True

    early_stop = (current_len == current_k)

    local_hist = kernel.invoke_filter_and_histogram(
        input_ids, candidate_mask, seq_dim=seq_dim)

    def _filter_counter_update(global_hist):
        kernel.accumulate(global_hist)
        if early_stop:
            kernel._counter.previous_len = 0
            kernel._counter.current_len = 0
        else:
            kernel._counter.previous_len = current_len
            kernel._counter.filter_cnt = 0

    global_hist, bucket_idx, bin_offsets = kernel.finalize_pass(
        local_hist, current_k, pass_idx, is_last_pass, _filter_counter_update)

    if early_stop:
        return [1.0 / sp_size] * sp_size, True

    ratios = kernel.compute_optimal_chunk_ratios(global_hist, sp_size)
    return ratios, False
