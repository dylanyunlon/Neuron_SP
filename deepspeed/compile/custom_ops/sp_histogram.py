import time
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import deepspeed.comm as dist


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


class SPHistogramKernel:

    def __init__(self, num_bins: int = 256, device: Optional[torch.device] = None):
        self._num_bins = num_bins
        self._device = device or (torch.device(f"cuda:{torch.cuda.current_device()}")
                                  if torch.cuda.is_available() else torch.device("cpu"))
        self._accumulator = HistogramAccumulator()
        self._pass_count = 0

    def compute_histogram(self, input_ids: torch.Tensor, seq_dim: int = 1) -> SequenceHistogram:
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

    def reduce_histograms(self, local_hist: SequenceHistogram) -> SequenceHistogram:
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

    def compute_optimal_chunk_ratios(self, histogram: SequenceHistogram,
                                     sp_size: int) -> List[float]:
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

    def extract_candidate_bins(self, histogram: SequenceHistogram,
                               threshold_ratio: float = 0.01) -> torch.Tensor:
        counts = histogram.bin_counts.float()
        total = counts.sum()
        if total == 0:
            return torch.arange(histogram.num_bins, device=self._device)

        ratios = counts / total
        mask = ratios >= threshold_ratio
        return torch.where(mask)[0]

    def early_stop_check(self, histogram: SequenceHistogram,
                         sp_size: int) -> bool:
        counts = histogram.bin_counts
        nonzero_bins = (counts > 0).sum().item()
        return nonzero_bins <= sp_size

    def accumulate(self, histogram: SequenceHistogram):
        if self._accumulator.global_histogram is None:
            self._accumulator.global_histogram = histogram.bin_counts.clone()
        else:
            self._accumulator.global_histogram += histogram.bin_counts
        self._accumulator.per_rank_histograms[histogram.rank] = histogram.bin_counts.clone()
        self._accumulator.num_passes += 1

    def get_accumulator(self) -> HistogramAccumulator:
        return self._accumulator

    def reset(self):
        self._accumulator = HistogramAccumulator()
        self._pass_count = 0


_GLOBAL_HISTOGRAM_KERNEL: Optional[SPHistogramKernel] = None


def get_histogram_kernel(num_bins: int = 256) -> SPHistogramKernel:
    global _GLOBAL_HISTOGRAM_KERNEL
    if _GLOBAL_HISTOGRAM_KERNEL is None:
        _GLOBAL_HISTOGRAM_KERNEL = SPHistogramKernel(num_bins=num_bins)
    return _GLOBAL_HISTOGRAM_KERNEL


def run_first_pass_histogram(input_ids: torch.Tensor, sp_size: int,
                             seq_dim: int = 1) -> Tuple[List[float], bool]:
    kernel = get_histogram_kernel()
    local_hist = kernel.compute_histogram(input_ids, seq_dim=seq_dim)
    global_hist = kernel.reduce_histograms(local_hist)
    kernel.accumulate(global_hist)

    early_stop = kernel.early_stop_check(global_hist, sp_size)
    if early_stop:
        return [1.0 / sp_size] * sp_size, True

    ratios = kernel.compute_optimal_chunk_ratios(global_hist, sp_size)
    return ratios, False
