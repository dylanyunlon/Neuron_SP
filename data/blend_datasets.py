# SPDX-License-Identifier: Apache-2.0
"""Blend multiple mmap token datasets with weighted sampling.

Usage:
    from data.blend_datasets import build_blended_dataloader

    loader = build_blended_dataloader(
        sources=[
            {"path": "data/stack_v2.bin", "weight": 0.5},
            {"path": "data/commitpack.bin", "weight": 0.3},
            {"path": "data/commitpackft.bin", "weight": 0.2},
        ],
        batch_size=4,
        seq_len=2048,
    )
"""
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class MmapTokenDataset(Dataset):
    """Memory-mapped uint16 token file."""

    def __init__(self, path: str, seq_len: int = 2048, dtype=np.uint16):
        self.seq_len = seq_len
        self.tokens = np.memmap(path, dtype=dtype, mode="r")
        self.num_samples = max(0, len(self.tokens) // seq_len - 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start : start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


class BlendedDataset(Dataset):
    """Concatenates multiple MmapTokenDatasets."""

    def __init__(self, datasets: List[MmapTokenDataset], weights: List[float]):
        self.datasets = datasets
        self.weights = weights
        self.cum_sizes = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cum_sizes.append(total)
        self.total_size = total

        # Build per-sample weights for WeightedRandomSampler
        self._sample_weights = []
        for ds, w in zip(datasets, weights):
            n = len(ds)
            if n > 0:
                self._sample_weights.extend([w / n] * n)

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        for i, cum in enumerate(self.cum_sizes):
            if idx < cum:
                local_idx = idx - (self.cum_sizes[i - 1] if i > 0 else 0)
                return self.datasets[i][local_idx]
        raise IndexError(f"Index {idx} out of range")


def build_blended_dataloader(
    sources: List[Dict],
    batch_size: int = 4,
    seq_len: int = 2048,
    num_workers: int = 2,
    dtype=np.uint16,
) -> DataLoader:
    """Build a DataLoader that samples from multiple .bin files by weight.

    Args:
        sources: list of {"path": str, "weight": float}
        batch_size: micro batch size
        seq_len: sequence length
    """
    datasets = []
    weights = []
    for src in sources:
        path = src["path"]
        w = src.get("weight", 1.0)
        if not os.path.isfile(path):
            print(f"[blend] WARNING: {path} not found, skipping")
            continue
        ds = MmapTokenDataset(path, seq_len=seq_len, dtype=dtype)
        print(f"[blend] {path}: {len(ds)} samples, weight={w}")
        datasets.append(ds)
        weights.append(w)

    if not datasets:
        raise FileNotFoundError("No valid data sources found")

    blended = BlendedDataset(datasets, weights)
    sampler = WeightedRandomSampler(
        blended._sample_weights,
        num_samples=len(blended),
        replacement=True,
    )
    return DataLoader(
        blended,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
