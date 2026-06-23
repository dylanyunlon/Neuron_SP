# SPDX-License-Identifier: Apache-2.0
"""Blend multiple mmap token datasets with weighted sampling.

DES-LOC recommended blend ratios for commit-message pretraining:
    - CommitPackFT:              40%  (high quality, GPT-4 filtered)
    - StarCoder git-commits:     30%  (deduplicated)
    - CommitPack (streaming):    20%  (large volume, variable quality)
    - The Stack v2 PR/commit:    10%  (supplements PR merge style)

Usage:
    from data.blend_datasets import build_blended_dataloader

    # Use DES-LOC defaults:
    loader = build_blended_dataloader(batch_size=4, seq_len=2048)

    # Override via YAML config:
    loader = build_blended_dataloader(
        blend_config="configs/my_blend.yaml",
        batch_size=4,
        seq_len=2048,
    )

    # Or pass explicit sources (overrides defaults and YAML):
    loader = build_blended_dataloader(
        sources=[
            {"path": "data/commitpackft.bin", "weight": 0.4},
            {"path": "data/starcoder_commits.bin", "weight": 0.6},
        ],
        batch_size=4,
        seq_len=2048,
    )
"""
import os
from typing import Dict, List, Optional

import yaml

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# DES-LOC recommended blend ratios for commit-message pretraining.
# Weights must sum to 1.0; adjust paths to match local data layout.
DESLOC_DEFAULT_BLEND: List[Dict] = [
    {"path": "data/commitpackft.bin", "weight": 0.40, "tag": "CommitPackFT"},
    {"path": "data/starcoder_commits.bin", "weight": 0.30, "tag": "StarCoder-git-commits-cleaned"},
    {"path": "data/commitpack_stream.bin", "weight": 0.20, "tag": "CommitPack-streaming"},
    {"path": "data/stack_v2_pr.bin", "weight": 0.10, "tag": "TheStackV2-PR-commit"},
]


def _load_blend_config(config_path: str) -> List[Dict]:
    """Parse a YAML blend config into a sources list.

    Expected YAML schema::

        sources:
          - path: data/commitpackft.bin
            weight: 0.4
          - path: data/starcoder_commits.bin
            weight: 0.3
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "sources" not in cfg:
        raise ValueError(f"blend config {config_path} must contain a 'sources' key")
    sources = cfg["sources"]
    total_w = sum(s.get("weight", 1.0) for s in sources)
    if abs(total_w - 1.0) > 1e-6:
        print(f"[blend] WARNING: weights sum to {total_w:.4f}, renormalising to 1.0")
        for s in sources:
            s["weight"] = s.get("weight", 1.0) / total_w
    return sources


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
    sources: Optional[List[Dict]] = None,
    batch_size: int = 4,
    seq_len: int = 2048,
    num_workers: int = 2,
    dtype=np.uint16,
    blend_config: Optional[str] = None,
) -> DataLoader:
    """Build a DataLoader that samples from multiple .bin files by weight.

    Resolution order for sources:
        1. Explicit *sources* argument (highest priority)
        2. YAML file passed via *blend_config* (``--blend-config``)
        3. ``DESLOC_DEFAULT_BLEND`` (fallback)

    Args:
        sources: list of {"path": str, "weight": float}
        batch_size: micro batch size
        seq_len: sequence length
        num_workers: DataLoader worker count
        dtype: numpy dtype of the mmap token files
        blend_config: path to a YAML file that overrides default blend ratios
    """
    if sources is not None:
        resolved = sources
    elif blend_config is not None:
        resolved = _load_blend_config(blend_config)
        print(f"[blend] loaded config from {blend_config}")
    else:
        resolved = DESLOC_DEFAULT_BLEND
        print("[blend] using DES-LOC default blend ratios")

    datasets = []
    weights = []
    for src in resolved:
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
