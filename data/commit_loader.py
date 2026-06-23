# SPDX-License-Identifier: Apache-2.0
"""Commit data loader — synthetic and real modes.

Usage:
    # Synthetic (for testing):
    loader = build_dataloader(mode="synthetic", batch_size=4, seq_len=2048)

    # Real from .jsonl:
    loader = build_dataloader(mode="real", data_path="data/commits.jsonl",
                               batch_size=4, seq_len=2048)
"""
import json
import os
from typing import Iterator, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


class SyntheticCommitDataset(Dataset):
    """Random token dataset for testing."""

    def __init__(self, vocab_size: int = 32000, seq_len: int = 2048,
                 num_samples: int = 100_000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        tokens = torch.randint(0, self.vocab_size, (self.seq_len + 1,))
        return tokens[:-1], tokens[1:]


class JsonlCommitDataset(IterableDataset):
    """Stream .jsonl files with {"text": "..."} format."""

    def __init__(self, data_path: str, tokenizer=None,
                 seq_len: int = 2048, vocab_size: int = 32000):
        self.data_path = data_path
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self._buffer = []

    def _tokenize(self, text: str):
        if self.tokenizer is not None:
            return self.tokenizer.encode(text, add_special_tokens=False)
        # Fallback: hash each char to a token id
        return [hash(c) % self.vocab_size for c in text]

    def __iter__(self):
        buffer = []
        paths = [self.data_path] if os.path.isfile(self.data_path) else \
            sorted(os.path.join(self.data_path, f)
                   for f in os.listdir(self.data_path) if f.endswith('.jsonl'))

        for path in paths:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = obj.get("text", "")
                    tokens = self._tokenize(text)
                    buffer.extend(tokens)

                    while len(buffer) >= self.seq_len + 1:
                        chunk = buffer[:self.seq_len + 1]
                        buffer = buffer[self.seq_len:]
                        t = torch.tensor(chunk, dtype=torch.long)
                        yield t[:-1], t[1:]


def build_dataloader(
    mode: str = "synthetic",
    data_path: Optional[str] = None,
    batch_size: int = 4,
    seq_len: int = 2048,
    vocab_size: int = 32000,
    num_workers: int = 2,
    tokenizer=None,
) -> DataLoader:
    if mode == "synthetic":
        ds = SyntheticCommitDataset(vocab_size=vocab_size, seq_len=seq_len)
        return DataLoader(ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True)
    elif mode == "real":
        assert data_path, "data_path required for real mode"
        ds = JsonlCommitDataset(data_path=data_path, tokenizer=tokenizer,
                                seq_len=seq_len, vocab_size=vocab_size)
        return DataLoader(ds, batch_size=batch_size,
                          num_workers=num_workers, pin_memory=True)
    else:
        raise ValueError(f"Unknown mode: {mode}")
