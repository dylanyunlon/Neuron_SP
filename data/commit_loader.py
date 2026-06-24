# SPDX-License-Identifier: Apache-2.0
"""Commit data loader — synthetic, real, mmap, and commit-boundary-packed modes.

Usage:
    # Synthetic (for testing):
    loader = build_dataloader(mode="synthetic", batch_size=4, seq_len=2048)

    # Real from .jsonl (naive token-stream packing, may cross commit boundaries):
    loader = build_dataloader(mode="real", data_path="data/commits.jsonl",
                               batch_size=4, seq_len=2048)

    # Mmap — load real int32 token ids from a flat binary mmap file:
    #   Expects a flat binary file of int32 token ids (e.g. produced by
    #   Megatron-style preprocess_data.py).  Uses numpy.memmap for zero-copy
    #   memory-mapped access — the OS pages in only the data actually read,
    #   making this suitable for files larger than available RAM.
    loader = build_dataloader(mode="mmap", data_path="data/tokens.bin",
                               batch_size=4, seq_len=2048)

    # Packed — commit-boundary-aware packing via CommitSequencePacker:
    #   Short commits (≤256 tokens) are merged within the same sequence;
    #   long commits (>seq_len) are split with a sliding window.  No sequence
    #   ever straddles two commits.
    loader = build_dataloader(mode="packed", data_path="data/commits.jsonl",
                               batch_size=4, seq_len=2048, tokenizer=tok)
"""
import json
import os
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

# CommitSequencePacker lives in datasets/bigcode/commit_packing.py.
# Import lazily inside the classes/functions that need it so the module
# remains importable even if that path isn't on sys.path yet.
try:
    from datasets.bigcode.commit_packing import (
        CommitSequencePacker,
        compute_packing_stats,
    )
    _PACKER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PACKER_AVAILABLE = False


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
    """Stream .jsonl files with {"text": "..."} format.

    NOTE: this class concatenates token streams across commit boundaries.
    Use CommitPackerDataset (mode="packed") if you need commit-boundary
    guarantees.
    """

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



# ---------------------------------------------------------------------------
# Mmap-backed dataset: reads int32 token ids from a flat binary file
# ---------------------------------------------------------------------------

class MmapTokenDataset(Dataset):
    """Map-style dataset backed by a memory-mapped flat binary token file.

    The file is expected to contain a flat array of ``int32`` token ids, e.g.
    as produced by Megatron-style ``preprocess_data.py``.  Uses
    ``numpy.memmap`` so the OS pages in only the data actually read — suitable
    for files larger than available RAM.

    Each sample returns (input_ids, labels) of length ``seq_len``, where
    ``labels = input_ids`` shifted by one (standard causal LM convention).
    Samples are taken from non-overlapping windows of size ``seq_len + 1``
    starting at offset 0.

    Args:
        data_path : path to the flat binary ``.bin`` / ``.npy`` file of int32
                    token ids.
        seq_len   : sequence length in tokens (default 2048).
    """

    def __init__(self, data_path: str, seq_len: int = 2048) -> None:
        self.data_path = data_path
        self.seq_len = seq_len

        # Memory-map the file as read-only int32.  numpy.memmap does not load
        # the entire file into RAM; the OS maps virtual pages on demand.
        self._tokens: np.memmap = np.memmap(data_path, dtype=np.int32, mode="r")

        self._chunk = seq_len + 1  # tokens needed per sample (shifted by 1)
        n_total = len(self._tokens)
        self._n_samples = n_total // self._chunk
        if self._n_samples == 0:
            raise ValueError(
                f"MmapTokenDataset: file '{data_path}' contains {n_total} "
                f"int32 tokens but at least {self._chunk} are needed for a "
                f"single sample (seq_len={seq_len})."
            )

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self._chunk
        # Slice the memmap — copies only this window into host memory
        chunk = torch.from_numpy(
            self._tokens[start : start + self._chunk].astype(np.int64)
        )
        return chunk[:-1], chunk[1:]


# ---------------------------------------------------------------------------
# Commit-boundary-aware dataset (wraps CommitSequencePacker)
# ---------------------------------------------------------------------------

def _iter_jsonl_samples(data_path: str) -> Iterator[dict]:
    """Yield raw JSON objects from a .jsonl file or directory of .jsonl files."""
    paths = (
        [data_path]
        if os.path.isfile(data_path)
        else sorted(
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.endswith(".jsonl")
        )
    )
    for path in paths:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


class CommitPackerDataset(Dataset):
    """Map-style dataset built by packing commit samples with CommitSequencePacker.

    Each item in the dataset corresponds to one ``PackedSequence`` — a fixed-
    length token tensor whose contents never span more than one commit
    boundary.  Short commits (≤ SHORT_COMMIT_THRESHOLD tokens) may be merged
    into the same sequence; commits longer than ``seq_len`` are split via a
    sliding window (see ``CommitSequencePacker`` for full policy details).

    Args:
        data_path   : path to a single .jsonl file or a directory containing
                      .jsonl files.  Each line must be a JSON object with at
                      least one of the following fields, checked in order:
                        - ``"text"``            — pre-formatted commit text
                        - ``"old_contents"`` + ``"new_contents"`` + ``"subject"``
                          — raw commit fields (CommitSequencePacker will format)
        tokenizer   : HF-compatible tokenizer (must have ``__call__`` returning
                      ``{"input_ids": [...]}``).  If None, falls back to a
                      character-level approximation.
        seq_len     : target sequence length in tokens (default 2048).
        pad_token_id: token ID used for padding (default: tokenizer.eos_token_id
                      if available, else 0).
        max_sequences: optional cap on the number of packed sequences produced
                      (useful for smoke tests).
        verbose     : if True, print packing statistics after packing.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer=None,
        seq_len: int = 2048,
        pad_token_id: Optional[int] = None,
        max_sequences: Optional[int] = None,
        verbose: bool = True,
    ):
        if not _PACKER_AVAILABLE:
            raise ImportError(
                "CommitPackerDataset requires "
                "datasets.bigcode.commit_packing.CommitSequencePacker. "
                "Make sure the datasets/bigcode directory is on sys.path."
            )

        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None) or 0

        self.data_path = data_path
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id

        # Run the packer eagerly so __len__ is available for the DataLoader.
        packer = CommitSequencePacker(
            tokenizer=tokenizer,
            seq_len=seq_len,
            pad_token_id=pad_token_id,
        )
        self.sequences = packer.pack_dataset(
            _iter_jsonl_samples(data_path),
            max_sequences=max_sequences,
        )

        if verbose:
            stats = compute_packing_stats(self.sequences)
            print(f"[CommitPackerDataset] packed {len(self.sequences)} sequences "
                  f"from {data_path}")
            print(f"[CommitPackerDataset] packing stats: {stats}")
            if not stats.get("meets_5pct_target", True):
                print(
                    f"[CommitPackerDataset] WARNING: padding_ratio="
                    f"{stats['padding_ratio']:.3%} exceeds 5% target. "
                    f"Check commit length distribution."
                )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.tensor(self.sequences[idx].tokens, dtype=torch.long)
        # input_ids = tokens[:-1], labels = tokens[1:]  (standard LM shift)
        # Sequences are already seq_len long (padded), so we return the full
        # tensor as both input and label and let the training loop handle the
        # causal shift.
        return tokens, tokens


# ---------------------------------------------------------------------------
# build_dataloader — unified entry-point
# ---------------------------------------------------------------------------

def build_dataloader(
    mode: str = "synthetic",
    data_path: Optional[str] = None,
    batch_size: int = 4,
    seq_len: int = 2048,
    vocab_size: int = 32000,
    num_workers: int = 2,
    tokenizer=None,
    # packed-mode extras
    pad_token_id: Optional[int] = None,
    max_sequences: Optional[int] = None,
    verbose: bool = True,
) -> DataLoader:
    """Build a DataLoader for commit data.

    Args:
        mode        : one of ``"synthetic"``, ``"real"``, ``"mmap"``, or
                      ``"packed"``.

                      * ``"synthetic"`` — random token data, no tokenizer needed.
                      * ``"real"``      — naïve token-stream concatenation from
                        .jsonl; sequences **may cross commit boundaries**.
                      * ``"mmap"``      — memory-mapped flat binary file of
                        ``int32`` token ids (e.g. Megatron-style .bin files).
                        Uses :class:`MmapTokenDataset` / ``numpy.memmap`` for
                        zero-copy, RAM-efficient access.  Requires
                        ``data_path``.
                      * ``"packed"``    — commit-boundary-aware packing via
                        :class:`CommitPackerDataset` /
                        ``CommitSequencePacker``.  Requires ``data_path`` and
                        (optionally) a ``tokenizer``.

        data_path   : required for ``"real"``, ``"mmap"``, and ``"packed"``
                      modes.
        batch_size  : number of packed sequences per batch.
        seq_len     : target sequence length in tokens.
        vocab_size  : vocabulary size (used only in ``"synthetic"`` mode).
        num_workers : DataLoader worker processes.
        tokenizer   : HF-compatible tokenizer for ``"real"`` / ``"packed"`` modes.
        pad_token_id: padding token ID for ``"packed"`` mode (defaults to
                      ``tokenizer.eos_token_id`` or 0).
        max_sequences: cap on packed sequences for ``"packed"`` mode.
        verbose     : print packing stats for ``"packed"`` mode.
    """
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

    elif mode == "mmap":
        assert data_path, "data_path required for mmap mode"
        ds = MmapTokenDataset(data_path=data_path, seq_len=seq_len)
        return DataLoader(ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True)

    elif mode == "packed":
        assert data_path, "data_path required for packed mode"
        ds = CommitPackerDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            pad_token_id=pad_token_id,
            max_sequences=max_sequences,
            verbose=verbose,
        )

        def _collate(items: List[Tuple[torch.Tensor, torch.Tensor]]):
            input_ids = torch.stack([inp for inp, _ in items])
            labels = torch.stack([lbl for _, lbl in items])
            return {"input_ids": input_ids, "labels": labels}

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=_collate,
        )

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Choose 'synthetic', 'real', 'mmap', or 'packed'.")
