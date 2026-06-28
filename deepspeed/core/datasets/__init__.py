# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Dataset classes for pretraining with heterogeneous batch sizes.

Three public classes are exposed:
    GPTDatasetConfig  — configuration dataclass
    GPTDataset        — single-source dataset (mmap / streaming)
    BlendedDataset    — weighted mixture of multiple GPTDataset instances

DES-LOC extension
-----------------
The data layer must support per-rank different ``micro_batch_size`` values
(heterogeneous batch).  This is achieved at the *DataLoader* level: every rank
calls ``GPTDataset.__getitem__`` with its own indices, and the batch assembly
happens locally.  Nothing inside these classes hard-codes a batch size.

Data formats supported
----------------------
*.bin   — flat array of ``int32`` token ids (Megatron-style, memory-mapped)
*.npy   — NumPy array file (memory-mapped via ``mmap_mode='r'``)
streaming (jsonl) — iterable via ``CommitPackerDataset`` in data/commit_loader.py
                    (GPTDataset delegates to that path when data_path ends in .jsonl)
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPTDatasetConfig
# ---------------------------------------------------------------------------

@dataclass
class GPTDatasetConfig:
    """Configuration for GPT pretraining dataset.

    Args:
        sequence_length: Target sequence length in tokens.
        split: One of ``"train"``, ``"valid"``, or ``"test"``.
        random_seed: Seed for shuffle reproducibility.
        fim_rate: Fill-in-the-Middle augmentation rate in ``[0, 1)``.
            When > 0, each sequence is independently transformed to the
            ``<PRE><SUF><MID>`` format used by StarCoder / CommitPack.
        ctx_loss_weight: Per-sample loss weight scalar for context tokens.
            Useful for down-weighting boilerplate context vs. the actual
            changed code in a commit message.  A weight of 1.0 is neutral.
        path_to_cache: Optional directory to cache pre-built index arrays.
            If None, indices are rebuilt from scratch every run (slower but
            avoids stale-cache bugs during development).
        eod_token_id: End-of-document token id used for FIM boundary marking
            and loss masking (default 0 — override to match your tokenizer).
        pad_token_id: Token id used for padding short sequences.
    """

    sequence_length: int = 2048
    split: str = "train"
    random_seed: int = 42
    fim_rate: float = 0.0        # Fill-in-the-Middle augmentation rate
    ctx_loss_weight: float = 1.0  # Loss weight for context tokens
    path_to_cache: Optional[str] = None
    eod_token_id: int = 0
    pad_token_id: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_sample_index(
    token_count: int,
    sequence_length: int,
    drop_last: bool = True,
) -> np.ndarray:
    """Return starting token offsets for each fixed-length sample.

    Each sample consumes ``sequence_length + 1`` tokens (the +1 gives us
    the final label token via a simple shift).  Samples are non-overlapping.

    Args:
        token_count: Total number of tokens in the backing array.
        sequence_length: Target sequence length.
        drop_last: If True, discard the tail fraction that does not fill
            a complete sample (default: True, matching Megatron behaviour).

    Returns:
        1-D int64 array of starting offsets, one per sample.
    """
    chunk = sequence_length + 1
    n_samples = token_count // chunk
    if not drop_last:
        n_samples = (token_count + chunk - 1) // chunk
    return np.arange(n_samples, dtype=np.int64) * chunk


def _build_shuffle_index(
    n_samples: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Return a shuffled permutation of ``[0, n_samples)``."""
    idx = np.arange(n_samples, dtype=np.int64)
    rng.shuffle(idx)
    return idx


def _apply_fim(
    tokens: np.ndarray,
    rng: np.random.RandomState,
    fim_rate: float,
    eod_id: int,
) -> np.ndarray:
    """Apply Fill-in-the-Middle transformation with probability ``fim_rate``.

    Transforms ``[prefix | suffix]`` into ``[<PRE> prefix <SUF> suffix <MID>]``
    following the StarCoder / SantaCoder convention.  The special token ids are
    synthesised as ``eod_id + {1,2,3}`` — callers should ensure their tokenizer
    has these reserved.

    When ``fim_rate == 0`` this is a no-op.
    """
    if fim_rate <= 0.0 or rng.random() >= fim_rate:
        return tokens

    PRE = eod_id + 1
    SUF = eod_id + 2
    MID = eod_id + 3

    n = len(tokens)
    if n < 4:
        return tokens

    # Choose a random split point (exclude 0 and n so both parts are non-empty)
    split = int(rng.randint(1, n))
    prefix = tokens[:split]
    suffix = tokens[split:]

    out = np.concatenate([[PRE], prefix, [SUF], suffix, [MID]])
    # Truncate / pad back to original length to maintain uniform shape
    if len(out) > n:
        out = out[:n]
    elif len(out) < n:
        out = np.concatenate([out, np.full(n - len(out), eod_id, dtype=out.dtype)])
    return out


# ---------------------------------------------------------------------------
# GPTDataset
# ---------------------------------------------------------------------------

class GPTDataset(Dataset):
    """GPT pretraining dataset with document-level packing.

    Supports two backends:
    * **mmap** — reads a flat binary file of ``int32`` or ``int64`` tokens
      via ``numpy.memmap`` (for ``.bin`` files) or ``numpy.load(...,
      mmap_mode='r')`` (for ``.npy`` files).  This is the primary path for
      production training on pre-tokenised CommitPack shards.
    * **streaming** — delegates to ``data/commit_loader.py``'s
      ``JsonlCommitDataset`` for ``.jsonl`` source files (useful during
      dataset construction / debugging).

    DES-LOC extension
    -----------------
    Each rank calls ``__getitem__`` independently with its local indices.
    The ``micro_batch_size`` can differ per rank without any changes here;
    the heterogeneity is handled by the engine-level sampler.

    Args:
        config: ``GPTDatasetConfig`` instance.
        data_path: Path to a ``.bin``, ``.npy``, or ``.jsonl`` file.
    """

    # ------------------------------------------------------------------ init

    def __init__(self, config: GPTDatasetConfig, data_path: str) -> None:
        self.config = config
        self.data_path = data_path
        self._rng = np.random.RandomState(config.random_seed)

        # --- resolve backend -------------------------------------------
        if data_path.endswith(".jsonl") or os.path.isdir(data_path):
            self._init_streaming(data_path)
        else:
            self._init_mmap(data_path)

    # ---------------------------------------------------------------- mmap path

    def _init_mmap(self, data_path: str) -> None:
        """Load and index a flat binary token file."""
        if data_path.endswith(".npy"):
            self._tokens: np.ndarray = np.load(data_path, mmap_mode="r")
        else:
            # Assume Megatron-style flat int32 binary
            self._tokens = np.memmap(data_path, dtype=np.int32, mode="r")

        self._streaming_sequences: Optional[List[np.ndarray]] = None

        token_count = len(self._tokens)
        drop_last = (self.config.split != "valid")
        offsets = _build_sample_index(token_count, self.config.sequence_length, drop_last)

        # Build / load shuffle index (with optional cache)
        self._shuffle_index = self._load_or_build_shuffle_index(offsets, data_path)
        self._offsets = offsets

        logger.info(
            "[GPTDataset] %s | split=%s | tokens=%d | samples=%d | fim_rate=%.2f",
            data_path, self.config.split, token_count,
            len(self._shuffle_index), self.config.fim_rate,
        )

    def _load_or_build_shuffle_index(
        self, offsets: np.ndarray, data_path: str
    ) -> np.ndarray:
        """Return shuffle index, loading from cache if available."""
        if self.config.path_to_cache is None:
            return _build_shuffle_index(len(offsets), self._rng)

        # Build a stable cache key from (data_path, config fields that affect ordering)
        key_str = (
            f"{os.path.abspath(data_path)}"
            f"|seq={self.config.sequence_length}"
            f"|split={self.config.split}"
            f"|seed={self.config.random_seed}"
        )
        cache_hash = hashlib.md5(key_str.encode(), usedforsecurity=False).hexdigest()[:16]
        cache_file = os.path.join(
            self.config.path_to_cache,
            f"gpt_shuffle_{cache_hash}.npy",
        )

        if os.path.isfile(cache_file):
            logger.debug("[GPTDataset] Loading cached shuffle index from %s", cache_file)
            return np.load(cache_file, mmap_mode="r")

        # Build and optionally save
        idx = _build_shuffle_index(len(offsets), self._rng)
        try:
            os.makedirs(self.config.path_to_cache, exist_ok=True)
            np.save(cache_file, idx)
            logger.debug("[GPTDataset] Saved shuffle index to %s", cache_file)
        except OSError as exc:
            logger.warning("[GPTDataset] Could not save shuffle index: %s", exc)
        return idx

    # --------------------------------------------------------------- streaming path

    def _init_streaming(self, data_path: str) -> None:
        """Eagerly materialise sequences from a .jsonl source.

        We call into ``data/commit_loader.py``'s ``JsonlCommitDataset``
        (the same iterator already used by the rest of Neuron_SP).  Because
        that class is an IterableDataset, we eagerly collect all sequences
        into a list so that ``__len__`` and random access both work.
        """
        from data.commit_loader import JsonlCommitDataset  # lazy import

        stream_ds = JsonlCommitDataset(
            data_path=data_path,
            seq_len=self.config.sequence_length,
        )
        seqs: List[np.ndarray] = []
        for inp, _lbl in stream_ds:
            seqs.append(inp.numpy().astype(np.int32))

        self._streaming_sequences = seqs
        self._tokens = None  # type: ignore[assignment]
        self._offsets = np.arange(len(seqs), dtype=np.int64)
        self._shuffle_index = _build_shuffle_index(len(seqs), self._rng)

        logger.info(
            "[GPTDataset] streaming %s | split=%s | sequences=%d",
            data_path, self.config.split, len(seqs),
        )

    # ---------------------------------------------------------------- Dataset API

    def __len__(self) -> int:
        """Number of samples in this dataset."""
        return len(self._shuffle_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single training sample.

        Args:
            idx: Logical index into the (shuffled) dataset.

        Returns:
            A dict with keys:
            * ``"tokens"``      — input token ids, shape ``(seq_len,)``
            * ``"labels"``      — target token ids (shifted by 1), shape ``(seq_len,)``
            * ``"loss_mask"``   — float mask (1.0 = compute loss), shape ``(seq_len,)``
            * ``"position_ids"``— position ids, shape ``(seq_len,)``
        """
        # Shuffle mapping
        raw_idx = int(self._shuffle_index[idx])

        # Retrieve raw token sequence of length seq_len + 1
        text = self._fetch_raw(raw_idx)

        # Optionally apply FIM augmentation (uses a per-sample RNG derived
        # from the global seed + idx so the result is deterministic/reproducible)
        if self.config.fim_rate > 0.0:
            sample_rng = np.random.RandomState(self.config.random_seed + raw_idx)
            text = _apply_fim(
                text, sample_rng, self.config.fim_rate, self.config.eod_token_id
            )

        # Standard causal LM shift
        tokens = torch.from_numpy(text[:-1].astype(np.int64))
        labels = torch.from_numpy(text[1:].astype(np.int64))

        seq_len = self.config.sequence_length
        # Pad or truncate to exactly seq_len (should already be exact for
        # mmap path; streaming path may occasionally be off by one)
        tokens  = _pad_or_trunc(tokens,  seq_len, self.config.pad_token_id)
        labels  = _pad_or_trunc(labels,  seq_len, self.config.pad_token_id)

        # Loss mask: 0.0 on pad positions, ctx_loss_weight elsewhere
        loss_mask = torch.ones(seq_len, dtype=torch.float32)
        loss_mask[labels == self.config.pad_token_id] = 0.0
        if self.config.ctx_loss_weight != 1.0:
            loss_mask = loss_mask * self.config.ctx_loss_weight

        position_ids = torch.arange(seq_len, dtype=torch.long)

        return {
            "tokens":       tokens,
            "labels":       labels,
            "loss_mask":    loss_mask,
            "position_ids": position_ids,
        }

    # ---------------------------------------------------------------- internals

    def _fetch_raw(self, raw_idx: int) -> np.ndarray:
        """Return seq_len+1 raw token ids for the given (pre-shuffle) index."""
        if self._streaming_sequences is not None:
            # Streaming path: each element is already seq_len tokens; append
            # a padding token to obtain seq_len+1 for the label shift.
            seq = self._streaming_sequences[raw_idx]
            if len(seq) < self.config.sequence_length + 1:
                seq = np.concatenate(
                    [seq, np.full(
                        self.config.sequence_length + 1 - len(seq),
                        self.config.pad_token_id, dtype=seq.dtype
                    )]
                )
            return seq[: self.config.sequence_length + 1]

        # mmap path
        # From Megatron M3247: cast to Python int before use as a slice index
        # to avoid numpy integer overflow on very large datasets (numpy uint32/int32
        # arithmetic can silently overflow when multiplied by seq_len > 2^15).
        offset = int(self._offsets[raw_idx])
        chunk = self.config.sequence_length + 1
        end = offset + chunk
        raw = self._tokens[offset:end]

        # Handle edge: last chunk may be short (only when drop_last=False)
        if len(raw) < chunk:
            raw = np.concatenate(
                [raw, np.full(chunk - len(raw), self.config.pad_token_id, dtype=raw.dtype)]
            )
        return raw.astype(np.int64)


def _pad_or_trunc(t: torch.Tensor, length: int, pad_id: int) -> torch.Tensor:
    """Pad or truncate a 1-D tensor to exactly ``length`` elements."""
    if t.shape[0] == length:
        return t
    if t.shape[0] > length:
        return t[:length]
    return torch.cat([t, torch.full((length - t.shape[0],), pad_id, dtype=t.dtype)])


# ---------------------------------------------------------------------------
# BlendedDataset
# ---------------------------------------------------------------------------

class BlendedDataset(Dataset):
    """Blended dataset that mixes multiple data sources with weights.

    Implements the same weighted-interleaving strategy as Megatron-LM's
    ``BlendedDataset`` but without the dependency on the C++ helpers extension.
    The blending index is built in pure NumPy and optionally cached to disk.

    The index arrays are:
    * ``_dataset_index``        — maps global sample id → which sub-dataset
    * ``_dataset_sample_index`` — maps global sample id → local sample id
                                  within that sub-dataset

    DES-LOC extension
    -----------------
    ``size`` controls the *total* number of samples returned by this dataset
    across all ranks.  Each rank draws its own slice via the DataLoader
    sampler; nothing inside ``BlendedDataset`` is rank-aware.

    Args:
        datasets: List of ``Dataset`` instances (typically ``GPTDataset``).
        weights: Relative sampling weights (positive floats or ints).
        size: Total number of samples the blended dataset should contain.
        config: ``GPTDatasetConfig`` — used for cache path and random seed.
    """

    def __init__(
        self,
        datasets: List[Dataset],
        weights: List[float],
        size: int,
        config: GPTDatasetConfig,
    ) -> None:
        if len(datasets) != len(weights):
            raise ValueError(
                f"len(datasets)={len(datasets)} != len(weights)={len(weights)}"
            )
        if len(datasets) == 0:
            raise ValueError("datasets list must not be empty")
        if any(w <= 0 for w in weights):
            raise ValueError("all weights must be strictly positive")
        if size <= 0:
            raise ValueError("size must be a positive integer")

        self.datasets = datasets
        self.weights = weights
        self.size = size
        self.config = config

        # Normalise weights so they sum to 1.0
        total_w = sum(weights)
        self._norm_weights = [w / total_w for w in weights]

        # Build (or load) the blending indices
        self._dataset_index, self._dataset_sample_index = self._build_indices()

    # ---------------------------------------------------------------- Dataset API

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return the blended sample at ``idx``.

        Looks up which sub-dataset and local sample id correspond to
        global ``idx``, then delegates to that sub-dataset.

        Returns:
            Dict with same keys as ``GPTDataset.__getitem__`` plus
            ``"dataset_id"`` (int tensor scalar).
        """
        dataset_id = int(self._dataset_index[idx])
        sample_id  = int(self._dataset_sample_index[idx])
        sample = self.datasets[dataset_id][sample_id]
        sample["dataset_id"] = torch.tensor(dataset_id, dtype=torch.long)
        return sample

    # ---------------------------------------------------------------- index building

    def _build_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build the dataset_index and dataset_sample_index arrays.

        The algorithm:
        1. Allocate one slot per sample (total = ``self.size``).
        2. Use a priority-queue approach (adapted from Megatron's C++ helper)
           implemented in pure NumPy: track a per-dataset "target" count
           based on normalised weights, and greedily assign each slot to the
           dataset that is most "behind" its target.

        Returns:
            Tuple of (dataset_index, dataset_sample_index), each of length
            ``self.size``.
        """
        cache_result = self._try_load_cache()
        if cache_result is not None:
            return cache_result

        n = self.size
        k = len(self.datasets)

        dataset_index        = np.zeros(n, dtype=np.int16)
        dataset_sample_index = np.zeros(n, dtype=np.int64)

        # Counters per dataset
        assigned = np.zeros(k, dtype=np.int64)
        # Cyclic per-dataset sample cursors (wrap around if oversampled)
        dataset_sizes = np.array([len(ds) for ds in self.datasets], dtype=np.int64)

        # Fractional "debt": how many more samples dataset i is owed.
        # Each step we award one sample to the dataset with the highest debt.
        debt = np.array(self._norm_weights, dtype=np.float64)  # initial budget share

        for i in range(n):
            # Choose the dataset with the highest accumulated debt
            chosen = int(np.argmax(debt))
            dataset_index[i] = chosen

            # Local sample index: wrap around to allow oversampling
            local_idx = int(assigned[chosen] % dataset_sizes[chosen])
            dataset_sample_index[i] = local_idx

            assigned[chosen] += 1
            # Reduce debt for chosen, increase for all (proportional to weight)
            debt -= self._norm_weights[chosen] * np.array(
                [1.0 if j == chosen else 0.0 for j in range(k)], dtype=np.float64
            )
            debt += np.array(self._norm_weights, dtype=np.float64)
            # Normalise to avoid unbounded growth (subtract the min)
            debt -= debt.min()

        self._try_save_cache(dataset_index, dataset_sample_index)
        return dataset_index, dataset_sample_index

    # ---------------------------------------------------------------- cache helpers

    def _cache_paths(self) -> Optional[Tuple[str, str]]:
        """Return (dataset_index_path, sample_index_path) or None if no cache dir."""
        if self.config.path_to_cache is None:
            return None
        key_str = (
            f"blend|n={self.size}|k={len(self.datasets)}"
            f"|w={self._norm_weights}|seed={self.config.random_seed}"
        )
        cache_hash = hashlib.md5(key_str.encode(), usedforsecurity=False).hexdigest()[:16]
        base = os.path.join(self.config.path_to_cache, f"blended_{cache_hash}")
        return f"{base}_dataset_index.npy", f"{base}_sample_index.npy"

    def _try_load_cache(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        paths = self._cache_paths()
        if paths is None:
            return None
        di_path, si_path = paths
        if os.path.isfile(di_path) and os.path.isfile(si_path):
            logger.debug("[BlendedDataset] Loading cached indices from %s", di_path)
            return (
                np.load(di_path, mmap_mode="r"),
                np.load(si_path, mmap_mode="r"),
            )
        return None

    def _try_save_cache(
        self,
        dataset_index: np.ndarray,
        dataset_sample_index: np.ndarray,
    ) -> None:
        paths = self._cache_paths()
        if paths is None:
            return
        di_path, si_path = paths
        try:
            os.makedirs(self.config.path_to_cache, exist_ok=True)
            np.save(di_path, dataset_index)
            np.save(si_path, dataset_sample_index)
            logger.debug("[BlendedDataset] Saved blending indices to %s", di_path)
        except OSError as exc:
            logger.warning("[BlendedDataset] Could not save blending indices: %s", exc)


__all__ = ["GPTDatasetConfig", "GPTDataset", "BlendedDataset"]
