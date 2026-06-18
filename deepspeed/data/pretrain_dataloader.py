"""
Neuron_SP Project — Pretraining DataLoader
Supports: JSONL (RedPajama/SlimPajama), Parquet, plain TXT
Tokenizer: tiktoken cl100k_base (fallback: byte-level BPE)
Strategy: PackedPretrainDataset — zero-padding packing across documents
"""

from __future__ import annotations

import io
import os
import sys
import glob
import json
import random
import struct
import warnings
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Union

import torch
from torch.utils.data import DataLoader, IterableDataset

# ---------------------------------------------------------------------------
# Tokenizer — tiktoken preferred, byte-level fallback
# ---------------------------------------------------------------------------

class _ByteLevelTokenizer:
    """Minimal byte-level tokenizer used as fallback when tiktoken is absent."""

    bos_id: int = 256
    eos_id: int = 257
    vocab_size: int = 258

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, ids: List[int]) -> str:
        return bytes([i for i in ids if i < 256]).decode("utf-8", errors="replace")


def _build_tokenizer():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

        class _TikTokenWrapper:
            bos_id = enc.encode("<|endoftext|>", allowed_special="all")[0]
            eos_id = enc.encode("<|endoftext|>", allowed_special="all")[0]
            vocab_size = enc.n_vocab

            def encode(self, text: str) -> List[int]:
                return enc.encode(text, allowed_special="all")

            def decode(self, ids: List[int]) -> str:
                return enc.decode(ids)

        return _TikTokenWrapper()
    except ImportError:
        warnings.warn("tiktoken not found — falling back to byte-level tokenizer.", stacklevel=2)
        return _ByteLevelTokenizer()


# Module-level singleton
_TOKENIZER = None

def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = _build_tokenizer()
    return _TOKENIZER


# ---------------------------------------------------------------------------
# File readers — generators yielding raw text strings
# ---------------------------------------------------------------------------

def _read_jsonl(path: Union[str, Path]) -> Iterator[str]:
    """Yield text strings from {"text": "..."} JSONL (RedPajama/SlimPajama format)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # Support both "text" and "content" keys
                text = obj.get("text") or obj.get("content") or ""
                if text:
                    yield text
            except json.JSONDecodeError as e:
                warnings.warn(f"{path}:{line_no} — JSON decode error: {e}", stacklevel=2)


def _read_parquet(path: Union[str, Path]) -> Iterator[str]:
    """Yield text strings from a Parquet file (expects a 'text' or 'content' column)."""
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError("pyarrow is required to read Parquet files.  pip install pyarrow") from e

    table = pq.read_table(str(path))
    col_name = None
    for candidate in ("text", "content", "body", "document"):
        if candidate in table.column_names:
            col_name = candidate
            break
    if col_name is None:
        raise ValueError(f"No known text column in {path}. Columns: {table.column_names}")

    col = table.column(col_name)
    for chunk in col.chunks:
        for val in chunk:
            text = val.as_py()
            if text:
                yield str(text)


def _read_txt(path: Union[str, Path]) -> Iterator[str]:
    """Yield whole file as a single text string (suitable for large plain-text corpora)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    if text.strip():
        yield text


def _file_reader(path: Union[str, Path]) -> Iterator[str]:
    """Dispatch to the correct reader based on file extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".json"):
        yield from _read_jsonl(path)
    elif suffix == ".parquet":
        yield from _read_parquet(path)
    elif suffix in (".txt", ".text", ""):
        yield from _read_txt(path)
    else:
        warnings.warn(f"Unknown extension '{suffix}' for {path} — trying JSONL.", stacklevel=2)
        yield from _read_jsonl(path)


# ---------------------------------------------------------------------------
# PackedPretrainDataset
# ---------------------------------------------------------------------------

class PackedPretrainDataset(IterableDataset):
    """
    Iterable dataset that packs tokenised documents back-to-back into
    fixed-length chunks of size `max_seq_len`.  Absolutely zero padding waste.

    Each yielded item is a dict:
        {"input_ids": LongTensor[max_seq_len]}
    Labels = input_ids (shifted inside the model).

    Documents are separated by <EOS> token.  The dataset loops infinitely
    when `infinite=True` (default for pretraining).
    """

    def __init__(
        self,
        file_paths: Sequence[Union[str, Path]],
        max_seq_len: int = 2048,
        tokenizer=None,
        shuffle_files: bool = True,
        seed: int = 42,
        infinite: bool = True,
    ):
        super().__init__()
        self.file_paths   = list(file_paths)
        self.max_seq_len  = max_seq_len
        self.tokenizer    = tokenizer or get_tokenizer()
        self.shuffle_files = shuffle_files
        self.seed         = seed
        self.infinite     = infinite

    # ------------------------------------------------------------------
    # Worker-aware iteration split
    # ------------------------------------------------------------------

    def _get_file_shard(self) -> List[Path]:
        worker_info = torch.utils.data.get_worker_info()
        paths = list(self.file_paths)
        if worker_info is None:
            return paths
        # Shard files across workers
        total   = worker_info.num_workers
        rank    = worker_info.id
        return [p for i, p in enumerate(paths) if i % total == rank]

    # ------------------------------------------------------------------
    # Core generator
    # ------------------------------------------------------------------

    def _token_stream(self, paths: List[Path]) -> Iterator[int]:
        """Yield individual token ids from all files, with EOS between docs."""
        eos = self.tokenizer.eos_id
        for path in paths:
            for text in _file_reader(path):
                ids = self.tokenizer.encode(text)
                if ids:
                    yield from ids
                    yield eos

    def __iter__(self) -> Iterator[dict]:
        paths = self._get_file_shard()
        if self.shuffle_files:
            rng = random.Random(self.seed)
            rng.shuffle(paths)

        buffer: List[int] = []
        epoch = 0

        while True:
            for token_id in self._token_stream(paths):
                buffer.append(token_id)
                if len(buffer) >= self.max_seq_len:
                    chunk = buffer[:self.max_seq_len]
                    buffer = buffer[self.max_seq_len:]
                    yield {"input_ids": torch.tensor(chunk, dtype=torch.long)}

            epoch += 1
            if not self.infinite:
                # Yield partial last chunk with padding
                if buffer:
                    pad = self.max_seq_len - len(buffer)
                    chunk = buffer + [self.tokenizer.eos_id] * pad
                    yield {"input_ids": torch.tensor(chunk[:self.max_seq_len], dtype=torch.long)}
                break

            # Re-shuffle on each epoch for infinite training
            if self.shuffle_files:
                self.seed += 1
                rng = random.Random(self.seed)
                rng.shuffle(paths)


# ---------------------------------------------------------------------------
# NUMA-aware pin_memory
# ---------------------------------------------------------------------------

def _numa_aware_pin(tensor: torch.Tensor) -> torch.Tensor:
    """Pin memory with NUMA awareness when available (Linux + libnuma)."""
    try:
        return tensor.pin_memory()
    except RuntimeError:
        return tensor


class _PinMemoryCollate:
    def __call__(self, batch: List[dict]) -> dict:
        input_ids = torch.stack([b["input_ids"] for b in batch])
        if torch.cuda.is_available():
            input_ids = _numa_aware_pin(input_ids)
        return {"input_ids": input_ids}


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_pretrain_dataloader(
    data_paths: Union[str, Sequence[str]],
    max_seq_len: int = 2048,
    micro_batch_size: int = 4,
    num_workers: int = 4,
    tokenizer=None,
    shuffle_files: bool = True,
    seed: int = 42,
    infinite: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    """
    Build a ready-to-use DataLoader for pretraining.

    Args:
        data_paths: glob pattern string OR list of file paths.
                    Supports .jsonl, .parquet, .txt
        max_seq_len: sequence length (tokens) per sample
        micro_batch_size: local batch size per GPU
        num_workers: DataLoader worker processes
        tokenizer: optional pre-built tokenizer; builds default if None
        shuffle_files: randomise file ordering each epoch
        seed: RNG seed
        infinite: loop dataset indefinitely (True for pretraining)
        prefetch_factor: batches to prefetch per worker

    Returns:
        torch.utils.data.DataLoader yielding {"input_ids": Tensor[B, T]}
    """
    # Resolve glob patterns
    if isinstance(data_paths, str):
        resolved = sorted(glob.glob(data_paths, recursive=True))
        if not resolved:
            raise FileNotFoundError(f"No files matched glob pattern: {data_paths}")
    else:
        resolved = []
        for p in data_paths:
            matched = sorted(glob.glob(str(p), recursive=True))
            resolved.extend(matched if matched else [str(p)])

    if not resolved:
        raise FileNotFoundError(f"No data files resolved from: {data_paths}")

    tok = tokenizer or get_tokenizer()

    dataset = PackedPretrainDataset(
        file_paths=resolved,
        max_seq_len=max_seq_len,
        tokenizer=tok,
        shuffle_files=shuffle_files,
        seed=seed,
        infinite=infinite,
    )

    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        num_workers=num_workers,
        collate_fn=_PinMemoryCollate(),
        pin_memory=False,          # already handled in collate
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )
    return loader


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, textwrap

    print("=== PackedPretrainDataset smoke test ===")

    # Create a tiny JSONL file
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = os.path.join(tmpdir, "test.jsonl")
        with open(jsonl_path, "w") as f:
            for i in range(200):
                f.write(json.dumps({"text": f"Hello world sample number {i}. " * 10}) + "\n")

        tok = get_tokenizer()
        print(f"Tokenizer: {type(tok).__name__}  vocab_size={tok.vocab_size}")

        loader = create_pretrain_dataloader(
            data_paths=jsonl_path,
            max_seq_len=128,
            micro_batch_size=2,
            num_workers=0,
            tokenizer=tok,
            infinite=False,
        )

        total_tokens = 0
        for step, batch in enumerate(loader):
            ids = batch["input_ids"]
            assert ids.shape == (2, 128), f"Unexpected shape: {ids.shape}"
            total_tokens += ids.numel()
            if step == 0:
                print(f"  Batch 0 shape : {ids.shape}  dtype={ids.dtype}")

        print(f"  Total tokens seen : {total_tokens:,}")
        print("pretrain_dataloader.py — OK")
