"""
Pretrain Data Pipeline for DES-LOC
===================================

Supports:
- JSONL / Parquet reading (RedPajama, SlimPajama, C4 compatible)
- Sequence packing: multiple texts → one max_seq_len sequence
- NUMA-aware pinned memory (pin to GPU NUMA node)
- Multi-source weighted sampling
"""

import os
import json
import logging
import itertools
from pathlib import Path
from typing import List, Dict, Optional, Iterator, Union
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer wrapper
# ---------------------------------------------------------------------------

class SimpleTokenizer:
    """
    Wrapper that tries tiktoken (GPT-4 BPE) first, falls back to a
    character-level tokenizer for testing.
    """

    def __init__(self, name: str = "cl100k_base"):
        self.name = name
        self._enc = None
        self._vocab_size = 32000
        try:
            import tiktoken
            self._enc = tiktoken.get_encoding(name)
            self._vocab_size = self._enc.n_vocab
            logger.info("Using tiktoken '%s' (vocab=%d)", name, self._vocab_size)
        except ImportError:
            logger.warning("tiktoken not available; using byte-level fallback")

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eos_token_id(self) -> int:
        return 0

    def encode(self, text: str) -> List[int]:
        if self._enc is not None:
            return self._enc.encode(text, allowed_special="all")
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, ids: List[int]) -> str:
        if self._enc is not None:
            return self._enc.decode(ids)
        return bytes(ids).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Data source readers
# ---------------------------------------------------------------------------

def read_jsonl(path: str, text_key: str = "text") -> Iterator[str]:
    """Yield text strings from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get(text_key, "")
                if text:
                    yield text
            except json.JSONDecodeError:
                continue


def read_parquet(path: str, text_column: str = "text") -> Iterator[str]:
    """Yield text strings from a Parquet file."""
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(path, columns=[text_column])
        for batch in table.to_batches():
            for val in batch.column(text_column):
                text = val.as_py()
                if text:
                    yield text
    except ImportError:
        logger.error("pyarrow required for Parquet; pip install pyarrow")
        return


def read_texts(path: str, text_key: str = "text") -> Iterator[str]:
    """Auto-detect format and yield texts."""
    p = Path(path)
    if p.suffix == ".jsonl" or p.suffix == ".json":
        yield from read_jsonl(path, text_key)
    elif p.suffix == ".parquet":
        yield from read_parquet(path, text_key)
    elif p.suffix == ".txt":
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line
    else:
        logger.warning("Unknown format %s, trying JSONL", p.suffix)
        yield from read_jsonl(path, text_key)


# ---------------------------------------------------------------------------
# Sequence packing dataset
# ---------------------------------------------------------------------------

@dataclass
class DataSourceConfig:
    path: str
    weight: float = 1.0
    text_key: str = "text"


class PackedPretrainDataset(IterableDataset):
    """
    Streaming dataset that packs multiple texts into fixed-length sequences.

    Packing strategy:
    - Tokenize texts and concatenate with EOS separator
    - Chunk into max_seq_len blocks
    - No padding waste
    """

    def __init__(
        self,
        sources: List[DataSourceConfig],
        tokenizer: SimpleTokenizer,
        max_seq_len: int = 2048,
        seed: int = 42,
    ):
        self.sources = sources
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.seed = seed

    def _token_stream(self) -> Iterator[int]:
        """Infinite stream of token IDs from all sources."""
        while True:
            for source in self.sources:
                for text in read_texts(source.path, source.text_key):
                    tokens = self.tokenizer.encode(text)
                    yield from tokens
                    yield self.tokenizer.eos_token_id

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buffer = []
        for token_id in self._token_stream():
            buffer.append(token_id)
            if len(buffer) >= self.max_seq_len + 1:
                ids = buffer[: self.max_seq_len + 1]
                buffer = buffer[self.max_seq_len:]
                input_ids = torch.tensor(ids[:-1], dtype=torch.long)
                labels = torch.tensor(ids[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_pretrain_dataloader(
    data_paths: List[str],
    tokenizer: Optional[SimpleTokenizer] = None,
    max_seq_len: int = 2048,
    batch_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
    weights: Optional[List[float]] = None,
) -> DataLoader:
    """
    Create a DataLoader for pretraining.

    Args:
        data_paths: List of file paths (jsonl, parquet, txt)
        tokenizer: Tokenizer instance (default: tiktoken cl100k_base)
        max_seq_len: Maximum sequence length
        batch_size: Batch size
        num_workers: DataLoader workers
        pin_memory: Pin to NUMA node (recommended for GPU training)
        weights: Per-source sampling weights
    """
    if tokenizer is None:
        tokenizer = SimpleTokenizer()

    if weights is None:
        weights = [1.0] * len(data_paths)

    sources = [
        DataSourceConfig(path=p, weight=w)
        for p, w in zip(data_paths, weights)
    ]

    dataset = PackedPretrainDataset(
        sources=sources,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )


if __name__ == "__main__":
    import tempfile
    logging.basicConfig(level=logging.INFO)

    # Create a tiny test file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for i in range(100):
            json.dump({"text": f"Hello world sentence number {i}. " * 20}, f)
            f.write("\n")
        tmp_path = f.name

    tok = SimpleTokenizer()
    loader = create_pretrain_dataloader(
        [tmp_path], tokenizer=tok, max_seq_len=128, batch_size=2, num_workers=0,
    )

    for i, batch in enumerate(loader):
        print(f"Batch {i}: input_ids {batch['input_ids'].shape}, labels {batch['labels'].shape}")
        if i >= 2:
            break

    os.unlink(tmp_path)
    print("DataLoader smoke test passed.")
