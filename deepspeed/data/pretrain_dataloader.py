"""
deepspeed/data/pretrain_dataloader.py

Production pretrain data pipeline with sequence packing for DES-LOC (Neuron_SP).
Supports JSONL, Parquet, and plain TXT formats with zero-padding-free packing.

COMMIT_MSG: production pretrain data pipeline with sequence packing for DES-LOC
"""

from __future__ import annotations

import io
import os
import glob
import logging
from pathlib import Path
from typing import Iterator, List, Optional, Union

import torch
from torch.utils.data import DataLoader, IterableDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SimpleTokenizer
# ---------------------------------------------------------------------------

class SimpleTokenizer:
    """
    Thin wrapper: tries tiktoken cl100k_base first, falls back to raw-bytes.
    Interface: encode(text) -> List[int], decode(ids) -> str, vocab_size -> int
    """

    def __init__(self) -> None:
        self._backend: str
        try:
            import tiktoken
            self._enc = tiktoken.get_encoding("cl100k_base")
            self._backend = "tiktoken"
            logger.info("[SimpleTokenizer] Using tiktoken cl100k_base.")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[SimpleTokenizer] tiktoken unavailable (%s); falling back to bytes.", exc
            )
            self._enc = None
            self._backend = "bytes"

    # ------------------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        if self._backend == "tiktoken":
            return self._enc.encode_ordinary(text)
        # Bytes fallback: each byte → token id in [0, 255]
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, ids: List[int]) -> str:
        if self._backend == "tiktoken":
            return self._enc.decode(ids)
        return bytes(i & 0xFF for i in ids).decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        if self._backend == "tiktoken":
            return self._enc.n_vocab
        return 256

    @property
    def eot_token(self) -> int:
        """End-of-text separator inserted between packed documents."""
        if self._backend == "tiktoken":
            return self._enc.eot_token  # 100257 for cl100k_base
        return 0  # '\x00' as EOT in bytes mode


# ---------------------------------------------------------------------------
# Raw-text iterators per file format
# ---------------------------------------------------------------------------

def _iter_jsonl(path: str) -> Iterator[str]:
    """Yield text strings from a JSONL file with {"text": "..."} records."""
    import json

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text", "")
                if text:
                    yield text
            except json.JSONDecodeError as exc:
                logger.debug("JSONL parse error at %s:%d – %s", path, lineno, exc)


def _iter_parquet(path: str) -> Iterator[str]:
    """Yield text strings from a Parquet file (expects a 'text' column)."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required to read Parquet files. "
            "Install it with: pip install pyarrow"
        ) from exc

    table = pq.read_table(path, columns=["text"])
    for batch in table.to_batches():
        col = batch.column("text")
        for val in col:
            text = val.as_py()
            if text:
                yield text


def _iter_txt(path: str) -> Iterator[str]:
    """Yield non-empty lines from a plain text file."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line:
                yield line


def _iter_file(path: str) -> Iterator[str]:
    """Dispatch to the correct iterator based on file extension."""
    ext = Path(path).suffix.lower()
    if ext in (".jsonl", ".json"):
        yield from _iter_jsonl(path)
    elif ext in (".parquet", ".pq"):
        yield from _iter_parquet(path)
    elif ext in (".txt", ".text", ""):
        yield from _iter_txt(path)
    else:
        logger.warning("Unknown extension '%s' for %s; treating as TXT.", ext, path)
        yield from _iter_txt(path)


def _resolve_paths(paths: List[str]) -> List[str]:
    """
    Expand glob patterns and directories into a flat list of file paths.
    Directories are searched one level deep for supported extensions.
    """
    supported_exts = {".jsonl", ".json", ".parquet", ".pq", ".txt", ".text"}
    resolved: List[str] = []
    for p in paths:
        expanded = glob.glob(p, recursive=True)
        if not expanded:
            expanded = [p]  # treat as literal even if glob found nothing
        for ep in expanded:
            ep_path = Path(ep)
            if ep_path.is_dir():
                for child in sorted(ep_path.iterdir()):
                    if child.suffix.lower() in supported_exts:
                        resolved.append(str(child))
            elif ep_path.exists():
                resolved.append(str(ep_path))
            else:
                logger.warning("Path does not exist and will be skipped: %s", ep)
    return resolved


# ---------------------------------------------------------------------------
# PackedPretrainDataset
# ---------------------------------------------------------------------------

class PackedPretrainDataset(IterableDataset):
    """
    IterableDataset that streams tokens from multiple files and packs them into
    fixed-length chunks of `max_seq_len` tokens with *zero padding*.

    Each yielded sample is a dict:
        {
            "input_ids":  LongTensor [max_seq_len],
            "labels":     LongTensor [max_seq_len],   # same as input_ids (CLM)
            "attention_mask": LongTensor [max_seq_len],  # all-ones (no padding)
        }

    Documents are separated by the tokenizer's EOT token.
    """

    def __init__(
        self,
        paths: List[str],
        tokenizer: SimpleTokenizer,
        max_seq_len: int = 2048,
        shuffle_files: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.paths = _resolve_paths(paths)
        if not self.paths:
            raise ValueError("PackedPretrainDataset received an empty file list.")
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.shuffle_files = shuffle_files
        self.seed = seed

        logger.info(
            "[PackedPretrainDataset] %d file(s), max_seq_len=%d, eot=%d",
            len(self.paths),
            max_seq_len,
            tokenizer.eot_token,
        )

    # ------------------------------------------------------------------
    def _token_stream(self) -> Iterator[int]:
        """
        Yield individual token IDs across all files.
        Appends an EOT token at the end of every document.
        Handles multi-worker sharding transparently.
        """
        worker_info = torch.utils.data.get_worker_info()
        file_list = list(self.paths)

        if self.shuffle_files:
            import random
            rng = random.Random(self.seed)
            rng.shuffle(file_list)

        # Shard files across DataLoader workers to avoid duplicate samples.
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            file_list = [f for i, f in enumerate(file_list) if i % num_workers == worker_id]
            logger.debug(
                "[Worker %d/%d] assigned %d files.", worker_id, num_workers, len(file_list)
            )

        eot = self.tokenizer.eot_token
        for path in file_list:
            try:
                for text in _iter_file(path):
                    ids = self.tokenizer.encode(text)
                    if ids:
                        yield from ids
                        yield eot
            except Exception as exc:  # noqa: BLE001
                logger.error("Error reading %s: %s", path, exc)

    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[dict]:
        buf: List[int] = []
        for token_id in self._token_stream():
            buf.append(token_id)
            if len(buf) == self.max_seq_len:
                yield self._make_sample(buf)
                buf = []
        # Drop the last incomplete chunk (zero-padding-free guarantee).

    # ------------------------------------------------------------------
    @staticmethod
    def _make_sample(ids: List[int]) -> dict:
        t = torch.tensor(ids, dtype=torch.long)
        return {
            "input_ids": t,
            "labels": t.clone(),
            "attention_mask": torch.ones_like(t),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_pretrain_dataloader(
    paths: List[str],
    tokenizer: Optional[SimpleTokenizer] = None,
    seq_len: int = 2048,
    batch_size: int = 4,
    num_workers: int = 2,
    shuffle_files: bool = False,
    seed: int = 42,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    """
    Factory that wires together tokenizer, dataset, and DataLoader.

    Args:
        paths:          List of file paths / glob patterns / directories.
        tokenizer:      Optional pre-built SimpleTokenizer; created if None.
        seq_len:        Token sequence length per sample (no padding).
        batch_size:     Samples per batch.
        num_workers:    DataLoader worker processes.
        shuffle_files:  Randomise file order each epoch.
        seed:           RNG seed for file shuffling.
        pin_memory:     Enable CUDA pinned memory.
        prefetch_factor: Batches prefetched per worker (ignored when workers=0).

    Returns:
        A configured DataLoader instance.
    """
    if tokenizer is None:
        tokenizer = SimpleTokenizer()

    dataset = PackedPretrainDataset(
        paths=paths,
        tokenizer=tokenizer,
        max_seq_len=seq_len,
        shuffle_files=shuffle_files,
        seed=seed,
    )

    loader_kwargs: dict = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,  # keeps batch sizes uniform (important for DeepSpeed)
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = True

    loader = DataLoader(dataset, **loader_kwargs)
    logger.info(
        "[create_pretrain_dataloader] batch=%d seq_len=%d workers=%d pin=%s",
        batch_size,
        seq_len,
        num_workers,
        loader_kwargs["pin_memory"],
    )
    return loader


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    # ── 1. Build temp corpus ──────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # JSONL
        jsonl_path = tmp / "corpus.jsonl"
        with jsonl_path.open("w") as f:
            for i in range(200):
                f.write(json.dumps({"text": f"Neuron_SP sample document number {i}. " * 8}) + "\n")

        # TXT
        txt_path = tmp / "corpus.txt"
        with txt_path.open("w") as f:
            for i in range(100):
                f.write(f"Plain text line {i}: The quick brown fox jumps over the lazy dog.\n")

        # Parquet (optional – skipped if pyarrow absent)
        parquet_path: Optional[Path] = None
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.table(
                {"text": [f"Parquet record {i}: {'token ' * 20}" for i in range(50)]}
            )
            parquet_path = tmp / "corpus.parquet"
            pq.write_table(table, str(parquet_path))
            logger.info("Parquet file written: %s", parquet_path)
        except ImportError:
            logger.warning("pyarrow not installed – skipping Parquet smoke test.")

        # ── 2. Tokenizer ──────────────────────────────────────────────────
        tok = SimpleTokenizer()
        logger.info("Backend: %s | vocab_size: %d", tok._backend, tok.vocab_size)

        sample_enc = tok.encode("Hello, Neuron_SP!")
        logger.info("Encode test → %s", sample_enc[:10])
        logger.info("Decode test → %r", tok.decode(sample_enc))

        # ── 3. Dataloader ─────────────────────────────────────────────────
        all_paths = [str(jsonl_path), str(txt_path)]
        if parquet_path:
            all_paths.append(str(parquet_path))

        SEQ_LEN = 128
        BATCH = 2

        loader = create_pretrain_dataloader(
            paths=all_paths,
            tokenizer=tok,
            seq_len=SEQ_LEN,
            batch_size=BATCH,
            num_workers=0,   # 0 = main process (safe in __main__)
            shuffle_files=True,
            seed=1337,
        )

        seen_batches = 0
        for batch in loader:
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attn_mask = batch["attention_mask"]

            assert input_ids.shape == (BATCH, SEQ_LEN), f"Shape mismatch: {input_ids.shape}"
            assert (attn_mask == 1).all(), "Expected all-ones attention mask (no padding)!"
            assert (input_ids == labels).all(), "input_ids and labels must match for CLM!"

            if seen_batches == 0:
                logger.info(
                    "First batch – input_ids[0,:16]: %s", input_ids[0, :16].tolist()
                )

            seen_batches += 1
            if seen_batches >= 5:
                break

        logger.info("✓ Smoke test passed: %d batches verified (shape=%s).",
                    seen_batches, (BATCH, SEQ_LEN))
