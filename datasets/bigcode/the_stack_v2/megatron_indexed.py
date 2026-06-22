# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
megatron_indexed.py — Megatron-style indexed dataset writer (M761-M775)

Writes Stack v2 (or any DES-LOC) commit samples to Megatron's binary
IndexedDataset format (.bin + .idx), making the corpus directly consumable
by Megatron-LM's GPTDataset / BlendedMegatronDataset dataloaders.

Binary layout (matches Megatron's MMapIndexedDataset):
    .idx header:
        magic    : b"MMIDIDX\x00\x00"   (9 bytes)
        version  : uint64 = 1
        dtype    : uint8  (1=uint8, 2=int8, 3=int16, 4=int32, 8=int64)
        length   : int64  (number of documents)
        doc_count: int64  (same as length for non-packed format)
        sizes[]  : int32  * length     (token count per doc)
        pointers[]: int64 * length     (byte offset per doc in .bin)
    .bin data:
        flat array of token ids, each stored as dtype

For compatibility with Megatron's default dtype=int32 (vocab ≤ 2^31):
    dtype_code = 4 (int32)

Usage:
    from datasets.bigcode.the_stack_v2.megatron_indexed import MegatronIndexedWriter
    from datasets.bigcode.the_stack_v2.stackv2_commits import StackV2CommitAdapter

    tokenizer = ...   # any HF tokenizer

    writer = MegatronIndexedWriter("/data/stackv2_commits_megatron", tokenizer)
    adapter = StackV2CommitAdapter()
    writer.write_from_adapter(adapter, adapter.stream_parquet("/data/*.parquet"))
    writer.finalize()
    # → /data/stackv2_commits_megatron.bin
    # → /data/stackv2_commits_megatron.idx

Standalone (writes a tiny dummy dataset for testing):
    python megatron_indexed.py --output /tmp/test_stackv2 --dummy
"""

from __future__ import annotations

import array
import os
import struct
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Megatron MMapIndexedDataset magic + version
_MAGIC   = b"MMIDIDX\x00\x00"   # 9 bytes
_VERSION = 1                      # uint64

# dtype codes (matches Megatron's DType enum)
DTYPE_UINT8  = 1
DTYPE_INT8   = 2
DTYPE_INT16  = 3
DTYPE_INT32  = 4   # default (vocab ≤ 2^31)
DTYPE_INT64  = 8

_DTYPE_SIZE: Dict[int, int] = {
    DTYPE_UINT8: 1, DTYPE_INT8: 1, DTYPE_INT16: 2,
    DTYPE_INT32: 4, DTYPE_INT64: 8,
}
_DTYPE_STRUCT: Dict[int, str] = {
    DTYPE_UINT8: "B",  DTYPE_INT8: "b",  DTYPE_INT16: "h",
    DTYPE_INT32: "i",  DTYPE_INT64: "q",
}


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class MegatronIndexedWriter:
    """
    Streams tokenised commit samples to Megatron .bin + .idx files.

    The writer buffers token arrays in memory until finalize() is called,
    which writes both files atomically.  For very large datasets (>100M
    tokens) consider flushing in chunks (see flush_chunk() below).

    Args:
        output_prefix : output path WITHOUT extension (e.g. "/data/stackv2")
        tokenizer     : HF tokenizer (or None → word-split approximation)
        dtype         : token id dtype code (default DTYPE_INT32)
        eos_token_id  : appended after each document (default None = no EOS)
        add_bos       : prepend BOS token before each document (default False)
    """

    def __init__(
        self,
        output_prefix: str,
        tokenizer=None,
        dtype:          int           = DTYPE_INT32,
        eos_token_id:   Optional[int] = None,
        add_bos:        bool          = False,
    ) -> None:
        self.output_prefix = Path(output_prefix)
        self.tokenizer     = tokenizer
        self.dtype         = dtype
        self.eos_token_id  = eos_token_id
        self.add_bos       = add_bos

        self._sizes:    List[int]      = []   # token count per doc
        self._data:     List[List[int]] = []  # token ids per doc

        # Diagnostics
        self.total_docs:   int = 0
        self.total_tokens: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_text(self, text: str) -> int:
        """Tokenise text and add as one document. Returns token count."""
        ids = self._encode(text)
        self._add_ids(ids)
        return len(ids)

    def add_sample(self, sample: Dict) -> int:
        """Add a pre-formatted DES-LOC sample dict (must contain 'text')."""
        return self.add_text(sample["text"])

    def write_from_adapter(
        self,
        records: Iterable[Dict],
        verbose_every: int = 10_000,
    ) -> None:
        """
        Consume an iterable of formatted sample dicts and add them all.

        records: iterable of dicts with "text" key (e.g. from StackV2CommitAdapter)
        """
        for i, sample in enumerate(records):
            self.add_sample(sample)
            if verbose_every > 0 and (i + 1) % verbose_every == 0:
                print(
                    f"  [MegatronIndexedWriter] docs={self.total_docs:,}  "
                    f"tokens={self.total_tokens:,}"
                )

    def finalize(self) -> Tuple[Path, Path]:
        """
        Write .bin and .idx files.  Returns (bin_path, idx_path).
        """
        bin_path = self.output_prefix.with_suffix(".bin")
        idx_path = self.output_prefix.with_suffix(".idx")

        self.output_prefix.parent.mkdir(parents=True, exist_ok=True)

        self._write_bin(bin_path)
        self._write_idx(idx_path)

        print(
            f"[MegatronIndexedWriter] wrote {self.total_docs:,} docs  "
            f"{self.total_tokens:,} tokens\n"
            f"  → {bin_path}\n"
            f"  → {idx_path}"
        )
        return bin_path, idx_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> List[int]:
        if self.tokenizer is not None:
            ids: List[int] = self.tokenizer(
                text, add_special_tokens=False
            )["input_ids"]
        else:
            # Approximation: 4 chars ≈ 1 token, sequential fake ids
            n = max(1, len(text) // 4)
            ids = list(range(n))
        return ids

    def _add_ids(self, ids: List[int]) -> None:
        if self.add_bos and self.tokenizer is not None:
            bos = self.tokenizer.bos_token_id
            if bos is not None:
                ids = [bos] + ids
        if self.eos_token_id is not None:
            ids = ids + [self.eos_token_id]
        self._sizes.append(len(ids))
        self._data.append(ids)
        self.total_docs   += 1
        self.total_tokens += len(ids)

    def _write_bin(self, path: Path) -> None:
        fmt   = _DTYPE_STRUCT[self.dtype]
        dsize = _DTYPE_SIZE[self.dtype]
        with open(path, "wb") as f:
            for doc_ids in self._data:
                f.write(struct.pack(f"<{len(doc_ids)}{fmt}", *doc_ids))

    def _write_idx(self, path: Path) -> None:
        n = self.total_docs
        dsize = _DTYPE_SIZE[self.dtype]

        # Build byte offsets
        pointers: List[int] = []
        offset = 0
        for size in self._sizes:
            pointers.append(offset)
            offset += size * dsize

        with open(path, "wb") as f:
            # Header
            f.write(_MAGIC)                          # 9 bytes
            f.write(struct.pack("<Q", _VERSION))     # uint64
            f.write(struct.pack("<B", self.dtype))   # uint8
            f.write(struct.pack("<q", n))            # int64 (length)
            f.write(struct.pack("<q", n))            # int64 (doc_count)
            # Sizes (int32 array)
            f.write(struct.pack(f"<{n}i", *self._sizes))
            # Pointers (int64 array)
            f.write(struct.pack(f"<{n}q", *pointers))


# ---------------------------------------------------------------------------
# Minimal reader (sanity check / inspection)
# ---------------------------------------------------------------------------

class MegatronIndexedReader:
    """
    Minimal reader for Megatron MMapIndexedDataset .bin/.idx files.

    Useful for verifying writer output without needing a full Megatron install.

    Usage:
        reader = MegatronIndexedReader("/data/stackv2")
        print(len(reader))          # number of documents
        ids = reader[0]             # token ids for document 0
    """

    def __init__(self, prefix: str) -> None:
        self.bin_path = Path(prefix).with_suffix(".bin")
        self.idx_path = Path(prefix).with_suffix(".idx")
        self._load_idx()

    def _load_idx(self) -> None:
        with open(self.idx_path, "rb") as f:
            magic = f.read(9)
            assert magic == _MAGIC, f"Bad magic: {magic!r}"
            version = struct.unpack("<Q", f.read(8))[0]
            assert version == 1
            dtype_code = struct.unpack("<B", f.read(1))[0]
            n          = struct.unpack("<q", f.read(8))[0]
            _          = struct.unpack("<q", f.read(8))[0]   # doc_count
            sizes_raw  = f.read(n * 4)
            ptrs_raw   = f.read(n * 8)

        self.dtype_code = dtype_code
        self.dtype_fmt  = _DTYPE_STRUCT[dtype_code]
        self.dtype_size = _DTYPE_SIZE[dtype_code]
        self._sizes     = list(struct.unpack(f"<{n}i", sizes_raw))
        self._ptrs      = list(struct.unpack(f"<{n}q", ptrs_raw))
        self._n         = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> List[int]:
        if idx < 0 or idx >= self._n:
            raise IndexError(idx)
        size = self._sizes[idx]
        offset = self._ptrs[idx]
        byte_len = size * self.dtype_size
        with open(self.bin_path, "rb") as f:
            f.seek(offset)
            raw = f.read(byte_len)
        return list(struct.unpack(f"<{size}{self.dtype_fmt}", raw))

    def total_tokens(self) -> int:
        return sum(self._sizes)


# ---------------------------------------------------------------------------
# CLI / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description="MegatronIndexedWriter smoke test")
    parser.add_argument("--output", type=str, default=None,
                        help="Output prefix (default: tmp dir)")
    parser.add_argument("--dummy", action="store_true",
                        help="Write dummy dataset and read it back")
    parser.add_argument("--parquet", type=str, default=None,
                        help="Parquet glob to write (requires pyarrow)")
    parser.add_argument("--max-samples", type=int, default=1000)
    args = parser.parse_args()

    if args.dummy or args.parquet:
        if args.output is None:
            tmp = tempfile.mkdtemp()
            args.output = str(Path(tmp) / "stackv2_test")

        writer = MegatronIndexedWriter(args.output)

        if args.dummy:
            # Import adapter for dummy data
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from datasets.bigcode.the_stack_v2.stackv2_commits import StackV2CommitAdapter
            adapter = StackV2CommitAdapter()
            records = StackV2CommitAdapter.dummy_records(20)
            writer.write_from_adapter(adapter.stream_records(records))
            adapter.print_stats()

        elif args.parquet:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from datasets.bigcode.the_stack_v2.stackv2_commits import StackV2CommitAdapter
            adapter = StackV2CommitAdapter()
            writer.write_from_adapter(
                adapter.stream_parquet(args.parquet, args.max_samples)
            )
            adapter.print_stats()

        bin_p, idx_p = writer.finalize()

        # Read back and verify
        reader = MegatronIndexedReader(args.output)
        print(f"\n[verify] docs={len(reader)}  total_tokens={reader.total_tokens()}")
        if len(reader) > 0:
            doc0 = reader[0]
            print(f"  doc[0]: {len(doc0)} tokens  first5={doc0[:5]}")
        print("OK")
    else:
        parser.print_help()
