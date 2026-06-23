#!/usr/bin/env python3
# coding=utf-8
"""
count_tokens.py — Count total tokens across dataset sources.

Supported formats:
  1. Megatron indexed binary  (.bin + .idx pairs)
  2. JSONL                    (.jsonl, optionally gzipped)
  3. Hugging Face dataset     (local path or Hub repo, streaming)

Usage examples:
  # Binary (.idx auto-discovered alongside .bin)
  python tools/count_tokens.py --bin data/train_text_document.bin

  # Multiple .bin files
  python tools/count_tokens.py --bin data/shard0.bin data/shard1.bin

  # JSONL (field "input_ids" or specify with --field)
  python tools/count_tokens.py --jsonl data/train.jsonl --field input_ids

  # JSONL with plain text + tokenizer
  python tools/count_tokens.py --jsonl data/train.jsonl --field text --tokenizer gpt2

  # Hugging Face Hub (streaming)
  python tools/count_tokens.py --hf EleutherAI/pile --hf-split train \
      --field text --tokenizer gpt2

  # Mix sources
  python tools/count_tokens.py --bin data/a.bin --jsonl data/b.jsonl --field input_ids
"""

import argparse
import gzip
import json
import logging
import os
import struct
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Megatron indexed-binary helpers
# ─────────────────────────────────────────────────────────────────────────────

_IDX_HEADER = b"MMIDIDX\x00\x00"

_DTYPE_MAP = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float64,
    7: np.float32,
    8: np.uint16,
}


def _read_idx_sequence_lengths(idx_path: str) -> np.ndarray:
    """Parse a Megatron .idx file and return the per-sequence token counts."""
    with open(idx_path, "rb") as f:
        header = f.read(9)
        if header != _IDX_HEADER:
            raise ValueError(
                f"{idx_path}: unexpected header {header!r} "
                f"(expected {_IDX_HEADER!r})"
            )
        # version (8 bytes, little-endian uint64) — vestigial, always 1
        f.read(8)
        # dtype code (1 byte)
        (dtype_code,) = struct.unpack("<B", f.read(1))
        if dtype_code not in _DTYPE_MAP:
            raise ValueError(f"{idx_path}: unknown dtype code {dtype_code}")
        # sequence count and document count
        (seq_count,) = struct.unpack("<Q", f.read(8))
        (doc_count,) = struct.unpack("<Q", f.read(8))
        # sequence lengths are stored as int32
        lengths_bytes = f.read(seq_count * 4)
    lengths = np.frombuffer(lengths_bytes, dtype=np.int32)
    return lengths


def count_tokens_bin(bin_path: str) -> int:
    """Count tokens in a Megatron binary dataset (.bin + .idx)."""
    bin_path = str(bin_path)
    # Derive .idx path: strip trailing .bin if present and add .idx
    base = bin_path[:-4] if bin_path.endswith(".bin") else bin_path
    idx_path = base + ".idx"
    if not os.path.isfile(idx_path):
        raise FileNotFoundError(f"Index file not found: {idx_path}")

    log.info("Reading index: %s", idx_path)
    lengths = _read_idx_sequence_lengths(idx_path)
    total = int(lengths.sum())
    log.info(
        "  sequences=%d  tokens=%s",
        len(lengths),
        f"{total:,}",
    )
    return total


# ─────────────────────────────────────────────────────────────────────────────
# JSONL helpers
# ─────────────────────────────────────────────────────────────────────────────


def _open_jsonl(path: str):
    """Open a plain or gzip-compressed JSONL file."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def count_tokens_jsonl(
    jsonl_path: str,
    field: str = "input_ids",
    tokenizer=None,
    log_every: int = 500_000,
) -> int:
    """
    Count tokens in a JSONL file.

    If the target field contains a list (e.g. input_ids), its length is used.
    If it contains a string and a tokenizer is provided, the string is tokenized.
    """
    total = 0
    lines = 0
    log.info("Scanning JSONL: %s  (field=%r)", jsonl_path, field)
    with _open_jsonl(jsonl_path) as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                log.warning("Skipping malformed line %d: %s", lines + 1, e)
                continue

            value = obj.get(field)
            if value is None:
                continue

            if isinstance(value, list):
                total += len(value)
            elif isinstance(value, str):
                if tokenizer is None:
                    raise RuntimeError(
                        f"Field {field!r} is a string but no --tokenizer was given. "
                        "Pass --tokenizer <name-or-path> to tokenize on the fly."
                    )
                total += len(tokenizer.encode(value))
            else:
                log.warning(
                    "Line %d: field %r has unexpected type %s — skipping",
                    lines + 1,
                    field,
                    type(value).__name__,
                )

            lines += 1
            if lines % log_every == 0:
                log.info("  processed %s lines  tokens so far: %s", f"{lines:,}", f"{total:,}")

    log.info("  total lines=%s  tokens=%s", f"{lines:,}", f"{total:,}")
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Hugging Face streaming helper
# ─────────────────────────────────────────────────────────────────────────────


def count_tokens_hf(
    dataset_name: str,
    split: str = "train",
    field: str = "input_ids",
    tokenizer=None,
    hf_config: Optional[str] = None,
    log_every: int = 100_000,
) -> int:
    """
    Count tokens in a Hugging Face dataset using streaming (no full download).

    If the target field contains a list (e.g. input_ids), its length is used.
    If it contains a string, tokenizer must be provided.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for HF streaming. "
            "Install it with: pip install datasets"
        )

    log.info(
        "Streaming HF dataset: %s  config=%s  split=%s  field=%r",
        dataset_name,
        hf_config or "(default)",
        split,
        field,
    )
    load_kwargs = dict(streaming=True, split=split, trust_remote_code=True)
    if hf_config:
        load_kwargs["name"] = hf_config

    ds = load_dataset(dataset_name, **load_kwargs)

    total = 0
    count = 0
    for sample in ds:
        value = sample.get(field)
        if value is None:
            continue
        if isinstance(value, list):
            total += len(value)
        elif isinstance(value, str):
            if tokenizer is None:
                raise RuntimeError(
                    f"Field {field!r} is a string but no --tokenizer was given."
                )
            total += len(tokenizer.encode(value))
        count += 1
        if count % log_every == 0:
            log.info("  streamed %s samples  tokens so far: %s", f"{count:,}", f"{total:,}")

    log.info("  total samples=%s  tokens=%s", f"{count:,}", f"{total:,}")
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer loader
# ─────────────────────────────────────────────────────────────────────────────


def load_tokenizer(name_or_path: str):
    """Load a HuggingFace tokenizer, falling back to tiktoken for known names."""
    # Try transformers first
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
        log.info("Loaded tokenizer via transformers: %s", name_or_path)
        return tok
    except Exception:
        pass

    # Fall back to tiktoken (e.g. "gpt2", "cl100k_base")
    try:
        import tiktoken
        enc = tiktoken.get_encoding(name_or_path)
        # Wrap to provide a .encode() method returning a list
        class _TiktokenWrapper:
            def encode(self, text):
                return enc.encode(text)
        log.info("Loaded tokenizer via tiktoken: %s", name_or_path)
        return _TiktokenWrapper()
    except Exception:
        pass

    raise ValueError(
        f"Could not load tokenizer {name_or_path!r}. "
        "Install 'transformers' or 'tiktoken' and verify the name/path."
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Count total tokens in .bin/.idx, .jsonl, or HF streaming datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Sources
    p.add_argument(
        "--bin",
        metavar="PATH",
        nargs="+",
        default=[],
        help="One or more Megatron .bin file paths (paired .idx files are auto-discovered).",
    )
    p.add_argument(
        "--jsonl",
        metavar="PATH",
        nargs="+",
        default=[],
        help="One or more .jsonl (or .jsonl.gz) file paths.",
    )
    p.add_argument(
        "--hf",
        metavar="DATASET",
        default=None,
        help="Hugging Face dataset name or local path for streaming.",
    )

    # HF options
    p.add_argument(
        "--hf-split",
        metavar="SPLIT",
        default="train",
        help="Dataset split to stream (default: train).",
    )
    p.add_argument(
        "--hf-config",
        metavar="CONFIG",
        default=None,
        help="Dataset config/subset name (e.g. 'en' for Wikipedia).",
    )

    # Common options
    p.add_argument(
        "--field",
        metavar="FIELD",
        default="input_ids",
        help=(
            "JSON field name containing token IDs (list) or raw text (string). "
            "Default: input_ids"
        ),
    )
    p.add_argument(
        "--tokenizer",
        metavar="NAME_OR_PATH",
        default=None,
        help=(
            "Tokenizer name or path (transformers or tiktoken) used when the target "
            "field contains raw text instead of token IDs."
        ),
    )

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.bin and not args.jsonl and not args.hf:
        parser.error("Provide at least one of --bin, --jsonl, or --hf.")

    tokenizer = None
    if args.tokenizer:
        tokenizer = load_tokenizer(args.tokenizer)

    t0 = time.perf_counter()
    grand_total = 0
    results = []

    # ── Binary datasets ───────────────────────────────────────────────────
    for bin_path in args.bin:
        n = count_tokens_bin(bin_path)
        grand_total += n
        results.append((bin_path, n))

    # ── JSONL datasets ────────────────────────────────────────────────────
    for jsonl_path in args.jsonl:
        n = count_tokens_jsonl(jsonl_path, field=args.field, tokenizer=tokenizer)
        grand_total += n
        results.append((jsonl_path, n))

    # ── HF streaming ─────────────────────────────────────────────────────
    if args.hf:
        n = count_tokens_hf(
            args.hf,
            split=args.hf_split,
            field=args.field,
            tokenizer=tokenizer,
            hf_config=args.hf_config,
        )
        label = f"{args.hf} ({args.hf_split})"
        grand_total += n
        results.append((label, n))

    elapsed = time.perf_counter() - t0

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Token count summary")
    print("=" * 60)
    for src, n in results:
        print(f"  {src}")
        print(f"      {n:>20,} tokens")
    if len(results) > 1:
        print("-" * 60)
        print(f"  TOTAL  {grand_total:>20,} tokens")
    print("=" * 60)
    # Human-readable scale
    for unit, divisor in [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if grand_total >= divisor:
            print(f"  ≈ {grand_total / divisor:.2f} {unit} tokens")
            break
    print(f"  Elapsed: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
