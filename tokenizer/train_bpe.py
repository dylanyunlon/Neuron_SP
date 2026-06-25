# SPDX-License-Identifier: Apache-2.0
# Neuron-SP Team
"""
train_bpe.py — Train a ByteLevel BPE tokenizer on CommitPack (Python + JavaScript)

Vocab layout (32 000 total):
┌──────────────────────────────────────────────────────────────────┐
│  ByteLevel BPE  vocab_size = 31 744                              │
│    ids   0 –   255 : 256 raw-byte tokens (built-in to ByteLevel) │
│    ids 256 – 31743 : 31 488 learned BPE merge tokens             │
│  added_tokens                                                    │
│    ids 31744 – 31999 : 256 explicit byte_0 … byte_255 aliases    │
│    ids 32000 – 32008 :   9 commit special tokens                 │
│  TOTAL  31 744 + 256 = 32 000  (headline figure)                 │
│  get_vocab_size() ≈ 32 009     (including commit specials)       │
└──────────────────────────────────────────────────────────────────┘

Steps:
  1. Stream 500 K records from bigcode/commitpack (python + javascript).
  2. Train ByteLevelBPETokenizer with vocab_size=31 744.
  3. Inject 256 named byte-fallback tokens  <byte_0> … <byte_255>.
  4. Inject 9 commit special tokens (mirrors pipeline/unified_tokenizer.py).
  5. Save to tokenizer/neuron_sp_32k/tokenizer.json.

Usage:
    python tokenizer/train_bpe.py [--samples 500000] [--output tokenizer/neuron_sp_32k]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Iterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Commit special tokens — mirrors pipeline/unified_tokenizer.py _LEGACY_COMMIT_TOKENS
# + <pad> for padding support
# ---------------------------------------------------------------------------
COMMIT_SPECIAL_TOKENS: list[str] = [
    "<|diff_start|>",
    "<|diff_end|>",
    "<|old|>",
    "<|new|>",
    "<|commit_msg|>",
    "<|file_path|>",
    "<|lang|>",
    "<|endoftext|>",
    "<pad>",
]

# 256 explicit byte-fallback token aliases for deterministic byte coverage
BYTE_FALLBACK_TOKENS: list[str] = [f"<byte_{i}>" for i in range(256)]

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

_LANG_FILE_PATTERNS = {
    "python":     "hf://datasets/bigcode/commitpack/data/python/python-{shard:04d}.jsonl",
    "javascript": "hf://datasets/bigcode/commitpack/data/javascript/javascript-{shard:04d}.jsonl",
}

# Known shard counts (as of training time; fetched dynamically if huggingface_hub available)
_KNOWN_SHARD_COUNTS = {"python": 458, "javascript": 516}


def _resolve_data_files() -> list[str]:
    """Return hf:// paths for all python + javascript shards in bigcode/commitpack."""
    try:
        from huggingface_hub import list_repo_tree  # type: ignore[import]

        files: list[str] = []
        for lang, prefix in [
            ("python",     "data/python"),
            ("javascript", "data/javascript"),
        ]:
            entries = list(list_repo_tree(
                "bigcode/commitpack", repo_type="dataset", path_in_repo=prefix
            ))
            for entry in entries:
                if entry.path.endswith(".jsonl"):
                    files.append(f"hf://datasets/bigcode/commitpack/{entry.path}")
        log.info("Resolved %d shard files via huggingface_hub.", len(files))
        return files

    except Exception as exc:  # noqa: BLE001
        log.warning("huggingface_hub listing failed (%s); using hardcoded shard counts.", exc)

    # Fallback: build paths from known shard counts
    files = []
    for lang, n_shards in _KNOWN_SHARD_COUNTS.items():
        pattern = _LANG_FILE_PATTERNS[lang]
        for i in range(1, n_shards + 1):
            files.append(pattern.format(shard=i))
    log.info("Built %d shard file paths from hardcoded counts.", len(files))
    return files


def iter_commit_texts(num_samples: int, batch_log: int = 50_000) -> Iterator[str]:
    """Stream text from bigcode/commitpack (python + javascript).

    Each record yields up to three text chunks:
      - old_contents  (original file before commit)
      - new_contents  (modified file after commit)
      - message       (commit message / subject)

    This exposes the tokenizer to real code diffs and natural-language summaries.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError:
        log.error("datasets not installed — run:  pip install datasets")
        sys.exit(1)

    data_files = _resolve_data_files()
    log.info(
        "Loading bigcode/commitpack  langs=[python, javascript]  "
        "shards=%d  streaming=True  target=%d records …",
        len(data_files),
        num_samples,
    )

    ds = load_dataset(
        "json",
        data_files={"train": data_files},
        split="train",
        streaming=True,
    )

    count = 0
    t0 = time.monotonic()

    for record in ds:
        if count >= num_samples:
            break

        for field in ("old_contents", "new_contents", "message"):
            text: str = record.get(field, "") or ""
            if text.strip():
                yield text

        count += 1

        if count % batch_log == 0:
            elapsed = time.monotonic() - t0
            log.info(
                "  … streamed %d / %d records  (%.1f rec/s)",
                count, num_samples, count / max(elapsed, 1e-9),
            )

    elapsed = time.monotonic() - t0
    log.info(
        "Streaming complete: %d records in %.1f s  (%.1f rec/s)",
        count, elapsed, count / max(elapsed, 1e-9),
    )


# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def train_tokenizer(
    num_samples: int = 500_000,
    bpe_vocab_size: int = 31_744,
    output_dir: str = "tokenizer/neuron_sp_32k",
) -> str:
    """Train ByteLevelBPE, inject byte + commit tokens, save tokenizer.json.

    Returns:
        Absolute path to the written tokenizer.json.
    """
    try:
        from tokenizers import ByteLevelBPETokenizer  # type: ignore[import]
        from tokenizers import Tokenizer as _Tokenizer  # type: ignore[import]
    except ImportError:
        log.error("tokenizers not installed — run:  pip install tokenizers")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1.  Train base ByteLevel BPE  (vocab_size = 31 744)
    # -----------------------------------------------------------------------
    log.info(
        "Training ByteLevelBPETokenizer  bpe_vocab_size=%d  num_samples=%d",
        bpe_vocab_size, num_samples,
    )

    tokenizer = ByteLevelBPETokenizer(
        add_prefix_space=False,
        lowercase=False,
        unicode_normalizer=None,
    )

    # Reserve all special tokens upfront so BPE never splits them
    all_reserved = BYTE_FALLBACK_TOKENS + COMMIT_SPECIAL_TOKENS

    tokenizer.train_from_iterator(
        iterator=iter_commit_texts(num_samples),
        vocab_size=bpe_vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=all_reserved,
        length=num_samples * 3,  # tqdm hint: ~3 text chunks per record
    )

    base_vocab_size = tokenizer.get_vocab_size()
    log.info("BPE training complete.  model.vocab_size = %d", base_vocab_size)

    # -----------------------------------------------------------------------
    # 2.  Serialise to JSON for surgical token injection
    # -----------------------------------------------------------------------
    tmp_path = os.path.join(output_dir, "_tmp_base.json")
    tokenizer.save(tmp_path)

    with open(tmp_path, "r", encoding="utf-8") as fh:
        tok_data: dict = json.load(fh)

    vocab: dict[str, int] = tok_data["model"]["vocab"]
    added_tokens_list: list[dict] = tok_data.get("added_tokens", [])

    existing_added_contents: set[str] = {t["content"] for t in added_tokens_list}
    existing_vocab_tokens: set[str] = set(vocab.keys())

    next_id: int = max(vocab.values()) + 1

    def _make_special_entry(content: str, token_id: int) -> dict:
        return {
            "id": token_id,
            "content": content,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }

    # -----------------------------------------------------------------------
    # 3.  Inject 256 named byte-fallback tokens  <byte_0> … <byte_255>
    # -----------------------------------------------------------------------
    new_byte_entries: list[dict] = []
    for bname in BYTE_FALLBACK_TOKENS:
        if bname in existing_vocab_tokens or bname in existing_added_contents:
            continue
        entry = _make_special_entry(bname, next_id)
        new_byte_entries.append(entry)
        existing_added_contents.add(bname)
        next_id += 1

    log.info(
        "Injected %d byte-fallback tokens  (<byte_0> … <byte_255>)",
        len(new_byte_entries),
    )

    # -----------------------------------------------------------------------
    # 4.  Inject 9 commit special tokens
    # -----------------------------------------------------------------------
    new_special_entries: list[dict] = []
    for stoken in COMMIT_SPECIAL_TOKENS:
        if stoken in existing_vocab_tokens or stoken in existing_added_contents:
            continue
        entry = _make_special_entry(stoken, next_id)
        new_special_entries.append(entry)
        existing_added_contents.add(stoken)
        next_id += 1

    log.info(
        "Injected %d commit special tokens  (%s … %s)",
        len(new_special_entries),
        COMMIT_SPECIAL_TOKENS[0],
        COMMIT_SPECIAL_TOKENS[-1],
    )

    # Merge and re-sort added_tokens by id
    existing_ids: set[int] = {t["id"] for t in added_tokens_list}
    for entry in new_byte_entries + new_special_entries:
        if entry["id"] not in existing_ids:
            added_tokens_list.append(entry)
            existing_ids.add(entry["id"])
    added_tokens_list.sort(key=lambda x: x["id"])

    tok_data["added_tokens"] = added_tokens_list

    # -----------------------------------------------------------------------
    # 5.  Save final tokenizer.json
    # -----------------------------------------------------------------------
    out_path = os.path.join(output_dir, "tokenizer.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(tok_data, fh, ensure_ascii=False, indent=2)

    headline_vocab = base_vocab_size + len(new_byte_entries)
    log.info(
        "Saved → %s\n"
        "        BPE model vocab        = %d\n"
        "        byte aliases injected  = %d\n"
        "        commit specials added  = %d\n"
        "        headline 31744+256     = %d",
        out_path,
        base_vocab_size,
        len(new_byte_entries),
        len(new_special_entries),
        headline_vocab,
    )

    # Clean up temp file
    os.remove(tmp_path)

    # -----------------------------------------------------------------------
    # 6.  Sanity checks
    # -----------------------------------------------------------------------
    log.info("Running sanity checks …")
    loaded = _Tokenizer.from_file(out_path)

    total_size = loaded.get_vocab_size()
    log.info("  tokenizer.get_vocab_size() = %d", total_size)

    # Encode a typical Python snippet
    sample = "def hello():\n    return 'world'  # 42\n"
    enc = loaded.encode(sample)
    log.info("  encode test: '%s' → %d tokens  %s", sample[:30], len(enc.ids), enc.tokens[:10])

    ok_special = True
    for stoken in COMMIT_SPECIAL_TOKENS:
        sid = loaded.token_to_id(stoken)
        if sid is None:
            log.warning("  ✗ MISSING special token: %s", stoken)
            ok_special = False
        else:
            log.info("  ✓ %-22s → id %d", stoken, sid)

    ok_bytes = True
    for i in (0, 64, 127, 200, 255):
        bname = f"<byte_{i}>"
        bid = loaded.token_to_id(bname)
        if bid is None:
            log.warning("  ✗ MISSING byte token: %s", bname)
            ok_bytes = False
        else:
            log.info("  ✓ %-14s → id %d", bname, bid)

    if ok_special and ok_bytes:
        log.info("All special and byte tokens verified ✓")
    else:
        log.warning("Some tokens missing — inspect tokenizer.json carefully")

    abs_path = os.path.abspath(out_path)
    log.info("Done.  tokenizer.json: %s", abs_path)
    return abs_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train a ByteLevel BPE tokenizer on bigcode/commitpack "
            "(Python + JavaScript) and save to tokenizer/neuron_sp_32k/tokenizer.json."
        )
    )
    p.add_argument(
        "--samples",
        type=int,
        default=500_000,
        help="Number of commit records to stream (default: 500 000)",
    )
    p.add_argument(
        "--vocab-size",
        type=int,
        default=31_744,
        dest="vocab_size",
        help=(
            "BPE vocab_size (default: 31 744). "
            "256 byte-fallback tokens are appended after training, "
            "giving a 31 744 + 256 = 32 000 headline vocabulary."
        ),
    )
    p.add_argument(
        "--output",
        type=str,
        default="tokenizer/neuron_sp_32k",
        help="Output directory for tokenizer.json (default: tokenizer/neuron_sp_32k)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_tokenizer(
        num_samples=args.samples,
        bpe_vocab_size=args.vocab_size,
        output_dir=args.output,
    )
