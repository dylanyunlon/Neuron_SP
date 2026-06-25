# SPDX-License-Identifier: Apache-2.0
# DeepSpeed / Neuron-SP Team
"""
train_bpe.py — Train a ByteLevel BPE tokenizer on CommitPack (Python + JavaScript)

Vocab layout (32 000 total accessible tokens):
┌──────────────────────────────────────────────────────────────┐
│  ByteLevel BPE model.vocab  = 31 744 entries                 │
│    ├─ ids  0 – 255  : 256 built-in byte tokens (Ġ…)          │
│    └─ ids 256 – 31743 : 31 488 learned BPE merges            │
│  added_tokens (special / named byte aliases)                 │
│    ├─ ids 31744 – 31999 : 256 explicit <byte_0> … <byte_255> │
│    └─ ids 32000 – 32008 : 9  commit special tokens           │
│  TOTAL get_vocab_size()   ≈ 32 009                           │
│  "32 000" headline figure = 31 744 BPE + 256 byte aliases    │
└──────────────────────────────────────────────────────────────┘

Steps:
  1. Stream 500K records from bigcode/commitpackft (python + javascript).
  2. Train ByteLevelBPETokenizer with vocab_size=31 744.
  3. Inject 256 named byte-fallback tokens  <byte_0> … <byte_255>.
  4. Inject 9 commit special tokens from COMMIT_SPECIAL_TOKENS (incl. <pad>).
  5. Save to tokenizer/neuron_sp_32k/tokenizer.json.

Usage:
    python tokenizer/train_bpe.py [--samples 500000] [--output tokenizer/neuron_sp_32k]
"""

import argparse
import json
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Commit special tokens (mirrors pipeline/unified_tokenizer.py)
# ---------------------------------------------------------------------------
COMMIT_SPECIAL_TOKENS = [
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

# Byte-fallback token names  (explicit named aliases for deterministic coverage)
BYTE_FALLBACK_TOKENS = [f"<byte_{i}>" for i in range(256)]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def iter_commit_texts(num_samples: int, batch_log: int = 50_000):
    """Stream text from bigcode/commitpackft (python + javascript).

    Each commit record yields up to three text chunks:
      - old_contents  (original file content before the commit)
      - new_contents  (modified file content after the commit)
      - message       (commit message / subject line)

    This gives the tokenizer realistic exposure to Python/JS code diffs as
    well as natural-language commit summaries.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("datasets not installed; run:  pip install datasets")
        sys.exit(1)

    log.info(
        "Loading bigcode/commitpackft (python + javascript) — streaming mode …"
    )

    ds = load_dataset(
        "json",
        data_files={
            "train": [
                "hf://datasets/bigcode/commitpackft/data/python/data.jsonl",
                "hf://datasets/bigcode/commitpackft/data/javascript/data.jsonl",
            ]
        },
        split="train",
        streaming=True,
    )

    count = 0
    t0 = time.time()

    for record in ds:
        if count >= num_samples:
            break

        for field in ("old_contents", "new_contents", "message"):
            text = record.get(field, "")
            if isinstance(text, str) and text.strip():
                yield text

        count += 1

        if count % batch_log == 0:
            elapsed = time.time() - t0
            log.info(
                "  … streamed %d / %d records  (%.1f rec/s)",
                count,
                num_samples,
                count / max(elapsed, 1e-9),
            )

    elapsed = time.time() - t0
    log.info(
        "Streaming complete: %d records in %.1f s  (%.1f rec/s)",
        count,
        elapsed,
        count / max(elapsed, 1e-9),
    )


# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def train_tokenizer(
    num_samples: int = 500_000,
    bpe_vocab_size: int = 31_744,
    output_dir: str = "tokenizer/neuron_sp_32k",
) -> str:
    """Train ByteLevelBPE, inject byte + special tokens, save tokenizer.json.

    Vocab accounting
    ----------------
    ByteLevelBPETokenizer always allocates ids 0–255 for the 256 raw bytes,
    then fills ids 256 … (bpe_vocab_size – 1) with learned merge tokens.
    After training we append:
      - 256 explicitly-named byte-fallback tokens  <byte_0> … <byte_255>
        (ids bpe_vocab_size … bpe_vocab_size + 255)
      - 9 commit special tokens
        (ids bpe_vocab_size + 256 … bpe_vocab_size + 264)

    So final tokenizer.get_vocab_size() == bpe_vocab_size + 256 + 9.
    The headline "32 000" = bpe_vocab_size (31 744) + 256 byte aliases.

    Args:
        num_samples:   Number of commit records to stream.
        bpe_vocab_size: vocab_size parameter passed to ByteLevelBPETokenizer.
                        Default 31 744 gives the 31 744 + 256 = 32 000 total.
        output_dir:    Directory where tokenizer.json is written.

    Returns:
        Absolute path to the saved tokenizer.json.
    """
    try:
        from tokenizers import ByteLevelBPETokenizer
    except ImportError:
        log.error("tokenizers not installed; run:  pip install tokenizers")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1.  Train base ByteLevel BPE  (vocab_size = 31 744)
    # ------------------------------------------------------------------
    log.info(
        "Training ByteLevelBPETokenizer  bpe_vocab_size=%d  num_samples=%d",
        bpe_vocab_size,
        num_samples,
    )

    tokenizer = ByteLevelBPETokenizer(
        add_prefix_space=False,
        lowercase=False,
        unicode_normalizer=None,
    )

    # Reserve all special tokens so they are never split by the BPE algorithm
    all_reserved = BYTE_FALLBACK_TOKENS + COMMIT_SPECIAL_TOKENS

    tokenizer.train_from_iterator(
        iterator=iter_commit_texts(num_samples),
        vocab_size=bpe_vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=all_reserved,
        length=num_samples * 3,   # hint for tqdm progress bar
    )

    actual_base = tokenizer.get_vocab_size()
    log.info("BPE training complete.  model.vocab size = %d", actual_base)

    # ------------------------------------------------------------------
    # 2.  Serialise to JSON for direct manipulation of the token table
    # ------------------------------------------------------------------
    tmp_path = os.path.join(output_dir, "_tmp_base.json")
    tokenizer.save(tmp_path)

    with open(tmp_path, "r", encoding="utf-8") as fh:
        tok_data = json.load(fh)

    vocab: dict = tok_data["model"]["vocab"]

    # ------------------------------------------------------------------
    # 3.  Inject 256 named byte-fallback tokens  <byte_0> … <byte_255>
    # ------------------------------------------------------------------
    added_tokens_list: list = tok_data.get("added_tokens", [])
    existing_added_contents = {t["content"] for t in added_tokens_list}
    existing_vocab_tokens = set(vocab.keys())

    next_id = max(vocab.values()) + 1

    new_byte_entries: list = []
    for bname in BYTE_FALLBACK_TOKENS:
        # Skip if already present in model vocab or added_tokens
        if bname in existing_vocab_tokens or bname in existing_added_contents:
            continue
        entry = {
            "id": next_id,
            "content": bname,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        new_byte_entries.append(entry)
        existing_added_contents.add(bname)
        next_id += 1

    log.info(
        "Injected %d byte-fallback tokens  (<byte_0> … <byte_255>)",
        len(new_byte_entries),
    )

    # ------------------------------------------------------------------
    # 4.  Inject 9 commit special tokens
    # ------------------------------------------------------------------
    new_special_entries: list = []
    for stoken in COMMIT_SPECIAL_TOKENS:
        if stoken in existing_vocab_tokens or stoken in existing_added_contents:
            continue
        entry = {
            "id": next_id,
            "content": stoken,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        new_special_entries.append(entry)
        existing_added_contents.add(stoken)
        next_id += 1

    log.info("Injected %d commit special tokens", len(new_special_entries))

    # Merge new entries into added_tokens (sorted by id)
    existing_ids = {t["id"] for t in added_tokens_list}
    for entry in new_byte_entries + new_special_entries:
        if entry["id"] not in existing_ids:
            added_tokens_list.append(entry)
            existing_ids.add(entry["id"])
    added_tokens_list.sort(key=lambda x: x["id"])

    tok_data["added_tokens"] = added_tokens_list

    # ------------------------------------------------------------------
    # 5.  Save final tokenizer.json
    # ------------------------------------------------------------------
    out_path = os.path.join(output_dir, "tokenizer.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(tok_data, fh, ensure_ascii=False, indent=2)

    # Compute headline vocab size: BPE model vocab + the 256 byte aliases
    byte_aliases_added = len(new_byte_entries)
    # If byte aliases were already in vocab (= 0 new), they are in the base vocab
    headline_vocab = actual_base + byte_aliases_added
    log.info(
        "Saved tokenizer.json → %s  |  "
        "BPE model vocab = %d  |  "
        "byte aliases added = %d  |  "
        "commit specials added = %d  |  "
        "headline 31744+256 = %d",
        out_path,
        actual_base,
        byte_aliases_added,
        len(new_special_entries),
        headline_vocab,
    )

    # Clean up temporary file
    os.remove(tmp_path)

    # ------------------------------------------------------------------
    # 6.  Sanity checks
    # ------------------------------------------------------------------
    log.info("Running sanity checks …")
    from tokenizers import Tokenizer as _Tokenizer
    loaded = _Tokenizer.from_file(out_path)

    total_size = loaded.get_vocab_size()
    log.info("  tokenizer.get_vocab_size() = %d", total_size)

    sample = "def hello():\n    return 'world'  # 42"
    enc = loaded.encode(sample)
    log.info("  encode test: %d tokens → %s", len(enc.ids), enc.tokens[:12])

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
        log.warning("Some tokens missing — inspect tokenizer.json")

    log.info("Done.  tokenizer.json: %s", os.path.abspath(out_path))
    return os.path.abspath(out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Train a ByteLevel BPE tokenizer on CommitPack (Python + JS) "
            "and save to tokenizer/neuron_sp_32k/tokenizer.json"
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
        help=(
            "BPE vocab_size parameter (default: 31 744). "
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
    args = parse_args()
    train_tokenizer(
        num_samples=args.samples,
        bpe_vocab_size=args.vocab_size,
        output_dir=args.output,
    )
