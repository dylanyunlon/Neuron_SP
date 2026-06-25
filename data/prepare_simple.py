"""Minimal data preparation — no external tokenizer needed.

Downloads CommitPack text, builds a simple byte-level tokenizer
(256 byte values + special tokens, mapped to vocab_size=32000 range),
outputs commitpack_train.npy for run_pretrain.py.

Usage:
    pip install datasets    # only dependency
    python data/prepare_simple.py --num-samples 200000
"""
import argparse
import os
import sys
import time
import numpy as np


def prepare(
    output_dir: str = "data",
    num_samples: int = 200_000,
    seq_len: int = 1024,
    languages: list = None,
    text_field: str = "message",
    train_ratio: float = 0.99,
):
    from datasets import load_dataset

    if languages is None:
        languages = ["python", "javascript"]

    os.makedirs(output_dir, exist_ok=True)

    # ---- Byte-level "tokenizer" — zero dependencies ----
    # Map each byte (0-255) to token IDs 1-256, 0 = padding, 257 = EOS
    # All within vocab_size=32000
    EOS_ID = 257

    def encode(text: str) -> list:
        """Encode text as byte values + EOS."""
        return [b + 1 for b in text.encode("utf-8", errors="replace")] + [EOS_ID]

    # ---- Stream and tokenize ----
    all_ids = []
    total_tokens = 0
    t0 = time.time()

    for lang in languages:
        print(f"[prepare] Loading commitpack/{lang} (streaming)...")
        try:
            ds = load_dataset(
                "bigcode/commitpack", lang,
                split="train", streaming=True,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"[prepare] WARNING: could not load {lang}: {e}", file=sys.stderr)
            continue

        count = 0
        for sample in ds:
            text = sample.get(text_field, "")
            if not text or len(text) < 10:
                continue
            ids = encode(text)
            all_ids.extend(ids)
            total_tokens += len(ids)
            count += 1
            if count >= num_samples // len(languages):
                break
            if count % 10000 == 0:
                elapsed = time.time() - t0
                print(f"  [{lang}] {count:,} samples, {total_tokens:,} tokens ({elapsed:.0f}s)")

    print(f"[prepare] Total: {total_tokens:,} tokens from {len(languages)} languages")

    # ---- Write .npy ----
    arr = np.array(all_ids, dtype=np.int32)
    split_idx = int(len(arr) * train_ratio)

    train_path = os.path.join(output_dir, "commitpack_train.npy")
    valid_path = os.path.join(output_dir, "commitpack_valid.npy")

    np.save(train_path, arr[:split_idx])
    np.save(valid_path, arr[split_idx:])

    train_mb = os.path.getsize(train_path) / (1024 * 1024)
    print(f"[prepare] Wrote {train_path} ({train_mb:.1f} MB, {split_idx:,} tokens)")
    print(f"[prepare] Wrote {valid_path} ({len(arr) - split_idx:,} tokens)")
    print(f"[prepare] Tokenizer: byte-level (0-255 → 1-256, EOS=257, vocab ⊂ 32000)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=200_000)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--languages", nargs="+", default=["python", "javascript"])
    args = parser.parse_args()
    prepare(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        languages=args.languages,
    )
