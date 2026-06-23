# SPDX-License-Identifier: Apache-2.0
"""Prepare CommitPack dataset into tokenized numpy mmap .bin files.

Downloads bigcode/commitpack from HuggingFace in streaming mode,
tokenizes with a HuggingFace AutoTokenizer, and writes uint16 token ids
into a numpy memmap .bin file.

Usage:
    python data/prepare_commits.py \
        --output-dir data/commits_bin \
        --num-samples 100000 \
        --tokenizer-name gpt2

The output files are:
    <output-dir>/commits_train.bin   -- token ids (uint16, shape [N])
    <output-dir>/commits_train.idx   -- plain-text with total token count
"""

import argparse
import os
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# Lazy imports so the script fails early with a clear message if deps missing
# ---------------------------------------------------------------------------

def _import_deps():
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not found. Install via: pip install datasets", file=sys.stderr)
        sys.exit(1)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: 'transformers' package not found. Install via: pip install transformers", file=sys.stderr)
        sys.exit(1)

    return load_dataset, AutoTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_num(n: int) -> str:
    """Return a human-readable number string (e.g. 1_234_567)."""
    return f"{n:,}"


def _elapsed(start: float) -> str:
    secs = time.time() - start
    if secs < 60:
        return f"{secs:.1f}s"
    mins = secs / 60
    if mins < 60:
        return f"{mins:.1f}m"
    return f"{mins / 60:.1f}h"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare(
    output_dir: str,
    num_samples: int,
    tokenizer_name: str,
    split: str,
    languages: list,
    shard_size: int,
    text_field: str,
) -> None:
    load_dataset, AutoTokenizer = _import_deps()

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load tokenizer
    # ------------------------------------------------------------------
    print(f"[prepare_commits] Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
    eos_id = tokenizer.eos_token_id
    vocab_size = tokenizer.vocab_size
    print(f"[prepare_commits] Tokenizer vocab size: {_format_num(vocab_size)}, EOS id: {eos_id}")

    # uint16 can represent ids 0–65535; warn if vocab exceeds that
    if vocab_size > 65535:
        print(
            f"[prepare_commits] WARNING: vocab_size={vocab_size} > 65535. "
            "Tokens will be truncated to uint16. Consider using uint32.",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # Stream dataset
    # ------------------------------------------------------------------
    print(f"[prepare_commits] Streaming bigcode/commitpack (split={split}) ...")
    if languages:
        # CommitPack has a language sub-config; stream first language then concat
        # For simplicity, iterate over each language config sequentially
        datasets_iter = []
        for lang in languages:
            try:
                ds = load_dataset(
                    "bigcode/commitpack",
                    lang,
                    split=split,
                    streaming=True,
                    trust_remote_code=True,
                )
                datasets_iter.append(ds)
            except Exception as e:
                print(f"[prepare_commits] WARNING: could not load language '{lang}': {e}", file=sys.stderr)

        if not datasets_iter:
            print("ERROR: No valid language datasets loaded.", file=sys.stderr)
            sys.exit(1)

        # Chain them
        from itertools import chain
        dataset_stream = chain.from_iterable(datasets_iter)
    else:
        # Load the 'all' config (may be large)
        dataset_stream = load_dataset(
            "bigcode/commitpack",
            "all",
            split=split,
            streaming=True,
            trust_remote_code=True,
        )

    # ------------------------------------------------------------------
    # Prepare output mmap file
    # ------------------------------------------------------------------
    # We pre-allocate a large buffer; actual written length saved in .idx
    max_tokens = num_samples * 512  # rough upper bound (avg ~256 tokens/commit + eos)
    bin_path = os.path.join(output_dir, "commits_train.bin")
    idx_path = os.path.join(output_dir, "commits_train.idx")

    print(f"[prepare_commits] Pre-allocating mmap buffer: {_format_num(max_tokens)} tokens -> {bin_path}")
    mmap_arr = np.memmap(bin_path, dtype=np.uint16, mode="w+", shape=(max_tokens,))

    # ------------------------------------------------------------------
    # Tokenize and write
    # ------------------------------------------------------------------
    start_time = time.time()
    total_tokens = 0
    samples_done = 0
    log_every = max(1, num_samples // 100)  # log ~100 times

    for sample in dataset_stream:
        if samples_done >= num_samples:
            break

        text = sample.get(text_field, "")
        if not text:
            continue

        # Tokenize (no truncation; we want full commits)
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(eos_id)  # document separator

        token_count = len(ids)

        # Grow mmap if needed
        if total_tokens + token_count > len(mmap_arr):
            new_size = max(len(mmap_arr) * 2, total_tokens + token_count)
            print(
                f"[prepare_commits] Expanding mmap from {_format_num(len(mmap_arr))} "
                f"to {_format_num(new_size)} tokens ..."
            )
            mmap_arr.flush()
            del mmap_arr
            mmap_arr = np.memmap(bin_path, dtype=np.uint16, mode="r+", shape=(new_size,))

        # Clamp ids to uint16 range
        ids_np = np.array(ids, dtype=np.uint16)
        mmap_arr[total_tokens : total_tokens + token_count] = ids_np
        total_tokens += token_count
        samples_done += 1

        if samples_done % log_every == 0 or samples_done == num_samples:
            pct = 100.0 * samples_done / num_samples
            tps = total_tokens / max(1e-6, time.time() - start_time)
            print(
                f"[prepare_commits] {pct:5.1f}% | "
                f"samples={_format_num(samples_done)}/{_format_num(num_samples)} | "
                f"tokens={_format_num(total_tokens)} | "
                f"elapsed={_elapsed(start_time)} | "
                f"tok/s={_format_num(int(tps))}"
            )
            sys.stdout.flush()

    # ------------------------------------------------------------------
    # Flush and truncate to actual size
    # ------------------------------------------------------------------
    mmap_arr.flush()
    del mmap_arr

    # Truncate the file to the actual number of tokens written
    actual_size_bytes = total_tokens * np.dtype(np.uint16).itemsize
    with open(bin_path, "r+b") as f:
        f.truncate(actual_size_bytes)

    # Write index file with metadata
    with open(idx_path, "w") as f:
        f.write(f"total_tokens={total_tokens}\n")
        f.write(f"num_samples={samples_done}\n")
        f.write(f"tokenizer={tokenizer_name}\n")
        f.write(f"dtype=uint16\n")
        f.write(f"eos_id={eos_id}\n")

    elapsed = _elapsed(start_time)
    print()
    print(f"[prepare_commits] Done!")
    print(f"  samples   : {_format_num(samples_done)}")
    print(f"  tokens    : {_format_num(total_tokens)}")
    print(f"  bin file  : {bin_path}  ({actual_size_bytes / 1e9:.3f} GB)")
    print(f"  idx file  : {idx_path}")
    print(f"  elapsed   : {elapsed}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tokenize bigcode/commitpack into numpy mmap .bin files (uint16)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/commits_bin",
        help="Directory to write output .bin and .idx files (default: data/commits_bin).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100_000,
        help="Number of commit samples to process (default: 100000).",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="gpt2",
        help="HuggingFace tokenizer name or local path (default: gpt2).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train).",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="*",
        default=["python", "javascript"],
        help=(
            "CommitPack language sub-configs to load. "
            "Pass 'all' as a single value to load the full dataset. "
            "Default: python javascript."
        ),
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=10_000,
        help="(Reserved) Future shard size parameter (default: 10000).",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="message",
        help=(
            "Field in each dataset row to tokenize. "
            "CommitPack rows have: subject, message, diff, old_contents, new_contents. "
            "Default: message."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    languages = args.languages
    if languages == ["all"]:
        languages = []  # signals 'load all config'

    prepare(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        tokenizer_name=args.tokenizer_name,
        split=args.split,
        languages=languages,
        shard_size=args.shard_size,
        text_field=args.text_field,
    )
