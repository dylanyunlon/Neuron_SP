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
# .npy output with 99:1 train/valid split and CodeLlama tokenizer support
# ---------------------------------------------------------------------------

def prepare_npy(
    output_dir: str = "data",
    num_samples: int = 500_000,
    tokenizer_name: str = "codellama/CodeLlama-7b-hf",
    seq_len: int = 2048,
    languages: list = None,
    text_field: str = "message",
    train_ratio: float = 0.99,
) -> None:
    """Tokenize CommitPack and write memory-mapped .npy arrays for run_pretrain.py.

    Outputs:
        <output_dir>/commitpack_train.npy  -- int32 token ids, shape (N_train,)
        <output_dir>/commitpack_valid.npy  -- int32 token ids, shape (N_valid,)
        <output_dir>/commitpack_meta.json  -- total_tokens, split, tokenizer info

    The files can be loaded in run_pretrain.py via:
        tokens = np.load(path, mmap_mode='r')

    Args:
        output_dir:     Where to write the .npy files.
        num_samples:    Target number of commit samples to process.
        tokenizer_name: HuggingFace tokenizer; defaults to CodeLlama-7b-hf.
                        Falls back to 'gpt2' if the model is not accessible.
        seq_len:        Sequence length for packing; not used during tokenization
                        (run_pretrain.py handles packing at load time).
        languages:      CommitPack language sub-configs (default: python, javascript).
        text_field:     Dataset field to tokenize (default: message).
        train_ratio:    Fraction of tokens for train split (default: 0.99).
    """
    import json  # noqa: PLC0415
    load_dataset, AutoTokenizer = _import_deps()

    if languages is None:
        languages = ["python", "javascript"]

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Tokenizer — CodeLlama preferred; fall back to gpt2
    # ------------------------------------------------------------------
    print(f"[prepare_npy] Loading tokenizer: {tokenizer_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    except Exception as tok_exc:
        fallback = "gpt2"
        print(
            f"[prepare_npy] WARNING: could not load '{tokenizer_name}' ({tok_exc}). "
            f"Falling back to '{fallback}'.",
            file=sys.stderr,
        )
        tokenizer = AutoTokenizer.from_pretrained(fallback)
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
    eos_id = tokenizer.eos_token_id
    vocab_size = tokenizer.vocab_size
    print(f"[prepare_npy] vocab_size={_format_num(vocab_size)}, eos_id={eos_id}")

    # ------------------------------------------------------------------
    # Stream dataset
    # ------------------------------------------------------------------
    datasets_iter = []
    for lang in languages:
        try:
            ds = load_dataset(
                "bigcode/commitpack",
                lang,
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            datasets_iter.append(ds)
            print(f"[prepare_npy] Streaming commitpack/{lang}")
        except Exception as e:
            print(f"[prepare_npy] WARNING: could not load '{lang}': {e}", file=sys.stderr)

    if not datasets_iter:
        print("ERROR: No valid language datasets loaded.", file=sys.stderr)
        sys.exit(1)

    from itertools import chain  # noqa: PLC0415
    dataset_stream = chain.from_iterable(datasets_iter)

    # ------------------------------------------------------------------
    # Collect all tokens into a list, then split and save as .npy
    # ------------------------------------------------------------------
    all_tokens: list = []
    start_time = time.time()
    samples_done = 0
    log_every = max(1, num_samples // 100)

    for sample in dataset_stream:
        if samples_done >= num_samples:
            break
        text = sample.get(text_field, "")
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(eos_id)
        all_tokens.extend(ids)
        samples_done += 1
        if samples_done % log_every == 0 or samples_done == num_samples:
            pct = 100.0 * samples_done / num_samples
            tps = len(all_tokens) / max(1e-6, time.time() - start_time)
            print(
                f"[prepare_npy] {pct:5.1f}% | samples={_format_num(samples_done)} | "
                f"tokens={_format_num(len(all_tokens))} | "
                f"elapsed={_elapsed(start_time)} | tok/s={_format_num(int(tps))}"
            )
            sys.stdout.flush()

    total_tokens = len(all_tokens)
    print(f"[prepare_npy] Total tokens collected: {_format_num(total_tokens)}")

    if total_tokens < 1_000_000:
        print(
            f"[prepare_npy] WARNING: only {_format_num(total_tokens)} tokens; "
            "target is ≥1B tokens for meaningful pretraining.",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # 99:1 train/valid split
    # ------------------------------------------------------------------
    split_idx = int(total_tokens * train_ratio)
    train_tokens = np.array(all_tokens[:split_idx], dtype=np.int32)
    valid_tokens = np.array(all_tokens[split_idx:], dtype=np.int32)

    train_path = os.path.join(output_dir, "commitpack_train.npy")
    valid_path = os.path.join(output_dir, "commitpack_valid.npy")

    print(f"[prepare_npy] Saving train: {_format_num(len(train_tokens))} tokens → {train_path}")
    np.save(train_path, train_tokens)

    print(f"[prepare_npy] Saving valid: {_format_num(len(valid_tokens))} tokens → {valid_path}")
    np.save(valid_path, valid_tokens)

    # Metadata
    meta = {
        "total_tokens":  total_tokens,
        "train_tokens":  int(len(train_tokens)),
        "valid_tokens":  int(len(valid_tokens)),
        "train_ratio":   train_ratio,
        "num_samples":   samples_done,
        "tokenizer":     tokenizer_name,
        "eos_id":        int(eos_id),
        "vocab_size":    int(vocab_size),
        "seq_len":       seq_len,
        "dtype":         "int32",
        "format":        "npy",
    }
    meta_path = os.path.join(output_dir, "commitpack_meta.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[prepare_npy] Done!")
    print(f"  train  : {train_path}  ({os.path.getsize(train_path) / 1e9:.3f} GB)")
    print(f"  valid  : {valid_path}  ({os.path.getsize(valid_path) / 1e9:.3f} GB)")
    print(f"  meta   : {meta_path}")
    print(f"  elapsed: {_elapsed(start_time)}")
    print()
    print("Load in run_pretrain.py with:")
    print(f"  python run_pretrain.py --data-path {train_path}")

# ---------------------------------------------------------------------------
# CommitPackFT
# ---------------------------------------------------------------------------

def prepare_commitpackft(
    output_dir: str = "data/processed",
    tokenizer_name: str = "bigcode/starcoderbase",
    split: str = "train",
) -> None:
    """Tokenize bigcode/commitpackft (Python subset) into a numpy mmap .bin file.

    Each sample is formatted as:
        <commit_before> {old_contents} <commit_after> {new_contents} <commit_msg> {message}

    Tokens are written as uint16 to ``<output_dir>/commitpackft_python.bin``.

    Args:
        output_dir:      Directory to write the output .bin file.
        tokenizer_name:  HuggingFace tokenizer to use (default: bigcode/starcoderbase).
        split:           Dataset split (default: train).
    """
    load_dataset, AutoTokenizer = _import_deps()

    os.makedirs(output_dir, exist_ok=True)
    bin_path = os.path.join(output_dir, "commitpackft_python.bin")
    idx_path = os.path.join(output_dir, "commitpackft_python.idx")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    print(f"[prepare_commitpackft] Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
    eos_id = tokenizer.eos_token_id
    vocab_size = tokenizer.vocab_size
    print(f"[prepare_commitpackft] vocab_size={_format_num(vocab_size)}, eos_id={eos_id}")

    if vocab_size > 65535:
        print(
            f"[prepare_commitpackft] WARNING: vocab_size={vocab_size} > 65535. "
            "Tokens will be truncated to uint16.",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    print(f"[prepare_commitpackft] Loading bigcode/commitpackft ('python', split='{split}') ...")
    dataset = load_dataset("bigcode/commitpackft", "python", split=split, trust_remote_code=True)
    total_samples = len(dataset)
    print(f"[prepare_commitpackft] Dataset has {_format_num(total_samples)} samples.")

    # ------------------------------------------------------------------
    # Pre-allocate mmap (avg ~512 tokens/sample is a safe upper bound)
    # ------------------------------------------------------------------
    initial_capacity = total_samples * 512
    print(f"[prepare_commitpackft] Pre-allocating mmap: {_format_num(initial_capacity)} tokens -> {bin_path}")
    mmap_arr = np.memmap(bin_path, dtype=np.uint16, mode="w+", shape=(initial_capacity,))

    # ------------------------------------------------------------------
    # Tokenize and write
    # ------------------------------------------------------------------
    start_time = time.time()
    total_tokens = 0
    log_every = max(1, total_samples // 100)

    for i, sample in enumerate(dataset):
        old_contents = sample.get("old_contents") or ""
        new_contents = sample.get("new_contents") or ""
        message      = sample.get("message") or ""

        text = (
            f"<commit_before>{old_contents}"
            f"<commit_after>{new_contents}"
            f"<commit_msg>{message}"
        )

        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(eos_id)
        token_count = len(ids)

        # Grow mmap if needed
        if total_tokens + token_count > len(mmap_arr):
            new_size = max(len(mmap_arr) * 2, total_tokens + token_count)
            print(
                f"[prepare_commitpackft] Expanding mmap: "
                f"{_format_num(len(mmap_arr))} -> {_format_num(new_size)} tokens ..."
            )
            mmap_arr.flush()
            del mmap_arr
            mmap_arr = np.memmap(bin_path, dtype=np.uint16, mode="r+", shape=(new_size,))

        ids_np = np.array(ids, dtype=np.uint16)
        mmap_arr[total_tokens : total_tokens + token_count] = ids_np
        total_tokens += token_count

        samples_done = i + 1
        if samples_done % log_every == 0 or samples_done == total_samples:
            pct = 100.0 * samples_done / total_samples
            tps = total_tokens / max(1e-6, time.time() - start_time)
            print(
                f"[prepare_commitpackft] {pct:5.1f}% | "
                f"samples={_format_num(samples_done)}/{_format_num(total_samples)} | "
                f"tokens={_format_num(total_tokens)} | "
                f"elapsed={_elapsed(start_time)} | "
                f"tok/s={_format_num(int(tps))}"
            )
            sys.stdout.flush()

    # ------------------------------------------------------------------
    # Flush, truncate, write index
    # ------------------------------------------------------------------
    mmap_arr.flush()
    del mmap_arr

    actual_bytes = total_tokens * np.dtype(np.uint16).itemsize
    with open(bin_path, "r+b") as f:
        f.truncate(actual_bytes)

    with open(idx_path, "w") as f:
        f.write(f"total_tokens={total_tokens}\n")
        f.write(f"num_samples={total_samples}\n")
        f.write(f"tokenizer={tokenizer_name}\n")
        f.write(f"dtype=uint16\n")
        f.write(f"eos_id={eos_id}\n")
        f.write(f"format=<commit_before>old_contents<commit_after>new_contents<commit_msg>message\n")

    print()
    print("[prepare_commitpackft] Done!")
    print(f"  samples : {_format_num(total_samples)}")
    print(f"  tokens  : {_format_num(total_tokens)}")
    print(f"  bin     : {bin_path}  ({actual_bytes / 1e9:.3f} GB)")
    print(f"  idx     : {idx_path}")
    print(f"  elapsed : {_elapsed(start_time)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tokenize bigcode/commitpack or bigcode/commitpackft into numpy mmap .bin files (uint16)."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="commitpack",
        choices=["commitpack", "commitpackft", "npy"],
        help=(
            "Which preparation task to run. "
            "'commitpack' runs prepare() (uint16 .bin output). "
            "'commitpackft' runs prepare_commitpackft() (CommitPackFT, Python subset). "
            "'npy' runs prepare_npy() — CodeLlama tokenizer, int32, "
            "99:1 train/valid split, outputs commitpack_train.npy / commitpack_valid.npy. "
            "Default: commitpack."
        ),
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.99,
        help="Fraction of tokens for train split in 'npy' task (default: 0.99 = 99%%).",
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

    if args.task == "npy":
        languages = args.languages
        if languages == ["all"]:
            languages = []
        prepare_npy(
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            tokenizer_name=args.tokenizer_name or "codellama/CodeLlama-7b-hf",
            seq_len=2048,
            languages=languages if languages else ["python", "javascript"],
            text_field=args.text_field,
            train_ratio=getattr(args, "train_ratio", 0.99),
        )
    elif args.task == "commitpackft":
        prepare_commitpackft(
            output_dir=args.output_dir,
            tokenizer_name=args.tokenizer_name,
            split=args.split,
        )
    else:
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
