"""Prepare CommitPack with GLM-130B 70/30 mixed objective.

Downloads bigcode/commitpack, applies:
  - 70% span corruption: mask diff spans → model predicts masked content
  - 30% causal LM: commit_before → commit_after (natural causal structure)

Output: data/commitpack_train.npy (int32 token IDs, byte-level encoding)

References:
  - GLM-130B §3 (mixed training objective)
  - Commit c0f4e819 (MixedCommitDataset)
  - Commit 16405f8b (3-stage pipeline: Stage2 = CommitPack)

Usage:
    pip install datasets
    python data/prepare_mixed.py --num-samples 200000
"""
import argparse
import os
import sys
import time
import random
import re
import numpy as np


# ── Byte-level tokenizer (vocab ⊂ 32000, no external model needed) ──
EOS_ID = 257
MASK_ID = 258  # [MASK] for span corruption

def encode(text: str) -> list:
    return [b + 1 for b in text.encode("utf-8", errors="replace")] + [EOS_ID]


# ── GLM-130B span corruption ──
def span_corrupt(text: str, mask_ratio: float = 0.15, mean_span_len: int = 3,
                 seed: int = None) -> tuple:
    """T5/GLM-style geometric span masking.
    Returns (masked_text_ids, target_ids)."""
    ids = [b + 1 for b in text.encode("utf-8", errors="replace")]
    if len(ids) < 10:
        return ids + [EOS_ID], []

    rng = random.Random(seed)
    n_mask = max(1, int(len(ids) * mask_ratio))
    masked = list(ids)
    targets = []
    i = 0
    masked_count = 0
    result = []

    while i < len(ids) and masked_count < n_mask:
        if rng.random() < mask_ratio:
            # Start a span
            span_len = min(max(1, int(rng.expovariate(1.0 / mean_span_len))),
                          len(ids) - i, n_mask - masked_count)
            result.append(MASK_ID)
            targets.extend(ids[i:i + span_len])
            targets.append(MASK_ID)
            masked_count += span_len
            i += span_len
        else:
            result.append(ids[i])
            i += 1
    result.extend(ids[i:])
    result.append(EOS_ID)
    targets.append(EOS_ID)
    return result, targets


# ── Commit formatting ──
COMMIT_BEFORE = encode("<commit_before>\n")[:-1]  # strip EOS
COMMIT_AFTER = encode("<commit_after>\n")[:-1]
COMMIT_MSG = encode("<commit_msg>\n")[:-1]
CORRUPT_DIFF = encode("<corrupt_diff>\n")[:-1]
TARGETS_TAG = encode("<targets>\n")[:-1]


def format_causal(sample: dict) -> list:
    """30% path: before → after causal structure."""
    old = sample.get("old_contents", "") or ""
    new = sample.get("new_contents", "") or ""
    msg = sample.get("message", "") or ""
    ids = (COMMIT_BEFORE + encode(old[:2000])[:-1] +
           COMMIT_MSG + encode(msg[:500])[:-1] +
           COMMIT_AFTER + encode(new[:2000]))
    return ids


def format_span_corrupt(sample: dict) -> list:
    """70% path: mask diff spans, predict from context + message."""
    old = sample.get("old_contents", "") or ""
    new = sample.get("new_contents", "") or ""
    msg = sample.get("message", "") or ""
    diff = f"{old[:1500]}\n---\n{new[:1500]}"
    masked_ids, target_ids = span_corrupt(diff)
    ids = (CORRUPT_DIFF + masked_ids +
           COMMIT_MSG + encode(msg[:500])[:-1] +
           TARGETS_TAG + target_ids)
    return ids


def prepare(
    output_dir: str = "data",
    num_samples: int = 200_000,
    languages: list = None,
    alpha: float = 0.7,
    train_ratio: float = 0.99,
    seed: int = 42,
):
    from datasets import load_dataset

    if languages is None:
        languages = ["python", "javascript"]

    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)

    all_ids = []
    total_tokens = 0
    span_count = 0
    causal_count = 0
    t0 = time.time()

    for lang in languages:
        print(f"[prepare_mixed] Loading commitpack/{lang} (streaming)...")
        try:
            ds = load_dataset(
                "bigcode/commitpack", lang,
                split="train", streaming=True,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"[prepare_mixed] WARNING: {lang}: {e}", file=sys.stderr)
            continue

        count = 0
        target = num_samples // len(languages)
        for sample in ds:
            msg = sample.get("message", "")
            if not msg or len(msg) < 5:
                continue

            # ── GLM-130B 70/30 mixed objective ──
            if rng.random() < alpha:
                ids = format_span_corrupt(sample)
                span_count += 1
            else:
                ids = format_causal(sample)
                causal_count += 1

            all_ids.extend(ids)
            total_tokens += len(ids)
            count += 1
            if count >= target:
                break
            if count % 20000 == 0:
                elapsed = time.time() - t0
                print(f"  [{lang}] {count:,}/{target:,} samples, "
                      f"{total_tokens:,} tokens, "
                      f"span={span_count} causal={causal_count} ({elapsed:.0f}s)")

    print(f"\n[prepare_mixed] Total: {total_tokens:,} tokens")
    print(f"  span_corruption: {span_count:,} ({span_count/(span_count+causal_count):.0%})")
    print(f"  causal_lm:       {causal_count:,} ({causal_count/(span_count+causal_count):.0%})")

    arr = np.array(all_ids, dtype=np.int32)
    split_idx = int(len(arr) * train_ratio)

    train_path = os.path.join(output_dir, "commitpack_train.npy")
    valid_path = os.path.join(output_dir, "commitpack_valid.npy")
    np.save(train_path, arr[:split_idx])
    np.save(valid_path, arr[split_idx:])

    train_mb = os.path.getsize(train_path) / (1024 * 1024)
    print(f"\n[prepare_mixed] {train_path} ({train_mb:.1f} MB, {split_idx:,} tokens)")
    print(f"[prepare_mixed] Byte-level tokenizer: 1-256=bytes, 257=EOS, 258=MASK")
    print(f"[prepare_mixed] GLM-130B mixed: alpha={alpha} (70% span + 30% causal)")

    # Write metadata
    import json
    meta = {
        "total_tokens": total_tokens, "train_tokens": split_idx,
        "valid_tokens": len(arr) - split_idx,
        "span_samples": span_count, "causal_samples": causal_count,
        "alpha": alpha, "languages": languages,
        "tokenizer": "byte-level (1-256=bytes, 257=EOS, 258=MASK)",
        "vocab_subset": "⊂ 32000",
    }
    with open(os.path.join(output_dir, "commitpack_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=200_000)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--alpha", type=float, default=0.7, help="span corruption ratio")
    parser.add_argument("--languages", nargs="+", default=["python", "javascript"])
    args = parser.parse_args()
    prepare(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        alpha=args.alpha,
        languages=args.languages,
    )
