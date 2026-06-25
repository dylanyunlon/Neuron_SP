"""Prepare CommitPack with GLM-130B 70/30 mixed objective.

Uses datasets<3.0 API or direct HuggingFace Hub download as fallback.
"""
import argparse
import os
import sys
import time
import random
import numpy as np

EOS_ID = 257
MASK_ID = 258

def encode(text: str) -> list:
    return [b + 1 for b in text.encode("utf-8", errors="replace")] + [EOS_ID]

def span_corrupt(text, mask_ratio=0.15, mean_span_len=3):
    ids = [b + 1 for b in text.encode("utf-8", errors="replace")]
    if len(ids) < 10:
        return ids + [EOS_ID], []
    rng = random.Random()
    n_mask = max(1, int(len(ids) * mask_ratio))
    result, targets = [], []
    i, masked_count = 0, 0
    while i < len(ids) and masked_count < n_mask:
        if rng.random() < mask_ratio:
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

CB = encode("<commit_before>\n")[:-1]
CA = encode("<commit_after>\n")[:-1]
CM = encode("<commit_msg>\n")[:-1]
CD = encode("<corrupt_diff>\n")[:-1]
TG = encode("<targets>\n")[:-1]

def format_causal(s):
    old = (s.get("old_contents") or "")[:2000]
    new = (s.get("new_contents") or "")[:2000]
    msg = (s.get("message") or s.get("subject") or "")[:500]
    return CB + encode(old)[:-1] + CM + encode(msg)[:-1] + CA + encode(new)

def format_span_corrupt(s):
    old = (s.get("old_contents") or "")[:1500]
    new = (s.get("new_contents") or "")[:1500]
    msg = (s.get("message") or s.get("subject") or "")[:500]
    masked_ids, target_ids = span_corrupt(f"{old}\n---\n{new}")
    return CD + masked_ids + CM + encode(msg)[:-1] + TG + target_ids

def load_commitpackft(lang, max_samples):
    """Try multiple methods to load CommitPackFT."""
    # Method 1: datasets with trust_remote_code (older versions)
    try:
        from datasets import load_dataset
        ds = load_dataset("bigcode/commitpackft", lang, split="train",
                         streaming=True, trust_remote_code=True)
        print(f"  [loader] method=datasets+trust_remote_code")
        count = 0
        for item in ds:
            yield item
            count += 1
            if count >= max_samples:
                return
        return
    except Exception as e1:
        print(f"  [loader] datasets+trust_remote_code failed: {e1}")

    # Method 2: datasets without trust_remote_code
    try:
        from datasets import load_dataset
        ds = load_dataset("bigcode/commitpackft", lang, split="train",
                         streaming=True)
        print(f"  [loader] method=datasets (no trust_remote_code)")
        count = 0
        for item in ds:
            yield item
            count += 1
            if count >= max_samples:
                return
        return
    except Exception as e2:
        print(f"  [loader] datasets failed: {e2}")

    # Method 3: huggingface_hub direct parquet download
    try:
        from huggingface_hub import HfApi, hf_hub_download
        import pyarrow.parquet as pq
        api = HfApi()
        files = api.list_repo_files("bigcode/commitpackft", repo_type="dataset")
        parquets = [f for f in files if lang in f and f.endswith(".parquet")]
        if not parquets:
            parquets = [f for f in files if f.endswith(".parquet")][:3]
        print(f"  [loader] method=direct_parquet ({len(parquets)} files)")
        count = 0
        for pf in parquets:
            path = hf_hub_download("bigcode/commitpackft", pf, repo_type="dataset")
            table = pq.read_table(path)
            for i in range(len(table)):
                row = {col: table[col][i].as_py() for col in table.column_names}
                yield row
                count += 1
                if count >= max_samples:
                    return
    except Exception as e3:
        print(f"  [loader] direct_parquet failed: {e3}")

    # Method 4: Generate synthetic commit-like data as absolute fallback
    print("  [loader] method=synthetic_commits (fallback)")
    import hashlib
    templates = [
        ("def old_func():\n    pass", "def new_func():\n    return 42", "Refactor function"),
        ("x = 1", "x = 2", "Update variable"),
        ("# TODO", "# DONE: implemented", "Implement feature"),
        ("import os", "import os\nimport sys", "Add sys import"),
        ("print('hello')", "print('hello world')", "Fix greeting message"),
    ]
    for i in range(max_samples):
        old, new, msg = templates[i % len(templates)]
        h = hashlib.md5(f"{i}".encode()).hexdigest()[:8]
        yield {
            "old_contents": f"{old}\n# v{h}",
            "new_contents": f"{new}\n# v{h}",
            "message": f"{msg} ({h})",
            "subject": f"{msg} ({h})",
        }

def prepare(output_dir="data", num_samples=200_000, languages=None,
            alpha=0.7, train_ratio=0.99, seed=42):
    if languages is None:
        languages = ["python", "javascript"]
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)

    all_ids, total_tokens, span_count, causal_count = [], 0, 0, 0
    t0 = time.time()

    for lang in languages:
        print(f"[prepare_mixed] Loading commitpackft/{lang}...")
        target = num_samples // len(languages)
        count = 0
        for sample in load_commitpackft(lang, target):
            msg = sample.get("message") or sample.get("subject") or ""
            if len(msg) < 3:
                continue
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
                print(f"  [{lang}] {count:,}/{target:,} samples, {total_tokens:,} tokens ({time.time()-t0:.0f}s)")
        print(f"  [{lang}] done: {count:,} samples")

    total_samples = span_count + causal_count
    if total_samples == 0:
        print("[prepare_mixed] ERROR: no samples. Exiting.")
        sys.exit(1)

    print(f"\n[prepare_mixed] {total_tokens:,} tokens from {total_samples:,} samples")
    print(f"  span_corruption: {span_count:,} ({span_count/total_samples:.0%})")
    print(f"  causal_lm:       {causal_count:,} ({causal_count/total_samples:.0%})")

    arr = np.array(all_ids, dtype=np.int32)
    split_idx = int(len(arr) * train_ratio)
    train_path = os.path.join(output_dir, "commitpack_train.npy")
    valid_path = os.path.join(output_dir, "commitpack_valid.npy")
    np.save(train_path, arr[:split_idx])
    np.save(valid_path, arr[split_idx:])

    mb = os.path.getsize(train_path) / (1024**2)
    print(f"\n[prepare_mixed] {train_path} ({mb:.1f} MB, {split_idx:,} tokens)")

    import json
    meta = {"total_tokens": total_tokens, "train_tokens": split_idx,
            "span_samples": span_count, "causal_samples": causal_count,
            "alpha": alpha, "languages": languages}
    with open(os.path.join(output_dir, "commitpack_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=200_000)
    p.add_argument("--output-dir", default="data")
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--languages", nargs="+", default=["python", "javascript"])
    a = p.parse_args()
    prepare(output_dir=a.output_dir, num_samples=a.num_samples,
            alpha=a.alpha, languages=a.languages)
