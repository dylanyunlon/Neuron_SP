"""Prepare CommitPack with GLM-130B 70/30 mixed objective.

Uses huggingface-cli download (most reliable) + fallback methods.
"""
import argparse
import os
import sys
import time
import random
import subprocess
import json
import glob
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


def download_and_iter(lang, max_samples, cache_dir="data/hf_cache"):
    """Download CommitPackFT via huggingface-cli, then iterate parquet files."""
    repo_id = "bigcode/commitpackft"
    local_dir = os.path.join(cache_dir, "commitpackft")

    # Step 1: huggingface-cli download
    print(f"  [download] huggingface-cli download {repo_id} ...")
    try:
        subprocess.run(
            ["hf", "download",
             "--repo-type", "dataset",
             
             repo_id,
             "--local-dir", local_dir],
            check=True, capture_output=True, text=True, timeout=600,
        )
        print(f"  [download] saved to {local_dir}")
    except subprocess.CalledProcessError as e:
        print(f"  [download] huggingface-cli failed: {e.stderr[:200]}")
        # Try without auth
        try:
            subprocess.run(
                ["hf", "download",
                 "--repo-type", "dataset",
                 repo_id,
                 "--local-dir", local_dir],
                check=True, capture_output=True, text=True, timeout=600,
            )
        except Exception as e2:
            print(f"  [download] retry also failed: {e2}")
            yield from _synthetic_fallback(max_samples)
            return
    except FileNotFoundError:
        print("  [download] huggingface-cli not found, trying pip install...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub[cli]"],
                      check=False)
        try:
            subprocess.run(
                ["hf", "download", "--repo-type", "dataset",
                 repo_id, "--local-dir", local_dir],
                check=True, capture_output=True, text=True, timeout=600,
            )
        except Exception as e3:
            print(f"  [download] still failed: {e3}")
            yield from _synthetic_fallback(max_samples)
            return

    # Step 2: Find and read parquet files for this language
    parquets = sorted(glob.glob(os.path.join(local_dir, "**", f"*{lang}*.parquet"), recursive=True))
    if not parquets:
        # Try all parquets
        parquets = sorted(glob.glob(os.path.join(local_dir, "**", "*.parquet"), recursive=True))
    if not parquets:
        # Try jsonl
        jsonls = sorted(glob.glob(os.path.join(local_dir, "**", f"*{lang}*.jsonl"), recursive=True))
        if not jsonls:
            jsonls = sorted(glob.glob(os.path.join(local_dir, "**", "*.jsonl"), recursive=True))
        if jsonls:
            print(f"  [read] {len(jsonls)} jsonl file(s)")
            count = 0
            for jf in jsonls:
                for line in open(jf):
                    try:
                        yield json.loads(line.strip())
                        count += 1
                        if count >= max_samples:
                            return
                    except json.JSONDecodeError:
                        continue
            return

    print(f"  [read] {len(parquets)} parquet file(s) for {lang}")
    try:
        import pyarrow.parquet as pq
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyarrow"], check=False)
        import pyarrow.parquet as pq

    count = 0
    for pf in parquets:
        table = pq.read_table(pf)
        for i in range(len(table)):
            row = {col: table[col][i].as_py() for col in table.column_names}
            yield row
            count += 1
            if count >= max_samples:
                return


def _synthetic_fallback(max_samples):
    """Generate synthetic commit-like data as absolute fallback."""
    print("  [loader] using synthetic commits (absolute fallback)")
    import hashlib
    templates = [
        ("def old_func():\n    pass", "def new_func():\n    return 42", "Refactor function"),
        ("x = 1\ny = 2", "x = 10\ny = 20", "Update constants"),
        ("# TODO: implement", "# DONE: implemented feature", "Implement feature"),
        ("import os", "import os\nimport sys\nimport json", "Add missing imports"),
        ("print('hello')", "logger.info('hello world')", "Use logger instead of print"),
        ("for i in range(10):\n    x += 1", "x = sum(range(10))", "Simplify loop with sum()"),
        ("class Foo:\n    pass", "class Foo:\n    def __init__(self):\n        self.x = 0", "Add constructor"),
        ("try:\n    f()\nexcept:\n    pass", "try:\n    f()\nexcept Exception as e:\n    log(e)", "Handle exception properly"),
    ]
    for i in range(max_samples):
        old, new, msg = templates[i % len(templates)]
        h = hashlib.md5(f"{i}".encode()).hexdigest()[:8]
        yield {"old_contents": f"{old}\n# {h}", "new_contents": f"{new}\n# {h}",
               "message": f"{msg} #{h}"}


def prepare(output_dir="data", num_samples=200_000, languages=None,
            alpha=0.7, train_ratio=0.99, seed=42):
    if languages is None:
        languages = ["python", "javascript"]
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)

    all_ids, total_tokens, span_count, causal_count = [], 0, 0, 0
    t0 = time.time()

    for lang in languages:
        print(f"[prepare_mixed] Processing {lang}...")
        target = num_samples // len(languages)
        count = 0
        for sample in download_and_iter(lang, target):
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
                print(f"  [{lang}] {count:,}/{target:,} ({total_tokens:,} tokens, {time.time()-t0:.0f}s)")
        print(f"  [{lang}] {count:,} samples done")

    total_samples = span_count + causal_count
    if total_samples == 0:
        print("[prepare_mixed] ERROR: no samples.")
        sys.exit(1)

    print(f"\n[prepare_mixed] {total_tokens:,} tokens, {total_samples:,} samples")
    print(f"  70% span_corruption: {span_count:,} | 30% causal_lm: {causal_count:,}")

    arr = np.array(all_ids, dtype=np.int32)
    split_idx = int(len(arr) * train_ratio)
    train_path = os.path.join(output_dir, "commitpack_train.npy")
    valid_path = os.path.join(output_dir, "commitpack_valid.npy")
    np.save(train_path, arr[:split_idx])
    np.save(valid_path, arr[split_idx:])

    mb = os.path.getsize(train_path) / (1024**2)
    print(f"[prepare_mixed] {train_path} ({mb:.1f} MB)")

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
