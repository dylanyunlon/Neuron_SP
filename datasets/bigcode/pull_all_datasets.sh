#!/usr/bin/env bash
# pull_all_datasets.sh — 在 ags1 服务器上执行，拉取 BigCode 四大 commit 数据集
# 前置: pip install datasets huggingface_hub
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

echo "=========================================="
echo "BigCode Commit Datasets Downloader"
echo "=========================================="

# ── 1. CommitPackFT (2GB, 高质量子集, GPT-4 筛选) ──
echo ""
echo "[1/4] bigcode/commitpackft — 2GB high-quality instruction commits"
python3 << 'PY1'
from datasets import load_dataset
import os, json

out = "commitpackft"
os.makedirs(out, exist_ok=True)

# 所有语言列表
langs = ["python", "javascript", "typescript", "java", "go", "rust", "c", "cpp",
         "ruby", "php", "swift", "kotlin", "scala", "r", "julia", "lua",
         "haskell", "perl", "shell", "powershell"]

total = 0
for lang in langs:
    try:
        ds = load_dataset("bigcode/commitpackft", lang, split="train")
        n = len(ds)
        total += n
        ds.to_json(f"{out}/{lang}.jsonl")
        print(f"  {lang}: {n} samples -> {out}/{lang}.jsonl")
    except Exception as e:
        print(f"  {lang}: SKIP ({e})")

print(f"  TOTAL: {total} samples across {len(langs)} languages")
PY1

# ── 2. StarCoder commits (32-64GB subset of starcoderdata) ──
echo ""
echo "[2/4] bigcode/starcoderdata — git-commits + git-commits-cleaned splits (~32-64GB)"
echo "  NOTE: Two splits available:"
echo "    git-commits         (~32GB) raw single-file commits"
echo "    git-commits-cleaned (~64GB) deduplicated + near-dedup filtered"
python3 << 'PY2'
from datasets import load_dataset
import os
import json

out = "starcoderdata_commits"
os.makedirs(out, exist_ok=True)

# ── 2a. git-commits (raw, ~32GB) ──
print("  [2a] git-commits split (raw, ~32GB) — streaming sample")
ds_raw = load_dataset("bigcode/starcoderdata", data_dir="git-commits", split="train", streaming=True)

count = 0
with open(f"{out}/sample_10k.jsonl", "w") as f:
    for item in ds_raw:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        count += 1
        if count >= 10000:
            break

print(f"    Saved {count} samples to {out}/sample_10k.jsonl")
print(f"    Full dataset: ~32GB, use streaming=True for processing")

# ── 2b. git-commits-cleaned (deduped, ~64GB) ──
# This split applies StarCoder near-dedup + exact-dedup filtering on top of
# git-commits; it is ~2x larger in token count due to retained full-file context
# for 20% of samples, but higher quality for pretraining objectives.
print("  [2b] git-commits-cleaned split (near-dedup filtered, ~64GB) — streaming sample")
ds_cleaned = load_dataset("bigcode/starcoderdata", data_dir="git-commits-cleaned", split="train", streaming=True)

count = 0
with open(f"{out}/sample_cleaned_10k.jsonl", "w") as f:
    for item in ds_cleaned:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        count += 1
        if count >= 10000:
            break

print(f"    Saved {count} samples to {out}/sample_cleaned_10k.jsonl")
print(f"    Full dataset: ~64GB, use streaming=True for processing")

# Save download script for full dataset (both splits)
with open(f"{out}/download_full.py", "w") as f:
    f.write('''from datasets import load_dataset

# git-commits: raw single-file commits (~32GB)
ds_raw = load_dataset("bigcode/starcoderdata", data_dir="git-commits", split="train")
ds_raw.save_to_disk("starcoderdata_commits_full")
print(f"git-commits total: {len(ds_raw)} samples")

# git-commits-cleaned: near-dedup + exact-dedup filtered (~64GB, preferred for pretraining)
ds_cleaned = load_dataset("bigcode/starcoderdata", data_dir="git-commits-cleaned", split="train")
ds_cleaned.save_to_disk("starcoderdata_commits_cleaned_full")
print(f"git-commits-cleaned total: {len(ds_cleaned)} samples")
''')
PY2

# ── 3. CommitPack (4TB, 完整 commit 数据) ──
echo ""
echo "[3/4] bigcode/commitpack — 4TB full Git commits"
echo "  NOTE: 4TB corpus — streaming=True mandatory; only sample subsets are written to disk."
python3 << 'PY3'
# CommitPack streaming pull.
#
# Why streaming=True is non-negotiable here:
#   Without it, HuggingFace would try to download and cache all ~4 TB before
#   yielding a single sample.  streaming=True flips to an IterableDataset that
#   fetches one Arrow shard at a time (~128 MB each), so resident memory and
#   disk usage stay bounded regardless of language corpus size.
#
# DATASET_REGISTRY in load_commits.py enforces this constraint programmatically
# so callers cannot accidentally omit streaming=True for this dataset.

from datasets import load_dataset
import os, json

out = "commitpack"
os.makedirs(out, exist_ok=True)

# Representative sample set: cover top-5 languages for integration tests
# and CI smoke runs without pulling the full 4 TB corpus.
SAMPLE_LANGS = ["python", "javascript", "java", "go", "rust"]
SAMPLE_SIZE  = 10_000   # per language

for lang in SAMPLE_LANGS:
    try:
        # streaming=True: shards are fetched on demand; no full-corpus download
        ds = load_dataset("bigcode/commitpack", lang, split="train", streaming=True)
        count = 0
        out_file = f"{out}/{lang}_sample_{SAMPLE_SIZE}.jsonl"
        with open(out_file, "w") as f:
            for item in ds:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
                if count >= SAMPLE_SIZE:
                    break
        print(f"  {lang}: {count} samples -> {out_file}")
    except Exception as e:
        print(f"  {lang}: SKIP ({e})")

# Streaming download helper used by training pipeline (load_commits.py registry)
with open(f"{out}/download_by_language.py", "w") as f:
    f.write('''"""Stream CommitPack one language at a time (4TB total).

Usage:
    python download_by_language.py python          # stream to disk
    python download_by_language.py javascript --sample 50000

streaming=True is enforced; omitting it on a 4TB corpus would exhaust disk.
"""
import sys, argparse
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("lang", default="python", nargs="?")
parser.add_argument("--sample", type=int, default=None,
                    help="Cap: only materialise this many rows (streaming walk)")
args = parser.parse_args()

# streaming=True: load one shard at a time — mandatory for 4TB corpus
ds = load_dataset("bigcode/commitpack", args.lang, split="train", streaming=True)

if args.sample:
    import itertools, json
    rows = list(itertools.islice(ds, args.sample))
    out  = f"commitpack_{args.lang}_sample{args.sample}.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\\n")
    print(f"{args.lang}: {len(rows)} samples -> {out}")
else:
    # Fully materialise language shard to disk (may be hundreds of GB)
    ds_full = load_dataset("bigcode/commitpack", args.lang, split="train")
    ds_full.save_to_disk(f"commitpack_{args.lang}")
    print(f"{args.lang}: {len(ds_full)} samples saved to commitpack_{args.lang}/")
''')

print(f"  streaming helper written to {out}/download_by_language.py")
print(f"  Use load_commits.DATASET_REGISTRY['commitpack'] for programmatic access")
PY3

# ── 4. The Stack v2 (PR/commit 数据) ──
echo ""
echo "[4/4] bigcode/the-stack-v2 — PR + commit data"
echo "  NOTE: Requires agreement to dataset terms on HuggingFace first"
python3 << 'PY4'
from huggingface_hub import HfApi
import os, json

out = "the_stack_v2"
os.makedirs(out, exist_ok=True)

api = HfApi()
try:
    info = api.dataset_info("bigcode/the-stack-v2")
    meta = {
        "id": info.id,
        "downloads": info.downloads,
        "likes": info.likes,
        "tags": info.tags[:20] if info.tags else [],
        "size_category": info.card_data.get("size_categories", "unknown") if info.card_data else "unknown"
    }
    with open(f"{out}/dataset_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  Metadata saved to {out}/dataset_meta.json")
except Exception as e:
    print(f"  Could not fetch metadata: {e}")

# Save access instructions
with open(f"{out}/README.md", "w") as f:
    f.write("""# The Stack v2

## Access
1. Go to https://huggingface.co/datasets/bigcode/the-stack-v2
2. Accept the agreement
3. Run: `huggingface-cli login`

## Download specific subsets
```python
from datasets import load_dataset
# Full dataset is too large, use streaming
ds = load_dataset("bigcode/the-stack-v2", streaming=True, split="train")
```
""")
print(f"  README saved to {out}/README.md")

# PR/commit streaming download script (requires accepted agreement + HF token)
with open(f"{out}/stream_pr_commits.py", "w") as f:
    f.write('''# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
Stream The Stack v2 PR/commit subset into DES-LOC pretraining format.

Requires:
  - huggingface-cli login (accepted agreement on HF Hub)
  - pip install datasets huggingface_hub pyarrow

Usage:
    python stream_pr_commits.py --max-samples 50000 --out-dir /data/stackv2_commits
"""
import argparse
import os
import json
import sys

# Adapter is in the sibling package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from datasets.bigcode.the_stack_v2.stackv2_commits import StackV2CommitAdapter


def main():
    parser = argparse.ArgumentParser(description="Stream Stack v2 PR/commit subset")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Stop after N formatted samples (None = unlimited)")
    parser.add_argument("--out-dir", type=str, default="stackv2_commits_jsonl",
                        help="Output directory for per-language JSONL files")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token (defaults to cached login)")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Disable directory_id deduplication")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    adapter = StackV2CommitAdapter(dedup=not args.no_dedup)
    handles = {}
    count = 0

    print(f"Streaming bigcode/the-stack-v2 ({args.split}) → {args.out_dir}/")
    for sample in adapter.stream_hf(
        split=args.split,
        max_samples=args.max_samples,
        hf_token=args.hf_token,
    ):
        lang = sample.get("lang", "unknown")
        if lang not in handles:
            handles[lang] = open(os.path.join(args.out_dir, f"{lang}.jsonl"), "w")
        handles[lang].write(json.dumps(sample, ensure_ascii=False) + "\\n")
        count += 1
        if count % 5000 == 0:
            print(f"  {count} samples written …")

    for fh in handles.values():
        fh.close()

    adapter.print_stats()
    print(f"Done. {count} samples saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
''')
print(f"  stream_pr_commits.py saved to {out}/stream_pr_commits.py")
PY4

echo ""
echo "=========================================="
echo "Done. Summary:"
echo "  commitpackft/     — ready to use (full download ~2GB)"
echo "  starcoderdata_commits/ — 10K samples (raw + cleaned) + full download script"
echo "  commitpack/       — 10K Python sample + per-language download script"
echo "  the_stack_v2/     — metadata + access instructions"
echo "=========================================="
