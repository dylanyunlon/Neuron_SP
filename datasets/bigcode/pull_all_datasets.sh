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
echo "  WARNING: 4TB is too large for full download. Pulling metadata + Python subset."
python3 << 'PY3'
from datasets import load_dataset
import os, json

out = "commitpack"
os.makedirs(out, exist_ok=True)

# Only pull Python subset in streaming mode
ds = load_dataset("bigcode/commitpack", "python", split="train", streaming=True)

count = 0
with open(f"{out}/python_sample_10k.jsonl", "w") as f:
    for item in ds:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        count += 1
        if count >= 10000:
            break

print(f"  Saved {count} Python commit samples to {out}/python_sample_10k.jsonl")

# Save full download script
with open(f"{out}/download_by_language.py", "w") as f:
    f.write('''"""Download CommitPack by language (4TB total, do one lang at a time)"""
import sys
from datasets import load_dataset

lang = sys.argv[1] if len(sys.argv) > 1 else "python"
ds = load_dataset("bigcode/commitpack", lang, split="train")
ds.save_to_disk(f"commitpack_{lang}")
print(f"{lang}: {len(ds)} samples")
''')
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
PY4

echo ""
echo "=========================================="
echo "Done. Summary:"
echo "  commitpackft/     — ready to use (full download ~2GB)"
echo "  starcoderdata_commits/ — 10K samples (raw + cleaned) + full download script"
echo "  commitpack/       — 10K Python sample + per-language download script"
echo "  the_stack_v2/     — metadata + access instructions"
echo "=========================================="
