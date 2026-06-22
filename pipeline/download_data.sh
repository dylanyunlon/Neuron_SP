#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# download_data.sh — 下载并预处理三阶段训练数据
#
# 在 ags1 上执行:
#   bash pipeline/download_data.sh
#
# 前置:
#   pip install datasets huggingface_hub transformers
#   huggingface-cli login   (需同意 Stack v2 和 StarCoder 协议)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

DATA_DIR="${DATA_DIR:-datasets/bigcode}"
CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

echo "╔══════════════════════════════════════════╗"
echo "║  DES-LOC 三阶段数据下载 + 预处理         ║"
echo "╚══════════════════════════════════════════╝"
echo "  DATA_DIR: $DATA_DIR"
echo "  CACHE:    $CACHE_DIR"
echo ""

# ── 检查 HuggingFace 登录 ──
if ! python3 -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
    echo "⚠️  未登录 HuggingFace. 运行: huggingface-cli login"
    echo "   需同意 bigcode/the-stack-v2 和 bigcode/starcoderdata 使用协议"
    exit 1
fi
echo "✓ HuggingFace 已登录"

# ── Stage 1: The Stack v2 (完整代码文件) ──
echo ""
echo "═══ Stage 1: The Stack v2 (代码基座预训练) ═══"
python3 << 'STAGE1'
import os, json, sys
sys.path.insert(0, ".")
from pipeline.unified_tokenizer import build_megatron_tokenizer
from datasets.bigcode.the_stack_v2.megatron_indexed import MegatronIndexedWriter

DATA_DIR = os.environ.get("DATA_DIR", "datasets/bigcode")
OUT = os.path.join(DATA_DIR, "the_stack_v2", "indexed")
os.makedirs(OUT, exist_ok=True)

tok = build_megatron_tokenizer()

# 按语言优先级下载
priority_langs = ["python", "javascript", "java", "go", "c", "cpp", "rust", "typescript"]

from datasets import load_dataset
total_tokens = 0

for lang in priority_langs:
    out_prefix = os.path.join(OUT, f"stack_v2_{lang}_text_document")
    if os.path.exists(out_prefix + ".bin"):
        print(f"  {lang}: already indexed, skip")
        continue
    try:
        print(f"  {lang}: loading from HuggingFace (streaming)...")
        ds = load_dataset("bigcode/the-stack-v2", data_dir=f"data/{lang}", split="train", streaming=True)
        writer = MegatronIndexedWriter(out_prefix, dtype_code=4)
        count = 0
        for item in ds:
            content = item.get("content", "")
            ids = tok.tokenize(content)
            if len(ids) < 32:
                continue
            writer.add_document(ids)
            count += 1
            total_tokens += len(ids)
            if count % 10000 == 0:
                print(f"    {lang}: {count} docs, {total_tokens/1e9:.2f}B tokens")
            # 每语言上限 (防止磁盘爆满, 可调)
            if count >= 500000:
                print(f"    {lang}: hit 500K doc limit")
                break
        writer.finalize()
        print(f"  ✓ {lang}: {count} docs indexed")
    except Exception as e:
        print(f"  ✗ {lang}: {e}")

print(f"\nStage 1 total: {total_tokens/1e9:.2f}B tokens")
STAGE1

# ── Stage 2: CommitPack (4TB commit diff) ──
echo ""
echo "═══ Stage 2: CommitPack (commit 续训) ═══"

# CommitPackFT 已下载 (392MB), 检查
if [ -f "$DATA_DIR/commitpackft/python.jsonl" ]; then
    echo "✓ CommitPackFT already downloaded"
    wc -l "$DATA_DIR/commitpackft/"*.jsonl 2>/dev/null | tail -1
else
    echo "  downloading CommitPackFT..."
    python3 << 'DL_FT'
from huggingface_hub import hf_hub_download
import os, shutil
out = os.environ.get("DATA_DIR", "datasets/bigcode") + "/commitpackft"
os.makedirs(out, exist_ok=True)
for lang in ["python","javascript","java","go","c++","rust","typescript","c"]:
    try:
        path = hf_hub_download("bigcode/commitpackft", f"data/{lang}/data.jsonl",
                               repo_type="dataset", local_dir=out+"_cache")
        shutil.copy2(path, f"{out}/{lang}.jsonl")
        print(f"  ✓ {lang}")
    except Exception as e:
        print(f"  ✗ {lang}: {e}")
DL_FT
fi

# CommitPack Python (大规模, 按 shard 下载)
echo "  CommitPack Python shards..."
python3 << 'DL_CP'
from huggingface_hub import hf_hub_download, HfApi
import os
out = os.environ.get("DATA_DIR", "datasets/bigcode") + "/commitpack"
os.makedirs(out, exist_ok=True)
api = HfApi()
files = list(api.list_repo_tree("bigcode/commitpack", path_in_repo="data/python", repo_type="dataset"))
for f in files[:5]:  # 前5个 shard (~2.5GB)
    name = f.path.split("/")[-1]
    dst = os.path.join(out, name)
    if os.path.exists(dst):
        print(f"  {name}: exists, skip")
        continue
    try:
        path = hf_hub_download("bigcode/commitpack", f.path, repo_type="dataset", local_dir=out+"_cache")
        import shutil; shutil.copy2(path, dst)
        print(f"  ✓ {name} ({f.size/1e6:.0f}MB)")
    except Exception as e:
        print(f"  ✗ {name}: {e}")
DL_CP

# ── Stage 3: CommitPackFT (已下载, 无需额外操作) ──
echo ""
echo "═══ Stage 3: CommitPackFT (指令微调) ═══"
echo "✓ Already available at $DATA_DIR/commitpackft/"

# ── StarCoder commits (bonus) ──
echo ""
echo "═══ Bonus: StarCoder commits ═══"
python3 << 'DL_SC'
try:
    from datasets import load_dataset
    import os, json
    out = os.environ.get("DATA_DIR", "datasets/bigcode") + "/starcoderdata_commits"
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "sample_10k.jsonl")):
        print("  ✓ 10K sample already exists")
    else:
        ds = load_dataset("bigcode/starcoderdata", data_dir="git-commits", split="train", streaming=True)
        count = 0
        with open(os.path.join(out, "sample_10k.jsonl"), "w") as f:
            for item in ds:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
                if count >= 10000: break
        print(f"  ✓ {count} samples saved")
except Exception as e:
    print(f"  ✗ StarCoder gated: {e}")
DL_SC

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  数据就绪. 运行训练:                      ║"
echo "║  python -m pipeline.smoke_test           ║"
echo "║  python -m pipeline.train_three_stage    ║"
echo "╚══════════════════════════════════════════╝"
