#!/bin/bash
# Neuron_SP 数据准备 + 训练
# 所有代码有 upstream commit 依据，无自研 tokenizer 训练
set -e
cd "$(cd "$(dirname "$0")" && pwd)"

echo "╔═══════════════════════════════════════════════════╗"
echo "║  Neuron_SP pipeline (CommitPackFT → train)        ║"
echo "╚═══════════════════════════════════════════════════╝"

# Step 1: Download CommitPackFT
if [ ! -d "data/hf_cache/commitpackft" ] || [ -z "$(find data/hf_cache/commitpackft -name '*.parquet' 2>/dev/null)" ]; then
    echo ""
    echo "═══ Step 1/3: Downloading CommitPackFT ═══"
    pip install pyarrow --quiet 2>/dev/null
    hf download --repo-type dataset bigcode/commitpackft --local-dir data/hf_cache/commitpackft 2>/dev/null \
        || echo "⚠ hf download failed, will use synthetic fallback"
else
    echo "═══ Step 1/3: CommitPackFT cached ═══"
fi

# Step 2: Tokenize with 70/30 mixed objective (byte-level, no external tokenizer)
if [ ! -f "data/commitpack_train.npy" ]; then
    echo ""
    echo "═══ Step 2/3: Tokenizing (70/30 GLM mixed objective) ═══"
    python3 data/prepare_mixed.py \
        --num-samples 200000 \
        --output-dir data \
        --alpha 0.7 \
        --languages python javascript
else
    echo "═══ Step 2/3: Training data exists ═══"
fi

echo ""
echo "════════════════════════════════════════"
ls -lh data/commitpack_*.npy data/commitpack_meta.json 2>/dev/null
echo "════════════════════════════════════════"

# Step 3: Launch training
echo ""
echo "═══ Step 3/3: Launching 3-GPU training ═══"
exec bash launch_7b_3gpu.sh
