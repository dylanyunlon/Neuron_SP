#!/bin/bash
# Neuron_SP 完整数据准备 pipeline
# Step 1: 下载 CommitPackFT
# Step 2: 训练 BPE tokenizer (vocab=32000, sentencepiece)
# Step 3: 用 BPE tokenizer 生成 70/30 mixed training data
set -e
cd "$(cd "$(dirname "$0")" && pwd)"

echo "╔═══════════════════════════════════════════════════╗"
echo "║  Neuron_SP 数据准备 (BPE tokenizer + GLM 70/30)  ║"
echo "╚═══════════════════════════════════════════════════╝"

# Dependencies
for pkg in sentencepiece pyarrow; do
    python3 -c "import ${pkg}" 2>/dev/null || pip install ${pkg} --quiet
done

# Step 1: Download CommitPackFT (if not cached)
if [ ! -d "data/hf_cache/commitpackft" ] || [ -z "$(find data/hf_cache/commitpackft -name '*.parquet' 2>/dev/null)" ]; then
    echo ""
    echo "═══ Step 1/3: Downloading CommitPackFT ═══"
    hf download --repo-type dataset bigcode/commitpackft --local-dir data/hf_cache/commitpackft 2>/dev/null \
        || echo "⚠ hf download failed, will use synthetic fallback"
fi

# Step 2: Train tokenizer (if not already trained)
if [ ! -f "data/neuron_sp.model" ]; then
    echo ""
    echo "═══ Step 2/3: Training BPE tokenizer (vocab=32000) ═══"
    python3 data/train_tokenizer.py \
        --vocab-size 32000 \
        --max-lines 500000 \
        --output-dir data
else
    echo "═══ Step 2/3: Tokenizer exists (data/neuron_sp.model) ═══"
fi

# Step 3: Tokenize with BPE + 70/30 mixed objective
echo ""
echo "═══ Step 3/3: Generating training data (70/30 mixed) ═══"
python3 data/prepare_mixed.py \
    --num-samples 200000 \
    --output-dir data \
    --alpha 0.7 \
    --languages python javascript \
    --tokenizer data/neuron_sp.model

echo ""
echo "════════════════════════════════════════"
ls -lh data/neuron_sp.model data/commitpack_*.npy data/commitpack_meta.json 2>/dev/null
echo "════════════════════════════════════════"
echo "✓ bash launch_7b_3gpu.sh"
