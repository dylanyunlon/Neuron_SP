#!/bin/bash
# Neuron_SP 一键 pipeline — 用已有的 upstream tokenizer 代码
set -e
cd "$(cd "$(dirname "$0")" && pwd)"

echo "╔═══════════════════════════════════════════════════╗"
echo "║  Neuron_SP pipeline (upstream tokenizer + train)  ║"
echo "╚═══════════════════════════════════════════════════╝"

# Dependencies
for pkg in tokenizers pyarrow; do
    python3 -c "import ${pkg}" 2>/dev/null || pip install ${pkg} --quiet
done

# Step 1: Download CommitPackFT (if not cached)
if [ ! -d "data/hf_cache/commitpackft" ] || [ -z "$(find data/hf_cache/commitpackft -name '*.parquet' 2>/dev/null)" ]; then
    echo ""
    echo "═══ Step 1/4: Downloading CommitPackFT ═══"
    hf download --repo-type dataset bigcode/commitpackft --local-dir data/hf_cache/commitpackft 2>/dev/null \
        || echo "⚠ hf download failed, will use synthetic fallback"
fi

# Step 2: Train BPE tokenizer (commit 1d99e0ad: tokenizer/train_bpe.py)
if [ ! -f "tokenizer/neuron_sp_32k/tokenizer.json" ]; then
    echo ""
    echo "═══ Step 2/4: Training ByteLevel BPE tokenizer (tokenizer/train_bpe.py) ═══"
    python3 tokenizer/train_bpe.py \
        --samples 500000 \
        --output tokenizer/neuron_sp_32k
else
    echo "═══ Step 2/4: Tokenizer exists (tokenizer/neuron_sp_32k/) ═══"
fi

# Step 3: Tokenize with BPE + 70/30 mixed objective
if [ ! -f "data/commitpack_train.npy" ]; then
    echo ""
    echo "═══ Step 3/4: Tokenizing data (70/30 GLM mixed) ═══"
    python3 data/prepare_mixed.py \
        --num-samples 200000 \
        --output-dir data \
        --alpha 0.7 \
        --languages python javascript \
        --tokenizer tokenizer/neuron_sp_32k/tokenizer.json
else
    echo "═══ Step 3/4: Training data exists (data/commitpack_train.npy) ═══"
fi

echo ""
echo "════════════════════════════════════════"
ls -lh tokenizer/neuron_sp_32k/tokenizer.json data/commitpack_*.npy data/commitpack_meta.json 2>/dev/null
echo "════════════════════════════════════════"

# Step 4: Launch training
echo ""
echo "═══ Step 4/4: Launching 3-GPU training ═══"
exec bash launch_7b_3gpu.sh
