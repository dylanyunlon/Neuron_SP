#!/bin/bash
# Neuron_SP 真实数据准备 — GLM-130B 70/30 mixed objective
# 下载 CommitPack + 70% span corruption + 30% causal LM
# 输出: data/commitpack_train.npy
set -e
cd "$(cd "$(dirname "$0")" && pwd)"

echo "╔══════════════════════════════════════════════╗"
echo "║  Neuron_SP 数据准备 (GLM-130B mixed 70/30)   ║"
echo "╚══════════════════════════════════════════════╝"

# Check dependency
if ! python3 -c "import datasets" 2>/dev/null; then
    echo "Installing datasets..."
    pip install datasets --quiet
fi

python3 data/prepare_mixed.py \
    --num-samples 200000 \
    --output-dir data \
    --alpha 0.7 \
    --languages python javascript

echo ""
ls -lh data/commitpack_*.npy data/commitpack_meta.json 2>/dev/null
echo ""
echo "✓ 运行: bash launch_7b_3gpu.sh"
