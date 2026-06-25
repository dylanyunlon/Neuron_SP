#!/bin/bash
set -e
cd "$(cd "$(dirname "$0")" && pwd)"

echo "=== Neuron_SP 数据准备 ==="
echo "下载 CommitPack + byte-level 分词（无需外部 tokenizer）"

pip install datasets --quiet 2>/dev/null

python3 data/prepare_simple.py \
    --num-samples 200000 \
    --output-dir data \
    --languages python javascript

echo ""
echo "=== 完成 ==="
ls -lh data/commitpack_*.npy 2>/dev/null
echo ""
echo "现在可以运行: bash launch_7b_3gpu.sh"
