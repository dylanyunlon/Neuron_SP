#!/bin/bash
# Neuron_SP 真实数据准备脚本
# 从 HuggingFace 下载 CommitPack，用 CodeLlama tokenizer 分词
# 输出: data/commitpack_train.npy (~500K samples, ~200M tokens)
set -e

echo "=== Installing dependencies ==="
pip install datasets transformers sentencepiece protobuf --quiet

echo "=== Preparing CommitPack data ==="
python3 data/prepare_commits.py \
    --task npy \
    --num-samples 500000 \
    --tokenizer-name codellama/CodeLlama-7b-hf \
    --output-dir data \
    --languages python javascript

echo "=== Done ==="
ls -lh data/commitpack_*.npy data/commitpack_meta.json 2>/dev/null
