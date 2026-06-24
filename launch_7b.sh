#!/usr/bin/env bash
# launch_7b.sh — single H100 DES-LOC 7B pretrain
# A6000 (47GB) can't fit 7B model+optimizer (needs ~50GB).
# Run on H100 only until ZeRO-3 sharding is implemented.
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/7b_pretrain_${TIMESTAMP}.log"

# H100 only (physical GPU2)
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=16

echo "=== Neuron_SP 7B DES-LOC (H100 single GPU) ==="
echo "Log: $LOG"

python run_pretrain.py \
    --model-size 7b \
    --steps 100000 \
    --batch-size 1 \
    --seq-len 2048 \
    --use-desloc \
    --gradient-checkpointing \
    --log-every 10 \
    --save-every 500 \
    --checkpoint-dir checkpoints/7b_${TIMESTAMP} \
    2>&1 | tee "$LOG"
