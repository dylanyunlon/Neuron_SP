#!/usr/bin/env bash
# launch_7b.sh — 3-GPU heterogeneous DES-LOC pretrain
# 3B model fits all GPUs: bf16=5.6GB + Adam=11.2GB + act~5GB = ~22GB (A6000 47GB OK)
# 7B needs ZeRO-3 sharding (not yet implemented) — A6000 can't fit full replica
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/3b_pretrain_${TIMESTAMP}.log"

# H100 (GPU2) + 2×A6000 (GPU3, GPU4) — skip Blackwell (GPU0, GPU1)
export CUDA_VISIBLE_DEVICES=2,3,4
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "=== Neuron_SP 3B DES-LOC (H100 + 2×A6000) ==="
echo "GPUs: $CUDA_VISIBLE_DEVICES (3 devices)"
echo "Log: $LOG"

torchrun --nproc_per_node=3 --master_port=29500 \
    run_pretrain.py \
    --model-size 3b \
    --steps 100000 \
    --batch-size 1 \
    --seq-len 2048 \
    --use-desloc \
    --gradient-checkpointing \
    --log-every 10 \
    --save-every 500 \
    --checkpoint-dir checkpoints/3b_${TIMESTAMP} \
    2>&1 | tee "$LOG"
