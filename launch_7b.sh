#!/usr/bin/env bash
# launch_7b.sh — ags1 3-GPU DES-LOC 7B pretrain (H100 + 2×A6000)
# Blackwell SM120 excluded: PyTorch cu118 has no kernel image for SM120
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/7b_pretrain_${TIMESTAMP}.log"

# Only H100 (GPU2) + A6000 (GPU3, GPU4) — skip Blackwell (GPU0, GPU1)
export CUDA_VISIBLE_DEVICES=2,3,4
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "=== Neuron_SP 7B DES-LOC (H100 + 2×A6000) ==="
echo "GPUs: $CUDA_VISIBLE_DEVICES (3 devices)"
echo "Log: $LOG"

torchrun --nproc_per_node=3 --master_port=29500 \
    run_pretrain.py \
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
