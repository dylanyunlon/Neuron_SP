#!/usr/bin/env bash
# launch_7b.sh — 3-GPU heterogeneous DES-LOC 7B pretrain with FSDP
#
# Hardware topology (from nvidia-smi 2026-06-24):
#   GPU0: A6000      49GB  (01:00.0)
#   GPU1: Blackwell  96GB  (21:00.0) — SKIP (no cu118 kernel)
#   GPU2: H100 NVL   93GB  (61:00.0)
#   GPU3: A6000      49GB  (81:00.0)
#   GPU4: Blackwell  96GB  (A1:00.0) — SKIP (no cu118 kernel)
#
# Use GPU0 + GPU2 + GPU3 = A6000 + H100 + A6000
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/7b_pretrain_${TIMESTAMP}.log"

# A6000(GPU0) + H100(GPU2) + A6000(GPU3) — skip Blackwell GPU1,GPU4
export CUDA_VISIBLE_DEVICES=0,2,3
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "=== Neuron_SP 7B DES-LOC (H100 + 2×A6000, FSDP) ==="
echo "GPUs: $CUDA_VISIBLE_DEVICES (A6000+H100+A6000)"
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
