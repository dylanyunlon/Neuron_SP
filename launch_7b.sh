#!/usr/bin/env bash
# launch_7b.sh — 在 ags1 的 5 张 GPU 上用 DES-LOC 异构引擎启动 7B 预训练
# 用法: bash launch_7b.sh [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/7b_pretrain_${TIMESTAMP}.log"

# GPU config: all 5 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

# NCCL: disable P2P (no NVLink between GPUs), use SHM for intra-node
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=WARN

# Torch config
export OMP_NUM_THREADS=8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

NGPUS=5
MODEL_SIZE="7b"
TOTAL_STEPS=100000
MICRO_BS=1
SEQ_LEN=2048

echo "=== Neuron_SP 7B Pretrain (DES-LOC) ==="
echo "GPUs: $CUDA_VISIBLE_DEVICES ($NGPUS)"
echo "Model: $MODEL_SIZE"
echo "Log: $LOG_FILE"
echo ""

if [[ "${1:-}" == "--dry-run" ]]; then
    echo "[DRY RUN] Would execute:"
    echo "  torchrun --nproc_per_node=$NGPUS run_pretrain.py \\"
    echo "    --model-size $MODEL_SIZE --steps $TOTAL_STEPS \\"
    echo "    --batch-size $MICRO_BS --seq-len $SEQ_LEN \\"
    echo "    --use-desloc --gradient-checkpointing"
    echo ""
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"
    exit 0
fi

echo "Starting DES-LOC training..." | tee "$LOG_FILE"

torchrun \
    --nproc_per_node=$NGPUS \
    --master_port=29500 \
    run_pretrain.py \
    --model-size "$MODEL_SIZE" \
    --steps "$TOTAL_STEPS" \
    --batch-size "$MICRO_BS" \
    --seq-len "$SEQ_LEN" \
    --use-desloc \
    --gradient-checkpointing \
    --log-every 10 \
    --save-every 500 \
    --checkpoint-dir checkpoints/7b_${TIMESTAMP} \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== Training complete. Log: $LOG_FILE ===" | tee -a "$LOG_FILE"
