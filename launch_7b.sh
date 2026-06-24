#!/usr/bin/env bash
# launch_7b.sh — ags1 3-GPU heterogeneous DES-LOC 7B pretrain
#
# Topology (cu118, no Blackwell):
#   GPU2: H100 NVL  93GB SM9.0  — primary compute
#   GPU3: A6000     47GB SM8.6  — FSDP shard + CPU offload
#   GPU4: A6000     47GB SM8.6  — FSDP shard + CPU offload
#
# FSDP FULL_SHARD splits model params + optimizer states across all 3 GPUs.
# A6000 uses CPU offload for optimizer states (1.5TB DRAM available).
# This is the Neuron_SP heterogeneous design — NOT single-GPU or 3B fallback.
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/7b_pretrain_${TIMESTAMP}.log"

# H100 + 2×A6000 (skip Blackwell SM120 — PyTorch cu118 limit)
export CUDA_VISIBLE_DEVICES=2,3,4
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800000
export OMP_NUM_THREADS=8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256

EXTRA_ARGS=("$@")

if [[ " ${EXTRA_ARGS[*]:-} " == *" --dry-run "* ]]; then
    echo "=== DRY RUN ==="
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader -i 2,3,4 2>/dev/null
    EXTRA_ARGS=("${EXTRA_ARGS[@]/--dry-run/}" --steps 3 --log-every 1 --save-every 0)
fi

echo "=== Neuron_SP 7B DES-LOC (H100 + 2×A6000, FSDP sharded) ==="
echo "GPUs: $CUDA_VISIBLE_DEVICES (3 devices)"
echo "Log: $LOG"

numactl --interleave=all \
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
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG"
