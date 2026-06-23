#!/usr/bin/env bash
# launch_7b.sh — ags1 5-GPU DES-LOC 7B pretrain
# Topology: GPU0(A6000) GPU1(BW6000) GPU2(H100) → NUMA0
#           GPU3(A6000) GPU4(BW6000)             → NUMA1
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/7b_pretrain_${TIMESTAMP}.log"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4
# No NVLink — disable P2P, use SHM
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=WARN
# 30min timeout: H100 finishes 20x faster than A6000, slow rank needs time
export NCCL_TIMEOUT=1800000
export OMP_NUM_THREADS=8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# A6000 48GB is tight — reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256

EXTRA_ARGS=("$@")

# --dry-run: 3 steps then exit
if [[ " ${EXTRA_ARGS[*]:-} " == *" --dry-run "* ]]; then
    echo "=== DRY RUN ==="
    nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.current --format=csv,noheader 2>/dev/null
    echo ""
    EXTRA_ARGS=("${EXTRA_ARGS[@]/--dry-run/}" --steps 3 --log-every 1 --save-every 0)
fi

echo "=== Neuron_SP 7B DES-LOC ==="
echo "Log: $LOG"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# NUMA-aware launch: bind memory to local node
numactl --interleave=all \
torchrun --nproc_per_node=5 --master_port=29500 \
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
