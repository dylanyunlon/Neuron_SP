#!/usr/bin/env bash
# launch_7b.sh — auto-detect compatible GPUs, skip Blackwell SM120
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

# Auto-detect: pick only GPUs with SM < 12.0 (skip Blackwell)
COMPAT_GPUS=$(python3 -c "
import torch
ids = []
for i in range(torch.cuda.device_count()):
    cap = torch.cuda.get_device_capability(i)
    if cap[0] < 12:  # skip SM 12.0 (Blackwell)
        ids.append(str(i))
print(','.join(ids))
")

NUM_GPUS=$(echo "$COMPAT_GPUS" | tr ',' '\n' | wc -l)
echo "Auto-detected compatible GPUs: $COMPAT_GPUS ($NUM_GPUS devices)"

export CUDA_VISIBLE_DEVICES="$COMPAT_GPUS"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_ENABLE_MONITORING=0
# Disable torch.compile / inductor — JIT compilation of 7B FSDP model
# spawns 32 compile workers and takes 10-20 min on first forward.
# Run eager mode first to verify correctness, enable compile later.
export TORCHDYNAMO_DISABLE=1

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/7b_pretrain_${TIMESTAMP}.log"

echo "=== Neuron_SP 7B DES-LOC (FSDP, ${NUM_GPUS} GPUs) ==="
echo "Log: $LOG"

torchrun --nproc_per_node="$NUM_GPUS" --master_port=29500 \
    run_pretrain.py \
    --model-size 7b \
    --steps 100000 \
    --batch-size 1 \
    --seq-len 1024 \
    --use-desloc \
    --gradient-checkpointing \
    --log-every 1 \
    --save-every 500 \
    --checkpoint-dir checkpoints/7b_${TIMESTAMP} \
    2>&1 | tee "$LOG"
