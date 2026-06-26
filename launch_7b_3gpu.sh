#!/usr/bin/env bash
# launch_7b_3gpu.sh — ags1 3-GPU DES-LOC 7B pretrain (H100 + 2×A6000)
#
# Blackwell RTX PRO 6000 (SM120) excluded: PyTorch 2.7.1+cu118 only supports up to SM90.
# Upgrade to PyTorch cu126+ to enable Blackwell. Until then, train on 3 GPUs:
#   GPU2: H100 NVL (93GB, SM9.0) — primary compute
#   GPU3: A6000   (47GB, SM8.6) — secondary
#   GPU4: A6000   (47GB, SM8.6) — secondary
#
# To enable all 5 GPUs with Blackwell:
#   pip install torch --index-url https://download.pytorch.org/whl/cu126
#   Then use launch_7b.sh (CUDA_VISIBLE_DEVICES=0,1,2,3,4)

set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/7b_pretrain_3gpu_${TIMESTAMP}.log"

# Only H100 + 2×A6000 (skip Blackwell SM120)
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
    echo "=== DRY RUN (3-GPU: H100 + 2×A6000) ==="
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader -i 2,3,4 2>/dev/null
    EXTRA_ARGS=("${EXTRA_ARGS[@]/--dry-run/}" --steps 3 --log-every 1 --save-every 0)
fi

echo "=== Neuron_SP 7B DES-LOC (3-GPU: H100+2×A6000) ==="
echo "Log: $LOG"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Note: Blackwell GPUs excluded (PyTorch cu118 < SM120). Upgrade to cu126+ for 5-GPU."

# Auto-detect real data
DATA_PATH="data/commitpack_train.npy"
if [ ! -f "$DATA_PATH" ]; then
    echo "⚠ Real data not found at $DATA_PATH — using synthetic. Run: bash prepare_data.sh"
    DATA_PATH=""
fi

DATA_ARGS=()
if [ -n "$DATA_PATH" ]; then
    DATA_ARGS=(--data-path "$DATA_PATH")
    echo "Data: $DATA_PATH"
else
    echo "Data: synthetic"
fi

numactl --interleave=all \
torchrun --nproc_per_node=3 --master_port=29500 \
    run_pretrain.py \
    --model-size 7b \
    --steps 100000 \
    --batch-size 1 \
    --seq-len 4096 \
    --use-desloc \
    --gradient-checkpointing \
    --log-every 10 \
    --save-every 500 \
    --checkpoint-dir checkpoints/7b_3gpu_${TIMESTAMP} \
    "${DATA_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG"
