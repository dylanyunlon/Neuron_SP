#!/usr/bin/env bash
# launch_7b_5gpu.sh — ags1 5-GPU DES-LOC 7B pretrain (Blackwell + H100 + 2×A6000)
#
# Requires PyTorch cu126+ to support Blackwell RTX PRO 6000 (SM120).
# This script guards against older PyTorch builds and exits with an error
# if torch.version.cuda < 12.6.
#
# GPU layout (all 5):
#   GPU0: Blackwell RTX PRO 6000 (SM120) — requires cu126+
#   GPU1: Blackwell RTX PRO 6000 (SM120) — requires cu126+
#   GPU2: H100 NVL (93GB, SM9.0)
#   GPU3: A6000   (47GB, SM8.6)
#   GPU4: A6000   (47GB, SM8.6)
#
# To install the required PyTorch build:
#   pip install torch --index-url https://download.pytorch.org/whl/cu126

set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

# ── CUDA version guard ────────────────────────────────────────────────────────
echo "=== Checking PyTorch CUDA version (requires >= 12.6 for Blackwell) ==="
if ! python3 -c "
import sys, torch
cuda_ver = torch.version.cuda or ''
print(f'  torch={torch.__version__}  cuda={cuda_ver}')
if not cuda_ver:
    print('ERROR: torch.version.cuda is None — CPU-only build detected.', file=sys.stderr)
    sys.exit(1)
major, minor = (int(x) for x in cuda_ver.split('.')[:2])
if (major, minor) < (12, 6):
    print(f'ERROR: CUDA {cuda_ver} < 12.6. Blackwell (SM120) requires cu126+.', file=sys.stderr)
    print('  Fix: pip install torch --index-url https://download.pytorch.org/whl/cu126', file=sys.stderr)
    sys.exit(1)
print('  OK: CUDA version satisfies >= 12.6')
"; then
    echo "ABORT: PyTorch CUDA version check failed. See error above." >&2
    exit 1
fi
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/7b_pretrain_5gpu_${TIMESTAMP}.log"

# All 5 GPUs (Blackwell × 2 + H100 + A6000 × 2)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
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
    echo "=== DRY RUN (5-GPU: 2×Blackwell + H100 + 2×A6000) ==="
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader -i 0,1,2,3,4 2>/dev/null
    EXTRA_ARGS=("${EXTRA_ARGS[@]/--dry-run/}" --steps 3 --log-every 1 --save-every 0)
fi

echo "=== Neuron_SP 7B DES-LOC (5-GPU: 2×Blackwell + H100 + 2×A6000) ==="
echo "Log: $LOG"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

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
    --checkpoint-dir checkpoints/7b_5gpu_${TIMESTAMP} \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG"
