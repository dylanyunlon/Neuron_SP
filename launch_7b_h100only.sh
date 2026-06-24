#!/usr/bin/env bash
# launch_7b_h100only.sh — single-GPU H100 NVL (GPU2) baseline for MFU comparison
#
# Purpose: establish a clean single-GPU MFU baseline on the H100 NVL (96 GB VRAM).
#   No NCCL, no FSDP sharding, no inter-GPU communication overhead — pure compute.
#   Use this to compare against multi-GPU DES-LOC / SP runs.
#
# GPU layout on ags1:
#   GPU0: Blackwell RTX PRO 6000 (SM120)
#   GPU1: Blackwell RTX PRO 6000 (SM120)
#   GPU2: H100 NVL  (96 GB, SM9.0)  ← this script
#   GPU3: A6000 (47 GB, SM8.6)
#   GPU4: A6000 (47 GB, SM8.6)
#
# micro_batch_size=4 is conservative; bump to 8 if VRAM allows at your seq-len.

set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

# ── Single GPU: H100 NVL only ─────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=2

# No NCCL needed for single-GPU — unset any inherited vars that could interfere
unset NCCL_P2P_DISABLE   2>/dev/null || true
unset NCCL_IB_DISABLE    2>/dev/null || true
unset NCCL_SOCKET_IFNAME 2>/dev/null || true
unset NCCL_DEBUG         2>/dev/null || true
unset NCCL_TIMEOUT       2>/dev/null || true
# Silence the "no NCCL" warning that torchrun emits on single-process launches
export TORCH_DISTRIBUTED_DEBUG=OFF

# Performance
export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

# Eager mode — no JIT compilation stall on first forward
export TORCHDYNAMO_DISABLE=1
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/7b_h100only_${TIMESTAMP}.log"

EXTRA_ARGS=("$@")

# ── Dry-run support ───────────────────────────────────────────────────────────
if [[ " ${EXTRA_ARGS[*]:-} " == *" --dry-run "* ]]; then
    echo "=== DRY RUN (single GPU: H100 NVL, GPU2) ==="
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader -i 2 2>/dev/null || true
    EXTRA_ARGS=("${EXTRA_ARGS[@]/--dry-run/}" --steps 3 --log-every 1 --save-every 0)
fi
# ─────────────────────────────────────────────────────────────────────────────

echo "=== Neuron_SP 7B — H100 NVL single-GPU MFU baseline ==="
echo "Log:  $LOG"
echo "GPU:  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  (H100 NVL, 96 GB)"
echo "Note: micro_batch_size=4 (~24 GB at seq_len=2048). Bump to 8 for higher MFU."
echo ""

# nproc_per_node=1 → single process, no NCCL collective ops
torchrun --nproc_per_node=1 --master_port=29600 \
    run_pretrain.py \
    --model-size 7b \
    --steps 100000 \
    --batch-size 4 \
    --seq-len 2048 \
    --gradient-checkpointing \
    --log-every 10 \
    --save-every 500 \
    --checkpoint-dir checkpoints/7b_h100only_${TIMESTAMP} \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG"
