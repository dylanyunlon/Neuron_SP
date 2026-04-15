#!/bin/bash
# =============================================================================
# DES-LOC Real GPU Benchmark - Launch Script
# =============================================================================
# NO SIMULATION. NO FALLBACK. FAIL HARD.
# =============================================================================

set -e  # Exit on any error

echo "=============================================="
echo "DES-LOC REAL GPU BENCHMARK"
echo "=============================================="

# Check CUDA
nvidia-smi || { echo "FATAL: nvidia-smi failed"; exit 1; }

# Check PyTorch
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || { echo "FATAL: PyTorch CUDA check failed"; exit 1; }

# Configuration
MODEL_SIZE=${MODEL_SIZE:-"125M"}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-8}
MAX_STEPS=${MAX_STEPS:-500}
KX=${KX:-32}
KU=${KU:-96}
KV=${KV:-192}
OUTPUT_DIR=${OUTPUT_DIR:-"./real_benchmark_results"}

# Get number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected GPUs: $NUM_GPUS"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Running distributed training with $NUM_GPUS GPUs..."
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        REAL_GPU_BENCHMARK.py \
        --model_size $MODEL_SIZE \
        --batch_size $BATCH_SIZE \
        --grad_accum $GRAD_ACCUM \
        --max_steps $MAX_STEPS \
        --Kx $KX \
        --Ku $KU \
        --Kv $KV \
        --output $OUTPUT_DIR \
        --methods DDP LocalAdam DESLOC
else
    echo "Running single GPU training..."
    python3 REAL_GPU_BENCHMARK.py \
        --model_size $MODEL_SIZE \
        --batch_size $BATCH_SIZE \
        --grad_accum $GRAD_ACCUM \
        --max_steps $MAX_STEPS \
        --Kx $KX \
        --Ku $KU \
        --Kv $KV \
        --output $OUTPUT_DIR \
        --methods DDP LocalAdam DESLOC
fi

echo ""
echo "=============================================="
echo "BENCHMARK COMPLETE"
echo "=============================================="
echo "Results: $OUTPUT_DIR"
