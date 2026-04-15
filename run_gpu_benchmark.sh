#!/bin/bash
# =============================================================================
# DES-LOC GPU Benchmark - Distributed Training Script
# =============================================================================
# For running on Yotta A100 or other GPU clusters
# Usage: ./run_gpu_benchmark.sh
# =============================================================================

set -e

# Configuration
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29500}

# Model configurations
MODEL_SIZE=${MODEL_SIZE:-"125M"}
MICRO_BATCH=${MICRO_BATCH:-4}
GRAD_ACCUM=${GRAD_ACCUM:-8}
SEQ_LEN=${SEQ_LEN:-2048}
MAX_STEPS=${MAX_STEPS:-1000}

# DES-LOC parameters
KX=${KX:-32}
KU_RATIO=${KU_RATIO:-3}
KV_RATIO=${KV_RATIO:-6}

# Output
OUTPUT_DIR=${OUTPUT_DIR:-"./gpu_benchmark_results"}
mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "DES-LOC GPU Benchmark"
echo "========================================="
echo "Nodes: $NNODES x $NPROC_PER_NODE GPUs"
echo "Model: $MODEL_SIZE"
echo "Batch: $MICRO_BATCH x $GRAD_ACCUM"
echo "Sequence Length: $SEQ_LEN"
echo "Max Steps: $MAX_STEPS"
echo "DES-LOC: Kx=$KX, Ku=$((KX*KU_RATIO)), Kv=$((KX*KV_RATIO))"
echo "Output: $OUTPUT_DIR"
echo "========================================="

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found, GPU benchmark may fail"
fi

# Show GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "Could not query GPU info"
echo ""

# Run with torchrun
echo "Starting distributed training..."
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    FULL_PATCH.py \
    --output "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "========================================="
echo "GPU Benchmark Complete!"
echo "========================================="
echo "Results: $OUTPUT_DIR"
echo "Log: $OUTPUT_DIR/training.log"
