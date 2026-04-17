#!/bin/bash
# =============================================================================
# M061: DES-LOC Real GPU Benchmark Runner (400 lines)
# =============================================================================
# Launches real distributed training with DES-LOC optimizer.
# Uses REAL_GPU_BENCHMARK.py — no simulation, no FULL_PATCH.py.
#
# Usage:
#   ./run_desloc_benchmark.sh                    # Single GPU test
#   ./run_desloc_benchmark.sh --gpus 2           # 2-GPU distributed
#   ./run_desloc_benchmark.sh --ablation         # Kx/Ku/Kv sweep
#   ./run_desloc_benchmark.sh --all              # Full benchmark suite
#
# Architecture reference: CCCL ci/bench/bench.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_PY="${SCRIPT_DIR}/REAL_GPU_BENCHMARK.py"
OUTPUT_DIR="${SCRIPT_DIR}/real_benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Defaults
NUM_GPUS=1
MODEL_SIZE="125M"
BATCH_SIZE=4
GRAD_ACCUM=8
MAX_STEPS=200
METHODS="DDP LocalAdam DESLOC"
KX=32
KU=96
KV=192
RUN_ABLATION=false
RUN_ALL=false
DRY_RUN=false
MASTER_PORT=29500

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)      NUM_GPUS="$2"; shift 2;;
        --model)     MODEL_SIZE="$2"; shift 2;;
        --batch)     BATCH_SIZE="$2"; shift 2;;
        --accum)     GRAD_ACCUM="$2"; shift 2;;
        --steps)     MAX_STEPS="$2"; shift 2;;
        --methods)   METHODS="$2"; shift 2;;
        --Kx)        KX="$2"; shift 2;;
        --Ku)        KU="$2"; shift 2;;
        --Kv)        KV="$2"; shift 2;;
        --output)    OUTPUT_DIR="$2"; shift 2;;
        --ablation)  RUN_ABLATION=true; shift;;
        --all)       RUN_ALL=true; shift;;
        --dry-run)   DRY_RUN=true; shift;;
        --port)      MASTER_PORT="$2"; shift 2;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Training Options:"
            echo "  --gpus N        Number of GPUs (default: 1)"
            echo "  --model SIZE    Model size: 125M, 350M, 700M, 1.3B (default: 125M)"
            echo "  --batch N       Batch size per GPU (default: 4)"
            echo "  --accum N       Gradient accumulation steps (default: 8)"
            echo "  --steps N       Max training steps (default: 200)"
            echo "  --methods STR   Space-separated methods (default: 'DDP LocalAdam DESLOC')"
            echo ""
            echo "DES-LOC Options:"
            echo "  --Kx N          Parameter sync period (default: 32)"
            echo "  --Ku N          First moment sync period (default: 96)"
            echo "  --Kv N          Second moment sync period (default: 192)"
            echo ""
            echo "Modes:"
            echo "  --ablation      Run Kx/Ku/Kv ablation sweep"
            echo "  --all           Run full benchmark suite"
            echo "  --dry-run       Print commands without executing"
            echo ""
            echo "Output:"
            echo "  --output DIR    Output directory (default: ./real_benchmark_results)"
            echo "  --port N        Master port for distributed (default: 29500)"
            exit 0;;
        *) echo -e "${RED}Unknown option: $1${NC}"; exit 1;;
    esac
done

# Verify prerequisites
echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║   DES-LOC Real GPU Benchmark Suite                      ║"
echo "║   Model: ${MODEL_SIZE}  GPUs: ${NUM_GPUS}  Steps: ${MAX_STEPS}"
echo "║   DES-LOC: Kx=${KX} Ku=${KU} Kv=${KV}"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

if [ ! -f "$BENCHMARK_PY" ]; then
    echo -e "${RED}ERROR: $BENCHMARK_PY not found${NC}"
    exit 1
fi

python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
    echo -e "${RED}ERROR: CUDA not available${NC}"; exit 1
}

AVAILABLE_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo -e "${CYAN}Available GPUs: ${AVAILABLE_GPUS}${NC}"
if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    NUM_GPUS=$AVAILABLE_GPUS
fi

python3 -c "
import torch
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name} ({p.total_memory/1e9:.1f} GB)')
"

mkdir -p "$OUTPUT_DIR"

run_benchmark() {
    local methods="$1"
    local suffix="$2"
    local extra_args="${3:-}"
    local run_dir="${OUTPUT_DIR}/${suffix}_${TIMESTAMP}"
    mkdir -p "$run_dir"

    echo ""
    echo -e "${GREEN}━━━ Running: ${suffix} ━━━${NC}"
    echo -e "${GREEN}  Methods: ${methods} | Output: ${run_dir}${NC}"

    local CMD=""
    if [ "$NUM_GPUS" -gt 1 ]; then
        CMD="torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT}"
    else
        CMD="python3"
    fi

    CMD="$CMD $BENCHMARK_PY \
        --model_size $MODEL_SIZE \
        --batch_size $BATCH_SIZE \
        --grad_accum $GRAD_ACCUM \
        --max_steps $MAX_STEPS \
        --Kx $KX --Ku $KU --Kv $KV \
        --output $run_dir \
        --methods $methods \
        $extra_args"

    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN] $CMD${NC}"
    else
        echo -e "${CYAN}$ $CMD${NC}"
        eval $CMD 2>&1 | tee "${run_dir}/train.log"
        echo -e "${GREEN}  COMPLETE${NC}"
    fi
}

if [ "$RUN_ALL" = true ]; then
    echo -e "${BLUE}Running FULL benchmark suite...${NC}"

    run_benchmark "DDP LocalAdam DESLOC" "standard"

    echo -e "${BLUE}=== Kx Ablation ===${NC}"
    for kx in 4 8 16 32 64 128; do
        run_benchmark "DESLOC" "ablation_Kx${kx}" "--Kx $kx"
    done

    echo -e "${BLUE}=== Ku Ablation ===${NC}"
    for ku in 16 32 64 96 192 384; do
        run_benchmark "DESLOC" "ablation_Ku${ku}" "--Ku $ku"
    done

    echo -e "${BLUE}=== Kv Ablation ===${NC}"
    for kv in 32 64 96 192 384 768; do
        run_benchmark "DESLOC" "ablation_Kv${kv}" "--Kv $kv"
    done

    echo -e "${BLUE}=== Model Scaling ===${NC}"
    for model in 125M 350M; do
        run_benchmark "DDP DESLOC" "scale_${model}" "--model_size $model"
    done

elif [ "$RUN_ABLATION" = true ]; then
    echo -e "${BLUE}Running ablation sweep...${NC}"
    for kx in 4 16 32 64 128; do
        run_benchmark "DESLOC" "ablation_Kx${kx}" "--Kx $kx"
    done
    for ku in 32 96 192; do
        run_benchmark "DESLOC" "ablation_Ku${ku}" "--Ku $ku"
    done
    for kv in 96 192 384; do
        run_benchmark "DESLOC" "ablation_Kv${kv}" "--Kv $kv"
    done

else
    run_benchmark "$METHODS" "benchmark"
fi

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   BENCHMARK COMPLETE                                    ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Results: $OUTPUT_DIR"
find "$OUTPUT_DIR" -name "*.json" -o -name "*.jsonl" -o -name "*.csv" 2>/dev/null | wc -l
echo " data files generated"
