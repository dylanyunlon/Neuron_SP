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

# =============================================================================
# M076: 100-Config Experiment Sweep Launcher (400 lines)
# =============================================================================
# Extends the benchmark runner with the full experiment suite.
# Launches 100+ configurations across RQ1-RQ6 with proper
# environment setup, GPU assignment, and log collection.
#
# Architecture reference: CCCL ci/bench/bench.sh + submit_benchmark_job.sh
# Environment setup: follows llm4walking.sh pattern
# =============================================================================

# --- Experiment Suite Configuration ---

EXP_LOG_DIR="${SCRIPT_DIR}/desloc_experiment_logs"
EXP_FIGURE_DIR="${SCRIPT_DIR}/desloc_figures"
EXP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Model sizes for benchmarks
MODELS=("125M" "350M")
LARGE_MODELS=("1.3B")

# DES-LOC sync period sweeps
KX_SWEEP=(1 2 4 8 16 32 64 128)
KU_MULTIPLIERS=(1 2 3 4 6 8)
KV_MULTIPLIERS=(1 2 4 6 8 12)
BETA2_SWEEP=(0.95 0.98 0.99 0.995 0.999)
SEEDS=(42 137 256)

# Methods
METHODS_FULL=("ddp" "local_adam" "desloc")
OUTER_OPTS=("averaging" "nesterov")

setup_experiment_env() {
    # ---- Environment Setup (follows llm4walking.sh pattern) ----
    echo -e "${CYAN}Setting up experiment environment...${NC}"

    # Detect CUDA
    if ! command -v nvidia-smi &>/dev/null; then
        echo -e "${RED}ERROR: nvidia-smi not found${NC}"
        exit 1
    fi

    # Count GPUs
    NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
    echo -e "${GREEN}Detected ${NUM_GPUS} GPU(s)${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

    # Set CUDA environment
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s',' 0 $((NUM_GPUS-1)))}
    export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
    export PYTHONUNBUFFERED=1

    # DES-LOC specific
    export DESLOC_ENABLED=1
    export DESLOC_LOG_DIR="${EXP_LOG_DIR}"

    # Create output dirs
    mkdir -p "${EXP_LOG_DIR}"
    mkdir -p "${EXP_FIGURE_DIR}"
    mkdir -p "${OUTPUT_DIR}/logs"

    echo -e "${GREEN}Environment ready${NC}"
}

run_single_experiment() {
    # Run a single experiment configuration
    local BENCHMARK_ID="$1"
    local MODEL="$2"
    local KX="$3"
    local KU="$4"
    local KV="$5"
    local SEED="$6"
    local METHOD="${7:-desloc}"
    local STEPS="${8:-1000}"
    local EXTRA_ARGS="${9:-}"

    local RUN_ID="${BENCHMARK_ID}_${MODEL}_Kx${KX}_Ku${KU}_Kv${KV}_s${SEED}_${METHOD}"
    local LOG_FILE="${EXP_LOG_DIR}/${RUN_ID}.log"
    local JSON_LOG="${EXP_LOG_DIR}/${RUN_ID}.json"

    echo -e "${BLUE}[RUN] ${RUN_ID}${NC}"

    # Set DES-LOC environment for this run
    export DESLOC_KX=${KX}
    export DESLOC_KU=${KU}
    export DESLOC_KV=${KV}

    local GPU_ARG=""
    if [[ ${NUM_GPUS} -gt 1 ]]; then
        GPU_ARG="--nproc_per_node=${NUM_GPUS}"
    fi

    # Launch training
    local CMD=""
    if [[ ${NUM_GPUS} -gt 1 ]]; then
        CMD="torchrun ${GPU_ARG} ${BENCHMARK_PY}"
    else
        CMD="python3 ${BENCHMARK_PY}"
    fi

    ${CMD} \
        --model ${MODEL} \
        --batch-size 4 \
        --max-steps ${STEPS} \
        --Kx ${KX} --Ku ${KU} --Kv ${KV} \
        --methods ${METHOD} \
        --output "${OUTPUT_DIR}/${RUN_ID}" \
        ${EXTRA_ARGS} \
        2>&1 | tee "${LOG_FILE}"

    local EXIT_CODE=${PIPESTATUS[0]}

    if [[ ${EXIT_CODE} -eq 0 ]]; then
        echo -e "${GREEN}[DONE] ${RUN_ID}${NC}"
    else
        echo -e "${RED}[FAIL] ${RUN_ID} (exit ${EXIT_CODE})${NC}"
    fi

    return ${EXIT_CODE}
}

run_rq1_rate_of_change() {
    # RQ1: Rate of change sweep over beta2
    echo -e "${CYAN}━━━ RQ1: Empirical Rate of Change ━━━${NC}"

    for BETA2 in "${BETA2_SWEEP[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            run_single_experiment \
                "rq1_rate_of_change_adam" "125M" \
                32 96 192 ${SEED} "desloc" 1000 \
                "--beta2 ${BETA2}" || true
        done
    done
}

run_rq2_sync_frequency() {
    # RQ2: Sync frequency ablation
    echo -e "${CYAN}━━━ RQ2: Sync Frequency Ablation ━━━${NC}"

    # Sweep Kx with fixed Ku=3*Kx, Kv=6*Kx
    for KX in "${KX_SWEEP[@]}"; do
        local KU=$((KX * 3))
        local KV=$((KX * 6))
        for SEED in "${SEEDS[@]}"; do
            run_single_experiment \
                "rq2_sync_freq_Kx" "125M" \
                ${KX} ${KU} ${KV} ${SEED} "desloc" 1000 || true
        done
    done

    # Sweep Ku/Kv multipliers with fixed Kx=32
    for KU_M in "${KU_MULTIPLIERS[@]}"; do
        for KV_M in "${KV_MULTIPLIERS[@]}"; do
            local KU=$((32 * KU_M))
            local KV=$((32 * KV_M))
            for SEED in "${SEEDS[@]}"; do
                run_single_experiment \
                    "rq2_sync_freq_Ku_Kv" "125M" \
                    32 ${KU} ${KV} ${SEED} "desloc" 1000 || true
            done
        done
    done
}

run_rq3_comm_reduction() {
    # RQ3: Communication reduction comparison
    echo -e "${CYAN}━━━ RQ3: Communication Reduction ━━━${NC}"

    for MODEL in "${MODELS[@]}"; do
        for METHOD in "${METHODS_FULL[@]}"; do
            local KX=32
            local KU=96
            local KV=192
            if [[ "${METHOD}" == "ddp" ]]; then
                KX=1; KU=1; KV=1
            elif [[ "${METHOD}" == "local_adam" ]]; then
                KU=${KX}; KV=${KX}
            fi
            for SEED in "${SEEDS[@]}"; do
                run_single_experiment \
                    "rq3_comm_reduction_${MODEL}" "${MODEL}" \
                    ${KX} ${KU} ${KV} ${SEED} "${METHOD}" 1000 || true
            done
        done
    done
}

run_rq4_billion_scale() {
    # RQ4: Large-scale training
    echo -e "${CYAN}━━━ RQ4: Billion-Scale Training ━━━${NC}"

    for MODEL in "${LARGE_MODELS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            run_single_experiment \
                "rq4_billion_scale" "${MODEL}" \
                32 96 192 ${SEED} "desloc" 2000 || true
        done
    done
}

run_rq5_nesterov() {
    # RQ5: Nesterov outer optimizer
    echo -e "${CYAN}━━━ RQ5: Nesterov Outer Optimizer ━━━${NC}"

    for OUTER in "${OUTER_OPTS[@]}"; do
        for KX in 8 32 128; do
            local KU=$((KX * 3))
            local KV=$((KX * 6))
            for SEED in "${SEEDS[@]}"; do
                run_single_experiment \
                    "rq5_nesterov_${OUTER}" "125M" \
                    ${KX} ${KU} ${KV} ${SEED} "desloc" 1000 \
                    "--outer-optimizer ${OUTER}" || true
            done
        done
    done
}

run_rq6_muon() {
    # RQ6: Muon inner optimizer
    echo -e "${CYAN}━━━ RQ6: Muon Inner Optimizer ━━━${NC}"

    for MODEL in "${MODELS[@]}"; do
        for METHOD in "local_muon" "desloc_muon"; do
            local KX=32
            local KU=96
            if [[ "${METHOD}" == "local_muon" ]]; then
                KU=${KX}
            fi
            for SEED in "${SEEDS[@]}"; do
                run_single_experiment \
                    "rq6_muon_${MODEL}" "${MODEL}" \
                    ${KX} ${KU} 192 ${SEED} "desloc" 1000 \
                    "--inner-optimizer muon" || true
            done
        done
    done
}

generate_figures() {
    echo -e "${CYAN}━━━ Generating Figures ━━━${NC}"

    python3 -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from REAL_GPU_BENCHMARK import DESLOCFigureGenerator

gen = DESLOCFigureGenerator(
    log_dir='${EXP_LOG_DIR}',
    figure_dir='${EXP_FIGURE_DIR}')
generated = gen.generate_all()
print(f'Generated {len(generated)} figures')

table = gen.generate_summary_table()
with open('${EXP_FIGURE_DIR}/results_table.tex', 'w') as f:
    f.write(table)
print('Summary table saved')
"
}

count_experiment_configs() {
    # Count total configurations that will be run
    local COUNT=0

    # RQ1: beta2_sweep * seeds
    COUNT=$((COUNT + ${#BETA2_SWEEP[@]} * ${#SEEDS[@]}))

    # RQ2: Kx_sweep * seeds + Ku_mult * Kv_mult * seeds
    COUNT=$((COUNT + ${#KX_SWEEP[@]} * ${#SEEDS[@]}))
    COUNT=$((COUNT + ${#KU_MULTIPLIERS[@]} * ${#KV_MULTIPLIERS[@]} * ${#SEEDS[@]}))

    # RQ3: models * methods * seeds
    COUNT=$((COUNT + ${#MODELS[@]} * ${#METHODS_FULL[@]} * ${#SEEDS[@]}))

    # RQ4: large_models * seeds
    COUNT=$((COUNT + ${#LARGE_MODELS[@]} * ${#SEEDS[@]}))

    # RQ5: outer_opts * 3_Kx * seeds
    COUNT=$((COUNT + ${#OUTER_OPTS[@]} * 3 * ${#SEEDS[@]}))

    # RQ6: models * 2_methods * seeds
    COUNT=$((COUNT + ${#MODELS[@]} * 2 * ${#SEEDS[@]}))

    echo ${COUNT}
}

run_full_suite() {
    setup_experiment_env

    local TOTAL=$(count_experiment_configs)
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║   DES-LOC FULL EXPERIMENT SUITE                         ║${NC}"
    echo -e "${CYAN}║   Total configurations: ${TOTAL}                            ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════╝${NC}"

    local START_TIME=$(date +%s)

    run_rq1_rate_of_change
    run_rq2_sync_frequency
    run_rq3_comm_reduction
    run_rq4_billion_scale
    run_rq5_nesterov
    run_rq6_muon

    generate_figures

    local END_TIME=$(date +%s)
    local ELAPSED=$((END_TIME - START_TIME))
    local HOURS=$((ELAPSED / 3600))
    local MINS=$(((ELAPSED % 3600) / 60))

    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║   FULL SUITE COMPLETE                                   ║${NC}"
    echo -e "${GREEN}║   Time: ${HOURS}h ${MINS}m                                        ║${NC}"
    echo -e "${GREEN}║   Logs: ${EXP_LOG_DIR}                                  ║${NC}"
    echo -e "${GREEN}║   Figures: ${EXP_FIGURE_DIR}                            ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
}

run_quick_suite() {
    setup_experiment_env
    echo -e "${CYAN}Running quick suite (3 benchmarks)...${NC}"

    # Quick: one config per RQ
    run_single_experiment "rq1_rate_of_change_adam" "125M" 32 96 192 42 "desloc" 500 || true
    run_single_experiment "rq3_comm_reduction_125M" "125M" 32 96 192 42 "desloc" 500 || true
    run_single_experiment "rq6_muon_125M" "125M" 32 96 192 42 "desloc" 500 || true

    generate_figures
}

# --- Extended CLI ---
if [[ "${1:-}" == "--full-suite" ]]; then
    run_full_suite
    exit 0
elif [[ "${1:-}" == "--quick-suite" ]]; then
    run_quick_suite
    exit 0
elif [[ "${1:-}" == "--rq1" ]]; then
    setup_experiment_env
    run_rq1_rate_of_change
    generate_figures
    exit 0
elif [[ "${1:-}" == "--rq2" ]]; then
    setup_experiment_env
    run_rq2_sync_frequency
    generate_figures
    exit 0
elif [[ "${1:-}" == "--rq3" ]]; then
    setup_experiment_env
    run_rq3_comm_reduction
    generate_figures
    exit 0
elif [[ "${1:-}" == "--rq4" ]]; then
    setup_experiment_env
    run_rq4_billion_scale
    generate_figures
    exit 0
elif [[ "${1:-}" == "--rq5" ]]; then
    setup_experiment_env
    run_rq5_nesterov
    generate_figures
    exit 0
elif [[ "${1:-}" == "--rq6" ]]; then
    setup_experiment_env
    run_rq6_muon
    generate_figures
    exit 0
elif [[ "${1:-}" == "--count" ]]; then
    echo "Total experiment configurations: $(count_experiment_configs)"
    exit 0
elif [[ "${1:-}" == "--generate-figures" ]]; then
    generate_figures
    exit 0
fi

# =============================================================================
# End M076
# =============================================================================
