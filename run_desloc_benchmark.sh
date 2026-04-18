#!/bin/bash
# =============================================================================
# DES-LOC Benchmark Suite - Launch Script
# =============================================================================
# Usage: ./run_desloc_benchmark.sh [OPTIONS]
#
# Options:
#   --all           Run all 7 figures (default)
#   --figure N      Run specific figure (1-7)
#   --gpu           Run GPU benchmarks (requires CUDA)
#   --output DIR    Output directory
#   --help          Show this help
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
OUTPUT_DIR="./desloc_benchmark_results"
RUN_ALL=true
RUN_GPU=false
FIGURES=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            RUN_ALL=true
            shift
            ;;
        --figure)
            RUN_ALL=false
            FIGURES+=("figure$2")
            shift 2
            ;;
        --gpu)
            RUN_GPU=true
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --all           Run all 7 figures (default)"
            echo "  --figure N      Run specific figure (1-7)"
            echo "  --gpu           Run GPU benchmarks"
            echo "  --output DIR    Output directory"
            echo "  --help          Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Banner
echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     ██████╗ ███████╗███████╗      ██╗      ██████╗  ██████╗   ║"
echo "║     ██╔══██╗██╔════╝██╔════╝      ██║     ██╔═══██╗██╔════╝   ║"
echo "║     ██║  ██║█████╗  ███████╗█████╗██║     ██║   ██║██║        ║"
echo "║     ██║  ██║██╔══╝  ╚════██║╚════╝██║     ██║   ██║██║        ║"
echo "║     ██████╔╝███████╗███████║      ███████╗╚██████╔╝╚██████╗   ║"
echo "║     ╚═════╝ ╚══════╝╚══════╝      ╚══════╝ ╚═════╝  ╚═════╝   ║"
echo "║                                                               ║"
echo "║     Desynced Low-Communication Optimizer Benchmark Suite      ║"
echo "║                         v1.0.0                                ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check Python
echo -e "${YELLOW}Checking dependencies...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 not found${NC}"
    exit 1
fi

# Check required packages
python3 -c "import numpy, matplotlib" 2>/dev/null || {
    echo -e "${YELLOW}Installing required packages...${NC}"
    pip install numpy matplotlib --quiet
}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run benchmark
echo -e "${GREEN}Starting DES-LOC Benchmark Suite...${NC}"
echo "Output directory: $OUTPUT_DIR"
echo ""

if [ "$RUN_ALL" = true ]; then
    echo -e "${BLUE}Running all 7 figures...${NC}"
    python3 FULL_PATCH.py --output "$OUTPUT_DIR"
else
    echo -e "${BLUE}Running figures: ${FIGURES[*]}${NC}"
    python3 FULL_PATCH.py --figures "${FIGURES[@]}" --output "$OUTPUT_DIR"
fi

# Summary
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Benchmark Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
find "$OUTPUT_DIR" -name "*.png" -o -name "*.pdf" | head -20
echo ""
echo -e "${YELLOW}View summary: cat $OUTPUT_DIR/SUMMARY.md${NC}"

# ═══════════════════════════════════════════════════════════════
# DES-LOC Experiment Pipeline (M196)
# Full benchmark automation: detect → check → benchmark → plot
# ═══════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/desloc_experiment_logs"
FIG_DIR="${SCRIPT_DIR}/desloc_figures"
SEEDS=(42 137 2024)
MODELS=("125M" "350M")
NC='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'

mkdir -p "${LOG_DIR}" "${FIG_DIR}"

detect_hardware() {
    echo -e "${CYAN}━━━ Hardware Detection ━━━${NC}"
    python3 -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {p.name}, {p.total_mem/1024**3:.1f}GB, SM={p.multi_processor_count}')
else:
    print('No CUDA GPUs detected')
"
    echo ""
}

run_single_experiment() {
    local BENCH_ID=$1
    local MODEL=$2
    local KX=$3
    local KU=$4
    local KV=$5
    local SEED=$6
    local METHOD=$7
    local STEPS=$8
    local EXTRA_ARGS="${9:-}"

    local LOG_FILE="${LOG_DIR}/${BENCH_ID}_Kx${KX}_seed${SEED}_$(date +%s).log"

    echo -e "${GREEN}[RUN] ${BENCH_ID} Kx=${KX} Ku=${KU} Kv=${KV} seed=${SEED}${NC}"

    DESLOC_ENABLED=1 \
    DESLOC_KX=${KX} \
    DESLOC_KU=${KU} \
    DESLOC_KV=${KV} \
    PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}" \
    python3 -u "${SCRIPT_DIR}/REAL_GPU_BENCHMARK.py" \
        --benchmark-id "${BENCH_ID}" \
        --model "${MODEL}" \
        --method "${METHOD}" \
        --Kx ${KX} --Ku ${KU} --Kv ${KV} \
        --seed ${SEED} \
        --total-steps ${STEPS} \
        --log-dir "${LOG_DIR}" \
        ${EXTRA_ARGS} \
        2>&1 | tee "${LOG_FILE}"

    local EXIT_CODE=$?
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo -e "${GREEN}[OK] ${BENCH_ID} completed${NC}"
    else
        echo -e "${RED}[FAIL] ${BENCH_ID} exit code ${EXIT_CODE}${NC}"
    fi
    return ${EXIT_CODE}
}

# ── RQ1: Half-life measurement ──────────────────────────────
run_rq1_halflife() {
    echo -e "${CYAN}━━━ RQ1: Half-Life Measurement (Section 5.1) ━━━${NC}"
    for SEED in "${SEEDS[@]}"; do
        run_single_experiment \
            "rq1_half_life_125M" "125M" \
            32 96 192 ${SEED} "desloc" 5000 || true
    done
}

# ── RQ2: Sync frequency sweep ───────────────────────────────
run_rq2_sync_sweep() {
    echo -e "${CYAN}━━━ RQ2: Sync Frequency Sweep (Section 5.2) ━━━${NC}"
    for KX in 1 2 4 8 16 32 64 128; do
        local KU=$((KX * 3))
        local KV=$((KX * 6))
        for SEED in "${SEEDS[@]}"; do
            run_single_experiment \
                "rq2_Kx_sweep" "125M" \
                ${KX} ${KU} ${KV} ${SEED} "desloc" 5000 || true
        done
    done
}

# ── RQ3: Communication reduction ────────────────────────────
run_rq3_comm_reduction() {
    echo -e "${CYAN}━━━ RQ3: Communication Reduction (Section 5.3) ━━━${NC}"
    for MODEL in "${MODELS[@]}"; do
        local STEPS=5000
        if [ "${MODEL}" = "350M" ]; then STEPS=3000; fi

        # DDP baseline (Kx=1)
        for SEED in "${SEEDS[@]}"; do
            run_single_experiment \
                "rq3_ddp_${MODEL}" "${MODEL}" \
                1 1 1 ${SEED} "ddp" ${STEPS} || true
        done

        # Local Adam (Kx=Ku=Kv)
        for SEED in "${SEEDS[@]}"; do
            run_single_experiment \
                "rq3_local_adam_${MODEL}" "${MODEL}" \
                32 32 32 ${SEED} "local_adam" ${STEPS} || true
        done

        # DES-LOC (Kx < Ku < Kv)
        for SEED in "${SEEDS[@]}"; do
            run_single_experiment \
                "rq3_desloc_${MODEL}" "${MODEL}" \
                32 96 192 ${SEED} "desloc" ${STEPS} || true
        done
    done
}

# ── RQ5: Nesterov outer optimizer ────────────────────────────
OUTER_OPTS=("averaging" "nesterov")

run_rq5_nesterov() {
    echo -e "${CYAN}━━━ RQ5: Nesterov Outer Optimizer (Section 5.5) ━━━${NC}"

    for OUTER in "${OUTER_OPTS[@]}"; do
        for KX in 8 32 128; do
            local KU=$((KX * 3))
            local KV=$((KX * 6))
            for SEED in "${SEEDS[@]}"; do
                run_single_experiment \
                    "rq5_nesterov_${OUTER}" "125M" \
                    ${KX} ${KU} ${KV} ${SEED} "desloc" 5000 \
                    "--outer-optimizer ${OUTER}" || true
            done
        done
    done

    # Nesterov momentum sweep (momentum = 0.7, 0.9, 0.95)
    for MOM in 0.7 0.9 0.95; do
        for SEED in "${SEEDS[@]}"; do
            run_single_experiment \
                "rq5_nesterov_mom${MOM}" "125M" \
                32 96 192 ${SEED} "desloc" 5000 \
                "--outer-optimizer nesterov --nesterov-momentum ${MOM}" || true
        done
    done
}

# ── RQ6: Muon inner optimizer ────────────────────────────────
run_rq6_muon() {
    echo -e "${CYAN}━━━ RQ6: Muon Inner Optimizer (Section 5.6) ━━━${NC}"

    for MODEL in "${MODELS[@]}"; do
        for METHOD in "local_muon" "desloc_muon"; do
            local KX=32
            local KU=96
            if [ "${METHOD}" = "local_muon" ]; then
                KU=${KX}
            fi
            for SEED in "${SEEDS[@]}"; do
                run_single_experiment \
                    "rq6_muon_${MODEL}" "${MODEL}" \
                    ${KX} ${KU} ${KU} ${SEED} "desloc" 5000 \
                    "--inner-optimizer muon --muon-compat" || true
            done
        done
    done

    # Muon Ku sweep (Ku = Kx, 3*Kx, 6*Kx)
    for KU_MULT in 1 3 6; do
        local KX=32
        local KU=$((KX * KU_MULT))
        for SEED in "${SEEDS[@]}"; do
            run_single_experiment \
                "rq6_muon_Ku_${KU_MULT}x" "125M" \
                ${KX} ${KU} ${KU} ${SEED} "desloc" 5000 \
                "--inner-optimizer muon --muon-compat" || true
        done
    done
}

# ── Figure Generation ────────────────────────────────────────
generate_figures() {
    echo -e "${CYAN}━━━ Generating Figures ━━━${NC}"

    python3 -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from REAL_GPU_BENCHMARK import DESLOCFigureGenerator
gen = DESLOCFigureGenerator('${LOG_DIR}', '${FIG_DIR}')
paths = gen.generate_all()
for p in paths:
    print(f'  Generated: {p}')
if not paths:
    print('  No figures generated (no experiment data found)')
"
}

# ── Summary ──────────────────────────────────────────────────
print_summary() {
    echo -e "${CYAN}━━━ Experiment Summary ━━━${NC}"
    echo "Log directory: ${LOG_DIR}"
    echo "Figure directory: ${FIG_DIR}"
    echo "Total log files: $(ls ${LOG_DIR}/*.log 2>/dev/null | wc -l)"
    echo "Total figures: $(ls ${FIG_DIR}/*.json 2>/dev/null | wc -l)"
}

# ── Main ─────────────────────────────────────────────────────
run_all() {
    echo -e "${CYAN}═══ DES-LOC Full Benchmark Suite ═══${NC}"
    echo "Started: $(date)"
    echo ""

    detect_hardware
    run_rq1_halflife
    run_rq2_sync_sweep
    run_rq3_comm_reduction
    run_rq5_nesterov
    run_rq6_muon
    generate_figures
    print_summary

    echo ""
    echo -e "${GREEN}═══ All experiments complete ═══${NC}"
    echo "Finished: $(date)"
}

run_quick() {
    echo -e "${CYAN}═══ DES-LOC Quick Benchmark (smoke test) ═══${NC}"
    detect_hardware
    run_single_experiment "smoke_ddp" "125M" 1 1 1 42 "ddp" 100 || true
    run_single_experiment "smoke_desloc" "125M" 8 24 48 42 "desloc" 100 || true
    run_single_experiment "smoke_nesterov" "125M" 8 24 48 42 "desloc" 100 "--outer-optimizer nesterov" || true
    run_single_experiment "smoke_muon" "125M" 8 24 24 42 "desloc" 100 "--inner-optimizer muon --muon-compat" || true
    generate_figures
    print_summary
}

# Parse arguments
case "${1:-all}" in
    all) run_all ;;
    quick) run_quick ;;
    rq1) run_rq1_halflife ;;
    rq2) run_rq2_sync_sweep ;;
    rq3) run_rq3_comm_reduction ;;
    rq5) run_rq5_nesterov ;;
    rq6) run_rq6_muon ;;
    figures) generate_figures ;;
    summary) print_summary ;;
    *)
        echo "Usage: $0 {all|quick|rq1|rq2|rq3|rq5|rq6|figures|summary}"
        exit 1 ;;
esac
