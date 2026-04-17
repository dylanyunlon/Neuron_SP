#!/bin/bash
#===============================================================================
# M039: Master Batch Runner Script
#===============================================================================
# NeurIPS 2026 Submission - DES-LOC Benchmark Framework
#
# This script orchestrates the execution of 100+ experiments across multiple
# GPUs, manages checkpoints, and aggregates results.
#
# Author: Claude (M026-M050)
# Date: April 2026
# License: MIT
#
# Usage:
#   ./run_all_experiments.sh [--config CONFIG_FILE] [--gpus GPU_IDS]
#
# Hardware Requirements:
#   - 2x NVIDIA RTX A6000 (48GB)
#   - 1x NVIDIA H100 NVL (96GB)
#===============================================================================

set -e  # Exit on error
set -o pipefail

#===============================================================================
# CONFIGURATION
#===============================================================================

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
SRC_DIR="${PROJECT_ROOT}/src"
OUTPUT_DIR="${PROJECT_ROOT}/outputs"
LOG_DIR="${PROJECT_ROOT}/logs"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"
CONFIG_DIR="${PROJECT_ROOT}/configs"

# Default settings
DEFAULT_GPUS="0,1,2"  # A6000, A6000, H100
CONDA_ENV="nips2026"
PYTHON_EXEC="python"
MAX_PARALLEL_JOBS=3
RETRY_LIMIT=3
TIMEOUT_HOURS=24

# Experiment settings
TOTAL_EXPERIMENTS=100
EXPERIMENTS_PER_GPU=4

# Timestamps
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="run_${TIMESTAMP}"

#===============================================================================
# HELPER FUNCTIONS
#===============================================================================

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1"
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $1" >&2
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2
}

create_directories() {
    log_info "Creating directories..."
    mkdir -p "${OUTPUT_DIR}/${RUN_ID}"
    mkdir -p "${LOG_DIR}/${RUN_ID}"
    mkdir -p "${CHECKPOINT_DIR}/${RUN_ID}"
    mkdir -p "${CONFIG_DIR}"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Python
    if ! command -v ${PYTHON_EXEC} &> /dev/null; then
        log_error "Python not found: ${PYTHON_EXEC}"
        exit 1
    fi
    
    # Check CUDA
    if ! command -v nvidia-smi &> /dev/null; then
        log_warn "nvidia-smi not found, running in CPU mode"
        CUDA_AVAILABLE=false
    else
        CUDA_AVAILABLE=true
        log_info "CUDA available: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    fi
    
    # Check PyTorch
    ${PYTHON_EXEC} -c "import torch; print(f'PyTorch {torch.__version__}')" || {
        log_error "PyTorch not installed"
        exit 1
    }
    
    # Check available GPUs
    if [ "${CUDA_AVAILABLE}" = true ]; then
        NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        log_info "Available GPUs: ${NUM_GPUS}"
    fi
}

setup_environment() {
    log_info "Setting up environment..."
    
    # Activate conda environment if available
    if command -v conda &> /dev/null; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        if conda env list | grep -q "${CONDA_ENV}"; then
            conda activate "${CONDA_ENV}"
            log_info "Activated conda environment: ${CONDA_ENV}"
        fi
    fi
    
    # Set environment variables
    export PYTHONPATH="${SRC_DIR}:${PYTHONPATH}"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export NCCL_DEBUG=WARN
    export NCCL_IB_DISABLE=0
    export OMP_NUM_THREADS=8
    
    # For H100
    export CUDA_VISIBLE_DEVICES="${DEFAULT_GPUS}"
}

get_gpu_for_experiment() {
    local exp_id=$1
    local model_size=$2
    
    # H100 for large models, A6000 for small
    case "${model_size}" in
        "1.7B"|"7B")
            echo "2"  # H100
            ;;
        "360M")
            echo "$(( exp_id % 2 ))"  # Rotate A6000s
            ;;
        *)
            echo "$(( exp_id % 2 ))"  # 125M on A6000s
            ;;
    esac
}

#===============================================================================
# EXPERIMENT DEFINITIONS
#===============================================================================

declare -a EXPERIMENTS

# Figure 1: Rosenbrock
EXPERIMENTS+=("rosenbrock:125M:32:64:128")

# Figure 2: Momentum Ablation (β₁ sweep)
for beta1 in 0.8 0.9 0.95 0.99; do
    EXPERIMENTS+=("momentum_ablation:125M:${beta1}")
done

# Figure 3a: Kx Ablation
for kx in 16 32 64 128; do
    EXPERIMENTS+=("sync_ablation:kx:125M:${kx}:64:128")
done

# Figure 3b: Ku Ablation
for ku in 32 64 128 256; do
    EXPERIMENTS+=("sync_ablation:ku:125M:32:${ku}:128")
done

# Figure 3c: Kv Ablation
for kv in 64 128 256 512; do
    EXPERIMENTS+=("sync_ablation:kv:125M:32:64:${kv}")
done

# Figure 4: Communication Reduction
for model_size in 125M 360M; do
    for method in ddp desloc_k32 desloc_k64 desloc_k128; do
        EXPERIMENTS+=("comm_reduction:${model_size}:${method}")
    done
done

# Figure 5: Billion Scale
for model_size in 125M 360M; do
    for method in ddp desloc; do
        EXPERIMENTS+=("billion_scale:${model_size}:${method}")
    done
done

# Figure 6: Outer Optimizer
for outer in simple_avg weighted_avg nesterov heavy_ball polyak; do
    EXPERIMENTS+=("outer_optimizer:125M:${outer}")
done

# Figure 7: Muon Integration
for opt in adam muon desloc_adam desloc_muon; do
    EXPERIMENTS+=("muon:125M:${opt}")
done

# Table 2: Wall-Clock Time
for model_size in 125M 360M; do
    for method in ddp desloc_k32 desloc_k64 desloc_k128; do
        EXPERIMENTS+=("wallclock:${model_size}:${method}")
    done
done

# Additional ablations for robustness
for seed in 42 123 456 789; do
    EXPERIMENTS+=("rosenbrock:125M:32:64:128:seed${seed}")
done

#===============================================================================
# EXPERIMENT RUNNER
#===============================================================================

run_experiment() {
    local exp_spec=$1
    local exp_id=$2
    local retry_count=${3:-0}
    
    # Parse experiment specification
    IFS=':' read -ra SPEC <<< "${exp_spec}"
    local exp_type="${SPEC[0]}"
    
    log_info "Running experiment ${exp_id}: ${exp_spec}"
    
    # Determine GPU and model size
    local model_size="${SPEC[1]}"
    local gpu_id=$(get_gpu_for_experiment ${exp_id} ${model_size})
    
    # Log file
    local log_file="${LOG_DIR}/${RUN_ID}/exp_${exp_id}_${exp_type}.log"
    
    # Build command based on experiment type
    local cmd=""
    local timeout_secs=$((TIMEOUT_HOURS * 3600))
    
    case "${exp_type}" in
        "rosenbrock")
            local kx="${SPEC[2]}"
            local ku="${SPEC[3]}"
            local kv="${SPEC[4]}"
            local seed="${SPEC[5]:-42}"
            cmd="${PYTHON_EXEC} ${SRC_DIR}/experiments/run_rosenbrock.py \
                --output-dir ${OUTPUT_DIR}/${RUN_ID} \
                --num-workers 256 \
                --total-steps 1000"
            ;;
        
        "momentum_ablation")
            local beta1="${SPEC[2]}"
            cmd="${PYTHON_EXEC} ${SRC_DIR}/experiments/run_momentum_ablation.py \
                --output-dir ${OUTPUT_DIR}/${RUN_ID} \
                --beta1-values ${beta1} \
                --total-steps 2000"
            ;;
        
        "sync_ablation")
            local ablation_type="${SPEC[1]}"
            model_size="${SPEC[2]}"
            cmd="${PYTHON_EXEC} ${SRC_DIR}/experiments/run_sync_ablation.py \
                --output-dir ${OUTPUT_DIR}/${RUN_ID} \
                --ablation-type ${ablation_type} \
                --total-steps 2000"
            ;;
        
        "comm_reduction")
            local method="${SPEC[2]}"
            cmd="${PYTHON_EXEC} ${SRC_DIR}/experiments/run_comm_reduction.py \
                --output-dir ${OUTPUT_DIR}/${RUN_ID} \
                --model-sizes ${model_size} \
                --total-steps 5000"
            ;;
        
        "billion_scale")
            local method="${SPEC[2]}"
            cmd="${PYTHON_EXEC} ${SRC_DIR}/experiments/run_billion_scale.py \
                --output-dir ${OUTPUT_DIR}/${RUN_ID} \
                --scale ${model_size}"
            ;;
        
        "outer_optimizer")
            local outer_type="${SPEC[2]}"
            cmd="${PYTHON_EXEC} ${SRC_DIR}/experiments/run_outer_optimizer.py \
                --output-dir ${OUTPUT_DIR}/${RUN_ID} \
                --outer-type ${outer_type} \
                --total-steps 2000"
            ;;
        
        "muon")
            local optimizer="${SPEC[2]}"
            cmd="${PYTHON_EXEC} ${SRC_DIR}/experiments/run_muon.py \
                --output-dir ${OUTPUT_DIR}/${RUN_ID} \
                --optimizer ${optimizer} \
                --total-steps 2000"
            ;;
        
        "wallclock")
            local method="${SPEC[2]}"
            cmd="${PYTHON_EXEC} ${SRC_DIR}/experiments/run_wallclock.py \
                --output-dir ${OUTPUT_DIR}/${RUN_ID} \
                --model-sizes ${model_size} \
                --total-steps 300"
            ;;
        
        *)
            log_error "Unknown experiment type: ${exp_type}"
            return 1
            ;;
    esac
    
    # Run with GPU assignment and timeout
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    
    log_info "GPU ${gpu_id}: ${cmd}"
    
    if timeout ${timeout_secs} bash -c "${cmd}" > "${log_file}" 2>&1; then
        log_info "Experiment ${exp_id} completed successfully"
        echo "SUCCESS:${exp_spec}" >> "${OUTPUT_DIR}/${RUN_ID}/status.log"
        return 0
    else
        local exit_code=$?
        log_warn "Experiment ${exp_id} failed with code ${exit_code}"
        
        if [ ${retry_count} -lt ${RETRY_LIMIT} ]; then
            log_info "Retrying experiment ${exp_id} (attempt $((retry_count + 1))/${RETRY_LIMIT})"
            sleep 10
            run_experiment "${exp_spec}" ${exp_id} $((retry_count + 1))
            return $?
        else
            log_error "Experiment ${exp_id} failed after ${RETRY_LIMIT} retries"
            echo "FAILED:${exp_spec}" >> "${OUTPUT_DIR}/${RUN_ID}/status.log"
            return 1
        fi
    fi
}

run_experiments_parallel() {
    local num_jobs=${1:-${MAX_PARALLEL_JOBS}}
    local total=${#EXPERIMENTS[@]}
    
    log_info "Running ${total} experiments with ${num_jobs} parallel jobs"
    
    local pids=()
    local exp_id=0
    local running=0
    
    for exp_spec in "${EXPERIMENTS[@]}"; do
        # Wait if max parallel jobs reached
        while [ ${running} -ge ${num_jobs} ]; do
            for i in "${!pids[@]}"; do
                if ! kill -0 ${pids[$i]} 2>/dev/null; then
                    wait ${pids[$i]} || true
                    unset pids[$i]
                    ((running--))
                fi
            done
            sleep 1
        done
        
        # Start new job
        run_experiment "${exp_spec}" ${exp_id} &
        pids+=($!)
        ((running++))
        ((exp_id++))
        
        log_info "Progress: ${exp_id}/${total} experiments started"
    done
    
    # Wait for remaining jobs
    log_info "Waiting for remaining jobs..."
    for pid in "${pids[@]}"; do
        wait ${pid} || true
    done
    
    log_info "All experiments completed"
}

run_experiments_sequential() {
    local total=${#EXPERIMENTS[@]}
    local exp_id=0
    local success=0
    local failed=0
    
    log_info "Running ${total} experiments sequentially"
    
    for exp_spec in "${EXPERIMENTS[@]}"; do
        if run_experiment "${exp_spec}" ${exp_id}; then
            ((success++))
        else
            ((failed++))
        fi
        ((exp_id++))
        
        log_info "Progress: ${exp_id}/${total} (${success} success, ${failed} failed)"
    done
    
    log_info "Completed: ${success} success, ${failed} failed"
}

#===============================================================================
# RESULTS AGGREGATION
#===============================================================================

aggregate_results() {
    log_info "Aggregating results..."
    
    local results_file="${OUTPUT_DIR}/${RUN_ID}/aggregated_results.json"
    
    ${PYTHON_EXEC} << EOF
import json
import os
from pathlib import Path
from datetime import datetime

output_dir = Path("${OUTPUT_DIR}/${RUN_ID}")
results = {
    'run_id': '${RUN_ID}',
    'timestamp': datetime.now().isoformat(),
    'experiments': []
}

# Collect all JSON results
for json_file in output_dir.glob("*_results.json"):
    try:
        with open(json_file) as f:
            data = json.load(f)
            results['experiments'].append({
                'file': json_file.name,
                'data': data
            })
    except Exception as e:
        print(f"Error reading {json_file}: {e}")

# Save aggregated results
with open("${results_file}", 'w') as f:
    json.dump(results, f, indent=2)

print(f"Aggregated {len(results['experiments'])} experiment results")
EOF
    
    log_info "Results saved to ${results_file}"
}

generate_summary() {
    log_info "Generating summary..."
    
    local summary_file="${OUTPUT_DIR}/${RUN_ID}/summary.txt"
    
    {
        echo "==============================================================================="
        echo "NeurIPS 2026 - DES-LOC Benchmark Experiment Summary"
        echo "Run ID: ${RUN_ID}"
        echo "Timestamp: $(date)"
        echo "==============================================================================="
        echo ""
        
        # Count status
        if [ -f "${OUTPUT_DIR}/${RUN_ID}/status.log" ]; then
            echo "Experiment Status:"
            echo "  Success: $(grep -c "SUCCESS" "${OUTPUT_DIR}/${RUN_ID}/status.log" || echo 0)"
            echo "  Failed: $(grep -c "FAILED" "${OUTPUT_DIR}/${RUN_ID}/status.log" || echo 0)"
            echo ""
        fi
        
        echo "Output files:"
        ls -la "${OUTPUT_DIR}/${RUN_ID}/"*.json 2>/dev/null || echo "  (none)"
        echo ""
        
        echo "Log files:"
        ls -la "${LOG_DIR}/${RUN_ID}/"*.log 2>/dev/null | wc -l
        echo ""
        
    } > "${summary_file}"
    
    cat "${summary_file}"
}

#===============================================================================
# MAIN
#===============================================================================

main() {
    local mode="sequential"
    local config_file=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --parallel)
                mode="parallel"
                shift
                ;;
            --config)
                config_file="$2"
                shift 2
                ;;
            --gpus)
                DEFAULT_GPUS="$2"
                shift 2
                ;;
            --jobs)
                MAX_PARALLEL_JOBS="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [--parallel] [--config FILE] [--gpus IDS] [--jobs N]"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Load config if provided
    if [ -n "${config_file}" ] && [ -f "${config_file}" ]; then
        source "${config_file}"
    fi
    
    log_info "Starting DES-LOC Benchmark Suite"
    log_info "Run ID: ${RUN_ID}"
    log_info "Mode: ${mode}"
    
    create_directories
    check_prerequisites
    setup_environment
    
    # Run experiments
    if [ "${mode}" = "parallel" ]; then
        run_experiments_parallel ${MAX_PARALLEL_JOBS}
    else
        run_experiments_sequential
    fi
    
    # Post-processing
    aggregate_results
    generate_summary
    
    log_info "Benchmark suite completed"
    log_info "Results: ${OUTPUT_DIR}/${RUN_ID}"
}

# Run main
main "$@"
