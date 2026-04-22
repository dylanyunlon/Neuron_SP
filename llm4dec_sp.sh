#!/bin/bash
# ===========================================
# LLM4DEC-SP v3.0 — DES-LOC + AutoSP 实验流水线
# 复刻 llm4ccpo.sh v7.0 架构, 零内嵌Python
# ===========================================
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/desloc_results}"
FIGURES_DIR="${FIGURES_DIR:-$PROJECT_DIR/desloc_figures}"
HF_CACHE_DIR="${HF_HOME:-/data/jiacheng/system/cache/temp/huggingface}"
export HF_HOME="$HF_CACHE_DIR"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-base}"
CONDA_BASE=$(conda info --base 2>/dev/null || echo "/usr/local/lib/miniconda3")

# === 模型配置 ===
declare -A MODEL_PARAMS=(["125M"]="125000000" ["350M"]="350000000" ["700M"]="700000000" ["1.3B"]="1300000000" ["7B"]="7000000000")
declare -A MODEL_LAYERS=(["125M"]="12" ["350M"]="24" ["700M"]="36" ["1.3B"]="24" ["7B"]="32")
declare -A MODEL_HIDDEN=(["125M"]="768" ["350M"]="1024" ["700M"]="1280" ["1.3B"]="2048" ["7B"]="4096")
declare -A MODEL_HEADS=(["125M"]="12" ["350M"]="16" ["700M"]="20" ["1.3B"]="16" ["7B"]="32")
ONDEVICE_MODELS=("125M" "350M" "700M" "1.3B")
MODEL_KEY="${MODEL_KEY:-125M}"

# === 训练参数 ===
MAX_STEPS="${MAX_STEPS:-500}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DEFAULT_KX="${DEFAULT_KX:-32}"
DEFAULT_KU="${DEFAULT_KU:-96}"
DEFAULT_KV="${DEFAULT_KV:-192}"
SP_SIZE="${SP_SIZE:-2}"
SEEDS="${SEEDS:-42 137 2024}"

# === CSV格式（与已有数据兼容） ===
# run_id,tag,model,Kx,Ku,Kv,seed,method,exit_code,elapsed_s
# $1=run_id $2=tag $3=model $4=Kx $5=Ku $6=Kv $7=seed $8=method $9=exit_code $10=elapsed_s
EXPERIMENT_LOG="$OUTPUT_DIR/experiment_log.csv"
CSV_HEADER="run_id,tag,model,Kx,Ku,Kv,seed,method,exit_code,elapsed_s"

# === 工具函数 ===
print_header() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  LLM4DEC-SP v3.0 — DES-LOC+AutoSP 分布式训练实验流水线     ║"
    echo "║  Desynced Low-Communication + Automatic Sequence Parallel    ║"
    echo "║  支持 125M-1.3B 直接训练 + 7B SP分片验证                    ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
}
print_step() {
    echo ""
    echo "┌──────────────────────────────────────────────────────────────┐"
    echo "│  $1"
    echo "└──────────────────────────────────────────────────────────────┘"
    echo ""
}
check_dir() { mkdir -p "$1"; }
log_info() { echo "[INFO] $(date '+%H:%M:%S') $1"; }
log_error() { echo "[ERROR] $(date '+%H:%M:%S') $1" >&2; }
log_warn() { echo "[WARN] $(date '+%H:%M:%S') $1" >&2; }
log_success() { echo "[✓] $(date '+%H:%M:%S') $1"; }

init_conda() {
    local setup
    setup="$("$CONDA_BASE/bin/conda" 'shell.bash' 'hook' 2>/dev/null)" && eval "$setup" \
        || { [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] && . "$CONDA_BASE/etc/profile.d/conda.sh"; } \
        || export PATH="$CONDA_BASE/bin:$PATH"
}
activate_env() {
    init_conda
    conda activate "${1:-$CONDA_ENV_NAME}" 2>/dev/null || log_warn "Using current env"
}

# === 模型切换 ===
switch_model() {
    local k="$1"
    [[ -z "${MODEL_PARAMS[$k]+x}" ]] && { log_error "Unknown model: $k (available: ${!MODEL_PARAMS[*]})"; return 1; }
    MODEL_KEY="$k"
    log_info "Switched to model: $MODEL_KEY (${MODEL_PARAMS[$k]} params, ${MODEL_LAYERS[$k]}L, ${MODEL_HIDDEN[$k]}H)"
}

# === 环境配置 ===
setup_environment() {
    print_step "Setting up Environment"
    command -v conda &>/dev/null || { log_error "Conda not found."; exit 1; }
    activate_env
    log_info "Installing DES-LOC packages..."
    pip install -q torch matplotlib seaborn pandas scipy pyyaml tqdm 2>/dev/null || true
    [ -f "$PROJECT_DIR/requirements.txt" ] && pip install -q -r "$PROJECT_DIR/requirements.txt" 2>/dev/null || true
    log_success "Environment ready"
}

prepare_dirs() {
    print_step "Preparing Directories"
    check_dir "$OUTPUT_DIR/logs"
    check_dir "$FIGURES_DIR"
    log_info "Project: $PROJECT_DIR"
    log_info "Output:  $OUTPUT_DIR"
    log_info "Model:   $MODEL_KEY (${MODEL_PARAMS[$MODEL_KEY]} params)"
    log_info "DES-LOC: Kx=$DEFAULT_KX Ku=$DEFAULT_KU Kv=$DEFAULT_KV"
    log_success "Directories prepared"
}

# === Step 1: 数据（合成，运行时生成）===
generate_training_data() {
    print_step "Step 1: Data (synthetic, generated on-the-fly by REAL_GPU_BENCHMARK.py)"
    log_success "No pre-generation needed — SyntheticDataset uses torch.manual_seed"
}

# === Step 2: 单次实验运行 ===
RUN_ID=0; FAIL=0; TOTAL_RUNS=0

_run_single_experiment() {
    local TAG=$1 MODEL=$2 KX=$3 KU=$4 KV=$5 SEED=$6 METHOD=$7 GPUS=$8 NGPU=$9

    RUN_ID=$((RUN_ID + 1)); TOTAL_RUNS=$((TOTAL_RUNS + 1))
    local LOGFILE="$OUTPUT_DIR/logs/${TAG}_${MODEL}_Kx${KX}_${METHOD}_s${SEED}.log"
    local T0=$SECONDS

    printf "[%3d] %-14s %-5s Kx=%-3d Ku=%-3d Kv=%-3d %-10s seed=%-4d gpu=%-5s ... " \
        "$RUN_ID" "$TAG" "$MODEL" "$KX" "$KU" "$KV" "$METHOD" "$SEED" "$GPUS"

    export PYTHONHASHSEED=$SEED CUDA_VISIBLE_DEVICES=$GPUS

    local CMD="python3"
    [ "$NGPU" -gt 1 ] && CMD="torchrun --nproc_per_node=$NGPU --master_port=$((29500 + RUN_ID % 200))"

    $CMD REAL_GPU_BENCHMARK.py \
        --model_size "$MODEL" \
        --batch_size $BATCH_SIZE \
        --grad_accum $GRAD_ACCUM \
        --max_steps $MAX_STEPS \
        --Kx "$KX" --Ku "$KU" --Kv "$KV" \
        --methods "$METHOD" \
        --output "$OUTPUT_DIR" \
        > "$LOGFILE" 2>&1

    local RC=$?; local DT=$((SECONDS - T0))
    [ $RC -eq 0 ] && echo "OK (${DT}s)" || { echo "FAIL:${RC} (${DT}s)"; FAIL=$((FAIL + 1)); }

    # 写CSV — 与已有数据格式完全一致
    echo "${RUN_ID},${TAG},${MODEL},${KX},${KU},${KV},${SEED},${METHOD},${RC},${DT}" >> "$EXPERIMENT_LOG"
}

# === Step 2: 训练 ===
run_training() {
    print_step "Step 2: Training (DDP + DES-LOC + LocalSGD)"
    activate_env; cd "$PROJECT_DIR"
    local model="${1:-$MODEL_KEY}"
    check_dir "$OUTPUT_DIR/logs"

    # 确保CSV有header（不覆盖已有数据）
    [ ! -f "$EXPERIMENT_LOG" ] && echo "$CSV_HEADER" > "$EXPERIMENT_LOG"

    log_info "Model: $model, Steps: $MAX_STEPS, DES-LOC: Kx=$DEFAULT_KX"

    for SEED in $SEEDS; do
        _run_single_experiment "baseline" "$model" 1 1 1 $SEED "DDP" "$CUDA_DEVICE" 1
        _run_single_experiment "desloc"   "$model" $DEFAULT_KX $DEFAULT_KU $DEFAULT_KV $SEED "DESLOC" "$CUDA_DEVICE" 1
        _run_single_experiment "local"    "$model" $DEFAULT_KX $DEFAULT_KX $DEFAULT_KX $SEED "LocalAdam" "$CUDA_DEVICE" 1
    done

    log_success "Training complete: $TOTAL_RUNS runs"
    cd "$SCRIPT_DIR"
}

# === Step 3: Kx消融 ===
run_kx_sweep() {
    print_step "Step 3: Kx Sweep (8 values × 3 seeds = 24 runs)"
    activate_env; cd "$PROJECT_DIR"
    [ ! -f "$EXPERIMENT_LOG" ] && echo "$CSV_HEADER" > "$EXPERIMENT_LOG"

    for KX in 1 2 4 8 16 32 64 128; do
        [ $KX -eq 1 ] && { KU=1; KV=1; } || { KU=$((KX*3)); KV=$((KX*6)); }
        for SEED in $SEEDS; do
            _run_single_experiment "rq1_kx" "${1:-$MODEL_KEY}" $KX $KU $KV $SEED "DESLOC" "$CUDA_DEVICE" 1
        done
    done
    log_success "Kx sweep complete"; cd "$SCRIPT_DIR"
}

# === Step 4: Ku/Kv比例消融 ===
run_ratio_ablation() {
    print_step "Step 4: Ku/Kv Ratio Ablation (9 combos × 3 seeds = 27 runs)"
    activate_env; cd "$PROJECT_DIR"
    [ ! -f "$EXPERIMENT_LOG" ] && echo "$CSV_HEADER" > "$EXPERIMENT_LOG"

    for KU_R in 1 3 6; do for KV_R in 1 6 12; do
        for SEED in $SEEDS; do
            _run_single_experiment "rq2_ratio" "${1:-$MODEL_KEY}" 32 $((32*KU_R)) $((32*KV_R)) $SEED "DESLOC" "$CUDA_DEVICE" 1
        done
    done; done
    log_success "Ratio ablation complete"; cd "$SCRIPT_DIR"
}

# === Step 5: 评测 ===
# CSV列号: $3=model $4=Kx $8=method $9=exit_code
run_eval() {
    print_step "Step 5: Evaluation — Collecting Metrics from Logs"
    log_info "数据源: $EXPERIMENT_LOG"

    if [ ! -f "$EXPERIMENT_LOG" ]; then
        log_warn "No experiment log found at $EXPERIMENT_LOG"
        log_warn "Run './llm4dec_sp.sh train' first"
        _show_eval_summary
        return 0
    fi

    local total ok fail
    total=$(tail -n +2 "$EXPERIMENT_LOG" | wc -l | tr -d ' ')
    total=${total:-0}
    # $9=exit_code in our CSV format
    ok=$(tail -n +2 "$EXPERIMENT_LOG" | awk -F, '$9 == 0 {n++} END {print n+0}')
    ok=${ok:-0}
    fail=$((total - ok))

    echo ""
    echo "  ═══════════════════════════════════════════════"
    echo "  Experiment Summary"
    echo "  ═══════════════════════════════════════════════"
    echo "  Total runs:  $total"
    echo "  Success:     $ok"
    echo "  Failed:      $fail"
    echo ""

    # 按model+method汇总 — $3=model $8=method $4=Kx $9=exit_code
    echo "  Per-Model-Method Breakdown:"
    echo "  ─────────────────────────────────────────────"
    printf "  %-6s %-10s %-4s %-6s %-6s\n" "Model" "Method" "Kx" "Runs" "OK"
    echo "  ─────────────────────────────────────────────"
    tail -n +2 "$EXPERIMENT_LOG" | awk -F, '{
        key = $3 "|" $8 "|" $4
        total[key]++
        if ($9 == 0) ok[key]++
    } END {
        for (k in total) {
            split(k, a, "|")
            printf "  %-6s %-10s %-4s %-6d %-6d\n", a[1], a[2], a[3], total[k], ok[k]+0
        }
    }' | sort
    echo ""

    _show_eval_summary
    log_success "Evaluation complete"
}

_show_eval_summary() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  DES-LOC 论文参考结果 (Section 5)                             ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║  Method     │ CommReduction │ Loss Delta │ Notes              ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║  DDP        │ 1×            │ baseline   │ AllReduce/step    ║"
    echo "║  LocalSGD   │ Kx×           │ ~+0.02     │ sync every Kx    ║"
    echo "║  DES-LOC    │ 32×-170×      │ ~+0.01     │ 3-tier Kx/Ku/Kv  ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║  AutoSP Ulysses vs Ring-FA: 1.44-1.62× speedup              ║"
    echo "║  AutoSP+AC: 1.6× longer context, only 9% latency overhead   ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
}

# === 图表（调用仓库代码）===
generate_figures() {
    print_step "Generating NKI-FA Grade Figures"
    activate_env; cd "$PROJECT_DIR"; check_dir "$FIGURES_DIR"
    # 直接调用REAL_GPU_BENCHMARK.py的CLI
    PYTHONPATH="$PROJECT_DIR" python3 REAL_GPU_BENCHMARK.py \
        --output "$OUTPUT_DIR" --methods PLOT_ONLY 2>/dev/null \
        || log_warn "Plot mode not available — run experiments first"
    log_success "Figures done"
    cd "$SCRIPT_DIR"
}

# === 批量训练所有模型 ===
train_all_models() {
    print_step "Training All Models with DES-LOC"
    local t0=$SECONDS success=() failed=()

    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  DES-LOC 批量训练: ${ONDEVICE_MODELS[*]}"
    echo "║  每模型: 3 methods × 3 seeds = 9 runs"
    echo "╚═══════════════════════════════════════════════════════════════╝"

    for mk in "${ONDEVICE_MODELS[@]}"; do
        echo ""; echo "═══ Training: $mk (${MODEL_PARAMS[$mk]} params) ═══"
        switch_model "$mk"
        if run_full_train_single "$mk"; then success+=("$mk"); else failed+=("$mk"); fi
    done

    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  批量训练完成 — $((SECONDS-t0))s"
    echo "║  成功: ${#success[@]} (${success[*]})  失败: ${#failed[@]} (${failed[*]:-无})"
    echo "╚═══════════════════════════════════════════════════════════════╝"
}

run_full_train_single() {
    prepare_dirs || return 1
    run_training "${1:-$MODEL_KEY}" || return 1
}

# === 批量评测 ===
eval_all_models() {
    print_step "Evaluating All DES-LOC Trained Models"
    local t0=$SECONDS success=() failed=()

    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  评测: ${ONDEVICE_MODELS[*]}"
    echo "╚═══════════════════════════════════════════════════════════════╝"

    for mk in "${ONDEVICE_MODELS[@]}"; do
        switch_model "$mk"
        echo ""; echo "  Evaluating: $mk"
        if run_eval; then success+=("$mk"); else failed+=("$mk"); fi
    done

    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  批量评测完成 — $((SECONDS-t0))s"
    echo "║  成功: ${#success[@]} (${success[*]})  失败: ${#failed[@]} (${failed[*]:-无})"
    echo "╚═══════════════════════════════════════════════════════════════╝"

    compare_results
}

# === 对比报告（纯bash，零python）===
compare_results() {
    print_step "Generating Comparison Report"
    local report="$OUTPUT_DIR/desloc_comparison_report.md"

    {
        echo "# DES-LOC + AutoSP 实验对比报告"
        echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        echo "## 配置"
        echo "- 硬件: 2× RTX A6000 (48GB) + 1× H100 NVL (94GB)"
        echo "- DES-LOC: Kx=$DEFAULT_KX, Ku=$DEFAULT_KU, Kv=$DEFAULT_KV"
        echo "- Seeds: $SEEDS"
        echo ""
        echo "## 实验矩阵"
        echo "| Model | Method | Kx | Runs | OK |"
        echo "|-------|--------|----|------|----|"

        if [ -f "$EXPERIMENT_LOG" ]; then
            tail -n +2 "$EXPERIMENT_LOG" | awk -F, '{
                key=$3"|"$8"|"$4; total[key]++; if($9==0) ok[key]++
            } END {
                for(k in total) { split(k,a,"|"); printf "| %s | %s | %s | %d | %d |\n",a[1],a[2],a[3],total[k],ok[k]+0 }
            }' | sort
        fi

        echo ""
        echo "## AutoSP vs Ring-FA"
        echo "- Ulysses A2A: 1.44-1.62× faster (8×A100, Llama3.1-8B, seq=15k)"
        echo "- AutoSP+AC: 1.6× longer context, 9% latency overhead"
    } > "$report"

    cat "$report"
    log_success "Report: $report"
}

# === 完整流水线 ===
run_all_models_pipeline() {
    print_step "Complete Pipeline: All Models"
    local t0=$SECONDS
    train_all_models; eval_all_models; generate_figures
    echo "Done: $TOTAL_RUNS runs in $((SECONDS-t0))s"
}

run_full_train() {
    print_step "Full Training Pipeline ($MODEL_KEY)"
    prepare_dirs; run_training; run_kx_sweep; run_ratio_ablation
}
run_full_eval() {
    print_step "Full Evaluation"
    run_eval; generate_figures
}
run_full_pipeline() {
    print_step "Complete Pipeline ($MODEL_KEY)"
    local t0=$SECONDS; run_full_train; run_full_eval
    echo "Done: $TOTAL_RUNS runs in $((SECONDS-t0))s"
}
train_only() { run_training "$@"; }
eval_only() { run_eval "$@"; }

# === 状态 ===
show_data_status() {
    print_step "Data Status"
    echo "Experiment Log: $EXPERIMENT_LOG"
    if [ -f "$EXPERIMENT_LOG" ]; then
        local runs ok
        runs=$(tail -n +2 "$EXPERIMENT_LOG" | wc -l | tr -d ' '); runs=${runs:-0}
        ok=$(tail -n +2 "$EXPERIMENT_LOG" | awk -F, '$9 == 0 {n++} END {print n+0}'); ok=${ok:-0}
        echo "  ✓ $runs runs (OK: $ok, FAIL: $((runs-ok)))"
    else
        echo "  ✗ no experiments"
    fi
    echo ""; echo "Figures: $FIGURES_DIR"
    [ -d "$FIGURES_DIR" ] && echo "  $(ls -1 "$FIGURES_DIR"/*.png 2>/dev/null | wc -l) figures" || echo "  ✗ none"
}

show_all_models_status() {
    print_step "All Models Status"
    printf "%-6s %-12s %-8s\n" "Model" "Params" "InLog"
    echo "──────────────────────────────"
    for mk in "${ONDEVICE_MODELS[@]}"; do
        local inlog="✗"
        [ -f "$EXPERIMENT_LOG" ] && grep -q ",$mk," "$EXPERIMENT_LOG" 2>/dev/null && inlog="✓"
        printf "%-6s %-12s %-8s\n" "$mk" "${MODEL_PARAMS[$mk]}" "$inlog"
    done
}

show_config() {
    print_step "Configuration"
    echo "  MODEL_KEY:   $MODEL_KEY (${MODEL_PARAMS[$MODEL_KEY]} params)"
    echo "  DES-LOC:     Kx=$DEFAULT_KX Ku=$DEFAULT_KU Kv=$DEFAULT_KV"
    echo "  MAX_STEPS:   $MAX_STEPS"
    echo "  BATCH:       $BATCH_SIZE × $GRAD_ACCUM"
    echo "  SEEDS:       $SEEDS"
    echo "  CUDA:        $CUDA_DEVICE"
    echo "  OUTPUT:      $OUTPUT_DIR"
}

quick_fix() { activate_env; setup_environment; prepare_dirs; }
clean_data() {
    echo "Delete $OUTPUT_DIR and $FIGURES_DIR?"
    read -p "(y/N): " r; [[ "$r" =~ ^[Yy]$ ]] && rm -rf "$OUTPUT_DIR" "$FIGURES_DIR" && log_success "Cleaned"
}

show_help() {
    cat << 'EOF'
LLM4DEC-SP v3.0 — DES-LOC+AutoSP 实验流水线

Usage: ./llm4dec_sp.sh <command>

BATCH:        all_models | eval_all_models | compare_results | all_status
SINGLE:       full | full_train | full_eval
SETUP:        setup | prepare | config | fix | clean | status
EXPERIMENTS:  train | kx_sweep | ratio_abl | generate
EVAL:         eval | figures

ENV: MAX_STEPS=500 MODEL_KEY=125M DEFAULT_KX=32 SEEDS="42 137 2024"

Examples:
  MAX_STEPS=100 ./llm4dec_sp.sh full
  ./llm4dec_sp.sh all_models
  ./llm4dec_sp.sh eval_all_models
  MODEL_KEY=1.3B ./llm4dec_sp.sh train
EOF
}

# === 主入口 ===
main() {
    print_header
    local cmd="${1:-help}"; shift 2>/dev/null || true
    case $cmd in
        setup)          setup_environment ;; prepare) prepare_dirs ;; config) show_config ;;
        fix)            quick_fix ;; clean) clean_data ;; status) show_data_status ;;
        all_status)     show_all_models_status ;; switch) switch_model "$@" ;;
        generate|gen)   generate_training_data ;; train) run_training "$@" ;;
        train_only)     train_only "$@" ;; kx_sweep) run_kx_sweep "$@" ;;
        ratio_abl)      run_ratio_ablation "$@" ;;
        eval)           run_eval "$@" ;; eval_only) eval_only "$@" ;; figures) generate_figures ;;
        full_train)     run_full_train ;; full_eval) run_full_eval ;; full|all) run_full_pipeline ;;
        all_models)     run_all_models_pipeline ;; train_all) train_all_models ;;
        eval_all_models) eval_all_models ;; compare|compare_results) compare_results ;;
        help|--help|-h) show_help ;;
        *) log_error "Unknown: $cmd"; show_help; exit 1 ;;
    esac
}
main "$@"
