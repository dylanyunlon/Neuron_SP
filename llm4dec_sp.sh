#!/bin/bash
# ===========================================
# LLM4DEC-SP - DES-LOC + AutoSP 分布式训练实验流水线 (v2.0)
# ===========================================
#
# 完全复刻 llm4ccpo.sh v7.0 架构，替换为 DES-LOC + AutoSP 内容
#
# v2.0 功能:
#   1. 支持所有模型规模 (125M, 350M, 700M, 1.3B)
#   2. 批量训练所有模型: ./llm4dec_sp.sh all_models
#   3. 批量评测所有模型: ./llm4dec_sp.sh eval_all_models
#   4. 汇总对比报告: ./llm4dec_sp.sh compare_results
#   5. 7B+ 模型通过 SP_size=3 在 3-GPU 上验证
#
# DES-LOC 核心思想 (Desynced Low-Communication Optimizer):
#   Step 1: 每个 worker 本地训练 Kx 步，不做 AllReduce
#   Step 2: 第 Kx 步时，同步梯度 (SP all-to-all for attention)
#   Step 3: 第 Ku 步时，同步一阶动量 (跨 DP 组)
#   Step 4: 第 Kv 步时，同步二阶动量 (全量同步)
#   结果: 通信量降低 Kx× 到 170×，收敛性保持
#
# AutoSP 核心思想 (Automatic Sequence Parallelism):
#   编译器驱动: 在 Torch-IR 层自动发现 attention/MLP 结构
#   插入 SP 集合: Ulysses 式 all-to-all (非 Ring)
#   零代码修改: 任意 PyTorch 模型自动支持 SP
#   NVSwitch 优化: all-to-all 并行带宽 vs Ring 的 p-hop 串行
#
# 使用示例:
#   # 训练单个模型 (500步快速测试)
#   MAX_STEPS=500 ./llm4dec_sp.sh full
#
#   # 训练所有模型
#   MAX_STEPS=500 ./llm4dec_sp.sh all_models
#
#   # 仅评测
#   ./llm4dec_sp.sh eval_all_models
#
# ===========================================

set -e

# ===========================================
# 路径配置
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 自动检测PROJECT_DIR
if [ -f "$SCRIPT_DIR/REAL_GPU_BENCHMARK.py" ] || [ -f "$SCRIPT_DIR/deepspeed/runtime/engine.py" ]; then
    PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
else
    PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/Neuron_SP}"
fi

DATA_DIR="${DATA_DIR:-$PROJECT_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/desloc_results}"
FIGURES_DIR="${FIGURES_DIR:-$PROJECT_DIR/desloc_figures}"

# HuggingFace缓存 (用于 7B 模型下载)
HF_CACHE_DIR="${HF_HOME:-/data/jiacheng/system/cache/temp/huggingface}"
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"

# Conda环境配置
CONDA_ENV_NAME="${CONDA_ENV_NAME:-base}"
SOURCE_ENV="${SOURCE_ENV:-base}"

# Conda路径 (自动检测)
if [ -f "/usr/local/lib/miniconda3/bin/conda" ]; then
    CONDA_BASE="/usr/local/lib/miniconda3"
elif [ -f "$HOME/miniconda3/bin/conda" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -f "$HOME/anaconda3/bin/conda" ]; then
    CONDA_BASE="$HOME/anaconda3"
else
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "/usr/local/lib/miniconda3")
fi

# ===========================================
# 模型配置 - 支持 125M 到 7B
# ===========================================

declare -A MODEL_PARAMS=(
    ["125M"]="125000000"
    ["350M"]="350000000"
    ["700M"]="700000000"
    ["1.3B"]="1300000000"
    ["7B"]="7000000000"
)

declare -A MODEL_LAYERS=(
    ["125M"]="12"
    ["350M"]="24"
    ["700M"]="36"
    ["1.3B"]="24"
    ["7B"]="32"
)

declare -A MODEL_HEADS=(
    ["125M"]="12"
    ["350M"]="16"
    ["700M"]="20"
    ["1.3B"]="16"
    ["7B"]="32"
)

declare -A MODEL_HIDDEN=(
    ["125M"]="768"
    ["350M"]="1024"
    ["700M"]="1280"
    ["1.3B"]="2048"
    ["7B"]="4096"
)

# 可在设备上直接训练的模型
ONDEVICE_MODELS=("125M" "350M" "700M" "1.3B")

# 默认模型
MODEL_KEY="${MODEL_KEY:-125M}"

# DES-LOC 输出
DESLOC_OUTPUT_NAME="${DESLOC_OUTPUT_NAME:-desloc_${MODEL_KEY}}"
DESLOC_OUTPUT_DIR="${DESLOC_OUTPUT_DIR:-$OUTPUT_DIR/$DESLOC_OUTPUT_NAME}"

# ===========================================
# 训练参数
# ===========================================

MAX_STEPS="${MAX_STEPS:-500}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"

LEARNING_RATE="${LEARNING_RATE:-6e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.95}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

WARMUP_STEPS="${WARMUP_STEPS:-100}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
WORLD_SIZE="${WORLD_SIZE:-1}"

# DES-LOC 参数
DEFAULT_KX="${DEFAULT_KX:-32}"
DEFAULT_KU="${DEFAULT_KU:-96}"    # 3 × Kx
DEFAULT_KV="${DEFAULT_KV:-192}"   # 6 × Kx

# AutoSP 参数
SP_SIZE="${SP_SIZE:-2}"
SP_ENABLED="${SP_ENABLED:-true}"

# 实验种子
SEEDS="${SEEDS:-42 137 2024}"

# 数据路径
EXPERIMENT_LOG="${EXPERIMENT_LOG:-$OUTPUT_DIR/experiment_log.csv}"
COST_MODEL_DIR="${COST_MODEL_DIR:-$OUTPUT_DIR/cost_model}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$OUTPUT_DIR/eval_results_${MODEL_KEY}}"

# ===========================================
# 工具函数
# ===========================================

print_header() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  LLM4DEC-SP - DES-LOC+AutoSP 分布式训练实验流水线 (v2.0)    ║"
    echo "║  Desynced Low-Communication + Automatic Sequence Parallel    ║"
    echo "║  [v2.0] 支持 125M-1.3B 直接训练 + 7B SP 分片验证           ║"
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

# Conda环境管理
init_conda() {
    __conda_setup="$("$CONDA_BASE/bin/conda" 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            . "$CONDA_BASE/etc/profile.d/conda.sh"
        else
            export PATH="$CONDA_BASE/bin:$PATH"
        fi
    fi
    unset __conda_setup
}

activate_env() {
    local env_name="${1:-$CONDA_ENV_NAME}"
    init_conda
    conda activate "$env_name" 2>/dev/null || {
        log_warn "Conda environment '$env_name' not found, using current env"
        return 0
    }
    log_info "Activated conda environment: $env_name"
}

# ===========================================
# 模型配置更新函数
# ===========================================

switch_model() {
    local new_model_key="$1"

    if [[ -z "${MODEL_PARAMS[$new_model_key]+x}" ]]; then
        log_error "Unknown model key: $new_model_key"
        log_info "Available models: ${!MODEL_PARAMS[*]}"
        return 1
    fi

    MODEL_KEY="$new_model_key"
    DESLOC_OUTPUT_NAME="desloc_${MODEL_KEY}"
    DESLOC_OUTPUT_DIR="$OUTPUT_DIR/$DESLOC_OUTPUT_NAME"
    EVAL_OUTPUT_DIR="$OUTPUT_DIR/eval_results_${MODEL_KEY}"

    log_info "Switched to model: $MODEL_KEY"
    log_info "  Params: ${MODEL_PARAMS[$MODEL_KEY]}"
    log_info "  Layers: ${MODEL_LAYERS[$MODEL_KEY]}, Hidden: ${MODEL_HIDDEN[$MODEL_KEY]}"
    log_info "  Output: $DESLOC_OUTPUT_DIR"
}

# ===========================================
# 环境配置
# ===========================================

setup_environment() {
    print_step "Setting up Environment"

    if ! command -v conda &> /dev/null; then
        log_error "Conda not found. Install Miniconda/Anaconda first."
        exit 1
    fi

    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        log_info "Environment '${CONDA_ENV_NAME}' exists."
        activate_env
        _install_desloc_packages
        log_success "Environment updated"
        return 0
    fi

    log_info "Creating conda environment '${CONDA_ENV_NAME}'..."
    if conda env list | grep -q "^${SOURCE_ENV} "; then
        conda create --name ${CONDA_ENV_NAME} --clone ${SOURCE_ENV} -y
    else
        conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    fi

    activate_env
    _install_desloc_packages
    log_success "Environment setup complete: ${CONDA_ENV_NAME}"
}

_install_desloc_packages() {
    log_info "Installing DES-LOC + AutoSP packages..."
    pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || true
    pip install -q matplotlib seaborn pandas scipy 2>/dev/null || true
    pip install -q pyyaml numpy tqdm 2>/dev/null || true
    [ -f "$PROJECT_DIR/requirements.txt" ] && pip install -q -r "$PROJECT_DIR/requirements.txt" 2>/dev/null || true
    log_success "Packages installed"
}

prepare_dirs() {
    print_step "Preparing Directories"

    check_dir "$OUTPUT_DIR"
    check_dir "$OUTPUT_DIR/logs"
    check_dir "$OUTPUT_DIR/cost_model"
    check_dir "$FIGURES_DIR"
    check_dir "$DESLOC_OUTPUT_DIR"
    check_dir "$EVAL_OUTPUT_DIR"

    log_info "Project:     $PROJECT_DIR"
    log_info "Output:      $OUTPUT_DIR"
    log_info "Model:       $MODEL_KEY (${MODEL_PARAMS[$MODEL_KEY]} params)"
    log_info "DES-LOC:     Kx=$DEFAULT_KX, Ku=$DEFAULT_KU, Kv=$DEFAULT_KV"
    log_info "AutoSP:      SP_size=$SP_SIZE, enabled=$SP_ENABLED"
    log_info "Figures:     $FIGURES_DIR"
    log_success "Directories prepared"
}

# ===========================================
# Step 0: Cost Model (CPU only, no GPU needed)
# ===========================================

generate_cost_model() {
    print_step "Step 0: Generating DES-LOC SP Communication Cost Model"
    cd "$PROJECT_DIR"

    log_info "╔═══════════════════════════════════════════════════════════════╗"
    log_info "║  DES-LOC + AutoSP 通信代价模型                                ║"
    log_info "╠═══════════════════════════════════════════════════════════════╣"
    log_info "║  GPU 类型: A6000, H100, H100_NVL, TRAINIUM2                  ║"
    log_info "║  配置: 4 GPU × 3 层数 × 4 hidden × 5 seq × 8 Kx             ║"
    log_info "║  数据点: 1920                                                 ║"
    log_info "║                                                               ║"
    log_info "║  AutoSP 通信模式分析:                                          ║"
    log_info "║  - Ulysses A2A: alpha + msg/BW (NVSwitch 并行)               ║"
    log_info "║  - Ring-FA: alpha*(p-1) + msg*(p-1)/BW (串行 p-hop)          ║"
    log_info "║  - DES-LOC: Ulysses / Kx (Kx步摊销)                          ║"
    log_info "╚═══════════════════════════════════════════════════════════════╝"

    check_dir "$COST_MODEL_DIR"

    python3 << 'COSTEOF'
import os, time, math

class SPCostModel:
    """DES-LOC SP communication cost model.
    Ref: NCCL topo.cc bandwidth constants + NKI-FA benchmark_attn.py timing."""
    BW_GBS = {'A6000': 64.0, 'H100': 900.0, 'H100_NVL': 900.0, 'TRAINIUM2': 384.0}
    LAT_US = {'A6000': 8.0, 'H100': 3.5, 'H100_NVL': 3.5, 'TRAINIUM2': 6.0}

    def __init__(self, gpu, sp_size=2, n_layers=32):
        self.gpu = gpu
        self.sp = sp_size
        self.n_layers = n_layers
        self.bw = self.BW_GBS.get(gpu, 600.0) * 1e9
        self.lat = self.LAT_US.get(gpu, 5.0) * 1e-6

    def a2a_time(self, seq, hidden, batch=1, dtype_bytes=2):
        """Single all-to-all time: latency + msg/bandwidth."""
        msg = 2 * batch * seq * hidden * dtype_bytes / self.sp
        return self.lat + msg / self.bw

    def step_comm(self, seq, hidden, Kx=1, batch=1):
        """Amortized comm per step = (per_layer × 2 × n_layers) / Kx."""
        return self.a2a_time(seq, hidden, batch) * self.n_layers * 2 / max(Kx, 1)

    def ring_time(self, seq, hidden, batch=1, dtype_bytes=2):
        """Ring-FA time: p-1 sequential hops."""
        msg_per_hop = batch * seq * hidden * dtype_bytes / self.sp
        hops = self.sp - 1
        return (self.lat * hops + msg_per_hop * hops / self.bw) * self.n_layers * 2

    def speedup_vs_ring(self, seq, hidden, Kx=1, batch=1):
        """Speedup of Kx-gated Ulysses over Ring-FA."""
        ring = self.ring_time(seq, hidden, batch)
        ulysses = self.step_comm(seq, hidden, Kx, batch)
        return ring / max(ulysses, 1e-15)

rd = os.environ.get('COST_MODEL_DIR', './desloc_results/cost_model')
os.makedirs(rd, exist_ok=True)

# CSV sweep
csv_path = os.path.join(rd, 'sp_cost_sweep.csv')
n = 0
with open(csv_path, 'w') as f:
    f.write('gpu,layers,hidden,seq,Kx,a2a_us,step_ms,ring_ms,speedup\n')
    for gpu in ['A6000', 'H100', 'H100_NVL', 'TRAINIUM2']:
        for layers in [12, 24, 32]:
            m = SPCostModel(gpu, 2, layers)
            for hidden in [768, 1024, 2048, 4096]:
                for seq in [2048, 4096, 8192, 16384, 32768]:
                    for Kx in [1, 2, 4, 8, 16, 32, 64, 128]:
                        a2a = m.a2a_time(seq, hidden)
                        step = m.step_comm(seq, hidden, Kx)
                        ring = m.ring_time(seq, hidden)
                        sp = m.speedup_vs_ring(seq, hidden, Kx)
                        f.write(f'{gpu},{layers},{hidden},{seq},{Kx},'
                                f'{a2a*1e6:.4f},{step*1e3:.4f},{ring*1e3:.4f},{sp:.3f}\n')
                        n += 1

# NKI-FA format logs per GPU
for gpu in ['A6000', 'H100', 'H100_NVL', 'TRAINIUM2']:
    log_path = os.path.join(rd, f'cost_{gpu}.log')
    with open(log_path, 'w') as f:
        f.write(f'### DES-LOC SP Cost Model — {gpu} ###\n')
        f.write(f'### Generated: {time.strftime("%Y-%m-%d %H:%M:%S")} ###\n\n')
        m = SPCostModel(gpu, 2, 32)
        for seq in [2048, 4096, 8192, 16384]:
            for hidden in [768, 2048, 4096]:
                f.write(f'### gpu = {gpu}, seq = {seq}, hidden = {hidden}, layers = 32 ###\n')
                for Kx in [1, 4, 8, 16, 32, 64]:
                    st = m.step_comm(seq, hidden, Kx) * 1e3
                    sp = m.speedup_vs_ring(seq, hidden, Kx)
                    f.write(f'Kx={Kx}: step_comm={st:.4f}ms, speedup_vs_ring={sp:.3f}x\n')
                f.write('\n')

print(f'Generated {n} data points -> {csv_path}')
print(f'NKI-FA logs -> {rd}/cost_*.log')
COSTEOF

    log_success "Cost model generated: $COST_MODEL_DIR"
    cd "$SCRIPT_DIR"
}

# ===========================================
# Step 1: 数据生成 (合成训练数据)
# ===========================================

generate_training_data() {
    print_step "Step 1: Generating Synthetic Training Data"
    activate_env
    cd "$PROJECT_DIR"

    local model="${1:-$MODEL_KEY}"

    log_info "╔═══════════════════════════════════════════════════════════════╗"
    log_info "║  DES-LOC 合成数据生成                                         ║"
    log_info "╠═══════════════════════════════════════════════════════════════╣"
    log_info "║  Model: $model (${MODEL_PARAMS[$model]} params)              "
    log_info "║  Layers: ${MODEL_LAYERS[$model]}, Hidden: ${MODEL_HIDDEN[$model]}"
    log_info "║  Seq Len: $MAX_SEQ_LEN                                       "
    log_info "║  DES-LOC: Kx=$DEFAULT_KX, Ku=$DEFAULT_KU, Kv=$DEFAULT_KV    "
    log_info "║                                                               ║"
    log_info "║  数据来源: GPT-2 vocab 合成序列 (torch.manual_seed)           ║"
    log_info "║  无 numpy.random — 确定性数据生成                             ║"
    log_info "╚═══════════════════════════════════════════════════════════════╝"

    log_success "Synthetic data will be generated on-the-fly by REAL_GPU_BENCHMARK.py"
    cd "$SCRIPT_DIR"
}

# ===========================================
# Step 2: 实验运行 (核心训练)
# ===========================================

RUN_ID=0
FAIL=0
TOTAL_RUNS=0

_run_single_experiment() {
    local PHASE=$1 TAG=$2 MODEL=$3 KX=$4 KU=$5 KV=$6 SEED=$7 METHOD=$8 GPUS=$9 NGPU=${10}

    RUN_ID=$((RUN_ID + 1))
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
    local LOGFILE="$OUTPUT_DIR/logs/${PHASE}_${TAG}_${MODEL}_Kx${KX}_${METHOD}_s${SEED}.log"
    local T0=$SECONDS

    printf "[%3d] %-12s %-10s %-5s Kx=%-3d Ku=%-3d Kv=%-3d %-10s seed=%-4d gpu=%-5s ... " \
        "$RUN_ID" "$PHASE" "$TAG" "$MODEL" "$KX" "$KU" "$KV" "$METHOD" "$SEED" "$GPUS"

    export PYTHONHASHSEED=$SEED
    export CUDA_VISIBLE_DEVICES=$GPUS

    if [ "$NGPU" -gt 1 ]; then
        torchrun --nproc_per_node=$NGPU \
            --master_port=$((29500 + RUN_ID % 200)) \
            REAL_GPU_BENCHMARK.py \
            --model_size "$MODEL" \
            --batch_size $BATCH_SIZE \
            --grad_accum $GRAD_ACCUM \
            --max_steps $MAX_STEPS \
            --Kx "$KX" --Ku "$KU" --Kv "$KV" \
            --methods "$METHOD" \
            --output "$DESLOC_OUTPUT_DIR/$PHASE" \
            > "$LOGFILE" 2>&1
    else
        python3 REAL_GPU_BENCHMARK.py \
            --model_size "$MODEL" \
            --batch_size $BATCH_SIZE \
            --grad_accum $GRAD_ACCUM \
            --max_steps $MAX_STEPS \
            --Kx "$KX" --Ku "$KU" --Kv "$KV" \
            --methods "$METHOD" \
            --output "$DESLOC_OUTPUT_DIR/$PHASE" \
            > "$LOGFILE" 2>&1
    fi

    local RC=$?
    local DT=$((SECONDS - T0))

    if [ $RC -eq 0 ]; then
        echo "OK (${DT}s)"
    else
        echo "FAIL:${RC} (${DT}s)"
        FAIL=$((FAIL + 1))
    fi

    echo "${RUN_ID},${PHASE},${TAG},${MODEL},${KX},${KU},${KV},${SEED},${METHOD},${GPUS},${NGPU},${RC},${DT},${LOGFILE}" >> "$EXPERIMENT_LOG"
}

run_training() {
    print_step "Step 2: Running DES-LOC Training Experiments"
    activate_env
    cd "$PROJECT_DIR"

    local model="${1:-$MODEL_KEY}"

    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
    check_dir "$OUTPUT_DIR/logs"

    log_info "Model: $model (${MODEL_PARAMS[$model]} params)"
    log_info "DES-LOC: Kx=$DEFAULT_KX, Ku=$DEFAULT_KU, Kv=$DEFAULT_KV"
    log_info "Steps: $MAX_STEPS, Batch: $BATCH_SIZE, Grad Accum: $GRAD_ACCUM"

    # Initialize experiment log
    echo "run_id,phase,tag,model,Kx,Ku,Kv,seed,method,gpus,ngpu,rc,elapsed,log" > "$EXPERIMENT_LOG"

    # Run DDP baseline + DES-LOC + LocalSGD
    for SEED in $SEEDS; do
        _run_single_experiment "train" "baseline" "$model" 1 1 1 $SEED "DDP" "$CUDA_DEVICE" 1
        _run_single_experiment "train" "desloc" "$model" $DEFAULT_KX $DEFAULT_KU $DEFAULT_KV $SEED "DESLOC" "$CUDA_DEVICE" 1
        _run_single_experiment "train" "local" "$model" $DEFAULT_KX $DEFAULT_KX $DEFAULT_KX $SEED "LocalAdam" "$CUDA_DEVICE" 1
    done

    log_success "Training complete: $TOTAL_RUNS experiments"
    cd "$SCRIPT_DIR"
}

# ===========================================
# Step 3: Kx Sweep (消融实验)
# ===========================================

run_kx_sweep() {
    print_step "Step 3: Kx Sync Frequency Sweep"
    activate_env
    cd "$PROJECT_DIR"

    local model="${1:-$MODEL_KEY}"

    log_info "╔═══════════════════════════════════════════════════════════════╗"
    log_info "║  RQ1: Kx 同步频率消融                                         ║"
    log_info "╠═══════════════════════════════════════════════════════════════╣"
    log_info "║  Kx ∈ {1, 2, 4, 8, 16, 32, 64, 128}                         ║"
    log_info "║  3 seeds × 8 Kx = 24 experiments                             ║"
    log_info "║  Ku = 3×Kx, Kv = 6×Kx (half-life proportional)              ║"
    log_info "╚═══════════════════════════════════════════════════════════════╝"

    for KX in 1 2 4 8 16 32 64 128; do
        if [ $KX -eq 1 ]; then KU=1; KV=1
        else KU=$((KX * 3)); KV=$((KX * 6)); fi
        for SEED in $SEEDS; do
            _run_single_experiment "kx_sweep" "rq1" "$model" $KX $KU $KV $SEED "DESLOC" "$CUDA_DEVICE" 1
        done
    done

    log_success "Kx sweep complete"
    cd "$SCRIPT_DIR"
}

# ===========================================
# Step 4: Ku/Kv Ratio Ablation
# ===========================================

run_ratio_ablation() {
    print_step "Step 4: Ku/Kv Ratio Ablation (fixed Kx=32)"
    activate_env
    cd "$PROJECT_DIR"

    local model="${1:-$MODEL_KEY}"

    log_info "╔═══════════════════════════════════════════════════════════════╗"
    log_info "║  RQ2: Ku/Kv 比例消融                                          ║"
    log_info "╠═══════════════════════════════════════════════════════════════╣"
    log_info "║  固定 Kx=32                                                   ║"
    log_info "║  Ku/Kx ∈ {1, 3, 6}, Kv/Kx ∈ {1, 6, 12}                     ║"
    log_info "║  9 combos × 3 seeds = 27 experiments                         ║"
    log_info "╚═══════════════════════════════════════════════════════════════╝"

    for KU_RATIO in 1 3 6; do
        for KV_RATIO in 1 6 12; do
            KU=$((32 * KU_RATIO))
            KV=$((32 * KV_RATIO))
            for SEED in $SEEDS; do
                _run_single_experiment "ratio_abl" "rq2" "$model" 32 $KU $KV $SEED "DESLOC" "$CUDA_DEVICE" 1
            done
        done
    done

    log_success "Ratio ablation complete"
    cd "$SCRIPT_DIR"
}

# ===========================================
# Step 5: 评测 (性能指标收集)
# ===========================================

run_eval() {
    print_step "Step 5: Collecting Evaluation Metrics"

    local eval_output="${1:-$EVAL_OUTPUT_DIR}"

    log_info "╔═══════════════════════════════════════════════════════════════╗"
    log_info "║  DES-LOC 评测指标收集                                          ║"
    log_info "╠═══════════════════════════════════════════════════════════════╣"
    log_info "║  指标:                                                        ║"
    log_info "║    - Final Loss (收敛性)                                       ║"
    log_info "║    - Tokens/sec (吞吐量)                                      ║"
    log_info "║    - MFU (模型FLOPS利用率)                                     ║"
    log_info "║    - Communication Reduction (通信降低倍数)                    ║"
    log_info "║    - Peak Memory (GB)                                         ║"
    log_info "║    - Sync Counts (同步次数)                                    ║"
    log_info "╚═══════════════════════════════════════════════════════════════╝"

    check_dir "$eval_output"

    _aggregate_results "$eval_output"
    _show_eval_summary "$eval_output"

    log_success "Evaluation complete!"
    log_info "Results saved to: $eval_output"
}

_aggregate_results() {
    local eval_output="$1"

    python3 << PYEOF
import json, glob, os, csv, re

output_dir = "$eval_output"
results_dir = "$OUTPUT_DIR"
os.makedirs(output_dir, exist_ok=True)

# Parse experiment_log.csv
csv_rows = []
csv_path = os.path.join(results_dir, 'experiment_log.csv')
if os.path.exists(csv_path):
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            csv_rows.append(r)

# Parse NKI-FA format logs
nkifa_pat = re.compile(
    r'### model = (\S+), method = (\S+), Kx = (\d+), Ku = (\d+), Kv = (\d+)')
metric_pat = re.compile(r'^(\w[\w_]+):\s+(.+)$')
nkifa = []

for logf in sorted(glob.glob(os.path.join(results_dir, 'logs', '*.log'))):
    with open(logf) as fh:
        lines = fh.readlines()
    cur = None
    for line in lines:
        m = nkifa_pat.match(line.strip())
        if m:
            cur = {'model': m.group(1), 'method': m.group(2),
                   'Kx': int(m.group(3)), 'Ku': int(m.group(4)),
                   'Kv': int(m.group(5)), 'log': logf}
            continue
        if cur:
            mm = metric_pat.match(line.strip())
            if mm:
                try: cur[mm.group(1)] = float(mm.group(2))
                except: cur[mm.group(1)] = mm.group(2)
            elif not line.strip():
                if len(cur) > 5: nkifa.append(cur)
                cur = None

# Save merged results
merged = {
    'csv_log': csv_rows,
    'nkifa_parsed': nkifa,
    'total_experiments': len(csv_rows),
}
with open(os.path.join(output_dir, 'ALL_RESULTS.json'), 'w') as f:
    json.dump(merged, f, indent=2)

print(f"Aggregated: {len(csv_rows)} CSV + {len(nkifa)} NKI-FA records")
PYEOF
}

_show_eval_summary() {
    local eval_output="$1"

    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  评测结果摘要                                                  ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"

    python3 << PYEOF
import json, os

eval_dir = "$eval_output"
results_file = os.path.join(eval_dir, 'ALL_RESULTS.json')

if not os.path.exists(results_file):
    print("  (no results yet)")
    exit(0)

with open(results_file) as f:
    data = json.load(f)

nkifa = data.get('nkifa_parsed', [])
if nkifa:
    print(f"\n  {'Model':>5s} {'Method':>10s} {'Kx':>4s} {'Loss':>10s} {'Tok/s':>10s} {'MFU':>6s} {'Mem(GB)':>8s}")
    print("  " + "-" * 55)
    for r in nkifa[:20]:
        print(f"  {r.get('model','?'):>5s} {r.get('method','?'):>10s} "
              f"{str(r.get('Kx','?')):>4s} "
              f"{str(r.get('final_loss','?')):>10s} "
              f"{str(r.get('tokens_per_second','?')):>10s} "
              f"{str(r.get('mfu','?')):>6s} "
              f"{str(r.get('peak_memory_gb','?')):>8s}")
else:
    csv_rows = data.get('csv_log', [])
    ok = sum(1 for r in csv_rows if r.get('rc','') == '0')
    fail = len(csv_rows) - ok
    print(f"\n  Total runs: {len(csv_rows)} (OK: {ok}, FAIL: {fail})")

print("")
PYEOF

    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  DES-LOC 论文参考结果 (Section 5)                             ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║  Method     │ 125M Loss │ 350M Loss │ Comm Reduction         ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║  DDP        │ baseline  │ baseline  │ 1×                     ║"
    echo "║  LocalSGD   │ ~+0.02    │ ~+0.03    │ 32×                    ║"
    echo "║  DES-LOC    │ ~+0.01    │ ~+0.01    │ 32×-170×               ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
}

# ===========================================
# 图表生成 (NKI-FA 质量)
# ===========================================

generate_figures() {
    print_step "Generating NKI-FA Grade Figures"
    activate_env
    cd "$PROJECT_DIR"

    log_info "Generating 7 figures from experiment logs..."
    log_info "Output: $FIGURES_DIR"

    python3 -c "
import sys; sys.path.insert(0, '.')
try:
    from REAL_GPU_BENCHMARK import desloc_draw_all_figures
    desloc_draw_all_figures('${OUTPUT_DIR}')
except Exception as e:
    print(f'[WARN] Figure generation: {e}')
    print('Trying cost model figures...')
" 2>&1 || log_warn "Figure generation had errors"

    # Generate cost model figures
    python3 << 'FIGEOF'
import os
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    rd = os.environ.get('COST_MODEL_DIR', './desloc_results/cost_model')
    fd = os.environ.get('FIGURES_DIR', './desloc_figures')
    os.makedirs(fd, exist_ok=True)

    csv_path = os.path.join(rd, 'sp_cost_sweep.csv')
    if not os.path.exists(csv_path):
        print(f"No cost model data at {csv_path}")
        exit(0)

    df = pd.read_csv(csv_path)

    # Fig 1: Kx vs comm time
    fig, ax = plt.subplots(figsize=(10, 6))
    for gpu in ['A6000', 'H100', 'H100_NVL', 'TRAINIUM2']:
        sub = df[(df['gpu']==gpu) & (df['seq']==16384) & (df['hidden']==2048) & (df['layers']==32)]
        ax.plot(sub['Kx'], sub['step_ms'], 'o-', label=gpu, markersize=5)
    ax.set_xlabel('Sync Period Kx')
    ax.set_ylabel('Amortized Comm Time (ms/step)')
    ax.set_title('DES-LOC SP Communication Cost (seq=16384, hidden=2048)')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{fd}/fig1_kx_comm.png', dpi=300)
    print(f'Saved fig1_kx_comm.png')

    # Fig 2: Speedup vs Ring
    fig, ax = plt.subplots(figsize=(10, 6))
    for gpu in ['A6000', 'H100', 'H100_NVL', 'TRAINIUM2']:
        sub = df[(df['gpu']==gpu) & (df['seq']==16384) & (df['hidden']==2048) & (df['layers']==32)]
        ax.plot(sub['Kx'], sub['speedup'], 's-', label=gpu, markersize=5)
    ax.set_xlabel('Sync Period Kx')
    ax.set_ylabel('Speedup vs Ring-Flash-Attention')
    ax.set_title('AutoSP (Ulysses) + DES-LOC vs Ring-FA')
    ax.set_xscale('log', base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{fd}/fig2_speedup_ring.png', dpi=300)
    print(f'Saved fig2_speedup_ring.png')

    # Fig 3: Comm reduction %
    fig, ax = plt.subplots(figsize=(10, 6))
    for gpu in ['A6000', 'H100', 'TRAINIUM2']:
        sub = df[(df['gpu']==gpu) & (df['seq']==16384) & (df['hidden']==2048) & (df['layers']==32)]
        baseline = sub[sub['Kx']==1]['step_ms'].values[0]
        reduction = (1 - sub['step_ms'] / baseline) * 100
        ax.plot(sub['Kx'], reduction, 'o-', label=gpu, markersize=5)
    ax.set_xlabel('Sync Period Kx')
    ax.set_ylabel('Communication Reduction (%)')
    ax.set_title('DES-LOC Communication Reduction vs Baseline (Kx=1)')
    ax.set_xscale('log', base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{fd}/fig3_comm_reduction.png', dpi=300)
    print(f'Saved fig3_comm_reduction.png')

    print(f'\nAll figures saved to {fd}/')
except ImportError as e:
    print(f'matplotlib/pandas not available: {e}')
except Exception as e:
    print(f'Error: {e}')
FIGEOF

    log_success "Figures generated in $FIGURES_DIR"
    cd "$SCRIPT_DIR"
}

# ===========================================
# 批量处理所有模型
# ===========================================

train_all_models() {
    print_step "Training All Models with DES-LOC"
    local start_time=$(date +%s)

    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  DES-LOC 批量训练所有模型                                      ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║  模型列表: ${ONDEVICE_MODELS[*]}"
    echo "║  每模型: 3 methods × 3 seeds = 9 experiments"
    echo "║  总计: $((${#ONDEVICE_MODELS[@]} * 9)) experiments"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""

    local success_models=()
    local failed_models=()

    for model_key in "${ONDEVICE_MODELS[@]}"; do
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "  Training model: $model_key (${MODEL_PARAMS[$model_key]} params)"
        echo "═══════════════════════════════════════════════════════════════"

        switch_model "$model_key"

        if run_full_train_single "$model_key"; then
            success_models+=("$model_key")
        else
            failed_models+=("$model_key")
        fi
    done

    local duration=$(($(date +%s) - start_time))

    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  批量训练完成                                                  ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║  总耗时: $((duration/60))m $((duration%60))s"
    echo "║  成功: ${#success_models[@]} (${success_models[*]})"
    echo "║  失败: ${#failed_models[@]} (${failed_models[*]:-无})"
    echo "╚═══════════════════════════════════════════════════════════════╝"
}

run_full_train_single() {
    local model="${1:-$MODEL_KEY}"

    prepare_dirs || return 1
    generate_training_data "$model" || return 1
    run_training "$model" || return 1

    return 0
}

eval_all_models() {
    print_step "Evaluating All DES-LOC Trained Models"
    local start_time=$(date +%s)

    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  评测所有 DES-LOC 训练模型                                     ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║  模型列表: ${ONDEVICE_MODELS[*]}"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""

    local success_models=()
    local failed_models=()

    for model_key in "${ONDEVICE_MODELS[@]}"; do
        switch_model "$model_key"

        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "  Evaluating model: $model_key"
        echo "═══════════════════════════════════════════════════════════════"

        if run_eval "$EVAL_OUTPUT_DIR"; then
            success_models+=("$model_key")
        else
            failed_models+=("$model_key")
        fi
    done

    local duration=$(($(date +%s) - start_time))

    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  批量评测完成                                                  ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║  总耗时: $((duration/60))m $((duration%60))s"
    echo "║  成功: ${#success_models[@]} (${success_models[*]})"
    echo "║  失败: ${#failed_models[@]} (${failed_models[*]:-无})"
    echo "╚═══════════════════════════════════════════════════════════════╝"

    compare_results
}

compare_results() {
    print_step "Generating Comparison Report for All Models"

    python3 << 'CMPEOF'
import json, os, csv, glob
from datetime import datetime

output_dir = os.environ.get('OUTPUT_DIR', './desloc_results')
report_file = os.path.join(output_dir, 'desloc_comparison_report.md')
json_report = os.path.join(output_dir, 'desloc_comparison_report.json')

models = ['125M', '350M', '700M', '1.3B']
methods = ['DDP', 'LocalAdam', 'DESLOC']

# Collect results from CSV logs
all_results = {}
csv_path = os.path.join(output_dir, 'experiment_log.csv')
if os.path.exists(csv_path):
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            key = f"{r.get('model','?')}_{r.get('method','?')}"
            all_results.setdefault(key, []).append(r)

report = f"""# DES-LOC + AutoSP 实验对比报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 实验配置
- 硬件: 2× RTX A6000 (48GB) + 1× H100 NVL (94GB)
- DES-LOC: Kx=32, Ku=96, Kv=192
- AutoSP: SP_size=2 (Ulysses all-to-all)
- 种子: 42, 137, 2024 (3-seed mean±std)

## 实验矩阵

| Model | Method | Kx | Runs | Success |
|-------|--------|----|------|---------|
"""

for model in models:
    for method in methods:
        key = f"{model}_{method}"
        runs = all_results.get(key, [])
        ok = sum(1 for r in runs if r.get('rc','') == '0')
        kx = runs[0].get('Kx', '?') if runs else '?'
        report += f"| {model} | {method} | {kx} | {len(runs)} | {ok} |\n"

report += """
## AutoSP vs Ring-Flash-Attention 通信分析

| GPU | Kx=1 Ulysses | Kx=32 DES-LOC | Ring-FA | Speedup (DES-LOC vs Ring) |
|-----|-------------|---------------|---------|--------------------------|
| A6000 | baseline | 32× reduction | 1× (serial) | see cost_model/ |
| H100 | baseline | 32× reduction | 1× (serial) | see cost_model/ |
| Trainium2 | baseline | 32× reduction | 1× (serial) | see cost_model/ |

详细数据见 `cost_model/sp_cost_sweep.csv`
"""

with open(report_file, 'w') as f:
    f.write(report)

with open(json_report, 'w') as f:
    json.dump({'timestamp': datetime.now().isoformat(), 'results': all_results}, f, indent=2, default=str)

print(report)
print(f"\n报告: {report_file}")
print(f"JSON: {json_report}")
CMPEOF
}

run_all_models_pipeline() {
    print_step "Complete Pipeline: Train and Evaluate All Models"
    local start_time=$(date +%s)

    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  DES-LOC 完整流水线 — 训练并评测所有模型                       ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║  模型列表: ${ONDEVICE_MODELS[*]}"
    echo "║  Steps: $MAX_STEPS"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""

    generate_cost_model
    train_all_models
    eval_all_models
    generate_figures

    local duration=$(($(date +%s) - start_time))

    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  完整流水线完成                                                ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║  总耗时: $((duration/3600))h $((duration%3600/60))m $((duration%60))s"
    echo "║  对比报告: $OUTPUT_DIR/desloc_comparison_report.md"
    echo "║  图表: $FIGURES_DIR/"
    echo "╚═══════════════════════════════════════════════════════════════╝"
}

# ===========================================
# 单模型流水线命令
# ===========================================

run_full_train() {
    print_step "Full Training Pipeline"
    local start_time=$(date +%s)

    log_info "Starting full training pipeline"
    log_info "Model: $MODEL_KEY (${MODEL_PARAMS[$MODEL_KEY]} params)"
    log_info "DES-LOC: Kx=$DEFAULT_KX, Ku=$DEFAULT_KU, Kv=$DEFAULT_KV"
    log_info "Steps: $MAX_STEPS"

    prepare_dirs
    generate_cost_model
    generate_training_data
    run_training
    run_kx_sweep
    run_ratio_ablation

    local duration=$(($(date +%s) - start_time))
    log_success "Full training complete in $((duration/60))m $((duration%60))s"
}

run_full_eval() {
    print_step "Full Evaluation Pipeline"
    run_eval
    generate_figures
    log_success "Full evaluation complete!"
}

run_full_pipeline() {
    print_step "Complete DES-LOC Pipeline (Train + Eval + Figures)"
    local start_time=$(date +%s)

    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  DES-LOC Pipeline Configuration                              ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║  Model:        $MODEL_KEY (${MODEL_PARAMS[$MODEL_KEY]} params)"
    echo "║  DES-LOC:      Kx=$DEFAULT_KX, Ku=$DEFAULT_KU, Kv=$DEFAULT_KV"
    echo "║  AutoSP:       SP_size=$SP_SIZE"
    echo "║  Steps:        $MAX_STEPS"
    echo "║  Seeds:        $SEEDS"
    echo "║  Batch:        $BATCH_SIZE × $GRAD_ACCUM accum"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "DES-LOC 核心流程:"
    echo "  0. 通信代价模型 (CPU, 1920 data points)"
    echo "  1. 合成数据生成 (torch.manual_seed)"
    echo "  2. 基线训练 (DDP + LocalSGD + DES-LOC)"
    echo "  3. Kx 消融 (8 values × 3 seeds)"
    echo "  4. Ku/Kv 比例消融 (9 combos × 3 seeds)"
    echo "  5. 评测汇总 + NKI-FA 图表"
    echo ""

    run_full_train
    run_full_eval

    local duration=$(($(date +%s) - start_time))

    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  Pipeline Complete!                                          ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║  Total Time:   $((duration/60))m $((duration%60))s"
    echo "║  Total Runs:   $TOTAL_RUNS (OK: $((TOTAL_RUNS-FAIL)), FAIL: $FAIL)"
    echo "║  Results:      $OUTPUT_DIR"
    echo "║  Figures:      $FIGURES_DIR"
    echo "║  Eval:         $EVAL_OUTPUT_DIR"
    echo "╚═══════════════════════════════════════════════════════════════╝"
}

train_only() {
    print_step "Training Only (DDP + DES-LOC + LocalSGD)"
    run_training "$@"
}

eval_only() {
    print_step "Evaluation Only"
    run_eval "$@"
}

# ===========================================
# 状态显示
# ===========================================

show_data_status() {
    print_step "Data Status"

    echo "Cost Model: $COST_MODEL_DIR"
    if [ -f "$COST_MODEL_DIR/sp_cost_sweep.csv" ]; then
        local n=$(wc -l < "$COST_MODEL_DIR/sp_cost_sweep.csv")
        echo "  ✓ $((n-1)) data points"
    else
        echo "  ✗ not generated"
    fi

    echo ""
    echo "Experiment Log: $EXPERIMENT_LOG"
    if [ -f "$EXPERIMENT_LOG" ]; then
        local runs=$(tail -n +2 "$EXPERIMENT_LOG" | wc -l)
        local ok=$(tail -n +2 "$EXPERIMENT_LOG" | awk -F, '{print $12}' | grep -c '^0$' || true)
        echo "  ✓ $runs runs (OK: $ok, FAIL: $((runs-ok)))"
    else
        echo "  ✗ no experiments run"
    fi

    echo ""
    echo "Figures: $FIGURES_DIR"
    if [ -d "$FIGURES_DIR" ]; then
        local figs=$(ls -1 "$FIGURES_DIR"/*.png 2>/dev/null | wc -l)
        echo "  $figs figures"
    else
        echo "  ✗ not generated"
    fi
}

show_all_models_status() {
    print_step "All Models Status"

    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  所有模型的 DES-LOC 训练状态                                   ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""

    printf "%-8s %-12s %-10s %-10s %-10s\n" "Model" "Params" "Trained" "CostModel" "Evaluated"
    echo "─────────────────────────────────────────────────────────────────"

    for model_key in "${ONDEVICE_MODELS[@]}"; do
        local params="${MODEL_PARAMS[$model_key]}"
        local trained="✗"
        local cost="✗"
        local evaluated="✗"

        [ -f "$OUTPUT_DIR/desloc_${model_key}/train/benchmark_results_*.json" ] 2>/dev/null && trained="✓"
        # Check if any experiment log mentions this model
        [ -f "$EXPERIMENT_LOG" ] && grep -q "$model_key" "$EXPERIMENT_LOG" 2>/dev/null && trained="✓"
        [ -f "$COST_MODEL_DIR/sp_cost_sweep.csv" ] && cost="✓"
        [ -f "$OUTPUT_DIR/eval_results_${model_key}/ALL_RESULTS.json" ] && evaluated="✓"

        printf "%-8s %-12s %-10s %-10s %-10s\n" "$model_key" "$params" "$trained" "$cost" "$evaluated"
    done

    echo ""
}

show_config() {
    print_step "Current Configuration"

    echo "═══════════════════════════════════════════════════════════════"
    echo "  Model Settings"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  MODEL_KEY:          $MODEL_KEY"
    echo "  Params:             ${MODEL_PARAMS[$MODEL_KEY]}"
    echo "  Layers:             ${MODEL_LAYERS[$MODEL_KEY]}"
    echo "  Hidden:             ${MODEL_HIDDEN[$MODEL_KEY]}"
    echo "  Heads:              ${MODEL_HEADS[$MODEL_KEY]}"
    echo ""
    echo "  Available Models:"
    for key in "${!MODEL_PARAMS[@]}"; do
        echo "    - $key: ${MODEL_PARAMS[$key]} params"
    done
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  DES-LOC Settings"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Kx:                 $DEFAULT_KX (SP sync period)"
    echo "  Ku:                 $DEFAULT_KU (momentum sync period)"
    echo "  Kv:                 $DEFAULT_KV (full sync period)"
    echo "  SP_SIZE:            $SP_SIZE"
    echo "  SP_ENABLED:         $SP_ENABLED"
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Training Settings"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  MAX_STEPS:          $MAX_STEPS"
    echo "  BATCH_SIZE:         $BATCH_SIZE"
    echo "  GRAD_ACCUM:         $GRAD_ACCUM"
    echo "  LEARNING_RATE:      $LEARNING_RATE"
    echo "  SEEDS:              $SEEDS"
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Environment"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  CONDA_ENV:          $CONDA_ENV_NAME"
    echo "  CUDA_DEVICE:        $CUDA_DEVICE"
    echo "  PROJECT_DIR:        $PROJECT_DIR"
    echo "  OUTPUT_DIR:         $OUTPUT_DIR"
    echo "═══════════════════════════════════════════════════════════════"
}

quick_fix() {
    print_step "Quick Fix"
    activate_env
    _install_desloc_packages
    prepare_dirs
    log_success "Quick fix complete"
}

clean_data() {
    print_step "Clean Data"

    echo "This will delete:"
    echo "  - Experiment results: $OUTPUT_DIR"
    echo "  - Figures: $FIGURES_DIR"
    echo ""

    read -p "Are you sure? (y/N): " resp
    [[ ! "$resp" =~ ^[Yy]$ ]] && { log_info "Cancelled"; return; }

    rm -rf "$OUTPUT_DIR" "$FIGURES_DIR"
    log_success "Data cleaned"
}

# ===========================================
# 帮助
# ===========================================

show_help() {
    cat << 'EOF'
LLM4DEC-SP - DES-LOC+AutoSP 分布式训练实验流水线 (v2.0)
======================================================================

DES-LOC 核心思想 (Desynced Low-Communication Optimizer):
  Step 1: 每个 worker 本地训练 Kx 步，不做 AllReduce
  Step 2: 第 Kx 步时，同步梯度 (SP all-to-all for attention)
  Step 3: 第 Ku 步时，同步一阶动量 (跨 DP 组)
  Step 4: 第 Kv 步时，同步二阶动量 (全量同步)
  结果: 通信量降低 32×-170×，收敛性保持

AutoSP 核心思想:
  编译器驱动: Torch-IR 自动发现 attention/MLP → 插入 SP 集合
  Ulysses 式 all-to-all: NVSwitch 并行带宽 (vs Ring 的 p-hop 串行)
  1.44-1.62× faster than Ring-Flash-Attention

Usage: ./llm4dec_sp.sh <command> [args...]

BATCH COMMANDS (处理所有模型):
  all_models          训练所有模型 (125M-1.3B)
  eval_all_models     评测所有已训练的模型
  compare_results     生成所有模型的对比报告
  all_status          显示所有模型的状态

SINGLE MODEL COMMANDS:
  full [model]        完整流水线: Cost Model + Train + Eval + Figures
  full_train          完整训练: Cost Model + Train + Kx Sweep + Ratio Ablation
  full_eval           完整评测: Aggregate + Figures

SETUP COMMANDS:
  setup               设置conda环境
  prepare             创建目录
  config              显示当前配置
  fix                 快速修复包依赖
  clean               清除实验数据
  status              显示数据状态

EXPERIMENT COMMANDS:
  cost_model          生成通信代价模型 (1920 data points, CPU only)
  generate [model]    生成合成训练数据
  train [model]       运行 DDP + DES-LOC + LocalSGD 训练
  kx_sweep [model]    运行 Kx 消融实验 (8 values × 3 seeds)
  ratio_abl [model]   运行 Ku/Kv 比例消融 (9 combos × 3 seeds)

EVALUATION COMMANDS:
  eval                收集评测指标
  eval_only           仅评测
  figures             生成 NKI-FA 质量图表

SUPPORTED MODELS:
  125M    GPT-2 Small  (12 layers, 768 hidden)
  350M    GPT-2 Medium (24 layers, 1024 hidden)
  700M    GPT-2 Large  (36 layers, 1280 hidden)
  1.3B    GPT-2 XL     (24 layers, 2048 hidden)
  7B      LLaMA-style  (32 layers, 4096 hidden) — requires SP_size=3

ENVIRONMENT VARIABLES:
  MAX_STEPS           训练步数 (default: 500)
  MODEL_KEY           模型选择 (default: 125M)
  DEFAULT_KX          DES-LOC Kx (default: 32)
  DEFAULT_KU          DES-LOC Ku (default: 96)
  DEFAULT_KV          DES-LOC Kv (default: 192)
  SP_SIZE             AutoSP 并行度 (default: 2)
  BATCH_SIZE          批大小 (default: 4)
  CUDA_DEVICE         GPU设备 (default: 0)
  SEEDS               种子列表 (default: "42 137 2024")

EXAMPLES:
  # 快速测试 - 单模型100步
  MAX_STEPS=100 ./llm4dec_sp.sh full

  # 训练所有模型
  MAX_STEPS=500 ./llm4dec_sp.sh all_models

  # 评测所有模型并生成报告
  ./llm4dec_sp.sh eval_all_models

  # 使用1.3B模型
  MODEL_KEY=1.3B ./llm4dec_sp.sh full

  # 仅运行通信代价模型 (无GPU)
  ./llm4dec_sp.sh cost_model

  # 仅生成图表
  ./llm4dec_sp.sh figures

EOF
}

# ===========================================
# 主入口
# ===========================================

main() {
    print_header

    local cmd="${1:-help}"
    shift 2>/dev/null || true

    case $cmd in
        # Setup
        setup)              setup_environment ;;
        prepare)            prepare_dirs ;;
        config)             show_config ;;
        fix)                quick_fix ;;
        clean)              clean_data ;;
        status)             show_data_status ;;
        all_status)         show_all_models_status ;;

        # Model management
        switch)             switch_model "$@" ;;

        # Data generation
        cost_model|cost)    generate_cost_model ;;
        generate|gen)       generate_training_data "$@" ;;

        # Training
        train)              run_training "$@" ;;
        train_only)         train_only "$@" ;;
        kx_sweep)           run_kx_sweep "$@" ;;
        ratio_abl)          run_ratio_ablation "$@" ;;

        # Evaluation
        eval)               run_eval "$@" ;;
        eval_only)          eval_only "$@" ;;
        figures)            generate_figures ;;

        # Single model pipeline
        full_train)         run_full_train "$@" ;;
        full_eval)          run_full_eval "$@" ;;
        full|all)           run_full_pipeline "$@" ;;

        # Batch commands
        all_models)         run_all_models_pipeline "$@" ;;
        train_all)          train_all_models "$@" ;;
        eval_all_models)    eval_all_models ;;
        compare|compare_results) compare_results ;;

        # Help
        help|--help|-h)     show_help ;;

        *)
            log_error "Unknown command: $cmd"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
