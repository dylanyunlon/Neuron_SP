#!/usr/bin/env bash
# ===========================================================================
# DES-LOC 实验 — 2×H20 同构集群 (阿里云 gn8v_2x_8xlarge)
# 硬件:
#   GPU 0: NVIDIA H20  96GB HBM3  (BF16 ~148 TFLOPS, 4TB/s BW, NVLink 900GB/s)
#   GPU 1: NVIDIA H20  96GB HBM3  (同上)
#
# 优势 vs ags1 (2×A6000 + H100 NVL):
#   - 同构 → 无GradScaler死锁 (M340 issue)
#   - 96GB/GPU → 1.3B模型轻松
#   - NVLink 900GB/s → AllReduce极快
#
# 用法:
#   git clone https://github.com/dylanyunlon/Neuron_SP.git
#   cd Neuron_SP
#   # 如需打patch: git apply claude29_bugfix.patch
#   # 覆盖最新REAL_GPU_BENCHMARK.py (含M339-M341修复)
#   bash run_experiment_h20.sh
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# NCCL settings for H20 NVLink cluster
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_DEBUG=WARN

RESULTS_DIR="./desloc_results_h20"
mkdir -p "$RESULTS_DIR"

NGPU=2
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs_h20_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo " DES-LOC SP+DEC+AC 实验 — 2×H20 同构集群"
echo " 启动: $(date)"
echo " NGPU: $NGPU"
echo "================================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "================================================================"

PORT=29500

run_exp() {
    local NAME="$1"
    local MODEL="$2"
    local KX="$3"
    local METHODS="$4"
    local STEPS="$5"
    local BATCH="${6:-4}"
    local GRAD_ACCUM="${7:-4}"
    local EXTRA="${8:-}"

    local KU=$((KX * 3))
    local KV=$((KX * 6))
    if [ "$KX" -eq 1 ]; then KU=1; KV=1; fi

    echo ""
    echo ">>> [$(date +%H:%M:%S)] $NAME | model=$MODEL Kx=$KX Ku=$KU Kv=$KV methods=$METHODS steps=$STEPS batch=$BATCH×$GRAD_ACCUM $EXTRA"

    CUDA_VISIBLE_DEVICES=0,1 torchrun \
        --nproc_per_node=$NGPU \
        --master_addr=127.0.0.1 \
        --master_port=$PORT \
        REAL_GPU_BENCHMARK.py \
        --model_size "$MODEL" \
        --batch_size "$BATCH" \
        --grad_accum "$GRAD_ACCUM" \
        --max_steps "$STEPS" \
        --Kx "$KX" \
        --Ku "$KU" \
        --Kv "$KV" \
        --methods $METHODS \
        --output "$RESULTS_DIR" \
        $EXTRA \
        2>&1 | tee "$LOG_DIR/${NAME}.log"

    local EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -ne 0 ]; then
        echo "!!! [$(date +%H:%M:%S)] $NAME FAILED (exit=$EXIT_CODE)"
    else
        echo "<<< [$(date +%H:%M:%S)] $NAME OK"
    fi
    PORT=$((PORT + 1))
    sleep 2
}

# ===================================================================
# Phase 0: 冒烟测试 (~1分钟)
# ===================================================================
echo ""
echo "===== Phase 0: 冒烟测试 (10 steps) ====="
run_exp "smoke_ddp" "125M" 1 "DDP" 10 4 1
run_exp "smoke_desloc" "125M" 32 "DESLOC" 10 4 1

# ===================================================================
# Phase 1: 125M DDP vs DESLOC (5-seed, 交替, ~10分钟)
# ===================================================================
echo ""
echo "===== Phase 1: 125M baseline (5-seed, interleaved) ====="
for S in 1 2 3 4 5; do
    PYTHONHASHSEED=$((S * 7)) run_exp "p1_ddp_125m_s${S}" "125M" 1 "DDP" 500 4 4
    PYTHONHASHSEED=$((S * 7)) run_exp "p1_desloc_125m_Kx32_s${S}" "125M" 32 "DESLOC" 500 4 4
done

# ===================================================================
# Phase 2: 700M DDP vs DESLOC (5-seed, 交替, ~50分钟)
# H20 96GB轻松跑700M, batch可以加大
# ===================================================================
echo ""
echo "===== Phase 2: 700M baseline (5-seed, interleaved) ====="
for S in 1 2 3 4 5; do
    PYTHONHASHSEED=$((S * 7)) run_exp "p2_ddp_700m_s${S}" "700M" 1 "DDP" 500 4 4
    PYTHONHASHSEED=$((S * 7)) run_exp "p2_desloc_700m_Kx32_s${S}" "700M" 32 "DESLOC" 500 4 4
done

# ===================================================================
# Phase 3: Kx 消融 (700M, 3-seed, ~24分钟)
# ===================================================================
echo ""
echo "===== Phase 3: Kx ablation (700M) ====="
for KX in 8 16 64 128; do
    for S in 1 2 3; do
        PYTHONHASHSEED=$((S * 7)) run_exp "p3_desloc_700m_Kx${KX}_s${S}" "700M" "$KX" "DESLOC" 500 4 4
    done
done

# ===================================================================
# Phase 4: Nesterov outer optimizer (700M, 3-seed, ~12分钟)
# ===================================================================
echo ""
echo "===== Phase 4: Nesterov vs Average ====="
for S in 1 2 3; do
    PYTHONHASHSEED=$((S * 7)) run_exp "p4_nesterov_700m_s${S}" "700M" 32 "DESLOC" 500 4 4 \
        "--outer_optimizer nesterov --outer_momentum 0.9"
done

# ===================================================================
# Phase 5: SP+DEC (Sequence Parallel + Desynced Communication)
# 125M + 700M, 验证SP与DEC的正交性
# SP沿seq维度切分, DEC沿worker维度控制通信频率
# ===================================================================
echo ""
echo "===== Phase 5: SP+DEC (125M + 700M, 3-seed) ====="

# 5a: 125M SP+DEC vs 非SP DEC
for S in 1 2 3; do
    PYTHONHASHSEED=$((S * 7)) run_exp "p5a_sp_desloc_125m_s${S}" "125M" 32 "DESLOC" 500 4 4 \
        "--use_autosp"
    PYTHONHASHSEED=$((S * 7)) run_exp "p5a_sp_ddp_125m_s${S}" "125M" 1 "DDP" 500 4 4 \
        "--use_autosp"
done

# 5b: 700M SP+DEC (H20 96GB够用)
for S in 1 2 3; do
    PYTHONHASHSEED=$((S * 7)) run_exp "p5b_sp_desloc_700m_s${S}" "700M" 32 "DESLOC" 500 4 4 \
        "--use_autosp"
done

# ===================================================================
# Phase 6: AC+DEC (Activation Checkpointing + Desynced Communication)
# 验证AC与DEC正交: AC省显存, DEC省通信
# ===================================================================
echo ""
echo "===== Phase 6: AC+DEC (700M, 3-seed) ====="

for S in 1 2 3; do
    # AC+DEC: DES-LOC with activation checkpointing
    PYTHONHASHSEED=$((S * 7)) run_exp "p6_ac_desloc_700m_s${S}" "700M" 32 "DESLOC" 500 4 4 \
        "--use_ac"
    # AC+DDP baseline
    PYTHONHASHSEED=$((S * 7)) run_exp "p6_ac_ddp_700m_s${S}" "700M" 1 "DDP" 500 4 4 \
        "--use_ac"
done

# ===================================================================
# Phase 7: SP+DEC+AC 三维正交 (700M, 3-seed)
# 全部组合: sequence parallel + desynced comm + activation ckpt
# ===================================================================
echo ""
echo "===== Phase 7: SP+DEC+AC full combo (700M, 3-seed) ====="

for S in 1 2 3; do
    PYTHONHASHSEED=$((S * 7)) run_exp "p7_sp_ac_desloc_700m_s${S}" "700M" 32 "DESLOC" 500 4 4 \
        "--use_autosp --use_ac"
done

# ===================================================================
# Phase 8: 1.3B 大模型验证 (AC必须开, 3-seed)
# H20 96GB: 1.3B + AC → ~50GB/GPU, 够用
# ===================================================================
echo ""
echo "===== Phase 8: 1.3B scale-up (3-seed) ====="

for S in 1 2 3; do
    PYTHONHASHSEED=$((S * 7)) run_exp "p8_ddp_1.3b_s${S}" "1.3B" 1 "DDP" 300 2 8 "--use_ac"
    PYTHONHASHSEED=$((S * 7)) run_exp "p8_desloc_1.3b_s${S}" "1.3B" 32 "DESLOC" 300 2 8 "--use_ac"
done

# ===================================================================
# 汇总
# ===================================================================
echo ""
echo "================================================================"
echo " 全部实验完成 — $(date)"
echo " JSON 数量: $(ls -1 $RESULTS_DIR/*.json 2>/dev/null | wc -l)"
echo "================================================================"

python3 << 'PYEOF'
import json, glob, statistics

results = {}
for f in sorted(glob.glob('desloc_results_h20/*.json')):
    d = json.load(open(f))
    cfg = d.get('config', {})
    model = cfg.get('model_size', '?')
    kx = cfg.get('Kx', 1)
    steps = cfg.get('max_steps', 0)
    if steps < 50: continue
    for method, data in d.get('results', {}).items():
        if not isinstance(data, dict): continue
        loss = data.get('final_loss')
        mfu = data.get('mfu', 0)
        sp = data.get('sp_mode', 'none')
        if loss is None: continue
        key = f'{method}_{model}_Kx{kx}'
        if sp != 'none':
            key += f'_SP'
        results.setdefault(key, []).append({'loss': loss, 'mfu': mfu})

print(f"\n{'Key':<40} {'N':>3} {'Loss':>14} {'MFU':>10}")
print('-' * 72)
for key in sorted(results.keys()):
    runs = results[key]
    losses = [r['loss'] for r in runs]
    mfus = [r['mfu'] for r in runs]
    ml = statistics.mean(losses)
    sl = statistics.stdev(losses) if len(losses) > 1 else 0
    mm = statistics.mean(mfus)
    print(f'{key:<40} {len(runs):>3} {ml:>7.2f}±{sl:<5.2f} {mm:>9.4f}')

# DDP vs DESLOC speedup
for model in ['125M', '700M', '1.3B']:
    dk = f'DDP_{model}_Kx1'
    lk = f'DESLOC_{model}_Kx32'
    if dk in results and lk in results:
        ddp_mfu = statistics.mean([r['mfu'] for r in results[dk]])
        des_mfu = statistics.mean([r['mfu'] for r in results[lk]])
        if ddp_mfu > 0:
            print(f'\n{model} MFU speedup: DESLOC/DDP = {des_mfu/ddp_mfu:.2f}x')
PYEOF
