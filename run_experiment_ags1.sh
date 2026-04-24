#!/usr/bin/env bash
# ===========================================================================
# DES-LOC 实验 — 2×A6000 + 1×H100 NVL 异构 3-GPU
# 硬件: ags1 服务器
#   GPU 0: NVIDIA RTX A6000  49GB  (BF16 ~38.7 TFLOPS)
#   GPU 1: NVIDIA RTX A6000  49GB  (BF16 ~38.7 TFLOPS)
#   GPU 2: NVIDIA H100 NVL   96GB  (BF16 ~835 TFLOPS)
#
# 用法:
#   cd /data/jiacheng/system/cache/temp/nips2026/Neuron_SP
#   git apply claude29_bugfix.patch   # 先打补丁
#   bash run_experiment_ags1.sh       # 开始实验
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# M340: NCCL settings for heterogeneous GPU cluster
# The 600s default timeout is too short when GradScaler causes
# step skips that desync AllReduce participation across ranks.
# With the global_step fix (M340), deadlocks should not occur,
# but we increase timeout as a safety net.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1   # replaces deprecated NCCL_ASYNC_ERROR_HANDLING
export NCCL_TIMEOUT=1800000                # 30 min timeout (was 600s default)
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800  # match
export NCCL_DEBUG=WARN                     # show NCCL warnings (INFO is too verbose)

RESULTS_DIR="./desloc_results"
mkdir -p "$RESULTS_DIR"

NGPU=3
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo " DES-LOC 实验 — 2×A6000 + 1×H100 NVL"
echo " 启动: $(date)"
echo " NGPU: $NGPU"
echo "================================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "================================================================"

# ===================================================================
# 端口管理: 每个实验递增, 避免残留端口冲突
# ===================================================================
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

    # Ku = 3×Kx, Kv = 6×Kx (论文默认)
    local KU=$((KX * 3))
    local KV=$((KX * 6))
    # Kx=1 时 Ku=Kv=1 (DDP baseline)
    if [ "$KX" -eq 1 ]; then KU=1; KV=1; fi

    echo ""
    echo ">>> [$(date +%H:%M:%S)] $NAME | model=$MODEL Kx=$KX Ku=$KU Kv=$KV methods=$METHODS steps=$STEPS batch=$BATCH×$GRAD_ACCUM $EXTRA"

    CUDA_VISIBLE_DEVICES=0,1,2 torchrun \
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
    sleep 2  # 让端口释放
}

# ===================================================================
# Phase 0: 快速冒烟测试 (2分钟)
# ===================================================================
echo ""
echo "===== Phase 0: 冒烟测试 (10 steps) ====="
run_exp "smoke_ddp" "125M" 1 "DDP" 10 4 1
run_exp "smoke_desloc" "125M" 32 "DESLOC" 10 4 1

# ===================================================================
# Phase 1: 125M 基线 (5-seed, ~20分钟)
# 修改: 每个seed交替跑DDP+DESLOC, 立即得到对比
# ===================================================================
echo ""
echo "===== Phase 1: 125M DDP vs DESLOC (5-seed, interleaved) ====="

for S in 1 2 3 4 5; do
    PYTHONHASHSEED=$((S * 7)) run_exp "p1_ddp_125m_s${S}" "125M" 1 "DDP" 500 4 4
    PYTHONHASHSEED=$((S * 7)) run_exp "p1_desloc_125m_Kx32_s${S}" "125M" 32 "DESLOC" 500 4 4
done

# ===================================================================
# Phase 2: 700M 核心实验 (5-seed, ~120分钟)
# 注意: A6000 49GB 跑 700M 需要 grad_accum>=4
# 修改: 每个seed先跑DDP再跑DESLOC (交替), 而非全部DDP先跑完
#       这样seed1的对比结果24分钟就出来, 不用等70分钟
# ===================================================================
echo ""
echo "===== Phase 2: 700M DDP vs DESLOC (5-seed, interleaved) ====="

for S in 1 2 3 4 5; do
    PYTHONHASHSEED=$((S * 7)) run_exp "p2_ddp_700m_s${S}" "700M" 1 "DDP" 500 4 4
    PYTHONHASHSEED=$((S * 7)) run_exp "p2_desloc_700m_Kx32_s${S}" "700M" 32 "DESLOC" 500 4 4
done

# ===================================================================
# Phase 3: Kx 消融 (700M, 3-seed, ~40分钟)
# ===================================================================
echo ""
echo "===== Phase 3: Kx 消融 (700M) ====="

for KX in 8 16 64 128; do
    for S in 1 2 3; do
        PYTHONHASHSEED=$((S * 7)) run_exp "p3_desloc_700m_Kx${KX}_s${S}" "700M" "$KX" "DESLOC" 500 4 4
    done
done

# ===================================================================
# Phase 4: Nesterov outer optimizer (700M, 3-seed, ~20分钟)
# ===================================================================
echo ""
echo "===== Phase 4: Nesterov vs Average ====="

for S in 1 2 3; do
    PYTHONHASHSEED=$((S * 7)) run_exp "p4_nesterov_700m_s${S}" "700M" 32 "DESLOC" 500 4 4 \
        "--outer_optimizer nesterov --outer_momentum 0.9"
done

# ===================================================================
# Phase 5: SP+DEC (Sequence Parallel + Desynced Communication)
# 125M only — SP halves per-GPU seq_len, enabling longer contexts
# or doubling effective batch size per memory unit.
# 注意: SP+DEC正交 — SP沿seq维度切分, DEC沿worker维度控制通信频率
# ===================================================================
echo ""
echo "===== Phase 5: SP+DEC (125M, Kx=32, 3-seed) ====="

for S in 1 2 3; do
    PYTHONHASHSEED=$((S * 7)) run_exp "p5_sp_desloc_125m_s${S}" "125M" 32 "DESLOC" 500 4 4 \
        "--use_autosp"
done

# Phase 5b: SP+DDP baseline for comparison
for S in 1 2 3; do
    PYTHONHASHSEED=$((S * 7)) run_exp "p5_sp_ddp_125m_s${S}" "125M" 1 "DDP" 500 4 4 \
        "--use_autosp"
done

# ===================================================================
# Phase 6: 1.3B — 异构实验 (3-seed, ~180分钟)
# Memory: 1.3B params = ~15GB → fits A6000 49GB easily
# 使用 activation checkpointing 以留更多空间给 batch
# ===================================================================
echo ""
echo "===== Phase 6: 1.3B DDP vs DESLOC (3-seed, interleaved, use_ac) ====="

for S in 1 2 3; do
    PYTHONHASHSEED=$((S * 7)) run_exp "p6_ddp_1.3B_s${S}" "1.3B" 1 "DDP" 500 2 8 "--use_ac"
    PYTHONHASHSEED=$((S * 7)) run_exp "p6_desloc_1.3B_Kx32_s${S}" "1.3B" 32 "DESLOC" 500 2 8 "--use_ac"
done

# Phase 6b: 1.3B Kx消融
echo "===== Phase 6b: 1.3B Kx 消融 ====="
for KX in 16 64; do
    PYTHONHASHSEED=7 run_exp "p6b_desloc_1.3B_Kx${KX}" "1.3B" "$KX" "DESLOC" 500 2 8 "--use_ac"
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
for f in sorted(glob.glob('desloc_results/*.json')):
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
        if loss is None: continue
        key = f'{method}_{model}_Kx{kx}'
        results.setdefault(key, []).append({'loss': loss, 'mfu': mfu})

print(f"\n{'Key':<32} {'N':>3} {'Loss':>14} {'MFU':>10}")
print('-' * 64)
for key in sorted(results.keys()):
    runs = results[key]
    losses = [r['loss'] for r in runs]
    mfus = [r['mfu'] for r in runs]
    ml = statistics.mean(losses)
    sl = statistics.stdev(losses) if len(losses) > 1 else 0
    mm = statistics.mean(mfus)
    print(f'{key:<32} {len(runs):>3} {ml:>7.2f}±{sl:<5.2f} {mm:>9.4f}')

# DDP vs DESLOC speedup
for model in ['125M', '700M', '1.3B']:
    dk = f'DDP_{model}_Kx1'
    lk = f'DESLOC_{model}_Kx32'
    if dk in results and lk in results:
        ddp_mfu = statistics.mean([r['mfu'] for r in results[dk]])
        des_mfu = statistics.mean([r['mfu'] for r in results[lk]])
        print(f'\n{model} MFU speedup: DESLOC/DDP = {des_mfu/ddp_mfu:.2f}x')
PYEOF
