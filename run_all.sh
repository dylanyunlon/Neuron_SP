#!/bin/bash
# =============================================================================
# DES-LOC 完整实验矩阵 — 2x A100 PCIe 80G
# 42 experiments (RQ2 + RQ3) × 500 steps, 3 seeds each
# 预计用时: ~35-40 min
# =============================================================================
set -uo pipefail

SECONDS=0
TS=$(date +%Y%m%d_%H%M%S)
export RESULTS_DIR="/workspace/desloc_results_${TS}"
mkdir -p "$RESULTS_DIR"

echo "================================================================"
echo " DES-LOC Full Experiment Matrix"
echo " Started: $(date)"
echo " Results: $RESULTS_DIR"
echo "================================================================"

# ── Phase 0: 环境 ──────────────────────────────────────────
cd /workspace 2>/dev/null || cd ~

if [ ! -d "Neuron_SP" ]; then
    echo "[SETUP] Cloning repo..."
    git clone --depth 1 https://github.com/dylanyunlon/Neuron_SP.git
fi
cd Neuron_SP

# 安装依赖（静默）
pip install -q deepspeed 2>/dev/null || pip install deepspeed --break-system-packages -q 2>/dev/null || true

NGPU=$(python3 -c "import torch; print(torch.cuda.device_count())")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "[INFO] ${NGPU}x ${GPU_NAME}"

STEPS=500
LOG="$RESULTS_DIR/experiment_log.csv"
echo "run_id,tag,model,Kx,Ku,Kv,seed,method,exit_code,elapsed_s" > "$LOG"
RUN_ID=0
FAIL=0

# ── 运行函数 ───────────────────────────────────────────────
run_exp() {
    local TAG=$1 MODEL=$2 KX=$3 KU=$4 KV=$5 SEED=$6 METHOD=$7

    RUN_ID=$((RUN_ID + 1))
    local LOGFILE="$RESULTS_DIR/${TAG}_${MODEL}_Kx${KX}_${METHOD}_s${SEED}.log"
    local T0=$SECONDS

    printf "[%2d/42] %-12s %-5s Kx=%-3d %-10s seed=%-4d ... " \
        "$RUN_ID" "$TAG" "$MODEL" "$KX" "$METHOD" "$SEED"

    # seed 控制
    export PYTHONHASHSEED=$SEED

    torchrun --nproc_per_node=$NGPU \
        --master_port=$((29500 + RUN_ID % 100)) \
        REAL_GPU_BENCHMARK.py \
        --model_size "$MODEL" \
        --batch_size 4 \
        --grad_accum 8 \
        --max_steps $STEPS \
        --Kx "$KX" --Ku "$KU" --Kv "$KV" \
        --methods "$METHOD" \
        --output "$RESULTS_DIR" \
        > "$LOGFILE" 2>&1

    local RC=$?
    local DT=$((SECONDS - T0))

    if [ $RC -eq 0 ]; then
        echo "OK (${DT}s)"
    else
        echo "FAIL:${RC} (${DT}s)"
        FAIL=$((FAIL + 1))
    fi

    echo "${RUN_ID},${TAG},${MODEL},${KX},${KU},${KV},${SEED},${METHOD},${RC},${DT}" >> "$LOG"
}

# ══════════════════════════════════════════════════════════════
# RQ2: Sync Frequency Sweep — Kx ∈ {1,2,4,8,16,32,64,128}
# 8 Kx values × 3 seeds = 24 runs
# ══════════════════════════════════════════════════════════════
echo ""
echo "━━━ RQ2: Sync Frequency Sweep (Section 5.2) ━━━"
for KX in 1 2 4 8 16 32 64 128; do
    KU=$((KX > 0 ? KX * 3 : 3))
    KV=$((KX > 0 ? KX * 6 : 6))
    # Kx=1 时 Ku/Kv也=1 表示DDP等价
    if [ $KX -eq 1 ]; then KU=1; KV=1; fi
    for SEED in 42 137 2024; do
        run_exp "rq2_sweep" "125M" $KX $KU $KV $SEED "DESLOC"
    done
done

# ══════════════════════════════════════════════════════════════
# RQ3: Communication Reduction — DDP vs LocalAdam vs DES-LOC
# 2 models × 3 methods × 3 seeds = 18 runs
# ══════════════════════════════════════════════════════════════
echo ""
echo "━━━ RQ3: Communication Reduction (Section 5.3) ━━━"
for MODEL in 125M 350M; do
    for SEED in 42 137 2024; do
        run_exp "rq3_comm" "$MODEL" 1   1   1   $SEED "DDP"
        run_exp "rq3_comm" "$MODEL" 32  32  32  $SEED "LocalAdam"
        run_exp "rq3_comm" "$MODEL" 32  96  192 $SEED "DESLOC"
    done
done

# ══════════════════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo " COMPLETED — $(date)"
echo " Total: ${RUN_ID} experiments in $((SECONDS/60))m${SECONDS##*[0-9]}s"
echo " Success: $((RUN_ID - FAIL)) / ${RUN_ID}"
echo " Failed: ${FAIL}"
echo " Results: $RESULTS_DIR"
echo "================================================================"

# 汇总所有JSON结果
echo ""
echo "[MERGE] Collecting results..."
python3 << 'PYEOF'
import json, glob, os, sys

results_dir = os.environ.get('RESULTS_DIR', '.')
files = sorted(glob.glob(f'{results_dir}/benchmark_results_*.json'))

if not files:
    # 尝试从log文件提取
    files = sorted(glob.glob(f'{results_dir}/*.log'))

all_data = []
for f in files:
    try:
        with open(f) as fh:
            data = json.load(fh)
            all_data.append(data)
    except (json.JSONDecodeError, Exception):
        continue

merged = {
    'total_experiments': len(all_data),
    'experiments': all_data
}

out = f'{results_dir}/ALL_RESULTS.json'
with open(out, 'w') as fh:
    json.dump(merged, fh, indent=2)

print(f"  Merged {len(all_data)} results → {out}")
PYEOF

# 打包
tar czf "${RESULTS_DIR}.tar.gz" -C "$(dirname $RESULTS_DIR)" "$(basename $RESULTS_DIR)"
echo "[DONE] Packed: ${RESULTS_DIR}.tar.gz ($(du -h ${RESULTS_DIR}.tar.gz | cut -f1))"
echo ""
echo "下载结果:"
echo "  scp -P <PORT> root@<HOST>:${RESULTS_DIR}.tar.gz ."
