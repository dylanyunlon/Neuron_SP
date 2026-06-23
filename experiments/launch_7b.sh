#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
#
# Launch script for 7B pre-training experiments on ags1 cluster.
# Cluster: 2×A6000 + 1×H100-NVL + 2×RTX PRO 6000 Blackwell (5 GPUs total)
#
# Usage:
#   ./experiments/launch_7b.sh                  # Run both DES-LOC + baseline
#   ./experiments/launch_7b.sh desloc           # DES-LOC only
#   ./experiments/launch_7b.sh baseline         # Baseline only
#   ./experiments/launch_7b.sh desloc 200       # DES-LOC, 200 steps (quick test)

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

eval "$(conda shell.bash hook)"
conda activate walking3
set -u

# ─── Environment ─────────────────────────────────────────────────────────────
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_DEBUG=WARN
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DESLOC_SP_A2A_TIMEOUT_MS=120000

# LOC cache: pre-allocate NUMA-local shared memory
export DESLOC_LOC_CACHE_NUMA0=/dev/shm/loc_cache_numa0
export DESLOC_LOC_CACHE_NUMA1=/dev/shm/loc_cache_numa1
export DESLOC_LOC_CACHE_SIZE_GB=200

NGPU=5
RESULTS_DIR="./desloc_results"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs_7b_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

MODE="${1:-all}"
OVERRIDE_STEPS="${2:-}"
PORT=29700

echo "================================================================"
echo " 7B Pre-training — ags1 5-GPU Heterogeneous Cluster"
echo " $(date)"
echo " Mode: $MODE"
echo " NGPU: $NGPU"
echo "================================================================"
nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.current --format=csv,noheader
echo "================================================================"

# ─── Helper ──────────────────────────────────────────────────────────────────
run_exp() {
    local NAME="$1"
    local KX="$2"
    local METHODS="$3"
    local STEPS="$4"
    local EXTRA="${5:-}"

    # Override steps if provided on command line
    if [ -n "$OVERRIDE_STEPS" ]; then
        STEPS="$OVERRIDE_STEPS"
    fi

    # DES-LOC sync periods: Ku=Kx/2, Kv=Kx*2
    local KU=$((KX / 2))
    local KV=$((KX * 2))
    if [ "$KX" -eq 1 ]; then KU=1; KV=1; fi
    # Task specifies Kx=8, Ku=4, Kv=16 — the formula above yields exactly that.

    echo ""
    echo ">>> [$(date +%H:%M:%S)] $NAME | Kx=$KX Ku=$KU Kv=$KV methods=$METHODS steps=$STEPS $EXTRA"

    CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun \
        --nproc_per_node=$NGPU \
        --master_addr=127.0.0.1 \
        --master_port=$PORT \
        REAL_GPU_BENCHMARK.py \
        --model_size "7B" \
        --batch_size 2 \
        --grad_accum 8 \
        --max_steps "$STEPS" \
        --Kx "$KX" \
        --Ku "$KU" \
        --Kv "$KV" \
        --methods $METHODS \
        --output "$RESULTS_DIR" \
        --cpu_offload \
        --use_ac \
        $EXTRA \
        2>&1 | tee "$LOG_DIR/${NAME}.log"

    local EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -ne 0 ]; then
        echo "!!! [$(date +%H:%M:%S)] $NAME FAILED (exit=$EXIT_CODE)"
    else
        echo "<<< [$(date +%H:%M:%S)] $NAME OK"
    fi
    PORT=$((PORT + 1))
    sleep 5
}

# ─── Phase 1: Baseline (DDP, no DES-LOC) ────────────────────────────────────
if [ "$MODE" = "all" ] || [ "$MODE" = "baseline" ]; then
    echo ""
    echo "===== Phase 7a: 7B DDP baseline (Kx=1 → full sync every step) ====="
    PYTHONHASHSEED=42 run_exp "p7a_ddp_7B_baseline" 1 "DDP" 200 "--use_autosp"
    sleep 5

    echo ""
    echo "===== Phase 7b: 7B DDP baseline extended (1000 steps) ====="
    PYTHONHASHSEED=42 run_exp "p7b_ddp_7B_baseline_long" 1 "DDP" 1000 "--use_autosp"
    sleep 5
fi

# ─── Phase 2: DES-LOC (Kx=8, Ku=4, Kv=16) ──────────────────────────────────
if [ "$MODE" = "all" ] || [ "$MODE" = "desloc" ]; then
    echo ""
    echo "===== Phase 7c: 7B DES-LOC Kx=8 (200 steps, quick validation) ====="
    PYTHONHASHSEED=42 run_exp "p7c_desloc_7B_Kx8" 8 "DESLOC" 200 "--use_autosp"
    sleep 5

    echo ""
    echo "===== Phase 7d: 7B DES-LOC Kx=8 extended (1000 steps) ====="
    PYTHONHASHSEED=42 run_exp "p7d_desloc_7B_Kx8_long" 8 "DESLOC" 1000 "--use_autosp"
    sleep 5

    echo ""
    echo "===== Phase 7e: 7B DES-LOC Kx ablation ====="
    for KX in 4 16 32; do
        PYTHONHASHSEED=42 run_exp "p7e_desloc_7B_Kx${KX}" "$KX" "DESLOC" 200 "--use_autosp"
        sleep 5
    done

    echo ""
    echo "===== Phase 7f: 7B DES-LOC Nesterov outer optimizer ====="
    PYTHONHASHSEED=42 run_exp "p7f_desloc_7B_nesterov" 8 "DESLOC" 200 \
        "--use_autosp --outer_optimizer nesterov --outer_momentum 0.9"
    sleep 5

    echo ""
    echo "===== Phase 7g: 7B DES-LOC seq_len=2048 stress test ====="
    PYTHONHASHSEED=42 run_exp "p7g_desloc_7B_seq2048" 8 "DESLOC" 200 \
        "--use_autosp --max_seq_len 2048"
    sleep 5
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " 7B experiments done — $(date)"
echo " Logs: $LOG_DIR"
echo " JSON: $(ls -1 $RESULTS_DIR/*7B*.json 2>/dev/null | wc -l) result files"
echo "================================================================"

python3 << 'PYEOF'
import json, glob, statistics

results = {}
for f in sorted(glob.glob('desloc_results/*.json')):
    d = json.load(open(f))
    cfg = d.get('config', {})
    model = cfg.get('model_size', '?')
    if model != '7B':
        continue
    kx = cfg.get('Kx', 1)
    steps = cfg.get('max_steps', 0)
    if steps < 50:
        continue
    for method, data in d.get('results', {}).items():
        if not isinstance(data, dict):
            continue
        loss = data.get('final_loss')
        mfu = data.get('mfu', 0)
        tps = data.get('tokens_per_sec', 0)
        if loss is None:
            continue
        key = f'{method}_7B_Kx{kx}'
        results.setdefault(key, []).append({'loss': loss, 'mfu': mfu, 'tps': tps})

print(f"\n{'Key':<36} {'N':>3} {'Loss':>14} {'MFU':>10} {'tok/s':>10}")
print('-' * 78)
for key in sorted(results.keys()):
    runs = results[key]
    losses = [r['loss'] for r in runs]
    mfus = [r['mfu'] for r in runs]
    tps_list = [r['tps'] for r in runs]
    ml = statistics.mean(losses)
    sl = statistics.stdev(losses) if len(losses) > 1 else 0
    mm = statistics.mean(mfus)
    mt = statistics.mean(tps_list)
    print(f'{key:<36} {len(runs):>3} {ml:>7.2f}+/-{sl:<5.2f} {mm:>9.4f} {mt:>9.0f}')
PYEOF
