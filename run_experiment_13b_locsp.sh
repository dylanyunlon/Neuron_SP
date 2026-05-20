#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook)"
conda activate walking3
set -u

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_DEBUG=WARN
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DESLOC_SP_A2A_TIMEOUT_MS=120000

RESULTS_DIR="./desloc_results"
mkdir -p "$RESULTS_DIR"

NGPU=3
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs_13b_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo " LOC+SP 13B — 2×A6000 + 1×H100 NVL"
echo " $(date)"
echo " NGPU: $NGPU"
echo "================================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "================================================================"

PORT=29600

run_exp() {
    local NAME="$1"
    local MODEL="$2"
    local KX="$3"
    local METHODS="$4"
    local STEPS="$5"
    local BATCH="${6:-2}"
    local GRAD_ACCUM="${7:-8}"
    local EXTRA="${8:-}"

    local KU=$((KX * 3))
    local KV=$((KX * 6))
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
    sleep 5
}

echo ""
echo "===== Phase 13a: 13B DDP baseline ====="
PYTHONHASHSEED=13 run_exp "p13a_ddp_13B" "13B" 1 "DDP" 200 1 8 "--cpu_offload --use_ac"
sleep 5

PYTHONHASHSEED=13 run_exp "p13a_sp_ddp_13B" "13B" 1 "DDP" 200 1 8 "--cpu_offload --use_ac --use_autosp"
sleep 5

echo ""
echo "===== Phase 13b: 13B Kx ablation (200 steps) ====="
for KX in 16 32 64; do
    PYTHONHASHSEED=13 run_exp "p13b_sp_desloc_13B_Kx${KX}" "13B" "$KX" "DESLOC" 200 2 8 "--cpu_offload --use_ac --use_autosp"
    sleep 5

    PYTHONHASHSEED=13 run_exp "p13b_sp_desloc_13B_Kx${KX}_bs4" "13B" "$KX" "DESLOC" 200 4 4 "--cpu_offload --use_ac --use_autosp"
    sleep 5
done

echo ""
echo "===== Phase 13c: 13B extended (1536 steps, Kx=32) ====="
PYTHONHASHSEED=13 run_exp "p13c_sp_desloc_13B_Kx32_long" "13B" 32 "DESLOC" 1536 2 8 "--cpu_offload --use_ac --use_autosp"
sleep 5

PYTHONHASHSEED=13 run_exp "p13c_sp_desloc_13B_Kx32_seq2048" "13B" 32 "DESLOC" 200 2 8 "--cpu_offload --use_ac --use_autosp --max_seq_len 2048"
sleep 5

echo ""
echo "===== Phase 13d: 13B Nesterov vs Avg ====="
PYTHONHASHSEED=13 run_exp "p13d_sp_nesterov_13B" "13B" 32 "DESLOC" 200 2 8 "--cpu_offload --use_ac --use_autosp --outer_optimizer nesterov --outer_momentum 0.9"
sleep 5
PYTHONHASHSEED=13 run_exp "p13d_sp_avg_13B" "13B" 32 "DESLOC" 200 2 8 "--cpu_offload --use_ac --use_autosp"

echo ""
echo "================================================================"
echo " 13B LOC+SP done — $(date)"
echo " JSON: $(ls -1 $RESULTS_DIR/*.json 2>/dev/null | wc -l)"
echo "================================================================"

python3 << 'PYEOF'
import json, glob, statistics

results = {}
for f in sorted(glob.glob('desloc_results/*.json')):
    d = json.load(open(f))
    cfg = d.get('config', {})
    model = cfg.get('model_size', '?')
    if model != '13B':
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
        if loss is None:
            continue
        key = f'{method}_13B_Kx{kx}'
        results.setdefault(key, []).append({'loss': loss, 'mfu': mfu})

print(f"\n{'Key':<36} {'N':>3} {'Loss':>14} {'MFU':>10}")
print('-' * 68)
for key in sorted(results.keys()):
    runs = results[key]
    losses = [r['loss'] for r in runs]
    mfus = [r['mfu'] for r in runs]
    ml = statistics.mean(losses)
    sl = statistics.stdev(losses) if len(losses) > 1 else 0
    mm = statistics.mean(mfus)
    print(f'{key:<36} {len(runs):>3} {ml:>7.2f}+/-{sl:<5.2f} {mm:>9.4f}')
PYEOF
