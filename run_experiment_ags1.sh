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
#   bash run_experiment_ags1.sh
# ===========================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook)"
conda activate walking3
set -u

# M340: NCCL settings for heterogeneous GPU cluster
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_DEBUG=WARN
# M363: PCI_BUS_ID → nvidia-smi index = torch cuda index
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# M363: NCCL flight recorder for crash diagnostics
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
    sleep 2
}

# ===================================================================
# Phase 7b: ALREADY COMPLETED (2-seed, 200 steps) — skip
# ===================================================================
# echo "===== Phase 7b: 7B long context seq=2048 (2-seed) ====="
# for S in 1 2; do
#     PYTHONHASHSEED=$((S * 7)) run_exp "p7b_sp_desloc_7B_seq2048_s${S}" "7B" 32 "DESLOC" 200 1 8 "--use_autosp --use_ac --max_seq_len 2048"
# done

# ===================================================================
# Phase 7c: 7B Kx ablation — Figure 14 (1536 steps, compute-optimal)
#
# M363: NO --zero_stage 1. With ZeRO-1, gradient AllReduce happens
# EVERY step (ZeRO-1 requirement), so DES-LOC Kx gating provides
# NO communication reduction — defeating the entire purpose.
# Without ZeRO-1: DESLOCAdamW baseline path → true Kx gating.
# DESLOCAdamW auto-offloads optimizer to CPU when 7B > GPU memory.
#
# Figure 14a (high freq): Kx=16, Kx=32
# Figure 14b (low freq):  Kx=64, Kx=128
# ===================================================================
# echo ""
# echo "===== Phase 7c: 7B Kx ablation — Figure 14 (1536 steps) ====="
# for KX in 16 32 64 128; do
#     PYTHONHASHSEED=7 run_exp "p7c_desloc_7B_Kx${KX}" "7B" "$KX" "DESLOC" 1536 4 4 "--cpu_offload --use_ac"
# done
# # DDP baseline (Kx=1) — AC required to fit 7B on A6000
# PYTHONHASHSEED=7 run_exp "p7c_ddp_7B" "7B" 1 "DDP" 1536 1 8 "--use_ac"
# sleep 5

# # ===================================================================
# # Phase 7d: 7B + AC push (bs=8, higher GPU util)
# # M363: removed --zero_stage 1 — same reason as Phase 7c
# # ===================================================================
# echo ""
# echo "===== Phase 7d: 7B + AC push (bs=8, 1-seed) ====="
# PYTHONHASHSEED=7 run_exp "p7d_desloc_7B_ac_bs8" "7B" 32 "DESLOC" 200 8 4 "--cpu_offload --use_ac"
# sleep 5

# # ===================================================================
# # Phase 7e: 7B Nesterov vs Avg (1536 steps)
# # ===================================================================
# echo ""
# echo "===== Phase 7e: 7B Nesterov vs Avg 1536 steps ====="
# PYTHONHASHSEED=7 run_exp "p7e_nesterov_7B" "7B" 32 "DESLOC" 1536 4 4 "--cpu_offload --use_ac --outer_optimizer nesterov --outer_momentum 0.9"
# PYTHONHASHSEED=7 run_exp "p7e_avg_7B" "7B" 32 "DESLOC" 1536 4 4 "--cpu_offload --use_ac"


echo ""
echo "===== Phase 7c: 7B Kx ablation — Figure 14 (1536 steps) ====="
for KX in 16 32 64 128; do
    # 无SP: bs=4 seq=1024
    # PYTHONHASHSEED=7 run_exp "p7c_desloc_7B_Kx${KX}" "7B" "$KX" "DESLOC" 1536 4 4 "--cpu_offload --use_ac"

    sleep 5

    # SP: bs=4 seq=1024 (same config, SP saves memory)
    PYTHONHASHSEED=7 run_exp "p7c_sp_desloc_7B_Kx${KX}" "7B" "$KX" "DESLOC" 1536 4 4 "--cpu_offload --use_ac --use_autosp"

    sleep 5

    # SP+大batch: SP省出的内存用来推bs=8
    PYTHONHASHSEED=7 run_exp "p7c_sp_desloc_7B_Kx${KX}_bs8" "7B" "$KX" "DESLOC" 1536 8 4 "--cpu_offload --use_ac --use_autosp"
    
    sleep 5
    # SP+长seq: SP省出的内存用来推seq=2048
    PYTHONHASHSEED=7 run_exp "p7c_sp_desloc_7B_Kx${KX}_seq2048" "7B" "$KX" "DESLOC" 1536 4 4 "--cpu_offload --use_ac --use_autosp --max_seq_len 2048"
done
# DDP baselines
PYTHONHASHSEED=7 run_exp "p7c_ddp_7B" "7B" 1 "DDP" 1536 1 8 "--use_ac"

sleep 5


PYTHONHASHSEED=7 run_exp "p7c_sp_ddp_7B" "7B" 1 "DDP" 1536 1 8 "--use_ac --use_autosp"

sleep 5


PYTHONHASHSEED=7 run_exp "p7c_sp_ddp_7B_seq2048" "7B" 1 "DDP" 1536 1 8 "--use_ac --use_autosp --max_seq_len 2048"
sleep 5

echo ""
echo "===== Phase 7d: 7B + AC push (1-seed) ====="
PYTHONHASHSEED=7 run_exp "p7d_desloc_7B_ac_bs8" "7B" 32 "DESLOC" 200 8 4 "--cpu_offload --use_ac"

sleep 5


PYTHONHASHSEED=7 run_exp "p7d_sp_desloc_7B_ac_bs8" "7B" 32 "DESLOC" 200 8 4 "--cpu_offload --use_ac --use_autosp"

sleep 5


PYTHONHASHSEED=7 run_exp "p7d_sp_desloc_7B_ac_bs16" "7B" 32 "DESLOC" 200 16 4 "--cpu_offload --use_ac --use_autosp"
sleep 5

echo ""
echo "===== Phase 7e: 7B Nesterov vs Avg 1536 steps ====="
PYTHONHASHSEED=7 run_exp "p7e_nesterov_7B" "7B" 32 "DESLOC" 1536 4 4 "--cpu_offload --use_ac --outer_optimizer nesterov --outer_momentum 0.9"
sleep 5
PYTHONHASHSEED=7 run_exp "p7e_sp_nesterov_7B" "7B" 32 "DESLOC" 1536 4 4 "--cpu_offload --use_ac --use_autosp --outer_optimizer nesterov --outer_momentum 0.9"
sleep 5
PYTHONHASHSEED=7 run_exp "p7e_sp_nesterov_7B_bs8" "7B" 32 "DESLOC" 1536 8 4 "--cpu_offload --use_ac --use_autosp --outer_optimizer nesterov --outer_momentum 0.9"
sleep 5
PYTHONHASHSEED=7 run_exp "p7e_avg_7B" "7B" 32 "DESLOC" 1536 4 4 "--cpu_offload --use_ac"
sleep 5
PYTHONHASHSEED=7 run_exp "p7e_sp_avg_7B" "7B" 32 "DESLOC" 1536 4 4 "--cpu_offload --use_ac --use_autosp"
sleep 5
PYTHONHASHSEED=7 run_exp "p7e_sp_avg_7B_bs8" "7B" 32 "DESLOC" 1536 8 4 "--cpu_offload --use_ac --use_autosp"
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

for model in ['125M', '700M', '1.3B', '7B']:
    dk = f'DDP_{model}_Kx1'
    lk = f'DESLOC_{model}_Kx32'
    if dk in results and lk in results:
        ddp_mfu = statistics.mean([r['mfu'] for r in results[dk]])
        des_mfu = statistics.mean([r['mfu'] for r in results[lk]])
        print(f'\n{model} MFU speedup: DESLOC/DDP = {des_mfu/ddp_mfu:.2f}x')
PYEOF