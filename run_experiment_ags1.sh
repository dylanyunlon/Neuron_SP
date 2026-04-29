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
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate the walking3 conda environment
# Note: set -u is disabled because conda activate reads $PS1 which is unset in scripts
eval "$(conda shell.bash hook)"
conda activate walking3
set -u

# M340: NCCL settings for heterogeneous GPU cluster
# The 600s default timeout is too short when GradScaler causes
# step skips that desync AllReduce participation across ranks.
# With the global_step fix (M340), deadlocks should not occur,
# but we increase timeout as a safety net.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1   # replaces deprecated NCCL_ASYNC_ERROR_HANDLING
export NCCL_TIMEOUT=1800000                # 30 min timeout (was 600s default)
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800  # match
export NCCL_DEBUG=WARN                     # show NCCL warnings (INFO is too verbose)
# Reduce CUDA memory fragmentation for tight-memory 7B runs (A6000 48GB)
# Requires PyTorch 2.4+ (walking3 env). Falls back gracefully if unsupported.
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
# echo ""
# echo "===== Phase 0: 冒烟测试 (10 steps) ====="
# run_exp "smoke_ddp" "125M" 1 "DDP" 10 4 1
# run_exp "smoke_desloc" "125M" 32 "DESLOC" 10 4 1

# ===================================================================
# Phase 1: 125M 基线 (5-seed, ~20分钟)
# 修改: 每个seed交替跑DDP+DESLOC, 立即得到对比
# ===================================================================
# echo ""
# echo "===== Phase 1: 125M DDP vs DESLOC (5-seed, interleaved) ====="

# for S in 1 2 3 4 5; do
#     PYTHONHASHSEED=$((S * 7)) run_exp "p1_ddp_125m_s${S}" "125M" 1 "DDP" 500 4 4
#     PYTHONHASHSEED=$((S * 7)) run_exp "p1_desloc_125m_Kx32_s${S}" "125M" 32 "DESLOC" 500 4 4
# done

# ===================================================================
# Phase 2: 700M (5-seed)
# base = wt(1.4)+grad(1.4)+opt(1.9/3gpu) = 4.7GB
# bs=8 seq=2048 no-AC → act=18.1GB → 22.8GB/A6000 (48%)
# bs=8 seq=1024 no-AC → act=9.1GB → 13.8GB/A6000 (29%) ← too low
# ===================================================================
# echo ""
# echo "===== Phase 2: 700M DDP vs DESLOC (5-seed, 48% GPU) ====="

# for S in 1 2 3 4 5; do
#     PYTHONHASHSEED=$((S * 7)) run_exp "p2_ddp_700m_s${S}" "700M" 1 "DDP" 500 8 4 "--zero_stage 1 --max_seq_len 2048"
#     PYTHONHASHSEED=$((S * 7)) run_exp "p2_desloc_700m_Kx32_s${S}" "700M" 32 "DESLOC" 500 8 4 "--zero_stage 1 --max_seq_len 2048"
# done

# # # ===================================================================
# # # Phase 3: Kx 消融 (700M, 3-seed, ~40分钟)
# # # ===================================================================
# # echo ""
# # echo "===== Phase 3: Kx 消融 (700M) ====="

# # for KX in 8 16 64 128; do
# #     for S in 1 2 3; do
# #         PYTHONHASHSEED=$((S * 7)) run_exp "p3_desloc_700m_Kx${KX}_s${S}" "700M" "$KX" "DESLOC" 500 4 4
# #     done
# # done

# # # ===================================================================
# # # Phase 4: Nesterov outer optimizer (700M, 3-seed, ~20分钟)
# # # ===================================================================
# # echo ""
# # echo "===== Phase 4: Nesterov vs Average ====="

# # for S in 1 2 3; do
# #     PYTHONHASHSEED=$((S * 7)) run_exp "p4_nesterov_700m_s${S}" "700M" 32 "DESLOC" 500 4 4 \
# #         "--outer_optimizer nesterov --outer_momentum 0.9"
# # done

# # # ===================================================================
# # # Phase 5: SP+DEC (Sequence Parallel + Desynced Communication)
# # # 125M only — SP halves per-GPU seq_len, enabling longer contexts
# # # or doubling effective batch size per memory unit.
# # # 注意: SP+DEC正交 — SP沿seq维度切分, DEC沿worker维度控制通信频率
# # # ===================================================================
# # echo ""
# # echo "===== Phase 5: SP+DEC (125M, Kx=32, 3-seed) ====="

# # for S in 1 2 3; do
# #     PYTHONHASHSEED=$((S * 7)) run_exp "p5_sp_desloc_125m_s${S}" "125M" 32 "DESLOC" 500 4 4 \
# #         "--use_autosp"
# # done

# # # Phase 5b: SP+DDP baseline for comparison
# # for S in 1 2 3; do
# #     PYTHONHASHSEED=$((S * 7)) run_exp "p5_sp_ddp_125m_s${S}" "125M" 1 "DDP" 500 4 4 \
# #         "--use_autosp"
# # done

# # ===================================================================
# # Phase 6: 1.3B on 3-GPU heterogeneous
# # OLD: bs=2 ga=8 AC → 27GB/A6000 (56%) — GPU idle
# # NEW: no-AC, ZeRO-1, bigger batch
# #   base(ZeRO-1/3gpu) = wt(2.6) + grad(2.6) + opt(3.5) = 8.7GB
# #   bs=8 seq=2048 no-AC → act=19.3GB → 28GB/A6000 (58%)
# #   bs=16 seq=1024 no-AC → act=19.3GB → 28GB/A6000 (58%)
# # ===================================================================
# echo ""
# echo "===== Phase 6: 1.3B DDP vs DESLOC (3-seed, 58% GPU) ====="

# for S in 1 2 3; do
#     PYTHONHASHSEED=$((S * 7)) run_exp "p6_ddp_1.3B_s${S}" "1.3B" 1 "DDP" 500 8 4 "--zero_stage 1 --max_seq_len 2048"
#     PYTHONHASHSEED=$((S * 7)) run_exp "p6_desloc_1.3B_Kx32_s${S}" "1.3B" 32 "DESLOC" 500 8 4 "--zero_stage 1 --max_seq_len 2048"
# done

# # Phase 6b: 1.3B Kx ablation
# echo "===== Phase 6b: 1.3B Kx ablation ====="
# for KX in 16 64; do
#     PYTHONHASHSEED=7 run_exp "p6b_desloc_1.3B_Kx${KX}" "1.3B" "$KX" "DESLOC" 500 8 4 "--zero_stage 1 --max_seq_len 2048"
# done

# ===================================================================
# Phase 7: 7B on 3-GPU heterogeneous (2×A6000 48GB + 1×H100 96GB)
#
# Memory strategy:
#   CPU offload optimizer → model(14GB) + grads(14GB) = 28GB on GPU
#   This leaves 20GB for activations on A6000 (48GB)
#
# Target configs (A6000 bottleneck = 48GB):
#   bs=2 seq=2048 no-AC → 40.9GB (85%) ← maximum push
#   bs=4 seq=1024 no-AC → 40.9GB (85%) ← same memory, higher throughput
#   bs=2 seq=1024 no-AC → 34.4GB (72%) ← safe
#   bs=1 seq=2048 no-AC → 34.4GB (72%) ← long context
#
# ZeRO-1 alone won't work: 46.7GB base > 48GB A6000
# CPU offload is required for 7B on A6000
# ===================================================================
# echo ""
# echo "===== Phase 7: 7B smoke test (50 steps) ====="
# # DDP: use_ac to fit on A6000 (AC reduces activations, stays under 48GB)
# run_exp "p7_smoke_ddp_7B" "7B" 1 "DDP" 50 1 8 "--use_ac"
# # DESLOC: cpu_offload pushes optimizer to CPU → room for bigger batch
# run_exp "p7_smoke_desloc_7B" "7B" 32 "DESLOC" 50 4 4 "--cpu_offload --zero_stage 1"

# sleep 5

# echo ""
# echo "===== Phase 7a: 7B DDP vs DESLOC (2-seed) ====="
# for S in 1 2; do
#     # DDP: AC + bs=2 → ~29GB/A6000 (61%)
#     PYTHONHASHSEED=$((S * 7)) run_exp "p7a_ddp_7B_s${S}" "7B" 1 "DDP" 200 2 8 "--use_ac"
#     # DESLOC: cpu_offload + no-AC + bs=4 → ~41GB/A6000 (85%)
#     PYTHONHASHSEED=$((S * 7)) run_exp "p7a_desloc_7B_Kx32_s${S}" "7B" 32 "DESLOC" 200 4 4 "--cpu_offload --zero_stage 1"
# done

# sleep 5

# ===================================================================
# Phase 0: AutoSP compile验证 (50步, ~3分钟)
# 目的: 确认AutoSP在walking3(2.7.1)上能compile通过
# 看terminal有没有 [AUTOSP-COMPILE] SUCCESS
# ===================================================================
# echo ""
# echo "===== Phase 0: AutoSP compile验证 (125M, 50 steps) ====="
# run_exp "p0_autosp_desloc_125m" "125M" 32 "DESLOC" 50 4 4 "--use_autosp"
# run_exp "p0_autosp_ddp_125m" "125M" 1 "DDP" 50 4 4 "--use_autosp"
 
# sleep 3
 
# # ===================================================================
# # Phase 5: SP+DEC 正式实验 (AutoSP + DESLOC, 125M, 3-seed)
# # ===================================================================
# echo ""
# echo "===== Phase 5: SP+DEC (125M, Kx=32, 3-seed, 500 steps) ====="
# for S in 1 2 3; do
#     PYTHONHASHSEED=$((S * 7)) run_exp "p5_sp_desloc_125m_s${S}" "125M" 32 "DESLOC" 500 4 4 "--use_autosp"
# done
 
# # Phase 5b: SP+DDP baseline
# for S in 1 2 3; do
#     PYTHONHASHSEED=$((S * 7)) run_exp "p5_sp_ddp_125m_s${S}" "125M" 1 "DDP" 500 4 4 "--use_autosp"
# done
 
# sleep 3
 
# ===================================================================
# Phase 7b: 7B long context seq=2048 (2-seed)
# ===================================================================
echo "===== Phase 7b: 7B long context seq=2048 (2-seed) ====="
for S in 1 2; do
    PYTHONHASHSEED=$((S * 7)) run_exp "p7b_ddp_7B_seq2048_s${S}" "7B" 1 "DDP" 200 1 8 "--use_ac --max_seq_len 2048"
    PYTHONHASHSEED=$((S * 7)) run_exp "p7b_sp_desloc_7B_seq2048_s${S}" "7B" 32 "DESLOC" 200 1 8 "--use_autosp --use_ac --max_seq_len 2048"
done
sleep 5
 
echo ""
echo "===== Phase 7c: 7B Kx ablation (1-seed) ====="
for KX in 16 64; do
    PYTHONHASHSEED=7 run_exp "p7c_desloc_7B_Kx${KX}" "7B" "$KX" "DESLOC" 200 4 4 "--cpu_offload --zero_stage 1"
done
sleep 5

echo ""
echo "===== Phase 7d: 7B + AC push (bs=8, 67% GPU, 1-seed) ====="
PYTHONHASHSEED=7 run_exp "p7d_desloc_7B_ac_bs8" "7B" 32 "DESLOC" 200 8 4 "--cpu_offload --zero_stage 1 --use_ac"

sleep 5

echo ""
echo "===== Phase 7e: 7B Nesterov vs Avg 1536 steps (85% GPU) ====="
PYTHONHASHSEED=7 run_exp "p7e_nesterov_7B" "7B" 32 "DESLOC" 1536 4 4 "--cpu_offload --zero_stage 1 --outer_optimizer nesterov --outer_momentum 0.9"
PYTHONHASHSEED=7 run_exp "p7e_avg_7B" "7B" 32 "DESLOC" 1536 4 4 "--cpu_offload --zero_stage 1"
 
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