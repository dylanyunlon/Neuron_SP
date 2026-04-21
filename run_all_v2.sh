#!/bin/bash
# =============================================================================
# DES-LOC Full Experiment Matrix v2 — Claude-22 (M317)
# Hardware: 2× RTX A6000 (48GB) + 1× H100 NVL (94GB)
# Strategy:
#   Phase 1: 2×A6000 (CUDA_VISIBLE_DEVICES=0,1) — RQ2+RQ3 core experiments
#   Phase 2: H100 solo (CUDA_VISIBLE_DEVICES=2)  — large model (1.3B) experiments
#   Phase 3: 3-GPU mixed (CUDA_VISIBLE_DEVICES=0,1,2) — heterogeneous scaling
#
# Total: 108 experiments across 3 phases
# Est. time: ~90 min
# =============================================================================
set -uo pipefail

SECONDS=0
TS=$(date +%Y%m%d_%H%M%S)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RESULTS_DIR="${SCRIPT_DIR}/desloc_results_${TS}"
mkdir -p "$RESULTS_DIR"/{logs,phase1_2xA6000,phase2_H100,phase3_hetero}

echo "================================================================"
echo " DES-LOC Full Experiment Matrix v2 (Claude-22)"
echo " Started: $(date)"
echo " Results: $RESULTS_DIR"
echo "================================================================"

# ── Phase 0: Environment ──────────────────────────────────
cd "$SCRIPT_DIR"

# Dependencies (quiet)
pip install -q matplotlib seaborn pandas 2>/dev/null \
  || pip install matplotlib seaborn pandas --break-system-packages -q 2>/dev/null \
  || true

# GPU inventory
echo ""
echo "[HW] GPU inventory:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

TOTAL_GPU=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "[INFO] Total GPUs visible: $TOTAL_GPU"

STEPS=500
LOG="$RESULTS_DIR/experiment_log.csv"
echo "run_id,phase,tag,model,Kx,Ku,Kv,seed,method,gpus,exit_code,elapsed_s,log_file" > "$LOG"
RUN_ID=0
FAIL=0
TOTAL_RUNS=0

# ── Run function ──────────────────────────────────────────
run_exp() {
    local PHASE=$1 TAG=$2 MODEL=$3 KX=$4 KU=$5 KV=$6 SEED=$7 METHOD=$8 GPUS=$9 NGPU=${10}

    RUN_ID=$((RUN_ID + 1))
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
    local LOGFILE="$RESULTS_DIR/logs/${PHASE}_${TAG}_${MODEL}_Kx${KX}_${METHOD}_s${SEED}.log"
    local T0=$SECONDS
    local OUTDIR="$RESULTS_DIR/${PHASE}"

    printf "[%3d] %-8s %-12s %-5s Kx=%-3d %-10s seed=%-4d gpu=%s ... " \
        "$RUN_ID" "$PHASE" "$TAG" "$MODEL" "$KX" "$METHOD" "$SEED" "$GPUS"

    export PYTHONHASHSEED=$SEED
    export CUDA_VISIBLE_DEVICES=$GPUS

    torchrun --nproc_per_node=$NGPU \
        --master_port=$((29500 + RUN_ID % 200)) \
        REAL_GPU_BENCHMARK.py \
        --model_size "$MODEL" \
        --batch_size 4 \
        --grad_accum 8 \
        --max_steps $STEPS \
        --Kx "$KX" --Ku "$KU" --Kv "$KV" \
        --methods "$METHOD" \
        --output "$OUTDIR" \
        > "$LOGFILE" 2>&1

    local RC=$?
    local DT=$((SECONDS - T0))

    if [ $RC -eq 0 ]; then
        echo "OK (${DT}s)"
    else
        echo "FAIL:${RC} (${DT}s)"
        FAIL=$((FAIL + 1))
    fi

    echo "${RUN_ID},${PHASE},${TAG},${MODEL},${KX},${KU},${KV},${SEED},${METHOD},${GPUS},${RC},${DT},${LOGFILE}" >> "$LOG"
}

# ══════════════════════════════════════════════════════════════
# PHASE 1: 2× A6000 (GPU 0,1) — Main experiments
# ══════════════════════════════════════════════════════════════
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  PHASE 1: 2× RTX A6000 — Core DES-LOC Experiments       ║"
echo "╚════════════════════════════════════════════════════════════╝"

# ── RQ2: Sync Frequency Sweep ────────────────────────────
# Kx ∈ {1,2,4,8,16,32,64,128} × 3 seeds = 24 runs
echo ""
echo "━━━ RQ2: Sync Frequency Sweep (Section 5.2) ━━━"
for KX in 1 2 4 8 16 32 64 128; do
    if [ $KX -eq 1 ]; then
        KU=1; KV=1
    else
        KU=$((KX * 3))
        KV=$((KX * 6))
    fi
    for SEED in 42 137 2024; do
        run_exp "phase1_2xA6000" "rq2_sweep" "125M" $KX $KU $KV $SEED "DESLOC" "0,1" 2
    done
done

# ── RQ3: Communication Reduction ─────────────────────────
# 2 models × 3 methods × 3 seeds = 18 runs
echo ""
echo "━━━ RQ3: Communication Reduction (Section 5.3) ━━━"
for MODEL in 125M 350M; do
    for SEED in 42 137 2024; do
        run_exp "phase1_2xA6000" "rq3_comm" "$MODEL" 1   1   1   $SEED "DDP"       "0,1" 2
        run_exp "phase1_2xA6000" "rq3_comm" "$MODEL" 32  32  32  $SEED "LocalAdam"  "0,1" 2
        run_exp "phase1_2xA6000" "rq3_comm" "$MODEL" 32  96  192 $SEED "DESLOC"     "0,1" 2
    done
done

# ── RQ4: Ku/Kv Ratio Ablation ────────────────────────────
# Fixed Kx=32, vary Ku_ratio ∈ {1,3,6}, Kv_ratio ∈ {1,6,12}
# 9 combos × 3 seeds = 27 runs
echo ""
echo "━━━ RQ4: Ku/Kv Ratio Ablation (Section 5.4) ━━━"
for KU_RATIO in 1 3 6; do
    for KV_RATIO in 1 6 12; do
        KU=$((32 * KU_RATIO))
        KV=$((32 * KV_RATIO))
        for SEED in 42 137 2024; do
            run_exp "phase1_2xA6000" "rq4_ratio" "125M" 32 $KU $KV $SEED "DESLOC" "0,1" 2
        done
    done
done

# ══════════════════════════════════════════════════════════════
# PHASE 2: H100 solo (GPU 2) — Large model experiments
# ══════════════════════════════════════════════════════════════
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  PHASE 2: H100 NVL Solo — Large Model Scaling            ║"
echo "╚════════════════════════════════════════════════════════════╝"

# Single-GPU scaling: 125M, 350M, 700M, 1.3B × 3 methods × 1 seed
# 4 models × 3 methods = 12 runs (single GPU, so DDP=baseline)
echo ""
echo "━━━ RQ5: Model Scale vs Loss (Section 5.5) ━━━"
for MODEL in 125M 350M 700M 1.3B; do
    SEED=42
    run_exp "phase2_H100" "rq5_scale" "$MODEL" 1   1   1   $SEED "DDP"       "2" 1
    run_exp "phase2_H100" "rq5_scale" "$MODEL" 32  96  192 $SEED "DESLOC"    "2" 1
    run_exp "phase2_H100" "rq5_scale" "$MODEL" 32  32  32  $SEED "LocalAdam" "2" 1
done

# ══════════════════════════════════════════════════════════════
# PHASE 3: 3-GPU heterogeneous (GPU 0,1,2) — Hetero scaling
# ══════════════════════════════════════════════════════════════
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  PHASE 3: 2×A6000 + H100 — Heterogeneous Scaling         ║"
echo "╚════════════════════════════════════════════════════════════╝"

# 2 models × 3 methods × 3 seeds = 18 runs
echo ""
echo "━━━ RQ6: Heterogeneous GPU Training (Section 5.6) ━━━"
for MODEL in 125M 350M; do
    for SEED in 42 137 2024; do
        run_exp "phase3_hetero" "rq6_hetero" "$MODEL" 1   1   1   $SEED "DDP"       "0,1,2" 3
        run_exp "phase3_hetero" "rq6_hetero" "$MODEL" 32  96  192 $SEED "DESLOC"    "0,1,2" 3
        run_exp "phase3_hetero" "rq6_hetero" "$MODEL" 32  32  32  $SEED "LocalAdam" "0,1,2" 3
    done
done

# ══════════════════════════════════════════════════════════════
# AGGREGATE + PLOT
# ══════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo " ALL PHASES COMPLETED — $(date)"
echo " Total: ${TOTAL_RUNS} experiments in $((SECONDS/60))m$((SECONDS%60))s"
echo " Success: $((TOTAL_RUNS - FAIL)) / ${TOTAL_RUNS}"
echo " Failed: ${FAIL}"
echo " Results: $RESULTS_DIR"
echo "================================================================"

# Merge all JSON results
echo ""
echo "[MERGE] Collecting results..."
python3 << 'PYEOF'
import json, glob, os, csv

results_dir = os.environ.get('RESULTS_DIR', '.')

# Collect all benchmark JSON files across phases
all_data = []
for phase_dir in ['phase1_2xA6000', 'phase2_H100', 'phase3_hetero']:
    pattern = os.path.join(results_dir, phase_dir, 'benchmark_results_*.json')
    for f in sorted(glob.glob(pattern)):
        try:
            with open(f) as fh:
                data = json.load(fh)
                data['source_file'] = f
                data['phase'] = phase_dir
                all_data.append(data)
        except Exception as e:
            print(f"  WARN: {f}: {e}")

# Also parse NKI-FA format logs
import re
nkifa_pattern = re.compile(
    r'### model = (\S+), method = (\S+), Kx = (\d+), Ku = (\d+), Kv = (\d+), world_size = (\d+) ###'
)
metric_pattern = re.compile(r'^(\w+):\s+(.+)$')

nkifa_records = []
for logf in sorted(glob.glob(os.path.join(results_dir, 'logs', '*.log'))):
    with open(logf) as fh:
        lines = fh.readlines()
    current = None
    for line in lines:
        m = nkifa_pattern.match(line.strip())
        if m:
            current = {
                'model': m.group(1), 'method': m.group(2),
                'Kx': int(m.group(3)), 'Ku': int(m.group(4)),
                'Kv': int(m.group(5)), 'world_size': int(m.group(6)),
                'log_file': logf
            }
            continue
        if current:
            mm = metric_pattern.match(line.strip())
            if mm:
                key, val = mm.group(1), mm.group(2)
                try:
                    current[key] = float(val)
                except ValueError:
                    current[key] = val
            elif line.strip() == '' or line.startswith('=') or line.startswith('-'):
                if len(current) > 6:
                    nkifa_records.append(current)
                current = None

merged = {
    'total_experiments': len(all_data),
    'total_nkifa_records': len(nkifa_records),
    'experiments': all_data,
    'nkifa_parsed': nkifa_records,
}

out = os.path.join(results_dir, 'ALL_RESULTS.json')
with open(out, 'w') as fh:
    json.dump(merged, fh, indent=2)

print(f"  Merged {len(all_data)} JSON + {len(nkifa_records)} NKI-FA records → {out}")

# Quick summary table
print("\n  === Quick Summary ===")
for rec in nkifa_records[:20]:
    print(f"  {rec.get('model','?'):>5s} {rec.get('method','?'):>10s} "
          f"Kx={rec.get('Kx','?'):>3s} loss={rec.get('final_loss','?')}")
PYEOF

# Pack results
tar czf "${RESULTS_DIR}.tar.gz" -C "$(dirname $RESULTS_DIR)" "$(basename $RESULTS_DIR)"
echo ""
echo "[DONE] Packed: ${RESULTS_DIR}.tar.gz ($(du -h ${RESULTS_DIR}.tar.gz | cut -f1))"

# Generate figures
echo ""
echo "[PLOT] Generating NKI-FA grade figures..."
python3 -c "
import sys; sys.path.insert(0, '.')
from REAL_GPU_BENCHMARK import desloc_draw_all_figures
desloc_draw_all_figures('${RESULTS_DIR}')
" 2>&1 || echo "[WARN] Figure generation had errors"

# Re-pack with figures
tar czf "${RESULTS_DIR}.tar.gz" -C "$(dirname $RESULTS_DIR)" "$(basename $RESULTS_DIR)"
echo ""
echo "[FINAL] ${RESULTS_DIR}.tar.gz ($(du -h ${RESULTS_DIR}.tar.gz | cut -f1))"
echo "下载结果:"
echo "  scp user@host:${RESULTS_DIR}.tar.gz ."
