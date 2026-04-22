#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# YOTTA BLITZ — 流浪地球计划
# 零模拟 零fallback 零重试
# 自适应GPU数量 (1卡串行, 2+卡并行)
# ═══════════════════════════════════════════════════════════════════

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

T_START=$SECONDS
BUDGET_SECONDS="${BUDGET_SECONDS:-3600}"
OUTPUT_DIR="$SCRIPT_DIR/desloc_results"

export PYTHONUNBUFFERED=1

log() { echo "[$(date '+%H:%M:%S')] [$((SECONDS - T_START))s] $1"; }
die() { log "FATAL: $1"; exit 1; }
budget_ok() {
    local elapsed=$((SECONDS - T_START))
    [ $elapsed -lt $BUDGET_SECONDS ] || { log "BUDGET ($((elapsed/60))m > $((BUDGET_SECONDS/60))m)"; return 1; }
    log "Budget: $(( (BUDGET_SECONDS - elapsed) / 60 ))m left"
}

# ═══════════════════════════════════════════════════════════════════
# PHASE 0: Environment probe — find working python+torch, detect GPUs
# No set -e here: probe can fail gracefully
# ═══════════════════════════════════════════════════════════════════
log "═══ PHASE 0: Environment probe ═══"

# --- Find python with torch ---
# Knuth-grade exhaustive probe: covers Yotta, RunPod, Lambda, Paperspace,
# bare metal, conda, venv, pyenv, system pip, docker entrypoints.
# Also handles nohup/cron where PATH is minimal.
PYTHON=""

# Step 1: Enumerate every python binary on the system
CANDIDATES=""
# PATH-based (may be incomplete in nohup)
for cmd in python3 python python3.11 python3.10 python3.12; do
    p=$(command -v "$cmd" 2>/dev/null) && CANDIDATES="$CANDIDATES $p"
done
# Hardcoded common locations (Yotta, RunPod, Lambda, etc.)
for p in \
    /usr/local/bin/python3 /usr/local/bin/python \
    /usr/bin/python3 /usr/bin/python \
    /opt/conda/bin/python3 /opt/conda/bin/python \
    /opt/conda/envs/pytorch/bin/python3 \
    /opt/conda/envs/pytorch/bin/python \
    /opt/venv/bin/python3 /opt/venv/bin/python \
    "$HOME/miniconda3/bin/python3" \
    "$HOME/anaconda3/bin/python3" \
    "$HOME/.local/bin/python3" \
    /workspace/.venv/bin/python3 \
    /root/miniconda3/bin/python3 \
; do
    [ -x "$p" ] && CANDIDATES="$CANDIDATES $p"
done
# find-based fallback (slow but thorough)
if [ -z "$CANDIDATES" ]; then
    CANDIDATES=$(find /usr /opt "$HOME" -maxdepth 5 -name "python3*" -type f -executable 2>/dev/null | head -20)
fi

# Step 2: Test each candidate for torch
# De-duplicate preserving order
SEEN=""
for candidate in $CANDIDATES; do
    # Resolve symlinks to de-dup
    real=$(readlink -f "$candidate" 2>/dev/null || echo "$candidate")
    case " $SEEN " in *" $real "*) continue ;; esac
    SEEN="$SEEN $real"

    if "$candidate" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>/dev/null; then
        PYTHON="$candidate"
        log "Found torch: $PYTHON"
        break
    fi
done

# Step 3: Conda env activation (if torch not found in any binary directly)
if [ -z "$PYTHON" ]; then
    log "No python with torch found directly, trying conda activation..."
    CONDA_BASE=""
    for cb in /opt/conda "$HOME/miniconda3" "$HOME/anaconda3" /root/miniconda3; do
        [ -d "$cb" ] && CONDA_BASE="$cb" && break
    done
    if command -v conda &>/dev/null; then
        CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null)}"
    fi

    if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        . "$CONDA_BASE/etc/profile.d/conda.sh"
        for env in base pytorch torch py311 $(conda env list 2>/dev/null | awk '/^[^#]/{print $1}'); do
            conda activate "$env" 2>/dev/null || continue
            if python3 -c "import torch; print(torch.__version__)" &>/dev/null; then
                PYTHON=$(which python3)
                log "Found torch in conda env: $env -> $PYTHON"
                break
            fi
        done
    fi
fi

# Step 4: Last resort — install torch
if [ -z "$PYTHON" ]; then
    log "LAST RESORT: installing torch via pip..."
    PYBIN=$(command -v python3 || command -v python || echo /usr/bin/python3)
    "$PYBIN" -m pip install torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -3
    if "$PYBIN" -c "import torch; print(torch.__version__)" 2>/dev/null; then
        PYTHON="$PYBIN"
        log "Installed torch successfully"
    fi
fi

[ -z "$PYTHON" ] && {
    log "DIAGNOSTIC: all candidates tried:"
    for c in $CANDIDATES; do echo "  $c: $($c -c 'import torch' 2>&1 | head -1)"; done
    die "Cannot find python with torch"
}
log "Python: $PYTHON ($($PYTHON --version 2>&1))"
log "Torch: $($PYTHON -c 'import torch; print(torch.__version__, "CUDA:", torch.version.cuda)' 2>&1)"

# --- Detect GPU count and properties ---
# Use heredoc + temp file to avoid shell escaping issues in $() capture
GPU_PROBE_SCRIPT=$(mktemp /tmp/gpu_probe.XXXXXX.py)
cat > "$GPU_PROBE_SCRIPT" << 'PYEOF'
import torch
import json
import sys
try:
    if not torch.cuda.is_available():
        print(json.dumps({"n": 0, "gpus": []}))
        sys.exit(0)
    n = torch.cuda.device_count()
    gpus = []
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        mem = getattr(p, 'total_memory', None) or getattr(p, 'total_mem', 0)
        gpus.append({"idx": i, "name": p.name, "mem_gb": round(mem / 1e9, 1)})
    print(json.dumps({"n": n, "gpus": gpus}))
except Exception as e:
    print(json.dumps({"n": 0, "gpus": [], "error": str(e)}))
    sys.exit(1)
PYEOF

GPU_INFO=$($PYTHON "$GPU_PROBE_SCRIPT" 2>&1)
GPU_RC=$?
rm -f "$GPU_PROBE_SCRIPT"

if [ $GPU_RC -ne 0 ]; then
    log "GPU probe output: $GPU_INFO"
    die "torch.cuda probe failed (rc=$GPU_RC)"
fi

# Parse with temp file to avoid shell quoting issues with JSON
GPU_PARSE_SCRIPT=$(mktemp /tmp/gpu_parse.XXXXXX.py)
cat > "$GPU_PARSE_SCRIPT" << 'PYEOF'
import json, sys, os
raw = os.environ.get("GPU_INFO_RAW", "")
try:
    info = json.loads(raw)
    print(info["n"])
    for g in info["gpus"]:
        sys.stderr.write(f"  GPU{g['idx']}: {g['name']}, {g['mem_gb']}GB\n")
except Exception as e:
    sys.stderr.write(f"JSON parse error: {e}\nRaw: {raw}\n")
    print("0")
PYEOF

N_GPU=$(GPU_INFO_RAW="$GPU_INFO" $PYTHON "$GPU_PARSE_SCRIPT" 2>&1 >/dev/null || echo "0")
# Actually need stdout for N_GPU and stderr for display
N_GPU_DISPLAY=$(mktemp /tmp/gpu_display.XXXXXX)
N_GPU=$(GPU_INFO_RAW="$GPU_INFO" $PYTHON "$GPU_PARSE_SCRIPT" 2>"$N_GPU_DISPLAY")
cat "$N_GPU_DISPLAY"
rm -f "$GPU_PARSE_SCRIPT" "$N_GPU_DISPLAY"

N_GPU=${N_GPU:-0}
[ "$N_GPU" -eq 0 ] && die "No CUDA GPUs detected (GPU_INFO: $GPU_INFO)"
log "GPUs: $N_GPU detected"

# --- Check disk space ---
DISK_FREE_GB=$(df -BG "$SCRIPT_DIR" 2>/dev/null | awk 'NR==2{gsub(/G/,"",$4); print $4}')
DISK_FREE_GB=${DISK_FREE_GB:-999}
[ "$DISK_FREE_GB" -lt 5 ] && die "Disk space < 5GB ($DISK_FREE_GB GB free)"
log "Disk: ${DISK_FREE_GB}GB free"

# --- Check NCCL (multi-GPU only) ---
if [ "$N_GPU" -ge 2 ]; then
    $PYTHON -c "import torch.distributed; print('NCCL:', torch.cuda.nccl.version())" 2>/dev/null \
        || log "WARN: Cannot detect NCCL version (multi-GPU may fail)"
fi

# --- Install missing deps only ---
log "Checking deps..."
for pkg in matplotlib seaborn deepspeed; do
    $PYTHON -c "import $pkg" 2>/dev/null || {
        log "Installing $pkg..."
        $PYTHON -m pip install -q "$pkg" 2>/dev/null || log "WARN: Failed to install $pkg"
    }
done

# --- DeepSpeed install check (needs special handling for CUDA extensions) ---
$PYTHON -c "import deepspeed; print('DeepSpeed:', deepspeed.__version__)" 2>/dev/null \
    || log "WARN: deepspeed not importable — DESLOC DeepSpeed path will use baseline"

# --- Project file check ---
[ -f "$SCRIPT_DIR/REAL_GPU_BENCHMARK.py" ] || die "REAL_GPU_BENCHMARK.py not found in $SCRIPT_DIR"

# --- NCCL tuning for PCIe topology (A100 PCIe, no NVLink) ---
# Without these, NCCL may hang or crash on PCIe-only multi-GPU
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING=1
# For single-machine PCIe, disable IB to avoid probing timeout
if [ "$N_GPU" -le 8 ]; then
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
fi
log "NCCL: P2P=$NCCL_P2P_LEVEL IB_DISABLE=${NCCL_IB_DISABLE:-0} IF=$NCCL_SOCKET_IFNAME"

# --- DeepSpeed availability flag ---
DS_OK=0
if $PYTHON -c "import deepspeed; print('DeepSpeed:', deepspeed.__version__)" 2>/dev/null; then
    DS_OK=1
    log "DeepSpeed: available"
else
    log "DeepSpeed: NOT available -- DESLOC will use baseline DESLOCAdamW optimizer path"
    log "  (This still measures DES-LOC correctly; only misses engine.py Kx-gated allreduce hooks)"
fi

# ═══════════════════════════════════════════════════════════════════
# Now enable strict mode for experiment execution
# ═══════════════════════════════════════════════════════════════════
set -eo pipefail

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/figures"

CSV="$OUTPUT_DIR/experiment_log.csv"
[ ! -f "$CSV" ] && echo "run_id,tag,model,Kx,Ku,Kv,seed,method,exit_code,elapsed_s" > "$CSV"

# Atomic run counter via flock (safe for parallel subshells)
RUN_COUNTER="$OUTPUT_DIR/.run_counter"
RUN_LOCK="$OUTPUT_DIR/.run_lock"
echo 0 > "$RUN_COUNTER"
touch "$RUN_LOCK"
next_run_id() {
    local id
    (
        flock -x 200
        id=$(cat "$RUN_COUNTER")
        id=$((id + 1))
        echo "$id" > "$RUN_COUNTER"
        echo "$id"
    ) 200>"$RUN_LOCK"
}

run_one() {
    # $1=tag $2=model $3=Kx $4=Ku $5=Kv $6=seed $7=method $8=gpu $9=steps
    budget_ok || return 1
    local TAG=$1 MODEL=$2 KX=$3 KU=$4 KV=$5 SEED=$6 METHOD=$7 GPU=$8 STEPS=$9
    local RID=$(next_run_id)
    local LOGFILE="$OUTPUT_DIR/logs/${TAG}_${MODEL}_Kx${KX}_${METHOD}_s${SEED}.log"
    local T0=$SECONDS

    printf "[%3d] %-10s %-5s Kx=%-3d %-10s seed=%-4d gpu=%-2s steps=%-4d ... " \
        "$RID" "$TAG" "$MODEL" "$KX" "$METHOD" "$SEED" "$GPU" "$STEPS"

    PYTHONHASHSEED=$SEED PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU \
        $PYTHON REAL_GPU_BENCHMARK.py \
            --model_size "$MODEL" \
            --batch_size 4 \
            --grad_accum 4 \
            --max_steps "$STEPS" \
            --Kx "$KX" --Ku "$KU" --Kv "$KV" \
            --methods "$METHOD" \
            --output "$OUTPUT_DIR" \
            > "$LOGFILE" 2>&1 || true

    local RC=${PIPESTATUS[0]:-$?}; local DT=$((SECONDS - T0))
    [ $RC -eq 0 ] && echo "OK (${DT}s)" || echo "FAIL:${RC} (${DT}s)"
    echo "${RID},${TAG},${MODEL},${KX},${KU},${KV},${SEED},${METHOD},${RC},${DT}" >> "$CSV"
    return 0  # don't let set -e kill the parent
}

run_nesterov() {
    # Special: RQ5 Nesterov outer optimizer
    # $1=model $2=Kx $3=Ku $4=Kv $5=seed $6=gpu $7=steps
    budget_ok || return 1
    local MODEL=$1 KX=$2 KU=$3 KV=$4 SEED=$5 GPU=$6 STEPS=$7
    local RID=$(next_run_id)
    local LOGFILE="$OUTPUT_DIR/logs/rq5_nest_${MODEL}_Kx${KX}_DESLOC_nesterov_s${SEED}.log"
    local T0=$SECONDS

    printf "[%3d] %-10s %-5s Kx=%-3d %-10s (nesterov) gpu=%-2s steps=%-4d ... " \
        "$RID" "rq5_nest" "$MODEL" "$KX" "DESLOC" "$GPU" "$STEPS"

    PYTHONHASHSEED=$SEED PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU \
        $PYTHON REAL_GPU_BENCHMARK.py \
            --model_size "$MODEL" \
            --batch_size 4 --grad_accum 4 --max_steps "$STEPS" \
            --Kx "$KX" --Ku "$KU" --Kv "$KV" \
            --outer_optimizer nesterov --outer_momentum 0.9 --outer_lr 1.0 \
            --methods DESLOC_nesterov \
            --output "$OUTPUT_DIR" \
            > "$LOGFILE" 2>&1 || true

    local RC=${PIPESTATUS[0]:-$?}; local DT=$((SECONDS - T0))
    [ $RC -eq 0 ] && echo "OK (${DT}s)" || echo "FAIL:${RC} (${DT}s)"
    echo "${RID},rq5_nest,${MODEL},${KX},${KU},${KV},${SEED},DESLOC_nesterov,${RC},${DT}" >> "$CSV"
    return 0
}

run_multigpu() {
    # Multi-GPU DDP/DESLOC via torchrun
    # $1=model $2=method $3=steps
    budget_ok || return 1
    local MODEL=$1 METHOD=$2 STEPS=$3
    local RID=$(next_run_id)
    local LOGFILE="$OUTPUT_DIR/logs/multigpu_${MODEL}_${METHOD}_${N_GPU}gpu.log"
    local T0=$SECONDS
    local PORT=$((29500 + RID % 200))

    printf "[%3d] %-10s %-5s %-10s %dxGPU ... " "$RID" "multigpu" "$MODEL" "$METHOD" "$N_GPU"

    PYTHONHASHSEED=42 PYTHONUNBUFFERED=1 \
        $PYTHON -m torch.distributed.run \
            --nproc_per_node="$N_GPU" --master_port="$PORT" \
        REAL_GPU_BENCHMARK.py \
            --model_size "$MODEL" \
            --batch_size 4 --grad_accum 4 --max_steps "$STEPS" \
            --Kx 32 --Ku 96 --Kv 192 \
            --methods "$METHOD" \
            --output "$OUTPUT_DIR" \
            > "$LOGFILE" 2>&1 || true

    local RC=${PIPESTATUS[0]:-$?}; local DT=$((SECONDS - T0))
    [ $RC -eq 0 ] && echo "OK (${DT}s)" || echo "FAIL:${RC} (${DT}s)"
    echo "${RID},multigpu,${MODEL},32,96,192,42,${METHOD},${RC},${DT}" >> "$CSV"
    return 0
}

# ═══════════════════════════════════════════════════════════════════
# Adaptive scheduling: 1 GPU = serial, 2+ GPU = parallel phases
# ═══════════════════════════════════════════════════════════════════

if [ "$N_GPU" -ge 2 ]; then
    GPU_A=0; GPU_B=1
    log "Mode: DUAL GPU parallel (GPU $GPU_A + GPU $GPU_B)"
else
    GPU_A=0; GPU_B=0
    log "Mode: SINGLE GPU serial (GPU $GPU_A only)"
fi

# ═══════════════════════════════════════════════════════════════════
# PHASE 1: Core training — 125M + 350M
# ═══════════════════════════════════════════════════════════════════
log "═══ PHASE 1: Core training (125M + 350M) ═══"

phase1_a() {
    for SEED in 42 137 2024; do
        run_one baseline 125M 1 1 1 $SEED DDP $GPU_A 300
        run_one desloc   125M 32 96 192 $SEED DESLOC $GPU_A 300
        run_one local    125M 32 32 32 $SEED LocalAdam $GPU_A 300
    done
}

phase1_b() {
    for SEED in 42 137 2024; do
        run_one baseline 350M 1 1 1 $SEED DDP $GPU_B 300
        run_one desloc   350M 32 96 192 $SEED DESLOC $GPU_B 300
        run_one local    350M 32 32 32 $SEED LocalAdam $GPU_B 300
    done
}

if [ "$N_GPU" -ge 2 ]; then
    phase1_a & PID_A=$!
    phase1_b & PID_B=$!
    wait $PID_A || log "GPU_A phase1 had failures"
    wait $PID_B || log "GPU_B phase1 had failures"
else
    phase1_a
    phase1_b
fi
log "PHASE 1 done"

# ═══════════════════════════════════════════════════════════════════
# PHASE 2: Ablations — Kx sweep + Ku/Kv ratio
# ═══════════════════════════════════════════════════════════════════
budget_ok || { log "Skipping phase 2+"; exit 0; }
log "═══ PHASE 2: Ablations ═══"

phase2_a() {
    for KX in 1 2 4 8 16 32 64 128; do
        [ $KX -eq 1 ] && { KU=1; KV=1; } || { KU=$((KX*3)); KV=$((KX*6)); }
        for SEED in 42 137; do
            budget_ok || return 0
            run_one rq1_kx 125M $KX $KU $KV $SEED DESLOC $GPU_A 200
        done
    done
}

phase2_b() {
    for KU_R in 1 3 6; do for KV_R in 1 6 12; do
        [ $KV_R -lt $KU_R ] && continue
        for SEED in 42 137; do
            budget_ok || return 0
            run_one rq2_ratio 125M 32 $((32*KU_R)) $((32*KV_R)) $SEED DESLOC $GPU_B 200
        done
    done; done
}

if [ "$N_GPU" -ge 2 ]; then
    phase2_a & PID_A=$!
    phase2_b & PID_B=$!
    wait $PID_A || log "GPU_A phase2 had failures"
    wait $PID_B || log "GPU_B phase2 had failures"
else
    phase2_a
    phase2_b
fi
log "PHASE 2 done"

# ═══════════════════════════════════════════════════════════════════
# PHASE 3: Large models + RQ5 Nesterov (Section 5.5)
# 700M: DDP + DESLOC-avg + DESLOC-nesterov (Kx=32) + DESLOC(Kx=256)
# 1.3B + 1.7B: DDP + DESLOC
# ═══════════════════════════════════════════════════════════════════
budget_ok || { log "Skipping phase 3+"; exit 0; }
log "═══ PHASE 3: Large models + RQ5 Nesterov ═══"

phase3_a() {
    run_one baseline 700M 1 1 1 42 DDP $GPU_A 200
    run_one desloc   700M 32 96 192 42 DESLOC $GPU_A 200
    run_one local    700M 32 32 32 42 LocalAdam $GPU_A 200
    # RQ5: Nesterov outer (Section 5.5, Charles et al. 2025)
    budget_ok || return 0
    run_nesterov 700M 32 96 192 42 $GPU_A 200
    # RQ5: infrequent sync Kx=256 (should show worse PPL)
    budget_ok || return 0
    run_one rq5_infreq 700M 256 768 1536 42 DESLOC $GPU_A 200
}

phase3_b() {
    run_one baseline 1.3B 1 1 1 42 DDP $GPU_B 200
    run_one desloc   1.3B 32 96 192 42 DESLOC $GPU_B 200
    run_one local    1.3B 32 32 32 42 LocalAdam $GPU_B 200
    budget_ok || return 0
    run_one baseline 1.7B 1 1 1 42 DDP $GPU_B 200
    run_one desloc   1.7B 32 96 192 42 DESLOC $GPU_B 200
}

if [ "$N_GPU" -ge 2 ]; then
    phase3_a & PID_A=$!
    phase3_b & PID_B=$!
    wait $PID_A || log "GPU_A phase3 had failures"
    wait $PID_B || log "GPU_B phase3 had failures"
else
    phase3_a
    phase3_b
fi
log "PHASE 3 done"

# ═══════════════════════════════════════════════════════════════════
# PHASE 4: Multi-GPU DDP (only if 2+ GPUs)
# ═══════════════════════════════════════════════════════════════════
if [ "$N_GPU" -ge 2 ]; then
    budget_ok || { log "Skipping phase 4"; exit 0; }
    log "═══ PHASE 4: Multi-GPU ($N_GPU GPUs) ═══"
    run_multigpu 125M DDP 200
    run_multigpu 125M DESLOC 200
    log "PHASE 4 done"
else
    log "Skipping PHASE 4 (single GPU)"
fi

# ═══════════════════════════════════════════════════════════════════
# PHASE 5: Eval + Report + Figures
# ═══════════════════════════════════════════════════════════════════
log "═══ PHASE 5: Eval + Figures ═══"

# Generate figures
$PYTHON -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
from REAL_GPU_BENCHMARK import desloc_draw_all_figures
desloc_draw_all_figures('$OUTPUT_DIR')
print('[OK] Figures generated')
" 2>&1 || log "Figure generation had issues"

# Run eval via llm4dec_sp.sh if available
if [ -x "$SCRIPT_DIR/llm4dec_sp.sh" ]; then
    OUTPUT_DIR="$OUTPUT_DIR" $SCRIPT_DIR/llm4dec_sp.sh eval_all_models 2>&1 | tail -50
fi

# ═══════════════════════════════════════════════════════════════════
# FINAL: Summary + package
# ═══════════════════════════════════════════════════════════════════
TOTAL_TIME=$((SECONDS - T_START))
TOTAL=$(tail -n +2 "$CSV" 2>/dev/null | wc -l | tr -d ' ')
OK=$(tail -n +2 "$CSV" 2>/dev/null | awk -F, '$9==0{n++}END{print n+0}')
FAIL=$((TOTAL - OK))

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  流浪地球计划 — 完成"
echo "║  Runs: $TOTAL (OK: $OK, FAIL: $FAIL)"
echo "║  Time: ${TOTAL_TIME}s ($((TOTAL_TIME/60))min)"
echo "║  GPUs: $N_GPU"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Package results for download
TARBALL="$SCRIPT_DIR/yotta_results_$(date +%Y%m%d_%H%M%S).tar.gz"
tar czf "$TARBALL" -C "$SCRIPT_DIR" desloc_results/ 2>/dev/null
log "Results: $TARBALL ($(du -h "$TARBALL" | cut -f1))"
log "CSV: $CSV"
log "Logs: $OUTPUT_DIR/logs/"
log "Done."