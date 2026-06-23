#!/usr/bin/env bash
# ===========================================================================
# Neuron_SP: Scaling Law Experiment Runner
# ===========================================================================
#
# Runs two short training probes for Chinchilla scaling-law fitting:
#   - tiny_70m  : 500 steps  (hidden=512,  layers=8,  heads=8)
#   - large_1b  : 200 steps  (hidden=2048, layers=16, heads=16)
#
# For each model the script:
#   1. Runs run_pretrain.py and captures stdout
#   2. Parses per-step loss values from the training log
#   3. Writes a JSON loss record to experiments/scaling_law/
#
# After both runs, calls:
#   python experiments/scaling_law/fit_scaling_curve.py --predict-7b
# which reads experiments/scaling_law/scaling_fit_results.json (pre-existing
# or just generated) and writes scaling_7b_predictions.json.
#
# Usage:
#   bash scripts/launch_scaling_law.sh
#   bash scripts/launch_scaling_law.sh --dry-run   # echo commands only
#
# Environment:
#   CUDA_VISIBLE_DEVICES  override GPU selection (default: best available)
#   SCALING_BATCH_SIZE    micro-batch size for both runs (default: 2)
#   SCALING_SEQ_LEN       sequence length (default: 2048)
#   NEURON_SP_DIR         root of repo (auto-detected from script location)
#
# Outputs (under experiments/scaling_law/):
#   70m_loss_log.json   — step-by-step loss for the 70M probe
#   1b_loss_log.json    — step-by-step loss for the 1B probe
#   scaling_7b_predictions.json  — predicted 7B loss at various token budgets
# ===========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEURON_SP_DIR="${NEURON_SP_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SCALING_DIR="${NEURON_SP_DIR}/experiments/scaling_law"
LOG_DIR="${NEURON_SP_DIR}/logs/scaling_law"
PRETRAIN_SCRIPT="${NEURON_SP_DIR}/run_pretrain.py"
FIT_SCRIPT="${NEURON_SP_DIR}/experiments/scaling_law/fit_scaling_curve.py"

mkdir -p "${SCALING_DIR}" "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BATCH_SIZE="${SCALING_BATCH_SIZE:-2}"
SEQ_LEN="${SCALING_SEQ_LEN:-2048}"
LOG_EVERY=10        # log interval passed to run_pretrain.py

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "[launch_scaling_law] DRY-RUN mode — commands will be echoed, not executed."
fi

# ---------------------------------------------------------------------------
# Helper: run or echo
# ---------------------------------------------------------------------------
run() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "  [DRY-RUN] $*"
    else
        "$@"
    fi
}

# ---------------------------------------------------------------------------
# Helper: parse loss log and emit JSON
#
# Expects lines of the form printed by run_pretrain.py:
#   "   500   2.3451   3.00e-04   12345   ..."
#   step  avg_loss  lr  tok_s  ...
#
# Writes:
#   {
#     "model_size": "70m",
#     "steps": 500,
#     "batch_size": 2,
#     "seq_len": 2048,
#     "params_approx": 70000000,
#     "tokens_per_step": 4096,
#     "loss_curve": [
#       {"step": 10,  "loss": 9.1234},
#       {"step": 20,  "loss": 8.9012},
#       ...
#     ],
#     "final_loss": 2.3451,
#     "initial_loss": 9.1234
#   }
# ---------------------------------------------------------------------------
parse_and_save_loss_json() {
    local raw_log="$1"
    local out_json="$2"
    local model_size="$3"
    local steps="$4"
    local params="$5"

    python3 - <<PYEOF
import json, re, sys

raw_log   = """$(cat "${raw_log}" 2>/dev/null || echo "")"""
out_json  = "${out_json}"
model_size = "${model_size}"
steps     = int("${steps}")
params    = int("${params}")
batch     = int("${BATCH_SIZE}")
seq_len   = int("${SEQ_LEN}")

# run_pretrain.py prints training rows as:
#   "  <step>  <avg_loss>  <lr>  <tok/s>  <mem>"
# The header line starts with "step" so we skip it.
# We match: leading whitespace, integer step, float loss.
pattern = re.compile(r'^\s{0,8}(\d+)\s+([\d]+\.[\d]+)\s+[\d.e+\-]+')

curve = []
for line in raw_log.splitlines():
    m = pattern.match(line)
    if m:
        step_val = int(m.group(1))
        loss_val = float(m.group(2))
        curve.append({"step": step_val, "loss": loss_val})

tokens_per_step = batch * seq_len
total_tokens    = steps * tokens_per_step

result = {
    "model_size":      model_size,
    "steps":           steps,
    "batch_size":      batch,
    "seq_len":         seq_len,
    "params_approx":   params,
    "tokens_per_step": tokens_per_step,
    "total_tokens":    total_tokens,
    "loss_curve":      curve,
    "final_loss":      curve[-1]["loss"] if curve else None,
    "initial_loss":    curve[0]["loss"]  if curve else None,
}

with open(out_json, "w") as f:
    json.dump(result, f, indent=2)

print(f"[parse_loss] {len(curve)} log points written to {out_json}")
if curve:
    print(f"[parse_loss]   initial_loss={result['initial_loss']:.4f}  "
          f"final_loss={result['final_loss']:.4f}")
PYEOF
}

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo "==========================================="
echo " Neuron_SP: Scaling Law Experiment Runner"
echo "==========================================="
echo " Repo    : ${NEURON_SP_DIR}"
echo " Out dir : ${SCALING_DIR}"
echo " Batch   : ${BATCH_SIZE}  |  SeqLen: ${SEQ_LEN}"
echo " Models  : 70m (500 steps)  +  1b (200 steps)"
echo "==========================================="
echo ""

# ---------------------------------------------------------------------------
# ① 70M — 500 steps
# ---------------------------------------------------------------------------
MODEL_70M="70m"
STEPS_70M=500
PARAMS_70M=70000000
LOG_70M="${LOG_DIR}/run_70m_$(date +%Y%m%d_%H%M%S).log"
JSON_70M="${SCALING_DIR}/70m_loss_log.json"

echo "[1/3] Training 70M model for ${STEPS_70M} steps..."
echo "      Log: ${LOG_70M}"

run python3 "${PRETRAIN_SCRIPT}" \
    --model-size "${MODEL_70M}" \
    --steps      "${STEPS_70M}" \
    --batch-size "${BATCH_SIZE}" \
    --seq-len    "${SEQ_LEN}" \
    --log-every  "${LOG_EVERY}" \
    --save-every 0 \
    2>&1 | tee "${LOG_70M}"

echo ""
echo "[1/3] Parsing 70M loss log -> ${JSON_70M}"
if [[ "${DRY_RUN}" -eq 0 ]]; then
    parse_and_save_loss_json "${LOG_70M}" "${JSON_70M}" "${MODEL_70M}" "${STEPS_70M}" "${PARAMS_70M}"
else
    echo "  [DRY-RUN] Would parse ${LOG_70M} -> ${JSON_70M}"
fi
echo ""

# ---------------------------------------------------------------------------
# ② 1B — 200 steps
# ---------------------------------------------------------------------------
MODEL_1B="1b"
STEPS_1B=200
PARAMS_1B=1000000000
LOG_1B="${LOG_DIR}/run_1b_$(date +%Y%m%d_%H%M%S).log"
JSON_1B="${SCALING_DIR}/1b_loss_log.json"

echo "[2/3] Training 1B model for ${STEPS_1B} steps..."
echo "      Log: ${LOG_1B}"

run python3 "${PRETRAIN_SCRIPT}" \
    --model-size "${MODEL_1B}" \
    --steps      "${STEPS_1B}" \
    --batch-size "${BATCH_SIZE}" \
    --seq-len    "${SEQ_LEN}" \
    --log-every  "${LOG_EVERY}" \
    --save-every 0 \
    2>&1 | tee "${LOG_1B}"

echo ""
echo "[2/3] Parsing 1B loss log -> ${JSON_1B}"
if [[ "${DRY_RUN}" -eq 0 ]]; then
    parse_and_save_loss_json "${LOG_1B}" "${JSON_1B}" "${MODEL_1B}" "${STEPS_1B}" "${PARAMS_1B}"
else
    echo "  [DRY-RUN] Would parse ${LOG_1B} -> ${JSON_1B}"
fi
echo ""

# ---------------------------------------------------------------------------
# ③ Fit scaling curve and predict 7B
# ---------------------------------------------------------------------------
echo "[3/3] Fitting scaling curve and predicting 7B loss..."
echo "      Script: ${FIT_SCRIPT}"

# fit_scaling_curve.py --predict-7b reads scaling_fit_results.json from the
# same directory and writes scaling_7b_predictions.json.
# If scaling_fit_results.json doesn't exist yet, run --demo first to generate it.
FIT_RESULTS="${SCALING_DIR}/scaling_fit_results.json"

if [[ "${DRY_RUN}" -eq 0 ]]; then
    if [[ ! -f "${FIT_RESULTS}" ]]; then
        echo "  [warning] ${FIT_RESULTS} not found — running --demo to generate initial fit."
        python3 "${FIT_SCRIPT}" \
            --demo \
            --out_dir "${SCALING_DIR}"
    fi

    python3 "${FIT_SCRIPT}" --predict-7b

else
    echo "  [DRY-RUN] Would run: python3 ${FIT_SCRIPT} --predict-7b"
fi

echo ""
echo "==========================================="
echo " Done."
echo " Loss logs:"
echo "   70M  ->  ${JSON_70M}"
echo "   1B   ->  ${JSON_1B}"
echo " 7B prediction:"
echo "   ${SCALING_DIR}/scaling_7b_predictions.json"
echo "==========================================="
