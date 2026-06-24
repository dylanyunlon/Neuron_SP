#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
# scripts/download_and_tokenize.sh
#
# One-click pipeline: install deps → download bigcode/commitpackft (Python)
# → tokenize with prepare_commits.py → write mmap to data/processed/
#
# Usage:
#   bash scripts/download_and_tokenize.sh [--tokenizer <name>] [--split <split>]
#
# Options:
#   --tokenizer   HuggingFace tokenizer (default: bigcode/starcoderbase)
#   --split       Dataset split to use  (default: train)
#   --output-dir  Output directory      (default: data/processed)
#   -h|--help     Show this help
#
# Outputs (written to data/processed/):
#   commitpackft_python.bin  — uint16 token ids as numpy mmap
#   commitpackft_python.idx  — metadata (total_tokens, num_samples, …)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
TOKENIZER="bigcode/starcoderbase"
SPLIT="train"
OUTPUT_DIR="data/processed"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tokenizer)   TOKENIZER="$2";  shift 2 ;;
        --split)       SPLIT="$2";      shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
BOLD="\033[1m"
GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[0;33m"
RESET="\033[0m"

step() {
    local num="$1"; local total="$2"; local msg="$3"
    local bar_done=$(( num * 20 / total ))
    local bar_left=$(( 20 - bar_done ))
    local bar
    bar="$(printf '%0.s█' $(seq 1 "$bar_done") 2>/dev/null || printf '=%.0s' $(seq 1 "$bar_done"))"
    bar+="$(printf '%0.s░' $(seq 1 "$bar_left") 2>/dev/null || printf '-%.0s' $(seq 1 "$bar_left"))"
    printf "\n${BOLD}${CYAN}[%d/%d]${RESET} ${BOLD}%s${RESET}\n" "$num" "$total" "$msg"
    printf "       ${GREEN}%s${RESET} %d%%\n" "$bar" $(( num * 100 / total ))
}

info()    { printf "       ${CYAN}→${RESET} %s\n"         "$*"; }
success() { printf "       ${GREEN}✔${RESET}  %s\n"        "$*"; }
warn()    { printf "       ${YELLOW}⚠${RESET}  %s\n"        "$*" >&2; }
die()     { printf "\n[ERROR] %s\n" "$*" >&2; exit 1; }

TOTAL_STEPS=4

# ---------------------------------------------------------------------------
# Step 0: Resolve repo root (script lives in <root>/scripts/)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
info "Repo root: ${REPO_ROOT}"
info "Output dir: ${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Step 1: Install Python dependencies
# ---------------------------------------------------------------------------
step 1 "$TOTAL_STEPS" "Installing Python dependencies"

info "pip install datasets transformers numpy tqdm"
pip install --quiet --upgrade datasets transformers numpy tqdm \
    || die "pip install failed. Check your Python / network environment."
success "Dependencies installed."

# ---------------------------------------------------------------------------
# Step 2: Validate that prepare_commits.py exists
# ---------------------------------------------------------------------------
step 2 "$TOTAL_STEPS" "Validating project layout"

PREPARE_SCRIPT="${REPO_ROOT}/data/prepare_commits.py"
[[ -f "${PREPARE_SCRIPT}" ]] \
    || die "data/prepare_commits.py not found at expected path: ${PREPARE_SCRIPT}"
success "data/prepare_commits.py found."

# ---------------------------------------------------------------------------
# Step 3: Pre-flight dataset check (fast metadata fetch, no real download)
# ---------------------------------------------------------------------------
step 3 "$TOTAL_STEPS" "Pre-flight: fetching dataset card for bigcode/commitpackft"

python - <<'PYEOF'
import sys
try:
    from datasets import load_dataset_builder
    builder = load_dataset_builder("bigcode/commitpackft", "python", trust_remote_code=True)
    info = builder.info
    split_info = info.splits or {}
    if split_info:
        for s, si in split_info.items():
            print(f"       → split={s}  num_examples={si.num_examples:,}")
    else:
        print("       → (split metadata unavailable — proceeding anyway)")
except Exception as e:
    print(f"       ⚠  Could not read dataset metadata: {e}", file=sys.stderr)
    print("       ⚠  Proceeding with download regardless.", file=sys.stderr)
PYEOF

success "Pre-flight complete."

# ---------------------------------------------------------------------------
# Step 4: Download + tokenize → mmap .bin
# ---------------------------------------------------------------------------
step 4 "$TOTAL_STEPS" "Downloading & tokenizing CommitPackFT (Python)"

info "tokenizer : ${TOKENIZER}"
info "split     : ${SPLIT}"
info "output    : ${OUTPUT_DIR}/"

mkdir -p "${OUTPUT_DIR}"

# Fallback tokenizer: if starcoderbase is not accessible (no HF token),
# auto-detect and warn, then retry with gpt2.
run_prepare() {
    local tok="$1"
    python "${PREPARE_SCRIPT}" \
        --task commitpackft \
        --tokenizer-name "${tok}" \
        --split "${SPLIT}" \
        --output-dir "${OUTPUT_DIR}"
}

if ! run_prepare "${TOKENIZER}"; then
    warn "Tokenizer '${TOKENIZER}' failed (possibly requires HuggingFace login)."
    warn "Retrying with fallback tokenizer: gpt2"
    TOKENIZER="gpt2"
    run_prepare "${TOKENIZER}" \
        || die "Tokenization failed with fallback tokenizer '${TOKENIZER}' too."
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
printf "${BOLD}${GREEN}══════════════════════════════════════════════════════${RESET}\n"
printf "${BOLD}${GREEN}  ✔  Pipeline complete!${RESET}\n"
printf "${BOLD}${GREEN}══════════════════════════════════════════════════════${RESET}\n"
echo ""
info "Output files in ${OUTPUT_DIR}/:"
ls -lh "${OUTPUT_DIR}/" | grep "commitpackft" | while read -r line; do
    printf "       %s\n" "$line"
done
echo ""
info "Load in Python:"
printf "       ${CYAN}import numpy as np${RESET}\n"
printf "       ${CYAN}tokens = np.memmap('%s/commitpackft_python.bin', dtype='uint16', mode='r')${RESET}\n" \
    "${OUTPUT_DIR}"
echo ""
