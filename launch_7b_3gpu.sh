#!/usr/bin/env bash
# launch_7b_3gpu.sh — ags1 3-GPU DES-LOC 7B pretrain (H100 + 2×A6000)
#
# Blackwell RTX PRO 6000 (SM120) excluded: PyTorch 2.7.1+cu118 only supports up to SM90.
# Upgrade to PyTorch cu126+ to enable Blackwell. Until then, train on 3 GPUs:
#   GPU2: H100 NVL (93GB, SM9.0) — primary compute
#   GPU3: A6000   (47GB, SM8.6) — secondary
#   GPU4: A6000   (47GB, SM8.6) — secondary
#
# To enable all 5 GPUs with Blackwell:
#   pip install torch --index-url https://download.pytorch.org/whl/cu126
#   Then use launch_7b.sh (CUDA_VISIBLE_DEVICES=0,1,2,3,4)

set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/7b_pretrain_3gpu_${TIMESTAMP}.log"

# Only H100 + 2×A6000 (skip Blackwell SM120)
export CUDA_VISIBLE_DEVICES=2,3,4
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800000
export OMP_NUM_THREADS=8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256

# Pin CUDA arch list to SM8.6 (A6000) + SM9.0 (H100).
# Without this, nvcc sees all GPUs (including Blackwell SM120) and tries to
# compile for SM120 which PyTorch cu124 doesn't support → silent hang.
export TORCH_CUDA_ARCH_LIST="8.6;9.0"

# ── AutoSP kill-switch ────────────────────────────────────────────────────────
# PCIe-only topology (H100 + 2×A6000, no NVLink): Ulysses SP=3 all-to-all
# collectives deadlock inside model.forward() at step 0.  Force DP-only mode
# until the all-to-all / process-group wiring is fixed for heterogeneous PCIe.
#
# To re-enable SP once the deadlock is resolved, either:
#   export NEURON_SP_DISABLE_AUTOSP=0   (before calling this script), or
#   remove / comment-out the line below.
export NEURON_SP_DISABLE_AUTOSP=${NEURON_SP_DISABLE_AUTOSP:-0}

# ── Optional per-layer forward logging ───────────────────────────────────────
# Set NEURON_SP_LAYER_LOG=1 to emit a log line before/after every transformer
# layer during forward.  Useful for pinpointing the exact layer that hangs
# when SP is re-enabled.  Off by default (adds barrier overhead).
export NEURON_SP_LAYER_LOG=${NEURON_SP_LAYER_LOG:-0}

# ── Pre-build DeepSpeed CPU Adam C++ extension ──────────────────────────────
# A6000 ranks use DeepSpeedCPUAdam which JIT-compiles csrc/adam/cpu_adam.cpp.
# Problem: CPUAdamBuilder inherits from CUDAOpBuilder, so jit_load() passes
# with_cuda=True and clears TORCH_CUDA_ARCH_LIST.  nvcc 13.0 (system) then
# auto-detects ALL GPUs including Blackwell SM120, which PyTorch cu124 can't
# compile for → silent hang.
#
# Fix: compile directly via torch.utils.cpp_extension.load(with_cuda=False).
# cpu_adam is pure C++ (no .cu files) — it never needed nvcc.
echo "Pre-building DeepSpeed CPU Adam extension (CPU-only, no nvcc)..."
python -c "
import os, sys, torch
from torch.utils.cpp_extension import load

# csrc/ lives at the project root, not under deepspeed/
project_root = os.path.dirname(os.path.abspath('launch_7b_3gpu.sh'))
sources = [
    os.path.join(project_root, 'csrc', 'adam', 'cpu_adam.cpp'),
    os.path.join(project_root, 'csrc', 'adam', 'cpu_adam_impl.cpp'),
]
includes = [os.path.join(project_root, 'csrc', 'includes')]

# Verify files exist
for s in sources:
    if not os.path.isfile(s):
        raise FileNotFoundError(f'Source not found: {s}')

# Detect CPU SIMD flags
simd = '-mavx512f -mavx512dq' if 'avx512' in open('/proc/cpuinfo').read() else '-mavx2'
cxx_args = ['-O3', '-fopenmp', '-std=c++17', simd]

print(f'  sources: {[os.path.basename(s) for s in sources]}')
print(f'  SIMD: {simd}')
sys.stdout.flush()

op = load(
    name='cpu_adam',
    sources=sources,
    extra_include_paths=includes,
    extra_cflags=cxx_args,
    extra_ldflags=['-fopenmp'],
    with_cuda=False,
    verbose=True,
)
print(f'  cpu_adam pre-built OK: {type(op)}')
" 2>&1 || echo "  WARNING: cpu_adam pre-build failed; will JIT on first use (may be slow)"

EXTRA_ARGS=("$@")

if [[ " ${EXTRA_ARGS[*]:-} " == *" --dry-run "* ]]; then
    echo "=== DRY RUN (3-GPU: H100 + 2×A6000) ==="
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader -i 2,3,4 2>/dev/null
    EXTRA_ARGS=("${EXTRA_ARGS[@]/--dry-run/}" --steps 3 --log-every 1 --save-every 0)
fi

echo "=== Neuron_SP 7B DES-LOC (3-GPU: H100+2×A6000) ==="
echo "Log: $LOG"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Note: Blackwell GPUs excluded (PyTorch cu118 < SM120). Upgrade to cu126+ for 5-GPU."

# Auto-detect real data
DATA_PATH="data/commitpack_train.npy"
if [ ! -f "$DATA_PATH" ]; then
    echo "⚠ Real data not found at $DATA_PATH — using synthetic. Run: bash prepare_data.sh"
    DATA_PATH=""
fi

DATA_ARGS=()
if [ -n "$DATA_PATH" ]; then
    DATA_ARGS=(--data-path "$DATA_PATH")
    echo "Data: $DATA_PATH"
else
    echo "Data: synthetic"
fi

numactl --interleave=all \
torchrun --nproc_per_node=3 --master_port=29500 \
    run_pretrain.py \
    --model-size 7b \
    --steps 100000 \
    --batch-size 1 \
    --seq-len 2048 \
    --use-desloc \
    --gradient-checkpointing \
    --log-every 10 \
    --save-every 500 \
    --checkpoint-dir checkpoints/7b_3gpu_${TIMESTAMP} \
    "${DATA_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG"