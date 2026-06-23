#!/usr/bin/env bash
# setup_ags1.sh — 在 ags1 上一键配置 Neuron_SP 训练环境
# 用法: bash setup_ags1.sh
set -euo pipefail

echo "=== Neuron_SP Environment Setup for ags1 ==="
echo "$(date)"
echo ""

# 1. Conda env
if ! conda env list | grep -q neuron_sp; then
    echo "[1/7] Creating conda env neuron_sp (Python 3.11)..."
    conda create -y -n neuron_sp python=3.11
else
    echo "[1/7] conda env neuron_sp already exists"
fi

eval "$(conda shell.bash hook)"
conda activate neuron_sp

# 2. PyTorch (CUDA 12.4 — compatible with CUDA 13.0 driver)
echo "[2/7] Installing PyTorch..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Training deps
echo "[3/7] Installing training dependencies..."
pip install -q transformers datasets tokenizers sentencepiece
pip install -q wandb tensorboard tqdm pyyaml
pip install -q deepspeed
pip install -q numpy scipy ninja packaging
pip install -q flash-attn --no-build-isolation 2>/dev/null || echo "  (flash-attn install skipped — may need manual build for SM 12.0)"

# 4. Verify GPU and CUDA compatibility
echo "[4/7] Verifying GPU setup and CUDA compatibility..."
python3 -c "
import torch
n = torch.cuda.device_count()
print(f'  GPUs detected: {n}')
cuda_ver = torch.version.cuda
print(f'  CUDA version: {cuda_ver}')
print(f'  cuDNN version: {torch.backends.cudnn.version()}')
has_blackwell = False
for i in range(n):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_mem / 1e9
    cc = torch.cuda.get_device_capability(i)
    print(f'  GPU {i}: {name} ({mem:.0f} GB, SM {cc[0]}.{cc[1]})')
    if cc[0] >= 12:
        has_blackwell = True
if has_blackwell:
    major, minor = (int(x) for x in cuda_ver.split('.')[:2])
    if major < 13:
        print(f'  WARNING: Blackwell SM 12.0 detected but CUDA {cuda_ver} < 13.0')
        print(f'  Blackwell GPUs need CUDA >= 13.0 for native SM 12.0 support.')
        print(f'  Current toolkit may fall back to PTX JIT — expect slower first launch.')
    else:
        print(f'  CUDA {cuda_ver} supports Blackwell SM 12.0 natively — OK')
"

# 5. Verify model import
echo "[5/7] Verifying model import..."
python3 -c "
import sys; sys.path.insert(0, '.')
from models.llama_pretrain import LlamaConfig, LlamaForPreTraining
cfg = LlamaConfig()  # default 7B
print(f'  LlamaForPreTraining: {cfg.hidden_size}h, {cfg.num_layers}L, {cfg.num_heads}H')
m = LlamaForPreTraining(LlamaConfig(hidden_size=64, num_layers=2, num_heads=2, intermediate_size=128))
n = sum(p.numel() for p in m.parameters())
print(f'  Tiny model test: {n/1e3:.0f}K params — OK')
"

# 6. Verify DeepSpeed installation
echo "[6/7] Verifying DeepSpeed..."
python3 -c "
import deepspeed
print(f'  DeepSpeed version: {deepspeed.__version__}')
print(f'  DeepSpeed ops: {deepspeed.ops.__path__}')
" 2>/dev/null || echo "  WARNING: DeepSpeed import failed — run 'pip install deepspeed' manually"

# 7. Dataset preprocessing
echo "[7/7] Preparing dataset (SlimPajama tokenization)..."
DATA_DIR="data/slimpajama_bin"
if [[ -f "${DATA_DIR}/train.bin" ]]; then
    echo "  Tokenized dataset already exists at ${DATA_DIR}/train.bin — skipping"
else
    mkdir -p "$DATA_DIR"
    echo "  Tokenizing SlimPajama subset (100k samples) with LLaMA SentencePiece tokenizer..."
    python3 data/prepare_commits.py \
        --output-dir "$DATA_DIR" \
        --num-samples 100000 \
        --tokenizer-name meta-llama/Llama-2-7b-hf \
        2>&1 || echo "  Dataset preprocessing deferred — run manually if HF token is needed"
fi

echo ""
echo "=== Setup Complete ==="
echo "To train: conda activate neuron_sp && bash launch_7b.sh --dry-run"
