#!/usr/bin/env bash
# setup_ags1.sh — 在 ags1 上一键配置 Neuron_SP 训练环境
# 用法: bash setup_ags1.sh
set -euo pipefail

echo "=== Neuron_SP Environment Setup for ags1 ==="
echo "$(date)"
echo ""

# 1. Conda env
if ! conda env list | grep -q neuron_sp; then
    echo "[1/5] Creating conda env neuron_sp (Python 3.11)..."
    conda create -y -n neuron_sp python=3.11
else
    echo "[1/5] conda env neuron_sp already exists"
fi

eval "$(conda shell.bash hook)"
conda activate neuron_sp

# 2. PyTorch (CUDA 12.4 — compatible with CUDA 13.0 driver)
echo "[2/5] Installing PyTorch..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Training deps (no deepspeed, no apex)
echo "[3/5] Installing training dependencies..."
pip install -q transformers datasets tokenizers sentencepiece
pip install -q wandb tensorboard tqdm pyyaml

# 4. Verify GPU
echo "[4/5] Verifying GPU setup..."
python3 -c "
import torch
n = torch.cuda.device_count()
print(f'  GPUs detected: {n}')
for i in range(n):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_mem / 1e9
    print(f'  GPU {i}: {name} ({mem:.0f} GB)')
print(f'  CUDA version: {torch.version.cuda}')
print(f'  cuDNN version: {torch.backends.cudnn.version()}')
"

# 5. Verify model import
echo "[5/5] Verifying model import..."
python3 -c "
import sys; sys.path.insert(0, '.')
from models.llama_pretrain import LlamaConfig, LlamaForPreTraining
cfg = LlamaConfig()  # default 7B
print(f'  LlamaForPreTraining: {cfg.hidden_size}h, {cfg.num_layers}L, {cfg.num_heads}H')
m = LlamaForPreTraining(LlamaConfig(hidden_size=64, num_layers=2, num_heads=2, intermediate_size=128))
n = sum(p.numel() for p in m.parameters())
print(f'  Tiny model test: {n/1e3:.0f}K params — OK')
"

echo ""
echo "=== Setup Complete ==="
echo "To train: conda activate neuron_sp && python run_pretrain.py --model-size 70m"
