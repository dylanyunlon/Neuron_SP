#!/bin/bash
# 在 walking3 conda 环境里升级 torch 到 cu124（兼容系统 CUDA 13.0）
set -e
echo "=== Upgrading torch in walking3 to cu124 ==="
echo "Current: $(python3 -c 'import torch; print(torch.__version__, torch.version.cuda)')"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --upgrade
echo "After: $(python3 -c 'import torch; print(torch.__version__, torch.version.cuda)')"
echo "=== Verifying GPUs ==="
python3 -c "
import torch
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_mem / 1e9
    cc = torch.cuda.get_device_capability(i)
    print(f'  GPU {i}: {name} ({mem:.0f} GB, SM {cc[0]}.{cc[1]})')
print(f'CUDA: {torch.version.cuda}')
"
echo "=== Done. Now run: bash launch_7b_3gpu.sh ==="
