#!/bin/bash
set -e
echo "=== Upgrading torch in walking3: cu118 → cu124 ==="
echo "Before: $(python3 -c 'import torch; print(torch.__version__, torch.version.cuda)')"

# pip 认为版本号相同(2.7.1)不需要升级，必须先卸载
pip uninstall -y torch torchvision torchaudio 2>/dev/null
pip uninstall -y nvidia-cuda-nvrtc-cu11 nvidia-cuda-runtime-cu11 nvidia-cuda-cupti-cu11 \
    nvidia-cudnn-cu11 nvidia-cublas-cu11 nvidia-cufft-cu11 nvidia-curand-cu11 \
    nvidia-cusolver-cu11 nvidia-cusparse-cu11 nvidia-nccl-cu11 nvidia-nvtx-cu11 2>/dev/null || true

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "After: $(python3 -c 'import torch; print(torch.__version__, torch.version.cuda)')"
echo "=== Verifying GPUs ==="
python3 -c "
import torch
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    cc = torch.cuda.get_device_capability(i)
    print(f'  GPU {i}: {name} ({mem:.0f} GB, SM {cc[0]}.{cc[1]})')
print(f'CUDA: {torch.version.cuda}')
"
echo "=== Done. Now run: bash launch_7b_3gpu.sh ==="
