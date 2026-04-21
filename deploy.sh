#!/bin/bash
# 部署脚本 — 在服务器上解压并覆盖
# 用法: cd /data/jiacheng/system/cache/temp/nips2026/Neuron_SP && bash deploy.sh
set -e

echo "[DEPLOY] Setting git author..."
git config user.name "dylanyunlong"
git config user.email "dylanyunlong@gmail.com"

echo "[DEPLOY] Backing up originals..."
for f in deepspeed/runtime/engine.py REAL_GPU_BENCHMARK.py; do
    cp "$f" "${f}.bak.$(date +%s)" 2>/dev/null || true
done

echo "[DEPLOY] Overwriting files..."
cp -v engine.py deepspeed/runtime/engine.py
cp -v REAL_GPU_BENCHMARK.py .
cp -v run_all_v2.sh .
cp -v CLAUDE_M317_M331.md .
cp -v CLAUDE23_SKILL_M318_M331.sh .
chmod +x run_all_v2.sh

echo "[DEPLOY] Verifying Python syntax..."
python3 -c "import ast; ast.parse(open('deepspeed/runtime/engine.py').read()); print('engine.py OK')"
python3 -c "import ast; ast.parse(open('REAL_GPU_BENCHMARK.py').read()); print('REAL_GPU_BENCHMARK.py OK')"

echo "[DEPLOY] Done."
echo ""
echo "Git commands:"
echo '  git add -A'
echo '  git commit --author="dylanyunlong <dylanyunlong@gmail.com>" -m "feat(desloc/M317): engine.py — Megatron-style bucket AllReduce, Kx-gated async comm, 3-tier momentum sync, NKI-FA logging. REAL_GPU_BENCHMARK.py — fix SyntheticDataset/sync_counts/tokens_per_sec/MFU. 15 infra repos surveyed."'
echo '  git push origin main'
echo ""
echo "To run experiments:"
echo "  ./run_all_v2.sh"
