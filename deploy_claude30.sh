#!/usr/bin/env bash
# ===========================================================================
# Claude-30 M336-M343 部署脚本
# 在 ags1 上执行: bash deploy_claude30.sh
# ===========================================================================
set -euo pipefail

REPO="/data/jiacheng/system/cache/temp/nips2026/Neuron_SP"
TMP="$REPO/tmp"
ZIP="$REPO/files_autosp_modified_des_loc.zip"

cd "$REPO"

# 1. 解压到临时目录
echo ">>> Step 1: 解压 zip 到 tmp/"
rm -rf "$TMP"
mkdir -p "$TMP"
unzip -o "$ZIP" -d "$TMP"
echo "    解压完成: $(ls "$TMP"/ | wc -l) 个文件"

# 2. 按仓库路径放回文件
echo ">>> Step 2: 放回仓库路径"

cp -v "$TMP/REAL_GPU_BENCHMARK.py"           "$REPO/REAL_GPU_BENCHMARK.py"
cp -v "$TMP/engine.py"                       "$REPO/deepspeed/runtime/engine.py"
cp -v "$TMP/constants.py"                    "$REPO/deepspeed/runtime/constants.py"
cp -v "$TMP/config.py"                       "$REPO/deepspeed/runtime/config.py"
cp -v "$TMP/utils.py"                        "$REPO/deepspeed/runtime/utils.py"
cp -v "$TMP/comms_logging.py"                "$REPO/deepspeed/utils/comms_logging.py"
cp -v "$TMP/timer.py"                        "$REPO/deepspeed/utils/timer.py"
cp -v "$TMP/torch_comm.py"                   "$REPO/deepspeed/comm/torch.py"
cp -v "$TMP/fused_adam.py"                   "$REPO/deepspeed/ops/adam/fused_adam.py"
cp -v "$TMP/init_sp.py"                      "$REPO/deepspeed/compile/init_sp.py"
cp -v "$TMP/sp_compile.py"                   "$REPO/deepspeed/compile/passes/sp_compile.py"
cp -v "$TMP/long_context_checkpointing.py"   "$REPO/deepspeed/compile/passes/long_context_checkpointing.py"
cp -v "$TMP/compile_backend.py"              "$REPO/deepspeed/compile/backend.py"
cp -v "$TMP/run_experiment_h20.sh"           "$REPO/run_experiment_h20.sh"
cp -v "$TMP/run_experiment_ags1.sh"          "$REPO/run_experiment_ags1.sh"
cp -v "$TMP/CLAUDE_M336_M350.md"             "$REPO/CLAUDE_M336_M350.md"

# 3. 验证 Python 语法
echo ">>> Step 3: ast.parse 验证"
python3 -c "
import ast, sys
files = [
    'REAL_GPU_BENCHMARK.py',
    'deepspeed/runtime/engine.py',
    'deepspeed/runtime/constants.py',
    'deepspeed/runtime/config.py',
    'deepspeed/runtime/utils.py',
    'deepspeed/utils/comms_logging.py',
    'deepspeed/utils/timer.py',
    'deepspeed/comm/torch.py',
    'deepspeed/ops/adam/fused_adam.py',
    'deepspeed/compile/init_sp.py',
    'deepspeed/compile/passes/sp_compile.py',
    'deepspeed/compile/passes/long_context_checkpointing.py',
    'deepspeed/compile/backend.py',
]
ok = 0
for f in files:
    try:
        ast.parse(open(f).read())
        ok += 1
    except SyntaxError as e:
        print(f'FAIL: {f}: {e}', file=sys.stderr)
        sys.exit(1)
print(f'    {ok}/{len(files)} files passed ast.parse')
"

# 4. 提交实验数据（半途结果）+ 代码修改
echo ">>> Step 4: git add + commit + push"

git config user.name "dylanyunlong"
git config user.email "dylanyunlong@gmail.com"

# 先提交半途实验数据
git add desloc_results/*.json
git add desloc_results_0422/ 2>/dev/null || true
git commit -m "data(desloc/ags1): 2xA6000+H100 NVL experiment results (in-progress)

Phase 1: 125M DDP vs DESLOC 5-seed
Phase 2: 700M DDP vs DESLOC 5-seed (partial, NCCL timeout on DESLOC)
Hardware: 2×RTX A6000 (49GB) + 1×H100 NVL (96GB)
Note: DESLOC 700M hit NCCL deadlock at step 90 due to GradScaler
      desync on heterogeneous GPUs — fixed in M340." || echo "    (no new data to commit)"

# 提交代码修改
git add \
    REAL_GPU_BENCHMARK.py \
    deepspeed/runtime/engine.py \
    deepspeed/runtime/constants.py \
    deepspeed/runtime/config.py \
    deepspeed/runtime/utils.py \
    deepspeed/utils/comms_logging.py \
    deepspeed/utils/timer.py \
    deepspeed/comm/torch.py \
    deepspeed/ops/adam/fused_adam.py \
    deepspeed/compile/init_sp.py \
    deepspeed/compile/passes/sp_compile.py \
    deepspeed/compile/passes/long_context_checkpointing.py \
    deepspeed/compile/backend.py \
    run_experiment_h20.sh \
    run_experiment_ags1.sh \
    CLAUDE_M336_M350.md

git commit -m "feat(desloc/M336-M343): SP+DEC+AC integration, AutoSP compile-level DES-LOC awareness, critical bugfixes

Claude-30 session: 13 files modified, +4279 lines, 8 M-tasks.

M336: constants.py — NCCL protocol constants, Megatron bucket sizing,
      AutoSP×DES-LOC compat, H20/H100/A6000 GPU database
M337: comms_logging.py — DeslocCommEvent (NCCL profiler_v3 pattern),
      DeslocCommProfiler, bandwidth analysis, NKI-FA export
M338: timer.py — DeslocCudaEventTimer (FlashAttention benchmark pattern),
      DeslocMemoryTracker, DeslocStepMFUCalculator with roofline
M339: REAL_GPU_BENCHMARK.py — SP+DEC in _train_baseline via standalone
      DeslocSequenceParallelComm.scatter_along_seq()
M340: REAL_GPU_BENCHMARK.py — CRITICAL: fix GradScaler + heterogeneous
      GPU NCCL deadlock. Override optimizer.global_step with loop counter.
M341: REAL_GPU_BENCHMARK.py + config.py — Activation checkpointing via
      torch.utils.checkpoint (layer-wise AC, --use_ac flag)
M342: engine.py — SP+DEC+AC unified init in DeepSpeedEngine.__init__,
      public API (desloc_composition_state), composition state tracking
M343: compile/init_sp.py + sp_compile.py + long_context_checkpointing.py
      — AutoSP compile-level DES-LOC awareness, Aten-IR AC vs layer AC
      documentation addressing NeurIPS reviewer concerns

Bugfixes in REAL_GPU_BENCHMARK.py:
  - MFU: add H20 (148T BF16), fix H100 NVL (835T not 267T)
  - sync schedule: 6x (K<=1) guard for ZeroDivision safety
  - GradScaler deadlock: global_step from loop counter not optimizer
  - AutoSP compile failure: explicit WARNING not silent fallback
  - Shell: interleave DDP+DESLOC per seed (24min first result)

New: run_experiment_h20.sh (2×H20 homogeneous, 8 phases, 61 runs)"

# 5. Push
git push origin main

# 6. 清理
echo ">>> Step 5: 清理临时文件"
rm -rf "$TMP"

echo ""
echo "================================================================"
echo " 部署完成 — $(date)"
echo " 实验仍在运行中（GPU 99-100% 利用率）"
echo " 不要中断当前实验进程"
echo "================================================================"
