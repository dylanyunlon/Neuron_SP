# Claude-1 任务: M447-M460 — 服务器 13B 完整训练 + 基线对比

Session: Claude-1 (M447-M460) | Base: commit 4fd5cf44

## 目标
在 ags1 服务器运行 DES-LOC 13B 完整训练, 并与 DDP/LocalAdam 基线对比。

## 步骤

### 1. 环境准备
```bash
cd /data/jiacheng/system/cache/temp/nips2026/Neuron_SP
git pull origin main
# 确认 HEAD = 4fd5cf44 或更新
git log --oneline -3
```

### 2. DES-LOC 13B 训练
```bash
# 使用 3-GPU 异构 (2x A6000 + 1x H100)
bash run_13B_ags1.sh
```
关注诊断:
- DES-LOC comm_reduction_ratio (目标: >3x vs DDP)
- per-step timing (目标: wall-clock 加速)
- convergence: loss curve 不应发散

### 3. DDP Baseline
修改配置 Kx=Ku=Kv=1 (等效 DDP), 重新运行:
```bash
DESLOC_KX=1 DESLOC_KU=1 DESLOC_KV=1 bash run_13B_ags1.sh
```

### 4. LocalAdam Baseline
修改配置 Kx=Ku=Kv=同一值 (非分解):
```bash
DESLOC_KX=8 DESLOC_KU=8 DESLOC_KV=8 bash run_13B_ags1.sh
```

### 5. 结果保存
```bash
mkdir -p desloc_results/phase4
# 将3组实验结果 JSON 放入该目录
git add desloc_results/phase4/
git commit --signoff -m "Claude-1 M447: 13B baseline comparison results"
git push origin main
```

## 判据
- DES-LOC 相比 DDP: comm 降低 >50%, wall-clock 加速 >20%
- DES-LOC 相比 LocalAdam: 收敛质量更好 (final loss 更低)
- 如果 OOM: 记录错误, 尝试 gradient_accumulation_steps 加倍

## 铁律
- 不改代码, 只跑实验和记录结果
- 结果必须有原始日志支撑
- 如实报告, 不夸大
