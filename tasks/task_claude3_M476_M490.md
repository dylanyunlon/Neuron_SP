# Claude-3 任务: M476-M490 — AutoSP 序列并行实验

Session: Claude-3 (M476-M490) | Base: 待 Claude-2 后确定

## 目标
验证 AutoSP 在异构 3-GPU 环境的 A2A 开销和与 DES-LOC 的联合效果。

## 步骤

### 1. SP-only 实验
```bash
for SP_SIZE in 1 2 3; do
    AUTOSP_SIZE=$SP_SIZE bash run_13B_ags1.sh
done
```

### 2. SP + DES-LOC 联合
```bash
AUTOSP_SIZE=2 DESLOC_KX=8 DESLOC_KU=16 DESLOC_KV=32 bash run_13B_ags1.sh
```

### 3. 结果写入 desloc_results/phase4/autosp_experiments.json

## 判据
- SP=2 应处理更长序列而不 OOM
- SP + DES-LOC 联合应同时降低 comm 和支持长序列
