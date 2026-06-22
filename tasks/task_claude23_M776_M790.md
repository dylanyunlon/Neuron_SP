# Claude-23 任务: M776-M790 — ColossalAI Gemini 异构内存管理迁移

Session: Claude-23 (M776-M790) | Base: latest main

## 目标
参考 ColossalAI 的 Gemini 异构内存管理，优化 DES-LOC 在 A6000(49GB)+H100(96GB) 上的参数放置。

## 参考 (远程)
```bash
# 需先 clone ColossalAI 查看
git clone --depth 1 https://github.com/hpcaitech/ColossalAI.git /tmp/colossalai
cat /tmp/colossalai/colossalai/zero/gemini/ -r | head -500
```

## 步骤

### 1. 分析 ColossalAI Gemini 策略
- Gemini: 动态在 GPU/CPU 间搬运参数 chunks
- StatefulTensorMgr: 按访问频率决定 evict/prefetch
- DES-LOC 已有 LOCActivationCache，可与 Gemini 互补

### 2. 实现 HeteroMemoryManager
- 49GB GPU: 参数 + 梯度 + 激活值都紧张 → 激极 offload
- 96GB GPU: 可以缓存更多 optimizer states
- 修改 `deepspeed/runtime/zero/stage3.py` 的 param partition 逻辑
- 添加 print() 诊断: 每个 forward step 的 GPU 内存水位

### 3. 与 ZeRO-Offload 对比
- 基线: DeepSpeed 原生 offload_param + offload_optimizer
- 改进: 基于实际显存差异的 asymmetric offload

## 交付物
- 修改的文件 + commit push
