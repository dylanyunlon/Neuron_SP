# Claude-15 任务: M656-M670 — 2022 预训练框架 Benchmark 复现

Session: Claude-15 (M656-M670) | Base: latest main

## 目标
在 ags1 异构集群上复现 2022 年代预训练框架的 benchmark，作为 DES-LOC 的基线对比。

## 步骤

### 1. 基线框架选择
从以下选 3 个在异构环境可运行的：
- GPT-NeoX (EleutherAI) — 纯 DeepSpeed ZeRO
- Megatron-DeepSpeed — TP + PP + ZeRO
- ColossalAI — Gemini 自动并行

### 2. 7B 模型配置
- 统一模型架构: LLaMA-7B (32 layers, 4096 hidden, 32 heads)
- 统一数据: 1B tokens from RedPajama sample
- 统一 batch size: global batch 256, seq_len 2048

### 3. 指标采集
- throughput (tokens/sec/GPU)
- peak VRAM per GPU
- communication volume
- time to 1000 steps

## 交付物
- benchmark 脚本 + commit push
- 结果写入 desloc_results/phase5/baselines/
