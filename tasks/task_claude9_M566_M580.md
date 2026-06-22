# Claude-9 任务: M566-M580 — Megatron Pipeline Parallel 调度器对标分析

Session: Claude-9 (M566-M580) | Base: latest main

## 目标
对标 Megatron-LM 的 pipeline parallel 调度器，优化 DES-LOC 在异构 GPU 上的 pipeline bubble。

## 步骤

### 1. 分析 Megatron 的调度实现
```bash
cat references/pretrain_frameworks/megatron_pipeline_parallel/schedules.py | head -100
cat references/pretrain_frameworks/megatron_pipeline_parallel/combined_1f1b.py | head -100
```

### 2. 与 DES-LOC 现有实现对比
```bash
cat deepspeed/runtime/pipe/schedule.py
cat deepspeed/runtime/desloc_engine.py | grep -A 20 "class.*Schedule"
```

### 3. 实现异构 pipeline 优化
- 基于 GPU 算力比分配不同 micro-batch 数量
- H100 NVL (快) 处理更多 micro-batch, A6000 (慢) 处理更少
- 减少 pipeline bubble 的理论推导写入注释

## 交付物
- 优化后的 schedule 代码 + commit push
- bubble 率对比数据写入 desloc_results/phase5/
