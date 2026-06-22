# Claude-19 任务: M716-M730 — BLOOM ALiBi 位置编码集成

Session: Claude-19 (M716-M730) | Base: latest main

## 目标
将 BLOOM 的 ALiBi (Attention with Linear Biases) 实现从 Megatron-DeepSpeed 迁移到 DES-LOC。

## 参考文件
```bash
cat references/pretrain_frameworks/bloom_megatron_ds/gpt_model.py | grep -B5 -A30 "alibi"
cat references/pretrain_frameworks/bloom_megatron_ds/transformer.py | grep -B5 -A50 "alibi"
```

## 步骤

### 1. 提取 ALiBi 核心逻辑
- BLOOM 的 ALiBi: 每个 head 有固定的 slope (2^(-8/n_heads) 系列)
- 无需学习的参数 → pipeline 分割时不占显存

### 2. 在 DES-LOC 模型定义中添加 ALiBi 选项
- 修改 `deepspeed/pipe_module.py` 或对应的 transformer block
- ALiBi 的 slope 在 pipeline stage 间共享，但 attention mask offset 按 stage 偏移
- 异构设备: ALiBi bias 矩阵统一用 FP32 计算，再 cast 到设备精度

### 3. 与 Rotary (Claude-17) 做 A/B 对比框架
- 配置开关: `position_encoding_type: [rotary, alibi, learned]`

## 交付物
- 修改的文件 + commit push
