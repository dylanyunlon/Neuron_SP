# Claude-18 任务: M701-M715 — Metaseq OPT FSDP 策略适配 DES-LOC ZeRO

Session: Claude-18 (M701-M715) | Base: latest main

## 目标
分析 Meta OPT 的 FSDP 分片策略，将其 auto-wrap 逻辑适配到 DES-LOC 的异构 ZeRO-3 分片。

## 参考文件
```bash
cat references/pretrain_frameworks/metaseq_opt/distributed_model.py
cat references/pretrain_frameworks/metaseq_opt/trainer.py | head -200
```

## 步骤

### 1. 提取 OPT 的 FSDP auto-wrap policy
- Metaseq 用 `size_based_auto_wrap_policy` 按参数量自动包裹
- DES-LOC 需要: 按 GPU 显存差异 (49GB vs 96GB) 动态调整包裹粒度

### 2. 修改 deepspeed/runtime/zero/ 下的 stage3 分片器
- 添加 `hetero_shard_ratio` 配置: A6000 分更少参数, H100 分更多
- 参考 OPT 的 `flatten_parameters()` 技巧减少通信碎片
- 添加 print() 诊断: 各 GPU 的实际参数分片量

### 3. 验证
- 3-GPU mock 测试: 两个 "small" rank + 一个 "large" rank
- 梯度一致性检查

## 交付物
- 修改的文件 + commit push
