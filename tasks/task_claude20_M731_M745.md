# Claude-20 任务: M731-M745 — StarCoder Commit 数据 Sequence Packing 优化

Session: Claude-20 (M731-M745) | Base: latest main

## 目标
优化 BigCode StarCoder commit 数据的 sequence packing，适配异构 batch 分配。

## 参考
```bash
cat datasets/bigcode/DATASETS.md
cat datasets/bigcode/load_commits.py
cat references/data_pipelines/bigcode_commit_filtering/
```

## 步骤

### 1. 实现 commit-diff-aware packing
- StarCoder commit 格式: `<commit_before>old<commit_msg>msg<commit_after>new`
- Packing 时不能跨 commit 边界切割
- 短 commit (≤256 tokens) 合并到同一 sequence
- 长 commit (>2048 tokens) 做 sliding window

### 2. 异构 batch 分配
- H100 (96GB) 的 micro-batch 是 A6000 (49GB) 的 ~2x
- `hetero_batch_sampler`: 按 GPU 显存比例分配不同数量的 packed sequences
- 添加 print() 诊断: 每个 rank 的实际 token 数和 padding ratio

### 3. 验证
- 用 CommitPackFT (2GB) 的 Python subset 测试
- padding ratio < 5% 目标

## 交付物
- 修改的文件 + commit push
- packing 效率统计写入 desloc_results/phase6/
