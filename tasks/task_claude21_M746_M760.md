# Claude-21 任务: M746-M760 — CommitPack 4TB 数据流式加载器

Session: Claude-21 (M746-M760) | Base: latest main

## 目标
为 4TB CommitPack 数据集实现流式加载 (streaming)，避免内存溢出。

## 参考
```bash
cat datasets/bigcode/pull_all_datasets.sh
cat references/data_pipelines/bigcode_commit_filtering/
# HuggingFace streaming API: datasets.load_dataset(..., streaming=True)
```

## 步骤

### 1. 实现 HuggingFace streaming → DeepSpeed DataLoader 桥接
- `datasets.load_dataset("bigcode/commitpack", streaming=True)` 按需加载
- 桥接到 `deepspeed.runtime.data_pipeline` 的 DataSampler
- 支持 resume (断点续传): 记录已消费的 shard index

### 2. 分布式预处理 pipeline
- 每个 GPU 独立拉取不同 shard (按 rank 分配)
- 在 CPU 上做 tokenization (EPYC 128核够用)
- 预取: GPU 训练时异步加载下一批

### 3. 1.5TB CPU RAM 利用
- 用 NUMA-aware 分配: NUMA0 缓存给 GPU0/1/2, NUMA1 给 GPU3/4
- mmap 大文件到 RAM，让 OS 管理

## 交付物
- 修改的文件 + commit push
