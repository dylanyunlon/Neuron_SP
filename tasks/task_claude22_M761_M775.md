# Claude-22 任务: M761-M775 — The Stack v2 PR/Commit 子集适配

Session: Claude-22 (M761-M775) | Base: latest main

## 目标
将 BigCode The Stack v2 的 PR/commit 子集适配为 DES-LOC 预训练语料格式。

## 参考
```bash
cat references/data_pipelines/stackv2_README.md
cat references/data_pipelines/stackv2_pr_commits/
cat datasets/bigcode/DATASETS.md
```

## 步骤

### 1. 数据格式标准化
- The Stack v2 commit 格式与 StarCoder/CommitPack 不同
- 统一为: `<|diff_start|> <|file_path|> path <|old|> code <|new|> code <|msg|> message <|diff_end|>`
- 添加 language tag: `<|lang|>python` / `<|lang|>cpp` 等

### 2. 去重与过滤
- 按 `directory_id` hash 去重 (Stack v2 的去重策略)
- 过滤: max 100K chars, 非 merge commit, 有意义的 diff (>10 changed lines)
- 参考 Stack v2 论文的 PII 去除 pipeline

### 3. 生成 Megatron-style indexed dataset
- 输出 `.bin` + `.idx` 格式 (兼容 Megatron 的 IndexedDataset)
- 或 DeepSpeed 的 data_efficiency 格式

## 交付物
- 修改的文件 + commit push
