# C4: Wire CommitPackStreamingDataset for 4TB CommitPack streaming

## 目标
改 `datasets/bigcode/load_commits.py`，接入 `Megatron-LM/megatron/core/datasets/commitpack_streaming_dataset.py` 的 `CommitPackStreamingDataset`，实现 4TB CommitPack 流式加载。

## 具体改动
1. 在 `load_commits.py` 中 import `CommitPackStreamingDataset, CommitPackStreamingConfig, build_commitpack_dataloader`
2. 添加一个 `build_streaming_dataloader()` 函数，配置 CommitPackStreamingConfig（指定 HuggingFace dataset name "bigcode/commitpack"，token reader 用 mmap 模式）
3. 在 `pull_all_datasets.sh` 中添加 huggingface-cli download 命令拉取 bigcode/commitpack 的 metadata

## 文件
改 `datasets/bigcode/load_commits.py` 和 `datasets/bigcode/pull_all_datasets.sh`
