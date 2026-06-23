# C3: Wire Megatron CommitDataset into train_three_stage.py Stage 2

## 目标
改 `pipeline/train_three_stage.py` 的 Stage 2 (continue_commit)，让它使用 `Megatron-LM/megatron/core/datasets/commit_dataset.py` 里的 `CommitDataset` 和 `build_commit_datasets()`。

## 具体改动
1. 在 import 区添加: `from megatron.core.datasets.commit_dataset import CommitDataset, build_commit_datasets`
2. 在 Stage 2 的数据加载部分，替换现有的 dummy dataloader，调用 `build_commit_datasets()`
3. 确保 tokenizer 使用 `pipeline/unified_tokenizer.py` 的 `build_megatron_tokenizer()`
4. Stage 2 的 data_path 参数指向 CommitPack 格式数据

## 文件
只改 `pipeline/train_three_stage.py`
