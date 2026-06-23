# C6: Wire The Stack v2 PR/commit adapter into data pipeline

## 目标
改 `datasets/bigcode/the_stack_v2/stackv2_commits.py`，接入 `Megatron-LM/megatron/core/datasets/` 的索引格式，让 The Stack v2 的 PR commit 数据能被 Megatron 的 data pipeline 读取。

## 具体改动
1. 在 `stackv2_commits.py` 中 import `megatron_indexed.py` 的写入器
2. 添加 `convert_to_megatron_indexed()` 函数，将 HuggingFace streaming dataset 转为 Megatron .bin/.idx 格式
3. 在 `datasets/bigcode/the_stack_v2/README.md` 中记录转换流程

## 文件
改 `datasets/bigcode/the_stack_v2/stackv2_commits.py` 和 `datasets/bigcode/the_stack_v2/README.md`
