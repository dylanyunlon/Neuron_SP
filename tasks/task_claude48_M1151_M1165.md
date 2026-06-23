# Claude-48: M1151-M1165 — 数据集拉取: CommitPack 4TB

## 任务
在 `datasets/bigcode/pull_all_datasets.sh` 和 `datasets/bigcode/load_commits.py` 中添加 CommitPack (bigcode/commitpack) 数据集支持。

## 具体工作
1. `cat datasets/bigcode/pull_all_datasets.sh` 先读
2. 添加 `bigcode/commitpack` 流式下载(4TB, 需要 streaming=True)
3. 在 `datasets/bigcode/load_commits.py` 的 `DATASET_REGISTRY` 添加 commitpack 条目，指定 streaming 模式
4. 在 `datasets/bigcode/DATASETS.md` 更新注册表

## 铁律
- MODIFY EXISTING FILES ONLY
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
