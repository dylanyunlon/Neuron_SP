# Claude-47: M1136-M1150 — 数据集拉取: StarCoder commits

## 任务
在 `datasets/bigcode/pull_all_datasets.sh` 中添加 StarCoder commits 数据集的下载逻辑。

## 具体工作
1. `cat datasets/bigcode/pull_all_datasets.sh` 先读
2. 添加 `bigcode/starcoderdata` 的 HuggingFace 下载(仅 commit 子集, git-commits-cleaned split)
3. 在 `datasets/bigcode/DATASETS.md` 中更新数据集注册表，加入 StarCoder commits (32-64GB, HuggingFace 来源)
4. 修改 `datasets/bigcode/load_commits.py` 的 `DATASET_REGISTRY` 字典，添加 starcoderdata 条目

## 铁律
- MODIFY EXISTING FILES ONLY
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
