# Claude-49: M1166-M1180 — 数据集拉取: CommitPackFT + The Stack v2

## 任务
添加 CommitPackFT (bigcode/commitpackft, 2GB 高质量子集) 和 The Stack v2 PR/commit (bigcode/the-stack-v2) 数据集。

## 具体工作
1. `cat datasets/bigcode/pull_all_datasets.sh` 先读
2. 添加 commitpackft 和 the-stack-v2 下载条目
3. `cat datasets/bigcode/load_commits.py` 先读，在 DATASET_REGISTRY 添加两个条目
4. `cat datasets/bigcode/the_stack_v2/stackv2_commits.py` 先读，确保和新注册条目兼容
5. 在 `datasets/bigcode/DATASETS.md` 更新

## 铁律
- MODIFY EXISTING FILES ONLY
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
