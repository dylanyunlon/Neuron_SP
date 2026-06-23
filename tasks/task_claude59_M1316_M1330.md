# Claude-59: M1316-M1330 — 数据集拉取: The Stack v2 PR/commit

## 任务
完善 The Stack v2 PR/commit 数据集适配。

## 具体工作
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat datasets/bigcode/the_stack_v2/ — 读现有文件
3. cat datasets/bigcode/pull_all_datasets.sh — 读
4. 在 pull_all_datasets.sh 添加 bigcode/the-stack-v2 的下载(streaming模式,仅 commit 相关 split)
5. 在 load_commits.py DATASET_REGISTRY 添加 the-stack-v2: {"name": "the-stack-v2", "source": "GHArchive+Software Heritage", "size": "TBD", "hf_id": "bigcode/the-stack-v2"}
6. 在 DATASETS.md 更新表格

## 铁律
- 只改已有文件,不新建文件
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff
- git push origin main
