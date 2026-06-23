# Claude-57: M1286-M1300 — 数据集拉取: CommitPack 4TB 流式下载器

## 任务
在 datasets/bigcode/ 下完善 CommitPack 4TB 数据集的流式拉取。

## 具体工作
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat datasets/bigcode/pull_all_datasets.sh — 先读现有逻辑
3. cat datasets/bigcode/load_commits.py — 读 DATASET_REGISTRY
4. 在 pull_all_datasets.sh 添加 bigcode/commitpack 的 HuggingFace streaming 下载(用 huggingface-cli download --repo-type dataset bigcode/commitpack)
5. 在 load_commits.py 的 DATASET_REGISTRY 添加 commitpack 条目: {"name": "commitpack", "source": "GHArchive+GitHub API", "size": "4TB", "hf_id": "bigcode/commitpack"}
6. 在 DATASETS.md 更新表格加入 CommitPack 行

## 铁律
- 只改已有文件,不新建文件
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff
- git push origin main
