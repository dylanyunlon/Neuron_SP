# Claude-71: Pull CommitPack dataset + wire into data pipeline

## 任务
完善 CommitPack 4TB 数据集的拉取和预处理流程。

## 步骤
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
3. 读 datasets/bigcode/DATASETS.md 了解数据集注册表
4. 读 datasets/bigcode/load_commits.py 和 pull_all_datasets.sh
5. 完善 pull_all_datasets.sh:  添加 bigcode/commitpack 的 HuggingFace streaming 拉取
6. 在 load_commits.py 中添加 CommitPack 格式解析 (diff + message 字段)
7. 确保 CommitSequencePacker (commit_packing.py) 能处理 CommitPack 格式
8. git add -A && git commit --signoff -m "wire CommitPack 4TB dataset into data pipeline"
9. git push origin main

## 铁律
- 只改 datasets/bigcode/ 下的文件
- 作者: dylanyunlon <dogechat@163.com>
