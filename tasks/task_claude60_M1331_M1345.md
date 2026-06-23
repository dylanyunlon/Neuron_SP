# Claude-60: M1331-M1345 — 数据集拉取: StarCoder commits

## 任务
添加 StarCoder commits (32-64GB) 数据集下载和注册。

## 具体工作
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat datasets/bigcode/pull_all_datasets.sh — 先读
3. cat datasets/bigcode/load_commits.py — 读 DATASET_REGISTRY
4. 在 pull_all_datasets.sh 添加 bigcode/starcoderdata 的 git-commits-cleaned split 下载
5. 在 load_commits.py DATASET_REGISTRY 添加 starcoderdata: {"name": "starcoderdata", "source": "HuggingFace/BigQuery", "size": "32-64GB", "hf_id": "bigcode/starcoderdata"}
6. 在 DATASETS.md 更新表格

## 铁律
- 只改已有文件,不新建文件
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff
- git push origin main
