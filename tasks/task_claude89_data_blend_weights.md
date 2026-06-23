# Claude-89: 数据集混合权重配置

## 任务
在 configs/7b_commitpack.yaml 中配置数据集混合比例。

## 具体工作
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat configs/7b_commitpack.yaml — 读现有配置
3. cat datasets/bigcode/DATASETS.md — 读数据集信息
4. 添加 data_blend 配置: CommitPack 70%, StarCoder commits 20%, CommitPackFT 10%
5. 确保 train_three_stage.py 或 run_pretrain.py 能读到 data_blend 配置

## 铁律
- 只改已有文件
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff
- git push origin main
