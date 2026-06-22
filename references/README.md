# References — 预训练 Pipeline & 框架参考代码

本目录包含从业界开源项目中提取的关键参考实现，用于 DES-LOC 异构训练框架的设计参考。

## data_pipelines/

### bigcode_commit_filtering/
来源: `bigcode-project/bigcode-dataset` (MIT License)
- **filtering_git_commits.ipynb**: StarCoder 的 GitHub commit 数据清洗 pipeline
- 完整流程: BigQuery → diff 下载 → 质量过滤 → opt-out → 窗口采样

### bigcode_dedup/
来源: `bigcode-project/bigcode-dataset`
- MinHash LSH 近似去重 pipeline

### bigcode_pii/
来源: `bigcode-project/bigcode-dataset`
- PII（个人信息）检测和匿名化

### stackv2_pr_commits/
来源: `bigcode-project/the-stack-v2` (Apache 2.0)
- The Stack v2 的 PR + commit pairs 处理 pipeline
- 使用 60 节点 Ray 集群处理 GHArchive 数据
- 包含许可证过滤 + opt-out + commit pair 提取

## pretrain_frameworks/

### megatron_datasets/
来源: `NVIDIA/Megatron-LM` (MIT License)
- GPT 数据集加载器 + BlendedMegatronDataset

### megatron_pipeline_parallel/
来源: `NVIDIA/Megatron-LM`
- Pipeline 并行调度器（1F1B, interleaved）
- P2P 通信原语

### megatron_tensor_parallel/
来源: `NVIDIA/Megatron-LM`
- 张量并行分片策略

## 2022 年代大厂预训练仓库清单（完整 URL）

| # | 仓库 | 公司 |
|---|------|------|
| 1 | github.com/NVIDIA/Megatron-LM | NVIDIA |
| 2 | github.com/facebookresearch/metaseq | Meta (OPT-175B) |
| 3 | github.com/bigscience-workshop/bigscience | BigScience (BLOOM-176B) |
| 4 | github.com/EleutherAI/gpt-neox | EleutherAI (GPT-NeoX-20B) |
| 5 | github.com/THUDM/GLM-130B | 清华 (GLM-130B) |
| 6 | github.com/google-research/t5x | Google (PaLM via t5x/Pax) |
| 7 | github.com/facebookresearch/galactica | Meta FAIR (Galactica-120B) |
| 8 | github.com/THUDM/CodeGeeX | 清华 (CodeGeeX-13B) |
| 9 | github.com/amazon-science/alexa-teacher-models | Amazon (AlexaTM-20B) |
| 10 | github.com/yandex/YaLM-100B | Yandex (YaLM-100B) |
