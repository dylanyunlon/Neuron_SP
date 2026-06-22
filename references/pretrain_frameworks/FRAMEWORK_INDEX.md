# 2022 年代大厂预训练框架索引

本目录收集了 2022 年前后发布的主要预训练框架/仓库的关键文件，供 DES-LOC 异构训练框架参考。

## 已拉取到本仓库的框架 (key files)

| # | 目录 | 公司/组织 | 模型 | 年份 | 上游仓库 |
|---|------|---------|------|------|---------|
| 1 | `megatron_*` (顶层Megatron-LM) | NVIDIA | GPT-3级训练 | 2019→ | github.com/NVIDIA/Megatron-LM |
| 2 | `gpt_neox/` | EleutherAI | GPT-NeoX-20B, Pythia | 2022 | github.com/EleutherAI/gpt-neox |
| 3 | `metaseq_opt/` | Meta (Facebook) | OPT-175B | 2022 | github.com/facebookresearch/metaseq |
| 4 | `bloom_megatron_ds/` | BigScience/HuggingFace | BLOOM-176B | 2022 | github.com/bigscience-workshop/Megatron-DeepSpeed |

## 已拉取 key files (Phase 6 补充)

| # | 目录 | 公司/组织 | 模型/框架 | 年份 | 上游仓库 |
|---|------|---------|---------|------|---------|
| 5 | `google_t5x/` | Google | PaLM (t5x框架) | 2022 | github.com/google-research/t5x |
| 6 | `salesforce_codegen/` | Salesforce | CodeGen-16B | 2022 | github.com/salesforce/CodeGen |
| 7 | (本仓库自身) | Microsoft | DeepSpeed (ZeRO系列) | 2020→ | github.com/microsoft/DeepSpeed |
| 8 | `glm130b/` | 智谱/清华 | GLM-130B → ChatGLM | 2022 | github.com/THUDM/GLM-130B |
| 9 | (参考 t5x partitioning) | DeepMind | Chinchilla Scaling Laws | 2022 | (论文驱动，无独立代码) |
| 10 | `colossalai/` | HPC-AI Tech | ColossalAI Gemini | 2021-2022 | github.com/hpcaitech/ColossalAI |

## 对 DES-LOC 的参考价值

### GPT-NeoX (EleutherAI)
- `gpt_neox/transformer.py`: 手写 Attention + MLP，无 Megatron 的 TP/PP 耦合，适合做异构 baseline
- `gpt_neox/training.py`: 训练循环简洁，DeepSpeed 原生集成，DES-LOC 可直接参考

### Metaseq OPT (Meta)
- `metaseq_opt/distributed_model.py`: FSDP 封装，与 DeepSpeed ZeRO-3 对标
- `metaseq_opt/transformer_decoder.py`: OPT 175B 的 decoder 实现，pre-norm 风格

### BLOOM Megatron-DeepSpeed (BigScience)
- `bloom_megatron_ds/pretrain_gpt.py`: BLOOM 训练入口，ALiBi 位置编码
- `bloom_megatron_ds/transformer.py`: Megatron-DS 融合的 transformer，含 deepspeed 优化器钩子
- **关键参考**: 在 8× A6000 上训练过 BLOOM-1B1 (见 arXiv:2303.04715)

## 数据集对照

详见 `../../datasets/bigcode/DATASETS.md`

### ColossalAI Gemini (HPC-AI Tech)
- `colossalai/gemini_ddp.py`: GPU↔CPU 参数搬运，按 chunk 动态管理
- `colossalai/gemini_mgr.py`: StatefulTensorMgr — 访问模式驱动的内存调度
- **关键参考**: 异构显存 (49GB vs 96GB) 的 asymmetric chunk allocation

### Google t5x (Google Research / PaLM)
- `google_t5x/partitioning.py`: PjitPartitioner — JAX 声明式并行分区
- `google_t5x/trainer.py`: Flax 训练循环，learning rate schedule
- **关键参考**: PaLM 的 mesh 概念可迁移到 DES-LOC 异构 mesh

### GLM-130B (智谱AI / 清华)
- `glm130b/quantization_layers.py`: INT4/INT8 量化层
- `glm130b/configs/`: 含 V100 配置 (A6000 算力近似 V100×2)
- **关键参考**: bidirectional + causal 混合训练目标

### Salesforce CodeGen
- `salesforce_codegen/train_deepspeed.py`: **直接用 DeepSpeed 训练**! 可对标
- `salesforce_codegen/modeling_codegen.py`: GPT-J 变体 (rotary embedding)
- `salesforce_codegen/mtpb_*.py`: 多轮编程评估基准
- **关键参考**: 代码预训练 NL+PL 数据混合策略
