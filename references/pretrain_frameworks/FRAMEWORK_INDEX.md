# 2022 年代大厂预训练框架索引

本目录收集了 2022 年前后发布的主要预训练框架/仓库的关键文件，供 DES-LOC 异构训练框架参考。

## 已拉取到本仓库的框架 (key files)

| # | 目录 | 公司/组织 | 模型 | 年份 | 上游仓库 |
|---|------|---------|------|------|---------|
| 1 | `megatron_*` (顶层Megatron-LM) | NVIDIA | GPT-3级训练 | 2019→ | github.com/NVIDIA/Megatron-LM |
| 2 | `gpt_neox/` | EleutherAI | GPT-NeoX-20B, Pythia | 2022 | github.com/EleutherAI/gpt-neox |
| 3 | `metaseq_opt/` | Meta (Facebook) | OPT-175B | 2022 | github.com/facebookresearch/metaseq |
| 4 | `bloom_megatron_ds/` | BigScience/HuggingFace | BLOOM-176B | 2022 | github.com/bigscience-workshop/Megatron-DeepSpeed |

## 需远程参考（未拉取完整代码）

| # | 公司/组织 | 模型/框架 | 年份 | 仓库 |
|---|---------|---------|------|------|
| 5 | Google | PaLM (t5x框架) | 2022 | github.com/google-research/t5x |
| 6 | Salesforce | CodeGen | 2022 | github.com/salesforce/CodeGen |
| 7 | Microsoft | DeepSpeed (ZeRO系列) | 2020→ | github.com/microsoft/DeepSpeed (=本仓库上游) |
| 8 | 智谱/清华 | GLM-130B → ChatGLM | 2022 | github.com/THUDM/GLM-130B |
| 9 | DeepMind | Chinchilla/Gopher (Scaling Laws) | 2022 | github.com/google-deepmind/jax |
| 10 | HPC-AI Tech | ColossalAI | 2021-2022 | github.com/hpcaitech/ColossalAI |

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
