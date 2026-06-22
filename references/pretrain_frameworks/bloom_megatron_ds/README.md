# BLOOM Megatron-DeepSpeed 参考文件 (BigScience/HuggingFace)
上游: github.com/bigscience-workshop/Megatron-DeepSpeed (2022)
模型: BLOOM-176B (多语言)

## 文件说明
- `pretrain_gpt.py` — BLOOM 预训练入口
- `gpt_model.py` — BLOOM GPT 模型定义 (含 ALiBi)
- `transformer.py` — Transformer block (Megatron-DS 融合版)

## DES-LOC 参考点
- BLOOM-1B1 曾在 8× RTX A6000 上训练 (arXiv:2303.04715)
- ALiBi 位置编码: 线性偏置无需学习，适合异构 pipeline (不存参数)
- Megatron-DS 混合并行: TP + PP + DP 同时使用
