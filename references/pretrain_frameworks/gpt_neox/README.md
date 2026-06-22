# GPT-NeoX 参考文件 (EleutherAI)
上游: github.com/EleutherAI/gpt-neox (2022)
模型: GPT-NeoX-20B, Pythia suite

## 文件说明
- `train.py` — 训练入口 (DeepSpeed launcher)
- `training.py` — 训练循环核心 (forward/backward/optimizer step)
- `transformer.py` — Transformer 层实现 (含 rotary/alibi 位置编码)
- `gpt2_model.py` — GPT-2/NeoX 模型组装
- `word_embeddings.py` — 词嵌入 + 位置嵌入
- `init_functions.py` — 参数初始化策略

## DES-LOC 参考点
- Rotary embedding 实现可迁移到异构 pipeline 的每个 stage
- DeepSpeed 原生训练循环，不依赖 Megatron 的 mpu
- Pythia 的 data dedup 策略可参考
