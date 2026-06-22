# Metaseq OPT 参考文件 (Meta/Facebook)
上游: github.com/facebookresearch/metaseq (2022)
模型: OPT-175B (Open Pre-trained Transformer)

## 文件说明
- `transformer_lm.py` — OPT 语言模型封装
- `transformer_decoder.py` — Decoder-only Transformer 实现
- `distributed_model.py` — FSDP 分布式模型封装
- `trainer.py` — 训练管理器 (checkpoint, logging, optimization)

## DES-LOC 参考点
- OPT 用 FSDP 训练 175B，与 ZeRO-3 对标
- pre-LayerNorm 架构 (比 GPT-3 的 post-LN 稳定)
- Meta 公开了完整训练日志 (35次重启记录)，异构训练的容错可参考
