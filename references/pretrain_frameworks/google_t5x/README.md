# Google t5x 参考文件 (Google Research)
上游: github.com/google-research/t5x (2022)
框架: PaLM-540B 的训练框架 (JAX/Flax 生态)

## 文件说明
- `train.py` — 训练入口 (gin 配置驱动)
- `trainer.py` — 训练循环核心 (Flax train_state)
- `train_state.py` — 训练状态管理 (optimizer + params)
- `models.py` — 模型抽象基类 (EncoderDecoder / DecoderOnly)
- `partitioning.py` — JAX 分区策略 (PjitPartitioner)
- `main.py` — 总入口

## DES-LOC 参考点
- JAX 的 pjit 分区: 声明式并行 (model_parallel_submesh)
- PaLM 用的是 JAX 而非 PyTorch, 但分区思想可迁移
- Chinchilla scaling law 实验基于此框架
- partitioning.py 的 mesh 概念 → DES-LOC 的异构 mesh 设计
