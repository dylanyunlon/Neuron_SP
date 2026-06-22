# ColossalAI Gemini 参考文件 (HPC-AI Tech)
上游: github.com/hpcaitech/ColossalAI (2021-2022→)
核心: Gemini 异构内存管理 + ZeRO-style 低级优化器

## 文件说明
- `gemini_ddp.py` — Gemini DDP 封装 (GPU/CPU 自动参数搬运)
- `gemini_mgr.py` — Gemini 内存管理器 (StatefulTensorMgr)
- `gemini_optimizer.py` — Gemini 优化器 (梯度分片 + offload)
- `low_level_optim.py` — 底层 ZeRO 优化器实现
- `bucket_store.py` — 梯度 bucket 存储管理

## DES-LOC 参考点
- Gemini 的 chunk 机制: 将参数按 chunk 在 GPU↔CPU 间移动
- StatefulTensorMgr: 按访问模式 (COMPUTE/HOLD/FREE) 动态管理
- 可与 DES-LOC 的 LOCActivationCache 互补
- 异构显存 (49GB vs 96GB) 的 asymmetric chunk 分配策略
