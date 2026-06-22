# Claude-16 任务: M671-M685 — NeurIPS 论文 Related Work + Experiment Section

Session: Claude-16 (M671-M685) | Base: latest main

## 目标
撰写 NeurIPS 论文的 Related Work 和 Experiment 框架。

## 步骤

### 1. Related Work 覆盖范围
- Heterogeneous distributed training (FedBuff, SlowMo, DiLoCo)
- Local SGD variants (PostLocalSGD, SlowMo-LocalSGD)
- Sequence parallelism (Megatron-SP, DeepSpeed-Ulysses, Ring-Attention)
- Code data pipelines (The Stack, StarCoder, CommitPack) — 引用 references/ 中的数据

### 2. Experiment Section 框架
- Table 1: 硬件拓扑 (A6000×2 + H100 NVL, 无 NVLink, PCIe-only)
- Table 2: DES-LOC vs DDP vs LocalAdam vs ZeRO-3 (从 desloc_results/ 提取)
- Table 3: Kx/Ku/Kv sweep 消融
- Table 4: 与 2022 框架基线对比 (Claude-15 的结果)
- Figure 规划对接 Claude-5 的图表

### 3. BibTeX
- 确保所有引用有 bibtex entry
- NeurIPS 2026 格式 (neurips_2026.sty)

## 交付物
- .tex 文件更新 + commit push
- bibtex 文件更新
