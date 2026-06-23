# Claude-82: NeurIPS 2026 论文骨架

## 任务
在 `FAUST_nips2026/` 中创建论文 LaTeX 骨架。

## 具体工作
1. 创建 `FAUST_nips2026/main.tex` — NeurIPS 2026 格式
2. 标题: "DES-LOC: Decomposed Local SGD for Heterogeneous GPU Clusters with Automatic Sequence Parallelism"
3. 作者: Jiacheng Yuan (对应 dylanyunlon)
4. Section 结构:
   - Abstract
   - 1. Introduction
   - 2. Related Work (Local SGD, Heterogeneous Training, Sequence Parallelism)
   - 3. Method: DES-LOC (3.1 Decomposed Synchronization, 3.2 LOC Cache, 3.3 AutoSP Integration)
   - 4. System Design (4.1 Heterogeneous GPU Tiers, 4.2 PCIe-aware Communication, 4.3 MIMO Training Loop)
   - 5. Experiments (5.1 Setup, 5.2 Scaling Law, 5.3 7B Pretraining, 5.4 Ablation)
   - 6. Conclusion
   - References
5. 创建 `FAUST_nips2026/neurips_2026.sty` (标准 NeurIPS 样式)
6. 创建 `FAUST_nips2026/Makefile` (`make pdf`)

## 铁律
- 可以创建新文件（在 FAUST_nips2026/ 目录下）
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
