# Claude-105: 项目 README 更新

## 任务
更新根目录 `README.md` (如果不存在则创建)，反映 Neuron_SP 的当前状态。

## 具体工作
1. 检查是否有 README.md: `ls README.md`
2. 写/更新 README:
   - 标题: Neuron_SP: DES-LOC Heterogeneous GPU Training Framework
   - 描述: DeepSpeed fork with Decomposed Local SGD for heterogeneous GPU clusters
   - Features: DES-LOC, AutoSP, LOC Cache, 5-tier GPU support, commit-centric pretraining
   - Quick Start: setup_ags1.sh → launch_7b.sh
   - Architecture: 描述 desloc_engine.py, train_three_stage.py, 62 hetero modules
   - Paper: link to FAUST_nips2026/
   - Hardware: A6000×2 + H100 NVL + RTX PRO 6000 Blackwell×2
3. 简洁有力，不超过 200 行

## 铁律
- 可创建/修改 README.md
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
