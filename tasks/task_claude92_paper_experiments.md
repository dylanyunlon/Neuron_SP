# Claude-92: 论文 Experiments section 框架

## 任务
填充 `FAUST_nips2026/main.tex` 的 Section 5 (Experiments)。

## 具体工作
1. `cat FAUST_nips2026/main.tex` 先读
2. `cat experiments/scaling_law/scaling_7b_predictions.json` 读预测数据
3. 填充:
   - 5.1 Setup: 硬件描述 (ags1 cluster), 数据集 (4TB commits), 模型 (7B LLaMA-style)
   - 5.2 Scaling Law: 引用 scaling_7b_predictions.json 的拟合结果
   - 5.3 7B Pretraining: 训练曲线、MFU、tokens/s（先用占位 \textbf{TBD}）
   - 5.4 Ablation: DES-LOC vs uniform sync, LOC cache hit rate, AutoSP vs manual SP
   - 添加 Table 1: GPU specifications, Table 2: Training config
   - 引用 fig1-fig5 图表

## 铁律
- MODIFY EXISTING FILES ONLY
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
