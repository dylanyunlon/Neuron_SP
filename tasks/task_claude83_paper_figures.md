# Claude-83: 论文图表生成脚本

## 任务
创建 `experiments/plot_figures.py`，生成论文需要的 matplotlib 图表。

## 具体工作
1. 先 `cat experiments/scaling_law/scaling_7b_predictions.json` 读数据
2. 创建 `experiments/plot_figures.py`，包含:
   - Fig 1: Scaling law curve (loss vs tokens) with 7B prediction overlay
   - Fig 2: Training throughput comparison (DES-LOC vs baseline) across heterogeneous vs homogeneous
   - Fig 3: Communication reduction ratio vs convergence (Kx, Ku, Kv sweep)
   - Fig 4: Per-GPU utilization heatmap (A6000 vs H100 vs Blackwell)
   - Fig 5: Ablation — LOC cache hit rate vs training step
3. 输出到 `experiments/figures/`
4. 使用占位数据（后续用真实实验数据替换）

## 铁律
- 可以创建 experiments/plot_figures.py 和 experiments/figures/ 目录
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
