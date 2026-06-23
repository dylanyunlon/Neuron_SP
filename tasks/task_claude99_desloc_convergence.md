# Claude-99: DES-LOC 收敛性分析模块

## 任务
创建 `experiments/convergence_analysis.py` — 分析训练 log 并验证 DES-LOC 收敛性。

## 具体工作
1. 读 `experiments/scaling_law/scaling_7b_predictions.json` 了解预期 loss
2. 创建脚本，从训练 log 中:
   - 提取 loss 曲线并 fit power law
   - 计算 effective Chinchilla ratio
   - 比较 DES-LOC vs baseline 收敛速度
   - 输出 convergence_report.json
3. 支持 `--log-dir` 和 `--output` 参数

## 铁律
- 可创建 experiments/convergence_analysis.py
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
