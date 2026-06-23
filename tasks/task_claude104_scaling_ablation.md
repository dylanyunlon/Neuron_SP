# Claude-104: Scaling Law + Ablation 实验脚本

## 任务
创建 `experiments/run_ablation.py` — 自动化 DES-LOC ablation 实验。

## 具体工作
1. 读 `experiments/scaling_law/experiment_matrix.yaml`
2. 创建 ablation 脚本:
   - Sweep Kx in {1,2,4,8,16,32}
   - Sweep Ku in {1,2,4,8}
   - Sweep Kv in {4,8,16,32,64}
   - 每个配置跑 1000 steps, 记录 loss + tokens/s + communication_bytes
   - 输出 experiments/ablation_results/
3. 创建 `experiments/plot_ablation.py` 绘制热力图

## 铁律
- 可创建新文件在 experiments/ 下
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
