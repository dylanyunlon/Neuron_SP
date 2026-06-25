# Claude-141: 拟合 scaling law 曲线

git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP

## 任务
用 experiments/convergence_report.json 的数据拟合 Chinchilla scaling law。

1. 读 experiments/scaling_law/fit_scaling_curve.py 和 scaling_fit_results.json
2. 读 experiments/convergence_report.json 提取 loss_trajectories 和 run_details
3. 更新 fit_scaling_curve.py: 从 convergence_report.json 加载数据, 拟合 L(N,D) = E + A/N^α + B/D^β
4. 写入 scaling_fit_results.json: A, B, α, β, E, R², 各模型size的拟合曲线
5. 更新 scaling_7b_predictions.json: 7B 模型在不同 token budget 下的 loss 预测
6. python experiments/scaling_law/fit_scaling_curve.py 验证运行
7. pip install scipy --break-system-packages

## 铁律
- 不开新分支,直接 main。push前 git pull --rebase origin main
- git config user.email "dogechat@163.com" && git config user.name "dylanyunlon"
- commit --signoff -m "scaling: fit Chinchilla law from convergence data"
