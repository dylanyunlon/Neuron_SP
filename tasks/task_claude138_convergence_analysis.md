# Claude-138: 收敛分析 + convergence_report.json

git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP && tree -L 1

## 任务
处理 desloc_results/ 下58个 benchmark json，生成完整收敛分析报告。

1. 读 `experiments/convergence_analysis.py` 理解现有框架
2. 遍历所有 `desloc_results/benchmark_results_*.json`，按 config 分组:
   - 提取: model_size, Kx/Ku/Kv, final_loss, avg_loss, tokens_per_second, mfu, peak_memory
3. 更新 `experiments/convergence_report.json`:
   - best_config: 哪组 Kx/Ku/Kv 达到最低 final_loss
   - throughput_comparison: DES-LOC vs baseline (Kx=Ku=Kv=1)
   - comm_reduction: 计算通信量缩减比例 = 1 - 1/(Kx*Ku*Kv) 的加权版本
   - loss_trajectories: 按配置分组的 loss 曲线数据
4. 更新 `experiments/convergence_analysis.py` 让它能独立运行 `python experiments/convergence_analysis.py` 生成报告
5. 验证: python -c "import json; d=json.load(open('experiments/convergence_report.json')); print(len(d))"

## 铁律
- 不开新分支，直接 main
- git config user.email "dogechat@163.com" && git config user.name "dylanyunlon"
- - 用 dispatch prompt 里提供的 GIT_TOKEN 设置 remote
- push 前: git pull --rebase origin main
- commit: git commit --signoff -m "analysis: convergence report from 58 benchmark runs"
