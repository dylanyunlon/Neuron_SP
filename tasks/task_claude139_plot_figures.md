# Claude-139: 生成论文图表

git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP && tree -L 1

## 任务
用 matplotlib 从 desloc_results/ 生成5张论文级图表，保存到 experiments/figures/

1. 读 `experiments/plot_figures.py` 理解现有框架
2. 读 `desloc_results/benchmark_results_*.json` 提取数据
3. 更新 plot_figures.py 生成:
   - fig1_scaling_law.png: loss vs tokens (不同 Kx 配置的曲线对比)
   - fig2_throughput.png: 柱状图 tok/s (DDP vs FSDP vs DES-LOC)
   - fig3_comm_reduction.png: 通信量缩减 bar chart (Kx sweep)
   - fig4_gpu_heatmap.png: 5-GPU内存使用热力图
   - fig5_cache_hit_rate.png: LOC缓存命中率 vs step
4. 图表要求: NeurIPS 风格 (单列宽3.25in, 双列宽6.75in), 字号≥8pt, PDF-safe 颜色
5. 运行: python experiments/plot_figures.py 确认生成成功
6. pip install matplotlib --break-system-packages (如果需要)

## 铁律
- 不开新分支，直接 main
- git config user.email "dogechat@163.com" && git config user.name "dylanyunlon"
- - 用 dispatch prompt 里提供的 GIT_TOKEN 设置 remote
- push 前: git pull --rebase origin main
- commit: git commit --signoff -m "figures: generate NeurIPS plots from benchmark data"
