# Claude-5 任务: M506-M520 — NeurIPS 图表生成

Session: Claude-5 (M506-M520) | Base: 待 Claude-4 后确定

## 目标
从实验结果生成 NeurIPS 论文图表 (REAL_GPU_BENCHMARK.py)。

## 图表清单
1. **Figure 1**: Loss vs Step curves (DDP / LocalAdam / DES-LOC 3条线)
2. **Figure 2**: Communication reduction bars (3方法, 标注 reduction ratio)
3. **Figure 3**: Kx sensitivity (x=Kx, y=final_loss, 同步 vs 分解两条线)
4. **Figure 4**: SP scalability (x=SP_size, y=throughput)
5. **Figure 5**: Wall-clock comparison bars
6. **Figure 6**: Convergence bound validation (理论 vs 实测)

## 风格要求
- seaborn whitegrid
- NKI-FA draw_plot.py 风格: bar annotations, 4+ decimal places
- 使用 desloc_parse_nkifa_logfile() 和 desloc_aggregate_experiments() 读数据
- 图表保存到 desloc_results/figures/

## 铁律
- 所有数据必须来自 desloc_results/ 的已提交 JSON
- 不硬编码数字
