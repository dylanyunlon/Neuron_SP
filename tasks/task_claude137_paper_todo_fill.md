# Claude-137: 填充论文 TODO 占位符

git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP && tree -L 1

## 任务
FAUST_nips2026/main.tex 有12个 \todo{} 占位符需要填真实数字。

1. 读 `tools/mfu_calculator.py` 理解 MFU 计算逻辑
2. 读 `desloc_results/benchmark_results_20260520_092158.json` 等最新3个json，提取: tokens/s, MFU%, peak_memory_gb
3. 读 `benchmarks/mfu_hetero.py` 里的硬件TFLOPS定义
4. 计算填充:
   - Table 1 (throughput): DDP baseline tok/s, FSDP tok/s, Single-H100 tok/s, DES-LOC tok/s, speedup 倍数
   - Table 2 (per-GPU memory): A6000 peak GB, H100 peak GB, Blackwell peak GB, utilization%
   - 用 benchmark json 里的实测数据，缺失的用 mfu_calculator.py 公式推算
5. 替换所有 \todo{XX.X} 为实际数字
6. 同时把 "Replace with measured values" 的注释行删掉

## 铁律
- 不开新分支，直接 main
- git config user.email "dogechat@163.com" && git config user.name "dylanyunlon"
- - 用 dispatch prompt 里提供的 GIT_TOKEN 设置 remote
- push 前: git pull --rebase origin main
- commit: git commit --signoff -m "paper: fill TODO placeholders with computed MFU numbers"
