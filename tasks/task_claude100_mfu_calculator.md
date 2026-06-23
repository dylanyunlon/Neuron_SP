# Claude-100: MFU 计算器 — 异构集群版

## 任务
创建 `tools/mfu_calculator.py` — 计算异构集群的 Model FLOP Utilization。

## 具体工作
1. 读 `experiments/configs/7b_ags1_desloc.yaml` 获取 GPU 配置
2. 创建 MFU 计算器:
   - 每种 GPU 的理论 BF16 TFLOPS (A6000=39, H100=990, RTX PRO 6000 BW=~250)
   - 从训练 log 计算 achieved tokens/s
   - Per-GPU MFU = achieved_flops / theoretical_peak
   - Aggregate MFU = sum(achieved) / sum(theoretical)
   - 输出 per-device breakdown 表
3. 支持命令行: `python tools/mfu_calculator.py --config experiments/configs/7b_ags1_desloc.yaml --tokens-per-sec 1200`

## 铁律
- 可创建 tools/mfu_calculator.py
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
