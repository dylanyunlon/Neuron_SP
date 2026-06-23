# Claude-95: 混合精度策略 — BF16 for H100/Blackwell, FP16 for A6000

## 任务
配置 per-device 精度策略。

## 具体工作
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat run_pretrain.py | grep -A20 "dtype\|precision\|bf16\|fp16" — 读精度配置
3. H100 NVL (SM9.0) 和 Blackwell (SM12.0): 用 BF16 (原生支持)
4. A6000 (SM8.6): BF16 也支持但 FP16 tensor core 更快,需要 loss scaling
5. 确保 FSDP MixedPrecision 配置正确

## 铁律
- 只改已有文件
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff
- git push origin main
