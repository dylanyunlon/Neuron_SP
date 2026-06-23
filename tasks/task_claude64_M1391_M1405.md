# Claude-64: M1391-M1405 — Wiring: HeteroFP32GradAccumManager 接入 backward

## 任务
在 DesLocEngine 的 backward pass 中接入 HeteroFP32GradAccumManager。

## 具体工作
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat deepspeed/runtime/desloc_engine.py — 找 backward 相关代码
3. cat deepspeed/runtime/hetero_fp32_grad_accum.py | head -100 — 读接口
4. 在 backward 后、optimizer step 前插入 FP32 梯度累积调用
5. 确保三级精度策略(FP16 compute → FP32 accum → BF16 comm)正确连接

## 铁律
- 只改 deepspeed/runtime/desloc_engine.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff
- git push origin main
