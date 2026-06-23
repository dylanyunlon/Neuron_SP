# Claude-63: M1376-M1390 — Wiring: integrate_with_deepspeed_engine (GradNorm)

## 任务
在 DesLocEngine 中接入 HeteroGradNormSkipController。

## 具体工作
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat deepspeed/runtime/desloc_engine.py — 找 step() 方法
3. cat deepspeed/runtime/hetero_grad_norm_skip.py | head -100 — 读 integrate_with_deepspeed_engine 签名
4. 在 DesLocEngine.__init__() 中调用 integrate_with_deepspeed_engine(self)
5. 确保 step() 中的 grad clipping 通过 HeteroGradNormSkipController 路由

## 铁律
- 只改 deepspeed/runtime/desloc_engine.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff
- git push origin main
