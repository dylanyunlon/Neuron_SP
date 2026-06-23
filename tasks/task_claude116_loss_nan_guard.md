# Claude-116: loss NaN/Inf 检测和安全恢复

## Bug
异构集群精度不统一 (A6000 skips FP32 accum), 可能产生 NaN gradient。
NaN 一旦进入 optimizer state 就污染整个训练, 必须检测并跳过。

## 修复
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat deepspeed/runtime/desloc_engine.py | grep -n "step_loss\|loss.item\|backward"
3. 在 train() 的 backward 后、optimizer.step() 前加:
   if torch.isnan(loss) or torch.isinf(loss):
       logger.warning("NaN/Inf loss at step %d, skipping update", step)
       optimizer.zero_grad()
       continue
4. 对 grad_norm 也检测: if torch.isnan(gnorm) or gnorm > 1000: skip
5. 连续 10 次 NaN → 自动 reload 上一个 checkpoint

## 铁律
- 只改 deepspeed/runtime/desloc_engine.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff -m "fix: NaN/Inf loss detection and skip in heterogeneous train loop"
- git push origin main
