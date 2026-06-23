# Claude-125: A6000 跳过 FP32 accum 导致梯度同步不一致

## Bug
A6000 (48GB) 跳过 FP32GradAccumManager, H100/Blackwell 不跳。
all_reduce 时两边的梯度精度不同 (FP32 vs BF16), 可能导致数值不一致。
需要确保 all_reduce 前所有 rank 的梯度是同一精度。

## 修复
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. grep -n "fp32_grad_manager" deepspeed/runtime/desloc_engine.py | head -20
3. 在 all_reduce 前, 如果 fp32_grad_manager is None (A6000):
   - 梯度已经是 BF16, 不需要额外处理
   - 但 H100/Blackwell 的梯度经过 FP32 累积后会被 cast 回 BF16 再 all_reduce
4. 确认: after_backward() 里做了 FP32→BF16 cast 后才 all_reduce
5. 如果没有, 在 all_reduce 前加显式 cast: param.grad = param.grad.to(torch.bfloat16)

## 铁律
- 只改 deepspeed/runtime/desloc_engine.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff -m "fix: ensure gradient dtype consistency before all_reduce across hetero GPUs"
- git push origin main
