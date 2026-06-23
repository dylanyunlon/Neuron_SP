# Claude-108: fix 3个hetero模块里的 .cuda() 无device参数

## Bug
- hetero_ddp_overlap_grad_fix.py:909: model = model.cuda() → 多卡时堆到cuda:0
- hetero_mtp_grad_clipper.py:461: total_sq_tensor = total_sq.cuda() → 同上
- hetero_rl_optimizer_offload.py 里有 .cuda() 在注释里,确认是否还有在代码里的

## 修复
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. 对每个文件: 把 .cuda() 改为 .to(torch.device(f"cuda:{torch.cuda.current_device()}"))
3. 或者如果有 engine/device 参数可用,用那个

## 铁律
- 只改这3个 hetero_*.py 文件
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff -m "fix: replace bare .cuda() with device-aware .to() in hetero modules"
- git push origin main
