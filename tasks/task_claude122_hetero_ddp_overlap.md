# Claude-122: hetero_ddp_overlap_grad_fix.py model.cuda() 修复

## Bug
hetero_ddp_overlap_grad_fix.py:909: model = model.cuda() 多卡堆到 cuda:0。

## 修复
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. sed -n '905,915p' deepspeed/runtime/hetero_ddp_overlap_grad_fix.py
3. 改 model.cuda() → model.to(torch.device(f"cuda:{torch.cuda.current_device()}"))
4. 同时检查文件里其他 .cuda() 调用

## 铁律
- 只改 deepspeed/runtime/hetero_ddp_overlap_grad_fix.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff -m "fix: hetero_ddp_overlap model.cuda() → device-aware .to()"
- git push origin main
