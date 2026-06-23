# Claude-123: hetero_mtp_grad_clipper.py total_sq.cuda() 修复

## Bug
hetero_mtp_grad_clipper.py:461: total_sq_tensor = total_sq.cuda() 多卡堆到 cuda:0。

## 修复
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. sed -n '457,465p' deepspeed/runtime/hetero_mtp_grad_clipper.py
3. 改 total_sq.cuda() → total_sq.to(device=torch.device(f"cuda:{torch.cuda.current_device()}"))
4. 检查文件里其他 .cuda() 调用

## 铁律
- 只改 deepspeed/runtime/hetero_mtp_grad_clipper.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff -m "fix: hetero_mtp_grad_clipper .cuda() → device-aware"
- git push origin main
