# Claude-109: fix torch.autocast 缺少 dtype 参数

## Bug
desloc_engine.py line ~1496: torch.autocast("cuda") 没指定 dtype。
BF16 和 FP16 的 autocast 行为不同。A6000 上用 BF16, H100/Blackwell 也用 BF16。
应该显式指定 dtype=torch.bfloat16。

## 修复
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. grep -n "torch.autocast" deepspeed/runtime/desloc_engine.py
3. 所有 torch.autocast("cuda") → torch.autocast("cuda", dtype=torch.bfloat16)
4. 删除 GradScaler import (line 34) 如果没被使用 — BF16 不需要 loss scaling
5. 搜索是否有 GradScaler() 实例化,如果有删掉 (BF16 不需要)

## 铁律
- 只改 deepspeed/runtime/desloc_engine.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff -m "fix: explicit bf16 dtype in autocast, remove unused GradScaler"
- git push origin main
