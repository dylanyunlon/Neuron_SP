# Claude-92: 异构梯度累积步数配置

## 任务
配置 per-device gradient accumulation 适配不同显存 GPU。

## 具体工作
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat run_pretrain.py | grep -A20 "gradient.accum" — 读现有逻辑
3. cat configs/7b_commitpack.yaml | grep -i accum — 读配置
4. A6000 (48GB): grad_accum=8, H100/Blackwell (96GB): grad_accum=4, 保持 effective batch 相同
5. 确保 global_batch_size = micro_batch * grad_accum * world_size 正确

## 铁律
- 只改已有文件
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff
- git push origin main
