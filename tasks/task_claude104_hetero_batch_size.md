# Claude-104: per-GPU micro batch size 自动计算

## 任务
根据 GPU 显存自动设置不同的 micro_batch_size。

## 具体工作
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat run_pretrain.py | grep -A20 "micro_batch\|batch_size" — 读现有逻辑
3. 添加自动计算:
   - A6000 (48GB): micro_batch=2
   - H100/Blackwell (96GB): micro_batch=4
   - 用 torch.cuda.get_device_properties(device).total_mem 检测
4. 确保 effective_batch_size = sum(micro_batch_i * grad_accum_i) 在所有 GPU 上一致

## 铁律
- 只改 run_pretrain.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff
- git push origin main
