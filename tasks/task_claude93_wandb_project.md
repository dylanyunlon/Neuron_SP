# Claude-93: WandB project 配置 + 训练 dashboard

## 任务
完善 wandb 日志配置,添加关键监控指标。

## 具体工作
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat run_pretrain.py | grep -A30 "wandb" — 读现有 wandb 集成
3. 确保 log: loss, lr, grad_norm, tokens_per_sec, MFU, per-GPU memory
4. 添加 per-GPU memory utilization 和 per-tier throughput 的日志
5. 项目名: neuron-sp-7b-commitpack

## 铁律
- 只改 run_pretrain.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff
- git push origin main
