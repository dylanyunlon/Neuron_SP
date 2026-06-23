# Claude-105: checkpoint 定期备份到远端

## 任务
在训练循环中添加 checkpoint 自动备份功能。

## 具体工作
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat run_pretrain.py | grep -A20 "save\|checkpoint" — 读现有逻辑
3. 添加:
   - 每 N 步保存 checkpoint 到 --checkpoint-dir
   - 保留最近 3 个 checkpoint, 删除旧的 (节省1.5TB磁盘)
   - 可选: rsync 到远端 (--backup-host)
4. 确保 checkpoint 包含: model_state, optimizer_state, step, lr_schedule, rng_state

## 铁律
- 只改 run_pretrain.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff
- git push origin main
