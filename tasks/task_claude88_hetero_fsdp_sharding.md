# Claude-88: 异构FSDP sharding策略 — 按显存比例分配

## 任务
在 run_pretrain.py 的 FSDP wrap 中加入异构 shard_ratio。

## 具体工作
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat run_pretrain.py | grep -B5 -A30 "FSDP" — 读 FSDP wrap 逻辑
3. cat pipeline/engine_bridge.py | grep -A20 "detect_gpu_tiers" — 读 tier 检测
4. 在 FSDP 初始化时,根据 GPU 显存设置不同的 shard ratio (A6000=1.0, H100/Blackwell=2.0)
5. 确保 ShardingStrategy 用 FULL_SHARD 而不是 NO_SHARD

## 铁律
- 只改 run_pretrain.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff
- git push origin main
