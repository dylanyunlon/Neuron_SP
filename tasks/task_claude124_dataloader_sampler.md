# Claude-124: DataLoader DistributedSampler 缺失

## Bug
多卡训练 DataLoader 需要 DistributedSampler 来分片数据, 否则所有 rank 读同一批。
检查 run_pretrain.py 和 train_three_stage.py 的 DataLoader 是否有 sampler。

## 修复
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. grep -n "DataLoader\|DistributedSampler" run_pretrain.py pipeline/train_three_stage.py
3. 如果 DataLoader 没有 sampler=DistributedSampler, 加上:
   from torch.utils.data.distributed import DistributedSampler
   sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
   loader = DataLoader(dataset, sampler=sampler, ...)
4. 每个 epoch 要调 sampler.set_epoch(epoch)

## 铁律
- 只改 run_pretrain.py 和 pipeline/train_three_stage.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff -m "fix: add DistributedSampler to DataLoader for multi-GPU data sharding"
- git push origin main
