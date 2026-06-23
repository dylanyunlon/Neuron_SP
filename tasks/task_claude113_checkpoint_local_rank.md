# Claude-113: checkpoint save/load 只在 rank 0 执行

## Bug
所有 rank 同时写 checkpoint 会冲突。save 应该只在 rank 0。
load 所有 rank 都要做, 但要 map_location 到各自的 local device。

## 修复
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. grep -n "save_checkpoint\|load_checkpoint\|torch.save\|torch.load" run_pretrain.py | head -20
3. save: 加 if rank == 0 guard
4. load: torch.load(..., map_location=f"cuda:{local_rank}")
5. save 后加 dist.barrier() 让其他 rank 等 rank 0 写完

## 铁律
- 只改 run_pretrain.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff -m "fix: checkpoint save on rank0 only, load with map_location=local"
- git push origin main
