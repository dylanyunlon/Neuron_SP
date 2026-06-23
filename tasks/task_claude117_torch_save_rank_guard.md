# Claude-117: torch.save 缺 rank 0 guard

## Bug
desloc_engine.py:1914 和 train_three_stage.py:395 的 torch.save 没有 rank guard。
5个rank同时写同一个文件 → 文件损坏。

## 修复
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. sed -n '1910,1920p' deepspeed/runtime/desloc_engine.py — 看上下文
3. 在 torch.save(payload, path) 前加: if dist.is_initialized() and dist.get_rank() != 0: return
4. save 后加: if dist.is_initialized(): dist.barrier()
5. 同样修 train_three_stage.py:395

## 铁律
- 只改这两个文件
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff -m "fix: torch.save only on rank 0 with barrier after"
- git push origin main
