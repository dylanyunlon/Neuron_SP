# Claude-126: 训练循环中定期 empty_cache 防 A6000 OOM

## Bug
A6000 48GB 跑 7B model 非常紧张。PyTorch CUDA allocator 碎片化会逐步吃掉余量。
需要在关键点 empty_cache。

## 修复
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat deepspeed/runtime/desloc_engine.py | grep -n "empty_cache\|cuda.memory"
3. 在 train() 循环里每 log_every 步加:
   if step % log_every == 0:
       torch.cuda.empty_cache()  # reclaim fragmented memory
4. 在 checkpoint save 前后加 empty_cache
5. 在 train() 开头加 torch.cuda.reset_peak_memory_stats()
6. 在 log 时打印 per-GPU memory: torch.cuda.max_memory_allocated() / 1e9

## 铁律
- 只改 deepspeed/runtime/desloc_engine.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff -m "fix: periodic empty_cache + memory logging in train loop"
- git push origin main
