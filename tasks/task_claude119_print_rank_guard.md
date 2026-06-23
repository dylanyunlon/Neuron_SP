# Claude-119: train loop print 在所有 rank 输出 → 日志爆炸

## Bug
desloc_engine.py 的 train() 里有多处 print(f"[hetero_grad]...") 在所有5个rank都输出。
每个 step 打5行相同日志, 100000步 = 50万行垃圾。

## 修复
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. grep -n 'print(f"\[hetero_grad\]' deepspeed/runtime/desloc_engine.py
3. 在 train() 开头获取 _is_main = (not dist.is_initialized()) or (dist.get_rank() == 0)
4. 所有 print 改为: if _is_main: print(...)
5. 或者用 logger.info 代替 print, 加 rank prefix

## 铁律
- 只改 deepspeed/runtime/desloc_engine.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff -m "fix: limit train loop prints to rank 0 only"
- git push origin main
