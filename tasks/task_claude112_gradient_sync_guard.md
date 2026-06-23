# Claude-112: dist.all_reduce 缺 is_initialized guard

## Bug
desloc_engine.py 里有多处 dist.barrier() 和 dist.all_reduce() 没检查 dist.is_initialized()。
单GPU调试时会 crash。

## 修复
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. grep -n "dist\.barrier\|dist\.all_reduce\|dist\.broadcast" deepspeed/runtime/desloc_engine.py
3. 对每个调用,如果前面没有 is_initialized check, 加上:
   if dist.is_initialized(): dist.barrier()
4. 或者在文件顶部写一个 helper: _is_dist = dist.is_initialized() if HAS_DIST else False

## 铁律
- 只改 deepspeed/runtime/desloc_engine.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff -m "fix: guard all dist calls with is_initialized check"
- git push origin main
