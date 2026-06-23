# Claude-114: CUDA memory fragmentation 防护

## Bug
5卡异构, A6000 只有 48GB, 7B model BF16 ~14GB + activations, 剩余空间很紧。
需要: expandable_segments, max_split_size, 提前 empty_cache。

## 修复
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat launch_7b.sh
3. 在 launch_7b.sh 添加:
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
4. 在 run_pretrain.py 的 run_desloc() 开头加:
   torch.cuda.empty_cache()
   torch.cuda.reset_peak_memory_stats()
5. 在每个 checkpoint save 后加 torch.cuda.empty_cache()

## 铁律
- 只改 launch_7b.sh 和 run_pretrain.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff -m "fix: CUDA memory fragmentation defenses for 48GB A6000"
- git push origin main
