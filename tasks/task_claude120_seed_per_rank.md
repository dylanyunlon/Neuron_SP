# Claude-120: random seed 所有 rank 相同 → 数据重复

## Bug
如果 torch.manual_seed() 在所有 rank 用同一个 seed, 数据采样完全相同。
5个 GPU 训练的是同一批数据, 等于只用了1个GPU的数据量。

## 修复
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. grep -n "manual_seed\|seed(" run_pretrain.py
3. 改为: seed = base_seed + rank (rank 从 dist.get_rank() 获取)
4. torch.manual_seed(seed)
5. numpy.random.seed(seed) 如果有用 numpy
6. DataLoader 的 worker_init_fn 也要按 rank 区分

## 铁律
- 只改 run_pretrain.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff -m "fix: per-rank seed offset to avoid duplicate data across GPUs"
- git push origin main
