# Claude-110: fix hetero_mimo_topology.py 硬编码 world_size=8

## Bug
hetero_mimo_topology.py line 1686: world_size=8 硬编码在测试/示例中。
ags1 集群是 5 GPU, 不是 8。

## 修复
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. grep -n "world_size=8\|world_size = 8\|world_size==8" deepspeed/runtime/hetero_mimo_topology.py
3. 把硬编码的 world_size=8 改为从 dist.get_world_size() 动态获取
4. 如果是在 doctest/example 里,改为 world_size=5 并加注释 "ags1: 2xA6000 + 1xH100 + 2xBlackwell"
5. 同样检查 hetero_ddp_bucket_pg_collection.py 的 world_size=2 假设

## 铁律
- 只改 hetero_mimo_topology.py 和 hetero_ddp_bucket_pg_collection.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff -m "fix: replace hardcoded world_size with dynamic dist.get_world_size()"
- git push origin main
