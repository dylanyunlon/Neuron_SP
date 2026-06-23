# Claude-98: launch_7b.sh NUMA affinity 绑定

## 任务
在 launch_7b.sh 中添加 NUMA affinity 绑定。

## 具体工作
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat launch_7b.sh — 读现有逻辑
3. 添加 NUMA 绑定:
   - GPU0/1/2 在 NUMA0: numactl --cpunodebind=0 --membind=0
   - GPU3/4 在 NUMA1: numactl --cpunodebind=1 --membind=1
4. 用 CUDA_VISIBLE_DEVICES + numactl 对 torchrun 的每个 rank 正确绑定
5. 加注释说明 ags1 拓扑

## 铁律
- 只改 launch_7b.sh
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff
- git push origin main
