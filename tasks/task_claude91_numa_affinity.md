# Claude-91: NUMA affinity 自动绑定

## 任务
在 launch_7b.sh 中添加 NUMA affinity 绑定。

## 具体工作
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat launch_7b.sh — 读现有启动脚本
3. GPU0/1/2 绑定 NUMA0 (cpus 0-31,64-95), GPU3/4 绑定 NUMA1 (cpus 32-63,96-127)
4. 用 numactl --cpunodebind=0 --membind=0 给 NUMA0 的 GPU 进程
5. 确保 CUDA_VISIBLE_DEVICES 和 numactl 正确对应

## 铁律
- 只改 launch_7b.sh
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff
- git push origin main
