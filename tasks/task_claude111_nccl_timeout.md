# Claude-111: NCCL timeout 和 异步错误处理

## Bug
PCIe-only 5卡异构集群, 跨NUMA通信慢 (SYS level), NCCL 默认 timeout 5min 可能不够。
梯度同步时 H100 比 A6000 快20倍, 快的卡等慢的卡会超时。

## 修复
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat run_pretrain.py | grep -n "init_process_group\|timeout\|NCCL"
3. 在 dist.init_process_group 里加 timeout=datetime.timedelta(minutes=30)
4. 在 launch_7b.sh 里加 export NCCL_TIMEOUT=1800000 (30分钟毫秒)
5. 在 run_pretrain.py 两个 init_process_group 调用都加 timeout

## 铁律
- 只改 run_pretrain.py 和 launch_7b.sh
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff -m "fix: extend NCCL timeout to 30min for heterogeneous PCIe cluster"
- git push origin main
