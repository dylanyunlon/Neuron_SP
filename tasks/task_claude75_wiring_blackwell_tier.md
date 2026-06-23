# Claude-75: Add RTX PRO 6000 Blackwell tier to TierDiscovery + PartitionSolver

## 任务
在 TierDiscovery 和 PartitionSolver 中添加 Blackwell 工作站卡支持。

## 步骤
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
3. 读 deepspeed/runtime/desloc_engine.py 找 TierClass enum 和 TierDiscovery
4. 添加 TierClass.RTX_PRO_6000_BW = "RTX_PRO_6000_BW"
5. 在 TierDiscovery 中添加 SM12.0 / 96GB / compute_cap 12.0 的识别逻辑
6. 在 desloc_partition.py 的 FLOPS 表中添加 (估500 TFLOPS BF16)
7. 确保 PartitionSolver 能处理 5卡3种架构 (2xA6000 + 1xH100 + 2xBlackwell)
8. git add -A && git commit --signoff -m "add RTX PRO 6000 Blackwell tier to TierDiscovery + PartitionSolver"
9. git push origin main

## 铁律
- 只改 deepspeed/runtime/desloc_engine.py 和 deepspeed/runtime/desloc_partition.py
- 作者: dylanyunlon <dogechat@163.com>
