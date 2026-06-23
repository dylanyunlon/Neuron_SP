# Claude-66: M1421-M1435 — Wiring: HeteroRegistry 完善 + Blackwell RTX PRO 6000 tier

## 任务
在 HeteroRegistry 和 TierDiscovery 中添加 RTX PRO 6000 Blackwell (SM12.0, 96GB) 支持。

## 具体工作
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat deepspeed/runtime/desloc_engine.py — 找 TierClass enum 和 TierDiscovery
3. 在 TierClass 中添加 RTX_PRO_6000_BW = "RTX_PRO_6000_BW"
4. 在 TierDiscovery 中添加 SM12.0 / 96GB 的识别逻辑
5. 在 desloc_partition.py 的 FLOPS 表中添加 RTX PRO 6000 BW 的 BF16 TFLOPS (估300)
6. 确保 PartitionSolver 能处理5卡3种架构的集群

## 铁律
- 只改 deepspeed/runtime/desloc_engine.py 和 deepspeed/runtime/desloc_partition.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff
- git push origin main
