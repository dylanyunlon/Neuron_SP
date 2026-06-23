# Claude-91: 论文 System Design section 填充

## 任务
填充 `FAUST_nips2026/main.tex` 的 Section 4 (System Design)。

## 具体工作
1. `cat FAUST_nips2026/main.tex` 先读
2. 填充 4.1 Heterogeneous GPU Tiers:
   - A6000 (48GB, SM86, PCIe Gen4), H100 NVL (94GB, SM90, PCIe Gen5), RTX PRO 6000 Blackwell (96GB, SM120, PCIe Gen5)
   - TierClass 分配: compute tier, memory tier, bandwidth tier
3. 填充 4.2 PCIe-aware Communication:
   - 无 NVLink, 全 PCIe topology
   - NODE vs SYS 跨 NUMA 延迟差异
   - PCIeP2PCommunicator 的 ring/tree 策略选择
4. 填充 4.3 MIMO Training Loop:
   - 多输入多输出训练循环适配异构设备

## 铁律
- MODIFY EXISTING FILES ONLY
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
