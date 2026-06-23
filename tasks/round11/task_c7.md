# C7: Wire PartitionSolver 的 ZeRO-3 vs Pipeline 自动选择

## 目标
改 `deepspeed/runtime/desloc_partition.py`，让 PartitionSolver 根据实际 GPU 拓扑（从 nvidia-smi topo -m 解析）自动选择 ZeRO-3 或 Pipeline 策略，然后在 DesLocEngine.__init__() 中调用。

## 具体改动
1. 在 `desloc_partition.py` 中添加 `auto_select_strategy()` 函数，读取 GPU 拓扑，计算两种策略的预估 MFU，返回最优策略
2. 在 `desloc_engine.py` 的 `__init__()` 中调用 `auto_select_strategy()` 并根据结果初始化对应的并行策略
3. 策略选择逻辑: 如果 GPU 间有 NVLink → ZeRO-3; 纯 PCIe + 算力差 >5x → Pipeline with uneven split

## 文件
改 `deepspeed/runtime/desloc_partition.py` 和 `deepspeed/runtime/desloc_engine.py`
