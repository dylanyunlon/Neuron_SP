# Claude-43 (M1076-M1090): Wire PCIeP2PCommunicator into allreduce_gradients

## 目标
在 `engine.py` 的 `allreduce_gradients()` 中用 `PCIeP2PCommunicator` 替代默认 NCCL allreduce。

## 具体步骤
1. `cat deepspeed/runtime/hetero_mimo_training_loop.py | grep -A 30 "class PCIeP2PCommunicator"` 读取
2. 在 `allreduce_gradients()` 中：
   - 若 `self._hetero_p2p` 存在，用其 `staged_allreduce()` 替代默认路径
   - 大 tensor (>staging_threshold) 走 CPU DRAM staging
   - 同 NUMA 内用直接 PCIe P2P
3. `self._hetero_p2p` 在 `__init__()` 中由 `setup_hetero_mimo_training()` 创建
4. 验证语法

## 铁律
同 Claude-37
