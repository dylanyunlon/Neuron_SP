# Claude-33: Wire PCIeP2PCommunicator into pipeline stage boundaries

在 `deepspeed/runtime/hetero_mimo_training_loop.py` 的 `HeteroMIMOTrainingLoop` 中:
1. `grep -n "class PCIeP2PCommunicator" deepspeed/runtime/hetero_mimo_training_loop.py`
2. 确认 `PCIeP2PCommunicator` 已被 `HeteroMIMOTrainingLoop.forward()` 调用
3. 如未接入, 在 pipeline stage boundary 的 activation 传递中插入 PCIeP2PCommunicator
4. 特别处理跨 NUMA (GPU0-2 ↔ GPU3-4) 的传输: 用 CPU DRAM staging
