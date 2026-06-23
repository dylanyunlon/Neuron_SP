# C1: Wire HeteroMIMOTrainingLoop into DesLocEngine.train()

## 目标
改 `deepspeed/runtime/desloc_engine.py` 的 `train()` 方法，让它调用 `setup_hetero_mimo_training()` 初始化 PCIeP2PCommunicator 和 SharedLocalityCache，然后在训练循环中使用 HeteroMIMOTrainingLoop。

## 具体改动
1. 在 `DesLocEngine.__init__()` 末尾（约 L860 后），调用 `setup_hetero_mimo_training(self)` 并保存返回的 `HeteroMIMOTrainingLoop` 对象到 `self.mimo_loop`
2. 在 `train()` 中，forward/backward 步骤替换为 `self.mimo_loop.step(batch)` 如果 `self.mimo_loop is not None`
3. import 已经在文件顶部了（L44-49 的 `from deepspeed.runtime.hetero_mimo_training_loop import ...`），确认 import 正确

## 文件
只改 `deepspeed/runtime/desloc_engine.py`
