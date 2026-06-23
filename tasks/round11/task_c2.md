# C2: Wire HeteroFP32GradAccumManager into backward pass

## 目标
改 `deepspeed/runtime/desloc_engine.py`，在 backward pass 中接入 `HeteroFP32GradAccumManager`，实现三级精度策略和 PCIe 感知的梯度拷贝。

## 具体改动
1. 在文件顶部添加 import: `from deepspeed.runtime.hetero_fp32_grad_accum import HeteroFP32GradAccumManager`
2. 在 `DesLocEngine.__init__()` 中，创建 `self.fp32_grad_manager = HeteroFP32GradAccumManager(self)`
3. 在 `train()` 的 backward 部分（`scaled_loss.backward()` 之后），调用 `self.fp32_grad_manager.accumulate()` 做精度对齐
4. 在 optimizer.step() 前，调用 `self.fp32_grad_manager.sync()` 做跨设备梯度同步

## 文件
只改 `deepspeed/runtime/desloc_engine.py`
