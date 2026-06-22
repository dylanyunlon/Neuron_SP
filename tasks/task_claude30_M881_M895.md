# Claude-30: Wire HeteroFP32GradAccumManager into backward pass

在 `deepspeed/runtime/desloc_engine.py` 的 `DesLocEngine.train()` 的 backward 部分:
1. `cat deepspeed/runtime/hetero_fp32_grad_accum.py | grep -n "class HeteroFP32GradAccum"` 找到类
2. 在 `__init__` 中创建 `HeteroFP32GradAccumManager`
3. 在每个 micro-batch 的 `scaled_loss.backward()` 之后, 用 manager 将梯度提升到 FP32
4. 在 `optimizer.step()` 之前, 用 manager 同步所有 rank 的 FP32 梯度
