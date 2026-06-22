# Claude-31: Wire HeteroStepBatchScheduler into training loop

在 `deepspeed/runtime/desloc_engine.py` 的 `DesLocEngine.train()` 中:
1. `cat deepspeed/runtime/hetero_step_batch_scheduler.py | grep -n "class HeteroStepBatch"` 找到类
2. 在 `__init__` 中创建 `HeteroStepBatchScheduler`
3. 每步开始时用 scheduler 决定每个 GPU 的 micro-batch 数量 (H100 多跑几个 micro-batch)
4. 替换固定的 `for micro in range(self.grad_accum)` 为 scheduler 动态分配
