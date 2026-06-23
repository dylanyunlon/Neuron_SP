# Claude-76: Wire HeteroOptimizerRouter + StepBatchScheduler integration

## 任务
把异构优化器路由和批次调度器完整接入训练循环。

## 步骤
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
3. 读 deepspeed/runtime/zero/hetero_optimizer_router.py
4. 读 deepspeed/runtime/hetero_step_batch_scheduler.py (已部分接入)
5. 在 DesLocEngine.__init__() 中初始化 HeteroOptimizerRouter
6. 在 train() 的 optimizer.step() 替换为通过 router 的 per-device step
7. 确保 StepBatchScheduler 的 per_device_assignment 正确传递给 data loader
8. git add -A && git commit --signoff -m "wire HeteroOptimizerRouter + StepBatchScheduler into DesLocEngine"
9. git push origin main

## 铁律
- 只改 deepspeed/runtime/desloc_engine.py
- 作者: dylanyunlon <dogechat@163.com>
