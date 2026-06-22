# Claude-44 (M1091-M1105): Wire HeteroStepBatchScheduler + HeteroBatchSampler into DataLoader

## 目标
在 `deepspeed/runtime/dataloader.py` (DeepSpeedDataLoader) 中接入异构批量调度。

## 具体步骤
1. `cat deepspeed/runtime/hetero_step_batch_scheduler.py | head -80` 读取接口
2. `cat datasets/bigcode/commit_packing.py | grep -A 20 "class HeteroBatchSampler"` 读取接口
3. 在 `DeepSpeedDataLoader.__init__()` 中：
   - 检测异构配置时，用 `HeteroStepBatchScheduler` 包装 batch size 调度
   - 用 `HeteroBatchSampler` 按 GPU 显存比例分配 micro-batch
4. 在 `__iter__()` 中每步调用 scheduler 获取当前 batch size
5. 验证语法

## 铁律
同 Claude-37
