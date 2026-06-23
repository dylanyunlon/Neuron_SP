# C8: Wire activation checkpointing with LOC cache

## 目标
改 `deepspeed/runtime/desloc_engine.py`，在模型初始化后调用 `build_neuron_sp_config()` 返回的 `HeteroRecomputeConfig` 来设置 per-layer 的 activation checkpointing 策略。

## 具体改动
1. `self.neuron_sp_config` 已经在 __init__() L1043 创建了，但没有实际 apply 到模型
2. 添加 `_apply_hetero_recompute(model, config)` 函数，遍历模型的 transformer layers，根据 config 中每层的设备类型设置不同的 checkpoint 策略:
   - A6000 层: 全 recompute (省显存)
   - H100 层: 选择性 recompute (只 recompute attention)
3. 在 __init__() 中模型创建后调用这个函数

## 文件
只改 `deepspeed/runtime/desloc_engine.py`
