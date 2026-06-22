# Claude-45 (M1106-M1120): Wire SharedLocalityCache into checkpoint save/load

## 目标
将 LOC cache 状态纳入 checkpoint，使恢复训练时 cache 命中率不归零。

## 具体步骤
1. `cat deepspeed/checkpoint/hetero_checkpoint_config.py | head -60` 读取
2. `cat deepspeed/checkpoint/hetero_async_checkpoint_save.py | head -60` 读取
3. 在 `engine.py` 的 `save_checkpoint()` 中：
   - 序列化 `SharedLocalityCache` 的热度统计到 checkpoint
4. 在 `load_checkpoint()` 中：
   - 恢复 cache 热度统计，预热关键 entry
5. 验证语法

## 铁律
同 Claude-37
