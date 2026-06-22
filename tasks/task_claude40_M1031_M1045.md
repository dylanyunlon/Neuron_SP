# Claude-40 (M1031-M1045): Wire HeteroMemoryManager into DeepSpeedEngine for CPU offload

## 目标
将 `deepspeed/runtime/zero/hetero_memory_manager.py` 的 Gemini-style 非对称 offload 接入引擎。

## 具体步骤
1. `cat deepspeed/runtime/zero/hetero_memory_manager.py | head -80` 读取接口
2. 在 `engine.py` 的 `DeepSpeedEngine.__init__()` 中初始化 `HeteroMemoryManager`
3. 在 `_take_model_step()` 中的 optimizer.step() 之后调用 `memory_manager.post_step_evict()`
4. 在 `forward()` 的 prologue 中调用 `memory_manager.pre_forward_prefetch()`
5. 验证语法

## 铁律
同 Claude-37
