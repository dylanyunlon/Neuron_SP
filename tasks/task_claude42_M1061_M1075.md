# Claude-42 (M1061-M1075): Wire HeteroOptimizerRouter into optimizer construction

## 目标
在 `engine.py` 的 `_configure_optimizer()` 中接入 `hetero_mimo_training_loop.py` 的 `HeteroOptimizerRouter`。

## 具体步骤
1. `grep -n "def _configure_optimizer" engine.py` 定位
2. `cat deepspeed/runtime/hetero_mimo_training_loop.py | grep -A 20 "class HeteroOptimizerRouter"` 读取接口
3. 在 optimizer 构建逻辑中：
   - 检测到异构 GPU 时，使用 HeteroOptimizerRouter 替代单一 optimizer
   - H100 用 fused AdamW，A6000 用 foreach Adam，Blackwell 用 FP8 优化器
4. 验证语法

## 铁律
同 Claude-37
