# Claude-28: Wire setup_hetero_mimo_training() into train_three_stage.py

在 `pipeline/train_three_stage.py` 中:
1. `cat deepspeed/runtime/hetero_mimo_training_loop.py | grep -n "def setup_hetero_mimo_training"` 找到签名
2. 删除 `DESLOCEngine` 的使用, 改为调用 `setup_hetero_mimo_training(model)` 获取 `HeteroMIMOTrainingLoop`
3. 用 `HeteroMIMOTrainingLoop` 替换 `train_one_stage()` 中的训练循环
4. 三个 stage 之间通过 checkpoint 传递模型权重
