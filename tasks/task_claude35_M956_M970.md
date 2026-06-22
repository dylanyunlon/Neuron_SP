# Claude-35: Rewrite train_three_stage.py to use real DesLocEngine + HeteroMIMOTrainingLoop

在 `pipeline/train_three_stage.py` 中:
1. 删除 `from pipeline.engine_bridge import DESLOCEngine` (这是玩具)
2. 改为 `from deepspeed.runtime.hetero_mimo_training_loop import setup_hetero_mimo_training`
3. 改为 `from deepspeed.models.llama_7b import LLaMAConfig, LLaMA` (而非 GPT2LMHeadModel)
4. `build_model()` 改为构建 LLaMA 7B (hidden=4096, layers=32, heads=32, kv_heads=32)
5. 每个 stage 的训练循环改为调用 `HeteroMIMOTrainingLoop.run()`
6. 删除 `pipeline/engine_bridge.py`
