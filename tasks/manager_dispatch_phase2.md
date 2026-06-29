# Phase 2 任务分工 — Megatron core 吸收（实现补全）

## 当前状态
骨架已完成，核心文件都在 `deepspeed/core/` 下。`desloc_engine.py` 已经 import 并调用这些文件。
还有几个关键的 NotImplementedError 需要填上，以及 transformer_config 需要对齐 Megatron HEAD。

## 任务分配

### Worker A: Pipeline Schedule 补全
文件: `deepspeed/core/pipeline_parallel/schedules.py`
参考: `Megatron-LM/megatron/core/pipeline_parallel/schedules.py` (2471行)
目标:
- 实现 combined_1f1b_schedule_for_no_pipelining (line 155)
- 实现 combined_1f1b_schedule_for_interleaved_pipelining (line 156)
- 保留 DES-LOC 的异构 bubble 填充逻辑
- 确保与 `desloc_engine.py` 的 `_run_pipeline_step()` 方法兼容

### Worker B: Transformer Config 对齐 + Attention CP
文件:
- `deepspeed/core/transformer/transformer_config.py` (补全缺失字段)
- `deepspeed/core/transformer/dot_product_attention.py` (实现 CP 路径)
参考:
- `Megatron-LM/megatron/core/transformer/transformer_config.py` (2805行)
- `Megatron-LM/megatron/core/transformer/dot_product_attention.py`
目标:
- 对齐 FSDP/CP/MTP 相关配置字段
- 实现 dot_product_attention 的 context_parallel 路径

### Worker C: 验证和集成测试
文件: `tests/test_core_integration.py` (新建)
目标:
- 写 smoke test 验证 deepspeed.core 的 import chain 完整
- 验证 parallel_state init → DDP → finalize_model_grads → DistributedOptimizer 这条链能跑通
- 在 CPU 上跑（不需要 GPU），验证接口对齐
