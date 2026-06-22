# Claude-36: Integration smoke test — verify all wiring on mock setup

在 `pipeline/smoke_test.py` 中:
1. 添加 test: `test_wiring_recompute_config` — 调用 `build_neuron_sp_config()` 验证返回 HeteroRecomputeConfig
2. 添加 test: `test_wiring_grad_norm_skip` — 调用 `integrate_with_deepspeed_engine()` 验证 monkey-patch
3. 添加 test: `test_wiring_mimo_setup` — 调用 `setup_hetero_mimo_training(model)` 验证返回 HeteroMIMOTrainingLoop
4. 添加 test: `test_llama_model` — 验证 `deepspeed/models/llama_7b.py` 能构建带 RoPE+GQA+RMSNorm 的模型
5. 所有 test 用 CPU mock (不需要真 GPU)
