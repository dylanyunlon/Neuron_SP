# Claude-46 (M1121-M1135): Integration smoke test + push 验证

## 目标
在 `pipeline/smoke_test.py` 中添加所有 wiring 的集成测试。

## 具体步骤
1. `cat pipeline/smoke_test.py` 读取现有测试
2. 添加以下测试（全部 CPU mock，不需 GPU）：
   - `test_engine_has_hetero_recompute` — DeepSpeedEngine 初始化后有 `_hetero_recompute_config`
   - `test_engine_has_grad_controller` — 有 `_hetero_grad_controller`
   - `test_engine_has_memory_manager` — 有 `_hetero_memory_manager`
   - `test_engine_has_registry` — 有 `_hetero_registry` 且发现 >100 模块
   - `test_pcie_p2p_in_allreduce` — allreduce 路径包含 PCIe staging
   - `test_checkpoint_saves_loc_cache` — checkpoint 包含 cache 统计
   - `test_three_stage_wiring` — train_three_stage.py 的三阶段都能初始化
3. 验证全部通过：`python -m pytest pipeline/smoke_test.py -v`
4. 验证语法

## 铁律
同 Claude-37
