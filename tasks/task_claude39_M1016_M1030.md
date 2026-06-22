# Claude-39 (M1016-M1030): Wire HeteroFP32GradAccumManager into ZeRO backward

## 目标
在 `deepspeed/runtime/zero/stage_1_and_2.py` 或 `stage3.py` 中接入 `hetero_fp32_grad_accum.py` 的三级精度策略。

## 具体步骤
1. `cat deepspeed/runtime/hetero_fp32_grad_accum.py | head -80` 读取 HeteroFP32GradAccumManager 接口
2. 在 ZeRO optimizer 的 `backward()` 或 `reduce_gradients()` 中：
   - 创建 `HeteroFP32GradAccumManager` 实例（在 `__init__` 时）
   - 在梯度 reduce 前调用 `manager.pre_reduce(bucket)`
   - 在 reduce 后调用 `manager.post_reduce(bucket)`
3. 确保 PCIe 感知的拷贝调度被激活（读取 manager 的 `pcie_aware_copy` 方法）
4. 验证语法

## 铁律
同 Claude-37
