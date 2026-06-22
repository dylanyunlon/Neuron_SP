# Claude-38 (M1001-M1015): Wire HeteroGradNormSkipController into DeepSpeedEngine.step()

## 目标
在 `engine.py` 的 `DeepSpeedEngine.step()` 中调用 `integrate_with_deepspeed_engine()` 接入梯度异常跳步。

## 具体步骤
1. `cat deepspeed/runtime/hetero_grad_norm_skip.py | head -100` 读取接口
2. 在 `DeepSpeedEngine.__init__()` 中（M986 之后），创建 `HeteroGradNormConfig` 并调用 `integrate_with_deepspeed_engine(self, config)` —— 此函数会 monkey-patch `self.step()`
3. 将返回的 `HeteroGradNormSkipController` 存为 `self._hetero_grad_controller`
4. 在 `DeepSpeedEngine._take_model_step()` 中，step 之后调用 `self._hetero_grad_controller.record_step(grad_norm)`
5. 验证语法

## 铁律
同 Claude-37
