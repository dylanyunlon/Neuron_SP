# Claude-29: Wire integrate_with_deepspeed_engine() into DesLocEngine.train()

在 `deepspeed/runtime/desloc_engine.py` 的 `DesLocEngine.train()` 中:
1. `cat deepspeed/runtime/hetero_grad_norm_skip.py | grep -n "def integrate_with_deepspeed"` 找到签名
2. 在 train() 开始时调用 `integrate_with_deepspeed_engine(self, config)` 获取 controller
3. 在每步的 gradient clipping 之后, 用 `controller.should_skip()` 决定是否跳过这步 optimizer.step()
4. 加 print 诊断: skip 计数和 grad norm 值
