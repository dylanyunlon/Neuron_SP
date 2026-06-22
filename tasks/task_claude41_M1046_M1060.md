# Claude-41 (M1046-M1060): Wire HeteroRegistry.discover_and_register() into init

## 目标
让 `hetero_registry.py` 的自动发现机制在引擎初始化时运行，注册所有150个hetero模块。

## 具体步骤
1. `cat deepspeed/runtime/hetero_registry.py` 完整读取
2. 在 `engine.py` 的 `DeepSpeedEngine.__init__()` 中：
   - 实例化 `HeteroRegistry()`
   - 调用 `registry.discover_modules()`（扫描 deepspeed/ 下所有 hetero_*.py）
   - 调用 `registry.register_hooks(self)` 为每个模块注册到引擎
   - 存为 `self._hetero_registry`
3. 对没有 `register()` 函数的模块，在 registry 中添加 fallback 逻辑
4. 验证语法

## 铁律
同 Claude-37
