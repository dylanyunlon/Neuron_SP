# Claude-37 (M986-M1000): Wire HeteroRecomputeConfig into DeepSpeedEngine.__init__

## 目标
在 `engine.py` 的 `DeepSpeedEngine.__init__()` 中，检测到 DES-LOC 配置时调用 `build_neuron_sp_config()` 并将返回的 `HeteroRecomputeConfig` 存入 `self._hetero_recompute_config`。

## 具体步骤
1. `cat deepspeed/runtime/hetero_gdn_selective_recompute.py` 读取 `build_neuron_sp_config()` 签名
2. 在 `engine.py` 的 `DeepSpeedEngine.__init__()` 末尾（`self._do_args_sanity_check()` 之后），加入：
   - 导入 `build_neuron_sp_config`
   - 检测 `self._config.hetero_enabled`（如不存在，检测 GPU 异构性）
   - 调用 `build_neuron_sp_config()` 并存 `self._hetero_recompute_config`
3. 在 `forward()` 方法中，若 `self._hetero_recompute_config` 存在，调用其 `apply_recompute_policy()`
4. `python -c "import ast; ast.parse(open('engine.py').read())"` 验证语法

## 铁律
- 作者: dylanyunlon <dogechat@163.com>
- 只改 engine.py，不建新文件
- git commit --signoff -m "M986-M1000: wire build_neuron_sp_config into DeepSpeedEngine.__init__"
