# Claude-50: M1181-M1195 — Wiring: DesLocEngine.__init__ 接入 build_neuron_sp_config

## 任务
在 `deepspeed/runtime/desloc_engine.py` 的 `__init__` 中调用 `build_neuron_sp_config()`。

## 具体工作
1. `cat deepspeed/runtime/desloc_engine.py` 先读(1330行)
2. `grep -n "def build_neuron_sp_config" deepspeed/runtime/hetero_gdn_selective_recompute.py` 找函数签名
3. 在 DesLocEngine.__init__ 中 import 并调用 build_neuron_sp_config(), 将返回的 config 存为 self.neuron_sp_config
4. `python3 -c "import ast; ast.parse(open('deepspeed/runtime/desloc_engine.py').read())"` 验证

## 铁律
- MODIFY EXISTING FILES ONLY
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
