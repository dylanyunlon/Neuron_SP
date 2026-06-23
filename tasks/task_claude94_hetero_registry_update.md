# Claude-94: HeteroRegistry 更新 — 自动发现所有 register() 模块

## 任务
更新 `deepspeed/runtime/hetero_registry.py` 的 discover_and_register() 函数。

## 具体工作
1. `cat deepspeed/runtime/hetero_registry.py` 先读
2. 让 discover_and_register() 自动扫描 deepspeed/runtime/ 和 deepspeed/runtime/zero/ 下所有 hetero_*.py
3. 对每个找到的模块，尝试 importlib.import_module 并调用 register(engine)
4. 记录成功/失败/跳过的模块数
5. `python3 -c "import ast; ast.parse(open('deepspeed/runtime/hetero_registry.py').read())"` 验证

## 铁律
- MODIFY EXISTING FILES ONLY
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
