# Claude-54: M1241-M1255 — Wiring: HeteroRegistry.register_hooks 完善

## 任务
让 HeteroRegistry 的 discover_and_register() 真正把所有 hetero 模块挂载到 engine hooks。

## 具体工作
1. `cat deepspeed/runtime/hetero_registry.py` 先读
2. 找到 discover_and_register() 函数，检查哪些模块缺 register() 方法
3. 为缺 register() 的 top-10 关键模块添加注册逻辑(在各自的 hetero_*.py 中)
4. 验证语法

## 铁律
- MODIFY EXISTING FILES ONLY
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
