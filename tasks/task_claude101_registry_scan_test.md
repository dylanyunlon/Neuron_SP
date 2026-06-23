# Claude-101: HeteroRegistry 自动发现测试

## 任务
验证 HeteroRegistry 现在能发现并注册所有 62 个 hetero 模块。

## 具体工作
1. `cat deepspeed/runtime/hetero_registry.py | head -100` 看 discover 逻辑
2. 在 `tests/` 中创建 `test_hetero_registry.py`:
   - test_discover_all_modules: 扫描 deepspeed/runtime/hetero_*.py 和 zero/hetero_*.py，确认全部有 register()
   - test_register_returns_bool: mock engine, 调用 register(), 检查返回值
   - test_no_duplicate_registrations: 连续调用两次 discover_and_register(), 确认不重复
3. `python3 -c "import ast; ast.parse(open('tests/test_hetero_registry.py').read())"` 验证

## 铁律
- 可创建 tests/test_hetero_registry.py
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
