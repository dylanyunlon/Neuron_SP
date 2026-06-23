# Claude-89: 补全剩余所有缺 register() 的模块

## 任务
找出所有缺 register() 的 hetero 模块并补全:
```bash
for f in deepspeed/runtime/hetero_*.py deepspeed/runtime/zero/hetero_*.py; do
    grep -q "def register(" "$f" 2>/dev/null || echo "MISSING: $f"
done
```
为每个 MISSING 的添加 register(engine)。

## 参考: `grep -B2 -A25 "def register(" deepspeed/runtime/hetero_elastic_batch.py`

## 铁律
- MODIFY EXISTING FILES ONLY
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
