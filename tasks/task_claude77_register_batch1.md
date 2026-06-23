# Claude-77: 补全 register() — 第1批 (6个模块)

## 任务
为以下 6 个 hetero 模块添加 `register(engine)` 函数，让 HeteroRegistry 能发现并挂载它们。

## 目标文件（每个都要先 cat 再改）
1. `deepspeed/runtime/hetero_fp32_grad_accum.py`
2. `deepspeed/runtime/hetero_gdn_selective_recompute.py`
3. `deepspeed/runtime/hetero_grad_norm_skip.py`
4. `deepspeed/runtime/hetero_step_batch_scheduler.py`
5. `deepspeed/runtime/hetero_mimo_training_loop.py`
6. `deepspeed/runtime/hetero_mimo_topology.py`

## register() 规范
参考已有的实现: `grep -A 20 "def register(" deepspeed/runtime/hetero_elastic_batch.py`

每个 register(engine) 应:
1. 检查 engine 是否有对应属性
2. 将模块实例挂到 engine 上
3. 返回注册成功/跳过的 bool

## 铁律
- MODIFY EXISTING FILES ONLY — 在每个文件末尾添加 register() 函数
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
