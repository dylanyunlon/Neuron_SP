# Claude-87: 补全核心6个模块的 register()

## 目标文件（先cat再改）
1. `deepspeed/runtime/hetero_fp32_grad_accum.py`
2. `deepspeed/runtime/hetero_gdn_selective_recompute.py`
3. `deepspeed/runtime/hetero_grad_norm_skip.py`
4. `deepspeed/runtime/hetero_step_batch_scheduler.py`
5. `deepspeed/runtime/hetero_mimo_training_loop.py`
6. `deepspeed/runtime/hetero_pretrain_config.py`

## register() 模板
先看已有实现: `grep -B2 -A25 "def register(" deepspeed/runtime/hetero_elastic_batch.py`
每个 register(engine) 应: 检查 engine 属性 → 创建模块实例 → 挂到 engine → 返回 bool

## 铁律
- MODIFY EXISTING FILES ONLY
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
