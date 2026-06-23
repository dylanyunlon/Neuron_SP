# Claude-80: 补全 register() — 第4批 (7个 zero 子目录模块)

## 目标文件 (注意在 zero/ 子目录)
1. `deepspeed/runtime/zero/hetero_allgather_pipeline.py`
2. `deepspeed/runtime/zero/hetero_grad_reduce_double_buffer.py`
3. `deepspeed/runtime/zero/hetero_optimizer_router.py`
4. `deepspeed/runtime/zero/hetero_wgrad_race_fix.py`
5. `deepspeed/runtime/zero/hetero_memory_manager.py`
6. `deepspeed/runtime/zero/hetero_fsdp_double_buffer.py`
7. `deepspeed/runtime/zero/hetero_overlap_param_gather.py`

## register() 规范
同 Claude-77。参考: `grep -A 20 "def register(" deepspeed/runtime/hetero_elastic_batch.py`

## 铁律
- MODIFY EXISTING FILES ONLY
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
