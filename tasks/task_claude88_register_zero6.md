# Claude-88: 补全 zero/ 子目录 6 个模块的 register()

## 目标文件
1. `deepspeed/runtime/zero/hetero_allgather_pipeline.py`
2. `deepspeed/runtime/zero/hetero_grad_reduce_double_buffer.py`
3. `deepspeed/runtime/zero/hetero_wgrad_race_fix.py`
4. `deepspeed/runtime/zero/hetero_memory_manager.py`
5. `deepspeed/runtime/zero/hetero_fsdp_double_buffer.py`
6. `deepspeed/runtime/zero/hetero_overlap_param_gather.py`

## 参考: `grep -B2 -A25 "def register(" deepspeed/runtime/hetero_elastic_batch.py`

## 铁律
- MODIFY EXISTING FILES ONLY
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
