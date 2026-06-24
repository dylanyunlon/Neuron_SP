# Claude-132: Gradient AllReduce Sync Across Heterogeneous Ranks

## Context
Currently each rank does forward+backward independently. Gradients are NOT synced across ranks — each rank trains on its own data with its own grads. This is NOT data parallel, it's 3 independent training runs.

## Task
1. After backward on each rank, add `dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)` for all params
2. Do this BEFORE optimizer.step()
3. Use bucketed all-reduce for efficiency (bucket_size_mb=25)
4. Handle the case where fp32_grad_manager is None on A6000 ranks — still need to all-reduce the bf16 grads
5. Add a barrier before optimizer.step() to ensure all ranks have synced grads

## Location in code
`desloc_engine.py` train() method, after the backward pass section, before `self.step()`.

## Files
- `deepspeed/runtime/desloc_engine.py`

## Git
Branch: main. Commit with `--signoff`. Author: `dylanyunlon <dogechat@163.com>`.
Token: `SEE_CONFIG`
