# Claude-129: ZeRO-3 Backward Grad Reduce-Scatter

## Context
After backward, each rank has grads for the full model (from the all-gathered forward). Need reduce-scatter so each rank keeps only grads for its shard.

## Task
1. Register backward hooks on parameters: after `.grad` is computed, do `dist.reduce_scatter_tensor`
2. Each rank's grad buffer = grad_shard (1/N of full grads), already reduced (sum) across ranks
3. Scale by 1/world_size for averaging
4. Integrate with existing `HeteroFP32GradAccumManager` — if fp32_grad_manager is not None, accumulate into FP32 buffer after scatter

## Files
- `deepspeed/runtime/zero3_hetero_shard.py`
- `deepspeed/runtime/desloc_engine.py`

## Git
Branch: main. Commit with `--signoff`. Author: `dylanyunlon <dogechat@163.com>`.
Token: `SEE_CONFIG`
