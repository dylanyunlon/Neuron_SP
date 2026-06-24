# Claude-127: ZeRO-3 Parameter Sharding Init for DesLocEngine

## Context
Currently each rank holds a full copy of the model. 7B model + Adam needs ~50GB, A6000 (47GB) OOMs. Need ZeRO-3: each rank holds 1/N of parameters, gathering full params on-demand for forward/backward.

## Task
In `desloc_engine.py`, after model creation and before optimizer init:
1. Partition `model.parameters()` into `world_size` shards by flattening all params into a 1D buffer, then each rank keeps only `shard[rank]`
2. Store `self.param_shard` (local FP32 master copy) and `self.param_offsets` (mapping from flat index to original param)
3. Add `gather_full_params(module)` context manager that does all-gather before forward, releases after
4. Add `scatter_grads()` that reduces grads and keeps only local shard's portion

## Files to edit
- `deepspeed/runtime/desloc_engine.py` — add sharding in `__init__` between model.to() and optimizer init
- `deepspeed/runtime/zero3_hetero_shard.py` — new file for shard utilities

## Constraints
- Use `dist.all_gather_into_tensor` / `dist.reduce_scatter_tensor` (NCCL)
- Heterogeneous: H100 gets larger shard (proportional to VRAM), A6000 gets smaller
- Must be backward-compatible: if world_size=1, skip sharding entirely

## Test
After sharding, `sum(shard.numel() for all ranks)` must equal original `sum(p.numel() for p in model.parameters())`.

## Git
Branch: main. Commit with `--signoff`. Author: `dylanyunlon <dogechat@163.com>`.
Token: `SEE_CONFIG`
