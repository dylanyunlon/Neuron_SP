# Claude-130: Heterogeneous Shard Sizing — Proportional to VRAM

## Context
ZeRO-3 defaults to equal shards. But H100 (93GB) can hold 2x more than A6000 (47GB). Waste to give them equal shards.

## Task
1. In `zero3_hetero_shard.py`, compute shard sizes proportional to available VRAM:
   - Available = total_memory - reserved_overhead (16GB for activations/framework)
   - H100: (93-16)=77GB → ratio 77/(77+31+31)=0.554
   - A6000: (47-16)=31GB → ratio 31/139=0.223 each
2. Partition flat param buffer according to ratios
3. Log shard sizes: "GPU0(H100): 3.65B params, GPU1(A6000): 1.47B params, GPU2(A6000): 1.47B params"
4. All-gather/reduce-scatter must handle unequal shard sizes (use `dist.all_gather` with list of tensors, not `all_gather_into_tensor`)

## Files
- `deepspeed/runtime/zero3_hetero_shard.py`

## Git
Branch: main. Commit with `--signoff`. Author: `dylanyunlon <dogechat@163.com>`.
Token: `SEE_CONFIG`
