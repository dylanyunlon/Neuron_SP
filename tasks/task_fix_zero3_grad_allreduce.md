# Task: Fix ZeRO-3 gradient synchronization — add all-reduce before optimizer.step()

## Current State (commit 57f9dc5a)
- Neuron_SP native ZeRO-3 is running WITHOUT FSDP
- Full BF16 model loaded to each GPU (~12GB)  
- Backward hooks extract local grad slice into param_shard.grad
- Optimizer operates on param_shard (FP32 master copy)
- Step 1 produced loss=2621.4, grad_norm=90.7 — forward+backward works
- Crash: "CUDA illegal memory access" at clip_grad_norm_ on param_shard

## Root Cause
Two issues:

### 1. Missing gradient all-reduce
After backward, each rank's `param_shard.grad` only contains gradients from
that rank's own micro-batches. No `dist.all_reduce()` synchronizes gradients
across ranks. Without this, each rank optimizes with partial gradients.

The fix: add `dist.all_reduce(param_shard.grad, op=dist.ReduceOp.AVG)` 
(or SUM then divide) after all micro-batches' backward passes complete 
and before `clip_grad_norm_` + `optimizer.step()`.

Location: `deepspeed/runtime/desloc_engine.py`, train() method, between
the micro-batch loop and gradient clipping. Look for the comment
"Backward hooks have already written reduced grads into param_shard.grad".

### 2. Potential CUDA illegal memory access  
The backward hooks in `zero3_hetero_shard.py` line ~408 do:
```python
_param_shard.grad[_shard_start:_shard_start + take].add_(
    local_grad[:take].to(dtype=_param_shard.dtype, device=_param_shard.device)
)
```
The `local_grad` comes from `flat[p_lo:p_hi]` where `flat = param.grad.reshape(-1)`.
Since the FULL model is on GPU, `param.grad` is full-size on GPU too.
But `p_lo:p_hi` indices are computed relative to the global flat offset.

VERIFY that for every parameter:
- `p_lo` and `p_hi` do not exceed `flat.numel()` 
- `_shard_start + take` does not exceed `_param_shard.grad.numel()`
- The `.to(device=_param_shard.device)` is correct (both should be same GPU)

Add bounds checking with asserts for debugging:
```python
assert p_hi <= flat.numel(), f"p_hi={p_hi} > flat.numel()={flat.numel()} for {name}"
assert _shard_start + take <= _param_shard.numel(), f"shard overflow for {name}"
```

## Files to Modify
1. `deepspeed/runtime/desloc_engine.py` — add all-reduce in train()
2. `deepspeed/runtime/zero3_hetero_shard.py` — add bounds checks in backward hook

## Acceptance Criteria  
- `dist.all_reduce(param_shard.grad)` before clip_grad_norm_
- No illegal memory access crash
- Loss decreases over steps (proves optimizer is learning)
- Works with heterogeneous micro-batch counts (H100=7, A6000=1)

## Key Files to Read First
- `deepspeed/runtime/zero3_hetero_shard.py` lines 350-440 (backward hook)
- `deepspeed/runtime/desloc_engine.py` train() method (search for "clip_grad_norm_")
- `deepspeed/runtime/zero3_hetero_shard.py` ShardState.build() (shard_offsets, param_offsets)

## Constraints
- Do NOT use FSDP
- Do NOT modify the forward path (full model on GPU is correct)
- Keep heterogeneous micro-batch scheduling (allocation.num_microbatches per rank)
- The all-reduce must be on param_shard.grad only (small: ~12GB H100, ~6GB A6000)
