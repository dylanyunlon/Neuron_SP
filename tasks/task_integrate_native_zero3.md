# Task: Integrate Neuron_SP Native ZeRO-3 with Optimizer

## Context
DesLocEngine has removed FSDP and restored native ZeRO-3 via `zero3_hetero_shard.py`.
The model is BF16 on CPU. Each rank holds a FP32 `param_shard` on GPU (VRAM-proportional).
Forward hooks do per-layer all-gather; backward hooks do per-param reduce-scatter.

## Problem
Phase 5 optimizer is `AdamW(self.model.parameters())` — but `model.parameters()` are
BF16 on CPU. After backward, gradients are reduce-scattered into `param_shard` (FP32 on GPU),
NOT into `model.parameters().grad`. So `optimizer.step()` operates on stale CPU params
with no gradients → no learning.

## Required Fix (desloc_engine.py Phase 5)

1. **Optimizer must operate on `param_shard`** (FP32 master copy on GPU):
   ```python
   # param_shard is 1-D FP32 tensor on local GPU
   self.optimizer = AdamW(
       [self.param_shard_state.param_shard],  # single param group
       lr=config.max_lr, ...
   )
   ```
   Note: param_shard needs `.requires_grad_(True)` and `.grad` set manually
   from the reduce-scattered gradient buffer after backward.

2. **After optimizer.step(), sync updated FP32 shard back to model BF16 params**:
   After `self.optimizer.step()`, each rank needs to update the subset of
   `model.parameters()` that intersects with its shard window. This is the
   reverse of `ShardState.build()` line 201-217 (the init copy).
   
   Add a method `ShardState.update_model_params(model)` that copies
   `param_shard[s_start:s_end].to(bf16)` back into the corresponding
   `param.data[p_start:p_end]` for each param that intersects this rank's window.

3. **Gradient wiring**: After backward + reduce-scatter hooks fire,
   `param_shard.grad` should contain the reduced gradient shard.
   Verify `register_backward_hooks()` actually sets `param_shard.grad`
   (or accumulates into a separate buffer that we then assign).

## Files to modify
- `deepspeed/runtime/desloc_engine.py` (Phase 5 optimizer init, train loop optimizer.step)
- `deepspeed/runtime/zero3_hetero_shard.py` (add `update_model_params` method)

## Acceptance criteria
- `optimizer.step()` updates `param_shard` (FP32 on GPU)
- Updated params are synced back to model BF16 params after each step
- Loss should decrease on synthetic data (proves learning is happening)
- No NCCL deadlocks (all ranks must participate in same collectives same number of times)

## Key constraint
- Do NOT use FSDP. This is Neuron_SP's native ZeRO-3 path.
- Do NOT modify zero3_hetero_shard.py's existing hooks — only add the new method.
