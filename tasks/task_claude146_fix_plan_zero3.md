# Task C146: Fix PartitionSolver._plan_zero3 hardcoded 22:1 grad_accum

## Context
git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP

In `deepspeed/runtime/desloc_engine.py` around line 570, `_plan_zero3()`
hardcodes `grad_accum[H100] = 22` and `grad_accum[A6000] = 1`.

This was the design from commit 93504931 which violates upstream Megatron's
pattern: num_microbatches must be a SINGLE global value, identical on all ranks.
efa47b85 fixed the training loop to use the global value, but the
PartitionSolver still generates 22 — meaning ALL ranks now run 22 forward
passes per step (wasteful on A6000).

## Task
Change `_plan_zero3()` to use `cfg.grad_accum_steps` (from config, default 8)
uniformly for all devices. Differentiate via micro_batch_size instead:
H100 gets larger micro_batch (e.g. 4), A6000 gets micro_batch=1.

Also update the `notes` string and `_estimate_zero3_throughput` call accordingly.

## Constraint
Do NOT open new branches. Keep num_microbatches uniform across ranks.
