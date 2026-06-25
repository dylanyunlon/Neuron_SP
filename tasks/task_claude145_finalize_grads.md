# Task C145: Add finalize_model_grads (upstream grad all-reduce)

## Context
git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP

Read `Megatron-LM/megatron/core/pipeline_parallel/schedules.py` line ~793 and
`Megatron-LM/megatron/core/distributed/finalize_model_grads.py` line ~449.

Upstream Megatron calls `finalize_model_grads` after ALL microbatches complete,
BEFORE optimizer.step(). This does all-reduce on gradients across DP replicas.

Neuron_SP's `desloc_engine.py` train() loop has NO such gradient all-reduce.
The backward hooks in `zero3_hetero_shard.py` only extract the local shard
slice — no cross-rank gradient reduction. Each rank optimizes independently
on its own data's gradient, which is incorrect for data-parallel training.

## Task
In `deepspeed/runtime/desloc_engine.py`, after the microbatch for-loop
(line ~2150, after the async norm drain) and BEFORE the gradient clipping
section, add gradient all-reduce on param_shard.grad.

Handle heterogeneous shard sizes: use `scatter_grads()` pattern from
`zero3_hetero_shard.py` (all_reduce full flat buffer then slice), or
pad to max shard size.

## Constraint
Do NOT change forward count logic. Do NOT open new branches.
