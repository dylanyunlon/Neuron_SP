# Task C150: Fix HeteroStepBatchScheduler to produce uniform num_microbatches

## Context
git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP

`deepspeed/runtime/hetero_step_batch_scheduler.py` HeteroMicrobatchAllocator
distributes microbatches by weight, producing different per-device counts
(e.g. {0: 6, 1: 1, 2: 1}). But efa47b85 made the engine use a single global
`allocation.num_microbatches` — so the per-device assignments are ignored.

The scheduler still sets `per_rank_microbatches` with different values, and
`DEFAULT_DES_LOC_DEVICE_PROFILES` has `capacity_weight=6.0` for H100.

## Task
Align the scheduler output: `per_device_assignment` should give the SAME
value to all devices (= num_microbatches). Differentiate via micro_batch_size
per device, not forward count. Remove or deprecate `per_rank_microbatches`.

## Constraint
Do NOT open new branches.
