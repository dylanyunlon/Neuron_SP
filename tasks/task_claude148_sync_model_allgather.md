# Task C148: sync_shard_to_model must all_gather to all ranks

## Context
git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP

In `deepspeed/runtime/zero3_hetero_shard.py`, `sync_shard_to_model()` writes
only this rank's FP32 shard slice back to model BF16 params. But each rank
holds the FULL model on GPU. After optimizer.step(), only 1/N of each rank's
model copy gets updated — the rest is stale.

## Task
After the local shard-to-model copy loop in `sync_shard_to_model()`, add an
all_gather (or per-param broadcast) so all ranks get the full updated model.

Use `_build_full_buffer()` as reference for handling uneven shard sizes.
Update `sync_shard_to_model_async()` similarly — the all_gather must happen
synchronously (NCCL collectives cannot run on private streams across ranks).

## Constraint
Do NOT open new branches.
