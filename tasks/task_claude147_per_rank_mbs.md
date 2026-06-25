# Task C147: Wire per-rank micro_batch_size into train loop

## Context
git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP

PartitionPlan has `micro_batch_sizes: Dict[int, int]` with per-device values,
but the train loop in desloc_engine.py uses only `cfg.micro_batch_size` (single
value). H100 should use micro_batch_size=4, A6000 should use 1.

## Task
In `desloc_engine.py` train():
1. Look up this rank's micro_batch_size from `self.plan.micro_batch_sizes`
2. Use it when reading from data iterator (batch size)
3. Pass correct value for loss scaling: `loss / num_microbatches`
4. Fix tokens_per_step accounting: `num_microbatches * per_rank_mbs * seq_len`

Also update `forward()` to accept variable batch sizes.

## Constraint
Do NOT open new branches. Do NOT change num_microbatches (keep uniform).
