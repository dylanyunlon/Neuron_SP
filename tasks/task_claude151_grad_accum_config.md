# Task C151: Set reasonable grad_accum_steps in launch scripts and config

## Context
git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP

After fixing PartitionSolver (C146), grad_accum uses cfg.grad_accum_steps
which defaults to 8. With 3 GPUs, micro_bs=1, seq_len=1024:
  global_batch = 3 * 1 * 8 * 1024 = 24576 tokens/step — reasonable.

But the launch script `launch_7b_3gpu.sh` doesn't pass --grad-accum-steps,
and configs/7b_commitpack.yaml may not have it.

## Task
1. In launch_7b_3gpu.sh: add --grad-accum-steps 4
2. In configs/7b_commitpack.yaml: set training.grad_accum_steps: 4
3. In run_pretrain.py: verify --grad-accum-steps is parsed correctly

## Constraint
Do NOT open new branches.
