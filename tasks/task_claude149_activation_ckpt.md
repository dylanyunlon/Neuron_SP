# Task C149: Enable activation checkpointing on A6000 tiers

## Context
git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP

Training log shows ALL layers on ALL ranks have `ckpt=OFF [OFF (master disabled)]`.
A6000 (47GB) needs activation checkpointing to fit 7B model with grad_accum > 1.

In `desloc_engine.py`, find where `[ActCkpt]` master flag is set.
The config says `activation_checkpointing: true` but master is forced False.

## Task
1. Find the activation checkpointing master flag in desloc_engine.py
2. Set it True for A6000 tier (SM8.6), keep False/selective for H100
3. Wrap transformer layers with `torch.utils.checkpoint.checkpoint` for
   A6000 ranks

## Constraint
Do NOT open new branches.
