# Task C154: Fix hetero_offload_throttle and hetero_cudagraph_optimizer registration

## Context
git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP

From log:
- "Hook registration failed for deepspeed.runtime.hetero_offload_throttle:
   DESLOCOffloadConfig.__init__() got an unexpected keyword argument 'base_cap_per_group'"
- "[register] engine.module is None; cannot build HeteroCudaGraphOptimizer."

## Task
1. In deepspeed/runtime/hetero_offload_throttle.py: fix DESLOCOffloadConfig
   to accept base_cap_per_group, or fix the caller
2. In deepspeed/runtime/hetero_cudagraph_optimizer.py: defer init to after
   engine.module is assigned (lazy build pattern)

## Constraint
Do NOT open new branches.
