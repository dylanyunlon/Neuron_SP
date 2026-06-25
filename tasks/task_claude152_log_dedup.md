# Task C152: Deduplicate training logs (3x per message)

## Context
git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP

Every logger.info() prints 3 times (once per rank). Fix by setting non-rank-0
loggers to WARNING level after dist.init_process_group().

## Task
1. In desloc_engine.py train() start: if rank > 0, set deepspeed logger to WARNING
2. In run_pretrain.py after init: same
3. Keep _is_main guard for print() statements

## Constraint
Do NOT open new branches.
