# Claude-133: WandB + TensorBoard Logging for DES-LOC Training

## Context
Training runs produce no visible metrics. Need logging for the paper.

## Task
1. In `desloc_engine.py` train(), add optional wandb and tensorboard logging
2. Log every `log_every` steps (rank 0 only):
   - `train/loss` (averaged over grad_accum steps)
   - `train/lr` (current learning rate)
   - `train/grad_norm` (before clipping)
   - `train/throughput_tokens_per_sec` (tokens processed / wall time)
   - `train/gpu0_mem_gb`, `train/gpu1_mem_gb`, `train/gpu2_mem_gb` (allocated)
   - `train/mfu` (model FLOPs utilization estimate)
3. wandb init: project="neuron_sp_desloc", run name from timestamp
4. tensorboard: write to `logs/tb_{timestamp}/`
5. Controlled by `--wandb` and `--tensorboard` CLI flags (both default off)

## Files
- `deepspeed/runtime/desloc_engine.py`
- `run_pretrain.py` — add CLI flags, pass to TrainingConfig

## Git
Branch: main. Commit with `--signoff`. Author: `dylanyunlon <dogechat@163.com>`.
Token: `SEE_CONFIG`
