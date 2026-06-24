# Claude-131: Fix 3B Model Training Loop End-to-End

## Context
3B model (h=3200, L=26, heads=32) was added to _MODEL_CONFIGS. Need to ensure the full train loop works: forward → backward → optimizer.step → log → repeat.

## Task
1. Run `torchrun --nproc_per_node=3` with `--model-size 3b` on CUDA_VISIBLE_DEVICES=2,3,4
2. Fix any crashes in the training loop (device mismatch, NCCL hangs, etc)
3. Ensure all 3 ranks complete at least 10 steps without crash
4. Print loss every 10 steps (rank 0 only)
5. If NCCL hangs: check that all ranks enter the same collective ops in the same order. The mimo_loop path and non-mimo path must be consistent across ranks.

## Key issue to watch
`self.mimo_loop` might be None on some ranks but not others → deadlock on NCCL collectives. Ensure all ranks take the same code path.

## Files
- `deepspeed/runtime/desloc_engine.py` — train() method
- `run_pretrain.py` — model creation, data iter

## Git
Branch: main. Commit with `--signoff`. Author: `dylanyunlon <dogechat@163.com>`.
Token: `SEE_CONFIG`
