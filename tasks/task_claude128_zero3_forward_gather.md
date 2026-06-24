# Claude-128: ZeRO-3 Forward All-Gather Hook

## Context
With ZeRO-3 sharding (Claude-127), each rank only has 1/N params. Before forward pass, need to all-gather full params temporarily.

## Task
1. Create `ZeRO3ForwardHook` that wraps each `nn.Module` with `register_forward_pre_hook` and `register_forward_hook`
2. Pre-hook: `dist.all_gather_into_tensor` to reconstruct full params for that layer
3. Post-hook: free the gathered params, keep only local shard
4. Wire into DesLocEngine.train() — wrap model with hooks before training loop

## Files
- `deepspeed/runtime/zero3_hetero_shard.py` (extend from Claude-127)
- `deepspeed/runtime/desloc_engine.py` — call hook registration

## Key detail
Layer-by-layer gather, not all-at-once. Only gather params for current layer being computed. This keeps peak memory = model_shard + 1_full_layer instead of full_model.

## Git
Branch: main. Commit with `--signoff`. Author: `dylanyunlon <dogechat@163.com>`.
Token: `SEE_CONFIG`
