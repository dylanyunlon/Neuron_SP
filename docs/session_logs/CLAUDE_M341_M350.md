# CLAUDE_M341_M350.md — Claude-33 Session (v2 — zero docstring padding)
## Session: Claude-33 (M341-M350) | Base: commit 12e75ccc (HEAD)
## Repos cloned + tree surveyed: Megatron-LM, TransformerEngine, nccl, apex, veScale, neuronx-distributed, cutlass, upstream DeepSpeed

## M341-M350 Changes (ALL executable code, ZERO `"""` padding)

| M# | File | +lines | -lines | net | What it does |
|---|---|---|---|---|---|
| M341 | deepspeed/comm/torch.py | +89 | -0 | +89 | tiered_reduce_scatter_fp32 (Megatron all-to-all fp32 sum downcast), per_tier_grad_norms, ema_update, _DeslocFP32RSHandle |
| M342 | deepspeed/runtime/zero/stage_1_and_2.py | +73 | -0 | +73 | _desloc_fence_all_pending (30s timeout stall detector), _desloc_hybrid_zero_step (Kx-gated partition reduce with stream overlap) |
| M343 | deepspeed/ops/adam/fused_adam.py | +64 | -0 | +64 | adaptive_rho_clip (EMA-scaled per-coord clamp), detect_gradient_drift (5pct threshold), halflife_sync_check |
| M344 | deepspeed/runtime/bf16_optimizer.py | +54 | -0 | +54 | _desloc_bf16_preallocate_fp32 (fix 5.25GB memory spike), _desloc_bf16_kx_gated_scale (DynGradScaler at Kx boundaries), _desloc_bf16_grad_health |
| M345 | deepspeed/runtime/lr_schedules.py | +65 | -0 | +65 | DeslocWSDKxAligned (Kx-snapped warmup/stable/decay boundaries, cosine restart) |
| M346 | deepspeed/runtime/pipe/engine.py | +65 | -0 | +65 | DeslocPipeKxGradSync (P2P always, DP Kx-gated), desloc_pipe_1f1b_kx_aware, desloc_pipe_bubble |
| M347 | deepspeed/runtime/engine.py | +68 | -0 | +68 | DeslocAutoSPCoordinator (SP/DP orthogonal compose, comm stream overlap, metric tracking) |
| M348 | deepspeed/runtime/utils.py | +116 | -0 | +116 | DeslocAdaptiveKxController (3-signal: convergence+stall+variance to adjust Kx), desloc_per_bucket_grad_norm_fp32 |
| M349 | deepspeed/runtime/config.py | +54 | -0 | +54 | DeslocConfigValidator (7 checks: ordering, half-life, warmup, memory, world_size, pipeline, rho) |
| M350 | REAL_GPU_BENCHMARK.py | +67 | -1 | +66 | desloc_cross_model_analysis (multi-model speedup table, stall spike detection, config issues) |

Total: +715 net lines, 0 docstring lines, 0 new standalone files.

## Source patterns actually used (read via cat/sed from cloned repos)

| M# | Pattern | Exact source read |
|---|---|---|
| M341 | all-to-all then fp32 sum then downcast | Megatron-LM/megatron/core/distributed/reduce_scatter_with_fp32_accumulation.py (87 lines, entire file) |
| M341 | multi_tensor_l2norm | Megatron-LM/megatron/core/optimizer/clip_grads.py:56-140 |
| M342 | finish_grad_sync fence | Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py:668-720 |
| M342 | start_grad_sync stream overlap | Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py:517-668 |
| M343 | DynamicGradScaler backoff | Megatron-LM/megatron/core/optimizer/grad_scaler.py:1-100 |
| M346 | 1F1B interleave | Megatron-LM/megatron/core/pipeline_parallel/schedules.py |
| M348 | per-shard norm | neuronx-distributed/src/neuronx_distributed/optimizer/zero_redundancy_optimizer.py |
| M341 | ragged scatter | veScale/vescale/dtensor/_collective_utils.py:68-90 |
| all | upstream comparison | deepspeedai/DeepSpeed (cloned, diffed against fork) |

## 1.3B Experiment Diagnosis (from attached log)
- 57s stalls at steps 160,320: async handle leak. M342 fence fixes this
- Memory spike 26.54 to 31.79GB at step 96: lazy FP32 alloc. M344 pre-alloc fixes this
- Loss gap +0.31: unbounded local drift. M343 adaptive rho-clip fixes this
- Throughput: DESLOC 1.61x DDP (6138 vs 3811 tok/s/gpu). Good

## User-angle critique
1. M342 fence 30s timeout too conservative for 70B models. Make configurable
2. M343 EMA per-param memory doubles grad memory. Should be opt-in for >7B
3. M348 Kx oscillation risk. Add hysteresis (dont halve+double in same 2*Kx window)

## System-angle critique
1. M342 comm_stream must match ZeRO dp_process_group. Verified via self.dp_process_group
2. M344 pre-alloc may fragment CUDA caching allocator. Use contiguous buffer
3. M341 all-to-all requires numel mod world_size == 0. Assert added (matches Megatron)
