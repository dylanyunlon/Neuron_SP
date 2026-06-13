# CLAUDE_M317_M331.md — Claude-22 Session
## Session: Claude-22 (M317+) | Base: commit b19e1a95 (HEAD)
## Theme: Fix Training Pipeline + NKI-FA-Grade Experiment Suite for 2xA6000+H100

## Large-Scale Infra Code Survey (15 Repos, tree+git branch)

| # | Repo | Org | Files | Key Module for DES-LOC |
|---|------|-----|-------|----------------------|
| 1 | nccl | NVIDIA | 481 | `src/device/all_reduce.h` — Ring AllReduce kernel |
| 2 | Megatron-LM | NVIDIA | 932 | `core/distributed/param_and_grad_buffer.py` — bucket化梯度通信 |
| 3 | cccl | NVIDIA | 3486 | Thrust/CUB/libcudacxx 基础原语 |
| 4 | TransformerEngine | NVIDIA | 529 | FP8 GEMM + comm_gemm_overlap |
| 5 | cutlass | NVIDIA | 2220 | GEMM微内核 (Hopper/Blackwell) |
| 6 | apex | NVIDIA | — | mixed precision + distributed utils |
| 7 | DeepSpeed | Microsoft | 1130 | `runtime/zero/stage_1_and_2.py` — 我们的上游 |
| 8 | jax | Google | 1244 | XLA编译 + pjit分布式 |
| 9 | veScale | ByteDance | 351 | DTensor redistribute — 分布式tensor |
| 10 | pytorch | Meta | — | FSDP + c10d通信后端 |
| 11 | aws-neuron-sdk | AWS | — | Trainium芯片文档+benchmark |
| 12 | neuronx-nemo-megatron | AWS | — | Megatron→Trainium移植 |
| 13 | neuronx-distributed | AWS | 388 | ZeRO-1 on Trainium + TP/PP |
| 14 | triton | OpenAI | 623 | GPU compiler (kernel fusion) |
| 15 | flash-attention | Dao-AILab | 833 | FA3 Hopper kernel (NKI-FA参考) |

## Nick Joseph → 大厂代码映射 (技术链)

### NJ-1: "写all-reduce通信,不能完全依赖现有包"
从 NCCL `src/device/all_reduce.h` 的 Ring AllReduce 开始。
遵循该模式实现 DES-LOC 的 **gated AllReduce**，
让 `sync_if_needed` 可以按 Kx/Ku/Kv 独立调度，并能跳过不需要同步的步骤。
接着 Megatron 引入 `_ParamAndGradBuffer` 的 bucket 化机制，
使 DES-LOC 能够按 bucket 粒度决定是否通信，同时 NCCL overlap 优化通信延迟。
随后 neuronx-distributed 整合 `NeuronZero1Optimizer`，
令 DES-LOC 支持 Trainium2 上的 XLA AllReduce，进而 veScale 增强 DTensor redistribute。
最终 TransformerEngine 完善 comm_gemm_overlap，
确保 DES-LOC 兼容 FP8 混合精度训练，全面升级通信效率以达成 170× reduction。

### NJ-2: "用纸笔算出理论效率(MFU)"
从 cutlass 的 GEMM roofline model 开始。
遵循该模式实现 DES-LOC 的 **MFU calculator**，
让 benchmark 可以报告实际 vs 理论 TFLOPS，并能对比不同 GPU (A6000 vs H100)。
接着 Megatron 引入 `profiling/` 的 per-step timing，
使 DES-LOC 能够分解 compute/comm/idle 三相时间，
同时 FlashAttention benchmark_attn.py 优化 TFLOPS 测量精度。
随后 TransformerEngine 整合 FP8 GEMM benchmark，
令 roofline 支持 FP8/BF16/FP16 多精度对比。
最终 NKI-FA draw_plot.py 完善可视化，
确保图表兼容 NeurIPS 审稿标准，全面升级 MFU 报告以达成论文级质量。

### NJ-3: "把分布式框架调到极致…数据并行、流水线并行、模型并行"
从 Megatron `core/distributed/distributed_data_parallel.py` 的 DDP 开始。
遵循该模式实现 DES-LOC 的 **异步梯度同步**，
让各 worker 可以独立训练 Kx 步再同步，并能保持收敛性。
接着 DeepSpeed 引入 ZeRO Stage 1/2 的分片优化器状态，
使 DES-LOC 能够在 ZeRO 基础上叠加 Kx gating，
同时 neuronx-distributed 优化 Trainium 上的 tensor parallel。
随后 veScale 整合 ByteDance 的 DTensor 自动分片，
令 DES-LOC 支持异构 GPU 拓扑 (2×A6000+H100)，
进而 FSDP 增强参数分片与通信重叠。
最终 Triton 完善 kernel fusion，
确保 DES-LOC 兼容 custom CUDA kernel 路径，
全面升级并行策略以达成万卡级可扩展性。

### NJ-4: "一个bug就可能让你搁置几个月"
从 DeepSpeed `REAL_GPU_BENCHMARK.py` 的 SyntheticDataset bug 开始（loss=10.825 flatline）。
遵循该模式实现 **可学习数据集** + sync 计数修复，
让实验可以产生有意义的 loss 曲线，并能在 500 步内看到收敛趋势。
接着 Megatron 引入 deterministic training 的 seed 控制，
使跨 seed 实验能够产生可复现的 mean±std 误差棒。
最终 NKI-FA 的 `### config ### metric: value` 格式完善日志解析，
确保 draw_plot.py 兼容所有实验输出。

## M317 Changes Applied (Claude-22)
- engine.py: +348 lines — Megatron-style bucket AllReduce, 3-tier momentum sync, NKI-FA logging
- REAL_GPU_BENCHMARK.py: 4 bugs fixed + MFU + NKI-FA logs + draw_plot (873→1141 lines)
- run_all_v2.sh: 3-phase 99-run experiment matrix for 2×A6000+H100

## Rules for Claude-23 (M318-M331)
1. MODIFY EXISTING FILES ONLY — no new standalone .py files
2. cat FILE FIRST — always read target before modifying
3. ast.parse AFTER — verify Python syntax after every modification
4. ZERO numpy.random — use torch.manual_seed only
5. Net +400 lines per M# (批判性merge: add N, delete M, net N-M≈400)
6. Data from NKI-FA logs only — no hardcoded dummy results
7. git clone the infra repos below, diff the EXACT file:line listed, then merge

## Infra Repos to Clone (Claude-22 already surveyed these)
```bash
cd /home/claude/infra_repos  # or wherever you want
git clone --depth 1 https://github.com/NVIDIA/nccl.git
git clone --depth 1 https://github.com/NVIDIA/Megatron-LM.git
git clone --depth 1 https://github.com/NVIDIA/TransformerEngine.git
git clone --depth 1 https://github.com/volcengine/veScale.git
git clone --depth 1 https://github.com/aws-neuron/neuronx-distributed.git
git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git
git clone --depth 1 https://github.com/dylanyunlon/NKI-FA.git
```

---

## M318: deepspeed/runtime/utils.py (2419→~2819, net +400)
**DELETE** lines 1485-1600: `desloc_half_life()`, `desloc_recommend_periods()`, `desloc_psi_factor()`, `desloc_comm_reduction_ratio()` — these are math-only simulations with no real training integration.
**DIFF FROM (cat these, adapt into utils.py):**

| Repo file:line | Function/Class | What to adapt |
|---|---|---|
| `Megatron-LM/megatron/core/optimizer/clip_grads.py:57` | `get_grad_norm_fp32()` | → `desloc_bucket_grad_norm(bucket)` per-bucket L2 norm |
| `Megatron-LM/megatron/core/optimizer/clip_grads.py` (search `clip_grad_by_total_norm`) | `clip_grad_by_total_norm_fp32()` | → `desloc_clip_grad_per_tier(params, max_norm, tier)` |
| `Megatron-LM/megatron/core/optimizer/distrib_optimizer.py:62` | `class Range` | → `desloc_param_range(start, end)` for bucket offset tracking |
| `Megatron-LM/megatron/core/optimizer/distrib_optimizer.py:118` | `_build_model_gbuf_param_range_map()` | → `desloc_build_param_range_map(model)` |
| `neuronx-distributed/src/neuronx_distributed/optimizer/zero_redundancy_optimizer.py:96` | `_clip_grad_norm()` | → `desloc_zero1_clip_with_Kx_gate()` |
| `TransformerEngine/examples/pytorch/comm_gemm_overlap/te_layer_with_overlap.py:175` | `_train()` | → `desloc_roofline_model(gpu_name, n_params)` |
| `Megatron-LM/megatron/core/optimizer/emerging_optimizers.py:154` | `class TensorParallelMuon` | → `desloc_adaptive_Kx(grad_variance, loss_trend)` |

---

## M319: deepspeed/runtime/config.py (1085→~1485, net +400)
**DELETE** unused desloc config stubs that just set defaults without validation.
**DIFF FROM:**

| Repo file:line | Function/Class | What to adapt |
|---|---|---|
| `Megatron-LM/megatron/core/distributed/distributed_data_parallel_config.py:10` | `@dataclass DistributedDataParallelConfig` | → `@dataclass DeslocDistributedConfig` with Kx/Ku/Kv + all DDP fields |
| `Megatron-LM/megatron/core/distributed/distributed_data_parallel_config.py:16` | `overlap_grad_reduce: bool` | → `desloc_overlap_grad_reduce` |
| `Megatron-LM/megatron/core/distributed/distributed_data_parallel_config.py:46` | `bucket_size: Optional[int]` | → `desloc_bucket_size` |
| `Megatron-LM/megatron/core/optimizer/optimizer_config.py:139` | `@dataclass OptimizerConfig` | → `DeslocOptimizerConfig` with tier periods, half_life |

---

## M320: deepspeed/runtime/constants.py (713→~1113, net +400)
**DELETE** ~100 lines of reserved slot comments and duplicate constant defs.
**DIFF FROM:**

| Repo file:line | What | What to adapt |
|---|---|---|
| `nccl/src/device/all_reduce.h:234` | `RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_RING>`, `ALLREDUCE_CHUNKSTEPS`, `ALLREDUCE_SLICESTEPS` | → `DESLOC_ALLREDUCE_CHUNK_SIZE`, `DESLOC_RING_PROTOCOL` |
| `nccl/src/collectives.cc:111` | `ncclAllReduce()` — `struct ncclInfo` fields | → `DESLOC_COMM_INFO_FIELDS` dict |
| `Megatron-LM/megatron/core/distributed/distributed_data_parallel_config.py:46` | `bucket_size` | → `DESLOC_DEFAULT_BUCKET_SIZE` |
| `Megatron-LM/megatron/core/distributed/distributed_data_parallel_config.py:51` | `pad_buckets_for_high_nccl_busbw` | → `DESLOC_BUCKET_PAD_ALIGNMENT` |

---

## M321: deepspeed/utils/comms_logging.py (716→~1116, net +400)
**DELETE** `desloc_classify_op()` dummy implementation (~lines 141-165).
**DIFF FROM:**

| Repo file | What | What to adapt |
|---|---|---|
| `nccl/plugins/profiler/example/nccl/profiler_v3.h` | profiler event structure: startColl, stopColl | → `DeslocCommEvent` dataclass |
| `nccl/plugins/profiler/example/event.h` | event lifecycle: init→start→stop | → `DeslocCommProfiler` class |
| `Megatron-LM/megatron/training/dgrad_logging.py` | gradient logging patterns | → `desloc_log_comm_event()` NKI-FA format |
| `Megatron-LM/megatron/core/config_logger.py` | `log_config_to_disk()` | → `desloc_export_comm_log()` |

---

## M322: deepspeed/utils/timer.py (723→~1123, net +400)
**DELETE** empty `_desloc_phases` dict, unused timer categories (~30 lines).
**DIFF FROM:**

| Repo file:line | Function | What to adapt |
|---|---|---|
| `flash-attention/flash_attn/utils/benchmark.py:8` | `benchmark_forward()` | → `DeslocStepTimer` with CUDA event timing + warmup |
| `flash-attention/flash_attn/utils/benchmark.py:30` | `benchmark_backward()` | → backward phase timing |
| `flash-attention/flash_attn/utils/benchmark.py:258` | `benchmark_memory()` | → peak memory tracking per step |
| `TransformerEngine/benchmarks/attention/benchmark_dot_product_attention.py:44` | `benchmark_dot_product_attention()` | → `desloc_benchmark_step()` with TFLOPS |
| `Megatron-LM/megatron/core/transformer/cuda_graphs.py` | `is_graph_capturing()` | → `DeslocMFUCalculator` roofline from GPU specs |

---

## M323: deepspeed/comm/comm.py (1029→~1429, net +400)
**DIFF FROM:**

| Repo file:line | Function/Class | What to adapt |
|---|---|---|
| `veScale/vescale/dtensor/_collective_utils.py:66` | `mesh_scatter_ragged()` | → `desloc_gated_allreduce(tensor, Kx, step)` |
| `nccl/src/collectives.cc:111` | `ncclAllReduce()` dispatch | → `desloc_tier_aware_reduce()` ring vs tree per tier |
| `Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py:~160` | `class _ParamAndGradBucketGroup` | → `desloc_async_bucket_reduce()` |
| `neuronx-distributed/src/neuronx_distributed/parallel_layers/comm.py:200` | `all_reduce()` | → Trainium-compatible path |
| `neuronx-distributed/src/neuronx_distributed/parallel_layers/comm.py:124` | `reduce_scatter()` | → `desloc_trainium_reduce_scatter()` |

---

## M324: deepspeed/comm/backend.py (net +400)
**DIFF FROM:**

| Repo file | What | What to adapt |
|---|---|---|
| `TransformerEngine/transformer_engine/pytorch/ops/fused/userbuffers_forward_linear.py` | userbuffers comm/compute overlap | → `DeslocAsyncBackend` |
| `nccl/src/enqueue.cc` | `ncclEnqueueCheck()` async enqueue | → `desloc_enqueue_allreduce()` |
| `Megatron-LM/megatron/core/distributed/distributed_data_parallel.py` | overlap_grad_reduce hooks in `__init__` | → `DeslocOverlapManager` |

---

## M325: deepspeed/comm/torch.py (net +400)
**DIFF FROM:**

| Repo file:line | Function/Class | What to adapt |
|---|---|---|
| `Megatron-LM/megatron/core/distributed/reduce_scatter_with_fp32_accumulation.py:9` | `class _ReduceScatterWithFP32AccumulationWorkHandle` | → `desloc_reduce_scatter_fp32()` |
| `Megatron-LM/megatron/core/distributed/reduce_scatter_with_fp32_accumulation.py:42` | `reduce_scatter_with_fp32_accumulation()` | → FP32 accumulation for DES-LOC |
| `neuronx-distributed/src/neuronx_distributed/parallel_layers/comm.py:163` | `all_gather()` | → `desloc_torch_tiered_allreduce()` |
| `neuronx-distributed/src/neuronx_distributed/parallel_layers/comm.py:200` | `all_reduce()` | → torch.distributed tier-aware wrapper |
| `veScale/vescale/dtensor/_redistribute.py:130` | `class Redistribute(torch.autograd.Function)` | → `DeslocRedistribute` autograd-safe |

---

## M326: deepspeed/runtime/zero/stage_1_and_2.py (3237→~3637, net +400)
**DIFF FROM:**

| Repo file:line | Function/Class | What to adapt |
|---|---|---|
| `Megatron-LM/megatron/core/optimizer/distrib_optimizer.py:2696` | `step_with_ready_grads()` | → ZeRO + DES-LOC hybrid step |
| `Megatron-LM/megatron/core/optimizer/distrib_optimizer.py:62` | `class Range` | → partition range for Kx-gated reduce |
| `Megatron-LM/megatron/core/optimizer/distrib_optimizer.py:118` | `_build_model_gbuf_param_range_map()` | → param-to-bucket mapping |
| `neuronx-distributed/src/neuronx_distributed/optimizer/zero_redundancy_optimizer.py:59` | `_shard_parameters()` | → `desloc_zero_shard_with_Kx()` |
| `neuronx-distributed/src/neuronx_distributed/optimizer/zero_redundancy_optimizer.py:270` | `_reduce_gradients()` (in NeuronEPZero1Optimizer) | → `desloc_zero_reduce_with_Kx()` |

---

## M327: deepspeed/ops/adam/fused_adam.py (244→~644, net +400)
**DIFF FROM:**

| Repo file:line | Function/Class | What to adapt |
|---|---|---|
| `Megatron-LM/megatron/core/optimizer/emerging_optimizers.py:154` | `class TensorParallelMuon` | → `DeslocFusedAdam` with ρ-clipping |
| `Megatron-LM/megatron/core/optimizer/emerging_optimizers.py:179` | `scaled_orthogonalize_fn` (Newton-Schulz) | → `desloc_coordinate_clip()` |
| `Megatron-LM/megatron/core/optimizer/emerging_optimizers.py:227` | `orthogonalize()` | → half-life tracking for Ku/Kv |
| `TransformerEngine/transformer_engine/common/multi_tensor/adam.cu` | CUDA fused Adam kernel | → Python fallback with same semantics |

---

## M328: deepspeed/runtime/bf16_optimizer.py (727→~1127, net +400)
**DIFF FROM:**

| Repo file:line | Function/Class | What to adapt |
|---|---|---|
| `Megatron-LM/megatron/core/optimizer/grad_scaler.py:61` | `class DynamicGradScaler` | → `DeslocBF16GradScaler` aware of Kx boundaries |
| `Megatron-LM/megatron/core/optimizer/grad_scaler.py:84` | `__init__()` with hysteresis, growth_interval | → loss scaling + Kx sync |
| `Megatron-LM/megatron/core/optimizer/optimizer.py` | BF16 master weight management | → `desloc_bf16_sync_master_weights()` |

---

## M329: deepspeed/runtime/lr_schedules.py (975→~1375, net +400)
**DIFF FROM:**

| Repo file | What | What to adapt |
|---|---|---|
| `Megatron-LM/megatron/training/config/training_config.py` | LR schedule fields (warmup, decay, min_lr) | → `DeslocWSDSchedule` aligned with Kx |
| `Megatron-LM/megatron/core/optimizer/optimizer.py` | LR schedule in `step()` | → `desloc_lr_at_boundary()` |

---

## M330: deepspeed/runtime/pipe/engine.py (1467→~1867, net +400)
**DIFF FROM:**

| Repo file:line | Function | What to adapt |
|---|---|---|
| `Megatron-LM/megatron/core/pipeline_parallel/schedules.py:592` | `forward_backward_no_pipelining()` | → DES-LOC Kx gating in pipe grad sync |
| `Megatron-LM/megatron/core/pipeline_parallel/schedules.py:460` | `backward_step()` | → `desloc_pipe_backward_with_Kx()` |
| `Megatron-LM/megatron/core/pipeline_parallel/schedules.py:804` | `get_pp_rank_microbatches()` | → hetero microbatch for A6000+H100 |
| `Megatron-LM/megatron/core/pipeline_parallel/combined_1f1b.py` | combined 1F1B schedule | → `desloc_pipe_hetero_schedule()` |

---

## M331: REAL_GPU_BENCHMARK.py (1141→~1541, net +400)
**DELETE** M315 simulation at EOF: `desloc_abl_cfgs()`, `desloc_run_mx()`, `desloc_hw_rep()` (~80 lines).
**DIFF FROM:**

| Repo file | What | What to adapt |
|---|---|---|
| `NKI-FA/exp_utils/draw_plot.py` (commit da964f3) | `parse_data()` regex parser, seaborn barplot with `f"{value:.1f}"` annotations | → full 7-figure pipeline |
| `NKI-FA/hopper/benchmark_attn.py` | `time_fwd()` (Triton do_bench), `flops()` | → `desloc_compute_tflops()` |
| `flash-attention/flash_attn/utils/benchmark.py:8` | `benchmark_forward()` CUDA event timing | → integrate into experiment runner |
| `flash-attention/flash_attn/utils/benchmark.py:258` | `benchmark_memory()` peak memory | → per-method memory tracking |
