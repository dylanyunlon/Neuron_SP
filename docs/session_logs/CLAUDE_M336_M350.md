# CLAUDE_M336_M350.md — Claude-30 Session
## Session: Claude-30 (M336-M350) | Base: commit 16b49c37 (M335)
## Theme: NCCL Protocol Integration + AutoSP×DES-LOC Orthogonality + Distributed Experiment Scheduling
## Machine: ecs.gn8v-2x.8xlarge (32vCPU, 192GiB, 2×NVIDIA H20 96GB HBM3, ¥71.58/h)
## Note: 阿里云gn8v系列的"GPU H"实际型号为NVIDIA H20 (Hopper阉割版, 非H100)
##   H20: FP32=39.5T, BF16=148T, FP8=296T, 96GB HBM3, 4TB/s BW, NVLink 900GB/s
##   对比H100 SXM: FP32=67T, BF16=989.5T → H20算力约为H100的15%
##   优势: 96GB大显存(>H100 80GB), 4TB/s高带宽, NVLink 900GB/s, 价格低40%

## 20 Infra Repos Cloned (tree + git branch surveyed)

| # | Repo | Org | Branch | Key Module for DES-LOC |
|---|------|-----|--------|----------------------|
| 1 | nccl | NVIDIA | master | `src/device/all_reduce.h` — Ring/Tree/NVLS AllReduce kernel specializations |
| 2 | Megatron-LM | NVIDIA | main | `core/distributed/distributed_data_parallel.py` — bucket化梯度通信 |
| 3 | cccl | NVIDIA | main | Thrust/CUB/libcudacxx — CUDA core primitives (8783 files) |
| 4 | TransformerEngine | NVIDIA | main | `pytorch/ops/fused/userbuffers_forward_linear.py` — comm/compute overlap |
| 5 | cutlass | NVIDIA | main | GEMM microkernel (Hopper/Blackwell, 6899 files) |
| 6 | apex | NVIDIA | master | `multi_tensor_apply` — fused gradient ops |
| 7 | DeepSpeed | Microsoft | master | `runtime/zero/stage_1_and_2.py` — our upstream (1919 files) |
| 8 | jax | Google | main | XLA compilation + pjit distributed (1872 files) |
| 9 | veScale | ByteDance | main | `dtensor/_collective_utils.py` — mesh_scatter_ragged |
| 10 | pytorch | Meta | main | `torch/distributed/` — FSDP + c10d backend (21718 files) |
| 11 | aws-neuron-sdk | AWS | master | Trainium chip docs + benchmark (2170 files) |
| 12 | neuronx-nemo-megatron | AWS | main | Megatron→Trainium port (3177 files) |
| 13 | neuronx-distributed | AWS | main | `optimizer/zero_redundancy_optimizer.py` — ZeRO-1 on Trainium |
| 14 | triton | OpenAI | main | GPU compiler (kernel fusion, 1592 files) |
| 15 | flash-attention | Dao-AILab | main | `flash_attn/utils/benchmark.py` — CUDA event timing |
| 16 | NKI-FA | dylanyunlon | main | `exp_utils/draw_plot.py` — NeurIPS figure pipeline |

## Nick Joseph 访谈 → 大厂代码映射 (NJ→Code Technical Chain)

### NJ-1: "写all-reduce通信,不能完全依赖现有包" (自己写分布式通信)

从 NCCL `src/device/all_reduce.h` 的 **Ring AllReduce ALLREDUCE_CHUNKSTEPS/SLICESTEPS 协议** 这个好例子开始。
然后,遵循该模式实现一个新的 **DES-LOC gated AllReduce dispatcher**,
让 `sync_if_needed()` 可以按 **Kx/Ku/Kv 三层独立调度通信**,并能 **跳过非同步步的AllReduce**。
接着 **Megatron `_ParamAndGradBuffer`** 引入 **bucket化梯度缓冲**,
使 DES-LOC 能够按 **bucket粒度判断是否需要通信**,同时 **NCCL NVLS算法** 优化 **NVLink intra-node 通信延迟**。
随后 **neuronx-distributed `NeuronZero1Optimizer._shard_parameters()`** 整合 **XLA AllReduce 路径**,
令 DES-LOC 支持 **Trainium2 cc.allreduce**,进而 **veScale DTensor redistribute** 增强 **异构worker间tensor再分布**。
最终 **TransformerEngine `userbuffers_forward_linear.py` comm_gemm_overlap** 完善 **通信计算重叠**,
确保 DES-LOC 兼容 **FP8混合精度通信+计算 overlap pipeline**,全面 **减少通信暴露时间** 升级 **端到端训练吞吐** 以达成 **170×通信量削减**。

### NJ-2: "用纸笔算出理论效率(MFU)" (roofline分析)

从 **FlashAttention `flash_attn/utils/benchmark.py:8` benchmark_forward()** 的 **CUDA event timing + warmup pattern** 这个好例子开始。
然后,遵循该模式实现一个新的 **DeslocCudaEventTimer**,
让 每个训练step 可以 **sub-microsecond精度分相(forward/backward/comm)计时**,并能 **区分warmup和steady-state**。
接着 **TransformerEngine `benchmark_dot_product_attention.py`** 引入 **TFLOPS测量标准**,
使 DES-LOC 能够 **自动检测GPU型号并查表peak TFLOPS**,同时 **FlashAttention benchmark_memory()** 优化 **peak memory tracking**。
随后 **cutlass GEMM roofline model** 整合 **arithmetic intensity分析**,
令 roofline 支持 **compute-bound vs memory-bound自动判定**,进而 **Megatron cuda_graphs.py is_graph_capturing()** 增强 **CUDA graph phase检测**。
最终 **NKI-FA `draw_plot.py` parse_data()** 完善 **NeurIPS-quality figure生成**,
确保 MFU报告 兼容 **seaborn whitegrid 300dpi serif-font格式**,全面 **标准化性能指标** 升级 **实验复现性** 以达成 **论文级质量MFU报告**。

## M336-M338 Changes Applied (Claude-30)

| M# | File | Lines +/- | Net | Key Additions |
|---|---|---|---|---|
| M336 | constants.py | +501/-22 | +479 | NCCL protocol (algo/proto/chunk), Megatron bucket sizing, AutoSP+DES-LOC compat, GPU instance DB, grad clip per-tier, roofline constants, NKI-FA extended log keys |
| M337 | comms_logging.py | +488/-0 | +488 | DeslocCommEvent (NCCL profiler_v3 pattern), DeslocCommProfiler (lifecycle), log_comm_event (Megatron config_logger), cl_parse_v2 (algo/proto), bandwidth_analysis, overlap_ratio |
| M338 | timer.py | +482/-0 | +482 | DeslocCudaEventTimer (FA benchmark.py), DeslocMemoryTracker (FA memory), DeslocStepMFUCalculator (TE+roofline), DeslocPhaseProfiler (Megatron DDP overlap) |

## M339-M350 Plan for Claude-31+

| M# | Target File | Source Repos | Task |
|---|---|---|---|
| M339 | deepspeed/comm/comm.py | veScale `_collective_utils.py`, nccl `collectives.cc`, Megatron `param_and_grad_buffer.py` | Gated AllReduce with tier routing + async bucket reduce |
| M340 | deepspeed/comm/backend.py | TransformerEngine `userbuffers_forward_linear.py`, nccl `enqueue.cc`, Megatron DDP overlap hooks | DeslocAsyncBackend with userbuffer overlap |
| M341 | deepspeed/comm/torch.py | Megatron `reduce_scatter_with_fp32_accumulation.py`, neuronx-distributed `comm.py`, veScale `_redistribute.py` | FP32 accumulation + tier-aware torch.distributed wrapper |
| M342 | deepspeed/runtime/zero/stage_1_and_2.py | Megatron `distrib_optimizer.py`, neuronx-distributed `zero_redundancy_optimizer.py` | ZeRO + DES-LOC hybrid step with Kx-gated reduce |
| M343 | deepspeed/ops/adam/fused_adam.py | Megatron `emerging_optimizers.py` (TensorParallelMuon), TransformerEngine `adam.cu` | DeslocFusedAdam with ρ-clipping + half-life tracking |
| M344 | deepspeed/runtime/bf16_optimizer.py | Megatron `grad_scaler.py` (DynamicGradScaler), Megatron `optimizer.py` | DeslocBF16GradScaler with Kx-boundary awareness |
| M345 | deepspeed/runtime/lr_schedules.py | Megatron `training_config.py`, Megatron `optimizer.py` | WSD schedule aligned with Kx boundaries |
| M346 | deepspeed/runtime/pipe/engine.py | Megatron `schedules.py`, Megatron `combined_1f1b.py` | DES-LOC Kx gating in pipeline grad sync + hetero MB |
| M347 | deepspeed/runtime/engine.py | Megatron DDP, DeepSpeed compile/passes, NKI-FA | AutoSP×DES-LOC integration in main engine loop |
| M348 | deepspeed/runtime/utils.py | Megatron `clip_grads.py`, neuronx-distributed, cutlass roofline | Per-bucket grad norm + adaptive Kx controller |
| M349 | deepspeed/runtime/config.py | Megatron `DistributedDataParallelConfig`, `OptimizerConfig` | AutoSP+DES-LOC unified config with validation |
| M350 | REAL_GPU_BENCHMARK.py | NKI-FA `draw_plot.py`, FlashAttention benchmark, all M336-M349 | Full 7-figure pipeline + distributed experiment runner |

## 分布式实验计划 (流浪地球计划)

### GPU Instance Selection Analysis
| Instance | Cost/hr | GPU Mem | vCPU | Best For |
|---|---|---|---|---|
| sgn8ia_m2 | ¥1.10 | 2GB | 4 | ❌ Too small |
| sgn8ia_m4 | ¥24.40 | 48GB | 8 | ✅ Single-GPU 125M/350M |
| gn8v_4x | ¥36.39 | 96GB | 16 | ✅ Single-GPU 700M/1.3B |
| gn8v_2x_8x | ¥71.58 | 192GB (2×96) | 32 | ✅✅ 2-GPU DDP/DESLOC 700M+ |
| gn8v_2x_12x | ¥76.98 | 192GB (2×96) | 48 | ✅ Same GPU, more CPU |

**Selected**: gn8v_2x_8xlarge — best ¥/GPU-hour for 2-GPU experiments.



### Claude 开发进度 (27位Claude分工)
| Claude# | M Range | Status |
|---|---|---|
| Claude-17 | M257-M271 | ✅ Cleanup + experiment scheduler |
| Claude-18 | M272-M286 | ✅ NeurIPS figure pipeline |
| Claude-19 | M287-M301 | ✅ Protocol v2 + Trainium2 |
| Claude-20 | M302-M316 | ✅ Hetero GPU scheduling |
| Claude-22 | M317 | ✅ Engine bucket AllReduce |
| Claude-23 | M318-M319 | ✅ Utils/config DES-LOC |
| Claude-24 | M320-M331(partial) | ✅ Shell + DDP baseline |
| Claude-25 | M332-M334 | ✅ 2xA100 DDP baseline |
| Claude-26-29 | M335 | ✅ 700M 3-seed results (7.31±0.49 vs 7.67±0.05) |
| **Claude-30** | **M336-M338** | **✅ NCCL protocol + comm profiler + MFU timer** |