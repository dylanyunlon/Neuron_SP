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

### NJ-3: "把分布式框架调到极致…数据并行、流水线并行、模型并行"

从 **Megatron `core/distributed/distributed_data_parallel.py` DistributedDataParallel.__init__()** 的 **bucket_size自动缩放(max(40M, 1M×dp_size))** 这个好例子开始。
然后,遵循该模式实现一个新的 **DES-LOC tier-aware bucket sizing**,
让 不同通信层 可以 **使用不同bucket大小(param 0.5×, mom1 1.0×, mom2 2.0×)**,并能 **按NCCL总线带宽对齐padding(2^16)**。
接着 **DeepSpeed `runtime/zero/stage_1_and_2.py`** 引入 **ZeRO-1分片优化器状态**,
使 DES-LOC 能够 **在ZeRO stage 0/1基础上叠加Kx gating**,同时 **neuronx-distributed** 优化 **Trainium TP/PP支持**。
随后 **veScale `dtensor/_redistribute.py`** 整合 **ByteDance的DTensor自动分片**,
令 DES-LOC 支持 **异构GPU拓扑(2×A6000+H100)**,进而 **PyTorch FSDP** 增强 **参数分片与通信重叠**。
最终 **Triton kernel fusion** 完善 **custom CUDA kernel路径**,
确保 DES-LOC 兼容 **fused optimizer + comm kernel**,全面 **统一并行策略** 升级 **万卡级可扩展性** 以达成 **线性扩展效率**。

### NJ-4: "一个bug就可能让你搁置几个月"

从 **M335 warmup u-sync bug (100→5)** 的 **95次多余AllReduce修复** 这个好例子开始。
然后,遵循该模式实现一个新的 **DES-LOC validation suite**,
让 每个sync boundary 可以 **自动验证loss monotonicity和gradient norm bounds**,并能 **检测数值发散(NaN/Inf)**。
接着 **Megatron `DynamicGradScaler`** 引入 **hysteresis-based loss scaling**,
使 DES-LOC 能够 **在Kx boundary做deterministic grad scaling**,同时 **NCCL profiler_v3.h event lifecycle** 优化 **通信事件追踪精度**。
随后 **NKI-FA `### config ### metric: value` format** 整合 **结构化日志标准**,
令 所有实验 支持 **自动化log解析和multi-seed聚合(mean±std)**,进而 **FlashAttention benchmark.py** 增强 **reproducible timing**。
最终 **Megatron deterministic training seed控制** 完善 **跨seed实验可复现性**,
确保 3-seed实验 兼容 **NeurIPS审稿标准(error bars)**,全面 **消除隐蔽bug** 升级 **实验可信度** 以达成 **生产级可靠性**。

### NJ-5: "芯片计算错误检测" + "不同芯片之间差异"

从 **Megatron `DistributedDataParallelConfig` @dataclass** 的 **配置验证__post_init__()** 这个好例子开始。
然后,遵循该模式实现一个新的 **DES-LOC HeteroGPU config**,
让 多芯片集群 可以 **自动检测GPU型号/显存/带宽并推荐Kx**,并能 **按计算能力比例分配micro-batch**。
接着 **neuronx-distributed** 引入 **XLA device抽象**,
使 DES-LOC 能够 **透明支持CUDA/XLA/NeuronCore后端**,同时 **aws-neuron-sdk** 优化 **Trainium2 NKI kernel路径**。
随后 **jax pjit** 整合 **device mesh分布式编译**,
令 DES-LOC 支持 **TPU/GPU混合训练**,进而 **accelerator/cuda_accelerator.py** 增强 **12-SKU GPU数据库**。
最终 **TransformerEngine FP8** 完善 **混合精度路径**,
确保 DES-LOC 兼容 **H100 FP8 + A100 BF16 异构精度**,全面 **统一多芯片抽象** 升级 **硬件利用率** 以达成 **跨芯片最优配置**。

### NJ-6: "AutoSP沿sequence维度切分, DES-LOC沿worker维度控制通信频率 → 正交"

从 **DeepSpeed AutoSP `compile/passes/sp_compile.py` prepare_autosp_inputs()** 的 **sequence维度自动分片** 这个好例子开始。
然后,遵循该模式实现一个新的 **DES-LOC × AutoSP 联合调度器**,
让 每个worker 可以 **在Kx步内独立做sequence-parallel forward/backward**,并能 **只在Kx边界执行AllReduce**。
接着 **Megatron `core/pipeline_parallel/schedules.py` forward_backward_no_pipelining()** 引入 **pipeline-aware同步点**,
使 DES-LOC 能够 **在PP stage boundary和Kx boundary对齐同步**,同时 **DeepSpeed ZeRO stage 0** 优化 **与AutoSP的兼容性(两者都要求stage 0)**。
随后 **Megatron `core/optimizer/distrib_optimizer.py` step_with_ready_grads()** 整合 **梯度就绪检测**,
令 DES-LOC+AutoSP 支持 **异步梯度reduce与SP comm重叠**,进而 **veScale DTensor** 增强 **sequence shard placement**。
最终 **Megatron `emerging_optimizers.py` TensorParallelMuon** 完善 **TP-aware Newton-Schulz正交化**,
确保 DES-LOC ρ-clipping 兼容 **Muon momentum正交化pipeline**,全面 **统一SP+DEC并行策略** 升级 **长序列训练效率** 以达成 **DES-LOC×AutoSP正交组合的1.5×加速**。

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

## AutoSP × DES-LOC Integration Architecture

```
AutoSP (sequence parallel) ←→ DES-LOC (worker-level comm gating)
    │ splits input along seq dim=1     │ gates comm along worker dim=0
    │ requires ZeRO stage 0            │ works with ZeRO stage 0/1
    │ uses torch.compile + inductor    │ uses manual AllReduce gating
    │                                  │
    └──────────── ORTHOGONAL ──────────┘
                    │
    Combined flow per worker:
    for step in range(total_steps):
        # AutoSP: shard input along sequence
        input_ids = prepare_autosp_inputs(batch, seq_dim=1)
        # DES-LOC: check if this step requires AllReduce
        if step % Kx == 0:  # Kx boundary
            allreduce(param_grads)     # sync params
        if step % Ku == 0:  # Ku boundary
            allreduce(momentum_states)  # sync momentum
        if step % Kv == 0:  # Kv boundary
            allreduce(variance_states)  # sync variance
        # Local step: forward + backward (SP-parallel, comm-free)
        loss = engine(input_ids)
        engine.backward(loss)
        engine.step()  # local optimizer update
```

## Rules for Claude-31 (M339+)
1. **MODIFY EXISTING FILES ONLY** — no new standalone .py files
2. **cat FILE FIRST** — always read target before modifying
3. **ast.parse AFTER** — verify Python syntax after every modification
4. **ZERO numpy.random** — use torch.manual_seed only
5. **Net +400 lines per M#** (批判性merge: add N, delete M, net ≈ 400)
6. **Clone repos before merge** — git clone --depth 1, tree, git branch
7. **NKI-FA FORMAT** — logs use `### config ### \n metric: value` format
8. **AutoSP compat** — DES-LOC must NOT break AutoSP (both ZeRO stage 0)

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

### Experiment Budget (流浪地球: 最大化榨取)
- 700M model: ~20min/seed × 3 seeds × 7 Kx × 4 methods = 1680 min ≈ 28 GPU-hours
- 1.3B model: ~60min/seed × 3 seeds × 4 Kx × 4 methods = 2880 min ≈ 48 GPU-hours
- Total: ~76 GPU-hours × ¥71.58/hr = ¥5,440 estimated

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
| Claude-31 | M339-M341 | 🔲 comm/backend/torch integration |
| Claude-32 | M342-M344 | 🔲 ZeRO+DES-LOC + fused_adam + bf16 |
| Claude-33 | M345-M347 | 🔲 LR schedule + pipe engine + main engine |
| Claude-34 | M348-M350 | 🔲 utils + config + benchmark |

## 批判性分析 (《计算机程序设计艺术》视角)

### 1. 用户角度Bug风险
- **M336 DESLOC_TIER_BUCKET_SCALE**: 将param tier bucket缩小到0.5×可能导致latency-bound comm。缓解: 添加了DESLOC_MIN_BUCKET_SIZE=1M下限。
- **M337 DeslocCommProfiler max_events=100000**: 长时间训练可能内存泄漏。缓解: 设置上限并在to_nkifa_str中懒序列化。
- **M338 DeslocCudaEventTimer**: torch.cuda.synchronize()在每个phase boundary会阻塞pipeline。缓解: 仅在warmup后启用,生产模式可关闭。

### 2. 系统角度批判
- **AutoSP兼容性**: AutoSP要求ZeRO stage 0, DES-LOC在stage 0/1工作。stage 1场景需验证AutoSP的torch.compile与DES-LOC的手动AllReduce gating不冲突。
- **NCCL algo选择**: M336中DESLOC_TIER_ALGO_MAP硬编码Ring/Tree选择, 但实际最优算法取决于cluster topology (NVSwitch vs PCIe)。应在runtime根据NCCL环境变量动态选择。
- **Gradient clipping at Kx boundary**: M336 DESLOC_CLIP_AT_KX_BOUNDARY_ONLY=True意味着非boundary步的gradient可能unclipped爆炸。需要在fused_adam中增加per-step local clip作为安全网。

## Claude-29 Bugfix Patch Applied

已成功 `git apply claude29_bugfix.patch`, 修复了2个致命bug:

1. **REAL_GPU_BENCHMARK.py comm_bytes计算**: 旧版用单一`syncs`计数器混合3个tier, 新版拆分为`sync_x/sync_u/sync_v`, 并修正了warmup期间的`eKx`边界判断(`eKx <= 1`条件)
2. **deepspeed/runtime/utils.py desloc_comm_reduction_ratio()**: 旧版简单用`total_steps // Kx`估算, 新版用逐步模拟匹配REAL_GPU_BENCHMARK的精确调度(包括warmup ramp和v-on-x piggyback)

Patch同时引入了关键新模块:
- **deepspeed/comm/torch.py +1197行**: M325 完整实现 — `DeslocTieredAllReduceTorch`, `DeslocAsyncBucketManager`, `DeslocCommOverlap`, `DeslocSequenceParallelComm`, `DeslocHeteroBalancer`, `init_desloc_torch_extensions()`
- **deepspeed/ops/adam/fused_adam.py +751行**: M327 完整实现 — `DeslocCoordinateClipper`, `DeslocFusedAdam`, `DeslocHalfLifeCalc`, `DeslocMomentumInterpolator`, `DeslocGradAccumulator`
- **deepspeed/runtime/utils.py +62行**: `desloc_comm_bytes()` 精确通信量计算

## 500步实验分析 (ags1: 2×A6000 + 1×H100 NVL)

### 实验设计批判 (《计算机程序设计艺术》视角)
**关键问题: 500步是否足以验证收敛性?**

从Figure 3的证据看:
- **Perplexity 100→20的平稳过渡**在~1000步完成, 500步只能观测到前半段衰减
- 但500步**足以检测致命bug**(loss explosion, NaN, 不收敛), 这正是smoke test的目的
- Figure 3(a)显示Kx=256时perplexity在3000步稳定到17.6±0.6, 而Kx=1536/3072则发散到260/294 → **500步足以区分收敛vs发散**
- Figure 5左图(billion-scale)显示DES-LOC在5小时(~2500步)达到perplexity~10, DDP需要15小时 → **2.2×加速**

### 500步足够验证的方面:
1. ✅ 通信正确性 (AllReduce不死锁)
2. ✅ Loss单调下降 (非发散)
3. ✅ 3-GPU异构DDP工作 (A6000+H100 mixed)
4. ✅ Kx gating生效 (通信量减少)
5. ✅ MFU测量合理

### 500步不足以验证的方面:
1. ❌ 最终收敛质量 (需3000+步, 参Figure 3)
2. ❌ Kx过大导致的延迟发散 (Figure 3(c) Ku=1024开始发散在~1500步)
3. ❌ FAVG+OPT的activation growth (Figure 5(b)在10000+步才显现)
4. ❌ ICL downstream评估 (Table 1需完整训练)

### Figure 3 关键洞察 (指导M339+实现):
- **(a) Kx最敏感**: Kx=256→17.6ppl, Kx=1024→20.1ppl, Kx=1536→260ppl → **Kx不应超过1024**
- **(b) Kv最不敏感**: Kv从256到3072, ppl仅从17.4到17.4 → **Kv可以非常大(variance半衰期长)**
- **(c) Ku中等敏感**: Ku=16/32→13.7/13.9ppl, Ku=256→39.4ppl → **Ku应≤3×Kx**
- **(d) Ku在大Kx下不敏感**: Kx=256时Ku从256到3072, ppl 17.6→17.7 → **大Kx下momentum同步不关键**

这验证了我们M336中设定的:
- `DESLOC_DEFAULT_KU_MULT = 3` (Ku = 3×Kx) ← Figure 3(c)支持
- `DESLOC_DEFAULT_KV_MULT = 6` (Kv = 6×Kx) ← Figure 3(b)支持
- `DESLOC_MAX_KX = 256` ← Figure 3(a)支持, 但实际可安全用到~512

### Figure 5 关键洞察:
- DES-LOC (8.96±0.22) vs DDP (8.45±0.18): **仅6%ppl代价换2.2×加速**
- DES-LOC activation L2 (0.12) 远低于FAVG+OPT (0.44) → **训练稳定性优势**
- DES-LOC与Local Adam ppl完全一致 (8.96 vs 8.96) 但通信量170×less
