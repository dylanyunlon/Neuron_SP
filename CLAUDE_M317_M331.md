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

## M317 Changes Applied
- REAL_GPU_BENCHMARK.py: 4 bugs fixed + MFU + NKI-FA logs + draw_plot (951→1141 lines)
- run_all_v2.sh: 3-phase 99-run experiment matrix for 2×A6000+H100
- CLAUDE_M317_M331.md: this document
