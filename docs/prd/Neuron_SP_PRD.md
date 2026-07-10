# Neuron_SP — Product Requirements Document

> **Modeled after NVIDIA/CCCL Project #6 (1615 items)**
> 
> DES-LOC + AutoSP heterogeneous training · 5-GPU (2×A6000 + H100 NVL + 2×Blackwell) · NeurIPS 2026

## Summary

| Metric | Count |
|--------|-------|
| **Total items** | **182** |
| Modules | 13 |
| THEMEs | 24 |
| EPICs | 33 |
| FEAs | 123 |
| BUGs | 14 |
| DOCs | 12 |
| P0 (critical) | 63 |
| P1 (important) | 84 |
| P2 (nice-to-have) | 35 |

## Module Breakdown

| Module | Items | Description |
|--------|-------|-------------|
| `core/parallel_state` | 23 | Tier-aware TP/PP/DP/SP/CP process group management for heterogeneous GPU cluster… |
| `desloc_engine` | 34 | DES-LOC heterogeneous training engine — runtime layer scheduling, partition, and… |
| `autosp` | 19 | Automatic Sequence Parallelism — dynamic SP degree detection, fusion, and hetero… |
| `csrc` | 23 | C++/CUDA kernels for heterogeneous multi-GPU training — fused ops, communication… |
| `core/transformer` | 13 | Megatron TransformerLayer with heterogeneous attention, MLP, and tier-specific p… |
| `core/distributed` | 7 | DDP + gradient sync + tier-aware bucketing for heterogeneous training… |
| `core/optimizer` | 8 | Heterogeneous optimizer with ZeRO-3, tier-proportional sharding, and mixed preci… |
| `core/pipeline_parallel` | 6 | Heterogeneous 1F1B schedule with tier-aware stage assignment and P2P communicati… |
| `core/datasets` | 8 | CommitPack data pipeline + Megatron indexed dataset for code pretraining… |
| `hetero_bridge` | 6 | Integration layer connecting DES-LOC engine with Megatron core modules and DeepS… |
| `benchmarks` | 15 | Performance benchmarks, MFU tracking, and ablation experiments for NeurIPS 2026… |
| `infra` | 13 | CI/CD, testing infrastructure, build system, and developer tooling… |
| `paper` | 7 | NeurIPS 2026 paper: FAUST — Flexible Asynchronous Unified Scheduling for Trainin… |

---

## Module: `core/parallel_state`

> Tier-aware TP/PP/DP/SP/CP process group management for heterogeneous GPU clusters

### [THEME] Heterogeneous Process Group Topology

| # | Type | Priority | Title | Labels |
|---|------|----------|-------|--------|
| 1 | EPIC | P0 | [EPIC] Tier-aware process group initialization for mixed GPU clusters | `core/parallel_state`, `megatron-migration`, `P0` |
| 2 | EPIC | P0 | [EPIC] Dynamic topology discovery and runtime GPU tier classification | `core/parallel_state`, `desloc`, `P0` |
| 3 | EPIC | P1 | [EPIC] Context-parallel group management with heterogeneous sequence lengths | `core/parallel_state`, `autosp`, `P1` |
| 4 | FEA | P0 | [FEA] VRAM-proportional rank assignment across GPU tiers | `core/parallel_state`, `enhancement`, `P0` |
| 5 | FEA | P0 | [FEA] Bandwidth-aware process group rebalancing at runtime | `core/parallel_state`, `enhancement`, `P0` |
| 6 | FEA | P1 | [FEA] SM capability detection (SM8.6/9.0/12.0) for tier classification | `core/parallel_state`, `enhancement`, `P1` |
| 7 | FEA | P1 | [FEA] PCIe vs NVLink topology graph construction per tier | `core/parallel_state`, `enhancement`, `P1` |
| 8 | FEA | P2 | [FEA] Fallback to single-tier mode when heterogeneous init fails | `core/parallel_state`, `enhancement`, `P2` |
| 9 | FEA | P1 | [FEA] Process group health monitoring and heartbeat protocol | `core/parallel_state`, `enhancement`, `P1` |
| 10 | FEA | P2 | [FEA] Hot-swap tier re-registration without full restart | `core/parallel_state`, `enhancement`, `P2` |
| 11 | FEA | P2 | [FEA] Expose tier topology as JSON for external monitoring | `core/parallel_state`, `enhancement`, `P2` |
| 12 | BUG | P0 | [BUG] get_context_parallel_group() crashes when Megatron parallel init skipped | `core/parallel_state`, `bug`, `P0` |
| 13 | BUG | P1 | [BUG] HeteroRegistry activates 0/10 modules on rank 0 (H100) | `core/parallel_state`, `bug`, `P1` |
| 14 | BUG | P1 | [BUG] Race condition in tier_map initialization with >4 GPUs | `core/parallel_state`, `bug`, `P1` |
| 15 | DOC | P2 | [DOC] Document tier classification algorithm and SM capability mapping | `core/parallel_state`, `documentation`, `P2` |
| 16 | DOC | P2 | [DOC] Process group lifecycle diagram for heterogeneous clusters | `core/parallel_state`, `documentation`, `P2` |

### [THEME] Pubsub-Loop Coordination Protocol

| # | Type | Priority | Title | Labels |
|---|------|----------|-------|--------|
| 1 | EPIC | P0 | [EPIC] Blockchain-inspired pubsub-loop for tier capability broadcasting | `core/parallel_state`, `desloc`, `P0` |
| 2 | FEA | P0 | [FEA] Tier capability publish message format (VRAM/BW/SM/interconnect) | `core/parallel_state`, `enhancement`, `P0` |
| 3 | FEA | P0 | [FEA] Subscribe-to-gradient-event protocol per rank | `core/parallel_state`, `enhancement`, `P0` |
| 4 | FEA | P1 | [FEA] Training-step-as-block consensus mechanism | `core/parallel_state`, `enhancement`, `P1` |
| 5 | FEA | P1 | [FEA] Async tier capability update during training without barrier | `core/parallel_state`, `enhancement`, `P1` |
| 6 | FEA | P2 | [FEA] Pubsub message compression for bandwidth-constrained PCIe links | `core/parallel_state`, `enhancement`, `P2` |
| 7 | DOC | P1 | [DOC] Pubsub-loop protocol specification (message format, consensus rules) | `core/parallel_state`, `documentation`, `P1` |

---

## Module: `desloc_engine`

> DES-LOC heterogeneous training engine — runtime layer scheduling, partition, and checkpoint management

### [THEME] Runtime Layer Scheduling

| # | Type | Priority | Title | Labels |
|---|------|----------|-------|--------|
| 1 | EPIC | P0 | [EPIC] DES-LOC engine core — tier-aware forward/backward scheduling | `desloc_engine`, `desloc`, `P0` |
| 2 | EPIC | P0 | [EPIC] Heterogeneous 1F1B pipeline schedule with tier-aware stage assignment | `desloc_engine`, `core/pipeline_parallel`, `P0` |
| 3 | EPIC | P0 | [EPIC] Dynamic layer-to-tier assignment based on runtime profiling | `desloc_engine`, `desloc`, `P0` |
| 4 | FEA | P0 | [FEA] Tier-aware activation checkpointing with VRAM budget constraints | `desloc_engine`, `enhancement`, `P0` |
| 5 | FEA | P0 | [FEA] CPU offload scheduling for A6000 tier (48GB VRAM budget) | `desloc_engine`, `enhancement`, `P0` |
| 6 | FEA | P1 | [FEA] H100 NVL fast-path bypass for NVLink-connected layers | `desloc_engine`, `enhancement`, `P1` |
| 7 | FEA | P1 | [FEA] Blackwell SM12.0 tensor core utilization path | `desloc_engine`, `enhancement`, `P1` |
| 8 | FEA | P1 | [FEA] Runtime MFU (Model FLOPs Utilization) tracker per tier | `desloc_engine`, `enhancement`, `benchmark`, `P1` |
| 9 | FEA | P1 | [FEA] Straggler detection and dynamic load rebalancing | `desloc_engine`, `enhancement`, `P1` |
| 10 | FEA | P1 | [FEA] Micro-batch size auto-scaling per tier based on VRAM pressure | `desloc_engine`, `enhancement`, `P1` |
| 11 | FEA | P0 | [FEA] Gradient accumulation step alignment across heterogeneous tiers | `desloc_engine`, `enhancement`, `P0` |
| 12 | BUG | P0 | [BUG] Training hangs at step 0 when rank 1/2 never reach forward | `desloc_engine`, `bug`, `P0` |
| 13 | BUG | P0 | [BUG] NameError: config not defined in desloc_engine.py:1747 | `desloc_engine`, `bug`, `P0` |
| 14 | BUG | P1 | [BUG] Activation checkpointing tier assignment wrong — all layers show A6000 | `desloc_engine`, `bug`, `P1` |
| 15 | BUG | P0 | [BUG] finalize_model_grads crashes when parallel_state not initialized | `desloc_engine`, `bug`, `P0` |

### [THEME] Partition and Shard Management

| # | Type | Priority | Title | Labels |
|---|------|----------|-------|--------|
| 1 | EPIC | P0 | [EPIC] VRAM-proportional model partitioning across GPU tiers | `desloc_engine`, `desloc`, `P0` |
| 2 | EPIC | P0 | [EPIC] Heterogeneous ZeRO-3 with tier-proportional sharding | `desloc_engine`, `core/optimizer`, `P0` |
| 3 | FEA | P0 | [FEA] Shard planner with bandwidth-aware placement | `desloc_engine`, `enhancement`, `P0` |
| 4 | FEA | P1 | [FEA] Dynamic shard rebalancing on tier capability change | `desloc_engine`, `enhancement`, `P1` |
| 5 | FEA | P2 | [FEA] Parameter-group-aware sharding (freeze/unfreeze support) | `desloc_engine`, `enhancement`, `P2` |
| 6 | FEA | P1 | [FEA] Async parameter gather with overlap on backward pass | `desloc_engine`, `enhancement`, `P1` |
| 7 | BUG | P0 | [BUG] double weight_decay — decoupled_weight_decay + optimizer wd stacks | `desloc_engine`, `core/optimizer`, `bug`, `P0` |

### [THEME] Checkpointing and Fault Tolerance

| # | Type | Priority | Title | Labels |
|---|------|----------|-------|--------|
| 1 | EPIC | P1 | [EPIC] Heterogeneous async checkpoint save/load across tiers | `desloc_engine`, `desloc`, `P1` |
| 2 | FEA | P1 | [FEA] Tier-aware checkpoint sharding to match VRAM layout | `desloc_engine`, `enhancement`, `P1` |
| 3 | FEA | P1 | [FEA] Checkpoint format for dynamic tier topology (portable across configs) | `desloc_engine`, `enhancement`, `P1` |
| 4 | FEA | P2 | [FEA] Auto-resume from last good checkpoint on tier failure | `desloc_engine`, `enhancement`, `P2` |
| 5 | FEA | P2 | [FEA] Streaming checkpoint to CPU/NVMe during training | `desloc_engine`, `enhancement`, `P2` |
| 6 | DOC | P2 | [DOC] Checkpoint format specification for heterogeneous clusters | `desloc_engine`, `documentation`, `P2` |

### [THEME] World Model / Cyber-Physical Runtime

| # | Type | Priority | Title | Labels |
|---|------|----------|-------|--------|
| 1 | EPIC | P1 | [EPIC] Continuous topology adaptation — runtime GPU add/remove handling | `desloc_engine`, `desloc`, `P1` |
| 2 | FEA | P0 | [FEA] Runtime bandwidth probing between all tier pairs | `desloc_engine`, `enhancement`, `P0` |
| 3 | FEA | P2 | [FEA] Thermal throttling detection and tier capability downgrade | `desloc_engine`, `enhancement`, `P2` |
| 4 | FEA | P2 | [FEA] ECC error monitoring and preemptive tier exclusion | `desloc_engine`, `enhancement`, `P2` |
| 5 | FEA | P2 | [FEA] Live topology visualization via JSON event stream | `desloc_engine`, `enhancement`, `P2` |
| 6 | DOC | P1 | [DOC] World model architecture: cyber-physical loop specification | `desloc_engine`, `documentation`, `P1` |

---

## Module: `autosp`

> Automatic Sequence Parallelism — dynamic SP degree detection, fusion, and heterogeneous context-parallel coordination

### [THEME] SP Degree Auto-Detection

| # | Type | Priority | Title | Labels |
|---|------|----------|-------|--------|
| 1 | EPIC | P0 | [EPIC] AutoSP detector — runtime optimal SP degree selection per tier | `autosp`, `P0` |
| 2 | EPIC | P1 | [EPIC] Heterogeneous context-parallel coordination (CP across mixed tiers) | `autosp`, `core/parallel_state`, `P1` |
| 3 | FEA | P0 | [FEA] Per-layer SP degree based on attention head count and tier VRAM | `autosp`, `enhancement`, `P0` |
| 4 | FEA | P0 | [FEA] Sequence length adaptive partitioning for variable-length inputs | `autosp`, `enhancement`, `P0` |
| 5 | FEA | P1 | [FEA] SP/TP interaction matrix — valid SP degrees per TP configuration | `autosp`, `enhancement`, `P1` |
| 6 | FEA | P1 | [FEA] Profiling-guided SP degree calibration (one-shot warmup) | `autosp`, `enhancement`, `benchmark`, `P1` |
| 7 | FEA | P2 | [FEA] Runtime SP degree switching without gradient invalidation | `autosp`, `enhancement`, `P2` |
| 8 | BUG | P0 | [BUG] SP + RoPE shape mismatch after all-to-all scatter | `autosp`, `bug`, `P0` |
| 9 | BUG | P2 | [BUG] AutoSP transformers version warning — pin or test with 5.12.1 | `autosp`, `bug`, `P2` |

### [THEME] SP Fusion and Optimization

| # | Type | Priority | Title | Labels |
|---|------|----------|-------|--------|
| 1 | EPIC | P1 | [EPIC] AutoSP operator fusion — fuse attention scatter with SP communication | `autosp`, `P1` |
| 2 | FEA | P0 | [FEA] All-to-all scatter/gather overlap with computation | `autosp`, `enhancement`, `P0` |
| 3 | FEA | P1 | [FEA] SP-aware flash attention integration | `autosp`, `core/attention`, `enhancement`, `P1` |
| 4 | FEA | P2 | [FEA] ViT-specific AutoSP path (patch-based sequence splitting) | `autosp`, `enhancement`, `P2` |
| 5 | FEA | P2 | [FEA] Multimodal SP — mixed text+image sequence partitioning | `autosp`, `enhancement`, `P2` |
| 6 | DOC | P1 | [DOC] AutoSP algorithm specification: detection → calibration → runtime loop | `autosp`, `documentation`, `P1` |

### [THEME] Pubsub-Loop Integration

| # | Type | Priority | Title | Labels |
|---|------|----------|-------|--------|
| 1 | EPIC | P1 | [EPIC] AutoSP ↔ pubsub-loop: SP degree as published capability | `autosp`, `core/parallel_state`, `P1` |
| 2 | FEA | P1 | [FEA] SP degree change event publishing to coordination layer | `autosp`, `enhancement`, `P1` |
| 3 | FEA | P1 | [FEA] Subscribe to tier bandwidth changes to trigger SP recalibration | `autosp`, `enhancement`, `P1` |
| 4 | FEA | P0 | [FEA] Consensus check: all ranks agree on SP degree before step | `autosp`, `enhancement`, `P0` |

---

## Module: `csrc`

> C++/CUDA kernels for heterogeneous multi-GPU training — fused ops, communication kernels, and tier-specific code paths

### [THEME] Heterogeneous Gradient Communication Kernels

| # | Type | Priority | Title | Labels |
|---|------|----------|-------|--------|
| 1 | EPIC | P0 | [EPIC] Fused heterogeneous gradient reduce-scatter kernel suite | `csrc`, `cuda-kernel`, `hetero_reduce`, `P0` |
| 2 | EPIC | P0 | [EPIC] PCIe-aware NCCL allreduce with adaptive bucketing | `csrc`, `cuda-kernel`, `P0` |
| 3 | FEA | P0 | [FEA] Warp-level cooperative reduce-scatter with adaptive bucket sizing | `csrc`, `cuda-kernel`, `enhancement`, `P0` |
| 4 | FEA | P0 | [FEA] Runtime bandwidth probing kernel for PCIe/NVLink classification | `csrc`, `cuda-kernel`, `enhancement`, `P0` |
| 5 | FEA | P1 | [FEA] Ring allreduce with per-link bandwidth weighting | `csrc`, `cuda-kernel`, `enhancement`, `P1` |
| 6 | FEA | P2 | [FEA] Async gradient compression (FP16→FP8) for PCIe-bottlenecked links | `csrc`, `cuda-kernel`, `enhancement`, `P2` |
| 7 | FEA | P1 | [FEA] NCCL group call batching for heterogeneous tier pairs | `csrc`, `cuda-kernel`, `enhancement`, `P1` |
| 8 | BUG | P1 | [BUG] hetero_reduce 2357 lines of CUDA with no benchmark proving they beat NCCL | `csrc`, `bug`, `benchmark`, `P1` |

### [THEME] Fused Compute Kernels

| # | Type | Priority | Title | Labels |
|---|------|----------|-------|--------|
| 1 | EPIC | P0 | [EPIC] Fused SwiGLU + LayerNorm kernel for SM8.6/9.0/12.0 | `csrc`, `cuda-kernel`, `P0` |
| 2 | EPIC | P1 | [EPIC] Fused RoPE kernel for heterogeneous attention head counts | `csrc`, `cuda-kernel`, `csrc/rope`, `P1` |
| 3 | FEA | P0 | [FEA] SM-specific memory access patterns (shared mem tiling per arch) | `csrc`, `cuda-kernel`, `enhancement`, `P0` |
| 4 | FEA | P1 | [FEA] Blackwell TMA-based BlockLoadToShared for PCIe-bottlenecked scatter | `csrc`, `cuda-kernel`, `enhancement`, `P1` |
| 5 | FEA | P1 | [FEA] SM12.0 (Blackwell) code paths for all csrc/hetero_reduce kernels | `csrc`, `cuda-kernel`, `enhancement`, `P1` |
| 6 | FEA | P1 | [FEA] Fused cross-entropy for heterogeneous loss computation | `csrc`, `cuda-kernel`, `enhancement`, `P1` |
| 7 | FEA | P1 | [FEA] Tier-aware activation checkpointing offload kernel (GPU→CPU async) | `csrc`, `cuda-kernel`, `enhancement`, `P1` |
| 8 | FEA | P0 | [FEA] Fused heterogeneous Adam optimizer (mixed precision per tier) | `csrc`, `cuda-kernel`, `enhancement`, `P0` |
| 9 | FEA | P1 | [FEA] Fused LayerNorm + residual connection with tier-specific vectorization | `csrc`, `cuda-kernel`, `enhancement`, `P1` |

### [THEME] Kernel Benchmarking and Validation

| # | Type | Priority | Title | Labels |
|---|------|----------|-------|--------|
| 1 | EPIC | P0 | [EPIC] Comprehensive kernel benchmark suite: hetero_reduce vs NCCL baseline | `csrc`, `benchmark`, `P0` |
| 2 | FEA | P1 | [FEA] Per-kernel roofline analysis (compute vs memory bound per SM arch) | `csrc`, `benchmark`, `enhancement`, `P1` |
| 3 | FEA | P2 | [FEA] Automated SASS analysis for register pressure and occupancy | `csrc`, `benchmark`, `enhancement`, `P2` |
| 4 | FEA | P1 | [FEA] CI integration: kernel correctness tests on SM8.6 + SM9.0 + SM12.0 | `csrc`, `benchmark`, `enhancement`, `P1` |
| 5 | FEA | P1 | [FEA] Numerical stability validation for fused kernels (FP16/BF16/FP8) | `csrc`, `benchmark`, `enhancement`, `P1` |
| 6 | DOC | P2 | [DOC] Kernel API reference: launch configs, shared memory requirements, SM compatibility | `csrc`, `documentation`, `P2` |

---

## Module: `core/transformer`

> Megatron TransformerLayer with heterogeneous attention, MLP, and tier-specific precision paths

### [THEME] Heterogeneous Transformer Architecture

| # | Type | Priority | Title | Labels |
|---|------|----------|-------|--------|
| 1 | EPIC | P0 | [EPIC] Megatron TransformerLayer migration with heterogeneous attention + MLP | `core/transformer`, `megatron-migration`, `P0` |
| 2 | EPIC | P1 | [EPIC] Heterogeneous attention head assignment across GPU tiers | `core/transformer`, `core/attention`, `P1` |
| 3 | FEA | P0 | [FEA] Per-tier precision path: BF16 on H100/Blackwell, FP16 on A6000 | `core/transformer`, `enhancement`, `P0` |
| 4 | FEA | P1 | [FEA] GQA (Grouped Query Attention) support with heterogeneous KV heads | `core/transformer`, `core/attention`, `enhancement`, `P1` |
| 5 | FEA | P2 | [FEA] MLA (Multi-Latent Attention) heterogeneous implementation | `core/transformer`, `core/attention`, `enhancement`, `P2` |
| 6 | FEA | P1 | [FEA] Flash Attention v3 integration with tier-specific configs | `core/transformer`, `core/attention`, `enhancement`, `P1` |
| 7 | FEA | P1 | [FEA] Heterogeneous MLP: SwiGLU width scaling per tier VRAM | `core/transformer`, `enhancement`, `P1` |
| 8 | FEA | P1 | [FEA] FP8 training path for Blackwell tiers | `core/transformer`, `enhancement`, `P1` |
| 9 | BUG | P0 | [BUG] RoPE fallback crash — megatron not installed path broken | `core/transformer`, `bug`, `P0` |

### [THEME] Tensor Parallelism

| # | Type | Priority | Title | Labels |
|---|------|----------|-------|--------|
| 1 | EPIC | P1 | [EPIC] ColumnParallel + RowParallel + VocabEmbed for heterogeneous TP | `core/tensor_parallel`, `megatron-migration`, `P1` |
| 2 | FEA | P2 | [FEA] Asymmetric TP: different TP degrees for attention vs MLP | `core/tensor_parallel`, `enhancement`, `P2` |
| 3 | FEA | P1 | [FEA] TP-aware gradient bucketing for heterogeneous allreduce | `core/tensor_parallel`, `enhancement`, `P1` |
| 4 | DOC | P2 | [DOC] TP + SP interaction matrix — valid configurations per tier count | `core/tensor_parallel`, `documentation`, `P2` |

---

## Module: `core/distributed`

> DDP + gradient sync + tier-aware bucketing for heterogeneous training

### [THEME] Heterogeneous DDP

| # | Type | Priority | Title | Labels |
|---|------|----------|-------|--------|
| 1 | EPIC | P0 | [EPIC] Tier-aware DDP with bandwidth-proportional gradient bucketing | `core/distributed`, `megatron-migration`, `P0` |
| 2 | FEA | P0 | [FEA] Per-tier bucket size tuning based on measured PCIe/NVLink bandwidth | `core/distributed`, `enhancement`, `P0` |
| 3 | FEA | P1 | [FEA] Gradient compression (TopK/Random-K) for PCIe-bottlenecked pairs | `core/distributed`, `enhancement`, `P1` |
| 4 | FEA | P1 | [FEA] Overlap computation with allreduce via pipelined bucketing | `core/distributed`, `enhancement`, `P1` |
| 5 | FEA | P1 | [FEA] Heterogeneous DDP gradient norm clipping across tiers | `core/distributed`, `enhancement`, `P1` |
| 6 | FEA | P0 | [FEA] DeepSpeedCPUAdam compilation and tier-specific optimizer dispatch | `core/distributed`, `enhancement`, `P0` |
| 7 | BUG | P0 | [BUG] DeepSpeedCPUAdam not compiled — A6000 falls back to CPU AdamW | `core/distributed`, `bug`, `P0` |

---

## Module: `core/optimizer`

> Heterogeneous optimizer with ZeRO-3, tier-proportional sharding, and mixed precision per tier

### [THEME] Heterogeneous Optimizer Suite

| # | Type | Priority | Title | Labels |
|---|------|----------|-------|--------|
| 1 | EPIC | P0 | [EPIC] DistributedOptimizer (ZeRO-3) for heterogeneous GPU clusters | `core/optimizer`, `megatron-migration`, `P0` |
| 2 | EPIC | P0 | [EPIC] Heterogeneous ZeRO-3 shard rebalancing with VRAM-proportional distribution | `core/optimizer`, `P0` |
| 3 | FEA | P1 | [FEA] Per-tier learning rate scaling based on effective batch size | `core/optimizer`, `enhancement`, `P1` |
| 4 | FEA | P1 | [FEA] Mixed precision master weights: FP32 on CPU, BF16 on GPU per tier | `core/optimizer`, `enhancement`, `P1` |
| 5 | FEA | P1 | [FEA] Gradient accumulation with tier-aware loss scaling | `core/optimizer`, `enhancement`, `P1` |
| 6 | FEA | P1 | [FEA] Optimizer state CPU offload for A6000 tier | `core/optimizer`, `enhancement`, `P1` |
| 7 | FEA | P1 | [FEA] FP32 gradient accumulation buffer for numerical stability across tiers | `core/optimizer`, `enhancement`, `P1` |
| 8 | FEA | P2 | [FEA] Gradient norm skip for heterogeneous training divergence detection | `core/optimizer`, `enhancement`, `P2` |

---

## Module: `core/pipeline_parallel`

> Heterogeneous 1F1B schedule with tier-aware stage assignment and P2P communication

### [THEME] Heterogeneous Pipeline Schedule

| # | Type | Priority | Title | Labels |
|---|------|----------|-------|--------|
| 1 | EPIC | P1 | [EPIC] 1F1B + P2P communication for heterogeneous pipeline | `core/pipeline_parallel`, `megatron-migration`, `P1` |
| 2 | FEA | P0 | [FEA] VRAM-proportional stage assignment (more layers on H100/Blackwell) | `core/pipeline_parallel`, `enhancement`, `P0` |
| 3 | FEA | P1 | [FEA] Pipeline bubble minimization for asymmetric stage compute times | `core/pipeline_parallel`, `enhancement`, `P1` |
| 4 | FEA | P1 | [FEA] Inter-tier P2P with automatic format conversion (BF16↔FP16) | `core/pipeline_parallel`, `enhancement`, `P1` |
| 5 | FEA | P2 | [FEA] Pipeline interleaving for heterogeneous micro-batch sizes | `core/pipeline_parallel`, `enhancement`, `P2` |
| 6 | FEA | P1 | [FEA] Bridge communicator for cross-tier pipeline stages | `core/pipeline_parallel`, `enhancement`, `P1` |

---

## Module: `core/datasets`

> CommitPack data pipeline + Megatron indexed dataset for code pretraining

### [THEME] CommitPack Data Pipeline

| # | Type | Priority | Title | Labels |
|---|------|----------|-------|--------|
| 1 | EPIC | P1 | [EPIC] CommitPack streaming dataset with Megatron-compatible indexing | `core/transformer`, `megatron-migration`, `P1` |
| 2 | FEA | P0 | [FEA] JSONL → Megatron indexed binary conversion pipeline | `core/transformer`, `enhancement`, `P0` |
| 3 | FEA | P1 | [FEA] CommitPackFT (fine-tuning split) dataset integration | `core/transformer`, `enhancement`, `P1` |
| 4 | FEA | P1 | [FEA] Language-filtered streaming (Python-only subset for code pretraining) | `core/transformer`, `enhancement`, `P1` |
| 5 | FEA | P0 | [FEA] Heterogeneous data loading: per-tier batch size with global shuffle | `core/transformer`, `enhancement`, `P0` |
| 6 | FEA | P1 | [FEA] Elastic batch size scheduling per tier based on VRAM availability | `core/transformer`, `enhancement`, `P1` |
| 7 | FEA | P2 | [FEA] Data pipeline throughput benchmark (tokens/sec per tier) | `core/transformer`, `benchmark`, `enhancement`, `P2` |
| 8 | DOC | P2 | [DOC] Data preparation guide: CommitPack → indexed binary → training | `core/transformer`, `documentation`, `P2` |

---

## Module: `hetero_bridge`

> Integration layer connecting DES-LOC engine with Megatron core modules and DeepSpeed runtime

### [THEME] Bridge Architecture

| # | Type | Priority | Title | Labels |
|---|------|----------|-------|--------|
| 1 | EPIC | P0 | [EPIC] hetero_bridge: unified adapter layer for DES-LOC ↔ Megatron ↔ DeepSpeed | `desloc_engine`, `integration`, `P0` |
| 2 | FEA | P1 | [FEA] AutoSP hook registration in hetero_bridge | `desloc_engine`, `autosp`, `enhancement`, `P1` |
| 3 | FEA | P0 | [FEA] DES-LOC sync policy adapter (gradient sync timing per tier) | `desloc_engine`, `enhancement`, `P0` |
| 4 | FEA | P0 | [FEA] Distributed optimizer adapter for heterogeneous ZeRO | `desloc_engine`, `core/optimizer`, `enhancement`, `P0` |
| 5 | FEA | P1 | [FEA] Pipeline schedule adapter for 1F1B with heterogeneous stages | `desloc_engine`, `core/pipeline_parallel`, `enhancement`, `P1` |
| 6 | FEA | P1 | [FEA] Engine integration test harness (mock 5-GPU config on CPU) | `desloc_engine`, `enhancement`, `P1` |

---

## Module: `benchmarks`

> Performance benchmarks, MFU tracking, and ablation experiments for NeurIPS 2026

### [THEME] Training Performance Benchmarks

| # | Type | Priority | Title | Labels |
|---|------|----------|-------|--------|
| 1 | EPIC | P0 | [EPIC] End-to-end training benchmark suite: 7B/13B models on 5-GPU heterogeneous cluster | `benchmark`, `P0` |
| 2 | FEA | P0 | [FEA] MFU (Model FLOPs Utilization) measurement per tier and aggregate | `benchmark`, `enhancement`, `P0` |
| 3 | FEA | P0 | [FEA] Communication overhead breakdown: PCIe vs NVLink per operation | `benchmark`, `enhancement`, `P0` |
| 4 | FEA | P0 | [FEA] hetero_reduce vs NCCL baseline benchmark (all message sizes) | `benchmark`, `csrc`, `enhancement`, `P0` |
| 5 | FEA | P1 | [FEA] Scaling efficiency: 2-GPU → 3-GPU → 5-GPU heterogeneous | `benchmark`, `enhancement`, `P1` |
| 6 | FEA | P1 | [FEA] Memory usage tracking per tier across training steps | `benchmark`, `enhancement`, `P1` |
| 7 | FEA | P0 | [FEA] Training convergence comparison: heterogeneous vs homogeneous baseline | `benchmark`, `enhancement`, `P0` |

### [THEME] NeurIPS 2026 Ablation Experiments

| # | Type | Priority | Title | Labels |
|---|------|----------|-------|--------|
| 1 | EPIC | P0 | [EPIC] NeurIPS 2026 paper ablation experiment suite | `benchmark`, `P0` |
| 2 | FEA | P0 | [FEA] Ablation: SP degree impact on throughput (SP=1,2,4,8) | `benchmark`, `autosp`, `enhancement`, `P0` |
| 3 | FEA | P0 | [FEA] Ablation: tier-proportional vs equal sharding impact | `benchmark`, `enhancement`, `P0` |
| 4 | FEA | P1 | [FEA] Ablation: fused kernel vs PyTorch baseline per operation | `benchmark`, `csrc`, `enhancement`, `P1` |
| 5 | FEA | P1 | [FEA] Ablation: dynamic vs static layer assignment | `benchmark`, `desloc_engine`, `enhancement`, `P1` |
| 6 | FEA | P2 | [FEA] Ablation: checkpoint frequency impact on training time | `benchmark`, `enhancement`, `P2` |
| 7 | FEA | P0 | [FEA] Comparison table generation: DES-LOC vs Megatron vs DeepSpeed baselines | `benchmark`, `enhancement`, `P0` |
| 8 | DOC | P0 | [DOC] NeurIPS experiment protocol: hardware config, seeds, hyperparameters | `benchmark`, `documentation`, `P0` |

---

## Module: `infra`

> CI/CD, testing infrastructure, build system, and developer tooling

### [THEME] CI/CD and Testing

| # | Type | Priority | Title | Labels |
|---|------|----------|-------|--------|
| 1 | EPIC | P1 | [EPIC] CI pipeline for heterogeneous GPU testing (mock + real hardware) | `build`, `P1` |
| 2 | FEA | P1 | [FEA] CPU-only mock test for tier-aware process groups | `build`, `enhancement`, `P1` |
| 3 | FEA | P1 | [FEA] CUDA kernel unit tests with compute-sanitizer integration | `build`, `csrc`, `enhancement`, `P1` |
| 4 | FEA | P2 | [FEA] Automated regression benchmark on PR merge | `build`, `benchmark`, `enhancement`, `P2` |
| 5 | FEA | P2 | [FEA] Docker image with 5-GPU mock topology for local testing | `build`, `enhancement`, `P2` |
| 6 | FEA | P1 | [FEA] Blackwell (SM12.0) CI coverage when cu126+ available | `build`, `enhancement`, `P1` |
| 7 | BUG | P1 | [BUG] Blackwell GPUs excluded — need PyTorch cu126+ for SM12.0 | `build`, `bug`, `P1` |

### [THEME] Developer Experience

| # | Type | Priority | Title | Labels |
|---|------|----------|-------|--------|
| 1 | FEA | P2 | [FEA] Suppress 32-layer activation checkpoint log spam | `build`, `enhancement`, `P2` |
| 2 | FEA | P2 | [FEA] Suppress [M] tags — hundreds of silent milestone markers in stdout | `build`, `enhancement`, `P2` |
| 3 | FEA | P2 | [FEA] Structured logging with per-tier log levels | `build`, `enhancement`, `P2` |
| 4 | FEA | P1 | [FEA] CLAUDE.md and AGENTS.md maintenance for multi-Claude dispatch | `build`, `enhancement`, `P1` |
| 5 | DOC | P1 | [DOC] ARCHITECTURE.md update: full module dependency graph | `build`, `documentation`, `P1` |
| 6 | DOC | P2 | [DOC] Contributing guide for heterogeneous kernel development | `build`, `documentation`, `P2` |

---

## Module: `paper`

> NeurIPS 2026 paper: FAUST — Flexible Asynchronous Unified Scheduling for Training

### [THEME] Paper Writing and Submission

| # | Type | Priority | Title | Labels |
|---|------|----------|-------|--------|
| 1 | EPIC | P0 | [EPIC] FAUST NeurIPS 2026 paper: writing, experiments, camera-ready | `documentation`, `P0` |
| 2 | FEA | P1 | [FEA] Related work survey: heterogeneous training systems (2023-2026) | `documentation`, `enhancement`, `P1` |
| 3 | FEA | P1 | [FEA] Figure generation: system architecture diagram | `documentation`, `enhancement`, `P1` |
| 4 | FEA | P0 | [FEA] Figure generation: training throughput comparison charts | `documentation`, `enhancement`, `P0` |
| 5 | FEA | P1 | [FEA] Figure generation: pubsub-loop protocol visualization | `documentation`, `enhancement`, `P1` |
| 6 | FEA | P1 | [FEA] Appendix: full experiment hyperparameter tables | `documentation`, `enhancement`, `P1` |
| 7 | FEA | P2 | [FEA] Supplementary: kernel benchmark raw data and analysis | `documentation`, `enhancement`, `P2` |
