# deepspeed/core/ — Architecture Design Document

## Overview

This directory implements Neuron_SP's core training infrastructure, modeled after
NVIDIA Megatron-LM's `megatron/core/` but adapted for **heterogeneous GPU clusters**
running DES-LOC (Decomposed Local SGD) with AutoSP (Automatic Sequence Parallelism).

Target hardware: mixed-tier GPU clusters (e.g. 2×A6000 + 1×H100 NVL, PCIe-only,
no NVLink). All modules must support:
- Per-tier compute/memory budgets (different GPUs get different workloads)
- Kx/Ku/Kv decomposed synchronization (skip gradient sync on non-Kx steps)
- PCIe-aware communication (no NVLink assumptions)
- NUMA-aware buffer placement

## Module Dependency Graph (build order)

```
Layer 0 (no deps):
  parallel_state.py           — process group management
  model_parallel_config.py    — parallelism config dataclass

Layer 1 (depends on Layer 0):
  tensor_parallel/            — TP layers (Column/RowParallelLinear)
  fusions/                    — fused kernels (bias+activation)
  quantization/               — precision management

Layer 2 (depends on Layer 0+1):
  distributed/                — DDP, FSDP, grad finalization, param buffers
  optimizer/                  — distributed optimizer, grad clipping, CPU offload
  pipeline_parallel/          — 1F1B schedules, P2P comm, activation offload
  dist_checkpointing/         — sharded checkpoint save/load

Layer 3 (depends on all above):
  transformer/                — attention, MLP, transformer layer/block, MoE
  datasets/                   — GPT dataset, blended dataset, data scheduling

Layer 4 (depends on all above):
  models/                     — GPT model, hybrid models
  ssm/                        — Mamba/SSM layers

Layer 5 (integration):
  Used by: deepspeed/runtime/desloc_engine.py
           deepspeed/runtime/desloc_partition.py
           deepspeed/runtime/zero3_hetero_shard.py
           deepspeed/sequence/auto_sp.py
           run_pretrain.py
```

## File Naming Convention

- NO `hetero_` prefix. The heterogeneity support is built into every class.
- Follow Megatron naming: `distributed_data_parallel.py`, `transformer_layer.py`, etc.
- One class per file when the class is >200 lines. Small helper classes can share a file.

## API Contract Rules

1. Every public class has a `__init__` that accepts a config dataclass (not loose args)
2. Every module that participates in DES-LOC must accept `desloc_config: Optional[DesLocConfig]`
3. Gradient sync methods must have a `skip_sync: bool` parameter for Kx skip
4. All collective operations go through `torch.distributed`, not `deepspeed.comm`
5. No file may exist without being imported by at least one other file in the tree
6. Type hints on all public method signatures

## DES-LOC Integration Points

Each module must expose hooks for the DES-LOC engine:

- `distributed/`: `finish_grad_sync(force_all_reduce=False)` — Kx sync gating
- `optimizer/`: `step()` must support per-shard param writeback + broadcast
- `pipeline_parallel/`: schedules must handle unequal micro-batch counts per stage
- `transformer/`: layers must support selective activation checkpointing per tier
- `datasets/`: data loading must support per-rank different batch sizes

## AutoSP Integration Points

- `transformer/attention.py`: sequence dim must be shardable via A2A
- `parallel_state.py`: must support SP process groups alongside TP/PP/DP
- `tensor_parallel/`: must route A2A through SP groups when SP is active

## Phase 2 Task Assignments (skeleton → implementation)

Each task reads the FULL evolution history of the corresponding Megatron module
(from first commit to HEAD), then fills the skeleton implementation. NO new
`hetero_xxx.py` files. All code goes into existing files.

### Task A: recompute.py + fusions/
- Megatron source: `megatron/core/recompute.py`, `megatron/core/fusions/`
- Skeleton: `deepspeed/core/recompute.py`, `deepspeed/core/fusions/`
- Key: tier-aware activation checkpointing (A6000=full recompute, H100=selective)
- Must integrate with: `transformer/transformer_layer.py` forward pass

### Task B: optimizer_param_scheduler.py
- Megatron source: `megatron/core/optimizer_param_scheduler.py`
- Skeleton: `deepspeed/core/optimizer_param_scheduler.py`
- Key: cosine/WSD LR schedule with per-tier LR multiplier
- Must integrate with: `optimizer/distrib_optimizer.py`, `desloc_engine.py`

### Task C: distrib_optimizer.py completion
- Megatron source: `megatron/core/optimizer/distrib_optimizer.py` (4700+ lines)
- Target: `deepspeed/core/optimizer/distrib_optimizer.py` (current 3886 lines)
- Key: verify grad bucket allreduce, param shard sync, CPU offload
- Must verify: compatible with `zero3_hetero_shard.py` and `hetero_optimizer_router.py`

### Task D: pipeline_parallel/schedules.py completion
- Megatron source: `megatron/core/pipeline_parallel/schedules.py`
- Target: `deepspeed/core/pipeline_parallel/schedules.py` (current 2314 lines)
- Key: interleaved 1F1B, unequal micro-batch per stage (critical for hetero PP)
- Must integrate with: `desloc_engine.py` training loop

### Task E: distributed/ finalize_model_grads.py audit
- Megatron source: `megatron/core/distributed/finalize_model_grads.py`
- Target: `deepspeed/core/distributed/finalize_model_grads.py`
- Key: Kx sync gating, PCIe-aware allreduce fusion
- Must verify: no gradient norm mismatch between tiers

### Task F: dist_checkpointing/ completion
- Megatron source: `megatron/core/dist_checkpointing/` (full directory)
- Target: `deepspeed/core/dist_checkpointing/`
- Key: async save, per-tier state sharding, resume from heterogeneous layout
- Must integrate with: `checkpoint/hetero_async_checkpoint_save.py`
