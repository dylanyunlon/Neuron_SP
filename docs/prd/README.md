# Neuron_SP Product Requirements Document

> CCCL-grade taxonomy · 13 modules · 24 themes · 1615 items on Project Board

## Files

- `Neuron_SP_PRD.md` — Full product requirements document with 182 core items
- `task_instruction.txt` — Task specification for Claude Code sub-agents
- `project4_items_snapshot.json` — Snapshot of all 1615 Project #4 items

## Project Board

Live board: https://github.com/users/dylanyunlon/projects/4

## Architecture

```
Neuron_SP Product Requirements
├── core/parallel_state    (23 items) — Tier-aware process groups + pubsub-loop
├── desloc_engine          (34 items) — DES-LOC runtime scheduling/partitioning
├── autosp                 (19 items) — Automatic sequence parallelism
├── csrc                   (23 items) — C++/CUDA heterogeneous kernels
├── core/transformer       (13 items) — Megatron TransformerLayer
├── core/distributed       (7 items)  — Tier-aware DDP
├── core/optimizer         (8 items)  — Heterogeneous ZeRO-3
├── core/pipeline_parallel (6 items)  — 1F1B pipeline
├── core/datasets          (8 items)  — CommitPack pipeline
├── hetero_bridge          (6 items)  — Integration adapter
├── benchmarks             (15 items) — Performance + NeurIPS ablation
├── infra                  (13 items) — CI/CD and build
└── paper                  (7 items)  — FAUST NeurIPS 2026
```
