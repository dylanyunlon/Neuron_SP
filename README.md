# Neuron_SP: DES-LOC Heterogeneous GPU Training Framework

DeepSpeed fork implementing **Decomposed Local SGD (DES-LOC)** with **Automatic Sequence Parallelism (AutoSP)** for heterogeneous GPU clusters. Neuron_SP enables efficient large-model pretraining across mixed-generation GPUs without NVLink, achieving near-linear scaling on commodity PCIe topologies.

## Overview

Standard distributed training frameworks assume homogeneous hardware — identical GPUs connected by high-bandwidth interconnects. Real-world clusters rarely look like that. Neuron_SP bridges the gap by decomposing synchronization into parameter-specific local SGD rounds, automatically partitioning sequence-parallel workloads across GPUs with different compute and memory profiles, and caching local optimizer states to minimize cross-device communication.

The framework powers experiments for the NeurIPS 2026 paper:
*DES-LOC: Decomposed Local SGD for Heterogeneous GPU Clusters with Automatic Sequence Parallelism* (see [`FAUST_nips2026/`](FAUST_nips2026/)).

## Features

**DES-LOC Engine** — Decomposed Local SGD that assigns per-parameter synchronization periods based on gradient variance, reducing all-reduce traffic by up to 4× while preserving convergence. Implemented in `deepspeed/runtime/desloc_engine.py` with tier-aware partition solving and heterogeneous gradient accumulation.

**AutoSP** — Automatic Sequence Parallelism built on DeepSpeed-Ulysses. Detects attention patterns and splits SP/DP groups to maximize throughput on mixed hardware. Modules in `deepspeed/sequence/` (auto_sp.py, autosp_detector.py, autosp_fusion.py) with Ulysses SP runtime in `deepspeed/runtime/sequence_parallel/`.

**LOC Cache** — Local optimizer-state caching across heterogeneous devices. Keeps Adam moments pinned on the fastest available memory tier, reducing H2D transfers during mixed-precision ZeRO-2/3 training. Integrated throughout the hetero runtime modules.

**5-Tier GPU Support** — Automatic hardware discovery and classification (H100, A6000, RTX PRO 6000 Blackwell, and more) via `TierDiscovery`. The partition solver adapts micro-batch sizes, gradient accumulation steps, and memory budgets per tier.

**Commit-Centric Pretraining** — Three-stage pipeline (`pipeline/train_three_stage.py`): base pretraining on code corpora → continued training on CommitPack diffs → instruction tuning on CommitPackFT. Designed for code-understanding models that learn incremental edits.

## Quick Start

```bash
# 1. Set up the training environment on your cluster
bash setup_ags1.sh

# 2. Launch 7B pretraining with DES-LOC on all available GPUs
bash launch_7b.sh
```

`setup_ags1.sh` creates a conda environment, installs dependencies (PyTorch, DeepSpeed, FlashAttention), and validates GPU topology. `launch_7b.sh` configures NCCL for PCIe-only communication, binds processes to NUMA nodes, and starts distributed training via the DeepSpeed launcher.

For 13B training:

```bash
bash run_13B_ags1.sh
```

## Architecture

```
Neuron_SP/
├── deepspeed/
│   ├── runtime/
│   │   ├── desloc_engine.py          # DES-LOC core: TierDiscovery + PartitionSolver + training loop
│   │   ├── sequence_parallel/        # Ulysses SP runtime (AllToAll-based)
│   │   └── hetero_*.py               # 62+ heterogeneous support modules
│   │       ├── hetero_mimo_topology   # Multi-input multi-output GPU topology
│   │       ├── hetero_elastic_batch   # Dynamic batch sizing per tier
│   │       ├── hetero_h2d_stream_sync # Host-to-device stream synchronization
│   │       └── ...
│   └── sequence/
│       ├── auto_sp.py                 # AutoSP: automatic SP/DP group selection
│       ├── autosp_detector.py         # Attention pattern detection
│       └── autosp_fusion.py           # Fused SP kernels
├── pipeline/
│   └── train_three_stage.py           # Three-stage pretraining orchestrator
├── experiments/                       # Scaling law experiments + convergence analysis
├── FAUST_nips2026/                    # NeurIPS 2026 paper (LaTeX)
├── configs/                           # DeepSpeed JSON configs
├── eval/                              # Evaluation harness
├── setup_ags1.sh                      # Environment setup
└── launch_7b.sh                       # Training launcher
```

The hetero runtime modules handle the full spectrum of mixed-GPU concerns: gradient bucketing with per-device process groups, CUDA graph compatibility across SM versions, pinned buffer lifecycle management, FSDP sharding strategies for asymmetric memory, and elastic batch scheduling that accounts for per-tier throughput.

## Hardware

Development and benchmarking target the following cluster (ags1):

| GPU | Count | VRAM | SM | Interconnect | BF16 TFLOPS |
|-----|-------|------|----|--------------|-------------|
| NVIDIA A6000 | 2 | 48 GB | 8.6 | PCIe 4.0 | 38.7 |
| NVIDIA H100 NVL | 1 | 96 GB | 9.0 | PCIe 5.0 | 835 |
| NVIDIA RTX PRO 6000 Blackwell | 2 | 96 GB | 12.0 | PCIe 5.0 | TBD |

No NVLink between devices. CPU: 2× AMD EPYC 9354 (128 cores), 1.5 TB DDR5. NCCL communicates over shared memory with P2P disabled.

## Paper

The DES-LOC method and experimental results are described in:

> **DES-LOC: Decomposed Local SGD for Heterogeneous GPU Clusters with Automatic Sequence Parallelism**
> NeurIPS 2026 submission — [`FAUST_nips2026/`](FAUST_nips2026/)

## License

Apache-2.0 — see [LICENSE](LICENSE).
