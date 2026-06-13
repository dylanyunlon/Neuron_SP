# CLAUDE_M302_M316.md — Claude-20 Handoff
## Session: Claude-20 (M302-M316) | Base: commit 70830c7a (M301)
## Theme: Heterogeneous GPU Topology + Scheduling
## All 14 Python ast.parse OK. Shell bash -n OK. Zero numpy.random.

| M# | File | Key Additions |
|---|---|---|
| M302 | cuda_accelerator.py | GPU DB (12 SKUs), topo detect, hetero batch alloc, config recommend |
| M303 | engine.py | coord_clip, half_life, spike detect, recovery, NKI-FA export, scaling law |
| M304 | config.py | hetero config, env overrides, optimal periods, presets |
| M305 | utils.py | hetero step times, adaptive Kx, savings estimator |
| M306 | comm.py | AllReduce gate, tier classify, comm budget, CommTracker |
| M307 | backend.py | HealthMon, NaNDet, Watchdog |
| M308 | torch.py | TorchAdapt, BWTrack |
| M309 | pipe/module.py | topo partition, bubble, DES-LOC interaction |
| M310 | pipe/engine.py | 1F1B schedule, hetero mb, pipe timing |
| M311 | dataloader.py | HeteroDataLoader, grad accum, throughput report |
| M312 | stage_1_and_2.py | ZeRO Kx gate, hetero partition, comm volume |
| M313 | comms_logging.py | tier classify, NKI-FA logger, figure data |
| M314 | timer.py | StepTimer, MFU, roofline, progress tracker |
| M315 | REAL_GPU_BENCHMARK.py | detect_gpus, bench_matmul, ablation configs |
| M316 | run_desloc_benchmark.sh | full pipeline: detect→bench→matrix→run→aggregate |

## Rules for Claude-21 (M317+)
1. NO standalone classes at EOF
2. cat first, ast.parse after
3. Zero numpy.random, Kx=1 = original DeepSpeed
4. Data from NKI-FA logs only
