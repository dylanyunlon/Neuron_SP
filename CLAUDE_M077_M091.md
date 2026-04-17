# CLAUDE_M077_M091.md — Handoff to Claude-6 (M092-M106)

## Completed M000-M091 Summary

### Claude-1 (M001-M025): DES-LOC template extraction + LaTeX mapping
### Claude-2 (M026-M050): Benchmark framework + convergence theory (Section 3)
### Claude-3 (M050-M061): DES-LOC engine integration (12 files, 3159 lines)
### Claude-4 (M062-M076): GPU benchmark + comm optimization + runtime (14 files, 5505 lines)
### Claude-5 (M077-M091): Experiment engine + 14 benchmarks + NeurIPS viz (15 files, 4776 lines)

## Files Modified by Claude-5 (M077-M091)

Each M number = modify ONE existing repo source file + ~300-450 lines appended.
No new standalone files created. No numpy.random anywhere.

| M# | File | Lines Added | Key Classes/Functions |
|---|---|---|---|
| M077 | deepspeed/runtime/engine.py | +431 | DeslocExperimentLogger, WatchdogMonitor, HalfLifeAnalyzer, DeslocExperimentScheduler, DeslocMFUCalculator |
| M078 | deepspeed/runtime/config.py | +269 | HardwareProfile, BenchmarkDefinition, DESLOC_BENCHMARK_DEFINITIONS (14 benchmarks), DeslocBenchmarkSuite |
| M079 | deepspeed/runtime/utils.py | +342 | ExperimentLogParser (CSV/JSON/stdout/NKI), StatisticalAggregator, BenchmarkComparator, ResultsExporter |
| M080 | deepspeed/comm/comm.py | +346 | CommBandwidthProfiler, TopologyDetector, DeslocCommTracker |
| M081 | deepspeed/comm/torch.py | +293 | NeuronCommOpsMapper, NeuronCollectiveWrapper, EFABandwidthEstimator |
| M082 | deepspeed/comm/backend.py | +263 | PrecisionValidator, ThermalMonitor, HardwareDiagnostics |
| M083 | deepspeed/ops/adam/fused_adam.py | +268 | DeslocSyncController, GradientClipperDesloc, DeslocAdoptStep, WSDScheduleParams |
| M084 | deepspeed/ops/adam/cpu_adam.py | +272 | DeslocCPUStateManager, StateCheckpointer |
| M085 | deepspeed/runtime/lr_schedules.py | +311 | DeslocWSDSchedule, NesterovOuterOptimizer, CheckpointInitializer, DeslocPsiCalculator |
| M086 | deepspeed/runtime/zero/stage_1_and_2.py | +292 | DeslocZeroSyncManager, PartitionedMomentumSync, CommunicationCounter, GradientAccumulationCompat |
| M087 | deepspeed/utils/comms_logging.py | +273 | NeurIPSFigureSpec, DESLOC_FIGURE_SPECS (11 figures), FigureDataValidator, PlotDataPreparer |
| M088 | deepspeed/utils/timer.py | +245 | RooflineAnalyzer, ThroughputTracker, DeslocMFUTracker |
| M089 | REAL_GPU_BENCHMARK.py | +299 | EndToEndBenchmarkRunner (14 benchmarks × 3 seeds) |
| M090 | run_desloc_benchmark.sh | +465 | Full pipeline: detect→check→benchmark→aggregate→plot |
| M091 | accelerator/cuda_accelerator.py | +407 | AcceleratorCapabilityProbe, DeslocHardwareOptimizer, NeuronAcceleratorBridge |

## MANDATORY RULES FOR CLAUDE-6 (M092-M106)

1. **MODIFY EXISTING FILES ONLY** — no new standalone files. Each M = append to one existing .py/.sh
2. **NO numpy.random** — use torch.manual_seed or deterministic algorithms
3. **cat FILE FIRST** — always read the target file before modifying
4. **ast.parse AFTER** — verify Python syntax after every modification
5. **PRESERVE ALL PRIOR CODE** — grep for "End M0xx" markers to confirm nothing deleted
6. **DATA FROM LOGS** — all benchmark data must come from experiment log parsing, never hardcoded

## Claude-6 Task: M092-M106 — Pipeline Parallel + MoE + Profiling Integration

Target files (from the REPO, not experiment/src):

| M# | Target File | Task |
|---|---|---|
| M092 | deepspeed/runtime/pipe/engine.py | DES-LOC sync periods in pipeline parallel micro-batches |
| M093 | deepspeed/runtime/pipe/module.py | Layer-aware sync scheduling (different Kx per pipeline stage) |
| M094 | deepspeed/moe/sharded_moe.py | Expert-parallel DES-LOC: per-expert sync periods |
| M095 | deepspeed/moe/layer.py | MoE top-k gate with DES-LOC communication reduction |
| M096 | deepspeed/profiling/flops_profiler/profiler.py | DES-LOC communication overhead in FLOPS profiling |
| M097 | deepspeed/runtime/bf16_optimizer.py | BF16 precision DES-LOC convergence verification |
| M098 | deepspeed/runtime/fp16/fused_optimizer.py | FP16 mixed precision DES-LOC integration |
| M099 | deepspeed/checkpoint/deepspeed_checkpoint.py | DES-LOC state in distributed checkpoint format |
| M100 | deepspeed/runtime/dataloader.py | Data loading alignment with DES-LOC sync boundaries |
| M101 | deepspeed/launcher/runner.py | Multi-node DES-LOC experiment launcher CLI args |
| M102 | deepspeed/launcher/launch.py | torchrun-compatible DES-LOC launch parameters |
| M103 | deepspeed/runtime/activation_checkpointing/checkpointing.py | Activation checkpoint + DES-LOC memory optimization |
| M104 | deepspeed/runtime/compiler.py | torch.compile compatibility with DES-LOC sync hooks |
| M105 | deepspeed/runtime/config_utils.py | DES-LOC config validation utilities |
| M106 | deepspeed/runtime/constants.py | DES-LOC constant definitions (sync period defaults, etc.) |

CRITICAL: Read each target file with `cat` before modifying. These are REAL DeepSpeed source files with existing functionality that MUST NOT break.
