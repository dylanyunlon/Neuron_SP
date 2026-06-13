# CLAUDE_M167_M181.md — Claude-11 Handoff to Claude-12 (M182+)

## What This Commit Series Does

**Claude-11 (M167-M181)**: 15 critical merges + 1 cleanup commit

### Key Decisions Made
1. **DELETED ~13,500 lines** of appended virtual/simulation code from Claude-1 through Claude-8
2. **Restored ALL 36 modified files** to their original DeepSpeed baseline (commit ee088280)
3. **Re-merged DES-LOC** as surgical injections: total Δ = +386 lines across 15 files
4. **Zero numpy.random** in any DES-LOC code (original DeepSpeed's set_random_seed preserved)

### Files Modified (M167-M181)

| M# | File | Δ Lines | What Was Injected |
|---|---|---|---|
| M167 | deepspeed/utils/comms_logging.py | +108 | DESLOC_COMM_TIERS, classify_comm_tier(), should_sync_at_step(), compute_comm_reduction_ratio() |
| M168 | deepspeed/utils/timer.py | +73 | DESLOC_PHASE_* constants, accumulate_desloc_phase(), get_desloc_mfu_breakdown(), numpy→pure-python |
| M169 | deepspeed/runtime/utils.py | +58 | desloc_half_life(), desloc_recommend_sync_periods(), overflow check annotation |
| M170 | deepspeed/runtime/config.py | +49 | DESLOC config constants, get_desloc_config(), wired into _initialize_params |
| M171 | deepspeed/runtime/engine.py | +42 | **CORE**: desloc_is_sync_boundary(), Kx gate in allreduce_gradients(), desloc_step counter |
| M172 | deepspeed/comm/comm.py | +0 | desloc_tier param added to all_reduce/reduce_scatter_fn signatures |
| M173 | deepspeed/runtime/lr_schedules.py | +4 | WSD schedule Kx-alignment annotation |
| M174 | REAL_GPU_BENCHMARK.py | +6 | NKI-FA log format config, clip_rho param |
| M175 | deepspeed/comm/torch.py | +3 | TorchBackend DES-LOC docstring |
| M176 | deepspeed/comm/backend.py | +0 | desloc_bytes_sent/ops_count in Backend.__init__ |
| M177 | deepspeed/ops/adam/fused_adam.py | +7 | desloc_clip_rho param in step(), per-coordinate clipping annotation |
| M178 | deepspeed/ops/adam/cpu_adam.py | +4 | CPU offload sync period annotation |
| M179 | deepspeed/runtime/zero/stage_1_and_2.py | +9 | Kx gating annotations in allreduce_bucket, reduce_ready_partitions |
| M180 | accelerator/cuda_accelerator.py | +23 | desloc_device_info() for GPU capability probing |
| M181 | run_desloc_benchmark.sh | +257 | Complete experiment runner: detect→matrix→run→parse→summarize |

### Architecture After M181

```
DES-LOC Data Flow:
  config.py (Kx/Ku/Kv parsing)
    → engine.py (desloc_is_sync_boundary + allreduce gating)
      → comm.py (desloc_tier parameter on all_reduce)
        → comms_logging.py (tier-level byte tracking)
          → timer.py (phase timing: compute vs comm)
            → run_desloc_benchmark.sh (experiment matrix runner)
```

## MANDATORY RULES FOR CLAUDE-12 (M182+)

1. **CRITICAL MERGE ONLY** — inject at correct points, do NOT append blocks at EOF
2. **cat FILE FIRST** — always read the target file before modifying
3. **ast.parse AFTER** — verify Python syntax after every modification
4. **ZERO numpy.random** — use torch.manual_seed or deterministic math
5. **Kx=1 MUST degrade** to original DeepSpeed behavior (no extra overhead)
6. **Data from logs** — all benchmark data parsed from experiment output, never hardcoded
7. **NKI-FA format** — logs use `### config = {...} ###\nmetric: value` format
8. **Reference infra repos** — cite Megatron/NCCL/CCCL patterns when injecting code

## Claude-12 Task: M182-M196 — Scaling Law Predictor

Target: inject DES-LOC-aware scaling law utilities into existing files.

| M# | Target File | Task |
|---|---|---|
| M182 | deepspeed/runtime/engine.py | Scaling law loss predictor: given FLOPS→predicted loss |
| M183 | deepspeed/runtime/config.py | Compute-optimal {N,D,Kx} config generator |
| M184 | deepspeed/runtime/utils.py | Power-law regression: loss vs compute fitting |
| M185 | deepspeed/comm/comm.py | Communication overhead correction term in scaling |
| M186 | deepspeed/runtime/lr_schedules.py | WSD params scaling with model size |
| M187 | deepspeed/utils/timer.py | MFU prediction as f(N,D,Kx) |
| M188 | deepspeed/profiling/flops_profiler/profiler.py | FLOPS split: compute/comm/idle |
| M189 | deepspeed/runtime/bf16_optimizer.py | BF16 scaling law precision correction |
| M190 | deepspeed/runtime/config_utils.py | Scaling law config validation |
| M191 | deepspeed/runtime/constants.py | Scaling law constant definitions |
| M192 | REAL_GPU_BENCHMARK.py | Multi-scale sweep: 125M→350M→1B |
| M193 | deepspeed/ops/adam/fused_adam.py | Adam hyperparams scaling with model size |
| M194 | deepspeed/runtime/pipe/engine.py | Pipeline parallel scaling correction |
| M195 | deepspeed/moe/sharded_moe.py | MoE expert count scaling |
| M196 | run_desloc_benchmark.sh | Auto scaling law verification experiment |

CRITICAL: Each M number = ~400 lines of surgical edits (not appends). Read the file first, find the correct injection point, merge cleanly.
