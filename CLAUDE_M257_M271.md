# CLAUDE_M272+ Handoff — Claude-18 Task Assignment

## What Claude-17 Did (M257-M271)

**CRITICAL CLEANUP**: Deleted 9,534 lines of bloated standalone DES-LOC classes
across 16 files. Added 1,078 lines of clean, properly-factored code.

### Key Classes Added (engine.py):
- `DeslocExperimentScheduler` — 108-config ablation matrix generator
- `DeslocStepProfiler` — per-step timing/memory/comm instrumentation  
- `DeslocNKIFAExporter` — NKI-FA format log writer for draw_plot.py
- `DeslocConvergenceBoundChecker` — Theorem 1 bound validation
- `desloc_comm_overhead_model()` — α+βN AllReduce latency model
- `desloc_sweep_Kx()` — RQ2 Kx sweep with ψ factor computation

### Key Functions Added (utils.py):
- `desloc_comm_reduction_ratio()` — 3/(1/Kx+1/Ku+1/Kv)
- `desloc_parse_nkifa_logfile()` — full NKI-FA + DES-LOC log parser
- `desloc_aggregate_experiments()` — multi-seed mean±std
- `desloc_power_law_fit_simple()` — pure-Python log-log regression
- `desloc_scan_log_directory()` — batch log parsing

### Key Additions (config.py):
- `DESLOC_FIGURE_SPECS` — 7 figure specifications (size/dpi/style/labels)
- `DESLOC_COLORS` — palettes for DDP/LocalAdam/DES-LOC
- `DESLOC_PRESETS` — quick_test/standard/large_scale experiment presets
- `desloc_validate_config()` — config validation with 6 constraint checks
- `desloc_config_to_cli_args()` — config→CLI arg string conversion

## MANDATORY RULES FOR CLAUDE-18 (M272+)

1. **MODIFY EXISTING FILES ONLY** — no new standalone .py files
2. **cat FILE FIRST** — always read target before modifying
3. **ast.parse AFTER** — verify Python syntax after every modification
4. **ZERO numpy.random** — use torch.manual_seed only
5. **SURGICAL MERGE** — inject at correct points, never append at EOF
6. **DELETE BEFORE ADD** — if you find bloated classes, delete them first
7. **NKI-FA FORMAT** — logs use `### config ### \n metric: value` format
8. **≥4 DECIMAL PLACES** — all loss/metric annotations (e.g., 3.2147)

## Claude-18 Task: M272-M286 — NeurIPS Figure Generation Pipeline

The experiment scheduler and log parser are done. Now build the plotting pipeline.

| M# | Target File | Task |
|---|---|---|
| M272 | REAL_GPU_BENCHMARK.py | Figure 1: Loss vs Step curves (7 Kx values, 2 models) |
| M273 | REAL_GPU_BENCHMARK.py | Figure 2: Comm reduction bars (DDP vs LocalAdam vs DES-LOC) |
| M274 | REAL_GPU_BENCHMARK.py | Figure 3: Half-life validation scatter (β₂ vs change rate) |
| M275 | REAL_GPU_BENCHMARK.py | Figure 4: Sync sensitivity bars (final loss vs Kx) |
| M276 | REAL_GPU_BENCHMARK.py | Figure 5: Large-scale training (125M/350M/1.3B curves) |
| M277 | REAL_GPU_BENCHMARK.py | Figure 6: Nesterov vs Averaging comparison |
| M278 | REAL_GPU_BENCHMARK.py | Figure 7: Adam vs ADOPT inner optimizer |
| M279 | deepspeed/runtime/engine.py | Integrate profiler into training loop |
| M280 | deepspeed/runtime/utils.py | Result validation + outlier detection |
| M281 | deepspeed/runtime/config.py | Auto Kx recommendation from hardware probe |
| M282 | deepspeed/comm/comm.py | Bandwidth measurement + tier-aware scheduling |
| M283 | deepspeed/utils/comms_logging.py | Structured comm event logger |
| M284 | deepspeed/utils/timer.py | Phase timer integration |
| M285 | run_desloc_benchmark.sh | Full 108-config sweep launcher script |
| M286 | deepspeed/runtime/constants.py | Figure spec constants cleanup |

CRITICAL: Each figure MUST read data from parsed experiment logs.
Use `desloc_parse_nkifa_logfile()` and `desloc_aggregate_experiments()`.
Follow NKI-FA da964f3 draw_plot.py style: seaborn whitegrid, bar annotations.
