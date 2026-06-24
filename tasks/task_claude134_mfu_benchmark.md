# Claude-134: MFU Benchmark Script for Heterogeneous Cluster

## Context
Paper needs MFU (Model FLOPs Utilization) numbers comparing homogeneous vs heterogeneous training.

## Task
Create `benchmarks/mfu_hetero.py`:
1. Run 50 steps of 3B model on each config:
   - H100 only (1 GPU)
   - A6000 only (1 GPU)
   - H100 + 2×A6000 (3 GPU, DES-LOC)
   - H100 + 2×A6000 (3 GPU, naive DDP baseline)
2. Measure: tokens/sec, MFU, peak memory per GPU, step latency
3. MFU = actual_throughput / theoretical_peak
   - Theoretical for 3B LLaMA: 6 * N * T FLOPs per step (N=params, T=tokens)
   - H100 BF16 peak: 835 TFLOPS, A6000 BF16 peak: 38.7 TFLOPS
4. Output: markdown table + JSON for plotting

## Files
- `benchmarks/mfu_hetero.py` — new file

## Git
Branch: main. Commit with `--signoff`. Author: `dylanyunlon <dogechat@163.com>`.
Token: `SEE_CONFIG`
