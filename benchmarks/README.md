# Benchmark: Fused SwiGLU + RMSNorm vs Unfused Baseline

Partially addresses [#25](https://github.com/dylanyunlon/Neuron_SP/issues/25).

## What it measures

| Variant | Kernel launches | DRAM passes over data |
|---|---|---|
| **Fused** (`fused_swiglu_ln_kernel`) | 1 | 1 (single-pass) or 2 (two-pass, large hidden) |
| **Unfused** (separate SwiGLU + separate RMSNorm) | 3 | 3 (SwiGLU write + RMSNorm pass1 + pass2) |

The benchmark reports latency, effective HBM bandwidth, roofline percentage,
and numerical correctness (max / mean absolute error between fused and unfused outputs).

## Build

```bash
# For SM 8.6 (A6000 / RTX 3090)
make SM=86

# For SM 9.0 (H100)
make SM=90

# For SM 12.0 (Blackwell)
make SM=120
```

## Run

```bash
./bench_swiglu_ln [batch] [hidden] [warmup_iters] [bench_iters]

# defaults: batch=128, hidden=4096, warmup=50, bench=200
./bench_swiglu_ln
./bench_swiglu_ln 256 8192 100 500
```

## Roofline model

The fused kernel is **memory-bandwidth-bound** (arithmetic intensity ≈ 0.83 FLOP/byte).

**Estimated DRAM traffic per invocation:**

- Fused: `batch × hidden × 6 + hidden × 4` bytes
- Unfused: `batch × hidden × 12 + hidden × 4` bytes

The fused kernel achieves ~2× memory traffic reduction by eliminating the
intermediate BF16 buffer between SwiGLU and RMSNorm, and by using in-register
fusion (single-pass path) to avoid re-reading gate/up projections.

## Expected output

```
═══════════════════════════════════════════════════════════════
  Benchmark: Fused SwiGLU+RMSNorm  vs  Unfused Baseline
  batch=128  hidden=4096  warmup=50  iters=200  eps=1.0e-06
═══════════════════════════════════════════════════════════════
  ...
─────────────────────────────────────────────────────────────
  RESULTS
─────────────────────────────────────────────────────────────
                         Fused      Unfused      Ratio
  Latency              XX.X µs     XX.X µs     X.XXx
  Eff. bandwidth      XXX.X GB/s  XXX.X GB/s   X.XXx
  Roofline (% peak)    XX.X %      XX.X %
─────────────────────────────────────────────────────────────
```
