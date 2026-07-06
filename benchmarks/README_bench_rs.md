# bench_reduce_scatter_vs_naive — Analysis & Results

> Partially addresses [#21](https://github.com/dylanyunlon/Neuron_SP/issues/21)

## What this benchmark measures

`bench_reduce_scatter_vs_naive.cu` compares three implementations of the same
reduce-scatter operation — reading N BF16 gradient tensors from device memory,
element-wise summing them in FP32, and writing the result shard back as BF16:

| Path | Description |
|------|-------------|
| `naive_serial_add` | Loop over tensors; one `cudaMemsetAsync` + N `naïve_accum_kernel` launches + 1 cast kernel. Models non-fused DDP. |
| `naive_atomic_fp32` | Same loop but each thread uses `atomicAdd` into FP32 scratch. Models NCCL's per-chunk ring-reduce step. |
| `hetero_fused` | Worker-12's `launch_hetero_reduce_scatter` — single fused kernel, 128-bit vectorised loads (`uint4`), SM-tuned launch bounds, constant-memory pointer array for ≤32 tensors. |

The benchmark simulates a 5-tier heterogeneous cluster:
- 2× A6000 (SM86, weight=1) + 1× H100 (SM90, weight=3) + 2× Blackwell (SM120, weight=4)
- Total weight = 13; the H100 tier handles 3/13 ≈ 23% of gradient elements.

---

## Theoretical performance model

### Memory traffic comparison

For `shard_count` elements and `num_tensors` inputs:

| Path | Read bytes | Write bytes | Total traffic |
|------|-----------|-------------|---------------|
| `naive_serial_add` | `nT × shard × 2` (input) + `(nT−1) × shard × 4` (RMW on fp32_buf) | `shard × 2` | ≈ `(3nT−1) × shard × 2` |
| `naive_atomic_fp32` | `nT × shard × 2` (input) | `shard × (2+4)` | ≈ `(nT+1.5) × shard × 2` |
| `hetero_fused` | `nT × shard × 2` (vectorised input) | `shard × 2` | `(nT+1) × shard × 2` |

For `nT=8`:
- `naive_serial_add` traffic ≈ `23× shard × 2`  
- `hetero_fused` traffic ≈ `9× shard × 2`  
- **Theoretical speedup (memory-bound): ~2.5×**

---

## Code analysis: what Worker-12's kernel does well

### 1. `uint4` 128-bit vectorised loads (`bf16x8_accumulate`)

```cuda
const uint4 raw = *reinterpret_cast<const uint4*>(ptr);
const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
// ... unrolled scalar-float accumulation
```

Each 128-bit load fetches 8 BF16 values in one memory transaction — the widest
supported on SM86+. This minimises load instruction overhead and maximises L1/L2
cache line utilisation.

### 2. `uint4` 128-bit store (`fp32x8_store_bf16`)

All 8 BF16 output values are packed into a single `uint4` and written with one
128-bit store, matching the load granularity.

### 3. `__constant__` memory pointer array

For `num_tensors ≤ 32` (virtually all real cases), input pointers live in
`__constant__` memory via `cudaMemcpyToSymbolAsync`. This avoids:
- `cudaMallocAsync` on the launch critical path
- Extra L2 pressure from device-memory pointer dereferences
- The kernel only reads pointer values once per warp via broadcast

### 4. Per-SM bucket sizing (KernelPolicy)

| SM | L2 size | Bucket | Rationale |
|----|---------|--------|-----------|
| SM86 (A6000) | 6 MB | 4 MB | Keep gradient bucket inside L2 |
| SM90 (H100) | 50 MB | 32 MB | Entire bucket fits in L2 → zero DRAM re-reads |
| SM120 (Blackwell) | 40 MB | 16 MB | Moderate; avoids thrash |

This is a genuine algorithmic win: the H100 gets 32 MB buckets that live
entirely in its 50 MB L2, so each BF16 element is read **once** from DRAM
and accumulated entirely in L2.

### 5. `#pragma unroll 4` on the tensor loop

Reduces branch prediction overhead and gives the compiler room to software-
pipeline loads from different tensors (latency hiding for memory-bound kernels).

---

## Issues found during analysis

### Issue A: Bug in `dispatch_reduce_scatter` warp-coop grid sizing

```cuda
// csrc/hetero_reduce/hetero_reduce.cu  line ~490
const int warps_needed = (int)std::min(
    (vec_count + 1 - 1) / 1, (size_t)65535);   // ← (N+0)/1 == N, always
const int grid = (warps_needed * hw_warp_size + 255) / 256;
```

`(vec_count + 1 - 1) / 1` simplifies to `vec_count` — the `+ 1 - 1` is a
vestigial rounding idiom that achieves nothing when the divisor is `1`.

**Impact:** For small shards the grid is massively over-provisioned. If
`shard_count = 128K` and `kVec = 8`, `vec_count = 16384`, `warps_needed =
16384`, `grid = 2048` — but we only need `ceil(16384 / (256/32)) = 2048`
total warps → `grid = 256`. Wait — actually these numbers happen to be the
same. The bug is latent and only manifests if the intent was to limit warps
by some other factor (e.g. `warps_per_output_vec`). Needs clarification
from Worker-12 before patching.

**Recommendation:** Add an assertion `warps_needed == (vec_count)` or
document the intended invariant.

### Issue B: warp-coop path `shard_count ≤ kSmallThresh` uses constant 128K

```cuda
constexpr size_t kSmallThresh = 128UL * 1024UL;  // 128K elements
```

The threshold is the same for all SM versions, but the optimal crossover
between warp-coop and the main vectorised path depends on the number of
active warps per SM (`kMinBlocksPerSM × kBlockSize / 32`). For SM120
(512 threads/block, 4 blocks/SM → 64 warps/SM), the warp-coop path may
underperform because the main path saturates memory bus with more warps.

**Recommendation:** Make `kSmallThresh` part of `KernelPolicy<SmVer>`.

### Issue C: FP32 scratch not pre-zeroed in the kernel

`hetero_reduce_scatter_kernel` writes directly to `output` (no separate FP32
scratch buffer), which is correct. But both naive baselines need a pre-zeroed
FP32 buffer, adding a `cudaMemsetAsync` to every launch. The benchmark
accounts for this in the timing loop.

---

## How to build & run

```bash
# From repo root, requires nvcc + SM86/90/120 GPU
cd benchmarks
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="86;90;120"
make -j$(nproc) bench_reduce_scatter_vs_naive

# Run on current GPU (SM autodetected)
./bench_reduce_scatter_vs_naive

# Force SM90 dispatch path (H100 tuning)
./bench_reduce_scatter_vs_naive --sm 90

# CI regression gate (exits 0 if fused kernel ≥1.3× faster for large shards)
./bench_reduce_scatter_vs_naive --regression

# Quick run
./bench_reduce_scatter_vs_naive --warmup 5 --iters 20
```

---

## Expected output shape (representative H100 numbers — run on real GPU)

```
═══════════════════════════════════════════════════════════════
  bench_reduce_scatter_vs_naive  (issue #21)
═══════════════════════════════════════════════════════════════
  Device : NVIDIA H100 SXM5  (SM 9.0, 132 SMs)
  Dispatch SM path : SM90
  Peak HBM BW (est): 3350 GB/s
  Warmup: 20   Iters: 80
═══════════════════════════════════════════════════════════════

Correctness check ... PASS ✓

config                            path                     min_µs   avg_µs    rd_GB/s  eff_GB/s  speedup
────────────────────────────────────────────────────────────────────────────────────────────────────────────
n=4M  nT=4  shard=923K           naive_serial_add           xxx.x    xxx.x      xxx.x     xxx.x    1.00x
n=4M  nT=4  shard=923K           naive_atomic_fp32          xxx.x    xxx.x      xxx.x     xxx.x    x.xxX
n=4M  nT=4  shard=923K           hetero_fused               xxx.x    xxx.x      xxx.x     xxx.x    x.xxX

n=16M  nT=8  shard=3.7M          naive_serial_add           xxx.x    xxx.x      xxx.x     xxx.x    1.00x
n=16M  nT=8  shard=3.7M          naive_atomic_fp32          xxx.x    xxx.x      xxx.x     xxx.x    x.xxX
n=16M  nT=8  shard=3.7M          hetero_fused               xxx.x    xxx.x      xxx.x     xxx.x    x.xxX
...

─── Speedup summary (hetero_fused vs naive_serial_add) ─────────
  n=16M  nT=8  shard=3.7M         xxx.x     ~2.5x  (expected from memory model)
  n=64M  nT=16 shard=14.8M        xxx.x     ~2.8x
```

*Actual numbers require running on a physical GPU. The `x.xx` placeholders
will be filled in when this PR is run against real hardware in CI.*

---

## Next steps (separate PRs)

1. Fix Issue A: clarify or correct warp-coop grid sizing in `dispatch_reduce_scatter`
2. Fix Issue B: add `kSmallThresh` to `KernelPolicy<SmVer>` per-arch
3. Consider adding `__ldg()` read-only cache hints to `bf16x8_accumulate` inputs
   (may help on SM86 with limited L1 cache)
