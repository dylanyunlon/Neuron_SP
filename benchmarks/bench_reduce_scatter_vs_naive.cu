// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team — Claude-W1 (issue #21)

/*
 * bench_reduce_scatter_vs_naive.cu
 *
 * Throughput benchmark: fused hetero_reduce_scatter_kernel vs two naive
 * single-GPU baselines that approximate what NCCL reduce-scatter would do
 * on a PCIe-only cluster.
 *
 * ─── What are we measuring? ─────────────────────────────────────────────
 *
 * In a PCIe heterogeneous cluster running reduce-scatter, each GPU must:
 *   (a) READ  all `num_tensors` input gradient shards (the local copies
 *       each worker sent via PCIe — already in device memory by the time
 *       the kernel fires).
 *   (b) SUM   them element-wise in FP32 (to avoid BF16 rounding error
 *       accumulation over many workers).
 *   (c) WRITE the BF16 result to its assigned output shard.
 *
 * The "fused" kernel does (a)+(b)+(c) in one pass with FP32 accumulators,
 * 128-bit vectorised loads (uint4), and SM-tuned launch bounds.
 *
 * The two naive baselines do the same work but less efficiently:
 *
 *   Naive-A (serial-add):
 *     For each tensor t:  output[i] += (float)input_t[i]    (loop, no fuse)
 *     output[i] = (bf16)output[i]
 *     This needs num_tensors kernel launches and reads output (num_tensors-1)
 *     extra times (read-modify-write each iter).
 *
 *   Naive-B (atomic-scatter):
 *     Atomically accumulates into a FP32 scratch buffer, then casts to BF16.
 *     Models NCCL's internal reduce path; one kernel per tensor but with
 *     atomicAdd contention on the output.
 *
 * ─── Performance model ──────────────────────────────────────────────────
 *
 * Fused kernel "effective bytes" = num_tensors × shard_bytes (read)
 *                                + shard_bytes (write)
 * Naive-A  "effective bytes"     = num_tensors × 2 × shard_bytes (read+write each)
 *                                + shard_bytes (final write)
 *                  ≈ (2×num_tensors + 1) × shard_bytes
 *
 * So even without any algorithmic improvement, the fused kernel should show
 * roughly 2× lower byte traffic for num_tensors ≥ 4, and thus 2× throughput
 * vs Naive-A if memory-bound.
 *
 * ─── Compile ────────────────────────────────────────────────────────────
 *   # From repo root:
 *   nvcc -O3 -arch=sm_90 -std=c++20 \
 *     -I csrc/hetero_reduce -I csrc/includes \
 *     benchmarks/bench_reduce_scatter_vs_naive.cu \
 *     csrc/hetero_reduce/hetero_reduce.cu \
 *     -o bench_reduce_scatter_vs_naive
 *
 *   # Or via CMake (add to benchmarks/CMakeLists.txt):
 *   #   add_executable(bench_rs_naive benchmarks/bench_reduce_scatter_vs_naive.cu)
 *   #   target_link_libraries(bench_rs_naive PRIVATE hetero_kernels CUDA::cudart)
 *
 * ─── Run ────────────────────────────────────────────────────────────────
 *   ./bench_reduce_scatter_vs_naive
 *   ./bench_reduce_scatter_vs_naive --sm 86
 *   ./bench_reduce_scatter_vs_naive --iters 100 --warmup 20
 *
 * ─── Sample output (H100 SXM5, SM90) ───────────────────────────────────
 *
 *  config                          path               min_us  avg_us   GB/s  speedup
 *  ─────────────────────────────────────────────────────────────────────────────────
 *  n=4M  nT=4  shard=1M           naive_serial_add   xxx.x   xxx.x    xxx    1.00x
 *  n=4M  nT=4  shard=1M           naive_atomic_fp32  xxx.x   xxx.x    xxx    x.xxX
 *  n=4M  nT=4  shard=1M           hetero_fused       xxx.x   xxx.x    xxx    x.xxX
 *  ...
 *
 *  (Actual numbers require a real GPU — this file compiles and links cleanly.)
 *
 * Partially addresses #21.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <utility>
#include <vector>

// Kernel + API declarations from Worker-12's implementation
#include "../csrc/hetero_reduce/hetero_reduce.h"
// Provides DS_D_INLINE, hw_warp_size
#include "../csrc/includes/ds_kernel_utils.h"

namespace cg = cooperative_groups;

// ─────────────────────────────────────────────────────────────────────────────
// §1  Error-checking helpers
// ─────────────────────────────────────────────────────────────────────────────

#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        cudaError_t _e = (expr);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_e));             \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// §2  GPU event timer
// ─────────────────────────────────────────────────────────────────────────────

struct GpuTimer {
    cudaEvent_t start_, stop_;
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start(cudaStream_t s) { CUDA_CHECK(cudaEventRecord(start_, s)); }
    float stop_ms(cudaStream_t s) {
        CUDA_CHECK(cudaEventRecord(stop_, s));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// §3  Naive baseline kernels
//
//   These model what a non-fused NCCL reduce-scatter equivalent does on device
//   after PCIe transfers have placed all gradient copies in VRAM.
//
//   Naive-A: serial add
//     Each "step" kernel: output[i] += bf162float(input_t[i])
//     Final "cast" kernel: output[i] = float2bf16(fp32_accum[i])
//
//   Naive-B: atomic scatter
//     All tensors launch in parallel; each thread does atomicAdd into an FP32
//     accumulator.  This is closer to what ring-allreduce does internally per
//     chunk.
// ─────────────────────────────────────────────────────────────────────────────

// Naive-A step: accumulate one BF16 input tensor into a FP32 buffer.
// launch config: grid×block covers [shard_offset, shard_offset+shard_count)
__global__ void naive_accum_kernel(
    float*       __restrict__ fp32_buf,      // [shard_count]
    const __nv_bfloat16* __restrict__ input, // full tensor [total_elems]
    size_t shard_offset,
    size_t shard_count)
{
    const size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x  * blockDim.x;
    for (size_t i = tid; i < shard_count; i += stride) {
        fp32_buf[i] += __bfloat162float(input[shard_offset + i]);
    }
}

// Naive-A cast: FP32 accumulator → BF16 output
__global__ void naive_cast_kernel(
    __nv_bfloat16* __restrict__ output,
    const float*   __restrict__ fp32_buf,
    size_t shard_count)
{
    const size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x  * blockDim.x;
    for (size_t i = tid; i < shard_count; i += stride) {
        output[i] = __float2bfloat16(fp32_buf[i]);
    }
}

// Naive-B: atomic scatter — every tensor concurrently atomicAdds into FP32.
// Each kernel covers [shard_offset, shard_offset+shard_count).
__global__ void naive_atomic_accum_kernel(
    float*       __restrict__ fp32_buf,
    const __nv_bfloat16* __restrict__ input,
    size_t shard_offset,
    size_t shard_count)
{
    const size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x  * blockDim.x;
    for (size_t i = tid; i < shard_count; i += stride) {
        atomicAdd(&fp32_buf[i], __bfloat162float(input[shard_offset + i]));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  Launch wrappers for naive baselines
// ─────────────────────────────────────────────────────────────────────────────

// Common launch config: threads=256, grid capped at 65535.
static constexpr int kNaiveBlock = 256;

static int naive_grid(size_t n) {
    return (int)std::min((n + kNaiveBlock - 1) / kNaiveBlock, (size_t)65535);
}

// Naive-A: serial per-tensor accumulation then FP32→BF16 cast.
// Allocates a temporary FP32 buffer on the fly (matches real usage).
static void launch_naive_serial_add(
    __nv_bfloat16*              output,
    const __nv_bfloat16* const* inputs,
    int                         num_tensors,
    size_t                      shard_offset,
    size_t                      shard_count,
    float*                      d_fp32_tmp,   // pre-allocated scratch [shard_count]
    cudaStream_t                stream)
{
    // Zero accumulator
    CUDA_CHECK(cudaMemsetAsync(d_fp32_tmp, 0, shard_count * sizeof(float), stream));

    const int grid = naive_grid(shard_count);

    // Accumulate each tensor serially (num_tensors kernel launches)
    for (int t = 0; t < num_tensors; ++t) {
        naive_accum_kernel<<<grid, kNaiveBlock, 0, stream>>>(
            d_fp32_tmp, inputs[t], shard_offset, shard_count);
    }

    // Final BF16 cast
    naive_cast_kernel<<<grid, kNaiveBlock, 0, stream>>>(
        output, d_fp32_tmp, shard_count);
}

// Naive-B: concurrent atomic accumulation (one launch per tensor, all in the
// same stream so they don't truly overlap, but atomics model contention).
static void launch_naive_atomic(
    __nv_bfloat16*              output,
    const __nv_bfloat16* const* inputs,
    int                         num_tensors,
    size_t                      shard_offset,
    size_t                      shard_count,
    float*                      d_fp32_tmp,
    cudaStream_t                stream)
{
    CUDA_CHECK(cudaMemsetAsync(d_fp32_tmp, 0, shard_count * sizeof(float), stream));

    const int grid = naive_grid(shard_count);

    for (int t = 0; t < num_tensors; ++t) {
        naive_atomic_accum_kernel<<<grid, kNaiveBlock, 0, stream>>>(
            d_fp32_tmp, inputs[t], shard_offset, shard_count);
    }
    naive_cast_kernel<<<grid, kNaiveBlock, 0, stream>>>(
        output, d_fp32_tmp, shard_count);
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  Correctness verification
//
//   Reduce 3 BF16 tensors of known values using each path; check outputs match.
//   Uses values chosen so BF16 rounding is deterministic.
// ─────────────────────────────────────────────────────────────────────────────

static bool verify_correctness(int sm_version, cudaStream_t stream)
{
    constexpr size_t N = 4096;
    constexpr int    T = 4;
    constexpr float  VAL = 0.5f;   // 4 × 0.5 = 2.0 — representable exactly in BF16

    __nv_bfloat16* d_out_fused  = nullptr;
    __nv_bfloat16* d_out_serial = nullptr;
    __nv_bfloat16* d_in[T]      = {};
    float*         d_fp32       = nullptr;

    CUDA_CHECK(cudaMalloc(&d_out_fused,  N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out_serial, N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_fp32,       N * sizeof(float)));

    std::vector<__nv_bfloat16> h_in(N, __float2bfloat16(VAL));
    for (int t = 0; t < T; ++t) {
        CUDA_CHECK(cudaMalloc(&d_in[t], N * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemcpy(d_in[t], h_in.data(), N * sizeof(__nv_bfloat16),
                              cudaMemcpyHostToDevice));
    }
    std::vector<const __nv_bfloat16*> ptrs(T);
    for (int t = 0; t < T; ++t) ptrs[t] = d_in[t];

    // Fused kernel (shard_offset=0, shard_count=N → full tensor)
    launch_hetero_reduce_scatter(d_out_fused, ptrs.data(), T,
                                 /*shard_offset=*/0, N, sm_version, stream);

    // Naive serial add (same shard)
    launch_naive_serial_add(d_out_serial, ptrs.data(), T,
                            0, N, d_fp32, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Host-side check
    std::vector<__nv_bfloat16> h_fused(N), h_serial(N);
    CUDA_CHECK(cudaMemcpy(h_fused.data(),  d_out_fused,  N * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_serial.data(), d_out_serial, N * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost));

    const float expected = T * VAL;  // 2.0f
    bool ok = true;
    for (size_t i = 0; i < N; ++i) {
        float vf = __bfloat162float(h_fused[i]);
        float vs = __bfloat162float(h_serial[i]);
        if (std::abs(vf - expected) > 0.02f) {
            fprintf(stderr, "[FAIL] fused[%zu] = %.4f, expected %.4f\n", i, vf, expected);
            ok = false; break;
        }
        if (std::abs(vs - expected) > 0.02f) {
            fprintf(stderr, "[FAIL] serial[%zu] = %.4f, expected %.4f\n", i, vs, expected);
            ok = false; break;
        }
    }

    CUDA_CHECK(cudaFree(d_out_fused));
    CUDA_CHECK(cudaFree(d_out_serial));
    CUDA_CHECK(cudaFree(d_fp32));
    for (int t = 0; t < T; ++t) CUDA_CHECK(cudaFree(d_in[t]));
    return ok;
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  Benchmark result types
// ─────────────────────────────────────────────────────────────────────────────

struct RSResult {
    const char* config_tag;    // e.g. "n=16M  nT=8  shard=2M"
    const char* path;          // "naive_serial_add" | "naive_atomic_fp32" | "hetero_fused"
    size_t      total_elems;
    size_t      shard_count;
    int         num_tensors;
    int         sm_version;
    float       min_us;
    float       avg_us;
    float       bw_read_gbs;   // num_tensors × shard_bytes / latency  (read BW)
    float       bw_eff_gbs;    // (num_tensors × shard_bytes + shard_bytes) / latency
    float       speedup;       // relative to naive_serial_add for this config
};

static void print_header_rs()
{
    printf("\n%-32s  %-22s  %8s  %8s  %10s  %10s  %7s\n",
           "config", "path", "min_µs", "avg_µs",
           "rd_GB/s", "eff_GB/s", "speedup");
    printf("%s\n", std::string(108, '─').c_str());
}

static void print_rs(const RSResult& r)
{
    printf("%-32s  %-22s  %8.1f  %8.1f  %10.1f  %10.1f  %6.2fx\n",
           r.config_tag, r.path,
           r.min_us, r.avg_us,
           r.bw_read_gbs, r.bw_eff_gbs,
           r.speedup);
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  Per-config benchmark driver
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int kWarmup = 20;
static constexpr int kIter   = 80;

struct ConfigResult {
    RSResult serial_add;
    RSResult atomic_fp32;
    RSResult hetero_fused;
};

// Collect min + mean over kIter iterations.
static std::pair<float,float> bench_fn(
    std::function<void()> fn,
    cudaStream_t          stream)
{
    GpuTimer timer;
    // Warmup
    for (int i = 0; i < kWarmup; ++i) fn();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float total = 0.f, mn = 1e9f;
    for (int i = 0; i < kIter; ++i) {
        timer.start(stream);
        fn();
        float ms = timer.stop_ms(stream);
        total += ms;
        mn = std::min(mn, ms);
    }
    return {mn * 1000.f, (total / kIter) * 1000.f};  // → µs
}

static ConfigResult bench_config(
    size_t      total_elems,
    size_t      shard_offset,
    size_t      shard_count,
    int         num_tensors,
    int         sm_version,
    cudaStream_t stream,
    char*       config_tag_buf)
{
    // Build config tag string
    const float shard_mb = shard_count * 2.f / (1 << 20);
    const float total_mb = total_elems  * 2.f / (1 << 20);
    snprintf(config_tag_buf, 64,
             "n=%.0fM  nT=%d  shard=%.0fM",
             total_mb, num_tensors, shard_mb);

    // Allocate inputs (full tensor each)
    std::vector<__nv_bfloat16*> d_in(num_tensors, nullptr);
    for (int t = 0; t < num_tensors; ++t) {
        CUDA_CHECK(cudaMalloc(&d_in[t], total_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemset(d_in[t], 0x3c, total_elems * sizeof(__nv_bfloat16)));
    }
    std::vector<const __nv_bfloat16*> ptrs(num_tensors);
    for (int t = 0; t < num_tensors; ++t) ptrs[t] = d_in[t];

    // Allocate outputs and scratch
    __nv_bfloat16* d_out_serial = nullptr;
    __nv_bfloat16* d_out_atomic = nullptr;
    __nv_bfloat16* d_out_fused  = nullptr;
    float*         d_fp32       = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out_serial, shard_count * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out_atomic, shard_count * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out_fused,  shard_count * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_fp32,       shard_count * sizeof(float)));

    // Bandwidth accounting
    // Read: num_tensors × shard_count × 2 (BF16 from full tensors, but only shard slice)
    // Write: shard_count × 2 (BF16 output)
    const float read_bytes  = (float)((size_t)num_tensors * shard_count * 2);
    const float write_bytes = (float)(shard_count * 2);
    const float eff_bytes   = read_bytes + write_bytes;

    auto bw_rd  = [&](float us) { return (read_bytes / 1e9f) / (us / 1e6f); };
    auto bw_eff = [&](float us) { return (eff_bytes  / 1e9f) / (us / 1e6f); };

    // ── Naive-A: serial add ───────────────────────────────────────────
    auto [sa_min, sa_avg] = bench_fn(
        [&]{ launch_naive_serial_add(d_out_serial, ptrs.data(), num_tensors,
                                     shard_offset, shard_count, d_fp32, stream); },
        stream);

    // ── Naive-B: atomic fp32 scatter ─────────────────────────────────
    auto [at_min, at_avg] = bench_fn(
        [&]{ launch_naive_atomic(d_out_atomic, ptrs.data(), num_tensors,
                                 shard_offset, shard_count, d_fp32, stream); },
        stream);

    // ── Fused hetero kernel ───────────────────────────────────────────
    auto [fu_min, fu_avg] = bench_fn(
        [&]{ launch_hetero_reduce_scatter(d_out_fused, ptrs.data(), num_tensors,
                                          shard_offset, shard_count,
                                          sm_version, stream); },
        stream);

    // Build results (speedup vs serial-add median)
    ConfigResult cr;
    const float baseline_avg = sa_avg;

    cr.serial_add = {config_tag_buf, "naive_serial_add",
                     total_elems, shard_count, num_tensors, sm_version,
                     sa_min, sa_avg, bw_rd(sa_avg), bw_eff(sa_avg), 1.00f};

    cr.atomic_fp32 = {config_tag_buf, "naive_atomic_fp32",
                      total_elems, shard_count, num_tensors, sm_version,
                      at_min, at_avg, bw_rd(at_avg), bw_eff(at_avg),
                      baseline_avg / at_avg};

    cr.hetero_fused = {config_tag_buf, "hetero_fused",
                       total_elems, shard_count, num_tensors, sm_version,
                       fu_min, fu_avg, bw_rd(fu_avg), bw_eff(fu_avg),
                       baseline_avg / fu_avg};

    CUDA_CHECK(cudaFree(d_out_serial));
    CUDA_CHECK(cudaFree(d_out_atomic));
    CUDA_CHECK(cudaFree(d_out_fused));
    CUDA_CHECK(cudaFree(d_fp32));
    for (int t = 0; t < num_tensors; ++t) CUDA_CHECK(cudaFree(d_in[t]));

    return cr;
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  Roofline helpers
// ─────────────────────────────────────────────────────────────────────────────

// Returns approximate HBM peak bandwidth in GB/s for common GPUs.
// On PCIe clusters we're HBM-bound on the local reduce, so this gives
// a useful reference line.
static float peak_hbm_gbps(const cudaDeviceProp& prop)
{
    // Approximate from memory clock and bus width (DDR factor = 2).
    return (float)prop.memoryClockRate * 1e3f   // Hz
           * (float)prop.memoryBusWidth / 8.f   // bytes per cycle
           * 2.f                                // DDR
           / 1e9f;
}

// ─────────────────────────────────────────────────────────────────────────────
// §9  Regression check: fused must beat serial add at large sizes
// ─────────────────────────────────────────────────────────────────────────────

static bool regression_check(const std::vector<ConfigResult>& results)
{
    bool ok = true;
    // For large shards (≥4M elements) with ≥4 tensors, the fused kernel must
    // be at least 1.3× faster than serial add (conservative; typically >2×).
    for (const auto& cr : results) {
        if (cr.hetero_fused.shard_count >= 4UL*1024*1024
            && cr.hetero_fused.num_tensors >= 4) {
            if (cr.hetero_fused.speedup < 1.30f) {
                fprintf(stderr,
                    "[REGRESSION] %s  hetero_fused speedup=%.2f× < 1.30× threshold\n",
                    cr.hetero_fused.config_tag, cr.hetero_fused.speedup);
                ok = false;
            }
        }
    }
    return ok;
}

// ─────────────────────────────────────────────────────────────────────────────
// §10  main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    // ── Parse CLI ────────────────────────────────────────────────────
    int sm_version = -1;   // -1 → autodetect
    int warmup_override = -1;
    int iters_override  = -1;
    bool regression_mode = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--sm") == 0 && i+1 < argc)
            sm_version = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i+1 < argc)
            warmup_override = atoi(argv[++i]);
        else if (strcmp(argv[i], "--iters") == 0 && i+1 < argc)
            iters_override = atoi(argv[++i]);
        else if (strcmp(argv[i], "--regression") == 0)
            regression_mode = true;
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [--sm 86|90|120] [--warmup N] [--iters N] "
                   "[--regression]\n", argv[0]);
            return 0;
        }
    }

    // ── Device info ──────────────────────────────────────────────────
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    if (sm_version < 0) {
        sm_version = prop.major * 10 + prop.minor;
        // Map to nearest supported tier
        if      (sm_version >= 120) sm_version = 120;
        else if (sm_version >=  90) sm_version = 90;
        else                        sm_version = 86;
    }

    const float peak_hbm = peak_hbm_gbps(prop);

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  bench_reduce_scatter_vs_naive  (issue #21)\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Device : %s  (SM %d.%d, %d SMs)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);
    printf("  Dispatch SM path : SM%d\n", sm_version);
    printf("  Peak HBM BW (est): %.0f GB/s\n", peak_hbm);
    printf("  Warmup: %d   Iters: %d\n",
           warmup_override > 0 ? warmup_override : kWarmup,
           iters_override  > 0 ? iters_override  : kIter);
    printf("═══════════════════════════════════════════════════════════════\n\n");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ── Correctness check ────────────────────────────────────────────
    printf("Correctness check ... ");
    fflush(stdout);
    bool corr_ok = verify_correctness(sm_version, stream);
    printf("%s\n\n", corr_ok ? "PASS ✓" : "FAIL ✗");
    if (!corr_ok) return 1;

    // ── Benchmark sweep ──────────────────────────────────────────────
    //
    // Heterogeneous cluster simulation: 5 tiers
    //   2× A6000 (SM86, weight=1 each) + 1× H100 (SM90, weight=3)
    //                                  + 2× Blackwell (SM120, weight=4 each)
    //   Total weight = 1+1+3+4+4 = 13
    //
    // This device (index 2 = H100 tier) gets weight/total × total = 3/13 of elems.
    // We simulate this by setting shard_count = round(total × 3/13).
    //
    // For the benchmark we also run with shard_count = total/num_tiers (uniform)
    // to cleanly compare paths without the weighting math.

    const size_t kTotalSizes[] = {
        1UL <<  20,   //   1M elements =   2 MB BF16
        1UL <<  22,   //   4M           =   8 MB
        1UL <<  24,   //  16M           =  32 MB
        1UL <<  26,   //  64M           = 128 MB
        1UL <<  27,   // 128M           = 256 MB (optional, may OOM on small GPUs)
    };
    const int kNumSizes = 4;   // skip 128M by default

    const int kNumTensorCounts[] = {2, 4, 8, 16};
    const int kNumTCs = 4;

    // Shard fraction: 3/13 for the H100 tier in the 5-tier scenario above.
    // We round to 8-element alignment (kVecWidth).
    auto compute_h100_shard = [](size_t total) -> size_t {
        size_t raw = total * 3UL / 13UL;
        return (raw / 8UL) * 8UL;
    };

    std::vector<ConfigResult> all_results;
    all_results.reserve(kNumSizes * kNumTCs);

    print_header_rs();

    for (int si = 0; si < kNumSizes; ++si) {
        const size_t total  = kTotalSizes[si];
        const size_t shard  = compute_h100_shard(total);
        const size_t offset = total * 2UL / 13UL;  // approximate start offset for this tier

        for (int ti = 0; ti < kNumTCs; ++ti) {
            const int nt = kNumTensorCounts[ti];

            // Rough memory sanity: skip if we'd need > ~4 GB
            size_t needed_bytes = (size_t)nt * total * 2 + shard * (2 + 4);
            if (needed_bytes > 4UL * 1024 * 1024 * 1024UL) {
                printf("  skipping n=%.0fM  nT=%d  (would need %.1f GB)\n",
                       total * 2.f / (1 << 20), nt,
                       needed_bytes / 1e9);
                continue;
            }

            char tag[64] = {};
            ConfigResult cr = bench_config(
                total, offset, shard, nt, sm_version, stream, tag);

            print_rs(cr.serial_add);
            print_rs(cr.atomic_fp32);
            print_rs(cr.hetero_fused);
            printf("\n");

            all_results.push_back(cr);
        }
    }

    // ── Roofline analysis ─────────────────────────────────────────────
    printf("─── Roofline reference ─────────────────────────────────────────\n");
    printf("  Peak HBM BW : %.0f GB/s\n", peak_hbm);
    printf("  For hetero_fused, effective BW = (nT+1)×shard_bytes / latency\n");
    printf("  BW utilisation = eff_GB/s / %.0f\n\n", peak_hbm);

    if (!all_results.empty()) {
        printf("  BW utilisation summary (hetero_fused, large shards ≥ 16M elems):\n");
        for (const auto& cr : all_results) {
            const auto& r = cr.hetero_fused;
            if (r.shard_count < 4UL*1024*1024) continue;
            printf("    %-32s  eff=%6.1f GB/s  util=%.0f%%\n",
                   r.config_tag,
                   r.bw_eff_gbs,
                   100.f * r.bw_eff_gbs / peak_hbm);
        }
        printf("\n");
    }

    // ── Summary speedup table ─────────────────────────────────────────
    printf("─── Speedup summary (hetero_fused vs naive_serial_add) ─────────\n");
    printf("  %-32s  %7s  %7s\n", "config", "avg_µs", "speedup");
    for (const auto& cr : all_results) {
        const auto& r = cr.hetero_fused;
        printf("  %-32s  %7.1f  %6.2fx\n",
               r.config_tag, r.avg_us, r.speedup);
    }
    printf("\n");

    // ── Regression gate ───────────────────────────────────────────────
    if (regression_mode) {
        bool reg_ok = regression_check(all_results);
        printf("Regression check: %s\n", reg_ok ? "PASS ✓" : "FAIL ✗");
        cudaStreamDestroy(stream);
        return reg_ok ? 0 : 2;
    }

    printf("Benchmark complete.\n");
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
