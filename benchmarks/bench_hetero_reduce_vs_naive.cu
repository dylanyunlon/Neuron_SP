// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * bench_hetero_reduce_vs_naive.cu
 *
 * Comparative benchmark: hetero_reduce_scatter_kernel (fused vectorised,
 * policy-driven) vs a naive scalar baseline.
 *
 * Purpose:
 *   Quantify the throughput gain from the algorithmic innovations in
 *   hetero_reduce.cu (vectorised BF16x8 loads, __constant__ pointer
 *   array, warp-cooperative small-tensor path, per-SM-tier policy).
 *
 * Methodology:
 *   - Naive baseline: one thread per output element, scalar BF16→FP32
 *     accumulation, no vectorisation, pointer array in device memory.
 *   - Fused kernel: launch_hetero_reduce_scatter() from the production
 *     API (auto-selects warp-coop vs standard path).
 *   - Each configuration runs 5 warmup + 20 timed iterations; reports
 *     median latency and effective memory bandwidth.
 *   - Correctness is verified per-config (relative tolerance 1%).
 *
 * Compile:
 *   nvcc -O3 -std=c++17 -arch=sm_86 \
 *        -I../csrc/includes -I../csrc/hetero_reduce \
 *        bench_hetero_reduce_vs_naive.cu \
 *        ../csrc/hetero_reduce/hetero_reduce.cu \
 *        -o bench_hetero_reduce_vs_naive
 *
 * Run:
 *   ./bench_hetero_reduce_vs_naive [sm_version]
 *   # sm_version defaults to auto-detect from current device
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#include "hetero_reduce.h"

// ──────────────────────────────────────────────────────────────────────
// Error checking
// ──────────────────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                   \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(_e));                                 \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// ──────────────────────────────────────────────────────────────────────
// Device query
// ──────────────────────────────────────────────────────────────────────

static int detect_sm_version() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    return prop.major * 10 + prop.minor;
}

static void print_device_info() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("  Device:     %s\n", prop.name);
    printf("  SM count:   %d\n", prop.multiProcessorCount);
    printf("  L2 cache:   %.1f MB\n", prop.l2CacheSize / (1024.0 * 1024.0));
    printf("  Mem BW:     %.0f GB/s (theoretical peak)\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
}

// ──────────────────────────────────────────────────────────────────────
// Data initialisation
// ──────────────────────────────────────────────────────────────────────

static void fill_random_bf16(__nv_bfloat16* d_ptr, size_t n, unsigned seed) {
    std::vector<__nv_bfloat16> host(n);
    srand(seed);
    for (size_t i = 0; i < n; ++i)
        host[i] = __float2bfloat16(((float)rand() / RAND_MAX - 0.5f) * 2.0f);
    CUDA_CHECK(cudaMemcpy(d_ptr, host.data(), n * sizeof(__nv_bfloat16),
                           cudaMemcpyHostToDevice));
}

// ──────────────────────────────────────────────────────────────────────
// Naive baseline kernel
//
// One thread per output element.  Scalar loads, no vectorisation,
// pointer array in device memory.  This is the simplest correct
// implementation — it establishes the performance floor.
// ──────────────────────────────────────────────────────────────────────

__global__ void naive_reduce_scatter_kernel(
    __nv_bfloat16* __restrict__              output,
    const __nv_bfloat16* const* __restrict__ d_inputs,
    int    num_tensors,
    size_t shard_offset,
    size_t shard_count)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= shard_count) return;

    float acc = 0.f;
    const size_t gidx = shard_offset + idx;
    for (int t = 0; t < num_tensors; ++t)
        acc += __bfloat162float(d_inputs[t][gidx]);
    output[idx] = __float2bfloat16(acc);
}

static void launch_naive(
    __nv_bfloat16* output,
    const __nv_bfloat16* const* h_inputs,
    int num_tensors,
    size_t shard_offset,
    size_t shard_count,
    cudaStream_t stream)
{
    const __nv_bfloat16** d_ptrs = nullptr;
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_ptrs),
               num_tensors * sizeof(const __nv_bfloat16*), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_ptrs, h_inputs,
               num_tensors * sizeof(const __nv_bfloat16*),
               cudaMemcpyHostToDevice, stream));

    constexpr int kBlock = 256;
    const int grid = (int)std::min(
        (shard_count + kBlock - 1) / kBlock, (size_t)65535);
    naive_reduce_scatter_kernel<<<grid, kBlock, 0, stream>>>(
        output, d_ptrs, num_tensors, shard_offset, shard_count);

    CUDA_CHECK(cudaFreeAsync(d_ptrs, stream));
}

// ──────────────────────────────────────────────────────────────────────
// Correctness verification
// ──────────────────────────────────────────────────────────────────────

static bool verify(
    __nv_bfloat16* d_ref, __nv_bfloat16* d_test, size_t n,
    float rtol = 1e-2f)
{
    std::vector<__nv_bfloat16> h_ref(n), h_test(n);
    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref, n * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_test.data(), d_test, n * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToHost));
    int bad = 0;
    for (size_t i = 0; i < n && bad < 5; ++i) {
        float a = __bfloat162float(h_ref[i]);
        float b = __bfloat162float(h_test[i]);
        float diff = fabsf(a - b);
        float denom = fmaxf(fabsf(a), 1e-6f);
        if (diff / denom > rtol) {
            if (bad == 0)
                fprintf(stderr, "    mismatch [%zu]: ref=%.6f got=%.6f\n",
                        i, a, b);
            ++bad;
        }
    }
    return bad == 0;
}

// ──────────────────────────────────────────────────────────────────────
// Timing: returns median of N runs in milliseconds
// ──────────────────────────────────────────────────────────────────────

template <typename Fn>
static float time_median(Fn fn, cudaStream_t stream,
                         int warmup = 5, int iters = 20)
{
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> times(iters);
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(t0, stream));
        fn();
        CUDA_CHECK(cudaEventRecord(t1, stream));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], t0, t1));
    }
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));

    std::sort(times.begin(), times.end());
    return times[iters / 2];
}

// ──────────────────────────────────────────────────────────────────────
// Main benchmark loop
// ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    int sm = (argc >= 2) ? atoi(argv[1]) : detect_sm_version();

    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  hetero_reduce_scatter: fused kernel vs naive baseline\n");
    printf("  SM version (dispatch): %d\n", sm);
    print_device_info();
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ── Test matrix ──────────────────────────────────────────────────
    struct Config {
        int    num_tensors;
        size_t shard_count;
        const char* tag;
    };

    Config cfgs[] = {
        // Small (warp-coop path)
        { 4,   64UL * 1024,          " 4T x   64K" },
        { 8,   64UL * 1024,          " 8T x   64K" },
        { 16,  64UL * 1024,          "16T x   64K" },
        // Medium (standard vectorised path)
        { 4,  512UL * 1024,          " 4T x  512K" },
        { 8,  512UL * 1024,          " 8T x  512K" },
        { 16, 512UL * 1024,          "16T x  512K" },
        // Large (bandwidth-bound)
        { 4,  4UL * 1024 * 1024,     " 4T x    4M" },
        { 8,  4UL * 1024 * 1024,     " 8T x    4M" },
        { 16, 4UL * 1024 * 1024,     "16T x    4M" },
        // XL stress
        { 4,  16UL * 1024 * 1024,    " 4T x   16M" },
        { 8,  16UL * 1024 * 1024,    " 8T x   16M" },
    };
    const int N = sizeof(cfgs) / sizeof(cfgs[0]);

    printf("%-14s  %9s  %9s  %8s  %10s  %10s  %5s\n",
           "Config", "Naive(ms)", "Fused(ms)", "Speedup",
           "Naive GB/s", "Fused GB/s", "Check");
    printf("─────────────────────────────────────────────────────"
           "─────────────────────────────────\n");

    for (int ci = 0; ci < N; ++ci) {
        const auto& c = cfgs[ci];

        // Allocate inputs
        std::vector<__nv_bfloat16*> d_in(c.num_tensors);
        std::vector<const __nv_bfloat16*> h_ptrs(c.num_tensors);
        for (int t = 0; t < c.num_tensors; ++t) {
            CUDA_CHECK(cudaMalloc(&d_in[t],
                       c.shard_count * sizeof(__nv_bfloat16)));
            fill_random_bf16(d_in[t], c.shard_count, 42 + t);
            h_ptrs[t] = d_in[t];
        }

        __nv_bfloat16 *d_naive = nullptr, *d_fused = nullptr;
        CUDA_CHECK(cudaMalloc(&d_naive, c.shard_count * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_fused, c.shard_count * sizeof(__nv_bfloat16)));

        // Total bytes moved: num_tensors reads + 1 write, all in BF16
        const double total_bytes =
            (double)(c.num_tensors + 1) * c.shard_count * sizeof(__nv_bfloat16);

        // Time naive
        float ms_naive = time_median([&]() {
            launch_naive(d_naive, h_ptrs.data(), c.num_tensors,
                         0, c.shard_count, stream);
        }, stream);

        // Time fused
        float ms_fused = time_median([&]() {
            launch_hetero_reduce_scatter(
                d_fused, h_ptrs.data(), c.num_tensors,
                0, c.shard_count, sm, stream);
        }, stream);

        CUDA_CHECK(cudaStreamSynchronize(stream));
        bool ok = verify(d_naive, d_fused, c.shard_count);

        float speedup   = ms_naive / ms_fused;
        float bw_naive  = (float)(total_bytes / (ms_naive * 1e6));
        float bw_fused  = (float)(total_bytes / (ms_fused * 1e6));

        printf("%-14s  %9.3f  %9.3f  %7.2fx  %9.1f  %9.1f  %5s\n",
               c.tag, ms_naive, ms_fused, speedup,
               bw_naive, bw_fused,
               ok ? "OK" : "FAIL");

        for (int t = 0; t < c.num_tensors; ++t)
            CUDA_CHECK(cudaFree(d_in[t]));
        CUDA_CHECK(cudaFree(d_naive));
        CUDA_CHECK(cudaFree(d_fused));
    }

    printf("\n");
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
