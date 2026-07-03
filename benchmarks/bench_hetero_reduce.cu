// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * bench_hetero_reduce.cu
 *
 * Benchmark harness for hetero_reduce.cu kernels.
 * Measures throughput (GB/s effective BF16) and latency across:
 *   - Different tensor sizes (1M → 512M elements)
 *   - Different num_tensors (1, 2, 4, 8, 16, 32)
 *   - SM version dispatch paths (SM86, SM90, SM120)
 *   - Fast path (num_tensors ≤ kMaxInlinePointers, constant memory)
 *   - Warp-coop path (small tensors, many inputs)
 *
 * Compile:
 *   nvcc -O3 -arch=sm_90 -std=c++20 \
 *     -I../csrc/hetero_reduce -I../csrc/includes \
 *     bench_hetero_reduce.cu ../csrc/hetero_reduce/hetero_reduce.cu \
 *     -o bench_hetero_reduce
 *
 * Run:
 *   ./bench_hetero_reduce
 *   ./bench_hetero_reduce --sm 86    # force SM86 dispatch path
 *   ./bench_hetero_reduce --sm 90    # force SM90 dispatch path (default)
 *
 * Output format:
 *   tensor_elems  num_tensors  path            latency_us  throughput_GBs
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

// Include the kernel header
#include "../csrc/hetero_reduce/hetero_reduce.h"

// ─────────────────────────────────────────────────────────────────────────────
// Timing utilities
// ─────────────────────────────────────────────────────────────────────────────

struct GpuTimer {
    cudaEvent_t start_, stop_;
    GpuTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start(cudaStream_t s = 0) { cudaEventRecord(start_, s); }
    float stop_ms(cudaStream_t s = 0) {
        cudaEventRecord(stop_, s);
        cudaEventSynchronize(stop_);
        float ms;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }
};

static constexpr int kWarmupIter = 10;
static constexpr int kBenchIter  = 50;

// ─────────────────────────────────────────────────────────────────────────────
// Result struct
// ─────────────────────────────────────────────────────────────────────────────

struct BenchResult {
    size_t n_elems;
    int    num_tensors;
    int    sm_version;
    const char* path;
    float  min_latency_us;
    float  avg_latency_us;
    float  throughput_GBs;   // effective read bandwidth (num_tensors × n_elems × 2 bytes)
};

void print_result(const BenchResult& r)
{
    printf("%-12zu  %-4d  %-20s  SM%-3d  %8.2f us  %8.2f us  %8.2f GB/s\n",
           r.n_elems, r.num_tensors, r.path, r.sm_version,
           r.min_latency_us, r.avg_latency_us, r.throughput_GBs);
}

void print_header()
{
    printf("%-12s  %-4s  %-20s  %-5s  %8s  %8s  %10s\n",
           "tensor_elems", "nT", "path", "SM",
           "min_us", "avg_us", "GB/s");
    printf("%s\n", std::string(80, '-').c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// Correctness check: reduce 2 tensors of all-1 BF16 → expect all-2 BF16
// ─────────────────────────────────────────────────────────────────────────────

bool correctness_check(int sm_version, cudaStream_t stream)
{
    constexpr size_t N = 1024;
    const int num_tensors = 2;

    __nv_bfloat16* d_out = nullptr;
    __nv_bfloat16* d_in[2] = {nullptr, nullptr};
    cudaMalloc(&d_out, N * sizeof(__nv_bfloat16));
    for (int i = 0; i < 2; ++i) {
        cudaMalloc(&d_in[i], N * sizeof(__nv_bfloat16));
        // Fill with 1.0
        std::vector<__nv_bfloat16> h(N, __float2bfloat16(1.f));
        cudaMemcpy(d_in[i], h.data(), N * sizeof(__nv_bfloat16),
                   cudaMemcpyHostToDevice);
    }

    const __nv_bfloat16* ptrs[2] = {d_in[0], d_in[1]};
    launch_fused_bf16_reduce(d_out, ptrs, num_tensors, N, sm_version, stream);
    cudaStreamSynchronize(stream);

    std::vector<__nv_bfloat16> h_out(N);
    cudaMemcpy(h_out.data(), d_out, N * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);

    bool ok = true;
    for (size_t i = 0; i < N; ++i) {
        float v = __bfloat162float(h_out[i]);
        if (std::abs(v - 2.f) > 0.01f) { ok = false; break; }
    }

    cudaFree(d_out);
    for (int i = 0; i < 2; ++i) cudaFree(d_in[i]);
    return ok;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main benchmark
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    int sm_version = 90;  // default: H100 path
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--sm") == 0 && i + 1 < argc)
            sm_version = atoi(argv[++i]);
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Print device info
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("Device: %s  SM: %d.%d\n", prop.name,
           prop.major, prop.minor);
    printf("Forcing SM dispatch path: SM%d\n\n", sm_version);

    // Correctness check
    bool ok = correctness_check(sm_version, stream);
    printf("Correctness check: %s\n\n", ok ? "PASS" : "FAIL");
    if (!ok) return 1;

    print_header();

    // Tensor sizes to sweep (in elements, must be multiple of 8)
    const size_t sizes[] = {
        1UL   << 17,  // 128K elements = 256 KB BF16
        1UL   << 20,  //   1M elements =   2 MB
        1UL   << 22,  //   4M elements =   8 MB
        1UL   << 24,  //  16M elements =  32 MB
        1UL   << 26,  //  64M elements = 128 MB
    };
    const int num_tensor_counts[] = {1, 2, 4, 8, 16, 32};

    constexpr int kMaxT = 32;

    for (size_t n : sizes) {
        for (int nt : num_tensor_counts) {

            // Allocate output
            __nv_bfloat16* d_out = nullptr;
            cudaMalloc(&d_out, n * sizeof(__nv_bfloat16));
            cudaMemset(d_out, 0, n * sizeof(__nv_bfloat16));

            // Allocate inputs
            std::vector<__nv_bfloat16*> d_in(nt, nullptr);
            for (int t = 0; t < nt; ++t) {
                cudaMalloc(&d_in[t], n * sizeof(__nv_bfloat16));
                cudaMemset(d_in[t], 0x3f, n * sizeof(__nv_bfloat16));  // random data
            }
            std::vector<const __nv_bfloat16*> ptrs(nt);
            for (int t = 0; t < nt; ++t) ptrs[t] = d_in[t];

            GpuTimer timer;
            float total_ms = 0.f, min_ms = 1e9f;

            // Warmup
            for (int w = 0; w < kWarmupIter; ++w)
                launch_fused_bf16_reduce(d_out, ptrs.data(), nt, n, sm_version, stream);
            cudaStreamSynchronize(stream);

            // Benchmark
            for (int it = 0; it < kBenchIter; ++it) {
                timer.start(stream);
                launch_fused_bf16_reduce(d_out, ptrs.data(), nt, n, sm_version, stream);
                float ms = timer.stop_ms(stream);
                total_ms += ms;
                min_ms = std::min(min_ms, ms);
            }

            // Determine path name
            const char* path;
            if (n <= 128 * 1024 && nt <= 32)  path = "warp_coop";
            else if (nt <= kMaxT)              path = "const_mem_fast";
            else                               path = "device_mem";

            // Effective read bandwidth: nt inputs + 1 output, each n × 2 bytes
            float bytes = (float)((size_t)(nt + 1) * n * sizeof(__nv_bfloat16));
            float avg_ms = total_ms / kBenchIter;

            BenchResult r;
            r.n_elems         = n;
            r.num_tensors     = nt;
            r.sm_version      = sm_version;
            r.path            = path;
            r.min_latency_us  = min_ms * 1000.f;
            r.avg_latency_us  = avg_ms * 1000.f;
            r.throughput_GBs  = bytes / (avg_ms * 1e-3f) / 1e9f;
            print_result(r);

            cudaFree(d_out);
            for (int t = 0; t < nt; ++t) cudaFree(d_in[t]);
        }
        printf("\n");
    }

    // ─── Reduce-scatter benchmark ───────────────────────────────────────
    printf("\n─── Reduce-scatter (shard = 1/5 of tensor) ─────────────────────\n");
    print_header();

    // Simulate 5 tiers; this device gets shard index 2 (H100 tier)
    HeteroTierDesc tiers[5] = {
        {0, 86, 0}, {1, 86, 0}, {2, 90, 0}, {3, 120, 0}, {4, 120, 0}
    };
    size_t offsets[5], counts[5];

    const size_t rs_size = 1UL << 24;  // 16M elements
    compute_hetero_shard_ranges(tiers, 5, rs_size, offsets, counts);

    printf("Shard ranges for %zu total elements:\n", rs_size);
    for (int i = 0; i < 5; ++i)
        printf("  tier%d SM%d: offset=%-8zu count=%-8zu (%.1f MB)\n",
               i, tiers[i].sm_version, offsets[i], counts[i],
               counts[i] * 2.f / (1 << 20));

    for (int nt : {2, 4, 8}) {
        __nv_bfloat16* d_out = nullptr;
        cudaMalloc(&d_out, counts[2] * sizeof(__nv_bfloat16));

        std::vector<__nv_bfloat16*> d_in(nt);
        for (int t = 0; t < nt; ++t) {
            cudaMalloc(&d_in[t], rs_size * sizeof(__nv_bfloat16));
            cudaMemset(d_in[t], 0, rs_size * sizeof(__nv_bfloat16));
        }
        std::vector<const __nv_bfloat16*> ptrs(nt);
        for (int t = 0; t < nt; ++t) ptrs[t] = d_in[t];

        GpuTimer timer;
        float total_ms = 0.f, min_ms = 1e9f;
        for (int w = 0; w < kWarmupIter; ++w)
            launch_hetero_reduce_scatter(d_out, ptrs.data(), nt,
                offsets[2], counts[2], sm_version, stream);
        cudaStreamSynchronize(stream);
        for (int it = 0; it < kBenchIter; ++it) {
            timer.start(stream);
            launch_hetero_reduce_scatter(d_out, ptrs.data(), nt,
                offsets[2], counts[2], sm_version, stream);
            float ms = timer.stop_ms(stream);
            total_ms += ms; min_ms = std::min(min_ms, ms);
        }
        float bytes = (float)((size_t)nt * rs_size + counts[2]) * 2;
        float avg_ms = total_ms / kBenchIter;
        BenchResult r;
        r.n_elems = rs_size; r.num_tensors = nt; r.sm_version = sm_version;
        r.path = "reduce_scatter";
        r.min_latency_us = min_ms * 1000.f;
        r.avg_latency_us = avg_ms * 1000.f;
        r.throughput_GBs = bytes / (avg_ms * 1e-3f) / 1e9f;
        print_result(r);

        cudaFree(d_out);
        for (int t = 0; t < nt; ++t) cudaFree(d_in[t]);
    }

    // ─── Per-tier bucket size report ────────────────────────────────────
    printf("\n─── Per-tier adaptive bucket sizes ─────────────────────────────\n");
    for (int sm : {86, 90, 120}) {
        size_t elems = hetero_bucket_size_elems(sm);
        printf("  SM%d: %zu elements = %.1f MB BF16\n",
               sm, elems, elems * 2.f / (1 << 20));
    }

    cudaStreamDestroy(stream);
    printf("\nBenchmark complete.\n");
    return 0;
}
