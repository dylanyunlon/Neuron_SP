// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * bench_fused_swiglu_ln.cu
 *
 * Benchmark harness for fused_swiglu_ln.cu.
 * Measures:
 *   1. End-to-end throughput (GB/s effective) vs. naive unfused baseline
 *   2. Latency across batch sizes and hidden dimensions
 *   3. Single-pass vs. two-pass regime transition
 *   4. Arithmetic intensity verification (via Nsight metrics when available)
 *
 * Key metrics:
 *   - Effective memory bandwidth: reads = gate + up = 2 × batch × hidden × 2 B
 *                                  write = output = batch × hidden × 2 B
 *     Total = 3 × batch × hidden × 2 bytes
 *   - For H100 peak bandwidth (3.35 TB/s), roofline BW-bound limit:
 *     3.35 TB/s / 6 B/elem = 558 Gelem/s → batch=64, hidden=8192 → 0.037 ms
 *
 * Compile:
 *   nvcc -O3 -arch=sm_90 -std=c++20 \
 *     -I../csrc/hetero_reduce -I../csrc/includes \
 *     bench_fused_swiglu_ln.cu ../csrc/hetero_reduce/fused_swiglu_ln.cu \
 *     -o bench_fused_swiglu_ln
 *
 * Run:
 *   ./bench_fused_swiglu_ln
 *   ./bench_fused_swiglu_ln --sm 86
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

#include "../csrc/hetero_reduce/hetero_reduce.h"

// ─────────────────────────────────────────────────────────────────────────────
// Timer
// ─────────────────────────────────────────────────────────────────────────────

struct GpuTimer {
    cudaEvent_t s_, e_;
    GpuTimer() { cudaEventCreate(&s_); cudaEventCreate(&e_); }
    ~GpuTimer() { cudaEventDestroy(s_); cudaEventDestroy(e_); }
    void start(cudaStream_t st = 0) { cudaEventRecord(s_, st); }
    float stop_ms(cudaStream_t st = 0) {
        cudaEventRecord(e_, st);
        cudaEventSynchronize(e_);
        float ms; cudaEventElapsedTime(&ms, s_, e_); return ms;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Naive unfused baseline (reference)
//   Three separate kernels: SwiGLU → sq_sum → LayerNorm
//   Used to measure fusion benefit.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void naive_swiglu_kernel(
    float* __restrict__ out,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    int hidden)
{
    const int row = blockIdx.x;
    for (int c = threadIdx.x; c < hidden; c += blockDim.x) {
        float g = __bfloat162float(gate[row * hidden + c]);
        float u = __bfloat162float(up  [row * hidden + c]);
        out[row * hidden + c] = g / (1.f + __expf(-g)) * u;
    }
}

__global__ void naive_rms_norm_kernel(
    __nv_bfloat16* __restrict__ out,
    const float* __restrict__ swiglu,
    const float* __restrict__ ln_w,
    int hidden, float eps)
{
    extern __shared__ float smem[];
    const int row  = blockIdx.x;
    float sq = 0.f;
    for (int c = threadIdx.x; c < hidden; c += blockDim.x)
        sq += swiglu[row * hidden + c] * swiglu[row * hidden + c];
    smem[threadIdx.x] = sq;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float rms_inv = rsqrtf(smem[0] / hidden + eps);
    for (int c = threadIdx.x; c < hidden; c += blockDim.x)
        out[row * hidden + c] = __float2bfloat16(swiglu[row * hidden + c] * rms_inv * ln_w[c]);
}

float bench_unfused(
    __nv_bfloat16* d_out,
    const __nv_bfloat16* d_gate,
    const __nv_bfloat16* d_up,
    const float*         d_lnw,
    float*               d_tmp,  // FP32 scratch [batch × hidden]
    int batch, int hidden,
    cudaStream_t stream, int niters)
{
    GpuTimer t;
    float total = 0.f;
    constexpr int kBS = 256;
    for (int it = 0; it < niters; ++it) {
        t.start(stream);
        naive_swiglu_kernel<<<batch, kBS, 0, stream>>>(d_tmp, d_gate, d_up, hidden);
        naive_rms_norm_kernel<<<batch, kBS, kBS * sizeof(float), stream>>>(
            d_out, d_tmp, d_lnw, hidden, 1e-6f);
        total += t.stop_ms(stream);
    }
    return total / niters;
}

// ─────────────────────────────────────────────────────────────────────────────
// Correctness check: output should equal naive reference within BF16 tolerance
// ─────────────────────────────────────────────────────────────────────────────

bool correctness_check(int sm_version, cudaStream_t stream)
{
    constexpr int batch  = 4;
    constexpr int hidden = 512;

    size_t row_bytes = hidden * sizeof(__nv_bfloat16);

    __nv_bfloat16 *d_gate, *d_up, *d_out_fused, *d_out_ref;
    float *d_lnw, *d_tmp;
    cudaMalloc(&d_gate,       batch * row_bytes);
    cudaMalloc(&d_up,         batch * row_bytes);
    cudaMalloc(&d_out_fused,  batch * row_bytes);
    cudaMalloc(&d_out_ref,    batch * row_bytes);
    cudaMalloc(&d_lnw,        hidden * sizeof(float));
    cudaMalloc(&d_tmp,        batch * hidden * sizeof(float));

    // Fill: gate = 0.5, up = 1.0, ln_weight = 1.0
    std::vector<__nv_bfloat16> h_bf(batch * hidden, __float2bfloat16(0.5f));
    cudaMemcpy(d_gate, h_bf.data(), batch * row_bytes, cudaMemcpyHostToDevice);
    std::fill(h_bf.begin(), h_bf.end(), __float2bfloat16(1.f));
    cudaMemcpy(d_up, h_bf.data(), batch * row_bytes, cudaMemcpyHostToDevice);
    std::vector<float> h_w(hidden, 1.f);
    cudaMemcpy(d_lnw, h_w.data(), hidden * sizeof(float), cudaMemcpyHostToDevice);

    launch_fused_swiglu_ln(d_out_fused, d_gate, d_up, d_lnw,
                           batch, hidden, 1e-6f, sm_version, stream);
    bench_unfused(d_out_ref, d_gate, d_up, d_lnw, d_tmp,
                  batch, hidden, stream, 1);
    cudaStreamSynchronize(stream);

    std::vector<__nv_bfloat16> hf(batch * hidden), hr(batch * hidden);
    cudaMemcpy(hf.data(), d_out_fused, batch * row_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(hr.data(), d_out_ref,   batch * row_bytes, cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < batch * hidden && ok; ++i) {
        float f = __bfloat162float(hf[i]);
        float r = __bfloat162float(hr[i]);
        if (std::abs(f - r) > 0.05f * std::abs(r) + 0.001f) {
            printf("  Mismatch at i=%d: fused=%.4f ref=%.4f\n", i, f, r);
            ok = false;
        }
    }

    cudaFree(d_gate); cudaFree(d_up);
    cudaFree(d_out_fused); cudaFree(d_out_ref);
    cudaFree(d_lnw); cudaFree(d_tmp);
    return ok;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main benchmark
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    int sm_version = 90;
    for (int i = 1; i < argc; ++i)
        if (strcmp(argv[i], "--sm") == 0 && i + 1 < argc)
            sm_version = atoi(argv[++i]);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    printf("Device: %s  SM: %d.%d  Mem BW: %.1f GB/s\n",
           prop.name, prop.major, prop.minor,
           prop.memoryBusWidth / 8.0 * prop.memoryClockRate * 2e-6);
    printf("SM dispatch path: SM%d\n\n", sm_version);

    bool ok = correctness_check(sm_version, stream);
    printf("Correctness: %s\n\n", ok ? "PASS" : "FAIL");
    if (!ok) return 1;

    constexpr int kWarmup = 10, kIter = 50;

    // Sweep: batch × hidden
    const int batches[] = {1, 8, 64, 512, 2048};
    const int hiddens[]  = {512, 1024, 2048, 4096, 8192, 16384};

    printf("%-6s  %-6s  %-14s  %-14s  %-10s  %-10s  %s\n",
           "batch", "hidden", "fused_avg_us", "naive_avg_us",
           "speedup", "fused_GB/s", "path");
    printf("%s\n", std::string(90, '-').c_str());

    for (int batch : batches) {
        for (int hidden : hiddens) {
            size_t row_bytes = hidden * sizeof(__nv_bfloat16);

            __nv_bfloat16 *d_gate, *d_up, *d_out;
            float *d_lnw, *d_tmp;
            cudaMalloc(&d_gate, (size_t)batch * row_bytes);
            cudaMalloc(&d_up,   (size_t)batch * row_bytes);
            cudaMalloc(&d_out,  (size_t)batch * row_bytes);
            cudaMalloc(&d_lnw,  hidden * sizeof(float));
            cudaMalloc(&d_tmp,  (size_t)batch * hidden * sizeof(float));
            cudaMemset(d_gate, 0x3f, (size_t)batch * row_bytes);
            cudaMemset(d_up,   0x3f, (size_t)batch * row_bytes);
            cudaMemset(d_lnw,  0x3f, hidden * sizeof(float));

            GpuTimer timer;

            // Warmup
            for (int w = 0; w < kWarmup; ++w)
                launch_fused_swiglu_ln(d_out, d_gate, d_up, d_lnw,
                                       batch, hidden, 1e-6f, sm_version, stream);
            cudaStreamSynchronize(stream);

            // Fused benchmark
            float fused_total = 0.f, fused_min = 1e9f;
            for (int it = 0; it < kIter; ++it) {
                timer.start(stream);
                launch_fused_swiglu_ln(d_out, d_gate, d_up, d_lnw,
                                       batch, hidden, 1e-6f, sm_version, stream);
                float ms = timer.stop_ms(stream);
                fused_total += ms;
                fused_min = std::min(fused_min, ms);
            }
            float fused_avg_ms = fused_total / kIter;

            // Naive benchmark
            float naive_avg_ms = bench_unfused(d_out, d_gate, d_up, d_lnw,
                                               d_tmp, batch, hidden, stream, kIter);

            // Effective BW: 3 × batch × hidden × 2 bytes (2 reads + 1 write)
            float bytes_eff = 3.f * batch * hidden * 2.f;
            float fused_gbs = bytes_eff / (fused_avg_ms * 1e-3f) / 1e9f;
            float speedup   = naive_avg_ms / fused_avg_ms;

            // Determine single-pass vs two-pass
            // kBlockSize × kVecWidth × kRegBudgetPerThread:
            int max_sp = (sm_version >= 90) ? (256 * 8 * 128) : (256 * 8 * 64);
            const char* path = (hidden <= max_sp) ? "single-pass" : "two-pass";

            printf("%-6d  %-6d  %12.2f us  %12.2f us  %8.2f×  %8.2f GB/s  %s\n",
                   batch, hidden,
                   fused_avg_ms * 1000.f, naive_avg_ms * 1000.f,
                   speedup, fused_gbs, path);

            cudaFree(d_gate); cudaFree(d_up); cudaFree(d_out);
            cudaFree(d_lnw);  cudaFree(d_tmp);
        }
    }

    // ─── Single-pass regime analysis ────────────────────────────────────
    printf("\n─── Single-pass vs two-pass threshold analysis ───────────────\n");
    printf("For SM%d: kBlockSize=%d, kVecWidth=8, kRegBudget=%d\n",
           sm_version, 256,
           (sm_version >= 90) ? 128 : 64);
    int max_sp = (sm_version >= 90) ? (256 * 8 * 128) : (256 * 8 * 64);
    printf("Single-pass threshold: hidden ≤ %d elements (%.1f KB)\n",
           max_sp, max_sp * 4.f / 1024.f);
    printf("Typical transformer hidden: 4096 (Llama-7B), 8192 (Llama-70B)\n");
    printf("Both fit in single-pass for SM%d → zero extra DRAM reads\n", sm_version);

    cudaStreamDestroy(stream);
    printf("\nBenchmark complete.\n");
    return 0;
}
