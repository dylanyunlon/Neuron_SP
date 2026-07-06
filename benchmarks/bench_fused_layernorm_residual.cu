// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * bench_fused_layernorm_residual.cu
 *
 * Benchmark harness for fused_layernorm_residual.cu.
 * Measures:
 *   1. End-to-end throughput (effective GB/s) vs. naive two-kernel baseline
 *   2. Latency sweep across batch sizes and hidden dimensions
 *   3. Single-pass vs. two-pass regime identification
 *   4. Correctness check against naive reference (BF16 tolerance)
 *
 * Effective bandwidth accounting:
 *   Reads  : input [batch×hidden×2 B] + residual_in [batch×hidden×2 B]
 *            + ln_weight [hidden×4 B]  ≈ 4×B×H bytes for large B
 *   Writes : residual_out [batch×hidden×2 B] + output [batch×hidden×2 B]
 *            = 4×B×H bytes
 *   Total  ≈ 8 × batch × hidden bytes (single-pass avoids re-reading residual)
 *   Naive unfused total = 10 × batch × hidden bytes (extra residual read)
 *
 * Compile (standalone):
 *   nvcc -O3 -arch=sm_90 -std=c++20 \
 *     -I../csrc/hetero_reduce -I../csrc/includes \
 *     bench_fused_layernorm_residual.cu \
 *     ../csrc/hetero_reduce/fused_layernorm_residual.cu \
 *     -o bench_fused_layernorm_residual
 *
 * Run:
 *   ./bench_fused_layernorm_residual
 *   ./bench_fused_layernorm_residual --sm 86
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
#include <string>

#include "../csrc/hetero_reduce/hetero_reduce.h"

// ─────────────────────────────────────────────────────────────────────────────
// Timer
// ─────────────────────────────────────────────────────────────────────────────

struct GpuTimer {
    cudaEvent_t s_, e_;
    GpuTimer()  { cudaEventCreate(&s_); cudaEventCreate(&e_); }
    ~GpuTimer() { cudaEventDestroy(s_); cudaEventDestroy(e_); }
    void start(cudaStream_t st = 0) { cudaEventRecord(s_, st); }
    float stop_ms(cudaStream_t st = 0) {
        cudaEventRecord(e_, st);
        cudaEventSynchronize(e_);
        float ms; cudaEventElapsedTime(&ms, s_, e_); return ms;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Naive unfused baseline
//   Kernel 1: residual_out[i] = input[i] + residual_in[i]
//   Kernel 2: rms_sq = mean(residual_out²)
//   Kernel 3: output[i] = residual_out[i] * rsqrt(rms_sq + eps) * w[i]
//   Two-pass separation is what the fused kernel eliminates.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void naive_residual_add_kernel(
    __nv_bfloat16*       __restrict__ res_out,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ res_in,
    int hidden)
{
    const int row = blockIdx.x;
    for (int c = threadIdx.x; c < hidden; c += blockDim.x)
        res_out[row * hidden + c] = __float2bfloat16(
            __bfloat162float(input [row * hidden + c]) +
            __bfloat162float(res_in[row * hidden + c]));
}

// Uses dynamic shared memory for the per-block RMS reduction
__global__ void naive_rms_norm_kernel(
    __nv_bfloat16*       __restrict__ output,
    const __nv_bfloat16* __restrict__ residual,
    const float*         __restrict__ ln_weight,
    int hidden, float eps)
{
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    float sq = 0.f;
    for (int c = threadIdx.x; c < hidden; c += blockDim.x) {
        float v = __bfloat162float(residual[row * hidden + c]);
        sq += v * v;
    }
    smem[threadIdx.x] = sq;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float rms_inv = rsqrtf(smem[0] / (float)hidden + eps);
    for (int c = threadIdx.x; c < hidden; c += blockDim.x) {
        float v = __bfloat162float(residual[row * hidden + c]);
        output[row * hidden + c] = __float2bfloat16(v * rms_inv * ln_weight[c]);
    }
}

// Returns average latency in ms over niters iterations
float bench_unfused(
    __nv_bfloat16*       d_output,
    __nv_bfloat16*       d_residual,    // in/out (updated in-place)
    const __nv_bfloat16* d_input,
    const float*         d_lnw,
    int batch, int hidden,
    cudaStream_t stream, int niters)
{
    constexpr int kBS = 256;
    const size_t smem = kBS * sizeof(float);
    GpuTimer t;
    float total = 0.f;
    for (int it = 0; it < niters; ++it) {
        t.start(stream);
        naive_residual_add_kernel<<<batch, kBS, 0, stream>>>(
            d_residual, d_input, d_residual, hidden);
        naive_rms_norm_kernel<<<batch, kBS, smem, stream>>>(
            d_output, d_residual, d_lnw, hidden, 1e-6f);
        total += t.stop_ms(stream);
    }
    return total / (float)niters;
}

// ─────────────────────────────────────────────────────────────────────────────
// Correctness check
//   Fill input = 0.5, residual = 0.3, ln_weight = 1.0
//   Compare fused vs. naive reference output element-wise.
// ─────────────────────────────────────────────────────────────────────────────

bool correctness_check(int sm_version, cudaStream_t stream)
{
    constexpr int batch  = 4;
    constexpr int hidden = 512;
    const size_t row_b = hidden * sizeof(__nv_bfloat16);

    __nv_bfloat16 *d_in, *d_res_f, *d_res_n, *d_out_f, *d_out_n;
    float* d_lnw;
    cudaMalloc(&d_in,    batch * row_b);
    cudaMalloc(&d_res_f, batch * row_b);
    cudaMalloc(&d_res_n, batch * row_b);
    cudaMalloc(&d_out_f, batch * row_b);
    cudaMalloc(&d_out_n, batch * row_b);
    cudaMalloc(&d_lnw,   hidden * sizeof(float));

    // Host initialisation
    std::vector<__nv_bfloat16> h_in(batch * hidden), h_res(batch * hidden);
    std::vector<float>          h_lnw(hidden, 1.f);
    for (int i = 0; i < batch * hidden; ++i) {
        h_in [i] = __float2bfloat16(0.5f);
        h_res[i] = __float2bfloat16(0.3f);
    }
    cudaMemcpy(d_in,    h_in.data(),  batch * row_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res_f, h_res.data(), batch * row_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res_n, h_res.data(), batch * row_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lnw,   h_lnw.data(), hidden * sizeof(float), cudaMemcpyHostToDevice);

    // Fused path
    launch_fused_layernorm_residual(d_out_f, d_res_f, d_in, d_lnw,
                                     batch, hidden, 1e-6f, sm_version, stream);
    // Naive path
    bench_unfused(d_out_n, d_res_n, d_in, d_lnw, batch, hidden, stream, 1);
    cudaStreamSynchronize(stream);

    std::vector<__nv_bfloat16> hof(batch * hidden), hon(batch * hidden);
    std::vector<__nv_bfloat16> hrf(batch * hidden), hrn(batch * hidden);
    cudaMemcpy(hof.data(), d_out_f, batch * row_b, cudaMemcpyDeviceToHost);
    cudaMemcpy(hon.data(), d_out_n, batch * row_b, cudaMemcpyDeviceToHost);
    cudaMemcpy(hrf.data(), d_res_f, batch * row_b, cudaMemcpyDeviceToHost);
    cudaMemcpy(hrn.data(), d_res_n, batch * row_b, cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < batch * hidden && ok; ++i) {
        float f = __bfloat162float(hof[i]);
        float n = __bfloat162float(hon[i]);
        if (std::abs(f - n) > 0.05f * std::abs(n) + 0.001f) {
            printf("  output mismatch at i=%d: fused=%.4f naive=%.4f\n", i, f, n);
            ok = false;
        }
        // Also verify residual stream was updated identically
        float rf = __bfloat162float(hrf[i]);
        float rn = __bfloat162float(hrn[i]);
        if (ok && std::abs(rf - rn) > 0.001f) {
            printf("  residual mismatch at i=%d: fused=%.4f naive=%.4f\n", i, rf, rn);
            ok = false;
        }
    }

    cudaFree(d_in); cudaFree(d_res_f); cudaFree(d_res_n);
    cudaFree(d_out_f); cudaFree(d_out_n); cudaFree(d_lnw);
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

    constexpr int kWarmup = 10;
    constexpr int kIter   = 50;

    const int batches[] = {1, 8, 64, 512, 2048};
    const int hiddens[]  = {512, 1024, 2048, 4096, 8192, 16384};

    printf("%-6s  %-6s  %-14s  %-14s  %-10s  %-12s  %s\n",
           "batch", "hidden", "fused_avg_us", "naive_avg_us",
           "speedup", "fused_GB/s", "path");
    printf("%s\n", std::string(96, '-').c_str());

    for (int batch : batches) {
        for (int hidden : hiddens) {
            const size_t row_b = (size_t)hidden * sizeof(__nv_bfloat16);

            __nv_bfloat16 *d_in, *d_res, *d_out;
            float* d_lnw;
            cudaMalloc(&d_in,  (size_t)batch * row_b);
            cudaMalloc(&d_res, (size_t)batch * row_b);
            cudaMalloc(&d_out, (size_t)batch * row_b);
            cudaMalloc(&d_lnw, (size_t)hidden * sizeof(float));

            cudaMemset(d_in,  0x3f, (size_t)batch * row_b);
            cudaMemset(d_res, 0x38, (size_t)batch * row_b);
            cudaMemset(d_lnw, 0x3f, (size_t)hidden * sizeof(float));

            GpuTimer timer;

            // Warmup
            for (int w = 0; w < kWarmup; ++w)
                launch_fused_layernorm_residual(d_out, d_res, d_in, d_lnw,
                                                batch, hidden, 1e-6f, sm_version, stream);
            cudaStreamSynchronize(stream);

            // Fused benchmark
            float fused_total = 0.f;
            for (int it = 0; it < kIter; ++it) {
                timer.start(stream);
                launch_fused_layernorm_residual(d_out, d_res, d_in, d_lnw,
                                                batch, hidden, 1e-6f, sm_version, stream);
                fused_total += timer.stop_ms(stream);
            }
            float fused_avg_ms = fused_total / (float)kIter;

            // Naive benchmark (re-init residual each call to avoid divergence)
            cudaMemset(d_res, 0x38, (size_t)batch * row_b);
            float naive_avg_ms = bench_unfused(d_out, d_res, d_in, d_lnw,
                                               batch, hidden, stream, kIter);

            // Effective BW:
            //   Fused reads:  input(2B) + residual_in(2B) + ln_weight(4B≈0)
            //   Fused writes: residual_out(2B) + output(2B)
            //   → 8 × batch × hidden bytes (approx, ignoring ln_weight for large B)
            float bytes_fused = 8.f * (float)batch * (float)hidden;
            // Naive adds an extra residual re-read in the norm kernel:
            //   +2 × batch × hidden bytes (residual_in for norm pass)
            float fused_gbs  = bytes_fused / (fused_avg_ms * 1e-3f) / 1e9f;
            float speedup    = naive_avg_ms / fused_avg_ms;

            // Single-pass threshold: kBlockSize × kVecWidth × kRegBudget
            int reg_budget = (sm_version >= 90) ? 128 : 64;
            int block_size = (sm_version >= 120) ? 512 : 256;
            int max_sp     = block_size * 8 * reg_budget;
            const char* path = (hidden <= max_sp) ? "single-pass" : "two-pass";

            printf("%-6d  %-6d  %12.2f us  %12.2f us  %8.2f×  %10.2f GB/s  %s\n",
                   batch, hidden,
                   fused_avg_ms * 1000.f, naive_avg_ms * 1000.f,
                   speedup, fused_gbs, path);

            cudaFree(d_in); cudaFree(d_res); cudaFree(d_out); cudaFree(d_lnw);
        }
    }

    // ─── Single-pass regime summary ───────────────────────────────────────
    printf("\n─── Single-pass vs two-pass threshold analysis ───────────────\n");
    int reg_budget = (sm_version >= 90) ? 128 : 64;
    int block_size = (sm_version >= 120) ? 512 : 256;
    int max_sp     = block_size * 8 * reg_budget;
    printf("SM%d: kBlockSize=%d, kVecWidth=8, kRegBudgetPerThread=%d\n",
           sm_version, block_size, reg_budget);
    printf("Single-pass threshold: hidden ≤ %d elements (%.1f KB)\n",
           max_sp, max_sp * 4.f / 1024.f);
    printf("Llama-7B (hidden=4096), Llama-70B (hidden=8192): ");
    printf(4096 <= max_sp ? "both single-pass\n" : "may require two-pass\n");
    printf("Single-pass eliminates one full DRAM read of the post-add residual.\n");

    cudaStreamDestroy(stream);
    printf("\nBenchmark complete.\n");
    return 0;
}
