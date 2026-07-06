// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

/*
 * bench_swiglu_ln.cu — Benchmark: fused SwiGLU+RMSNorm vs unfused baseline
 *
 * Partially addresses https://github.com/dylanyunlon/Neuron_SP/issues/25
 *
 * Compiles as a standalone binary.  Requires CUDA ≥ 11.0 with BF16 support.
 *
 *   nvcc -O3 -std=c++17 -arch=sm_86              \
 *        -I../csrc/includes -I../csrc/hetero_reduce \
 *        bench_swiglu_ln.cu -o bench_swiglu_ln
 *
 * Usage:
 *   ./bench_swiglu_ln [batch] [hidden] [warmup_iters] [bench_iters]
 *   defaults: batch=128, hidden=4096, warmup=50, bench=200
 *
 * Output:
 *   - Fused kernel latency (µs)
 *   - Unfused baseline latency (µs)
 *   - Speedup ratio
 *   - Effective bandwidth (GB/s) for each
 *   - Roofline: fraction of theoretical HBM peak
 *   - Numerical correctness check (max abs error)
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>

// ─── Project headers (for DS_D_INLINE, hw_warp_size) ─────────────────────────
#include "ds_kernel_utils.h"
#include "hetero_reduce.h"

// ─── Error-checking macro ────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

namespace cg = cooperative_groups;

// ═════════════════════════════════════════════════════════════════════════════
// Section A: Unfused baseline kernels (separate SwiGLU + separate RMSNorm)
// ═════════════════════════════════════════════════════════════════════════════

// Kernel 1: SwiGLU activation only — writes intermediate buffer
__global__ void unfused_swiglu_kernel(
    __nv_bfloat16* __restrict__       swiglu_out,
    const __nv_bfloat16* __restrict__ gate_proj,
    const __nv_bfloat16* __restrict__ up_proj,
    int hidden)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const __nv_bfloat16* g_row = gate_proj + (size_t)row * hidden;
    const __nv_bfloat16* u_row = up_proj   + (size_t)row * hidden;
    __nv_bfloat16*       o_row = swiglu_out + (size_t)row * hidden;

    for (int col = tid; col < hidden; col += blockDim.x) {
        float gv = __bfloat162float(g_row[col]);
        float uv = __bfloat162float(u_row[col]);
        float sigmoid_g = 1.f / (1.f + __expf(-gv));
        float sw = gv * sigmoid_g * uv;
        o_row[col] = __float2bfloat16(sw);
    }
}

// Kernel 2: RMSNorm on pre-computed SwiGLU output
__global__ void unfused_rmsnorm_kernel(
    __nv_bfloat16* __restrict__       output,
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__         ln_weight,
    int hidden,
    float eps)
{
    const int row = blockIdx.x;
    const __nv_bfloat16* in_row  = input  + (size_t)row * hidden;
    __nv_bfloat16*       out_row = output + (size_t)row * hidden;

    // Pass 1: compute sum of squares
    float sq_sum = 0.f;
    for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
        float v = __bfloat162float(in_row[col]);
        sq_sum += v * v;
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sq_sum += __shfl_xor_sync(0xffffffff, sq_sum, offset);

    // Block reduction via shared memory
    __shared__ float smem[32];
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;
    if (lane == 0) smem[wid] = sq_sum;
    __syncthreads();

    sq_sum = (threadIdx.x < blockDim.x / warpSize) ? smem[threadIdx.x] : 0.f;
    if (wid == 0) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sq_sum += __shfl_xor_sync(0xffffffff, sq_sum, offset);
    }
    if (threadIdx.x == 0) smem[0] = sq_sum;
    __syncthreads();
    sq_sum = smem[0];

    float rms_inv = rsqrtf(sq_sum / (float)hidden + eps);

    // Pass 2: normalize
    for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
        float v = __bfloat162float(in_row[col]);
        float w = ln_weight[col];
        out_row[col] = __float2bfloat16(v * rms_inv * w);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Section B: Include the fused kernel from the main source
//   We re-declare the host-side launcher declared in fused_swiglu_ln.cu.
// ═════════════════════════════════════════════════════════════════════════════

extern void launch_fused_swiglu_ln(
    __nv_bfloat16*       output,
    const __nv_bfloat16* gate_proj,
    const __nv_bfloat16* up_proj,
    const float*         ln_weight,
    int                  batch,
    int                  hidden,
    float                eps,
    int                  sm_version,
    cudaStream_t         stream);

// ═════════════════════════════════════════════════════════════════════════════
// Section C: Roofline helper
// ═════════════════════════════════════════════════════════════════════════════

struct DeviceRoofline {
    double peak_bw_gb_s;   // theoretical HBM bandwidth (GB/s)
    double peak_flops_gf;  // theoretical BF16 TFLOPS  (unused for mem-bound)
    const char* gpu_name;
};

static DeviceRoofline get_device_roofline()
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // peak bandwidth = memory_clock_rate (kHz) × bus_width (bits) × 2 (DDR) / 8
    double bw = (double)prop.memoryClockRate * 1e3          // Hz
                * ((double)prop.memoryBusWidth / 8.0)       // bytes per clock per dir
                * 2.0                                       // DDR
                / 1e9;                                      // → GB/s

    DeviceRoofline r;
    r.peak_bw_gb_s  = bw;
    r.peak_flops_gf = 0.0;  // not used for this bandwidth-bound kernel
    r.gpu_name      = prop.name;
    return r;
}

// ═════════════════════════════════════════════════════════════════════════════
// Section D: Timing utility
// ═════════════════════════════════════════════════════════════════════════════

struct BenchResult {
    double mean_us;
    double min_us;
    double max_us;
    double eff_bw_gb_s;
    double roofline_pct;
};

// ═════════════════════════════════════════════════════════════════════════════
// Section E: main()
// ═════════════════════════════════════════════════════════════════════════════

int main(int argc, char** argv)
{
    // ── Parse CLI args ──────────────────────────────────────────────────────
    int batch        = (argc > 1) ? atoi(argv[1]) : 128;
    int hidden       = (argc > 2) ? atoi(argv[2]) : 4096;
    int warmup_iters = (argc > 3) ? atoi(argv[3]) : 50;
    int bench_iters  = (argc > 4) ? atoi(argv[4]) : 200;
    float eps        = 1e-6f;

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Benchmark: Fused SwiGLU+RMSNorm  vs  Unfused Baseline\n");
    printf("  batch=%d  hidden=%d  warmup=%d  iters=%d  eps=%.1e\n",
           batch, hidden, warmup_iters, bench_iters, eps);
    printf("═══════════════════════════════════════════════════════════════\n\n");

    // ── Device info & roofline ──────────────────────────────────────────────
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int sm_version = prop.major * 10 + prop.minor;

    DeviceRoofline roof = get_device_roofline();
    printf("  GPU:            %s  (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("  Peak HBM BW:    %.1f GB/s (theoretical)\n", roof.peak_bw_gb_s);
    printf("\n");

    // ── Allocate device memory ──────────────────────────────────────────────
    size_t n_elem       = (size_t)batch * hidden;
    size_t bf16_bytes   = n_elem * sizeof(__nv_bfloat16);
    size_t weight_bytes = (size_t)hidden * sizeof(float);

    __nv_bfloat16 *d_gate, *d_up, *d_out_fused, *d_out_unfused, *d_swiglu_tmp;
    float *d_ln_weight;

    CUDA_CHECK(cudaMalloc(&d_gate,        bf16_bytes));
    CUDA_CHECK(cudaMalloc(&d_up,          bf16_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_fused,   bf16_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_unfused, bf16_bytes));
    CUDA_CHECK(cudaMalloc(&d_swiglu_tmp,  bf16_bytes));   // unfused intermediate
    CUDA_CHECK(cudaMalloc(&d_ln_weight,   weight_bytes));

    // ── Initialise with random data ─────────────────────────────────────────
    std::vector<__nv_bfloat16> h_gate(n_elem), h_up(n_elem);
    std::vector<float> h_weight(hidden);

    srand(42);
    for (size_t i = 0; i < n_elem; ++i) {
        h_gate[i] = __float2bfloat16((float)(rand() % 1000 - 500) / 500.f);
        h_up[i]   = __float2bfloat16((float)(rand() % 1000 - 500) / 500.f);
    }
    for (int i = 0; i < hidden; ++i)
        h_weight[i] = 0.5f + (float)(rand() % 1000) / 1000.f;

    CUDA_CHECK(cudaMemcpy(d_gate,      h_gate.data(),   bf16_bytes,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up,        h_up.data(),     bf16_bytes,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ln_weight, h_weight.data(), weight_bytes, cudaMemcpyHostToDevice));

    // ── Memory traffic calculation (bytes) ──────────────────────────────────
    //
    // Fused kernel (single-pass):
    //   reads:  gate(bf16) + up(bf16) + ln_weight(f32)
    //   writes: output(bf16)
    //   total = batch*hidden*2 (gate) + batch*hidden*2 (up)
    //         + hidden*4 (weight, amortised) + batch*hidden*2 (output)
    //         = batch*hidden*6 + hidden*4
    //
    // Unfused baseline:
    //   swiglu kernel:  reads gate+up, writes tmp
    //     = batch*hidden*(2+2+2) = batch*hidden*6
    //   rmsnorm kernel: reads tmp + weight, writes output
    //     Pass1: reads tmp → batch*hidden*2
    //     Pass2: reads tmp + weight, writes output → batch*hidden*(2+4/batch+2)
    //     ≈ batch*hidden*6 + hidden*4  (two full passes over tmp)
    //   total ≈ batch*hidden*12 + hidden*4
    //
    double fused_bytes   = (double)batch * hidden * 6.0 + (double)hidden * 4.0;
    double unfused_bytes = (double)batch * hidden * 12.0 + (double)hidden * 4.0;

    printf("  Memory traffic (estimated):\n");
    printf("    Fused:   %.2f MB\n", fused_bytes / 1e6);
    printf("    Unfused: %.2f MB\n", unfused_bytes / 1e6);
    printf("\n");

    // ── CUDA events for timing ──────────────────────────────────────────────
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Block size for unfused kernels
    const int unfused_block = 256;

    // ── Warmup: fused ────────────────────────────────────────────────────────
    for (int i = 0; i < warmup_iters; ++i) {
        launch_fused_swiglu_ln(d_out_fused, d_gate, d_up, d_ln_weight,
                               batch, hidden, eps, sm_version, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── Benchmark: fused ────────────────────────────────────────────────────
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < bench_iters; ++i) {
        launch_fused_swiglu_ln(d_out_fused, d_gate, d_up, d_ln_weight,
                               batch, hidden, eps, sm_version, stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float fused_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&fused_ms, start, stop));
    double fused_us = (double)fused_ms * 1000.0 / bench_iters;
    double fused_bw = fused_bytes / (fused_us * 1e-6) / 1e9;   // GB/s
    double fused_roof = fused_bw / roof.peak_bw_gb_s * 100.0;

    // ── Warmup: unfused ─────────────────────────────────────────────────────
    for (int i = 0; i < warmup_iters; ++i) {
        unfused_swiglu_kernel<<<batch, unfused_block, 0, stream>>>(
            d_swiglu_tmp, d_gate, d_up, hidden);
        unfused_rmsnorm_kernel<<<batch, unfused_block, 0, stream>>>(
            d_out_unfused, d_swiglu_tmp, d_ln_weight, hidden, eps);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── Benchmark: unfused ──────────────────────────────────────────────────
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < bench_iters; ++i) {
        unfused_swiglu_kernel<<<batch, unfused_block, 0, stream>>>(
            d_swiglu_tmp, d_gate, d_up, hidden);
        unfused_rmsnorm_kernel<<<batch, unfused_block, 0, stream>>>(
            d_out_unfused, d_swiglu_tmp, d_ln_weight, hidden, eps);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float unfused_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&unfused_ms, start, stop));
    double unfused_us = (double)unfused_ms * 1000.0 / bench_iters;
    double unfused_bw = unfused_bytes / (unfused_us * 1e-6) / 1e9;
    double unfused_roof = unfused_bw / roof.peak_bw_gb_s * 100.0;

    // ── Numerical correctness ───────────────────────────────────────────────
    std::vector<__nv_bfloat16> h_fused(n_elem), h_unfused(n_elem);
    CUDA_CHECK(cudaMemcpy(h_fused.data(),   d_out_fused,   bf16_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_unfused.data(), d_out_unfused, bf16_bytes, cudaMemcpyDeviceToHost));

    float max_abs_err = 0.f;
    double sum_abs_err = 0.0;
    for (size_t i = 0; i < n_elem; ++i) {
        float fv = __bfloat162float(h_fused[i]);
        float uv = __bfloat162float(h_unfused[i]);
        float ae = fabsf(fv - uv);
        if (ae > max_abs_err) max_abs_err = ae;
        sum_abs_err += ae;
    }
    float mean_abs_err = (float)(sum_abs_err / n_elem);

    // ── Print results ───────────────────────────────────────────────────────
    printf("─────────────────────────────────────────────────────────────\n");
    printf("  RESULTS\n");
    printf("─────────────────────────────────────────────────────────────\n");
    printf("  %-22s %10s %10s %10s\n", "", "Fused", "Unfused", "Ratio");
    printf("  %-22s %9.1f µs %8.1f µs %9.2fx\n",
           "Latency",    fused_us, unfused_us, unfused_us / fused_us);
    printf("  %-22s %8.1f GB/s %6.1f GB/s %9.2fx\n",
           "Eff. bandwidth", fused_bw, unfused_bw, fused_bw / unfused_bw);
    printf("  %-22s %9.1f %%  %8.1f %%\n",
           "Roofline (%% peak)", fused_roof, unfused_roof);
    printf("─────────────────────────────────────────────────────────────\n");
    printf("  Correctness:  max|err|=%.6f  mean|err|=%.8f\n",
           max_abs_err, mean_abs_err);
    printf("  (BF16 has ~0.01 relative precision; small diffs expected)\n");
    printf("─────────────────────────────────────────────────────────────\n\n");

    // ── Roofline analysis ───────────────────────────────────────────────────
    printf("  ROOFLINE ANALYSIS\n");
    printf("─────────────────────────────────────────────────────────────\n");
    printf("  This kernel is memory-bandwidth-bound (arithmetic intensity\n");
    printf("  ≈ 5 FLOPs/element ÷ 6 bytes/element ≈ 0.83 FLOP/byte).\n");
    printf("  The fused kernel eliminates the intermediate BF16 buffer,\n");
    printf("  cutting DRAM traffic from ~%.0f MB to ~%.0f MB (%.1fx).\n",
           unfused_bytes / 1e6, fused_bytes / 1e6,
           unfused_bytes / fused_bytes);
    printf("\n");
    printf("  Theoretical peak HBM bandwidth: %.1f GB/s\n", roof.peak_bw_gb_s);
    printf("  Fused achieved:   %6.1f GB/s  (%.1f%% of peak)\n",
           fused_bw, fused_roof);
    printf("  Unfused achieved: %6.1f GB/s  (%.1f%% of peak)\n",
           unfused_bw, unfused_roof);
    printf("─────────────────────────────────────────────────────────────\n");

    // ── Sweep hidden sizes ──────────────────────────────────────────────────
    printf("\n  HIDDEN-SIZE SWEEP (batch=%d)\n", batch);
    printf("─────────────────────────────────────────────────────────────\n");
    printf("  %8s  %10s  %10s  %8s\n", "hidden", "fused(µs)", "unfused(µs)", "speedup");

    int hidden_sizes[] = {1024, 2048, 4096, 8192, 11008, 13824, 16384};
    for (int h : hidden_sizes) {
        size_t ne = (size_t)batch * h;
        size_t nb = ne * sizeof(__nv_bfloat16);
        size_t wb = (size_t)h * sizeof(float);

        // Realloc if bigger
        __nv_bfloat16 *dg, *du, *do_f, *do_u, *dt;
        float *dw;
        CUDA_CHECK(cudaMalloc(&dg,   nb));
        CUDA_CHECK(cudaMalloc(&du,   nb));
        CUDA_CHECK(cudaMalloc(&do_f, nb));
        CUDA_CHECK(cudaMalloc(&do_u, nb));
        CUDA_CHECK(cudaMalloc(&dt,   nb));
        CUDA_CHECK(cudaMalloc(&dw,   wb));
        CUDA_CHECK(cudaMemset(dg, 0x3f, nb));  // fill with non-zero BF16
        CUDA_CHECK(cudaMemset(du, 0x3f, nb));
        // init weight to 1.0f
        std::vector<float> ones(h, 1.0f);
        CUDA_CHECK(cudaMemcpy(dw, ones.data(), wb, cudaMemcpyHostToDevice));

        // warmup
        for (int i = 0; i < 20; ++i) {
            launch_fused_swiglu_ln(do_f, dg, du, dw, batch, h, eps, sm_version, stream);
            unfused_swiglu_kernel<<<batch, unfused_block, 0, stream>>>(dt, dg, du, h);
            unfused_rmsnorm_kernel<<<batch, unfused_block, 0, stream>>>(do_u, dt, dw, h, eps);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // bench fused
        CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < bench_iters; ++i)
            launch_fused_swiglu_ln(do_f, dg, du, dw, batch, h, eps, sm_version, stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        float f_ms;
        CUDA_CHECK(cudaEventElapsedTime(&f_ms, start, stop));
        double f_us = (double)f_ms * 1000.0 / bench_iters;

        // bench unfused
        CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < bench_iters; ++i) {
            unfused_swiglu_kernel<<<batch, unfused_block, 0, stream>>>(dt, dg, du, h);
            unfused_rmsnorm_kernel<<<batch, unfused_block, 0, stream>>>(do_u, dt, dw, h, eps);
        }
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        float u_ms;
        CUDA_CHECK(cudaEventElapsedTime(&u_ms, start, stop));
        double u_us = (double)u_ms * 1000.0 / bench_iters;

        printf("  %8d  %9.1f µs  %9.1f µs  %7.2fx\n",
               h, f_us, u_us, u_us / f_us);

        CUDA_CHECK(cudaFree(dg));
        CUDA_CHECK(cudaFree(du));
        CUDA_CHECK(cudaFree(do_f));
        CUDA_CHECK(cudaFree(do_u));
        CUDA_CHECK(cudaFree(dt));
        CUDA_CHECK(cudaFree(dw));
    }
    printf("─────────────────────────────────────────────────────────────\n");

    // ── Cleanup ─────────────────────────────────────────────────────────────
    CUDA_CHECK(cudaFree(d_gate));
    CUDA_CHECK(cudaFree(d_up));
    CUDA_CHECK(cudaFree(d_out_fused));
    CUDA_CHECK(cudaFree(d_out_unfused));
    CUDA_CHECK(cudaFree(d_swiglu_tmp));
    CUDA_CHECK(cudaFree(d_ln_weight));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));

    printf("\nDone.\n");
    return 0;
}
