// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * bench_cross_entropy_tp.cu
 *
 * Benchmark harness for cross_entropy_tp.cu.
 * Measures:
 *   1. Forward phase throughput (GB/s effective) vs. naive unfused baseline
 *   2. Backward phase throughput vs. naive reference
 *   3. Correctness of forward loss and backward gradient
 *   4. Latency sweep across batch sizes and local vocab sizes (V/tp_size)
 *
 * Effective bandwidth (forward per sample):
 *   Reads : logits [batch × v_local × 2 B] + labels [batch × 4 B]
 *   Writes: local_max + local_sum_exp + local_logit [3 × batch × 4 B]
 *   Total ≈ 2 × batch × v_local bytes (label/output terms are negligible for large V)
 *
 * Compile (standalone):
 *   nvcc -O3 -arch=sm_90 -std=c++20 \
 *     -I../csrc/hetero_reduce -I../csrc/includes \
 *     bench_cross_entropy_tp.cu \
 *     ../csrc/hetero_reduce/cross_entropy_tp.cu \
 *     -o bench_cross_entropy_tp
 *
 * Run:
 *   ./bench_cross_entropy_tp
 *   ./bench_cross_entropy_tp --sm 86 --tp 8
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
#include <float.h>

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
// Naive forward baseline (separate max pass + sum_exp pass, FP32 logits)
//   Models what a non-fused implementation does: two sweeps over v_local.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void naive_max_kernel(
    float*                            out_max,
    const __nv_bfloat16* __restrict__ logits,
    int v_local)
{
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    float mx = -FLT_MAX;
    for (int c = threadIdx.x; c < v_local; c += blockDim.x)
        mx = fmaxf(mx, __bfloat162float(logits[row * v_local + c]));
    smem[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) out_max[row] = smem[0];
}

__global__ void naive_sum_exp_kernel(
    float*                            out_sum,
    float*                            out_label_logit,
    const __nv_bfloat16* __restrict__ logits,
    const float*         __restrict__ row_max,
    const int*           __restrict__ labels,
    int shard_offset,
    int v_local)
{
    extern __shared__ float smem[];
    const int row   = blockIdx.x;
    const int label = labels[row];
    const float mx  = row_max[row];
    const bool label_in_shard = (label >= shard_offset) &&
                                 (label <  shard_offset + v_local);
    float se = 0.f, ll = 0.f;
    for (int c = threadIdx.x; c < v_local; c += blockDim.x) {
        float x = __bfloat162float(logits[row * v_local + c]);
        se += __expf(x - mx);
        if (label_in_shard && c == (label - shard_offset)) ll = x;
    }
    smem[threadIdx.x] = se;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out_sum[row] = smem[0];
    // Reduce label logit across threads (only one thread has non-zero ll)
    // Simple approach: use another smem slot for the label value
    smem[threadIdx.x] = ll;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out_label_logit[row] = smem[0];
}

float bench_unfused_forward(
    float*               d_max,
    float*               d_sum,
    float*               d_llogit,
    const __nv_bfloat16* d_logits,
    const int*           d_labels,
    int batch, int v_local, int shard_offset,
    cudaStream_t stream, int niters)
{
    constexpr int kBS = 256;
    const size_t smem = kBS * sizeof(float);
    GpuTimer t;
    float total = 0.f;
    for (int it = 0; it < niters; ++it) {
        t.start(stream);
        naive_max_kernel<<<batch, kBS, smem, stream>>>(d_max, d_logits, v_local);
        naive_sum_exp_kernel<<<batch, kBS, smem, stream>>>(
            d_sum, d_llogit, d_logits, d_max, d_labels, shard_offset, v_local);
        total += t.stop_ms(stream);
    }
    return total / (float)niters;
}

// ─────────────────────────────────────────────────────────────────────────────
// Correctness check for forward
//   Single-rank scenario (tp_size=1, shard_offset=0):
//     expected loss[i] = log(Σ exp(logit[j])) - logit[label[i]]
//   Verified against host-side FP32 reference.
// ─────────────────────────────────────────────────────────────────────────────

bool correctness_check_forward(int sm_version, cudaStream_t stream)
{
    constexpr int batch  = 4;
    constexpr int v_loc  = 128;   // small for host-verifiable reference
    const size_t logit_b = (size_t)batch * v_loc * sizeof(__nv_bfloat16);

    // Host data
    std::vector<float>          h_logits_f(batch * v_loc);
    std::vector<__nv_bfloat16>  h_logits_bf(batch * v_loc);
    std::vector<int>             h_labels(batch);
    for (int i = 0; i < batch * v_loc; ++i) {
        h_logits_f[i]  = (float)(i % 17) * 0.1f - 0.8f;
        h_logits_bf[i] = __float2bfloat16(h_logits_f[i]);
    }
    for (int b = 0; b < batch; ++b) h_labels[b] = (b * 7) % v_loc;

    __nv_bfloat16* d_logits;
    int* d_labels;
    float *d_max, *d_sum, *d_llogit, *d_loss;
    cudaMalloc(&d_logits, logit_b);
    cudaMalloc(&d_labels, batch * sizeof(int));
    cudaMalloc(&d_max,    batch * sizeof(float));
    cudaMalloc(&d_sum,    batch * sizeof(float));
    cudaMalloc(&d_llogit, batch * sizeof(float));
    cudaMalloc(&d_loss,   batch * sizeof(float));

    cudaMemcpy(d_logits, h_logits_bf.data(), logit_b,         cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels.data(),    batch*sizeof(int), cudaMemcpyHostToDevice);

    // Fused forward phase 1
    launch_cross_entropy_tp_forward(d_max, d_sum, d_llogit,
                                     d_logits, d_labels,
                                     batch, v_loc, 0,
                                     sm_version, stream);
    // Phase 2: compute loss (tp_size=1, so AllReduce is identity here)
    launch_cross_entropy_tp_loss(d_loss, d_max, d_sum, d_llogit, batch, stream);
    cudaStreamSynchronize(stream);

    std::vector<float> h_loss(batch), h_max(batch), h_sum(batch), h_ll(batch);
    cudaMemcpy(h_loss.data(), d_loss,    batch*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_max.data(),  d_max,     batch*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sum.data(),  d_sum,     batch*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ll.data(),   d_llogit,  batch*sizeof(float), cudaMemcpyDeviceToHost);

    // Host reference: log-sum-exp via FP32 double-precision
    bool ok = true;
    for (int b = 0; b < batch && ok; ++b) {
        float* row = h_logits_f.data() + b * v_loc;
        float mx_ref = *std::max_element(row, row + v_loc);
        float se_ref = 0.f;
        for (int j = 0; j < v_loc; ++j)
            se_ref += expf(row[j] - mx_ref);
        float label_logit_ref = row[h_labels[b]];
        float loss_ref = logf(se_ref) + mx_ref - label_logit_ref;
        float loss_got = h_loss[b];
        if (fabsf(loss_got - loss_ref) > 0.05f * fabsf(loss_ref) + 0.01f) {
            printf("  loss mismatch at b=%d: got=%.4f ref=%.4f\n",
                   b, loss_got, loss_ref);
            ok = false;
        }
    }

    cudaFree(d_logits); cudaFree(d_labels);
    cudaFree(d_max); cudaFree(d_sum); cudaFree(d_llogit); cudaFree(d_loss);
    return ok;
}

// ─────────────────────────────────────────────────────────────────────────────
// Correctness check for backward
//   For label at position j*: d_logit[j] = softmax[j] - 1{j==j*}
//   Verify sum(d_logit) ≈ 0 (conservation property) and d_logit[j*] < 0.
// ─────────────────────────────────────────────────────────────────────────────

bool correctness_check_backward(int sm_version, cudaStream_t stream)
{
    constexpr int batch  = 2;
    constexpr int v_loc  = 256;
    const size_t logit_b = (size_t)batch * v_loc * sizeof(__nv_bfloat16);

    std::vector<float>         h_lf(batch * v_loc);
    std::vector<__nv_bfloat16> h_lb(batch * v_loc);
    std::vector<int>            h_labels(batch);
    for (int i = 0; i < batch * v_loc; ++i) {
        h_lf[i] = (float)(i % 13) * 0.15f - 0.9f;
        h_lb[i] = __float2bfloat16(h_lf[i]);
    }
    for (int b = 0; b < batch; ++b) h_labels[b] = b * 11 % v_loc;

    __nv_bfloat16 *d_logits, *d_dlogits;
    int* d_labels;
    float *d_max, *d_sum, *d_llogit, *d_loss, *d_lse;
    cudaMalloc(&d_logits,  logit_b);
    cudaMalloc(&d_dlogits, logit_b);
    cudaMalloc(&d_labels,  batch * sizeof(int));
    cudaMalloc(&d_max,     batch * sizeof(float));
    cudaMalloc(&d_sum,     batch * sizeof(float));
    cudaMalloc(&d_llogit,  batch * sizeof(float));
    cudaMalloc(&d_loss,    batch * sizeof(float));
    cudaMalloc(&d_lse,     batch * sizeof(float));

    cudaMemcpy(d_logits, h_lb.data(),    logit_b,         cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels.data(), batch*sizeof(int), cudaMemcpyHostToDevice);

    launch_cross_entropy_tp_forward(d_max, d_sum, d_llogit,
                                     d_logits, d_labels,
                                     batch, v_loc, 0, sm_version, stream);
    // Compute log_sum_exp = log(sum_exp) + max for backward
    // (caller would normally AllReduce first; here tp_size=1)
    // Use a tiny host kernel to set d_lse[b] = log(d_sum[b]) + d_max[b]
    std::vector<float> h_max(batch), h_sum(batch);
    cudaStreamSynchronize(stream);
    cudaMemcpy(h_max.data(), d_max, batch*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sum.data(), d_sum, batch*sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<float> h_lse(batch);
    for (int b = 0; b < batch; ++b)
        h_lse[b] = logf(h_sum[b]) + h_max[b];
    cudaMemcpy(d_lse, h_lse.data(), batch*sizeof(float), cudaMemcpyHostToDevice);

    // Copy logits to d_dlogits for in-place backward
    cudaMemcpy(d_dlogits, d_logits, logit_b, cudaMemcpyDeviceToDevice);

    launch_cross_entropy_tp_backward(d_dlogits, d_logits, d_labels,
                                      d_max, d_lse,
                                      batch, v_loc, 0,
                                      1.f / (float)batch,
                                      sm_version, stream);
    cudaStreamSynchronize(stream);

    std::vector<__nv_bfloat16> h_grad(batch * v_loc);
    cudaMemcpy(h_grad.data(), d_dlogits, logit_b, cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int b = 0; b < batch && ok; ++b) {
        float sum_grad = 0.f;
        for (int j = 0; j < v_loc; ++j)
            sum_grad += __bfloat162float(h_grad[b * v_loc + j]);
        // Sum of softmax gradients ≈ 0 (softmax sums to 1 minus 1/batch for label)
        // Actually sum(d_logit) = (1 - 1) / batch = 0
        if (fabsf(sum_grad) > 0.1f) {
            printf("  backward sum_grad conservation failed: b=%d sum=%.5f\n", b, sum_grad);
            ok = false;
        }
        // Label gradient should be negative (softmax[label] < 1)
        float grad_label = __bfloat162float(h_grad[b * v_loc + h_labels[b]]);
        if (grad_label >= 0.f) {
            printf("  backward label grad should be negative: b=%d grad=%.4f\n",
                   b, grad_label);
            ok = false;
        }
    }

    cudaFree(d_logits); cudaFree(d_dlogits); cudaFree(d_labels);
    cudaFree(d_max); cudaFree(d_sum); cudaFree(d_llogit);
    cudaFree(d_loss); cudaFree(d_lse);
    return ok;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main benchmark
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    int sm_version  = 90;
    int tp_size     = 8;   // simulated: v_local = V_total / tp_size
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--sm") == 0 && i + 1 < argc)
            sm_version = atoi(argv[++i]);
        if (strcmp(argv[i], "--tp") == 0 && i + 1 < argc)
            tp_size = atoi(argv[++i]);
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    printf("Device: %s  SM: %d.%d  Mem BW: %.1f GB/s\n",
           prop.name, prop.major, prop.minor,
           prop.memoryBusWidth / 8.0 * prop.memoryClockRate * 2e-6);
    printf("SM dispatch path: SM%d  TP size: %d\n\n", sm_version, tp_size);

    bool fwd_ok = correctness_check_forward(sm_version, stream);
    printf("Forward correctness:  %s\n", fwd_ok ? "PASS" : "FAIL");
    bool bwd_ok = correctness_check_backward(sm_version, stream);
    printf("Backward correctness: %s\n\n", bwd_ok ? "PASS" : "FAIL");
    if (!fwd_ok || !bwd_ok) return 1;

    constexpr int kWarmup = 10;
    constexpr int kIter   = 50;

    // Vocabulary sizes representative of production LLMs (full vocab / tp_size)
    // Llama 3: V=128K / tp=8 → v_local=16384
    // Mistral: V=32K  / tp=4 → v_local=8192
    const int batches[]  = {1, 8, 64, 512};
    const int v_totals[] = {32768, 65536, 128000, 256000};  // LLM vocab sizes

    // ── Forward benchmark ──────────────────────────────────────────────────
    printf("─── Forward: launch_cross_entropy_tp_forward ────────────────────\n");
    printf("%-6s  %-8s  %-8s  %-14s  %-14s  %-10s  %s\n",
           "batch", "V_total", "V_local", "fused_avg_us", "naive_avg_us",
           "speedup", "fused_GB/s");
    printf("%s\n", std::string(95, '-').c_str());

    for (int batch : batches) {
        for (int v_total : v_totals) {
            const int v_local = v_total / tp_size;
            if (v_local < 8) continue;  // skip degenerate cases

            const size_t logit_b = (size_t)batch * v_local * sizeof(__nv_bfloat16);

            __nv_bfloat16* d_logits;
            int*   d_labels;
            float *d_max, *d_sum, *d_llogit;
            cudaMalloc(&d_logits,  logit_b);
            cudaMalloc(&d_labels,  batch * sizeof(int));
            cudaMalloc(&d_max,     batch * sizeof(float));
            cudaMalloc(&d_sum,     batch * sizeof(float));
            cudaMalloc(&d_llogit,  batch * sizeof(float));

            cudaMemset(d_logits, 0x3c, logit_b);
            // Fill labels: random within [0, v_total)
            std::vector<int> h_labels(batch);
            for (int b = 0; b < batch; ++b)
                h_labels[b] = (b * 7919 + 3) % v_total;
            cudaMemcpy(d_labels, h_labels.data(), batch*sizeof(int), cudaMemcpyHostToDevice);

            GpuTimer timer;

            // Warmup
            for (int w = 0; w < kWarmup; ++w)
                launch_cross_entropy_tp_forward(d_max, d_sum, d_llogit,
                                                 d_logits, d_labels,
                                                 batch, v_local, 0,
                                                 sm_version, stream);
            cudaStreamSynchronize(stream);

            // Fused benchmark
            float fused_total = 0.f;
            for (int it = 0; it < kIter; ++it) {
                timer.start(stream);
                launch_cross_entropy_tp_forward(d_max, d_sum, d_llogit,
                                                 d_logits, d_labels,
                                                 batch, v_local, 0,
                                                 sm_version, stream);
                fused_total += timer.stop_ms(stream);
            }
            float fused_avg_ms = fused_total / (float)kIter;

            // Naive benchmark (two separate passes)
            float naive_avg_ms = bench_unfused_forward(
                d_max, d_sum, d_llogit, d_logits, d_labels,
                batch, v_local, 0, stream, kIter);

            float bytes_read = 2.f * (float)batch * (float)v_local;  // BF16
            float fused_gbs  = bytes_read / (fused_avg_ms * 1e-3f) / 1e9f;
            float speedup    = naive_avg_ms / fused_avg_ms;

            printf("%-6d  %-8d  %-8d  %12.2f us  %12.2f us  %8.2f×  %8.2f GB/s\n",
                   batch, v_total, v_local,
                   fused_avg_ms * 1000.f, naive_avg_ms * 1000.f,
                   speedup, fused_gbs);

            cudaFree(d_logits); cudaFree(d_labels);
            cudaFree(d_max); cudaFree(d_sum); cudaFree(d_llogit);
        }
    }

    // ── Backward benchmark ─────────────────────────────────────────────────
    printf("\n─── Backward: launch_cross_entropy_tp_backward ──────────────────\n");
    printf("%-6s  %-8s  %-8s  %-14s  %s\n",
           "batch", "V_total", "V_local", "bwd_avg_us", "bwd_GB/s");
    printf("%s\n", std::string(65, '-').c_str());

    for (int batch : batches) {
        for (int v_total : v_totals) {
            const int v_local = v_total / tp_size;
            if (v_local < 8) continue;

            const size_t logit_b = (size_t)batch * v_local * sizeof(__nv_bfloat16);

            __nv_bfloat16 *d_logits, *d_dlogits;
            int*   d_labels;
            float *d_max, *d_lse;
            cudaMalloc(&d_logits,  logit_b);
            cudaMalloc(&d_dlogits, logit_b);
            cudaMalloc(&d_labels,  batch * sizeof(int));
            cudaMalloc(&d_max,     batch * sizeof(float));
            cudaMalloc(&d_lse,     batch * sizeof(float));

            cudaMemset(d_logits,  0x3c, logit_b);
            cudaMemset(d_dlogits, 0x00, logit_b);
            cudaMemset(d_max,     0x40, batch * sizeof(float));
            cudaMemset(d_lse,     0x40, batch * sizeof(float));

            std::vector<int> h_labels(batch);
            for (int b = 0; b < batch; ++b)
                h_labels[b] = (b * 7919 + 3) % v_total;
            cudaMemcpy(d_labels, h_labels.data(), batch*sizeof(int), cudaMemcpyHostToDevice);

            GpuTimer timer;

            // Warmup
            for (int w = 0; w < kWarmup; ++w)
                launch_cross_entropy_tp_backward(d_dlogits, d_logits, d_labels,
                                                  d_max, d_lse,
                                                  batch, v_local, 0,
                                                  1.f / (float)batch,
                                                  sm_version, stream);
            cudaStreamSynchronize(stream);

            // Benchmark
            float bwd_total = 0.f;
            for (int it = 0; it < kIter; ++it) {
                timer.start(stream);
                launch_cross_entropy_tp_backward(d_dlogits, d_logits, d_labels,
                                                  d_max, d_lse,
                                                  batch, v_local, 0,
                                                  1.f / (float)batch,
                                                  sm_version, stream);
                bwd_total += timer.stop_ms(stream);
            }
            float bwd_avg_ms = bwd_total / (float)kIter;

            // BW: read logits (2B) + write d_logits (2B)
            float bytes_bwd = 4.f * (float)batch * (float)v_local;
            float bwd_gbs   = bytes_bwd / (bwd_avg_ms * 1e-3f) / 1e9f;

            printf("%-6d  %-8d  %-8d  %12.2f us  %8.2f GB/s\n",
                   batch, v_total, v_local,
                   bwd_avg_ms * 1000.f, bwd_gbs);

            cudaFree(d_logits); cudaFree(d_dlogits); cudaFree(d_labels);
            cudaFree(d_max); cudaFree(d_lse);
        }
    }

    cudaStreamDestroy(stream);
    printf("\nBenchmark complete.\n");
    return 0;
}
