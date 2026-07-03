// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * bench_pcie_allreduce.cu
 *
 * Benchmark harness for pcie_adaptive_allreduce.cu.
 * Measures:
 *   1. Ring-reduce kernel throughput vs. tensor size
 *   2. Gradient packing throughput (scatter-gather efficiency)
 *   3. Adaptive chunk size across simulated PCIe bandwidths
 *   4. Finalisation kernel throughput
 *   5. End-to-end simulated ring-allreduce timing with overlap model
 *
 * The PCIe bandwidth probe (probe_pcie_bandwidth) requires a multi-GPU
 * system; we simulate it with a bandwidth override flag for single-GPU
 * environments.
 *
 * Compile:
 *   nvcc -O3 -arch=sm_90 -std=c++20 \
 *     -I../csrc/hetero_reduce -I../csrc/includes \
 *     bench_pcie_allreduce.cu ../csrc/hetero_reduce/pcie_adaptive_allreduce.cu \
 *     -o bench_pcie_allreduce
 *
 * Run (single GPU):
 *   ./bench_pcie_allreduce --bw 10.0     # simulate 10 GB/s PCIe
 *   ./bench_pcie_allreduce --bw 32.0     # simulate 32 GB/s PCIe
 *   ./bench_pcie_allreduce               # probe actual bandwidth (multi-GPU)
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cmath>

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

static constexpr int kWarmup = 10, kIter = 50;

// ─────────────────────────────────────────────────────────────────────────────
// Correctness check for ring_reduce:
//   dst initially all-1, src all-1 → expect all-2
// ─────────────────────────────────────────────────────────────────────────────

bool check_ring_reduce(int sm_version, cudaStream_t stream)
{
    constexpr size_t N = 1024;
    __nv_bfloat16 *d_dst, *d_src;
    cudaMalloc(&d_dst, N * 2);
    cudaMalloc(&d_src, N * 2);
    std::vector<__nv_bfloat16> h(N, __float2bfloat16(1.f));
    cudaMemcpy(d_dst, h.data(), N * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_src, h.data(), N * 2, cudaMemcpyHostToDevice);

    launch_pcie_ring_reduce(d_dst, d_src, N, sm_version, stream);
    cudaStreamSynchronize(stream);

    std::vector<__nv_bfloat16> hr(N);
    cudaMemcpy(hr.data(), d_dst, N * 2, cudaMemcpyDeviceToHost);
    bool ok = true;
    for (size_t i = 0; i < N; ++i)
        if (std::abs(__bfloat162float(hr[i]) - 2.f) > 0.01f) { ok = false; break; }

    cudaFree(d_dst); cudaFree(d_src);
    return ok;
}

// ─────────────────────────────────────────────────────────────────────────────
// Correctness check for finalise:
//   src all-4, world_size=4 → expect all-1
// ─────────────────────────────────────────────────────────────────────────────

bool check_finalise(int sm_version, cudaStream_t stream)
{
    constexpr size_t N = 1024;
    __nv_bfloat16 *d_out, *d_src;
    cudaMalloc(&d_out, N * 2);
    cudaMalloc(&d_src, N * 2);
    std::vector<__nv_bfloat16> h(N, __float2bfloat16(4.f));
    cudaMemcpy(d_src, h.data(), N * 2, cudaMemcpyHostToDevice);

    launch_pcie_allreduce_finalise(d_out, d_src, N, 4, sm_version, stream);
    cudaStreamSynchronize(stream);

    std::vector<__nv_bfloat16> hr(N);
    cudaMemcpy(hr.data(), d_out, N * 2, cudaMemcpyDeviceToHost);
    bool ok = true;
    for (size_t i = 0; i < N; ++i)
        if (std::abs(__bfloat162float(hr[i]) - 1.f) > 0.01f) { ok = false; break; }

    cudaFree(d_out); cudaFree(d_src);
    return ok;
}

// ─────────────────────────────────────────────────────────────────────────────
// Double-buffered ring allreduce simulation
//   Simulates world_size-1 ring steps on a single GPU using two buffers.
//   This validates the pipeline logic without requiring multi-GPU.
//   Each step: reduce accum with recv_buf (alternating ping/pong).
// ─────────────────────────────────────────────────────────────────────────────

float bench_ring_allreduce_sim(
    __nv_bfloat16* d_accum,      // [chunk_elems]
    __nv_bfloat16* d_ping,       // recv buffer A
    __nv_bfloat16* d_pong,       // recv buffer B
    size_t chunk_elems,
    int world_size,
    int sm_version,
    cudaStream_t compute_stream,
    cudaStream_t transfer_stream)
{
    // Use CUDA events to model compute-transfer overlap
    cudaEvent_t ping_ready, pong_ready, reduce_done;
    cudaEventCreate(&ping_ready);
    cudaEventCreate(&pong_ready);
    cudaEventCreate(&reduce_done);

    GpuTimer timer;
    float total_ms = 0.f;

    for (int iter = 0; iter < kIter; ++iter) {
        timer.start(compute_stream);

        for (int step = 0; step < world_size - 1; ++step) {
            __nv_bfloat16* recv = (step & 1) ? d_pong : d_ping;
            cudaEvent_t*   ready = (step & 1) ? &pong_ready : &ping_ready;

            // Transfer: simulate async copy (same device in this sim)
            cudaMemcpyAsync(recv, d_accum,
                            chunk_elems * sizeof(__nv_bfloat16),
                            cudaMemcpyDeviceToDevice, transfer_stream);
            cudaEventRecord(*ready, transfer_stream);

            // Compute: wait for transfer to finish, then reduce
            cudaStreamWaitEvent(compute_stream, *ready, 0);
            launch_pcie_ring_reduce_step(d_accum, recv, chunk_elems,
                                         sm_version, compute_stream);
        }

        total_ms += timer.stop_ms(compute_stream);
    }

    cudaEventDestroy(ping_ready);
    cudaEventDestroy(pong_ready);
    cudaEventDestroy(reduce_done);

    return total_ms / kIter;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main benchmark
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    int   sm_version  = 90;
    float bw_override = 0.f;  // 0 = probe actual BW

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--sm")  == 0 && i + 1 < argc) sm_version  = atoi(argv[++i]);
        if (strcmp(argv[i], "--bw")  == 0 && i + 1 < argc) bw_override = atof(argv[++i]);
    }

    cudaStream_t stream, xfer_stream;
    cudaStreamCreate(&stream);
    cudaStreamCreate(&xfer_stream);

    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    printf("Device: %s  SM: %d.%d\n", prop.name, prop.major, prop.minor);
    printf("SM dispatch path: SM%d\n\n", sm_version);

    // Correctness
    bool ok1 = check_ring_reduce(sm_version, stream);
    bool ok2 = check_finalise(sm_version, stream);
    printf("ring_reduce correctness: %s\n", ok1 ? "PASS" : "FAIL");
    printf("finalise   correctness: %s\n\n", ok2 ? "PASS" : "FAIL");
    if (!ok1 || !ok2) return 1;

    // ─── Adaptive chunk size table ───────────────────────────────────────
    printf("─── Adaptive chunk sizes for simulated PCIe bandwidths ─────\n");
    const float bws[] = {4.f, 8.f, 10.f, 16.f, 32.f};
    printf("%-12s  %-14s  %-10s\n", "BW (GB/s)", "chunk_bytes", "chunk_elems");
    for (float bw : bws) {
        size_t chunk_bytes = compute_pcie_bucket_size(bw);
        size_t chunk_elems = chunk_bytes / sizeof(__nv_bfloat16);
        printf("%-12.1f  %-14zu  %-10zu  (%.1f MB)\n",
               bw, chunk_bytes, chunk_elems, chunk_bytes / 1024.f / 1024.f);
    }
    printf("\n");

    // Use override BW or default
    float bw_gbps = (bw_override > 0.f) ? bw_override : 10.f;
    size_t chunk_bytes = compute_pcie_bucket_size(bw_gbps);
    size_t chunk_elems = chunk_bytes / sizeof(__nv_bfloat16);
    printf("Using %.1f GB/s → chunk_elems = %zu (%.1f MB BF16)\n\n",
           bw_gbps, chunk_elems, chunk_bytes / 1024.f / 1024.f);

    // ─── Ring-reduce kernel throughput sweep ─────────────────────────────
    printf("─── pcie_ring_reduce_kernel throughput ──────────────────────\n");
    printf("%-12s  %-10s  %-10s\n", "n_elems", "avg_us", "GB/s_eff");
    const size_t sizes[] = {
        1UL << 17,   // 128K
        1UL << 20,   //   1M
        1UL << 22,   //   4M
        1UL << 24,   //  16M
        1UL << 26,   //  64M
    };
    for (size_t n : sizes) {
        __nv_bfloat16 *d_dst, *d_src;
        cudaMalloc(&d_dst, n * 2);
        cudaMalloc(&d_src, n * 2);
        cudaMemset(d_dst, 0x3c, n * 2);
        cudaMemset(d_src, 0x3c, n * 2);

        GpuTimer timer;
        for (int w = 0; w < kWarmup; ++w)
            launch_pcie_ring_reduce(d_dst, d_src, n, sm_version, stream);
        cudaStreamSynchronize(stream);

        float total = 0.f;
        for (int it = 0; it < kIter; ++it) {
            timer.start(stream);
            launch_pcie_ring_reduce(d_dst, d_src, n, sm_version, stream);
            total += timer.stop_ms(stream);
        }
        float avg_ms = total / kIter;
        float gbs = (n * 2.f * 2.f) / (avg_ms * 1e-3f) / 1e9f;  // 1 read + 1 write dst, 1 read src
        printf("%-12zu  %8.2f us  %8.2f GB/s\n", n, avg_ms * 1000.f, gbs);

        cudaFree(d_dst); cudaFree(d_src);
    }

    // ─── Double-buffered ring allreduce simulation ────────────────────────
    printf("\n─── Double-buffered ring allreduce (simulated, world_size=5) ─\n");
    const int world_sizes[] = {2, 4, 5, 8};
    printf("%-12s  %-8s  %-12s  %-12s  %-10s\n",
           "chunk_elems", "ws", "total_us", "per_step_us", "overlap_eff");

    for (size_t ce : {chunk_elems, chunk_elems / 2, chunk_elems * 2}) {
        if (ce == 0 || ce > 128UL * 1024 * 1024) continue;
        __nv_bfloat16 *d_accum, *d_ping, *d_pong;
        cudaMalloc(&d_accum, ce * 2);
        cudaMalloc(&d_ping,  ce * 2);
        cudaMalloc(&d_pong,  ce * 2);
        cudaMemset(d_accum, 0x3c, ce * 2);

        for (int ws : world_sizes) {
            float avg_ms = bench_ring_allreduce_sim(
                d_accum, d_ping, d_pong, ce, ws,
                sm_version, stream, xfer_stream);
            float step_ms = avg_ms / (ws - 1);
            // Theoretical transfer time per step at bw_gbps
            float xfer_ms = (ce * 2.f) / (bw_gbps * 1e9f) * 1e3f;
            float overlap = std::min(1.f, step_ms / xfer_ms);
            printf("%-12zu  %-8d  %10.2f us  %10.2f us  %8.1f%%\n",
                   ce, ws, avg_ms * 1000.f, step_ms * 1000.f, overlap * 100.f);
        }
    }

    // ─── Gradient packing throughput ─────────────────────────────────────
    printf("\n─── Gradient packing (gather) throughput ────────────────────\n");
    printf("%-12s  %-4s  %-10s\n", "bucket_elems", "nC", "GB/s");
    for (size_t n : {size_t(1) << 20, size_t(1) << 22, size_t(1) << 24}) {
        for (int nc : {1, 4, 8}) {
            __nv_bfloat16 *d_bucket, *d_src;
            cudaMalloc(&d_bucket, n * 2);
            cudaMalloc(&d_src,    n * 2);
            cudaMemset(d_src, 0x3c, n * 2);

            // Build chunks: split n into nc equal contiguous pieces
            std::vector<PcieGradChunk> chunks(nc);
            size_t per = n / nc;
            for (int c = 0; c < nc; ++c) {
                chunks[c].src    = d_src;
                chunks[c].offset = c * per;
                chunks[c].length = (c == nc-1) ? (n - c*per) : per;
            }

            GpuTimer timer;
            for (int w = 0; w < kWarmup; ++w)
                launch_pcie_gradient_pack(d_bucket, chunks.data(), nc, n, sm_version, stream);
            cudaStreamSynchronize(stream);

            float total = 0.f;
            for (int it = 0; it < kIter; ++it) {
                timer.start(stream);
                launch_pcie_gradient_pack(d_bucket, chunks.data(), nc, n, sm_version, stream);
                total += timer.stop_ms(stream);
            }
            float avg_ms = total / kIter;
            float gbs = (n * 2.f * 2.f) / (avg_ms * 1e-3f) / 1e9f;
            printf("%-12zu  %-4d  %8.2f GB/s  (%6.2f us)\n", n, nc, gbs, avg_ms * 1000.f);

            cudaFree(d_bucket); cudaFree(d_src);
        }
    }

    cudaStreamDestroy(stream);
    cudaStreamDestroy(xfer_stream);
    printf("\nBenchmark complete.\n");
    return 0;
}
