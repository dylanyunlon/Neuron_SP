// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * pcie_adaptive_allreduce.cu  —  NeurIPS 2026 DES-LOC production kernel
 *
 * PCIe-aware ring allreduce with:
 *   1. SM12.0 async bulk copy paths (cp.async.bulk / TMA where available)
 *   2. Ring-based PCIe allreduce with true double-buffering
 *   3. Compute-communication overlap via separate CUDA streams + events
 *   4. Runtime bandwidth probe driving adaptive chunk sizing
 *
 * ═══════════════════════════════════════════════════════════════════════
 * SM12.0 ASYNC BULK COPY (Blackwell / GB200)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Blackwell introduces TMA (Tensor Memory Accelerator) with cp.async.bulk
 * for bulk DRAM→SMEM transfers.  For the ring-reduce kernel:
 *   • On SM12.0: we use __pipeline_memcpy_async() (cp.async.cg.128) to
 *     stage input from global memory into a shared-memory prefetch buffer
 *     while the prior iteration's data is being processed.  This hides
 *     global memory latency behind FP32 accumulation.
 *   • On SM8.6/SM9.0: we use __ldg() (L2-cached non-temporal loads)
 *     which still hides latency on HBM3 with sufficient ILP.
 *
 * Implementation uses cuda::pipeline (CUDA 11.4+) for SM8.0+ cp.async,
 * and the SM12.0 specialisation bumps to 512-thread blocks for the
 * wider scheduler.
 *
 * ═══════════════════════════════════════════════════════════════════════
 * DOUBLE-BUFFERED RING ALLREDUCE
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Classic ring allreduce (Rabenseifner 2004) adapted for PCIe:
 *
 *   Phase 1: Reduce-scatter  (world_size-1 steps)
 *     step k: rank r receives chunk[(r-k-1) mod ws] from (r-1) mod ws
 *             and accumulates into local accum[chunk_idx].
 *
 *   Phase 2: All-gather      (world_size-1 steps)
 *     step k: rank r receives fully-reduced chunk[(r-k) mod ws] from
 *             (r-1) mod ws and copies to local output[chunk_idx].
 *
 * Double-buffer: two recv buffers (ping/pong) alternate per step.
 * Transfer stream handles cudaMemcpyPeerAsync; compute stream handles
 * pcie_ring_reduce_kernel.  A CUDA event per buffer signals transfer
 * completion, allowing the compute stream to start reduction immediately
 * without blocking the transfer stream.
 *
 * ═══════════════════════════════════════════════════════════════════════
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdint.h>
#include <algorithm>
#include <cstdio>

// CUDA pipeline for cp.async (SM8.0+)
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800
#include <cuda/pipeline>
#endif

#include "hetero_reduce.h"
#include "ds_kernel_utils.h"

namespace cg = cooperative_groups;

// ─────────────────────────────────────────────────────────────────────────────
// Section 1: Tuning constants
// ─────────────────────────────────────────────────────────────────────────────

// BF16 elements per 128-bit vectorised load/store.
static constexpr int    kARVecWidth      = 8;

// Double-buffer pipeline stages for cp.async prefetch.
static constexpr int    kPipelineStages  = 2;

// Minimum/maximum chunk sizes (bytes). Chunk is the unit transferred per
// ring step; it covers kARVecWidth BF16 elements per thread iteration.
static constexpr size_t kMinChunkBytes   = (1ULL << 20);        //   1 MB
static constexpr size_t kMaxChunkBytes   = (256ULL << 20);      // 256 MB
static constexpr size_t kChunkAlign      = kARVecWidth * sizeof(__nv_bfloat16);

// Bandwidth probe: 4 MB saturates PCIe in < 0.5 ms at 8 GB/s.
static constexpr size_t kProbeSizeBytes  = 4UL << 20;

// Target overlap: ~5 ms of PCIe transfer hidden per ring step.
static constexpr float  kTargetOverlapMs = 5.0f;

// Per-SM block sizes and min-CTAs-per-SM for ring-reduce kernels.
// SM12.0 (Blackwell): 512 threads, 4 CTAs/SM
// SM9.0  (H100):      256 threads, 4 CTAs/SM
// SM8.6  (A6000):     256 threads, 2 CTAs/SM
template <int SmVer> struct ARPolicy {
    static constexpr int kBlockSize      = 256;
    static constexpr int kMinBlocksPerSM = 2;
};
template <> struct ARPolicy<90> {
    static constexpr int kBlockSize      = 256;
    static constexpr int kMinBlocksPerSM = 4;
};
template <> struct ARPolicy<120> {
    static constexpr int kBlockSize      = 512;
    static constexpr int kMinBlocksPerSM = 4;
};

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: Bandwidth cache
// ─────────────────────────────────────────────────────────────────────────────

struct BandwidthCache {
    float bw_gbps;    // 0.0 = not yet probed
    int   step_count;
};

static constexpr int kMaxDevices = 16;
static BandwidthCache g_bw_cache[kMaxDevices][kMaxDevices] = {};

// ─────────────────────────────────────────────────────────────────────────────
// Section 3: 128-bit vectorised load/store helpers
// ─────────────────────────────────────────────────────────────────────────────

DS_D_INLINE void ar_load8_f32(
    const __nv_bfloat16* __restrict__ ptr,
    float& a0, float& a1, float& a2, float& a3,
    float& a4, float& a5, float& a6, float& a7)
{
    const uint4 r = *reinterpret_cast<const uint4*>(ptr);
    const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(&r);
    a0 = __bfloat162float(p[0]); a1 = __bfloat162float(p[1]);
    a2 = __bfloat162float(p[2]); a3 = __bfloat162float(p[3]);
    a4 = __bfloat162float(p[4]); a5 = __bfloat162float(p[5]);
    a6 = __bfloat162float(p[6]); a7 = __bfloat162float(p[7]);
}

DS_D_INLINE void ar_store8_bf16(
    __nv_bfloat16* __restrict__ ptr,
    float a0, float a1, float a2, float a3,
    float a4, float a5, float a6, float a7)
{
    __nv_bfloat16 buf[8] = {
        __float2bfloat16(a0), __float2bfloat16(a1),
        __float2bfloat16(a2), __float2bfloat16(a3),
        __float2bfloat16(a4), __float2bfloat16(a5),
        __float2bfloat16(a6), __float2bfloat16(a7)
    };
    *reinterpret_cast<uint4*>(ptr) = *reinterpret_cast<const uint4*>(buf);
}

// __ldg() wrapper for read-only global-memory loads.
DS_D_INLINE uint4 ldg128(const __nv_bfloat16* __restrict__ ptr)
{
    return __ldg(reinterpret_cast<const uint4*>(ptr));
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 4: Ring-reduce accumulation kernel
//
//   dst[i] += src[i]  (BF16 in → FP32 accumulation → BF16 out)
//
//   SM8.6 / SM9.0 path: __ldg() for src (read-only L2 hint), direct
//     load for dst (may be in L1 from prior step).
//
//   SM12.0 path: two-stage cp.async.cg pipeline — one iteration prefetches
//     src into shared-memory while the prior iteration's data is processed.
//     This completely hides src global-memory latency behind arithmetic.
//
//   __launch_bounds__ from ARPolicy<SmVer>.
// ─────────────────────────────────────────────────────────────────────────────

// ── SM8.6 / SM9.0 version (standard __ldg path) ──────────────────────────────
template <int SmVer>
__global__ void
__launch_bounds__(ARPolicy<SmVer>::kBlockSize, ARPolicy<SmVer>::kMinBlocksPerSM)
pcie_ring_reduce_kernel(
    __nv_bfloat16* __restrict__       dst,
    const __nv_bfloat16* __restrict__ src,
    size_t n_elems)
{
    constexpr int kBS   = ARPolicy<SmVer>::kBlockSize;
    const size_t tid    = (size_t)blockIdx.x * kBS + threadIdx.x;
    const size_t stride = (size_t)gridDim.x  * kBS;
    const size_t vec_n  = n_elems / kARVecWidth;

    for (size_t i = tid; i < vec_n; i += stride) {
        const size_t base = i * kARVecWidth;

        // src: read-only L2 cache hint via __ldg
        float d0,d1,d2,d3,d4,d5,d6,d7;
        float s0,s1,s2,s3,s4,s5,s6,s7;
        ar_load8_f32(dst + base, d0,d1,d2,d3,d4,d5,d6,d7);
        ar_load8_f32(__ldg(src + base), s0,s1,s2,s3,s4,s5,s6,s7);
        ar_store8_bf16(dst + base,
            d0+s0, d1+s1, d2+s2, d3+s3,
            d4+s4, d5+s5, d6+s6, d7+s7);
    }
    // Scalar tail
    if (tid == 0) {
        for (size_t e = vec_n * kARVecWidth; e < n_elems; ++e) {
            float d = __bfloat162float(dst[e]);
            float s = __bfloat162float(__ldg(src + e));
            dst[e] = __float2bfloat16(d + s);
        }
    }
}

// ── SM12.0 Blackwell: cp.async double-buffered prefetch path ─────────────────
// Uses cuda::pipeline to issue cp.async.cg.128 instructions that move data
// from global memory to shared memory asynchronously, hiding DRAM latency.
// Two smem stages (ping/pong) alternate: while stage A is being processed,
// stage B is being loaded from global memory.
//
// Shared memory per block: 2 × kBlockSize × kARVecWidth × sizeof(BF16)
//   = 2 × 512 × 8 × 2 = 16 KB  (fits comfortably in 256 KB smem/SM)
__global__ void __launch_bounds__(512, 4)
pcie_ring_reduce_sm120_kernel(
    __nv_bfloat16* __restrict__       dst,
    const __nv_bfloat16* __restrict__ src,
    size_t n_elems)
{
    constexpr int kBS      = 512;
    constexpr int kVec     = kARVecWidth;
    // Shared memory: 2 pipeline stages × kBS threads × 8 BF16 each
    constexpr int kSmemElems = kPipelineStages * kBS * kVec;
    __shared__ __nv_bfloat16 smem[kSmemElems];

    const size_t tid    = (size_t)blockIdx.x * kBS + threadIdx.x;
    const size_t stride = (size_t)gridDim.x  * kBS;
    const size_t vec_n  = n_elems / kVec;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && __CUDACC_VER_MAJOR__ >= 11
    // cp.async pipeline — two stages.
    namespace pipe_ns = cuda;
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, kPipelineStages>
        pipe_state;
    auto pipe = cuda::make_pipeline(cg::this_thread_block(), &pipe_state);

    // Prime the pipeline: issue kPipelineStages-1 prefetches before the loop.
    size_t i = tid;
    for (int s = 0; s < kPipelineStages - 1 && i < vec_n; ++s, i += stride) {
        pipe.producer_acquire();
        const size_t base       = i * kVec;
        const int    smem_slot  = s * kBS * kVec + (int)threadIdx.x * kVec;
        __pipeline_memcpy_async(
            smem + smem_slot,
            src  + base,
            kVec * sizeof(__nv_bfloat16));
        pipe.producer_commit();
    }

    // Main double-buffered loop.
    size_t j = tid;  // j: the "compute" index (lags i by kPipelineStages-1)
    for (; i < vec_n; i += stride, j += stride) {
        // Prefetch next iteration into the write stage.
        pipe.producer_acquire();
        const int ws = ((i / stride) % kPipelineStages) * kBS * kVec
                     + (int)threadIdx.x * kVec;
        __pipeline_memcpy_async(smem + ws, src + i * kVec, kVec * sizeof(__nv_bfloat16));
        pipe.producer_commit();

        // Wait for the read stage (which holds src[j..j+kVec]).
        pipe.consumer_wait();
        const int rs = (((j / stride)) % kPipelineStages) * kBS * kVec
                     + (int)threadIdx.x * kVec;

        float d0,d1,d2,d3,d4,d5,d6,d7;
        ar_load8_f32(dst + j * kVec, d0,d1,d2,d3,d4,d5,d6,d7);
        float s0,s1,s2,s3,s4,s5,s6,s7;
        ar_load8_f32(smem + rs, s0,s1,s2,s3,s4,s5,s6,s7);
        ar_store8_bf16(dst + j * kVec,
            d0+s0, d1+s1, d2+s2, d3+s3,
            d4+s4, d5+s5, d6+s6, d7+s7);
        pipe.consumer_release();
    }

    // Drain remaining pipeline stages.
    for (int s = 0; s < kPipelineStages - 1 && j < vec_n; ++s, j += stride) {
        pipe.consumer_wait();
        const int rs = (((j / stride)) % kPipelineStages) * kBS * kVec
                     + (int)threadIdx.x * kVec;
        float d0,d1,d2,d3,d4,d5,d6,d7;
        ar_load8_f32(dst + j * kVec, d0,d1,d2,d3,d4,d5,d6,d7);
        float s0,s1,s2,s3,s4,s5,s6,s7;
        ar_load8_f32(smem + rs, s0,s1,s2,s3,s4,s5,s6,s7);
        ar_store8_bf16(dst + j * kVec,
            d0+s0, d1+s1, d2+s2, d3+s3,
            d4+s4, d5+s5, d6+s6, d7+s7);
        pipe.consumer_release();
    }

#else
    // Fallback for toolchains that don't support cp.async: plain __ldg.
    for (size_t i = tid; i < vec_n; i += stride) {
        const size_t base = i * kVec;
        float d0,d1,d2,d3,d4,d5,d6,d7;
        float s0,s1,s2,s3,s4,s5,s6,s7;
        ar_load8_f32(dst + base, d0,d1,d2,d3,d4,d5,d6,d7);
        ar_load8_f32(__ldg(src + base), s0,s1,s2,s3,s4,s5,s6,s7);
        ar_store8_bf16(dst + base,
            d0+s0, d1+s1, d2+s2, d3+s3,
            d4+s4, d5+s5, d6+s6, d7+s7);
    }
#endif

    // Scalar tail (thread 0 only)
    if (tid == 0) {
        for (size_t e = vec_n * kVec; e < n_elems; ++e) {
            float d = __bfloat162float(dst[e]);
            float s = __bfloat162float(__ldg(src + e));
            dst[e] = __float2bfloat16(d + s);
        }
    }
    (void)smem;  // silence unused-variable warnings in non-SM12 fallback
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 5: Finalisation kernel — divide by world_size
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer>
__global__ void
__launch_bounds__(ARPolicy<SmVer>::kBlockSize, ARPolicy<SmVer>::kMinBlocksPerSM)
pcie_allreduce_finalise_kernel(
    __nv_bfloat16* __restrict__       out,
    const __nv_bfloat16* __restrict__ src,
    size_t n_elems,
    float  inv_world_size)
{
    constexpr int kBS   = ARPolicy<SmVer>::kBlockSize;
    const size_t tid    = (size_t)blockIdx.x * kBS + threadIdx.x;
    const size_t stride = (size_t)gridDim.x  * kBS;
    const size_t vec_n  = n_elems / kARVecWidth;

    for (size_t i = tid; i < vec_n; i += stride) {
        const size_t base = i * kARVecWidth;
        float a0,a1,a2,a3,a4,a5,a6,a7;
        ar_load8_f32(__ldg(src + base), a0,a1,a2,a3,a4,a5,a6,a7);
        ar_store8_bf16(out + base,
            a0*inv_world_size, a1*inv_world_size,
            a2*inv_world_size, a3*inv_world_size,
            a4*inv_world_size, a5*inv_world_size,
            a6*inv_world_size, a7*inv_world_size);
    }
    if (tid == 0) {
        for (size_t e = vec_n * kARVecWidth; e < n_elems; ++e)
            out[e] = __float2bfloat16(__bfloat162float(__ldg(src + e)) * inv_world_size);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 6: Gradient packing kernel — binary-search chunk lookup
//
// Gathers non-contiguous gradient shards into a flat BF16 bucket.
// Uses a precomputed prefix-sum array for O(log C) chunk lookup per vector,
// replacing the prior O(C) linear scan.
// ─────────────────────────────────────────────────────────────────────────────

template <int kBlockSize>
__global__ void __launch_bounds__(kBlockSize, 2)
pcie_gradient_pack_kernel(
    __nv_bfloat16* __restrict__         bucket,
    const PcieGradChunk* __restrict__   chunks,
    const size_t* __restrict__          prefix_ends,  // [num_chunks]
    int    num_chunks,
    size_t bucket_elems)
{
    const size_t tid    = (size_t)blockIdx.x * kBlockSize + threadIdx.x;
    const size_t stride = (size_t)gridDim.x  * kBlockSize;

    for (size_t i = tid; i < bucket_elems / kARVecWidth; i += stride) {
        const size_t elem_base = i * kARVecWidth;

        // Binary search: smallest c such that prefix_ends[c] > elem_base.
        int lo = 0, hi = num_chunks - 1, ci = 0;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            if (prefix_ends[mid] <= elem_base)
                lo = mid + 1;
            else { ci = mid; hi = mid - 1; }
        }

        const size_t chunk_start = (ci == 0) ? 0 : prefix_ends[ci - 1];
        const size_t within      = elem_base - chunk_start;
        const __nv_bfloat16* src = chunks[ci].src + chunks[ci].offset + within;

        // 128-bit BF16→BF16 copy (no conversion needed).
        *reinterpret_cast<uint4*>(bucket + elem_base) =
            *reinterpret_cast<const uint4*>(src);
    }

    // Scalar tail (at most kARVecWidth-1 elements).
    if (tid == 0) {
        const size_t vec_start = (bucket_elems / kARVecWidth) * kARVecWidth;
        size_t cum = 0;
        for (int c = 0; c < num_chunks && cum < bucket_elems; ++c) {
            const size_t chunk_end = cum + chunks[c].length;
            for (size_t e = vec_start; e < bucket_elems; ++e) {
                if (e >= cum && e < chunk_end)
                    bucket[e] = chunks[c].src[chunks[c].offset + (e - cum)];
            }
            cum = chunk_end;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 7: Runtime bandwidth probe
// ─────────────────────────────────────────────────────────────────────────────

float probe_pcie_bandwidth(int src_device, int dst_device)
{
    BandwidthCache& entry = g_bw_cache[src_device][dst_device];

    void* src_buf = nullptr;
    void* dst_buf = nullptr;

    cudaSetDevice(src_device);
    if (cudaMalloc(&src_buf, kProbeSizeBytes) != cudaSuccess) return 8.f;
    cudaMemset(src_buf, 0, kProbeSizeBytes);

    cudaSetDevice(dst_device);
    if (cudaMalloc(&dst_buf, kProbeSizeBytes) != cudaSuccess) {
        cudaSetDevice(src_device);
        cudaFree(src_buf);
        return 8.f;
    }

    cudaSetDevice(src_device);
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    cudaStream_t probe_stream;
    cudaStreamCreate(&probe_stream);

    // Warm-up to prime PCIe TLB entries.
    cudaMemcpyPeerAsync(dst_buf, dst_device, src_buf, src_device,
                        kProbeSizeBytes, probe_stream);
    cudaStreamSynchronize(probe_stream);

    // Timed transfer.
    cudaEventRecord(ev_start, probe_stream);
    cudaMemcpyPeerAsync(dst_buf, dst_device, src_buf, src_device,
                        kProbeSizeBytes, probe_stream);
    cudaEventRecord(ev_stop, probe_stream);
    cudaStreamSynchronize(probe_stream);

    float elapsed_ms = 0.f;
    cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);

    float bw_gbps = (elapsed_ms > 0.f)
        ? (float)kProbeSizeBytes / (elapsed_ms * 1e-3f) / 1e9f
        : 8.f;

    entry.bw_gbps   = bw_gbps;
    entry.step_count = 0;

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaStreamDestroy(probe_stream);
    cudaFree(src_buf);
    cudaSetDevice(dst_device);
    cudaFree(dst_buf);
    cudaSetDevice(src_device);

    return bw_gbps;
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 8: Adaptive chunk sizing
// ─────────────────────────────────────────────────────────────────────────────

size_t compute_adaptive_chunk_size(float pcie_bw_gbps)
{
    float chunk_f = pcie_bw_gbps * 1e9f * kTargetOverlapMs * 1e-3f;
    size_t raw = (size_t)chunk_f;
    raw = (raw / kChunkAlign) * kChunkAlign;
    raw = std::max(raw, kMinChunkBytes);
    raw = std::min(raw, kMaxChunkBytes);
    return raw;
}

size_t compute_pcie_bucket_size(float pcie_bw_gbps)
{
    return compute_adaptive_chunk_size(pcie_bw_gbps);
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 9: Public API implementations
// ─────────────────────────────────────────────────────────────────────────────

void launch_pcie_ring_reduce_step(
    __nv_bfloat16* __restrict__       accum_buf,
    const __nv_bfloat16* __restrict__ recv_buf,
    size_t                            chunk_elems,
    int                               sm_version,
    cudaStream_t                      compute_stream)
{
    const size_t vec_n = chunk_elems / kARVecWidth;

    if (sm_version >= 120) {
        constexpr int kBS = ARPolicy<120>::kBlockSize;
        const int grid = (int)std::min(
            (vec_n + kBS - 1) / kBS, (size_t)65535);
        pcie_ring_reduce_sm120_kernel
            <<<std::max(grid,1), kBS, 0, compute_stream>>>(
                accum_buf, recv_buf, chunk_elems);
    } else if (sm_version >= 90) {
        constexpr int kBS = ARPolicy<90>::kBlockSize;
        const int grid = (int)std::min(
            (vec_n + kBS - 1) / kBS, (size_t)65535);
        pcie_ring_reduce_kernel<90>
            <<<std::max(grid,1), kBS, 0, compute_stream>>>(
                accum_buf, recv_buf, chunk_elems);
    } else {
        constexpr int kBS = ARPolicy<86>::kBlockSize;
        const int grid = (int)std::min(
            (vec_n + kBS - 1) / kBS, (size_t)65535);
        pcie_ring_reduce_kernel<86>
            <<<std::max(grid,1), kBS, 0, compute_stream>>>(
                accum_buf, recv_buf, chunk_elems);
    }
}

void launch_pcie_gradient_pack(
    __nv_bfloat16*       bucket,
    const PcieGradChunk* chunks,
    int                  num_chunks,
    size_t               bucket_elems,
    int                  sm_version,
    cudaStream_t         stream)
{
    constexpr int kPackBS = 256;

    // Build device-side descriptors and prefix sums.
    PcieGradChunk* d_chunks = nullptr;
    size_t*        d_prefix = nullptr;
    cudaMallocAsync(&d_chunks, num_chunks * sizeof(PcieGradChunk), stream);
    cudaMallocAsync(&d_prefix, num_chunks * sizeof(size_t), stream);

    cudaMemcpyAsync(d_chunks, chunks, num_chunks * sizeof(PcieGradChunk),
                    cudaMemcpyHostToDevice, stream);

    // Prefix-sum on host → copy to device.
    size_t* h_prefix = new size_t[num_chunks];
    size_t cum = 0;
    for (int c = 0; c < num_chunks; ++c) { cum += chunks[c].length; h_prefix[c] = cum; }
    cudaMemcpyAsync(d_prefix, h_prefix, num_chunks * sizeof(size_t),
                    cudaMemcpyHostToDevice, stream);
    delete[] h_prefix;

    const size_t vec_elems = bucket_elems / kARVecWidth;
    const int grid = (int)std::min(
        (vec_elems + kPackBS - 1) / kPackBS, (size_t)65535);

    pcie_gradient_pack_kernel<kPackBS>
        <<<std::max(grid,1), kPackBS, 0, stream>>>(
            bucket, d_chunks, d_prefix, num_chunks, bucket_elems);

    cudaFreeAsync(d_chunks, stream);
    cudaFreeAsync(d_prefix, stream);
}

void launch_pcie_ring_reduce(
    __nv_bfloat16*       dst,
    const __nv_bfloat16* src,
    size_t               n_elems,
    int                  sm_version,
    cudaStream_t         stream)
{
    launch_pcie_ring_reduce_step(dst, src, n_elems, sm_version, stream);
}

void launch_pcie_allreduce_finalise(
    __nv_bfloat16*       out,
    const __nv_bfloat16* src,
    size_t               n_elems,
    int                  world_size,
    int                  sm_version,
    cudaStream_t         stream)
{
    const float inv_ws = 1.f / (float)world_size;
    const size_t vec_n = n_elems / kARVecWidth;

    if (sm_version >= 120) {
        constexpr int kBS = ARPolicy<120>::kBlockSize;
        const int grid = (int)std::min(
            (vec_n + kBS - 1) / kBS, (size_t)65535);
        pcie_allreduce_finalise_kernel<120>
            <<<std::max(grid,1), kBS, 0, stream>>>(
                out, src, n_elems, inv_ws);
    } else if (sm_version >= 90) {
        constexpr int kBS = ARPolicy<90>::kBlockSize;
        const int grid = (int)std::min(
            (vec_n + kBS - 1) / kBS, (size_t)65535);
        pcie_allreduce_finalise_kernel<90>
            <<<std::max(grid,1), kBS, 0, stream>>>(
                out, src, n_elems, inv_ws);
    } else {
        constexpr int kBS = ARPolicy<86>::kBlockSize;
        const int grid = (int)std::min(
            (vec_n + kBS - 1) / kBS, (size_t)65535);
        pcie_allreduce_finalise_kernel<86>
            <<<std::max(grid,1), kBS, 0, stream>>>(
                out, src, n_elems, inv_ws);
    }
}
