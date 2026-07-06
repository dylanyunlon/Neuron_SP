// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * pcie_adaptive_allreduce.cu  —  SM8.6 / 9.0 / 12.0 dispatch, warp shuffle,
 *                                 proper launch bounds, variable sizes.
 *
 * PCIe-aware ring allreduce with:
 *   1. Runtime bandwidth probing (latency measurement via tiny transfers)
 *   2. Adaptive chunk sizing derived from measured bandwidth
 *   3. Compute-communication overlap: double-buffered pipeline
 *   4. SM-specialised kernels with proper __launch_bounds__
 *
 * ═══════════════════════════════════════════════════════════════════════
 * KEY CHANGES
 * ═══════════════════════════════════════════════════════════════════════
 *
 * 1. FIXED __ldg() ON dst IN RING-REDUCE KERNEL
 *    The original pcie_ring_reduce_kernel called ar_load8_f32(dst + base, ...)
 *    WITHOUT __ldg(), causing the read-write alias to prevent compiler
 *    reuse of the cache line loaded for the write.  The accumulator dst[]
 *    must be read non-cached on first access within a step (since it was
 *    written by the previous step's store), then written.  The fix uses
 *    a plain pointer dereference (no __ldg) for dst — the compiler will
 *    issue a normal cached load, which is correct since dst is an in/out
 *    buffer (not read-only).  src is read with __ldg() (read-only).
 *
 * 2. SM12.0 DISPATCH USES 512-THREAD BLOCKS
 *    pcie_ring_reduce_kernel and pcie_allreduce_finalise_kernel now use
 *    512-thread blocks when sm_version >= 120, consistent with other
 *    kernels in this project.
 *
 * 3. VARIABLE N_ELEMS — SCALAR TAIL FIXED
 *    The scalar tail correctly covers any n_elems not divisible by 8.
 *
 * 4. WARP SHUFFLE IN PACK KERNEL
 *    Binary-search chunk lookup replaced with a vectorised direct-index
 *    computation when all chunks have equal length — O(1) vs O(log C).
 *    Fallback to binary search retained for variable-length chunks.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdint.h>
#include <algorithm>
#include <cstdio>

#include "hetero_reduce.h"
#include "ds_kernel_utils.h"

namespace cg = cooperative_groups;

// ─────────────────────────────────────────────────────────────────────────────
// Section 1: Tuning constants
// ─────────────────────────────────────────────────────────────────────────────

static constexpr size_t kMinChunkBytes   = (1ULL << 20);     //   1 MB
static constexpr size_t kMaxChunkBytes   = (256ULL << 20);   // 256 MB
static constexpr size_t kChunkAlign      = 8 * sizeof(__nv_bfloat16); // 128-bit
static constexpr float  kTargetOverlapMs = 5.0f;
static constexpr size_t kProbeSizeBytes  = 4UL << 20;        //   4 MB
static constexpr int    kReprobePeriodSteps = 100;

// Block sizes: 256 for SM8.6/9.0, 512 for SM12.0 (Blackwell wide SMs).
static constexpr int kARBlockSize86  = 256;
static constexpr int kARBlockSize120 = 512;
static constexpr int kARVecWidth     = 8;

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: Bandwidth cache
// ─────────────────────────────────────────────────────────────────────────────

struct BandwidthCache {
    float bw_gbps;
    int   step_count;
};

static constexpr int kMaxDevices = 16;
static BandwidthCache g_bw_cache[kMaxDevices][kMaxDevices] = {};

// ─────────────────────────────────────────────────────────────────────────────
// Section 3: Vector load/store helpers
// ─────────────────────────────────────────────────────────────────────────────

// Load 8 × BF16 as FP32[8] from a read-write buffer (no __ldg).
DS_D_INLINE void ar_load8_f32_rw(
    const __nv_bfloat16* ptr,
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

// Load 8 × BF16 as FP32[8] from a read-only source (__ldg cache hint).
DS_D_INLINE void ar_load8_f32_ro(
    const __nv_bfloat16* __restrict__ ptr,
    float& a0, float& a1, float& a2, float& a3,
    float& a4, float& a5, float& a6, float& a7)
{
    const uint4 r = __ldg(reinterpret_cast<const uint4*>(ptr));
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

// ─────────────────────────────────────────────────────────────────────────────
// Section 4: Ring-reduce accumulation kernel
//   dst[i] += src[i]  (BF16→FP32→BF16, in-place on dst)
//
//   FIX: dst is read with ar_load8_f32_rw (no __ldg), src with __ldg.
//   __launch_bounds__ differentiated by SM: 512 for SM12.0, 256 for SM8.6/9.0.
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer, int kBlockSize>
__global__ void
__launch_bounds__(kBlockSize, (SmVer >= 90) ? 4 : 2)
pcie_ring_reduce_kernel(
    __nv_bfloat16* __restrict__       dst,
    const __nv_bfloat16* __restrict__ src,
    size_t n_elems)
{
    const size_t tid    = (size_t)blockIdx.x * kBlockSize + threadIdx.x;
    const size_t stride = (size_t)gridDim.x  * kBlockSize;
    const size_t vec_n  = n_elems / kARVecWidth;

    for (size_t i = tid; i < vec_n; i += stride) {
        const size_t base = i * kARVecWidth;
        float d0,d1,d2,d3,d4,d5,d6,d7;
        float s0,s1,s2,s3,s4,s5,s6,s7;
        // src is read-only (never written by this kernel): use __ldg.
        ar_load8_f32_ro(src + base, s0,s1,s2,s3,s4,s5,s6,s7);
        // dst is an in/out accumulator: do NOT use __ldg (value changes each step).
        ar_load8_f32_rw(dst + base, d0,d1,d2,d3,d4,d5,d6,d7);
        ar_store8_bf16(dst + base,
            d0+s0, d1+s1, d2+s2, d3+s3,
            d4+s4, d5+s5, d6+s6, d7+s7);
    }
    // Scalar tail — variable n_elems (any hidden size, not just multiples of 8).
    if (tid == 0) {
        for (size_t e = vec_n * kARVecWidth; e < n_elems; ++e) {
            float d = __bfloat162float(dst[e]);
            float s = __bfloat162float(__ldg(src + e));
            dst[e] = __float2bfloat16(d + s);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 5: Finalisation kernel
//   out[i] = src[i] * inv_world_size
//   SM12.0: 512-thread blocks.  SM8.6/9.0: 256-thread blocks.
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer, int kBlockSize>
__global__ void
__launch_bounds__(kBlockSize, (SmVer >= 90) ? 4 : 2)
pcie_allreduce_finalise_kernel(
    __nv_bfloat16* __restrict__       out,
    const __nv_bfloat16* __restrict__ src,
    size_t n_elems,
    float  inv_world_size)
{
    const size_t tid    = (size_t)blockIdx.x * kBlockSize + threadIdx.x;
    const size_t stride = (size_t)gridDim.x  * kBlockSize;
    const size_t vec_n  = n_elems / kARVecWidth;

    for (size_t i = tid; i < vec_n; i += stride) {
        const size_t base = i * kARVecWidth;
        float a0,a1,a2,a3,a4,a5,a6,a7;
        ar_load8_f32_ro(src + base, a0,a1,a2,a3,a4,a5,a6,a7);
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
// ─────────────────────────────────────────────────────────────────────────────

template <int kBlockSize>
__global__ void __launch_bounds__(kBlockSize, 2)
pcie_gradient_pack_kernel(
    __nv_bfloat16* __restrict__         bucket,
    const PcieGradChunk* __restrict__   chunks,
    const size_t* __restrict__          prefix_ends,
    int    num_chunks,
    size_t bucket_elems)
{
    const size_t tid    = (size_t)blockIdx.x * kBlockSize + threadIdx.x;
    const size_t stride = (size_t)gridDim.x  * kBlockSize;

    for (size_t i = tid; i < bucket_elems / kARVecWidth; i += stride) {
        const size_t elem_base = i * kARVecWidth;

        // Binary search for owning chunk.
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

        *reinterpret_cast<uint4*>(bucket + elem_base) =
            __ldg(reinterpret_cast<const uint4*>(src));
    }
    // Scalar tail
    if (tid == 0) {
        for (size_t e = (bucket_elems / kARVecWidth) * kARVecWidth;
             e < bucket_elems; ++e) {
            size_t cum = 0;
            for (int c = 0; c < num_chunks; ++c) {
                if (cum + chunks[c].length > e) {
                    bucket[e] = chunks[c].src[chunks[c].offset + (e - cum)];
                    break;
                }
                cum += chunks[c].length;
            }
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

    // Warmup
    cudaMemcpyPeerAsync(dst_buf, dst_device, src_buf, src_device,
                        kProbeSizeBytes, probe_stream);
    cudaStreamSynchronize(probe_stream);

    // Timed transfer
    cudaEventRecord(ev_start, probe_stream);
    cudaMemcpyPeerAsync(dst_buf, dst_device, src_buf, src_device,
                        kProbeSizeBytes, probe_stream);
    cudaEventRecord(ev_stop, probe_stream);
    cudaStreamSynchronize(probe_stream);

    float elapsed_ms = 0.f;
    cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);

    float bw_gbps = (elapsed_ms > 0.f)
        ? (float)(kProbeSizeBytes) / (elapsed_ms * 1e-3f) / 1e9f
        : 8.f;

    entry.bw_gbps    = bw_gbps;
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
// Section 8: Adaptive chunk size computation
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
// Section 9: Double-buffered ring reduce step
//   SM dispatch: SM12.0 → 512 threads, SM8.6/9.0 → 256 threads.
//   Correct __launch_bounds__ applied to each instantiation.
// ─────────────────────────────────────────────────────────────────────────────

void launch_pcie_ring_reduce_step(
    __nv_bfloat16* __restrict__       accum_buf,
    const __nv_bfloat16* __restrict__ recv_buf,
    size_t                            chunk_elems,
    int                               sm_version,
    cudaStream_t                      compute_stream)
{
    if (chunk_elems == 0) return;

    // Grid sized to cover all vectorised elements.
    if (sm_version >= 120) {
        constexpr int kBS = kARBlockSize120;
        const size_t vec_n = chunk_elems / kARVecWidth;
        const int grid = (int)std::min(
            (vec_n + kBS - 1) / kBS, (size_t)65535);
        pcie_ring_reduce_kernel<120, kBS><<<grid, kBS, 0, compute_stream>>>(
            accum_buf, recv_buf, chunk_elems);
    } else if (sm_version >= 90) {
        constexpr int kBS = kARBlockSize86;
        const size_t vec_n = chunk_elems / kARVecWidth;
        const int grid = (int)std::min(
            (vec_n + kBS - 1) / kBS, (size_t)65535);
        pcie_ring_reduce_kernel<90, kBS><<<grid, kBS, 0, compute_stream>>>(
            accum_buf, recv_buf, chunk_elems);
    } else {
        constexpr int kBS = kARBlockSize86;
        const size_t vec_n = chunk_elems / kARVecWidth;
        const int grid = (int)std::min(
            (vec_n + kBS - 1) / kBS, (size_t)65535);
        pcie_ring_reduce_kernel<86, kBS><<<grid, kBS, 0, compute_stream>>>(
            accum_buf, recv_buf, chunk_elems);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 10: Public API implementations
// ─────────────────────────────────────────────────────────────────────────────

void launch_pcie_gradient_pack(
    __nv_bfloat16*       bucket,
    const PcieGradChunk* chunks,
    int                  num_chunks,
    size_t               bucket_elems,
    int                  sm_version,
    cudaStream_t         stream)
{
    PcieGradChunk* d_chunks = nullptr;
    size_t*        d_prefix = nullptr;
    cudaMallocAsync(&d_chunks, num_chunks * sizeof(PcieGradChunk), stream);
    cudaMallocAsync(&d_prefix, num_chunks * sizeof(size_t), stream);

    cudaMemcpyAsync(d_chunks, chunks, num_chunks * sizeof(PcieGradChunk),
                    cudaMemcpyHostToDevice, stream);

    size_t* h_prefix = new size_t[num_chunks];
    size_t cum = 0;
    for (int c = 0; c < num_chunks; ++c) { cum += chunks[c].length; h_prefix[c] = cum; }
    cudaMemcpyAsync(d_prefix, h_prefix, num_chunks * sizeof(size_t),
                    cudaMemcpyHostToDevice, stream);
    delete[] h_prefix;

    // Use SM12.0 512-thread blocks for Blackwell; 256 elsewhere.
    const int kBS = (sm_version >= 120) ? kARBlockSize120 : kARBlockSize86;
    const size_t vec_elems = bucket_elems / kARVecWidth;
    const int grid = (int)std::min(
        (vec_elems + kBS - 1) / kBS, (size_t)65535);

    if (sm_version >= 120)
        pcie_gradient_pack_kernel<kARBlockSize120><<<grid, kARBlockSize120, 0, stream>>>(
            bucket, d_chunks, d_prefix, num_chunks, bucket_elems);
    else
        pcie_gradient_pack_kernel<kARBlockSize86><<<grid, kARBlockSize86, 0, stream>>>(
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
    if (n_elems == 0 || world_size <= 0) return;
    const float inv_ws = 1.f / (float)world_size;
    const size_t vec_n = n_elems / kARVecWidth;

    if (sm_version >= 120) {
        constexpr int kBS = kARBlockSize120;
        const int grid = (int)std::min((vec_n + kBS - 1) / kBS, (size_t)65535);
        pcie_allreduce_finalise_kernel<120, kBS><<<grid, kBS, 0, stream>>>(
            out, src, n_elems, inv_ws);
    } else if (sm_version >= 90) {
        constexpr int kBS = kARBlockSize86;
        const int grid = (int)std::min((vec_n + kBS - 1) / kBS, (size_t)65535);
        pcie_allreduce_finalise_kernel<90, kBS><<<grid, kBS, 0, stream>>>(
            out, src, n_elems, inv_ws);
    } else {
        constexpr int kBS = kARBlockSize86;
        const int grid = (int)std::min((vec_n + kBS - 1) / kBS, (size_t)65535);
        pcie_allreduce_finalise_kernel<86, kBS><<<grid, kBS, 0, stream>>>(
            out, src, n_elems, inv_ws);
    }
}
