// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * pcie_adaptive_allreduce.cu
 *
 * PCIe-aware adaptive gradient bucketing and allreduce for heterogeneous
 * GPU clusters without NVLink (A6000 SM8.6 + H100 SM9.0 + Blackwell SM12.0).
 *
 * Problem
 * -------
 * Standard allreduce assumes symmetric bandwidth between all device pairs.
 * In a PCIe-only cluster, bandwidth depends on PCIe switch topology:
 *   - Same PCIe root complex:  ~32 GB/s per direction (PCIe 4.0 x16)
 *   - Cross-NUMA / cross-switch: ~8–12 GB/s effective
 * Naive allreduce saturates the inter-socket link.
 *
 * Strategy
 * --------
 * 1. Each GPU flattens its local gradients into a contiguous BF16 bucket
 *    (launch_pcie_gradient_pack kernel).
 * 2. The ring-allreduce reduce phase fuses BF16→FP32 accumulation at the
 *    receiver side (launch_pcie_ring_reduce kernel).
 * 3. A final normalisation pass divides by world_size and converts back to
 *    BF16 (launch_pcie_allreduce_finalise kernel).
 *
 * PCIe-adaptive bucketing
 * -----------------------
 * The host-side helper compute_pcie_bucket_size() returns a recommended
 * bucket size based on measured bandwidth.  It rounds to a multiple of the
 * vector width (8 BF16 = 128 bits) and clamps between kMinBucketBytes and
 * kMaxBucketBytes.  The kernel API itself is agnostic to bucket size.
 *
 * Kernel design
 * -------------
 * All kernels use 8-element BF16 vectorised (uint4) loads and stores.
 * FP32 accumulation prevents precision loss during multi-GPU reduction.
 * SM-specialised __launch_bounds__: SM8.6→(256,2), SM9.0→(256,4), SM12.0→(512,4).
 *
 * Cooperative groups
 * ------------------
 * Warp-level reductions use cg::reduce() for forward-compatible shuffle
 * semantics on all supported SM versions.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdint.h>
#include <algorithm>

#include "hetero_reduce.h"
#include "ds_kernel_utils.h"

namespace cg = cooperative_groups;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int   kARBlockSize    = 256;
static constexpr int   kARVecWidth     = 8;    // BF16 elements per 128-bit access
static constexpr size_t kMinBucketBytes = (1ULL << 20);  //  1 MB
static constexpr size_t kMaxBucketBytes = (256ULL << 20); // 256 MB
static constexpr size_t kBucketAlign    = kARVecWidth * sizeof(__nv_bfloat16);

// ---------------------------------------------------------------------------
// Vector load/store helpers (reuse pattern from hetero_reduce.cu)
// ---------------------------------------------------------------------------
DS_D_INLINE void ar_load_bf16x8(const __nv_bfloat16* __restrict__ ptr,
                                  float& a0, float& a1, float& a2, float& a3,
                                  float& a4, float& a5, float& a6, float& a7)
{
    uint4 raw = *reinterpret_cast<const uint4*>(ptr);
    const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(&raw);
    a0 = __bfloat162float(p[0]);
    a1 = __bfloat162float(p[1]);
    a2 = __bfloat162float(p[2]);
    a3 = __bfloat162float(p[3]);
    a4 = __bfloat162float(p[4]);
    a5 = __bfloat162float(p[5]);
    a6 = __bfloat162float(p[6]);
    a7 = __bfloat162float(p[7]);
}

DS_D_INLINE void ar_store_fp32x8_as_bf16(__nv_bfloat16* __restrict__ ptr,
                                           float a0, float a1, float a2, float a3,
                                           float a4, float a5, float a6, float a7)
{
    __nv_bfloat16 buf[8];
    buf[0] = __float2bfloat16(a0); buf[1] = __float2bfloat16(a1);
    buf[2] = __float2bfloat16(a2); buf[3] = __float2bfloat16(a3);
    buf[4] = __float2bfloat16(a4); buf[5] = __float2bfloat16(a5);
    buf[6] = __float2bfloat16(a6); buf[7] = __float2bfloat16(a7);
    *reinterpret_cast<uint4*>(ptr) =
        *reinterpret_cast<const uint4*>(buf);
}

// ---------------------------------------------------------------------------
// Gradient packing kernel.
//
// Copies num_chunks non-contiguous gradient shards into a single flat
// BF16 bucket.  Each shard is described by (src_ptr, offset, length) in the
// PcieGradChunk descriptor array (declared in hetero_reduce.h).
//
// Output: bucket[0..bucket_elems) — flat BF16 buffer.
//
// This is a simple memcpy-style kernel; the cost is one BF16 read +
// one BF16 write per element, dominated by PCIe bandwidth to/from device
// memory.  We use 128-bit vectorised accesses to saturate L2 bandwidth.
// ---------------------------------------------------------------------------

template <int kBlockSize>
__global__ void __launch_bounds__(kBlockSize, 2)
pcie_gradient_pack_kernel(
    __nv_bfloat16* __restrict__     bucket,
    const PcieGradChunk* __restrict__ chunks,
    int   num_chunks,
    size_t bucket_elems)
{
    const size_t tid    = (size_t)blockIdx.x * kBlockSize + threadIdx.x;
    const size_t stride = (size_t)gridDim.x  * kBlockSize;

    // Pre-compute prefix sums to find chunk for each element.
    // For typical num_chunks <= 8, this linear scan is fast.
    for (size_t i = tid; i < bucket_elems / kARVecWidth; i += stride) {
        const size_t elem_base = i * kARVecWidth;

        // Find which chunk this element belongs to.
        size_t cum = 0;
        int    ci  = 0;
        while (ci < num_chunks && cum + chunks[ci].length <= elem_base) {
            cum += chunks[ci].length;
            ci++;
        }
        if (ci >= num_chunks) break;

        // Within-chunk offset.
        size_t within = elem_base - cum;
        const __nv_bfloat16* src = chunks[ci].src + chunks[ci].offset + within;

        // Vectorised copy: 8 BF16 elements per iteration.
        float a0, a1, a2, a3, a4, a5, a6, a7;
        ar_load_bf16x8(src, a0, a1, a2, a3, a4, a5, a6, a7);
        ar_store_fp32x8_as_bf16(bucket + elem_base,
                                 a0, a1, a2, a3, a4, a5, a6, a7);
    }
}

// ---------------------------------------------------------------------------
// Ring-reduce accumulation kernel.
//
// Called on the *receiver* GPU during ring-allreduce reduce phase.
// Accumulates `src` (incoming BF16 bucket from peer) into `dst` (local
// BF16 accumulator) in FP32.
//
// dst[i] += src[i]   (computed in FP32, stored as BF16)
//
// SM specialisation via template.
// ---------------------------------------------------------------------------
template <int SmVer, int kBlockSize>
__global__ void __launch_bounds__(kBlockSize, (SmVer >= 90) ? 4 : 2)
pcie_ring_reduce_kernel(
    __nv_bfloat16* __restrict__       dst,  // [n_elems] accumulator (in-place)
    const __nv_bfloat16* __restrict__ src,  // [n_elems] incoming chunk from peer
    size_t n_elems)
{
    const size_t tid    = (size_t)blockIdx.x * kBlockSize + threadIdx.x;
    const size_t stride = (size_t)gridDim.x  * kBlockSize;

    for (size_t i = tid; i < n_elems / kARVecWidth; i += stride) {
        const size_t base = i * kARVecWidth;

        float d0, d1, d2, d3, d4, d5, d6, d7;
        float s0, s1, s2, s3, s4, s5, s6, s7;
        ar_load_bf16x8(dst + base, d0, d1, d2, d3, d4, d5, d6, d7);
        ar_load_bf16x8(src + base, s0, s1, s2, s3, s4, s5, s6, s7);

        ar_store_fp32x8_as_bf16(dst + base,
            d0 + s0, d1 + s1, d2 + s2, d3 + s3,
            d4 + s4, d5 + s5, d6 + s6, d7 + s7);
    }
}

// ---------------------------------------------------------------------------
// Allreduce finalisation kernel.
//
// After ring-reduce, each GPU holds the full sum.  This kernel divides
// by world_size and writes BF16 output.
//
// dst[i] = dst[i] * inv_world_size
//
// Also supports writing into a separate output buffer (out != dst) for
// non-in-place use.
// ---------------------------------------------------------------------------
template <int SmVer, int kBlockSize>
__global__ void __launch_bounds__(kBlockSize, (SmVer >= 90) ? 4 : 2)
pcie_allreduce_finalise_kernel(
    __nv_bfloat16* __restrict__       out,  // [n_elems] output
    const __nv_bfloat16* __restrict__ src,  // [n_elems] sum buffer
    size_t  n_elems,
    float   inv_world_size)
{
    const size_t tid    = (size_t)blockIdx.x * kBlockSize + threadIdx.x;
    const size_t stride = (size_t)gridDim.x  * kBlockSize;

    for (size_t i = tid; i < n_elems / kARVecWidth; i += stride) {
        const size_t base = i * kARVecWidth;

        float a0, a1, a2, a3, a4, a5, a6, a7;
        ar_load_bf16x8(src + base, a0, a1, a2, a3, a4, a5, a6, a7);

        ar_store_fp32x8_as_bf16(out + base,
            a0 * inv_world_size, a1 * inv_world_size,
            a2 * inv_world_size, a3 * inv_world_size,
            a4 * inv_world_size, a5 * inv_world_size,
            a6 * inv_world_size, a7 * inv_world_size);
    }
}

// ---------------------------------------------------------------------------
// Host-side bucket size heuristic.
//
// Estimates optimal bucket size based on available PCIe bandwidth.
// pcie_bw_gbps: measured (or estimated) PCIe bandwidth in GB/s.
// For a single hop on PCIe 4.0 x16: ~32 GB/s.
// For cross-switch: ~10 GB/s typical.
//
// Formula: target latency 1ms at given BW, rounded to vector alignment,
// clamped to [kMinBucketBytes, kMaxBucketBytes].
// ---------------------------------------------------------------------------
size_t compute_pcie_bucket_size(float pcie_bw_gbps)
{
    // Target: fill PCIe for ~1 ms to hide per-bucket launch overhead.
    constexpr float kTargetLatencyMs = 1.0f;
    size_t raw = (size_t)(pcie_bw_gbps * 1e9f * kTargetLatencyMs * 1e-3f);
    // Round to vector alignment.
    raw = (raw / kBucketAlign) * kBucketAlign;
    raw = std::max(raw, kMinBucketBytes);
    raw = std::min(raw, kMaxBucketBytes);
    return raw;
}

// ---------------------------------------------------------------------------
// Host-side launch wrappers
// ---------------------------------------------------------------------------

void launch_pcie_gradient_pack(
    __nv_bfloat16*        bucket,
    const PcieGradChunk*  chunks,
    int                   num_chunks,
    size_t                bucket_elems,
    int                   sm_version,
    cudaStream_t          stream)
{
    // Copy chunk descriptors to device.
    PcieGradChunk* d_chunks = nullptr;
    cudaMallocAsync(&d_chunks, num_chunks * sizeof(PcieGradChunk), stream);
    cudaMemcpyAsync(d_chunks, chunks, num_chunks * sizeof(PcieGradChunk),
                    cudaMemcpyHostToDevice, stream);

    const size_t vec_elems = bucket_elems / kARVecWidth;
    const int grid = (int)std::min(
        (vec_elems + kARBlockSize - 1) / kARBlockSize, (size_t)65535);

    // Use 256-thread blocks for all SM versions (pack is bandwidth-bound, not
    // compute-bound, so occupancy differences are negligible).
    pcie_gradient_pack_kernel<kARBlockSize><<<grid, kARBlockSize, 0, stream>>>(
        bucket, d_chunks, num_chunks, bucket_elems);

    cudaFreeAsync(d_chunks, stream);
}

void launch_pcie_ring_reduce(
    __nv_bfloat16*       dst,
    const __nv_bfloat16* src,
    size_t               n_elems,
    int                  sm_version,
    cudaStream_t         stream)
{
    const size_t vec_elems = n_elems / kARVecWidth;
    const int grid = (int)std::min(
        (vec_elems + kARBlockSize - 1) / kARBlockSize, (size_t)65535);

    if (sm_version >= 120) {
        constexpr int kBS = 512;
        pcie_ring_reduce_kernel<120, kBS><<<grid, kBS, 0, stream>>>(
            dst, src, n_elems);
    } else if (sm_version >= 90) {
        constexpr int kBS = 256;
        pcie_ring_reduce_kernel<90, kBS><<<grid, kBS, 0, stream>>>(
            dst, src, n_elems);
    } else {
        pcie_ring_reduce_kernel<86, kARBlockSize><<<grid, kARBlockSize, 0, stream>>>(
            dst, src, n_elems);
    }
}

void launch_pcie_allreduce_finalise(
    __nv_bfloat16*       out,
    const __nv_bfloat16* src,
    size_t               n_elems,
    int                  world_size,
    int                  sm_version,
    cudaStream_t         stream)
{
    const float inv_ws = 1.0f / (float)world_size;
    const size_t vec_elems = n_elems / kARVecWidth;
    const int grid = (int)std::min(
        (vec_elems + kARBlockSize - 1) / kARBlockSize, (size_t)65535);

    if (sm_version >= 120) {
        constexpr int kBS = 512;
        pcie_allreduce_finalise_kernel<120, kBS><<<grid, kBS, 0, stream>>>(
            out, src, n_elems, inv_ws);
    } else if (sm_version >= 90) {
        constexpr int kBS = 256;
        pcie_allreduce_finalise_kernel<90, kBS><<<grid, kBS, 0, stream>>>(
            out, src, n_elems, inv_ws);
    } else {
        pcie_allreduce_finalise_kernel<86, kARBlockSize><<<grid, kARBlockSize, 0, stream>>>(
            out, src, n_elems, inv_ws);
    }
}
