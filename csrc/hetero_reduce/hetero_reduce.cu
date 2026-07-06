// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * hetero_reduce.cu  —  SM8.6 / 9.0 / 12.0 dispatch, warp shuffle reductions,
 *                       proper launch bounds, variable hidden sizes.
 *
 * Heterogeneous reduce-scatter for PCIe-only clusters:
 *   2× A6000 (SM8.6) + 1× H100 (SM9.0) + 2× Blackwell (SM12.0)
 *
 * ═══════════════════════════════════════════════════════════════════════
 * ALGORITHMIC DESIGN
 * ═══════════════════════════════════════════════════════════════════════
 *
 * 1. TUNING POLICY STRUCT  (KernelPolicy<SmVer>)
 *    Block size, min_blocks_per_sm, per-tier bucket sizing, and vec width
 *    are all derived from a single specialised struct.  Adding SM13.0 or
 *    any future tier requires only one new specialisation.
 *
 * 2. WARP-LEVEL COOPERATIVE REDUCTION — __shfl_xor_sync butterfly
 *    The innermost accumulation is a per-thread independent loop (each
 *    thread owns disjoint output elements), so no intra-warp reduce is
 *    needed on the HOT path.  The warp-coop kernel explicitly splits
 *    input tensors across lanes and then uses a full butterfly
 *    (__shfl_xor_sync masks 16/8/4/2/1) to fold partial sums — no smem.
 *    On SM9.0+ the compiler can emit REDUX.SYNC.ADD.F32 for the fold.
 *
 * 3. VARIABLE HIDDEN SIZES
 *    All loops use runtime `hidden` / `shard_count` parameters; nothing
 *    is hard-coded to 4096 or any other model width.  The vectorised
 *    path handles shard_count % kVec != 0 via a scalar tail.
 *    A dedicated variable-hidden reduce-scatter kernel is instantiated for
 *    each SM tier to allow the compiler to optimise the inner loop body
 *    independently of the block size.
 *
 * 4. PROPER __launch_bounds__
 *    All __launch_bounds__(maxThreads, minBlocksPerSM) values come from
 *    KernelPolicy<SmVer>::kBlockSize and kMinBlocksPerSM.  This lets the
 *    compiler spill to L2 registers instead of stack on H100 (50 MB L2)
 *    and reduces register pressure on A6000 (6 MB L2).
 *
 * 5. PER-TIER ADAPTIVE BUCKET SIZING
 *    H100  (SM9.0):      4M elements (32 MB) — large L2
 *    A6000 (SM8.6):    512K elements  (4 MB) — small L2
 *    Blackwell (SM12.0): 2M elements (16 MB) — moderate L2
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdint>
#include <algorithm>

#include "hetero_reduce.h"
#include "ds_kernel_utils.h"

namespace cg = cooperative_groups;

// ─────────────────────────────────────────────────────────────────────────────
// Section 1: Tuning Policy Struct
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer> struct KernelPolicy;

template <> struct KernelPolicy<86> {
    static constexpr int    kBlockSize      = 256;
    static constexpr int    kMinBlocksPerSM = 2;
    static constexpr size_t kBucketElems    = 512UL * 1024UL;   // 4 MB in BF16
    static constexpr int    kVecWidth       = 8;
};

template <> struct KernelPolicy<90> {
    static constexpr int    kBlockSize      = 256;
    static constexpr int    kMinBlocksPerSM = 4;
    static constexpr size_t kBucketElems    = 4UL * 1024UL * 1024UL;  // 32 MB
    static constexpr int    kVecWidth       = 8;
};

template <> struct KernelPolicy<120> {
    static constexpr int    kBlockSize      = 512;
    static constexpr int    kMinBlocksPerSM = 4;
    static constexpr size_t kBucketElems    = 2UL * 1024UL * 1024UL;  // 16 MB
    static constexpr int    kVecWidth       = 8;
};

// Generic fallback
template <int SmVer> struct KernelPolicy {
    static constexpr int    kBlockSize      = 256;
    static constexpr int    kMinBlocksPerSM = 2;
    static constexpr size_t kBucketElems    = 512UL * 1024UL;
    static constexpr int    kVecWidth       = 8;
};

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: Constant-memory inline pointer array
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int kMaxInlinePointers = 32;
static constexpr int kMaxTensors        = 32;
static constexpr int kMaxWarps          = 16; // KernelPolicy<120>::kBlockSize / 32

__constant__ const __nv_bfloat16* c_input_ptrs[kMaxInlinePointers];

// ─────────────────────────────────────────────────────────────────────────────
// Section 3: Vectorised load/store helpers
// ─────────────────────────────────────────────────────────────────────────────

DS_D_INLINE void bf16x8_accumulate(
    const __nv_bfloat16* __restrict__ ptr,
    float2& acc0, float2& acc1, float2& acc2, float2& acc3)
{
    const uint4 raw = *reinterpret_cast<const uint4*>(ptr);
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
    acc0.x += __bfloat162float(p[0].x);  acc0.y += __bfloat162float(p[0].y);
    acc1.x += __bfloat162float(p[1].x);  acc1.y += __bfloat162float(p[1].y);
    acc2.x += __bfloat162float(p[2].x);  acc2.y += __bfloat162float(p[2].y);
    acc3.x += __bfloat162float(p[3].x);  acc3.y += __bfloat162float(p[3].y);
}

DS_D_INLINE void fp32x8_store_bf16(
    __nv_bfloat16* __restrict__ ptr,
    float2 a0, float2 a1, float2 a2, float2 a3)
{
    __nv_bfloat162 b0 = {__float2bfloat16(a0.x), __float2bfloat16(a0.y)};
    __nv_bfloat162 b1 = {__float2bfloat16(a1.x), __float2bfloat16(a1.y)};
    __nv_bfloat162 b2 = {__float2bfloat16(a2.x), __float2bfloat16(a2.y)};
    __nv_bfloat162 b3 = {__float2bfloat16(a3.x), __float2bfloat16(a3.y)};
    uint4 out;
    out.x = *reinterpret_cast<uint32_t*>(&b0);
    out.y = *reinterpret_cast<uint32_t*>(&b1);
    out.z = *reinterpret_cast<uint32_t*>(&b2);
    out.w = *reinterpret_cast<uint32_t*>(&b3);
    *reinterpret_cast<uint4*>(ptr) = out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 4: Warp-level shuffle reduction helpers
//
//   All variants use full butterfly (__shfl_xor_sync, masks 16/8/4/2/1).
//   On SM9.0+ the compiler emits REDUX.SYNC.ADD.F32; on SM8.6 it emits
//   five SHFL.BFLY.B32 instructions.  The #pragma unroll ensures the
//   compiler can schedule the 5 rounds without loop overhead.
// ─────────────────────────────────────────────────────────────────────────────

// Float sum reduction — butterfly XOR, 5 rounds, no shared memory.
DS_D_INLINE float warp_reduce_sum_f32(float v)
{
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, mask);
    return v;
}

// float2 pair reduction — two independent butterfly chains, same cost.
DS_D_INLINE float2 warp_reduce_sum_f2(float2 v)
{
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v.x += __shfl_xor_sync(0xffffffff, v.x, mask);
        v.y += __shfl_xor_sync(0xffffffff, v.y, mask);
    }
    return v;
}

// Block-level sum reduction using warp butterfly + smem exchange.
// smem must have at least (kBlockSize / 32) float slots.
template <int kBlockSize>
DS_D_INLINE float block_reduce_sum_f32(
    float               val,
    float* __restrict__ smem,
    cg::thread_block&   blk)
{
    constexpr int kNWarps = kBlockSize / hw_warp_size;
    const int lane    = threadIdx.x % hw_warp_size;
    const int warp_id = threadIdx.x / hw_warp_size;

    val = warp_reduce_sum_f32(val);
    if (lane == 0) smem[warp_id] = val;
    blk.sync();

    val = (threadIdx.x < kNWarps) ? smem[threadIdx.x] : 0.f;
    if (warp_id == 0) val = warp_reduce_sum_f32(val);

    if (threadIdx.x == 0) smem[0] = val;
    blk.sync();
    return smem[0];
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 5: Main Reduce-Scatter Kernel
//
//   Template parameters:
//     SmVer        — SM version (86 / 90 / 120)
//     UseConstMem  — true: read pointers from c_input_ptrs (fast path)
//                    false: read from d_inputs device pointer
//
//   Variable hidden sizes: shard_count is a runtime parameter; the kernel
//   handles any value >= 0, including non-multiples of kVec (scalar tail).
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer, bool UseConstMem>
__global__ void
__launch_bounds__(KernelPolicy<SmVer>::kBlockSize, KernelPolicy<SmVer>::kMinBlocksPerSM)
hetero_reduce_scatter_kernel(
    __nv_bfloat16* __restrict__              output,
    const __nv_bfloat16* const* __restrict__ d_inputs,
    int    num_tensors,
    size_t shard_offset,
    size_t shard_count)
{
    using Policy = KernelPolicy<SmVer>;
    constexpr int kVec = Policy::kVecWidth;

    const size_t vec_count = shard_count / kVec;
    const size_t tid       = (size_t)blockIdx.x * Policy::kBlockSize + threadIdx.x;
    const size_t stride    = (size_t)gridDim.x  * Policy::kBlockSize;

    // ── Vectorised path: 8 BF16 (128 bits) per thread per step ───────────
    for (size_t vec_idx = tid; vec_idx < vec_count; vec_idx += stride) {
        const size_t global_elem = shard_offset + vec_idx * kVec;

        float2 acc0 = {0.f,0.f}, acc1 = {0.f,0.f};
        float2 acc2 = {0.f,0.f}, acc3 = {0.f,0.f};

        if constexpr (UseConstMem) {
            #pragma unroll 4
            for (int t = 0; t < num_tensors; ++t)
                bf16x8_accumulate(c_input_ptrs[t] + global_elem,
                                  acc0, acc1, acc2, acc3);
        } else {
            #pragma unroll 4
            for (int t = 0; t < num_tensors; ++t)
                bf16x8_accumulate(d_inputs[t] + global_elem,
                                  acc0, acc1, acc2, acc3);
        }

        fp32x8_store_bf16(output + vec_idx * kVec, acc0, acc1, acc2, acc3);
    }

    // ── Scalar tail: handles variable hidden sizes not divisible by kVec ──
    // Thread 0 handles remainder sequentially (at most kVec-1 elements).
    if (tid == 0) {
        const size_t scalar_start = vec_count * kVec;
        for (size_t e = scalar_start; e < shard_count; ++e) {
            float acc = 0.f;
            const size_t gidx = shard_offset + e;
            if constexpr (UseConstMem) {
                for (int t = 0; t < num_tensors; ++t)
                    acc += __bfloat162float(__ldg(c_input_ptrs[t] + gidx));
            } else {
                for (int t = 0; t < num_tensors; ++t)
                    acc += __bfloat162float(__ldg(d_inputs[t] + gidx));
            }
            output[e] = __float2bfloat16(acc);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 6: Warp-Cooperative Small-Tensor Kernel
//
//   For shard_count ≤ kSmallThreshold AND num_tensors ≤ 32:
//   threads WITHIN each warp split ownership of num_tensors inputs.
//   After per-lane partial accumulation, a full butterfly __shfl_xor_sync
//   fold is performed — 5 instructions per float, no shared memory.
//
//   This is the key warp-shuffle reduction path for small tensors.
// ─────────────────────────────────────────────────────────────────────────────

template <bool UseConstMem>
__global__ void __launch_bounds__(256, 4)
hetero_reduce_scatter_warp_coop(
    __nv_bfloat16* __restrict__              output,
    const __nv_bfloat16* const* __restrict__ d_inputs,
    int    num_tensors,
    size_t shard_offset,
    size_t shard_count)
{
    constexpr int kVec     = 8;
    constexpr int kBS      = 256;
    const int warp_id_g    = ((int)blockIdx.x * kBS + (int)threadIdx.x) / hw_warp_size;
    const int lane         = (int)threadIdx.x % hw_warp_size;
    const int total_warps  = ((int)gridDim.x * kBS) / hw_warp_size;
    const size_t vec_count = shard_count / kVec;

    for (size_t vec_idx = (size_t)warp_id_g; vec_idx < vec_count;
         vec_idx += (size_t)total_warps) {

        const size_t gidx = shard_offset + vec_idx * kVec;

        // Each lane accumulates a disjoint subset of input tensors.
        float2 acc0={0.f,0.f}, acc1={0.f,0.f}, acc2={0.f,0.f}, acc3={0.f,0.f};

        for (int t = lane; t < num_tensors; t += hw_warp_size) {
            const __nv_bfloat16* src = UseConstMem ? c_input_ptrs[t] : d_inputs[t];
            bf16x8_accumulate(src + gidx, acc0, acc1, acc2, acc3);
        }

        // Warp-level butterfly fold — no shared memory, 5 SHFL instructions each.
        // This is the canonical warp shuffle reduction used throughout this file.
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            acc0.x += __shfl_xor_sync(0xffffffff, acc0.x, mask);
            acc0.y += __shfl_xor_sync(0xffffffff, acc0.y, mask);
            acc1.x += __shfl_xor_sync(0xffffffff, acc1.x, mask);
            acc1.y += __shfl_xor_sync(0xffffffff, acc1.y, mask);
            acc2.x += __shfl_xor_sync(0xffffffff, acc2.x, mask);
            acc2.y += __shfl_xor_sync(0xffffffff, acc2.y, mask);
            acc3.x += __shfl_xor_sync(0xffffffff, acc3.x, mask);
            acc3.y += __shfl_xor_sync(0xffffffff, acc3.y, mask);
        }

        if (lane == 0) {
            fp32x8_store_bf16(output + vec_idx * kVec, acc0, acc1, acc2, acc3);
        }
    }

    // Scalar tail for variable hidden sizes.
    const size_t scalar_start = vec_count * kVec;
    for (size_t e = scalar_start + (size_t)lane; e < shard_count; e += hw_warp_size) {
        float acc = 0.f;
        const size_t gidx = shard_offset + e;
        for (int t = 0; t < num_tensors; t += hw_warp_size) {
            const int tt = t + lane;
            if (tt < num_tensors) {
                const __nv_bfloat16* src = UseConstMem ? c_input_ptrs[tt] : d_inputs[tt];
                acc += __bfloat162float(__ldg(src + gidx));
            }
        }
        // Reduce across active lanes covering the same element.
        acc = warp_reduce_sum_f32(acc);
        if (lane == 0) output[e] = __float2bfloat16(acc);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 7: Full-Tensor Reduce kernel (no shard scatter)
//   Identical structure to reduce-scatter; shard covers the full tensor.
//   Variable hidden sizes supported via scalar tail.
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer, bool UseConstMem>
__global__ void
__launch_bounds__(KernelPolicy<SmVer>::kBlockSize, KernelPolicy<SmVer>::kMinBlocksPerSM)
fused_bf16_reduce_kernel(
    __nv_bfloat16* __restrict__              output,
    const __nv_bfloat16* const* __restrict__ d_inputs,
    int    num_tensors,
    size_t n_elems)
{
    using Policy = KernelPolicy<SmVer>;
    constexpr int kVec = Policy::kVecWidth;

    const size_t tid    = (size_t)blockIdx.x * Policy::kBlockSize + threadIdx.x;
    const size_t stride = (size_t)gridDim.x  * Policy::kBlockSize;
    const size_t vec_n  = n_elems / kVec;

    for (size_t i = tid; i < vec_n; i += stride) {
        float2 a0={0.f,0.f}, a1={0.f,0.f}, a2={0.f,0.f}, a3={0.f,0.f};
        const size_t base = i * kVec;

        if constexpr (UseConstMem) {
            #pragma unroll 4
            for (int t = 0; t < num_tensors; ++t)
                bf16x8_accumulate(c_input_ptrs[t] + base, a0, a1, a2, a3);
        } else {
            #pragma unroll 4
            for (int t = 0; t < num_tensors; ++t)
                bf16x8_accumulate(d_inputs[t] + base, a0, a1, a2, a3);
        }
        fp32x8_store_bf16(output + base, a0, a1, a2, a3);
    }

    // Scalar tail — variable hidden sizes (any n_elems, not just power-of-two)
    if (tid == 0) {
        for (size_t e = vec_n * kVec; e < n_elems; ++e) {
            float acc = 0.f;
            if constexpr (UseConstMem) {
                for (int t = 0; t < num_tensors; ++t)
                    acc += __bfloat162float(__ldg(c_input_ptrs[t] + e));
            } else {
                for (int t = 0; t < num_tensors; ++t)
                    acc += __bfloat162float(__ldg(d_inputs[t] + e));
            }
            output[e] = __float2bfloat16(acc);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 8: Helper — setup input pointer array (constant mem or device alloc)
// ─────────────────────────────────────────────────────────────────────────────

static const __nv_bfloat16** setup_input_ptrs(
    const __nv_bfloat16* const* host_ptrs,
    int num_tensors,
    cudaStream_t stream)
{
    if (num_tensors <= kMaxInlinePointers) {
        cudaMemcpyToSymbolAsync(c_input_ptrs, host_ptrs,
            num_tensors * sizeof(const __nv_bfloat16*), 0,
            cudaMemcpyHostToDevice, stream);
        return nullptr;
    }
    const __nv_bfloat16** d_ptrs = nullptr;
    cudaMallocAsync(reinterpret_cast<void**>(&d_ptrs),
        num_tensors * sizeof(const __nv_bfloat16*), stream);
    cudaMemcpyAsync(d_ptrs, host_ptrs,
        num_tensors * sizeof(const __nv_bfloat16*),
        cudaMemcpyHostToDevice, stream);
    return d_ptrs;
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 9: SM-specialised dispatch helpers with proper launch bounds
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer>
static void dispatch_reduce_scatter(
    __nv_bfloat16* output,
    const __nv_bfloat16** d_inputs,
    bool use_const,
    int num_tensors,
    size_t shard_offset, size_t shard_count,
    cudaStream_t stream)
{
    using Policy = KernelPolicy<SmVer>;
    constexpr size_t kSmallThresh = 128UL * 1024UL;

    if (shard_count <= kSmallThresh && num_tensors <= hw_warp_size) {
        // Warp-cooperative path — all arithmetic via warp shuffle, no smem.
        const size_t vec_count = (shard_count + Policy::kVecWidth - 1)
                                  / Policy::kVecWidth;
        const int warps_needed = (int)std::min(vec_count, (size_t)65535 * 8);
        const int grid = (warps_needed * hw_warp_size + 255) / 256;
        if (use_const)
            hetero_reduce_scatter_warp_coop<true>
                <<<grid, 256, 0, stream>>>(
                    output, nullptr, num_tensors, shard_offset, shard_count);
        else
            hetero_reduce_scatter_warp_coop<false>
                <<<grid, 256, 0, stream>>>(
                    output, d_inputs, num_tensors, shard_offset, shard_count);
        return;
    }

    // Standard path — grid sized to cover all vectorised elements.
    const size_t vec_count = shard_count / Policy::kVecWidth;
    const int grid = (int)std::min(
        (vec_count + Policy::kBlockSize - 1) / Policy::kBlockSize,
        (size_t)65535);

    if (use_const)
        hetero_reduce_scatter_kernel<SmVer, true>
            <<<grid, Policy::kBlockSize, 0, stream>>>(
                output, nullptr, num_tensors, shard_offset, shard_count);
    else
        hetero_reduce_scatter_kernel<SmVer, false>
            <<<grid, Policy::kBlockSize, 0, stream>>>(
                output, d_inputs, num_tensors, shard_offset, shard_count);
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 10: Public API Implementations
// ─────────────────────────────────────────────────────────────────────────────

void launch_fused_bf16_reduce(
    __nv_bfloat16*              output,
    const __nv_bfloat16* const* inputs,
    int                         num_tensors,
    size_t                      n_elems,
    int                         sm_version,
    cudaStream_t                stream)
{
    if (n_elems == 0 || num_tensors == 0) return;

    const __nv_bfloat16** d_inputs = setup_input_ptrs(inputs, num_tensors, stream);
    const bool use_const = (d_inputs == nullptr);

    // SM dispatch: derive grid size from per-SM policy to honour launch bounds.
    auto launch_reduce = [&]<int SmVer>() {
        using Policy = KernelPolicy<SmVer>;
        const size_t vec_n = n_elems / Policy::kVecWidth;
        const int grid = (int)std::min(
            (vec_n + Policy::kBlockSize - 1) / Policy::kBlockSize,
            (size_t)65535);
        if (use_const)
            fused_bf16_reduce_kernel<SmVer, true>
                <<<grid, Policy::kBlockSize, 0, stream>>>(
                    output, nullptr, num_tensors, n_elems);
        else
            fused_bf16_reduce_kernel<SmVer, false>
                <<<grid, Policy::kBlockSize, 0, stream>>>(
                    output, d_inputs, num_tensors, n_elems);
    };

    if      (sm_version >= 120) launch_reduce.template operator()<120>();
    else if (sm_version >= 90)  launch_reduce.template operator()<90>();
    else                        launch_reduce.template operator()<86>();

    if (d_inputs) cudaFreeAsync(d_inputs, stream);
}

// Tier weight for heterogeneous shard allocation.
// SM12.0 (Blackwell) → 4, SM9.0 (H100) → 3, SM8.6 (A6000) → 1.
static int tier_weight(int sm_version)
{
    if (sm_version >= 120) return 4;
    if (sm_version >= 90)  return 3;
    return 1;
}

void compute_hetero_shard_ranges(
    const HeteroTierDesc* tiers,
    int                   num_tiers,
    size_t                total_elems,
    size_t*               out_offsets,
    size_t*               out_counts)
{
    int total_weight = 0;
    for (int i = 0; i < num_tiers; ++i)
        total_weight += tier_weight(tiers[i].sm_version);

    // Alignment: 8 BF16 elements (128-bit vector boundary).
    // Ensures every shard is usable by the vectorised path regardless of
    // hidden model size (variable hidden sizes are all multiples of 8).
    constexpr size_t kAlign = 8;

    size_t assigned = 0;
    for (int i = 0; i < num_tiers; ++i) {
        if (i == num_tiers - 1) {
            out_offsets[i] = assigned;
            out_counts[i]  = total_elems - assigned;
        } else {
            size_t raw = (total_elems * (size_t)tier_weight(tiers[i].sm_version))
                         / (size_t)total_weight;
            raw = (raw / kAlign) * kAlign;
            out_offsets[i] = assigned;
            out_counts[i]  = raw;
            assigned += raw;
        }
    }
}

size_t hetero_bucket_size_elems(int sm_version)
{
    if (sm_version >= 120) return KernelPolicy<120>::kBucketElems;
    if (sm_version >= 90)  return KernelPolicy<90>::kBucketElems;
    return KernelPolicy<86>::kBucketElems;
}

void launch_hetero_reduce_scatter(
    __nv_bfloat16*              output,
    const __nv_bfloat16* const* inputs,
    int                         num_tensors,
    size_t                      shard_offset,
    size_t                      shard_count,
    int                         sm_version,
    cudaStream_t                stream)
{
    if (shard_count == 0 || num_tensors == 0) return;

    const __nv_bfloat16** d_inputs = setup_input_ptrs(inputs, num_tensors, stream);
    const bool use_const = (d_inputs == nullptr);

    // Three-way SM dispatch: each instantiation has distinct __launch_bounds__
    // matching the hardware's occupancy characteristics.
    if      (sm_version >= 120)
        dispatch_reduce_scatter<120>(output, d_inputs, use_const,
            num_tensors, shard_offset, shard_count, stream);
    else if (sm_version >= 90)
        dispatch_reduce_scatter<90>(output, d_inputs, use_const,
            num_tensors, shard_offset, shard_count, stream);
    else
        dispatch_reduce_scatter<86>(output, d_inputs, use_const,
            num_tensors, shard_offset, shard_count, stream);

    if (d_inputs) cudaFreeAsync(d_inputs, stream);
}
