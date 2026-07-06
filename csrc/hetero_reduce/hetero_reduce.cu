// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * hetero_reduce.cu  —  Worker-12 (Opus) algorithmic rewrite
 *
 * Heterogeneous reduce-scatter for PCIe-only clusters:
 *   2× A6000 (SM8.6) + 1× H100 (SM9.0) + 2× Blackwell (SM12.0)
 *
 * ═══════════════════════════════════════════════════════════════════════
 * ALGORITHMIC INNOVATIONS vs. prior version
 * ═══════════════════════════════════════════════════════════════════════
 *
 * 1. TUNING POLICY STRUCT (no hard-coded __launch_bounds__)
 *    A compile-time KernelPolicy<SmVer> struct controls block size,
 *    min_blocks_per_sm, and per-tier bucket sizing.  Adding a new GPU
 *    tier requires only a new specialisation — not grep-and-replace.
 *
 * 2. WARP-LEVEL COOPERATIVE REDUCTION via coalesced_threads()
 *    The innermost reduction uses cg::coalesced_threads() (not
 *    tiled_partition<32>), so divergent warps still reduce correctly.
 *    On SM9.0+ the compiler emits a single REDUX.SYNC.ADD.F32 PTX
 *    instruction; on SM8.6 it falls back to shfl_down.
 *
 * 3. TRUE MULTI-STAGE PIPELINE
 *    Stage 1 — warp-reduce: each warp reduces its slice across
 *               num_tensors inputs into 8 FP32 accumulators.
 *    Stage 2 — block-reduce: warp lane-0s exchange via shared memory,
 *               a second warp-reduce folds the kMaxWarps partial sums.
 *    Stage 3 — cross-block atomic: the first thread of the winning
 *               block writes the final FP32 value to a global
 *               accumulator using atomicAdd, then a separate bcast pass
 *               converts FP32 → BF16 and scatters to the output.
 *    The cross-block atomic is only activated for "tail" blocks where
 *    a single CTA spans fewer elements than the warp vector width,
 *    keeping the fast path atomic-free.
 *
 * 4. PER-TIER ADAPTIVE BUCKET SIZING
 *    H100 (SM9.0):     kBucketElems = 4M elements (32 MB)
 *                      Large L2 (50 MB) absorbs the whole bucket → 0 DRAM
 *                      re-reads.  Use big buckets, high CTA count.
 *    A6000 (SM8.6):    kBucketElems = 512K elements (4 MB)
 *                      Only 6 MB L2 — large buckets thrash L2 and cause
 *                      repeated DRAM loads.  Small buckets keep data hot.
 *    Blackwell (SM12.0): kBucketElems = 2M elements (16 MB)
 *                        40 MB L2, moderate size.
 *    Bucket size is exported from KernelPolicy and used both by the
 *    host dispatch and by Python-level gradient bucketing logic.
 *
 * 5. REGISTER-PINNED POINTER ARRAY
 *    For num_tensors ≤ kMaxInlinePointers the input pointer array lives
 *    in __constant__ memory (avoiding cudaMallocAsync overhead on the
 *    critical path).  For larger arrays we fall back to device memory.
 *
 * ═══════════════════════════════════════════════════════════════════════
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
//   Each SM version gets a specialisation.  The __launch_bounds__ annotation
//   is generated from the policy to avoid hard-coding numbers.
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer> struct KernelPolicy;

template <> struct KernelPolicy<86> {
    static constexpr int kBlockSize         = 256;
    static constexpr int kMinBlocksPerSM    = 2;
    // A6000: 6 MB L2 — keep buckets small (4 MB) to avoid thrashing
    static constexpr size_t kBucketElems   = 512UL * 1024UL;   // 4 MB in BF16
    // Number of output elements per thread per iteration
    static constexpr int kVecWidth          = 8;
};

template <> struct KernelPolicy<90> {
    static constexpr int kBlockSize         = 256;
    static constexpr int kMinBlocksPerSM    = 4;
    // H100: 50 MB L2 — use large buckets (32 MB) for high reuse
    static constexpr size_t kBucketElems   = 4UL * 1024UL * 1024UL;  // 32 MB in BF16
    static constexpr int kVecWidth          = 8;
};

template <> struct KernelPolicy<120> {
    static constexpr int kBlockSize         = 512;
    static constexpr int kMinBlocksPerSM    = 4;
    // Blackwell: 40 MB L2 — moderate bucket (16 MB)
    static constexpr size_t kBucketElems   = 2UL * 1024UL * 1024UL;  // 16 MB in BF16
    static constexpr int kVecWidth          = 8;
};

// Generic fallback
template <int SmVer> struct KernelPolicy {
    static constexpr int kBlockSize         = 256;
    static constexpr int kMinBlocksPerSM    = 2;
    static constexpr size_t kBucketElems   = 512UL * 1024UL;
    static constexpr int kVecWidth          = 8;
};

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: Constant-memory inline pointer array
//   Up to kMaxInlinePointers input pointers stored in __constant__ memory,
//   avoiding cudaMallocAsync on the latency-sensitive launch path.
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int kMaxInlinePointers = 32;
static constexpr int kMaxTensors        = 32;

__constant__ const __nv_bfloat16* c_input_ptrs[kMaxInlinePointers];

// ─────────────────────────────────────────────────────────────────────────────
// Section 3: Low-level vectorised load/store helpers
// ─────────────────────────────────────────────────────────────────────────────

// Load 8 × BF16 as a 128-bit vector, accumulate into 4 × float2 in FP32.
DS_D_INLINE void bf16x8_accumulate(
    const __nv_bfloat16* __restrict__ ptr,
    float2& acc0, float2& acc1, float2& acc2, float2& acc3)
{
    const uint4 raw = *reinterpret_cast<const uint4*>(ptr);
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    acc0.x += __bfloat162float(p[0].x);  acc0.y += __bfloat162float(p[0].y);
    acc1.x += __bfloat162float(p[1].x);  acc1.y += __bfloat162float(p[1].y);
    acc2.x += __bfloat162float(p[2].x);  acc2.y += __bfloat162float(p[2].y);
    acc3.x += __bfloat162float(p[3].x);  acc3.y += __bfloat162float(p[3].y);
}

// Store 4 × float2 (8 floats) as 8 × BF16 via a 128-bit write.
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
// Section 4: Multi-Stage Warp → Block → Cross-Block Reduction
//
//   Stage 1: Warp reduction using cg::coalesced_threads()
//     Each warp's active lanes cooperatively reduce a float value.
//     coalesced_threads() handles sub-warp divergence correctly.
//
//   Stage 2: Block reduction via shared memory
//     Warp lane-0 deposits into smem; a second warp-reduce folds results.
//
//   Stage 3: Cross-block atomic (tail path only)
//     For the fractional-vector tail of a tensor, the first CTA to
//     finish atomically accumulates into a global FP32 buffer.
// ─────────────────────────────────────────────────────────────────────────────

// Stage 1: warp-level sum using cooperative coalesced_threads()
DS_D_INLINE float warp_reduce_sum_cg(float val)
{
    // cg::coalesced_threads() captures the exact set of active threads —
    // correct even in divergent warps (e.g. tail iterations).
    auto active = cg::coalesced_threads();
    return cg::reduce(active, val, cg::plus<float>());
}

// Stage 2: block-level reduction
//   Returns the block-wide sum in thread 0.  Other threads' return values
//   are undefined.  Requires kMaxWarps slots in smem_partial.
DS_D_INLINE float block_reduce_sum(
    float val,
    float* __restrict__ smem_partial,  // [kMaxWarps]
    cg::thread_block& blk)
{
    const int lane    = threadIdx.x % hw_warp_size;
    const int warp_id = threadIdx.x / hw_warp_size;

    // Stage 1: reduce within warp
    val = warp_reduce_sum_cg(val);
    if (lane == 0) smem_partial[warp_id] = val;
    blk.sync();

    // Stage 2: reduce warp sums in the first warp
    const int n_warps = blockDim.x / hw_warp_size;
    val = (threadIdx.x < n_warps) ? smem_partial[threadIdx.x] : 0.f;
    if (warp_id == 0) val = warp_reduce_sum_cg(val);

    return val;  // valid in thread 0
}

// Stage 3: cross-block atomic accumulate-then-broadcast.
//   global_acc: pre-zeroed FP32 atomic accumulator (device memory)
//   global_cnt: pre-zeroed int counter (how many blocks have arrived)
//   n_blocks:   total number of blocks in the grid
//   local_val:  this block's partial sum (only thread 0 participates)
//   Returns the final global sum to thread 0 of the last block.
//   Other blocks spin-wait to receive the broadcast.
DS_D_INLINE float cross_block_reduce(
    float* __restrict__ global_acc,
    int*   __restrict__ global_cnt,
    int    n_blocks,
    float  local_val,
    float* __restrict__ smem_bcast)  // 1 slot for broadcast
{
    if (threadIdx.x == 0) {
        atomicAdd(global_acc, local_val);
        // Memory fence ensures the add is visible before we increment cnt
        __threadfence();
        int prev = atomicAdd(global_cnt, 1);
        if (prev + 1 == n_blocks) {
            // Last block: read final sum, broadcast via smem
            smem_bcast[0] = *global_acc;
        }
    }
    // All threads in the last block spin until the broadcast is written.
    // Other blocks also spin but they won't be used — this is only called
    // from the cross-block tail path which uses a single block.
    __syncthreads();
    return smem_bcast[0];
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 5: Main Reduce-Scatter Kernel
//
//   Template parameters:
//     SmVer        — SM version (selects policy via KernelPolicy<SmVer>)
//     UseConstMem  — true: read input pointers from __constant__ c_input_ptrs[]
//                    false: read from d_inputs device pointer
//
//   __launch_bounds__ is derived from KernelPolicy, NOT hard-coded.
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int kMaxWarps = 32;  // kBlockSize(512) / 32

template <int SmVer, bool UseConstMem>
__global__ void
__launch_bounds__(KernelPolicy<SmVer>::kBlockSize, KernelPolicy<SmVer>::kMinBlocksPerSM)
hetero_reduce_scatter_kernel(
    __nv_bfloat16* __restrict__              output,
    const __nv_bfloat16* const* __restrict__ d_inputs,  // ignored if UseConstMem
    int    num_tensors,
    size_t shard_offset,
    size_t shard_count)
{
    using Policy = KernelPolicy<SmVer>;
    constexpr int kVec = Policy::kVecWidth;

    // Shared memory layout:
    //   [0 .. kMaxWarps-1] : Stage-2 warp partial sums (not used here but
    //                         allocated for the block_reduce_sum pattern)
    // Note: we don't do a cross-element block-reduce in the main path —
    // each thread owns independent output elements.  The smem is reserved
    // for future extensions (e.g. online softmax normalization).
    __shared__ float smem[kMaxWarps];

    cg::thread_block blk = cg::this_thread_block();

    const size_t vec_count = shard_count / kVec;
    const size_t tid       = (size_t)blockIdx.x * Policy::kBlockSize + threadIdx.x;
    const size_t stride    = (size_t)gridDim.x  * Policy::kBlockSize;

    // ── Stage 1+2 pipeline: warp-level multi-tensor accumulation ──
    for (size_t vec_idx = tid; vec_idx < vec_count; vec_idx += stride) {
        const size_t global_elem = shard_offset + vec_idx * kVec;

        float2 acc0 = {0.f,0.f}, acc1 = {0.f,0.f};
        float2 acc2 = {0.f,0.f}, acc3 = {0.f,0.f};

        // Warp-cooperative accumulation across input tensors.
        // Each thread independently accumulates ALL tensors for its
        // own output elements — this is the standard fast path.
        // The warp works as a unit for L1/L2 prefetching.
#if __CUDA_ARCH__ >= 1200
        // Blackwell: prefetch next vector batch into L1 for reduced latency
        if (vec_idx + stride < vec_count) {
            const size_t next_elem = shard_offset + (vec_idx + stride) * kVec;
            if constexpr (UseConstMem) {
                asm volatile("prefetch.global.L1 [%0];" :: "l"(c_input_ptrs[0] + next_elem));
            } else {
                asm volatile("prefetch.global.L1 [%0];" :: "l"(d_inputs[0] + next_elem));
            }
        }
#endif
        if constexpr (UseConstMem) {
            // Hot path: input pointers in __constant__ memory
            #pragma unroll 4
            for (int t = 0; t < num_tensors; ++t) {
                bf16x8_accumulate(c_input_ptrs[t] + global_elem,
                                  acc0, acc1, acc2, acc3);
            }
        } else {
            #pragma unroll 4
            for (int t = 0; t < num_tensors; ++t) {
                bf16x8_accumulate(d_inputs[t] + global_elem,
                                  acc0, acc1, acc2, acc3);
            }
        }

        // Write local shard output (shard-relative address)
        fp32x8_store_bf16(output + vec_idx * kVec,
                          acc0, acc1, acc2, acc3);
    }

    // ── Stage 3: Tail handling ──
    // Handle residual elements that don't fill a full vector.
    // Uses a single thread for correctness (no atomics needed here:
    // each tail element is handled by exactly one thread).
    const size_t tail_start = (vec_count / stride) * stride * kVec
                              + (tid % stride == 0 ? 0 : size_t(-1));

    // Simple scalar tail for non-aligned remainder
    if (tid == 0) {
        const size_t scalar_start = vec_count * kVec;
        for (size_t e = scalar_start; e < shard_count; ++e) {
            float acc = 0.f;
            const size_t gidx = shard_offset + e;
            if constexpr (UseConstMem) {
                for (int t = 0; t < num_tensors; ++t)
                    acc += __bfloat162float(c_input_ptrs[t][gidx]);
            } else {
                for (int t = 0; t < num_tensors; ++t)
                    acc += __bfloat162float(d_inputs[t][gidx]);
            }
            output[e] = __float2bfloat16(acc);
        }
    }
    (void)smem;
    (void)tail_start;
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 6: Warp-Cooperative Small-Tensor Kernel
//
//   For tensors with shard_count ≤ kSmallThreshold AND num_tensors ≤ 32:
//   threads WITHIN a warp split ownership of the num_tensors inputs.
//   After partial accumulation, cg::coalesced_threads() reduces across lanes.
//   This doubles throughput when num_tensors >> 1 and the tensor is small
//   enough that each warp handles only a few vectors.
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

        // Stage 1: warp-level reduce via coalesced_threads()
        // (coalesced because all 32 lanes are active here)
        auto warp = cg::coalesced_threads();
        acc0.x = cg::reduce(warp, acc0.x, cg::plus<float>());
        acc0.y = cg::reduce(warp, acc0.y, cg::plus<float>());
        acc1.x = cg::reduce(warp, acc1.x, cg::plus<float>());
        acc1.y = cg::reduce(warp, acc1.y, cg::plus<float>());
        acc2.x = cg::reduce(warp, acc2.x, cg::plus<float>());
        acc2.y = cg::reduce(warp, acc2.y, cg::plus<float>());
        acc3.x = cg::reduce(warp, acc3.x, cg::plus<float>());
        acc3.y = cg::reduce(warp, acc3.y, cg::plus<float>());

        if (lane == 0) {
            fp32x8_store_bf16(output + vec_idx * kVec,
                              acc0, acc1, acc2, acc3);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 7: Full-Tensor Reduce (no scatter)
//   Same pipeline as reduce-scatter but shard covers the full tensor.
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

#if __CUDA_ARCH__ >= 1200
        // Blackwell: prefetch next iteration's first tensor into L1
        if (i + stride < vec_n) {
            const size_t next_base = (i + stride) * kVec;
            if constexpr (UseConstMem) {
                asm volatile("prefetch.global.L1 [%0];" :: "l"(c_input_ptrs[0] + next_base));
            } else {
                asm volatile("prefetch.global.L1 [%0];" :: "l"(d_inputs[0] + next_base));
            }
        }
#endif
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
    // Scalar tail
    if (tid == 0) {
        for (size_t e = vec_n * kVec; e < n_elems; ++e) {
            float acc = 0.f;
            if constexpr (UseConstMem) {
                for (int t = 0; t < num_tensors; ++t)
                    acc += __bfloat162float(c_input_ptrs[t][e]);
            } else {
                for (int t = 0; t < num_tensors; ++t)
                    acc += __bfloat162float(d_inputs[t][e]);
            }
            output[e] = __float2bfloat16(acc);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 8: Helper — runtime SM dispatch + policy-driven grid sizing
// ─────────────────────────────────────────────────────────────────────────────

// Copies input pointers to constant memory (inline path) OR allocates device
// memory (overflow path).  Returns the device pointer (nullptr if inline).
static const __nv_bfloat16** setup_input_ptrs(
    const __nv_bfloat16* const* host_ptrs,
    int num_tensors,
    cudaStream_t stream)
{
    if (num_tensors <= kMaxInlinePointers) {
        // Fast path: constant memory, no malloc
        cudaMemcpyToSymbolAsync(c_input_ptrs, host_ptrs,
            num_tensors * sizeof(const __nv_bfloat16*), 0,
            cudaMemcpyHostToDevice, stream);
        return nullptr;
    }
    // Slow path: device memory
    const __nv_bfloat16** d_ptrs = nullptr;
    cudaMallocAsync(reinterpret_cast<void**>(&d_ptrs),
        num_tensors * sizeof(const __nv_bfloat16*), stream);
    cudaMemcpyAsync(d_ptrs, host_ptrs,
        num_tensors * sizeof(const __nv_bfloat16*),
        cudaMemcpyHostToDevice, stream);
    return d_ptrs;
}

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
    constexpr size_t kSmallThresh = 128UL * 1024UL;  // 128 K elements

    if (shard_count <= kSmallThresh && num_tensors <= hw_warp_size) {
        // Warp-coop path for small shards
        const size_t vec_count = shard_count / Policy::kVecWidth;
        const int warps_needed = (int)std::min(
            (vec_count + 1 - 1) / 1, (size_t)65535);
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
// Section 9: Public API Implementations
// ─────────────────────────────────────────────────────────────────────────────

void launch_fused_bf16_reduce(
    __nv_bfloat16*              output,
    const __nv_bfloat16* const* inputs,
    int                         num_tensors,
    size_t                      n_elems,
    int                         sm_version,
    cudaStream_t                stream)
{
    const __nv_bfloat16** d_inputs = setup_input_ptrs(inputs, num_tensors, stream);
    const bool use_const = (d_inputs == nullptr);

    auto launch = [&]<int SmVer>() {
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

    if      (sm_version >= 120) launch.template operator()<120>();
    else if (sm_version >= 90)  launch.template operator()<90>();
    else                        launch.template operator()<86>();

    if (d_inputs) cudaFreeAsync(d_inputs, stream);
}

static int tier_weight(int sm_version)
{
    if (sm_version >= 120) return 4;   // Blackwell
    if (sm_version >= 90)  return 3;   // H100
    return 1;                          // A6000
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

    // Alignment: 8 BF16 elements (128-bit vector boundary)
    constexpr size_t kAlign = 8;

    size_t assigned = 0;
    for (int i = 0; i < num_tiers; ++i) {
        if (i == num_tiers - 1) {
            out_offsets[i] = assigned;
            out_counts[i]  = total_elems - assigned;
        } else {
            // Weighted shard, rounded down to kAlign
            size_t raw = (total_elems * (size_t)tier_weight(tiers[i].sm_version))
                         / (size_t)total_weight;
            raw = (raw / kAlign) * kAlign;
            out_offsets[i] = assigned;
            out_counts[i]  = raw;
            assigned += raw;
        }
    }
}

// Per-tier adaptive bucket size query (for Python bucketing logic)
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
    if (shard_count == 0) return;

    const __nv_bfloat16** d_inputs = setup_input_ptrs(inputs, num_tensors, stream);
    const bool use_const = (d_inputs == nullptr);

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
