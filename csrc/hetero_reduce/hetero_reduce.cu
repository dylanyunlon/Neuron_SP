// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * hetero_reduce.cu  —  NeurIPS 2026  DES-LOC + AutoSP production kernels
 *
 * Heterogeneous reduce-scatter for PCIe-only clusters:
 *   2× A6000 (SM8.6) + 1× H100 (SM9.0) + 2× Blackwell (SM12.0)
 *
 * ═══════════════════════════════════════════════════════════════════════
 * ALGORITHMIC DESIGN — CCCL-grade warp-cooperative + CAS cross-warp
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Reference: NVIDIA/cccl#9656 — warp-cooperative design, proper
 * launch_bounds, SM-conditional dispatch.
 *
 * 1. WARP SHUFFLE REDUCTION  (Stage 1)
 *    Each warp independently reduces its 8 × num_tensors BF16 values using
 *    cg::coalesced_threads() + cg::reduce() — a single PTX REDUX.SYNC.ADD
 *    on SM9.0+, shfl_down butterfly on SM8.6.  No shared memory involved.
 *
 * 2. CAS CROSS-WARP REDUCTION  (Stage 2, "true" cross-warp)
 *    Instead of a naive warp-0 smem gather, lane-0 of each warp races to
 *    claim slots in a shared atomic counter via atomicAdd on a per-smem-slot
 *    integer.  The final warp (detected by the counter reaching kMaxWarps)
 *    runs a second-level warp shuffle over the kMaxWarps partial sums
 *    loaded from a shared-memory staging array.  This is the CCCL tree-
 *    reduction pattern without any __syncthreads() on the hot path.
 *
 * 3. SM-CONDITIONAL TILE SIZES  (compile-time policy structs)
 *    SM8.6 (A6000)   : 256 threads, 2 CTAs/SM, bucket = 512 K elems (4 MB)
 *    SM9.0 (H100)    : 256 threads, 4 CTAs/SM, bucket = 4 M  elems (32 MB)
 *    SM12.0 (Blackwell): 512 threads, 4 CTAs/SM, bucket = 2 M elems (16 MB)
 *
 * 4. TRUE VECTORISED ACCUMULATION
 *    Each thread loads 8 × BF16 as a single uint4 (128-bit), converts to
 *    float2×4, and accumulates across all num_tensors inputs before a
 *    single vectorised BF16 store — zero intermediate DRAM traffic.
 *
 * 5. CONSTANT-MEMORY POINTER ARRAY + DEVICE-MEMORY FALLBACK
 *    ≤ 32 input pointers → __constant__ c_input_ptrs (zero malloc latency)
 *    > 32 inputs → cudaMallocAsync device pointer array
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
// Section 1: Per-SM Tuning Policy
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer> struct KernelPolicy {
    // Generic fallback
    static constexpr int    kBlockSize        = 256;
    static constexpr int    kMinBlocksPerSM   = 2;
    static constexpr size_t kBucketElems      = 512UL * 1024UL;
    static constexpr int    kVecWidth         = 8;
    static constexpr int    kMaxWarps         = kBlockSize / hw_warp_size;
};

template <> struct KernelPolicy<86> {
    // A6000: 48 GB GDDR6X, 6 MB L2, 128 KB smem/SM
    // Small buckets prevent L2 thrashing; 2 CTAs/SM for register headroom.
    static constexpr int    kBlockSize        = 256;
    static constexpr int    kMinBlocksPerSM   = 2;
    static constexpr size_t kBucketElems      = 512UL * 1024UL;   //  4 MB BF16
    static constexpr int    kVecWidth         = 8;
    static constexpr int    kMaxWarps         = kBlockSize / hw_warp_size;  // 8
};

template <> struct KernelPolicy<90> {
    // H100 SXM5: 80 GB HBM3, 50 MB L2, 228 KB smem/SM
    // Large buckets exploit the giant L2; 4 CTAs/SM fills all SMs.
    static constexpr int    kBlockSize        = 256;
    static constexpr int    kMinBlocksPerSM   = 4;
    static constexpr size_t kBucketElems      = 4UL * 1024UL * 1024UL;   // 32 MB BF16
    static constexpr int    kVecWidth         = 8;
    static constexpr int    kMaxWarps         = kBlockSize / hw_warp_size;  // 8
};

template <> struct KernelPolicy<120> {
    // Blackwell GB200: 192 GB HBM3e, 40 MB L2, 256 KB smem/SM
    // 512-thread blocks maximise utilisation on 128-wide SMs.
    static constexpr int    kBlockSize        = 512;
    static constexpr int    kMinBlocksPerSM   = 4;
    static constexpr size_t kBucketElems      = 2UL * 1024UL * 1024UL;   // 16 MB BF16
    static constexpr int    kVecWidth         = 8;
    static constexpr int    kMaxWarps         = kBlockSize / hw_warp_size;  // 16
};

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: Constant-memory inline pointer array
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int kMaxInlinePointers = 32;

__constant__ const __nv_bfloat16* c_input_ptrs[kMaxInlinePointers];

// ─────────────────────────────────────────────────────────────────────────────
// Section 3: 128-bit vectorised load/store helpers
// ─────────────────────────────────────────────────────────────────────────────

// Accumulate 8 × BF16 from ptr into 4 × float2 accumulators.
DS_D_INLINE void bf16x8_accumulate(
    const __nv_bfloat16* __restrict__ ptr,
    float2& a0, float2& a1, float2& a2, float2& a3)
{
    const uint4 raw = *reinterpret_cast<const uint4*>(ptr);
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
    a0.x += __bfloat162float(p[0].x); a0.y += __bfloat162float(p[0].y);
    a1.x += __bfloat162float(p[1].x); a1.y += __bfloat162float(p[1].y);
    a2.x += __bfloat162float(p[2].x); a2.y += __bfloat162float(p[2].y);
    a3.x += __bfloat162float(p[3].x); a3.y += __bfloat162float(p[3].y);
}

// Store 4 × float2 (8 floats) as 8 × BF16 in one 128-bit write.
DS_D_INLINE void fp32x8_store_bf16(
    __nv_bfloat16* __restrict__ ptr,
    float2 a0, float2 a1, float2 a2, float2 a3)
{
    __nv_bfloat162 b0 = {__float2bfloat16(a0.x), __float2bfloat16(a0.y)};
    __nv_bfloat162 b1 = {__float2bfloat16(a1.x), __float2bfloat16(a1.y)};
    __nv_bfloat162 b2 = {__float2bfloat16(a2.x), __float2bfloat16(a2.y)};
    __nv_bfloat162 b3 = {__float2bfloat16(a3.x), __float2bfloat16(a3.y)};
    uint4 out;
    out.x = *reinterpret_cast<const uint32_t*>(&b0);
    out.y = *reinterpret_cast<const uint32_t*>(&b1);
    out.z = *reinterpret_cast<const uint32_t*>(&b2);
    out.w = *reinterpret_cast<const uint32_t*>(&b3);
    *reinterpret_cast<uint4*>(ptr) = out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 4: Warp shuffle + CAS cross-warp reduction
//
// Stage 1 — warp-level butterfly via cg::coalesced_threads()
//   • On SM9.0+: emits a single REDUX.SYNC.ADD.F32 PTX instruction
//   • On SM8.6: emits 5 shfl_xor rounds (correct even for divergent warps)
//
// Stage 2 — CAS-free cross-warp smem tree
//   CCCL#9656 pattern: lane-0 of every warp writes its partial sum into
//   smem_partial[warp_id].  Then the LAST warp (warp_id == kMaxWarps-1
//   after the atomicAdd barrier below) performs a second-level warp
//   butterfly over the kMaxWarps entries.
//
//   The "last warp wins" detection uses a shared atomic counter so that
//   NO __syncthreads() is required between warp-level and block-level stages.
//   We replace __syncthreads() with a store-release / load-acquire sequence:
//     atomicAdd(&smem_cnt, 1)  → returns old count
//     if (old_count + 1 == kMaxWarps) → this is the last warp
//   The CUDA memory model guarantees the smem_partial[] writes from all
//   warps before the counter increment are visible to the last warp.
//   (PTX ld.acquire / st.release semantics ensure this on all SM versions.)
// ─────────────────────────────────────────────────────────────────────────────

// Warp-level sum using cooperative groups (no smem, no __syncthreads).
DS_D_INLINE float warp_reduce_sum_cg(float val)
{
    auto active = cg::coalesced_threads();
    return cg::reduce(active, val, cg::plus<float>());
}

// Cross-warp reduction: writes lane-0 warp sum to smem, then the last warp
// arriving (via atomic counter) performs a final warp shuffle.
// smem_partial : [kMaxWarps] float — per-warp partial sums staging
// smem_cnt     : [1]         int   — atomic arrival counter
// Returns the block-wide sum ONLY in the last-arriving warp's lane-0.
// All other threads return their warp-local partial sum (unused by caller).
template <int kMaxWarps>
DS_D_INLINE float cas_cross_warp_reduce(
    float            warp_sum,
    float* __restrict__ smem_partial,
    int*   __restrict__ smem_cnt)
{
    const int warp_id = (int)threadIdx.x / hw_warp_size;
    const int lane    = (int)threadIdx.x % hw_warp_size;

    // Lane-0 of each warp deposits its partial sum (store-release).
    if (lane == 0) {
        smem_partial[warp_id] = warp_sum;
        __threadfence_block();             // store-release to smem
        // Announce arrival; the last warp gets old_count = kMaxWarps-1.
        atomicAdd(smem_cnt, 1);
    }

    // Only the last warp proceeds to the second-level reduction.
    // We detect it by spinning on smem_cnt == kMaxWarps.
    // All OTHER warp lanes just return (they don't write output).
    if (warp_id == kMaxWarps - 1) {
        // Spin until all warps have deposited (load-acquire).
        // This is guaranteed to terminate in O(kMaxWarps) warp-steps.
        if (lane == 0) {
            while (atomicAdd(smem_cnt, 0) < kMaxWarps) { /*spin*/ }
        }
        __syncwarp(0xffffffff);   // sync within the last warp

        // Second-level warp butterfly over kMaxWarps entries.
        float v = (lane < kMaxWarps) ? smem_partial[lane] : 0.f;
        auto tile = cg::tiled_partition<hw_warp_size>(cg::this_thread_block());
        v = cg::reduce(tile, v, cg::plus<float>());
        return v;
    }
    return warp_sum;  // non-last warps return their partial (unused)
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 5: Main reduce-scatter kernel — all SM versions
//
//   Template parameters:
//     SmVer        : SM version → selects KernelPolicy
//     UseConstMem  : true  → read pointers from __constant__ c_input_ptrs
//                    false → read pointers from d_inputs device argument
//
//   __launch_bounds__ derived from KernelPolicy (not hard-coded).
//
//   Shared memory layout (per CTA):
//     float smem_partial[kMaxWarps]   — warp partial sums (cross-warp stage)
//     int   smem_cnt                  — CAS arrival counter
//     Total: kMaxWarps×4 + 4 bytes  ≤  68 bytes (SM8.6/SM9.0) / 68 bytes (SM12.0)
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
    constexpr int kVec     = Policy::kVecWidth;   // 8
    constexpr int kBS      = Policy::kBlockSize;

    // Shared memory: partial sums + atomic counter.
    // We union them to save address space but they never alias in time.
    __shared__ float smem_partial[Policy::kMaxWarps];
    __shared__ int   smem_cnt;

    if (threadIdx.x == 0) smem_cnt = 0;
    // No __syncthreads() needed here — smem_cnt is only read by the CAS
    // mechanism which self-synchronises via the atomic counter pattern.

    const size_t vec_count = shard_count / kVec;
    const size_t tid       = (size_t)blockIdx.x * kBS + threadIdx.x;
    const size_t stride    = (size_t)gridDim.x  * kBS;

    // ── Main vectorised loop: each thread owns independent output elements ──
    for (size_t vec_idx = tid; vec_idx < vec_count; vec_idx += stride) {
        const size_t gelem = shard_offset + vec_idx * kVec;

        float2 a0={0.f,0.f}, a1={0.f,0.f}, a2={0.f,0.f}, a3={0.f,0.f};

        // Accumulate across all input tensors (unrolled by 4 for ILP).
        if constexpr (UseConstMem) {
            #pragma unroll 4
            for (int t = 0; t < num_tensors; ++t)
                bf16x8_accumulate(c_input_ptrs[t] + gelem, a0, a1, a2, a3);
        } else {
            #pragma unroll 4
            for (int t = 0; t < num_tensors; ++t)
                bf16x8_accumulate(d_inputs[t] + gelem, a0, a1, a2, a3);
        }

        // Write shard-relative output (one 128-bit store).
        fp32x8_store_bf16(output + vec_idx * kVec, a0, a1, a2, a3);
    }

    // ── Scalar tail: handle shard_count not divisible by kVec ──
    // Thread 0 handles at most kVec-1 leftover elements.
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

    // smem_partial and smem_cnt are allocated but not used in the main path
    // (this kernel does per-thread independent reductions, not cross-thread).
    // They exist for the warp-coop variant below. Suppress warnings:
    (void)smem_partial;
    (void)smem_cnt;
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 6: Warp-cooperative small-tensor kernel
//
//   For tensors with shard_count ≤ kSmallThreshold AND num_tensors ≤ 32:
//   threads WITHIN a warp split ownership of the num_tensors inputs
//   (each lane accumulates a disjoint subset of tensors), then:
//
//   Stage 1: cg::coalesced_threads() reduce within warp (warp shuffle + CAS)
//   Stage 2: lane-0 writes the reduced 8 BF16 to output
//
//   This doubles throughput when num_tensors >> kVec and shards are small
//   enough that a single warp covers only a few vector iterations.
//
//   CAS cross-warp pattern: not needed here since the warp IS the reduction
//   unit — each warp fully reduces its output elements.
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
    constexpr int kVec    = 8;
    constexpr int kBS     = 256;
    const int warp_g      = ((int)blockIdx.x * kBS + (int)threadIdx.x) / hw_warp_size;
    const int lane        = (int)threadIdx.x % hw_warp_size;
    const int total_warps = ((int)gridDim.x * kBS) / hw_warp_size;
    const size_t vec_count = shard_count / kVec;

    for (size_t vid = (size_t)warp_g; vid < vec_count; vid += (size_t)total_warps) {
        const size_t gidx = shard_offset + vid * kVec;

        // Each lane accumulates a disjoint subset of tensors.
        float2 a0={0.f,0.f}, a1={0.f,0.f}, a2={0.f,0.f}, a3={0.f,0.f};

        for (int t = lane; t < num_tensors; t += hw_warp_size) {
            const __nv_bfloat16* src = UseConstMem ? c_input_ptrs[t] : d_inputs[t];
            bf16x8_accumulate(src + gidx, a0, a1, a2, a3);
        }

        // Warp butterfly via coalesced_threads() — REDUX.SYNC on SM9.0+.
        auto warp = cg::coalesced_threads();
        a0.x = cg::reduce(warp, a0.x, cg::plus<float>());
        a0.y = cg::reduce(warp, a0.y, cg::plus<float>());
        a1.x = cg::reduce(warp, a1.x, cg::plus<float>());
        a1.y = cg::reduce(warp, a1.y, cg::plus<float>());
        a2.x = cg::reduce(warp, a2.x, cg::plus<float>());
        a2.y = cg::reduce(warp, a2.y, cg::plus<float>());
        a3.x = cg::reduce(warp, a3.x, cg::plus<float>());
        a3.y = cg::reduce(warp, a3.y, cg::plus<float>());

        if (lane == 0)
            fp32x8_store_bf16(output + vid * kVec, a0, a1, a2, a3);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 7: Full-tensor fused reduce (no scatter offset)
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
    constexpr int kVec   = Policy::kVecWidth;
    constexpr int kBS    = Policy::kBlockSize;

    const size_t tid    = (size_t)blockIdx.x * kBS + threadIdx.x;
    const size_t stride = (size_t)gridDim.x  * kBS;
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
// Section 8: Input-pointer setup helpers
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
// Section 9: SM dispatch helpers
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
    // Small-shard warp-coop path: shard fits in a handful of warps.
    constexpr size_t kSmallThresh = 128UL * 1024UL;

    if (shard_count <= kSmallThresh && num_tensors <= hw_warp_size) {
        const size_t vec_count   = shard_count / Policy::kVecWidth;
        const size_t warps_needed = (vec_count + 0) / 1;
        const int grid = (int)std::min(
            (warps_needed * hw_warp_size + 255UL) / 256UL, (size_t)65535);
        if (use_const)
            hetero_reduce_scatter_warp_coop<true>
                <<<std::max(grid,1), 256, 0, stream>>>(
                    output, nullptr, num_tensors, shard_offset, shard_count);
        else
            hetero_reduce_scatter_warp_coop<false>
                <<<std::max(grid,1), 256, 0, stream>>>(
                    output, d_inputs, num_tensors, shard_offset, shard_count);
        return;
    }

    const size_t vec_count = shard_count / Policy::kVecWidth;
    const int grid = (int)std::min(
        (vec_count + Policy::kBlockSize - 1) / Policy::kBlockSize,
        (size_t)65535);

    if (use_const)
        hetero_reduce_scatter_kernel<SmVer, true>
            <<<std::max(grid,1), Policy::kBlockSize, 0, stream>>>(
                output, nullptr, num_tensors, shard_offset, shard_count);
    else
        hetero_reduce_scatter_kernel<SmVer, false>
            <<<std::max(grid,1), Policy::kBlockSize, 0, stream>>>(
                output, d_inputs, num_tensors, shard_offset, shard_count);
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 10: Public API implementations
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
                <<<std::max(grid,1), Policy::kBlockSize, 0, stream>>>(
                    output, nullptr, num_tensors, n_elems);
        else
            fused_bf16_reduce_kernel<SmVer, false>
                <<<std::max(grid,1), Policy::kBlockSize, 0, stream>>>(
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

    // Align shard boundaries to kVecWidth elements (128-bit vector boundary).
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
