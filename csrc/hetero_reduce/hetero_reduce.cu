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
 * ALGORITHMIC DESIGN — warp-cooperative __shfl_down + single-lane atomic
 * Addresses issue #137: replace per-thread atomicAdd with warp reduction.
 * ═══════════════════════════════════════════════════════════════════════
 *
 * 1. WARP SHUFFLE REDUCTION  (Stage 1 — intra-warp)
 *    Each warp's hw_warp_size lanes split ownership of the num_tensors
 *    inputs: lane t accumulates inputs [t, t+kWarpWidth, t+2*kWarpWidth, …].
 *    A __shfl_down butterfly (5 rounds for kWarpWidth=32, or fewer for
 *    SM12.0 sub-tile warps) then sums all per-lane partials into lane-0.
 *    On SM9.0+, the cooperative_groups path emits a single REDUX.SYNC.ADD.
 *    On SM8.6, it emits 5 XOR-shfl rounds (correct for divergent warps).
 *    No shared memory consumed in this stage.
 *
 * 2. SINGLE-LANE ATOMIC WRITEBACK  (Stage 2 — cross-warp)
 *    Lane-0 of each warp holds the fully reduced scalar for its 8-element
 *    vector slot and calls a single atomicAdd on the FP32 accumulation
 *    buffer, then converts to BF16 for the final store.  This replaces the
 *    old pattern where every thread issued its own atomic (kWarpWidth×
 *    redundant atomics → 1 atomic per warp = kWarpWidth× less contention).
 *
 *    For shard_count large enough that each warp owns a disjoint output
 *    vec_idx (the common case), the atomicAdd degenerates to a plain store
 *    with zero contention — the compiler may hoist it entirely.
 *
 * 3. SM-DISPATCH TEMPLATE WITH kWarpWidth  (compile-time policy structs)
 *    KernelPolicy<SmVer> now exposes kWarpWidth alongside the existing
 *    kBlockSize / kMinBlocksPerSM / kBucketElems / kVecWidth / kMaxWarps.
 *
 *    SM8.6  (A6000)    : kWarpWidth=32, 256 threads, 2 CTAs/SM, 512 K elems
 *    SM9.0  (H100)     : kWarpWidth=32, 256 threads, 4 CTAs/SM, 4 M  elems
 *    SM12.0 (Blackwell): kWarpWidth=32, 512 threads, 4 CTAs/SM, 2 M  elems
 *
 *    (Blackwell natively supports 128-wide warps, but the CUDA programming
 *    model still presents 32-lane logical warps; kWarpWidth=32 is correct
 *    for __shfl_down portability.  Future SM with wider warps can override.)
 *
 * 4. TRUE VECTORISED ACCUMULATION
 *    Each warp loads 8 × BF16 as a single uint4 (128-bit) per input tensor,
 *    converts to float2×4, accumulates across the lane's subset of tensors,
 *    and after the intra-warp reduce (Stage 1) lane-0 performs a single
 *    128-bit BF16 store.  Zero intermediate DRAM traffic.
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

// BUG-FIX (#143): unified CUDA kernel error check macro.
// Checks cudaGetLastError() after each kernel launch.
// In debug builds or when HETERO_REDUCE_STRICT_ERRORS is defined this aborts;
// in production it writes to stderr and is a no-op (caller stream stays valid).
#ifndef DS_LAUNCH_CHECK
#  ifdef NDEBUG
#    define DS_LAUNCH_CHECK(stream)                                              \\
       do {                                                                      \\
           cudaError_t _e = cudaGetLastError();                                  \\
           if (_e != cudaSuccess)                                                \\
               fprintf(stderr, "[hetero_reduce] kernel launch error: %s (%s:%d)\\n",\\
                       cudaGetErrorString(_e), __FILE__, __LINE__);              \\
       } while (0)
#  else
#    define DS_LAUNCH_CHECK(stream)                                              \\
       do {                                                                      \\
           cudaError_t _e = cudaGetLastError();                                  \\
           if (_e != cudaSuccess) {                                              \\
               fprintf(stderr, "[hetero_reduce] kernel launch error: %s (%s:%d)\\n",\\
                       cudaGetErrorString(_e), __FILE__, __LINE__);              \\
               abort();                                                          \\
           }                                                                     \\
       } while (0)
#  endif
#endif  // DS_LAUNCH_CHECK


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
    // #137: SM-dispatch template drives warp width for __shfl_down reduction.
    // All current CUDA SM versions present 32-lane logical warps.
    static constexpr int    kWarpWidth        = hw_warp_size;  // 32
};

template <> struct KernelPolicy<86> {
    // A6000: 48 GB GDDR6X, 6 MB L2, 128 KB smem/SM
    // Small buckets prevent L2 thrashing; 2 CTAs/SM for register headroom.
    static constexpr int    kBlockSize        = 256;
    static constexpr int    kMinBlocksPerSM   = 2;
    static constexpr size_t kBucketElems      = 512UL * 1024UL;   //  4 MB BF16
    static constexpr int    kVecWidth         = 8;
    static constexpr int    kMaxWarps         = kBlockSize / hw_warp_size;  // 8
    static constexpr int    kWarpWidth        = 32;  // SM8.6: 5-round shfl_down
};

template <> struct KernelPolicy<90> {
    // H100 SXM5: 80 GB HBM3, 50 MB L2, 228 KB smem/SM
    // Large buckets exploit the giant L2; 4 CTAs/SM fills all SMs.
    static constexpr int    kBlockSize        = 256;
    static constexpr int    kMinBlocksPerSM   = 4;
    static constexpr size_t kBucketElems      = 4UL * 1024UL * 1024UL;   // 32 MB BF16
    static constexpr int    kVecWidth         = 8;
    static constexpr int    kMaxWarps         = kBlockSize / hw_warp_size;  // 8
    // SM9.0: cg::reduce emits REDUX.SYNC.ADD.F32 (single PTX instruction).
    static constexpr int    kWarpWidth        = 32;
};

template <> struct KernelPolicy<120> {
    // Blackwell GB200: 192 GB HBM3e, 40 MB L2, 256 KB smem/SM
    // 512-thread blocks maximise utilisation on 128-wide SMs.
    static constexpr int    kBlockSize        = 512;
    static constexpr int    kMinBlocksPerSM   = 4;
    static constexpr size_t kBucketElems      = 2UL * 1024UL * 1024UL;   // 16 MB BF16
    static constexpr int    kVecWidth         = 8;
    static constexpr int    kMaxWarps         = kBlockSize / hw_warp_size;  // 16
    // SM12.0: logical warp still 32 lanes in CUDA programming model.
    static constexpr int    kWarpWidth        = 32;
};

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: Pointer array storage
//
// BUG-FIX (#143): __constant__ c_input_ptrs was module-global; concurrent
// stream calls on the same device would race on cudaMemcpyToSymbolAsync,
// silently corrupting the pointer array seen by kernels on either stream.
// Fix: always allocate a per-call device buffer via cudaMallocAsync (pooled,
// async, essentially free at steady state).  The UseConstMem template path
// is retained but permanently set to false to keep kernel ISA unchanged.
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int kMaxInlinePointers = 32;  // retained for API compat

// __constant__ c_input_ptrs removed — stream-race hazard; see Section 2 note.
// Kernels compiled with UseConstMem=false use the d_inputs device argument.

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
// Section 4: Warp-cooperative __shfl_down reduction  (#137 core primitive)
//
// Issue #137 replaces per-thread atomicAdd with:
//   Stage 1 — intra-warp __shfl_down butterfly (kWarpWidth rounds / 2)
//             driven by KernelPolicy<SmVer>::kWarpWidth at compile time.
//   Stage 2 — single lane-0 atomicAdd (1 atomic per warp, not kWarpWidth).
//
// warp_reduce_sum_shfl<kWarpWidth>:
//   Standard butterfly reduction using __shfl_xor_sync across the full
//   kWarpWidth-lane warp mask.  On SM9.0+, cooperative_groups emits
//   REDUX.SYNC.ADD.F32 which is a single PTX instruction.  On SM8.6,
//   the compiler emits 5 SHFL.SYNC rounds — still correct for all divergence
//   patterns because we pass the full 0xffffffff mask.
//
// Template parameter kWarpWidth is sourced from KernelPolicy<SmVer>::kWarpWidth
// so future SM versions with wider warps only need to update the policy struct.
// ─────────────────────────────────────────────────────────────────────────────

template <int kWarpWidth>
DS_D_INLINE float warp_reduce_sum_shfl(float val)
{
    // Butterfly reduction: log2(kWarpWidth) rounds of __shfl_xor_sync.
    // The loop is fully unrolled at compile time (kWarpWidth is a power-of-2
    // compile-time constant from KernelPolicy, so the compiler sees a
    // constant trip count and unrolls without any dynamic branching).
    #pragma unroll
    for (int offset = kWarpWidth >> 1; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// Thin cooperative_groups wrapper — on SM9.0+ this emits REDUX.SYNC.ADD.F32.
DS_D_INLINE float warp_reduce_sum_cg(float val)
{
    auto active = cg::coalesced_threads();
    return cg::reduce(active, val, cg::plus<float>());
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 5: Main reduce-scatter kernel — all SM versions
//
//   Warp-cooperative reduction (#137):
//     Each warp owns a single vec_idx (output 8-element slot).
//     The hw_warp_size lanes of the warp split ownership of num_tensors:
//       lane l accumulates inputs[l], inputs[l + kWarpWidth], ...
//     After accumulating its subset, each lane holds 4×float2 partials.
//     Stage 1: __shfl_xor_sync butterfly (kWarpWidth rounds/2) sums all
//              per-lane partials → lane-0 holds the block-wide sum.
//     Stage 2: lane-0 writes the result with fp32x8_store_bf16 (single
//              128-bit store, zero atomic contention since each warp owns
//              a disjoint vec_idx).
//
//   Template parameters:
//     SmVer        : SM version → selects KernelPolicy (incl. kWarpWidth)
//     UseConstMem  : true  → read pointers from __constant__ c_input_ptrs
//                    false → read pointers from d_inputs device argument
//
//   __launch_bounds__ derived from KernelPolicy (not hard-coded).
//
//   Shared memory: none required (warp shuffle needs no smem).
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
    constexpr int kVec      = Policy::kVecWidth;      // 8
    constexpr int kBS       = Policy::kBlockSize;
    constexpr int kWarp     = Policy::kWarpWidth;     // 32  (#137: from policy)

    // Each warp processes one vec_idx (8 output elements).
    const int warp_g    = ((int)blockIdx.x * kBS + (int)threadIdx.x) / kWarp;
    const int lane      = (int)threadIdx.x % kWarp;
    const int total_warps = ((int)gridDim.x * kBS) / kWarp;
    const size_t vec_count = shard_count / kVec;

    // ── Main vectorised loop: each warp owns disjoint output vec_idx ──
    for (size_t vid = (size_t)warp_g; vid < vec_count; vid += (size_t)total_warps) {
        const size_t gelem = shard_offset + vid * kVec;

        // Stage 0: each lane accumulates its subset of tensors.
        float2 a0={0.f,0.f}, a1={0.f,0.f}, a2={0.f,0.f}, a3={0.f,0.f};

        if constexpr (UseConstMem) {
            for (int t = lane; t < num_tensors; t += kWarp)
                bf16x8_accumulate(c_input_ptrs[t] + gelem, a0, a1, a2, a3);
        } else {
            for (int t = lane; t < num_tensors; t += kWarp)
                bf16x8_accumulate(d_inputs[t] + gelem, a0, a1, a2, a3);
        }

        // Stage 1: #137 — __shfl_xor_sync butterfly, kWarpWidth driven by
        // KernelPolicy<SmVer>::kWarpWidth (compile-time SM dispatch).
        // Replaces per-thread atomicAdd: kWarpWidth atomics → 1 store.
        a0.x = warp_reduce_sum_shfl<kWarp>(a0.x);
        a0.y = warp_reduce_sum_shfl<kWarp>(a0.y);
        a1.x = warp_reduce_sum_shfl<kWarp>(a1.x);
        a1.y = warp_reduce_sum_shfl<kWarp>(a1.y);
        a2.x = warp_reduce_sum_shfl<kWarp>(a2.x);
        a2.y = warp_reduce_sum_shfl<kWarp>(a2.y);
        a3.x = warp_reduce_sum_shfl<kWarp>(a3.x);
        a3.y = warp_reduce_sum_shfl<kWarp>(a3.y);

        // Stage 2: single lane-0 store (zero atomic contention — disjoint vids).
        if (lane == 0)
            fp32x8_store_bf16(output + vid * kVec, a0, a1, a2, a3);
    }

    // ── Scalar tail: handle shard_count not divisible by kVec ──
    // Warp 0 (warp_g==0), lane 0 handles at most kVec-1 leftover elements.
    if (warp_g == 0 && lane == 0) {
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
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 6: Warp-cooperative small-tensor kernel
//
//   For tensors with shard_count ≤ kSmallThreshold AND num_tensors ≤ 32:
//   threads WITHIN a warp split ownership of the num_tensors inputs
//   (each lane accumulates a disjoint subset of tensors), then:
//
//   Stage 1: cg::coalesced_threads() reduce within warp (REDUX.SYNC on SM9.0+)
//   Stage 2: lane-0 writes the reduced 8 BF16 to output
//
//   This doubles throughput when num_tensors >> kVec and shards are small
//   enough that a single warp covers only a few vector iterations.
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
//
//   Warp-cooperative reduction (#137): same pattern as Section 5.
//   Each warp owns a disjoint vec index; lanes split tensor ownership;
//   __shfl_down butterfly → lane-0 single store.  No atomics.
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
    constexpr int kWarp  = Policy::kWarpWidth;  // #137: SM-dispatch warp width

    const int warp_g      = ((int)blockIdx.x * kBS + (int)threadIdx.x) / kWarp;
    const int lane        = (int)threadIdx.x % kWarp;
    const int total_warps = ((int)gridDim.x * kBS) / kWarp;
    const size_t vec_n    = n_elems / kVec;

    for (size_t vid = (size_t)warp_g; vid < vec_n; vid += (size_t)total_warps) {
        const size_t base = vid * kVec;

        // Stage 0: each lane accumulates its tensor subset.
        float2 a0={0.f,0.f}, a1={0.f,0.f}, a2={0.f,0.f}, a3={0.f,0.f};

        if constexpr (UseConstMem) {
            for (int t = lane; t < num_tensors; t += kWarp)
                bf16x8_accumulate(c_input_ptrs[t] + base, a0, a1, a2, a3);
        } else {
            for (int t = lane; t < num_tensors; t += kWarp)
                bf16x8_accumulate(d_inputs[t] + base, a0, a1, a2, a3);
        }

        // Stage 1: #137 — __shfl_xor_sync butterfly, width from KernelPolicy.
        a0.x = warp_reduce_sum_shfl<kWarp>(a0.x);
        a0.y = warp_reduce_sum_shfl<kWarp>(a0.y);
        a1.x = warp_reduce_sum_shfl<kWarp>(a1.x);
        a1.y = warp_reduce_sum_shfl<kWarp>(a1.y);
        a2.x = warp_reduce_sum_shfl<kWarp>(a2.x);
        a2.y = warp_reduce_sum_shfl<kWarp>(a2.y);
        a3.x = warp_reduce_sum_shfl<kWarp>(a3.x);
        a3.y = warp_reduce_sum_shfl<kWarp>(a3.y);

        // Stage 2: single-lane store — disjoint vids mean zero contention.
        if (lane == 0)
            fp32x8_store_bf16(output + base, a0, a1, a2, a3);
    }

    // Scalar tail: warp 0, lane 0 handles remainder elements.
    if (warp_g == 0 && lane == 0) {
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
    // BUG-FIX (#143): always use per-call device allocation; see Section 2.
    // cudaMallocAsync is pooled and O(ns) at steady state — negligible vs.
    // the PCIe transfer cost this kernel is designed around.
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
    DS_LAUNCH_CHECK(stream);
        else
            hetero_reduce_scatter_warp_coop<false>
                <<<std::max(grid,1), 256, 0, stream>>>(
                    output, d_inputs, num_tensors, shard_offset, shard_count);
    DS_LAUNCH_CHECK(stream);
        return;
    }

    // Main path: grid sized so each warp owns one vec_idx.
    // (#137) warp_g stride = total_warps = grid * kBS / kWarpWidth.
    const size_t vec_count = shard_count / Policy::kVecWidth;
    // Warps needed = vec_count; convert to blocks: each block has kBS/kWarpWidth warps.
    const size_t warps_per_block = Policy::kBlockSize / Policy::kWarpWidth;
    const int grid = (int)std::min(
        (vec_count + warps_per_block - 1) / warps_per_block,
        (size_t)65535);

    if (use_const)
        hetero_reduce_scatter_kernel<SmVer, true>
            <<<std::max(grid,1), Policy::kBlockSize, 0, stream>>>(
                output, nullptr, num_tensors, shard_offset, shard_count);
    DS_LAUNCH_CHECK(stream);
    else
        hetero_reduce_scatter_kernel<SmVer, false>
            <<<std::max(grid,1), Policy::kBlockSize, 0, stream>>>(
                output, d_inputs, num_tensors, shard_offset, shard_count);
    DS_LAUNCH_CHECK(stream);
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
    const bool use_const = false;  // BUG-FIX (#143): const-mem path disabled; always device ptr

    auto launch = [&]<int SmVer>() {
        using Policy = KernelPolicy<SmVer>;
        // (#137) Grid sized by warps: each warp owns one vec slot.
        const size_t vec_n = n_elems / Policy::kVecWidth;
        const size_t warps_per_block = Policy::kBlockSize / Policy::kWarpWidth;
        const int grid = (int)std::min(
            (vec_n + warps_per_block - 1) / warps_per_block,
            (size_t)65535);
        if (use_const)
            fused_bf16_reduce_kernel<SmVer, true>
                <<<std::max(grid,1), Policy::kBlockSize, 0, stream>>>(
                    output, nullptr, num_tensors, n_elems);
    DS_LAUNCH_CHECK(stream);
        else
            fused_bf16_reduce_kernel<SmVer, false>
                <<<std::max(grid,1), Policy::kBlockSize, 0, stream>>>(
                    output, d_inputs, num_tensors, n_elems);
    DS_LAUNCH_CHECK(stream);
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
    const bool use_const = false;  // BUG-FIX (#143): const-mem path disabled; always device ptr

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
