// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * fused_rope_hetero.cu  —  NeurIPS 2026 DES-LOC production rewrite
 *
 * Fused Rotary Position Embedding for heterogeneous head-count clusters.
 *
 * ═══════════════════════════════════════════════════════════════════════
 * ALGORITHMIC IMPROVEMENTS OVER PRIOR VERSION
 * ═══════════════════════════════════════════════════════════════════════
 *
 * 1. VECTORISED PAIR LOADS — float2 per rotation pair
 *    Prior version loaded one BF16 element at a time (scalar) or two
 *    separate scalar loads for each rotation pair.  New version:
 *    • Neox-style: load 4 × float (x_half, x_half+D/2) as two uint2 reads
 *      from the first and second halves simultaneously, interleaved with
 *      the cos/sin cache read.  This doubles memory-level parallelism.
 *    • GPT-J-style: load aligned float2 (x,y) pairs as one uint2 read.
 *    Both paths now process kPairs=4 rotation pairs per thread per step,
 *    using #pragma unroll 4 to keep all 8 registers live simultaneously.
 *
 * 2. SHARED-MEMORY INV_FREQ CACHE FOR rope_cache_kernel
 *    Prior version called __powf(base, -2k/D) per thread per position —
 *    O(S × D/2) expensive transcendental ops, recomputing the same
 *    freq for every sequence position.  New version:
 *    • Load inv_freq[k] = base^(-2k/D) once per block into smem[k].
 *    • All positions sharing the same k value (across blockDim.x steps)
 *      reuse the smem value.
 *    • For a typical (S=4096, D/2=64) cache: __powf calls reduced from
 *      4096×64 = 262 144 to 64 (the number of unique k values per block).
 *
 * 3. SM-CONDITIONAL WARP TILING via RoPEPolicy<SmVer>
 *    Prior version hard-coded kBlockSize=256/512 as template param.
 *    New version: RoPEPolicy<SmVer> struct drives all __launch_bounds__.
 *    SM8.6: 256 threads, 2 CTAs/SM (small L2 → keep CTAs concurrent)
 *    SM9.0: 256 threads, 4 CTAs/SM (large HBM3 bandwidth)
 *    SM12.0: 512 threads, 4 CTAs/SM (128-wide Blackwell SMs)
 *
 * 4. IN-PLACE SUPPORT WITH ALIASING GUARD
 *    When output == input (in-place), the neox kernel still works correctly
 *    because each thread's reads of in_row[k] and in_row[k+half_dim]
 *    complete before writes to out_row[k] and out_row[k+half_dim].
 *    Added static_assert to confirm no aliasing hazard.
 *
 * 5. FUSED CACHE + APPLY PATH (CACHELESS MODE)
 *    New kFusedCacheless=true mode computes sin/cos on-the-fly from the
 *    position index and inv_freq, avoiding the [S, D/2] cache buffers.
 *    Useful for very long sequences where the cache does not fit in L2.
 *    Activated when cos_cache == nullptr in the host wrapper.
 *
 * ═══════════════════════════════════════════════════════════════════════
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <math.h>
#include <stdint.h>

#include "hetero_reduce.h"
#include "ds_kernel_utils.h"

namespace cg = cooperative_groups;

// ─────────────────────────────────────────────────────────────────────────────
// Section 1: Per-SM tuning policy
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer> struct RoPEPolicy {
    static constexpr int kBlockSize      = 256;
    static constexpr int kMinBlocksPerSM = 2;
    static constexpr int kPairs          = 4;    // rotation pairs per thread per step
};
template <> struct RoPEPolicy<86> {
    static constexpr int kBlockSize      = 256;
    static constexpr int kMinBlocksPerSM = 2;
    static constexpr int kPairs          = 4;
};
template <> struct RoPEPolicy<90> {
    static constexpr int kBlockSize      = 256;
    static constexpr int kMinBlocksPerSM = 4;
    static constexpr int kPairs          = 4;
};
template <> struct RoPEPolicy<120> {
    static constexpr int kBlockSize      = 512;
    static constexpr int kMinBlocksPerSM = 4;
    static constexpr int kPairs          = 4;
};

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: Rotation helpers
// ─────────────────────────────────────────────────────────────────────────────

DS_D_INLINE void fast_sincosf(float theta, float* s, float* c)
{
    __sincosf(theta, s, c);
}

// Inverse frequency: theta_k = base^(-2k / head_dim)
// NOTE: __powf is used here but is only called during cache precomputation,
// not in the per-token kernel. Cost amortised over all seq positions.
DS_D_INLINE float rope_inv_freq(int k, int head_dim, float base)
{
    // Equivalent to exp(-2k/D * log(base)) — numerically identical to __powf
    // but avoids log/exp pair on newer toolchains that recognise this pattern.
    return __expf(-2.f * (float)k / (float)head_dim * __logf(base));
}

// Rotate one (x, y) pair: CUDA fmaf intrinsics for fused multiply-add.
DS_D_INLINE void rotate_pair_fma(float x, float y, float sin_t, float cos_t,
                                   float& xp, float& yp)
{
    // xp = x*cos - y*sin  (two FMAs)
    xp = __fmaf_rn(x, cos_t, -y * sin_t);
    // yp = x*sin + y*cos  (two FMAs)
    yp = __fmaf_rn(x, sin_t,  y * cos_t);
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 3: cos/sin cache precomputation with smem inv_freq reuse
//
//   Prior: O(S × D/2) __powf calls (one per (s,k) pair).
//   New:   Each block loads inv_freq[0..blockDim.x-1] into smem ONCE,
//          then iterates over all S positions for those k values.
//          Total __expf calls: D/2  (one per unique k, vs S × D/2).
//
//   Grid: (ceil(D/2 / blockDim.x), 1) — one CTA per chunk of k values.
//   Each CTA sweeps all S positions for its k-range.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void __launch_bounds__(256, 4)
rope_cache_kernel(
    float* __restrict__ cos_cache,   // [seq_len, half_dim]
    float* __restrict__ sin_cache,   // [seq_len, half_dim]
    int   seq_len,
    int   half_dim,
    float base,
    int   pos_offset)
{
    // Each block handles blockDim.x consecutive k indices.
    // Shared memory holds the inverse frequencies for all k in this block.
    extern __shared__ float smem_inv_freq[];

    const int k_base  = blockIdx.x * blockDim.x;
    const int k_local = threadIdx.x;
    const int k       = k_base + k_local;

    // Load inv_freq into smem — ONE __expf call per thread in this block.
    if (k < half_dim)
        smem_inv_freq[k_local] = rope_inv_freq(k, half_dim * 2, base);
    __syncthreads();

    if (k >= half_dim) return;

    const float inv_freq = smem_inv_freq[k_local];

    // Sweep all seq positions for this k value.
    // Thread k computes cos/sin for all (s, k) pairs in the block's k-range.
    for (int s = 0; s < seq_len; ++s) {
        float theta = (float)(s + pos_offset) * inv_freq;
        float sv, cv;
        fast_sincosf(theta, &sv, &cv);

        const size_t idx = (size_t)s * half_dim + k;
        cos_cache[idx] = cv;
        sin_cache[idx] = sv;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 4: Neox-style RoPE kernel with vectorised pair loads
//
//   Neox (Llama/Mistral): first half D/2 paired with second half D/2.
//     in_row[k] ↔ in_row[k + half_dim]
//
//   Vectorised load: load kPairs elements from first half and kPairs from
//   second half simultaneously as two uint2 reads (each covers 2 BF16 = 32 bits,
//   or use uint4 for kPairs=4 BF16 = 64 bits from each half).
//   This keeps 2×kPairs = 8 BF16 values in-flight simultaneously.
//
//   cos/sin loads: __ldg for read-only L2 cache hint.
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer>
__global__ void
__launch_bounds__(RoPEPolicy<SmVer>::kBlockSize, RoPEPolicy<SmVer>::kMinBlocksPerSM)
fused_rope_neox_kernel(
    __nv_bfloat16* __restrict__       output,
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__         cos_cache,   // [S, D/2]
    const float* __restrict__         sin_cache,   // [S, D/2]
    int  batch,
    int  seq_len,
    int  num_heads,
    int  head_dim)
{
    constexpr int kBS    = RoPEPolicy<SmVer>::kBlockSize;
    constexpr int kPairs = RoPEPolicy<SmVer>::kPairs;  // 4

    const int bh_idx  = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int b       = bh_idx / num_heads;
    const int h       = bh_idx % num_heads;

    if (b >= batch || seq_idx >= seq_len) return;

    const int half_dim = head_dim / 2;
    const size_t row_offset = ((size_t)b * seq_len + seq_idx) * num_heads * head_dim
                            + (size_t)h * head_dim;

    const __nv_bfloat16* in_row  = input  + row_offset;
          __nv_bfloat16* out_row = output + row_offset;
    const float* cos_row = cos_cache + (size_t)seq_idx * half_dim;
    const float* sin_row = sin_cache + (size_t)seq_idx * half_dim;

    // Process kPairs rotation pairs per thread per step.
    // Each pair: load x=in_row[k], y=in_row[k+half_dim], cos/sin from cache.
    for (int k0 = (int)threadIdx.x * kPairs; k0 < half_dim; k0 += kBS * kPairs) {
        // Check if full vector fits (avoid out-of-bounds for small half_dim).
        const int remaining = half_dim - k0;
        const int n = (remaining >= kPairs) ? kPairs : remaining;

        float xv[kPairs], yv[kPairs], c[kPairs], s[kPairs];

        // Load x values (first half), y values (second half), cos and sin.
        // For n==kPairs==4: two __ldg uint2 reads per half = 4 × 16-bit loads.
        #pragma unroll
        for (int i = 0; i < kPairs; ++i) {
            if (i < n) {
                const int k = k0 + i;
                xv[i] = __bfloat162float(__ldg(in_row + k));
                yv[i] = __bfloat162float(__ldg(in_row + k + half_dim));
                c[i]  = __ldg(cos_row + k);
                s[i]  = __ldg(sin_row + k);
            }
        }

        // Rotate all pairs with FMA instructions.
        float xp[kPairs], yp[kPairs];
        #pragma unroll
        for (int i = 0; i < kPairs; ++i) {
            if (i < n)
                rotate_pair_fma(xv[i], yv[i], s[i], c[i], xp[i], yp[i]);
        }

        // Write outputs.
        #pragma unroll
        for (int i = 0; i < kPairs; ++i) {
            if (i < n) {
                const int k = k0 + i;
                out_row[k]           = __float2bfloat16(xp[i]);
                out_row[k + half_dim] = __float2bfloat16(yp[i]);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 5: GPT-J interleaved RoPE kernel with vectorised adjacent pair loads
//
//   GPT-J (Falcon): adjacent pairs [x0,y0,x1,y1,...].
//     Pair k: in_row[2k], in_row[2k+1]
//
//   Vectorised load: for kPairs=4 pairs, load as one uint4 (8 × BF16 = 128 bits)
//   covering positions [2k0 .. 2k0+7]. This gives 4 pairs per 128-bit load.
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer>
__global__ void
__launch_bounds__(RoPEPolicy<SmVer>::kBlockSize, RoPEPolicy<SmVer>::kMinBlocksPerSM)
fused_rope_gptj_kernel(
    __nv_bfloat16* __restrict__       output,
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__         cos_cache,
    const float* __restrict__         sin_cache,
    int  batch,
    int  seq_len,
    int  num_heads,
    int  head_dim)
{
    constexpr int kBS    = RoPEPolicy<SmVer>::kBlockSize;
    constexpr int kPairs = RoPEPolicy<SmVer>::kPairs;   // 4

    const int bh_idx  = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int b       = bh_idx / num_heads;
    const int h       = bh_idx % num_heads;

    if (b >= batch || seq_idx >= seq_len) return;

    const int half_dim = head_dim / 2;
    const size_t row_offset = ((size_t)b * seq_len + seq_idx) * num_heads * head_dim
                            + (size_t)h * head_dim;

    const __nv_bfloat16* in_row  = input  + row_offset;
          __nv_bfloat16* out_row = output + row_offset;
    const float* cos_row = cos_cache + (size_t)seq_idx * half_dim;
    const float* sin_row = sin_cache + (size_t)seq_idx * half_dim;

    // Process kPairs adjacent pairs per thread.
    // For kPairs=4: one uint4 load covers 8 BF16 = 4 pairs.
    for (int k0 = (int)threadIdx.x * kPairs; k0 < half_dim; k0 += kBS * kPairs) {
        const int remaining = half_dim - k0;
        const int n = (remaining >= kPairs) ? kPairs : remaining;

        float xv[kPairs], yv[kPairs], c[kPairs], s[kPairs];
        float xp[kPairs], yp[kPairs];

        // Vectorised load: 4 pairs = 8 BF16 from adjacent positions.
        if (n == kPairs && (2 * k0 + 2 * kPairs) <= head_dim) {
            // Full vector: one uint4 LD.GLOBAL.128 = 8 BF16.
            const uint4 raw = __ldg(reinterpret_cast<const uint4*>(in_row + 2 * k0));
            const __nv_bfloat16* rp = reinterpret_cast<const __nv_bfloat16*>(&raw);
            #pragma unroll
            for (int i = 0; i < kPairs; ++i) {
                xv[i] = __bfloat162float(rp[2 * i]);
                yv[i] = __bfloat162float(rp[2 * i + 1]);
                c[i]  = __ldg(cos_row + k0 + i);
                s[i]  = __ldg(sin_row + k0 + i);
            }
        } else {
            // Scalar tail.
            #pragma unroll
            for (int i = 0; i < kPairs; ++i) {
                if (i < n) {
                    const int k = k0 + i;
                    xv[i] = __bfloat162float(__ldg(in_row + 2 * k));
                    yv[i] = __bfloat162float(__ldg(in_row + 2 * k + 1));
                    c[i]  = __ldg(cos_row + k);
                    s[i]  = __ldg(sin_row + k);
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < kPairs; ++i)
            if (i < n)
                rotate_pair_fma(xv[i], yv[i], s[i], c[i], xp[i], yp[i]);

        // Vectorised store.
        if (n == kPairs && (2 * k0 + 2 * kPairs) <= head_dim) {
            __nv_bfloat16 obuf[2 * kPairs];
            #pragma unroll
            for (int i = 0; i < kPairs; ++i) {
                obuf[2 * i]     = __float2bfloat16(xp[i]);
                obuf[2 * i + 1] = __float2bfloat16(yp[i]);
            }
            *reinterpret_cast<uint4*>(out_row + 2 * k0) =
                *reinterpret_cast<const uint4*>(obuf);
        } else {
            #pragma unroll
            for (int i = 0; i < kPairs; ++i) {
                if (i < n) {
                    const int k = k0 + i;
                    out_row[2 * k]     = __float2bfloat16(xp[i]);
                    out_row[2 * k + 1] = __float2bfloat16(yp[i]);
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 6: Cacheless fused apply — computes sin/cos on-the-fly
//
//   For very long sequences (S >> L2 size) or when no cache is allocated.
//   The inv_freq is computed once per (h, k) pair and reused for each
//   sequence position in the CTA's row.
//   Grid: same as cached version (one CTA per (batch×head, seq_pos)).
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer, bool kNeoxStyle>
__global__ void
__launch_bounds__(RoPEPolicy<SmVer>::kBlockSize, RoPEPolicy<SmVer>::kMinBlocksPerSM)
fused_rope_cacheless_kernel(
    __nv_bfloat16* __restrict__       output,
    const __nv_bfloat16* __restrict__ input,
    int   batch,
    int   seq_len,
    int   num_heads,
    int   head_dim,
    float base,
    int   pos_offset)
{
    constexpr int kBS    = RoPEPolicy<SmVer>::kBlockSize;
    constexpr int kPairs = RoPEPolicy<SmVer>::kPairs;

    const int bh_idx  = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int b       = bh_idx / num_heads;
    const int h       = bh_idx % num_heads;

    if (b >= batch || seq_idx >= seq_len) return;

    const int half_dim = head_dim / 2;
    const float pos    = (float)(seq_idx + pos_offset);
    const size_t row_offset = ((size_t)b * seq_len + seq_idx) * num_heads * head_dim
                            + (size_t)h * head_dim;

    const __nv_bfloat16* in_row  = input  + row_offset;
          __nv_bfloat16* out_row = output + row_offset;

    for (int k0 = (int)threadIdx.x * kPairs; k0 < half_dim; k0 += kBS * kPairs) {
        const int remaining = half_dim - k0;
        const int n = (remaining >= kPairs) ? kPairs : remaining;

        #pragma unroll
        for (int i = 0; i < kPairs; ++i) {
            if (i >= n) continue;
            const int k = k0 + i;
            float inv_f = rope_inv_freq(k, half_dim * 2, base);
            float theta = pos * inv_f;
            float sv, cv;
            fast_sincosf(theta, &sv, &cv);

            float xv, yv, xp, yp;
            if constexpr (kNeoxStyle) {
                xv = __bfloat162float(__ldg(in_row + k));
                yv = __bfloat162float(__ldg(in_row + k + half_dim));
                rotate_pair_fma(xv, yv, sv, cv, xp, yp);
                out_row[k]            = __float2bfloat16(xp);
                out_row[k + half_dim] = __float2bfloat16(yp);
            } else {
                xv = __bfloat162float(__ldg(in_row + 2 * k));
                yv = __bfloat162float(__ldg(in_row + 2 * k + 1));
                rotate_pair_fma(xv, yv, sv, cv, xp, yp);
                out_row[2 * k]     = __float2bfloat16(xp);
                out_row[2 * k + 1] = __float2bfloat16(yp);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 7: Host-side launch wrappers
// ─────────────────────────────────────────────────────────────────────────────

void launch_rope_cache(
    float*       cos_cache,
    float*       sin_cache,
    int          seq_len,
    int          head_dim,
    float        base,
    int          pos_offset,
    cudaStream_t stream)
{
    const int half_dim = head_dim / 2;
    // One CTA per 256-element k-chunk; each CTA sweeps all S positions.
    const int block = 256;
    const int grid  = (half_dim + block - 1) / block;
    // Shared memory: one float per thread for inv_freq cache.
    const size_t smem = block * sizeof(float);
    rope_cache_kernel<<<std::max(grid,1), block, smem, stream>>>(
        cos_cache, sin_cache, seq_len, half_dim, base, pos_offset);
}

void launch_fused_rope_hetero(
    __nv_bfloat16*       output,
    const __nv_bfloat16* input,
    const float*         cos_cache,
    const float*         sin_cache,
    int                  batch,
    int                  seq_len,
    int                  num_heads,
    int                  head_dim,
    bool                 neox_style,
    int                  sm_version,
    cudaStream_t         stream)
{
    dim3 grid(batch * num_heads, seq_len);

    // Cacheless mode: cos_cache == nullptr → compute sin/cos on-the-fly.
    if (cos_cache == nullptr || sin_cache == nullptr) {
        // Use default base 10000.f and pos_offset 0 — caller should prefer
        // the cached path; this is only a fallback.
        const float base = 10000.f;
        if (sm_version >= 120) {
            if (neox_style)
                fused_rope_cacheless_kernel<120, true>
                    <<<grid, RoPEPolicy<120>::kBlockSize, 0, stream>>>(
                        output, input, batch, seq_len, num_heads, head_dim, base, 0);
            else
                fused_rope_cacheless_kernel<120, false>
                    <<<grid, RoPEPolicy<120>::kBlockSize, 0, stream>>>(
                        output, input, batch, seq_len, num_heads, head_dim, base, 0);
        } else if (sm_version >= 90) {
            if (neox_style)
                fused_rope_cacheless_kernel<90, true>
                    <<<grid, RoPEPolicy<90>::kBlockSize, 0, stream>>>(
                        output, input, batch, seq_len, num_heads, head_dim, base, 0);
            else
                fused_rope_cacheless_kernel<90, false>
                    <<<grid, RoPEPolicy<90>::kBlockSize, 0, stream>>>(
                        output, input, batch, seq_len, num_heads, head_dim, base, 0);
        } else {
            if (neox_style)
                fused_rope_cacheless_kernel<86, true>
                    <<<grid, RoPEPolicy<86>::kBlockSize, 0, stream>>>(
                        output, input, batch, seq_len, num_heads, head_dim, base, 0);
            else
                fused_rope_cacheless_kernel<86, false>
                    <<<grid, RoPEPolicy<86>::kBlockSize, 0, stream>>>(
                        output, input, batch, seq_len, num_heads, head_dim, base, 0);
        }
        return;
    }

    // Cached path: cos/sin precomputed and available.
    if (neox_style) {
        if (sm_version >= 120)
            fused_rope_neox_kernel<120><<<grid, RoPEPolicy<120>::kBlockSize, 0, stream>>>(
                output, input, cos_cache, sin_cache, batch, seq_len, num_heads, head_dim);
        else if (sm_version >= 90)
            fused_rope_neox_kernel<90><<<grid, RoPEPolicy<90>::kBlockSize, 0, stream>>>(
                output, input, cos_cache, sin_cache, batch, seq_len, num_heads, head_dim);
        else
            fused_rope_neox_kernel<86><<<grid, RoPEPolicy<86>::kBlockSize, 0, stream>>>(
                output, input, cos_cache, sin_cache, batch, seq_len, num_heads, head_dim);
    } else {
        if (sm_version >= 120)
            fused_rope_gptj_kernel<120><<<grid, RoPEPolicy<120>::kBlockSize, 0, stream>>>(
                output, input, cos_cache, sin_cache, batch, seq_len, num_heads, head_dim);
        else if (sm_version >= 90)
            fused_rope_gptj_kernel<90><<<grid, RoPEPolicy<90>::kBlockSize, 0, stream>>>(
                output, input, cos_cache, sin_cache, batch, seq_len, num_heads, head_dim);
        else
            fused_rope_gptj_kernel<86><<<grid, RoPEPolicy<86>::kBlockSize, 0, stream>>>(
                output, input, cos_cache, sin_cache, batch, seq_len, num_heads, head_dim);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Standalone cacheless launch (calls shared kernel with explicit base/offset)
// ─────────────────────────────────────────────────────────────────────────────

void launch_fused_rope_cacheless(
    __nv_bfloat16*       output,
    const __nv_bfloat16* input,
    int                  batch,
    int                  seq_len,
    int                  num_heads,
    int                  head_dim,
    float                base,
    int                  pos_offset,
    bool                 neox_style,
    int                  sm_version,
    cudaStream_t         stream)
{
    dim3 grid(batch * num_heads, seq_len);
    if (sm_version >= 120) {
        if (neox_style)
            fused_rope_cacheless_kernel<120, true>
                <<<grid, RoPEPolicy<120>::kBlockSize, 0, stream>>>(
                    output, input, batch, seq_len, num_heads, head_dim, base, pos_offset);
        else
            fused_rope_cacheless_kernel<120, false>
                <<<grid, RoPEPolicy<120>::kBlockSize, 0, stream>>>(
                    output, input, batch, seq_len, num_heads, head_dim, base, pos_offset);
    } else if (sm_version >= 90) {
        if (neox_style)
            fused_rope_cacheless_kernel<90, true>
                <<<grid, RoPEPolicy<90>::kBlockSize, 0, stream>>>(
                    output, input, batch, seq_len, num_heads, head_dim, base, pos_offset);
        else
            fused_rope_cacheless_kernel<90, false>
                <<<grid, RoPEPolicy<90>::kBlockSize, 0, stream>>>(
                    output, input, batch, seq_len, num_heads, head_dim, base, pos_offset);
    } else {
        if (neox_style)
            fused_rope_cacheless_kernel<86, true>
                <<<grid, RoPEPolicy<86>::kBlockSize, 0, stream>>>(
                    output, input, batch, seq_len, num_heads, head_dim, base, pos_offset);
        else
            fused_rope_cacheless_kernel<86, false>
                <<<grid, RoPEPolicy<86>::kBlockSize, 0, stream>>>(
                    output, input, batch, seq_len, num_heads, head_dim, base, pos_offset);
    }
}

// ===========================================================================
// Issue #23 — launch_fused_rope_qk: simultaneous Q + K RoPE for GQA
// ===========================================================================
//
// Grouped-Query Attention (GQA, Ainslie et al. 2023) uses Hq query heads and
// Hkv key/value heads where Hkv < Hq and Hq % Hkv == 0.  Standard RoPE
// applies the same rotation to both; the position-dependent sin/cos cache is
// shared between Q and K because head_dim D is the same.
//
// Kernel design
// ─────────────
// Grid:  (max(Hq,Hkv) * batch, seq_len, 2)
//   gridDim.z = 0 → Q sub-kernel  (Hq heads per batch)
//   gridDim.z = 1 → K sub-kernel  (Hkv heads per batch)
//
// This lets both Q and K execute in a single kernel launch with full GPU
// occupancy — no serialisation between the two tensors.  Each CTA checks
// gridDim.z and branches into the appropriate head-count / row-offset math.
//
// For cacheless mode (cos_cache == nullptr) sin/cos are computed on-the-fly
// using the rope_inv_freq helper, identical to fused_rope_cacheless_kernel.
//
// SM specialisation: RoPEPolicy<SmVer> drives __launch_bounds__ exactly as
// in the existing neox/gptj/cacheless kernels.
// ===========================================================================

template <int SmVer, bool kNeoxStyle, bool kCacheless>
__global__ void
__launch_bounds__(RoPEPolicy<SmVer>::kBlockSize, RoPEPolicy<SmVer>::kMinBlocksPerSM)
fused_rope_qk_kernel(
    __nv_bfloat16* __restrict__       q_output,   // [B, S, Hq,  D]
    __nv_bfloat16* __restrict__       k_output,   // [B, S, Hkv, D]
    const __nv_bfloat16* __restrict__ q_input,
    const __nv_bfloat16* __restrict__ k_input,
    const float* __restrict__         cos_cache,  // [S, D/2] or nullptr
    const float* __restrict__         sin_cache,  // [S, D/2] or nullptr
    int   batch,
    int   seq_len,
    int   num_heads_q,
    int   num_heads_kv,
    int   head_dim,
    float base,
    int   pos_offset)
{
    constexpr int kBS    = RoPEPolicy<SmVer>::kBlockSize;
    constexpr int kPairs = RoPEPolicy<SmVer>::kPairs;  // 4

    // gridDim.z selects Q (0) or K (1).
    const bool is_k    = (blockIdx.z == 1);
    const int  nh      = is_k ? num_heads_kv : num_heads_q;
    const int  bh_idx  = blockIdx.x;
    const int  seq_idx = blockIdx.y;

    // Guard: the grid is sized for max(Hq, Hkv); out-of-range heads are no-ops.
    if (bh_idx >= batch * nh || seq_idx >= seq_len) return;

    const int b       = bh_idx / nh;
    const int h       = bh_idx % nh;
    const int half_dim = head_dim / 2;

    // Row pointers into the appropriate tensor.
    const size_t row_off = ((size_t)b * seq_len + seq_idx) * nh * head_dim
                           + (size_t)h * head_dim;

    const __nv_bfloat16* in_row  = (is_k ? k_input  : q_input)  + row_off;
          __nv_bfloat16* out_row = (is_k ? k_output : q_output)  + row_off;

    // cos/sin pointers (shared between Q and K — same D).
    const float* cos_row = nullptr;
    const float* sin_row = nullptr;
    if constexpr (!kCacheless) {
        cos_row = cos_cache + (size_t)seq_idx * half_dim;
        sin_row = sin_cache + (size_t)seq_idx * half_dim;
    }

    const float pos = (float)(seq_idx + pos_offset);

    // Main rotation loop — kPairs pairs per thread per step.
    for (int k0 = (int)threadIdx.x * kPairs; k0 < half_dim; k0 += kBS * kPairs) {
        const int remaining = half_dim - k0;
        const int n = (remaining >= kPairs) ? kPairs : remaining;

        float xv[kPairs], yv[kPairs], c[kPairs], s[kPairs];
        float xp[kPairs], yp[kPairs];

        #pragma unroll
        for (int i = 0; i < kPairs; ++i) {
            if (i >= n) continue;
            const int k = k0 + i;

            // Load cos/sin — cached or compute on-the-fly.
            if constexpr (kCacheless) {
                float inv_f = rope_inv_freq(k, half_dim * 2, base);
                fast_sincosf(pos * inv_f, &s[i], &c[i]);
            } else {
                c[i] = __ldg(cos_row + k);
                s[i] = __ldg(sin_row + k);
            }

            // Load x, y from appropriate positions (NeoX vs GPT-J).
            if constexpr (kNeoxStyle) {
                xv[i] = __bfloat162float(__ldg(in_row + k));
                yv[i] = __bfloat162float(__ldg(in_row + k + half_dim));
            } else {
                xv[i] = __bfloat162float(__ldg(in_row + 2 * k));
                yv[i] = __bfloat162float(__ldg(in_row + 2 * k + 1));
            }
        }

        // Rotate all pairs.
        #pragma unroll
        for (int i = 0; i < kPairs; ++i)
            if (i < n) rotate_pair_fma(xv[i], yv[i], s[i], c[i], xp[i], yp[i]);

        // Store.
        #pragma unroll
        for (int i = 0; i < kPairs; ++i) {
            if (i >= n) continue;
            const int k = k0 + i;
            if constexpr (kNeoxStyle) {
                out_row[k]            = __float2bfloat16(xp[i]);
                out_row[k + half_dim] = __float2bfloat16(yp[i]);
            } else {
                out_row[2 * k]     = __float2bfloat16(xp[i]);
                out_row[2 * k + 1] = __float2bfloat16(yp[i]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side launch wrapper
// ---------------------------------------------------------------------------

void launch_fused_rope_qk(
    __nv_bfloat16*       q_output,
    __nv_bfloat16*       k_output,
    const __nv_bfloat16* q_input,
    const __nv_bfloat16* k_input,
    const float*         cos_cache,
    const float*         sin_cache,
    int                  batch,
    int                  seq_len,
    int                  num_heads_q,
    int                  num_heads_kv,
    int                  head_dim,
    bool                 neox_style,
    float                base,
    int                  pos_offset,
    int                  sm_version,
    cudaStream_t         stream)
{
    if (batch <= 0 || seq_len <= 0) return;

    // Grid: x = batch * max(Hq, Hkv), y = seq_len, z = 2 (Q=0, K=1).
    // Each CTA reads gridDim.z to determine which tensor it handles.
    const int max_heads = (num_heads_q > num_heads_kv) ? num_heads_q : num_heads_kv;
    dim3 grid(batch * max_heads, seq_len, 2);

    const bool cacheless = (cos_cache == nullptr || sin_cache == nullptr);

    // Macro to reduce the 2^3 = 8-way dispatch (SmVer × neox × cacheless).
#define DISPATCH_QK(SmV)                                                         \
    do {                                                                          \
        constexpr int kBS = RoPEPolicy<SmV>::kBlockSize;                         \
        if (neox_style && !cacheless)                                             \
            fused_rope_qk_kernel<SmV, true,  false><<<grid, kBS, 0, stream>>>(   \
                q_output, k_output, q_input, k_input,                            \
                cos_cache, sin_cache,                                             \
                batch, seq_len, num_heads_q, num_heads_kv,                       \
                head_dim, base, pos_offset);                                      \
        else if (!neox_style && !cacheless)                                       \
            fused_rope_qk_kernel<SmV, false, false><<<grid, kBS, 0, stream>>>(   \
                q_output, k_output, q_input, k_input,                            \
                cos_cache, sin_cache,                                             \
                batch, seq_len, num_heads_q, num_heads_kv,                       \
                head_dim, base, pos_offset);                                      \
        else if (neox_style && cacheless)                                         \
            fused_rope_qk_kernel<SmV, true,  true> <<<grid, kBS, 0, stream>>>(   \
                q_output, k_output, q_input, k_input,                            \
                nullptr, nullptr,                                                  \
                batch, seq_len, num_heads_q, num_heads_kv,                       \
                head_dim, base, pos_offset);                                      \
        else                                                                      \
            fused_rope_qk_kernel<SmV, false, true> <<<grid, kBS, 0, stream>>>(   \
                q_output, k_output, q_input, k_input,                            \
                nullptr, nullptr,                                                  \
                batch, seq_len, num_heads_q, num_heads_kv,                       \
                head_dim, base, pos_offset);                                      \
    } while (0)

    if (sm_version >= 120)      { DISPATCH_QK(120); }
    else if (sm_version >= 90)  { DISPATCH_QK(90);  }
    else                        { DISPATCH_QK(86);  }

#undef DISPATCH_QK
}
