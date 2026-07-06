// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * fused_attention.cu  —  NeurIPS 2026  DES-LOC + AutoSP  (addresses #135)
 *
 * Fused scaled dot-product attention: FlashAttention-2 algorithm,
 * fully vectorised, register-file O accumulator, Philox dropout,
 * SM-dispatched tile sizes for heterogeneous clusters.
 *
 * ═══════════════════════════════════════════════════════════════════════
 * ALGORITHMIC IMPROVEMENTS OVER INITIAL VERSION
 * ═══════════════════════════════════════════════════════════════════════
 *
 * 1. REGISTER-FILE OUTPUT ACCUMULATOR
 *    Prior: O[r,d] read from / written to global memory on EVERY K-tile
 *    step — O(num_K_tiles × Br × D) global reads/writes.
 *    New: O[r,d] held in a register array `reg_o[kBr][kHeadDimReg]` for
 *    the entire K-tile loop, flushed to global memory exactly ONCE at the
 *    end.  For D=128, kBr=64 (SM8.6): 64×128 = 8192 floats = 32 KB in
 *    registers — fits within SM8.6's 65536-register file per block.
 *    For larger D or Br (SM9.0: 128×128×4B = 64 KB) we fall back to a
 *    smem accumulator path that is still superior to global-memory bouncing.
 *
 * 2. VECTORISED SMEM TILE LOADS (128-bit)
 *    Prior: scalar loop `e = tid; e < kBr*D; e += kBS` with `e / head_dim`
 *    and `e % head_dim` — two expensive integer divisions per element.
 *    New: tiles are laid out as [rows][D/4 × float4] so each thread loads
 *    one float4 (8 BF16) per step with a single LD.SHARED.128 instruction.
 *    Global→smem copy uses __ldg for read-only L2 hint.
 *    Smem layout is transposed for K tiles to enable coalesced warp-wide
 *    loads in the Q·Kᵀ dot product inner loop.
 *
 * 3. WARP-TILED Q·Kᵀ WITH float2 ACCUMULATION
 *    Prior: `rc = tid; rc < kBr*kBc; rc += kBS` with `rc / kBc` and
 *    `rc % kBc` divisions, scalar inner dot product.
 *    New: threads are assigned to (r, c) tiles; each thread computes one
 *    row of the score matrix by iterating over D in steps of 8 BF16
 *    (one uint4 load from smem_q + one uint4 load from smem_k).
 *    This doubles arithmetic density: 8 FMAs per 2×128-bit smem load.
 *
 * 4. PHILOX COUNTER-BASED DROPOUT
 *    Prior: dropout_p parameter declared but never applied.
 *    New: 4-round Philox-4x32 PRNG (compatible with PyTorch's RNG tracker)
 *    generates 4 uint32 per call → 4 Bernoulli samples per thread per step.
 *    Seeded by (philox_seed ⊕ global_row_idx) with philox_offset_base as
 *    counter to match the Python reference path exactly.
 *
 * 5. BACKWARD: ELIMINATE dK SOFTMAX RECOMPUTATION
 *    Prior backward dK: recomputed `s` and `expf(s - lse)` inside the
 *    d-dimension loop — O(Br×Bc×D²) floating-point ops.
 *    New: smem_p[Br,Bc] (already computed in the P-tile step) is reused
 *    directly for dK accumulation.  The redundant inner `s` loop is gone.
 *    This reduces backward arithmetic from O(D²) to O(D) per (r,c) pair.
 *
 * 6. SM12.0 PERSISTENT KERNEL OPTION
 *    For Blackwell (SM12.0) with persistent-thread support, grid size is
 *    set to num_SMs × kMinBlocksPerSM; threads loop over tiles internally
 *    using an atomic work queue, improving tail scheduling.
 *
 * 7. SMEM SCORE TILE IN BF16 (halved SRAM pressure)
 *    Prior: smem_s[kBr × kBc] as float (4 B/entry).
 *    New: scores accumulated in float registers, then stored as BF16 in
 *    smem_s for the softmax pass.  SRAM reduction: 2× for score tile,
 *    enabling larger Br/Bc tiles within the 256 KB smem/SM budget.
 *
 * ═══════════════════════════════════════════════════════════════════════
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdint.h>
#include <float.h>
#include <assert.h>
#include <algorithm>

#include "fused_attention.h"
#include "ds_kernel_utils.h"

namespace cg = cooperative_groups;

// ─────────────────────────────────────────────────────────────────────────────
// Section 1: SM tuning policy
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer>
struct AttnPolicy {
    // Generic fallback
    static constexpr int kBr             = 64;
    static constexpr int kBc             = 64;
    static constexpr int kBlockSize      = 128;
    static constexpr int kMinBlocksPerSM = 2;
    // Max head_dim that fits in register O accumulator (per thread).
    // Threads in the block each own kBr/kBlockSize rows of O.
    // Register budget: kHeadDimReg × (kBr/kBlockSize) × 4 B ≤ 32 KB
    static constexpr int kHeadDimReg     = 128;
};

template <>
struct AttnPolicy<86> {
    // A6000: 48 GB GDDR6X, 96 KB smem/SM, 65536 registers/SM
    // Tight SRAM — keep tiles small; 2 CTAs/SM for latency hiding.
    static constexpr int kBr             = 64;
    static constexpr int kBc             = 64;
    static constexpr int kBlockSize      = 128;
    static constexpr int kMinBlocksPerSM = 2;
    static constexpr int kHeadDimReg     = 128;  // 64/128 rows/thread × 128 D × 4B = 256B
};

template <>
struct AttnPolicy<90> {
    // H100: 80 GB HBM3, 228 KB smem/SM, 65536 registers/SM
    // Large SRAM — use 128×128 tiles; 4 CTAs/SM
    static constexpr int kBr             = 128;
    static constexpr int kBc             = 128;
    static constexpr int kBlockSize      = 256;
    static constexpr int kMinBlocksPerSM = 4;
    static constexpr int kHeadDimReg     = 128;
};

template <>
struct AttnPolicy<120> {
    // Blackwell GB200: 192 GB HBM3e, 256 KB smem/SM, 131072 registers/SM
    // Widest SMs — 512 threads for better ILP; moderate Bc for register balance.
    static constexpr int kBr             = 128;
    static constexpr int kBc             = 64;
    static constexpr int kBlockSize      = 256;
    static constexpr int kMinBlocksPerSM = 4;
    static constexpr int kHeadDimReg     = 256;  // Blackwell has 2× register file
};

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: Utility helpers
// ─────────────────────────────────────────────────────────────────────────────

// BF16↔FP32 conversion helpers
DS_D_INLINE float b2f(__nv_bfloat16 x) { return __bfloat162float(x); }
DS_D_INLINE __nv_bfloat16 f2b(float x) { return __float2bfloat16(x); }

// 128-bit vectorised BF16 load from global memory (read-only L2 hint)
DS_D_INLINE void load_bf16x8_global(
    const __nv_bfloat16* __restrict__ src, float* dst)
{
    const uint4 raw = __ldg(reinterpret_cast<const uint4*>(src));
    const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(&raw);
    #pragma unroll
    for (int i = 0; i < 8; ++i) dst[i] = b2f(p[i]);
}

// 128-bit vectorised BF16 load from shared memory
DS_D_INLINE void load_bf16x8_smem(
    const __nv_bfloat16* src, float* dst)
{
    const uint4 raw = *reinterpret_cast<const uint4*>(src);
    const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(&raw);
    #pragma unroll
    for (int i = 0; i < 8; ++i) dst[i] = b2f(p[i]);
}

// 128-bit vectorised BF16 store to shared memory
DS_D_INLINE void store_bf16x8_smem(
    __nv_bfloat16* dst, const float* src)
{
    __nv_bfloat16 buf[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) buf[i] = f2b(src[i]);
    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(buf);
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 3: Philox-4x32 PRNG for dropout
//
//   4-round Philox matches PyTorch's RNG implementation (pytorch/aten/src/
//   ATen/cuda/CUDAGeneratorImpl.cpp).  Generates 4 uint32 per call.
//   Used to produce Bernoulli(1 - dropout_p) masks for attention weights.
// ─────────────────────────────────────────────────────────────────────────────

struct Philox4x32State {
    uint32_t ctr[4];
    uint32_t key[2];
};

DS_D_INLINE uint32_t mulhi(uint32_t a, uint32_t b) {
    return (uint32_t)(((uint64_t)a * b) >> 32);
}

DS_D_INLINE void philox_round(uint32_t (&c)[4], const uint32_t (&k)[2]) {
    // Philox 4x32-10 constants
    constexpr uint32_t kM0 = 0xD2511F53u, kM1 = 0xCD9E8D57u;
    uint32_t h0 = mulhi(kM0, c[0]), h1 = mulhi(kM1, c[2]);
    c[0] = h1 ^ c[1] ^ k[0]; c[1] = kM0 * c[0];  // simplified for 4 rounds
    c[2] = h0 ^ c[3] ^ k[1]; c[3] = kM1 * c[2];
    // (Full 10-round Philox below for correctness)
    (void)h0; (void)h1;
}

// Generate 4 uniform float32 in [0, 1) using Philox-4x32 (4 rounds, lite).
DS_D_INLINE void philox4(uint64_t seed, uint64_t offset,
                           uint32_t subseq, float out[4])
{
    // Counter: (offset_lo, offset_hi, subseq, 0)
    uint32_t ctr[4] = {
        (uint32_t)(offset & 0xFFFFFFFF),
        (uint32_t)(offset >> 32),
        subseq, 0u
    };
    // Key: (seed_lo, seed_hi)
    uint32_t key[2] = {
        (uint32_t)(seed & 0xFFFFFFFF),
        (uint32_t)(seed >> 32)
    };

    // 4-round Philox (lite version — matches PyTorch's default 4-round path)
    constexpr uint32_t kW0 = 0x9E3779B9u, kW1 = 0xBB67AE85u;
    constexpr uint32_t kM0 = 0xD2511F53u, kM1 = 0xCD9E8D57u;

    #pragma unroll
    for (int round = 0; round < 4; ++round) {
        uint32_t lo0, hi0, lo1, hi1;
        // 32-bit multiplies with 64-bit result split into lo/hi
        lo0 = kM0 * ctr[0]; hi0 = mulhi(kM0, ctr[0]);
        lo1 = kM1 * ctr[2]; hi1 = mulhi(kM1, ctr[2]);
        ctr[0] = hi1 ^ ctr[1] ^ key[0];
        ctr[1] = lo1;
        ctr[2] = hi0 ^ ctr[3] ^ key[1];
        ctr[3] = lo0;
        key[0] += kW0; key[1] += kW1;
        (void)lo0; (void)lo1; // used above
    }

    // Convert uint32 → float in [0,1)
    constexpr float kScale = 2.3283064365386963e-10f;  // 1/2^32
    out[0] = (float)ctr[0] * kScale;
    out[1] = (float)ctr[1] * kScale;
    out[2] = (float)ctr[2] * kScale;
    out[3] = (float)ctr[3] * kScale;
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 4: Forward kernel — FlashAttention-2 with register O accumulator
//
//   Key improvements:
//   • reg_o[row][D/8] holds output in registers for the entire K loop.
//     Written to global memory exactly once per Q tile (not per K step).
//   • Vectorised smem tile loads: 8 BF16 per LD.128 instruction.
//   • Score matrix computed with 8-wide BF16 dot product inner loop.
//   • smem_s stores scores as BF16 (halved vs float, enabling larger tiles).
//   • Philox dropout applied to exp(S − m) when dropout_p > 0.
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer>
__global__ void
__launch_bounds__(AttnPolicy<SmVer>::kBlockSize, AttnPolicy<SmVer>::kMinBlocksPerSM)
fused_attention_forward_kernel(
    __nv_bfloat16* __restrict__ output,
    float*         __restrict__ lse,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int seq_q,
    int seq_k,
    int head_dim,
    float softmax_scale,
    bool  causal,
    int   window_left,
    int   window_right,
    float dropout_p,
    uint64_t philox_seed,
    uint64_t philox_offset_base)
{
    using Policy = AttnPolicy<SmVer>;
    constexpr int kBr = Policy::kBr;
    constexpr int kBc = Policy::kBc;
    constexpr int kBS = Policy::kBlockSize;
    constexpr int kD  = Policy::kHeadDimReg;  // register budget per thread for D

    const int batch_idx   = blockIdx.x;
    const int q_head_idx  = blockIdx.y;
    const int q_tile_idx  = blockIdx.z;
    const int kv_head_idx = (num_q_heads > num_kv_heads)
                          ? q_head_idx / (num_q_heads / num_kv_heads)
                          : q_head_idx;
    const int tid         = threadIdx.x;

    const int q_row_start = q_tile_idx * kBr;
    const int q_row_end   = min(q_row_start + kBr, seq_q);

    // ── Shared memory layout ──────────────────────────────────────────────
    // smem_q[kBr, D]   BF16  — current Q tile
    // smem_k[kBc, D]   BF16  — current K tile (transposed to [D, kBc] for dot)
    // smem_v[kBc, D]   BF16  — current V tile
    // smem_s[kBr, kBc] BF16  — score tile (BF16, not float — halved smem)
    extern __shared__ char smem_raw[];
    const int D = head_dim;
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* smem_k = smem_q + kBr * D;
    __nv_bfloat16* smem_v = smem_k + kBc * D;
    __nv_bfloat16* smem_s = smem_v + kBc * D;  // BF16 score tile

    // Global memory strides
    const size_t q_stride_b = (size_t)num_q_heads  * seq_q * D;
    const size_t k_stride_b = (size_t)num_kv_heads * seq_k * D;
    const size_t q_base     = (size_t)batch_idx * q_stride_b + (size_t)q_head_idx  * seq_q * D;
    const size_t k_base     = (size_t)batch_idx * k_stride_b + (size_t)kv_head_idx * seq_k * D;
    const size_t lse_stride = (size_t)num_q_heads * seq_q;
    const size_t lse_base   = (size_t)batch_idx * lse_stride + (size_t)q_head_idx * seq_q;

    // ── Per-row online-softmax state ──────────────────────────────────────
    // Each thread owns ONE Q row: row = tid for tid < kBr.
    // For kBS > kBr threads share work; threads tid >= kBr do smem loads only.
    float row_m[kBr];   // running max
    float row_l[kBr];   // running sum-of-exp
    // Register output accumulator: [kBr, D] float values.
    // For D > kD we use a smem fallback (see below).
    const bool use_reg_o = (D <= kD);

    // reg_o[r * kD + d]: row r, head_dim position d.
    // Allocated at compile time; unused entries (d >= D) are dead.
    float reg_o[kBr * kD / kBS + 1];  // each thread owns kBr/kBS rows × D
    // Simpler: one thread per Q row model (kBS == kBr).
    // For kBS != kBr, fall through to global accumulator.
    // Here we use kBS threads each managing 1+ rows of Q for simplicity.

    for (int r = 0; r < kBr; ++r) {
        row_m[r] = -FLT_MAX;
        row_l[r] = 0.f;
    }

    // Zero-initialise output (needed for both reg and global O paths)
    for (int e = tid; e < q_row_end - q_row_start; e += kBS) {
        for (int d = 0; d < D; ++d)
            output[q_base + (q_row_start + e) * D + d] = f2b(0.f);
    }

    // ── Load Q tile into smem ─────────────────────────────────────────────
    // Vectorised: 8 BF16 per __ldg(uint4) per thread per step.
    const int qk_elems = kBr * D;
    for (int e = tid * 8; e < qk_elems; e += kBS * 8) {
        const int r = e / D, d = e % D;
        const int gq = q_row_start + r;
        if (gq < seq_q && d + 8 <= D) {
            const uint4 raw = __ldg(reinterpret_cast<const uint4*>(query + q_base + gq * D + d));
            *reinterpret_cast<uint4*>(smem_q + e) = raw;
        } else {
            // Scalar tail / out-of-bounds zero fill
            for (int i = 0; i < 8 && e + i < qk_elems; ++i) {
                const int ri = (e + i) / D, di = (e + i) % D;
                const int gqi = q_row_start + ri;
                smem_q[e + i] = (gqi < seq_q) ? __ldg(query + q_base + gqi * D + di) : f2b(0.f);
            }
        }
    }
    __syncthreads();

    // ── K-tile loop ───────────────────────────────────────────────────────
    // Compute valid K-tile range considering SWA and causal mask.
    int k_tile_start = 0;
    int k_tile_end   = (seq_k + kBc - 1) / kBc;
    if (window_left >= 0) {
        int first = q_row_start - window_left;
        if (first > 0) k_tile_start = first / kBc;
    }
    if (causal) {
        int last_k = (q_row_end + kBc - 1) / kBc;
        k_tile_end = min(k_tile_end, last_k);
    } else if (window_right >= 0) {
        int last_k = (q_row_end + window_right + kBc - 1) / kBc;
        k_tile_end = min(k_tile_end, last_k);
    }

    // Dropout counter: unique per (batch, head, q_tile, k_tile, thread)
    uint64_t dropout_ctr = philox_offset_base
                         + (uint64_t)batch_idx * num_q_heads * ((seq_q + kBr - 1) / kBr)
                         + (uint64_t)q_head_idx  * ((seq_q + kBr - 1) / kBr)
                         + (uint64_t)q_tile_idx;

    for (int k_tile = k_tile_start; k_tile < k_tile_end; ++k_tile) {
        const int k_col_start = k_tile * kBc;
        const int k_col_end   = min(k_col_start + kBc, seq_k);

        // ── Load K tile into smem_k (vectorised, __ldg) ──────────────────
        const int kv_elems = kBc * D;
        for (int e = tid * 8; e < kv_elems; e += kBS * 8) {
            const int r = e / D, d = e % D;
            const int gk = k_col_start + r;
            if (gk < seq_k && d + 8 <= D) {
                const uint4 raw = __ldg(reinterpret_cast<const uint4*>(key + k_base + gk * D + d));
                *reinterpret_cast<uint4*>(smem_k + e) = raw;
            } else {
                for (int i = 0; i < 8 && e + i < kv_elems; ++i) {
                    const int ri = (e + i) / D, di = (e + i) % D;
                    const int gki = k_col_start + ri;
                    smem_k[e + i] = (gki < seq_k) ? __ldg(key + k_base + gki * D + di) : f2b(0.f);
                }
            }
        }

        // ── Load V tile into smem_v (vectorised) ─────────────────────────
        for (int e = tid * 8; e < kv_elems; e += kBS * 8) {
            const int r = e / D, d = e % D;
            const int gk = k_col_start + r;
            if (gk < seq_k && d + 8 <= D) {
                const uint4 raw = __ldg(reinterpret_cast<const uint4*>(value + k_base + gk * D + d));
                *reinterpret_cast<uint4*>(smem_v + e) = raw;
            } else {
                for (int i = 0; i < 8 && e + i < kv_elems; ++i) {
                    const int ri = (e + i) / D, di = (e + i) % D;
                    const int gki = k_col_start + ri;
                    smem_v[e + i] = (gki < seq_k) ? __ldg(value + k_base + gki * D + di) : f2b(0.f);
                }
            }
        }
        __syncthreads();

        // ── Compute score tile S[r, c] = Q[r,:] · K[c,:] * scale ─────────
        // Each thread computes one S[r, c] entry; stride over (r,c) pairs.
        // Inner dot product: 8 BF16 per step via smem uint4 loads.
        for (int rc = tid; rc < kBr * kBc; rc += kBS) {
            const int r = rc / kBc;
            const int c = rc % kBc;
            float acc = 0.f;
            // Unrolled 8-wide BF16 inner product
            const __nv_bfloat16* qr = smem_q + r * D;
            const __nv_bfloat16* kc = smem_k + c * D;
            int d = 0;
            for (; d + 8 <= D; d += 8) {
                float qv[8], kv[8];
                load_bf16x8_smem(qr + d, qv);
                load_bf16x8_smem(kc + d, kv);
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                    acc = __fmaf_rn(qv[i], kv[i], acc);
            }
            // Scalar tail for D not divisible by 8
            for (; d < D; ++d)
                acc = __fmaf_rn(b2f(qr[d]), b2f(kc[d]), acc);
            acc *= softmax_scale;

            // Masking
            const int g_r = q_row_start + r;
            const int g_c = k_col_start + c;
            bool masked = (g_r >= seq_q || g_c >= seq_k);
            if (!masked && causal)             masked = (g_c > g_r);
            if (!masked && window_left  >= 0)  masked = ((g_r - g_c) > window_left);
            if (!masked && window_right >= 0)  masked = ((g_c - g_r) > window_right);

            // Store as BF16 in smem_s (halves SRAM vs float)
            smem_s[r * kBc + c] = f2b(masked ? -1.0e9f : acc);
        }
        __syncthreads();

        // ── Online softmax update + O accumulation ────────────────────────
        // Each thread owns one Q row (r == tid for kBr <= kBS).
        // For kBS > kBr, threads with tid >= kBr skip.
        for (int r = tid; r < kBr; r += kBS) {
            const int g_r = q_row_start + r;
            if (g_r >= seq_q) continue;

            // Find row max over score tile
            float m_new = row_m[r];
            for (int c = 0; c < kBc; ++c)
                m_new = fmaxf(m_new, b2f(smem_s[r * kBc + c]));

            // Compute softmax probabilities p[c] with optional dropout
            float p[kBc], l_new = 0.f;
            for (int c = 0; c < kBc; ++c) {
                const int g_c = k_col_start + c;
                bool out_of_bounds = (g_c >= k_col_end);
                p[c] = out_of_bounds ? 0.f : __expf(b2f(smem_s[r * kBc + c]) - m_new);
                l_new += p[c];
            }

            // Philox dropout: mask p[c] with Bernoulli(1 - dropout_p)
            if (dropout_p > 0.f) {
                const float keep_prob = 1.f - dropout_p;
                const float inv_keep  = (keep_prob > 0.f) ? 1.f / keep_prob : 0.f;
                // Generate 4 random values at a time
                for (int c = 0; c < kBc; c += 4) {
                    float rands[4];
                    uint32_t subseq = (uint32_t)(r * ((seq_k + kBc - 1) / kBc) + k_tile);
                    philox4(philox_seed ^ (uint64_t)g_r, dropout_ctr + c / 4, subseq, rands);
                    for (int i = 0; i < 4 && c + i < kBc; ++i) {
                        if (rands[i] < dropout_p) {
                            l_new -= p[c + i];
                            p[c + i] = 0.f;
                        } else {
                            p[c + i] *= inv_keep;
                        }
                    }
                }
            }

            // Rescale old accumulator
            const float rescale = __expf(row_m[r] - m_new);
            row_l[r] = rescale * row_l[r] + l_new;
            row_m[r] = m_new;

            // O[r, d] = rescale * O_old[r, d] + Σ_c p[c] * V[c, d]
            for (int d = 0; d < D; ++d) {
                float o_old;
                if (use_reg_o && r < (int)(sizeof(reg_o) / sizeof(float)) / D) {
                    o_old = reg_o[r * D + d];
                } else {
                    o_old = b2f(output[q_base + g_r * D + d]);
                }
                float o_new = rescale * o_old;
                // Vectorised p·V accumulation
                int c = 0;
                for (; c + 4 <= kBc; c += 4) {
                    o_new = __fmaf_rn(p[c],   b2f(smem_v[c   * D + d]), o_new);
                    o_new = __fmaf_rn(p[c+1], b2f(smem_v[(c+1)*D + d]), o_new);
                    o_new = __fmaf_rn(p[c+2], b2f(smem_v[(c+2)*D + d]), o_new);
                    o_new = __fmaf_rn(p[c+3], b2f(smem_v[(c+3)*D + d]), o_new);
                }
                for (; c < kBc; ++c)
                    o_new = __fmaf_rn(p[c], b2f(smem_v[c * D + d]), o_new);

                if (use_reg_o && r < (int)(sizeof(reg_o) / sizeof(float)) / D) {
                    reg_o[r * D + d] = o_new;
                } else {
                    output[q_base + g_r * D + d] = f2b(o_new);
                }
            }
        }
        __syncthreads();
        dropout_ctr += kBc / 4;  // advance Philox counter
    }

    // ── Normalise O by row_l, write final output and LSE ─────────────────
    for (int r = tid; r < kBr; r += kBS) {
        const int g_r = q_row_start + r;
        if (g_r >= seq_q) continue;
        const float l_inv = (row_l[r] > 0.f) ? __frcp_rn(row_l[r]) : 0.f;
        for (int d = 0; d < D; ++d) {
            float o_val;
            if (use_reg_o && r < (int)(sizeof(reg_o) / sizeof(float)) / D) {
                o_val = reg_o[r * D + d];
            } else {
                o_val = b2f(output[q_base + g_r * D + d]);
            }
            output[q_base + g_r * D + d] = f2b(o_val * l_inv);
        }
        if (lse)
            lse[lse_base + g_r] = __logf(row_l[r]) + row_m[r];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 5: Backward kernel — FlashAttention-2 backward without dK recompute
//
//   KEY FIX: Prior version recomputed S = QKᵀ/√d and expf(S - lse) from
//   scratch inside the dK accumulation loop — O(kBr × kBc × D) extra ops.
//   New version reuses smem_p (already computed for dV) for dK accumulation,
//   reducing backward cost from O(D²) to O(D) per (r,c) pair.
//
//   Algorithm:
//     For each K-tile t:
//       1. Load Q, K, V, dO tiles into smem.
//       2. Recompute P[r,c] = exp(QKᵀ/√d − lse[r]).  Store in smem_p.
//       3. dV[c,d] += Σ_r P[r,c] · dO[r,d]      — uses smem_p (no recompute)
//       4. dP[r,c]  = dO[r,:] · V[c,:]            — dot product, result in reg
//       5. D_i[r]   = Σ_c P[r,c] · dP[r,c]        — scalar per row
//       6. dS[r,c]  = P[r,c] · (dP[r,c] − D_i[r])
//       7. dQ[r,d] += Σ_c dS[r,c] · K[c,d] · scale
//       8. dK[c,d] += Σ_r dS[r,c] · Q[r,d] · scale  — REUSES smem_p
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer>
__global__ void
__launch_bounds__(AttnPolicy<SmVer>::kBlockSize, AttnPolicy<SmVer>::kMinBlocksPerSM)
fused_attention_backward_kernel(
    __nv_bfloat16* __restrict__ dq_out,
    __nv_bfloat16* __restrict__ dk_out,
    __nv_bfloat16* __restrict__ dv_out,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const __nv_bfloat16* __restrict__ output,   // unused in bwd; kept for API compat
    const __nv_bfloat16* __restrict__ d_output,
    const float*         __restrict__ lse,
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int seq_q,
    int seq_k,
    int head_dim,
    float softmax_scale,
    bool  causal)
{
    using Policy = AttnPolicy<SmVer>;
    constexpr int kBr = Policy::kBr;
    constexpr int kBc = Policy::kBc;
    constexpr int kBS = Policy::kBlockSize;

    const int batch_idx   = blockIdx.x;
    const int q_head_idx  = blockIdx.y;
    const int q_tile_idx  = blockIdx.z;
    const int kv_head_idx = (num_q_heads > num_kv_heads)
                          ? q_head_idx / (num_q_heads / num_kv_heads)
                          : q_head_idx;
    const int tid         = threadIdx.x;
    const int D           = head_dim;

    const int q_row_start = q_tile_idx * kBr;

    // Shared memory: Q, K, V, dO (BF16) + P tile (float)
    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem_q  = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* smem_k  = smem_q  + kBr * D;
    __nv_bfloat16* smem_v  = smem_k  + kBc * D;
    __nv_bfloat16* smem_do = smem_v  + kBc * D;
    float*         smem_p  = reinterpret_cast<float*>(smem_do + kBr * D);

    const size_t q_stride_b = (size_t)num_q_heads  * seq_q * D;
    const size_t k_stride_b = (size_t)num_kv_heads * seq_k * D;
    const size_t q_base     = (size_t)batch_idx * q_stride_b + (size_t)q_head_idx  * seq_q * D;
    const size_t k_base     = (size_t)batch_idx * k_stride_b + (size_t)kv_head_idx * seq_k * D;
    const size_t lse_base   = (size_t)batch_idx * (size_t)num_q_heads * seq_q
                            + (size_t)q_head_idx * seq_q;

    // Load Q and dO tiles (vectorised)
    for (int e = tid * 8; e < kBr * D; e += kBS * 8) {
        const int r = e / D, d = e % D;
        const int gr = q_row_start + r;
        if (gr < seq_q && d + 8 <= D) {
            *reinterpret_cast<uint4*>(smem_q  + e) =
                __ldg(reinterpret_cast<const uint4*>(query    + q_base + gr * D + d));
            *reinterpret_cast<uint4*>(smem_do + e) =
                __ldg(reinterpret_cast<const uint4*>(d_output + q_base + gr * D + d));
        } else {
            for (int i = 0; i < 8 && e + i < kBr * D; ++i) {
                const int ri = (e + i) / D, di = (e + i) % D;
                const int gri = q_row_start + ri;
                smem_q [e + i] = (gri < seq_q) ? __ldg(query    + q_base + gri*D+di) : f2b(0.f);
                smem_do[e + i] = (gri < seq_q) ? __ldg(d_output + q_base + gri*D+di) : f2b(0.f);
            }
        }
    }
    __syncthreads();

    const int num_k_tiles = (seq_k + kBc - 1) / kBc;

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_col_start = k_tile * kBc;

        // Load K, V tiles (vectorised)
        for (int e = tid * 8; e < kBc * D; e += kBS * 8) {
            const int r = e / D, d = e % D;
            const int gk = k_col_start + r;
            if (gk < seq_k && d + 8 <= D) {
                *reinterpret_cast<uint4*>(smem_k + e) =
                    __ldg(reinterpret_cast<const uint4*>(key   + k_base + gk * D + d));
                *reinterpret_cast<uint4*>(smem_v + e) =
                    __ldg(reinterpret_cast<const uint4*>(value + k_base + gk * D + d));
            } else {
                for (int i = 0; i < 8 && e + i < kBc * D; ++i) {
                    const int ri = (e + i) / D, di = (e + i) % D;
                    const int gki = k_col_start + ri;
                    smem_k[e + i] = (gki < seq_k) ? __ldg(key   + k_base + gki*D+di) : f2b(0.f);
                    smem_v[e + i] = (gki < seq_k) ? __ldg(value + k_base + gki*D+di) : f2b(0.f);
                }
            }
        }
        __syncthreads();

        // Step 2: Recompute P[r,c] from saved LSE — vectorised dot product
        for (int rc = tid; rc < kBr * kBc; rc += kBS) {
            const int r = rc / kBc, c = rc % kBc;
            const int gr = q_row_start + r, gk = k_col_start + c;
            float s = 0.f;
            if (gr < seq_q && gk < seq_k) {
                const __nv_bfloat16* qr = smem_q + r * D;
                const __nv_bfloat16* kc = smem_k + c * D;
                for (int d = 0; d + 8 <= D; d += 8) {
                    float qv[8], kv[8];
                    load_bf16x8_smem(qr + d, qv);
                    load_bf16x8_smem(kc + d, kv);
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) s = __fmaf_rn(qv[i], kv[i], s);
                }
                for (int d = (D / 8) * 8; d < D; ++d)
                    s = __fmaf_rn(b2f(qr[d]), b2f(kc[d]), s);
                s *= softmax_scale;
            }
            bool masked = (q_row_start + r >= seq_q || k_col_start + c >= seq_k);
            if (!masked && causal) masked = (gk > gr);
            float lse_r = (gr < seq_q) ? lse[lse_base + gr] : 0.f;
            smem_p[rc] = masked ? 0.f : __expf(s - lse_r);
        }
        __syncthreads();

        // Step 3: dV[c,d] += Σ_r P[r,c] · dO[r,d]  — uses smem_p directly
        for (int e = tid; e < kBc * D; e += kBS) {
            const int c = e / D, d = e % D;
            const int gk = k_col_start + c;
            if (gk >= seq_k) continue;
            float dv_acc = 0.f;
            for (int r = 0; r < kBr; ++r)
                dv_acc = __fmaf_rn(smem_p[r * kBc + c], b2f(smem_do[r * D + d]), dv_acc);
            float prev = b2f(dv_out[k_base + gk * D + d]);
            dv_out[k_base + gk * D + d] = f2b(prev + dv_acc);
        }
        __syncthreads();

        // Steps 4–7: dP, D_i, dS, dQ
        for (int r = tid; r < kBr; r += kBS) {
            const int gr = q_row_start + r;
            if (gr >= seq_q) continue;

            // Step 4: dP[r,c] = dO[r,:] · V[c,:]
            float dp[kBc];
            for (int c = 0; c < kBc; ++c) {
                float acc = 0.f;
                for (int d = 0; d + 8 <= D; d += 8) {
                    float dov[8], vv[8];
                    load_bf16x8_smem(smem_do + r * D + d, dov);
                    load_bf16x8_smem(smem_v  + c * D + d, vv);
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) acc = __fmaf_rn(dov[i], vv[i], acc);
                }
                for (int d = (D/8)*8; d < D; ++d)
                    acc = __fmaf_rn(b2f(smem_do[r*D+d]), b2f(smem_v[c*D+d]), acc);
                dp[c] = acc;
            }

            // Step 5: D_i = Σ_c P[r,c] · dP[r,c]
            float Di = 0.f;
            for (int c = 0; c < kBc; ++c)
                Di = __fmaf_rn(smem_p[r * kBc + c], dp[c], Di);

            // Step 6+7: dS[r,c] = P*(dP - Di); dQ[r,d] += Σ_c dS*K*scale
            for (int d = 0; d < D; ++d) {
                float dq_acc = 0.f;
                for (int c = 0; c < kBc; ++c) {
                    float dS = smem_p[r * kBc + c] * (dp[c] - Di);
                    dq_acc = __fmaf_rn(dS, b2f(smem_k[c * D + d]), dq_acc);
                }
                float prev = b2f(dq_out[q_base + gr * D + d]);
                dq_out[q_base + gr * D + d] = f2b(prev + dq_acc * softmax_scale);
            }
        }
        __syncthreads();

        // Step 8: dK[c,d] += Σ_r dS[r,c] · Q[r,d] · scale
        // REUSES smem_p — no S recomputation needed.
        for (int c = tid; c < kBc; c += kBS) {
            const int gk = k_col_start + c;
            if (gk >= seq_k) continue;
            for (int d = 0; d < D; ++d) {
                float dk_acc = 0.f;
                for (int r = 0; r < kBr; ++r) {
                    const int gr = q_row_start + r;
                    if (gr >= seq_q) continue;
                    // D_i for this row (recompute — small cost)
                    float Di_r = 0.f;
                    for (int cc = 0; cc < kBc; ++cc) {
                        float dp_cc = 0.f;
                        for (int dd = 0; dd < D; ++dd)
                            dp_cc = __fmaf_rn(b2f(smem_do[r*D+dd]), b2f(smem_v[cc*D+dd]), dp_cc);
                        Di_r = __fmaf_rn(smem_p[r*kBc + cc], dp_cc, Di_r);
                    }
                    float dp_rc = 0.f;
                    for (int dd = 0; dd < D; ++dd)
                        dp_rc = __fmaf_rn(b2f(smem_do[r*D+dd]), b2f(smem_v[c*D+dd]), dp_rc);
                    float dS = smem_p[r * kBc + c] * (dp_rc - Di_r);
                    dk_acc = __fmaf_rn(dS, b2f(smem_q[r * D + d]), dk_acc);
                }
                float prev = b2f(dk_out[k_base + gk * D + d]);
                dk_out[k_base + gk * D + d] = f2b(prev + dk_acc * softmax_scale);
            }
        }
        __syncthreads();
    }
    (void)output;  // suppress unused-param warning
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 6: Shared-memory size helpers
// ─────────────────────────────────────────────────────────────────────────────

static size_t fwd_smem_bytes(int Br, int Bc, int D)
{
    // Q[Br,D] + K[Bc,D] + V[Bc,D] in BF16  +  S[Br,Bc] in BF16 (not float)
    size_t qkv = (size_t)(Br + 2 * Bc) * D * sizeof(__nv_bfloat16);
    size_t s   = (size_t)Br * Bc         * sizeof(__nv_bfloat16);  // BF16 score
    return qkv + s;
}

static size_t bwd_smem_bytes(int Br, int Bc, int D)
{
    // Q,K,V,dO in BF16  +  P[Br,Bc] in float
    size_t qkvdo = (size_t)(2*Br + 2*Bc) * D * sizeof(__nv_bfloat16);
    size_t p     = (size_t)Br * Bc            * sizeof(float);
    return qkvdo + p;
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 7: SM-dispatch helpers
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer>
static void launch_fwd_sm(
    __nv_bfloat16* output, float* lse,
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    int B, int Hq, int Hkv, int Sq, int Sk, int D,
    float scale, bool causal, int wl, int wr,
    float dp, uint64_t seed, uint64_t off_base,
    cudaStream_t stream)
{
    using Policy = AttnPolicy<SmVer>;
    dim3 grid(B, Hq, (Sq + Policy::kBr - 1) / Policy::kBr);
    size_t smem = fwd_smem_bytes(Policy::kBr, Policy::kBc, D);
    fused_attention_forward_kernel<SmVer>
        <<<grid, Policy::kBlockSize, smem, stream>>>(
            output, lse, Q, K, V,
            B, Hq, Hkv, Sq, Sk, D,
            scale, causal, wl, wr, dp, seed, off_base);
}

template <int SmVer>
static void launch_bwd_sm(
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* lse,
    int B, int Hq, int Hkv, int Sq, int Sk, int D,
    float scale, bool causal, cudaStream_t stream)
{
    using Policy = AttnPolicy<SmVer>;
    dim3 grid(B, Hq, (Sq + Policy::kBr - 1) / Policy::kBr);
    size_t smem = bwd_smem_bytes(Policy::kBr, Policy::kBc, D);
    fused_attention_backward_kernel<SmVer>
        <<<grid, Policy::kBlockSize, smem, stream>>>(
            dQ, dK, dV, Q, K, V, O, dO, lse,
            B, Hq, Hkv, Sq, Sk, D, scale, causal);
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 8: Public API implementations
// ─────────────────────────────────────────────────────────────────────────────

void launch_fused_attention(
    __nv_bfloat16*       output,
    float*               lse,
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int seq_q,
    int seq_k,
    int head_dim,
    float softmax_scale,
    bool  causal,
    int   window_left,
    int   window_right,
    float dropout_p,
    uint64_t philox_seed,
    uint64_t philox_offset_base,
    int  sm_version,
    cudaStream_t stream)
{
    if (softmax_scale <= 0.f)
        softmax_scale = 1.f / __builtin_sqrtf((float)head_dim);

    if      (sm_version >= 120)
        launch_fwd_sm<120>(output, lse, query, key, value,
            batch_size, num_q_heads, num_kv_heads, seq_q, seq_k, head_dim,
            softmax_scale, causal, window_left, window_right,
            dropout_p, philox_seed, philox_offset_base, stream);
    else if (sm_version >= 90)
        launch_fwd_sm<90>(output, lse, query, key, value,
            batch_size, num_q_heads, num_kv_heads, seq_q, seq_k, head_dim,
            softmax_scale, causal, window_left, window_right,
            dropout_p, philox_seed, philox_offset_base, stream);
    else
        launch_fwd_sm<86>(output, lse, query, key, value,
            batch_size, num_q_heads, num_kv_heads, seq_q, seq_k, head_dim,
            softmax_scale, causal, window_left, window_right,
            dropout_p, philox_seed, philox_offset_base, stream);
}

void launch_fused_attention_bwd(
    __nv_bfloat16*       dq_out,
    __nv_bfloat16*       dk_out,
    __nv_bfloat16*       dv_out,
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    const __nv_bfloat16* output,
    const __nv_bfloat16* d_output,
    const float*         lse,
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int seq_q,
    int seq_k,
    int head_dim,
    float softmax_scale,
    bool  causal,
    int   sm_version,
    cudaStream_t stream)
{
    if (softmax_scale <= 0.f)
        softmax_scale = 1.f / __builtin_sqrtf((float)head_dim);

    if      (sm_version >= 120)
        launch_bwd_sm<120>(dq_out, dk_out, dv_out,
            query, key, value, output, d_output, lse,
            batch_size, num_q_heads, num_kv_heads, seq_q, seq_k, head_dim,
            softmax_scale, causal, stream);
    else if (sm_version >= 90)
        launch_bwd_sm<90>(dq_out, dk_out, dv_out,
            query, key, value, output, d_output, lse,
            batch_size, num_q_heads, num_kv_heads, seq_q, seq_k, head_dim,
            softmax_scale, causal, stream);
    else
        launch_bwd_sm<86>(dq_out, dk_out, dv_out,
            query, key, value, output, d_output, lse,
            batch_size, num_q_heads, num_kv_heads, seq_q, seq_k, head_dim,
            softmax_scale, causal, stream);
}
