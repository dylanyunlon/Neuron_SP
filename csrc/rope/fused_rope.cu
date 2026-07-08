// Copyright (c) 2026 Neuron_SP Project. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Addresses issue #23: fused RoPE for heterogeneous head counts

/*
 * fused_rope.cu — Fused Q+K RoPE kernel for GQA models
 *
 * ═══════════════════════════════════════════════════════════════════════
 * DESIGN
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Standard GQA (Grouped Query Attention) uses fewer KV heads than Q heads
 * (e.g. Llama-2-70B: 64 Q, 8 KV).  Applying RoPE to Q and K separately
 * means two kernel launches and two independent reads of the cos/sin cache.
 *
 * This kernel fuses both into a single launch:
 *   Grid X = B × (num_heads_q + num_heads_kv)
 *   Grid Y = S
 *
 * Each CTA's blockIdx.x maps to a (batch, head) pair.  If the head index
 * falls in [0, num_heads_q), the CTA processes Q; if in [num_heads_q,
 * num_heads_q + num_heads_kv), it processes K.  The cos/sin cache read
 * at position S is shared across all CTAs at that Y index via L2 cache.
 *
 * Benefits:
 *   - Single kernel launch (PCIe launch overhead is 5-15 µs per launch).
 *   - cos/sin cache stays L2-hot across Q and K CTAs at the same seq pos.
 *   - Better SM occupancy when num_heads_kv < num_heads_q (K CTAs fill gaps).
 *
 * Implementation reuses the rotation helpers and per-SM policy from
 * csrc/hetero_reduce/fused_rope_hetero.cu.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>

#include "fused_rope.h"
#include "../includes/ds_kernel_utils.h"

// ─────────────────────────────────────────────────────────────────────────────
// Per-SM tuning policy (mirrors fused_rope_hetero.cu)
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer> struct QKRoPEPolicy {
    static constexpr int kBlockSize      = 256;
    static constexpr int kMinBlocksPerSM = 2;
    static constexpr int kPairs          = 4;
};
template <> struct QKRoPEPolicy<90> {
    static constexpr int kBlockSize      = 256;
    static constexpr int kMinBlocksPerSM = 4;
    static constexpr int kPairs          = 4;
};
template <> struct QKRoPEPolicy<120> {
    static constexpr int kBlockSize      = 512;
    static constexpr int kMinBlocksPerSM = 4;
    static constexpr int kPairs          = 4;
};

// ─────────────────────────────────────────────────────────────────────────────
// Rotation helpers
// ─────────────────────────────────────────────────────────────────────────────

DS_D_INLINE void qk_sincosf(float theta, float* s, float* c)
{
    __sincosf(theta, s, c);
}

DS_D_INLINE float qk_inv_freq(int k, int head_dim, float base)
{
    return __expf(-2.f * (float)k / (float)head_dim * __logf(base));
}

DS_D_INLINE void qk_rotate_pair(float x, float y, float sin_t, float cos_t,
                                  float& xp, float& yp)
{
    xp = __fmaf_rn(x, cos_t, -y * sin_t);
    yp = __fmaf_rn(x, sin_t,  y * cos_t);
}

// ─────────────────────────────────────────────────────────────────────────────
// Fused Q+K RoPE kernel — NeoX style (cached)
//
// Grid:  (B × (Hq + Hkv), S)
// Block: QKRoPEPolicy<SmVer>::kBlockSize
//
// blockIdx.x ∈ [0, B × (Hq + Hkv))
//   If head_global < Hq: process Q tensor
//   If head_global >= Hq: process K tensor (head_kv = head_global - Hq)
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer>
__global__ void
__launch_bounds__(QKRoPEPolicy<SmVer>::kBlockSize, QKRoPEPolicy<SmVer>::kMinBlocksPerSM)
fused_rope_qk_neox_kernel(
    __nv_bfloat16* __restrict__       q_output,
    __nv_bfloat16* __restrict__       k_output,
    const __nv_bfloat16* __restrict__ q_input,
    const __nv_bfloat16* __restrict__ k_input,
    const float* __restrict__         cos_cache,
    const float* __restrict__         sin_cache,
    int batch,
    int seq_len,
    int num_heads_q,
    int num_heads_kv,
    int head_dim)
{
    constexpr int kBS    = QKRoPEPolicy<SmVer>::kBlockSize;
    constexpr int kPairs = QKRoPEPolicy<SmVer>::kPairs;

    const int total_heads = num_heads_q + num_heads_kv;
    const int bh_idx  = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int b       = bh_idx / total_heads;
    const int h_global = bh_idx % total_heads;

    if (b >= batch || seq_idx >= seq_len) return;

    const int half_dim = head_dim / 2;

    // Determine whether this CTA processes Q or K.
    const bool is_q = (h_global < num_heads_q);
    const int  h    = is_q ? h_global : (h_global - num_heads_q);
    const int  H    = is_q ? num_heads_q : num_heads_kv;

    const __nv_bfloat16* in_base  = is_q ? q_input  : k_input;
          __nv_bfloat16* out_base = is_q ? q_output : k_output;

    const size_t row_offset = ((size_t)b * seq_len + seq_idx) * H * head_dim
                            + (size_t)h * head_dim;

    const __nv_bfloat16* in_row  = in_base  + row_offset;
          __nv_bfloat16* out_row = out_base + row_offset;
    const float* cos_row = cos_cache + (size_t)seq_idx * half_dim;
    const float* sin_row = sin_cache + (size_t)seq_idx * half_dim;

    for (int k0 = (int)threadIdx.x * kPairs; k0 < half_dim; k0 += kBS * kPairs) {
        const int remaining = half_dim - k0;
        const int n = (remaining >= kPairs) ? kPairs : remaining;

        float xv[kPairs], yv[kPairs], c[kPairs], s[kPairs];

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

        float xp[kPairs], yp[kPairs];
        #pragma unroll
        for (int i = 0; i < kPairs; ++i) {
            if (i < n)
                qk_rotate_pair(xv[i], yv[i], s[i], c[i], xp[i], yp[i]);
        }

        #pragma unroll
        for (int i = 0; i < kPairs; ++i) {
            if (i < n) {
                const int k = k0 + i;
                out_row[k]            = __float2bfloat16(xp[i]);
                out_row[k + half_dim] = __float2bfloat16(yp[i]);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fused Q+K RoPE kernel — GPT-J interleaved style (cached)
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer>
__global__ void
__launch_bounds__(QKRoPEPolicy<SmVer>::kBlockSize, QKRoPEPolicy<SmVer>::kMinBlocksPerSM)
fused_rope_qk_gptj_kernel(
    __nv_bfloat16* __restrict__       q_output,
    __nv_bfloat16* __restrict__       k_output,
    const __nv_bfloat16* __restrict__ q_input,
    const __nv_bfloat16* __restrict__ k_input,
    const float* __restrict__         cos_cache,
    const float* __restrict__         sin_cache,
    int batch,
    int seq_len,
    int num_heads_q,
    int num_heads_kv,
    int head_dim)
{
    constexpr int kBS    = QKRoPEPolicy<SmVer>::kBlockSize;
    constexpr int kPairs = QKRoPEPolicy<SmVer>::kPairs;

    const int total_heads = num_heads_q + num_heads_kv;
    const int bh_idx  = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int b       = bh_idx / total_heads;
    const int h_global = bh_idx % total_heads;

    if (b >= batch || seq_idx >= seq_len) return;

    const int half_dim = head_dim / 2;
    const bool is_q = (h_global < num_heads_q);
    const int  h    = is_q ? h_global : (h_global - num_heads_q);
    const int  H    = is_q ? num_heads_q : num_heads_kv;

    const __nv_bfloat16* in_base  = is_q ? q_input  : k_input;
          __nv_bfloat16* out_base = is_q ? q_output : k_output;

    const size_t row_offset = ((size_t)b * seq_len + seq_idx) * H * head_dim
                            + (size_t)h * head_dim;

    const __nv_bfloat16* in_row  = in_base  + row_offset;
          __nv_bfloat16* out_row = out_base + row_offset;
    const float* cos_row = cos_cache + (size_t)seq_idx * half_dim;
    const float* sin_row = sin_cache + (size_t)seq_idx * half_dim;

    for (int k0 = (int)threadIdx.x * kPairs; k0 < half_dim; k0 += kBS * kPairs) {
        const int remaining = half_dim - k0;
        const int n = (remaining >= kPairs) ? kPairs : remaining;

        float xv[kPairs], yv[kPairs], c[kPairs], s[kPairs];

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

        float xp[kPairs], yp[kPairs];
        #pragma unroll
        for (int i = 0; i < kPairs; ++i) {
            if (i < n)
                qk_rotate_pair(xv[i], yv[i], s[i], c[i], xp[i], yp[i]);
        }

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

// ─────────────────────────────────────────────────────────────────────────────
// Fused Q+K RoPE kernel — cacheless (on-the-fly sin/cos)
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer, bool kNeoxStyle>
__global__ void
__launch_bounds__(QKRoPEPolicy<SmVer>::kBlockSize, QKRoPEPolicy<SmVer>::kMinBlocksPerSM)
fused_rope_qk_cacheless_kernel(
    __nv_bfloat16* __restrict__       q_output,
    __nv_bfloat16* __restrict__       k_output,
    const __nv_bfloat16* __restrict__ q_input,
    const __nv_bfloat16* __restrict__ k_input,
    int batch,
    int seq_len,
    int num_heads_q,
    int num_heads_kv,
    int head_dim,
    float base,
    int pos_offset)
{
    constexpr int kBS    = QKRoPEPolicy<SmVer>::kBlockSize;
    constexpr int kPairs = QKRoPEPolicy<SmVer>::kPairs;

    const int total_heads = num_heads_q + num_heads_kv;
    const int bh_idx  = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int b       = bh_idx / total_heads;
    const int h_global = bh_idx % total_heads;

    if (b >= batch || seq_idx >= seq_len) return;

    const int half_dim = head_dim / 2;
    const float pos = (float)(seq_idx + pos_offset);
    const bool is_q = (h_global < num_heads_q);
    const int  h    = is_q ? h_global : (h_global - num_heads_q);
    const int  H    = is_q ? num_heads_q : num_heads_kv;

    const __nv_bfloat16* in_base  = is_q ? q_input  : k_input;
          __nv_bfloat16* out_base = is_q ? q_output : k_output;

    const size_t row_offset = ((size_t)b * seq_len + seq_idx) * H * head_dim
                            + (size_t)h * head_dim;

    const __nv_bfloat16* in_row  = in_base  + row_offset;
          __nv_bfloat16* out_row = out_base + row_offset;

    for (int k0 = (int)threadIdx.x * kPairs; k0 < half_dim; k0 += kBS * kPairs) {
        const int remaining = half_dim - k0;
        const int n = (remaining >= kPairs) ? kPairs : remaining;

        #pragma unroll
        for (int i = 0; i < kPairs; ++i) {
            if (i >= n) continue;
            const int k = k0 + i;
            float inv_f = qk_inv_freq(k, half_dim * 2, base);
            float theta = pos * inv_f;
            float sv, cv;
            qk_sincosf(theta, &sv, &cv);

            float xv, yv, xp, yp;
            if constexpr (kNeoxStyle) {
                xv = __bfloat162float(__ldg(in_row + k));
                yv = __bfloat162float(__ldg(in_row + k + half_dim));
                qk_rotate_pair(xv, yv, sv, cv, xp, yp);
                out_row[k]            = __float2bfloat16(xp);
                out_row[k + half_dim] = __float2bfloat16(yp);
            } else {
                xv = __bfloat162float(__ldg(in_row + 2 * k));
                yv = __bfloat162float(__ldg(in_row + 2 * k + 1));
                qk_rotate_pair(xv, yv, sv, cv, xp, yp);
                out_row[2 * k]     = __float2bfloat16(xp);
                out_row[2 * k + 1] = __float2bfloat16(yp);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host-side dispatch
// ─────────────────────────────────────────────────────────────────────────────

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
    const int total_heads = num_heads_q + num_heads_kv;
    dim3 grid(batch * total_heads, seq_len);

    // Cacheless mode: cos_cache or sin_cache is nullptr.
    if (cos_cache == nullptr || sin_cache == nullptr) {
        #define DISPATCH_CACHELESS(SmVer_)                                     \
            do {                                                                \
                constexpr int kBS = QKRoPEPolicy<SmVer_>::kBlockSize;          \
                if (neox_style)                                                 \
                    fused_rope_qk_cacheless_kernel<SmVer_, true>               \
                        <<<grid, kBS, 0, stream>>>(                             \
                            q_output, k_output, q_input, k_input,              \
                            batch, seq_len, num_heads_q, num_heads_kv,         \
                            head_dim, base, pos_offset);                        \
                else                                                            \
                    fused_rope_qk_cacheless_kernel<SmVer_, false>              \
                        <<<grid, kBS, 0, stream>>>(                             \
                            q_output, k_output, q_input, k_input,              \
                            batch, seq_len, num_heads_q, num_heads_kv,         \
                            head_dim, base, pos_offset);                        \
            } while (0)

        if      (sm_version >= 120) { DISPATCH_CACHELESS(120); }
        else if (sm_version >=  90) { DISPATCH_CACHELESS(90);  }
        else                        { DISPATCH_CACHELESS(86);  }
        #undef DISPATCH_CACHELESS
        return;
    }

    // Cached mode.
    #define DISPATCH_CACHED(SmVer_)                                            \
        do {                                                                    \
            constexpr int kBS = QKRoPEPolicy<SmVer_>::kBlockSize;              \
            if (neox_style)                                                     \
                fused_rope_qk_neox_kernel<SmVer_>                              \
                    <<<grid, kBS, 0, stream>>>(                                 \
                        q_output, k_output, q_input, k_input,                  \
                        cos_cache, sin_cache,                                   \
                        batch, seq_len, num_heads_q, num_heads_kv, head_dim);  \
            else                                                                \
                fused_rope_qk_gptj_kernel<SmVer_>                              \
                    <<<grid, kBS, 0, stream>>>(                                 \
                        q_output, k_output, q_input, k_input,                  \
                        cos_cache, sin_cache,                                   \
                        batch, seq_len, num_heads_q, num_heads_kv, head_dim);  \
        } while (0)

    if      (sm_version >= 120) { DISPATCH_CACHED(120); }
    else if (sm_version >=  90) { DISPATCH_CACHED(90);  }
    else                        { DISPATCH_CACHED(86);  }
    #undef DISPATCH_CACHED
}
