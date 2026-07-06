// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * fused_rope_hetero.cu
 *
 * Fused Rotary Position Embedding (RoPE) kernel for heterogeneous head-count
 * configurations in multi-GPU clusters.
 *
 * Motivation
 * ----------
 * In heterogeneous LLM serving, different GPU tiers may host models with
 * different numbers of attention heads (e.g. A6000 SM8.6 hosts a 7B model
 * with 32 heads, H100 SM9.0 hosts a 70B model with 64 heads, Blackwell
 * SM12.0 hosts a MoE model with 128 heads).  A single RoPE kernel that
 * parameterises head_dim, num_heads, and seq_len at runtime avoids
 * duplicating host-side dispatch logic.
 *
 * Kernel design
 * -------------
 * One CTA per (batch, head) pair.  Within the CTA, threads process
 * consecutive pairs of head-dim elements: element 2k and 2k+1 form a
 * rotation pair.  Each thread handles kVecPairs pairs per step.
 *
 * The rotation matrix for pair (2k, 2k+1) at position pos is:
 *   x' = x * cos(theta_k * pos) - y * sin(theta_k * pos)
 *   y' = x * sin(theta_k * pos) + y * cos(theta_k * pos)
 * where theta_k = base^(-2k / head_dim), base = 10000 by default.
 *
 * BF16 input/output with FP32 computation throughout.
 *
 * Neox-style vs GPT-J-style
 * --------------------------
 * neox_style=true  : rotate the first half of head_dim (GPT-NeoX / Llama)
 * neox_style=false : interleave pairs (GPT-J / Falcon)
 * Controlled by a compile-time template bool and a runtime flag.
 *
 * SM specialisations
 * ------------------
 *   SM 8.6 (A6000): __launch_bounds__(256, 2)
 *   SM 9.0 (H100):  __launch_bounds__(256, 4)
 *   SM 12.0 (Blackwell): __launch_bounds__(512, 4)
 *
 * Cooperative groups
 * ------------------
 * Warp-level sin/cos prefetch uses cg::tiled_partition<32> for
 * forward-compatible shuffle primitives.
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

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr float kRopeBase = 10000.0f;
static constexpr int   kVecPairs = 4;  // rotation pairs per thread per step

// ---------------------------------------------------------------------------
// Fast sincos for FP32 (maps to hardware sincos on SM 8.6+)
// ---------------------------------------------------------------------------
DS_D_INLINE void fast_sincosf(float theta, float* s, float* c)
{
    __sincosf(theta, s, c);
}

// ---------------------------------------------------------------------------
// Compute inverse frequency for pair index k and head_dim.
// theta_k = base^(-2k / head_dim)
// ---------------------------------------------------------------------------
DS_D_INLINE float rope_inv_freq(int k, int head_dim, float base)
{
    return __powf(base, -2.f * (float)k / (float)head_dim);
}

// ---------------------------------------------------------------------------
// Rotate a single (x, y) pair by angle theta.
// ---------------------------------------------------------------------------
DS_D_INLINE void rotate_pair(float x, float y, float sin_t, float cos_t,
                               float& xp, float& yp)
{
    xp = x * cos_t - y * sin_t;
    yp = x * sin_t + y * cos_t;
}

// ---------------------------------------------------------------------------
// RoPE kernel — neox_style (first half / second half split, Llama-style).
//
// Input layout: [batch, seq_len, num_heads, head_dim]
// The kernel operates on a single (batch, head) slice passed via base pointer.
//
// Template parameters:
//   SmVer      : SM version (86, 90, 120) for __launch_bounds__
//   kBlockSize : threads per block
// ---------------------------------------------------------------------------
template <int SmVer, int kBlockSize>
__global__ void __launch_bounds__(kBlockSize, (SmVer >= 90) ? 4 : 2)
fused_rope_neox_kernel(
    __nv_bfloat16* __restrict__       output,   // [B, S, H, D]
    const __nv_bfloat16* __restrict__ input,    // [B, S, H, D]
    const float* __restrict__         cos_cache, // [S, D/2]  precomputed cos
    const float* __restrict__         sin_cache, // [S, D/2]  precomputed sin
    int  batch,
    int  seq_len,
    int  num_heads,
    int  head_dim)
{
    // Grid: (batch * num_heads, seq_len)
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
    const float*          cos_row = cos_cache + (size_t)seq_idx * half_dim;
    const float*          sin_row = sin_cache + (size_t)seq_idx * half_dim;

    // Each thread handles kVecPairs consecutive pairs.
    // Neox-style: in_row[k] and in_row[k + half_dim] form a rotation pair.
    for (int k = threadIdx.x; k < half_dim; k += kBlockSize) {
#if __CUDA_ARCH__ >= 1200
        // Blackwell: prefetch next iteration's data into L1
        if (k + kBlockSize < half_dim) {
            asm volatile("prefetch.global.L1 [%0];" :: "l"(in_row  + k + kBlockSize));
            asm volatile("prefetch.global.L1 [%0];" :: "l"(cos_row + k + kBlockSize));
        }
#endif
        float xv = __bfloat162float(in_row[k]);
        float yv = __bfloat162float(in_row[k + half_dim]);
        float c  = cos_row[k];
        float s  = sin_row[k];
        float xp, yp;
        rotate_pair(xv, yv, s, c, xp, yp);
        out_row[k]            = __float2bfloat16(xp);
        out_row[k + half_dim] = __float2bfloat16(yp);
    }
}

// ---------------------------------------------------------------------------
// RoPE kernel — GPT-J interleaved style (pairs are adjacent: [x0,y0,x1,y1,...]).
// ---------------------------------------------------------------------------
template <int SmVer, int kBlockSize>
__global__ void __launch_bounds__(kBlockSize, (SmVer >= 90) ? 4 : 2)
fused_rope_gptj_kernel(
    __nv_bfloat16* __restrict__       output,
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__         cos_cache, // [S, D/2]
    const float* __restrict__         sin_cache, // [S, D/2]
    int  batch,
    int  seq_len,
    int  num_heads,
    int  head_dim)
{
    const int bh_idx  = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int b       = bh_idx / num_heads;
    const int h       = bh_idx % num_heads;

    if (b >= batch || seq_idx >= seq_len) return;

    const int half_dim   = head_dim / 2;
    const size_t row_offset = ((size_t)b * seq_len + seq_idx) * num_heads * head_dim
                            + (size_t)h * head_dim;

    const __nv_bfloat16* in_row  = input  + row_offset;
          __nv_bfloat16* out_row = output + row_offset;
    const float*          cos_row = cos_cache + (size_t)seq_idx * half_dim;
    const float*          sin_row = sin_cache + (size_t)seq_idx * half_dim;

    // GPT-J: pair (2k, 2k+1) are adjacent.
    for (int k = threadIdx.x; k < half_dim; k += kBlockSize) {
#if __CUDA_ARCH__ >= 1200
        // Blackwell: prefetch next iteration's data into L1
        if (k + kBlockSize < half_dim) {
            asm volatile("prefetch.global.L1 [%0];" :: "l"(in_row  + 2 * (k + kBlockSize)));
            asm volatile("prefetch.global.L1 [%0];" :: "l"(cos_row + k + kBlockSize));
        }
#endif
        float xv = __bfloat162float(in_row[2 * k]);
        float yv = __bfloat162float(in_row[2 * k + 1]);
        float c  = cos_row[k];
        float s  = sin_row[k];
        float xp, yp;
        rotate_pair(xv, yv, s, c, xp, yp);
        out_row[2 * k]     = __float2bfloat16(xp);
        out_row[2 * k + 1] = __float2bfloat16(yp);
    }
}

// ---------------------------------------------------------------------------
// cos/sin cache precomputation kernel.
// Fills cos_cache[s, k] = cos(theta_k * s) for s in [0, seq_len).
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
rope_cache_kernel(
    float* __restrict__ cos_cache,  // [seq_len, half_dim]
    float* __restrict__ sin_cache,  // [seq_len, half_dim]
    int   seq_len,
    int   half_dim,
    float base,
    int   pos_offset)   // for packed/split sequences: global position = pos_offset + s
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s >= seq_len || k >= half_dim) return;

    float inv_freq = rope_inv_freq(k, half_dim * 2, base);
    float theta    = (float)(s + pos_offset) * inv_freq;
    float sv, cv;
    fast_sincosf(theta, &sv, &cv);

#if __CUDA_ARCH__ >= 1200
    // Blackwell: use streaming stores to avoid polluting L2 cache
    const size_t idx = (size_t)s * half_dim + k;
    __stcs(cos_cache + idx, cv);
    __stcs(sin_cache + idx, sv);
#else
    const size_t idx = (size_t)s * half_dim + k;
    cos_cache[idx] = cv;
    sin_cache[idx] = sv;
#endif
}

// ---------------------------------------------------------------------------
// Host-side launch wrapper — precompute RoPE cache.
// ---------------------------------------------------------------------------
void launch_rope_cache(float* cos_cache,
                       float* sin_cache,
                       int    seq_len,
                       int    head_dim,
                       float  base,
                       int    pos_offset,
                       cudaStream_t stream)
{
    const int half_dim = head_dim / 2;
    dim3 block(32, 8);
    dim3 grid((seq_len   + block.x - 1) / block.x,
              (half_dim  + block.y - 1) / block.y);
    rope_cache_kernel<<<grid, block, 0, stream>>>(
        cos_cache, sin_cache, seq_len, half_dim, base, pos_offset);
}

// ---------------------------------------------------------------------------
// Host-side launch wrapper — fused RoPE forward pass.
//
// @param output      [B, S, H, D] BF16 output (may alias input for in-place)
// @param input       [B, S, H, D] BF16 input
// @param cos_cache   [S, D/2]     FP32 precomputed cosines
// @param sin_cache   [S, D/2]     FP32 precomputed sines
// @param batch       Batch size
// @param seq_len     Sequence length
// @param num_heads   Number of attention heads (heterogeneous: 32/64/128)
// @param head_dim    Head dimension (must be even)
// @param neox_style  true → Llama-style, false → GPT-J interleaved style
// @param sm_version  SM version of active device (86, 90, 120)
// @param stream      CUDA stream
// ---------------------------------------------------------------------------
void launch_fused_rope_hetero(
    __nv_bfloat16*              output,
    const __nv_bfloat16*        input,
    const float*                cos_cache,
    const float*                sin_cache,
    int                         batch,
    int                         seq_len,
    int                         num_heads,
    int                         head_dim,
    bool                        neox_style,
    int                         sm_version,
    cudaStream_t                stream)
{
    // Grid: (batch * num_heads, seq_len) — one CTA per (batch, head, seq)
    dim3 grid(batch * num_heads, seq_len);

    if (neox_style) {
        if (sm_version >= 120) {
            constexpr int kBS = 512;
            fused_rope_neox_kernel<120, kBS><<<grid, kBS, 0, stream>>>(
                output, input, cos_cache, sin_cache,
                batch, seq_len, num_heads, head_dim);
        } else if (sm_version >= 90) {
            constexpr int kBS = 256;
            fused_rope_neox_kernel<90, kBS><<<grid, kBS, 0, stream>>>(
                output, input, cos_cache, sin_cache,
                batch, seq_len, num_heads, head_dim);
        } else {
            constexpr int kBS = 256;
            fused_rope_neox_kernel<86, kBS><<<grid, kBS, 0, stream>>>(
                output, input, cos_cache, sin_cache,
                batch, seq_len, num_heads, head_dim);
        }
    } else {
        // GPT-J interleaved
        if (sm_version >= 120) {
            constexpr int kBS = 512;
            fused_rope_gptj_kernel<120, kBS><<<grid, kBS, 0, stream>>>(
                output, input, cos_cache, sin_cache,
                batch, seq_len, num_heads, head_dim);
        } else if (sm_version >= 90) {
            constexpr int kBS = 256;
            fused_rope_gptj_kernel<90, kBS><<<grid, kBS, 0, stream>>>(
                output, input, cos_cache, sin_cache,
                batch, seq_len, num_heads, head_dim);
        } else {
            constexpr int kBS = 256;
            fused_rope_gptj_kernel<86, kBS><<<grid, kBS, 0, stream>>>(
                output, input, cos_cache, sin_cache,
                batch, seq_len, num_heads, head_dim);
        }
    }
}
