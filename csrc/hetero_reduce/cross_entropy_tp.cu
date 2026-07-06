// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * cross_entropy_tp.cu  —  addresses #110
 *
 * Tensor-parallel (TP) cross-entropy loss with fused log-softmax for
 * heterogeneous GPU clusters (SM 8.6 / 9.0 / 12.0).
 *
 * ═══════════════════════════════════════════════════════════════════════
 * KEY CHANGES in this revision
 * ═══════════════════════════════════════════════════════════════════════
 *
 * 1. FIXED TAIL-LOOP INDEX CALCULATION
 *    The original vectorised loop stride was (int)threadIdx.x * kVec,
 *    making the scalar tail loop index non-monotone.  Fixed: the tail
 *    loop starts at vec_count * kVec and advances by kBS (one element
 *    per thread per step), with a correct guard (col < v_local).
 *
 * 2. FIXED LABEL-LOGIT ACCUMULATION
 *    The original code tried to use warp_reduce_max_sum() to extract a
 *    sum, which is wrong (max ≠ sum for negative logits).  Replaced with
 *    a clean warp-butterfly __shfl_xor_sync sum that handles negative
 *    logit values correctly.
 *
 * 3. VARIABLE VOCABULARY SIZES
 *    v_local is a runtime parameter; all loops guard with col < v_local.
 *    Works for any vocab shard size, not just powers of two.
 *
 * 4. PROPER __launch_bounds__ PER SM TIER
 *    SM9.0  (H100):      256 threads / block, 4 CTAs/SM
 *    SM8.6  (A6000):     256 threads / block, 2 CTAs/SM
 *    SM12.0 (Blackwell): 512 threads / block, 4 CTAs/SM
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <math.h>
#include <stdint.h>
#include <float.h>

#include "hetero_reduce.h"
#include "ds_kernel_utils.h"

namespace cg = cooperative_groups;

// ─────────────────────────────────────────────────────────────────────────────
// Section 1: Per-SM tuning policy
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer> struct CETPPolicy;

template <> struct CETPPolicy<86> {
    static constexpr int kBlockSize      = 256;
    static constexpr int kMinBlocksPerSM = 2;
    static constexpr int kVecWidthBF16   = 8;   // 8 × BF16 = 128-bit load
    static constexpr int kVecWidthFP32   = 4;   // 4 × FP32 = 128-bit load
};

template <> struct CETPPolicy<90> {
    static constexpr int kBlockSize      = 256;
    static constexpr int kMinBlocksPerSM = 4;
    static constexpr int kVecWidthBF16   = 8;
    static constexpr int kVecWidthFP32   = 4;
};

template <> struct CETPPolicy<120> {
    static constexpr int kBlockSize      = 512;
    static constexpr int kMinBlocksPerSM = 4;
    static constexpr int kVecWidthBF16   = 8;
    static constexpr int kVecWidthFP32   = 4;
};

// Generic fallback
template <int SmVer> struct CETPPolicy {
    static constexpr int kBlockSize      = 256;
    static constexpr int kMinBlocksPerSM = 2;
    static constexpr int kVecWidthBF16   = 8;
    static constexpr int kVecWidthFP32   = 4;
};

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: Online numerically-stable (max, sum_exp) pair
//
//   Milakov & Gimelshein (2018) online stable algorithm:
//     merge(a=(m_a,s_a), b=(m_b,s_b)):
//       m_out = max(m_a, m_b)
//       s_out = s_a * exp(m_a - m_out) + s_b * exp(m_b - m_out)
//   Exactly one exp() per merge (the other operand is exp(0) = 1).
// ─────────────────────────────────────────────────────────────────────────────

struct MaxSumPair {
    float m;   // running max
    float s;   // Σ exp(x - m)
};

DS_D_INLINE MaxSumPair merge_max_sum(MaxSumPair a, MaxSumPair b)
{
    if (a.m >= b.m) {
        return {a.m, a.s + b.s * __expf(b.m - a.m)};
    } else {
        return {b.m, a.s * __expf(a.m - b.m) + b.s};
    }
}

// Warp-level butterfly reduction over MaxSumPair.
// Uses __shfl_xor_sync with masks 16/8/4/2/1 — 5 rounds, #pragma unroll.
DS_D_INLINE MaxSumPair warp_reduce_max_sum(MaxSumPair p)
{
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        MaxSumPair peer;
        peer.m = __shfl_xor_sync(0xffffffff, p.m, mask);
        peer.s = __shfl_xor_sync(0xffffffff, p.s, mask);
        p = merge_max_sum(p, peer);
    }
    return p;
}

// Block-level reduction: warp butterfly → shared memory exchange.
template <int kBlockSize>
DS_D_INLINE MaxSumPair block_reduce_max_sum(
    MaxSumPair          p,
    float* __restrict__ smem_m,
    float* __restrict__ smem_s,
    cg::thread_block&   blk)
{
    constexpr int kMaxWarps = kBlockSize / hw_warp_size;
    const int lane    = threadIdx.x % hw_warp_size;
    const int warp_id = threadIdx.x / hw_warp_size;

    p = warp_reduce_max_sum(p);

    if (lane == 0) {
        smem_m[warp_id] = p.m;
        smem_s[warp_id] = p.s;
    }
    blk.sync();

    MaxSumPair q;
    q.m = (threadIdx.x < kMaxWarps) ? smem_m[threadIdx.x] : -FLT_MAX;
    q.s = (threadIdx.x < kMaxWarps) ? smem_s[threadIdx.x] : 0.f;
    if (warp_id == 0) q = warp_reduce_max_sum(q);

    if (threadIdx.x == 0) {
        smem_m[0] = q.m;
        smem_s[0] = q.s;
    }
    blk.sync();
    return {smem_m[0], smem_s[0]};
}

// Warp-level sum reduction using butterfly __shfl_xor_sync.
// Correct for negative values (unlike max-based reduction).
DS_D_INLINE float warp_reduce_sum(float v)
{
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, mask);
    return v;
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 3: Forward kernel — local max + sum_exp over BF16 logit shard
//
//   Variable v_local: works for any vocab shard size (not power-of-two).
//   Grid:  (batch,) blocks — one CTA per sample.
//   Block: kBlockSize threads covering v_local in strides of kBS * kVec.
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer>
__global__ void
__launch_bounds__(CETPPolicy<SmVer>::kBlockSize,
                  CETPPolicy<SmVer>::kMinBlocksPerSM)
cross_entropy_tp_forward_kernel(
    float*                            local_max,
    float*                            local_sum_exp,
    float*                            local_logit,
    const __nv_bfloat16* __restrict__ logits,      // [batch, v_local]
    const int*           __restrict__ labels,       // [batch]
    int                               shard_offset,
    int                               v_local)
{
    using Policy = CETPPolicy<SmVer>;
    constexpr int kVec      = Policy::kVecWidthBF16;
    constexpr int kBS       = Policy::kBlockSize;
    constexpr int kMaxWarps = kBS / hw_warp_size;

    __shared__ float smem_m[kMaxWarps];
    __shared__ float smem_s[kMaxWarps];

    cg::thread_block blk = cg::this_thread_block();

    const int row   = blockIdx.x;
    const int label = __ldg(labels + row);

    const bool label_in_shard = (label >= shard_offset) &&
                                 (label <  shard_offset + v_local);
    const int  local_label    = label - shard_offset;

    const __nv_bfloat16* __restrict__ row_ptr = logits + (size_t)row * v_local;

    // ── Online max+sum accumulation over variable v_local ─────────────────
    MaxSumPair acc = {-FLT_MAX, 0.f};
    float      label_logit_val = 0.f;

    // Vectorised loop: kVec BF16 elements per thread per iteration.
    // Stride = kBS * kVec, handles any v_local (tail handled separately).
    const int start_vec = (int)threadIdx.x * kVec;
    for (int col = start_vec; col + kVec <= v_local; col += kBS * kVec) {
        const uint4 raw = __ldg(reinterpret_cast<const uint4*>(row_ptr + col));
        const __nv_bfloat16* lp = reinterpret_cast<const __nv_bfloat16*>(&raw);

        #pragma unroll
        for (int v = 0; v < kVec; ++v) {
            float x = __bfloat162float(lp[v]);
            if (x > acc.m) {
                acc.s = acc.s * __expf(acc.m - x) + 1.f;
                acc.m = x;
            } else {
                acc.s += __expf(x - acc.m);
            }
            if (label_in_shard && (col + v) == local_label) {
                label_logit_val = x;
            }
        }
    }

    // Scalar tail: handles variable v_local not divisible by kVec * kBS.
    // Fixed stride: advance by kBS (not kBS*kVec) for scalar elements.
    {
        // Compute correct tail start for this thread.
        const int vec_rounds  = v_local / kVec;       // complete vec rounds total
        const int tail_start  = vec_rounds * kVec;    // first scalar element
        // Each thread handles elements: tail_start + threadIdx.x, + kBS, ...
        for (int col = tail_start + (int)threadIdx.x; col < v_local; col += kBS) {
            float x = __bfloat162float(__ldg(row_ptr + col));
            if (x > acc.m) {
                acc.s = acc.s * __expf(acc.m - x) + 1.f;
                acc.m = x;
            } else {
                acc.s += __expf(x - acc.m);
            }
            if (label_in_shard && col == local_label) {
                label_logit_val = x;
            }
        }
    }

    // ── Block-level reduction of (max, sum_exp) ────────────────────────────
    acc = block_reduce_max_sum<kBS>(acc, smem_m, smem_s, blk);

    // ── Label logit: warp butterfly sum, then cross-warp sum via smem ─────
    // Only one thread can have label_in_shard && col == local_label.
    // Butterfly sum is correct for negative logits (unlike max-based reduce).
    float lv_warp = warp_reduce_sum(label_logit_val);

    const int lane    = threadIdx.x % hw_warp_size;
    const int warp_id = threadIdx.x / hw_warp_size;

    // Reuse smem_m for label logit aggregation (smem_s already done).
    if (lane == 0) smem_m[warp_id] = lv_warp;
    blk.sync();

    float block_label_logit = (threadIdx.x < kMaxWarps) ? smem_m[threadIdx.x] : 0.f;
    block_label_logit = warp_reduce_sum(block_label_logit);

    if (threadIdx.x == 0) {
        local_max    [row] = acc.m;
        local_sum_exp[row] = acc.s;
        local_logit  [row] = block_label_logit;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 4: Loss finalise kernel
//   loss[i] = log(global_sum_exp[i]) + global_max[i] - global_logit[i]
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int kLFBlockSize = 256;

__global__ void cross_entropy_tp_loss_kernel(
    float*       loss,
    const float* global_max,
    const float* global_sum_exp,
    const float* global_logit,
    int          batch)
{
    const int i = blockIdx.x * kLFBlockSize + threadIdx.x;
    if (i >= batch) return;

    float log_norm = __logf(__ldg(global_sum_exp + i)) + __ldg(global_max + i);
    loss[i] = log_norm - __ldg(global_logit + i);
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 5: Backward kernel — softmax gradient w.r.t. local logit shard
//
//   d_logit[row, j] = (exp(logit[row,j] - max_g - lse) -
//                      1{shard_offset + j == label[row]}) / batch_size
//
//   Variable v_local: scalar tail handles any shard size.
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer>
__global__ void
__launch_bounds__(CETPPolicy<SmVer>::kBlockSize,
                  CETPPolicy<SmVer>::kMinBlocksPerSM)
cross_entropy_tp_backward_kernel(
    __nv_bfloat16* __restrict__       d_logits,
    const __nv_bfloat16* __restrict__ logits,
    const int*           __restrict__ labels,
    const float*         __restrict__ global_max,
    const float*         __restrict__ log_sum_exp,
    int                               shard_offset,
    int                               v_local,
    float                             inv_batch)
{
    using Policy = CETPPolicy<SmVer>;
    constexpr int kVec = Policy::kVecWidthBF16;
    constexpr int kBS  = Policy::kBlockSize;

    const int row     = blockIdx.x;
    const int label   = __ldg(labels + row);
    const float max_g = __ldg(global_max  + row);
    const float lse   = __ldg(log_sum_exp + row);

    const bool label_in_shard = (label >= shard_offset) &&
                                 (label <  shard_offset + v_local);
    const int local_label = label - shard_offset;

    const __nv_bfloat16* __restrict__ row_in  = logits   + (size_t)row * v_local;
          __nv_bfloat16* __restrict__ row_out = d_logits + (size_t)row * v_local;

    // Vectorised loop: 8 BF16 per thread per iteration.
    for (int col = (int)threadIdx.x * kVec; col + kVec <= v_local; col += kBS * kVec) {
        const uint4 raw = __ldg(reinterpret_cast<const uint4*>(row_in + col));
        const __nv_bfloat16* lp = reinterpret_cast<const __nv_bfloat16*>(&raw);

        __nv_bfloat16 out_buf[kVec];
        #pragma unroll
        for (int v = 0; v < kVec; ++v) {
            float logit_val = __bfloat162float(lp[v]);
            float softmax_j = __expf(logit_val - max_g - lse);
            float grad = softmax_j;
            if (label_in_shard && (col + v) == local_label) {
                grad -= 1.f;
            }
            out_buf[v] = __float2bfloat16(grad * inv_batch);
        }
        *reinterpret_cast<uint4*>(row_out + col) =
            *reinterpret_cast<const uint4*>(out_buf);
    }

    // Scalar tail for variable v_local sizes.
    {
        const int vec_rounds = v_local / kVec;
        const int tail_start = vec_rounds * kVec;
        for (int col = tail_start + (int)threadIdx.x; col < v_local; col += kBS) {
            float logit_val = __bfloat162float(__ldg(row_in + col));
            float grad      = __expf(logit_val - max_g - lse);
            if (label_in_shard && col == local_label) grad -= 1.f;
            row_out[col] = __float2bfloat16(grad * inv_batch);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 6: Host-side launch wrappers — SM dispatch
// ─────────────────────────────────────────────────────────────────────────────

void launch_cross_entropy_tp_forward(
    float*               local_max,
    float*               local_sum_exp,
    float*               local_logit,
    const __nv_bfloat16* logits,
    const int*           labels,
    int                  batch,
    int                  v_local,
    int                  shard_offset,
    int                  sm_version,
    cudaStream_t         stream)
{
    if (batch == 0) return;

    if (sm_version >= 120) {
        using P = CETPPolicy<120>;
        cross_entropy_tp_forward_kernel<120>
            <<<batch, P::kBlockSize, 0, stream>>>(
                local_max, local_sum_exp, local_logit,
                logits, labels, shard_offset, v_local);
    } else if (sm_version >= 90) {
        using P = CETPPolicy<90>;
        cross_entropy_tp_forward_kernel<90>
            <<<batch, P::kBlockSize, 0, stream>>>(
                local_max, local_sum_exp, local_logit,
                logits, labels, shard_offset, v_local);
    } else {
        using P = CETPPolicy<86>;
        cross_entropy_tp_forward_kernel<86>
            <<<batch, P::kBlockSize, 0, stream>>>(
                local_max, local_sum_exp, local_logit,
                logits, labels, shard_offset, v_local);
    }
}

void launch_cross_entropy_tp_loss(
    float*       loss,
    const float* global_max,
    const float* global_sum_exp,
    const float* global_logit,
    int          batch,
    cudaStream_t stream)
{
    if (batch == 0) return;
    const int grid = (batch + kLFBlockSize - 1) / kLFBlockSize;
    cross_entropy_tp_loss_kernel<<<grid, kLFBlockSize, 0, stream>>>(
        loss, global_max, global_sum_exp, global_logit, batch);
}

void launch_cross_entropy_tp_backward(
    __nv_bfloat16*       d_logits,
    const __nv_bfloat16* logits,
    const int*           labels,
    const float*         global_max,
    const float*         log_sum_exp,
    int                  batch,
    int                  v_local,
    int                  shard_offset,
    float                inv_batch,
    int                  sm_version,
    cudaStream_t         stream)
{
    if (batch == 0) return;

    if (sm_version >= 120) {
        using P = CETPPolicy<120>;
        cross_entropy_tp_backward_kernel<120>
            <<<batch, P::kBlockSize, 0, stream>>>(
                d_logits, logits, labels,
                global_max, log_sum_exp,
                shard_offset, v_local, inv_batch);
    } else if (sm_version >= 90) {
        using P = CETPPolicy<90>;
        cross_entropy_tp_backward_kernel<90>
            <<<batch, P::kBlockSize, 0, stream>>>(
                d_logits, logits, labels,
                global_max, log_sum_exp,
                shard_offset, v_local, inv_batch);
    } else {
        using P = CETPPolicy<86>;
        cross_entropy_tp_backward_kernel<86>
            <<<batch, P::kBlockSize, 0, stream>>>(
                d_logits, logits, labels,
                global_max, log_sum_exp,
                shard_offset, v_local, inv_batch);
    }
}
