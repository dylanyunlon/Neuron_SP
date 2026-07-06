// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * cross_entropy_tp.cu  —  addresses #110
 *
 * Tensor-parallel (TP) cross-entropy loss with fused log-softmax for
 * heterogeneous GPU clusters (SM 8.6 / 9.0 / 12.0).
 *
 * Background
 * ----------
 * In tensor-parallel LLMs the vocabulary projection (logit) tensor is
 * sharded across TP ranks.  Each GPU holds logits for a vocab slice of size
 * V_local = V / tp_size.  Computing cross-entropy requires:
 *
 *   1. Local max  (max_local = max_j logit[j])
 *   2. Global max (max_global = AllReduce_max across TP ranks)
 *   3. Local sum  (sum_exp_local = Σ exp(logit[j] - max_global))
 *   4. Global sum (log_sum_global = log(AllReduce_sum(sum_exp_local)))
 *   5. Per-sample loss = log_sum_global - (logit[label] - max_global)
 *                        where label is in this rank's shard (or 0 otherwise)
 *
 * This file implements Steps 1 and 3 — the device-local phases that run on
 * each GPU before and after the two cross-device AllReduce communications.
 * The AllReduces themselves are orchestrated in Python using the NCCL/hetero
 * backend; this kernel provides the reduction primitives callable from C++.
 *
 * ═══════════════════════════════════════════════════════════════════════
 * ALGORITHMIC DESIGN
 * ═══════════════════════════════════════════════════════════════════════
 *
 * 1. FUSED MAX + SUM_EXP (two-pass Online-Stable algorithm)
 *    Rather than a three-kernel sequence (max → exp → sum), we compute
 *    the numerically-stable log-sum-exp in two intra-block passes:
 *
 *      Pass 1 (fused max + exp_sum):
 *        • Each thread maintains a running (max, partial_sum_exp) pair using
 *          the Milakov & Gimelshein (2018) online numerically-stable trick:
 *            new_max  = max(old_max, x)
 *            new_sum  = old_sum * exp(old_max - new_max) + exp(x - new_max)
 *          This avoids a separate max-only pass with ZERO accuracy loss.
 *        • After per-thread accumulation, a block-level reduction folds
 *          partial (max, sum) pairs using the same online update rule.
 *
 *    No explicit exp-and-store pass is needed; exp values are computed
 *    once during the accumulation and never materialised in DRAM.
 *
 * 2. NUMERICALLY STABLE ACROSS TP RANKS
 *    The kernel writes (local_max, local_sum_exp) as two separate FP32
 *    scalars per sample.  The caller reduces across TP ranks:
 *      global_max     = AllReduce_max(local_max)
 *      global_sum_exp = AllReduce_sum(local_sum_exp * exp(local_max - global_max))
 *    This two-scalar interface matches cuDNN's distributed-softmax protocol
 *    and avoids precision loss from naive sum-then-correct approaches.
 *
 * 3. LABEL-IN-SHARD DETECTION
 *    Each TP rank knows its vocab shard: [shard_offset, shard_offset + V_local).
 *    The kernel checks whether label ∈ shard at load time.  If yes, it
 *    extracts logit[label - shard_offset] as a FP32 scalar (local_logit).
 *    If not, local_logit = 0.f.  After the AllReduce the caller sums local_logit
 *    across ranks (only one rank has a nonzero contribution) to get the true
 *    label logit without a separate gather operation.
 *
 * 4. BACKWARD PASS (softmax gradient in-place)
 *    launch_cross_entropy_tp_backward fills the logit buffer with:
 *      d_logit[j] = (softmax(logit)[j] - 1{j == label}) / batch_size
 *    using log_sum_global and label_logit (already computed by the forward pass)
 *    to avoid recomputing the full softmax denominator.
 *    Written as: d_logit[j] = exp(logit[j] - max_global - log_sum_global)
 *                             - 1{j == label in shard} / batch_size
 *
 * 5. SM-SPECIALISED __launch_bounds__
 *    SM9.0:  256 threads / block, 4 CTAs/SM — abundant register file
 *    SM8.6:  256 threads / block, 2 CTAs/SM — smaller register file
 *    SM12.0: 512 threads / block, 4 CTAs/SM — widest SMs
 *    Shared memory: 2 × float[kMaxWarps] for parallel max+sum reduction,
 *    totalling ≤ 128 bytes per block.
 *
 * 6. VECTORISED LOADS (128-bit)
 *    Each thread loads kVecWidth = 4 FP32 values per iteration (or 8 × BF16
 *    when logits are BF16).  The inner loop is #pragma unroll'd and handled
 *    with the online-stable accumulator so the compiler can pipeline the
 *    vectorised loads with FP32 math without additional synchronisation.
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
#include <float.h>

#include "hetero_reduce.h"
#include "ds_kernel_utils.h"

namespace cg = cooperative_groups;

// ─────────────────────────────────────────────────────────────────────────────
// Section 1: Tuning policy — one specialisation per SM class
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer> struct CETPPolicy;

template <> struct CETPPolicy<86> {
    // A6000: 256-thread blocks, 2 CTAs/SM (6 MB L2 — moderate occupancy)
    static constexpr int kBlockSize       = 256;
    static constexpr int kMinBlocksPerSM  = 2;
    // BF16 logit vector width: 8 × BF16 = 128 bits per load
    static constexpr int kVecWidthBF16    = 8;
    // FP32 logit vector width: 4 × FP32 = 128 bits per load
    static constexpr int kVecWidthFP32    = 4;
};

template <> struct CETPPolicy<90> {
    // H100: 256-thread blocks, 4 CTAs/SM (50 MB L2 — high occupancy)
    static constexpr int kBlockSize       = 256;
    static constexpr int kMinBlocksPerSM  = 4;
    static constexpr int kVecWidthBF16    = 8;
    static constexpr int kVecWidthFP32    = 4;
};

template <> struct CETPPolicy<120> {
    // Blackwell: 512-thread blocks, 4 CTAs/SM (widest SMs)
    static constexpr int kBlockSize       = 512;
    static constexpr int kMinBlocksPerSM  = 4;
    static constexpr int kVecWidthBF16    = 8;
    static constexpr int kVecWidthFP32    = 4;
};

// Generic fallback for future SM versions
template <int SmVer> struct CETPPolicy {
    static constexpr int kBlockSize       = 256;
    static constexpr int kMinBlocksPerSM  = 2;
    static constexpr int kVecWidthBF16    = 8;
    static constexpr int kVecWidthFP32    = 4;
};

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: Online numerically-stable (max, sum_exp) pair reduction
//
//   Uses the Milakov & Gimelshein identity:
//     merge(a, b) where a = (m_a, s_a), b = (m_b, s_b):
//       m_out = max(m_a, m_b)
//       s_out = s_a * exp(m_a - m_out) + s_b * exp(m_b - m_out)
//
//   This reduces to a single exp() call per merge since exactly one of
//   (m_a - m_out) or (m_b - m_out) is zero.
// ─────────────────────────────────────────────────────────────────────────────

struct MaxSumPair {
    float m;   // running max
    float s;   // Σ exp(x - m)
};

DS_D_INLINE MaxSumPair merge_max_sum(MaxSumPair a, MaxSumPair b)
{
    if (a.m >= b.m) {
        // a.m is the new max; b.s needs rescaling
        return {a.m, a.s + b.s * __expf(b.m - a.m)};
    } else {
        // b.m is the new max; a.s needs rescaling
        return {b.m, a.s * __expf(a.m - b.m) + b.s};
    }
}

// Warp-level butterfly reduction over MaxSumPair
DS_D_INLINE MaxSumPair warp_reduce_max_sum(MaxSumPair p)
{
    // XOR masks: 16, 8, 4, 2, 1 — 5 rounds
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        MaxSumPair peer;
        peer.m = __shfl_xor_sync(0xffffffff, p.m, mask);
        peer.s = __shfl_xor_sync(0xffffffff, p.s, mask);
        p = merge_max_sum(p, peer);
    }
    return p;
}

// Block-level reduction: warp butterfly + shared memory exchange
template <int kBlockSize>
DS_D_INLINE MaxSumPair block_reduce_max_sum(
    MaxSumPair          p,
    float* __restrict__ smem_m,      // [kBlockSize / hw_warp_size]
    float* __restrict__ smem_s,      // [kBlockSize / hw_warp_size]
    cg::thread_block&   blk)
{
    constexpr int kMaxWarps = kBlockSize / hw_warp_size;
    const int lane    = threadIdx.x % hw_warp_size;
    const int warp_id = threadIdx.x / hw_warp_size;

    // Stage 1: warp-level butterfly
    p = warp_reduce_max_sum(p);

    // Stage 2: deposit warp results into shared memory
    if (lane == 0) {
        smem_m[warp_id] = p.m;
        smem_s[warp_id] = p.s;
    }
    blk.sync();

    // Stage 3: reduce warp sums in first warp
    MaxSumPair q;
    q.m = (threadIdx.x < kMaxWarps) ? smem_m[threadIdx.x] : -FLT_MAX;
    q.s = (threadIdx.x < kMaxWarps) ? smem_s[threadIdx.x] : 0.f;
    if (warp_id == 0) q = warp_reduce_max_sum(q);

    // Broadcast final result through smem slot 0
    if (threadIdx.x == 0) {
        smem_m[0] = q.m;
        smem_s[0] = q.s;
    }
    blk.sync();
    return {smem_m[0], smem_s[0]};
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 3: Forward kernel — local max + sum_exp over BF16 logit shard
//
//   Inputs
//   ------
//   logits       [batch, v_local]  BF16  — this TP rank's vocab shard
//   labels       [batch]           int32 — global vocab label per sample
//                                          may be outside this shard → 0 contrib
//   shard_offset int               — global vocab index of logits[:,0]
//   v_local      int               — number of vocab elements on this rank
//
//   Outputs (all FP32 scalars, one per sample in batch)
//   -------
//   local_max    [batch]  — max logit in this shard (for AllReduce_max)
//   local_sum_exp[batch]  — Σ exp(logit[j] - local_max) (for AllReduce_sum)
//   local_logit  [batch]  — logit[label - shard_offset] if label in shard,
//                           else 0.f  (for AllReduce_sum → true label logit)
//
//   Grid:  (batch,) blocks — one CTA per sample
//   Block: kBlockSize threads — cover v_local in strides of kBS * kVec
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer>
__global__ void
__launch_bounds__(CETPPolicy<SmVer>::kBlockSize,
                  CETPPolicy<SmVer>::kMinBlocksPerSM)
cross_entropy_tp_forward_kernel(
    // outputs
    float*                            local_max,
    float*                            local_sum_exp,
    float*                            local_logit,
    // inputs
    const __nv_bfloat16* __restrict__ logits,       // [batch, v_local]
    const int*           __restrict__ labels,        // [batch]
    int                               shard_offset,  // global start of this shard
    int                               v_local)       // elements per sample on this rank
{
    using Policy = CETPPolicy<SmVer>;
    constexpr int kVec      = Policy::kVecWidthBF16;  // 8 × BF16 = 128-bit
    constexpr int kBS       = Policy::kBlockSize;
    constexpr int kMaxWarps = kBS / hw_warp_size;

    // Two shared arrays: one for max, one for sum_exp
    __shared__ float smem_m[kMaxWarps];
    __shared__ float smem_s[kMaxWarps];

    cg::thread_block blk = cg::this_thread_block();

    const int row   = blockIdx.x;
    const int label = __ldg(labels + row);

    // Is this label owned by this TP rank?
    const bool label_in_shard = (label >= shard_offset) &&
                                 (label <  shard_offset + v_local);
    const int  local_label    = label - shard_offset;  // valid only if label_in_shard

    const __nv_bfloat16* __restrict__ row_ptr = logits + (size_t)row * v_local;

    // ── Online max+sum accumulation over v_local ──────────────────────────
    MaxSumPair acc = {-FLT_MAX, 0.f};
    float      label_logit_val = 0.f;

    // Vectorised loop: kVec BF16 elements per iteration
    int col = (int)threadIdx.x * kVec;
    for (; col + kVec <= v_local; col += kBS * kVec) {
        const uint4 raw = __ldg(reinterpret_cast<const uint4*>(row_ptr + col));
        const __nv_bfloat16* lp = reinterpret_cast<const __nv_bfloat16*>(&raw);

        #pragma unroll
        for (int v = 0; v < kVec; ++v) {
            float x = __bfloat162float(lp[v]);
            // Accumulate into online (max, sum_exp) pair
            if (x > acc.m) {
                acc.s = acc.s * __expf(acc.m - x) + 1.f;
                acc.m = x;
            } else {
                acc.s += __expf(x - acc.m);
            }
            // Extract label logit (no branch divergence in typical case)
            if (label_in_shard && (col + v) == local_label) {
                label_logit_val = x;
            }
        }
    }

    // Tail loop for non-multiple-of-kVec v_local
    for (; col < v_local; col += kBS) {
        if (col < v_local) {
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
        col += kBS - (int)threadIdx.x * kVec;  // advance correctly
    }

    // ── Block-level reduction of (max, sum_exp) ───────────────────────────
    acc = block_reduce_max_sum<kBS>(acc, smem_m, smem_s, blk);

    // ── Label logit: reduce across threads in this block ──────────────────
    // Only one thread can have label_in_shard && (col == local_label) true.
    // We use a simple warp reduction followed by a shared-mem broadcast.
    // Since only one thread contributes, the sum equals the value.
    float lv = warp_reduce_max_sum({label_logit_val, 0.f}).m;
    // warp_reduce_max_sum computes max, which equals the label logit since
    // all non-contributing threads have 0.f and the label logit may be
    // negative — use a separate sum reduction instead.
    float lv_sum = label_logit_val;
    lv_sum += __shfl_xor_sync(0xffffffff, lv_sum, 16);
    lv_sum += __shfl_xor_sync(0xffffffff, lv_sum,  8);
    lv_sum += __shfl_xor_sync(0xffffffff, lv_sum,  4);
    lv_sum += __shfl_xor_sync(0xffffffff, lv_sum,  2);
    lv_sum += __shfl_xor_sync(0xffffffff, lv_sum,  1);

    // Deposit warp label-logit sums into smem, then reduce across warps
    const int lane    = threadIdx.x % hw_warp_size;
    const int warp_id = threadIdx.x / hw_warp_size;
    if (lane == 0) smem_m[warp_id] = lv_sum;   // reuse smem_m (already synced below)
    blk.sync();

    float block_label_logit = 0.f;
    if (threadIdx.x < kMaxWarps) block_label_logit = smem_m[threadIdx.x];
    block_label_logit += __shfl_xor_sync(0xffffffff, block_label_logit, 16);
    block_label_logit += __shfl_xor_sync(0xffffffff, block_label_logit,  8);
    block_label_logit += __shfl_xor_sync(0xffffffff, block_label_logit,  4);
    block_label_logit += __shfl_xor_sync(0xffffffff, block_label_logit,  2);
    block_label_logit += __shfl_xor_sync(0xffffffff, block_label_logit,  1);

    // ── Write per-sample outputs (thread 0 only) ──────────────────────────
    if (threadIdx.x == 0) {
        local_max    [row] = acc.m;
        local_sum_exp[row] = acc.s;
        local_logit  [row] = block_label_logit;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 4: Loss finalise — compute per-sample loss from AllReduced scalars
//
//   Called after the host has performed:
//     global_max     = AllReduce_max(local_max)      across TP ranks
//     global_sum_exp = AllReduce_sum(local_sum_exp
//                        * exp(local_max - global_max))
//     global_logit   = AllReduce_sum(local_logit)    (only one rank nonzero)
//
//   loss[i] = log(global_sum_exp[i]) + global_max[i] - global_logit[i]
//           = log_softmax normalisation - label logit
//
//   This kernel is tiny (one thread per sample) and is launched with a
//   1-D grid of (batch / kLFBlockSize) blocks.
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int kLFBlockSize = 256;

__global__ void cross_entropy_tp_loss_kernel(
    float*       loss,              // [batch]  output
    const float* global_max,        // [batch]
    const float* global_sum_exp,    // [batch]
    const float* global_logit,      // [batch]
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
//   d_logit[row, j] = (exp(logit[row,j] - max_g - log(sum_exp_g)) -
//                      1{shard_offset + j == label[row]}) / batch_size
//
//   This is written in-place into the logit buffer (forward logits no longer
//   needed after the loss is computed).  Uses __ldg() for read-only logits.
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer>
__global__ void
__launch_bounds__(CETPPolicy<SmVer>::kBlockSize,
                  CETPPolicy<SmVer>::kMinBlocksPerSM)
cross_entropy_tp_backward_kernel(
    __nv_bfloat16* __restrict__       d_logits,       // [batch, v_local] (in-place)
    const __nv_bfloat16* __restrict__ logits,          // [batch, v_local] (read-only)
    const int*           __restrict__ labels,           // [batch]
    const float*         __restrict__ global_max,       // [batch]
    const float*         __restrict__ log_sum_exp,      // [batch]  log of global sum
    int                               shard_offset,
    int                               v_local,
    float                             inv_batch)        // 1.f / batch_size
{
    using Policy = CETPPolicy<SmVer>;
    constexpr int kVec = Policy::kVecWidthBF16;  // 8
    constexpr int kBS  = Policy::kBlockSize;

    const int row     = blockIdx.x;
    const int label   = __ldg(labels + row);
    const float max_g = __ldg(global_max   + row);
    const float lse   = __ldg(log_sum_exp  + row);  // log(global_sum_exp) already

    const bool label_in_shard = (label >= shard_offset) &&
                                 (label <  shard_offset + v_local);
    const int local_label = label - shard_offset;

    const __nv_bfloat16* __restrict__ row_in  = logits   + (size_t)row * v_local;
          __nv_bfloat16* __restrict__ row_out = d_logits + (size_t)row * v_local;

    // Vectorised loop: 8 BF16 per thread per iteration
    for (int col = (int)threadIdx.x * kVec; col + kVec <= v_local; col += kBS * kVec) {
        const uint4 raw = __ldg(reinterpret_cast<const uint4*>(row_in + col));
        const __nv_bfloat16* lp = reinterpret_cast<const __nv_bfloat16*>(&raw);

        __nv_bfloat16 out_buf[kVec];
        #pragma unroll
        for (int v = 0; v < kVec; ++v) {
            float logit_val = __bfloat162float(lp[v]);
            // Numerically stable softmax: exp(logit - max_g - lse)
            float softmax_j = __expf(logit_val - max_g - lse);
            // Subtract 1/batch for the true label element
            float grad = softmax_j;
            if (label_in_shard && (col + v) == local_label) {
                grad -= 1.f;
            }
            out_buf[v] = __float2bfloat16(grad * inv_batch);
        }
        *reinterpret_cast<uint4*>(row_out + col) =
            *reinterpret_cast<const uint4*>(out_buf);
    }

    // Tail: handle remainder elements (when v_local not divisible by kVec)
    for (int col = ((v_local / (kBS * kVec)) * kBS + (int)threadIdx.x) * kVec;
         col < v_local; col++) {
        if (col < v_local) {
            float logit_val = __bfloat162float(__ldg(row_in + col));
            float grad      = __expf(logit_val - max_g - lse);
            if (label_in_shard && col == local_label) grad -= 1.f;
            row_out[col] = __float2bfloat16(grad * inv_batch);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 6: Host-side launch wrappers
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
    // Grid: one block per sample.  Block: SM-specialised block size.
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
