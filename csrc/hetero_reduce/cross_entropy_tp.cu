// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * cross_entropy_tp.cu  —  NeurIPS 2026 DES-LOC production kernel
 *
 * Tensor-parallel (TP) cross-entropy with fused online log-softmax.
 *
 * ═══════════════════════════════════════════════════════════════════════
 * ALGORITHMIC DESIGN
 * ═══════════════════════════════════════════════════════════════════════
 *
 * 1. ONLINE MILAKOV-GIMELSHEIN (2018) NUMERICALLY STABLE LOG-SOFTMAX
 *    Single-pass accumulation of (max_i, sum_exp_i) using the merge rule:
 *      merge((m_a, s_a), (m_b, s_b)):
 *        m = max(m_a, m_b)
 *        s = s_a * exp(m_a - m) + s_b * exp(m_b - m)
 *    This eliminates the need for a separate max-finding pass — the max and
 *    sum_exp are computed in ONE pass over the logit shard, not two.
 *    Cost: 1 extra __expf() per element vs. the naive two-pass approach,
 *    but saves the entire second pass of DRAM reads.
 *
 * 2. WARP-BUTTERFLY + SMEM CROSS-WARP REDUCTION
 *    Stage 1: 5-round XOR butterfly within each warp (zero smem):
 *      for mask in [16, 8, 4, 2, 1]:
 *        peer = shfl_xor(p, mask)
 *        p = merge(p, peer)
 *    Stage 2: lane-0 deposits into smem_m/smem_s; block sync; first warp
 *    loads kMaxWarps pairs and runs a second butterfly.
 *
 * 3. SM-CONDITIONAL TILE SIZES (KernelPolicy from hetero_reduce.cu style)
 *    SM8.6:  256 threads, 2 CTAs/SM — conservative for A6000 register file
 *    SM9.0:  256 threads, 4 CTAs/SM — H100 abundant registers + HBM3
 *    SM12.0: 512 threads, 4 CTAs/SM — Blackwell widest SMs
 *    Shared memory: 2 × float[kMaxWarps]  ≤ 128 bytes per block.
 *
 * 4. VECTORISED BF16 LOADS (128-bit, 8 elements per load)
 *    The inner accumulation loop loads uint4 (8 × BF16 = 128 bits) and
 *    processes them with #pragma unroll 8 — compiler pipelines 4 loads per
 *    cycle on SM9.0/SM12.0 (4-issue scheduler).
 *
 * 5. BACKWARD PASS — IN-PLACE SOFTMAX GRADIENT
 *    d_logit[j] = (exp(logit[j] - max_g - log_sum) - 1{j==label}) / batch
 *    Written in-place into the BF16 logit buffer after forward is complete.
 *    Uses 128-bit reads + writes for maximum memory bandwidth.
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
// Section 1: Per-SM tuning policy
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer> struct CETPPolicy {
    static constexpr int kBlockSize      = 256;
    static constexpr int kMinBlocksPerSM = 2;
    static constexpr int kVecBF16        = 8;   // elements per 128-bit BF16 load
};
template <> struct CETPPolicy<86> {
    static constexpr int kBlockSize      = 256;
    static constexpr int kMinBlocksPerSM = 2;
    static constexpr int kVecBF16        = 8;
};
template <> struct CETPPolicy<90> {
    static constexpr int kBlockSize      = 256;
    static constexpr int kMinBlocksPerSM = 4;
    static constexpr int kVecBF16        = 8;
};
template <> struct CETPPolicy<120> {
    static constexpr int kBlockSize      = 512;
    static constexpr int kMinBlocksPerSM = 4;
    static constexpr int kVecBF16        = 8;
};

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: Online (max, sum_exp) pair — Milakov-Gimelshein merge
// ─────────────────────────────────────────────────────────────────────────────

struct MaxSumPair {
    float m;   // running max
    float s;   // Σ exp(x_i - m)
};

// Merge two (max, sum_exp) pairs — the core of the online-stable algorithm.
// Uses only ONE __expf() call per merge (not two) since exactly one of
// (m_a - m_out) or (m_b - m_out) is zero.
DS_D_INLINE MaxSumPair merge_max_sum(MaxSumPair a, MaxSumPair b)
{
    if (a.m >= b.m)
        return { a.m, a.s + b.s * __expf(b.m - a.m) };
    else
        return { b.m, a.s * __expf(a.m - b.m) + b.s };
}

// Warp-level butterfly reduction of (max, sum_exp) pair.
// 5 rounds of XOR shuffle — produces correct result in all 32 lanes.
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

// Block-level reduction: warp butterfly → smem deposit → second warp butterfly.
// Returns final pair in ALL threads (broadcast via smem slot 0).
template <int kBlockSize>
DS_D_INLINE MaxSumPair block_reduce_max_sum(
    MaxSumPair          p,
    float* __restrict__ smem_m,   // [kBlockSize / 32]
    float* __restrict__ smem_s,   // [kBlockSize / 32]
    cg::thread_block&   blk)
{
    constexpr int kMaxWarps = kBlockSize / hw_warp_size;
    const int lane    = threadIdx.x % hw_warp_size;
    const int warp_id = threadIdx.x / hw_warp_size;

    // Stage 1: warp butterfly (no smem).
    p = warp_reduce_max_sum(p);

    // Stage 2: deposit warp results into smem.
    if (lane == 0) {
        smem_m[warp_id] = p.m;
        smem_s[warp_id] = p.s;
    }
    blk.sync();

    // Stage 3: first warp reduces smem entries.
    MaxSumPair q;
    q.m = (threadIdx.x < kMaxWarps) ? smem_m[threadIdx.x] : -FLT_MAX;
    q.s = (threadIdx.x < kMaxWarps) ? smem_s[threadIdx.x] : 0.f;
    if (warp_id == 0) q = warp_reduce_max_sum(q);

    // Broadcast via smem slot 0.
    if (threadIdx.x == 0) {
        smem_m[0] = q.m;
        smem_s[0] = q.s;
    }
    blk.sync();
    return { smem_m[0], smem_s[0] };
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 3: Forward kernel — local (max, sum_exp, label_logit) per sample
//
//   One CTA per sample (row of [batch, v_local]).
//   Threads stride over v_local in steps of kBS × kVec.
//
//   Online accumulation avoids a separate max-only pass:
//     • Each thread maintains acc = (max_so_far, sum_exp_so_far).
//     • Per element x: acc = merge(acc, (x, 1.f))
//   After block reduction, we have (local_max, local_sum_exp).
//
//   Label logit extraction: only one thread (and only on the rank that owns
//   the label) has a non-zero contribution.  We accumulate label_logit_val
//   as a separate float and reduce via warp shuffle sum.
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer>
__global__ void
__launch_bounds__(CETPPolicy<SmVer>::kBlockSize,
                  CETPPolicy<SmVer>::kMinBlocksPerSM)
cross_entropy_tp_forward_kernel(
    float*                            local_max,
    float*                            local_sum_exp,
    float*                            local_logit,
    const __nv_bfloat16* __restrict__ logits,
    const int*           __restrict__ labels,
    int                               shard_offset,
    int                               v_local)
{
    using Policy = CETPPolicy<SmVer>;
    constexpr int kVec      = Policy::kVecBF16;
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

    // ── Online (max, sum_exp) accumulation over v_local ──
    MaxSumPair acc = { -FLT_MAX, 0.f };
    float label_logit_val = 0.f;

    // Vectorised loop: 8 BF16 per thread per step.
    for (int col = (int)threadIdx.x * kVec; col < v_local; col += kBS * kVec) {
        // Guard for incomplete final vector.
        if (col + kVec <= v_local) {
            const uint4 raw = __ldg(reinterpret_cast<const uint4*>(row_ptr + col));
            const __nv_bfloat16* lp = reinterpret_cast<const __nv_bfloat16*>(&raw);
            #pragma unroll
            for (int v = 0; v < kVec; ++v) {
                const float x = __bfloat162float(lp[v]);
                // Online merge: acc = merge(acc, (x, 1.f))
                acc = merge_max_sum(acc, { x, 1.f });
                // Extract label logit if this element is the true label.
                if (label_in_shard && (col + v) == local_label)
                    label_logit_val = x;
            }
        } else {
            // Scalar tail of this thread's elements.
            for (int v = 0; v < kVec && col + v < v_local; ++v) {
                const float x = __bfloat162float(__ldg(row_ptr + col + v));
                acc = merge_max_sum(acc, { x, 1.f });
                if (label_in_shard && (col + v) == local_label)
                    label_logit_val = x;
            }
        }
    }

    // ── Block-level (max, sum_exp) reduction ──
    acc = block_reduce_max_sum<kBS>(acc, smem_m, smem_s, blk);

    // ── Label-logit reduction across the block ──
    // Only one thread contributed a non-zero label_logit_val; we reduce
    // via warp shuffle sum (all lanes sum → result in lane 0) then
    // cross-warp via smem reuse.
    const int lane    = threadIdx.x % hw_warp_size;
    const int warp_id = threadIdx.x / hw_warp_size;

    float lv = label_logit_val;
    // Warp sum butterfly (5 rounds).
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        lv += __shfl_xor_sync(0xffffffff, lv, mask);

    if (lane == 0) smem_m[warp_id] = lv;  // reuse smem_m (already done with max)
    blk.sync();

    float block_label_logit = (threadIdx.x < kMaxWarps) ? smem_m[threadIdx.x] : 0.f;
    if (warp_id == 0) {
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            block_label_logit += __shfl_xor_sync(0xffffffff, block_label_logit, mask);
    }
    if (threadIdx.x == 0) smem_m[0] = block_label_logit;
    blk.sync();
    block_label_logit = smem_m[0];

    // ── Write outputs (thread 0 only) ──
    if (threadIdx.x == 0) {
        local_max    [row] = acc.m;
        local_sum_exp[row] = acc.s;
        local_logit  [row] = block_label_logit;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 4: Loss finalise kernel
//
//   loss[i] = log(global_sum_exp[i]) + global_max[i] - global_logit[i]
//
//   Tiny kernel: one thread per sample.  Grid = ceil(batch / 256).
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
    const float g_max = __ldg(global_max     + i);
    const float g_sum = __ldg(global_sum_exp + i);
    const float g_log = __ldg(global_logit   + i);
    loss[i] = __logf(g_sum) + g_max - g_log;
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 5: Backward kernel — in-place softmax gradient
//
//   d_logit[row, j] = (exp(logit[row,j] - max_g - log_sum_g)
//                     - 1{shard_offset+j == label[row]}) / batch_size
//
//   Written in-place using 128-bit vectorised reads and writes.
//   Template SmVer → block size + min-CTAs-per-SM.
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
    constexpr int kVec = Policy::kVecBF16;
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

    for (int col = (int)threadIdx.x * kVec; col < v_local; col += kBS * kVec) {
        if (col + kVec <= v_local) {
            const uint4 raw = __ldg(reinterpret_cast<const uint4*>(row_in + col));
            const __nv_bfloat16* lp = reinterpret_cast<const __nv_bfloat16*>(&raw);

            __nv_bfloat16 out_buf[kVec];
            #pragma unroll
            for (int v = 0; v < kVec; ++v) {
                float logit_v = __bfloat162float(lp[v]);
                // Numerically stable softmax: exp(logit - max_g - log(sum_exp))
                float grad = __expf(logit_v - max_g - lse);
                // Subtract 1 for the ground-truth class (on the rank that owns it).
                if (label_in_shard && (col + v) == local_label)
                    grad -= 1.f;
                out_buf[v] = __float2bfloat16(grad * inv_batch);
            }
            // 128-bit store.
            *reinterpret_cast<uint4*>(row_out + col) =
                *reinterpret_cast<const uint4*>(out_buf);
        } else {
            // Scalar tail.
            for (int v = 0; v < kVec && col + v < v_local; ++v) {
                float logit_v = __bfloat162float(__ldg(row_in + col + v));
                float grad = __expf(logit_v - max_g - lse);
                if (label_in_shard && (col + v) == local_label)
                    grad -= 1.f;
                row_out[col + v] = __float2bfloat16(grad * inv_batch);
            }
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
    if (batch <= 0) return;  // BUG-FIX: guard zero-grid launch
    // One CTA per sample; all samples are fully independent.
    if (sm_version >= 120) {
        using P = CETPPolicy<120>;
        cross_entropy_tp_forward_kernel<120>
            <<<batch, P::kBlockSize, 0, stream>>>(
                local_max, local_sum_exp, local_logit,
                logits, labels, shard_offset, v_local);
    DS_LAUNCH_CHECK(stream);
    } else if (sm_version >= 90) {
        using P = CETPPolicy<90>;
        cross_entropy_tp_forward_kernel<90>
            <<<batch, P::kBlockSize, 0, stream>>>(
                local_max, local_sum_exp, local_logit,
                logits, labels, shard_offset, v_local);
    DS_LAUNCH_CHECK(stream);
    } else {
        using P = CETPPolicy<86>;
        cross_entropy_tp_forward_kernel<86>
            <<<batch, P::kBlockSize, 0, stream>>>(
                local_max, local_sum_exp, local_logit,
                logits, labels, shard_offset, v_local);
    DS_LAUNCH_CHECK(stream);
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
    if (batch <= 0) return;  // BUG-FIX: guard zero-grid launch
    const int grid = (batch + kLFBlockSize - 1) / kLFBlockSize;
    cross_entropy_tp_loss_kernel<<<std::max(grid,1), kLFBlockSize, 0, stream>>>(
        loss, global_max, global_sum_exp, global_logit, batch);
    DS_LAUNCH_CHECK(stream);
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
    if (batch <= 0) return;  // BUG-FIX: guard zero-grid launch
    if (sm_version >= 120) {
        using P = CETPPolicy<120>;
        cross_entropy_tp_backward_kernel<120>
            <<<batch, P::kBlockSize, 0, stream>>>(
                d_logits, logits, labels,
                global_max, log_sum_exp,
                shard_offset, v_local, inv_batch);
    DS_LAUNCH_CHECK(stream);
    } else if (sm_version >= 90) {
        using P = CETPPolicy<90>;
        cross_entropy_tp_backward_kernel<90>
            <<<batch, P::kBlockSize, 0, stream>>>(
                d_logits, logits, labels,
                global_max, log_sum_exp,
                shard_offset, v_local, inv_batch);
    DS_LAUNCH_CHECK(stream);
    } else {
        using P = CETPPolicy<86>;
        cross_entropy_tp_backward_kernel<86>
            <<<batch, P::kBlockSize, 0, stream>>>(
                d_logits, logits, labels,
                global_max, log_sum_exp,
                shard_offset, v_local, inv_batch);
    DS_LAUNCH_CHECK(stream);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 7: Heterogeneous VocabPartition support  (issue #141)
//
// Heterogeneous TP splits assign different shard widths (v_local) to each
// rank.  The kernel logic is identical to the uniform path; only v_local and
// shard_offset come from a VocabPartition descriptor instead of being derived
// from V/tp_size.
//
// Two-pass strategy (matches the uniform path, fully allreduce-compatible):
//   Pass A — this file: per-rank local (max, sum_exp, label_logit) over the
//             rank's own shard.  Kernel is templated on SmVer; v_local is now
//             a runtime value that may differ across ranks.
//   Pass B — caller: AllReduce_max(local_max), AllReduce_sum(...), then
//             launch_cross_entropy_tp_loss (unchanged).
//
// compute_hetero_vocab_partition()
// ─────────────────────────────────────────────────────────────────────────────
// Weight table: SM 12.0 → 4, SM 9.0 → 3, everything else → 1.
// Boundaries aligned to kVecBF16=8 elements.  Last rank gets the residual.
// ─────────────────────────────────────────────────────────────────────────────

static inline int sm_to_weight(int sm_version)
{
    if (sm_version >= 120) return 4;
    if (sm_version >= 90)  return 3;
    return 1;
}

void compute_hetero_vocab_partition(VocabPartition* out_parts,
                                     const int*      sm_versions,
                                     int             tp_size,
                                     int             vocab_size)
{
    constexpr int kAlign = 8;   // must match kVecBF16

    // Sum all weights.
    int total_weight = 0;
    for (int r = 0; r < tp_size; ++r)
        total_weight += sm_to_weight(sm_versions[r]);

    // Assign aligned shard widths.  Accumulate offset as we go.
    int offset = 0;
    for (int r = 0; r < tp_size; ++r) {
        int w     = sm_to_weight(sm_versions[r]);
        // Proportional share, floored to kAlign.
        int share = (int)((long long)vocab_size * w / total_weight);
        share = (share / kAlign) * kAlign;
        if (share < kAlign) share = kAlign;  // minimum one vector's worth

        // Last rank absorbs residual to ensure full vocab coverage.
        if (r == tp_size - 1)
            share = vocab_size - offset;

        out_parts[r].v_local      = share;
        out_parts[r].shard_offset = offset;
        out_parts[r].tp_size      = tp_size;
        out_parts[r].rank         = r;

        offset += share;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 7a: Forward hetero kernel
//
// Structurally identical to cross_entropy_tp_forward_kernel<SmVer>.
// The only change is that v_local is now a runtime parameter sourced from
// VocabPartition rather than an implicit constant.
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer>
__global__ void
__launch_bounds__(CETPPolicy<SmVer>::kBlockSize,
                  CETPPolicy<SmVer>::kMinBlocksPerSM)
cross_entropy_tp_forward_hetero_kernel(
    float*                            local_max,
    float*                            local_sum_exp,
    float*                            local_logit,
    const __nv_bfloat16* __restrict__ logits,
    const int*           __restrict__ labels,
    int                               shard_offset,
    int                               v_local)   // runtime — may differ per rank
{
    using Policy = CETPPolicy<SmVer>;
    constexpr int kVec      = Policy::kVecBF16;
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

    const __nv_bfloat16* __restrict__ row_ptr =
        logits + (size_t)row * v_local;

    // ── Pass A: online (max, sum_exp) over this rank's shard ──
    MaxSumPair acc = { -FLT_MAX, 0.f };
    float label_logit_val = 0.f;

    for (int col = (int)threadIdx.x * kVec; col < v_local; col += kBS * kVec) {
        if (col + kVec <= v_local) {
            const uint4 raw = __ldg(reinterpret_cast<const uint4*>(row_ptr + col));
            const __nv_bfloat16* lp = reinterpret_cast<const __nv_bfloat16*>(&raw);
            #pragma unroll
            for (int v = 0; v < kVec; ++v) {
                const float x = __bfloat162float(lp[v]);
                acc = merge_max_sum(acc, { x, 1.f });
                if (label_in_shard && (col + v) == local_label)
                    label_logit_val = x;
            }
        } else {
            for (int v = 0; v < kVec && col + v < v_local; ++v) {
                const float x = __bfloat162float(__ldg(row_ptr + col + v));
                acc = merge_max_sum(acc, { x, 1.f });
                if (label_in_shard && (col + v) == local_label)
                    label_logit_val = x;
            }
        }
    }

    // ── Block reduce: (max, sum_exp) ──
    acc = block_reduce_max_sum<kBS>(acc, smem_m, smem_s, blk);

    // ── Block reduce: label logit ──
    const int lane    = threadIdx.x % hw_warp_size;
    const int warp_id = threadIdx.x / hw_warp_size;

    float lv = label_logit_val;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        lv += __shfl_xor_sync(0xffffffff, lv, mask);

    if (lane == 0) smem_m[warp_id] = lv;
    blk.sync();

    float blk_lv = (threadIdx.x < kMaxWarps) ? smem_m[threadIdx.x] : 0.f;
    if (warp_id == 0) {
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            blk_lv += __shfl_xor_sync(0xffffffff, blk_lv, mask);
    }
    if (threadIdx.x == 0) smem_m[0] = blk_lv;
    blk.sync();
    blk_lv = smem_m[0];

    // ── Write outputs ──
    if (threadIdx.x == 0) {
        local_max    [row] = acc.m;
        local_sum_exp[row] = acc.s;
        local_logit  [row] = blk_lv;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 7b: Backward hetero kernel
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer>
__global__ void
__launch_bounds__(CETPPolicy<SmVer>::kBlockSize,
                  CETPPolicy<SmVer>::kMinBlocksPerSM)
cross_entropy_tp_backward_hetero_kernel(
    __nv_bfloat16* __restrict__       d_logits,
    const __nv_bfloat16* __restrict__ logits,
    const int*           __restrict__ labels,
    const float*         __restrict__ global_max,
    const float*         __restrict__ log_sum_exp,
    int                               shard_offset,
    int                               v_local,   // runtime — may differ per rank
    float                             inv_batch)
{
    using Policy = CETPPolicy<SmVer>;
    constexpr int kVec = Policy::kVecBF16;
    constexpr int kBS  = Policy::kBlockSize;

    const int row     = blockIdx.x;
    const int label   = __ldg(labels + row);
    const float max_g = __ldg(global_max  + row);
    const float lse   = __ldg(log_sum_exp + row);

    const bool label_in_shard = (label >= shard_offset) &&
                                 (label <  shard_offset + v_local);
    const int local_label = label - shard_offset;

    const __nv_bfloat16* __restrict__ row_in  =
        logits   + (size_t)row * v_local;
          __nv_bfloat16* __restrict__ row_out =
        d_logits + (size_t)row * v_local;

    for (int col = (int)threadIdx.x * kVec; col < v_local; col += kBS * kVec) {
        if (col + kVec <= v_local) {
            const uint4 raw = __ldg(reinterpret_cast<const uint4*>(row_in + col));
            const __nv_bfloat16* lp = reinterpret_cast<const __nv_bfloat16*>(&raw);

            __nv_bfloat16 out_buf[kVec];
            #pragma unroll
            for (int v = 0; v < kVec; ++v) {
                float logit_v = __bfloat162float(lp[v]);
                float grad = __expf(logit_v - max_g - lse);
                if (label_in_shard && (col + v) == local_label)
                    grad -= 1.f;
                out_buf[v] = __float2bfloat16(grad * inv_batch);
            }
            *reinterpret_cast<uint4*>(row_out + col) =
                *reinterpret_cast<const uint4*>(out_buf);
        } else {
            for (int v = 0; v < kVec && col + v < v_local; ++v) {
                float logit_v = __bfloat162float(__ldg(row_in + col + v));
                float grad = __expf(logit_v - max_g - lse);
                if (label_in_shard && (col + v) == local_label)
                    grad -= 1.f;
                row_out[col + v] = __float2bfloat16(grad * inv_batch);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 7c: Host-side hetero launch wrappers
// ─────────────────────────────────────────────────────────────────────────────

void launch_cross_entropy_tp_forward_hetero(
    float*               local_max,
    float*               local_sum_exp,
    float*               local_logit,
    const __nv_bfloat16* logits,
    const int*           labels,
    int                  batch,
    VocabPartition       vp,
    int                  sm_version,
    cudaStream_t         stream)
{
    if (batch <= 0) return;  // BUG-FIX: guard zero-grid launch
    const int v_local      = vp.v_local;
    const int shard_offset = vp.shard_offset;

    if (sm_version >= 120) {
        using P = CETPPolicy<120>;
        cross_entropy_tp_forward_hetero_kernel<120>
            <<<batch, P::kBlockSize, 0, stream>>>(
                local_max, local_sum_exp, local_logit,
                logits, labels, shard_offset, v_local);
    DS_LAUNCH_CHECK(stream);
    } else if (sm_version >= 90) {
        using P = CETPPolicy<90>;
        cross_entropy_tp_forward_hetero_kernel<90>
            <<<batch, P::kBlockSize, 0, stream>>>(
                local_max, local_sum_exp, local_logit,
                logits, labels, shard_offset, v_local);
    DS_LAUNCH_CHECK(stream);
    } else {
        using P = CETPPolicy<86>;
        cross_entropy_tp_forward_hetero_kernel<86>
            <<<batch, P::kBlockSize, 0, stream>>>(
                local_max, local_sum_exp, local_logit,
                logits, labels, shard_offset, v_local);
    DS_LAUNCH_CHECK(stream);
    }
}

void launch_cross_entropy_tp_backward_hetero(
    __nv_bfloat16*       d_logits,
    const __nv_bfloat16* logits,
    const int*           labels,
    const float*         global_max,
    const float*         log_sum_exp,
    int                  batch,
    VocabPartition       vp,
    float                inv_batch,
    int                  sm_version,
    cudaStream_t         stream)
{
    if (batch <= 0) return;  // BUG-FIX: guard zero-grid launch
    const int v_local      = vp.v_local;
    const int shard_offset = vp.shard_offset;

    if (sm_version >= 120) {
        using P = CETPPolicy<120>;
        cross_entropy_tp_backward_hetero_kernel<120>
            <<<batch, P::kBlockSize, 0, stream>>>(
                d_logits, logits, labels,
                global_max, log_sum_exp,
                shard_offset, v_local, inv_batch);
    DS_LAUNCH_CHECK(stream);
    } else if (sm_version >= 90) {
        using P = CETPPolicy<90>;
        cross_entropy_tp_backward_hetero_kernel<90>
            <<<batch, P::kBlockSize, 0, stream>>>(
                d_logits, logits, labels,
                global_max, log_sum_exp,
                shard_offset, v_local, inv_batch);
    DS_LAUNCH_CHECK(stream);
    } else {
        using P = CETPPolicy<86>;
        cross_entropy_tp_backward_hetero_kernel<86>
            <<<batch, P::kBlockSize, 0, stream>>>(
                d_logits, logits, labels,
                global_max, log_sum_exp,
                shard_offset, v_local, inv_batch);
    DS_LAUNCH_CHECK(stream);
    }
}
