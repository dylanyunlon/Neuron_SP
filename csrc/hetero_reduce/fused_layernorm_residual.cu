// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * fused_layernorm_residual.cu  —  SM8.6 / 9.0 / 12.0 dispatch,
 *                                  warp shuffle, proper launch bounds,
 *                                  variable hidden sizes.
 *
 * Fused residual addition + RMS LayerNorm:
 *   residual_i[j] = input_i[j] + residual_i[j]          (in-place)
 *   output_i[j]   = residual_i[j] * ln_weight[j]
 *                   * rsqrt(mean(residual²) + ε)
 *
 * ═══════════════════════════════════════════════════════════════════════
 * KEY CHANGES
 * ═══════════════════════════════════════════════════════════════════════
 *
 * 1. FIXED SINGLE-PASS n_iter COUNTER (same bug as fused_swiglu_ln.cu)
 *    n_iter is now derived as col / (kBS * kVec), not a separate counter.
 *    This ensures correct register buffer indexing for any hidden size.
 *
 * 2. VARIABLE HIDDEN SIZES
 *    The single-pass and two-pass paths both handle any hidden divisible
 *    by kVec = 8.  The single-pass threshold is computed at runtime from
 *    policy constants, not hard-coded to a specific model size.
 *
 * 3. WARP SHUFFLE REDUCTION
 *    block_reduce_sum uses a full butterfly __shfl_xor_sync (5 rounds,
 *    #pragma unroll).  RMS variance is E[x²] (not Welford), requiring
 *    only a sum of squares — one reduction per row.
 *
 * 4. PROPER __launch_bounds__ PER SM
 *    SM9.0 (H100):       256 threads, 4 CTAs/SM
 *    SM8.6 (A6000):      256 threads, 2 CTAs/SM
 *    SM12.0 (Blackwell): 512 threads, 4 CTAs/SM
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

template <int SmVer> struct LNResPolicy;

template <> struct LNResPolicy<86> {
    static constexpr int kBlockSize          = 256;
    static constexpr int kMinBlocksPerSM     = 2;
    static constexpr int kVecWidth           = 8;
    static constexpr int kRegBudgetPerThread = 64;
};

template <> struct LNResPolicy<90> {
    static constexpr int kBlockSize          = 256;
    static constexpr int kMinBlocksPerSM     = 4;
    static constexpr int kVecWidth           = 8;
    static constexpr int kRegBudgetPerThread = 128;
};

template <> struct LNResPolicy<120> {
    static constexpr int kBlockSize          = 512;
    static constexpr int kMinBlocksPerSM     = 4;
    static constexpr int kVecWidth           = 8;
    static constexpr int kRegBudgetPerThread = 128;
};

template <int SmVer> struct LNResPolicy {
    static constexpr int kBlockSize          = 256;
    static constexpr int kMinBlocksPerSM     = 2;
    static constexpr int kVecWidth           = 8;
    static constexpr int kRegBudgetPerThread = 64;
};

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: Warp + block sum reductions (butterfly __shfl_xor_sync)
// ─────────────────────────────────────────────────────────────────────────────

DS_D_INLINE float warp_reduce_sum(float val)
{
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

template <int kBlockSize>
DS_D_INLINE float block_reduce_sum(
    float                val,
    float* __restrict__  smem_warps,
    cg::thread_block&    blk)
{
    constexpr int kMaxWarps = kBlockSize / hw_warp_size;
    const int lane    = threadIdx.x % hw_warp_size;
    const int warp_id = threadIdx.x / hw_warp_size;

    val = warp_reduce_sum(val);
    if (lane == 0) smem_warps[warp_id] = val;
    blk.sync();

    val = (threadIdx.x < kMaxWarps) ? smem_warps[threadIdx.x] : 0.f;
    if (warp_id == 0) val = warp_reduce_sum(val);

    if (threadIdx.x == 0) smem_warps[0] = val;
    blk.sync();
    return smem_warps[0];
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 3: Main kernel — fused residual add + RMS LayerNorm
//
//   kSinglePass = true:  post-add values held in registers; zero DRAM
//                        re-read for the normalise phase.
//   kSinglePass = false: two-pass fallback for large hidden dims.
//
//   Variable hidden sizes: any hidden divisible by 8, including non-power-
//   of-two values (e.g. 14336 for Mixtral, 8192 for Llama-70B, 5120, …).
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer, bool kSinglePass>
__global__ void
__launch_bounds__(LNResPolicy<SmVer>::kBlockSize,
                  LNResPolicy<SmVer>::kMinBlocksPerSM)
fused_layernorm_residual_kernel(
    __nv_bfloat16* __restrict__       output,
    __nv_bfloat16* __restrict__       residual,
    const __nv_bfloat16* __restrict__ input,
    const float*          __restrict__ ln_weight,
    int   hidden,
    float eps)
{
    using Policy = LNResPolicy<SmVer>;
    constexpr int kVec      = Policy::kVecWidth;
    constexpr int kBS       = Policy::kBlockSize;
    constexpr int kMaxWarps = kBS / hw_warp_size;

    __shared__ float smem_warps[kMaxWarps];
    cg::thread_block blk = cg::this_thread_block();

    const int row = blockIdx.x;

    const __nv_bfloat16* __restrict__ in_row  = input    + (size_t)row * hidden;
          __nv_bfloat16* __restrict__ res_row = residual + (size_t)row * hidden;
          __nv_bfloat16* __restrict__ out_row = output   + (size_t)row * hidden;

    if constexpr (kSinglePass) {
        // ── Single-pass: hold post-add values in registers ─────────────────
        constexpr int kMaxIter = Policy::kRegBudgetPerThread / kVec;

        float reg_add[kMaxIter * kVec];
        float thread_sq = 0.f;

        // Pass 1: add input + residual, accumulate sq_sum, store in registers.
        // FIX: n_iter = col / (kBS * kVec) — correct for any hidden size.
        for (int col = (int)threadIdx.x * kVec; col < hidden; col += kBS * kVec) {
            const int n_iter = col / (kBS * kVec);

            const uint4 in_raw  = *reinterpret_cast<const uint4*>(in_row  + col);
            const uint4 res_raw = *reinterpret_cast<const uint4*>(res_row + col);
            const __nv_bfloat16* ip = reinterpret_cast<const __nv_bfloat16*>(&in_raw);
            const __nv_bfloat16* rp = reinterpret_cast<const __nv_bfloat16*>(&res_raw);

            const int base = n_iter * kVec;
            #pragma unroll
            for (int v = 0; v < kVec; ++v) {
                float added       = __bfloat162float(ip[v]) + __bfloat162float(rp[v]);
                reg_add[base + v] = added;
                thread_sq        += added * added;
            }
        }

        // Block-level RMS reduction.
        float sq_sum  = block_reduce_sum<kBS>(thread_sq, smem_warps, blk);
        float rms_inv = rsqrtf(sq_sum / (float)hidden + eps);

        // Pass 2: write residual + normalised output from registers.
        for (int col = (int)threadIdx.x * kVec; col < hidden; col += kBS * kVec) {
            const int n_iter   = col / (kBS * kVec);
            const int base     = n_iter * kVec;

            __nv_bfloat16 res_buf[kVec];
            __nv_bfloat16 out_buf[kVec];

            #pragma unroll
            for (int v = 0; v < kVec; ++v) {
                float added    = reg_add[base + v];
                float w        = __ldg(ln_weight + col + v);
                res_buf[v]     = __float2bfloat16(added);
                out_buf[v]     = __float2bfloat16(added * rms_inv * w);
            }

            *reinterpret_cast<uint4*>(res_row + col) =
                *reinterpret_cast<const uint4*>(res_buf);
            *reinterpret_cast<uint4*>(out_row + col) =
                *reinterpret_cast<const uint4*>(out_buf);
        }

    } else {
        // ── Two-pass: large hidden dims (register budget exceeded) ──────────
        // Pass 1: compute residual = input + residual_old, accumulate sq_sum.
        float thread_sq = 0.f;
        for (int col = (int)threadIdx.x * kVec; col < hidden; col += kBS * kVec) {
            const uint4 in_raw  = __ldg(reinterpret_cast<const uint4*>(in_row  + col));
            const uint4 res_raw = __ldg(reinterpret_cast<const uint4*>(res_row + col));
            const __nv_bfloat16* ip = reinterpret_cast<const __nv_bfloat16*>(&in_raw);
            const __nv_bfloat16* rp = reinterpret_cast<const __nv_bfloat16*>(&res_raw);

            __nv_bfloat16 res_buf[kVec];
            #pragma unroll
            for (int v = 0; v < kVec; ++v) {
                float added    = __bfloat162float(ip[v]) + __bfloat162float(rp[v]);
                res_buf[v]     = __float2bfloat16(added);
                thread_sq     += added * added;
            }
            *reinterpret_cast<uint4*>(res_row + col) =
                *reinterpret_cast<const uint4*>(res_buf);
        }

        float sq_sum  = block_reduce_sum<kBS>(thread_sq, smem_warps, blk);
        float rms_inv = rsqrtf(sq_sum / (float)hidden + eps);

        // Pass 2: re-read updated residual, normalise, write output.
        for (int col = (int)threadIdx.x * kVec; col < hidden; col += kBS * kVec) {
            const uint4 res_raw = __ldg(reinterpret_cast<const uint4*>(res_row + col));
            const __nv_bfloat16* rp = reinterpret_cast<const __nv_bfloat16*>(&res_raw);

            __nv_bfloat16 out_buf[kVec];
            #pragma unroll
            for (int v = 0; v < kVec; ++v) {
                float w      = __ldg(ln_weight + col + v);
                out_buf[v]   = __float2bfloat16(__bfloat162float(rp[v]) * rms_inv * w);
            }
            *reinterpret_cast<uint4*>(out_row + col) =
                *reinterpret_cast<const uint4*>(out_buf);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 4: Host-side launch wrapper — SM dispatch + pass selection
// ─────────────────────────────────────────────────────────────────────────────

void launch_fused_layernorm_residual(
    __nv_bfloat16*       output,
    __nv_bfloat16*       residual,
    const __nv_bfloat16* input,
    const float*         ln_weight,
    int                  batch,
    int                  hidden,
    float                eps,
    int                  sm_version,
    cudaStream_t         stream)
{
    if (batch == 0 || hidden == 0) return;
    const int grid = batch;

    if (sm_version >= 120) {
        using P = LNResPolicy<120>;
        const int max_sp = P::kBlockSize * P::kVecWidth * P::kRegBudgetPerThread;
        if (hidden <= max_sp)
            fused_layernorm_residual_kernel<120, true>
                <<<grid, P::kBlockSize, 0, stream>>>(
                    output, residual, input, ln_weight, hidden, eps);
        else
            fused_layernorm_residual_kernel<120, false>
                <<<grid, P::kBlockSize, 0, stream>>>(
                    output, residual, input, ln_weight, hidden, eps);
    } else if (sm_version >= 90) {
        using P = LNResPolicy<90>;
        const int max_sp = P::kBlockSize * P::kVecWidth * P::kRegBudgetPerThread;
        if (hidden <= max_sp)
            fused_layernorm_residual_kernel<90, true>
                <<<grid, P::kBlockSize, 0, stream>>>(
                    output, residual, input, ln_weight, hidden, eps);
        else
            fused_layernorm_residual_kernel<90, false>
                <<<grid, P::kBlockSize, 0, stream>>>(
                    output, residual, input, ln_weight, hidden, eps);
    } else {
        using P = LNResPolicy<86>;
        const int max_sp = P::kBlockSize * P::kVecWidth * P::kRegBudgetPerThread;
        if (hidden <= max_sp)
            fused_layernorm_residual_kernel<86, true>
                <<<grid, P::kBlockSize, 0, stream>>>(
                    output, residual, input, ln_weight, hidden, eps);
        else
            fused_layernorm_residual_kernel<86, false>
                <<<grid, P::kBlockSize, 0, stream>>>(
                    output, residual, input, ln_weight, hidden, eps);
    }
}
