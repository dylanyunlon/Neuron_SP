// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * fused_layernorm_residual.cu  —  addresses #110
 *
 * Fused pre-LN residual addition + RMS LayerNorm kernel for heterogeneous
 * GPU clusters (SM 8.6 / 9.0 / 12.0).
 *
 * Operation (per row i of [batch × hidden]):
 *   residual_i[j] = input_i[j] + residual_i[j]                 (in-place)
 *   output_i[j]   = residual_i[j] * ln_weight[j]
 *                   * rsqrt(mean(residual²) + ε)
 *
 * This pattern is ubiquitous in transformer decoder blocks where the
 * residual stream accumulates across MHA and MLP sub-layers:
 *
 *   x = attn_out + x                         ← residual add
 *   x = rmsnorm(x) * weight                  ← pre-LN (Llama / Mistral style)
 *
 * Fusing both into a single kernel eliminates one round-trip to DRAM:
 * the post-add value is computed in registers and passed directly to the
 * variance accumulator, saving N × hidden × sizeof(bf16) bytes of DRAM
 * bandwidth relative to a two-kernel approach.
 *
 * ═══════════════════════════════════════════════════════════════════════
 * ALGORITHMIC DESIGN
 * ═══════════════════════════════════════════════════════════════════════
 *
 * 1. SINGLE-PASS IN-REGISTER FUSION (for hidden ≤ kRegBudgetThreshold)
 *    Each thread processes kVecWidth = 8 BF16 elements per iteration.
 *    After loading gate/residual and computing elem = input + residual,
 *    the result is held in a thread-local float register array.  The
 *    squared sum (for RMS) is accumulated without any DRAM write.
 *    After block-level RMS reduction, normalization is applied directly
 *    from the register array — zero additional DRAM reads of the
 *    post-add data.
 *
 *    For hidden > threshold we fall back to a two-pass scheme:
 *      Pass 1: load input+residual, accumulate sq_sum, write residual.
 *      Pass 2: read residual, normalise, write output.
 *    The two-pass path is taken when the register budget would exceed
 *    kRegBudgetPerThread floats per thread.
 *
 * 2. WARP-BUTTERFLY + SHARED-MEMORY BLOCK REDUCTION (identical to
 *    fused_swiglu_ln.cu — zero divergence in critical path)
 *    Welford is NOT used here because variance = E[x²] (not E[(x-μ)²]):
 *    pre-LN uses zero-mean assumption (centering is in the weight bias),
 *    so only E[x²] is needed.  That eliminates the Welford term and one
 *    SHFL per lane.
 *
 * 3. IN-PLACE RESIDUAL WRITEBACK
 *    residual is written through a __restrict__ pointer separate from
 *    input/output.  In the single-pass path, the write is deferred to a
 *    second vectorised store loop (avoiding a write-then-read-back
 *    pattern that would pollute L2 before the normalise phase).
 *    In the two-pass path, the residual write happens in Pass 1 so that
 *    Pass 2 can read it with __ldg() (read-only cache).
 *
 * 4. SM-SPECIALISED __launch_bounds__
 *    SM9.0  (H100):      256-thread blocks, 4 CTAs/SM
 *    SM8.6  (A6000):     256-thread blocks, 2 CTAs/SM
 *    SM12.0 (Blackwell): 512-thread blocks, 4 CTAs/SM
 *    Shared memory: one float[kMaxWarps] array, ≤ 64 bytes per block.
 *
 * 5. DUAL OUTPUT POINTERS
 *    output   [batch, hidden] — normalised result (next sub-layer input)
 *    residual [batch, hidden] — updated residual stream (for skip at
 *                               the next add-and-norm boundary)
 *    They may alias only if output == residual (single-buffer mode).
 *    The kernel handles aliasing correctly because the register file
 *    decouples the read of residual_old and the write of residual_new.
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
// Section 1: Per-SM tuning policy (mirrors fused_swiglu_ln.cu structure)
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer> struct LNResPolicy;

template <> struct LNResPolicy<86> {
    // A6000: 48 GB GDDR6X, 6 MB L2, 256 KB shared mem per SM
    static constexpr int kBlockSize          = 256;
    static constexpr int kMinBlocksPerSM     = 2;
    static constexpr int kVecWidth           = 8;    // BF16 elements per load
    // Register budget per thread (floats): 64 × 4 B = 256 B
    // → max single-pass hidden = 256 × 8 × 64 = 131 072
    static constexpr int kRegBudgetPerThread = 64;
};

template <> struct LNResPolicy<90> {
    // H100: 80 GB HBM3, 50 MB L2, 228 KB shared mem per SM
    static constexpr int kBlockSize          = 256;
    static constexpr int kMinBlocksPerSM     = 4;
    static constexpr int kVecWidth           = 8;
    static constexpr int kRegBudgetPerThread = 128; // 128 × 4 B = 512 B
};

template <> struct LNResPolicy<120> {
    // Blackwell: wider SMs, more registers, 40 MB L2
    static constexpr int kBlockSize          = 512;
    static constexpr int kMinBlocksPerSM     = 4;
    static constexpr int kVecWidth           = 8;
    static constexpr int kRegBudgetPerThread = 128;
};

// Generic fallback for untested SM versions
template <int SmVer> struct LNResPolicy {
    static constexpr int kBlockSize          = 256;
    static constexpr int kMinBlocksPerSM     = 2;
    static constexpr int kVecWidth           = 8;
    static constexpr int kRegBudgetPerThread = 64;
};

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: Warp-level + block-level sum reduction
//   Identical butterfly pattern used by fused_swiglu_ln.cu; see that file
//   for detailed rationale.  Repeated here to keep each TU self-contained.
// ─────────────────────────────────────────────────────────────────────────────

DS_D_INLINE float warp_reduce_sum(float val)
{
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val,  8);
    val += __shfl_xor_sync(0xffffffff, val,  4);
    val += __shfl_xor_sync(0xffffffff, val,  2);
    val += __shfl_xor_sync(0xffffffff, val,  1);
    return val;
}

template <int kBlockSize>
DS_D_INLINE float block_reduce_sum(
    float                val,
    float* __restrict__  smem_warps,   // [kBlockSize / hw_warp_size]
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

    // Broadcast via slot 0 — all threads read the same smem location
    if (threadIdx.x == 0) smem_warps[0] = val;
    blk.sync();
    return smem_warps[0];
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 3: Main kernel — fused residual add + RMS LayerNorm
//
//   Template parameters:
//     SmVer       : SM version for policy dispatch
//     kSinglePass : true  → hold post-add values in registers for zero
//                           extra DRAM reads in the normalise phase.
//                   false → two-pass; write residual in Pass 1, re-read in 2.
//
//   Grid/block: one CTA per row (batch dimension); threads cover hidden.
//
//   Shared memory: float[kMaxWarps] for block reduction (≤ 64 B).
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer, bool kSinglePass>
__global__ void
__launch_bounds__(LNResPolicy<SmVer>::kBlockSize,
                  LNResPolicy<SmVer>::kMinBlocksPerSM)
fused_layernorm_residual_kernel(
    __nv_bfloat16* __restrict__       output,      // [batch, hidden]  LN output
    __nv_bfloat16* __restrict__       residual,    // [batch, hidden]  updated residual
    const __nv_bfloat16* __restrict__ input,       // [batch, hidden]  new contribution
    const float*          __restrict__ ln_weight,   // [hidden]         RMS scale
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

    // Row pointers
    const __nv_bfloat16* __restrict__ in_row  = input    + (size_t)row * hidden;
          __nv_bfloat16* __restrict__ res_row = residual + (size_t)row * hidden;
          __nv_bfloat16* __restrict__ out_row = output   + (size_t)row * hidden;

    // ──────────────────────────────────────────────────────────────────────
    // SINGLE-PASS path — hidden fits in thread registers
    // ──────────────────────────────────────────────────────────────────────
    if constexpr (kSinglePass) {
        constexpr int kMaxIter = Policy::kRegBudgetPerThread / kVec;

        // Register file: store post-add BF16 values as FP32 for normalise pass
        float reg_add[kMaxIter * kVec];
        float thread_sq = 0.f;

        // ── Pass 1: load input + residual, accumulate sq_sum ──────────────
        int n_iter = 0;
        for (int col = (int)threadIdx.x * kVec; col < hidden;
             col += kBS * kVec, ++n_iter) {

            const uint4 in_raw  = *reinterpret_cast<const uint4*>(in_row  + col);
            const uint4 res_raw = *reinterpret_cast<const uint4*>(res_row + col);

            const __nv_bfloat16* ip  = reinterpret_cast<const __nv_bfloat16*>(&in_raw);
            const __nv_bfloat16* rp  = reinterpret_cast<const __nv_bfloat16*>(&res_raw);

            const int base = n_iter * kVec;
            #pragma unroll
            for (int v = 0; v < kVec; ++v) {
                float added         = __bfloat162float(ip[v]) + __bfloat162float(rp[v]);
                reg_add[base + v]   = added;
                thread_sq          += added * added;
            }
        }

        // ── Block-level RMS denominator ───────────────────────────────────
        float sq_sum  = block_reduce_sum<kBS>(thread_sq, smem_warps, blk);
        float rms_inv = rsqrtf(sq_sum / (float)hidden + eps);

        // ── Pass 2: write residual + normalised output from registers ─────
        // Residual write must precede output write in case they alias, but
        // they may only alias when output == residual (single-buffer mode),
        // in which case order does not matter because we read from reg_add.
        n_iter = 0;
        for (int col = (int)threadIdx.x * kVec; col < hidden;
             col += kBS * kVec, ++n_iter) {

            const int base = n_iter * kVec;
            __nv_bfloat16 res_buf[kVec];
            __nv_bfloat16 out_buf[kVec];

            #pragma unroll
            for (int v = 0; v < kVec; ++v) {
                float added    = reg_add[base + v];
                float w        = __ldg(ln_weight + col + v);
                res_buf[v]     = __float2bfloat16(added);
                out_buf[v]     = __float2bfloat16(added * rms_inv * w);
            }

            // 128-bit vectorised store — residual stream
            *reinterpret_cast<uint4*>(res_row + col) =
                *reinterpret_cast<const uint4*>(res_buf);
            // 128-bit vectorised store — LN output
            *reinterpret_cast<uint4*>(out_row + col) =
                *reinterpret_cast<const uint4*>(out_buf);
        }

    } else {
        // ──────────────────────────────────────────────────────────────────
        // TWO-PASS path — large hidden; register budget would overflow
        //   Pass 1: compute residual = input + residual_old, accumulate sq_sum,
        //           write updated residual to DRAM.
        //   Pass 2: read updated residual with __ldg(), normalise, write output.
        // ──────────────────────────────────────────────────────────────────

        // ── Pass 1: residual update + sq_sum ─────────────────────────────
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

        // ── Block-level RMS reduction ─────────────────────────────────────
        float sq_sum  = block_reduce_sum<kBS>(thread_sq, smem_warps, blk);
        float rms_inv = rsqrtf(sq_sum / (float)hidden + eps);

        // ── Pass 2: re-read residual (L2-warm), normalise, write output ───
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
// Section 4: Host-side launch wrapper
//   Selects single-pass vs. two-pass based on register budget.
//   Dispatches to the correct SM specialisation.
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
    // One CTA per row; all rows are independent.
    const int grid = batch;

    // Single-pass threshold: each thread can hold at most
    //   kBlockSize × kVecWidth × kRegBudgetPerThread elements in registers.
    // In practice hidden ∈ {4096, 8192, 14336, 16384} → single-pass for all.

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
        // SM 8.6 (A6000) and any older / unknown SM
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
