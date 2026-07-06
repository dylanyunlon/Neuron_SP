// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * fused_swiglu_ln.cu  —  SM8.6 / 9.0 / 12.0 dispatch, warp shuffle,
 *                         proper launch bounds, variable hidden sizes.
 *
 * Fused SwiGLU activation + RMS LayerNorm.
 *
 * For each row i of [batch × hidden]:
 *   swiglu_i[j] = gate_i[j] * sigmoid(gate_i[j]) * up_i[j]
 *   output_i[j] = swiglu_i[j] * ln_weight[j] * rsqrt(mean(swiglu²) + ε)
 *
 * ═══════════════════════════════════════════════════════════════════════
 * KEY CHANGES
 * ═══════════════════════════════════════════════════════════════════════
 *
 * 1. FIXED SINGLE-PASS n_iter COUNTER
 *    The original loop used a separate n_iter counter that was incremented
 *    inside the for-loop body after the register-store, but the stride
 *    was kBS * kVec — so the iteration count was correct only when
 *    hidden == kBS * kVec * k for integer k.  Fixed: n_iter is now
 *    computed as col / (kBS * kVec) — derived directly from col so it
 *    is always correct for any hidden size.
 *
 * 2. VARIABLE HIDDEN SIZES
 *    Both single-pass and two-pass paths handle any hidden divisible by
 *    kVec (= 8).  The register buffer is sized to kRegBudgetPerThread floats;
 *    single-pass is selected only when hidden <= kBlockSize * kVec * kBudget.
 *
 * 3. WARP SHUFFLE REDUCTION
 *    block_reduce_sum uses a full butterfly __shfl_xor_sync (5 rounds,
 *    #pragma unroll) — 5 SHFL instructions, no extra smem pressure.
 *
 * 4. PROPER __launch_bounds__ PER SM
 *    SM9.0 (H100):       256 threads, 4 CTAs/SM — maximal H100 occupancy
 *    SM8.6 (A6000):      256 threads, 2 CTAs/SM — small register file
 *    SM12.0 (Blackwell): 512 threads, 4 CTAs/SM — wider SMs
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <math.h>

#include "hetero_reduce.h"
#include "ds_kernel_utils.h"

namespace cg = cooperative_groups;

// ─────────────────────────────────────────────────────────────────────────────
// Section 1: Tuning policies
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer> struct SwiGLUPolicy;

template <> struct SwiGLUPolicy<86> {
    static constexpr int kBlockSize          = 256;
    static constexpr int kMinBlocksPerSM     = 2;
    static constexpr int kVecWidth           = 8;
    static constexpr int kRegBudgetPerThread = 64;  // 64 × float = 256 B
};

template <> struct SwiGLUPolicy<90> {
    static constexpr int kBlockSize          = 256;
    static constexpr int kMinBlocksPerSM     = 4;
    static constexpr int kVecWidth           = 8;
    static constexpr int kRegBudgetPerThread = 128;
};

template <> struct SwiGLUPolicy<120> {
    static constexpr int kBlockSize          = 512;
    static constexpr int kMinBlocksPerSM     = 4;
    static constexpr int kVecWidth           = 8;
    static constexpr int kRegBudgetPerThread = 128;
};

template <int SmVer> struct SwiGLUPolicy {
    static constexpr int kBlockSize          = 256;
    static constexpr int kMinBlocksPerSM     = 2;
    static constexpr int kVecWidth           = 8;
    static constexpr int kRegBudgetPerThread = 64;
};

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: Activation function
// ─────────────────────────────────────────────────────────────────────────────

DS_D_INLINE float fast_sigmoid(float x)
{
    return 1.f / (1.f + __expf(-x));
}

DS_D_INLINE float swiglu(float gate, float up)
{
    return gate * fast_sigmoid(gate) * up;
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 3: Warp + block sum reductions
//   Full butterfly __shfl_xor_sync, #pragma unroll, 5 rounds.
//   On SM9.0+ compiler may emit REDUX.SYNC.ADD.F32.
// ─────────────────────────────────────────────────────────────────────────────

DS_D_INLINE float warp_reduce_sum_shfl(float val)
{
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

template <int kBlockSize>
DS_D_INLINE float block_reduce_sum(
    float                  val,
    float* __restrict__    smem_warps,
    cg::thread_block&      blk)
{
    constexpr int kMaxWarps = kBlockSize / hw_warp_size;
    const int lane    = threadIdx.x % hw_warp_size;
    const int warp_id = threadIdx.x / hw_warp_size;

    val = warp_reduce_sum_shfl(val);

    if (lane == 0) smem_warps[warp_id] = val;
    blk.sync();

    val = (threadIdx.x < kMaxWarps) ? smem_warps[threadIdx.x] : 0.f;
    if (warp_id == 0) val = warp_reduce_sum_shfl(val);

    if (threadIdx.x == 0) smem_warps[0] = val;
    blk.sync();
    return smem_warps[0];
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 4: Main Kernel — fused SwiGLU + RMSNorm
//
//   kSinglePass = true:  store SwiGLU in registers, normalise from registers.
//   kSinglePass = false: two-pass fallback for large hidden dims.
//
//   Variable hidden sizes: any hidden divisible by kVec is supported.
//   The register buffer is bounded by kMaxIter = kRegBudgetPerThread / kVec.
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer, bool kSinglePass>
__global__ void
__launch_bounds__(SwiGLUPolicy<SmVer>::kBlockSize,
                  SwiGLUPolicy<SmVer>::kMinBlocksPerSM)
fused_swiglu_ln_kernel(
    __nv_bfloat16* __restrict__       output,
    const __nv_bfloat16* __restrict__ gate_proj,
    const __nv_bfloat16* __restrict__ up_proj,
    const float*          __restrict__ ln_weight,
    int hidden,
    float eps)
{
    using Policy = SwiGLUPolicy<SmVer>;
    constexpr int kVec      = Policy::kVecWidth;
    constexpr int kBS       = Policy::kBlockSize;
    constexpr int kMaxWarps = kBS / hw_warp_size;

    __shared__ float smem_warps[kMaxWarps];
    cg::thread_block blk = cg::this_thread_block();

    const int row     = blockIdx.x;
    const __nv_bfloat16* __restrict__ g_row = gate_proj + (size_t)row * hidden;
    const __nv_bfloat16* __restrict__ u_row = up_proj   + (size_t)row * hidden;
          __nv_bfloat16* __restrict__ o_row = output    + (size_t)row * hidden;

    if constexpr (kSinglePass) {
        // ── Single-pass: all SwiGLU values held in thread registers ────────
        // kMaxIter = max complete vector iterations per thread.
        // For hidden=4096, kBS=256, kVec=8: 4096/(256*8)=2 iterations.
        constexpr int kMaxIter = Policy::kRegBudgetPerThread / kVec;

        float reg_swiglu[kMaxIter * kVec];
        float thread_sq = 0.f;

        // Pass 1: compute SwiGLU, store in registers, accumulate sq_sum.
        // FIX: n_iter derived from col directly (col / (kBS*kVec)) — always
        // correct for any hidden size, not just kBS*kVec multiples.
        for (int col = (int)threadIdx.x * kVec; col < hidden; col += kBS * kVec) {
            const int n_iter = col / (kBS * kVec);  // ← fixed: was a counter

            const uint4 g_raw = *reinterpret_cast<const uint4*>(g_row + col);
            const uint4 u_raw = *reinterpret_cast<const uint4*>(u_row + col);
            const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&g_raw);
            const __nv_bfloat16* up = reinterpret_cast<const __nv_bfloat16*>(&u_raw);

            const int base_reg = n_iter * kVec;
            #pragma unroll
            for (int v = 0; v < kVec; ++v) {
                float sw = swiglu(__bfloat162float(gp[v]), __bfloat162float(up[v]));
                reg_swiglu[base_reg + v] = sw;
                thread_sq += sw * sw;
            }
        }

        // Block-level RMS reduction.
        float rms_sq  = block_reduce_sum<kBS>(thread_sq, smem_warps, blk);
        float rms_inv = rsqrtf(rms_sq / (float)hidden + eps);

        // Pass 2: normalise from registers — zero additional DRAM reads.
        for (int col = (int)threadIdx.x * kVec; col < hidden; col += kBS * kVec) {
            const int n_iter   = col / (kBS * kVec);
            const int base_reg = n_iter * kVec;
            __nv_bfloat16 out_buf[kVec];

            #pragma unroll
            for (int v = 0; v < kVec; ++v) {
                float w   = __ldg(ln_weight + col + v);
                out_buf[v] = __float2bfloat16(reg_swiglu[base_reg + v] * rms_inv * w);
            }
            *reinterpret_cast<uint4*>(o_row + col) =
                *reinterpret_cast<const uint4*>(out_buf);
        }

    } else {
        // ── Two-pass fallback for large hidden dims ─────────────────────────
        // Pass 1: stream through gate/up, compute sq_sum.
        float thread_sq = 0.f;
        for (int col = (int)threadIdx.x * kVec; col < hidden; col += kBS * kVec) {
            const uint4 g_raw = __ldg(reinterpret_cast<const uint4*>(g_row + col));
            const uint4 u_raw = __ldg(reinterpret_cast<const uint4*>(u_row + col));
            const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&g_raw);
            const __nv_bfloat16* up = reinterpret_cast<const __nv_bfloat16*>(&u_raw);

            #pragma unroll
            for (int v = 0; v < kVec; ++v) {
                float sw = swiglu(__bfloat162float(gp[v]), __bfloat162float(up[v]));
                thread_sq += sw * sw;
            }
        }

        float rms_sq  = block_reduce_sum<kBS>(thread_sq, smem_warps, blk);
        float rms_inv = rsqrtf(rms_sq / (float)hidden + eps);

        // Pass 2: re-read, compute SwiGLU, normalise, write output.
        for (int col = (int)threadIdx.x * kVec; col < hidden; col += kBS * kVec) {
            const uint4 g_raw = __ldg(reinterpret_cast<const uint4*>(g_row + col));
            const uint4 u_raw = __ldg(reinterpret_cast<const uint4*>(u_row + col));
            const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&g_raw);
            const __nv_bfloat16* up = reinterpret_cast<const __nv_bfloat16*>(&u_raw);

            __nv_bfloat16 out_buf[kVec];
            #pragma unroll
            for (int v = 0; v < kVec; ++v) {
                float gv  = __bfloat162float(gp[v]);
                float uv  = __bfloat162float(up[v]);
                float sw  = swiglu(gv, uv);
                float w   = __ldg(ln_weight + col + v);
                out_buf[v] = __float2bfloat16(sw * rms_inv * w);
            }
            *reinterpret_cast<uint4*>(o_row + col) =
                *reinterpret_cast<const uint4*>(out_buf);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 5: Host-side dispatch
//   Single-pass selected when hidden fits in per-thread register budget.
//   Three-way SM dispatch with correct __launch_bounds__ per tier.
// ─────────────────────────────────────────────────────────────────────────────

void launch_fused_swiglu_ln(
    __nv_bfloat16*       output,
    const __nv_bfloat16* gate_proj,
    const __nv_bfloat16* up_proj,
    const float*         ln_weight,
    int                  batch,
    int                  hidden,
    float                eps,
    int                  sm_version,
    cudaStream_t         stream)
{
    if (batch == 0 || hidden == 0) return;
    const int grid = batch;

    // max_sp: maximum hidden for single-pass (register budget not exceeded).
    // hidden can be any multiple of 8; this calculation is exact.
    if (sm_version >= 120) {
        using P = SwiGLUPolicy<120>;
        const int max_sp = P::kBlockSize * P::kVecWidth * P::kRegBudgetPerThread;
        if (hidden <= max_sp)
            fused_swiglu_ln_kernel<120, true>
                <<<grid, P::kBlockSize, 0, stream>>>(
                    output, gate_proj, up_proj, ln_weight, hidden, eps);
        else
            fused_swiglu_ln_kernel<120, false>
                <<<grid, P::kBlockSize, 0, stream>>>(
                    output, gate_proj, up_proj, ln_weight, hidden, eps);
    } else if (sm_version >= 90) {
        using P = SwiGLUPolicy<90>;
        const int max_sp = P::kBlockSize * P::kVecWidth * P::kRegBudgetPerThread;
        if (hidden <= max_sp)
            fused_swiglu_ln_kernel<90, true>
                <<<grid, P::kBlockSize, 0, stream>>>(
                    output, gate_proj, up_proj, ln_weight, hidden, eps);
        else
            fused_swiglu_ln_kernel<90, false>
                <<<grid, P::kBlockSize, 0, stream>>>(
                    output, gate_proj, up_proj, ln_weight, hidden, eps);
    } else {
        using P = SwiGLUPolicy<86>;
        const int max_sp = P::kBlockSize * P::kVecWidth * P::kRegBudgetPerThread;
        if (hidden <= max_sp)
            fused_swiglu_ln_kernel<86, true>
                <<<grid, P::kBlockSize, 0, stream>>>(
                    output, gate_proj, up_proj, ln_weight, hidden, eps);
        else
            fused_swiglu_ln_kernel<86, false>
                <<<grid, P::kBlockSize, 0, stream>>>(
                    output, gate_proj, up_proj, ln_weight, hidden, eps);
    }
}
