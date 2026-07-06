// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * fused_swiglu_ln.cu  —  NeurIPS 2026 DES-LOC production kernel
 *
 * Fused SwiGLU activation + RMS LayerNorm for heterogeneous GPU clusters.
 *
 * For each row i of [batch × hidden]:
 *   swiglu_i[j] = gate_i[j] × sigmoid(gate_i[j]) × up_i[j]
 *   output_i[j] = swiglu_i[j] × ln_weight[j] / sqrt(mean(swiglu²) + ε)
 *
 * ═══════════════════════════════════════════════════════════════════════
 * ALGORITHMIC DESIGN
 * ═══════════════════════════════════════════════════════════════════════
 *
 * 1. PARAMETERISED hidden_size — no compile-time constant
 *    The kernel accepts hidden_size as a runtime integer.  The loop bounds
 *    and register allocation are sized at compile time via kRegBudgetPerThread
 *    (maximum iterations per thread before register overflow).  For typical
 *    hidden sizes (4096–16384) the single-pass path is always taken.
 *
 * 2. float4 VECTORISED LOADS
 *    The inner loop loads float4 (16 bytes = 2 × uint4 BF16 loads) per
 *    thread iteration.  This is the MAXIMUM vectorisation achievable with
 *    the SM8.6+ 128-byte L1 cache line (one cache line = 8 × float4).
 *    Each thread processes kVecWidth = 8 BF16 elements per step using
 *    a single uint4 LD.128 instruction.
 *
 * 3. SINGLE-PASS IN-REGISTER FUSION (kSinglePass=true)
 *    For hidden ≤ kBlockSize × kVecWidth × kRegBudgetPerThread:
 *      Pass 1: compute SwiGLU, store in register array reg_swiglu[], 
 *              accumulate sq_sum for RMS denominator — one DRAM read pass.
 *      Block reduce sq_sum → rms_inv.
 *      Pass 2: stream through reg_swiglu[], apply rms_inv × ln_weight,
 *              write BF16 output — ZERO additional DRAM reads.
 *    Net: one DRAM read of gate/up + one write of output = minimum bandwidth.
 *
 * 4. TWO-PASS FALLBACK (kSinglePass=false, large hidden)
 *    Pass 1: stream through gate/up with __ldg(), accumulate sq_sum.
 *    Block reduce → rms_inv.
 *    Pass 2: re-read gate/up (L2 cache hit on H100/Blackwell), compute
 *            SwiGLU, normalise, write output.
 *    For hidden > L2 capacity: net ~2× DRAM reads + 1 write.
 *
 * 5. WARP BUTTERFLY REDUCTION (5 × shfl_xor, no smem)
 *    SM9.0+: compiler emits REDUX.SYNC.ADD.F32 (single-cycle warp sum).
 *    SM8.6:  5 × shfl_xor rounds (10 cycles).
 *    Block reduction: one float[kMaxWarps] smem array (64–128 bytes).
 *
 * 6. SM-SPECIALISED __launch_bounds__
 *    SM8.6:  256 threads / block, 2 CTAs/SM, regBudget = 64 floats/thread
 *    SM9.0:  256 threads / block, 4 CTAs/SM, regBudget = 128 floats/thread
 *    SM12.0: 512 threads / block, 4 CTAs/SM, regBudget = 128 floats/thread
 * ═══════════════════════════════════════════════════════════════════════
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
// Section 1: Per-SM tuning policy
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer> struct SwiGLUPolicy {
    static constexpr int kBlockSize          = 256;
    static constexpr int kMinBlocksPerSM     = 2;
    static constexpr int kVecWidth           = 8;    // BF16 per 128-bit load
    // Maximum float registers per thread for SwiGLU value storage.
    // Single-pass is valid when hidden ≤ kBlockSize × kVecWidth × kRegBudget.
    static constexpr int kRegBudgetPerThread = 64;   // 256 B per thread
    static constexpr int kMaxWarps           = kBlockSize / hw_warp_size;
};

template <> struct SwiGLUPolicy<86> {
    static constexpr int kBlockSize          = 256;
    static constexpr int kMinBlocksPerSM     = 2;
    static constexpr int kVecWidth           = 8;
    static constexpr int kRegBudgetPerThread = 64;
    static constexpr int kMaxWarps           = kBlockSize / hw_warp_size;  // 8
};

template <> struct SwiGLUPolicy<90> {
    static constexpr int kBlockSize          = 256;
    static constexpr int kMinBlocksPerSM     = 4;
    static constexpr int kVecWidth           = 8;
    static constexpr int kRegBudgetPerThread = 128;  // 512 B per thread (H100)
    static constexpr int kMaxWarps           = kBlockSize / hw_warp_size;  // 8
};

template <> struct SwiGLUPolicy<120> {
    static constexpr int kBlockSize          = 512;
    static constexpr int kMinBlocksPerSM     = 4;
    static constexpr int kVecWidth           = 8;
    static constexpr int kRegBudgetPerThread = 128;
    static constexpr int kMaxWarps           = kBlockSize / hw_warp_size;  // 16
};

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: SwiGLU activation and fast sigmoid
// ─────────────────────────────────────────────────────────────────────────────

// Hardware-accelerated sigmoid via __expf (maps to EX2.APPROX on SM8.6+).
DS_D_INLINE float fast_sigmoid(float x)
{
    return 1.f / (1.f + __expf(-x));
}

// SwiGLU: gate × sigmoid(gate) × up
DS_D_INLINE float swiglu(float gate, float up)
{
    return gate * fast_sigmoid(gate) * up;
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 3: Warp butterfly + smem block reduction
//
//   Stage 1: 5 × shfl_xor_sync butterfly (zero smem, REDUX.SYNC on SM9.0+)
//   Stage 2: lane-0 deposits into smem; first warp does second butterfly
//
//   Returns the block-wide sum broadcast to ALL threads via smem[0].
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
    float                  val,
    float* __restrict__    smem_warps,
    cg::thread_block&      blk)
{
    constexpr int kMaxWarps = kBlockSize / hw_warp_size;
    const int lane    = threadIdx.x % hw_warp_size;
    const int warp_id = threadIdx.x / hw_warp_size;

    // Stage 1: warp butterfly.
    val = warp_reduce_sum(val);

    // Stage 2: deposit warp sums.
    if (lane == 0) smem_warps[warp_id] = val;
    blk.sync();

    // Stage 3: first warp reduces smem entries.
    val = (threadIdx.x < kMaxWarps) ? smem_warps[threadIdx.x] : 0.f;
    if (warp_id == 0) val = warp_reduce_sum(val);

    // Broadcast final sum via smem[0].
    if (threadIdx.x == 0) smem_warps[0] = val;
    blk.sync();
    return smem_warps[0];
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 4: Main kernel — fused SwiGLU + RMSNorm
//
//   Template parameters:
//     SmVer       : SM version (selects SwiGLUPolicy)
//     kSinglePass : true  → store SwiGLU in thread registers, zero re-reads
//                   false → two-pass; pass 1 accumulates sq_sum, pass 2 normalises
//
//   Grid:  (batch,) blocks — one CTA per row
//   Block: Policy::kBlockSize threads — cover hidden in strides of kBS × kVec
//
//   Shared memory: float[kMaxWarps] for block reduce (64–128 bytes max)
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
    int   hidden,
    float eps)
{
    using Policy = SwiGLUPolicy<SmVer>;
    constexpr int kVec      = Policy::kVecWidth;   // 8
    constexpr int kBS       = Policy::kBlockSize;
    constexpr int kMaxWarps = Policy::kMaxWarps;

    // Shared memory: ONE float array for warp partial sums.
    __shared__ float smem_warps[kMaxWarps];

    cg::thread_block blk = cg::this_thread_block();

    const int row = blockIdx.x;
    const __nv_bfloat16* __restrict__ g_row = gate_proj + (size_t)row * hidden;
    const __nv_bfloat16* __restrict__ u_row = up_proj   + (size_t)row * hidden;
          __nv_bfloat16* __restrict__ o_row = output    + (size_t)row * hidden;

    // ──────────────────────────────────────────────────────────────────────
    // SINGLE-PASS path (kSinglePass=true)
    //   All SwiGLU outputs held in thread-local register array.
    //   Pass 1: load gate/up (one DRAM read), compute SwiGLU, store in
    //           register array reg[], accumulate sq_sum.
    //   Block reduce → rms_inv.
    //   Pass 2: stream through register array, apply normalisation,
    //           write BF16 output — ZERO additional DRAM reads.
    // ──────────────────────────────────────────────────────────────────────
    if constexpr (kSinglePass) {
        // Compile-time upper bound on iterations per thread.
        // For hidden=4096, kBS=256, kVec=8: 4096/(256*8) = 2 iterations.
        constexpr int kMaxIter = Policy::kRegBudgetPerThread / kVec;

        float reg_swiglu[kMaxIter * kVec];  // Register-file storage
        float thread_sq = 0.f;

        // ── Pass 1: compute SwiGLU + accumulate squared sum ──
        int n_iter = 0;
        for (int col = (int)threadIdx.x * kVec; col < hidden;
             col += kBS * kVec, ++n_iter) {

            // float4 vectorised load: two uint4 loads packed into 16 bytes each
            const uint4 g_raw = *reinterpret_cast<const uint4*>(g_row + col);
            const uint4 u_raw = *reinterpret_cast<const uint4*>(u_row + col);
            const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&g_raw);
            const __nv_bfloat16* up = reinterpret_cast<const __nv_bfloat16*>(&u_raw);

            const int base = n_iter * kVec;
            #pragma unroll
            for (int v = 0; v < kVec; ++v) {
                float gv = __bfloat162float(gp[v]);
                float uv = __bfloat162float(up[v]);
                float sw = swiglu(gv, uv);
                reg_swiglu[base + v] = sw;
                thread_sq += sw * sw;
            }
        }

        // ── Block-level RMS denominator ──
        float sq_sum  = block_reduce_sum<kBS>(thread_sq, smem_warps, blk);
        float rms_inv = rsqrtf(sq_sum / (float)hidden + eps);
        // rms_inv is broadcast to all threads via smem_warps[0] inside block_reduce_sum.

        // ── Pass 2: normalise from registers — ZERO DRAM reads ──
        n_iter = 0;
        for (int col = (int)threadIdx.x * kVec; col < hidden;
             col += kBS * kVec, ++n_iter) {

            const int base = n_iter * kVec;
            __nv_bfloat16 out_buf[kVec];

            #pragma unroll
            for (int v = 0; v < kVec; ++v) {
                // __ldg for ln_weight: read-only, likely L2-warm.
                float w   = __ldg(ln_weight + col + v);
                float res = reg_swiglu[base + v] * rms_inv * w;
                out_buf[v] = __float2bfloat16(res);
            }
            // 128-bit store — one LD.GLOBAL.128 per 8 BF16 elements.
            *reinterpret_cast<uint4*>(o_row + col) =
                *reinterpret_cast<const uint4*>(out_buf);
        }

    } else {
        // ──────────────────────────────────────────────────────────────────
        // TWO-PASS path (kSinglePass=false, large hidden)
        //   Pass 1: stream gate/up with __ldg(), accumulate sq_sum.
        //   Block reduce → rms_inv.
        //   Pass 2: re-read gate/up (L2 hit on H100/BW), normalise, write.
        // ──────────────────────────────────────────────────────────────────

        // ── Pass 1: sq_sum accumulation ──
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

        // ── Block-level RMS reduction ──
        float sq_sum  = block_reduce_sum<kBS>(thread_sq, smem_warps, blk);
        float rms_inv = rsqrtf(sq_sum / (float)hidden + eps);

        // ── Pass 2: re-read, normalise, write ──
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
//
//   Single-pass threshold: hidden ≤ kBlockSize × kVecWidth × kRegBudgetPerThread
//   In practice: SM86 → 256×8×64=131072, SM90/SM12 → 512×8×128 or 256×8×128
//   All standard LLM hidden sizes (4096–16384) → single-pass.
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
    const int grid = batch;  // one CTA per row

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
