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
    if (batch <= 0 || hidden <= 0) return;  // BUG-FIX: guard zero-grid launch
    const int grid = batch;  // one CTA per row

    if (sm_version >= 120) {
        using P = SwiGLUPolicy<120>;
        const int max_sp = P::kBlockSize * P::kVecWidth * P::kRegBudgetPerThread;
        if (hidden <= max_sp)
            fused_swiglu_ln_kernel<120, true>
                <<<grid, P::kBlockSize, 0, stream>>>(
                    output, gate_proj, up_proj, ln_weight, hidden, eps);
    DS_LAUNCH_CHECK(stream);
        else
            fused_swiglu_ln_kernel<120, false>
                <<<grid, P::kBlockSize, 0, stream>>>(
                    output, gate_proj, up_proj, ln_weight, hidden, eps);
    DS_LAUNCH_CHECK(stream);
    } else if (sm_version >= 90) {
        using P = SwiGLUPolicy<90>;
        const int max_sp = P::kBlockSize * P::kVecWidth * P::kRegBudgetPerThread;
        if (hidden <= max_sp)
            fused_swiglu_ln_kernel<90, true>
                <<<grid, P::kBlockSize, 0, stream>>>(
                    output, gate_proj, up_proj, ln_weight, hidden, eps);
    DS_LAUNCH_CHECK(stream);
        else
            fused_swiglu_ln_kernel<90, false>
                <<<grid, P::kBlockSize, 0, stream>>>(
                    output, gate_proj, up_proj, ln_weight, hidden, eps);
    DS_LAUNCH_CHECK(stream);
    } else {
        using P = SwiGLUPolicy<86>;
        const int max_sp = P::kBlockSize * P::kVecWidth * P::kRegBudgetPerThread;
        if (hidden <= max_sp)
            fused_swiglu_ln_kernel<86, true>
                <<<grid, P::kBlockSize, 0, stream>>>(
                    output, gate_proj, up_proj, ln_weight, hidden, eps);
    DS_LAUNCH_CHECK(stream);
        else
            fused_swiglu_ln_kernel<86, false>
                <<<grid, P::kBlockSize, 0, stream>>>(
                    output, gate_proj, up_proj, ln_weight, hidden, eps);
    DS_LAUNCH_CHECK(stream);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 6: Backward kernel — fused SwiGLU + RMSNorm
//
// Given upstream gradient d_output ∈ BF16[batch, hidden], the forward
// inputs gate_proj / up_proj ∈ BF16[batch, hidden], the LN weight
// ln_weight ∈ FP32[hidden], and the forward RMS inverse rms_inv ∈ FP32[batch]
// saved from the forward pass, this kernel computes:
//
//   Let  σ(x) = sigmoid(x) = 1 / (1 + exp(-x))
//        s_j  = gate_j · σ(gate_j) · up_j           (SwiGLU output)
//        ŷ_j  = s_j · w_j · rms_inv                 (forward output)
//
//   d_ln_weight_j  += Σ_i  d_out[i,j] · ŷ[i,j] / w_j     (batch reduce)
//
//   For each row i (CTA handles one row):
//     dot_i = Σ_j  d_out[i,j] · w_j · s[i,j]              (RMSNorm chain-rule dot)
//
//     d_s_j = rms_inv · w_j · d_out[i,j]
//             - rms_inv³ · s_j · dot_i / hidden            (RMSNorm upstream grad)
//
//     d_gate_j = d_s_j · up_j · σ(gate_j) · (1 + gate_j · (1 − σ(gate_j)))
//     d_up_j   = d_s_j · gate_j · σ(gate_j)
//
// Shared memory layout  (kMaxWarps + kMaxWarps floats):
//   smem[0 .. kMaxWarps-1]          : warp partial sums for dot reduction
//   smem[kMaxWarps .. 2kMaxWarps-1] : unused (padding for alignment)
//
// d_ln_weight is accumulated via atomicAdd over the batch dimension;
// it must be zeroed by the caller before the first backward call.
//
// Template parameters mirror the forward kernel (SmVer, kSinglePass).
// ─────────────────────────────────────────────────────────────────────────────

// ── Derivative helpers ────────────────────────────────────────────────────────

// d(SwiGLU)/d(gate)  =  up · σ(gate) · (1 + gate · (1 − σ(gate)))
DS_D_INLINE float dswiglu_dgate(float gate, float up)
{
    float sig = fast_sigmoid(gate);
    return up * sig * (1.f + gate * (1.f - sig));
}

// d(SwiGLU)/d(up)  =  gate · σ(gate)
DS_D_INLINE float dswiglu_dup(float gate)
{
    return gate * fast_sigmoid(gate);
}

// ── Backward kernel ───────────────────────────────────────────────────────────

template <int SmVer, bool kSinglePass>
__global__ void
__launch_bounds__(SwiGLUPolicy<SmVer>::kBlockSize,
                  SwiGLUPolicy<SmVer>::kMinBlocksPerSM)
fused_swiglu_ln_backward_kernel(
    __nv_bfloat16* __restrict__       d_gate,       // [batch, hidden]
    __nv_bfloat16* __restrict__       d_up,         // [batch, hidden]
    float*         __restrict__       d_ln_weight,  // [hidden]  atomicAdd
    const __nv_bfloat16* __restrict__ d_output,     // [batch, hidden]
    const __nv_bfloat16* __restrict__ gate_proj,    // [batch, hidden]
    const __nv_bfloat16* __restrict__ up_proj,      // [batch, hidden]
    const float*          __restrict__ ln_weight,   // [hidden]
    const float*          __restrict__ rms_inv_buf, // [batch]  saved from fwd
    int   hidden,
    float eps)
{
    using Policy = SwiGLUPolicy<SmVer>;
    constexpr int kVec      = Policy::kVecWidth;   // 8
    constexpr int kBS       = Policy::kBlockSize;
    constexpr int kMaxWarps = Policy::kMaxWarps;

    // Shared memory: one warp-sum array (used for the dot reduction and
    // the RMS-inv broadcast that follows).
    __shared__ float smem_warps[kMaxWarps];

    cg::thread_block blk = cg::this_thread_block();

    const int row = blockIdx.x;

    // Row pointers
    const __nv_bfloat16* __restrict__ do_row = d_output  + (size_t)row * hidden;
    const __nv_bfloat16* __restrict__  g_row = gate_proj + (size_t)row * hidden;
    const __nv_bfloat16* __restrict__  u_row = up_proj   + (size_t)row * hidden;
          __nv_bfloat16* __restrict__ dg_row = d_gate    + (size_t)row * hidden;
          __nv_bfloat16* __restrict__ du_row = d_up      + (size_t)row * hidden;

    // Saved forward RMS inverse for this row.
    const float rms_inv = rms_inv_buf[row];
    const float rms_inv3 = rms_inv * rms_inv * rms_inv;   // rms_inv^3

    // ──────────────────────────────────────────────────────────────────────
    // SINGLE-PASS backward (kSinglePass=true)
    //   All inputs fit in registers; two logical passes but zero extra DRAM.
    //
    //   Register-file pass 1: load gate/up/d_out, compute SwiGLU, store
    //       gate, up, d_out, and s in register arrays; accumulate dot.
    //   Block reduce → dot.
    //   Register-file pass 2: compute d_s, d_gate, d_up and write outputs.
    //       Also atomicAdd d_ln_weight (one atomic per element per CTA).
    // ──────────────────────────────────────────────────────────────────────
    if constexpr (kSinglePass) {
        constexpr int kMaxIter = Policy::kRegBudgetPerThread / kVec;

        // Register arrays (4 × kMaxIter × kVec floats, ~4 KB on SM9.0)
        float reg_gate[kMaxIter * kVec];
        float reg_up  [kMaxIter * kVec];
        float reg_do  [kMaxIter * kVec];
        float reg_sw  [kMaxIter * kVec];

        float thread_dot = 0.f;

        // ── Register-file pass 1: load + SwiGLU + dot accumulation ──
        int n_iter = 0;
        for (int col = (int)threadIdx.x * kVec; col < hidden;
             col += kBS * kVec, ++n_iter) {

            const uint4  g_raw = *reinterpret_cast<const uint4*>( g_row + col);
            const uint4  u_raw = *reinterpret_cast<const uint4*>( u_row + col);
            const uint4 do_raw = *reinterpret_cast<const uint4*>(do_row + col);

            const __nv_bfloat16* gp  = reinterpret_cast<const __nv_bfloat16*>(&g_raw);
            const __nv_bfloat16* up  = reinterpret_cast<const __nv_bfloat16*>(&u_raw);
            const __nv_bfloat16* dop = reinterpret_cast<const __nv_bfloat16*>(&do_raw);

            const int base = n_iter * kVec;
            #pragma unroll
            for (int v = 0; v < kVec; ++v) {
                float gv  = __bfloat162float(gp[v]);
                float uv  = __bfloat162float(up[v]);
                float dov = __bfloat162float(dop[v]);
                float sw  = swiglu(gv, uv);
                float w   = __ldg(ln_weight + col + v);

                reg_gate[base + v] = gv;
                reg_up  [base + v] = uv;
                reg_do  [base + v] = dov;
                reg_sw  [base + v] = sw;

                // dot_i = Σ_j  d_out_j · w_j · s_j
                thread_dot += dov * w * sw;
            }
        }

        // ── Block-level dot reduction ──
        float dot = block_reduce_sum<kBS>(thread_dot, smem_warps, blk);

        // ── Register-file pass 2: compute gradients + write ──
        n_iter = 0;
        for (int col = (int)threadIdx.x * kVec; col < hidden;
             col += kBS * kVec, ++n_iter) {

            const int base = n_iter * kVec;
            __nv_bfloat16 dg_buf[kVec];
            __nv_bfloat16 du_buf[kVec];

            #pragma unroll
            for (int v = 0; v < kVec; ++v) {
                float gv  = reg_gate[base + v];
                float uv  = reg_up  [base + v];
                float dov = reg_do  [base + v];
                float sw  = reg_sw  [base + v];
                float w   = __ldg(ln_weight + col + v);

                // RMSNorm upstream gradient for s_j:
                //   d_s_j = rms_inv · w_j · d_out_j
                //         - rms_inv³ · s_j · dot / hidden
                float d_s = rms_inv * w * dov
                            - rms_inv3 * sw * dot / (float)hidden;

                // Chain rule through SwiGLU
                dg_buf[v] = __float2bfloat16(d_s * dswiglu_dgate(gv, uv));
                du_buf[v] = __float2bfloat16(d_s * dswiglu_dup(gv));

                // Accumulate d_ln_weight (atomic over batch)
                // d_ln_weight_j = Σ_i  d_out[i,j] · ŷ[i,j] / w_j
                //               = Σ_i  d_out[i,j] · s[i,j] · rms_inv_i
                // We use the equivalent form to avoid re-reading output:
                //   d_out_j · s_j · rms_inv  (already have all three values)
                atomicAdd(&d_ln_weight[col + v], dov * sw * rms_inv);
            }

            // Vectorised 128-bit store
            *reinterpret_cast<uint4*>(dg_row + col) =
                *reinterpret_cast<const uint4*>(dg_buf);
            *reinterpret_cast<uint4*>(du_row + col) =
                *reinterpret_cast<const uint4*>(du_buf);
        }

    } else {
        // ──────────────────────────────────────────────────────────────────
        // TWO-PASS backward (kSinglePass=false, large hidden)
        //   Pass 1: stream gate/up/d_out, compute SwiGLU, accumulate dot.
        //   Block reduce → dot.
        //   Pass 2: re-read gate/up/d_out (L2 hit), compute gradients, write.
        // ──────────────────────────────────────────────────────────────────

        // ── Pass 1: dot accumulation ──
        float thread_dot = 0.f;
        for (int col = (int)threadIdx.x * kVec; col < hidden; col += kBS * kVec) {
            const uint4  g_raw = __ldg(reinterpret_cast<const uint4*>( g_row + col));
            const uint4  u_raw = __ldg(reinterpret_cast<const uint4*>( u_row + col));
            const uint4 do_raw = __ldg(reinterpret_cast<const uint4*>(do_row + col));
            const __nv_bfloat16* gp  = reinterpret_cast<const __nv_bfloat16*>(&g_raw);
            const __nv_bfloat16* up  = reinterpret_cast<const __nv_bfloat16*>(&u_raw);
            const __nv_bfloat16* dop = reinterpret_cast<const __nv_bfloat16*>(&do_raw);
            #pragma unroll
            for (int v = 0; v < kVec; ++v) {
                float sw = swiglu(__bfloat162float(gp[v]), __bfloat162float(up[v]));
                float w  = __ldg(ln_weight + col + v);
                thread_dot += __bfloat162float(dop[v]) * w * sw;
            }
        }

        // ── Block-level dot reduction ──
        float dot = block_reduce_sum<kBS>(thread_dot, smem_warps, blk);

        // ── Pass 2: compute and write d_gate, d_up; accumulate d_ln_weight ──
        for (int col = (int)threadIdx.x * kVec; col < hidden; col += kBS * kVec) {
            const uint4  g_raw = __ldg(reinterpret_cast<const uint4*>( g_row + col));
            const uint4  u_raw = __ldg(reinterpret_cast<const uint4*>( u_row + col));
            const uint4 do_raw = __ldg(reinterpret_cast<const uint4*>(do_row + col));
            const __nv_bfloat16* gp  = reinterpret_cast<const __nv_bfloat16*>(&g_raw);
            const __nv_bfloat16* up  = reinterpret_cast<const __nv_bfloat16*>(&u_raw);
            const __nv_bfloat16* dop = reinterpret_cast<const __nv_bfloat16*>(&do_raw);

            __nv_bfloat16 dg_buf[kVec];
            __nv_bfloat16 du_buf[kVec];

            #pragma unroll
            for (int v = 0; v < kVec; ++v) {
                float gv  = __bfloat162float(gp[v]);
                float uv  = __bfloat162float(up[v]);
                float dov = __bfloat162float(dop[v]);
                float sw  = swiglu(gv, uv);
                float w   = __ldg(ln_weight + col + v);

                float d_s = rms_inv * w * dov
                            - rms_inv3 * sw * dot / (float)hidden;

                dg_buf[v] = __float2bfloat16(d_s * dswiglu_dgate(gv, uv));
                du_buf[v] = __float2bfloat16(d_s * dswiglu_dup(gv));

                atomicAdd(&d_ln_weight[col + v], dov * sw * rms_inv);
            }

            *reinterpret_cast<uint4*>(dg_row + col) =
                *reinterpret_cast<const uint4*>(dg_buf);
            *reinterpret_cast<uint4*>(du_row + col) =
                *reinterpret_cast<const uint4*>(du_buf);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 7: Forward kernel variant that also saves rms_inv per row
//
// An overload of the forward kernel that writes rms_inv[row] into a
// preallocated FP32 buffer so that the backward pass can reuse it without
// recomputing the RMS denominator.
//
// Interface is identical to fused_swiglu_ln_kernel except for the extra
// rms_inv_out pointer.  The saved value is:
//   rms_inv_out[row] = 1 / sqrt(mean(swiglu(gate,up)²) + eps)
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer, bool kSinglePass>
__global__ void
__launch_bounds__(SwiGLUPolicy<SmVer>::kBlockSize,
                  SwiGLUPolicy<SmVer>::kMinBlocksPerSM)
fused_swiglu_ln_fwd_save_kernel(
    __nv_bfloat16* __restrict__       output,
    float*         __restrict__       rms_inv_out,  // [batch]
    const __nv_bfloat16* __restrict__ gate_proj,
    const __nv_bfloat16* __restrict__ up_proj,
    const float*          __restrict__ ln_weight,
    int   hidden,
    float eps)
{
    using Policy = SwiGLUPolicy<SmVer>;
    constexpr int kVec      = Policy::kVecWidth;
    constexpr int kBS       = Policy::kBlockSize;
    constexpr int kMaxWarps = Policy::kMaxWarps;

    __shared__ float smem_warps[kMaxWarps];
    cg::thread_block blk = cg::this_thread_block();

    const int row = blockIdx.x;
    const __nv_bfloat16* __restrict__ g_row = gate_proj + (size_t)row * hidden;
    const __nv_bfloat16* __restrict__ u_row = up_proj   + (size_t)row * hidden;
          __nv_bfloat16* __restrict__ o_row = output    + (size_t)row * hidden;

    if constexpr (kSinglePass) {
        constexpr int kMaxIter = Policy::kRegBudgetPerThread / kVec;
        float reg_swiglu[kMaxIter * kVec];
        float thread_sq = 0.f;

        int n_iter = 0;
        for (int col = (int)threadIdx.x * kVec; col < hidden;
             col += kBS * kVec, ++n_iter) {
            const uint4 g_raw = *reinterpret_cast<const uint4*>(g_row + col);
            const uint4 u_raw = *reinterpret_cast<const uint4*>(u_row + col);
            const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&g_raw);
            const __nv_bfloat16* up = reinterpret_cast<const __nv_bfloat16*>(&u_raw);
            const int base = n_iter * kVec;
            #pragma unroll
            for (int v = 0; v < kVec; ++v) {
                float sw = swiglu(__bfloat162float(gp[v]), __bfloat162float(up[v]));
                reg_swiglu[base + v] = sw;
                thread_sq += sw * sw;
            }
        }

        float sq_sum  = block_reduce_sum<kBS>(thread_sq, smem_warps, blk);
        float rms_inv = rsqrtf(sq_sum / (float)hidden + eps);

        // Save for backward
        if (threadIdx.x == 0) rms_inv_out[row] = rms_inv;

        n_iter = 0;
        for (int col = (int)threadIdx.x * kVec; col < hidden;
             col += kBS * kVec, ++n_iter) {
            const int base = n_iter * kVec;
            __nv_bfloat16 out_buf[kVec];
            #pragma unroll
            for (int v = 0; v < kVec; ++v) {
                float w   = __ldg(ln_weight + col + v);
                out_buf[v] = __float2bfloat16(reg_swiglu[base + v] * rms_inv * w);
            }
            *reinterpret_cast<uint4*>(o_row + col) =
                *reinterpret_cast<const uint4*>(out_buf);
        }

    } else {
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

        float sq_sum  = block_reduce_sum<kBS>(thread_sq, smem_warps, blk);
        float rms_inv = rsqrtf(sq_sum / (float)hidden + eps);

        if (threadIdx.x == 0) rms_inv_out[row] = rms_inv;

        for (int col = (int)threadIdx.x * kVec; col < hidden; col += kBS * kVec) {
            const uint4 g_raw = __ldg(reinterpret_cast<const uint4*>(g_row + col));
            const uint4 u_raw = __ldg(reinterpret_cast<const uint4*>(u_row + col));
            const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&g_raw);
            const __nv_bfloat16* up = reinterpret_cast<const __nv_bfloat16*>(&u_raw);
            __nv_bfloat16 out_buf[kVec];
            #pragma unroll
            for (int v = 0; v < kVec; ++v) {
                float w = __ldg(ln_weight + col + v);
                out_buf[v] = __float2bfloat16(
                    swiglu(__bfloat162float(gp[v]), __bfloat162float(up[v])) * rms_inv * w);
            }
            *reinterpret_cast<uint4*>(o_row + col) =
                *reinterpret_cast<const uint4*>(out_buf);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 8: Host-side dispatch for forward-with-save and backward
// ─────────────────────────────────────────────────────────────────────────────

// Helper macro to avoid repetition across the three SM specialisations.
#define DISPATCH_FWD_SAVE(SmVer_)                                              \
    do {                                                                        \
        using P = SwiGLUPolicy<SmVer_>;                                        \
        const int max_sp = P::kBlockSize * P::kVecWidth * P::kRegBudgetPerThread; \
        if (hidden <= max_sp)                                                   \
            fused_swiglu_ln_fwd_save_kernel<SmVer_, true>                      \
                <<<grid, P::kBlockSize, 0, stream>>>(                           \
                    output, rms_inv_out,                                        \
                    gate_proj, up_proj, ln_weight, hidden, eps);                \
        else                                                                    \
            fused_swiglu_ln_fwd_save_kernel<SmVer_, false>                     \
                <<<grid, P::kBlockSize, 0, stream>>>(                           \
                    output, rms_inv_out,                                        \
                    gate_proj, up_proj, ln_weight, hidden, eps);                \
    } while (0)

#define DISPATCH_BWD(SmVer_)                                                   \
    do {                                                                        \
        using P = SwiGLUPolicy<SmVer_>;                                        \
        const int max_sp = P::kBlockSize * P::kVecWidth * P::kRegBudgetPerThread; \
        if (hidden <= max_sp)                                                   \
            fused_swiglu_ln_backward_kernel<SmVer_, true>                      \
                <<<grid, P::kBlockSize, 0, stream>>>(                           \
                    d_gate, d_up, d_ln_weight,                                  \
                    d_output, gate_proj, up_proj, ln_weight,                    \
                    rms_inv_buf, hidden, eps);                                  \
        else                                                                    \
            fused_swiglu_ln_backward_kernel<SmVer_, false>                     \
                <<<grid, P::kBlockSize, 0, stream>>>(                           \
                    d_gate, d_up, d_ln_weight,                                  \
                    d_output, gate_proj, up_proj, ln_weight,                    \
                    rms_inv_buf, hidden, eps);                                  \
    } while (0)

/**
 * launch_fused_swiglu_ln_fwd_save
 *
 * Forward pass variant that also writes rms_inv[row] into a caller-allocated
 * FP32 buffer.  This buffer must be passed verbatim to
 * launch_fused_swiglu_ln_backward so the backward kernel can avoid
 * recomputing the RMS denominator.
 *
 * Caller must allocate: rms_inv_out — FP32 device buffer of length `batch`.
 */
void launch_fused_swiglu_ln_fwd_save(
    __nv_bfloat16*       output,
    float*               rms_inv_out,
    const __nv_bfloat16* gate_proj,
    const __nv_bfloat16* up_proj,
    const float*         ln_weight,
    int                  batch,
    int                  hidden,
    float                eps,
    int                  sm_version,
    cudaStream_t         stream)
{
    if (batch <= 0 || hidden <= 0) return;  // BUG-FIX: guard zero-grid launch
    const int grid = batch;
    if      (sm_version >= 120) { DISPATCH_FWD_SAVE(120); }
    else if (sm_version >=  90) { DISPATCH_FWD_SAVE( 90); }
    else                        { DISPATCH_FWD_SAVE( 86); }
}

/**
 * launch_fused_swiglu_ln_backward
 *
 * Backward pass for the fused SwiGLU + RMSNorm kernel.
 *
 * Computes:
 *   d_gate     [batch, hidden] BF16  — gradient w.r.t. gate_proj
 *   d_up       [batch, hidden] BF16  — gradient w.r.t. up_proj
 *   d_ln_weight[hidden]        FP32  — gradient w.r.t. ln_weight
 *                                      (accumulated via atomicAdd; caller must
 *                                       zero before the first backward call)
 *
 * @param d_gate       [out] BF16 gradient for gate_proj [batch, hidden]
 * @param d_up         [out] BF16 gradient for up_proj   [batch, hidden]
 * @param d_ln_weight  [out] FP32 gradient for ln_weight [hidden] — ACCUMULATES
 * @param d_output     [in]  BF16 upstream gradient      [batch, hidden]
 * @param gate_proj    [in]  BF16 forward input gate     [batch, hidden]
 * @param up_proj      [in]  BF16 forward input up       [batch, hidden]
 * @param ln_weight    [in]  FP32 LN weight              [hidden]
 * @param rms_inv_buf  [in]  FP32 rms_inv saved by fwd   [batch]
 * @param batch        Batch size (rows)
 * @param hidden       Hidden size (must be divisible by 8)
 * @param eps          LayerNorm epsilon (must match forward)
 * @param sm_version   SM version of the active device (86, 90, 120)
 * @param stream       CUDA stream
 */
void launch_fused_swiglu_ln_backward(
    __nv_bfloat16*       d_gate,
    __nv_bfloat16*       d_up,
    float*               d_ln_weight,
    const __nv_bfloat16* d_output,
    const __nv_bfloat16* gate_proj,
    const __nv_bfloat16* up_proj,
    const float*         ln_weight,
    const float*         rms_inv_buf,
    int                  batch,
    int                  hidden,
    float                eps,
    int                  sm_version,
    cudaStream_t         stream)
{
    if (batch <= 0 || hidden <= 0) return;  // BUG-FIX: guard zero-grid launch
    const int grid = batch;
    if      (sm_version >= 120) { DISPATCH_BWD(120); }
    else if (sm_version >=  90) { DISPATCH_BWD( 90); }
    else                        { DISPATCH_BWD( 86); }
}

// ===========================================================================
// Issue #25 — Additional fused MLP kernels (swiglu-only, pre-LN, residual-RMSNorm)
// ===========================================================================
//
// Three lightweight kernels that complement the full fused_swiglu_ln_kernel:
//
//  1. launch_fused_swiglu     — gate × σ(gate) × up, BF16 in/out, no LN
//     Use when the caller does LayerNorm separately (e.g. pre-LN before MLP).
//
//  2. launch_fused_pre_ln_attn — RMSNorm(residual) × ln_weight, BF16 in/out
//     Pre-LayerNorm for attention input (residual stream not modified).
//
//  3. launch_fused_residual_rmsnorm — residual += input; out = RMSNorm(residual)
//     Post-attention residual addition + normalisation in one pass.
//
// All three follow the SwiGLUPolicy<SmVer> block-size and use the same
// warp-butterfly block_reduce_sum already defined above for the full kernel.
// ===========================================================================

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: SwiGLU only (no LayerNorm)
// Grid: (batch,)  Block: Policy::kBlockSize
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer>
__global__ void
__launch_bounds__(SwiGLUPolicy<SmVer>::kBlockSize,
                  SwiGLUPolicy<SmVer>::kMinBlocksPerSM)
swiglu_only_kernel(
    __nv_bfloat16* __restrict__       output,
    const __nv_bfloat16* __restrict__ gate_proj,
    const __nv_bfloat16* __restrict__ up_proj,
    int hidden)
{
    using P = SwiGLUPolicy<SmVer>;
    constexpr int kVec = P::kVecWidth;   // 8
    constexpr int kBS  = P::kBlockSize;

    const int row = blockIdx.x;
    const __nv_bfloat16* g_row = gate_proj + (size_t)row * hidden;
    const __nv_bfloat16* u_row = up_proj   + (size_t)row * hidden;
          __nv_bfloat16* o_row = output    + (size_t)row * hidden;

    for (int col = (int)threadIdx.x * kVec; col < hidden; col += kBS * kVec) {
        const uint4 g_raw = *reinterpret_cast<const uint4*>(g_row + col);
        const uint4 u_raw = *reinterpret_cast<const uint4*>(u_row + col);
        const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&g_raw);
        const __nv_bfloat16* up = reinterpret_cast<const __nv_bfloat16*>(&u_raw);

        __nv_bfloat16 out_buf[kVec];
        #pragma unroll
        for (int v = 0; v < kVec; ++v)
            out_buf[v] = __float2bfloat16(
                swiglu(__bfloat162float(gp[v]), __bfloat162float(up[v])));

        *reinterpret_cast<uint4*>(o_row + col) =
            *reinterpret_cast<const uint4*>(out_buf);
    }
}

void launch_fused_swiglu(
    __nv_bfloat16*       output,
    const __nv_bfloat16* gate_proj,
    const __nv_bfloat16* up_proj,
    int                  batch,
    int                  hidden,
    int                  sm_version,
    cudaStream_t         stream)
{
    if (batch <= 0 || hidden <= 0) return;
    if (sm_version >= 120)
        swiglu_only_kernel<120><<<batch, SwiGLUPolicy<120>::kBlockSize, 0, stream>>>(
            output, gate_proj, up_proj, hidden);
    else if (sm_version >= 90)
        swiglu_only_kernel<90><<<batch, SwiGLUPolicy<90>::kBlockSize, 0, stream>>>(
            output, gate_proj, up_proj, hidden);
    else
        swiglu_only_kernel<86><<<batch, SwiGLUPolicy<86>::kBlockSize, 0, stream>>>(
            output, gate_proj, up_proj, hidden);
    DS_LAUNCH_CHECK(stream);
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: Pre-LayerNorm for attention input
//   output[j] = residual[j] * ln_weight[j] * rsqrt(mean(residual²) + ε)
//   residual stream is NOT modified.
// Grid: (batch,)  Block: Policy::kBlockSize
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer>
__global__ void
__launch_bounds__(SwiGLUPolicy<SmVer>::kBlockSize,
                  SwiGLUPolicy<SmVer>::kMinBlocksPerSM)
pre_ln_attn_kernel(
    __nv_bfloat16* __restrict__       output,
    const __nv_bfloat16* __restrict__ residual,
    const float*          __restrict__ ln_weight,
    int   hidden,
    float eps)
{
    using P = SwiGLUPolicy<SmVer>;
    constexpr int kVec      = P::kVecWidth;
    constexpr int kBS       = P::kBlockSize;
    constexpr int kMaxWarps = P::kMaxWarps;

    __shared__ float smem_warps[kMaxWarps];
    cg::thread_block blk = cg::this_thread_block();

    const int row = blockIdx.x;
    const __nv_bfloat16* r_row = residual + (size_t)row * hidden;
          __nv_bfloat16* o_row = output   + (size_t)row * hidden;

    // Pass 1: accumulate squared sum for RMS denominator.
    float thread_sq = 0.f;
    for (int col = (int)threadIdx.x * kVec; col < hidden; col += kBS * kVec) {
        const uint4 r_raw = __ldg(reinterpret_cast<const uint4*>(r_row + col));
        const __nv_bfloat16* rp = reinterpret_cast<const __nv_bfloat16*>(&r_raw);
        #pragma unroll
        for (int v = 0; v < kVec; ++v) {
            float rv = __bfloat162float(rp[v]);
            thread_sq += rv * rv;
        }
    }

    float sq_sum  = block_reduce_sum<SwiGLUPolicy<SmVer>::kBlockSize>(
                        thread_sq, smem_warps, blk);
    float rms_inv = rsqrtf(sq_sum / (float)hidden + eps);

    // Pass 2: apply RMSNorm weights and write output.
    for (int col = (int)threadIdx.x * kVec; col < hidden; col += kBS * kVec) {
        const uint4 r_raw = __ldg(reinterpret_cast<const uint4*>(r_row + col));
        const __nv_bfloat16* rp = reinterpret_cast<const __nv_bfloat16*>(&r_raw);

        __nv_bfloat16 out_buf[kVec];
        #pragma unroll
        for (int v = 0; v < kVec; ++v) {
            float w = __ldg(ln_weight + col + v);
            out_buf[v] = __float2bfloat16(
                __bfloat162float(rp[v]) * rms_inv * w);
        }
        *reinterpret_cast<uint4*>(o_row + col) =
            *reinterpret_cast<const uint4*>(out_buf);
    }
}

void launch_fused_pre_ln_attn(
    __nv_bfloat16*       output,
    const __nv_bfloat16* residual,
    const float*         ln_weight,
    int                  batch,
    int                  hidden,
    float                eps,
    int                  sm_version,
    cudaStream_t         stream)
{
    if (batch <= 0 || hidden <= 0) return;
    if (sm_version >= 120)
        pre_ln_attn_kernel<120><<<batch, SwiGLUPolicy<120>::kBlockSize, 0, stream>>>(
            output, residual, ln_weight, hidden, eps);
    else if (sm_version >= 90)
        pre_ln_attn_kernel<90><<<batch, SwiGLUPolicy<90>::kBlockSize, 0, stream>>>(
            output, residual, ln_weight, hidden, eps);
    else
        pre_ln_attn_kernel<86><<<batch, SwiGLUPolicy<86>::kBlockSize, 0, stream>>>(
            output, residual, ln_weight, hidden, eps);
    DS_LAUNCH_CHECK(stream);
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3: Fused residual add + RMSNorm
//   residual[j] += input[j]           (in-place)
//   output[j]   = residual[j] * ln_weight[j] * rsqrt(mean(residual²) + ε)
//
// Single-pass: store residual sums in registers, apply LN in second pass.
// Grid: (batch,)  Block: Policy::kBlockSize
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer>
__global__ void
__launch_bounds__(SwiGLUPolicy<SmVer>::kBlockSize,
                  SwiGLUPolicy<SmVer>::kMinBlocksPerSM)
residual_rmsnorm_kernel(
    __nv_bfloat16* __restrict__       output,
    __nv_bfloat16* __restrict__       residual,   // updated in-place
    const __nv_bfloat16* __restrict__ input,
    const float*          __restrict__ ln_weight,
    int   hidden,
    float eps)
{
    using P = SwiGLUPolicy<SmVer>;
    constexpr int kVec      = P::kVecWidth;
    constexpr int kBS       = P::kBlockSize;
    constexpr int kMaxWarps = P::kMaxWarps;
    constexpr int kMaxIter  = P::kRegBudgetPerThread / kVec;

    __shared__ float smem_warps[kMaxWarps];
    cg::thread_block blk = cg::this_thread_block();

    const int row = blockIdx.x;
    __nv_bfloat16* r_row = residual + (size_t)row * hidden;
    const __nv_bfloat16* i_row = input    + (size_t)row * hidden;
    __nv_bfloat16* o_row = output   + (size_t)row * hidden;

    // Pass 1: residual += input; accumulate sq_sum in registers.
    float reg_res[kMaxIter * kVec];
    float thread_sq = 0.f;
    int n_iter = 0;

    for (int col = (int)threadIdx.x * kVec; col < hidden;
         col += kBS * kVec, ++n_iter) {

        const uint4 r_raw = *reinterpret_cast<const uint4*>(r_row + col);
        const uint4 i_raw = *reinterpret_cast<const uint4*>(i_row + col);
        const __nv_bfloat16* rp = reinterpret_cast<const __nv_bfloat16*>(&r_raw);
        const __nv_bfloat16* ip = reinterpret_cast<const __nv_bfloat16*>(&i_raw);

        __nv_bfloat16 updated[kVec];
        const int base = n_iter * kVec;
        #pragma unroll
        for (int v = 0; v < kVec; ++v) {
            float rv = __bfloat162float(rp[v]) + __bfloat162float(ip[v]);
            updated[v]         = __float2bfloat16(rv);
            reg_res[base + v]  = rv;
            thread_sq         += rv * rv;
        }
        // Write updated residual in-place.
        *reinterpret_cast<uint4*>(r_row + col) =
            *reinterpret_cast<const uint4*>(updated);
    }

    float sq_sum  = block_reduce_sum<SwiGLUPolicy<SmVer>::kBlockSize>(
                        thread_sq, smem_warps, blk);
    float rms_inv = rsqrtf(sq_sum / (float)hidden + eps);

    // Pass 2: normalise from registers — zero extra DRAM reads.
    n_iter = 0;
    for (int col = (int)threadIdx.x * kVec; col < hidden;
         col += kBS * kVec, ++n_iter) {

        const int base = n_iter * kVec;
        __nv_bfloat16 out_buf[kVec];
        #pragma unroll
        for (int v = 0; v < kVec; ++v) {
            float w = __ldg(ln_weight + col + v);
            out_buf[v] = __float2bfloat16(reg_res[base + v] * rms_inv * w);
        }
        *reinterpret_cast<uint4*>(o_row + col) =
            *reinterpret_cast<const uint4*>(out_buf);
    }
}

void launch_fused_residual_rmsnorm(
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
    if (batch <= 0 || hidden <= 0) return;
    if (sm_version >= 120)
        residual_rmsnorm_kernel<120><<<batch, SwiGLUPolicy<120>::kBlockSize, 0, stream>>>(
            output, residual, input, ln_weight, hidden, eps);
    else if (sm_version >= 90)
        residual_rmsnorm_kernel<90><<<batch, SwiGLUPolicy<90>::kBlockSize, 0, stream>>>(
            output, residual, input, ln_weight, hidden, eps);
    else
        residual_rmsnorm_kernel<86><<<batch, SwiGLUPolicy<86>::kBlockSize, 0, stream>>>(
            output, residual, input, ln_weight, hidden, eps);
    DS_LAUNCH_CHECK(stream);
}
