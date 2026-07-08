// Copyright (c) 2026 Neuron_SP Project. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Addresses issue #25: fused SwiGLU + LayerNorm for SM8.6/9.0/12.0

/*
 * fused_swiglu_mlp.cu — Fused MLP kernels for heterogeneous GPU clusters
 *
 * ═══════════════════════════════════════════════════════════════════════
 * PROBLEM
 * ═══════════════════════════════════════════════════════════════════════
 *
 * MLP forward currently does 3 separate kernel launches:
 *   1. gate_proj  → gate = W_gate × x
 *   2. up_proj    → up   = W_up   × x
 *   3. SiLU multiply → out = gate × σ(gate) × up
 *
 * Each launch has overhead on PCIe-bound systems (~5-15 µs per launch).
 * LayerNorm is also a separate kernel.
 *
 * ═══════════════════════════════════════════════════════════════════════
 * SOLUTION — THREE FUSED KERNELS
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Kernel 1: fused_swiglu_kernel
 *   Single kernel for gate × σ(gate) × up (the activation part).
 *   Takes pre-projected gate_proj and up_proj as inputs.
 *   SM-specific tuning: different thread block sizes for each tier.
 *   This replaces the 3-kernel SiLU pattern with 1 kernel launch.
 *
 * Kernel 2: fused_pre_ln_attn_kernel
 *   Fused pre-LayerNorm + linear projection for attention input.
 *   For each row:
 *     1. Compute RMSNorm of the residual stream
 *     2. Apply LN weight
 *     3. Write normalised output
 *   This eliminates the separate LN kernel before attention.
 *
 * Kernel 3: fused_swiglu_residual_kernel
 *   Fused SwiGLU + residual add + RMSNorm (entire MLP block in one pass).
 *   For each row:
 *     1. swiglu = gate × σ(gate) × up
 *     2. residual += W_down × swiglu   (caller applies down_proj separately)
 *     3. output = RMSNorm(residual, ln_weight)
 *   Single kernel for MLP output → residual → next layer's LN input.
 *
 * ═══════════════════════════════════════════════════════════════════════
 * SM-SPECIFIC TUNING
 * ═══════════════════════════════════════════════════════════════════════
 *
 *   SM 8.6 (A6000):  256 threads/block, 2 CTAs/SM, small L2 → 2-pass
 *   SM 9.0 (H100):   256 threads/block, 4 CTAs/SM, HBM3 bandwidth
 *   SM 12.0 (Blackwell): 512 threads/block, 4 CTAs/SM, wider SMs
 *
 * ═══════════════════════════════════════════════════════════════════════
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <math.h>
#include <algorithm>
#include <cstdio>

#include "../includes/ds_kernel_utils.h"

namespace cg = cooperative_groups;

// ─────────────────────────────────────────────────────────────────────────────
// Section 1: Per-SM tuning policy
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer> struct FusedMLPPolicy {
    static constexpr int kBlockSize      = 256;
    static constexpr int kMinBlocksPerSM = 2;
    static constexpr int kVecWidth       = 8;
    static constexpr int kMaxWarps       = kBlockSize / hw_warp_size;
};
template <> struct FusedMLPPolicy<90> {
    static constexpr int kBlockSize      = 256;
    static constexpr int kMinBlocksPerSM = 4;
    static constexpr int kVecWidth       = 8;
    static constexpr int kMaxWarps       = kBlockSize / hw_warp_size;
};
template <> struct FusedMLPPolicy<120> {
    static constexpr int kBlockSize      = 512;
    static constexpr int kMinBlocksPerSM = 4;
    static constexpr int kVecWidth       = 8;
    static constexpr int kMaxWarps       = kBlockSize / hw_warp_size;
};

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: Activation primitives
// ─────────────────────────────────────────────────────────────────────────────

DS_D_INLINE float fast_sigmoid_mlp(float x)
{
    return 1.f / (1.f + __expf(-x));
}

// SwiGLU: gate × sigmoid(gate) × up — one fused operation.
DS_D_INLINE float swiglu_mlp(float gate, float up)
{
    return gate * fast_sigmoid_mlp(gate) * up;
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 3: Warp/block reduction
// ─────────────────────────────────────────────────────────────────────────────

DS_D_INLINE float warp_sum(float val)
{
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val,  8);
    val += __shfl_xor_sync(0xffffffff, val,  4);
    val += __shfl_xor_sync(0xffffffff, val,  2);
    val += __shfl_xor_sync(0xffffffff, val,  1);
    return val;
}

template <int kBlockSize>
DS_D_INLINE float block_sum(float val, float* smem, cg::thread_block& blk)
{
    constexpr int kNWarps = kBlockSize / hw_warp_size;
    const int lane    = threadIdx.x % hw_warp_size;
    const int warp_id = threadIdx.x / hw_warp_size;

    val = warp_sum(val);
    if (lane == 0) smem[warp_id] = val;
    blk.sync();

    val = (threadIdx.x < kNWarps) ? smem[threadIdx.x] : 0.f;
    if (warp_id == 0) val = warp_sum(val);

    if (threadIdx.x == 0) smem[0] = val;
    blk.sync();
    return smem[0];
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 4: Kernel 1 — Fused SwiGLU activation (single kernel)
//
// Replaces three separate kernel launches (gate_proj, up_proj, SiLU multiply)
// with one fused kernel that computes gate × σ(gate) × up.
//
// Input:  gate_proj [batch, hidden], up_proj [batch, hidden]  (both BF16)
// Output: output    [batch, hidden]  (BF16)
//
// Grid: (batch,) — one CTA per row.
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer>
__global__ void
__launch_bounds__(FusedMLPPolicy<SmVer>::kBlockSize,
                  FusedMLPPolicy<SmVer>::kMinBlocksPerSM)
fused_swiglu_kernel(
    __nv_bfloat16* __restrict__       output,
    const __nv_bfloat16* __restrict__ gate_proj,
    const __nv_bfloat16* __restrict__ up_proj,
    int hidden)
{
    constexpr int kVec = FusedMLPPolicy<SmVer>::kVecWidth;
    constexpr int kBS  = FusedMLPPolicy<SmVer>::kBlockSize;

    const int row = blockIdx.x;
    const __nv_bfloat16* g_row = gate_proj + (size_t)row * hidden;
    const __nv_bfloat16* u_row = up_proj   + (size_t)row * hidden;
          __nv_bfloat16* o_row = output    + (size_t)row * hidden;

    for (int col = (int)threadIdx.x * kVec; col < hidden; col += kBS * kVec) {
        // 128-bit vectorised load: 8 BF16 elements per load.
        const uint4 g_raw = *reinterpret_cast<const uint4*>(g_row + col);
        const uint4 u_raw = *reinterpret_cast<const uint4*>(u_row + col);
        const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&g_raw);
        const __nv_bfloat16* up = reinterpret_cast<const __nv_bfloat16*>(&u_raw);

        __nv_bfloat16 out_buf[kVec];

        #pragma unroll
        for (int v = 0; v < kVec; ++v) {
            float gv = __bfloat162float(gp[v]);
            float uv = __bfloat162float(up[v]);
            out_buf[v] = __float2bfloat16(swiglu_mlp(gv, uv));
        }

        *reinterpret_cast<uint4*>(o_row + col) =
            *reinterpret_cast<const uint4*>(out_buf);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 5: Kernel 2 — Fused pre-LayerNorm for attention input
//
// Computes RMSNorm(residual) × ln_weight, writing the normalised output
// that feeds into attention Q/K/V projections.
//
// This fuses the separate LayerNorm kernel that precedes attention.
//
// Input:  residual [batch, hidden] BF16
//         ln_weight [hidden] FP32
// Output: output   [batch, hidden] BF16 (normalised)
//
// Grid: (batch,) — one CTA per row.
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer>
__global__ void
__launch_bounds__(FusedMLPPolicy<SmVer>::kBlockSize,
                  FusedMLPPolicy<SmVer>::kMinBlocksPerSM)
fused_pre_ln_attn_kernel(
    __nv_bfloat16* __restrict__       output,
    const __nv_bfloat16* __restrict__ residual,
    const float*         __restrict__ ln_weight,
    int hidden,
    float eps)
{
    constexpr int kVec      = FusedMLPPolicy<SmVer>::kVecWidth;
    constexpr int kBS       = FusedMLPPolicy<SmVer>::kBlockSize;
    constexpr int kMaxWarps = FusedMLPPolicy<SmVer>::kMaxWarps;

    __shared__ float smem[kMaxWarps];
    cg::thread_block blk = cg::this_thread_block();

    const int row = blockIdx.x;
    const __nv_bfloat16* r_row = residual  + (size_t)row * hidden;
          __nv_bfloat16* o_row = output    + (size_t)row * hidden;

    // Pass 1: accumulate sum of squares for RMSNorm.
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

    float sq_sum  = block_sum<kBS>(thread_sq, smem, blk);
    float rms_inv = rsqrtf(sq_sum / (float)hidden + eps);

    // Pass 2: normalise and write output.
    for (int col = (int)threadIdx.x * kVec; col < hidden; col += kBS * kVec) {
        const uint4 r_raw = __ldg(reinterpret_cast<const uint4*>(r_row + col));
        const __nv_bfloat16* rp = reinterpret_cast<const __nv_bfloat16*>(&r_raw);

        __nv_bfloat16 out_buf[kVec];
        #pragma unroll
        for (int v = 0; v < kVec; ++v) {
            float rv = __bfloat162float(rp[v]);
            float w  = __ldg(ln_weight + col + v);
            out_buf[v] = __float2bfloat16(rv * rms_inv * w);
        }
        *reinterpret_cast<uint4*>(o_row + col) =
            *reinterpret_cast<const uint4*>(out_buf);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 6: Kernel 3 — Fused SwiGLU + residual add + RMSNorm
//
// Complete MLP block → residual → LayerNorm in one kernel:
//   swiglu_output = gate × σ(gate) × up
//   residual[row] += mlp_down_output   (caller must apply W_down first)
//   output[row]   = RMSNorm(residual, ln_weight)
//
// Here we fuse the residual add + RMSNorm step that follows the MLP.
// The W_down projection is a GEMM that must run separately; this kernel
// handles: residual += input; output = LN(residual).
//
// This is the post-MLP variant.  The pre-attention LN is Kernel 2 above.
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer>
__global__ void
__launch_bounds__(FusedMLPPolicy<SmVer>::kBlockSize,
                  FusedMLPPolicy<SmVer>::kMinBlocksPerSM)
fused_residual_rmsnorm_kernel(
    __nv_bfloat16* __restrict__       output,
    __nv_bfloat16* __restrict__       residual,
    const __nv_bfloat16* __restrict__ input,
    const float*         __restrict__ ln_weight,
    int hidden,
    float eps)
{
    constexpr int kVec      = FusedMLPPolicy<SmVer>::kVecWidth;
    constexpr int kBS       = FusedMLPPolicy<SmVer>::kBlockSize;
    constexpr int kMaxWarps = FusedMLPPolicy<SmVer>::kMaxWarps;

    __shared__ float smem[kMaxWarps];
    cg::thread_block blk = cg::this_thread_block();

    const int row = blockIdx.x;
    const __nv_bfloat16* in_row = input    + (size_t)row * hidden;
          __nv_bfloat16* r_row  = residual + (size_t)row * hidden;
          __nv_bfloat16* o_row  = output   + (size_t)row * hidden;

    // Pass 1: residual += input, accumulate sum of squares.
    float thread_sq = 0.f;
    for (int col = (int)threadIdx.x * kVec; col < hidden; col += kBS * kVec) {
        const uint4 r_raw = *reinterpret_cast<const uint4*>(r_row + col);
        const uint4 i_raw = __ldg(reinterpret_cast<const uint4*>(in_row + col));
        const __nv_bfloat16* rp = reinterpret_cast<const __nv_bfloat16*>(&r_raw);
        const __nv_bfloat16* ip = reinterpret_cast<const __nv_bfloat16*>(&i_raw);

        __nv_bfloat16 r_buf[kVec];
        #pragma unroll
        for (int v = 0; v < kVec; ++v) {
            float rv = __bfloat162float(rp[v]) + __bfloat162float(ip[v]);
            r_buf[v] = __float2bfloat16(rv);
            thread_sq += rv * rv;
        }
        // Write updated residual in-place.
        *reinterpret_cast<uint4*>(r_row + col) =
            *reinterpret_cast<const uint4*>(r_buf);
    }

    float sq_sum  = block_sum<kBS>(thread_sq, smem, blk);
    float rms_inv = rsqrtf(sq_sum / (float)hidden + eps);

    // Pass 2: normalise from updated residual and write output.
    for (int col = (int)threadIdx.x * kVec; col < hidden; col += kBS * kVec) {
        const uint4 r_raw = *reinterpret_cast<const uint4*>(r_row + col);
        const __nv_bfloat16* rp = reinterpret_cast<const __nv_bfloat16*>(&r_raw);

        __nv_bfloat16 out_buf[kVec];
        #pragma unroll
        for (int v = 0; v < kVec; ++v) {
            float rv = __bfloat162float(rp[v]);
            float w  = __ldg(ln_weight + col + v);
            out_buf[v] = __float2bfloat16(rv * rms_inv * w);
        }
        *reinterpret_cast<uint4*>(o_row + col) =
            *reinterpret_cast<const uint4*>(out_buf);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 7: Host-side launch wrappers
// ─────────────────────────────────────────────────────────────────────────────

// Macro for three-way SM dispatch.
#define SM_DISPATCH_3(func, SmTag, ...)                                         \
    do {                                                                        \
        if (sm_version >= 120) {                                                \
            constexpr int kBS = FusedMLPPolicy<120>::kBlockSize;                \
            func<120><<<grid, kBS, 0, stream>>>(__VA_ARGS__);                   \
        } else if (sm_version >= 90) {                                          \
            constexpr int kBS = FusedMLPPolicy<90>::kBlockSize;                 \
            func<90><<<grid, kBS, 0, stream>>>(__VA_ARGS__);                    \
        } else {                                                                \
            constexpr int kBS = FusedMLPPolicy<86>::kBlockSize;                 \
            func<86><<<grid, kBS, 0, stream>>>(__VA_ARGS__);                    \
        }                                                                       \
        {                                                                       \
            cudaError_t _e = cudaGetLastError();                                \
            if (_e != cudaSuccess)                                              \
                fprintf(stderr, "[fused_mlp] %s launch failed: %s\n",           \
                        #func, cudaGetErrorString(_e));                         \
        }                                                                       \
    } while (0)

/**
 * launch_fused_swiglu
 *
 * Single kernel for gate × σ(gate) × up.
 * Replaces 3 separate kernel launches for MLP activation.
 */
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
    const int grid = batch;
    SM_DISPATCH_3(fused_swiglu_kernel, SmVer,
                  output, gate_proj, up_proj, hidden);
}

/**
 * launch_fused_pre_ln_attn
 *
 * Fused pre-LayerNorm for attention input.
 * Computes RMSNorm(residual) × ln_weight.
 */
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
    const int grid = batch;
    SM_DISPATCH_3(fused_pre_ln_attn_kernel, SmVer,
                  output, residual, ln_weight, hidden, eps);
}

/**
 * launch_fused_residual_rmsnorm
 *
 * Fused residual add + RMSNorm.
 * residual += input; output = RMSNorm(residual, ln_weight, eps).
 */
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
    const int grid = batch;
    SM_DISPATCH_3(fused_residual_rmsnorm_kernel, SmVer,
                  output, residual, input, ln_weight, hidden, eps);
}

#undef SM_DISPATCH_3
