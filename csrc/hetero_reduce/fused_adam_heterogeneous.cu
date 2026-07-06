// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * fused_adam_heterogeneous.cu
 *
 * Fused Adam optimizer with per-tier learning-rate scaling for heterogeneous
 * GPU clusters (A6000 SM8.6 / H100 SM9.0 / Blackwell SM12.0) running over PCIe.
 *
 * ═══════════════════════════════════════════════════════════════════════
 * DESIGN RATIONALE
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Mixed-hardware training introduces a gradient-staleness asymmetry: slower
 * tiers (A6000) complete fewer micro-steps per wall-clock second than faster
 * tiers (H100 / Blackwell).  Applying a uniform learning rate to all tiers
 * yields systematically larger effective step sizes on slower tiers relative
 * to their contributed gradient quality.  Per-tier LR scaling compensates:
 *
 *   lr_effective(tier) = lr_base × lr_scale(tier)
 *
 * where lr_scale is supplied by the Python scheduler (e.g. proportional to
 * the tier's measured throughput or TFLOP/s rating).
 *
 * ═══════════════════════════════════════════════════════════════════════
 * ALGORITHMIC OVERVIEW
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Standard Adam update (Kingma & Ba, 2015):
 *   m_t   = β₁ · m_{t-1} + (1 − β₁) · g_t
 *   v_t   = β₂ · v_{t-1} + (1 − β₂) · g_t²
 *   m̂_t   = m_t / (1 − β₁ᵗ)      [bias correction]
 *   v̂_t   = v_t / (1 − β₂ᵗ)
 *   θ_t   = θ_{t-1} − lr · m̂_t / (√v̂_t + ε)
 *
 * Per-tier extension:
 *   θ_t   = θ_{t-1} − (lr_base × lr_scale) · m̂_t / (√v̂_t + ε)
 *
 * Optional weight decay (decoupled, AdamW style):
 *   θ_t   = (1 − lr · wd) · θ_{t-1} − lr_eff · m̂_t / (√v̂_t + ε)
 *
 * ═══════════════════════════════════════════════════════════════════════
 * IMPLEMENTATION HIGHLIGHTS
 * ═══════════════════════════════════════════════════════════════════════
 *
 * 1. COMPILE-TIME POLICY STRUCT
 *    AdamPolicy<SmVer> controls block size, min-CTAs-per-SM, and vec width —
 *    same idiom as KernelPolicy<SmVer> in hetero_reduce.cu.
 *
 * 2. VECTORISED 128-BIT LOADS / STORES
 *    Parameters (θ) are stored in BF16; moments (m, v) in FP32.
 *    Each thread processes kVecWidth = 8 elements per iteration:
 *      - Load 8 × BF16 param as uint4 (128-bit)
 *      - Load 8 × FP32 m-moment as two float4 (128-bit each)
 *      - Load 8 × FP32 v-moment as two float4
 *    Accumulate in FP32, store BF16 param and FP32 moments back.
 *
 * 3. BIAS-CORRECTION PRECOMPUTATION
 *    bc1 = 1 / (1 − β₁ᵗ)  and  bc2 = 1 / (1 − β₂ᵗ)  are computed on the
 *    host and passed as scalar arguments, avoiding redundant __powf() calls
 *    in every thread.
 *
 * 4. OPTIONAL MASTER-WEIGHT FP32 PATH
 *    When master_params != nullptr the kernel also reads/writes a full-FP32
 *    master copy and updates the BF16 working copy from it.  This guards
 *    against precision loss in the param update at very small LR (< 1e-4)
 *    where fp32(bf16 + small_delta) == bf16, a common pathology.
 *
 * 5. THREE-WAY SM DISPATCH
 *    The host-side launch wrapper dispatches to a compile-time-specialised
 *    kernel for SM 8.6 / SM 9.0 / SM 12.0.  Adding a new architecture
 *    requires only a new AdamPolicy<N> specialisation.
 *
 * 6. GRID-STRIDE LOOP
 *    The kernel loops over the parameter shard with a grid-stride to support
 *    arbitrary tensor sizes without separate tail handling.  The tail (last
 *    <8 elements) is handled scalar by thread 0.
 *
 * 7. PER-TIER LR SCALING SURFACE
 *    lr_scale is a float scalar.  Callers set it via the Python binding or
 *    C++ helper `hetero_adam_lr_scale(sm_version)` which returns a default
 *    weight proportional to tier throughput (SM12.0→4.0, SM9.0→3.0, SM8.6→1.0).
 *    The Python scheduler can override this per-step.
 *
 * ═══════════════════════════════════════════════════════════════════════
 * DATA LAYOUT
 * ═══════════════════════════════════════════════════════════════════════
 *
 * params        [n_elems]  BF16   (in/out)  working parameters
 * master_params [n_elems]  FP32   (in/out)  optional FP32 master copy
 * exp_avg       [n_elems]  FP32   (in/out)  first-moment estimate  (m)
 * exp_avg_sq    [n_elems]  FP32   (in/out)  second-moment estimate (v)
 * grads         [n_elems]  BF16   (in)      gradient for this step
 *
 * All buffers live on the same CUDA device.  Cross-device copies (reduce-
 * scatter → local shard) are handled upstream by hetero_reduce.cu.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdint.h>
#include <math.h>
#include <algorithm>

#include "hetero_reduce.h"
#include "ds_kernel_utils.h"

namespace cg = cooperative_groups;

// ─────────────────────────────────────────────────────────────────────────────
// Section 1: Compile-time tuning policy
//   Mirrors KernelPolicy<SmVer> in hetero_reduce.cu.
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer> struct AdamPolicy;

template <> struct AdamPolicy<86> {
    // A6000: 84 SMs @ SM8.6 — 256 threads, 2 resident CTAs/SM keep occupancy
    // while leaving register file headroom (≈64 regs/thread for Adam).
    static constexpr int kBlockSize      = 256;
    static constexpr int kMinBlocksPerSM = 2;
    // 8 BF16 elements per 128-bit vectorised load.
    static constexpr int kVecWidth       = 8;
};

template <> struct AdamPolicy<90> {
    // H100: 132 SMs @ SM9.0 — 256 threads, 4 CTAs/SM for higher occupancy.
    // H100 register file is twice as wide as A6000, so we can afford 4 CTAs.
    static constexpr int kBlockSize      = 256;
    static constexpr int kMinBlocksPerSM = 4;
    static constexpr int kVecWidth       = 8;
};

template <> struct AdamPolicy<120> {
    // Blackwell (GB200/B200): 128-wide SMs benefit from 512-thread blocks.
    // Larger CTA fills the wider issue slots.  4 CTAs/SM for full utilisation.
    static constexpr int kBlockSize      = 512;
    static constexpr int kMinBlocksPerSM = 4;
    static constexpr int kVecWidth       = 8;
};

// Generic fallback (forward-compatible with future SM versions).
template <int SmVer> struct AdamPolicy {
    static constexpr int kBlockSize      = 256;
    static constexpr int kMinBlocksPerSM = 2;
    static constexpr int kVecWidth       = 8;
};

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: Vectorised load / store helpers
// ─────────────────────────────────────────────────────────────────────────────

// Load 8 × BF16 elements from ptr as a single 128-bit read.
// Outputs: 8 floats via four float2 accumulators (matches bf16x8_accumulate
// convention from hetero_reduce.cu for consistency).
DS_D_INLINE void load_bf16x8(
    const __nv_bfloat16* __restrict__ ptr,
    float& f0, float& f1, float& f2, float& f3,
    float& f4, float& f5, float& f6, float& f7)
{
    const uint4 raw = *reinterpret_cast<const uint4*>(ptr);
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
    f0 = __bfloat162float(p[0].x);
    f1 = __bfloat162float(p[0].y);
    f2 = __bfloat162float(p[1].x);
    f3 = __bfloat162float(p[1].y);
    f4 = __bfloat162float(p[2].x);
    f5 = __bfloat162float(p[2].y);
    f6 = __bfloat162float(p[3].x);
    f7 = __bfloat162float(p[3].y);
}

// Store 8 floats as 8 × BF16 via a 128-bit write.
DS_D_INLINE void store_bf16x8(
    __nv_bfloat16* __restrict__ ptr,
    float f0, float f1, float f2, float f3,
    float f4, float f5, float f6, float f7)
{
    __nv_bfloat162 b0 = {__float2bfloat16(f0), __float2bfloat16(f1)};
    __nv_bfloat162 b1 = {__float2bfloat16(f2), __float2bfloat16(f3)};
    __nv_bfloat162 b2 = {__float2bfloat16(f4), __float2bfloat16(f5)};
    __nv_bfloat162 b3 = {__float2bfloat16(f6), __float2bfloat16(f7)};
    uint4 out;
    out.x = *reinterpret_cast<uint32_t*>(&b0);
    out.y = *reinterpret_cast<uint32_t*>(&b1);
    out.z = *reinterpret_cast<uint32_t*>(&b2);
    out.w = *reinterpret_cast<uint32_t*>(&b3);
    *reinterpret_cast<uint4*>(ptr) = out;
}

// Load 4 × FP32 via a 128-bit read.
DS_D_INLINE void load_fp32x4(
    const float* __restrict__ ptr,
    float& f0, float& f1, float& f2, float& f3)
{
    const float4 v = *reinterpret_cast<const float4*>(ptr);
    f0 = v.x; f1 = v.y; f2 = v.z; f3 = v.w;
}

// Store 4 × FP32 via a 128-bit write.
DS_D_INLINE void store_fp32x4(
    float* __restrict__ ptr,
    float f0, float f1, float f2, float f3)
{
    float4 v; v.x = f0; v.y = f1; v.z = f2; v.w = f3;
    *reinterpret_cast<float4*>(ptr) = v;
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 3: Core Adam update helper (scalar, called per element)
//
//   Applies:
//     m'  = β₁·m + (1−β₁)·g
//     v'  = β₂·v + (1−β₂)·g²
//     m̂   = m' · bc1          (bc1 = 1/(1−β₁ᵗ))
//     v̂   = v' · bc2          (bc2 = 1/(1−β₂ᵗ))
//     Δθ  = −lr_eff · m̂ / (√v̂ + ε) − lr_eff · wd · θ
//     θ'  = θ + Δθ
// ─────────────────────────────────────────────────────────────────────────────

DS_D_INLINE void adam_step_scalar(
    float&       param,          // in/out: FP32 parameter
    float&       m,              // in/out: first  moment
    float&       v,              // in/out: second moment
    float        grad,           // gradient (already FP32)
    float        lr_eff,         // effective learning rate = lr_base × lr_scale
    float        beta1,
    float        beta2,
    float        one_minus_beta1,
    float        one_minus_beta2,
    float        bc1,            // bias-correction-1 = 1 / (1 − β₁ᵗ)
    float        bc2,            // bias-correction-2 = 1 / (1 − β₂ᵗ)
    float        eps,
    float        weight_decay)
{
    // Moment updates (standard EMA).
    m = beta1 * m + one_minus_beta1 * grad;
    v = beta2 * v + one_minus_beta2 * (grad * grad);

    // Bias-corrected estimates.
    const float m_hat = m * bc1;
    const float v_hat = v * bc2;

    // Denominator: √v̂ + ε.
    const float denom = __fsqrt_rn(v_hat) + eps;

    // Decoupled weight-decay + Adam update.
    param = param - lr_eff * (m_hat / denom + weight_decay * param);
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 4: Main vectorised kernel
//
//   Template parameters:
//     SmVer           — SM version selects AdamPolicy specialisation
//     HasMasterParams — true: maintain FP32 master copy alongside BF16 params
//
//   Grid-stride loop: each thread processes kVecWidth elements per iteration,
//   spinning until all n_elems / kVecWidth vectors are processed.
//   Tail (<kVecWidth elements) handled scalar in thread 0 of the first block.
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer, bool HasMasterParams>
__global__ void
__launch_bounds__(AdamPolicy<SmVer>::kBlockSize, AdamPolicy<SmVer>::kMinBlocksPerSM)
fused_adam_hetero_kernel(
    __nv_bfloat16* __restrict__  params,          // [n_elems] BF16 working params (in/out)
    float*         __restrict__  master_params,   // [n_elems] FP32 master params (in/out, nullable)
    float*         __restrict__  exp_avg,         // [n_elems] first  moment m (in/out)
    float*         __restrict__  exp_avg_sq,      // [n_elems] second moment v (in/out)
    const __nv_bfloat16* __restrict__ grads,      // [n_elems] BF16 gradients  (in)
    size_t         n_elems,
    float          lr_eff,          // lr_base × lr_scale(tier)
    float          beta1,
    float          beta2,
    float          bc1,             // 1 / (1 − β₁ᵗ)
    float          bc2,             // 1 / (1 − β₂ᵗ)
    float          eps,
    float          weight_decay)
{
    constexpr int  kVec    = AdamPolicy<SmVer>::kVecWidth;   // 8
    constexpr int  kBS     = AdamPolicy<SmVer>::kBlockSize;

    // Precompute (1 − βᵢ) to avoid repeated 1.f − β in the inner loop.
    const float one_minus_beta1 = 1.f - beta1;
    const float one_minus_beta2 = 1.f - beta2;

    const size_t vec_n  = n_elems / kVec;
    const size_t stride = (size_t)gridDim.x * kBS;
    size_t       vid    = (size_t)blockIdx.x * kBS + threadIdx.x;

    // ── Vectorised main loop ──────────────────────────────────────────────────
    for (; vid < vec_n; vid += stride) {
        const size_t e = vid * kVec;

        // Load 8 × BF16 gradient elements.
        float g0, g1, g2, g3, g4, g5, g6, g7;
        load_bf16x8(grads + e, g0, g1, g2, g3, g4, g5, g6, g7);

        // Load 8 × FP32 parameter values.
        float p0, p1, p2, p3, p4, p5, p6, p7;
        if constexpr (HasMasterParams) {
            // Use the FP32 master copy as the authoritative parameter.
            load_fp32x4(master_params + e,     p0, p1, p2, p3);
            load_fp32x4(master_params + e + 4, p4, p5, p6, p7);
        } else {
            // Upcast BF16 working params to FP32 for the update.
            load_bf16x8(params + e, p0, p1, p2, p3, p4, p5, p6, p7);
        }

        // Load 8 × FP32 first moments (two float4 reads = 256-bit total).
        float m0, m1, m2, m3, m4, m5, m6, m7;
        load_fp32x4(exp_avg + e,     m0, m1, m2, m3);
        load_fp32x4(exp_avg + e + 4, m4, m5, m6, m7);

        // Load 8 × FP32 second moments.
        float v0, v1, v2, v3, v4, v5, v6, v7;
        load_fp32x4(exp_avg_sq + e,     v0, v1, v2, v3);
        load_fp32x4(exp_avg_sq + e + 4, v4, v5, v6, v7);

        // Apply Adam update for each of the 8 elements.
        // Unrolled manually to maximise ILP; the compiler can keep all
        // accumulators live in registers.
        adam_step_scalar(p0, m0, v0, g0, lr_eff, beta1, beta2, one_minus_beta1, one_minus_beta2, bc1, bc2, eps, weight_decay);
        adam_step_scalar(p1, m1, v1, g1, lr_eff, beta1, beta2, one_minus_beta1, one_minus_beta2, bc1, bc2, eps, weight_decay);
        adam_step_scalar(p2, m2, v2, g2, lr_eff, beta1, beta2, one_minus_beta1, one_minus_beta2, bc1, bc2, eps, weight_decay);
        adam_step_scalar(p3, m3, v3, g3, lr_eff, beta1, beta2, one_minus_beta1, one_minus_beta2, bc1, bc2, eps, weight_decay);
        adam_step_scalar(p4, m4, v4, g4, lr_eff, beta1, beta2, one_minus_beta1, one_minus_beta2, bc1, bc2, eps, weight_decay);
        adam_step_scalar(p5, m5, v5, g5, lr_eff, beta1, beta2, one_minus_beta1, one_minus_beta2, bc1, bc2, eps, weight_decay);
        adam_step_scalar(p6, m6, v6, g6, lr_eff, beta1, beta2, one_minus_beta1, one_minus_beta2, bc1, bc2, eps, weight_decay);
        adam_step_scalar(p7, m7, v7, g7, lr_eff, beta1, beta2, one_minus_beta1, one_minus_beta2, bc1, bc2, eps, weight_decay);

        // Write updated moments (FP32).
        store_fp32x4(exp_avg + e,     m0, m1, m2, m3);
        store_fp32x4(exp_avg + e + 4, m4, m5, m6, m7);
        store_fp32x4(exp_avg_sq + e,     v0, v1, v2, v3);
        store_fp32x4(exp_avg_sq + e + 4, v4, v5, v6, v7);

        // Write updated parameters.
        if constexpr (HasMasterParams) {
            // Commit full-precision master copy.
            store_fp32x4(master_params + e,     p0, p1, p2, p3);
            store_fp32x4(master_params + e + 4, p4, p5, p6, p7);
        }
        // Always write BF16 working copy (truncates FP32 → BF16).
        store_bf16x8(params + e, p0, p1, p2, p3, p4, p5, p6, p7);
    }

    // ── Scalar tail (elements not covered by the vectorised loop) ────────────
    // Thread 0 of block 0 cleans up the last (n_elems % kVecWidth) elements.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (size_t e = vec_n * kVec; e < n_elems; ++e) {
            float g = __bfloat162float(grads[e]);
            float p;
            if constexpr (HasMasterParams) {
                p = master_params[e];
            } else {
                p = __bfloat162float(params[e]);
            }
            float m = exp_avg[e];
            float v = exp_avg_sq[e];

            adam_step_scalar(p, m, v, g, lr_eff, beta1, beta2,
                             one_minus_beta1, one_minus_beta2,
                             bc1, bc2, eps, weight_decay);

            exp_avg[e]    = m;
            exp_avg_sq[e] = v;
            if constexpr (HasMasterParams) master_params[e] = p;
            params[e] = __float2bfloat16(p);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 5: SM-dispatch helper
//
//   Selects AdamPolicy<SmVer> at compile time, then computes an occupancy-
//   aware grid size targeting full SM utilisation.
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer>
static void dispatch_fused_adam(
    __nv_bfloat16*       params,
    float*               master_params,
    float*               exp_avg,
    float*               exp_avg_sq,
    const __nv_bfloat16* grads,
    size_t               n_elems,
    float                lr_eff,
    float                beta1,
    float                beta2,
    float                bc1,
    float                bc2,
    float                eps,
    float                weight_decay,
    cudaStream_t         stream)
{
    using Policy           = AdamPolicy<SmVer>;
    constexpr int kVec     = Policy::kVecWidth;
    constexpr int kBS      = Policy::kBlockSize;

    const size_t vec_n     = (n_elems + kVec - 1) / kVec;   // ceil
    const int    grid      = (int)std::min(
                                 (vec_n + (size_t)kBS - 1) / (size_t)kBS,
                                 (size_t)65535);

    if (master_params != nullptr) {
        fused_adam_hetero_kernel<SmVer, true>
            <<<grid, kBS, 0, stream>>>(
                params, master_params, exp_avg, exp_avg_sq, grads,
                n_elems, lr_eff, beta1, beta2, bc1, bc2, eps, weight_decay);
    } else {
        fused_adam_hetero_kernel<SmVer, false>
            <<<grid, kBS, 0, stream>>>(
                params, master_params, exp_avg, exp_avg_sq, grads,
                n_elems, lr_eff, beta1, beta2, bc1, bc2, eps, weight_decay);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 6: Public API
// ─────────────────────────────────────────────────────────────────────────────

/**
 * hetero_adam_lr_scale
 *
 * Returns the default per-tier learning-rate scale factor derived from each
 * GPU tier's normalised throughput (matches tier_weight() in hetero_reduce.cu):
 *
 *   SM 12.0 (Blackwell) → 4.0  (highest TFLOP/s tier)
 *   SM  9.0 (H100)      → 3.0
 *   SM  8.6 (A6000)     → 1.0  (baseline)
 *
 * The Python-level scheduler may override this per-step to account for dynamic
 * load imbalance or pipeline depth differences.
 *
 * @param sm_version  SM version of the current device (86, 90, 120, …)
 * @returns           float scale factor ≥ 1.0
 */
float hetero_adam_lr_scale(int sm_version)
{
    if (sm_version >= 120) return 4.0f;
    if (sm_version >= 90)  return 3.0f;
    return 1.0f;
}

/**
 * launch_fused_adam_heterogeneous
 *
 * Launches the fused Adam update kernel on the current device.  Dispatches to
 * a compile-time-specialised kernel for SM 8.6 / 9.0 / 12.0.
 *
 * The effective learning rate applied is:
 *   lr_eff = lr_base × lr_scale
 *
 * Bias corrections bc1 and bc2 are pre-computed by the caller to avoid
 * per-kernel __powf() overhead:
 *   bc1 = 1.0 / (1.0 - pow(beta1, step))
 *   bc2 = 1.0 / (1.0 - pow(beta2, step))
 *
 * @param params        [in/out] BF16 working parameters, length n_elems
 * @param master_params [in/out] FP32 master copy (may be nullptr to skip)
 * @param exp_avg       [in/out] FP32 first-moment buffer,  length n_elems
 * @param exp_avg_sq    [in/out] FP32 second-moment buffer, length n_elems
 * @param grads         [in]     BF16 gradient tensor, length n_elems
 * @param n_elems       Number of parameter elements (must be > 0)
 * @param lr_base       Base learning rate (before tier scaling)
 * @param lr_scale      Per-tier LR scale (use hetero_adam_lr_scale() for default)
 * @param beta1         Adam β₁ (typically 0.9)
 * @param beta2         Adam β₂ (typically 0.999)
 * @param bc1           Bias correction 1 = 1/(1−β₁^step)
 * @param bc2           Bias correction 2 = 1/(1−β₂^step)
 * @param eps           Adam ε (typically 1e-8)
 * @param weight_decay  Decoupled weight-decay coefficient (0.0 to disable)
 * @param sm_version    SM version of the current device (86, 90, 120, …)
 * @param stream        CUDA stream to launch on
 */
void launch_fused_adam_heterogeneous(
    __nv_bfloat16*       params,
    float*               master_params,
    float*               exp_avg,
    float*               exp_avg_sq,
    const __nv_bfloat16* grads,
    size_t               n_elems,
    float                lr_base,
    float                lr_scale,
    float                beta1,
    float                beta2,
    float                bc1,
    float                bc2,
    float                eps,
    float                weight_decay,
    int                  sm_version,
    cudaStream_t         stream)
{
    if (n_elems == 0) return;

    const float lr_eff = lr_base * lr_scale;

    if (sm_version >= 120)
        dispatch_fused_adam<120>(params, master_params, exp_avg, exp_avg_sq,
                                  grads, n_elems, lr_eff,
                                  beta1, beta2, bc1, bc2, eps, weight_decay,
                                  stream);
    else if (sm_version >= 90)
        dispatch_fused_adam<90>(params, master_params, exp_avg, exp_avg_sq,
                                 grads, n_elems, lr_eff,
                                 beta1, beta2, bc1, bc2, eps, weight_decay,
                                 stream);
    else
        dispatch_fused_adam<86>(params, master_params, exp_avg, exp_avg_sq,
                                 grads, n_elems, lr_eff,
                                 beta1, beta2, bc1, bc2, eps, weight_decay,
                                 stream);
}
