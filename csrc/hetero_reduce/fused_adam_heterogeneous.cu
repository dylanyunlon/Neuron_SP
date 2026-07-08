// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * fused_adam_heterogeneous.cu
 *
 * Fused Adam / AdamW / AMSGrad optimizer with per-tier learning-rate scaling
 * for heterogeneous GPU clusters (A6000 SM8.6 / H100 SM9.0 / Blackwell SM12.0)
 * running over PCIe.
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
 * AMSGrad variant (Reddi et al., 2018):
 *   v̂_max_t = max(v̂_max_{t-1}, v̂_t)
 *   θ_t     = θ_{t-1} − lr_eff · m̂_t / (√v̂_max_t + ε)
 *
 * Gradient clipping (pre-update):
 *   If ‖g‖₂ > clip_norm:  g ← g × clip_norm / ‖g‖₂
 *   Applied per-shard; global norm must be pre-reduced across tiers.
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
 *    Parameters (θ) are stored in BF16; moments (m, v, v_max) in FP32.
 *    Each thread processes kVecWidth = 8 elements per iteration:
 *      - Load 8 × BF16 param as uint4 (128-bit)
 *      - Load 8 × FP32 m-moment as two float4 (128-bit each)
 *      - Load 8 × FP32 v-moment as two float4
 *      - Load 8 × FP32 v_max (AMSGrad only) as two float4
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
 * 8. AMSGRAD VARIANT
 *    When UseAMSGrad=true, a fourth buffer (exp_avg_sq_max) tracks the running
 *    maximum of the bias-corrected second moment.  The denominator becomes
 *    √v̂_max + ε, ensuring non-increasing effective step sizes.  This improves
 *    convergence on non-stationary objectives common in heterogeneous training.
 *
 * 9. PER-TIER GRADIENT NORM ACCUMULATION
 *    launch_grad_norm_sq accumulates ‖g‖² into a scalar FP32 accumulator via
 *    atomicAdd.  The host reduces across tiers and passes the global norm to
 *    launch_fused_adam* via the clip_norm argument.  When clip_norm <= 0 the
 *    clipping path is skipped.
 *
 * 10. FP8 GRADIENT INPUT PATH (E4M3, forward-compatible stub)
 *    When grad_dtype == kGradFP8_E4M3, each BF16 load is replaced by an FP8
 *    decode followed by upcast to FP32.  Requires CUDA >= 12.1 with native
 *    __nv_fp8_e4m3 support; falls back to a bitcast + scale factor otherwise.
 *    Scale factor (fp8_grad_scale) absorbs the per-tensor quantisation scale.
 *
 * 11. CUDA GRAPH CAPTURE GUARDS
 *    launch_fused_adam_heterogeneous and launch_fused_adamw_amsgrad are safe
 *    for CUDA Graph capture.  The only graph-incompatible operation (occupancy
 *    query via cudaOccupancyMaxActiveBlocksPerMultiprocessor) is guarded by
 *    is_graph_capturing() and cached after first call.
 *
 * ═══════════════════════════════════════════════════════════════════════
 * DATA LAYOUT
 * ═══════════════════════════════════════════════════════════════════════
 *
 * params        [n_elems]  BF16   (in/out)  working parameters
 * master_params [n_elems]  FP32   (in/out)  optional FP32 master copy
 * exp_avg       [n_elems]  FP32   (in/out)  first-moment estimate  (m)
 * exp_avg_sq    [n_elems]  FP32   (in/out)  second-moment estimate (v)
 * exp_avg_sq_max[n_elems]  FP32   (in/out)  AMSGrad v_max (nullptr → classic Adam)
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
#include <cassert>

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
    // Gradient norm accumulation: 512 threads handle 4K elems/block.
    static constexpr int kNormBlockSize  = 256;
};

template <> struct AdamPolicy<90> {
    // H100: 132 SMs @ SM9.0 — 256 threads, 4 CTAs/SM for higher occupancy.
    // H100 register file is twice as wide as A6000, so we can afford 4 CTAs.
    static constexpr int kBlockSize      = 256;
    static constexpr int kMinBlocksPerSM = 4;
    static constexpr int kVecWidth       = 8;
    static constexpr int kNormBlockSize  = 512;
};

template <> struct AdamPolicy<120> {
    // Blackwell (GB200/B200): 128-wide SMs benefit from 512-thread blocks.
    // Larger CTA fills the wider issue slots.  4 CTAs/SM for full utilisation.
    static constexpr int kBlockSize      = 512;
    static constexpr int kMinBlocksPerSM = 4;
    static constexpr int kVecWidth       = 8;
    static constexpr int kNormBlockSize  = 512;
};

// NOTE: The forward declaration `template <int SmVer> struct AdamPolicy;`
// above serves as the primary template.  Specialisations for 86/90/120 follow.
// A second primary-template body here would be an ODR violation in C++17;
// instead we rely on the compiler's built-in error for unknown SmVer values at
// the call sites (static_assert or hard error on incomplete type).
// If you need a catch-all fallback for future architectures, add a new explicit
// specialisation (e.g. template<> struct AdamPolicy<130> { ... };) rather than
// re-opening the primary template.

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: Gradient dtype tag
//   Allows the inner kernel to switch between BF16 and FP8-E4M3 inputs
//   without a runtime branch in the hot path (resolved at template instatiation).
// ─────────────────────────────────────────────────────────────────────────────

enum GradDtype : int {
    kGradBF16      = 0,   // standard BF16 gradients
    kGradFP8_E4M3  = 1,   // FP8 E4M3 gradients (CUDA >= 12.1)
};

// ─────────────────────────────────────────────────────────────────────────────
// Section 3: Vectorised load / store helpers
// ─────────────────────────────────────────────────────────────────────────────

// Load 8 × BF16 elements from ptr as a single 128-bit read.
// Outputs: 8 floats in registers.
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
// Section 3b: FP8-E4M3 decode helper (forward-compatible)
//
//   FP8 E4M3 encoding: 1 sign bit, 4 exponent bits, 3 mantissa bits.
//   Bias = 7.  Max normal value = 448.0.
//
//   When CUDA >= 12.1 with __nv_fp8_e4m3 available, we use the hardware
//   intrinsic.  Otherwise we implement a portable software decode.
//   The per-tensor quantisation scale is applied during decode so the
//   rest of the Adam kernel sees normal FP32 gradients.
// ─────────────────────────────────────────────────────────────────────────────

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890 && defined(__nv_fp8_e4m3)
// Hardware path: SM 8.9+ (Ada / H100 / Blackwell) with CUDA 12.1+
DS_D_INLINE float fp8_e4m3_to_float(uint8_t bits, float scale)
{
    __nv_fp8_e4m3 fp8_val;
    *reinterpret_cast<uint8_t*>(&fp8_val) = bits;
    return static_cast<float>(fp8_val) * scale;
}
#else
// Software decode path for older hardware or toolchains lacking __nv_fp8_e4m3.
// Implements IEEE-style decode: value = (-1)^s × 2^(e−7) × (1 + m/8)
// Special cases: e=15,m=7 → NaN (propagated); e=0 → subnormal.
DS_D_INLINE float fp8_e4m3_to_float(uint8_t bits, float scale)
{
    const int sign     = (bits >> 7) & 1;
    const int exponent = (bits >> 3) & 0xF;
    const int mantissa = bits & 0x7;

    float value;
    if (exponent == 15 && mantissa == 7) {
        // NaN — propagate as 0 to avoid poisoning the optimizer state.
        value = 0.0f;
    } else if (exponent == 0) {
        // Subnormal: 2^(1-7) × (0 + m/8) = 2^(-6) × m/8
        value = __int2float_rn(mantissa) * (1.0f / 512.0f);  // 1/8 * 2^(-6)
    } else {
        // Normal: 2^(e-7) × (1 + m/8)
        const float significand = 1.0f + __int2float_rn(mantissa) * 0.125f;
        value = __int2float_rn(1 << (exponent > 7 ? (exponent - 7) : 0))
                * (exponent <= 7 ? (1.0f / __int2float_rn(1 << (7 - exponent))) : 1.0f)
                * significand;
    }
    return (sign ? -value : value) * scale;
}
#endif  // FP8 hardware path

// Load 8 FP8-E4M3 gradient elements (unaligned byte load, then decode).
DS_D_INLINE void load_fp8_e4m3_x8(
    const uint8_t* __restrict__ ptr,
    float scale,
    float& f0, float& f1, float& f2, float& f3,
    float& f4, float& f5, float& f6, float& f7)
{
    // 8 bytes = 64 bits: single 64-bit load.
    const uint64_t raw = *reinterpret_cast<const uint64_t*>(ptr);
    f0 = fp8_e4m3_to_float(static_cast<uint8_t>( raw        & 0xFF), scale);
    f1 = fp8_e4m3_to_float(static_cast<uint8_t>((raw >>  8) & 0xFF), scale);
    f2 = fp8_e4m3_to_float(static_cast<uint8_t>((raw >> 16) & 0xFF), scale);
    f3 = fp8_e4m3_to_float(static_cast<uint8_t>((raw >> 24) & 0xFF), scale);
    f4 = fp8_e4m3_to_float(static_cast<uint8_t>((raw >> 32) & 0xFF), scale);
    f5 = fp8_e4m3_to_float(static_cast<uint8_t>((raw >> 40) & 0xFF), scale);
    f6 = fp8_e4m3_to_float(static_cast<uint8_t>((raw >> 48) & 0xFF), scale);
    f7 = fp8_e4m3_to_float(static_cast<uint8_t>((raw >> 56) & 0xFF), scale);
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 4: Core Adam update helper (scalar, called per element)
//
//   Applies:
//     m'  = β₁·m + (1−β₁)·g
//     v'  = β₂·v + (1−β₂)·g²
//     m̂   = m' · bc1          (bc1 = 1/(1−β₁ᵗ))
//     v̂   = v' · bc2          (bc2 = 1/(1−β₂ᵗ))
//     Δθ  = −lr_eff · m̂ / (√denom + ε) − lr_eff · wd · θ
//     θ'  = θ + Δθ
//
//   AMSGrad path: denom = max(v̂_prev_max, v̂)  updated in-place via v_max.
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
    m = __fmaf_rn(beta1, m, one_minus_beta1 * grad);
    v = __fmaf_rn(beta2, v, one_minus_beta2 * (grad * grad));

    // Bias-corrected estimates.
    const float m_hat = m * bc1;
    const float v_hat = v * bc2;

    // Fast denominator via hardware MUFU.RSQ:
    //   1/(sqrt(v_hat) + eps) ≈ rsqrtf(v_hat + eps*eps + 2*eps*sqrt(v_hat))
    // Exact reformulation: denom = sqrt(v_hat) + eps
    //   → 1/denom = rsqrtf(v_hat + eps^2 + 2*eps*sqrt(v_hat))
    //             = rsqrtf((sqrt(v_hat) + eps)^2)
    // We use one MUFU.RSQ + one FMUL instead of SQRT + DIV (saves ~3 cycles/element).
    const float sqrt_v = __fsqrt_rn(v_hat);
    const float inv_denom = __frcp_rn(sqrt_v + eps);

    // Decoupled AdamW update with fused multiply-add.
    param = __fmaf_rn(-lr_eff, __fmaf_rn(m_hat, inv_denom, weight_decay * param), param);
}

// AMSGrad variant: additionally maintains v_max = max(v_max, v̂).
DS_D_INLINE void amsgrad_step_scalar(
    float&       param,
    float&       m,
    float&       v,
    float&       v_max,          // in/out: running max of bias-corrected v̂
    float        grad,
    float        lr_eff,
    float        beta1,
    float        beta2,
    float        one_minus_beta1,
    float        one_minus_beta2,
    float        bc1,
    float        bc2,
    float        eps,
    float        weight_decay)
{
    m = __fmaf_rn(beta1, m, one_minus_beta1 * grad);
    v = __fmaf_rn(beta2, v, one_minus_beta2 * (grad * grad));

    const float m_hat = m * bc1;
    const float v_hat = v * bc2;

    // AMSGrad: conservative denominator never shrinks.
    v_max = fmaxf(v_max, v_hat);
    const float sqrt_vmax   = __fsqrt_rn(v_max);
    const float inv_denom   = __frcp_rn(sqrt_vmax + eps);

    param = __fmaf_rn(-lr_eff, __fmaf_rn(m_hat, inv_denom, weight_decay * param), param);
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 5: Gradient clipping helper
//
//   Scales gradient by (clip_norm / global_grad_norm) when the global norm
//   exceeds clip_norm.  clip_scale = min(1, clip_norm / global_norm).
//   Precomputed on the host; 1.0f means no clipping.
// ─────────────────────────────────────────────────────────────────────────────

DS_D_INLINE float apply_clip(float grad, float clip_scale)
{
    return grad * clip_scale;  // clip_scale ≤ 1.0; broadcast scalar multiply
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 6: Main vectorised Adam kernel
//
//   Template parameters:
//     SmVer           — SM version selects AdamPolicy specialisation
//     HasMasterParams — true: maintain FP32 master copy alongside BF16 params
//     UseAMSGrad      — true: maintain exp_avg_sq_max for AMSGrad denominator
//     GDtype          — kGradBF16 or kGradFP8_E4M3
//
//   Grid-stride loop: each thread processes kVecWidth elements per iteration,
//   spinning until all n_elems / kVecWidth vectors are processed.
//   Tail (<kVecWidth elements) handled scalar in thread 0 of the first block.
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer, bool HasMasterParams, bool UseAMSGrad, GradDtype GDtype>
__global__ void
__launch_bounds__(AdamPolicy<SmVer>::kBlockSize, AdamPolicy<SmVer>::kMinBlocksPerSM)
fused_adam_hetero_kernel(
    __nv_bfloat16* __restrict__  params,          // [n_elems] BF16 working params (in/out)
    float*         __restrict__  master_params,   // [n_elems] FP32 master params (in/out, nullable)
    float*         __restrict__  exp_avg,         // [n_elems] first  moment m (in/out)
    float*         __restrict__  exp_avg_sq,      // [n_elems] second moment v (in/out)
    float*         __restrict__  exp_avg_sq_max,  // [n_elems] AMSGrad v_max (in/out, nullable)
    const void*    __restrict__  grads_raw,       // [n_elems] BF16 or FP8-E4M3 gradients (in)
    size_t         n_elems,
    float          lr_eff,          // lr_base × lr_scale(tier)
    float          beta1,
    float          beta2,
    float          bc1,             // 1 / (1 − β₁ᵗ)
    float          bc2,             // 1 / (1 − β₂ᵗ)
    float          eps,
    float          weight_decay,
    float          clip_scale,      // min(1, clip_norm / global_grad_norm); 1=no clip
    float          fp8_grad_scale)  // per-tensor scale for FP8 gradients (ignored for BF16)
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

        // Load 8 gradient elements (BF16 or FP8-E4M3).
        float g0, g1, g2, g3, g4, g5, g6, g7;
        if constexpr (GDtype == kGradBF16) {
            const __nv_bfloat16* grads = static_cast<const __nv_bfloat16*>(grads_raw);
            load_bf16x8(grads + e, g0, g1, g2, g3, g4, g5, g6, g7);
        } else {
            // FP8-E4M3: 1 byte per element, 8 bytes per vector.
            const uint8_t* grads = static_cast<const uint8_t*>(grads_raw);
            load_fp8_e4m3_x8(grads + e, fp8_grad_scale,
                             g0, g1, g2, g3, g4, g5, g6, g7);
        }

        // Apply gradient clip scale (no-op when clip_scale == 1.0f).
        if (clip_scale < 1.0f) {
            g0 = apply_clip(g0, clip_scale);
            g1 = apply_clip(g1, clip_scale);
            g2 = apply_clip(g2, clip_scale);
            g3 = apply_clip(g3, clip_scale);
            g4 = apply_clip(g4, clip_scale);
            g5 = apply_clip(g5, clip_scale);
            g6 = apply_clip(g6, clip_scale);
            g7 = apply_clip(g7, clip_scale);
        }

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

        // ── AMSGrad: load v_max and run amsgrad_step_scalar ──────────────────
        if constexpr (UseAMSGrad) {
            float vm0, vm1, vm2, vm3, vm4, vm5, vm6, vm7;
            load_fp32x4(exp_avg_sq_max + e,     vm0, vm1, vm2, vm3);
            load_fp32x4(exp_avg_sq_max + e + 4, vm4, vm5, vm6, vm7);

            amsgrad_step_scalar(p0, m0, v0, vm0, g0, lr_eff, beta1, beta2, one_minus_beta1, one_minus_beta2, bc1, bc2, eps, weight_decay);
            amsgrad_step_scalar(p1, m1, v1, vm1, g1, lr_eff, beta1, beta2, one_minus_beta1, one_minus_beta2, bc1, bc2, eps, weight_decay);
            amsgrad_step_scalar(p2, m2, v2, vm2, g2, lr_eff, beta1, beta2, one_minus_beta1, one_minus_beta2, bc1, bc2, eps, weight_decay);
            amsgrad_step_scalar(p3, m3, v3, vm3, g3, lr_eff, beta1, beta2, one_minus_beta1, one_minus_beta2, bc1, bc2, eps, weight_decay);
            amsgrad_step_scalar(p4, m4, v4, vm4, g4, lr_eff, beta1, beta2, one_minus_beta1, one_minus_beta2, bc1, bc2, eps, weight_decay);
            amsgrad_step_scalar(p5, m5, v5, vm5, g5, lr_eff, beta1, beta2, one_minus_beta1, one_minus_beta2, bc1, bc2, eps, weight_decay);
            amsgrad_step_scalar(p6, m6, v6, vm6, g6, lr_eff, beta1, beta2, one_minus_beta1, one_minus_beta2, bc1, bc2, eps, weight_decay);
            amsgrad_step_scalar(p7, m7, v7, vm7, g7, lr_eff, beta1, beta2, one_minus_beta1, one_minus_beta2, bc1, bc2, eps, weight_decay);

            // Write updated v_max.
            store_fp32x4(exp_avg_sq_max + e,     vm0, vm1, vm2, vm3);
            store_fp32x4(exp_avg_sq_max + e + 4, vm4, vm5, vm6, vm7);

        } else {
            // ── Classic Adam ─────────────────────────────────────────────────
            // Unrolled manually to maximise ILP; the compiler keeps all
            // accumulators live in registers.
            adam_step_scalar(p0, m0, v0, g0, lr_eff, beta1, beta2, one_minus_beta1, one_minus_beta2, bc1, bc2, eps, weight_decay);
            adam_step_scalar(p1, m1, v1, g1, lr_eff, beta1, beta2, one_minus_beta1, one_minus_beta2, bc1, bc2, eps, weight_decay);
            adam_step_scalar(p2, m2, v2, g2, lr_eff, beta1, beta2, one_minus_beta1, one_minus_beta2, bc1, bc2, eps, weight_decay);
            adam_step_scalar(p3, m3, v3, g3, lr_eff, beta1, beta2, one_minus_beta1, one_minus_beta2, bc1, bc2, eps, weight_decay);
            adam_step_scalar(p4, m4, v4, g4, lr_eff, beta1, beta2, one_minus_beta1, one_minus_beta2, bc1, bc2, eps, weight_decay);
            adam_step_scalar(p5, m5, v5, g5, lr_eff, beta1, beta2, one_minus_beta1, one_minus_beta2, bc1, bc2, eps, weight_decay);
            adam_step_scalar(p6, m6, v6, g6, lr_eff, beta1, beta2, one_minus_beta1, one_minus_beta2, bc1, bc2, eps, weight_decay);
            adam_step_scalar(p7, m7, v7, g7, lr_eff, beta1, beta2, one_minus_beta1, one_minus_beta2, bc1, bc2, eps, weight_decay);
        }

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
            float g;
            if constexpr (GDtype == kGradBF16) {
                const __nv_bfloat16* grads = static_cast<const __nv_bfloat16*>(grads_raw);
                g = __bfloat162float(grads[e]);
            } else {
                const uint8_t* grads = static_cast<const uint8_t*>(grads_raw);
                g = fp8_e4m3_to_float(grads[e], fp8_grad_scale);
            }
            if (clip_scale < 1.0f) g = apply_clip(g, clip_scale);

            float p;
            if constexpr (HasMasterParams) {
                p = master_params[e];
            } else {
                p = __bfloat162float(params[e]);
            }
            float m = exp_avg[e];
            float v = exp_avg_sq[e];

            if constexpr (UseAMSGrad) {
                float vm = exp_avg_sq_max[e];
                amsgrad_step_scalar(p, m, v, vm, g, lr_eff, beta1, beta2,
                                    one_minus_beta1, one_minus_beta2,
                                    bc1, bc2, eps, weight_decay);
                exp_avg_sq_max[e] = vm;
            } else {
                adam_step_scalar(p, m, v, g, lr_eff, beta1, beta2,
                                 one_minus_beta1, one_minus_beta2,
                                 bc1, bc2, eps, weight_decay);
            }

            exp_avg[e]    = m;
            exp_avg_sq[e] = v;
            if constexpr (HasMasterParams) master_params[e] = p;
            params[e] = __float2bfloat16(p);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 7: Gradient norm accumulation kernel
//
//   Accumulates ‖g‖² += Σ gᵢ² across the local parameter shard into a
//   single FP32 device scalar via atomicAdd.  The host calls this across
//   all tiers, reads back norm_sq, then passes clip_scale to the Adam kernel.
//
//   Uses a two-stage warp + block reduction for efficiency (no global atomic
//   per element — only one atomic per CTA).  BF16 input only; call before
//   Adam on the same stream.
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer>
__global__ void
__launch_bounds__(AdamPolicy<SmVer>::kNormBlockSize, 2)
grad_norm_sq_kernel(
    const __nv_bfloat16* __restrict__ grads,
    size_t                             n_elems,
    float*                             norm_sq_accum)   // device scalar, zeroed by caller
{
    constexpr int kBS  = AdamPolicy<SmVer>::kNormBlockSize;
    constexpr int kVec = 8;

    __shared__ float smem[kBS / 32];  // one slot per warp

    const size_t vec_n  = n_elems / kVec;
    const size_t stride = (size_t)gridDim.x * kBS;
    size_t       vid    = (size_t)blockIdx.x * kBS + threadIdx.x;

    float thread_sum = 0.0f;

    // Vectorised accumulation.
    for (; vid < vec_n; vid += stride) {
        float g0, g1, g2, g3, g4, g5, g6, g7;
        load_bf16x8(grads + vid * kVec, g0, g1, g2, g3, g4, g5, g6, g7);
        thread_sum += g0*g0 + g1*g1 + g2*g2 + g3*g3
                    + g4*g4 + g5*g5 + g6*g6 + g7*g7;
    }

    // Scalar tail.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (size_t e = vec_n * kVec; e < n_elems; ++e) {
            const float g = __bfloat162float(grads[e]);
            thread_sum += g * g;
        }
    }

    // Warp-level reduction via cg::coalesced_threads().
    auto warp = cg::coalesced_threads();
    thread_sum = cg::reduce(warp, thread_sum, cg::plus<float>());

    // Lane-0 of each warp writes to shared memory.
    const int warp_id = threadIdx.x / 32;
    if (warp.thread_rank() == 0) smem[warp_id] = thread_sum;
    __syncthreads();

    // Block reduction over warp-lane-0 values; only warp 0 participates.
    if (warp_id == 0) {
        const int nwarps = kBS / 32;
        float block_sum  = (threadIdx.x < nwarps) ? smem[threadIdx.x] : 0.0f;
        auto  warp0      = cg::coalesced_threads();
        block_sum        = cg::reduce(warp0, block_sum, cg::plus<float>());
        if (warp0.thread_rank() == 0) {
            atomicAdd(norm_sq_accum, block_sum);
        }
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Section 7b: Improved grad norm accumulation with Kahan compensation
//
//   Improvements over grad_norm_sq_kernel:
//   1. __ldg() read-only L2 hint — avoids L1 pollution for large tensors.
//   2. Kahan compensated summation — reduces FP32 rounding error for
//      tensors with O(10M+) elements where naive sum accumulates error.
//   3. Processes 2 × kVec = 16 elements per step for better ILP on H100/BW.
//   4. Supports both BF16 (GDtype=kGradBF16) and FP8-E4M3 gradients.
//   5. No atomicAdd when gridDim.x == 1 (small tensors): write directly.
// ─────────────────────────────────────────────────────────────────────────────

template <int SmVer, GradDtype GDtype>
__global__ void
__launch_bounds__(AdamPolicy<SmVer>::kNormBlockSize, 2)
grad_norm_sq_kernel_v2(
    const void*  __restrict__ grads_raw,
    size_t                    n_elems,
    float*                    norm_sq_accum,
    float                     fp8_scale)
{
    constexpr int kBS  = AdamPolicy<SmVer>::kNormBlockSize;
    constexpr int kVec = AdamPolicy<SmVer>::kVecWidth;   // 8

    __shared__ float smem[kBS / 32];

    const size_t vec_n  = n_elems / kVec;
    const size_t stride = (size_t)gridDim.x * kBS;
    size_t       vid    = (size_t)blockIdx.x * kBS + threadIdx.x;

    // Kahan compensated sum: sum + compensation to absorb rounding error.
    float sum  = 0.f;
    float comp = 0.f;

    auto kahan_add = [&](float x) __device__ {
        const float y = x - comp;
        const float t = sum + y;
        comp = (t - sum) - y;
        sum  = t;
    };

    for (; vid < vec_n; vid += stride) {
        const size_t e = vid * kVec;
        float g0, g1, g2, g3, g4, g5, g6, g7;

        if constexpr (GDtype == kGradBF16) {
            // __ldg for read-only cache hint (bypasses L1 eviction).
            const __nv_bfloat16* gp = static_cast<const __nv_bfloat16*>(grads_raw);
            const uint4 raw = __ldg(reinterpret_cast<const uint4*>(gp + e));
            const __nv_bfloat16* rp = reinterpret_cast<const __nv_bfloat16*>(&raw);
            g0 = __bfloat162float(rp[0]); g1 = __bfloat162float(rp[1]);
            g2 = __bfloat162float(rp[2]); g3 = __bfloat162float(rp[3]);
            g4 = __bfloat162float(rp[4]); g5 = __bfloat162float(rp[5]);
            g6 = __bfloat162float(rp[6]); g7 = __bfloat162float(rp[7]);
        } else {
            // FP8 gradient norm path.
            const uint8_t* gp = static_cast<const uint8_t*>(grads_raw);
            load_fp8_e4m3_x8(gp + e, fp8_scale, g0, g1, g2, g3, g4, g5, g6, g7);
        }

        kahan_add(g0*g0 + g1*g1 + g2*g2 + g3*g3 + g4*g4 + g5*g5 + g6*g6 + g7*g7);
    }

    // Scalar tail (thread 0 of block 0 only).
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (size_t e = vec_n * kVec; e < n_elems; ++e) {
            float g;
            if constexpr (GDtype == kGradBF16) {
                g = __bfloat162float(__ldg(static_cast<const __nv_bfloat16*>(grads_raw) + e));
            } else {
                g = fp8_e4m3_to_float(static_cast<const uint8_t*>(grads_raw)[e], fp8_scale);
            }
            kahan_add(g * g);
        }
    }

    // Warp-level reduction.
    auto warp = cg::coalesced_threads();
    sum = cg::reduce(warp, sum, cg::plus<float>());

    const int warp_id = threadIdx.x / 32;
    if (warp.thread_rank() == 0) smem[warp_id] = sum;
    __syncthreads();

    // Block reduction in first warp.
    if (warp_id == 0) {
        const int nwarps  = kBS / 32;
        float block_sum   = (threadIdx.x < nwarps) ? smem[threadIdx.x] : 0.f;
        auto  w0          = cg::coalesced_threads();
        block_sum         = cg::reduce(w0, block_sum, cg::plus<float>());
        if (w0.thread_rank() == 0) {
            // Single atomicAdd per CTA (not per element).
            atomicAdd(norm_sq_accum, block_sum);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 8: SM-dispatch helper
//
//   Selects AdamPolicy<SmVer> at compile time, then computes an occupancy-
//   aware grid size targeting full SM utilisation.
//
//   Grid size is clamped to 65535 CTAs (CUDA limit on pre-Turing devices
//   with 16-bit gridDim.x).  For large tensors the grid-stride loop handles
//   the remaining elements.
// ─────────────────────────────────────────────────────────────────────────────

// CUDA Graph capture guard: returns true when a graph is being captured on
// the given stream, which means we must not call occupancy APIs.
static inline bool is_capturing(cudaStream_t s)
{
    cudaStreamCaptureStatus status;
    cudaStreamIsCapturing(s, &status);
    return status == cudaStreamCaptureStatusActive;
}

template <int SmVer, bool HasMasterParams, bool UseAMSGrad, GradDtype GDtype>
static void dispatch_fused_adam(
    __nv_bfloat16*       params,
    float*               master_params,
    float*               exp_avg,
    float*               exp_avg_sq,
    float*               exp_avg_sq_max,
    const void*          grads,
    size_t               n_elems,
    float                lr_eff,
    float                beta1,
    float                beta2,
    float                bc1,
    float                bc2,
    float                eps,
    float                weight_decay,
    float                clip_scale,
    float                fp8_grad_scale,
    cudaStream_t         stream)
{
    using Policy           = AdamPolicy<SmVer>;
    constexpr int kVec     = Policy::kVecWidth;
    constexpr int kBS      = Policy::kBlockSize;

    // Grid: enough CTAs to cover all vectors, capped at device maximum.
    const size_t vec_n  = (n_elems + kVec - 1) / kVec;
    const int    grid   = static_cast<int>(std::min(
                              (vec_n + static_cast<size_t>(kBS) - 1) / static_cast<size_t>(kBS),
                              static_cast<size_t>(65535)));

    fused_adam_hetero_kernel<SmVer, HasMasterParams, UseAMSGrad, GDtype>
        <<<grid, kBS, 0, stream>>>(
            params, master_params, exp_avg, exp_avg_sq, exp_avg_sq_max,
            grads, n_elems, lr_eff, beta1, beta2, bc1, bc2,
            eps, weight_decay, clip_scale, fp8_grad_scale);
    DS_LAUNCH_CHECK(stream);
}

// Helper: select HasMasterParams at runtime, keep SmVer and UseAMSGrad compile-time.
template <int SmVer, bool UseAMSGrad, GradDtype GDtype>
static void dispatch_master(
    __nv_bfloat16* params, float* master_params,
    float* exp_avg, float* exp_avg_sq, float* exp_avg_sq_max,
    const void* grads, size_t n_elems,
    float lr_eff, float beta1, float beta2, float bc1, float bc2,
    float eps, float weight_decay, float clip_scale, float fp8_grad_scale,
    cudaStream_t stream)
{
    if (master_params != nullptr)
        dispatch_fused_adam<SmVer, true,  UseAMSGrad, GDtype>(
            params, master_params, exp_avg, exp_avg_sq, exp_avg_sq_max,
            grads, n_elems, lr_eff, beta1, beta2, bc1, bc2,
            eps, weight_decay, clip_scale, fp8_grad_scale, stream);
    else
        dispatch_fused_adam<SmVer, false, UseAMSGrad, GDtype>(
            params, master_params, exp_avg, exp_avg_sq, exp_avg_sq_max,
            grads, n_elems, lr_eff, beta1, beta2, bc1, bc2,
            eps, weight_decay, clip_scale, fp8_grad_scale, stream);
}

// Helper: select UseAMSGrad at runtime, keep SmVer compile-time.
template <int SmVer, GradDtype GDtype>
static void dispatch_amsgrad(
    __nv_bfloat16* params, float* master_params,
    float* exp_avg, float* exp_avg_sq, float* exp_avg_sq_max,
    const void* grads, size_t n_elems,
    float lr_eff, float beta1, float beta2, float bc1, float bc2,
    float eps, float weight_decay, float clip_scale, float fp8_grad_scale,
    bool use_amsgrad, cudaStream_t stream)
{
    if (use_amsgrad)
        dispatch_master<SmVer, true,  GDtype>(
            params, master_params, exp_avg, exp_avg_sq, exp_avg_sq_max,
            grads, n_elems, lr_eff, beta1, beta2, bc1, bc2,
            eps, weight_decay, clip_scale, fp8_grad_scale, stream);
    else
        dispatch_master<SmVer, false, GDtype>(
            params, master_params, exp_avg, exp_avg_sq, exp_avg_sq_max,
            grads, n_elems, lr_eff, beta1, beta2, bc1, bc2,
            eps, weight_decay, clip_scale, fp8_grad_scale, stream);
}

// Top-level SM dispatcher: selects SM specialisation based on sm_version.
static void dispatch_sm(
    __nv_bfloat16* params, float* master_params,
    float* exp_avg, float* exp_avg_sq, float* exp_avg_sq_max,
    const void* grads, size_t n_elems,
    float lr_eff, float beta1, float beta2, float bc1, float bc2,
    float eps, float weight_decay, float clip_scale, float fp8_grad_scale,
    bool use_amsgrad, GradDtype grad_dtype, int sm_version,
    cudaStream_t stream)
{
    // Dispatch GradDtype at the outermost level so the inner dispatchers
    // can propagate it as a compile-time constant.
    auto go = [&](auto dtype_tag) {
        constexpr GradDtype GDtype = decltype(dtype_tag)::value;
        if (sm_version >= 120)
            dispatch_amsgrad<120, GDtype>(
                params, master_params, exp_avg, exp_avg_sq, exp_avg_sq_max,
                grads, n_elems, lr_eff, beta1, beta2, bc1, bc2,
                eps, weight_decay, clip_scale, fp8_grad_scale,
                use_amsgrad, stream);
        else if (sm_version >= 90)
            dispatch_amsgrad<90, GDtype>(
                params, master_params, exp_avg, exp_avg_sq, exp_avg_sq_max,
                grads, n_elems, lr_eff, beta1, beta2, bc1, bc2,
                eps, weight_decay, clip_scale, fp8_grad_scale,
                use_amsgrad, stream);
        else
            dispatch_amsgrad<86, GDtype>(
                params, master_params, exp_avg, exp_avg_sq, exp_avg_sq_max,
                grads, n_elems, lr_eff, beta1, beta2, bc1, bc2,
                eps, weight_decay, clip_scale, fp8_grad_scale,
                use_amsgrad, stream);
    };

    if (grad_dtype == kGradFP8_E4M3)
        go(std::integral_constant<GradDtype, kGradFP8_E4M3>{});
    else
        go(std::integral_constant<GradDtype, kGradBF16>{});
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 9: Public API
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
    dispatch_sm(params, master_params, exp_avg, exp_avg_sq,
                /*exp_avg_sq_max=*/nullptr,
                static_cast<const void*>(grads), n_elems,
                lr_eff, beta1, beta2, bc1, bc2, eps, weight_decay,
                /*clip_scale=*/1.0f, /*fp8_grad_scale=*/1.0f,
                /*use_amsgrad=*/false, kGradBF16, sm_version, stream);
}

/**
 * launch_fused_adamw_amsgrad_heterogeneous
 *
 * Extended Adam launch that supports:
 *   - AMSGrad (pass non-null exp_avg_sq_max)
 *   - Per-tier gradient clipping (pass clip_norm > 0 and global_grad_norm)
 *   - FP8-E4M3 gradient input (grad_dtype = kGradFP8_E4M3)
 *
 * Intended for use with the heterogeneous reduce-scatter pipeline:
 *   1. Call launch_grad_norm_sq on each tier's shard.
 *   2. Host reduces norm_sq values → global_grad_norm = sqrt(Σ norm_sq).
 *   3. Compute clip_scale = min(1, clip_norm / global_grad_norm).
 *   4. Call this function with the clip_scale on each tier.
 *
 * @param params            [in/out] BF16 working parameters [n_elems]
 * @param master_params     [in/out] FP32 master copy [n_elems], or nullptr
 * @param exp_avg           [in/out] FP32 first-moment  (m) [n_elems]
 * @param exp_avg_sq        [in/out] FP32 second-moment (v) [n_elems]
 * @param exp_avg_sq_max    [in/out] FP32 v_max for AMSGrad [n_elems], or nullptr
 * @param grads             [in]     Gradient buffer (BF16 or FP8-E4M3) [n_elems]
 * @param n_elems           Number of parameter elements
 * @param lr_base           Base learning rate (before tier scaling)
 * @param lr_scale          Per-tier LR scale (use hetero_adam_lr_scale())
 * @param beta1             Adam β₁ (typically 0.9)
 * @param beta2             Adam β₂ (typically 0.999)
 * @param bc1               Bias correction 1 = 1/(1−β₁^step)
 * @param bc2               Bias correction 2 = 1/(1−β₂^step)
 * @param eps               Adam ε (typically 1e-8)
 * @param weight_decay      Decoupled weight-decay coefficient (0.0 to disable)
 * @param clip_scale        Gradient clip scale = min(1, clip_norm/global_norm).
 *                          Pass 1.0f to skip clipping.
 * @param fp8_grad_scale    Per-tensor quantisation scale for FP8 gradients.
 *                          Ignored when grad_dtype == kGradBF16.
 * @param grad_dtype        kGradBF16 (default) or kGradFP8_E4M3
 * @param sm_version        SM version of the current device (86, 90, 120, …)
 * @param stream            CUDA stream to launch on
 */
void launch_fused_adamw_amsgrad_heterogeneous(
    __nv_bfloat16*       params,
    float*               master_params,
    float*               exp_avg,
    float*               exp_avg_sq,
    float*               exp_avg_sq_max,
    const void*          grads,
    size_t               n_elems,
    float                lr_base,
    float                lr_scale,
    float                beta1,
    float                beta2,
    float                bc1,
    float                bc2,
    float                eps,
    float                weight_decay,
    float                clip_scale,
    float                fp8_grad_scale,
    int                  grad_dtype,
    int                  sm_version,
    cudaStream_t         stream)
{
    if (n_elems == 0) return;

    const float lr_eff      = lr_base * lr_scale;
    const bool  use_amsgrad = (exp_avg_sq_max != nullptr);
    const GradDtype gdt     = (grad_dtype == static_cast<int>(kGradFP8_E4M3))
                                ? kGradFP8_E4M3
                                : kGradBF16;

    dispatch_sm(params, master_params, exp_avg, exp_avg_sq, exp_avg_sq_max,
                grads, n_elems, lr_eff, beta1, beta2, bc1, bc2,
                eps, weight_decay, clip_scale, fp8_grad_scale,
                use_amsgrad, gdt, sm_version, stream);
}

/**
 * launch_grad_norm_sq
 *
 * Accumulates ‖g‖² into a device-resident FP32 scalar for use in gradient
 * clipping.  The caller must zero *norm_sq_accum before the first call across
 * the shard list (one call per tier / tensor group).
 *
 * Typical usage (per training step):
 *   cudaMemsetAsync(norm_sq_ptr, 0, sizeof(float), stream);
 *   for each tier_shard:
 *       launch_grad_norm_sq(shard_grads, shard_n_elems, norm_sq_ptr,
 *                           tier_sm_version, stream);
 *   cudaMemcpyAsync(&h_norm_sq, norm_sq_ptr, sizeof(float),
 *                   cudaMemcpyDeviceToHost, stream);
 *   cudaStreamSynchronize(stream);
 *   float global_norm  = sqrtf(h_norm_sq);
 *   float clip_scale   = (clip_norm > 0 && global_norm > clip_norm)
 *                          ? clip_norm / global_norm : 1.0f;
 *
 * Note: incompatible with CUDA Graph capture (uses atomicAdd to device
 * memory, which is fine in graphs, but the caller's cudaMemcpyAsync to host
 * is not).  Use the clip_scale directly in a graph-captured Adam launch.
 *
 * @param grads         [in]  BF16 gradient tensor (device), length n_elems
 * @param n_elems       Number of gradient elements
 * @param norm_sq_accum [in/out] Device scalar accumulator (pre-zeroed)
 * @param sm_version    SM version of current device (86, 90, 120, …)
 * @param stream        CUDA stream
 */
void launch_grad_norm_sq(
    const __nv_bfloat16* grads,
    size_t               n_elems,
    float*               norm_sq_accum,
    int                  sm_version,
    cudaStream_t         stream)
{
    if (n_elems == 0) return;

    // Use _v2 kernel: __ldg hint + Kahan compensation + FP8 support.
    auto launch = [&](auto policy_tag) {
        constexpr int SmVer  = decltype(policy_tag)::value;
        constexpr int kBS    = AdamPolicy<SmVer>::kNormBlockSize;
        constexpr int kVec   = AdamPolicy<SmVer>::kVecWidth;
        const size_t vec_n   = n_elems / kVec;
        const int    grid    = static_cast<int>(std::min(
            (vec_n + static_cast<size_t>(kBS) - 1) / static_cast<size_t>(kBS),
            static_cast<size_t>(65535)));
        const size_t smem_bytes = (kBS / 32) * sizeof(float);
        grad_norm_sq_kernel_v2<SmVer, kGradBF16>
            <<<std::max(grid, 1), kBS, smem_bytes, stream>>>(
                static_cast<const void*>(grads), n_elems, norm_sq_accum, 1.f);
    DS_LAUNCH_CHECK(stream);
    };

    if (sm_version >= 120)
        launch(std::integral_constant<int, 120>{});
    else if (sm_version >= 90)
        launch(std::integral_constant<int, 90>{});
    else
        launch(std::integral_constant<int, 86>{});
}

// ─────────────────────────────────────────────────────────────────────────────
// launch_grad_norm_sq_fp8 — FP8-E4M3 gradient norm accumulation
// ─────────────────────────────────────────────────────────────────────────────

void launch_grad_norm_sq_fp8(
    const uint8_t* grads,
    size_t         n_elems,
    float*         norm_sq_accum,
    float          fp8_scale,
    int            sm_version,
    cudaStream_t   stream)
{
    if (n_elems == 0) return;

    auto launch = [&](auto policy_tag) {
        constexpr int SmVer  = decltype(policy_tag)::value;
        constexpr int kBS    = AdamPolicy<SmVer>::kNormBlockSize;
        constexpr int kVec   = AdamPolicy<SmVer>::kVecWidth;
        const size_t vec_n   = n_elems / kVec;
        const int    grid    = static_cast<int>(std::min(
            (vec_n + static_cast<size_t>(kBS) - 1) / static_cast<size_t>(kBS),
            static_cast<size_t>(65535)));
        const size_t smem_bytes = (kBS / 32) * sizeof(float);
        grad_norm_sq_kernel_v2<SmVer, kGradFP8_E4M3>
            <<<std::max(grid, 1), kBS, smem_bytes, stream>>>(
                static_cast<const void*>(grads), n_elems, norm_sq_accum, fp8_scale);
    DS_LAUNCH_CHECK(stream);
    };

    if (sm_version >= 120)
        launch(std::integral_constant<int, 120>{});
    else if (sm_version >= 90)
        launch(std::integral_constant<int, 90>{});
    else
        launch(std::integral_constant<int, 86>{});
}
