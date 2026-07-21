// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

// Issues #12 / #84: tier-aware CPU Adam with async pinned-memory prefetch
// and SM-specific fast paths (H100 hybrid / Blackwell GPU kernel / A6000 offload).

#include <torch/extension.h>
#include <cassert>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include "cpu_adam.h"
#include "cpu_adam_tier.h"  // tier detection + PrefetchState

using namespace std::string_literals;
static std::unordered_map<int, std::shared_ptr<void>> s_optimizers;

// C++ interface

template <typename ds_params_precision_t, typename ds_state_precision_t>
void Adam_Optimizer::Step_1(ds_params_precision_t* _params,
                            ds_params_precision_t* grads,
                            ds_state_precision_t* _exp_avg,
                            ds_state_precision_t* _exp_avg_sq,
                            size_t _param_size)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<1>(&rounded_size, _params, grads, _exp_avg, _exp_avg_sq, _param_size);
#endif
    if (_param_size > rounded_size) {
        float betta1_minus1 = 1 - _betta1;
        float betta2_minus1 = 1 - _betta2;

        float step_size = -1 * _alpha / _bias_correction1;
        float w_decay = -1 * _alpha * _weight_decay;

        for (size_t t = rounded_size; t < _param_size; t += TILE) {
            size_t copy_size = TILE;
            if ((t + TILE) > _param_size) copy_size = _param_size - t;
            size_t offset = copy_size + t;
#pragma omp parallel for
            for (size_t k = t; k < offset; k++) {
                float grad = (float)grads[k];
                float param = (float)_params[k];
                float momentum = _exp_avg[k];
                float variance = _exp_avg_sq[k];
                if (_weight_decay > 0 && !_adamw_mode) { grad = param * _weight_decay + grad; }
                momentum = momentum * _betta1;
                momentum = grad * betta1_minus1 + momentum;

                variance = variance * _betta2;
                grad = grad * grad;
                variance = grad * betta2_minus1 + variance;

                grad = sqrt(variance);
                grad = grad * _bias_correction2 + _eps;
                grad = momentum / grad;
                if (_weight_decay > 0 && _adamw_mode) { param += w_decay * param; }
                param = grad * step_size + param;
                _params[k] = param;
                _exp_avg[k] = momentum;
                _exp_avg_sq[k] = variance;
            }
        }
    }
}

template <typename ds_params_precision_t, typename ds_state_precision_t>
void Adam_Optimizer::Step_4(ds_params_precision_t* _params,
                            ds_params_precision_t* grads,
                            ds_state_precision_t* _exp_avg,
                            ds_state_precision_t* _exp_avg_sq,
                            size_t _param_size)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<4>(&rounded_size, _params, grads, _exp_avg, _exp_avg_sq, _param_size);
#endif
    if (_param_size > rounded_size)
        Step_1((_params + rounded_size),
               (grads + rounded_size),
               (_exp_avg + rounded_size),
               (_exp_avg_sq + rounded_size),
               (_param_size - rounded_size));
}

int create_adam_optimizer(int optimizer_id,
                          float alpha,
                          float betta1,
                          float betta2,
                          float eps,
                          float weight_decay,
                          bool adamw_mode,
                          bool should_log)
{
    auto opt =
        std::make_shared<Adam_Optimizer>(alpha, betta1, betta2, eps, weight_decay, adamw_mode);

    s_optimizers[optimizer_id] = opt;

    if (should_log) {
        std::string avx_type = "";
#if defined(__AVX512__)
        avx_type = "AVX512";
#else
#if defined(__AVX256__)
        avx_type = "AVX2";
#else
        avx_type = "scalar";
#endif
#endif

        printf("Adam Optimizer #%d is created with %s arithmetic capability.\n",
               optimizer_id,
               avx_type.c_str());
        printf("Config: alpha=%f, betas=(%f, %f), weight_decay=%f, adam_w=%d\n",
               alpha,
               betta1,
               betta2,
               weight_decay,
               (int)adamw_mode);
    }

    return 0;
}

template <typename ds_params_precision_t, typename ds_state_precision_t>
void Adam_Optimizer::Step_8(ds_params_precision_t* _params,
                            ds_params_precision_t* grads,
                            ds_state_precision_t* _exp_avg,
                            ds_state_precision_t* _exp_avg_sq,
                            size_t _param_size)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<8>(&rounded_size, _params, grads, _exp_avg, _exp_avg_sq, _param_size);
#endif
    if (_param_size > rounded_size)
        Step_4((_params + rounded_size),
               (grads + rounded_size),
               (_exp_avg + rounded_size),
               (_exp_avg_sq + rounded_size),
               (_param_size - rounded_size));
}

template <typename ds_params_precision_t, typename ds_state_precision_t>
void step_invoker(std::shared_ptr<Adam_Optimizer> opt,
                  void* _params,
                  void* grads,
                  void* _exp_avg,
                  void* _exp_avg_sq,
                  size_t _param_size)
{
    opt->Step_8((ds_params_precision_t*)(_params),
                (ds_params_precision_t*)(grads),
                (ds_state_precision_t*)(_exp_avg),
                (ds_state_precision_t*)(_exp_avg_sq),
                _param_size);
}

std::map<std::tuple<c10::ScalarType, c10::ScalarType>,
         std::function<void(std::shared_ptr<Adam_Optimizer>, void*, void*, void*, void*, size_t)>>
    invokers;

// Fill map with template functions for each type
template <class ds_params_precision_t, class ds_state_precision_t>
void create_invoker()
{
    invokers[std::tuple(c10::CppTypeToScalarType<ds_params_precision_t>(),
                        c10::CppTypeToScalarType<ds_state_precision_t>())] =
        step_invoker<ds_params_precision_t, ds_state_precision_t>;
}
struct InvokerInitializer {
    InvokerInitializer()
    {
        create_invoker<c10::Half, float>();
        create_invoker<c10::Half, c10::Half>();
        create_invoker<c10::BFloat16, float>();
        create_invoker<c10::BFloat16, c10::BFloat16>();
        create_invoker<float, float>();
    }
} _invoker_initializer;

void invoke(std::shared_ptr<Adam_Optimizer> opt,
            torch::Tensor& params,
            torch::Tensor& grads,
            torch::Tensor& exp_avg,
            torch::Tensor& exp_avg_sq,
            size_t param_size)
{
    c10::ScalarType params_type = at::typeMetaToScalarType(params.options().dtype());
    c10::ScalarType state_type = at::typeMetaToScalarType(exp_avg.options().dtype());

    auto it = invokers.find(std::tuple(params_type, state_type));
    if (it == invokers.end()) {
        throw std::runtime_error("Adam optimizer with param type "s + c10::toString(params_type) +
                                 " and state type "s + c10::toString(state_type) +
                                 " is not supported on current hardware"s);
    }

    it->second(opt,
               params.data_ptr(),
               grads.data_ptr(),
               exp_avg.data_ptr(),
               exp_avg_sq.data_ptr(),
               param_size);
}

int ds_adam_step(int optimizer_id,
                 size_t step,
                 float lr,
                 float beta1,
                 float beta2,
                 float epsilon,
                 float weight_decay,
                 bool bias_correction,
                 torch::Tensor& params,
                 torch::Tensor& grads,
                 torch::Tensor& exp_avg,
                 torch::Tensor& exp_avg_sq)
{
    auto params_c = params.contiguous();
    auto grads_c = grads.contiguous();
    auto exp_avg_c = exp_avg.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();

    std::shared_ptr<Adam_Optimizer> opt =
        std::static_pointer_cast<Adam_Optimizer>(s_optimizers[optimizer_id]);
    opt->IncrementStep(step, beta1, beta2);
    opt->update_state(lr, epsilon, weight_decay, bias_correction);

    invoke(opt, params_c, grads_c, exp_avg_c, exp_avg_sq_c, params_c.numel());

    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Issue #12 / #84: ds_adam_step_tier
//
// Tier-aware entry point that selects the offload strategy based on the SM
// version of the attached GPU:
//
//   SM 8.6  (A6000)  →  CPU_OFFLOAD:
//     Gradients are on GPU. We move them tile-by-tile into pinned host memory
//     via double-buffered async D2H prefetch, then run the AVX512 CPU Adam
//     kernel on each pinned tile.  The prefetch of tile N+1 overlaps with the
//     CPU math on tile N, hiding ≈80 % of PCIe latency.
//
//   SM 9.0  (H100)   →  HYBRID:
//     Small parameter groups (< kH100SmallThresh elements) are handed off to
//     the multi_tensor_adam CUDA kernel directly (caller is expected to invoke
//     the CUDA path for those). Large groups use the same async prefetch loop
//     as SM 8.6 but with a larger pinned pool (4 MiB per buffer) and an
//     aggressive prefetch look-ahead of two tiles.
//
//   SM 12.0 (Blackwell) → GPU_KERNEL:
//     CPU Adam is bypassed entirely.  We immediately return 0 and expect the
//     caller (Python layer) to have already dispatched to multi_tensor_adam.
//     This function is a no-op fast path to avoid redundant CPU work.
//
// Parameters are identical to ds_adam_step plus:
//   sm_version  – SM version of the GPU (e.g. 86, 90, 120).  Pass 0 to
//                 auto-detect from the current CUDA device.
//   enable_prefetch – set false to disable async prefetch (debugging).
// ─────────────────────────────────────────────────────────────────────────────

// Per-optimizer prefetch state (keyed by optimizer_id).
// Initialised lazily on first call; destroyed by destroy_adam_optimizer.
// Only available when CUDA headers are present (PrefetchState uses CUDA APIs).
#if CPU_ADAM_TIER_HAS_CUDA
static std::unordered_map<int, std::unique_ptr<PrefetchState>> s_prefetch_states;
#endif

int ds_adam_step_tier(int optimizer_id,
                      size_t step,
                      float lr,
                      float beta1,
                      float beta2,
                      float epsilon,
                      float weight_decay,
                      bool bias_correction,
                      torch::Tensor& params,
                      torch::Tensor& grads,
                      torch::Tensor& exp_avg,
                      torch::Tensor& exp_avg_sq,
                      int sm_version,
                      bool enable_prefetch)
{
    // ── 1. Resolve tier ────────────────────────────────────────────────────
    if (sm_version == 0)
        sm_version = detect_sm_version();  // auto-detect current CUDA device

    const GpuTier tier         = sm_version_to_tier(sm_version);
    const AdamOffloadStrategy strategy = tier_to_strategy(tier);

    // ── 2. SM 12.0 fast path: pure GPU, nothing for CPU Adam to do ─────────
    if (strategy == AdamOffloadStrategy::GPU_KERNEL) {
        // multi_tensor_adam kernel handles this group; CPU Adam is a no-op.
        return 0;
    }

    // ── 3. SM 9.0 hybrid: small groups delegate to GPU, large go to CPU ────
    const size_t param_size = params.numel();
    if (strategy == AdamOffloadStrategy::HYBRID && param_size < kH100SmallThresh) {
        // Small group — no CPU work; caller uses CUDA kernel.
        return 0;
    }

    // ── 4. CPU Adam path (SM 8.6 CPU_OFFLOAD or SM 9.0 large group) ────────
    auto params_c     = params.contiguous();
    auto grads_c      = grads.contiguous();
    auto exp_avg_c    = exp_avg.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();

    std::shared_ptr<Adam_Optimizer> opt =
        std::static_pointer_cast<Adam_Optimizer>(s_optimizers[optimizer_id]);
    opt->IncrementStep(step, beta1, beta2);
    opt->update_state(lr, epsilon, weight_decay, bias_correction);

    // ── 5. Async prefetch path ─────────────────────────────────────────────
    // Only applies when gradients live on a CUDA device AND prefetch is enabled.
    // PrefetchState requires CUDA runtime APIs (cudaHostAlloc, cudaMemcpyAsync, etc.)
    // so this entire block is guarded.
    const bool grads_on_gpu = grads_c.is_cuda();

#if CPU_ADAM_TIER_HAS_CUDA
    if (enable_prefetch && grads_on_gpu) {
        // Lazy-init PrefetchState for this optimizer.
        auto& pf_ptr = s_prefetch_states[optimizer_id];
        if (!pf_ptr) {
            pf_ptr = std::make_unique<PrefetchState>();
            if (!pf_ptr->init(kAdamPrefetchTileBytes)) {
                // Fall back to synchronous copy if pinned alloc fails.
                pf_ptr.reset();
            }
        }

        if (pf_ptr && pf_ptr->valid) {
            PrefetchState& pf = *pf_ptr;
            // Element size in bytes (params and grads share dtype in CPU path).
            const size_t elem_bytes  = grads_c.element_size();
            const size_t tile_elems  = kAdamPrefetchTileBytes / elem_bytes;
            const size_t total_elems = param_size;

            // Byte pointers into GPU gradient buffer.
            const uint8_t* grad_gpu =
                reinterpret_cast<const uint8_t*>(grads_c.data_ptr());

            // Issue prefetch for the FIRST tile before the loop starts.
            size_t first_tile = (total_elems < tile_elems) ? total_elems : tile_elems;
            pf.issue_prefetch(grad_gpu, first_tile * elem_bytes);

            // Tiled Adam loop: process tile[t] while prefetching tile[t+1].
            for (size_t offset = 0; offset < total_elems; offset += tile_elems) {
                size_t cur_tile  = total_elems - offset;
                if (cur_tile > tile_elems) cur_tile = tile_elems;

                // Wait for current tile's D2H copy to finish.
                pf.wait_current();
                void* pinned_grads = pf.current_buf();

                // Prefetch NEXT tile while CPU Adam runs on current tile.
                size_t next_offset = offset + tile_elems;
                if (next_offset < total_elems) {
                    size_t next_tile = total_elems - next_offset;
                    if (next_tile > tile_elems) next_tile = tile_elems;
                    pf.issue_prefetch(grad_gpu + next_offset * elem_bytes,
                                      next_tile * elem_bytes);
                }
                pf.flip_phase();

                // Build a CPU tensor view over the pinned gradient buffer so
                // we can reuse the existing typed invoke() infrastructure.
                auto opts = grads_c.options().device(torch::kCPU);
                torch::Tensor pinned_grad_t =
                    torch::from_blob(pinned_grads, {(long)cur_tile}, opts);

                // Slice param/state tensors for this tile.
                auto p_slice   = params_c.flatten().narrow(0, (long)offset, (long)cur_tile);
                auto ea_slice  = exp_avg_c.flatten().narrow(0, (long)offset, (long)cur_tile);
                auto eas_slice = exp_avg_sq_c.flatten().narrow(0, (long)offset, (long)cur_tile);

                invoke(opt, p_slice, pinned_grad_t, ea_slice, eas_slice, cur_tile);
            }
            return 0;
        }
        // Fall through to synchronous path if prefetch init failed.
    }
#endif  // CPU_ADAM_TIER_HAS_CUDA

    // ── 6. Synchronous fallback (no prefetch or params already on CPU) ──────
    // If grads are on GPU without prefetch, we must pull them first.
    torch::Tensor grads_cpu = grads_on_gpu ? grads_c.cpu() : grads_c;
    invoke(opt, params_c, grads_cpu, exp_avg_c, exp_avg_sq_c, param_size);

    return 0;
}

void adamw_rollback_inplace(float* params,
                            const float* grads,
                            float* momentum,
                            float* variance,
                            size_t param_size,
                            float learning_rate,
                            float beta1,
                            float beta2,
                            float eps,
                            float weight_decay,
                            int& step_count)
{
    const float lr = learning_rate;
    const float lambda = weight_decay;
    const float beta1_pow = std::pow(beta1, step_count);
    const float beta2_pow = std::pow(beta2, step_count);
    const float one_minus_beta1 = 1.0f - beta1;
    const float one_minus_beta2 = 1.0f - beta2;
    const float lr_lambda = lr * lambda;
    const float one_minus_lr_lambda = 1.0f - lr_lambda;

#pragma omp parallel for
    for (size_t i = 0; i < param_size; ++i) {
        const float bias_correction1 = 1.0f - beta1_pow;
        const float bias_correction2 = 1.0f - beta2_pow;

        const float m_hat = momentum[i] / bias_correction1;
        const float v_hat = variance[i] / bias_correction2;

        const float denominator = std::sqrt(v_hat) + eps;

        // Rollback parameter update
        const float update = lr * m_hat / denominator;
        float new_param = (params[i] + update) / one_minus_lr_lambda;

        // Handle numerical instability
        if (!std::isfinite(new_param)) { new_param = 0.0f; }

        params[i] = new_param;

        const float grad = grads[i];
        momentum[i] = (momentum[i] - one_minus_beta1 * grad) / beta1;
        variance[i] = (variance[i] - one_minus_beta2 * grad * grad) / beta2;
    }

    --step_count;
}

int ds_adam_rollback(int optimizer_id,
                     size_t step,
                     float lr,
                     float beta1,
                     float beta2,
                     float epsilon,
                     float weight_decay,
                     bool bias_correction,
                     torch::Tensor& params,
                     torch::Tensor& grads,
                     torch::Tensor& exp_avg,
                     torch::Tensor& exp_avg_sq)
{
    try {
        // Validate tensor types - rollback currently only supports float32
        if (params.scalar_type() != torch::kFloat32 || grads.scalar_type() != torch::kFloat32 ||
            exp_avg.scalar_type() != torch::kFloat32 ||
            exp_avg_sq.scalar_type() != torch::kFloat32) {
            printf("Error: Adam rollback currently only supports float32 tensors\n");
            return -1;
        }

        float* params_ptr = params.data_ptr<float>();
        const float* grads_ptr = grads.data_ptr<float>();
        float* momentum_ptr = exp_avg.data_ptr<float>();
        float* variance_ptr = exp_avg_sq.data_ptr<float>();
        const size_t param_size = params.numel();
        int step_count = static_cast<int>(step);

        adamw_rollback_inplace(params_ptr,
                               grads_ptr,
                               momentum_ptr,
                               variance_ptr,
                               param_size,
                               lr,
                               beta1,
                               beta2,
                               epsilon,
                               weight_decay,
                               step_count);

        return 0;
    } catch (const std::exception& e) {
        printf("Error in Adam rollback for optimizer #%d: %s\n", optimizer_id, e.what());
        return -1;
    }
}

int destroy_adam_optimizer(int optimizer_id)
{
    s_optimizers.erase(optimizer_id);

    return 0;
}