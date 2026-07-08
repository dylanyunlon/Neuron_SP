// Copyright (c) 2026 Neuron_SP Project. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Addresses issue #23: fused RoPE for heterogeneous head counts

/*
 * binding.cpp — PyTorch / pybind11 bindings for csrc/rope/ kernels.
 *
 * Python API:
 *   fused_rope.apply_qk(q_out, k_out, q_in, k_in, cos, sin,
 *                        num_heads_q, num_heads_kv, neox_style, sm_version)
 *   fused_rope.apply_qk_cacheless(q_out, k_out, q_in, k_in,
 *                                  num_heads_q, num_heads_kv,
 *                                  base, pos_offset, neox_style, sm_version)
 */

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include "fused_rope.h"

namespace py = pybind11;

static void check_bf16_cuda(const at::Tensor& t, const char* name)
{
    TORCH_CHECK(t.scalar_type() == at::ScalarType::BFloat16,
                name, " must be BFloat16, got ", t.scalar_type());
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

static void apply_qk_py(
    at::Tensor q_output,
    at::Tensor k_output,
    at::Tensor q_input,
    at::Tensor k_input,
    at::Tensor cos_cache,
    at::Tensor sin_cache,
    int num_heads_q,
    int num_heads_kv,
    bool neox_style,
    int sm_version)
{
    check_bf16_cuda(q_output, "q_output");
    check_bf16_cuda(k_output, "k_output");
    check_bf16_cuda(q_input,  "q_input");
    check_bf16_cuda(k_input,  "k_input");

    TORCH_CHECK(q_input.dim() == 4, "q_input must be 4D [B, S, Hq, D]");
    TORCH_CHECK(k_input.dim() == 4, "k_input must be 4D [B, S, Hkv, D]");

    const int B = q_input.size(0);
    const int S = q_input.size(1);
    const int D = q_input.size(3);

    TORCH_CHECK(k_input.size(0) == B && k_input.size(1) == S && k_input.size(3) == D,
                "k_input shape must match q_input in B, S, D dimensions");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    launch_fused_rope_qk(
        reinterpret_cast<__nv_bfloat16*>(q_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(k_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(q_input.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k_input.data_ptr<at::BFloat16>()),
        cos_cache.data_ptr<float>(),
        sin_cache.data_ptr<float>(),
        B, S, num_heads_q, num_heads_kv, D,
        neox_style,
        10000.f, 0,   // base/pos_offset unused in cached mode
        sm_version,
        stream);
}

static void apply_qk_cacheless_py(
    at::Tensor q_output,
    at::Tensor k_output,
    at::Tensor q_input,
    at::Tensor k_input,
    int num_heads_q,
    int num_heads_kv,
    float base,
    int pos_offset,
    bool neox_style,
    int sm_version)
{
    check_bf16_cuda(q_output, "q_output");
    check_bf16_cuda(k_output, "k_output");
    check_bf16_cuda(q_input,  "q_input");
    check_bf16_cuda(k_input,  "k_input");

    TORCH_CHECK(q_input.dim() == 4, "q_input must be 4D [B, S, Hq, D]");
    TORCH_CHECK(k_input.dim() == 4, "k_input must be 4D [B, S, Hkv, D]");

    const int B = q_input.size(0);
    const int S = q_input.size(1);
    const int D = q_input.size(3);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    launch_fused_rope_qk(
        reinterpret_cast<__nv_bfloat16*>(q_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(k_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(q_input.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k_input.data_ptr<at::BFloat16>()),
        nullptr, nullptr,  // cacheless mode
        B, S, num_heads_q, num_heads_kv, D,
        neox_style,
        base, pos_offset,
        sm_version,
        stream);
}

PYBIND11_MODULE(fused_rope, m) {
    m.doc() = "Fused Q+K RoPE for GQA models (issue #23)";

    m.def("apply_qk", &apply_qk_py,
          "Apply RoPE to Q and K simultaneously (cached mode).\n"
          "Handles GQA with different Q/K head counts in a single kernel launch.",
          py::arg("q_output"), py::arg("k_output"),
          py::arg("q_input"),  py::arg("k_input"),
          py::arg("cos_cache"), py::arg("sin_cache"),
          py::arg("num_heads_q"), py::arg("num_heads_kv"),
          py::arg("neox_style"), py::arg("sm_version"));

    m.def("apply_qk_cacheless", &apply_qk_cacheless_py,
          "Apply RoPE to Q and K simultaneously (cacheless mode).\n"
          "Computes sin/cos on-the-fly for very long sequences.",
          py::arg("q_output"), py::arg("k_output"),
          py::arg("q_input"),  py::arg("k_input"),
          py::arg("num_heads_q"), py::arg("num_heads_kv"),
          py::arg("base"), py::arg("pos_offset"),
          py::arg("neox_style"), py::arg("sm_version"));
}
