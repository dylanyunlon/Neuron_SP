// Copyright (c) 2026 Neuron_SP Project. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Addresses issue #25: fused SwiGLU + LayerNorm for SM8.6/9.0/12.0

/*
 * binding.cpp — PyTorch / pybind11 bindings for csrc/fused_mlp/ kernels.
 *
 * Python API:
 *   fused_mlp.swiglu(output, gate_proj, up_proj, sm_version) -> None
 *   fused_mlp.pre_ln_attn(output, residual, ln_weight, eps, sm_version) -> None
 *   fused_mlp.residual_rmsnorm(output, residual, input, ln_weight, eps, sm_version) -> None
 */

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations from fused_swiglu_mlp.cu.
void launch_fused_swiglu(
    __nv_bfloat16* output, const __nv_bfloat16* gate_proj,
    const __nv_bfloat16* up_proj, int batch, int hidden,
    int sm_version, cudaStream_t stream);

void launch_fused_pre_ln_attn(
    __nv_bfloat16* output, const __nv_bfloat16* residual,
    const float* ln_weight, int batch, int hidden, float eps,
    int sm_version, cudaStream_t stream);

void launch_fused_residual_rmsnorm(
    __nv_bfloat16* output, __nv_bfloat16* residual,
    const __nv_bfloat16* input, const float* ln_weight,
    int batch, int hidden, float eps,
    int sm_version, cudaStream_t stream);

// ─── Helpers ─────────────────────────────────────────────────────────────────

static void check_bf16(const at::Tensor& t, const char* name)
{
    TORCH_CHECK(t.scalar_type() == at::ScalarType::BFloat16,
                name, " must be BFloat16");
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

static void check_fp32(const at::Tensor& t, const char* name)
{
    TORCH_CHECK(t.scalar_type() == at::ScalarType::Float,
                name, " must be Float32");
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// ─── Python wrappers ─────────────────────────────────────────────────────────

static void swiglu_py(at::Tensor output,
                      at::Tensor gate_proj,
                      at::Tensor up_proj,
                      int sm_version)
{
    check_bf16(output,    "output");
    check_bf16(gate_proj, "gate_proj");
    check_bf16(up_proj,   "up_proj");

    TORCH_CHECK(gate_proj.dim() == 2, "gate_proj must be 2D [batch, hidden]");
    TORCH_CHECK(up_proj.dim() == 2,   "up_proj must be 2D [batch, hidden]");
    TORCH_CHECK(output.dim() == 2,    "output must be 2D [batch, hidden]");

    const int batch  = gate_proj.size(0);
    const int hidden = gate_proj.size(1);

    TORCH_CHECK(hidden % 8 == 0, "hidden must be divisible by 8");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    launch_fused_swiglu(
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(gate_proj.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(up_proj.data_ptr<at::BFloat16>()),
        batch, hidden, sm_version, stream);
}

static void pre_ln_attn_py(at::Tensor output,
                            at::Tensor residual,
                            at::Tensor ln_weight,
                            float eps,
                            int sm_version)
{
    check_bf16(output,   "output");
    check_bf16(residual, "residual");
    check_fp32(ln_weight, "ln_weight");

    TORCH_CHECK(residual.dim() == 2, "residual must be 2D [batch, hidden]");
    TORCH_CHECK(output.dim() == 2,   "output must be 2D [batch, hidden]");

    const int batch  = residual.size(0);
    const int hidden = residual.size(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    launch_fused_pre_ln_attn(
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
        ln_weight.data_ptr<float>(),
        batch, hidden, eps, sm_version, stream);
}

static void residual_rmsnorm_py(at::Tensor output,
                                 at::Tensor residual,
                                 at::Tensor input,
                                 at::Tensor ln_weight,
                                 float eps,
                                 int sm_version)
{
    check_bf16(output,   "output");
    check_bf16(residual, "residual");
    check_bf16(input,    "input");
    check_fp32(ln_weight, "ln_weight");

    TORCH_CHECK(residual.dim() == 2, "residual must be 2D [batch, hidden]");

    const int batch  = residual.size(0);
    const int hidden = residual.size(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    launch_fused_residual_rmsnorm(
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        ln_weight.data_ptr<float>(),
        batch, hidden, eps, sm_version, stream);
}

// ─── Module registration ────────────────────────────────────────────────────

PYBIND11_MODULE(fused_mlp, m) {
    m.doc() = "Fused SwiGLU + LayerNorm for SM8.6/9.0/12.0 (issue #25)";

    m.def("swiglu", &swiglu_py,
          "Fused SwiGLU activation: gate × σ(gate) × up in one kernel.\n"
          "Replaces 3 separate kernel launches.",
          py::arg("output"), py::arg("gate_proj"), py::arg("up_proj"),
          py::arg("sm_version"));

    m.def("pre_ln_attn", &pre_ln_attn_py,
          "Fused pre-LayerNorm for attention input.\n"
          "Computes RMSNorm(residual) × ln_weight.",
          py::arg("output"), py::arg("residual"), py::arg("ln_weight"),
          py::arg("eps") = 1e-5f, py::arg("sm_version") = 90);

    m.def("residual_rmsnorm", &residual_rmsnorm_py,
          "Fused residual add + RMSNorm.\n"
          "residual += input; output = RMSNorm(residual, ln_weight, eps).",
          py::arg("output"), py::arg("residual"), py::arg("input"),
          py::arg("ln_weight"), py::arg("eps") = 1e-5f,
          py::arg("sm_version") = 90);
}
