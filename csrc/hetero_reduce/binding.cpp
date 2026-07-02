// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
 * binding.cpp
 *
 * PyTorch / pybind11 bindings for the hetero_reduce CUDA kernels.
 *
 * Exposed Python API
 * ------------------
 *   hetero_reduce.fused_bf16_reduce(output, inputs, sm_version) -> None
 *       output : torch.Tensor  BF16, device tensor, shape [N]
 *       inputs : List[torch.Tensor]  BF16 device tensors, each shape [N]
 *       sm_version : int  e.g. 86, 90, 120
 *
 *   hetero_reduce.hetero_reduce_scatter(output, inputs, shard_offset,
 *                                        shard_count, sm_version) -> None
 *       output       : torch.Tensor BF16, device tensor [shard_count]
 *       inputs       : List[torch.Tensor] BF16 device tensors [N]
 *       shard_offset : int  starting element index in the full tensor
 *       shard_count  : int  number of elements this device writes
 *       sm_version   : int  e.g. 86, 90, 120
 *
 *   hetero_reduce.compute_shard_ranges(sm_versions, total_elems)
 *                                        -> List[Tuple[int, int]]
 *       sm_versions : List[int]  per-tier SM versions
 *       total_elems : int        total BF16 elements
 *       Returns list of (offset, count) tuples, one per tier.
 *
 *   hetero_reduce.fused_swiglu_ln(output, gate_proj, up_proj, ln_weight,
 *                                  eps, sm_version) -> None
 *       output     : torch.Tensor  BF16  [batch, hidden]
 *       gate_proj  : torch.Tensor  BF16  [batch, hidden]
 *       up_proj    : torch.Tensor  BF16  [batch, hidden]
 *       ln_weight  : torch.Tensor  FP32  [hidden]
 *       eps        : float
 *       sm_version : int
 */

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <vector>
#include <stdexcept>

#include "hetero_reduce.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void check_bf16(const at::Tensor& t, const char* name)
{
    TORCH_CHECK(t.scalar_type() == at::ScalarType::BFloat16,
                name, " must be BFloat16, got ", t.scalar_type());
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

static void check_fp32(const at::Tensor& t, const char* name)
{
    TORCH_CHECK(t.scalar_type() == at::ScalarType::Float,
                name, " must be Float32, got ", t.scalar_type());
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// ---------------------------------------------------------------------------
// fused_bf16_reduce binding
// ---------------------------------------------------------------------------

void fused_bf16_reduce_py(at::Tensor output,
                           std::vector<at::Tensor> inputs,
                           int sm_version)
{
    check_bf16(output, "output");
    TORCH_CHECK(!inputs.empty(), "inputs list must not be empty");
    TORCH_CHECK(inputs.size() <= 32,
                "fused_bf16_reduce supports at most 32 input tensors, got ", inputs.size());

    const size_t n_elems = static_cast<size_t>(output.numel());
    TORCH_CHECK(n_elems % 8 == 0,
                "output numel must be divisible by 8 for vectorised loads, got ", n_elems);

    // Collect raw device pointers.
    std::vector<const __nv_bfloat16*> ptrs;
    ptrs.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
        check_bf16(inputs[i], ("inputs[" + std::to_string(i) + "]").c_str());
        TORCH_CHECK(static_cast<size_t>(inputs[i].numel()) == n_elems,
                    "inputs[", i, "] numel mismatch: expected ", n_elems,
                    " got ", inputs[i].numel());
        ptrs.push_back(reinterpret_cast<const __nv_bfloat16*>(inputs[i].data_ptr<at::BFloat16>()));
    }

    __nv_bfloat16* out_ptr =
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_fused_bf16_reduce(out_ptr, ptrs.data(),
                              static_cast<int>(ptrs.size()),
                              n_elems, sm_version, stream);
}

// ---------------------------------------------------------------------------
// hetero_reduce_scatter binding
// ---------------------------------------------------------------------------

void hetero_reduce_scatter_py(at::Tensor output,
                               std::vector<at::Tensor> inputs,
                               int64_t shard_offset,
                               int64_t shard_count,
                               int sm_version)
{
    check_bf16(output, "output");
    TORCH_CHECK(!inputs.empty(), "inputs list must not be empty");
    TORCH_CHECK(inputs.size() <= 32,
                "supports at most 32 input tensors, got ", inputs.size());

    TORCH_CHECK(shard_offset >= 0, "shard_offset must be >= 0");
    TORCH_CHECK(shard_count > 0,   "shard_count must be > 0");
    TORCH_CHECK(shard_count % 8 == 0,
                "shard_count must be divisible by 8, got ", shard_count);
    TORCH_CHECK(static_cast<size_t>(output.numel()) >= static_cast<size_t>(shard_count),
                "output numel must be >= shard_count");

    std::vector<const __nv_bfloat16*> ptrs;
    ptrs.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
        check_bf16(inputs[i], ("inputs[" + std::to_string(i) + "]").c_str());
        TORCH_CHECK(static_cast<size_t>(inputs[i].numel()) >=
                    static_cast<size_t>(shard_offset + shard_count),
                    "inputs[", i, "] numel too small for shard range");
        ptrs.push_back(reinterpret_cast<const __nv_bfloat16*>(inputs[i].data_ptr<at::BFloat16>()));
    }

    __nv_bfloat16* out_ptr =
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_hetero_reduce_scatter(out_ptr, ptrs.data(),
                                  static_cast<int>(ptrs.size()),
                                  static_cast<size_t>(shard_offset),
                                  static_cast<size_t>(shard_count),
                                  sm_version, stream);
}

// ---------------------------------------------------------------------------
// compute_shard_ranges binding
// ---------------------------------------------------------------------------

std::vector<std::tuple<int64_t, int64_t>>
compute_shard_ranges_py(std::vector<int> sm_versions, int64_t total_elems)
{
    TORCH_CHECK(!sm_versions.empty(), "sm_versions must not be empty");
    TORCH_CHECK(total_elems > 0, "total_elems must be > 0");
    TORCH_CHECK(total_elems % 8 == 0,
                "total_elems must be divisible by 8, got ", total_elems);

    const int num_tiers = static_cast<int>(sm_versions.size());
    std::vector<HeteroTierDesc> tiers(num_tiers);
    for (int i = 0; i < num_tiers; i++) {
        tiers[i].device_id   = i;
        tiers[i].sm_version  = sm_versions[i];
        tiers[i].bucket_size = 0;
    }

    std::vector<size_t> offsets(num_tiers);
    std::vector<size_t> counts(num_tiers);
    compute_hetero_shard_ranges(tiers.data(), num_tiers,
                                 static_cast<size_t>(total_elems),
                                 offsets.data(), counts.data());

    std::vector<std::tuple<int64_t, int64_t>> result;
    result.reserve(num_tiers);
    for (int i = 0; i < num_tiers; i++) {
        result.emplace_back(static_cast<int64_t>(offsets[i]),
                            static_cast<int64_t>(counts[i]));
    }
    return result;
}

// ---------------------------------------------------------------------------
// fused_swiglu_ln binding
// ---------------------------------------------------------------------------

void fused_swiglu_ln_py(at::Tensor output,
                         at::Tensor gate_proj,
                         at::Tensor up_proj,
                         at::Tensor ln_weight,
                         float eps,
                         int sm_version)
{
    check_bf16(output,    "output");
    check_bf16(gate_proj, "gate_proj");
    check_bf16(up_proj,   "up_proj");
    check_fp32(ln_weight, "ln_weight");

    TORCH_CHECK(output.dim() == 2,    "output must be 2-D [batch, hidden]");
    TORCH_CHECK(gate_proj.dim() == 2, "gate_proj must be 2-D [batch, hidden]");
    TORCH_CHECK(up_proj.dim() == 2,   "up_proj must be 2-D [batch, hidden]");

    const int batch  = static_cast<int>(output.size(0));
    const int hidden = static_cast<int>(output.size(1));

    TORCH_CHECK(hidden % 8 == 0,
                "hidden must be divisible by 8, got ", hidden);
    TORCH_CHECK(gate_proj.size(0) == batch && gate_proj.size(1) == hidden,
                "gate_proj shape mismatch");
    TORCH_CHECK(up_proj.size(0) == batch && up_proj.size(1) == hidden,
                "up_proj shape mismatch");
    TORCH_CHECK(ln_weight.numel() == hidden,
                "ln_weight must have numel == hidden, got ", ln_weight.numel());

    __nv_bfloat16* out_ptr =
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
    const __nv_bfloat16* gate_ptr =
        reinterpret_cast<const __nv_bfloat16*>(gate_proj.data_ptr<at::BFloat16>());
    const __nv_bfloat16* up_ptr =
        reinterpret_cast<const __nv_bfloat16*>(up_proj.data_ptr<at::BFloat16>());
    const float* w_ptr = ln_weight.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_fused_swiglu_ln(out_ptr, gate_ptr, up_ptr, w_ptr,
                            batch, hidden, eps, sm_version, stream);
}

// ---------------------------------------------------------------------------
// PYBIND11_MODULE
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "DeepSpeed hetero_reduce: fused BF16 reduce-scatter + SwiGLU-LN "
              "kernels for heterogeneous GPU clusters (SM 8.6 / 9.0 / 12.0).";

    m.def("fused_bf16_reduce",
          &fused_bf16_reduce_py,
          "Fused BF16→FP32 reduce + FP32→BF16 writeback across multiple tensors.\n"
          "Args:\n"
          "  output     (Tensor BF16): in-place reduction destination\n"
          "  inputs     (List[Tensor BF16]): tensors to reduce\n"
          "  sm_version (int): SM version of active device (86, 90, 120, …)",
          py::arg("output"),
          py::arg("inputs"),
          py::arg("sm_version") = 86);

    m.def("hetero_reduce_scatter",
          &hetero_reduce_scatter_py,
          "Heterogeneous reduce-scatter: reduces all inputs but writes only the\n"
          "local shard [shard_offset, shard_offset + shard_count) to output.\n"
          "Args:\n"
          "  output       (Tensor BF16): shard output buffer [shard_count]\n"
          "  inputs       (List[Tensor BF16]): full-length input gradient tensors\n"
          "  shard_offset (int): starting element index in full tensor\n"
          "  shard_count  (int): number of elements to reduce and write\n"
          "  sm_version   (int): SM version of active device",
          py::arg("output"),
          py::arg("inputs"),
          py::arg("shard_offset"),
          py::arg("shard_count"),
          py::arg("sm_version") = 86);

    m.def("compute_shard_ranges",
          &compute_shard_ranges_py,
          "Compute non-uniform shard ranges for heterogeneous GPU tiers.\n"
          "Returns List[Tuple[offset, count]] with one entry per tier.\n"
          "Weight: SM12.0=4, SM9.0=3, SM8.6=1.\n"
          "Args:\n"
          "  sm_versions (List[int]): per-tier SM versions\n"
          "  total_elems (int): total BF16 elements in gradient tensor",
          py::arg("sm_versions"),
          py::arg("total_elems"));

    m.def("fused_swiglu_ln",
          &fused_swiglu_ln_py,
          "Fused SwiGLU activation + RMS LayerNorm.\n"
          "Args:\n"
          "  output     (Tensor BF16  [B, H]): output buffer\n"
          "  gate_proj  (Tensor BF16  [B, H]): gate projection\n"
          "  up_proj    (Tensor BF16  [B, H]): up   projection\n"
          "  ln_weight  (Tensor FP32  [H])   : RMSNorm scale\n"
          "  eps        (float)               : RMSNorm epsilon\n"
          "  sm_version (int)                 : 86, 90, or 120",
          py::arg("output"),
          py::arg("gate_proj"),
          py::arg("up_proj"),
          py::arg("ln_weight"),
          py::arg("eps") = 1e-6f,
          py::arg("sm_version") = 86);
}
