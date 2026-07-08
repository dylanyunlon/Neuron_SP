// Copyright (c) 2026 Neuron_SP Project. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Addresses issue #22: tier-aware activation checkpointing offload

/*
 * binding.cpp — PyTorch / pybind11 bindings for the OffloadManager.
 *
 * Python API:
 *   activation_offload.init(sm_version, total_act_bytes, vram_free_bytes,
 *                           headroom_frac, use_quant, num_layers) -> bool
 *   activation_offload.push(tensor, layer_idx) -> None
 *   activation_offload.pop(tensor, layer_idx) -> None
 *   activation_offload.sync() -> None
 *   activation_offload.flip_phase() -> None
 */

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations from offload_manager.cu.
extern "C" {
    bool offload_manager_init(int sm_version,
                              size_t total_act_bytes,
                              size_t vram_free_bytes,
                              float headroom_frac,
                              bool use_quant,
                              int num_layers);
    void offload_manager_push(const void* dev_ptr,
                              size_t n_elems,
                              int layer_idx,
                              cudaStream_t compute_stream);
    void offload_manager_pop(void* dev_ptr,
                             int layer_idx,
                             cudaStream_t compute_stream);
    void offload_manager_sync();
    void offload_manager_flip_phase();
}

// ─── Python-facing wrappers ──────────────────────────────────────────────────

static bool init_py(int sm_version,
                    int64_t total_act_bytes,
                    int64_t vram_free_bytes,
                    float headroom_frac,
                    bool use_quant,
                    int num_layers)
{
    return offload_manager_init(
        sm_version,
        static_cast<size_t>(total_act_bytes),
        static_cast<size_t>(vram_free_bytes),
        headroom_frac, use_quant, num_layers);
}

static void push_py(at::Tensor tensor, int layer_idx)
{
    TORCH_CHECK(tensor.scalar_type() == at::ScalarType::BFloat16,
                "activation_offload.push: tensor must be BFloat16");
    TORCH_CHECK(tensor.is_cuda(), "activation_offload.push: tensor must be CUDA");
    TORCH_CHECK(tensor.is_contiguous(), "activation_offload.push: tensor must be contiguous");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    offload_manager_push(
        tensor.data_ptr(),
        static_cast<size_t>(tensor.numel()),
        layer_idx, stream);
}

static void pop_py(at::Tensor tensor, int layer_idx)
{
    TORCH_CHECK(tensor.scalar_type() == at::ScalarType::BFloat16,
                "activation_offload.pop: tensor must be BFloat16");
    TORCH_CHECK(tensor.is_cuda(), "activation_offload.pop: tensor must be CUDA");
    TORCH_CHECK(tensor.is_contiguous(), "activation_offload.pop: tensor must be contiguous");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    offload_manager_pop(
        tensor.data_ptr(),
        layer_idx, stream);
}

static void sync_py() { offload_manager_sync(); }
static void flip_phase_py() { offload_manager_flip_phase(); }

// ─── Module registration ────────────────────────────────────────────────────

PYBIND11_MODULE(activation_offload, m) {
    m.doc() = "Tier-aware activation checkpoint offload (issue #22)";

    m.def("init", &init_py,
          "Initialise the offload manager for the current GPU tier.\n"
          "\n"
          "Args:\n"
          "    sm_version: SM version (86, 90, 120)\n"
          "    total_act_bytes: Total activation storage required\n"
          "    vram_free_bytes: Current free VRAM\n"
          "    headroom_frac: Fraction of VRAM to keep free\n"
          "    use_quant: Enable INT8 quantisation for bandwidth reduction\n"
          "    num_layers: Number of transformer layers\n"
          "\n"
          "Returns: True on success",
          py::arg("sm_version"),
          py::arg("total_act_bytes"),
          py::arg("vram_free_bytes"),
          py::arg("headroom_frac") = 0.10f,
          py::arg("use_quant") = false,
          py::arg("num_layers") = 32);

    m.def("push", &push_py,
          "Offload a layer's activation from GPU to host (async D2H).\n"
          "Call during forward pass after layer completes.",
          py::arg("tensor"),
          py::arg("layer_idx"));

    m.def("pop", &pop_py,
          "Prefetch a layer's activation from host to GPU (async H2D).\n"
          "Call 1 layer ahead during backward pass.",
          py::arg("tensor"),
          py::arg("layer_idx"));

    m.def("sync", &sync_py,
          "Wait for all pending offload/prefetch transfers.");

    m.def("flip_phase", &flip_phase_py,
          "Advance double-buffer phase (call between forward and backward).");
}
