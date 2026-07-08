// Copyright (c) 2026 Neuron_SP Project. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Addresses issue #24: PCIe-aware NCCL allreduce with adaptive bucketing

/*
 * binding.cpp — PyTorch / pybind11 bindings for pcie_comm BucketManager.
 *
 * Python API:
 *   pcie_comm.init(device_ids, world_size, rank, sm_version) -> bool
 *   pcie_comm.enqueue(gradient_tensor) -> None
 *   pcie_comm.flush() -> None
 *   pcie_comm.sync() -> None
 *   pcie_comm.get_bucket_elems() -> int
 */

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

// Forward declarations from pcie_bucket_manager.cu.
extern "C" {
    bool bucket_manager_init(const int* device_ids,
                             int world_size,
                             int rank,
                             int sm_version);
    void bucket_manager_enqueue(void* grad_ptr, size_t n_elems);
    void bucket_manager_flush();
    void bucket_manager_sync();
    size_t bucket_manager_get_bucket_elems();
}

// ─── Python wrappers ─────────────────────────────────────────────────────────

static bool init_py(std::vector<int> device_ids,
                    int world_size,
                    int rank,
                    int sm_version)
{
    TORCH_CHECK((int)device_ids.size() == world_size,
                "device_ids length must equal world_size");
    return bucket_manager_init(device_ids.data(), world_size, rank, sm_version);
}

static void enqueue_py(at::Tensor gradient)
{
    TORCH_CHECK(gradient.scalar_type() == at::ScalarType::BFloat16,
                "pcie_comm.enqueue: gradient must be BFloat16");
    TORCH_CHECK(gradient.is_cuda(),
                "pcie_comm.enqueue: gradient must be CUDA");
    TORCH_CHECK(gradient.is_contiguous(),
                "pcie_comm.enqueue: gradient must be contiguous");

    bucket_manager_enqueue(
        gradient.data_ptr(),
        static_cast<size_t>(gradient.numel()));
}

static void flush_py() { bucket_manager_flush(); }
static void sync_py()  { bucket_manager_sync(); }

static int64_t get_bucket_elems_py()
{
    return static_cast<int64_t>(bucket_manager_get_bucket_elems());
}

// ─── Module registration ────────────────────────────────────────────────────

PYBIND11_MODULE(pcie_comm, m) {
    m.doc() = "PCIe-aware gradient allreduce with adaptive bucketing (issue #24)";

    m.def("init", &init_py,
          "Initialise the bucket manager.\n"
          "\n"
          "Probes PCIe topology, measures bandwidth, and computes adaptive\n"
          "bucket sizes.  Call once per training run.\n"
          "\n"
          "Args:\n"
          "    device_ids: CUDA device ordinals for all ranks\n"
          "    world_size: Number of GPUs\n"
          "    rank: This rank's index\n"
          "    sm_version: SM version (86, 90, 120)\n"
          "\n"
          "Returns: True on success",
          py::arg("device_ids"),
          py::arg("world_size"),
          py::arg("rank"),
          py::arg("sm_version"));

    m.def("enqueue", &enqueue_py,
          "Enqueue a gradient tensor for allreduce.\n"
          "Auto-flushes when bucket is full.",
          py::arg("gradient"));

    m.def("flush", &flush_py,
          "Flush pending gradients: pack + allreduce + unpack.");

    m.def("sync", &sync_py,
          "Synchronise all streams. Call at end of training step.");

    m.def("get_bucket_elems", &get_bucket_elems_py,
          "Query the configured bucket size in BF16 elements.");
}
