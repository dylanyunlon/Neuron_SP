// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

// Issues #12 / #84: expose tier-aware Adam entry point alongside the original.

#include "cpu_adam.h"
#include "cpu_adam_tier.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("adam_update", &ds_adam_step, "DeepSpeed CPU Adam update (C++)");
    m.def("adam_rollback", &ds_adam_rollback, "DeepSpeed CPU Adam rollback (C++)");
    m.def("create_adam", &create_adam_optimizer, "DeepSpeed CPU Adam (C++)");
    m.def("destroy_adam", &destroy_adam_optimizer, "DeepSpeed CPU Adam destroy (C++)");

    // Issue #12 / #84: tier-aware update with async prefetch.
    // sm_version=0 → auto-detect; enable_prefetch=True → double-buffered D2H.
    m.def("adam_update_tier",
          &ds_adam_step_tier,
          "DeepSpeed CPU Adam — tier-aware (SM8.6/9.0/12.0) with async prefetch (C++)",
          pybind11::arg("optimizer_id"),
          pybind11::arg("step"),
          pybind11::arg("lr"),
          pybind11::arg("beta1"),
          pybind11::arg("beta2"),
          pybind11::arg("epsilon"),
          pybind11::arg("weight_decay"),
          pybind11::arg("bias_correction"),
          pybind11::arg("params"),
          pybind11::arg("grads"),
          pybind11::arg("exp_avg"),
          pybind11::arg("exp_avg_sq"),
          pybind11::arg("sm_version") = 0,
          pybind11::arg("enable_prefetch") = true);

    // Expose tier detection utility to Python for diagnostics.
    m.def("detect_sm_version",
          []() { return detect_sm_version(); },
          "Return SM version of the current CUDA device (e.g. 86, 90, 120)");
    m.def("tier_strategy_name",
          [](int sm_ver) {
              TierAwareAdamConfig cfg;
              cfg.sm_ver   = sm_ver;
              cfg.tier     = sm_version_to_tier(sm_ver);
              cfg.strategy = tier_to_strategy(cfg.tier);
              return std::string(cfg.strategy_name());
          },
          "Return human-readable offload strategy for given SM version");
}
