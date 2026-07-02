# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import sys
from .builder import CUDAOpBuilder


class HeteroReduceBuilder(CUDAOpBuilder):
    """
    Builds the hetero_reduce CUDA extension:

      * csrc/hetero_reduce/hetero_reduce.cu   — BF16→FP32 reduce-scatter + cast
      * csrc/hetero_reduce/fused_swiglu_ln.cu — fused SwiGLU + RMSNorm
      * csrc/hetero_reduce/binding.cpp         — pybind11 glue

    Heterogeneous targets (all PCIe, no NVLink):
      SM 8.6  — RTX A6000
      SM 9.0  — H100
      SM 12.0 — Blackwell (requires CUDA ≥ 12.8 toolchain)

    The sm_version argument passed to launch_fused_bf16_reduce /
    launch_fused_swiglu_ln at *runtime* (not build time) selects the
    occupancy-tuned kernel variant for the active device.
    """

    BUILD_VAR = "DS_BUILD_HETERO_REDUCE"
    NAME = "hetero_reduce"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f"deepspeed.ops.hetero_reduce.{self.NAME}_op"

    def sources(self):
        return [
            "csrc/hetero_reduce/binding.cpp",
            "csrc/hetero_reduce/hetero_reduce.cu",
            "csrc/hetero_reduce/fused_swiglu_ln.cu",
        ]

    def include_paths(self):
        return [
            "csrc/includes",
            "csrc/hetero_reduce",
        ]

    def cxx_args(self):
        args = super().cxx_args()
        return args + self.version_dependent_macros()

    def nvcc_args(self):
        nvcc_flags = ["-O3"] + self.version_dependent_macros()
        if not self.is_rocm_pytorch():
            base_flags = [
                "-lineinfo",
                "--use_fast_math",
                # SM 8.6 — RTX A6000
                "-gencode", "arch=compute_86,code=sm_86",
                # SM 9.0 — H100
                "-gencode", "arch=compute_90,code=sm_90",
                # Forward-compatible PTX for unknown SM ≥ 9.0 (covers SM 12.0
                # when compiled with CUDA < 12.8 that lacks sm_120 target).
                "-gencode", "arch=compute_90,code=compute_90",
            ]
            # SM 12.0 (Blackwell) needs CUDA ≥ 12.8 for native SASS codegen.
            try:
                cuda_major, cuda_minor = self.installed_cuda_version()
                if cuda_major > 12 or (cuda_major == 12 and cuda_minor >= 8):
                    base_flags += ["-gencode", "arch=compute_120,code=sm_120"]
            except Exception:
                pass  # toolchain absent; skip sm_120

            if sys.platform == "win32":
                base_flags = ["-allow-unsupported-compiler"] + base_flags

            nvcc_flags.extend(base_flags)
            nvcc_flags = [f for f in nvcc_flags if f]
        return nvcc_flags

    def extra_ldflags(self):
        return []
