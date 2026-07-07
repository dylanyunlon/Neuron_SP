# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# addresses #145 — FlashAttention-2 heterogeneous SM kernel

import sys
from .builder import CUDAOpBuilder


class FusedAttentionBuilder(CUDAOpBuilder):
    """
    Builds the ds_fused_attention CUDA extension for heterogeneous GPU clusters.

    Source files
    ------------
    * csrc/attention/fused_attention.cu      — FA-2 online-softmax MHA/GQA kernel
    * csrc/attention/fused_gqa_attention.cu  — warp-group GQA kernel (KV-sharing)
    * csrc/attention/binding.cpp             — pybind11 / PyTorch extension glue

    Heterogeneous targets (all PCIe, no NVLink):
      SM 8.6  — RTX A6000  (48 GB VRAM, Ampere)
      SM 9.0  — H100       (80 GB VRAM, Hopper)
      SM 12.0 — Blackwell  (192 GB VRAM NVL72, requires CUDA >= 12.8)

    The sm_version argument passed to each launch_* function at *runtime*
    selects the occupancy-tuned kernel variant (AttnPolicy<86/90/120>).
    A single binary serves all three tiers — no recompile needed when
    migrating attention layers between devices in the DES-LOC scheduler.

    SM Occupancy Summary
    --------------------
    SM 8.6 (A6000, 84 SMs):
      AttnPolicy<86>: Br=64, Bc=64, 128 threads/block (4 warps)
      smem: 57 KB (smem_q + smem_k + smem_v + smem_s in BF16) < 64 KB limit
      kMinBlocksPerSM=2 → 16 warps/SM → ~50% warp occupancy (smem-limited)

    SM 9.0 (H100, 132 SMs):
      AttnPolicy<90>: Br=128, Bc=128, 256 threads/block (8 warps)
      smem: 131 KB < H100's 228 KB per-SM budget; cp.async overlaps HBM reads
      kMinBlocksPerSM=2 → ~75% warp occupancy at peak sequence lengths

    SM 12.0 (Blackwell):
      AttnPolicy<120>: Br=128, Bc=64, persistent kernel over atomic tile queue
      cp.async.bulk (TMA-style) for K/V loads; grid=num_SMs×kMinBlocksPerSM

    Exposed Python API (after load())
    ------------------------------------
    op.fused_attention_forward(query, key, value, softmax_scale,
                               causal, window_left, window_right,
                               dropout_p, philox_seed, philox_offset,
                               sm_version)
        -> Tuple[Tensor, Tensor]   # (output [B,Hq,Sq,D] BF16, lse [B,Hq,Sq] FP32)

    op.fused_attention_backward(d_output, query, key, value, output, lse,
                                softmax_scale, causal, sm_version)
        -> Tuple[Tensor, Tensor, Tensor]  # (dq, dk, dv) all BF16

    op.fused_gqa_attention_forward(query, key, value, num_kv_heads,
                                   causal, sm_scale, sm_version)
        -> Tensor  # output [B,Hq,Sq,D] BF16
    """

    BUILD_VAR = "DS_BUILD_FUSED_ATTENTION"
    NAME = "fused_attention"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f"deepspeed.ops.fused_attention.{self.NAME}_op"

    def sources(self):
        return [
            # pybind11 glue — compiled by host CXX
            "csrc/attention/binding.cpp",
            # FA-2 MHA/GQA online-softmax kernel (SM-dispatched tile sizes)
            "csrc/attention/fused_attention.cu",
            # Warp-group GQA kernel: one block per KV-head group,
            # K/V tile shared across all gqa_ratio Q-head warps
            "csrc/attention/fused_gqa_attention.cu",
        ]

    def include_paths(self):
        return [
            # DeepSpeed shared headers: DS_D_INLINE, hw_warp_size, etc.
            "csrc/includes",
            # fused_attention.h, fused_gqa_attention.h (launch_* declarations)
            "csrc/attention",
        ]

    def cxx_args(self):
        # Inherit base CXX flags (C++17, -fPIC, etc.) and add BF16/version macros
        args = super().cxx_args()
        return args + self.version_dependent_macros()

    def nvcc_args(self):
        # Start from base optimisation flags and version macros
        nvcc_flags = ["-O3", "--expt-relaxed-constexpr"] + self.version_dependent_macros()

        if not self.is_rocm_pytorch():
            base_flags = [
                # Line info for Nsight / cuda-gdb
                "-lineinfo",
                # Fast intrinsics (sincos, reciprocal, etc.)
                "--use_fast_math",
                # C++17 in device code (needed for if constexpr in AttnPolicy templates)
                "-std=c++17",
                # ----------------------------------------------------------------
                # Code-generation targets — one per supported SM generation.
                # ----------------------------------------------------------------
                # SM 8.6 — RTX A6000 (Ampere)
                "-gencode", "arch=compute_86,code=sm_86",
                # SM 9.0 — H100 (Hopper); enables cp.async path in kernel
                "-gencode", "arch=compute_90,code=sm_90",
                # Forward-compatible PTX: handles SM 9.x variants and future
                # architectures that the current CUDA toolchain doesn't yet know.
                "-gencode", "arch=compute_90,code=compute_90",
            ]

            # SM 12.0 (Blackwell) requires CUDA >= 12.8 for native SASS.
            # Add the native target when the toolchain supports it; the PTX
            # fallback above still works with older toolchains.
            try:
                cuda_major, cuda_minor = self.installed_cuda_version()
                if cuda_major > 12 or (cuda_major == 12 and cuda_minor >= 8):
                    base_flags += ["-gencode", "arch=compute_120,code=sm_120"]
            except Exception:
                pass  # nvcc absent or version query failed; PTX fallback suffices

            if sys.platform == "win32":
                base_flags = ["-allow-unsupported-compiler"] + base_flags

            nvcc_flags.extend(base_flags)
            # Remove any accidental empty strings that confuse nvcc
            nvcc_flags = [f for f in nvcc_flags if f]

        return nvcc_flags

    def extra_ldflags(self):
        # No extra libraries beyond what PyTorch already links (libcuda, libcudart)
        return []

    def is_compatible(self, verbose=True):
        """
        fused_attention requires CUDA 11.0+ for BF16 (__nv_bfloat16) and
        cooperative groups (used for warp-level row-max reductions).
        """
        try:
            cuda_major, cuda_minor = self.installed_cuda_version()
            if cuda_major < 11:
                if verbose:
                    self.warning(
                        f"{self.NAME}: CUDA {cuda_major}.{cuda_minor} < 11.0; "
                        "BF16 (__nv_bfloat16) and cooperative groups require CUDA >= 11.0."
                    )
                return False
        except Exception:
            pass  # Let the parent class handle the absence of nvcc
        return super().is_compatible(verbose)
