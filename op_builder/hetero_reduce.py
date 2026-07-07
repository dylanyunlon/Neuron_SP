# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import sys
from .builder import CUDAOpBuilder


class HeteroReduceBuilder(CUDAOpBuilder):
    """
    Builds the hetero_reduce CUDA extension for heterogeneous GPU clusters.

    Source files
    ------------
    * csrc/hetero_reduce/hetero_reduce.cu              — BF16→FP32 fused reduce-scatter
    * csrc/hetero_reduce/fused_rope_hetero.cu          — RoPE with heterogeneous head counts
    * csrc/hetero_reduce/pcie_adaptive_allreduce.cu    — PCIe-aware bucketed allreduce
    * csrc/hetero_reduce/fused_swiglu_ln.cu            — fused SwiGLU + RMSNorm
    * csrc/hetero_reduce/tier_activation_offload.cu    — activation checkpoint pack/unpack
    * csrc/hetero_reduce/fused_layernorm_residual.cu   — fused RMSNorm + residual add (#110)
    * csrc/hetero_reduce/cross_entropy_tp.cu           — TP cross-entropy loss (#110)
    * csrc/hetero_reduce/fused_gradient_allreduce.cu   — INT8 compressed ring allreduce
    * csrc/hetero_reduce/fused_adam_heterogeneous.cu   — per-tier LR-scaled Adam
    * csrc/hetero_reduce/binding.cpp                   — pybind11 / PyTorch extension glue

    Heterogeneous targets (all PCIe, no NVLink):
      SM 8.6  — RTX A6000  (48 GB VRAM)
      SM 9.0  — H100       (80 GB VRAM)
      SM 12.0 — Blackwell  (192 GB VRAM NVL72, requires CUDA ≥ 12.8)

    The sm_version argument passed to each launch_* function at *runtime*
    (not build time) selects the occupancy-tuned kernel variant for the
    active device.  This lets a single binary serve all three tiers.

    Exposed Python API (after import deepspeed; op = deepspeed.ops.op_builder.HeteroReduceBuilder().load())
    ----------------------------------------
    op.fused_bf16_reduce(output, inputs, sm_version)
    op.hetero_reduce_scatter(output, inputs, shard_offset, shard_count, sm_version)
    op.compute_shard_ranges(sm_versions, total_elems) -> List[Tuple[int,int]]
    op.fused_swiglu_ln(output, gate_proj, up_proj, ln_weight, eps, sm_version)
    op.rope_cache(cos_cache, sin_cache, seq_len, head_dim, base, pos_offset)
    op.fused_rope_hetero(output, input, cos_cache, sin_cache, neox_style, sm_version)
    op.pcie_ring_reduce(dst, src, sm_version)
    op.pcie_allreduce_finalise(out, src, world_size, sm_version)
    op.pcie_bucket_size(pcie_bw_gbps) -> int
    op.activation_pack(output, inputs, sm_version)
    op.activation_unpack(outputs, flat, sm_version)
    op.quantise_bf16_to_int8(output, scales, input)
    op.dequantise_int8_to_bf16(output, input, scales)
    op.compute_offload_budget(total_act_bytes, vram_free_bytes, headroom_frac) -> int
    op.fused_layernorm_residual(output, residual, input, ln_weight, eps, sm_version)
    op.cross_entropy_tp_forward(logits, labels, shard_offset, sm_version) -> Tuple[Tensor,Tensor,Tensor]
    op.cross_entropy_tp_loss(global_max, global_sum_exp, global_logit) -> Tensor
    op.cross_entropy_tp_backward(d_logits, logits, labels, global_max, log_sum_exp, shard_offset, inv_batch, sm_version)
    op.fused_gradient_allreduce(grad, int8_staging, scale_staging, ping_int8, pong_int8, ping_scale, pong_scale, rank, world_size, sm_version)
    op.gradient_compress(out_int8, out_scale, input, sm_version)
    op.gradient_decompress(output, int8_data, scale_buf, sm_version)
    op.int8_ring_reduce_step(dst_int8, dst_scale, src_int8, src_scale, sm_version)
    op.gradient_allreduce_finalise(scale_buf, n_elems, world_size)
    op.gradient_compress_bytes(n_elems) -> int
    op.gradient_scale_bytes(n_elems) -> int
    op.fused_adam_heterogeneous(params, exp_avg, exp_avg_sq, grads, lr_base, lr_scale, beta1, beta2, bc1, bc2, eps, weight_decay, sm_version, master_params)
    op.hetero_adam_lr_scale(sm_version) -> float
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
            # C++ pybind11 glue — compiled by the host CXX compiler.
            "csrc/hetero_reduce/binding.cpp",
            # CUDA kernels — compiled by nvcc.
            "csrc/hetero_reduce/hetero_reduce.cu",
            "csrc/hetero_reduce/fused_rope_hetero.cu",
            "csrc/hetero_reduce/pcie_adaptive_allreduce.cu",
            "csrc/hetero_reduce/fused_swiglu_ln.cu",
            "csrc/hetero_reduce/tier_activation_offload.cu",
            # Additional kernels (#110 / #134)
            "csrc/hetero_reduce/fused_layernorm_residual.cu",
            "csrc/hetero_reduce/cross_entropy_tp.cu",
            "csrc/hetero_reduce/fused_gradient_allreduce.cu",
            "csrc/hetero_reduce/fused_adam_heterogeneous.cu",
            # 5-GPU 2-NUMA heterogeneous ring allreduce (#143)
            "csrc/hetero_reduce/hetero_ring_allreduce.cu",
        ]

    def include_paths(self):
        return [
            # DeepSpeed shared headers: DS_D_INLINE, hw_warp_size, …
            "csrc/includes",
            # hetero_reduce.h (HeteroTierDesc, PcieGradChunk, launch_* declarations)
            "csrc/hetero_reduce",
        ]

    def cxx_args(self):
        # Inherit base CXX flags (C++17, position-independent, etc.) and add
        # version-dependent macros (-DBF16_AVAILABLE, -DVERSION_GE_1_1, …).
        args = super().cxx_args()
        return args + self.version_dependent_macros()

    def nvcc_args(self):
        # Start with optimisation level and version macros.
        nvcc_flags = ["-O3", "--expt-relaxed-constexpr"] + self.version_dependent_macros()

        if not self.is_rocm_pytorch():
            base_flags = [
                # Debug line info for nsight / cuda-gdb profiling.
                "-lineinfo",
                # Fused sincos, fast division, etc.
                "--use_fast_math",
                # Allow C++ 17 in device code (required for if-constexpr in templates).
                "-std=c++17",
                # ----------------------------------------------------------------
                # Code generation targets — one per supported SM generation.
                # ----------------------------------------------------------------
                # SM 8.6 — RTX A6000 (Ampere)
                "-gencode", "arch=compute_86,code=sm_86",
                # SM 9.0 — H100 (Hopper)
                "-gencode", "arch=compute_90,code=sm_90",
                # Forward-compatible PTX: covers SM 9.x variants and any future
                # architecture that the CUDA toolchain doesn't yet know about,
                # including SM 12.0 when built with CUDA < 12.8.
                "-gencode", "arch=compute_90,code=compute_90",
            ]

            # SM 12.0 (Blackwell) requires CUDA ≥ 12.8 for native SASS.
            # When available, add the native target for maximum performance;
            # the PTX fallback above still works with older toolchains.
            try:
                cuda_major, cuda_minor = self.installed_cuda_version()
                if cuda_major > 12 or (cuda_major == 12 and cuda_minor >= 8):
                    base_flags += ["-gencode", "arch=compute_120,code=sm_120"]
            except Exception:
                pass  # nvcc absent or version query failed; PTX fallback suffices.

            if sys.platform == "win32":
                base_flags = ["-allow-unsupported-compiler"] + base_flags

            nvcc_flags.extend(base_flags)
            # Remove any accidental empty strings that confuse nvcc.
            nvcc_flags = [f for f in nvcc_flags if f]

        return nvcc_flags

    def extra_ldflags(self):
        # No extra libraries beyond what PyTorch already links (libcuda, libcudart).
        return []

    def is_compatible(self, verbose=True):
        """hetero_reduce requires CUDA 11.0+ for BF16 and cooperative groups."""
        try:
            cuda_major, cuda_minor = self.installed_cuda_version()
            if cuda_major < 11:
                if verbose:
                    self.warning(
                        f"{self.NAME}: CUDA {cuda_major}.{cuda_minor} < 11.0; "
                        "BF16 and cooperative groups require CUDA ≥ 11.0."
                    )
                return False
        except Exception:
            pass  # Let the parent class handle the absence of nvcc.
        return super().is_compatible(verbose)
