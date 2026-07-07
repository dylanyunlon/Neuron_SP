# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""
HeteroReduceOp — lazy-loading wrapper around the hetero_reduce CUDA extension.

The extension is JIT-compiled on first use if it was not pre-built during
``pip install deepspeed``.
"""

import torch

# Compiled CUDA module, loaded on first call.
_hetero_reduce_module = None


def _load_module():
    global _hetero_reduce_module
    if _hetero_reduce_module is None:
        # Try to import from the installed deepspeed package first; fall back
        # to the top-level op_builder used during development / editable installs.
        try:
            from deepspeed.ops.op_builder import HeteroReduceBuilder
        except ImportError:
            import sys
            import os
            # Resolve repo root (two levels above this file's directory).
            _repo_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            )
            if _repo_root not in sys.path:
                sys.path.insert(0, _repo_root)
            from op_builder.hetero_reduce import HeteroReduceBuilder  # noqa: F401

        _hetero_reduce_module = HeteroReduceBuilder().load()
    return _hetero_reduce_module


class HeteroReduceOp:
    """
    Thin Python wrapper for the fused hetero-reduce CUDA kernels.

    Covers:
      - fused_bf16_reduce / hetero_reduce_scatter / compute_shard_ranges
      - fused_swiglu_ln / fused_layernorm_residual / fused_layernorm_residual_ex
      - rope_cache / fused_rope_hetero / fused_rope_cacheless
      - pcie_ring_reduce / pcie_allreduce_finalise / pcie_ring_reduce_step
      - hetero_ring_reduce_step / hetero_ring_gather_step (ring allreduce, #143)
      - fused_adam_heterogeneous / hetero_adam_lr_scale
      - cross_entropy_tp_forward / cross_entropy_tp_loss / cross_entropy_tp_backward
      - compute_hetero_vocab_partition / cross_entropy_tp_{forward,backward}_hetero
      - activation_pack / activation_unpack / quantise_bf16_to_int8 / dequantise_int8_to_bf16
      - compute_offload_budget / pcie_bucket_size / probe_pcie_bandwidth
      - hetero_bucket_size_elems / compute_adaptive_chunk_size
      - grad_norm_sq_fp8

    Usage
    -----
    >>> import torch
    >>> from deepspeed.ops.hetero_reduce import HeteroReduceOp
    >>> op = HeteroReduceOp()
    >>> grads = [torch.randn(1024, dtype=torch.bfloat16, device='cuda') for _ in range(4)]
    >>> out = torch.zeros(1024, dtype=torch.bfloat16, device='cuda')
    >>> op.fused_bf16_reduce(out, grads, sm_version=86)
    """

    def fused_bf16_reduce(self,
                           output: torch.Tensor,
                           inputs: list,
                           sm_version: int = 86) -> None:
        """
        Reduce a list of BF16 gradient tensors into *output* with FP32 accumulation.

        All tensors must be on the same CUDA device.  The caller is responsible for
        cross-device staging when orchestrating the full reduce-scatter.

        Parameters
        ----------
        output     : BF16 CUDA tensor, shape ``[N]``.  Written in-place.
        inputs     : list[BF16 CUDA tensor], each shape ``[N]``.
        sm_version : SM version of the current device (86, 90, 120, …).
        """
        _load_module().fused_bf16_reduce(output, inputs, sm_version)

    def hetero_reduce_scatter(self,
                               output: torch.Tensor,
                               inputs: list,
                               shard_offset: int,
                               shard_count: int,
                               sm_version: int = 86) -> None:
        """Heterogeneous reduce-scatter: reduce all inputs, write only local shard."""
        _load_module().hetero_reduce_scatter(output, inputs, shard_offset, shard_count, sm_version)

    def compute_shard_ranges(self, sm_versions: list, total_elems: int) -> list:
        """Compute non-uniform per-tier shard (offset, count) pairs."""
        return _load_module().compute_shard_ranges(sm_versions, total_elems)

    def fused_swiglu_ln(self,
                         output: torch.Tensor,
                         gate_proj: torch.Tensor,
                         up_proj: torch.Tensor,
                         ln_weight: torch.Tensor,
                         eps: float = 1e-6,
                         sm_version: int = 86) -> None:
        """
        Compute SwiGLU(gate_proj, up_proj) + RMS LayerNorm in one kernel.

        Parameters
        ----------
        output    : BF16 CUDA ``[batch, hidden]`` — result written here.
        gate_proj : BF16 CUDA ``[batch, hidden]``.
        up_proj   : BF16 CUDA ``[batch, hidden]``.
        ln_weight : FP32 CUDA ``[hidden]``  — RMSNorm weight (gamma).
        eps       : float — RMSNorm epsilon (default 1e-6).
        sm_version: int  — device SM version: 86, 90, or 120.
        """
        _load_module().fused_swiglu_ln(output, gate_proj, up_proj, ln_weight, eps, sm_version)

    def fused_layernorm_residual(self,
                                  output: torch.Tensor,
                                  residual: torch.Tensor,
                                  input: torch.Tensor,
                                  ln_weight: torch.Tensor,
                                  eps: float = 1e-6,
                                  sm_version: int = 86) -> None:
        """Fused residual add + RMS LayerNorm (pre-LN Llama/Mistral style)."""
        _load_module().fused_layernorm_residual(output, residual, input, ln_weight, eps, sm_version)

    def fused_layernorm_residual_ex(self,
                                     output: torch.Tensor,
                                     residual: torch.Tensor,
                                     input: torch.Tensor,
                                     ln_weight: torch.Tensor,
                                     eps: float = 1e-6,
                                     full_ln: bool = False,
                                     sm_version: int = 86,
                                     bias=None,
                                     output_fp32=None) -> None:
        """Extended fused residual + LN: full LN or RMSNorm, optional bias/FP32 out."""
        _load_module().fused_layernorm_residual_ex(
            output, residual, input, ln_weight, eps, full_ln, sm_version, bias, output_fp32)

    # -----------------------------------------------------------------------
    # RoPE
    # -----------------------------------------------------------------------
    def rope_cache(self,
                   cos_cache: torch.Tensor,
                   sin_cache: torch.Tensor,
                   seq_len: int,
                   head_dim: int,
                   base: float = 10000.0,
                   pos_offset: int = 0) -> None:
        """Precompute RoPE cos/sin tables on device."""
        _load_module().rope_cache(cos_cache, sin_cache, seq_len, head_dim, base, pos_offset)

    def fused_rope_hetero(self,
                           output: torch.Tensor,
                           input: torch.Tensor,
                           cos_cache: torch.Tensor,
                           sin_cache: torch.Tensor,
                           neox_style: bool = True,
                           sm_version: int = 86) -> None:
        """Fused RoPE for heterogeneous head counts."""
        _load_module().fused_rope_hetero(output, input, cos_cache, sin_cache, neox_style, sm_version)

    def fused_rope_cacheless(self,
                              output: torch.Tensor,
                              input: torch.Tensor,
                              base: float = 10000.0,
                              pos_offset: int = 0,
                              neox_style: bool = True,
                              sm_version: int = 86) -> None:
        """RoPE with on-the-fly sin/cos (no precomputed cache); for very long sequences."""
        _load_module().fused_rope_cacheless(output, input, base, pos_offset, neox_style, sm_version)

    # -----------------------------------------------------------------------
    # PCIe allreduce
    # -----------------------------------------------------------------------
    def pcie_ring_reduce(self, dst: torch.Tensor, src: torch.Tensor, sm_version: int = 86) -> None:
        """PCIe ring reduce step: dst += src (BF16, in-place)."""
        _load_module().pcie_ring_reduce(dst, src, sm_version)

    def pcie_allreduce_finalise(self,
                                 out: torch.Tensor,
                                 src: torch.Tensor,
                                 world_size: int,
                                 sm_version: int = 86) -> None:
        """Divide allreduce sum by world_size and write BF16 output."""
        _load_module().pcie_allreduce_finalise(out, src, world_size, sm_version)

    def pcie_ring_reduce_step(self,
                               accum_buf: torch.Tensor,
                               recv_buf: torch.Tensor,
                               sm_version: int = 86) -> None:
        """Double-buffered ring reduce step: accum_buf += recv_buf."""
        _load_module().pcie_ring_reduce_step(accum_buf, recv_buf, sm_version)

    def pcie_bucket_size(self, pcie_bw_gbps: float = 32.0) -> int:
        """Recommended gradient bucket size in bytes for given PCIe bandwidth."""
        return _load_module().pcie_bucket_size(pcie_bw_gbps)

    def probe_pcie_bandwidth(self, src_device: int, dst_device: int) -> float:
        """Measure PCIe bandwidth between two CUDA devices (GB/s)."""
        return _load_module().probe_pcie_bandwidth(src_device, dst_device)

    # -----------------------------------------------------------------------
    # Heterogeneous ring allreduce kernels (#143)
    # -----------------------------------------------------------------------
    def hetero_ring_reduce_step(self,
                                 accum_buf: torch.Tensor,
                                 recv_buf: torch.Tensor,
                                 sm_version: int = 86) -> None:
        """
        Single reduce-scatter step in a heterogeneous PCIe ring.

        accum_buf[i] += recv_buf[i]  (BF16 → FP32 accumulation → BF16).
        Dispatches SM-specialised kernel: SM8.6, SM9.0, or SM12.0 cp.async.

        Parameters
        ----------
        accum_buf  : BF16 CUDA tensor ``[chunk_elems]`` — accumulator, modified in-place.
        recv_buf   : BF16 CUDA tensor ``[chunk_elems]`` — received from ring predecessor.
        sm_version : SM version of active device (86, 90, 120).
        """
        _load_module().hetero_ring_reduce_step(accum_buf, recv_buf, sm_version)

    def hetero_ring_gather_step(self,
                                 output: torch.Tensor,
                                 recv_buf: torch.Tensor,
                                 sm_version: int = 86) -> None:
        """
        Single all-gather step in a heterogeneous PCIe ring.

        output[i] = recv_buf[i]  (vectorised copy, no accumulation).
        Used in the gather phase after reduce-scatter completes.

        Parameters
        ----------
        output     : BF16 CUDA tensor ``[chunk_elems]`` — destination buffer.
        recv_buf   : BF16 CUDA tensor ``[chunk_elems]`` — fully-reduced chunk.
        sm_version : SM version of active device (86, 90, 120).
        """
        _load_module().hetero_ring_gather_step(output, recv_buf, sm_version)

    def hetero_ring_intra_numa_chunk_bytes(self) -> int:
        """Intra-NUMA chunk size in bytes (4 MB, targets ~0.5 ms at 32 GB/s PCIe 4.0 x16)."""
        return _load_module().hetero_ring_intra_numa_chunk_bytes()

    def hetero_ring_cross_numa_chunk_bytes(self) -> int:
        """Cross-NUMA chunk size in bytes (2 MB, targets ~0.5 ms at 16 GB/s cross-switch PCIe)."""
        return _load_module().hetero_ring_cross_numa_chunk_bytes()

    def hetero_ring_max_chunk_bytes(self) -> int:
        """Maximum chunk size in bytes; allocate ping/pong buffers with at least this size."""
        return _load_module().hetero_ring_max_chunk_bytes()

    def hetero_ring_sm_block_size(self, sm_version: int = 86) -> int:
        """Thread-block size for hetero ring kernels (SM8.6→128, SM9.0→256, SM12.0→512)."""
        return _load_module().hetero_ring_sm_block_size(sm_version)

    # -----------------------------------------------------------------------
    # Adam optimizer
    # -----------------------------------------------------------------------
    def fused_adam_heterogeneous(self,
                                  params: torch.Tensor,
                                  exp_avg: torch.Tensor,
                                  exp_avg_sq: torch.Tensor,
                                  grads: torch.Tensor,
                                  lr_base: float,
                                  lr_scale: float,
                                  beta1: float = 0.9,
                                  beta2: float = 0.999,
                                  bc1: float = 1.0,
                                  bc2: float = 1.0,
                                  eps: float = 1e-8,
                                  weight_decay: float = 0.0,
                                  sm_version: int = 86,
                                  master_params: torch.Tensor = None) -> None:
        """Fused AdamW with per-tier LR scaling. BF16 params/grads, FP32 moments."""
        mp = master_params if master_params is not None else torch.Tensor()
        _load_module().fused_adam_heterogeneous(
            params, exp_avg, exp_avg_sq, grads,
            lr_base, lr_scale, beta1, beta2, bc1, bc2, eps, weight_decay,
            sm_version, mp)

    def hetero_adam_lr_scale(self, sm_version: int) -> float:
        """Default per-tier LR scale (SM12.0→4.0, SM9.0→3.0, SM8.6→1.0)."""
        return _load_module().hetero_adam_lr_scale(sm_version)

    # -----------------------------------------------------------------------
    # Cross-entropy TP
    # -----------------------------------------------------------------------
    def cross_entropy_tp_forward(self,
                                  logits: torch.Tensor,
                                  labels: torch.Tensor,
                                  shard_offset: int = 0,
                                  sm_version: int = 86):
        """Phase-1 TP cross-entropy: returns (local_max, local_sum_exp, local_logit)."""
        return _load_module().cross_entropy_tp_forward(logits, labels, shard_offset, sm_version)

    def cross_entropy_tp_loss(self,
                               global_max: torch.Tensor,
                               global_sum_exp: torch.Tensor,
                               global_logit: torch.Tensor) -> torch.Tensor:
        """Phase-2 TP cross-entropy: per-sample CE loss from globally-reduced scalars."""
        return _load_module().cross_entropy_tp_loss(global_max, global_sum_exp, global_logit)

    def cross_entropy_tp_backward(self,
                                   d_logits: torch.Tensor,
                                   logits: torch.Tensor,
                                   labels: torch.Tensor,
                                   global_max: torch.Tensor,
                                   log_sum_exp: torch.Tensor,
                                   shard_offset: int = 0,
                                   inv_batch: float = 1.0,
                                   sm_version: int = 86) -> None:
        """TP cross-entropy backward: softmax gradient w.r.t. local logit shard."""
        _load_module().cross_entropy_tp_backward(
            d_logits, logits, labels, global_max, log_sum_exp,
            shard_offset, inv_batch, sm_version)

    def compute_hetero_vocab_partition(self, sm_versions: list, vocab_size: int) -> list:
        """Compute per-rank VocabPartition for non-uniform TP vocab split."""
        return _load_module().compute_hetero_vocab_partition(sm_versions, vocab_size)

    # -----------------------------------------------------------------------
    # Activation offload
    # -----------------------------------------------------------------------
    def activation_pack(self,
                         output: torch.Tensor,
                         inputs: list,
                         sm_version: int = 86) -> None:
        """Pack activation tensors into a flat BF16 offload buffer."""
        _load_module().activation_pack(output, inputs, sm_version)

    def activation_unpack(self,
                           outputs: list,
                           flat: torch.Tensor,
                           sm_version: int = 86) -> None:
        """Unpack a flat BF16 buffer back to individual activation tensors."""
        _load_module().activation_unpack(outputs, flat, sm_version)

    def compute_offload_budget(self,
                                total_act_bytes: int,
                                vram_free_bytes: int,
                                headroom_frac: float = 0.1) -> int:
        """Bytes of activations to offload to host/peer given free VRAM."""
        return _load_module().compute_offload_budget(total_act_bytes, vram_free_bytes, headroom_frac)

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------
    def hetero_bucket_size_elems(self, sm_version: int = 86) -> int:
        """Policy-recommended gradient bucket size in BF16 elements for this SM version."""
        return _load_module().hetero_bucket_size_elems(sm_version)

    def compute_adaptive_chunk_size(self, pcie_bw_gbps: float = 32.0) -> int:
        """Adaptive ring chunk size targeting 5 ms of PCIe transfer per step."""
        return _load_module().compute_adaptive_chunk_size(pcie_bw_gbps)
