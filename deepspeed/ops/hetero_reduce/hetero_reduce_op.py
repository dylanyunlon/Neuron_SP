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

    # Issue #124: TP-aware fused cross-entropy completions
    def cross_entropy_tp_log_finalise(self, global_sum_exp: torch.Tensor) -> None:
        """Issue #124: In-place log(global_sum_exp) → log_sum_exp for backward.

        Call AFTER dist.all_reduce(local_sum_exp, op=sum) to produce the
        log_sum_exp tensor required by cross_entropy_tp_backward.

        Args:
            global_sum_exp: FP32 Tensor [batch], modified in-place.
        """
        _load_module().cross_entropy_tp_log_finalise(global_sum_exp)

    def cross_entropy_tp_forward_with_log(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        shard_offset: int = 0,
        sm_version: int = 86,
    ):
        """Issue #124: Forward pass returning (max, sum_exp, log_sum, logit).

        Returns a 4-tuple:
            local_max      (FP32 [batch]) — AllReduce_max across TP ranks
            local_sum_exp  (FP32 [batch]) — AllReduce_sum across TP ranks
            local_log_sum  (FP32 [batch]) — log(local_sum_exp); valid for
                           tp_size=1 without further AllReduce.  For tp>1:
                           AllReduce sum_exp, then call cross_entropy_tp_log_finalise.
            local_logit    (FP32 [batch]) — AllReduce_sum across TP ranks
        """
        return _load_module().cross_entropy_tp_forward_with_log(
            logits, labels, shard_offset, sm_version)

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
    # Fused gradient allreduce (INT8-compressed ring, #146)
    # -----------------------------------------------------------------------

    def gradient_compress(self,
                           output_int8: torch.Tensor,
                           output_scale: torch.Tensor,
                           input: torch.Tensor,
                           sm_version: int = 86) -> None:
        """Compress a BF16 gradient tensor to INT8 with per-block FP32 scales.

        Parameters
        ----------
        output_int8  : Int8  CUDA tensor ``[N]``   — compressed output (written in-place).
        output_scale : FP32  CUDA tensor ``[ceil(N/256)]`` — per-block ℓ∞/127 scales.
        input        : BF16  CUDA tensor ``[N]``   — source gradient.
        sm_version   : SM version of active device (86, 90, 120).
        """
        _load_module().gradient_compress(output_int8, output_scale, input, sm_version)

    def int8_ring_reduce_step(self,
                               dst_int8: torch.Tensor,
                               dst_scale: torch.Tensor,
                               src_int8: torch.Tensor,
                               src_scale: torch.Tensor,
                               sm_version: int = 86) -> None:
        """One ring-allreduce accumulation step: fused INT8 dequant + sum + re-quant.

        dst_int8/dst_scale are modified in-place to hold the accumulated result
        after absorbing src_int8/src_scale.  The rescaled sum is stored back as
        INT8 so compression is maintained across all ring steps.

        Parameters
        ----------
        dst_int8  : Int8 CUDA tensor ``[N]`` — accumulator, modified in-place.
        dst_scale : FP32 CUDA tensor ``[ceil(N/256)]`` — per-block scales for dst.
        src_int8  : Int8 CUDA tensor ``[N]`` — received chunk from ring predecessor.
        src_scale : FP32 CUDA tensor ``[ceil(N/256)]`` — per-block scales for src.
        sm_version: SM version of active device (86, 90, 120).
        """
        _load_module().int8_ring_reduce_step(dst_int8, dst_scale, src_int8, src_scale, sm_version)

    def gradient_decompress(self,
                             output: torch.Tensor,
                             int8_data: torch.Tensor,
                             scale_buf: torch.Tensor,
                             sm_version: int = 86) -> None:
        """Decompress INT8 + per-block FP32 scales back to BF16.

        Parameters
        ----------
        output   : BF16  CUDA tensor ``[N]`` — decompressed gradient (written in-place).
        int8_data: Int8  CUDA tensor ``[N]`` — compressed INT8 values.
        scale_buf: FP32  CUDA tensor ``[ceil(N/256)]`` — per-block ℓ∞/127 scales.
        sm_version: SM version (86, 90, 120).
        """
        _load_module().gradient_decompress(output, int8_data, scale_buf, sm_version)

    def gradient_allreduce_finalise(self,
                                     scale_buf: torch.Tensor,
                                     n_elems: int,
                                     world_size: int) -> None:
        """Divide per-block scales by world_size after the ring reduce completes.

        Must be called once on each rank after the reduce-scatter + gather
        phases so that the decompressed gradient is correctly averaged.

        Parameters
        ----------
        scale_buf  : FP32 CUDA tensor ``[ceil(n_elems/256)]`` — scales (modified in-place).
        n_elems    : int — total number of BF16 gradient elements.
        world_size : int — number of participating ranks.
        """
        _load_module().gradient_allreduce_finalise(scale_buf, n_elems, world_size)

    def gradient_compress_bytes(self, n_elems: int) -> int:
        """Return the INT8 staging buffer size in bytes for *n_elems* BF16 elements."""
        return _load_module().gradient_compress_bytes(n_elems)

    def gradient_scale_bytes(self, n_elems: int) -> int:
        """Return the per-block FP32 scale buffer size in bytes for *n_elems* elements."""
        return _load_module().gradient_scale_bytes(n_elems)

    def fused_gradient_allreduce(self,
                                  grad: torch.Tensor,
                                  group: 'torch.distributed.ProcessGroup',
                                  sm_version: int = 86) -> None:
        """High-level helper: INT8-compressed ring allreduce of a single BF16 gradient.

        Allocates staging buffers, runs compress → ring-reduce → finalise →
        decompress in sequence.  For multi-tensor callers prefer the lower-level
        ``gradient_compress / int8_ring_reduce_step / gradient_decompress`` API
        to amortise buffer allocation across gradient tensors.

        Parameters
        ----------
        grad       : BF16 CUDA tensor of arbitrary shape (treated as flat).
        group      : process group for dist.send/recv during the ring steps.
        sm_version : SM version of active device (86, 90, 120).
        """
        import torch.distributed as dist

        world_size = dist.get_world_size(group=group)
        if world_size <= 1:
            return  # nothing to reduce

        flat = grad.view(-1)
        n = flat.numel()
        device = flat.device

        int8_self = torch.empty(n, dtype=torch.int8, device=device)
        scale_self = torch.empty((n + 255) // 256, dtype=torch.float32, device=device)
        int8_recv = torch.empty(n, dtype=torch.int8, device=device)
        scale_recv = torch.empty_like(scale_self)

        # Phase 1: compress local gradient
        self.gradient_compress(int8_self, scale_self, flat, sm_version)

        rank = dist.get_rank(group=group)
        # Ring reduce-scatter: (world_size - 1) steps
        for step in range(world_size - 1):
            send_to = (rank + 1) % world_size
            recv_from = (rank - 1) % world_size
            send_req_data = dist.isend(int8_self, dst=send_to, group=group, tag=step * 2)
            send_req_scale = dist.isend(scale_self, dst=send_to, group=group, tag=step * 2 + 1)
            dist.recv(int8_recv, src=recv_from, group=group, tag=step * 2)
            dist.recv(scale_recv, src=recv_from, group=group, tag=step * 2 + 1)
            send_req_data.wait()
            send_req_scale.wait()
            self.int8_ring_reduce_step(int8_self, scale_self, int8_recv, scale_recv, sm_version)

        # Finalise: divide scales by world_size
        self.gradient_allreduce_finalise(scale_self, n, world_size)

        # Phase 3: decompress back to BF16 in-place
        self.gradient_decompress(flat, int8_self, scale_self, sm_version)

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------
    def hetero_bucket_size_elems(self, sm_version: int = 86) -> int:
        """Policy-recommended gradient bucket size in BF16 elements for this SM version."""
        return _load_module().hetero_bucket_size_elems(sm_version)

    def compute_adaptive_chunk_size(self, pcie_bw_gbps: float = 32.0) -> int:
        """Adaptive ring chunk size targeting 5 ms of PCIe transfer per step."""
        return _load_module().compute_adaptive_chunk_size(pcie_bw_gbps)
