# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""
deepspeed.ops.hetero_reduce
===========================

Fused CUDA kernels for heterogeneous GPU gradient synchronisation across
mixed SM 8.6 (A6000) / SM 9.0 (H100) / SM 12.0 (Blackwell) PCIe clusters.

Kernel groups
-------------
Gradient reduction
  HeteroReduceOp.fused_bf16_reduce       — BF16→FP32 reduce + BF16 writeback
  HeteroReduceOp.hetero_reduce_scatter   — non-uniform shard reduce-scatter
  HeteroReduceOp.compute_shard_ranges    — heterogeneous shard layout helper

PCIe ring allreduce (issue #143)
  HeteroReduceOp.pcie_ring_reduce        — ring reduce step (dst += src)
  HeteroReduceOp.pcie_allreduce_finalise — post-ring averaging
  HeteroReduceOp.pcie_ring_reduce_step   — double-buffered ring reduce step

Heterogeneous ring allreduce (issue #143)
  HeteroReduceOp.hetero_ring_reduce_step — 5-GPU 2-NUMA ring reduce step
  HeteroReduceOp.hetero_ring_gather_step — 5-GPU 2-NUMA ring gather step

Fused activations
  HeteroReduceOp.fused_swiglu_ln                — SwiGLU + RMSNorm
  HeteroReduceOp.fused_layernorm_residual        — residual + RMSNorm (pre-LN)
  HeteroReduceOp.fused_layernorm_residual_ex     — extended: full LN / bias / FP32 out

RoPE
  HeteroReduceOp.rope_cache                 — precompute cos/sin tables
  HeteroReduceOp.fused_rope_hetero          — cached RoPE for hetero head counts
  HeteroReduceOp.fused_rope_cacheless       — on-the-fly RoPE (long sequences)

Optimizer
  HeteroReduceOp.fused_adam_heterogeneous   — per-tier LR-scaled AdamW
  HeteroReduceOp.hetero_adam_lr_scale       — default tier LR multiplier

Cross-entropy TP
  HeteroReduceOp.cross_entropy_tp_forward   — local (max, sum_exp, logit) pass
  HeteroReduceOp.cross_entropy_tp_loss      — CE loss from reduced scalars
  HeteroReduceOp.cross_entropy_tp_backward  — softmax gradient for TP shard
  HeteroReduceOp.compute_hetero_vocab_partition — non-uniform vocab split

Activation offload
  HeteroReduceOp.activation_pack / activation_unpack
  HeteroReduceOp.compute_offload_budget

Utilities
  HeteroReduceOp.hetero_bucket_size_elems
  HeteroReduceOp.compute_adaptive_chunk_size
  HeteroReduceOp.probe_pcie_bandwidth / pcie_bucket_size
"""

from .hetero_reduce_op import HeteroReduceOp

# Re-export HeteroReduceBuilder so callers can do:
#   from deepspeed.ops.hetero_reduce import HeteroReduceBuilder
try:
    from deepspeed.ops.op_builder.hetero_reduce import HeteroReduceBuilder
except ImportError:
    try:
        from op_builder.hetero_reduce import HeteroReduceBuilder
    except ImportError:
        HeteroReduceBuilder = None

__all__ = ["HeteroReduceOp", "HeteroReduceBuilder"]
