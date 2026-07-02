# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""
deepspeed.ops.hetero_reduce
===========================

Fused CUDA kernels for heterogeneous GPU gradient synchronisation.

  * fused_bf16_reduce  — BF16→FP32 reduce + FP32→BF16 writeback across tensors
  * fused_swiglu_ln    — SwiGLU activation fused with RMS LayerNorm

Targets SM 8.6 (A6000), SM 9.0 (H100), SM 12.0 (Blackwell) over pure PCIe.
"""

from .hetero_reduce_op import HeteroReduceOp

__all__ = ["HeteroReduceOp"]
