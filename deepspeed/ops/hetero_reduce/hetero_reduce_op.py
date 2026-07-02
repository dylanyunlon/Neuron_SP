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
