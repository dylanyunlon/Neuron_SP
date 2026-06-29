"""Gradient clipping for heterogeneous GPU training.

Adapted from Megatron megatron/core/optimizer/clip_grads.py.
Key change from Megatron M2335: removes host/device synchronization in
get_grad_norm_fp32 — critical for PCIe-only topologies where sync is costly.

No apex/TE dependency. Uses pure PyTorch for portability across SM86/SM90.
"""
from __future__ import annotations

from typing import List, Optional, Union

import torch
from torch import inf

import deepspeed.core.parallel_state as parallel_state


def get_grad_norm_fp32(
    grads_for_norm: Union[List[torch.Tensor], torch.Tensor],
    norm_type: Union[int, float] = 2,
    grad_stats_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Calculate the p-norm of gradients in FP32 — fully on device, no sync.

    Returns a CUDA tensor (not .item()) to avoid host/device synchronization.
    This is the M2335 pattern: desloc_engine can compare grad_norm > clip_value
    entirely on GPU.

    Args:
        grads_for_norm: Gradient tensors to compute norm over.
        norm_type: Type of p-norm (2 for L2, inf for max).
        grad_stats_parallel_group: Process group for reducing norm across
            model-parallel ranks. If None, no reduction.

    Returns:
        total_norm: Scalar CUDA tensor with the gradient norm.
    """
    if isinstance(grads_for_norm, torch.Tensor):
        grads_for_norm = [grads_for_norm]

    norm_type = float(norm_type)

    if norm_type == inf:
        if grads_for_norm:
            total_norm = torch.stack(
                [g.detach().abs().max() for g in grads_for_norm]
            ).max()
        else:
            total_norm = torch.tensor(0.0, device="cuda")

        if grad_stats_parallel_group is not None:
            total_norm = total_norm.unsqueeze(0)
            torch.distributed.all_reduce(
                total_norm,
                op=torch.distributed.ReduceOp.MAX,
                group=grad_stats_parallel_group,
            )
            total_norm = total_norm.squeeze(0)

    elif norm_type == 2.0:
        # Fused L2 norm: sum of squares, then sqrt after allreduce
        if grads_for_norm:
            total_norm_sq = torch.stack(
                [g.detach().float().norm(2.0).square() for g in grads_for_norm]
            ).sum()
        else:
            total_norm_sq = torch.tensor(0.0, device="cuda")

        if grad_stats_parallel_group is not None:
            total_norm_sq = total_norm_sq.unsqueeze(0)
            torch.distributed.all_reduce(
                total_norm_sq,
                op=torch.distributed.ReduceOp.SUM,
                group=grad_stats_parallel_group,
            )
            total_norm_sq = total_norm_sq.squeeze(0)

        total_norm = total_norm_sq.sqrt()

    else:
        # General p-norm
        if grads_for_norm:
            total_norm_p = torch.stack(
                [g.detach().float().norm(norm_type).pow(norm_type) for g in grads_for_norm]
            ).sum()
        else:
            total_norm_p = torch.tensor(0.0, device="cuda")

        if grad_stats_parallel_group is not None:
            total_norm_p = total_norm_p.unsqueeze(0)
            torch.distributed.all_reduce(
                total_norm_p,
                op=torch.distributed.ReduceOp.SUM,
                group=grad_stats_parallel_group,
            )
            total_norm_p = total_norm_p.squeeze(0)

        total_norm = total_norm_p.pow(1.0 / norm_type)

    return total_norm


def clip_grad_norm(
    parameters: Union[List[torch.nn.Parameter], torch.nn.Parameter],
    max_norm: float,
    norm_type: float = 2.0,
    grad_stats_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Clip gradients by total norm — replaces torch.nn.utils.clip_grad_norm_.

    Difference from torch version: computes norm across model-parallel group
    and avoids host/device sync (stays on GPU).

    Args:
        parameters: Model parameters whose gradients to clip.
        max_norm: Maximum allowed gradient norm.
        norm_type: Type of norm (default L2).
        grad_stats_parallel_group: Process group for norm reduction.

    Returns:
        total_norm: The gradient norm before clipping (CUDA tensor).
    """
    if isinstance(parameters, torch.nn.Parameter):
        parameters = [parameters]

    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0, device="cuda")

    total_norm = get_grad_norm_fp32(grads, norm_type, grad_stats_parallel_group)

    # Clip: scale grads by (max_norm / total_norm) if total_norm > max_norm
    clip_coeff = max_norm / (total_norm + 1e-6)
    clip_coeff = torch.clamp(clip_coeff, max=1.0)
    for grad in grads:
        grad.mul_(clip_coeff)

    return total_norm
