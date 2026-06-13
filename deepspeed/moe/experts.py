# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# M1920: Megatron c13f08a11 — Remove duplicated gradient scaling for MoE weight
# Megatron's GroupedMLP used ScaleGradient (1/expert_parallel_world_size) to correct
# gradients under pure data parallelism vs expert model parallelism.  DDP already
# handles the averaging across data-parallel ranks, so an explicit ScaleGradient class
# creates a second (incorrect) scale and should not exist here.
# DES-LOC note: expert params carry param.allreduce=False; the DES-LOC Kx-gated
# allreduce logic in engine.py respects this flag and skips the redundant scale.

import copy
from typing import List, Optional

import torch
from torch import nn

# M1920: diagnostic step counter — prints every 50 forward passes
_experts_diag_step = 0


class Experts(nn.Module):

    def __init__(self, expert: nn.Module, num_local_experts: int = 1, expert_group_name: Optional[str] = None) -> None:
        super(Experts, self).__init__()

        self.deepspeed_experts = nn.ModuleList([copy.deepcopy(expert) for _ in range(num_local_experts)])
        self.num_local_experts = num_local_experts

        # M1920: expert params are NOT allreduced across dp_group — DDP already averages
        # gradients for shared (non-expert) params; expert params use ep_group instead.
        # No manual gradient scaling (ScaleGradient) is needed here.
        for expert in self.deepspeed_experts:
            # TODO: Create param groups to handle expert + data case (e.g. param.group = moe_group)
            for param in expert.parameters():
                param.allreduce = False
                param.group_name = expert_group_name

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        global _experts_diag_step
        _experts_diag_step += 1

        # M1920 DES-LOC diag: log input shape + expert count every 50 steps
        if _experts_diag_step % 50 == 1:
            print(
                f"[M1920 Experts.forward step={_experts_diag_step}] "
                f"inputs.shape={tuple(inputs.shape)} num_local_experts={self.num_local_experts} "
                f"dtype={inputs.dtype} device={inputs.device} "
                f"no_grad_scale=True (DDP handles averaging, Megatron c13f08a11 aligned)"
            )

        chunks = inputs.chunk(self.num_local_experts, dim=1)
        expert_outputs: List[torch.Tensor] = []

        for chunk, expert in zip(chunks, self.deepspeed_experts):
            out = expert(chunk)
            if isinstance(out, tuple):
                out = out[0]  # Ignore the bias term for now
            expert_outputs += [out]

        result = torch.cat(expert_outputs, dim=1)

        # M1920 DES-LOC diag: log output norm every 50 steps to detect gradient explosion
        if _experts_diag_step % 50 == 1:
            print(
                f"[M1920 Experts.forward step={_experts_diag_step}] "
                f"output.shape={tuple(result.shape)} output_norm={result.detach().float().norm().item():.4f}"
            )

        return result
