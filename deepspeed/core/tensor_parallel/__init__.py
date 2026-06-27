# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Tensor parallelism layers."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from deepspeed.core.model_parallel_config import ModelParallelConfig


class VocabParallelEmbedding(nn.Module):
    """Embedding parallelized across TP ranks."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        config: ModelParallelConfig,
        init_method: Optional[callable] = None,
    ) -> None:
        raise NotImplementedError("Claude task: tensor_parallel")

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Claude task: tensor_parallel")


class ColumnParallelLinear(nn.Module):
    """Linear layer with column parallelism.

    Splits weight matrix along the output dimension across TP ranks.
    Each rank computes a partial result, then all-gather if needed.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: ModelParallelConfig,
        bias: bool = True,
        gather_output: bool = True,
        init_method: Optional[callable] = None,
        skip_bias_add: bool = False,
    ) -> None:
        raise NotImplementedError("Claude task: tensor_parallel")

    def forward(self, input_: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError("Claude task: tensor_parallel")


class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism.

    Splits weight matrix along the input dimension across TP ranks.
    Each rank computes a partial result, then all-reduce.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: ModelParallelConfig,
        bias: bool = True,
        input_is_parallel: bool = False,
        init_method: Optional[callable] = None,
        skip_bias_add: bool = False,
    ) -> None:
        raise NotImplementedError("Claude task: tensor_parallel")

    def forward(self, input_: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError("Claude task: tensor_parallel")
