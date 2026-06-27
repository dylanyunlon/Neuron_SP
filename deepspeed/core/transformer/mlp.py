# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""MLP module — SwiGLU feed-forward network with TP sharding.

Ported from Megatron-LM with the following adaptations:

* Self-contained: does not depend on Megatron's ``tensor_parallel`` module.
* TP sharding via standard ``nn.Linear`` with parameter attributes.
* DES-LOC: each MLP instance logs its tier assignment at init time.

Architecture
------------
::

    hidden_states  →  gate_proj  ─┐
                    →  up_proj   ─┤─→ SiLU(gate) * up  →  down_proj  →  output
                                   └─────────────────────────────────────────────

Both ``gate_proj`` and ``up_proj`` are *column-parallel* (output split across
TP ranks).  ``down_proj`` is *row-parallel* (input split, all-reduce on output).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_config import TransformerConfig
from .module import MegatronModule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy parallel-state helpers
# ---------------------------------------------------------------------------

def _get_tp_world_size() -> int:
    try:
        from deepspeed.core.parallel_state import get_tensor_model_parallel_world_size
        return get_tensor_model_parallel_world_size()
    except Exception:
        return 1


def _get_tp_group():
    try:
        from deepspeed.core.parallel_state import get_tensor_model_parallel_group
        return get_tensor_model_parallel_group()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(MegatronModule):
    """SwiGLU MLP with tensor-parallel sharding.

    Structure::

        gate_proj: Linear(hidden,  ffn/tp)   — column-parallel
        up_proj:   Linear(hidden,  ffn/tp)   — column-parallel
        down_proj: Linear(ffn/tp,  hidden)   — row-parallel + all-reduce

    When TP > 1 the output dimension of gate/up is split across TP ranks
    and down_proj's input is likewise split.  An all-reduce is applied to
    the down_proj output to sum contributions from all TP ranks.

    DES-LOC integration:
        MLP does not hold a ``layer_number`` by itself (it is owned by
        TransformerLayer).  The parent TransformerLayer is responsible for
        tier-logging.  If needed, callers can pass ``layer_number`` as a
        constructor argument to enable per-MLP logging.

    Args:
        config: TransformerConfig driving hidden sizes and dropout.
        layer_number: Optional 1-based global layer index for DES-LOC
            logging.  Pass 0 (default) to skip tier logging.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int = 0,
    ) -> None:
        super().__init__(config)
        self.layer_number = layer_number

        tp_size = _get_tp_world_size()
        ffn_per_tp = config.ffn_hidden_size // tp_size

        # --- Column-parallel gate/up projections --------------------------
        self.gate_proj = nn.Linear(
            config.hidden_size, ffn_per_tp, bias=config.add_bias_linear
        )
        self.up_proj = nn.Linear(
            config.hidden_size, ffn_per_tp, bias=config.add_bias_linear
        )

        # --- Row-parallel down projection ---------------------------------
        self.down_proj = nn.Linear(
            ffn_per_tp, config.hidden_size, bias=config.add_bias_linear
        )

        # Mark TP-sharded parameters for checkpoint tooling
        for proj in (self.gate_proj, self.up_proj):
            proj.weight.tensor_model_parallel = True
            proj.weight.partition_dim = 0   # output dimension sharded

        self.down_proj.weight.tensor_model_parallel = True
        self.down_proj.weight.partition_dim = 1  # input dimension sharded

        self.activation_func: Callable = config.activation_func

        # DES-LOC tier log
        if layer_number > 0:
            tier = config.get_layer_tier(layer_number - 1)
            if tier is not None:
                logger.debug(
                    "MLP layer %d assigned to tier: %s", layer_number, tier.upper()
                )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """SwiGLU forward pass.

        Args:
            hidden_states: ``[..., hidden_size]``

        Returns:
            output: ``[..., hidden_size]``
        """
        # SwiGLU: SiLU(gate) * up
        gate = self.gate_proj(hidden_states)           # [..., ffn/tp]
        up   = self.up_proj(hidden_states)             # [..., ffn/tp]
        activated = self.activation_func(gate) * up    # element-wise

        # Down projection
        output = self.down_proj(activated)             # [..., hidden]

        # All-reduce across TP ranks (RowParallelLinear pattern)
        tp_size = _get_tp_world_size()
        if tp_size > 1:
            tp_group = _get_tp_group()
            if tp_group is not None:
                torch.distributed.all_reduce(output, group=tp_group)

        return output
