# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
deepspeed.core — Neuron_SP core training infrastructure.

Modeled after NVIDIA Megatron-LM's megatron/core/, adapted for
heterogeneous GPU clusters with DES-LOC and AutoSP.

Module hierarchy:
    core.desloc_config          — DES-LOC configuration
    core.model_parallel_config  — parallelism configuration
    core.parallel_state         — process group management
    core.hyper_comm_grid        — N-dimensional communication grid
    core.tensor_parallel        — TP layers
    core.distributed            — DDP, FSDP, grad finalization
    core.optimizer              — distributed optimizer
    core.pipeline_parallel      — pipeline schedules
    core.dist_checkpointing     — sharded checkpointing
    core.transformer            — attention, MLP, transformer layers, MoE
    core.datasets               — pretraining datasets
    core.models                 — GPT, hybrid models
"""

from deepspeed.core.desloc_config import DesLocConfig, TierSpec, TierType
from deepspeed.core.hyper_comm_grid import HyperCommGrid
from deepspeed.core.model_parallel_config import ModelParallelConfig

# parallel_state: TP/PP/DP/CP/EP process group management + DES-LOC tier groups.
# Import the module (not individual symbols) so callers can do:
#   import deepspeed.core.parallel_state as parallel_state
#   parallel_state.initialize_model_parallel(...)
# This matches Megatron's usage pattern throughout training code.
from deepspeed.core import parallel_state  # noqa: F401

# distributed: DDP wrapper, param/grad buffer, finalize_model_grads.
from deepspeed.core import distributed  # noqa: F401
from deepspeed.core.distributed import (  # noqa: F401
    DistributedDataParallel,
    DistributedDataParallelConfig,
    finalize_model_grads,
    ParamAndGradBuffer,
    ParamAndGradBucketGroup,
)
