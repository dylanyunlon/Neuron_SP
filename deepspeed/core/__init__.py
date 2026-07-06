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

Imports are lazy (torch-free at module-load time) so that
``import deepspeed.core.parallel_state`` and
``from deepspeed.core.desloc_config import DesLocConfig`` both work
without triggering torch's CUDA SO loading — critical for dry-run tests
and import-chain validation in CI.

Torch-dependent submodules (hyper_comm_grid, distributed, …) are only
imported when first accessed via __getattr__.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Torch-free dataclass / config re-exports — always safe at import time
# ---------------------------------------------------------------------------

from deepspeed.core.desloc_config import DesLocConfig, TierSpec, TierType  # noqa: F401
from deepspeed.core.model_parallel_config import ModelParallelConfig        # noqa: F401

# ---------------------------------------------------------------------------
# Lazy attribute loader for torch-dependent submodules
# ---------------------------------------------------------------------------

_LAZY_SUBMODULES = {
    "HyperCommGrid": "deepspeed.core.hyper_comm_grid",
    "parallel_state": "deepspeed.core.parallel_state",
    "distributed": "deepspeed.core.distributed",
    "DistributedDataParallel": "deepspeed.core.distributed",
    "DistributedDataParallelConfig": "deepspeed.core.distributed",
    "finalize_model_grads": "deepspeed.core.distributed",
    "ParamAndGradBuffer": "deepspeed.core.distributed",
    "ParamAndGradBucketGroup": "deepspeed.core.distributed",
}


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        import importlib
        mod = importlib.import_module(_LAZY_SUBMODULES[name])
        # For submodule names return the module; for symbol names return the attr
        if name in ("parallel_state", "distributed"):
            return mod
        return getattr(mod, name)
    raise AttributeError(f"module 'deepspeed.core' has no attribute {name!r}")
