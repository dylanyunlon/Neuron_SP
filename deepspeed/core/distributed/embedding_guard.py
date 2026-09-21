"""
Embedding Grad-Sync Guard — prevent asymmetric embedding allreduce hangs.

Fix for issue #591 Blocker 3:
    finalize_model_grads.py has an embedding gradient allreduce that only
    fires when config.share_embeddings_and_output_weights = True.  If some
    ranks have this flag and others don't (or if the flag is derived from
    model attributes that differ across ranks), the embedding allreduce
    fires on a subset of ranks → NCCL hang.

Solution:
    1. validate_embedding_sync_flags() broadcasts the share_embeddings flag
       from rank 0 so all ranks agree before finalize_model_grads runs.
    2. EmbeddingGradSyncConfig stores the resolved flags, decoupling the
       decision from per-rank model inspection.

AST call chain (6 nodes):
    DesLocEngine.train()
      → finalize_model_grads()
        → _allreduce_all_embedding_grads()
          → _allreduce_word_embedding_grads()        ← conditional on share_embeddings
            → _allreduce_embedding_grad()
              → dist.all_reduce()                     ← deadlocks if asymmetric

Public API:
    EmbeddingGradSyncConfig
    validate_embedding_sync_flags(config, group) → EmbeddingGradSyncConfig
    safe_model_parallel_config(**overrides) → ModelParallelConfig
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingGradSyncConfig:
    """Resolved, rank-consistent embedding grad sync flags.

    After validate_embedding_sync_flags() all ranks hold identical values
    for these flags, preventing the conditional-collective asymmetry that
    causes NCCL hangs.

    Attributes:
        share_embeddings_and_output_weights: Whether word-embedding and
            output-weight gradients are shared and need cross-PP allreduce.
            False for the DES-LOC gate run (no PP, no weight tying).
        has_position_embeddings: Whether position-embedding allreduce is
            needed.  False for RoPE-based models (LLaMA 7B).
        has_cond_embedder: Whether conditional (DiT-style) embedding
            allreduce is needed.  False for decoder-only LMs.
        validated: True after collective validation has run.
    """
    share_embeddings_and_output_weights: bool = False
    has_position_embeddings: bool = False
    has_cond_embedder: bool = False
    validated: bool = False


def validate_embedding_sync_flags(
    share_embeddings: bool = False,
    has_position_embeddings: bool = False,
    has_cond_embedder: bool = False,
    group: Optional[dist.ProcessGroup] = None,
    *,
    device: Optional[torch.device] = None,
) -> EmbeddingGradSyncConfig:
    """Broadcast embedding sync flags from rank 0 to ensure all-rank agreement.

    Must be called by ALL ranks in the group before finalize_model_grads().
    Uses broadcast (not allreduce) so rank 0's flags are authoritative —
    this matches the Megatron convention where rank 0 owns the config.

    Args:
        share_embeddings: This rank's view of share_embeddings_and_output_weights.
        has_position_embeddings: This rank's view of position embedding status.
        has_cond_embedder: This rank's view of conditional embedder status.
        group: Process group (None = WORLD).
        device: CUDA device (auto-detected if None).

    Returns:
        EmbeddingGradSyncConfig with rank-0's authoritative values.
    """
    if not dist.is_initialized():
        return EmbeddingGradSyncConfig(
            share_embeddings_and_output_weights=share_embeddings,
            has_position_embeddings=has_position_embeddings,
            has_cond_embedder=has_cond_embedder,
            validated=True,
        )

    if device is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    # Pack 3 bools into a single int64 tensor for one broadcast.
    flags = torch.tensor(
        [
            int(share_embeddings),
            int(has_position_embeddings),
            int(has_cond_embedder),
        ],
        dtype=torch.int64,
        device=device,
    )

    # Determine source rank within this group.
    if group is not None:
        src_rank = dist.get_global_rank(group, 0)
    else:
        src_rank = 0

    dist.broadcast(flags, src=src_rank, group=group)

    resolved = EmbeddingGradSyncConfig(
        share_embeddings_and_output_weights=bool(flags[0].item()),
        has_position_embeddings=bool(flags[1].item()),
        has_cond_embedder=bool(flags[2].item()),
        validated=True,
    )

    rank = dist.get_rank()
    if rank != src_rank:
        # Check for mismatches and warn.
        local_flags = (share_embeddings, has_position_embeddings, has_cond_embedder)
        resolved_flags = (
            resolved.share_embeddings_and_output_weights,
            resolved.has_position_embeddings,
            resolved.has_cond_embedder,
        )
        if local_flags != resolved_flags:
            logger.warning(
                "[EmbeddingGuard] rank=%d: local flags %s overridden by "
                "rank-0 broadcast %s (prevents asymmetric NCCL collective)",
                rank, local_flags, resolved_flags,
            )

    return resolved


def safe_model_parallel_config(**overrides):
    """Construct a ModelParallelConfig with safe defaults for DES-LOC gate run.

    Explicitly disables all conditional embedding allreduces that can cause
    asymmetric NCCL hangs in the heterogeneous 3-GPU setup.

    The returned config is safe for finalize_model_grads() — every flag
    that gates a conditional collective is set to a value that either
    fires on ALL ranks or fires on NONE.

    Args:
        **overrides: Any ModelParallelConfig field to override.

    Returns:
        A ModelParallelConfig instance.
    """
    from deepspeed.core.model_parallel_config import ModelParallelConfig

    # Safe defaults for gate run: disable all conditional embedding collectives.
    safe_defaults = {
        "sequence_parallel": False,
        "pipeline_model_parallel_size": 1,
        "tensor_model_parallel_size": 1,
    }
    safe_defaults.update(overrides)

    config = ModelParallelConfig(**safe_defaults)

    # Belt-and-suspenders: ensure share_embeddings is explicitly False
    # when not set.  The ModelParallelConfig doesn't have this field
    # (it's a model attribute), but finalize_model_grads reads it from
    # the model.  We add it here for documentation and testing.
    if not hasattr(config, "share_embeddings_and_output_weights"):
        config.share_embeddings_and_output_weights = False

    return config
