"""Adapters that wire deepspeed/core/ modules into the DES-LOC training loop.

Each adapter is gated by a TrainingConfig flag so it can be enabled with a
single config change, no code edits to desloc_engine.py required.

Usage in desloc_engine.py:
    from deepspeed.runtime.core_adapters import (
        maybe_build_core_scheduler,
        maybe_get_pipeline_forward_backward,
        maybe_build_dist_checkpoint_saver,
    )
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. LR Scheduler adapter
# ---------------------------------------------------------------------------
def maybe_build_core_scheduler(
    optimizer: Any,
    config: Any,
) -> Optional[Any]:
    """If config.use_core_scheduler is True, build OptimizerParamScheduler.

    Otherwise returns None and desloc_engine falls back to its existing
    build_warmup_cosine_scheduler (torch LambdaLR).

    The OptimizerParamScheduler supports WSD decay, per-tier LR multipliers,
    and weight decay scheduling — features the LambdaLR path lacks.
    """
    if not getattr(config, "use_core_scheduler", False):
        return None

    try:
        from deepspeed.core.optimizer_param_scheduler import OptimizerParamScheduler

        tier_mult = getattr(config, "tier_lr_multiplier", 1.0)
        scheduler = OptimizerParamScheduler(
            optimizer=optimizer,
            init_lr=0.0,
            max_lr=config.max_lr,
            min_lr=config.min_lr,
            lr_warmup_steps=config.warmup_steps,
            lr_decay_steps=config.total_steps - config.warmup_steps,
            lr_decay_style=getattr(config, "lr_decay_style", "cosine"),
            start_wd=config.weight_decay,
            end_wd=config.weight_decay,
            wd_incr_steps=0,
            wd_incr_style="constant",
            tier_lr_multiplier=tier_mult,
        )
        logger.info(
            "core_adapter: OptimizerParamScheduler active "
            "(style=%s, tier_mult=%.2f)",
            getattr(config, "lr_decay_style", "cosine"),
            tier_mult,
        )
        return scheduler
    except Exception as exc:
        logger.warning("core_adapter: OptimizerParamScheduler failed (%s), fallback", exc)
        return None


# ---------------------------------------------------------------------------
# 2. Pipeline schedule adapter
# ---------------------------------------------------------------------------
def maybe_get_pipeline_forward_backward(
    config: Any,
    default_fn: Callable,
) -> Callable:
    """If config.use_pipeline_schedule is True, return the Megatron-style
    get_forward_backward_func() dispatch. Otherwise return default_fn unchanged.

    This replaces desloc_engine's inline _forward_backward_func closure with
    combined_1f1b / interleaved_1f1b from deepspeed/core/pipeline_parallel/.
    """
    if not getattr(config, "use_pipeline_schedule", False):
        return default_fn

    try:
        from deepspeed.core.pipeline_parallel.schedules import get_forward_backward_func
        import deepspeed.core.parallel_state as ps

        pp_size = ps.get_pipeline_model_parallel_world_size() if ps.is_initialized() else 1
        vp_size = getattr(config, "virtual_pipeline_model_parallel_size", None)

        if pp_size <= 1:
            logger.info("core_adapter: pp_size=1, pipeline schedule not needed")
            return default_fn

        fb_func = get_forward_backward_func(pp_size=pp_size, vp_size=vp_size)
        logger.info(
            "core_adapter: pipeline schedule active (pp=%d, vp=%s)",
            pp_size, vp_size,
        )
        return fb_func
    except Exception as exc:
        logger.warning("core_adapter: pipeline schedule failed (%s), fallback", exc)
        return default_fn


# ---------------------------------------------------------------------------
# 3. dist_checkpointing adapter
# ---------------------------------------------------------------------------
def maybe_build_dist_checkpoint_saver(
    config: Any,
) -> Optional[Any]:
    """If config.use_dist_checkpointing is True, return a wrapper that
    delegates save/load to deepspeed/core/dist_checkpointing/.

    The wrapper exposes .save(state_dict, path) and .load(path) methods
    with the same contract as torch.save/torch.load, but uses async
    sharded strategies underneath.
    """
    if not getattr(config, "use_dist_checkpointing", False):
        return None

    try:
        from deepspeed.core.dist_checkpointing import core as dc_core

        class DistCheckpointAdapter:
            """Thin wrapper around dist_checkpointing.save/load."""

            def save(self, state_dict: dict, path) -> None:
                dc_core.save(state_dict, str(path))

            def load(self, path) -> dict:
                return dc_core.load(str(path))

        logger.info("core_adapter: dist_checkpointing active")
        return DistCheckpointAdapter()
    except Exception as exc:
        logger.warning("core_adapter: dist_checkpointing failed (%s), fallback", exc)
        return None


# ---------------------------------------------------------------------------
# 4. BridgeCommunicator adapter
# ---------------------------------------------------------------------------
def maybe_build_bridge_communicator(
    config: Any,
    existing_p2p: Any,
) -> Any:
    """If config.use_bridge_communicator is True AND pipeline_parallel_size > 1,
    wrap or replace the existing PCIeP2PCommunicator with BridgeCommunicator.

    Otherwise return existing_p2p unchanged.
    """
    if not getattr(config, "use_bridge_communicator", False):
        return existing_p2p

    try:
        from deepspeed.core.pipeline_parallel.p2p_communication import (
            BridgeCommunicator,
            CommRole,
        )
        import deepspeed.core.parallel_state as ps

        if not ps.is_initialized():
            return existing_p2p

        pp_size = ps.get_pipeline_model_parallel_world_size() if hasattr(ps, "get_pipeline_model_parallel_world_size") else 1
        if pp_size <= 1:
            return existing_p2p

        # BridgeCommunicator needs a bridge_pg. For now, use the DP group
        # as cross-grid bridge (will be replaced with dedicated bridge PG later).
        bridge_pg = ps.get_data_parallel_group()
        my_rank = ps.get_data_parallel_rank()

        # In a 2-tier setup: rank 0,1 = A6000 pool, rank 2 = H100 pool
        # The bridge connects the last rank of pool 0 to the first rank of pool 1
        peer_rank = (my_rank + 1) % ps.get_data_parallel_world_size()

        from deepspeed.core.model_parallel_config import ModelParallelConfig
        bridge = BridgeCommunicator(
            model_parallel_config=ModelParallelConfig(),
            bridge_pg=bridge_pg,
            my_role=CommRole.BOTH,
            peer_rank=peer_rank,
        )
        logger.info("core_adapter: BridgeCommunicator active (peer=%d)", peer_rank)
        return bridge
    except Exception as exc:
        logger.warning("core_adapter: BridgeCommunicator failed (%s), keeping PCIeP2P", exc)
        return existing_p2p
