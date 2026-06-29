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
        from deepspeed.core.dist_checkpointing import save as dc_save, load as dc_load

        class DistCheckpointAdapter:
            """Thin wrapper around dist_checkpointing.save/load."""

            def save(self, state_dict: dict, path) -> None:
                dc_save(state_dict, str(path))

            def load(self, path) -> dict:
                return dc_load(str(path))

        logger.info("core_adapter: dist_checkpointing active")
        return DistCheckpointAdapter()
    except Exception as exc:
        logger.warning("core_adapter: dist_checkpointing failed (%s), fallback", exc)
        return None


# ---------------------------------------------------------------------------
# 4. BridgeCommunicator adapter + duck-typing wrapper
# ---------------------------------------------------------------------------

class BridgeToP2PWrapper:
    """Duck-typing wrapper: makes BridgeCommunicator quack like PCIeP2PCommunicator.

    PCIeP2PCommunicator interface (used by desloc_engine + hetero_mimo_training_loop):
        send_activation(tensor, src_device, dst_device, cache_key=None) -> Tensor

    BridgeCommunicator interface (Megatron pipeline):
        send_forward(tensor) / recv_forward() -> Tensor
        send_backward(tensor) / recv_backward() -> Tensor

    Mapping: src_device < dst_device ⇒ forward direction (lower stage → higher stage).
             src_device > dst_device ⇒ backward direction.
             src_device == dst_device ⇒ local copy, no bridge needed.
    """

    def __init__(self, bridge, locality_cache=None):
        self._bridge = bridge
        self._cache = locality_cache

    def send_activation(
        self,
        tensor: "torch.Tensor",
        src_device: int,
        dst_device: int,
        cache_key: Optional[str] = None,
    ) -> "torch.Tensor":
        import torch

        # Cache hit → skip transfer
        if cache_key is not None and self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached.to(f"cuda:{dst_device}", non_blocking=True)

        if src_device == dst_device:
            # Local copy, no bridge needed
            result = tensor.to(f"cuda:{dst_device}", non_blocking=True)
        elif src_device < dst_device:
            # Forward direction: this rank sends, peer receives
            self._bridge.send_forward(tensor)
            result = self._bridge.recv_forward()
        else:
            # Backward direction: this rank sends grad, peer receives
            self._bridge.send_backward(tensor)
            result = self._bridge.recv_backward()

        # Cache the result
        if cache_key is not None and self._cache is not None:
            self._cache.put(cache_key, result.detach())

        return result


def maybe_build_bridge_communicator(
    config: Any,
    existing_p2p: Any,
) -> Any:
    """If config.use_bridge_communicator is True AND pipeline_parallel_size > 1,
    wrap BridgeCommunicator in BridgeToP2PWrapper so it exposes send_activation().

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

        bridge_pg = ps.get_data_parallel_group()
        my_rank = ps.get_data_parallel_rank()
        peer_rank = (my_rank + 1) % ps.get_data_parallel_world_size()

        from deepspeed.core.model_parallel_config import ModelParallelConfig
        bridge = BridgeCommunicator(
            model_parallel_config=ModelParallelConfig(),
            bridge_pg=bridge_pg,
            my_role=CommRole.BOTH,
            peer_rank=peer_rank,
        )

        # Extract locality_cache from existing_p2p if it has one
        locality_cache = getattr(existing_p2p, "_cache", None)

        wrapper = BridgeToP2PWrapper(bridge, locality_cache=locality_cache)
        logger.info(
            "core_adapter: BridgeToP2PWrapper active (peer=%d, cache=%s)",
            peer_rank,
            locality_cache is not None,
        )
        return wrapper
    except Exception as exc:
        logger.warning("core_adapter: BridgeCommunicator failed (%s), keeping PCIeP2P", exc)
        return existing_p2p
