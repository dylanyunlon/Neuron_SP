"""Adapters that wire deepspeed/core/ modules into the DES-LOC training loop.

Each adapter is gated by a TrainingConfig flag so it can be enabled with a
single config change, no code edits to desloc_engine.py required.

Usage in desloc_engine.py:
    from deepspeed.runtime.core_adapters import (
        build_core_scheduler,
        get_pipeline_forward_backward,
        build_dist_checkpoint_saver,
        build_hybrid_cp_schedule,
        maybe_enable_activation_offload,
    )
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. LR Scheduler adapter
# ---------------------------------------------------------------------------
def build_core_scheduler(
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
def get_pipeline_forward_backward(
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
def build_dist_checkpoint_saver(
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


def build_bridge_communicator(
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


# ---------------------------------------------------------------------------
# 5. Hybrid Context Parallel schedule adapter
# ---------------------------------------------------------------------------

def build_hybrid_cp_schedule(
    config: Any,
) -> Optional[Callable]:
    """If config.use_context_parallel is True, return a wrapper around
    hybrid_context_parallel_forward_backward that is signature-compatible
    with desloc_engine's forward_backward_func calls.

    The wrapper signature is::

        _cp_fb(
            forward_only,
            p2p_communicator,
            data_iterator,
            model,
            config,
            iteration,
        ) -> list[torch.Tensor]   # per-sample losses

    When config.use_context_parallel is False (or the module cannot be
    imported), returns None so the caller falls back to the default
    micro-batch loop.
    """
    if not getattr(config, "use_context_parallel", False):
        return None

    try:
        from deepspeed.core.pipeline_parallel.hybrid_cp_schedule import (
            hybrid_context_parallel_forward_backward,
        )

        def _hybrid_cp_wrapper(
            forward_only: bool,
            p2p_communicator: Any,
            data_iterator: Any,
            model: Any,
            config: Any,  # TrainingConfig passed at call-site
            iteration: int,
        ):
            """Thin shim that maps desloc_engine's calling convention to
            hybrid_context_parallel_forward_backward's parameter list.

            Required positional args for hybrid_context_parallel_forward_backward:
                forward_step_func       – per-sample forward callable
                data_iterator           – iterator over batches
                model                   – nn.Module
                num_microbatches        – total micro-batches in this step
                input_tensor            – pipeline input (None for PP=1)
                output_tensor_grad      – upstream grad  (None for PP=1)
                forward_data_store      – accumulator list (modified in-place)
                config                  – ModelParallelConfig / TrainingConfig
                collect_non_loss_data   – bool flag (False in training)
                first_val_step          – bool (False in training)
                forward_only            – passed through
                no_sync_func            – DDP no-sync context manager
                total_num_tokens        – running token counter (int)
                check_first_val_step    – callable(first_val_step, fw_only, is_first)
                model_type              – str tag for schedule variant selection
            """
            # Resolve model-parallel config for no_sync_func.
            # When CoreDDP is active it provides a no_sync() context manager;
            # otherwise we use a no-op context so the shim is safe in all modes.
            import contextlib

            _no_sync_fn: Callable
            if hasattr(model, "no_sync"):
                _no_sync_fn = model.no_sync
            else:
                @contextlib.contextmanager
                def _noop_sync():
                    yield
                _no_sync_fn = _noop_sync

            # forward_data_store accumulates per-sample loss tensors returned
            # by forward_step_func.  We create a fresh list each call so that
            # the caller (desloc_engine) can read back the losses.
            _fwd_store: list = []

            # Resolve num_microbatches from the config when possible.
            _num_mb = getattr(config, "num_microbatches", 1)

            # Minimal forward_step_func compatible with hybrid_cp's expectations:
            # must return (loss_tensor, num_tokens_tensor).
            def _forward_step(data_iter, _model):
                import torch
                batch = next(data_iter)
                if isinstance(batch, dict):
                    input_ids = batch["tokens"]
                    labels = batch.get("labels")
                else:
                    input_ids, labels = batch[0], (batch[1] if len(batch) > 1 else None)
                # Use model's own forward; assume it returns a scalar loss.
                if labels is not None:
                    loss = _model(input_ids, labels)
                else:
                    loss = _model(input_ids)
                # Scalar loss → 0-dim tensor; token count ≈ sequence length
                if not isinstance(loss, tuple):
                    loss_t = loss if isinstance(loss, type(loss)) and hasattr(loss, "shape") else loss
                else:
                    loss_t = loss[0]
                seq_len = input_ids.shape[-1]
                import torch as _t
                num_tokens = _t.tensor(seq_len, dtype=_t.int64, device=input_ids.device)
                _fwd_store.append(loss_t.detach())
                return loss_t, num_tokens

            hybrid_context_parallel_forward_backward(
                forward_step_func=_forward_step,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=_num_mb,
                input_tensor=None,          # PP=1: no pipeline input tensor
                output_tensor_grad=None,    # PP=1: no upstream gradient
                forward_data_store=_fwd_store,
                config=config,
                collect_non_loss_data=False,
                first_val_step=False,
                forward_only=forward_only,
                no_sync_func=_no_sync_fn,
                total_num_tokens=0,
                check_first_val_step=lambda fvs, fw, is_first: is_first,
                model_type="decoder",
            )

            return _fwd_store

        logger.info("core_adapter: hybrid_context_parallel_forward_backward active")
        return _hybrid_cp_wrapper

    except Exception as exc:
        logger.warning(
            "core_adapter: hybrid_cp_schedule failed (%s), fallback to default microbatch loop",
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# 6. Fine-grained activation offload adapter (DES-LOC tier-aware)
# ---------------------------------------------------------------------------

def maybe_enable_activation_offload(
    config: Any,
    tier_type: Any = None,
) -> Optional[Any]:
    """Conditionally initialise fine-grained activation offloading.

    Decision logic (DES-LOC spec)
    ──────────────────────────────
    * ``config.use_activation_offload`` must be True (opt-in flag).
    * ``tier_type`` drives the tier check:
        - PROFESSIONAL / CONSUMER  (≤ ~49 GB VRAM, e.g. A6000, RTX 4090)
          → offload **enabled** – GPU memory is scarce.
        - DATACENTER (H100, A100, ≥ 80 GB VRAM)
          → offload **skipped** – ample VRAM, adding PCIe traffic hurts MFU.
      When *tier_type* is None the adapter falls back to interrogating
      ``torch.cuda.get_device_properties`` directly.

    Returns
    ────────
    ``FineGrainedActivationOffloadingInterface`` singleton if offload is
    active, otherwise ``None``.  The caller (desloc_engine) can store the
    return value and use it as a context manager around each transformer layer::

        _offload = maybe_enable_activation_offload(cfg, tier_type)
        if _offload is not None:
            _offload.init_chunk_handler(vp_size, vp_stage, min_tensor_size)
        …
        with _offload or nullcontext():
            y = transformer_layer(x)

    Config keys recognised (all optional with sensible defaults):
        use_activation_offload      bool   – master on/off switch (default False)
        activation_offload_min_size int    – minimum tensor elements to offload
                                             (default 1 048 576 = 1 M elements)
        activation_offload_max_inflight int – max pending D2H per group name
                                              (default None = unlimited)
        virtual_pipeline_model_parallel_size int – VPP size (default None = 1)
    """
    if not getattr(config, "use_activation_offload", False):
        return None

    try:
        from deepspeed.core.pipeline_parallel.fine_grained_activation_offload import (
            FineGrainedActivationOffloadingInterface,
            PipelineOffloadManager,
            offload_required_for_tier,
        )

        # --- Tier check: skip on high-VRAM GPUs ---
        if not offload_required_for_tier(tier_type):
            logger.info(
                "core_adapter: activation offload SKIPPED "
                "(tier=%s indicates sufficient VRAM; disable use_activation_offload "
                "or set tier_type=PROFESSIONAL to force)",
                tier_type,
            )
            return None

        # --- Read config knobs ---
        min_size: int = getattr(
            config, "activation_offload_min_size", 1024 * 1024
        )
        max_inflight: Optional[int] = getattr(
            config, "activation_offload_max_inflight", None
        )
        vp_size: Optional[int] = getattr(
            config, "virtual_pipeline_model_parallel_size", None
        )

        # Ensure singleton is fresh
        PipelineOffloadManager.reset_instance()

        # Pre-initialise the first chunk handler for vp_stage 0 so that
        # callers that do not call init_chunk_handler explicitly still work.
        FineGrainedActivationOffloadingInterface.init_chunk_handler(
            vp_size=vp_size,
            vp_stage=0,
            min_offloaded_tensor_size=min_size,
            max_inflight_offloads=max_inflight,
        )

        logger.info(
            "core_adapter: fine-grained activation offload ACTIVE "
            "(tier=%s, min_size=%d, max_inflight=%s, vp_size=%s)",
            tier_type,
            min_size,
            max_inflight,
            vp_size,
        )
        return FineGrainedActivationOffloadingInterface

    except Exception as exc:
        logger.warning(
            "core_adapter: activation offload failed (%s), skipping",
            exc,
        )
        return None
