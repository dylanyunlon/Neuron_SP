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
        build_moe_adapter,
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


# ---------------------------------------------------------------------------
# 7. MoE adapter — wire deepspeed/core/transformer/moe/ into the engine
# ---------------------------------------------------------------------------

class MoEAdapter:
    """Manages MoE layers inside a model built with MiniTransformer/TransformerBlock.

    Responsibilities
    ----------------
    * **patch_model** — replace the dense ``MLP`` sub-module in every
      TransformerBlock with an ``MoELayer`` according to the engine config.
      Only every ``moe_layer_freq``-th block gets MoE (default: every block
      when freq=1).  Layers owned by H100 (high-VRAM) tiers always get MoE;
      A6000 layers only get MoE when ``moe_on_all_tiers=True``.

    * **collect_aux_loss** — walk all ``MoELayer`` instances in the model and
      sum their router auxiliary losses into a single scalar.  Called once per
      micro-batch backward so the aux loss participates in the gradient graph.

    * **log_utilization** — emit per-expert token-count statistics at the
      requested logging cadence (gated by ``moe_log_every``).

    Config keys recognised in ``TrainingConfig`` (all optional):
        num_moe_experts         int   – number of experts per MoE layer (default 8)
        moe_router_topk         int   – tokens routed to top-k experts (default 2)
        moe_aux_loss_coeff      float – load-balancing aux loss weight (default 0.01)
        moe_z_loss_coeff        float – z-loss weight for router stability (default 0.0)
        moe_token_capacity_factor float – optional expert capacity factor (default None)
        moe_layer_freq          int   – replace every N-th block with MoE (default 1)
        moe_on_all_tiers        bool  – enable MoE on A6000 tiers too (default True)
        moe_num_shared_experts  int   – shared (dense) expert count per layer (default 0)
        ffn_hidden_size         int   – expert intermediate size (default hidden*4)
        activation_func_type    str   – 'swiglu' or 'gelu' (default 'swiglu')
    """

    def __init__(self, config: Any, tiers: Any = None) -> None:
        self.config = config
        self.tiers = tiers or []
        self._moe_layers: list = []   # populated by patch_model()

    # ------------------------------------------------------------------
    # Internal: build a synthetic config object the MoELayer / TopKRouter
    # constructors understand.  They only need attribute access, so a
    # simple namespace is enough — no dataclass or TypedDict required.
    # ------------------------------------------------------------------
    def _make_moe_config(self) -> Any:
        """Return an attribute namespace with all fields MoELayer expects."""
        import types
        cfg = self.config
        ns = types.SimpleNamespace(
            hidden_size=cfg.hidden_size,
            ffn_hidden_size=getattr(cfg, "ffn_hidden_size", cfg.hidden_size * 4),
            num_moe_experts=getattr(cfg, "num_moe_experts", 8),
            moe_router_topk=getattr(cfg, "moe_router_topk", 2),
            moe_aux_loss_coeff=getattr(cfg, "moe_aux_loss_coeff", 0.01),
            moe_z_loss_coeff=getattr(cfg, "moe_z_loss_coeff", 0.0),
            moe_token_capacity_factor=getattr(cfg, "moe_token_capacity_factor", None),
            moe_num_shared_experts=getattr(cfg, "moe_num_shared_experts", 0),
            activation_func_type=getattr(cfg, "activation_func_type", "swiglu"),
        )
        return ns

    def patch_model(self, model: Any) -> int:
        """Replace dense MLP sub-modules with MoELayer in eligible blocks.

        Args:
            model: The nn.Module to patch (typically ``DesLocEngine.model``).

        Returns:
            Number of blocks that were converted to MoE.
        """
        try:
            from deepspeed.core.transformer.moe.moe_layer import MoELayer
        except Exception as exc:  # pragma: no cover
            logger.warning("MoEAdapter.patch_model: cannot import MoELayer (%s)", exc)
            return 0

        cfg = self.config
        moe_config = self._make_moe_config()
        freq: int = getattr(cfg, "moe_layer_freq", 1)
        on_all_tiers: bool = getattr(cfg, "moe_on_all_tiers", True)

        # Build device-index → TierClass lookup from discovered tiers so we
        # can skip MoE on memory-constrained A6000 tiers when requested.
        tier_values: dict = {}
        for spec in self.tiers:
            tier_val = getattr(getattr(spec, "tier", None), "value", "UNKNOWN")
            tier_values[spec.device_index] = tier_val

        # Support both MiniTransformer (.blocks) and models with .layers.
        block_list = getattr(model, "blocks", None) or getattr(model, "layers", None)
        if block_list is None:
            logger.warning(
                "MoEAdapter.patch_model: model has no .blocks/.layers; "
                "MoE patching skipped."
            )
            return 0

        converted = 0
        for layer_idx, block in enumerate(block_list):
            # Frequency gate: only replace every freq-th layer.
            if freq > 1 and layer_idx % freq != 0:
                continue

            # Tier gate: when moe_on_all_tiers=False, skip A6000 layers.
            if not on_all_tiers:
                dev_idx = getattr(block, "_device_idx", -1)
                tier_name = tier_values.get(dev_idx, "UNKNOWN")
                if "A6000" in tier_name:
                    logger.debug(
                        "MoEAdapter: layer %d on A6000 — skipped (moe_on_all_tiers=False)",
                        layer_idx,
                    )
                    continue

            # Replace .mlp sub-module if present; fall back to .ffn.
            target_attr = None
            if hasattr(block, "mlp"):
                target_attr = "mlp"
            elif hasattr(block, "ffn"):
                target_attr = "ffn"
            else:
                logger.debug(
                    "MoEAdapter: layer %d has no .mlp/.ffn — cannot convert",
                    layer_idx,
                )
                continue

            moe_layer = MoELayer(moe_config, layer_number=layer_idx)
            setattr(block, target_attr, moe_layer)
            self._moe_layers.append(moe_layer)
            converted += 1
            logger.debug(
                "MoEAdapter: layer %d .%s replaced with MoELayer "
                "(%d experts, top-%d)",
                layer_idx, target_attr,
                moe_config.num_moe_experts, moe_config.moe_router_topk,
            )

        logger.info(
            "MoEAdapter.patch_model: %d/%d blocks converted to MoE "
            "(experts=%d, topk=%d, freq=%d)",
            converted, len(block_list),
            moe_config.num_moe_experts, moe_config.moe_router_topk, freq,
        )
        return converted

    def collect_aux_loss(self) -> Any:
        """Sum router auxiliary losses from all live MoELayer instances.

        Returns a zero-dimensional scalar tensor on the same device as the
        router weights, or Python float 0.0 when there are no MoE layers.
        The returned value can be added directly to the main loss so that
        aux-loss gradients flow through the router gate weights.
        """
        import torch

        total = None
        for moe_layer in self._moe_layers:
            layer_aux = getattr(moe_layer, "get_aux_loss", lambda: None)()
            if layer_aux is None:
                continue
            if not isinstance(layer_aux, torch.Tensor):
                continue
            if total is None:
                total = layer_aux
            else:
                total = total + layer_aux

        if total is None:
            return 0.0
        return total

    def log_utilization(self, step: int, moe_log_every: int = 100) -> None:
        """Log per-expert token-count statistics at the requested cadence.

        Args:
            step: Current global training step.
            moe_log_every: Emit utilisation logs every this many steps.
        """
        if step % moe_log_every != 0:
            return

        for i, moe_layer in enumerate(self._moe_layers):
            router = getattr(moe_layer, "router", None)
            if router is None:
                continue
            aux = getattr(router, "aux_loss", None)
            if aux is None:
                continue
            try:
                aux_val = float(aux.item()) if hasattr(aux, "item") else float(aux)
            except Exception:
                aux_val = float("nan")
            logger.info(
                "[MoE] step=%d layer=%d aux_loss=%.6f experts=%d topk=%d",
                step, i, aux_val,
                getattr(moe_layer, "num_experts", -1),
                getattr(moe_layer, "topk", -1),
            )


def build_moe_adapter(
    config: Any,
    model: Any,
    tiers: Any = None,
) -> Optional["MoEAdapter"]:
    """Build and apply a MoEAdapter if ``config.use_moe`` is True.

    Gate flag
    ---------
    ``config.use_moe: bool`` — master on/off switch.  When False (default),
    this function is a no-op and returns ``None`` so the rest of the engine
    is completely unaffected.

    On success, ``MoEAdapter.patch_model(model)`` is called immediately so
    the caller receives a model already populated with MoE layers.

    Args:
        config: ``TrainingConfig`` (or any object with ``use_moe`` attr).
        model:  The ``nn.Module`` to patch in-place.
        tiers:  Optional list of ``TierSpec`` objects (from ``TierDiscovery``).
                Used to apply tier-aware MoE placement (skip A6000 when
                ``moe_on_all_tiers=False``).

    Returns:
        Populated :class:`MoEAdapter` instance, or ``None`` when MoE is
        disabled or the import chain fails.
    """
    if not getattr(config, "use_moe", False):
        return None

    try:
        adapter = MoEAdapter(config=config, tiers=tiers or [])
        n_converted = adapter.patch_model(model)
        if n_converted == 0:
            logger.warning(
                "build_moe_adapter: use_moe=True but no blocks were converted; "
                "check model architecture and moe_layer_freq setting."
            )
        logger.info(
            "build_moe_adapter: MoEAdapter active (%d MoE layers, "
            "experts=%d, topk=%d, aux_coeff=%.4f)",
            n_converted,
            getattr(config, "num_moe_experts", 8),
            getattr(config, "moe_router_topk", 2),
            getattr(config, "moe_aux_loss_coeff", 0.01),
        )
        return adapter
    except Exception as exc:
        logger.warning(
            "build_moe_adapter: failed (%s); MoE disabled for this run.", exc
        )
        return None


# ---------------------------------------------------------------------------
# 7. Multi-Latent Attention (MLA) adapter
# ---------------------------------------------------------------------------

class _LightweightMLA(object):
    """Pure-PyTorch MLA drop-in for MiniTransformer's CausalSelfAttention.

    Implements the DeepSeek-style low-rank KV compression without requiring
    Megatron-LM or any external RoPE library.  This is the *training path*
    that always works; for inference or tensor-parallel workloads the caller
    can swap in ``MLASelfAttention`` from deepspeed.core.transformer.

    Architecture summary
    --------------------
    Queries      : x → W_q_down (hidden → q_lora_rank) → W_q_up (→ n_heads * head_dim)
    Keys/Values  : x → W_kv_down (hidden → kv_lora_rank) →
                       W_k_up  (→ n_heads * head_dim)
                       W_v_up  (→ n_heads * v_head_dim)
    Output       : concat(head_outputs) → W_proj (n_heads * v_head_dim → hidden)

    RoPE is *not* applied here; the MiniTransformer does not use positional
    embeddings in its smoke-test configuration.  Add a rotary layer before
    W_q_up / W_k_up when porting to a production model.
    """

    def __new__(
        cls,
        hidden: int,
        n_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        head_dim: int,
        v_head_dim: int,
    ):
        """Return an ``nn.Module`` (not a plain Python object)."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class _MLA(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_heads   = n_heads
                self.head_dim  = head_dim
                self.v_head_dim = v_head_dim

                # Q low-rank path: hidden → q_lora_rank → n_heads * head_dim
                self.w_q_down = nn.Linear(hidden, q_lora_rank, bias=False)
                self.w_q_up   = nn.Linear(q_lora_rank, n_heads * head_dim, bias=False)

                # KV shared down-projection: hidden → kv_lora_rank (the "latent")
                self.w_kv_down = nn.Linear(hidden, kv_lora_rank, bias=False)
                # K up-projection from latent
                self.w_k_up   = nn.Linear(kv_lora_rank, n_heads * head_dim, bias=False)
                # V up-projection from latent
                self.w_v_up   = nn.Linear(kv_lora_rank, n_heads * v_head_dim, bias=False)
                # Output projection: concat of all V heads → hidden
                self.proj      = nn.Linear(n_heads * v_head_dim, hidden, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                B, T, _ = x.shape
                h, d, vd = self.n_heads, self.head_dim, self.v_head_dim

                q = self.w_q_up(self.w_q_down(x))          # (B, T, h*d)
                latent = self.w_kv_down(x)                  # (B, T, kv_lora_rank)
                k = self.w_k_up(latent)                     # (B, T, h*d)
                v = self.w_v_up(latent)                     # (B, T, h*vd)

                # Reshape to (B, h, T, dim_per_head)
                q = q.view(B, T, h, d).transpose(1, 2)
                k = k.view(B, T, h, d).transpose(1, 2)
                v = v.view(B, T, h, vd).transpose(1, 2)

                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                out = out.transpose(1, 2).contiguous().view(B, T, h * vd)
                return self.proj(out)

        return _MLA()


class MLAAdapter:
    """Patches a MiniTransformer/TransformerBlock model to use MLA attention.

    Responsibilities
    ----------------
    * **patch_model** — replace the ``CausalSelfAttention`` (``.attn``) in
      every TransformerBlock with a :class:`_LightweightMLA` module.
      Only every ``mla_layer_freq``-th block is converted (default: every
      block when freq=1).

    * **log_info** — emit a one-line summary of the conversion.

    Config keys recognised in ``TrainingConfig`` (all optional):
        use_mla             bool  – master on/off switch (default False)
        mla_q_lora_rank     int   – Q bottleneck rank (default hidden // 4)
        mla_kv_lora_rank    int   – KV shared latent rank (default hidden // 8)
        mla_head_dim        int   – per-head QK dimension (default hidden // n_heads)
        mla_v_head_dim      int   – per-head V dimension  (default hidden // n_heads)
        mla_layer_freq      int   – convert every N-th block (default 1)
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self._mla_layers: list = []  # populated by patch_model()

    def _resolve_dims(self, hidden: int, n_heads: int):
        """Return (q_lora_rank, kv_lora_rank, head_dim, v_head_dim) from config."""
        cfg = self.config
        head_dim    = getattr(cfg, "mla_head_dim",    hidden // n_heads)
        v_head_dim  = getattr(cfg, "mla_v_head_dim",  hidden // n_heads)
        q_lora_rank = getattr(cfg, "mla_q_lora_rank",  hidden // 4)
        kv_lora_rank = getattr(cfg, "mla_kv_lora_rank", hidden // 8)
        return q_lora_rank, kv_lora_rank, head_dim, v_head_dim

    def patch_model(self, model: Any) -> int:
        """Replace CausalSelfAttention with _LightweightMLA in eligible blocks.

        Args:
            model: The ``nn.Module`` to patch (typically ``DesLocEngine.model``).

        Returns:
            Number of blocks that were converted to MLA.
        """
        cfg  = self.config
        freq: int = getattr(cfg, "mla_layer_freq", 1)

        block_list = getattr(model, "blocks", None) or getattr(model, "layers", None)
        if block_list is None:
            logger.warning(
                "MLAAdapter.patch_model: model has no .blocks/.layers; "
                "MLA patching skipped."
            )
            return 0

        converted = 0
        for layer_idx, block in enumerate(block_list):
            if freq > 1 and layer_idx % freq != 0:
                continue

            if not hasattr(block, "attn"):
                logger.debug(
                    "MLAAdapter: layer %d has no .attn — cannot convert", layer_idx
                )
                continue

            # Infer hidden / n_heads from the existing attention module.
            existing_attn = block.attn
            # CausalSelfAttention stores .qkv: Linear(hidden, 3*hidden)
            if hasattr(existing_attn, "qkv"):
                hidden   = existing_attn.qkv.in_features
                n_heads  = existing_attn.n_heads
            elif hasattr(existing_attn, "proj"):
                hidden  = existing_attn.proj.out_features
                n_heads = getattr(existing_attn, "n_heads", 1)
            else:
                logger.debug(
                    "MLAAdapter: layer %d .attn has unknown shape — skipping",
                    layer_idx,
                )
                continue

            q_lora_rank, kv_lora_rank, head_dim, v_head_dim = self._resolve_dims(
                hidden, n_heads
            )

            mla_layer = _LightweightMLA(
                hidden=hidden,
                n_heads=n_heads,
                q_lora_rank=q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                head_dim=head_dim,
                v_head_dim=v_head_dim,
            )
            # Move to same device/dtype as the existing module.
            param0 = next(existing_attn.parameters(), None)
            if param0 is not None:
                mla_layer = mla_layer.to(dtype=param0.dtype, device=param0.device)

            block.attn = mla_layer
            self._mla_layers.append(mla_layer)
            converted += 1
            logger.debug(
                "MLAAdapter: layer %d .attn replaced with _LightweightMLA "
                "(q_lora=%d, kv_lora=%d, heads=%d, head_dim=%d, v_head_dim=%d)",
                layer_idx, q_lora_rank, kv_lora_rank, n_heads, head_dim, v_head_dim,
            )

        logger.info(
            "MLAAdapter.patch_model: %d/%d blocks converted to MLA "
            "(q_lora_rank=%d, kv_lora_rank=%d, freq=%d)",
            converted,
            len(block_list),
            getattr(cfg, "mla_q_lora_rank",  getattr(cfg, "hidden_size", 0) // 4),
            getattr(cfg, "mla_kv_lora_rank", getattr(cfg, "hidden_size", 0) // 8),
            freq,
        )
        return converted

    def param_count(self) -> int:
        """Return total parameter count across all patched MLA layers."""
        return sum(p.numel() for layer in self._mla_layers for p in layer.parameters())


def build_mla_adapter(
    config: Any,
    model: Any,
) -> "Optional[MLAAdapter]":
    """Build and apply an MLAAdapter if ``config.use_mla`` is True.

    Gate flag
    ---------
    ``config.use_mla: bool`` — master on/off switch.  When False (default),
    this function is a no-op and returns ``None`` so the rest of the engine
    is completely unaffected.

    On success, ``MLAAdapter.patch_model(model)`` is called immediately so
    the caller receives a model already populated with MLA attention layers.

    Args:
        config: ``TrainingConfig`` (or any object with ``use_mla`` attr).
        model:  The ``nn.Module`` to patch in-place.

    Returns:
        Populated :class:`MLAAdapter` instance, or ``None`` when MLA is
        disabled or the patch step fails.
    """
    if not getattr(config, "use_mla", False):
        return None

    try:
        adapter = MLAAdapter(config=config)
        n_converted = adapter.patch_model(model)
        if n_converted == 0:
            logger.warning(
                "build_mla_adapter: use_mla=True but no blocks were converted; "
                "check model architecture and mla_layer_freq setting."
            )
        logger.info(
            "build_mla_adapter: MLAAdapter active (%d MLA layers, "
            "q_lora_rank=%d, kv_lora_rank=%d)",
            n_converted,
            getattr(config, "mla_q_lora_rank",  getattr(config, "hidden_size", 0) // 4),
            getattr(config, "mla_kv_lora_rank", getattr(config, "hidden_size", 0) // 8),
        )
        return adapter
    except Exception as exc:
        logger.warning(
            "build_mla_adapter: failed (%s); MLA disabled for this run.", exc
        )
        return None
