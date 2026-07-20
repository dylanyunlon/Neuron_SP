# SPDX-License-Identifier: Apache-2.0
"""engine_integration.py — single entrypoint that wires hetero_bridge into desloc_engine.

Call order
----------
install(engine) does exactly this, in order:

  1. TierMap.discover()           — all-gather GPU topology
  2. HeteroShardPlanner.plan()    — assign fp32 shards by VRAM budget
  3. DistOptAdapter.build()       — per-rank optimizer (CPUAdam / fused AdamW)
  4. Attach to engine             — set engine.optimizer, engine._dist_optimizer, etc.
  5. DesLocSyncPolicy.classify()  — classify params into Kx/Ku/Kv sync classes
  6. PPScheduleAdapter            — constructed; layer_split() called and cached
  7. AutoSPHook                   — wrap_grad_reduction + sp_aware_sync installed

Additionally installs:
  - ``finalize_grads_step(engine, step)`` as ``engine._hetero_finalize_grads``
    for the training loop to call instead of the raw finalize_model_grads.
  - ``engine._pp_layer_split`` populated from PPScheduleAdapter.layer_split().

After install() returns:
  - engine.optimizer              is DistOptAdapter  (has .step, .zero_grad, .defaults)
  - engine._dist_optimizer        is the underlying DistributedOptimizer (or None)
  - engine._cpu_offload_optim     is True on A6000 ranks
  - engine._optim_gpu_device      is the local CUDA device
  - engine._hetero_bridge_adapter is the DistOptAdapter (for reduce_scatter / all_gather)
  - engine._desloc_policy         is the DesLocSyncPolicy (classify done on install)
  - engine._pp_adapter            is the PPScheduleAdapter (layer_split cached)
  - engine._pp_layer_split        is the list[int] from layer_split()
  - engine._autosp_hook           is the AutoSPHook (hooks installed if SP active)
  - engine._hetero_finalize_grads is a callable(step) → None for the training loop
  - engine.model                  has been moved to the local CUDA device
"""
from __future__ import annotations
import torch

import logging
import os
from typing import Callable, List, Optional, TYPE_CHECKING

from .desloc_sync_policy import SyncPeriods

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# finalize_grads_step  — DES-LOC-aware grad finalization callable
# ---------------------------------------------------------------------------

def _make_finalize_grads_step(
    engine,
    adapter,
    policy,
) -> Callable[[int], None]:
    """Return a callable that runs finalize_model_grads with DES-LOC Kx gating.

    The returned function is stored as ``engine._hetero_finalize_grads`` and
    should be called from the training loop *after* backward() but *before*
    optimizer.step().

    It:
      1. Updates the gradient variance EMA (for future classify() calls).
      2. Determines whether this is a Kx sync step.
      3. Calls ``core.distributed.finalize_model_grads`` with the appropriate
         ``skip_grad_sync`` and ``desloc_step`` arguments.
      4. Calls ``adapter.reduce_scatter_grads()`` when using DistributedOptimizer.

    Args:
        engine:  The DesLocEngine instance.
        adapter: The DistOptAdapter wrapping the optimizer.
        policy:  The DesLocSyncPolicy holding Kx/Ku/Kv periods.

    Returns:
        Callable[[int], None] — takes the current training step index.
    """
    try:
        from deepspeed.core.distributed import finalize_model_grads as _fmg
        _HAS_FMG = True
    except ImportError:
        _fmg = None
        _HAS_FMG = False

    kx = policy.periods.kx

    def finalize_grads_step(step: int) -> None:
        """Run DES-LOC-gated finalize_model_grads for training step *step*."""
        model = engine.model
        named_params = [
            (n, p) for n, p in model.named_parameters() if p.requires_grad
        ]

        # ── Update variance EMA (cheap; no allreduce) ─────────────
        try:
            policy.update_variance_ema(named_params)
        except Exception:
            pass  # non-critical; EMA will catch up

        is_kx = (step + 1) % kx == 0

        # ── finalize_model_grads via core.distributed ─────────────
        if _HAS_FMG:
            model_list = model if isinstance(model, list) else [model]
            try:
                # Derive desloc_config from engine if available.
                desloc_cfg = getattr(engine, "_desloc_config_obj", None)
                _fmg(
                    model=model_list,
                    skip_grad_sync=(not is_kx),
                    desloc_step=step,
                    desloc_config=desloc_cfg,
                )
            except Exception as exc:
                # finalize_model_grads can fail if parallel_state is not
                # initialised (single-rank test mode).  Log and continue.
                logger.debug(
                    "[hetero_bridge] finalize_model_grads failed (%s); "
                    "continuing without finalization.", exc
                )

        # ── reduce_scatter_grads (DistributedOptimizer path) ──────
        if is_kx:
            try:
                adapter.reduce_scatter_grads()
            except Exception as exc:
                logger.debug(
                    "[hetero_bridge] reduce_scatter_grads failed (%s).", exc
                )

        logger.debug(
            "[hetero_bridge] finalize_grads_step: step=%d Kx=%s",
            step + 1, "SYNC" if is_kx else "skip",
        )

    return finalize_grads_step


# ---------------------------------------------------------------------------
# install()  — frozen public API
# ---------------------------------------------------------------------------

def install(
    engine,
    *,
    lr: float,
    betas: "tuple[float, float]" = (0.9, 0.95),
    weight_decay: float = 0.1,
    sync_periods: Optional["SyncPeriods"] = None,
) -> None:
    """Wire the full hetero_bridge stack onto *engine*.

    Order: TierMap.discover() → HeteroShardPlanner.plan() → DistOptAdapter.build()
    → DesLocSyncPolicy.classify() → PPScheduleAdapter.layer_split()
    → AutoSPHook.wrap_grad_reduction() + sp_aware_sync()
    → install finalize_grads_step callable.

    The function is idempotent if called a second time — it simply re-runs the
    discovery and rebuilds the adapter.

    Args:
        engine:       DesLocEngine instance to wire up.
        lr:           Learning rate for the inner optimizer.
        betas:        (beta1, beta2) Adam coefficients.
        weight_decay: AdamW weight decay.
        sync_periods: Override DES-LOC Kx/Ku/Kv periods.  Defaults to
                      (kx=32, ku=96, kv=192) mirroring desloc_engine.py.

    Raises:
        RuntimeError: If DistOptAdapter.build() raises and no fallback is
                      available (in practice build() always falls back to the
                      inner optimizer, so this should not propagate).
    """
    import torch

    # ------------------------------------------------------------------
    # Lazy imports — keep them local so that merely importing this module
    # does NOT trigger the deepspeed/__init__.py → apex chain.
    # ------------------------------------------------------------------
    from .tier_map import TierMap, TierInfo, GPUTier
    from .shard_planner import HeteroShardPlanner, ShardPlan
    from .dist_opt_adapter import DistOptAdapter
    from .desloc_sync_policy import DesLocSyncPolicy, SyncPeriods as _SP
    from .pp_schedule_adapter import PPScheduleAdapter
    from .autosp_hook import AutoSPHook

    # ── 1. GPU topology discovery ──────────────────────────────────────
    logger.info("[hetero_bridge] install() — discovering GPU topology …")
    try:
        tier_map = TierMap.discover()
    except Exception as exc:
        logger.warning(
            "[hetero_bridge] TierMap.discover() failed (%s); "
            "falling back to single-rank UNKNOWN tier.", exc
        )
        _vram = (
            torch.cuda.get_device_properties(0).total_memory
            if torch.cuda.is_available() else 0
        )
        tier_map = TierMap.from_infos([
            TierInfo(rank=0, tier=GPUTier.UNKNOWN,
                     total_vram_bytes=_vram,
                     numa_node=0, peak_bf16_tflops=100.0)
        ])

    # ── 2. Shard plan ──────────────────────────────────────────────────
    named_params = [
        (n, p) for n, p in engine.model.named_parameters() if p.requires_grad
    ]
    planner = HeteroShardPlanner(tier_map)
    try:
        shard_plan = planner.plan(named_params)
    except Exception as exc:
        logger.warning(
            "[hetero_bridge] HeteroShardPlanner.plan() failed (%s); "
            "using empty ShardPlan (adapter will still work).", exc
        )
        shard_plan = ShardPlan(rationale=f"fallback-empty: {exc}")

    # ── 3. Build hetero DistributedOptimizer ───────────────────────────
    adapter = DistOptAdapter(
        model=engine.model,
        shard_plan=shard_plan,
        tier_map=tier_map,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )
    optimizer = adapter.build()  # never raises — has internal fallback

    # ── 3b. Barrier: all ranks must complete build() before any rank
    # proceeds to engine.model.to(device) and train().  Without this,
    # the fastest rank (H100, no JIT) reaches train() and starts ZeRO-3
    # collectives while slower ranks (A6000, DeepSpeedCPUAdam JIT) are
    # still in build(), causing NCCL collective asymmetry → deadlock.
    import torch.distributed as _dist
    if _dist.is_initialized():
        logger.info("[hetero_bridge] barrier after adapter.build() — waiting for all ranks")
        _dist.barrier()
        logger.info("[hetero_bridge] barrier passed — all ranks ready")

    # ── 4. Attach to engine ────────────────────────────────────────────
    engine.optimizer = adapter
    engine._hetero_bridge_adapter = adapter

    # _dist_optimizer: expose DistributedOptimizer directly when available.
    if (adapter._opt is not None
            and hasattr(adapter._opt, "data_parallel_rank")
            and hasattr(adapter._opt, "sync_moments")):
        engine._dist_optimizer = adapter._opt
        logger.info(
            "[hetero_bridge] engine._dist_optimizer = DistributedOptimizer "
            "(rank=%d/%d)",
            adapter._opt.data_parallel_rank,
            adapter._opt.data_parallel_world_size,
        )
    else:
        engine._dist_optimizer = None
        logger.info(
            "[hetero_bridge] engine._dist_optimizer = None "
            "(DistributedOptimizer not constructed; single-rank or test mode)"
        )

    # _cpu_offload_optim — A6000 ranks offload optimizer states to CPU.
    try:
        import torch.distributed as dist
        local_rank = dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        local_rank = int(os.environ.get("RANK", "0"))

    engine._cpu_offload_optim = tier_map.is_low_vram(local_rank)
    engine._optim_type = "hetero_bridge"

    # _optim_gpu_device — local CUDA device for param copies.
    local_device = torch.device(f"cuda:{torch.cuda.current_device()}")
    engine._optim_gpu_device = local_device

    # Move model to the local CUDA device.
    engine.model = engine.model.to(local_device)

    # ── 5. DES-LOC sync policy ─────────────────────────────────────────
    if sync_periods is None:
        # Mirror defaults from desloc_engine.py lines 770-772.
        sync_periods = _SP(kx=32, ku=96, kv=192)
    desloc_policy = DesLocSyncPolicy(sync_periods)

    # Classify parameters immediately so should_sync() works from step 0.
    try:
        desloc_policy.classify(named_params)
    except Exception as exc:
        logger.warning(
            "[hetero_bridge] DesLocSyncPolicy.classify() failed (%s); "
            "policy will use pre-classification Kx fallback.", exc
        )

    engine._desloc_policy = desloc_policy

    # ── 6. Pipeline schedule adapter ──────────────────────────────────
    num_layers = getattr(getattr(engine, "config", None), "num_layers", 32)
    pp_adapter = PPScheduleAdapter(tier_map, num_layers)

    # Cache layer_split immediately; result stored on engine for the training
    # loop and for logging (mirrors configs/7b_5gpu.yaml pp_layer_split).
    try:
        layer_split = pp_adapter.layer_split()
        engine._pp_layer_split = layer_split
        logger.info(
            "[hetero_bridge] pp_layer_split=%s (num_layers=%d, world_size=%d)",
            layer_split, num_layers, tier_map.world_size,
        )
    except Exception as exc:
        logger.warning(
            "[hetero_bridge] PPScheduleAdapter.layer_split() failed (%s); "
            "engine._pp_layer_split not set.", exc
        )
        engine._pp_layer_split = None

    engine._pp_adapter = pp_adapter

    # ── 7. AutoSP hook ─────────────────────────────────────────────────
    sp_group = getattr(engine, "_sp_group", None)
    sp_hook = AutoSPHook(sp_group, tier_map)

    # wrap_grad_reduction: installs per-param SP allreduce hooks.
    try:
        sp_hook.wrap_grad_reduction(adapter)
    except Exception as exc:
        logger.debug(
            "[hetero_bridge] AutoSPHook.wrap_grad_reduction() raised (%s).", exc
        )

    # sp_aware_sync: upgrades 'v'-class SP params to at least 'u'.
    try:
        sp_hook.sp_aware_sync(desloc_policy)
    except Exception as exc:
        logger.debug(
            "[hetero_bridge] AutoSPHook.sp_aware_sync() raised (%s).", exc
        )

    engine._autosp_hook = sp_hook

    # ── 8. finalize_grads_step callable ───────────────────────────────
    engine._hetero_finalize_grads = _make_finalize_grads_step(
        engine, adapter, desloc_policy
    )

    logger.info(
        "[hetero_bridge] install() complete — "
        "optimizer=%s, _cpu_offload=%s, tier_map=%s, "
        "kx=%d ku=%d kv=%d, pp_layer_split=%s",
        type(optimizer).__name__,
        engine._cpu_offload_optim,
        tier_map,
        sync_periods.kx, sync_periods.ku, sync_periods.kv,
        getattr(engine, "_pp_layer_split", None),
    )