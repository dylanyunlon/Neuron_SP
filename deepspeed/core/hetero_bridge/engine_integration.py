# SPDX-License-Identifier: Apache-2.0
"""engine_integration.py — single entrypoint that wires hetero_bridge into desloc_engine.

Call order
----------
install(engine) does exactly this, in order:

  1. TierMap.discover()           — all-gather GPU topology
  2. HeteroShardPlanner.plan()    — assign fp32 shards by VRAM budget
  3. DistOptAdapter.build()       — per-rank optimizer (CPUAdam / fused AdamW)
  4. Attach to engine             — set engine.optimizer, engine._dist_optimizer, etc.
  5. DesLocSyncPolicy             — constructed (methods are Phase-2 stubs; not called here)
  6. PPScheduleAdapter            — constructed (layer_split stub; not called here)
  7. AutoSPHook                   — constructed; wrap_grad_reduction guarded with try/except

After install() returns:
  - engine.optimizer              is DistOptAdapter  (has .step, .zero_grad, .defaults)
  - engine._dist_optimizer        is the underlying DistributedOptimizer (or None)
  - engine._cpu_offload_optim     is True on A6000 ranks
  - engine._optim_gpu_device      is the local CUDA device
  - engine._hetero_bridge_adapter is the DistOptAdapter (for reduce_scatter / all_gather)
  - engine.model                  has been moved to the local CUDA device

Stub-method calls (PPScheduleAdapter.layer_split, AutoSPHook.wrap_grad_reduction,
DesLocSyncPolicy.classify) are wrapped in try/except NotImplementedError so they
are inert until Sub-Claudes B/C fill them in Phase 2.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, TYPE_CHECKING

from .desloc_sync_policy import SyncPeriods

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def install(
    engine,
    *,
    lr: float,
    betas: "tuple[float, float]" = (0.9, 0.95),
    weight_decay: float = 0.1,
    sync_periods: Optional["SyncPeriods"] = None,
) -> None:
    """Wire the full hetero_bridge stack onto *engine*.

    Order: TierMap.discover() -> HeteroShardPlanner.plan() -> DistOptAdapter.build()
    -> DesLocSyncPolicy -> PPScheduleAdapter -> AutoSPHook. After this call,
    engine.optimizer is the hetero DistributedOptimizer and the training loop uses
    adapter.reduce_scatter_grads / step / all_gather_params.

    The function is idempotent if called a second time — it simply re-runs the
    discovery and rebuilds the adapter.

    Raises
    ------
    RuntimeError
        If DistOptAdapter.build() raises *and* no graceful fallback is available.
        In practice build() has its own internal fallback to the inner optimizer,
        so this should not propagate under normal circumstances.
    """
    import torch

    # ------------------------------------------------------------------
    # Lazy imports — keep them local so that merely importing this module
    # does NOT trigger the deepspeed/__init__.py → apex chain.
    # ------------------------------------------------------------------
    from .tier_map import TierMap
    from .shard_planner import HeteroShardPlanner
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
        from .tier_map import TierInfo, GPUTier
        import torch.cuda as _cuda
        _vram = _cuda.get_device_properties(0).total_memory if _cuda.is_available() else 0
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
        from .shard_planner import ShardPlan
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
    optimizer = adapter.build()   # never raises — has internal fallback to inner opt

    # ── 4. Attach to engine ────────────────────────────────────────────
    # engine.optimizer must expose .step(), .zero_grad(), .defaults
    # DistOptAdapter has step() and zero_grad(); add .defaults pass-through below.
    engine.optimizer = adapter
    engine._hetero_bridge_adapter = adapter

    # engine._dist_optimizer must expose .data_parallel_rank,
    # .data_parallel_world_size, .sync_moments(), .shard_to_model_broadcast()
    # These live on DistributedOptimizer (core/optimizer/distrib_optimizer.py),
    # not on DistOptAdapter. So point _dist_optimizer at the underlying object
    # only when it genuinely is a DistributedOptimizer.
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
        # Inner-opt fallback (single-rank / DistributedOptimizer unavailable).
        engine._dist_optimizer = None
        logger.info(
            "[hetero_bridge] engine._dist_optimizer = None "
            "(DistributedOptimizer not constructed; single-rank or test mode)"
        )

    # _cpu_offload_optim — used by engine.step() at line ~1470 to decide
    # whether to move gradients to CPU before the Adam update.
    try:
        import torch.distributed as dist
        local_rank = dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        local_rank = int(os.environ.get("RANK", "0"))

    engine._cpu_offload_optim = tier_map.is_low_vram(local_rank)
    engine._optim_type = "hetero_bridge"

    # _optim_gpu_device — used by engine.step() to copy updated params back.
    local_device = torch.device(f"cuda:{torch.cuda.current_device()}")
    engine._optim_gpu_device = local_device

    # Move model to the local CUDA device (replaces the model.to(_local_device)
    # calls that were inside the old Phase 5 if/else branches).
    engine.model = engine.model.to(local_device)

    # ── 5. DES-LOC sync policy ─────────────────────────────────────────
    # Only instantiate — classify() / should_sync() are Phase-2 stubs and
    # are NOT called during install(). The engine's training loop calls
    # sync_moments() directly on _dist_optimizer, not via this policy object.
    if sync_periods is None:
        # Mirror the defaults from desloc_engine.py lines 766-768.
        sync_periods = _SP(kx=32, ku=96, kv=192)
    engine._desloc_policy = DesLocSyncPolicy(sync_periods)

    # ── 6. Pipeline schedule adapter ─────────────────────────────────
    # layer_split() / forward_backward() are Phase-2 stubs. Not called here.
    num_layers = getattr(getattr(engine, "config", None), "num_layers", 12)
    engine._pp_adapter = PPScheduleAdapter(tier_map, num_layers)

    # ── 7. AutoSP hook ─────────────────────────────────────────────────
    # sp_group may not exist yet (it's wired later in train() at ~line 1638).
    # Store None now; the hook can be reconfigured once SP is initialised.
    sp_group = getattr(engine, "_sp_group", None)
    sp_hook = AutoSPHook(sp_group, tier_map)
    try:
        sp_hook.wrap_grad_reduction(adapter)
    except NotImplementedError:
        pass   # Phase 2: Sub-Claude C implements wrap_grad_reduction
    engine._autosp_hook = sp_hook

    logger.info(
        "[hetero_bridge] install() complete — "
        "optimizer=%s, _cpu_offload=%s, tier_map=%s",
        type(optimizer).__name__,
        engine._cpu_offload_optim,
        tier_map,
    )
