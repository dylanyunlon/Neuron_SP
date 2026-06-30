"""
Checkpoint save/load logic for DesLocEngine.

Split out of deepspeed/runtime/desloc_engine.py, following the Megatron-LM
pattern of keeping checkpointing.py as a sibling module to the main training
loop (see Megatron-LM/megatron/training/{checkpointing,training}.py).

Both functions take the owning DesLocEngine instance as their first
argument (mirroring Megatron's free-function style, e.g.
`save_checkpoint(iteration, model, optimizer, ...)`) rather than being bound
methods, so they can be tested, profiled, and evolved independently of the
rest of the engine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

import deepspeed.core.parallel_state as parallel_state

logger = logging.getLogger(__name__)


def save_checkpoint(engine, path: Path) -> None:
    """
    Save a full training checkpoint to disk.

    When config.use_dist_checkpointing is True, delegates to the async
    sharded strategy from deepspeed/core/dist_checkpointing/ (faster,
    non-blocking saves). Otherwise falls through to the existing
    hetero/torch.save path.

    When a :class:`HeteroCheckpointConfig` is active the save is routed
    through the per-tier async pipeline:

    * **CACHE tier (H100)** — optimizer state is first staged to the
      host-DRAM locality cache (``cfg.locality_cache_path(step)``), which
      maps to a ramdisk / tmpfs on the 1.5 TB DDR5 host.  A
      ``hetero_metadata.pt`` index is written there so that
      :meth:`load_checkpoint` can rediscover the staged tensors on resume.
      The staged state is then persisted asynchronously to *path* by
      :class:`~deepspeed.checkpoint.hetero_async_checkpoint_save.HeteroAsyncCheckpointScheduler`,
      allowing the next forward pass to begin immediately.
    * **WORKER tiers (A6000)** — parameter shards are written
      synchronously to *path* when ``worker_offload_optim=True`` (optimizer
      state is owned by the CACHE tier and omitted here).

    When no hetero config is available (CPU-only fallback) the method
    reverts to a plain :func:`torch.save`.

    Args:
        path: Destination file/directory path (parent dirs created as needed).
    """
    # Re-use pre-imported symbols from engine._lazy to avoid repeated import
    # overhead and stay consistent with the lazy-import contract.
    _build_async_pipeline = engine._lazy["build_hetero_async_save_pipeline"]
    _validate_async_cfg   = engine._lazy["validate_async_checkpoint_config"]
    _detect_arch          = engine._lazy["detect_device_arch"]
    _DeviceArch           = engine._lazy["DeviceArch"]
    _TierRole             = engine._lazy["TierRole"]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # --- Core dist_checkpointing adapter (gated by config.use_dist_checkpointing) ---
    from deepspeed.runtime.core_adapters import build_dist_checkpoint_saver
    _dc_saver = build_dist_checkpoint_saver(engine.config)

    payload = {
        "global_step":       engine.global_step,
        "tokens_seen":       engine.tokens_seen,
        "model_state":       engine.model.state_dict(),
        "optimizer_state":   engine.optimizer.state_dict(),
        "scheduler_state":   engine.scheduler.state_dict(),
        "plan":              engine.plan,
        "config":            engine.config,
    }

    # From Megatron M2869 (PR #2658): RNG state must be sharded by
    # (PP, TP, DP) when EP > 1 to avoid aliased RNG state across EP ranks,
    # which causes different experts to use the same dropout / noise seeds.
    # On DES-LOC (heterogeneous tiers, no NVLink), EP ranks may live on
    # different GPU models, making RNG aliasing especially hard to debug.
    # We save (pp_rank, tp_rank, ep_rank) as a shard key alongside the
    # RNG tensors so the loader can broadcast the correct state per-rank.
    try:
        import torch
        from deepspeed.core import parallel_state as _ps
        _ep_size = _ps.get_expert_model_parallel_world_size() \
            if hasattr(_ps, "get_expert_model_parallel_world_size") else 1
        _rng_shard_key = {
            "pp_rank": _ps.get_pipeline_model_parallel_rank()
                if hasattr(_ps, "get_pipeline_model_parallel_rank") else 0,
            "tp_rank": _ps.get_tensor_model_parallel_rank()
                if hasattr(_ps, "get_tensor_model_parallel_rank") else 0,
            # Include ep_rank when EP>1 so each expert-parallel rank saves
            # a distinct RNG state (M2869: avoid RNG aliasing across EP).
            "ep_rank": _ps.get_expert_model_parallel_rank()
                if (_ep_size > 1 and hasattr(_ps, "get_expert_model_parallel_rank")) else 0,
            "ep_size": _ep_size,
        }
        payload["rng_state"] = {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "shard_key": _rng_shard_key,
        }
    except Exception:
        pass  # RNG state save is best-effort

    cfg = engine._hetero_ckpt_cfg  # may be None in CPU-only mode

    if cfg is not None and False:  # DISABLED: DCP async save deadlocks with training all_reduce (dcp.save calls gather_object→allgather in background thread while main thread does grad all_reduce)
        # ------------------------------------------------------------------
        # Hetero async path
        # Step 1: classify the current device into a TierRole using the
        #         SM architecture reported by HeteroCheckpointConfig.
        # ------------------------------------------------------------------
        current_device = engine.primary_device
        if current_device.type == "cuda":
            arch = _detect_arch(current_device)
            tier_role = (
                _TierRole.CACHE
                if arch == _DeviceArch.SM90_H100
                else _TierRole.WORKER
            )
        else:
            tier_role = _TierRole.WORKER   # CPU fallback treated as WORKER

        tier_policy = cfg.get_policy(tier_role)

        # Step 2: apply per-tier save policy.
        # WORKER tiers skip optimizer state when worker_offload_optim=True
        # (the CACHE tier owns the full optimizer checkpoint).
        if not tier_policy.save_optim and "optimizer_state" in payload:
            logger.info(
                "[hetero_ckpt] WORKER tier: omitting optimizer_state "
                "(offloaded to CACHE tier per worker_offload_optim)."
            )
            payload.pop("optimizer_state")

        ckpt_format = cfg.hetero_ckpt_format.value  # e.g. "torch_dist"

        # Step 3 (CACHE tier only): stage the full payload to the
        # host-DRAM locality cache before handing off to async IO.
        # This gives sub-second fast-resume capability from /dev/shm.
        # The staging itself runs in a background thread so the training
        # loop returns immediately after submitting the IO work.
        if tier_role == _TierRole.CACHE and cfg.locality_cache_dir is not None:
            lc_path = cfg.locality_cache_path(engine.global_step)
            if lc_path is not None:
                # Detach all tensors to CPU before the thread takes them —
                # GPU tensors cannot be safely serialised from a background
                # thread while the main thread uses the same CUDA context.
                _stage_payload = {
                    k: (v.cpu().detach().clone() if isinstance(v, torch.Tensor) else v)
                    for k, v in payload.items()
                }
                # Recursively CPU-detach nested optimizer state tensors so
                # that CUDA context access from the worker thread is safe.
                if "optimizer_state" in _stage_payload:
                    def _cpu_detach_state(obj):
                        if isinstance(obj, torch.Tensor):
                            return obj.cpu().detach().clone()
                        if isinstance(obj, dict):
                            return {k: _cpu_detach_state(v) for k, v in obj.items()}
                        if isinstance(obj, (list, tuple)):
                            return type(obj)(_cpu_detach_state(v) for v in obj)
                        return obj
                    _stage_payload["optimizer_state"] = _cpu_detach_state(
                        _stage_payload["optimizer_state"]
                    )

                def _do_stage(lc_path_=lc_path, payload_=_stage_payload):
                    try:
                        lc_path_.mkdir(parents=True, exist_ok=True)
                        meta_file = lc_path_ / "hetero_metadata.pt"
                        torch.save(payload_, meta_file)
                        logger.info(
                            "[hetero_ckpt] CACHE tier: staged payload to "
                            "locality cache %s (%.0f MB).",
                            meta_file,
                            meta_file.stat().st_size / (1 << 20),
                        )
                    except Exception as _lc_exc:  # noqa: BLE001
                        logger.warning(
                            "[hetero_ckpt] locality_cache staging failed (%s); "
                            "async stage thread exiting.",
                            _lc_exc,
                        )

                _fut = engine._cpu_stage_executor.submit(_do_stage)
                with engine._cpu_stage_lock:
                    # Prune already-done futures to avoid unbounded growth.
                    engine._cpu_stage_futures = [
                        f for f in engine._cpu_stage_futures if not f.done()
                    ]
                    engine._cpu_stage_futures.append(_fut)
                logger.info(
                    "[hetero_ckpt] CACHE tier: locality-cache stage submitted "
                    "asynchronously → %s (step %d).",
                    lc_path, engine.global_step,
                )

        # Step 3b (WORKER tier): CPU-stage FP32 shard to locality cache.
        if (
            tier_role == _TierRole.WORKER
            and cfg.locality_cache_dir is not None
            and engine._dist_optimizer is not None
        ):
            lc_path_w = cfg.locality_cache_path(engine.global_step)
            if lc_path_w is not None:
                _shard_cpu = (
                    engine._dist_optimizer._fp32_shards[0].cpu().detach().clone()
                    if engine._dist_optimizer._fp32_shards
                    else None
                )
                if _shard_cpu is not None:
                    def _do_worker_stage(
                        lc_path_=lc_path_w,
                        shard_=_shard_cpu,
                        step_=engine.global_step,
                    ):
                        try:
                            lc_path_.mkdir(parents=True, exist_ok=True)
                            shard_file = lc_path_ / "param_shard.pt"
                            torch.save({"param_shard": shard_, "global_step": step_},
                                       shard_file)
                            logger.info(
                                "[hetero_ckpt] WORKER tier: param shard staged to "
                                "locality cache %s (%.1f MB).",
                                shard_file,
                                shard_file.stat().st_size / (1 << 20),
                            )
                        except Exception as _ws_exc:  # noqa: BLE001
                            logger.warning(
                                "[hetero_ckpt] WORKER param-shard stage failed (%s).",
                                _ws_exc,
                            )

                    _fut_w = engine._cpu_stage_executor.submit(_do_worker_stage)
                    with engine._cpu_stage_lock:
                        engine._cpu_stage_futures = [
                            f for f in engine._cpu_stage_futures if not f.done()
                        ]
                        engine._cpu_stage_futures.append(_fut_w)
                    logger.info(
                        "[hetero_ckpt] WORKER tier: param-shard stage submitted "
                        "asynchronously → %s (step %d).",
                        lc_path_w, engine.global_step,
                    )

        # Step 4: launch the async save pipeline (CACHE and WORKER tiers).
        try:
            _validate_async_cfg(
                ckpt_format,
                async_save=True,
                require_nvrx_for_dcp=False,  # graceful fallback if no NVRx
            )
            logger.info(
                "[hetero_ckpt] Launching async save to %s "
                "(tier=%s, format=%s, async=%s, scheduler=%s).",
                path, tier_role.value, ckpt_format,
                tier_policy.async_save,
                "reused" if engine._hetero_ckpt_scheduler is not None else "new",
            )
            engine._hetero_ckpt_scheduler = _build_async_pipeline(
                state_dict=payload,
                checkpoint_path=str(path),
                ckpt_format=ckpt_format,
                iteration=engine.global_step,
                scheduler=engine._hetero_ckpt_scheduler,
            )
            logger.info(
                "[hetero_ckpt] Async save scheduled: %s (step %d, tier=%s).",
                path, engine.global_step, tier_role.value,
            )
            return
        except (NotImplementedError, RuntimeError) as _async_err:
            # Format not async-eligible or NVRx missing: fall through to
            # synchronous save so training is never blocked.
            logger.warning(
                "[hetero_ckpt] Async save unavailable (%s); "
                "falling back to torch.save.",
                _async_err,
            )

    # ------------------------------------------------------------------
    # Dispatch: dist_checkpointing (when adapter is active) or legacy
    # synchronous per-rank torch.save.
    # ------------------------------------------------------------------
    if _dc_saver is not None:
        # Phase-1 wire-up (PROPOSAL_checkpoint_migration.md §3 Step 1):
        # route S3/S4 through the DistCheckpointAdapter so that
        # dist_checkpointing.save() handles sharded IO instead of each
        # rank writing a full monolithic .pt file.
        #
        # path must be a directory for dist_checkpointing; create it here
        # (empty) so serialization.save()'s non-empty check passes.
        ckpt_dir = Path(path) if not Path(path).suffix else Path(path).parent / Path(path).stem
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        _dc_saver.save(payload, ckpt_dir)
        logger.info(
            "dist_checkpoint saved: %s (step %d)",
            ckpt_dir, engine.global_step,
        )
    else:
        # ------------------------------------------------------------------
        # Legacy synchronous per-rank save.  Each rank saves its own shard
        # to a rank-suffixed file.  No collective communication needed.
        # ------------------------------------------------------------------
        _rank = (
            parallel_state.get_data_parallel_rank()
            if parallel_state.is_initialized()
            else (dist.get_rank() if dist.is_initialized() else 0)
        )
        _world = (
            parallel_state.get_data_parallel_world_size()
            if parallel_state.is_initialized()
            else (dist.get_world_size() if dist.is_initialized() else 1)
        )
        # For rank 0 (or single-GPU), save full payload.
        # For other ranks, save only param_shard + step (optimizer is redundant
        # since each rank has its own shard's Adam states).
        if _world == 1:
            torch.save(payload, path)
            logger.info("Checkpoint saved: %s (step %d)", path, engine.global_step)
        else:
            path = Path(path)
            ckpt_dir = path.parent / path.stem
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            rank_path = ckpt_dir / f"rank_{_rank}.pt"
            torch.save(payload, rank_path)
            logger.info(
                "Checkpoint saved: %s (rank %d/%d, step %d, %.1f MB)",
                rank_path, _rank, _world, engine.global_step,
                rank_path.stat().st_size / (1 << 20),
            )
    if dist.is_initialized():
        dist.barrier()  # all ranks wait for IO to finish


def load_checkpoint(engine, path: Path) -> None:
    """
    Resume training from a saved checkpoint.

    Tier-aware loading strategy driven by :class:`HeteroCheckpointConfig`:

    1. **Locality-cache fast-resume** — when ``cfg.locality_cache_dir`` is
       set, the method first scans for a ``hetero_metadata.pt`` written by
       the CACHE tier into the host-DRAM ramdisk during the most recent
       async save.  Because the ramdisk is in CPU DRAM (not on disk), this
       path avoids storage IO entirely and resumes in <1 s on the target
       3-GPU cluster.

    2. **Tier-aware HeteroAsyncCheckpointLoad** — when
       ``cfg.shard_rebalance_on_load=True`` and *path* is a directory,
       uses :class:`~deepspeed.checkpoint.hetero_async_checkpoint_load.HeteroAsyncCheckpointLoad`
       to restore tensors through the CPU-DRAM staging pipeline with SM-arch
       routing (H100 vs A6000).  This handles heterogeneous resume when the
       saved tier layout differs from the current one.

    3. **Legacy synchronous fallback** — plain :func:`torch.load` from a
       ``.pt`` file.

    Post-load behaviour is governed by ``cfg.load_optim``,
    ``cfg.load_rng``, and ``cfg.dist_ckpt_strictness``.

    Args:
        path: Path to the checkpoint directory (hetero format) or ``.pt``
              file (legacy format).

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    # Re-use pre-imported symbols from engine._lazy.
    _HeteroLoad   = engine._lazy["HeteroAsyncCheckpointLoad"]
    _detect_arch  = engine._lazy["detect_device_arch"]
    _DeviceArch   = engine._lazy["DeviceArch"]

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    cfg = engine._hetero_ckpt_cfg  # may be None when called from __init__

    # ------------------------------------------------------------------
    # Stage 1: locality-cache fast-resume (host-DRAM, sub-second).
    # Try every step subdirectory under locality_cache_dir, newest first.
    # ------------------------------------------------------------------
    if cfg is not None and cfg.locality_cache_dir is not None:
        import glob as _glob  # noqa: PLC0415
        lc_base = Path(cfg.locality_cache_dir)
        # Enumerate step_XXXXXXXXXX subdirectories and sort descending
        # so we always try the most recent staged checkpoint first.
        lc_step_dirs = sorted(lc_base.glob("step_*"), reverse=True)
        for lc_step_dir in lc_step_dirs:
            meta_file = lc_step_dir / "hetero_metadata.pt"
            # Also accept rank-specific shards written by locality_cache_path()
            if not meta_file.exists():
                rank_shards = list(lc_step_dir.glob("rank_*"))
                for shard_dir in rank_shards:
                    candidate = shard_dir / "hetero_metadata.pt"
                    if candidate.exists():
                        meta_file = candidate
                        break
            if not meta_file.exists():
                continue
            try:
                logger.info(
                    "[hetero_ckpt] Fast-resume: loading from locality cache %s.",
                    meta_file,
                )
                payload = torch.load(meta_file, map_location="cpu")
                engine._apply_loaded_state(payload, cfg)
                logger.info(
                    "[hetero_ckpt] Fast-resume complete from locality cache "
                    "(step %d, %.2fM tokens seen).",
                    engine.global_step, engine.tokens_seen / 1e6,
                )
                return
            except Exception as _lc_exc:  # noqa: BLE001
                logger.warning(
                    "[hetero_ckpt] locality-cache load failed (%s); "
                    "continuing to persistent-path load.",
                    _lc_exc,
                )
                break  # one failure → skip remaining cache entries

    # ------------------------------------------------------------------
    # Stage 2: tier-aware HeteroAsyncCheckpointLoad from persistent path.
    # ------------------------------------------------------------------
    if (
        cfg is not None
        and cfg.shard_rebalance_on_load
        and path.is_dir()   # hetero checkpoints are directories
    ):
        logger.info(
            "[hetero_ckpt] Attempting tier-aware load from %s "
            "(shard_rebalance_on_load=True).", path
        )

        # Discover device topology from tiers (populated by TierDiscovery).
        h100_device: Optional[torch.device] = None
        a6000_devices: List[torch.device] = []
        for spec in getattr(engine, "tiers", []):
            dev = spec.device
            arch = _detect_arch(dev)
            if arch == _DeviceArch.SM90_H100 and h100_device is None:
                h100_device = dev
            elif arch == _DeviceArch.SM86_A6000:
                a6000_devices.append(dev)

        if h100_device is None:
            h100_device = engine.primary_device

        try:
            loader = _HeteroLoad(
                checkpoint_dir=str(path),
                h100_device=h100_device,
                a6000_devices=a6000_devices or [engine.primary_device],
                slc_capacity_gb=min(cfg.locality_cache_max_gb, 64.0),
                io_workers=8,
                h100_budget_gb=80.0,
                a6000_budget_gb=40.0,
            )

            # Build shard metadata from hetero_metadata.pt written by the
            # CACHE tier during save_checkpoint().
            meta_file = path / "hetero_metadata.pt"
            if meta_file.exists():
                state_dict_meta: Dict[str, Any] = torch.load(
                    meta_file, map_location="cpu"
                )
                loaded_state = loader.load(state_dict_meta)
                loader.shutdown()
                engine._apply_loaded_state(loaded_state, cfg)
                logger.info(
                    "[hetero_ckpt] Tier-aware load complete: %s "
                    "(step %d, %.2fM tokens seen).",
                    path, engine.global_step, engine.tokens_seen / 1e6,
                )
                return
            else:
                loader.shutdown()
                logger.info(
                    "[hetero_ckpt] No hetero_metadata.pt in %s; "
                    "falling back to torch.load.",
                    path,
                )
        except Exception as _load_exc:  # noqa: BLE001
            logger.warning(
                "[hetero_ckpt] HeteroAsyncCheckpointLoad failed (%s); "
                "falling back to torch.load.",
                _load_exc,
            )

    # ------------------------------------------------------------------
    # Stage 3: dist_checkpointing directory or legacy torch.load fallback.
    # ------------------------------------------------------------------
    # Build the adapter here (mirrors save_checkpoint).  Returns None when
    # use_dist_checkpointing=False (default) so legacy behaviour is unchanged.
    from deepspeed.runtime.core_adapters import build_dist_checkpoint_saver  # noqa: PLC0415
    _dc_saver = build_dist_checkpoint_saver(engine.config)

    from deepspeed.core.dist_checkpointing import check_is_distributed_checkpoint  # noqa: PLC0415
    if _dc_saver is not None and path.is_dir() and check_is_distributed_checkpoint(str(path)):
        # Phase-1 wire-up (PROPOSAL_checkpoint_migration.md §3 Step 1):
        # checkpoint was written by dist_checkpointing.save(); use the
        # adapter to load so common.pt + shard_*.pt are handled correctly.
        logger.info(
            "dist_checkpoint load: %s", path,
        )
        payload = _dc_saver.load(path)
        engine._apply_loaded_state(payload, cfg)
        logger.info(
            "dist_checkpoint loaded: %s (step %d, %.2fM tokens seen)",
            path, engine.global_step, engine.tokens_seen / 1e6,
        )
    else:
        # Legacy / synchronous fallback — torch.load of a .pt file.
        pt_file = path if path.suffix == ".pt" else path
        payload = torch.load(
            pt_file,
            map_location=f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available() else "cpu",
        )
        engine._apply_loaded_state(payload, cfg)
        logger.info(
            "Checkpoint loaded (legacy): %s (step %d, %.2fM tokens seen)",
            path,
            engine.global_step,
            engine.tokens_seen / 1e6,
        )
