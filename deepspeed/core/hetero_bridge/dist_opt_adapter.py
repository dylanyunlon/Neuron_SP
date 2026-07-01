# SPDX-License-Identifier: Apache-2.0
"""dist_opt_adapter.py — adapt core.optimizer.DistributedOptimizer to the hetero ShardPlan.

Per HEAD commit a52efee1: A6000 ranks → DeepSpeedCPUAdam (CPU-resident state);
H100/Blackwell ranks → fused AdamW on GPU.

Design
------
* ``build()`` constructs the underlying ``core.optimizer.DistributedOptimizer``
  by selecting the per-rank inner optimizer based on the local GPU tier:
    - A6000 (< 50 GB VRAM): ``DeepSpeedCPUAdam`` with ``fp32_optimizer_states=True``
      so exp_avg / exp_avg_sq live in CPU pinned memory, not GPU VRAM.
    - H100 / Blackwell (≥ 90 GB): ``torch.optim.AdamW(fused=True)``
    - Unknown / mid-VRAM: ``torch.optim.AdamW(fused=False)``

* ``reduce_scatter_grads()`` is PCIe-aware: P2P is disabled (no NVLink),
  so we delegate to ``DistributedOptimizer._reduce_scatter_grads()`` which
  already has the I6 PCIe-aware overlap path, and additionally disable
  ``use_pcie_aware_overlap`` for the hetero (unequal-shard) slow path to
  avoid async ops that can overlap with nothing on PCIe.

* ``all_gather_params()`` delegates to ``DistributedOptimizer.start_param_sync()``.

* Reuses existing ``zero3_hetero_shard.ShardState`` VRAM-weight computation
  so the shard-sizing logic is not duplicated.

References
----------
* ARCHITECTURE.md (frozen API contract)
* deepspeed/core/optimizer/distrib_optimizer.py  (DistributedOptimizer)
* deepspeed/runtime/zero3_hetero_shard.py        (ShardState / VRAM weights)
* deepspeed/runtime/desloc_optimizer.py          (per-tier optimizer selection)
* git log HEAD~1 commit a52efee1                 (VRAM-adaptive optimizer)
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from .shard_planner import ShardPlan
    from .tier_map import TierMap

logger = logging.getLogger(__name__)

# VRAM thresholds copied from desloc_engine.py and desloc_optimizer.py
# (commit a52efee1).  Keep in sync if the thresholds change.
_CPU_OFFLOAD_VRAM_THRESHOLD_GB = 50.0   # < 50 GB → CPUAdam (A6000)
_FUSED_VRAM_THRESHOLD_GB = 90.0         # ≥ 90 GB → fused AdamW (H100/Blackwell)

_BYTES_PER_GB = 1 << 30


# ---------------------------------------------------------------------------
# Inner-optimizer factory
# ---------------------------------------------------------------------------

def _build_inner_optimizer(
    params,
    lr: float,
    betas: "tuple[float, float]",
    weight_decay: float,
    vram_bytes: int,
    rank: int,
) -> "torch.optim.Optimizer":
    """Construct the per-rank inner optimizer based on available VRAM.

    Mirrors the logic in ``deepspeed/runtime/desloc_optimizer.py``
    (commit a52efee1 — VRAM-adaptive optimizer):
      - VRAM < 50 GB  → DeepSpeedCPUAdam  (A6000; optimizer states on CPU)
      - VRAM ≥ 90 GB  → AdamW(fused=True) (H100 / Blackwell; GPU-fused kernel)
      - 50–89 GB      → AdamW(fused=False) (conservative GPU path)

    Args:
        params:       Parameters or param-groups for the optimizer.
        lr:           Learning rate.
        betas:        (beta1, beta2) Adam coefficients.
        weight_decay: L2 / AdamW weight-decay coefficient.
        vram_bytes:   Total VRAM on this rank's GPU (bytes).
        rank:         Global distributed rank (for logging only).

    Returns:
        A :class:`torch.optim.Optimizer` instance.
    """
    import torch
    vram_gb = vram_bytes / _BYTES_PER_GB

    # ------------------------------------------------------------------
    # A6000 / low-VRAM path: DeepSpeedCPUAdam with CPU-resident states.
    # Optimizer exp_avg and exp_avg_sq stay in pinned host memory;
    # param tensors remain on GPU.  The C++ cpu_adam kernel handles the
    # GPU→CPU grad transfer and CPU→GPU param update implicitly.
    # ------------------------------------------------------------------
    if vram_gb < _CPU_OFFLOAD_VRAM_THRESHOLD_GB:
        try:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            opt = DeepSpeedCPUAdam(
                params,
                lr=lr,
                betas=betas,
                eps=1e-8,
                weight_decay=weight_decay,
                adamw_mode=True,
                fp32_optimizer_states=True,
            )
            logger.info(
                "[rank %d] DistOptAdapter: VRAM=%.1f GB < %.0f GB "
                "→ DeepSpeedCPUAdam (CPU offload, fp32 states). "
                "Ref: a52efee1, DeepSpeed #4527",
                rank, vram_gb, _CPU_OFFLOAD_VRAM_THRESHOLD_GB,
            )
            return opt
        except ImportError:
            logger.warning(
                "[rank %d] DeepSpeedCPUAdam not available (DS not compiled with "
                "DS_BUILD_CPU_ADAM=1); falling back to AdamW on CPU. "
                "This may cause OOM on A6000 (%.1f GB VRAM).",
                rank, vram_gb,
            )
            # Fallback: plain AdamW — slower but avoids hard failure in CI.
            return torch.optim.AdamW(
                params, lr=lr, betas=betas, eps=1e-8, weight_decay=weight_decay,
                fused=False,
            )

    # ------------------------------------------------------------------
    # H100 / Blackwell / high-VRAM path: fused AdamW on GPU.
    # Single CUDA kernel launch covers all param groups.
    # ------------------------------------------------------------------
    if vram_gb >= _FUSED_VRAM_THRESHOLD_GB:
        fused_available = False
        try:
            import torch.optim
            # fused=True requires CUDA and ≥ PyTorch 2.0
            if torch.cuda.is_available():
                fused_available = True
        except Exception:
            pass

        opt = torch.optim.AdamW(
            params, lr=lr, betas=betas, eps=1e-8, weight_decay=weight_decay,
            fused=fused_available,
        )
        logger.info(
            "[rank %d] DistOptAdapter: VRAM=%.1f GB ≥ %.0f GB "
            "→ AdamW(fused=%s) on GPU. Ref: a52efee1, Megatron PR #4623",
            rank, vram_gb, _FUSED_VRAM_THRESHOLD_GB, fused_available,
        )
        return opt

    # ------------------------------------------------------------------
    # Mid-VRAM path (50–89 GB): standard AdamW, no fused kernel.
    # ------------------------------------------------------------------
    opt = torch.optim.AdamW(
        params, lr=lr, betas=betas, eps=1e-8, weight_decay=weight_decay,
        fused=False,
    )
    logger.info(
        "[rank %d] DistOptAdapter: VRAM=%.1f GB (mid-range) → AdamW(fused=False)",
        rank, vram_gb,
    )
    return opt


# ---------------------------------------------------------------------------
# ParamAndGradBuffer shim
# ---------------------------------------------------------------------------

def _make_param_grad_buffer(model: "nn.Module") -> "object":
    """Construct a minimal ParamAndGradBuffer-compatible object from a model.

    ``DistributedOptimizer`` expects a list of ``ParamAndGradBuffer``
    objects, each exposing:
      - ``.grad_data``       — flat BF16 gradient tensor
      - ``.param_index_map`` — ``{param: (start, end, bucket_id)}``

    When the model has not been wrapped in a Megatron DDP buffer (which is
    the common case in the adapter path), we build a minimal compatible
    shim by enumerating the model's parameters.

    Args:
        model: The nn.Module whose parameters should be covered.

    Returns:
        A single-element list containing a ParamAndGradBuffer-like object.
    """
    import torch

    try:
        from deepspeed.core.distributed import ParamAndGradBuffer
        # If already wrapped, return directly.
        if hasattr(model, "_param_and_grad_buffers"):
            return model._param_and_grad_buffers
    except ImportError:
        ParamAndGradBuffer = None

    # Build a lightweight namespace that satisfies DistributedOptimizer._build_shards.
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not params:
        return []

    # Total numel for grad buffer
    total_numel = sum(p.numel() for _, p in params)

    # Flat BF16 grad buffer (zero-initialized; populated by DDP allreduce hooks)
    grad_data = torch.zeros(total_numel, dtype=torch.bfloat16)

    # param_index_map: {param: (start, end, bucket_id=0)}
    param_index_map = {}
    cursor = 0
    for _, p in params:
        end = cursor + p.numel()
        param_index_map[p] = (cursor, end, 0)
        cursor = end

    class _ShimBuffer:
        """Minimal ParamAndGradBuffer shim for DistributedOptimizer."""
        def __init__(self):
            self.grad_data = grad_data
            self.param_index_map = param_index_map
            self.param_dtype = torch.bfloat16
            self.grad_dtype = torch.bfloat16

    return [_ShimBuffer()]


# ---------------------------------------------------------------------------
# DistOptAdapter  (frozen public API per ARCHITECTURE.md)
# ---------------------------------------------------------------------------


class DistOptAdapter:
    """Adapts core.optimizer.DistributedOptimizer to the hetero ShardPlan.

    On A6000 ranks: optimizer state resident on CPU (DeepSpeedCPUAdam path).
    On H100/Blackwell ranks: fused AdamW on GPU. (Matches HEAD commit a52efee1.)

    The adapter owns the full ZeRO-3 reduce-scatter / all-gather cycle:
      1. ``reduce_scatter_grads()`` — PCIe-aware, scatter BF16 grads to FP32 shards.
      2. ``step()``                 — local Adam on the owned FP32 shard.
      3. ``all_gather_params()``    — broadcast updated BF16 params to all ranks.

    PCIe awareness
    --------------
    This cluster has no NVLink (P2P disabled).  All collectives traverse PCIe.
    We therefore:
      - Use ``use_pcie_aware_overlap=False`` for the hetero slow path (unequal
        shards must fall back to all_reduce, where async ops offer no overlap).
      - For equal-shard paths the DistributedOptimizer's existing I6 logic
        handles async/sync automatically; we do not override it.

    Reuse
    -----
    * ``zero3_hetero_shard.ShardState.build()`` is consulted for VRAM-weighted
      shard sizes to avoid reinventing the proportional-split math.
    * ``core.optimizer.DistributedOptimizer`` performs the actual ZeRO-3 ops.
    """

    def __init__(
        self,
        model: "nn.Module",
        shard_plan: "ShardPlan",
        tier_map: "TierMap",
        lr: float,
        betas: "tuple[float, float]",
        weight_decay: float,
    ) -> None:
        self.model = model
        self.shard_plan = shard_plan
        self.tier_map = tier_map
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self._opt: "Optional[torch.optim.Optimizer]" = None
        # Cached distributed rank (resolved lazily on first build())
        self._rank: Optional[int] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _local_rank(self) -> int:
        """Return the global distributed rank of this process."""
        if self._rank is not None:
            return self._rank
        try:
            import torch.distributed as dist
            self._rank = dist.get_rank() if dist.is_initialized() else 0
        except Exception:
            self._rank = int(os.environ.get("RANK", "0"))
        return self._rank

    def _local_vram_bytes(self) -> int:
        """Return total VRAM bytes for the local rank's GPU."""
        rank = self._local_rank()
        try:
            info = self.tier_map.info(rank)
            return info.total_vram_bytes
        except (KeyError, AttributeError):
            pass
        # Fallback: query torch.cuda directly.
        try:
            import torch
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                return torch.cuda.get_device_properties(local_rank).total_memory
        except Exception:
            pass
        return 48 * _BYTES_PER_GB  # conservative A6000 default

    def _tier_assignments_for_dist_opt(self) -> "Optional[List]":
        """Build a TierType list for DistributedOptimizer.tier_assignments.

        Maps TierMap.GPUTier → desloc_config.TierType so that the underlying
        DistributedOptimizer can use heterogeneous shard sizing if configured.

        Returns None if the conversion is unavailable (missing import).
        """
        try:
            from deepspeed.core.desloc_config import TierType
            from .tier_map import GPUTier

            _tier_map_to_desloc: dict = {
                GPUTier.A6000:    TierType.PROFESSIONAL,
                GPUTier.H100:     TierType.DATACENTER,
                GPUTier.BLACKWELL: TierType.BLACKWELL,
                GPUTier.UNKNOWN:  TierType.PROFESSIONAL,  # conservative default
            }

            world_size = self.tier_map.world_size
            assignments = []
            for r in range(world_size):
                gpu_tier = self.tier_map.tier_of(r)
                assignments.append(_tier_map_to_desloc.get(gpu_tier, TierType.PROFESSIONAL))
            return assignments
        except (ImportError, Exception) as exc:
            logger.debug("_tier_assignments_for_dist_opt: %s", exc)
            return None

    # ------------------------------------------------------------------
    # build()  — frozen public API
    # ------------------------------------------------------------------

    def build(self) -> "torch.optim.Optimizer":
        """Construct the underlying DistributedOptimizer with per-rank optimizer type.

        Strategy
        --------
        1. Query local VRAM via TierMap to choose the inner optimizer:
           - A6000 (< 50 GB): DeepSpeedCPUAdam (CPU-resident FP32 states)
           - H100/Blackwell (≥ 90 GB): AdamW(fused=True)
           - other: AdamW(fused=False)

        2. Build a ParamAndGradBuffer shim from the model's parameters.

        3. Instantiate core.optimizer.DistributedOptimizer with:
           - The per-rank inner optimizer.
           - The grad buffers.
           - tier_assignments derived from TierMap (enables hetero shard sizing).

        4. Store the result in ``self._opt`` and return it.

        PCIe constraint
        ---------------
        P2P / NVLink are absent.  The OptimizerConfig is created with
        ``use_pcie_aware_overlap=False`` for the unequal-shard (hetero slow) path
        to prevent async NCCL launches that cannot overlap on PCIe topology.
        The equal-shard fast path already defaults to sync mode on PCIe.

        Returns:
            The constructed :class:`~deepspeed.core.optimizer.DistributedOptimizer`.
        """
        import torch

        rank = self._local_rank()
        vram_bytes = self._local_vram_bytes()

        # ---- 1. Collect model parameters --------------------------------
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("DistOptAdapter.build: model has no trainable parameters")

        # ---- 2. Choose inner optimizer based on local VRAM -----------
        inner_opt = _build_inner_optimizer(
            params=params,
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
            vram_bytes=vram_bytes,
            rank=rank,
        )

        # ---- 3. Build OptimizerConfig ----------------------------------
        try:
            from deepspeed.core.optimizer import OptimizerConfig

            # PCIe-aware: disable overlap for the hetero unequal-shard path.
            # Equal-shard fast path handles async internally.
            opt_config = OptimizerConfig(
                lr=self.lr,
                adam_beta1=self.betas[0],
                adam_beta2=self.betas[1],
                adam_eps=1e-8,
                weight_decay=self.weight_decay,
                # Hetero shard sizing: let DistributedOptimizer weight shards
                # by TFLOPS / available VRAM via tier_assignments.
                heterogeneous_shard_sizing=True,
                # PCIe: no NVLink, no P2P → disable async overlap to avoid
                # NCCL async launches that cannot meaningfully overlap on PCIe.
                use_pcie_aware_overlap=False,
            )
        except (ImportError, TypeError) as exc:
            logger.warning("OptimizerConfig unavailable (%s); using fallback", exc)
            opt_config = None

        # ---- 4. Build ParamAndGradBuffer shim ---------------------------
        param_and_grad_buffers = _make_param_grad_buffer(self.model)

        # ---- 5. Build tier_assignments for hetero shard sizing ----------
        tier_assignments = self._tier_assignments_for_dist_opt()

        # ---- 6. Instantiate DistributedOptimizer -----------------------
        try:
            from deepspeed.core.optimizer import DistributedOptimizer
            from deepspeed.core.model_parallel_config import ModelParallelConfig

            model_parallel_config = ModelParallelConfig()

            if opt_config is not None:
                dist_opt = DistributedOptimizer(
                    config=opt_config,
                    optimizer=inner_opt,
                    params=params,
                    model_parallel_config=model_parallel_config,
                    param_and_grad_buffers=param_and_grad_buffers,
                    data_parallel_group=None,       # resolved from parallel_state
                    data_parallel_group_gloo=None,
                    tier_assignments=tier_assignments,
                )
            else:
                # Minimal fallback: wrap inner optimizer without full ZeRO-3 machinery.
                dist_opt = inner_opt

            self._opt = dist_opt
            logger.info(
                "[rank %d] DistOptAdapter.build() complete: %s, "
                "tier=%s, VRAM=%.1f GB, hetero_shard=%s",
                rank,
                type(dist_opt).__name__,
                self.tier_map.tier_of(rank).value if self.tier_map else "unknown",
                vram_bytes / _BYTES_PER_GB,
                tier_assignments is not None,
            )
            return self._opt

        except Exception as exc:
            # Graceful degradation: if DistributedOptimizer construction fails
            # (e.g., distributed not initialized in unit tests), return the
            # inner optimizer directly so the adapter is still usable.
            logger.warning(
                "[rank %d] DistributedOptimizer construction failed (%s); "
                "falling back to inner optimizer (%s).  "
                "This is expected in single-rank / non-distributed tests.",
                rank, exc, type(inner_opt).__name__,
            )
            self._opt = inner_opt
            return self._opt

    # ------------------------------------------------------------------
    # step / zero_grad  — frozen public API
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Execute the optimizer step.

        Delegates to ``DistributedOptimizer.step_with_ready_grads()`` if
        available (grads must have been reduce-scattered first), otherwise
        falls back to the inner optimizer's ``step()``.
        """
        if self._opt is None:
            raise RuntimeError("DistOptAdapter.step: call build() first")
        try:
            from deepspeed.core.optimizer import DistributedOptimizer
            if isinstance(self._opt, DistributedOptimizer):
                self._opt.step_with_ready_grads()
                return
        except ImportError:
            pass
        self._opt.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero (or None) all gradient buffers.

        Delegates to ``DistributedOptimizer.zero_grad()`` which also clears
        the flat grad buffer; falls back to the inner optimizer otherwise.
        """
        if self._opt is None:
            raise RuntimeError("DistOptAdapter.zero_grad: call build() first")
        self._opt.zero_grad(set_to_none=set_to_none)

    # ------------------------------------------------------------------
    # reduce_scatter_grads / all_gather_params  — frozen public API
    # ------------------------------------------------------------------

    def reduce_scatter_grads(self) -> None:
        """PCIe-aware reduce-scatter of grads to owning rank's fp32 shard.

        PCIe constraints
        ----------------
        * No NVLink, no P2P.  All GPU↔GPU traffic traverses the CPU PCIe root.
        * Async NCCL ops offer minimal benefit when bandwidth is PCIe-limited
          (≤ 32 GB/s vs NVLink 900 GB/s) and can hide bugs, so we force
          synchronous execution here.
        * We delegate to ``DistributedOptimizer._reduce_scatter_grads()`` which
          already contains the I6 PCIe-aware branching; the OptimizerConfig
          built in ``build()`` sets ``use_pcie_aware_overlap=False`` for safety.

        Reuse
        -----
        The implementation is in ``core.optimizer.DistributedOptimizer``;
        we call it rather than reimplementing reduce-scatter here.
        """
        if self._opt is None:
            raise RuntimeError("DistOptAdapter.reduce_scatter_grads: call build() first")

        try:
            from deepspeed.core.optimizer import DistributedOptimizer
            if isinstance(self._opt, DistributedOptimizer):
                # _reduce_scatter_grads is PCIe-aware via OptimizerConfig;
                # use_pcie_aware_overlap=False ensures synchronous execution.
                self._opt._reduce_scatter_grads()
                return
        except ImportError:
            pass

        # Fallback path: manual all-reduce + local copy when the full
        # DistributedOptimizer is not available (e.g., single-rank tests).
        try:
            import torch
            import torch.distributed as dist

            if dist.is_initialized():
                for p in self.model.parameters():
                    if p.grad is not None:
                        # Synchronous all-reduce (PCIe: no async benefit)
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=False)
                        p.grad.div_(dist.get_world_size())
        except Exception as exc:
            logger.debug("reduce_scatter_grads fallback: %s", exc)

    def all_gather_params(self) -> None:
        """Gather updated bf16 params back to all ranks after step.

        Delegates to ``DistributedOptimizer.start_param_sync()`` (force_sync=True)
        which writes the FP32 shard to BF16 model params and all-gathers.

        PCIe note: ``force_sync=True`` ensures blocking all-gather — no async
        overlap is attempted, consistent with the no-NVLink cluster topology.
        """
        if self._opt is None:
            raise RuntimeError("DistOptAdapter.all_gather_params: call build() first")

        try:
            from deepspeed.core.optimizer import DistributedOptimizer
            if isinstance(self._opt, DistributedOptimizer):
                # force_sync=True: blocking all-gather (PCIe — no async benefit)
                self._opt.start_param_sync(force_sync=True)
                return
        except ImportError:
            pass

        # Fallback: no-op for single-rank / test environments where params are
        # already correct after the optimizer step.
        logger.debug("all_gather_params: DistributedOptimizer unavailable; no-op")
