# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Distributed optimizer — ZeRO-3 style param/grad/state sharding.

Reimplemented from Megatron-LM/megatron/core/optimizer/distrib_optimizer.py
with DES-LOC extensions for heterogeneous GPU tiers (H100 / A6000).

Design overview
---------------
Each data-parallel rank owns a contiguous *shard* of every grad buffer.
The shard boundaries are computed as equal slices by default, or weighted
by each GPU's BF16 TFLOPS when ``OptimizerConfig.heterogeneous_shard_sizing``
is enabled (DES-LOC extension):

  H100 shard fraction  = H100_tflops / (H100_tflops + A6000_tflops)
  A6000 shard fraction = A6000_tflops / (H100_tflops + A6000_tflops)

On every step:
  1. Reduce-scatter BF16 gradients → each rank gets its averaged FP32 shard.
  2. Optional grad clipping on the local shard.
  3. Adam step on the local FP32 shard.
  4. All-gather updated FP32 shards → reconstruct BF16 model weights.
  5. (DES-LOC) Every Ku steps: all-reduce first moments across DP ranks.
  6. (DES-LOC) Every Kv steps: all-reduce second moments across DP ranks.

Public API
----------
  DistributedOptimizer  — main class
  MegatronOptimizer     — abstract base
  MixedPrecisionOptimizer — BF16/FP16 mixed-precision base
  clip_grad_norm        — model-parallel-aware gradient clipping
  Range                 — (start, end, size) index helper

Import path guaranteed:
  from deepspeed.core.optimizer import DistributedOptimizer  ✓
  from deepspeed.core.optimizer.distrib_optimizer import DistributedOptimizer  ✓
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from deepspeed.core.model_parallel_config import ModelParallelConfig
from deepspeed.core.desloc_config import DesLocConfig, TierType
from deepspeed.core.distributed import ParamAndGradBuffer
from deepspeed.core.optimizer.optimizer_config import OptimizerConfig
import deepspeed.core.parallel_state as parallel_state

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------


def clip_grad_norm(
    parameters: Union[torch.Tensor, List[torch.Tensor]],
    max_norm: float,
    norm_type: float = 2.0,
    model_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
) -> float:
    """Clip gradient norm across model-parallel ranks.

    Args:
        parameters:           Model parameters (or a single tensor).
        max_norm:             Maximum gradient norm after clipping.
        norm_type:            Lp norm type (default L2).
        model_parallel_group: Process group over which to reduce the norm.

    Returns:
        Total gradient norm *before* clipping.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    grads: List[torch.Tensor] = []
    for p in parameters:
        if not isinstance(p, torch.Tensor):
            continue
        grad = getattr(p, "main_grad", None)
        if grad is None:
            grad = p.grad
        if grad is not None:
            grads.append(grad.detach())

    norm_type = float(norm_type)
    device = grads[0].device if grads else torch.device("cpu")

    if norm_type == float("inf"):
        local_norm = max(g.abs().max().item() for g in grads) if grads else 0.0
        total_norm_t = torch.tensor([local_norm], dtype=torch.float32, device=device)
        if model_parallel_group is not None and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                total_norm_t,
                op=torch.distributed.ReduceOp.MAX,
                group=model_parallel_group,
            )
        total_norm = total_norm_t.item()
    else:
        total_norm_t = torch.zeros(1, dtype=torch.float32, device=device)
        for g in grads:
            total_norm_t += g.float().norm(norm_type) ** norm_type
        if model_parallel_group is not None and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                total_norm_t,
                op=torch.distributed.ReduceOp.SUM,
                group=model_parallel_group,
            )
        total_norm = float(total_norm_t.item() ** (1.0 / norm_type))

    clip_coeff = max_norm / (total_norm + 1e-6)
    if clip_coeff < 1.0:
        for g in grads:
            g.mul_(clip_coeff)

    return total_norm


# ---------------------------------------------------------------------------
# Range helper  (mirrors Megatron's Range class)
# ---------------------------------------------------------------------------


class Range:
    """A [start, end) index range over a flat buffer.

    Attributes:
        start (int): First index (inclusive).
        end   (int): Last index (exclusive).
        size  (int): Number of elements = end - start.
    """

    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end
        self.size = end - start

    def normalize(self, start: int = 0) -> "Range":
        """Return a new Range shifted so that self.start maps to *start*."""
        return Range(start, start + self.size)

    def __len__(self) -> int:
        return self.size

    def __str__(self) -> str:
        return f"{self.start},{self.end} [{self.size}]"

    def __repr__(self) -> str:
        return str(self)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class MegatronOptimizer(ABC):
    """Abstract base for all Neuron_SP optimizers.

    Provides the interface contract for gradient preparation, stepping,
    state management, and loss-scale queries.  All concrete optimizers
    must implement ``prepare_grads``, ``step``, and ``zero_grad``.
    """

    def __init__(
        self,
        config: OptimizerConfig,
        optimizer: torch.optim.Optimizer,
        params: List[torch.nn.Parameter],
        model_parallel_config: ModelParallelConfig,
    ) -> None:
        self.config = config
        self.optimizer = optimizer
        self.params = list(params)
        self.model_parallel_config = model_parallel_config
        # Unity loss scale (BF16 path has no dynamic scaler)
        self._scale_one = torch.tensor([1.0], dtype=torch.float32)

    @abstractmethod
    def prepare_grads(self) -> bool:
        """Prepare gradients; return True if optimizer step should be skipped."""
        ...

    @abstractmethod
    def step(self) -> bool:
        """Execute optimizer step; return True if a step was taken."""
        ...

    @abstractmethod
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero (or None) all gradients."""
        ...

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clip_grad_norm(self, clip_grad: float) -> float:
        """Compute grad norm and clip gradients in place."""
        return clip_grad_norm(
            parameters=self.params,
            max_norm=clip_grad,
            norm_type=2.0,
            model_parallel_group=None,
        )

    def get_loss_scale(self) -> torch.Tensor:
        """Current loss scale (1.0 for BF16, dynamic for FP16)."""
        return self._scale_one

    def get_parameters(self) -> List[torch.nn.Parameter]:
        """Flat list of all optimizer parameters."""
        all_params: List[torch.nn.Parameter] = []
        for pg in self.optimizer.param_groups:
            all_params.extend(pg["params"])
        return all_params

    def state_dict(self) -> dict:
        return {"optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def reload_model_params(self) -> None:
        """Reload model params from FP32 master copy (override in subclasses)."""
        pass


# ---------------------------------------------------------------------------
# Mixed precision base
# ---------------------------------------------------------------------------


class MixedPrecisionOptimizer(MegatronOptimizer):
    """Optimizer with FP32 master weights and BF16/FP16 model weights.

    Maintains a shadow copy of every BF16/FP16 model parameter in FP32
    and performs Adam updates on those FP32 copies.  After each step the
    updated FP32 masters are cast back to BF16/FP16 for the forward pass.

    For BF16 training (the common case) ``grad_scaler`` should be ``None``.
    For FP16 training a :class:`~torch.cuda.amp.GradScaler`-compatible
    object must be supplied.
    """

    def __init__(
        self,
        config: OptimizerConfig,
        optimizer: torch.optim.Optimizer,
        params: List[torch.nn.Parameter],
        model_parallel_config: ModelParallelConfig,
        grad_scaler: Optional[Any] = None,
    ) -> None:
        super().__init__(config, optimizer, params, model_parallel_config)
        self.grad_scaler = grad_scaler

        if self.grad_scaler is None:
            assert not config.fp16, "fp16 training requires a grad_scaler."

        if self.grad_scaler is not None:
            self._found_inf = torch.tensor([0.0], dtype=torch.float32)

        # Separate params into three lists:
        #   bf16_model_params   — original BF16 model tensors
        #   fp32_master_params  — FP32 copies held by the optimizer
        #   fp32_model_params   — params that are already FP32
        self.bf16_model_params: List[torch.nn.Parameter] = []
        self.fp32_master_params: List[torch.nn.Parameter] = []
        self.fp32_model_params: List[torch.nn.Parameter] = []

        for param_group in self.optimizer.param_groups:
            bf16_grp: List[torch.nn.Parameter] = []
            fp32_master_grp: List[torch.nn.Parameter] = []
            fp32_model_grp: List[torch.nn.Parameter] = []

            for i, param in enumerate(param_group["params"]):
                if not param.requires_grad:
                    continue
                if param.dtype in (torch.float16, torch.bfloat16):
                    bf16_grp.append(param)
                    master = param.detach().clone().float()
                    master.requires_grad_(True)
                    param_group["params"][i] = master
                    param.main_param = master
                    fp32_master_grp.append(master)
                    if param in self.optimizer.state:
                        self.optimizer.state[master] = self.optimizer.state.pop(param)
                elif param.dtype == torch.float32:
                    fp32_model_grp.append(param)
                else:
                    raise TypeError(
                        f"Unsupported param dtype {param.dtype}. "
                        "Expected float16, bfloat16, or float32."
                    )

            self.bf16_model_params.extend(bf16_grp)
            self.fp32_master_params.extend(fp32_master_grp)
            self.fp32_model_params.extend(fp32_model_grp)

    # ------------------------------------------------------------------
    # Internal grad helpers
    # ------------------------------------------------------------------

    def _copy_model_grads_to_main_grads(self) -> None:
        """Copy BF16 gradients into FP32 master params' ``.grad`` field."""
        for bf16, fp32 in zip(self.bf16_model_params, self.fp32_master_params):
            if hasattr(bf16, "main_grad") and bf16.main_grad is not None:
                fp32.grad = bf16.main_grad.float()
            elif bf16.grad is not None:
                fp32.grad = bf16.grad.float()
            bf16.grad = None
        for fp32 in self.fp32_model_params:
            if hasattr(fp32, "main_grad"):
                fp32.grad = fp32.main_grad

    def _copy_main_params_to_model_params(self) -> None:
        """Write FP32 master weights back into BF16 model tensors."""
        for bf16, fp32 in zip(self.bf16_model_params, self.fp32_master_params):
            bf16.data.copy_(fp32.data)

    def _copy_model_params_to_main_params(self) -> None:
        """Reload FP32 masters from BF16 model params (used at resume)."""
        for bf16, fp32 in zip(self.bf16_model_params, self.fp32_master_params):
            fp32.data.copy_(bf16.data.float())

    def _unscale_and_check_inf(self) -> bool:
        """Unscale main grads and return True if inf/nan was detected."""
        if self.grad_scaler is None:
            return False
        main_grads = [
            fp32.grad.data
            for fp32 in (self.fp32_master_params + self.fp32_model_params)
            if fp32.grad is not None
        ]
        self._found_inf.fill_(0.0)
        if main_grads:
            torch._amp_foreach_non_finite_check_and_unscale_(
                main_grads, self._found_inf, self.grad_scaler.inv_scale
            )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                self._found_inf, op=torch.distributed.ReduceOp.MAX
            )
        return bool(self._found_inf.item() > 0)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prepare_grads(self) -> bool:
        """Copy BF16 grads to FP32 masters; unscale and check for inf.

        Returns:
            True if inf/nan detected (caller should skip optimizer step).
        """
        self._copy_model_grads_to_main_grads()
        if self.grad_scaler is not None:
            found_inf = self._unscale_and_check_inf()
            self.grad_scaler.update(found_inf)
            return found_inf
        return False

    @torch.no_grad()
    def step(self) -> bool:
        found_inf = self.prepare_grads()
        if found_inf:
            return False
        if self.config.clip_grad > 0.0:
            self.clip_grad_norm(self.config.clip_grad)
        self.optimizer.step()
        self._copy_main_params_to_model_params()
        return True

    def zero_grad(self, set_to_none: bool = True) -> None:
        for param in self.bf16_model_params + self.fp32_model_params:
            if set_to_none:
                param.grad = None
                if hasattr(param, "main_grad"):
                    param.main_grad = None
            else:
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()
        for param in self.fp32_master_params:
            if set_to_none:
                param.grad = None
            else:
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()

    def get_loss_scale(self) -> torch.Tensor:
        if self.grad_scaler is None:
            return self._scale_one
        return self.grad_scaler.scale

    def reload_model_params(self) -> None:
        self._copy_main_params_to_model_params()

    def state_dict(self) -> dict:
        sd: dict = {"optimizer": self.optimizer.state_dict()}
        if self.grad_scaler is not None:
            sd["grad_scaler"] = self.grad_scaler.state_dict()
        sd["fp32_master_params"] = [p.data for p in self.fp32_master_params]
        return sd

    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict["optimizer"])
        if "grad_scaler" in state_dict and self.grad_scaler is not None:
            self.grad_scaler.load_state_dict(state_dict["grad_scaler"])
        if "fp32_master_params" in state_dict:
            for cur, saved in zip(
                self.fp32_master_params, state_dict["fp32_master_params"]
            ):
                cur.data.copy_(saved)


# ---------------------------------------------------------------------------
# Heterogeneous shard sizing helpers
# ---------------------------------------------------------------------------


def _compute_hetero_shard_boundaries(
    total_numel: int,
    dp_world_size: int,
    config: OptimizerConfig,
    tier_assignments: Optional[List[Optional[TierType]]] = None,
) -> List[Tuple[int, int]]:
    """Compute per-rank shard [start, end) boundaries.

    When ``config.heterogeneous_shard_sizing`` is False (or
    ``tier_assignments`` is None/empty), every rank gets an equal slice of
    ``ceil(total_numel / dp_world_size)`` elements.

    When heterogeneous sizing is enabled, ranks are divided into two tiers:
      - H100 ranks get ``h100_fraction`` of the buffer each.
      - A6000 ranks share the remainder equally.

    The fractions are derived from BF16 TFLOPS:
      h100_fraction  = H100_tflops  / sum_tflops_per_rank
      a6000_fraction = A6000_tflops / sum_tflops_per_rank

    Each partition is padded to 16-element alignment for NCCL efficiency.

    Args:
        total_numel:       Total number of elements in the flat grad buffer.
        dp_world_size:     Number of data-parallel ranks.
        config:            :class:`OptimizerConfig` instance.
        tier_assignments:  Per-rank tier type; ``None`` = homogeneous.

    Returns:
        List of ``(start, end)`` pairs, one per DP rank.
    """
    ALIGN = 16  # elements; 16 × 2 bytes = 32 bytes aligns to cache line

    if (
        not config.heterogeneous_shard_sizing
        or tier_assignments is None
        or len(tier_assignments) != dp_world_size
    ):
        # Uniform equal-slice partitioning (original behaviour)
        padded = _round_up(total_numel, dp_world_size)
        shard_size = padded // dp_world_size
        boundaries: List[Tuple[int, int]] = []
        for r in range(dp_world_size):
            s = r * shard_size
            e = min(s + shard_size, total_numel)
            boundaries.append((s, e))
        return boundaries

    # -----------------------------------------------------------------------
    # Heterogeneous partitioning: assign sizes proportional to FLOPS.
    # -----------------------------------------------------------------------
    h100_tflops = config.h100_bf16_tflops
    a6000_tflops = config.a6000_bf16_tflops

    # Tflops weight for each rank
    rank_tflops = [
        h100_tflops if t == TierType.DATACENTER else a6000_tflops
        for t in tier_assignments
    ]
    total_tflops = sum(rank_tflops)

    # Raw (unaligned) shard sizes proportional to FLOPS
    raw_sizes = [
        int(total_numel * w / total_tflops) for w in rank_tflops
    ]

    # Align each shard to ALIGN; distribute rounding error to first rank
    aligned_sizes = [_round_up(s, ALIGN) for s in raw_sizes]
    allocated = sum(aligned_sizes)
    if allocated > total_numel:
        # Trim back from the last rank's padding
        aligned_sizes[-1] = max(
            aligned_sizes[-1] - (allocated - total_numel), 0
        )

    # Build [start, end) pairs
    boundaries = []
    cur = 0
    for sz in aligned_sizes:
        end = min(cur + sz, total_numel)
        boundaries.append((cur, end))
        cur = end

    # Ensure the last rank reaches total_numel
    if boundaries and boundaries[-1][1] < total_numel:
        s, _ = boundaries[-1]
        boundaries[-1] = (s, total_numel)

    return boundaries


def _round_up(n: int, multiple: int) -> int:
    """Round *n* up to the next multiple of *multiple*."""
    return ((n + multiple - 1) // multiple) * multiple


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class DistributedOptimizer(MixedPrecisionOptimizer):
    """ZeRO-style distributed optimizer with DES-LOC Ku/Kv decomposed sync.

    Param sharding
    ~~~~~~~~~~~~~~
    Each data-parallel rank is responsible for a contiguous slice of every
    grad buffer.  After a reduce-scatter the rank updates its local FP32
    shard with Adam, then all-gathers the updated values back into every
    rank's BF16 model tensors.

    DES-LOC extension — heterogeneous shard sizing
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    When ``OptimizerConfig.heterogeneous_shard_sizing`` is ``True`` the shard
    size of each rank is proportional to its GPU's BF16 TFLOPS:

      H100  shard fraction ≈ 989 / (989 + 309.7) ≈ 76 %
      A6000 shard fraction ≈ 309.7 / (989 + 309.7) ≈ 24 %

    This ensures the compute time of the local Adam update is approximately
    equal across heterogeneous tiers, eliminating the stragglers that arise
    from equal-sized shards.

    DES-LOC extension — moment synchronisation
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    First  moments (exp_avg)    are all-reduced every ``config.ku`` steps.
    Second moments (exp_avg_sq) are all-reduced every ``config.kv`` steps.
    Parameters are broadcast after every Adam step (not just every kx steps)
    to avoid the "Frankenstein model" divergence.

    Args:
        config:                  Optimizer hyper-parameters and DES-LOC flags.
        optimizer:               Underlying :class:`torch.optim.Optimizer`
                                 (Adam/AdamW); operates on FP32 shards.
        params:                  Full list of model parameters.
        model_parallel_config:   TP/PP/DP configuration + DES-LOC config.
        param_and_grad_buffers:  Contiguous grad buffers from the DDP wrapper;
                                 one per model chunk / buffer dtype.
        data_parallel_group:     NCCL process group for reduce-scatter / all-gather.
        data_parallel_group_gloo: (Optional) CPU Gloo group for checkpoint I/O.
        tier_assignments:        Per-rank :class:`~deepspeed.core.desloc_config.TierType`;
                                 supply when ``heterogeneous_shard_sizing=True``.
    """

    def __init__(
        self,
        config: OptimizerConfig,
        optimizer: torch.optim.Optimizer,
        params: List[torch.nn.Parameter],
        model_parallel_config: ModelParallelConfig,
        param_and_grad_buffers: List[ParamAndGradBuffer],
        data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        data_parallel_group_gloo: Optional[torch.distributed.ProcessGroup] = None,
        tier_assignments: Optional[List[Optional[TierType]]] = None,
    ) -> None:
        # Resolve data_parallel_group from parallel_state when not explicitly given
        if data_parallel_group is None:
            if parallel_state.is_initialized():
                data_parallel_group = parallel_state.get_data_parallel_group()
            else:
                data_parallel_group = torch.distributed.GroupMember.WORLD
        if data_parallel_group_gloo is None and parallel_state.is_initialized():
            data_parallel_group_gloo = parallel_state.get_data_parallel_group_gloo()

        super().__init__(
            config=config,
            optimizer=optimizer,
            params=params,
            model_parallel_config=model_parallel_config,
            grad_scaler=None,  # BF16 path: no dynamic loss scale
        )

        self.param_and_grad_buffers = param_and_grad_buffers
        self.data_parallel_group = data_parallel_group
        self.data_parallel_group_gloo = data_parallel_group_gloo
        self.tier_assignments = tier_assignments

        self.data_parallel_world_size: int = torch.distributed.get_world_size(
            group=data_parallel_group
        )
        self.data_parallel_rank: int = torch.distributed.get_rank(
            group=data_parallel_group
        )

        # DES-LOC legacy config (DesLocConfig on model_parallel_config, may be None)
        self._desloc: Optional[DesLocConfig] = getattr(
            model_parallel_config, "desloc", None
        )

        # Global step counter — incremented after each successful Adam step.
        self._step_count: int = 0

        # Build per-rank param/grad/state shards.
        self._build_shards()

    # ------------------------------------------------------------------
    # Shard construction
    # ------------------------------------------------------------------

    def _build_shards(self) -> None:
        """Partition each grad buffer into heterogeneous per-rank shards.

        For each :class:`ParamAndGradBuffer` we allocate:
        - ``_fp32_shards``     — FP32 master weights for this rank's slice.
        - ``_fp32_grad_shards``— scratch buffer for reduce-scattered grads.

        Shard boundaries are computed via
        :func:`_compute_hetero_shard_boundaries`, which honours the FLOPS-
        weighted heterogeneous sizing if requested.
        """
        dp_rank = self.data_parallel_rank
        dp_world = self.data_parallel_world_size

        self._fp32_shards: List[torch.Tensor] = []
        self._fp32_grad_shards: List[torch.Tensor] = []
        # per-buffer list of per-rank (start, end) boundaries
        self._buf_boundaries: List[List[Tuple[int, int]]] = []

        for buf in self.param_and_grad_buffers:
            total_numel: int = buf.grad_data.numel()

            # Compute per-rank boundaries (hetero-aware)
            boundaries = _compute_hetero_shard_boundaries(
                total_numel=total_numel,
                dp_world_size=dp_world,
                config=self.config,
                tier_assignments=self.tier_assignments,
            )
            self._buf_boundaries.append(boundaries)

            shard_start, shard_end = boundaries[dp_rank]
            shard_size = shard_end - shard_start

            _device = buf.grad_data.device

            # FP32 master weight shard — initialised from model params
            fp32_shard = torch.zeros(shard_size, dtype=torch.float32, device=_device)
            for param, (ps, pe, _) in buf.param_index_map.items():
                overlap_s = max(ps, shard_start)
                overlap_e = min(pe, shard_end)
                if overlap_s >= overlap_e:
                    continue
                local_s = overlap_s - shard_start
                local_e = overlap_e - shard_start
                model_s = overlap_s - ps
                model_e = overlap_e - ps
                fp32_shard[local_s:local_e].copy_(
                    param.data.view(-1)[model_s:model_e].float()
                )

            # FP32 grad scratch (receives reduce-scattered grads)
            fp32_grad_shard = torch.zeros(shard_size, dtype=torch.float32, device=_device)

            self._fp32_shards.append(fp32_shard)
            self._fp32_grad_shards.append(fp32_grad_shard)

        # Rewire the inner optimizer's param_groups to point at FP32 shards.
        shard_params: List[torch.nn.Parameter] = [
            torch.nn.Parameter(shard, requires_grad=True)
            for shard in self._fp32_shards
        ]
        self.optimizer.param_groups = [
            {
                "params": shard_params,
                "lr": self.config.lr or 1e-4,
                "betas": (self.config.adam_beta1, self.config.adam_beta2),
                "eps": self.config.adam_eps,
                "weight_decay": self.config.weight_decay,
            }
        ]
        self._shard_params = shard_params

        logger.debug(
            "DistributedOptimizer shards built: rank=%d/%d, "
            "hetero=%s, buffers=%d",
            self.data_parallel_rank,
            self.data_parallel_world_size,
            self.config.heterogeneous_shard_sizing,
            len(self.param_and_grad_buffers),
        )
        for i, (buf, boundaries) in enumerate(
            zip(self.param_and_grad_buffers, self._buf_boundaries)
        ):
            s, e = boundaries[self.data_parallel_rank]
            logger.debug(
                "  buf[%d]: total=%d, my_shard=[%d,%d) size=%d",
                i, buf.grad_data.numel(), s, e, e - s,
            )

    # ------------------------------------------------------------------
    # Grad preparation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _reduce_scatter_grads(self) -> None:
        """Reduce-scatter BF16 grads; store averaged FP32 result in grad shards.

        Uses ``reduce_scatter_tensor`` for a single contiguous operation
        when the total buffer size is a multiple of dp_world_size.
        Otherwise falls back to an equivalent chunked scatter.
        """
        dp_world = self.data_parallel_world_size
        dp_rank = self.data_parallel_rank

        for buf_idx, (buf, fp32_grad_shard, boundaries) in enumerate(
            zip(
                self.param_and_grad_buffers,
                self._fp32_grad_shards,
                self._buf_boundaries,
            )
        ):
            total_numel = buf.grad_data.numel()
            shard_start, shard_end = boundaries[dp_rank]
            shard_size = shard_end - shard_start

            # Maximum shard size across ranks (needed for equal-shard collectives)
            max_shard_size = max(e - s for s, e in boundaries)

            if all((e - s) == max_shard_size for s, e in boundaries):
                # -------------------------------------------------------
                # Fast path: equal shards → use reduce_scatter_tensor
                # -------------------------------------------------------
                padded_numel = max_shard_size * dp_world
                if padded_numel > total_numel:
                    grad_padded = torch.zeros(
                        padded_numel,
                        dtype=buf.grad_data.dtype,
                        device=buf.grad_data.device,
                    )
                    grad_padded[:total_numel].copy_(buf.grad_data)
                else:
                    grad_padded = buf.grad_data

                output_shard = torch.zeros(
                    max_shard_size,
                    dtype=grad_padded.dtype,
                    device=grad_padded.device,
                )
                torch.distributed.reduce_scatter_tensor(
                    output_shard,
                    grad_padded,
                    op=torch.distributed.ReduceOp.SUM,
                    group=self.data_parallel_group,
                )
                fp32_grad_shard.copy_(output_shard[:shard_size].float().div_(dp_world))

            else:
                # -------------------------------------------------------
                # Slow path: heterogeneous shards → chunked reduce + local copy
                # Uses all_reduce on the full buffer then extracts our slice.
                # This is bandwidth-equivalent to reduce_scatter + re-broadcast
                # for small worlds; for large worlds a proper variable-chunk
                # reduce_scatter would be more efficient.
                # -------------------------------------------------------
                grad_work = buf.grad_data.clone()
                torch.distributed.all_reduce(
                    grad_work,
                    op=torch.distributed.ReduceOp.SUM,
                    group=self.data_parallel_group,
                )
                fp32_grad_shard.copy_(
                    grad_work[shard_start:shard_end].float().div_(dp_world)
                )

        # Attach grad shards to shard params for the Adam step
        for shard_param, fp32_grad_shard in zip(
            self._shard_params, self._fp32_grad_shards
        ):
            shard_param.grad = fp32_grad_shard

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prepare_grads(self) -> bool:
        """Reduce-scatter gradients into FP32 shards.

        Returns:
            Always ``False`` (no loss scaler in BF16 path).
        """
        self._reduce_scatter_grads()
        return False

    # ------------------------------------------------------------------
    # DES-LOC moment synchronisation
    # ------------------------------------------------------------------

    def sync_moments(
        self, sync_first: bool = False, sync_second: bool = False
    ) -> None:
        """All-reduce optimizer moments across data-parallel ranks.

        This implements the DES-LOC Ku / Kv synchronisation protocol:
        first moments are averaged every Ku steps, second moments every Kv
        steps.  The communication volume is reduced by (1 - 1/Ku) + (1 - 1/Kv)
        compared to syncing every step.

        Args:
            sync_first:  All-reduce ``exp_avg``    (first moment).
            sync_second: All-reduce ``exp_avg_sq`` (second moment).
        """
        if not (sync_first or sync_second):
            return

        dp_world = self.data_parallel_world_size

        for shard_param in self._shard_params:
            state = self.optimizer.state.get(shard_param)
            if state is None:
                continue

            if sync_first and "exp_avg" in state:
                m = state["exp_avg"]
                torch.distributed.all_reduce(
                    m,
                    op=torch.distributed.ReduceOp.SUM,
                    group=self.data_parallel_group,
                )
                m.div_(dp_world)

            if sync_second and "exp_avg_sq" in state:
                v = state["exp_avg_sq"]
                torch.distributed.all_reduce(
                    v,
                    op=torch.distributed.ReduceOp.SUM,
                    group=self.data_parallel_group,
                )
                v.div_(dp_world)

    # ------------------------------------------------------------------
    # Shard → model broadcast
    # ------------------------------------------------------------------

    @torch.no_grad()
    def shard_to_model_broadcast(self) -> None:
        """Write local FP32 shard to BF16 model and broadcast to all ranks.

        Called every step to keep all ranks' model weights in sync.
        Avoids the "Frankenstein model" bug where only 1/N of params get
        updated on each rank.

        Protocol (per buffer):
          1. Convert this rank's FP32 shard to BF16 and zero-fill the rest
             of a padded world-size buffer.
          2. All-reduce (SUM) — only one rank has non-zero in each slot.
          3. Copy reconstructed BF16 values back to model param tensors.
          4. Refresh the local FP32 shard from the broadcast result.

        For heterogeneous shards the buffer is assembled with each rank's
        slice at the correct offset before the all-reduce.
        """
        dp_rank = self.data_parallel_rank

        for fp32_shard, boundaries, buf in zip(
            self._fp32_shards,
            self._buf_boundaries,
            self.param_and_grad_buffers,
        ):
            total_numel = buf.grad_data.numel()
            _device = buf.grad_data.device

            full_bf16 = torch.zeros(total_numel, dtype=torch.bfloat16, device=_device)

            shard_start, shard_end = boundaries[dp_rank]
            copy_len = shard_end - shard_start
            full_bf16[shard_start:shard_end].copy_(fp32_shard[:copy_len].bfloat16())

            # SUM all-reduce: each rank contributes to its own region only
            torch.distributed.all_reduce(
                full_bf16,
                op=torch.distributed.ReduceOp.SUM,
                group=self.data_parallel_group,
            )

            # Write back to model param tensors
            for param, (ps, pe, _) in buf.param_index_map.items():
                param.data.copy_(full_bf16[ps:pe].view(param.data.shape))

            # Keep FP32 shard consistent with the broadcast result
            fp32_shard[:copy_len].copy_(full_bf16[shard_start:shard_end].float())

    # ------------------------------------------------------------------
    # Zero grad
    # ------------------------------------------------------------------

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradient buffers and shard param grads."""
        for buf in self.param_and_grad_buffers:
            buf.grad_data.zero_()
        for sp in self._shard_params:
            if set_to_none:
                sp.grad = None
            elif sp.grad is not None:
                sp.grad.detach_()
                sp.grad.zero_()

    # ------------------------------------------------------------------
    # State dict / checkpoint helpers
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return serialisable optimizer state.

        Includes the inner Adam state (moments + step), the FP32 shard
        tensors for warm-start, and the current step count for DES-LOC
        period tracking.
        """
        return {
            "optimizer": self.optimizer.state_dict(),
            "step_count": self._step_count,
            "fp32_shards": [s.cpu() for s in self._fp32_shards],
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore optimizer state from a checkpoint dict."""
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self._step_count = state_dict.get("step_count", 0)
        if "fp32_shards" in state_dict:
            for shard, saved in zip(self._fp32_shards, state_dict["fp32_shards"]):
                shard.copy_(saved.to(shard.device))

    # ------------------------------------------------------------------
    # _copy_main_params_to_model_params  (Megatron evolution: M2307→M4019)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _copy_main_params_to_model_params(self) -> None:
        """Copy updated FP32 shard back to BF16 model param tensors.

        This mirrors Megatron's ``_copy_main_params_to_model_params`` but uses
        our flat ``_fp32_shards`` / ``param_and_grad_buffers`` data structures
        instead of Megatron's ``shard_fp32_from_float16_groups``.

        For each buffer, every model parameter whose range overlaps with this
        rank's shard receives updated data from the corresponding slice of the
        FP32 shard.  This must be called **before** the all-gather so the
        correct values are broadcast to all DP ranks.

        Evolution notes
        ~~~~~~~~~~~~~~~
        - M2307: initial port of copy_group_params inner helper.
        - M3715: decoupled_grad / MFSDP awareness.
        - M3737: NVFP4 native weight guard (skip quantized params here; they
          are handled by a dedicated ``quantize_nvfp4_param_shard`` call).
        - M4019: MXFP8 param-buffer copy path; in our simplified stack we
          keep the standard BF16 path only and skip FP8/NVFP4 guards because
          those quantized param types require TE which is not in scope.
        """
        dp_rank = self.data_parallel_rank

        for fp32_shard, boundaries, buf in zip(
            self._fp32_shards,
            self._buf_boundaries,
            self.param_and_grad_buffers,
        ):
            shard_start, shard_end = boundaries[dp_rank]

            for param, (ps, pe, _) in buf.param_index_map.items():
                overlap_s = max(ps, shard_start)
                overlap_e = min(pe, shard_end)
                if overlap_s >= overlap_e:
                    # This parameter is not owned by this rank's shard.
                    continue

                # Slice inside the local shard
                local_s = overlap_s - shard_start
                local_e = overlap_e - shard_start
                # Slice inside the full parameter tensor
                param_s = overlap_s - ps
                param_e = overlap_e - ps

                # Write FP32 shard slice back to BF16 model param.
                param.data.view(-1)[param_s:param_e].copy_(
                    fp32_shard[local_s:local_e].to(param.data.dtype)
                )

    # ------------------------------------------------------------------
    # start_param_sync / start_param_sync_for_bucket_group_subset
    # (Megatron evolution: M3998 — Route non-Muon params #4771)
    # ------------------------------------------------------------------

    def start_param_sync(self, force_sync: bool = True) -> None:
        """All-gather updated FP32 shards into every rank's BF16 model weights.

        This is the canonical post-step param broadcast.  It replaces the
        earlier ``shard_to_model_broadcast`` and aligns with Megatron's
        ``model_chunk.start_param_sync()`` contract.

        First copies the local FP32 shard slice into BF16 model params (via
        ``_copy_main_params_to_model_params``), then performs the all-gather
        so every rank sees the full, up-to-date weight tensor.

        When ``force_sync=False`` the call is a *no-op* if async overlap is
        requested by the config (future extension hook; currently always
        synchronous).

        Args:
            force_sync: If True (default) perform a blocking all-gather.
                        If False, skip when async param-gather overlap is on.
        """
        if not force_sync and getattr(self.config, "overlap_param_gather", False):
            # Async overlap path: the caller will trigger the gather later.
            return

        # 1. Write local FP32 shard to model params on this rank.
        self._copy_main_params_to_model_params()

        # 2. All-gather so every rank has all updated params.
        self.shard_to_model_broadcast()

    def start_param_sync_for_bucket_group_subset(self) -> None:
        """Trigger ``start_param_sync`` for DistributedOptimizer-managed params only.

        In a system where a ``LayerWiseDistributedOptimizer`` sibling manages
        some buckets, we must avoid double-syncing those buckets.  In our
        single-optimizer stack all params belong to this optimizer, so this
        method is equivalent to ``start_param_sync``.

        This method exists to match the Megatron API surface introduced in
        M3998 (commit 9e60da33) so that callers written against the Megatron
        interface continue to work when ported to Neuron_SP.

        Evolution note (M3998)
        ~~~~~~~~~~~~~~~~~~~~~~
        Megatron walks ``model_chunk.bucket_groups`` and calls
        ``_start_bucket_group_param_sync`` per bucket, skipping buckets tagged
        ``is_managed_by_layer_wise_optimizer=True``.  Our flat-buffer design
        has no per-bucket lazy tagging, so we delegate to ``start_param_sync``
        directly.
        """
        self.start_param_sync(force_sync=True)

    # ------------------------------------------------------------------
    # step_with_ready_grads  (Megatron evolution: M2307 → M3998)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        """Optimizer step assuming gradients are already reduced/scattered.

        This is the Megatron-style entry point for the optimizer step when the
        DDP wrapper has already performed the reduce-scatter and the grad shards
        are ready in ``self._fp32_grad_shards``.  Callers that manage DDP
        overlap externally use this instead of the all-in-one ``step()``.

        Sequence
        ~~~~~~~~
        1.  Clip gradients on the local FP32 shard (if configured).
        2.  Local Adam update on the FP32 shard parameters.
        3.  Increment step counter.
        4.  DES-LOC moment synchronisation (Ku / Kv cadence).
        5.  All-gather updated FP32 shards → BF16 model weights, using
            ``start_param_sync_for_bucket_group_subset()`` (aligns with
            Megatron M3998: only sync DistOpt-managed buckets so a sibling
            LayerWise optimizer doesn't double-sync).

        Returns:
            True if the step was taken successfully; False if skipped.

        Evolution notes
        ~~~~~~~~~~~~~~~
        - M2307: ``step_with_ready_grads`` split off from ``step``.
        - M3998: switched inner sync call from
          ``model_chunk.start_param_sync()`` to
          ``start_param_sync_for_bucket_group_subset()`` to avoid duplicate
          bucket syncs when a LayerWise sibling is present.
        - M3811: ``_defer_param_sync`` flag added for FSDP overlap.
        """
        # Gradient clipping on local shard params.
        if self.config.clip_grad > 0.0:
            clip_grad_norm(
                parameters=self._shard_params,
                max_norm=self.config.clip_grad,
                norm_type=2.0,
                model_parallel_group=self.data_parallel_group,
            )

        # Local Adam update on FP32 shard.
        self.optimizer.step()

        self._step_count += 1

        # DES-LOC: synchronise moments at Ku / Kv cadence.
        if self.config.desloc_enabled:
            sync_u = self.config.is_ku_step(self._step_count)
            sync_v = self.config.is_kv_step(self._step_count)
            if sync_u or sync_v:
                self.sync_moments(sync_first=sync_u, sync_second=sync_v)
        elif self._desloc is not None and self._desloc.enabled:
            sync_u = self._desloc.is_ku_step(self._step_count)
            sync_v = self._desloc.is_kv_step(self._step_count)
            if sync_u or sync_v:
                self.sync_moments(sync_first=sync_u, sync_second=sync_v)

        # All-gather updated shards to model weights (only DistOpt-managed).
        # When _defer_param_sync is set (FSDP overlap path), skip here; the
        # caller is responsible for triggering the gather at the right moment.
        if not getattr(self, "_defer_param_sync", False):
            self.start_param_sync_for_bucket_group_subset()

        return True

    @torch.no_grad()
    def step(self) -> bool:
        """Full distributed optimizer step.

        Sequence:
          1. Reduce-scatter BF16 grads → FP32 grad shards.
          2. Execute ``step_with_ready_grads`` (clip + Adam + moment sync + all-gather).

        Returns:
            True if a step was taken, False if skipped due to overflow.
        """
        found_inf = self.prepare_grads()
        if found_inf:
            return False
        return self.step_with_ready_grads()

    # ------------------------------------------------------------------
    # save_parameter_state / load_parameter_state
    # (Megatron evolution: M2307 → M3385 → M3356 → M3175)
    # ------------------------------------------------------------------

    def get_parameter_state_dp_zero(
        self,
        use_gloo_comm: bool = True,
        return_on_all_ranks: bool = False,
    ) -> Optional[dict]:
        """Gather per-rank FP32 shards and optimizer moments onto DP rank 0.

        Mirrors Megatron's ``get_parameter_state_dp_zero``.  For each
        ``ParamAndGradBuffer`` we collect ``(param, exp_avg, exp_avg_sq)`` from
        every DP rank and assemble them into world-sized CPU tensors on rank 0.

        Uses Gloo by default (``use_gloo_comm=True``) to avoid occupying the
        NCCL stream during checkpointing.

        Evolution notes
        ~~~~~~~~~~~~~~~
        - M2307: initial implementation with buckets_coalesced format.
        - M3175: ``step`` key in optimizer state handled separately to avoid
          saving a scalar as a tensor (fixes CPU-offload checkpoint crash).
        - M3356: skip non-tensor optimizer state entries (fixes crash when the
          inner optimizer stores Python scalars like ``step`` as plain ints).
        - M3385: ``--load-main-params-from-ckpt`` flag support — pass through
          here via the ``state_dict`` checkpoint structure.

        Args:
            use_gloo_comm:       Use Gloo process group for CPU gather.
            return_on_all_ranks: Return the full state on every rank (for
                                 parallel saving); default False returns None
                                 on non-zero DP ranks.

        Returns:
            State dict on DP rank 0 (or all ranks if ``return_on_all_ranks``),
            None on other ranks.
        """
        # Choose comm group: prefer Gloo CPU transport for checkpoint I/O
        # (does not stall the CUDA NCCL stream).
        if use_gloo_comm and self.data_parallel_group_gloo is not None:
            comm_group = self.data_parallel_group_gloo
        else:
            comm_group = self.data_parallel_group

        dp_world = torch.distributed.get_world_size(group=comm_group)
        dp_rank = torch.distributed.get_rank(group=comm_group)
        global_ranks = torch.distributed.get_process_group_ranks(comm_group)

        state: dict = {"buckets_coalesced": True}

        for buf_idx, (fp32_shard, buf, boundaries) in enumerate(
            zip(self._fp32_shards, self.param_and_grad_buffers, self._buf_boundaries)
        ):
            shard_start, shard_end = boundaries[dp_rank]
            shard_size = shard_end - shard_start
            total_numel = buf.grad_data.numel()

            # --- Build local shard buffers (param + moments) ---
            local_param = fp32_shard.detach().cpu()

            # Retrieve exp_avg / exp_avg_sq from the inner Adam state.
            # The shard param corresponding to this buffer's index.
            shard_p = self._shard_params[buf_idx]
            adam_state = self.optimizer.state.get(shard_p, {})

            local_exp_avg = adam_state.get("exp_avg", torch.zeros(shard_size))
            local_exp_avg_sq = adam_state.get("exp_avg_sq", torch.zeros(shard_size))

            # M3356: skip non-tensor entries silently (e.g. scalar ``step``).
            if not isinstance(local_exp_avg, torch.Tensor):
                local_exp_avg = torch.zeros(shard_size, dtype=torch.float32)
            if not isinstance(local_exp_avg_sq, torch.Tensor):
                local_exp_avg_sq = torch.zeros(shard_size, dtype=torch.float32)

            local_exp_avg = local_exp_avg.detach().cpu().float()
            local_exp_avg_sq = local_exp_avg_sq.detach().cpu().float()

            # --- Gather all ranks' shards onto rank 0 ---
            # Each rank's shard may differ in size (hetero path), so we use
            # all_gather_object which handles variable-size tensors via CPU.
            local_tensors = {
                "param": local_param,
                "exp_avg": local_exp_avg,
                "exp_avg_sq": local_exp_avg_sq,
                "shard_start": shard_start,
                "shard_end": shard_end,
                "total_numel": total_numel,
            }

            if return_on_all_ranks:
                gathered: List[Optional[dict]] = [None] * dp_world
                torch.distributed.all_gather_object(gathered, local_tensors, group=comm_group)
            else:
                if dp_rank == 0:
                    gathered = [None] * dp_world
                else:
                    gathered = None
                torch.distributed.gather_object(
                    local_tensors,
                    gathered,
                    dst=global_ranks[0],
                    group=comm_group,
                )

            if dp_rank == 0 or return_on_all_ranks:
                # Assemble world tensors from all ranks' shards.
                world_param = torch.zeros(total_numel, dtype=torch.float32)
                world_exp_avg = torch.zeros(total_numel, dtype=torch.float32)
                world_exp_avg_sq = torch.zeros(total_numel, dtype=torch.float32)

                for rank_data in gathered:
                    if rank_data is None:
                        continue
                    rs, re = rank_data["shard_start"], rank_data["shard_end"]
                    copy_len = re - rs
                    world_param[rs:re].copy_(rank_data["param"][:copy_len])
                    world_exp_avg[rs:re].copy_(rank_data["exp_avg"][:copy_len])
                    world_exp_avg_sq[rs:re].copy_(rank_data["exp_avg_sq"][:copy_len])

                dtype_state = {
                    torch.bfloat16: {
                        "param": world_param,
                        "exp_avg": world_exp_avg,
                        "exp_avg_sq": world_exp_avg_sq,
                        "numel_unpadded": total_numel,
                    }
                }
                state[buf_idx] = dtype_state
            else:
                state[buf_idx] = None

        return state if (dp_rank == 0 or return_on_all_ranks) else None

    def save_parameter_state(self, filename: str) -> None:
        """Gather all FP32 shards + Adam moments onto DP rank 0 and save.

        Only DP rank 0 writes to *filename*.  All ranks participate in the
        gather via ``get_parameter_state_dp_zero``.

        Uses Gloo comm group for CPU gather (does not stall NCCL stream).

        Evolution notes
        ~~~~~~~~~~~~~~~
        - M2307: initial implementation (NCCL-based gather).
        - M3175: switched to ``get_parameter_state_dp_zero`` helper which also
          saves optimizer moments alongside main params.
        - M3385: added ``step_count`` to the checkpoint payload for DES-LOC
          period tracking.

        Args:
            filename: Path on the local filesystem where the checkpoint is
                      written (rank 0 only).
        """
        state_dict = self.get_parameter_state_dp_zero(use_gloo_comm=True)
        if self.data_parallel_rank == 0 and state_dict is not None:
            state_dict["step_count"] = self._step_count
            torch.save(state_dict, filename)
            logger.info(
                "DistributedOptimizer: saved parameter state to %s "
                "(rank 0, dp_world=%d)",
                filename,
                self.data_parallel_world_size,
            )

    def load_parameter_state_from_dp_zero(
        self, state_dict: Optional[dict], *, update_legacy_format: bool = False
    ) -> None:
        """Scatter world-sized checkpoint tensors from DP rank 0 to all ranks.

        Mirrors Megatron's ``load_parameter_state_from_dp_zero``.  DP rank 0
        holds the full ``(param, exp_avg, exp_avg_sq)`` tensors; we scatter the
        appropriate slice to every rank.

        Evolution notes
        ~~~~~~~~~~~~~~~
        - M2307: initial implementation.
        - M3175: ``step`` key handled separately (scalar, not a tensor).
        - M3356: non-tensor optimizer state entries are skipped gracefully to
          handle checkpoints written by Adam variants that store ``step`` as
          a Python int rather than a 0-D tensor.
        - M3385: ``--load-main-params-from-ckpt`` path: after scattering the
          world tensors the FP32 shard is re-broadcast from the param slice
          so the model weights are consistent without requiring a forward pass.

        Args:
            state_dict:            Checkpoint dict (only non-None on DP rank 0).
            update_legacy_format:  When True, call legacy loader for
                                   pre-Feb-2024 checkpoint format.
        """
        if update_legacy_format:
            # Legacy format: state_dict is just a list of gathered shards.
            return self._load_parameter_state_legacy(state_dict)

        # Choose Gloo comm group for scatter (same as save path).
        if self.data_parallel_group_gloo is not None:
            comm_group = self.data_parallel_group_gloo
        else:
            comm_group = self.data_parallel_group

        dp_world = torch.distributed.get_world_size(group=comm_group)
        dp_rank = torch.distributed.get_rank(group=comm_group)
        global_ranks = torch.distributed.get_process_group_ranks(comm_group)

        for buf_idx, (fp32_shard, boundaries) in enumerate(
            zip(self._fp32_shards, self._buf_boundaries)
        ):
            shard_start, shard_end = boundaries[dp_rank]
            shard_size = shard_end - shard_start

            # DP rank 0 prepares per-rank slice dicts for scatter.
            if dp_rank == 0:
                if state_dict is None or state_dict.get(buf_idx) is None:
                    logger.warning(
                        "load_parameter_state_from_dp_zero: missing buf_idx=%d in state_dict",
                        buf_idx,
                    )
                    send_objects = [{} for _ in range(dp_world)]
                else:
                    buf_state = list(state_dict[buf_idx].values())[0]  # first (only) dtype
                    world_param = buf_state["param"]
                    world_exp_avg = buf_state["exp_avg"]
                    world_exp_avg_sq = buf_state["exp_avg_sq"]

                    send_objects: List[dict] = []
                    for r in range(dp_world):
                        rs, re = boundaries[r]
                        send_objects.append(
                            {
                                "param": world_param[rs:re].clone(),
                                "exp_avg": world_exp_avg[rs:re].clone(),
                                "exp_avg_sq": world_exp_avg_sq[rs:re].clone(),
                            }
                        )
            else:
                send_objects = None

            recv_obj: dict = {}
            torch.distributed.scatter_object_list(
                [recv_obj],
                send_objects,
                src=global_ranks[0],
                group=comm_group,
            )

            # Restore FP32 shard from checkpoint.
            if "param" in recv_obj:
                copy_len = min(shard_size, recv_obj["param"].numel())
                fp32_shard[:copy_len].copy_(recv_obj["param"][:copy_len].to(fp32_shard.device))

            # Restore Adam moments for this buffer's shard param.
            shard_p = self._shard_params[buf_idx]
            if shard_p not in self.optimizer.state:
                self.optimizer.state[shard_p] = {}
            opt_state = self.optimizer.state[shard_p]

            # M3356: guard — only update tensor entries.
            for key in ("exp_avg", "exp_avg_sq"):
                if key in recv_obj and isinstance(recv_obj[key], torch.Tensor):
                    tensor = recv_obj[key].to(fp32_shard.device)
                    if key in opt_state and isinstance(opt_state[key], torch.Tensor):
                        opt_state[key].copy_(tensor)
                    else:
                        opt_state[key] = tensor.clone()

    def _load_parameter_state_legacy(self, state_dict: Optional[dict]) -> None:
        """Load from the pre-M2307 checkpoint format (plain gathered-shard list).

        The legacy format stores a list of per-buffer gathered FP32 tensors
        (no moments) under the key ``"fp32_shards"``.  We split them back into
        per-rank slices using the current shard boundaries.

        Args:
            state_dict: Dict with ``"fp32_shards"`` key (only on DP rank 0).
        """
        dp_rank = self.data_parallel_rank
        dp_world = self.data_parallel_world_size

        comm_group = self.data_parallel_group

        for buf_idx, (fp32_shard, boundaries) in enumerate(
            zip(self._fp32_shards, self._buf_boundaries)
        ):
            shard_start, shard_end = boundaries[dp_rank]

            if dp_rank == 0 and state_dict is not None:
                gathered_list = state_dict.get("fp32_shards", [])
                if buf_idx < len(gathered_list) and gathered_list[buf_idx] is not None:
                    world_tensor = gathered_list[buf_idx]
                    send_objects = []
                    for r in range(dp_world):
                        rs, re = boundaries[r]
                        send_objects.append({"param": world_tensor[rs:re].clone()})
                else:
                    send_objects = [{} for _ in range(dp_world)]
            else:
                send_objects = None

            recv_obj: dict = {}
            torch.distributed.scatter_object_list(
                [recv_obj],
                send_objects,
                src=torch.distributed.get_global_rank(comm_group, 0),
                group=comm_group,
            )

            if "param" in recv_obj:
                shard_size = shard_end - shard_start
                copy_len = min(shard_size, recv_obj["param"].numel())
                fp32_shard[:copy_len].copy_(recv_obj["param"][:copy_len].to(fp32_shard.device))

    def load_parameter_state(
        self, filename: str, *, update_legacy_format: bool = False
    ) -> None:
        """Load distributed parameter state from *filename*.

        DP rank 0 reads the checkpoint; all ranks receive their slice via
        ``load_parameter_state_from_dp_zero``.

        Evolution notes
        ~~~~~~~~~~~~~~~
        - M2307: initial implementation.
        - M3385: added ``step_count`` restore for DES-LOC period tracking.

        Args:
            filename:             Path to the checkpoint saved by
                                  ``save_parameter_state``.
            update_legacy_format: Pass through to
                                  ``load_parameter_state_from_dp_zero`` for
                                  pre-Feb-2024 checkpoints.
        """
        state_dict: Optional[dict] = None
        if self.data_parallel_rank == 0:
            state_dict = torch.load(filename, map_location="cpu")
            logger.info(
                "DistributedOptimizer: loaded parameter state from %s", filename
            )

        # Restore step counter for DES-LOC period alignment.
        if self.data_parallel_rank == 0 and state_dict is not None:
            self._step_count = state_dict.get("step_count", 0)

        # Broadcast step_count to all ranks.
        step_count_t = torch.tensor(
            [self._step_count], dtype=torch.int64,
            device=next(iter(self._fp32_shards)).device
            if self._fp32_shards
            else torch.device("cpu"),
        )
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(
                step_count_t,
                src=torch.distributed.get_global_rank(self.data_parallel_group, 0),
                group=self.data_parallel_group,
            )
        self._step_count = int(step_count_t.item())

        self.load_parameter_state_from_dp_zero(
            state_dict, update_legacy_format=update_legacy_format
        )

    def get_model_param_range_map(
        self, param: torch.nn.Parameter
    ) -> Optional[Dict[str, Range]]:
        """Return the grad-buffer range info for *param* on this rank.

        Returns a dict with key ``"gbuf_local"`` containing the Range of
        elements this rank owns for the parameter, or ``None`` if the param
        does not overlap with this rank's shard.
        """
        dp_rank = self.data_parallel_rank

        for buf, boundaries in zip(
            self.param_and_grad_buffers, self._buf_boundaries
        ):
            if param not in buf.param_index_map:
                continue
            ps, pe, _ = buf.param_index_map[param]
            shard_start, shard_end = boundaries[dp_rank]
            overlap_s = max(ps, shard_start)
            overlap_e = min(pe, shard_end)
            if overlap_s >= overlap_e:
                return None
            local_s = overlap_s - shard_start
            local_e = overlap_e - shard_start
            return {
                "gbuf_local": Range(local_s, local_e),
                "gbuf_world": Range(overlap_s, overlap_e),
                "param": Range(overlap_s - ps, overlap_e - ps),
            }
        return None
