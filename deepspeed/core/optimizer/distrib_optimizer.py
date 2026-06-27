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

    @torch.no_grad()
    def step(self) -> bool:
        """Full distributed optimizer step.

        Sequence:
          1. Reduce-scatter BF16 grads → FP32 grad shards.
          2. Clip gradients on local shard (if configured).
          3. Local Adam update on FP32 shard.
          4. Increment step counter.
          5. DES-LOC moment sync (Ku / Kv steps).
          6. All-gather FP32 shards → BF16 model + broadcast.

        Returns:
            True if a step was taken, False if skipped due to overflow.
        """
        found_inf = self.prepare_grads()
        if found_inf:
            return False

        # Gradient clipping on shard params
        if self.config.clip_grad > 0.0:
            clip_grad_norm(
                parameters=self._shard_params,
                max_norm=self.config.clip_grad,
                norm_type=2.0,
                model_parallel_group=self.data_parallel_group,
            )

        # Local Adam update
        self.optimizer.step()

        self._step_count += 1

        # DES-LOC: synchronise moments when appropriate
        if self.config.desloc_enabled:
            sync_u = self.config.is_ku_step(self._step_count)
            sync_v = self.config.is_kv_step(self._step_count)
            if sync_u or sync_v:
                self.sync_moments(sync_first=sync_u, sync_second=sync_v)
        elif self._desloc is not None and self._desloc.enabled:
            # Legacy DesLocConfig path (from model_parallel_config.desloc)
            sync_u = self._desloc.is_ku_step(self._step_count)
            sync_v = self._desloc.is_kv_step(self._step_count)
            if sync_u or sync_v:
                self.sync_moments(sync_first=sync_u, sync_second=sync_v)

        # Reconstruct BF16 model weights from updated FP32 shards
        self.shard_to_model_broadcast()

        return True

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

    def save_parameter_state(self, filename: str) -> None:
        """Gather all FP32 shards onto DP rank 0 and save to *filename*.

        Only DP rank 0 writes; all ranks participate in the gather.
        """
        dp_rank = self.data_parallel_rank
        dp_world = self.data_parallel_world_size

        gathered_shards: List[Optional[torch.Tensor]] = []
        for fp32_shard in self._fp32_shards:
            shard_size = fp32_shard.numel()
            if dp_rank == 0:
                all_shards = [
                    torch.zeros(shard_size, dtype=torch.float32, device=fp32_shard.device)
                    for _ in range(dp_world)
                ]
            else:
                all_shards = None

            torch.distributed.gather(
                fp32_shard,
                all_shards,
                dst=torch.distributed.get_global_rank(self.data_parallel_group, 0),
                group=self.data_parallel_group,
            )

            if dp_rank == 0 and all_shards is not None:
                gathered_shards.append(torch.cat(all_shards, dim=0))
            else:
                gathered_shards.append(None)

        if dp_rank == 0:
            torch.save(
                {"fp32_shards": gathered_shards, "step_count": self._step_count},
                filename,
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
