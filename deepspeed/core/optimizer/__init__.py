# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Distributed optimizer with DES-LOC Ku/Kv decomposed moment sync."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from deepspeed.core.model_parallel_config import ModelParallelConfig
from deepspeed.core.desloc_config import DesLocConfig
from deepspeed.core.distributed import ParamAndGradBuffer

logger = logging.getLogger(__name__)


# ===========================================================================
# optimizer_config.py
# ===========================================================================

@dataclass
class OptimizerConfig:
    """Optimizer configuration.

    Mirrors Megatron's OptimizerConfig with DES-LOC moment sync extensions.
    """
    lr: Optional[float] = None
    min_lr: Optional[float] = None
    weight_decay: float = 0.01
    params_dtype: torch.dtype = torch.float32
    loss_scale: Optional[float] = None
    initial_loss_scale: float = 2**32
    min_loss_scale: float = 1.0
    loss_scale_window: int = 1000
    hysteresis: int = 2
    fp16: bool = False
    bf16: bool = True
    clip_grad: float = 1.0
    use_distributed_optimizer: bool = True
    overlap_param_gather: bool = False
    overlap_grad_reduce: bool = False

    # Adam hyperparams
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8


# ===========================================================================
# clip_grads.py
# ===========================================================================

def clip_grad_norm(
    parameters: Union[torch.Tensor, List[torch.Tensor]],
    max_norm: float,
    norm_type: float = 2.0,
    model_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
) -> float:
    """Clip gradient norm across model-parallel ranks.

    Args:
        parameters: Model parameters.
        max_norm: Max gradient norm.
        norm_type: Norm type (default L2).
        model_parallel_group: Process group for norm all-reduce.

    Returns:
        Total gradient norm before clipping.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # Collect grads (prefer .main_grad over .grad for contiguous-buffer case)
    grads: List[torch.Tensor] = []
    for p in parameters:
        if not isinstance(p, torch.Tensor):
            continue
        grad = getattr(p, 'main_grad', None)
        if grad is None:
            grad = p.grad
        if grad is not None:
            grads.append(grad.detach())

    norm_type = float(norm_type)
    device = grads[0].device if grads else torch.device('cpu')

    if norm_type == float('inf'):
        # Inf-norm: max of absolute values
        if grads:
            local_norm = max(g.abs().max().item() for g in grads)
        else:
            local_norm = 0.0
        total_norm_tensor = torch.tensor([local_norm], dtype=torch.float32, device=device)
        if model_parallel_group is not None and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                total_norm_tensor,
                op=torch.distributed.ReduceOp.MAX,
                group=model_parallel_group,
            )
        total_norm = total_norm_tensor.item()
    else:
        # p-norm: sum of (|g|^p), then take 1/p root after all-reduce
        if grads:
            total_norm_tensor = torch.zeros(1, dtype=torch.float32, device=device)
            for g in grads:
                total_norm_tensor += g.float().norm(norm_type) ** norm_type
        else:
            total_norm_tensor = torch.zeros(1, dtype=torch.float32, device=device)

        if model_parallel_group is not None and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                total_norm_tensor,
                op=torch.distributed.ReduceOp.SUM,
                group=model_parallel_group,
            )
        total_norm = total_norm_tensor.item() ** (1.0 / norm_type)

    # Clip in-place
    clip_coeff = max_norm / (total_norm + 1e-6)
    if clip_coeff < 1.0:
        for g in grads:
            g.mul_(clip_coeff)

    return total_norm


# ===========================================================================
# optimizer.py — base class
# ===========================================================================

class MegatronOptimizer(ABC):
    """Base optimizer class. All Neuron_SP optimizers inherit from this.

    Provides the interface for gradient preparation, stepping, state
    management, and DES-LOC moment synchronization.
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
        # Unity loss scale tensor (BF16 path has no dynamic scaler)
        self._scale_one = torch.tensor([1.0], dtype=torch.float32)

    @abstractmethod
    def prepare_grads(self) -> bool:
        """Prepare gradients for optimizer step. Returns False if step should be skipped."""
        ...

    @abstractmethod
    def step(self) -> bool:
        """Execute optimizer step. Returns True if step was taken."""
        ...

    @abstractmethod
    def zero_grad(self, set_to_none: bool = True) -> None:
        ...

    def clip_grad_norm(self, clip_grad: float) -> float:
        """Compute grad norm and clip gradients in place."""
        return clip_grad_norm(
            parameters=self.params,
            max_norm=clip_grad,
            norm_type=2.0,
            model_parallel_group=None,
        )

    def get_loss_scale(self) -> torch.Tensor:
        """Return current loss scale (1.0 for BF16, dynamic for FP16)."""
        return self._scale_one

    def state_dict(self) -> dict:
        sd: dict = {'optimizer': self.optimizer.state_dict()}
        return sd

    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def reload_model_params(self) -> None:
        """Reload model params from optimizer's FP32 master copy."""
        # Subclasses that maintain a separate FP32 copy must override.
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_parameters(self) -> List[torch.nn.Parameter]:
        """Return flat list of all optimizer parameters."""
        all_params: List[torch.nn.Parameter] = []
        for pg in self.optimizer.param_groups:
            all_params.extend(pg['params'])
        return all_params


class MixedPrecisionOptimizer(MegatronOptimizer):
    """Optimizer with mixed-precision (FP32 master weights, BF16 model weights)."""

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

        # BF16 must not have FP16-style scaler; FP16 must have one.
        if self.grad_scaler is None:
            assert not config.fp16, "fp16 training requires a grad scaler."

        # Overflow detection tensor (only needed when a scaler is active)
        if self.grad_scaler is not None:
            self._found_inf = torch.tensor([0.0], dtype=torch.float32)

        # Build three param lists from the provided params:
        #   bf16_model_params   — original BF16/FP16 model tensors
        #   fp32_master_params  — FP32 copies used by the optimizer
        #   fp32_model_params   — params that are already FP32 in the model
        self.bf16_model_params: List[torch.nn.Parameter] = []
        self.fp32_master_params: List[torch.nn.Parameter] = []
        self.fp32_model_params: List[torch.nn.Parameter] = []

        for param_group in self.optimizer.param_groups:
            bf16_group: List[torch.nn.Parameter] = []
            fp32_master_group: List[torch.nn.Parameter] = []
            fp32_model_group: List[torch.nn.Parameter] = []

            for i, param in enumerate(param_group['params']):
                if not param.requires_grad:
                    continue
                if param.dtype in (torch.float16, torch.bfloat16):
                    bf16_group.append(param)
                    # Create FP32 master copy
                    master = param.detach().clone().float()
                    master.requires_grad_(True)
                    # Point the optimizer to the FP32 copy
                    param_group['params'][i] = master
                    # Cross-reference: BF16 param knows its master
                    param.main_param = master
                    fp32_master_group.append(master)
                    # Transfer optimizer state key if present
                    if param in self.optimizer.state:
                        self.optimizer.state[master] = self.optimizer.state.pop(param)
                elif param.dtype == torch.float32:
                    fp32_model_group.append(param)
                else:
                    raise TypeError(
                        f"Unsupported param dtype {param.dtype}. "
                        "Expected float16, bfloat16, or float32."
                    )

            self.bf16_model_params.extend(bf16_group)
            self.fp32_master_params.extend(fp32_master_group)
            self.fp32_model_params.extend(fp32_model_group)

    # ------------------------------------------------------------------
    # Gradient preparation
    # ------------------------------------------------------------------

    def _copy_model_grads_to_main_grads(self) -> None:
        """Copy BF16 gradients into the FP32 master params' .grad field."""
        for bf16_param, fp32_param in zip(self.bf16_model_params, self.fp32_master_params):
            # Prefer pre-allocated contiguous buffer (main_grad) if present
            if hasattr(bf16_param, 'main_grad') and bf16_param.main_grad is not None:
                fp32_param.grad = bf16_param.main_grad.float()
            elif bf16_param.grad is not None:
                fp32_param.grad = bf16_param.grad.float()
            # Clear the BF16 grad to free memory
            bf16_param.grad = None

        # FP32 params: their grad is already usable directly
        for fp32_param in self.fp32_model_params:
            if hasattr(fp32_param, 'main_grad'):
                fp32_param.grad = fp32_param.main_grad

    def _copy_main_params_to_model_params(self) -> None:
        """Copy updated FP32 master weights back to BF16 model params."""
        for bf16_param, fp32_param in zip(self.bf16_model_params, self.fp32_master_params):
            bf16_param.data.copy_(fp32_param.data)

    def _copy_model_params_to_main_params(self) -> None:
        """Reload FP32 master weights from BF16 model params (for resume)."""
        for bf16_param, fp32_param in zip(self.bf16_model_params, self.fp32_master_params):
            fp32_param.data.copy_(bf16_param.data.float())

    def _unscale_and_check_inf(self) -> bool:
        """Unscale main grads and return True if inf/nan detected."""
        if self.grad_scaler is None:
            return False

        # Collect all FP32 master grads
        main_grads = [
            fp32.grad.data
            for fp32 in (self.fp32_master_params + self.fp32_model_params)
            if fp32.grad is not None
        ]
        self._found_inf.fill_(0.0)
        if main_grads:
            torch._amp_foreach_non_finite_check_and_unscale_(
                main_grads,
                self._found_inf,
                self.grad_scaler.inv_scale,
            )
        # All-reduce inf flag across model-parallel ranks if initialized
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                self._found_inf,
                op=torch.distributed.ReduceOp.MAX,
            )
        return bool(self._found_inf.item() > 0)

    @torch.no_grad()
    def prepare_grads(self) -> bool:
        """Copy BF16 grads → FP32 master grads; unscale + check inf.

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
        """Full optimizer step: prepare → clip → step → writeback."""
        # 1. Copy BF16 grads → FP32 master grads and check overflow
        found_inf = self.prepare_grads()
        if found_inf:
            return False

        # 2. Gradient clipping on master params
        if self.config.clip_grad > 0.0:
            self.clip_grad_norm(self.config.clip_grad)

        # 3. Adam step on FP32 masters
        self.optimizer.step()

        # 4. Write FP32 masters back to BF16 model weights
        self._copy_main_params_to_model_params()

        return True

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients on BF16 model params and FP32 masters."""
        for param in self.bf16_model_params + self.fp32_model_params:
            if set_to_none:
                param.grad = None
                if hasattr(param, 'main_grad'):
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
        """Reload BF16 model params from FP32 master copy."""
        self._copy_main_params_to_model_params()

    def state_dict(self) -> dict:
        sd: dict = {'optimizer': self.optimizer.state_dict()}
        if self.grad_scaler is not None:
            sd['grad_scaler'] = self.grad_scaler.state_dict()
        # Save FP32 master params (needed to warm-start from checkpoint)
        sd['fp32_master_params'] = [p.data for p in self.fp32_master_params]
        return sd

    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if 'grad_scaler' in state_dict and self.grad_scaler is not None:
            self.grad_scaler.load_state_dict(state_dict['grad_scaler'])
        if 'fp32_master_params' in state_dict:
            for current, saved in zip(
                self.fp32_master_params, state_dict['fp32_master_params']
            ):
                current.data.copy_(saved)


# ===========================================================================
# distrib_optimizer.py
# ===========================================================================

class DistributedOptimizer(MixedPrecisionOptimizer):
    """ZeRO-style distributed optimizer with DES-LOC Ku/Kv decomposed sync.

    Each rank holds 1/N of the FP32 master params + optimizer states.
    On optimizer step:
      1. Reduce-scatter gradients (or skip if non-Kx step)
      2. Update local FP32 shard
      3. Write updated shard back to BF16 model
      4. Broadcast so all ranks have consistent model (every step, not just Kx)

    DES-LOC extension: first moments (exp_avg) sync every Ku steps,
    second moments (exp_avg_sq) sync every Kv steps. This reduces
    optimizer state communication by (1 - 1/Ku) + (1 - 1/Kv) volume.
    """

    def __init__(
        self,
        config: OptimizerConfig,
        optimizer: torch.optim.Optimizer,
        params: List[torch.nn.Parameter],
        model_parallel_config: ModelParallelConfig,
        param_and_grad_buffers: List[ParamAndGradBuffer],
        data_parallel_group: torch.distributed.ProcessGroup,
        data_parallel_group_gloo: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        super().__init__(
            config=config,
            optimizer=optimizer,
            params=params,
            model_parallel_config=model_parallel_config,
            grad_scaler=None,  # DistributedOptimizer defaults to BF16 (no dynamic scale)
        )

        self.param_and_grad_buffers = param_and_grad_buffers
        self.data_parallel_group = data_parallel_group
        self.data_parallel_group_gloo = data_parallel_group_gloo

        self.data_parallel_world_size: int = torch.distributed.get_world_size(
            group=data_parallel_group
        )
        self.data_parallel_rank: int = torch.distributed.get_rank(
            group=data_parallel_group
        )

        # DES-LOC config (may be None when DES-LOC is disabled)
        self._desloc: Optional[DesLocConfig] = model_parallel_config.desloc

        # Global step counter — incremented in step() after each successful update.
        # Used to gate Ku/Kv moment synchronization.
        self._step_count: int = 0

        # Build per-rank param shards from the grad buffers.
        # Each rank is responsible for a contiguous 1/N slice of each buffer.
        #
        # shard_fp32_params: list of FP32 tensors that are the local shard of the
        #                    FP32 master weight buffer.
        # shard_param_views:  list of (buffer_grad_data, shard_start, shard_end)
        #                     used to scatter updated values back to the buffer.
        self._build_shards()

    # ------------------------------------------------------------------
    # Shard construction
    # ------------------------------------------------------------------

    def _build_shards(self) -> None:
        """Partition each grad/param buffer into per-rank shards.

        For each ParamAndGradBuffer we allocate a FP32 shard of size
        ceil(total_numel / dp_world_size) on this rank.  The shard covers
        elements [rank * shard_size, (rank+1) * shard_size) of the flattened
        FP32 master buffer.
        """
        dp_rank = self.data_parallel_rank
        dp_world = self.data_parallel_world_size

        # Master FP32 weight buffer shards and grad shards, one entry per
        # ParamAndGradBuffer.
        self._fp32_shards: List[torch.Tensor] = []      # FP32 param shard
        self._fp32_grad_shards: List[torch.Tensor] = [] # grad shard
        self._shard_sizes: List[int] = []               # elements in each shard
        self._padded_sizes: List[int] = []              # padded total buffer size

        for buf in self.param_and_grad_buffers:
            total_numel: int = buf.grad_data.numel()
            # Pad to a multiple of dp_world so every rank gets equal-sized shard
            padded = _round_up(total_numel, dp_world)
            shard_size = padded // dp_world
            shard_start = dp_rank * shard_size
            shard_end = min(shard_start + shard_size, total_numel)

            # Allocate FP32 master param shard (initialise from BF16 buffer if
            # param_data exists; otherwise from zero — grads are freshly synced
            # before any Adam step is taken).
            fp32_shard = torch.zeros(shard_size, dtype=torch.float32,
                                     device=buf.grad_data.device)
            # Initialise master weights from the BF16 model params that fall in
            # this rank's shard.  We iterate the param_index_map to find which
            # params overlap with [shard_start, shard_end).
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

            # FP32 grad shard (receives reduce-scattered grads before Adam step)
            fp32_grad_shard = torch.zeros(shard_size, dtype=torch.float32,
                                          device=buf.grad_data.device)

            self._fp32_shards.append(fp32_shard)
            self._fp32_grad_shards.append(fp32_grad_shard)
            self._shard_sizes.append(shard_size)
            self._padded_sizes.append(padded)

        # Build param_groups for the inner torch optimizer pointing at FP32 shards.
        # We replace the param_groups so that optimizer.step() operates on the shards.
        shard_params: List[torch.nn.Parameter] = [
            torch.nn.Parameter(shard, requires_grad=True)
            for shard in self._fp32_shards
        ]
        # The caller-supplied optimizer may have been built with the original params;
        # replace all param groups with a single group covering our shards.
        self.optimizer.param_groups = [
            {
                'params': shard_params,
                'lr': self.config.lr or 1e-4,
                'betas': (self.config.adam_beta1, self.config.adam_beta2),
                'eps': self.config.adam_eps,
                'weight_decay': self.config.weight_decay,
            }
        ]
        self._shard_params = shard_params

    # ------------------------------------------------------------------
    # Grad preparation (reduce-scatter then copy into shard)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _reduce_scatter_grads(self) -> None:
        """Reduce-scatter BF16 gradients and store in FP32 grad shards.

        Each rank ends up with the averaged gradients for its local shard.
        """
        dp_world = self.data_parallel_world_size
        dp_rank = self.data_parallel_rank

        for buf, fp32_grad_shard, shard_size, padded in zip(
            self.param_and_grad_buffers,
            self._fp32_grad_shards,
            self._shard_sizes,
            self._padded_sizes,
        ):
            total_numel = buf.grad_data.numel()

            # Pad the grad buffer to padded size
            if padded > total_numel:
                grad_padded = torch.zeros(padded, dtype=buf.grad_data.dtype,
                                          device=buf.grad_data.device)
                grad_padded[:total_numel].copy_(buf.grad_data)
            else:
                grad_padded = buf.grad_data

            # Allocate output tensor for reduce-scatter (same dtype as buffer)
            output_shard = torch.zeros(shard_size, dtype=grad_padded.dtype,
                                       device=grad_padded.device)

            # reduce-scatter: sum grads across ranks, each rank gets its slice
            torch.distributed.reduce_scatter_tensor(
                output_shard,
                grad_padded,
                op=torch.distributed.ReduceOp.SUM,
                group=self.data_parallel_group,
            )
            # Average and cast to FP32
            fp32_grad_shard.copy_(output_shard.float().div_(dp_world))

        # Point each shard param's .grad at the corresponding grad shard
        for shard_param, fp32_grad_shard in zip(self._shard_params, self._fp32_grad_shards):
            shard_param.grad = fp32_grad_shard

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prepare_grads(self) -> bool:
        """Reduce-scatter grads into FP32 shards. Always returns False (no scaler)."""
        self._reduce_scatter_grads()
        return False

    @torch.no_grad()
    def step(self) -> bool:
        """Execute optimizer step with DES-LOC moment sync.

        After base Adam step on local shard:
        - Always: write FP32 shard → BF16 model + broadcast (prevent Kx spike)
        - Every Ku steps: all-reduce exp_avg across ranks
        - Every Kv steps: all-reduce exp_avg_sq across ranks
        """
        # 1. Prepare: reduce-scatter grads into FP32 grad shards
        found_inf = self.prepare_grads()
        if found_inf:
            return False

        # 2. Optional grad clipping on shard params
        if self.config.clip_grad > 0.0:
            grads_for_clip = [
                sp.grad for sp in self._shard_params if sp.grad is not None
            ]
            if grads_for_clip:
                total_norm = clip_grad_norm(
                    parameters=self._shard_params,
                    max_norm=self.config.clip_grad,
                    norm_type=2.0,
                    model_parallel_group=self.data_parallel_group,
                )

        # 3. Local Adam update on FP32 shard
        self.optimizer.step()

        # 4. Increment step counter
        self._step_count += 1

        # 5. DES-LOC: synchronize moments when configured
        if self._desloc is not None and self._desloc.enabled:
            sync_u = self._desloc.is_ku_step(self._step_count)
            sync_v = self._desloc.is_kv_step(self._step_count)
            if sync_u or sync_v:
                self.sync_moments(sync_first=sync_u, sync_second=sync_v)

        # 6. ALWAYS: write FP32 shard → BF16 model + broadcast to prevent spike
        self.shard_to_model_broadcast()

        return True

    # ------------------------------------------------------------------
    # DES-LOC moment sync
    # ------------------------------------------------------------------

    def sync_moments(self, sync_first: bool = False, sync_second: bool = False) -> None:
        """Synchronize optimizer moments across DES-LOC tiers.

        Args:
            sync_first: All-reduce first moment (exp_avg) — Ku step.
            sync_second: All-reduce second moment (exp_avg_sq) — Kv step.
        """
        if not (sync_first or sync_second):
            return

        dp_world = self.data_parallel_world_size

        for shard_param in self._shard_params:
            state = self.optimizer.state.get(shard_param)
            if state is None:
                continue

            if sync_first and 'exp_avg' in state:
                m = state['exp_avg']
                torch.distributed.all_reduce(
                    m,
                    op=torch.distributed.ReduceOp.SUM,
                    group=self.data_parallel_group,
                )
                m.div_(dp_world)

            if sync_second and 'exp_avg_sq' in state:
                v = state['exp_avg_sq']
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

        This is called EVERY step (not just Kx steps) to prevent the
        Frankenstein model bug where each rank's model has only 1/N
        params updated.

        Flow:
          1. Copy this rank's FP32 shard → its slice of the BF16 grad buffer
             (which is also the param buffer when param_data is not separate)
          2. All-gather the full BF16 buffer across DP ranks
          3. Copy the all-gathered buffer back to model param tensors
        """
        dp_world = self.data_parallel_world_size
        dp_rank = self.data_parallel_rank

        for buf, fp32_shard, shard_size, padded in zip(
            self.param_and_grad_buffers,
            self._fp32_shards,
            self._shard_sizes,
            self._padded_sizes,
        ):
            total_numel = buf.grad_data.numel()

            # 1. Allocate a BF16 output buffer (padded to multiple of dp_world)
            if padded > total_numel:
                full_bf16 = torch.zeros(padded, dtype=torch.bfloat16,
                                        device=buf.grad_data.device)
            else:
                full_bf16 = torch.zeros(total_numel, dtype=torch.bfloat16,
                                        device=buf.grad_data.device)

            # 2. Convert our shard to BF16 and place it in the correct slot
            shard_bf16 = fp32_shard[:shard_size].bfloat16()
            shard_start = dp_rank * shard_size
            shard_end = min(shard_start + shard_size, total_numel)
            copy_len = shard_end - shard_start
            full_bf16[shard_start: shard_start + copy_len].copy_(
                shard_bf16[:copy_len]
            )

            # 3. All-gather: each rank contributes its shard slice into full_bf16
            #    We use all_reduce SUM since exactly one rank has non-zero in each
            #    slot after the slice-copy above. (A true all-gather would need a
            #    list-of-tensors API which is less portable; SUM is equivalent when
            #    only one rank contributes to each region.)
            torch.distributed.all_reduce(
                full_bf16[:padded] if padded > total_numel else full_bf16,
                op=torch.distributed.ReduceOp.SUM,
                group=self.data_parallel_group,
            )

            # 4. Write the reconstructed BF16 buffer back to model param tensors
            result = full_bf16[:total_numel]
            for param, (ps, pe, _) in buf.param_index_map.items():
                param.data.copy_(result[ps:pe].view(param.data.shape))

            # 5. Also update the FP32 shard from the freshly broadcast data
            #    (ensures all ranks' shards are consistent after moment sync)
            shard_start2 = dp_rank * shard_size
            shard_end2 = min(shard_start2 + shard_size, total_numel)
            copy_len2 = shard_end2 - shard_start2
            fp32_shard[:copy_len2].copy_(result[shard_start2:shard_end2].float())

    # ------------------------------------------------------------------
    # Zero grad
    # ------------------------------------------------------------------

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients: clear grad buffers and shard param grads."""
        # Zero the contiguous grad buffers
        for buf in self.param_and_grad_buffers:
            buf.grad_data.zero_()

        # Zero shard param grads
        for sp in self._shard_params:
            if set_to_none:
                sp.grad = None
            elif sp.grad is not None:
                sp.grad.detach_()
                sp.grad.zero_()

    # ------------------------------------------------------------------
    # State dict
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return optimizer state including FP32 shards and step count."""
        return {
            'optimizer': self.optimizer.state_dict(),
            'step_count': self._step_count,
            'fp32_shards': [s.cpu() for s in self._fp32_shards],
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore optimizer state from checkpoint."""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self._step_count = state_dict.get('step_count', 0)
        if 'fp32_shards' in state_dict:
            for shard, saved in zip(self._fp32_shards, state_dict['fp32_shards']):
                shard.copy_(saved.to(shard.device))

    def save_parameter_state(self, filename: str) -> None:
        """Save FP32 parameter shards from DP-rank-0 to disk.

        Only DP rank 0 writes; other ranks contribute via gather.
        """
        dp_rank = self.data_parallel_rank

        # Gather all shards to rank 0
        gathered_shards: List[Optional[torch.Tensor]] = []
        for fp32_shard, padded in zip(self._fp32_shards, self._padded_sizes):
            dp_world = self.data_parallel_world_size
            shard_size = fp32_shard.numel()

            if dp_rank == 0:
                all_shards = [
                    torch.zeros(shard_size, dtype=torch.float32,
                                device=fp32_shard.device)
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
                {
                    'fp32_shards': gathered_shards,
                    'step_count': self._step_count,
                },
                filename,
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _round_up(n: int, multiple: int) -> int:
    """Round n up to the next multiple of `multiple`."""
    return ((n + multiple - 1) // multiple) * multiple
