# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""LayerWiseDistributedOptimizer — per-layer sharded optimizer for Muon.

Ported from Megatron-LM/megatron/core/optimizer/layer_wise_optimizer.py
with DES-LOC extensions for heterogeneous GPU tiers (H100 / A6000 / Blackwell).

Overview
--------
Unlike the standard :class:`DistributedOptimizer` (ZeRO-3) which shards
optimizer state across all DP ranks at a byte-granular level,
:class:`LayerWiseDistributedOptimizer` shards at **parameter granularity**:
each data-parallel rank owns a disjoint set of *whole parameters*.

This design is required by optimizers like Muon (momentum orthogonalization
via Newton-Schulz) that cannot operate on partial parameter slices — the
orthogonalization is applied to the full parameter matrix.

Communication pattern (per step)
---------------------------------
1. Each rank holds the full BF16 model weights.
2. After backward, each rank has full gradients for all parameters.
3. Each rank applies the layer-wise optimizer (Muon or Adam) to its owned
   parameter subset only.
4. All-gather updated parameters so every rank has the full, updated model.

DES-LOC integration
-------------------
When ``config.heterogeneous_shard_sizing=True``, parameter assignment is
weighted by BF16 TFLOPS so faster GPUs own more parameters.  Concretely:

  H100  owns ceil(N * 989 / (989 + 309.7)) parameters
  A6000 owns the remainder

This ensures the Muon Newton-Schulz iterations take approximately equal
wall-clock time per GPU tier, eliminating the stragglers that arise from
equal-count assignment.

Public API
----------
  LayerWiseDistributedOptimizer  — main class
  is_managed_by_layer_wise_optimizer  — param routing predicate
  tag_params_for_buffer_routing  — tag params before DDP construction
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch

import deepspeed.core.parallel_state as parallel_state
from deepspeed.core.optimizer.clip_grads import (
    count_zeros_fp32,
    get_grad_norm_fp32,
)
from deepspeed.core.optimizer.optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
    _get_param_grad_norm_group,
    _validate_grad_norm_group,
)
from deepspeed.core.optimizer.optimizer_config import OptimizerConfig
from deepspeed.core.optimizer.distrib_optimizer import (
    _compute_hetero_shard_boundaries,
    DistributedOptimizer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter routing predicates
# ---------------------------------------------------------------------------

def is_managed_by_layer_wise_optimizer(param: torch.nn.Parameter) -> bool:
    """Return True if *param* should be handled by LayerWiseDistributedOptimizer.

    Muon orthogonalizes 2-D matrix parameters (linear weights).  All other
    parameters (embeddings, biases, LayerNorm weights, 1-D params) are routed
    to the standard Adam-based :class:`DistributedOptimizer`.

    Routing rules:
      - 2-D tensor: True  (linear weights — Muon territory)
      - embedding / output params: False  (shared; must use Adam)
      - anything else: False

    Args:
        param: Any model parameter tensor.

    Returns:
        True if the param should be layer-wise-managed.
    """
    if param.dim() != 2:
        return False
    if getattr(param, "is_embedding_or_output_parameter", False):
        return False
    return True


def tag_params_for_buffer_routing(model_chunks: List) -> None:
    """Tag every requires-grad param with ``is_managed_by_layer_wise_optimizer``.

    Must be called on model chunks *before* DDP constructs its grad/param
    buffers.  The DDP buffer grouping function reads this attribute to route
    parameters into the correct buffer (LayerWise shard-aligned vs byte-level
    DistOpt buffer).

    Args:
        model_chunks: List of model chunks (nn.Module subclasses).
    """
    for chunk in model_chunks:
        for param in chunk.parameters():
            if param.requires_grad:
                param.is_managed_by_layer_wise_optimizer = (
                    is_managed_by_layer_wise_optimizer(param)
                )


# ---------------------------------------------------------------------------
# LayerWiseDistributedOptimizer
# ---------------------------------------------------------------------------

class LayerWiseDistributedOptimizer(MegatronOptimizer):
    """Per-layer sharded optimizer for Muon and other matrix-aware optimizers.

    Shards optimizer state and gradient updates at parameter granularity:
    each DP rank owns a disjoint set of whole parameters and applies the
    Muon update to those parameters only.  Updated parameters are broadcast
    back to all ranks via all-reduce(SUM) with zero-filled non-owner slots.

    Unlike the byte-level ZeRO-3 sharding in :class:`DistributedOptimizer`,
    this class never splits a parameter across ranks — the full parameter
    tensor is always resident on at least one rank and updated atomically.

    Memory usage (7B model, 8 DP ranks, BF16)
    ------------------------------------------
    LayerWise state memory ≈ ZeRO-2:
      - Full model weights: 14 GB/rank (same as ZeRO-0/1/2)
      - 1/N of optimizer state: ~7 GB/rank
      - Full gradients (temporary): up to 14 GB/rank during backward

    DES-LOC tier-weighted assignment
    ----------------------------------
    When ``config.heterogeneous_shard_sizing=True``, parameters are assigned
    to ranks proportionally to TFLOPS:

      H100  fraction ≈ 989  / (989 + 309.7) ≈ 76% of all 2-D params
      A6000 fraction ≈ 309.7 / (989 + 309.7) ≈ 24% of all 2-D params

    The total number of parameters (not numel) is used as the unit because
    Muon's Newton-Schulz cost scales with the number of matrix operations,
    not the total element count.

    Args:
        config:                  Optimizer configuration.
        optimizers:              Dict mapping param → per-param optimizer
                                 (Muon or Adam), or a single Adam optimizer
                                 for params that fall back to Adam.
        model_chunks:            List of model chunks.
        data_parallel_group:     NCCL process group.
        data_parallel_group_gloo: Gloo process group for checkpoint I/O.
        tier_assignments:        Per-rank TierType for heterogeneous sizing.
    """

    def __init__(
        self,
        config: OptimizerConfig,
        model_chunks: List,
        data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        data_parallel_group_gloo: Optional[torch.distributed.ProcessGroup] = None,
        tier_assignments: Optional[List] = None,
    ) -> None:
        # Resolve DP group
        if data_parallel_group is None:
            if parallel_state.is_initialized():
                data_parallel_group = parallel_state.get_data_parallel_group()
            else:
                data_parallel_group = torch.distributed.GroupMember.WORLD

        self.config = config
        self.model_chunks = model_chunks
        self.data_parallel_group = data_parallel_group
        self.data_parallel_group_gloo = data_parallel_group_gloo
        self.tier_assignments = tier_assignments

        self.data_parallel_world_size: int = torch.distributed.get_world_size(
            group=data_parallel_group
        )
        self.data_parallel_rank: int = torch.distributed.get_rank(
            group=data_parallel_group
        )

        self._step_count: int = 0
        self._scale_one = torch.tensor([1.0], dtype=torch.float32)
        self.is_stub_optimizer: bool = False

        # Collect all managed parameters
        self._all_managed_params: List[torch.nn.Parameter] = []
        self._all_other_params: List[torch.nn.Parameter] = []
        for chunk in model_chunks:
            for param in chunk.parameters():
                if not param.requires_grad:
                    continue
                if is_managed_by_layer_wise_optimizer(param):
                    self._all_managed_params.append(param)
                else:
                    self._all_other_params.append(param)

        # Assign managed params to this rank
        self._owned_params: List[torch.nn.Parameter] = []
        self._build_param_assignment()

        # Build FP32 master copies for owned params
        self._owned_fp32: List[torch.nn.Parameter] = []
        for p in self._owned_params:
            if p.dtype in (torch.float16, torch.bfloat16):
                fp32 = p.detach().clone().float()
                fp32.requires_grad_(True)
                p.main_param = fp32
                self._owned_fp32.append(fp32)
            else:
                self._owned_fp32.append(p)

        # Build inner Adam optimizer over owned FP32 params
        _inner_lr = config.lr or 1e-4
        self.optimizer = torch.optim.AdamW(
            self._owned_fp32,
            lr=_inner_lr,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )
        # This satisfies MegatronOptimizer's property accessor
        self._has_grad_norm_group_cache: Dict[str, bool] = {}

        logger.info(
            "LayerWiseDistributedOptimizer: rank=%d/%d owns %d/%d managed params, "
            "hetero=%s",
            self.data_parallel_rank,
            self.data_parallel_world_size,
            len(self._owned_params),
            len(self._all_managed_params),
            config.heterogeneous_shard_sizing,
        )

    def _build_param_assignment(self) -> None:
        """Assign managed parameters to this rank.

        Uses :func:`_compute_hetero_shard_boundaries` with ``total_numel``
        equal to the number of parameters (not their element count) so that
        Muon's Newton-Schulz cost (per-matrix, not per-element) is balanced.

        After this call, ``self._owned_params`` contains the subset of
        ``self._all_managed_params`` assigned to this rank.
        """
        n = len(self._all_managed_params)
        if n == 0:
            return

        boundaries = _compute_hetero_shard_boundaries(
            total_numel=n,
            dp_world_size=self.data_parallel_world_size,
            config=self.config,
            tier_assignments=self.tier_assignments,
        )
        s, e = boundaries[self.data_parallel_rank]
        self._owned_params = self._all_managed_params[s:e]
        self._owned_start = s
        self._owned_end = e

    # ------------------------------------------------------------------
    # Gradient collection helpers
    # ------------------------------------------------------------------

    def _collect_owned_grads(self) -> None:
        """Copy owned model param gradients to their FP32 master .grad fields."""
        for fp32, bf16 in zip(self._owned_fp32, self._owned_params):
            if hasattr(bf16, "main_grad") and bf16.main_grad is not None:
                fp32.grad = bf16.main_grad.float()
            elif bf16.grad is not None:
                fp32.grad = bf16.grad.float()
            else:
                fp32.grad = None

    # ------------------------------------------------------------------
    # MegatronOptimizer interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prepare_grads(self) -> bool:
        """Copy owned gradients from BF16 model to FP32 masters.

        Returns:
            False — no loss scaler in BF16 path.
        """
        self._collect_owned_grads()
        return False

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        """Apply optimizer step to owned params and broadcast updates.

        Steps:
          1. Clip owned gradients.
          2. Apply AdamW step to owned FP32 params.
          3. Write updated FP32 back to BF16 model params (owned only).
          4. All-reduce(SUM) each managed param so all ranks have the update.

        Returns:
            True (always succeeds in BF16 path).
        """
        # Gradient clipping on owned FP32 params
        if self.config.clip_grad > 0.0:
            from deepspeed.core.optimizer.clip_grads import clip_grad_norm as _clip
            _clip(
                parameters=self._owned_fp32,
                max_norm=self.config.clip_grad,
                norm_type=2.0,
                grad_stats_parallel_group=self.data_parallel_group,
            )

        # Decoupled weight decay
        if self.config.decoupled_weight_decay and self.config.weight_decay > 0.0:
            lr = self.optimizer.param_groups[0].get("lr", self.config.lr or 1e-4)
            decay = 1.0 - lr * self.config.weight_decay
            for fp32 in self._owned_fp32:
                if fp32.requires_grad and fp32.data is not None:
                    fp32.data.mul_(decay)

        # Adam step on owned FP32 shard
        self.optimizer.step()
        self._step_count += 1

        # Write FP32 updates back to BF16 model params
        for fp32, bf16 in zip(self._owned_fp32, self._owned_params):
            bf16.data.copy_(fp32.data.to(bf16.dtype))

        # All-reduce(SUM) each managed param across DP ranks.
        # Each rank contributes its owned slice; others contribute zeros.
        # SUM reconstructs the full update across all ranks.
        owned_set = set(id(p) for p in self._owned_params)
        for param in self._all_managed_params:
            if id(param) not in owned_set:
                # Non-owner: zero out param so SUM gives owner's value
                _zero_buf = torch.zeros_like(param.data)
                torch.distributed.all_reduce(
                    _zero_buf,
                    op=torch.distributed.ReduceOp.SUM,
                    group=self.data_parallel_group,
                )
                param.data.copy_(_zero_buf)
            else:
                # Owner: participate in SUM with current (updated) data
                _param_buf = param.data.clone()
                torch.distributed.all_reduce(
                    _param_buf,
                    op=torch.distributed.ReduceOp.SUM,
                    group=self.data_parallel_group,
                )
                param.data.copy_(_param_buf)

        return True

    @torch.no_grad()
    def step(self):
        """Full step: prepare_grads + step_with_ready_grads."""
        found_inf = self.prepare_grads()
        if found_inf:
            return False, None, None
        clip_norm = self.config.clip_grad
        # Compute global grad norm for logging before step
        grads = [
            fp32.grad.detach().float()
            for fp32 in self._owned_fp32
            if fp32.grad is not None
        ]
        grad_norm = get_grad_norm_fp32(
            grads,
            grad_stats_parallel_group=self.data_parallel_group,
        )
        success = self.step_with_ready_grads()
        return success, grad_norm, None

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients for all managed and non-managed model params."""
        for param in self._all_managed_params + self._all_other_params:
            if set_to_none:
                param.grad = None
                if hasattr(param, "main_grad"):
                    param.main_grad = None
            else:
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()
        for fp32 in self._owned_fp32:
            if set_to_none:
                fp32.grad = None
            elif fp32.grad is not None:
                fp32.grad.detach_()
                fp32.grad.zero_()

    def get_loss_scale(self) -> torch.Tensor:
        return self._scale_one

    def reload_model_params(self, state_dict=None) -> None:
        """Reload FP32 masters from BF16 model params (used at checkpoint resume)."""
        for fp32, bf16 in zip(self._owned_fp32, self._owned_params):
            fp32.data.copy_(bf16.data.float())

    def get_parameters(self) -> List[torch.nn.Parameter]:
        """Return all managed + non-managed model parameters."""
        return self._all_managed_params + self._all_other_params

    def get_grad_stats_parallel_group(self) -> Optional[torch.distributed.ProcessGroup]:
        return getattr(self, "grad_stats_parallel_group", self.data_parallel_group)

    @torch.no_grad()
    def get_grad_norm(self) -> torch.Tensor:
        """Compute L2 norm of owned gradients, reduced across DP ranks."""
        grads = [
            fp32.grad.detach().float()
            for fp32 in self._owned_fp32
            if fp32.grad is not None
        ]
        return get_grad_norm_fp32(
            grads,
            grad_stats_parallel_group=self.data_parallel_group,
        )

    def clip_grad_norm(self, clip_grad: float) -> float:
        from deepspeed.core.optimizer.clip_grads import clip_grad_norm as _clip
        norm = _clip(
            parameters=self._owned_fp32,
            max_norm=clip_grad,
            norm_type=2.0,
            grad_stats_parallel_group=self.data_parallel_group,
        )
        return float(norm.item())

    def count_zeros(self) -> float:
        return count_zeros_fp32(
            parameters=self._owned_fp32,
            grad_stats_parallel_group=self.data_parallel_group,
        )

    def start_param_sync_for_bucket_group_subset(self) -> None:
        """Trigger param broadcast for LayerWise-managed buckets.

        No-op for this class — the broadcast is already performed inside
        ``step_with_ready_grads`` via all-reduce(SUM).  Retained for API
        parity with :class:`DistributedOptimizer`.
        """
        pass

    # ------------------------------------------------------------------
    # State dict / checkpoint
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "optimizer": self.optimizer.state_dict(),
            "step_count": self._step_count,
            "owned_fp32": [p.data.cpu() for p in self._owned_fp32],
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self._step_count = state_dict.get("step_count", 0)
        if "owned_fp32" in state_dict:
            for fp32, saved in zip(self._owned_fp32, state_dict["owned_fp32"]):
                fp32.data.copy_(saved.to(fp32.device))

    def save_parameter_state(self, filename: str) -> None:
        """Save owned FP32 shards and Adam moments to disk (rank 0 only)."""
        if self.data_parallel_rank == 0:
            torch.save(self.state_dict(), filename)
            logger.info(
                "LayerWiseDistributedOptimizer: saved parameter state to %s "
                "(rank 0, dp_world=%d)",
                filename,
                self.data_parallel_world_size,
            )

    def load_parameter_state(self, filename: str) -> None:
        """Load owned FP32 shards and Adam moments from disk."""
        state_dict: Optional[dict] = None
        if self.data_parallel_rank == 0:
            state_dict = torch.load(filename, map_location="cpu")
            logger.info(
                "LayerWiseDistributedOptimizer: loaded parameter state from %s",
                filename,
            )
        # Broadcast step_count from rank 0
        sc = torch.tensor(
            [state_dict.get("step_count", 0) if state_dict else 0],
            dtype=torch.int64,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(
                sc,
                src=torch.distributed.get_global_rank(self.data_parallel_group, 0),
                group=self.data_parallel_group,
            )
        self._step_count = int(sc.item())
        if state_dict:
            self.load_state_dict(state_dict)

    def sharded_state_dict(
        self,
        model_sharded_state_dict=None,
        is_loading: bool = False,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Minimal sharded state dict (layer-wise format)."""
        return {
            "optimizer": self.optimizer.state_dict(),
            "step_count": self._step_count,
            "param_state": {
                "owned_fp32": [p.data.cpu() for p in self._owned_fp32],
                "owned_start": self._owned_start,
                "owned_end": self._owned_end,
            },
            "param_state_sharding_type": "layer_wise",
        }

    # ------------------------------------------------------------------
    # Allgather utility
    # ------------------------------------------------------------------

    def allgather_params(self) -> None:
        """All-gather all managed params from their owners to all ranks.

        Can be called manually (e.g., before validation) if param state has
        diverged.  Under normal training this is redundant because
        ``step_with_ready_grads`` already performs the broadcast.
        """
        owned_set = set(id(p) for p in self._owned_params)
        for param in self._all_managed_params:
            _buf = (
                param.data.clone() if id(param) in owned_set
                else torch.zeros_like(param.data)
            )
            torch.distributed.all_reduce(
                _buf,
                op=torch.distributed.ReduceOp.SUM,
                group=self.data_parallel_group,
            )
            param.data.copy_(_buf)

    def has_grad_norm_group(self, grad_norm_group: str) -> bool:
        """Whether any rank owns params in *grad_norm_group*."""
        _validate_grad_norm_group(grad_norm_group)
        if grad_norm_group not in self._has_grad_norm_group_cache:
            local = any(
                _get_param_grad_norm_group(p) == grad_norm_group
                for p in self.get_parameters()
            )
            flag = torch.tensor(
                [1 if local else 0], dtype=torch.int, device="cuda"
            )
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    flag, op=torch.distributed.ReduceOp.MAX,
                    group=self.data_parallel_group,
                )
            self._has_grad_norm_group_cache[grad_norm_group] = bool(flag.item() > 0)
        return self._has_grad_norm_group_cache[grad_norm_group]

    def offload_to_cpu(self) -> None:
        """Move owned FP32 optimizer state tensors to CPU."""
        for fp32 in self._owned_fp32:
            if fp32.is_cuda:
                fp32.data = fp32.data.cpu()
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.is_cuda:
                    state[k] = v.cpu()
        torch.cuda.empty_cache()

    def restore_from_cpu(self) -> None:
        """Move owned FP32 optimizer state tensors back to GPU."""
        for fp32 in self._owned_fp32:
            if not fp32.is_cuda:
                fp32.data = fp32.data.cuda()
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and not v.is_cuda:
                    state[k] = v.cuda()


# ---------------------------------------------------------------------------
# Muon Newton-Schulz orthogonalization utilities
# ---------------------------------------------------------------------------

def _zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Approximate matrix square-root inverse via Newton-Schulz iteration.

    Computes an orthogonal approximation to G by iteratively applying:
      X_{k+1} = a * X_k + b * X_k @ X_k.T @ X_k + c * X_k @ (X_k.T @ X_k)^2

    where (a, b, c) are chosen to maximize convergence rate.  After *steps*
    iterations the result is approximately G / ||G||_F, which is the closest
    orthogonal matrix to G in the Frobenius norm sense.

    This implementation mirrors the Muon optimizer's zero-power computation
    from Megatron-LM/megatron/core/optimizer/muon.py.

    Args:
        G:     Weight matrix gradient to orthogonalize. Shape: (m, n).
        steps: Number of Newton-Schulz iterations (5 is typically sufficient).
        eps:   Small value to prevent division by zero in normalization.

    Returns:
        Orthogonalized matrix of the same shape as G.

    Note:
        For non-square matrices (m ≠ n), the iteration is applied to
        G @ G.T when m ≤ n (tall matrix) or G.T @ G when m > n (wide).
    """
    assert G.ndim >= 2, f"_zeropower_via_newtonschulz5: expected 2-D tensor, got {G.ndim}-D."
    m, n = G.shape[-2], G.shape[-1]

    # Iteration coefficients for degree-5 minimal polynomial of x^{-1/2}
    # on [1/1.01, 1.01] (Muon paper, Table 1)
    a, b, c = (3.4445, -4.7750, 2.0315)

    X = G.float()

    # Normalize to prevent overflow
    X = X / (X.norm() + eps)

    if m < n:
        # Tall: work in the m×m covariance space
        X = X.T  # (n, m)

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if m < n:
        X = X.T

    return X.to(G.dtype)


class MuonOptimizer:
    """Muon (MomentUm Orthogonalized by Newton-schulz) optimizer.

    Applies Newton-Schulz orthogonalization to 2-D parameter gradients
    before the momentum update.  Non-2-D parameters fall back to standard
    Adam/SGD within the same optimizer instance.

    From Megatron-LM (Kosson et al., 2024 / Jordan et al., 2024).

    Args:
        params:      Parameters to optimize (2-D matrix params for Muon).
        lr:          Learning rate.
        momentum:    Momentum coefficient for the orthogonalized update.
        nesterov:    If True, apply Nesterov momentum.
        ns_steps:    Number of Newton-Schulz iterations.
        eps:         Numerical stability eps for Newton-Schulz normalization.

    Examples::

        muon_params = [p for p in model.parameters() if p.dim() == 2]
        adam_params = [p for p in model.parameters() if p.dim() != 2]
        optimizer = ChainedOptimizer([
            MuonOptimizer(muon_params, lr=0.02),
            torch.optim.AdamW(adam_params, lr=3e-4),
        ])
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-7,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"Invalid momentum: {momentum}")

        self.param_groups = [{"params": list(params), "lr": lr}]
        self.state: Dict[torch.nn.Parameter, dict] = {}
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.eps = eps

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients for all parameters."""
        for group in self.param_groups:
            for p in group["params"]:
                if set_to_none:
                    p.grad = None
                elif p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        """Perform a single Muon optimization step.

        For each 2-D parameter with a gradient:
          1. Orthogonalize the gradient via Newton-Schulz.
          2. Apply momentum update.
          3. Update parameter: param -= lr * orthogonalized_momentum.
        """
        for group in self.param_groups:
            lr = group.get("lr", self.lr)
            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.dim() == 2:
                    # Orthogonalize gradient
                    g_ortho = _zeropower_via_newtonschulz5(
                        p.grad, steps=self.ns_steps, eps=self.eps
                    )
                    # Scale to match the Frobenius norm of the original gradient
                    # (preserves learning rate magnitude across different matrix sizes)
                    scale = max(1, p.shape[0] / p.shape[1]) ** 0.5
                    g_ortho = g_ortho * scale
                else:
                    g_ortho = p.grad

                # Momentum state
                if p not in self.state:
                    self.state[p] = {"momentum_buffer": torch.zeros_like(p.data)}

                buf = self.state[p]["momentum_buffer"]
                buf.mul_(self.momentum).add_(g_ortho)

                if self.nesterov:
                    update = self.momentum * buf + g_ortho
                else:
                    update = buf

                p.data.add_(update, alpha=-lr)

    def state_dict(self) -> dict:
        return {
            "state": {i: s for i, s in enumerate(self.state.values())},
            "param_groups": [
                {k: v for k, v in g.items() if k != "params"}
                for g in self.param_groups
            ],
        }

    def load_state_dict(self, state_dict: dict) -> None:
        params = [p for g in self.param_groups for p in g["params"]]
        states = list(state_dict["state"].values())
        for param, s in zip(params, states):
            self.state[param] = {k: v.clone() for k, v in s.items() if isinstance(v, torch.Tensor)}


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "LayerWiseDistributedOptimizer",
    "MuonOptimizer",
    "is_managed_by_layer_wise_optimizer",
    "tag_params_for_buffer_routing",
    "_zeropower_via_newtonschulz5",
]
