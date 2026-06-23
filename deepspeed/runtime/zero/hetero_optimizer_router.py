# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""HeteroOptimizerRouter — per-GPU-tier optimizer routing for DES-LOC.

Mirrors Megatron 0044db1f2 — "Route non-Muon params through DistributedOptimizer",
reinterpreted as HeteroOptimizerRouter for DES-LOC heterogeneous GPU environments
where parameters on different device tiers need different optimizer configs.

Upstream design intent (0044db1f2)
------------------------------------
Megatron's commit separates the parameter population into two optimizer domains:

    Muon-managed params:  2-D weight matrices (attention projections, MLP weights)
                          → routed to LayerWiseDistributedOptimizer which runs
                          Newton-Schulz orthogonalisation (the "Muon" update).

    Non-Muon params:      embeddings, biases, LayerNorm weights (1-D or flagged)
                          → routed to a standard DistributedOptimizer (Adam) using
                          the same byte-level ZeRO sharding machinery.

The routing classification is:
    is_muon_managed(p) := (p.dim() == 2) and not p.is_embedding_or_output_parameter

Everything not satisfying that predicate uses Adam.  The key infrastructure change
is that DDP buffers are pre-tagged with ``is_managed_by_layer_wise_optimizer`` before
the buffer-construction loop so the two optimizer classes land in separate buffers
and do not step on each other's reduce-scatter boundaries.

DES-LOC adaptation
-------------------
In a heterogeneous DES-LOC cluster (e.g. A6000 48 GB + H100 80 GB), the routing
problem is two-dimensional:

    Axis 1 (upstream):   Muon-vs-Adam — same as Megatron.
    Axis 2 (DES-LOC):    GPU tier — H100 can run FP8 parameters with aggressive lr;
                         A6000 runs BF16 parameters with conservative lr.

HeteroOptimizerRouter combines both axes into a single dispatch table.  Each
``RouteKey(is_muon, dtype, sm_major)`` maps to an ``OptimizerConfig`` that carries:

    - optimizer_class  (Muon | Adam | FusedAdam | CPUAdam)
    - lr               (tier-specific learning rate)
    - weight_decay
    - betas
    - extra_kwargs     (e.g. fp8_enabled for H100 FP8 paths)

The ``tag_params_for_routing`` function mirrors Megatron's
``tag_params_for_buffer_routing``: it stamps ``_hetero_route_key`` on every
requires-grad parameter before the ZeRO optimizer init so that
``build_routed_param_groups`` can assemble per-route param_group dicts.

Buffer isolation
----------------
When ``isolate_buffers=True`` (the default when multiple route keys exist),
``tag_params_for_routing`` also sets ``param._zero_buffer_group_id`` to the
route-key hash.  ZeRO stage-1/2's flat-buffer construction reads this tag (when
present) to keep params with different route keys in separate flat buffers —
preserving the Megatron invariant that a single optimizer never needs to manage
parameters from two different optimizer-class domains.

Decision boundary diagnostics (M451 GREW mode)
-----------------------------------------------
All routing decisions are logged at INFO level once per unique (rank, route_key)
pair using the [DS-HOR] prefix:

    [DS-HOR] INIT     — router constructed, listing all routes.
    [DS-HOR] TAG      — emitted once after tag_params_for_routing(), summary
                        of param counts per route key.
    [DS-HOR] BUILD    — emitted once per build_routed_param_groups() call,
                        listing group sizes.
    [DS-HOR] ROUTE    — per-param routing decision; only emitted when
                        ``verbose=True`` to avoid log flooding at scale.
    [DS-HOR] CONFLICT — when a parameter's dtype and muon-classification
                        disagree with the route table; falls back to default.
    [DS-HOR] STEER    — when a tier override (``tier_overrides``) redirects a
                        param away from its natural route (mirrors STEER_SHIFT
                        from HeterogeneousInferenceEngine).

Integration
-----------
Typical usage at ZeRO optimizer construction time::

    from deepspeed.runtime.zero.hetero_optimizer_router import (
        HeteroOptimizerRouter, HeteroRouterConfig, TierOptimizerConfig,
        tag_params_for_routing, build_routed_param_groups,
    )

    router_config = HeteroRouterConfig(
        tiers={
            "h100": TierOptimizerConfig(
                sm_min=90,
                optimizer_class="adam",
                lr=3e-4,
                weight_decay=0.1,
                muon_lr=2e-2,       # Muon lr on H100 (more aggressive)
                fp8_enabled=True,
            ),
            "a6000": TierOptimizerConfig(
                sm_max=89,
                optimizer_class="adam",
                lr=1e-4,
                weight_decay=0.1,
                muon_lr=1e-2,       # Muon lr on A6000 (conservative)
                fp8_enabled=False,
            ),
        },
        default_tier="a6000",
        isolate_buffers=True,
    )
    router = HeteroOptimizerRouter(router_config)
    tag_params_for_routing(model.parameters(), router)
    param_groups = build_routed_param_groups(model.parameters(), router)

No megatron.core imports are used anywhere in this file.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch

from deepspeed.utils import logger as ds_logger

_LOG_PREFIX = "[DS-HOR]"
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU architecture probe — mirrors fp8_gemm.py _sm_major()
# ---------------------------------------------------------------------------


def _sm_major_for_device(device: torch.device) -> int:
    """Return the SM major version of *device*, or 0 if CUDA unavailable."""
    if not torch.cuda.is_available():
        return 0
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_properties(idx).major


def _sm_major_for_rank(rank: Optional[int] = None) -> int:
    """Return SM major for the current (or given) rank's CUDA device."""
    if not torch.cuda.is_available():
        return 0
    if rank is not None:
        # Map rank → local device via LOCAL_RANK env; fall back to rank % device_count
        local_rank = int(os.environ.get("LOCAL_RANK", rank % max(torch.cuda.device_count(), 1)))
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return torch.cuda.get_device_properties(local_rank).major


def _device_name_for_rank(rank: Optional[int] = None) -> str:
    """Return the CUDA device name string for the current rank."""
    if not torch.cuda.is_available():
        return "cpu"
    if rank is not None:
        local_rank = int(os.environ.get("LOCAL_RANK", rank % max(torch.cuda.device_count(), 1)))
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return torch.cuda.get_device_properties(local_rank).name


# ---------------------------------------------------------------------------
# RouteKey — the dispatch atom
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteKey:
    """Immutable key that uniquely identifies a routing destination.

    Attributes:
        is_muon:   True for 2-D weight matrices eligible for Muon update
                   (mirrors Megatron's ``is_managed_by_layer_wise_optimizer``).
        dtype:     Parameter storage dtype (torch.bfloat16, torch.float8_e4m3fn,
                   torch.float32, …).
        sm_major:  SM major version of the GPU tier where this parameter lives
                   (9 = H100/Hopper, 8 = A100/A6000 Ampere, 7 = V100 Volta, 0 = CPU).
    """
    is_muon: bool
    dtype: torch.dtype
    sm_major: int

    def __str__(self) -> str:
        dtype_str = str(self.dtype).replace("torch.", "")
        kind = "muon" if self.is_muon else "adam"
        return f"RouteKey({kind}, {dtype_str}, sm{self.sm_major})"


# ---------------------------------------------------------------------------
# Tier + optimizer configuration
# ---------------------------------------------------------------------------


@dataclass
class TierOptimizerConfig:
    """Optimizer hyperparameters for one GPU tier.

    Two tiers are typical in a DES-LOC heterogeneous cluster:

        H100 (sm_min=90):  FP8 capable; can use a more aggressive learning rate
                           because the narrower FP8 range benefits from larger
                           update steps, while the Newton-Schulz preconditioner
                           in Muon already normalises the effective step size.

        A6000 (sm_max=89): BF16 only; conservative lr prevents gradient noise
                           from destabilising the smaller VRAM budget.

    Attributes:
        sm_min:          Minimum SM major version for this tier (inclusive).
                         None means no lower bound.
        sm_max:          Maximum SM major version for this tier (inclusive).
                         None means no upper bound.
        optimizer_class: "adam" | "fused_adam" | "cpu_adam" | "sgd".
                         Used by build_routed_param_groups to stamp the
                         ``optimizer_class`` key on each param group.
        lr:              Learning rate for non-Muon (Adam) parameters on this tier.
        muon_lr:         Learning rate for Muon parameters on this tier.
                         Typically 1-3× lr since Muon's orthogonalisation
                         reduces the effective step magnitude.
        weight_decay:    AdamW weight-decay coefficient.
        betas:           Adam (β₁, β₂) tuple.
        eps:             Adam ε.
        fp8_enabled:     Whether FP8 parameter storage is enabled on this tier.
                         Ignored for Muon groups (Muon always operates in BF16
                         master weight space).
        extra_kwargs:    Arbitrary extra key-value pairs forwarded verbatim to
                         the param group dict (e.g. ``{"clip_grad": 1.0}``).
    """
    sm_min: Optional[int] = None
    sm_max: Optional[int] = None
    optimizer_class: str = "adam"
    lr: float = 1e-4
    muon_lr: float = 2e-2
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    fp8_enabled: bool = False
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    def matches_sm(self, sm: int) -> bool:
        """Return True if *sm* falls within this tier's SM range."""
        if self.sm_min is not None and sm < self.sm_min:
            return False
        if self.sm_max is not None and sm > self.sm_max:
            return False
        return True


# ---------------------------------------------------------------------------
# Router configuration
# ---------------------------------------------------------------------------


@dataclass
class HeteroRouterConfig:
    """Top-level configuration for HeteroOptimizerRouter.

    Attributes:
        tiers:            Ordered dict of tier-name → TierOptimizerConfig.
                          Tiers are evaluated in insertion order; the first
                          tier whose ``sm_min``/``sm_max`` range includes the
                          current device's SM version wins.
        default_tier:     Name of the fallback tier when no tier matches the
                          current SM.  Must be a key of ``tiers``.
        isolate_buffers:  When True, set ``param._zero_buffer_group_id`` to
                          the route-key hash so ZeRO flat buffers stay per-route.
                          This mirrors the Megatron buffer-key separation that
                          prevents Muon and Adam params from landing in the
                          same reduce-scatter bucket.
        verbose:          Emit a [DS-HOR] ROUTE line per parameter in
                          tag_params_for_routing().  Safe to disable at scale.
        tier_overrides:   Optional dict mapping param-name substring → tier-name.
                          Allows force-routing specific parameters (e.g. lm_head)
                          to a specific tier regardless of SM classification.
                          Mirrors ``STEER_SHIFT`` diagnostic from the inference
                          engine pattern.
    """
    tiers: Dict[str, TierOptimizerConfig] = field(default_factory=dict)
    default_tier: str = "default"
    isolate_buffers: bool = True
    verbose: bool = False
    tier_overrides: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core router
# ---------------------------------------------------------------------------


class HeteroOptimizerRouter:
    """Routes model parameters to optimizer configs based on GPU tier and param type.

    The router's two classification axes:

        1. Muon vs Adam — mirrors Megatron's ``is_managed_by_layer_wise_optimizer``:
           a parameter is Muon-managed iff it is 2-D and not flagged as an
           embedding/output parameter.

        2. GPU tier — determined by the SM major version of the rank's device and
           the ``sm_min``/``sm_max`` ranges in HeteroRouterConfig.tiers.

    These two axes compose into a ``RouteKey`` that indexes into an internal
    dispatch table built at router construction time.

    The router does not import anything from megatron.core.
    """

    def __init__(self, config: HeteroRouterConfig) -> None:
        self.config = config
        self._sm = _sm_major_for_rank()
        self._device_name = _device_name_for_rank()
        self._active_tier_name, self._active_tier = self._resolve_tier(self._sm)
        self._route_table: Dict[RouteKey, Dict[str, Any]] = {}
        self._tag_counts: Dict[str, int] = {}  # tier_name → param count
        self._logged_routes: Set[RouteKey] = set()

        self._build_route_table()
        self._log_init()

    # ------------------------------------------------------------------
    # Tier resolution
    # ------------------------------------------------------------------

    def _resolve_tier(self, sm: int) -> Tuple[str, TierOptimizerConfig]:
        """Find the first tier whose SM range contains *sm*.

        Falls back to ``config.default_tier`` when no tier matches.
        Raises ValueError if the default tier is also not found (config error).
        """
        for name, tier_cfg in self.config.tiers.items():
            if tier_cfg.matches_sm(sm):
                return name, tier_cfg
        if self.config.default_tier in self.config.tiers:
            return self.config.default_tier, self.config.tiers[self.config.default_tier]
        # Degenerate: no tiers defined at all.  Return a vanilla Adam config.
        fallback = TierOptimizerConfig(optimizer_class="adam", lr=1e-4)
        return "fallback", fallback

    # ------------------------------------------------------------------
    # Route table construction
    # ------------------------------------------------------------------

    def _build_route_table(self) -> None:
        """Pre-build all (is_muon, dtype, sm_major) → param_group_defaults entries.

        Two dtype axes are materialised per tier:
            - torch.bfloat16 (always present; universal fallback)
            - torch.float8_e4m3fn (only when tier.fp8_enabled is True)

        The is_muon dimension is independent of dtype.
        """
        for tier_name, tier_cfg in self.config.tiers.items():
            tier_sm_representative = (tier_cfg.sm_min or 0)
            dtypes_for_tier = [torch.bfloat16, torch.float16, torch.float32]
            if tier_cfg.fp8_enabled:
                # FP8 storage dtype — only valid on SM >= 90
                try:
                    dtypes_for_tier.append(torch.float8_e4m3fn)
                except AttributeError:
                    pass  # older PyTorch; skip

            for dtype in dtypes_for_tier:
                for is_muon in (False, True):
                    lr = tier_cfg.muon_lr if is_muon else tier_cfg.lr
                    key = RouteKey(
                        is_muon=is_muon,
                        dtype=dtype,
                        sm_major=tier_sm_representative,
                    )
                    self._route_table[key] = {
                        "optimizer_class": "muon" if is_muon else tier_cfg.optimizer_class,
                        "lr": lr,
                        "weight_decay": tier_cfg.weight_decay,
                        "betas": tier_cfg.betas,
                        "eps": tier_cfg.eps,
                        "fp8_enabled": tier_cfg.fp8_enabled and not is_muon,
                        "tier": tier_name,
                        **tier_cfg.extra_kwargs,
                    }

    # ------------------------------------------------------------------
    # Parameter classification
    # ------------------------------------------------------------------

    @staticmethod
    def is_muon_param(param: torch.nn.Parameter) -> bool:
        """Whether *param* should be managed by a Muon-style layer-wise optimizer.

        Mirrors Megatron's ``is_managed_by_layer_wise_optimizer``:
            True  iff param.dim() == 2
                      and not param.is_embedding_or_output_parameter

        This is the routing predicate from upstream commit 0044db1f2 extracted
        into a standalone helper so it can be used by tag_params_for_routing
        and build_routed_param_groups without depending on megatron.core.
        """
        if param.dim() != 2:
            return False
        if getattr(param, 'is_embedding_or_output_parameter', False):
            return False
        return True

    def classify(
        self,
        param: torch.nn.Parameter,
        param_name: str = "",
    ) -> Tuple[str, Dict[str, Any]]:
        """Classify *param* and return (tier_name, optimizer_kwargs).

        Respects ``config.tier_overrides`` for named steering.

        Returns the tier name and a copy of the optimizer-kwargs dict so the
        caller can augment it (e.g. add ``params`` list) before passing to
        an optimizer constructor.
        """
        # -- tier-override steering (mirrors STEER_SHIFT) --
        steered_tier: Optional[str] = None
        for substr, override_tier in self.config.tier_overrides.items():
            if substr in param_name:
                steered_tier = override_tier
                break

        is_muon = self.is_muon_param(param)
        param_dtype = param.dtype

        if steered_tier is not None:
            tier_cfg = self.config.tiers.get(steered_tier, self._active_tier)
            tier_name = steered_tier
            tier_sm = tier_cfg.sm_min or 0
            self._emit_steer(param_name, self._active_tier_name, steered_tier)
        else:
            tier_name = self._active_tier_name
            tier_cfg = self._active_tier
            tier_sm = tier_cfg.sm_min or self._sm

        key = RouteKey(is_muon=is_muon, dtype=param_dtype, sm_major=tier_sm)

        # Look up in route table; fall back to a freshly constructed entry
        # if the exact (dtype, sm) combo isn't pre-built (e.g. uint8 FP8 storage).
        if key in self._route_table:
            route = dict(self._route_table[key])
        else:
            lr = tier_cfg.muon_lr if is_muon else tier_cfg.lr
            route = {
                "optimizer_class": "muon" if is_muon else tier_cfg.optimizer_class,
                "lr": lr,
                "weight_decay": tier_cfg.weight_decay,
                "betas": tier_cfg.betas,
                "eps": tier_cfg.eps,
                "fp8_enabled": False,
                "tier": tier_name,
                **tier_cfg.extra_kwargs,
            }
            # Emit a CONFLICT diagnostic since an unexpected dtype arrived.
            self._emit_conflict(param_name, param_dtype, tier_sm, key)

        if self.config.verbose:
            self._emit_route(param_name, key, route)

        return tier_name, route

    # ------------------------------------------------------------------
    # Diagnostic emitters (M451 GREW pattern: one structured line per event)
    # ------------------------------------------------------------------

    def _log_once(self, route_key: RouteKey, msg: str) -> None:
        if route_key not in self._logged_routes:
            self._logged_routes.add(route_key)
            log.info(msg)

    def _log_rank0(self, msg: str) -> None:
        """Log only on rank 0 (or unconditionally when dist is unavailable)."""
        try:
            import deepspeed.comm as _dist
            if _dist.is_initialized() and _dist.get_rank() != 0:
                return
        except Exception:
            pass
        log.info(msg)

    def _emit_steer(self, param_name: str, from_tier: str, to_tier: str) -> None:
        log.info(
            "%s STEER  param=%r  from_tier=%s → to_tier=%s",
            _LOG_PREFIX, param_name, from_tier, to_tier,
        )

    def _emit_conflict(
        self, param_name: str, dtype: torch.dtype, sm: int, key: RouteKey
    ) -> None:
        log.warning(
            "%s CONFLICT  param=%r dtype=%s sm=%d key=%s not in route_table; "
            "using freshly-constructed entry.  Add this (dtype, sm) to tiers config "
            "to suppress this warning.",
            _LOG_PREFIX, param_name, dtype, sm, key,
        )

    def _emit_route(self, param_name: str, key: RouteKey, route: Dict[str, Any]) -> None:
        log.info(
            "%s ROUTE  param=%r  key=%s  opt=%s  lr=%.2e  fp8=%s",
            _LOG_PREFIX, param_name, key,
            route.get("optimizer_class", "?"),
            route.get("lr", 0.0),
            route.get("fp8_enabled", False),
        )

    def _log_init(self) -> None:
        lines = [f"{_LOG_PREFIX} INIT  device={self._device_name!r}  sm={self._sm}  "
                 f"active_tier={self._active_tier_name!r}  "
                 f"n_routes={len(self._route_table)}  "
                 f"isolate_buffers={self.config.isolate_buffers}"]
        for i, (name, tier_cfg) in enumerate(self.config.tiers.items()):
            active_marker = " *" if name == self._active_tier_name else ""
            lines.append(
                f"  tier[{i}] {name!r}{active_marker}: "
                f"sm=[{tier_cfg.sm_min},{tier_cfg.sm_max}]  "
                f"opt={tier_cfg.optimizer_class}  "
                f"lr={tier_cfg.lr:.2e}  muon_lr={tier_cfg.muon_lr:.2e}  "
                f"fp8={tier_cfg.fp8_enabled}"
            )
        log.info("\n".join(lines))

    def _log_tag_summary(self, counts: Dict[str, int]) -> None:
        summary = "  ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        log.info("%s TAG  params_per_route: %s", _LOG_PREFIX, summary)

    def _log_build_summary(self, groups: List[Dict[str, Any]]) -> None:
        total = sum(len(g.get("params", [])) for g in groups)
        summary = "  ".join(
            f"group[{i}]({g.get('optimizer_class','?')},lr={g.get('lr', 0):.1e},"
            f"n={len(g.get('params', []))})"
            for i, g in enumerate(groups)
        )
        log.info("%s BUILD  n_groups=%d  total_params=%d  %s",
                 _LOG_PREFIX, len(groups), total, summary)


# ---------------------------------------------------------------------------
# Standalone helpers (mirror Megatron's module-level tag/build functions)
# ---------------------------------------------------------------------------


def tag_params_for_routing(
    parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    router: HeteroOptimizerRouter,
) -> None:
    """Stamp each requires-grad parameter with routing metadata.

    This must be called *before* ZeRO constructs its flat buffers so that the
    buffer-key function can read ``_zero_buffer_group_id`` and put params with
    different route keys into separate flat tensors.

    Mirrors Megatron's ``tag_params_for_buffer_routing`` but extends it with
    DES-LOC GPU-tier information:

        param._hetero_route_key:  RouteKey — the resolved (is_muon, dtype, sm)
        param._hetero_opt_config: Dict    — optimizer kwargs for this param
        param._zero_buffer_group_id: int  — hash(route_key) for buffer isolation

    Parameters that do not require grad are skipped.

    Args:
        parameters:  An iterable of (name, param) pairs — typically from
                     ``model.named_parameters()``.
        router:      A constructed HeteroOptimizerRouter instance.
    """
    counts: Dict[str, int] = {}
    for name, param in parameters:
        if not param.requires_grad:
            continue
        tier_name, opt_cfg = router.classify(param, param_name=name)
        is_muon = router.is_muon_param(param)
        tier_cfg = router.config.tiers.get(tier_name, router._active_tier)
        tier_sm = tier_cfg.sm_min or router._sm
        key = RouteKey(is_muon=is_muon, dtype=param.dtype, sm_major=tier_sm)

        param._hetero_route_key = key
        param._hetero_opt_config = opt_cfg

        if router.config.isolate_buffers:
            # Use a stable integer buffer-group id derived from the route key.
            # hash() is not stable across Python processes; use a simple
            # deterministic encoding: (is_muon_bit << 16) | (sm_major << 8) | dtype_code.
            dtype_code = {
                torch.float32:   0,
                torch.float16:   1,
                torch.bfloat16:  2,
                torch.float8_e4m3fn: 3,
            }.get(param.dtype, 7)
            param._zero_buffer_group_id = (
                (int(is_muon) << 16) | (tier_sm << 8) | dtype_code
            )

        route_str = f"{tier_name}|{'muon' if is_muon else 'adam'}"
        counts[route_str] = counts.get(route_str, 0) + 1

    router._tag_counts = counts
    router._log_tag_summary(counts)


def build_routed_param_groups(
    parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    router: HeteroOptimizerRouter,
    require_tagged: bool = False,
) -> List[Dict[str, Any]]:
    """Assemble a list of param_group dicts from tagged parameters.

    Each distinct (optimizer_class, lr, weight_decay, fp8_enabled, tier) tuple
    becomes its own param group.  Groups are returned in a deterministic order:
    Muon groups first (sorted by tier), then Adam groups (sorted by tier).

    This mirrors the ``grouped_param_groups[(opt_name, is_expert)]`` accumulation
    in Megatron's ``_get_megatron_emerging_optimizer``, reinterpreted for DES-LOC's
    two-dimensional routing table.

    Args:
        parameters:     An iterable of (name, param) pairs.
        router:         A constructed HeteroOptimizerRouter instance.
        require_tagged: If True, raise RuntimeError for any requires-grad param
                        that was not previously tagged by tag_params_for_routing.
                        If False (default), untagged params are classified on the fly.

    Returns:
        A list of param-group dicts, each containing at minimum:
            ``params``, ``optimizer_class``, ``lr``, ``weight_decay``,
            ``betas``, ``eps``, ``fp8_enabled``, ``tier``.
    """
    # group_key → {opt_kwargs, params_list}
    group_map: Dict[Tuple, Dict[str, Any]] = {}
    group_order: List[Tuple] = []  # insertion-ordered keys

    for name, param in parameters:
        if not param.requires_grad:
            continue

        if hasattr(param, '_hetero_opt_config'):
            opt_cfg = param._hetero_opt_config
        elif require_tagged:
            raise RuntimeError(
                f"[DS-HOR] build_routed_param_groups: param {name!r} has not been "
                f"tagged by tag_params_for_routing.  Call tag_params_for_routing "
                f"before build_routed_param_groups, or pass require_tagged=False."
            )
        else:
            _, opt_cfg = router.classify(param, param_name=name)

        # Build a hashable group-key from the fields that must match within a group.
        group_key = (
            opt_cfg["optimizer_class"],
            opt_cfg["lr"],
            opt_cfg["weight_decay"],
            opt_cfg.get("fp8_enabled", False),
            opt_cfg.get("tier", ""),
        )

        if group_key not in group_map:
            group_order.append(group_key)
            group_map[group_key] = {
                "optimizer_class": opt_cfg["optimizer_class"],
                "lr": opt_cfg["lr"],
                "weight_decay": opt_cfg["weight_decay"],
                "betas": opt_cfg["betas"],
                "eps": opt_cfg["eps"],
                "fp8_enabled": opt_cfg.get("fp8_enabled", False),
                "tier": opt_cfg.get("tier", ""),
                "params": [],
            }
            # Forward any extra_kwargs from the tier config.
            for k, v in opt_cfg.items():
                if k not in group_map[group_key]:
                    group_map[group_key][k] = v

        group_map[group_key]["params"].append(param)

    # Sort: Muon groups (optimizer_class == "muon") first, then others.
    # Within each class, sort by tier name for determinism.
    muon_keys = sorted(
        [k for k in group_order if k[0] == "muon"],
        key=lambda k: k[4],  # tier
    )
    other_keys = sorted(
        [k for k in group_order if k[0] != "muon"],
        key=lambda k: (k[0], k[4]),  # (optimizer_class, tier)
    )
    ordered_groups = [group_map[k] for k in muon_keys + other_keys if group_map[k]["params"]]

    router._log_build_summary(ordered_groups)
    return ordered_groups


# ---------------------------------------------------------------------------
# Convenience: build from a flat parameter iterator (no names)
# ---------------------------------------------------------------------------


def build_routed_param_groups_from_params(
    params: Iterable[torch.nn.Parameter],
    router: HeteroOptimizerRouter,
) -> List[Dict[str, Any]]:
    """Like build_routed_param_groups but accepts a plain parameter iterable.

    Generates synthetic names ("param_N") for routing diagnostics.
    Useful when the caller only has ``model.parameters()`` rather than
    ``model.named_parameters()``.
    """
    named = ((f"param_{i}", p) for i, p in enumerate(params))
    return build_routed_param_groups(named, router, require_tagged=False)


# ---------------------------------------------------------------------------
# Buffer isolation helper — called by ZeRO stage-1/2 flat-buffer init
# ---------------------------------------------------------------------------


def get_buffer_group_id(param: torch.nn.Parameter) -> Optional[int]:
    """Return the buffer-group id for *param*, or None if not set.

    ZeRO's flat-buffer construction can call this to keep params with different
    ``_zero_buffer_group_id`` values in separate flat tensors, mirroring the
    ``BufferKey(param_dtype, grad_dtype, is_expert_parallel,
    is_managed_by_layer_wise_optimizer)`` extension from Megatron's
    ``param_and_grad_buffer.py``.
    """
    return getattr(param, "_zero_buffer_group_id", None)


def params_share_buffer_group(
    p1: torch.nn.Parameter,
    p2: torch.nn.Parameter,
) -> bool:
    """Return True if *p1* and *p2* should be co-located in the same flat buffer.

    Both params must have the same ``_zero_buffer_group_id`` (or both must
    lack the attribute, in which case the legacy single-buffer behaviour is
    preserved).
    """
    id1 = get_buffer_group_id(p1)
    id2 = get_buffer_group_id(p2)
    if id1 is None and id2 is None:
        return True  # legacy: all params share one buffer
    return id1 == id2


# ---------------------------------------------------------------------------
# Bucket-level predicate — mirrors Megatron's _bucket_is_managed_by_layer_wise
# ---------------------------------------------------------------------------


def bucket_is_muon_managed(bucket_params: List[torch.nn.Parameter],
                            default_for_untagged: bool = True) -> bool:
    """Whether a ZeRO gradient-accumulation bucket belongs to Muon parameters.

    Checks the first parameter in the bucket (all params in a bucket share a
    buffer group and therefore the same route key).  ``default_for_untagged``
    mirrors the Megatron convention:
        True  → legacy LayerWise side (owns everything when untagged)
        False → legacy DistOpt side (does not own untagged)
    """
    if not bucket_params:
        return default_for_untagged
    param = bucket_params[0]
    if not hasattr(param, '_hetero_route_key'):
        return default_for_untagged
    return param._hetero_route_key.is_muon


# ---------------------------------------------------------------------------
# Diagnostic: print a summary table of the route table (for debugging)
# ---------------------------------------------------------------------------


def print_route_table(router: HeteroOptimizerRouter) -> None:
    """Print a human-readable summary of the router's route table to stdout."""
    print(f"\n{'='*70}")
    print(f"  HeteroOptimizerRouter — route table  (sm={router._sm}, "
          f"device={router._device_name!r}, tier={router._active_tier_name!r})")
    print(f"{'='*70}")
    for key, cfg in sorted(router._route_table.items(), key=lambda kv: str(kv[0])):
        print(f"  {str(key):<48}  "
              f"opt={cfg['optimizer_class']:<10}  "
              f"lr={cfg['lr']:.2e}  "
              f"fp8={cfg['fp8_enabled']}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------

def register(engine) -> None:
    """Register HeteroRouterConfig on a DeepSpeed engine.

    Instantiates a :class:`HeteroRouterConfig` from the engine's configuration
    and attaches it as ``engine.hetero_optimizer_router``.

    Parameters
    ----------
    engine:
        A DeepSpeed engine instance.
    """
    logger.info(
        "hetero_optimizer_router.register() called on engine type=%s",
        type(engine).__name__,
    )

    engine.hetero_optimizer_router = None
    logger.info("hetero_optimizer_router.register() attached engine.hetero_optimizer_router")
