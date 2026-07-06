# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Emerging optimizer registry for Neuron_SP.

Ported from Megatron-LM/megatron/core/optimizer/emerging_optimizers.py
with DES-LOC extensions for heterogeneous GPU tier routing.

Overview
--------
The emerging optimizer registry maps optimizer names (e.g. ``"muon"``,
``"lion"``, ``"soap"``) to their constructor, state-initialisation function,
and default parameter routing overrides.  This design lets callers switch
between optimizers by changing a single config field
(``OptimizerConfig.optimizer = "muon"``) without touching any other code.

Optimizer routing on heterogeneous clusters
--------------------------------------------
On DES-LOC clusters, different GPU tiers may use different optimizers:

  - H100 ranks (high memory, high FLOPS): Muon for 2-D weight matrices;
    Adam for embeddings, biases, and 1-D params.
  - A6000 ranks (memory-limited, lower FLOPS): Adam only (Muon's
    Newton-Schulz iteration is VRAM-intensive at large matrix sizes).
  - Blackwell / Consumer ranks: same as H100 (Muon capable).

The registry provides ``route_params_by_tier()`` which partitions model
parameters into (muon_params, adam_params) based on the current rank's
tier assignment.

Public API
----------
  EmergingOptimizerEntry     — registry entry dataclass
  register_emerging_optimizer — add a new entry to the registry
  get_emerging_optimizer      — retrieve an entry by name
  list_emerging_optimizers    — list registered names
  build_emerging_optimizer    — construct optimizer from config
  route_params_by_tier        — tier-aware param routing
  EMERGING_OPTIMIZER_REGISTRY — the global registry dict
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional emerging_optimizers package (NVIDIA)
# ---------------------------------------------------------------------------
try:
    from emerging_optimizers.orthogonalized_optimizers import (
        AdaptiveMuon,
        OrthogonalizedOptimizer,
        get_muon_scale_factor,
    )
    _HAS_EMERGING_OPTIMIZERS = True
except ImportError:
    _HAS_EMERGING_OPTIMIZERS = False
    AdaptiveMuon = None
    OrthogonalizedOptimizer = None

# ---------------------------------------------------------------------------
# Tier constants (mirrors desloc_config.TierType)
# ---------------------------------------------------------------------------
TIER_H100       = "h100"
TIER_A6000      = "a6000"
TIER_BLACKWELL  = "blackwell"
TIER_CONSUMER   = "consumer"

# Tiers that support Muon (sufficient VRAM for Newton-Schulz on large matrices)
MUON_CAPABLE_TIERS = {TIER_H100, TIER_BLACKWELL}


# ---------------------------------------------------------------------------
# Registry entry dataclass
# ---------------------------------------------------------------------------

@dataclass
class EmergingOptimizerEntry:
    """Registration record for a single emerging optimizer.

    Attributes:
        name:                  Short name used in ``OptimizerConfig.optimizer``.
        optimizer_cls:         The torch.optim.Optimizer subclass to instantiate.
        init_state_fn:         Callable that lazily initialises optimizer state
                               (needed by distributed checkpoint formats that
                               require state to exist before loading tensors).
        config_to_kwargs:      ``(config) -> dict`` mapping config fields to
                               constructor keyword arguments.
        default_param_overrides: Dict mapping :class:`ParamKey` → override dict
                               applied to parameters before group construction
                               (e.g. route non-linear params to Adam).
        muon_capable_tiers:    Tier names where this optimizer is supported;
                               other tiers fall back to Adam.
        description:           Human-readable description for logging.
    """
    name: str
    optimizer_cls: Any
    init_state_fn: Callable
    config_to_kwargs: Callable
    default_param_overrides: Dict = field(default_factory=dict)
    muon_capable_tiers: List[str] = field(default_factory=lambda: list(MUON_CAPABLE_TIERS))
    description: str = ""


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

EMERGING_OPTIMIZER_REGISTRY: Dict[str, EmergingOptimizerEntry] = {}


def register_emerging_optimizer(entry: EmergingOptimizerEntry) -> None:
    """Register *entry* in the global emerging optimizer registry.

    Args:
        entry: :class:`EmergingOptimizerEntry` to register.

    Raises:
        ValueError: If an entry with the same name already exists.
    """
    if entry.name in EMERGING_OPTIMIZER_REGISTRY:
        raise ValueError(
            f"register_emerging_optimizer: optimizer '{entry.name}' is already "
            "registered. Use a different name or call the registry directly to override."
        )
    EMERGING_OPTIMIZER_REGISTRY[entry.name] = entry
    logger.debug("register_emerging_optimizer: registered '%s'.", entry.name)


def get_emerging_optimizer(name: str) -> EmergingOptimizerEntry:
    """Return the registry entry for *name*.

    Args:
        name: Optimizer name (e.g. ``"muon"``, ``"lion"``).

    Returns:
        :class:`EmergingOptimizerEntry` for *name*.

    Raises:
        KeyError: If *name* is not registered.
    """
    if name not in EMERGING_OPTIMIZER_REGISTRY:
        available = list(EMERGING_OPTIMIZER_REGISTRY.keys())
        raise KeyError(
            f"Emerging optimizer '{name}' is not registered. "
            f"Available: {available}"
        )
    return EMERGING_OPTIMIZER_REGISTRY[name]


def list_emerging_optimizers() -> List[str]:
    """Return a sorted list of registered optimizer names."""
    return sorted(EMERGING_OPTIMIZER_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Muon registration (using our local MuonOptimizer as fallback)
# ---------------------------------------------------------------------------

def _muon_init_state_fn(opt: Any, config: Any = None) -> None:
    """Lazily initialise Muon momentum buffers for all parameters.

    Called before checkpoint loading so that state tensors exist in memory
    before in-place copy operations fill them.  Without this call,
    ``opt.state[param]`` is empty and the checkpoint loader raises KeyError.
    """
    for group in opt.param_groups:
        for p in group["params"]:
            if p not in opt.state:
                opt.state[p] = {}
            if "momentum_buffer" not in opt.state[p]:
                opt.state[p]["momentum_buffer"] = torch.zeros_like(p.data)


def _muon_config_to_kwargs(config: Any) -> dict:
    """Extract Muon constructor kwargs from *config*.

    Reads ``config.lr``, ``config.muon_momentum``, ``config.muon_nesterov``,
    and ``config.muon_ns_steps`` (all with sensible defaults when absent).
    """
    return {
        "lr":       getattr(config, "lr", 0.02),
        "momentum": getattr(config, "muon_momentum", 0.95),
        "nesterov": getattr(config, "muon_nesterov", True),
        "ns_steps": getattr(config, "muon_ns_steps", 5),
    }


def _lion_init_state_fn(opt: Any, config: Any = None) -> None:
    """Lazily initialise Lion exp_avg buffers for all parameters."""
    for group in opt.param_groups:
        for p in group["params"]:
            if p not in opt.state:
                opt.state[p] = {}
            if "exp_avg" not in opt.state[p]:
                opt.state[p]["exp_avg"] = torch.zeros_like(p.data)


def _lion_config_to_kwargs(config: Any) -> dict:
    return {
        "lr":           getattr(config, "lr", 1e-4),
        "betas":        (getattr(config, "lion_beta1", 0.9), getattr(config, "lion_beta2", 0.99)),
        "weight_decay": getattr(config, "weight_decay", 0.0),
    }


def _soap_init_state_fn(opt: Any, config: Any = None) -> None:
    """No-op init state for SOAP (lazy init handled internally)."""
    pass


def _soap_config_to_kwargs(config: Any) -> dict:
    return {
        "lr":               getattr(config, "lr", 3e-4),
        "betas":            (getattr(config, "adam_beta1", 0.95), getattr(config, "adam_beta2", 0.95)),
        "eps":              getattr(config, "adam_eps", 1e-8),
        "weight_decay":     getattr(config, "weight_decay", 1e-2),
        "precondition_frequency": getattr(config, "soap_precondition_freq", 10),
    }


# Register built-in optimizers
def _register_builtin_optimizers() -> None:
    """Register Muon, Lion, and SOAP into the global registry.

    Called once at module import.  Gracefully skips any optimizer whose
    dependency is not installed.
    """
    # -----------------------------------------------------------------------
    # Muon — use our local MuonOptimizer as a fallback when emerging_optimizers
    # is not installed; use the upstream AdaptiveMuon when it is available.
    # -----------------------------------------------------------------------
    try:
        if _HAS_EMERGING_OPTIMIZERS and AdaptiveMuon is not None:
            muon_cls = AdaptiveMuon
        else:
            from deepspeed.core.optimizer.layer_wise_optimizer import MuonOptimizer
            muon_cls = MuonOptimizer
    except ImportError:
        from deepspeed.core.optimizer.layer_wise_optimizer import MuonOptimizer
        muon_cls = MuonOptimizer

    register_emerging_optimizer(EmergingOptimizerEntry(
        name="muon",
        optimizer_cls=muon_cls,
        init_state_fn=_muon_init_state_fn,
        config_to_kwargs=_muon_config_to_kwargs,
        muon_capable_tiers=list(MUON_CAPABLE_TIERS),
        description="Momentum Orthogonalized by Newton-Schulz (Kosson et al. 2024).",
    ))

    # -----------------------------------------------------------------------
    # Lion — Evolved Sign Momentum
    # -----------------------------------------------------------------------
    try:
        try:
            from lion_pytorch import Lion as _Lion
        except ImportError:
            from emerging_optimizers.scalar_optimizers import Lion as _Lion

        register_emerging_optimizer(EmergingOptimizerEntry(
            name="lion",
            optimizer_cls=_Lion,
            init_state_fn=_lion_init_state_fn,
            config_to_kwargs=_lion_config_to_kwargs,
            muon_capable_tiers=list(MUON_CAPABLE_TIERS) + [TIER_A6000, TIER_CONSUMER],
            description="Evolved Sign Momentum optimizer (Chen et al. 2023).",
        ))
    except ImportError:
        logger.debug("Lion optimizer not available (neither lion-pytorch nor emerging_optimizers installed).")

    # -----------------------------------------------------------------------
    # SOAP — Shampoo-like second-order optimizer
    # -----------------------------------------------------------------------
    try:
        from emerging_optimizers.soap import SOAP as _SOAP
        register_emerging_optimizer(EmergingOptimizerEntry(
            name="soap",
            optimizer_cls=_SOAP,
            init_state_fn=_soap_init_state_fn,
            config_to_kwargs=_soap_config_to_kwargs,
            muon_capable_tiers=list(MUON_CAPABLE_TIERS),
            description="Shampoo as Adam Preconditioner (Vyas et al. 2024).",
        ))
    except ImportError:
        logger.debug("SOAP optimizer not available (emerging_optimizers not installed).")


_register_builtin_optimizers()


# ---------------------------------------------------------------------------
# Tier-aware parameter routing
# ---------------------------------------------------------------------------

def route_params_by_tier(
    params: List[torch.nn.Parameter],
    current_tier: Optional[str],
    optimizer_name: str = "muon",
) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """Partition *params* into (emerging_optimizer_params, adam_params).

    On Muon-capable tiers (H100, Blackwell) 2-D matrix parameters are routed
    to Muon; all other parameters (embeddings, biases, 1-D params) go to Adam.
    On non-capable tiers (A6000, Consumer) all parameters go to Adam.

    Args:
        params:         Full list of model parameters.
        current_tier:   This rank's tier name (``TIER_H100``, ``TIER_A6000``, …).
        optimizer_name: Name of the emerging optimizer to route to.

    Returns:
        (emerging_params, adam_params) — two disjoint lists that together
        contain all of *params*.
    """
    try:
        entry = get_emerging_optimizer(optimizer_name)
        capable_tiers = entry.muon_capable_tiers
    except KeyError:
        # Unknown optimizer: send everything to Adam
        return [], list(params)

    if current_tier not in capable_tiers:
        logger.debug(
            "route_params_by_tier: tier '%s' not Muon-capable → all params → Adam.",
            current_tier,
        )
        return [], list(params)

    from deepspeed.core.optimizer.layer_wise_optimizer import is_managed_by_layer_wise_optimizer

    emerging_params: List[torch.nn.Parameter] = []
    adam_params: List[torch.nn.Parameter] = []
    for p in params:
        if is_managed_by_layer_wise_optimizer(p):
            emerging_params.append(p)
        else:
            adam_params.append(p)

    logger.debug(
        "route_params_by_tier: tier='%s' optimizer='%s' → %d emerging / %d adam params.",
        current_tier, optimizer_name, len(emerging_params), len(adam_params),
    )
    return emerging_params, adam_params


# ---------------------------------------------------------------------------
# Factory: build_emerging_optimizer
# ---------------------------------------------------------------------------

def build_emerging_optimizer(
    config: Any,
    params: List[torch.nn.Parameter],
    optimizer_name: Optional[str] = None,
    current_tier: Optional[str] = None,
) -> Tuple[Any, Any, List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """Build the emerging optimizer and an Adam fallback for *params*.

    Routing:
      - 2-D matrix params on Muon-capable tiers → emerging optimizer.
      - All other params → Adam fallback.

    Args:
        config:         ``OptimizerConfig`` (or any object with ``lr``, etc.).
        params:         All model parameters to optimise.
        optimizer_name: Override name; defaults to ``config.optimizer``.
        current_tier:   This rank's tier name for routing decisions.

    Returns:
        ``(emerging_opt, adam_opt, emerging_params, adam_params)`` 4-tuple.
        ``emerging_opt`` is ``None`` when all params are routed to Adam.
        ``adam_opt`` is ``None`` when all params are routed to the emerging opt.

    Raises:
        KeyError: If the optimizer name is not registered.
    """
    name = optimizer_name or getattr(config, "optimizer", "muon")
    entry = get_emerging_optimizer(name)

    emerging_params, adam_params = route_params_by_tier(params, current_tier, name)

    # Build emerging optimizer
    emerging_opt: Optional[Any] = None
    if emerging_params:
        kwargs = entry.config_to_kwargs(config)
        emerging_opt = entry.optimizer_cls(emerging_params, **kwargs)
        entry.init_state_fn(emerging_opt, config)
        logger.info(
            "build_emerging_optimizer: '%s' optimizer created "
            "(%d params, tier='%s').",
            name, len(emerging_params), current_tier,
        )

    # Build Adam fallback
    adam_opt: Optional[Any] = None
    if adam_params:
        try:
            from transformer_engine.pytorch.optimizers import FusedAdam as _Adam
        except ImportError:
            try:
                from apex.optimizers import FusedAdam as _Adam
            except ImportError:
                from torch.optim import AdamW as _Adam
        adam_opt = _Adam(
            adam_params,
            lr=getattr(config, "lr", 1e-4),
            betas=(getattr(config, "adam_beta1", 0.9), getattr(config, "adam_beta2", 0.999)),
            eps=getattr(config, "adam_eps", 1e-8),
            weight_decay=getattr(config, "weight_decay", 0.0),
        )
        logger.info(
            "build_emerging_optimizer: Adam fallback created "
            "(%d params, tier='%s').",
            len(adam_params), current_tier,
        )

    return emerging_opt, adam_opt, emerging_params, adam_params


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "EmergingOptimizerEntry",
    "EMERGING_OPTIMIZER_REGISTRY",
    "register_emerging_optimizer",
    "get_emerging_optimizer",
    "list_emerging_optimizers",
    "build_emerging_optimizer",
    "route_params_by_tier",
    # Tier constants
    "TIER_H100",
    "TIER_A6000",
    "TIER_BLACKWELL",
    "TIER_CONSUMER",
    "MUON_CAPABLE_TIERS",
]
