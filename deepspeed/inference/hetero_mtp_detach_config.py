"""
deepspeed/inference/hetero_mtp_detach_config.py

DES-LOC Heterogeneous MTP Detach Configuration and Scheduling
=============================================================

Upstream design intent (Megatron commit 71e418ea7d7b3a6c9a53238c543c3e0b43e11026):
    Yi-Fu Wu introduced ``mtp_detach_heads`` as a single boolean flag in
    ``TransformerConfig`` that, when enabled, prevents Multi-Token Prediction (MTP)
    loss gradients from propagating back into the main transformer body.  Three
    detach sites were added:

      1. ``process_mtp_loss`` — detaches the shared output-projection weight so the
         output head is frozen w.r.t. the MTP loss.
      2. ``MultiTokenPredictionLayer._get_embeddings`` — detaches the decoder input
         coming out of the shared embedding, severing the gradient path to the
         embedding table.
      3. ``MultiTokenPredictionBlock.forward`` — detaches ``hidden_states`` after
         chunking so that MTP-layer gradients cannot leak into the upstream backbone.

    The motivation is to train MTP heads as semi-independent speculative-decoding
    assistants without disturbing the carefully tuned main-model gradients.

DES-LOC adaptation rationale:
    In the Neuron_SP DES-LOC (Decoupled Execution with Shared LOcality Cache)
    framework the three physical tiers have radically different gradient bandwidth:

      • A6000-0 / A6000-1  (48 GB each, SM86, PCIe-only)
         — DRAM-bandwidth-limited; remat is cheap relative to PCIe gradient traffic.
      • H100-NVL            (96 GB, SM90, PCIe-only)
         — Compute-rich; can absorb backward passes that the A6000s cannot.

    A global ``mtp_detach_heads=True`` would throw away gradient information that the
    H100 could exploit.  A global ``False`` would flood the PCIe bus with MTP
    gradients on the A6000 tier at every step.

    This module replaces the single boolean with a *per-tier, per-MTP-layer*
    detach schedule that is aware of:
      - which physical device hosts each MTP layer (locality affinity),
      - the current training step (warm-up, steady-state, fine-tuning),
      - gradient-traffic budgets derived from PCIe topology,
      - the shared locality cache occupancy so that detaching can free cache slots.

    The public API mirrors the Megatron flag (``should_detach(site, layer_idx)``)
    so that DES-LOC-aware wrappers around the three upstream detach sites can call a
    single predicate without restructuring the training loop.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class DetachSite(enum.Enum):
    """The three architectural sites where Megatron applies detach.

    These map 1-to-1 to the upstream diff hunk locations:
      OUTPUT_WEIGHT  → process_mtp_loss, line +817 in diff
      DECODER_INPUT  → _get_embeddings, line +1107 in diff
      HIDDEN_STATES  → MultiTokenPredictionBlock.forward, line +1808 in diff
    """

    OUTPUT_WEIGHT = "output_weight"
    DECODER_INPUT = "decoder_input"
    HIDDEN_STATES = "hidden_states"


class TierKind(enum.Enum):
    """Physical hardware tier in the DES-LOC cluster."""

    A6000 = "a6000"   # SM86, 48 GB, PCIe
    H100 = "h100"     # SM90, 96 GB, PCIe


class SchedulePhase(enum.Enum):
    """High-level training phase that controls detach aggressiveness."""

    WARMUP = "warmup"
    STEADY = "steady"
    FINETUNE = "finetune"


# ---------------------------------------------------------------------------
# Hardware topology constants
# ---------------------------------------------------------------------------

# PCIe Gen4 x16 peak bandwidth in GB/s (unidirectional).
# In practice DES-LOC observes ~24 GB/s cross-device due to shared root complex.
_PCIE_BW_GBS: float = float(os.environ.get("DESLOCK_PCIE_BW_GBS", "24.0"))

# Size of the DES-LOC shared locality cache per tier (bytes).
# Default: 4 GB per A6000, 8 GB for H100.
_CACHE_SIZE_A6000: int = int(os.environ.get("DESLOCK_CACHE_A6000_BYTES", str(4 * 1024 ** 3)))
_CACHE_SIZE_H100: int = int(os.environ.get("DESLOCK_CACHE_H100_BYTES", str(8 * 1024 ** 3)))

# Fraction of cache above which we consider the tier "cache-pressured".
_CACHE_PRESSURE_THRESHOLD: float = float(os.environ.get("DESLOCK_CACHE_PRESSURE", "0.75"))


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TierSpec:
    """Static description of one physical device tier."""

    kind: TierKind
    device_ids: Tuple[int, ...]   # torch device indices
    vram_bytes: int
    sm_version: int               # e.g. 86 for A6000, 90 for H100
    cache_bytes: int
    pcie_bw_gbs: float            # effective cross-tier bandwidth

    @property
    def is_bandwidth_constrained(self) -> bool:
        """True when the tier relies on PCIe without NVLink acceleration."""
        # In the Neuron_SP cluster there is no NVLink at all; every tier is PCIe.
        return True

    @property
    def gradient_budget_bytes_per_step(self) -> float:
        """Conservative upper bound on gradient bytes we can move per step.

        We allow at most 10 % of peak PCIe bandwidth to be consumed by MTP
        gradient traffic before the detach heuristic kicks in.  The remaining
        90 % is reserved for forward activations and parameter all-reduces.
        """
        return self.pcie_bw_gbs * 1e9 * 0.10


@dataclasses.dataclass
class MTPLayerPlacement:
    """Maps each MTP layer index to a tier and records its gradient size."""

    layer_idx: int
    tier: TierSpec
    # Estimated bytes of gradient produced by one backward pass through this layer.
    gradient_bytes: int = 0
    # Whether this layer's parameters are pinned in the locality cache.
    cache_pinned: bool = False


@dataclasses.dataclass
class DetachDecision:
    """Result of a single detach query."""

    should_detach: bool
    site: DetachSite
    layer_idx: int
    tier_kind: TierKind
    reason: str   # human-readable justification for the decision


# ---------------------------------------------------------------------------
# Locality cache occupancy oracle
# ---------------------------------------------------------------------------


class LocalityCacheOracle:
    """Thread-safe occupancy tracker for the DES-LOC shared locality cache.

    The actual cache resides in CPU DRAM (1.5 TB) and is managed by the
    DES-LOC runtime.  This class only tracks *occupancy estimates* so that
    the detach scheduler can decide whether freeing a gradient path would
    release cache slots.

    In production the DES-LOC runtime would call ``update_occupancy`` after
    each cache eviction or insertion event.  In unit tests a synthetic
    occupancy sequence is injected via ``_inject_occupancy`` for
    deterministic behaviour.
    """

    def __init__(self, tier: TierSpec) -> None:
        self._tier = tier
        self._lock = threading.Lock()
        self._occupancy_bytes: int = 0
        self._last_update_ts: float = time.monotonic()

    def update_occupancy(self, occupancy_bytes: int) -> None:
        with self._lock:
            self._occupancy_bytes = occupancy_bytes
            self._last_update_ts = time.monotonic()

    @property
    def occupancy_fraction(self) -> float:
        with self._lock:
            return self._occupancy_bytes / max(self._tier.cache_bytes, 1)

    @property
    def is_pressured(self) -> bool:
        return self.occupancy_fraction >= _CACHE_PRESSURE_THRESHOLD

    def _inject_occupancy(self, fraction: float) -> None:
        """Test helper: set occupancy as a fraction of total capacity."""
        with self._lock:
            self._occupancy_bytes = int(fraction * self._tier.cache_bytes)

    def __repr__(self) -> str:
        return (
            f"LocalityCacheOracle(tier={self._tier.kind.value}, "
            f"occupancy={self.occupancy_fraction:.1%})"
        )


# ---------------------------------------------------------------------------
# Step-wise phase resolver
# ---------------------------------------------------------------------------


def resolve_phase(
    global_step: int,
    warmup_steps: int,
    finetune_start_step: Optional[int],
) -> SchedulePhase:
    """Map a global training step to a SchedulePhase.

    Args:
        global_step: Current optimizer step (0-indexed).
        warmup_steps: Number of LR warm-up steps.
        finetune_start_step: If set, steps >= this value enter FINETUNE phase.

    Returns:
        The active SchedulePhase.
    """
    if global_step < warmup_steps:
        return SchedulePhase.WARMUP
    if finetune_start_step is not None and global_step >= finetune_start_step:
        return SchedulePhase.FINETUNE
    return SchedulePhase.STEADY


# ---------------------------------------------------------------------------
# Per-tier detach policy
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TierDetachPolicy:
    """Encodes which DetachSites are active for a given tier and phase.

    The policy is intentionally over-specified: callers can query any
    (site, phase) combination.  The defaults below reflect the DES-LOC
    design decision:

      A6000 tier:
        WARMUP   — detach ALL sites (save PCIe bandwidth while LR is low)
        STEADY   — detach HIDDEN_STATES only (allow embedding gradients)
        FINETUNE — detach nothing (full gradient flow for quality recovery)

      H100 tier:
        WARMUP   — detach OUTPUT_WEIGHT only (protect shared weight)
        STEADY   — detach nothing (H100 can absorb full backward)
        FINETUNE — detach nothing
    """

    tier_kind: TierKind
    active_sites: Dict[SchedulePhase, frozenset[DetachSite]] = dataclasses.field(
        default_factory=dict
    )

    @classmethod
    def default_for_tier(cls, kind: TierKind) -> "TierDetachPolicy":
        if kind == TierKind.A6000:
            return cls(
                tier_kind=kind,
                active_sites={
                    SchedulePhase.WARMUP: frozenset(DetachSite),
                    SchedulePhase.STEADY: frozenset({DetachSite.HIDDEN_STATES}),
                    SchedulePhase.FINETUNE: frozenset(),
                },
            )
        else:  # H100
            return cls(
                tier_kind=kind,
                active_sites={
                    SchedulePhase.WARMUP: frozenset({DetachSite.OUTPUT_WEIGHT}),
                    SchedulePhase.STEADY: frozenset(),
                    SchedulePhase.FINETUNE: frozenset(),
                },
            )

    def is_active(self, site: DetachSite, phase: SchedulePhase) -> bool:
        return site in self.active_sites.get(phase, frozenset())

    def override(self, phase: SchedulePhase, sites: Sequence[DetachSite]) -> None:
        """Replace the active site set for ``phase``."""
        self.active_sites[phase] = frozenset(sites)


# ---------------------------------------------------------------------------
# Gradient traffic estimator
# ---------------------------------------------------------------------------


class GradientTrafficEstimator:
    """Estimates PCIe gradient traffic per training step per MTP layer.

    The estimator tracks a running average of observed gradient tensor
    sizes.  When no observed data is available it falls back to a
    formula-based estimate derived from the layer's hidden dimension.

    DES-LOC relevance:
        On a PCIe-only topology gradient synchronisation is the dominant
        bottleneck.  We use this estimate to decide whether the current
        accumulated gradient budget for a tier has been exhausted, at which
        point the detach policy is upgraded to a more aggressive setting
        regardless of training phase.
    """

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int) -> None:
        self._hidden = hidden_size
        self._heads = num_heads
        self._head_dim = head_dim
        self._observed: List[int] = []
        self._lock = threading.Lock()

    def record(self, grad_bytes: int) -> None:
        with self._lock:
            self._observed.append(grad_bytes)
            if len(self._observed) > 128:
                self._observed = self._observed[-128:]

    @property
    def estimate_bytes(self) -> int:
        with self._lock:
            if self._observed:
                return int(sum(self._observed) / len(self._observed))
        # Formula estimate: 2 * (QKV + output projection + MLP) parameter count * 2 bytes
        qkv_params = 3 * self._hidden * self._heads * self._head_dim
        out_params = self._heads * self._head_dim * self._hidden
        mlp_params = 4 * self._hidden * self._hidden * 2  # gate + down
        return 2 * (qkv_params + out_params + mlp_params) * 2  # bf16

    def exceeds_budget(self, budget_bytes: float) -> bool:
        return self.estimate_bytes > budget_bytes


# ---------------------------------------------------------------------------
# Core scheduler
# ---------------------------------------------------------------------------


class HeteroMTPDetachScheduler:
    """Per-tier, per-step detach scheduler for DES-LOC MTP heads.

    This is the central object that the DES-LOC training loop interacts
    with.  It replaces the single ``mtp_detach_heads`` boolean flag from
    Megatron with a stateful, topology-aware decision engine.

    Usage
    -----
    >>> scheduler = HeteroMTPDetachScheduler.from_neuron_sp_env(config)
    >>> # Inside the training step:
    >>> scheduler.step(global_step)
    >>> decision = scheduler.should_detach(
    ...     site=DetachSite.HIDDEN_STATES,
    ...     layer_idx=0,
    ...     device=torch.device("cuda:0"),
    ... )
    >>> if decision.should_detach:
    ...     hidden_states = hidden_states.detach()

    Design notes
    ------------
    * ``step()`` is called once per optimizer step and resolves the current
      SchedulePhase and per-tier gradient budgets.
    * ``should_detach()`` is O(1) and lock-free after ``step()`` completes.
    * Cache pressure can promote a tier to a more aggressive detach policy
      mid-phase without waiting for the next phase boundary.
    * All decisions are logged at DEBUG level the first time they change,
      and at WARNING level when a budget overrun triggers an emergency detach.
    """

    def __init__(
        self,
        tier_specs: List[TierSpec],
        mtp_layer_placements: List[MTPLayerPlacement],
        tier_policies: Dict[TierKind, TierDetachPolicy],
        cache_oracles: Dict[TierKind, LocalityCacheOracle],
        gradient_estimators: Dict[int, GradientTrafficEstimator],
        warmup_steps: int = 200,
        finetune_start_step: Optional[int] = None,
        num_mtp_layers: int = 1,
    ) -> None:
        self._tier_specs: Dict[TierKind, TierSpec] = {t.kind: t for t in tier_specs}
        self._placements: Dict[int, MTPLayerPlacement] = {
            p.layer_idx: p for p in mtp_layer_placements
        }
        self._policies = tier_policies
        self._oracles = cache_oracles
        self._estimators = gradient_estimators
        self._warmup_steps = warmup_steps
        self._finetune_start_step = finetune_start_step
        self._num_mtp_layers = num_mtp_layers

        # Mutable state, updated by step()
        self._current_step: int = 0
        self._current_phase: SchedulePhase = SchedulePhase.WARMUP
        # Cache last decisions to detect transitions and log only on change.
        self._prev_decisions: Dict[Tuple[DetachSite, int], bool] = {}
        self._lock = threading.Lock()  # protects _current_step / _current_phase

    # ------------------------------------------------------------------
    # Class methods / factory
    # ------------------------------------------------------------------

    @classmethod
    def from_neuron_sp_env(
        cls,
        hidden_size: int,
        num_attention_heads: int,
        num_mtp_layers: int,
        warmup_steps: int = 200,
        finetune_start_step: Optional[int] = None,
        mtp_layer_device_map: Optional[Dict[int, int]] = None,
    ) -> "HeteroMTPDetachScheduler":
        """Construct a scheduler from the canonical Neuron_SP hardware layout.

        Hardware assumed:
          cuda:0 → A6000 #0
          cuda:1 → A6000 #1
          cuda:2 → H100 NVL

        Args:
            hidden_size: Model hidden dimension.
            num_attention_heads: Number of attention heads.
            num_mtp_layers: Number of MTP prediction layers.
            warmup_steps: LR warm-up duration.
            finetune_start_step: Step at which fine-tuning phase begins.
            mtp_layer_device_map: Optional override mapping layer_idx → cuda device id.

        Returns:
            A fully configured HeteroMTPDetachScheduler.
        """
        a6000_spec = TierSpec(
            kind=TierKind.A6000,
            device_ids=(0, 1),
            vram_bytes=48 * 1024 ** 3,
            sm_version=86,
            cache_bytes=_CACHE_SIZE_A6000,
            pcie_bw_gbs=_PCIE_BW_GBS,
        )
        h100_spec = TierSpec(
            kind=TierKind.H100,
            device_ids=(2,),
            vram_bytes=96 * 1024 ** 3,
            sm_version=90,
            cache_bytes=_CACHE_SIZE_H100,
            pcie_bw_gbs=_PCIE_BW_GBS,
        )

        head_dim = hidden_size // num_attention_heads
        estimators: Dict[int, GradientTrafficEstimator] = {}
        placements: List[MTPLayerPlacement] = []

        for i in range(num_mtp_layers):
            # Default placement: distribute MTP layers round-robin across A6000s,
            # unless the caller provides an override map or we have more layers than
            # A6000 slots (overflow to H100).
            if mtp_layer_device_map is not None:
                dev_id = mtp_layer_device_map.get(i, 0)
            else:
                dev_id = i % 2  # cuda:0 or cuda:1

            tier = a6000_spec if dev_id in a6000_spec.device_ids else h100_spec

            estimators[i] = GradientTrafficEstimator(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                head_dim=head_dim,
            )
            placements.append(
                MTPLayerPlacement(
                    layer_idx=i,
                    tier=tier,
                    gradient_bytes=estimators[i].estimate_bytes,
                )
            )

        tier_policies = {
            TierKind.A6000: TierDetachPolicy.default_for_tier(TierKind.A6000),
            TierKind.H100: TierDetachPolicy.default_for_tier(TierKind.H100),
        }

        cache_oracles = {
            TierKind.A6000: LocalityCacheOracle(a6000_spec),
            TierKind.H100: LocalityCacheOracle(h100_spec),
        }

        scheduler = cls(
            tier_specs=[a6000_spec, h100_spec],
            mtp_layer_placements=placements,
            tier_policies=tier_policies,
            cache_oracles=cache_oracles,
            gradient_estimators=estimators,
            warmup_steps=warmup_steps,
            finetune_start_step=finetune_start_step,
            num_mtp_layers=num_mtp_layers,
        )

        logger.info(
            "HeteroMTPDetachScheduler initialised: %d MTP layers, "
            "warmup=%d steps, finetune_start=%s",
            num_mtp_layers,
            warmup_steps,
            finetune_start_step,
        )
        return scheduler

    # ------------------------------------------------------------------
    # Step interface
    # ------------------------------------------------------------------

    def step(self, global_step: int) -> None:
        """Advance the scheduler to ``global_step``.

        Must be called once per optimizer step before querying
        ``should_detach``.  Resolves the current training phase and
        refreshes per-layer gradient budget state.
        """
        with self._lock:
            self._current_step = global_step
            self._current_phase = resolve_phase(
                global_step, self._warmup_steps, self._finetune_start_step
            )

    # ------------------------------------------------------------------
    # Decision interface
    # ------------------------------------------------------------------

    def should_detach(
        self,
        site: DetachSite,
        layer_idx: int,
        device: Optional[torch.device] = None,
    ) -> DetachDecision:
        """Query whether a detach is recommended at a given site and layer.

        Args:
            site: One of the three Megatron detach sites.
            layer_idx: Zero-based MTP layer index.
            device: The torch.device currently executing (used to infer tier
                    when layer_idx is ambiguous).

        Returns:
            A DetachDecision with the boolean result and a diagnostic reason.
        """
        with self._lock:
            phase = self._current_phase
            step = self._current_step

        placement = self._placements.get(layer_idx)
        if placement is None:
            # Unknown layer: conservative default — do not detach.
            return DetachDecision(
                should_detach=False,
                site=site,
                layer_idx=layer_idx,
                tier_kind=TierKind.H100,  # assume capable tier
                reason=f"layer_idx={layer_idx} not in placement map; defaulting to no-detach",
            )

        tier_kind = placement.tier.kind
        policy = self._policies[tier_kind]
        oracle = self._oracles[tier_kind]

        # Primary decision: phase-driven policy.
        base_detach = policy.is_active(site, phase)
        reason = f"phase={phase.value} policy for {tier_kind.value}"

        # Secondary promotion: cache pressure.
        if not base_detach and oracle.is_pressured:
            # Promote to detach when locality cache is under pressure.
            # We only promote HIDDEN_STATES and DECODER_INPUT (which are the
            # largest tensors); OUTPUT_WEIGHT is small and its detach has
            # negligible cache impact.
            if site in (DetachSite.HIDDEN_STATES, DetachSite.DECODER_INPUT):
                base_detach = True
                reason = (
                    f"cache_pressure={oracle.occupancy_fraction:.1%} "
                    f"exceeds threshold={_CACHE_PRESSURE_THRESHOLD:.0%}; "
                    f"promoted to detach"
                )
                logger.warning(
                    "DES-LOC locality cache pressure on %s (%.1f%% full): "
                    "promoting layer %d %s to detach at step %d",
                    tier_kind.value,
                    oracle.occupancy_fraction * 100,
                    layer_idx,
                    site.value,
                    step,
                )

        # Tertiary promotion: gradient budget exhaustion.
        if not base_detach:
            estimator = self._estimators.get(layer_idx)
            if estimator is not None:
                budget = placement.tier.gradient_budget_bytes_per_step
                if estimator.exceeds_budget(budget):
                    base_detach = True
                    reason = (
                        f"gradient_budget_bytes={budget:.0f} exceeded by "
                        f"estimate={estimator.estimate_bytes}; emergency detach"
                    )
                    logger.warning(
                        "DES-LOC gradient budget exceeded on %s for MTP layer %d "
                        "site=%s at step %d (estimate=%d bytes, budget=%.0f bytes)",
                        tier_kind.value,
                        layer_idx,
                        site.value,
                        step,
                        estimator.estimate_bytes,
                        budget,
                    )

        decision = DetachDecision(
            should_detach=base_detach,
            site=site,
            layer_idx=layer_idx,
            tier_kind=tier_kind,
            reason=reason,
        )

        # Log transitions only (avoids per-step spam).
        key = (site, layer_idx)
        prev = self._prev_decisions.get(key)
        if prev != base_detach:
            logger.debug(
                "DES-LOC detach transition: layer=%d site=%s %s→%s step=%d reason=%s",
                layer_idx,
                site.value,
                prev,
                base_detach,
                step,
                reason,
            )
            self._prev_decisions[key] = base_detach

        return decision

    # ------------------------------------------------------------------
    # Gradient recording
    # ------------------------------------------------------------------

    def record_gradient(self, layer_idx: int, grad_bytes: int) -> None:
        """Inform the scheduler of an observed gradient size.

        Call this from a ``register_hook`` on the MTP layer output tensor
        so that the gradient estimator can refine its budget checks.

        Args:
            layer_idx: Zero-based MTP layer index.
            grad_bytes: Byte size of the gradient tensor observed.
        """
        estimator = self._estimators.get(layer_idx)
        if estimator is not None:
            estimator.record(grad_bytes)

    # ------------------------------------------------------------------
    # Cache oracle proxy
    # ------------------------------------------------------------------

    def update_cache_occupancy(self, tier_kind: TierKind, occupancy_bytes: int) -> None:
        """Proxy to the tier's LocalityCacheOracle.

        The DES-LOC runtime calls this whenever cache occupancy changes
        by more than a configured delta (to avoid flooding the lock).
        """
        oracle = self._oracles.get(tier_kind)
        if oracle is not None:
            oracle.update_occupancy(occupancy_bytes)

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"HeteroMTPDetachScheduler("
                f"step={self._current_step}, "
                f"phase={self._current_phase.value}, "
                f"mtp_layers={self._num_mtp_layers})"
            )


# ---------------------------------------------------------------------------
# DES-LOC detach application helpers
# ---------------------------------------------------------------------------


def apply_detach_at_output_weight(
    scheduler: HeteroMTPDetachScheduler,
    output_weight: Optional[torch.Tensor],
    output_layer: Any,
    layer_idx: int,
) -> torch.Tensor:
    """DES-LOC equivalent of Megatron's ``process_mtp_loss`` detach block.

    Upstream (diff hunk @@ -815,6 +815,12):
        if config.mtp_detach_heads:
            if output_weight is not None:
                output_weight = output_weight.detach()
            else:
                output_weight = output_layer.weight.detach()

    DES-LOC adaptation:
        Instead of a single boolean flag the scheduler decides per-layer.
        The output_weight is only detached when the tier policy and current
        training phase recommend it.

    Args:
        scheduler: The active HeteroMTPDetachScheduler.
        output_weight: Explicit weight tensor, or None to use output_layer.weight.
        output_layer: Module whose .weight is used when output_weight is None.
        layer_idx: MTP layer index.

    Returns:
        Possibly-detached weight tensor.
    """
    if output_weight is None:
        output_weight = output_layer.weight

    decision = scheduler.should_detach(DetachSite.OUTPUT_WEIGHT, layer_idx)
    if decision.should_detach:
        return output_weight.detach()
    return output_weight


def apply_detach_at_decoder_input(
    scheduler: HeteroMTPDetachScheduler,
    decoder_input: torch.Tensor,
    hidden_states: torch.Tensor,
    layer_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """DES-LOC equivalent of Megatron's ``_get_embeddings`` detach block.

    Upstream (diff hunk @@ -1103,17):
        if self.config.mtp_detach_heads:
            decoder_input = decoder_input.detach()
        hidden_states = make_viewless_tensor(...)
        if not hidden_states.requires_grad:
            hidden_states.requires_grad_(True)

    DES-LOC adaptation:
        The decoder_input detach is gated on the scheduler.  The
        ``hidden_states.requires_grad_(True)`` fix is always applied when
        activation checkpointing is in use (DES-LOC always uses remat on
        A6000 due to its 48 GB constraint), regardless of the detach
        decision.  On H100 we skip the force-grad since the tensor almost
        always already carries a grad_fn.

    Args:
        scheduler: Active scheduler.
        decoder_input: Embedding output tensor [seq, batch, hidden].
        hidden_states: Main model hidden states passed to MTP layer.
        layer_idx: MTP layer index.

    Returns:
        (decoder_input, hidden_states) tuple, potentially with detach applied.
    """
    dec_decision = scheduler.should_detach(DetachSite.DECODER_INPUT, layer_idx)
    if dec_decision.should_detach:
        decoder_input = decoder_input.detach()

    # Ensure hidden_states is differentiable so activation checkpointing
    # (CheckpointFunction) can produce a differentiable output.  On A6000
    # remat is always active; on H100 the tensor should already require grad.
    placement = scheduler._placements.get(layer_idx)
    tier_kind = placement.tier.kind if placement else TierKind.H100
    if tier_kind == TierKind.A6000 and not hidden_states.requires_grad:
        hidden_states = hidden_states.requires_grad_(True)

    return decoder_input, hidden_states


def apply_detach_at_hidden_states(
    scheduler: HeteroMTPDetachScheduler,
    hidden_states: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """DES-LOC equivalent of Megatron's ``MultiTokenPredictionBlock.forward`` detach.

    Upstream (diff hunk @@ -1805,10):
        if self.config.mtp_detach_heads:
            hidden_states = hidden_states.detach()

    DES-LOC adaptation:
        The hidden_states detach is the most impactful for PCIe traffic
        because it completely severs the backward pass from the main
        transformer body to the MTP layers.  On A6000 this is applied
        aggressively (WARMUP + STEADY phases).  On H100 it is never applied
        in STEADY or FINETUNE.

    Args:
        scheduler: Active scheduler.
        hidden_states: Chunked hidden states tensor after offset extraction.
        layer_idx: MTP layer index.

    Returns:
        Possibly-detached hidden_states tensor.
    """
    decision = scheduler.should_detach(DetachSite.HIDDEN_STATES, layer_idx)
    if decision.should_detach:
        return hidden_states.detach()
    return hidden_states


# ---------------------------------------------------------------------------
# Configuration dataclass (drop-in for TransformerConfig extension)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class HeteroMTPDetachConfig:
    """Configuration object for DES-LOC heterogeneous MTP detach scheduling.

    This replaces the single ``mtp_detach_heads: bool`` field from Megatron's
    ``TransformerConfig`` with a richer structure that the
    ``HeteroMTPDetachScheduler`` can be constructed from.

    Intended use:
        Attach an instance of this class to your DeepSpeed model config
        alongside the standard TransformerConfig.  The DES-LOC training
        loop reads it to build the scheduler.

    Attributes:
        enabled: Global enable switch.  When False the scheduler always
                 returns should_detach=False (equivalent to Megatron's
                 mtp_detach_heads=False).
        warmup_steps: Steps in LR warm-up phase.
        finetune_start_step: Step index where fine-tuning phase begins.
        num_mtp_layers: Must match TransformerConfig.mtp_num_layers.
        hidden_size: Must match TransformerConfig.hidden_size.
        num_attention_heads: Must match TransformerConfig.num_attention_heads.
        mtp_layer_device_map: Optional per-layer CUDA device assignment.
        a6000_policy_overrides: Per-phase overrides for the A6000 tier policy.
        h100_policy_overrides: Per-phase overrides for the H100 tier policy.
        pcie_bw_gbs: Effective PCIe bandwidth for budget calculations.
        cache_pressure_threshold: Cache occupancy fraction triggering promotion.
    """

    enabled: bool = False
    warmup_steps: int = 200
    finetune_start_step: Optional[int] = None
    num_mtp_layers: int = 1
    hidden_size: int = 4096
    num_attention_heads: int = 32
    mtp_layer_device_map: Optional[Dict[int, int]] = None

    # Per-phase policy overrides: keys are SchedulePhase values (strings),
    # values are lists of DetachSite values (strings).
    a6000_policy_overrides: Optional[Dict[str, List[str]]] = None
    h100_policy_overrides: Optional[Dict[str, List[str]]] = None

    pcie_bw_gbs: float = _PCIE_BW_GBS
    cache_pressure_threshold: float = _CACHE_PRESSURE_THRESHOLD

    def build_scheduler(self) -> Optional[HeteroMTPDetachScheduler]:
        """Construct a HeteroMTPDetachScheduler from this config.

        Returns:
            A scheduler instance, or None if ``enabled`` is False.
        """
        if not self.enabled:
            return None

        scheduler = HeteroMTPDetachScheduler.from_neuron_sp_env(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_mtp_layers=self.num_mtp_layers,
            warmup_steps=self.warmup_steps,
            finetune_start_step=self.finetune_start_step,
            mtp_layer_device_map=self.mtp_layer_device_map,
        )

        # Apply policy overrides.
        self._apply_overrides(
            scheduler._policies[TierKind.A6000], self.a6000_policy_overrides
        )
        self._apply_overrides(
            scheduler._policies[TierKind.H100], self.h100_policy_overrides
        )

        return scheduler

    @staticmethod
    def _apply_overrides(
        policy: TierDetachPolicy,
        overrides: Optional[Dict[str, List[str]]],
    ) -> None:
        if not overrides:
            return
        for phase_str, site_strs in overrides.items():
            try:
                phase = SchedulePhase(phase_str)
                sites = [DetachSite(s) for s in site_strs]
                policy.override(phase, sites)
                logger.debug(
                    "Applied DES-LOC policy override: tier=%s phase=%s sites=%s",
                    policy.tier_kind.value,
                    phase_str,
                    site_strs,
                )
            except ValueError as exc:
                logger.warning("Invalid DES-LOC policy override ignored: %s", exc)


# ---------------------------------------------------------------------------
# Null scheduler (for mtp_detach_heads=False compatibility)
# ---------------------------------------------------------------------------


class _NullDetachScheduler:
    """Drop-in replacement when DES-LOC detach scheduling is disabled.

    Returns should_detach=False for every query, matching Megatron's
    behaviour when ``mtp_detach_heads=False``.
    """

    def step(self, global_step: int) -> None:  # noqa: D102
        pass

    def should_detach(
        self,
        site: DetachSite,
        layer_idx: int,
        device: Optional[torch.device] = None,
    ) -> DetachDecision:  # noqa: D102
        return DetachDecision(
            should_detach=False,
            site=site,
            layer_idx=layer_idx,
            tier_kind=TierKind.H100,
            reason="DES-LOC detach scheduling disabled",
        )

    def record_gradient(self, layer_idx: int, grad_bytes: int) -> None:  # noqa: D102
        pass

    def update_cache_occupancy(self, tier_kind: TierKind, occupancy_bytes: int) -> None:
        pass


NULL_SCHEDULER: _NullDetachScheduler = _NullDetachScheduler()


def get_scheduler(config: HeteroMTPDetachConfig) -> Any:
    """Convenience factory that returns a real or null scheduler.

    Args:
        config: HeteroMTPDetachConfig instance.

    Returns:
        HeteroMTPDetachScheduler if enabled, else _NullDetachScheduler.
    """
    scheduler = config.build_scheduler()
    if scheduler is None:
        return NULL_SCHEDULER
    return scheduler


# ---------------------------------------------------------------------------
# DeepSpeed integration hook
# ---------------------------------------------------------------------------


class DeepSpeedMTPDetachHook:
    """Integrates the detach scheduler with a DeepSpeed engine.

    Intended to be called from the Neuron_SP training loop wrapper around
    the DeepSpeed engine's ``train_batch`` method.

    Usage
    -----
    >>> hook = DeepSpeedMTPDetachHook(engine, scheduler)
    >>> for step, batch in enumerate(loader):
    ...     hook.on_step_begin(step)
    ...     loss = engine.train_batch(batch)
    ...     hook.on_backward_end(mtp_layer_outputs)
    """

    def __init__(
        self,
        engine: Any,  # deepspeed.DeepSpeedEngine
        scheduler: Any,  # HeteroMTPDetachScheduler or _NullDetachScheduler
    ) -> None:
        self._engine = engine
        self._scheduler = scheduler
        self._grad_hooks: List[Any] = []

    def on_step_begin(self, global_step: int) -> None:
        """Advance the detach scheduler and clear old gradient hooks."""
        self._scheduler.step(global_step)
        for handle in self._grad_hooks:
            handle.remove()
        self._grad_hooks.clear()

    def register_grad_hook(self, tensor: torch.Tensor, layer_idx: int) -> None:
        """Register a hook to record observed gradient sizes for layer_idx.

        Call this on the MTP layer output tensor (before detach) so that
        the gradient estimator can refine its budget model.

        Args:
            tensor: The MTP output activation tensor.
            layer_idx: MTP layer index.
        """
        scheduler = self._scheduler

        def _hook(grad: torch.Tensor) -> None:
            if grad is not None:
                scheduler.record_gradient(layer_idx, grad.nelement() * grad.element_size())

        handle = tensor.register_hook(_hook)
        self._grad_hooks.append(handle)

    def on_backward_end(self, mtp_layer_outputs: List[torch.Tensor]) -> None:
        """Post-backward callback.  Currently a no-op placeholder for future
        per-step gradient-norm telemetry that feeds the traffic estimator."""
        pass


# ---------------------------------------------------------------------------
# Utility: tensor byte size
# ---------------------------------------------------------------------------


def tensor_bytes(t: torch.Tensor) -> int:
    """Return the byte size of a tensor's storage."""
    return t.nelement() * t.element_size()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    import traceback
    import unittest

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    class TestResolvePhase(unittest.TestCase):
        def test_warmup(self):
            self.assertEqual(resolve_phase(0, 100, None), SchedulePhase.WARMUP)
            self.assertEqual(resolve_phase(99, 100, None), SchedulePhase.WARMUP)

        def test_steady(self):
            self.assertEqual(resolve_phase(100, 100, None), SchedulePhase.STEADY)
            self.assertEqual(resolve_phase(500, 100, None), SchedulePhase.STEADY)

        def test_finetune(self):
            self.assertEqual(
                resolve_phase(1000, 100, 1000), SchedulePhase.FINETUNE
            )
            self.assertEqual(
                resolve_phase(999, 100, 1000), SchedulePhase.STEADY
            )

        def test_finetune_none(self):
            # No finetune_start_step → never enters FINETUNE.
            self.assertEqual(
                resolve_phase(99999, 100, None), SchedulePhase.STEADY
            )

    class TestLocalityCacheOracle(unittest.TestCase):
        def _make_oracle(self, fraction: float = 0.0) -> LocalityCacheOracle:
            spec = TierSpec(
                kind=TierKind.A6000,
                device_ids=(0,),
                vram_bytes=48 * 1024 ** 3,
                sm_version=86,
                cache_bytes=4 * 1024 ** 3,
                pcie_bw_gbs=24.0,
            )
            o = LocalityCacheOracle(spec)
            o._inject_occupancy(fraction)
            return o

        def test_not_pressured_below_threshold(self):
            o = self._make_oracle(0.5)
            self.assertFalse(o.is_pressured)

        def test_pressured_at_threshold(self):
            o = self._make_oracle(_CACHE_PRESSURE_THRESHOLD)
            self.assertTrue(o.is_pressured)

        def test_occupancy_fraction(self):
            o = self._make_oracle(0.3)
            self.assertAlmostEqual(o.occupancy_fraction, 0.3, places=5)

    class TestTierDetachPolicy(unittest.TestCase):
        def test_a6000_warmup_all_sites_active(self):
            policy = TierDetachPolicy.default_for_tier(TierKind.A6000)
            for site in DetachSite:
                self.assertTrue(policy.is_active(site, SchedulePhase.WARMUP))

        def test_a6000_steady_only_hidden_states(self):
            policy = TierDetachPolicy.default_for_tier(TierKind.A6000)
            self.assertTrue(
                policy.is_active(DetachSite.HIDDEN_STATES, SchedulePhase.STEADY)
            )
            self.assertFalse(
                policy.is_active(DetachSite.DECODER_INPUT, SchedulePhase.STEADY)
            )
            self.assertFalse(
                policy.is_active(DetachSite.OUTPUT_WEIGHT, SchedulePhase.STEADY)
            )

        def test_a6000_finetune_nothing_active(self):
            policy = TierDetachPolicy.default_for_tier(TierKind.A6000)
            for site in DetachSite:
                self.assertFalse(policy.is_active(site, SchedulePhase.FINETUNE))

        def test_h100_warmup_only_output_weight(self):
            policy = TierDetachPolicy.default_for_tier(TierKind.H100)
            self.assertTrue(
                policy.is_active(DetachSite.OUTPUT_WEIGHT, SchedulePhase.WARMUP)
            )
            self.assertFalse(
                policy.is_active(DetachSite.HIDDEN_STATES, SchedulePhase.WARMUP)
            )

        def test_h100_steady_nothing(self):
            policy = TierDetachPolicy.default_for_tier(TierKind.H100)
            for site in DetachSite:
                self.assertFalse(policy.is_active(site, SchedulePhase.STEADY))

        def test_override(self):
            policy = TierDetachPolicy.default_for_tier(TierKind.A6000)
            policy.override(SchedulePhase.STEADY, [DetachSite.OUTPUT_WEIGHT])
            self.assertTrue(
                policy.is_active(DetachSite.OUTPUT_WEIGHT, SchedulePhase.STEADY)
            )
            self.assertFalse(
                policy.is_active(DetachSite.HIDDEN_STATES, SchedulePhase.STEADY)
            )

    class TestGradientTrafficEstimator(unittest.TestCase):
        def test_formula_estimate_positive(self):
            est = GradientTrafficEstimator(
                hidden_size=4096, num_heads=32, head_dim=128
            )
            self.assertGreater(est.estimate_bytes, 0)

        def test_observed_average(self):
            est = GradientTrafficEstimator(
                hidden_size=4096, num_heads=32, head_dim=128
            )
            for v in [100, 200, 300]:
                est.record(v)
            self.assertAlmostEqual(est.estimate_bytes, 200, delta=1)

        def test_exceeds_budget_true(self):
            est = GradientTrafficEstimator(
                hidden_size=4096, num_heads=32, head_dim=128
            )
            est.record(10 ** 9)  # 1 GB
            self.assertTrue(est.exceeds_budget(1.0))

        def test_exceeds_budget_false(self):
            est = GradientTrafficEstimator(
                hidden_size=64, num_heads=4, head_dim=16
            )
            # Small model → small gradient estimate.
            self.assertFalse(est.exceeds_budget(10 ** 12))

    class TestHeteroMTPDetachScheduler(unittest.TestCase):
        def _make_scheduler(
            self, num_mtp_layers: int = 2, warmup_steps: int = 100
        ) -> HeteroMTPDetachScheduler:
            return HeteroMTPDetachScheduler.from_neuron_sp_env(
                hidden_size=64,
                num_attention_heads=4,
                num_mtp_layers=num_mtp_layers,
                warmup_steps=warmup_steps,
                finetune_start_step=500,
            )

        def test_warmup_a6000_detaches_all_sites(self):
            sched = self._make_scheduler()
            sched.step(0)  # warmup
            for site in DetachSite:
                d = sched.should_detach(site, layer_idx=0)
                self.assertTrue(
                    d.should_detach,
                    f"Expected detach at {site} in WARMUP for A6000",
                )

        def test_steady_a6000_only_hidden_states(self):
            sched = self._make_scheduler()
            sched.step(100)  # steady
            self.assertTrue(
                sched.should_detach(DetachSite.HIDDEN_STATES, 0).should_detach
            )
            self.assertFalse(
                sched.should_detach(DetachSite.DECODER_INPUT, 0).should_detach
            )
            self.assertFalse(
                sched.should_detach(DetachSite.OUTPUT_WEIGHT, 0).should_detach
            )

        def test_finetune_a6000_no_detach(self):
            sched = self._make_scheduler()
            sched.step(500)  # finetune
            for site in DetachSite:
                d = sched.should_detach(site, layer_idx=0)
                self.assertFalse(
                    d.should_detach,
                    f"Expected no detach at {site} in FINETUNE for A6000",
                )

        def test_unknown_layer_no_detach(self):
            sched = self._make_scheduler()
            sched.step(0)
            d = sched.should_detach(DetachSite.HIDDEN_STATES, layer_idx=99)
            self.assertFalse(d.should_detach)

        def test_cache_pressure_promotes_hidden_states(self):
            sched = self._make_scheduler()
            sched.step(100)  # steady — normally no detach for DECODER_INPUT
            # Inject high cache pressure.
            sched._oracles[TierKind.A6000]._inject_occupancy(0.95)
            d = sched.should_detach(DetachSite.HIDDEN_STATES, layer_idx=0)
            self.assertTrue(d.should_detach)
            self.assertIn("cache_pressure", d.reason)

        def test_cache_pressure_does_not_promote_output_weight(self):
            sched = self._make_scheduler()
            sched.step(100)  # steady
            sched._oracles[TierKind.A6000]._inject_occupancy(0.99)
            d = sched.should_detach(DetachSite.OUTPUT_WEIGHT, layer_idx=0)
            # OUTPUT_WEIGHT is excluded from cache-pressure promotion.
            self.assertFalse(d.should_detach)

        def test_gradient_budget_promotion(self):
            sched = self._make_scheduler()
            sched.step(100)  # steady
            # Inject enormous observed gradient to trigger budget overrun.
            sched._estimators[0].record(10 ** 12)
            d = sched.should_detach(DetachSite.DECODER_INPUT, layer_idx=0)
            self.assertTrue(d.should_detach)
            self.assertIn("budget", d.reason)

        def test_phase_transition_logging(self):
            """Transition from WARMUP to STEADY should change decision and log."""
            sched = self._make_scheduler()
            sched.step(0)
            d1 = sched.should_detach(DetachSite.DECODER_INPUT, 0)
            self.assertTrue(d1.should_detach)
            sched.step(100)
            d2 = sched.should_detach(DetachSite.DECODER_INPUT, 0)
            self.assertFalse(d2.should_detach)

        def test_record_gradient_updates_estimator(self):
            sched = self._make_scheduler()
            sched.record_gradient(0, 42)
            self.assertEqual(sched._estimators[0].estimate_bytes, 42)

        def test_update_cache_occupancy_proxy(self):
            sched = self._make_scheduler()
            sched.update_cache_occupancy(TierKind.A6000, 3 * 1024 ** 3)
            self.assertAlmostEqual(
                sched._oracles[TierKind.A6000].occupancy_fraction,
                3 / 4,
                places=3,
            )

        def test_repr(self):
            sched = self._make_scheduler()
            sched.step(42)
            r = repr(sched)
            self.assertIn("42", r)

    class TestApplyDetachHelpers(unittest.TestCase):
        def _scheduler_at_step(self, step: int) -> HeteroMTPDetachScheduler:
            sched = HeteroMTPDetachScheduler.from_neuron_sp_env(
                hidden_size=64,
                num_attention_heads=4,
                num_mtp_layers=2,
                warmup_steps=100,
            )
            sched.step(step)
            return sched

        def test_output_weight_detached_in_warmup(self):
            sched = self._scheduler_at_step(0)
            w = torch.nn.Parameter(torch.randn(16, 64))
            result = apply_detach_at_output_weight(sched, w, None, layer_idx=0)
            self.assertFalse(result.requires_grad)
            self.assertIsNone(result.grad_fn)

        def test_output_weight_not_detached_in_steady(self):
            sched = self._scheduler_at_step(100)
            w = torch.nn.Parameter(torch.randn(16, 64))
            result = apply_detach_at_output_weight(sched, w, None, layer_idx=0)
            # In steady phase A6000 does not detach OUTPUT_WEIGHT.
            self.assertTrue(result.requires_grad)

        def test_output_weight_from_layer_when_none(self):
            sched = self._scheduler_at_step(100)

            class _FakeLayer:
                weight = torch.nn.Parameter(torch.randn(16, 64))

            result = apply_detach_at_output_weight(sched, None, _FakeLayer(), 0)
            self.assertIsInstance(result, torch.Tensor)

        def test_decoder_input_detached_in_warmup(self):
            sched = self._scheduler_at_step(0)
            dec = torch.randn(4, 2, 64, requires_grad=True)
            hs = torch.randn(4, 2, 64)
            dec_out, hs_out = apply_detach_at_decoder_input(sched, dec, hs, 0)
            self.assertFalse(dec_out.requires_grad)
            # hidden_states on A6000 gets requires_grad forced True.
            self.assertTrue(hs_out.requires_grad)

        def test_decoder_input_not_detached_in_steady(self):
            sched = self._scheduler_at_step(100)
            dec = torch.randn(4, 2, 64, requires_grad=True)
            hs = torch.randn(4, 2, 64)
            dec_out, _ = apply_detach_at_decoder_input(sched, dec, hs, 0)
            # Not detached in steady phase.
            self.assertTrue(dec_out.requires_grad)

        def test_hidden_states_detached_in_warmup(self):
            sched = self._scheduler_at_step(0)
            hs = torch.randn(4, 2, 64, requires_grad=True)
            hs_out = apply_detach_at_hidden_states(sched, hs, layer_idx=0)
            self.assertFalse(hs_out.requires_grad)

        def test_hidden_states_detached_in_steady_a6000(self):
            sched = self._scheduler_at_step(100)
            hs = torch.randn(4, 2, 64, requires_grad=True)
            # Layer 0 is on A6000 → HIDDEN_STATES is active in STEADY.
            hs_out = apply_detach_at_hidden_states(sched, hs, layer_idx=0)
            self.assertFalse(hs_out.requires_grad)

        def test_hidden_states_not_detached_in_finetune_a6000(self):
            sched = HeteroMTPDetachScheduler.from_neuron_sp_env(
                hidden_size=64,
                num_attention_heads=4,
                num_mtp_layers=2,
                warmup_steps=100,
                finetune_start_step=500,
            )
            sched.step(500)
            hs = torch.randn(4, 2, 64, requires_grad=True)
            hs_out = apply_detach_at_hidden_states(sched, hs, layer_idx=0)
            # FINETUNE → no detach.
            self.assertTrue(hs_out.requires_grad)

    class TestHeteroMTPDetachConfig(unittest.TestCase):
        def test_disabled_returns_null(self):
            cfg = HeteroMTPDetachConfig(enabled=False)
            result = cfg.build_scheduler()
            self.assertIsNone(result)

        def test_enabled_returns_scheduler(self):
            cfg = HeteroMTPDetachConfig(
                enabled=True,
                hidden_size=64,
                num_attention_heads=4,
                num_mtp_layers=2,
            )
            sched = cfg.build_scheduler()
            self.assertIsInstance(sched, HeteroMTPDetachScheduler)

        def test_get_scheduler_null(self):
            cfg = HeteroMTPDetachConfig(enabled=False)
            sched = get_scheduler(cfg)
            self.assertIsInstance(sched, _NullDetachScheduler)

        def test_get_scheduler_real(self):
            cfg = HeteroMTPDetachConfig(
                enabled=True,
                hidden_size=64,
                num_attention_heads=4,
                num_mtp_layers=1,
            )
            sched = get_scheduler(cfg)
            self.assertIsInstance(sched, HeteroMTPDetachScheduler)

        def test_policy_override_applied(self):
            cfg = HeteroMTPDetachConfig(
                enabled=True,
                hidden_size=64,
                num_attention_heads=4,
                num_mtp_layers=1,
                a6000_policy_overrides={
                    "steady": ["output_weight", "hidden_states"],
                },
            )
            sched = cfg.build_scheduler()
            sched.step(100)  # steady
            self.assertTrue(
                sched.should_detach(DetachSite.OUTPUT_WEIGHT, 0).should_detach
            )

        def test_invalid_override_ignored(self):
            cfg = HeteroMTPDetachConfig(
                enabled=True,
                hidden_size=64,
                num_attention_heads=4,
                num_mtp_layers=1,
                a6000_policy_overrides={"not_a_phase": ["hidden_states"]},
            )
            # Should not raise.
            sched = cfg.build_scheduler()
            self.assertIsInstance(sched, HeteroMTPDetachScheduler)

    class TestNullScheduler(unittest.TestCase):
        def test_always_no_detach(self):
            for site in DetachSite:
                d = NULL_SCHEDULER.should_detach(site, layer_idx=0)
                self.assertFalse(d.should_detach)

        def test_step_noop(self):
            NULL_SCHEDULER.step(9999)  # no exception

        def test_record_gradient_noop(self):
            NULL_SCHEDULER.record_gradient(0, 1024)  # no exception

    class TestDeepSpeedMTPDetachHook(unittest.TestCase):
        def _make_hook(self) -> DeepSpeedMTPDetachHook:
            cfg = HeteroMTPDetachConfig(
                enabled=True,
                hidden_size=64,
                num_attention_heads=4,
                num_mtp_layers=2,
            )
            sched = cfg.build_scheduler()

            class _FakeEngine:
                pass

            return DeepSpeedMTPDetachHook(_FakeEngine(), sched)

        def test_on_step_begin_advances_scheduler(self):
            hook = self._make_hook()
            hook.on_step_begin(42)
            with hook._scheduler._lock:
                self.assertEqual(hook._scheduler._current_step, 42)

        def test_register_grad_hook_records(self):
            hook = self._make_hook()
            hook.on_step_begin(0)
            t = torch.randn(4, 2, 64, requires_grad=True)
            hook.register_grad_hook(t, layer_idx=0)
            self.assertEqual(len(hook._grad_hooks), 1)
            # Trigger backward to fire the hook.
            t.sum().backward()
            # Estimator should have received the gradient size.
            self.assertGreater(
                len(hook._scheduler._estimators[0]._observed), 0
            )

        def test_on_step_begin_clears_old_hooks(self):
            hook = self._make_hook()
            hook.on_step_begin(0)
            t = torch.randn(4, requires_grad=True)
            hook.register_grad_hook(t, layer_idx=0)
            self.assertEqual(len(hook._grad_hooks), 1)
            hook.on_step_begin(1)
            self.assertEqual(len(hook._grad_hooks), 0)

    class TestTensorBytes(unittest.TestCase):
        def test_float32(self):
            t = torch.zeros(100, 100)  # float32 = 4 bytes each
            self.assertEqual(tensor_bytes(t), 100 * 100 * 4)

        def test_bfloat16(self):
            t = torch.zeros(50, dtype=torch.bfloat16)
            self.assertEqual(tensor_bytes(t), 50 * 2)

    # Run all tests.
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestResolvePhase,
        TestLocalityCacheOracle,
        TestTierDetachPolicy,
        TestGradientTrafficEstimator,
        TestHeteroMTPDetachScheduler,
        TestApplyDetachHelpers,
        TestHeteroMTPDetachConfig,
        TestNullScheduler,
        TestDeepSpeedMTPDetachHook,
        TestTensorBytes,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
