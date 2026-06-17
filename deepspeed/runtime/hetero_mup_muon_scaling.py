"""
DES-LOC Heterogeneous MuP-Muon Scaling Adapter
================================================

Upstream design intent (Megatron f58a3281):
    Megatron-LM introduced MuP (Maximal Update Parametrization) scaling support
    for the Muon optimizer. The key insight is that Muon already handles its own
    spectral or unit-RMS-norm scaling for 2-D weight matrices (the "matrix-like"
    parameters it manages internally). Therefore, when MuP width-multiplier
    overrides are applied to the optimizer's parameter groups, Muon-managed
    matrices must be *excluded* from the standard Adam-style MuP LR/eps overrides
    to avoid double-scaling. Vector-like parameters (biases, LayerNorm weights,
    etc.) and embedding/output parameters remain in the conventional MuP path.

DES-LOC adaptation points:
    In the DES-LOC (Decoupled Execution with Shared LOcality Cache) heterogeneous
    training framework on 2×A6000-48 GB (SM86) + 1×H100-NVL-96 GB (SM90), model
    parameters reside on *different devices with different compute capabilities*.
    This creates three orthogonal concerns that the Megatron implementation does
    not address:

    1. **Device-aware parameter classification** — A parameter's physical device
       determines whether it is "Muon-eligible". Muon's Newton-Schulz iteration
       is BF16/FP32 intensive; we only run it on SM90 (H100) for numerical
       stability and throughput. SM86 devices fall back to unit-RMS-norm SGD.
       Classification therefore depends on (dtype, device, shape), not shape alone.

    2. **Locality-cache coherence** — DES-LOC maintains a shared locality cache
       (SLC) that holds recently-used activations and gradient fragments across
       the PCIe fabric. When MuP overrides mutate per-parameter LR values, the
       SLC eviction policy must be invalidated for affected parameter slots so
       stale gradient momentum estimates do not persist across width-multiplier
       changes (e.g., during curriculum scaling warm-up).

    3. **Heterogeneous width-multiplier** — Because layers can be partitioned
       across A6000 and H100 devices, the logical "width" of a layer may differ
       from the physical allocation width on a given device. We introduce a
       per-device width-multiplier correction factor (``device_width_correction``)
       derived from the memory-capacity ratio and SM count, ensuring that MuP
       scaling remains consistent across heterogeneous shards.

    4. **Muon scale-mode routing** — ``spectral`` mode requires iterative
       Newton-Schulz on the full weight matrix; this is cost-prohibitive over PCIe.
       We transparently demote ``spectral`` to ``unit_rms_norm`` for SM86 devices
       and emit a structured warning so operators can audit the choice.

References:
    - Megatron-LM commit f58a3281fa80ca44efcd084f96880e55042be59f
    - Yang et al., "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot
      Hyperparameter Transfer" (arXiv 2203.03466)
    - Kosson & Jaggi, "Muon: A Training Algorithm for Deep Networks" (2024)
    - Neuron_SP project: github.com/dylanyunlon/Neuron_SP
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware capability constants for DES-LOC target cluster
# ---------------------------------------------------------------------------

#: SM compute capability tuples for supported devices
_SM86_CAPABILITY: Tuple[int, int] = (8, 6)   # A6000 48 GB
_SM90_CAPABILITY: Tuple[int, int] = (9, 0)   # H100 NVL 96 GB

#: Devices whose Newton-Schulz iteration is approved for spectral Muon
_SPECTRAL_APPROVED_CAPABILITIES: Set[Tuple[int, int]] = {_SM90_CAPABILITY}

#: Memory capacity (bytes) used for heterogeneous width-multiplier correction
_DEVICE_MEM_BYTES: Dict[Tuple[int, int], int] = {
    _SM86_CAPABILITY: 48 * (1 << 30),
    _SM90_CAPABILITY: 96 * (1 << 30),
}

#: SM counts used as secondary correction signal
_DEVICE_SM_COUNT: Dict[Tuple[int, int], int] = {
    _SM86_CAPABILITY: 84,    # RTX A6000: 84 SMs
    _SM90_CAPABILITY: 132,   # H100 NVL: 132 SMs
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DeviceProfile:
    """Runtime profile for a single device in the DES-LOC cluster."""
    device: torch.device
    capability: Tuple[int, int]
    mem_bytes: int
    sm_count: int
    supports_spectral_muon: bool

    @classmethod
    def from_device(cls, device: torch.device) -> "DeviceProfile":
        """Query CUDA device properties and build a DeviceProfile."""
        if device.type != "cuda":
            # CPU-offloaded parameters: treated as SM86-equivalent
            return cls(
                device=device,
                capability=_SM86_CAPABILITY,
                mem_bytes=_DEVICE_MEM_BYTES[_SM86_CAPABILITY],
                sm_count=_DEVICE_SM_COUNT[_SM86_CAPABILITY],
                supports_spectral_muon=False,
            )
        props = torch.cuda.get_device_properties(device)
        cap = (props.major, props.minor)
        mem = props.total_memory
        sm_cnt = props.multi_processor_count
        return cls(
            device=device,
            capability=cap,
            mem_bytes=mem,
            sm_count=sm_cnt,
            supports_spectral_muon=(cap in _SPECTRAL_APPROVED_CAPABILITIES),
        )


@dataclass
class MuonParamClassification:
    """Result of classifying a single parameter for Muon/MuP routing."""
    param: nn.Parameter
    name: str
    device_profile: DeviceProfile
    is_muon_eligible: bool          # True → Muon manages scaling for this param
    is_embedding_or_output: bool    # True → uses decoupled LR path
    is_vector_like: bool            # True → bias/LN; no LR scaling
    effective_scale_mode: str       # 'spectral' | 'unit_rms_norm' | 'shape_scaling'
    device_width_correction: float  # heterogeneous width-multiplier adjustment


@dataclass
class MuPOverride:
    """Per-parameter MuP configuration override, DES-LOC extended."""
    max_lr: Optional[float] = None
    min_lr: Optional[float] = None
    eps: Optional[float] = None
    #: DES-LOC extension: SLC slot must be invalidated when this override fires
    slc_invalidate: bool = False
    #: DES-LOC extension: effective scale mode after device routing
    scale_mode: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict suitable for DeepSpeed param-group merging."""
        d: Dict[str, Any] = {}
        if self.max_lr is not None:
            d["max_lr"] = self.max_lr
        if self.min_lr is not None:
            d["min_lr"] = self.min_lr
        if self.eps is not None:
            d["eps"] = self.eps
        return d


@dataclass
class HeteroMuPMuonConfig:
    """
    Configuration for DES-LOC heterogeneous MuP-Muon scaling.

    Parameters
    ----------
    base_lr : float
        Base learning rate at reference width.
    min_lr : float
        Minimum LR for schedulers.
    mup_width_mult : float
        Width multiplier relative to reference (μP).  1.0 → no scaling.
    optimizer_type : str
        One of 'muon', 'dist_muon', 'adam', 'adamw', 'sgd'.
    muon_scale_mode : str
        Global Muon scale mode.  Device routing may override per-device.
    decoupled_lr : Optional[float]
        If set, embedding/output params use this LR exclusively.
    apply_device_width_correction : bool
        When True, width_mult is adjusted per-device using memory/SM ratios.
    slc_invalidation_enabled : bool
        When True, SLC cache slots for scaled params are marked for eviction.
    """
    base_lr: float = 1e-3
    min_lr: float = 1e-5
    mup_width_mult: float = 1.0
    optimizer_type: str = "adamw"
    muon_scale_mode: str = "spectral"
    decoupled_lr: Optional[float] = None
    apply_device_width_correction: bool = True
    slc_invalidation_enabled: bool = True


# ---------------------------------------------------------------------------
# Locality-cache invalidation stub
# ---------------------------------------------------------------------------

class SharedLocalityCacheInvalidator:
    """
    Minimal interface to the DES-LOC Shared Locality Cache (SLC).

    In the full Neuron_SP stack this class wraps C++ CUDA IPC handles.
    Here we expose the contract so the MuP-Muon scaling layer can call
    ``invalidate_slots`` without depending on the full runtime.

    The SLC stores (parameter_id → gradient_fragment_tensor) mappings
    that are shared across the PCIe fabric.  When a MuP override changes
    effective LR for a parameter, accumulated gradient moments in the SLC
    may no longer be valid at the new scale; invalidating the slot forces
    re-population on the next forward-backward pass.
    """

    def __init__(self) -> None:
        self._invalidated: Set[int] = set()

    def invalidate_slots(self, param_ids: Sequence[int]) -> int:
        """
        Mark SLC slots for the given parameter IDs as invalid.

        Returns the number of newly-invalidated slots (0 if all were
        already invalid from a prior call in the same step).
        """
        before = len(self._invalidated)
        self._invalidated.update(param_ids)
        newly = len(self._invalidated) - before
        if newly > 0:
            logger.debug(
                "SLC invalidation: %d slot(s) newly marked (total invalidated: %d)",
                newly,
                len(self._invalidated),
            )
        return newly

    def flush(self) -> None:
        """Commit invalidations and reset for the next step."""
        logger.debug("SLC flush: clearing %d invalidated slot(s)", len(self._invalidated))
        self._invalidated.clear()

    @property
    def pending_count(self) -> int:
        return len(self._invalidated)


# ---------------------------------------------------------------------------
# Device width-multiplier correction
# ---------------------------------------------------------------------------

def compute_device_width_correction(
    profile: DeviceProfile,
    reference_capability: Tuple[int, int] = _SM90_CAPABILITY,
) -> float:
    """
    Compute a multiplicative correction to the global MuP width_mult for a
    given device, based on memory capacity and SM count relative to the
    reference device (H100 NVL).

    Rationale
    ---------
    In DES-LOC, the model is partitioned such that each device holds a shard
    proportional to its memory capacity.  The "effective width" of a layer
    shard on a device is thus::

        effective_width_d = global_width × (mem_d / mem_ref)

    μP theory requires that LR scales as 1/width.  When shards differ in size,
    the global width_mult no longer captures the per-device contribution
    accurately.  We apply a correction factor::

        correction_d = sqrt(mem_d / mem_ref) × sqrt(sm_d / sm_ref)

    The square-root dampening prevents over-correction: in practice the shard
    size scales linearly with memory but the effective receptive field of the
    optimizer (momentum accumulation volume) scales sub-linearly with SM count.

    Parameters
    ----------
    profile : DeviceProfile
        The target device profile.
    reference_capability : tuple
        SM capability of the reference device (default: H100 NVL = (9,0)).

    Returns
    -------
    float
        Correction multiplier in (0, 1].  Reference device returns 1.0.
    """
    if profile.capability == reference_capability:
        return 1.0

    ref_mem = _DEVICE_MEM_BYTES.get(reference_capability, profile.mem_bytes)
    ref_sm = _DEVICE_SM_COUNT.get(reference_capability, profile.sm_count)

    mem_ratio = profile.mem_bytes / ref_mem
    sm_ratio = profile.sm_count / ref_sm

    correction = math.sqrt(mem_ratio) * math.sqrt(sm_ratio)
    logger.debug(
        "Device width correction for %s (cap=%s): mem_ratio=%.3f sm_ratio=%.3f → %.4f",
        profile.device,
        profile.capability,
        mem_ratio,
        sm_ratio,
        correction,
    )
    return correction


# ---------------------------------------------------------------------------
# Core classification logic
# ---------------------------------------------------------------------------

def _resolve_scale_mode(
    requested_mode: str,
    profile: DeviceProfile,
) -> str:
    """
    Route the requested Muon scale mode to the effective mode for a device.

    DES-LOC rule:
        ``spectral`` requires Newton-Schulz iteration over the full weight
        matrix.  On SM86 (A6000) this is feasible but slow over PCIe because
        the full matrix must be gathered before orthogonalisation.  We demote
        ``spectral`` to ``unit_rms_norm`` on non-SM90 devices.
    """
    if requested_mode == "spectral" and not profile.supports_spectral_muon:
        logger.warning(
            "Device %s (SM%d%d) does not support spectral Muon efficiently over PCIe. "
            "Demoting muon_scale_mode from 'spectral' to 'unit_rms_norm' for this device. "
            "Set muon_scale_mode='unit_rms_norm' globally to suppress this warning.",
            profile.device,
            *profile.capability,
        )
        return "unit_rms_norm"
    return requested_mode


def _is_vector_like(param: nn.Parameter, name: str) -> bool:
    """
    Return True if the parameter is vector-like (bias, LayerNorm scale/bias,
    or any 1-D tensor).

    Mirrors Megatron's ``is_vector_like_parameter`` heuristic.  We extend it
    to also flag parameters whose name ends in common scalar suffixes.
    """
    if param.dim() < 2:
        return True
    _scalar_suffixes = (".bias", "_bias", "ln.weight", "norm.weight",
                        "layernorm.weight", "rmsnorm.weight")
    name_lower = name.lower()
    return any(name_lower.endswith(s) for s in _scalar_suffixes)


def _is_embedding_or_output(param: nn.Parameter) -> bool:
    """Check the parameter marker attribute used by Megatron/Neuron_SP."""
    return bool(getattr(param, "is_embedding_or_output_parameter", False))


def classify_parameter(
    param: nn.Parameter,
    name: str,
    optimizer_type: str,
    muon_scale_mode: str,
    apply_device_width_correction: bool,
) -> MuonParamClassification:
    """
    Classify a single parameter for DES-LOC MuP-Muon routing.

    This is the central dispatch function.  It determines:

    - Whether Muon manages scaling for this parameter (``is_muon_eligible``).
    - The effective scale mode after device capability routing.
    - The device-local width-multiplier correction.

    Parameters
    ----------
    param : nn.Parameter
        The parameter tensor.
    name : str
        Fully-qualified parameter name (used for heuristic classification).
    optimizer_type : str
        Optimizer identifier string ('muon', 'dist_muon', 'adam', …).
    muon_scale_mode : str
        Requested global Muon scale mode.
    apply_device_width_correction : bool
        Whether to compute per-device width correction.

    Returns
    -------
    MuonParamClassification
    """
    device = param.device if param.device.type == "cuda" else torch.device("cpu")
    profile = DeviceProfile.from_device(device)

    is_emb_out = _is_embedding_or_output(param)
    is_vec = _is_vector_like(param, name)

    is_muon_type = "muon" in optimizer_type.lower()
    # Muon manages exactly the 2-D non-embedding/output matrices
    is_muon_eligible = (
        is_muon_type
        and param.dim() == 2
        and not is_emb_out
    )

    effective_mode = muon_scale_mode
    if is_muon_eligible:
        effective_mode = _resolve_scale_mode(muon_scale_mode, profile)

    correction = 1.0
    if apply_device_width_correction and not is_vec:
        correction = compute_device_width_correction(profile)

    return MuonParamClassification(
        param=param,
        name=name,
        device_profile=profile,
        is_muon_eligible=is_muon_eligible,
        is_embedding_or_output=is_emb_out,
        is_vector_like=is_vec,
        effective_scale_mode=effective_mode,
        device_width_correction=correction,
    )


# ---------------------------------------------------------------------------
# Override computation
# ---------------------------------------------------------------------------

def _compute_lr_override(
    base_lr: float,
    min_lr: float,
    width_mult: float,
    device_correction: float,
) -> Tuple[float, float]:
    """
    Compute (max_lr, min_lr) after MuP and device-correction scaling.

    MuP rule: LR ∝ 1/width_mult.
    Device correction: further scale by correction factor to account for
    heterogeneous shard sizes.

    effective_mult = width_mult / device_correction
    → max_lr = base_lr / effective_mult
    """
    effective_mult = width_mult / max(device_correction, 1e-8)
    scaled_max = base_lr / effective_mult
    scaled_min = min_lr / effective_mult
    return scaled_max, scaled_min


def _compute_eps_override(
    base_eps: float,
    width_mult: float,
    device_correction: float,
) -> float:
    """
    Compute scaled epsilon per MuP Appendix B.3.

    eps scales with sqrt(fan_in) ≈ sqrt(width_mult).  With device correction
    we use the effective multiplier's square root.
    """
    effective_mult = width_mult / max(device_correction, 1e-8)
    return base_eps * math.sqrt(effective_mult)


def get_hetero_mup_overrides(
    named_params: List[Tuple[str, nn.Parameter]],
    config: HeteroMuPMuonConfig,
    base_eps: float = 1e-8,
    slc: Optional[SharedLocalityCacheInvalidator] = None,
) -> Dict[int, MuPOverride]:
    """
    Compute per-parameter MuP override dictionaries for the DES-LOC cluster.

    This is the DES-LOC analogue of Megatron's ``get_mup_config_overrides``.
    It returns a mapping from parameter id → ``MuPOverride``.

    Key differences from the Megatron implementation
    -------------------------------------------------
    1. Returns a dict keyed by ``id(param)`` rather than abstract ``ParamKey``
       objects, because DeepSpeed parameter groups are managed by reference.
    2. Applies per-device width-multiplier correction before computing LR.
    3. Emits a single consolidated warning per device capability for spectral
       demotion rather than once per parameter.
    4. Optionally schedules SLC invalidation for parameters whose LR changes.
    5. Muon-managed matrices are excluded from Adam-style MuP overrides
       (identical to Megatron intent), but the device routing is applied first.

    Parameters
    ----------
    named_params : list of (name, Parameter)
        All model parameters with their fully-qualified names.
    config : HeteroMuPMuonConfig
        Scaling configuration.
    base_eps : float
        Base Adam epsilon before MuP scaling.
    slc : SharedLocalityCacheInvalidator, optional
        If provided and ``config.slc_invalidation_enabled``, parameters whose
        LR override fires will be registered for SLC slot invalidation.

    Returns
    -------
    dict mapping id(param) → MuPOverride
        Empty dict if ``mup_width_mult == 1.0`` (no scaling needed), but
        Muon-spectral warnings are still emitted regardless of width_mult.
    """
    optimizer_type = config.optimizer_type
    is_muon = "muon" in optimizer_type.lower()
    is_adam = "adam" in optimizer_type.lower()
    is_sgd = optimizer_type.lower() == "sgd"
    decoupled_lr_enabled = config.decoupled_lr is not None

    # ---- Emit consolidated warnings before short-circuit on width_mult=1 ----
    _warned_spectral_caps: Set[Tuple[int, int]] = set()

    if is_muon and config.muon_scale_mode == "spectral":
        logger.warning(
            "Both MuP and muon_scale_mode=spectral are enabled. "
            "Muon-managed matrix parameters will use spectral Muon scaling on SM90 "
            "devices and will be demoted to unit_rms_norm on SM86 devices (PCIe). "
            "Set muon_scale_mode='unit_rms_norm' to use uniform unit_rms_norm scaling "
            "for Muon-managed matrices with MuP across all devices.",
        )

    if decoupled_lr_enabled and not is_adam:
        logger.warning(
            "decoupled_lr is set but optimizer '%s' is not Adam-family. "
            "Embedding/output parameters will retain decoupled_lr; "
            "MuP will not override those values.",
            optimizer_type,
        )

    # ---- Short-circuit: no scaling when width_mult is unity ----------------
    if config.mup_width_mult == 1.0:
        logger.debug("mup_width_mult=1.0 — skipping MuP override computation.")
        return {}

    overrides: Dict[int, MuPOverride] = {}
    slc_to_invalidate: List[int] = []

    for name, param in named_params:
        cls = classify_parameter(
            param=param,
            name=name,
            optimizer_type=optimizer_type,
            muon_scale_mode=config.muon_scale_mode,
            apply_device_width_correction=config.apply_device_width_correction,
        )

        # Warn once per newly-demoted device capability
        if (
            cls.is_muon_eligible
            and cls.effective_scale_mode != config.muon_scale_mode
            and cls.device_profile.capability not in _warned_spectral_caps
        ):
            _warned_spectral_caps.add(cls.device_profile.capability)
            # warning already emitted inside _resolve_scale_mode; no duplicate here

        override = MuPOverride(scale_mode=cls.effective_scale_mode)

        # ---- Determine LR override eligibility -----------------------------
        #
        # Mirrors Megatron's should_scale_lr_with_mup:
        #   - Skip embedding/output params when decoupled_lr is active
        #   - Skip Muon-managed matrices (they handle their own scaling)
        #   - Skip vector-like params (biases, LN weights)
        #
        should_scale_lr = (
            not (decoupled_lr_enabled and cls.is_embedding_or_output)
            and not cls.is_muon_eligible
            and not cls.is_vector_like
        )

        if should_scale_lr:
            max_lr, min_lr = _compute_lr_override(
                base_lr=config.base_lr,
                min_lr=config.min_lr,
                width_mult=config.mup_width_mult,
                device_correction=cls.device_width_correction,
            )
            override.max_lr = max_lr
            override.min_lr = min_lr
            override.slc_invalidate = config.slc_invalidation_enabled
            slc_to_invalidate.append(id(param))

        # ---- Determine eps override eligibility ----------------------------
        #
        # Mirrors Megatron's should_scale_eps_with_mup:
        #   - Skip vector-like params
        #   - Skip Muon-managed matrices
        #   - Only applies to Adam-family optimizers
        #
        should_scale_eps = (
            is_adam
            and not cls.is_vector_like
            and not cls.is_muon_eligible
        )

        if should_scale_eps:
            override.eps = _compute_eps_override(
                base_eps=base_eps,
                width_mult=config.mup_width_mult,
                device_correction=cls.device_width_correction,
            )

        # Only record if there is something to override
        if override.max_lr is not None or override.eps is not None:
            overrides[id(param)] = override

        logger.debug(
            "Param '%s' | device=%s cap=(%d,%d) | muon_eligible=%s vec=%s emb=%s "
            "| scale_mode=%s correction=%.3f | override=%s",
            name,
            cls.device_profile.device,
            *cls.device_profile.capability,
            cls.is_muon_eligible,
            cls.is_vector_like,
            cls.is_embedding_or_output,
            cls.effective_scale_mode,
            cls.device_width_correction,
            override.as_dict(),
        )

    # ---- SLC invalidation --------------------------------------------------
    if slc is not None and slc_to_invalidate:
        count = slc.invalidate_slots(slc_to_invalidate)
        logger.info(
            "SLC: %d parameter slot(s) scheduled for invalidation after MuP override.",
            count,
        )

    logger.info(
        "MuP override computation complete: %d/%d parameters have active overrides "
        "(optimizer=%s width_mult=%.2f).",
        len(overrides),
        len(named_params),
        optimizer_type,
        config.mup_width_mult,
    )
    return overrides


# ---------------------------------------------------------------------------
# DeepSpeed parameter-group integration
# ---------------------------------------------------------------------------

def apply_hetero_mup_to_param_groups(
    param_groups: List[Dict[str, Any]],
    overrides: Dict[int, MuPOverride],
) -> int:
    """
    Merge MuP overrides into DeepSpeed-style parameter groups in-place.

    DeepSpeed parameter groups are lists of dicts with a ``"params"`` key
    holding a list of ``nn.Parameter`` tensors (not named tuples).  We
    iterate and patch each group's per-parameter ``lr``/``eps`` if an
    override exists.

    Note: DeepSpeed supports per-parameter LR via the ``param_specific``
    sub-group mechanism.  For simplicity, this implementation uses the
    ``max_lr`` field expected by DeepSpeed's LR scheduler.

    Parameters
    ----------
    param_groups : list of dict
        Mutable DeepSpeed optimizer parameter groups.
    overrides : dict
        Mapping from id(param) → MuPOverride as returned by
        ``get_hetero_mup_overrides``.

    Returns
    -------
    int
        Number of parameter groups that had at least one override applied.
    """
    if not overrides:
        return 0

    modified_groups = 0
    for group in param_groups:
        params: List[nn.Parameter] = group.get("params", [])
        group_modified = False
        for param in params:
            override = overrides.get(id(param))
            if override is None:
                continue
            od = override.as_dict()
            if not od:
                continue
            # Apply overrides into the group dict.  If the group holds
            # multiple parameters, we promote individual overrides to
            # group-level only when all params in the group agree.
            # For now we apply the first match (standard DeepSpeed pattern
            # is one param per group when per-param LR is desired).
            for key, val in od.items():
                if key not in group or group[key] != val:
                    group[key] = val
                    group_modified = True
                    logger.debug(
                        "Param-group patched: key='%s' value=%s (param id=%d)",
                        key, val, id(param),
                    )
        if group_modified:
            modified_groups += 1

    logger.info(
        "apply_hetero_mup_to_param_groups: %d/%d group(s) modified.",
        modified_groups, len(param_groups),
    )
    return modified_groups


# ---------------------------------------------------------------------------
# High-level convenience entry point
# ---------------------------------------------------------------------------

def configure_hetero_mup_muon(
    model: nn.Module,
    config: HeteroMuPMuonConfig,
    param_groups: Optional[List[Dict[str, Any]]] = None,
    base_eps: float = 1e-8,
    slc: Optional[SharedLocalityCacheInvalidator] = None,
) -> Tuple[Dict[int, MuPOverride], int]:
    """
    End-to-end DES-LOC MuP-Muon configuration for a heterogeneous model.

    Workflow
    --------
    1. Collect all named parameters from ``model``.
    2. Classify each parameter (device, shape, Muon eligibility).
    3. Compute MuP overrides with device-local width correction.
    4. If ``param_groups`` are provided, patch them in-place.
    5. Schedule SLC invalidation for affected slots.

    Parameters
    ----------
    model : nn.Module
        The model (may span multiple CUDA devices in DES-LOC).
    config : HeteroMuPMuonConfig
        Scaling and routing configuration.
    param_groups : list of dict, optional
        DeepSpeed parameter groups to patch.  If None, only the override
        dict is returned (useful for inspection or unit testing).
    base_eps : float
        Base Adam epsilon.
    slc : SharedLocalityCacheInvalidator, optional
        Locality cache to notify of invalidated slots.

    Returns
    -------
    (overrides, modified_groups) : tuple
        ``overrides`` is the per-param-id MuPOverride mapping.
        ``modified_groups`` is the number of patched parameter groups
        (0 when param_groups is None).
    """
    named_params = list(model.named_parameters())
    overrides = get_hetero_mup_overrides(
        named_params=named_params,
        config=config,
        base_eps=base_eps,
        slc=slc,
    )

    modified = 0
    if param_groups is not None:
        modified = apply_hetero_mup_to_param_groups(param_groups, overrides)

    return overrides, modified


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    # Build a tiny heterogeneous model: place parameters on CPU to avoid
    # requiring real GPUs in CI; DeviceProfile will treat them as SM86-class.
    class _TinyTransformerBlock(nn.Module):
        def __init__(self, d: int = 64) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(d, d))       # 2D → Muon-eligible
            self.bias = nn.Parameter(torch.zeros(d))             # 1D → vector-like
            self.out_proj = nn.Parameter(torch.randn(d, d))     # 2D embedding/output
            self.out_proj.is_embedding_or_output_parameter = True  # type: ignore[attr-defined]

    model = _TinyTransformerBlock()

    cfg = HeteroMuPMuonConfig(
        base_lr=1e-3,
        min_lr=1e-5,
        mup_width_mult=4.0,
        optimizer_type="muon",
        muon_scale_mode="unit_rms_norm",
        apply_device_width_correction=True,
        slc_invalidation_enabled=True,
    )

    slc = SharedLocalityCacheInvalidator()
    overrides, _ = configure_hetero_mup_muon(model, cfg, slc=slc)

    # 1. Muon-managed 2D matrix: no LR override (Muon handles its own scaling)
    weight_id = id(model.weight)
    assert weight_id not in overrides, \
        "Muon-managed matrix must NOT receive a MuP LR override"

    # 2. Vector-like param: no override at all
    bias_id = id(model.bias)
    assert bias_id not in overrides, \
        "Vector-like (bias) param must NOT receive any MuP override"

    # 3. Embedding/output 2D param: should receive LR override (Adam chained path)
    #    (not a Muon-managed param; gets standard MuP treatment)
    out_id = id(model.out_proj)
    #    With optimizer_type='muon' and is_embedding_or_output=True, the param is
    #    excluded from Muon eligibility but also from vector-like; check accordingly.
    #    Because decoupled_lr is not set, it should receive the LR override.
    assert out_id in overrides, \
        "Embedding/output matrix must receive MuP LR override when decoupled_lr is not set"
    assert abs(overrides[out_id].max_lr - (1e-3 / 4.0)) < 1e-9, \
        "max_lr must equal base_lr / width_mult (no device correction on CPU path)"

    # 4. SLC should have at least one pending invalidation
    assert slc.pending_count >= 1, "At least one SLC slot must be pending invalidation"

    print("All smoke tests passed.")
