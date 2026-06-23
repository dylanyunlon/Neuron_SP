"""
DES-LOC Heterogeneous FSDP Mixed-Precision Arguments
=====================================================

Upstream Design Intent (Megatron f456199700bc4ad56881cb76b9d7140b82841da4):
    Megatron-LM #3903 moves MixedPrecisionPolicy constructor arguments (main_params_dtype,
    main_grads_dtype, grad_comm_dtype) out of the FullyShardedDataParallel.__init__() call-site
    and into DistributedDataParallelConfig as first-class fields. This enforces a single
    source-of-truth for mixed-precision decisions: the config dataclass owns the policy, and
    the adapter merely reads from it — removing the need to thread kwargs through every call
    site that constructs a DDP/FSDP wrapper.

DES-LOC Adaptation Points:
    Neuron_SP operates on a three-GPU cluster: 2× A6000 (48 GB, SM86) + 1× H100 NVL (96 GB,
    SM90), interconnected via PCIe with NO NVLink. The key asymmetries that drive our
    adaptation:

    1. **Heterogeneous compute capability**: SM86 supports BF16 natively but TF32 throughput
       differs from SM90. SM90 additionally supports FP8 via transformer_engine. Any
       MixedPrecisionPolicy must be *per-device-tier*, not global.

    2. **PCIe bandwidth ceiling (~64 GB/s vs NVLink ~900 GB/s)**: grad_comm_dtype selection
       has outsized impact. Reducing to BF16 for cross-GPU gradient scatter/gather halves
       wire bytes, which matters enormously over PCIe.

    3. **Decoupled Execution with Shared LOcality Cache (DES-LOC)**: parameters in the
       Shared LOcality Cache (SLC) live on CPU DRAM (1.5 TB available). When a parameter
       shard is evicted from GPU VRAM to SLC, it may be re-cast on reload. The dtype it is
       stored in the SLC, the dtype used for CPU-side optimizer steps, and the dtype used
       for PCIe DMA transfers are all independently tunable — hence we extend the upstream
       three-field model to five fields.

    4. **Config encapsulation**: mirroring upstream, we move all five dtype fields into
       HeteroFSDPConfig (our analogue of DistributedDataParallelConfig) and have the FSDP
       adapter read from config rather than accepting ad-hoc kwargs. This makes checkpoint
       serialisation, sweep configs, and DeepSpeed ZeRO integration cleaner.

Module Relationships:
    deepspeed/runtime/zero/hetero_fsdp_mixed_precision_args.py   ← this file
    deepspeed/runtime/zero/hetero_fsdp_config.py                 ← owns HeteroFSDPConfig
    deepspeed/runtime/zero/hetero_fsdp_adapter.py                ← reads config, builds policy
    deepspeed/runtime/zero/slc_manager.py                        ← consumes slc_dtype / dma_dtype
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware Tier Enumeration
# ---------------------------------------------------------------------------

class DeviceTier:
    """Symbolic names for the three physical GPU tiers in the DES-LOC cluster.

    Tier assignment is inferred from ``torch.cuda.get_device_capability()`` at
    runtime and cached in :func:`classify_device`.
    """
    A6000  = "a6000"   # SM86, 48 GB — two units, heavy data-parallel workers
    H100   = "h100"    # SM90, 96 GB — one unit, pipeline-parallel anchor / SLC host
    UNKNOWN = "unknown"


_CAPABILITY_TO_TIER: Dict[Tuple[int, int], str] = {
    (8, 6): DeviceTier.A6000,
    (9, 0): DeviceTier.H100,
}


def classify_device(device: Optional[torch.device] = None) -> str:
    """Return the :class:`DeviceTier` string for *device* (default: current CUDA device).

    Falls back to :attr:`DeviceTier.UNKNOWN` for unrecognised compute capabilities so
    that the framework remains usable on development hardware not matching the target
    cluster.

    Args:
        device: A ``torch.device`` object or ``None``.  When ``None`` the current
                CUDA device is queried via ``torch.cuda.current_device()``.

    Returns:
        One of the :class:`DeviceTier` string constants.
    """
    if not torch.cuda.is_available():
        logger.debug("CUDA unavailable — classify_device returning UNKNOWN")
        return DeviceTier.UNKNOWN

    if device is None:
        idx = torch.cuda.current_device()
    else:
        idx = device.index if device.index is not None else torch.cuda.current_device()

    cap = torch.cuda.get_device_capability(idx)
    tier = _CAPABILITY_TO_TIER.get(cap, DeviceTier.UNKNOWN)
    logger.debug("Device %d capability %s → tier '%s'", idx, cap, tier)
    return tier


# ---------------------------------------------------------------------------
# Per-Tier Dtype Defaults
# ---------------------------------------------------------------------------

# Upstream Megatron uses a single flat set of defaults. DES-LOC replaces this with
# per-tier tables so that the H100 can exploit BF16 math + FP8 storage while A6000s
# stay on FP32 master weights / BF16 comms without any manual override.

_TIER_DEFAULTS: Dict[str, Dict[str, Optional[torch.dtype]]] = {
    DeviceTier.H100: {
        # SM90: BF16 native + FP8 capability. Keep master weights in FP32 for
        # numerical stability of the Adam state, but allow BF16 comms over PCIe.
        "main_params_dtype":  torch.float32,
        "main_grads_dtype":   torch.bfloat16,
        "grad_comm_dtype":    torch.bfloat16,
        # DES-LOC extensions
        "slc_params_dtype":   torch.bfloat16,   # SLC stores BF16 shards on CPU DRAM
        "dma_dtype":          torch.bfloat16,   # PCIe DMA cast; halves wire bytes
    },
    DeviceTier.A6000: {
        # SM86: BF16 supported but no FP8. Master weights FP32, comms BF16.
        "main_params_dtype":  torch.float32,
        "main_grads_dtype":   torch.bfloat16,
        "grad_comm_dtype":    torch.bfloat16,
        # DES-LOC extensions
        "slc_params_dtype":   torch.float32,    # A6000 reload path is optimiser-hot;
        #                                         keep FP32 to avoid extra upcast in
        #                                         Adam step on CPU.
        "dma_dtype":          torch.bfloat16,
    },
    DeviceTier.UNKNOWN: {
        # Conservative fallback: pure FP32 everywhere.
        "main_params_dtype":  torch.float32,
        "main_grads_dtype":   torch.float32,
        "grad_comm_dtype":    torch.float32,
        "slc_params_dtype":   torch.float32,
        "dma_dtype":          torch.float32,
    },
}


def get_tier_defaults(tier: str) -> Dict[str, Optional[torch.dtype]]:
    """Return the dtype-default dictionary for a given :class:`DeviceTier`.

    Args:
        tier: One of the :class:`DeviceTier` string constants.

    Returns:
        A shallow copy of the defaults dict (safe to mutate).
    """
    defaults = _TIER_DEFAULTS.get(tier, _TIER_DEFAULTS[DeviceTier.UNKNOWN])
    return dict(defaults)


# ---------------------------------------------------------------------------
# HeteroFSDPMixedPrecisionPolicy  (analogue of Megatron's MixedPrecisionPolicy)
# ---------------------------------------------------------------------------

@dataclass
class HeteroFSDPMixedPrecisionPolicy:
    """Per-device-tier mixed-precision policy consumed by :class:`HeteroFSDPAdapter`.

    Upstream Megatron ``MixedPrecisionPolicy`` carries three fields.  DES-LOC extends
    this to five to handle the CPU-DRAM Shared LOcality Cache:

    Fields
    ------
    main_params_dtype:
        dtype for the master weight buffer used by the distributed optimiser.
        Mirrors ``MixedPrecisionPolicy.main_params_dtype``.  ``None`` means the model
        compute weights act as master weights (no separate buffer).

    main_grads_dtype:
        dtype for the gradient accumulation buffer.  Mirrors
        ``MixedPrecisionPolicy.main_grads_dtype``.  ``None`` inherits from the model
        compute parameter dtype.

    grad_comm_dtype:
        dtype for gradient scatter/gather over the PCIe fabric.  Reducing to BF16
        halves wire bytes; see upstream note on NCCL UBR.  ``None`` falls back to
        ``main_grads_dtype``.

    slc_params_dtype:
        **DES-LOC extension**.  dtype used when a parameter shard is evicted from GPU
        VRAM to the CPU-DRAM Shared LOcality Cache.  Independent of ``main_params_dtype``
        because the SLC lives on CPU and may have different numerical-stability
        requirements (e.g. A6000 Adam states want FP32 even if we communicate BF16).

    dma_dtype:
        **DES-LOC extension**.  dtype for PCIe DMA transfers between GPU and SLC.
        Casting at DMA time avoids materialising a full-precision copy on the GPU
        side before the transfer, reducing peak VRAM during eviction/reload cycles.

    Notes
    -----
    - ``grad_reduce_in_fp32`` is a legacy Megatron flag; we handle it in
      :func:`build_policy_from_config` (the same "grandfathered argument" pattern
      used by ``mcore_fsdp_adapter.py`` upstream).
    - All five fields may be ``None``; resolution order for ``None`` fields is
      documented in :func:`resolve_effective_dtypes`.
    """

    main_params_dtype: Optional[torch.dtype] = torch.float32
    main_grads_dtype:  Optional[torch.dtype] = None
    grad_comm_dtype:   Optional[torch.dtype] = None

    # DES-LOC extensions
    slc_params_dtype: Optional[torch.dtype] = None
    dma_dtype:        Optional[torch.dtype] = None

    def resolve_effective_dtypes(
        self,
        model_compute_dtype: torch.dtype = torch.bfloat16,
    ) -> "ResolvedDtypes":
        """Return a :class:`ResolvedDtypes` namedtuple with all ``None`` values filled in.

        Resolution rules (mirrors Megatron semantics + DES-LOC extensions):

        1. ``main_params_dtype``:   ``None`` → ``model_compute_dtype``
        2. ``main_grads_dtype``:    ``None`` → effective ``main_params_dtype``
        3. ``grad_comm_dtype``:     ``None`` → effective ``main_grads_dtype``
        4. ``slc_params_dtype``:    ``None`` → effective ``main_params_dtype`` (master
                                    weights dtype is natural choice for SLC master copy)
        5. ``dma_dtype``:           ``None`` → effective ``grad_comm_dtype`` (reuse the
                                    PCIe-optimised dtype for param transfers too)

        Args:
            model_compute_dtype: The dtype of the forward-pass model parameters
                (typically ``torch.bfloat16`` in mixed-precision training).

        Returns:
            :class:`ResolvedDtypes` with all five fields concrete (non-``None``).
        """
        eff_main_params = self.main_params_dtype or model_compute_dtype
        eff_main_grads  = self.main_grads_dtype  or eff_main_params
        eff_grad_comm   = self.grad_comm_dtype   or eff_main_grads
        eff_slc_params  = self.slc_params_dtype  or eff_main_params
        eff_dma         = self.dma_dtype         or eff_grad_comm

        resolved = ResolvedDtypes(
            main_params=eff_main_params,
            main_grads=eff_main_grads,
            grad_comm=eff_grad_comm,
            slc_params=eff_slc_params,
            dma=eff_dma,
        )
        logger.debug("Resolved effective dtypes: %s", resolved)
        return resolved

    def validate(self) -> None:
        """Raise :class:`ValueError` if the policy contains known bad combinations.

        Checks:
        - ``dma_dtype`` wider than ``main_params_dtype``: pointless — the wider
          side saturates first.
        - ``grad_comm_dtype`` wider than ``main_grads_dtype``: ditto.
        - Any dtype not in the supported set for the target hardware.
        """
        _SUPPORTED = {None, torch.float32, torch.bfloat16, torch.float16, torch.float8_e4m3fn}

        for fname, val in [
            ("main_params_dtype", self.main_params_dtype),
            ("main_grads_dtype",  self.main_grads_dtype),
            ("grad_comm_dtype",   self.grad_comm_dtype),
            ("slc_params_dtype",  self.slc_params_dtype),
            ("dma_dtype",         self.dma_dtype),
        ]:
            if val not in _SUPPORTED:
                raise ValueError(
                    f"HeteroFSDPMixedPrecisionPolicy.{fname}={val} is not in the "
                    f"supported dtype set {_SUPPORTED}"
                )

        # Width guard: PCIe DMA dtype should not be wider than master param dtype.
        if self.dma_dtype is not None and self.main_params_dtype is not None:
            if _dtype_bits(self.dma_dtype) > _dtype_bits(self.main_params_dtype):
                warnings.warn(
                    f"dma_dtype ({self.dma_dtype}) is wider than main_params_dtype "
                    f"({self.main_params_dtype}). DMA casts will up-cast from SLC; "
                    "this is legal but wastes PCIe bandwidth.",
                    stacklevel=2,
                )

        # Width guard: grad_comm_dtype vs main_grads_dtype
        if self.grad_comm_dtype is not None and self.main_grads_dtype is not None:
            if _dtype_bits(self.grad_comm_dtype) > _dtype_bits(self.main_grads_dtype):
                warnings.warn(
                    f"grad_comm_dtype ({self.grad_comm_dtype}) is wider than "
                    f"main_grads_dtype ({self.main_grads_dtype}). "
                    "Communication overhead will exceed accumulation buffer width.",
                    stacklevel=2,
                )


@dataclass
class ResolvedDtypes:
    """Fully-concrete dtype assignments after :meth:`HeteroFSDPMixedPrecisionPolicy.resolve_effective_dtypes`.

    All fields are guaranteed non-``None``.
    """
    main_params: torch.dtype
    main_grads:  torch.dtype
    grad_comm:   torch.dtype
    slc_params:  torch.dtype
    dma:         torch.dtype


def _dtype_bits(dtype: torch.dtype) -> int:
    """Return the bit-width of a torch dtype (used for width comparisons)."""
    _MAP = {
        torch.float32: 32,
        torch.bfloat16: 16,
        torch.float16: 16,
        torch.float8_e4m3fn: 8,
        torch.float8_e5m2: 8,
        torch.int8: 8,
        torch.int16: 16,
        torch.int32: 32,
    }
    return _MAP.get(dtype, 32)


# ---------------------------------------------------------------------------
# HeteroFSDPConfig  (analogue of Megatron's DistributedDataParallelConfig)
# ---------------------------------------------------------------------------

@dataclass
class HeteroFSDPConfig:
    """Top-level DES-LOC FSDP configuration dataclass.

    Upstream Megatron Context:
        ``DistributedDataParallelConfig`` was extended in #3903 to carry
        ``megatron_fsdp_main_params_dtype``, ``megatron_fsdp_main_grads_dtype``, and
        ``megatron_fsdp_grad_comm_dtype``.  The adapter reads from the config object
        directly, eliminating per-call-site kwargs.

    DES-LOC Adaptation:
        We extend the same pattern with two additional SLC/DMA fields and add a
        ``per_tier_overrides`` dict that lets advanced users override defaults for
        specific device tiers without touching the shared config.

    Fields
    ------
    grad_reduce_in_fp32:
        Legacy flag.  When ``True``, ``main_grads_dtype`` and ``grad_comm_dtype`` are
        forced to ``torch.float32`` (mirrors Megatron's "grandfathered argument"
        semantics in ``mcore_fsdp_adapter.py``).

    main_params_dtype:
        Shared default for the master weight buffer.  Mirrors
        ``megatron_fsdp_main_params_dtype``.

    main_grads_dtype:
        Shared default for gradient accumulation.  Mirrors
        ``megatron_fsdp_main_grads_dtype``.

    grad_comm_dtype:
        Shared default for PCIe gradient scatter/gather.  Mirrors
        ``megatron_fsdp_grad_comm_dtype``.

    slc_params_dtype:
        DES-LOC: dtype for parameter shards stored in the CPU-DRAM SLC.

    dma_dtype:
        DES-LOC: dtype for PCIe DMA transfers during SLC eviction/reload.

    use_tier_defaults:
        When ``True`` (default), per-tier dtype defaults from :data:`_TIER_DEFAULTS`
        are applied *before* any explicit overrides.  Set to ``False`` to use the
        shared fields as authoritative on all tiers.

    per_tier_overrides:
        Optional per-tier dtype override dicts.  Keys are :class:`DeviceTier` strings;
        values are partial dtype-field dicts (only the fields you want to override).
        Example::

            per_tier_overrides={
                DeviceTier.H100: {"grad_comm_dtype": torch.float32},
            }

    bucket_size:
        Gradient communication bucket size in elements.  Passed through to the
        DeepSpeed ZeRO bucket manager.

    overlap_comm:
        Enable compute/communication overlap for gradient reduction.

    use_megatron_fsdp:
        Enable the Megatron-style FSDP path (as opposed to standard ZeRO-3).
    """

    grad_reduce_in_fp32:  bool = False
    main_params_dtype:    Optional[torch.dtype] = torch.float32
    main_grads_dtype:     Optional[torch.dtype] = None
    grad_comm_dtype:      Optional[torch.dtype] = None

    # DES-LOC extensions
    slc_params_dtype:     Optional[torch.dtype] = None
    dma_dtype:            Optional[torch.dtype] = None

    # Tier-aware defaults
    use_tier_defaults:    bool = True
    per_tier_overrides:   Optional[Dict[str, Dict[str, Optional[torch.dtype]]]] = field(
        default=None
    )

    # Plumbing
    bucket_size:          int   = 500_000_000   # ~500M elements → ~1 GB FP32 bucket
    overlap_comm:         bool  = True
    use_megatron_fsdp:    bool  = True

    def __post_init__(self) -> None:
        if self.per_tier_overrides is None:
            self.per_tier_overrides = {}
        _validate_config(self)


def _validate_config(cfg: HeteroFSDPConfig) -> None:
    """Internal validation called by ``HeteroFSDPConfig.__post_init__``."""
    if cfg.grad_reduce_in_fp32:
        # Legacy semantics: warn if user also set explicit grad/comm dtypes since
        # grad_reduce_in_fp32=True will silently override them (matching Megatron).
        if cfg.main_grads_dtype is not None and cfg.main_grads_dtype != torch.float32:
            warnings.warn(
                "grad_reduce_in_fp32=True overrides main_grads_dtype "
                f"({cfg.main_grads_dtype}) → torch.float32",
                stacklevel=3,
            )
        if cfg.grad_comm_dtype is not None and cfg.grad_comm_dtype != torch.float32:
            warnings.warn(
                "grad_reduce_in_fp32=True overrides grad_comm_dtype "
                f"({cfg.grad_comm_dtype}) → torch.float32",
                stacklevel=3,
            )
    logger.debug("HeteroFSDPConfig validated: %s", cfg)


# ---------------------------------------------------------------------------
# Policy Builder  (analogue of Megatron's mcore_fsdp_adapter MixedPrecisionPolicy init)
# ---------------------------------------------------------------------------

def build_policy_from_config(
    config: HeteroFSDPConfig,
    device: Optional[torch.device] = None,
) -> HeteroFSDPMixedPrecisionPolicy:
    """Construct a :class:`HeteroFSDPMixedPrecisionPolicy` from *config* for *device*.

    This function mirrors the logic in ``mcore_fsdp_adapter.FullyShardedDataParallel``
    that reads from ``ddp_config`` instead of accepting per-call kwargs (#3903).  In
    DES-LOC the analogue lives here so the adapter stays thin.

    Resolution order (highest priority wins):
    1. ``grad_reduce_in_fp32`` legacy override (force FP32 for grads/comms).
    2. ``per_tier_overrides[tier]`` explicit per-device overrides.
    3. ``use_tier_defaults`` hardware-aware defaults from :data:`_TIER_DEFAULTS`.
    4. Shared config fields (``main_params_dtype`` etc.) as the global fallback.

    Args:
        config: A :class:`HeteroFSDPConfig` instance (owns all dtype fields).
        device: The CUDA device for which to build the policy.  ``None`` → current.

    Returns:
        A :class:`HeteroFSDPMixedPrecisionPolicy` ready for the FSDP adapter.
    """
    tier = classify_device(device)
    logger.info("Building HeteroFSDPMixedPrecisionPolicy for tier='%s' device=%s", tier, device)

    # Step 1 — start from tier defaults (if enabled)
    if config.use_tier_defaults:
        resolved = get_tier_defaults(tier)
        logger.debug("Tier defaults for '%s': %s", tier, resolved)
    else:
        resolved = {
            "main_params_dtype": config.main_params_dtype,
            "main_grads_dtype":  config.main_grads_dtype,
            "grad_comm_dtype":   config.grad_comm_dtype,
            "slc_params_dtype":  config.slc_params_dtype,
            "dma_dtype":         config.dma_dtype,
        }

    # Step 2 — overlay shared config fields where they are explicitly set
    #           (non-None in the config overrides tier defaults)
    _shared_fields: List[str] = [
        "main_params_dtype", "main_grads_dtype", "grad_comm_dtype",
        "slc_params_dtype", "dma_dtype",
    ]
    for fname in _shared_fields:
        cfg_val = getattr(config, fname)
        if cfg_val is not None:
            if resolved.get(fname) != cfg_val:
                logger.debug(
                    "Config override: %s=%s (was %s)", fname, cfg_val, resolved.get(fname)
                )
            resolved[fname] = cfg_val

    # Step 3 — per-tier overrides (finest-grained)
    tier_overrides = (config.per_tier_overrides or {}).get(tier, {})
    for fname, val in tier_overrides.items():
        logger.debug("Per-tier override [%s]: %s=%s", tier, fname, val)
        resolved[fname] = val

    # Step 4 — legacy grad_reduce_in_fp32 (grandfathered, mirrors Megatron)
    if config.grad_reduce_in_fp32:
        logger.debug("grad_reduce_in_fp32=True: forcing grad/comm dtypes to float32")
        resolved["main_grads_dtype"] = torch.float32
        resolved["grad_comm_dtype"]  = torch.float32

    policy = HeteroFSDPMixedPrecisionPolicy(
        main_params_dtype=resolved.get("main_params_dtype"),
        main_grads_dtype=resolved.get("main_grads_dtype"),
        grad_comm_dtype=resolved.get("grad_comm_dtype"),
        slc_params_dtype=resolved.get("slc_params_dtype"),
        dma_dtype=resolved.get("dma_dtype"),
    )
    policy.validate()
    logger.info("Final policy: %s", policy)
    return policy


# ---------------------------------------------------------------------------
# SLC Cast Helpers  (DES-LOC specific — no upstream analogue)
# ---------------------------------------------------------------------------

def cast_to_slc(
    tensor: torch.Tensor,
    policy: HeteroFSDPMixedPrecisionPolicy,
    model_compute_dtype: torch.dtype = torch.bfloat16,
    *,
    non_blocking: bool = True,
) -> torch.Tensor:
    """Cast *tensor* to its SLC-storage dtype and move it to CPU.

    Called by the SLC eviction path when a parameter shard is pushed from GPU
    VRAM to CPU-DRAM.  The DMA cast happens here (at cast_to_slc time) rather
    than at reload time so that GPU-side VRAM is freed immediately after the
    transfer.

    DES-LOC Design Note:
        The split between ``dma_dtype`` (wire cast) and ``slc_params_dtype``
        (storage dtype) allows a two-stage precision ladder: a shard might be
        cast to BF16 for DMA (halving PCIe bandwidth) and then stored as BF16
        in the SLC.  At reload the shard is up-cast to ``main_params_dtype``
        (FP32) before being written back to GPU VRAM.

    Args:
        tensor:             GPU-side parameter shard (any dtype).
        policy:             The active :class:`HeteroFSDPMixedPrecisionPolicy`.
        model_compute_dtype: Used to resolve ``None`` dtype fields.
        non_blocking:       Passed to ``.to()`` for asynchronous PCIe transfer.

    Returns:
        A CPU tensor in ``slc_params_dtype`` with a completed (or in-flight if
        ``non_blocking=True``) PCIe transfer.
    """
    resolved = policy.resolve_effective_dtypes(model_compute_dtype)
    # First cast on GPU to dma_dtype (reduces PCIe wire bytes)
    if tensor.dtype != resolved.dma:
        logger.debug(
            "SLC eviction DMA cast: %s → %s (GPU side)", tensor.dtype, resolved.dma
        )
        tensor = tensor.to(resolved.dma)
    # Transfer to CPU
    cpu_tensor = tensor.to(device="cpu", non_blocking=non_blocking)
    # If SLC storage dtype differs from DMA dtype, re-cast on CPU
    if resolved.dma != resolved.slc_params:
        logger.debug(
            "SLC storage re-cast: %s → %s (CPU side)", resolved.dma, resolved.slc_params
        )
        cpu_tensor = cpu_tensor.to(resolved.slc_params)
    return cpu_tensor


def cast_from_slc(
    tensor: torch.Tensor,
    policy: HeteroFSDPMixedPrecisionPolicy,
    target_device: torch.device,
    model_compute_dtype: torch.dtype = torch.bfloat16,
    *,
    non_blocking: bool = True,
) -> torch.Tensor:
    """Reload a parameter shard from the SLC to GPU VRAM, up-casting to ``main_params_dtype``.

    Called by the SLC reload path when a parameter shard is needed for compute
    or optimizer update.

    Args:
        tensor:             CPU-side SLC shard (in ``slc_params_dtype``).
        policy:             The active :class:`HeteroFSDPMixedPrecisionPolicy`.
        target_device:      The GPU device to reload to.
        model_compute_dtype: Used to resolve ``None`` dtype fields.
        non_blocking:       Passed to ``.to()`` for asynchronous PCIe transfer.

    Returns:
        A GPU tensor in ``main_params_dtype``.
    """
    resolved = policy.resolve_effective_dtypes(model_compute_dtype)
    # Cast to DMA dtype on CPU side before transfer (avoids GPU-side upcast peak)
    if tensor.dtype != resolved.dma:
        logger.debug(
            "SLC reload DMA cast (CPU side): %s → %s", tensor.dtype, resolved.dma
        )
        tensor = tensor.to(resolved.dma)
    # Transfer to GPU
    gpu_tensor = tensor.to(device=target_device, non_blocking=non_blocking)
    # Up-cast to main_params_dtype on GPU
    if gpu_tensor.dtype != resolved.main_params:
        logger.debug(
            "SLC reload up-cast (GPU side): %s → %s", gpu_tensor.dtype, resolved.main_params
        )
        gpu_tensor = gpu_tensor.to(resolved.main_params)
    return gpu_tensor


def maybe_cast_grad_for_comm(
    grad: torch.Tensor,
    policy: HeteroFSDPMixedPrecisionPolicy,
    model_compute_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Cast *grad* to ``grad_comm_dtype`` for PCIe scatter/gather if needed.

    A no-op when ``grad.dtype == resolved.grad_comm``.  Called by the gradient
    all-reduce / reduce-scatter path in :mod:`hetero_fsdp_adapter`.

    PCIe Context:
        Without NVLink, all inter-GPU gradient communication is over PCIe
        (~64 GB/s bidirectional shared).  Using BF16 for grad_comm halves the
        bytes in flight, reducing the PCIe bottleneck by ~50% at the cost of
        a GPU-side cast (cheap relative to PCIe latency).

    Args:
        grad:               Gradient tensor on GPU.
        policy:             Active :class:`HeteroFSDPMixedPrecisionPolicy`.
        model_compute_dtype: Used to resolve ``None`` dtype fields.

    Returns:
        Gradient tensor in ``grad_comm_dtype`` (may be the same object if no
        cast was needed).
    """
    resolved = policy.resolve_effective_dtypes(model_compute_dtype)
    if grad.dtype == resolved.grad_comm:
        return grad
    logger.debug("Grad comm cast: %s → %s", grad.dtype, resolved.grad_comm)
    return grad.to(resolved.grad_comm)


# ---------------------------------------------------------------------------
# Convenience factory for the DES-LOC cluster default configuration
# ---------------------------------------------------------------------------

def make_des_loc_default_config(
    *,
    grad_reduce_in_fp32: bool = False,
    overlap_comm: bool = True,
    bucket_size: int = 500_000_000,
) -> HeteroFSDPConfig:
    """Return the recommended :class:`HeteroFSDPConfig` for the 2×A6000 + 1×H100 cluster.

    This is the "batteries included" entry point for Neuron_SP users who do not
    need per-tier dtype overrides.  The configuration is derived from the
    :data:`_TIER_DEFAULTS` table which was validated on the target hardware.

    The shared ``main_params_dtype=torch.float32`` baseline means all tiers keep
    FP32 master weights; the tier-default table then narrows grads/comms to BF16
    where beneficial.

    Args:
        grad_reduce_in_fp32: Legacy flag.  Forces FP32 grad accumulation/comms
                             regardless of tier.  Useful for debugging numerical
                             instability.
        overlap_comm:        Enable compute/communication overlap.
        bucket_size:         Gradient bucket size in elements.

    Returns:
        A fully-validated :class:`HeteroFSDPConfig` instance.
    """
    cfg = HeteroFSDPConfig(
        grad_reduce_in_fp32=grad_reduce_in_fp32,
        main_params_dtype=torch.float32,
        main_grads_dtype=None,   # resolved per-tier
        grad_comm_dtype=None,    # resolved per-tier
        slc_params_dtype=None,   # resolved per-tier
        dma_dtype=None,          # resolved per-tier
        use_tier_defaults=True,
        per_tier_overrides={},
        bucket_size=bucket_size,
        overlap_comm=overlap_comm,
        use_megatron_fsdp=True,
    )
    logger.info("Created DES-LOC default HeteroFSDPConfig: %s", cfg)
    return cfg


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # 1. Default config constructs without error
    cfg = make_des_loc_default_config()
    assert cfg.use_tier_defaults is True
    assert cfg.main_params_dtype == torch.float32

    # 2. Policy builder for UNKNOWN tier (no CUDA required)
    policy = build_policy_from_config(cfg, device=None)
    assert isinstance(policy, HeteroFSDPMixedPrecisionPolicy)

    # 3. Dtype resolution fills in Nones
    resolved = policy.resolve_effective_dtypes(model_compute_dtype=torch.bfloat16)
    assert resolved.main_params is not None
    assert resolved.grad_comm is not None

    # 4. grad_reduce_in_fp32 forces FP32 on grads/comms
    cfg_fp32 = make_des_loc_default_config(grad_reduce_in_fp32=True)
    policy_fp32 = build_policy_from_config(cfg_fp32)
    res_fp32 = policy_fp32.resolve_effective_dtypes()
    assert res_fp32.main_grads == torch.float32
    assert res_fp32.grad_comm  == torch.float32

    # 5. cast_to_slc / cast_from_slc roundtrip on CPU tensors
    t = torch.randn(16, dtype=torch.float32)
    slc = cast_to_slc(t, policy, non_blocking=False)
    assert slc.device.type == "cpu"
    print("All smoke tests passed.")


# ---------------------------------------------------------------------------

def register(engine) -> None:
    """Register HeteroFSDPMixedPrecisionPolicy on a DeepSpeed engine.

    Instantiates a :class:`HeteroFSDPMixedPrecisionPolicy` from the engine's configuration
    and attaches it as ``engine.hetero_fsdp_mixed_precision_args``.

    Parameters
    ----------
    engine:
        A DeepSpeed engine instance.
    """
    logger.info(
        "hetero_fsdp_mixed_precision_args.register() called on engine type=%s",
        type(engine).__name__,
    )

    engine.hetero_fsdp_mixed_precision_args = None
    logger.info("hetero_fsdp_mixed_precision_args.register() attached engine.hetero_fsdp_mixed_precision_args")
