"""
deepspeed/inference/hetero_mamba_state_dtype.py
================================================

DES-LOC Heterogeneous Mamba State Dtype Management
----------------------------------------------------

Upstream intent (Megatron da47e64):
    Megatron added per-tensor dtype flags for Mamba inference states (conv_states and
    ssm_states), allowing them to be stored at a different precision than the model's
    params_dtype.  The motivation is memory efficiency on large hybrid SSM-Transformer
    models: conv states are typically small with low dynamic range (can use fp16/bf16
    safely), while SSM states benefit from fp32 for numerical stability in long
    sequences.  The patch separates dtype tracking from shape tracking, removes an
    implicit assumption that both state tensors share the model dtype, and propagates
    separate dtype metadata through config → context → mixer.

DES-LOC adaptation (HeteroMambaStateDtype):
    In the DES-LOC framework (Decoupled Execution with Shared LOcality Cache) running on
    2×A6000-48GB (SM86) + 1×H100-NVL-96GB (SM90), memory pressure is asymmetric:

    1. **Device-aware dtype selection** – A6000s have 48 GB VRAM with high bandwidth but
       no BF16 tensor-core advantage on SM86 (BF16 works but is slower than on SM90).
       H100 NVL supports BF16 natively with full tensor-core throughput.  DES-LOC
       selects conv_states dtype per device placement rather than globally.

    2. **Locality-cache pressure** – DES-LOC's Shared LOcality Cache (SLC) prefetches
       Mamba states from CPU DRAM (1.5 TB) to GPU on-device SRAM.  Smaller dtypes
       reduce PCIe bandwidth demand.  DES-LOC tracks dtype *size* separately from shape
       to budget SLC eviction thresholds accurately.

    3. **Decoupled execution** – Mamba layers run on a different device than attention
       layers in DES-LOC's pipeline.  State tensors that cross device boundaries must
       have their dtype negotiated at boundary transition points.  This module provides
       `HeteroMambaStateDtypeConfig` as the single source of truth for those decisions.

    4. **Cast-restore pattern** – The mixer cast (`xBC → conv_state dtype → xBC_dtype`)
       from Megatron is preserved but gated on whether src == dst dtype to avoid
       unnecessary kernel launches on the hot path.

Hardware context:
    - PCIe interconnect only (no NVLink): dtype downcast before PCIe transfer is
      profitable even at 3–4× compute overhead because PCIe BW is the bottleneck.
    - CPU DRAM (1.5 TB): SLC uses pinned memory staging; dtype chosen for CPU-side
      staging must be compatible with numpy/ctypes serialization (fp32 preferred for
      CPU-side checkpointing, bf16 acceptable for transient state).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants – hardware-specific dtype preferences for DES-LOC target cluster
# ---------------------------------------------------------------------------

# SM86 (A6000): bf16 is supported but has no tensor-core throughput advantage
# for Mamba conv ops (which are mostly GEMM-free).  fp16 avoids some precision
# edge-cases on SM86 matmuls inside mamba_mixer.  fp32 is used for SSM states
# on A6000 for numerical safety in long sequences.
_SM86_PREFERRED_CONV_DTYPE = torch.float16
_SM86_PREFERRED_SSM_DTYPE = torch.float32

# SM90 (H100 NVL): bf16 is first-class; tensor cores accelerate bf16 matmuls
# inside selective-scan.  SSM states can be bf16 without stability loss because
# H100's accumulation path is wider.
_SM90_PREFERRED_CONV_DTYPE = torch.bfloat16
_SM90_PREFERRED_SSM_DTYPE = torch.bfloat16

# PCIe transfer budget: when a state tensor crosses PCIe, prefer fp16 to halve
# bandwidth.  fp32 is only tolerated when the tensor stays on-device.
_PCIE_TRANSFER_DTYPE = torch.float16

# SLC (Shared LOcality Cache) CPU-side staging: must be fp32 for stable
# serialization / deserialization from pinned buffers.
_SLC_CPU_STAGING_DTYPE = torch.float32

# Map capability string → (conv_dtype, ssm_dtype)
_SM_CAP_TO_DTYPE: Dict[str, Tuple[torch.dtype, torch.dtype]] = {
    "sm86": (_SM86_PREFERRED_CONV_DTYPE, _SM86_PREFERRED_SSM_DTYPE),
    "sm90": (_SM90_PREFERRED_CONV_DTYPE, _SM90_PREFERRED_SSM_DTYPE),
}

# Dtype sizes in bytes (torch.dtype.itemsize not always available on older builds)
_DTYPE_SIZES: Dict[torch.dtype, int] = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float64: 8,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 4,
    torch.int64: 8,
}


def _dtype_size(dtype: torch.dtype) -> int:
    """Return byte size of *dtype*, with fallback to torch.finfo/iinfo."""
    if dtype in _DTYPE_SIZES:
        return _DTYPE_SIZES[dtype]
    try:
        return dtype.itemsize  # torch >= 2.0
    except AttributeError:
        # Older torch: compute from finfo
        try:
            return torch.finfo(dtype).bits // 8
        except TypeError:
            return torch.iinfo(dtype).bits // 8


def _detect_sm_capability(device: Optional[torch.device] = None) -> str:
    """
    Detect SM capability string for *device* (defaults to current CUDA device).

    Returns one of ``'sm86'``, ``'sm90'``, or ``'unknown'``.

    DES-LOC note: capability detection is used to select optimal Mamba state
    dtypes per device in the heterogeneous cluster without manual configuration.
    """
    if not torch.cuda.is_available():
        return "unknown"
    if device is None:
        idx = torch.cuda.current_device()
    elif isinstance(device, torch.device):
        idx = device.index or 0
    else:
        idx = int(device)

    try:
        major, minor = torch.cuda.get_device_capability(idx)
        cap = f"sm{major}{minor}"
        logger.debug("Device %d SM capability: %s", idx, cap)
        return cap
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not detect SM capability for device %d: %s", idx, exc)
        return "unknown"


# ---------------------------------------------------------------------------
# Core config dataclass
# ---------------------------------------------------------------------------


@dataclass
class HeteroMambaStateDtypeConfig:
    """
    Per-layer dtype configuration for Mamba inference state tensors in DES-LOC.

    Mirrors Megatron's ``MambaInferenceStateConfig`` (da47e64) but adds:

    - Device-aware dtype inference via ``from_device()``
    - Per-tensor dtype *size* bookkeeping for SLC budget calculations
    - PCIe transfer dtype negotiation via ``pcie_transfer_dtype()``
    - CPU staging dtype override for SLC pinned-memory buffers

    Fields
    ------
    conv_states_shape : Tuple[int, ...]
        Shape of a single conv state tensor (no batch/layer dims).
    ssm_states_shape : Tuple[int, ...]
        Shape of a single SSM state tensor (no batch/layer dims).
    conv_states_dtype : torch.dtype
        Dtype for conv state tensors on the compute device.
    ssm_states_dtype : torch.dtype
        Dtype for SSM state tensors on the compute device.
    pcie_dtype : torch.dtype
        Dtype used when streaming states across PCIe (A6000 ↔ H100).
        Defaults to fp16 to reduce PCIe pressure.
    cpu_staging_dtype : torch.dtype
        Dtype used when staging states in CPU DRAM (SLC eviction path).
        Defaults to fp32 for stable serialization.
    num_conv_bytes_per_layer_request : int
        Cached byte count for one (layer, request) conv state slice.
        Used by SLC to compute eviction thresholds.
    num_ssm_bytes_per_layer_request : int
        Cached byte count for one (layer, request) SSM state slice.
    """

    conv_states_shape: Tuple[int, ...]
    ssm_states_shape: Tuple[int, ...]
    conv_states_dtype: torch.dtype
    ssm_states_dtype: torch.dtype
    pcie_dtype: torch.dtype = field(default=_PCIE_TRANSFER_DTYPE)
    cpu_staging_dtype: torch.dtype = field(default=_SLC_CPU_STAGING_DTYPE)

    # Computed in __post_init__; not part of constructor kwargs
    num_conv_bytes_per_layer_request: int = field(init=False)
    num_ssm_bytes_per_layer_request: int = field(init=False)

    def __post_init__(self) -> None:
        self.num_conv_bytes_per_layer_request = (
            math.prod(self.conv_states_shape) * _dtype_size(self.conv_states_dtype)
        )
        self.num_ssm_bytes_per_layer_request = (
            math.prod(self.ssm_states_shape) * _dtype_size(self.ssm_states_dtype)
        )
        logger.debug(
            "HeteroMambaStateDtypeConfig: conv %s %s (%d B/req), ssm %s %s (%d B/req), "
            "pcie_dtype=%s, cpu_staging_dtype=%s",
            self.conv_states_shape,
            self.conv_states_dtype,
            self.num_conv_bytes_per_layer_request,
            self.ssm_states_shape,
            self.ssm_states_dtype,
            self.num_ssm_bytes_per_layer_request,
            self.pcie_dtype,
            self.cpu_staging_dtype,
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_model_dtype(
        cls,
        conv_states_shape: Tuple[int, ...],
        ssm_states_shape: Tuple[int, ...],
        model_dtype: torch.dtype,
        *,
        conv_states_dtype: Optional[torch.dtype] = None,
        ssm_states_dtype: Optional[torch.dtype] = None,
        pcie_dtype: Optional[torch.dtype] = None,
        cpu_staging_dtype: Optional[torch.dtype] = None,
    ) -> "HeteroMambaStateDtypeConfig":
        """
        Construct from model params_dtype with optional per-tensor overrides.

        This factory mirrors Megatron's ``MambaInferenceStateConfig.from_model()``
        (da47e64) in that it falls back to ``model_dtype`` when no explicit dtype
        is provided, but additionally validates that the chosen dtypes are safe for
        DES-LOC's heterogeneous execution path.

        Parameters
        ----------
        conv_states_shape : tuple
            Shape of conv state tensor (per request, per layer).
        ssm_states_shape : tuple
            Shape of SSM state tensor (per request, per layer).
        model_dtype : torch.dtype
            The model's ``params_dtype``; used as fallback for state dtypes.
        conv_states_dtype : optional torch.dtype
            Explicit override for conv state dtype.  Defaults to ``model_dtype``.
        ssm_states_dtype : optional torch.dtype
            Explicit override for SSM state dtype.  Defaults to ``model_dtype``.
        pcie_dtype : optional torch.dtype
            Dtype for PCIe transfers.  Defaults to ``_PCIE_TRANSFER_DTYPE``.
        cpu_staging_dtype : optional torch.dtype
            Dtype for CPU SLC staging.  Defaults to ``_SLC_CPU_STAGING_DTYPE``.
        """
        if conv_states_dtype is None:
            conv_states_dtype = model_dtype
            logger.debug(
                "conv_states_dtype not specified; inheriting model dtype %s", model_dtype
            )
        if ssm_states_dtype is None:
            ssm_states_dtype = model_dtype
            logger.debug(
                "ssm_states_dtype not specified; inheriting model dtype %s", model_dtype
            )

        kwargs: Dict = {}
        if pcie_dtype is not None:
            kwargs["pcie_dtype"] = pcie_dtype
        if cpu_staging_dtype is not None:
            kwargs["cpu_staging_dtype"] = cpu_staging_dtype

        return cls(
            conv_states_shape=conv_states_shape,
            ssm_states_shape=ssm_states_shape,
            conv_states_dtype=conv_states_dtype,
            ssm_states_dtype=ssm_states_dtype,
            **kwargs,
        )

    @classmethod
    def from_device(
        cls,
        conv_states_shape: Tuple[int, ...],
        ssm_states_shape: Tuple[int, ...],
        device: Optional[torch.device] = None,
        *,
        conv_states_dtype: Optional[torch.dtype] = None,
        ssm_states_dtype: Optional[torch.dtype] = None,
    ) -> "HeteroMambaStateDtypeConfig":
        """
        Construct with device-optimal dtypes for DES-LOC heterogeneous cluster.

        Detects SM capability of *device* and picks dtype defaults accordingly:

        - SM86 (A6000 48 GB): conv=fp16, ssm=fp32
        - SM90 (H100 NVL):    conv=bf16, ssm=bf16
        - Unknown:             conv=bf16, ssm=fp32  (conservative)

        Explicit overrides (``conv_states_dtype``, ``ssm_states_dtype``) always
        win over the auto-detected defaults.

        DES-LOC note: this factory is called once per device in
        ``HeteroMambaStateManager.build()``, so each device in the heterogeneous
        pool has its own ``HeteroMambaStateDtypeConfig``.
        """
        cap = _detect_sm_capability(device)
        default_conv, default_ssm = _SM_CAP_TO_DTYPE.get(
            cap, (torch.bfloat16, torch.float32)
        )
        logger.info(
            "Device SM capability=%s → default conv_dtype=%s, ssm_dtype=%s",
            cap,
            default_conv,
            default_ssm,
        )

        final_conv = conv_states_dtype if conv_states_dtype is not None else default_conv
        final_ssm = ssm_states_dtype if ssm_states_dtype is not None else default_ssm

        return cls(
            conv_states_shape=conv_states_shape,
            ssm_states_shape=ssm_states_shape,
            conv_states_dtype=final_conv,
            ssm_states_dtype=final_ssm,
        )

    # ------------------------------------------------------------------
    # Memory accounting helpers (used by SLC eviction logic)
    # ------------------------------------------------------------------

    def total_mamba_bytes_per_request(self, num_mamba_layers: int) -> int:
        """
        Total bytes for all Mamba state tensors for a single inference request.

        DES-LOC's SLC uses this value to decide whether to evict a request's
        Mamba states to CPU DRAM or to keep them in GPU SRAM.

        Parameters
        ----------
        num_mamba_layers : int
            Number of Mamba layers in the hybrid model.
        """
        return (
            self.num_conv_bytes_per_layer_request + self.num_ssm_bytes_per_layer_request
        ) * num_mamba_layers

    def pcie_transfer_bytes_per_request(self, num_mamba_layers: int) -> int:
        """
        Byte cost to stream all Mamba states for one request across PCIe.

        Unlike ``total_mamba_bytes_per_request``, this uses ``pcie_dtype`` sizing
        because states are cast before transfer.  This asymmetry matters for
        SLC prefetch scheduling: the on-device footprint may be fp32 while the
        wire size is fp16.
        """
        pcie_size = _dtype_size(self.pcie_dtype)
        conv_elems = math.prod(self.conv_states_shape)
        ssm_elems = math.prod(self.ssm_states_shape)
        return (conv_elems + ssm_elems) * pcie_size * num_mamba_layers

    def memory_ratio_for_budget(
        self, num_mamba_layers: int, total_gpu_bytes: int
    ) -> float:
        """
        Fraction of GPU memory consumed by Mamba states for a single request.

        Used by the SLC memory manager to replicate Megatron's ``mamba_memory_ratio``
        calculation, extended to account for dtype-aware state sizes.
        """
        state_bytes = self.total_mamba_bytes_per_request(num_mamba_layers)
        if total_gpu_bytes <= 0:
            return 0.0
        return state_bytes / total_gpu_bytes


# ---------------------------------------------------------------------------
# Cast utilities – mirrors Megatron mamba_mixer.py cast-restore pattern
# ---------------------------------------------------------------------------


def cast_for_conv_op(
    tensor: torch.Tensor,
    conv_state: Optional[torch.Tensor],
    target_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.dtype]:
    """
    Cast *tensor* to conv state dtype if needed; return (casted_tensor, original_dtype).

    DES-LOC adaptation of Megatron's causal_conv1d cast pattern (da47e64,
    mamba_mixer.py lines 776–795).  The upstream code unconditionally casts
    ``xBC → conv_state.dtype`` even when they already match.  DES-LOC gates the
    cast to avoid a no-op kernel on the hot inference path.

    Parameters
    ----------
    tensor : torch.Tensor
        Activation tensor ``xBC`` entering causal_conv1d.
    conv_state : optional torch.Tensor
        Current conv state tensor (may be None on prefill).
    target_dtype : torch.dtype
        Dtype from ``HeteroMambaStateDtypeConfig.conv_states_dtype`` (authoritative
        when ``conv_state`` is None).

    Returns
    -------
    casted : torch.Tensor
        Tensor in the conv state dtype.
    original_dtype : torch.dtype
        Dtype of *tensor* before casting (used to restore after op).

    Notes
    -----
    When ``conv_state`` is provided its dtype takes precedence over
    ``target_dtype`` to remain compatible with states loaded from SLC that may
    have been serialized at a different dtype.
    """
    original_dtype = tensor.dtype
    conv_state_dtype = original_dtype if conv_state is None else conv_state.dtype

    # Prefer live conv_state dtype; fall back to config target
    effective_dtype = conv_state_dtype if conv_state is not None else target_dtype

    if tensor.dtype == effective_dtype:
        # No-op path: avoid kernel launch overhead on H100 hot path
        logger.debug(
            "cast_for_conv_op: tensor already in target dtype %s, skipping cast",
            effective_dtype,
        )
        return tensor, original_dtype

    logger.debug(
        "cast_for_conv_op: casting %s → %s (will restore to %s after op)",
        original_dtype,
        effective_dtype,
        original_dtype,
    )
    return tensor.to(effective_dtype), original_dtype


def restore_after_conv_op(
    tensor: torch.Tensor, original_dtype: torch.dtype
) -> torch.Tensor:
    """
    Restore *tensor* to *original_dtype* after a conv op.

    Pair with ``cast_for_conv_op``.  No-op when dtypes already match.
    """
    if tensor.dtype == original_dtype:
        return tensor
    return tensor.to(original_dtype)


def cast_weight_for_conv_op(
    weight: torch.Tensor, target_dtype: torch.dtype
) -> torch.Tensor:
    """
    Cast conv1d weight to *target_dtype* for causal_conv1d_fn/update compatibility.

    DES-LOC note: weight casting is separate from activation casting because
    weight tensors on A6000 may be pinned in fp32 for gradient accumulation
    during online fine-tuning, while inference state ops need fp16/bf16.
    """
    if weight.dtype == target_dtype:
        return weight
    return weight.to(target_dtype)


# ---------------------------------------------------------------------------
# HeteroMambaStateManager – per-device state tensor lifecycle
# ---------------------------------------------------------------------------


class HeteroMambaStateManager:
    """
    Manages Mamba conv and SSM state tensors for DES-LOC's heterogeneous device pool.

    In DES-LOC, Mamba layers may be assigned to different GPUs (e.g., A6000s for
    layers 0-N/2, H100 for layers N/2-N).  Each device has its own optimal dtype
    for state tensors.  This manager:

    1. Allocates per-device state tensor pools at the correct dtype.
    2. Handles PCIe transfers between device pools when a layer is migrated.
    3. Interfaces with the SLC eviction system via ``evict_to_cpu()`` /
       ``restore_from_cpu()``.

    Upstream analogue: Megatron ``DynamicInferenceContext`` (da47e64,
    dynamic_context.py lines 610–625).  The upstream allocates a single global
    ``mamba_conv_states`` and ``mamba_ssm_states`` tensor on the current CUDA
    device with a single dtype.  DES-LOC replaces this with per-device pools and
    dtype-aware allocation.

    Parameters
    ----------
    num_mamba_layers : int
        Number of Mamba layers in the hybrid model.
    max_requests : int
        Maximum concurrent inference requests.
    dtype_config : HeteroMambaStateDtypeConfig
        Dtype and shape configuration for state tensors.
    device : torch.device
        The device this manager is responsible for.
    enable_slc : bool
        Whether to enable Shared LOcality Cache (SLC) CPU offload support.
        When True, allocates pinned CPU tensors for eviction staging.
    """

    def __init__(
        self,
        num_mamba_layers: int,
        max_requests: int,
        dtype_config: HeteroMambaStateDtypeConfig,
        device: torch.device,
        enable_slc: bool = True,
    ) -> None:
        self.num_mamba_layers = num_mamba_layers
        self.max_requests = max_requests
        self.dtype_config = dtype_config
        self.device = device
        self.enable_slc = enable_slc

        # Allocated lazily to avoid CUDA context creation at import time
        self._conv_states: Optional[torch.Tensor] = None
        self._ssm_states: Optional[torch.Tensor] = None

        # CPU pinned staging buffers for SLC eviction (allocated on demand)
        self._cpu_conv_staging: Optional[torch.Tensor] = None
        self._cpu_ssm_staging: Optional[torch.Tensor] = None

        # Tracks which request slots are currently evicted to CPU
        self._evicted_slots: set = set()

        logger.info(
            "HeteroMambaStateManager: device=%s, layers=%d, max_req=%d, "
            "conv_dtype=%s, ssm_dtype=%s, slc=%s",
            device,
            num_mamba_layers,
            max_requests,
            dtype_config.conv_states_dtype,
            dtype_config.ssm_states_dtype,
            enable_slc,
        )

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def allocate(self) -> None:
        """
        Allocate GPU state tensors on ``self.device``.

        Shape: ``(num_mamba_layers, max_requests, *state_shape)``

        DES-LOC note: allocation is deferred to this explicit call (rather than
        ``__init__``) so that the manager can be constructed before the CUDA
        context is ready, e.g. during config parsing in DeepSpeed engine init.
        """
        dc = self.dtype_config
        conv_shape = (self.num_mamba_layers, self.max_requests) + dc.conv_states_shape
        ssm_shape = (self.num_mamba_layers, self.max_requests) + dc.ssm_states_shape

        logger.debug(
            "Allocating Mamba states: conv %s %s, ssm %s %s on %s",
            conv_shape,
            dc.conv_states_dtype,
            ssm_shape,
            dc.ssm_states_dtype,
            self.device,
        )

        self._conv_states = torch.empty(
            conv_shape, dtype=dc.conv_states_dtype, device=self.device
        )
        self._ssm_states = torch.empty(
            ssm_shape, dtype=dc.ssm_states_dtype, device=self.device
        )

        gpu_bytes = (
            self._conv_states.nbytes + self._ssm_states.nbytes
        )
        logger.info(
            "Mamba state allocation complete on %s: %.2f MB total "
            "(conv %.2f MB @%s, ssm %.2f MB @%s)",
            self.device,
            gpu_bytes / 1e6,
            self._conv_states.nbytes / 1e6,
            dc.conv_states_dtype,
            self._ssm_states.nbytes / 1e6,
            dc.ssm_states_dtype,
        )

    @property
    def conv_states(self) -> torch.Tensor:
        """GPU conv state tensor; raises if not yet allocated."""
        if self._conv_states is None:
            raise RuntimeError(
                "HeteroMambaStateManager.allocate() must be called before accessing states"
            )
        return self._conv_states

    @property
    def ssm_states(self) -> torch.Tensor:
        """GPU SSM state tensor; raises if not yet allocated."""
        if self._ssm_states is None:
            raise RuntimeError(
                "HeteroMambaStateManager.allocate() must be called before accessing states"
            )
        return self._ssm_states

    # ------------------------------------------------------------------
    # PCIe transfer helpers (device ↔ device via host)
    # ------------------------------------------------------------------

    def transfer_to_device(
        self,
        dst_manager: "HeteroMambaStateManager",
        layer_idx: int,
        request_indices: torch.Tensor,
    ) -> None:
        """
        Transfer Mamba states for a subset of requests to *dst_manager*'s device.

        Uses PCIe dtype (fp16) to minimise bandwidth: cast → transfer → cast back.

        DES-LOC pipeline migration scenario: when a Mamba layer is dynamically
        re-assigned from A6000 to H100 NVL, its state tensors must follow.
        This method handles the cast-transfer-restore sequence:

            src_dtype → pcie_dtype → dst_dtype

        Parameters
        ----------
        dst_manager : HeteroMambaStateManager
            The destination device's state manager.
        layer_idx : int
            Which Mamba layer's states to transfer.
        request_indices : torch.Tensor
            1-D tensor of request slot indices to transfer.
        """
        pcie_dtype = self.dtype_config.pcie_dtype
        n = request_indices.numel()
        logger.debug(
            "PCIe transfer: layer %d, %d requests, %s → %s (wire dtype %s)",
            layer_idx,
            n,
            self.device,
            dst_manager.device,
            pcie_dtype,
        )

        # Gather src slices
        src_conv = self.conv_states[layer_idx, request_indices]  # (n, *conv_shape)
        src_ssm = self.ssm_states[layer_idx, request_indices]  # (n, *ssm_shape)

        # Cast to PCIe wire dtype before transfer
        wire_conv = src_conv.to(pcie_dtype).cpu()
        wire_ssm = src_ssm.to(pcie_dtype).cpu()

        # Transfer to destination device; restore to dst dtype
        dst_conv = wire_conv.to(
            device=dst_manager.device, dtype=dst_manager.dtype_config.conv_states_dtype
        )
        dst_ssm = wire_ssm.to(
            device=dst_manager.device, dtype=dst_manager.dtype_config.ssm_states_dtype
        )

        # Scatter into dst_manager's state tensors
        dst_manager.conv_states[layer_idx, request_indices] = dst_conv
        dst_manager.ssm_states[layer_idx, request_indices] = dst_ssm

        logger.debug("PCIe transfer complete for layer %d", layer_idx)

    # ------------------------------------------------------------------
    # SLC eviction / restore
    # ------------------------------------------------------------------

    def _ensure_cpu_staging(self) -> None:
        """Lazily allocate pinned CPU staging buffers for SLC eviction."""
        if self._cpu_conv_staging is not None:
            return

        dc = self.dtype_config
        cpu_dtype = dc.cpu_staging_dtype
        conv_shape = (self.num_mamba_layers, self.max_requests) + dc.conv_states_shape
        ssm_shape = (self.num_mamba_layers, self.max_requests) + dc.ssm_states_shape

        logger.debug(
            "Allocating SLC CPU staging buffers: conv %s, ssm %s, dtype=%s",
            conv_shape,
            ssm_shape,
            cpu_dtype,
        )
        self._cpu_conv_staging = torch.empty(
            conv_shape, dtype=cpu_dtype, pin_memory=True
        )
        self._cpu_ssm_staging = torch.empty(
            ssm_shape, dtype=cpu_dtype, pin_memory=True
        )
        logger.info(
            "SLC CPU staging allocated: conv %.2f MB + ssm %.2f MB (pinned, dtype=%s)",
            self._cpu_conv_staging.nbytes / 1e6,
            self._cpu_ssm_staging.nbytes / 1e6,
            cpu_dtype,
        )

    def evict_to_cpu(
        self, request_slot: int, stream: Optional[torch.cuda.Stream] = None
    ) -> None:
        """
        Evict a request's Mamba states from GPU to CPU DRAM (SLC staging).

        DES-LOC SLC eviction: when a request is paused (waiting for KV-cache
        space or during speculative decoding rollback), its Mamba states are
        pushed to 1.5 TB CPU DRAM to free GPU SRAM for active requests.

        States are cast to ``cpu_staging_dtype`` (fp32) for stable round-trip
        serialisation even if the GPU dtype is fp16/bf16.

        Parameters
        ----------
        request_slot : int
            The request slot index to evict.
        stream : optional cuda.Stream
            CUDA stream for async copy.  Uses current stream if None.
        """
        if not self.enable_slc:
            logger.debug("SLC disabled; evict_to_cpu is a no-op")
            return

        if request_slot in self._evicted_slots:
            logger.warning("Request slot %d already evicted; skipping", request_slot)
            return

        self._ensure_cpu_staging()
        cpu_dtype = self.dtype_config.cpu_staging_dtype

        logger.debug(
            "Evicting request slot %d to CPU DRAM (dtype %s)", request_slot, cpu_dtype
        )

        ctx = torch.cuda.stream(stream) if stream is not None else _null_context()
        with ctx:
            # All layers for this request slot
            self._cpu_conv_staging[:, request_slot].copy_(
                self.conv_states[:, request_slot].to(cpu_dtype), non_blocking=True
            )
            self._cpu_ssm_staging[:, request_slot].copy_(
                self.ssm_states[:, request_slot].to(cpu_dtype), non_blocking=True
            )

        self._evicted_slots.add(request_slot)
        logger.debug("Eviction enqueued for request slot %d", request_slot)

    def restore_from_cpu(
        self, request_slot: int, stream: Optional[torch.cuda.Stream] = None
    ) -> None:
        """
        Restore a request's Mamba states from CPU DRAM back to GPU SRAM.

        Paired with ``evict_to_cpu``.  Casts from ``cpu_staging_dtype`` back to
        the GPU conv/ssm dtypes.

        Parameters
        ----------
        request_slot : int
            The request slot index to restore.
        stream : optional cuda.Stream
            CUDA stream for async copy.
        """
        if not self.enable_slc:
            logger.debug("SLC disabled; restore_from_cpu is a no-op")
            return

        if request_slot not in self._evicted_slots:
            logger.warning(
                "Request slot %d not in evicted set; restore may be incorrect",
                request_slot,
            )

        dc = self.dtype_config
        logger.debug(
            "Restoring request slot %d from CPU DRAM (conv→%s, ssm→%s)",
            request_slot,
            dc.conv_states_dtype,
            dc.ssm_states_dtype,
        )

        ctx = torch.cuda.stream(stream) if stream is not None else _null_context()
        with ctx:
            self.conv_states[:, request_slot].copy_(
                self._cpu_conv_staging[:, request_slot]
                .to(dc.conv_states_dtype)
                .to(self.device),
                non_blocking=True,
            )
            self.ssm_states[:, request_slot].copy_(
                self._cpu_ssm_staging[:, request_slot]
                .to(dc.ssm_states_dtype)
                .to(self.device),
                non_blocking=True,
            )

        self._evicted_slots.discard(request_slot)
        logger.debug("Restore enqueued for request slot %d", request_slot)

    # ------------------------------------------------------------------
    # Memory reporting
    # ------------------------------------------------------------------

    def memory_summary(self) -> Dict:
        """Return a dict of memory usage stats for logging/monitoring."""
        evicted_count = len(self._evicted_slots)
        active_count = self.max_requests - evicted_count

        gpu_bytes = 0
        if self._conv_states is not None:
            gpu_bytes += self._conv_states.nbytes + self._ssm_states.nbytes

        cpu_bytes = 0
        if self._cpu_conv_staging is not None:
            cpu_bytes += (
                self._cpu_conv_staging.nbytes + self._cpu_ssm_staging.nbytes
            )

        return {
            "device": str(self.device),
            "gpu_mb": gpu_bytes / 1e6,
            "cpu_staging_mb": cpu_bytes / 1e6,
            "active_requests": active_count,
            "evicted_requests": evicted_count,
            "conv_dtype": str(self.dtype_config.conv_states_dtype),
            "ssm_dtype": str(self.dtype_config.ssm_states_dtype),
        }


# ---------------------------------------------------------------------------
# Argument parsing helpers (mirrors Megatron arguments.py da47e64)
# ---------------------------------------------------------------------------


_DTYPE_MAP: Dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def map_mamba_state_dtype(dtype_str: Optional[str]) -> Optional[torch.dtype]:
    """
    Convert a dtype string ('bf16', 'fp16', 'fp32') to a torch.dtype.

    Returns None if *dtype_str* is None (meaning "inherit from model dtype").

    DES-LOC adds None support on top of Megatron's ``map_dtype`` (arguments.py
    da47e64) so that the device-aware factory path can trigger when no explicit
    dtype is given.
    """
    if dtype_str is None:
        return None
    key = dtype_str.lower().strip()
    if key not in _DTYPE_MAP:
        raise ValueError(
            f"Unknown Mamba state dtype '{dtype_str}'. "
            f"Choose from: {list(_DTYPE_MAP.keys())}"
        )
    return _DTYPE_MAP[key]


def add_mamba_inference_dtype_args(parser) -> None:
    """
    Add DES-LOC Mamba state dtype CLI arguments to *parser*.

    Mirrors Megatron ``_add_inference_args`` (arguments.py da47e64) but adds a
    ``'auto'`` choice that triggers ``HeteroMambaStateDtypeConfig.from_device()``.

    Usage
    -----
    .. code-block:: bash

        deepspeed train.py \\
            --mamba-inference-conv-states-dtype auto \\
            --mamba-inference-ssm-states-dtype fp32
    """
    group = parser.add_argument_group("DES-LOC Mamba inference state dtypes")
    group.add_argument(
        "--mamba-inference-conv-states-dtype",
        type=str,
        choices=["bf16", "fp16", "fp32", "auto"],
        default="auto",
        help=(
            "Dtype for Mamba inference conv state tensors. "
            "'auto' selects the optimal dtype for the current device "
            "(fp16 for SM86/A6000, bf16 for SM90/H100)."
        ),
    )
    group.add_argument(
        "--mamba-inference-ssm-states-dtype",
        type=str,
        choices=["bf16", "fp16", "fp32", "auto"],
        default="auto",
        help=(
            "Dtype for Mamba inference SSM state tensors. "
            "'auto' selects the optimal dtype for the current device "
            "(fp32 for SM86/A6000, bf16 for SM90/H100)."
        ),
    )
    return group


def resolve_mamba_dtype_from_args(
    args,
    conv_states_shape: Tuple[int, ...],
    ssm_states_shape: Tuple[int, ...],
    model_dtype: torch.dtype,
    device: Optional[torch.device] = None,
) -> HeteroMambaStateDtypeConfig:
    """
    Build a ``HeteroMambaStateDtypeConfig`` from parsed CLI args.

    Handles the ``'auto'`` sentinel: when either dtype arg is ``'auto'``,
    ``HeteroMambaStateDtypeConfig.from_device()`` is used for that dimension so
    device-optimal defaults apply.  When both are explicit, uses
    ``from_model_dtype()``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed args with ``mamba_inference_conv_states_dtype`` and
        ``mamba_inference_ssm_states_dtype`` attributes.
    conv_states_shape : tuple
        Conv state shape from model.
    ssm_states_shape : tuple
        SSM state shape from model.
    model_dtype : torch.dtype
        Fallback dtype (model's params_dtype).
    device : optional torch.device
        Target device for auto-detection.
    """
    raw_conv = getattr(args, "mamba_inference_conv_states_dtype", "auto")
    raw_ssm = getattr(args, "mamba_inference_ssm_states_dtype", "auto")

    explicit_conv: Optional[torch.dtype] = None if raw_conv == "auto" else map_mamba_state_dtype(raw_conv)
    explicit_ssm: Optional[torch.dtype] = None if raw_ssm == "auto" else map_mamba_state_dtype(raw_ssm)

    if explicit_conv is None or explicit_ssm is None:
        # At least one dimension is auto → use device-aware factory
        logger.info(
            "resolve_mamba_dtype_from_args: auto mode, detecting device dtype (device=%s)",
            device,
        )
        cfg = HeteroMambaStateDtypeConfig.from_device(
            conv_states_shape=conv_states_shape,
            ssm_states_shape=ssm_states_shape,
            device=device,
            conv_states_dtype=explicit_conv,
            ssm_states_dtype=explicit_ssm,
        )
    else:
        cfg = HeteroMambaStateDtypeConfig.from_model_dtype(
            conv_states_shape=conv_states_shape,
            ssm_states_shape=ssm_states_shape,
            model_dtype=model_dtype,
            conv_states_dtype=explicit_conv,
            ssm_states_dtype=explicit_ssm,
        )

    logger.info(
        "Resolved Mamba state dtypes: conv=%s, ssm=%s",
        cfg.conv_states_dtype,
        cfg.ssm_states_dtype,
    )
    return cfg


# ---------------------------------------------------------------------------
# Null context helper for optional stream context manager
# ---------------------------------------------------------------------------


class _null_context:
    """Trivial no-op context manager used when no CUDA stream is provided."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    conv_shape = (544, 4)
    ssm_shape = (8, 64, 16)

    # 1. from_model_dtype with explicit overrides
    cfg = HeteroMambaStateDtypeConfig.from_model_dtype(
        conv_shape, ssm_shape, model_dtype=torch.bfloat16,
        conv_states_dtype=torch.float16, ssm_states_dtype=torch.float32,
    )
    assert cfg.conv_states_dtype == torch.float16, "conv dtype mismatch"
    assert cfg.ssm_states_dtype == torch.float32, "ssm dtype mismatch"

    # 2. Byte accounting
    expected_conv_bytes = math.prod(conv_shape) * 2  # fp16 = 2 bytes
    assert cfg.num_conv_bytes_per_layer_request == expected_conv_bytes, \
        f"conv bytes: {cfg.num_conv_bytes_per_layer_request} != {expected_conv_bytes}"

    # 3. map_mamba_state_dtype round-trip
    assert map_mamba_state_dtype("bf16") == torch.bfloat16
    assert map_mamba_state_dtype(None) is None

    # 4. cast_for_conv_op no-op when dtypes match
    t = torch.randn(4, 544, dtype=torch.float16)
    casted, orig = cast_for_conv_op(t, None, torch.float16)
    assert casted.data_ptr() == t.data_ptr(), "should be same tensor (no-op cast)"
    assert orig == torch.float16

    # 5. total_mamba_bytes_per_request scales with num_layers
    total = cfg.total_mamba_bytes_per_request(num_mamba_layers=4)
    per_layer = cfg.num_conv_bytes_per_layer_request + cfg.num_ssm_bytes_per_layer_request
    assert total == per_layer * 4, f"total bytes mismatch: {total} != {per_layer * 4}"

    logger.info("All smoke-test assertions passed.")
    sys.exit(0)
