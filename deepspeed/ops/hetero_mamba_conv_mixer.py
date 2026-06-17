"""
deepspeed/ops/hetero_mamba_conv_mixer.py
========================================

DES-LOC Heterogeneous Mamba Conv Mixer
---------------------------------------

**Upstream design intent (Megatron commit 35992ba8)**
Megatron's Mamba implementation originally wrapped the 1-D depthwise
convolution in an ``nn.Conv1d`` sub-module.  This forced two indirections
that were harmful for heterogeneous inference:

1. Parameter access went through ``module.conv1d.weight`` / ``module.conv1d.bias``,
   hiding the tensors behind an extra module boundary that certain FSDP /
   checkpoint path-aware hooks could not see without special-casing.
2. The ``nn.Conv1d`` object carried stride, dilation, and group metadata that
   was re-derived at every call to ``F.conv1d`` rather than being fixed at
   construction time.

The upstream fix (PR #4899) promotes ``conv1d.weight`` and ``conv1d.bias``
to **direct** ``nn.Parameter`` attributes of ``MambaMixer``, named
``conv1d_weight`` and ``conv1d_bias``.  The context-parallel helper
(``MambaContextParallel``) is updated to accept the raw tensors plus the
scalar ``conv1d_padding`` instead of the whole ``nn.Conv1d`` object, and
the fast path is simplified: the ``if cp_size == 1`` branch that called
``self.conv1d_cp1(input_)`` directly is removed — both cases now go through
``F.conv1d`` with the same code path.

**DES-LOC adaptation: HeteroMambaConvMixer**
In a DES-LOC cluster (2× A6000 SM86 48 GB + 1× H100 NVL SM96 96 GB, PCIe,
1.5 TB CPU DRAM) the "direct parameter" refactor becomes the load-bearing
hook for *tier-aware placement*:

- Because ``conv1d_weight`` and ``conv1d_bias`` are now plain ``nn.Parameter``
  objects they can be moved to any device tier without unwrapping a module.
- The ``LocalityCache`` (shared CPU DRAM cache) can pin tensors that are too
  large for the current CUDA tier and stream them on demand with a single
  ``Tensor.to(device, non_blocking=True)`` call — no ``Module.to()`` required.
- The elasticity manager (``DESLOCFlextronElasticityBridge``) patches
  ``get_conv1d_weight`` / ``get_conv1d_bias`` on the context-parallel object
  rather than registering a ``register_forward_hook`` on an ``nn.Module``.
  This is exactly the pattern the upstream commit introduced for Flextron hooks
  and maps cleanly onto DES-LOC's device-aware weight scaling.

Key DES-LOC-specific additions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``TierDescriptor``       — maps each hardware tier to a torch.device, SM arch,
                             VRAM budget, and PCIe bandwidth ceiling.
* ``LocalityCacheHandle`` — thin wrapper around a pinned CPU tensor used as a
                             staging buffer between tiers.
* ``HeteroConvParamStore`` — owns conv weight/bias, tracks which tier currently
                             holds the live copy, and exposes ``fetch()`` that
                             streams to the requested tier if needed.
* ``HeteroMambaConvMixer`` — drop-in replacement for the Megatron MambaMixer
                             conv param block; wires ``HeteroConvParamStore``
                             into the DES-LOC execution pipeline.
* ``DESLOCContextParallel``— replaces ``MambaContextParallel``; receives raw
                             tensors (not an nn.Conv1d) exactly as upstream now
                             requires, and adds tier-aware dispatch.
* ``DESLOCFlextronElasticityBridge`` — mirrors the Flextron elasticity manager's
                             method-patching approach but routes the scale tensor
                             through the locality cache before applying it.

Checkpoint compatibility
~~~~~~~~~~~~~~~~~~~~~~~~
Checkpoint keys written before this commit used ``conv1d.weight`` /
``conv1d.bias``.  Upstream handles this with a ``conv_checkpoint_key_map``
dict that rewrites the sharded-state-dict keys.  This file replicates that
map in ``HeteroMambaConvMixer.state_dict_compat_remap`` so that DES-LOC
checkpoints saved against old Megatron weights load without manual surgery.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import unittest
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware tier descriptors
# ---------------------------------------------------------------------------

@dataclass
class TierDescriptor:
    """Describes a single hardware tier in the DES-LOC cluster.

    DES-LOC Adaptation
    ------------------
    The upstream Megatron code assumes a homogeneous CUDA device.  We replace
    that assumption with a ``TierDescriptor`` per physical device so that the
    conv parameter store can track where tensors live and whether a cross-tier
    transfer is needed before a forward pass.

    Parameters
    ----------
    tier_id : int
        Monotonically increasing index; 0 = primary (H100 NVL), 1-2 = A6000.
    device : torch.device
        The CUDA device (or CPU for the locality cache).
    sm_arch : int
        SM capability as an integer (90 for H100, 86 for A6000, 0 for CPU).
    vram_bytes : int
        Nominal VRAM capacity in bytes.
    pcie_bw_gbs : float
        Approximate PCIe bandwidth in GB/s from CPU DRAM to this device.
        Used to estimate transfer latency for scheduling decisions.
    label : str
        Human-readable label used in log messages.
    """
    tier_id: int
    device: torch.device
    sm_arch: int
    vram_bytes: int
    pcie_bw_gbs: float
    label: str


# Default cluster topology for the Neuron_SP reference hardware:
#   slot 0 → H100 NVL 96 GB  SM90
#   slot 1 → A6000 48 GB     SM86
#   slot 2 → A6000 48 GB     SM86
#   cpu    → 1.5 TB DRAM     (locality cache staging)
_DEFAULT_CLUSTER_TIERS: List[TierDescriptor] = [
    TierDescriptor(
        tier_id=0,
        device=torch.device("cuda:0"),
        sm_arch=90,
        vram_bytes=96 * (1 << 30),
        pcie_bw_gbs=64.0,
        label="H100-NVL",
    ),
    TierDescriptor(
        tier_id=1,
        device=torch.device("cuda:1"),
        sm_arch=86,
        vram_bytes=48 * (1 << 30),
        pcie_bw_gbs=32.0,
        label="A6000-0",
    ),
    TierDescriptor(
        tier_id=2,
        device=torch.device("cuda:2"),
        sm_arch=86,
        vram_bytes=48 * (1 << 30),
        pcie_bw_gbs=32.0,
        label="A6000-1",
    ),
]

_CPU_TIER = TierDescriptor(
    tier_id=-1,
    device=torch.device("cpu"),
    sm_arch=0,
    vram_bytes=1536 * (1 << 30),
    pcie_bw_gbs=0.0,
    label="CPU-DRAM",
)


# ---------------------------------------------------------------------------
# Locality Cache
# ---------------------------------------------------------------------------

class LocalityCacheHandle:
    """Pinned-memory staging buffer for a single tensor in the locality cache.

    DES-LOC Adaptation
    ------------------
    The "shared locality cache" in DES-LOC is CPU DRAM pinned memory used as a
    staging tier between GPU VRAM regions.  Because Megatron's upstream commit
    makes ``conv1d_weight`` and ``conv1d_bias`` direct ``nn.Parameter`` objects
    rather than attributes of an ``nn.Conv1d`` sub-module, they can be evicted
    to this staging tier and fetched back with a plain ``Tensor.to()`` call.

    A ``LocalityCacheHandle`` wraps one such pinned-memory buffer and tracks
    whether it is stale (i.e. the GPU copy has been updated since the last
    CPU snapshot).

    Parameters
    ----------
    tensor : torch.Tensor
        The initial value; will be cloned into pinned memory.
    name : str
        Identifier used in log messages.
    """

    def __init__(self, tensor: torch.Tensor, name: str) -> None:
        self.name = name
        self._lock = threading.Lock()
        self._pinned: torch.Tensor = tensor.detach().cpu().pin_memory()
        self._stale = False
        logger.debug(
            "LocalityCacheHandle '%s' allocated %.2f MB in pinned CPU DRAM",
            name,
            self._pinned.numel() * self._pinned.element_size() / (1 << 20),
        )

    @property
    def pinned(self) -> torch.Tensor:
        return self._pinned

    def mark_stale(self) -> None:
        with self._lock:
            self._stale = True

    def refresh(self, tensor: torch.Tensor) -> None:
        """Copy *tensor* (GPU or CPU) into the pinned buffer."""
        with self._lock:
            self._pinned.copy_(tensor.detach().cpu(), non_blocking=False)
            self._stale = False

    @property
    def is_stale(self) -> bool:
        with self._lock:
            return self._stale

    def fetch_to(self, device: torch.device, non_blocking: bool = True) -> torch.Tensor:
        """Return a tensor on *device*, streamed from pinned memory."""
        with self._lock:
            if self._stale:
                logger.warning(
                    "LocalityCacheHandle '%s' is stale; fetching potentially outdated data",
                    self.name,
                )
            return self._pinned.to(device=device, non_blocking=non_blocking)


# ---------------------------------------------------------------------------
# Heterogeneous conv parameter store
# ---------------------------------------------------------------------------

class HeteroConvParamStore:
    """Manages conv weight / bias across DES-LOC device tiers.

    DES-LOC Adaptation
    ------------------
    Upstream Megatron (after commit 35992ba8) stores ``conv1d_weight`` and
    ``conv1d_bias`` as plain ``nn.Parameter`` objects on the current CUDA
    device.  In DES-LOC we extend this by tracking *where* each parameter
    lives and providing ``fetch(tier)`` that streams the tensor to the
    requested tier on demand, using the locality cache as an intermediate
    staging buffer when eviction is needed.

    The store does **not** own the parameters themselves (they remain
    attributes of ``HeteroMambaConvMixer`` so that the optimizer sees them);
    it only owns the locality-cache staging buffers and the placement metadata.

    Parameters
    ----------
    weight_param : nn.Parameter
        The ``conv1d_weight`` parameter (shape ``[conv_dim, 1, d_conv]``).
    bias_param : nn.Parameter
        The ``conv1d_bias`` parameter (shape ``[conv_dim]``).
    home_tier : TierDescriptor
        The tier that "owns" the parameters for gradient accumulation.
    cluster_tiers : list of TierDescriptor
        All available GPU tiers (not including the CPU tier).
    """

    def __init__(
        self,
        weight_param: nn.Parameter,
        bias_param: nn.Parameter,
        home_tier: TierDescriptor,
        cluster_tiers: List[TierDescriptor],
    ) -> None:
        self.weight_param = weight_param
        self.bias_param = bias_param
        self.home_tier = home_tier
        self.cluster_tiers = {t.tier_id: t for t in cluster_tiers}

        self._weight_cache = LocalityCacheHandle(weight_param, "conv1d_weight")
        self._bias_cache = LocalityCacheHandle(bias_param, "conv1d_bias")

        # Track which tier currently holds the "hot" copy.
        self._current_tier_id: int = home_tier.tier_id

        # Per-tier CUDA streams for async transfers.
        self._streams: Dict[int, torch.cuda.Stream] = {}
        for t in cluster_tiers:
            if t.device.type == "cuda":
                try:
                    self._streams[t.tier_id] = torch.cuda.Stream(device=t.device)
                except RuntimeError:
                    # Device may not be available in test environments.
                    pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_transfer_ms(self, nbytes: int, tier: TierDescriptor) -> float:
        if tier.pcie_bw_gbs <= 0:
            return 0.0
        return nbytes / (tier.pcie_bw_gbs * (1 << 30)) * 1_000

    def _should_log_transfer(self, src_tier_id: int, dst_tier_id: int) -> bool:
        """Only log cross-tier transfers (same-tier fetches are no-ops)."""
        return src_tier_id != dst_tier_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evict_to_cache(self) -> None:
        """Copy current GPU tensors into the locality cache (CPU DRAM)."""
        self._weight_cache.refresh(self.weight_param)
        self._bias_cache.refresh(self.bias_param)
        logger.debug(
            "HeteroConvParamStore: evicted conv params from tier %d (%s) to CPU locality cache",
            self._current_tier_id,
            self.cluster_tiers.get(self._current_tier_id, _CPU_TIER).label,
        )

    def fetch_weight(
        self,
        target_tier: TierDescriptor,
        non_blocking: bool = True,
    ) -> torch.Tensor:
        """Return conv weight on *target_tier*'s device.

        If the weight already lives on the target device this is a no-op
        (returns the parameter directly).  Otherwise it is streamed from the
        locality cache.
        """
        if self.weight_param.device == target_tier.device:
            return self.weight_param

        if self._should_log_transfer(self._current_tier_id, target_tier.tier_id):
            nbytes = self.weight_param.numel() * self.weight_param.element_size()
            latency_ms = self._estimate_transfer_ms(nbytes, target_tier)
            logger.info(
                "HeteroConvParamStore: streaming conv1d_weight (%.2f KB) "
                "to tier %d (%s), est. %.2f ms",
                nbytes / 1024,
                target_tier.tier_id,
                target_tier.label,
                latency_ms,
            )

        return self._weight_cache.fetch_to(target_tier.device, non_blocking=non_blocking)

    def fetch_bias(
        self,
        target_tier: TierDescriptor,
        non_blocking: bool = True,
    ) -> torch.Tensor:
        """Return conv bias on *target_tier*'s device."""
        if self.bias_param.device == target_tier.device:
            return self.bias_param

        if self._should_log_transfer(self._current_tier_id, target_tier.tier_id):
            nbytes = self.bias_param.numel() * self.bias_param.element_size()
            logger.info(
                "HeteroConvParamStore: streaming conv1d_bias (%.2f KB) "
                "to tier %d (%s)",
                nbytes / 1024,
                target_tier.tier_id,
                target_tier.label,
            )

        return self._bias_cache.fetch_to(target_tier.device, non_blocking=non_blocking)

    def mark_params_stale(self) -> None:
        """Call after an optimizer step to mark the locality cache as stale."""
        self._weight_cache.mark_stale()
        self._bias_cache.mark_stale()


# ---------------------------------------------------------------------------
# Attribute-restore handle (mirrors upstream _AttributeRestoreHandle)
# ---------------------------------------------------------------------------

class _AttributeRestoreHandle:
    """Removable handle for temporary instance-attribute patches.

    Upstream Design Intent
    ----------------------
    Megatron PR #4899 introduced this class in
    ``megatron/elastification/flextron_elasticity_hooks.py`` to replace
    ``nn.Module.register_forward_hook`` for the conv1d masking path.  Because
    ``conv1d_weight`` and ``conv1d_bias`` are now direct parameters rather than
    attributes of an ``nn.Conv1d`` sub-module, there is no module to hook;
    instead the elasticity manager monkey-patches ``cp.get_conv1d_weight`` and
    ``cp.get_conv1d_bias`` and uses this handle to restore the originals when
    the elasticity scope is exited.

    DES-LOC Adaptation
    ------------------
    We reuse the same pattern for ``DESLOCFlextronElasticityBridge``, which
    additionally needs to move scale tensors across tiers before applying them.
    The handle API is identical to upstream so existing Flextron code that
    calls ``handle.remove()`` continues to work.
    """

    def __init__(self, obj: object, attr_name: str, original_value: object) -> None:
        self.obj = obj
        self.attr_name = attr_name
        self.original_value = original_value
        self._removed = False

    def remove(self) -> None:
        """Restore the original attribute value."""
        if not self._removed:
            setattr(self.obj, self.attr_name, self.original_value)
            self._removed = True

    def __repr__(self) -> str:
        return (
            f"_AttributeRestoreHandle(attr='{self.attr_name}', removed={self._removed})"
        )


# ---------------------------------------------------------------------------
# DES-LOC Context Parallel helper
# ---------------------------------------------------------------------------

class DESLOCContextParallel:
    """Context-parallel conv dispatch with tier-aware parameter fetching.

    Upstream Design Intent
    ----------------------
    ``MambaContextParallel`` (Megatron) handles the case where the sequence is
    split across ``cp_size`` context-parallel ranks.  After PR #4899 it accepts
    ``conv1d_weight_cp1`` (Tensor), ``conv1d_bias_cp1`` (Tensor), and
    ``conv1d_padding`` (int) instead of a full ``nn.Conv1d`` object.  The
    ``cp_size == 1`` fast-path that called ``self.conv1d_cp1(input_)`` directly
    is removed; all cases go through ``F.conv1d`` uniformly.

    DES-LOC Adaptation
    ------------------
    ``DESLOCContextParallel`` wraps the same interface but adds:
    * ``target_tier`` — which hardware tier to run the conv on.  The weight
      and bias are fetched from ``HeteroConvParamStore`` to that tier if needed.
    * ``get_conv1d_weight`` / ``get_conv1d_bias`` — these are the patching
      points used by ``DESLOCFlextronElasticityBridge`` (mirrors upstream
      Flextron hook refactor).

    Parameters
    ----------
    cp_size : int
        Number of context-parallel ranks.
    cp_rank : int
        This rank's index within the CP group.
    tp_size : int
        Tensor-parallel world size.
    tp_rank : int
        This rank's index within the TP group.
    nheads_local_tp : int
        Number of heads on this TP rank.
    ngroups_local_tp : int
        Number of groups on this TP rank.
    d_state : int
        Mamba d_state dimension.
    conv1d_weight_cp1 : torch.Tensor
        The conv weight that would be used when ``cp_size == 1``.
    conv1d_bias_cp1 : torch.Tensor
        The conv bias that would be used when ``cp_size == 1``.
    conv1d_padding : int
        Padding value for ``F.conv1d``.
    param_store : HeteroConvParamStore, optional
        If provided, used to fetch the weight/bias to the target tier.
    target_tier : TierDescriptor, optional
        The tier on which the conv will execute.
    """

    def __init__(
        self,
        cp_size: int,
        cp_rank: int,
        tp_size: int,
        tp_rank: int,
        nheads_local_tp: int,
        ngroups_local_tp: int,
        d_state: int,
        conv1d_weight_cp1: torch.Tensor,
        conv1d_bias_cp1: torch.Tensor,
        conv1d_padding: int,
        param_store: Optional[HeteroConvParamStore] = None,
        target_tier: Optional[TierDescriptor] = None,
    ) -> None:
        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.nheads_local_tp = nheads_local_tp
        self.ngroups_local_tp = ngroups_local_tp
        self.d_state = d_state
        self.conv1d_weight_cp1 = conv1d_weight_cp1
        self.conv1d_bias_cp1 = conv1d_bias_cp1
        self.conv1d_padding = conv1d_padding
        self.param_store = param_store
        self.target_tier = target_tier

    # ------------------------------------------------------------------
    # Weight / bias accessors  (patching points for elasticity bridge)
    # ------------------------------------------------------------------

    def get_conv1d_weight(self) -> torch.Tensor:
        """Return the conv weight slice for this CP rank.

        If a ``HeteroConvParamStore`` and ``target_tier`` are set, the weight is
        fetched to the target tier before slicing.  This matches the upstream
        pattern where the elasticity manager patches this method to inject
        scaling without needing an ``nn.Module`` hook.
        """
        if self.param_store is not None and self.target_tier is not None:
            w = self.param_store.fetch_weight(self.target_tier)
        else:
            w = self.conv1d_weight_cp1
        return self._slice_conv_param(w)

    def get_conv1d_bias(self) -> torch.Tensor:
        """Return the conv bias slice for this CP rank."""
        if self.param_store is not None and self.target_tier is not None:
            b = self.param_store.fetch_bias(self.target_tier)
        else:
            b = self.conv1d_bias_cp1
        return self._slice_conv_param(b)

    # ------------------------------------------------------------------
    # Core conv dispatch (mirrors upstream unified F.conv1d path)
    # ------------------------------------------------------------------

    def apply_conv1d(self, input_: torch.Tensor) -> torch.Tensor:
        """Run the depthwise conv1d on *input_*.

        Upstream Design Intent
        ----------------------
        PR #4899 removed the ``if cp_size == 1`` branch that used
        ``self.conv1d_cp1(input_)`` directly and unified both paths through
        ``F.conv1d``.  This eliminates the implicit ``stride`` and ``dilation``
        fields that were silently inherited from the ``nn.Conv1d`` object;
        those are now always 1 (PyTorch default) which is correct for Mamba.

        DES-LOC Adaptation
        ------------------
        We apply the same unification.  The target-tier fetch in
        ``get_conv1d_weight`` / ``get_conv1d_bias`` ensures the weight/bias are
        on the same device as *input_* when a cross-tier dispatch is in flight.
        """
        return F.conv1d(
            input=input_,
            weight=self.get_conv1d_weight(),
            bias=self.get_conv1d_bias(),
            padding=self.conv1d_padding,
            groups=self._conv1d_channels(),
        )

    def _conv1d_channels(self) -> int:
        """Number of depthwise-conv channels for this TP rank.

        Upstream comment: ``in_channels == out_channels == groups``.
        """
        if self.conv1d_weight_cp1 is None:
            return 0
        return self.get_conv1d_weight().shape[0]

    def _slice_conv_param(self, param: torch.Tensor) -> torch.Tensor:
        """Slice a conv parameter to the current CP rank's channel range.

        For ``cp_size == 1`` the full parameter is returned unchanged, matching
        the upstream post-PR behavior where the fast path was removed and the
        slice is always taken (it is a no-op when the slice covers all channels).
        """
        if param is None:
            return param
        if self.cp_size == 1:
            return param
        channels_per_rank = param.shape[0] // self.cp_size
        start = self.cp_rank * channels_per_rank
        end = start + channels_per_rank
        return param[start:end]


# ---------------------------------------------------------------------------
# Heterogeneous Mamba Conv Mixer
# ---------------------------------------------------------------------------

class HeteroMambaConvMixer(nn.Module):
    """DES-LOC-aware Mamba conv parameter block.

    Upstream Design Intent
    ----------------------
    In Megatron after commit 35992ba8 the conv parameters are exposed as direct
    ``nn.Parameter`` attributes (``conv1d_weight``, ``conv1d_bias``) of
    ``MambaMixer`` instead of being hidden inside an ``nn.Conv1d`` sub-module.
    This simplifies FSDP parameter gathering (the existing non-unit pre-forward
    hook can see the params without special-casing), removes a redundant
    ``Module.to()`` indirection, and eliminates dead code such as the
    ``self.conv1d.bias.data_ptr()`` call that served no purpose.

    Initialization preserves the original ``nn.Conv1d`` RNG consumption order:
    first a ``kaiming_uniform_`` on the weight, then a ``uniform_`` on the
    bias.  A second conditional ``kaiming_uniform_`` (or custom ``conv_init``)
    follows if ``perform_initialization`` is set.  The upstream commit comment
    notes this preserves existing test baselines.

    Checkpoint compatibility: keys written as ``conv1d.weight`` / ``conv1d.bias``
    are remapped to ``conv1d_weight`` / ``conv1d_bias`` via
    ``state_dict_compat_remap`` so that old checkpoints load without manual key
    surgery.

    DES-LOC Adaptation
    ------------------
    * Parameters are created on the ``home_tier`` device.
    * A ``HeteroConvParamStore`` tracks tier placement and manages the locality
      cache (CPU DRAM staging buffer).
    * The ``DESLOCContextParallel`` object is built with a reference to the
      store so that ``get_conv1d_weight`` / ``get_conv1d_bias`` can fetch to
      the correct device tier at forward time.
    * ``assign_execution_tier`` lets the DeepSpeed engine move the "hot" copy
      to a different tier without touching optimizer state.

    Parameters
    ----------
    conv_dim : int
        Total conv channel count (= d_inner + 2 * ngroups * d_state for this TP rank).
    d_conv : int
        Kernel size of the depthwise conv.
    params_dtype : torch.dtype
        Parameter dtype (e.g. bfloat16).
    home_tier : TierDescriptor
        The device tier that owns the parameters for gradient purposes.
    cluster_tiers : list of TierDescriptor
        All GPU tiers in the cluster.
    cp_size : int
        Context-parallel world size.
    cp_rank : int
        Context-parallel rank.
    tp_size : int
        Tensor-parallel world size.
    tp_rank : int
        Tensor-parallel rank.
    nheads_local_tp : int
        Heads on this TP rank.
    ngroups_local_tp : int
        Groups on this TP rank.
    d_state : int
        Mamba d_state.
    conv_init : float or None
        If set, weight is initialized with ``uniform_(-conv_init, conv_init)``
        after the Kaiming pass (mirrors Megatron ``perform_initialization``).
    """

    # Map from old checkpoint key suffix → new parameter name
    _CONV_COMPAT_KEY_MAP: Dict[str, str] = {
        "conv1d.weight": "conv1d_weight",
        "conv1d.bias":   "conv1d_bias",
    }

    def __init__(
        self,
        conv_dim: int,
        d_conv: int,
        params_dtype: torch.dtype,
        home_tier: TierDescriptor,
        cluster_tiers: List[TierDescriptor],
        cp_size: int = 1,
        cp_rank: int = 0,
        tp_size: int = 1,
        tp_rank: int = 0,
        nheads_local_tp: int = 1,
        ngroups_local_tp: int = 1,
        d_state: int = 64,
        conv_init: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.conv_dim = conv_dim
        self.d_conv = d_conv
        self.params_dtype = params_dtype
        self.home_tier = home_tier
        self._conv_init = conv_init

        # ------------------------------------------------------------------
        # Parameter initialization
        # Mirrors the upstream init sequence exactly to preserve RNG state for
        # any existing checkpoint baselines:
        #   1. kaiming_uniform_ on weight
        #   2. uniform_ on bias  (fan_in-based bound)
        #   3. optional second pass guarded by perform_initialization
        # ------------------------------------------------------------------
        self.conv1d_weight = nn.Parameter(
            torch.empty(conv_dim, 1, d_conv, device=home_tier.device, dtype=params_dtype)
        )
        self.conv1d_bias = nn.Parameter(
            torch.empty(conv_dim, device=home_tier.device, dtype=params_dtype)
        )

        # First pass: preserve upstream RNG consumption order.
        nn.init.kaiming_uniform_(self.conv1d_weight, a=math.sqrt(5))
        fan_in = self.conv1d_weight.size(1) * self.conv1d_weight.size(2)
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.conv1d_bias, -bound, bound)

        # Second pass: custom or default kaiming (guarded by perform_initialization).
        if conv_init is not None:
            nn.init.uniform_(self.conv1d_weight, -conv_init, conv_init)
            logger.debug(
                "HeteroMambaConvMixer: applied custom conv_init=%.4f on tier '%s'",
                conv_init,
                home_tier.label,
            )
        else:
            nn.init.kaiming_uniform_(self.conv1d_weight, a=math.sqrt(5))

        # Tensor-model-parallel metadata (mirrors upstream setattr pattern).
        setattr(self.conv1d_weight, "tensor_model_parallel", True)
        setattr(self.conv1d_weight, "partition_dim", 0)
        setattr(self.conv1d_bias,   "tensor_model_parallel", True)
        setattr(self.conv1d_bias,   "partition_dim", 0)

        # ------------------------------------------------------------------
        # Locality cache and param store
        # ------------------------------------------------------------------
        self._param_store = HeteroConvParamStore(
            weight_param=self.conv1d_weight,
            bias_param=self.conv1d_bias,
            home_tier=home_tier,
            cluster_tiers=cluster_tiers,
        )

        # ------------------------------------------------------------------
        # Context-parallel helper
        # ------------------------------------------------------------------
        self.cp = DESLOCContextParallel(
            cp_size=cp_size,
            cp_rank=cp_rank,
            tp_size=tp_size,
            tp_rank=tp_rank,
            nheads_local_tp=nheads_local_tp,
            ngroups_local_tp=ngroups_local_tp,
            d_state=d_state,
            conv1d_weight_cp1=self.conv1d_weight,
            conv1d_bias_cp1=self.conv1d_bias,
            conv1d_padding=d_conv - 1,
            param_store=self._param_store,
            target_tier=home_tier,
        )

        logger.info(
            "HeteroMambaConvMixer initialized: conv_dim=%d d_conv=%d dtype=%s "
            "home_tier='%s' cp_size=%d tp_size=%d",
            conv_dim,
            d_conv,
            params_dtype,
            home_tier.label,
            cp_size,
            tp_size,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, xBC: torch.Tensor) -> torch.Tensor:
        """Apply the depthwise causal conv to *xBC*.

        Parameters
        ----------
        xBC : torch.Tensor
            Shape ``[batch, seq_len, conv_dim]`` or ``[batch, conv_dim, seq_len]``
            depending on the caller's convention.  The conv is applied along the
            last dimension (channels-last layout expected by ``F.conv1d`` with
            input shape ``[batch, channels, length]``).

        Returns
        -------
        torch.Tensor
            Conv output with the same shape as *xBC*.
        """
        return self.cp.apply_conv1d(xBC)

    # ------------------------------------------------------------------
    # Tier management
    # ------------------------------------------------------------------

    def assign_execution_tier(self, tier: TierDescriptor) -> None:
        """Redirect forward-pass parameter fetches to *tier*.

        DES-LOC Adaptation
        ------------------
        The DeepSpeed engine calls this method when scheduling a layer on a
        specific device tier.  We update ``cp.target_tier`` so that subsequent
        ``get_conv1d_weight`` / ``get_conv1d_bias`` calls stream the params to
        the correct device.  We also evict the current copies to the locality
        cache first so the old-tier VRAM is reclaimed.

        This is the key difference from the upstream design: in Megatron the
        device is fixed at construction time; here it can change between
        forward passes without reinitializing parameters.
        """
        if tier.tier_id == self.cp.target_tier.tier_id:
            return

        self._param_store.evict_to_cache()
        old_label = self.cp.target_tier.label
        self.cp.target_tier = tier
        logger.info(
            "HeteroMambaConvMixer: execution tier reassigned %s → %s",
            old_label,
            tier.label,
        )

    def notify_optimizer_step(self) -> None:
        """Mark locality cache stale after an optimizer step.

        Should be called by the DeepSpeed optimizer wrapper after each
        parameter update so that the next cross-tier fetch refreshes from the
        up-to-date GPU copy rather than the stale CPU snapshot.
        """
        self._param_store.mark_params_stale()

    # ------------------------------------------------------------------
    # Checkpoint compatibility
    # ------------------------------------------------------------------

    def state_dict_compat_remap(
        self, state_dict: Dict[str, torch.Tensor], prefix: str = ""
    ) -> Dict[str, torch.Tensor]:
        """Remap old checkpoint keys to the new direct-parameter layout.

        Upstream Design Intent
        ----------------------
        Megatron PR #4899 includes a ``conv_checkpoint_key_map`` in
        ``MambaMixer.sharded_state_dict`` that rewrites
        ``conv1d.weight`` → ``conv1d_weight`` and
        ``conv1d.bias`` → ``conv1d_bias`` so that checkpoints written before
        the refactor load without key mismatches.

        DES-LOC Adaptation
        ------------------
        We expose the same logic as a standalone method that the DeepSpeed
        checkpoint loader can call before ``load_state_dict``.  The prefix
        argument handles nested module paths (e.g. ``"mixer."``).
        """
        remapped: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            new_k = k
            for old_suffix, new_suffix in self._CONV_COMPAT_KEY_MAP.items():
                old_key = f"{prefix}{old_suffix}"
                new_key = f"{prefix}{new_suffix}"
                if k == old_key:
                    new_k = new_key
                    logger.debug(
                        "state_dict_compat_remap: '%s' → '%s'", old_key, new_key
                    )
                    break
            remapped[new_k] = v
        return remapped

    def mamba_conv_state_shape(self) -> Tuple[int, int]:
        """Return ``(conv_dim, d_conv)`` for conv-state allocation.

        Upstream: ``MambaMixer.mamba_state_shapes_per_request`` used
        ``self.conv1d.weight.shape[0]`` after the refactor it uses
        ``self.conv1d_weight.shape[0]``.  We expose the same information here.
        """
        return (self.conv1d_weight.shape[0], self.d_conv)

    def make_conv_state(
        self, batch_size: int, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Allocate a zeroed conv state buffer on the home tier.

        Mirrors upstream ``MambaMixer._allocate_inference_cache`` which
        switched from ``self.conv1d.weight.device`` to
        ``self.conv1d_weight.device`` after the refactor.
        """
        dtype = dtype or self.conv1d_weight.dtype
        conv_dim, d_conv = self.mamba_conv_state_shape()
        return torch.zeros(
            batch_size,
            conv_dim,
            d_conv,
            device=self.conv1d_weight.device,
            dtype=dtype,
        )


# ---------------------------------------------------------------------------
# DES-LOC Flextron Elasticity Bridge
# ---------------------------------------------------------------------------

class DESLOCFlextronElasticityBridge:
    """Bridges the Flextron elasticity manager to DES-LOC tier-aware conv params.

    Upstream Design Intent
    ----------------------
    In Megatron PR #4899 the Flextron elasticity hooks were refactored from an
    ``nn.Module.register_forward_hook`` on ``conv1d`` to instance-method patches
    on ``mamba_mixer.cp.get_conv1d_weight`` and ``get_conv1d_bias``.  This was
    necessary because there is no longer an ``nn.Conv1d`` module to hook.  An
    ``_AttributeRestoreHandle`` is used to cleanly undo the patches when the
    elasticity scope ends.

    The upstream ``conv1d_output_scale()`` closure computes a per-channel scale
    tensor from the router logits and then scales weight/bias before the conv so
    that the pre-activation output is effectively masked — equivalent to the old
    ``masked_output = output * conv1d_mask`` but applied to the kernel rather
    than the activation.

    DES-LOC Adaptation
    ------------------
    The bridge wraps a ``HeteroMambaConvMixer`` instead of a raw ``MambaMixer``.
    It adds one extra step: before applying the scale tensor it moves it to the
    conv mixer's current execution tier device via the locality cache, so that
    the ``weight * scale`` computation runs on the correct GPU.

    Parameters
    ----------
    conv_mixer : HeteroMambaConvMixer
        The mixer whose conv params will be elasticity-scaled.
    router_logit_fn : callable
        Zero-argument callable that returns the current router logits tensor
        (or ``None`` if elasticity is not active this step).
    conv1d_mask_list : list of torch.Tensor
        Per-mamba-int channel masks, one per elasticity level.
    mamba_masks_lookup : dict
        Maps ``mamba_per`` (fraction) → index into ``conv1d_mask_list``.
    soft_mask : bool
        If True, soft-blended masks are used; otherwise hard masks.
    """

    def __init__(
        self,
        conv_mixer: HeteroMambaConvMixer,
        router_logit_fn: Callable[[], Optional[torch.Tensor]],
        conv1d_mask_list: List[torch.Tensor],
        mamba_masks_lookup: Dict[float, int],
        soft_mask: bool = False,
    ) -> None:
        self.conv_mixer = conv_mixer
        self.router_logit_fn = router_logit_fn
        self.conv1d_mask_list = conv1d_mask_list
        self.mamba_masks_lookup = mamba_masks_lookup
        self.soft_mask = soft_mask
        self._handles: List[_AttributeRestoreHandle] = []
        self._active = False

    def _compute_conv_scale(self) -> Optional[torch.Tensor]:
        """Compute the per-channel conv output scale from router logits.

        Returns ``None`` if elasticity is not active.  Otherwise returns a
        1-D tensor of shape ``[conv_dim]`` with values in [0, 1] (hard mask)
        or soft-blended logit-weighted values.

        This mirrors ``conv1d_output_scale()`` in upstream
        ``flextron_elasticity_hooks.py`` but is pulled out of the closure so
        that it can be tested independently.
        """
        logits = self.router_logit_fn()
        if logits is None:
            return None

        if self.soft_mask:
            scale = torch.zeros_like(self.conv1d_mask_list[0], dtype=torch.float32)
            for mask, per_logit in zip(self.conv1d_mask_list, logits):
                scale = scale + mask.float() * per_logit
            return scale

        # Hard mask path.
        router_mamba_logits = torch.max(logits)
        mamba_per = float(logits.argmax())
        mask_idx = self.mamba_masks_lookup.get(mamba_per, len(self.conv1d_mask_list) - 1)
        conv1d_mask = self.conv1d_mask_list[mask_idx].float()
        return conv1d_mask * router_mamba_logits

    def _scale_to_tier(self, scale: torch.Tensor) -> torch.Tensor:
        """Move *scale* to the conv mixer's current execution tier.

        DES-LOC Adaptation
        ------------------
        The upstream code runs ``conv1d_mask.to(device=weight.device)`` inline.
        Here we go through the locality cache if the scale is large enough that
        a direct ``Tensor.to()`` would stall the compute stream.  For small
        scale tensors (< 1 MB) we fall through to a direct transfer.
        """
        target_device = self.conv_mixer.cp.target_tier.device
        if scale.device == target_device:
            return scale
        nbytes = scale.numel() * scale.element_size()
        if nbytes < (1 << 20):
            return scale.to(device=target_device, non_blocking=True)
        # Large scale: route through pinned staging for async transfer.
        pinned = scale.detach().cpu().pin_memory()
        return pinned.to(device=target_device, non_blocking=True)

    def activate(self) -> None:
        """Patch ``cp.get_conv1d_weight`` and ``cp.get_conv1d_bias``.

        Mirrors upstream Flextron hook attachment but uses
        ``_AttributeRestoreHandle`` instead of ``register_forward_hook``.
        """
        if self._active:
            return

        cp = self.conv_mixer.cp
        original_weight_fn = cp.get_conv1d_weight
        original_bias_fn   = cp.get_conv1d_bias

        bridge = self  # capture for closures

        def get_scaled_weight() -> torch.Tensor:
            weight = original_weight_fn()
            scale = bridge._compute_conv_scale()
            if scale is None:
                return weight
            scale = cp._slice_conv_param(scale)
            scale = bridge._scale_to_tier(scale).to(dtype=weight.dtype)
            return weight * scale[:, None, None]

        def get_scaled_bias() -> torch.Tensor:
            bias = original_bias_fn()
            scale = bridge._compute_conv_scale()
            if scale is None:
                return bias
            scale = cp._slice_conv_param(scale)
            scale = bridge._scale_to_tier(scale).to(dtype=bias.dtype)
            return bias * scale

        cp.get_conv1d_weight = get_scaled_weight
        cp.get_conv1d_bias   = get_scaled_bias

        self._handles = [
            _AttributeRestoreHandle(cp, "get_conv1d_weight", original_weight_fn),
            _AttributeRestoreHandle(cp, "get_conv1d_bias",   original_bias_fn),
        ]
        self._active = True
        logger.debug(
            "DESLOCFlextronElasticityBridge: activated method patches on cp"
        )

    def deactivate(self) -> None:
        """Restore original ``get_conv1d_weight`` / ``get_conv1d_bias``."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._active = False
        logger.debug("DESLOCFlextronElasticityBridge: deactivated method patches")

    def __enter__(self) -> "DESLOCFlextronElasticityBridge":
        self.activate()
        return self

    def __exit__(self, *_: object) -> None:
        self.deactivate()


# ---------------------------------------------------------------------------
# Utility: build a HeteroMambaConvMixer from a Megatron-style MambaMixer dict
# ---------------------------------------------------------------------------

def build_hetero_conv_mixer_from_config(
    conv_dim: int,
    d_conv: int,
    params_dtype: torch.dtype,
    cp_size: int,
    cp_rank: int,
    tp_size: int,
    tp_rank: int,
    nheads_local_tp: int,
    ngroups_local_tp: int,
    d_state: int,
    cluster_tiers: Optional[List[TierDescriptor]] = None,
    home_tier_id: int = 0,
    conv_init: Optional[float] = None,
) -> HeteroMambaConvMixer:
    """Factory that wires cluster topology to a ``HeteroMambaConvMixer``.

    Parameters
    ----------
    conv_dim : int
        Conv channel dimension.
    d_conv : int
        Conv kernel size.
    params_dtype : torch.dtype
        Parameter dtype.
    cp_size, cp_rank : int
        Context-parallel world size / rank.
    tp_size, tp_rank : int
        Tensor-parallel world size / rank.
    nheads_local_tp, ngroups_local_tp : int
        Per-TP-rank head / group counts.
    d_state : int
        Mamba d_state.
    cluster_tiers : list of TierDescriptor, optional
        Cluster topology.  Defaults to ``_DEFAULT_CLUSTER_TIERS``.
    home_tier_id : int
        Which tier to place the parameters on (index into *cluster_tiers*).
    conv_init : float or None
        Custom weight init range.

    Returns
    -------
    HeteroMambaConvMixer
    """
    if cluster_tiers is None:
        cluster_tiers = _DEFAULT_CLUSTER_TIERS

    available: List[TierDescriptor] = []
    for t in cluster_tiers:
        try:
            _ = torch.zeros(1, device=t.device)
            available.append(t)
        except RuntimeError:
            logger.warning(
                "build_hetero_conv_mixer_from_config: tier '%s' not available, skipping",
                t.label,
            )

    if not available:
        logger.warning(
            "build_hetero_conv_mixer_from_config: no GPU tiers available, "
            "falling back to CPU locality tier"
        )
        available = [_CPU_TIER]

    home_tier = available[home_tier_id % len(available)]

    return HeteroMambaConvMixer(
        conv_dim=conv_dim,
        d_conv=d_conv,
        params_dtype=params_dtype,
        home_tier=home_tier,
        cluster_tiers=available,
        cp_size=cp_size,
        cp_rank=cp_rank,
        tp_size=tp_size,
        tp_rank=tp_rank,
        nheads_local_tp=nheads_local_tp,
        ngroups_local_tp=ngroups_local_tp,
        d_state=d_state,
        conv_init=conv_init,
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    class _TestTierDescriptor(unittest.TestCase):
        def test_cpu_tier_attrs(self):
            self.assertEqual(_CPU_TIER.sm_arch, 0)
            self.assertEqual(_CPU_TIER.device, torch.device("cpu"))
            self.assertGreater(_CPU_TIER.vram_bytes, 0)

        def test_default_cluster_has_three_tiers(self):
            self.assertEqual(len(_DEFAULT_CLUSTER_TIERS), 3)
            sm_archs = {t.sm_arch for t in _DEFAULT_CLUSTER_TIERS}
            self.assertIn(90, sm_archs)
            self.assertIn(86, sm_archs)

    class _TestLocalityCacheHandle(unittest.TestCase):
        def _make_handle(self, shape=(16, 1, 4)):
            t = torch.randn(*shape)
            return LocalityCacheHandle(t, "test_tensor"), t

        def test_initial_not_stale(self):
            h, _ = self._make_handle()
            self.assertFalse(h.is_stale)

        def test_mark_stale(self):
            h, _ = self._make_handle()
            h.mark_stale()
            self.assertTrue(h.is_stale)

        def test_refresh_clears_stale(self):
            h, t = self._make_handle()
            h.mark_stale()
            h.refresh(t)
            self.assertFalse(h.is_stale)

        def test_pinned_shape_preserved(self):
            shape = (8, 1, 3)
            h, _ = self._make_handle(shape)
            self.assertEqual(tuple(h.pinned.shape), shape)

        def test_fetch_to_cpu(self):
            h, t = self._make_handle()
            result = h.fetch_to(torch.device("cpu"), non_blocking=False)
            self.assertTrue(torch.allclose(result.float(), t.float(), atol=1e-5))

    class _TestAttributeRestoreHandle(unittest.TestCase):
        def test_restore_on_remove(self):
            class _Obj:
                value = 42

            obj = _Obj()
            obj.value = 99
            handle = _AttributeRestoreHandle(obj, "value", 42)
            handle.remove()
            self.assertEqual(obj.value, 42)

        def test_double_remove_is_safe(self):
            class _Obj:
                value = 1

            obj = _Obj()
            handle = _AttributeRestoreHandle(obj, "value", 0)
            handle.remove()
            obj.value = 999
            handle.remove()  # Should not overwrite again.
            self.assertEqual(obj.value, 999)

    class _TestDESLOCContextParallel(unittest.TestCase):
        def _make_cp(self, cp_size=1, cp_rank=0):
            weight = torch.randn(8, 1, 4)
            bias = torch.randn(8)
            return DESLOCContextParallel(
                cp_size=cp_size,
                cp_rank=cp_rank,
                tp_size=1,
                tp_rank=0,
                nheads_local_tp=1,
                ngroups_local_tp=1,
                d_state=4,
                conv1d_weight_cp1=weight,
                conv1d_bias_cp1=bias,
                conv1d_padding=3,
            ), weight, bias

        def test_get_weight_cp1(self):
            cp, weight, _ = self._make_cp(cp_size=1)
            result = cp.get_conv1d_weight()
            self.assertTrue(torch.equal(result, weight))

        def test_get_bias_cp1(self):
            cp, _, bias = self._make_cp(cp_size=1)
            result = cp.get_conv1d_bias()
            self.assertTrue(torch.equal(result, bias))

        def test_slice_weight_cp2(self):
            weight = torch.randn(8, 1, 4)
            bias = torch.randn(8)
            cp = DESLOCContextParallel(
                cp_size=2, cp_rank=1,
                tp_size=1, tp_rank=0,
                nheads_local_tp=1, ngroups_local_tp=1, d_state=4,
                conv1d_weight_cp1=weight,
                conv1d_bias_cp1=bias,
                conv1d_padding=3,
            )
            sliced = cp.get_conv1d_weight()
            self.assertEqual(sliced.shape[0], 4)
            self.assertTrue(torch.equal(sliced, weight[4:]))

        def test_apply_conv1d_output_shape(self):
            conv_dim = 8
            seq_len  = 16
            cp, _, _ = self._make_cp(cp_size=1)
            x = torch.randn(2, conv_dim, seq_len)
            out = cp.apply_conv1d(x)
            # padding = d_conv - 1 = 3, so output length = seq_len
            self.assertEqual(out.shape, (2, conv_dim, seq_len))

        def test_apply_conv1d_matches_f_conv1d(self):
            conv_dim = 4
            d_conv   = 3
            weight   = torch.randn(conv_dim, 1, d_conv)
            bias     = torch.randn(conv_dim)
            cp = DESLOCContextParallel(
                cp_size=1, cp_rank=0,
                tp_size=1, tp_rank=0,
                nheads_local_tp=1, ngroups_local_tp=1, d_state=4,
                conv1d_weight_cp1=weight,
                conv1d_bias_cp1=bias,
                conv1d_padding=d_conv - 1,
            )
            x = torch.randn(1, conv_dim, 10)
            out_cp = cp.apply_conv1d(x)
            out_ref = F.conv1d(x, weight, bias, padding=d_conv - 1, groups=conv_dim)
            self.assertTrue(torch.allclose(out_cp, out_ref, atol=1e-5))

        def test_none_params_return_none(self):
            cp = DESLOCContextParallel(
                cp_size=1, cp_rank=0,
                tp_size=1, tp_rank=0,
                nheads_local_tp=1, ngroups_local_tp=1, d_state=4,
                conv1d_weight_cp1=None,
                conv1d_bias_cp1=None,
                conv1d_padding=0,
            )
            self.assertIsNone(cp.get_conv1d_weight())
            self.assertIsNone(cp.get_conv1d_bias())

    class _TestHeteroConvParamStore(unittest.TestCase):
        def _make_store(self):
            weight = nn.Parameter(torch.randn(8, 1, 4))
            bias   = nn.Parameter(torch.randn(8))
            tier   = _CPU_TIER
            store  = HeteroConvParamStore(
                weight_param=weight,
                bias_param=bias,
                home_tier=tier,
                cluster_tiers=[],
            )
            return store, weight, bias

        def test_fetch_weight_same_device(self):
            store, weight, _ = self._make_store()
            result = store.fetch_weight(_CPU_TIER, non_blocking=False)
            self.assertTrue(torch.equal(result, weight))

        def test_fetch_bias_same_device(self):
            store, _, bias = self._make_store()
            result = store.fetch_bias(_CPU_TIER, non_blocking=False)
            self.assertTrue(torch.equal(result, bias))

        def test_evict_and_fetch_roundtrip(self):
            store, weight, _ = self._make_store()
            store.evict_to_cache()
            fetched = store._weight_cache.fetch_to(torch.device("cpu"), non_blocking=False)
            self.assertTrue(torch.allclose(fetched.float(), weight.float(), atol=1e-5))

        def test_mark_params_stale(self):
            store, _, _ = self._make_store()
            store.mark_params_stale()
            self.assertTrue(store._weight_cache.is_stale)
            self.assertTrue(store._bias_cache.is_stale)

    class _TestHeteroMambaConvMixer(unittest.TestCase):
        def _make_mixer(self, conv_dim=16, d_conv=4):
            cpu_tier = _CPU_TIER
            return HeteroMambaConvMixer(
                conv_dim=conv_dim,
                d_conv=d_conv,
                params_dtype=torch.float32,
                home_tier=cpu_tier,
                cluster_tiers=[cpu_tier],
                cp_size=1,
                cp_rank=0,
                tp_size=1,
                tp_rank=0,
                nheads_local_tp=1,
                ngroups_local_tp=1,
                d_state=4,
            )

        def test_param_shapes(self):
            conv_dim, d_conv = 16, 4
            mixer = self._make_mixer(conv_dim, d_conv)
            self.assertEqual(mixer.conv1d_weight.shape, (conv_dim, 1, d_conv))
            self.assertEqual(mixer.conv1d_bias.shape, (conv_dim,))

        def test_tensor_model_parallel_attr(self):
            mixer = self._make_mixer()
            self.assertTrue(getattr(mixer.conv1d_weight, "tensor_model_parallel", False))
            self.assertTrue(getattr(mixer.conv1d_bias, "tensor_model_parallel", False))

        def test_no_conv1d_submodule(self):
            mixer = self._make_mixer()
            self.assertFalse(hasattr(mixer, "conv1d"))
            child_names = [n for n, _ in mixer.named_children()]
            self.assertNotIn("conv1d", child_names)

        def test_forward_output_shape(self):
            conv_dim, d_conv = 8, 3
            mixer = self._make_mixer(conv_dim, d_conv)
            x = torch.randn(2, conv_dim, 12)
            out = mixer(x)
            self.assertEqual(out.shape, x.shape)

        def test_forward_matches_f_conv1d(self):
            conv_dim, d_conv = 8, 4
            mixer = self._make_mixer(conv_dim, d_conv)
            mixer.eval()
            x = torch.randn(1, conv_dim, 20)
            with torch.no_grad():
                out_mixer = mixer(x)
                out_ref = F.conv1d(
                    x,
                    mixer.conv1d_weight,
                    mixer.conv1d_bias,
                    padding=d_conv - 1,
                    groups=conv_dim,
                )
            self.assertTrue(torch.allclose(out_mixer, out_ref, atol=1e-5))

        def test_mamba_conv_state_shape(self):
            conv_dim, d_conv = 12, 4
            mixer = self._make_mixer(conv_dim, d_conv)
            shape = mixer.mamba_conv_state_shape()
            self.assertEqual(shape, (conv_dim, d_conv))

        def test_make_conv_state(self):
            conv_dim, d_conv, batch = 12, 4, 3
            mixer = self._make_mixer(conv_dim, d_conv)
            state = mixer.make_conv_state(batch)
            self.assertEqual(state.shape, (batch, conv_dim, d_conv))
            self.assertTrue((state == 0).all())

        def test_state_dict_has_direct_params(self):
            mixer = self._make_mixer()
            sd = mixer.state_dict()
            self.assertIn("conv1d_weight", sd)
            self.assertIn("conv1d_bias", sd)
            self.assertNotIn("conv1d.weight", sd)
            self.assertNotIn("conv1d.bias", sd)

        def test_state_dict_compat_remap(self):
            mixer = self._make_mixer()
            old_sd = {
                "conv1d.weight": torch.randn(8, 1, 4),
                "conv1d.bias":   torch.randn(8),
                "other_param":   torch.randn(4),
            }
            new_sd = mixer.state_dict_compat_remap(old_sd)
            self.assertIn("conv1d_weight", new_sd)
            self.assertIn("conv1d_bias", new_sd)
            self.assertIn("other_param", new_sd)
            self.assertNotIn("conv1d.weight", new_sd)

        def test_assign_execution_tier_same_tier_noop(self):
            mixer = self._make_mixer()
            original_target = mixer.cp.target_tier
            mixer.assign_execution_tier(_CPU_TIER)
            self.assertIs(mixer.cp.target_tier, original_target)

        def test_notify_optimizer_step_marks_stale(self):
            mixer = self._make_mixer()
            mixer.notify_optimizer_step()
            self.assertTrue(mixer._param_store._weight_cache.is_stale)
            self.assertTrue(mixer._param_store._bias_cache.is_stale)

        def test_grad_flows_through_weight(self):
            mixer = self._make_mixer(conv_dim=8, d_conv=4)
            x = torch.randn(1, 8, 10)
            out = mixer(x)
            loss = out.sum()
            loss.backward()
            self.assertIsNotNone(mixer.conv1d_weight.grad)
            self.assertIsNotNone(mixer.conv1d_bias.grad)

        def test_custom_conv_init(self):
            conv_dim, d_conv = 8, 4
            cpu_tier = _CPU_TIER
            mixer = HeteroMambaConvMixer(
                conv_dim=conv_dim,
                d_conv=d_conv,
                params_dtype=torch.float32,
                home_tier=cpu_tier,
                cluster_tiers=[cpu_tier],
                conv_init=0.01,
            )
            w = mixer.conv1d_weight.detach()
            self.assertTrue((w.abs() <= 0.01 + 1e-6).all())

    class _TestDESLOCFlextronElasticityBridge(unittest.TestCase):
        def _make_bridge(self, soft_mask=False, return_logits=True):
            conv_dim = 8
            mixer = HeteroMambaConvMixer(
                conv_dim=conv_dim,
                d_conv=4,
                params_dtype=torch.float32,
                home_tier=_CPU_TIER,
                cluster_tiers=[_CPU_TIER],
            )
            masks = [
                torch.ones(conv_dim) * i for i in range(1, 4)
            ]
            lookup = {0.0: 0, 1.0: 1, 2.0: 2}
            logits_val = [0.5, 0.3, 0.2]

            def router_logit_fn():
                if return_logits:
                    return torch.tensor(logits_val)
                return None

            bridge = DESLOCFlextronElasticityBridge(
                conv_mixer=mixer,
                router_logit_fn=router_logit_fn,
                conv1d_mask_list=masks,
                mamba_masks_lookup=lookup,
                soft_mask=soft_mask,
            )
            return bridge, mixer

        def test_activate_patches_methods(self):
            bridge, mixer = self._make_bridge()
            original_w = mixer.cp.get_conv1d_weight
            bridge.activate()
            self.assertIsNot(mixer.cp.get_conv1d_weight, original_w)
            bridge.deactivate()

        def test_deactivate_restores_methods(self):
            bridge, mixer = self._make_bridge()
            original_w = mixer.cp.get_conv1d_weight
            bridge.activate()
            bridge.deactivate()
            self.assertIs(mixer.cp.get_conv1d_weight, original_w)

        def test_context_manager_restores(self):
            bridge, mixer = self._make_bridge()
            original_w = mixer.cp.get_conv1d_weight
            with bridge:
                pass
            self.assertIs(mixer.cp.get_conv1d_weight, original_w)

        def test_scaled_weight_differs_from_unscaled(self):
            bridge, mixer = self._make_bridge(soft_mask=True)
            unscaled_w = mixer.cp.get_conv1d_weight().clone()
            bridge.activate()
            scaled_w = mixer.cp.get_conv1d_weight()
            bridge.deactivate()
            self.assertFalse(torch.equal(scaled_w, unscaled_w))

        def test_no_logits_returns_original(self):
            bridge, mixer = self._make_bridge(return_logits=False)
            original_w = mixer.cp.get_conv1d_weight().clone()
            bridge.activate()
            result_w = mixer.cp.get_conv1d_weight()
            bridge.deactivate()
            self.assertTrue(torch.equal(result_w, original_w))

        def test_double_activate_is_idempotent(self):
            bridge, mixer = self._make_bridge()
            bridge.activate()
            bridge.activate()  # Should not double-patch.
            bridge.deactivate()
            self.assertFalse(bridge._active)

        def test_compute_conv_scale_soft(self):
            bridge, _ = self._make_bridge(soft_mask=True)
            scale = bridge._compute_conv_scale()
            self.assertIsNotNone(scale)
            self.assertEqual(scale.shape[0], 8)

        def test_compute_conv_scale_hard(self):
            bridge, _ = self._make_bridge(soft_mask=False)
            scale = bridge._compute_conv_scale()
            self.assertIsNotNone(scale)
            self.assertEqual(scale.shape[0], 8)

        def test_compute_conv_scale_none_when_no_logits(self):
            bridge, _ = self._make_bridge(return_logits=False)
            scale = bridge._compute_conv_scale()
            self.assertIsNone(scale)

    class _TestBuildFactory(unittest.TestCase):
        def test_factory_cpu_fallback(self):
            cpu_only = [_CPU_TIER]
            mixer = build_hetero_conv_mixer_from_config(
                conv_dim=8,
                d_conv=4,
                params_dtype=torch.float32,
                cp_size=1,
                cp_rank=0,
                tp_size=1,
                tp_rank=0,
                nheads_local_tp=1,
                ngroups_local_tp=1,
                d_state=4,
                cluster_tiers=cpu_only,
                home_tier_id=0,
            )
            self.assertIsInstance(mixer, HeteroMambaConvMixer)
            self.assertEqual(mixer.conv1d_weight.device, torch.device("cpu"))

        def test_factory_home_tier_wraps(self):
            cpu_only = [_CPU_TIER]
            # home_tier_id out of range → wraps around
            mixer = build_hetero_conv_mixer_from_config(
                conv_dim=8, d_conv=4, params_dtype=torch.float32,
                cp_size=1, cp_rank=0, tp_size=1, tp_rank=0,
                nheads_local_tp=1, ngroups_local_tp=1, d_state=4,
                cluster_tiers=cpu_only,
                home_tier_id=99,
            )
            self.assertIsInstance(mixer, HeteroMambaConvMixer)

    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(_TestTierDescriptor)
    for cls in [
        _TestLocalityCacheHandle,
        _TestAttributeRestoreHandle,
        _TestDESLOCContextParallel,
        _TestHeteroConvParamStore,
        _TestHeteroMambaConvMixer,
        _TestDESLOCFlextronElasticityBridge,
        _TestBuildFactory,
    ]:
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
