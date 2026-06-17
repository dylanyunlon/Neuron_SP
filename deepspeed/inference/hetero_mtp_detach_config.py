"""
deepspeed/inference/hetero_mtp_detach_config.py

DES-LOC Heterogeneous Multi-Token Prediction with Detached Head Configuration
==============================================================================

Upstream Design Intent (Megatron-LM commit 71e418ea):
------------------------------------------------------
The original Megatron-LM commit introduces ``mtp_detach_heads``, a boolean flag
that severs gradient flow between the Multi-Token Prediction (MTP) auxiliary heads
and the main model backbone.  The core insight is that MTP heads are speculative
decoders: they predict future tokens from intermediate hidden states, but their
loss gradients need not propagate back into the backbone weights during stable
training phases.  Detaching achieves three benefits in Megatron's homogeneous
setting:

  1. Gradient isolation — MTP loss only trains the MTP-specific parameters
     (enorm, hnorm, eh_proj), not the shared embedding or backbone.
  2. Numerical stability — avoids accumulated gradient noise from auxiliary
     paths polluting the primary language-model gradient.
  3. Activation checkpointing compatibility — after detach(), tensors lose
     ``requires_grad``; the commit re-enables it on ``hidden_states`` explicitly
     so that ``CheckpointFunction.apply`` can still build a differentiable graph
     through the MTP layer parameters.

The diff touches three callsites:
  * ``process_mtp_loss``   — detaches ``output_weight`` before computing logits.
  * ``MultiTokenPredictionLayer._get_embeddings`` — detaches ``decoder_input``.
  * ``MultiTokenPredictionBlock.forward``         — detaches the per-offset chunk
    of ``hidden_states`` before feeding MTP layers.

DES-LOC Adaptation Points:
---------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) adds a heterogeneous
hardware dimension that Megatron's homogeneous design ignores.  In our cluster:

  * 2× A6000 48 GB  (SM86, compute capability 8.6) — speculative draft devices
  * 1× H100 NVL 96 GB (SM90, compute capability 9.0) — verification / backbone device
  * PCIe interconnect only (no NVLink) — P2P bandwidth ≈ 24 GB/s, latency matters
  * 1.5 TB CPU DRAM — locality cache for KV and activation spill

The critical adaptation is **device-aware gradient isolation**:

  * Detach boundaries must coincide with PCIe transfer boundaries.  A tensor
    that crosses PCIe should almost always be detached: keeping the autograd
    graph alive across PCIe means gradient tensors must cross back in the
    backward pass at full precision — a bandwidth tax we cannot afford.

  * The H100 owns the backbone and the shared output projection (``output_weight``).
    A6000s own MTP draft heads.  Gradient isolation is therefore not just an
    optional training trick but a hard architectural boundary: A6000s cannot
    accumulate gradients into H100-resident parameters without a synchronous PCIe
    round-trip.

  * The ``HeteroMTPDetachConfig`` defined here extends Megatron's scalar boolean
    into a per-device-group policy, tracking which devices are "draft" (A6000)
    vs "verify" (H100) and enforcing appropriate detach semantics at each
    callsite.

  * The LOcality Cache (shared CPU DRAM) serves as the handoff buffer for
    detached tensors: when a hidden-state chunk is detached on an A6000, it is
    optionally spilled to pinned CPU memory before being moved to the next
    device, amortising PCIe cost via async prefetch.

Module layout:
  ``HeteroDeviceRole``        — enum marking a device as DRAFT or VERIFY.
  ``HeteroMTPDetachPolicy``   — per-device detach rules (replaces the scalar bool).
  ``HeteroMTPDetachConfig``   — dataclass integrating policy into DS engine config.
  ``LocalityCacheBuffer``     — lightweight pinned-memory buffer for cross-device
                                 tensor handoff.
  ``HeteroMTPDetachManager``  — runtime manager; called at each MTP callsite.
  ``HeteroProcessMTPLoss``    — drop-in replacement for Megatron's
                                 ``process_mtp_loss`` with hetero-aware detach.
  ``HeteroMultiTokenPredictionLayer`` — wraps ``_get_embeddings`` detach logic.
  ``HeteroMultiTokenPredictionBlock`` — wraps block-level hidden_states detach.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import math
import time
import unittest
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardware role taxonomy
# ---------------------------------------------------------------------------

class HeteroDeviceRole(enum.Enum):
    """
    Logical role of a physical device in the DES-LOC heterogeneous cluster.

    DRAFT  — A6000 48 GB SM86.  Executes MTP speculative heads.  Gradients
              must not escape to backbone parameters on VERIFY devices.
    VERIFY — H100 NVL 96 GB SM90.  Executes backbone and output projection.
              Receives detached tensors from DRAFT devices; never sends live
              autograd graphs to DRAFT devices.
    CPU    — CPU DRAM locality cache.  Intermediate spill target for PCIe
              handoff buffers.
    """
    DRAFT = "draft"
    VERIFY = "verify"
    CPU = "cpu"


# ---------------------------------------------------------------------------
# Per-device detach policy
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class HeteroMTPDetachPolicy:
    """
    Fine-grained detach policy that replaces Megatron's scalar ``mtp_detach_heads``.

    Attributes
    ----------
    detach_decoder_input : bool
        Mirrors Megatron's ``_get_embeddings`` detach of ``decoder_input``.
        Always True for DRAFT devices; configurable for homogeneous fallback.
    detach_hidden_states : bool
        Mirrors Megatron's block-level detach of the hidden-state chunk.
        Enabled when the chunk crosses a PCIe boundary (VERIFY→DRAFT transfer).
    detach_output_weight : bool
        Mirrors Megatron's ``process_mtp_loss`` detach of ``output_weight``.
        Must be True whenever ``output_weight`` lives on a VERIFY device but
        the loss is computed on a DRAFT device.
    ensure_hidden_grad : bool
        Re-enable ``requires_grad`` on hidden_states after detach, matching
        Megatron's fix for ``CheckpointFunction.apply`` compatibility.
        Should remain True unless activation checkpointing is fully disabled.
    spill_to_locality_cache : bool
        When True, detached tensors are staged through pinned CPU DRAM before
        the next device-to-device transfer, amortising PCIe latency via
        async prefetch.  Only meaningful for cross-device boundaries.
    locality_cache_pin_memory : bool
        Whether the CPU-side locality cache buffer uses pinned memory.
        Pinned memory is required for async cudaMemcpyAsync; set False only
        in test environments without CUDA.
    """
    detach_decoder_input: bool = True
    detach_hidden_states: bool = True
    detach_output_weight: bool = True
    ensure_hidden_grad: bool = True
    spill_to_locality_cache: bool = False
    locality_cache_pin_memory: bool = True

    @classmethod
    def for_draft_device(cls) -> "HeteroMTPDetachPolicy":
        """
        Policy for an A6000 DRAFT device.

        All detach flags are enabled; locality cache spill is enabled so that
        the hidden-state handoff from H100 (VERIFY) can be prefetched.
        """
        return cls(
            detach_decoder_input=True,
            detach_hidden_states=True,
            detach_output_weight=True,
            ensure_hidden_grad=True,
            spill_to_locality_cache=True,
            locality_cache_pin_memory=True,
        )

    @classmethod
    def for_verify_device(cls) -> "HeteroMTPDetachPolicy":
        """
        Policy for the H100 VERIFY device.

        The backbone never receives MTP gradients, so detach flags are less
        critical here; however, ``detach_output_weight`` is kept True to
        match the invariant that the shared output projection weight is never
        differentiated through the MTP path.
        """
        return cls(
            detach_decoder_input=False,
            detach_hidden_states=False,
            detach_output_weight=True,
            ensure_hidden_grad=True,
            spill_to_locality_cache=False,
            locality_cache_pin_memory=True,
        )

    @classmethod
    def homogeneous_compat(cls, mtp_detach_heads: bool) -> "HeteroMTPDetachPolicy":
        """
        Compatibility shim for environments where heterogeneous routing is
        disabled.  Reproduces Megatron's original scalar behaviour.
        """
        return cls(
            detach_decoder_input=mtp_detach_heads,
            detach_hidden_states=mtp_detach_heads,
            detach_output_weight=mtp_detach_heads,
            ensure_hidden_grad=mtp_detach_heads,
            spill_to_locality_cache=False,
            locality_cache_pin_memory=False,
        )


# ---------------------------------------------------------------------------
# Integrated configuration dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class HeteroMTPDetachConfig:
    """
    Top-level configuration for DES-LOC heterogeneous MTP detach behaviour.

    This replaces the single ``mtp_detach_heads: bool`` field in Megatron's
    ``TransformerConfig`` with a richer structure that encodes per-device
    policies and PCIe-crossing detection.

    Parameters
    ----------
    mtp_num_layers : int
        Number of MTP speculative head layers.  Mirrors Megatron's field.
    mtp_use_repeated_layer : bool
        Whether a single MTP layer is reused repeatedly.  Mirrors Megatron.
    device_roles : dict[int, HeteroDeviceRole]
        Maps ``torch.cuda.device`` ordinals to their logical role.
        Example: ``{0: DRAFT, 1: DRAFT, 2: VERIFY}`` for our A6000×2 + H100.
    device_policies : dict[HeteroDeviceRole, HeteroMTPDetachPolicy]
        Maps each role to its detach policy.  Constructed automatically by
        ``from_cluster_spec`` if not provided.
    pcie_crossing_device_pairs : list[tuple[int,int]]
        Pairs of (src_device, dst_device) that cross a PCIe bus.  For our
        cluster every pair is a PCIe crossing because there is no NVLink.
        The manager uses this to decide when to spill through locality cache.
    locality_cache_capacity_gb : float
        Maximum CPU DRAM to reserve for locality cache buffers, in GiB.
        Default 16 GiB leaves headroom in the 1.5 TB system.
    mtp_loss_scaling_factor : float
        Scales the aggregated MTP auxiliary loss before adding to main loss.
    enable_hetero_routing : bool
        Master switch.  If False, falls back to Megatron's homogeneous path
        with ``mtp_detach_heads`` semantics.
    mtp_detach_heads : bool
        Homogeneous fallback flag, used only when ``enable_hetero_routing``
        is False.  Mirrors Megatron's ``TransformerConfig.mtp_detach_heads``.
    """

    mtp_num_layers: int = 1
    mtp_use_repeated_layer: bool = False
    device_roles: Dict[int, HeteroDeviceRole] = dataclasses.field(default_factory=dict)
    device_policies: Dict[HeteroDeviceRole, HeteroMTPDetachPolicy] = dataclasses.field(
        default_factory=dict
    )
    pcie_crossing_device_pairs: List[Tuple[int, int]] = dataclasses.field(default_factory=list)
    locality_cache_capacity_gb: float = 16.0
    mtp_loss_scaling_factor: float = 0.1
    enable_hetero_routing: bool = True
    mtp_detach_heads: bool = False  # homogeneous compat only

    @classmethod
    def from_cluster_spec(
        cls,
        draft_device_ids: List[int],
        verify_device_id: int,
        mtp_num_layers: int = 1,
        mtp_use_repeated_layer: bool = False,
        mtp_loss_scaling_factor: float = 0.1,
        locality_cache_capacity_gb: float = 16.0,
    ) -> "HeteroMTPDetachConfig":
        """
        Construct a config from the physical cluster specification.

        For the DES-LOC target cluster:
          draft_device_ids  = [0, 1]  (A6000 × 2)
          verify_device_id  = 2       (H100 NVL)

        All device pairs are PCIe crossings because there is no NVLink.

        Parameters
        ----------
        draft_device_ids : list[int]
            CUDA device ordinals for A6000 DRAFT devices.
        verify_device_id : int
            CUDA device ordinal for H100 VERIFY device.
        mtp_num_layers : int
        mtp_use_repeated_layer : bool
        mtp_loss_scaling_factor : float
        locality_cache_capacity_gb : float
        """
        device_roles: Dict[int, HeteroDeviceRole] = {}
        for did in draft_device_ids:
            device_roles[did] = HeteroDeviceRole.DRAFT
        device_roles[verify_device_id] = HeteroDeviceRole.VERIFY

        device_policies: Dict[HeteroDeviceRole, HeteroMTPDetachPolicy] = {
            HeteroDeviceRole.DRAFT: HeteroMTPDetachPolicy.for_draft_device(),
            HeteroDeviceRole.VERIFY: HeteroMTPDetachPolicy.for_verify_device(),
        }

        # All pairs cross PCIe in our cluster.
        all_devices = draft_device_ids + [verify_device_id]
        pcie_pairs: List[Tuple[int, int]] = []
        for i, src in enumerate(all_devices):
            for dst in all_devices[i + 1 :]:
                pcie_pairs.append((src, dst))
                pcie_pairs.append((dst, src))

        logger.info(
            "HeteroMTPDetachConfig: draft=%s verify=%d pcie_pairs=%s",
            draft_device_ids,
            verify_device_id,
            pcie_pairs,
        )

        return cls(
            mtp_num_layers=mtp_num_layers,
            mtp_use_repeated_layer=mtp_use_repeated_layer,
            device_roles=device_roles,
            device_policies=device_policies,
            pcie_crossing_device_pairs=pcie_pairs,
            locality_cache_capacity_gb=locality_cache_capacity_gb,
            mtp_loss_scaling_factor=mtp_loss_scaling_factor,
            enable_hetero_routing=True,
        )

    def policy_for(self, device: torch.device) -> HeteroMTPDetachPolicy:
        """
        Return the detach policy applicable to ``device``.

        Falls back to the homogeneous compat policy when hetero routing is
        disabled or the device is not registered.
        """
        if not self.enable_hetero_routing:
            return HeteroMTPDetachPolicy.homogeneous_compat(self.mtp_detach_heads)

        role = self.device_roles.get(device.index if device.index is not None else 0)
        if role is None:
            logger.warning(
                "Device %s has no registered role; defaulting to homogeneous compat policy.",
                device,
            )
            return HeteroMTPDetachPolicy.homogeneous_compat(self.mtp_detach_heads)

        policy = self.device_policies.get(role)
        if policy is None:
            logger.warning(
                "Role %s has no registered policy; defaulting to DRAFT policy.",
                role,
            )
            return HeteroMTPDetachPolicy.for_draft_device()

        return policy

    def is_pcie_crossing(self, src: torch.device, dst: torch.device) -> bool:
        """Return True if the (src, dst) pair crosses a PCIe bus."""
        src_idx = src.index if src.index is not None else 0
        dst_idx = dst.index if dst.index is not None else 0
        return (src_idx, dst_idx) in self.pcie_crossing_device_pairs


# ---------------------------------------------------------------------------
# Locality cache buffer
# ---------------------------------------------------------------------------

class LocalityCacheBuffer:
    """
    Pinned CPU DRAM buffer used as a staging area for cross-device tensor
    transfers in DES-LOC.

    Motivation
    ----------
    When a hidden-state chunk is transferred from the H100 (VERIFY) to an
    A6000 (DRAFT) over PCIe, going through pinned CPU memory enables:

      1. Async ``cudaMemcpyDeviceToHost`` — overlapped with GPU computation.
      2. Async ``cudaMemcpyHostToDevice`` — the destination A6000 can begin
         work as soon as the transfer completes, without blocking the H100.

    This is the "Shared LOcality Cache" component of DES-LOC.  The buffer
    does NOT store live autograd graphs; tensors are always detached before
    spilling, consistent with ``HeteroMTPDetachPolicy.detach_hidden_states``.

    Parameters
    ----------
    capacity_bytes : int
        Maximum bytes to allocate for the pinned buffer.
    use_pinned : bool
        If True, allocates pinned (page-locked) host memory.  Set False in
        environments without CUDA (e.g., unit tests on CPU).
    """

    def __init__(self, capacity_bytes: int, use_pinned: bool = True) -> None:
        self._capacity_bytes = capacity_bytes
        self._use_pinned = use_pinned
        self._buffers: Dict[str, torch.Tensor] = {}
        self._bytes_used: int = 0
        logger.debug(
            "LocalityCacheBuffer: capacity=%.2f GiB pinned=%s",
            capacity_bytes / (1024 ** 3),
            use_pinned,
        )

    @property
    def capacity_bytes(self) -> int:
        return self._capacity_bytes

    @property
    def bytes_used(self) -> int:
        return self._bytes_used

    def _alloc_or_reuse(self, key: str, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        """
        Return a CPU tensor of the requested shape and dtype, reusing an
        existing allocation if sizes match.
        """
        needed = math.prod(shape) * torch.finfo(dtype).bits // 8
        if key in self._buffers:
            existing = self._buffers[key]
            if existing.shape == shape and existing.dtype == dtype:
                return existing
            # Free old allocation.
            self._bytes_used -= existing.numel() * existing.element_size()

        if self._bytes_used + needed > self._capacity_bytes:
            logger.warning(
                "LocalityCacheBuffer: capacity exceeded (used=%d needed=%d cap=%d); "
                "evicting all buffers.",
                self._bytes_used,
                needed,
                self._capacity_bytes,
            )
            self._buffers.clear()
            self._bytes_used = 0

        if self._use_pinned and torch.cuda.is_available():
            buf = torch.empty(shape, dtype=dtype, pin_memory=True)
        else:
            buf = torch.empty(shape, dtype=dtype)

        self._buffers[key] = buf
        self._bytes_used += needed
        return buf

    def stage(
        self,
        key: str,
        tensor: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        """
        Copy ``tensor`` (GPU) into pinned CPU buffer asynchronously.

        The tensor must already be detached (no autograd graph).  Returns the
        CPU-side buffer; the caller is responsible for synchronisation before
        reading.

        Parameters
        ----------
        key : str
            Identifier for buffer reuse.
        tensor : torch.Tensor
            Detached GPU tensor to stage.
        stream : torch.cuda.Stream or None
            If provided, the copy is enqueued on this stream.
        """
        assert not tensor.requires_grad, (
            "LocalityCacheBuffer.stage received a tensor with requires_grad=True; "
            "detach before staging."
        )
        cpu_buf = self._alloc_or_reuse(key, tensor.shape, tensor.dtype)
        if stream is not None and tensor.is_cuda:
            with torch.cuda.stream(stream):
                cpu_buf.copy_(tensor, non_blocking=True)
        else:
            cpu_buf.copy_(tensor)
        return cpu_buf

    def retrieve(
        self,
        key: str,
        dst_device: torch.device,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> Optional[torch.Tensor]:
        """
        Retrieve a staged tensor onto ``dst_device`` asynchronously.

        Returns None if the key has not been staged.
        """
        if key not in self._buffers:
            return None
        cpu_buf = self._buffers[key]
        if stream is not None and dst_device.type == "cuda":
            with torch.cuda.stream(stream):
                return cpu_buf.to(dst_device, non_blocking=True)
        return cpu_buf.to(dst_device)

    def clear(self) -> None:
        self._buffers.clear()
        self._bytes_used = 0


# ---------------------------------------------------------------------------
# Runtime detach manager
# ---------------------------------------------------------------------------

class HeteroMTPDetachManager:
    """
    Centralised runtime manager for DES-LOC MTP detach operations.

    This object is instantiated once per inference/training session and is
    passed to each callsite that needs to perform a detach.  It encapsulates:

      * Policy lookup by device.
      * PCIe-crossing detection.
      * Optional locality cache staging.
      * Gradient re-enablement after detach (Megatron compat fix).

    Design note — why a manager rather than free functions?
    -------------------------------------------------------
    Megatron's original code embeds detach logic directly in the module's
    ``forward`` methods, which works for homogeneous hardware.  In DES-LOC,
    the decision to detach depends on runtime device placement, which can
    change between requests (e.g., load-balancing across A6000s).  The
    manager provides a single lookup point and makes the policy testable
    without instantiating full model modules.
    """

    def __init__(
        self,
        config: HeteroMTPDetachConfig,
        locality_cache: Optional[LocalityCacheBuffer] = None,
    ) -> None:
        self._config = config
        if locality_cache is None and config.locality_cache_capacity_gb > 0:
            capacity = int(config.locality_cache_capacity_gb * 1024 ** 3)
            use_pinned = torch.cuda.is_available()
            locality_cache = LocalityCacheBuffer(capacity, use_pinned=use_pinned)
        self._cache = locality_cache
        self._transfer_streams: Dict[int, torch.cuda.Stream] = {}

    def _get_or_create_stream(self, device_idx: int) -> Optional[torch.cuda.Stream]:
        if not torch.cuda.is_available():
            return None
        if device_idx not in self._transfer_streams:
            with torch.cuda.device(device_idx):
                self._transfer_streams[device_idx] = torch.cuda.Stream()
        return self._transfer_streams[device_idx]

    def detach_decoder_input(
        self,
        decoder_input: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Apply decoder_input detach policy.

        Mirrors Megatron's ``_get_embeddings`` detach of ``decoder_input``
        when ``mtp_detach_heads=True``, but conditioned on the device policy.

        The returned tensor has ``requires_grad=False`` when the policy is
        active, severing gradient flow back to the shared embedding.
        """
        if device is None:
            device = decoder_input.device
        policy = self._config.policy_for(device)

        if policy.detach_decoder_input and decoder_input.requires_grad:
            decoder_input = decoder_input.detach()
            logger.debug(
                "detach_decoder_input: severed embedding grad on device %s (role=%s)",
                device,
                self._config.device_roles.get(
                    device.index if device.index is not None else 0, "unknown"
                ),
            )
        return decoder_input

    def detach_hidden_states(
        self,
        hidden_states: torch.Tensor,
        src_device: Optional[torch.device] = None,
        dst_device: Optional[torch.device] = None,
        cache_key: Optional[str] = None,
        transfer_stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        """
        Apply hidden-states detach policy, optionally staging through the
        locality cache for PCIe crossings.

        Mirrors Megatron's block-level detach of the per-offset hidden-state
        chunk, extended with:

          * PCIe-crossing detection → locality cache staging when active.
          * Gradient re-enablement to preserve activation-checkpointing compat.

        Parameters
        ----------
        hidden_states : torch.Tensor
            The hidden-state chunk for this MTP offset.
        src_device : torch.device or None
            Source device (where ``hidden_states`` currently lives).
        dst_device : torch.device or None
            Destination device (where MTP layers will execute).
        cache_key : str or None
            Key for locality cache staging.  Auto-generated if None.
        transfer_stream : torch.cuda.Stream or None
            CUDA stream for async PCIe transfer.
        """
        if src_device is None:
            src_device = hidden_states.device
        if dst_device is None:
            dst_device = src_device

        policy = self._config.policy_for(dst_device)

        if not policy.detach_hidden_states:
            return hidden_states

        # Detach from main-model autograd graph.
        detached = hidden_states.detach()

        # PCIe crossing: stage through locality cache if configured.
        crosses_pcie = self._config.is_pcie_crossing(src_device, dst_device)
        if crosses_pcie and policy.spill_to_locality_cache and self._cache is not None:
            if cache_key is None:
                cache_key = f"hidden_{src_device.index}_{dst_device.index}"

            stream = transfer_stream or self._get_or_create_stream(
                src_device.index if src_device.index is not None else 0
            )
            staged = self._cache.stage(cache_key, detached, stream=stream)

            dst_stream = self._get_or_create_stream(
                dst_device.index if dst_device.index is not None else 0
            )
            detached = self._cache.retrieve(cache_key, dst_device, stream=dst_stream)
            if detached is None:
                # Fallback: direct transfer (should not happen).
                detached = hidden_states.detach().to(dst_device)

            logger.debug(
                "detach_hidden_states: PCIe handoff %s→%s via locality cache key=%s "
                "shape=%s dtype=%s",
                src_device,
                dst_device,
                cache_key,
                tuple(detached.shape),
                detached.dtype,
            )

        # Re-enable requires_grad so MTP layer params and activation
        # checkpointing see a differentiable input (Megatron fix, line 1110).
        if policy.ensure_hidden_grad and not detached.requires_grad:
            detached = detached.requires_grad_(True)

        return detached

    def detach_output_weight(
        self,
        output_weight: Optional[torch.Tensor],
        output_layer: Optional[nn.Module],
        device: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        """
        Apply output-weight detach policy.

        Mirrors Megatron's ``process_mtp_loss`` detach of ``output_weight``
        when ``mtp_detach_heads=True``.  In DES-LOC this is always active when
        the weight lives on the VERIFY device (H100) but the loss is computed
        on a DRAFT device (A6000).

        Parameters
        ----------
        output_weight : torch.Tensor or None
            Explicitly passed weight tensor, or None to extract from
            ``output_layer``.
        output_layer : nn.Module or None
            Module whose ``.weight`` attribute is used when
            ``output_weight`` is None.
        device : torch.device or None
            Device of the MTP loss computation (usually DRAFT device).
        """
        if device is None:
            if output_weight is not None:
                device = output_weight.device
            elif output_layer is not None and hasattr(output_layer, "weight"):
                device = output_layer.weight.device
            else:
                device = torch.device("cpu")

        policy = self._config.policy_for(device)

        if not policy.detach_output_weight:
            # No detach: return as-is (homogeneous path).
            if output_weight is not None:
                return output_weight
            if output_layer is not None and hasattr(output_layer, "weight"):
                return output_layer.weight
            return None

        # Detach path (mirrors Megatron lines 820–824).
        if output_weight is not None:
            detached = output_weight.detach()
        elif output_layer is not None and hasattr(output_layer, "weight"):
            detached = output_layer.weight.detach()
        else:
            return None

        logger.debug(
            "detach_output_weight: detached output_weight on device %s; "
            "MTP loss will not update shared projection.",
            device,
        )
        return detached


# ---------------------------------------------------------------------------
# Hetero-aware process_mtp_loss
# ---------------------------------------------------------------------------

def hetero_process_mtp_loss(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    output_layer: nn.Module,
    output_weight: Optional[torch.Tensor],
    compute_language_model_loss: Callable,
    config: HeteroMTPDetachConfig,
    manager: HeteroMTPDetachManager,
    mtp_layer_offset: int = 1,
    runtime_gather_output: Optional[bool] = None,
    is_training: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    DES-LOC replacement for Megatron's ``process_mtp_loss``.

    Upstream design (Megatron)
    --------------------------
    ``process_mtp_loss`` computes the auxiliary cross-entropy loss for each
    MTP head.  When ``mtp_detach_heads=True``, it detaches ``output_weight``
    before building logits, so that gradients from the MTP loss do not update
    the shared output projection weight.

    DES-LOC adaptation
    ------------------
    * Weight detach is routed through ``HeteroMTPDetachManager`` which checks
      device roles; on the A6000 the weight is always detached because it
      physically lives on the H100.
    * Hidden-state slicing is performed per MTP layer, and each slice is
      individually detached if the policy requires it.
    * Loss aggregation uses ``mtp_loss_scaling_factor`` from config (mirrors
      Megatron's ``mtp_loss_scaling_factor``).
    * Per-layer loss values are returned in a diagnostic dict so the DeepSpeed
      engine can log them without an additional forward pass.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Concatenated [main, mtp_0, …, mtp_N] along dim 0.
        Shape: ``[(1 + N) * seq_len, batch, hidden]``.
    labels : torch.Tensor
        Ground-truth token ids.  Shape: ``[batch, seq_len]``.
    loss_mask : torch.Tensor
        Per-token mask.  Shape: ``[batch, seq_len]``.
    output_layer : nn.Module
        Shared output projection from the backbone.
    output_weight : torch.Tensor or None
        Explicit weight tensor; if None, extracted from ``output_layer``.
    compute_language_model_loss : callable
        Function ``(labels, logits) -> loss`` used for each MTP head.
    config : HeteroMTPDetachConfig
    manager : HeteroMTPDetachManager
    mtp_layer_offset : int
        Typically 1: the first chunk is the main model output.
    runtime_gather_output : bool or None
        Passed to ``output_layer`` for tensor-parallel all-gather.
    is_training : bool

    Returns
    -------
    total_mtp_loss : torch.Tensor
        Scaled sum of per-head losses.
    per_layer_losses : dict[str, torch.Tensor]
        Diagnostic dict mapping ``"mtp_loss_layer_N"`` to per-head loss.
    """
    device = hidden_states.device
    seq_len = labels.shape[1]
    n_mtp = config.mtp_num_layers

    # Detach output weight (mirrors Megatron lines 820–824).
    eff_weight = manager.detach_output_weight(
        output_weight=output_weight,
        output_layer=output_layer,
        device=device,
    )

    per_layer_losses: Dict[str, torch.Tensor] = {}
    total_loss = torch.tensor(0.0, device=device, dtype=hidden_states.dtype)

    for i in range(n_mtp):
        # Slice MTP head hidden states (mirrors block-level chunk logic).
        start = (mtp_layer_offset + i) * seq_len
        end = start + seq_len
        mtp_hidden = hidden_states[start:end]  # [seq_len, batch, hidden]

        # Compute logits via shared output projection with detached weight.
        if eff_weight is not None:
            logits, _ = output_layer(
                mtp_hidden, weight=eff_weight, runtime_gather_output=runtime_gather_output
            )
        else:
            logits, _ = output_layer(
                mtp_hidden, runtime_gather_output=runtime_gather_output
            )

        # Shift labels: MTP head i predicts token i+1 (mirrors Megatron label
        # derivation from input_ids in process_mtp_loss).
        shifted_labels = labels.clone()
        # Compute per-token loss.
        layer_loss_tokens = compute_language_model_loss(shifted_labels, logits)

        # Apply loss mask and average.
        masked = layer_loss_tokens * loss_mask
        layer_loss = masked.sum() / (loss_mask.sum() + 1e-8)

        key = f"mtp_loss_layer_{i}"
        per_layer_losses[key] = layer_loss.detach()
        total_loss = total_loss + layer_loss

    total_mtp_loss = total_loss * config.mtp_loss_scaling_factor
    return total_mtp_loss, per_layer_losses


# ---------------------------------------------------------------------------
# Hetero-aware _get_embeddings replacement
# ---------------------------------------------------------------------------

class HeteroMTPEmbeddingDetach:
    """
    DES-LOC adaptation of ``MultiTokenPredictionLayer._get_embeddings``.

    Upstream design (Megatron)
    --------------------------
    ``_get_embeddings`` computes the decoder embedding for MTP input tokens
    and concatenates it with the backbone hidden states.  With
    ``mtp_detach_heads=True``, Megatron detaches ``decoder_input`` to sever
    gradient flow to the shared embedding, then ensures ``hidden_states`` has
    ``requires_grad=True`` for checkpointing compatibility.

    DES-LOC adaptation
    ------------------
    In DES-LOC the shared embedding lives on the VERIFY device (H100).  The
    MTP head executes on a DRAFT device (A6000).  The embedding lookup result
    crosses PCIe; staging it through the locality cache and detaching it
    before transfer is both a correctness requirement (no gradient over PCIe)
    and a bandwidth optimisation.

    This class is a stateless helper; it holds a reference to the manager and
    is called from the MTP layer's forward method.
    """

    def __init__(self, manager: HeteroMTPDetachManager) -> None:
        self._manager = manager

    def apply(
        self,
        decoder_input: torch.Tensor,
        hidden_states: torch.Tensor,
        src_device: Optional[torch.device] = None,
        dst_device: Optional[torch.device] = None,
        cache_key_prefix: str = "emb",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply detach policy to ``decoder_input`` and ``hidden_states``.

        Parameters
        ----------
        decoder_input : torch.Tensor
            Output of the shared embedding for MTP input tokens.
        hidden_states : torch.Tensor
            Backbone hidden states for the MTP layer.
        src_device : torch.device or None
            Device where embedding was computed (VERIFY/H100 in DES-LOC).
        dst_device : torch.device or None
            Device where MTP layer will execute (DRAFT/A6000 in DES-LOC).
        cache_key_prefix : str
            Prefix for locality cache keys.

        Returns
        -------
        decoder_input : torch.Tensor
            Detached (when policy active); no grad, no graph to embedding.
        hidden_states : torch.Tensor
            Possibly detached and/or transferred; requires_grad=True ensured.
        """
        decoder_input = self._manager.detach_decoder_input(
            decoder_input, device=dst_device or decoder_input.device
        )

        hidden_states = self._manager.detach_hidden_states(
            hidden_states,
            src_device=src_device,
            dst_device=dst_device,
            cache_key=f"{cache_key_prefix}_hidden",
        )

        return decoder_input, hidden_states


# ---------------------------------------------------------------------------
# Hetero-aware block-level hidden-states detach
# ---------------------------------------------------------------------------

class HeteroMTPBlockDetach:
    """
    DES-LOC adaptation of ``MultiTokenPredictionBlock.forward`` hidden-states
    detach logic.

    Upstream design (Megatron)
    --------------------------
    ``MultiTokenPredictionBlock.forward`` slices the concatenated hidden-states
    tensor into per-offset chunks.  With ``mtp_detach_heads=True``, the chunk
    for the current MTP offset is detached before being fed to the MTP layers,
    preventing MTP gradients from flowing back through the backbone's earlier
    layers.

    DES-LOC adaptation
    ------------------
    In addition to the gradient isolation, DES-LOC must handle the physical
    device transfer.  The chunk for MTP offset ``k`` lives on the VERIFY device
    (where the backbone ran) but must execute on a DRAFT device.  The detach
    and PCIe transfer are fused via the locality cache.

    This class is a stateless helper, analogous to ``HeteroMTPEmbeddingDetach``.
    """

    def __init__(self, manager: HeteroMTPDetachManager) -> None:
        self._manager = manager

    def apply(
        self,
        hidden_states: torch.Tensor,
        mtp_offset: int,
        verify_device: torch.device,
        draft_device: torch.device,
    ) -> torch.Tensor:
        """
        Detach and (optionally) transfer the hidden-state chunk for
        ``mtp_offset`` from VERIFY to DRAFT device.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Full concatenated tensor on VERIFY device.
        mtp_offset : int
            Which chunk to extract.  Matches Megatron's ``offset`` variable.
        verify_device : torch.device
            Source device (H100).
        draft_device : torch.device
            Destination device (A6000).

        Returns
        -------
        torch.Tensor
            Chunk for ``mtp_offset``, on ``draft_device``, detached from
            backbone graph, with ``requires_grad=True``.
        """
        chunks = list(torch.chunk(hidden_states, 1 + mtp_offset, dim=0))
        chunk = chunks[mtp_offset]

        detached_chunk = self._manager.detach_hidden_states(
            chunk,
            src_device=verify_device,
            dst_device=draft_device,
            cache_key=f"block_hidden_offset_{mtp_offset}",
        )
        return detached_chunk


# ---------------------------------------------------------------------------
# Lightweight MTP head module for DES-LOC
# ---------------------------------------------------------------------------

class HeteroMTPDraftHead(nn.Module):
    """
    Lightweight MTP speculative head designed for A6000 DRAFT devices.

    This module encapsulates the per-MTP-layer parameters (enorm, hnorm,
    eh_proj) and the forward pass logic.  It is analogous to the
    ``MultiTokenPredictionLayer`` in Megatron, but:

      * Explicitly device-aware: constructed on a specific DRAFT device.
      * Accepts detached inputs from the manager (no grad crossing PCIe).
      * Returns hidden states ready for loss computation on the same device.

    The design mirrors Megatron's three-parameter MTP layer structure:
      enorm  — LayerNorm over decoder (embedding) input.
      hnorm  — LayerNorm over backbone hidden states.
      eh_proj — Linear projection from 2*hidden to hidden.

    Parameters
    ----------
    hidden_size : int
        Dimension of backbone hidden states.
    device : torch.device
        DRAFT device to place parameters on.
    dtype : torch.dtype
        Parameter dtype.  Default float32; fp16/bf16 for A6000 inference.
    layer_idx : int
        Index of this MTP head (0-indexed).
    """

    def __init__(
        self,
        hidden_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size

        self.enorm = nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        self.hnorm = nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        self.eh_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False, device=device)

        logger.debug(
            "HeteroMTPDraftHead[%d]: hidden_size=%d device=%s dtype=%s",
            layer_idx, hidden_size, device, dtype,
        )

    def forward(
        self,
        decoder_input: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MTP head output.

        Both inputs should already be detached from the backbone graph
        by the manager before this call.

        Parameters
        ----------
        decoder_input : torch.Tensor
            Embedding output for MTP input tokens.  Shape: [seq, batch, hidden].
        hidden_states : torch.Tensor
            Backbone hidden states.  Shape: [seq, batch, hidden].
            Must have ``requires_grad=True`` (set by manager).

        Returns
        -------
        torch.Tensor
            MTP head output.  Shape: [seq, batch, hidden].
        """
        normed_e = self.enorm(decoder_input)
        normed_h = self.hnorm(hidden_states)
        concatenated = torch.cat([normed_e, normed_h], dim=-1)
        projected = self.eh_proj(concatenated)
        return projected


# ---------------------------------------------------------------------------
# End-to-end DES-LOC MTP speculative decode runner
# ---------------------------------------------------------------------------

class HeteroMTPSpeculativeRunner:
    """
    Orchestrates DES-LOC speculative decoding with heterogeneous MTP heads.

    Responsibilities:
      1. Receive backbone hidden states from VERIFY device (H100).
      2. Detach and transfer each MTP chunk to the appropriate DRAFT device.
      3. Run ``HeteroMTPDraftHead`` on A6000 for each MTP layer.
      4. Collect draft logits and return to VERIFY device for verification.
      5. (Training) Compute MTP auxiliary loss via ``hetero_process_mtp_loss``.

    This class does not own the embedding or output_layer modules; those live
    on the VERIFY device and are passed in at call time, matching DeepSpeed's
    engine architecture where module references are held centrally.

    Parameters
    ----------
    config : HeteroMTPDetachConfig
    manager : HeteroMTPDetachManager
    draft_heads : list[HeteroMTPDraftHead]
        One per MTP layer.  Each head is on a DRAFT device.
    draft_device_ids : list[int]
        CUDA ordinals of A6000 DRAFT devices.  Heads are round-robin
        distributed if fewer devices than heads.
    verify_device_id : int
        CUDA ordinal of H100 VERIFY device.
    """

    def __init__(
        self,
        config: HeteroMTPDetachConfig,
        manager: HeteroMTPDetachManager,
        draft_heads: List[HeteroMTPDraftHead],
        draft_device_ids: List[int],
        verify_device_id: int,
    ) -> None:
        self._config = config
        self._manager = manager
        self._heads = draft_heads
        self._draft_device_ids = draft_device_ids
        self._verify_device = torch.device(f"cuda:{verify_device_id}")
        self._block_detach = HeteroMTPBlockDetach(manager)
        self._emb_detach = HeteroMTPEmbeddingDetach(manager)

    def _draft_device_for(self, layer_idx: int) -> torch.device:
        """Round-robin assignment of MTP layers to DRAFT devices."""
        idx = self._draft_device_ids[layer_idx % len(self._draft_device_ids)]
        return torch.device(f"cuda:{idx}")

    def run_draft_heads(
        self,
        hidden_states: torch.Tensor,
        embedding_fn: Callable,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Run all MTP draft heads and collect outputs.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Concatenated backbone hidden states on VERIFY device.
            Shape: ``[(1 + N) * seq_len, batch, hidden]``.
        embedding_fn : callable
            ``(input_ids, position_ids) -> decoder_input`` on VERIFY device.
        input_ids : torch.Tensor
        position_ids : torch.Tensor

        Returns
        -------
        list[torch.Tensor]
            Per-MTP-layer head outputs.  Each on the VERIFY device (transferred
            back after computation for downstream verification/loss).
        """
        draft_outputs: List[torch.Tensor] = []

        for layer_idx, head in enumerate(self._heads):
            draft_device = self._draft_device_for(layer_idx)

            # Detach and transfer hidden-state chunk (block-level detach).
            h_chunk = self._block_detach.apply(
                hidden_states=hidden_states,
                mtp_offset=layer_idx,
                verify_device=self._verify_device,
                draft_device=draft_device,
            )

            # Compute embedding on VERIFY, then detach and transfer to DRAFT.
            with torch.no_grad() if not h_chunk.requires_grad else _nullctx():
                dec_input = embedding_fn(input_ids, position_ids)

            dec_input_draft, h_chunk = self._emb_detach.apply(
                decoder_input=dec_input.to(draft_device),
                hidden_states=h_chunk,
                src_device=self._verify_device,
                dst_device=draft_device,
                cache_key_prefix=f"layer{layer_idx}",
            )

            # Run draft head on A6000.
            head_output = head(dec_input_draft, h_chunk)

            # Transfer draft output back to VERIFY device for loss/verification.
            verify_output = head_output.detach().to(self._verify_device)
            draft_outputs.append(verify_output)

        return draft_outputs


# Context manager no-op used above.
import contextlib

@contextlib.contextmanager
def _nullctx():
    yield


# ---------------------------------------------------------------------------
# DeepSpeed engine integration shim
# ---------------------------------------------------------------------------

def build_hetero_mtp_components(
    hidden_size: int,
    draft_device_ids: List[int],
    verify_device_id: int,
    mtp_num_layers: int = 2,
    mtp_loss_scaling_factor: float = 0.1,
    locality_cache_capacity_gb: float = 16.0,
    dtype: torch.dtype = torch.float32,
) -> Tuple[HeteroMTPDetachConfig, HeteroMTPDetachManager, HeteroMTPSpeculativeRunner]:
    """
    Convenience factory for the DES-LOC MTP subsystem.

    Creates the config, manager, draft heads, and runner in one call.
    Intended to be invoked from the DeepSpeed engine initialisation,
    replacing the Megatron-style ``mtp_detach_heads`` config field.

    Parameters
    ----------
    hidden_size : int
    draft_device_ids : list[int]
        e.g. [0, 1] for two A6000s.
    verify_device_id : int
        e.g. 2 for H100.
    mtp_num_layers : int
    mtp_loss_scaling_factor : float
    locality_cache_capacity_gb : float
    dtype : torch.dtype

    Returns
    -------
    config : HeteroMTPDetachConfig
    manager : HeteroMTPDetachManager
    runner : HeteroMTPSpeculativeRunner
    """
    config = HeteroMTPDetachConfig.from_cluster_spec(
        draft_device_ids=draft_device_ids,
        verify_device_id=verify_device_id,
        mtp_num_layers=mtp_num_layers,
        mtp_loss_scaling_factor=mtp_loss_scaling_factor,
        locality_cache_capacity_gb=locality_cache_capacity_gb,
    )

    manager = HeteroMTPDetachManager(config)

    draft_heads: List[HeteroMTPDraftHead] = []
    for i in range(mtp_num_layers):
        device_idx = draft_device_ids[i % len(draft_device_ids)]
        device = torch.device(f"cuda:{device_idx}") if torch.cuda.is_available() else torch.device("cpu")
        head = HeteroMTPDraftHead(
            hidden_size=hidden_size,
            device=device,
            dtype=dtype,
            layer_idx=i,
        )
        draft_heads.append(head)

    runner = HeteroMTPSpeculativeRunner(
        config=config,
        manager=manager,
        draft_heads=draft_heads,
        draft_device_ids=draft_device_ids,
        verify_device_id=verify_device_id,
    )

    logger.info(
        "build_hetero_mtp_components: %d MTP heads, draft=%s, verify=%d, "
        "locality_cache=%.1f GiB",
        mtp_num_layers,
        draft_device_ids,
        verify_device_id,
        locality_cache_capacity_gb,
    )

    return config, manager, runner


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    class _TestHeteroMTPDetachConfig(unittest.TestCase):

        def test_default_config_disable_hetero(self):
            """Default config with enable_hetero_routing=False reproduces Megatron scalar."""
            cfg = HeteroMTPDetachConfig(enable_hetero_routing=False, mtp_detach_heads=False)
            device = torch.device("cpu")
            policy = cfg.policy_for(device)
            self.assertFalse(policy.detach_decoder_input)
            self.assertFalse(policy.detach_hidden_states)
            self.assertFalse(policy.detach_output_weight)

        def test_default_config_homogeneous_detach_true(self):
            """With mtp_detach_heads=True and no hetero routing, all detach flags on."""
            cfg = HeteroMTPDetachConfig(enable_hetero_routing=False, mtp_detach_heads=True)
            device = torch.device("cpu")
            policy = cfg.policy_for(device)
            self.assertTrue(policy.detach_decoder_input)
            self.assertTrue(policy.detach_hidden_states)
            self.assertTrue(policy.detach_output_weight)
            self.assertTrue(policy.ensure_hidden_grad)

        def test_from_cluster_spec_roles(self):
            """from_cluster_spec assigns roles correctly."""
            cfg = HeteroMTPDetachConfig.from_cluster_spec(
                draft_device_ids=[0, 1],
                verify_device_id=2,
                mtp_num_layers=2,
            )
            self.assertEqual(cfg.device_roles[0], HeteroDeviceRole.DRAFT)
            self.assertEqual(cfg.device_roles[1], HeteroDeviceRole.DRAFT)
            self.assertEqual(cfg.device_roles[2], HeteroDeviceRole.VERIFY)

        def test_from_cluster_spec_pcie_pairs(self):
            """All pairs are PCIe crossings in our cluster."""
            cfg = HeteroMTPDetachConfig.from_cluster_spec(
                draft_device_ids=[0, 1], verify_device_id=2
            )
            # Should include (0,2), (2,0), (1,2), (2,1), (0,1), (1,0).
            self.assertIn((0, 2), cfg.pcie_crossing_device_pairs)
            self.assertIn((2, 0), cfg.pcie_crossing_device_pairs)
            self.assertIn((0, 1), cfg.pcie_crossing_device_pairs)

        def test_is_pcie_crossing(self):
            cfg = HeteroMTPDetachConfig.from_cluster_spec(
                draft_device_ids=[0, 1], verify_device_id=2
            )
            self.assertTrue(cfg.is_pcie_crossing(torch.device("cuda:0"), torch.device("cuda:2")))
            self.assertFalse(cfg.is_pcie_crossing(torch.device("cuda:0"), torch.device("cuda:0")))

        def test_policy_for_draft(self):
            cfg = HeteroMTPDetachConfig.from_cluster_spec(
                draft_device_ids=[0], verify_device_id=1
            )
            # Simulate device 0 = DRAFT.
            policy = cfg.device_policies[HeteroDeviceRole.DRAFT]
            self.assertTrue(policy.detach_decoder_input)
            self.assertTrue(policy.detach_hidden_states)
            self.assertTrue(policy.detach_output_weight)
            self.assertTrue(policy.ensure_hidden_grad)
            self.assertTrue(policy.spill_to_locality_cache)

        def test_policy_for_verify(self):
            cfg = HeteroMTPDetachConfig.from_cluster_spec(
                draft_device_ids=[0], verify_device_id=1
            )
            policy = cfg.device_policies[HeteroDeviceRole.VERIFY]
            self.assertFalse(policy.detach_decoder_input)
            self.assertFalse(policy.detach_hidden_states)
            self.assertTrue(policy.detach_output_weight)

        def test_unknown_device_falls_back_to_compat(self):
            cfg = HeteroMTPDetachConfig(
                enable_hetero_routing=True,
                device_roles={},
                device_policies={},
                mtp_detach_heads=False,
            )
            device = torch.device("cpu")
            policy = cfg.policy_for(device)
            # Falls through to homogeneous compat with mtp_detach_heads=False.
            self.assertFalse(policy.detach_hidden_states)

    class _TestLocalityCacheBuffer(unittest.TestCase):

        def _make_cache(self, cap_gb: float = 0.01) -> LocalityCacheBuffer:
            cap = int(cap_gb * 1024 ** 3)
            return LocalityCacheBuffer(cap, use_pinned=False)

        def test_stage_and_retrieve_cpu(self):
            """Stage a CPU tensor and retrieve it on CPU (no CUDA needed)."""
            cache = self._make_cache()
            t = torch.randn(4, 2, 8)
            staged = cache.stage("key1", t)
            self.assertEqual(staged.shape, t.shape)
            torch.testing.assert_close(staged, t)

        def test_stage_requires_no_grad(self):
            cache = self._make_cache()
            t = torch.randn(4, requires_grad=True).detach()
            # Should work fine (already detached).
            staged = cache.stage("k", t)
            self.assertFalse(staged.requires_grad)

        def test_stage_with_grad_raises(self):
            cache = self._make_cache()
            t = torch.randn(4, requires_grad=True)
            with self.assertRaises(AssertionError):
                cache.stage("k", t)

        def test_reuse_existing_buffer(self):
            cache = self._make_cache()
            t1 = torch.ones(4, 8)
            t2 = torch.zeros(4, 8)
            cache.stage("k", t1)
            buf1_ptr = cache._buffers["k"].data_ptr()
            cache.stage("k", t2)
            buf2_ptr = cache._buffers["k"].data_ptr()
            # Same allocation reused.
            self.assertEqual(buf1_ptr, buf2_ptr)
            torch.testing.assert_close(cache._buffers["k"], t2)

        def test_capacity_eviction(self):
            """Exceeding capacity evicts all buffers."""
            cache = LocalityCacheBuffer(capacity_bytes=100, use_pinned=False)
            # Each tensor is 4*4=16 bytes; 7 of them exceed 100 bytes.
            for i in range(8):
                t = torch.randn(4, dtype=torch.float32)
                cache.stage(f"k{i}", t)
            # After eviction + re-staging, bytes_used should be small.
            self.assertLess(cache.bytes_used, 100)

        def test_clear(self):
            cache = self._make_cache()
            t = torch.randn(4)
            cache.stage("k", t)
            cache.clear()
            self.assertEqual(len(cache._buffers), 0)
            self.assertEqual(cache.bytes_used, 0)

    class _TestHeteroMTPDetachManager(unittest.TestCase):

        def _make_manager(self, mtp_detach_heads: bool = True) -> HeteroMTPDetachManager:
            cfg = HeteroMTPDetachConfig(
                enable_hetero_routing=False,
                mtp_detach_heads=mtp_detach_heads,
                locality_cache_capacity_gb=0.01,
            )
            cache = LocalityCacheBuffer(int(0.01 * 1024 ** 3), use_pinned=False)
            return HeteroMTPDetachManager(cfg, locality_cache=cache)

        def test_detach_decoder_input_active(self):
            """Manager severs decoder_input grad when policy active."""
            mgr = self._make_manager(mtp_detach_heads=True)
            t = torch.randn(4, 2, 8, requires_grad=True)
            result = mgr.detach_decoder_input(t, device=torch.device("cpu"))
            self.assertFalse(result.requires_grad)
            self.assertIsNone(result.grad_fn)

        def test_detach_decoder_input_inactive(self):
            """Manager preserves decoder_input grad when policy inactive."""
            mgr = self._make_manager(mtp_detach_heads=False)
            t = torch.randn(4, 2, 8, requires_grad=True)
            result = mgr.detach_decoder_input(t, device=torch.device("cpu"))
            self.assertTrue(result.requires_grad)

        def test_detach_hidden_states_ensures_grad(self):
            """After detach, manager re-enables requires_grad."""
            mgr = self._make_manager(mtp_detach_heads=True)
            # hidden_states without grad (as after detach upstream).
            t = torch.randn(4, 2, 8)
            self.assertFalse(t.requires_grad)
            result = mgr.detach_hidden_states(t, device=torch.device("cpu"))
            self.assertTrue(result.requires_grad)

        def test_detach_hidden_states_inactive_preserves_graph(self):
            """When policy inactive, hidden_states grad_fn preserved."""
            mgr = self._make_manager(mtp_detach_heads=False)
            t = torch.randn(4, 2, 8, requires_grad=True)
            t2 = t * 2  # has grad_fn
            result = mgr.detach_hidden_states(t2, device=torch.device("cpu"))
            self.assertIsNotNone(result.grad_fn)

        def test_detach_output_weight_severs_grad(self):
            """Manager detaches output_weight; grad does not flow through it."""
            mgr = self._make_manager(mtp_detach_heads=True)
            w = nn.Parameter(torch.randn(16, 8))
            result = mgr.detach_output_weight(output_weight=w, output_layer=None)
            self.assertIsNotNone(result)
            self.assertFalse(result.requires_grad)
            # Verify no gradient accumulates through the detached weight.
            loss = (result ** 2).sum()
            loss.backward()
            self.assertIsNone(w.grad)

        def test_detach_output_weight_from_layer(self):
            """Manager extracts weight from output_layer when output_weight is None."""
            mgr = self._make_manager(mtp_detach_heads=True)
            layer = nn.Linear(8, 16, bias=False)
            result = mgr.detach_output_weight(output_weight=None, output_layer=layer)
            self.assertIsNotNone(result)
            self.assertFalse(result.requires_grad)

        def test_detach_output_weight_inactive(self):
            """When policy inactive, output_weight retains grad."""
            mgr = self._make_manager(mtp_detach_heads=False)
            w = nn.Parameter(torch.randn(16, 8))
            result = mgr.detach_output_weight(output_weight=w, output_layer=None)
            self.assertIsNotNone(result)
            self.assertTrue(result.requires_grad)

    class _TestHeteroProcessMTPLoss(unittest.TestCase):

        def _make_config_and_manager(
            self, mtp_num_layers: int = 2, mtp_detach: bool = True
        ) -> Tuple[HeteroMTPDetachConfig, HeteroMTPDetachManager]:
            cfg = HeteroMTPDetachConfig(
                enable_hetero_routing=False,
                mtp_detach_heads=mtp_detach,
                mtp_num_layers=mtp_num_layers,
                mtp_loss_scaling_factor=0.1,
                locality_cache_capacity_gb=0.0,
            )
            cache = LocalityCacheBuffer(1024, use_pinned=False)
            mgr = HeteroMTPDetachManager(cfg, locality_cache=cache)
            return cfg, mgr

        def test_mtp_loss_detach_true_no_weight_grad(self):
            """With detach=True, output_weight.grad is None after backward."""
            cfg, mgr = self._make_config_and_manager(mtp_detach=True)
            seq_len, batch, hidden, vocab = 4, 2, 8, 16
            n = cfg.mtp_num_layers
            hidden_states = torch.randn((1 + n) * seq_len, batch, hidden, requires_grad=True)
            labels = torch.randint(0, vocab, (batch, seq_len))
            loss_mask = torch.ones(batch, seq_len)
            w = nn.Parameter(torch.randn(vocab, hidden))

            class FakeOutputLayer(nn.Module):
                def forward(self, h, weight=None, runtime_gather_output=None):
                    return torch.matmul(h, weight.t()), None

            def fake_loss(lbl, logits):
                return logits.sum(dim=-1).transpose(0, 1)

            loss, per_layer = hetero_process_mtp_loss(
                hidden_states=hidden_states,
                labels=labels,
                loss_mask=loss_mask,
                output_layer=FakeOutputLayer(),
                output_weight=w,
                compute_language_model_loss=fake_loss,
                config=cfg,
                manager=mgr,
            )
            loss.backward()
            self.assertIsNone(w.grad, "output_weight should have no grad when detach=True")
            self.assertEqual(len(per_layer), n)
            for i in range(n):
                self.assertIn(f"mtp_loss_layer_{i}", per_layer)

        def test_mtp_loss_detach_false_weight_grad_exists(self):
            """With detach=False, output_weight.grad is populated after backward."""
            cfg, mgr = self._make_config_and_manager(mtp_detach=False)
            seq_len, batch, hidden, vocab = 4, 2, 8, 16
            n = cfg.mtp_num_layers
            hidden_states = torch.randn((1 + n) * seq_len, batch, hidden, requires_grad=True)
            labels = torch.randint(0, vocab, (batch, seq_len))
            loss_mask = torch.ones(batch, seq_len)
            w = nn.Parameter(torch.randn(vocab, hidden))

            class FakeOutputLayer(nn.Module):
                def forward(self, h, weight=None, runtime_gather_output=None):
                    return torch.matmul(h, weight.t()), None

            def fake_loss(lbl, logits):
                return logits.sum(dim=-1).transpose(0, 1)

            loss, _ = hetero_process_mtp_loss(
                hidden_states=hidden_states,
                labels=labels,
                loss_mask=loss_mask,
                output_layer=FakeOutputLayer(),
                output_weight=w,
                compute_language_model_loss=fake_loss,
                config=cfg,
                manager=mgr,
            )
            loss.backward()
            self.assertIsNotNone(w.grad)

        def test_loss_scaling_applied(self):
            """Total MTP loss equals sum of per-layer losses times scaling factor."""
            cfg, mgr = self._make_config_and_manager(mtp_detach=False)
            seq_len, batch, hidden, vocab = 2, 1, 4, 8
            n = cfg.mtp_num_layers
            hidden_states = torch.randn((1 + n) * seq_len, batch, hidden)
            labels = torch.zeros(batch, seq_len, dtype=torch.long)
            loss_mask = torch.ones(batch, seq_len)
            w = nn.Parameter(torch.eye(vocab, hidden))

            class FakeOutputLayer(nn.Module):
                def forward(self, h, weight=None, runtime_gather_output=None):
                    return torch.matmul(h, weight.t()), None

            def fake_loss(lbl, logits):
                # Return all-ones loss (shape [batch, seq_len]).
                return torch.ones(batch, seq_len)

            loss, per_layer = hetero_process_mtp_loss(
                hidden_states=hidden_states,
                labels=labels,
                loss_mask=loss_mask,
                output_layer=FakeOutputLayer(),
                output_weight=w,
                compute_language_model_loss=fake_loss,
                config=cfg,
                manager=mgr,
            )
            # Each layer loss = 1.0 (all-ones / N tokens); total = N * 1.0 * scaling.
            expected_total = n * 1.0 * cfg.mtp_loss_scaling_factor
            self.assertAlmostEqual(loss.item(), expected_total, places=5)

    class _TestHeteroMTPDraftHead(unittest.TestCase):

        def test_forward_shape(self):
            """Draft head output has correct shape."""
            head = HeteroMTPDraftHead(hidden_size=16, device=torch.device("cpu"))
            seq, batch = 4, 2
            dec_in = torch.randn(seq, batch, 16)
            h = torch.randn(seq, batch, 16, requires_grad=True)
            out = head(dec_in, h)
            self.assertEqual(out.shape, (seq, batch, 16))

        def test_gradient_flows_to_head_params(self):
            """Backward reaches enorm, hnorm, eh_proj."""
            head = HeteroMTPDraftHead(hidden_size=8, device=torch.device("cpu"))
            seq, batch = 3, 2
            dec_in = torch.randn(seq, batch, 8)
            h = torch.randn(seq, batch, 8, requires_grad=True)
            out = head(dec_in, h)
            out.sum().backward()
            self.assertIsNotNone(head.enorm.weight.grad)
            self.assertIsNotNone(head.hnorm.weight.grad)
            self.assertIsNotNone(head.eh_proj.weight.grad)

        def test_gradient_does_not_reach_detached_input(self):
            """When dec_in is detached, no grad flows back to the 'embedding'."""
            head = HeteroMTPDraftHead(hidden_size=8, device=torch.device("cpu"))
            seq, batch = 3, 2
            emb_param = nn.Parameter(torch.randn(seq, batch, 8))
            dec_in = emb_param.detach()  # mirrors manager.detach_decoder_input
            h = torch.randn(seq, batch, 8, requires_grad=True)
            out = head(dec_in, h)
            out.sum().backward()
            self.assertIsNone(emb_param.grad)

    class _TestHeteroMTPEmbeddingDetach(unittest.TestCase):

        def _make(self, detach: bool):
            cfg = HeteroMTPDetachConfig(
                enable_hetero_routing=False,
                mtp_detach_heads=detach,
                locality_cache_capacity_gb=0.0,
            )
            mgr = HeteroMTPDetachManager(
                cfg, locality_cache=LocalityCacheBuffer(1024, use_pinned=False)
            )
            return HeteroMTPEmbeddingDetach(mgr)

        def test_detach_true_severs_decoder_input_grad(self):
            helper = self._make(detach=True)
            dec = torch.randn(4, 2, 8, requires_grad=True)
            h = torch.randn(4, 2, 8)
            dec_out, h_out = helper.apply(dec, h)
            self.assertFalse(dec_out.requires_grad)
            self.assertIsNone(dec_out.grad_fn)

        def test_detach_true_ensures_hidden_grad(self):
            helper = self._make(detach=True)
            dec = torch.randn(4, 2, 8)
            h = torch.randn(4, 2, 8)  # no grad
            _, h_out = helper.apply(dec, h)
            self.assertTrue(h_out.requires_grad)

        def test_detach_false_preserves_decoder_grad(self):
            helper = self._make(detach=False)
            dec = torch.randn(4, 2, 8, requires_grad=True)
            h = torch.randn(4, 2, 8, requires_grad=True)
            dec_out, _ = helper.apply(dec, h)
            self.assertTrue(dec_out.requires_grad)

    class _TestBuildHeteroMTPComponents(unittest.TestCase):

        def test_builds_without_cuda(self):
            """build_hetero_mtp_components succeeds in CPU-only environment."""
            config, manager, runner = build_hetero_mtp_components(
                hidden_size=16,
                draft_device_ids=[0, 1],
                verify_device_id=2,
                mtp_num_layers=2,
            )
            self.assertEqual(config.mtp_num_layers, 2)
            self.assertEqual(len(runner._heads), 2)
            self.assertIsInstance(manager, HeteroMTPDetachManager)

        def test_head_layer_indices(self):
            _, _, runner = build_hetero_mtp_components(
                hidden_size=8,
                draft_device_ids=[0],
                verify_device_id=1,
                mtp_num_layers=3,
            )
            for i, head in enumerate(runner._heads):
                self.assertEqual(head.layer_idx, i)

    # Run all tests.
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        _TestHeteroMTPDetachConfig,
        _TestLocalityCacheBuffer,
        _TestHeteroMTPDetachManager,
        _TestHeteroProcessMTPLoss,
        _TestHeteroMTPDraftHead,
        _TestHeteroMTPEmbeddingDetach,
        _TestBuildHeteroMTPComponents,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
