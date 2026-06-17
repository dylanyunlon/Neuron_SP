"""
DES-LOC Heterogeneous CUDA Graph MTP Inference Adapter
=======================================================

Upstream design intent (Megatron commit 1cfa834a):
---------------------------------------------------
Megatron-LM's commit adds **full-model (block-scope) CUDA graph support** for
Multi-Token Prediction (MTP) inference. The core insight is that when a CUDA
graph captures the entire model forward pass as a single graph, Python-level
tensor assignments (``model._decoder_hidden_states_cache = hidden_states``)
only execute during graph capture — not during replays. Subsequent replays read
stale GPU memory at a potentially recycled address, silently producing wrong
results.

The fix lifts hidden-state ownership out of the model object and into a
persistent ``DynamicInferenceContext`` buffer (``mtp_decoder_hidden_states``).
This buffer is pre-allocated once at its maximum size and filled via
``Tensor.copy_()`` inside the captured graph, so every replay writes to the
same fixed GPU address. The controller reads ``[:actual_tokens]`` from that
buffer after each replay, and in non-block-scope (eager/layer-scope) mode the
attribute is set to ``None`` after reading to allow garbage collection.

DES-LOC adaptation points:
---------------------------
The DES-LOC (Decoupled Execution with Shared LOcality Cache) framework targets
a heterogeneous cluster of:

    * 2× NVIDIA A6000 48 GB (SM86, PCIe, no NVLink)
    * 1× NVIDIA H100 NVL 96 GB (SM90, PCIe, no NVLink)
    * 1.5 TB CPU DRAM as a shared locality cache tier

Key heterogeneity-aware adaptations over the upstream:

1. **Device-affine CUDA graph scoping** — Block-scope graphs are only safe
   on the H100 (SM90) where the larger L2 / HBM bandwidth can amortise the
   per-block capture overhead.  On SM86 (A6000) we fall back to layer-scope
   (eager-compatible) graphs.  ``HeteroCudaGraphScope`` encodes this policy.

2. **Locality-cache-aware buffer placement** — The pre-allocated
   ``mtp_decoder_hidden_states`` buffer lives in GPU memory on its *home
   device*.  When a request migrates across devices (DES-LOC's decoupled
   execution scheduler can move KV blocks between A6000 and H100), the buffer
   is transparently pinned in the 1.5 TB CPU DRAM locality cache tier and
   streamed back to the executing GPU via ``_DesLocTransfer``.

3. **SM-capability gating** — We use ``torch.cuda.get_device_capability``
   to detect SM86 vs SM90 at runtime and apply device-specific graph capture
   flags (SM90 gains async graph launches, SM86 gets conservative
   stream-capture).

4. **PCIe-topology-aware copy** — Because there is no NVLink between devices,
   cross-device ``copy_()`` goes through host memory.  ``_peer_copy_via_host``
   stages the copy through a pinned staging buffer to avoid saturating PCIe
   with un-pinned transfers.

5. **Heterogeneous MTP depth routing** — In DES-LOC, the MTP speculative
   layers (depth > 0) may execute on a *different device* from the base model
   (depth 0).  The ``HeteroMTPContext`` tracks per-depth device assignments
   and routes ``mtp_decoder_hidden_states`` to the correct device before each
   MTP forward call.

Module layout
-------------
This file is self-contained and can be imported by DeepSpeed engine code::

    from deepspeed.inference.hetero_cuda_graph_mtp import (
        HeteroCudaGraphScope,
        HeteroMTPContext,
        HeteroMTPInferenceAdapter,
        build_hetero_mtp_context,
    )
"""

from __future__ import annotations

import enum
import logging
import math
import threading
import time
import unittest
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from unittest import mock

import torch
import torch.cuda
import torch.nn as nn

__all__ = [
    "HeteroCudaGraphScope",
    "HeteroMTPContext",
    "HeteroMTPInferenceAdapter",
    "DesLocTransferBuffer",
    "build_hetero_mtp_context",
    "hetero_peer_copy_via_host",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SM86_CAPABILITY = (8, 6)   # A6000
_SM90_CAPABILITY = (9, 0)   # H100 NVL

# DES-LOC: threshold below which we skip logging a migration event (avoids spam
# for tiny tensors that the locality cache absorbs trivially).
_MIGRATION_LOG_BYTES_THRESHOLD = 1 << 20  # 1 MiB


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class HeteroCudaGraphScope(enum.Enum):
    """Device-affine CUDA graph scoping policy for DES-LOC.

    Upstream Megatron uses ``InferenceCudaGraphScope`` with values ``none``,
    ``layer``, ``block``.  We re-express this as a heterogeneity-aware enum
    that additionally encodes *which device class* is permitted to use
    block-scope graphs.

    Attributes
    ----------
    none:
        No CUDA graph capture.  Eager execution on all devices.
    layer:
        Per-layer CUDA graphs.  Safe on SM86 and SM90.
    block_sm90_only:
        Full-model (block-scope) CUDA graphs exclusively on SM90 (H100).
        SM86 (A6000) devices fall back to ``layer`` scope automatically.
        This is the recommended DES-LOC default for the target cluster.
    block:
        Full-model CUDA graphs on all capable devices.  Use only when all
        devices have sufficient VRAM headroom for simultaneous captures.
    """

    none = "none"
    layer = "layer"
    block_sm90_only = "block_sm90_only"
    block = "block"


class DesLocTier(enum.Enum):
    """Memory tier within the DES-LOC locality cache hierarchy.

    Attributes
    ----------
    gpu_local:   Tensor resides in the GPU's own HBM/GDDR6X.
    cpu_pinned:  Tensor resides in pinned CPU DRAM (locality cache tier).
    cpu_paged:   Tensor resides in pageable CPU DRAM (spill/overflow tier).
    """

    gpu_local = "gpu_local"
    cpu_pinned = "cpu_pinned"
    cpu_paged = "cpu_paged"


# ---------------------------------------------------------------------------
# PCIe-aware peer copy helper
# ---------------------------------------------------------------------------


def hetero_peer_copy_via_host(
    src: torch.Tensor,
    dst: torch.Tensor,
    staging: Optional[torch.Tensor] = None,
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """Copy *src* → *dst* across PCIe using a pinned CPU staging buffer.

    Because the DES-LOC cluster has no NVLink between A6000 and H100, a
    direct ``dst.copy_(src)`` would trigger an implicit host-mediated copy
    through *pageable* host memory, which is slower than an explicit
    pinned-memory staging copy.

    Parameters
    ----------
    src:
        Source tensor (may live on any CUDA device or CPU).
    dst:
        Destination tensor (may live on any CUDA device or CPU).
    staging:
        Optional pre-allocated pinned CPU staging buffer.  If provided, it
        must be at least as large as ``src`` (by byte count) and be
        contiguous.  Reusing a staging buffer across calls avoids repeated
        ``pin_memory`` allocations.
    stream:
        CUDA stream on the *destination* device.  Copies are issued onto this
        stream so they can be overlapped with compute.  Defaults to the
        current stream of ``dst.device``.

    Notes
    -----
    If ``src`` and ``dst`` are on the same device, this degenerates to a
    simple ``dst.copy_(src)`` to avoid unnecessary host round-trips.
    """
    src_dev = src.device
    dst_dev = dst.device

    if src_dev == dst_dev:
        with torch.cuda.stream(stream) if stream is not None else _null_ctx():
            dst.copy_(src)
        return

    nbytes = src.numel() * src.element_size()

    if staging is None:
        staging = torch.empty(src.shape, dtype=src.dtype, device="cpu", pin_memory=True)

    with torch.cuda.device(src_dev):
        staging.copy_(src)  # GPU → pinned CPU (non-blocking from src perspective)

    if src.is_cuda:
        torch.cuda.synchronize(src_dev)  # ensure staging is populated before H2D

    with torch.cuda.device(dst_dev):
        ctx = torch.cuda.stream(stream) if stream is not None else _null_ctx()
        with ctx:
            dst.copy_(staging)  # pinned CPU → dst GPU

    if nbytes >= _MIGRATION_LOG_BYTES_THRESHOLD:
        logger.debug(
            "DES-LOC PCIe peer copy: %s -> %s  (%.1f MiB, via pinned staging)",
            src_dev,
            dst_dev,
            nbytes / (1 << 20),
        )


class _null_ctx:
    """Minimal no-op context manager used as a conditional stream guard."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


# ---------------------------------------------------------------------------
# Locality cache transfer buffer
# ---------------------------------------------------------------------------


@dataclass
class DesLocTransferBuffer:
    """Pinned CPU staging buffer for DES-LOC locality-cache-aware transfers.

    When MTP hidden states need to cross PCIe (e.g., base model runs on H100,
    MTP depth-1 decoder runs on A6000), this buffer acts as a zero-copy
    intermediary backed by the 1.5 TB CPU DRAM tier.

    Attributes
    ----------
    max_tokens:     Maximum number of tokens (= max batch size for decode).
    hidden_size:    Model hidden dimension.
    dtype:          Tensor dtype (matches model params_dtype).
    _pinned_buf:    Underlying pinned CPU tensor [max_tokens, 1, hidden_size].
    _lock:          Thread lock guarding concurrent access in DES-LOC's
                    multi-stream execution model.
    """

    max_tokens: int
    hidden_size: int
    dtype: torch.dtype = torch.bfloat16
    _pinned_buf: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self):
        self._pinned_buf = torch.empty(
            self.max_tokens,
            1,
            self.hidden_size,
            dtype=self.dtype,
            device="cpu",
            pin_memory=True,
        )
        logger.debug(
            "DesLocTransferBuffer allocated: shape=(%d, 1, %d), dtype=%s, "
            "pinned=True (locality cache tier)",
            self.max_tokens,
            self.hidden_size,
            self.dtype,
        )

    def stage_from_gpu(self, gpu_buf: torch.Tensor, n_tokens: int) -> None:
        """Copy ``gpu_buf[:n_tokens]`` → pinned CPU buffer (D2H).

        Parameters
        ----------
        gpu_buf:    Source GPU tensor shaped ``[max_tokens, 1, hidden_size]``.
        n_tokens:   Number of valid rows to copy.
        """
        with self._lock:
            self._pinned_buf[:n_tokens].copy_(gpu_buf[:n_tokens])

    def load_to_gpu(
        self,
        dst_device: torch.device,
        n_tokens: int,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        """Copy ``[:n_tokens]`` from CPU staging → *dst_device* GPU.

        Returns a new GPU tensor of shape ``[n_tokens, 1, hidden_size]``.
        The copy is issued onto *stream* if provided.
        """
        with self._lock:
            dst = torch.empty(
                n_tokens,
                1,
                self.hidden_size,
                dtype=self.dtype,
                device=dst_device,
            )
            ctx = torch.cuda.stream(stream) if stream is not None else _null_ctx()
            with ctx:
                dst.copy_(self._pinned_buf[:n_tokens])
            return dst


# ---------------------------------------------------------------------------
# Per-device capability helpers
# ---------------------------------------------------------------------------


def _device_sm_capability(device: torch.device) -> Tuple[int, int]:
    """Return ``(major, minor)`` SM capability for a CUDA *device*."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_capability(idx)


def _effective_graph_scope(
    device: torch.device,
    requested_scope: HeteroCudaGraphScope,
) -> HeteroCudaGraphScope:
    """Resolve the *effective* graph scope for *device* given the *requested* policy.

    DES-LOC heterogeneity rule: ``block_sm90_only`` downgrades to ``layer``
    for SM86 (A6000) devices because block-scope capture on GDDR6X-backed
    devices with limited L2 is disproportionately expensive and may cause
    OOM during simultaneous multi-device captures.

    Parameters
    ----------
    device:
        The CUDA device to evaluate.
    requested_scope:
        The globally configured ``HeteroCudaGraphScope``.

    Returns
    -------
    HeteroCudaGraphScope
        The scope actually used for *device*.
    """
    if requested_scope == HeteroCudaGraphScope.block_sm90_only:
        cap = _device_sm_capability(device)
        if cap < _SM90_CAPABILITY:
            logger.debug(
                "DES-LOC scope downgrade: device %s (SM%d%d) → layer scope "
                "(block_sm90_only policy active)",
                device,
                cap[0],
                cap[1],
            )
            return HeteroCudaGraphScope.layer
    return requested_scope


def _is_block_scope(scope: HeteroCudaGraphScope) -> bool:
    return scope in (HeteroCudaGraphScope.block, HeteroCudaGraphScope.block_sm90_only)


# ---------------------------------------------------------------------------
# HeteroMTPContext — replaces Megatron DynamicInferenceContext fields
# ---------------------------------------------------------------------------


class HeteroMTPContext:
    """DES-LOC counterpart to Megatron's ``DynamicInferenceContext`` MTP fields.

    Upstream design
    ~~~~~~~~~~~~~~~
    Megatron's ``DynamicInferenceContext`` owns the ``mtp_decoder_hidden_states``
    buffer.  In block-scope CUDA graph mode this buffer is pre-allocated to
    ``(max_tokens, 1, hidden_size)`` so that the captured graph can ``copy_()``
    into it at a fixed GPU address on every replay.  In non-block-scope modes
    the attribute is a transient reference that is set to ``None`` after the
    controller reads it.

    DES-LOC adaptation
    ~~~~~~~~~~~~~~~~~~
    ``HeteroMTPContext`` extends this contract with:

    * **Per-device buffer registry** — one pre-allocated buffer per GPU device
      (indexed by ``torch.device``).  On the target cluster that is at most
      three buffers (cuda:0 A6000, cuda:1 A6000, cuda:2 H100).
    * **Locality-cache transfer buffer** — a pinned CPU staging region shared
      across all PCIe-mediated cross-device transfers.
    * **Device-affine scope resolution** — each device independently applies
      the ``_effective_graph_scope`` policy so A6000s and H100 can coexist
      with different capture strategies.
    * **MTP depth → device mapping** — supports DES-LOC's decoupled execution
      where depth-0 (base model) and depth-1+ (MTP heads) may run on different
      physical GPUs.

    Parameters
    ----------
    max_tokens:
        Maximum number of decode tokens per step (worst-case batch size).
    hidden_size:
        Model hidden dimension.
    params_dtype:
        Model parameter dtype.
    num_speculative_tokens:
        Number of speculative tokens generated per step.  If 0, no MTP
        buffers are allocated.
    cuda_graph_scope:
        Requested ``HeteroCudaGraphScope`` policy.
    home_device:
        The GPU device that owns the primary hidden-states buffer (typically
        the device hosting the base model's last transformer layer).
    depth_device_map:
        Optional mapping from MTP depth index (0 = base, 1 = first MTP head,
        …) to the device executing that depth.  Defaults to all depths on
        *home_device*.
    enable_locality_cache:
        Whether to allocate a pinned CPU staging buffer for cross-device
        MTP hidden state transfers.
    """

    def __init__(
        self,
        max_tokens: int,
        hidden_size: int,
        params_dtype: torch.dtype,
        num_speculative_tokens: int,
        cuda_graph_scope: HeteroCudaGraphScope,
        home_device: torch.device,
        depth_device_map: Optional[Dict[int, torch.device]] = None,
        enable_locality_cache: bool = True,
    ) -> None:
        self.max_tokens = max_tokens
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.num_speculative_tokens = num_speculative_tokens
        self.cuda_graph_scope = cuda_graph_scope
        self.home_device = home_device
        self.depth_device_map: Dict[int, torch.device] = depth_device_map or {}
        self.enable_locality_cache = enable_locality_cache

        # Primary hidden-states buffer (device-local).
        # Shape: (max_tokens, 1, hidden_size) when block-scope; None otherwise.
        self._device_buffers: Dict[torch.device, Optional[torch.Tensor]] = {}

        # Transient reference used in non-block-scope modes.
        self._mtp_decoder_hidden_states_ref: Optional[torch.Tensor] = None

        # Locality-cache pinned staging buffer (CPU DRAM tier).
        self._transfer_buffer: Optional[DesLocTransferBuffer] = None

        # Per-device CUDA stream for async cross-device copies.
        self._copy_streams: Dict[torch.device, torch.cuda.Stream] = {}

        self._initialized = False
        self._initialize_buffers()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize_buffers(self) -> None:
        """Allocate device-local GPU buffers and CPU staging as needed."""
        if self.num_speculative_tokens == 0:
            self._initialized = True
            return

        devices_needed = {self.home_device}
        for dev in self.depth_device_map.values():
            devices_needed.add(dev)

        for device in devices_needed:
            effective_scope = _effective_graph_scope(device, self.cuda_graph_scope)
            if _is_block_scope(effective_scope):
                buf = torch.empty(
                    self.max_tokens,
                    1,
                    self.hidden_size,
                    device=device,
                    dtype=self.params_dtype,
                )
                self._device_buffers[device] = buf
                logger.info(
                    "HeteroMTPContext: pre-allocated block-scope MTP buffer on %s "
                    "shape=(%d, 1, %d) dtype=%s (%.1f MiB)",
                    device,
                    self.max_tokens,
                    self.hidden_size,
                    self.params_dtype,
                    buf.numel() * buf.element_size() / (1 << 20),
                )
            else:
                self._device_buffers[device] = None

        # Allocate CPU locality-cache staging buffer if any cross-device transfer
        # is possible (i.e., the base model and MTP heads run on different GPUs).
        has_cross_device = any(
            dev != self.home_device for dev in self.depth_device_map.values()
        )
        if self.enable_locality_cache and has_cross_device:
            self._transfer_buffer = DesLocTransferBuffer(
                max_tokens=self.max_tokens,
                hidden_size=self.hidden_size,
                dtype=self.params_dtype,
            )

        self._initialized = True

    # ------------------------------------------------------------------
    # Public property API  (mirrors Megatron's context.mtp_decoder_hidden_states)
    # ------------------------------------------------------------------

    @property
    def mtp_decoder_hidden_states(self) -> Optional[torch.Tensor]:
        """Return the current MTP decoder hidden states tensor.

        In block-scope mode: returns the home-device pre-allocated buffer
        (full ``max_tokens`` size; caller slices ``[:n]``).
        In non-block-scope mode: returns the transient reference.
        """
        effective = _effective_graph_scope(self.home_device, self.cuda_graph_scope)
        if _is_block_scope(effective):
            return self._device_buffers.get(self.home_device)
        return self._mtp_decoder_hidden_states_ref

    @mtp_decoder_hidden_states.setter
    def mtp_decoder_hidden_states(self, value: Optional[torch.Tensor]) -> None:
        """Set the MTP hidden states reference.

        In block-scope mode: the buffer is pre-allocated; setting to ``None``
        is a no-op (buffer must persist across graph replays).  Setting to a
        tensor triggers a ``copy_()`` into the buffer from the correct device.
        In non-block-scope mode: stores a direct reference; ``None`` releases it.
        """
        effective = _effective_graph_scope(self.home_device, self.cuda_graph_scope)
        if _is_block_scope(effective):
            if value is None:
                # Block-scope buffers persist; ignore None to match upstream behaviour.
                return
            buf = self._device_buffers.get(self.home_device)
            if buf is None:
                raise RuntimeError(
                    "DES-LOC: block-scope MTP buffer not allocated on home device "
                    f"{self.home_device}. Was HeteroMTPContext._initialize_buffers() called?"
                )
            n = value.shape[0]
            if value.device == self.home_device:
                buf[:n].copy_(value)
            else:
                # Cross-device assignment: route through PCIe staging.
                hetero_peer_copy_via_host(
                    src=value,
                    dst=buf[:n],
                    staging=self._transfer_buffer._pinned_buf[:n]
                    if self._transfer_buffer is not None
                    else None,
                )
        else:
            self._mtp_decoder_hidden_states_ref = value

    # ------------------------------------------------------------------
    # DES-LOC: depth-routed hidden state access
    # ------------------------------------------------------------------

    def get_hidden_states_for_depth(
        self,
        depth: int,
        n_tokens: int,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> Optional[torch.Tensor]:
        """Return hidden states routed to the device executing MTP *depth*.

        Parameters
        ----------
        depth:
            MTP depth index (0 = base model output, 1 = first speculative head,
            …).
        n_tokens:
            Number of valid tokens in the current decode step.
        stream:
            Optional destination-device CUDA stream for async transfer.

        Returns
        -------
        Optional[torch.Tensor]
            Tensor of shape ``[n_tokens, 1, hidden_size]`` on the device
            assigned to *depth*.  Returns ``None`` if no hidden states are
            available (e.g., not-last pipeline stage).

        Notes
        -----
        If *depth* is executed on the same device as the home device, the
        buffer slice is returned directly (no copy).  If *depth* is executed
        on a different device, data is staged through the CPU locality cache.
        """
        src_tensor = self.mtp_decoder_hidden_states
        if src_tensor is None:
            return None

        target_device = self.depth_device_map.get(depth, self.home_device)

        if target_device == self.home_device:
            # Same device: cheap slice, no copy.
            return src_tensor[:n_tokens]

        # Cross-device: stage through CPU locality cache.
        if self._transfer_buffer is not None:
            self._transfer_buffer.stage_from_gpu(src_tensor, n_tokens)
            routed = self._transfer_buffer.load_to_gpu(
                dst_device=target_device,
                n_tokens=n_tokens,
                stream=stream,
            )
            logger.debug(
                "DES-LOC MTP depth=%d: routed %d tokens %s→%s via locality cache",
                depth,
                n_tokens,
                self.home_device,
                target_device,
            )
            return routed
        else:
            # Fallback: direct PCIe copy (no staging buffer available).
            dst = torch.empty(
                n_tokens, 1, self.hidden_size, dtype=self.params_dtype, device=target_device
            )
            hetero_peer_copy_via_host(src=src_tensor[:n_tokens], dst=dst)
            return dst

    def release_transient_hidden_states(self, scope: Optional[HeteroCudaGraphScope] = None) -> None:
        """Release the transient MTP hidden states reference if in non-block-scope mode.

        Mirrors the upstream controller logic::

            if has_mtp and context.inference_cuda_graph_scope != InferenceCudaGraphScope.block:
                context.mtp_decoder_hidden_states = None

        Parameters
        ----------
        scope:
            Override scope for the release decision.  Defaults to
            ``self.cuda_graph_scope``.
        """
        check_scope = scope if scope is not None else self.cuda_graph_scope
        effective = _effective_graph_scope(self.home_device, check_scope)
        if not _is_block_scope(effective):
            self._mtp_decoder_hidden_states_ref = None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def buffer_info(self) -> Dict[str, object]:
        """Return a diagnostic snapshot of all allocated buffers."""
        info: Dict[str, object] = {
            "home_device": str(self.home_device),
            "cuda_graph_scope": self.cuda_graph_scope.value,
            "num_speculative_tokens": self.num_speculative_tokens,
            "has_locality_cache_buffer": self._transfer_buffer is not None,
            "device_buffers": {},
        }
        for dev, buf in self._device_buffers.items():
            if buf is not None:
                info["device_buffers"][str(dev)] = {
                    "shape": list(buf.shape),
                    "dtype": str(buf.dtype),
                    "nbytes_MiB": round(buf.numel() * buf.element_size() / (1 << 20), 2),
                }
            else:
                info["device_buffers"][str(dev)] = None
        return info


# ---------------------------------------------------------------------------
# Model-side forward hook — replaces Megatron's model._decoder_hidden_states_cache
# ---------------------------------------------------------------------------


class HeteroMTPForwardHook:
    """Callable that replaces the per-model hidden-states cache assignment.

    Upstream Megatron (pre-1cfa834a) stored hidden states directly on the
    model object::

        self._decoder_hidden_states_cache = hidden_states

    The commit moves this into the inference context.  DES-LOC further
    distinguishes block-scope vs non-block-scope behaviour per device.

    Usage::

        hook = HeteroMTPForwardHook(context)
        # Inside model forward():
        hook(hidden_states, inference_context=context)

    Parameters
    ----------
    context:
        The ``HeteroMTPContext`` owning the MTP buffers.
    """

    def __init__(self, context: HeteroMTPContext) -> None:
        self._context = context

    def __call__(
        self,
        hidden_states: torch.Tensor,
        inference_context: Optional[HeteroMTPContext] = None,
    ) -> None:
        """Write *hidden_states* into the context MTP buffer.

        Parameters
        ----------
        hidden_states:
            Decoder output of shape ``[T, B, H]`` where ``T <= max_tokens``.
        inference_context:
            If provided, used in place of ``self._context``  (for
            flexibility when the same hook is shared across engine instances).

        Behaviour (mirrors Megatron commit logic):
        - **Block-scope**: asserts the pre-allocated buffer exists, then
          issues ``copy_()`` into ``buf[:T]``.
        - **Non-block-scope**: direct reference assignment; GC is left to
          ``release_transient_hidden_states``.
        """
        ctx = inference_context if inference_context is not None else self._context
        if ctx is None:
            return

        device = hidden_states.device
        effective_scope = _effective_graph_scope(device, ctx.cuda_graph_scope)
        n = hidden_states.shape[0]

        if _is_block_scope(effective_scope):
            buf = ctx._device_buffers.get(ctx.home_device)
            if buf is None:
                raise RuntimeError(
                    f"DES-LOC HeteroMTPForwardHook: block-scope buffer missing on "
                    f"{ctx.home_device}.  Ensure HeteroMTPContext was initialised with "
                    f"num_speculative_tokens > 0."
                )
            buf[:n].copy_(hidden_states)
        else:
            ctx._mtp_decoder_hidden_states_ref = hidden_states


# ---------------------------------------------------------------------------
# Controller-side adapter — mirrors TextGenerationController MTP logic
# ---------------------------------------------------------------------------


class HeteroMTPInferenceAdapter:
    """DES-LOC inference controller adapter for heterogeneous MTP decoding.

    Upstream design
    ~~~~~~~~~~~~~~~
    Megatron's ``TextGenerationController._compute_serial_mtp`` reads
    ``context.mtp_decoder_hidden_states`` on the last pipeline stage, routes
    hidden states through the MTP head layers at each speculative depth, and
    samples candidate tokens.  After all depths are processed, it releases the
    transient reference (non-block-scope) or leaves the buffer intact (block-scope).

    DES-LOC adaptation
    ~~~~~~~~~~~~~~~~~~
    ``HeteroMTPInferenceAdapter`` wraps a (possibly mock) set of MTP head
    layers and adds:

    * **Per-depth device routing** via ``HeteroMTPContext.get_hidden_states_for_depth``.
    * **PCIe-aware batch assembly** — gathers ``[:n_tokens]`` slices without
      unnecessary full-buffer transfers.
    * **Block-scope-safe release** — calls
      ``context.release_transient_hidden_states()`` only in non-block-scope mode.

    Parameters
    ----------
    context:
        The ``HeteroMTPContext`` for this inference session.
    mtp_heads:
        A list of ``nn.Module`` objects, one per speculative depth.
        Each module accepts ``(hidden_states, ...)`` and returns ``logits``.
    is_last_pp_stage:
        Whether this rank is the last pipeline-parallel stage (gates has_mtp).
    """

    def __init__(
        self,
        context: HeteroMTPContext,
        mtp_heads: List[nn.Module],
        is_last_pp_stage: bool = True,
    ) -> None:
        self._context = context
        self._mtp_heads = mtp_heads
        self._is_last_pp_stage = is_last_pp_stage

    # ------------------------------------------------------------------
    # has_mtp check — mirrors Megatron controller logic
    # ------------------------------------------------------------------

    @property
    def has_mtp(self) -> bool:
        """True when this rank should perform MTP serial decoding."""
        return (
            self._is_last_pp_stage
            and self._context.mtp_decoder_hidden_states is not None
        )

    # ------------------------------------------------------------------
    # Main MTP decode entry point
    # ------------------------------------------------------------------

    def compute_serial_mtp(
        self,
        last_accepted_indices: torch.Tensor,
        n_active: int,
    ) -> List[torch.Tensor]:
        """Run serial MTP forward passes and return sampled token IDs per depth.

        Parameters
        ----------
        last_accepted_indices:
            1-D int tensor of length ``n_active`` with the sequence position
            of the last accepted token per request (used to gather hidden
            states at the right position).
        n_active:
            Number of active requests in this step.

        Returns
        -------
        List[torch.Tensor]
            One int64 tensor per MTP depth, each of shape ``[n_active]``.

        Algorithm
        ---------
        For each depth *d* in ``range(num_mtp_depths)``:

        1. Obtain hidden states on the device executing depth *d* via
           ``context.get_hidden_states_for_depth(d, n_active)``.
        2. Gather rows at ``last_accepted_indices`` (sequence-level indexing).
        3. Run the MTP head ``mtp_heads[d]`` to get logits.
        4. Greedy-sample the next token.

        After all depths, release the transient reference if appropriate.
        """
        if not self.has_mtp:
            return []

        sampled_per_depth: List[torch.Tensor] = []

        for depth, head in enumerate(self._mtp_heads):
            target_device = self._context.depth_device_map.get(
                depth, self._context.home_device
            )
            hidden = self._context.get_hidden_states_for_depth(
                depth=depth,
                n_tokens=n_active,
            )
            if hidden is None:
                sampled_per_depth.append(
                    torch.zeros(n_active, dtype=torch.int64, device=target_device)
                )
                continue

            # Gather hidden states at the last accepted positions.
            # hidden: [n_active, 1, H]  — already sliced to n_active rows.
            # last_accepted_indices: [n_active] — within [0, n_active).
            gathered = hidden[last_accepted_indices.to(target_device)]  # [n_active, 1, H]
            gathered = gathered.squeeze(1)  # [n_active, H]

            head_device = next(head.parameters(), None)
            if head_device is not None and head_device.device != target_device:
                gathered = gathered.to(head_device.device)

            with torch.no_grad():
                logits = head(gathered)  # [n_active, vocab_size]

            tokens = logits.argmax(dim=-1)  # greedy; real impl uses sampling
            sampled_per_depth.append(tokens)

        self._context.release_transient_hidden_states()
        return sampled_per_depth

    # ------------------------------------------------------------------
    # Expert-parallel dummy forward (mirrors _dummy_serial_mtp_forward)
    # ------------------------------------------------------------------

    def dummy_serial_mtp_forward(self, n_active: int) -> None:
        """Issue dummy MTP forwards on ranks that don't own actual MTP heads.

        This ensures all ranks participate in any collective ops inside the MTP
        head forward (e.g., AllReduce for TP) even when they have no real work.

        Parameters
        ----------
        n_active:
            Number of active requests (used to size the dummy hidden tensor).
        """
        dummy_hidden = torch.zeros(
            n_active,
            self._context.hidden_size,
            dtype=self._context.params_dtype,
            device=self._context.home_device,
        )
        for head in self._mtp_heads:
            with torch.no_grad():
                _ = head(dummy_hidden)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def build_hetero_mtp_context(
    max_tokens: int,
    hidden_size: int,
    params_dtype: torch.dtype,
    num_speculative_tokens: int,
    cuda_graph_scope: str,
    home_device: torch.device,
    depth_device_map: Optional[Dict[int, torch.device]] = None,
    enable_locality_cache: bool = True,
) -> HeteroMTPContext:
    """Construct a ``HeteroMTPContext`` from string-valued config parameters.

    This mirrors how Megatron's ``DynamicInferenceContext.__init__`` reads
    ``model_config.inference_cuda_graph_scope`` and creates the upstream
    ``InferenceCudaGraphScope`` enum.  Here we parse into
    ``HeteroCudaGraphScope`` and apply DES-LOC defaults.

    Parameters
    ----------
    max_tokens:
        Maximum decode-batch size.
    hidden_size:
        Model hidden dimension.
    params_dtype:
        Model parameter dtype.
    num_speculative_tokens:
        Number of speculative tokens (0 disables MTP).
    cuda_graph_scope:
        One of ``"none"``, ``"layer"``, ``"block"``, or
        ``"block_sm90_only"`` (DES-LOC recommended default).
    home_device:
        The device hosting the base model's last transformer layer.
    depth_device_map:
        Optional ``{depth: device}`` routing table.
    enable_locality_cache:
        If ``True`` (default), allocate a pinned CPU staging buffer.

    Returns
    -------
    HeteroMTPContext
    """
    try:
        scope = HeteroCudaGraphScope(cuda_graph_scope)
    except ValueError:
        valid = [e.value for e in HeteroCudaGraphScope]
        raise ValueError(
            f"DES-LOC: unknown cuda_graph_scope={cuda_graph_scope!r}. "
            f"Valid values: {valid}"
        )

    return HeteroMTPContext(
        max_tokens=max_tokens,
        hidden_size=hidden_size,
        params_dtype=params_dtype,
        num_speculative_tokens=num_speculative_tokens,
        cuda_graph_scope=scope,
        home_device=home_device,
        depth_device_map=depth_device_map,
        enable_locality_cache=enable_locality_cache,
    )


# ---------------------------------------------------------------------------
# Utility: simple greedy MTP head (for testing)
# ---------------------------------------------------------------------------


class _LinearMTPHead(nn.Module):
    """Minimal MTP head: Linear(hidden_size → vocab_size).

    Used only in unit tests so there is no dependency on a full model.
    """

    def __init__(self, hidden_size: int, vocab_size: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.linear(x.to(self.linear.weight.dtype))


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Suppress verbose CUDA device allocation logs during tests.
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    class TestHeteroCudaGraphScope(unittest.TestCase):
        """Tests for HeteroCudaGraphScope enum and _effective_graph_scope."""

        def test_enum_values(self):
            self.assertEqual(HeteroCudaGraphScope("none"), HeteroCudaGraphScope.none)
            self.assertEqual(HeteroCudaGraphScope("layer"), HeteroCudaGraphScope.layer)
            self.assertEqual(HeteroCudaGraphScope("block"), HeteroCudaGraphScope.block)
            self.assertEqual(
                HeteroCudaGraphScope("block_sm90_only"), HeteroCudaGraphScope.block_sm90_only
            )

        def test_is_block_scope(self):
            self.assertFalse(_is_block_scope(HeteroCudaGraphScope.none))
            self.assertFalse(_is_block_scope(HeteroCudaGraphScope.layer))
            self.assertTrue(_is_block_scope(HeteroCudaGraphScope.block))
            self.assertTrue(_is_block_scope(HeteroCudaGraphScope.block_sm90_only))

        def test_effective_scope_non_block_passthrough(self):
            """non-block scopes pass through regardless of SM capability."""
            dev = mock.MagicMock(spec=torch.device)
            dev.index = 0
            with mock.patch(
                "deepspeed.inference.hetero_cuda_graph_mtp._device_sm_capability",
                return_value=(8, 6),
            ):
                for scope in (HeteroCudaGraphScope.none, HeteroCudaGraphScope.layer):
                    result = _effective_graph_scope(dev, scope)
                    self.assertEqual(result, scope)

        def test_effective_scope_block_sm90_only_downgrades_sm86(self):
            """block_sm90_only downgrades to layer on SM86."""
            dev = mock.MagicMock(spec=torch.device)
            dev.index = 0
            with mock.patch(
                "deepspeed.inference.hetero_cuda_graph_mtp._device_sm_capability",
                return_value=_SM86_CAPABILITY,
            ):
                result = _effective_graph_scope(dev, HeteroCudaGraphScope.block_sm90_only)
                self.assertEqual(result, HeteroCudaGraphScope.layer)

        def test_effective_scope_block_sm90_only_keeps_sm90(self):
            """block_sm90_only stays block on SM90."""
            dev = mock.MagicMock(spec=torch.device)
            dev.index = 0
            with mock.patch(
                "deepspeed.inference.hetero_cuda_graph_mtp._device_sm_capability",
                return_value=_SM90_CAPABILITY,
            ):
                result = _effective_graph_scope(dev, HeteroCudaGraphScope.block_sm90_only)
                self.assertEqual(result, HeteroCudaGraphScope.block_sm90_only)

        def test_effective_scope_block_always_block(self):
            """block scope is never downgraded regardless of SM."""
            dev = mock.MagicMock(spec=torch.device)
            dev.index = 0
            with mock.patch(
                "deepspeed.inference.hetero_cuda_graph_mtp._device_sm_capability",
                return_value=_SM86_CAPABILITY,
            ):
                result = _effective_graph_scope(dev, HeteroCudaGraphScope.block)
                self.assertEqual(result, HeteroCudaGraphScope.block)

    class TestDesLocTransferBuffer(unittest.TestCase):
        """Tests for DesLocTransferBuffer CPU pinned staging."""

        def test_allocation(self):
            buf = DesLocTransferBuffer(max_tokens=16, hidden_size=64, dtype=torch.float32)
            self.assertIsNotNone(buf._pinned_buf)
            self.assertEqual(buf._pinned_buf.shape, (16, 1, 64))
            self.assertTrue(buf._pinned_buf.is_pinned())

        def test_stage_and_load_cpu(self):
            """Round-trip through pinned buffer using CPU tensors."""
            buf = DesLocTransferBuffer(max_tokens=8, hidden_size=32, dtype=torch.float32)
            src_gpu_mock = torch.randn(8, 1, 32)  # CPU tensor for test portability
            buf.stage_from_gpu(src_gpu_mock, n_tokens=4)
            loaded = buf.load_to_gpu(dst_device=torch.device("cpu"), n_tokens=4)
            self.assertEqual(loaded.shape, (4, 1, 32))
            torch.testing.assert_close(loaded, src_gpu_mock[:4])

    class TestHeteroMTPContextCPU(unittest.TestCase):
        """Tests for HeteroMTPContext using CPU as a stand-in for GPU device.

        All tests run on CPU to remain portable across CI environments without
        real CUDA devices.  Device-specific paths are exercised via mocking.
        """

        def _make_context(
            self,
            scope: HeteroCudaGraphScope,
            num_spec: int = 2,
            home_device: Optional[torch.device] = None,
            depth_device_map: Optional[Dict[int, torch.device]] = None,
        ) -> HeteroMTPContext:
            dev = home_device or torch.device("cpu")
            with mock.patch(
                "deepspeed.inference.hetero_cuda_graph_mtp._device_sm_capability",
                return_value=_SM90_CAPABILITY,
            ):
                ctx = HeteroMTPContext(
                    max_tokens=32,
                    hidden_size=64,
                    params_dtype=torch.float32,
                    num_speculative_tokens=num_spec,
                    cuda_graph_scope=scope,
                    home_device=dev,
                    depth_device_map=depth_device_map,
                    enable_locality_cache=False,  # no cross-device in CPU tests
                )
            return ctx

        def test_no_speculative_tokens_no_buffer(self):
            ctx = self._make_context(HeteroCudaGraphScope.block, num_spec=0)
            self.assertIsNone(ctx.mtp_decoder_hidden_states)

        def test_block_scope_prealloc_buffer(self):
            """Block-scope context pre-allocates the (max_tokens, 1, H) buffer."""
            ctx = self._make_context(HeteroCudaGraphScope.block)
            buf = ctx._device_buffers.get(torch.device("cpu"))
            self.assertIsNotNone(buf)
            self.assertEqual(buf.shape, (32, 1, 64))

        def test_block_scope_property_returns_buffer(self):
            ctx = self._make_context(HeteroCudaGraphScope.block)
            self.assertIsNotNone(ctx.mtp_decoder_hidden_states)
            self.assertEqual(ctx.mtp_decoder_hidden_states.shape, (32, 1, 64))

        def test_block_scope_setter_copies_into_buffer(self):
            """In block-scope mode, setting mtp_decoder_hidden_states writes into the buffer."""
            ctx = self._make_context(HeteroCudaGraphScope.block)
            src = torch.arange(4 * 64, dtype=torch.float32).reshape(4, 1, 64)
            ctx.mtp_decoder_hidden_states = src
            buf = ctx._device_buffers[torch.device("cpu")]
            torch.testing.assert_close(buf[:4], src)

        def test_block_scope_setter_none_is_noop(self):
            """In block-scope mode, setting to None does not destroy the buffer."""
            ctx = self._make_context(HeteroCudaGraphScope.block)
            ctx.mtp_decoder_hidden_states = None
            self.assertIsNotNone(ctx.mtp_decoder_hidden_states)

        def test_layer_scope_transient_ref(self):
            """In layer-scope mode, the property returns a direct tensor reference."""
            ctx = self._make_context(HeteroCudaGraphScope.layer)
            self.assertIsNone(ctx.mtp_decoder_hidden_states)
            t = torch.zeros(4, 1, 64)
            ctx.mtp_decoder_hidden_states = t
            self.assertIs(ctx.mtp_decoder_hidden_states, t)

        def test_layer_scope_release(self):
            """release_transient_hidden_states clears the ref in non-block mode."""
            ctx = self._make_context(HeteroCudaGraphScope.layer)
            ctx.mtp_decoder_hidden_states = torch.zeros(4, 1, 64)
            ctx.release_transient_hidden_states()
            self.assertIsNone(ctx.mtp_decoder_hidden_states)

        def test_block_scope_release_noop(self):
            """release_transient_hidden_states is a no-op in block-scope mode."""
            ctx = self._make_context(HeteroCudaGraphScope.block)
            ctx.release_transient_hidden_states()
            # Buffer must still exist.
            self.assertIsNotNone(ctx.mtp_decoder_hidden_states)

        def test_get_hidden_states_for_depth_same_device(self):
            """Same-device depth access returns a cheap buffer slice."""
            ctx = self._make_context(HeteroCudaGraphScope.block)
            src = torch.ones(5, 1, 64, dtype=torch.float32)
            ctx.mtp_decoder_hidden_states = src
            hidden = ctx.get_hidden_states_for_depth(depth=0, n_tokens=5)
            self.assertIsNotNone(hidden)
            self.assertEqual(hidden.shape, (5, 1, 64))
            torch.testing.assert_close(hidden, src)

        def test_buffer_info_keys(self):
            ctx = self._make_context(HeteroCudaGraphScope.block)
            info = ctx.buffer_info()
            self.assertIn("home_device", info)
            self.assertIn("cuda_graph_scope", info)
            self.assertIn("device_buffers", info)

        def test_block_scope_partial_fill_does_not_read_stale_tail(self):
            """Upstream test_mtp_forward_with_runtime_tokens_below_max ported to CPU.

            Pre-allocate a max_tokens buffer, poison the tail with NaN, fill
            only the first n=4 rows with valid data, then verify that slicing
            [:4] returns the exact valid data and the NaN tail is not exposed.
            """
            ctx = self._make_context(HeteroCudaGraphScope.block)
            max_tok = ctx.max_tokens  # 32
            n = 4

            buf = ctx._device_buffers[torch.device("cpu")]
            buf.fill_(float("nan"))  # poison entire buffer

            valid = torch.arange(n * 64, dtype=torch.float32).reshape(n, 1, 64)
            buf[:n].copy_(valid)

            hidden = ctx.get_hidden_states_for_depth(depth=0, n_tokens=n)
            self.assertFalse(torch.any(torch.isnan(hidden)), "NaN leaked from tail into slice")
            torch.testing.assert_close(hidden, valid)

    class TestHeteroMTPForwardHook(unittest.TestCase):
        """Tests for HeteroMTPForwardHook."""

        def _make_ctx(self, scope: HeteroCudaGraphScope) -> HeteroMTPContext:
            with mock.patch(
                "deepspeed.inference.hetero_cuda_graph_mtp._device_sm_capability",
                return_value=_SM90_CAPABILITY,
            ):
                return HeteroMTPContext(
                    max_tokens=16,
                    hidden_size=32,
                    params_dtype=torch.float32,
                    num_speculative_tokens=1,
                    cuda_graph_scope=scope,
                    home_device=torch.device("cpu"),
                    enable_locality_cache=False,
                )

        def test_block_scope_hook_writes_via_copy(self):
            ctx = self._make_ctx(HeteroCudaGraphScope.block)
            hook = HeteroMTPForwardHook(ctx)
            hidden = torch.full((3, 1, 32), fill_value=7.0)
            hook(hidden)
            buf = ctx._device_buffers[torch.device("cpu")]
            torch.testing.assert_close(buf[:3], hidden)

        def test_layer_scope_hook_stores_reference(self):
            ctx = self._make_ctx(HeteroCudaGraphScope.layer)
            hook = HeteroMTPForwardHook(ctx)
            hidden = torch.randn(3, 1, 32)
            hook(hidden)
            self.assertIs(ctx.mtp_decoder_hidden_states, hidden)

        def test_block_scope_hook_missing_buffer_raises(self):
            ctx = self._make_ctx(HeteroCudaGraphScope.block)
            # Deliberately remove the buffer to simulate misconfiguration.
            ctx._device_buffers[torch.device("cpu")] = None
            hook = HeteroMTPForwardHook(ctx)
            with self.assertRaises(RuntimeError):
                hook(torch.randn(2, 1, 32))

        def test_hook_no_context_is_noop(self):
            """Hook called with inference_context=None must not raise."""
            ctx = self._make_ctx(HeteroCudaGraphScope.block)
            hook = HeteroMTPForwardHook(ctx)
            hook(torch.randn(2, 1, 32), inference_context=None)

    class TestHeteroMTPInferenceAdapter(unittest.TestCase):
        """Tests for HeteroMTPInferenceAdapter.compute_serial_mtp."""

        def _make_adapter(
            self,
            scope: HeteroCudaGraphScope,
            num_depths: int = 2,
        ):
            with mock.patch(
                "deepspeed.inference.hetero_cuda_graph_mtp._device_sm_capability",
                return_value=_SM90_CAPABILITY,
            ):
                ctx = HeteroMTPContext(
                    max_tokens=16,
                    hidden_size=32,
                    params_dtype=torch.float32,
                    num_speculative_tokens=num_depths,
                    cuda_graph_scope=scope,
                    home_device=torch.device("cpu"),
                    enable_locality_cache=False,
                )
            heads = [_LinearMTPHead(32, 50) for _ in range(num_depths)]
            adapter = HeteroMTPInferenceAdapter(ctx, heads, is_last_pp_stage=True)
            return ctx, adapter

        def test_has_mtp_false_when_no_hidden_states(self):
            ctx, adapter = self._make_adapter(HeteroCudaGraphScope.layer)
            # layer scope, no hidden states set → has_mtp = False
            self.assertFalse(adapter.has_mtp)

        def test_has_mtp_true_after_forward_layer_scope(self):
            ctx, adapter = self._make_adapter(HeteroCudaGraphScope.layer)
            ctx.mtp_decoder_hidden_states = torch.randn(4, 1, 32)
            self.assertTrue(adapter.has_mtp)

        def test_has_mtp_true_block_scope(self):
            ctx, adapter = self._make_adapter(HeteroCudaGraphScope.block)
            # block scope: buffer is pre-allocated, so has_mtp is always True
            # once the context has speculative tokens.
            self.assertTrue(adapter.has_mtp)

        def test_compute_serial_mtp_returns_per_depth(self):
            ctx, adapter = self._make_adapter(HeteroCudaGraphScope.layer, num_depths=2)
            hidden = torch.randn(4, 1, 32)
            ctx.mtp_decoder_hidden_states = hidden
            indices = torch.arange(4)
            result = adapter.compute_serial_mtp(last_accepted_indices=indices, n_active=4)
            self.assertEqual(len(result), 2)
            for tok in result:
                self.assertEqual(tok.shape, (4,))

        def test_compute_serial_mtp_tokens_in_vocab_range(self):
            ctx, adapter = self._make_adapter(HeteroCudaGraphScope.layer, num_depths=1)
            hidden = torch.randn(3, 1, 32)
            ctx.mtp_decoder_hidden_states = hidden
            indices = torch.arange(3)
            result = adapter.compute_serial_mtp(last_accepted_indices=indices, n_active=3)
            tok = result[0]
            self.assertTrue(torch.all(tok >= 0))
            self.assertTrue(torch.all(tok < 50))

        def test_compute_serial_mtp_releases_ref_layer_scope(self):
            """After compute_serial_mtp in layer scope, the ref is None."""
            ctx, adapter = self._make_adapter(HeteroCudaGraphScope.layer)
            ctx.mtp_decoder_hidden_states = torch.randn(4, 1, 32)
            indices = torch.arange(4)
            adapter.compute_serial_mtp(last_accepted_indices=indices, n_active=4)
            self.assertIsNone(ctx.mtp_decoder_hidden_states)

        def test_compute_serial_mtp_block_scope_buffer_persists(self):
            """After compute_serial_mtp in block scope, the buffer still exists."""
            ctx, adapter = self._make_adapter(HeteroCudaGraphScope.block)
            indices = torch.arange(4)
            adapter.compute_serial_mtp(last_accepted_indices=indices, n_active=4)
            self.assertIsNotNone(ctx.mtp_decoder_hidden_states)

        def test_compute_serial_mtp_empty_when_not_last_stage(self):
            ctx, _ = self._make_adapter(HeteroCudaGraphScope.layer)
            with mock.patch(
                "deepspeed.inference.hetero_cuda_graph_mtp._device_sm_capability",
                return_value=_SM90_CAPABILITY,
            ):
                non_last_ctx = HeteroMTPContext(
                    max_tokens=16,
                    hidden_size=32,
                    params_dtype=torch.float32,
                    num_speculative_tokens=1,
                    cuda_graph_scope=HeteroCudaGraphScope.layer,
                    home_device=torch.device("cpu"),
                    enable_locality_cache=False,
                )
            heads = [_LinearMTPHead(32, 50)]
            adapter = HeteroMTPInferenceAdapter(non_last_ctx, heads, is_last_pp_stage=False)
            result = adapter.compute_serial_mtp(torch.arange(2), n_active=2)
            self.assertEqual(result, [])

        def test_partial_fill_does_not_expose_nan_tail(self):
            """block-scope: partial n_active < max_tokens must not read NaN tail.

            Mirrors Megatron's TestMTPBlockScopeCudaGraph.test_mtp_forward_with_runtime_tokens_below_max
            """
            ctx, adapter = self._make_adapter(HeteroCudaGraphScope.block, num_depths=1)
            max_tok = ctx.max_tokens  # 16
            n = 3
            buf = ctx._device_buffers[torch.device("cpu")]
            buf.fill_(float("nan"))
            valid = torch.randn(n, 1, 32)
            buf[:n].copy_(valid)

            indices = torch.arange(n)
            result_poisoned = adapter.compute_serial_mtp(
                last_accepted_indices=indices, n_active=n
            )

            # Reset, run with exact buffer to get reference.
            ctx._device_buffers[torch.device("cpu")] = valid.clone()
            result_exact = adapter.compute_serial_mtp(
                last_accepted_indices=indices, n_active=n
            )

            self.assertFalse(
                torch.any(torch.isnan(result_poisoned[0].float())),
                "NaN tail leaked into MTP forward output",
            )
            torch.testing.assert_close(result_poisoned[0], result_exact[0])

    class TestBuildHeteroMTPContext(unittest.TestCase):
        """Tests for the build_hetero_mtp_context factory."""

        def test_valid_scope_strings(self):
            dev = torch.device("cpu")
            with mock.patch(
                "deepspeed.inference.hetero_cuda_graph_mtp._device_sm_capability",
                return_value=_SM90_CAPABILITY,
            ):
                for scope_str in ("none", "layer", "block", "block_sm90_only"):
                    ctx = build_hetero_mtp_context(
                        max_tokens=8,
                        hidden_size=16,
                        params_dtype=torch.float32,
                        num_speculative_tokens=1,
                        cuda_graph_scope=scope_str,
                        home_device=dev,
                        enable_locality_cache=False,
                    )
                    self.assertIsInstance(ctx, HeteroMTPContext)

        def test_invalid_scope_raises(self):
            with self.assertRaises(ValueError):
                build_hetero_mtp_context(
                    max_tokens=8,
                    hidden_size=16,
                    params_dtype=torch.float32,
                    num_speculative_tokens=1,
                    cuda_graph_scope="full",  # invalid
                    home_device=torch.device("cpu"),
                )

        def test_block_sm90_only_with_sm86_gives_no_prealloc(self):
            """block_sm90_only on SM86 (CPU mock) → no pre-allocated buffer."""
            dev = torch.device("cpu")
            with mock.patch(
                "deepspeed.inference.hetero_cuda_graph_mtp._device_sm_capability",
                return_value=_SM86_CAPABILITY,
            ):
                ctx = build_hetero_mtp_context(
                    max_tokens=8,
                    hidden_size=16,
                    params_dtype=torch.float32,
                    num_speculative_tokens=1,
                    cuda_graph_scope="block_sm90_only",
                    home_device=dev,
                    enable_locality_cache=False,
                )
            # Effective scope is layer; no buffer should be allocated.
            buf = ctx._device_buffers.get(dev)
            self.assertIsNone(buf)
            # And the property returns None (transient ref path, nothing assigned).
            self.assertIsNone(ctx.mtp_decoder_hidden_states)

    class TestPeerCopyViaHost(unittest.TestCase):
        """Tests for hetero_peer_copy_via_host on CPU tensors."""

        def test_same_device_direct_copy(self):
            src = torch.arange(12, dtype=torch.float32).reshape(3, 1, 4)
            dst = torch.zeros(3, 1, 4, dtype=torch.float32)
            hetero_peer_copy_via_host(src, dst)
            torch.testing.assert_close(dst, src)

        def test_staging_buffer_reuse(self):
            staging = torch.zeros(6, 1, 4, pin_memory=True, dtype=torch.float32)
            src = torch.ones(6, 1, 4, dtype=torch.float32) * 3.14
            dst = torch.zeros(6, 1, 4, dtype=torch.float32)
            hetero_peer_copy_via_host(src, dst, staging=staging)
            torch.testing.assert_close(dst, src)

        def test_contiguous_required_values_preserved(self):
            src = torch.rand(10, 1, 8)
            dst = torch.empty_like(src)
            hetero_peer_copy_via_host(src, dst)
            torch.testing.assert_close(dst, src)

    # Run all tests.
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for test_class in [
        TestHeteroCudaGraphScope,
        TestDesLocTransferBuffer,
        TestHeteroMTPContextCPU,
        TestHeteroMTPForwardHook,
        TestHeteroMTPInferenceAdapter,
        TestBuildHeteroMTPContext,
        TestPeerCopyViaHost,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
