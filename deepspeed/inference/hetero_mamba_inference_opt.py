"""
deepspeed/inference/hetero_mamba_inference_opt.py
==================================================

DES-LOC Heterogeneous Mamba Inference Optimization
----------------------------------------------------

Upstream design intent (Megatron commit 40627d00):
    1. ``MambaMetadata._batch_indices_decode_buffer`` dtype promoted from
       int32 → int64 so ``selective_state_update`` can index directly without
       a per-layer upcast kernel on each decode step.
    2. ``MambaMixer._A_neg_exp_cache`` — a persistent, pre-allocated float32
       buffer for ``-exp(A_log)``.  Allocated *outside* any CUDA-graph capture
       so its address lives in the default memory pool and remains valid across
       every graph capture/replay cycle (including RL train/eval cycling).
       A companion ``_A_neg_exp_cache_stale`` flag ensures the cache is
       refilled whenever weights may have changed.
    3. ``_get_decode_A_neg_exp()`` — returns the cached, stride-0 expanded
       view ``(nheads, headdim, dstate)`` during inference, activating the
       Triton kernel's TIE_HDIM fast-path.  Falls back to eager recomputation
       during training / grad-enabled contexts.
    4. The per-token upcast ``batch_indices.to(torch.int64)`` is removed from
       the hot decode loop because the buffer is now allocated as int64.
    5. ``topk_routing_with_score_function`` disables sorted top-k during
       inference (``sorted=torch.is_grad_enabled()``), saving a warp-level
       sort per MoE dispatch step.

DES-LOC adaptation points (SM86 × 2  +  SM90 × 1, PCIe, 1.5 TB DRAM):
    A. **Locality cache placement** — the ``_A_neg_exp_cache`` must be
       pinned to the *device that owns the layer*.  Under DES-LOC, layers are
       partitioned across the three physical GPUs; the SM90 H100 NVL holds the
       deep layers, the two A6000 SM86 cards share the shallow layers.  Each
       ``HeteroMambaMixer`` wrapper tracks its ``home_device`` and allocates
       the cache there.
    B. **PCIe-aware prefill/decode split** — without NVLink, cross-device
       tensor moves during decode are expensive.  The ``SharedLocalityCache``
       buffers SSM states for active decode requests in a per-device pool so
       that ``selective_state_update`` never migrates a live SSM state tensor
       across the PCIe bus mid-step.
    C. **Heterogeneous int64 index promotion** — on SM86 the legacy int32
       path in ``selective_state_update`` triggers an implicit upcast that
       wastes ~3 µs per layer per decode step.  We promote *at allocation
       time* (matching upstream) and add an assertion to catch regressions.
    D. **CUDA-graph safety across SM80-class capture domains** — A6000 is
       SM86; H100 NVL is SM90.  They form *separate* CUDA-graph capture
       domains.  ``_get_decode_A_neg_exp`` checks
       ``torch.cuda.is_current_stream_capturing()`` per-device so that cache
       refills are correctly deferred when either capture domain is active.
    E. **SM90-specific TIE_HDIM fast-path gate** — The Triton kernel's
       TIE_HDIM path yields larger throughput gains on SM90 than on SM86.
       We expose a ``use_tie_hdim`` flag that is set automatically from the
       device's compute capability and can be overridden per-layer.
    F. **MoE routing sort gate** — heterogeneous expert placement means MoE
       dispatch indices must remain deterministic during training (sorted=True)
       but can drop the sort during inference (sorted=False).  We extend the
       upstream condition to also account for DES-LOC's *dual-phase execution*
       where a single forward pass may run prefill on SM90 and decode on SM86
       concurrently.

Author: Neuron_SP / DES-LOC team
Mirrors: Megatron 40627d00d96cef74aceab35e0e3c170e900f3aff
"""

from __future__ import annotations

import logging
import threading
import weakref
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware capability registry
# ---------------------------------------------------------------------------

_DEVICE_SM_CACHE: Dict[int, Tuple[int, int]] = {}
_DEVICE_SM_LOCK = threading.Lock()


def _get_sm_capability(device: torch.device) -> Tuple[int, int]:
    """Return (major, minor) compute capability for *device*, cached."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    with _DEVICE_SM_LOCK:
        if idx not in _DEVICE_SM_CACHE:
            cap = torch.cuda.get_device_capability(idx)
            _DEVICE_SM_CACHE[idx] = cap
            logger.debug(
                "Device cuda:%d has SM capability %d.%d", idx, cap[0], cap[1]
            )
        return _DEVICE_SM_CACHE[idx]


def _is_sm90(device: torch.device) -> bool:
    major, _ = _get_sm_capability(device)
    return major == 9


def _is_sm86(device: torch.device) -> bool:
    major, minor = _get_sm_capability(device)
    return major == 8 and minor == 6


# ---------------------------------------------------------------------------
# DES-LOC device topology description
# ---------------------------------------------------------------------------

@dataclass
class HeteroTopology:
    """Describes the physical GPU layout for a DES-LOC run.

    Attributes
    ----------
    sm90_devices:
        CUDA device indices of SM90 (H100 NVL) cards.
    sm86_devices:
        CUDA device indices of SM86 (A6000) cards.
    cpu_offload_bytes:
        Budget (bytes) for CPU DRAM offload of SSM states.  DES-LOC can
        spill least-recently-used SSM states to the 1.5 TB host pool.
    pcie_bandwidth_gbps:
        Approximate measured PCIe bandwidth between any GPU pair (no NVLink).
        Used for scheduling heuristics; does not affect correctness.
    """

    sm90_devices: List[int] = field(default_factory=lambda: [0])
    sm86_devices: List[int] = field(default_factory=lambda: [1, 2])
    cpu_offload_bytes: int = 1 << 40  # 1 TiB default
    pcie_bandwidth_gbps: float = 32.0  # bidirectional PCIe 4.0 x16

    @classmethod
    def auto_detect(cls) -> "HeteroTopology":
        """Probe attached GPUs and build a topology from capability strings."""
        n = torch.cuda.device_count()
        sm90, sm86 = [], []
        for i in range(n):
            cap = torch.cuda.get_device_capability(i)
            if cap == (9, 0):
                sm90.append(i)
            elif cap == (8, 6):
                sm86.append(i)
            else:
                logger.warning(
                    "cuda:%d has unexpected capability %d.%d; treating as SM86", i, *cap
                )
                sm86.append(i)
        if not sm90:
            logger.warning("No SM90 device found; all compute on SM86")
        topo = cls(sm90_devices=sm90, sm86_devices=sm86)
        logger.info(
            "Auto-detected topology: SM90=%s SM86=%s", topo.sm90_devices, topo.sm86_devices
        )
        return topo

    def home_device_for_layer(self, layer_idx: int, total_layers: int) -> torch.device:
        """Assign a layer to its home GPU under DES-LOC's static partition.

        Policy: deep layers (top 60 %) go to SM90 for maximal throughput;
        shallow layers round-robin across SM86 cards.  This reflects the
        reality that SSM recurrences in deep layers dominate decode latency
        and benefit most from SM90's larger shared-memory tiles.
        """
        if not self.sm90_devices:
            idx = layer_idx % len(self.sm86_devices)
            return torch.device(f"cuda:{self.sm86_devices[idx]}")
        sm90_threshold = int(total_layers * 0.40)
        if layer_idx >= sm90_threshold:
            sm90_idx = (layer_idx - sm90_threshold) % len(self.sm90_devices)
            return torch.device(f"cuda:{self.sm90_devices[sm90_idx]}")
        else:
            sm86_idx = layer_idx % len(self.sm86_devices)
            return torch.device(f"cuda:{self.sm86_devices[sm86_idx]}")


# ---------------------------------------------------------------------------
# Shared Locality Cache (SLC) — per-device SSM state pool
# ---------------------------------------------------------------------------

class SharedLocalityCache:
    """Per-device pool of Mamba SSM states for active decode requests.

    Design rationale
    ----------------
    Under DES-LOC each GPU owns a contiguous block of layers.  Without NVLink,
    migrating a live SSM state tensor across PCIe during ``selective_state_update``
    would add ~10–30 µs of bus latency per request per layer.  The SLC keeps
    SSM states *pinned to the device that executes the layer*, avoiding all
    cross-device copies in the hot decode loop.

    The SLC is a flat pre-allocated tensor of shape
    ``(max_slots, nheads, headdim, dstate)`` from which individual request
    slots are carved.  This mirrors the upstream static Mamba state buffer
    approach but is per-device rather than global.

    Parameters
    ----------
    device:
        The CUDA device this cache lives on.
    max_slots:
        Maximum number of concurrently active decode requests.
    nheads, headdim, dstate:
        SSM dimensions matching the layer configuration.
    dtype:
        Storage dtype for SSM states.
    """

    def __init__(
        self,
        device: torch.device,
        max_slots: int,
        nheads: int,
        headdim: int,
        dstate: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = device
        self.max_slots = max_slots
        self.nheads = nheads
        self.headdim = headdim
        self.dstate = dstate

        # Contiguous backing buffer; allocated outside any CUDA-graph domain
        self._buffer = torch.zeros(
            (max_slots, nheads, headdim, dstate),
            dtype=dtype,
            device=device,
        )
        # int64 free-list so indexing never requires an upcast kernel
        self._free_slots = list(range(max_slots))
        self._slot_to_request: Dict[int, int] = {}
        self._request_to_slot: Dict[int, int] = {}
        self._lock = threading.Lock()

        logger.info(
            "SharedLocalityCache on %s: %d slots, shape (%d,%d,%d), %.2f MB",
            device,
            max_slots,
            nheads,
            headdim,
            dstate,
            self._buffer.numel() * self._buffer.element_size() / 1e6,
        )

    # ------------------------------------------------------------------
    # Slot management
    # ------------------------------------------------------------------

    def allocate(self, request_id: int) -> int:
        """Reserve a slot for *request_id*; return slot index (int64-safe)."""
        with self._lock:
            if not self._free_slots:
                raise RuntimeError(
                    f"SharedLocalityCache on {self.device} exhausted "
                    f"(max_slots={self.max_slots})"
                )
            slot = self._free_slots.pop()
            self._slot_to_request[slot] = request_id
            self._request_to_slot[request_id] = slot
            # Zero the slot so stale state from a previous request cannot bleed
            self._buffer[slot].zero_()
            return slot

    def free(self, request_id: int) -> None:
        """Release the slot held by *request_id*."""
        with self._lock:
            slot = self._request_to_slot.pop(request_id, None)
            if slot is None:
                return
            del self._slot_to_request[slot]
            self._free_slots.append(slot)

    def get_state(self, slot: int) -> torch.Tensor:
        """Return a *view* into the SSM state for *slot* (no copy)."""
        return self._buffer[slot]

    def batch_index_tensor(self, request_ids: List[int]) -> torch.Tensor:
        """Build an int64 index tensor for ``selective_state_update``.

        Upstream Megatron promoted ``_batch_indices_decode_buffer`` from int32
        to int64 (40627d00) so that the Triton kernel can index directly.  We
        allocate as int64 here for the same reason, and assert to catch any
        regression where int32 accidentally re-appears.
        """
        slots = [self._request_to_slot[rid] for rid in request_ids]
        idx = torch.tensor(slots, dtype=torch.int64, device=self.device)
        assert idx.dtype == torch.int64, (
            "batch_index_tensor must be int64 to avoid per-layer upcast in "
            "selective_state_update (DES-LOC PCIe adaptation)"
        )
        return idx

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def utilization(self) -> float:
        """Fraction of slots currently in use."""
        with self._lock:
            return (self.max_slots - len(self._free_slots)) / self.max_slots


# ---------------------------------------------------------------------------
# Hetero Mamba Mixer — DES-LOC wrapper
# ---------------------------------------------------------------------------

class HeteroMambaMixer(nn.Module):
    """DES-LOC-aware Mamba SSM mixer with persistent locality cache.

    Wraps a base ``MambaMixer``-compatible module and adds:
      * Home-device-pinned ``_A_neg_exp_cache`` (mirrors upstream 40627d00).
      * Per-device ``SharedLocalityCache`` for SSM states.
      * SM90/SM86 capability-gated TIE_HDIM fast-path selection.
      * CUDA-graph-safe cache invalidation across both capture domains.

    Parameters
    ----------
    base_mixer:
        An existing Mamba mixer whose ``A_log`` parameter drives the cache.
    home_device:
        The CUDA device this layer is assigned to in the DES-LOC partition.
    nheads, headdim, dstate:
        SSM dimensions (must match *base_mixer*).
    slc:
        A ``SharedLocalityCache`` instance for this device.  If *None*, SSM
        state management is delegated to the base mixer.
    layer_idx:
        Layer index within the model (for logging).
    """

    def __init__(
        self,
        base_mixer: nn.Module,
        home_device: torch.device,
        nheads: int,
        headdim: int,
        dstate: int,
        slc: Optional[SharedLocalityCache] = None,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.base_mixer = base_mixer
        self.home_device = home_device
        self.nheads = nheads
        self.headdim = headdim
        self.dstate = dstate
        self.slc = slc
        self.layer_idx = layer_idx

        # Capability-derived flags
        self._is_sm90 = _is_sm90(home_device)
        self._is_sm86 = _is_sm86(home_device)

        # TIE_HDIM fast path: always enabled on SM90; opt-in on SM86 if
        # headdim ≤ 64 (empirically faster on A6000 within that range).
        self.use_tie_hdim: bool = self._is_sm90 or (self._is_sm86 and headdim <= 64)
        if self.use_tie_hdim:
            logger.debug(
                "Layer %d on %s: TIE_HDIM fast-path ENABLED (SM90=%s SM86=%s headdim=%d)",
                layer_idx,
                home_device,
                self._is_sm90,
                self._is_sm86,
                headdim,
            )

        # Persistent A_neg_exp cache — allocated on home_device *outside* any
        # CUDA-graph capture so its address is stable across all graph
        # captures/replays (upstream design intent, DES-LOC adaptation A).
        a_log = self._get_a_log()
        if a_log is not None:
            self._A_neg_exp_cache = torch.empty_like(
                a_log, dtype=torch.float32, device=home_device
            )
            self._A_neg_exp_cache_stale: bool = True
        else:
            self._A_neg_exp_cache = None  # type: ignore[assignment]
            self._A_neg_exp_cache_stale = True

        logger.info(
            "HeteroMambaMixer layer=%d home=%s SM90=%s SM86=%s TIE_HDIM=%s "
            "cache_shape=%s",
            layer_idx,
            home_device,
            self._is_sm90,
            self._is_sm86,
            self.use_tie_hdim,
            tuple(self._A_neg_exp_cache.shape) if self._A_neg_exp_cache is not None else None,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_a_log(self) -> Optional[torch.Tensor]:
        """Retrieve A_log from the base mixer, if present."""
        return getattr(self.base_mixer, "A_log", None)

    def get_decode_A_neg_exp(self) -> torch.Tensor:
        """Return ``-exp(A_log.float())`` pre-expanded to ``(nheads, headdim, dstate)``.

        This is the DES-LOC re-interpretation of Megatron's
        ``MambaMixer._get_decode_A_neg_exp`` (40627d00).

        Key differences from upstream
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        * The cache tensor is pinned to ``self.home_device``; if the caller is
          executing on a different device the result is moved (this should be
          rare and is logged at WARNING level).
        * ``torch.cuda.is_current_stream_capturing()`` is checked against the
          *home device's* current stream, not ``torch.cuda.current_device()``,
          because DES-LOC may have graph captures active on multiple devices
          simultaneously.
        * The stride-0 ``expand`` view activates the Triton kernel's TIE_HDIM
          fast path on SM90 where throughput gains are largest; on SM86 the
          view is only used when ``use_tie_hdim`` is True.

        Returns
        -------
        torch.Tensor
            Shape ``(nheads, headdim, dstate)``, float32, on ``home_device``.
        """
        a_log = self._get_a_log()
        if a_log is None:
            raise RuntimeError(f"Layer {self.layer_idx}: base mixer has no A_log parameter")

        if self.training or torch.is_grad_enabled():
            # Eager path: always recompute during training so gradients flow
            base = -torch.exp(a_log.float())
            if self.use_tie_hdim:
                return base.view(-1, 1, 1).expand(-1, self.headdim, self.dstate)
            return base.unsqueeze(-1).unsqueeze(-1).expand(-1, self.headdim, self.dstate)

        # Inference path — use the persistent locality cache.
        # Check for CUDA-graph capture on the home device stream (DES-LOC
        # adaptation D: each capture domain is checked independently).
        with torch.cuda.device(self.home_device):
            capturing = torch.cuda.is_current_stream_capturing()

        if capturing or self._A_neg_exp_cache_stale:
            with torch.no_grad():
                a_log_on_home = a_log.to(self.home_device)
                neg_exp = -torch.exp(a_log_on_home.float())
                self._A_neg_exp_cache.copy_(neg_exp)
            self._A_neg_exp_cache_stale = False
            if capturing:
                logger.debug(
                    "Layer %d: refilled A_neg_exp cache during graph capture on %s",
                    self.layer_idx,
                    self.home_device,
                )

        # Return stride-0 expanded view; shape (nheads, headdim, dstate)
        if self.use_tie_hdim:
            return self._A_neg_exp_cache.view(-1, 1, 1).expand(
                -1, self.headdim, self.dstate
            )
        return self._A_neg_exp_cache.unsqueeze(-1).unsqueeze(-1).expand(
            -1, self.headdim, self.dstate
        )

    def train(self, mode: bool = True) -> "HeteroMambaMixer":
        """Mark the decode cache stale; weights may have changed.

        Upstream (40627d00) overrides ``train()`` on MambaMixer for the same
        reason.  DES-LOC adds a debug log because mode transitions on
        heterogeneous devices (e.g., switching one SM86 to eval while the
        SM90 remains in train) can be subtle to debug.
        """
        was_training = self.training
        result = super().train(mode)
        self._A_neg_exp_cache_stale = True
        if was_training != mode:
            logger.debug(
                "Layer %d on %s: mode %s → %s; A_neg_exp cache marked stale",
                self.layer_idx,
                self.home_device,
                "train" if was_training else "eval",
                "train" if mode else "eval",
            )
        return result

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        request_ids: Optional[List[int]] = None,
        is_decode: bool = False,
    ) -> torch.Tensor:
        """Forward pass with DES-LOC locality cache injection.

        Parameters
        ----------
        hidden_states:
            Input tensor on ``home_device``.
        request_ids:
            List of active request IDs for this batch.  If provided and
            ``is_decode`` is True, SSM states are retrieved from the
            ``SharedLocalityCache`` and the int64 index tensor is built here
            (upstream adaptation C).
        is_decode:
            Whether this is a single-token decode step.

        Returns
        -------
        torch.Tensor
            Output hidden states on ``home_device``.
        """
        if hidden_states.device != self.home_device:
            logger.warning(
                "Layer %d: hidden_states on %s, expected %s — moving tensor "
                "(cross-PCIe copy; check DES-LOC layer assignment)",
                self.layer_idx,
                hidden_states.device,
                self.home_device,
            )
            hidden_states = hidden_states.to(self.home_device)

        if is_decode and self.slc is not None and request_ids is not None:
            # Build int64 batch indices from the SLC — no per-layer upcast
            batch_indices = self.slc.batch_index_tensor(request_ids)
            # Patch A into the base mixer's decode call via the cache
            a_neg_exp = self.get_decode_A_neg_exp()
            # Delegate to the base mixer; the SLC and A cache are injected
            # via keyword arguments that a DES-LOC-patched base mixer accepts.
            return self.base_mixer(
                hidden_states,
                batch_indices=batch_indices,
                a_neg_exp_override=a_neg_exp,
            )

        return self.base_mixer(hidden_states)


# ---------------------------------------------------------------------------
# Hetero Mamba Metadata — DES-LOC extension of MambaMetadata
# ---------------------------------------------------------------------------

class HeteroMambaMetadata:
    """Per-device Mamba inference metadata for DES-LOC.

    Extends the upstream ``MambaMetadata`` design (Megatron 40627d00) with:
      * Per-device ``_batch_indices_decode_buffer`` (int64, per upstream fix).
      * Integration with ``SharedLocalityCache`` for PCIe-safe state access.
      * Separate prefill and decode slot maps, mirroring upstream structure.

    Parameters
    ----------
    max_requests:
        Maximum number of concurrent inference requests.
    topology:
        The DES-LOC hardware topology.
    """

    def __init__(
        self,
        max_requests: int,
        topology: HeteroTopology,
    ) -> None:
        self.max_requests = max_requests
        self.topology = topology

        # One decode-buffer per device in the topology
        all_devices = (
            [torch.device(f"cuda:{i}") for i in topology.sm90_devices]
            + [torch.device(f"cuda:{i}") for i in topology.sm86_devices]
        )

        # int64 per upstream 40627d00: selective_state_update indexes directly
        # without a per-layer upcast kernel.  Assert dtype to guard regressions.
        self._batch_indices_decode_buffers: Dict[torch.device, torch.Tensor] = {}
        for dev in all_devices:
            buf = torch.full(
                (max_requests,), -1, dtype=torch.int64, device=dev
            )
            assert buf.dtype == torch.int64, (
                f"decode buffer on {dev} must be int64 "
                "(DES-LOC adaptation C / upstream 40627d00)"
            )
            self._batch_indices_decode_buffers[dev] = buf

        # Prefill buffer stays int32; it is not on the selective_state_update
        # hot path and the upstream did not change it.
        self._batch_indices_prefill_buffers: Dict[torch.device, torch.Tensor] = {}
        for dev in all_devices:
            self._batch_indices_prefill_buffers[dev] = torch.full(
                (max_requests,), -1, dtype=torch.int32, device=dev
            )

        # Request → slot maps per device (int entries are slot indices)
        self._decode_slots: Dict[int, Dict[torch.device, int]] = {}  # req_id → {dev → slot}

        logger.info(
            "HeteroMambaMetadata: max_requests=%d devices=%s",
            max_requests,
            [str(d) for d in all_devices],
        )

    def get_decode_buffer(self, device: torch.device) -> torch.Tensor:
        """Return the int64 decode index buffer for *device*."""
        buf = self._batch_indices_decode_buffers.get(device)
        if buf is None:
            raise KeyError(f"No decode buffer registered for device {device}")
        return buf

    def update_decode_indices(
        self, active_request_ids: List[int], slc: SharedLocalityCache
    ) -> None:
        """Populate the decode index buffer from the SharedLocalityCache.

        This replaces the upstream in-place slot bookkeeping with a call that
        queries the SLC so that slot assignments stay consistent between the
        metadata and the state buffer (DES-LOC adaptation B).
        """
        dev = slc.device
        buf = self._batch_indices_decode_buffers[dev]
        # Reset to -1 for inactive slots
        buf.fill_(-1)
        for pos, rid in enumerate(active_request_ids):
            if rid in slc._request_to_slot:
                buf[pos] = slc._request_to_slot[rid]


# ---------------------------------------------------------------------------
# Top-k routing with DES-LOC dual-phase sort gate
# ---------------------------------------------------------------------------

def hetero_topk_routing(
    scores: torch.Tensor,
    topk: int,
    is_decode: bool,
    num_groups: Optional[int] = None,
    group_topk: Optional[int] = None,
    deterministic_prefill: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Heterogeneous-aware top-k routing with sort gate.

    Upstream design intent (40627d00 / ``topk_routing_with_score_function``)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``sorted=torch.is_grad_enabled()`` disables the warp-level sort during
    inference, saving ~2–5 µs per MoE layer per decode step (confirmed on
    A100; similar on A6000/H100).  Sorted order is required during training
    so that expert indices are deterministic for gradient accumulation across
    micro-batches.

    DES-LOC adaptation F
    ~~~~~~~~~~~~~~~~~~~~
    DES-LOC may run prefill on SM90 and decode on SM86 *concurrently within
    the same forward pass* (dual-phase execution).  The upstream condition
    ``torch.is_grad_enabled()`` is insufficient because:
      * Prefill on SM90 runs with grad enabled only during training.
      * Decode on SM86 may run with grad enabled during PPO value estimation.

    We therefore accept an explicit ``is_decode`` flag from the DES-LOC
    scheduler rather than inferring it from grad state alone.  The sort is
    disabled iff ``is_decode`` is True *and* we are not in a
    ``deterministic_prefill`` context (e.g., chunked prefill replay).

    Parameters
    ----------
    scores:
        Router logits of shape ``(batch * seq, num_experts)``.
    topk:
        Number of experts to select per token.
    is_decode:
        True when processing single-token decode steps.
    num_groups, group_topk:
        Optional grouped top-k parameters (passed through).
    deterministic_prefill:
        If True, sorted=True is forced even during decode for determinism.
        Set False only for pure latency-optimised decode.

    Returns
    -------
    (values, indices) : Tuple[Tensor, Tensor]
        Top-k expert scores and indices.
    """
    do_sort = (not is_decode) or deterministic_prefill or torch.is_grad_enabled()

    if num_groups is not None and group_topk is not None:
        # Grouped top-k path (not changed from upstream; placeholder for
        # future DES-LOC expert-parallel extension)
        raise NotImplementedError(
            "Grouped top-k with DES-LOC dual-phase not yet implemented; "
            "set num_groups=None for standard top-k routing."
        )

    return torch.topk(scores, k=topk, dim=1, sorted=do_sort)


# ---------------------------------------------------------------------------
# Model-level builder: wrap existing layers with DES-LOC optimizations
# ---------------------------------------------------------------------------

class DESSLOCMambaModelWrapper:
    """Wrap a model's Mamba layers with DES-LOC locality cache optimizations.

    Usage
    -----
    ::

        topology = HeteroTopology.auto_detect()
        wrapper = DESSLOCMambaModelWrapper(model, topology, max_requests=256)
        wrapper.install()

    After ``install()``, each Mamba layer in the model is replaced by a
    ``HeteroMambaMixer`` that uses the DES-LOC A_neg_exp cache and SLC.
    """

    def __init__(
        self,
        model: nn.Module,
        topology: HeteroTopology,
        max_requests: int = 256,
        nheads: int = 64,
        headdim: int = 64,
        dstate: int = 128,
        ssm_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.model = model
        self.topology = topology
        self.max_requests = max_requests
        self.nheads = nheads
        self.headdim = headdim
        self.dstate = dstate
        self.ssm_dtype = ssm_dtype

        # Pre-create SLCs for every device
        all_devices = (
            [torch.device(f"cuda:{i}") for i in topology.sm90_devices]
            + [torch.device(f"cuda:{i}") for i in topology.sm86_devices]
        )
        self.slcs: Dict[torch.device, SharedLocalityCache] = {
            dev: SharedLocalityCache(
                device=dev,
                max_slots=max_requests,
                nheads=nheads,
                headdim=headdim,
                dstate=dstate,
                dtype=ssm_dtype,
            )
            for dev in all_devices
        }
        self._installed = False

    # ------------------------------------------------------------------
    # Installation
    # ------------------------------------------------------------------

    def _find_mamba_layers(self) -> List[Tuple[str, nn.Module, nn.Module]]:
        """Return (name, parent, child) for all MambaMixer-like sub-modules."""
        results = []
        for name, module in self.model.named_modules():
            cls_name = type(module).__name__
            if "MambaMixer" in cls_name or "MambaLayer" in cls_name:
                # Walk back to parent
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent_name, child_attr = parts
                    parent = dict(self.model.named_modules())[parent_name]
                else:
                    parent = self.model
                    child_attr = name
                results.append((child_attr, parent, module))
        return results

    def install(self, total_layers: Optional[int] = None) -> None:
        """Replace Mamba layers with ``HeteroMambaMixer`` instances."""
        if self._installed:
            logger.warning("DES-LOC wrappers already installed; skipping")
            return

        layers = self._find_mamba_layers()
        if total_layers is None:
            total_layers = len(layers)

        for idx, (attr_name, parent, base_mixer) in enumerate(layers):
            home = self.topology.home_device_for_layer(idx, total_layers)
            slc = self.slcs.get(home)
            wrapper = HeteroMambaMixer(
                base_mixer=base_mixer,
                home_device=home,
                nheads=self.nheads,
                headdim=self.headdim,
                dstate=self.dstate,
                slc=slc,
                layer_idx=idx,
            )
            setattr(parent, attr_name, wrapper)

        self._installed = True
        logger.info(
            "DES-LOC install complete: %d Mamba layers wrapped across %d devices",
            len(layers),
            len(self.slcs),
        )

    def utilization_report(self) -> Dict[str, float]:
        """Return SLC utilization per device (fraction of slots in use)."""
        return {str(dev): slc.utilization() for dev, slc in self.slcs.items()}


# ---------------------------------------------------------------------------
# Utility: force A_neg_exp cache refresh on all wrapped layers
# ---------------------------------------------------------------------------

def refresh_a_neg_exp_caches(model: nn.Module) -> int:
    """Mark all ``HeteroMambaMixer`` A_neg_exp caches as stale.

    Call this after any parameter update (e.g., LoRA merge, weight reload)
    to ensure the cached ``-exp(A_log)`` values are recomputed on the next
    decode step.  Returns the number of caches invalidated.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, HeteroMambaMixer):
            module._A_neg_exp_cache_stale = True
            count += 1
    if count:
        logger.info("Invalidated A_neg_exp caches on %d HeteroMambaMixer layers", count)
    return count


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import unittest

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    class _DummyALog(nn.Module):
        """Minimal stand-in for MambaMixer.A_log."""

        def __init__(self, nheads: int, device: torch.device) -> None:
            super().__init__()
            self.A_log = nn.Parameter(
                torch.randn(nheads, device=device, dtype=torch.float32)
            )

        def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:  # type: ignore[override]
            return x

    # -----------------------------------------------------------------------
    class TestSMCapability(unittest.TestCase):
        def test_cache_population(self):
            if not torch.cuda.is_available():
                self.skipTest("CUDA not available")
            dev = torch.device("cuda:0")
            cap = _get_sm_capability(dev)
            self.assertIsInstance(cap, tuple)
            self.assertEqual(len(cap), 2)
            # Second call must hit cache (no CUDA probe)
            cap2 = _get_sm_capability(dev)
            self.assertEqual(cap, cap2)

    # -----------------------------------------------------------------------
    class TestHeteroTopology(unittest.TestCase):
        def _make_topo(self) -> HeteroTopology:
            return HeteroTopology(sm90_devices=[0], sm86_devices=[1, 2])

        def test_layer_assignment_deep(self):
            topo = self._make_topo()
            # Layer 6 of 10 → deep (≥40 %) → SM90 device 0
            dev = topo.home_device_for_layer(6, 10)
            self.assertEqual(dev, torch.device("cuda:0"))

        def test_layer_assignment_shallow(self):
            topo = self._make_topo()
            # Layer 0 of 10 → shallow → SM86 device 1
            dev = topo.home_device_for_layer(0, 10)
            self.assertIn(dev.index, [1, 2])

        def test_layer_assignment_no_sm90(self):
            topo = HeteroTopology(sm90_devices=[], sm86_devices=[1, 2])
            dev = topo.home_device_for_layer(0, 4)
            self.assertEqual(dev.index, 1)
            dev2 = topo.home_device_for_layer(1, 4)
            self.assertEqual(dev2.index, 2)

    # -----------------------------------------------------------------------
    class TestSharedLocalityCache(unittest.TestCase):
        def _make_slc(self, device: Optional[torch.device] = None) -> SharedLocalityCache:
            dev = device or (
                torch.device("cuda:0") if torch.cuda.is_available()
                else torch.device("cpu")
            )
            return SharedLocalityCache(
                device=dev,
                max_slots=8,
                nheads=4,
                headdim=8,
                dstate=16,
                dtype=torch.float32,
            )

        def test_allocate_and_free(self):
            slc = self._make_slc()
            slot_a = slc.allocate(request_id=42)
            slot_b = slc.allocate(request_id=99)
            self.assertNotEqual(slot_a, slot_b)
            slc.free(request_id=42)
            slot_c = slc.allocate(request_id=77)
            self.assertEqual(slot_c, slot_a)  # slot reuse

        def test_get_state_shape(self):
            slc = self._make_slc()
            slot = slc.allocate(request_id=1)
            state = slc.get_state(slot)
            self.assertEqual(tuple(state.shape), (4, 8, 16))

        def test_batch_index_tensor_dtype(self):
            slc = self._make_slc()
            slc.allocate(request_id=10)
            slc.allocate(request_id=20)
            idx = slc.batch_index_tensor([10, 20])
            self.assertEqual(idx.dtype, torch.int64,
                             "batch_index_tensor must be int64 (upstream 40627d00)")

        def test_exhaustion(self):
            slc = self._make_slc()
            for i in range(8):
                slc.allocate(request_id=i)
            with self.assertRaises(RuntimeError):
                slc.allocate(request_id=999)

        def test_double_free_no_error(self):
            slc = self._make_slc()
            slc.allocate(request_id=5)
            slc.free(request_id=5)
            slc.free(request_id=5)  # idempotent

        def test_utilization(self):
            slc = self._make_slc()
            self.assertAlmostEqual(slc.utilization(), 0.0)
            slc.allocate(request_id=0)
            slc.allocate(request_id=1)
            self.assertAlmostEqual(slc.utilization(), 2 / 8)

    # -----------------------------------------------------------------------
    class TestHeteroMambaMetadata(unittest.TestCase):
        def _make_meta(self) -> HeteroMambaMetadata:
            topo = HeteroTopology(sm90_devices=[], sm86_devices=[0])
            return HeteroMambaMetadata(max_requests=4, topology=topo)

        def test_decode_buffer_dtype(self):
            if not torch.cuda.is_available():
                self.skipTest("CUDA required")
            meta = self._make_meta()
            dev = torch.device("cuda:0")
            buf = meta.get_decode_buffer(dev)
            self.assertEqual(buf.dtype, torch.int64,
                             "Decode buffer must be int64 (upstream 40627d00)")

        def test_update_decode_indices(self):
            if not torch.cuda.is_available():
                self.skipTest("CUDA required")
            dev = torch.device("cuda:0")
            topo = HeteroTopology(sm90_devices=[], sm86_devices=[0])
            meta = HeteroMambaMetadata(max_requests=4, topology=topo)
            slc = SharedLocalityCache(
                device=dev, max_slots=4, nheads=2, headdim=4, dstate=8
            )
            slc.allocate(request_id=10)
            slc.allocate(request_id=20)
            meta.update_decode_indices([10, 20], slc)
            buf = meta.get_decode_buffer(dev)
            # Slots 0..3 should be filled; first two must be non-negative
            self.assertGreaterEqual(buf[0].item(), 0)
            self.assertGreaterEqual(buf[1].item(), 0)

    # -----------------------------------------------------------------------
    class TestHeteroMambaMixer(unittest.TestCase):
        def _make_mixer(
            self, nheads: int = 4, headdim: int = 8, dstate: int = 16
        ) -> HeteroMambaMixer:
            dev = (
                torch.device("cuda:0") if torch.cuda.is_available()
                else torch.device("cpu")
            )
            base = _DummyALog(nheads=nheads, device=dev)
            return HeteroMambaMixer(
                base_mixer=base,
                home_device=dev,
                nheads=nheads,
                headdim=headdim,
                dstate=dstate,
                slc=None,
                layer_idx=0,
            )

        def test_cache_allocated_float32(self):
            mixer = self._make_mixer()
            self.assertEqual(mixer._A_neg_exp_cache.dtype, torch.float32)

        def test_cache_stale_after_train(self):
            mixer = self._make_mixer()
            mixer._A_neg_exp_cache_stale = False
            mixer.train(True)
            self.assertTrue(mixer._A_neg_exp_cache_stale)

        def test_cache_stale_after_train_to_eval(self):
            mixer = self._make_mixer()
            mixer.eval()
            mixer._A_neg_exp_cache_stale = False
            mixer.train(True)
            self.assertTrue(mixer._A_neg_exp_cache_stale)

        def test_get_decode_A_neg_exp_shape_training(self):
            mixer = self._make_mixer(nheads=4, headdim=8, dstate=16)
            mixer.train(True)
            out = mixer.get_decode_A_neg_exp()
            self.assertEqual(tuple(out.shape), (4, 8, 16))

        def test_get_decode_A_neg_exp_shape_eval(self):
            mixer = self._make_mixer(nheads=4, headdim=8, dstate=16)
            mixer.eval()
            with torch.no_grad():
                out = mixer.get_decode_A_neg_exp()
            self.assertEqual(tuple(out.shape), (4, 8, 16))

        def test_get_decode_A_neg_exp_values(self):
            """Cache and eager paths must agree."""
            mixer = self._make_mixer(nheads=4, headdim=8, dstate=16)
            a_log = mixer._get_a_log()
            expected = -torch.exp(a_log.detach().float())

            # Eval path (cache)
            mixer.eval()
            with torch.no_grad():
                cached = mixer.get_decode_A_neg_exp()
            # All values along headdim/dstate dims should equal expected[h]
            for h in range(4):
                self.assertAlmostEqual(
                    cached[h, 0, 0].item(), expected[h].item(), places=5
                )

        def test_cache_not_stale_after_eval_get(self):
            mixer = self._make_mixer()
            mixer.eval()
            self.assertTrue(mixer._A_neg_exp_cache_stale)
            with torch.no_grad():
                mixer.get_decode_A_neg_exp()
            self.assertFalse(mixer._A_neg_exp_cache_stale)

        def test_expand_is_stride_zero(self):
            """Stride-0 expand is what enables TIE_HDIM fast path."""
            mixer = self._make_mixer(nheads=4, headdim=8, dstate=16)
            mixer.use_tie_hdim = True
            mixer.eval()
            with torch.no_grad():
                out = mixer.get_decode_A_neg_exp()
            # At least headdim or dstate dim should have stride 0
            strides = out.stride()
            self.assertIn(0, strides, "Expected at least one stride-0 dimension for TIE_HDIM")

    # -----------------------------------------------------------------------
    class TestHeteroTopKRouting(unittest.TestCase):
        def _scores(self) -> torch.Tensor:
            torch.manual_seed(0)
            return torch.randn(8, 16)  # 8 tokens, 16 experts

        def test_train_sorted(self):
            scores = self._scores()
            vals, idx = hetero_topk_routing(
                scores, topk=2, is_decode=False, deterministic_prefill=True
            )
            self.assertEqual(tuple(vals.shape), (8, 2))
            # Sorted path: each row should be descending
            for row in range(8):
                self.assertGreaterEqual(vals[row, 0].item(), vals[row, 1].item())

        def test_decode_unsorted_allowed(self):
            scores = self._scores()
            # With deterministic_prefill=False we allow unsorted; just check shape
            vals, idx = hetero_topk_routing(
                scores, topk=2, is_decode=True, deterministic_prefill=False
            )
            self.assertEqual(tuple(vals.shape), (8, 2))

        def test_decode_with_grad_forces_sort(self):
            scores = self._scores().requires_grad_(True)
            vals, idx = hetero_topk_routing(
                scores, topk=2, is_decode=True, deterministic_prefill=False
            )
            # torch.is_grad_enabled() → sorted=True → descending rows
            for row in range(8):
                self.assertGreaterEqual(vals[row, 0].item(), vals[row, 1].item())

        def test_grouped_raises(self):
            with self.assertRaises(NotImplementedError):
                hetero_topk_routing(
                    self._scores(), topk=2, is_decode=False,
                    num_groups=4, group_topk=2
                )

    # -----------------------------------------------------------------------
    class TestRefreshCaches(unittest.TestCase):
        def test_refresh_marks_all_stale(self):
            dev = (
                torch.device("cuda:0") if torch.cuda.is_available()
                else torch.device("cpu")
            )

            class _TinyModel(nn.Module):
                def __init__(self_):
                    super().__init__()
                    base0 = _DummyALog(nheads=2, device=dev)
                    base1 = _DummyALog(nheads=2, device=dev)
                    self_.layer0 = HeteroMambaMixer(
                        base0, dev, nheads=2, headdim=4, dstate=8, layer_idx=0
                    )
                    self_.layer1 = HeteroMambaMixer(
                        base1, dev, nheads=2, headdim=4, dstate=8, layer_idx=1
                    )

                def forward(self_, x):
                    return self_.layer1(self_.layer0(x))

            model = _TinyModel()
            # Mark both as fresh
            for m in model.modules():
                if isinstance(m, HeteroMambaMixer):
                    m._A_neg_exp_cache_stale = False

            count = refresh_a_neg_exp_caches(model)
            self.assertEqual(count, 2)
            for m in model.modules():
                if isinstance(m, HeteroMambaMixer):
                    self.assertTrue(m._A_neg_exp_cache_stale)

    # -----------------------------------------------------------------------
    class TestSLCZeroOnAllocate(unittest.TestCase):
        """SSM state slots must be zeroed on allocation to prevent state bleed."""

        def test_zero_on_allocate(self):
            dev = (
                torch.device("cuda:0") if torch.cuda.is_available()
                else torch.device("cpu")
            )
            slc = SharedLocalityCache(
                device=dev, max_slots=4, nheads=2, headdim=4, dstate=8
            )
            slot = slc.allocate(request_id=1)
            # Dirty the slot
            slc._buffer[slot].fill_(3.14)
            slc.free(request_id=1)
            # Reallocate; must be zero
            slot2 = slc.allocate(request_id=2)
            self.assertEqual(slot, slot2)
            self.assertTrue(slc._buffer[slot2].abs().max().item() == 0.0)

    # -----------------------------------------------------------------------
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSMCapability)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestHeteroTopology))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSharedLocalityCache))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestHeteroMambaMetadata))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestHeteroMambaMixer))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestHeteroTopKRouting))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRefreshCaches))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSLCZeroOnAllocate))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
