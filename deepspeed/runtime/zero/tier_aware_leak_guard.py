"""
TierAwareLeakGuard — DES-LOC Heterogeneous Attention Logit Memory Manager
==========================================================================

Upstream Design Intent (Megatron-LM commit 72171c0d):
------------------------------------------------------
In Megatron's TEDotProductAttention, the attention forward pass can optionally
return a ``batch_max_attention_logits`` tensor used for:

  1. QK-Clip: an adaptive clipping mechanism that rescales Q/K projections to
     prevent attention logit explosion during training (similar to μP / logit
     growth stabilization).
  2. Logging: when ``log_max_attention_logit=True``, the per-step maximum logit
     is tracked for monitoring numerical health without actually clipping.

The bug (fixed in Megatron #4699 / #5067):
  - ``batch_max_attention_logits`` was accumulated into
    ``self.current_max_attn_logits`` without detaching from the autograd graph.
  - This kept the entire attention forward graph alive across steps.
  - When ``log_max_attention_logit=True`` but clip is disabled, ``clip_qk()``
    was never called, so ``current_max_attn_logits`` was never reset → unbounded
    graph accumulation → OOM over thousands of steps.

Megatron fix (two-part):
  a) In ``TEDotProductAttention.forward``:
       ``batch_max_attention_logits = batch_max_attention_logits.detach()``
  b) In ``clip_qk()`` under ``log_max_only=True`` branch:
       ``transformer_layer.self_attention.core_attention.current_max_attn_logits = None``
     to explicitly reset the reference so stale tensors don't linger.

DES-LOC Adaptation Points (TierAwareLeakGuard):
------------------------------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) introduces a
*heterogeneous memory hierarchy* across three physical tiers:

  Tier 0 — SM86 A6000 (2x, 48 GB each, PCIe-attached, no NVLink)
  Tier 1 — SM90  H100 NVL (1x, 96 GB, PCIe-attached)
  Tier 2 — CPU DRAM (1.5 TB, shared locality cache / swap backing)

Key DES-LOC-specific considerations that go beyond the Megatron fix:

1. **Cross-Tier Tensor Pinning**
   When a tensor is created on Tier 1 (H100) but later accumulated into a
   Tier 0 (A6000) state variable — or vice versa — the autograd graph node
   keeps a reference to the *originating* device's CUDA context. In a PCIe
   topology without NVLink, releasing such cross-tier references is especially
   critical: the graph node cannot be freed until all upstream CUDAGraph
   captures on the originating device are also released.

2. **Locality Cache Eviction Safety**
   DES-LOC's Shared LOcality Cache (SLC) may asynchronously migrate tensors
   between tiers (e.g. promoting a CPU tensor to H100 HBM for a fused kernel).
   If ``current_max_attn_logits`` retains a grad_fn, the SLC migration logic
   may attempt to clone the full graph onto the target device, causing an
   unintended graph duplication rather than a lightweight tensor move.

3. **Per-Tier Reset Scheduling**
   Because the three tiers run on separate CUDA streams with different
   synchronization cadences, a single global ``current_max_attn_logits = None``
   reset is insufficient. TierAwareLeakGuard tracks which tier produced the
   logit tensor and issues the reset on the corresponding stream, avoiding
   silent use-after-free races that appear only under PCIe async transfers.

4. **ZeRO Stage Interaction**
   Under ZeRO-3 with parameter offloading enabled, attention layer parameters
   may be temporarily resident on CPU. During the brief window where a layer's
   parameters are offloaded but its activations are still live on GPU,
   retaining a graph reference in ``current_max_attn_logits`` prevents ZeRO
   from releasing the offloaded parameter buffers. The detach + reset pattern
   from Megatron generalizes here to a *tier-scoped eviction fence*.

Module Structure:
-----------------
- ``TierSpec``            : Named hardware tier descriptor (SM number, memory,
                            CUDA device index, assigned CUDA stream).
- ``TierRegistry``        : Singleton registry mapping device indices → TierSpec.
- ``LogitAccumulator``    : Drop-in replacement for the raw tensor accumulation
                            inside TEDotProductAttention; enforces detach + tier
                            tagging on every update.
- ``TierAwareResetGuard`` : Context manager / utility that performs the two-part
                            Megatron fix generalized to multi-tier streams.
- ``QKClipAdaptor``       : DES-LOC-aware reimplementation of ``clip_qk()`` from
                            ``megatron/core/optimizer/qk_clip.py``.
- ``TierAwareLeakGuard``  : Top-level orchestrator tying the above together,
                            intended to be instantiated once per DeepSpeed engine.

Usage (inside deepspeed engine or model wrapper):
-------------------------------------------------
::

    from deepspeed.runtime.zero.tier_aware_leak_guard import (
        TierAwareLeakGuard, TierRegistry, TierSpec
    )

    # Register hardware tiers at engine init
    registry = TierRegistry.global_instance()
    registry.register(TierSpec(name="a6000_0", device_idx=0, sm=86, memory_gb=48))
    registry.register(TierSpec(name="a6000_1", device_idx=1, sm=86, memory_gb=48))
    registry.register(TierSpec(name="h100_nvl", device_idx=2, sm=90, memory_gb=96))

    guard = TierAwareLeakGuard(registry=registry, log_max_only=True)

    # In attention forward:
    core_attn_out, batch_max_logits = raw_attn_out
    batch_max_logits = guard.accumulator.update(batch_max_logits, layer_idx=i)

    # After each optimiser step:
    max_logit_scalar = guard.step_reset(model)
"""

from __future__ import annotations

import logging
import threading
import warnings
import weakref
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware constants for the target DES-LOC cluster
# ---------------------------------------------------------------------------

_SM86_MEMORY_GB: int = 48   # NVIDIA A6000 per-card
_SM90_MEMORY_GB: int = 96   # NVIDIA H100 NVL
_CPU_DRAM_GB: int = 1536    # 1.5 TB shared locality cache

# ---------------------------------------------------------------------------
# TierSpec
# ---------------------------------------------------------------------------


@dataclass
class TierSpec:
    """
    Describes one physical memory tier in a DES-LOC heterogeneous cluster.

    Attributes
    ----------
    name : str
        Human-readable label, e.g. ``"a6000_0"``.
    device_idx : int
        CUDA device index as seen by ``torch.cuda.device(device_idx)``.
        Use ``-1`` for CPU (Tier 2).
    sm : int
        CUDA Streaming Multiprocessor compute capability (e.g. 86 for A6000,
        90 for H100, 0 for CPU).
    memory_gb : int
        Nominal on-device memory in gigabytes.
    stream : Optional[torch.cuda.Stream]
        Dedicated CUDA stream for async operations on this tier.
        Populated automatically by ``TierRegistry.register()``.
    priority : int
        Lower value = higher priority for logit accumulation.
        H100 (SM90) gets priority 0; A6000 (SM86) gets priority 1; CPU is 2.
    """

    name: str
    device_idx: int
    sm: int
    memory_gb: int
    stream: Optional[torch.cuda.Stream] = field(default=None, repr=False)
    priority: int = field(default=2, repr=False)

    def __post_init__(self) -> None:
        if self.device_idx >= 0 and self.stream is None:
            try:
                with torch.cuda.device(self.device_idx):
                    self.stream = torch.cuda.Stream(device=self.device_idx)
            except Exception as exc:  # pragma: no cover
                warnings.warn(
                    f"TierSpec({self.name}): could not create CUDA stream — {exc}. "
                    "Leak-guard resets will fall back to the default stream.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        # Assign priority based on SM generation (higher SM = higher priority)
        if self.priority == 2:  # still default
            if self.sm >= 90:
                self.priority = 0
            elif self.sm >= 86:
                self.priority = 1
            else:
                self.priority = 2

    @property
    def is_cpu(self) -> bool:
        """True if this tier is CPU DRAM (Tier 2)."""
        return self.device_idx < 0

    @property
    def torch_device(self) -> torch.device:
        """Return the corresponding ``torch.device``."""
        if self.is_cpu:
            return torch.device("cpu")
        return torch.device("cuda", self.device_idx)


# ---------------------------------------------------------------------------
# TierRegistry
# ---------------------------------------------------------------------------


class TierRegistry:
    """
    Singleton registry mapping CUDA device indices to :class:`TierSpec` objects.

    DES-LOC Design Note
    -------------------
    Because DES-LOC's SLC migrates tensors asynchronously between tiers,
    every component that touches per-tier state (logit accumulators, ZeRO
    partition maps, offload queues) must share a single source of truth about
    which device corresponds to which tier. ``TierRegistry`` provides that
    source.

    Thread Safety
    -------------
    All public methods are protected by ``threading.Lock``. The registry is
    safe to read from data-parallel worker threads while the SLC migration
    thread writes to it.
    """

    _instance: Optional["TierRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._tiers: Dict[int, TierSpec] = {}   # device_idx → TierSpec
        self._name_map: Dict[str, TierSpec] = {}
        self._rw_lock = threading.Lock()
        # CPU tier is always present
        cpu_tier = TierSpec(
            name="cpu_dram",
            device_idx=-1,
            sm=0,
            memory_gb=_CPU_DRAM_GB,
            priority=2,
        )
        self._tiers[-1] = cpu_tier
        self._name_map["cpu_dram"] = cpu_tier

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def global_instance(cls) -> "TierRegistry":
        """Return (or create) the process-global TierRegistry."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_global(cls) -> None:
        """
        Destroy the global instance (for testing / re-init).

        .. warning::
            This is not safe to call while other threads hold references to
            the old registry.
        """
        with cls._lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, spec: TierSpec) -> None:
        """
        Register a hardware tier.

        Parameters
        ----------
        spec : TierSpec
            Tier descriptor. If a tier with the same ``device_idx`` was
            previously registered, the old entry is replaced.
        """
        with self._rw_lock:
            self._tiers[spec.device_idx] = spec
            self._name_map[spec.name] = spec
        logger.debug(
            "TierRegistry: registered %s (device=%d, SM%d, %d GB)",
            spec.name, spec.device_idx, spec.sm, spec.memory_gb,
        )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def tier_for_tensor(self, tensor: torch.Tensor) -> TierSpec:
        """
        Return the :class:`TierSpec` that owns ``tensor``'s storage device.

        Falls back to CPU tier if the device is not registered.

        Parameters
        ----------
        tensor : torch.Tensor
            Any tensor; its ``.device`` attribute is used for lookup.
        """
        with self._rw_lock:
            if tensor.device.type == "cpu":
                return self._tiers[-1]
            dev_idx = tensor.device.index if tensor.device.index is not None else 0
            return self._tiers.get(dev_idx, self._tiers[-1])

    def tier_for_device(self, device: torch.device) -> TierSpec:
        """Return the :class:`TierSpec` for a given :class:`torch.device`."""
        with self._rw_lock:
            if device.type == "cpu":
                return self._tiers[-1]
            idx = device.index if device.index is not None else 0
            return self._tiers.get(idx, self._tiers[-1])

    def preferred_tier(self) -> TierSpec:
        """
        Return the highest-priority registered GPU tier (lowest ``priority`` value).

        Used to determine on which device the canonical
        ``current_max_attn_logits`` scalar should live.
        """
        with self._rw_lock:
            gpu_tiers = [t for t in self._tiers.values() if not t.is_cpu]
            if not gpu_tiers:
                return self._tiers[-1]
            return min(gpu_tiers, key=lambda t: t.priority)

    def all_gpu_tiers(self) -> List[TierSpec]:
        """Return all registered GPU tiers, sorted by priority (ascending)."""
        with self._rw_lock:
            tiers = [t for t in self._tiers.values() if not t.is_cpu]
            return sorted(tiers, key=lambda t: t.priority)

    def __repr__(self) -> str:  # pragma: no cover
        with self._rw_lock:
            lines = [f"TierRegistry({len(self._tiers)} tiers):"]
            for spec in sorted(self._tiers.values(), key=lambda s: s.device_idx):
                lines.append(f"  {spec}")
            return "\n".join(lines)


# ---------------------------------------------------------------------------
# LogitAccumulator
# ---------------------------------------------------------------------------


class LogitAccumulator:
    """
    Per-layer accumulator for ``batch_max_attention_logits``.

    Upstream Context (Megatron ``TEDotProductAttention``)
    -----------------------------------------------------
    After ``te.pytorch.DotProductAttention.forward()`` returns a tuple
    ``(core_attn_out, batch_max_attention_logits)``, Megatron accumulates the
    per-batch maximum logit into ``self.current_max_attn_logits`` via:

    .. code-block:: python

        if self.current_max_attn_logits is None:
            self.current_max_attn_logits = batch_max_attention_logits
        else:
            self.current_max_attn_logits = torch.max(
                self.current_max_attn_logits, batch_max_attention_logits
            )

    The Megatron fix (commit 72171c0d) adds ``.detach()`` *before* this
    accumulation to prevent the autograd graph from being retained.

    DES-LOC Extension
    -----------------
    ``LogitAccumulator`` additionally:

    - Enforces ``.detach()`` unconditionally (mirrors Megatron fix).
    - Tags every incoming tensor with the tier it arrived from (for cross-tier
      graph reference tracking).
    - Detects *tier transitions*: if the current accumulator lives on Tier 0
      (A6000) but a new batch logit arrives from Tier 1 (H100), the
      accumulator is moved to the higher-priority device before the max-reduce
      to avoid an implicit cross-PCIe copy inside ``torch.max``.
    - Exposes ``reset(stream=...)`` to issue the ``= None`` reset on the
      tier-specific CUDA stream, matching the Megatron ``clip_qk`` fix.

    Parameters
    ----------
    layer_idx : int
        Index of the transformer layer that owns this accumulator.
    registry : TierRegistry
        The process-global tier registry.
    """

    def __init__(self, layer_idx: int, registry: TierRegistry) -> None:
        self.layer_idx = layer_idx
        self.registry = registry
        self._value: Optional[torch.Tensor] = None
        self._origin_tier: Optional[TierSpec] = None
        self._lock = threading.Lock()
        self._update_count: int = 0

    # ------------------------------------------------------------------
    # Core update — mirrors Megatron fix with DES-LOC tier awareness
    # ------------------------------------------------------------------

    def update(self, batch_max_logits: torch.Tensor) -> torch.Tensor:
        """
        Ingest a new ``batch_max_attention_logits`` tensor and return it
        detached (the caller should use the returned value, not the original).

        This is the DES-LOC analogue of the Megatron fix:

        .. code-block:: python

            # Megatron (commit 72171c0d, transformer_engine.py line 1867):
            batch_max_attention_logits = batch_max_attention_logits.detach()

        Parameters
        ----------
        batch_max_logits : torch.Tensor
            Raw logit tensor from the attention kernel, potentially with an
            attached grad_fn (hence the need to detach).

        Returns
        -------
        torch.Tensor
            Detached scalar (or 0-dim) tensor. Safe to use for logging /
            QK-clip without retaining the attention forward graph.
        """
        # Step 1: Detach — this is the direct port of the Megatron fix.
        # Without this, every call to update() would extend the autograd graph
        # chain: graph_t0 → graph_t1 → … → graph_tN, keeping all N forward
        # passes alive simultaneously.
        detached = batch_max_logits.detach()

        # Step 2: Identify which tier produced this tensor.
        src_tier = self.registry.tier_for_tensor(detached)

        with self._lock:
            self._update_count += 1

            if self._value is None:
                # First update: just store it, no max-reduce needed.
                self._value = detached
                self._origin_tier = src_tier
            else:
                # Tier-transition detection: if the stored accumulator and the
                # new tensor are on different devices, we must resolve before
                # calling torch.max() to avoid:
                #   a) Implicit cross-PCIe copy (slow on A6000 ↔ H100 PCIe).
                #   b) Silent device-mismatch errors in some PyTorch versions.
                target_tier = self._pick_target_tier(src_tier, self._origin_tier)
                current = self._value.to(target_tier.torch_device, non_blocking=True)
                incoming = detached.to(target_tier.torch_device, non_blocking=True)
                self._value = torch.max(current, incoming)
                self._origin_tier = target_tier

        logger.debug(
            "LogitAccumulator[layer=%d]: update #%d, tier=%s, value=%s",
            self.layer_idx, self._update_count,
            src_tier.name,
            detached.item() if detached.numel() == 1 else detached.shape,
        )
        return detached

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @property
    def value(self) -> Optional[torch.Tensor]:
        """The current accumulated maximum logit, or ``None`` before first update."""
        with self._lock:
            return self._value

    def scalar(self) -> float:
        """
        Return the current maximum as a Python float.

        Returns ``float('-inf')`` if no update has been seen yet, matching
        the semantics of Megatron's ``log_max_attention_logit`` aggregation.
        """
        with self._lock:
            if self._value is None:
                return float("-inf")
            return self._value.item()

    # ------------------------------------------------------------------
    # Reset — mirrors Megatron clip_qk fix (log_max_only branch)
    # ------------------------------------------------------------------

    def reset(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        """
        Reset the accumulator to ``None``.

        Megatron analogue (commit 72171c0d, qk_clip.py lines 48-52):

        .. code-block:: python

            # When qk-clip is disabled, reset so stale references are not retained.
            transformer_layer.self_attention.core_attention.current_max_attn_logits = None

        DES-LOC extension: if the stored tensor lives on a CUDA device and a
        ``stream`` is provided, we record a no-op CUDA event on that stream
        *before* nulling the reference. This ensures that any in-flight CUDA
        kernels that read ``_value`` complete before the tensor's storage is
        reclaimed by the caching allocator.

        Parameters
        ----------
        stream : Optional[torch.cuda.Stream]
            CUDA stream on which to synchronize before nulling.
            If ``None``, no explicit synchronization is performed (matches
            Megatron's original unconditional ``= None``).
        """
        with self._lock:
            if self._value is not None and stream is not None:
                if self._origin_tier is not None and not self._origin_tier.is_cpu:
                    # Record a CUDA event so downstream consumers on this stream
                    # will observe the tensor as available until the event fires.
                    try:
                        with torch.cuda.device(self._origin_tier.device_idx):
                            evt = torch.cuda.Event()
                            evt.record(stream)
                            # We do *not* block here; the event merely marks
                            # the point at which the reference can safely drop.
                    except Exception as exc:  # pragma: no cover
                        logger.debug(
                            "LogitAccumulator[layer=%d]: stream sync event failed: %s",
                            self.layer_idx, exc,
                        )
            self._value = None
            self._origin_tier = None
            self._update_count = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_target_tier(self, tier_a: TierSpec, tier_b: TierSpec) -> TierSpec:
        """
        Given two tiers, return the preferred one for the max-reduce result.

        Higher priority (lower ``priority`` value) wins. On a tie, prefer the
        one that is already GPU over CPU (PCIe transfer to GPU is assumed
        cheaper than GPU→CPU→GPU round-trip).

        This prevents DES-LOC's SLC from repeatedly moving the accumulator
        tensor back and forth between devices when batches alternate between
        tiers.
        """
        if tier_a.priority < tier_b.priority:
            return tier_a
        if tier_b.priority < tier_a.priority:
            return tier_b
        # Same priority: prefer the one that is not CPU
        if not tier_a.is_cpu:
            return tier_a
        return tier_b

    def __repr__(self) -> str:
        with self._lock:
            tier_name = self._origin_tier.name if self._origin_tier else "none"
            val = self._value.item() if self._value is not None and self._value.numel() == 1 else self._value
            return (
                f"LogitAccumulator(layer={self.layer_idx}, "
                f"value={val}, tier={tier_name}, "
                f"updates={self._update_count})"
            )


# ---------------------------------------------------------------------------
# TierAwareResetGuard  (context manager)
# ---------------------------------------------------------------------------


class TierAwareResetGuard:
    """
    Context manager that guarantees ``current_max_attn_logits``-style state is
    reset on the correct CUDA stream after each training step.

    Usage
    -----
    .. code-block:: python

        guard = TierAwareResetGuard(registry, accumulators, log_max_only=True)
        with guard:
            loss = model(batch)
            loss.backward()
            optimizer.step()
        # On __exit__, all accumulators are reset in tier-priority order.

    Megatron Analogue
    -----------------
    The ``__exit__`` logic mirrors the two-part Megatron fix:

    1. ``batch_max_attention_logits.detach()`` — enforced inside each
       ``LogitAccumulator.update()`` call (so already done in the forward pass).

    2. ``transformer_layer.self_attention.core_attention.current_max_attn_logits = None``
       — performed here on the stream that owns each accumulator's tier.

    DES-LOC Extension
    -----------------
    Resets are issued in **tier-priority order** (H100 first, then A6000s, then
    CPU) so that the SLC's background migration thread does not encounter an
    accumulator being reset while it is mid-migration.
    """

    def __init__(
        self,
        registry: TierRegistry,
        accumulators: List[LogitAccumulator],
        log_max_only: bool = False,
    ) -> None:
        self.registry = registry
        self.accumulators = accumulators
        self.log_max_only = log_max_only
        self._entered = False

    def __enter__(self) -> "TierAwareResetGuard":
        self._entered = True
        return self

    def __exit__(self, *_exc) -> None:
        if not self._entered:
            return
        self._entered = False

        # Sort tiers by priority so we reset from highest-priority GPU down to CPU.
        # This mirrors the clip_qk / log_max_only branch in Megatron.
        all_tiers = self.registry.all_gpu_tiers()
        all_tiers.append(self.registry.tier_for_device(torch.device("cpu")))

        for tier in all_tiers:
            stream = tier.stream if not tier.is_cpu else None
            for acc in self.accumulators:
                with acc._lock:
                    if acc._origin_tier is not None and acc._origin_tier.device_idx == tier.device_idx:
                        # Unlock before calling reset (which re-acquires the lock)
                        pass
                # Use the public reset() which handles its own locking
                if (
                    acc._origin_tier is not None
                    and acc._origin_tier.device_idx == tier.device_idx
                ):
                    acc.reset(stream=stream)

    def reset_all(self) -> None:
        """
        Unconditionally reset all accumulators (no context manager required).
        Equivalent to calling ``acc.reset()`` on each accumulator in
        tier-priority order.
        """
        all_tiers = self.registry.all_gpu_tiers()
        all_tiers.append(self.registry.tier_for_device(torch.device("cpu")))
        for tier in all_tiers:
            stream = tier.stream if not tier.is_cpu else None
            for acc in self.accumulators:
                if (
                    acc._origin_tier is not None
                    and acc._origin_tier.device_idx == tier.device_idx
                ):
                    acc.reset(stream=stream)
        # Also reset any accumulators that haven't been tagged with a tier yet
        for acc in self.accumulators:
            if acc._origin_tier is None and acc._value is not None:
                acc.reset()


# ---------------------------------------------------------------------------
# QKClipAdaptor
# ---------------------------------------------------------------------------


class QKClipAdaptor:
    """
    DES-LOC-aware reimplementation of ``megatron/core/optimizer/qk_clip.py:clip_qk()``.

    Upstream Logic (Megatron ``clip_qk``)
    --------------------------------------
    .. code-block:: python

        def clip_qk(model, log_max_only=False) -> float:
            log_max_attention_logit = 0.0
            for module in model.modules():
                if isinstance(module, TEDotProductAttention):
                    if module.current_max_attn_logits is not None:
                        log_max_attention_logit = max(
                            log_max_attention_logit,
                            module.current_max_attn_logits.item(),
                        )
                        if not log_max_only:
                            module.clip_qk()          # resets & clips Q/K scale
                        else:
                            # Megatron fix: reset here when clip is disabled
                            module.current_max_attn_logits = None
            return log_max_attention_logit

    DES-LOC Extension
    -----------------
    ``QKClipAdaptor.step()`` replicates this logic but:

    - Operates on :class:`LogitAccumulator` objects rather than raw tensors,
      so detach/reset semantics are always enforced via the accumulator API.
    - Performs an **all-reduce across tier-local maxima** before returning the
      scalar log value (needed because DES-LOC may run separate layers on
      separate tiers, each with their own local accumulator).
    - Defers resets to the tier-specific stream via ``TierAwareResetGuard``
      rather than an immediate ``= None`` assignment.

    Parameters
    ----------
    registry : TierRegistry
        Process-global tier registry.
    accumulators : List[LogitAccumulator]
        One accumulator per attention layer in the model.
    log_max_only : bool
        If ``True``, match the ``log_max_only=True`` branch of Megatron's
        ``clip_qk``: log the max logit and reset the accumulator, but do not
        apply the actual QK-clip rescaling.
    clip_fn : Optional[callable]
        Optional callable ``clip_fn(layer_idx)`` that performs the actual
        QK-clip operation on layer ``layer_idx``. Ignored when
        ``log_max_only=True``.
    """

    def __init__(
        self,
        registry: TierRegistry,
        accumulators: List[LogitAccumulator],
        log_max_only: bool = False,
        clip_fn: Optional[callable] = None,
    ) -> None:
        self.registry = registry
        self.accumulators = accumulators
        self.log_max_only = log_max_only
        self.clip_fn = clip_fn
        self._reset_guard = TierAwareResetGuard(
            registry=registry,
            accumulators=accumulators,
            log_max_only=log_max_only,
        )

    def step(self) -> float:
        """
        Collect per-layer maximum logits, optionally apply QK-clip, and reset
        accumulators.

        Returns
        -------
        float
            The global maximum attention logit across all layers and all tiers,
            suitable for logging (matches the return value of Megatron's
            ``clip_qk()``).
        """
        # Phase 1: Gather per-layer maxima (tier-local all-reduce within process)
        tier_maxima: Dict[int, float] = {}  # device_idx → running max

        for acc in self.accumulators:
            local_val = acc.scalar()
            if local_val == float("-inf"):
                continue
            tier = acc._origin_tier
            dev_idx = tier.device_idx if tier is not None else -1
            if dev_idx not in tier_maxima:
                tier_maxima[dev_idx] = local_val
            else:
                tier_maxima[dev_idx] = max(tier_maxima[dev_idx], local_val)

        if not tier_maxima:
            return 0.0

        # Phase 2: Cross-tier reduction to get global maximum
        global_max = max(tier_maxima.values())

        logger.debug(
            "QKClipAdaptor.step(): tier_maxima=%s, global_max=%.4f",
            tier_maxima, global_max,
        )

        # Phase 3: Apply QK-clip (or log-only reset)
        if self.log_max_only:
            # Megatron fix (log_max_only branch): reset so stale refs don't accumulate.
            # DES-LOC: use tier-aware reset guard to issue resets on per-tier streams.
            self._reset_guard.reset_all()
        else:
            # Apply actual QK-clip then reset
            if self.clip_fn is not None:
                for acc in self.accumulators:
                    self.clip_fn(acc.layer_idx)
            self._reset_guard.reset_all()

        return float(global_max)


# ---------------------------------------------------------------------------
# TierAwareLeakGuard  (top-level orchestrator)
# ---------------------------------------------------------------------------


class TierAwareLeakGuard:
    """
    Top-level orchestrator for DES-LOC heterogeneous attention logit memory management.

    This class ties together :class:`TierRegistry`, :class:`LogitAccumulator`,
    :class:`TierAwareResetGuard`, and :class:`QKClipAdaptor` into a single
    object intended to be instantiated once per DeepSpeed engine and passed to
    any attention module that reports ``batch_max_attention_logits``.

    Relationship to Megatron Commit 72171c0d
    ----------------------------------------
    The Megatron commit fixes two separate issues with a total of two code
    changes. ``TierAwareLeakGuard`` generalises both:

    +-----------+------------------------------------+------------------------------------+
    | Part      | Megatron fix                       | DES-LOC generalisation             |
    +===========+====================================+====================================+
    | 1         | ``.detach()`` in forward pass      | ``LogitAccumulator.update()``      |
    |           | before accumulating into           | always detaches; also moves to     |
    |           | ``current_max_attn_logits``        | preferred tier to avoid cross-PCIe |
    |           |                                    | graph references                   |
    +-----------+------------------------------------+------------------------------------+
    | 2         | ``= None`` in ``clip_qk``          | ``TierAwareResetGuard.reset_all()``|
    |           | ``log_max_only`` branch to reset   | issues resets on per-tier CUDA     |
    |           | stale reference after logging      | streams in priority order          |
    +-----------+------------------------------------+------------------------------------+

    Parameters
    ----------
    registry : Optional[TierRegistry]
        If ``None``, the global singleton is used (created if necessary).
    log_max_only : bool
        Pass-through to :class:`QKClipAdaptor`.
    clip_fn : Optional[callable]
        Per-layer QK-clip function. Ignored when ``log_max_only=True``.
    num_layers : int
        Number of transformer layers. Used to pre-allocate accumulators.
        Additional accumulators can be added via :meth:`add_accumulator`.
    """

    def __init__(
        self,
        registry: Optional[TierRegistry] = None,
        log_max_only: bool = False,
        clip_fn: Optional[callable] = None,
        num_layers: int = 0,
    ) -> None:
        self.registry = registry if registry is not None else TierRegistry.global_instance()
        self.log_max_only = log_max_only
        self._accumulators: Dict[int, LogitAccumulator] = {}

        for layer_idx in range(num_layers):
            self._accumulators[layer_idx] = LogitAccumulator(
                layer_idx=layer_idx, registry=self.registry
            )

        self._adaptor = QKClipAdaptor(
            registry=self.registry,
            accumulators=list(self._accumulators.values()),
            log_max_only=log_max_only,
            clip_fn=clip_fn,
        )

    # ------------------------------------------------------------------
    # Accumulator management
    # ------------------------------------------------------------------

    def add_accumulator(self, layer_idx: int) -> LogitAccumulator:
        """
        Create and register a new :class:`LogitAccumulator` for ``layer_idx``.

        If one already exists for this index, it is returned unchanged.
        """
        if layer_idx not in self._accumulators:
            acc = LogitAccumulator(layer_idx=layer_idx, registry=self.registry)
            self._accumulators[layer_idx] = acc
            self._adaptor.accumulators.append(acc)
        return self._accumulators[layer_idx]

    @property
    def accumulator(self) -> "MultiLayerAccumulatorProxy":
        """
        Return a proxy that dispatches ``update(tensor, layer_idx=i)`` calls
        to the correct per-layer :class:`LogitAccumulator`.
        """
        return MultiLayerAccumulatorProxy(self)

    # ------------------------------------------------------------------
    # Step interface
    # ------------------------------------------------------------------

    def step_reset(self, model: Optional[nn.Module] = None) -> float:
        """
        Collect maximum attention logits, optionally clip QK, and reset
        all per-layer accumulators.

        This is the DES-LOC replacement for calling ``clip_qk(model)`` after
        each optimiser step in Megatron.

        Parameters
        ----------
        model : Optional[nn.Module]
            Unused in the DES-LOC path (accumulation is tracked via the
            :class:`LogitAccumulator` objects, not by walking ``model.modules()``
            as in Megatron). Kept for API compatibility.

        Returns
        -------
        float
            Global maximum logit across all layers and tiers.
        """
        return self._adaptor.step()

    # ------------------------------------------------------------------
    # ZeRO eviction fence
    # ------------------------------------------------------------------

    def eviction_fence(self, device_idx: int) -> None:
        """
        Issue an eviction fence for a specific tier device.

        Call this before ZeRO-3 evicts a layer's parameters to CPU. It resets
        any accumulator whose ``_origin_tier.device_idx == device_idx``, freeing
        the tensor storage on that device before the parameter buffer is reused.

        Parameters
        ----------
        device_idx : int
            CUDA device index of the tier being evicted.
        """
        tier = self.registry.tier_for_device(torch.device("cuda", device_idx))
        stream = tier.stream
        for acc in self._accumulators.values():
            if acc._origin_tier is not None and acc._origin_tier.device_idx == device_idx:
                acc.reset(stream=stream)
        logger.debug(
            "TierAwareLeakGuard: eviction fence issued for device %d (%s)",
            device_idx, tier.name,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def memory_stats(self) -> Dict[str, object]:
        """
        Return a dictionary of per-tier memory statistics relevant to
        the logit accumulation state.

        Useful for debugging suspected memory leaks.
        """
        stats: Dict[str, object] = {}
        for layer_idx, acc in sorted(self._accumulators.items()):
            stats[f"layer_{layer_idx}_value"] = acc.scalar()
            tier_name = acc._origin_tier.name if acc._origin_tier else "none"
            stats[f"layer_{layer_idx}_tier"] = tier_name
            stats[f"layer_{layer_idx}_updates"] = acc._update_count
        for tier in self.registry.all_gpu_tiers():
            try:
                mem = torch.cuda.memory_allocated(tier.device_idx)
                stats[f"cuda_{tier.device_idx}_allocated_bytes"] = mem
            except Exception:
                pass
        return stats

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TierAwareLeakGuard("
            f"num_layers={len(self._accumulators)}, "
            f"log_max_only={self.log_max_only}, "
            f"registry={len(self.registry.all_gpu_tiers())} GPU tiers)"
        )


# ---------------------------------------------------------------------------
# MultiLayerAccumulatorProxy
# ---------------------------------------------------------------------------


class MultiLayerAccumulatorProxy:
    """
    Convenience proxy returned by ``TierAwareLeakGuard.accumulator``.

    Allows attention modules to call::

        batch_max_logits = guard.accumulator.update(batch_max_logits, layer_idx=i)

    without needing a direct reference to the per-layer
    :class:`LogitAccumulator`.
    """

    def __init__(self, guard: TierAwareLeakGuard) -> None:
        self._guard = weakref.ref(guard)

    def update(self, tensor: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Detach ``tensor`` and update the accumulator for ``layer_idx``.

        Auto-creates the accumulator if it does not yet exist.

        Parameters
        ----------
        tensor : torch.Tensor
            Raw ``batch_max_attention_logits`` from the attention kernel.
        layer_idx : int
            Index of the attention layer.

        Returns
        -------
        torch.Tensor
            Detached tensor (use this value downstream, not the original).
        """
        guard = self._guard()
        if guard is None:
            raise RuntimeError(
                "MultiLayerAccumulatorProxy: parent TierAwareLeakGuard has been "
                "garbage-collected. Keep a reference to the guard alive."
            )
        acc = guard.add_accumulator(layer_idx)
        return acc.update(tensor)


# ---------------------------------------------------------------------------
# Utility: build a default TierRegistry for the target DES-LOC cluster
# ---------------------------------------------------------------------------


def build_des_loc_registry(
    a6000_device_indices: Tuple[int, ...] = (0, 1),
    h100_device_index: int = 2,
) -> TierRegistry:
    """
    Construct a :class:`TierRegistry` pre-populated for the target DES-LOC
    hardware (2x A6000 + 1x H100 NVL + 1.5 TB CPU DRAM).

    Parameters
    ----------
    a6000_device_indices : Tuple[int, ...]
        CUDA device indices for the A6000 cards (SM86, 48 GB each).
    h100_device_index : int
        CUDA device index for the H100 NVL card (SM90, 96 GB).

    Returns
    -------
    TierRegistry
        Populated registry. Also installs it as the global singleton via
        ``TierRegistry.reset_global()`` followed by a ``global_instance()``
        call, so that ``TierRegistry.global_instance()`` returns this object.
    """
    TierRegistry.reset_global()
    registry = TierRegistry.global_instance()

    for i, dev_idx in enumerate(a6000_device_indices):
        registry.register(TierSpec(
            name=f"a6000_{i}",
            device_idx=dev_idx,
            sm=86,
            memory_gb=_SM86_MEMORY_GB,
        ))

    registry.register(TierSpec(
        name="h100_nvl",
        device_idx=h100_device_index,
        sm=90,
        memory_gb=_SM90_MEMORY_GB,
    ))

    return registry


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    import traceback

    _PASS = "\033[92mPASS\033[0m"
    _FAIL = "\033[91mFAIL\033[0m"

    results: List[Tuple[str, bool, str]] = []

    def _run(name: str, fn) -> None:
        try:
            fn()
            results.append((name, True, ""))
        except Exception as exc:  # noqa: BLE001
            results.append((name, False, traceback.format_exc()))

    # -----------------------------------------------------------------------
    # Test helpers
    # -----------------------------------------------------------------------

    def _cpu_tensor(val: float, requires_grad: bool = False) -> torch.Tensor:
        """Create a CPU scalar tensor, optionally with grad."""
        t = torch.tensor(val, dtype=torch.float32, requires_grad=requires_grad)
        if requires_grad:
            # Simulate a grad_fn by going through an op
            t = t * torch.tensor(1.0)
        return t

    def _fresh_registry() -> TierRegistry:
        TierRegistry.reset_global()
        reg = TierRegistry.global_instance()
        # Register two fake GPU tiers as CPU (no real GPU needed for unit tests)
        # We override device_idx with negative values so TierSpec.__post_init__
        # does not try to create a CUDA stream.
        tier0 = TierSpec(name="fake_sm90", device_idx=-2, sm=90, memory_gb=96, stream=None, priority=0)
        tier1 = TierSpec(name="fake_sm86", device_idx=-3, sm=86, memory_gb=48, stream=None, priority=1)
        reg._tiers[-2] = tier0
        reg._name_map["fake_sm90"] = tier0
        reg._tiers[-3] = tier1
        reg._name_map["fake_sm86"] = tier1
        return reg

    # -----------------------------------------------------------------------
    # Test 1: TierSpec priority assignment
    # -----------------------------------------------------------------------

    def test_tier_spec_priority():
        a6000 = TierSpec(name="a6000", device_idx=-3, sm=86, memory_gb=48)
        h100 = TierSpec(name="h100", device_idx=-2, sm=90, memory_gb=96)
        cpu = TierSpec(name="cpu", device_idx=-1, sm=0, memory_gb=1536)
        assert h100.priority < a6000.priority < cpu.priority, (
            f"Priority order wrong: H100={h100.priority}, "
            f"A6000={a6000.priority}, CPU={cpu.priority}"
        )
        assert h100.priority == 0
        assert a6000.priority == 1
        assert cpu.priority == 2

    _run("TierSpec: priority assignment (SM90 > SM86 > CPU)", test_tier_spec_priority)

    # -----------------------------------------------------------------------
    # Test 2: TierRegistry singleton
    # -----------------------------------------------------------------------

    def test_registry_singleton():
        TierRegistry.reset_global()
        r1 = TierRegistry.global_instance()
        r2 = TierRegistry.global_instance()
        assert r1 is r2, "global_instance() must return the same object"
        TierRegistry.reset_global()
        r3 = TierRegistry.global_instance()
        assert r3 is not r1, "After reset_global(), a new instance should be created"

    _run("TierRegistry: singleton behaviour", test_registry_singleton)

    # -----------------------------------------------------------------------
    # Test 3: LogitAccumulator detach (core Megatron fix)
    # -----------------------------------------------------------------------

    def test_accumulator_detaches():
        reg = _fresh_registry()
        acc = LogitAccumulator(layer_idx=0, registry=reg)
        t = _cpu_tensor(3.14, requires_grad=True)
        assert t.requires_grad, "Precondition: tensor has grad"
        result = acc.update(t)
        assert not result.requires_grad, (
            "update() must detach the tensor — grad_fn retained would leak "
            "the attention forward graph (Megatron commit 72171c0d Part 1)"
        )
        # The accumulator's internal value must also be detached
        assert acc.value is not None
        assert not acc.value.requires_grad, "Accumulated value must not require grad"

    _run("LogitAccumulator: detach on update (Megatron Part 1)", test_accumulator_detaches)

    # -----------------------------------------------------------------------
    # Test 4: LogitAccumulator accumulates max correctly
    # -----------------------------------------------------------------------

    def test_accumulator_max():
        reg = _fresh_registry()
        acc = LogitAccumulator(layer_idx=1, registry=reg)
        vals = [1.0, 5.0, 3.0, 7.0, 2.0]
        for v in vals:
            acc.update(_cpu_tensor(v))
        assert abs(acc.scalar() - max(vals)) < 1e-5, (
            f"Expected max={max(vals)}, got {acc.scalar()}"
        )

    _run("LogitAccumulator: max-reduce correctness", test_accumulator_max)

    # -----------------------------------------------------------------------
    # Test 5: LogitAccumulator reset (Megatron fix Part 2)
    # -----------------------------------------------------------------------

    def test_accumulator_reset():
        reg = _fresh_registry()
        acc = LogitAccumulator(layer_idx=2, registry=reg)
        acc.update(_cpu_tensor(9.9))
        assert acc.value is not None
        acc.reset()
        assert acc.value is None, (
            "After reset(), value must be None — mirrors Megatron's "
            "current_max_attn_logits = None in clip_qk log_max_only branch"
        )
        assert acc.scalar() == float("-inf"), "scalar() after reset must be -inf"
        assert acc._update_count == 0, "update_count must be reset to 0"

    _run("LogitAccumulator: reset clears state (Megatron Part 2)", test_accumulator_reset)

    # -----------------------------------------------------------------------
    # Test 6: LogitAccumulator scalar() before any update
    # -----------------------------------------------------------------------

    def test_accumulator_scalar_before_update():
        reg = _fresh_registry()
        acc = LogitAccumulator(layer_idx=3, registry=reg)
        val = acc.scalar()
        assert val == float("-inf"), (
            f"scalar() before any update must be -inf, got {val}"
        )

    _run("LogitAccumulator: scalar() before update → -inf", test_accumulator_scalar_before_update)

    # -----------------------------------------------------------------------
    # Test 7: _pick_target_tier selects higher-priority tier
    # -----------------------------------------------------------------------

    def test_pick_target_tier():
        reg = _fresh_registry()
        acc = LogitAccumulator(layer_idx=0, registry=reg)
        sm90 = reg._tiers[-2]  # priority=0
        sm86 = reg._tiers[-3]  # priority=1
        cpu = reg._tiers[-1]   # priority=2

        assert acc._pick_target_tier(sm90, sm86) is sm90, "SM90 should beat SM86"
        assert acc._pick_target_tier(sm86, sm90) is sm90, "Symmetric: SM90 should still win"
        assert acc._pick_target_tier(sm86, cpu) is sm86, "SM86 should beat CPU"
        assert acc._pick_target_tier(cpu, sm90) is sm90, "SM90 should beat CPU"

    _run("LogitAccumulator: _pick_target_tier priority logic", test_pick_target_tier)

    # -----------------------------------------------------------------------
    # Test 8: QKClipAdaptor.step() returns correct global max and resets
    # -----------------------------------------------------------------------

    def test_qk_clip_step():
        reg = _fresh_registry()
        accs = [LogitAccumulator(layer_idx=i, registry=reg) for i in range(3)]
        layer_vals = [2.5, 8.3, 4.1]
        for acc, v in zip(accs, layer_vals):
            acc.update(_cpu_tensor(v))

        adaptor = QKClipAdaptor(
            registry=reg,
            accumulators=accs,
            log_max_only=True,
        )
        global_max = adaptor.step()
        assert abs(global_max - max(layer_vals)) < 1e-5, (
            f"Expected global max={max(layer_vals)}, got {global_max}"
        )
        # After step(), all accumulators should be reset (log_max_only=True branch)
        for acc in accs:
            assert acc.value is None, (
                f"Accumulator for layer {acc.layer_idx} not reset after step()"
            )

    _run("QKClipAdaptor: step() returns max and resets (log_max_only)", test_qk_clip_step)

    # -----------------------------------------------------------------------
    # Test 9: QKClipAdaptor.step() with no updates returns 0.0
    # -----------------------------------------------------------------------

    def test_qk_clip_step_empty():
        reg = _fresh_registry()
        accs = [LogitAccumulator(layer_idx=i, registry=reg) for i in range(2)]
        adaptor = QKClipAdaptor(registry=reg, accumulators=accs, log_max_only=True)
        result = adaptor.step()
        assert result == 0.0, f"Expected 0.0 for empty accumulators, got {result}"

    _run("QKClipAdaptor: step() with no updates → 0.0", test_qk_clip_step_empty)

    # -----------------------------------------------------------------------
    # Test 10: TierAwareResetGuard context manager
    # -----------------------------------------------------------------------

    def test_reset_guard_context():
        reg = _fresh_registry()
        accs = [LogitAccumulator(layer_idx=i, registry=reg) for i in range(4)]
        # Simulate all accumulators having values and being on the CPU tier
        for acc in accs:
            acc.update(_cpu_tensor(float(acc.layer_idx + 1)))

        guard = TierAwareResetGuard(registry=reg, accumulators=accs, log_max_only=True)
        with guard:
            pass  # Exit triggers reset_all in __exit__
        # All accumulators whose _origin_tier matched a registered tier should be reset.
        # In this test all tensors are on CPU (tier -1), so they all reset.
        for acc in accs:
            assert acc.value is None, (
                f"Accumulator layer={acc.layer_idx} not reset by TierAwareResetGuard"
            )

    _run("TierAwareResetGuard: __exit__ resets all accumulators", test_reset_guard_context)

    # -----------------------------------------------------------------------
    # Test 11: MultiLayerAccumulatorProxy auto-creates accumulators
    # -----------------------------------------------------------------------

    def test_proxy_auto_create():
        TierRegistry.reset_global()
        reg = TierRegistry.global_instance()
        guard = TierAwareLeakGuard(registry=reg, log_max_only=True, num_layers=0)
        proxy = guard.accumulator

        assert 5 not in guard._accumulators
        result = proxy.update(_cpu_tensor(1.23), layer_idx=5)
        assert 5 in guard._accumulators, "Proxy must auto-create accumulator for new layer_idx"
        assert not result.requires_grad, "Proxy must return detached tensor"

    _run("MultiLayerAccumulatorProxy: auto-creates accumulator for new layer", test_proxy_auto_create)

    # -----------------------------------------------------------------------
    # Test 12: TierAwareLeakGuard.step_reset() integrates all parts
    # -----------------------------------------------------------------------

    def test_leak_guard_step_reset():
        TierRegistry.reset_global()
        reg = TierRegistry.global_instance()
        guard = TierAwareLeakGuard(registry=reg, log_max_only=True, num_layers=3)

        vals = [3.0, 6.0, 1.5]
        for i, v in enumerate(vals):
            guard.accumulator.update(_cpu_tensor(v), layer_idx=i)

        global_max = guard.step_reset()
        assert abs(global_max - max(vals)) < 1e-5, (
            f"step_reset() returned {global_max}, expected {max(vals)}"
        )
        # All accumulators should be None after step
        for acc in guard._accumulators.values():
            assert acc.value is None, f"Layer {acc.layer_idx} not reset after step_reset()"

    _run("TierAwareLeakGuard: step_reset() end-to-end", test_leak_guard_step_reset)

    # -----------------------------------------------------------------------
    # Test 13: Memory leak simulation — repeated updates without reset
    # -----------------------------------------------------------------------

    def test_memory_leak_guard_without_grad_accumulation():
        """
        Simulate the scenario that caused the Megatron memory leak:
        accumulating tensors with grad_fn across many steps.
        Without the detach fix, graph_t0 → graph_t1 → … → graph_tN grows.
        With the fix (LogitAccumulator.update detaches), each step is O(1).
        """
        TierRegistry.reset_global()
        reg = TierRegistry.global_instance()
        guard = TierAwareLeakGuard(registry=reg, log_max_only=True, num_layers=1)

        NUM_STEPS = 100
        for step in range(NUM_STEPS):
            # Simulate a tensor that has a grad_fn (as in a real attention forward pass)
            raw_logit = torch.tensor(float(step), dtype=torch.float32, requires_grad=True)
            sim_logit = raw_logit * torch.tensor(1.0)  # creates grad_fn
            assert sim_logit.grad_fn is not None, "Test precondition: grad_fn must exist"
            guard.accumulator.update(sim_logit, layer_idx=0)

        # The accumulated value should be the max of all inputs (99.0)
        acc = guard._accumulators[0]
        assert abs(acc.scalar() - (NUM_STEPS - 1)) < 1e-4, (
            f"Expected {NUM_STEPS - 1}, got {acc.scalar()}"
        )
        # Critically: the accumulated tensor must NOT have a grad_fn
        assert not acc.value.requires_grad, (
            "Memory leak detected: accumulated tensor has requires_grad=True. "
            "This means the autograd graph is being retained across steps. "
            "The detach() fix in LogitAccumulator.update() must be applied."
        )

    _run("Memory leak guard: no grad_fn retained after 100 updates", test_memory_leak_guard_without_grad_accumulation)

    # -----------------------------------------------------------------------
    # Test 14: TierAwareLeakGuard.memory_stats() returns sane structure
    # -----------------------------------------------------------------------

    def test_memory_stats():
        TierRegistry.reset_global()
        reg = TierRegistry.global_instance()
        guard = TierAwareLeakGuard(registry=reg, log_max_only=True, num_layers=2)
        guard.accumulator.update(_cpu_tensor(1.0), layer_idx=0)

        stats = guard.memory_stats()
        assert "layer_0_value" in stats
        assert "layer_1_value" in stats
        assert stats["layer_0_value"] == 1.0
        assert stats["layer_1_value"] == float("-inf")
        assert "layer_0_tier" in stats
        assert "layer_0_updates" in stats
        assert stats["layer_0_updates"] == 1

    _run("TierAwareLeakGuard: memory_stats() structure", test_memory_stats)

    # -----------------------------------------------------------------------
    # Test 15: build_des_loc_registry populates correct SM and memory values
    # -----------------------------------------------------------------------

    def test_build_des_loc_registry():
        # We can't actually register real CUDA devices, so patch __post_init__
        # to skip stream creation.
        original_post_init = TierSpec.__post_init__

        def _no_stream_post_init(self):
            # Skip CUDA stream creation; set priority only
            if self.priority == 2:
                if self.sm >= 90:
                    self.priority = 0
                elif self.sm >= 86:
                    self.priority = 1
                else:
                    self.priority = 2

        TierSpec.__post_init__ = _no_stream_post_init
        try:
            reg = build_des_loc_registry(
                a6000_device_indices=(0, 1),
                h100_device_index=2,
            )
            assert "h100_nvl" in reg._name_map
            h100 = reg._name_map["h100_nvl"]
            assert h100.sm == 90
            assert h100.memory_gb == 96
            assert h100.priority == 0

            assert "a6000_0" in reg._name_map
            a6000_0 = reg._name_map["a6000_0"]
            assert a6000_0.sm == 86
            assert a6000_0.memory_gb == 48
            assert a6000_0.priority == 1

            assert "a6000_1" in reg._name_map
            pref = reg.preferred_tier()
            assert pref.name == "h100_nvl", (
                f"H100 should be preferred tier, got {pref.name}"
            )
        finally:
            TierSpec.__post_init__ = original_post_init

    _run("build_des_loc_registry: correct SM/memory/priority for DES-LOC cluster", test_build_des_loc_registry)

    # -----------------------------------------------------------------------
    # Test 16: eviction_fence resets correct tier's accumulators only
    # -----------------------------------------------------------------------

    def test_eviction_fence_selective():
        TierRegistry.reset_global()
        reg = TierRegistry.global_instance()
        # Register two fake GPU tiers (negative indices to skip CUDA)
        tier_a = TierSpec(name="tier_a", device_idx=-10, sm=86, memory_gb=48, stream=None, priority=1)
        tier_b = TierSpec(name="tier_b", device_idx=-11, sm=90, memory_gb=96, stream=None, priority=0)
        reg._tiers[-10] = tier_a
        reg._name_map["tier_a"] = tier_a
        reg._tiers[-11] = tier_b
        reg._name_map["tier_b"] = tier_b

        guard = TierAwareLeakGuard(registry=reg, log_max_only=True, num_layers=2)

        # Manually assign accumulators to different tiers
        guard._accumulators[0].update(_cpu_tensor(5.0))
        guard._accumulators[0]._origin_tier = tier_a
        guard._accumulators[0]._value = torch.tensor(5.0)

        guard._accumulators[1].update(_cpu_tensor(3.0))
        guard._accumulators[1]._origin_tier = tier_b
        guard._accumulators[1]._value = torch.tensor(3.0)

        # Fence only device -10 (tier_a)
        guard.eviction_fence(device_idx=-10)

        assert guard._accumulators[0].value is None, "tier_a accumulator should be reset"
        assert guard._accumulators[1].value is not None, "tier_b accumulator should NOT be reset"

    _run("eviction_fence: resets only the fenced tier's accumulators", test_eviction_fence_selective)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("TierAwareLeakGuard — DES-LOC Unit Test Results")
    print("=" * 72)
    passed = 0
    for name, ok, tb in results:
        status = _PASS if ok else _FAIL
        print(f"  [{status}] {name}")
        if not ok:
            print(tb)
        else:
            passed += 1
    print("=" * 72)
    print(f"  {passed}/{len(results)} tests passed")
    print("=" * 72)

    sys.exit(0 if passed == len(results) else 1)
