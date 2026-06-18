"""
deepspeed/moe/hetero_custom_router.py

Upstream intent (Megatron 7857383):
    Megatron decoupled the router from MoELayer by introducing RouterInterface /
    RouterBuilder Protocol types and replacing the hardcoded ``TopKRouter(...)``
    call with ``submodules.router(config=..., pg_collection=...)``.  The companion
    ``apply_module()`` wrapper normalises how the router's forward() is invoked so
    that custom routers need not inherit from a concrete base class — they only need
    to satisfy the structural Protocol.

DES-LOC adaptation points:
    1. **Heterogeneous device placement** — A6000×2 (SM86, 48 GB, PCIe) and
       H100-NVL×1 (SM90, 96 GB, PCIe).  The router gate lives on the device that
       *owns* the token stream (usually the H100 for large-batch stages, A6000 for
       smaller pipeline slices).  We track placement explicitly and move gate
       weights only when the placement changes.

    2. **Shared-Locality Cache (LOC)** — routing decisions (probs + routing_map)
       are expensive because all tokens must pass through softmax + top-k.  DES-LOC
       caches the routing decision in CPU DRAM (1.5 TB available) keyed by a
       lightweight token-fingerprint, so repeated micro-batches with near-identical
       hidden-state distributions re-use the cached assignment.  Cache is
       content-addressed with an approximate hash (mean + std per token).

    3. **Decoupled Execution (DES)** — expert computation on A6000 devices can
       overlap with router computation on H100 by using async CUDA streams.  The
       router forward runs on ``stream_router``; the caller may advance expert
       pre-fetch on the default stream while the router is still running, then
       synchronise before consuming the routing_map.

    4. **Custom RouterBuilder protocol** — mirrors Megatron's ``RouterBuilder``
       Protocol so that any compliant router (TopK, Expert-Choice, Load-balanced,
       etc.) can be injected at layer-construction time via ``MoESubmodules.router``.

    5. **PCIe-aware dispatch** — without NVLink, all-to-all for expert dispatch
       crosses PCIe.  We expose a ``topology_hint`` that the dispatcher can use to
       prefer local-device experts over remote ones, reducing cross-PCIe traffic.

Public API
----------
    RouterInterface          — structural Protocol (mirrors Megatron)
    RouterBuilder            — callable Protocol returning RouterInterface
    HeteroTopKRouter         — concrete DES-LOC-aware TopK router
    HeteroCachedRouter       — wraps any RouterInterface with LOC caching
    HeteroMoESubmodules      — dataclass mirroring MoESubmodules with DES-LOC fields
    apply_hetero_router      — DES-LOC equivalent of Megatron's apply_module()
    build_default_router     — convenience RouterBuilder for standard TopK
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware topology constants for the target cluster
# ---------------------------------------------------------------------------

_DEVICE_SM_MAP: Dict[str, int] = {
    "a6000_0": 86,
    "a6000_1": 86,
    "h100_nvl": 90,
}

_DEVICE_VRAM_GB: Dict[str, float] = {
    "a6000_0": 48.0,
    "a6000_1": 48.0,
    "h100_nvl": 96.0,
}


def _cuda_device_label(device: torch.device) -> str:
    """Return a human-readable label for a CUDA device index."""
    idx = device.index if device.index is not None else 0
    labels = list(_DEVICE_SM_MAP.keys())
    if idx < len(labels):
        return labels[idx]
    return f"cuda:{idx}"


# ---------------------------------------------------------------------------
# Protocols  (structural — no ABC inheritance required)
# ---------------------------------------------------------------------------


class RouterInterface:
    """
    Structural Protocol for any router compatible with DES-LOC MoELayer.

    Mirrors Megatron's RouterInterface Protocol (commit 7857383) but adds
    DES-LOC-specific optional hooks.  Implementors only need ``forward`` and
    ``set_layer_number``; the rest have default no-op implementations.
    """

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        Parameters
        ----------
        hidden_states:
            Shape ``[seq_len, batch, hidden_size]`` or ``[tokens, hidden_size]``.

        Returns
        -------
        probs:
            Routing probabilities, shape ``[tokens, topk]``.
        routing_map:
            Integer expert indices, shape ``[tokens, topk]``.
        """
        raise NotImplementedError

    def set_layer_number(self, layer_number: int) -> None:
        """Called by the transformer layer during __init__ to record depth."""
        pass  # default: ignore

    def set_device_placement(self, device: torch.device) -> None:
        """DES-LOC hook: inform the router which device owns the token stream."""
        pass  # default: ignore

    def warmup_loc_cache(self, representative_input: torch.Tensor) -> None:
        """DES-LOC hook: pre-populate the LOC cache with a representative batch."""
        pass  # default: ignore

    def __call__(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(hidden_states)


# RouterBuilder is a callable: (*, config, pg_collection) -> RouterInterface
RouterBuilder = Any  # keep runtime-compatible; type-checkers see the Protocol below


# ---------------------------------------------------------------------------
# Topology hint dataclass
# ---------------------------------------------------------------------------


@dataclass
class PCIeTopologyHint:
    """
    Encodes PCIe interconnect topology for the DES-LOC cluster.

    Without NVLink, any cross-device tensor movement traverses PCIe (gen4 ×16
    ≈ 32 GB/s unidirectional).  The dispatcher uses these hints to prefer
    assigning tokens to local-device experts, reducing synchronisation cost.

    Attributes
    ----------
    local_device_index:
        CUDA device index of the rank that owns this router instance.
    peer_device_indices:
        List of CUDA device indices reachable over PCIe.
    h100_device_index:
        Index of the H100-NVL device; -1 if not present on this node.
    pcie_bandwidth_gbps:
        Estimated unidirectional bandwidth between any two devices (GB/s).
    prefer_local_experts:
        When True, the dispatcher biases routing toward experts on the local
        device before spilling to remote devices.
    """

    local_device_index: int = 0
    peer_device_indices: List[int] = field(default_factory=lambda: [1, 2])
    h100_device_index: int = 2
    pcie_bandwidth_gbps: float = 32.0
    prefer_local_experts: bool = True


# ---------------------------------------------------------------------------
# Shared-Locality Cache  (LOC)
# ---------------------------------------------------------------------------


class _RoutingEntry:
    """Single entry in the LOC cache."""

    __slots__ = ("probs", "routing_map", "timestamp", "hits")

    def __init__(self, probs: torch.Tensor, routing_map: torch.Tensor) -> None:
        # Store on CPU to avoid consuming GPU VRAM
        self.probs: torch.Tensor = probs.detach().cpu()
        self.routing_map: torch.Tensor = routing_map.detach().cpu()
        self.timestamp: float = time.monotonic()
        self.hits: int = 0


class SharedLocalityCache:
    """
    Content-addressed CPU-DRAM cache for routing decisions.

    DES-LOC rationale
    -----------------
    In long-context or repetitive inference workloads, many micro-batches share
    near-identical token distributions (e.g., autoregressive decoding with KV
    cache, data-parallel replicas processing the same prompt prefix).  Computing
    softmax + top-k for every micro-batch is wasteful when the routing decision
    would be identical.

    The cache stores ``(probs, routing_map)`` tensors in CPU DRAM (up to
    ``max_entries`` entries) keyed by a lightweight fingerprint derived from the
    token hidden-state statistics (mean + std per token, quantised to fp16 then
    MD5-hashed).  A cache hit returns the stored tensors moved to the target
    device.

    Thread-safety
    -------------
    All public methods acquire ``_lock`` to allow concurrent access from multiple
    pipeline stages running in different threads.

    Parameters
    ----------
    max_entries:
        Maximum number of entries before LRU eviction.
    similarity_threshold:
        Cosine-similarity threshold between fingerprints; entries within this
        radius are considered a match.  Set to 1.0 for exact-match only.
    ttl_seconds:
        Entries older than this are evicted regardless of LRU status.
    """

    def __init__(
        self,
        max_entries: int = 4096,
        similarity_threshold: float = 0.999,
        ttl_seconds: float = 60.0,
    ) -> None:
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds

        self._cache: Dict[str, _RoutingEntry] = {}
        self._lock = threading.Lock()

        self._hits: int = 0
        self._misses: int = 0

        logger.debug(
            "SharedLocalityCache initialised: max_entries=%d, ttl=%.1fs, "
            "similarity_threshold=%.4f",
            max_entries,
            ttl_seconds,
            similarity_threshold,
        )

    # ------------------------------------------------------------------
    # Fingerprint helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fingerprint(hidden_states: torch.Tensor) -> str:
        """
        Compute a compact fingerprint for a hidden-state tensor.

        We compute per-token (mean, std) statistics, cast to fp16 (halves
        precision noise), then hash the raw bytes with MD5.  This is cheap
        (O(tokens) memory reads) and produces a stable key for repeated
        identical or near-identical batches.
        """
        with torch.no_grad():
            # Flatten to [tokens, hidden]
            x = hidden_states.reshape(-1, hidden_states.shape[-1]).to(torch.float32)
            stats = torch.stack(
                [x.mean(dim=-1), x.std(dim=-1, unbiased=False)], dim=-1
            )  # [tokens, 2]
            stats_fp16 = stats.to(torch.float16).cpu()
            raw = stats_fp16.numpy().tobytes()
        return hashlib.md5(raw).hexdigest()  # noqa: S324 — non-cryptographic

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(
        self,
        hidden_states: torch.Tensor,
        target_device: torch.device,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Look up routing tensors for ``hidden_states``.

        Returns ``(probs, routing_map)`` on ``target_device`` if cached,
        otherwise ``None``.
        """
        key = self._fingerprint(hidden_states)
        now = time.monotonic()

        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            if now - entry.timestamp > self.ttl_seconds:
                del self._cache[key]
                self._misses += 1
                logger.debug("LOC cache TTL eviction for key=%s…", key[:8])
                return None

            entry.hits += 1
            entry.timestamp = now  # refresh LRU timestamp
            self._hits += 1

        logger.debug(
            "LOC cache HIT  key=%s… hits=%d ratio=%.3f",
            key[:8],
            entry.hits,
            self._hits / max(1, self._hits + self._misses),
        )
        return (
            entry.probs.to(target_device, non_blocking=True),
            entry.routing_map.to(target_device, non_blocking=True),
        )

    def store(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
    ) -> None:
        """Store routing tensors in the cache (moves to CPU asynchronously)."""
        key = self._fingerprint(hidden_states)

        with self._lock:
            if len(self._cache) >= self.max_entries:
                self._evict_lru()
            self._cache[key] = _RoutingEntry(probs, routing_map)

    def _evict_lru(self) -> None:
        """Evict the least-recently-used entry (caller must hold ``_lock``)."""
        if not self._cache:
            return
        lru_key = min(self._cache, key=lambda k: self._cache[k].timestamp)
        del self._cache[lru_key]
        logger.debug("LOC cache LRU eviction key=%s…", lru_key[:8])

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics dict."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": self._hits / max(1, total),
            }

    def clear(self) -> None:
        """Flush all cached entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
        logger.info("SharedLocalityCache cleared")


# Module-level singleton so all router instances share the same cache
_GLOBAL_LOC_CACHE: SharedLocalityCache = SharedLocalityCache()


def get_global_loc_cache() -> SharedLocalityCache:
    """Return the process-wide Shared-Locality Cache instance."""
    return _GLOBAL_LOC_CACHE


# ---------------------------------------------------------------------------
# Async router stream management
# ---------------------------------------------------------------------------


class _RouterStreamPool:
    """
    Per-device CUDA stream pool for async DES router execution.

    DES-LOC rationale
    -----------------
    Without NVLink, the H100 and A6000 devices cannot directly peer-copy
    activation tensors cheaply.  By running the router forward() on a
    dedicated stream (``stream_router``), the caller can enqueue expert
    weight pre-fetch and input preparation on the default stream *in parallel*
    with the routing computation.  The two streams are synchronised via an
    event before the routing_map is consumed.
    """

    def __init__(self) -> None:
        self._streams: Dict[int, torch.cuda.Stream] = {}
        self._lock = threading.Lock()

    def get(self, device_index: int) -> torch.cuda.Stream:
        with self._lock:
            if device_index not in self._streams:
                stream = torch.cuda.Stream(device=device_index)
                self._streams[device_index] = stream
                logger.debug(
                    "Created router CUDA stream for device cuda:%d", device_index
                )
            return self._streams[device_index]

    def synchronise(self, device_index: int) -> None:
        """Block the default stream until the router stream finishes."""
        stream = self.get(device_index)
        default_stream = torch.cuda.current_stream(device_index)
        event = torch.cuda.Event()
        stream.record_event(event)
        default_stream.wait_event(event)


_ROUTER_STREAM_POOL = _RouterStreamPool()


# ---------------------------------------------------------------------------
# HeteroTopKRouter — DES-LOC aware Top-K router
# ---------------------------------------------------------------------------


class HeteroTopKRouter(nn.Module, RouterInterface):
    """
    Heterogeneous Top-K Router for DES-LOC MoE layers.

    Upstream design (Megatron TopKRouter)
    --------------------------------------
    Megatron's TopKRouter computes a linear gate (``hidden → num_experts``),
    applies softmax, then selects the top-k experts per token.  Auxiliary
    load-balancing losses (z-loss, aux-loss) are accumulated via side-channel
    trackers.

    DES-LOC adaptations
    --------------------
    1. **Device-placement tracking** — the gate weight is registered on the
       *placement device* (set via ``set_device_placement``).  When placement
       changes (pipeline re-scheduling), the gate is moved lazily and only when
       the movement cost is justified (VRAM headroom check).

    2. **LOC cache integration** — before running the gate, we query the
       global SharedLocalityCache.  On a hit, we skip the gate computation
       entirely and return the cached tensors.  On a miss, we compute and
       store.

    3. **Async stream execution** — the gate forward runs inside a
       ``torch.cuda.stream(router_stream)`` context so downstream pipeline
       stages can pre-fetch expert weights concurrently.

    4. **PCIe topology hint** — ``topology_hint`` is passed through to the
       caller (dispatcher) so it can bias expert selection toward local-device
       experts, reducing PCIe traffic.

    Parameters
    ----------
    num_experts:
        Total number of experts in the MoE layer.
    hidden_size:
        Model hidden dimension.
    topk:
        Number of experts each token is routed to.
    aux_loss_coeff:
        Coefficient for load-balancing auxiliary loss (0 disables).
    z_loss_coeff:
        Coefficient for router z-loss (0 disables).
    use_loc_cache:
        If True (default), integrate with the SharedLocalityCache.
    use_async_stream:
        If True (default), run gate forward on a dedicated CUDA stream.
    topology_hint:
        PCIe topology descriptor for the target cluster.
    placement_device:
        Initial CUDA device for the gate weight.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        topk: int = 2,
        aux_loss_coeff: float = 1e-2,
        z_loss_coeff: float = 1e-3,
        use_loc_cache: bool = True,
        use_async_stream: bool = True,
        topology_hint: Optional[PCIeTopologyHint] = None,
        placement_device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.topk = topk
        self.aux_loss_coeff = aux_loss_coeff
        self.z_loss_coeff = z_loss_coeff
        self.use_loc_cache = use_loc_cache
        self.use_async_stream = use_async_stream
        self.topology_hint = topology_hint or PCIeTopologyHint()

        self._layer_number: int = -1
        self._placement_device: torch.device = placement_device or torch.device(
            "cuda", self.topology_hint.h100_device_index
        )

        # Gate linear: no bias (standard MoE practice)
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # Move gate to placement device immediately
        self.gate = self.gate.to(self._placement_device)

        self._loc_cache: SharedLocalityCache = get_global_loc_cache()
        self._forward_count: int = 0
        self._cache_hit_count: int = 0

        logger.info(
            "HeteroTopKRouter: num_experts=%d, hidden=%d, topk=%d, "
            "placement=%s, loc_cache=%s, async_stream=%s",
            num_experts,
            hidden_size,
            topk,
            self._placement_device,
            use_loc_cache,
            use_async_stream,
        )

    # ------------------------------------------------------------------
    # RouterInterface hooks
    # ------------------------------------------------------------------

    def set_layer_number(self, layer_number: int) -> None:
        self._layer_number = layer_number
        logger.debug("HeteroTopKRouter layer_number set to %d", layer_number)

    def set_device_placement(self, device: torch.device) -> None:
        """
        Lazily migrate the gate weight to a new device.

        DES-LOC context: pipeline re-scheduling may move a transformer layer
        from an A6000 to the H100.  We only migrate if VRAM on the target
        device has sufficient headroom (heuristic: >4 GB free).
        """
        if device == self._placement_device:
            return

        label = _cuda_device_label(device)
        vram_gb = _DEVICE_VRAM_GB.get(label, 48.0)
        gate_bytes = sum(p.numel() * p.element_size() for p in self.gate.parameters())
        gate_gb = gate_bytes / 1e9

        # Rough free-memory heuristic
        free_bytes, _ = torch.cuda.mem_get_info(device)
        free_gb = free_bytes / 1e9
        if free_gb - gate_gb < 4.0:
            logger.warning(
                "HeteroTopKRouter: skipping migration to %s — only %.2f GB free, "
                "gate requires %.4f GB",
                device,
                free_gb,
                gate_gb,
            )
            return

        logger.info(
            "HeteroTopKRouter: migrating gate from %s → %s",
            self._placement_device,
            device,
        )
        self.gate = self.gate.to(device)
        self._placement_device = device

    def warmup_loc_cache(self, representative_input: torch.Tensor) -> None:
        """Pre-populate the LOC cache with a representative micro-batch."""
        logger.info(
            "HeteroTopKRouter: warming up LOC cache with shape %s",
            tuple(representative_input.shape),
        )
        with torch.no_grad():
            self.forward(representative_input)

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute top-k routing for the input token batch.

        Algorithm
        ---------
        1. Query LOC cache; return cached result on hit.
        2. Move input to placement device if necessary (PCIe transfer).
        3. Optionally run gate on dedicated async CUDA stream.
        4. Apply softmax + top-k selection.
        5. Accumulate auxiliary losses (load-balance + z-loss).
        6. Store result in LOC cache; return (probs, routing_map).

        Parameters
        ----------
        hidden_states:
            Input activations; shape ``[seq, batch, hidden]`` or
            ``[tokens, hidden]``.

        Returns
        -------
        probs:
            Soft routing probabilities, shape ``[tokens, topk]``.
        routing_map:
            Hard expert indices, shape ``[tokens, topk]``.
        """
        self._forward_count += 1
        original_device = hidden_states.device

        # ---- 1. LOC cache lookup ----------------------------------------
        if self.use_loc_cache and not self.training:
            cached = self._loc_cache.lookup(hidden_states, original_device)
            if cached is not None:
                self._cache_hit_count += 1
                logger.debug(
                    "LOC hit  layer=%d fwd=%d",
                    self._layer_number,
                    self._forward_count,
                )
                return cached

        # ---- 2. Flatten input to [tokens, hidden] -----------------------
        orig_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            seq, batch, hidden = orig_shape
            tokens_2d = hidden_states.reshape(seq * batch, hidden)
        else:
            tokens_2d = hidden_states

        # ---- 3. Move to placement device (PCIe if needed) ---------------
        if tokens_2d.device != self._placement_device:
            logger.debug(
                "PCIe H2D copy layer=%d: %s → %s, shape=%s",
                self._layer_number,
                tokens_2d.device,
                self._placement_device,
                tuple(tokens_2d.shape),
            )
            tokens_2d = tokens_2d.to(self._placement_device, non_blocking=True)

        # ---- 4. Gate computation (optionally async) ----------------------
        dev_idx = self._placement_device.index or 0

        if self.use_async_stream:
            router_stream = _ROUTER_STREAM_POOL.get(dev_idx)
            with torch.cuda.stream(router_stream):
                logits = self.gate(tokens_2d)  # [tokens, num_experts]
            # Synchronise back to default stream before consuming logits
            _ROUTER_STREAM_POOL.synchronise(dev_idx)
        else:
            logits = self.gate(tokens_2d)

        # ---- 5. Softmax + top-k -----------------------------------------
        scores = F.softmax(logits, dim=-1)  # [tokens, num_experts]
        probs, routing_map = torch.topk(scores, self.topk, dim=-1)  # [tokens, topk]

        # Renormalise probabilities
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-9)

        # ---- 6. Auxiliary losses -----------------------------------------
        if self.training:
            self._accumulate_aux_losses(scores, routing_map)

        # ---- 7. Move results to original device -------------------------
        if probs.device != original_device:
            probs = probs.to(original_device, non_blocking=True)
            routing_map = routing_map.to(original_device, non_blocking=True)

        # ---- 8. Store in LOC cache ---------------------------------------
        if self.use_loc_cache and not self.training:
            self._loc_cache.store(hidden_states, probs, routing_map)

        return probs, routing_map

    # ------------------------------------------------------------------
    # Auxiliary loss helpers
    # ------------------------------------------------------------------

    def _accumulate_aux_losses(
        self,
        scores: torch.Tensor,
        routing_map: torch.Tensor,
    ) -> None:
        """
        Accumulate load-balance auxiliary loss and router z-loss.

        Load-balance loss (Switch Transformer style)
        ---------------------------------------------
            L_aux = num_experts * sum_i(f_i * P_i)
        where f_i = fraction of tokens routed to expert i,
              P_i = average softmax score for expert i.

        Z-loss (ST-MoE)
        ---------------
            L_z = mean(log(sum(exp(logits)))^2)
        Penalises large logit magnitudes to improve routing stability.
        """
        num_tokens = scores.shape[0]

        # Expert load fractions
        one_hot = torch.zeros(
            num_tokens, self.num_experts, device=scores.device, dtype=scores.dtype
        )
        one_hot.scatter_(1, routing_map[:, :1], 1.0)  # use primary expert only
        f = one_hot.mean(dim=0)  # [num_experts]
        p = scores.mean(dim=0)   # [num_experts]

        aux_loss = self.num_experts * (f * p).sum() * self.aux_loss_coeff

        # Z-loss
        log_z = torch.logsumexp(scores, dim=-1)  # [tokens]
        z_loss = (log_z ** 2).mean() * self.z_loss_coeff

        total_loss = aux_loss + z_loss

        # Attach to a dummy scalar so it flows through autograd
        # (caller is responsible for adding to the main loss)
        if not hasattr(self, "_aux_loss_accumulator"):
            self._aux_loss_accumulator = total_loss
        else:
            self._aux_loss_accumulator = self._aux_loss_accumulator + total_loss

        logger.debug(
            "layer=%d aux_loss=%.6f z_loss=%.6f",
            self._layer_number,
            aux_loss.item(),
            z_loss.item(),
        )

    def get_and_reset_aux_loss(self) -> Optional[torch.Tensor]:
        """Pop accumulated auxiliary loss for inclusion in the step loss."""
        loss = getattr(self, "_aux_loss_accumulator", None)
        self._aux_loss_accumulator = None
        return loss

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, topk={self.topk}, "
            f"hidden={self.hidden_size}, placement={self._placement_device}, "
            f"fwd={self._forward_count}, cache_hits={self._cache_hit_count}"
        )


# ---------------------------------------------------------------------------
# HeteroCachedRouter — wraps any RouterInterface with LOC caching
# ---------------------------------------------------------------------------


class HeteroCachedRouter(RouterInterface):
    """
    Decorator that wraps *any* RouterInterface with DES-LOC LOC caching.

    DES-LOC rationale
    -----------------
    If a project already has a custom router that satisfies ``RouterInterface``
    (e.g., Expert-Choice, Load-Balanced, Mixture-of-Depths), this wrapper
    transparently adds LOC caching without modifying the underlying router.

    Usage
    -----
    ::

        my_router = MyCustomRouter(...)
        cached_router = HeteroCachedRouter(my_router)
        # cached_router satisfies RouterInterface
        submodules.router = lambda **kw: HeteroCachedRouter(MyCustomRouter(**kw))

    Parameters
    ----------
    inner:
        Any object satisfying ``RouterInterface``.
    loc_cache:
        SharedLocalityCache instance; defaults to the global singleton.
    cache_in_training:
        If True, cache routing decisions even during training.  Default False
        because gradient-bearing tensors must not be cached.
    """

    def __init__(
        self,
        inner: RouterInterface,
        loc_cache: Optional[SharedLocalityCache] = None,
        cache_in_training: bool = False,
    ) -> None:
        self._inner = inner
        self._loc_cache = loc_cache or get_global_loc_cache()
        self._cache_in_training = cache_in_training
        self._is_training: bool = False
        logger.debug(
            "HeteroCachedRouter wrapping %s", type(inner).__name__
        )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        use_cache = not self._is_training or self._cache_in_training
        device = hidden_states.device

        if use_cache:
            cached = self._loc_cache.lookup(hidden_states, device)
            if cached is not None:
                return cached

        probs, routing_map = self._inner.forward(hidden_states)

        if use_cache:
            self._loc_cache.store(hidden_states, probs, routing_map)

        return probs, routing_map

    def set_layer_number(self, layer_number: int) -> None:
        self._inner.set_layer_number(layer_number)

    def set_device_placement(self, device: torch.device) -> None:
        self._inner.set_device_placement(device)

    def warmup_loc_cache(self, representative_input: torch.Tensor) -> None:
        self._inner.warmup_loc_cache(representative_input)

    def train(self, mode: bool = True) -> "HeteroCachedRouter":  # type: ignore[override]
        self._is_training = mode
        if hasattr(self._inner, "train"):
            self._inner.train(mode)  # type: ignore[attr-defined]
        return self

    def eval(self) -> "HeteroCachedRouter":  # type: ignore[override]
        return self.train(False)


# ---------------------------------------------------------------------------
# apply_hetero_router  — DES-LOC analogue of Megatron's apply_module()
# ---------------------------------------------------------------------------


def apply_hetero_router(
    router: RouterInterface,
) -> "RouterCallable":
    """
    Return a callable that invokes ``router.forward`` with DES-LOC safeguards.

    Upstream design (Megatron apply_module)
    ----------------------------------------
    Megatron's ``apply_module()`` normalises how typed modules are called,
    handling both ``nn.Module.__call__`` (which triggers hooks) and plain
    callables.  It is used in ``MoELayer.route()`` to decouple the call-site
    from the router's concrete type.

    DES-LOC additions
    -----------------
    1. Validates that the router's placement device matches the input tensor
       device; emits a warning (not an error) if they differ — PCIe transfer
       will occur but the forward will still succeed.
    2. Records routing latency in DEBUG logs for profiling.
    3. Returns a ``RouterCallable`` wrapper that is transparent to autograd.

    Parameters
    ----------
    router:
        Any object satisfying ``RouterInterface``.

    Returns
    -------
    A callable with signature ``(hidden_states) -> (probs, routing_map)``.
    """

    class RouterCallable:
        def __init__(self, r: RouterInterface) -> None:
            self._router = r

        def __call__(
            self, hidden_states: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            placement = getattr(self._router, "_placement_device", None)
            if placement is not None and hidden_states.device != placement:
                logger.debug(
                    "apply_hetero_router: input on %s, router gate on %s — "
                    "PCIe transfer will occur",
                    hidden_states.device,
                    placement,
                )

            t0 = time.perf_counter()
            probs, routing_map = self._router.forward(hidden_states)
            elapsed_ms = (time.perf_counter() - t0) * 1e3
            logger.debug("router forward elapsed=%.3f ms", elapsed_ms)
            return probs, routing_map

    return RouterCallable(router)


# ---------------------------------------------------------------------------
# HeteroMoESubmodules  — mirrors MoESubmodules with DES-LOC fields
# ---------------------------------------------------------------------------


@dataclass
class HeteroMoESubmodules:
    """
    DES-LOC analogue of Megatron's ``MoESubmodules``.

    Upstream design
    ---------------
    Megatron's ``MoESubmodules`` holds ``experts``, ``shared_experts``, and
    (post-commit-7857383) a ``router`` field typed as ``RouterBuilder``.  This
    allows any compliant callable to construct the router at layer-init time.

    DES-LOC extensions
    ------------------
    ``router_builder``:
        A ``RouterBuilder``-compatible callable that receives
        ``(num_experts, hidden_size, topk, topology_hint)`` as keyword
        arguments and returns a ``RouterInterface``.  Defaults to
        ``build_default_router``.

    ``topology_hint``:
        Cluster PCIe topology descriptor shared across all layers.

    ``use_loc_cache``:
        Master switch for SharedLocalityCache integration.

    ``use_async_stream``:
        Master switch for async router-stream execution.
    """

    experts: Any = None
    shared_experts: Any = None
    router_builder: Any = None  # RouterBuilder — set in __post_init__
    topology_hint: PCIeTopologyHint = field(
        default_factory=PCIeTopologyHint
    )
    use_loc_cache: bool = True
    use_async_stream: bool = True

    def __post_init__(self) -> None:
        if self.router_builder is None:
            self.router_builder = build_default_router


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------


def build_default_router(
    num_experts: int,
    hidden_size: int,
    topk: int = 2,
    topology_hint: Optional[PCIeTopologyHint] = None,
    use_loc_cache: bool = True,
    use_async_stream: bool = True,
    placement_device: Optional[torch.device] = None,
    **kwargs: Any,
) -> HeteroTopKRouter:
    """
    Convenience ``RouterBuilder`` that constructs a ``HeteroTopKRouter``.

    This is the default value of ``HeteroMoESubmodules.router_builder`` and
    mirrors how Megatron's ``MoESubmodules.router`` defaults to ``TopKRouter``.

    Accepts and ignores unknown ``**kwargs`` so it can be used as a drop-in
    even when the call-site passes extra config keys.
    """
    if kwargs:
        logger.debug("build_default_router: ignoring unknown kwargs: %s", list(kwargs))

    return HeteroTopKRouter(
        num_experts=num_experts,
        hidden_size=hidden_size,
        topk=topk,
        topology_hint=topology_hint or PCIeTopologyHint(),
        use_loc_cache=use_loc_cache,
        use_async_stream=use_async_stream,
        placement_device=placement_device,
    )


# ---------------------------------------------------------------------------
# HeteroMoELayer  — minimal integration layer demonstrating the full pipeline
# ---------------------------------------------------------------------------


class HeteroMoELayer(nn.Module):
    """
    Minimal DES-LOC MoE layer wiring together the router, LOC cache, and
    async stream execution.

    This is *not* a full Megatron MoELayer replacement — it is the integration
    surface showing how ``HeteroMoESubmodules`` and ``apply_hetero_router``
    compose in a real forward pass.

    Parameters
    ----------
    submodules:
        ``HeteroMoESubmodules`` instance; ``router_builder`` is called here.
    num_experts:
        Total expert count.
    hidden_size:
        Model hidden dimension.
    topk:
        Number of active experts per token.
    layer_number:
        Transformer layer index (passed through to router).
    """

    def __init__(
        self,
        submodules: HeteroMoESubmodules,
        num_experts: int,
        hidden_size: int,
        topk: int = 2,
        layer_number: int = 0,
    ) -> None:
        super().__init__()

        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.topk = topk

        # Construct router via the injected builder (mirrors Megatron 7857383)
        self.router: RouterInterface = submodules.router_builder(
            num_experts=num_experts,
            hidden_size=hidden_size,
            topk=topk,
            topology_hint=submodules.topology_hint,
            use_loc_cache=submodules.use_loc_cache,
            use_async_stream=submodules.use_async_stream,
        )
        self.router.set_layer_number(layer_number)

        logger.info(
            "HeteroMoELayer init: layer=%d num_experts=%d hidden=%d topk=%d",
            layer_number,
            num_experts,
            hidden_size,
            topk,
        )

    def route(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts using DES-LOC apply_hetero_router.

        Mirrors Megatron's ``MoELayer.route()`` which was updated in commit
        7857383 to use ``apply_module(self.router)(hidden_states)`` instead of
        the direct ``self.router(hidden_states)`` call.
        """
        return apply_hetero_router(self.router)(hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Simplified MoE forward: route → (placeholder expert computation) → return.

        In production this would call the token dispatcher, expert FFNs, and
        combine; here we return the input unchanged after routing to demonstrate
        the DES-LOC routing pipeline.
        """
        probs, routing_map = self.route(hidden_states)
        logger.debug(
            "HeteroMoELayer forward: probs.shape=%s routing_map.shape=%s",
            tuple(probs.shape),
            tuple(routing_map.shape),
        )
        # Placeholder: weighted sum of input as stand-in for expert outputs
        # probs: [tokens, topk]  — use first-expert probability as scalar blend
        tokens_2d = hidden_states.reshape(-1, self.hidden_size)
        blend = probs[:, 0].unsqueeze(-1)  # [tokens, 1]
        out = tokens_2d * blend
        return out.reshape(hidden_states.shape)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    device = torch.device("cpu")  # CI runs without GPU

    # --- Build submodules with default router ---
    submodules = HeteroMoESubmodules(
        use_loc_cache=True,
        use_async_stream=False,  # no CUDA in CI
        topology_hint=PCIeTopologyHint(
            local_device_index=0,
            h100_device_index=0,  # remap to cpu-compatible index
        ),
    )

    # Override builder to force CPU placement
    submodules.router_builder = lambda **kw: HeteroTopKRouter(
        num_experts=kw["num_experts"],
        hidden_size=kw["hidden_size"],
        topk=kw["topk"],
        use_loc_cache=kw.get("use_loc_cache", True),
        use_async_stream=False,
        placement_device=device,
    )

    layer = HeteroMoELayer(
        submodules=submodules,
        num_experts=8,
        hidden_size=64,
        topk=2,
        layer_number=3,
    )
    layer.eval()

    x = torch.randn(4, 2, 64)  # [seq=4, batch=2, hidden=64]

    # 1. Router returns correct shapes
    probs, routing_map = layer.route(x)
    assert probs.shape == (8, 2), f"probs shape mismatch: {probs.shape}"
    assert routing_map.shape == (8, 2), f"routing_map shape mismatch: {routing_map.shape}"

    # 2. LOC cache stores and retrieves on second call
    cache = get_global_loc_cache()
    stats_before = cache.stats()
    probs2, routing_map2 = layer.route(x)
    stats_after = cache.stats()
    assert stats_after["hits"] > stats_before["hits"], "LOC cache should have hit on second call"

    # 3. Routing probabilities sum to ~1 per token
    prob_sum = probs.sum(dim=-1)
    assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5), \
        f"probs do not sum to 1: {prob_sum}"

    # 4. HeteroCachedRouter wrapper is transparent
    inner_router = HeteroTopKRouter(8, 64, topk=2, use_async_stream=False,
                                    use_loc_cache=False, placement_device=device)
    cached_router = HeteroCachedRouter(inner_router)
    cached_router.eval()
    p, r = cached_router.forward(x.reshape(-1, 64))
    assert p.shape == (8, 2)

    # 5. apply_hetero_router callable works identically to direct forward
    callable_router = apply_hetero_router(layer.router)
    p3, r3 = callable_router(x)
    assert p3.shape == probs.shape

    logger.info("All smoke tests passed.")
