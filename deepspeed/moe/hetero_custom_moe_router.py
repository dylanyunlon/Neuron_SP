"""
DES-LOC Heterogeneous Custom MoE Router
========================================

Upstream design intent (Megatron 7857383):
    Megatron-LM's commit 7857383 decoupled the router from MoELayer by introducing
    ``RouterInterface`` and ``RouterBuilder`` Protocols, replacing the hard-coded
    ``TopKRouter`` instantiation with a pluggable ``submodules.router`` callable.
    The key insight is that ``apply_module()`` wraps any callable conforming to
    ``RouterInterface.forward`` so that CUDAGraph capture, TE fused kernels, and
    type-checked dispatch all work uniformly regardless of which concrete router
    class is provided.

DES-LOC adaptation points:
    1. **Device-aware router placement** – In our 2×A6000 (SM86) + 1×H100 NVL (SM90)
       topology, the gating network (router) runs on whichever device holds the
       hidden states for that micro-batch. We cannot assume a uniform CUDA device;
       ``HeteroRouterInterface`` therefore carries an explicit ``device`` attribute
       that is resolved at construction time by ``DeviceAffinityResolver``.

    2. **Shared Locality Cache (LOC)** – DES-LOC's key innovation: routing decisions
       are cheap to reuse across micro-batches when the token distribution is stable.
       ``LocalityCacheManager`` stores (probs, routing_map) keyed by a lightweight
       hash of the input statistics. Cache hits skip the gating forward pass entirely,
       reducing cross-device PCIe traffic (critical because A6000↔H100 share only
       PCIe bandwidth, no NVLink).

    3. **Decoupled Execution (DES)** – Expert computation is submitted as async work
       items to a per-device executor pool. The router's ``route_async`` method returns
       a ``RoutingFuture`` so that the caller can overlap token dispatch with other
       compute. This mirrors Megatron's ``apply_module`` wrapper but adds explicit
       stream management across heterogeneous SM generations.

    4. **SM-generation-aware top-k kernel selection** – SM90 (H100) supports warp-
       specialised top-k via CUTLASS; SM86 (A6000) falls back to the standard
       ``torch.topk`` path. ``RouterKernelDispatcher`` selects at construction time.

    5. **Custom router injection** – Mirrors ``RouterBuilder`` Protocol: any callable
       ``(config, device_profile) -> HeteroRouterInterface`` can be injected, enabling
       research into heterogeneity-aware gating (e.g. load-balancing across A6000/H100
       expert pools with different throughput characteristics).

References:
    - Megatron-LM commit 7857383d3a3c0fac00a87546eb9092a7bea36e53
    - DES-LOC design doc: docs/des_loc_design.md (Neuron_SP internal)
    - PCIe bandwidth budget: ~32 GB/s unidirectional A6000↔H100
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, NamedTuple, Optional, Protocol, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware topology constants for 2×A6000 + 1×H100 NVL cluster
# ---------------------------------------------------------------------------

_SM_GENERATION: Dict[int, str] = {
    86: "ampere",   # A6000 48GB
    90: "hopper",   # H100 NVL 96GB
}

_DEVICE_MEMORY_GB: Dict[str, float] = {
    "ampere": 48.0,
    "hopper": 96.0,
}

_PCIE_BW_GBPS: float = 32.0  # PCIe Gen4 ×16, ~32 GB/s unidirectional


# ---------------------------------------------------------------------------
# DeviceProfile: static description of one physical device
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeviceProfile:
    """Immutable description of a single CUDA device in the DES-LOC cluster.

    Attributes:
        device_index: CUDA device ordinal (0, 1, 2, ...).
        sm_major: SM major version (e.g. 8 for Ampere/SM86, 9 for Hopper/SM90).
        sm_minor: SM minor version.
        memory_gb: Total VRAM in gigabytes.
        arch_name: Human-readable architecture name.
        supports_warp_specialised_topk: True for SM90+ (H100) only.
    """
    device_index: int
    sm_major: int
    sm_minor: int
    memory_gb: float
    arch_name: str
    supports_warp_specialised_topk: bool

    @classmethod
    def from_device(cls, device_index: int) -> "DeviceProfile":
        """Construct from a live CUDA device index."""
        props = torch.cuda.get_device_properties(device_index)
        sm_ver = props.major * 10 + props.minor
        arch = _SM_GENERATION.get(sm_ver, f"sm{sm_ver}")
        mem_gb = props.total_memory / (1024 ** 3)
        return cls(
            device_index=device_index,
            sm_major=props.major,
            sm_minor=props.minor,
            memory_gb=round(mem_gb, 1),
            arch_name=arch,
            supports_warp_specialised_topk=(props.major >= 9),
        )

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.device_index)


# ---------------------------------------------------------------------------
# DeviceAffinityResolver: maps hidden-state tensors to DeviceProfiles
# ---------------------------------------------------------------------------

class DeviceAffinityResolver:
    """Resolves which DeviceProfile owns a tensor at runtime.

    In DES-LOC, each micro-batch's hidden states may live on a different
    physical device (A6000-0, A6000-1, or H100). The resolver caches
    DeviceProfile objects so the lookup is O(1) after the first call.

    Design note (mirrors Megatron's pg_collection awareness):
        Megatron passes ``pg_collection`` to the router builder so the router
        knows its tensor-parallel group. Here we extend that idea: the resolver
        is constructed once and injected into every router builder call, giving
        routers full topology awareness without global state.
    """

    def __init__(self) -> None:
        self._cache: Dict[int, DeviceProfile] = {}
        self._lock = threading.Lock()

    def resolve(self, tensor: torch.Tensor) -> DeviceProfile:
        """Return DeviceProfile for the device holding *tensor*."""
        idx = tensor.device.index
        if idx is None:
            raise ValueError("Tensor is on CPU; DES-LOC routers require CUDA tensors.")
        if idx not in self._cache:
            with self._lock:
                if idx not in self._cache:
                    profile = DeviceProfile.from_device(idx)
                    self._cache[idx] = profile
                    logger.debug(
                        "DeviceAffinityResolver: registered device %d "
                        "arch=%s mem=%.1f GB warp_topk=%s",
                        idx, profile.arch_name, profile.memory_gb,
                        profile.supports_warp_specialised_topk,
                    )
        return self._cache[idx]

    def all_profiles(self) -> Dict[int, DeviceProfile]:
        with self._lock:
            return dict(self._cache)


# ---------------------------------------------------------------------------
# Locality Cache (LOC) – core DES-LOC primitive
# ---------------------------------------------------------------------------

class _CacheEntry(NamedTuple):
    probs: torch.Tensor
    routing_map: torch.Tensor
    timestamp: float
    hit_count: int


@dataclass
class LocalityCacheConfig:
    """Tuning knobs for the Shared Locality Cache.

    Attributes:
        capacity: Maximum number of (probs, routing_map) pairs to keep.
        ttl_seconds: Entries older than this are evicted on next access.
        stat_hash_bins: Number of histogram bins used to fingerprint inputs.
            Coarser bins → more cache hits but lower routing quality.
        min_hit_rate_to_log: Log a warning when hit-rate drops below this.
    """
    capacity: int = 128
    ttl_seconds: float = 0.5
    stat_hash_bins: int = 16
    min_hit_rate_to_log: float = 0.10


class LocalityCacheManager:
    """Shared Locality Cache for routing decisions.

    Core DES-LOC mechanism: avoids re-running the gating network (and the
    associated PCIe transfer of routing tensors) when the token distribution
    has not meaningfully changed between consecutive micro-batches.

    Fingerprinting strategy:
        We compute a lightweight fingerprint of ``hidden_states`` using the
        mean and std of a coarsely quantised histogram of activation values.
        This is deliberately approximate: we trade recall for speed.  False
        positives (cache hit when distribution actually changed) produce
        slightly sub-optimal routing; false negatives (cache miss when a hit
        was possible) just add latency.  Both are acceptable in the PCIe-
        bandwidth-limited A6000↔H100 regime.

    Thread safety:
        All public methods acquire a per-instance lock, making the cache safe
        for use from multiple DES-LOC executor threads.
    """

    def __init__(self, cfg: LocalityCacheConfig = LocalityCacheConfig()) -> None:
        self.cfg = cfg
        self._store: Dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()
        self._total_queries = 0
        self._total_hits = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(
        self, key: str
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return cached (probs, routing_map) or None on miss/expiry."""
        with self._lock:
            self._total_queries += 1
            entry = self._store.get(key)
            if entry is None:
                return None
            if time.monotonic() - entry.timestamp > self.cfg.ttl_seconds:
                del self._store[key]
                return None
            # Refresh hit count
            self._store[key] = entry._replace(
                hit_count=entry.hit_count + 1,
                timestamp=time.monotonic(),
            )
            self._total_hits += 1
            return entry.probs, entry.routing_map

    def store(
        self,
        key: str,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
    ) -> None:
        """Insert or overwrite a cache entry, evicting LRU if at capacity."""
        with self._lock:
            if key in self._store:
                self._store[key] = _CacheEntry(
                    probs=probs.detach(),
                    routing_map=routing_map.detach(),
                    timestamp=time.monotonic(),
                    hit_count=self._store[key].hit_count,
                )
                return
            if len(self._store) >= self.cfg.capacity:
                self._evict_one()
            self._store[key] = _CacheEntry(
                probs=probs.detach(),
                routing_map=routing_map.detach(),
                timestamp=time.monotonic(),
                hit_count=0,
            )

    def fingerprint(self, hidden_states: torch.Tensor) -> str:
        """Compute a cheap string fingerprint of *hidden_states*.

        Uses a fixed-bin histogram of a random projection of the hidden
        states, making the fingerprint O(seq_len × proj_dim) in compute
        but only O(bins) in memory.
        """
        with torch.no_grad():
            # Flatten to [N, D], project to scalar per token, histogram
            x = hidden_states.detach().float().reshape(-1, hidden_states.shape[-1])
            # Deterministic projection: mean over last dim as proxy
            proj = x.mean(dim=-1)  # [N]
            hist = torch.histc(proj, bins=self.cfg.stat_hash_bins)
            # Normalise so scale-invariant
            hist = hist / (hist.sum() + 1e-9)
            digest = hashlib.md5(
                hist.cpu().numpy().tobytes(), usedforsecurity=False
            ).hexdigest()[:16]
        return digest

    def hit_rate(self) -> float:
        with self._lock:
            if self._total_queries == 0:
                return 0.0
            return self._total_hits / self._total_queries

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._store),
                "capacity": self.cfg.capacity,
                "total_queries": self._total_queries,
                "total_hits": self._total_hits,
                "hit_rate": round(self.hit_rate(), 4),
            }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evict_one(self) -> None:
        """Evict the entry with the oldest timestamp (LRU approximation)."""
        oldest_key = min(self._store, key=lambda k: self._store[k].timestamp)
        logger.debug("LocalityCacheManager: evicting key %s", oldest_key)
        del self._store[oldest_key]


# ---------------------------------------------------------------------------
# RouterKernelDispatcher: SM-generation-aware top-k selection
# ---------------------------------------------------------------------------

class RouterKernelDispatcher:
    """Selects the appropriate top-k kernel based on SM generation.

    SM90 (H100 NVL): uses ``_topk_hopper`` which calls into a warp-specialised
        kernel (falls back gracefully if CUTLASS is not available).
    SM86 (A6000): uses ``_topk_ampere`` which is standard ``torch.topk``.

    The dispatcher is constructed once per device and cached, so the
    ``isinstance`` / ``major`` checks do not appear in the hot path.
    """

    def __init__(self, profile: DeviceProfile) -> None:
        self.profile = profile
        if profile.supports_warp_specialised_topk:
            self._impl = self._topk_hopper
            logger.debug(
                "RouterKernelDispatcher: device %d (SM90) → warp-specialised topk",
                profile.device_index,
            )
        else:
            self._impl = self._topk_ampere
            logger.debug(
                "RouterKernelDispatcher: device %d (SM86) → standard topk",
                profile.device_index,
            )

    def topk(
        self, scores: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (values, indices) of top-k along last dimension."""
        return self._impl(scores, k)

    @staticmethod
    def _topk_ampere(
        scores: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.topk(scores, k, dim=-1, sorted=False)

    @staticmethod
    def _topk_hopper(
        scores: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Preferred path: warp-specialised CUTLASS top-k (SM90 only)
        # We attempt import; if unavailable, fall back to standard topk.
        try:
            import neuron_sp_kernels  # type: ignore[import]
            return neuron_sp_kernels.warp_topk(scores, k)
        except (ImportError, AttributeError):
            logger.debug(
                "RouterKernelDispatcher: neuron_sp_kernels unavailable, "
                "falling back to torch.topk on SM90"
            )
            return torch.topk(scores, k, dim=-1, sorted=False)


# ---------------------------------------------------------------------------
# HeteroRouterInterface – DES-LOC extension of Megatron's RouterInterface
# ---------------------------------------------------------------------------

class HeteroRouterInterface(Protocol):
    """DES-LOC extension of Megatron's RouterInterface.

    Adds:
        * ``device_profile`` property so callers know where this router lives.
        * ``route_async`` for non-blocking routing that returns a Future.
        * ``set_layer_number`` (mirrors Megatron RouterInterface).

    The ``forward`` signature is identical to Megatron's RouterInterface so
    that any HeteroRouterInterface is substitutable wherever a plain
    RouterInterface is expected.
    """

    @property
    def device_profile(self) -> DeviceProfile: ...

    def forward(
        self, hidden_states: torch.Tensor, /
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def route_async(
        self, hidden_states: torch.Tensor
    ) -> "RoutingFuture": ...

    def set_layer_number(self, layer_number: int) -> None: ...


class HeteroRouterBuilder(Protocol):
    """Protocol for building a HeteroRouter – mirrors Megatron's RouterBuilder."""

    def __call__(
        self,
        /,
        *,
        config: Any,
        device_profile: DeviceProfile,
        cache_manager: LocalityCacheManager,
    ) -> HeteroRouterInterface: ...


# ---------------------------------------------------------------------------
# RoutingFuture: async routing result with stream synchronisation
# ---------------------------------------------------------------------------

@dataclass
class RoutingFuture:
    """Wraps a concurrent.futures.Future holding (probs, routing_map).

    Callers can do other work while the gating network runs on a separate
    CUDA stream, then call ``.result()`` when the routing is needed.

    Attributes:
        _future: Underlying Python future.
        _stream: CUDA stream on which the routing computation was launched.
        layer_number: For logging / debugging.
        cache_hit: True if this result came from LocalityCacheManager.
    """
    _future: Future
    _stream: torch.cuda.Stream
    layer_number: int
    cache_hit: bool = False

    def result(
        self, timeout: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Block until routing is complete and return (probs, routing_map).

        Synchronises the calling thread's default CUDA stream with the
        stream on which routing was computed, ensuring correct ordering.
        """
        probs, routing_map = self._future.result(timeout=timeout)
        # Wait for routing stream to finish before current stream proceeds
        torch.cuda.current_stream().wait_stream(self._stream)
        return probs, routing_map


# ---------------------------------------------------------------------------
# BaseHeteroRouter: shared logic for all DES-LOC router implementations
# ---------------------------------------------------------------------------

class BaseHeteroRouter(nn.Module, ABC):
    """Abstract base for DES-LOC heterogeneous MoE routers.

    Handles:
        * Device affinity resolution and kernel dispatcher selection.
        * Shared Locality Cache integration (fingerprint → lookup → store).
        * Async routing via a per-device ThreadPoolExecutor + CUDA stream.
        * Hit-rate logging every ``_log_interval`` forward passes.

    Subclasses implement ``_compute_routing`` which receives pre-normalised
    scores and returns (probs, routing_map).

    Design mirrors Megatron's BaseMoELayer: shared state in the base,
    compute logic in the subclass.
    """

    _log_interval: int = 200  # log cache stats every N forward passes

    def __init__(
        self,
        num_experts: int,
        topk: int,
        hidden_size: int,
        device_profile: DeviceProfile,
        cache_manager: LocalityCacheManager,
        layer_number: int = 0,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.hidden_size = hidden_size
        self._device_profile = device_profile
        self._cache_manager = cache_manager
        self._layer_number = layer_number
        self._forward_count = 0

        # Gating weight: maps hidden_size → num_experts
        self.gate = nn.Linear(hidden_size, num_experts, bias=False,
                              device=device_profile.device)

        self._kernel_dispatcher = RouterKernelDispatcher(device_profile)
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"des_loc_router_dev{device_profile.device_index}",
        )
        self._routing_stream = torch.cuda.Stream(device=device_profile.device)

        logger.info(
            "BaseHeteroRouter: layer=%d device=%s arch=%s experts=%d topk=%d",
            layer_number, device_profile.device, device_profile.arch_name,
            num_experts, topk,
        )

    # ------------------------------------------------------------------
    # HeteroRouterInterface implementation
    # ------------------------------------------------------------------

    @property
    def device_profile(self) -> DeviceProfile:
        return self._device_profile

    def set_layer_number(self, layer_number: int) -> None:
        """Mirror Megatron RouterInterface.set_layer_number."""
        self._layer_number = layer_number
        logger.debug("BaseHeteroRouter: set layer_number=%d", layer_number)

    def forward(
        self, hidden_states: torch.Tensor, /
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Synchronous routing with LOC cache lookup.

        Flow:
            1. Fingerprint hidden_states → cache key.
            2. If LOC hit: return cached (probs, routing_map) immediately,
               avoiding gating network entirely (saves PCIe round-trip).
            3. If miss: run gating, store result in LOC.
            4. Log cache statistics every _log_interval calls.
        """
        self._forward_count += 1
        key = f"L{self._layer_number}_{self._cache_manager.fingerprint(hidden_states)}"
        cached = self._cache_manager.lookup(key)
        if cached is not None:
            if self._forward_count % self._log_interval == 0:
                self._log_cache_stats()
            return cached  # type: ignore[return-value]

        probs, routing_map = self._run_gating(hidden_states)
        self._cache_manager.store(key, probs, routing_map)

        if self._forward_count % self._log_interval == 0:
            self._log_cache_stats()
        return probs, routing_map

    def route_async(self, hidden_states: torch.Tensor) -> RoutingFuture:
        """Non-blocking routing: submit to executor, return RoutingFuture.

        The actual gating runs on ``_routing_stream`` inside the executor
        thread, so the caller can overlap token dispatch or other compute.
        """
        key = f"L{self._layer_number}_{self._cache_manager.fingerprint(hidden_states)}"
        cached = self._cache_manager.lookup(key)
        if cached is not None:
            f: Future = Future()
            f.set_result(cached)
            dummy_stream = torch.cuda.current_stream(self._device_profile.device)
            return RoutingFuture(_future=f, _stream=dummy_stream,
                                 layer_number=self._layer_number, cache_hit=True)

        stream = self._routing_stream

        def _work() -> Tuple[torch.Tensor, torch.Tensor]:
            with torch.cuda.stream(stream):
                p, r = self._run_gating(hidden_states)
            self._cache_manager.store(key, p, r)
            return p, r

        future = self._executor.submit(_work)
        return RoutingFuture(_future=future, _stream=stream,
                             layer_number=self._layer_number, cache_hit=False)

    # ------------------------------------------------------------------
    # Abstract compute hook
    # ------------------------------------------------------------------

    @abstractmethod
    def _compute_routing(
        self, scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given raw gate logits, return (probs, routing_map).

        Args:
            scores: [seq_len * batch, num_experts] gate logits.
        Returns:
            probs:       [seq_len * batch, topk] routing probabilities.
            routing_map: [seq_len * batch, num_experts] bool assignment map.
        """
        ...

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_gating(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run gate linear + _compute_routing."""
        orig_shape = hidden_states.shape
        x = hidden_states.reshape(-1, self.hidden_size)
        scores = self.gate(x)  # [N, E]
        probs, routing_map = self._compute_routing(scores)
        return probs, routing_map

    def _log_cache_stats(self) -> None:
        stats = self._cache_manager.stats()
        if stats["hit_rate"] < self._cache_manager.cfg.min_hit_rate_to_log:
            logger.warning(
                "LocalityCacheManager hit-rate below threshold: "
                "layer=%d hit_rate=%.3f queries=%d",
                self._layer_number, stats["hit_rate"], stats["total_queries"],
            )
        else:
            logger.debug(
                "LocalityCacheManager: layer=%d %s",
                self._layer_number, stats,
            )


# ---------------------------------------------------------------------------
# TopKHeteroRouter: DES-LOC reimplementation of Megatron TopKRouter
# ---------------------------------------------------------------------------

class TopKHeteroRouter(BaseHeteroRouter):
    """DES-LOC heterogeneous top-k router.

    Mirrors Megatron's TopKRouter gating logic while adding:
        - SM-aware top-k kernel dispatch via RouterKernelDispatcher.
        - Soft vs hard routing controlled by ``use_softmax_routing``.
        - Optional auxiliary load-balancing loss (mirrors Megatron's
          aux_loss_coeff, adapted for per-device expert pools).

    Upstream Megatron context:
        TopKRouter applies a linear gate, softmax normalisation, and
        selects the top-k experts per token, producing a sparse
        routing_map and associated probabilities.  DES-LOC preserves
        this semantic but dispatches the top-k computation to the
        SM-appropriate kernel.
    """

    def __init__(
        self,
        num_experts: int,
        topk: int,
        hidden_size: int,
        device_profile: DeviceProfile,
        cache_manager: LocalityCacheManager,
        layer_number: int = 0,
        use_softmax_routing: bool = True,
        aux_loss_coeff: float = 0.01,
        normalize_gate_prob_before_dropping: bool = False,
    ) -> None:
        super().__init__(
            num_experts=num_experts,
            topk=topk,
            hidden_size=hidden_size,
            device_profile=device_profile,
            cache_manager=cache_manager,
            layer_number=layer_number,
        )
        self.use_softmax_routing = use_softmax_routing
        self.aux_loss_coeff = aux_loss_coeff
        self.normalize_gate_prob_before_dropping = normalize_gate_prob_before_dropping

    def _compute_routing(
        self, scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-k gating with optional load-balancing aux loss.

        Args:
            scores: [N, E] raw gate logits.

        Returns:
            probs:       [N, topk] normalised routing weights.
            routing_map: [N, E] bool tensor, True where token routes to expert.
        """
        N, E = scores.shape

        if self.use_softmax_routing:
            gate_probs = F.softmax(scores, dim=-1)  # [N, E]
        else:
            gate_probs = scores

        # SM-aware top-k
        topk_vals, topk_indices = self._kernel_dispatcher.topk(gate_probs, self.topk)

        # Build sparse routing_map
        routing_map = torch.zeros(N, E, dtype=torch.bool, device=scores.device)
        routing_map.scatter_(1, topk_indices, True)

        # Normalise selected probabilities
        if self.normalize_gate_prob_before_dropping:
            probs = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)
        else:
            probs = topk_vals

        # Auxiliary load-balancing loss (only during training, matches Megatron)
        if self.training and self.aux_loss_coeff > 0.0:
            self._accumulate_aux_loss(gate_probs, routing_map)

        return probs, routing_map

    def _accumulate_aux_loss(
        self,
        gate_probs: torch.Tensor,
        routing_map: torch.Tensor,
    ) -> None:
        """Compute and store auxiliary load-balancing loss.

        Mirrors Megatron's switch_load_balancing_loss_func logic:
            aux_loss = coeff * E * sum_e(f_e * P_e)
        where f_e = fraction of tokens routed to expert e,
              P_e = mean gate probability for expert e.
        """
        N = gate_probs.shape[0]
        E = self.num_experts
        # f_e: fraction of tokens assigned to expert e
        f_e = routing_map.float().mean(dim=0)            # [E]
        # P_e: mean gate probability for expert e
        P_e = gate_probs.mean(dim=0)                     # [E]
        aux_loss = self.aux_loss_coeff * E * (f_e * P_e).sum()
        # Store on module for the training loop to retrieve
        self._last_aux_loss = aux_loss
        logger.debug(
            "TopKHeteroRouter: layer=%d aux_loss=%.6f",
            self._layer_number, aux_loss.item(),
        )


# ---------------------------------------------------------------------------
# HeteroMoESubmodules: DES-LOC equivalent of Megatron MoESubmodules
# ---------------------------------------------------------------------------

@dataclass
class HeteroMoESubmodules:
    """DES-LOC MoE layer submodule specification.

    Mirrors Megatron's MoESubmodules but replaces the plain ``RouterBuilder``
    with a ``HeteroRouterBuilder`` that receives ``device_profile`` and
    ``cache_manager`` instead of ``pg_collection``.

    Attributes:
        router: Callable matching HeteroRouterBuilder Protocol.
            Defaults to TopKHeteroRouter.
        experts: Expert module spec (same as Megatron).
        shared_experts: Shared expert spec (optional).
        cache_config: Tuning knobs for the Shared Locality Cache.
    """
    router: HeteroRouterBuilder = field(
        default_factory=lambda: _default_router_builder  # type: ignore[return-value]
    )
    experts: Optional[Any] = None
    shared_experts: Optional[Any] = None
    cache_config: LocalityCacheConfig = field(default_factory=LocalityCacheConfig)


def _default_router_builder(
    *,
    config: Any,
    device_profile: DeviceProfile,
    cache_manager: LocalityCacheManager,
) -> TopKHeteroRouter:
    """Default HeteroRouterBuilder: constructs a TopKHeteroRouter.

    Extracts hyperparameters from a DeepSpeed-style config dict or object.
    Falls back to sensible defaults so smoke tests run without a full config.
    """
    def _get(cfg: Any, key: str, default: Any) -> Any:
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    return TopKHeteroRouter(
        num_experts=_get(config, "num_moe_experts", 8),
        topk=_get(config, "moe_router_topk", 2),
        hidden_size=_get(config, "hidden_size", 1024),
        device_profile=device_profile,
        cache_manager=cache_manager,
        layer_number=0,
        use_softmax_routing=_get(config, "moe_use_softmax_routing", True),
        aux_loss_coeff=_get(config, "moe_aux_loss_coeff", 0.01),
    )


# ---------------------------------------------------------------------------
# HeteroMoELayer: top-level DES-LOC MoE layer
# ---------------------------------------------------------------------------

class HeteroMoELayer(nn.Module):
    """DES-LOC heterogeneous MoE layer.

    Mirrors Megatron's MoELayer structure but replaces:
        * ``TopKRouter`` hard-coded construction → pluggable HeteroRouterBuilder
          (mirrors commit 7857383's ``submodules.router`` injection).
        * ``apply_module(self.router)(hidden_states)`` → ``self.router(hidden_states)``
          via BaseHeteroRouter.forward (which handles stream/cache internally).
        * ``pg_collection`` → ``DeviceAffinityResolver`` + ``DeviceProfile``
          for heterogeneous device topology.

    Async usage:
        For maximum overlap, callers can use:
            future = layer.route_async(hidden_states)
            # ... other compute ...
            probs, routing_map = future.result()

    Args:
        config: DeepSpeed-compatible config dict or object.
        submodules: HeteroMoESubmodules specifying router builder and expert specs.
        layer_number: Transformer layer index (forwarded to router for logging).
    """

    def __init__(
        self,
        config: Any,
        submodules: Optional[HeteroMoESubmodules] = None,
        layer_number: int = 0,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        submodules = submodules or HeteroMoESubmodules()
        self._cache_manager = LocalityCacheManager(submodules.cache_config)
        self._affinity_resolver = DeviceAffinityResolver()
        self._router_builder = submodules.router
        self._router: Optional[BaseHeteroRouter] = None
        # Router is lazily initialised on first forward so device is known.
        logger.info(
            "HeteroMoELayer: layer=%d initialised (router lazy)", layer_number
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def router(self) -> Optional[BaseHeteroRouter]:
        return self._router

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full MoE layer forward pass.

        Returns:
            output:      [same shape as hidden_states] expert outputs (placeholder).
            probs:       [N, topk] routing probabilities.
            routing_map: [N, num_experts] bool routing assignment.

        Note:
            Expert computation (the actual FFN) is left as a placeholder stub
            (``_apply_experts``).  In Neuron_SP this is handled by DeepSpeed's
            MoE expert pipeline; the router output is the contract surface.
        """
        self._ensure_router(hidden_states)
        assert self._router is not None

        probs, routing_map = self._router(hidden_states)
        output = self._apply_experts(hidden_states, probs, routing_map)
        return output, probs, routing_map

    def route_async(self, hidden_states: torch.Tensor) -> RoutingFuture:
        """Async routing for DES overlap. See RoutingFuture for usage."""
        self._ensure_router(hidden_states)
        assert self._router is not None
        return self._router.route_async(hidden_states)

    def cache_stats(self) -> Dict[str, Any]:
        """Expose LOC statistics to monitoring/logging pipelines."""
        return self._cache_manager.stats()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_router(self, hidden_states: torch.Tensor) -> None:
        """Lazy router construction: deferred until we know the device."""
        if self._router is not None:
            return
        profile = self._affinity_resolver.resolve(hidden_states)
        self._router = self._router_builder(
            config=self.config,
            device_profile=profile,
            cache_manager=self._cache_manager,
        )
        self._router.set_layer_number(self.layer_number)
        logger.info(
            "HeteroMoELayer: layer=%d router constructed on device=%s arch=%s",
            self.layer_number, profile.device, profile.arch_name,
        )

    def _apply_experts(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
    ) -> torch.Tensor:
        """Placeholder expert dispatch. Returns hidden_states unchanged.

        In production Neuron_SP this delegates to DeepSpeed's MoE expert
        sharding layer (deepspeed.moe.layer.MoE).  The routing tensors
        produced by HeteroMoELayer.forward are passed directly into
        DeepSpeed's token dispatcher.
        """
        return hidden_states


# ---------------------------------------------------------------------------
# apply_hetero_router: DES-LOC equivalent of Megatron's apply_module(router)
# ---------------------------------------------------------------------------

def apply_hetero_router(
    router: HeteroRouterInterface,
) -> Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Wrap a HeteroRouterInterface for uniform call syntax.

    Mirrors Megatron's ``apply_module(self.router)`` pattern from commit
    7857383, but adds DES-LOC locality cache integration transparently.

    Usage:
        probs, routing_map = apply_hetero_router(layer.router)(hidden_states)

    This matches the Megatron call site pattern so code ported from upstream
    can substitute ``apply_module`` with ``apply_hetero_router`` with no
    other changes.
    """
    def _call(hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return router.forward(hidden_states)
    return _call


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Use CPU DeviceProfile for CI environments without GPUs
    if not torch.cuda.is_available():
        logger.warning("No CUDA device available; smoke test runs on CPU with mock profile.")

        # Mock profile simulating an A6000
        mock_profile = DeviceProfile(
            device_index=0,
            sm_major=8,
            sm_minor=6,
            memory_gb=48.0,
            arch_name="ampere",
            supports_warp_specialised_topk=False,
        )

        cfg = {
            "num_moe_experts": 4,
            "moe_router_topk": 2,
            "hidden_size": 64,
            "moe_use_softmax_routing": True,
            "moe_aux_loss_coeff": 0.01,
        }

        cache_mgr = LocalityCacheManager(LocalityCacheConfig(capacity=32, ttl_seconds=1.0))

        # Build router directly (CPU tensors)
        router = TopKHeteroRouter(
            num_experts=4,
            topk=2,
            hidden_size=64,
            device_profile=mock_profile,
            cache_manager=cache_mgr,
            layer_number=0,
        )
        # Move gate to CPU explicitly
        router.gate = nn.Linear(64, 4, bias=False)

        hidden = torch.randn(8, 64)  # [seq*batch, hidden]
        probs, routing_map = router._run_gating(hidden)

        assert probs.shape == (8, 2), f"probs shape wrong: {probs.shape}"
        assert routing_map.shape == (8, 4), f"routing_map shape wrong: {routing_map.shape}"
        assert routing_map.dtype == torch.bool, "routing_map must be bool"
        assert routing_map.sum(dim=-1).eq(2).all(), "Each token must route to exactly topk=2 experts"

        # LOC fingerprint determinism
        fp1 = cache_mgr.fingerprint(hidden)
        fp2 = cache_mgr.fingerprint(hidden)
        assert fp1 == fp2, "Fingerprint must be deterministic"

        logger.info("All smoke tests passed (CPU mode).")
    else:
        # Real GPU path
        dev_idx = 0
        profile = DeviceProfile.from_device(dev_idx)
        logger.info("Running smoke test on %s (SM%d%d)", profile.arch_name,
                    profile.sm_major, profile.sm_minor)

        cfg = {"num_moe_experts": 8, "moe_router_topk": 2, "hidden_size": 128}
        layer = HeteroMoELayer(config=cfg, layer_number=3)

        hidden = torch.randn(16, 128, device=profile.device)
        output, probs, routing_map = layer.forward(hidden)

        assert output.shape == hidden.shape
        assert probs.shape[0] == 16
        assert routing_map.shape == (16, 8)

        # Second forward should hit LOC cache
        _, probs2, _ = layer.forward(hidden)
        stats = layer.cache_stats()
        assert stats["total_hits"] >= 1, "LOC should have at least one hit on second forward"

        logger.info("Smoke tests passed. LOC stats: %s", stats)
