# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""Heterogeneous-device inference engine for DES-LOC.

Mirrors Megatron 3c39d98b5 ``_MegatronLLMBase`` / ``_CoordinatorRuntime`` /
``_EventLoopManager`` stable API, reinterpreted as a *device-routing layer*:
a single ``HeterogeneousInferenceEngine`` instance manages multiple GPU tiers
(e.g. A6000 48 GB + H100 96 GB) and routes each request to the tier whose
capacity best fits the request's token budget.

Megatron's upstream change separated the stable public API from the engine
internals by introducing ``_MegatronLLMBase`` (direct mode / coordinator mode)
and decoupling ``_CoordinatorRuntime`` from the engine pipeline construction.
The key insight extracted here is that the two execution modes are really a
*dispatch abstraction*: in direct mode every rank acts as primary and runs the
engine inline; in coordinator mode the primary submits requests through a
client and workers run an engine loop on a background thread.

DES-LOC reinterprets that dispatch abstraction at the *device* axis:
- A request whose total token count (prompt + max_new_tokens) fits within
  ``small_request_threshold`` is dispatched to the ``light`` tier (A6000).
- Requests exceeding the threshold go to the ``heavy`` tier (H100).
- The routing decision is extracted into ``_DeviceRouter._select_tier()`` —
  analogous to how M407 extracted ``_free_unlocked`` from the allocate path.
  This lets callers swap routing heuristics without touching the engine loop.

Two operational modes mirror Megatron:
1. **Inline mode** (``use_background_loop=False``): routing + forward run
   synchronously on the caller's thread.  Safe for single-process scripts.
2. **Background-loop mode** (``use_background_loop=True``): an ``_EngineLoop``
   daemon thread owns an asyncio event loop and serialises engine calls via a
   bounded ``asyncio.Queue``; the main thread submits futures and waits.  This
   mirrors ``_EventLoopManager`` and prevents concurrent CUDA calls from
   separate Python threads.

Diagnostic events (rank-0, logger + print, mirrors M451 pattern):
  [DS-HIE] ROUTE: per-request tier assignment with token budget and thresholds.
  [DS-HIE] TIER_SATURATED: when both tiers reject a batch, flagging back-pressure.
  [DS-HIE] STEER_SHIFT: when a tier assignment is overridden due to saturation.
  [DS-HIE] LOOP_READY / LOOP_STOP: background event-loop lifecycle transitions.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from deepspeed.utils import logger as ds_logger

_LOG_PREFIX = "[DS-HIE]"

# ---------------------------------------------------------------------------
# Tier descriptor
# ---------------------------------------------------------------------------

class DeviceTier(str, Enum):
    """Named GPU tier for routing decisions."""
    LIGHT = "light"   # e.g. A6000 — small-batch / short-sequence requests
    HEAVY = "heavy"   # e.g. H100  — large-batch / long-sequence requests


@dataclass
class TierConfig:
    """Configuration for a single device tier.

    Attributes:
        tier:           Tier identifier (LIGHT or HEAVY).
        device_ids:     CUDA device indices assigned to this tier.
        max_tokens:     Maximum total tokens (prompt + generation) this tier
                        will accept per request.  Requests exceeding this limit
                        are re-routed to the next tier.
        capacity_weight: Relative throughput capacity used by the saturated-tier
                         steer-shift logic (mirrors AsymmetricCPScheduler weights
                         from M3047 heterogeneous_cp.py).
    """
    tier: DeviceTier
    device_ids: List[int]
    max_tokens: int = 4096
    capacity_weight: float = 1.0


# ---------------------------------------------------------------------------
# Device router — decoupled from engine loop (mirrors M407 _free_unlocked)
# ---------------------------------------------------------------------------

class _DeviceRouter:
    """Select the appropriate device tier for a request.

    The routing decision is isolated here so it can be unit-tested and swapped
    without touching ``HeterogeneousInferenceEngine``.  Mirrors how M407
    extracted ``_free_unlocked`` from ``DoubleBufferA2A.allocate`` to decouple
    deallocation policy from the allocation fast-path.

    Routing heuristic:
      1. If request_tokens <= small_request_threshold → LIGHT tier preferred.
      2. Otherwise → HEAVY tier preferred.
      3. If the preferred tier is saturated (``saturated_tiers`` set), escalate:
         LIGHT → HEAVY; HEAVY → LIGHT.  Log a STEER_SHIFT event.
      4. If both tiers are saturated, raise ``TierSaturatedError`` and log a
         TIER_SATURATED event.
    """

    def __init__(
        self,
        small_request_threshold: int,
        tier_configs: Dict[DeviceTier, TierConfig],
        *,
        rank: int = 0,
    ) -> None:
        self._threshold = small_request_threshold
        self._tier_configs = tier_configs
        self._rank = rank

    def _select_tier(
        self,
        request_tokens: int,
        saturated_tiers: Optional[set] = None,
    ) -> DeviceTier:
        """Return the tier to route ``request_tokens`` to.

        Args:
            request_tokens:  Total tokens for this request (prompt + max_new).
            saturated_tiers: Set of :class:`DeviceTier` values that are
                             currently at capacity and should be avoided.

        Returns:
            :class:`DeviceTier` for this request.

        Raises:
            TierSaturatedError: if every tier is saturated.
        """
        if saturated_tiers is None:
            saturated_tiers = set()

        preferred = DeviceTier.LIGHT if request_tokens <= self._threshold else DeviceTier.HEAVY
        alternative = DeviceTier.HEAVY if preferred == DeviceTier.LIGHT else DeviceTier.LIGHT

        if preferred not in saturated_tiers:
            if self._rank == 0:
                _log_route(request_tokens, preferred, self._threshold, steered=False)
            return preferred

        if alternative not in saturated_tiers:
            if self._rank == 0:
                _log_steer_shift(request_tokens, preferred, alternative, self._threshold)
            return alternative

        # Both saturated
        if self._rank == 0:
            _log_tier_saturated(request_tokens, self._threshold)
        raise TierSaturatedError(
            f"All device tiers saturated for request with {request_tokens} tokens "
            f"(threshold={self._threshold})"
        )

    def device_ids_for_tier(self, tier: DeviceTier) -> List[int]:
        """Return the CUDA device IDs assigned to ``tier``."""
        cfg = self._tier_configs.get(tier)
        if cfg is None:
            raise KeyError(f"No TierConfig registered for tier {tier}")
        return cfg.device_ids


class TierSaturatedError(RuntimeError):
    """Raised when all device tiers are at capacity."""


# ---------------------------------------------------------------------------
# Diagnostic helpers (M451-style: one structured event at the decision point)
# ---------------------------------------------------------------------------

def _log_route(tokens: int, tier: DeviceTier, threshold: int, steered: bool) -> None:
    msg = (
        f"{_LOG_PREFIX} ROUTE: tokens={tokens} threshold={threshold} "
        f"-> tier={tier.value} steered={steered}"
    )
    print(msg)
    ds_logger.info(msg)


def _log_steer_shift(
    tokens: int,
    original: DeviceTier,
    assigned: DeviceTier,
    threshold: int,
) -> None:
    msg = (
        f"{_LOG_PREFIX} STEER_SHIFT: tokens={tokens} threshold={threshold} "
        f"original_tier={original.value} saturated, rerouting -> {assigned.value}"
    )
    print(msg)
    ds_logger.info(msg)


def _log_tier_saturated(tokens: int, threshold: int) -> None:
    msg = (
        f"{_LOG_PREFIX} TIER_SATURATED: tokens={tokens} threshold={threshold} "
        "all tiers saturated — request will be queued or dropped"
    )
    print(msg)
    ds_logger.warning(msg)


def _log_loop_event(event: str, detail: str = "") -> None:
    msg = f"{_LOG_PREFIX} {event}" + (f": {detail}" if detail else "")
    print(msg)
    ds_logger.info(msg)


# ---------------------------------------------------------------------------
# Background event loop (mirrors Megatron _EventLoopManager)
# ---------------------------------------------------------------------------

class _EngineLoop:
    """Background daemon thread with a persistent asyncio event loop.

    Mirrors Megatron ``_EventLoopManager`` from 3c39d98b5, reinterpreted to
    serialise CUDA kernel dispatch from multiple threads into a single async
    queue on a dedicated daemon thread.  This prevents concurrent CUDA calls
    that would otherwise arise when e.g. an HTTP server dispatches requests
    from a thread pool.

    Key difference from Megatron's manager: we explicitly bind the CUDA
    device on the daemon thread (same as the parent thread's current device)
    so that torchrun's per-rank device assignment is respected — Megatron
    added the same fix in 3c39d98b5's ``_EventLoopManager.__init__``.
    """

    def __init__(self, *, queue_depth: int = 64) -> None:
        self._queue_depth = queue_depth
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._stopped = False
        # Capture parent thread's CUDA device for transfer to daemon thread
        self._parent_device: Optional[int] = (
            torch.cuda.current_device() if torch.cuda.is_available() else None
        )

    def start(self) -> None:
        """Spawn the daemon thread and start the event loop. Idempotent."""
        if self._started:
            return

        loop_ready = threading.Event()
        parent_device = self._parent_device

        def _run() -> None:
            if parent_device is not None and torch.cuda.is_available():
                torch.cuda.set_device(parent_device)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            loop.call_soon(loop_ready.set)
            loop.run_forever()

        self._thread = threading.Thread(target=_run, daemon=True, name="ds-hie-engine-loop")
        self._thread.start()
        loop_ready.wait()
        self._started = True
        _log_loop_event("LOOP_READY", f"device={parent_device} queue_depth={self._queue_depth}")

    def submit(self, coro) -> "asyncio.Future":
        """Schedule ``coro`` on the background loop and return a Future."""
        if not self._started or self._loop is None:
            raise RuntimeError("_EngineLoop.start() must be called before submit()")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def run_sync(self, coro):
        """Submit ``coro`` and block until it completes."""
        return self.submit(coro).result()

    def stop(self) -> None:
        """Stop the background loop and join the daemon thread. Idempotent."""
        if not self._started or self._stopped:
            return
        assert self._loop is not None and self._thread is not None
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10)
        self._stopped = True
        self._started = False
        _log_loop_event("LOOP_STOP")


# ---------------------------------------------------------------------------
# Tier execution handle
# ---------------------------------------------------------------------------

@dataclass
class _TierHandle:
    """Wraps a DeepSpeed InferenceEngine instance with tier metadata.

    In production, ``engine`` is an ``InferenceEngine`` from
    ``deepspeed/inference/engine.py``.  We type it as ``Any`` to avoid a
    circular import and to keep this module importable without the full
    DeepSpeed stack (unit-testable with a mock engine).
    """
    tier: DeviceTier
    engine: Any          # deepspeed.inference.engine.InferenceEngine or mock
    config: TierConfig
    # Lightweight in-flight request counter; saturated when >= max_concurrent
    _in_flight: int = field(default=0, repr=False)
    max_concurrent: int = 32

    def is_saturated(self) -> bool:
        return self._in_flight >= self.max_concurrent

    def acquire(self) -> None:
        self._in_flight += 1

    def release(self) -> None:
        self._in_flight = max(0, self._in_flight - 1)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class HeterogeneousInferenceEngine:
    """Single entry-point managing multiple device tiers for DES-LOC inference.

    Mirrors Megatron 3c39d98b5 ``_MegatronLLMBase`` API shape:

    - ``generate(prompts, ...)`` as the primary callable.
    - Context-manager protocol (``__enter__`` / ``__exit__``).
    - ``shutdown()`` / ``wait_for_shutdown()`` lifecycle.
    - ``use_background_loop`` mirrors Megatron's ``use_coordinator`` flag:
      when True a ``_EngineLoop`` is started and all forward passes are
      serialised through it (coordinator = background-loop in DES-LOC terms).

    Reinterpreted addition (DES-LOC-specific):
    - ``_DeviceRouter._select_tier()`` is called per request *before* dispatch,
      factoring in both the token budget and current tier saturation state.
      This is the key algorithmic change that has no Megatron equivalent:
      Megatron's LLM base dispatches to one fixed engine; DES-LOC routes to
      the cheapest tier that can absorb the request.

    Usage::

        light_cfg = TierConfig(DeviceTier.LIGHT, device_ids=[0, 1], max_tokens=2048)
        heavy_cfg = TierConfig(DeviceTier.HEAVY, device_ids=[2, 3], max_tokens=32768,
                               capacity_weight=6.0)

        engine = HeterogeneousInferenceEngine(
            tier_handles={
                DeviceTier.LIGHT: _TierHandle(DeviceTier.LIGHT, light_ds_engine, light_cfg),
                DeviceTier.HEAVY: _TierHandle(DeviceTier.HEAVY, heavy_ds_engine, heavy_cfg),
            },
            small_request_threshold=512,
            use_background_loop=True,
        )

        with engine:
            results = engine.generate(prompts, max_new_tokens=128)
    """

    def __init__(
        self,
        *,
        tier_handles: Dict[DeviceTier, _TierHandle],
        small_request_threshold: int = 512,
        use_background_loop: bool = False,
        rank: int = 0,
        background_queue_depth: int = 64,
    ) -> None:
        """
        Args:
            tier_handles:            Mapping of tier → _TierHandle (engine + metadata).
            small_request_threshold: Requests with total tokens <= this go to the LIGHT
                                     tier; above it → HEAVY.  Tune based on observed A6000
                                     vs H100 latency crossover in your cluster.
            use_background_loop:     When True, a daemon ``_EngineLoop`` is started and
                                     all forward passes are serialised through it.  Mirrors
                                     Megatron ``use_coordinator=True`` mode.
            rank:                    Global distributed rank.  Diagnostic events are
                                     suppressed on rank != 0.
            background_queue_depth:  ``asyncio.Queue`` depth for the background loop.
        """
        if not tier_handles:
            raise ValueError("tier_handles must contain at least one _TierHandle entry")

        self._tier_handles = tier_handles
        self._rank = rank
        self._use_background_loop = use_background_loop
        self._shutdown_called = False

        tier_configs = {tier: h.config for tier, h in tier_handles.items()}
        self._router = _DeviceRouter(
            small_request_threshold=small_request_threshold,
            tier_configs=tier_configs,
            rank=rank,
        )

        self._loop_manager: Optional[_EngineLoop] = None
        if use_background_loop:
            lm = _EngineLoop(queue_depth=background_queue_depth)
            lm.start()
            self._loop_manager = lm

    # ---- context manager protocol (mirrors Megatron LLM __enter__/__exit__) ----

    def __enter__(self) -> "HeterogeneousInferenceEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()

    # ---- public API ----

    def generate(
        self,
        prompts: List[Any],
        max_new_tokens: int = 128,
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Route each prompt to the appropriate tier and run inference.

        Args:
            prompts:        List of prompts (strings or token-id lists).
            max_new_tokens: Maximum tokens to generate per prompt.
            sampling_params: Optional dict forwarded to the underlying engine.

        Returns:
            List of generation results in the same order as ``prompts``.

        Raises:
            TierSaturatedError: if all tiers are simultaneously saturated.
            RuntimeError:       if called after :meth:`shutdown`.
        """
        if self._shutdown_called:
            raise RuntimeError("HeterogeneousInferenceEngine has been shut down")

        if self._use_background_loop and self._loop_manager is not None:
            return self._loop_manager.run_sync(
                self._generate_async(prompts, max_new_tokens, sampling_params)
            )
        return self._generate_sync(prompts, max_new_tokens, sampling_params)

    def shutdown(self) -> None:
        """Shut down the background loop (if any) and release tier resources.

        Idempotent. Mirrors Megatron ``MegatronLLM.shutdown()``.
        """
        if self._shutdown_called:
            return
        self._shutdown_called = True
        if self._loop_manager is not None:
            self._loop_manager.stop()
            self._loop_manager = None

    def wait_for_shutdown(self) -> None:
        """Block until the background loop has fully stopped.

        A no-op in inline mode. Mirrors Megatron ``wait_for_shutdown()``.
        """
        # In our impl shutdown() is already synchronous (joins the thread).
        # This method exists so callers can use the same interface as Megatron.
        pass

    # ---- internal dispatch ----

    def _resolve_tier_for_prompt(
        self,
        prompt: Any,
        max_new_tokens: int,
        saturated: set,
    ) -> Tuple[DeviceTier, _TierHandle]:
        """Measure prompt length, call router, return tier + handle."""
        prompt_len = self._estimate_prompt_tokens(prompt)
        total = prompt_len + max_new_tokens
        tier = self._router._select_tier(total, saturated_tiers=saturated)
        handle = self._tier_handles[tier]
        return tier, handle

    def _generate_sync(
        self,
        prompts: List[Any],
        max_new_tokens: int,
        sampling_params: Optional[Dict[str, Any]],
    ) -> List[Any]:
        """Inline (non-background-loop) dispatch path."""
        results: List[Any] = [None] * len(prompts)

        # Group prompts by target tier first so we can batch per tier.
        tier_batches: Dict[DeviceTier, List[Tuple[int, Any]]] = {
            t: [] for t in self._tier_handles
        }
        saturated: set = set()

        for idx, prompt in enumerate(prompts):
            tier, handle = self._resolve_tier_for_prompt(prompt, max_new_tokens, saturated)
            tier_batches[tier].append((idx, prompt))

        # Execute each tier batch
        for tier, indexed_prompts in tier_batches.items():
            if not indexed_prompts:
                continue
            handle = self._tier_handles[tier]
            batch_prompts = [p for _, p in indexed_prompts]
            batch_indices = [i for i, _ in indexed_prompts]

            handle.acquire()
            try:
                tier_results = self._call_engine(handle, batch_prompts, max_new_tokens, sampling_params)
            finally:
                handle.release()

            for i, result in zip(batch_indices, tier_results):
                results[i] = result

        return results

    async def _generate_async(
        self,
        prompts: List[Any],
        max_new_tokens: int,
        sampling_params: Optional[Dict[str, Any]],
    ) -> List[Any]:
        """Background-loop (async) dispatch path.

        Mirrors Megatron ``_MegatronLLMBase._generate_impl`` which submits
        requests to the InferenceClient from a coroutine.  Here we acquire
        the tier handle, run the blocking engine call in an executor (to avoid
        blocking the event loop), and release.
        """
        loop = asyncio.get_event_loop()
        results: List[Any] = [None] * len(prompts)

        tier_batches: Dict[DeviceTier, List[Tuple[int, Any]]] = {
            t: [] for t in self._tier_handles
        }
        saturated: set = set()

        for idx, prompt in enumerate(prompts):
            tier, handle = self._resolve_tier_for_prompt(prompt, max_new_tokens, saturated)
            tier_batches[tier].append((idx, prompt))

        import concurrent.futures
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(self._tier_handles))

        async def _run_tier(tier: DeviceTier, indexed_prompts: List[Tuple[int, Any]]):
            handle = self._tier_handles[tier]
            batch_prompts = [p for _, p in indexed_prompts]
            batch_indices = [i for i, _ in indexed_prompts]
            handle.acquire()
            try:
                tier_results = await loop.run_in_executor(
                    executor,
                    lambda: self._call_engine(handle, batch_prompts, max_new_tokens, sampling_params),
                )
            finally:
                handle.release()
            for i, result in zip(batch_indices, tier_results):
                results[i] = result

        tasks = [
            _run_tier(tier, indexed_prompts)
            for tier, indexed_prompts in tier_batches.items()
            if indexed_prompts
        ]
        await asyncio.gather(*tasks)
        executor.shutdown(wait=False)
        return results

    def _call_engine(
        self,
        handle: _TierHandle,
        prompts: List[Any],
        max_new_tokens: int,
        sampling_params: Optional[Dict[str, Any]],
    ) -> List[Any]:
        """Dispatch a batch to the underlying DeepSpeed engine for ``handle.tier``.

        The engine is expected to expose the same interface as
        ``InferenceEngine.generate`` (from deepspeed/inference/engine.py) or
        ``InferenceEngineV2.put`` (from deepspeed/inference/v2/engine_v2.py).
        We check for ``generate`` first (HF-style), then fall back to ``put``
        (ragged V2 style).

        Callers that plug in a mock engine for testing can implement either
        interface.
        """
        engine = handle.engine
        if engine is None:
            # Stub for unit-tests / dry-run mode: return empty strings.
            return ["" for _ in prompts]

        if hasattr(engine, "generate"):
            # HF-compatible generate (also DeepSpeed's InferenceEngine._generate path)
            kwargs = sampling_params or {}
            return engine.generate(
                prompts,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )

        if hasattr(engine, "put"):
            # InferenceEngineV2.put expects token tensors; callers should
            # pre-tokenize or pass token IDs.
            raise NotImplementedError(
                "InferenceEngineV2 (ragged) path requires tokenized inputs; "
                "wrap prompts with token IDs and call engine.put() directly."
            )

        raise TypeError(
            f"Unsupported engine type for tier {handle.tier}: {type(engine).__name__}. "
            "Engine must expose .generate() or .put()."
        )

    @staticmethod
    def _estimate_prompt_tokens(prompt: Any) -> int:
        """Heuristic token-count estimate used for routing decisions.

        For pre-tokenized inputs (list of ints) this is exact.
        For strings we use the 4-chars-per-token rule-of-thumb that GPT-family
        tokenizers approximate.  A more accurate estimate requires an actual
        tokenizer; callers can subclass and override this method.
        """
        if isinstance(prompt, (list, tuple)) and all(isinstance(t, int) for t in prompt):
            return len(prompt)
        if isinstance(prompt, str):
            return max(1, len(prompt) // 4)
        # Tensor or other pre-tokenized form
        if hasattr(prompt, "numel"):
            return int(prompt.numel())
        return 256  # safe fallback

    # ---- properties ----

    @property
    def tier_handles(self) -> Dict[DeviceTier, _TierHandle]:
        """Mapping of registered tier handles."""
        return self._tier_handles

    @property
    def router(self) -> _DeviceRouter:
        """The ``_DeviceRouter`` instance (testable in isolation)."""
        return self._router
