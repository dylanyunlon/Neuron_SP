# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""Unit tests for deepspeed/inference/heterogeneous_engine.py

Mirrors Megatron tests/unit_tests/inference/high_level_api/test_apis.py and
test_event_loop_manager.py (from upstream commit 3c39d98b5), reinterpreted
for the DES-LOC device-routing layer.

Tests are intentionally CPU-only and import no CUDA dependencies so they pass
in CI environments without GPUs.  The actual CUDA dispatch is mocked via
stub engines (``_StubEngine``) that record calls and return dummy outputs.
"""

import asyncio
import threading
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from deepspeed.inference.heterogeneous_engine import (
    DeviceTier,
    HeterogeneousInferenceEngine,
    TierConfig,
    TierSaturatedError,
    _DeviceRouter,
    _EngineLoop,
    _TierHandle,
)


# ---------------------------------------------------------------------------
# Stub engine
# ---------------------------------------------------------------------------

class _StubEngine:
    """Mock inference engine that records generate() calls."""

    def __init__(self, tier_label: str) -> None:
        self.tier_label = tier_label
        self.calls: List[Dict[str, Any]] = []

    def generate(self, prompts, max_new_tokens=128, **kwargs) -> List[str]:
        self.calls.append({"prompts": list(prompts), "max_new_tokens": max_new_tokens})
        return [f"{self.tier_label}:{p}" for p in prompts]


def _make_engine(
    light_engine: Optional[Any] = None,
    heavy_engine: Optional[Any] = None,
    threshold: int = 512,
    use_background_loop: bool = False,
) -> HeterogeneousInferenceEngine:
    """Construct a ``HeterogeneousInferenceEngine`` with stub tier engines."""
    light_cfg = TierConfig(
        tier=DeviceTier.LIGHT,
        device_ids=[0],
        max_tokens=2048,
        capacity_weight=1.0,
    )
    heavy_cfg = TierConfig(
        tier=DeviceTier.HEAVY,
        device_ids=[1],
        max_tokens=32768,
        capacity_weight=6.0,
    )
    tier_handles = {
        DeviceTier.LIGHT: _TierHandle(
            tier=DeviceTier.LIGHT,
            engine=light_engine or _StubEngine("light"),
            config=light_cfg,
        ),
        DeviceTier.HEAVY: _TierHandle(
            tier=DeviceTier.HEAVY,
            engine=heavy_engine or _StubEngine("heavy"),
            config=heavy_cfg,
        ),
    }
    return HeterogeneousInferenceEngine(
        tier_handles=tier_handles,
        small_request_threshold=threshold,
        use_background_loop=use_background_loop,
        rank=0,
    )


# ---------------------------------------------------------------------------
# _DeviceRouter tests — routing decision logic is isolated here
# ---------------------------------------------------------------------------

class TestDeviceRouter:
    """Tests for ``_DeviceRouter._select_tier``.

    Mirrors test_event_loop_manager.py in Megatron (tests the isolated helper
    class before integration).
    """

    def _make_router(self, threshold: int = 512) -> _DeviceRouter:
        return _DeviceRouter(
            small_request_threshold=threshold,
            tier_configs={
                DeviceTier.LIGHT: TierConfig(DeviceTier.LIGHT, device_ids=[0]),
                DeviceTier.HEAVY: TierConfig(DeviceTier.HEAVY, device_ids=[1]),
            },
            rank=1,  # non-zero rank → diagnostics suppressed
        )

    def test_small_request_routes_to_light(self):
        router = self._make_router(512)
        assert router._select_tier(100) == DeviceTier.LIGHT
        assert router._select_tier(512) == DeviceTier.LIGHT

    def test_large_request_routes_to_heavy(self):
        router = self._make_router(512)
        assert router._select_tier(513) == DeviceTier.HEAVY
        assert router._select_tier(8192) == DeviceTier.HEAVY

    def test_saturated_light_steers_to_heavy(self):
        router = self._make_router(512)
        tier = router._select_tier(100, saturated_tiers={DeviceTier.LIGHT})
        assert tier == DeviceTier.HEAVY

    def test_saturated_heavy_steers_to_light(self):
        router = self._make_router(512)
        tier = router._select_tier(1000, saturated_tiers={DeviceTier.HEAVY})
        assert tier == DeviceTier.LIGHT

    def test_both_saturated_raises(self):
        router = self._make_router(512)
        with pytest.raises(TierSaturatedError):
            router._select_tier(100, saturated_tiers={DeviceTier.LIGHT, DeviceTier.HEAVY})

    def test_boundary_token_count(self):
        """Exactly at threshold → LIGHT; one above → HEAVY."""
        router = self._make_router(1024)
        assert router._select_tier(1024) == DeviceTier.LIGHT
        assert router._select_tier(1025) == DeviceTier.HEAVY

    def test_device_ids_for_tier(self):
        router = self._make_router()
        assert router.device_ids_for_tier(DeviceTier.LIGHT) == [0]
        assert router.device_ids_for_tier(DeviceTier.HEAVY) == [1]

    def test_unknown_tier_raises(self):
        router = self._make_router()
        with pytest.raises(KeyError):
            router.device_ids_for_tier("nonexistent")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _EngineLoop tests — mirrors test_event_loop_manager.py
# ---------------------------------------------------------------------------

class TestEngineLoop:
    """Tests for the background daemon event loop wrapper."""

    def test_start_stop_idempotent(self):
        el = _EngineLoop()
        el.start()
        assert el._started
        el.start()  # second call is a no-op
        assert el._started
        el.stop()
        assert el._stopped

    def test_run_sync_simple_coroutine(self):
        async def _add(a, b):
            return a + b

        el = _EngineLoop()
        el.start()
        try:
            result = el.run_sync(_add(3, 4))
            assert result == 7
        finally:
            el.stop()

    def test_run_sync_from_main_thread_is_not_deadlock(self):
        """Verify run_sync does not deadlock when called from the main thread."""
        el = _EngineLoop()
        el.start()
        try:
            result = el.run_sync(asyncio.coroutine(lambda: 42)())
        except TypeError:
            # asyncio.coroutine removed in Python 3.11; use async def
            async def _coro():
                return 42
            result = el.run_sync(_coro())
        finally:
            el.stop()
        assert result == 42

    def test_submit_before_start_raises(self):
        el = _EngineLoop()

        async def _noop():
            pass

        with pytest.raises(RuntimeError, match="start()"):
            el.submit(_noop())

    def test_daemon_thread_is_daemon(self):
        el = _EngineLoop()
        el.start()
        assert el._thread is not None
        assert el._thread.daemon
        el.stop()


# ---------------------------------------------------------------------------
# HeterogeneousInferenceEngine integration tests
# ---------------------------------------------------------------------------

class TestHeterogeneousInferenceEngine:

    def test_short_prompts_routed_to_light(self):
        """Short prompts (≤ threshold tokens) go to the LIGHT engine."""
        light = _StubEngine("light")
        heavy = _StubEngine("heavy")
        engine = _make_engine(light_engine=light, heavy_engine=heavy, threshold=512)

        with engine:
            results = engine.generate(["hi"] * 3, max_new_tokens=10)

        # Each short prompt → light tier
        assert len(results) == 3
        for r in results:
            assert r.startswith("light:")
        # Heavy engine untouched
        assert heavy.calls == []

    def test_long_prompts_routed_to_heavy(self):
        """Prompts whose token estimate exceeds threshold go to HEAVY."""
        light = _StubEngine("light")
        heavy = _StubEngine("heavy")
        # Threshold=50, prompt of 300 chars ≈ 75 tokens → heavy
        engine = _make_engine(light_engine=light, heavy_engine=heavy, threshold=50)

        long_prompt = "x" * 300  # ~75 tokens by 4-chars-per-token heuristic
        with engine:
            results = engine.generate([long_prompt], max_new_tokens=10)

        assert results[0].startswith("heavy:")
        assert light.calls == []

    def test_mixed_batch_split_across_tiers(self):
        """Mix of short and long prompts → split across LIGHT and HEAVY."""
        light = _StubEngine("light")
        heavy = _StubEngine("heavy")
        engine = _make_engine(light_engine=light, heavy_engine=heavy, threshold=50)

        short_prompt = "hello"        # ~1 token → LIGHT
        long_prompt = "x" * 300      # ~75 tokens → HEAVY

        with engine:
            results = engine.generate([short_prompt, long_prompt], max_new_tokens=10)

        assert len(results) == 2
        tiers = {r.split(":")[0] for r in results}
        assert tiers == {"light", "heavy"}

    def test_context_manager_shuts_down_cleanly(self):
        engine = _make_engine()
        with engine:
            pass
        assert engine._shutdown_called

    def test_shutdown_idempotent(self):
        engine = _make_engine()
        engine.shutdown()
        engine.shutdown()  # second call must not raise

    def test_generate_after_shutdown_raises(self):
        engine = _make_engine()
        engine.shutdown()
        with pytest.raises(RuntimeError, match="shut down"):
            engine.generate(["prompt"])

    def test_background_loop_mode(self):
        """Background-loop mode produces the same results as inline mode."""
        light_inline = _StubEngine("light")
        heavy_inline = _StubEngine("heavy")
        inline_engine = _make_engine(
            light_engine=light_inline, heavy_engine=heavy_inline, use_background_loop=False
        )

        light_bg = _StubEngine("light")
        heavy_bg = _StubEngine("heavy")
        bg_engine = _make_engine(
            light_engine=light_bg, heavy_engine=heavy_bg, use_background_loop=True
        )

        prompts = ["hi", "hello world"]
        with inline_engine:
            inline_results = inline_engine.generate(prompts, max_new_tokens=5)
        with bg_engine:
            bg_results = bg_engine.generate(prompts, max_new_tokens=5)

        assert inline_results == bg_results

    def test_no_tier_handles_raises(self):
        with pytest.raises(ValueError, match="tier_handles"):
            HeterogeneousInferenceEngine(tier_handles={})

    def test_token_estimate_from_list_of_ints(self):
        """Pre-tokenized inputs are counted exactly."""
        engine = _make_engine(threshold=10)
        tokens_short = list(range(5))   # 5 tokens → LIGHT
        tokens_long = list(range(100))  # 100 tokens → HEAVY

        tier_short, _ = engine._resolve_tier_for_prompt(tokens_short, max_new_tokens=0, saturated=set())
        tier_long, _ = engine._resolve_tier_for_prompt(tokens_long, max_new_tokens=0, saturated=set())

        assert tier_short == DeviceTier.LIGHT
        assert tier_long == DeviceTier.HEAVY

    def test_max_new_tokens_counts_toward_routing(self):
        """max_new_tokens is added to the prompt token estimate for routing."""
        engine = _make_engine(threshold=100)
        short_prompt = "hi"  # ~0–1 tokens

        # With max_new_tokens=200 → total > 100 → HEAVY
        tier, _ = engine._resolve_tier_for_prompt(short_prompt, max_new_tokens=200, saturated=set())
        assert tier == DeviceTier.HEAVY

    def test_wait_for_shutdown_is_noop(self):
        """wait_for_shutdown() should not raise in any mode."""
        engine = _make_engine(use_background_loop=False)
        engine.wait_for_shutdown()  # no-op inline

        engine2 = _make_engine(use_background_loop=True)
        engine2.shutdown()
        engine2.wait_for_shutdown()  # no-op after shutdown


# ---------------------------------------------------------------------------
# TierHandle saturation tests
# ---------------------------------------------------------------------------

class TestTierHandle:

    def test_acquire_release_in_flight(self):
        cfg = TierConfig(DeviceTier.LIGHT, device_ids=[0])
        handle = _TierHandle(DeviceTier.LIGHT, engine=None, config=cfg, max_concurrent=2)
        assert not handle.is_saturated()
        handle.acquire()
        assert not handle.is_saturated()
        handle.acquire()
        assert handle.is_saturated()
        handle.release()
        assert not handle.is_saturated()

    def test_release_never_goes_below_zero(self):
        cfg = TierConfig(DeviceTier.LIGHT, device_ids=[0])
        handle = _TierHandle(DeviceTier.LIGHT, engine=None, config=cfg, max_concurrent=2)
        handle.release()  # should not raise or go negative
        assert handle._in_flight == 0
