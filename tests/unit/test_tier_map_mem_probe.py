# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Unit tests for issue #593: replace _VRAM_RESERVE_FRACTION with runtime mem_probe.

Covers the full AST call chain introduced by the fix:

  engine_integration.install()
    -> TierMap.discover()
      -> _detect_local_tier()
        -> torch.cuda.mem_get_info(local_rank)   # runtime probe
      -> _gather_tier_infos()                      # all_gather_object
    -> TierMap.mem_budget(rank)
      -> free_vram_bytes * (1 - 0.15)             # runtime path
      -> OR total_vram_bytes * (1 - reserve)       # legacy fallback

Test categories:
  1. mem_probe.py — probe_free_vram, budget_from_probe, validate_tier_info_vram
  2. TierInfo dataclass — free_vram_bytes field present and serialisable
  3. TierMap.mem_budget — runtime path vs legacy fallback
  4. TierMap.from_infos — backward compat with free_vram_bytes
  5. Shard planner integration — VRAM weights use runtime budgets
  6. Edge cases — zero free, free > total, negative free

Running:
    pytest tests/unit/test_tier_map_mem_probe.py -v

No GPU required.  All probes are mocked.
"""
from __future__ import annotations

import pickle
import pytest
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from deepspeed.core.hetero_bridge.tier_map import (
    GPUTier,
    TierInfo,
    TierMap,
    _BYTES_PER_GB,
    _VRAM_RESERVE_FRACTION,
    _VRAM_SAFETY_MARGIN,
)
from deepspeed.core.hetero_bridge.mem_probe import (
    VRAM_SAFETY_MARGIN,
    budget_from_probe,
    probe_free_vram,
    validate_tier_info_vram,
)


# ---------------------------------------------------------------------------
# Helpers — build TierInfo / TierMap objects for tests
# ---------------------------------------------------------------------------

def _ti(rank, tier, total_gb, free_gb, numa=0, tflops=100.0):
    """Shorthand TierInfo constructor (GB -> bytes)."""
    return TierInfo(
        rank=rank,
        tier=tier,
        total_vram_bytes=int(total_gb * _BYTES_PER_GB),
        free_vram_bytes=int(free_gb * _BYTES_PER_GB),
        numa_node=numa,
        peak_bf16_tflops=tflops,
    )


def _ags1_tier_map():
    """5-GPU ags1 topology with realistic free VRAM after model load.

    The free values simulate discover() running *after* the 7B model has
    been placed on device (Phase 4) but *before* optimizer init (Phase 5).
    This is the timing guaranteed by engine_integration.install().
    """
    return TierMap.from_infos([
        _ti(0, GPUTier.A6000,    48,  5.2,  numa=0, tflops=309.7),
        _ti(1, GPUTier.A6000,    48,  4.8,  numa=0, tflops=309.7),
        _ti(2, GPUTier.H100,     94, 62.0,  numa=1, tflops=989.0),
        _ti(3, GPUTier.BLACKWELL, 96, 74.0, numa=1, tflops=2250.0),
        _ti(4, GPUTier.BLACKWELL, 96, 73.5, numa=1, tflops=2250.0),
    ])


def _legacy_tier_map():
    """TierMap with free_vram_bytes=0 (legacy / offline path)."""
    return TierMap.from_infos([
        _ti(0, GPUTier.A6000,    48, 0),
        _ti(1, GPUTier.A6000,    48, 0),
        _ti(2, GPUTier.H100,     94, 0),
    ])


# ===================================================================
# 1. mem_probe.py unit tests
# ===================================================================

class TestProbeFreevram:
    """Tests for probe_free_vram()."""

    def test_returns_tuple_of_two_ints_on_mock_cuda(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.mem_get_info.return_value = (32 * _BYTES_PER_GB, 48 * _BYTES_PER_GB)
        with patch.dict("sys.modules", {"torch": mock_torch}):
            # Re-import to pick up mock
            import importlib
            import deepspeed.core.hetero_bridge.mem_probe as mp
            importlib.reload(mp)
            free, total = mp.probe_free_vram(0)
        assert free == 32 * _BYTES_PER_GB
        assert total == 48 * _BYTES_PER_GB

    def test_no_cuda_returns_zero_zero(self):
        """When CUDA is unavailable, probe should return (0, 0)."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            import importlib
            import deepspeed.core.hetero_bridge.mem_probe as mp
            importlib.reload(mp)
            free, total = mp.probe_free_vram(0)
        assert free == 0
        assert total == 0

    def test_exception_returns_zero_zero(self):
        """If mem_get_info throws, probe returns (0, 0) gracefully."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.mem_get_info.side_effect = RuntimeError("driver error")
        with patch.dict("sys.modules", {"torch": mock_torch}):
            import importlib
            import deepspeed.core.hetero_bridge.mem_probe as mp
            importlib.reload(mp)
            free, total = mp.probe_free_vram(0)
        assert free == 0
        assert total == 0


class TestBudgetFromProbe:
    """Tests for budget_from_probe()."""

    def test_runtime_path_uses_safety_margin(self):
        free = 30 * _BYTES_PER_GB
        total = 48 * _BYTES_PER_GB
        expected = int(free * (1.0 - VRAM_SAFETY_MARGIN))
        assert budget_from_probe(free, total, tier_reserve=0.35) == expected

    def test_legacy_fallback_when_free_is_zero(self):
        total = 48 * _BYTES_PER_GB
        expected = int(total * (1.0 - 0.35))
        assert budget_from_probe(0, total, tier_reserve=0.35) == expected

    def test_zero_total_zero_free_returns_zero(self):
        assert budget_from_probe(0, 0, tier_reserve=0.40) == 0

    def test_budget_never_negative(self):
        """Even with extreme safety margin, budget must be >= 0."""
        assert budget_from_probe(1, 100, tier_reserve=0.99) >= 0


class TestValidateTierInfoVram:
    """Tests for validate_tier_info_vram()."""

    def test_valid_values_pass_through(self):
        assert validate_tier_info_vram(30 * _BYTES_PER_GB, 48 * _BYTES_PER_GB, rank=0) == 30 * _BYTES_PER_GB

    def test_negative_free_clamped_to_zero(self):
        assert validate_tier_info_vram(-100, 48 * _BYTES_PER_GB, rank=0) == 0

    def test_free_greater_than_total_clamped(self):
        total = 48 * _BYTES_PER_GB
        assert validate_tier_info_vram(total + 1000, total, rank=0) == total

    def test_zero_total_zero_free_ok(self):
        assert validate_tier_info_vram(0, 0, rank=0) == 0


# ===================================================================
# 2. TierInfo dataclass tests
# ===================================================================

class TestTierInfoField:
    """Verify free_vram_bytes field exists and is correctly typed."""

    def test_field_exists(self):
        ti = _ti(0, GPUTier.A6000, 48, 30)
        assert hasattr(ti, "free_vram_bytes")
        assert ti.free_vram_bytes == 30 * _BYTES_PER_GB

    def test_pickle_roundtrip(self):
        """TierInfo must be picklable for all_gather_object."""
        ti = _ti(0, GPUTier.A6000, 48, 30)
        restored = pickle.loads(pickle.dumps(ti))
        assert restored.free_vram_bytes == ti.free_vram_bytes
        assert restored.total_vram_bytes == ti.total_vram_bytes
        assert restored.tier == ti.tier
        assert restored.rank == ti.rank

    def test_zero_free_vram_is_valid(self):
        """free_vram_bytes=0 is the offline/test sentinel, not an error."""
        ti = _ti(0, GPUTier.UNKNOWN, 48, 0)
        assert ti.free_vram_bytes == 0


# ===================================================================
# 3. TierMap.mem_budget — runtime vs legacy
# ===================================================================

class TestMemBudgetRuntimePath:
    """mem_budget uses free_vram_bytes * (1 - 0.15) when probe data exists."""

    def test_a6000_runtime_budget(self):
        tm = _ags1_tier_map()
        info = tm.info(0)
        expected = int(info.free_vram_bytes * (1.0 - _VRAM_SAFETY_MARGIN))
        assert tm.mem_budget(0) == expected

    def test_h100_runtime_budget(self):
        tm = _ags1_tier_map()
        info = tm.info(2)
        expected = int(info.free_vram_bytes * (1.0 - _VRAM_SAFETY_MARGIN))
        assert tm.mem_budget(2) == expected

    def test_blackwell_runtime_budget(self):
        tm = _ags1_tier_map()
        info = tm.info(3)
        expected = int(info.free_vram_bytes * (1.0 - _VRAM_SAFETY_MARGIN))
        assert tm.mem_budget(3) == expected

    def test_budget_ordering_matches_free_vram(self):
        """A6000 (5.2 GB free) < H100 (62 GB free) < Blackwell (74 GB free)."""
        tm = _ags1_tier_map()
        assert tm.mem_budget(0) < tm.mem_budget(2) < tm.mem_budget(3)

    def test_runtime_budget_smaller_than_legacy_for_a6000(self):
        """With only 5.2 GB free, runtime budget << legacy 31.2 GB budget.

        This is the core fix: the old code would report 31.2 GB free on an
        A6000 that actually has 5.2 GB free, causing OOM.
        """
        tm = _ags1_tier_map()
        runtime_budget = tm.mem_budget(0)
        legacy_budget = int(48 * _BYTES_PER_GB * (1.0 - 0.35))
        assert runtime_budget < legacy_budget
        # Runtime should be roughly 5.2 * 0.85 = 4.42 GB
        assert runtime_budget < 5 * _BYTES_PER_GB


class TestMemBudgetLegacyFallback:
    """mem_budget falls back to total * (1 - reserve) when free_vram_bytes == 0."""

    def test_a6000_legacy_budget(self):
        tm = _legacy_tier_map()
        expected = int(48 * _BYTES_PER_GB * (1.0 - _VRAM_RESERVE_FRACTION[GPUTier.A6000]))
        assert tm.mem_budget(0) == expected

    def test_h100_legacy_budget(self):
        tm = _legacy_tier_map()
        expected = int(94 * _BYTES_PER_GB * (1.0 - _VRAM_RESERVE_FRACTION[GPUTier.H100]))
        assert tm.mem_budget(2) == expected

    def test_legacy_ordering(self):
        """A6000 < H100 even in legacy mode."""
        tm = _legacy_tier_map()
        assert tm.mem_budget(0) < tm.mem_budget(2)


class TestMemBudgetMixed:
    """A TierMap can have some ranks probed and others not (e.g. gap-fill)."""

    def test_mixed_probed_and_unprobed(self):
        tm = TierMap.from_infos([
            _ti(0, GPUTier.A6000,    48, 30),   # probed
            _ti(1, GPUTier.A6000,    48,  0),   # unprobed
            _ti(2, GPUTier.H100,     94, 72),   # probed
        ])
        # rank 0: runtime path
        assert tm.mem_budget(0) == int(30 * _BYTES_PER_GB * (1.0 - _VRAM_SAFETY_MARGIN))
        # rank 1: legacy fallback
        assert tm.mem_budget(1) == int(48 * _BYTES_PER_GB * (1.0 - 0.35))
        # rank 2: runtime path
        assert tm.mem_budget(2) == int(72 * _BYTES_PER_GB * (1.0 - _VRAM_SAFETY_MARGIN))


# ===================================================================
# 4. TierMap.from_infos — backward compat
# ===================================================================

class TestFromInfos:
    """from_infos must work with the new TierInfo that includes free_vram_bytes."""

    def test_world_size_matches(self):
        tm = _ags1_tier_map()
        assert tm.world_size == 5

    def test_info_roundtrip(self):
        tm = _ags1_tier_map()
        for r in range(5):
            info = tm.info(r)
            assert info.rank == r
            assert info.free_vram_bytes >= 0

    def test_repr_includes_free(self):
        tm = TierMap.from_infos([_ti(0, GPUTier.A6000, 48, 30)])
        r = repr(tm)
        assert "free=" in r


# ===================================================================
# 5. Shard planner integration (mem_budget propagation)
# ===================================================================

class TestShardPlannerIntegration:
    """HeteroShardPlanner uses mem_budget which now uses runtime free VRAM."""

    def test_planner_weights_reflect_free_vram(self):
        """Shard weights should be proportional to free VRAM, not total."""
        import torch
        from deepspeed.core.hetero_bridge.shard_planner import HeteroShardPlanner

        tm = TierMap.from_infos([
            _ti(0, GPUTier.A6000,    48,  5),    # very little free
            _ti(1, GPUTier.H100,     94, 70),    # lots free
        ])
        planner = HeteroShardPlanner(tm)

        # Fake 1000-element parameter
        params = [("w", torch.zeros(1000))]
        plan = planner.plan(params)

        # rank 1 (H100, 70 GB free) should own many more elements than
        # rank 0 (A6000, 5 GB free)
        r0_params = len(plan.rank_to_param_ids.get(0, []))
        r1_params = len(plan.rank_to_param_ids.get(1, []))
        # Both ranks get the single param (it straddles the boundary)
        # but rank 1 should own more bytes
        assert plan.rank_to_bytes[1] > plan.rank_to_bytes[0]

    def test_planner_with_legacy_tiermap_still_works(self):
        """Planner must not crash when free_vram_bytes == 0 (legacy path)."""
        import torch
        from deepspeed.core.hetero_bridge.shard_planner import HeteroShardPlanner

        tm = _legacy_tier_map()
        planner = HeteroShardPlanner(tm)
        params = [("w", torch.zeros(500))]
        plan = planner.plan(params)
        assert sum(plan.rank_to_bytes.values()) > 0


# ===================================================================
# 6. Edge cases
# ===================================================================

class TestEdgeCases:

    def test_free_vram_equals_total(self):
        """Fresh GPU with nothing loaded — free == total."""
        tm = TierMap.from_infos([
            _ti(0, GPUTier.H100, 94, 94),
        ])
        expected = int(94 * _BYTES_PER_GB * (1.0 - _VRAM_SAFETY_MARGIN))
        assert tm.mem_budget(0) == expected

    def test_free_vram_one_byte(self):
        """Extremely low free — budget should still be non-negative."""
        ti = TierInfo(
            rank=0, tier=GPUTier.A6000,
            total_vram_bytes=48 * _BYTES_PER_GB,
            free_vram_bytes=1,
            numa_node=0, peak_bf16_tflops=309.7,
        )
        tm = TierMap.from_infos([ti])
        assert tm.mem_budget(0) >= 0

    def test_unknown_tier_legacy_fallback(self):
        """UNKNOWN tier with free=0 uses the 0.40 reserve fraction."""
        ti = _ti(0, GPUTier.UNKNOWN, 48, 0)
        tm = TierMap.from_infos([ti])
        expected = int(48 * _BYTES_PER_GB * (1.0 - 0.40))
        assert tm.mem_budget(0) == expected

    def test_safety_margin_constant_matches(self):
        """Verify _VRAM_SAFETY_MARGIN in tier_map.py matches mem_probe.py."""
        assert _VRAM_SAFETY_MARGIN == VRAM_SAFETY_MARGIN == 0.15

    def test_all_ranks_different_free(self):
        """Budget ordering must strictly follow free_vram ordering."""
        tm = TierMap.from_infos([
            _ti(0, GPUTier.A6000,    48, 10),
            _ti(1, GPUTier.A6000,    48, 20),
            _ti(2, GPUTier.H100,     94, 30),
            _ti(3, GPUTier.BLACKWELL, 96, 40),
        ])
        budgets = [tm.mem_budget(r) for r in range(4)]
        assert budgets == sorted(budgets), \
            f"Budgets not monotonically increasing: {budgets}"

    def test_13b_cpuadam_scenario(self):
        """Issue #593 scenario: 13B + CPUAdam overestimates A6000 budget.

        Old code: 48 * 0.65 = 31.2 GB (claims 31 GB free for optimizer)
        New code: 3.1 GB free measured -> 3.1 * 0.85 = 2.6 GB
        The shard planner then correctly assigns fewer params to A6000.
        """
        tm = TierMap.from_infos([
            _ti(0, GPUTier.A6000,    48,  3.1),   # 13B model ate most VRAM
            _ti(1, GPUTier.H100,     94, 55.0),   # plenty left on H100
        ])
        a6000_budget_gb = tm.mem_budget(0) / _BYTES_PER_GB
        h100_budget_gb = tm.mem_budget(1) / _BYTES_PER_GB

        # A6000 budget must be < 4 GB (not the legacy ~31 GB)
        assert a6000_budget_gb < 4.0, f"A6000 budget too high: {a6000_budget_gb:.1f} GB"
        # H100 budget must be >> A6000
        assert h100_budget_gb > 10 * a6000_budget_gb

    def test_new_gpu_tier_needs_no_manual_constant(self):
        """A hypothetical B100 GPU (SM 13.0) works without adding to
        _VRAM_RESERVE_FRACTION because runtime probe provides free VRAM.
        """
        ti = TierInfo(
            rank=0, tier=GPUTier.UNKNOWN,  # unknown SM maps to UNKNOWN
            total_vram_bytes=192 * _BYTES_PER_GB,
            free_vram_bytes=180 * _BYTES_PER_GB,  # probe succeeds
            numa_node=0, peak_bf16_tflops=5000.0,
        )
        tm = TierMap.from_infos([ti])
        # Runtime path — does NOT need an entry in _VRAM_RESERVE_FRACTION
        expected = int(180 * _BYTES_PER_GB * (1.0 - _VRAM_SAFETY_MARGIN))
        assert tm.mem_budget(0) == expected
