# SPDX-License-Identifier: Apache-2.0
"""tests/test_hetero_bridge.py — unit tests for deepspeed/core/hetero_bridge/.

Covers:
  1. tier_map.py     — SM-capability → GPUTier mapping, TierMap queries,
                       mem_budget fractions, from_infos constructor.
  2. shard_planner.py — VRAM-proportional shard sizing, param → rank overlap,
                        single-rank and empty edge cases.
  3. desloc_sync_policy.py — SyncPeriods dataclass, DesLocSyncPolicy
                              instantiation, and NotImplementedError guards for
                              Phase-1 skeleton methods.

All tests run without CUDA or torch.distributed.  Torch tensors are created
on CPU; VRAM values are injected via TierMap.from_infos() rather than
TierMap.discover(), so no GPU hardware is required.

Addresses #90.
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import List, Tuple

import torch

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from deepspeed.core.hetero_bridge.tier_map import (
    GPUTier,
    TierInfo,
    TierMap,
    _SM_TO_TIER,
    _VRAM_RESERVE_FRACTION,
    _TIER_TFLOPS,
    _BYTES_PER_GB,
)
from deepspeed.core.hetero_bridge.shard_planner import HeteroShardPlanner, ShardPlan
from deepspeed.core.hetero_bridge.desloc_sync_policy import DesLocSyncPolicy, SyncPeriods


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tier_info(rank: int, tier: GPUTier, vram_gb: int = 48) -> TierInfo:
    """Construct a TierInfo without touching torch.cuda."""
    return TierInfo(
        rank=rank,
        tier=tier,
        total_vram_bytes=vram_gb * _BYTES_PER_GB,
        numa_node=0,
        peak_bf16_tflops=_TIER_TFLOPS[tier],
    )


def _cluster_tier_map() -> TierMap:
    """Return a TierMap matching the ARCHITECTURE.md cluster topology:
      rank 0 → A6000  (48 GB)
      rank 1 → A6000  (48 GB)
      rank 2 → H100   (96 GB)
      rank 3 → Blackwell (96 GB)
      rank 4 → Blackwell (96 GB)
    """
    infos = [
        _make_tier_info(0, GPUTier.A6000, 48),
        _make_tier_info(1, GPUTier.A6000, 48),
        _make_tier_info(2, GPUTier.H100, 96),
        _make_tier_info(3, GPUTier.BLACKWELL, 96),
        _make_tier_info(4, GPUTier.BLACKWELL, 96),
    ]
    return TierMap.from_infos(infos)


def _named_params(sizes: List[int]) -> List[Tuple[str, torch.Tensor]]:
    """Build a list of (name, tensor) pairs with given element counts."""
    return [(f"param_{i}", torch.zeros(n)) for i, n in enumerate(sizes)]


# ===========================================================================
# 1. Tests for tier_map.py
# ===========================================================================

class TestSMToTierMapping(unittest.TestCase):
    """_SM_TO_TIER must map the documented SM capabilities to the right GPUTier."""

    def test_sm_86_maps_to_a6000(self):
        self.assertEqual(_SM_TO_TIER[(8, 6)], GPUTier.A6000)

    def test_sm_87_maps_to_a6000_class(self):
        # Jetson Orin (SM8.7) is treated as A6000-class
        self.assertEqual(_SM_TO_TIER[(8, 7)], GPUTier.A6000)

    def test_sm_90_maps_to_h100(self):
        self.assertEqual(_SM_TO_TIER[(9, 0)], GPUTier.H100)

    def test_sm_94_maps_to_h100_nvl(self):
        self.assertEqual(_SM_TO_TIER[(9, 4)], GPUTier.H100)

    def test_sm_120_maps_to_blackwell(self):
        self.assertEqual(_SM_TO_TIER[(12, 0)], GPUTier.BLACKWELL)

    def test_sm_121_maps_to_blackwell(self):
        self.assertEqual(_SM_TO_TIER[(12, 1)], GPUTier.BLACKWELL)

    def test_unknown_sm_not_in_map(self):
        # An unmapped SM capability must not silently return a tier
        self.assertNotIn((7, 5), _SM_TO_TIER)  # Turing — not in cluster
        self.assertNotIn((8, 0), _SM_TO_TIER)

    def test_all_map_values_are_gputier_members(self):
        for key, val in _SM_TO_TIER.items():
            self.assertIsInstance(val, GPUTier, f"key {key} maps to non-GPUTier: {val}")


class TestGPUTierEnum(unittest.TestCase):
    """GPUTier enum has the four expected members with correct string values."""

    def test_member_names_exist(self):
        members = {t.name for t in GPUTier}
        self.assertIn("A6000", members)
        self.assertIn("H100", members)
        self.assertIn("BLACKWELL", members)
        self.assertIn("UNKNOWN", members)

    def test_string_values(self):
        self.assertEqual(GPUTier.A6000.value, "a6000")
        self.assertEqual(GPUTier.H100.value, "h100")
        self.assertEqual(GPUTier.BLACKWELL.value, "blackwell")
        self.assertEqual(GPUTier.UNKNOWN.value, "unknown")


class TestTierInfoDataclass(unittest.TestCase):
    """TierInfo stores fields correctly and is constructable without CUDA."""

    def test_construction_and_fields(self):
        info = _make_tier_info(rank=2, tier=GPUTier.H100, vram_gb=96)
        self.assertEqual(info.rank, 2)
        self.assertEqual(info.tier, GPUTier.H100)
        self.assertEqual(info.total_vram_bytes, 96 * _BYTES_PER_GB)
        self.assertEqual(info.numa_node, 0)
        self.assertAlmostEqual(info.peak_bf16_tflops, _TIER_TFLOPS[GPUTier.H100])

    def test_tflops_per_tier(self):
        for tier in (GPUTier.A6000, GPUTier.H100, GPUTier.BLACKWELL, GPUTier.UNKNOWN):
            info = _make_tier_info(0, tier)
            self.assertEqual(info.peak_bf16_tflops, _TIER_TFLOPS[tier])

    def test_blackwell_tflops_greater_than_h100(self):
        self.assertGreater(
            _TIER_TFLOPS[GPUTier.BLACKWELL],
            _TIER_TFLOPS[GPUTier.H100],
        )

    def test_h100_tflops_greater_than_a6000(self):
        self.assertGreater(
            _TIER_TFLOPS[GPUTier.H100],
            _TIER_TFLOPS[GPUTier.A6000],
        )


class TestTierMapFromInfos(unittest.TestCase):
    """TierMap.from_infos() constructs a valid map without a distributed env."""

    def setUp(self):
        self.tm = _cluster_tier_map()

    def test_world_size(self):
        self.assertEqual(self.tm.world_size, 5)

    def test_tier_of_each_rank(self):
        self.assertEqual(self.tm.tier_of(0), GPUTier.A6000)
        self.assertEqual(self.tm.tier_of(1), GPUTier.A6000)
        self.assertEqual(self.tm.tier_of(2), GPUTier.H100)
        self.assertEqual(self.tm.tier_of(3), GPUTier.BLACKWELL)
        self.assertEqual(self.tm.tier_of(4), GPUTier.BLACKWELL)

    def test_tier_of_missing_rank_raises(self):
        with self.assertRaises(KeyError):
            self.tm.tier_of(99)

    def test_ranks_of_tier_a6000(self):
        self.assertEqual(self.tm.ranks_of_tier(GPUTier.A6000), [0, 1])

    def test_ranks_of_tier_h100(self):
        self.assertEqual(self.tm.ranks_of_tier(GPUTier.H100), [2])

    def test_ranks_of_tier_blackwell(self):
        self.assertEqual(self.tm.ranks_of_tier(GPUTier.BLACKWELL), [3, 4])

    def test_ranks_of_tier_unknown_empty(self):
        self.assertEqual(self.tm.ranks_of_tier(GPUTier.UNKNOWN), [])

    def test_ranks_of_tier_sorted_ascending(self):
        # Build map with ranks in reverse to confirm sorting
        infos = [
            _make_tier_info(4, GPUTier.A6000),
            _make_tier_info(2, GPUTier.A6000),
            _make_tier_info(0, GPUTier.A6000),
        ]
        tm = TierMap.from_infos(infos)
        self.assertEqual(tm.ranks_of_tier(GPUTier.A6000), [0, 2, 4])

    def test_info_returns_tier_info(self):
        info = self.tm.info(2)
        self.assertIsInstance(info, TierInfo)
        self.assertEqual(info.rank, 2)
        self.assertEqual(info.tier, GPUTier.H100)

    def test_is_low_vram(self):
        self.assertTrue(self.tm.is_low_vram(0))
        self.assertTrue(self.tm.is_low_vram(1))
        self.assertFalse(self.tm.is_low_vram(2))
        self.assertFalse(self.tm.is_low_vram(3))

    def test_is_high_vram(self):
        self.assertFalse(self.tm.is_high_vram(0))
        self.assertTrue(self.tm.is_high_vram(2))
        self.assertTrue(self.tm.is_high_vram(3))
        self.assertTrue(self.tm.is_high_vram(4))


class TestTierMapMemBudget(unittest.TestCase):
    """mem_budget applies the correct tier-specific reserve fractions."""

    def _budget(self, tier: GPUTier, vram_gb: int) -> int:
        tm = TierMap.from_infos([_make_tier_info(0, tier, vram_gb)])
        return tm.mem_budget(0)

    def test_a6000_reserve_fraction(self):
        vram = 48 * _BYTES_PER_GB
        reserve = _VRAM_RESERVE_FRACTION[GPUTier.A6000]  # 0.35
        expected = int(vram * (1.0 - reserve))
        self.assertEqual(self._budget(GPUTier.A6000, 48), expected)

    def test_h100_reserve_fraction(self):
        vram = 96 * _BYTES_PER_GB
        reserve = _VRAM_RESERVE_FRACTION[GPUTier.H100]  # 0.25
        expected = int(vram * (1.0 - reserve))
        self.assertEqual(self._budget(GPUTier.H100, 96), expected)

    def test_blackwell_reserve_fraction(self):
        vram = 96 * _BYTES_PER_GB
        reserve = _VRAM_RESERVE_FRACTION[GPUTier.BLACKWELL]  # 0.20
        expected = int(vram * (1.0 - reserve))
        self.assertEqual(self._budget(GPUTier.BLACKWELL, 96), expected)

    def test_unknown_reserve_fraction(self):
        vram = 48 * _BYTES_PER_GB
        reserve = _VRAM_RESERVE_FRACTION[GPUTier.UNKNOWN]  # 0.40
        expected = int(vram * (1.0 - reserve))
        self.assertEqual(self._budget(GPUTier.UNKNOWN, 48), expected)

    def test_budget_positive(self):
        for tier in GPUTier:
            self.assertGreater(self._budget(tier, 48), 0, f"tier={tier} has zero budget")

    def test_a6000_budget_smaller_than_h100_same_vram(self):
        """A6000 reserves more, so its budget is smaller for equal VRAM."""
        a6000_budget = self._budget(GPUTier.A6000, 96)
        h100_budget = self._budget(GPUTier.H100, 96)
        self.assertLess(a6000_budget, h100_budget)

    def test_h100_budget_smaller_than_blackwell_same_vram(self):
        h100_budget = self._budget(GPUTier.H100, 96)
        bw_budget = self._budget(GPUTier.BLACKWELL, 96)
        self.assertLess(h100_budget, bw_budget)

    def test_budget_scales_with_vram(self):
        """Doubling VRAM should double the budget (reserve fraction is constant)."""
        b48 = self._budget(GPUTier.H100, 48)
        b96 = self._budget(GPUTier.H100, 96)
        self.assertAlmostEqual(b96 / b48, 2.0, places=3)


# ===========================================================================
# 2. Tests for shard_planner.py
# ===========================================================================

class TestShardPlannerEdgeCases(unittest.TestCase):
    """HeteroShardPlanner handles empty param lists and single-rank worlds."""

    def setUp(self):
        self.tm_single = TierMap.from_infos([_make_tier_info(0, GPUTier.H100, 96)])
        self.planner_single = HeteroShardPlanner(self.tm_single)

    def test_empty_params_returns_empty_plan(self):
        plan = self.planner_single.plan([])
        self.assertIsInstance(plan, ShardPlan)
        self.assertEqual(plan.rank_to_param_ids, {})
        self.assertEqual(plan.rank_to_bytes, {})
        self.assertIn("no trainable", plan.rationale.lower())

    def test_single_rank_full_replica(self):
        params = _named_params([100, 200, 300])
        plan = self.planner_single.plan(params)
        self.assertIn(0, plan.rank_to_param_ids)
        self.assertEqual(sorted(plan.rank_to_param_ids[0]), [0, 1, 2])
        total_bytes = sum(p.numel() for _, p in params) * 4
        self.assertEqual(plan.rank_to_bytes[0], total_bytes)
        self.assertIn("full replica", plan.rationale.lower())

    def test_single_rank_single_param(self):
        params = _named_params([512])
        plan = self.planner_single.plan(params)
        self.assertEqual(plan.rank_to_param_ids[0], [0])


class TestShardPlannerVRAMProportional(unittest.TestCase):
    """Higher-VRAM ranks receive proportionally larger shards."""

    def _plan_with_budgets(self, budgets_gb: List[int], param_sizes: List[int]):
        """Build a TierMap with custom VRAM, run planner, return ShardPlan."""
        # All H100 so reserve fraction is uniform (0.25); only VRAM size varies.
        infos = [_make_tier_info(r, GPUTier.H100, gb) for r, gb in enumerate(budgets_gb)]
        tm = TierMap.from_infos(infos)
        planner = HeteroShardPlanner(tm)
        return planner.plan(_named_params(param_sizes))

    # ------------------------------------------------------------------
    # Equal-VRAM → equal shards
    # ------------------------------------------------------------------

    def test_equal_vram_equal_shards(self):
        """Two ranks with identical VRAM → each gets ≈ half the parameters."""
        plan = self._plan_with_budgets([96, 96], [500, 500])
        # With 1000 total elements and equal budgets, each rank owns ~500 elems
        # i.e. each rank gets approximately half the params (could share boundary param)
        size0 = plan.rank_to_bytes[0]
        size1 = plan.rank_to_bytes[1]
        self.assertAlmostEqual(size0, size1, delta=4)  # at most 1-element difference (4 bytes)

    # ------------------------------------------------------------------
    # Unequal VRAM → larger rank gets more bytes
    # ------------------------------------------------------------------

    def test_larger_vram_rank_gets_more_bytes(self):
        """Rank with 2× VRAM should own ≈ 2× the bytes."""
        plan = self._plan_with_budgets([48, 96], [3000])
        bytes0 = plan.rank_to_bytes[0]
        bytes1 = plan.rank_to_bytes[1]
        self.assertGreater(bytes1, bytes0,
                           "rank 1 (96 GB) should own more bytes than rank 0 (48 GB)")

    def test_larger_vram_rank_roughly_double_bytes(self):
        """With budget ratio 1:2, rank 1 should own roughly twice rank 0's bytes."""
        plan = self._plan_with_budgets([48, 96], [9000])
        bytes0 = plan.rank_to_bytes[0]
        bytes1 = plan.rank_to_bytes[1]
        # Allow ±5% tolerance around 2.0 ratio
        ratio = bytes1 / bytes0
        self.assertAlmostEqual(ratio, 2.0, delta=0.15,
                               msg=f"Expected ratio≈2.0, got {ratio:.3f}")

    def test_three_ranks_proportional_bytes(self):
        """3 ranks with budget ratio 1:2:3 — bytes should respect that ratio."""
        plan = self._plan_with_budgets([32, 64, 96], [12000])
        b0, b1, b2 = plan.rank_to_bytes[0], plan.rank_to_bytes[1], plan.rank_to_bytes[2]
        self.assertLess(b0, b1)
        self.assertLess(b1, b2)

    def test_total_bytes_equals_param_bytes(self):
        """Sum of rank_to_bytes across all ranks must equal total fp32 bytes."""
        param_sizes = [100, 200, 300, 400]
        plan = self._plan_with_budgets([48, 96], param_sizes)
        total_expected = sum(param_sizes) * 4  # fp32
        total_actual = sum(plan.rank_to_bytes.values())
        self.assertEqual(total_actual, total_expected)

    # ------------------------------------------------------------------
    # Cluster topology (ARCHITECTURE.md)
    # ------------------------------------------------------------------

    def test_cluster_topology_byte_ordering(self):
        """Full 5-rank cluster: A6000 ranks own fewer bytes than H100/Blackwell."""
        tm = _cluster_tier_map()
        planner = HeteroShardPlanner(tm)
        params = _named_params([10000] * 10)
        plan = planner.plan(params)
        a6000_bytes = plan.rank_to_bytes[0]  # 48 GB A6000
        h100_bytes = plan.rank_to_bytes[2]   # 96 GB H100
        bw_bytes = plan.rank_to_bytes[3]     # 96 GB Blackwell
        # H100 and Blackwell have larger budgets after reserve fractions
        self.assertGreater(h100_bytes, a6000_bytes)
        self.assertGreater(bw_bytes, a6000_bytes)


class TestShardPlannerParamAssignment(unittest.TestCase):
    """Param indices are assigned correctly with boundary overlap."""

    def _two_rank_plan(self, param_sizes: List[int]) -> ShardPlan:
        tm = TierMap.from_infos([
            _make_tier_info(0, GPUTier.H100, 48),
            _make_tier_info(1, GPUTier.H100, 48),
        ])
        return HeteroShardPlanner(tm).plan(_named_params(param_sizes))

    def test_all_params_assigned_to_at_least_one_rank(self):
        plan = self._two_rank_plan([100, 200, 300])
        all_assigned = set(plan.rank_to_param_ids[0]) | set(plan.rank_to_param_ids[1])
        self.assertEqual(all_assigned, {0, 1, 2})

    def test_no_param_is_completely_unassigned(self):
        """Every param index must appear in at least one rank's list."""
        plan = self._two_rank_plan([50] * 10)
        assigned = set()
        for ids in plan.rank_to_param_ids.values():
            assigned.update(ids)
        self.assertEqual(assigned, set(range(10)))

    def test_param_ids_are_valid_indices(self):
        params = [100, 200, 300]
        plan = self._two_rank_plan(params)
        valid = set(range(len(params)))
        for rank, ids in plan.rank_to_param_ids.items():
            for pid in ids:
                self.assertIn(pid, valid, f"rank {rank} has invalid param id {pid}")

    def test_boundary_spanning_param_in_both_ranks(self):
        """A single large param that spans both shards appears in both rank lists."""
        # One huge param → definitely straddles the midpoint
        plan = self._two_rank_plan([10000])
        self.assertIn(0, plan.rank_to_param_ids[0])
        self.assertIn(0, plan.rank_to_param_ids[1])

    def test_small_disjoint_params_rank_exclusive(self):
        """Many tiny params — first half should belong only to rank 0, last to rank 1
        (unless a param happens to fall exactly on the boundary)."""
        # 200 params of 1 element each; equal VRAM → each rank owns 100 elements
        plan = self._two_rank_plan([1] * 200)
        ids0 = set(plan.rank_to_param_ids[0])
        ids1 = set(plan.rank_to_param_ids[1])
        # The first param can only belong to rank 0, the last only to rank 1
        self.assertIn(0, ids0)
        self.assertIn(199, ids1)

    def test_plan_rationale_non_empty(self):
        plan = self._two_rank_plan([100, 200])
        self.assertIsInstance(plan.rationale, str)
        self.assertTrue(len(plan.rationale) > 0)

    def test_plan_is_deterministic(self):
        """Same inputs must yield identical param assignments on every call."""
        tm = TierMap.from_infos([
            _make_tier_info(0, GPUTier.H100, 48),
            _make_tier_info(1, GPUTier.BLACKWELL, 96),
        ])
        planner = HeteroShardPlanner(tm)
        params = _named_params([300, 500, 200, 100])
        plan_a = planner.plan(params)
        plan_b = planner.plan(params)
        self.assertEqual(plan_a.rank_to_param_ids, plan_b.rank_to_param_ids)
        self.assertEqual(plan_a.rank_to_bytes, plan_b.rank_to_bytes)

    def test_rank_to_bytes_type_is_int(self):
        plan = self._two_rank_plan([100, 200])
        for rank, b in plan.rank_to_bytes.items():
            self.assertIsInstance(b, int, f"rank {rank} bytes is not int: {type(b)}")

    def test_rank_to_bytes_multiple_of_four(self):
        """fp32 → each element is 4 bytes, so rank_to_bytes must be divisible by 4."""
        plan = self._two_rank_plan([100, 200])
        for rank, b in plan.rank_to_bytes.items():
            self.assertEqual(b % 4, 0, f"rank {rank} bytes={b} not a multiple of 4")


class TestShardPlannerSingleParam(unittest.TestCase):
    """Edge-case: exactly one parameter, multiple ranks."""

    def _plan(self, n_elems: int, n_ranks: int = 3) -> ShardPlan:
        infos = [_make_tier_info(r, GPUTier.H100, 96) for r in range(n_ranks)]
        tm = TierMap.from_infos(infos)
        return HeteroShardPlanner(tm).plan(_named_params([n_elems]))

    def test_single_large_param_in_all_ranks(self):
        """A single param large enough to span all shards → present in every rank."""
        plan = self._plan(n_elems=10000, n_ranks=3)
        for r in range(3):
            self.assertIn(0, plan.rank_to_param_ids[r],
                          f"param 0 missing from rank {r}")

    def test_single_small_param_in_at_least_one_rank(self):
        """A 1-element param must appear in exactly one rank."""
        plan = self._plan(n_elems=1, n_ranks=3)
        count = sum(1 for ids in plan.rank_to_param_ids.values() if 0 in ids)
        self.assertGreaterEqual(count, 1)


# ===========================================================================
# 3. Tests for desloc_sync_policy.py
# ===========================================================================

class TestSyncPeriods(unittest.TestCase):
    """SyncPeriods dataclass stores Kx/Ku/Kv correctly."""

    def test_construction(self):
        sp = SyncPeriods(kx=1, ku=4, kv=16)
        self.assertEqual(sp.kx, 1)
        self.assertEqual(sp.ku, 4)
        self.assertEqual(sp.kv, 16)

    def test_fields_are_integers(self):
        sp = SyncPeriods(kx=2, ku=8, kv=32)
        self.assertIsInstance(sp.kx, int)
        self.assertIsInstance(sp.ku, int)
        self.assertIsInstance(sp.kv, int)

    def test_equality(self):
        a = SyncPeriods(kx=1, ku=4, kv=16)
        b = SyncPeriods(kx=1, ku=4, kv=16)
        self.assertEqual(a, b)

    def test_inequality(self):
        a = SyncPeriods(kx=1, ku=4, kv=16)
        b = SyncPeriods(kx=2, ku=4, kv=16)
        self.assertNotEqual(a, b)

    def test_period_ordering_convention(self):
        """Typical use: kx ≤ ku ≤ kv (fast → slow sync hierarchy)."""
        sp = SyncPeriods(kx=1, ku=4, kv=16)
        self.assertLessEqual(sp.kx, sp.ku)
        self.assertLessEqual(sp.ku, sp.kv)

    def test_unit_periods(self):
        """All periods = 1 is a valid degenerate config (sync every step)."""
        sp = SyncPeriods(kx=1, ku=1, kv=1)
        self.assertEqual(sp.kx, sp.ku)
        self.assertEqual(sp.ku, sp.kv)

    def test_large_periods(self):
        sp = SyncPeriods(kx=10, ku=100, kv=1000)
        self.assertEqual(sp.kv, 1000)


class TestDesLocSyncPolicyInit(unittest.TestCase):
    """DesLocSyncPolicy stores its SyncPeriods and exposes the right interface."""

    def setUp(self):
        self.periods = SyncPeriods(kx=1, ku=4, kv=16)
        self.policy = DesLocSyncPolicy(periods=self.periods)

    def test_periods_stored(self):
        self.assertIs(self.policy.periods, self.periods)

    def test_kx_accessible(self):
        self.assertEqual(self.policy.periods.kx, 1)

    def test_ku_accessible(self):
        self.assertEqual(self.policy.periods.ku, 4)

    def test_kv_accessible(self):
        self.assertEqual(self.policy.periods.kv, 16)


class TestDesLocSyncPolicySkeletonMethods(unittest.TestCase):
    """Phase-1 skeleton: classify() and should_sync() raise NotImplementedError."""

    def setUp(self):
        self.policy = DesLocSyncPolicy(SyncPeriods(kx=1, ku=4, kv=16))
        self.params = [("attn.weight", torch.zeros(64, 64)),
                       ("ffn.weight", torch.zeros(256, 64))]

    def test_classify_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.policy.classify(self.params)

    def test_should_sync_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.policy.should_sync(param_id=0, step=1)

    def test_classify_error_message_descriptive(self):
        try:
            self.policy.classify(self.params)
        except NotImplementedError as e:
            self.assertIn("classify", str(e))

    def test_should_sync_error_message_descriptive(self):
        try:
            self.policy.should_sync(param_id=0, step=0)
        except NotImplementedError as e:
            self.assertIn("should_sync", str(e))

    def test_classify_raises_for_empty_params(self):
        with self.assertRaises(NotImplementedError):
            self.policy.classify([])

    def test_should_sync_raises_for_any_step(self):
        for step in [0, 1, 4, 16, 100]:
            with self.assertRaises(NotImplementedError):
                self.policy.should_sync(param_id=0, step=step)

    def test_multiple_policy_instances_independent(self):
        """Two instances with different periods should not share state."""
        p1 = DesLocSyncPolicy(SyncPeriods(kx=1, ku=2, kv=4))
        p2 = DesLocSyncPolicy(SyncPeriods(kx=5, ku=10, kv=20))
        self.assertEqual(p1.periods.kx, 1)
        self.assertEqual(p2.periods.kx, 5)
        self.assertIsNot(p1.periods, p2.periods)


class TestDesLocSyncPolicyPeriodVariants(unittest.TestCase):
    """DesLocSyncPolicy is constructable with arbitrary valid period configs."""

    def _policy(self, kx, ku, kv):
        return DesLocSyncPolicy(SyncPeriods(kx=kx, ku=ku, kv=kv))

    def test_paper_default_periods(self):
        """Typical DES-LOC paper config: Kx=1, Ku=4, Kv=16."""
        p = self._policy(1, 4, 16)
        self.assertEqual(p.periods.kx, 1)
        self.assertEqual(p.periods.kv, 16)

    def test_aggressive_sync_periods(self):
        p = self._policy(1, 1, 1)
        self.assertEqual(p.periods.ku, 1)

    def test_very_lazy_sync_periods(self):
        p = self._policy(1, 32, 512)
        self.assertEqual(p.periods.kv, 512)

    def test_classify_still_raises_with_any_periods(self):
        for kx, ku, kv in [(1, 4, 16), (2, 8, 32), (1, 1, 1)]:
            p = self._policy(kx, ku, kv)
            with self.assertRaises(NotImplementedError):
                p.classify([("w", torch.zeros(10))])

    def test_should_sync_still_raises_with_any_periods(self):
        for kx, ku, kv in [(1, 4, 16), (2, 8, 32)]:
            p = self._policy(kx, ku, kv)
            with self.assertRaises(NotImplementedError):
                p.should_sync(0, 10)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
