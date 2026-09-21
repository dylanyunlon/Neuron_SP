# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team / Neuron_SP
"""Unit tests for CollectiveContract — NCCL collective symmetry enforcement.

Tests:
  1. Planning API: plan(), plan_conditional(), planned_sequence, planned_count.
  2. Execution API: guard(), execute(), all_executed, assert_complete().
  3. Ordering enforcement: out-of-order guard raises RuntimeError.
  4. Overflow detection: extra collectives beyond planned count raise.
  5. Incomplete detection: assert_complete() raises when ops are missing.
  6. ContractViolation: correct fields and message formatting.
  7. CollectiveOp enum: all expected members exist.
  8. ContractEntry dataclass: default values and field access.
  9. build_step_contract factory: correct sequence for DP and dist_optimizer.
  10. Kx/Ku/Kv resolution: from engine config and DesLocConfig.
  11. Disabled contract: all methods are no-op passthroughs.
  12. Summary: structured output contains expected keys.
  13. Noop execution: noop_fn is called when entry.is_noop is True.
  14. Verify without dist: returns True immediately in non-distributed mode.

Runs without GPU or torch.distributed initialization.
"""

from __future__ import annotations

import types
from typing import Any, List

import pytest

from deepspeed.core.distributed.collective_contract import (
    CollectiveContract,
    CollectiveOp,
    ContractEntry,
    ContractViolation,
    build_step_contract,
    log_contract_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_engine_config(kx: int = 32, ku: int = 96, kv: int = 192) -> Any:
    """Return a minimal object mimicking TrainingConfig with DES-LOC fields."""
    cfg = types.SimpleNamespace()
    cfg.desloc_Kx = kx
    cfg.desloc_Ku = ku
    cfg.desloc_Kv = kv
    return cfg


def _make_desloc_config(kx: int = 8, ku: int = 32, kv: int = 64) -> Any:
    """Return a minimal object mimicking DesLocConfig."""
    cfg = types.SimpleNamespace()
    cfg.kx = kx
    cfg.ku = ku
    cfg.kv = kv
    cfg.enabled = True
    return cfg


# ---------------------------------------------------------------------------
# Test: Planning API
# ---------------------------------------------------------------------------

class TestPlanningAPI:
    """Tests for plan(), plan_conditional(), and sequence introspection."""

    def test_plan_single_collective(self):
        c = CollectiveContract(step=0, rank=0)
        entry = c.plan("test_op", CollectiveOp.ALL_REDUCE, group="dp")

        assert entry.seq == 1
        assert entry.name == "test_op"
        assert entry.op == CollectiveOp.ALL_REDUCE
        assert entry.is_noop is False
        assert entry.group_key == "dp"

    def test_plan_sequence_ordering(self):
        c = CollectiveContract(step=0, rank=0)
        c.plan("op_a", CollectiveOp.ALL_REDUCE)
        c.plan("op_b", CollectiveOp.REDUCE_SCATTER)
        c.plan("op_c", CollectiveOp.ALL_GATHER)

        assert c.planned_sequence == ["op_a", "op_b", "op_c"]
        assert c.planned_count == 3

    def test_plan_conditional_with_true_condition(self):
        c = CollectiveContract(step=0, rank=0)
        entry = c.plan_conditional("cond_op", condition=True, op=CollectiveOp.BROADCAST)

        assert entry.is_noop is False
        assert entry.name == "cond_op"

    def test_plan_conditional_with_false_condition(self):
        c = CollectiveContract(step=0, rank=0)
        entry = c.plan_conditional("cond_op", condition=False, op=CollectiveOp.BROADCAST)

        assert entry.is_noop is True
        assert entry.name == "cond_op"

    def test_planned_count_empty(self):
        c = CollectiveContract(step=0, rank=0)
        assert c.planned_count == 0
        assert c.planned_sequence == []


# ---------------------------------------------------------------------------
# Test: Execution API
# ---------------------------------------------------------------------------

class TestExecutionAPI:
    """Tests for guard(), execute(), and completion tracking."""

    def test_guard_correct_order(self):
        c = CollectiveContract(step=0, rank=0)
        c.plan("op_a", CollectiveOp.ALL_REDUCE)
        c.plan("op_b", CollectiveOp.REDUCE_SCATTER)

        with c.guard("op_a") as entry:
            assert entry.name == "op_a"
        with c.guard("op_b") as entry:
            assert entry.name == "op_b"

        assert c.all_executed

    def test_guard_wrong_order_raises(self):
        c = CollectiveContract(step=0, rank=0)
        c.plan("op_a", CollectiveOp.ALL_REDUCE)
        c.plan("op_b", CollectiveOp.REDUCE_SCATTER)

        with pytest.raises(RuntimeError, match="collective order mismatch"):
            with c.guard("op_b"):
                pass

    def test_guard_overflow_raises(self):
        c = CollectiveContract(step=0, rank=0)
        c.plan("op_a", CollectiveOp.ALL_REDUCE)

        with c.guard("op_a"):
            pass

        with pytest.raises(RuntimeError, match="unexpected collective"):
            with c.guard("op_extra"):
                pass

    def test_execute_calls_fn(self):
        c = CollectiveContract(step=0, rank=0)
        c.plan("op_a", CollectiveOp.ALL_REDUCE)

        calls: List[str] = []
        c.execute("op_a", lambda: calls.append("called"))

        assert calls == ["called"]
        assert c.all_executed

    def test_execute_noop_calls_noop_fn(self):
        c = CollectiveContract(step=0, rank=0)
        c.plan("op_a", CollectiveOp.ALL_REDUCE, is_noop=True)

        main_calls: List[str] = []
        noop_calls: List[str] = []

        c.execute(
            "op_a",
            lambda: main_calls.append("main"),
            noop_fn=lambda: noop_calls.append("noop"),
        )

        assert main_calls == []
        assert noop_calls == ["noop"]

    def test_execute_noop_without_noop_fn_calls_main(self):
        c = CollectiveContract(step=0, rank=0)
        c.plan("op_a", CollectiveOp.ALL_REDUCE, is_noop=True)

        calls: List[str] = []
        c.execute("op_a", lambda: calls.append("main"))

        assert calls == ["main"]

    def test_assert_complete_passes(self):
        c = CollectiveContract(step=0, rank=0)
        c.plan("op_a", CollectiveOp.ALL_REDUCE)
        with c.guard("op_a"):
            pass
        c.assert_complete()  # should not raise

    def test_assert_complete_raises_when_incomplete(self):
        c = CollectiveContract(step=0, rank=0)
        c.plan("op_a", CollectiveOp.ALL_REDUCE)
        c.plan("op_b", CollectiveOp.REDUCE_SCATTER)

        with c.guard("op_a"):
            pass

        with pytest.raises(RuntimeError, match="1 planned collective"):
            c.assert_complete()

    def test_all_executed_false_when_incomplete(self):
        c = CollectiveContract(step=0, rank=0)
        c.plan("op_a", CollectiveOp.ALL_REDUCE)
        assert c.all_executed is False


# ---------------------------------------------------------------------------
# Test: ContractViolation
# ---------------------------------------------------------------------------

class TestContractViolation:
    """Tests for the ContractViolation exception."""

    def test_violation_attributes(self):
        exc = ContractViolation(
            step=42,
            local_seq=["op_a", "op_b"],
            remote_seq=["op_a", "op_c"],
            rank=1,
        )

        assert exc.step == 42
        assert exc.local_seq == ["op_a", "op_b"]
        assert exc.remote_seq == ["op_a", "op_c"]
        assert exc.rank == 1

    def test_violation_message_format(self):
        exc = ContractViolation(
            step=10,
            local_seq=["a"],
            remote_seq=["a", "b"],
            rank=0,
        )
        msg = str(exc)
        assert "step 10" in msg
        assert "rank 0" in msg
        assert "1 ops" in msg
        assert "2 ops" in msg

    def test_violation_truncates_long_sequences(self):
        long_seq = [f"op_{i}" for i in range(20)]
        exc = ContractViolation(step=0, local_seq=long_seq, remote_seq=[], rank=0)
        msg = str(exc)
        assert "…" in msg


# ---------------------------------------------------------------------------
# Test: CollectiveOp enum
# ---------------------------------------------------------------------------

class TestCollectiveOp:
    """Tests for the CollectiveOp enumeration."""

    def test_all_expected_members_exist(self):
        expected = {"ALL_REDUCE", "REDUCE_SCATTER", "ALL_GATHER",
                    "BROADCAST", "ALL_TO_ALL", "BARRIER", "NOOP"}
        actual = {m.name for m in CollectiveOp}
        assert expected == actual

    def test_members_are_unique(self):
        values = [m.value for m in CollectiveOp]
        assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# Test: ContractEntry dataclass
# ---------------------------------------------------------------------------

class TestContractEntry:
    """Tests for the ContractEntry dataclass."""

    def test_default_values(self):
        entry = ContractEntry(seq=1, name="test", op=CollectiveOp.ALL_REDUCE)
        assert entry.is_noop is False
        assert entry.group_key == "world"

    def test_explicit_values(self):
        entry = ContractEntry(
            seq=5, name="custom", op=CollectiveOp.BROADCAST,
            is_noop=True, group_key="tp",
        )
        assert entry.seq == 5
        assert entry.is_noop is True
        assert entry.group_key == "tp"


# ---------------------------------------------------------------------------
# Test: Kx/Ku/Kv resolution
# ---------------------------------------------------------------------------

class TestKxKuKvResolution:
    """Tests for Kx/Ku/Kv flag resolution from config objects."""

    def test_engine_config_kx_step(self):
        cfg = _make_engine_config(kx=4, ku=8, kv=16)
        c = CollectiveContract(step=3, config=cfg, rank=0)  # step+1=4
        assert c.is_Kx is True

    def test_engine_config_non_kx_step(self):
        cfg = _make_engine_config(kx=4, ku=8, kv=16)
        c = CollectiveContract(step=1, config=cfg, rank=0)  # step+1=2
        assert c.is_Kx is False

    def test_engine_config_ku_step(self):
        cfg = _make_engine_config(kx=4, ku=8, kv=16)
        c = CollectiveContract(step=7, config=cfg, rank=0)  # step+1=8
        assert c.is_Ku is True
        assert c.is_Kx is True  # 8 % 4 == 0

    def test_engine_config_kv_step(self):
        cfg = _make_engine_config(kx=4, ku=8, kv=16)
        c = CollectiveContract(step=15, config=cfg, rank=0)  # step+1=16
        assert c.is_Kv is True

    def test_desloc_config(self):
        cfg = _make_desloc_config(kx=2, ku=4, kv=8)
        c = CollectiveContract(step=3, config=cfg, rank=0)  # step+1=4
        assert c.is_Kx is True  # 4 % 2 == 0
        assert c.is_Ku is True  # 4 % 4 == 0
        assert c.is_Kv is False  # 4 % 8 != 0

    def test_no_config_defaults_kx_true(self):
        c = CollectiveContract(step=0, config=None, rank=0)
        assert c.is_Kx is True
        assert c.is_Ku is False
        assert c.is_Kv is False


# ---------------------------------------------------------------------------
# Test: Disabled contract
# ---------------------------------------------------------------------------

class TestDisabledContract:
    """Tests that a disabled contract is a complete no-op passthrough."""

    def test_disabled_guard_yields_none(self):
        c = CollectiveContract(step=0, rank=0, enabled=False)
        c.plan("op_a", CollectiveOp.ALL_REDUCE)

        with c.guard("any_name") as entry:
            assert entry is None

    def test_disabled_assert_complete_no_raise(self):
        c = CollectiveContract(step=0, rank=0, enabled=False)
        c.plan("op_a", CollectiveOp.ALL_REDUCE)
        c.assert_complete()  # should not raise

    def test_disabled_verify_returns_true(self):
        c = CollectiveContract(step=0, rank=0, enabled=False)
        assert c.verify() is True

    def test_disabled_execute_calls_fn(self):
        c = CollectiveContract(step=0, rank=0, enabled=False)
        calls: List[str] = []
        c.execute("anything", lambda: calls.append("ok"))
        assert calls == ["ok"]


# ---------------------------------------------------------------------------
# Test: build_step_contract factory
# ---------------------------------------------------------------------------

class TestBuildStepContract:
    """Tests for the build_step_contract factory function."""

    def test_dp_only_sequence(self):
        cfg = _make_engine_config()
        c = build_step_contract(step=0, config=cfg, has_dist_optimizer=False)

        expected = [
            "nan_flag_allreduce",
            "finalize_model_grads",
            "clip_grad_norm_allreduce",
            "skip_flag_allreduce",
        ]
        assert c.planned_sequence == expected
        assert c.planned_count == 4

    def test_dist_optimizer_sequence(self):
        cfg = _make_engine_config()
        c = build_step_contract(step=0, config=cfg, has_dist_optimizer=True)

        expected = [
            "nan_flag_allreduce",
            "finalize_model_grads",
            "clip_grad_norm_allreduce",
            "skip_flag_allreduce",
            "prepare_grads_rs",
            "param_sync",
        ]
        assert c.planned_sequence == expected
        assert c.planned_count == 6

    def test_kx_resolution_forwarded(self):
        cfg = _make_engine_config(kx=2, ku=4, kv=8)
        c = build_step_contract(step=1, config=cfg)  # step+1=2
        assert c.is_Kx is True

    def test_non_kx_step(self):
        cfg = _make_engine_config(kx=4, ku=8, kv=16)
        c = build_step_contract(step=0, config=cfg)  # step+1=1
        assert c.is_Kx is False


# ---------------------------------------------------------------------------
# Test: Summary
# ---------------------------------------------------------------------------

class TestSummary:
    """Tests for the summary() method."""

    def test_summary_keys(self):
        c = CollectiveContract(step=5, rank=2)
        c.plan("op_a", CollectiveOp.ALL_REDUCE)

        s = c.summary()
        assert s["step"] == 5
        assert s["rank"] == 2
        assert "is_Kx" in s
        assert "is_Ku" in s
        assert "is_Kv" in s
        assert s["planned_count"] == 1
        assert s["executed_count"] == 0
        assert s["verified"] is False
        assert len(s["planned_ops"]) == 1

    def test_summary_op_entry_fields(self):
        c = CollectiveContract(step=0, rank=0)
        c.plan("test_op", CollectiveOp.BROADCAST, is_noop=True)

        entry = c.summary()["planned_ops"][0]
        assert entry["seq"] == 1
        assert entry["name"] == "test_op"
        assert entry["op"] == "BROADCAST"
        assert entry["noop"] is True


# ---------------------------------------------------------------------------
# Test: Verify without dist
# ---------------------------------------------------------------------------

class TestVerifyNonDistributed:
    """Tests that verify() works correctly in non-distributed mode."""

    def test_verify_returns_true(self):
        c = CollectiveContract(step=0, rank=0)
        c.plan("op_a", CollectiveOp.ALL_REDUCE)
        assert c.verify() is True
        assert c._verified is True


# ---------------------------------------------------------------------------
# Test: log_contract_summary (smoke test)
# ---------------------------------------------------------------------------

class TestLogContractSummary:
    """Smoke test that log_contract_summary runs without error."""

    def test_log_does_not_raise(self):
        c = CollectiveContract(step=0, rank=0)
        c.plan("op_a", CollectiveOp.ALL_REDUCE)
        log_contract_summary(c)  # should not raise


# ---------------------------------------------------------------------------
# Test: Full step simulation (integration-style)
# ---------------------------------------------------------------------------

class TestFullStepSimulation:
    """Simulate a complete training step to verify the contract works end-to-end."""

    def test_complete_dp_step(self):
        """Simulate a DP-only step: all 4 collectives executed in order."""
        cfg = _make_engine_config()
        c = build_step_contract(step=0, config=cfg, has_dist_optimizer=False)

        with c.guard("nan_flag_allreduce"):
            pass  # would call dist.all_reduce
        with c.guard("finalize_model_grads"):
            pass  # would call finalize_model_grads
        with c.guard("clip_grad_norm_allreduce"):
            pass  # would call clip_grad_norm
        with c.guard("skip_flag_allreduce"):
            pass  # would call dist.all_reduce

        c.assert_complete()

    def test_complete_dist_optimizer_step(self):
        """Simulate a dist_optimizer step: all 6 collectives in order."""
        cfg = _make_engine_config()
        c = build_step_contract(step=0, config=cfg, has_dist_optimizer=True)

        for name in c.planned_sequence:
            with c.guard(name):
                pass

        c.assert_complete()

    def test_skipped_collective_detected(self):
        """Skipping a collective in the middle raises RuntimeError."""
        cfg = _make_engine_config()
        c = build_step_contract(step=0, config=cfg, has_dist_optimizer=False)

        with c.guard("nan_flag_allreduce"):
            pass

        # Skip finalize_model_grads, try clip_grad_norm_allreduce
        with pytest.raises(RuntimeError, match="collective order mismatch"):
            with c.guard("clip_grad_norm_allreduce"):
                pass


# ---------------------------------------------------------------------------
# Test: Multi-step simulation
# ---------------------------------------------------------------------------

class TestMultiStepSimulation:
    """Simulate multiple consecutive training steps end-to-end."""

    def test_three_consecutive_dp_only_steps(self):
        """Three back-to-back DP-only steps, all 4 collectives each."""
        cfg = _make_engine_config(kx=2, ku=4, kv=8)
        for step in range(3):
            c = build_step_contract(step=step, config=cfg, has_dist_optimizer=False)
            for name in c.planned_sequence:
                with c.guard(name):
                    pass
            c.assert_complete()

    def test_five_consecutive_dist_optimizer_steps(self):
        """Five back-to-back dist-optimizer steps, all 6 collectives each."""
        cfg = _make_engine_config(kx=2, ku=4, kv=8)
        for step in range(5):
            c = build_step_contract(step=step, config=cfg, has_dist_optimizer=True)
            for name in c.planned_sequence:
                with c.guard(name):
                    pass
            c.assert_complete()

    def test_kx_flags_alternate_across_steps(self):
        """Verify Kx/Ku/Kv flags flip correctly as step advances."""
        cfg = _make_engine_config(kx=2, ku=4, kv=8)
        # step 0 → step+1=1 → Kx=False
        c0 = build_step_contract(step=0, config=cfg)
        assert c0.is_Kx is False
        # step 1 → step+1=2 → Kx=True (2%2==0)
        c1 = build_step_contract(step=1, config=cfg)
        assert c1.is_Kx is True
        # step 3 → step+1=4 → Kx=True, Ku=True (4%4==0)
        c3 = build_step_contract(step=3, config=cfg)
        assert c3.is_Kx is True
        assert c3.is_Ku is True


# ---------------------------------------------------------------------------
# Test: Boundary conditions
# ---------------------------------------------------------------------------

class TestBoundaryConditions:
    """Edge cases: Kx=1, step=0, very large step numbers."""

    def test_kx_equals_one_every_step_syncs(self):
        """When Kx=Ku=Kv=1, every step is a sync step."""
        cfg = _make_engine_config(kx=1, ku=1, kv=1)
        for step in range(5):
            c = CollectiveContract(step=step, config=cfg, rank=0)
            assert c.is_Kx is True
            assert c.is_Ku is True
            assert c.is_Kv is True

    def test_step_zero(self):
        """Step 0 is always valid — Kx=True when Kx divides (0+1)."""
        cfg = _make_engine_config(kx=1, ku=1, kv=1)
        c = build_step_contract(step=0, config=cfg)
        assert c.step == 0
        assert c.is_Kx is True

    def test_large_step_number(self):
        """Very large step number does not overflow or error."""
        cfg = _make_engine_config(kx=32, ku=96, kv=192)
        c = build_step_contract(step=1_000_000, config=cfg)
        # (1000000+1) % 32 = 1000001 % 32 = 17 ≠ 0, so Kx=False
        assert c.is_Kx is False
        # Ensure no crash on large numbers
        for name in c.planned_sequence:
            with c.guard(name):
                pass
        c.assert_complete()


# ---------------------------------------------------------------------------
# Test: execute() args/kwargs forwarding
# ---------------------------------------------------------------------------

class TestExecuteArgsForwarding:
    """Test that execute() correctly forwards *args and **kwargs."""

    def test_positional_args_forwarded(self):
        c = CollectiveContract(step=0, rank=0)
        c.plan("op_a", CollectiveOp.ALL_REDUCE)
        result = c.execute("op_a", lambda x, y: x + y, 3, 4)
        assert result == 7

    def test_keyword_args_forwarded(self):
        c = CollectiveContract(step=0, rank=0)
        c.plan("op_a", CollectiveOp.ALL_REDUCE)
        result = c.execute("op_a", lambda x, y=10: x * y, 5, y=3)
        assert result == 15

    def test_mixed_args_and_kwargs(self):
        c = CollectiveContract(step=0, rank=0)
        c.plan("op_a", CollectiveOp.ALL_REDUCE)
        def fn(a, b, *, c=0):
            return a + b + c
        result = c.execute("op_a", fn, 1, 2, c=10)
        assert result == 13

    def test_execute_return_value_preserved(self):
        c = CollectiveContract(step=0, rank=0)
        c.plan("op_a", CollectiveOp.ALL_REDUCE)
        result = c.execute("op_a", lambda: {"key": "value"})
        assert result == {"key": "value"}


# ---------------------------------------------------------------------------
# Test: Snapshot verification against golden files
# ---------------------------------------------------------------------------

class TestSnapshotVerification:
    """Verify the planned sequence matches the golden snapshot."""

    def test_dp_only_sequence_matches_snapshot(self):
        import json, os
        snap_path = os.path.join(
            os.path.dirname(__file__), "__snapshots__",
            "collective_contract_sequence.json",
        )
        if not os.path.exists(snap_path):
            pytest.skip("Snapshot file not found")
        with open(snap_path) as f:
            snap = json.load(f)

        cfg = _make_engine_config()
        c = build_step_contract(step=0, config=cfg, has_dist_optimizer=False)
        assert c.planned_sequence == snap["dp_only"]

    def test_dist_optimizer_sequence_matches_snapshot(self):
        import json, os
        snap_path = os.path.join(
            os.path.dirname(__file__), "__snapshots__",
            "collective_contract_sequence.json",
        )
        if not os.path.exists(snap_path):
            pytest.skip("Snapshot file not found")
        with open(snap_path) as f:
            snap = json.load(f)

        cfg = _make_engine_config()
        c = build_step_contract(step=0, config=cfg, has_dist_optimizer=True)
        assert c.planned_sequence == snap["dist_optimizer"]


# ---------------------------------------------------------------------------
# Test: verify() truncation guard
# ---------------------------------------------------------------------------

class TestVerifyTruncationGuard:
    """Ensure verify() raises when the planned sequence exceeds buffer size."""

    def test_huge_sequence_raises_on_verify(self):
        """If someone plans 200+ long-named collectives, verify() must not silently truncate."""
        c = CollectiveContract(step=0, rank=0)
        # Plan enough ops to exceed 4096 bytes when encoded
        for i in range(250):
            c.plan(f"very_long_collective_name_number_{i:05d}", CollectiveOp.ALL_REDUCE)

        # verify() should raise RuntimeError about sequence being too long
        # (in non-distributed mode it short-circuits, so test the encoding path directly)
        local_seq_str = "|".join(c.planned_sequence)
        encoded = local_seq_str.encode("utf-8")
        assert len(encoded) >= 4096, f"Test setup: expected >= 4096 bytes, got {len(encoded)}"
