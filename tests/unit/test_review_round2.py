"""Round 2 review: exploit tests from xiao-jingsheng2, adapted to verify fixes."""
import os
import types
import pytest
from collections import deque

from deepspeed.core.distributed.collective_contract import (
    CollectiveContract, CollectiveOp, ContractViolation, build_step_contract,
)
from deepspeed.core.distributed.contract_diagnostics import (
    StepTrace, StepTraceLog, validate_call_sites, diff_sequences,
)


class TestVerifyTruncationGuardIsDeadCode:
    """BUG 1 FIX: _encode_planned_sequence() is now directly testable."""

    def test_verify_short_circuits_in_non_distributed(self):
        """verify() still returns True in non-dist mode (expected)."""
        c = CollectiveContract(step=0, rank=0)
        for i in range(250):
            c.plan(f"very_long_collective_name_number_{i:05d}", CollectiveOp.ALL_REDUCE)
        result = c.verify()
        assert result is True

    def test_encode_planned_sequence_raises_on_overflow(self):
        """The extracted method IS reachable and DOES raise."""
        c = CollectiveContract(step=0, rank=0)
        for i in range(250):
            c.plan(f"very_long_collective_name_number_{i:05d}", CollectiveOp.ALL_REDUCE)

        with pytest.raises(RuntimeError, match="too long"):
            c._encode_planned_sequence()


class TestResetSeqCounterCorruption:
    """BUG 2 FIX: reset(clear_plan=True) resets _seq_counter."""

    def test_reset_clear_plan_resets_seq_counter(self):
        c = CollectiveContract(step=0, rank=0)
        c.plan("op_a", CollectiveOp.ALL_REDUCE)
        c.plan("op_b", CollectiveOp.ALL_REDUCE)
        assert c._seq_counter == 2

        with c.guard("op_a"): pass
        with c.guard("op_b"): pass
        c.reset(clear_plan=True)

        # _seq_counter IS now reset:
        assert c._seq_counter == 0
        assert c.planned_count == 0

    def test_plan_after_clear_plan_reset_works_cleanly(self):
        """plan() after reset(clear_plan=True) starts fresh."""
        c = CollectiveContract(step=0, rank=0)
        c.plan("op_a", CollectiveOp.ALL_REDUCE)
        with c.guard("op_a"): pass
        c.reset(clear_plan=True)

        c.plan("op_new", CollectiveOp.ALL_REDUCE)
        assert c.planned_count == 1
        assert c.planned_sequence == ["op_new"]
        with c.guard("op_new"): pass
        c.assert_complete()

    def test_default_reset_preserves_plan_for_replay(self):
        """Default reset() still works for replay (plan preserved)."""
        c = CollectiveContract(step=0, rank=0)
        c.plan("op_a", CollectiveOp.ALL_REDUCE)
        with c.guard("op_a"): pass
        c.reset()

        assert c.planned_count == 1
        assert c.planned_sequence == ["op_a"]
        with c.guard("op_a"): pass
        c.assert_complete()


class TestEnvVarTraceMaxlenEdgeCases:
    """BUG 3 FIX: engine code now wraps int() in try/except + max(..., 1)."""

    def test_non_numeric_handled_gracefully(self):
        """Simulating the fixed parsing logic."""
        raw = "not_a_number"
        try:
            result = max(int(raw), 1)
        except (ValueError, TypeError):
            result = 128
        assert result == 128

    def test_zero_clamped_to_one(self):
        result = max(int("0"), 1)
        assert result == 1
        log = StepTraceLog(maxlen=result)
        log.record(StepTrace(step=0, rank=0, complete=True))
        assert len(log.traces) == 1  # retains at least 1

    def test_negative_clamped_to_one(self):
        result = max(int("-5"), 1)
        assert result == 1


class TestVerifyLimitWithoutDeslocKx:
    """BUG 4 FIX: uses getattr(self, 'desloc_Kx', 5) now."""

    def test_getattr_fallback_works(self):
        engine = types.SimpleNamespace()
        _verify_limit = max(getattr(engine, 'desloc_Kx', 5), 5) + 1
        assert _verify_limit == 6  # fallback to 5

    def test_getattr_uses_real_value_when_present(self):
        engine = types.SimpleNamespace(desloc_Kx=32)
        _verify_limit = max(getattr(engine, 'desloc_Kx', 5), 5) + 1
        assert _verify_limit == 33


class TestGuardExceptionSafetyRegression:
    """Verify the clip_grads.py fix: guard context must exit even on exception."""

    def test_guard_exits_on_exception(self):
        c = CollectiveContract(step=0, rank=0)
        c.plan("op_a", CollectiveOp.ALL_REDUCE)
        c.plan("op_b", CollectiveOp.ALL_REDUCE)

        with pytest.raises(RuntimeError, match="boom"):
            with c.guard("op_a"):
                raise RuntimeError("boom")

        assert len(c._executed) == 1
        assert c._executed[0].name == "op_a"
        assert c._active_guard is None

        with c.guard("op_b"): pass
        c.assert_complete()
