# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team / Neuron_SP
"""Unit tests for contract_diagnostics — diagnostic utilities for CollectiveContract.

Tests:
  1. format_violation_report: human-readable report with side-by-side diff.
  2. diff_sequences: symmetric, asymmetric, empty, and identical inputs.
  3. validate_call_sites: detects present and missing guard() invocations.
  4. StepTrace: construction from contract instance.
  5. StepTraceLog: ring buffer, incomplete_steps, summary.

Runs without GPU or torch.distributed initialization.
"""

from __future__ import annotations

import types
from typing import List

import pytest

from deepspeed.core.distributed.collective_contract import (
    CollectiveContract,
    CollectiveOp,
    ContractViolation,
    build_step_contract,
)
from deepspeed.core.distributed.contract_diagnostics import (
    format_violation_report,
    diff_sequences,
    validate_call_sites,
    StepTrace,
    StepTraceLog,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(kx: int = 32, ku: int = 96, kv: int = 192):
    cfg = types.SimpleNamespace()
    cfg.desloc_Kx = kx
    cfg.desloc_Ku = ku
    cfg.desloc_Kv = kv
    return cfg


# ---------------------------------------------------------------------------
# Test: format_violation_report
# ---------------------------------------------------------------------------

class TestFormatViolationReport:
    """Tests for the human-readable violation report."""

    def test_basic_report_contains_step_and_rank(self):
        report = format_violation_report(
            step=42, rank=1,
            local_seq=["op_a", "op_b"],
            remote_seq=["op_a", "op_c"],
            remote_rank=2,
        )
        assert "Step:        42" in report
        assert "Local rank:  1" in report
        assert "Remote rank: 2" in report

    def test_report_contains_diff_markers(self):
        report = format_violation_report(
            step=0, rank=0,
            local_seq=["op_a", "op_b"],
            remote_seq=["op_a", "op_c"],
        )
        assert "[✓]" in report  # op_a matches
        assert "[✗]" in report  # op_b vs op_c diverges

    def test_report_handles_length_mismatch(self):
        report = format_violation_report(
            step=0, rank=0,
            local_seq=["op_a", "op_b", "op_c"],
            remote_seq=["op_a"],
        )
        assert "<missing>" in report
        assert "Local ops:   3" in report
        assert "Remote ops:  1" in report

    def test_report_handles_empty_sequences(self):
        report = format_violation_report(
            step=0, rank=0,
            local_seq=[], remote_seq=[],
        )
        assert "Local ops:   0" in report
        assert "Remote ops:  0" in report

    def test_report_unknown_remote_rank(self):
        report = format_violation_report(
            step=0, rank=0,
            local_seq=["a"], remote_seq=["b"],
        )
        assert "unknown" in report


# ---------------------------------------------------------------------------
# Test: diff_sequences
# ---------------------------------------------------------------------------

class TestDiffSequences:
    """Tests for the side-by-side sequence diff."""

    def test_identical_sequences(self):
        result = diff_sequences(["a", "b", "c"], ["a", "b", "c"])
        assert all(match for _, _, match in result)
        assert len(result) == 3

    def test_divergent_sequences(self):
        result = diff_sequences(["a", "b"], ["a", "c"])
        assert result[0] == ("a", "a", True)
        assert result[1] == ("b", "c", False)

    def test_local_longer(self):
        result = diff_sequences(["a", "b", "c"], ["a"])
        assert len(result) == 3
        assert result[1] == ("b", "<missing>", False)
        assert result[2] == ("c", "<missing>", False)

    def test_remote_longer(self):
        result = diff_sequences(["a"], ["a", "b"])
        assert len(result) == 2
        assert result[1] == ("<missing>", "b", False)

    def test_both_empty(self):
        result = diff_sequences([], [])
        assert result == []

    def test_one_empty(self):
        result = diff_sequences([], ["a", "b"])
        assert len(result) == 2
        assert result[0] == ("<missing>", "a", False)


# ---------------------------------------------------------------------------
# Test: validate_call_sites
# ---------------------------------------------------------------------------

class TestValidateCallSites:
    """Tests for the static guard-name check."""

    def test_all_guards_present(self):
        source = '''
        with _contract.guard("nan_flag_allreduce"):
            pass
        with _contract.guard("finalize_model_grads"):
            pass
        with _contract.guard("clip_grad_norm_allreduce"):
            pass
        with _contract.guard("skip_flag_allreduce"):
            pass
        with _contract.guard("prepare_grads_rs"):
            pass
        with _contract.guard("param_sync"):
            pass
        '''
        results = validate_call_sites(source)
        assert all(results.values()), f"Missing guards: {[k for k, v in results.items() if not v]}"

    def test_missing_guards_detected(self):
        source = '''
        with _contract.guard("nan_flag_allreduce"):
            pass
        '''
        results = validate_call_sites(source)
        assert results["nan_flag_allreduce"] is True
        assert results["finalize_model_grads"] is False
        assert results["param_sync"] is False

    def test_custom_expected_guards(self):
        source = 'contract.guard("custom_op")'
        results = validate_call_sites(source, expected_guards=frozenset({"custom_op", "missing_op"}))
        assert results["custom_op"] is True
        assert results["missing_op"] is False

    def test_single_quotes_detected(self):
        source = "contract.guard('nan_flag_allreduce')"
        results = validate_call_sites(source, expected_guards=frozenset({"nan_flag_allreduce"}))
        assert results["nan_flag_allreduce"] is True

    def test_empty_source(self):
        results = validate_call_sites("")
        assert all(v is False for v in results.values())


# ---------------------------------------------------------------------------
# Test: StepTrace
# ---------------------------------------------------------------------------

class TestStepTrace:
    """Tests for StepTrace data class and from_contract factory."""

    def test_from_contract_basic(self):
        c = CollectiveContract(step=5, rank=2)
        c.plan("op_a", CollectiveOp.ALL_REDUCE)
        trace = StepTrace.from_contract(c)

        assert trace.step == 5
        assert trace.rank == 2
        assert trace.planned_count == 1
        assert trace.executed_count == 0
        assert trace.verified is False
        assert trace.complete is False

    def test_from_contract_after_execution(self):
        c = CollectiveContract(step=0, rank=0)
        c.plan("op_a", CollectiveOp.ALL_REDUCE)
        with c.guard("op_a"):
            pass
        trace = StepTrace.from_contract(c)

        assert trace.executed_count == 1
        assert trace.complete is True

    def test_from_contract_kx_flags(self):
        cfg = _make_config(kx=2, ku=4, kv=8)
        c = CollectiveContract(step=3, config=cfg, rank=0)  # step+1=4
        trace = StepTrace.from_contract(c)

        assert trace.is_Kx is True  # 4 % 2 == 0
        assert trace.is_Ku is True  # 4 % 4 == 0
        assert trace.is_Kv is False  # 4 % 8 != 0

    def test_direct_construction(self):
        trace = StepTrace(step=10, rank=1, planned_count=6, executed_count=6,
                          verified=True, complete=True)
        assert trace.step == 10
        assert trace.complete is True


# ---------------------------------------------------------------------------
# Test: StepTraceLog
# ---------------------------------------------------------------------------

class TestStepTraceLog:
    """Tests for the ring-buffer trace log."""

    def test_empty_log(self):
        log = StepTraceLog(maxlen=10)
        assert log.last is None
        assert log.traces == []
        assert log.incomplete_steps() == []

    def test_record_and_last(self):
        log = StepTraceLog(maxlen=10)
        t1 = StepTrace(step=0, rank=0, planned_count=4, executed_count=4, complete=True)
        t2 = StepTrace(step=1, rank=0, planned_count=4, executed_count=4, complete=True)
        log.record(t1)
        log.record(t2)

        assert log.last is t2
        assert len(log.traces) == 2

    def test_ring_buffer_eviction(self):
        log = StepTraceLog(maxlen=3)
        for i in range(5):
            log.record(StepTrace(step=i, rank=0, complete=True))

        assert len(log.traces) == 3
        assert log.traces[0].step == 2  # oldest surviving
        assert log.traces[-1].step == 4  # newest

    def test_incomplete_steps_filtering(self):
        log = StepTraceLog(maxlen=10)
        log.record(StepTrace(step=0, rank=0, planned_count=4, executed_count=4, complete=True))
        log.record(StepTrace(step=1, rank=0, planned_count=4, executed_count=3, complete=False))
        log.record(StepTrace(step=2, rank=0, planned_count=4, executed_count=4, complete=True))
        log.record(StepTrace(step=3, rank=0, planned_count=6, executed_count=5, complete=False))

        incomplete = log.incomplete_steps()
        assert len(incomplete) == 2
        assert incomplete[0].step == 1
        assert incomplete[1].step == 3

    def test_summary_keys(self):
        log = StepTraceLog(maxlen=10)
        log.record(StepTrace(step=5, rank=0, complete=True))
        log.record(StepTrace(step=6, rank=0, complete=False))

        s = log.summary()
        assert s["total_steps"] == 2
        assert s["incomplete_steps"] == 1
        assert s["oldest_step"] == 5
        assert s["newest_step"] == 6

    def test_summary_empty_log(self):
        log = StepTraceLog()
        s = log.summary()
        assert s["total_steps"] == 0
        assert s["oldest_step"] is None


# ---------------------------------------------------------------------------
# Test: ContractViolation.detailed_report integration
# ---------------------------------------------------------------------------

class TestViolationDetailedReport:
    """Test that ContractViolation.detailed_report() produces a useful report."""

    def test_detailed_report_contains_diff(self):
        exc = ContractViolation(
            step=10,
            local_seq=["nan_flag_allreduce", "finalize_model_grads", "clip_grad_norm_allreduce"],
            remote_seq=["nan_flag_allreduce", "finalize_model_grads"],
            rank=0,
            remote_rank=1,
        )
        report = exc.detailed_report()
        assert "Step:        10" in report
        assert "[✓]" in report
        assert "[✗]" in report
        assert "<missing>" in report

    def test_detailed_report_with_unknown_remote(self):
        exc = ContractViolation(
            step=0,
            local_seq=["a"], remote_seq=["b"],
            rank=0,
        )
        report = exc.detailed_report()
        assert "unknown" in report


# ---------------------------------------------------------------------------
# Test: CollectiveContract.__repr__
# ---------------------------------------------------------------------------

class TestContractRepr:
    """Test that CollectiveContract has a useful repr."""

    def test_repr_contains_key_fields(self):
        c = CollectiveContract(step=42, rank=3, enabled=True)
        c.plan("op_a", CollectiveOp.ALL_REDUCE)
        r = repr(c)

        assert "step=42" in r
        assert "rank=3" in r
        assert "planned=1" in r
        assert "enabled=True" in r


# ---------------------------------------------------------------------------
# Test: CollectiveContract.reset
# ---------------------------------------------------------------------------

class TestContractReset:
    """Test that reset() clears execution state but keeps the plan."""

    def test_reset_clears_executed(self):
        c = CollectiveContract(step=0, rank=0)
        c.plan("op_a", CollectiveOp.ALL_REDUCE)
        c.plan("op_b", CollectiveOp.REDUCE_SCATTER)

        with c.guard("op_a"):
            pass
        with c.guard("op_b"):
            pass
        assert c.all_executed

        c.reset()
        assert not c.all_executed
        assert c.planned_count == 2  # plan preserved
        assert c.planned_sequence == ["op_a", "op_b"]

    def test_reset_allows_replay(self):
        c = CollectiveContract(step=0, rank=0)
        c.plan("op_a", CollectiveOp.ALL_REDUCE)

        # First execution
        with c.guard("op_a"):
            pass
        c.assert_complete()

        # Reset and replay
        c.reset()
        with c.guard("op_a"):
            pass
        c.assert_complete()

    def test_reset_clears_verified_flag(self):
        c = CollectiveContract(step=0, rank=0)
        c.verify()  # passes in non-distributed mode
        assert c._verified is True

        c.reset()
        assert c._verified is False


# ---------------------------------------------------------------------------
# Test: validate_call_sites against actual desloc_engine.py
# ---------------------------------------------------------------------------

class TestValidateCallSitesIntegration:
    """Verify that the actual desloc_engine.py contains all expected guard sites.

    This is an AST-level call-chain verification: if a new collective is
    added to build_step_contract but the corresponding guard() call is
    missing from the engine loop, this test catches it.
    """

    def test_desloc_engine_has_all_guard_sites(self):
        import os
        # Guard sites are spread across the call chain:
        # engine → finalize_model_grads → clip_grads
        source_files = [
            os.path.join(os.path.dirname(__file__), "..", "..", "deepspeed", "runtime", "desloc_engine.py"),
            os.path.join(os.path.dirname(__file__), "..", "..", "deepspeed", "core", "distributed", "finalize_model_grads.py"),
            os.path.join(os.path.dirname(__file__), "..", "..", "deepspeed", "core", "optimizer", "clip_grads.py"),
        ]
        combined_source = ""
        for p in source_files:
            if not os.path.exists(p):
                pytest.skip(f"Source file not found: {p}")
            with open(p, "r") as f:
                combined_source += f.read() + "\n"

        # Check all 6 guards across the combined source
        results = validate_call_sites(combined_source)
        missing = [k for k, v in results.items() if not v]
        assert not missing, f"Call chain missing guard() sites for: {missing}"

    def test_desloc_engine_has_optimizer_guard_sites(self):
        import os
        engine_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "deepspeed", "runtime", "desloc_engine.py"
        )
        if not os.path.exists(engine_path):
            pytest.skip("desloc_engine.py not found at expected path")

        with open(engine_path, "r") as f:
            source = f.read()

        # Check the 2 optimizer guards (dist_optimizer path) — these are in engine
        opt_guards = frozenset({"prepare_grads_rs", "param_sync"})
        results = validate_call_sites(source, expected_guards=opt_guards)
        missing = [k for k, v in results.items() if not v]
        assert not missing, f"desloc_engine.py missing optimizer guard() sites for: {missing}"
