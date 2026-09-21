"""
Integration tests for issue #591 gate validation.

Tests the validate_gate.py script against synthetic log content
to verify all 5 gate criteria are correctly checked.

Test matrix:
  - All 5 criteria pass on a well-formed log
  - Criterion 1 fails: not enough steps
  - Criterion 1 fails: NCCL error in log
  - Criterion 2 fails: loss increases
  - Criterion 3 fails: missing rank
  - Criterion 3 fails: unequal step counts across ranks
  - Criterion 4 fails: no GPU memory entries
  - Criterion 5 passes: file exists
  - JSON output format
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

# Import from scripts/ — add to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from validate_gate import (
    GateReport,
    check_gpu_memory_logged,
    check_log_committed,
    check_loss_decreasing,
    check_rank_step_counts,
    check_steps_completed,
    parse_log,
    validate_gate,
    StepRecord,
)


def _write_log(content: str) -> str:
    """Write log content to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".log", prefix="gate_test_")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


# ---------------------------------------------------------------------------
# Synthetic log generators
# ---------------------------------------------------------------------------

def _make_passing_log(steps: int = 100, ranks: int = 3) -> str:
    """Generate a log that passes all 5 gate criteria."""
    lines = []
    for step in range(1, steps + 1):
        loss = 11.0 - step * 0.05  # monotonically decreasing
        lines.append(
            f"step=  {step:4d} | loss={loss:.4f} | lr=3.00e-04 | "
            f"grad_norm=1.234 | tok/s=  1234 | step_ms=50.0 | "
            f"tokens_seen_M=0.10 | MFU=0.0100"
        )
        # Rank logs
        for rank in range(ranks):
            lines.append(
                f"rank={rank}: num_microbatches=1, step={step}"
            )
        # GPU memory logs
        lines.append(f"train/gpu0_mem_gb=12.5")
        lines.append(f"train/gpu1_mem_gb=14.2")
        lines.append(f"train/gpu2_mem_gb=30.1")
    return "\n".join(lines)


def _make_nccl_error_log() -> str:
    """Generate a log with an NCCL error at step 50."""
    lines = []
    for step in range(1, 51):
        loss = 11.0 - step * 0.05
        lines.append(f"step=  {step:4d} | loss={loss:.4f}")
        for rank in range(3):
            lines.append(f"rank={rank}: num_microbatches=1, step={step}")
    lines.append("NCCL error: ncclInternalError at collective AllReduce")
    return "\n".join(lines)


def _make_loss_increasing_log() -> str:
    """Generate a log where loss increases."""
    lines = []
    for step in range(1, 101):
        loss = 5.0 + step * 0.01  # increasing
        lines.append(f"step=  {step:4d} | loss={loss:.4f}")
        for rank in range(3):
            lines.append(f"rank={rank}: num_microbatches=1, step={step}")
        lines.append(f"train/gpu0_mem_gb=12.5")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestValidateGateAllPass:
    """All 5 criteria should pass on a well-formed log."""

    def test_all_pass(self):
        path = _write_log(_make_passing_log())
        try:
            report = validate_gate(path, target_steps=100)
            assert report.all_passed, report.summary()
            assert len(report.results) == 5
            for r in report.results:
                assert r.passed, f"Criterion failed: {r.name}: {r.detail}"
        finally:
            os.unlink(path)


class TestCriterion1StepsCompleted:
    """Criterion 1: N steps completed without crash."""

    def test_not_enough_steps(self):
        path = _write_log(_make_passing_log(steps=50))
        try:
            report = validate_gate(path, target_steps=100)
            assert not report.results[0].passed
            assert "50" in report.results[0].detail
        finally:
            os.unlink(path)

    def test_nccl_error(self):
        path = _write_log(_make_nccl_error_log())
        try:
            report = validate_gate(path, target_steps=100)
            assert not report.results[0].passed
            assert "NCCL" in report.results[0].detail or "error" in report.results[0].detail.lower()
        finally:
            os.unlink(path)


class TestCriterion2LossDecreasing:
    """Criterion 2: loss decreasing over the run."""

    def test_loss_increasing_fails(self):
        path = _write_log(_make_loss_increasing_log())
        try:
            report = validate_gate(path, target_steps=100)
            assert not report.results[1].passed
        finally:
            os.unlink(path)

    def test_loss_decreasing_passes(self):
        records = [
            StepRecord(step=1, loss=11.0),
            StepRecord(step=100, loss=6.0),
        ]
        result = check_loss_decreasing(records)
        assert result.passed

    def test_too_few_records(self):
        result = check_loss_decreasing([StepRecord(step=1, loss=11.0)])
        assert not result.passed


class TestCriterion3RankStepCounts:
    """Criterion 3: all ranks log the same step count."""

    def test_missing_rank(self):
        result = check_rank_step_counts({0: 100, 1: 100}, expected_ranks=3)
        assert not result.passed

    def test_unequal_counts(self):
        result = check_rank_step_counts({0: 100, 1: 100, 2: 50}, expected_ranks=3)
        assert not result.passed

    def test_all_equal(self):
        result = check_rank_step_counts({0: 100, 1: 100, 2: 100}, expected_ranks=3)
        assert result.passed


class TestCriterion4GPUMemory:
    """Criterion 4: GPU memory logged."""

    def test_no_memory_entries(self):
        log = "\n".join([f"step= {i} | loss={10-i*0.1:.4f}" for i in range(1, 11)])
        path = _write_log(log)
        try:
            records, _, _ = parse_log(path)
            result = check_gpu_memory_logged(records, path)
            assert not result.passed
        finally:
            os.unlink(path)

    def test_has_memory_entries(self):
        log = "step= 1 | loss=10.0\ntrain/gpu0_mem_gb=12.5\ntrain/gpu2_mem_gb=30.1\n"
        path = _write_log(log)
        try:
            records, _, _ = parse_log(path)
            result = check_gpu_memory_logged(records, path)
            assert result.passed
            assert "0" in result.detail and "2" in result.detail
        finally:
            os.unlink(path)


class TestCriterion5LogExists:
    """Criterion 5: log file exists."""

    def test_file_exists(self):
        path = _write_log("test")
        try:
            result = check_log_committed(path)
            assert result.passed
        finally:
            os.unlink(path)

    def test_file_missing(self):
        result = check_log_committed("/nonexistent/path.log")
        assert not result.passed


class TestGateReportJSON:
    """JSON serialisation of the gate report."""

    def test_json_output(self):
        path = _write_log(_make_passing_log(steps=10))
        try:
            report = validate_gate(path, target_steps=10)
            j = json.loads(report.to_json())
            assert "all_passed" in j
            assert "results" in j
            assert len(j["results"]) == 5
        finally:
            os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
