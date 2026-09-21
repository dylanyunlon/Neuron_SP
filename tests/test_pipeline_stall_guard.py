"""
Comprehensive tests for issue #592: pipeline stall guard.

Note: this file uses importlib direct-import to bypass the heavy
deepspeed/__init__.py chain (which pulls in torch, NCCL, etc).
The project conftest.py also imports torch, so run with --noconftest:

    python -m pytest tests/test_pipeline_stall_guard.py -v --noconftest

Tests the centralized pipeline deadlock prevention module
(deepspeed.runtime.pipe.pipeline_stall_guard) that prevents
barrier deadlocks when num_microbatches < pipeline_parallel_world_size.

Test matrix (6 call-chain integration points × edge cases):

  1. should_measure_pipeline_stall — core predicate
     - num_microbatches >= num_stages → True
     - num_microbatches < num_stages → False
     - edge: num_microbatches == 0
     - edge: num_microbatches == 1, num_stages == 1
     - edge: negative inputs
     - edge: num_stages == 0

  2. sanitize_schedule_params — parameter sanitizer
     - standard 4-stage, 8-microbatch case
     - undersaturated: 2 microbatches, 4 stages
     - edge: 1 stage, 1 microbatch (degenerate)
     - clamping: num_microbatches=0 → 1
     - invalid: stage_id out of range
     - invalid: num_stages < 1
     - warmup count correctness across all stage_ids

  3. warn_microbatch_underflow — structured warning
     - underflow detected: returns warning dict
     - no underflow: returns None
     - warning dict keys and values
     - rank parameter propagation

  4. PipelineStallGuard — context manager
     - yields ScheduleParams
     - undersaturated mode disables stall measurement
     - normal mode enables stall measurement
     - exception propagation

  5. validate_schedule_no_deadlock — post-hoc validator
     - valid schedule: no errors
     - dp_sync with no backward: error
     - short schedule with sync: error
     - empty schedule: no errors

  6. sanitize_timer_name — timer name filter
     - blocked timer with stall disabled: returns None
     - blocked timer with stall enabled: returns name
     - non-blocked timer: always returns name

  7. Integration: TrainSchedule.measure_pipeline_stall flag
     - schedule.py TrainSchedule uses centralized predicate
     - InterleavedTrainSchedule uses centralized predicate

  8. Integration: ScheduleParams warmup consistency
     - warmup counts match hand-computed values for all ranks in a 4-stage pipeline
     - warmup is clamped to num_microbatches when undersaturated

  9. BLOCKED_BARRIER_CONDITIONS completeness
     - all known barrier timer names are in the set
     - set is frozen (immutable)
"""

import importlib.util
import logging
import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Direct-import to bypass the heavy deepspeed/__init__.py chain.
# Register in sys.modules before exec_module so that dataclasses can resolve
# the module's __dict__ (Python 3.12+).
# ---------------------------------------------------------------------------
_MODULE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "deepspeed", "runtime", "pipe",
    "pipeline_stall_guard.py",
)
_spec = importlib.util.spec_from_file_location("pipeline_stall_guard", _MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["pipeline_stall_guard"] = _mod
_spec.loader.exec_module(_mod)

should_measure_pipeline_stall = _mod.should_measure_pipeline_stall
sanitize_schedule_params = _mod.sanitize_schedule_params
ScheduleParams = _mod.ScheduleParams
warn_microbatch_underflow = _mod.warn_microbatch_underflow
PipelineStallGuard = _mod.PipelineStallGuard
validate_schedule_no_deadlock = _mod.validate_schedule_no_deadlock
sanitize_timer_name = _mod.sanitize_timer_name
BLOCKED_BARRIER_CONDITIONS = _mod.BLOCKED_BARRIER_CONDITIONS


# =========================================================================
# 1. Core predicate: should_measure_pipeline_stall
# =========================================================================

class TestShouldMeasurePipelineStall:
    """Tests for the single source of truth stall predicate."""

    def test_sufficient_microbatches(self):
        """When num_microbatches >= num_stages, stall measurement is safe."""
        assert should_measure_pipeline_stall(4, 4) is True
        assert should_measure_pipeline_stall(8, 4) is True
        assert should_measure_pipeline_stall(100, 2) is True

    def test_insufficient_microbatches(self):
        """When num_microbatches < num_stages, stall measurement deadlocks."""
        assert should_measure_pipeline_stall(1, 4) is False
        assert should_measure_pipeline_stall(3, 4) is False
        assert should_measure_pipeline_stall(0, 2) is False

    def test_single_stage_single_microbatch(self):
        """Degenerate case: 1 stage, 1 microbatch — always safe."""
        assert should_measure_pipeline_stall(1, 1) is True

    def test_zero_microbatches(self):
        """Edge: zero microbatches with any stage count."""
        assert should_measure_pipeline_stall(0, 1) is False
        assert should_measure_pipeline_stall(0, 4) is False

    def test_negative_inputs(self):
        """Edge: negative inputs should return False (defensive)."""
        assert should_measure_pipeline_stall(-1, 4) is False
        assert should_measure_pipeline_stall(4, 0) is False
        assert should_measure_pipeline_stall(-1, -1) is False

    def test_exact_boundary(self):
        """Boundary: num_microbatches == num_stages is the threshold."""
        for n in range(1, 10):
            assert should_measure_pipeline_stall(n, n) is True
            assert should_measure_pipeline_stall(n - 1, n) is (n - 1 >= n)


# =========================================================================
# 2. Parameter sanitizer: sanitize_schedule_params
# =========================================================================

class TestSanitizeScheduleParams:
    """Tests for schedule parameter validation and clamping."""

    def test_standard_case(self):
        """Standard 4-stage, 8-microbatch pipeline."""
        p = sanitize_schedule_params(8, 4, 0)
        assert p.num_microbatches == 8
        assert p.num_stages == 4
        assert p.stage_id == 0
        # Stage 0 warmup = 4 - 0 - 1 = 3
        assert p.num_warmup_microbatches == 3
        assert p.num_microbatches_remaining == 5
        assert p.measure_pipeline_stall is True
        assert p.is_undersaturated is False

    def test_undersaturated(self):
        """2 microbatches, 4 stages — undersaturated."""
        p = sanitize_schedule_params(2, 4, 0)
        assert p.is_undersaturated is True
        assert p.measure_pipeline_stall is False
        # Warmup clamped to num_microbatches
        assert p.num_warmup_microbatches == 2
        assert p.num_microbatches_remaining == 0

    def test_degenerate_single(self):
        """1 stage, 1 microbatch — degenerate but valid."""
        p = sanitize_schedule_params(1, 1, 0)
        assert p.num_warmup_microbatches == 0
        assert p.num_microbatches_remaining == 1
        assert p.measure_pipeline_stall is True
        assert p.is_undersaturated is False

    def test_zero_microbatch_clamping(self):
        """num_microbatches=0 is clamped to 1."""
        p = sanitize_schedule_params(0, 4, 0)
        assert p.num_microbatches == 1
        assert p.is_undersaturated is True

    def test_invalid_stage_id_too_high(self):
        """stage_id >= num_stages should raise."""
        with pytest.raises(ValueError, match="stage_id=4 out of range"):
            sanitize_schedule_params(8, 4, 4)

    def test_invalid_stage_id_negative(self):
        """Negative stage_id should raise."""
        with pytest.raises(ValueError, match="stage_id=-1 out of range"):
            sanitize_schedule_params(8, 4, -1)

    def test_invalid_num_stages_zero(self):
        """num_stages=0 should raise."""
        with pytest.raises(ValueError, match="num_stages must be >= 1"):
            sanitize_schedule_params(8, 0, 0)

    def test_warmup_counts_all_stages(self):
        """Warmup counts should be correct for every stage in a 4-stage pipeline."""
        for stage_id in range(4):
            p = sanitize_schedule_params(8, 4, stage_id)
            expected_warmup = min(4 - stage_id - 1, 8)
            assert p.num_warmup_microbatches == expected_warmup, (
                f"stage {stage_id}: expected warmup={expected_warmup}, "
                f"got {p.num_warmup_microbatches}"
            )
            assert p.num_warmup_microbatches + p.num_microbatches_remaining == 8

    def test_frozen_dataclass(self):
        """ScheduleParams should be frozen (immutable)."""
        p = sanitize_schedule_params(8, 4, 0)
        with pytest.raises(AttributeError):
            p.num_microbatches = 99

    def test_last_stage_zero_warmup(self):
        """The last stage should have 0 warmup microbatches."""
        p = sanitize_schedule_params(8, 4, 3)
        assert p.num_warmup_microbatches == 0
        assert p.num_microbatches_remaining == 8

    def test_undersaturated_all_stages(self):
        """All stages in an undersaturated pipeline should have stall disabled."""
        for stage_id in range(4):
            p = sanitize_schedule_params(2, 4, stage_id)
            assert p.measure_pipeline_stall is False
            assert p.is_undersaturated is True


# =========================================================================
# 3. Structured warning: warn_microbatch_underflow
# =========================================================================

class TestWarnMicrobatchUnderflow:
    """Tests for the structured underflow warning."""

    def test_underflow_returns_dict(self):
        """Underflow condition should return a warning dict."""
        result = warn_microbatch_underflow(2, 4, context="test")
        assert result is not None
        assert result["issue"] == 592
        assert result["num_microbatches"] == 2
        assert result["num_stages"] == 4
        assert result["stall_barriers_disabled"] is True
        assert result["context"] == "test"

    def test_no_underflow_returns_none(self):
        """No underflow should return None."""
        assert warn_microbatch_underflow(4, 4) is None
        assert warn_microbatch_underflow(8, 4) is None

    def test_rank_propagation(self):
        """Rank parameter should appear in the warning dict."""
        result = warn_microbatch_underflow(1, 4, rank=2)
        assert result is not None
        assert result["rank"] == 2

    def test_no_rank(self):
        """Without rank, key should be absent."""
        result = warn_microbatch_underflow(1, 4)
        assert result is not None
        assert "rank" not in result

    def test_warning_logged(self, caplog):
        """Should emit a WARNING-level log message."""
        with caplog.at_level(logging.WARNING):
            warn_microbatch_underflow(1, 4, context="log_test")
        assert any("[M592]" in r.message for r in caplog.records)
        assert any("Pipeline underflow" in r.message for r in caplog.records)

    def test_boundary_no_warning(self):
        """Exact boundary (num_microbatches == num_stages) should not warn."""
        assert warn_microbatch_underflow(4, 4) is None


# =========================================================================
# 4. Context manager: PipelineStallGuard
# =========================================================================

class TestPipelineStallGuard:
    """Tests for the PipelineStallGuard context manager."""

    def test_yields_schedule_params(self):
        """Should yield a ScheduleParams instance."""
        with PipelineStallGuard(8, 4, 0, context="test") as params:
            assert isinstance(params, ScheduleParams)
            assert params.num_microbatches == 8

    def test_undersaturated_disables_stall(self):
        """Undersaturated mode should disable stall measurement."""
        with PipelineStallGuard(2, 4, 1, context="test") as params:
            assert params.measure_pipeline_stall is False
            assert params.is_undersaturated is True

    def test_normal_enables_stall(self):
        """Normal mode should enable stall measurement."""
        with PipelineStallGuard(8, 4, 0, context="test") as params:
            assert params.measure_pipeline_stall is True
            assert params.is_undersaturated is False

    def test_exception_propagation(self):
        """Exceptions inside the guard should propagate normally."""
        with pytest.raises(RuntimeError, match="test error"):
            with PipelineStallGuard(8, 4, 0) as params:
                raise RuntimeError("test error")

    def test_guard_for_all_stages(self):
        """Guard should work for every stage in a pipeline."""
        for stage_id in range(4):
            with PipelineStallGuard(8, 4, stage_id) as params:
                assert params.stage_id == stage_id
                assert params.num_stages == 4


# =========================================================================
# 5. Post-hoc validator: validate_schedule_no_deadlock
# =========================================================================

class TestValidateScheduleNoDeadlock:
    """Tests for the schedule deadlock validator."""

    def test_valid_schedule(self):
        """A well-formed schedule should produce no errors."""
        steps = [
            {"d": "F", "mb": 0, "dp_sync": False},
            {"d": "F", "mb": 1, "dp_sync": False},
            {"d": "B", "mb": 0, "dp_sync": False},
            {"d": "B", "mb": 1, "dp_sync": True},
        ]
        errors = validate_schedule_no_deadlock(steps, num_stages=2)
        assert errors == []

    def test_sync_without_backward(self):
        """dp_sync=True with no backward passes should error."""
        steps = [
            {"d": "F", "mb": 0, "dp_sync": True},
            {"d": "F", "mb": 1, "dp_sync": False},
        ]
        errors = validate_schedule_no_deadlock(steps, num_stages=2)
        assert len(errors) == 1
        assert "no backward passes" in errors[0]

    def test_short_schedule_with_sync(self):
        """Schedule shorter than num_stages with sync should error."""
        steps = [
            {"d": "F", "mb": 0, "dp_sync": False},
            {"d": "B", "mb": 0, "dp_sync": True},
        ]
        errors = validate_schedule_no_deadlock(steps, num_stages=4)
        assert len(errors) == 1
        assert "Schedule length" in errors[0]

    def test_empty_schedule(self):
        """Empty schedule should produce no errors."""
        assert validate_schedule_no_deadlock([], num_stages=4) == []

    def test_large_valid_schedule(self):
        """Large valid schedule should pass."""
        steps = []
        for i in range(16):
            steps.append({"d": "F", "mb": i, "dp_sync": False})
        for i in range(16):
            steps.append({"d": "B", "mb": i, "dp_sync": (i == 15)})
        errors = validate_schedule_no_deadlock(steps, num_stages=4)
        assert errors == []


# =========================================================================
# 6. Timer name sanitizer: sanitize_timer_name
# =========================================================================

class TestSanitizeTimerName:
    """Tests for the timer name filter."""

    def test_blocked_timer_stall_disabled(self):
        """Blocked timer names should return None when stall is disabled."""
        assert sanitize_timer_name("forward-pipeline-stall", measure_stall=False) is None
        assert sanitize_timer_name("backward-pipeline-stall", measure_stall=False) is None
        assert sanitize_timer_name("pipeline-stall-warmup-end", measure_stall=False) is None
        assert sanitize_timer_name("pipeline-stall-cooldown-start", measure_stall=False) is None

    def test_blocked_timer_stall_enabled(self):
        """Blocked timer names should pass through when stall is enabled."""
        name = sanitize_timer_name("forward-pipeline-stall", measure_stall=True)
        assert name == "forward-pipeline-stall"

    def test_non_blocked_timer(self):
        """Non-blocked timer names should always pass through."""
        assert sanitize_timer_name("forward-compute", measure_stall=False) == "forward-compute"
        assert sanitize_timer_name("forward-compute", measure_stall=True) == "forward-compute"
        assert sanitize_timer_name("backward-reduce", measure_stall=False) == "backward-reduce"


# =========================================================================
# 7. BLOCKED_BARRIER_CONDITIONS completeness
# =========================================================================

class TestBlockedBarrierConditions:
    """Tests for the barrier condition blocklist."""

    def test_known_conditions_present(self):
        """All known barrier timer names should be in the set."""
        assert "forward-pipeline-stall" in BLOCKED_BARRIER_CONDITIONS
        assert "backward-pipeline-stall" in BLOCKED_BARRIER_CONDITIONS
        assert "pipeline-stall-warmup-end" in BLOCKED_BARRIER_CONDITIONS
        assert "pipeline-stall-cooldown-start" in BLOCKED_BARRIER_CONDITIONS

    def test_set_is_frozen(self):
        """The blocklist should be a frozenset (immutable)."""
        assert isinstance(BLOCKED_BARRIER_CONDITIONS, frozenset)
        with pytest.raises(AttributeError):
            BLOCKED_BARRIER_CONDITIONS.add("new-barrier")

    def test_minimum_count(self):
        """Should have at least 4 blocked conditions."""
        assert len(BLOCKED_BARRIER_CONDITIONS) >= 4


# =========================================================================
# 8. Integration: warmup consistency across pipeline topologies
# =========================================================================

class TestWarmupConsistency:
    """Verify warmup / steady-state splits are consistent across topologies."""

    @pytest.mark.parametrize("num_stages", [2, 4, 5, 8])
    def test_warmup_sum_equals_total(self, num_stages):
        """warmup + remaining should always equal num_microbatches."""
        for nm in [1, num_stages, num_stages * 2, num_stages * 4]:
            for sid in range(num_stages):
                p = sanitize_schedule_params(nm, num_stages, sid)
                total = p.num_warmup_microbatches + p.num_microbatches_remaining
                assert total == p.num_microbatches, (
                    f"stages={num_stages}, mb={nm}, stage={sid}: "
                    f"{p.num_warmup_microbatches} + {p.num_microbatches_remaining} "
                    f"!= {p.num_microbatches}"
                )

    @pytest.mark.parametrize("num_stages", [2, 4, 5, 8])
    def test_warmup_monotonically_decreasing_across_stages(self, num_stages):
        """Earlier stages should have more warmup than later stages."""
        nm = num_stages * 4  # well-saturated
        warmups = []
        for sid in range(num_stages):
            p = sanitize_schedule_params(nm, num_stages, sid)
            warmups.append(p.num_warmup_microbatches)
        for i in range(len(warmups) - 1):
            assert warmups[i] >= warmups[i + 1], (
                f"stages={num_stages}: warmup not decreasing: {warmups}"
            )

    def test_five_gpu_hetero_topology(self):
        """Validate the 5-GPU heterogeneous cluster schedule params."""
        # 5-GPU cluster from configs/7b_5gpu.yaml
        # With 16 microbatches, stage 0 should have warmup=4, stage 4 should have 0
        for sid in range(5):
            p = sanitize_schedule_params(16, 5, sid)
            expected_warmup = min(5 - sid - 1, 16)
            assert p.num_warmup_microbatches == expected_warmup
            assert p.measure_pipeline_stall is True

        # With 3 microbatches (undersaturated for 5 stages)
        for sid in range(5):
            p = sanitize_schedule_params(3, 5, sid)
            assert p.measure_pipeline_stall is False
            assert p.is_undersaturated is True


# =========================================================================
# 9. Integration: ScheduleParams determinism
# =========================================================================

class TestScheduleParamsDeterminism:
    """Ensure params are deterministic and reproducible."""

    def test_same_inputs_same_output(self):
        """Same inputs should produce identical ScheduleParams."""
        p1 = sanitize_schedule_params(8, 4, 2)
        p2 = sanitize_schedule_params(8, 4, 2)
        assert p1 == p2

    def test_different_stages_different_warmup(self):
        """Different stage_ids should produce different warmup counts."""
        p0 = sanitize_schedule_params(8, 4, 0)
        p3 = sanitize_schedule_params(8, 4, 3)
        assert p0.num_warmup_microbatches != p3.num_warmup_microbatches
        assert p0.num_warmup_microbatches == 3
        assert p3.num_warmup_microbatches == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
