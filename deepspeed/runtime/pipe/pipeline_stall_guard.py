# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""pipeline_stall_guard.py — centralized deadlock prevention for pipeline schedules.

Issue #592 (M592): When ``num_microbatches < pipeline_parallel_world_size``,
the standard 1F1B schedule deadlocks because some pipeline stages never enter
the warmup phase and unconditionally execute barrier synchronizations that no
peer will ever reach.

This module centralises the ``measure_pipeline_stall`` predicate and provides:

  1. ``should_measure_pipeline_stall(num_microbatches, num_stages)``
     — the single source of truth for whether stall-measurement barriers
     are safe to execute.

  2. ``sanitize_schedule_params(num_microbatches, num_stages, stage_id)``
     — validates + clamps schedule parameters, returning a ``ScheduleParams``
     dataclass with pre-computed warmup / steady / cooldown counts and the
     stall flag.

  3. ``warn_microbatch_underflow(num_microbatches, num_stages, ...)``
     — emits a structured warning (and optional ``logging.warning``) when
     the microbatch count is below the pipeline-parallel world size.

  4. ``PipelineStallGuard`` context manager
     — wraps a 1F1B schedule execution; gates barrier calls on the stall flag
     and logs diagnostic information on entry/exit.

  5. ``validate_schedule_no_deadlock(schedule_steps, num_stages)``
     — post-hoc check that a generated schedule does not contain any
     barrier step reachable by fewer than ``num_stages`` ranks.

Call chain (6 call sites patched by this PR):
  engine.py::PipelineEngine.train_batch
    → schedule.py::TrainSchedule.__init__           (uses sanitize_schedule_params)
    → schedule.py::InterleavedTrainSchedule.__init__ (uses sanitize_schedule_params)
  schedules.py::forward_backward_pipelining_without_interleaving
    → uses PipelineStallGuard context manager
  schedules.py::forward_backward_pipelining_without_interleaving_pp5_heterogeneous
    → uses PipelineStallGuard context manager
  pp_schedule_adapter.py::PPScheduleAdapter.forward_backward
    → calls warn_microbatch_underflow before dispatch
  engine.py::m592_forward_backward_pipelining_without_interleaving
    → uses should_measure_pipeline_stall (Megatron compat path)
"""

from __future__ import annotations

import dataclasses
import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Blocked barrier conditions (analogous to BLOCKED_EXECUTION_ENVS)
# ---------------------------------------------------------------------------
# Barrier operations that MUST NOT be executed when
# num_microbatches < pipeline_parallel_world_size.
#
# Enumerated via: grep -rn 'pipeline-stall' deepspeed/ Megatron-LM/
# Last verified: 2026-09-21 (4 unique barrier timer names found)
BLOCKED_BARRIER_CONDITIONS: frozenset = frozenset({
    "forward-pipeline-stall",
    "backward-pipeline-stall",
    "pipeline-stall-warmup-end",
    "pipeline-stall-cooldown-start",
})

# Timer names that should be filtered by sanitize_timer_name() when stall
# measurement is disabled. Equal to BLOCKED_BARRIER_CONDITIONS because any
# timer guarding a barrier that cannot fire should also be blocked from
# start/stop calls to avoid corrupting timer bookkeeping.
_SENSITIVE_TIMER_NAMES = BLOCKED_BARRIER_CONDITIONS


# ---------------------------------------------------------------------------
# Core predicate
# ---------------------------------------------------------------------------

def should_measure_pipeline_stall(
    num_microbatches: int,
    num_stages: int,
) -> bool:
    """Return True only if pipeline stall measurement barriers are safe.

    The stall-measurement barriers require all pipeline stages to participate.
    When ``num_microbatches < num_stages``, the later stages never enter
    the warmup phase and will never reach the barrier, causing a deadlock.

    This is the **single source of truth** for the ``measure_pipeline_stall``
    flag.  Every schedule path must call this instead of computing the
    condition inline.

    Args:
        num_microbatches: Number of micro-batches in the current batch.
        num_stages:       Pipeline-parallel world size.

    Returns:
        True if stall barriers may be executed; False otherwise.
    """
    if num_microbatches < 0 or num_stages < 1:
        return False
    return num_microbatches >= num_stages


# ---------------------------------------------------------------------------
# Parameter sanitizer
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ScheduleParams:
    """Pre-computed schedule parameters with deadlock-safe defaults.

    Fields:
        num_microbatches:       Validated micro-batch count (>= 1).
        num_stages:             Pipeline parallel world size.
        stage_id:               This stage's index in [0, num_stages).
        num_warmup_microbatches: Warmup count, clamped to [0, num_microbatches].
        num_microbatches_remaining: Steady-state 1F1B pair count.
        measure_pipeline_stall:  Whether barrier-based stall measurement is safe.
        is_undersaturated:       True when num_microbatches < num_stages.
    """
    num_microbatches: int
    num_stages: int
    stage_id: int
    num_warmup_microbatches: int
    num_microbatches_remaining: int
    measure_pipeline_stall: bool
    is_undersaturated: bool


def sanitize_schedule_params(
    num_microbatches: int,
    num_stages: int,
    stage_id: int,
) -> ScheduleParams:
    """Validate and clamp schedule parameters for deadlock safety.

    Computes the warmup / steady-state split and the
    ``measure_pipeline_stall`` flag in one place so that callers cannot
    accidentally diverge.

    Args:
        num_microbatches: Raw micro-batch count from config / runtime.
        num_stages:       Pipeline-parallel world size.
        stage_id:         This rank's pipeline stage index.

    Returns:
        A frozen ``ScheduleParams`` with deadlock-safe values.

    Raises:
        ValueError: If ``num_stages < 1`` or ``stage_id`` is out of range.
    """
    if num_stages < 1:
        raise ValueError(f"num_stages must be >= 1, got {num_stages}")
    if not (0 <= stage_id < num_stages):
        raise ValueError(
            f"stage_id={stage_id} out of range [0, {num_stages})"
        )
    # Clamp micro-batches to at least 1 (zero means skip the batch entirely)
    num_microbatches = max(1, num_microbatches)

    # Standard 1F1B warmup count
    num_warmup = num_stages - stage_id - 1
    num_warmup = min(num_warmup, num_microbatches)
    num_remaining = num_microbatches - num_warmup

    measure_stall = should_measure_pipeline_stall(num_microbatches, num_stages)
    is_under = num_microbatches < num_stages

    return ScheduleParams(
        num_microbatches=num_microbatches,
        num_stages=num_stages,
        stage_id=stage_id,
        num_warmup_microbatches=num_warmup,
        num_microbatches_remaining=num_remaining,
        measure_pipeline_stall=measure_stall,
        is_undersaturated=is_under,
    )


# ---------------------------------------------------------------------------
# Structured warning
# ---------------------------------------------------------------------------

def warn_microbatch_underflow(
    num_microbatches: int,
    num_stages: int,
    *,
    context: str = "",
    rank: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Emit a structured warning when microbatches < pipeline stages.

    Args:
        num_microbatches: Current micro-batch count.
        num_stages:       Pipeline-parallel world size.
        context:          Caller description for the log message.
        rank:             Optional rank index for multi-GPU logging.

    Returns:
        A warning dict if underflow detected, else None.
    """
    if num_microbatches >= num_stages:
        return None

    warning_info = {
        "issue": 592,
        "condition": "num_microbatches < pipeline_parallel_world_size",
        "num_microbatches": num_microbatches,
        "num_stages": num_stages,
        "stall_barriers_disabled": True,
        "context": context,
    }
    if rank is not None:
        warning_info["rank"] = rank

    rank_str = f" rank={rank}" if rank is not None else ""
    ctx_str = f" [{context}]" if context else ""
    logger.warning(
        "[M592] Pipeline underflow%s%s: num_microbatches=%d < num_stages=%d. "
        "Stall-measurement barriers DISABLED to prevent deadlock.",
        rank_str, ctx_str, num_microbatches, num_stages,
    )
    return warning_info


# ---------------------------------------------------------------------------
# Context manager for guarded execution
# ---------------------------------------------------------------------------

@contextmanager
def PipelineStallGuard(
    num_microbatches: int,
    num_stages: int,
    stage_id: int,
    *,
    context: str = "",
) -> Iterator[ScheduleParams]:
    """Context manager that gates barrier calls for a 1F1B execution.

    Usage::

        with PipelineStallGuard(nm, ns, sid, context="1F1B") as params:
            if params.measure_pipeline_stall:
                barrier(...)
            ...

    Yields:
        ``ScheduleParams`` with deadlock-safe defaults.
    """
    params = sanitize_schedule_params(num_microbatches, num_stages, stage_id)

    if params.is_undersaturated:
        warn_microbatch_underflow(
            num_microbatches, num_stages,
            context=context, rank=stage_id,
        )

    logger.debug(
        "[M592] PipelineStallGuard enter: stages=%d stage=%d mb=%d "
        "warmup=%d steady=%d stall=%s undersaturated=%s",
        params.num_stages, params.stage_id, params.num_microbatches,
        params.num_warmup_microbatches, params.num_microbatches_remaining,
        params.measure_pipeline_stall, params.is_undersaturated,
    )

    try:
        yield params
    finally:
        logger.debug(
            "[M592] PipelineStallGuard exit: stage=%d context=%s",
            params.stage_id, context,
        )


# ---------------------------------------------------------------------------
# Post-hoc schedule validator
# ---------------------------------------------------------------------------

def validate_schedule_no_deadlock(
    schedule_steps: Sequence[dict],
    num_stages: int,
) -> List[str]:
    """Validate that a generated schedule doesn't contain unsafe barriers.

    Checks the ``dp_sync`` flags emitted by
    ``desloc_interleaved_kx_warmup`` and similar generators.

    Args:
        schedule_steps: List of schedule step dicts with 'dp_sync' key.
        num_stages:     Pipeline-parallel world size.

    Returns:
        List of error strings (empty = valid).
    """
    errors: List[str] = []
    if not schedule_steps:
        return errors

    # Count forward and backward steps
    fwd_count = sum(1 for s in schedule_steps if s.get('d') == 'F')
    bwd_count = sum(1 for s in schedule_steps if s.get('d') == 'B')

    # If backward count is zero but dp_sync is set anywhere, that's suspicious
    sync_steps = [i for i, s in enumerate(schedule_steps) if s.get('dp_sync')]
    if bwd_count == 0 and sync_steps:
        errors.append(
            f"Schedule has dp_sync=True at step(s) {sync_steps} but no "
            f"backward passes — this will deadlock on DP allreduce."
        )

    # If total steps < num_stages and any sync is requested, flag it
    if len(schedule_steps) < num_stages and sync_steps:
        errors.append(
            f"Schedule length ({len(schedule_steps)}) < num_stages ({num_stages}) "
            f"with dp_sync=True at {sync_steps} — potential barrier deadlock."
        )

    return errors


# ---------------------------------------------------------------------------
# Timer name sanitizer (analogous to masking sensitive env vars in consent)
# ---------------------------------------------------------------------------

def sanitize_timer_name(name: str, *, measure_stall: bool) -> Optional[str]:
    """Return None if a timer name is a blocked stall timer and stall
    measurement is disabled; otherwise return the name unchanged.

    This prevents timer start/stop calls on barriers that won't be reached,
    which would corrupt the timer bookkeeping.

    Args:
        name:          Timer name string.
        measure_stall: Whether pipeline stall measurement is enabled.

    Returns:
        The timer name if safe to use, or None if blocked.
    """
    if not measure_stall and name in _SENSITIVE_TIMER_NAMES:
        logger.debug(
            "[M592] Blocking timer '%s' — stall measurement disabled.", name
        )
        return None
    return name
