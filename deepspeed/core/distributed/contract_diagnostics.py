# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team / Neuron_SP
"""CollectiveContract diagnostics — runtime analysis and violation reporting.

Companion module to ``collective_contract.py`` providing helpers that are
useful for debugging NCCL hang root causes but too verbose for the core
contract module.

Public API
----------
  format_violation_report  — human-readable multi-line report for a mismatch.
  diff_sequences           — side-by-side diff of two planned collective lists.
  validate_call_sites      — static-ish check that a source file contains the
                             expected ``contract.guard(...)`` call sites.
  StepTrace                — lightweight record of one step's contract lifecycle.
  StepTraceLog             — bounded ring-buffer of recent StepTraces.

Fixes: #589 (diagnostic tooling for collective symmetry enforcement).
"""

from __future__ import annotations

import logging
import textwrap
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# format_violation_report
# ---------------------------------------------------------------------------

def format_violation_report(
    step: int,
    rank: int,
    local_seq: Sequence[str],
    remote_seq: Sequence[str],
    remote_rank: int = -1,
) -> str:
    """Return a human-readable multi-line report for a contract violation.

    The report includes a side-by-side diff of the planned collective
    sequences so that the divergence point is immediately visible.

    Args:
        step:        Training step where the mismatch was detected.
        rank:        Local rank id.
        local_seq:   This rank's planned collective names.
        remote_seq:  The remote rank's planned collective names.
        remote_rank: Remote rank id (default -1 = unknown).

    Returns:
        A multi-line string suitable for logging or exception detail.
    """
    lines: List[str] = [
        f"═══ CollectiveContract Violation Report ═══",
        f"  Step:        {step}",
        f"  Local rank:  {rank}",
        f"  Remote rank: {remote_rank if remote_rank >= 0 else 'unknown'}",
        f"  Local ops:   {len(local_seq)}",
        f"  Remote ops:  {len(remote_seq)}",
        f"",
    ]

    diff = diff_sequences(list(local_seq), list(remote_seq))
    lines.append("  Sequence diff (← local | remote →):")
    for idx, (l_name, r_name, match) in enumerate(diff, 1):
        marker = "✓" if match else "✗"
        lines.append(f"    [{marker}] {idx:>3d}: {l_name:<35s} | {r_name}")
    lines.append(f"═══ End of Report ═══")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# diff_sequences
# ---------------------------------------------------------------------------

def diff_sequences(
    local: List[str],
    remote: List[str],
) -> List[Tuple[str, str, bool]]:
    """Produce a side-by-side diff of two collective-name sequences.

    Pads the shorter list with ``"<missing>"`` entries so both columns
    align.  Each tuple is ``(local_name, remote_name, match: bool)``.

    Args:
        local:  Local rank's planned names.
        remote: Remote rank's planned names.

    Returns:
        List of (local_name, remote_name, match) tuples.
    """
    max_len = max(len(local), len(remote))
    result: List[Tuple[str, str, bool]] = []
    for i in range(max_len):
        l_name = local[i] if i < len(local) else "<missing>"
        r_name = remote[i] if i < len(remote) else "<missing>"
        result.append((l_name, r_name, l_name == r_name))
    return result


# ---------------------------------------------------------------------------
# validate_call_sites
# ---------------------------------------------------------------------------

_EXPECTED_GUARD_NAMES = frozenset({
    "nan_flag_allreduce",
    "finalize_model_grads",
    "clip_grad_norm_allreduce",
    "skip_flag_allreduce",
    "prepare_grads_rs",
    "param_sync",
})


def validate_call_sites(
    source_text: str,
    expected_guards: Optional[frozenset] = None,
) -> Dict[str, bool]:
    """Check whether *source_text* contains ``guard("name")`` for each expected guard.

    This is a lightweight static analysis helper (grep-level, not AST-level)
    for CI checks or integration tests.  Matches any call of the form
    ``xxx.guard("name")`` or ``xxx.guard('name')`` or the bare string
    ``"name"`` after ``guard(`` — the variable holding the contract may
    differ across files (``_contract``, ``collective_contract``, etc.).

    Args:
        source_text:     Python source code to scan.
        expected_guards: Set of guard names to look for.  Defaults to the
                         standard DES-LOC training-loop guards.

    Returns:
        Dict mapping guard name → found (bool).
    """
    if expected_guards is None:
        expected_guards = _EXPECTED_GUARD_NAMES

    results: Dict[str, bool] = {}
    for name in sorted(expected_guards):
        # Match .guard("name") or .guard('name') with any receiver
        results[name] = (
            f'guard("{name}")' in source_text
            or f"guard('{name}')" in source_text
        )
    return results


# ---------------------------------------------------------------------------
# StepTrace / StepTraceLog
# ---------------------------------------------------------------------------

@dataclass
class StepTrace:
    """Lightweight record of one training step's contract lifecycle.

    Attributes:
        step:           Training step (0-indexed).
        rank:           Local rank id.
        planned_count:  Number of planned collectives.
        executed_count: Number of actually executed collectives.
        verified:       Whether verify() was called and passed.
        complete:       Whether assert_complete() would pass.
        is_Kx:          Kx flag for this step.
        is_Ku:          Ku flag for this step.
        is_Kv:          Kv flag for this step.
    """
    step: int
    rank: int
    planned_count: int = 0
    executed_count: int = 0
    verified: bool = False
    complete: bool = False
    is_Kx: bool = False
    is_Ku: bool = False
    is_Kv: bool = False

    @classmethod
    def from_contract(cls, contract: Any) -> "StepTrace":
        """Build a StepTrace from a CollectiveContract instance."""
        return cls(
            step=contract.step,
            rank=contract.rank,
            planned_count=contract.planned_count,
            executed_count=len(contract._executed),
            verified=contract._verified,
            complete=contract.all_executed,
            is_Kx=contract.is_Kx,
            is_Ku=contract.is_Ku,
            is_Kv=contract.is_Kv,
        )


class StepTraceLog:
    """Bounded ring-buffer of recent StepTraces for post-mortem analysis.

    Keeps the last *maxlen* traces so that when a hang occurs the prior
    step history is available without unbounded memory growth.

    Args:
        maxlen: Maximum number of traces to retain (default 64).
    """

    def __init__(self, maxlen: int = 64) -> None:
        self._traces: Deque[StepTrace] = deque(maxlen=maxlen)

    def record(self, trace: StepTrace) -> None:
        """Append a trace to the log."""
        self._traces.append(trace)

    @property
    def traces(self) -> List[StepTrace]:
        """Return all traces in chronological order."""
        return list(self._traces)

    @property
    def last(self) -> Optional[StepTrace]:
        """Return the most recent trace, or None if empty."""
        return self._traces[-1] if self._traces else None

    def incomplete_steps(self) -> List[StepTrace]:
        """Return all traces where the contract was not fully executed."""
        return [t for t in self._traces if not t.complete]

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict for logging."""
        total = len(self._traces)
        incomplete = len(self.incomplete_steps())
        return {
            "total_steps": total,
            "incomplete_steps": incomplete,
            "oldest_step": self._traces[0].step if self._traces else None,
            "newest_step": self._traces[-1].step if self._traces else None,
        }


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "format_violation_report",
    "diff_sequences",
    "validate_call_sites",
    "StepTrace",
    "StepTraceLog",
]
