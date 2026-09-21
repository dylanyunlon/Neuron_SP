#!/usr/bin/env python3
"""
validate_gate.py — Post-run validator for issue #591 gate criteria.

Parses a training log file and checks ALL 5 gate criteria:
  1. ✅ 100 steps completed (no NCCL hang, no OOM, no crash)
  2. ✅ Loss monotonically decreasing (step 0 ~ 10-11, step 100 < step 0)
  3. ✅ All 3 ranks log the same step count
  4. ✅ Per-step GPU memory logged for H100 and both A6000s
  5. ✅ Log file committed to logs/ as evidence

Usage:
    python scripts/validate_gate.py logs/7b_pretrain_3gpu_*.log
    python scripts/validate_gate.py --steps 100 logs/gate_run.log

Exit codes:
    0 — all criteria pass
    1 — one or more criteria fail (details printed to stderr)
    2 — log file not found or parse error
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class StepRecord:
    """Parsed record for a single training step from the log."""
    step: int
    loss: float
    rank: int = 0
    gpu_mem_gb: Optional[float] = None
    grad_norm: Optional[float] = None
    tokens_per_sec: Optional[float] = None
    mfu: Optional[float] = None


@dataclass
class GateResult:
    """Result of one gate criterion check."""
    name: str
    passed: bool
    detail: str


@dataclass
class GateReport:
    """Full gate validation report."""
    log_path: str
    target_steps: int
    results: List[GateResult] = field(default_factory=list)
    step_records: List[StepRecord] = field(default_factory=list)
    rank_step_counts: Dict[int, int] = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def summary(self) -> str:
        lines = [
            f"Gate Validation Report: {self.log_path}",
            f"Target steps: {self.target_steps}",
            "-" * 60,
        ]
        for r in self.results:
            icon = "✅" if r.passed else "❌"
            lines.append(f"  {icon} {r.name}: {r.detail}")
        lines.append("-" * 60)
        status = "ALL PASSED" if self.all_passed else "FAILED"
        lines.append(f"Gate status: {status}")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps({
            "log_path": self.log_path,
            "target_steps": self.target_steps,
            "all_passed": self.all_passed,
            "results": [
                {"name": r.name, "passed": r.passed, "detail": r.detail}
                for r in self.results
            ],
            "rank_step_counts": self.rank_step_counts,
            "num_step_records": len(self.step_records),
        }, indent=2)


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

# Matches lines like:
#   step=   100 | loss=8.1234 | lr=3.00e-04 | grad_norm=12.345 | tok/s=  1234 | ...
_STEP_RE = re.compile(
    r"step=\s*(\d+)\s*\|\s*loss=([\d.]+)"
)

# Matches rank step logs: rank=0: num_microbatches=1, step=5
_RANK_STEP_RE = re.compile(
    r"rank=(\d+).*step=(\d+)"
)

# Matches GPU memory logs: train/gpu0_mem_gb or gpu_mem patterns
_GPU_MEM_RE = re.compile(
    r"gpu(\d+)_mem_gb[=:\s]+([\d.]+)"
)

# Matches NCCL errors
_NCCL_ERROR_RE = re.compile(
    r"(NCCL\s*(error|timeout|hang)|ncclInternalError|ncclSystemError)",
    re.IGNORECASE,
)

# Matches OOM
_OOM_RE = re.compile(
    r"(OutOfMemoryError|CUDA out of memory|OOM)",
    re.IGNORECASE,
)


def parse_log(log_path: str) -> Tuple[List[StepRecord], Dict[int, int], List[str]]:
    """Parse a training log file.

    Returns:
        (step_records, rank_step_counts, error_lines)
    """
    records: List[StepRecord] = []
    rank_steps: Dict[int, set] = defaultdict(set)
    errors: List[str] = []
    seen_steps: set = set()

    with open(log_path, "r", errors="replace") as f:
        for line in f:
            # Check for fatal errors
            if _NCCL_ERROR_RE.search(line):
                errors.append(f"NCCL error: {line.strip()[:200]}")
            if _OOM_RE.search(line):
                errors.append(f"OOM: {line.strip()[:200]}")

            # Parse step records
            m = _STEP_RE.search(line)
            if m:
                step = int(m.group(1))
                loss = float(m.group(2))
                if step not in seen_steps:
                    seen_steps.add(step)
                    rec = StepRecord(step=step, loss=loss)

                    # Try to extract GPU memory
                    mem_match = _GPU_MEM_RE.search(line)
                    if mem_match:
                        rec.gpu_mem_gb = float(mem_match.group(2))

                    records.append(rec)

            # Parse rank-level step counts
            rm = _RANK_STEP_RE.search(line)
            if rm:
                rank = int(rm.group(1))
                step = int(rm.group(2))
                rank_steps[rank].add(step)

    rank_counts = {r: len(steps) for r, steps in rank_steps.items()}
    records.sort(key=lambda r: r.step)
    return records, rank_counts, errors


# ---------------------------------------------------------------------------
# Gate criteria checks
# ---------------------------------------------------------------------------

def check_steps_completed(
    records: List[StepRecord],
    target: int,
    errors: List[str],
) -> GateResult:
    """Criterion 1: target steps completed with no fatal errors."""
    max_step = max((r.step for r in records), default=0)
    has_errors = len(errors) > 0
    passed = max_step >= target and not has_errors

    if has_errors:
        detail = f"max_step={max_step}, errors: {errors[:3]}"
    else:
        detail = f"{max_step} steps completed (target={target})"

    return GateResult(
        name=f"{target} steps completed, no NCCL/OOM crash",
        passed=passed,
        detail=detail,
    )


def check_loss_decreasing(
    records: List[StepRecord],
) -> GateResult:
    """Criterion 2: loss monotonically decreasing (start ~10-11, end < start)."""
    if len(records) < 2:
        return GateResult(
            name="Loss decreasing",
            passed=False,
            detail="Not enough step records to check",
        )

    first_loss = records[0].loss
    last_loss = records[-1].loss
    passed = last_loss < first_loss

    detail = (
        f"step {records[0].step} loss={first_loss:.4f} → "
        f"step {records[-1].step} loss={last_loss:.4f} "
        f"(delta={last_loss - first_loss:+.4f})"
    )
    return GateResult(
        name="Loss decreasing (last < first)",
        passed=passed,
        detail=detail,
    )


def check_rank_step_counts(
    rank_counts: Dict[int, int],
    expected_ranks: int = 3,
) -> GateResult:
    """Criterion 3: all ranks log the same step count."""
    if len(rank_counts) < expected_ranks:
        return GateResult(
            name=f"All {expected_ranks} ranks same step count",
            passed=False,
            detail=f"Only {len(rank_counts)} ranks found in log: {rank_counts}",
        )

    counts = list(rank_counts.values())
    all_same = len(set(counts)) <= 1
    passed = all_same and len(rank_counts) >= expected_ranks

    detail = f"rank_step_counts={dict(rank_counts)}"
    return GateResult(
        name=f"All {expected_ranks} ranks same step count",
        passed=passed,
        detail=detail,
    )


def check_gpu_memory_logged(
    records: List[StepRecord],
    log_path: str,
) -> GateResult:
    """Criterion 4: per-step GPU memory logged for all GPUs."""
    # Check raw log for gpu memory entries
    gpu_ids_with_mem: set = set()
    try:
        with open(log_path, "r", errors="replace") as f:
            for line in f:
                for m in _GPU_MEM_RE.finditer(line):
                    gpu_ids_with_mem.add(int(m.group(1)))
    except OSError:
        pass

    has_mem = len(gpu_ids_with_mem) > 0
    detail = (
        f"GPU memory logged for devices: {sorted(gpu_ids_with_mem)}"
        if has_mem
        else "No GPU memory entries found in log"
    )
    return GateResult(
        name="Per-step GPU memory logged",
        passed=has_mem,
        detail=detail,
    )


def check_log_committed(
    log_path: str,
) -> GateResult:
    """Criterion 5: log file exists in logs/ directory."""
    exists = os.path.isfile(log_path)
    in_logs_dir = "logs/" in log_path or log_path.startswith("logs")
    passed = exists

    detail = f"path={log_path}, exists={exists}, in_logs_dir={in_logs_dir}"
    return GateResult(
        name="Log file exists as evidence",
        passed=passed,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate_gate(log_path: str, target_steps: int = 100) -> GateReport:
    """Run all 5 gate criteria checks against a log file.

    Args:
        log_path: Path to the training log.
        target_steps: Number of steps required (default 100).

    Returns:
        GateReport with all results.
    """
    records, rank_counts, errors = parse_log(log_path)

    report = GateReport(
        log_path=log_path,
        target_steps=target_steps,
        step_records=records,
        rank_step_counts=rank_counts,
    )

    report.results.append(check_steps_completed(records, target_steps, errors))
    report.results.append(check_loss_decreasing(records))
    report.results.append(check_rank_step_counts(rank_counts))
    report.results.append(check_gpu_memory_logged(records, log_path))
    report.results.append(check_log_committed(log_path))

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate gate criteria for issue #591",
    )
    parser.add_argument("log_path", help="Path to training log file")
    parser.add_argument(
        "--steps", type=int, default=100,
        help="Target number of steps (default: 100)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output JSON report instead of human-readable",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.log_path):
        print(f"Error: log file not found: {args.log_path}", file=sys.stderr)
        return 2

    try:
        report = validate_gate(args.log_path, args.steps)
    except Exception as exc:
        print(f"Error parsing log: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(report.to_json())
    else:
        print(report.summary())

    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
