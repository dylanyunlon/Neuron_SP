# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team / Neuron_SP
"""CollectiveContract — enforce NCCL collective symmetry across all ranks.

Fixes: #51, #157, #197, #589.

Root cause
----------
Every NCCL hang in the DES-LOC training loop shares one root cause:
rank A enters a collective that rank B skips.  NCCL requires ALL ranks
in a process group to call the same collective in the same order.
Any divergence results in a 30-minute timeout → hang.

The training loop has several "conditional collectives" — NCCL calls inside
``if`` branches that different ranks may evaluate differently:

  1. ``finalize_model_grads``: embedding allreduce only on ranks holding
     embedding params; conditional on ``_is_Kx_sync``.
  2. ``clip_grad_norm``: internal allreduce over ``grad_stats_parallel_group``.
  3. ``prepare_grads``: reduce-scatter always, but ``step_with_ready_grads``
     vs ``start_param_sync(force_sync=True)`` differ between skip/non-skip.
  4. ``Kx/Ku/Kv`` sync: gated by modular arithmetic on step + skip flag.

Solution
--------
``CollectiveContract`` makes the per-step collective sequence explicit and
verifiable.  All conditional collectives become unconditional: on non-Kx
steps we still call the collective but with a no-op payload (zero tensor).
A preflight check (``verify_contract``) has all ranks broadcast their
planned sequence of collective names; if any rank disagrees, we abort
with a clear error before NCCL can hang.

Public API
----------
  CollectiveContract   — declares and verifies the per-step sequence.
  ContractViolation    — raised when ranks disagree on the collective order.
  collective_guard     — context manager wrapping a single collective call.
  log_contract_summary — emit a structured log of the step's collective plan.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ContractViolation(RuntimeError):
    """Raised when ranks disagree on the collective sequence.

    Attributes:
        step:        Training step where the mismatch was detected.
        local_seq:   This rank's planned sequence.
        remote_seq:  The sequence received from the mismatched rank.
        rank:        Local rank id.
        remote_rank: Remote rank where the mismatch was found (-1 if unknown).
    """

    def __init__(
        self,
        step: int,
        local_seq: Sequence[str],
        remote_seq: Sequence[str],
        rank: int,
        remote_rank: int = -1,
    ) -> None:
        self.step = step
        self.local_seq = list(local_seq)
        self.remote_seq = list(remote_seq)
        self.rank = rank
        self.remote_rank = remote_rank
        super().__init__(
            f"CollectiveContract violation at step {step} on rank {rank}: "
            f"local sequence has {len(local_seq)} ops "
            f"({', '.join(local_seq[:5])}{'…' if len(local_seq) > 5 else ''}), "
            f"remote sequence has {len(remote_seq)} ops "
            f"({', '.join(remote_seq[:5])}{'…' if len(remote_seq) > 5 else ''})."
        )

    def detailed_report(self) -> str:
        """Return a human-readable multi-line violation report.

        Includes a side-by-side diff showing exactly where the planned
        collective sequences diverge.
        """
        from deepspeed.core.distributed.contract_diagnostics import (
            format_violation_report,
        )
        return format_violation_report(
            step=self.step,
            rank=self.rank,
            local_seq=self.local_seq,
            remote_seq=self.remote_seq,
            remote_rank=self.remote_rank,
        )


# ---------------------------------------------------------------------------
# Collective operation types
# ---------------------------------------------------------------------------

class CollectiveOp(Enum):
    """Enumeration of NCCL collective operation types tracked by the contract."""
    ALL_REDUCE = auto()
    REDUCE_SCATTER = auto()
    ALL_GATHER = auto()
    BROADCAST = auto()
    ALL_TO_ALL = auto()
    BARRIER = auto()
    NOOP = auto()  # placeholder when a conditional collective is skipped


# ---------------------------------------------------------------------------
# Contract entry: one logged collective call
# ---------------------------------------------------------------------------

@dataclass
class ContractEntry:
    """A single collective call in the contract sequence.

    Attributes:
        seq:       Monotonic sequence number within the step.
        name:      Human-readable label (e.g. ``"nan_flag_allreduce"``).
        op:        The NCCL operation type.
        is_noop:   True if this call is a no-op placeholder
                   (conditional collective with empty payload).
        group_key: Identifier for the process group used (e.g. ``"dp"``).
    """
    seq: int
    name: str
    op: CollectiveOp
    is_noop: bool = False
    group_key: str = "world"


# ---------------------------------------------------------------------------
# CollectiveContract
# ---------------------------------------------------------------------------

class CollectiveContract:
    """Declares and enforces the NCCL collective sequence for one training step.

    All ranks must execute the same sequence.  Conditional collectives
    become unconditional (with ``op=NOOP`` when the data is not needed).

    Usage::

        contract = CollectiveContract(step=42, config=desloc_config)

        # Declare planned collectives
        contract.plan("nan_allreduce", CollectiveOp.ALL_REDUCE, group="dp")
        contract.plan("skip_allreduce", CollectiveOp.ALL_REDUCE, group="dp")
        contract.plan("finalize_grads", CollectiveOp.ALL_REDUCE, group="dp")
        contract.plan("clip_norm_allreduce", CollectiveOp.ALL_REDUCE, group="dp")
        contract.plan("prepare_grads_rs", CollectiveOp.REDUCE_SCATTER, group="dp")
        contract.plan("param_sync_ag", CollectiveOp.ALL_GATHER, group="dp")

        # Verify all ranks agree (preflight)
        contract.verify(process_group=dp_group)

        # Execute
        with contract.guard("nan_allreduce"):
            dist.all_reduce(nan_tensor, op=dist.ReduceOp.MAX, group=dp_group)

    Args:
        step:    Current training step (0-indexed).
        config:  Optional DesLocConfig for Kx/Ku/Kv resolution.
        rank:    Local rank (auto-detected if None).
        enabled: Master switch; when False the contract is a no-op passthrough.
    """

    def __init__(
        self,
        step: int,
        config: Any = None,
        rank: Optional[int] = None,
        enabled: bool = True,
    ) -> None:
        self.step = step
        self.config = config
        self.rank = rank if rank is not None else (
            dist.get_rank() if dist.is_initialized() else 0
        )
        self.enabled = enabled

        # Kx/Ku/Kv resolution
        if config is not None and hasattr(config, 'desloc_Kx'):
            self.is_Kx = (step + 1) % config.desloc_Kx == 0
            self.is_Ku = (step + 1) % config.desloc_Ku == 0
            self.is_Kv = (step + 1) % config.desloc_Kv == 0
        elif config is not None and hasattr(config, 'kx'):
            self.is_Kx = (step + 1) % config.kx == 0
            self.is_Ku = (step + 1) % config.ku == 0
            self.is_Kv = (step + 1) % config.kv == 0
        else:
            self.is_Kx = True
            self.is_Ku = False
            self.is_Kv = False

        # Sequence tracking
        self._planned: List[ContractEntry] = []
        self._executed: List[ContractEntry] = []
        self._seq_counter: int = 0
        self._active_guard: Optional[str] = None
        self._verified: bool = False

    def __repr__(self) -> str:
        return (
            f"CollectiveContract(step={self.step}, rank={self.rank}, "
            f"planned={self.planned_count}, executed={len(self._executed)}, "
            f"Kx={self.is_Kx}, Ku={self.is_Ku}, Kv={self.is_Kv}, "
            f"enabled={self.enabled})"
        )

    def reset(self) -> None:
        """Reset execution state so the contract can be re-verified and re-run.

        Keeps the planned sequence intact; clears only the execution log,
        verified flag, and active guard.  Useful for test harnesses that
        replay the same contract multiple times.
        """
        self._executed.clear()
        self._active_guard = None
        self._verified = False

    # ------------------------------------------------------------------
    # Planning API
    # ------------------------------------------------------------------

    def plan(
        self,
        name: str,
        op: CollectiveOp = CollectiveOp.ALL_REDUCE,
        *,
        is_noop: bool = False,
        group: str = "world",
    ) -> ContractEntry:
        """Declare a collective that will be executed in this step.

        Must be called in the exact order the collectives will fire.
        All ranks must call ``plan()`` with the same sequence.

        Args:
            name:    Human-readable label.
            op:      The NCCL operation type.
            is_noop: True if this is a no-op placeholder (conditional skip).
            group:   Identifier for the process group.

        Returns:
            The created ContractEntry.
        """
        self._seq_counter += 1
        entry = ContractEntry(
            seq=self._seq_counter,
            name=name,
            op=op,
            is_noop=is_noop,
            group_key=group,
        )
        self._planned.append(entry)
        return entry

    def plan_conditional(
        self,
        name: str,
        condition: bool,
        op: CollectiveOp = CollectiveOp.ALL_REDUCE,
        *,
        group: str = "world",
    ) -> ContractEntry:
        """Plan a conditional collective — always enters, but payload may be no-op.

        This is the key mechanism: the collective is ALWAYS planned (and later
        executed) regardless of the condition.  When ``condition`` is False,
        the collective runs with a zero/no-op payload so that NCCL sees
        symmetric participation across all ranks.

        Args:
            name:      Human-readable label.
            condition: Whether the collective carries real data.
            op:        The NCCL operation type.
            group:     Identifier for the process group.

        Returns:
            The created ContractEntry.
        """
        return self.plan(name, op, is_noop=not condition, group=group)

    @property
    def planned_sequence(self) -> List[str]:
        """Return the planned collective names in order."""
        return [e.name for e in self._planned]

    @property
    def planned_count(self) -> int:
        """Number of planned collectives."""
        return len(self._planned)

    # ------------------------------------------------------------------
    # Verification API
    # ------------------------------------------------------------------

    def verify(
        self,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> bool:
        """Verify that all ranks planned the same collective sequence.

        Broadcasts each rank's planned sequence of names and checks for
        exact match.  If any rank disagrees, raises ``ContractViolation``
        before any NCCL collective can hang.

        This is a *collective call* — all ranks must enter it.

        Args:
            process_group: The process group to verify across.
                           Defaults to WORLD.

        Returns:
            True if verification passed.

        Raises:
            ContractViolation: If any rank has a different sequence.
        """
        if not self.enabled or not dist.is_initialized():
            self._verified = True
            return True

        world_size = (
            dist.get_world_size(group=process_group)
            if process_group is not None
            else dist.get_world_size()
        )
        if world_size <= 1:
            self._verified = True
            return True

        # Encode planned sequence as a single string
        local_seq_str = "|".join(self.planned_sequence)
        # Pad to fixed length for allgather
        max_len = 4096  # generous upper bound
        encoded = local_seq_str.encode("utf-8")
        if len(encoded) >= max_len:
            raise RuntimeError(
                f"CollectiveContract: planned sequence too long for verify() "
                f"({len(encoded)} >= {max_len} bytes, {self.planned_count} ops). "
                f"Increase max_len or reduce collective name lengths."
            )
        padded = encoded + b"\x00" * (max_len - len(encoded))

        local_tensor = torch.frombuffer(bytearray(padded), dtype=torch.uint8).cuda()
        gathered = [torch.zeros_like(local_tensor) for _ in range(world_size)]
        dist.all_gather(gathered, local_tensor, group=process_group)

        for remote_rank, remote_tensor in enumerate(gathered):
            if remote_rank == self.rank:
                continue
            remote_bytes = remote_tensor.cpu().numpy().tobytes()
            remote_str = remote_bytes.split(b"\x00")[0].decode("utf-8", errors="replace")
            remote_seq = remote_str.split("|") if remote_str else []

            if remote_seq != self.planned_sequence:
                violation = ContractViolation(
                    step=self.step,
                    local_seq=self.planned_sequence,
                    remote_seq=remote_seq,
                    rank=self.rank,
                    remote_rank=remote_rank,
                )
                logger.error(
                    "CollectiveContract VIOLATION detected:\n%s",
                    violation.detailed_report(),
                )
                raise violation

        self._verified = True
        logger.debug(
            "CollectiveContract verified: step=%d rank=%d ops=%d",
            self.step, self.rank, self.planned_count,
        )
        return True

    # ------------------------------------------------------------------
    # Execution API
    # ------------------------------------------------------------------

    @contextmanager
    def guard(self, name: str):
        """Context manager wrapping a single collective call.

        Validates that the collective name matches the next planned entry
        and records the execution.

        Args:
            name: Must match the next planned collective's name.

        Yields:
            The ContractEntry for this collective.

        Raises:
            RuntimeError: If name doesn't match the next planned entry.
        """
        if not self.enabled:
            yield None
            return

        next_idx = len(self._executed)
        if next_idx >= len(self._planned):
            raise RuntimeError(
                f"CollectiveContract: unexpected collective '{name}' at step {self.step} "
                f"(already executed {next_idx} of {len(self._planned)} planned ops)"
            )

        expected = self._planned[next_idx]
        if expected.name != name:
            raise RuntimeError(
                f"CollectiveContract: collective order mismatch at step {self.step}, "
                f"seq={next_idx + 1}: expected '{expected.name}', got '{name}'"
            )

        self._active_guard = name
        try:
            yield expected
        finally:
            self._executed.append(expected)
            self._active_guard = None

    def execute(
        self,
        name: str,
        fn: Callable[..., Any],
        *args: Any,
        noop_fn: Optional[Callable[..., Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute a collective and record it in the contract.

        If the corresponding planned entry has ``is_noop=True``, calls
        ``noop_fn`` instead (or ``fn`` with the same args if ``noop_fn``
        is None).

        Args:
            name:    Must match the next planned collective's name.
            fn:      The collective function to call.
            *args:   Positional args forwarded to fn.
            noop_fn: Alternative function for no-op entries.
            **kwargs: Keyword args forwarded to fn.

        Returns:
            The return value of fn (or noop_fn).
        """
        with self.guard(name) as entry:
            if entry is not None and entry.is_noop and noop_fn is not None:
                return noop_fn(*args, **kwargs)
            return fn(*args, **kwargs)

    @property
    def all_executed(self) -> bool:
        """True if all planned collectives have been executed."""
        return len(self._executed) == len(self._planned)

    def assert_complete(self) -> None:
        """Assert that all planned collectives were executed.

        Raises:
            RuntimeError: If any planned collective was not executed.
        """
        if not self.enabled:
            return
        if not self.all_executed:
            missing = self._planned[len(self._executed):]
            names = [e.name for e in missing]
            raise RuntimeError(
                f"CollectiveContract: {len(missing)} planned collective(s) "
                f"not executed at step {self.step}: {names}"
            )

    # ------------------------------------------------------------------
    # Summary / logging
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a structured summary of the contract for logging."""
        return {
            "step": self.step,
            "rank": self.rank,
            "is_Kx": self.is_Kx,
            "is_Ku": self.is_Ku,
            "is_Kv": self.is_Kv,
            "planned_count": self.planned_count,
            "executed_count": len(self._executed),
            "verified": self._verified,
            "planned_ops": [
                {"seq": e.seq, "name": e.name, "op": e.op.name, "noop": e.is_noop}
                for e in self._planned
            ],
        }


# ---------------------------------------------------------------------------
# Convenience: log_contract_summary
# ---------------------------------------------------------------------------

def log_contract_summary(contract: CollectiveContract) -> None:
    """Emit a structured log of the contract's collective plan."""
    s = contract.summary()
    logger.info(
        "[CollectiveContract] step=%d rank=%d Kx=%s Ku=%s Kv=%s "
        "planned=%d executed=%d verified=%s",
        s["step"], s["rank"],
        s["is_Kx"], s["is_Ku"], s["is_Kv"],
        s["planned_count"], s["executed_count"], s["verified"],
    )
    for entry in s["planned_ops"]:
        logger.debug(
            "  seq=%d %s (%s) noop=%s",
            entry["seq"], entry["name"], entry["op"], entry["noop"],
        )


# ---------------------------------------------------------------------------
# Factory: build_step_contract
# ---------------------------------------------------------------------------

def build_step_contract(
    step: int,
    config: Any,
    *,
    has_dist_optimizer: bool = False,
    step_has_nan: bool = False,
    should_skip: bool = False,
    rank: Optional[int] = None,
) -> CollectiveContract:
    """Build a CollectiveContract with the standard DES-LOC collective plan.

    This encodes the exact sequence of NCCL collectives that every rank
    must execute during one training step, making conditional collectives
    unconditional.

    The sequence matches the post-fix ordering in desloc_engine.train():

      1. nan_flag_allreduce      — always (MAX reduce of NaN flag)
      2. finalize_model_grads    — always (skip_grad_sync controls payload)
      3. clip_grad_norm_allreduce — always (gradient norm reduction)
      4. skip_flag_allreduce     — always (MAX reduce of skip decision)
      5. prepare_grads_rs        — always when dist_optimizer (reduce-scatter)
      6. param_sync              — always when dist_optimizer (all-gather/broadcast)

    Args:
        step:               Current step (0-indexed).
        config:             Engine config or DesLocConfig carrying Kx/Ku/Kv.
        has_dist_optimizer: Whether DistributedOptimizer is active.
        step_has_nan:       Whether this step detected NaN loss.
        should_skip:        Whether the optimizer update should be skipped.
        rank:               Local rank (auto-detected if None).

    Returns:
        A fully-planned CollectiveContract ready for verify() + execute().
    """
    contract = CollectiveContract(step=step, config=config, rank=rank)

    # 1. NaN flag allreduce — always, all ranks
    contract.plan("nan_flag_allreduce", CollectiveOp.ALL_REDUCE, group="dp")

    # 2. finalize_model_grads — always called, skip_grad_sync controls payload
    #    Internally: DDP finish_grad_sync (allreduce or reduce-scatter)
    #              + embedding allreduce + SP allreduce
    contract.plan("finalize_model_grads", CollectiveOp.ALL_REDUCE, group="dp")

    # 3. clip_grad_norm — always called, internal allreduce for norm
    contract.plan("clip_grad_norm_allreduce", CollectiveOp.ALL_REDUCE, group="dp")

    # 4. skip flag allreduce — always, all ranks
    contract.plan("skip_flag_allreduce", CollectiveOp.ALL_REDUCE, group="dp")

    # 5-6. Optimizer collectives (DistributedOptimizer path)
    if has_dist_optimizer:
        # prepare_grads → reduce_scatter_tensor: always entered
        contract.plan("prepare_grads_rs", CollectiveOp.REDUCE_SCATTER, group="dp")

        # param_sync: always entered — either step_with_ready_grads
        # (which internally does broadcast) or start_param_sync(force_sync=True)
        # on the skip path.  Both call the same NCCL collective.
        contract.plan("param_sync", CollectiveOp.ALL_GATHER, group="dp")

    return contract


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "CollectiveContract",
    "CollectiveOp",
    "ContractEntry",
    "ContractViolation",
    "build_step_contract",
    "log_contract_summary",
]
