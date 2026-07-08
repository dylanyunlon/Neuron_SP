# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Pipeline-parallel utility classes and helpers.

Ported from Megatron-LM/megatron/core/pipeline_parallel/utils.py and extended
with DES-LOC heterogeneous-cluster primitives.

Contents
--------
* ``is_pp_first_stage``, ``is_pp_last_stage`` — boundary predicates
* ``is_vp_first_stage``, ``is_vp_last_stage`` — virtual-pipeline predicates
* ``get_pp_first_rank``, ``get_pp_last_rank``, ``get_pp_next_rank``,
  ``get_pp_prev_rank`` — rank look-up helpers
* ``make_viewless`` — detach tensor views so autograd does not accumulate
  through the view chain (prevents AccumulateGrad stream warnings)
* ``set_ideal_affinity_for_current_gpu`` — optional NUMA pinning
* ``NoopScheduleNode``, ``ScheduleNode``, ``AbstractSchedulePlan`` —
  fine-grained scheduling primitives used by combined-1F1B / MoE overlap
* ``set_streams``, ``get_comp_stream``, ``get_comm_stream`` — CUDA stream
  management for overlapping communication with computation

DES-LOC additions
-----------------
* ``get_tier_for_rank`` — returns the compute tier (DATACENTER / PROFESSIONAL)
  for a given pipeline rank based on ``CUDA_VISIBLE_DEVICES`` capability.
* ``tier_priority_stream`` — creates a CUDA stream whose priority is set
  according to the tier of the *current* rank, so fast stages (H100) use
  high-priority streams for P2P while slow stages (A6000) use default
  priority and rely on the bubble filler to amortise the latency.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Callable, Optional

import torch
from torch.autograd import Variable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core utility helpers (mirror megatron.core.utils for the subset we need)
# ---------------------------------------------------------------------------

def _get_pg_rank(group: torch.distributed.ProcessGroup) -> int:
    """Return the rank of the current process in *group*."""
    return torch.distributed.get_rank(group=group)


def _get_pg_size(group: torch.distributed.ProcessGroup) -> int:
    """Return the world size of *group*."""
    return torch.distributed.get_world_size(group=group)


def _make_viewless_tensor(
    inp: torch.Tensor,
    requires_grad: bool = False,
    keep_graph: bool = True,
) -> torch.Tensor:
    """Create a new tensor that shares storage with *inp* but is *not* a view.

    This prevents autograd from walking up long view chains which would trigger
    the ``AccumulateGrad`` stream warning.
    """
    if not isinstance(inp, torch.Tensor):
        return inp
    if inp._base is None:
        return inp
    out = torch.Tensor._make_subclass(type(inp), inp, requires_grad)
    return out


def _nvtx_range_push(name: str = "", **_kw) -> None:  # noqa: D401 – thin wrapper
    """Push an NVTX range (no-op when NVTX is unavailable)."""
    try:
        torch.cuda.nvtx.range_push(name)
    except Exception:
        pass


def _nvtx_range_pop(name: str = "", **_kw) -> None:  # noqa: D401 – thin wrapper
    """Pop an NVTX range (no-op when NVTX is unavailable)."""
    try:
        torch.cuda.nvtx.range_pop()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Pipeline-stage predicates
# ---------------------------------------------------------------------------

def is_pp_first_stage(pp_group: torch.distributed.ProcessGroup) -> bool:
    """Return ``True`` if the current rank is the first pipeline stage."""
    return _get_pg_rank(pp_group) == 0


def is_pp_last_stage(pp_group: torch.distributed.ProcessGroup) -> bool:
    """Return ``True`` if the current rank is the last pipeline stage."""
    return _get_pg_rank(pp_group) == (_get_pg_size(pp_group) - 1)


# ---------------------------------------------------------------------------
# Virtual-pipeline stage predicates
# ---------------------------------------------------------------------------

def is_vp_first_stage(vp_stage: Optional[int], vp_size: Optional[int]) -> bool:
    """Return ``True`` if in the first virtual pipeline stage."""
    if vp_size is None or vp_size <= 1:
        assert vp_stage is None or vp_stage == 0, (
            f"Expected vp_stage to be 0 or None when vp_size is <= 1 or None, "
            f"but got vp_stage={vp_stage} and vp_size={vp_size}"
        )
        return True
    return vp_stage == 0


def is_vp_last_stage(vp_stage: Optional[int], vp_size: Optional[int]) -> bool:
    """Return ``True`` if in the last virtual pipeline stage."""
    if vp_size is None or vp_size <= 1:
        assert vp_stage is None or vp_stage == 0, (
            f"Expected vp_stage to be 0 or None when vp_size is <= 1 or None, "
            f"but got vp_stage={vp_stage} and vp_size={vp_size}"
        )
        return True
    return vp_stage == (vp_size - 1)


# ---------------------------------------------------------------------------
# Rank look-up helpers
# ---------------------------------------------------------------------------

def get_pp_first_rank(pp_group: torch.distributed.ProcessGroup) -> int:
    """Return the global rank of the first rank in the PP group."""
    pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
    return pp_ranks[0]


def get_pp_last_rank(pp_group: torch.distributed.ProcessGroup) -> int:
    """Return the global rank of the last rank in the PP group."""
    pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
    return pp_ranks[-1]


def get_pp_next_rank(
    pp_group: torch.distributed.ProcessGroup,
) -> Optional[int]:
    """Return the global rank of the next rank in the PP group, or ``None``."""
    if is_pp_last_stage(pp_group):
        return None
    current_rank_in_group = _get_pg_rank(pp_group)
    pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
    return pp_ranks[current_rank_in_group + 1]


def get_pp_prev_rank(
    pp_group: torch.distributed.ProcessGroup,
) -> Optional[int]:
    """Return the global rank of the previous rank in the PP group, or ``None``."""
    if is_pp_first_stage(pp_group):
        return None
    current_rank_in_group = _get_pg_rank(pp_group)
    pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
    return pp_ranks[current_rank_in_group - 1]


# ---------------------------------------------------------------------------
# Tensor utilities
# ---------------------------------------------------------------------------

def make_viewless(e: torch.Tensor) -> torch.Tensor:
    """Convenience wrapper around :func:`_make_viewless_tensor`."""
    if not isinstance(e, torch.Tensor):
        return e
    return _make_viewless_tensor(inp=e, requires_grad=e.requires_grad, keep_graph=True)


# ---------------------------------------------------------------------------
# NUMA affinity (optional, best-effort)
# ---------------------------------------------------------------------------

def set_ideal_affinity_for_current_gpu() -> None:
    """Pin the current process to the NUMA node closest to its CUDA device.

    Requires ``cuda-python`` and ``pynvml``.  Silently falls back to a no-op
    when the dependencies are missing.
    """
    import uuid as _uuid_mod

    try:
        import cuda.bindings.driver as cuda_driver
        import cuda.bindings.runtime as cuda_runtime
    except ImportError:
        try:
            import cuda.cuda as cuda_driver  # type: ignore[no-redef]
            import cuda.cudart as cuda_runtime  # type: ignore[no-redef]
        except ImportError:
            logger.debug("cuda-python not installed — skipping GPU affinity")
            return

    try:
        import pynvml
    except ImportError:
        logger.debug("pynvml not installed — skipping GPU affinity")
        return

    err, device_id = cuda_runtime.cudaGetDevice()
    assert err == cuda_runtime.cudaError_t.cudaSuccess
    err, device_uuid = cuda_driver.cuDeviceGetUuid(device_id)
    assert err == cuda_driver.CUresult.CUDA_SUCCESS
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByUUID(
        "GPU-" + str(_uuid_mod.UUID(bytes=device_uuid.bytes))
    )
    pynvml.nvmlDeviceSetCpuAffinity(handle)
    logger.warning("Set CPU affinity for optimal host-device transfer performance")


# ---------------------------------------------------------------------------
# DES-LOC tier helpers
# ---------------------------------------------------------------------------

_TIER_DATACENTER = "DATACENTER"
_TIER_PROFESSIONAL = "PROFESSIONAL"


def get_tier_for_rank(pp_rank: int = 0) -> str:
    """Return the compute tier of the GPU assigned to *pp_rank*.

    Falls back to ``DATACENTER`` when detection is unavailable.  The actual
    mapping in production is supplied by ``HeteroRegistry`` and persisted in
    ``parallel_state``.  This helper provides a lightweight alternative for
    pipeline utilities that do not have access to the full registry.
    """
    try:
        from deepspeed.core.parallel_state import get_tier_for_rank as _get_tier
        return _get_tier(pp_rank)
    except (ImportError, AttributeError):
        pass

    # Heuristic: check the *current* GPU's compute capability
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 9:
            return _TIER_DATACENTER
        return _TIER_PROFESSIONAL
    return _TIER_DATACENTER


def tier_priority_stream(
    device: str = "cuda",
    tier: Optional[str] = None,
) -> torch.cuda.Stream:
    """Create a CUDA stream whose priority reflects the compute tier.

    H100 (DATACENTER) → high priority so P2P transfers preempt compute.
    A6000 (PROFESSIONAL) → default priority; the bubble filler handles
    amortisation of any additional latency.
    """
    if tier is None:
        tier = get_tier_for_rank()
    if tier == _TIER_DATACENTER:
        _, high = torch.cuda.Stream.priority_range()
        return torch.cuda.Stream(device=device, priority=high)
    return torch.cuda.Stream(device=device)


# ---------------------------------------------------------------------------
# NoopScheduleNode — placeholder for sparse schedule graphs
# ---------------------------------------------------------------------------

class NoopScheduleNode:
    """A no-op node that passes inputs/outputs through unchanged.

    Used when a real computation node is not needed but the interface must be
    maintained (e.g. dense layers that skip MoE dispatch/combine).
    """

    def forward(self, inputs):  # noqa: D401 – pass-through
        """Return *inputs* unchanged."""
        return inputs

    def backward(self, outgrads):  # noqa: D401 – pass-through
        """Return *outgrads* unchanged."""
        return outgrads


# ---------------------------------------------------------------------------
# ScheduleNode — fine-grained compute/communicate scheduling
# ---------------------------------------------------------------------------

class ScheduleNode:
    """Computational node in a fine-grained pipeline schedule.

    Each node binds a forward (and optionally backward) function to a specific
    CUDA stream and synchronises with other nodes via a shared ``torch.cuda.Event``.
    """

    def __init__(
        self,
        forward_func: Callable,
        stream: "torch.cuda.Stream | Callable",
        event: torch.cuda.Event,
        backward_func: Optional[Callable] = None,
        free_input: bool = False,
        name: str = "schedule_node",
    ):
        self.name = name
        self.forward_func = forward_func
        self.backward_func = backward_func if backward_func else self.default_backward_func
        self.stream = stream
        self.event = event
        self.free_input = free_input
        self.inputs = None
        self.output = None

    # -- default backward ------------------------------------------------
    def default_backward_func(self, outputs, output_grad):
        Variable._execution_engine.run_backward(
            tensors=outputs,
            grad_tensors=output_grad,
            keep_graph=False,
            create_graph=False,
            inputs=tuple(),
            allow_unreachable=True,
            accumulate_grad=True,
        )
        return output_grad

    # -- forward ----------------------------------------------------------
    def forward(self, inputs=()):
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        return self._forward(*inputs)

    def _forward(self, *inputs):
        if isinstance(self.stream, Callable):
            self.stream = self.stream()
        with self.stream_acquire_context(f"{self.name} forward"):
            self.inputs = [
                make_viewless(e).detach() if e is not None else None
                for e in inputs
            ]
            for i, inp in enumerate(self.inputs):
                if inp is not None:
                    inp.requires_grad = inputs[i].requires_grad

            data = tuple(self.inputs)
            data = self.forward_func(*data)

            if not isinstance(data, tuple):
                data = make_viewless(data)
            else:
                data = tuple(
                    make_viewless(e) if isinstance(e, torch.Tensor) else e
                    for e in data
                )
            self.output = data

        if self.free_input:
            for inp in inputs:
                if inp is not None:
                    inp.record_stream(self.stream)
                    inp.untyped_storage().resize_(0)
        return self.output

    def get_output(self):
        return self.output

    # -- backward ---------------------------------------------------------
    def backward(self, output_grad):
        if not isinstance(output_grad, tuple):
            output_grad = (output_grad,)
        return self._backward(*output_grad)

    def _backward(self, *output_grad):
        if isinstance(self.stream, Callable):
            self.stream = self.stream()
        with self.stream_acquire_context(f"{self.name} backward"):
            outputs = self.output
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            assert len(outputs) == len(output_grad), (
                f"{len(outputs)} of {type(outputs[0])} is not equal to "
                f"{len(output_grad)} of {type(output_grad[0])}"
            )
            output_grad = self.backward_func(outputs, output_grad)

        if output_grad:
            for g in output_grad:
                if g is not None:
                    g.record_stream(self.stream)

        grads = self.get_grad()
        self._release_state()
        return grads

    def get_grad(self):
        grad = tuple(
            e.grad if e is not None else None for e in self.inputs
        )
        if len(grad) == 1:
            grad = grad[0]
        return grad

    @contextmanager
    def stream_acquire_context(self, name: Optional[str] = None):
        """Synchronise on the shared event, run inside the node's stream."""
        self.event.wait(self.stream)
        if name:
            _nvtx_range_push(name)
        try:
            with torch.cuda.stream(self.stream):
                yield
        finally:
            if name:
                _nvtx_range_pop(name)
            self.event.record(self.stream)

    def _release_state(self):
        self.inputs = None
        self.output = None
        del self.forward_func
        del self.backward_func


# ---------------------------------------------------------------------------
# AbstractSchedulePlan — protocol for combined-1F1B model integration
# ---------------------------------------------------------------------------

class AbstractSchedulePlan(ABC):
    """Protocol for models that support combined-1F1B scheduling.

    A model's ``build_schedule_plan`` method must return an instance of a
    concrete subclass.  The ``run`` class method drives forward/backward for
    a pair of schedule plans (one per microbatch in the overlap window).
    """

    @staticmethod
    @abstractmethod
    def run(
        f_schedule_plan,
        b_schedule_plan,
        grad=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
    ):
        """Run the forward plan *f_schedule_plan* overlapped with the backward
        plan *b_schedule_plan*."""
        ...


# ---------------------------------------------------------------------------
# CUDA stream management (global singletons)
# ---------------------------------------------------------------------------

_USE_DYNAMIC_COMP_STREAM: Optional[bool] = None
_COMP_STREAM: Optional[torch.cuda.Stream] = None
_COMM_STREAM: Optional[torch.cuda.Stream] = None


def set_streams(
    comm_stream: Optional[torch.cuda.Stream] = None,
    high_priority: bool = False,
) -> None:
    """Lazily initialise the global communication stream.

    In DES-LOC setups, ``high_priority=True`` is used on DATACENTER-tier
    ranks to give P2P traffic scheduling precedence.
    """
    global _COMM_STREAM
    if _COMM_STREAM is None:
        if comm_stream is None:
            if high_priority:
                _, high = torch.cuda.Stream.priority_range()
                comm_stream = torch.cuda.Stream(device="cuda", priority=high)
            else:
                comm_stream = torch.cuda.Stream(device="cuda")
        _COMM_STREAM = comm_stream


def get_comp_stream() -> torch.cuda.Stream:
    """Return the computation stream (always ``current_stream``)."""
    return torch.cuda.current_stream()


def get_comm_stream() -> Optional[torch.cuda.Stream]:
    """Return the communication stream (lazily created by :func:`set_streams`)."""
    return _COMM_STREAM


# ===========================================================================
# DES-LOC: Tier-proportional stage assignment + stage micro-batch sizing
# ===========================================================================

def get_pp_stage_compute_factor(
    pp_rank: int,
    config=None,
) -> float:
    """Return relative compute speed factor for a PP stage.

    Factor of 1.0 = slowest tier.  H100 (312 bf16 TFLOPS) vs A6000
    (77.4 bf16 TFLOPS) gives factor ≈ 4.03 for H100, 1.0 for A6000.
    """
    desloc = getattr(config, "desloc", None) if config is not None else None
    if desloc is None or not getattr(desloc, "enabled", False) or not desloc.tiers:
        return 1.0
    import os
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", pp_rank))
    except (ValueError, TypeError):
        local_rank = pp_rank
    tier = desloc.get_tier_for_gpu(local_rank)
    if tier is None:
        return 1.0
    min_tflops = min(t.bf16_tflops for t in desloc.tiers)
    if min_tflops <= 0:
        return 1.0
    return tier.bf16_tflops / min_tflops


def is_fast_stage(pp_rank: int, config=None, threshold: float = 1.5) -> bool:
    """True if the PP rank is on a "fast" GPU tier (factor >= threshold)."""
    return get_pp_stage_compute_factor(pp_rank, config) >= threshold


def is_slow_stage(pp_rank: int, config=None, threshold: float = 1.5) -> bool:
    """True if the PP rank is on a "slow" GPU tier."""
    return not is_fast_stage(pp_rank, config, threshold)


def optimal_pp_stage_assignment(
    num_layers: int,
    pp_size: int,
    config=None,
) -> "list[int]":
    """Compute optimal layer-to-stage assignment for heterogeneous PP.

    Fast GPUs get proportionally more layers so that wall-clock time
    per stage is balanced.  Uses BF16 TFLOPS as throughput proxy:

        layers_i = round(num_layers * tflops_i / sum(tflops))

    Returns list of length pp_size with per-stage layer counts.
    """
    if pp_size <= 0:
        return []
    factors = [get_pp_stage_compute_factor(r, config) for r in range(pp_size)]
    total_factor = sum(factors)
    if total_factor <= 0 or all(f == factors[0] for f in factors):
        base = num_layers // pp_size
        remainder = num_layers % pp_size
        return [base + (1 if r < remainder else 0) for r in range(pp_size)]
    raw = [num_layers * f / total_factor for f in factors]
    assignment = [max(1, round(r)) for r in raw]
    diff = num_layers - sum(assignment)
    if diff > 0:
        indices = sorted(range(pp_size), key=lambda i: factors[i], reverse=True)
        for i in range(diff):
            assignment[indices[i % pp_size]] += 1
    elif diff < 0:
        indices = sorted(range(pp_size), key=lambda i: factors[i])
        for i in range(-diff):
            if assignment[indices[i % pp_size]] > 1:
                assignment[indices[i % pp_size]] -= 1
    return assignment


def get_pp_stage_micro_batch_size(
    pp_rank: int,
    default_micro_batch_size: int,
    config=None,
) -> int:
    """Return per-stage micro-batch size, accounting for tier VRAM capacity.

    Slow-VRAM stages (A6000, 48 GB) may need smaller micro-batch sizes
    than fast-VRAM stages (H100, 80 GB) to fit activations.
    """
    if config is None:
        return default_micro_batch_size
    desloc = getattr(config, "desloc", None)
    if desloc is None or not getattr(desloc, "enabled", False):
        return default_micro_batch_size
    per_tier = getattr(desloc, "micro_batch_per_tier", {})
    import os
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", pp_rank))
    except (ValueError, TypeError):
        local_rank = pp_rank
    tier = desloc.get_tier_for_gpu(local_rank)
    if tier is not None and tier.tier_type.value in per_tier:
        return per_tier[tier.tier_type.value]
    per_stage = getattr(config, "per_stage_micro_batch_sizes", None)
    if per_stage is not None and pp_rank < len(per_stage):
        return per_stage[pp_rank]
    return default_micro_batch_size
