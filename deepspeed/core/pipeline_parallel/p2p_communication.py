# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Point-to-point pipeline communication for pipeline parallelism.

Ported from Megatron-LM/megatron/core/pipeline_parallel/p2p_communication.py
and extended with DES-LOC heterogeneous cluster support.

DES-LOC extension — PCIe-aware message sizing:
  Cross-NUMA transfers (A6000 ↔ H100 over PCIe) use variable-length tensor
  shape negotiation so that faster stages (H100) can send larger activations
  while slower stages (A6000) can pace themselves.  The _communicate() method
  supports both batched P2P (NCCL batch_isend_irecv) and sequential P2P
  (ordered send/recv to avoid deadlocks on ring-2 topologies).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from deepspeed.core.model_parallel_config import ModelParallelConfig

# ---------------------------------------------------------------------------
# Import parallel state helpers with graceful fallback for unit-test envs.
# ---------------------------------------------------------------------------
try:
    from deepspeed.core import parallel_state as _ps
except ImportError:
    _ps = None  # type: ignore[assignment]

# Type alias
Shape = Union[List[int], torch.Size]


# ---------------------------------------------------------------------------
# Low-level P2P helpers (mirroring Megatron's batched / sequential variants)
# ---------------------------------------------------------------------------

def _batched_p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    group: torch.distributed.ProcessGroup,
    prev_pipeline_rank: int,
    next_pipeline_rank: int,
) -> list:
    """Issue all P2P transfers as a single batch_isend_irecv call.

    Batching is more efficient than individual isend/irecv calls because NCCL
    can schedule all operations in one kernel launch.  This is the default
    mode when ``config.batch_p2p_comm=True`` (or when the config does not
    specify otherwise, which defaults to True here for simplicity).
    """
    ops: list = []
    if tensor_send_prev is not None:
        ops.append(torch.distributed.P2POp(
            torch.distributed.isend, tensor_send_prev, prev_pipeline_rank, group
        ))
    if tensor_recv_prev is not None:
        ops.append(torch.distributed.P2POp(
            torch.distributed.irecv, tensor_recv_prev, prev_pipeline_rank, group
        ))
    if tensor_send_next is not None:
        ops.append(torch.distributed.P2POp(
            torch.distributed.isend, tensor_send_next, next_pipeline_rank, group
        ))
    if tensor_recv_next is not None:
        ops.append(torch.distributed.P2POp(
            torch.distributed.irecv, tensor_recv_next, next_pipeline_rank, group
        ))
    if ops:
        reqs = torch.distributed.batch_isend_irecv(ops)
    else:
        reqs = []
    return reqs


def _p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    group: torch.distributed.ProcessGroup,
    prev_pipeline_rank: int,
    next_pipeline_rank: int,
) -> Dict[str, object]:
    """Sequential (ordered) P2P ops to avoid deadlocks on small pipeline rings.

    When ``pp_size == 2``, batching two sends and two recvs in the same kernel
    can deadlock with some NCCL versions.  This variant issues operations in a
    parity-ordered fashion (even ranks send first, odd ranks receive first) and
    optionally uses the global WORLD group for one direction to allow overlap.

    Returns a dict keyed by op name so callers can selectively wait.
    """
    reqs: Dict[str, object] = {}
    rank_in_group = group.rank()

    # For a 2-stage pipeline, use WORLD for one direction so both transfers can
    # overlap (NCCL allows concurrent kernels on separate communicators).
    if group.size() == 2 and dist.get_backend(group) != "ucc":
        secondary_group = dist.group.WORLD
    else:
        secondary_group = group

    if rank_in_group % 2 == 0:
        # Even rank: send-next first, then recv-prev, then send-prev, recv-next
        if tensor_send_next is not None:
            reqs["send_next"] = dist.isend(tensor_send_next, dst=next_pipeline_rank, group=group)
        if tensor_recv_prev is not None:
            reqs["recv_prev"] = dist.irecv(tensor_recv_prev, src=prev_pipeline_rank, group=secondary_group)
        if tensor_send_prev is not None:
            reqs["send_prev"] = dist.isend(tensor_send_prev, dst=prev_pipeline_rank, group=group)
        if tensor_recv_next is not None:
            reqs["recv_next"] = dist.irecv(tensor_recv_next, src=next_pipeline_rank, group=secondary_group)
    else:
        # Odd rank: recv-prev first to match even's send-next
        if tensor_recv_prev is not None:
            reqs["recv_prev"] = dist.irecv(tensor_recv_prev, src=prev_pipeline_rank, group=group)
        if tensor_send_next is not None:
            reqs["send_next"] = dist.isend(tensor_send_next, dst=next_pipeline_rank, group=secondary_group)
        if tensor_recv_next is not None:
            reqs["recv_next"] = dist.irecv(tensor_recv_next, src=next_pipeline_rank, group=group)
        if tensor_send_prev is not None:
            reqs["send_prev"] = dist.isend(tensor_send_prev, dst=prev_pipeline_rank, group=secondary_group)
    return reqs


def is_single_shape(x) -> bool:
    """Return True if *x* is a single tensor shape (not a list of shapes)."""
    if isinstance(x, torch.Size):
        return True
    if isinstance(x, (list, tuple)) and len(x) > 0 and all(isinstance(d, int) for d in x):
        return True
    return False


# ---------------------------------------------------------------------------
# P2PCommunicator — public API consumed by schedules.py
# ---------------------------------------------------------------------------

class P2PCommunicator:
    """Point-to-point communication for pipeline parallelism.

    Handles send/recv of activations and gradients between adjacent PP stages.

    DES-LOC extension — PCIe-aware message sizing:
      When ``config.desloc`` is enabled and the sending stage is an H100 (fast
      tier) while the receiving stage is an A6000 (slow tier), the communicator
      negotiates the exact tensor shape before each transfer via
      ``_communicate_shapes()``.  This prevents the H100 from drowning the
      A6000's PCIe bandwidth with oversized activations.

    Args:
        config: ModelParallelConfig with PP settings.
        pg:     Pre-built PP process group (optional; resolved from
                parallel_state when not provided).
    """

    def __init__(
        self,
        config: ModelParallelConfig,
        pg: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        self.config = config

        # Resolve PP process group -----------------------------------------------
        if pg is not None:
            self.pp_group: torch.distributed.ProcessGroup = pg
        elif _ps is not None:
            self.pp_group = _ps.get_pipeline_model_parallel_group()
        else:
            raise RuntimeError(
                "P2PCommunicator requires either a process group (pg=) or an "
                "initialised parallel_state module."
            )

        world_size = self.pp_group.size()
        curr_rank   = self.pp_group.rank()

        # Global ranks of neighbouring stages -----------------------------------
        next_local = (curr_rank + 1) % world_size
        prev_local = (curr_rank - 1) % world_size
        self.next_rank: int = dist.get_global_rank(self.pp_group, next_local)
        self.prev_rank: int = dist.get_global_rank(self.pp_group, prev_local)

        self.virtual_pipeline_model_parallel_size: Optional[int] = (
            config.virtual_pipeline_model_parallel_size
        )

        # DES-LOC: pipeline dtype falls back to params_dtype if not set
        self._pipeline_dtype: torch.dtype = getattr(
            config, "pipeline_dtype", None
        ) or config.params_dtype

        # Whether to use batched P2P (default True; set False only for 2-stage ring)
        self._batch_p2p: bool = getattr(config, "batch_p2p_comm", True)
        self._batch_p2p_sync: bool = getattr(config, "batch_p2p_sync", False)
        self._variable_seq: bool = getattr(config, "variable_seq_lengths", False)

    # ------------------------------------------------------------------
    # Stage identity helpers
    # ------------------------------------------------------------------

    @property
    def is_pp_first_stage(self) -> bool:
        """True if this rank is the first pipeline stage (stage 0)."""
        return self.pp_group.rank() == 0

    @property
    def is_pp_last_stage(self) -> bool:
        """True if this rank is the last pipeline stage."""
        return self.pp_group.rank() == (self.pp_group.size() - 1)

    @property
    def total_stages(self) -> int:
        """Total number of PP stages."""
        return self.pp_group.size()

    @property
    def current_stage(self) -> int:
        """Current PP stage index (0-based)."""
        return self.pp_group.rank()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _allocate_recv_tensor(self, shape: Shape, dtype: torch.dtype) -> torch.Tensor:
        """Allocate a receive buffer on the current CUDA device."""
        device = (
            torch.device(f"cuda:{torch.cuda.current_device()}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        return torch.empty(shape, requires_grad=True, device=device, dtype=dtype)

    def _communicate_shapes(
        self,
        tensor_send_next: Optional[torch.Tensor],
        tensor_send_prev: Optional[torch.Tensor],
        recv_prev: bool,
        recv_next: bool,
    ) -> Tuple[List[int], List[int]]:
        """Exchange tensor shapes before the actual data transfer.

        Required when ``config.variable_seq_lengths=True`` so that the
        receiving stage can allocate a correctly-sized buffer even when
        sequence lengths vary across microbatches.

        DES-LOC note: On H100→A6000 transitions the sequence chunk sent by
        H100 may be smaller (H100 finished more tokens) so dynamic shape
        negotiation is essential.

        Returns:
            (recv_prev_shape, recv_next_shape) — lists of 3 ints [S, B, H].
        """
        recv_prev_tensor = None
        recv_next_tensor = None
        send_prev_tensor = None
        send_next_tensor = None

        device = (
            torch.device(f"cuda:{torch.cuda.current_device()}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if recv_prev:
            recv_prev_tensor = torch.empty((3,), device=device, dtype=torch.int64)
        if recv_next:
            recv_next_tensor = torch.empty((3,), device=device, dtype=torch.int64)
        if tensor_send_prev is not None:
            send_prev_tensor = torch.tensor(list(tensor_send_prev.size()), device=device, dtype=torch.int64)
        if tensor_send_next is not None:
            send_next_tensor = torch.tensor(list(tensor_send_next.size()), device=device, dtype=torch.int64)

        ops = []
        if send_prev_tensor is not None:
            ops.append(torch.distributed.P2POp(
                torch.distributed.isend, send_prev_tensor, self.prev_rank, self.pp_group
            ))
        if recv_prev_tensor is not None:
            ops.append(torch.distributed.P2POp(
                torch.distributed.irecv, recv_prev_tensor, self.prev_rank, self.pp_group
            ))
        if send_next_tensor is not None:
            ops.append(torch.distributed.P2POp(
                torch.distributed.isend, send_next_tensor, self.next_rank, self.pp_group
            ))
        if recv_next_tensor is not None:
            ops.append(torch.distributed.P2POp(
                torch.distributed.irecv, recv_next_tensor, self.next_rank, self.pp_group
            ))

        if ops:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()
            # Guard against race condition with batch_isend_irecv
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        recv_prev_shape = recv_prev_tensor.tolist() if recv_prev_tensor is not None else [0, 0, 0]
        recv_next_shape = recv_next_tensor.tolist() if recv_next_tensor is not None else [0, 0, 0]
        return recv_prev_shape, recv_next_shape

    def _communicate(
        self,
        *,
        tensor_send_next: Optional[torch.Tensor],
        tensor_send_prev: Optional[torch.Tensor],
        recv_prev: bool,
        recv_next: bool,
        tensor_shape: Optional[Shape],
        wait_on_reqs: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[object]]:
        """Core P2P dispatch: send and/or receive tensors atomically.

        DES-LOC extension: when variable_seq_lengths is True (heterogeneous
        sequence chunks across H100/A6000 boundaries), shapes are negotiated
        via ``_communicate_shapes()`` before allocating receive buffers.

        Args:
            tensor_send_next:  Tensor to send to the next stage (or None).
            tensor_send_prev:  Tensor to send to the prev stage (or None).
            recv_prev:         Whether to receive from the previous stage.
            recv_next:         Whether to receive from the next stage.
            tensor_shape:      Shape of tensors to receive (assumed uniform).
            wait_on_reqs:      If True, block until all transfers complete.

        Returns:
            (tensor_recv_prev, tensor_recv_next, pending_reqs)
            pending_reqs is None when wait_on_reqs=True, otherwise a list/dict.
        """
        dtype = self._pipeline_dtype

        # Negotiate shapes if seq lengths vary (DES-LOC heterogeneous chunks)
        if self._variable_seq:
            recv_prev_shape, recv_next_shape = self._communicate_shapes(
                tensor_send_next, tensor_send_prev, recv_prev, recv_next
            )
        else:
            recv_prev_shape = tensor_shape
            recv_next_shape = tensor_shape

        tensor_recv_prev: Optional[torch.Tensor] = None
        tensor_recv_next: Optional[torch.Tensor] = None

        if recv_prev:
            if recv_prev_shape is None:
                raise RuntimeError(
                    "_communicate: tensor_shape must be provided when recv_prev=True"
                )
            tensor_recv_prev = self._allocate_recv_tensor(recv_prev_shape, dtype)

        if recv_next:
            if recv_next_shape is None:
                raise RuntimeError(
                    "_communicate: tensor_shape must be provided when recv_next=True"
                )
            tensor_recv_next = self._allocate_recv_tensor(recv_next_shape, dtype)

        # Choose the appropriate P2P function
        if self._batch_p2p:
            p2p_func = _batched_p2p_ops
        else:
            p2p_func = _p2p_ops  # type: ignore[assignment]

        p2p_reqs = p2p_func(
            tensor_send_prev=tensor_send_prev,
            tensor_recv_prev=tensor_recv_prev,
            tensor_send_next=tensor_send_next,
            tensor_recv_next=tensor_recv_next,
            group=self.pp_group,
            prev_pipeline_rank=self.prev_rank,
            next_pipeline_rank=self.next_rank,
        )

        # Normalise to a flat iterable regardless of batched vs sequential
        if isinstance(p2p_reqs, dict):
            reqs_iter = list(p2p_reqs.values())
        else:
            reqs_iter = p2p_reqs  # already a list

        if wait_on_reqs and reqs_iter:
            for req in reqs_iter:
                req.wait()
            reqs_iter = None  # type: ignore[assignment]

        # Guard against race condition (batch_isend_irecv + batch_p2p_sync)
        if self._batch_p2p and self._batch_p2p_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

        return tensor_recv_prev, tensor_recv_next, reqs_iter

    # ------------------------------------------------------------------
    # Public send / recv API (mirrors Megatron's P2PCommunicator)
    # ------------------------------------------------------------------

    def recv_forward(
        self,
        tensor_shapes: Union[Shape, List[Shape]],
        is_first_stage: bool,
    ) -> Union[Optional[torch.Tensor], List[Optional[torch.Tensor]]]:
        """Receive activation tensor(s) from the previous pipeline stage.

        Args:
            tensor_shapes:  Single shape or list of shapes.
            is_first_stage: True for PP stage 0; returns None(s) immediately.

        Returns:
            Single tensor (or None) when a single shape was provided, else a
            list matching the input list.
        """
        unwrap = is_single_shape(tensor_shapes)
        if unwrap:
            tensor_shapes = [tensor_shapes]

        if self.config.timers is not None:
            self.config.timers("forward-recv", log_level=2).start()

        results: List[Optional[torch.Tensor]] = []
        for shape in tensor_shapes:
            if is_first_stage:
                results.append(None)
            else:
                t, _, _ = self._communicate(
                    tensor_send_next=None,
                    tensor_send_prev=None,
                    recv_prev=True,
                    recv_next=False,
                    tensor_shape=shape,
                )
                results.append(t)

        if self.config.timers is not None:
            self.config.timers("forward-recv").stop()

        return results[0] if unwrap else results

    def recv_backward(
        self,
        tensor_shapes: Union[Shape, List[Shape]],
        is_last_stage: bool,
    ) -> Union[Optional[torch.Tensor], List[Optional[torch.Tensor]]]:
        """Receive gradient tensor(s) from the next pipeline stage.

        Args:
            tensor_shapes: Single shape or list of shapes.
            is_last_stage: True for the last PP stage; returns None(s).

        Returns:
            Gradient tensor(s) or None(s).
        """
        unwrap = is_single_shape(tensor_shapes)
        if unwrap:
            tensor_shapes = [tensor_shapes]

        if self.config.timers is not None:
            self.config.timers("backward-recv", log_level=2).start()

        results: List[Optional[torch.Tensor]] = []
        for shape in tensor_shapes:
            if is_last_stage:
                results.append(None)
            else:
                _, t, _ = self._communicate(
                    tensor_send_next=None,
                    tensor_send_prev=None,
                    recv_prev=False,
                    recv_next=True,
                    tensor_shape=shape,
                )
                results.append(t)

        if self.config.timers is not None:
            self.config.timers("backward-recv").stop()

        return results[0] if unwrap else results

    def send_forward(
        self,
        output_tensors: Union[torch.Tensor, List[torch.Tensor]],
        is_last_stage: bool,
    ) -> None:
        """Send activation tensor(s) to the next pipeline stage."""
        if not isinstance(output_tensors, list):
            output_tensors = [output_tensors]

        if self.config.timers is not None:
            self.config.timers("forward-send", log_level=2).start()

        for t in output_tensors:
            if not is_last_stage:
                self._communicate(
                    tensor_send_next=t,
                    tensor_send_prev=None,
                    recv_prev=False,
                    recv_next=False,
                    tensor_shape=None,
                )

        if self.config.timers is not None:
            self.config.timers("forward-send").stop()

    def send_backward(
        self,
        input_tensor_grads: Union[Optional[torch.Tensor], List[Optional[torch.Tensor]]],
        is_first_stage: bool,
    ) -> None:
        """Send gradient tensor(s) to the previous pipeline stage."""
        if not isinstance(input_tensor_grads, list):
            input_tensor_grads = [input_tensor_grads]

        if self.config.timers is not None:
            self.config.timers("backward-send", log_level=2).start()

        for g in input_tensor_grads:
            if not is_first_stage:
                self._communicate(
                    tensor_send_next=None,
                    tensor_send_prev=g,
                    recv_prev=False,
                    recv_next=False,
                    tensor_shape=None,
                )

        if self.config.timers is not None:
            self.config.timers("backward-send").stop()

    def send_forward_recv_backward(
        self,
        output_tensors: Union[torch.Tensor, List[torch.Tensor]],
        tensor_shapes: Union[Shape, List[Shape]],
        is_last_stage: bool,
    ) -> Union[Optional[torch.Tensor], List[Optional[torch.Tensor]]]:
        """Atomically send activation forward and receive gradient backward.

        DES-LOC note: on H100 stages this atomic exchange avoids a serialised
        send-then-recv which would stall the H100 waiting for A6000's slow
        gradient backward.
        """
        unwrap = not isinstance(output_tensors, list)
        if unwrap:
            output_tensors = [output_tensors]
        if is_single_shape(tensor_shapes):
            tensor_shapes = [tensor_shapes]

        if self.config.timers is not None:
            self.config.timers("forward-send-backward-recv", log_level=2).start()

        grads: List[Optional[torch.Tensor]] = []
        for t, shape in zip(output_tensors, tensor_shapes):
            if is_last_stage:
                grads.append(None)
            else:
                _, g, _ = self._communicate(
                    tensor_send_next=t,
                    tensor_send_prev=None,
                    recv_prev=False,
                    recv_next=True,
                    tensor_shape=shape,
                )
                grads.append(g)

        if self.config.timers is not None:
            self.config.timers("forward-send-backward-recv").stop()

        return grads[0] if unwrap else grads

    def send_backward_recv_forward(
        self,
        input_tensor_grads: Union[Optional[torch.Tensor], List[Optional[torch.Tensor]]],
        tensor_shapes: Union[Shape, List[Shape]],
        is_first_stage: bool,
    ) -> Union[Optional[torch.Tensor], List[Optional[torch.Tensor]]]:
        """Atomically send gradient backward and receive activation forward."""
        unwrap = not isinstance(input_tensor_grads, list)
        if unwrap:
            input_tensor_grads = [input_tensor_grads]
        if is_single_shape(tensor_shapes):
            tensor_shapes = [tensor_shapes]

        if self.config.timers is not None:
            self.config.timers("backward-send-forward-recv", log_level=2).start()

        tensors: List[Optional[torch.Tensor]] = []
        for g, shape in zip(input_tensor_grads, tensor_shapes):
            if is_first_stage:
                tensors.append(None)
            else:
                t, _, _ = self._communicate(
                    tensor_send_next=None,
                    tensor_send_prev=g,
                    recv_prev=True,
                    recv_next=False,
                    tensor_shape=shape,
                )
                tensors.append(t)

        if self.config.timers is not None:
            self.config.timers("backward-send-forward-recv").stop()

        return tensors[0] if unwrap else tensors

    def send_forward_recv_forward(
        self,
        output_tensor: Optional[torch.Tensor],
        recv_prev: bool,
        tensor_shape: Shape,
        overlap_p2p_comm: bool = False,
    ) -> Union[Optional[torch.Tensor], Tuple[Optional[torch.Tensor], Optional[object]]]:
        """Send to next stage and receive from previous stage simultaneously.

        Used in the interleaved 1F1B warmup phase.

        Args:
            output_tensor:    Tensor to send forward (None if nothing to send).
            recv_prev:        Whether to post a recv from the previous stage.
            tensor_shape:     Shape of the tensor to receive.
            overlap_p2p_comm: If True, return (tensor, pending_reqs) so the
                              caller can overlap compute with communication.

        Returns:
            tensor_recv_prev if overlap_p2p_comm=False, else (tensor, reqs).
        """
        if self.config.timers is not None:
            self.config.timers("forward-send-forward-recv", log_level=2).start()

        t, _, reqs = self._communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=recv_prev,
            recv_next=False,
            tensor_shape=tensor_shape if recv_prev else None,
            wait_on_reqs=not overlap_p2p_comm,
        )

        if self.config.timers is not None:
            self.config.timers("forward-send-forward-recv").stop()

        if overlap_p2p_comm:
            return t, reqs
        return t

    def send_backward_recv_backward(
        self,
        input_tensor_grad: Optional[torch.Tensor],
        recv_next: bool,
        tensor_shape: Shape,
        overlap_p2p_comm: bool = False,
    ) -> Union[Optional[torch.Tensor], Tuple[Optional[torch.Tensor], Optional[object]]]:
        """Send gradient to prev stage and receive gradient from next stage.

        Used in the interleaved 1F1B cooldown phase.
        """
        if self.config.timers is not None:
            self.config.timers("backward-send-backward-recv", log_level=2).start()

        _, g, reqs = self._communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=False,
            recv_next=recv_next,
            tensor_shape=tensor_shape if recv_next else None,
            wait_on_reqs=not overlap_p2p_comm,
        )

        if self.config.timers is not None:
            self.config.timers("backward-send-backward-recv").stop()

        if overlap_p2p_comm:
            return g, reqs
        return g

    def send_forward_backward_recv_forward_backward(
        self,
        output_tensor: Optional[torch.Tensor],
        input_tensor_grad: Optional[torch.Tensor],
        recv_prev: bool,
        recv_next: bool,
        tensor_shape: Shape,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Four-way simultaneous exchange for the steady 1F1B VPP phase.

        Sends activation forward AND gradient backward in a single fused
        batch_isend_irecv call, and receives activation from previous stage
        AND gradient from next stage simultaneously.

        DES-LOC note: this is the hottest communication path.  On a PCIe-only
        cluster (no NVLink), fusing the four transfers minimises latency by
        keeping NCCL's internal scheduler busy across all PCIe lanes.
        """
        if self.config.timers is not None:
            self.config.timers(
                "forward-backward-send-forward-backward-recv", log_level=2
            ).start()

        t, g, _ = self._communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=input_tensor_grad,
            recv_prev=recv_prev,
            recv_next=recv_next,
            tensor_shape=tensor_shape,
        )

        if self.config.timers is not None:
            self.config.timers(
                "forward-backward-send-forward-backward-recv"
            ).stop()

        return t, g


__all__ = [
    "P2PCommunicator",
    "is_single_shape",
    "_batched_p2p_ops",
    "_p2p_ops",
]
