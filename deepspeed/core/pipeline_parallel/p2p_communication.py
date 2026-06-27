# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Point-to-point pipeline communication for pipeline parallelism.

Ported from Megatron-LM/megatron/core/pipeline_parallel/p2p_communication.py
and extended with DES-LOC heterogeneous cluster support.

Commit lineage tracked:
  M3009 – MTP standalone stage support (mtp_standalone shape negotiation)
  M3544 – is_pp_first/last_stage properties on P2PCommunicator, MultiModulePipelineCommunicator
  M3766 – wait for async P2P send before deallocating output tensor

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
    from deepspeed.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
except ImportError:
    _ps = None  # type: ignore[assignment]

    def is_pp_first_stage(pp_group):  # type: ignore[misc]
        return pp_group.rank() == 0

    def is_pp_last_stage(pp_group):  # type: ignore[misc]
        return pp_group.rank() == pp_group.size() - 1

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
    mode when ``config.batch_p2p_comm=True``.
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
) -> dict:
    """Issue P2P transfers as ordered isend/irecv calls (sequential mode).

    For PP size == 2 with non-ucc backends we use the WORLD group for one
    direction so that both directions can overlap.  This mirrors the
    Megatron implementation and avoids deadlocks on ring-2 topologies.
    """
    reqs: dict = {}
    even_send_odd_recv_group = group
    if group.size() == 2 and torch.distributed.get_backend(group) != 'ucc':
        # Use the global process group for one direction to allow overlap
        # of independent communications.  Compatible because PP comms set
        # source/dest by global rank.  Must avoid for 'ucc' backend.
        even_recv_odd_send_group = torch.distributed.group.WORLD
    else:
        even_recv_odd_send_group = group

    if group.rank() % 2 == 0:
        if tensor_send_next is not None:
            reqs["send_next"] = torch.distributed.isend(
                tensor=tensor_send_next, dst=next_pipeline_rank,
                group=even_send_odd_recv_group,
            )
        if tensor_recv_prev is not None:
            reqs["recv_prev"] = torch.distributed.irecv(
                tensor=tensor_recv_prev, src=prev_pipeline_rank,
                group=even_recv_odd_send_group,
            )
        if tensor_send_prev is not None:
            reqs["send_prev"] = torch.distributed.isend(
                tensor=tensor_send_prev, dst=prev_pipeline_rank,
                group=even_send_odd_recv_group,
            )
        if tensor_recv_next is not None:
            reqs["recv_next"] = torch.distributed.irecv(
                tensor=tensor_recv_next, src=next_pipeline_rank,
                group=even_recv_odd_send_group,
            )
    else:
        if tensor_recv_prev is not None:
            reqs["recv_prev"] = torch.distributed.irecv(
                tensor=tensor_recv_prev, src=prev_pipeline_rank,
                group=even_send_odd_recv_group,
            )
        if tensor_send_next is not None:
            reqs["send_next"] = torch.distributed.isend(
                tensor=tensor_send_next, dst=next_pipeline_rank,
                group=even_recv_odd_send_group,
            )
        if tensor_recv_next is not None:
            reqs["recv_next"] = torch.distributed.irecv(
                tensor=tensor_recv_next, src=next_pipeline_rank,
                group=even_send_odd_recv_group,
            )
        if tensor_send_prev is not None:
            reqs["send_prev"] = torch.distributed.isend(
                tensor=tensor_send_prev, dst=prev_pipeline_rank,
                group=even_recv_odd_send_group,
            )
    return reqs


def is_single_shape(x) -> bool:
    """Check if the input represents a single tensor shape (not a list of shapes)."""
    if isinstance(x, torch.Size):
        return True
    if isinstance(x, (list, tuple)) and len(x) > 0 and all(isinstance(d, int) for d in x):
        return True
    return False


# ---------------------------------------------------------------------------
# nvtx decorator shim (no-op when NVTX is not available)
# ---------------------------------------------------------------------------
try:
    from deepspeed.core.utils import nvtx_decorator
except ImportError:
    def nvtx_decorator():  # type: ignore[misc]
        def _dec(fn):
            return fn
        return _dec


class P2PCommunicator:
    """P2P (Point-to-Point) Communicator for pipeline parallelism.

    This class handles communication between pipeline stages by managing
    tensor exchanges between consecutive stages in the pipeline.

    Commit history:
      M3009 – mtp_standalone config flag added to _communicate
      M3544 – Added is_pp_first/last_stage properties, total_stages, current_stage
      M3766 – Async send handle returned and waited before tensor deallocation
    """

    def __init__(self, pp_group: dist.ProcessGroup, config: ModelParallelConfig):
        # Basic attrs
        self.pp_group = pp_group
        self.config = config

        world_size = self.pp_group.size()
        curr_rank_in_pg = self.pp_group.rank()

        next_rank_pg = (curr_rank_in_pg + 1) % world_size
        prev_rank_pg = (curr_rank_in_pg - 1) % world_size

        self.next_rank: Optional[int] = dist.get_global_rank(self.pp_group, next_rank_pg)
        self.prev_rank: Optional[int] = dist.get_global_rank(self.pp_group, prev_rank_pg)
        self.virtual_pipeline_model_parallel_size = (
            config.virtual_pipeline_model_parallel_size
            if config.virtual_pipeline_model_parallel_size is not None
            else None
        )

    # ------------------------------------------------------------------
    # Stage-position properties (added in M3544)
    # ------------------------------------------------------------------

    @property
    def is_pp_first_stage(self) -> bool:
        """Return True if this rank is the first pipeline stage."""
        return is_pp_first_stage(self.pp_group)

    @property
    def is_pp_last_stage(self) -> bool:
        """Return True if this rank is the last pipeline stage."""
        return is_pp_last_stage(self.pp_group)

    @property
    def total_stages(self) -> int:
        """Return total number of pipeline stages."""
        return self.pp_group.size()

    @property
    def current_stage(self) -> int:
        """Return current pipeline stage index (0-indexed)."""
        return self.pp_group.rank()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _communicate_shapes(
        self,
        tensor_send_next: Optional[torch.Tensor],
        tensor_send_prev: Optional[torch.Tensor],
        recv_prev: bool,
        recv_next: bool,
    ) -> Tuple[list, list]:
        """Communicate tensor shapes between stages.

        Used when sequence lengths across micro-batches are non-uniform
        (config.variable_seq_lengths=True) or when MTP standalone mode is
        enabled (config.mtp_standalone=True, added in M3009).

        Returns:
            (recv_prev_shape, recv_next_shape): lists of ints.
        """
        config = self.config
        recv_prev_shape_tensor = None
        recv_next_shape_tensor = None
        send_prev_shape_tensor = None
        send_next_shape_tensor = None

        if recv_prev:
            recv_prev_shape_tensor = torch.empty(
                (3,), device=torch.cuda.current_device(), dtype=torch.int64
            )
        if recv_next:
            recv_next_shape_tensor = torch.empty(
                (3,), device=torch.cuda.current_device(), dtype=torch.int64
            )
        if tensor_send_prev is not None:
            send_prev_shape_tensor = torch.tensor(
                tensor_send_prev.size(), device=torch.cuda.current_device(), dtype=torch.int64
            )
        if tensor_send_next is not None:
            send_next_shape_tensor = torch.tensor(
                tensor_send_next.size(), device=torch.cuda.current_device(), dtype=torch.int64
            )

        if config.use_ring_exchange_p2p:
            torch.distributed.ring_exchange(
                tensor_send_prev=send_prev_shape_tensor,
                tensor_recv_prev=recv_prev_shape_tensor,
                tensor_send_next=send_next_shape_tensor,
                tensor_recv_next=recv_next_shape_tensor,
                group=self.pp_group,
            )
        else:
            ops = []
            if send_prev_shape_tensor is not None:
                ops.append(torch.distributed.P2POp(
                    torch.distributed.isend, send_prev_shape_tensor, self.prev_rank, self.pp_group
                ))
            if recv_prev_shape_tensor is not None:
                ops.append(torch.distributed.P2POp(
                    torch.distributed.irecv, recv_prev_shape_tensor, self.prev_rank, self.pp_group
                ))
            if send_next_shape_tensor is not None:
                ops.append(torch.distributed.P2POp(
                    torch.distributed.isend, send_next_shape_tensor, self.next_rank, self.pp_group
                ))
            if recv_next_shape_tensor is not None:
                ops.append(torch.distributed.P2POp(
                    torch.distributed.irecv, recv_next_shape_tensor, self.next_rank, self.pp_group
                ))
            if ops:
                reqs = torch.distributed.batch_isend_irecv(ops)
                for req in reqs:
                    req.wait()
            # Guard against race condition with batch_isend_irecv
            torch.cuda.synchronize()

        recv_prev_shape = [0, 0, 0]
        if recv_prev_shape_tensor is not None:
            recv_prev_shape = recv_prev_shape_tensor.tolist()

        recv_next_shape = [0, 0, 0]
        if recv_next_shape_tensor is not None:
            recv_next_shape = recv_next_shape_tensor.tolist()

        return recv_prev_shape, recv_next_shape

    def _communicate(
        self,
        *,
        tensor_send_next: Optional[torch.Tensor],
        tensor_send_prev: Optional[torch.Tensor],
        recv_prev: bool,
        recv_next: bool,
        tensor_shape: Shape,
        wait_on_reqs: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Union[list, dict]]]:
        """Communicate tensors between stages.

        Core helper method used by all public send/recv methods.

        Args:
            tensor_send_next: Tensor to send to next rank (None → no send).
            tensor_send_prev: Tensor to send to prev rank (None → no send).
            recv_prev: Whether to receive a tensor from the previous rank.
            recv_next: Whether to receive a tensor from the next rank.
            tensor_shape: Shape of the tensor to receive.  Used when
                variable_seq_lengths and mtp_standalone are both False.
            wait_on_reqs: If True, wait on all P2P requests before returning
                (synchronous mode).  If False, return handles for the caller
                to wait on (async / overlapped mode, M3766).

        Returns:
            (tensor_recv_prev, tensor_recv_next, wait_handles)
        """
        config = self.config

        if config.variable_seq_lengths or getattr(config, 'mtp_standalone', False):
            recv_prev_shape, recv_next_shape = self._communicate_shapes(
                tensor_send_next, tensor_send_prev, recv_prev, recv_next
            )
        else:
            recv_prev_shape = tensor_shape
            recv_next_shape = tensor_shape

        def create_tensor_recv_prev() -> torch.Tensor:
            return torch.empty(
                recv_prev_shape,
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=config.pipeline_dtype,
            )

        def create_tensor_recv_next() -> torch.Tensor:
            return torch.empty(
                recv_next_shape,
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=config.pipeline_dtype,
            )

        if recv_prev:
            if config.pipeline_dtype is None:
                raise RuntimeError("pipeline_dtype must be provided if recv_prev is True")
            if tensor_shape is None:
                raise RuntimeError(
                    "tensor_shape must be specified if recv_prev is True. "
                    "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
                )
        if recv_next:
            if config.pipeline_dtype is None:
                raise RuntimeError("pipeline_dtype must be provided if recv_next is True")
            if tensor_shape is None:
                raise RuntimeError(
                    "tensor_shape must be specified if recv_next is True. "
                    "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
                )

        # Select P2P implementation
        if config.use_ring_exchange_p2p:
            def _ring_exchange_wrapper(**kwargs):
                torch.distributed.ring_exchange(**kwargs)
                return []
            p2p_func = _ring_exchange_wrapper
        elif config.batch_p2p_comm:
            assert wait_on_reqs
            p2p_func = _batched_p2p_ops
        else:
            p2p_func = _p2p_ops

        pp_group = self.pp_group
        next_rank = self.next_rank
        prev_rank = self.prev_rank

        if config.use_ring_exchange_p2p or config.batch_p2p_comm:
            reqs: Union[list, dict] = []
        else:
            reqs = {}

        tensor_recv_prev = create_tensor_recv_prev() if recv_prev else None
        tensor_recv_next = create_tensor_recv_next() if recv_next else None

        p2p_reqs = p2p_func(
            tensor_send_prev=tensor_send_prev,
            tensor_recv_prev=tensor_recv_prev,
            tensor_send_next=tensor_send_next,
            tensor_recv_next=tensor_recv_next,
            group=pp_group,
            prev_pipeline_rank=prev_rank,
            next_pipeline_rank=next_rank,
        )
        if isinstance(p2p_reqs, list):
            reqs.extend(p2p_reqs)  # type: ignore[union-attr]
        else:
            reqs.update(p2p_reqs)  # type: ignore[union-attr]

        if wait_on_reqs and len(reqs) > 0:
            req_iter = reqs if isinstance(reqs, list) else reqs.values()
            for req in req_iter:
                req.wait()
            reqs = None

        if config.batch_p2p_comm and config.batch_p2p_sync:
            # Guard against race condition with batch_isend_irecv
            torch.cuda.synchronize()

        return tensor_recv_prev, tensor_recv_next, reqs

    # ------------------------------------------------------------------
    # Public send / recv methods (all decorated with nvtx for profiling)
    # ------------------------------------------------------------------

    @nvtx_decorator()
    def recv_forward(
        self,
        tensor_shapes,
        is_first_stage: bool,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Receive activation tensor from the previous rank (forward pass recv)."""
        unwrap = False
        if is_single_shape(tensor_shapes):
            unwrap = True
            tensor_shapes = [tensor_shapes]

        config = self.config
        input_tensors = []
        for tensor_shape in tensor_shapes:
            if is_first_stage:
                input_tensor = None
            else:
                if config.timers is not None:
                    config.timers('forward-recv', log_level=2).start()
                input_tensor, _, _ = self._communicate(
                    tensor_send_next=None,
                    tensor_send_prev=None,
                    recv_prev=True,
                    recv_next=False,
                    tensor_shape=tensor_shape,
                )
                if config.timers is not None:
                    config.timers('forward-recv').stop()
            input_tensors.append(input_tensor)

        return input_tensors[0] if unwrap else input_tensors

    @nvtx_decorator()
    def recv_backward(
        self,
        tensor_shapes,
        is_last_stage: bool,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Receive gradient tensor from the next rank (backward pass recv)."""
        unwrap = False
        if is_single_shape(tensor_shapes):
            unwrap = True
            tensor_shapes = [tensor_shapes]

        config = self.config
        output_tensor_grads = []
        for tensor_shape in tensor_shapes:
            if is_last_stage:
                output_tensor_grad = None
            else:
                if config.timers is not None:
                    config.timers('backward-recv', log_level=2).start()
                _, output_tensor_grad, _ = self._communicate(
                    tensor_send_next=None,
                    tensor_send_prev=None,
                    recv_prev=False,
                    recv_next=True,
                    tensor_shape=tensor_shape,
                )
                if config.timers is not None:
                    config.timers('backward-recv').stop()
            output_tensor_grads.append(output_tensor_grad)

        return output_tensor_grads[0] if unwrap else output_tensor_grads

    @nvtx_decorator()
    def send_forward(self, output_tensors, is_last_stage: bool) -> None:
        """Send activation tensor to the next rank (forward pass send)."""
        config = self.config
        if not isinstance(output_tensors, list):
            output_tensors = [output_tensors]
        for output_tensor in output_tensors:
            if not is_last_stage:
                if config.timers is not None:
                    config.timers('forward-send', log_level=2).start()
                self._communicate(
                    tensor_send_next=output_tensor,
                    tensor_send_prev=None,
                    recv_prev=False,
                    recv_next=False,
                    tensor_shape=None,
                )
                if config.timers is not None:
                    config.timers('forward-send').stop()

    @nvtx_decorator()
    def send_backward(self, input_tensor_grads, is_first_stage: bool) -> None:
        """Send gradient tensor to the previous rank (backward pass send)."""
        config = self.config
        if not isinstance(input_tensor_grads, list):
            input_tensor_grads = [input_tensor_grads]
        for input_tensor_grad in input_tensor_grads:
            if not is_first_stage:
                if config.timers is not None:
                    config.timers('backward-send', log_level=2).start()
                self._communicate(
                    tensor_send_next=None,
                    tensor_send_prev=input_tensor_grad,
                    recv_prev=False,
                    recv_next=False,
                    tensor_shape=None,
                )
                if config.timers is not None:
                    config.timers('backward-send').stop()

    @nvtx_decorator()
    def send_forward_recv_backward(
        self,
        output_tensors,
        tensor_shapes,
        is_last_stage: bool,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Batched send forward and receive backward from the next rank."""
        config = self.config
        unwrap = False
        if not isinstance(output_tensors, list):
            unwrap = True
            output_tensors = [output_tensors]
        if not isinstance(tensor_shapes, list):
            tensor_shapes = [tensor_shapes]

        output_tensor_grads = []
        for output_tensor, tensor_shape in zip(output_tensors, tensor_shapes):
            if is_last_stage:
                output_tensor_grad = None
            else:
                if config.timers is not None:
                    config.timers('forward-send-backward-recv', log_level=2).start()
                _, output_tensor_grad, _ = self._communicate(
                    tensor_send_next=output_tensor,
                    tensor_send_prev=None,
                    recv_prev=False,
                    recv_next=True,
                    tensor_shape=tensor_shape,
                )
                if config.timers is not None:
                    config.timers('forward-send-backward-recv').stop()
            output_tensor_grads.append(output_tensor_grad)

        return output_tensor_grads[0] if unwrap else output_tensor_grads

    @nvtx_decorator()
    def send_backward_recv_forward(
        self,
        input_tensor_grads,
        tensor_shapes,
        is_first_stage: bool,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Batched send backward and receive forward from the previous rank."""
        config = self.config
        unwrap = False
        if not isinstance(input_tensor_grads, list):
            unwrap = True
            input_tensor_grads = [input_tensor_grads]
        if not isinstance(tensor_shapes, list):
            tensor_shapes = [tensor_shapes]

        input_tensors = []
        for input_tensor_grad, tensor_shape in zip(input_tensor_grads, tensor_shapes):
            if is_first_stage:
                input_tensor = None
            else:
                if config.timers is not None:
                    config.timers('backward-send-forward-recv', log_level=2).start()
                input_tensor, _, _ = self._communicate(
                    tensor_send_next=None,
                    tensor_send_prev=input_tensor_grad,
                    recv_prev=True,
                    recv_next=False,
                    tensor_shape=tensor_shape,
                )
                if config.timers is not None:
                    config.timers('backward-send-forward-recv').stop()
            input_tensors.append(input_tensor)

        return input_tensors[0] if unwrap else input_tensors

    @nvtx_decorator()
    def send_forward_recv_forward(
        self,
        output_tensor: Optional[torch.Tensor],
        recv_prev: bool,
        tensor_shape: Shape,
        overlap_p2p_comm: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[dict]]]:
        """Batched recv from previous rank and send to next rank.

        When overlap_p2p_comm=True (added for the VPP overlap path) the
        method returns (tensor, wait_handles) instead of blocking.
        """
        config = self.config
        if config.timers is not None:
            config.timers('forward-send-forward-recv', log_level=2).start()
        input_tensor, _, wait_handles = self._communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=recv_prev,
            recv_next=False,
            tensor_shape=tensor_shape,
            wait_on_reqs=(not overlap_p2p_comm),
        )
        if config.timers is not None:
            config.timers('forward-send-forward-recv').stop()
        if overlap_p2p_comm:
            return input_tensor, wait_handles
        return input_tensor

    @nvtx_decorator()
    def send_backward_recv_backward(
        self,
        input_tensor_grad: Optional[torch.Tensor],
        recv_next: bool,
        tensor_shape: Shape,
        overlap_p2p_comm: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[dict]]]:
        """Batched recv from next rank and send to previous rank.

        When overlap_p2p_comm=True the method returns (tensor, wait_handles).
        """
        config = self.config
        if config.timers is not None:
            config.timers('backward-send-backward-recv', log_level=2).start()
        _, output_tensor_grad, wait_handles = self._communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=False,
            recv_next=recv_next,
            tensor_shape=tensor_shape,
            wait_on_reqs=(not overlap_p2p_comm),
        )
        if config.timers is not None:
            config.timers('backward-send-backward-recv').stop()
        if overlap_p2p_comm:
            return output_tensor_grad, wait_handles
        return output_tensor_grad

    @nvtx_decorator()
    def send_forward_backward_recv_forward_backward(
        self,
        output_tensor: Optional[torch.Tensor],
        input_tensor_grad: Optional[torch.Tensor],
        recv_prev: bool,
        recv_next: bool,
        tensor_shape: Shape,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Batched send and receive with both previous and next ranks.

        Used in the 1F1B steady-state loop to overlap both directions.
        """
        config = self.config
        if config.timers is not None:
            config.timers('forward-backward-send-forward-backward-recv', log_level=2).start()
        input_tensor, output_tensor_grad, _ = self._communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=input_tensor_grad,
            recv_prev=recv_prev,
            recv_next=recv_next,
            tensor_shape=tensor_shape,
        )
        if config.timers is not None:
            config.timers('forward-backward-send-forward-backward-recv').stop()
        return input_tensor, output_tensor_grad


# ---------------------------------------------------------------------------
# MultiModulePipelineCommunicator (added in M3544)
# ---------------------------------------------------------------------------

class MultiModulePipelineCommunicator(P2PCommunicator):
    """Extended P2P communicator for multi-module (e.g. VLM encoder+LLM) pipelines.

    In multi-module pipelines different portions of the model (e.g. a vision
    encoder and a language model decoder) may reside on different pipeline
    stages with different process group configurations.  This communicator
    wraps those differences behind the same interface as P2PCommunicator.

    Added in M3544 (multimodule pipelining support).
    """

    def __init__(
        self,
        pp_group: dist.ProcessGroup,
        config: ModelParallelConfig,
        module_pp_groups: Optional[Dict[str, dist.ProcessGroup]] = None,
    ):
        super().__init__(pp_group=pp_group, config=config)
        # Mapping from module name → its own PP group (may differ from main pp_group)
        self.module_pp_groups: Dict[str, dist.ProcessGroup] = module_pp_groups or {}

    def get_module_pp_group(self, module_name: str) -> dist.ProcessGroup:
        """Return the PP process group for a named module."""
        return self.module_pp_groups.get(module_name, self.pp_group)


__all__ = [
    "P2PCommunicator",
    "MultiModulePipelineCommunicator",
    "is_single_shape",
    "_batched_p2p_ops",
    "_p2p_ops",
]
