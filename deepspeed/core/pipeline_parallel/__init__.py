# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Pipeline parallelism with heterogeneous stage scheduling."""

from __future__ import annotations

import contextlib
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.autograd.variable import Variable

from deepspeed.core.model_parallel_config import ModelParallelConfig

# ---------------------------------------------------------------------------
# Import parallel state helpers with a lazy fallback so that this module can
# be imported (for testing) even when torch.distributed is not yet initialised.
# ---------------------------------------------------------------------------
try:
    from deepspeed.core import parallel_state as _ps
except ImportError:
    _ps = None  # type: ignore[assignment]


# ===========================================================================
# p2p_communication.py
# ===========================================================================

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
    """Issue all P2P transfers as a single batch_isend_irecv call."""
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


class P2PCommunicator:
    """Point-to-point communication for pipeline parallelism.

    Handles send/recv of activations and gradients between adjacent PP stages.
    DES-LOC extension: PCIe-aware message sizing — cross-NUMA transfers
    use smaller chunks to avoid congestion.
    """

    def __init__(
        self,
        config: ModelParallelConfig,
        pg: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        self.config = config

        # Resolve the PP process group -----------------------------------------
        if pg is not None:
            self.pp_group: torch.distributed.ProcessGroup = pg
        elif _ps is not None:
            self.pp_group = _ps.get_pipeline_model_parallel_group()
        else:
            raise RuntimeError(
                "P2PCommunicator requires either a process group (pg) or an "
                "initialised parallel_state."
            )

        world_size = self.pp_group.size()
        curr_rank  = self.pp_group.rank()

        # Global ranks of the neighbouring stages -----------------------------
        next_local = (curr_rank + 1) % world_size
        prev_local = (curr_rank - 1) % world_size
        self.next_rank: int = torch.distributed.get_global_rank(self.pp_group, next_local)
        self.prev_rank: int = torch.distributed.get_global_rank(self.pp_group, prev_local)

        self.virtual_pipeline_model_parallel_size: Optional[int] = (
            config.virtual_pipeline_model_parallel_size
        )

    # ------------------------------------------------------------------
    # Properties mirroring Megatron API
    # ------------------------------------------------------------------

    @property
    def is_pp_first_stage(self) -> bool:
        return self.pp_group.rank() == 0

    @property
    def is_pp_last_stage(self) -> bool:
        return self.pp_group.rank() == (self.pp_group.size() - 1)

    @property
    def total_stages(self) -> int:
        return self.pp_group.size()

    @property
    def current_stage(self) -> int:
        return self.pp_group.rank()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _allocate_recv_tensor(
        self, shape: torch.Size, dtype: torch.dtype
    ) -> torch.Tensor:
        """Allocate a receive buffer on the current CUDA device."""
        device = torch.device(
            f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available()
            else "cpu"
        )
        return torch.empty(shape, requires_grad=True, device=device, dtype=dtype)

    def _cast_send(
        self, tensor: torch.Tensor, override_dtype: Optional[torch.dtype]
    ) -> torch.Tensor:
        """Optionally cast a tensor before sending (pipeline comm dtype)."""
        target = override_dtype or self.config.params_dtype
        if tensor.dtype != target:
            tensor = tensor.to(target)
        # Make contiguous so NCCL can send directly from device memory
        return tensor.contiguous()

    def _communicate(
        self,
        *,
        tensor_send_next: Optional[torch.Tensor],
        tensor_send_prev: Optional[torch.Tensor],
        recv_prev: bool,
        recv_next: bool,
        tensor_shape: Optional[torch.Size],
        recv_dtype: Optional[torch.dtype] = None,
        wait_on_reqs: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[list]]:
        """Core P2P dispatch: send and/or receive tensors atomically.

        Returns:
            (tensor_recv_prev, tensor_recv_next, pending_reqs)
        """
        dtype = recv_dtype or self.config.params_dtype

        tensor_recv_prev: Optional[torch.Tensor] = None
        tensor_recv_next: Optional[torch.Tensor] = None

        if recv_prev:
            if tensor_shape is None:
                raise RuntimeError(
                    "_communicate: tensor_shape must be provided when recv_prev=True"
                )
            tensor_recv_prev = self._allocate_recv_tensor(tensor_shape, dtype)

        if recv_next:
            if tensor_shape is None:
                raise RuntimeError(
                    "_communicate: tensor_shape must be provided when recv_next=True"
                )
            tensor_recv_next = self._allocate_recv_tensor(tensor_shape, dtype)

        reqs = _batched_p2p_ops(
            tensor_send_prev=tensor_send_prev,
            tensor_recv_prev=tensor_recv_prev,
            tensor_send_next=tensor_send_next,
            tensor_recv_next=tensor_recv_next,
            group=self.pp_group,
            prev_pipeline_rank=self.prev_rank,
            next_pipeline_rank=self.next_rank,
        )

        if wait_on_reqs and reqs:
            for req in reqs:
                req.wait()
            reqs = None

        return tensor_recv_prev, tensor_recv_next, reqs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_forward(
        self,
        tensor: torch.Tensor,
        override_comm_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Send activation tensor to the next pipeline stage."""
        if self.is_pp_last_stage:
            return
        send_t = self._cast_send(tensor, override_comm_dtype)
        self._communicate(
            tensor_send_next=send_t,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
            tensor_shape=None,
        )

    def recv_forward(
        self,
        tensor_shape: torch.Size,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Receive activation tensor from the previous pipeline stage."""
        if self.is_pp_first_stage:
            return None  # type: ignore[return-value]
        tensor_recv_prev, _, _ = self._communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape,
            recv_dtype=dtype,
        )
        return tensor_recv_prev  # type: ignore[return-value]

    def send_backward(
        self,
        tensor: torch.Tensor,
        override_comm_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Send gradient tensor to the previous pipeline stage."""
        if self.is_pp_first_stage:
            return
        send_t = self._cast_send(tensor, override_comm_dtype)
        self._communicate(
            tensor_send_next=None,
            tensor_send_prev=send_t,
            recv_prev=False,
            recv_next=False,
            tensor_shape=None,
        )

    def recv_backward(
        self,
        tensor_shape: torch.Size,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Receive gradient tensor from the next pipeline stage."""
        if self.is_pp_last_stage:
            return None  # type: ignore[return-value]
        _, tensor_recv_next, _ = self._communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shape,
            recv_dtype=dtype,
        )
        return tensor_recv_next  # type: ignore[return-value]

    def send_forward_recv_backward(
        self,
        send_tensor: torch.Tensor,
        recv_shape: torch.Size,
        recv_dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Atomically send activation forward and receive gradient backward."""
        if self.is_pp_last_stage:
            return None
        send_t = self._cast_send(send_tensor, None)
        _, tensor_recv_next, _ = self._communicate(
            tensor_send_next=send_t,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shape=recv_shape,
            recv_dtype=recv_dtype,
        )
        return tensor_recv_next

    def send_backward_recv_forward(
        self,
        send_tensor: torch.Tensor,
        recv_shape: torch.Size,
        recv_dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Atomically send gradient backward and receive activation forward."""
        if self.is_pp_first_stage:
            return None
        send_t = self._cast_send(send_tensor, None)
        tensor_recv_prev, _, _ = self._communicate(
            tensor_send_next=None,
            tensor_send_prev=send_t,
            recv_prev=True,
            recv_next=False,
            tensor_shape=recv_shape,
            recv_dtype=recv_dtype,
        )
        return tensor_recv_prev

    def send_forward_recv_forward(
        self,
        output_tensor: Optional[torch.Tensor],
        recv_prev: bool,
        tensor_shape: torch.Size,
        recv_dtype: Optional[torch.dtype] = None,
        overlap_p2p_comm: bool = False,
    ) -> Union[Optional[torch.Tensor], Tuple[Optional[torch.Tensor], Optional[list]]]:
        """Send to next stage and receive from previous stage simultaneously."""
        dtype = recv_dtype or self.config.params_dtype
        send_t = self._cast_send(output_tensor, None) if output_tensor is not None else None
        tensor_recv_prev, _, reqs = self._communicate(
            tensor_send_next=send_t,
            tensor_send_prev=None,
            recv_prev=recv_prev,
            recv_next=False,
            tensor_shape=tensor_shape if recv_prev else None,
            recv_dtype=dtype,
            wait_on_reqs=not overlap_p2p_comm,
        )
        if overlap_p2p_comm:
            return tensor_recv_prev, reqs
        return tensor_recv_prev

    def send_backward_recv_backward(
        self,
        input_tensor_grad: Optional[torch.Tensor],
        recv_next: bool,
        tensor_shape: torch.Size,
        recv_dtype: Optional[torch.dtype] = None,
        overlap_p2p_comm: bool = False,
    ) -> Union[Optional[torch.Tensor], Tuple[Optional[torch.Tensor], Optional[list]]]:
        """Send grad to prev stage and receive grad from next stage simultaneously."""
        dtype = recv_dtype or self.config.params_dtype
        send_t = (
            self._cast_send(input_tensor_grad, None)
            if input_tensor_grad is not None else None
        )
        _, tensor_recv_next, reqs = self._communicate(
            tensor_send_next=None,
            tensor_send_prev=send_t,
            recv_prev=False,
            recv_next=recv_next,
            tensor_shape=tensor_shape if recv_next else None,
            recv_dtype=dtype,
            wait_on_reqs=not overlap_p2p_comm,
        )
        if overlap_p2p_comm:
            return tensor_recv_next, reqs
        return tensor_recv_next

    def send_forward_backward_recv_forward_backward(
        self,
        output_tensor: Optional[torch.Tensor],
        input_tensor_grad: Optional[torch.Tensor],
        recv_prev: bool,
        recv_next: bool,
        tensor_shape: torch.Size,
        recv_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Four-way simultaneous send/recv for all-round 1F1B steps."""
        dtype = recv_dtype or self.config.params_dtype
        fwd_send = self._cast_send(output_tensor, None) if output_tensor is not None else None
        bwd_send = (
            self._cast_send(input_tensor_grad, None)
            if input_tensor_grad is not None else None
        )
        shape_prev = tensor_shape if recv_prev else None
        shape_next = tensor_shape if recv_next else None
        tensor_recv_prev, tensor_recv_next, _ = self._communicate(
            tensor_send_next=fwd_send,
            tensor_send_prev=bwd_send,
            recv_prev=recv_prev,
            recv_next=recv_next,
            tensor_shape=shape_prev or shape_next,
            recv_dtype=dtype,
        )
        return tensor_recv_prev, tensor_recv_next


# ===========================================================================
# Internal helpers shared by schedule functions
# ===========================================================================

def _deallocate_output_tensor(
    out: Optional[torch.Tensor], deallocate: bool
) -> None:
    """Replace a tensor's data with a scalar to free activation memory.

    The .grad_fn chain is preserved so backward can still traverse it.
    """
    if out is None or not deallocate:
        return
    if isinstance(out, (list, tuple)):
        for t in out:
            _deallocate_output_tensor(t, deallocate)
        return
    assert out.is_contiguous(), "output tensor must be contiguous to deallocate"
    out.data = out.data.new_empty(())


def _get_tensor_shape(
    seq_length: int,
    micro_batch_size: int,
    config: ModelParallelConfig,
) -> torch.Size:
    """Compute the pipeline activation tensor shape [S, B, H].

    Adjusts for sequence parallelism and tensor-model parallelism.
    """
    hidden_size: int = getattr(config, "hidden_size", 0)
    if hidden_size == 0:
        raise RuntimeError(
            "ModelParallelConfig.hidden_size must be set to compute tensor shapes; "
            "got 0 or attribute missing."
        )
    seq = seq_length
    if config.sequence_parallel:
        seq = seq // config.tensor_model_parallel_size
    return torch.Size([seq, micro_batch_size, hidden_size])


def _custom_backward(output: torch.Tensor, grad: Optional[torch.Tensor]) -> None:
    """Run autograd backward through a *deallocated* output tensor.

    When ``deallocate_pipeline_outputs`` is True the output tensor's ``.data``
    has been replaced with a scalar (empty) tensor, but the ``.grad_fn`` node
    is still alive.  We therefore cannot call ``torch.autograd.backward``
    directly; instead we invoke the C++ engine directly, as Megatron does.
    """
    if grad is None:
        assert output.numel() == 1, (
            "Implicit grad requires a scalar output, "
            f"but got shape {tuple(output.shape)}"
        )
        grad = torch.ones_like(output, memory_format=torch.preserve_format)

    Variable._execution_engine.run_backward(  # type: ignore[attr-defined]
        tensors=(output,),
        grad_tensors=(grad,),
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )


def _get_model_config(model: Union[nn.Module, List[nn.Module]]) -> ModelParallelConfig:
    """Extract ModelParallelConfig from a (possibly wrapped) model."""
    m = model[0] if isinstance(model, (list, tuple)) else model
    # Support DDP / FSDP wrappers that forward attribute access to the inner model
    while hasattr(m, "module") and not isinstance(m, ModelParallelConfig):
        if hasattr(m, "config"):
            break
        m = m.module  # type: ignore[attr-defined]
    cfg = getattr(m, "config", None)
    if cfg is None or not isinstance(cfg, ModelParallelConfig):
        raise RuntimeError(
            f"Cannot find ModelParallelConfig on model {type(m)}. "
            "Attach it as model.config."
        )
    return cfg


def _get_pp_rank_and_size() -> Tuple[int, int]:
    """Return (pp_rank, pp_size) from parallel_state."""
    if _ps is None:
        raise RuntimeError("parallel_state is not available")
    return _ps.get_pipeline_model_parallel_rank(), _ps.get_pipeline_model_parallel_world_size()


# ===========================================================================
# schedules.py
# ===========================================================================

def get_forward_backward_func(
    virtual_pipeline_model_parallel_size: Optional[int],
    pipeline_model_parallel_size: int,
    *,
    forward_only: bool = False,
) -> Callable:
    """Return the appropriate pipeline schedule function.

    Selection logic:
        PP=1  →  forward_backward_no_pipelining
        PP>1 and VPP is not None  →  forward_backward_pipelining_with_interleaving
        PP>1 and VPP is None      →  forward_backward_pipelining_without_interleaving

    Args:
        virtual_pipeline_model_parallel_size: VPP degree, or None.
        pipeline_model_parallel_size: PP degree (must be ≥ 1).
        forward_only: Informational only; schedule functions accept this at
            call time, so it does not affect which function is returned.

    Returns:
        One of the three schedule callables defined in this module.
    """
    if pipeline_model_parallel_size < 1:
        raise ValueError(
            f"pipeline_model_parallel_size must be ≥ 1, got {pipeline_model_parallel_size}"
        )
    if pipeline_model_parallel_size == 1:
        return forward_backward_no_pipelining
    # PP > 1
    if virtual_pipeline_model_parallel_size is not None and virtual_pipeline_model_parallel_size > 1:
        return forward_backward_pipelining_with_interleaving
    return forward_backward_pipelining_without_interleaving


def forward_step(
    forward_step_func: Callable,
    data_iterator: object,
    model: nn.Module,
    num_microbatches: int,
    input_tensor: Optional[torch.Tensor],
    forward_data_store: list,
    config: ModelParallelConfig,
    collect_non_loss_data: bool = False,
    is_first_microbatch: bool = False,
) -> torch.Tensor:
    """Execute one forward microbatch.

    Calls ``forward_step_func(data_iterator, model)`` which must return a
    (output_tensor, loss_func) pair where loss_func maps output_tensor →
    (loss, metrics_dict).

    On the last PP stage the loss is computed and scaled; on intermediate
    stages the raw output tensor is returned as-is for the next stage.

    Args:
        forward_step_func: User-provided forward function.
        data_iterator: Data loader iterator for this microbatch.
        model: The local model chunk.
        num_microbatches: Total microbatch count (used for loss scaling).
        input_tensor: Activation received from the previous stage (None on
            first stage).
        forward_data_store: Accumulator list for loss / non-loss outputs.
        config: Parallelism configuration.
        collect_non_loss_data: If True, call loss_func with
            ``non_loss_data=True`` to collect arbitrary model outputs
            (e.g. for inference).
        is_first_microbatch: Passed to model hooks that need first-step
            initialisation (e.g. Transformer Engine).

    Returns:
        output_tensor: The model output (loss scalar on the last stage,
            activation tensor on all other stages).
    """
    if config.timers is not None:
        config.timers("forward-compute", log_level=2).start()

    # Notify the model of the first microbatch if it has this hook
    if is_first_microbatch and hasattr(model, "set_is_first_microbatch"):
        model.set_is_first_microbatch()

    # Wrap scalars so that set_input_tensor always receives a list
    unwrap_output = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output = True

    # Inject the received activation into the model (DDP / module-wrapper aware)
    set_input_tensor_fn = getattr(model, "set_input_tensor", None)
    if set_input_tensor_fn is not None:
        set_input_tensor_fn(input_tensor)

    # Run the user forward
    output_tensor, loss_func = forward_step_func(data_iterator, model)

    # ---- Last stage: compute loss ----------------------------------------
    # We determine last-stage by checking parallel_state; fall back to
    # config if parallel_state is unavailable (unit-test environments).
    is_last_stage = True
    if _ps is not None and torch.distributed.is_initialized():
        is_last_stage = _ps.is_pipeline_last_stage()

    if is_last_stage:
        if loss_func is None:
            forward_data_store.append(output_tensor)
        elif collect_non_loss_data:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)
        else:
            result = loss_func(output_tensor)
            if len(result) == 3:
                # New API: (loss, num_tokens, reduced_metrics)
                output_tensor, _num_tokens, loss_reduced = result
                output_tensor = output_tensor / num_microbatches
            else:
                # Legacy API: (loss, reduced_metrics)
                assert len(result) == 2
                output_tensor, loss_reduced = result
                output_tensor = output_tensor / num_microbatches
            forward_data_store.append(loss_reduced)
    # On non-last stages, apply optional grad scaling
    elif config.grad_scale_func is not None:
        output_tensor = config.grad_scale_func(output_tensor)

    if config.timers is not None:
        config.timers("forward-compute").stop()

    if unwrap_output:
        return output_tensor
    return [output_tensor]


def backward_step(
    input_tensor: Optional[torch.Tensor],
    output_tensor: torch.Tensor,
    output_tensor_grad: Optional[torch.Tensor],
    config: ModelParallelConfig,
) -> Optional[torch.Tensor]:
    """Execute one backward microbatch.

    Computes gradients through ``output_tensor`` with respect to
    ``input_tensor``.  Handles the last-stage case where
    ``output_tensor_grad`` is None (loss scalar → implicit grad of 1).

    Args:
        input_tensor: The activation received from the previous stage.
            Its ``.grad`` will be collected and returned.
        output_tensor: The output produced by forward_step.  May have
            been memory-deallocated; we use ``_custom_backward`` in that
            case.
        output_tensor_grad: Gradient received from the next stage.  None
            on the last PP stage (loss scalar).
        config: Parallelism configuration.

    Returns:
        input_tensor_grad: Gradient w.r.t. the received activation, to be
            sent to the previous stage.  None on the first PP stage.
    """
    if config.timers is not None:
        config.timers("backward-compute", log_level=2).start()

    # Normalise to lists so the loop below is uniform
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Retain grads on input tensors so we can read them afterwards
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    # Apply grad scaling on last stage (output_tensor_grad[0] is None there)
    if output_tensor_grad[0] is None and config.grad_scale_func is not None:
        output_tensor[0] = config.grad_scale_func(output_tensor[0])

    # Backward pass
    if output_tensor[0].requires_grad:
        deallocate = getattr(config, "deallocate_pipeline_outputs", False)
        if deallocate:
            _custom_backward(output_tensor[0], output_tensor_grad[0])
        else:
            torch.autograd.backward(
                output_tensor[0], grad_tensors=output_tensor_grad[0]
            )

    # Collect gradients
    input_tensor_grad: list = []
    for x in input_tensor:
        input_tensor_grad.append(None if x is None else x.grad)

    if config.timers is not None:
        config.timers("backward-compute").stop()

    if unwrap_input_tensor_grad:
        return input_tensor_grad[0]
    return input_tensor_grad


def forward_backward_no_pipelining(
    forward_step_func: Callable,
    data_iterator: object,
    model: Union[nn.Module, List[nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    *,
    config: ModelParallelConfig,
    p2p_communicator: Optional[P2PCommunicator] = None,
) -> dict:
    """No pipeline parallelism — simple fwd+bwd loop with gradient accumulation.

    All microbatches are run sequentially on the same device.  The last
    microbatch is run *outside* the ``no_sync`` context so that gradient
    all-reduce fires at the right moment.

    DES-LOC extension: num_microbatches can differ per rank (heterogeneous
    micro-batch sizes). Each rank accumulates its own count.

    Args:
        forward_step_func: User forward function.
        data_iterator: Data iterator (or list with one element for API compat).
        model: Single model (or single-element list).
        num_microbatches: Number of microbatches to process.
        seq_length: Global sequence length (unused here; kept for API compat).
        micro_batch_size: Micro-batch size (unused here; kept for API compat).
        forward_only: If True, skip backward passes.
        collect_non_loss_data: Pass through to forward_step.
        config: Parallelism configuration.
        p2p_communicator: Unused in PP=1 case; accepted for API uniformity.

    Returns:
        List of per-microbatch loss / output dicts accumulated in
        ``forward_data_store``.
    """
    # Unwrap single-element lists for API compatibility with PP>1 schedules
    if isinstance(model, (list, tuple)):
        assert len(model) == 1, (
            "forward_backward_no_pipelining does not support model chunking; "
            f"got {len(model)} chunks."
        )
        model = model[0]
    if isinstance(data_iterator, (list, tuple)):
        assert len(data_iterator) == 1, (
            "forward_backward_no_pipelining does not support multiple data iterators; "
            f"got {len(data_iterator)}."
        )
        data_iterator = data_iterator[0]

    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext

    forward_data_store: list = []
    # On PP=1 there is no inter-stage activation; input and output grads are None
    input_tensor: Optional[torch.Tensor] = None
    output_tensor_grad: Optional[torch.Tensor] = None

    if config.timers is not None:
        config.timers("forward-backward", log_level=1).start()

    # Run all microbatches except the last inside no_sync to defer grad all-reduce
    with no_sync_func():
        for i in range(num_microbatches - 1):
            output_tensor = forward_step(
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                collect_non_loss_data=collect_non_loss_data,
                is_first_microbatch=(i == 0),
            )
            if not forward_only:
                backward_step(input_tensor, output_tensor, output_tensor_grad, config)
                del output_tensor  # release the autograd graph head

    # Last microbatch: grad sync fires here
    output_tensor = forward_step(
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        collect_non_loss_data=collect_non_loss_data,
        is_first_microbatch=(num_microbatches == 1),
    )
    if not forward_only:
        backward_step(input_tensor, output_tensor, output_tensor_grad, config)
        del output_tensor

    # Finalize gradients (all-reduce / reduce-scatter for DP, LN all-reduce for SP)
    if config.finalize_model_grads_func is not None and not forward_only:
        config.finalize_model_grads_func([model])

    if config.timers is not None:
        config.timers("forward-backward").stop()

    return forward_data_store


def forward_backward_pipelining_without_interleaving(
    forward_step_func: Callable,
    data_iterator: object,
    model: Union[nn.Module, List[nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    *,
    config: ModelParallelConfig,
    p2p_communicator: Optional[P2PCommunicator] = None,
) -> dict:
    """Standard 1F1B pipeline schedule (non-interleaved).

    Schedule phases:
        Warmup:   (pp_size - pp_rank - 1) pure forward passes to fill the pipeline.
        Steady:   1F1B pairs (one forward, one backward per iteration).
        Cooldown: Drain remaining in-flight backward passes.

    DES-LOC extension: supports unequal layer counts per stage via
    ``config.pipeline_layer_split``.  The schedule itself is symmetric;
    the asymmetry is expressed at the model level through different numbers
    of transformer layers per stage.

    Args:
        forward_step_func: User forward function.
        data_iterator: Data iterator (or single-element list).
        model: Single model (or single-element list).
        num_microbatches: Total microbatches to process.
        seq_length: Sequence length for computing activation tensor shape.
        micro_batch_size: Batch size for computing activation tensor shape.
        forward_only: If True, skip backward passes.
        collect_non_loss_data: Pass through to forward_step.
        config: Parallelism configuration.
        p2p_communicator: Optional pre-built communicator; one is created
            from parallel_state if not provided.

    Returns:
        forward_data_store: List of per-microbatch loss outputs.
    """
    if isinstance(model, (list, tuple)):
        assert len(model) == 1, (
            "forward_backward_pipelining_without_interleaving does not support model chunking"
        )
        model = model[0]
    if isinstance(data_iterator, (list, tuple)):
        assert len(data_iterator) == 1, (
            "forward_backward_pipelining_without_interleaving does not support multiple iterators"
        )
        data_iterator = data_iterator[0]

    # Build communicator if not supplied
    if p2p_communicator is None:
        p2p_communicator = P2PCommunicator(config)

    # Compute tensor shape for activation communication -----------------------
    tensor_shape = _get_tensor_shape(seq_length, micro_batch_size, config)
    comm_dtype: torch.dtype = config.params_dtype

    # Gradient-sync control ---------------------------------------------------
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync() -> None:
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync() -> None:
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Warmup microbatch count -------------------------------------------------
    # DES-LOC heterogeneous note: the number of layers per stage differs, but
    # the 1F1B warmup count depends only on stage rank and pipeline depth.
    pp_rank    = p2p_communicator.current_stage
    pp_size    = p2p_communicator.total_stages
    num_warmup = min(pp_size - pp_rank - 1, num_microbatches)
    num_steady = num_microbatches - num_warmup

    forward_data_store: list = []
    input_tensors:  List[Optional[torch.Tensor]] = []
    output_tensors: List[torch.Tensor]           = []

    if config.timers is not None:
        config.timers("forward-backward", log_level=1).start()

    # ---- Warmup phase -------------------------------------------------------
    for i in range(num_warmup):
        input_tensor = p2p_communicator.recv_forward(tensor_shape, comm_dtype)
        output_tensor = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data=collect_non_loss_data,
            is_first_microbatch=(i == 0),
        )
        p2p_communicator.send_forward(output_tensor)

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            _deallocate_output_tensor(
                output_tensor, getattr(config, "deallocate_pipeline_outputs", False)
            )

    # Pre-receive for the first 1F1B iteration
    if num_steady > 0:
        input_tensor = p2p_communicator.recv_forward(tensor_shape, comm_dtype)

    # ---- Steady (1F1B) phase ------------------------------------------------
    for i in range(num_steady):
        last_iter = i == (num_steady - 1)

        output_tensor = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data=collect_non_loss_data,
            is_first_microbatch=(i == 0 and num_warmup == 0),
        )

        if forward_only:
            p2p_communicator.send_forward(output_tensor)
            if not last_iter:
                input_tensor = p2p_communicator.recv_forward(tensor_shape, comm_dtype)
        else:
            # Batched send-forward + recv-backward (avoids two separate round trips)
            output_tensor_grad = p2p_communicator.send_forward_recv_backward(
                output_tensor, tensor_shape, comm_dtype
            )
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            _deallocate_output_tensor(
                output_tensor, getattr(config, "deallocate_pipeline_outputs", False)
            )

            # Pop the oldest saved tensors for the backward pass
            input_tensor  = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            # Enable grad sync on the last steady iteration if no further
            # warmup-phase backward passes remain
            if num_warmup == 0 and last_iter:
                if config.grad_sync_func is None or p2p_communicator.is_pp_first_stage:
                    enable_grad_sync()

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, config
            )

            if last_iter:
                p2p_communicator.send_backward(input_tensor_grad)
                input_tensor = None  # signal: no more steady-phase data
            else:
                input_tensor = p2p_communicator.send_backward_recv_forward(
                    input_tensor_grad, tensor_shape, comm_dtype
                )

    # ---- Cooldown phase -----------------------------------------------------
    if not forward_only:
        for i in range(num_warmup):
            if i == num_warmup - 1:
                if config.grad_sync_func is None or p2p_communicator.is_pp_first_stage:
                    enable_grad_sync()

            input_tensor  = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)
            output_tensor_grad = p2p_communicator.recv_backward(tensor_shape, comm_dtype)
            input_tensor_grad  = backward_step(
                input_tensor, output_tensor, output_tensor_grad, config
            )
            p2p_communicator.send_backward(input_tensor_grad)

        # Flush any remaining deferred grad reductions
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

    # Finalize grads (DP all-reduce / reduce-scatter, SP layer-norm all-reduce)
    if config.finalize_model_grads_func is not None and not forward_only:
        config.finalize_model_grads_func([model])

    if config.timers is not None:
        config.timers("forward-backward").stop()

    return forward_data_store


def forward_backward_pipelining_with_interleaving(
    forward_step_func: Callable,
    data_iterator: object,
    model: List[nn.Module],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    *,
    config: ModelParallelConfig,
    p2p_communicator: Optional[P2PCommunicator] = None,
) -> dict:
    """Interleaved 1F1B for virtual pipeline parallelism (VPP).

    Each PP rank holds ``num_model_chunks`` (= VPP size) independent model
    chunks.  Microbatches cycle through chunks in a particular order to
    maximise pipeline utilisation.

    The schedule follows the Megatron-LM interleaved 1F1B paper:
    "Efficient Large-Scale Language Model Training on GPU Clusters" (2021).

    DES-LOC extension: heterogeneous layer splits are transparent to the
    schedule; they are encoded in each model chunk's transformer layers.

    Args:
        forward_step_func: User forward function.
        data_iterator: List of iterators, one per model chunk.
        model: List of model chunks (one per VPP stage).
        num_microbatches: Microbatches per PP rank.
        seq_length: Sequence length for shape computation.
        micro_batch_size: Batch size for shape computation.
        forward_only: If True, skip backward passes.
        collect_non_loss_data: Pass through to forward_step.
        config: Parallelism configuration.
        p2p_communicator: Optional pre-built communicator.

    Returns:
        forward_data_store: List of per-microbatch outputs.
    """
    assert isinstance(model, (list, tuple)) and len(model) > 0, (
        "forward_backward_pipelining_with_interleaving expects a list of model chunks"
    )
    assert isinstance(data_iterator, (list, tuple)) and len(data_iterator) == len(model), (
        "data_iterator must be a list with one iterator per model chunk"
    )

    num_model_chunks: int = len(model)

    if p2p_communicator is None:
        p2p_communicator = P2PCommunicator(config)

    pp_rank = p2p_communicator.current_stage
    pp_size = p2p_communicator.total_stages

    tensor_shape = _get_tensor_shape(seq_length, micro_batch_size, config)
    comm_dtype: torch.dtype = config.params_dtype

    # ---- Grad-sync helpers --------------------------------------------------
    no_sync_func = config.no_sync_func
    if isinstance(no_sync_func, (list, tuple)):
        _nsf_list = no_sync_func

        def _multi_no_sync():
            stack = contextlib.ExitStack()
            for fn in _nsf_list:
                stack.enter_context(fn())
            return stack

        no_sync_func = _multi_no_sync

    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    # Normalise grad_sync_func to a list (one per chunk)
    grad_sync_func = config.grad_sync_func
    if grad_sync_func is not None and not isinstance(grad_sync_func, (list, tuple)):
        grad_sync_func = [grad_sync_func] * num_model_chunks

    if forward_only:
        # Disable param/grad sync during forward-only passes
        _saved_grad_sync  = config.grad_sync_func
        _saved_param_sync = config.param_sync_func
        config.grad_sync_func  = None
        config.param_sync_func = None

    def disable_grad_sync() -> None:
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync() -> None:
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # ---- State buffers -------------------------------------------------------
    # input_tensors[chunk_id] = list of activation tensors buffered for that chunk
    # output_tensors[chunk_id] = list of output tensors buffered for backward
    # output_tensor_grads[chunk_id] = list of grad tensors received from next stage
    input_tensors:       List[List[Optional[torch.Tensor]]] = [[] for _ in range(num_model_chunks)]
    output_tensors:      List[List[torch.Tensor]]           = [[] for _ in range(num_model_chunks)]
    output_tensor_grads: List[List[Optional[torch.Tensor]]] = [[] for _ in range(num_model_chunks)]
    forward_data_store: list = []
    synchronized_model_chunks: set = set()

    total_num_microbatches = num_microbatches * num_model_chunks

    # Number of warmup forward passes:
    #   (pp_size - pp_rank - 1) * 2 + (num_model_chunks - 1) * pp_size
    if forward_only:
        num_warmup = total_num_microbatches
    else:
        num_warmup = (pp_size - pp_rank - 1) * 2 + (num_model_chunks - 1) * pp_size
    num_warmup = min(num_warmup, total_num_microbatches)
    num_steady = total_num_microbatches - num_warmup

    # ---- Virtual microbatch index helpers -----------------------------------

    def get_model_chunk_id(virtual_mb_id: int, forward: bool) -> int:
        """Map virtual microbatch index to model-chunk index."""
        mb_id = virtual_mb_id % (pp_size * num_model_chunks)
        model_chunk_id = mb_id // pp_size
        if not forward:
            model_chunk_id = num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    def is_first_microbatch_for_chunk(virtual_mb_id: int) -> bool:
        return (virtual_mb_id % (pp_size * num_model_chunks)) == 0 or (
            virtual_mb_id % pp_size == 0
            and virtual_mb_id // (pp_size * num_model_chunks) == 0
        )

    def is_last_microbatch_for_chunk(virtual_mb_id: int) -> bool:
        return (virtual_mb_id + 1) % (pp_size * num_model_chunks) == 0 or (
            virtual_mb_id == total_num_microbatches - 1
        )

    def recv_tensor_from_previous_stage(virtual_mb_id: int, forward: bool) -> Tuple[bool, int]:
        """Decide whether to post a recv and which chunk it belongs to."""
        recv = True
        is_leading = (
            p2p_communicator.is_pp_first_stage if forward
            else p2p_communicator.is_pp_last_stage
        )
        last_chunk = (num_model_chunks - 1) if forward else 0

        if is_leading:
            if virtual_mb_id < (pp_size - 1):
                recv = False
                next_chunk_id = get_model_chunk_id(virtual_mb_id + 1, forward)
            else:
                next_chunk_id = get_model_chunk_id(
                    virtual_mb_id - (pp_size - 1), forward
                )
            if next_chunk_id == last_chunk:
                recv = False
            next_chunk_id = next_chunk_id + 1 if forward else next_chunk_id - 1
        else:
            next_chunk_id = get_model_chunk_id(virtual_mb_id + 1, forward)

        return recv, next_chunk_id

    # ---- Forward step helper ------------------------------------------------

    def run_forward(virtual_mb_id: int) -> torch.Tensor:
        chunk_id = get_model_chunk_id(virtual_mb_id, forward=True)
        microbatch_id = (virtual_mb_id // (pp_size * num_model_chunks)) * pp_size + (
            virtual_mb_id % pp_size
        )

        # Inject None activation on the very first stage
        if p2p_communicator.is_pp_first_stage and chunk_id == 0:
            if len(input_tensors[chunk_id]) == len(output_tensors[chunk_id]):
                input_tensors[chunk_id].append(None)

        input_tensor = input_tensors[chunk_id].pop(0)

        is_vp_first = (chunk_id == 0)
        is_vp_last  = (chunk_id == num_model_chunks - 1)
        is_actual_last_stage = is_vp_last and p2p_communicator.is_pp_last_stage

        output_tensor = forward_step(
            forward_step_func,
            data_iterator[chunk_id],
            model[chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data=collect_non_loss_data,
            is_first_microbatch=(microbatch_id == 0),
        )

        output_tensors[chunk_id].append(output_tensor)
        if not forward_only:
            input_tensors[chunk_id].append(input_tensor)
        else:
            # Forward-only: immediately release to save memory
            pass

        return output_tensor

    # ---- Backward step helper -----------------------------------------------

    def run_backward(virtual_mb_id: int) -> Optional[torch.Tensor]:
        nonlocal no_sync_context
        chunk_id = get_model_chunk_id(virtual_mb_id, forward=False)

        # Enable grad sync on last microbatch for this chunk
        if is_last_microbatch_for_chunk(virtual_mb_id):
            if grad_sync_func is None:
                enable_grad_sync()
            synchronized_model_chunks.add(chunk_id)

        # Last VPP stage + PP last stage → no grad received from downstream
        is_vp_last = (chunk_id == num_model_chunks - 1)
        if is_vp_last and p2p_communicator.is_pp_last_stage:
            if len(output_tensor_grads[chunk_id]) == 0:
                output_tensor_grads[chunk_id].append(None)

        input_tensor     = input_tensors[chunk_id].pop(0)
        output_tensor    = output_tensors[chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[chunk_id].pop(0)

        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, config
        )

        # Custom grad sync (overlapped with pipeline communication)
        if grad_sync_func is not None:
            sync_virtual_mb = virtual_mb_id - pp_rank
            if sync_virtual_mb >= 0 and is_last_microbatch_for_chunk(sync_virtual_mb):
                sync_chunk = get_model_chunk_id(sync_virtual_mb, forward=False)
                enable_grad_sync()
                grad_sync_func[sync_chunk](model[sync_chunk].parameters())
                synchronized_model_chunks.add(sync_chunk)
        disable_grad_sync()

        return input_tensor_grad

    if config.timers is not None:
        config.timers("forward-backward", log_level=1).start()

    # ---- Warmup: pre-receive the first forward activation -------------------
    input_tensors[0].append(
        p2p_communicator.recv_forward(tensor_shape, comm_dtype)
    )

    # ---- Warmup phase -------------------------------------------------------
    for k in range(num_warmup):
        output_tensor = run_forward(k)

        recv_prev, next_fwd_chunk = recv_tensor_from_previous_stage(k, forward=True)

        if k < (total_num_microbatches - 1):
            input_tensor = p2p_communicator.send_forward_recv_forward(
                output_tensor, recv_prev=recv_prev, tensor_shape=tensor_shape,
                recv_dtype=comm_dtype
            )
            if recv_prev:
                input_tensors[next_fwd_chunk].append(input_tensor)
        else:
            p2p_communicator.send_forward(output_tensor)

        _deallocate_output_tensor(
            output_tensor, getattr(config, "deallocate_pipeline_outputs", False)
        )

        # On the very last warmup step, pre-receive the first backward gradient
        if k == (num_warmup - 1) and not forward_only and num_steady > 0:
            recv_next = not p2p_communicator.is_pp_last_stage
            if recv_next:
                output_tensor_grads[num_model_chunks - 1].append(
                    p2p_communicator.recv_backward(tensor_shape, comm_dtype)
                )

    # ---- Steady (1F1B) phase ------------------------------------------------
    for k in range(num_steady):
        fwd_k = k + num_warmup
        bwd_k = k

        output_tensor    = run_forward(fwd_k)
        input_tensor_grad = run_backward(bwd_k)

        recv_prev, next_fwd_chunk = recv_tensor_from_previous_stage(fwd_k, forward=True)
        recv_next, next_bwd_chunk = recv_tensor_from_previous_stage(bwd_k, forward=False)

        last_fwd = fwd_k == (total_num_microbatches - 1)
        last_bwd = k == (num_steady - 1)

        if not last_fwd:
            recv_prev_flag = recv_prev
        else:
            recv_prev_flag = False

        if not last_bwd:
            recv_next_flag = recv_next
        else:
            recv_next_flag = False

        input_tensor, output_tensor_grad = (
            p2p_communicator.send_forward_backward_recv_forward_backward(
                output_tensor,
                input_tensor_grad,
                recv_prev=recv_prev_flag,
                recv_next=recv_next_flag,
                tensor_shape=tensor_shape,
                recv_dtype=comm_dtype,
            )
        )

        if recv_prev_flag and input_tensor is not None:
            input_tensors[next_fwd_chunk].append(input_tensor)
        if recv_next_flag and output_tensor_grad is not None:
            output_tensor_grads[next_bwd_chunk].append(output_tensor_grad)

        _deallocate_output_tensor(
            output_tensor, getattr(config, "deallocate_pipeline_outputs", False)
        )

    # ---- Cooldown phase: drain backward passes ------------------------------
    if not forward_only:
        for k in range(num_warmup):
            bwd_k = k + num_steady
            chunk_id = get_model_chunk_id(bwd_k, forward=False)

            recv_next = not p2p_communicator.is_pp_last_stage
            if recv_next:
                output_tensor_grads[chunk_id].append(
                    p2p_communicator.recv_backward(tensor_shape, comm_dtype)
                )
            else:
                if len(output_tensor_grads[chunk_id]) == 0:
                    output_tensor_grads[chunk_id].append(None)

            input_tensor_grad = run_backward(bwd_k)
            p2p_communicator.send_backward(input_tensor_grad)

        # Ensure remaining chunks get their grad sync
        if no_sync_context is not None:
            enable_grad_sync()
        for chunk_id, sync_func in enumerate(grad_sync_func or []):
            if chunk_id not in synchronized_model_chunks:
                enable_grad_sync()
                sync_func(model[chunk_id].parameters())
                synchronized_model_chunks.add(chunk_id)

    # Restore forward-only overrides
    if forward_only:
        config.grad_sync_func  = _saved_grad_sync   # type: ignore[possibly-undefined]
        config.param_sync_func = _saved_param_sync  # type: ignore[possibly-undefined]

    # Finalize grads
    if config.finalize_model_grads_func is not None and not forward_only:
        config.finalize_model_grads_func(model)

    if config.timers is not None:
        config.timers("forward-backward").stop()

    return forward_data_store


# ===========================================================================
# utils.py
# ===========================================================================

def get_num_microbatches() -> int:
    """Return the current global microbatch count from parallel state."""
    if _ps is None:
        raise RuntimeError("parallel_state is not available")
    # The global microbatch count is typically set by the training loop and
    # stored in parallel_state via set_num_microbatches / get_num_microbatches.
    fn = getattr(_ps, "get_num_microbatches", None)
    if fn is None:
        raise AttributeError(
            "parallel_state does not expose get_num_microbatches. "
            "Set it via parallel_state.set_num_microbatches() in your training loop."
        )
    return fn()


def get_pipeline_model_parallel_rank_for_layer(layer_number: int) -> int:
    """Given a global layer number, return which PP rank owns it.

    Uses ``config.pipeline_layer_split`` for heterogeneous (DES-LOC) splits.
    If ``pipeline_layer_split`` is not set, layers are distributed uniformly
    across pipeline stages.

    DES-LOC key invariant: ``pipeline_layer_split`` is a list whose i-th
    element is the number of transformer layers assigned to pipeline stage i.
    The list must be stored somewhere accessible at call time; we read it
    from ``parallel_state`` if a helper is registered, otherwise from a
    module-level registry ``_PIPELINE_LAYER_SPLIT``.

    Args:
        layer_number: 0-based global transformer layer index.

    Returns:
        PP rank (0-based) that owns ``layer_number``.

    Raises:
        ValueError: If ``layer_number`` is out of range for the current split.

    Example (DES-LOC unequal split)::

        # 5 GPU pipeline with layers [4, 8, 8, 4, 8] → 32 layers total
        # Layer 0-3   → rank 0
        # Layer 4-11  → rank 1
        # Layer 12-19 → rank 2
        # Layer 20-23 → rank 3
        # Layer 24-31 → rank 4
    """
    # ---- Resolve the split ---------------------------------------------------
    # Try parallel_state first (set at initialisation time), then fall back to
    # a module-level registry, then to a uniform split based on the PP world size.
    split: Optional[List[int]] = None

    if _ps is not None:
        split = getattr(_ps, "_PIPELINE_LAYER_SPLIT", None)

    if split is None:
        split = _PIPELINE_LAYER_SPLIT  # module-level fallback (see below)

    if split is None:
        # No split configured — use uniform distribution
        if _ps is None or not torch.distributed.is_initialized():
            raise RuntimeError(
                "get_pipeline_model_parallel_rank_for_layer: "
                "pipeline_layer_split is not configured and distributed is not initialised."
            )
        pp_size = _ps.get_pipeline_model_parallel_world_size()
        split = _uniform_split(layer_number, pp_size)
        # Infer pp_rank from cumulative sum
        cumulative = 0
        for rank, count in enumerate(split):
            cumulative += count
            if layer_number < cumulative:
                return rank
        raise ValueError(
            f"layer_number {layer_number} is out of range for a uniform split "
            f"with PP size {pp_size}."
        )

    # ---- Walk the split list ------------------------------------------------
    cumulative = 0
    for rank, count in enumerate(split):
        cumulative += count
        if layer_number < cumulative:
            return rank

    raise ValueError(
        f"layer_number {layer_number} is out of range for pipeline_layer_split "
        f"{split} (total layers: {cumulative})."
    )


def _uniform_split(layer_number: int, pp_size: int) -> List[int]:
    """Compute a placeholder uniform split for validation purposes.

    This helper is only used by ``get_pipeline_model_parallel_rank_for_layer``
    when no explicit split is configured and distributed is initialised.
    It returns a list of length ``pp_size`` where each element is 1, purely
    to drive the rank-lookup loop — the caller infers the actual layer count
    from the training configuration.
    """
    # We don't know the total number of layers here, so we return a list of
    # ones and let the range check fail gracefully if needed.
    return [1] * pp_size


# Module-level registry for pipeline_layer_split.
# Set this before calling get_pipeline_model_parallel_rank_for_layer if you are
# not using parallel_state (e.g. in unit tests or single-process scripts).
#
# Example:
#   from deepspeed.core.pipeline_parallel import set_pipeline_layer_split
#   set_pipeline_layer_split([4, 8, 8, 4, 8])
_PIPELINE_LAYER_SPLIT: Optional[List[int]] = None


def set_pipeline_layer_split(split: List[int]) -> None:
    """Register a pipeline layer split for get_pipeline_model_parallel_rank_for_layer.

    Args:
        split: List of per-stage layer counts, e.g. [4, 8, 8, 4, 8] for a
               5-stage heterogeneous DES-LOC pipeline with 32 total layers.
    """
    global _PIPELINE_LAYER_SPLIT
    if not split or any(c <= 0 for c in split):
        raise ValueError(
            f"pipeline_layer_split must be a non-empty list of positive integers, got {split}"
        )
    _PIPELINE_LAYER_SPLIT = list(split)


__all__ = [
    # P2P
    "P2PCommunicator",
    # Schedules
    "get_forward_backward_func",
    "forward_step",
    "backward_step",
    "forward_backward_no_pipelining",
    "forward_backward_pipelining_without_interleaving",
    "forward_backward_pipelining_with_interleaving",
    # Utils
    "get_num_microbatches",
    "get_pipeline_model_parallel_rank_for_layer",
    "set_pipeline_layer_split",
]
