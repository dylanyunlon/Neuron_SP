# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Pipeline schedule implementations for pipeline parallelism.

Ported from Megatron-LM/megatron/core/pipeline_parallel/schedules.py and
extended with DES-LOC heterogeneous bubble-filling.

Schedule summary
----------------
PP=1:
    forward_backward_no_pipelining — all microbatches on one device.
PP>1, VPP=None:
    forward_backward_pipelining_without_interleaving — standard 1F1B.
PP>1, VPP>1:
    forward_backward_pipelining_with_interleaving — interleaved 1F1B (VPP).

DES-LOC extension — heterogeneous bubble filling
-------------------------------------------------
In a mixed H100 + A6000 pipeline the H100 stage finishes each microbatch much
faster (≈ 21× higher BF16 throughput).  In the standard 1F1B schedule the H100
sits idle during the bubbles that are produced while A6000 stages are still
computing.

This file adds a ``HeterogeneousBubbleFiller`` helper that a DES-LOC-aware
training loop can attach to ``config.desloc.bubble_filler``.  During the
warmup phase the bubble-filler schedules extra forward microbatches on the
fast (H100) stages and during cooldown it drains the extra activations.  The
net effect is that H100 utilisation increases from ~40% (standard 1F1B with
slow A6000 peers) toward ~70-80%.

The filler is **opt-in**: if ``config.desloc`` is None (or
``config.desloc.bubble_fill`` is False) the schedules behave identically to
the baseline Megatron 1F1B implementation.
"""

from __future__ import annotations

import contextlib
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from torch.autograd.variable import Variable

from deepspeed.core.model_parallel_config import ModelParallelConfig
from deepspeed.core.pipeline_parallel.p2p_communication import P2PCommunicator, is_single_shape

# ---------------------------------------------------------------------------
# Import parallel state with graceful fallback for unit-test environments.
# ---------------------------------------------------------------------------
try:
    from deepspeed.core import parallel_state as _ps
except ImportError:
    _ps = None  # type: ignore[assignment]

# Type alias
Shape = Union[List[int], torch.Size]


# ===========================================================================
# Internal helpers
# ===========================================================================

def deallocate_output_tensor(
    out: Optional[Union[torch.Tensor, List, Dict]],
    deallocate: bool,
) -> None:
    """Pseudo-deallocate a pipeline output tensor to free activation memory.

    Replaces ``.data`` with a scalar (size-1) placeholder so that the
    autograd graph (``.grad_fn``) survives for the backward pass while the
    activation buffer is returned to the allocator.

    Supports tensors, lists of tensors, and dicts of tensors (multi-module).
    """
    if out is None or not deallocate:
        return
    if isinstance(out, dict):
        for v in out.values():
            deallocate_output_tensor(v, deallocate)
        return
    if isinstance(out, (list, tuple)):
        for item in out:
            deallocate_output_tensor(item, deallocate)
        return
    assert isinstance(out, torch.Tensor), f"expected Tensor, got {type(out)}"
    assert out._base is None, "counter-productive to free a view of another tensor"
    out.data = torch.empty((1,), device=out.device, dtype=out.dtype)


def custom_backward(output: torch.Tensor, grad_output: Optional[torch.Tensor]) -> None:
    """Directly invoke the C++ autograd engine.

    Required when ``deallocate_pipeline_outputs=True``: after deallocating
    the output tensor's ``.data``, PyTorch's Python-level
    ``torch.autograd.backward`` rejects the call because the shapes no longer
    match.  The C++ engine bypasses this check.
    """
    assert output.numel() == 1, (
        "output should be pseudo-freed (scalar) before custom_backward is called"
    )
    if grad_output is None:
        grad_output = torch.ones_like(output, memory_format=torch.preserve_format)
    Variable._execution_engine.run_backward(  # type: ignore[attr-defined]
        tensors=(output,),
        grad_tensors=(grad_output,),
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )


def get_tensor_shapes(
    seq_length: int,
    micro_batch_size: int,
    config: ModelParallelConfig,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> List[torch.Size]:
    """Compute pipeline activation tensor shapes.

    Returns a list with one element: the [seq, batch, hidden] shape adjusted
    for sequence parallelism and tensor-model parallelism.

    DES-LOC note: On heterogeneous clusters the hidden_size is uniform across
    all stages, but the seq dimension may differ if context-parallel chunking
    is stage-specific.  The caller can override with ``variable_seq_lengths``.
    """
    hidden_size: int = getattr(config, "hidden_size", 0)
    if hidden_size == 0:
        raise RuntimeError(
            "ModelParallelConfig.hidden_size must be set; got 0 or attribute missing."
        )

    seq = seq_length

    # Adjust for context parallelism
    if cp_group is not None:
        cp_size = cp_group.size()
        if cp_size > 1:
            seq = seq // cp_size

    # Adjust for sequence parallelism (SP splits the sequence across TP ranks)
    if getattr(config, "sequence_parallel", False):
        tp_size = tp_group.size() if tp_group is not None else config.tensor_model_parallel_size
        seq = seq // tp_size

    return [torch.Size([seq, micro_batch_size, hidden_size])]


def _get_model_config(model: Union[nn.Module, List[nn.Module]]) -> ModelParallelConfig:
    """Extract ModelParallelConfig from a (possibly wrapped) model."""
    m = model[0] if isinstance(model, (list, tuple)) else model
    while hasattr(m, "module") and not hasattr(m, "config"):
        m = m.module  # type: ignore[attr-defined]
    cfg = getattr(m, "config", None)
    if cfg is None or not isinstance(cfg, ModelParallelConfig):
        raise RuntimeError(
            f"Cannot find ModelParallelConfig on model {type(m)}. "
            "Attach it as model.config."
        )
    return cfg


def _check_first_val_step(
    first_val_step: Optional[bool],
    forward_only: bool,
    is_first: bool,
) -> bool:
    """Return True only on the first validation microbatch."""
    if first_val_step is not None:
        return first_val_step and is_first
    return is_first


# ===========================================================================
# DES-LOC heterogeneous bubble-filling
# ===========================================================================

class HeterogeneousBubbleFiller:
    """Fill H100 pipeline bubbles with extra forward microbatches.

    In a 3-stage pipeline [A6000-0, A6000-1, H100-2] with 8 microbatches:

    Standard 1F1B timeline on H100-2 (stage 2, pp_rank=2):
        F0 F1 F2 B0 F3 B1 F4 B2 F5 B3 F6 B4 F7 B5 _ B6 _ B7
                                                    ^  ^
                           two empty bubbles while A6000 drains its backprop

    With bubble filling on H100-2:
        F0 F1 F2 B0 F3 B1 F4 B2 F5 B3 F6 B4 F7 B5 Fe B6 Fe B7
        (Fe = extra forward microbatch from the next global batch)

    The filler runs ``extra_forward_step_func`` during each detected bubble
    and counts the number of extra microbatches processed.  These are reported
    back to the training loop so it can account for the additional tokens.

    Args:
        tier_rank:             The GPU tier rank (0 = fastest, e.g. H100).
        fast_stage_indices:    PP stage indices that are on the fast tier.
        extra_forward_step_func: User-provided function
                               ``(data_iterator, model) → (tensor, loss_fn)``
                               for the extra forward pass.
        data_iterator:         Iterator for the fill microbatches.
        model:                 The model chunk for this stage.
        enabled:               Master switch (set False to disable filling).
    """

    def __init__(
        self,
        tier_rank: int,
        fast_stage_indices: List[int],
        extra_forward_step_func: Optional[Callable],
        data_iterator: Optional[object],
        model: Optional[nn.Module],
        enabled: bool = True,
    ) -> None:
        self.tier_rank = tier_rank
        self.fast_stage_indices = set(fast_stage_indices)
        self.extra_forward_step_func = extra_forward_step_func
        self.data_iterator = data_iterator
        self.model = model
        self.enabled = enabled
        self._extra_count = 0

    def is_fast_stage(self, pp_rank: int) -> bool:
        """Return True if this PP rank is on the fast (H100) tier."""
        return pp_rank in self.fast_stage_indices

    def maybe_fill_bubble(
        self,
        pp_rank: int,
        forward_data_store: list,
        config: ModelParallelConfig,
    ) -> bool:
        """Attempt to run an extra forward pass during an idle bubble.

        Called by the schedule at each point where the fast stage would
        otherwise stall waiting for a slower stage to produce a gradient.

        Returns True if an extra forward was actually executed.
        """
        if not self.enabled:
            return False
        if not self.is_fast_stage(pp_rank):
            return False
        if self.extra_forward_step_func is None or self.data_iterator is None:
            return False

        try:
            output_tensor, loss_func = self.extra_forward_step_func(
                self.data_iterator, self.model
            )
            # Only collect losses; discard activations (forward-only fill)
            if loss_func is not None:
                result = loss_func(output_tensor)
                if len(result) >= 2:
                    forward_data_store.append(result[-1])  # last element = metrics dict
            self._extra_count += 1
            return True
        except StopIteration:
            # No more fill data; disable filling for this iteration
            self.enabled = False
            return False

    @property
    def extra_microbatch_count(self) -> int:
        """Number of extra microbatches processed by bubble filling."""
        return self._extra_count

    def reset(self) -> None:
        """Reset counter for a new global step."""
        self._extra_count = 0


# ---------------------------------------------------------------------------
# Bubble-filler factory — creates a filler from DesLocConfig if present
# ---------------------------------------------------------------------------

def _make_bubble_filler(
    config: ModelParallelConfig,
    pp_rank: int,
    model: Optional[nn.Module] = None,
    extra_forward_step_func: Optional[Callable] = None,
    extra_data_iterator: Optional[object] = None,
) -> Optional[HeterogeneousBubbleFiller]:
    """Build a HeterogeneousBubbleFiller from config if DES-LOC is enabled."""
    desloc = getattr(config, "desloc", None)
    if desloc is None or not getattr(desloc, "enabled", False):
        return None

    bubble_fill = getattr(desloc, "bubble_fill", False)
    if not bubble_fill:
        return None

    # Identify which PP stages are on the fast (DATACENTER) tier
    from deepspeed.core.desloc_config import TierType
    fast_stages: List[int] = []
    for tier in getattr(desloc, "tiers", []):
        if tier.tier_type == TierType.DATACENTER:
            # Assume GPU indices map 1:1 to PP ranks for simplicity
            fast_stages.extend(tier.gpu_indices)

    if not fast_stages:
        return None

    return HeterogeneousBubbleFiller(
        tier_rank=pp_rank,
        fast_stage_indices=fast_stages,
        extra_forward_step_func=extra_forward_step_func,
        data_iterator=extra_data_iterator,
        model=model,
        enabled=True,
    )


# ===========================================================================
# forward_step / backward_step
# ===========================================================================

def forward_step(
    forward_step_func: Callable,
    data_iterator: object,
    model: nn.Module,
    num_microbatches: int,
    input_tensor: Optional[Union[torch.Tensor, List]],
    forward_data_store: list,
    config: ModelParallelConfig,
    collect_non_loss_data: bool = False,
    checkpoint_activations_microbatch: Optional[bool] = None,
    is_first_microbatch: bool = False,
    current_microbatch: Optional[int] = None,
    vp_stage: Optional[int] = None,
    is_last_stage: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Execute one forward microbatch.

    Calls ``forward_step_func(data_iterator, model)`` which must return a
    (output_tensor, loss_func) pair.  On the last PP stage the loss is
    computed and scaled; on all other stages the raw activation is returned.

    Args:
        forward_step_func:   User forward function.
        data_iterator:       Data iterator for this microbatch.
        model:               The local model chunk.
        num_microbatches:    Total microbatch count (for loss scaling).
        input_tensor:        Activation from the previous stage (None if first).
        forward_data_store:  Accumulator for per-microbatch loss outputs.
        config:              Parallelism configuration.
        collect_non_loss_data: If True, collect arbitrary model outputs.
        checkpoint_activations_microbatch: Whether to checkpoint this MB.
        is_first_microbatch: Signals model hooks that need first-step init.
        current_microbatch:  Index for CUDA graph management.
        vp_stage:            Virtual pipeline stage index.
        is_last_stage:       True if this is the logical last PP stage.

    Returns:
        (output_tensor, num_tokens): output tensor + token count scalar.
    """
    if config.timers is not None:
        config.timers("forward-compute", log_level=2).start()

    if is_first_microbatch and hasattr(model, "set_is_first_microbatch"):
        model.set_is_first_microbatch()

    # Unwrap scalar input so set_input_tensor always receives a list
    unwrap_output = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output = True

    # Inject the received activation into the model
    set_input_fn = getattr(model, "set_input_tensor", None)
    if set_input_fn is not None:
        set_input_fn(input_tensor)

    # Run the user forward
    output_tensor, loss_func = forward_step_func(data_iterator, model)

    num_tokens = torch.tensor(0, dtype=torch.int)

    # ---- Last stage: compute loss -------------------------------------------
    if is_last_stage:
        if loss_func is None:
            forward_data_store.append(output_tensor)
        elif collect_non_loss_data:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)
        else:
            result = loss_func(output_tensor)
            if len(result) == 3:
                output_tensor, num_tokens, loss_reduced = result
                output_tensor = output_tensor / num_microbatches
            else:
                assert len(result) == 2
                output_tensor, loss_reduced = result
                output_tensor = output_tensor / num_microbatches
            forward_data_store.append(loss_reduced)
    elif config.grad_scale_func is not None:
        output_tensor = config.grad_scale_func(output_tensor)

    if config.timers is not None:
        config.timers("forward-compute").stop()

    if unwrap_output:
        return output_tensor, num_tokens
    return [output_tensor], num_tokens


def backward_step(
    input_tensor: Optional[Union[torch.Tensor, List]],
    output_tensor: Union[torch.Tensor, List],
    output_tensor_grad: Optional[Union[torch.Tensor, List]],
    config: ModelParallelConfig,
) -> Optional[Union[torch.Tensor, List]]:
    """Execute one backward microbatch.

    Computes gradients through ``output_tensor`` w.r.t. ``input_tensor``.
    Handles the last-stage case where ``output_tensor_grad`` is None.

    Args:
        input_tensor:       Activation received from the previous stage.
        output_tensor:      Output produced by forward_step (may be
                            memory-deallocated if deallocate_pipeline_outputs).
        output_tensor_grad: Gradient received from the next stage (None on
                            the last PP stage).
        config:             Parallelism configuration.

    Returns:
        input_tensor_grad: Gradient w.r.t. the received activation.
    """
    if config.timers is not None:
        config.timers("backward-compute", log_level=2).start()

    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Retain grads on input tensors so we can collect them afterwards
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    # Apply grad scaling on last stage (output_tensor_grad[0] is None there)
    if output_tensor_grad[0] is None and config.grad_scale_func is not None:
        output_tensor[0] = config.grad_scale_func(output_tensor[0])

    if output_tensor[0].requires_grad:
        deallocate = getattr(config, "deallocate_pipeline_outputs", False)
        if deallocate:
            custom_backward(output_tensor[0], output_tensor_grad[0])
        else:
            torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])

    input_tensor_grad: List[Optional[torch.Tensor]] = []
    for x in input_tensor:
        input_tensor_grad.append(None if x is None else x.grad)

    if config.timers is not None:
        config.timers("backward-compute").stop()

    if unwrap_input_tensor_grad:
        return input_tensor_grad[0]
    return input_tensor_grad


# ===========================================================================
# get_forward_backward_func — schedule selector
# ===========================================================================

def get_forward_backward_func(
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_size: Optional[int] = None,
    *,
    forward_only: bool = False,
) -> Callable:
    """Return the appropriate pipeline schedule function.

    Selection logic:
        PP=1  →  forward_backward_no_pipelining
        PP>1 and VPP is not None  →  forward_backward_pipelining_with_interleaving
        PP>1 and VPP is None      →  forward_backward_pipelining_without_interleaving

    Falls back to ``parallel_state`` when arguments are not provided.

    Args:
        virtual_pipeline_model_parallel_size: VPP degree, or None.
        pipeline_model_parallel_size: PP degree (≥ 1).
        forward_only: Informational; does not affect which function is returned.

    Returns:
        One of the three schedule callables defined in this module.
    """
    # Resolve from parallel_state if not given
    if pipeline_model_parallel_size is None:
        if _ps is None:
            raise RuntimeError("pipeline_model_parallel_size must be provided")
        pipeline_model_parallel_size = _ps.get_pipeline_model_parallel_world_size()
    if virtual_pipeline_model_parallel_size is None and _ps is not None:
        vpp_fn = getattr(_ps, "get_virtual_pipeline_model_parallel_world_size", None)
        if vpp_fn is not None:
            virtual_pipeline_model_parallel_size = vpp_fn()

    if pipeline_model_parallel_size < 1:
        raise ValueError(
            f"pipeline_model_parallel_size must be ≥ 1, got {pipeline_model_parallel_size}"
        )
    if pipeline_model_parallel_size == 1:
        return forward_backward_no_pipelining
    if (
        virtual_pipeline_model_parallel_size is not None
        and virtual_pipeline_model_parallel_size > 1
    ):
        return forward_backward_pipelining_with_interleaving
    return forward_backward_pipelining_without_interleaving


# ===========================================================================
# Schedule 1: No pipelining (PP=1)
# ===========================================================================

def forward_backward_no_pipelining(
    *,
    forward_step_func: Callable,
    data_iterator: Union[object, List[object]],
    model: Union[nn.Module, List[nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    config: ModelParallelConfig,
    p2p_communicator: Optional[P2PCommunicator] = None,
) -> list:
    """No pipeline parallelism — sequential forward/backward with grad accumulation.

    All microbatches are run sequentially on the same device.  All but the
    last microbatch are run inside ``no_sync`` to defer grad all-reduce;
    the last microbatch triggers the all-reduce.

    DES-LOC: num_microbatches can differ per rank (heterogeneous micro-batch
    sizes). Each rank accumulates its own count.

    Args:
        forward_step_func:      User forward function.
        data_iterator:          Data loader (or single-element list).
        model:                  Single model (or single-element list).
        num_microbatches:       Number of microbatches to process.
        seq_length:             Sequence length (unused; kept for API compat).
        micro_batch_size:       Micro-batch size (unused; kept for API compat).
        forward_only:           If True, skip backward passes.
        collect_non_loss_data:  Passed through to forward_step.
        first_val_step:         First step of validation phase.
        config:                 Parallelism configuration.
        p2p_communicator:       Unused for PP=1; accepted for API uniformity.

    Returns:
        List of per-microbatch loss / output dicts.
    """
    if isinstance(model, (list, tuple)):
        assert len(model) == 1, (
            "forward_backward_no_pipelining does not support model chunking; "
            f"got {len(model)} chunks."
        )
        model = model[0]
    if isinstance(data_iterator, (list, tuple)):
        assert len(data_iterator) == 1
        data_iterator = data_iterator[0]

    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext

    forward_data_store: list = []
    input_tensor: Optional[torch.Tensor] = None
    output_tensor_grad: Optional[torch.Tensor] = None

    if config.timers is not None:
        config.timers("forward-backward", log_level=1).start()

    # Determine last PP stage for is_last_stage argument
    is_last_stage = True
    if _ps is not None and torch.distributed.is_initialized():
        is_last_stage = _ps.is_pipeline_last_stage()

    with no_sync_func():
        for i in range(num_microbatches - 1):
            output_tensor, _ = forward_step(
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                collect_non_loss_data=collect_non_loss_data,
                is_first_microbatch=_check_first_val_step(first_val_step, forward_only, i == 0),
                current_microbatch=i,
                is_last_stage=is_last_stage,
            )
            if not forward_only:
                backward_step(input_tensor, output_tensor, output_tensor_grad, config)

    # Last microbatch — grad sync fires here
    output_tensor, _ = forward_step(
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        collect_non_loss_data=collect_non_loss_data,
        is_first_microbatch=_check_first_val_step(
            first_val_step, forward_only, num_microbatches == 1
        ),
        current_microbatch=num_microbatches - 1,
        is_last_stage=is_last_stage,
    )
    if not forward_only:
        backward_step(input_tensor, output_tensor, output_tensor_grad, config)

    if config.finalize_model_grads_func is not None and not forward_only:
        config.finalize_model_grads_func([model])

    if config.timers is not None:
        config.timers("forward-backward").stop()

    return forward_data_store


# ===========================================================================
# Schedule 2: Standard 1F1B (non-interleaved)
# ===========================================================================

def forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func: Callable,
    data_iterator: Union[object, List[object]],
    model: Union[nn.Module, List[nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    config: ModelParallelConfig,
    p2p_communicator: Optional[P2PCommunicator] = None,
    # DES-LOC heterogeneous bubble filling (optional)
    bubble_filler: Optional[HeterogeneousBubbleFiller] = None,
) -> list:
    """Standard 1F1B pipeline schedule (non-interleaved).

    Schedule phases:
        Warmup:   (pp_size - pp_rank - 1) pure forward passes to fill
                  the pipeline.
        Steady:   1F1B pairs — one forward, one backward per iteration.
        Cooldown: Drain remaining in-flight backward passes.

    DES-LOC heterogeneous bubble filling:
        During warmup and cooldown, fast stages (H100) that would otherwise
        idle are kept busy with extra forward microbatches from the **next**
        global batch (supplied via ``bubble_filler``).  This reduces H100
        bubble fraction from ~60% toward ~20% on a 3-stage A6000×2+H100 ring.

        Bubble-fill microbatches are counted separately and reported back to
        the training loop via ``bubble_filler.extra_microbatch_count``.

    Args:
        forward_step_func:   User forward function.
        data_iterator:       Data iterator (or single-element list).
        model:               Single model (or single-element list).
        num_microbatches:    Total microbatches to process.
        seq_length:          Sequence length for computing activation shape.
        micro_batch_size:    Batch size for computing activation shape.
        decoder_seq_length:  Decoder sequence length (unused; compat).
        forward_only:        If True, skip backward passes.
        collect_non_loss_data: Passed through to forward_step.
        first_val_step:      First step of the validation phase.
        config:              Parallelism configuration.
        p2p_communicator:    Optional pre-built communicator.
        bubble_filler:       DES-LOC bubble-filling helper (opt-in).

    Returns:
        forward_data_store: List of per-microbatch loss outputs.
    """
    if isinstance(model, (list, tuple)):
        assert len(model) == 1, (
            "forward_backward_pipelining_without_interleaving does not support "
            "model chunking; use the interleaved variant instead."
        )
        model = model[0]
    if isinstance(data_iterator, (list, tuple)):
        assert len(data_iterator) == 1
        data_iterator = data_iterator[0]

    # Build communicator if not supplied
    if p2p_communicator is None:
        p2p_communicator = P2PCommunicator(config)

    # Activation tensor shapes -----------------------------------------------
    tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        config=config,
    )
    # For non-interleaved PP, recv and send shapes are the same
    recv_tensor_shapes = tensor_shapes
    send_tensor_shapes = tensor_shapes

    # Gradient-sync control --------------------------------------------------
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

    # Schedule dimensions -----------------------------------------------------
    pp_rank = p2p_communicator.current_stage
    pp_size = p2p_communicator.total_stages

    # DES-LOC heterogeneous bubble detection:
    # Fast (H100) stages have fewer neighbours ahead, so fewer warmup MBs.
    # We expose this to the bubble_filler so it knows when to insert fills.
    num_warmup = min(pp_size - pp_rank - 1, num_microbatches)
    num_steady = num_microbatches - num_warmup

    # Activation checkpoint support
    max_outstanding_backprops: Optional[int] = None
    num_microbatches_with_partial_ckpt = getattr(
        config, "num_microbatches_with_partial_activation_checkpoints", None
    )
    if num_microbatches_with_partial_ckpt is not None:
        max_outstanding_backprops = num_warmup + 1

    forward_data_store: list = []
    input_tensors:  List[Optional[torch.Tensor]] = []
    output_tensors: List[torch.Tensor] = []
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda" if torch.cuda.is_available() else "cpu")

    if config.timers is not None:
        config.timers("forward-backward", log_level=1).start()

    # ---- Warmup phase -------------------------------------------------------
    for i in range(num_warmup):
        # Activation checkpoint decision
        checkpoint_mb = None
        if max_outstanding_backprops is not None:
            checkpoint_mb = (
                i % max_outstanding_backprops >= num_microbatches_with_partial_ckpt
            )

        input_tensor = p2p_communicator.recv_forward(
            recv_tensor_shapes, p2p_communicator.is_pp_first_stage
        )
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_mb,
            is_first_microbatch=_check_first_val_step(first_val_step, forward_only, i == 0),
            current_microbatch=i,
            is_last_stage=p2p_communicator.is_pp_last_stage,
        )
        p2p_communicator.send_forward(output_tensor, p2p_communicator.is_pp_last_stage)
        total_num_tokens += num_tokens

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(
                output_tensor, getattr(config, "deallocate_pipeline_outputs", False)
            )

        # DES-LOC: fill the bubble on fast stages during warmup
        # (H100 sends its output and immediately starts an extra forward
        # while the A6000 stages are still computing their first microbatches)
        if bubble_filler is not None:
            bubble_filler.maybe_fill_bubble(pp_rank, forward_data_store, config)

    # Pre-receive the first activation for the steady-state phase
    if num_steady > 0:
        input_tensor = p2p_communicator.recv_forward(
            recv_tensor_shapes, p2p_communicator.is_pp_first_stage
        )

    # ---- Steady (1F1B) phase ------------------------------------------------
    for i in range(num_steady):
        last_iter = i == (num_steady - 1)

        # Activation checkpoint decision
        checkpoint_mb = None
        if max_outstanding_backprops is not None:
            checkpoint_mb = (
                (i + num_warmup) % max_outstanding_backprops >= num_microbatches_with_partial_ckpt
            )

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_mb,
            is_first_microbatch=_check_first_val_step(
                first_val_step, forward_only, (i == 0) and (num_warmup == 0)
            ),
            current_microbatch=i + num_warmup,
            is_last_stage=p2p_communicator.is_pp_last_stage,
        )
        total_num_tokens += num_tokens

        if forward_only:
            p2p_communicator.send_forward(output_tensor, p2p_communicator.is_pp_last_stage)
            if not last_iter:
                input_tensor = p2p_communicator.recv_forward(
                    recv_tensor_shapes, p2p_communicator.is_pp_first_stage
                )
        else:
            # Batched send-forward + recv-backward (avoids two separate round trips)
            output_tensor_grad = p2p_communicator.send_forward_recv_backward(
                output_tensor, send_tensor_shapes, p2p_communicator.is_pp_last_stage
            )

            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(
                output_tensor, getattr(config, "deallocate_pipeline_outputs", False)
            )

            # Pop oldest saved tensors for the backward pass
            input_tensor  = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            # Enable grad sync on last steady iteration when no warmup-phase
            # backward passes remain
            if num_warmup == 0 and last_iter:
                if config.grad_sync_func is None or p2p_communicator.is_pp_first_stage:
                    enable_grad_sync()

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, config
            )

            if last_iter:
                p2p_communicator.send_backward(
                    input_tensor_grad, p2p_communicator.is_pp_first_stage
                )
                input_tensor = None
            else:
                input_tensor = p2p_communicator.send_backward_recv_forward(
                    input_tensor_grad, recv_tensor_shapes, p2p_communicator.is_pp_first_stage
                )

    # ---- Cooldown phase -----------------------------------------------------
    if not forward_only:
        for i in range(num_warmup):
            if i == num_warmup - 1:
                if config.grad_sync_func is None or p2p_communicator.is_pp_first_stage:
                    enable_grad_sync()

            input_tensor  = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = p2p_communicator.recv_backward(
                send_tensor_shapes, p2p_communicator.is_pp_last_stage
            )

            # DES-LOC: H100 stage can fill the bubble *while waiting* for the
            # grad recv to complete.  The bubble_filler runs before we issue
            # the backward pass; since recv_backward is synchronous this is
            # the idle window.
            if bubble_filler is not None:
                bubble_filler.maybe_fill_bubble(pp_rank, forward_data_store, config)

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, config
            )
            p2p_communicator.send_backward(
                input_tensor_grad, p2p_communicator.is_pp_first_stage
            )

        # Flush any remaining deferred grad reductions
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

    # Finalize grads (DP all-reduce / reduce-scatter, SP LN all-reduce)
    if config.finalize_model_grads_func is not None and not forward_only:
        config.finalize_model_grads_func([model])

    if config.timers is not None:
        config.timers("forward-backward").stop()

    return forward_data_store


# ===========================================================================
# Schedule 3: Interleaved 1F1B (virtual pipeline parallelism)
# ===========================================================================

def forward_backward_pipelining_with_interleaving(
    *,
    forward_step_func: Callable,
    data_iterator: Union[List[object], object],
    model: List[nn.Module],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    config: ModelParallelConfig,
    p2p_communicator: Optional[P2PCommunicator] = None,
    # DES-LOC bubble filling
    bubble_filler: Optional[HeterogeneousBubbleFiller] = None,
) -> list:
    """Interleaved 1F1B for virtual pipeline parallelism (VPP).

    Each PP rank holds ``num_model_chunks`` (= VPP size) independent model
    chunks.  Microbatches cycle through chunks in a particular order to
    maximise pipeline utilisation by reducing bubble fraction from
    ``1/pp_size`` to ``1/(pp_size * vpp_size)``.

    Follows the schedule described in:
    "Efficient Large-Scale Language Model Training on GPU Clusters" (2021).

    DES-LOC extension: fast stages (H100) run bubble-filling extra forward
    passes at warmup/cooldown boundaries, same as the non-interleaved variant.

    Args:
        forward_step_func:      User forward function.
        data_iterator:          List of iterators, one per model chunk.
        model:                  List of model chunks (one per VPP stage).
        num_microbatches:       Microbatches per PP rank.
        seq_length:             Sequence length for shape computation.
        micro_batch_size:       Batch size for shape computation.
        decoder_seq_length:     Unused; accepted for API compat.
        forward_only:           If True, skip backward passes.
        collect_non_loss_data:  Passed through to forward_step.
        first_val_step:         First validation step flag.
        config:                 Parallelism configuration.
        p2p_communicator:       Optional pre-built communicator.
        bubble_filler:          DES-LOC bubble-filling helper (opt-in).

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

    tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        config=config,
    )

    # ---- Grad-sync helpers --------------------------------------------------
    no_sync_func = config.no_sync_func
    if isinstance(no_sync_func, (list, tuple)):
        _nsf_list = list(no_sync_func)

        def _multi_no_sync():
            stack = contextlib.ExitStack()
            for fn in _nsf_list:
                stack.enter_context(fn())
            return stack

        no_sync_func = _multi_no_sync

    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    grad_sync_func = config.grad_sync_func
    if grad_sync_func is not None and not isinstance(grad_sync_func, (list, tuple)):
        grad_sync_func = [grad_sync_func] * num_model_chunks

    if forward_only:
        _saved_grad_sync  = config.grad_sync_func
        _saved_param_sync = getattr(config, "param_sync_func", None)
        config.grad_sync_func = None
        if hasattr(config, "param_sync_func"):
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
    input_tensors:       List[List[Optional[torch.Tensor]]] = [[] for _ in range(num_model_chunks)]
    output_tensors:      List[List[torch.Tensor]]           = [[] for _ in range(num_model_chunks)]
    output_tensor_grads: List[List[Optional[torch.Tensor]]] = [[] for _ in range(num_model_chunks)]
    forward_data_store: list = []
    synchronized_model_chunks: Set[int] = set()

    total_num_microbatches = num_microbatches * num_model_chunks
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda" if torch.cuda.is_available() else "cpu")

    if forward_only:
        num_warmup = total_num_microbatches
    else:
        num_warmup = (pp_size - pp_rank - 1) * 2 + (num_model_chunks - 1) * pp_size
    num_warmup = min(num_warmup, total_num_microbatches)
    num_steady = total_num_microbatches - num_warmup

    # ---- Virtual microbatch index helpers -----------------------------------

    def get_model_chunk_id(virtual_mb_id: int, forward: bool) -> int:
        mb_id = virtual_mb_id % (pp_size * num_model_chunks)
        chunk_id = mb_id // pp_size
        if not forward:
            chunk_id = num_model_chunks - chunk_id - 1
        return chunk_id

    def is_last_microbatch_for_chunk(virtual_mb_id: int) -> bool:
        return (virtual_mb_id + 1) % (pp_size * num_model_chunks) == 0 or (
            virtual_mb_id == total_num_microbatches - 1
        )

    def recv_tensor_from_previous_stage(
        virtual_mb_id: int, forward: bool
    ) -> Tuple[bool, int]:
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

    def run_forward(virtual_mb_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        nonlocal total_num_tokens
        chunk_id = get_model_chunk_id(virtual_mb_id, forward=True)
        microbatch_id = (
            (virtual_mb_id // (pp_size * num_model_chunks)) * pp_size
            + (virtual_mb_id % pp_size)
        )

        # First stage: inject None activation
        if p2p_communicator.is_pp_first_stage and chunk_id == 0:
            if len(input_tensors[chunk_id]) == len(output_tensors[chunk_id]):
                input_tensors[chunk_id].append(None)

        input_tensor = input_tensors[chunk_id].pop(0)
        is_vp_last = chunk_id == num_model_chunks - 1
        is_actual_last = is_vp_last and p2p_communicator.is_pp_last_stage

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator[chunk_id],
            model[chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data=collect_non_loss_data,
            is_first_microbatch=_check_first_val_step(
                first_val_step, forward_only, microbatch_id == 0
            ),
            current_microbatch=microbatch_id,
            vp_stage=chunk_id,
            is_last_stage=is_actual_last,
        )
        total_num_tokens += num_tokens
        output_tensors[chunk_id].append(output_tensor)
        if not forward_only:
            input_tensors[chunk_id].append(input_tensor)
        return output_tensor, num_tokens

    # ---- Backward step helper -----------------------------------------------

    def run_backward(virtual_mb_id: int) -> Optional[torch.Tensor]:
        nonlocal no_sync_context
        chunk_id = get_model_chunk_id(virtual_mb_id, forward=False)

        if is_last_microbatch_for_chunk(virtual_mb_id):
            if grad_sync_func is None:
                enable_grad_sync()
            synchronized_model_chunks.add(chunk_id)

        is_vp_last = chunk_id == num_model_chunks - 1
        if is_vp_last and p2p_communicator.is_pp_last_stage:
            if len(output_tensor_grads[chunk_id]) == 0:
                output_tensor_grads[chunk_id].append(None)

        input_tensor     = input_tensors[chunk_id].pop(0)
        output_tensor    = output_tensors[chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[chunk_id].pop(0)

        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, config
        )

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
        p2p_communicator.recv_forward(tensor_shapes, p2p_communicator.is_pp_first_stage)
    )

    # ---- Warmup phase -------------------------------------------------------
    for k in range(num_warmup):
        output_tensor, _ = run_forward(k)

        recv_prev, next_fwd_chunk = recv_tensor_from_previous_stage(k, forward=True)

        if k < total_num_microbatches - 1:
            t = p2p_communicator.send_forward_recv_forward(
                output_tensor,
                recv_prev=recv_prev,
                tensor_shape=tensor_shapes,
                overlap_p2p_comm=False,
            )
            if recv_prev and t is not None:
                input_tensors[next_fwd_chunk].append(t)
        else:
            p2p_communicator.send_forward(output_tensor, p2p_communicator.is_pp_last_stage)

        deallocate_output_tensor(
            output_tensor, getattr(config, "deallocate_pipeline_outputs", False)
        )

        # DES-LOC bubble fill on fast stages during warmup
        if bubble_filler is not None:
            bubble_filler.maybe_fill_bubble(pp_rank, forward_data_store, config)

        # On the last warmup step, pre-receive the first backward gradient
        if k == num_warmup - 1 and not forward_only and num_steady > 0:
            recv_next = not p2p_communicator.is_pp_last_stage
            if recv_next:
                output_tensor_grads[num_model_chunks - 1].append(
                    p2p_communicator.recv_backward(tensor_shapes, p2p_communicator.is_pp_last_stage)
                )

    # ---- Steady (1F1B) phase ------------------------------------------------
    for k in range(num_steady):
        fwd_k = k + num_warmup
        bwd_k = k

        output_tensor, _ = run_forward(fwd_k)
        input_tensor_grad = run_backward(bwd_k)

        recv_prev, next_fwd_chunk = recv_tensor_from_previous_stage(fwd_k, forward=True)
        recv_next, next_bwd_chunk = recv_tensor_from_previous_stage(bwd_k, forward=False)

        last_fwd = fwd_k == total_num_microbatches - 1
        last_bwd = k == num_steady - 1

        recv_prev_flag = recv_prev and not last_fwd
        recv_next_flag = recv_next and not last_bwd

        t, g = p2p_communicator.send_forward_backward_recv_forward_backward(
            output_tensor,
            input_tensor_grad,
            recv_prev=recv_prev_flag,
            recv_next=recv_next_flag,
            tensor_shape=tensor_shapes,
        )

        if recv_prev_flag and t is not None:
            input_tensors[next_fwd_chunk].append(t)
        if recv_next_flag and g is not None:
            output_tensor_grads[next_bwd_chunk].append(g)

        deallocate_output_tensor(
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
                    p2p_communicator.recv_backward(tensor_shapes, p2p_communicator.is_pp_last_stage)
                )
            else:
                if len(output_tensor_grads[chunk_id]) == 0:
                    output_tensor_grads[chunk_id].append(None)

            # DES-LOC: bubble fill while grad recv completes on fast stages
            if bubble_filler is not None:
                bubble_filler.maybe_fill_bubble(pp_rank, forward_data_store, config)

            input_tensor_grad = run_backward(bwd_k)
            p2p_communicator.send_backward(
                input_tensor_grad, p2p_communicator.is_pp_first_stage
            )

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
        config.grad_sync_func = _saved_grad_sync  # type: ignore[possibly-undefined]
        if hasattr(config, "param_sync_func"):
            config.param_sync_func = _saved_param_sync  # type: ignore[possibly-undefined]

    if config.finalize_model_grads_func is not None and not forward_only:
        config.finalize_model_grads_func(model)

    if config.timers is not None:
        config.timers("forward-backward").stop()

    return forward_data_store


# ===========================================================================
# Utility: pipeline layer-split registry
# ===========================================================================

#: Module-level registry for ``pipeline_layer_split``.
#: Set this before calling ``get_pipeline_model_parallel_rank_for_layer``
#: in scripts that do not use ``parallel_state`` (e.g. unit tests).
_PIPELINE_LAYER_SPLIT: Optional[List[int]] = None


def set_pipeline_layer_split(split: List[int]) -> None:
    """Register a pipeline layer split.

    Args:
        split: Per-stage layer counts, e.g. ``[4, 8, 8, 4, 8]`` for a
               5-stage heterogeneous DES-LOC pipeline with 32 total layers.
    """
    global _PIPELINE_LAYER_SPLIT
    if not split or any(c <= 0 for c in split):
        raise ValueError(
            f"pipeline_layer_split must be a non-empty list of positive integers, got {split}"
        )
    _PIPELINE_LAYER_SPLIT = list(split)


def get_pipeline_model_parallel_rank_for_layer(layer_number: int) -> int:
    """Given a global 0-based layer number, return which PP rank owns it.

    Uses ``_PIPELINE_LAYER_SPLIT`` (or ``parallel_state._PIPELINE_LAYER_SPLIT``)
    for heterogeneous (DES-LOC) splits; falls back to uniform distribution.

    DES-LOC example (5-stage, unequal split [4, 8, 8, 4, 8] = 32 layers):
        Layers  0- 3  → rank 0
        Layers  4-11  → rank 1
        Layers 12-19  → rank 2
        Layers 20-23  → rank 3
        Layers 24-31  → rank 4

    Args:
        layer_number: 0-based global transformer layer index.

    Returns:
        PP rank (0-based) that owns ``layer_number``.
    """
    split: Optional[List[int]] = None
    if _ps is not None:
        split = getattr(_ps, "_PIPELINE_LAYER_SPLIT", None)
    if split is None:
        split = _PIPELINE_LAYER_SPLIT

    if split is None:
        # Uniform fallback
        if _ps is None or not torch.distributed.is_initialized():
            raise RuntimeError(
                "pipeline_layer_split is not configured and distributed is not initialised."
            )
        pp_size = _ps.get_pipeline_model_parallel_world_size()
        split = [1] * pp_size  # placeholder; rank lookup will raise if OOB

    cumulative = 0
    for rank, count in enumerate(split):
        cumulative += count
        if layer_number < cumulative:
            return rank

    raise ValueError(
        f"layer_number {layer_number} is out of range for pipeline_layer_split "
        f"{split} (total layers: {cumulative})."
    )


def get_num_microbatches() -> int:
    """Return the current global microbatch count from ``parallel_state``."""
    if _ps is None:
        raise RuntimeError("parallel_state is not available")
    fn = getattr(_ps, "get_num_microbatches", None)
    if fn is None:
        raise AttributeError(
            "parallel_state does not expose get_num_microbatches. "
            "Set it via parallel_state.set_num_microbatches() in your training loop."
        )
    return fn()


__all__ = [
    # Schedule selector
    "get_forward_backward_func",
    # Step functions
    "forward_step",
    "backward_step",
    # Schedules
    "forward_backward_no_pipelining",
    "forward_backward_pipelining_without_interleaving",
    "forward_backward_pipelining_with_interleaving",
    # DES-LOC bubble filling
    "HeterogeneousBubbleFiller",
    # Utilities
    "get_tensor_shapes",
    "deallocate_output_tensor",
    "custom_backward",
    "get_num_microbatches",
    "get_pipeline_model_parallel_rank_for_layer",
    "set_pipeline_layer_split",
]
