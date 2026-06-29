# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Combined 1F1B schedules for MoE Expert-Parallel A2A communication hiding.

Ported from Megatron-LM/megatron/core/pipeline_parallel/combined_1f1b.py and
adapted for DeepSpeed's abstractions (no GPTModel.build_schedule_plan / AbstractSchedulePlan).

The key idea is to overlap the Expert-Parallel All-to-All communications of
microbatch N's forward pass with the attention/MLP compute of microbatch N-1's
backward pass, and vice versa.  This is done by issuing both operations on
separate CUDA streams and synchronising only at the boundaries where tensors
are exchanged.

Schedule summary (no-pipeline, M microbatches):
    Phase 0 : forward(mb=0)                           — warmup, no overlap
    Phase 1 : forward(mb=1) || backward(mb=0)
    ...
    Phase M-1 : forward(mb=M-1) || backward(mb=M-2)
    Phase M   : backward(mb=M-1)                      — cooldown, no overlap

For interleaved pipeline (VPP>1), the same forward/backward pairing happens
inside each call to forward_backward_helper_wrapper(), with the pre_*/post_*
hooks handling p2p communication on the main stream while compute runs on the
secondary stream.

DES-LOC heterogeneous note
--------------------------
The HeterogeneousBubbleFiller in schedules.py hooks into the pipeline bubble
that would otherwise be idle on fast ranks (H100).  The combined-1f1b path
reduces that bubble further: because we overlap A2A with compute, fast ranks
have less idle time during the normal 1F1B steady state, leaving the residual
bubble for the filler to exploit.
"""

from __future__ import annotations

import contextlib
from typing import Callable, List, Optional, Union

import torch

# ---------------------------------------------------------------------------
# Optional imports — same pattern as schedules.py
# ---------------------------------------------------------------------------
try:
    from deepspeed.core import parallel_state as _ps
except ImportError:
    _ps = None

try:
    from deepspeed.core.utils import get_attr_wrapped_model, get_model_config
except ImportError:
    def get_attr_wrapped_model(model, attr, **k):
        return getattr(model, attr, None)
    def get_model_config(model):
        return getattr(model, 'config', model)

# Import forward_step and backward_step from schedules (import at call-time
# to avoid circular imports at module load time).
def _import_schedule_fns():
    from deepspeed.core.pipeline_parallel.schedules import (
        forward_step,
        backward_step,
        check_first_val_step,
        forward_step_calc_loss,
    )
    return forward_step, backward_step, check_first_val_step, forward_step_calc_loss


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_cp_size(config) -> int:
    """Resolve context-parallel group size from config or parallel_state."""
    # Some configs carry cp_size directly (DES-LOC injection point)
    if hasattr(config, 'context_parallel_size') and config.context_parallel_size:
        return config.context_parallel_size
    if _ps is not None:
        try:
            return _ps.get_context_parallel_world_size()
        except Exception:
            pass
    return 1


def _release_tensor_storage(tensors):
    """Release CUDA tensor storage after all backward users are done.

    Mirrors Megatron's _release_tensor_storage — records a stream event
    before resizing to zero so the allocator can safely reclaim.
    """
    if tensors is None:
        return
    current_stream = torch.cuda.current_stream()
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
            tensor.record_stream(current_stream)
            tensor.untyped_storage().resize_(0)


def _get_high_priority_stream():
    """Return a high-priority CUDA stream for A2A communication overlap.

    We create one stream per device and cache it on the module to avoid
    re-allocating across microbatches.
    """
    device = torch.cuda.current_device()
    key = f'_combined_1f1b_hp_stream_{device}'
    stream = getattr(_get_high_priority_stream, key, None)
    if stream is None:
        _, high = torch.cuda.Stream.priority_range()
        stream = torch.cuda.Stream(device=device, priority=high)
        setattr(_get_high_priority_stream, key, stream)
    return stream


# ---------------------------------------------------------------------------
# Public API — no-pipelining (PP=1)
# ---------------------------------------------------------------------------

def combined_1f1b_schedule_for_no_pipelining(
    forward_step_func: Callable,
    data_iterator,
    model: torch.nn.Module,
    num_microbatches: int,
    input_tensor,          # always None for PP=1
    output_tensor_grad,    # always None for PP=1
    forward_data_store: list,
    config,
    collect_non_loss_data: bool,
    first_val_step: Optional[bool],
    forward_only: bool,
    no_sync_func: Callable,
    total_num_tokens: torch.Tensor,
    check_first_val_step_fn: Callable,
):
    """Overlap Expert-Parallel A2A with compute for PP=1 (no pipelining).

    Mirrors Megatron's combined_1f1b_schedule_for_no_pipelining but uses
    DeepSpeed's forward_step / backward_step primitives instead of
    GPTModel.build_schedule_plan / AbstractSchedulePlan.

    The overlap is achieved at the CUDA-stream level:
    - The forward pass issues its Expert-Parallel A2A on the high-priority
      stream returned by _get_high_priority_stream().
    - The backward pass runs on the default (low-priority) stream.
    - After each combined phase we synchronise via an event to ensure data
      consistency before advancing to the next phase.

    Because DeepSpeed does not yet expose per-layer scheduling hooks, the
    actual kernel-level interleaving is limited to what CUDA's scheduler
    can achieve with the two streams.  The functional correctness guarantee
    is identical to the conventional sequential schedule; the performance
    benefit depends on the GPU's ability to multiplex compute and A2A kernels.

    DES-LOC note: On H100 ranks (fast tier) config.desloc.bubble_filler is
    checked at the end of the warmup and cooldown phases so that any residual
    idle time is filled with prefetch microbatches.

    Args:
        forward_step_func: Callable ``(data_iterator, model) -> (output, loss_func)``.
        data_iterator: Dataset iterator for this PP stage.
        model: The single model module (PP=1 guarantees a scalar module here).
        num_microbatches: Total microbatches in this step.
        input_tensor: Always ``None`` for PP=1; kept for interface parity.
        output_tensor_grad: Always ``None`` for PP=1; kept for interface parity.
        forward_data_store: Mutable list to accumulate per-microbatch loss values.
        config: Model/training configuration object (``ModelParallelConfig``).
        collect_non_loss_data: When True forward_step returns raw model outputs.
        first_val_step: See ``check_first_val_step``.
        forward_only: If True skip backward passes (inference / eval).
        no_sync_func: Context manager that disables DDP gradient sync.
        total_num_tokens: Running token count (mutated in place).
        check_first_val_step_fn: ``partial(check_first_val_step, first_val_step, forward_only)``.

    Returns:
        Tuple ``(forward_data_store, total_num_tokens)``.
    """
    forward_step_fn, backward_step_fn, _, _ = _import_schedule_fns()
    cp_size = _get_cp_size(config)

    # Retrieve optional DES-LOC bubble filler (no-op if not configured)
    desloc = getattr(config, 'desloc', None)
    bubble_filler = getattr(desloc, 'bubble_filler', None)
    pp_rank = 0  # PP=1 → always rank 0

    # -----------------------------------------------------------------------
    # Phase 0: warmup — run first microbatch forward alone (no overlap)
    # -----------------------------------------------------------------------
    output_tensor_prev, num_tokens = forward_step_fn(
        forward_step_func, data_iterator, model, num_microbatches,
        input_tensor, forward_data_store, config,
        cp_group_size=cp_size,
        collect_non_loss_data=collect_non_loss_data,
        is_first_microbatch=check_first_val_step_fn(True),
        current_microbatch=0,
    )
    total_num_tokens += num_tokens

    # DES-LOC: fast rank may fill bubble after warmup forward
    if bubble_filler is not None:
        bubble_filler.maybe_fill_bubble(pp_rank, forward_data_store, config)

    # -----------------------------------------------------------------------
    # Steady state: for microbatches 1..N-1, overlap forward(i) + backward(i-1)
    # -----------------------------------------------------------------------
    # We use a high-priority stream for the forward pass so that the Expert-
    # Parallel A2A kernels it generates can run concurrently with the backward
    # compute on the default stream.
    hp_stream = _get_high_priority_stream()
    default_stream = torch.cuda.current_stream()

    with no_sync_func():
        for i in range(1, num_microbatches):
            # --- Launch forward(i) on the high-priority stream ---
            fwd_event = torch.cuda.Event()
            with torch.cuda.stream(hp_stream):
                output_tensor_cur, num_tokens = forward_step_fn(
                    forward_step_func, data_iterator, model, num_microbatches,
                    input_tensor, forward_data_store, config,
                    cp_group_size=cp_size,
                    collect_non_loss_data=collect_non_loss_data,
                    is_first_microbatch=check_first_val_step_fn(i == 0),
                    current_microbatch=i,
                )
                fwd_event.record(hp_stream)

            # --- Run backward(i-1) on the default stream concurrently ---
            if not forward_only:
                backward_step_fn(
                    input_tensor, output_tensor_prev, output_tensor_grad, config
                )
                del output_tensor_prev  # M4063: free before next forward result lands

            # Wait for forward(i) to complete before the next iteration uses
            # its output tensor as output_tensor_prev.
            default_stream.wait_event(fwd_event)
            total_num_tokens += num_tokens
            output_tensor_prev = output_tensor_cur

    # -----------------------------------------------------------------------
    # Cooldown: run backward for the last microbatch alone (enables grad sync)
    # -----------------------------------------------------------------------
    if bubble_filler is not None:
        bubble_filler.maybe_fill_bubble(pp_rank, forward_data_store, config)

    if not forward_only:
        backward_step_fn(
            input_tensor, output_tensor_prev, output_tensor_grad, config
        )
        del output_tensor_prev  # M4063

    return forward_data_store, total_num_tokens


# ---------------------------------------------------------------------------
# Public API — interleaved pipelining (VPP > 1)
# ---------------------------------------------------------------------------

def combined_1f1b_schedule_for_interleaved_pipelining(
    config,
    forward_step_func: Callable,
    data_iterator: list,
    model: list,
    num_microbatches: int,
    forward_data_store: list,
    forward_step_helper_preprocess: Callable,
    forward_step_helper_postprocess: Callable,
    backward_step_helper_preprocess: Callable,
    backward_step_helper_postprocess: Callable,
    get_microbatch_id_in_model_chunk: Callable,
    get_model_chunk_id: Callable,
    check_first_val_step_fn: Callable,
    is_first_microbatch_for_model_chunk: Callable,
    collect_non_loss_data: bool,
    f_virtual_microbatch_id: Optional[int] = None,
    b_virtual_microbatch_id: Optional[int] = None,
    pre_forward: Optional[Callable] = None,
    pre_backward: Optional[Callable] = None,
    post_forward: Optional[Callable] = None,
    post_backward: Optional[Callable] = None,
):
    """Overlap Expert-Parallel A2A with compute for interleaved PP (VPP > 1).

    Called from ``forward_backward_helper_wrapper`` in
    ``forward_backward_pipelining_with_interleaving`` when
    ``config.overlap_moe_expert_parallel_comm`` is True.

    This function mirrors Megatron's combined_1f1b_schedule_for_interleaved_pipelining
    but uses DeepSpeed's helper callables (preprocess / postprocess / forward_step /
    backward_step) instead of GPTModel.build_schedule_plan / AbstractSchedulePlan.

    Overlap strategy
    ----------------
    When both a forward microbatch (f_virtual_microbatch_id) and a backward
    microbatch (b_virtual_microbatch_id) are provided, we:
    1. Run the forward preprocess (tensor bookkeeping) on the main thread.
    2. Launch the forward compute on the high-priority CUDA stream.
    3. Run the backward preprocess on the main thread.
    4. Run the backward compute on the default stream (concurrently with fwd).
    5. Synchronise (wait for the forward event) before calling forward postprocess.
    6. Call forward and backward postprocess.

    When only one of the two IDs is provided (warmup / cooldown phases), we
    fall back to the sequential path so that p2p communication hooks in
    pre_forward/pre_backward execute correctly on the main stream.

    DES-LOC note: The HeterogeneousBubbleFiller is *not* called here because
    bubble filling is orchestrated at the outer schedule level
    (``forward_backward_pipelining_with_interleaving``); the combined path
    reduces the bubble that the filler needs to fill.

    Args:
        config: Model/training configuration object.
        forward_step_func: User forward step callable.
        data_iterator: List of iterators, one per model chunk.
        model: List of model chunks (one per virtual pipeline stage).
        num_microbatches: Number of microbatches per pipeline stage.
        forward_data_store: Mutable list accumulating loss values.
        forward_step_helper_preprocess: Callable that returns input_tensor for
            the given (virtual_mb_id, chunk_id, microbatch_id).
        forward_step_helper_postprocess: Callable updating output_tensors bookkeeping.
        backward_step_helper_preprocess: Callable returning
            (input_tensor, output_tensor, output_tensor_grad) for backward.
        backward_step_helper_postprocess: Callable that triggers grad sync etc.
        get_microbatch_id_in_model_chunk: Maps virtual_mb_id → microbatch_id.
        get_model_chunk_id: Maps (virtual_mb_id, forward) → model_chunk_id.
        check_first_val_step_fn: ``partial(check_first_val_step, first_val_step, fwd_only)``.
        is_first_microbatch_for_model_chunk: Returns True if this is the first
            microbatch for the given model chunk.
        collect_non_loss_data: Collect raw outputs instead of losses.
        f_virtual_microbatch_id: Forward microbatch index (None → skip forward).
        b_virtual_microbatch_id: Backward microbatch index (None → skip backward).
        pre_forward: Hook called before the forward compute (p2p recv wait etc.).
        pre_backward: Hook called before the backward compute.
        post_forward: Hook called after the forward compute (p2p send etc.).
        post_backward: Hook called after the backward compute.

    Returns:
        Tuple ``(forward_output_tensor, backward_input_tensor_grad)``.
        Either element may be ``None`` when the corresponding pass is skipped.
    """
    forward_step_fn, backward_step_fn, _, _ = _import_schedule_fns()
    cp_size = _get_cp_size(config)

    hp_stream = _get_high_priority_stream()
    default_stream = torch.cuda.current_stream()

    # ------------------------------------------------------------------
    # Resolve PP / VPP stage info for is_last_stage determination
    # ------------------------------------------------------------------
    num_model_chunks = len(model)
    vpp_size = config.virtual_pipeline_model_parallel_size if hasattr(config, 'virtual_pipeline_model_parallel_size') else num_model_chunks

    def _is_last_stage_for_chunk(chunk_id: int) -> bool:
        """True iff this (PP rank, VPP stage) is the final stage in the pipeline."""
        is_last_vp = (chunk_id == num_model_chunks - 1)
        if _ps is not None:
            try:
                is_last_pp = _ps.is_pipeline_last_stage(ignore_virtual=True)
                return is_last_vp and is_last_pp
            except Exception:
                pass
        return is_last_vp

    # ------------------------------------------------------------------
    # Resolve forward metadata
    # ------------------------------------------------------------------
    f_model_chunk_id = None
    f_microbatch_id = None
    f_input_tensor = None

    if f_virtual_microbatch_id is not None:
        f_model_chunk_id = get_model_chunk_id(f_virtual_microbatch_id, forward=True)
        f_microbatch_id = get_microbatch_id_in_model_chunk(f_virtual_microbatch_id, forward=True)
        f_input_tensor = forward_step_helper_preprocess(
            f_virtual_microbatch_id, f_model_chunk_id, f_microbatch_id
        )

    # ------------------------------------------------------------------
    # Resolve backward metadata
    # ------------------------------------------------------------------
    b_model_chunk_id = None
    b_input_tensor = None
    b_output_tensor = None
    b_output_tensor_grad = None

    if b_virtual_microbatch_id is not None:
        b_model_chunk_id = get_model_chunk_id(b_virtual_microbatch_id, forward=False)
        b_input_tensor, b_output_tensor, b_output_tensor_grad = (
            backward_step_helper_preprocess(b_virtual_microbatch_id, b_model_chunk_id)
        )

    # ------------------------------------------------------------------
    # Determine whether to overlap or run sequentially
    # ------------------------------------------------------------------
    both_active = (f_virtual_microbatch_id is not None) and (b_virtual_microbatch_id is not None)

    forward_output_tensor = None
    f_num_tokens = torch.tensor(0, dtype=torch.int)
    backward_input_tensor_grad = None

    if both_active:
        # ----------------------------------------------------------------
        # Overlapped forward + backward
        # ----------------------------------------------------------------
        # 1. pre-forward hook (p2p communication) runs on the default stream
        if pre_forward is not None:
            pre_forward()

        # 2. Launch forward compute on the high-priority stream
        fwd_event = torch.cuda.Event()
        with torch.cuda.stream(hp_stream):
            _fwd_out, f_num_tokens = forward_step_fn(
                forward_step_func,
                data_iterator[f_model_chunk_id],
                model[f_model_chunk_id],
                num_microbatches,
                f_input_tensor,
                forward_data_store,
                config,
                cp_group_size=cp_size,
                collect_non_loss_data=collect_non_loss_data,
                checkpoint_activations_microbatch=None,
                is_first_microbatch=check_first_val_step_fn(
                    is_first_microbatch_for_model_chunk(f_virtual_microbatch_id)
                ),
                current_microbatch=f_microbatch_id,
                vp_stage=f_model_chunk_id,
                is_last_stage=_is_last_stage_for_chunk(f_model_chunk_id),
            )
            fwd_event.record(hp_stream)

        # 3. pre-backward hook on the default stream (may wait on p2p recv)
        if pre_backward is not None:
            pre_backward()

        # 4. Backward compute on the default stream — overlaps with fwd A2A
        if not isinstance(b_input_tensor, list):
            _b_in = [b_input_tensor]
        else:
            _b_in = b_input_tensor
        for x in _b_in:
            if x is not None:
                x.retain_grad()

        if not isinstance(b_output_tensor, list):
            _b_out = [b_output_tensor]
        else:
            _b_out = b_output_tensor
        if not isinstance(b_output_tensor_grad, list):
            _b_out_grad = [b_output_tensor_grad]
        else:
            _b_out_grad = b_output_tensor_grad

        if _b_out_grad[0] is None and config.grad_scale_func is not None:
            _b_out[0] = config.grad_scale_func(_b_out[0])

        if _b_out[0] is not None and _b_out[0].requires_grad:
            from deepspeed.core.pipeline_parallel.schedules import custom_backward, deallocate_output_tensor
            if config.deallocate_pipeline_outputs:
                custom_backward(_b_out[0], _b_out_grad[0])
            else:
                torch.autograd.backward(_b_out[0], grad_tensors=_b_out_grad[0])

        _b_in_grad = [None]
        if _b_in is not None:
            _b_in_grad = []
            for x in _b_in:
                _b_in_grad.append(None if x is None else x.grad)

        if not isinstance(b_input_tensor, list):
            backward_input_tensor_grad = _b_in_grad[0]
        else:
            backward_input_tensor_grad = _b_in_grad

        # 5. Synchronise: wait for the forward compute to finish
        default_stream.wait_event(fwd_event)
        forward_output_tensor = _fwd_out

        # 6. post-forward hook (p2p send) on the default stream
        if post_forward is not None:
            forward_output_tensor = post_forward(forward_output_tensor)

        # 7. post-backward hook (p2p send) on the default stream
        if post_backward is not None:
            backward_input_tensor_grad = post_backward(backward_input_tensor_grad)

    else:
        # ----------------------------------------------------------------
        # Sequential path (warmup / cooldown — only one side active)
        # ----------------------------------------------------------------
        if f_virtual_microbatch_id is not None:
            if pre_forward is not None:
                pre_forward()
            forward_output_tensor, f_num_tokens = forward_step_fn(
                forward_step_func,
                data_iterator[f_model_chunk_id],
                model[f_model_chunk_id],
                num_microbatches,
                f_input_tensor,
                forward_data_store,
                config,
                cp_group_size=cp_size,
                collect_non_loss_data=collect_non_loss_data,
                checkpoint_activations_microbatch=None,
                is_first_microbatch=check_first_val_step_fn(
                    is_first_microbatch_for_model_chunk(f_virtual_microbatch_id)
                ),
                current_microbatch=f_microbatch_id,
                vp_stage=f_model_chunk_id,
                is_last_stage=_is_last_stage_for_chunk(f_model_chunk_id),
            )
            if post_forward is not None:
                forward_output_tensor = post_forward(forward_output_tensor)

        if b_virtual_microbatch_id is not None:
            if pre_backward is not None:
                pre_backward()
            backward_input_tensor_grad = backward_step_fn(
                b_input_tensor, b_output_tensor, b_output_tensor_grad, config
            )
            if post_backward is not None:
                backward_input_tensor_grad = post_backward(backward_input_tensor_grad)

    # ------------------------------------------------------------------
    # Postprocess bookkeeping (both active or single-side)
    # ------------------------------------------------------------------
    if f_model_chunk_id is not None:
        forward_step_helper_postprocess(f_model_chunk_id, forward_output_tensor, f_num_tokens)

    if b_model_chunk_id is not None:
        backward_step_helper_postprocess(b_virtual_microbatch_id)
        # Verify: if backward received activation from upstream, grad must exist
        if b_input_tensor is not None:
            assert backward_input_tensor_grad is not None, (
                "backward_input_tensor_grad is None despite b_input_tensor being set; "
                "check that the backward pass is running and output requires_grad."
            )

    return forward_output_tensor, backward_input_tensor_grad


__all__ = [
    "combined_1f1b_schedule_for_no_pipelining",
    "combined_1f1b_schedule_for_interleaved_pipelining",
]
