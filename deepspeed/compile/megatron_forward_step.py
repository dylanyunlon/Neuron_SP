# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ===========================================================================
# M831: Megatron a7539b0f8 — pipelining works
# ===========================================================================
#
# Upstream source:
#   megatron/inference/forward_step.py
#   (NVIDIA/Megatron-LM commit a7539b0f863d0a5fc8acea8e4944ef76f2493a9f)
#   Author: mshoeybi <mshoeybi@nvidia.com>  Date: 2021-10-07
#
# Mapping: megatron/inference/forward_step.py
#          → deepspeed/compile/megatron_forward_step.py
#          (project convention: megatron top-level/inference → deepspeed/compile/)
#
# Summary of changes ported from upstream:
#
#   InferenceParams (redesigned):
#     - Drop micro_batch_size_list / micro_batch_index / allocate list
#       approach entirely.
#     - New init signature: __init__(max_batch_size, max_sequence_length)
#     - New attributes: sequence_len_offset = 0, batch_size_offset = 0
#     - Keep allocate_key_value_memory = True flag.
#
#   ForwardStep (replaces ForwardStepBase + SimplePipeliningForwardStep
#                         + NoPipeliningForwardStep + forward_step_provider):
#     - Unified class: __init__(model, max_batch_size, max_sequence_length)
#     - Sets self.constant = 512 to decide pipelining threshold.
#     - __call__ dispatches to _with_pipelining_forward_step when
#       batch*seq >= constant, else _no_pipelining_forward_step.
#
#   New module-level helpers (replace old class hierarchy):
#     - _get_recv_buffer_dtype(args)
#     - _allocate_recv_buffer(batch_size, sequence_length)
#     - _forward_step_helper(model, tokens, position_ids, attention_mask,
#                            inference_params, recv_buffer=None)
#       Uses torch.distributed.recv / send directly (not recv_forward /
#       send_forward from p2p_communication).
#     - _no_pipelining_forward_step(...)  — updates sequence_len_offset
#     - _with_pipelining_forward_step(...)
#       Divides batch into micro-batches, pre-allocates logits on last
#       stage, iterates micro-batches, updates batch_size_offset /
#       sequence_len_offset.
#
# Adaptation (DS import style):
#   • megatron.get_args() / megatron.mpu kept as injected references;
#     callers must supply get_args and mpu when constructing ForwardStep
#     or call the module-level functions.
#   • Original imports (megatron.get_args, megatron.mpu) replaced by
#     keyword arguments with defaults pointing to None — callers inject
#     them at construction time.  This keeps the module free of hard
#     Megatron dependencies.
# ===========================================================================

print('[M831]')

"""Forward step utilities — M831 port of Megatron a7539b0f8."""

from collections.abc import Iterable

import torch


# ---------------------------------------------------------------------------
# InferenceParams
# ---------------------------------------------------------------------------

class InferenceParams:
    """Inference-time key/value cache parameters.

    Megatron a7539b0f8 forward_step.py — redesigned from the old
    micro_batch_size_list / micro_batch_index scheme to a flat
    (max_batch_size, sequence_len_offset, batch_size_offset) model.

    Attributes:
        max_sequence_length: maximum sequence length (context + generation).
        max_batch_size:   maximum batch size supported by the kv allocation.
        sequence_len_offset: tokens processed so far (updated after each step).
        batch_size_offset:   offset into the batch dimension for micro-batch
                             slicing; reset to 0 after each full pass.
        allocate_key_value_memory: True on first forward pass; set to False
                                   after memory has been allocated.
    """

    def __init__(self, max_batch_size, max_sequence_length):
        assert max_sequence_length > 0
        assert max_batch_size > 0
        self.max_sequence_length = max_sequence_length
        print(f'[M1735][InferenceParams] max_sequence_length={max_sequence_length}')
        self.max_batch_size = max_batch_size
        self.sequence_len_offset = 0
        self.batch_size_offset = 0
        self.allocate_key_value_memory = True


# ---------------------------------------------------------------------------
# ForwardStep
# ---------------------------------------------------------------------------

class ForwardStep:
    """Unified inference forward-step wrapper.

    Megatron a7539b0f8 forward_step.py — replaces the old
    forward_step_provider() + ForwardStepBase + NoPipeliningForwardStep +
    SimplePipeliningForwardStep class hierarchy with a single class that
    automatically dispatches to the pipelining or no-pipelining path based on
    a simple threshold (self.constant = 512 tokens).

    Args:
        model:            model or list of pipeline-stage models.
        max_batch_size:   max batch size (passed to InferenceParams).
        max_sequence_length: max sequence length (passed to InferenceParams).
        get_args:         callable returning the global args namespace.
        mpu:              mpu module providing pipeline-parallel helpers.
    """

    def __init__(self, model, max_batch_size, max_sequence_length,
                 get_args=None, mpu=None):
        # Make sure model is in eval mode.
        if isinstance(model, Iterable):
            for this_model in model:
                this_model.eval()
        else:
            model.eval()
        self.model = model
        self._get_args = get_args
        self._mpu = mpu

        # Threshold: use pipelining when batch_size * seq_len >= constant.
        self.constant = 512

        # Initialize inference parameters.
        self.inference_params = InferenceParams(max_batch_size,
                                                max_sequence_length)

    def __call__(self, tokens, position_ids, attention_mask):
        if tokens.size(0) * tokens.size(1) >= self.constant:
            micro_batch_size = max(1, self.constant // tokens.size(1))
            return _with_pipelining_forward_step(
                self.model, tokens, position_ids, attention_mask,
                self.inference_params, micro_batch_size,
                get_args=self._get_args, mpu=self._mpu)
        else:
            return _no_pipelining_forward_step(
                self.model, tokens, position_ids, attention_mask,
                self.inference_params,
                mpu=self._mpu)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _get_recv_buffer_dtype(args):
    """Return the dtype for inter-stage receive buffers.

    Megatron a7539b0f8: use float32 when fp32_residual_connection is set,
    otherwise use the model's params_dtype.
    """
    if args.fp32_residual_connection:
        return torch.float
    return args.params_dtype


def _allocate_recv_buffer(batch_size, sequence_length, get_args=None, mpu=None):
    """Allocate inter-stage receive buffer of shape [s, b, h].

    Returns None on the first pipeline stage (no tensor received).
    """
    if mpu is not None and mpu.is_pipeline_first_stage():
        return None
    args = get_args()
    recv_size = (sequence_length, batch_size, args.hidden_size)
    return torch.empty(recv_size,
                       dtype=_get_recv_buffer_dtype(args),
                       device=torch.cuda.current_device())


def _forward_step_helper(model, tokens, position_ids, attention_mask,
                         inference_params, recv_buffer=None,
                         mpu=None):
    """Single forward step through one pipeline stage.

    Megatron a7539b0f8: uses torch.distributed.recv/send directly rather
    than the p2p_communication helpers so that the recv buffer can be
    pre-allocated and reused across micro-batches.

    After the forward pass, clears the allocate_key_value_memory flag so
    that subsequent calls reuse the already-allocated KV buffers.
    """
    batch_size = tokens.size(0)
    sequence_length = tokens.size(1)
    if recv_buffer is None:
        recv_buffer = _allocate_recv_buffer(batch_size, sequence_length,
                                            mpu=mpu)

    # Receive from previous stage.
    if mpu is not None and not mpu.is_pipeline_first_stage():
        torch.distributed.recv(
            recv_buffer,
            src=mpu.get_pipeline_model_parallel_prev_rank())

    # Forward pass through the model.
    model.set_input_tensor(recv_buffer)
    output_tensor = model(tokens, position_ids, attention_mask,
                          inference_params=inference_params)

    # Send output to the next stage.
    if mpu is not None and not mpu.is_pipeline_last_stage():
        torch.distributed.send(
            output_tensor,
            mpu.get_pipeline_model_parallel_next_rank())

    # Make sure we do not allocate context memory anymore.
    if inference_params.allocate_key_value_memory:
        inference_params.allocate_key_value_memory = False

    return output_tensor


def _no_pipelining_forward_step(model, tokens, position_ids, attention_mask,
                                inference_params, recv_buffer=None,
                                mpu=None):
    """Forward step without micro-batch pipelining.

    Megatron a7539b0f8: simple single-pass forward; updates
    inference_params.sequence_len_offset after the call.
    """
    # Run a simple forward pass.
    output_tensor = _forward_step_helper(model, tokens, position_ids,
                                         attention_mask, inference_params,
                                         recv_buffer=recv_buffer,
                                         mpu=mpu)
    # Update the sequence length offset.
    inference_params.sequence_len_offset += tokens.size(1)

    logits = None
    if mpu is None or mpu.is_pipeline_last_stage():
        logits = output_tensor

    return logits


def _with_pipelining_forward_step(model, tokens, position_ids, attention_mask,
                                  inference_params, micro_batch_size,
                                  get_args=None, mpu=None):
    """Forward step with micro-batch pipelining.

    Megatron a7539b0f8: splits the batch dimension into micro-batches of
    size micro_batch_size, pre-allocates a shared recv buffer for all but the
    last (possibly smaller) micro-batch, and accumulates logits on the last
    pipeline stage.

    After all micro-batches are processed:
      - inference_params.sequence_len_offset is incremented by sequence_length.
      - inference_params.batch_size_offset is reset to 0.
    """
    sequence_length = tokens.size(1)
    batch_size = tokens.size(0)

    # Divide the batch dimension into micro batches.
    num_micro_batches, last_chunk = divmod(batch_size, micro_batch_size)
    if last_chunk > 0:
        num_micro_batches += 1

    # Preallocate memory for output logits.
    logits = None
    if mpu is None or mpu.is_pipeline_last_stage():
        args = get_args()
        logits = torch.empty(
            (batch_size, sequence_length, args.padded_vocab_size),
            dtype=torch.float32, device=torch.cuda.current_device())

    # Preallocate recv buffer (reused across full-size micro-batches).
    recv_buffer = _allocate_recv_buffer(micro_batch_size, sequence_length,
                                        get_args=get_args, mpu=mpu)

    for micro_batch_index in range(num_micro_batches):
        # Slice along the batch dimension.
        start = micro_batch_index * micro_batch_size
        end = min(start + micro_batch_size, batch_size)
        this_micro_batch_size = end - start
        tokens2use = tokens[start:end, ...]
        position_ids2use = position_ids[start:end, ...]

        # For the last (possibly smaller) micro-batch, don't reuse the buffer.
        if this_micro_batch_size != micro_batch_size:
            recv_buffer = None
        output = _forward_step_helper(model, tokens2use, position_ids2use,
                                      attention_mask, inference_params,
                                      recv_buffer=recv_buffer,
                                      mpu=mpu)

        # Adjust the batch size offset to account for the micro-batch.
        inference_params.batch_size_offset += this_micro_batch_size

        # Copy logits.
        if mpu is None or mpu.is_pipeline_last_stage():
            logits[start:end, ...] = output

    # Once we are done with all the micro-batches, adjust offsets.
    inference_params.sequence_len_offset += sequence_length
    # Reset the batch size offset for the next call.
    inference_params.batch_size_offset = 0

    return logits
