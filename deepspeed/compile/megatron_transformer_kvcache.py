# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ===========================================================================
# M831: Megatron a7539b0f8 — pipelining works
# ===========================================================================
#
# Upstream source:
#   megatron/model/transformer.py  —  ParallelAttention class
#   (NVIDIA/Megatron-LM commit a7539b0f863d0a5fc8acea8e4944ef76f2493a9f)
#   Author: mshoeybi <mshoeybi@nvidia.com>  Date: 2021-10-07
#
# Mapping: megatron/model/transformer.py (ParallelAttention kv-cache section)
#          → deepspeed/compile/megatron_transformer_kvcache.py
#          (project convention: megatron/model/ → deepspeed/compile/)
#
# Summary of changes ported from upstream:
#
#   ParallelAttention.__init__():
#     - Remove: inference_key_memory_list / inference_value_memory_list /
#               inference_current_sequence_len_list  (list-per-micro-batch)
#     - Add:    inference_key_memory = None
#               inference_value_memory = None
#       (single unified allocation covering max_batch_size)
#
#   ParallelAttention.forward() — key-value pre-allocation block:
#     - Old: allocate a list of tensors, one per entry in
#       inference_params.micro_batch_size_list
#     - New: allocate a single tensor of shape
#       [max_sequence_len, max_batch_size, np, hn] once using
#       inference_params.max_batch_size
#
#   ParallelAttention.forward() — else (no inference_params):
#     - Old: reset inference_key_memory_list, inference_value_memory_list,
#            inference_current_sequence_len_list to None
#     - New: reset inference_value_memory, inference_current_sequence_len
#            (note: inference_current_sequence_len is the new per-call
#             tracking variable, distinct from the old list)
#
#   ParallelAttention.forward() — "Adjust key and value for inference" block:
#     - Old: index by micro_batch_index; use per-micro-batch memory;
#            advance inference_current_sequence_len_list[micro_batch_index]
#     - New: use batch_size_offset / sequence_len_offset from
#            InferenceParams (from megatron_forward_step.py M831);
#            slice [sequence_start:sequence_end, batch_start:batch_end, ...]
#            into the unified memory tensors.
#
# This file provides:
#   • apply_inference_kvcache_to_attention() — a functional helper that
#     applies the new kv-cache logic given pre-allocated buffers and an
#     InferenceParams object.  Callers (e.g. a ParallelAttention port)
#     should call this instead of re-implementing the indexing inline.
#   • allocate_kvcache(inference_params, allocate_memory_fn) — allocates
#     (or no-ops) the unified kv-cache tensors and returns them.
#
# Note: this file does NOT port the full ParallelAttention class since
#   deepspeed/compile/ does not yet contain a ParallelAttention.  It
#   documents and provides the functional core of the kv-cache change so
#   future ParallelAttention ports can import it directly.
# ===========================================================================

print('[M831]')

"""ParallelAttention kv-cache helpers — M831 port of Megatron a7539b0f8."""

import torch


# ---------------------------------------------------------------------------
# KV-cache allocation
# ---------------------------------------------------------------------------

def allocate_kvcache(inference_params, allocate_memory_fn):
    """Allocate unified kv-cache tensors for inference.

    Megatron a7539b0f8 transformer.py — replaces the old list-per-micro-batch
    allocation with a single pair of tensors sized by max_batch_size.

    Args:
        inference_params: InferenceParams instance (from megatron_forward_step).
            Must have .allocate_key_value_memory (bool), .max_sequence_len,
            .max_batch_size.
        allocate_memory_fn: callable(max_sequence_len, max_batch_size) → Tensor.
            Typically ParallelAttention._allocate_memory().

    Returns:
        (key_memory, value_memory): pair of tensors, or (None, None) if
            allocation is not needed.
    """
    if inference_params is None or not inference_params.allocate_key_value_memory:
        return None, None
    inf_max_seq_len = inference_params.max_sequence_len
    inf_max_batch_size = inference_params.max_batch_size
    key_memory = allocate_memory_fn(inf_max_seq_len, inf_max_batch_size)
    value_memory = allocate_memory_fn(inf_max_seq_len, inf_max_batch_size)
    return key_memory, value_memory


# ---------------------------------------------------------------------------
# KV-cache read/write during forward
# ---------------------------------------------------------------------------

def apply_inference_kvcache_to_attention(
        key_layer, value_layer,
        inference_params,
        inference_key_memory,
        inference_value_memory):
    """Apply kv-cache read/write for inference.

    Megatron a7539b0f8 transformer.py — "Adjust key and value for inference"
    block.  Replaces the old micro_batch_index / per-list indexing with
    batch_size_offset / sequence_len_offset from InferenceParams.

    Writes the current key/value slice into the unified kv-cache tensors, then
    reads back the full accumulated context (from position 0 to sequence_end)
    for the relevant batch slice.

    Args:
        key_layer:   current key tensor   [sq, b, np, hn]
        value_layer: current value tensor [sq, b, np, hn]
        inference_params: InferenceParams with .batch_size_offset and
            .sequence_len_offset.
        inference_key_memory:   pre-allocated tensor [max_seq, max_b, np, hn]
        inference_value_memory: pre-allocated tensor [max_seq, max_b, np, hn]

    Returns:
        (key_layer, value_layer): updated tensors covering the full context
            so far for the current batch slice.
    """
    batch_start = inference_params.batch_size_offset
    batch_end = batch_start + key_layer.size(1)
    assert batch_end <= inference_key_memory.size(1), (
        f"batch_end={batch_end} > inference_key_memory.size(1)="
        f"{inference_key_memory.size(1)}")

    sequence_start = inference_params.sequence_len_offset
    sequence_end = sequence_start + key_layer.size(0)
    assert sequence_end <= inference_key_memory.size(0), (
        f"sequence_end={sequence_end} > inference_key_memory.size(0)="
        f"{inference_key_memory.size(0)}")

    # Copy key and values into the unified cache.
    inference_key_memory[sequence_start:sequence_end,
                         batch_start:batch_end,
                         ...] = key_layer
    inference_value_memory[sequence_start:sequence_end,
                            batch_start:batch_end,
                            ...] = value_layer

    # Read back the full accumulated context for this batch slice.
    key_layer = inference_key_memory[
        :sequence_end, batch_start:batch_end, ...]
    value_layer = inference_value_memory[
        :sequence_end, batch_start:batch_end, ...]

    return key_layer, value_layer
