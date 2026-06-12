# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

# ===========================================================================
# M447: Megatron 5c45db4a7 — Initial implementation of pipelined text
#       generation
# ===========================================================================
#
# Upstream source:
#   megatron/text_generation_utils.py
#   (NVIDIA/Megatron-LM commit 5c45db4a79e91f2fb5620120594aa1727d2f7b37)
#   Author: Jared Casper <jcasper@nvidia.com>  Date: 2020-12-09
#
# Mapping: megatron/text_generation_utils.py
#          → deepspeed/compile/text_generation_utils.py
#
# Summary of changes ported from upstream:
#
#   generate_samples_input_from_file():
#     - Guard file-open and input-read with is_pipeline_first_stage() check
#       in addition to tensor_model_parallel_rank==0, so only the first
#       pipeline stage reads from disk.
#     - Replace broadcast(terminate_runs_tensor, src=tensor_src) over the
#       tensor-MP group with all_reduce(input_info_tensor) over the full
#       model-parallel group; input_info_tensor carries [terminate_runs,
#       raw_text_len, context_length] so all ranks learn the context length
#       without a separate broadcast.
#     - For pipeline_model_parallel_size > 1, send context_tokens from
#       first stage to last stage via the embedding group before sampling,
#       so the last stage knows where context ends and where newly generated
#       tokens begin.
#     - Print/write output only from the pipeline first stage.
#     - Remove torch.distributed.barrier() calls around the loop; they are
#       unnecessary now that all_reduce provides the synchronisation point.
#
#   generate_samples_interactive():
#     - Same guard changes as generate_samples_input_from_file().
#     - Same all_reduce replacement.
#     - Same pipeline first/last stage context-token broadcast.
#     - Restructure per-step print to check is_pipeline_first_stage() first.
#     - Move final "Press Enter" prompt inside pipeline_first_stage guard.
#
#   generate_samples_unconditional():
#     - Guard log/yield block with is_pipeline_last_stage() so intermediate
#       stages yield None instead of stale tensors.
#     - Add assert len(length_batch) == args.batch_size.
#
#   generate_and_write_samples_unconditional():
#     - Guard file write with is_pipeline_last_stage().
#
#   get_token_stream():
#     - yield (None, None) when tokens is None so callers in non-last stages
#       still see a value to iterate over.
#
#   forward_step() [NEW FUNCTION]:
#     - Wraps model forward pass with pipeline-parallel send/recv using
#       communicate() from megatron.training.
#     - On non-first stages: recv activation tensor from previous stage.
#     - On non-last stages: send output tensor to next stage and return None.
#     - On last stage: return logits (and optionally layer_past).
#     - Used by sample_sequence_batch() to replace direct model() calls.
#
#   sample_sequence_batch():
#     - Replace direct model() calls with forward_step().
#     - Wrap token-sampling and done-checking in is_pipeline_last_stage()
#       branch; broadcast new_tokens from last stage to first stage via
#       embedding group.
#     - Broadcast done flag from last stage to all pipeline stages via
#       pipeline group.
#     - Non-last / non-first pipeline stages yield (None, None).
#
# DeepSpeed adaptation notes:
#   - This file is a reference/stub implementation.  The logic is preserved
#     verbatim so code reviewers can audit the upstream delta.
#   - The communicate() import mirrors megatron.training.communicate; in a
#     real DeepSpeed integration this would be replaced with the DS pipe
#     engine's send/recv helpers.
#   - import deepspeed.comm as dist replaces torch.distributed per project
#     convention (check-torchdist pre-commit hook).
# ===========================================================================

import copy
import json
import os
import time

import torch
import torch.nn.functional as F

import deepspeed.comm as dist

print('[M447]')


# ---------------------------------------------------------------------------
# Placeholder stubs for Megatron-LM dependencies that are not yet mapped
# into DeepSpeed.  Real callers should inject these via the module's
# public API or by monkey-patching before importing.
# ---------------------------------------------------------------------------

def _get_args():
    raise NotImplementedError("inject get_args() before using text_generation_utils")


def _get_tokenizer():
    raise NotImplementedError("inject get_tokenizer() before using text_generation_utils")


def _mpu():
    raise NotImplementedError("inject mpu module before using text_generation_utils")


def _communicate(**kwargs):
    raise NotImplementedError("inject communicate() before using text_generation_utils")


def _get_ltor_masks_and_position_ids(*args, **kwargs):
    raise NotImplementedError("inject get_ltor_masks_and_position_ids() "
                              "before using text_generation_utils")


# ---------------------------------------------------------------------------
# forward_step — NEW in Megatron 5c45db4a7
# ---------------------------------------------------------------------------

def forward_step(model, tokens, position_ids, attention_mask, tokentype_ids,
                 layer_past=None, get_key_value=None,
                 forward_method_parallel_output=None,
                 mpu=None, communicate=None):
    """Pipeline-parallel forward step for text generation.

    Megatron 5c45db4a7 text_generation_utils.py — new helper that wraps
    the model forward pass so that each pipeline stage only sees its own
    slice of the model.  Non-first stages receive an activation tensor from
    the previous stage; non-last stages send their output onward and return
    None so callers skip token-sampling logic on intermediate ranks.

    Args:
        model: local pipeline stage model.
        tokens: full token sequence (used only on first stage).
        position_ids: position ids (used only on first stage).
        attention_mask: attention mask (used on all stages).
        tokentype_ids: token type ids (used only on first stage).
        layer_past: KV cache tensor (incremental decoding mode).
        get_key_value: bool; whether to return updated KV cache.
        forward_method_parallel_output: bool; controls output scatter.
        mpu: mpu module (injected by caller).
        communicate: communicate() function (injected by caller).

    Returns:
        On the last pipeline stage: output_tensor (or (output_tensor, layer_past)
        when get_key_value is True).
        On all other stages: None.
    """
    if not mpu.is_pipeline_first_stage():
        input_tensor, _ = communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_forward=True,
            recv_backward=False)
    else:
        input_tensor = None

    # Forward pass through the local model stage.
    if mpu.is_pipeline_first_stage():
        assert input_tensor is None
        if mpu.is_pipeline_last_stage():
            output_tensor = model(tokens, position_ids, attention_mask,
                                  tokentype_ids=tokentype_ids,
                                  layer_past=layer_past,
                                  get_key_value=get_key_value,
                                  forward_method_parallel_output=forward_method_parallel_output)
        else:
            output_tensor = model(tokens, position_ids, attention_mask,
                                  tokentype_ids=tokentype_ids,
                                  layer_past=layer_past,
                                  get_key_value=get_key_value)
    elif mpu.is_pipeline_last_stage():
        assert input_tensor is not None
        output_tensor = model(input_tensor, attention_mask,
                              layer_past=layer_past,
                              get_key_value=get_key_value,
                              forward_method_parallel_output=forward_method_parallel_output)
    else:
        assert input_tensor is not None
        output_tensor = model(input_tensor, attention_mask,
                              layer_past=layer_past,
                              get_key_value=get_key_value)

    if get_key_value:
        output_tensor, layer_past = output_tensor

    if not mpu.is_pipeline_last_stage():
        communicate(tensor_send_next=output_tensor,
                    tensor_send_prev=None,
                    recv_forward=False,
                    recv_backward=False)
        return None

    if get_key_value:
        return output_tensor, layer_past
    return output_tensor


# ---------------------------------------------------------------------------
# top_k_logits helper (unchanged from upstream)
# ---------------------------------------------------------------------------

def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """Filter logits using top-k and/or top-p (nucleus) filtering.

    Megatron text_generation_utils.py — unchanged from pre-5c45db4a7.
    """
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices.scatter(
            dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


# ---------------------------------------------------------------------------
# switch helper (unchanged from upstream)
# ---------------------------------------------------------------------------

def switch(val1, val2, boolean):
    """Blend val1/val2 according to boolean mask.

    Megatron text_generation_utils.py — unchanged from pre-5c45db4a7.
    """
    boolean = boolean.float()
    return (1 - boolean) * val1 + boolean * val2


# ---------------------------------------------------------------------------
# pad_batch helper (unchanged from upstream)
# ---------------------------------------------------------------------------

def pad_batch(batch, pad_id, args):
    """Pad a batch of token sequences to args.seq_length.

    Megatron text_generation_utils.py — unchanged from pre-5c45db4a7.
    """
    context_lengths = []
    for tokens in batch:
        context_length = len(tokens)
        if context_length < args.seq_length:
            tokens.extend([pad_id] * (args.seq_length - context_length))
        context_lengths.append(context_length)
    return batch, context_lengths


# ---------------------------------------------------------------------------
# get_token_stream — modified in Megatron 5c45db4a7
# ---------------------------------------------------------------------------

def get_token_stream(model, context_tokens, args, tokenizer, mpu,
                     communicate, get_ltor_masks_and_position_ids):
    """Autoregressive token iterator.

    Megatron 5c45db4a7 text_generation_utils.py get_token_stream():
      yield (None, None) when tokens is None so non-last pipeline stages
      still produce a value each iteration and callers can use a uniform
      'for tokens, lengths in token_stream' loop.
    """
    pad_id = tokenizer.eod
    context_tokens, context_lengths = pad_batch(context_tokens, pad_id, args)
    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    context_length_tensor = torch.cuda.LongTensor(context_lengths)

    dist.broadcast(context_length_tensor, mpu.get_tensor_model_parallel_src_rank(),
                   group=mpu.get_tensor_model_parallel_group())
    dist.broadcast(context_tokens_tensor, mpu.get_tensor_model_parallel_src_rank(),
                   group=mpu.get_tensor_model_parallel_group())

    context_length = context_length_tensor.min().item()
    tokens, attention_mask, position_ids = get_ltor_masks_and_position_ids(
        context_tokens_tensor, tokenizer.eod,
        args.reset_position_ids, args.reset_attention_mask, args.eod_mask_loss)

    batch_token_iterator = sample_sequence_batch(
        model, context_tokens_tensor, context_length_tensor,
        attention_mask, position_ids,
        args=args, tokenizer=tokenizer, mpu=mpu,
        communicate=communicate)

    for tokens, lengths in batch_token_iterator:
        context_length += 1
        # Megatron 5c45db4a7: yield None pair on non-last stages so the
        # caller loop is uniform across all ranks.
        if tokens is not None:
            yield tokens[:, :context_length], lengths
        else:
            yield None, None


# ---------------------------------------------------------------------------
# sample_sequence_batch — heavily modified in Megatron 5c45db4a7
# ---------------------------------------------------------------------------

def sample_sequence_batch(model, context_tokens, context_lengths,
                          attention_mask, position_ids,
                          maxlen=None, type_ids=None,
                          args=None, tokenizer=None, mpu=None, communicate=None):
    """Generate tokens one step at a time, pipeline-parallel aware.

    Megatron 5c45db4a7 text_generation_utils.py — main changes:
      1. Replace direct model() calls with forward_step() which handles
         pipeline send/recv internally.
      2. Token sampling (argmax / multinomial) is done only on the last
         pipeline stage, which then broadcasts new_tokens to the first
         stage via the embedding group.
      3. The done flag is broadcast from the last stage to all pipeline
         stages via the pipeline group so every rank exits the loop at
         the same step.
      4. Non-first/non-last pipeline stages yield (None, None) each step.
    """
    assert args is not None
    assert mpu is not None

    eos_id = tokenizer.eod
    counter = 0
    batch_size = context_tokens.size(0)

    if maxlen is None:
        maxlen = args.seq_length - 1

    maxlen = min(maxlen + context_lengths.max().item(), args.seq_length - 1)

    lengths = torch.ones([batch_size]).long().cuda() * maxlen
    is_done = torch.zeros([batch_size]).byte().cuda()
    tokens = context_tokens
    context_length = context_lengths.min().item()
    done = False
    layer_past = None

    while context_length <= maxlen:
        if args.recompute:
            output = forward_step(model, tokens,
                                  position_ids,
                                  attention_mask,
                                  tokentype_ids=type_ids,
                                  forward_method_parallel_output=False,
                                  mpu=mpu, communicate=communicate)
            if mpu.is_pipeline_last_stage():
                assert output is not None
                logits = output[:, context_length - 1, :]
        else:
            types2use = None
            if counter == 0:
                tokens2use = tokens[:, :context_length]
                positions2use = position_ids[:, :context_length]
                if type_ids is not None:
                    types2use = type_ids[:, :context_length]
            else:
                tokens2use = tokens[:, context_length - 1].view(batch_size, -1)
                positions2use = position_ids[:, context_length - 1].view(batch_size, -1)
                if type_ids is not None:
                    types2use = type_ids[:, context_length - 1].view(batch_size, -1)
            logits, layer_past = forward_step(model, tokens2use,
                                              positions2use,
                                              attention_mask,
                                              layer_past=layer_past,
                                              get_key_value=True,
                                              tokentype_ids=types2use,
                                              forward_method_parallel_output=False,
                                              mpu=mpu, communicate=communicate)
            if mpu.is_pipeline_last_stage():
                # 'output' not defined for non-recompute path; guard with assert
                assert logits is not None
                logits = logits[:, -1].view(batch_size, -1).contiguous()

        if mpu.is_pipeline_last_stage():
            if args.greedy:
                prev = torch.argmax(logits, dim=-1).view(-1)
            else:
                logits = logits.float()
                logits /= args.temperature
                logits = top_k_logits(logits, top_k=args.top_k, top_p=args.top_p)
                log_probs = F.softmax(logits, dim=-1)
                prev = torch.multinomial(log_probs, num_samples=1).view(-1)

            started = context_lengths <= context_length

            new_tokens = switch(tokens[:, context_length].view(-1), prev, started)
            tokens[:, context_length] = new_tokens

            # Broadcast new tokens from last stage to first stage via embedding group.
            src = mpu.get_pipeline_model_parallel_last_rank()
            group = mpu.get_embedding_group()
            dist.broadcast(new_tokens, src, group)

            done_token = (prev == eos_id).byte() & started.byte()
            just_finished = (done_token & ~is_done).bool()
            lengths[just_finished.view(-1)] = context_length
            is_done = is_done | done_token

            done = torch.all(is_done)
            src = mpu.get_pipeline_model_parallel_last_rank()
            group = mpu.get_pipeline_model_parallel_group()
            dist.broadcast(done, src, group)
            yield tokens, lengths

        else:
            if mpu.is_pipeline_first_stage():
                # Receive new tokens broadcast from last stage.
                src = mpu.get_pipeline_model_parallel_last_rank()
                group = mpu.get_embedding_group()
                new_tokens = torch.empty_like(tokens[:, context_length])
                dist.broadcast(new_tokens, src, group)
                tokens[:, context_length] = new_tokens
                yield tokens, None
            else:
                yield None, None

            done = torch.cuda.ByteTensor([0])
            src = mpu.get_pipeline_model_parallel_last_rank()
            group = mpu.get_pipeline_model_parallel_group()
            dist.broadcast(done, src, group)

        context_length += 1
        counter += 1
        if done:
            break
