# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ===========================================================================
# M831: Megatron a7539b0f8 — pipelining works
# ===========================================================================
#
# Upstream source:
#   megatron/inference/generation.py
#   (NVIDIA/Megatron-LM commit a7539b0f863d0a5fc8acea8e4944ef76f2493a9f)
#   Author: mshoeybi <mshoeybi@nvidia.com>  Date: 2021-10-07
#
# Mapping: megatron/inference/generation.py
#          → deepspeed/compile/megatron_generation.py
#          (project convention: megatron/inference/ → deepspeed/compile/)
#
# Summary of changes ported from upstream (delta from pre-a7539b0f8):
#
#   Imports:
#     - Replace `from .forward_step import forward_step_provider` with
#       `from .forward_step import ForwardStep`
#       (mapped here to: from megatron_forward_step import ForwardStep)
#
#   generate_tokens_probs_and_return_on_first_stage():
#     - Replace:
#         forward_step = forward_step_provider(model, batch_size, 4,
#                                              max_sequence_length)
#       with:
#         forward_step = ForwardStep(model, batch_size, max_sequence_length)
#       (hard-coded micro_batch_size=4 is gone; ForwardStep auto-selects
#        micro-batch size via self.constant = 512 threshold)
#
#     - Comment out the early-exit `if done: break` block:
#         #if done:
#         #    break
#       (left as commented out per upstream; generation always runs to
#        max_sequence_length to avoid pipeline stalls)
#
# Adaptation (DS import style):
#   • megatron.* imports replaced by injected callables (get_args,
#     get_tokenizer, mpu, etc.) to keep the module free of hard Megatron deps.
#   • communication helpers (copy_from_last_to_first_pipeline_stage, etc.)
#     mapped to deepspeed/compile/megatron_p2p_communication.py equivalents.
# ===========================================================================

print('[M831]')

"""Generation utilities — M831 port of Megatron a7539b0f8."""

import torch
import torch.nn.functional as F

from .megatron_forward_step import ForwardStep


# ---------------------------------------------------------------------------
# Main generation entry point
# ---------------------------------------------------------------------------

def generate_tokens_probs_and_return_on_first_stage(
        model, tokens, lengths,
        return_output_log_probs=False,
        return_all_log_probs=False,
        temperature=1.0,
        get_args=None,
        get_tokenizer=None,
        mpu=None,
        copy_from_last_to_first_pipeline_stage=None,
        broadcast_from_last_pipeline_stage=None,
        broadcast_from_last_to_first_pipeline_stage=None,
        get_ltor_masks_and_position_ids=None,
        sample=None):
    """Main token generation function.

    Megatron a7539b0f8 generation.py — key change vs prior version:
      - forward_step_provider() replaced by ForwardStep(model, batch_size,
        max_sequence_length).  No explicit micro_batch_size argument;
        ForwardStep auto-selects via constant=512 threshold.
      - `if done: break` early-exit commented out to avoid pipeline stalls.

    Arguments:
        model: XXX
        tokens: prompt tokens extended to be of size [b, max-sequence-length]
        lengths: original prompt length, size: [b]
        return_output_log_probs: flag to calculate the log probability of
            the generated tokens. Note that the log probability is the one
            after logits are modified for sampling.
        return_all_log_probs: flag to calculate the log probability of across
            all the tokens (vocab size). Note that the log probability is the
            one after logits are modified for sampling.
        temperature: sampling temperature.
        get_args, get_tokenizer, mpu: injected Megatron helpers.
        copy_from_last_to_first_pipeline_stage,
        broadcast_from_last_pipeline_stage,
        broadcast_from_last_to_first_pipeline_stage: pipeline comm helpers.
        get_ltor_masks_and_position_ids: mask builder.
        sample: token sampling function.

    Note: Outside of model, other parameters only need to be available on
          rank 0.

    Outputs: Note that size is adjusted to a lower value than
             max-sequence-length if generation is terminated early.
        tokens: prompt and generated tokens. size: [b, :]
        generated_sequence_lengths: total length (including prompt) of
            the generated sequence. size: [b]
        output_log_probs: log probability of the selected tokens. size: [b, s]
        all_log_probs: log probability of all the tokens.
            size: [b, s, vocab-size]
    """

    args = get_args()
    tokenizer = get_tokenizer()

    batch_size = tokens.size(0)
    min_prompt_length = lengths.min().item()
    max_sequence_length = tokens.size(1)
    max_sequence_length = min(max_sequence_length, args.max_position_embeddings)

    # M831: ForwardStep replaces forward_step_provider(model, batch_size, 4, ...)
    # The hard-coded micro_batch_size=4 is gone; ForwardStep picks it via
    # self.constant = 512.
    forward_step = ForwardStep(model, batch_size, max_sequence_length,
                               get_args=get_args, mpu=mpu)

    # Added termination_id to support the case that we want to terminate the
    # generation once that id is generated.
    if hasattr(args, 'eos_id'):
        termination_id = args.eos_id
    else:
        termination_id = tokenizer.eod

    # ===================
    # Pre-allocate memory
    # ===================

    # Log probability of the sequence (prompt + generated tokens).
    output_log_probs = None
    output_log_probs_size = (batch_size, max_sequence_length - 1)
    # Log probability of all tokens for the sequence.
    all_log_probs = None
    all_log_probs_size = (batch_size, max_sequence_length - 1,
                          args.padded_vocab_size)
    # Lengths of generated sequence including prompts.
    generated_sequence_lengths = None
    if mpu.is_pipeline_last_stage():
        if return_output_log_probs:
            output_log_probs = torch.empty(output_log_probs_size,
                                           dtype=torch.float32,
                                           device=torch.cuda.current_device())
        if return_all_log_probs:
            all_log_probs = torch.empty(all_log_probs_size,
                                        dtype=torch.float32,
                                        device=torch.cuda.current_device())
        generated_sequence_lengths = torch.ones(
            batch_size, dtype=torch.int64,
            device=torch.cuda.current_device()) * max_sequence_length
    # Whether we have reached a termination id.
    is_generation_done = torch.zeros(batch_size, dtype=torch.uint8,
                                     device=torch.cuda.current_device())

    # =============
    # Run inference
    # =============

    attention_mask, position_ids = _build_attention_mask_and_position_ids(
        tokens, get_ltor_masks_and_position_ids=get_ltor_masks_and_position_ids)

    with torch.no_grad():
        prev_context_length = 0
        for context_length in range(min_prompt_length, max_sequence_length):

            # Pick the slice that we need to pass through the network.
            tokens2use = tokens[:, prev_context_length:context_length]
            positions2use = position_ids[:, prev_context_length:context_length]
            attention_mask2use = attention_mask[
                ..., prev_context_length:context_length, :context_length]

            # logits will be meaningful only in the last pipeline stage.
            logits = forward_step(tokens2use, positions2use, attention_mask2use)

            if mpu.is_pipeline_last_stage():
                # Always the last stage should have an output.
                assert logits is not None

                # Sample.
                last_token_logits = logits[:, -1, :]
                new_sample, updated_last_token_logits = sample(
                    last_token_logits,
                    greedy=args.greedy,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    temperature=temperature,
                    vocab_size=tokenizer.vocab_size)
                # Now that we have the sample and updated logits,
                # update the main logits and input tokens.
                # If a prompt length is smaller or equal to current context
                # length, it means we have started generating tokens.
                started = lengths <= context_length
                # Update the logits.
                last_token_logits.masked_scatter_(
                    started.unsqueeze(1), updated_last_token_logits[started])
                # and the tokens.
                tokens[started, context_length] = new_sample[started]

                # Calculate the log probabilities.
                if return_output_log_probs or return_all_log_probs:
                    log_probs = F.log_softmax(logits, dim=2)
                    if return_all_log_probs:
                        all_log_probs[:,
                                      prev_context_length:context_length,
                                      :] = log_probs
                    if return_output_log_probs:
                        # Pick the tokens that we need to get the log
                        # probabilities for. Note that next input token is
                        # the token which we selected in the current logits,
                        # so shift by 1.
                        indices = torch.unsqueeze(
                            tokens[
                                :,
                                (prev_context_length + 1):(context_length + 1)],
                            2)
                        output_log_probs[:,
                                         prev_context_length:context_length] = \
                            torch.gather(log_probs, 2, indices).squeeze(2)

            # Update the tokens on the first stage so the next input to
            # the network is correct.
            copy_from_last_to_first_pipeline_stage(batch_size, torch.int64,
                                                   tokens[:, context_length])

            # Update the context length for the next token generation.
            prev_context_length = context_length

            # Check if all the sequences have hit the termination_id.
            done = None
            if mpu.is_pipeline_last_stage():
                done_token = (new_sample == termination_id).byte() & \
                    started.byte()
                just_finished = (done_token & ~is_generation_done).bool()
                generated_sequence_lengths[just_finished.view(-1)] = \
                    context_length + 1
                is_generation_done = is_generation_done | done_token
                done = torch.all(is_generation_done)
            done = broadcast_from_last_pipeline_stage(1, torch.uint8,
                                                      tensor=done)
            # M831: commented out to avoid pipeline stalls — generation always
            # runs to max_sequence_length.
            #if done:
            #    break

    # ===================================================
    # Update the length based on max generated length.
    # ===================================================

    tokens = tokens[:, :(context_length + 1)]
    if mpu.is_pipeline_last_stage():
        if return_output_log_probs:
            output_log_probs = output_log_probs[:, :context_length]
        if return_all_log_probs:
            all_log_probs = all_log_probs[:, :context_length, :]

    # ======================================
    # Broadcast to the first pipeline stage.
    # ======================================

    generated_sequence_lengths = broadcast_from_last_to_first_pipeline_stage(
        batch_size, torch.int64, generated_sequence_lengths)
    if return_output_log_probs:
        output_log_probs_size = (batch_size, context_length)
        output_log_probs = broadcast_from_last_to_first_pipeline_stage(
            output_log_probs_size, torch.float32, output_log_probs)
    if return_all_log_probs:
        all_log_probs_size = (batch_size, context_length,
                              args.padded_vocab_size)
        all_log_probs = broadcast_from_last_to_first_pipeline_stage(
            all_log_probs_size, torch.float32, all_log_probs)

    return tokens, generated_sequence_lengths, output_log_probs, \
        all_log_probs


# ---------------------------------------------------------------------------
# Attention mask and position ids builder
# ---------------------------------------------------------------------------

def _build_attention_mask_and_position_ids(
        tokens, get_ltor_masks_and_position_ids=None):
    """Build the attention mask and position ids for the input tokens.

    Megatron a7539b0f8 generation.py — unchanged helper; eod_token is not
    used so it is safe to set it to None.
    """
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=None,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False)

    return attention_mask, position_ids
