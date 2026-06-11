# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

import deepspeed.comm as dist


print('[M55]')


class _VocabSequenceParallelCrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_seq_parallel_logits, target, sp_group):
        # vocab_seq_parallel_logits: [S/P, B, V]
        # target: [S/P, B]
        # return: [S, B]

        # M461 (57064fd): memory opt — avoid full logits clone before softmax.
        # Megatron insight: subtract max in-place, extract predicted logit (scalar
        # slice) BEFORE materialising exp over the full vocab dimension.  The clone
        # + contiguous on the scalar slice is mandatory: the advanced-index result
        # is non-contiguous and shares storage with vocab_seq_parallel_logits, so
        # the subsequent in-place exp would silently corrupt it.
        # Knuth §7.1: "premature pessimisation is the root of all waste" — here the
        # original .clone() of the *full* [S/P,B,V] tensor wastes O(SBV) memory
        # before a single scalar is extracted.  The fix pays O(SB) instead.

        # Step 1: max across vocab (in-place friendly; no clone needed for max)
        logits_max = vocab_seq_parallel_logits.max(dim=-1)[0]   # [S/P, B]
        dist.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=sp_group)

        # Step 2: subtract max in-place — avoids O(SBV) clone
        vocab_seq_parallel_logits.sub_(logits_max.unsqueeze(dim=-1))

        ctx.vocab_size = vocab_seq_parallel_logits.size(2)
        S_loc, B, V = vocab_seq_parallel_logits.shape
        logits_2d = vocab_seq_parallel_logits.view(-1, V)          # [S/P*B, V]
        target_1d = target.view(-1)                                 # [S/P*B]

        # Step 3: extract predicted-logit BEFORE exp — O(S/P*B) gather, not O(SBV)
        arange_1d = torch.arange(logits_2d.size(0), device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, target_1d].clone().contiguous()
        # all_reduce predicted logits across SP ranks (zero out non-local tokens)
        predicted_logits_1d_full = predicted_logits_1d.clone()
        dist.all_reduce(predicted_logits_1d_full, op=torch.distributed.ReduceOp.SUM, group=sp_group)

        # Step 4: in-place exp (reuses vocab_seq_parallel_logits storage — no alloc)
        exp_logits = vocab_seq_parallel_logits
        torch.exp(vocab_seq_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)                    # [S/P, B]
        dist.all_reduce(sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=sp_group)

        # Step 5: loss = log(Σexp) − predicted_logit
        predicted_logits = predicted_logits_1d_full.view(S_loc, B)
        loss = torch.log(sum_exp_logits) - predicted_logits        # [S/P, B]

        # M461 DIAG: print memory-opt path stats periodically
        _r = dist.get_rank()
        if not hasattr(_VocabSequenceParallelCrossEntropy, '_call_n'):
            _VocabSequenceParallelCrossEntropy._call_n = 0
        _VocabSequenceParallelCrossEntropy._call_n += 1
        if _VocabSequenceParallelCrossEntropy._call_n % 200 == 1:
            with torch.no_grad():
                sp_world_size_tmp = dist.get_world_size(sp_group)
                sp_rank_tmp = dist.get_rank(sp_group)
                print(f"[SP-CE-M461] rank={_r} sp_rank={sp_rank_tmp}/{sp_world_size_tmp} "
                      f"call#{_VocabSequenceParallelCrossEntropy._call_n} "
                      f"logits_shape={list(vocab_seq_parallel_logits.shape)} "
                      f"local_loss_mean={loss.mean().item():.6f} "
                      f"local_loss_std={loss.std().item():.4f} "
                      f"sum_exp_norm={sum_exp_logits.float().norm().item():.4f} "
                      f"mem_opt=no_clone+inplace_exp")

        sp_world_size = dist.get_world_size(sp_group)
        sp_rank = dist.get_rank(sp_group)
        ctx.sp_world_size = sp_world_size
        ctx.sp_rank = sp_rank
        ctx.seqlen = S_loc * sp_world_size

        # Gather loss across SP ranks → [S, B]
        loss_all = torch.empty(ctx.seqlen,
                               B,
                               dtype=loss.dtype,
                               device=loss.device)
        dist.all_gather_into_tensor(loss_all, loss, group=sp_group)

        # M461 DIAG: post-gather full loss statistics
        if _VocabSequenceParallelCrossEntropy._call_n % 200 == 1:
            with torch.no_grad():
                print(f"[SP-CE-M461] rank={_r} full_loss={list(loss_all.shape)} "
                      f"full_loss_mean={loss_all.mean().item():.6f}")

        # Save softmax (needed by backward for grad computation).
        # Recompute from already-exponentiated exp_logits / sum_exp_logits;
        # avoids a second full-vocab exp.  Knuth §1.4: "The best programs are
        # written so that computing machines can perform them quickly and so that
        # human beings can understand them clearly" — both served here.
        with torch.no_grad():
            softmax_saved = exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(softmax_saved, target)

        return loss_all

    @staticmethod
    def backward(ctx, grad_output):
        softmax, target = ctx.saved_tensors

        step_seqlen = ctx.seqlen // ctx.sp_world_size
        sp_rank = ctx.sp_rank
        grad_output_part = grad_output[step_seqlen * sp_rank:step_seqlen * (sp_rank + 1), :]

        grad_input = softmax
        grad_2d = grad_input.view(-1, ctx.vocab_size)
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)

        grad_2d[arange_1d, target.view(-1)] -= 1
        grad_input.mul_(grad_output_part.unsqueeze(dim=-1))

        return grad_input, None, None, None


def vocab_sequence_parallel_cross_entropy(vocab_parallel_logits, target, sp_group):
    return _VocabSequenceParallelCrossEntropy.apply(vocab_parallel_logits, target, sp_group)
