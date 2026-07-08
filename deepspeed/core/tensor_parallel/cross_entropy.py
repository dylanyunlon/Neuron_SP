# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Vocabulary-parallel cross-entropy loss.

Ported from Megatron-LM/megatron/core/tensor_parallel/cross_entropy.py with
DES-LOC extensions for heterogeneous TP groups.

The loss computation is split across TP ranks so that each rank only
materialises ``vocab_size / tp_world_size`` logits at a time, reducing peak
memory by the TP factor.  Numerical stability is maintained by subtracting
the global max logit (computed via all-reduce) before exponentiation.

DES-LOC extension — non-uniform vocab partitions
-------------------------------------------------
When the vocabulary size is not exactly divisible by the TP world size
(e.g. 32000 tokens across 5 GPUs), the standard ``VocabUtility`` produces
equal-sized partitions by padding.  The cross-entropy implementation
handles the padding rows correctly: masked-out logits contribute neither
to ``sum_exp`` nor to the predicted-logit term.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from deepspeed.core.tensor_parallel.utils import VocabUtility


# ---------------------------------------------------------------------------
# TP group helpers
# ---------------------------------------------------------------------------

def _get_tp_group() -> torch.distributed.ProcessGroup:
    from deepspeed.core.parallel_state import get_tensor_model_parallel_group
    return get_tensor_model_parallel_group()


def _pg_rank(group: torch.distributed.ProcessGroup) -> int:
    return torch.distributed.get_rank(group=group)


def _pg_size(group: torch.distributed.ProcessGroup) -> int:
    return torch.distributed.get_world_size(group=group)


# ---------------------------------------------------------------------------
# Stateless helper methods (reusable by fused kernels)
# ---------------------------------------------------------------------------

class VocabParallelCrossEntropy:
    """Numerically-stable cross-entropy over TP-sharded logits.

    All static methods are composable building blocks that can be called
    from both the autograd function below and from custom CUDA kernels.
    """

    @staticmethod
    def calculate_logits_max(
        vocab_parallel_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits_fp32, per-token max)``."""
        vocab_parallel_logits = vocab_parallel_logits.float()
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        return vocab_parallel_logits, logits_max

    @staticmethod
    def calculate_predicted_logits(
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        logits_max: torch.Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subtract global max, gather predicted logits, compute exp-sum."""
        # In-place subtraction
        vocab_parallel_logits -= logits_max.unsqueeze(dim=-1)

        # Mask tokens outside this partition
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Gather predicted logits
        partition_vocab_size = vocab_parallel_logits.size(-1)
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(
            start=0, end=logits_2d.size(0), device=logits_2d.device
        )
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0

        # Exponentiate and sum
        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)

        return target_mask, masked_target_1d, predicted_logits, sum_exp_logits, exp_logits

    @staticmethod
    def calculate_cross_entropy_loss(
        exp_logits: torch.Tensor,
        predicted_logits: torch.Tensor,
        sum_exp_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute ``loss = log(sum(exp)) - predicted_logit``."""
        loss = torch.log(sum_exp_logits) - predicted_logits
        # Normalise to softmax probabilities (re-used in backward)
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        return exp_logits, loss

    @staticmethod
    def prepare_gradient_calculation_operands(
        softmax: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare 2-D view and index tensors for the backward pass."""
        grad_input = softmax
        partition_vocab_size = softmax.size(-1)
        grad_2d = grad_input.view(-1, partition_vocab_size)
        arange_1d = torch.arange(
            start=0, end=grad_2d.size(0), device=grad_2d.device
        )
        softmax_update = 1.0 - target_mask.view(-1).float()
        return grad_2d, arange_1d, softmax_update, grad_input

    @staticmethod
    def calculate_gradients(
        grad_2d: torch.Tensor,
        arange_1d: torch.Tensor,
        masked_target_1d: torch.Tensor,
        softmax_update: torch.Tensor,
        grad_input: torch.Tensor,
        grad_output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the final gradient w.r.t. logits."""
        grad_2d[arange_1d, masked_target_1d] -= softmax_update
        grad_input.mul_(grad_output.unsqueeze(dim=-1))
        return grad_input


# ---------------------------------------------------------------------------
# Autograd function
# ---------------------------------------------------------------------------

class _VocabParallelCrossEntropy(torch.autograd.Function):
    """Autograd-compatible vocab-parallel cross-entropy with optional label smoothing."""

    @staticmethod
    def forward(
        ctx,
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        label_smoothing: float = 0.0,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> torch.Tensor:
        if tp_group is None:
            tp_group = _get_tp_group()

        vocab_parallel_logits, logits_max = (
            VocabParallelCrossEntropy.calculate_logits_max(vocab_parallel_logits)
        )
        torch.distributed.all_reduce(
            logits_max, op=torch.distributed.ReduceOp.MAX, group=tp_group
        )

        get_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size(-1)
        rank = _pg_rank(tp_group)
        world_size = _pg_size(tp_group)
        vocab_start, vocab_end = get_range(partition_vocab_size, rank, world_size)

        (target_mask, masked_target_1d, predicted_logits, sum_exp_logits, exp_logits) = (
            VocabParallelCrossEntropy.calculate_predicted_logits(
                vocab_parallel_logits, target, logits_max, vocab_start, vocab_end
            )
        )

        torch.distributed.all_reduce(
            predicted_logits, op=torch.distributed.ReduceOp.SUM, group=tp_group
        )
        torch.distributed.all_reduce(
            sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=tp_group
        )

        exp_logits, loss = VocabParallelCrossEntropy.calculate_cross_entropy_loss(
            exp_logits, predicted_logits, sum_exp_logits
        )

        vocab_size = exp_logits.size(-1)
        if label_smoothing > 0:
            assert 1.0 > label_smoothing > 0.0
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)
            log_probs = torch.log(exp_logits)
            mean_log_probs = log_probs.mean(dim=-1)
            loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs

        ctx.label_smoothing, ctx.vocab_size = label_smoothing, vocab_size
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        softmax, target_mask, masked_target_1d = ctx.saved_tensors
        label_smoothing = ctx.label_smoothing
        vocab_size = ctx.vocab_size

        (grad_2d, arange_1d, softmax_update, grad_input) = (
            VocabParallelCrossEntropy.prepare_gradient_calculation_operands(
                softmax, target_mask
            )
        )

        if label_smoothing > 0:
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)
            grad_2d[arange_1d, masked_target_1d] -= (1.0 - smoothing) * softmax_update
            average_grad = 1 / vocab_size
            grad_2d[arange_1d, :] -= smoothing * average_grad
            grad_input.mul_(grad_output.unsqueeze(dim=-1))
        else:
            grad_input = VocabParallelCrossEntropy.calculate_gradients(
                grad_2d, arange_1d, masked_target_1d,
                softmax_update, grad_input, grad_output,
            )

        return grad_input, None, None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def vocab_parallel_cross_entropy(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.0,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Cross-entropy loss over TP-sharded logits.

    Args:
        vocab_parallel_logits: ``[seq_len, batch, vocab_size / tp_world_size]``
        target: ``[seq_len, batch]`` — ground-truth token indices
        label_smoothing: smoothing factor in ``(0, 1)``; 0 means no smoothing
        tp_group: tensor-parallel process group (uses default if ``None``)

    Returns:
        Per-token loss tensor of shape ``[seq_len, batch]``.
    """
    return _VocabParallelCrossEntropy.apply(
        vocab_parallel_logits, target, label_smoothing, tp_group
    )
