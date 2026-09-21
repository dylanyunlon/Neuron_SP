"""
Microbatch Uniformity Guard , enforce identical num_microbatches across all ranks.

Fix for issue #591 Blocker 2:
    desloc_engine.py:2156 , HeteroStepBatchScheduler.schedule() returns
    per-rank num_microbatches.  But gather_full_params() inside forward fires
    one all_gather_into_tensor per layer per microbatch.  If rank 0 does 2
    microbatches and rank 1 does 1, rank 0 fires 64 all_gathers and rank 1
    fires 32 -> NCCL deadlock.

Solution:
    broadcast_uniform_microbatch_count() all-reduces the MAX of all ranks'
    num_microbatches so every rank loops the same number of times.  Ranks
    whose real data is exhausted before the max run zero-loss dummy batches
    (forward + backward with loss *= 0.0).

AST call chain (6 nodes):
    launch_7b_3gpu.sh
      -> run_pretrain.py
        -> DesLocEngine.train()
          -> hetero_scheduler.schedule() -> MicrobatchAllocation
            -> broadcast_uniform_microbatch_count()   <-- THIS MODULE
              -> for micro in range(num_microbatches)
                -> gather_full_params() -> all_gather_into_tensor()

Public API:
    broadcast_uniform_microbatch_count(local_count, group) -> int
    PaddedMicrobatchIterator(real_iter, real_count, padded_count, ...)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def broadcast_uniform_microbatch_count(
    local_count: int,
    group: Optional[dist.ProcessGroup] = None,
    *,
    device: Optional[torch.device] = None,
) -> int:
    """All-reduce local num_microbatches to the MAX across all ranks.

    Every rank MUST call this with the same process group so the collective
    is symmetric.  The returned value is guaranteed identical on all ranks.

    Args:
        local_count: This rank's num_microbatches from the scheduler.
        group: Process group for the all-reduce (None = WORLD).
        device: CUDA device for the tensor (auto-detected if None).

    Returns:
        The global maximum num_microbatches (>= local_count on every rank).
    """
    if not dist.is_initialized():
        return local_count

    world_size = dist.get_world_size(group=group)
    if world_size <= 1:
        return local_count

    if device is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    count_tensor = torch.tensor([local_count], dtype=torch.int64, device=device)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.MAX, group=group)
    uniform_count = int(count_tensor.item())

    if uniform_count != local_count:
        rank = dist.get_rank()
        logger.warning(
            "[MicrobatchGuard] rank=%d local_count=%d -> uniform_count=%d "
            "(padded %d dummy microbatches to prevent ZeRO-3 all_gather deadlock)",
            rank, local_count, uniform_count, uniform_count - local_count,
        )

    return uniform_count


class PaddedMicrobatchIterator:
    """Iterator that yields real batches then zero-loss dummy batches.

    When a rank's real microbatch count is less than the uniform maximum
    (because it has a weaker GPU or different scheduler output), this
    iterator transparently pads with dummy batches that:
      1. Run forward + backward normally (keeping ZeRO-3 all_gather in sync).
      2. Scale loss by 0.0 so dummy batches contribute zero gradient.

    Usage::

        padded = PaddedMicrobatchIterator(
            real_iter=data_iter,
            real_count=allocation.num_microbatches,
            padded_count=uniform_count,
            seq_len=cfg.seq_len,
            vocab_size=cfg.vocab_size,
            device=local_dev,
        )
        for micro in range(uniform_count):
            input_ids, labels, is_dummy = next(padded)
            loss, scaled = engine.forward(input_ids, labels, ...)
            if is_dummy:
                scaled = scaled * 0.0  # zero gradient contribution

    Args:
        real_iter: The real data iterator.
        real_count: Number of real microbatches this rank should process.
        padded_count: Total (uniform) microbatch count after guard.
        seq_len: Sequence length for dummy tensors.
        vocab_size: Vocabulary size for dummy label range.
        device: Target CUDA device.
        micro_batch_size: Batch dimension for dummy tensors (default 1).
    """

    def __init__(
        self,
        real_iter: Iterator,
        real_count: int,
        padded_count: int,
        seq_len: int = 2048,
        vocab_size: int = 32000,
        device: Optional[torch.device] = None,
        micro_batch_size: int = 1,
    ) -> None:
        self._real_iter = real_iter
        self._real_count = real_count
        self._padded_count = padded_count
        self._seq_len = seq_len
        self._vocab_size = vocab_size
        self._device = device or torch.device("cuda")
        self._micro_batch_size = micro_batch_size
        self._pos = 0

        # Pre-allocate dummy tensors (reused across pad iterations to avoid
        # repeated allocation on the CUDA allocator).
        self._dummy_ids: Optional[torch.Tensor] = None
        self._dummy_labels: Optional[torch.Tensor] = None

    def _get_dummy_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return pre-allocated dummy input_ids and labels tensors."""
        if self._dummy_ids is None:
            self._dummy_ids = torch.zeros(
                (self._micro_batch_size, self._seq_len),
                dtype=torch.long,
                device=self._device,
            )
            # Labels set to -100 (ignored by CrossEntropyLoss) for extra safety,
            # even though loss will be scaled to 0.  Belt-and-suspenders.
            self._dummy_labels = torch.full(
                (self._micro_batch_size, self._seq_len),
                fill_value=-100,
                dtype=torch.long,
                device=self._device,
            )
        return self._dummy_ids, self._dummy_labels

    def __iter__(self) -> "PaddedMicrobatchIterator":
        self._pos = 0
        return self

    def __next__(self) -> Tuple[Any, Any, bool]:
        """Return (input_ids, labels, is_dummy).

        For real microbatches (pos < real_count): fetch from real_iter.
        For dummy microbatches (pos >= real_count): return pre-allocated zeros.
        """
        if self._pos >= self._padded_count:
            raise StopIteration

        is_dummy = self._pos >= self._real_count
        self._pos += 1

        if is_dummy:
            ids, labels = self._get_dummy_batch()
            return ids, labels, True

        # Real data
        raw = next(self._real_iter)
        if isinstance(raw, dict):
            ids = raw["tokens"]
            labels = raw.get("labels")
        elif isinstance(raw, (tuple, list)):
            ids = raw[0]
            labels = raw[1] if len(raw) > 1 else None
        else:
            ids = raw
            labels = None

        return ids, labels, False

    def __len__(self) -> int:
        return self._padded_count


def log_microbatch_guard_stats(
    step: int,
    local_count: int,
    uniform_count: int,
    rank: int,
) -> Dict[str, Any]:
    """Build a stats dict for logging/W&B tracking.

    Returns a dict suitable for wandb.log() or JSON serialisation.
    """
    stats = {
        "gate/step": step,
        "gate/rank": rank,
        "gate/local_microbatches": local_count,
        "gate/uniform_microbatches": uniform_count,
        "gate/dummy_microbatches": uniform_count - local_count,
        "gate/is_padded": uniform_count > local_count,
    }
    if uniform_count != local_count:
        logger.info(
            "[MicrobatchGuard] step=%d rank=%d: %d real + %d dummy = %d total",
            step, rank, local_count,
            uniform_count - local_count, uniform_count,
        )
    return stats
