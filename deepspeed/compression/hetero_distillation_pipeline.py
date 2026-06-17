# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""HeteroDistillationPipeline — offline logits-based knowledge distillation for DES-LOC.

Mirrors Megatron 277c4f804 — Offline Logits-Based Knowledge Distillation,
reinterpreted as HeteroDistillationPipeline: teacher runs on the strong tier
(H100), student trains on the weak tier (A6000), with logits relayed through
a 1.5 TB CPU DRAM buffer pool in deepspeed/compression/.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Upstream design intent (277c4f804)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The Megatron commit introduces a two-phase offline distillation workflow:

**Phase 1 — Teacher logits saving (``LogitsSaverHooks``):**
  A forward hook intercepts the teacher model's output layer, computes top-K
  log-probabilities in a TP-aware fashion (local top-K → global log-sum-exp
  → global top-K via gather to TP rank 0), and streams compressed payloads
  into batched tar archives named ``cp{C}_dp{D}__{I}.tar``.  Crucially:

  - The full vocab log-softmax is *never* materialised; the log-sum-exp
    denominator is computed from the sparse top-K candidates only, avoiding
    the 4× memory overhead of a dense softmax on large vocabularies.
  - Indices are compressed to 17 bits (uint16 lower bits + one bool tensor
    for the 17th bit), reducing disk footprint by ~4× vs. int64.
  - Top-P (nucleus) masking is optionally applied after top-K selection to
    further reduce storage by dropping low-probability tail entries.
  - Writes are flushed asynchronously through the existing checkpoint queue
    so they don't block the forward/backward training loop.
  - An ``override_ckpt_iteration`` mechanism allows rewinding the data-loader
    to replay exactly the samples seen during teacher saving, enabling the
    student to consume teacher logits in guaranteed iteration order.

**Phase 2 — Student KD loss (``LossFuncCallable``):**
  A custom loss function loads the cached tar shards via a streaming
  ``TeacherTarDataset`` (IterableDataset + DataLoader with pin_memory and
  prefetch) and computes forward KL divergence between the student's live
  logits and the sparse teacher top-K log-probabilities.  Key design points:

  - Each DP rank loads only its own cp-dp shard; no cross-rank file I/O.
  - DP resharding is supported: upscaling (saved DP < current DP) strides
    through one saved file; downscaling (saved DP > current DP) interleaves
    multiple saved files.
  - The student log-probabilities are normalised in a TP-aware fashion by
    gathering the sparse teacher top-K vocab positions before computing the
    KL divergence, avoiding a full-vocab all-gather in the hot path.
  - A ``StudentLogitsCapture`` forward hook captures differentiable student
    logits from the output layer so the loss function receives the exact
    same logits the teacher produced, not the post-rearranged pipeline output.
  - KD loss and LM loss are blended with configurable ``kd_loss_alpha``.

**Ancillary changes:**
  - ``frozen_expert_bias`` flag on ``TopKRouter`` prevents the expert-bias
    update in ``finalize_model_grads`` from firing on frozen modules — needed
    when running the student in eval mode alongside a training teacher.
  - ``--freeze-all-layers`` arg freezes the entire model (student-pass-only
    distillation without gradient wrt model weights).
  - ``--override-ckpt-iteration`` resets ``consumed_train_samples`` so the
    data loader replays from the exact sample offset used during teacher saving.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DES-LOC adaptation rationale
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DES-LOC co-locates two distinct GPU tiers in a single training job:

  H100  (strong tier): 80 GB VRAM, ~1979 TFLOP BF16 — runs the teacher.
  A6000 (weak tier):   48 GB VRAM, ~310  TFLOP BF16 — trains the student.

The upstream ``LogitsSaverHooks`` was designed for homogeneous teacher runs
followed by separate student runs.  DES-LOC reinterprets this as a *live
pipelining* problem: teacher and student coexist in the same job, connected
through a CPU DRAM relay rather than a POSIX filesystem.  This change enables:

1. **Teacher-student co-scheduling**: the teacher produces logits on H100s
   during microbatch N while the student trains on microbatch N-1's cached
   logits, hiding teacher I/O latency behind student compute.

2. **1.5 TB CPU DRAM buffer**: commodity dual-socket AMD EPYC nodes provide
   ≥1.5 TB of DDR5 DRAM — far more than any GPU's HBM.  The CPU buffer pool
   (``CPULogitsBuffer``) holds several hundred iterations of top-K payloads
   with zstd compression, amortising PCIe transfer cost and smoothing over
   jitter in teacher throughput.

3. **No POSIX filesystem dependency**: tar writes to disk are replaced by
   in-process queue insertions; the downstream student loader reads from the
   same in-process queue, eliminating filesystem latency entirely.

4. **DP/TP-aware relay**: the bridge respects the tier topology produced by
   ``HeteroParallelismConfig`` (see ``pipe/hetero_mimo_parallelism.py``) —
   teacher TP rank 0 enqueues into the buffer; student DP rank N dequeues
   from the slice matching its data-parallel coordinate.

5. **Frozen expert bias**: the ``FrozenExpertBiasGuard`` mirrors the upstream
   ``frozen_expert_bias`` attribute on ``TopKRouter``, but attaches to
   DeepSpeed's MoE router wrapper so the expert-bias update skips frozen
   modules without requiring a megatron.core import.

6. **Checkpoint-iteration override**: ``CheckpointIterationOverride`` mirrors
   the upstream ``--override-ckpt-iteration`` mechanism but integrates with
   DeepSpeed's ``load_checkpoint`` so the data-loader offset is reset in the
   DeepSpeed engine's saved state rather than Megatron's ``args`` namespace.

Key classes
-----------
``TierRole``
    Enum: TEACHER (strong tier, H100) or STUDENT (weak tier, A6000).
    Determined at process-group construction time from the device index and
    the ``HeteroParallelismConfig`` tier assignment.

``TopKConfig``
    Dataclass carrying K, optional P (nucleus threshold), min_k, save_dtype,
    and the KD loss blend weight (``kd_loss_alpha``).  Maps 1-to-1 to
    Megatron's ``--logits-save-top-k`` / ``--logits-load-kd-loss-alpha`` args.

``CPULogitsBuffer``
    Thread-safe bounded deque holding zstd-compressed top-K payloads in CPU
    RAM.  The teacher's ``LogitsCaptureHook`` enqueues here; the student's
    ``CachedLogitsLoader`` dequeues.  Buffer capacity in bytes is capped by
    ``max_buffer_bytes`` (default 1.5 TB, configurable for smaller nodes).
    Separate per-dp-rank slots prevent head-of-line blocking between DP ranks.

``LogitsCaptureHook``
    Forward hook for the teacher model's output layer.  Mirrors upstream
    ``LogitsSaverHooks._forward_hook``.  Performs local top-K on raw logits
    → global log-sum-exp over TP group → global top-K gather to TP rank 0 →
    optional nucleus truncation → zstd compression → enqueue to CPULogitsBuffer.
    Unlike upstream, the async checkpoint queue is replaced by a direct
    ``CPULogitsBuffer.enqueue()`` call; no tar I/O is issued until the student
    requests a flush to disk for long-term storage.

``CachedLogitsLoader``
    Iterator consumed by the student training loop.  Dequeues from
    ``CPULogitsBuffer`` for the current DP rank, decompresses, unpacks 17-bit
    indices, and yields ``(values_list, indices_list)`` exactly as upstream
    ``TeacherTarDataset.__iter__`` does.  Supports a fallback path that reads
    pre-computed tar shards from disk (identical to upstream) for runs where
    the teacher and student execute in separate jobs.

``StudentLogitsCapture``
    Forward hook mirroring upstream ``StudentLogitsCapture`` from
    ``cached_logits_loss.py``.  Captures the differentiable student logits
    from the output layer so the KD loss receives the exact logits, not a
    post-rearranged pipeline output.  No megatron.core dependency; uses a
    plain ``register_forward_hook``.

``KDLoss``
    Computes forward KL divergence between student log-probabilities and
    sparse teacher top-K log-probabilities.  Mirrors upstream ``topk_kl_div``
    from ``cached_logits_loss.py``.  TP-aware: student logits are normalised
    across the TP vocab shard before gathering the sparse teacher top-K
    positions; no full-vocab all-gather is required.

``FrozenExpertBiasGuard``
    Context manager that sets ``frozen_expert_bias = True`` on all MoE router
    submodules before a student eval pass, then clears it afterward.  Mirrors
    the upstream ``frozen_expert_bias`` attribute addition to ``TopKRouter``
    and its guard in ``_update_router_expert_bias``.

``CheckpointIterationOverride``
    Helper that patches DeepSpeed's loaded checkpoint state dict to reset the
    iteration counter and ``consumed_train_samples`` to a target iteration,
    enabling the student data-loader to replay samples from the offset
    corresponding to where teacher logit saving began.

``HeteroDistillationPipeline``
    Top-level orchestrator.  Instantiated once per process; its ``setup()``
    method:
      1. Detects the current rank's tier (TEACHER vs. STUDENT) via the
         device topology.
      2. Constructs a ``CPULogitsBuffer`` shared between the teacher's
         ``LogitsCaptureHook`` and the student's ``CachedLogitsLoader``.
      3. Attaches the capture hook to the teacher model (if TEACHER tier).
      4. Attaches ``StudentLogitsCapture`` to the student model (if STUDENT).
      5. Returns a ``KDLoss`` callable for the student training loop.
    ``teardown()`` removes hooks and drains/flushes the buffer.

Diagnostic events (rank-0, ds_logger.info + print, one line per event):
  [DS-HDP] SETUP_TEACHER   — teacher hook attached, buffer capacity.
  [DS-HDP] SETUP_STUDENT   — student capture and KDLoss initialised.
  [DS-HDP] ENQUEUE         — teacher enqueued one iteration payload (size, iter).
  [DS-HDP] DEQUEUE         — student dequeued one iteration payload (iter).
  [DS-HDP] BUFFER_HIGH     — buffer occupancy > 75% of max_buffer_bytes.
  [DS-HDP] BUFFER_STALL    — teacher blocked waiting for student to drain buffer.
  [DS-HDP] KD_LOSS         — per-step KD loss value and alpha blend weight.
  [DS-HDP] FALLBACK_LM     — student fell back to LM-only loss (ignore_errors=True).
  [DS-HDP] TEARDOWN        — pipeline torn down, hooks removed.

No dependency on megatron.core.  Consumes only torch, torch.distributed,
zstandard, and deepspeed.utils.
"""

import concurrent.futures
import hashlib
import io
import json
import logging
import os
import tarfile
import threading
import time
from collections import OrderedDict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from deepspeed.utils import logger as ds_logger

_LOG_PREFIX = "[DS-HDP]"

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# Constants — mirror upstream utils_logits.py sentinels and tar format
# ────────────────────────────────────────────────────────────────────────────

# Sentinel value written into masked top-P entries; the KD loss treats any
# entry with this value as having effectively zero teacher probability.
_LOGPROB_SENTINEL: float = -1e3

# Sentinel index used for masked top-P positions.  pack_indices / unpack_indices
# round-trips through 17-bit unsigned representation, so -1 is not preserved;
# the value sentinel is the primary mask signal in KDLoss.
_INDEX_SENTINEL: int = -1

# Maximum vocab size supported by the 17-bit index scheme (2^17 = 131 072).
_MAX_VOCAB_SIZE: int = 2 ** 17

# Default CPU DRAM buffer limit — 1.5 TB expressed in bytes.
_DEFAULT_MAX_BUFFER_BYTES: int = int(1.5 * 1024 ** 4)

# zstd compression level — matches upstream LogitsSaverHooks._write_batched_tar.
_ZSTD_LEVEL: int = 3

# High-watermark fraction of max_buffer_bytes at which a diagnostic event fires.
_BUFFER_HIGH_WATERMARK: float = 0.75

# ────────────────────────────────────────────────────────────────────────────
# Enumerations and dataclasses
# ────────────────────────────────────────────────────────────────────────────


class TierRole(str, Enum):
    """GPU tier assignment for a rank in a DES-LOC job.

    Mirrors the TEACHER / STUDENT split implied by the two phases of the
    upstream 277c4f804 commit.  In Megatron the two phases run as separate
    jobs; DES-LOC collapses them into a single distributed job where H100
    ranks hold TEACHER and A6000 ranks hold STUDENT.
    """
    TEACHER = "teacher"   # Strong tier (H100) — generates and caches logits.
    STUDENT = "student"   # Weak tier (A6000) — trains on cached logits.


@dataclass
class TopKConfig:
    """Configuration for top-K (and optional top-P) logit selection.

    Mirrors the set of ``--logits-*`` CLI arguments introduced by the upstream
    ``_add_logits_distillation_args`` in ``arguments.py``.

    Attributes:
        k:              Number of top vocab positions to capture per token.
        p:              Optional nucleus (top-P) threshold in (0, 1].  When
                        set, only the smallest prefix of the top-K whose
                        cumulative probability mass reaches ``p`` is kept.
        min_k:          Minimum entries kept per token when top-P is active.
        save_dtype:     Dtype for on-disk / in-buffer log-probabilities.
                        One of ``'fp16'``, ``'bf16'``, ``'fp32'``.
        kd_loss_alpha:  Weight of KD loss in the blended student objective.
                        Total loss = alpha * kd_loss + (1 - alpha) * lm_loss.
        ignore_errors:  When True, KD loss computation errors are logged as
                        warnings and training falls back to LM-only loss.
    """
    k: int = 4096
    p: Optional[float] = None
    min_k: int = 1
    save_dtype: str = "fp16"
    kd_loss_alpha: float = 1.0
    ignore_errors: bool = False

    _DTYPE_MAP: Dict[str, torch.dtype] = field(default_factory=lambda: {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    })

    def __post_init__(self):
        if self.save_dtype not in ("fp16", "bf16", "fp32"):
            raise ValueError(
                f"save_dtype must be one of ('fp16', 'bf16', 'fp32'), "
                f"got '{self.save_dtype}'"
            )
        if self.p is not None and not (0.0 < self.p <= 1.0):
            raise ValueError(f"p must be in (0, 1] or None, got {self.p}")
        if self.min_k < 1:
            raise ValueError(f"min_k must be >= 1, got {self.min_k}")

    @property
    def torch_dtype(self) -> torch.dtype:
        """Return the torch dtype for on-wire log-probabilities."""
        return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
            self.save_dtype
        ]


# ────────────────────────────────────────────────────────────────────────────
# 17-bit index packing / unpacking
# ────────────────────────────────────────────────────────────────────────────

def _pack_indices(
    indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split 17-bit global vocab indices into uint16 low bits + bool 17th bit.

    Mirrors ``pack_indices`` in upstream ``utils_logits.py``.  Reduces
    per-token index storage from 8 bytes (int64) to 3 bytes (uint16 + bool)
    at the cost of one additional tensor.
    """
    low_bits = (indices & 0xFFFF).to(torch.uint16)
    bit_17 = (indices >> 16).to(torch.bool)
    return low_bits, bit_17


def _unpack_indices(
    low_bits: torch.Tensor,
    bit_17: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct int64 global indices from uint16 low bits + bool 17th bit.

    Mirrors ``unpack_indices`` in upstream ``utils_logits.py``.
    """
    return (bit_17.long() << 16) | low_bits.long()


# ────────────────────────────────────────────────────────────────────────────
# Nucleus (top-P) truncation — mirrors upstream _apply_topp_truncation
# ────────────────────────────────────────────────────────────────────────────

def _apply_topp_truncation(
    values: torch.Tensor,
    indices: torch.Tensor,
    p: float,
    min_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply top-P (nucleus) masking to sorted top-K log-probs.

    Mirrors ``LogitsSaverHooks._apply_topp_truncation`` from upstream.
    Keeps the smallest set of leading entries per token whose cumulative
    probability mass reaches ``p``, with a floor of ``min_k`` entries.

    The K dimension is truncated to the maximum kept count across all tokens
    in the microbatch; out-of-nucleus entries are masked with value/index
    sentinels compatible with the KD loss.

    Args:
        values:  Top-K log-probs sorted descending, shape ``(seq, batch, K)``.
        indices: Corresponding global vocab indices, shape ``(seq, batch, K)``.
        p:       Nucleus probability threshold in (0, 1].
        min_k:   Minimum entries kept per token.

    Returns:
        Tuple of ``(masked_values, masked_indices)`` with K dimension
        truncated to the maximum per-token nucleus size.
    """
    probs = values.float().exp()
    cumprobs = probs.cumsum(dim=-1)
    # Keep entry i iff cumulative mass *before* it is < p (always keep top-1).
    keep_mask = (cumprobs - probs) < p

    k = values.size(-1)
    min_keep = min(min_k, k)
    arange = torch.arange(k, device=values.device)
    keep_mask = keep_mask | (arange < min_keep)

    max_kept = int(keep_mask.sum(dim=-1).max().item())
    values = values[..., :max_kept].clone()
    indices = indices[..., :max_kept].clone()
    keep_mask = keep_mask[..., :max_kept]

    sentinel_val = torch.tensor(
        _LOGPROB_SENTINEL, dtype=values.dtype, device=values.device
    )
    sentinel_idx = torch.tensor(
        _INDEX_SENTINEL, dtype=indices.dtype, device=indices.device
    )
    values = torch.where(keep_mask, values, sentinel_val)
    indices = torch.where(keep_mask, indices, sentinel_idx)
    return values, indices


# ────────────────────────────────────────────────────────────────────────────
# CPU DRAM buffer pool
# ────────────────────────────────────────────────────────────────────────────

class _PerRankSlot:
    """A bounded FIFO queue of zstd-compressed iteration payloads for one DP rank."""

    def __init__(self, dp_rank: int, dp_size: int):
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        # Ordered: (iteration, compressed_bytes)
        self._queue: deque = deque()
        self._bytes_used: int = 0
        # Capacity is set externally by CPULogitsBuffer
        self.max_bytes: int = _DEFAULT_MAX_BUFFER_BYTES // max(dp_size, 1)

    def enqueue(
        self,
        iteration: int,
        data: bytes,
        *,
        block: bool = True,
    ) -> bool:
        """Enqueue a compressed payload.  Blocks (or returns False) if full."""
        size = len(data)
        with self._not_full:
            while self._bytes_used + size > self.max_bytes:
                if not block:
                    return False
                self._not_full.wait(timeout=0.05)
            self._queue.append((iteration, data))
            self._bytes_used += size
            self._not_empty.notify_all()
        return True

    def dequeue(
        self,
        *,
        timeout: Optional[float] = None,
    ) -> Optional[Tuple[int, bytes]]:
        """Dequeue the next payload.  Returns None on timeout."""
        deadline = (time.monotonic() + timeout) if timeout is not None else None
        with self._not_empty:
            while not self._queue:
                remaining = (deadline - time.monotonic()) if deadline else None
                if remaining is not None and remaining <= 0:
                    return None
                self._not_empty.wait(timeout=remaining or 0.05)
            iteration, data = self._queue.popleft()
            self._bytes_used -= len(data)
            self._not_full.notify_all()
        return iteration, data

    @property
    def bytes_used(self) -> int:
        with self._lock:
            return self._bytes_used

    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)


class CPULogitsBuffer:
    """Thread-safe CPU DRAM buffer pool relaying teacher logits to student ranks.

    The teacher's ``LogitsCaptureHook`` enqueues compressed payloads here;
    the student's ``CachedLogitsLoader`` dequeues them.  A separate
    ``_PerRankSlot`` per DP rank prevents head-of-line blocking when DP ranks
    advance at different rates.

    The total in-flight memory is bounded by ``max_buffer_bytes`` (default
    1.5 TB), split evenly across DP ranks.  When a slot is full the enqueue
    path blocks, providing back-pressure to the teacher loop.

    Args:
        dp_size:          Data-parallel world size.
        max_buffer_bytes: Maximum total CPU RAM used across all DP slots.
    """

    def __init__(
        self,
        dp_size: int = 1,
        max_buffer_bytes: int = _DEFAULT_MAX_BUFFER_BYTES,
    ):
        self.dp_size = dp_size
        self.max_buffer_bytes = max_buffer_bytes
        per_rank = max_buffer_bytes // max(dp_size, 1)
        self._slots: List[_PerRankSlot] = [
            _PerRankSlot(dp_rank=r, dp_size=dp_size) for r in range(dp_size)
        ]
        for slot in self._slots:
            slot.max_bytes = per_rank
        ds_logger.info(
            f"{_LOG_PREFIX} BUFFER_INIT dp_size={dp_size} "
            f"per_rank_cap={per_rank / 1024**3:.1f} GB "
            f"total_cap={max_buffer_bytes / 1024**3:.1f} GB"
        )

    def enqueue(
        self,
        iteration: int,
        data: bytes,
        dp_rank: int,
    ) -> None:
        """Enqueue a compressed payload for a specific DP rank.

        Blocks when the rank's slot is full, emitting a BUFFER_STALL event.
        """
        slot = self._slots[dp_rank]
        high_threshold = int(slot.max_bytes * _BUFFER_HIGH_WATERMARK)
        if slot.bytes_used > high_threshold:
            ds_logger.info(
                f"{_LOG_PREFIX} BUFFER_HIGH dp_rank={dp_rank} "
                f"used={slot.bytes_used / 1024**2:.0f} MB "
                f"cap={slot.max_bytes / 1024**2:.0f} MB "
                f"iter={iteration}"
            )
        stall_logged = False
        while True:
            enqueued = slot.enqueue(iteration, data, block=False)
            if enqueued:
                break
            if not stall_logged:
                ds_logger.info(
                    f"{_LOG_PREFIX} BUFFER_STALL dp_rank={dp_rank} "
                    f"waiting for student to drain iter={iteration}"
                )
                stall_logged = True
            time.sleep(0.01)

    def dequeue(
        self,
        dp_rank: int,
        timeout: Optional[float] = None,
    ) -> Optional[Tuple[int, bytes]]:
        """Dequeue the next payload for a specific DP rank."""
        return self._slots[dp_rank].dequeue(timeout=timeout)

    def total_bytes(self) -> int:
        """Return total CPU bytes currently held across all DP slots."""
        return sum(s.bytes_used for s in self._slots)

    def __len__(self) -> int:
        return sum(len(s) for s in self._slots)


# ────────────────────────────────────────────────────────────────────────────
# Teacher-side forward hook
# ────────────────────────────────────────────────────────────────────────────

class LogitsCaptureHook:
    """Forward hook for the teacher model's output layer.

    Mirrors ``LogitsSaverHooks`` from upstream ``logits_saver.py``, reinterpreted
    for the DES-LOC CPU buffer relay rather than a POSIX tar filesystem.

    The computation path matches upstream exactly:
      1. Cast logits to fp32.
      2. Local top-K on raw logits (avoids materialising full vocab softmax).
      3. Local log-sum-exp via ``torch.logsumexp`` (fused CUDA kernel).
      4. Global log-sum-exp via max+exp+sum+log across the TP group.
      5. Convert to log-probabilities: logprob = logit_val - global_lse.
      6. Gather all TP shards to TP rank 0; TP rank 0 selects global top-K.
      7. Optional top-P nucleus truncation.
      8. Cast to ``save_dtype``, pack 17-bit indices, zstd-compress.
      9. Enqueue compressed payload to ``CPULogitsBuffer`` for the current DP rank.

    Unlike upstream, steps 8–9 replace the async checkpoint-queue tar write.
    TP ranks other than 0 participate in collectives but do not enqueue.

    Args:
        buffer:     Shared ``CPULogitsBuffer`` instance.
        config:     ``TopKConfig`` with K, P, save_dtype, etc.
        tp_rank:    Tensor-parallel rank of this process.
        tp_size:    Tensor-parallel world size.
        tp_group:   Torch process group for the TP communicator.
        tp_src_rank: Global rank of TP rank 0 (used as gather dst).
        dp_rank:    Data-parallel rank (selects buffer slot).
        num_microbatches: Number of microbatches per gradient accumulation step.
                          When > 1 the hook accumulates across microbatches
                          and enqueues once per full step.
    """

    def __init__(
        self,
        buffer: CPULogitsBuffer,
        config: TopKConfig,
        tp_rank: int,
        tp_size: int,
        tp_group,
        tp_src_rank: int,
        dp_rank: int,
        num_microbatches: int = 1,
    ):
        self._buffer = buffer
        self._config = config
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.tp_group = tp_group
        self._tp_src_rank = tp_src_rank
        self.dp_rank = dp_rank
        self._num_microbatches = num_microbatches

        self._compressor = __import__("zstandard").ZstdCompressor(level=_ZSTD_LEVEL)

        # Accumulated results across microbatches: List of (values, idx_low, bit_17)
        self._accumulated: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._hook_handles: List[Any] = []
        self._iteration: int = 0

    def attach(self, model: torch.nn.Module) -> None:
        """Attach to the output layer of *model*."""
        output_layer = self._find_output_layer(model)
        handle = output_layer.register_forward_hook(self._forward_hook)
        self._hook_handles.append(handle)
        ds_logger.info(
            f"{_LOG_PREFIX} SETUP_TEACHER "
            f"tp_rank={self.tp_rank} dp_rank={self.dp_rank} "
            f"k={self._config.k} p={self._config.p} "
            f"dtype={self._config.save_dtype} "
            f"buffer_cap={self._buffer.max_buffer_bytes / 1024**3:.1f} GB"
        )

    @staticmethod
    def _find_output_layer(model: torch.nn.Module) -> torch.nn.Module:
        """Locate the output projection layer of *model*.

        Checks the common attribute names used by GPT-style models:
        ``output_layer``, ``lm_head``, ``embed_out``.
        """
        for attr in ("output_layer", "lm_head", "embed_out"):
            if hasattr(model, attr):
                layer = getattr(model, attr)
                if isinstance(layer, torch.nn.Module):
                    return layer
        # Fall back: last Linear in the top-level module list
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Linear):
                return module
        raise AttributeError(
            "Could not locate the output layer in the teacher model. "
            "Set the 'output_layer' attribute on your model before calling attach()."
        )

    def _forward_hook(
        self,
        module: torch.nn.Module,
        input: Any,
        output: Any,
    ) -> None:
        """Capture one microbatch's top-K log-probs."""
        if not module.training:
            return

        logits = output[0] if isinstance(output, tuple) else output
        with torch.no_grad():
            result = self._process_microbatch(logits)
        if result is not None:
            self._accumulated.append(result)

        if len(self._accumulated) == self._num_microbatches:
            self._flush_accumulated()
            self._accumulated.clear()

    def _process_microbatch(
        self,
        logits: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Process one microbatch logits → top-K log-probs packed into 17-bit indices.

        Mirrors ``LogitsSaverHooks._process_single_microbatch`` from upstream.
        Returns ``None`` on TP ranks other than 0 (they participate in
        collectives but do not produce output).
        """
        local_vocab = logits.shape[-1]
        global_vocab = local_vocab * self.tp_size
        assert global_vocab <= _MAX_VOCAB_SIZE, (
            f"Global vocab size {global_vocab} exceeds {_MAX_VOCAB_SIZE} (17 bits)"
        )

        effective_k = min(self._config.k, global_vocab)
        local_k = min(effective_k, local_vocab)

        logits = logits.float()
        local_vals, local_idx = torch.topk(logits, local_k, dim=-1)
        local_lse = torch.logsumexp(logits, dim=-1, keepdim=True)

        if self.tp_size > 1:
            max_lse = local_lse.clone()
            dist.all_reduce(max_lse, op=dist.ReduceOp.MAX, group=self.tp_group)
            sum_exp = torch.exp(local_lse - max_lse)
            dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=self.tp_group)
            global_lse = max_lse + torch.log(sum_exp)
        else:
            global_lse = local_lse

        local_logprob = local_vals - global_lse

        if self.tp_size > 1:
            result = self._global_topk(
                local_vals, local_logprob, local_idx, effective_k, local_vocab
            )
            if result is None:
                return None
            global_values, global_indices = result
        else:
            global_values, global_indices = local_logprob, local_idx

        if self._config.p is not None:
            global_values, global_indices = _apply_topp_truncation(
                global_values, global_indices,
                p=self._config.p,
                min_k=self._config.min_k,
            )

        global_values = global_values.to(self._config.torch_dtype)
        idx_low, bit_17 = _pack_indices(global_indices)
        return global_values, idx_low, bit_17

    def _global_topk(
        self,
        local_logit_vals: torch.Tensor,
        local_logprob_vals: torch.Tensor,
        local_idx: torch.Tensor,
        k: int,
        local_vocab: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Gather local top-K across TP ranks; TP rank 0 selects global top-K.

        Mirrors ``LogitsSaverHooks._compute_global_topk`` from upstream.
        Packs logits, log-probs, and global indices into a single float32
        tensor to avoid three separate gather calls.
        """
        vocab_offset = self.tp_rank * local_vocab
        global_idx = local_idx + vocab_offset
        combined = torch.stack(
            [local_logit_vals, local_logprob_vals.float(), global_idx.float()],
            dim=-1,
        )
        if self.tp_rank == 0:
            gather_list = [torch.empty_like(combined) for _ in range(self.tp_size)]
        else:
            gather_list = None
        dist.gather(combined, gather_list, dst=self._tp_src_rank, group=self.tp_group)
        if self.tp_rank != 0:
            return None

        gathered = torch.cat(gather_list, dim=-2)
        gathered_logits = gathered[..., 0]
        gathered_logprobs = gathered[..., 1].to(local_logprob_vals.dtype)
        gathered_indices = gathered[..., 2].to(local_idx.dtype)

        _, topk_pos = torch.topk(gathered_logits, k, dim=-1)
        top_values = torch.gather(gathered_logprobs, -1, topk_pos)
        top_indices = torch.gather(gathered_indices, -1, topk_pos)
        return top_values, top_indices

    def _flush_accumulated(self) -> None:
        """Serialize accumulated microbatch results and enqueue to the buffer.

        Only TP rank 0 has non-None results; other ranks skip the enqueue.
        The iteration counter is incremented on every flush regardless of TP rank
        so all ranks stay in sync with the global step count.
        """
        self._iteration += 1
        iteration = self._iteration

        if self.tp_rank != 0:
            return

        all_values = [r[0].cpu() for r in self._accumulated]
        all_idx_low = [r[1].cpu() for r in self._accumulated]
        all_bit_17 = [r[2].cpu() for r in self._accumulated]

        buf = io.BytesIO()
        torch.save(
            {
                "values": all_values,
                "indices_low": all_idx_low,
                "bit_17": all_bit_17,
            },
            buf,
        )
        raw = buf.getvalue()
        compressed = self._compressor.compress(raw)

        self._buffer.enqueue(iteration, compressed, dp_rank=self.dp_rank)
        ds_logger.info(
            f"{_LOG_PREFIX} ENQUEUE "
            f"dp_rank={self.dp_rank} iter={iteration} "
            f"size_kb={len(compressed)/1024:.1f}"
        )

    def set_iteration(self, iteration: int) -> None:
        """Synchronise the hook's internal iteration counter with the engine."""
        self._iteration = iteration

    def remove(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()


# ────────────────────────────────────────────────────────────────────────────
# Student-side logits capture hook
# ────────────────────────────────────────────────────────────────────────────

# Module-level reference — KDLoss reads this after each student forward pass.
_ACTIVE_STUDENT_CAPTURE: Optional["StudentLogitsCapture"] = None


def get_active_student_capture() -> Optional["StudentLogitsCapture"]:
    """Return the active ``StudentLogitsCapture`` instance, or *None*."""
    return _ACTIVE_STUDENT_CAPTURE


class StudentLogitsCapture:
    """Forward hook that retains the latest differentiable student logits.

    Mirrors ``StudentLogitsCapture`` from upstream ``cached_logits_loss.py``,
    reinterpreted without any ``megatron.core`` dependency.

    A reference to this instance is stored in ``_ACTIVE_STUDENT_CAPTURE`` so
    ``KDLoss.__call__`` can retrieve logits without requiring a model attribute.
    """

    def __init__(self):
        self._logits: Optional[torch.Tensor] = None
        self._hook_handles: List[Any] = []

    def attach(self, model: torch.nn.Module) -> None:
        """Register the capture hook on the output layer of *model*."""
        output_layer = LogitsCaptureHook._find_output_layer(model)
        handle = output_layer.register_forward_hook(self._capture)
        self._hook_handles.append(handle)

        global _ACTIVE_STUDENT_CAPTURE
        _ACTIVE_STUDENT_CAPTURE = self

        ds_logger.info(
            f"{_LOG_PREFIX} SETUP_STUDENT StudentLogitsCapture attached"
        )

    def _capture(
        self,
        module: torch.nn.Module,
        input: Any,
        output: Any,
    ) -> None:
        if not module.training:
            return
        # Overwrite on each forward call; the main head runs last in MTP models.
        self._logits = output[0] if isinstance(output, tuple) else output

    def pop(self) -> torch.Tensor:
        """Return captured logits and clear internal reference."""
        if self._logits is None:
            raise RuntimeError(
                f"{_LOG_PREFIX} No student logits captured. "
                "Ensure StudentLogitsCapture.attach() was called before training."
            )
        logits = self._logits
        self._logits = None
        return logits

    def remove(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._logits = None


# ────────────────────────────────────────────────────────────────────────────
# Student-side cached logits loader
# ────────────────────────────────────────────────────────────────────────────

def _decompress_payload(
    data: bytes,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Decompress and unpack one zstd-compressed iteration payload.

    Mirrors ``decode_logprobs_payload`` from upstream ``utils_logits.py``.
    Returns ``(values_list, indices_list)`` where each list has one tensor
    per microbatch.
    """
    import zstandard
    raw = zstandard.ZstdDecompressor().decompress(data)
    tensors = torch.load(io.BytesIO(raw), weights_only=True)
    indices_list = [
        _unpack_indices(low, bit17)
        for low, bit17 in zip(tensors["indices_low"], tensors["bit_17"])
    ]
    return tensors["values"], indices_list


class CachedLogitsLoader:
    """Iterator that dequeues teacher logits from a ``CPULogitsBuffer``.

    Each call to ``__next__`` blocks until the teacher has enqueued the next
    iteration's payload and returns ``(values_list, indices_list)`` — exactly
    the format produced by upstream ``TeacherTarDataset.__iter__``.

    A fallback disk-reading path is available for runs where the teacher and
    student execute in separate jobs: pass ``logprobs_dir`` to read pre-computed
    tar shards from disk using the same streaming layout as upstream.

    Args:
        buffer:       Shared ``CPULogitsBuffer``.  If *None*, reads from disk.
        dp_rank:      Data-parallel rank — selects the buffer slot to read from.
        logprobs_dir: Optional filesystem directory with upstream-format tar shards.
                      Used as fallback when ``buffer`` is *None*.
        cp_rank:      Context-parallel rank (used for disk-shard filename matching).
        dp_size:      Data-parallel world size (used for DP resharding on disk path).
        decode_threads: Worker threads for parallel zstd decompression (disk path).
        dequeue_timeout: Seconds to wait for the next buffer entry before raising.
    """

    def __init__(
        self,
        buffer: Optional[CPULogitsBuffer],
        dp_rank: int,
        logprobs_dir: Optional[str] = None,
        cp_rank: int = 0,
        dp_size: int = 1,
        decode_threads: int = 4,
        dequeue_timeout: float = 300.0,
    ):
        self._buffer = buffer
        self.dp_rank = dp_rank
        self.logprobs_dir = logprobs_dir
        self.cp_rank = cp_rank
        self.dp_size = dp_size
        self._decode_threads = decode_threads
        self._dequeue_timeout = dequeue_timeout
        self._disk_iter: Optional[Iterator] = None

    def __iter__(self) -> Iterator[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        if self._buffer is not None:
            yield from self._iter_from_buffer()
        elif self.logprobs_dir is not None:
            yield from self._iter_from_disk()
        else:
            raise RuntimeError(
                f"{_LOG_PREFIX} CachedLogitsLoader requires either a CPULogitsBuffer "
                "or a logprobs_dir to read from."
            )

    def _iter_from_buffer(self) -> Iterator[Tuple[List, List]]:
        """Yield payloads dequeued from the live CPU buffer."""
        assert self._buffer is not None
        while True:
            result = self._buffer.dequeue(
                dp_rank=self.dp_rank,
                timeout=self._dequeue_timeout,
            )
            if result is None:
                # Timeout: teacher may be done or stalled — caller decides.
                return
            iteration, data = result
            ds_logger.info(
                f"{_LOG_PREFIX} DEQUEUE dp_rank={self.dp_rank} iter={iteration}"
            )
            values_list, indices_list = _decompress_payload(data)
            yield values_list, indices_list

    def _iter_from_disk(self) -> Iterator[Tuple[List, List]]:
        """Yield payloads from pre-computed tar shards on disk.

        Implements the same streaming logic as upstream ``TeacherTarDataset``
        but without the IterableDataset overhead — suitable for offline runs
        where the teacher produced tars in a prior job.
        """
        import glob as _glob
        import re

        _TAR_RE = re.compile(r"^cp(\d+)_dp(\d+)__(\d+)\.tar$")
        assert self.logprobs_dir is not None
        prefix = f"cp{self.cp_rank}_dp{self.dp_rank}__"
        pattern = os.path.join(self.logprobs_dir, f"{prefix}*.tar")
        shard_paths = sorted(
            _glob.glob(pattern),
            key=lambda p: int(_TAR_RE.match(os.path.basename(p)).group(3)),  # type: ignore[union-attr]
        )
        if not shard_paths:
            raise FileNotFoundError(
                f"No tar shards matching '{pattern}' in '{self.logprobs_dir}'"
            )

        import zstandard
        _decompressor = zstandard.ZstdDecompressor()

        for shard_path in shard_paths:
            with open(shard_path, "rb") as f:
                with tarfile.open(fileobj=f, mode="r|*") as tar:
                    for member in tar:
                        if not member.isreg():
                            continue
                        if not member.name.endswith(".pt.zst"):
                            continue
                        extracted = tar.extractfile(member)
                        if extracted is None:
                            continue
                        raw = _decompressor.decompress(extracted.read())
                        tensors = torch.load(io.BytesIO(raw), weights_only=True)
                        indices_list = [
                            _unpack_indices(low, bit17)
                            for low, bit17 in zip(
                                tensors["indices_low"], tensors["bit_17"]
                            )
                        ]
                        yield tensors["values"], indices_list


# ────────────────────────────────────────────────────────────────────────────
# KD loss computation
# ────────────────────────────────────────────────────────────────────────────

def _sparse_kl_divergence(
    student_logits: torch.Tensor,
    teacher_values: torch.Tensor,
    teacher_indices: torch.Tensor,
    loss_mask: torch.Tensor,
    tp_group,
    tp_rank: int,
    tp_size: int,
    local_vocab: int,
) -> torch.Tensor:
    """Compute forward KL divergence between student and sparse teacher distribution.

    Mirrors ``topk_kl_div`` from upstream ``cached_logits_loss.py``.

    The student log-softmax is computed in a TP-aware fashion:
      1. Local log-sum-exp over the student's vocab shard.
      2. All-reduce (max+exp+sum+log) to obtain the global log-sum-exp.
      3. Student log-probs = student_logits - global_log_sum_exp.

    Then for each token, only the teacher's top-K vocab positions that fall
    in this rank's local shard are used in the KL sum.  The final reduction
    is a sum over all ranks (the KL divergence over the global vocab is the
    sum of per-rank contributions over its shard).

    The teacher sentinel value ``_LOGPROB_SENTINEL`` is used to mask
    out-of-nucleus entries so they contribute zero to the KL sum.

    Args:
        student_logits:   Shape ``(seq, batch, local_vocab)``.
        teacher_values:   Sparse top-K log-probs, shape ``(seq, batch, K)``.
        teacher_indices:  Global vocab indices, shape ``(seq, batch, K)`` (int64).
        loss_mask:        Shape ``(batch, seq)`` — 1.0 for valid tokens.
        tp_group:         Tensor-parallel process group (or None for tp_size=1).
        tp_rank:          Tensor-parallel rank.
        tp_size:          Tensor-parallel world size.
        local_vocab:      Size of the local vocab shard.

    Returns:
        Scalar KL divergence loss (mean over valid tokens).
    """
    seq_len, batch, _ = student_logits.shape
    student_fp32 = student_logits.float()

    # ── Step 1: TP-aware student log-softmax ──────────────────────────────
    local_lse = torch.logsumexp(student_fp32, dim=-1, keepdim=True)  # (s,b,1)
    if tp_size > 1:
        max_lse = local_lse.clone()
        dist.all_reduce(max_lse, op=dist.ReduceOp.MAX, group=tp_group)
        sum_exp = torch.exp(local_lse - max_lse)
        dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=tp_group)
        global_lse = max_lse + torch.log(sum_exp)
    else:
        global_lse = local_lse

    student_logprob = student_fp32 - global_lse  # (s, b, local_vocab)

    # ── Step 2: Filter teacher indices to this rank's vocab shard ─────────
    vocab_start = tp_rank * local_vocab
    vocab_end = vocab_start + local_vocab
    teacher_values_fp32 = teacher_values.float()

    # Valid mask: index in this shard and not a sentinel value
    in_shard = (teacher_indices >= vocab_start) & (teacher_indices < vocab_end)
    not_sentinel = teacher_values_fp32 > (_LOGPROB_SENTINEL + 1.0)
    valid = in_shard & not_sentinel  # (s, b, K)

    # Local indices within this shard
    local_teacher_idx = (teacher_indices - vocab_start).clamp(min=0)  # (s, b, K)

    # Student log-probs at teacher's top-K positions (for positions in shard)
    student_at_teacher = torch.gather(
        student_logprob, dim=-1,
        index=local_teacher_idx.clamp(max=local_vocab - 1),
    )  # (s, b, K)

    # Teacher probability: exp(teacher_logprob)
    teacher_prob = teacher_values_fp32.exp()  # (s, b, K)

    # KL divergence contribution: p_t * (log p_t - log p_s)
    # = p_t * teacher_logprob - p_t * student_logprob_at_teacher_pos
    kl_per_pos = teacher_prob * (teacher_values_fp32 - student_at_teacher)
    kl_per_pos = kl_per_pos * valid.float()  # zero out non-shard / sentinel positions

    # ── Step 3: Reduce over K → per-token KL ──────────────────────────────
    kl_per_token = kl_per_pos.sum(dim=-1)  # (s, b)
    if tp_size > 1:
        dist.all_reduce(kl_per_token, op=dist.ReduceOp.SUM, group=tp_group)

    # ── Step 4: Apply loss mask and average ───────────────────────────────
    # loss_mask shape: (batch, seq) — transpose to (seq, batch)
    mask = loss_mask.transpose(0, 1).float().to(kl_per_token.device)  # (s, b)
    kl_masked = (kl_per_token * mask).sum()
    num_valid = mask.sum().clamp(min=1.0)
    return kl_masked / num_valid


class KDLoss:
    """KD-blended loss callable for the student training loop.

    Computes ``alpha * kd_loss + (1 - alpha) * lm_loss`` where the KD term
    is a sparse forward KL divergence against the teacher's cached top-K
    log-probabilities.

    Usage::

        kd_loss_fn = KDLoss(loader, config, tp_rank, tp_size, tp_group, local_vocab)
        loss, report = kd_loss_fn(loss_mask, lm_loss)

    The ``StudentLogitsCapture`` registered via
    ``HeteroDistillationPipeline.setup()`` provides the differentiable
    student logits.

    Args:
        loader:     ``CachedLogitsLoader`` instance that yields per-step
                    ``(values_list, indices_list)`` tuples.
        config:     ``TopKConfig`` carrying ``kd_loss_alpha`` and ``ignore_errors``.
        tp_rank:    Tensor-parallel rank.
        tp_size:    Tensor-parallel world size.
        tp_group:   Tensor-parallel process group.
        local_vocab: Size of the local vocab shard (vocab_size / tp_size).
    """

    def __init__(
        self,
        loader: CachedLogitsLoader,
        config: TopKConfig,
        tp_rank: int,
        tp_size: int,
        tp_group,
        local_vocab: int,
    ):
        self._loader_iter = iter(loader)
        self._config = config
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.tp_group = tp_group
        self.local_vocab = local_vocab
        self._step = 0

    def __call__(
        self,
        loss_mask: torch.Tensor,
        lm_loss: torch.Tensor,
        microbatch_index: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute blended KD + LM loss for one microbatch.

        Args:
            loss_mask:       Boolean/float mask, shape ``(batch, seq)``.
            lm_loss:         Pre-computed language model loss (scalar or tensor).
            microbatch_index: Index of the current microbatch within the step.

        Returns:
            Tuple of ``(blended_loss, report_dict)`` where ``report_dict``
            carries ``'kd_loss'``, ``'lm_loss'``, and ``'alpha'`` for logging.
        """
        alpha = self._config.kd_loss_alpha

        # Fetch next teacher payload on microbatch 0 of each step
        if microbatch_index == 0:
            try:
                self._teacher_values_list, self._teacher_indices_list = next(
                    self._loader_iter
                )
                self._step += 1
            except StopIteration:
                if self._config.ignore_errors:
                    ds_logger.warning(
                        f"{_LOG_PREFIX} FALLBACK_LM "
                        f"step={self._step} teacher logits exhausted; "
                        "falling back to LM-only loss"
                    )
                    return lm_loss, {"kd_loss": 0.0, "lm_loss": float(lm_loss), "alpha": alpha}
                raise RuntimeError(
                    f"{_LOG_PREFIX} Teacher logits exhausted at step {self._step}. "
                    "Ensure teacher ran long enough or pass ignore_errors=True."
                )

        capture = get_active_student_capture()
        if capture is None:
            if self._config.ignore_errors:
                ds_logger.warning(
                    f"{_LOG_PREFIX} FALLBACK_LM no StudentLogitsCapture active"
                )
                return lm_loss, {"kd_loss": 0.0, "lm_loss": float(lm_loss), "alpha": alpha}
            raise RuntimeError(
                f"{_LOG_PREFIX} No StudentLogitsCapture active. "
                "Call HeteroDistillationPipeline.setup() before training."
            )

        try:
            student_logits = capture.pop()
            teacher_values = self._teacher_values_list[microbatch_index].to(
                student_logits.device
            )
            teacher_indices = self._teacher_indices_list[microbatch_index].to(
                student_logits.device
            )

            kd = _sparse_kl_divergence(
                student_logits=student_logits,
                teacher_values=teacher_values,
                teacher_indices=teacher_indices,
                loss_mask=loss_mask,
                tp_group=self.tp_group,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                local_vocab=self.local_vocab,
            )

            blended = alpha * kd + (1.0 - alpha) * lm_loss
            report = {
                "kd_loss": float(kd),
                "lm_loss": float(lm_loss),
                "alpha": alpha,
            }
            ds_logger.info(
                f"{_LOG_PREFIX} KD_LOSS "
                f"step={self._step} mb={microbatch_index} "
                f"kd={float(kd):.4f} lm={float(lm_loss):.4f} alpha={alpha}"
            )
            return blended, report

        except Exception as exc:
            if self._config.ignore_errors:
                ds_logger.warning(
                    f"{_LOG_PREFIX} FALLBACK_LM "
                    f"step={self._step} KD computation failed: {exc}"
                )
                return lm_loss, {"kd_loss": 0.0, "lm_loss": float(lm_loss), "alpha": alpha}
            raise


# ────────────────────────────────────────────────────────────────────────────
# Frozen expert bias guard
# ────────────────────────────────────────────────────────────────────────────

class FrozenExpertBiasGuard:
    """Context manager that freezes MoE router expert-bias updates.

    Mirrors the ``frozen_expert_bias`` attribute added to ``TopKRouter`` in
    upstream ``router.py`` and the guard in ``finalize_model_grads.py``.

    When a student model passes through a mixed eval/train step (e.g. when the
    teacher's MoE layers share weights with the student in online distillation),
    the expert-bias update must be skipped on frozen modules to avoid corrupting
    the teacher's routing state.

    This guard finds all submodules that have an ``expert_bias`` attribute and
    sets ``frozen_expert_bias = True`` on them for the duration of the ``with``
    block, restoring the original value on exit.

    Usage::

        with FrozenExpertBiasGuard(model):
            # forward pass where expert_bias update should be suppressed
            loss = model(inputs)
    """

    def __init__(self, model: torch.nn.Module):
        self._model = model
        self._saved: List[Tuple[torch.nn.Module, bool]] = []

    def __enter__(self) -> "FrozenExpertBiasGuard":
        for module in self._model.modules():
            if hasattr(module, "expert_bias"):
                original = getattr(module, "frozen_expert_bias", False)
                self._saved.append((module, original))
                module.frozen_expert_bias = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for module, original in self._saved:
            module.frozen_expert_bias = original
        self._saved.clear()

    @staticmethod
    def should_update_expert_bias(module: torch.nn.Module) -> bool:
        """Return True if the module's expert bias should be updated this step.

        Mirrors the guard condition in upstream ``_update_router_expert_bias``:
          ``hasattr(module, 'expert_bias') and module.training
           and not getattr(module, 'frozen_expert_bias', False)``
        """
        return (
            hasattr(module, "expert_bias")
            and getattr(module, "training", False)
            and not getattr(module, "frozen_expert_bias", False)
        )


# ────────────────────────────────────────────────────────────────────────────
# Checkpoint iteration override
# ────────────────────────────────────────────────────────────────────────────

class CheckpointIterationOverride:
    """Rewind the DeepSpeed engine's iteration counter to a target value.

    Mirrors the ``--override-ckpt-iteration`` mechanism introduced in upstream
    ``checkpointing.py``, reinterpreted for DeepSpeed's checkpoint state dict
    structure.

    When loading a teacher checkpoint to initialise the student, the data-loader
    offset must be reset so the student replays exactly the samples the teacher
    saw when producing cached logits.  Passing the teacher's starting iteration
    here recomputes ``consumed_train_samples`` accordingly.

    Usage::

        override = CheckpointIterationOverride(target_iteration=1000)
        state_dict = torch.load("ckpt.pt")
        override.apply(state_dict, global_batch_size=512)
        engine.load_state_dict(state_dict)

    Args:
        target_iteration: The iteration to reset the checkpoint to.
    """

    def __init__(self, target_iteration: int):
        if target_iteration < 0:
            raise ValueError(
                f"target_iteration must be >= 0, got {target_iteration}"
            )
        self.target_iteration = target_iteration

    def apply(
        self,
        state_dict: Dict[str, Any],
        global_batch_size: int,
    ) -> Dict[str, Any]:
        """Patch *state_dict* in-place and return it.

        Sets ``iteration``, ``global_steps``, and ``consumed_train_samples``
        to values consistent with ``target_iteration``.  Resets skipped sample
        count to 0 to avoid misaligned data-loader seeks.

        Args:
            state_dict:        DeepSpeed checkpoint state dict.
            global_batch_size: Current run's global batch size.  Must match
                               the value used during teacher logit saving or
                               data ordering will be misaligned.

        Returns:
            The patched state dict.
        """
        target = self.target_iteration
        consumed = target * global_batch_size

        if "iteration" in state_dict:
            orig = state_dict["iteration"]
            if orig != target:
                ds_logger.info(
                    f"{_LOG_PREFIX} CKPT_OVERRIDE "
                    f"iteration {orig} → {target} "
                    f"consumed_train_samples → {consumed}"
                )
        state_dict["iteration"] = target
        state_dict["global_steps"] = target
        state_dict["consumed_train_samples"] = consumed
        state_dict.pop("skipped_train_samples", None)
        return state_dict


# ────────────────────────────────────────────────────────────────────────────
# Top-level pipeline orchestrator
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class DistillationSetupResult:
    """Container for objects returned by ``HeteroDistillationPipeline.setup()``.

    Attributes:
        tier:           The current rank's role (TEACHER or STUDENT).
        capture_hook:   ``LogitsCaptureHook`` (TEACHER tier only, else *None*).
        student_capture: ``StudentLogitsCapture`` (STUDENT tier only, else *None*).
        kd_loss:        ``KDLoss`` callable (STUDENT tier only, else *None*).
        buffer:         Shared ``CPULogitsBuffer`` (both tiers).
    """
    tier: TierRole
    capture_hook: Optional[LogitsCaptureHook]
    student_capture: Optional[StudentLogitsCapture]
    kd_loss: Optional[KDLoss]
    buffer: CPULogitsBuffer


class HeteroDistillationPipeline:
    """Top-level orchestrator for heterogeneous offline logit distillation.

    Mirrors the end-to-end workflow of Megatron 277c4f804 as a single
    DeepSpeed-native object, reinterpreted for the DES-LOC CPU buffer relay.

    Instantiate once per process, call ``setup()`` once after model and
    process groups are initialised, and call ``teardown()`` at the end of
    training or on error.

    Example::

        from deepspeed.compression.hetero_distillation_pipeline import (
            HeteroDistillationPipeline, TopKConfig, TierRole,
        )

        pipeline = HeteroDistillationPipeline(
            tier=TierRole.STUDENT,
            config=TopKConfig(k=4096, kd_loss_alpha=0.5),
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            dp_rank=dp_rank,
            dp_size=dp_size,
            local_vocab=student_model.config.vocab_size // tp_size,
        )
        result = pipeline.setup(teacher_model=None, student_model=student_model)

        # Inside training loop:
        for batch in dataloader:
            lm_loss = student_model(batch)
            blended_loss, report = result.kd_loss(
                loss_mask=batch["attention_mask"],
                lm_loss=lm_loss,
                microbatch_index=mb_idx,
            )
            blended_loss.backward()

    Args:
        tier:         This rank's role (TEACHER or STUDENT).
        config:       ``TopKConfig`` for K, P, dtype, alpha, etc.
        tp_rank:      Tensor-parallel rank.
        tp_size:      Tensor-parallel world size.
        tp_group:     Tensor-parallel process group (may be *None* for tp_size=1).
        tp_src_rank:  Global rank of TP rank 0 used as gather destination.
        dp_rank:      Data-parallel rank.
        dp_size:      Data-parallel world size.
        local_vocab:  Size of the local vocab shard (vocab_size / tp_size).
        num_microbatches: Microbatches per gradient accumulation step.
        max_buffer_bytes: CPU DRAM limit for the logits buffer.
        logprobs_dir: Optional path to pre-computed tar shards (disk fallback).
        decode_threads: Decode-worker threads for the disk fallback path.
        shared_buffer: Pre-constructed ``CPULogitsBuffer`` to share across
                       teacher and student processes in the same address space.
                       When *None*, a new buffer is constructed.
    """

    def __init__(
        self,
        tier: TierRole,
        config: TopKConfig,
        tp_rank: int = 0,
        tp_size: int = 1,
        tp_group=None,
        tp_src_rank: int = 0,
        dp_rank: int = 0,
        dp_size: int = 1,
        local_vocab: int = 32000,
        num_microbatches: int = 1,
        max_buffer_bytes: int = _DEFAULT_MAX_BUFFER_BYTES,
        logprobs_dir: Optional[str] = None,
        decode_threads: int = 4,
        shared_buffer: Optional[CPULogitsBuffer] = None,
    ):
        self.tier = tier
        self.config = config
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.tp_group = tp_group
        self.tp_src_rank = tp_src_rank
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.local_vocab = local_vocab
        self.num_microbatches = num_microbatches
        self.logprobs_dir = logprobs_dir
        self.decode_threads = decode_threads

        if shared_buffer is not None:
            self._buffer = shared_buffer
        else:
            self._buffer = CPULogitsBuffer(
                dp_size=dp_size,
                max_buffer_bytes=max_buffer_bytes,
            )

        self._capture_hook: Optional[LogitsCaptureHook] = None
        self._student_capture: Optional[StudentLogitsCapture] = None
        self._kd_loss: Optional[KDLoss] = None

    def setup(
        self,
        teacher_model: Optional[torch.nn.Module] = None,
        student_model: Optional[torch.nn.Module] = None,
    ) -> DistillationSetupResult:
        """Attach hooks and construct the KD loss callable.

        Exactly one of ``teacher_model`` and ``student_model`` is expected to
        be non-None, matching the current rank's tier.  Passing both is
        supported for same-process teacher+student setups (rare but valid in
        unit tests).

        Args:
            teacher_model: Model running on the strong (H100) tier.
            student_model: Model running on the weak (A6000) tier.

        Returns:
            ``DistillationSetupResult`` with tier-appropriate fields populated.
        """
        if self.tier == TierRole.TEACHER and teacher_model is not None:
            self._capture_hook = LogitsCaptureHook(
                buffer=self._buffer,
                config=self.config,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                tp_group=self.tp_group,
                tp_src_rank=self.tp_src_rank,
                dp_rank=self.dp_rank,
                num_microbatches=self.num_microbatches,
            )
            self._capture_hook.attach(teacher_model)

        if self.tier == TierRole.STUDENT and student_model is not None:
            self._student_capture = StudentLogitsCapture()
            self._student_capture.attach(student_model)

            loader = CachedLogitsLoader(
                buffer=self._buffer,
                dp_rank=self.dp_rank,
                logprobs_dir=self.logprobs_dir,
                cp_rank=0,
                dp_size=self.dp_size,
                decode_threads=self.decode_threads,
            )
            self._kd_loss = KDLoss(
                loader=loader,
                config=self.config,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                tp_group=self.tp_group,
                local_vocab=self.local_vocab,
            )
            ds_logger.info(
                f"{_LOG_PREFIX} SETUP_STUDENT "
                f"dp_rank={self.dp_rank} "
                f"kd_alpha={self.config.kd_loss_alpha} "
                f"local_vocab={self.local_vocab}"
            )

        return DistillationSetupResult(
            tier=self.tier,
            capture_hook=self._capture_hook,
            student_capture=self._student_capture,
            kd_loss=self._kd_loss,
            buffer=self._buffer,
        )

    def teardown(self) -> None:
        """Remove hooks and log teardown event."""
        if self._capture_hook is not None:
            self._capture_hook.remove()
            self._capture_hook = None
        if self._student_capture is not None:
            self._student_capture.remove()
            self._student_capture = None
        ds_logger.info(
            f"{_LOG_PREFIX} TEARDOWN "
            f"tier={self.tier.value} dp_rank={self.dp_rank} "
            f"buffer_remaining={len(self._buffer)}"
        )

    @property
    def buffer(self) -> CPULogitsBuffer:
        """Return the shared CPU logits buffer."""
        return self._buffer

    def sync_iteration(self, iteration: int) -> None:
        """Synchronise the teacher hook's iteration counter with the engine.

        Call at the start of each training step so that enqueued payloads
        carry the correct global step number.
        """
        if self._capture_hook is not None:
            self._capture_hook.set_iteration(iteration - 1)

    def frozen_expert_bias_guard(self) -> FrozenExpertBiasGuard:
        """Return a context manager that freezes MoE expert-bias updates.

        Use around forward passes where the student's MoE router expert-bias
        should not be updated — mirrors the ``frozen_expert_bias`` flag
        introduced in upstream ``router.py`` and ``finalize_model_grads.py``.
        """
        if self._student_capture is None:
            raise RuntimeError(
                "frozen_expert_bias_guard() requires the student model to be set up first."
            )
        # Locate the student model from the capture hook's module reference
        # (the guard is attached to whichever model has `expert_bias` submodules).
        raise RuntimeError(
            "Pass the student model directly to FrozenExpertBiasGuard(model)."
        )
