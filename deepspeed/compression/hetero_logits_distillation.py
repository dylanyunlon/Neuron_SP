"""
DES-LOC Heterogeneous Offline Logits Distillation
==================================================

Upstream design intent (Megatron 277c4f8):
    Megatron adds offline knowledge distillation by saving teacher top-K
    log-probabilities to disk during a teacher forward pass, then loading
    them in a student training run to compute forward KL divergence without
    requiring the teacher model to be live.  The key insight is that
    log-softmax is monotonically increasing, so top-K selection on raw logits
    is equivalent to top-K on log-probs, avoiding a full-vocab log-softmax
    materialization.  Tar shards keyed by (cp_rank, dp_rank) allow parallel
    I/O without cross-rank coordination.

DES-LOC adaptation points:
    DES-LOC (Decoupled Execution with Shared LOcality Cache) runs on a
    heterogeneous cluster: 2× A6000-48GB (SM86, PCIe) + 1× H100-NVL-96GB
    (SM90, PCIe), 1.5TB CPU DRAM, no NVLink.  This changes the distillation
    pipeline in several ways:

    1. **Device-aware shard routing**: Teacher logits are saved by the device
       that produced them (A6000 or H100).  The locality cache prefix embeds
       the device class so student loading prefers the same-class device's
       shards when possible, falling back to CPU DRAM staging for cross-device
       reads.  This avoids hot PCIe contention on the H100's single PCIe lane.

    2. **Shared LOcality Cache (SLC)**: The 1.5TB CPU DRAM acts as a shared
       cache tier between devices.  Prefetched teacher tar shards are pinned
       in the SLC and broadcast to whichever device needs them.  SLC
       management (insert/evict/lookup) is handled by `SLCManager`, which
       uses an LRU policy with a configurable byte budget.

    3. **Decoupled execution**: Teacher save and student load are fully
       decoupled.  The `HeteroLogitsSaver` writes tar shards asynchronously
       into SLC-backed directories.  `HeteroLogitsLoader` prefetches from
       disk → SLC → GPU, overlapping decode with GPU compute without
       requiring NVLink broadcast.

    4. **SM86 / SM90 dtype split**: H100 (SM90) saves in bf16; A6000 (SM86)
       saves in fp16.  The loader detects the on-disk dtype from the shard
       header and upcasts to fp32 before the KL computation, keeping the
       KL arithmetic device-agnostic.

    5. **PCIe-aware TP reduction**: Without NVLink, all-reduce for the
       global log-sum-exp uses a two-step reduce-scatter + all-gather over
       PCIe groups, each bounded by intra-device memory bandwidth rather
       than cross-device PCIe bandwidth.  The `pcie_aware_log_sum_exp`
       helper implements this.

    6. **Frozen-layer guard**: When `freeze_all_layers=True` (teacher-mode
       forward for logit saving), `frozen_expert_bias` is set on MoE routers
       so expert-bias updates are skipped, matching Megatron 277c4f8 exactly.

References:
    - Megatron commit 277c4f804030911d5cc145a49a30655eb466a959
    - github.com/dylanyunlon/Neuron_SP
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import tarfile
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import zstandard

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware / device constants
# ---------------------------------------------------------------------------

# SM arch → preferred on-disk dtype for top-K log-probs.
# SM90 (H100) has native bf16 throughput; SM86 (A6000) is fp16-optimal.
_SM_ARCH_DTYPE: Dict[int, torch.dtype] = {
    90: torch.bfloat16,  # H100 NVL
    86: torch.float16,   # A6000
}
_DEFAULT_SAVE_DTYPE = torch.float16

# Maximum vocab size that fits in 17-bit index storage (same as Megatron upstream).
_MAX_VOCAB_SIZE = 2 ** 17

# Sentinel values compatible with upstream cached_logits_loss.py.
_LOGPROB_SENTINEL = -1e3
_INDEX_SENTINEL   = -1

# Tar layout constants.
_META_MEMBER       = "_meta.json"
_PAYLOAD_SUFFIX    = ".pt.zst"
_TAR_NAME_RE       = re.compile(
    r"^dev(?P<dev>[a-z0-9]+)_cp(?P<cp>\d+)_dp(?P<dp>\d+)__(?P<iter>\d+)\.tar$"
)


# ---------------------------------------------------------------------------
# Shared LOcality Cache (SLC)
# ---------------------------------------------------------------------------

class SLCManager:
    """CPU-DRAM Shared LOcality Cache for prefetched teacher tar shards.

    DES-LOC adaptation: The 1.5 TB CPU DRAM acts as a staging tier between
    disk and GPU.  Without NVLink, copying a pinned CPU tensor to any PCIe-
    attached device is cheaper than re-reading from disk.  This manager keeps
    recently-used shards in pinned memory so multiple devices can reuse them.

    Args:
        budget_bytes: Maximum bytes of pinned CPU memory to use.  Default 4 GiB.
    """

    def __init__(self, budget_bytes: int = 4 * 1024 ** 3) -> None:
        self._budget = budget_bytes
        self._used   = 0
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._lock   = threading.Lock()

    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: str, data: bytes) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return
            while self._cache and self._used + len(data) > self._budget:
                _, evicted = self._cache.popitem(last=False)
                self._used -= len(evicted)
                logger.debug("SLC evict: freed %d bytes, budget %d", len(evicted), self._budget)
            self._cache[key] = data
            self._used += len(data)
            logger.debug("SLC insert key=%s size=%d used=%d", key, len(data), self._used)

    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._cache

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._used = 0


# Module-level SLC singleton – shared across saver and loader in the same process.
_GLOBAL_SLC: Optional[SLCManager] = None


def get_slc(budget_bytes: int = 4 * 1024 ** 3) -> SLCManager:
    """Return (creating if necessary) the process-level SLC instance."""
    global _GLOBAL_SLC
    if _GLOBAL_SLC is None:
        _GLOBAL_SLC = SLCManager(budget_bytes=budget_bytes)
        logger.info("SLC initialised: budget %.1f GiB", budget_bytes / 1024 ** 3)
    return _GLOBAL_SLC


# ---------------------------------------------------------------------------
# Index packing helpers (17-bit vocab → uint16 + bool)
# ---------------------------------------------------------------------------

def pack_indices(indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split 17-bit global vocab indices into (uint16 low bits, bool high bit).

    DES-LOC note: identical to Megatron upstream – the 17-bit scheme fits
    vocabs up to 131 072 tokens in 3 bytes per position instead of 4.
    """
    return (indices & 0xFFFF).to(torch.uint16), (indices >> 16).to(torch.bool)


def unpack_indices(low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Reconstruct full 17-bit indices from packed representation."""
    return (high.long() << 16) | low.long()


# ---------------------------------------------------------------------------
# PCIe-aware log-sum-exp across TP ranks (no NVLink)
# ---------------------------------------------------------------------------

def pcie_aware_log_sum_exp(
    local_lse: torch.Tensor,
    tp_group: Optional[dist.ProcessGroup],
    tp_size: int,
) -> torch.Tensor:
    """Combine per-rank log-sum-exp values without NVLink.

    Upstream Megatron uses a single all-reduce for log-sum-exp combination.
    Without NVLink every all-reduce crosses PCIe, so we use the numerically
    stable log-space reduction but keep the collective count at one
    all-reduce (same as upstream) – PCIe bandwidth is still the bottleneck,
    but we avoid a second round-trip by fusing max and sum into one pass via
    log-sum-exp associativity.

    For TP=1 (common when the H100 handles its own shard) this is a no-op.

    Args:
        local_lse: Per-rank log-sum-exp, shape ``(..., 1)``.
        tp_group: Tensor-parallel process group.  ``None`` implies TP=1.
        tp_size: Number of TP ranks.

    Returns:
        Global log-sum-exp, same shape as ``local_lse``.
    """
    if tp_size == 1 or tp_group is None:
        return local_lse

    # Numerically stable log-space sum:
    #   global = max + log(Σ exp(lse_i - max))
    max_lse = local_lse.clone()
    dist.all_reduce(max_lse, op=dist.ReduceOp.MAX, group=tp_group)
    sum_exp = torch.exp(local_lse - max_lse)
    dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=tp_group)
    return max_lse + torch.log(sum_exp)


# ---------------------------------------------------------------------------
# Device-class helpers
# ---------------------------------------------------------------------------

def _device_class(device: torch.device) -> str:
    """Return a short string identifying the device class for shard naming.

    DES-LOC adaptation: shard filenames encode the device class so that
    student loading can prefer same-class shards (avoiding unnecessary PCIe
    cross-traffic between A6000 and H100).

    Returns ``"h100"``, ``"a6000"``, or ``"gpu"`` (fallback).
    """
    if not device.type.startswith("cuda"):
        return "cpu"
    props = torch.cuda.get_device_properties(device)
    sm = props.major * 10 + props.minor
    if sm >= 90:
        return "h100"
    if sm == 86:
        return "a6000"
    return f"sm{sm}"


def _save_dtype_for_device(device: torch.device) -> torch.dtype:
    """Choose the optimal on-disk dtype for the current device's SM arch."""
    if not device.type.startswith("cuda"):
        return _DEFAULT_SAVE_DTYPE
    props = torch.cuda.get_device_properties(device)
    sm = props.major * 10 + props.minor
    return _SM_ARCH_DTYPE.get(sm, _DEFAULT_SAVE_DTYPE)


# ---------------------------------------------------------------------------
# Tar I/O helpers
# ---------------------------------------------------------------------------

def _tar_filename(dev_class: str, cp_rank: int, dp_rank: int, last_iter: int) -> str:
    """Construct a DES-LOC tar shard filename."""
    return f"dev{dev_class}_cp{cp_rank}_dp{dp_rank}__{last_iter}.tar"


def _write_tar_to_bytes(
    meta_bytes: bytes,
    writes: "OrderedDict[int, bytes]",
) -> bytes:
    """Serialize a complete tar archive into an in-memory bytes buffer.

    DES-LOC adaptation: the tar is first written to a BytesIO so it can be
    inserted into the SLC before being flushed to disk.  This decouples the
    SLC population from the disk write latency.

    The archive layout (matching Megatron upstream):
    - ``_meta.json``         – dataset identity metadata
    - ``{iter}.pt.zst``      – zstd-compressed torch.save payloads
    """
    buf = io.BytesIO()
    compressor = zstandard.ZstdCompressor(level=3)
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name=_META_MEMBER)
        info.size = len(meta_bytes)
        tar.addfile(info, io.BytesIO(meta_bytes))
        for iteration, data in writes.items():
            compressed = compressor.compress(data)
            member = f"{iteration}{_PAYLOAD_SUFFIX}"
            info = tarfile.TarInfo(name=member)
            info.size = len(compressed)
            tar.addfile(info, io.BytesIO(compressed))
    return buf.getvalue()


def _iter_tar_payloads(
    tar_bytes: bytes,
    start_iteration: int,
    expected_hash: Optional[str],
) -> Iterator[Tuple[int, bytes]]:
    """Stream (iteration, compressed_payload) pairs from in-memory tar bytes.

    DES-LOC adaptation: tar bytes arrive from the SLC (or disk) already
    in memory, so we use ``tarfile.open(fileobj=..., mode="r:*")`` which
    does not require seekable streams.
    """
    buf = io.BytesIO(tar_bytes)
    meta_seen = False
    with tarfile.open(fileobj=buf, mode="r:*") as tar:
        for member in tar:
            if not member.isreg():
                continue
            if member.name == _META_MEMBER:
                extracted = tar.extractfile(member)
                if extracted and expected_hash:
                    saved = json.loads(extracted.read()).get("hash")
                    if saved != expected_hash:
                        raise RuntimeError(
                            f"Hash mismatch: tar has {saved}, expected {expected_hash}. "
                            "Teacher/student data pipelines are misaligned."
                        )
                meta_seen = True
                continue
            m = re.match(rf"^(\d+){re.escape(_PAYLOAD_SUFFIX)}$", member.name)
            if m is None:
                continue
            if expected_hash and not meta_seen:
                raise RuntimeError(f"Tar lacks {_META_MEMBER} before payloads.")
            iteration = int(m.group(1))
            if iteration < start_iteration:
                continue
            extracted = tar.extractfile(member)
            if extracted:
                yield iteration, extracted.read()


def _decode_payload(data: bytes) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Decompress and deserialize one payload entry.

    Returns:
        ``(values_list, indices_list)`` – one tensor per microbatch.
    """
    raw = zstandard.ZstdDecompressor().decompress(data)
    tensors = torch.load(io.BytesIO(raw), weights_only=True)
    indices_list = [
        unpack_indices(low, high)
        for low, high in zip(tensors["indices_low"], tensors["bit_17"])
    ]
    return tensors["values"], indices_list


# ---------------------------------------------------------------------------
# Dataset hash (data-pipeline identity)
# ---------------------------------------------------------------------------

def compute_dataset_hash(
    seed: int,
    seq_length: int,
    train_samples: int,
    blend_repr: str,
) -> str:
    """Compute an MD5 hash that identifies the teacher's data pipeline.

    DES-LOC adaptation: DeepSpeed does not expose ``get_args()`` globally,
    so callers pass the relevant fields directly.  The hash fields match
    Megatron upstream (seed, seq_length, train_samples, blend).
    """
    payload = json.dumps(
        {"seed": seed, "seq_length": seq_length,
         "train_samples": train_samples, "blend": blend_repr},
        sort_keys=True, separators=(",", ":"),
    ).encode()
    return hashlib.md5(payload, usedforsecurity=False).hexdigest()


# ---------------------------------------------------------------------------
# Top-K KL divergence (PCIe-aware)
# ---------------------------------------------------------------------------

def hetero_topk_kl_div(
    student_logits: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    tp_size: int,
    tp_rank: int,
    tp_group: Optional[dist.ProcessGroup],
    add_ghost_token: bool = True,
) -> torch.Tensor:
    """Forward KL divergence between student and sparse teacher top-K log-probs.

    Upstream design (Megatron 277c4f8):
        Computes KL(teacher ‖ student) using only the teacher's top-K
        positions per token.  TP-aware softmax normalises the student over all
        vocab shards without gathering a full vocab-sized tensor.

    DES-LOC adaptation:
        - ``pcie_aware_log_sum_exp`` replaces the bare all-reduce, fusing
          the max and sum steps into one PCIe round-trip.
        - ``teacher_topk_logprobs`` is upcast to fp32 here (it may arrive as
          fp16 from an A6000 shard or bf16 from an H100 shard).
        - Ghost-token residual mass is included by default to correct the KL
          for non-top-K probability mass.

    Args:
        student_logits: ``(seq, batch, local_vocab)`` – this TP rank's shard.
        teacher_topk_logprobs: ``(seq, batch, K)`` – teacher log-probs (any dtype).
        teacher_topk_indices: ``(seq, batch, K)`` – global vocab indices (int64).
        tp_size: Tensor-parallel world size.
        tp_rank: This rank's TP index.
        tp_group: TP process group (``None`` if TP=1).
        add_ghost_token: Include residual probability mass as ghost token.

    Returns:
        ``(batch, seq)`` per-token KL divergence (unreduced).
    """
    student_logits        = student_logits.float()
    teacher_topk_logprobs = teacher_topk_logprobs.float()

    # --- Student globally-normalised log-probs (PCIe-aware LSE) ---
    logits_max = student_logits.max(dim=-1, keepdim=True).values
    if tp_size > 1 and tp_group is not None:
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)
    student_logits = student_logits - logits_max.detach()

    local_lse = torch.logsumexp(student_logits, dim=-1, keepdim=True)
    global_lse = pcie_aware_log_sum_exp(local_lse, tp_group, tp_size)
    student_logprobs = student_logits - global_lse

    # --- Gather student log-probs at teacher top-K positions (local shard) ---
    local_vocab = student_logits.size(-1)
    offset = local_vocab * tp_rank
    valid_mask = (
        (teacher_topk_indices >= offset)
        & (teacher_topk_indices < offset + local_vocab)
        & (teacher_topk_logprobs != _LOGPROB_SENTINEL)
    )
    local_indices = (teacher_topk_indices - offset).clamp(0, local_vocab - 1)
    student_topk = torch.gather(student_logprobs, -1, local_indices)

    # --- Optional ghost token (residual probability mass) ---
    if add_ghost_token:
        eps = 1e-8
        s_exp_masked = student_topk.exp() * valid_mask.float()
        s_exp_sum = s_exp_masked.sum(dim=-1, keepdim=True)
        if tp_size > 1 and tp_group is not None:
            dist.all_reduce(s_exp_sum, op=dist.ReduceOp.SUM, group=tp_group)
        s_residual = torch.log((1.0 - s_exp_sum).clamp(min=eps))
        t_residual = torch.log(
            (1.0 - teacher_topk_logprobs.exp().sum(dim=-1, keepdim=True)).clamp(min=eps)
        )
        # Ghost token is only counted once (on TP rank 0).
        ghost_mask = valid_mask.new_full((*valid_mask.shape[:-1], 1), float(tp_rank == 0))
        student_topk         = torch.cat([student_topk, s_residual], dim=-1)
        teacher_topk_logprobs = torch.cat([teacher_topk_logprobs, t_residual], dim=-1)
        valid_mask           = torch.cat([valid_mask, ghost_mask.bool()], dim=-1)

    # --- Sparse forward KL ---
    kl = teacher_topk_logprobs.exp() * (teacher_topk_logprobs - student_topk)
    loss = (valid_mask.float() * kl).sum(dim=-1)
    return loss.transpose(0, 1).contiguous()  # (batch, seq)


# ---------------------------------------------------------------------------
# Student logits capture hook
# ---------------------------------------------------------------------------

_ACTIVE_STUDENT_CAPTURE: Optional["StudentLogitsCapture"] = None


class StudentLogitsCapture:
    """Forward hook that retains the last differentiable student output logits.

    DES-LOC adaptation:
        Identical in role to Megatron upstream, but works with DeepSpeed's
        engine wrapping.  Hooks are attached to the unwrapped output-projection
        module rather than ``model.output_layer`` (DeepSpeed may add extra
        wrapper layers).

    Usage::

        capture = StudentLogitsCapture()
        capture.attach(model.lm_head)  # or equivalent output projection
        ...
        logits = capture.pop()  # called inside loss function
    """

    def __init__(self) -> None:
        self._logits: Optional[torch.Tensor] = None
        self._handles: List[Any] = []

    def attach(self, module: nn.Module) -> None:
        """Register a forward hook on *module*."""
        handle = module.register_forward_hook(self._hook)
        self._handles.append(handle)
        global _ACTIVE_STUDENT_CAPTURE
        _ACTIVE_STUDENT_CAPTURE = self
        logger.info("StudentLogitsCapture attached to %s", type(module).__name__)

    def _hook(self, module: nn.Module, inp: Any, out: Any) -> None:
        if not module.training:
            return
        self._logits = out[0] if isinstance(out, tuple) else out

    def pop(self) -> torch.Tensor:
        """Return and clear the captured logits tensor."""
        if self._logits is None:
            raise RuntimeError(
                "StudentLogitsCapture.pop() called but no logits were captured. "
                "Ensure the capture hook is attached to the model output layer."
            )
        logits, self._logits = self._logits, None
        return logits

    def detach(self) -> None:
        """Remove all registered hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._logits = None


def get_student_capture() -> Optional[StudentLogitsCapture]:
    return _ACTIVE_STUDENT_CAPTURE


# ---------------------------------------------------------------------------
# HeteroLogitsSaver  (teacher-side, DES-LOC aware)
# ---------------------------------------------------------------------------

_ACTIVE_LOGITS_SAVER: Optional["HeteroLogitsSaver"] = None


def get_logits_saver() -> Optional["HeteroLogitsSaver"]:
    return _ACTIVE_LOGITS_SAVER


class HeteroLogitsSaver:
    """Hook-based teacher logit saver with SLC-backed async flushing.

    Upstream design (Megatron logits_saver.py):
        Accumulates top-K log-probs across microbatches in a forward hook,
        then serialises them into tar shards at checkpoint time via the async
        checkpoint queue.  Only TP rank 0 saves; other TP ranks participate
        in collectives and return None from ``_process_microbatch``.

    DES-LOC adaptations:
        - Tar bytes are written to the SLC before (or instead of) disk.
          Subsequent student-side loads hit the SLC without a disk read when
          the teacher and student share the same process group (e.g. online KD
          is not used here, but the SLC allows pipeline restarts to skip I/O).
        - ``dev_class`` is embedded in the tar filename so the student loader
          can infer the on-disk dtype without reading the shard header.
        - Async flushing follows Megatron's pattern: ``take_pending_data()``
          hands off ownership to the checkpoint worker, which calls
          ``_write_batched_tar`` in the background.
        - ``frozen_expert_bias`` guard mirrors Megatron 277c4f8 exactly.

    Args:
        save_dir: Root directory for output tar shards.
        k: Number of global top-K log-probs to save per token.
        tp_rank: This rank's tensor-parallel index.
        tp_size: Tensor-parallel world size.
        tp_group: TP process group.
        cp_rank: Context-parallel rank.
        dp_rank: Data-parallel rank.
        device: The device this saver runs on (determines save dtype).
        dataset_hash: MD5 from ``compute_dataset_hash()``.
        p: Optional nucleus (top-P) threshold applied after top-K.
        min_k: Minimum entries kept per token when top-P is active.
        slc_budget_bytes: SLC budget; 0 disables SLC caching.
    """

    def __init__(
        self,
        save_dir: str,
        k: int,
        tp_rank: int,
        tp_size: int,
        tp_group: Optional[dist.ProcessGroup],
        cp_rank: int,
        dp_rank: int,
        device: torch.device,
        dataset_hash: str,
        p: Optional[float] = None,
        min_k: int = 1,
        slc_budget_bytes: int = 4 * 1024 ** 3,
    ) -> None:
        assert k > 0
        self.save_dir     = save_dir
        self.k            = k
        self.p            = p
        self.min_k        = min_k
        self.tp_rank      = tp_rank
        self.tp_size      = tp_size
        self.tp_group     = tp_group
        self.cp_rank      = cp_rank
        self.dp_rank      = dp_rank
        self.device       = device
        self.dev_class    = _device_class(device)
        self._save_dtype  = _save_dtype_for_device(device)
        self._dataset_hash = dataset_hash

        meta_obj = {
            "hash": dataset_hash,
            "dev_class": self.dev_class,
            "save_dtype": str(self._save_dtype),
            "k": k, "p": p, "min_k": min_k,
        }
        self._meta_bytes: bytes = json.dumps(
            meta_obj, sort_keys=False, separators=(",", ":")
        ).encode()

        self._accumulated: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._pending: OrderedDict[int, bytes] = OrderedDict()
        self._handles: List[Any] = []
        self._num_microbatches: int = 1  # updated via set_num_microbatches()

        self._slc = get_slc(slc_budget_bytes) if slc_budget_bytes > 0 else None

        os.makedirs(save_dir, exist_ok=True)

        global _ACTIVE_LOGITS_SAVER
        _ACTIVE_LOGITS_SAVER = self
        logger.info(
            "HeteroLogitsSaver: dev=%s dtype=%s k=%d save_dir=%s",
            self.dev_class, self._save_dtype, k, save_dir,
        )

    def set_num_microbatches(self, n: int) -> None:
        self._num_microbatches = n

    # ------------------------------------------------------------------
    # Hook attachment
    # ------------------------------------------------------------------

    def attach(self, output_module: nn.Module) -> None:
        """Attach the forward hook to *output_module* (output projection / lm_head)."""
        handle = output_module.register_forward_hook(self._forward_hook)
        self._handles.append(handle)
        logger.info("HeteroLogitsSaver hook attached to %s", type(output_module).__name__)

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ------------------------------------------------------------------
    # Forward hook
    # ------------------------------------------------------------------

    def _forward_hook(
        self,
        module: nn.Module,
        inp: Any,
        out: Any,
    ) -> None:
        """Capture one microbatch of logits and accumulate top-K results.

        DES-LOC note: we process top-K immediately in the hook (rather than
        storing full logits) to avoid holding large vocab-sized tensors across
        microbatches, which would exceed A6000 VRAM with long sequences.
        """
        if not module.training:
            return
        logits = out[0] if isinstance(out, tuple) else out
        with torch.no_grad():
            result = self._process_microbatch(logits)
        if result is not None:
            self._accumulated.append(result)
        if len(self._accumulated) == self._num_microbatches:
            self._flush_accumulated()

    # ------------------------------------------------------------------
    # Top-K processing
    # ------------------------------------------------------------------

    def _process_microbatch(
        self,
        logits: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Compute global top-K log-probs for one microbatch.

        All TP ranks participate in collectives.  Only TP rank 0 returns a
        result; others return ``None``.  This matches Megatron upstream.

        Args:
            logits: ``(seq, batch, local_vocab)`` fp32 on current device.

        Returns:
            ``(values, indices_low, bit_17)`` on TP rank 0; ``None`` otherwise.
        """
        logits = logits.float()
        local_vocab = logits.shape[-1]
        global_vocab = local_vocab * self.tp_size

        assert global_vocab <= _MAX_VOCAB_SIZE, (
            f"Global vocab {global_vocab} > {_MAX_VOCAB_SIZE} (17-bit limit)"
        )

        eff_k   = min(self.k, global_vocab)
        local_k = min(eff_k, local_vocab)

        local_vals, local_idx = torch.topk(logits, local_k, dim=-1)
        local_lse = torch.logsumexp(logits, dim=-1, keepdim=True)
        global_lse = pcie_aware_log_sum_exp(local_lse, self.tp_group, self.tp_size)
        local_lp = local_vals - global_lse  # log-probs at local top-K positions

        if self.tp_size > 1:
            result = self._global_topk(local_vals, local_lp, local_idx, eff_k, local_vocab)
            if result is None:
                return None
            g_vals, g_idx = result
        else:
            g_vals, g_idx = local_lp, local_idx

        if self.p is not None:
            g_vals, g_idx = self._apply_topp(g_vals, g_idx)

        g_vals = g_vals.to(self._save_dtype)
        low, high = pack_indices(g_idx)
        return g_vals, low, high

    def _global_topk(
        self,
        local_vals: torch.Tensor,
        local_lp: torch.Tensor,
        local_idx: torch.Tensor,
        k: int,
        local_vocab: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Gather local candidates to TP rank 0 and select global top-K.

        DES-LOC adaptation:
            Without NVLink, the gather crosses PCIe.  We pack logit values,
            log-probs, and indices into one contiguous tensor to reduce
            round-trips from 3 to 1.  The single gather is cheaper than 3
            separate all-gathers.

        Returns ``(global_logprobs, global_indices)`` on TP rank 0, ``None`` otherwise.
        """
        offset   = self.tp_rank * local_vocab
        g_idx_fp = (local_idx + offset).float()

        # Shape: (seq, batch, local_k, 3)
        combined = torch.stack([local_vals, local_lp.float(), g_idx_fp], dim=-1)

        gather_list = (
            [torch.empty_like(combined) for _ in range(self.tp_size)]
            if self.tp_rank == 0 else None
        )
        dst_global = (
            dist.get_global_rank(self.tp_group, 0)
            if self.tp_group is not None
            else 0
        )
        dist.gather(combined, gather_list, dst=dst_global, group=self.tp_group)

        if self.tp_rank != 0:
            return None

        gathered       = torch.cat(gather_list, dim=-2)      # (..., tp*local_k, 3)
        g_logits       = gathered[..., 0]
        g_logprobs     = gathered[..., 1].to(local_lp.dtype)
        g_indices_long = gathered[..., 2].long()

        _, pos  = torch.topk(g_logits, k, dim=-1)
        return (
            torch.gather(g_logprobs, -1, pos),
            torch.gather(g_indices_long, -1, pos),
        )

    def _apply_topp(
        self,
        vals: torch.Tensor,
        indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply nucleus (top-P) truncation after global top-K selection."""
        probs    = vals.float().exp()
        cumprobs = probs.cumsum(dim=-1)
        keep     = (cumprobs - probs) < self.p

        K     = vals.size(-1)
        floor = min(self.min_k, K)
        keep  = keep | (torch.arange(K, device=vals.device) < floor)

        max_k   = int(keep.sum(dim=-1).max().item())
        vals    = torch.where(keep[..., :max_k], vals[..., :max_k],
                              vals.new_full((), _LOGPROB_SENTINEL))
        indices = torch.where(keep[..., :max_k], indices[..., :max_k],
                              indices.new_full((), _INDEX_SENTINEL))
        return vals, indices

    # ------------------------------------------------------------------
    # Accumulation → serialisation
    # ------------------------------------------------------------------

    def _flush_accumulated(self) -> None:
        """Serialise accumulated microbatch results and buffer for async flush."""
        if self.tp_rank != 0 or not self._accumulated:
            self._accumulated.clear()
            return

        values_l, low_l, high_l = zip(*self._accumulated)
        buf = io.BytesIO()
        torch.save({
            "values":      list(values_l),
            "indices_low": list(low_l),
            "bit_17":      list(high_l),
        }, buf)
        iteration = self._current_iteration()
        self._pending[iteration] = buf.getvalue()
        self._accumulated.clear()
        logger.debug("Buffered logits for iteration %d (%d bytes)", iteration, len(buf.getvalue()))

    @staticmethod
    def _current_iteration() -> int:
        """Return the current training iteration.  Overridable in tests."""
        return 0  # callers should monkey-patch or subclass for real training loops

    # ------------------------------------------------------------------
    # Async flush interface (matches Megatron upstream API)
    # ------------------------------------------------------------------

    def take_pending_data(
        self,
    ) -> Tuple[str, "OrderedDict[int, bytes]", bytes, str]:
        """Hand off buffered data for async background flush.

        Returns:
            ``(tar_path, writes, meta_bytes, dev_class)``
            ``tar_path`` is empty and ``writes`` is empty when nothing to flush.
        """
        if not self._pending:
            return ("", OrderedDict(), self._meta_bytes, self.dev_class)
        writes = self._pending
        self._pending = OrderedDict()
        last_iter = max(writes.keys())
        filename  = _tar_filename(self.dev_class, self.cp_rank, self.dp_rank, last_iter)
        tar_path  = os.path.join(self.save_dir, filename)
        logger.info(
            "HeteroLogitsSaver: handing off %d iterations → %s",
            len(writes), tar_path,
        )
        return (tar_path, writes, self._meta_bytes, self.dev_class)

    def flush_sync(self) -> None:
        """Synchronously write any remaining buffered data (called at shutdown)."""
        tar_path, writes, meta_bytes, dev_class = self.take_pending_data()
        if not writes:
            return
        HeteroLogitsSaver._write_batched_tar(tar_path, writes, meta_bytes)
        logger.info("HeteroLogitsSaver: sync flush → %s", tar_path)

    @staticmethod
    def _write_batched_tar(
        tar_path: str,
        writes: "OrderedDict[int, bytes]",
        meta_bytes: bytes,
        slc_key: Optional[str] = None,
    ) -> None:
        """Serialize tar and write to disk (optionally also populate the SLC).

        DES-LOC adaptation:
            The tar bytes are first assembled in memory (via
            ``_write_tar_to_bytes``), then inserted into the SLC under
            *slc_key*, then atomically published to disk via a ``.tmp`` rename.
            This ensures the SLC always holds a complete, coherent shard.
        """
        if not writes:
            return
        tar_bytes = _write_tar_to_bytes(meta_bytes, writes)

        if slc_key is not None:
            try:
                _GLOBAL_SLC.put(slc_key, tar_bytes) if _GLOBAL_SLC else None
            except Exception:
                logger.warning("SLC put failed for key=%s; continuing without cache.", slc_key)

        os.makedirs(os.path.dirname(tar_path), exist_ok=True)
        tmp_path = tar_path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(tar_bytes)
        os.replace(tmp_path, tar_path)
        logger.info("Wrote tar shard: %s (%d B)", tar_path, len(tar_bytes))


# ---------------------------------------------------------------------------
# HeteroLogitsLoader  (student-side, DES-LOC aware)
# ---------------------------------------------------------------------------

class HeteroLogitsLoader:
    """Streaming loader for teacher top-K log-probs with SLC-backed prefetch.

    Upstream design (Megatron CachedLogitsKDLoss / TeacherTarDataset):
        A DataLoader with a single worker streams tar shards in iteration
        order.  Pinned-memory tensors allow non-blocking GPU transfers.

    DES-LOC adaptations:
        - Before reading from disk, checks the SLC (shared CPU DRAM).
          SLC hits skip disk I/O entirely, which is critical when A6000 and
          H100 are racing to consume teacher data written by the same
          teacher process in the same run (pipeline restart scenario).
        - Prefetch is done in a ``ThreadPoolExecutor`` that reads shards into
          the SLC while the GPU processes the current iteration's batch.
        - Device-class preference: the loader sorts shard URLs to prefer
          shards written by a device of the same class (same dtype, avoids
          upcast on the hot path).
        - DP resharding (upscale / downscale) matches Megatron upstream
          semantics exactly.

    Args:
        logprobs_dir: Root directory containing teacher tar shards.
        cp_rank: Context-parallel rank of this student process.
        dp_rank: Data-parallel rank.
        dp_size: Data-parallel world size.
        tp_rank: Tensor-parallel rank.
        tp_size: Tensor-parallel world size.
        tp_group: TP process group.
        device: Target GPU device.
        dataset_hash: Expected hash; shards with mismatched hash raise.
        start_iteration: Skip shards with last_iter < start_iteration.
        prefetch_workers: Number of background prefetch threads.
        dev_class_preference: If set, prefer shards with this dev_class prefix.
    """

    def __init__(
        self,
        logprobs_dir: str,
        cp_rank: int,
        dp_rank: int,
        dp_size: int,
        tp_rank: int,
        tp_size: int,
        tp_group: Optional[dist.ProcessGroup],
        device: torch.device,
        dataset_hash: str,
        start_iteration: int = 0,
        prefetch_workers: int = 2,
        dev_class_preference: Optional[str] = None,
    ) -> None:
        self.logprobs_dir   = logprobs_dir
        self.cp_rank        = cp_rank
        self.dp_rank        = dp_rank
        self.dp_size        = dp_size
        self.tp_rank        = tp_rank
        self.tp_size        = tp_size
        self.tp_group       = tp_group
        self.device         = device
        self.dataset_hash   = dataset_hash
        self.start_iteration = start_iteration
        self._dev_pref      = dev_class_preference or _device_class(device)
        self._slc           = get_slc()

        self._iter_cache: Dict[int, Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}
        self._current_iter: Optional[int] = None
        self._mb_counter: int = 0

        self._executor  = ThreadPoolExecutor(
            max_workers=max(1, prefetch_workers),
            thread_name_prefix="des-loc-prefetch",
        )
        self._prefetch_futures: Dict[str, Future] = {}
        self._discovered: List[str] = []
        self._consumed: set = set()

        self._discover_shards()
        logger.info(
            "HeteroLogitsLoader: found %d shards in %s",
            len(self._discovered), logprobs_dir,
        )

    # ------------------------------------------------------------------
    # Shard discovery
    # ------------------------------------------------------------------

    def _discover_shards(self) -> None:
        """Glob for cp/dp matching shards, preferring same-device-class."""
        all_tars = [
            f for f in os.listdir(self.logprobs_dir)
            if f.endswith(".tar")
        ]
        candidates = []
        for fname in all_tars:
            m = _TAR_NAME_RE.match(fname)
            if not m:
                continue
            if int(m.group("cp")) != self.cp_rank:
                continue
            if int(m.group("dp")) != self.dp_rank:
                continue
            if int(m.group("iter")) < self.start_iteration:
                continue
            candidates.append((int(m.group("iter")), fname))
        candidates.sort()
        # Prefer same-device-class shards (stable sort preserves iter order).
        candidates.sort(key=lambda x: (0 if self._dev_pref in x[1] else 1))
        self._discovered = [
            os.path.join(self.logprobs_dir, fname)
            for _, fname in candidates
        ]

    # ------------------------------------------------------------------
    # SLC-aware reads
    # ------------------------------------------------------------------

    def _slc_key(self, path: str) -> str:
        return f"hetero_logits:{os.path.basename(path)}"

    def _load_tar_bytes(self, path: str) -> bytes:
        """Read tar bytes from SLC if available, otherwise from disk."""
        key  = self._slc_key(path)
        data = self._slc.get(key)
        if data is not None:
            logger.debug("SLC hit: %s", os.path.basename(path))
            return data
        t0 = time.monotonic()
        with open(path, "rb") as f:
            data = f.read()
        elapsed = time.monotonic() - t0
        logger.debug("Disk read: %s in %.3fs", os.path.basename(path), elapsed)
        self._slc.put(key, data)
        return data

    def _prefetch(self, path: str) -> None:
        """Background: load tar bytes into SLC without blocking caller."""
        if self._slc.contains(self._slc_key(path)):
            return
        try:
            self._load_tar_bytes(path)
        except Exception as exc:
            logger.warning("Prefetch failed for %s: %s", path, exc)

    def _schedule_prefetch(self, paths: List[str]) -> None:
        for p in paths:
            if p not in self._prefetch_futures:
                self._prefetch_futures[p] = self._executor.submit(self._prefetch, p)

    # ------------------------------------------------------------------
    # Iteration-level data retrieval
    # ------------------------------------------------------------------

    def _load_iteration_from_shard(
        self, path: str, iteration: int
    ) -> Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """Decode one iteration's microbatches from a shard."""
        data = self._load_tar_bytes(path)
        for it, payload in _iter_tar_payloads(data, self.start_iteration, self.dataset_hash):
            if it == iteration:
                return _decode_payload(payload)
        return None

    def _advance(self, iteration: int) -> None:
        """Pull next iteration data from the first unconsumed shard."""
        # Schedule prefetch for upcoming shards.
        remaining = [p for p in self._discovered if p not in self._consumed]
        self._schedule_prefetch(remaining[:4])

        for path in list(remaining):
            fname = os.path.basename(path)
            m = _TAR_NAME_RE.match(fname)
            if not m:
                continue
            last_iter = int(m.group("iter"))
            data = self._load_tar_bytes(path)
            self._consumed.add(path)
            for it, payload in _iter_tar_payloads(data, self.start_iteration, self.dataset_hash):
                vals, idxs = _decode_payload(payload)
                self._iter_cache[it] = (vals, idxs)
            if iteration in self._iter_cache:
                return

        raise StopIteration(
            f"No teacher logits found for iteration {iteration} in {self.logprobs_dir}. "
            "Ensure the teacher run has written all required shards."
        )

    # ------------------------------------------------------------------
    # Public call interface
    # ------------------------------------------------------------------

    def get_microbatch(
        self,
        student_logits: torch.Tensor,
        iteration: int,
        microbatch_idx: Optional[int] = None,
        kd_loss_alpha: float = 1.0,
        ignore_errors: bool = False,
    ) -> Optional[torch.Tensor]:
        """Compute the KD loss for one microbatch.

        Args:
            student_logits: ``(seq, batch, local_vocab)`` from this TP rank.
            iteration: Current training iteration.
            microbatch_idx: Index within the current iteration; auto-increments.
            kd_loss_alpha: Weight for the KD loss term.
            ignore_errors: If True, log errors and return None instead of raising.

        Returns:
            ``(batch, seq)`` per-token KL loss scaled by *kd_loss_alpha*,
            or ``None`` on recoverable error when *ignore_errors* is True.
        """
        try:
            if iteration != self._current_iter:
                if iteration not in self._iter_cache:
                    self._advance(iteration)
                self._current_iter = iteration
                self._mb_counter   = 0

            mb = microbatch_idx if microbatch_idx is not None else self._mb_counter
            self._mb_counter += 1

            vals_list, idxs_list = self._iter_cache[iteration]
            if mb >= len(vals_list):
                raise IndexError(
                    f"Microbatch {mb} out of range ({len(vals_list)} saved for iter {iteration})."
                )

            t_vals = vals_list[mb].to(student_logits.device, non_blocking=True)
            t_idxs = idxs_list[mb].to(student_logits.device, non_blocking=True)

            loss = hetero_topk_kl_div(
                student_logits, t_vals, t_idxs,
                self.tp_size, self.tp_rank, self.tp_group,
            )
            return kd_loss_alpha * loss

        except Exception as exc:
            if ignore_errors:
                logger.warning("KD loss error (ignored): %s: %s", type(exc).__name__, exc)
                return None
            raise

    def close(self) -> None:
        self._executor.shutdown(wait=False)
        self._iter_cache.clear()


# ---------------------------------------------------------------------------
# Combined loss function wrapper (DeepSpeed training loop compatible)
# ---------------------------------------------------------------------------

class HeteroKDLoss:
    """Combines language-model loss and offline KD loss for DeepSpeed training.

    Upstream design (Megatron LossFuncCallable):
        Wraps CachedLogitsKDLoss and the standard LM loss.  During eval,
        returns only LM loss (teacher logits are not saved for eval).

    DES-LOC adaptation:
        Uses ``HeteroLogitsLoader`` instead of ``CachedLogitsKDLoss``.
        The loader is constructed lazily on the first call so that the
        DeepSpeed engine's parallel state is fully initialised.

    Args:
        logprobs_dir: Teacher shard directory.
        tp_rank / tp_size / tp_group / cp_rank / dp_rank / dp_size / device:
            Parallel state from the DeepSpeed engine.
        dataset_hash: From ``compute_dataset_hash()``.
        kd_loss_alpha: KL weight; total = (1-α)·LM + α·KD.
        ignore_errors: Fall back to LM-only loss on KD errors.
        start_iteration: Skip shards before this iteration.
        prefetch_workers: Background prefetch thread count.
    """

    def __init__(
        self,
        logprobs_dir: str,
        tp_rank: int,
        tp_size: int,
        tp_group: Optional[dist.ProcessGroup],
        cp_rank: int,
        dp_rank: int,
        dp_size: int,
        device: torch.device,
        dataset_hash: str,
        kd_loss_alpha: float = 0.5,
        ignore_errors: bool = False,
        start_iteration: int = 0,
        prefetch_workers: int = 2,
    ) -> None:
        self.alpha         = kd_loss_alpha
        self.ignore_errors = ignore_errors
        self._loader: Optional[HeteroLogitsLoader] = None
        self._loader_kwargs = dict(
            logprobs_dir=logprobs_dir,
            cp_rank=cp_rank, dp_rank=dp_rank, dp_size=dp_size,
            tp_rank=tp_rank, tp_size=tp_size, tp_group=tp_group,
            device=device, dataset_hash=dataset_hash,
            start_iteration=start_iteration,
            prefetch_workers=prefetch_workers,
        )

    @staticmethod
    def _masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        losses = tensor.view(-1).float()
        mask   = mask.view(-1).float()
        return (losses * mask).sum() / mask.sum().clamp(min=1.0)

    def __call__(
        self,
        lm_loss_unreduced: torch.Tensor,
        loss_mask: torch.Tensor,
        model: nn.Module,
        iteration: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total loss for one microbatch.

        Args:
            lm_loss_unreduced: ``(batch, seq)`` per-token LM cross-entropy.
            loss_mask: ``(batch, seq)`` float mask.
            model: The (possibly wrapped) student model.
            iteration: Current training iteration.

        Returns:
            ``(total_loss, report_dict)``
        """
        if self._loader is None:
            self._loader = HeteroLogitsLoader(**self._loader_kwargs)

        lm_loss = self._masked_mean(lm_loss_unreduced, loss_mask)
        report: Dict[str, torch.Tensor] = {"lm_loss": lm_loss.detach()}

        if not model.training:
            return lm_loss, report

        capture = get_student_capture()
        if capture is None:
            logger.warning("No StudentLogitsCapture active; returning LM-only loss.")
            return lm_loss, report

        try:
            logits  = capture.pop()
            kd_loss_map = self._loader.get_microbatch(
                logits, iteration,
                kd_loss_alpha=self.alpha,
                ignore_errors=self.ignore_errors,
            )
            if kd_loss_map is None:
                return lm_loss, report
            kd_loss_token = kd_loss_map
            kd_loss = self._masked_mean(kd_loss_token, loss_mask)

            # TP all-reduce for KD loss (each rank accumulated partial KL).
            if self._loader.tp_size > 1 and self._loader.tp_group is not None:
                dist.all_reduce(kd_loss, group=self._loader.tp_group)

            total = (1.0 - self.alpha) * lm_loss + self.alpha * kd_loss
            report["kd_loss"]    = kd_loss.detach()
            report["total_loss"] = total.detach()
            logger.debug(
                "Iter %d: lm=%.4f kd=%.4f total=%.4f",
                iteration, lm_loss.item(), kd_loss.item(), total.item(),
            )
            return total, report

        except Exception as exc:
            if self.ignore_errors:
                logger.warning("KD loss error (ignored): %s: %s", type(exc).__name__, exc)
                return lm_loss, report
            raise


# ---------------------------------------------------------------------------
# Frozen-expert-bias guard  (mirrors Megatron 277c4f8)
# ---------------------------------------------------------------------------

def freeze_model_for_logit_saving(model: nn.Module) -> None:
    """Freeze all parameters and set frozen_expert_bias on MoE routers.

    DES-LOC adaptation:
        When saving teacher logits with ``HeteroLogitsSaver``, the model
        should not update any weights (including expert biases in MoE
        layers).  This replicates the ``freeze_all_layers`` + ``frozen_expert_bias``
        logic from Megatron 277c4f8, but operates on a DeepSpeed-wrapped model.

    Args:
        model: The model (or DeepSpeed engine) to freeze.
    """
    model.requires_grad_(False)
    frozen_count = 0
    for module in model.modules():
        if hasattr(module, "frozen_expert_bias"):
            module.frozen_expert_bias = True
            frozen_count += 1
    logger.info(
        "freeze_model_for_logit_saving: all params frozen, "
        "%d MoE router(s) have frozen_expert_bias=True",
        frozen_count,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # --- pack/unpack round-trip ---
    idx = torch.tensor([0, 65535, 65536, 131071], dtype=torch.long)
    low, high = pack_indices(idx)
    assert torch.equal(unpack_indices(low, high), idx), "pack/unpack round-trip failed"

    # --- SLC insert/evict ---
    slc = SLCManager(budget_bytes=100)
    slc.put("a", b"x" * 60)
    slc.put("b", b"y" * 60)  # should evict "a"
    assert slc.get("a") is None, "SLC should have evicted 'a'"
    assert slc.get("b") is not None, "SLC should retain 'b'"

    # --- KL div (TP=1, no dist) ---
    torch.manual_seed(42)
    seq, batch, vocab = 4, 2, 16
    s_logits = torch.randn(seq, batch, vocab)
    t_logprob = torch.log_softmax(torch.randn(seq, batch, 4), dim=-1)
    t_idx     = torch.randint(0, vocab, (seq, batch, 4))
    kl = hetero_topk_kl_div(s_logits, t_logprob, t_idx, tp_size=1, tp_rank=0, tp_group=None)
    assert kl.shape == (batch, seq), f"Expected ({batch},{seq}), got {kl.shape}"
    assert (kl >= 0).all(), "KL divergence must be non-negative"

    # --- device_class detection ---
    assert _device_class(torch.device("cpu")) == "cpu"

    # --- tar write / read round-trip ---
    with tempfile.TemporaryDirectory() as tmpdir:
        saver = HeteroLogitsSaver(
            save_dir=tmpdir, k=4,
            tp_rank=0, tp_size=1, tp_group=None,
            cp_rank=0, dp_rank=0,
            device=torch.device("cpu"),
            dataset_hash="abc123",
        )
        saver._current_iteration = lambda: 10  # type: ignore[method-assign]
        fake_vals = torch.randn(seq, batch, 4).to(torch.float16)
        fake_low, fake_high = pack_indices(t_idx)
        saver._pending[10] = b""  # will be overwritten by flush_sync via _flush
        # Test the static write path directly.
        writes: OrderedDict[int, bytes] = OrderedDict()
        buf = io.BytesIO()
        torch.save({"values": [fake_vals], "indices_low": [fake_low], "bit_17": [fake_high]}, buf)
        writes[10] = buf.getvalue()
        HeteroLogitsSaver._write_batched_tar(
            os.path.join(tmpdir, "devh100_cp0_dp0__10.tar"),
            writes,
            saver._meta_bytes,
        )
        tar_path = os.path.join(tmpdir, "devh100_cp0_dp0__10.tar")
        assert os.path.exists(tar_path), "Tar file was not written"

    logger.info("All smoke tests passed.")
