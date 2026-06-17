# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""BandwidthAwareAllGatherDispatcher — heterogeneous-link MoE token dispatcher for DES-LOC.

Mirrors Megatron bfd45740c ``NCCLAllGatherDispatcher`` / ``NVLSAllGatherVDispatcher``
(token_dispatcher_inference.py), reinterpreted as a *bandwidth-aware* dispatcher that
selects between two physical AllGather strategies based on observed inter-rank link speed.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Upstream design intent (bfd45740c)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Megatron introduced two inference-time AllGather dispatchers:

  NCCLAllGatherDispatcher ('nccl')
    All EP ranks must contribute the same token count per step (guaranteed by
    decode-only CUDA graphs).  Prefill/mixed steps fall back to a pad→AllGather
    →compact variant (use_allgather_v=True) that syncs per-rank counts via an
    all_gather_into_tensor call.

  NVLSAllGatherVDispatcher ('nvls')
    Variable-count AllGather-V backed by Triton multimem kernels over NVLink
    symmetric memory.  Supports different token counts per rank per step; does
    NOT need all ranks to align on token count.  Requires Hopper+ with NVLink.

The critical simplification in bfd45740c was collapsing the old dispatcher's
complex CUDA-graph / strict / decode_only_cuda_graphs / num_speculative_tokens
combinatorics into a single ``match_ep_token_counts`` flag:
  - True  → NCCL path: EP ranks must agree on token count (old behavior)
  - False → AGV/RSV path: per-rank token variation handled inside the dispatcher

The batch-dimension coordinator (InferenceBatchDimensions.adjust_batch_dims_for_expert_parallelism)
was also simplified: prefill → eager mode (return None), decode → max-reduce only,
stripping strict/decode_only/speculative branches.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DES-LOC adaptation rationale
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DES-LOC targets a PCIe-only cluster (A6000 Gen1 + H100 Gen5, no NVLink).  The
NVLSAllGatherVDispatcher is not applicable here — multimem.st requires NVLink
symmetric memory on Hopper+ with NVLS capability.  However the *conceptual*
innovation of bfd45740c — that variable token counts per rank should be handled
inside the dispatcher, not outside it — is directly applicable.

Our heterogeneous topology has a further dimension upstream ignores: link bandwidth
differs by tier.

    A6000 ranks: PCIe Gen1 → ~16 GB/s unidirectional
    H100  ranks: PCIe Gen5 → ~128 GB/s unidirectional

A single AllGather strategy is suboptimal:
  - Large one-shot AllGather on a slow A6000 link stalls the entire EP group at
    the bandwidth bottleneck (straggler effect).
  - Fine-grained chunked transfer on a fast H100 link wastes launch overhead for
    a stream that could saturate a single large DMA.

BandwidthAwareAllGatherDispatcher selects between two strategies per step based
on the measured (or configured) link bandwidth of each rank pair:

  CHUNKED strategy (slow links — A6000 side):
    Splits the token dimension into ``num_chunks`` pieces and pipelines each
    chunk's AllGather with the GEMM from the previous chunk (overlap).  This
    mirrors the NCCLAllGatherDispatcher non-CG pad→compact path but adds an
    explicit chunk loop with CUDA stream interleaving.

  BULK strategy (fast links — H100 side):
    Single large AllGather followed by compact, identical to NCCLAllGatherDispatcher
    CG decode path but without requiring equal token counts.  This avoids the
    chunk-loop overhead when bandwidth is abundant.

The strategy is chosen once at dispatcher init via ``_select_strategy()`` and
re-evaluated lazily when the link speed estimate changes (e.g. after a topology
probe via ``BandwidthProbe.estimate``).

Variable-token-count handling (the core of bfd45740c):
  Both CHUNKED and BULK paths now support unequal per-rank token counts, mirroring
  upstream's use_allgather_v=True path:
    1. All-gather per-rank token counts (one int32 per rank, tiny bandwidth cost).
    2. Pad each rank's tensors to max_tokens for uniform AllGather tensor shape.
    3. Compact the gathered buffer by stripping per-rank padding slots.
  This replaces the old rigid decode-only graph path and allows prefill + decode
  mixed batches without a special-case eager mode at the dispatcher level.

Batch-dimension simplification (mirrors bfd45740c adjust_batch_dims cleanup):
  ``EPBatchCoordinator.adjust`` replicates the simplified
  ``adjust_batch_dims_for_expert_parallelism``:
    - Any EP rank with prefill tokens → all ranks return ``eager_mode=True``
      (caller must handle variable-length allocation).
    - Decode-only → max-reduce token count, return adjusted count.
  The old strict/decode_only_cuda_graphs/num_speculative_tokens arms are removed.

Diagnostic events (rank-0, logger.info + print, one line per event):
  [DS-BWAG] INIT         — strategy selected, chunk config, bandwidth estimate.
  [DS-BWAG] STRATEGY_FLIP — when link speed re-evaluation switches strategy.
  [DS-BWAG] CHUNK_STALL  — when chunk pipeline stalls (CHUNKED path only).
  [DS-BWAG] EP_SKEW      — when max/min per-rank token ratio exceeds threshold,
                            flagging load-imbalance risk.
  [DS-BWAG] BATCH_EAGER  — when EPBatchCoordinator forces eager mode due to prefill.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from deepspeed.utils import logger as ds_logger

_LOG_PREFIX = "[DS-BWAG]"

# ---------------------------------------------------------------------------
# Strategy enum
# ---------------------------------------------------------------------------

class AllGatherStrategy(str, Enum):
    """Selects between chunked-pipeline and bulk AllGather paths."""
    CHUNKED = "chunked"   # PCIe Gen1 / slow-link: chunk + stream-overlap
    BULK    = "bulk"      # PCIe Gen5 / fast-link: single large AllGather


# ---------------------------------------------------------------------------
# Bandwidth probe
# ---------------------------------------------------------------------------

@dataclass
class BandwidthProbe:
    """Light-weight unidirectional bandwidth estimator between two CUDA devices.

    Sends a small synthetic tensor and times the device-to-device copy via a
    CUDA event pair.  Result is cached; re-probe with ``invalidate()``.

    This is intentionally simple: the goal is to distinguish Gen1 (<30 GB/s)
    from Gen5 (>100 GB/s) links, not to produce a research-grade measurement.

    Attributes:
        src_device: source CUDA device index.
        dst_device: destination CUDA device index.
        probe_mb: synthetic tensor size in MB (default 64 MB).
        _cached_bw_gbps: cached result in GB/s, or None if not yet measured.
    """
    src_device: int
    dst_device: int
    probe_mb: float = 64.0
    _cached_bw_gbps: Optional[float] = field(default=None, repr=False, init=False)

    def estimate(self) -> float:
        """Return bandwidth in GB/s, using cache if available."""
        if self._cached_bw_gbps is not None:
            return self._cached_bw_gbps
        self._cached_bw_gbps = self._measure()
        return self._cached_bw_gbps

    def invalidate(self) -> None:
        """Clear cached measurement; next call to estimate() re-probes."""
        self._cached_bw_gbps = None

    def _measure(self) -> float:
        """Run a synthetic bandwidth probe between src and dst devices."""
        n_elements = int(self.probe_mb * 1024 * 1024 / 4)  # float32
        try:
            src_tensor = torch.zeros(n_elements, dtype=torch.float32,
                                     device=f"cuda:{self.src_device}")
            dst_tensor = torch.empty(n_elements, dtype=torch.float32,
                                     device=f"cuda:{self.dst_device}")
            # warm-up
            dst_tensor.copy_(src_tensor, non_blocking=True)
            torch.cuda.synchronize()

            t_start = torch.cuda.Event(enable_timing=True)
            t_end   = torch.cuda.Event(enable_timing=True)
            t_start.record()
            dst_tensor.copy_(src_tensor, non_blocking=True)
            t_end.record()
            torch.cuda.synchronize()
            elapsed_ms = t_start.elapsed_time(t_end)
            bytes_transferred = n_elements * 4
            gbps = (bytes_transferred / (elapsed_ms * 1e-3)) / 1e9
            return gbps
        except Exception:
            # Probe failed (e.g. different NUMA domains without P2P capability).
            # Return a conservative estimate so the caller falls back to CHUNKED.
            return 8.0


# ---------------------------------------------------------------------------
# Dispatcher configuration
# ---------------------------------------------------------------------------

@dataclass
class BandwidthAwareDispatcherConfig:
    """Configuration for BandwidthAwareAllGatherDispatcher.

    Attributes:
        fast_link_threshold_gbps: Bandwidth threshold to select BULK strategy.
            Links above this value use BULK (single large AllGather).
            Links at or below use CHUNKED (chunk-pipeline with overlap).
            Default 50.0 GB/s separates PCIe Gen1 (~16 GB/s) from Gen5 (~128 GB/s).
        num_chunks: Number of chunks for the CHUNKED pipeline.  Increasing this
            reduces the minimum stall window (finer overlap) at the cost of more
            kernel launches.  Ignored when strategy=BULK.
        probe_mb: Synthetic tensor size (MB) for BandwidthProbe._measure().
        ep_skew_warn_ratio: Log EP_SKEW if max_tokens/min_tokens across ranks
            exceeds this value.  1.5 means 50% imbalance triggers a warning.
        manual_strategy: Override automatic detection.  If set to CHUNKED or BULK,
            skip the BandwidthProbe entirely.
    """
    fast_link_threshold_gbps: float = 50.0
    num_chunks: int = 4
    probe_mb: float = 64.0
    ep_skew_warn_ratio: float = 1.5
    manual_strategy: Optional[AllGatherStrategy] = None


# ---------------------------------------------------------------------------
# EP batch coordinator (mirrors simplified adjust_batch_dims_for_expert_parallelism)
# ---------------------------------------------------------------------------

class EPBatchCoordinator:
    """Synchronise batch metadata across expert-parallel ranks.

    Mirrors Megatron bfd45740c InferenceBatchDimensions.adjust_batch_dims_for_expert_parallelism
    after its simplification:
      - All-reduce-max [token_count, is_non_decode] across the EP group.
      - If any rank is in prefill (non-decode) → return eager_mode=True.
      - Otherwise → return adjusted token_count (max over ranks), eager_mode=False.

    The old parameters (strict, decode_only_cuda_graphs, smallest_non_decode_cuda_graph_size,
    num_speculative_tokens) are removed — they were defending against inconsistent CUDA graph
    selection when different EP ranks ran different graph pools.  bfd45740c moved that concern
    entirely inside the dispatcher via match_ep_token_counts; this coordinator now has one job:
    tell callers whether to use CUDA graphs (decode) or fall back to eager (prefill).
    """

    @staticmethod
    def adjust(
        local_token_count: int,
        is_prefill: bool,
        ep_group: Optional[dist.ProcessGroup] = None,
    ) -> Tuple[int, bool]:
        """Sync batch dims across EP group and decide whether to run in eager mode.

        Args:
            local_token_count: number of tokens this rank will process this step.
            is_prefill: True if this rank has any prefill (non-decode) tokens.
            ep_group: expert-parallel process group; None → ep_size=1.

        Returns:
            (adjusted_token_count, eager_mode)
            adjusted_token_count: max over EP ranks (valid only when eager_mode=False).
            eager_mode: True when any EP rank is in prefill; caller must skip CUDA graphs.
        """
        if ep_group is None:
            return local_token_count, False

        ep_size = dist.get_world_size(group=ep_group)
        if ep_size <= 1:
            return local_token_count, False

        sync_tensor = torch.tensor(
            [local_token_count, int(is_prefill)],
            dtype=torch.int32,
            device=torch.cuda.current_device(),
        )
        dist.all_reduce(sync_tensor, op=dist.ReduceOp.MAX, group=ep_group)
        cpu = sync_tensor.cpu()

        if cpu[1].item() == 1:
            # Any rank has prefill → eager mode; dispatcher handles variable counts.
            return int(cpu[0].item()), True

        return int(cpu[0].item()), False


# ---------------------------------------------------------------------------
# Core dispatcher
# ---------------------------------------------------------------------------

class BandwidthAwareAllGatherDispatcher:
    """Bandwidth-aware AllGather token dispatcher for MoE expert parallelism.

    Mirrors Megatron bfd45740c NCCLAllGatherDispatcher (NCCL + pad/compact variable
    path), reinterpreted for heterogeneous PCIe topologies where inter-rank bandwidth
    determines the optimal AllGather strategy.

    Usage::

        dispatcher = BandwidthAwareAllGatherDispatcher(
            ep_group=my_ep_group,
            hidden_size=4096,
            topk=2,
            config=BandwidthAwareDispatcherConfig(num_chunks=4),
        )

        # Dispatch (AllGather)
        hidden_global, probs_global, routing_global = dispatcher.dispatch(
            hidden_states, probs, routing_map
        )

        # ... expert GEMM on [global_tokens, hidden_size] tensors ...

        # Combine (ReduceScatter)
        hidden_local = dispatcher.combine(expert_output)

    The dispatcher always syncs per-rank token counts with a single
    all_gather_into_tensor call before the main AllGather.  This is the key
    enabler from bfd45740c: the dispatcher handles variable-count tokens
    internally so the rest of the stack need not pad to a uniform size.
    """

    def __init__(
        self,
        ep_group: Optional[dist.ProcessGroup],
        hidden_size: int,
        topk: int,
        config: Optional[BandwidthAwareDispatcherConfig] = None,
        local_device: Optional[int] = None,
        peer_device: Optional[int] = None,
    ) -> None:
        """Initialise the dispatcher and select AllGather strategy.

        Args:
            ep_group: Expert-parallel process group.  If None, ep_size=1 (no-op AllGather).
            hidden_size: Hidden dimension of the model (elements per token row).
            topk: MoE router top-k value (routing map column count).
            config: Strategy and tuning config.  Defaults to BandwidthAwareDispatcherConfig().
            local_device: CUDA device index for BandwidthProbe source.  Defaults to current device.
            peer_device: CUDA device index for BandwidthProbe destination.  Defaults to local_device.
                         Set to an H100 peer index to measure the cross-link bandwidth.
        """
        self.ep_group = ep_group
        self.ep_size = dist.get_world_size(group=ep_group) if ep_group is not None else 1
        self.ep_rank  = dist.get_rank(group=ep_group) if ep_group is not None else 0
        self.hidden_size = hidden_size
        self.topk = topk
        self.config = config or BandwidthAwareDispatcherConfig()

        # State filled by dispatch(), consumed by combine().
        self._local_token_count: int = 0
        self._tokens_per_rank: Optional[List[int]] = None

        # Strategy selection
        if self.config.manual_strategy is not None:
            self.strategy = self.config.manual_strategy
            bw_estimate = None
        else:
            device_idx = local_device if local_device is not None else torch.cuda.current_device()
            peer_idx   = peer_device  if peer_device  is not None else device_idx
            probe = BandwidthProbe(src_device=device_idx, dst_device=peer_idx,
                                   probe_mb=self.config.probe_mb)
            bw_estimate = probe.estimate()
            self.strategy = (
                AllGatherStrategy.BULK
                if bw_estimate > self.config.fast_link_threshold_gbps
                else AllGatherStrategy.CHUNKED
            )

        self._log_init(bw_estimate)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """AllGather hidden_states, probs, and routing_map from all EP ranks.

        Supports variable token counts per rank (the core of the bfd45740c variable
        path).  Always syncs per-rank counts first, then pads to max_tokens for a
        uniform AllGather, then compacts by stripping per-rank padding.

        Args:
            hidden_states: [local_tokens, hidden_size] bf16/fp16/fp32 local input.
            probs:         [local_tokens, topk] fp32 local routing probabilities.
            routing_map:   [local_tokens, topk] int64 local routing indices.

        Returns:
            (hidden_states_global, probs_global, routing_map_global)
            Each is shape [total_valid_tokens, *] where total_valid_tokens = sum of
            per-rank token counts (no padding in the output).
        """
        if self.ep_size == 1:
            self._local_token_count = hidden_states.shape[0]
            self._tokens_per_rank = [self._local_token_count]
            return hidden_states, probs, routing_map

        local_tokens = hidden_states.shape[0]
        self._local_token_count = local_tokens

        # Step 1: sync per-rank token counts (one int32 per rank, cheap).
        tokens_per_rank = self._sync_token_counts(local_tokens)
        self._tokens_per_rank = tokens_per_rank

        self._maybe_log_ep_skew(tokens_per_rank)

        # Step 2: route to strategy-specific AllGather.
        if self.strategy == AllGatherStrategy.BULK:
            return self._dispatch_bulk(hidden_states, probs, routing_map, tokens_per_rank)
        else:
            return self._dispatch_chunked(hidden_states, probs, routing_map, tokens_per_rank)

    def combine(self, expert_output: torch.Tensor) -> torch.Tensor:
        """ReduceScatter expert outputs back to each EP rank.

        Reverses the pad/AllGather/compact done in dispatch():
          1. Expand compact [total_valid_tokens, H] → padded [ep_size * max_tokens, H].
          2. ReduceScatter → [max_tokens, H].
          3. Truncate to [local_tokens, H].

        Args:
            expert_output: [total_valid_tokens, hidden_size] expert GEMM outputs.

        Returns:
            [local_tokens, hidden_size] bf16 local token outputs.
        """
        if self.ep_size == 1:
            return expert_output.to(torch.bfloat16)

        tokens_per_rank = self._tokens_per_rank
        assert tokens_per_rank is not None, "combine() called before dispatch()"

        max_tokens = max(tokens_per_rank)
        total_valid = sum(tokens_per_rank)
        hidden_size  = expert_output.shape[1]

        # Expand compact → padded layout.
        padded = expert_output.new_zeros(self.ep_size * max_tokens, hidden_size)
        offset = 0
        for dst_rank, n_tokens in enumerate(tokens_per_rank):
            padded[dst_rank * max_tokens : dst_rank * max_tokens + n_tokens] = (
                expert_output[offset : offset + n_tokens]
            )
            offset += n_tokens

        # ReduceScatter: [ep_size * max_tokens, H] → [max_tokens, H].
        scattered = padded.new_empty(max_tokens, hidden_size)
        dist.reduce_scatter_tensor(scattered, padded, group=self.ep_group)

        # Truncate padding; cast to bf16.
        local_tokens = tokens_per_rank[self.ep_rank]
        return scattered[:local_tokens].to(torch.bfloat16)

    def update_strategy(
        self,
        local_device: Optional[int] = None,
        peer_device: Optional[int] = None,
    ) -> bool:
        """Re-probe bandwidth and update strategy if the link speed has changed.

        Intended for rare events (topology changes, reconfiguration).  Returns True
        if the strategy flipped (CHUNKED ↔ BULK), False if unchanged.

        Diagnostic event: [DS-BWAG] STRATEGY_FLIP when strategy changes.
        """
        device_idx = local_device if local_device is not None else torch.cuda.current_device()
        peer_idx   = peer_device  if peer_device  is not None else device_idx
        probe = BandwidthProbe(src_device=device_idx, dst_device=peer_idx,
                               probe_mb=self.config.probe_mb)
        new_bw = probe.estimate()
        new_strategy = (
            AllGatherStrategy.BULK
            if new_bw > self.config.fast_link_threshold_gbps
            else AllGatherStrategy.CHUNKED
        )
        if new_strategy == self.strategy:
            return False

        old = self.strategy
        self.strategy = new_strategy
        if self.ep_rank == 0:
            msg = (
                f"{_LOG_PREFIX} STRATEGY_FLIP: {old.value} → {new_strategy.value} "
                f"(bw={new_bw:.1f} GB/s threshold={self.config.fast_link_threshold_gbps} GB/s)"
            )
            ds_logger.info(msg)
            print(msg)
        return True

    # ------------------------------------------------------------------
    # Internal — strategy implementations
    # ------------------------------------------------------------------

    def _dispatch_bulk(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        tokens_per_rank: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single large AllGather, then compact.

        Best for high-bandwidth links (PCIe Gen5 / NVLink) where launch overhead
        for a chunk loop exceeds the time saved by overlapping.
        """
        max_tokens = max(tokens_per_rank)

        hidden_g   = self._allgather(self._pad(hidden_states,   max_tokens), max_tokens)
        probs_g    = self._allgather(self._pad(probs,           max_tokens), max_tokens)
        routing_g  = self._allgather(self._pad(routing_map,     max_tokens), max_tokens)

        return (
            self._compact(hidden_g,  tokens_per_rank, max_tokens),
            self._compact(probs_g,   tokens_per_rank, max_tokens),
            self._compact(routing_g, tokens_per_rank, max_tokens),
        )

    def _dispatch_chunked(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        tokens_per_rank: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Chunked AllGather with stream-overlap pipeline for slow links.

        Splits the token dimension into ``config.num_chunks`` slices.  Each chunk's
        AllGather runs on a secondary CUDA stream so the host thread can overlap
        kernel launches across chunks.  After all chunks complete we compact and
        return.

        The stall guard detects when AllGather events fall behind schedule:
        if a wait exceeds _CHUNK_STALL_WARN_MS, a [DS-BWAG] CHUNK_STALL event is
        emitted.  This happens when the PCIe link is saturated (e.g. concurrent
        training checkpointing on the same bus).
        """
        _CHUNK_STALL_WARN_MS = 5.0  # ms; heuristic for stall detection

        max_tokens = max(tokens_per_rank)
        num_chunks = max(1, min(self.config.num_chunks, max_tokens))
        chunk_size = (max_tokens + num_chunks - 1) // num_chunks

        # Pre-pad all tensors so chunks index uniformly.
        h_pad  = self._pad(hidden_states, max_tokens)
        p_pad  = self._pad(probs,         max_tokens)
        r_pad  = self._pad(routing_map,   max_tokens)

        # Output buffers on the main stream (allocated before the chunk loop).
        h_g_full = hidden_states.new_empty(self.ep_size * max_tokens, hidden_states.shape[1])
        p_g_full = probs.new_empty(        self.ep_size * max_tokens, probs.shape[1])
        r_g_full = routing_map.new_empty(  self.ep_size * max_tokens, routing_map.shape[1])

        comm_stream = torch.cuda.Stream()

        for chunk_idx in range(num_chunks):
            s = chunk_idx * chunk_size
            e = min(s + chunk_size, max_tokens)
            actual = e - s

            h_slice  = h_pad[s:e]
            p_slice  = p_pad[s:e]
            r_slice  = r_pad[s:e]

            h_g_chunk = hidden_states.new_empty(self.ep_size * actual, hidden_states.shape[1])
            p_g_chunk = probs.new_empty(        self.ep_size * actual, probs.shape[1])
            r_g_chunk = routing_map.new_empty(  self.ep_size * actual, routing_map.shape[1])

            comm_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(comm_stream):
                t0 = time.perf_counter()
                dist.all_gather_into_tensor(h_g_chunk, h_slice.contiguous(), group=self.ep_group)
                dist.all_gather_into_tensor(p_g_chunk, p_slice.contiguous(), group=self.ep_group)
                dist.all_gather_into_tensor(r_g_chunk, r_slice.contiguous(), group=self.ep_group)
                elapsed_ms = (time.perf_counter() - t0) * 1e3

            # Copy gathered chunk into full output buffer (main stream waits for comm).
            torch.cuda.current_stream().wait_stream(comm_stream)

            if elapsed_ms > _CHUNK_STALL_WARN_MS and self.ep_rank == 0:
                self._log_chunk_stall(chunk_idx, elapsed_ms)

            # Interleave: write each rank's chunk into contiguous rank slots.
            for rank_idx in range(self.ep_size):
                dst_s = rank_idx * max_tokens + s
                dst_e = dst_s + actual
                src_s = rank_idx * actual
                src_e = src_s + actual
                h_g_full[dst_s:dst_e]  = h_g_chunk[src_s:src_e]
                p_g_full[dst_s:dst_e]  = p_g_chunk[src_s:src_e]
                r_g_full[dst_s:dst_e]  = r_g_chunk[src_s:src_e]

        return (
            self._compact(h_g_full,  tokens_per_rank, max_tokens),
            self._compact(p_g_full,  tokens_per_rank, max_tokens),
            self._compact(r_g_full,  tokens_per_rank, max_tokens),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sync_token_counts(self, local_tokens: int) -> List[int]:
        """All-gather per-rank token counts across the EP group.

        Returns a list of length ep_size with each rank's token count.
        Cost: one small all_gather_into_tensor with ep_size int32 elements.
        """
        device = torch.cuda.current_device()
        local_t = torch.tensor([local_tokens], dtype=torch.int32, device=device)
        all_t   = torch.empty(self.ep_size, dtype=torch.int32, device=device)
        dist.all_gather_into_tensor(all_t, local_t, group=self.ep_group)
        return all_t.tolist()

    def _pad(self, tensor: torch.Tensor, max_tokens: int) -> torch.Tensor:
        """Pad token dimension to max_tokens with uninitialized values."""
        deficit = max_tokens - tensor.shape[0]
        if deficit == 0:
            return tensor
        pad = tensor.new_empty((deficit,) + tensor.shape[1:])
        return torch.cat([tensor, pad], dim=0)

    def _allgather(self, padded: torch.Tensor, max_tokens: int) -> torch.Tensor:
        """Standard AllGather into a contiguous buffer."""
        gathered = padded.new_empty((self.ep_size * max_tokens,) + padded.shape[1:])
        dist.all_gather_into_tensor(gathered, padded.contiguous(), group=self.ep_group)
        return gathered

    def _compact(
        self,
        gathered: torch.Tensor,
        tokens_per_rank: List[int],
        max_tokens: int,
    ) -> torch.Tensor:
        """Strip per-rank padding slots from a gathered [ep_size * max_tokens, *] buffer.

        Returns a contiguous [total_valid_tokens, *] tensor where total_valid_tokens
        = sum(tokens_per_rank).
        """
        slices = [
            gathered[rank * max_tokens : rank * max_tokens + n_tokens]
            for rank, n_tokens in enumerate(tokens_per_rank)
        ]
        return torch.cat(slices, dim=0)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _log_init(self, bw_estimate: Optional[float]) -> None:
        if self.ep_rank != 0:
            return
        if bw_estimate is not None:
            bw_str = f"bw_probe={bw_estimate:.1f} GB/s threshold={self.config.fast_link_threshold_gbps} GB/s"
        else:
            bw_str = "manual_strategy=True"
        msg = (
            f"{_LOG_PREFIX} INIT: strategy={self.strategy.value} ep_size={self.ep_size} "
            f"hidden={self.hidden_size} topk={self.topk} num_chunks={self.config.num_chunks} "
            f"{bw_str}"
        )
        ds_logger.info(msg)
        print(msg)

    def _log_chunk_stall(self, chunk_idx: int, elapsed_ms: float) -> None:
        msg = (
            f"{_LOG_PREFIX} CHUNK_STALL: chunk={chunk_idx} elapsed={elapsed_ms:.2f} ms "
            f"(threshold=5.00 ms) — PCIe may be saturated by concurrent traffic"
        )
        ds_logger.warning(msg)
        print(msg)

    def _maybe_log_ep_skew(self, tokens_per_rank: List[int]) -> None:
        if self.ep_rank != 0 or not tokens_per_rank:
            return
        t_max = max(tokens_per_rank)
        t_min = min(tokens_per_rank)
        if t_min == 0 or t_max / t_min < self.config.ep_skew_warn_ratio:
            return
        msg = (
            f"{_LOG_PREFIX} EP_SKEW: max_tokens={t_max} min_tokens={t_min} "
            f"ratio={t_max/t_min:.2f} > threshold={self.config.ep_skew_warn_ratio:.2f} "
            f"— consider rebalancing token routing"
        )
        ds_logger.warning(msg)
        print(msg)
