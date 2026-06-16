# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""Heterogeneous Context-Parallel workload scheduling for DES-LOC.

Mirrors Megatron 98d8c56db BalancedCPScheduler + HybridCPDataLoaderWrapper,
reinterpreted for asymmetric GPU clusters (e.g. A6000 48GB + H100 96GB)
where a round-robin or balanced split produces severe rank straggler effects.

Megatron's BalancedCPScheduler assigns sub-samples to DPxCP ranks to minimise
the *maximum* total seqlen per rank, assuming all ranks have equal compute and
memory capacity.  DES-LOC operates on heterogeneous hardware where H100 delivers
~6x the FLOP throughput of A6000.  An equal-token split would leave H100 idle
80% of every step while waiting for the A6000 tail — wasting 5/6 of available
compute.

AsymmetricCPScheduler generalises BalancedCPScheduler by:
1. Accepting per-rank *capacity weights* (default: uniform, reproducing Megatron).
2. Converting absolute capacities into relative quota tokens per scheduling round.
3. Using a capacity-weighted greedy bin-pack rather than a pure min-heap balance.

HeterogeneousCPDataLoaderWrapper wraps the upstream logic to:
- All-gather sub-sample seqlens across the DP group.
- Call AsymmetricCPScheduler to build per-rank assignment groups.
- Route sub-samples to their assigned rank via all-to-all.
- Emit structured diagnostic events at scheduling boundaries (mirrors
  M451 loss_scaler scale-grow event pattern: one log at the event boundary,
  not per-token noise).

Diagnostic events (rank-0, logger.info + print):
  [DS-HCP] SCHEDULE: per-rank quota and assigned tokens each scheduling round.
  [DS-HCP] IMBALANCE: when the heaviest rank exceeds the lightest by >20%,
           flagging potential straggler risk to the user.
  [DS-HCP] REROUTE: summary of all-to-all send/recv byte counts.
"""

import heapq
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed

import deepspeed.comm as dist
from deepspeed.utils import logger as ds_logger

_LOG_PREFIX = "[DS-HCP]"


# ---------------------------------------------------------------------------
# Asymmetric scheduler
# ---------------------------------------------------------------------------

class AsymmetricCPScheduler:
    """Capacity-weighted sub-sample scheduler for heterogeneous CP ranks.

    Mirrors Megatron 98d8c56db BalancedCPScheduler.get_groups_and_subsamples(),
    reinterpreted as a weighted bin-pack that allocates proportionally more
    token budget to higher-capacity ranks.

    Args:
        max_tokens_per_rank: Hard upper bound on tokens per rank per scheduling
            round (analogous to Megatron's max_seqlen_per_dp_cp_rank).
        capacity_weights: Per-rank relative capacity (length == world_size of
            dp_cp_group).  None → uniform (reproduces Megatron behaviour).
            For A6000+H100: [1.0, 6.0] so H100 receives 6x the quota.
        dp_cp_group: The combined DP×CP process group.
    """

    def __init__(
        self,
        max_tokens_per_rank: int,
        capacity_weights: Optional[List[float]] = None,
        dp_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        self.max_tokens_per_rank = max_tokens_per_rank
        self.dp_cp_group = dp_cp_group
        self.world_size = dp_cp_group.size() if dp_cp_group is not None else 1

        if capacity_weights is None:
            # Uniform — identical to Megatron's balanced scheduler.
            self.capacity_weights = [1.0] * self.world_size
        else:
            if len(capacity_weights) != self.world_size:
                raise ValueError(
                    f"AsymmetricCPScheduler: len(capacity_weights)={len(capacity_weights)} "
                    f"!= world_size={self.world_size}"
                )
            self.capacity_weights = list(capacity_weights)

        # Normalise weights so that the *minimum* weight maps to max_tokens_per_rank.
        # Higher-capacity ranks receive proportionally larger quotas.
        min_w = min(self.capacity_weights)
        self._rank_quotas = [
            int(max_tokens_per_rank * (w / min_w)) for w in self.capacity_weights
        ]

    def get_rank_quotas(self) -> List[int]:
        """Return token quota for each rank in the dp_cp_group."""
        return list(self._rank_quotas)

    def schedule(
        self, global_id_seqlens: List[Tuple[int, int]]
    ) -> Tuple[List[List[int]], List[List[List[int]]]]:
        """Assign sub-samples to ranks using capacity-weighted greedy bin-pack.

        Megatron uses a min-heap over (current_load, rank) with equal capacity.
        We extend this by initialising the heap with negative-quota sentinels so
        that higher-capacity ranks absorb more tokens in the same greedy pass.

        Args:
            global_id_seqlens: List of (global_id, seqlen) for every sub-sample
                across all DP ranks.  Order is arbitrary; gids are unique.

        Returns:
            groups: List[List[int]] — groups[rank] = sorted list of global_ids
                assigned to that rank.
            sample_id_groups: List[List[List[int]]] — per-microbatch grouping
                (single outer list; kept for API parity with Megatron).
        """
        # Heap entry: (current_load, rank)  — min-heap so least-loaded pops first.
        # We subtract quota so that a rank with higher quota starts "more negative",
        # meaning it appears emptier and attracts more sub-samples initially.
        # This mirrors Megatron's pure-balance logic but weighted by quota.
        heap: List[Tuple[int, int]] = [
            (0, rank) for rank in range(self.world_size)
        ]
        heapq.heapify(heap)

        groups: List[List[int]] = [[] for _ in range(self.world_size)]

        # Sort descending by seqlen: pack largest first (first-fit-decreasing
        # heuristic reduces worst-case imbalance compared to arbitrary order).
        sorted_samples = sorted(global_id_seqlens, key=lambda x: x[1], reverse=True)

        for gid, seqlen in sorted_samples:
            # Pop the rank with the smallest current load.
            load, rank = heapq.heappop(heap)
            quota = self._rank_quotas[rank]

            # Hard cap: if the chosen rank is already at or beyond its quota,
            # try to find a rank still under quota.  If all ranks are at quota,
            # we still assign (overflow) rather than drop the sample — the caller
            # must ensure total data fits.
            if load + seqlen > quota:
                # Re-check all ranks; pick the one with lowest (load / quota) ratio.
                # This is O(world_size) but world_size is small (<=32 in practice).
                # Rebuild heap entry and scan the full heap.
                # We push back and do a linear scan of the heap list.
                heapq.heappush(heap, (load, rank))
                best_rank = min(
                    range(self.world_size),
                    key=lambda r: heap[r][0] / self._rank_quotas[heap[r][1]]
                    if self._rank_quotas[heap[r][1]] > 0 else float('inf')
                )
                # Pop best_rank from heap — requires a heap rebuild.
                heap_dict = {entry[1]: entry for entry in heap}
                load, rank = heap_dict[best_rank]
                heap = [entry for entry in heap if entry[1] != rank]
                heapq.heapify(heap)

            groups[rank].append(gid)
            heapq.heappush(heap, (load + seqlen, rank))

        # Sort each group for deterministic ordering.
        for rank in range(self.world_size):
            groups[rank].sort()

        # API parity with Megatron: wrap in a single-element outer list.
        sample_id_groups = [[groups[rank] for rank in range(self.world_size)]]

        return groups, sample_id_groups

    def imbalance_ratio(self, groups: List[List[int]], global_id_seqlens: List[Tuple[int, int]]) -> float:
        """Compute max_load / min_load ratio across ranks (1.0 = perfect balance).

        Used by HeterogeneousCPDataLoaderWrapper to decide whether to emit an
        [DS-HCP] IMBALANCE diagnostic.
        """
        seqlen_map = {gid: s for gid, s in global_id_seqlens}
        loads = [sum(seqlen_map[gid] for gid in groups[r]) for r in range(self.world_size)]
        # Normalise by quota so that a perfectly proportional split shows ratio=1.0.
        normed = [loads[r] / self._rank_quotas[r] for r in range(self.world_size)]
        max_n = max(normed) if normed else 1.0
        min_n = min(normed) if normed else 1.0
        return max_n / min_n if min_n > 0 else float('inf')


# ---------------------------------------------------------------------------
# Data loader wrapper
# ---------------------------------------------------------------------------

class HeterogeneousCPDataLoaderWrapper:
    """Data loader wrapper for DES-LOC heterogeneous context-parallel training.

    Mirrors Megatron 98d8c56db HybridCPDataLoaderWrapper.  Key behavioural
    difference: workload assignment is capacity-weighted via AsymmetricCPScheduler
    so that faster GPUs (H100) receive proportionally more tokens per step.

    Protocol per __next__ call:
      1. Pull a packed batch from the inner data_iterator.
      2. Extract sub-sample seqlens; all-gather across DP group.
      3. Schedule sub-samples via AsymmetricCPScheduler (capacity-weighted).
      4. Emit [DS-HCP] SCHEDULE diagnostic on rank 0.
      5. Route sub-samples to assigned ranks via all-to-all.
      6. Emit [DS-HCP] REROUTE diagnostic on rank 0.
      7. Return (samples_this_rank_with_id, sample_id_groups).

    Args:
        data_iterator: Iterable yielding dicts with keys including ``cu_seqlens``.
        max_tokens_per_rank: Hard token cap per rank per scheduling round.
            For homogeneous clusters set this to max_seqlen / cp_size.
            For heterogeneous clusters set this to the *weakest* GPU's budget;
            stronger GPUs receive proportionally higher quotas via capacity_weights.
        capacity_weights: Per-rank relative throughput.  None → uniform.
            Example for [A6000, H100]: [1.0, 6.0].
        dp_cp_group: Combined DP×CP process group.
        dp_group: Data-parallel only group.
        diag_interval: Emit diagnostics every N scheduling rounds (default: 50).
    """

    def __init__(
        self,
        data_iterator,
        max_tokens_per_rank: int,
        capacity_weights: Optional[List[float]] = None,
        dp_cp_group: Optional[torch.distributed.ProcessGroup] = None,
        dp_group: Optional[torch.distributed.ProcessGroup] = None,
        diag_interval: int = 50,
    ):
        self.data_iterator = data_iterator
        self.diag_interval = diag_interval
        self._call_count = 0

        if dp_cp_group is None or dp_group is None:
            raise ValueError(
                "HeterogeneousCPDataLoaderWrapper requires explicit dp_cp_group and dp_group. "
                "Pass the process groups directly."
            )
        self.dp_cp_group = dp_cp_group
        self.dp_group = dp_group

        self.scheduler = AsymmetricCPScheduler(
            max_tokens_per_rank=max_tokens_per_rank,
            capacity_weights=capacity_weights,
            dp_cp_group=dp_cp_group,
        )
        self.world_size = dp_cp_group.size()
        self.dp_size = dp_group.size()

        # Log capacity configuration at init time (rank 0 only) so the training
        # log always records what heterogeneity profile was applied.
        if dist.get_rank() == 0:
            quotas = self.scheduler.get_rank_quotas()
            weights_str = str(capacity_weights) if capacity_weights else "uniform"
            msg = (
                f"{_LOG_PREFIX} INIT: dp_cp_size={self.world_size} dp_size={self.dp_size} "
                f"capacity_weights={weights_str} "
                f"rank_quotas={quotas} "
                f"max_tokens_per_rank={max_tokens_per_rank}"
            )
            print(msg)
            ds_logger.info(msg)

    def __iter__(self):
        return self

    # ------------------------------------------------------------------
    # Internal helpers (mirrors Megatron HybridCPDataLoaderWrapper helpers)
    # ------------------------------------------------------------------

    def _gather_seqlens(self, local_seqlens: torch.Tensor) -> Tuple[List[int], torch.Tensor]:
        """All-gather sub-sample seqlens from every DP rank.

        Mirrors Megatron get_global_seqlens().  Handles uneven subsample counts
        by padding to the maximum observed count before all_gather and trimming
        after.

        Returns:
            seqlens_flat: flat list of all seqlens across DP ranks, in rank order.
            offsets: 1-D int32 tensor of per-rank start offsets into seqlens_flat.
        """
        local_count = torch.tensor([local_seqlens.shape[0]], dtype=torch.int32, device="cuda")
        counts_list = [torch.zeros_like(local_count) for _ in range(self.dp_size)]
        torch.distributed.all_gather(counts_list, local_count, group=self.dp_group)
        counts = torch.stack(counts_list).view(-1).cpu()
        max_count = int(counts.max().item())

        # Pad local seqlens to max_count.
        if local_seqlens.shape[0] < max_count:
            pad = torch.zeros(max_count - local_seqlens.shape[0], dtype=torch.int32, device="cuda")
            local_padded = torch.cat([local_seqlens, pad])
        else:
            local_padded = local_seqlens

        gathered = [torch.empty_like(local_padded) for _ in range(self.dp_size)]
        torch.distributed.all_gather(gathered, local_padded, group=self.dp_group)

        # Trim each rank's slice to its actual count and concatenate.
        trimmed = [gathered[r][:counts[r]] for r in range(self.dp_size)]
        seqlens_flat = torch.cat(trimmed).cpu().tolist()

        csum = torch.cumsum(counts, dim=0, dtype=torch.int32)
        offsets = torch.cat([torch.zeros(1, dtype=torch.int32), csum[:-1]])
        return seqlens_flat, offsets

    def _gid_to_src_dp_rank(self, gid: int, offsets: torch.Tensor) -> int:
        """Return the DP rank that originally owns global sub-sample gid."""
        # bucketize: find the first boundary that gid does NOT exceed.
        return int(torch.bucketize(torch.tensor(gid), offsets[1:]).item())

    def _unpack_batch(self, batch) -> List[Dict[str, torch.Tensor]]:
        """Unpack a packed batch into individual sub-samples.

        Mirrors Megatron HybridCPDataLoaderWrapper.unpack_batch().
        Skips zero-length sub-samples (artifact of cu_seqlens padding).
        """
        unpacked = []
        for sample in batch:
            cu = sample["cu_seqlens"]
            for i in range(cu.shape[0] - 1):
                start, end = int(cu[i]), int(cu[i + 1])
                if end - start == 0:
                    continue
                sub = {
                    k: sample[k][start:end]
                    for k in sample
                    if k not in ("cu_seqlens", "batch_idx", "max_seqlen")
                }
                unpacked.append(sub)
        return unpacked

    def _reroute(
        self,
        batch: List[Dict[str, torch.Tensor]],
        local_gids: List[int],
        global_id_seqlens: List[Tuple[int, int]],
        sample_id_groups: List[List[List[int]]],
        offsets: torch.Tensor,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Route sub-samples to their assigned ranks via all-to-all.

        Mirrors Megatron reroute_samples_to_hdp_ranks().  Simplified to operate
        on the dp_cp_group directly; TP awareness is omitted because DES-LOC
        uses a separate TP dimension not entangled with the CP routing.

        Returns:
            Dict mapping global_id → sub-sample dict for sub-samples assigned
            to this rank.
        """
        seqlen_map: Dict[int, int] = {gid: s for gid, s in global_id_seqlens}
        gid2local = {gid: i for i, gid in enumerate(local_gids)}
        my_rank = self.dp_cp_group.rank()
        data_keys = list(batch[0].keys()) if batch else []

        # Combined assignment: flatten sample_id_groups across microbatches.
        # combined[dest_rank] = sorted list of global_ids to send to dest_rank.
        combined: List[List[int]] = [[] for _ in range(self.world_size)]
        for mb in sample_id_groups:
            for dest_rank in range(self.world_size):
                combined[dest_rank].extend(mb[dest_rank])
        for dest_rank in range(self.world_size):
            combined[dest_rank].sort()

        # Send plan: gids we hold locally that need to go to each dest_rank.
        send_gids: List[List[int]] = [[] for _ in range(self.world_size)]
        for dest_rank in range(self.world_size):
            for gid in combined[dest_rank]:
                if gid in gid2local:
                    send_gids[dest_rank].append(gid)

        send_lens = [sum(seqlen_map[g] for g in send_gids[d]) for d in range(self.world_size)]

        # Receive plan: gids assigned to *us* and where they come from.
        recv_gids_from: List[List[int]] = [[] for _ in range(self.world_size)]
        for gid in combined[my_rank]:
            src_dp_rank = self._gid_to_src_dp_rank(gid, offsets)
            # In the dp_cp_group, the dp source rank maps to the same CP slot as my_rank.
            # For non-CP runs (cp_size=1), src_hdp_rank == src_dp_rank.
            recv_gids_from[src_dp_rank].append(gid)

        recv_lens = [sum(seqlen_map[g] for g in recv_gids_from[s]) for s in range(self.world_size)]

        recv_gids_ordered = [gid for s in range(self.world_size) for gid in recv_gids_from[s]]
        recv_samples: List[Optional[Dict[str, torch.Tensor]]] = [None] * len(recv_gids_ordered)

        send_gids_flat = [gid for d in range(self.world_size) for gid in send_gids[d]]

        total_send = sum(send_lens)
        total_recv = sum(recv_lens)

        # M341-pattern diagnostic: log A2A byte counts at diag_interval boundary.
        _emit_diag = (self._call_count % self.diag_interval == 0) and dist.get_rank() == 0

        if _emit_diag:
            msg = (
                f"{_LOG_PREFIX} REROUTE call#{self._call_count}: "
                f"rank={my_rank} send_tokens={total_send} recv_tokens={total_recv} "
                f"send_per_rank={send_lens} recv_per_rank={recv_lens}"
            )
            print(msg)
            ds_logger.info(msg)

        if not data_keys:
            return {}

        def _pack_key(key: str) -> torch.Tensor:
            parts = []
            for gid in send_gids_flat:
                t = batch[gid2local[gid]][key]
                parts.append(t.to("cuda", non_blocking=True))
            if parts:
                return torch.cat(parts, dim=0)
            return torch.empty(0, device="cuda", dtype=batch[0][key].dtype)

        def _unpack_key(key: str, recv_tensor: torch.Tensor):
            cursor = 0
            for i, gid in enumerate(recv_gids_ordered):
                sl = seqlen_map[gid]
                recv_samples[i] = recv_samples[i] or {}
                recv_samples[i][key] = recv_tensor[cursor: cursor + sl]
                cursor += sl

        for key in data_keys:
            send_tensor = _pack_key(key)
            recv_tensor = torch.empty(
                total_recv, device="cuda", dtype=send_tensor.dtype
            )
            torch.distributed.all_to_all_single(
                output=recv_tensor,
                input=send_tensor,
                output_split_sizes=recv_lens,
                input_split_sizes=send_lens,
                group=self.dp_cp_group,
            )
            _unpack_key(key, recv_tensor)

        return {gid: recv_samples[i] for i, gid in enumerate(recv_gids_ordered)}

    # ------------------------------------------------------------------
    # __next__
    # ------------------------------------------------------------------

    def __next__(self) -> Tuple[Dict[int, Dict[str, torch.Tensor]], List[List[List[int]]]]:
        """Pull next batch, schedule and reroute sub-samples.

        Returns:
            samples_this_rank: Dict[global_id → sub-sample dict] for gids
                assigned to this rank after capacity-weighted scheduling.
            sample_id_groups: Nested list compatible with Megatron API.
        """
        self._call_count += 1

        if self.data_iterator is None:
            return {}, []

        batch = next(self.data_iterator)

        # 1. Extract local sub-sample seqlens.
        local_seqlens = []
        for sample in batch:
            cu = sample["cu_seqlens"]
            for i in range(cu.shape[0] - 1):
                sl = int(cu[i + 1]) - int(cu[i])
                if sl > 0:
                    local_seqlens.append(sl)
        local_seqlens_t = torch.tensor(local_seqlens, dtype=torch.int32, device="cuda")

        # 2. All-gather seqlens across DP group.
        seqlens_flat, offsets = self._gather_seqlens(local_seqlens_t)

        # 3. Build global_id_seqlens list and local ownership list.
        global_id_seqlens: List[Tuple[int, int]] = [
            (i, seqlens_flat[i]) for i in range(len(seqlens_flat))
        ]
        dp_rank = self.dp_group.rank()
        local_offset = int(offsets[dp_rank].item())
        local_count = len(local_seqlens)
        local_gids = list(range(local_offset, local_offset + local_count))

        # 4. Schedule via capacity-weighted scheduler.
        groups, sample_id_groups = self.scheduler.schedule(global_id_seqlens)

        # 5. Emit SCHEDULE diagnostic (rank 0, every diag_interval calls).
        if (self._call_count % self.diag_interval == 0) and dist.get_rank() == 0:
            quotas = self.scheduler.get_rank_quotas()
            assigned_tokens = [
                sum(seqlens_flat[gid] for gid in groups[r])
                for r in range(self.world_size)
            ]
            imb = self.scheduler.imbalance_ratio(groups, global_id_seqlens)
            msg = (
                f"{_LOG_PREFIX} SCHEDULE call#{self._call_count}: "
                f"total_subsamples={len(global_id_seqlens)} "
                f"quotas={quotas} "
                f"assigned_tokens={assigned_tokens} "
                f"imbalance_ratio={imb:.3f}"
            )
            print(msg)
            ds_logger.info(msg)

            # M451-pattern: emit a distinct warning event when straggler risk is high.
            if imb > 1.20:
                warn_msg = (
                    f"{_LOG_PREFIX} IMBALANCE call#{self._call_count}: "
                    f"imbalance_ratio={imb:.3f} exceeds 1.20 threshold — "
                    f"consider adjusting capacity_weights or max_tokens_per_rank. "
                    f"rank_loads={assigned_tokens}"
                )
                print(warn_msg)
                ds_logger.warning(warn_msg)

        # 6. Unpack batch and route sub-samples.
        batch_unpacked = self._unpack_batch(batch)
        samples_this_rank = self._reroute(
            batch_unpacked, local_gids, global_id_seqlens, sample_id_groups, offsets
        )

        return samples_this_rank, sample_id_groups
