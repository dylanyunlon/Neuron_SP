# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Hybrid Context Parallel schedule for DeepSpeed.

Ported from Megatron-LM/megatron/core/pipeline_parallel/hybrid_cp_schedule.py
(Megatron commit M3047 – Hybrid Context Parallel Feature).

The scheduler packs variable-length sub-samples across the DPxCP domain so
that every rank has a roughly balanced compute workload. It then runs
forward (and, optionally, backward) passes for all sub-samples in group
order, inserting a distributed barrier between groups to prevent
rank-level deadlocks when CP group membership changes between groups.
"""

from collections import deque
from functools import lru_cache
from math import ceil, log2
from typing import Callable, List, Optional, Tuple

import torch

try:
    from deepspeed.core import parallel_state
except ImportError:
    parallel_state = None


# ---------------------------------------------------------------------------
# A minimal stand-in for Megatron's RerunDataIterator.
# It wraps a plain Python iterator so the model's forward-step function
# receives an object it can call next() on repeatedly.
# ---------------------------------------------------------------------------
class _SimpleDataIterator:
    """Thin wrapper that turns any iterable into a re-usable data iterator."""

    def __init__(self, iterable):
        self._iter = iter(iterable)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)


# ---------------------------------------------------------------------------
# BalancedCPScheduler
# ---------------------------------------------------------------------------

class BalancedCPScheduler:
    """
    Forms groups of sub-samples so that all DPxCP ranks have a roughly
    balanced workload within each group.

    A *group* is a set of sub-samples that can be executed across the CP
    domain without a collective barrier in between.  Samples that require
    different CP sizes must belong to different groups (or at least be
    separated by a barrier) because changing CP membership mid-flight would
    dead-lock ranks that finish early.
    """

    def __init__(self, max_seq_len_per_rank: int, dp_cp_group: torch.distributed.ProcessGroup):
        self.max_seq_len_per_rank = max_seq_len_per_rank
        self.num_subsamples = 0
        self.num_subsamples_processed = 0
        self.free_resources = []
        self.total_hdp_gpus = dp_cp_group.size()

    # ------------------------------------------------------------------
    @lru_cache(maxsize=128)
    def get_total_workload(self, seq_length: int, cp_size: Optional[int] = None):
        """
        Estimate the relative compute cost of a sub-sample.

        Uses the O(seq^2 / cp_size) attention-dominates heuristic.
        """
        if cp_size is None:
            cp_size = self.gpus_needed(seq_length)
        return (seq_length * seq_length) / cp_size

    @lru_cache(maxsize=128)
    def gpus_needed(self, seq_len: int) -> int:
        """
        Number of CP ranks needed for *seq_len*, rounded up to the next
        power-of-two (to match available hybrid-CP group sizes).
        """
        return max(1, 2 ** ceil(log2(seq_len / self.max_seq_len_per_rank)))

    # ------------------------------------------------------------------
    def make_buckets_equal(
        self,
        sample_seqlens: List[Tuple[int, int]],
        compute_estimator: Callable[[int], float],
    ) -> List[deque]:
        """
        Partition *sample_seqlens* into k buckets of roughly equal total
        work, where k is the number of distinct CP sizes required.
        """
        seqlens = [seq_len for _, seq_len in sample_seqlens]
        k = len({self.gpus_needed(L) for L in seqlens})

        work_list = []
        for _, s in sample_seqlens:
            cp_size = self.gpus_needed(s)
            work_list.append(compute_estimator(s, cp_size))
        total_work = sum(work_list)
        target = total_work / k

        buckets, cur, cur_work = [], [], 0.0
        remaining_k = k

        for i, (sample_id, seq_len) in enumerate(sample_seqlens):
            work = compute_estimator(seq_len)
            projected = cur_work + work

            if cur and (
                projected > target * 1.1
                or len(sample_seqlens) - i <= remaining_k - len(buckets)
            ):
                buckets.append(deque(cur))
                cur, cur_work = [], 0.0
                remaining_k -= 1

            cur.append((sample_id, seq_len))
            cur_work += work

        if cur:
            buckets.append(deque(cur))

        return buckets

    # ------------------------------------------------------------------
    def next_hdp_group(
        self,
        sample_seqlens: List[Tuple[int, int]],
        compute_estimator: Callable[[int], float],
        total_gpus: int,
        delta: float = 0.05,
        strategy: str = "dp",
        eps_bucket: float = 0.10,
    ) -> Tuple[List[List[int]], List[Tuple[int, int]], List[float], List[List[int]]]:
        """
        Assign a balanced subset of *sample_seqlens* to GPUs and return
        the remaining unscheduled samples.

        Returns
        -------
        micro_batches       : per-GPU list of sequence lengths
        leftover            : unscheduled (sample_id, seq_len) tuples
        exec_times          : estimated cost per GPU
        sample_ids_per_gpu  : per-GPU list of sample IDs
        """
        if not sample_seqlens:
            return (
                [[] for _ in range(total_gpus)],
                [],
                [0.0 for _ in range(total_gpus)],
                [[] for _ in range(total_gpus)],
            )

        buckets = self.make_buckets_equal(sample_seqlens, compute_estimator)

        micro_batches      = [[] for _ in range(total_gpus)]
        exec_times         = [0.0 for _ in range(total_gpus)]
        sample_ids_per_gpu = [[] for _ in range(total_gpus)]

        gpu_group_id = [None] * total_gpus
        group_members: dict = {}
        group_size: dict    = {}
        next_gid = 0

        pp_cursor  = 0
        prev_needed = None
        check_balance = False

        while buckets:
            sample_seq_tuple = bucket_idx = None
            needed = None

            scan_order = (
                range(len(buckets))
                if strategy == "dp"
                else [(pp_cursor + i) % len(buckets) for i in range(len(buckets))]
            )

            for idx in scan_order:
                if not buckets[idx]:
                    continue
                cand_tuple   = buckets[idx][0]
                cand_seq_len = cand_tuple[1]
                needed       = self.gpus_needed(cand_seq_len)

                candidate_gids = [gid for gid, sz in group_size.items() if sz == needed]
                free_ranks     = [r for r, gid in enumerate(gpu_group_id) if gid is None]

                if candidate_gids or len(free_ranks) >= needed:
                    sample_seq_tuple, bucket_idx = cand_tuple, idx
                    break

            if sample_seq_tuple is None:
                break

            if strategy == "pp":
                pp_cursor = (bucket_idx + 1) % len(buckets)

            sample_id, seq_len = sample_seq_tuple
            needed = self.gpus_needed(seq_len)
            if prev_needed is None:
                prev_needed = needed

            candidate_gids = [gid for gid, sz in group_size.items() if sz == needed]
            if candidate_gids:
                best_gid, best_load = min(
                    (
                        (gid, max(exec_times[r] for r in group_members[gid]))
                        for gid in candidate_gids
                    ),
                    key=lambda t: t[1],
                )
            else:
                best_gid, best_load = None, float("inf")

            free_ranks = [r for r, gid in enumerate(gpu_group_id) if gid is None]
            if len(free_ranks) >= needed:
                free_sorted   = sorted(free_ranks, key=lambda r: exec_times[r])
                new_members   = free_sorted[:needed]
                new_load      = exec_times[new_members[-1]]

                if new_load < best_load:
                    best_gid       = None
                    chosen_members = new_members
                else:
                    chosen_members = group_members[best_gid]
            else:
                chosen_members = group_members[best_gid]

            if best_gid is None:
                best_gid = next_gid
                next_gid += 1
                group_members[best_gid] = chosen_members
                group_size[best_gid]    = needed
                for r in chosen_members:
                    gpu_group_id[r] = best_gid

            per_gpu_cost = compute_estimator(seq_len)
            for r in chosen_members:
                micro_batches[r].append(seq_len)
                exec_times[r]         += per_gpu_cost
                sample_ids_per_gpu[r].append(sample_id)

            buckets[bucket_idx].popleft()

            while buckets and not buckets[0]:
                buckets.pop(0)
                pp_cursor %= max(1, len(buckets))

            if needed < prev_needed:
                check_balance = True

            if (
                check_balance
                and buckets
                and max(exec_times) - min(exec_times) <= delta * max(exec_times)
            ):
                break

        # Gather leftovers
        leftovers = []
        for b in buckets:
            for item in b:
                leftovers.append(item)

        # ------------------------------------------------------------------
        def trim_overload():
            while True:
                cur_max   = max(exec_times)
                cur_min   = min(exec_times)
                cur_slack = cur_max - cur_min
                if cur_slack <= delta * cur_max:
                    break
                if cur_min == 0:
                    break

                max_r   = exec_times.index(cur_max)
                gid     = gpu_group_id[max_r]
                members = group_members[gid]

                if not micro_batches[max_r] or len(micro_batches[max_r]) <= 1:
                    break

                seq          = micro_batches[max_r][-1]
                per_gpu_cost = compute_estimator(seq)

                proj_times = exec_times[:]
                for r in members:
                    proj_times[r] -= per_gpu_cost

                proj_slack = max(proj_times) - min(proj_times)
                if proj_slack < cur_slack:
                    sample_id_to_remove = sample_ids_per_gpu[max_r][-1]
                    for r in members:
                        micro_batches[r].pop()
                        exec_times[r]          -= per_gpu_cost
                        sample_ids_per_gpu[r].pop()
                    leftovers.append((sample_id_to_remove, seq))
                else:
                    break

        trim_overload()

        total_work_before = sum(len(mb) for mb in micro_batches)

        # ------------------------------------------------------------------
        def fill_empty_gpus(micro_batches, exec_times, sample_ids_per_gpu, group_members, group_size):
            """
            Redistribute work to empty GPUs by doubling the CP size of the
            smallest existing group until no GPU is left idle.
            """
            empty_gpus = [i for i in range(total_gpus) if not micro_batches[i]]
            if not empty_gpus:
                return micro_batches, exec_times, sample_ids_per_gpu, group_members, group_size

            existing_group_sizes = set(group_size.values())
            assert existing_group_sizes, (
                "There should be at least one group; try increasing max_seq_len_per_rank."
            )

            min_group_size = min(existing_group_sizes)
            next_power     = min(min_group_size * 2, total_gpus)

            for gid, size in group_size.items():
                if size == min_group_size:
                    members       = group_members[gid]
                    needed_count  = next_power - min_group_size
                    group_start_gpu = members[0]
                    group_end_gpu   = members[-1]

                    empty_gpu = [idx for idx, work in enumerate(micro_batches) if not work][0]
                    assert not all(
                        work for work in micro_batches[empty_gpu: empty_gpu + needed_count]
                    ), "Empty GPUs detected but not enough to expand."

                    work_to_push         = micro_batches[group_end_gpu + 1: empty_gpu]
                    exec_times_to_push   = exec_times[group_end_gpu + 1: empty_gpu]
                    sample_ids_to_push   = sample_ids_per_gpu[group_end_gpu + 1: empty_gpu]

                    new_micro_batches      = [[]] * len(micro_batches)
                    new_exec_times         = [0.0] * len(exec_times)
                    new_sample_ids_per_gpu = [[]] * len(sample_ids_per_gpu)

                    for i in range(group_start_gpu):
                        new_micro_batches[i]      = micro_batches[i]
                        new_exec_times[i]         = exec_times[i]
                        new_sample_ids_per_gpu[i] = sample_ids_per_gpu[i]

                    for i in range(group_start_gpu, group_end_gpu + needed_count + 1):
                        new_micro_batches[i]      = micro_batches[group_end_gpu]
                        new_exec_times[i]         = self.get_total_workload(
                            micro_batches[group_end_gpu][0], next_power
                        )
                        new_sample_ids_per_gpu[i] = sample_ids_per_gpu[group_end_gpu]

                    for i, work in enumerate(work_to_push):
                        new_micro_batches[group_end_gpu + needed_count + 1 + i]      = work
                        new_exec_times[group_end_gpu + needed_count + 1 + i]         = exec_times_to_push[i]
                        new_sample_ids_per_gpu[group_end_gpu + needed_count + 1 + i] = sample_ids_to_push[i]

                    group_size[gid]    = next_power
                    group_members[gid] = list(range(members[0], members[-1] + needed_count + 1))
                    for pushed_gid in group_size:
                        if pushed_gid > gid:
                            group_members[pushed_gid] = [
                                x + needed_count for x in group_members[pushed_gid]
                            ]

                    return (
                        new_micro_batches,
                        new_exec_times,
                        new_sample_ids_per_gpu,
                        group_members,
                        group_size,
                    )

        empty_gpus = any(not micro_batches[i] for i in range(total_gpus))
        while empty_gpus:
            micro_batches, exec_times, sample_ids_per_gpu, group_members, group_size = fill_empty_gpus(
                micro_batches, exec_times, sample_ids_per_gpu, group_members, group_size
            )
            empty_gpus = any(not micro_batches[i] for i in range(total_gpus))

        total_work_after = sum(len(mb) for mb in micro_batches)
        assert total_work_after >= total_work_before, (
            f"Samples were removed during fill_empty_gpus: {total_work_before} -> {total_work_after}"
        )

        return micro_batches, leftovers, exec_times, sample_ids_per_gpu

    # ------------------------------------------------------------------
    def get_groups_and_subsamples(self, sample_id_seqlens, config):
        """
        Recursively form groups until all sub-samples are scheduled.

        Returns
        -------
        groups           : list of per-GPU seq-len lists, one entry per group
        sample_id_groups : list of per-GPU sample-ID lists, one entry per group
        """
        groups         = []
        sample_id_groups = []
        sample_id_seqlens = sorted(sample_id_seqlens, key=lambda x: x[1], reverse=True)

        while sample_id_seqlens:
            mb, sample_id_seqlens, exec_times, sample_ids = self.next_hdp_group(
                sample_id_seqlens, self.get_total_workload, self.total_hdp_gpus
            )
            groups.append(mb)
            if len(sample_ids) < self.total_hdp_gpus:
                sample_ids.extend([] * (self.total_hdp_gpus - len(sample_ids)))
            sample_id_groups.append(sample_ids)

        return groups, sample_id_groups


# ---------------------------------------------------------------------------
# Main entry-point called from schedules.py
# ---------------------------------------------------------------------------

def hybrid_context_parallel_forward_backward(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    output_tensor_grad,
    forward_data_store,
    config,
    collect_non_loss_data,
    first_val_step,
    forward_only,
    no_sync_func,
    total_num_tokens,
    check_first_val_step,
    model_type,
):
    """
    Scheduler for Hybrid Context Parallel (PP=1 path).

    Responsibilities
    ----------------
    1. Fetch one global batch from *data_iterator* (TP rank 0 only; others
       receive the schedule via broadcast).
    2. Determine how many groups and sub-samples each CP rank must execute
       by broadcasting the per-group sample counts from TP rank 0.
    3. Run forward (and optionally backward) passes sub-sample by sub-sample,
       inserting a distributed barrier between groups so that no rank starts
       the next group before all its partners have finished the current one.

    The last sub-sample of the last group is intentionally run *outside* the
    no_sync_func context so that the DDP/gradient reducer fires at the right
    moment for the optimizer step.

    Parameters mirror :func:`forward_backward_no_pipelining` in schedules.py.
    """
    from .schedules import backward_step, forward_step

    ps = parallel_state  # may be None in test environments

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _broadcast(item):
        """Broadcast a CUDA tensor from the TP-src rank within the TP group."""
        if item is not None:
            torch.distributed.broadcast(
                item,
                ps.get_tensor_model_parallel_src_rank(),
                group=ps.get_tensor_model_parallel_group(),
            )

    def _broadcast_num_samples_this_group(num_samples_this_group):
        """
        Broadcast the per-group sample-count vector from TP rank 0.

        First broadcasts the *length* of the vector (so non-TP-rank-0 ranks
        can allocate the buffer), then the vector itself.
        """
        dev = torch.cuda.current_device()
        torch.distributed.barrier()

        n = 0 if num_samples_this_group is None else int(num_samples_this_group.numel())
        n_tensor = torch.tensor([n], dtype=torch.int64, device=dev)
        _broadcast(n_tensor)
        n = int(n_tensor.item())

        assert n > 0, "Expected at least 1 sub-sample in the group broadcast."

        num_samples_broadcast = (
            torch.empty(n, dtype=torch.int32, device=dev)
            if num_samples_this_group is None
            else num_samples_this_group
        )
        _broadcast(num_samples_broadcast)
        return num_samples_broadcast

    def _get_new_data_iterator(sample_id_in_group, group_id):
        """
        Build a single-sample data iterator for the requested sub-sample.

        Only TP rank 0 has the actual data; other TP ranks return None so
        the model can rely on tensor-parallel broadcasts for the inputs.
        """
        if is_first_tp_rank:
            sub_sample_id = sample_ids_this_group[sample_id_in_group]
            sample        = batch[sub_sample_id]
            partner_cp_size = len(
                [
                    True
                    for sample_ids in sample_id_groups[group_id]
                    if sub_sample_id in sample_ids
                ]
            )
            sample["local_cp_size"] = torch.tensor(partner_cp_size, dtype=torch.int32)
            return _SimpleDataIterator([sample])
        else:
            return None

    # ------------------------------------------------------------------
    # Fetch data (TP rank 0 only)
    # ------------------------------------------------------------------
    hdp_rank         = ps.get_data_parallel_rank(with_context_parallel=True)
    is_first_tp_rank = ps.get_tensor_model_parallel_rank() == 0

    if is_first_tp_rank:
        data             = next(data_iterator)
        sample_id_groups = data[1]   # list[list[list[int]]] – groups x gpus x sample_ids
        batch            = data[0]   # dict / list of per-sample tensors
    else:
        data, sample_id_groups, batch = None, None, None

    # Build the per-group sample-count vector on TP rank 0, broadcast to all.
    num_samples_this_group = None
    if is_first_tp_rank:
        num_samples_this_group = torch.tensor(
            [len(group[hdp_rank]) for group in sample_id_groups],
            dtype=torch.int32,
            device="cuda",
        )

    num_samples_this_group = _broadcast_num_samples_this_group(num_samples_this_group)
    num_samples_this_group = num_samples_this_group.cpu().numpy()
    num_total_groups       = num_samples_this_group.shape[0]

    current_microbatch = 0

    # ------------------------------------------------------------------
    # Groups 0 … N-2  (inside no_sync_func to suppress grad reductions)
    # ------------------------------------------------------------------
    with no_sync_func():
        for j in range(num_total_groups - 1):
            sample_ids_this_group = sample_id_groups[j][hdp_rank] if is_first_tp_rank else None

            for i in range(num_samples_this_group[j]):
                new_data_iterator = _get_new_data_iterator(i, j)
                output_tensor, num_tokens = forward_step(
                    forward_step_func,
                    new_data_iterator,
                    model,
                    num_microbatches,
                    input_tensor,
                    forward_data_store,
                    config,
                    collect_non_loss_data=collect_non_loss_data,
                    is_first_microbatch=check_first_val_step(
                        first_val_step, forward_only, current_microbatch == 0
                    ),
                    current_microbatch=current_microbatch,
                )
                current_microbatch  += 1
                total_num_tokens    += num_tokens.item()

                if not forward_only:
                    backward_step(input_tensor, output_tensor, output_tensor_grad, config)

            # Barrier between groups: all DPxCP partners must reach this
            # point before any rank advances to the next group.
            torch.distributed.barrier(
                ps.get_data_parallel_group(with_context_parallel=True)
            )

    # ------------------------------------------------------------------
    # Last group, all but the last sub-sample  (still inside no_sync_func)
    # ------------------------------------------------------------------
    with no_sync_func():
        sample_ids_this_group = (
            sample_id_groups[-1][hdp_rank] if is_first_tp_rank else None
        )

        for i in range(num_samples_this_group[-1] - 1):
            new_data_iterator = _get_new_data_iterator(i, -1)
            output_tensor, num_tokens = forward_step(
                forward_step_func,
                new_data_iterator,
                model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                collect_non_loss_data=collect_non_loss_data,
                is_first_microbatch=check_first_val_step(
                    first_val_step, forward_only, current_microbatch == 0
                ),
                current_microbatch=current_microbatch,
            )
            current_microbatch  += 1
            total_num_tokens    += num_tokens.item()

            if not forward_only:
                backward_step(input_tensor, output_tensor, output_tensor_grad, config)

    # ------------------------------------------------------------------
    # Very last sub-sample of the very last group  (outside no_sync_func
    # so that DDP grad-reduction fires).
    # ------------------------------------------------------------------
    new_data_iterator = _get_new_data_iterator(-1, -1)
    output_tensor, num_tokens = forward_step(
        forward_step_func,
        new_data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        collect_non_loss_data=collect_non_loss_data,
        is_first_microbatch=check_first_val_step(
            first_val_step, forward_only, current_microbatch == 0
        ),
        current_microbatch=current_microbatch,
    )
    total_num_tokens += num_tokens.item()

    if not forward_only:
        backward_step(input_tensor, output_tensor, output_tensor_grad, config)

    return forward_data_store, total_num_tokens
