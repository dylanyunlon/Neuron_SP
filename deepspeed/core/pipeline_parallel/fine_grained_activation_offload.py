# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
#
# Ported from:
#   Megatron-LM/megatron/core/pipeline_parallel/fine_grained_activation_offload.py
#   Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# DES-LOC adaptation:
#   - Removed Megatron-specific imports (megatron.core.extensions.transformer_engine,
#     megatron.core.utils, megatron.core.transformer.cuda_graphs).
#   - Replaced TransformerEngine CPU offload hooks with pure PyTorch saved-tensor
#     default hooks via torch._C._autograd._push_saved_tensors_default_hooks.
#   - Added tier-aware helpers: offload_required_for_tier() and
#     maybe_enable_activation_offload() (see core_adapters.py adapter #6).
#   - A6000/PROFESSIONAL tier (≤49 GB VRAM) enables offload automatically.
#     H100/DATACENTER tier (≥80 GB VRAM) skips offload for maximum throughput.

from __future__ import annotations

from collections import defaultdict, deque
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------
DEBUG = False
DEBUG_RANK = 0


def _debug_rank(message: str) -> None:
    """Print *message* on DEBUG_RANK when DEBUG is enabled."""
    if not DEBUG:
        return
    if not torch.distributed.is_initialized():
        print(message)
        return
    if torch.distributed.get_rank() == DEBUG_RANK:
        print(message)  # noqa: T201


# ---------------------------------------------------------------------------
# Tier-aware helpers (DES-LOC extension)
# ---------------------------------------------------------------------------

# VRAM threshold (GB) below which activation offload is considered necessary.
# A6000 has ~48-49 GB; H100/A100 have ≥80 GB.
_OFFLOAD_VRAM_THRESHOLD_GB: float = 60.0


def offload_required_for_tier(tier_type) -> bool:
    """Return True if *tier_type* indicates a memory-constrained GPU.

    Decision table (DES-LOC spec):
        PROFESSIONAL (A6000, RTX PRO 6000 Blackwell, ~49 GB) → offload = True
        CONSUMER  (RTX 4090, 3090, ≤24 GB)                   → offload = True
        DATACENTER (H100, A100, ≥80 GB)                       → offload = False

    When *tier_type* is None or unknown, falls back to querying
    ``torch.cuda.get_device_properties`` for the current device.
    """
    if tier_type is not None:
        try:
            from deepspeed.core.desloc_config import TierType
            if tier_type == TierType.DATACENTER:
                return False
            # PROFESSIONAL and CONSUMER both benefit from offload
            return True
        except ImportError:
            pass  # desloc_config not available; fall through to VRAM check

    # Fallback: inspect VRAM directly
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            vram_gb = props.total_memory / (1024 ** 3)
            return vram_gb < _OFFLOAD_VRAM_THRESHOLD_GB
        except Exception:
            pass

    # Conservative default: enable offload if tier unknown
    return True


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_offload_summary_table(total_offload_bytes: Dict[str, int]) -> None:
    """Print an ASCII table summarising offload bytes across all ranks.

    Gathers offload data from all ranks and prints a formatted table on rank 0,
    with rows representing ranks and columns representing groups.
    """
    if not torch.distributed.is_initialized():
        return

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # Gather all group names across ranks
    local_names = list(total_offload_bytes.keys())
    all_names_list: List[Optional[List[str]]] = [None] * world_size
    torch.distributed.all_gather_object(all_names_list, local_names)
    all_group_names = sorted(
        set(name for names in all_names_list if names is not None for name in names)
    )

    # Gather offload bytes from all ranks
    local_bytes = [total_offload_bytes.get(name, 0) for name in all_group_names]
    all_bytes_list: List[Optional[List[int]]] = [None] * world_size
    torch.distributed.all_gather_object(all_bytes_list, local_bytes)

    if rank == 0:
        col_width = max(12, max((len(n) for n in all_group_names), default=8) + 2)
        rank_col_width = max(6, len(f"Rank {world_size - 1}") + 2)

        header = "Rank".ljust(rank_col_width)
        header += "".join(n.rjust(col_width) for n in all_group_names)
        header += "Total".rjust(col_width)
        sep = "-" * len(header)

        print("\n" + "=" * len(header))  # noqa: T201
        print("Activation Offload Summary (MB)".center(len(header)))  # noqa: T201
        print("=" * len(header))  # noqa: T201
        print(header)  # noqa: T201
        print(sep)  # noqa: T201

        grand_total = 0
        col_totals = [0] * len(all_group_names)
        for r in range(world_size):
            row_bytes = all_bytes_list[r] or []
            row_total = sum(row_bytes)
            grand_total += row_total
            for i, b in enumerate(row_bytes):
                col_totals[i] += b
            row_str = f"Rank {r}".ljust(rank_col_width)
            for b in row_bytes:
                row_str += f"{b / (1024 * 1024):.2f}".rjust(col_width)
            row_str += f"{row_total / (1024 * 1024):.2f}".rjust(col_width)
            print(row_str)  # noqa: T201

        print(sep)  # noqa: T201
        totals_row = "Total".ljust(rank_col_width)
        for ct in col_totals:
            totals_row += f"{ct / (1024 * 1024):.2f}".rjust(col_width)
        totals_row += f"{grand_total / (1024 * 1024):.2f}".rjust(col_width)
        print(totals_row)  # noqa: T201
        print("=" * len(header) + "\n")  # noqa: T201

    torch.distributed.barrier()


# ---------------------------------------------------------------------------
# GPU (CPU-side pinned) tensor pool
# ---------------------------------------------------------------------------

class GPUTensorPool:
    """Memory pool for efficient allocation / deallocation of pinned CPU tensors.

    Supports multiple (shape, dtype) combinations, each with its own sub-pool.
    Reuses buffers across iterations for zero-copy offload performance.
    """

    def __init__(self, device: str = "cpu", pin_memory: bool = True):
        self.device = torch.device(device)
        self.pin_memory = pin_memory
        # {(shape, dtype): {"free": deque, "all": list, "allocated_count": int}}
        self._pools: Dict[Tuple, Dict[str, Any]] = {}
        self._stats = {
            "total_allocated": 0,
            "current_in_use": 0,
            "allocation_requests": 0,
            "free_requests": 0,
            "pool_hits": 0,
            "pool_misses": 0,
        }

    # ------------------------------------------------------------------
    def _pool_key(self, shape: Tuple, dtype: torch.dtype) -> Tuple:
        return (shape, dtype)

    @staticmethod
    def _byte_size(shape: Tuple, dtype: torch.dtype) -> int:
        return torch.tensor([], dtype=dtype).element_size() * (
            1 if not shape else __import__("math").prod(shape)
        )

    # ------------------------------------------------------------------
    def allocate(self, shape: Tuple, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        self._stats["allocation_requests"] += 1
        key = self._pool_key(shape, dtype)
        if key not in self._pools:
            self._pools[key] = {"free": deque(), "all": [], "allocated_count": 0}
        pool = self._pools[key]

        if pool["free"]:
            tensor = pool["free"].popleft()
            self._stats["pool_hits"] += 1
        else:
            tensor = torch.empty(
                shape, dtype=dtype, device=self.device, pin_memory=self.pin_memory
            )
            pool["all"].append(tensor)
            self._stats["total_allocated"] += 1
            self._stats["pool_misses"] += 1

        pool["allocated_count"] += 1
        self._stats["current_in_use"] += 1
        return tensor

    def free(self, tensor: torch.Tensor) -> None:
        self._stats["free_requests"] += 1
        key = self._pool_key(tensor.shape, tensor.dtype)
        if key not in self._pools:
            raise ValueError(
                f"No pool for shape={tensor.shape}, dtype={tensor.dtype}"
            )
        pool = self._pools[key]
        if not any(tensor is t for t in pool["all"]):
            raise ValueError("Tensor does not belong to this pool")
        pool["free"].append(tensor)
        pool["allocated_count"] -= 1
        self._stats["current_in_use"] -= 1

    def reset(self) -> None:
        """Mark all tensors as available (called at iteration boundary)."""
        for pool in self._pools.values():
            pool["free"].clear()
            for t in pool["all"]:
                pool["free"].append(t)
            pool["allocated_count"] = 0
        self._stats["current_in_use"] = 0

    def clear(self) -> None:
        """Release all GPU/CPU memory held by the pool."""
        for pool in self._pools.values():
            pool["free"].clear()
            pool["all"].clear()
        self._pools.clear()
        self._stats["current_in_use"] = 0

    def __del__(self) -> None:
        self.clear()


# ---------------------------------------------------------------------------
# Offload tensor group
# ---------------------------------------------------------------------------

class OffloadTensorGroup:
    """A named collection of tensors to be offloaded/reloaded together."""

    # Groups whose tensor shapes cannot be known ahead of time → skip CPU pool
    _NO_POOL_GROUPS = frozenset({"expert_fc1", "moe_act"})

    def __init__(self, name: str):
        self._name = name
        self._tensors: Dict[Tuple, Any] = {}
        self._offload_event = torch.cuda.Event()
        self._reload_event = torch.cuda.Event()
        self.offload: bool = True
        self.total_offload_bytes: int = 0
        self.total_tensor_count: int = 0
        self.use_cpu_pool: bool = name not in self._NO_POOL_GROUPS

    # ------------------------------------------------------------------
    def push_tensor(self, tag: Tuple, tensor: Any) -> None:
        self._tensors[tag] = tensor

    def pop_tensor(self, tag: Tuple) -> Any:
        return self._tensors.pop(tag)

    def record_offload_event(self, stream: torch.cuda.Stream) -> None:
        self._offload_event.record(stream)

    def wait_offload_event(self, stream: torch.cuda.Stream) -> None:
        stream.wait_event(self._offload_event)

    def record_reload_event(self, stream: torch.cuda.Stream) -> None:
        self._reload_event.record(stream)

    def wait_reload_event(self, stream: torch.cuda.Stream) -> None:
        stream.wait_event(self._reload_event)

    def update_offload_info(self, tensor: torch.Tensor) -> None:
        self.total_offload_bytes += tensor.numel() * tensor.element_size()
        self.total_tensor_count += 1


# ---------------------------------------------------------------------------
# Per-chunk offload handler
# ---------------------------------------------------------------------------

class ChunkOffloadHandler:
    """Handles activation offload/reload for a single pipeline micro-batch chunk.

    Core API (called by autograd hooks inside PipelineOffloadManager):
        tensor_push(tensor) → tag        # called in forward on_save_for_backward
        tensor_pop(tag)     → tensor     # called in backward on_get_saved_tensor

    Group lifecycle:
        on_group_start_forward(name)     # at the start of each transformer layer
        on_group_commit_forward(forced)  # at the end   of each transformer layer
        on_group_start_backward()        # triggers async H2D reload
        on_group_commit_backward(name)   # waits for reload, syncs compute stream
    """

    # ------------------------------------------------------------------
    # Low-level D2H / H2D helpers (offload_activations_to_cpu /
    # reload_activations_from_cpu in DES-LOC terminology)
    # ------------------------------------------------------------------

    def offload_activations_to_cpu(
        self,
        src_tensor: torch.Tensor,
        pin_memory: bool = True,
        use_cpu_pool: bool = True,
    ) -> Tuple:
        """Copy *src_tensor* to pinned CPU memory asynchronously.

        Returns a state tuple ``(device, cpu_backup, use_cpu_pool)`` that
        ``reload_activations_from_cpu`` can later convert back to a GPU tensor.
        """
        _debug_rank("offload_activations_to_cpu")
        if not src_tensor.is_contiguous():
            src_tensor = src_tensor.contiguous()

        if use_cpu_pool:
            cpu_backup = self.cpu_tensor_pool.allocate(src_tensor.shape, dtype=src_tensor.dtype)
        else:
            cpu_backup = torch.empty(
                src_tensor.shape,
                dtype=src_tensor.dtype,
                device="cpu",
                pin_memory=pin_memory,
            )

        cpu_backup.copy_(src_tensor, non_blocking=pin_memory)
        return (src_tensor.device, cpu_backup, use_cpu_pool)

    def reload_activations_from_cpu(
        self,
        state: Tuple,
        non_blocking: Optional[bool] = None,
    ) -> torch.Tensor:
        """Reload a previously offloaded activation from CPU back to GPU.

        *state* must be the tuple returned by ``offload_activations_to_cpu``.
        """
        _debug_rank("reload_activations_from_cpu")
        dev, cpu_backup, use_cpu_pool = state
        if non_blocking is None:
            non_blocking = cpu_backup.is_pinned()
        gpu_tensor = torch.empty(
            cpu_backup.size(),
            dtype=cpu_backup.dtype,
            layout=cpu_backup.layout,
            device=dev,
        )
        gpu_tensor.copy_(cpu_backup, non_blocking=non_blocking)
        if use_cpu_pool:
            self.cpu_tensor_pool.free(cpu_backup)
        return gpu_tensor

    # Keep Megatron-compatible aliases so existing call sites compile unchanged.
    offload = offload_activations_to_cpu
    reload = reload_activations_from_cpu

    # ------------------------------------------------------------------
    def __init__(
        self,
        min_offloaded_tensor_size: int,
        cpu_tensor_pool: GPUTensorPool,
        max_inflight_offloads: Optional[int] = None,
    ):
        self.do_offload: bool = True
        self.offload_groups: List[OffloadTensorGroup] = []
        self._offloaded_group_index: int = 0
        self._groups_to_offload: List[OffloadTensorGroup] = []
        self._groups_to_reload: List[OffloadTensorGroup] = []
        self._tensor_count_current_group: int = 0
        self._max_group_size: int = 0
        self._reloading_group: List[OffloadTensorGroup] = []
        self.torch_tensor_count: int = 0
        self.d2h_stream: torch.cuda.Stream = PipelineOffloadManager.get_instance().d2h_stream
        self.h2d_stream: torch.cuda.Stream = PipelineOffloadManager.get_instance().h2d_stream
        self.min_offloaded_tensor_size = min_offloaded_tensor_size
        self.cpu_tensor_pool = cpu_tensor_pool
        self.is_warmup: bool = True
        self._max_inflight_offloads = max_inflight_offloads
        self._offload_pending_by_name: Dict[str, deque] = defaultdict(deque)
        # vpp_rank assigned by PipelineOffloadManager
        self.vpp_rank: int = 0

    def reset(self) -> None:
        self._offloaded_group_index = 0
        self._groups_to_offload = []
        self._groups_to_reload = []
        self._tensor_count_current_group = 0
        self._reloading_group = []
        self._offload_pending_by_name.clear()

    # ------------------------------------------------------------------
    def find_group_with_name(self, name: str, start_index: int = 0) -> Optional[OffloadTensorGroup]:
        return next(
            (g for g in self.offload_groups[start_index:] if g._name == name), None
        )

    def is_empty_chunk(self, name: Optional[str] = None) -> bool:
        if name is not None:
            return self.find_group_with_name(name) is None
        return self._max_group_size == 0

    def finish_all_groups(self, name: Optional[str] = None) -> bool:
        if (
            not self._groups_to_reload
            and not self._groups_to_offload
            and self._offloaded_group_index > 0
        ):
            return True
        assert name is not None, "Name required"
        return self.find_group_with_name(name, self._offloaded_group_index) is None

    def find_next_group(self, name: str) -> Optional[OffloadTensorGroup]:
        return self.find_group_with_name(name, self._offloaded_group_index)

    # ------------------------------------------------------------------
    def tensor_need_offloading_checker(self, tensor: torch.Tensor) -> bool:
        if tensor.numel() < self.min_offloaded_tensor_size:
            return False
        if hasattr(tensor, "offloading_activation") and not tensor.offloading_activation:
            return False
        return True

    # ------------------------------------------------------------------
    def tensor_push(self, tensor: torch.Tensor) -> Tuple[int, int]:
        """Save *tensor* for backward; returns a (group_id, position) tag."""
        assert not isinstance(
            tensor,
            (
                torch._subclasses.fake_tensor.FakeTensor,
                torch._subclasses.functional_tensor.FunctionalTensor,
            ),
        ), "Stray tensors must not be offloaded"
        tag = (self._offloaded_group_index, self._tensor_count_current_group)
        self._tensor_count_current_group += 1
        self.offload_groups[self._offloaded_group_index - 1].push_tensor(tag, tensor)
        return tag

    def tensor_pop(self, tensor_tag: Tuple[int, int]) -> torch.Tensor:
        """Retrieve (and if needed reload) the tensor identified by *tensor_tag*."""
        group_id, _ = tensor_tag
        tensor = self.offload_groups[group_id - 1].pop_tensor(tensor_tag)
        if isinstance(tensor, tuple):
            tensor = self.reload_activations_from_cpu(tensor)
        return tensor

    # ------------------------------------------------------------------
    def bulk_offload_group(self) -> None:
        """Async D2H copy of all tensors in the current offload group."""
        group = self._groups_to_offload[-1]
        with torch.cuda.stream(self.d2h_stream):
            for tag, t in group._tensors.items():
                if self.tensor_need_offloading_checker(t):
                    state = self.offload_activations_to_cpu(
                        t, use_cpu_pool=group.use_cpu_pool
                    )
                    if self.is_warmup:
                        group.update_offload_info(t)
                    t.record_stream(self.d2h_stream)
                    group.push_tensor(tag, state)
            group.record_offload_event(self.d2h_stream)
        self._groups_to_offload.pop()

        # Optional inflight cap per group name
        if self._max_inflight_offloads is not None:
            q = self._offload_pending_by_name[group._name]
            q.append(group._offload_event)
            self._drain_offload_pending(group._name)

    def _drain_offload_pending(self, group_name: str) -> None:
        if self._max_inflight_offloads is None:
            return
        cur = torch.cuda.current_stream()
        q = self._offload_pending_by_name[group_name]
        while len(q) > self._max_inflight_offloads:
            cur.wait_event(q.popleft())

    def bulk_reload_group(self) -> None:
        """Async H2D copy of all tensors in the current reload group."""
        group = self._groups_to_reload[-1]
        with torch.cuda.stream(self.h2d_stream):
            group.wait_offload_event(self.h2d_stream)
            for tag, state in group._tensors.items():
                if isinstance(state, tuple):
                    gpu_t = self.reload_activations_from_cpu(state)
                    group.push_tensor(tag, gpu_t)
            group.record_reload_event(self.h2d_stream)
        self._groups_to_reload.pop()
        self._reloading_group.append(group)

    # ------------------------------------------------------------------
    def should_bulk_offload(self) -> bool:
        assert self._groups_to_offload, "No groups queued for offload"
        group = self._groups_to_offload[-1]
        if self.is_warmup:
            return True
        if not group.offload:
            return False
        mgr = PipelineOffloadManager.get_instance()
        next_bwd = mgr.front_backward_chunk(group._name)
        if next_bwd is not None and next_bwd is self:
            if self.find_next_group(group._name) is None:
                return False
        return True

    def bulk_offload(self, forced_released_tensors: List[torch.Tensor]) -> None:
        if self.should_bulk_offload():
            self._groups_to_reload.append(self._groups_to_offload[-1])
            self.bulk_offload_group()
            if forced_released_tensors:
                cur = torch.cuda.current_stream()
                for t in forced_released_tensors:
                    if self.tensor_need_offloading_checker(t):
                        t.record_stream(cur)
                        t.untyped_storage().resize_(0)

    def bulk_reload(self) -> None:
        if self._groups_to_reload:
            self.bulk_reload_group()
        else:
            mgr = PipelineOffloadManager.get_instance()
            next_bwd = mgr.front_backward_chunk()
            if (
                next_bwd is not None
                and next_bwd._offloaded_group_index == next_bwd._max_group_size
            ):
                next_bwd.pre_reload_last_layer()

    def pre_reload_last_layer(self) -> None:
        if self._groups_to_reload:
            self.bulk_reload_group()

    def get_max_deduplicated_groups(self) -> int:
        seen: List[str] = []
        for g in self.offload_groups:
            if g._name not in seen:
                seen.append(g._name)
        return len(seen)

    # ------------------------------------------------------------------
    # Lifecycle callbacks (called by autograd hooks)
    # ------------------------------------------------------------------

    def on_group_start_forward(self, name: str) -> None:
        if not self.do_offload:
            return
        self._offloaded_group_index += 1
        if self.is_warmup:
            self.offload_groups.append(OffloadTensorGroup(name))
            self._max_group_size = max(self._max_group_size, self._offloaded_group_index)
        else:
            for g in self.offload_groups[self._offloaded_group_index - 1:]:
                if g._name == name:
                    break
                self._offloaded_group_index += 1
        self._tensor_count_current_group = 0
        self._groups_to_offload.append(self.offload_groups[self._offloaded_group_index - 1])

    def on_group_commit_forward(self, forced_released_tensors: List[torch.Tensor]) -> None:
        if not self.do_offload:
            return
        self.d2h_stream.wait_stream(torch.cuda.current_stream())
        self.bulk_offload(forced_released_tensors)

    def on_group_start_backward(self) -> None:
        if not self.do_offload:
            return
        self.h2d_stream.wait_stream(torch.cuda.current_stream())
        self.bulk_reload()

    def on_group_commit_backward(self, name: str) -> None:
        if not self.do_offload:
            return
        mgr = PipelineOffloadManager.get_instance()
        cur = mgr.cur_backward_chunk()
        if cur is not self:
            mgr.pop_backward_chunk(name)
        cur = mgr.cur_backward_chunk()
        assert cur is self, f"Chunk mismatch: {cur} vs {self}"
        if self._reloading_group:
            for rg in list(self._reloading_group):
                if rg._name == name:
                    rg.wait_reload_event(torch.cuda.current_stream())
                    self._reloading_group.remove(rg)
                    break


# ---------------------------------------------------------------------------
# Pipeline-level singleton manager
# ---------------------------------------------------------------------------

class PipelineOffloadManager:
    """Singleton that coordinates activation offloading across pipeline stages.

    Manages per-micro-batch ``ChunkOffloadHandler`` objects, routes forward /
    backward lifecycle callbacks, and owns the shared CUDA streams and CPU
    tensor pool.
    """

    OFFLOAD_MGR: Optional["PipelineOffloadManager"] = None

    @classmethod
    def get_instance(cls) -> "PipelineOffloadManager":
        if cls.OFFLOAD_MGR is None:
            cls.OFFLOAD_MGR = PipelineOffloadManager()
        return cls.OFFLOAD_MGR

    @classmethod
    def reset_instance(cls) -> None:
        cls.OFFLOAD_MGR = None
        cls.OFFLOAD_MGR = PipelineOffloadManager()

    def __init__(self) -> None:
        self._queue: deque = deque()
        self._stages: Optional[List[List[ChunkOffloadHandler]]] = None
        self._d2h_stream = torch.cuda.Stream()
        self._h2d_stream = torch.cuda.Stream()
        self._cpu_tensor_pool = GPUTensorPool(device="cpu", pin_memory=True)
        self._is_warmup: bool = True
        self._cached_chunks_forward: List[ChunkOffloadHandler] = []
        self._cached_chunks_backward: List[ChunkOffloadHandler] = []
        self._cached_chunks_index_backward: int = 0
        self._cached_chunks_index_forward: int = 0
        self.do_offload: bool = True
        self._offload_margin: int = 0
        self._delayed_offload_groups: List = []
        self._vpp: int = 1
        self._offload_summary_bytes: Dict[str, int] = {}
        self._offload_summary_total_bytes: int = 0
        self._cur_forward_chunk: Optional[ChunkOffloadHandler] = None
        self._cur_backward_chunk: Optional[ChunkOffloadHandler] = None
        self.inside_context: bool = False
        self.reset()

    # ------------------------------------------------------------------
    @property
    def d2h_stream(self) -> torch.cuda.Stream:
        return self._d2h_stream

    @property
    def h2d_stream(self) -> torch.cuda.Stream:
        return self._h2d_stream

    @property
    def cpu_tensor_pool(self) -> GPUTensorPool:
        return self._cpu_tensor_pool

    @property
    def offload_summary_bytes(self) -> Dict[str, int]:
        return self._offload_summary_bytes

    @property
    def offload_summary_total_bytes(self) -> int:
        return self._offload_summary_total_bytes

    # ------------------------------------------------------------------
    def push_offload_groups(
        self,
        group_hook: Any,
        forced_released_tensors: List[torch.Tensor],
    ) -> None:
        self._delayed_offload_groups.append((group_hook, forced_released_tensors))

    def flush_delayed_groups(self) -> None:
        for hook, forced in reversed(self._delayed_offload_groups):
            hook(forced)
        self._delayed_offload_groups = []

    def reset(self) -> None:
        self._cur_forward_chunk = None
        self._cur_backward_chunk = None
        self.inside_context = False
        if hasattr(self, "_cpu_tensor_pool"):
            self._cpu_tensor_pool.reset()
        if self._is_warmup and self._cached_chunks_forward:
            self.post_warmup_callback()
        self._cached_chunks_index_backward = 0
        self._cached_chunks_index_forward = 0
        for chunk in self._cached_chunks_forward:
            chunk.reset()
        self._delayed_offload_groups = []

    def flush(self) -> None:
        """Flush all staged VPP chunks to the backward queue in reverse."""
        if self._stages is None:
            return
        if len(self._stages[0]) == len(self._stages[-1]):
            lens = [len(e) for e in self._stages]
            assert min(lens) == max(lens), "All VPP stages must have same chunk count"
            self._stages[-1] = []
            for chunks in reversed(self._stages):
                for chunk in chunks:
                    self.push(chunk)
            for i in range(self._vpp):
                self._stages[i] = []

    def disable_offload(self) -> None:
        self.do_offload = False
        for chunk in self._cached_chunks_forward:
            chunk.do_offload = False

    def enable_offload(self) -> None:
        self.do_offload = True
        for chunk in self._cached_chunks_forward:
            chunk.do_offload = True

    def post_warmup_callback(self) -> None:
        self._is_warmup = False
        assert len(self._cached_chunks_forward) == len(self._cached_chunks_backward)
        for chunk in self._cached_chunks_forward:
            chunk.is_warmup = False
            assert chunk in self._cached_chunks_backward
            self._offload_margin = max(self._offload_margin, chunk.get_max_deduplicated_groups())

        last_group_with_same_name: Dict[str, OffloadTensorGroup] = {}
        for chunk in reversed(self._cached_chunks_backward):
            for group in chunk.offload_groups:
                last_group_with_same_name[group._name] = group

        for name, group in last_group_with_same_name.items():
            if self._offload_margin > 0:
                group.offload = False
                self._offload_margin -= 1
            else:
                break
        assert self._offload_margin == 0

        total_tensor_count: Dict[str, int] = {}
        total_offload_bytes: Dict[str, int] = {}
        for chunk in self._cached_chunks_forward:
            for group in chunk.offload_groups:
                if group.offload:
                    total_tensor_count[group._name] = (
                        total_tensor_count.get(group._name, 0) + group.total_tensor_count
                    )
                    total_offload_bytes[group._name] = (
                        total_offload_bytes.get(group._name, 0) + group.total_offload_bytes
                    )
            if chunk is self._cached_chunks_backward[0]:
                break

        self._offload_summary_bytes = dict(total_offload_bytes)
        self._offload_summary_total_bytes = int(sum(total_offload_bytes.values()))
        if torch.distributed.is_initialized():
            print_offload_summary_table(total_offload_bytes)

    # ------------------------------------------------------------------
    def push(self, handler: ChunkOffloadHandler) -> None:
        self._queue.append(handler)
        if self._is_warmup:
            self._cached_chunks_backward.append(handler)

    def pop_backward_chunk(self, name: Optional[str] = None) -> None:
        self._cur_backward_chunk = None
        for handler in self._cached_chunks_backward[self._cached_chunks_index_backward:]:
            self._cached_chunks_index_backward += 1
            if not handler.is_empty_chunk(name):
                self._cur_backward_chunk = handler
                break
        assert self._cur_backward_chunk is not None, "No non-empty backward chunk found"

    def front_backward_chunk(
        self, name: Optional[str] = None
    ) -> Optional[ChunkOffloadHandler]:
        for handler in self._cached_chunks_backward[self._cached_chunks_index_backward:]:
            if not handler.is_empty_chunk(name):
                return handler
        return None

    def init_model_chunk_offload_handler(
        self,
        vp_size: Optional[int],
        vp_stage: Optional[int],
        min_offloaded_tensor_size: int = 1024 * 1024,
        max_inflight_offloads: Optional[int] = None,
    ) -> None:
        if not self._is_warmup:
            return
        vp_size = 1 if vp_size is None else vp_size
        if self._stages is None:
            self._vpp = vp_size
            self._stages = [[] for _ in range(vp_size)]

        cur_vpp_rank = 0 if vp_stage is None else vp_stage
        if cur_vpp_rank == self._vpp - 1:
            self.flush()

        cur_chunk = ChunkOffloadHandler(
            min_offloaded_tensor_size,
            self._cpu_tensor_pool,
            max_inflight_offloads=max_inflight_offloads,
        )
        self._stages[cur_vpp_rank].append(cur_chunk)
        if cur_vpp_rank == self._vpp - 1:
            self.push(cur_chunk)
            self.flush()
        self._cur_forward_chunk = cur_chunk
        cur_chunk.vpp_rank = cur_vpp_rank
        self._cached_chunks_forward.append(cur_chunk)

    def pop_forward_chunk(
        self, name: Optional[str] = None
    ) -> Optional[ChunkOffloadHandler]:
        if not self.do_offload:
            return self._cur_forward_chunk
        while not self._is_warmup and (
            self._cur_forward_chunk is None
            or self._cur_forward_chunk.finish_all_groups(name)
        ):
            if self._cached_chunks_index_forward >= len(self._cached_chunks_forward):
                self._cur_forward_chunk = None
                break
            self._cur_forward_chunk = self._cached_chunks_forward[
                self._cached_chunks_index_forward
            ]
            self._cached_chunks_index_forward += 1
        return self._cur_forward_chunk

    def cur_forward_chunk(self) -> Optional[ChunkOffloadHandler]:
        return self._cur_forward_chunk

    def cur_backward_chunk(self) -> Optional[ChunkOffloadHandler]:
        return self._cur_backward_chunk

    def mark_not_offloadable(self, tensor: torch.Tensor) -> None:
        if tensor is not None:
            tensor.offloading_activation = False

    # ------------------------------------------------------------------
    # Saved-tensor hooks (pure PyTorch, no TransformerEngine dependency)
    # ------------------------------------------------------------------

    def __enter__(self) -> None:
        if self._cur_forward_chunk is None or not self.cur_forward_chunk().do_offload:
            return
        self.inside_context = True
        torch._C._autograd._push_saved_tensors_default_hooks(
            self.on_save_for_backward,
            self.on_get_saved_tensor,
        )

    def __exit__(self, *args: Any) -> None:
        if self._cur_forward_chunk is None or not self.cur_forward_chunk().do_offload:
            return
        self.inside_context = False
        torch._C._autograd._pop_saved_tensors_default_hooks()

    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        assert self.inside_context, "Must be inside offload context"
        return self.cur_forward_chunk().tensor_push(tensor)

    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        return self.cur_backward_chunk().tensor_pop(saved_state)


# ---------------------------------------------------------------------------
# Autograd function wrappers (identical logic to Megatron originals)
# ---------------------------------------------------------------------------

class FineGrainedOffloadingGroupCommitFunction(torch.autograd.Function):
    """Identity op marking end-of-layer for offload during forward,
    and reload synchronisation during backward."""

    @staticmethod
    def forward(ctx, tensor, cur_forward_chunk, name, forced_released_tensors, delay_offload):
        if delay_offload:
            PipelineOffloadManager.get_instance().push_offload_groups(
                cur_forward_chunk.on_group_commit_forward, forced_released_tensors
            )
        else:
            cur_forward_chunk.on_group_commit_forward(forced_released_tensors)
        ctx.cpu_offload_handler = cur_forward_chunk
        ctx.name = name
        return tensor

    @staticmethod
    def backward(ctx, *grad_output):
        ctx.cpu_offload_handler.on_group_commit_backward(ctx.name)
        return grad_output + (None, None, None, None)


def fine_grained_offloading_group_commit(
    tensor: Any,
    name: str,
    forced_released_tensors: Optional[List[torch.Tensor]] = None,
    delay_offload: bool = False,
) -> Any:
    """Commit the current layer group for offload at end-of-forward."""
    if forced_released_tensors is None:
        forced_released_tensors = []
    if isinstance(tensor, tuple):
        if not tensor:
            return tensor
        head = fine_grained_offloading_group_commit(
            tensor[0], name=name,
            forced_released_tensors=forced_released_tensors,
            delay_offload=delay_offload,
        )
        return (head,) + tensor[1:]
    if isinstance(tensor, list):
        if not tensor:
            return tensor
        head = fine_grained_offloading_group_commit(
            tensor[0], name=name,
            forced_released_tensors=forced_released_tensors,
            delay_offload=delay_offload,
        )
        return [head] + tensor[1:]

    cur = PipelineOffloadManager.get_instance().cur_forward_chunk()
    if cur is None:
        return tensor
    return FineGrainedOffloadingGroupCommitFunction.apply(
        tensor, cur, name, forced_released_tensors, delay_offload
    )


def fine_grained_offloading_group_flush_delayed_groups() -> None:
    """Flush any groups whose offload was deferred with *delay_offload=True*."""
    PipelineOffloadManager.get_instance().flush_delayed_groups()


class FineGrainedOffloadingGroupStartFunction(torch.autograd.Function):
    """Identity op marking start-of-layer for offload tracking."""

    @staticmethod
    def forward(ctx, tensor, cpu_offload_handler, name):
        ctx.cpu_offload_handler = cpu_offload_handler
        cpu_offload_handler.on_group_start_forward(name)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        ctx.cpu_offload_handler.on_group_start_backward()
        return grad_output, None, None, None


def fine_grained_offloading_group_start(
    tensor: torch.Tensor,
    name: Optional[str] = None,
) -> torch.Tensor:
    """Mark the start of a layer group and prepare for offload/reload."""
    cur = PipelineOffloadManager.get_instance().pop_forward_chunk(name=name)
    if cur is None:
        return tensor
    return FineGrainedOffloadingGroupStartFunction.apply(tensor, cur, name)


class FineGrainedOffloadingBackwardRecordFunction(torch.autograd.Function):
    """Records CUDA events for cuda-graph-captured forward/backward streams."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, event: torch.cuda.Event) -> torch.Tensor:
        ctx.event = event
        return tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        h2d = PipelineOffloadManager.get_instance().h2d_stream
        torch.cuda.current_stream().record_event(ctx.event)
        torch.cuda.current_stream().wait_stream(h2d)
        return grad_output, None


def fine_grained_offloading_backward_record(
    tensor: torch.Tensor,
    event: torch.cuda.Event,
) -> torch.Tensor:
    return FineGrainedOffloadingBackwardRecordFunction.apply(tensor, event)


def fine_grained_offloading_forward_record(event: torch.cuda.Event) -> None:
    d2h = PipelineOffloadManager.get_instance().d2h_stream
    torch.cuda.current_stream().record_event(event)
    torch.cuda.current_stream().wait_stream(d2h)


# ---------------------------------------------------------------------------
# Disable / enable helpers
# ---------------------------------------------------------------------------

def fine_grained_offloading_disable_offload() -> None:
    PipelineOffloadManager.get_instance().disable_offload()


def fine_grained_offloading_enable_offload() -> None:
    PipelineOffloadManager.get_instance().enable_offload()


# ---------------------------------------------------------------------------
# High-level context-manager interface (mirrors Megatron's)
# ---------------------------------------------------------------------------

class FineGrainedActivationOffloadingInterface:
    """Convenience context manager for per-layer activation offloading.

    Usage::

        FineGrainedActivationOffloadingInterface.init_chunk_handler(vp_size, vp_stage, min_size)
        with FineGrainedActivationOffloadingInterface(offload=True, tensor=x, name="attn") as x:
            # transformer layer forward …
            x = FineGrainedActivationOffloadingInterface.group_commit(x, "attn")
    """

    def __init__(self, offload: bool, tensor: torch.Tensor, name: str):
        self.offload = offload
        self.tensor = tensor
        self.name = name

    def __enter__(self) -> torch.Tensor:
        if self.offload:
            self.tensor = fine_grained_offloading_group_start(self.tensor, self.name)
            PipelineOffloadManager.get_instance().__enter__()
        return self.tensor

    def __exit__(self, *args: Any) -> None:
        if self.offload:
            PipelineOffloadManager.get_instance().__exit__()

    @staticmethod
    def init_chunk_handler(
        vp_size: Optional[int],
        vp_stage: Optional[int],
        min_offloaded_tensor_size: int,
        max_inflight_offloads: Optional[int] = None,
    ) -> None:
        PipelineOffloadManager.get_instance().init_model_chunk_offload_handler(
            vp_size, vp_stage, min_offloaded_tensor_size,
            max_inflight_offloads=max_inflight_offloads,
        )

    @staticmethod
    def get_context(flag: bool) -> Any:
        return PipelineOffloadManager.get_instance() if flag else nullcontext()

    @staticmethod
    def group_commit(
        tensor: Any,
        name: str,
        forced_released_tensors: Optional[List[torch.Tensor]] = None,
        delay_offload: bool = False,
    ) -> Any:
        return fine_grained_offloading_group_commit(
            tensor, name, forced_released_tensors, delay_offload
        )

    @staticmethod
    def mark_not_offloadable(tensor: torch.Tensor) -> None:
        PipelineOffloadManager.get_instance().mark_not_offloadable(tensor)

    @staticmethod
    def forward_record(event: torch.cuda.Event) -> None:
        fine_grained_offloading_forward_record(event)

    @staticmethod
    def reset() -> None:
        PipelineOffloadManager.get_instance().reset()

    @staticmethod
    def reset_instance() -> None:
        PipelineOffloadManager.reset_instance()
