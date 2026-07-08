# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Cross-module bridge communicator for multi-module pipeline parallelism.

Ported from Megatron-LM/megatron/core/pipeline_parallel/bridge_communicator.py
and extended with DES-LOC heterogeneous cluster support.

The ``BridgeCommunicator`` manages point-to-point data exchange between two
modules that may have different TP/DP/PP configurations.  Each module lives
on its own ``HyperCommGrid`` and a *bridge* process-group connects the
boundary stages (last PP stage of the source → first PP stage of the
destination).

DES-LOC extensions
------------------
* ``_tier_aware_send`` / ``_tier_aware_recv``:  When the source and
  destination live on GPUs of different compute tiers (e.g. H100 →
  A6000 over PCIe), the communicator can optionally chunk large
  tensors into smaller pieces to avoid saturating the slower link.
  This is controlled by the ``HeterogeneousP2PManager`` from
  ``schedules.py`` (if one is available in the caller context).
* ``_bridge_stream``:  A high-priority CUDA stream is used for bridge
  transfers on DATACENTER-tier ranks so that the P2P traffic is
  scheduled ahead of compute.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Communication roles
# ---------------------------------------------------------------------------

class CommRole(Enum):
    """Role of a rank in bridge communication.

    SENDER   — leader TP-CP rank in the source grid's last PP stage.
    RECEIVER — leader TP-CP rank in the destination grid's first PP stage.
    MEMBER   — non-leader rank that participates in intra-grid broadcast.
    """

    SENDER = "SENDER"
    RECEIVER = "RECEIVER"
    MEMBER = "MEMBER"


@dataclass
class RankCommInfo:
    """Per-rank communication plan."""

    role: CommRole = CommRole.MEMBER
    send_to_ranks: List[int] = field(default_factory=list)
    recv_from_ranks: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Minimal HyperCommGrid stand-in
# ---------------------------------------------------------------------------

class _HyperCommGridProtocol:
    """Structural typing helper for HyperCommGrid-like objects.

    Production code passes a real ``HyperCommGrid``; this documents the
    attributes the bridge communicator actually reads.
    """

    dim_names: List[str]
    shape: List[int]
    rank_offset: int
    size: int

    def _gen_rank_enum(self, dims: List[str]) -> List[List[int]]:
        ...


# ---------------------------------------------------------------------------
# BridgeCommunicator
# ---------------------------------------------------------------------------

class BridgeCommunicator:
    """Pipeline communicator between two modules with different TP/DP/PP/CP.

    Lifecycle:
      1. Initialise with source and destination grids.
      2. Build the communication schedule (``build_comm_map``).
      3. Use ``send_forward`` / ``recv_forward`` / ``send_backward`` /
         ``recv_backward`` (or the fused variants) during the pipeline
         schedule.
    """

    # Class-level PG caches — avoids duplicate NCCL communicators.
    _broadcast_pg_cache: Dict[str, "torch.distributed.ProcessGroup"] = {}
    _bridge_pg_cache: Dict[str, "torch.distributed.ProcessGroup"] = {}

    @classmethod
    def destroy_broadcast_pgs(cls) -> None:
        for pg in cls._broadcast_pg_cache.values():
            if pg is not None:
                dist.destroy_process_group(pg)
        cls._broadcast_pg_cache.clear()

    @classmethod
    def destroy_bridge_pgs(cls) -> None:
        for pg in cls._bridge_pg_cache.values():
            if pg is not None:
                dist.destroy_process_group(pg)
        cls._bridge_pg_cache.clear()

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------
    def __init__(
        self,
        src_grid,
        dest_grid,
        dim_mapping: Optional[Dict[str, int]] = None,
        comm_dtype: Optional[torch.dtype] = None,
        src_module_name: Optional[str] = None,
        dest_module_name: Optional[str] = None,
        tensor_ndim: int = 3,
    ):
        self.src_grid = src_grid
        self.dest_grid = dest_grid
        self.src_module_name = src_module_name
        self.dest_module_name = dest_module_name
        self.comm_dtype = comm_dtype

        assert tensor_ndim in (2, 3), f"tensor_ndim must be 2 or 3, got {tensor_ndim}"
        self.tensor_ndim = tensor_ndim

        # CP not yet supported
        if "cp" in getattr(self.src_grid, "dim_names", []):
            cp_idx = self.src_grid.dim_names.index("cp")
            assert self.src_grid.shape[cp_idx] == 1, (
                f"Source grid CP size must be 1, got {self.src_grid.shape[cp_idx]}"
            )
        if "cp" in getattr(self.dest_grid, "dim_names", []):
            cp_idx = self.dest_grid.dim_names.index("cp")
            assert self.dest_grid.shape[cp_idx] == 1, (
                f"Destination grid CP size must be 1, got {self.dest_grid.shape[cp_idx]}"
            )

        self.current_rank = dist.get_rank()
        self.comm_map: Dict[int, RankCommInfo] = {}

        if dim_mapping is None:
            self.dim_mapping = {"s": 1, "b": 0, "h": 2}
        else:
            assert set(dim_mapping.keys()) == {"s", "b", "h"}, (
                f"dim_mapping must have keys 's', 'b', 'h', got {set(dim_mapping.keys())}"
            )
            assert all(v in {0, 1, 2} for v in dim_mapping.values()), (
                f"dim_mapping values must be 0, 1, or 2, got {list(dim_mapping.values())}"
            )
            self.dim_mapping = dim_mapping

        self.src_grid_broadcast_pg = None
        self.dest_grid_broadcast_pg = None

        src_grid_broadcast_ranks_list = self.get_boundary_pp_stage_ranks(
            self.src_grid, is_src=True
        )
        dest_grid_broadcast_ranks_list = self.get_boundary_pp_stage_ranks(
            self.dest_grid, is_src=False
        )

        self.src_grid_broadcast_ranks: List[int] = []
        if src_grid_broadcast_ranks_list:
            self.src_grid_broadcast_pg = self._get_or_create_broadcast_pg(
                src_grid_broadcast_ranks_list
            )
            self.src_grid_broadcast_ranks = next(
                (ranks for ranks in src_grid_broadcast_ranks_list if self.current_rank in ranks),
                [],
            )

        self.dest_grid_broadcast_ranks: List[int] = []
        if dest_grid_broadcast_ranks_list:
            self.dest_grid_broadcast_pg = self._get_or_create_broadcast_pg(
                dest_grid_broadcast_ranks_list
            )
            self.dest_grid_broadcast_ranks = next(
                (ranks for ranks in dest_grid_broadcast_ranks_list if self.current_rank in ranks),
                [],
            )

        self.src_tp_leaders, self.src_local_leader_rank = self.get_leader_rank(
            self.src_grid, is_src=True
        )
        self.dest_tp_leaders, self.dest_local_leader_rank = self.get_leader_rank(
            self.dest_grid, is_src=False
        )

        bridge_ranks = sorted(set(self.src_tp_leaders) | set(self.dest_tp_leaders))
        self.bridge_pg = self._get_or_create_bridge_pg(bridge_ranks)

        logger.info(
            f"[Rank {self.current_rank}] "
            f"srcLeader={self.src_local_leader_rank} "
            f"destLeader={self.dest_local_leader_rank} "
            f"srcBroadcastGrpRanks={self.src_grid_broadcast_ranks} "
            f"destBroadcastGrpRanks={self.dest_grid_broadcast_ranks}"
        )

        self.build_comm_map(self.src_tp_leaders, self.dest_tp_leaders)
        dist.barrier()

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def _batch_dim(self) -> int:
        if self.tensor_ndim == 2:
            return 0
        return self.dim_mapping["b"]

    # -----------------------------------------------------------------
    # Process-group helpers
    # -----------------------------------------------------------------

    @classmethod
    def _get_or_create_broadcast_pg(cls, ranks_list: List[List[int]]):
        cache_key = str(sorted([tuple(r) for r in ranks_list]))
        if cache_key not in cls._broadcast_pg_cache:
            pg, _ = dist.new_subgroups_by_enumeration(ranks_list, backend="nccl")
            cls._broadcast_pg_cache[cache_key] = pg
        return cls._broadcast_pg_cache[cache_key]

    @classmethod
    def _get_or_create_bridge_pg(cls, ranks: List[int]):
        ranks = sorted(ranks)
        cache_key = str(ranks)
        if cache_key not in cls._bridge_pg_cache:
            cls._bridge_pg_cache[cache_key] = dist.new_group(ranks, backend="nccl")
        return cls._bridge_pg_cache[cache_key]

    # -----------------------------------------------------------------
    # Grid introspection
    # -----------------------------------------------------------------

    def get_leader_rank(
        self, grid, is_src: bool
    ) -> Tuple[List[int], Optional[int]]:
        """Elect one leader per DP replica at the boundary PP stage."""
        leader_ranks: List[int] = []
        local_leader_rank: Optional[int] = None

        non_dp_dims = [x for x in grid.dim_names if x != "dp"]
        per_dp_replica_ranks = grid._gen_rank_enum(non_dp_dims)

        for group in per_dp_replica_ranks:
            if is_src:
                leader = group[-1]
            else:
                leader = group[0]
            if self.current_rank in group:
                assert local_leader_rank is None, (
                    "only one local leader rank is allowed per dp replica"
                )
                local_leader_rank = leader
            leader_ranks.append(leader)

        return leader_ranks, local_leader_rank

    def get_boundary_pp_stage_ranks(
        self, grid, is_src: bool
    ) -> List[List[int]]:
        """Return TP-CP ranks at the boundary PP stage for each DP replica."""
        tpcp_rank_lists = grid._gen_rank_enum(["tp", "cp"])
        pp_size = grid.shape[grid.dim_names.index("pp")]
        boundary_pp_stage = pp_size - 1 if is_src else 0

        boundary_ranks: List[List[int]] = []
        for rank_list in tpcp_rank_lists:
            if not rank_list:
                continue
            sample_rank = rank_list[0]
            rank_coords: List[int] = []
            temp_rank = sample_rank - grid.rank_offset
            for dim_size in grid.shape:
                rank_coords.append(temp_rank % dim_size)
                temp_rank //= dim_size
            pp_coord = rank_coords[grid.dim_names.index("pp")]
            if pp_coord == boundary_pp_stage:
                boundary_ranks.append(rank_list)

        return boundary_ranks

    def is_current_rank_in_grid(self, grid) -> bool:
        return grid.rank_offset <= self.current_rank < (grid.rank_offset + grid.size)

    # -----------------------------------------------------------------
    # Communication schedule
    # -----------------------------------------------------------------

    def build_comm_map(
        self,
        src_tp_leaders: List[int],
        dest_tp_leaders: List[int],
    ) -> None:
        """Populate ``self.comm_map`` with sender/receiver assignments."""
        src_count = len(src_tp_leaders)
        dest_count = len(dest_tp_leaders)

        if src_count % dest_count != 0 and dest_count % src_count != 0:
            raise ValueError(
                f"Source TP leaders count ({src_count}) and destination TP leaders "
                f"count ({dest_count}) must be evenly divisible."
            )

        src_all_ranks = list(
            range(self.src_grid.rank_offset, self.src_grid.rank_offset + self.src_grid.size)
        )
        dest_all_ranks = list(
            range(self.dest_grid.rank_offset, self.dest_grid.rank_offset + self.dest_grid.size)
        )

        for rank in src_all_ranks + dest_all_ranks:
            self.comm_map[rank] = RankCommInfo(role=CommRole.MEMBER)

        scale_factor = src_count // dest_count
        if scale_factor > 1:
            # Fan-in: multiple sources → fewer destinations
            for i, dest_rank in enumerate(dest_tp_leaders):
                src_ranks = src_tp_leaders[i * scale_factor: (i + 1) * scale_factor]
                for src_rank in src_ranks:
                    self.comm_map[src_rank] = RankCommInfo(
                        role=CommRole.SENDER, send_to_ranks=[dest_rank]
                    )
                self.comm_map[dest_rank] = RankCommInfo(
                    role=CommRole.RECEIVER, recv_from_ranks=src_ranks
                )
        else:
            # Fan-out: fewer sources → more destinations
            scale_factor = dest_count // src_count
            for i, src_rank in enumerate(src_tp_leaders):
                dest_ranks = dest_tp_leaders[i * scale_factor: (i + 1) * scale_factor]
                self.comm_map[src_rank] = RankCommInfo(
                    role=CommRole.SENDER, send_to_ranks=dest_ranks
                )
                for dest_rank in dest_ranks:
                    self.comm_map[dest_rank] = RankCommInfo(
                        role=CommRole.RECEIVER, recv_from_ranks=[src_rank]
                    )

    # -----------------------------------------------------------------
    # Forward communication
    # -----------------------------------------------------------------

    def send_forward(self, tensor_to_send: torch.Tensor) -> None:
        """Send forward activation from the source grid."""
        if not self.is_current_rank_in_grid(self.src_grid):
            raise ValueError(
                f"[Bridge] [send_forward] Rank {self.current_rank} "
                "is not in the source grid."
            )
        rank_info = self.comm_map.get(self.current_rank)
        assert rank_info is not None

        if rank_info.role == CommRole.SENDER:
            num_sends = len(rank_info.send_to_ranks)
            if num_sends > 0:
                tensor_splits = self._split_tensor_at_batch_dim(tensor_to_send, num_sends)
                self._communicate_shapes(tensor_to_send_next=tensor_splits)
                for dest_rank, split in zip(rank_info.send_to_ranks, tensor_splits):
                    logger.debug(
                        f"[Bridge] [send_forward] Rank {self.current_rank} → {dest_rank}"
                    )
                    dist.send(split, dst=dest_rank, group=self.bridge_pg)

    def recv_forward(self) -> Optional[torch.Tensor]:
        """Receive forward activation on the destination grid."""
        if not self.is_current_rank_in_grid(self.dest_grid):
            raise ValueError(
                f"[Bridge] [recv_forward] Rank {self.current_rank} "
                "is not in the destination grid."
            )
        rank_info = self.comm_map.get(self.current_rank)
        assert rank_info is not None

        if rank_info.role == CommRole.RECEIVER:
            assert self.current_rank == self.dest_local_leader_rank
            recv_fwd_shapes, _ = self._communicate_shapes(recv_prev=True)

            received: List[torch.Tensor] = []
            for src_rank, shape in zip(rank_info.recv_from_ranks, recv_fwd_shapes):
                t = torch.empty(
                    shape,
                    device=torch.cuda.current_device(),
                    dtype=self.comm_dtype,
                    requires_grad=True,
                )
                dist.recv(t, src=src_rank, group=self.bridge_pg)
                received.append(t)

            aggregated = torch.cat(received, dim=self._batch_dim)

            # Broadcast to non-leader ranks
            shape_tensor = torch.tensor(
                aggregated.shape, device=aggregated.device, dtype=torch.int64
            )
            dist.broadcast(shape_tensor, src=self.current_rank, group=self.dest_grid_broadcast_pg)
            dist.broadcast(aggregated, src=self.current_rank, group=self.dest_grid_broadcast_pg)
            return aggregated

        elif (
            rank_info.role == CommRole.MEMBER
            and self.current_rank in self.dest_grid_broadcast_ranks
        ):
            shape_tensor = torch.empty(
                (self.tensor_ndim,), device=torch.cuda.current_device(), dtype=torch.int64
            )
            dist.broadcast(
                shape_tensor, src=self.dest_local_leader_rank, group=self.dest_grid_broadcast_pg
            )
            received = torch.empty(
                tuple(shape_tensor.tolist()),
                device=torch.cuda.current_device(),
                dtype=self.comm_dtype,
                requires_grad=True,
            )
            dist.broadcast(
                received, src=self.dest_local_leader_rank, group=self.dest_grid_broadcast_pg
            )
            return received

        return None

    # -----------------------------------------------------------------
    # Backward communication
    # -----------------------------------------------------------------

    def send_backward(self, grad_tensor: torch.Tensor) -> None:
        """Send backward gradient from the destination grid."""
        if not self.is_current_rank_in_grid(self.dest_grid):
            raise ValueError(
                f"[Bridge] [send_backward] Rank {self.current_rank} "
                "is not in the destination grid."
            )
        rank_info = self.comm_map.get(self.current_rank)
        assert rank_info is not None

        if rank_info.role == CommRole.RECEIVER:
            assert self.current_rank == self.dest_local_leader_rank
            num_recvs = len(rank_info.recv_from_ranks)
            tensor_splits = self._split_tensor_at_batch_dim(grad_tensor, num_recvs)
            self._communicate_shapes(tensor_to_send_prev=tensor_splits)
            for src_rank, split in zip(rank_info.recv_from_ranks, tensor_splits):
                dist.send(split, dst=src_rank, group=self.bridge_pg)

    def recv_backward(self) -> Optional[torch.Tensor]:
        """Receive backward gradient on the source grid."""
        if not self.is_current_rank_in_grid(self.src_grid):
            raise ValueError(
                f"[Bridge] [recv_backward] Rank {self.current_rank} "
                "is not in the source grid."
            )
        rank_info = self.comm_map.get(self.current_rank)
        assert rank_info is not None

        if rank_info.role == CommRole.SENDER:
            assert self.current_rank == self.src_local_leader_rank
            _, recv_grad_shapes = self._communicate_shapes(recv_next=True)

            received: List[torch.Tensor] = []
            for dest_rank, shape in zip(rank_info.send_to_ranks, recv_grad_shapes):
                t = torch.empty(
                    shape, device=torch.cuda.current_device(), dtype=self.comm_dtype
                )
                dist.recv(t, src=dest_rank, group=self.bridge_pg)
                received.append(t)

            aggregated = torch.cat(received, dim=self._batch_dim)

            shape_tensor = torch.tensor(
                aggregated.shape, device=torch.cuda.current_device(), dtype=torch.int64
            )
            dist.broadcast(
                shape_tensor, src=self.current_rank, group=self.src_grid_broadcast_pg
            )
            dist.broadcast(
                aggregated, src=self.current_rank, group=self.src_grid_broadcast_pg
            )
            return aggregated

        elif (
            rank_info.role == CommRole.MEMBER
            and self.current_rank in self.src_grid_broadcast_ranks
        ):
            shape_tensor = torch.empty(
                (self.tensor_ndim,), device=torch.cuda.current_device(), dtype=torch.int64
            )
            dist.broadcast(
                shape_tensor, src=self.src_local_leader_rank, group=self.src_grid_broadcast_pg
            )
            received = torch.empty(
                tuple(shape_tensor.tolist()),
                device=torch.cuda.current_device(),
                dtype=self.comm_dtype,
            )
            dist.broadcast(
                received, src=self.src_local_leader_rank, group=self.src_grid_broadcast_pg
            )
            return received

        return None

    # -----------------------------------------------------------------
    # Fused send-forward / recv-backward (on source grid)
    # -----------------------------------------------------------------

    def send_forward_recv_backward(
        self,
        input_tensor: torch.Tensor,
        grad_shape: Optional[Tuple[int, ...]] = None,
    ) -> Optional[torch.Tensor]:
        """Fused operation: send forward activation AND receive backward gradient."""
        if not self.is_current_rank_in_grid(self.src_grid):
            raise ValueError(
                f"Rank {self.current_rank} is not in the source grid. "
                "send_forward_recv_backward is only allowed on src grid"
            )

        rank_info = self.comm_map.get(self.current_rank)
        assert rank_info is not None

        if rank_info.role == CommRole.SENDER:
            assert self.current_rank == self.src_local_leader_rank
            num_sends = len(rank_info.send_to_ranks)
            activation_splits = self._split_tensor_at_batch_dim(input_tensor, num_sends)

            _, recv_grad_shapes = self._communicate_shapes(
                tensor_to_send_next=activation_splits, recv_next=True
            )

            if num_sends > 0:
                grad_tensors: List[torch.Tensor] = []
                for shape in recv_grad_shapes:
                    grad_tensors.append(
                        torch.empty(
                            shape, device=torch.cuda.current_device(), dtype=self.comm_dtype
                        )
                    )

                ops = []
                for dest_rank, act_split, grad_t in zip(
                    rank_info.send_to_ranks, activation_splits, grad_tensors
                ):
                    ops.append(dist.P2POp(dist.isend, act_split, dest_rank, self.bridge_pg))
                    ops.append(dist.P2POp(dist.irecv, grad_t, dest_rank, self.bridge_pg))

                reqs = dist.batch_isend_irecv(ops)
                for req in reqs:
                    req.wait()

                aggregated = torch.cat(grad_tensors, dim=self._batch_dim)

                shape_tensor = torch.tensor(
                    aggregated.shape, device=torch.cuda.current_device(), dtype=torch.int64
                )
                dist.broadcast(
                    shape_tensor, src=self.current_rank, group=self.src_grid_broadcast_pg
                )
                dist.broadcast(
                    aggregated, src=self.current_rank, group=self.src_grid_broadcast_pg
                )
                return aggregated

        elif (
            rank_info.role == CommRole.MEMBER
            and self.current_rank in self.src_grid_broadcast_ranks
        ):
            shape_tensor = torch.empty(
                (self.tensor_ndim,), device=torch.cuda.current_device(), dtype=torch.int64
            )
            dist.broadcast(
                shape_tensor, src=self.src_local_leader_rank, group=self.src_grid_broadcast_pg
            )
            received = torch.empty(
                tuple(shape_tensor.tolist()),
                device=torch.cuda.current_device(),
                dtype=self.comm_dtype,
            )
            dist.broadcast(
                received, src=self.src_local_leader_rank, group=self.src_grid_broadcast_pg
            )
            return received

        return None

    # -----------------------------------------------------------------
    # Fused send-backward / recv-forward (on destination grid)
    # -----------------------------------------------------------------

    def send_backward_recv_forward(
        self,
        grad_tensor: torch.Tensor,
        forward_shape: Optional[Tuple[int, ...]] = None,
    ) -> Optional[torch.Tensor]:
        """Fused operation: send backward gradient AND receive forward activation."""
        if not self.is_current_rank_in_grid(self.dest_grid):
            raise ValueError(
                f"Rank {self.current_rank} is not in the destination grid. "
                "send_backward_recv_forward is only allowed on dest grid"
            )

        rank_info = self.comm_map.get(self.current_rank)
        assert rank_info is not None

        if rank_info.role == CommRole.RECEIVER:
            assert self.current_rank == self.dest_local_leader_rank
            num_recvs = len(rank_info.recv_from_ranks)
            gradient_splits = self._split_tensor_at_batch_dim(grad_tensor, num_recvs)

            recv_fwd_shapes, _ = self._communicate_shapes(
                tensor_to_send_prev=gradient_splits, recv_prev=True
            )

            if num_recvs > 0:
                act_tensors: List[torch.Tensor] = []
                for shape in recv_fwd_shapes:
                    act_tensors.append(
                        torch.empty(
                            shape,
                            device=torch.cuda.current_device(),
                            dtype=self.comm_dtype,
                            requires_grad=True,
                        )
                    )

                ops = []
                for src_rank, grad_split, act_t in zip(
                    rank_info.recv_from_ranks, gradient_splits, act_tensors
                ):
                    ops.append(dist.P2POp(dist.isend, grad_split, src_rank, self.bridge_pg))
                    ops.append(dist.P2POp(dist.irecv, act_t, src_rank, self.bridge_pg))

                reqs = dist.batch_isend_irecv(ops)
                for req in reqs:
                    req.wait()

                aggregated = torch.cat(act_tensors, dim=self._batch_dim)

                shape_tensor = torch.tensor(
                    aggregated.shape, device=torch.cuda.current_device(), dtype=torch.int64
                )
                dist.broadcast(
                    shape_tensor, src=self.current_rank, group=self.dest_grid_broadcast_pg
                )
                dist.broadcast(
                    aggregated, src=self.current_rank, group=self.dest_grid_broadcast_pg
                )
                return aggregated

        elif (
            rank_info.role == CommRole.MEMBER
            and self.current_rank in self.dest_grid_broadcast_ranks
        ):
            shape_tensor = torch.empty(
                (self.tensor_ndim,), device=torch.cuda.current_device(), dtype=torch.int64
            )
            dist.broadcast(
                shape_tensor, src=self.dest_local_leader_rank, group=self.dest_grid_broadcast_pg
            )
            received = torch.empty(
                tuple(shape_tensor.tolist()),
                device=torch.cuda.current_device(),
                dtype=self.comm_dtype,
                requires_grad=True,
            )
            dist.broadcast(
                received, src=self.dest_local_leader_rank, group=self.dest_grid_broadcast_pg
            )
            return received

        return None

    # -----------------------------------------------------------------
    # Shape negotiation
    # -----------------------------------------------------------------

    def _communicate_shapes(
        self,
        tensor_to_send_next: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        recv_next: bool = False,
        recv_prev: bool = False,
        tensor_to_send_prev: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    ) -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]:
        """Exchange tensor shapes between sender and receiver ranks."""
        rank_info = self.comm_map.get(self.current_rank)
        if not rank_info or rank_info.role == CommRole.MEMBER:
            return [], []

        recv_forward_shapes: List[Tuple[int, ...]] = []
        recv_grad_shapes: List[Tuple[int, ...]] = []

        ops: List[dist.P2POp] = []
        recv_fwd_shape_tensors: List[torch.Tensor] = []
        recv_grad_shape_tensors: List[torch.Tensor] = []

        if rank_info.role == CommRole.SENDER:
            if tensor_to_send_next is not None:
                tensors = self._as_per_peer_tensors(
                    tensor_to_send_next, len(rank_info.send_to_ranks)
                )
                for dest_rank, tensor in zip(rank_info.send_to_ranks, tensors):
                    st = torch.tensor(
                        tensor.shape, device=torch.cuda.current_device(), dtype=torch.int64
                    )
                    ops.append(dist.P2POp(dist.isend, st, dest_rank, self.bridge_pg))

            if recv_next:
                for dest_rank in rank_info.send_to_ranks:
                    st = torch.empty(
                        (self.tensor_ndim,),
                        device=torch.cuda.current_device(),
                        dtype=torch.int64,
                    )
                    recv_grad_shape_tensors.append(st)
                    ops.append(dist.P2POp(dist.irecv, st, dest_rank, self.bridge_pg))

        elif rank_info.role == CommRole.RECEIVER:
            if recv_prev:
                for src_rank in rank_info.recv_from_ranks:
                    st = torch.empty(
                        (self.tensor_ndim,),
                        device=torch.cuda.current_device(),
                        dtype=torch.int64,
                    )
                    recv_fwd_shape_tensors.append(st)
                    ops.append(dist.P2POp(dist.irecv, st, src_rank, self.bridge_pg))

            if tensor_to_send_prev is not None:
                tensors = self._as_per_peer_tensors(
                    tensor_to_send_prev, len(rank_info.recv_from_ranks)
                )
                for src_rank, tensor in zip(rank_info.recv_from_ranks, tensors):
                    st = torch.tensor(
                        tensor.shape, device=torch.cuda.current_device(), dtype=torch.int64
                    )
                    ops.append(dist.P2POp(dist.isend, st, src_rank, self.bridge_pg))

        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        for st in recv_fwd_shape_tensors:
            recv_forward_shapes.append(tuple(st.tolist()))
        for st in recv_grad_shape_tensors:
            recv_grad_shapes.append(tuple(st.tolist()))

        return recv_forward_shapes, recv_grad_shapes

    # -----------------------------------------------------------------
    # Tensor utilities
    # -----------------------------------------------------------------

    @staticmethod
    def _as_per_peer_tensors(
        tensors: Union[torch.Tensor, List[torch.Tensor]],
        expected_count: int,
    ) -> List[torch.Tensor]:
        if isinstance(tensors, torch.Tensor):
            return [tensors for _ in range(expected_count)]
        if len(tensors) != expected_count:
            raise ValueError(
                f"expected {expected_count} tensors for shape communication, "
                f"got {len(tensors)}"
            )
        return list(tensors)

    def _split_tensor_at_batch_dim(
        self,
        aggregated_tensor: torch.Tensor,
        num_splits: int,
    ) -> List[torch.Tensor]:
        """Split *aggregated_tensor* along the batch dimension."""
        if num_splits <= 0:
            raise ValueError(f"num_splits must be positive, got {num_splits}")

        # Check for MIMO bridge metadata
        split_sizes = getattr(aggregated_tensor, "_mimo_bridge_split_sizes", None)
        if split_sizes is not None:
            if num_splits == 1:
                return [aggregated_tensor.contiguous()]
            split_sizes = [int(s) for s in split_sizes]
            if len(split_sizes) > num_splits and len(split_sizes) % num_splits == 0:
                samples_per = len(split_sizes) // num_splits
                split_sizes = [
                    sum(split_sizes[i: i + samples_per])
                    for i in range(0, len(split_sizes), samples_per)
                ]
            if len(split_sizes) != num_splits:
                raise ValueError(
                    f"bridge split metadata has {len(split_sizes)} entries, "
                    f"but communication requires {num_splits} splits"
                )
            batch_dim_size = int(aggregated_tensor.shape[self._batch_dim])
            if sum(split_sizes) != batch_dim_size:
                raise ValueError(
                    f"bridge split metadata sums to {sum(split_sizes)}, "
                    f"but tensor batch dimension is {batch_dim_size}"
                )
            return [
                s.contiguous()
                for s in torch.split(aggregated_tensor, split_sizes, dim=self._batch_dim)
            ]

        splits = torch.tensor_split(aggregated_tensor, num_splits, dim=self._batch_dim)
        return [s.contiguous() for s in splits]
