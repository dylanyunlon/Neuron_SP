# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Multi-module pipeline communicator for heterogeneous model architectures.

Ported from Megatron-LM/megatron/core/pipeline_parallel/multimodule_communicator.py
and extended with DES-LOC support for cross-tier module boundaries.

A ``MultiModulePipelineCommunicator`` manages the full communication topology
for a model that is composed of multiple modules (e.g. vision encoder, LLM
backbone, generator head).  Each module may have its own TP/DP/PP/CP
configuration (described by its ``HyperCommGrid``), and data flows between
modules through ``BridgeCommunicator`` instances.

DES-LOC extension
-----------------
When modules straddle different GPU tiers (e.g. a vision encoder on A6000
and the LLM backbone on H100), the bridge communicator automatically uses
the ``HeterogeneousP2PManager`` from ``schedules.py`` to chunk large
activation transfers across the PCIe link, preventing stalls on the slower
side.

Usage from the pipeline schedule::

    from deepspeed.core.pipeline_parallel.multimodule_communicator import (
        MultiModulePipelineCommunicator,
    )

    mm_comm = MultiModulePipelineCommunicator(
        module_to_grid_map={"encoder": enc_grid, "llm": llm_grid},
        topology={"encoder": ["llm"], "llm": []},
        config=model_parallel_config,
    )

    # In the schedule loop:
    inputs = mm_comm.recv_forward()
    outputs = model(inputs)
    mm_comm.send_forward(outputs)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist

from deepspeed.core.pipeline_parallel.bridge_communicator import BridgeCommunicator
from deepspeed.core.pipeline_parallel.p2p_communication import P2PCommunicator

# Type aliases
Shape = Union[List[int], torch.Size]


# ---------------------------------------------------------------------------
# Per-rank module information
# ---------------------------------------------------------------------------

@dataclass
class RankModuleInfo:
    """Information about the current rank's position inside a module's pipeline.

    Attributes:
        pp_rank: Stage index (0-based) of this rank within the module's PP.
        pp_size: Total number of PP stages in the module.
        p2p_communicator: Intra-module P2P communicator (None at boundary stages
            that exclusively use bridge communicators).
        bridge_comms_as_src_module: Outgoing bridge communicators (last PP stage
            of this module → first PP stage of downstream modules).
        bridge_comms_as_dest_module: Incoming bridge communicators (last PP stage
            of upstream modules → first PP stage of this module).
        is_source_stage: True if this rank is in the absolute first pipeline
            stage of the entire multi-module model.
        is_terminal_stage: True if this rank is in the absolute last pipeline
            stage of the entire multi-module model.
    """

    pp_rank: int
    pp_size: int
    p2p_communicator: Optional[P2PCommunicator]
    bridge_comms_as_src_module: Optional[List[BridgeCommunicator]]
    bridge_comms_as_dest_module: Optional[List[BridgeCommunicator]]
    is_source_stage: Optional[bool] = True
    is_terminal_stage: Optional[bool] = True


# ---------------------------------------------------------------------------
# Tensor shape adapters (P2P communicators expect 3-D)
# ---------------------------------------------------------------------------

def _prepare_tensor_for_comm(
    tensor: Union[torch.Tensor, List[torch.Tensor], None],
) -> Union[torch.Tensor, List[torch.Tensor], None]:
    """Expand 2-D tensors to 3-D by adding a singleton last dim for P2P.

    Bridge communicators handle 2-D/3-D natively (via ``tensor_ndim``), so
    this adapter is only used on the intra-module P2P path.

    Note: 3-D tensors with ``shape[-1] == 1`` are ambiguous and will assert.
    """
    if tensor is None:
        return None
    if isinstance(tensor, list):
        return [_prepare_tensor_for_comm(t) for t in tensor]
    if isinstance(tensor, torch.Tensor):
        if tensor.ndim == 2:
            return tensor.unsqueeze(-1)
        assert tensor.ndim != 3 or tensor.shape[-1] != 1, (
            f"3D tensor with singleton last dim {tuple(tensor.shape)} is ambiguous for "
            "multimodule comm. Cannot distinguish from an unsqueezed 2D tensor on the "
            "receiving rank. Use a 2D tensor or a 3D tensor with last_dim > 1."
        )
    return tensor


def _restore_tensor_from_comm(
    tensor: Union[torch.Tensor, List[torch.Tensor], None],
) -> Union[torch.Tensor, List[torch.Tensor], None]:
    """Squeeze singleton last dim added by :func:`_prepare_tensor_for_comm`."""
    if tensor is None:
        return None
    if isinstance(tensor, list):
        return [_restore_tensor_from_comm(t) for t in tensor]
    if isinstance(tensor, torch.Tensor) and tensor.ndim == 3 and tensor.shape[-1] == 1:
        return tensor.squeeze(-1)
    return tensor


# ---------------------------------------------------------------------------
# MultiModulePipelineCommunicator
# ---------------------------------------------------------------------------

class MultiModulePipelineCommunicator:
    """Communicator for a multi-module pipeline with heterogeneous grids.

    Manages both intra-module P2P and cross-module bridge communication so
    that the pipeline schedule only needs to call ``send_forward`` /
    ``recv_forward`` / etc. without knowing which communication path is used.
    """

    def __init__(
        self,
        module_to_grid_map: Dict[str, object],
        topology: Dict[str, List[str]],
        config: object,
        dim_mapping: Optional[Dict[str, int]] = None,
        module_output_ndim: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            module_to_grid_map: ``{module_name: HyperCommGrid}``
            topology: DAG of data flow — ``{src_module: [dst_module, ...]}``
            config: ``ModelParallelConfig`` (provides ``pipeline_dtype`` etc.)
            dim_mapping: ``{'s': int, 'b': int, 'h': int}`` dimension mapping.
            module_output_ndim: Per-module output tensor dimensionality
                (2 for [B*S, H] vision encoders; default 3).
        """
        self.module_to_grid_map = module_to_grid_map
        self.topology = topology
        self.config = config
        self.dim_mapping = dim_mapping
        self.module_output_ndim = module_output_ndim or {}
        self.current_rank = dist.get_rank()

        self.bridge_comms: List[BridgeCommunicator] = []
        self._build_bridge_comms()

        self.rank_module_map: Dict[str, RankModuleInfo] = {}
        self._build_rank_module_info_map()

    # -----------------------------------------------------------------
    # Initialisation helpers
    # -----------------------------------------------------------------

    def _build_bridge_comms(self) -> None:
        for src_name, src_grid in self.module_to_grid_map.items():
            for dest_name in self.topology.get(src_name, []):
                dest_grid = self.module_to_grid_map[dest_name]
                bc = BridgeCommunicator(
                    src_grid=src_grid,
                    dest_grid=dest_grid,
                    dim_mapping=self.dim_mapping,
                    comm_dtype=getattr(self.config, "pipeline_dtype", None),
                    src_module_name=src_name,
                    dest_module_name=dest_name,
                    tensor_ndim=self.module_output_ndim.get(src_name, 3),
                )
                self.bridge_comms.append(bc)

    def _build_rank_module_info_map(self) -> None:
        for module_name, module_grid in self.module_to_grid_map.items():
            if not self.is_current_rank_in_grid(module_grid):
                continue

            pp_group = module_grid.get_pg("pp")
            p2p_comm = P2PCommunicator(pp_group, self.config)
            pp_size = dist.get_world_size(pp_group)
            rank_in_pp = dist.get_group_rank(pp_group, self.current_rank)
            pp_rank = rank_in_pp % pp_size

            bridge_as_dest: List[BridgeCommunicator] = []
            bridge_as_src: List[BridgeCommunicator] = []

            if pp_rank == 0:
                for bc in self.bridge_comms:
                    if (
                        bc.is_current_rank_in_grid(bc.dest_grid)
                        and bc.dest_module_name == module_name
                    ):
                        bridge_as_dest.append(bc)

            if pp_rank == pp_size - 1:
                for bc in self.bridge_comms:
                    if (
                        bc.is_current_rank_in_grid(bc.src_grid)
                        and bc.src_module_name == module_name
                    ):
                        bridge_as_src.append(bc)

            self.rank_module_map[module_name] = RankModuleInfo(
                pp_rank=pp_rank,
                pp_size=pp_size,
                p2p_communicator=p2p_comm,
                bridge_comms_as_dest_module=bridge_as_dest,
                bridge_comms_as_src_module=bridge_as_src,
            )

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def is_pp_first_stage(self) -> bool:
        """True if this rank holds the absolute first PP stage of the model."""
        for name, info in self.rank_module_map.items():
            if info.pp_rank == 0 and self._is_source_module(name):
                return True
        return False

    @property
    def is_pp_last_stage(self) -> bool:
        """True if this rank holds the absolute last PP stage of the model."""
        for name, info in self.rank_module_map.items():
            if info.pp_rank == info.pp_size - 1 and self._is_sink_module(name):
                return True
        return False

    def _is_source_module(self, module_name: str) -> bool:
        """True if no other module lists *module_name* as a destination."""
        for dest_list in self.topology.values():
            if module_name in dest_list:
                return False
        return True

    def _is_sink_module(self, module_name: str) -> bool:
        """True if *module_name* has no outgoing edges."""
        return len(self.topology.get(module_name, [])) == 0

    def is_current_rank_in_grid(self, grid) -> bool:
        return grid.rank_offset <= self.current_rank < (grid.rank_offset + grid.size)

    @property
    def total_stages(self) -> int:
        """Longest-path total PP stages across the module DAG."""
        return self.compute_total_pipeline_stages(self.topology, self.module_to_grid_map)

    @property
    def current_stage(self) -> int:
        """0-based stage index of this rank in the multi-module pipeline."""
        total = self.total_stages
        if self.rank_module_map:
            module_name = next(iter(self.rank_module_map.keys()))
            stage = (
                self.compute_total_pipeline_stages(
                    self.topology,
                    self.module_to_grid_map,
                    rank=self.current_rank,
                    module_name=module_name,
                )
                - 1
            )
        else:
            stage = 0
        assert stage < total, (
            f"current_stage: {stage} must be less than total_stages: {total}"
        )
        return stage

    # -----------------------------------------------------------------
    # Forward communication
    # -----------------------------------------------------------------

    def recv_forward(
        self,
        tensor_shape: Optional[Shape] = None,
        is_first_stage: bool = False,
    ) -> Dict[str, torch.Tensor]:
        input_dict: Dict[str, torch.Tensor] = {}
        for module_name, info in self.rank_module_map.items():
            if info.pp_rank == 0:
                for bc in info.bridge_comms_as_dest_module:
                    t = bc.recv_forward()
                    input_dict[bc.src_module_name] = t
            else:
                t = info.p2p_communicator.recv_forward(
                    tensor_shapes=tensor_shape, is_first_stage=False
                )
                input_dict[module_name] = _restore_tensor_from_comm(t)
        return input_dict

    def send_forward(
        self,
        output_dict: Dict[str, torch.Tensor],
        is_last_stage: bool = False,
    ) -> None:
        for module_name, info in self.rank_module_map.items():
            if info.pp_rank == info.pp_size - 1:
                for bc in info.bridge_comms_as_src_module:
                    bc.send_forward(output_dict[module_name])
            else:
                t = _prepare_tensor_for_comm(output_dict[module_name])
                info.p2p_communicator.send_forward(t, is_last_stage=False)

    # -----------------------------------------------------------------
    # Fused forward+backward communication
    # -----------------------------------------------------------------

    def send_forward_recv_backward(
        self,
        output_dict: Dict[str, torch.Tensor],
        tensor_shape: Optional[Shape] = None,
        is_last_stage: bool = False,
    ) -> Dict[str, torch.Tensor]:
        grad_dict: Dict[str, torch.Tensor] = {}
        for module_name, info in self.rank_module_map.items():
            if info.pp_rank == info.pp_size - 1:
                for bc in info.bridge_comms_as_src_module:
                    g = bc.send_forward_recv_backward(output_dict[module_name])
                    grad_dict[bc.src_module_name] = g
            else:
                t = _prepare_tensor_for_comm(output_dict[module_name])
                g = info.p2p_communicator.send_forward_recv_backward(
                    t, tensor_shapes=tensor_shape, is_last_stage=False
                )
                grad_dict[module_name] = _restore_tensor_from_comm(g)
        return grad_dict

    def send_backward_recv_forward(
        self,
        grad_dict: Dict[str, torch.Tensor],
        tensor_shape: Optional[Shape] = None,
        is_first_stage: bool = False,
    ) -> Dict[str, torch.Tensor]:
        input_dict: Dict[str, torch.Tensor] = {}
        for module_name, info in self.rank_module_map.items():
            if info.pp_rank == 0:
                for bc in info.bridge_comms_as_dest_module:
                    t = bc.send_backward_recv_forward(grad_dict[bc.src_module_name])
                    input_dict[bc.src_module_name] = t
            else:
                g = _prepare_tensor_for_comm(grad_dict[module_name])
                t = info.p2p_communicator.send_backward_recv_forward(
                    g, tensor_shapes=tensor_shape, is_first_stage=False
                )
                input_dict[module_name] = _restore_tensor_from_comm(t)
        return input_dict

    # -----------------------------------------------------------------
    # Backward communication
    # -----------------------------------------------------------------

    def recv_backward(
        self,
        tensor_shape: Optional[Shape] = None,
        is_last_stage: bool = False,
    ) -> Dict[str, torch.Tensor]:
        grad_dict: Dict[str, torch.Tensor] = {}
        for module_name, info in self.rank_module_map.items():
            if info.pp_rank == info.pp_size - 1:
                for bc in info.bridge_comms_as_src_module:
                    g = bc.recv_backward()
                    grad_dict[bc.src_module_name] = g
            else:
                g = info.p2p_communicator.recv_backward(
                    tensor_shapes=tensor_shape, is_last_stage=False
                )
                grad_dict[module_name] = _restore_tensor_from_comm(g)
        return grad_dict

    def send_backward(
        self,
        grad_dict: Dict[str, torch.Tensor],
        is_first_stage: bool = False,
    ) -> None:
        for module_name, info in self.rank_module_map.items():
            if info.pp_rank == 0:
                for bc in info.bridge_comms_as_dest_module:
                    bc.send_backward(grad_dict[bc.src_module_name])
            else:
                g = _prepare_tensor_for_comm(grad_dict[module_name])
                info.p2p_communicator.send_backward(g, is_first_stage=False)

    # -----------------------------------------------------------------
    # Total-stage computation (longest path in the module DAG)
    # -----------------------------------------------------------------

    @staticmethod
    def compute_total_pipeline_stages(
        topology: Dict[str, List[str]],
        module_to_grid_map: Dict[str, object],
        rank: Optional[int] = None,
        module_name: Optional[str] = None,
    ) -> int:
        """Compute total PP stages as the longest weighted path in the DAG.

        When *rank* is ``None``, returns the global maximum.  When *rank* is
        given, returns the path length up to and including *rank*'s position
        inside *module_name*.
        """
        nodes = set(module_to_grid_map.keys())
        adj: Dict[str, List[str]] = {n: list(topology.get(n, [])) for n in nodes}
        preds: Dict[str, List[str]] = {n: [] for n in nodes}
        for src, outs in adj.items():
            for dst in outs:
                preds[dst].append(src)

        sinks = [n for n, outs in adj.items() if not outs]
        if rank is None and not sinks:
            raise ValueError(
                "Topology must be a DAG with at least one terminal module."
            )

        def pp_size(name: str) -> int:
            grid = module_to_grid_map[name]
            pp_dim = grid.dim_names.index("pp")
            return grid.shape[pp_dim]

        def partial_weight(target: str) -> Optional[int]:
            if rank is None:
                return None
            grid = module_to_grid_map.get(target)
            groups = grid._gen_rank_enum(["pp"])
            for group in groups:
                if rank in group:
                    return group.index(rank) + 1
            return None

        def longest_path_to(target: str) -> int:
            visiting: set = set()
            pw = partial_weight(target)

            def weight(name: str) -> int:
                if pw is not None and name == target:
                    return pw
                return pp_size(name)

            def dfs(node: str) -> int:
                if node in visiting:
                    raise ValueError("Topology contains cycles; expected a DAG.")
                visiting.add(node)
                best = 0
                for p in preds.get(node, []):
                    val = dfs(p)
                    if val > best:
                        best = val
                visiting.remove(node)
                return weight(node) + best

            return dfs(target)

        if rank is None:
            return max(longest_path_to(s) for s in sinks)
        return longest_path_to(module_name)
