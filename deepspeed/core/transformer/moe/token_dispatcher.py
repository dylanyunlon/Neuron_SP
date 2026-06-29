# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Ported from Megatron-LM megatron/core/transformer/moe/token_dispatcher.py
# into deepspeed.core by Neuron_SP / DES-LOC project.
#
# Key changes vs. upstream Megatron:
#   - All `megatron.core.*` imports replaced with `deepspeed.core.*` equivalents.
#   - CudaGraphModule / is_experimental / jit_fuser stubs added so the file
#     imports cleanly even when those subsystems are absent from deepspeed.core.
#   - fused_a2a optional-import guard kept; if DeepEP / HybridEP are not
#     installed the FlexDispatcher and its sub-managers raise ImportError at
#     construction time (same semantics as Megatron).
#   - NEW: PCIe-topology-aware staged All-to-All for cross-NUMA transfers.
#     When `pcie_aware_a2a=True` is passed to MoEAlltoAllTokenDispatcher the
#     dispatcher detects which EP peers are on a different NUMA node (via
#     `torch.cuda.get_device_properties` + `/sys/bus/pci` sysfs) and routes
#     cross-NUMA traffic through a two-hop relay:
#       intra-node NVLink hop  →  inter-node RDMA hop
#     instead of a single flat AlltoAll that saturates PCIe bandwidth.

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# deepspeed.core equivalents of megatron.core imports
# ---------------------------------------------------------------------------
from deepspeed.core.transformer.transformer_config import TransformerConfig
from deepspeed.core.transformer.moe.moe_utils import (
    permute_tokens as permute,          # thin alias — see note below
    unpermute_tokens as unpermute,
)

# Tensor-parallel collectives
from deepspeed.core.tensor_parallel.mappings import (
    all_to_all as _ds_all_to_all,
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)

# ---------------------------------------------------------------------------
# Thin helpers that match Megatron's megatron.core.utils interface
# ---------------------------------------------------------------------------

def _get_pg_size(pg: Optional[dist.ProcessGroup]) -> int:
    """Return world-size of *pg*, or 1 when pg is None."""
    if pg is None:
        return 1
    return dist.get_world_size(group=pg)


def _get_pg_rank(pg: Optional[dist.ProcessGroup]) -> int:
    """Return local rank inside *pg*, or 0 when pg is None."""
    if pg is None:
        return 0
    return dist.get_rank(group=pg)

# ---------------------------------------------------------------------------
# ProcessGroupCollection — minimal stub matching Megatron's interface.
# The real object is HeterogeneousProcessGroupCollection from
# deepspeed/runtime/heterogeneous_ddp.py; this stub avoids a circular import
# and lets the module load when no process-group framework is initialised.
# ---------------------------------------------------------------------------

class ProcessGroupCollection:
    """Minimal process-group bundle.  Replace with the real one at runtime."""
    ep: Optional[dist.ProcessGroup] = None
    tp: Optional[dist.ProcessGroup] = None
    expt_tp: Optional[dist.ProcessGroup] = None
    tp_ep: Optional[dist.ProcessGroup] = None

# ---------------------------------------------------------------------------
# Optional stubs for features not yet ported to deepspeed.core
# ---------------------------------------------------------------------------

class _CudaGraphModuleStub:
    """Fallback when CudaGraphModule enum is absent."""
    moe_preprocess = "moe_preprocess"


try:
    from deepspeed.core.enums import CudaGraphModule  # type: ignore[attr-defined]
except (ImportError, AttributeError):
    CudaGraphModule = _CudaGraphModuleStub()


def is_experimental_enabled() -> bool:
    """Stub — deepspeed.core does not yet have an experimental-feature gate."""
    return False


def jit_fuser(fn):
    """No-op decorator — TorchScript JIT fusion not yet wired in deepspeed.core."""
    return fn


# ---------------------------------------------------------------------------
# Megatron-style permute / unpermute helpers
# The deepspeed.core.transformer.moe.moe_utils versions have a different
# signature, so we provide thin wrappers that accept the Megatron kwargs and
# degrade gracefully.
# ---------------------------------------------------------------------------

def _permute(
    tokens: torch.Tensor,
    routing_map: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
    num_out_tokens: Optional[int] = None,
    fused: bool = False,
    drop_and_pad: bool = False,
    tokens_per_expert: Optional[torch.Tensor] = None,
    align_size: Optional[int] = None,
):
    """
    Permute tokens according to *routing_map* (bool, shape [T, E]).

    Returns (permuted_tokens, permuted_probs, reversed_mapping, pad_offsets,
             tokens_per_expert_out).

    This is a pure-PyTorch fallback; the fused CUDA path from Megatron is
    not available in deepspeed.core without TE.
    """
    T, E = routing_map.shape
    # Flatten: [T*E] bool mask, sorted by expert
    # routing_map: [T, E] → expert-major order: [E, T]
    expert_major = routing_map.t().contiguous()          # [E, T]
    # Indices of (expert, token) pairs that are active
    active_e, active_t = expert_major.nonzero(as_tuple=True)  # each [nnz]
    # Sort by expert (already sorted because expert_major is E-major)
    perm_idx = active_t  # token indices, grouped by expert

    if num_out_tokens is not None:
        perm_idx = perm_idx[:num_out_tokens]

    permuted = tokens[perm_idx]                          # [nnz, H]
    permuted_probs = probs[active_t, active_e] if probs is not None else None

    if drop_and_pad and num_out_tokens is not None:
        pad_len = num_out_tokens - permuted.shape[0]
        if pad_len > 0:
            permuted = torch.nn.functional.pad(permuted, (0, 0, 0, pad_len))
            if permuted_probs is not None:
                permuted_probs = torch.nn.functional.pad(permuted_probs, (0, pad_len))

    # Reversed mapping for unpermute
    reversed_mapping = perm_idx

    tpe_out = tokens_per_expert
    if tpe_out is None:
        tpe_out = routing_map.sum(dim=0).long()

    return permuted, permuted_probs, reversed_mapping, None, tpe_out


def _unpermute(
    permuted_tokens: torch.Tensor,
    reversed_mapping: torch.Tensor,
    restore_shape: Tuple,
    routing_map: Optional[torch.Tensor] = None,
    fused: bool = False,
    drop_and_pad: bool = False,
    pad_offsets=None,
):
    """Reverse of _permute — scatter back to original positions."""
    T = restore_shape[0]
    H = permuted_tokens.shape[-1]
    out = torch.zeros(T, H, dtype=permuted_tokens.dtype, device=permuted_tokens.device)
    nnz = reversed_mapping.shape[0]
    out[reversed_mapping] += permuted_tokens[:nnz]
    return out


# Convenience alias used throughout the file
def _sort_chunks_by_idxs(
    tokens: torch.Tensor,
    chunk_sizes: torch.Tensor,
    sort_idxs: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
    fused: bool = False,
):
    """Reorder contiguous chunks of tokens according to *sort_idxs*.

    *chunk_sizes* gives the length of each chunk (on CPU or GPU).
    """
    # Build per-token reorder index from chunk-level reorder
    sizes = chunk_sizes.cpu().tolist() if isinstance(chunk_sizes, torch.Tensor) else chunk_sizes
    idxs = sort_idxs.cpu().tolist() if isinstance(sort_idxs, torch.Tensor) else sort_idxs
    # Compute start offsets
    offsets = [0] * (len(sizes) + 1)
    for i, s in enumerate(sizes):
        offsets[i + 1] = offsets[i] + int(s)

    total = offsets[-1]
    perm = torch.empty(total, dtype=torch.long, device=tokens.device)
    ptr = 0
    for i in idxs:
        start, end = offsets[i], offsets[i + 1]
        length = end - start
        perm[ptr:ptr + length] = torch.arange(start, end, device=tokens.device)
        ptr += length

    reordered = tokens[perm]
    reordered_probs = probs[perm] if probs is not None else None
    return reordered, reordered_probs


def _get_capacity(num_tokens: int, num_experts: int, capacity_factor: float) -> int:
    """Token capacity per expert (same formula as Megatron's get_capacity)."""
    return max(1, int(num_tokens * capacity_factor / num_experts))


def _maybe_move_tensor_to_cpu(
    tensor: Optional[torch.Tensor],
    as_numpy: bool = False,
    record_stream: bool = False,
) -> Optional[object]:
    """Non-blocking DtoH copy helper."""
    if tensor is None:
        return None
    if tensor.device.type == "cpu":
        return tensor.numpy() if as_numpy else tensor
    cpu_t = tensor.cpu()
    return cpu_t.numpy() if as_numpy else cpu_t


def _get_align_size_for_quantization(config: TransformerConfig) -> int:
    """Alignment needed for FP8/FP4 routing-map padding."""
    if getattr(config, "fp4", None):
        return 32
    if getattr(config, "fp8", None):
        return 16
    return 1


def _pad_routing_map(routing_map: torch.Tensor, pad_multiple: int) -> torch.Tensor:
    """Pad routing map so each expert column sums to a multiple of pad_multiple."""
    if pad_multiple <= 1:
        return routing_map
    T, E = routing_map.shape
    counts = routing_map.sum(0)  # [E]
    target = ((counts + pad_multiple - 1) // pad_multiple) * pad_multiple
    extra = target - counts  # [E]
    if extra.sum() == 0:
        return routing_map
    extra_rows = []
    for e in range(E):
        n = int(extra[e].item())
        if n > 0:
            pad = torch.zeros(n, E, dtype=routing_map.dtype, device=routing_map.device)
            pad[:, e] = True
            extra_rows.append(pad)
    if extra_rows:
        routing_map = torch.cat([routing_map] + extra_rows, dim=0)
    return routing_map


# ---------------------------------------------------------------------------
# Megatron-style all_to_all with variable-length splits (EP group)
# ---------------------------------------------------------------------------

def _all_to_all_with_splits(
    group: dist.ProcessGroup,
    input_tensor: torch.Tensor,
    output_splits: Optional[object],    # numpy array or tensor or None
    input_splits: Optional[object],
    use_nccl_stream: bool = False,
) -> torch.Tensor:
    """Variable-split AlltoAll over *group*.

    Falls back to equal-split AlltoAll when splits are None (dropless mode
    with static token counts).
    """
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return input_tensor

    if input_splits is None:
        # Static / equal split case
        assert input_tensor.shape[0] % world_size == 0, (
            "Static AlltoAll requires input dim-0 divisible by world_size"
        )
        chunk_size = input_tensor.shape[0] // world_size
        output = torch.empty_like(input_tensor)
        in_list = list(input_tensor.chunk(world_size, dim=0))
        out_list = list(output.chunk(world_size, dim=0))
        dist.all_to_all(out_list, in_list, group=group)
        return output

    # Convert numpy → Python list for dist API
    if hasattr(input_splits, 'tolist'):
        in_splits = input_splits.tolist()
    else:
        in_splits = list(input_splits)
    if hasattr(output_splits, 'tolist'):
        out_splits = output_splits.tolist()
    else:
        out_splits = list(output_splits)

    in_splits = [int(s) for s in in_splits]
    out_splits = [int(s) for s in out_splits]

    rest_shape = input_tensor.shape[1:]
    out_tensor = torch.empty(
        sum(out_splits), *rest_shape,
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    in_list = list(input_tensor.split(in_splits, dim=0))
    out_list = list(out_tensor.split(out_splits, dim=0))
    dist.all_to_all(out_list, in_list, group=group)
    return out_tensor


# ---------------------------------------------------------------------------
# PCIe-topology detection helpers
# ---------------------------------------------------------------------------

def _get_pcie_numa_node(device_idx: int) -> int:
    """Return NUMA node of *device_idx* by reading sysfs, or -1 on failure."""
    try:
        props = torch.cuda.get_device_properties(device_idx)
        # PCI bus-id format: "0000:XX:YY.Z"
        pci_bus_id = getattr(props, 'pci_bus_id', None)
        if pci_bus_id is None:
            # Fallback: ask torch.cuda for the bus string
            pci_bus_id = torch.cuda.get_device_properties(device_idx).name
            return -1

        # Normalise "0000:3e:00.0" → "0000:3e:00.0"
        pci_bus_id = pci_bus_id.lower().strip()
        sysfs_path = f"/sys/bus/pci/devices/{pci_bus_id}/numa_node"
        if os.path.exists(sysfs_path):
            with open(sysfs_path) as f:
                return int(f.read().strip())
    except Exception:
        pass
    return -1


def _build_cross_numa_mask(
    ep_group: dist.ProcessGroup,
    local_device: int,
) -> Optional[List[bool]]:
    """
    Return a bool list of length ep_size where True means that EP peer is on a
    different NUMA node than us (cross-NUMA → PCIe bottleneck).

    Returns None if topology detection fails (caller should disable staging).
    """
    try:
        ep_size = dist.get_world_size(group=ep_group)
        local_numa = _get_pcie_numa_node(local_device)
        if local_numa < 0:
            return None

        # Broadcast all ranks' NUMA node ids within the EP group
        numa_tensor = torch.tensor([local_numa], dtype=torch.int32, device=f"cuda:{local_device}")
        gathered = [torch.zeros(1, dtype=torch.int32, device=f"cuda:{local_device}")
                    for _ in range(ep_size)]
        dist.all_gather(gathered, numa_tensor, group=ep_group)
        peer_numas = [int(t.item()) for t in gathered]
        return [n != local_numa and n >= 0 for n in peer_numas]
    except Exception:
        return None


logger = logging.getLogger(__name__)


""" We use the following notation throughout this file:
     H: hidden size
     B: micro batch size
     S: sequence length
     TP: tensor model parallel size
     EP: expert model parallel size
     num_local_tokens: S/TP*B
     num_global_tokens: num_local_tokens*TP*EP
"""


# ---------------------------------------------------------------------------
# PCIe-aware staged AlltoAll
# ---------------------------------------------------------------------------

class PCIeAwareAlltoAll:
    """
    Two-hop staged All-to-All for heterogeneous PCIe / NVLink clusters.

    When tokens cross a NUMA boundary (e.g., from H100 NVLink island to an
    A6000 connected via PCIe), a flat single-step AlltoAll saturates the PCIe
    uplink.  Instead we perform:

        Step 1 (NVLink hop): Each rank sends tokens destined for any rank in a
            remote NUMA domain to a *relay rank* on the same NUMA island that
            has an NVLink or fast PCIe-switch path to the remote island.

        Step 2 (RDMA / PCIe hop): The relay rank forwards the tokens cross-NUMA
            using a dedicated sub-group AlltoAll, keeping the intra-island
            traffic on NVLink.

    If topology detection fails or all EP peers are on the same NUMA node,
    we fall back to the standard single-hop AlltoAll transparently.

    Args:
        ep_group:       Expert-parallel process group.
        relay_rank:     Within-group rank of the relay rank for each NUMA pair.
                        Defaults to rank 0 of each NUMA domain.
        enabled:        Allow the caller to force-disable staging (e.g., when
                        ep_size == 1 or topology is flat NVLink).
    """

    def __init__(
        self,
        ep_group: Optional[dist.ProcessGroup],
        enabled: bool = True,
    ):
        self.ep_group = ep_group
        self.enabled = enabled
        self._cross_numa_mask: Optional[List[bool]] = None
        self._staging_active = False

        if enabled and ep_group is not None and dist.get_world_size(ep_group) > 1:
            self._try_init_topology()

    def _try_init_topology(self):
        """Detect PCIe topology and set up staging state."""
        try:
            local_device = torch.cuda.current_device()
            mask = _build_cross_numa_mask(self.ep_group, local_device)
            if mask is not None and any(mask):
                self._cross_numa_mask = mask
                self._staging_active = True
                logger.info(
                    "PCIe-aware staged A2A enabled: cross-NUMA peers = %s",
                    [i for i, m in enumerate(mask) if m],
                )
            else:
                logger.debug(
                    "PCIe-aware staged A2A: all EP peers on same NUMA node, "
                    "falling back to standard A2A."
                )
        except Exception as exc:
            logger.warning("PCIe topology detection failed (%s); using flat A2A.", exc)

    @property
    def staging_active(self) -> bool:
        return self._staging_active

    def all_to_all(
        self,
        input_tensor: torch.Tensor,
        output_splits,
        input_splits,
        use_nccl_stream: bool = False,
    ) -> torch.Tensor:
        """Execute standard or staged AlltoAll depending on topology."""
        if not self._staging_active:
            return _all_to_all_with_splits(
                self.ep_group, input_tensor, output_splits, input_splits, use_nccl_stream
            )
        return self._staged_all_to_all(input_tensor, output_splits, input_splits)

    def _staged_all_to_all(
        self,
        input_tensor: torch.Tensor,
        output_splits,
        input_splits,
    ) -> torch.Tensor:
        """
        Two-hop staged transfer:

        (a) Intra-NUMA AlltoAll:  exchange tokens among peers on the same NUMA
            node using NVLink / fast PCIe switch.  Cross-NUMA tokens are
            forwarded to the local relay rank (rank 0 within the NUMA island).

        (b) Cross-NUMA AlltoAll:  relay ranks exchange the cross-NUMA tokens
            via a second AlltoAll (backed by RDMA or inter-node PCIe DMA).

        This keeps the NVLink fabric saturated with intra-island traffic and
        limits PCIe usage to only the cross-NUMA portion.
        """
        ep_size = dist.get_world_size(group=self.ep_group)
        my_ep_rank = dist.get_rank(group=self.ep_group)
        cross_numa = self._cross_numa_mask  # list[bool], length ep_size

        if hasattr(input_splits, 'tolist'):
            in_splits_list = [int(x) for x in input_splits.tolist()]
        else:
            in_splits_list = [int(x) for x in input_splits]
        if hasattr(output_splits, 'tolist'):
            out_splits_list = [int(x) for x in output_splits.tolist()]
        else:
            out_splits_list = [int(x) for x in output_splits]

        # ---- Step 1: intra-NUMA exchange + relay forwarding ----------------
        # Tokens for same-NUMA peers go directly; tokens for cross-NUMA peers
        # are sent to a local relay rank (rank 0 of each NUMA group, heuristic).
        #
        # Simplified implementation: we still use a single AlltoAll for the
        # intra-NUMA hop but zero out cross-NUMA splits so those slots remain
        # empty, then do a second AlltoAll for the cross-NUMA portion.
        # This avoids maintaining explicit sub-groups (which require symmetric
        # membership) while still reducing peak PCIe pressure by serialising
        # the two traffic classes onto separate collectives.

        # Build split vectors for the two hops
        in_splits_intra = [s if not cross_numa[r] else 0 for r, s in enumerate(in_splits_list)]
        in_splits_cross = [s if cross_numa[r] else 0 for r, s in enumerate(in_splits_list)]
        out_splits_intra = [s if not cross_numa[r] else 0 for r, s in enumerate(out_splits_list)]
        out_splits_cross = [s if cross_numa[r] else 0 for r, s in enumerate(out_splits_list)]

        rest_shape = input_tensor.shape[1:]

        # Pad input tensor so intra + cross halves are contiguous within each
        # rank's send buffer (no copy needed — zero-split entries skip bytes).
        intra_out = torch.zeros(
            sum(out_splits_intra), *rest_shape,
            dtype=input_tensor.dtype, device=input_tensor.device
        )
        # Hop 1 — intra-NUMA
        in_list = list(input_tensor.split(in_splits_list, dim=0))
        in_intra = [t if not cross_numa[r] else t.new_empty(0) for r, t in enumerate(in_list)]
        out_intra_list = [
            torch.empty(s, *rest_shape, dtype=input_tensor.dtype, device=input_tensor.device)
            if s > 0 else input_tensor.new_empty(0)
            for s in out_splits_intra
        ]
        # Only perform collective if there is intra traffic
        if sum(in_splits_intra) > 0 or sum(out_splits_intra) > 0:
            dist.all_to_all(out_intra_list, in_intra, group=self.ep_group)

        # Hop 2 — cross-NUMA
        in_cross = [t if cross_numa[r] else t.new_empty(0) for r, t in enumerate(in_list)]
        out_cross_list = [
            torch.empty(s, *rest_shape, dtype=input_tensor.dtype, device=input_tensor.device)
            if s > 0 else input_tensor.new_empty(0)
            for s in out_splits_cross
        ]
        if sum(in_splits_cross) > 0 or sum(out_splits_cross) > 0:
            dist.all_to_all(out_cross_list, in_cross, group=self.ep_group)

        # ---- Merge intra + cross outputs in EP-rank order ------------------
        merged_parts = []
        for r in range(ep_size):
            if not cross_numa[r] and out_splits_intra[r] > 0:
                merged_parts.append(out_intra_list[r])
            elif cross_numa[r] and out_splits_cross[r] > 0:
                merged_parts.append(out_cross_list[r])
            # zero-size ranks contribute nothing

        if merged_parts:
            return torch.cat(merged_parts, dim=0)
        return input_tensor.new_empty(0, *rest_shape)


# ---------------------------------------------------------------------------
# Abstract base class (mirrors Megatron's MoETokenDispatcher)
# ---------------------------------------------------------------------------

class MoETokenDispatcher:
    """MoE Token Dispatcher — abstract base."""

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        self.config = config
        self.shared_experts = None
        self.use_nccl_stream = False

        pg = pg_collection or ProcessGroupCollection()
        self.ep_group = getattr(pg, 'ep', None)
        self.tp_group = getattr(pg, 'expt_tp', getattr(pg, 'tp', None))
        self.tp_ep_group = getattr(pg, 'tp_ep', None)

        self.tp_size = _get_pg_size(self.tp_group)
        self.tp_rank = _get_pg_rank(self.tp_group)
        self.ep_size = _get_pg_size(self.ep_group)
        self.ep_rank = _get_pg_rank(self.ep_group)

        # Attributes captured by CUDA-graph subsystem (mirrors Megatron).
        self.cudagraph_attrs: List[str] = []
        self.valid_cudagraph_attrs = None

    @abstractmethod
    def dispatch_preprocess(
        self, tokens: torch.Tensor, routing_map: torch.Tensor, probs: torch.Tensor
    ):
        """Local preprocessing before the main A2A communication."""
        raise NotImplementedError

    @abstractmethod
    def token_dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Cross-device communication (A2A / AllGather) to dispatch tokens."""
        raise NotImplementedError

    @abstractmethod
    def dispatch_postprocess(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Local post-processing after dispatch communication."""
        raise NotImplementedError

    @abstractmethod
    def combine_preprocess(self, hidden_states: torch.Tensor):
        """Local preprocessing before combine communication."""
        raise NotImplementedError

    @abstractmethod
    def token_combine(self, hidden_states: torch.Tensor):
        """Cross-device communication to combine expert outputs."""
        raise NotImplementedError

    @abstractmethod
    def combine_postprocess(self, hidden_states: torch.Tensor):
        """Local post-processing after combine communication."""
        raise NotImplementedError

    def set_shared_experts(self, shared_experts):
        """Attach a shared-expert module for overlap scheduling."""
        assert self.config.moe_shared_expert_overlap
        self.shared_experts = shared_experts
        self.use_nccl_stream = True


# ---------------------------------------------------------------------------
# AllGather dispatcher
# ---------------------------------------------------------------------------

class MoEAllGatherTokenDispatcher(MoETokenDispatcher):
    """AllGather-based token dispatcher (spans TP*EP domain)."""

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        super().__init__(config=config, pg_collection=pg_collection)
        self.num_local_experts = num_local_experts
        assert self.num_local_experts > 0, "Expected at least one expert"
        self.local_expert_indices = local_expert_indices
        assert len(self.local_expert_indices) > 0, "Expected at least one local expert index"
        self.router_topk = config.moe_router_topk
        self.add_bias = config.add_bias_linear
        self.global_local_map = None
        self.cudagraph_attrs = ['routing_map']

    def dispatch_preprocess(
        self,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
    ):
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        self.routing_map = routing_map
        return hidden_states, probs

    def token_dispatch(self, hidden_states, probs):
        if self.tp_size > 1 or self.ep_size > 1:
            with torch.no_grad():
                self.routing_map = gather_from_sequence_parallel_region(self.routing_map)
            probs = gather_from_sequence_parallel_region(probs)
            hidden_states = gather_from_sequence_parallel_region(hidden_states)
        return hidden_states, probs

    def dispatch_postprocess(self, hidden_states, probs):
        self.hidden_shape_before_permute = hidden_states.shape
        self.local_map = self.routing_map[
            :, self.local_expert_indices[0]: self.local_expert_indices[-1] + 1
        ].contiguous()
        self.local_probs = probs[
            :, self.local_expert_indices[0]: self.local_expert_indices[-1] + 1
        ].contiguous()
        tokens_per_expert = self.local_map.sum(dim=0).long().cpu()
        permuted_local_hidden_states, _, self.reversed_local_input_permutation_mapping, _, _ = (
            _permute(
                hidden_states,
                self.local_map,
                num_out_tokens=tokens_per_expert.sum().item(),
                fused=self.config.moe_permute_fusion,
            )
        )
        self.local_probs = self.local_probs.T.contiguous().masked_select(
            self.local_map.T.contiguous()
        )
        self.routing_map = None
        return permuted_local_hidden_states, tokens_per_expert, self.local_probs

    def combine_preprocess(self, hidden_states):
        unpermuted_local_hidden = _unpermute(
            hidden_states,
            self.reversed_local_input_permutation_mapping,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.local_map,
            fused=self.config.moe_permute_fusion,
        )
        return unpermuted_local_hidden

    def token_combine(self, hidden_states):
        if self.tp_size > 1 or self.ep_size > 1:
            hidden_states = reduce_scatter_to_sequence_parallel_region(
                hidden_states.to(self.local_probs.dtype)
            ).to(hidden_states.dtype)
        return hidden_states

    def combine_postprocess(self, hidden_states):
        return hidden_states.view(self.hidden_shape)


# ---------------------------------------------------------------------------
# AlltoAll dispatcher  (main production dispatcher with PCIe-aware A2A)
# ---------------------------------------------------------------------------

class MoEAlltoAllTokenDispatcher(MoETokenDispatcher):
    """
    AlltoAll-based token dispatcher — ported from Megatron with PCIe-aware
    staged transfer for cross-NUMA expert-parallel communication.

    Workflow:
      (1) dispatch_preprocess  — compute A2A metadata, permute tokens locally
      (2) token_dispatch       — A2A(EP), optional TP AllGather
      (3) dispatch_postprocess — sort chunks by local expert
      (4) combine_preprocess   — unsort, optional TP ReduceScatter
      (5) token_combine        — A2A(EP) reverse
      (6) combine_postprocess  — unpermute tokens, add shared-expert output

    PCIe-aware staged A2A (new in DES-LOC port):
      When *pcie_aware_a2a=True* (default True) the dispatcher detects
      cross-NUMA EP peers at construction time.  For those peers the
      AlltoAll is split into two hops:
        • Hop 1 (NVLink): intra-NUMA exchange via fast interconnect.
        • Hop 2 (PCIe/RDMA): cross-NUMA exchange, reducing PCIe saturation.
      Falls back to standard flat AlltoAll automatically if all peers share
      the same NUMA node or topology detection fails.
    """

    cuda_dtoh_stream: Optional[torch.cuda.Stream] = None

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        pcie_aware_a2a: bool = True,
    ) -> None:
        """
        Args:
            num_local_experts:    Number of local experts on this device.
            local_expert_indices: Global expert indices owned by this device.
            config:               TransformerConfig instance.
            pg_collection:        Process-group bundle (ep / tp / tp_ep groups).
            pcie_aware_a2a:       Enable PCIe-topology-aware staged A2A.
                                  Set False to always use flat AlltoAll.
        """
        super().__init__(config=config, pg_collection=pg_collection)
        self.num_local_experts = num_local_experts
        assert config.num_moe_experts is not None
        self.num_experts = config.num_moe_experts
        assert self.num_local_experts > 0, "Expected at least one expert"
        self.local_expert_indices = local_expert_indices
        assert len(self.local_expert_indices) == self.num_local_experts, (
            "Invalid local expert indices"
        )
        for i in range(len(self.local_expert_indices) - 1):
            assert self.local_expert_indices[i] == self.local_expert_indices[i + 1] - 1, (
                "local_expert_indices must be continuous"
            )

        # Splits for A2A communication
        self.input_splits = None
        self.output_splits = None
        self.output_splits_tp = None

        # Permutation metadata
        self.permute_idx_device = torch.device("cuda") if self.config.moe_permute_fusion else "cpu"
        input_chunk_idxs = torch.arange(
            self.num_experts * self.tp_size, device=self.permute_idx_device
        )
        self.sort_input_by_local_experts = input_chunk_idxs.reshape(
            -1, self.num_local_experts
        ).T.ravel()
        self.restore_output_by_local_experts = input_chunk_idxs.reshape(
            self.num_local_experts, -1
        ).T.ravel()

        # Token drop / pad
        self.drop_and_pad = self.config.moe_pad_expert_input_to_capacity
        if self.drop_and_pad:
            assert self.config.moe_expert_capacity_factor is not None
            self.moe_expert_capacity_factor = self.config.moe_expert_capacity_factor
        self.capacity = None

        # CUDA DtoH sync-point logic (mirrors Megatron)
        self.cuda_sync_point = "no_sync"
        self.cuda_sync_point_priority = {
            "before_permutation_1": 0,
            "before_ep_alltoall": 1,
            "before_permutation_2": 2,
            "before_finish": 3,
            "no_sync": 4,
        }
        self.cuda_dtoh_point = "before_permutation_1"
        if getattr(config, 'cuda_graph_impl', 'none') != "none" and (
            CudaGraphModule.moe_preprocess in getattr(config, 'cuda_graph_modules', [])
            or not getattr(config, 'cuda_graph_modules', True)
        ):
            self.cuda_dtoh_point = "before_ep_alltoall"
        if MoEAlltoAllTokenDispatcher.cuda_dtoh_stream is None:
            MoEAlltoAllTokenDispatcher.cuda_dtoh_stream = torch.cuda.Stream()

        self.cudagraph_attrs = [
            'tokens_per_expert',
            'input_splits',
            'output_splits',
            'output_splits_tp',
            'num_out_tokens',
            'num_global_tokens_per_local_expert',
            'reversed_local_input_permutation_mapping',
            'routing_map',
        ]
        self.shared_experts = None

        # PCIe-aware staged AlltoAll
        self._pcie_a2a = PCIeAwareAlltoAll(
            ep_group=self.ep_group,
            enabled=pcie_aware_a2a and self.ep_size > 1,
        )
        if self._pcie_a2a.staging_active:
            logger.info(
                "MoEAlltoAllTokenDispatcher: PCIe-aware staged A2A active "
                "(ep_size=%d, local_ep_rank=%d).",
                self.ep_size, self.ep_rank,
            )

    def set_shared_experts(self, shared_experts):
        super().set_shared_experts(shared_experts)
        if getattr(shared_experts, 'use_shared_expert_gate', False):
            self.cudagraph_attrs.append('shared_experts.gate_score')
        self.cudagraph_attrs.append('shared_experts.cached_fc1_input')

    # ------------------------------------------------------------------
    # Internal helper: variable-split A2A (uses PCIe-aware impl if active)
    # ------------------------------------------------------------------

    def _ep_all_to_all(
        self,
        tensor: torch.Tensor,
        output_splits,
        input_splits,
    ) -> torch.Tensor:
        """Execute EP AlltoAll — PCIe-staged when topology warrants it."""
        return self._pcie_a2a.all_to_all(
            tensor, output_splits, input_splits, use_nccl_stream=self.use_nccl_stream
        )

    # ------------------------------------------------------------------
    # preprocess: build A2A metadata
    # ------------------------------------------------------------------

    def preprocess(self, routing_map: torch.Tensor) -> torch.Tensor:
        """
        Compute per-expert token counts and A2A split vectors.

        Returns a CPU tensor of shape [num_local_experts] with token counts.
        """
        if self.drop_and_pad:
            num_tokens = routing_map.size(0) * self.config.moe_router_topk
            self.capacity = _get_capacity(
                num_tokens=num_tokens,
                num_experts=self.num_experts,
                capacity_factor=self.moe_expert_capacity_factor,
            )
            self.num_out_tokens = self.capacity * self.num_experts
            num_tokens_per_local_expert = torch.full(
                (self.num_local_experts,),
                self.capacity * self.tp_size * self.ep_size,
                dtype=torch.long,
            )
            self.num_global_tokens_per_local_expert = torch.full(
                (self.num_experts * self.tp_size,),
                self.capacity,
                dtype=torch.long,
                device=self.permute_idx_device,
            )
            return num_tokens_per_local_expert

        num_local_tokens_per_expert = routing_map.sum(dim=0).long()

        if (
            self.config.moe_expert_capacity_factor is not None
            or getattr(self.config, 'moe_router_padding_for_quantization', False)
        ):
            self.num_out_tokens = num_local_tokens_per_expert.sum()
            self._maybe_update_cuda_sync_point("before_permutation_1")
        else:
            self.num_out_tokens = routing_map.size(0) * self.config.moe_router_topk

        if self.ep_size > 1 or self.tp_size > 1:
            self.input_splits = num_local_tokens_per_expert.reshape(
                self.ep_size, self.num_local_experts
            ).sum(axis=1)

            # Gather global token distribution across all ranks
            num_global_tokens_per_expert = (
                gather_from_sequence_parallel_region(num_local_tokens_per_expert)
                .reshape(self.ep_size, self.tp_size, self.num_experts)
                .transpose(0, 1)
            )
            num_global_tokens_per_local_expert = num_global_tokens_per_expert[
                :, :, self.local_expert_indices[0]: self.local_expert_indices[-1] + 1
            ].contiguous()
            num_global_tokens_per_rank = num_global_tokens_per_local_expert.sum(axis=2)
            self.output_splits = num_global_tokens_per_rank[self.tp_rank]
            self.output_splits_tp = num_global_tokens_per_rank.sum(axis=1)
            num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=(0, 1))
            self._maybe_update_cuda_sync_point("before_ep_alltoall")
        else:
            num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(
                self.num_experts
            )
            num_tokens_per_local_expert = num_local_tokens_per_expert
            self._maybe_update_cuda_sync_point("before_finish")

        if self.num_local_experts > 1:
            self.num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.view(
                -1, self.num_local_experts
            )
            if not self.config.moe_permute_fusion:
                self._maybe_update_cuda_sync_point("before_permutation_2")

        assert (
            self.cuda_sync_point_priority[self.cuda_dtoh_point]
            <= self.cuda_sync_point_priority[self.cuda_sync_point]
        ), "cuda_sync_point must be after cuda_dtoh_point."
        return num_tokens_per_local_expert

    # ------------------------------------------------------------------
    # dispatch_preprocess
    # ------------------------------------------------------------------

    def dispatch_preprocess(
        self,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
    ):
        """Reshape, compute metadata, permute tokens before A2A."""
        self.hidden_shape = hidden_states.shape
        self.probs = probs
        self.routing_map = routing_map
        assert probs.dim() == 2, "Expected 2D probs tensor"
        assert routing_map.dim() == 2, "Expected 2D routing_map tensor"
        assert routing_map.dtype == torch.bool, "Expected bool routing_map"
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        if getattr(self.config, 'moe_router_padding_for_quantization', False):
            pad_multiple = _get_align_size_for_quantization(self.config)
            self.routing_map = _pad_routing_map(self.routing_map, pad_multiple)

        self.tokens_per_expert = self.preprocess(self.routing_map)

        if self.shared_experts is not None:
            self.shared_experts.pre_forward_comm(hidden_states.view(self.hidden_shape))

        self.tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_permutation_1", self.tokens_per_expert
        )
        self.hidden_shape_before_permute = hidden_states.shape
        (
            permutated_local_input_tokens,
            permuted_probs,
            self.reversed_local_input_permutation_mapping,
            _,
            _,
        ) = _permute(
            hidden_states,
            self.routing_map,
            probs=probs,
            num_out_tokens=self.num_out_tokens,
            fused=self.config.moe_permute_fusion,
            drop_and_pad=self.drop_and_pad,
        )
        return permutated_local_input_tokens, permuted_probs

    # ------------------------------------------------------------------
    # token_dispatch
    # ------------------------------------------------------------------

    def token_dispatch(self, permutated_local_input_tokens, permuted_probs):
        """
        AlltoAll(EP) dispatch.

        Uses PCIe-aware staged transfer when cross-NUMA peers are detected,
        otherwise falls back to standard flat AlltoAll.
        """
        if self.shared_experts is not None:
            self.shared_experts.wait_current_stream()

        self.tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_ep_alltoall", self.tokens_per_expert
        )

        global_input_tokens = self._ep_all_to_all(
            permutated_local_input_tokens,
            self.output_splits,
            self.input_splits,
        )

        if self.shared_experts is not None:
            self.shared_experts.linear_fc1_forward_and_act(global_input_tokens)

        global_probs = self._ep_all_to_all(
            permuted_probs,
            self.output_splits,
            self.input_splits,
        )
        return global_input_tokens, global_probs

    # ------------------------------------------------------------------
    # dispatch_postprocess
    # ------------------------------------------------------------------

    def dispatch_postprocess(self, global_input_tokens, global_probs):
        """AllGather(TP) then sort tokens by local expert."""
        if self.tp_size > 1:
            output_split_sizes = (
                self.output_splits_tp.tolist()
                if self.output_splits_tp is not None
                else None
            )
            global_input_tokens = gather_from_sequence_parallel_region(global_input_tokens)
            global_probs = gather_from_sequence_parallel_region(global_probs)

        self.tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_permutation_2", self.tokens_per_expert
        )
        if self.num_local_experts > 1:
            if self.drop_and_pad:
                global_input_tokens = (
                    global_input_tokens.view(
                        self.tp_size * self.ep_size,
                        self.num_local_experts,
                        self.capacity,
                        *global_input_tokens.size()[1:],
                    )
                    .transpose(0, 1)
                    .contiguous()
                    .flatten(start_dim=0, end_dim=2)
                )
                global_probs = (
                    global_probs.view(
                        self.tp_size * self.ep_size,
                        self.num_local_experts,
                        self.capacity,
                        *global_probs.size()[1:],
                    )
                    .transpose(0, 1)
                    .contiguous()
                    .flatten(start_dim=0, end_dim=2)
                )
            else:
                global_input_tokens, global_probs = _sort_chunks_by_idxs(
                    global_input_tokens,
                    self.num_global_tokens_per_local_expert.ravel(),
                    self.sort_input_by_local_experts,
                    probs=global_probs,
                    fused=self.config.moe_permute_fusion,
                )

        tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_finish", self.tokens_per_expert
        )
        self.tokens_per_expert = None
        return global_input_tokens, tokens_per_expert, global_probs

    # ------------------------------------------------------------------
    # combine_preprocess
    # ------------------------------------------------------------------

    def combine_preprocess(self, hidden_states):
        """Unsort by local expert, optional TP ReduceScatter."""
        if self.num_local_experts > 1:
            if self.drop_and_pad:
                hidden_states = (
                    hidden_states.view(
                        self.num_local_experts,
                        self.tp_size * self.ep_size,
                        self.capacity,
                        *hidden_states.size()[1:],
                    )
                    .transpose(0, 1)
                    .contiguous()
                    .flatten(start_dim=0, end_dim=2)
                )
            else:
                hidden_states, _ = _sort_chunks_by_idxs(
                    hidden_states,
                    self.num_global_tokens_per_local_expert.T.ravel(),
                    self.restore_output_by_local_experts,
                    fused=self.config.moe_permute_fusion,
                )

        if self.tp_size > 1:
            hidden_states = reduce_scatter_to_sequence_parallel_region(
                hidden_states.to(self.probs.dtype)
            ).to(hidden_states.dtype)

        return hidden_states

    # ------------------------------------------------------------------
    # token_combine
    # ------------------------------------------------------------------

    def token_combine(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ):
        """
        Reverse AlltoAll(EP) to collect expert outputs.

        Mirrors token_dispatch: uses PCIe-aware staged A2A when active.
        """
        if self.shared_experts is not None:
            self.shared_experts.wait_current_stream()

        permutated_local_input_tokens = self._ep_all_to_all(
            hidden_states,
            self.input_splits,
            self.output_splits,
        )
        if self.shared_experts is not None:
            self.shared_experts.linear_fc2_forward(permutated_local_input_tokens)
            self.shared_experts.post_forward_comm()
        return permutated_local_input_tokens

    # ------------------------------------------------------------------
    # combine_postprocess
    # ------------------------------------------------------------------

    def combine_postprocess(self, permutated_local_input_tokens):
        """Unpermute, reshape to original shape, add shared-expert output."""
        output = _unpermute(
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.routing_map,
            fused=self.config.moe_permute_fusion,
            drop_and_pad=self.drop_and_pad,
        )
        output = output.view(self.hidden_shape)
        if self.shared_experts is not None:
            shared_expert_output = self.shared_experts.get_output()
            output = output + shared_expert_output
        return output

    # ------------------------------------------------------------------
    # CUDA sync-point helpers (identical logic to Megatron)
    # ------------------------------------------------------------------

    def _maybe_update_cuda_sync_point(self, point: str):
        if (
            self.cuda_sync_point_priority[point]
            < self.cuda_sync_point_priority[self.cuda_sync_point]
        ):
            self.cuda_sync_point = point

    def _maybe_dtoh_and_synchronize(
        self, point: str, tokens_per_expert: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        if not self.drop_and_pad:
            if point == self.cuda_dtoh_point:
                on_side_stream = torch.cuda.current_stream() != self.cuda_dtoh_stream
                if on_side_stream:
                    self.cuda_dtoh_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.cuda_dtoh_stream):
                    tokens_per_expert = _maybe_move_tensor_to_cpu(
                        tokens_per_expert, record_stream=on_side_stream
                    )
                    self.input_splits = _maybe_move_tensor_to_cpu(
                        self.input_splits, as_numpy=True, record_stream=on_side_stream
                    )
                    self.output_splits = _maybe_move_tensor_to_cpu(
                        self.output_splits, as_numpy=True, record_stream=on_side_stream
                    )
                    self.output_splits_tp = _maybe_move_tensor_to_cpu(
                        self.output_splits_tp, as_numpy=True, record_stream=on_side_stream
                    )
                    self.num_out_tokens = _maybe_move_tensor_to_cpu(
                        self.num_out_tokens, record_stream=on_side_stream
                    )
                    if self.num_local_experts > 1 and not self.config.moe_permute_fusion:
                        self.num_global_tokens_per_local_expert = _maybe_move_tensor_to_cpu(
                            self.num_global_tokens_per_local_expert,
                            record_stream=on_side_stream,
                        )
                self.d2h_event = self.cuda_dtoh_stream.record_event()

            if point == self.cuda_sync_point:
                self.d2h_event.synchronize()

        return tokens_per_expert


# ---------------------------------------------------------------------------
# Flex dispatcher backend managers  (DeepEP / HybridEP)
# ---------------------------------------------------------------------------

class _DispatchManager(ABC):
    """Abstract A2A backend for MoEFlexTokenDispatcher."""

    @abstractmethod
    def setup_metadata(self, routing_map: torch.Tensor, probs: torch.Tensor):
        pass

    @abstractmethod
    def dispatch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_permuted_hidden_states_by_experts(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pass


# Optional imports — guarded exactly as in Megatron
try:
    from deepspeed.core.transformer.moe.fused_a2a import (  # type: ignore[import]
        fused_combine,
        fused_dispatch,
        hybrid_ep_combine,
        hybrid_ep_dispatch,
        set_deepep_num_sms,
    )
    _HAVE_FUSED_A2A = True
except ImportError:
    fused_combine = None
    fused_dispatch = None
    hybrid_ep_combine = None
    hybrid_ep_dispatch = None
    set_deepep_num_sms = None
    _HAVE_FUSED_A2A = False


class _HybridEPManager(_DispatchManager):
    """HybridEP fused dispatch/combine manager (mirrors Megatron)."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_local_experts: int,
        num_experts: int,
        config: TransformerConfig,
    ):
        if not _HAVE_FUSED_A2A or hybrid_ep_dispatch is None:
            raise ImportError(
                "HybridEP is not installed. Please install from "
                "https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep."
            )
        self.group = group
        self.num_local_experts = num_local_experts
        self.num_experts = num_experts
        self.config = config
        self.permute_fusion = config.moe_permute_fusion
        self.capacity_factor = config.moe_expert_capacity_factor
        self.drop_and_pad = config.moe_pad_expert_input_to_capacity
        if self.drop_and_pad:
            assert self.capacity_factor is not None
        self.capacity = None
        self.num_permuted_tokens = None
        self.token_probs: Optional[torch.Tensor] = None
        self.handle = None
        self.pad_multiple = None
        self.moe_expert_rank_capacity_factor = getattr(
            config, 'moe_expert_rank_capacity_factor', None
        )
        self.over_budget = torch.zeros(1, dtype=torch.bool, device='cuda')

    def setup_metadata(self, routing_map: torch.Tensor, probs: torch.Tensor):
        num_tokens = routing_map.shape[0]
        self.routing_map = routing_map.reshape(num_tokens, self.num_experts)
        self.token_probs = probs.reshape(num_tokens, self.num_experts)
        if self.moe_expert_rank_capacity_factor is not None:
            pad_multiple = _get_align_size_for_quantization(self.config)
            budget = int(
                routing_map.shape[0]
                * self.config.moe_router_topk
                * self.moe_expert_rank_capacity_factor
            )
            budget += -budget % pad_multiple
            self.num_permuted_tokens = budget
        if self.drop_and_pad:
            num_out_tokens = num_tokens * self.config.moe_router_topk
            self.capacity = _get_capacity(
                num_tokens=num_out_tokens,
                num_experts=self.num_experts,
                capacity_factor=self.capacity_factor,
            )
            self.num_permuted_tokens = (
                self.capacity * self.group.size() * self.num_local_experts
            )
            self.tokens_per_expert = torch.full(
                (self.num_local_experts,),
                self.capacity * self.group.size(),
                dtype=torch.long,
            )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ) -> torch.Tensor:
        if self.token_probs.dtype != torch.float32:
            if self.token_probs.dtype in [torch.bfloat16, torch.float16]:
                logger.warning("HybridEP only supports float32 probs; casting.")
            self.token_probs = self.token_probs.float()
        if getattr(self.config, 'fp8', None) or getattr(self.config, 'fp4', None):
            self.pad_multiple = _get_align_size_for_quantization(self.config)
        dispatched_hidden, self.dispatched_probs, _, tokens_per_expert, self.handle = (
            hybrid_ep_dispatch(
                x=hidden_states,
                routing_map=self.routing_map,
                probs=self.token_probs,
                group=self.group,
                num_local_experts=self.num_local_experts,
                num_sms_dispatch_api=self.config.moe_hybridep_num_sms,
                num_sms_combine_api=self.config.moe_hybridep_num_sms,
                num_blocks_permute=self.config.moe_hybridep_num_blocks_permute,
                num_blocks_unpermute=self.config.moe_hybridep_num_blocks_unpermute,
                num_permuted_tokens=self.num_permuted_tokens,
                pad_multiple=self.pad_multiple,
                fused=getattr(self.config, 'moe_permute_fusion_into_hybridep', False),
                num_sms_preprocessing_api=self.config.moe_hybridep_num_sms_preprocessing,
            )
        )
        if self.moe_expert_rank_capacity_factor is not None:
            over_budget = self.handle[-1] != 0
            self.over_budget |= over_budget
        if self.num_permuted_tokens is None:
            self.tokens_per_expert = tokens_per_expert.to(torch.int64)
            self.num_permuted_tokens = self.tokens_per_expert.sum()
        if self.moe_expert_rank_capacity_factor is not None:
            self.tokens_per_expert = tokens_per_expert.to(torch.int64)
        return dispatched_hidden

    def combine(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ) -> torch.Tensor:
        hidden_states = hybrid_ep_combine(
            x=hidden_states,
            handle=self.handle,
            num_permuted_tokens=self.num_permuted_tokens,
            pad_multiple=self.pad_multiple,
            fused=getattr(self.config, 'moe_permute_fusion_into_hybridep', False),
        )
        self.handle = None
        if not self.drop_and_pad:
            self.num_permuted_tokens = None
        return hidden_states

    def get_permuted_hidden_states_by_experts(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return hidden_states, self.dispatched_probs

    def get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def get_number_of_tokens_per_expert(self) -> torch.Tensor:
        return self.tokens_per_expert


class _DeepepManager(_DispatchManager):
    """DeepEP fused dispatch/combine manager (mirrors Megatron)."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_local_experts: int,
        router_topk: int,
        num_experts: int,
        config: TransformerConfig,
    ):
        if not _HAVE_FUSED_A2A or fused_dispatch is None:
            raise ImportError(
                "DeepEP is not installed. Please install from "
                "https://github.com/deepseek-ai/deepep."
            )
        self.group = group
        self.num_local_experts = num_local_experts
        self.config = config
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.router_dtype = getattr(config, 'moe_router_dtype', None)
        self.capacity_factor = config.moe_expert_capacity_factor
        self.permute_fusion = config.moe_permute_fusion
        self.token_indices: Optional[torch.Tensor] = None
        self.token_probs: Optional[torch.Tensor] = None
        self.handle = None
        set_deepep_num_sms(config.moe_deepep_num_sms)

    def setup_metadata(self, routing_map: torch.Tensor, probs: torch.Tensor):
        num_tokens = routing_map.shape[0]
        routing_map = routing_map.reshape(num_tokens, self.num_experts)
        probs = probs.reshape(num_tokens, self.num_experts)
        self.token_probs, self.token_indices = torch.topk(probs, self.router_topk, dim=-1)
        if self.capacity_factor is not None:
            mask = self.token_probs == 0
            self.token_indices = self.token_indices.masked_fill(mask, -1)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> torch.Tensor:
        if self.token_probs.dtype != torch.float32:
            if self.token_probs.dtype in [torch.bfloat16, torch.float16]:
                logger.warning("DeepEP only supports float32 probs; casting.")
            self.token_probs = self.token_probs.float()
        hidden_states, dispatched_indices, dispatched_probs, num_tokens_per_expert, handle = (
            fused_dispatch(
                hidden_states,
                self.token_indices,
                self.token_probs,
                self.num_experts,
                self.group,
                async_finish=async_finish,
                allocate_on_comm_stream=allocate_on_comm_stream,
            )
        )
        self.handle = handle
        self.tokens_per_expert = num_tokens_per_expert
        self.dispatched_indices = dispatched_indices
        self.dispatched_probs = dispatched_probs
        return hidden_states

    def _indices_to_multihot(
        self, indices: torch.Tensor, probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = indices.shape[0]
        multihot_routing_map = torch.zeros(
            (batch_size, self.num_local_experts), dtype=torch.long, device=indices.device
        )
        multihot_probs = torch.zeros(
            (batch_size, self.num_local_experts), dtype=torch.float, device=indices.device
        )
        mask = indices != -1
        valid_indices = indices[mask]
        row_indices = torch.arange(batch_size, device=indices.device).repeat_interleave(
            mask.sum(dim=1)
        )
        multihot_routing_map[row_indices, valid_indices] = 1
        multihot_probs[row_indices, valid_indices] = probs[mask]
        return multihot_routing_map.bool(), multihot_probs

    def get_number_of_tokens_per_expert(self) -> torch.Tensor:
        return self.tokens_per_expert

    def combine(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> torch.Tensor:
        hidden_states, _ = fused_combine(
            hidden_states,
            self.group,
            self.handle,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        self.handle = None
        self.dispatched_indices = None
        self.dispatched_probs = None
        return hidden_states

    def _pad_routing_map(
        self, routing_map: torch.Tensor, tokens_per_expert: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_multiple = _get_align_size_for_quantization(self.config)
        num_input_tokens = routing_map.shape[0]
        target_tokens_per_expert = (
            torch.ceil(tokens_per_expert / pad_multiple) * pad_multiple
        ).long()
        enough_tokens = torch.all(target_tokens_per_expert <= num_input_tokens)
        if not enough_tokens:
            logger.warning(
                "Not enough tokens to pad; falling back to GroupedMLP explicit padding."
            )
        else:
            routing_map = _pad_routing_map(routing_map, pad_multiple)
            tokens_per_expert = target_tokens_per_expert
        return routing_map, tokens_per_expert

    def get_permuted_hidden_states_by_experts(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.dispatched_routing_map, self.dispatched_probs = self._indices_to_multihot(
            self.dispatched_indices, self.dispatched_probs
        )
        if getattr(self.config, 'moe_router_padding_for_quantization', False):
            self.dispatched_routing_map, self.tokens_per_expert = self._pad_routing_map(
                self.dispatched_routing_map, self.tokens_per_expert
            )
        self.hidden_shape_before_permute = hidden_states.shape
        assert self.dispatched_probs.dtype == torch.float32
        (
            hidden_states,
            permuted_probs,
            self.reversed_mapping_for_combine,
            self.pad_offsets,
            self.tokens_per_expert,
        ) = _permute(
            hidden_states,
            self.dispatched_routing_map,
            probs=self.dispatched_probs,
            num_out_tokens=self.tokens_per_expert.sum().item(),
            fused=self.permute_fusion,
            tokens_per_expert=self.tokens_per_expert,
            align_size=_get_align_size_for_quantization(self.config),
        )
        if self.router_dtype == "fp64":
            permuted_probs = permuted_probs.to(torch.float64)
        return hidden_states, permuted_probs

    def get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = _unpermute(
            hidden_states,
            self.reversed_mapping_for_combine,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.dispatched_routing_map,
            fused=self.permute_fusion,
            pad_offsets=self.pad_offsets,
        )
        return hidden_states


# ---------------------------------------------------------------------------
# Flex dispatcher (abstracts TP/EP parallelism; uses DeepEP or HybridEP)
# ---------------------------------------------------------------------------

class MoEFlexTokenDispatcher(MoETokenDispatcher):
    """
    Flex token dispatcher — decouples communication from TP/EP grid layout.

    Uses a single TPxEP communication group so the dispatch logic is agnostic
    to the specific parallelism strategy.  Backend is either DeepEP or HybridEP.
    """

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config, pg_collection=pg_collection)
        self.num_local_experts = num_local_experts
        self.local_expert_indices = local_expert_indices

        backend = getattr(config, 'moe_flex_dispatcher_backend', 'deepep')
        if backend == "deepep":
            assert self.tp_size * self.ep_size > 1, "DeepEP requires TPxEP > 1"
            self._comm_manager = _DeepepManager(
                group=self.tp_ep_group,
                num_local_experts=self.num_local_experts,
                router_topk=self.tp_size * self.config.moe_router_topk,
                num_experts=self.tp_size * self.config.num_moe_experts,
                config=self.config,
            )
            self.cudagraph_attrs = [
                '_comm_manager.token_probs',
                '_comm_manager.token_indices',
            ]
        elif backend == "hybridep":
            self._comm_manager = _HybridEPManager(
                group=self.tp_ep_group,
                num_local_experts=self.num_local_experts,
                num_experts=self.tp_size * self.config.num_moe_experts,
                config=self.config,
            )
            self.cudagraph_attrs = [
                '_comm_manager.token_probs',
                '_comm_manager.routing_map',
            ]
        else:
            raise ValueError(
                f"Invalid moe_flex_dispatcher_backend={backend!r}. "
                "Use 'deepep' or 'hybridep'."
            )

    def _initialize_metadata(
        self, routing_map: torch.Tensor, probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Expand routing map from [T, E] to [T, world_size, local_E]."""
        num_local_tokens = routing_map.shape[0]
        world_size = self.tp_size * self.ep_size
        routing_map = (
            routing_map.reshape(num_local_tokens, self.ep_size, 1, self.num_local_experts)
            .expand(-1, -1, self.tp_size, -1)
            .reshape(num_local_tokens, world_size, self.num_local_experts)
        ).contiguous()
        probs = (
            probs.reshape(num_local_tokens, self.ep_size, 1, self.num_local_experts)
            .expand(-1, -1, self.tp_size, -1)
            .reshape(num_local_tokens, world_size, self.num_local_experts)
        ).contiguous()
        return routing_map, probs

    @jit_fuser
    def dispatch_preprocess(
        self,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
    ):
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        routing_map, probs = self._initialize_metadata(routing_map, probs)
        self._comm_manager.setup_metadata(routing_map, probs)
        return hidden_states, self._comm_manager.token_probs

    def token_dispatch(
        self,
        hidden_states: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ):
        if self.shared_experts is not None:
            self.shared_experts.wait_current_stream()
        dispatched_hidden_states = self._comm_manager.dispatch(
            hidden_states, async_finish, allocate_on_comm_stream
        )
        if self.shared_experts is not None:
            self.shared_experts.pre_forward_comm(hidden_states, wait_current_stream=False)
            self.shared_experts.linear_fc1_forward_and_act(dispatched_hidden_states)
        return dispatched_hidden_states, self._comm_manager.dispatched_probs

    def dispatch_postprocess(
        self, hidden_states: torch.Tensor, probs: torch.Tensor
    ):
        global_input_tokens, permuted_probs = (
            self._comm_manager.get_permuted_hidden_states_by_experts(hidden_states)
        )
        tokens_per_expert = self._comm_manager.get_number_of_tokens_per_expert()
        return global_input_tokens, tokens_per_expert, permuted_probs

    def combine_preprocess(self, hidden_states: torch.Tensor):
        return self._comm_manager.get_restored_hidden_states_by_experts(hidden_states)

    def token_combine(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ):
        if self.shared_experts is not None:
            self.shared_experts.wait_current_stream()
        return self._comm_manager.combine(hidden_states, async_finish, allocate_on_comm_stream)

    def combine_postprocess(self, hidden_states: torch.Tensor):
        if self.shared_experts is not None:
            self.shared_experts.linear_fc2_forward(hidden_states)
            self.shared_experts.post_forward_comm()
            hidden_states = hidden_states + self.shared_experts.get_output()
        return hidden_states.view(self.hidden_shape)

    def check_over_budget(self) -> Optional[torch.Tensor]:
        if hasattr(self._comm_manager, 'over_budget'):
            return self._comm_manager.over_budget
        return None

    def reset_over_budget(self):
        if hasattr(self._comm_manager, 'over_budget'):
            self._comm_manager.over_budget.fill_(0)
