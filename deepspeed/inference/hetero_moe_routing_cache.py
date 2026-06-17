"""
HeteroMoERoutingCache: DES-LOC-aware Per-Block MoE Routing Storage
====================================================================

Upstream Design Intent (Megatron 64870c14577a84267b9c89dc130124ddf8a72d42)
---------------------------------------------------------------------------
Megatron-LM introduced per-block MoE routing storage to support prefix caching
with routing replay. The core insight: when KV cache blocks are reused across
requests (prefix caching), the MoE routing decisions for those tokens must also
be replayable. Instead of accumulating routing tensors per-request across steps
(which wastes memory and breaks with prefix cache hits), routing indices are
scattered into per-block storage immediately after each forward pass, aligned
with the token-to-block mapping. At request completion, routing is reconstructed
by reading back the blocks in order.

Key upstream changes:
  1. KVBlockAllocator gains `block_routing: Dict[int, np.ndarray]` keyed by
     block ID, with store/get/scatter/reconstruct primitives.
  2. Per-step routing accumulation (concatenating tensors per request) is
     replaced by a single scatter into block storage before request transitions.
  3. `routing_indices` on DynamicInferenceRequest changes from torch.Tensor
     (accumulated live) to np.ndarray (reconstructed at completion).
  4. TextGenerationController._router_record_bookkeeping() returns a flat numpy
     array [active_token_count, num_layers, topk] instead of a per-request dict.
  5. Serialization extended to handle np.ndarray alongside torch.Tensor.

DES-LOC Adaptation Points
--------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) adds three dimensions
of heterogeneity absent in Megatron's homogeneous GPU cluster:

1. **Device-Tiered Routing Storage**
   The cluster has 2x A6000 (48 GB, SM86) and 1x H100 NVL (96 GB, SM90).
   Routing data is tiny relative to KV cache (int16, ~KB per block), but
   accumulates across thousands of blocks. We assign routing storage to the
   device tier that owns the corresponding KV blocks:
     - H100 NVL: prefill-side blocks (large batches, complex MoE topologies)
     - A6000: decode-side blocks (long tails, smaller routing footprints)
   This prevents routing metadata from competing with KV activations on the
   memory-constrained A6000s.

2. **Shared LOcality Cache (LOC) Integration**
   DES-LOC's LOC layer is a 1.5 TB CPU DRAM slab shared between the prefill
   and decode execution streams. Prefix-cached blocks that have been evicted
   from GPU but retained in LOC keep their routing data alive in a CPU-resident
   `loc_routing` dict. On GPU re-promotion, routing is moved back with the KV
   data. This is the primary divergence from Megatron: routing is a first-class
   citizen of the LOC eviction/promotion pipeline.

3. **PCIe-Topology-Aware Scatter**
   Without NVLink, inter-GPU routing transfers are expensive. The scatter logic
   (store_routing_per_block) is annotated with device-origin information so the
   caller can batch cross-device writes via PCIe DMA rather than issuing
   per-token transfers. Routing data is always converted to CPU numpy before
   storage (matching Megatron's final design) which naturally avoids GPU→GPU
   PCIe round-trips.

4. **SM86/SM90 dtype selection**
   SM90 (H100) supports efficient int16 arithmetic and BF16 accumulation for
   MoE gates. SM86 (A6000) benefits from int16 for routing indices but may
   fall back to int32 for large expert counts. The dtype is selected per-device
   tier at construction time.

5. **Reconstruction with LOC fallback**
   reconstruct_routing_from_blocks() first checks GPU-resident block_routing,
   then falls back to loc_routing (CPU DRAM). This supports the common DES-LOC
   pattern where prefill completes on H100, evicts some early blocks to LOC,
   and decode on A6000 needs to reconstruct routing for the full sequence.

Module Structure
----------------
  HeteroDeviceTier          - enum for A6000 / H100 / CPU-LOC tiers
  DeviceRoutingConfig       - per-tier dtype and memory budget
  HeteroMoERoutingCache     - main class (drop-in for KVBlockAllocator.block_routing
                              + methods, extended for DES-LOC)
  RoutingScatterPlan        - batched PCIe-aware scatter descriptor
  LOCRoutingEvictionPolicy  - LRU eviction for CPU LOC routing slab
  DESLOCRoutingSerializer   - serialize/deserialize routing indices (extends
                              Megatron's ndarray serialization for LOC metadata)
"""

from __future__ import annotations

import logging
import time
import unittest
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device tier definitions
# ---------------------------------------------------------------------------

class HeteroDeviceTier(Enum):
    """Physical device tiers in the DES-LOC 2xA6000+H100 cluster.

    Tier assignment determines:
      - Routing dtype (int16 vs int32)
      - Whether routing lives on GPU or CPU LOC
      - PCIe transfer priority for scatter operations
    """
    H100_NVL = auto()   # SM90, 96 GB, prefill-side
    A6000 = auto()       # SM86, 48 GB, decode-side (x2)
    CPU_LOC = auto()     # 1.5 TB DRAM, shared LOcality Cache


@dataclass(frozen=True)
class DeviceRoutingConfig:
    """Per-tier configuration for routing index storage.

    Attributes:
        tier: The device tier this config applies to.
        routing_dtype: numpy dtype for routing indices. int16 suffices for
            up to 32768 experts (Megatron's threshold); H100 can handle
            larger models so we allow int32 there.
        max_blocks: Maximum number of blocks whose routing we keep in the
            hot dict before spilling to LOC or evicting.
        device_index: CUDA device index (-1 for CPU).
    """
    tier: HeteroDeviceTier
    routing_dtype: np.dtype
    max_blocks: int
    device_index: int

    @classmethod
    def for_h100(cls, num_experts: int = 8, device_index: int = 0) -> DeviceRoutingConfig:
        """H100 NVL config: large expert counts tolerated, int32 available."""
        dtype = np.dtype(np.int16) if num_experts <= 32768 else np.dtype(np.int32)
        return cls(
            tier=HeteroDeviceTier.H100_NVL,
            routing_dtype=dtype,
            max_blocks=65536,  # 96 GB headroom allows large routing hot set
            device_index=device_index,
        )

    @classmethod
    def for_a6000(cls, num_experts: int = 8, device_index: int = 1) -> DeviceRoutingConfig:
        """A6000 config: conservative block budget to avoid KV pressure."""
        dtype = np.dtype(np.int16) if num_experts <= 32768 else np.dtype(np.int32)
        return cls(
            tier=HeteroDeviceTier.A6000,
            routing_dtype=dtype,
            max_blocks=16384,  # tighter: 48 GB shared with KV cache
            device_index=device_index,
        )

    @classmethod
    def for_loc(cls) -> DeviceRoutingConfig:
        """CPU LOC config: essentially unlimited, backed by 1.5 TB DRAM."""
        return cls(
            tier=HeteroDeviceTier.CPU_LOC,
            routing_dtype=np.dtype(np.int16),
            max_blocks=1 << 20,  # ~1M blocks; LOC is DRAM-bounded not count-bounded
            device_index=-1,
        )


# ---------------------------------------------------------------------------
# LOC eviction policy
# ---------------------------------------------------------------------------

class LOCRoutingEvictionPolicy:
    """LRU eviction for the CPU LOcality Cache routing slab.

    The LOC is a shared CPU DRAM region. When routing blocks spill from GPU
    (either A6000 or H100) to LOC, they enter this LRU. When LOC capacity
    is exceeded, least-recently-used blocks are dropped. Blocks re-promoted
    to GPU are bumped to MRU position.

    This mirrors the KV block LRU in Megatron's PrefixCachingEvictionPolicy.LRU
    but operates on routing metadata rather than KV activations.

    Attributes:
        capacity: Maximum number of block routing entries in LOC.
        _cache: OrderedDict used as LRU (oldest at front, newest at back).
    """

    def __init__(self, capacity: int = 1 << 18) -> None:
        self.capacity = capacity
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._eviction_count = 0

    def put(self, block_id: int, routing: np.ndarray) -> Optional[int]:
        """Insert or update a block's routing in LOC.

        Returns the evicted block_id if capacity was exceeded, else None.
        """
        if block_id in self._cache:
            self._cache.move_to_end(block_id)
            self._cache[block_id] = routing
            return None

        evicted_id = None
        if len(self._cache) >= self.capacity:
            evicted_id, _ = self._cache.popitem(last=False)
            self._eviction_count += 1
            logger.debug(
                "LOC routing eviction: block %d evicted (total evictions: %d)",
                evicted_id,
                self._eviction_count,
            )

        self._cache[block_id] = routing
        return evicted_id

    def get(self, block_id: int) -> Optional[np.ndarray]:
        """Retrieve routing for block_id, bumping it to MRU. Returns None if absent."""
        if block_id not in self._cache:
            return None
        self._cache.move_to_end(block_id)
        return self._cache[block_id]

    def pop(self, block_id: int) -> Optional[np.ndarray]:
        """Remove and return routing for block_id (used on GPU re-promotion)."""
        return self._cache.pop(block_id, None)

    def clear(self) -> None:
        """Evict all LOC routing data."""
        count = len(self._cache)
        self._cache.clear()
        if count > 0:
            logger.info("LOC routing cache cleared (%d blocks evicted)", count)

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, block_id: int) -> bool:
        return block_id in self._cache


# ---------------------------------------------------------------------------
# PCIe-aware scatter plan
# ---------------------------------------------------------------------------

@dataclass
class RoutingScatterPlan:
    """Descriptor for a batched PCIe-aware scatter operation.

    In the Megatron upstream, store_routing_per_block() operates token-by-token
    within a single GPU's context. In DES-LOC, prefill runs on H100 and decode
    runs on A6000; routing data produced on H100 may need to be written into
    A6000-owned blocks (for sequences whose KV tail migrates mid-generation).

    Rather than issuing individual cross-device writes, we batch them into a
    RoutingScatterPlan which the caller can execute asynchronously via
    PCIe DMA streams. This avoids stalling the prefill pipeline on PCIe latency.

    Attributes:
        source_tier: Where routing data originates (device that ran forward pass).
        block_id: Target block ID.
        target_tier: Which device tier owns this block's KV cache.
        positions: Token positions within the block.
        routing: Routing indices to write [num_positions, num_layers, topk].
        priority: Higher priority plans are executed first (prefill > decode).
    """
    source_tier: HeteroDeviceTier
    block_id: int
    target_tier: HeteroDeviceTier
    positions: np.ndarray
    routing: np.ndarray
    priority: int = 0

    @property
    def is_cross_device(self) -> bool:
        """True if this scatter crosses device tiers (requires PCIe transfer)."""
        return self.source_tier != self.target_tier and self.target_tier != HeteroDeviceTier.CPU_LOC


# ---------------------------------------------------------------------------
# Serialization helpers (extends Megatron's ndarray serialization)
# ---------------------------------------------------------------------------

class DESLOCRoutingSerializer:
    """Serialize/deserialize routing indices with DES-LOC LOC metadata.

    Upstream Megatron added serialize_ndarray / deserialize_ndarray to handle
    routing_indices as np.ndarray on InferenceRequest. DES-LOC extends this
    to carry tier provenance metadata, enabling the receiver to route the
    array to the correct storage tier on deserialization.

    The wire format is a plain dict compatible with Megatron's format when
    `loc_meta` is absent, enabling interoperability with upstream consumers.
    """

    @staticmethod
    def serialize(arr: np.ndarray, source_tier: Optional[HeteroDeviceTier] = None) -> dict:
        """Serialize routing array with optional DES-LOC tier metadata.

        Args:
            arr: Routing indices [total_tokens, num_layers, topk].
            source_tier: Originating device tier (A6000 / H100 / CPU_LOC).

        Returns:
            JSON-serializable dict with 'data', 'dtype', and optional 'loc_meta'.
        """
        result: dict = {"data": arr.tolist(), "dtype": str(arr.dtype)}
        if source_tier is not None:
            result["loc_meta"] = {"source_tier": source_tier.name}
        return result

    @staticmethod
    def deserialize(obj: dict) -> Tuple[np.ndarray, Optional[HeteroDeviceTier]]:
        """Deserialize routing array and extract tier metadata if present.

        Args:
            obj: Dict from serialize().

        Returns:
            Tuple of (ndarray, source_tier or None).
        """
        arr = np.array(obj["data"], dtype=np.dtype(obj["dtype"]))
        tier = None
        if "loc_meta" in obj:
            try:
                tier = HeteroDeviceTier[obj["loc_meta"]["source_tier"]]
            except (KeyError, ValueError):
                logger.warning(
                    "Unknown DES-LOC tier in routing serialization: %s",
                    obj["loc_meta"].get("source_tier"),
                )
        return arr, tier

    @staticmethod
    def serialize_ndarray(arr: np.ndarray) -> dict:
        """Megatron-compatible serialization (no tier metadata).

        Drop-in replacement for megatron.core.inference.inference_request.serialize_ndarray.
        """
        return {"data": arr.tolist(), "dtype": str(arr.dtype)}

    @staticmethod
    def deserialize_ndarray(obj: dict) -> np.ndarray:
        """Megatron-compatible deserialization.

        Drop-in replacement for megatron.core.inference.inference_request.deserialize_ndarray.
        """
        return np.array(obj["data"], dtype=np.dtype(obj["dtype"]))


# ---------------------------------------------------------------------------
# Main cache: HeteroMoERoutingCache
# ---------------------------------------------------------------------------

class HeteroMoERoutingCache:
    """DES-LOC-aware per-block MoE routing storage.

    Upstream Context (Megatron 64870c14)
    -------------------------------------
    Megatron added block_routing: Dict[int, np.ndarray] to KVBlockAllocator,
    with four primitive operations:
      - store_block_routing(block_id, positions, routing): write token routing
        into a block's storage array at specific positions.
      - get_block_routing(block_id): retrieve a block's routing array.
      - store_routing_per_block(flat_routing): scatter a step's flat routing
        across all active blocks using the context's token-to-block mapping.
      - reconstruct_routing_from_blocks(block_ids, total_tokens): rebuild a
        request's full routing array from block-ordered storage.

    The motivation: prefix caching requires that routing decisions for cached
    tokens be reproducible (routing replay). Accumulating per-request tensors
    across steps is incompatible with prefix cache hits (where tokens appear
    in the context without an active forward pass). Per-block storage solves
    this by co-locating routing with the KV blocks that carry the tokens.

    DES-LOC Extensions
    ------------------
    This class is the DES-LOC adaptation of block_routing + its methods.
    It is instantiated by DeepSpeed's heterogeneous KV block allocator and
    holds routing for blocks owned by any device tier.

    Key differences from Megatron:
      1. Two-level hot/cold storage: GPU-resident hot dict + CPU LOC cold dict.
         Blocks evicted from GPU KV cache spill routing to LOC; reconstruction
         checks both layers.
      2. Per-tier dtype: H100 and A6000 may use different int dtypes depending
         on num_experts (matches Megatron's _ri_dtype selection but extended
         to be per-tier).
      3. scatter_plan_buffer: Rather than immediately writing cross-device
         routing (expensive over PCIe), deferred scatter plans are accumulated
         and can be flushed in bulk by the execution engine.
      4. block_tier: Optional Dict[int, HeteroDeviceTier] tracking which device
         tier owns each block, enabling tier-aware reconstruction.
      5. LOC eviction via LOCRoutingEvictionPolicy (LRU).

    Attributes:
        block_size_tokens: Number of tokens per KV block (from block allocator config).
        num_layers: Number of MoE layers in the model.
        topk: Number of experts selected per token per layer.
        dummy_block_idx: Index of the dummy/padding block (never stored).
        device_configs: Per-tier DeviceRoutingConfig.
        block_routing: GPU-hot routing storage {block_id -> ndarray}.
        block_tier: {block_id -> HeteroDeviceTier} for reconstruction routing.
        loc_routing: CPU LOC eviction-backed cold storage.
        scatter_plan_buffer: Pending cross-device scatter plans (PCIe deferred).
        _stats: Internal counters for monitoring.
    """

    def __init__(
        self,
        block_size_tokens: int,
        num_layers: int,
        topk: int,
        dummy_block_idx: int,
        device_configs: Optional[Dict[HeteroDeviceTier, DeviceRoutingConfig]] = None,
        loc_capacity: int = 1 << 18,
    ) -> None:
        """Initialise the routing cache.

        Args:
            block_size_tokens: Tokens per KV block (must match KV allocator).
            num_layers: Number of MoE layers (routing depth).
            topk: Experts selected per token per layer.
            dummy_block_idx: Padding block index; routing is never stored here.
            device_configs: Per-tier configs. Defaults to standard 2xA6000+H100.
            loc_capacity: Max blocks in CPU LOC routing slab.
        """
        self.block_size_tokens = block_size_tokens
        self.num_layers = num_layers
        self.topk = topk
        self.dummy_block_idx = dummy_block_idx

        if device_configs is None:
            device_configs = {
                HeteroDeviceTier.H100_NVL: DeviceRoutingConfig.for_h100(),
                HeteroDeviceTier.A6000: DeviceRoutingConfig.for_a6000(device_index=1),
                HeteroDeviceTier.CPU_LOC: DeviceRoutingConfig.for_loc(),
            }
        self.device_configs = device_configs

        # GPU-hot storage: matches Megatron's block_routing dict semantics
        self.block_routing: Dict[int, np.ndarray] = {}

        # Tier assignment for each stored block
        self.block_tier: Dict[int, HeteroDeviceTier] = {}

        # CPU LOC cold storage with LRU eviction
        self.loc_routing = LOCRoutingEvictionPolicy(capacity=loc_capacity)

        # Deferred PCIe scatter plans (cross-device writes)
        self.scatter_plan_buffer: List[RoutingScatterPlan] = []

        # Monitoring counters
        self._stats = {
            "store_calls": 0,
            "get_hits_hot": 0,
            "get_hits_loc": 0,
            "get_misses": 0,
            "loc_spills": 0,
            "loc_promotions": 0,
            "reconstruct_ok": 0,
            "reconstruct_miss": 0,
            "scatter_cross_device": 0,
        }

    # -----------------------------------------------------------------------
    # Primitive block-level operations (mirrors Megatron KVBlockAllocator)
    # -----------------------------------------------------------------------

    def store_block_routing(
        self,
        block_id: int,
        positions: np.ndarray,
        routing: np.ndarray,
        tier: Optional[HeteroDeviceTier] = None,
    ) -> None:
        """Store routing indices for specific token positions in a block.

        Mirrors Megatron's KVBlockAllocator.store_block_routing(), extended to
        record tier provenance and select the appropriate dtype per tier.

        If the block already has routing in LOC (e.g., spilled from a previous
        occupant), the hot dict takes precedence — LOC entry is NOT promoted
        here, only when explicitly requested via promote_from_loc().

        Args:
            block_id: The KV block ID.
            positions: 1-D int array of token positions within the block.
            routing: Routing indices [num_positions, num_layers, topk].
            tier: Device tier that owns this block. Defaults to H100_NVL.
        """
        if tier is None:
            tier = HeteroDeviceTier.H100_NVL

        cfg = self.device_configs.get(tier, self.device_configs[HeteroDeviceTier.H100_NVL])
        target_dtype = cfg.routing_dtype

        if routing.dtype != target_dtype:
            routing = routing.astype(target_dtype)

        if block_id not in self.block_routing:
            self.block_routing[block_id] = np.zeros(
                (self.block_size_tokens, self.num_layers, self.topk),
                dtype=target_dtype,
            )

        self.block_routing[block_id][positions] = routing
        self.block_tier[block_id] = tier
        self._stats["store_calls"] += 1

    def get_block_routing(self, block_id: int) -> Optional[np.ndarray]:
        """Retrieve routing indices for a block.

        Checks GPU-hot storage first, then falls back to CPU LOC cold storage.
        A LOC hit triggers a promotion back to the hot dict (warming the block).

        Mirrors Megatron's KVBlockAllocator.get_block_routing(), extended with
        the LOC fallback layer.

        Args:
            block_id: The KV block ID.

        Returns:
            ndarray [block_size_tokens, num_layers, topk] or None if not found.
        """
        # Hot path: GPU-resident
        if block_id in self.block_routing:
            self._stats["get_hits_hot"] += 1
            return self.block_routing[block_id]

        # Cold path: CPU LOC
        loc_entry = self.loc_routing.get(block_id)
        if loc_entry is not None:
            # Promote to GPU-hot storage
            self.block_routing[block_id] = loc_entry
            self._stats["get_hits_loc"] += 1
            self._stats["loc_promotions"] += 1
            logger.debug(
                "LOC routing promotion: block %d moved to hot storage", block_id
            )
            return loc_entry

        self._stats["get_misses"] += 1
        return None

    def pop_block_routing(self, block_id: int) -> None:
        """Remove routing for block_id from hot storage.

        Called when a block is re-allocated (Megatron clears routing on
        allocate). Does NOT touch LOC: LOC data is retained until the
        block is explicitly evicted from LOC or LOC is full.

        Args:
            block_id: The KV block ID.
        """
        self.block_routing.pop(block_id, None)
        self.block_tier.pop(block_id, None)

    def spill_to_loc(self, block_id: int) -> bool:
        """Spill a block's routing from GPU-hot to CPU LOC.

        Called by the DeepSpeed KV eviction pipeline when a KV block is
        evicted from GPU to CPU LOC. Routing follows the KV data so that
        reconstruction can still succeed from LOC.

        Args:
            block_id: The KV block ID being evicted.

        Returns:
            True if the block had routing to spill, False otherwise.
        """
        routing = self.block_routing.pop(block_id, None)
        tier = self.block_tier.pop(block_id, None)
        if routing is None:
            return False

        evicted = self.loc_routing.put(block_id, routing)
        self._stats["loc_spills"] += 1

        if evicted is not None:
            logger.debug(
                "LOC routing full: block %d evicted from LOC to make room for block %d",
                evicted,
                block_id,
            )

        logger.debug(
            "Routing spilled to LOC: block %d (tier=%s, shape=%s)",
            block_id,
            tier.name if tier else "unknown",
            routing.shape,
        )
        return True

    def promote_from_loc(self, block_id: int) -> bool:
        """Promote a block's routing from CPU LOC back to GPU-hot storage.

        Called when a KV block is promoted from CPU LOC back to GPU (e.g.,
        prefix cache re-hit after eviction). Routing is restored alongside KV.

        Args:
            block_id: The KV block ID being promoted.

        Returns:
            True if LOC had routing to promote, False otherwise.
        """
        routing = self.loc_routing.pop(block_id)
        if routing is None:
            return False

        self.block_routing[block_id] = routing
        self._stats["loc_promotions"] += 1
        logger.debug("Routing promoted from LOC to hot: block %d", block_id)
        return True

    def clear(self) -> None:
        """Clear all routing storage (hot + LOC).

        Mirrors Megatron's KVBlockAllocator.reset() which calls
        block_routing.clear(). Extended to also clear LOC and pending
        scatter plans.
        """
        hot_count = len(self.block_routing)
        loc_count = len(self.loc_routing)

        self.block_routing.clear()
        self.block_tier.clear()
        self.loc_routing.clear()
        self.scatter_plan_buffer.clear()

        if hot_count > 0 or loc_count > 0:
            logger.info(
                "HeteroMoERoutingCache cleared: %d hot blocks, %d LOC blocks",
                hot_count,
                loc_count,
            )

    # -----------------------------------------------------------------------
    # Scatter: flat routing → per-block storage
    # (mirrors Megatron's store_routing_per_block)
    # -----------------------------------------------------------------------

    def store_routing_per_block(
        self,
        flat_routing: Optional[np.ndarray],
        token_to_block_idx: np.ndarray,
        token_to_local_position: np.ndarray,
        active_token_count: int,
        source_tier: HeteroDeviceTier = HeteroDeviceTier.H100_NVL,
        block_tier_map: Optional[Dict[int, HeteroDeviceTier]] = None,
        defer_cross_device: bool = True,
    ) -> List[RoutingScatterPlan]:
        """Scatter flat routing indices into per-block storage.

        This is the DES-LOC adaptation of Megatron's
        KVBlockAllocator.store_routing_per_block(). The upstream version
        uses numpy argsort-based grouping to batch token writes per block.
        We preserve that algorithm and extend it with:
          - Per-block tier lookup for dtype selection
          - Deferred PCIe scatter plans for cross-device writes
          - LOC awareness: blocks in LOC that receive new routing are
            updated in LOC directly (no GPU round-trip)

        Must be called while token-to-block mappings are still valid
        (before update_requests / request transitions in the engine step).

        Args:
            flat_routing: [active_token_count, num_layers, topk] CPU numpy array,
                or None (no-op).
            token_to_block_idx: [active_token_count] int array mapping each
                active token to its KV block ID.
            token_to_local_position: [active_token_count] int array of each
                token's position within its block.
            active_token_count: Number of active tokens this step.
            source_tier: Device tier that produced this routing (ran fwd pass).
            block_tier_map: Optional override for per-block tier assignment.
                If None, all blocks are assumed to belong to source_tier.
            defer_cross_device: If True, cross-device writes are deferred into
                scatter_plan_buffer rather than written immediately.

        Returns:
            List of RoutingScatterPlan for cross-device writes (may be empty).
        """
        if flat_routing is None:
            return []

        if active_token_count == 0:
            return []

        assert flat_routing.shape[0] == active_token_count, (
            f"flat_routing token count {flat_routing.shape[0]} "
            f"!= active_token_count {active_token_count}"
        )

        dummy = self.dummy_block_idx
        deferred: List[RoutingScatterPlan] = []

        # Megatron's sort-based grouping: O(N log N) but cache-friendly
        block_ids_np = token_to_block_idx[:active_token_count]
        positions_np = token_to_local_position[:active_token_count]

        unique_blocks, inverse, counts = np.unique(
            block_ids_np, return_inverse=True, return_counts=True
        )
        sorted_indices = np.argsort(inverse, kind="stable")
        sorted_positions = positions_np[sorted_indices]
        sorted_routing = flat_routing[sorted_indices]

        offset = 0
        for bid_raw, count in zip(unique_blocks, counts):
            bid = int(bid_raw)
            count = int(count)

            if bid == dummy:
                offset += count
                continue

            block_pos = sorted_positions[offset: offset + count]
            block_rout = sorted_routing[offset: offset + count]
            offset += count

            # Determine target tier for this block
            if block_tier_map is not None and bid in block_tier_map:
                target_tier = block_tier_map[bid]
            else:
                target_tier = source_tier

            # Cross-device write: defer if requested
            is_cross = (source_tier != target_tier and
                        target_tier != HeteroDeviceTier.CPU_LOC)
            if is_cross and defer_cross_device:
                plan = RoutingScatterPlan(
                    source_tier=source_tier,
                    block_id=bid,
                    target_tier=target_tier,
                    positions=block_pos.copy(),
                    routing=block_rout.copy(),
                    priority=1 if source_tier == HeteroDeviceTier.H100_NVL else 0,
                )
                deferred.append(plan)
                self.scatter_plan_buffer.append(plan)
                self._stats["scatter_cross_device"] += 1
                continue

            # Check if this block is currently in LOC (not hot)
            if bid not in self.block_routing and bid in self.loc_routing:
                # Update in LOC directly (avoid spurious GPU allocation)
                existing = self.loc_routing.get(bid)
                if existing is not None:
                    cfg = self.device_configs.get(
                        target_tier, self.device_configs[HeteroDeviceTier.H100_NVL]
                    )
                    rout_typed = block_rout.astype(cfg.routing_dtype)
                    existing[block_pos] = rout_typed
                    # LOC LRU already bumped by get() above
                    continue

            # Normal path: write to hot storage
            self.store_block_routing(bid, block_pos, block_rout, tier=target_tier)

        return deferred

    def flush_scatter_plans(
        self,
        execute_fn: Optional[object] = None,
    ) -> int:
        """Execute and clear all pending cross-device scatter plans.

        In production, execute_fn is a callable that performs the actual
        PCIe DMA transfer (provided by the DeepSpeed execution engine).
        In tests or when execute_fn is None, plans are applied locally
        using store_block_routing().

        Args:
            execute_fn: Optional callable(plan: RoutingScatterPlan) -> None.
                If None, plans are applied in-process (test mode).

        Returns:
            Number of plans executed.
        """
        plans = sorted(self.scatter_plan_buffer, key=lambda p: -p.priority)
        count = 0

        for plan in plans:
            if execute_fn is not None:
                execute_fn(plan)
            else:
                # Fallback: apply locally (test / single-process mode)
                self.store_block_routing(
                    plan.block_id,
                    plan.positions,
                    plan.routing,
                    tier=plan.target_tier,
                )
            count += 1

        if count > 0:
            logger.debug("Flushed %d cross-device scatter plans", count)

        self.scatter_plan_buffer.clear()
        return count

    # -----------------------------------------------------------------------
    # Reconstruction: per-block storage → flat routing
    # (mirrors Megatron's reconstruct_routing_from_blocks)
    # -----------------------------------------------------------------------

    def reconstruct_routing_from_blocks(
        self,
        block_ids: List[int],
        total_routing_tokens: int,
    ) -> Optional[np.ndarray]:
        """Reconstruct routing indices from per-block storage.

        Mirrors Megatron's KVBlockAllocator.reconstruct_routing_from_blocks().
        Extended with LOC fallback: blocks not in GPU-hot storage are checked
        in CPU LOC before returning None.

        Upstream semantics:
          - Block list is ordered (same order as token sequence).
          - Last block is trimmed to exactly total_routing_tokens entries.
          - total_routing_tokens = total_tokens - 1 (last generated token
            has no forward-pass routing).

        DES-LOC extension:
          - If a block is missing from both hot and LOC, returns None.
          - LOC hits are logged at DEBUG for monitoring.
          - Routing arrays from different tiers may have different dtypes;
            the result is cast to the first block's dtype for consistency.

        Args:
            block_ids: Ordered list of KV block IDs for the request.
            total_routing_tokens: Expected number of tokens in result.

        Returns:
            ndarray [total_routing_tokens, num_layers, topk] or None.
        """
        routing_parts: List[np.ndarray] = []
        tokens_collected = 0
        loc_hits = 0

        for bid in block_ids:
            if tokens_collected >= total_routing_tokens:
                break

            # Hot lookup first
            routing = self.block_routing.get(bid)

            if routing is None:
                # LOC fallback (critical DES-LOC path: prefill evicted to LOC)
                routing = self.loc_routing.get(bid)
                if routing is not None:
                    loc_hits += 1
                else:
                    logger.debug(
                        "reconstruct_routing_from_blocks: block %d missing from hot+LOC",
                        bid,
                    )
                    self._stats["reconstruct_miss"] += 1
                    return None

            remaining = total_routing_tokens - tokens_collected
            take = min(self.block_size_tokens, remaining)
            routing_parts.append(routing[:take])
            tokens_collected += take

        if not routing_parts or tokens_collected != total_routing_tokens:
            self._stats["reconstruct_miss"] += 1
            return None

        if loc_hits > 0:
            logger.debug(
                "Routing reconstruction used %d LOC fallback(s) for %d-block request",
                loc_hits,
                len(block_ids),
            )

        result = np.concatenate(routing_parts, axis=0)
        self._stats["reconstruct_ok"] += 1
        return result

    # -----------------------------------------------------------------------
    # Router bookkeeping: GPU tensor → CPU numpy
    # (mirrors TextGenerationController._router_record_bookkeeping())
    # -----------------------------------------------------------------------

    @staticmethod
    def collect_flat_routing(
        stacked_routing: Optional[torch.Tensor],
        active_token_count: int,
        padded_active_token_count: int,
        tp_size: int,
        use_sequence_parallel: bool,
        num_experts: int,
        source_tier: HeteroDeviceTier = HeteroDeviceTier.H100_NVL,
        tp_group: Optional[object] = None,
    ) -> Optional[np.ndarray]:
        """Collect and convert per-step routing to a flat CPU numpy array.

        This is the DES-LOC adaptation of Megatron's
        TextGenerationController._router_record_bookkeeping().

        Upstream change (64870c14): Instead of splitting routing by request
        and returning Dict[request_id, Tensor], return a flat numpy array
        [active_token_count, num_layers, topk] aligned with the context's
        active-token layout. The split-by-request happens implicitly via
        the token-to-block mapping in store_routing_per_block.

        DES-LOC extension: dtype is selected per source_tier (H100 allows
        int32 for very large expert counts; A6000 uses int16 for memory
        efficiency).

        Args:
            stacked_routing: [local_token_count_or_active, num_layers, topk]
                GPU tensor from model's routing metadata. None → returns None.
            active_token_count: Global unpadded active token count.
            padded_active_token_count: Padded count (for SP all-gather sizing).
            tp_size: Tensor parallel size.
            use_sequence_parallel: Whether SP is active (requires all-gather).
            num_experts: Total number of MoE experts (for dtype selection).
            source_tier: Device tier that ran the forward pass.
            tp_group: Process group for all-gather (None in single-GPU tests).

        Returns:
            np.ndarray [active_token_count, num_layers, topk] on CPU, or None.
        """
        if stacked_routing is None:
            return None

        # SP all-gather: each TP rank processes padded_active_token_count // tp_size tokens.
        # Megatron upstream fix (64870c14): truncate to true per-rank count before gather
        # to avoid gathering stale buffer entries from the static CUDA graph buffer.
        if tp_size > 1 and use_sequence_parallel:
            local_token_count = padded_active_token_count // tp_size
            stacked_routing = stacked_routing[:local_token_count]

            if tp_group is not None:
                # All-gather across TP ranks
                try:
                    from megatron.core.tensor_parallel.mappings import (
                        gather_from_sequence_parallel_region,
                    )
                    stacked_routing = gather_from_sequence_parallel_region(
                        stacked_routing, group=tp_group
                    )
                except ImportError:
                    logger.warning(
                        "megatron gather_from_sequence_parallel_region unavailable; "
                        "skipping TP all-gather in routing collection"
                    )

        # Remove CUDA padding, convert to CPU numpy
        stacked_routing = stacked_routing[:active_token_count]

        # Dtype selection: per-tier (DES-LOC extension of Megatron's _ri_dtype logic)
        if source_tier == HeteroDeviceTier.H100_NVL:
            ri_dtype = np.int16 if num_experts <= 32768 else np.int32
        else:
            # A6000: conservative int16 regardless of expert count to save memory
            ri_dtype = np.int16

        return stacked_routing.cpu().numpy().astype(ri_dtype)

    # -----------------------------------------------------------------------
    # Utility / introspection
    # -----------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        """Return a copy of internal statistics counters."""
        return dict(self._stats)

    def hot_block_count(self) -> int:
        """Number of blocks with routing in GPU-hot storage."""
        return len(self.block_routing)

    def loc_block_count(self) -> int:
        """Number of blocks with routing in CPU LOC storage."""
        return len(self.loc_routing)

    def __len__(self) -> int:
        """Total blocks with routing across hot + LOC (may double-count promotions)."""
        return len(self.block_routing) + len(self.loc_routing)

    def __contains__(self, block_id: int) -> bool:
        """True if block has routing in hot or LOC storage."""
        return block_id in self.block_routing or block_id in self.loc_routing

    def __repr__(self) -> str:
        return (
            f"HeteroMoERoutingCache("
            f"hot={self.hot_block_count()}, "
            f"loc={self.loc_block_count()}, "
            f"block_size={self.block_size_tokens}, "
            f"layers={self.num_layers}, "
            f"topk={self.topk})"
        )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    class TestLOCRoutingEvictionPolicy(unittest.TestCase):
        """Tests for LOCRoutingEvictionPolicy."""

        def _make_routing(self, seed: int = 0) -> np.ndarray:
            rng = np.random.default_rng(seed)
            return rng.integers(-100, 100, size=(16, 4, 2), dtype=np.int16)

        def test_put_and_get(self):
            lru = LOCRoutingEvictionPolicy(capacity=4)
            r = self._make_routing()
            lru.put(10, r)
            got = lru.get(10)
            self.assertIsNotNone(got)
            np.testing.assert_array_equal(got, r)

        def test_eviction_on_overflow(self):
            lru = LOCRoutingEvictionPolicy(capacity=2)
            lru.put(1, self._make_routing(1))
            lru.put(2, self._make_routing(2))
            evicted = lru.put(3, self._make_routing(3))
            # Block 1 is oldest, should be evicted
            self.assertEqual(evicted, 1)
            self.assertIsNone(lru.get(1))
            self.assertIsNotNone(lru.get(2))
            self.assertIsNotNone(lru.get(3))

        def test_lru_bump_on_get(self):
            lru = LOCRoutingEvictionPolicy(capacity=2)
            lru.put(1, self._make_routing(1))
            lru.put(2, self._make_routing(2))
            # Access block 1 to make it MRU
            lru.get(1)
            # Now inserting block 3 should evict block 2 (now LRU)
            evicted = lru.put(3, self._make_routing(3))
            self.assertEqual(evicted, 2)
            self.assertIsNotNone(lru.get(1))

        def test_pop_removes_entry(self):
            lru = LOCRoutingEvictionPolicy(capacity=4)
            r = self._make_routing()
            lru.put(5, r)
            popped = lru.pop(5)
            np.testing.assert_array_equal(popped, r)
            self.assertIsNone(lru.get(5))

        def test_clear(self):
            lru = LOCRoutingEvictionPolicy(capacity=4)
            for i in range(4):
                lru.put(i, self._make_routing(i))
            lru.clear()
            self.assertEqual(len(lru), 0)

        def test_update_existing_key(self):
            lru = LOCRoutingEvictionPolicy(capacity=4)
            r1 = self._make_routing(1)
            r2 = self._make_routing(2)
            lru.put(7, r1)
            lru.put(7, r2)
            got = lru.get(7)
            np.testing.assert_array_equal(got, r2)
            self.assertEqual(len(lru), 1)

    class TestHeteroMoERoutingCacheBasic(unittest.TestCase):
        """Basic store/get/reconstruct tests (mirrors Megatron TestPerBlockRouting)."""

        BLOCK_SIZE = 16
        NUM_LAYERS = 4
        TOPK = 2
        DUMMY = -1

        def _make_cache(self) -> HeteroMoERoutingCache:
            return HeteroMoERoutingCache(
                block_size_tokens=self.BLOCK_SIZE,
                num_layers=self.NUM_LAYERS,
                topk=self.TOPK,
                dummy_block_idx=self.DUMMY,
            )

        def _rand_routing(self, n: int, seed: int = 42) -> np.ndarray:
            rng = np.random.default_rng(seed)
            return rng.integers(-1000, 1000, size=(n, self.NUM_LAYERS, self.TOPK), dtype=np.int16)

        def test_store_and_get_round_trip(self):
            """store_block_routing / get_block_routing round-trip."""
            cache = self._make_cache()
            positions = np.array([0, 1, 2])
            routing = self._rand_routing(3)

            cache.store_block_routing(100, positions, routing)
            got = cache.get_block_routing(100)

            self.assertIsNotNone(got)
            self.assertIsInstance(got, np.ndarray)
            self.assertEqual(got.shape, (self.BLOCK_SIZE, self.NUM_LAYERS, self.TOPK))
            np.testing.assert_array_equal(got[:3], routing)
            np.testing.assert_array_equal(got[3:], 0)

        def test_pop_clears_hot(self):
            """pop_block_routing removes from hot storage."""
            cache = self._make_cache()
            cache.store_block_routing(42, np.array([0]), self._rand_routing(1))
            self.assertIsNotNone(cache.get_block_routing(42))
            cache.pop_block_routing(42)
            self.assertIsNone(cache.block_routing.get(42))

        def test_clear_removes_everything(self):
            """clear() wipes hot, LOC, and scatter buffer."""
            cache = self._make_cache()
            cache.store_block_routing(1, np.array([0]), self._rand_routing(1))
            cache.store_block_routing(2, np.array([0]), self._rand_routing(1))
            cache.spill_to_loc(1)
            cache.clear()
            self.assertEqual(len(cache.block_routing), 0)
            self.assertEqual(len(cache.loc_routing), 0)
            self.assertEqual(len(cache.scatter_plan_buffer), 0)

        def test_reconstruct_full(self):
            """Reconstruct across 3 blocks: 2 full + 1 partial."""
            cache = self._make_cache()
            bs = self.BLOCK_SIZE

            for bid in [10, 20]:
                rout = self._rand_routing(bs, seed=bid)
                cache.store_block_routing(bid, np.arange(bs), rout)

            partial = 5
            rout_partial = self._rand_routing(partial, seed=30)
            cache.store_block_routing(30, np.arange(partial), rout_partial)

            total = 2 * bs + partial
            result = cache.reconstruct_routing_from_blocks([10, 20, 30], total)

            self.assertIsNotNone(result)
            self.assertEqual(result.shape, (total, self.NUM_LAYERS, self.TOPK))

            rout_10 = self._rand_routing(bs, seed=10)
            np.testing.assert_array_equal(result[:bs], rout_10)
            np.testing.assert_array_equal(result[2 * bs:], rout_partial)

        def test_reconstruct_returns_none_missing_block(self):
            """Missing block → reconstruct returns None."""
            cache = self._make_cache()
            bs = self.BLOCK_SIZE

            cache.store_block_routing(10, np.arange(bs), self._rand_routing(bs, seed=10))
            # Block 20 intentionally not stored

            result = cache.reconstruct_routing_from_blocks([10, 20], 2 * bs)
            self.assertIsNone(result)
            self.assertEqual(cache.stats()["reconstruct_miss"], 1)

        def test_reconstruct_empty_block_list(self):
            """Empty block list → reconstruct returns None."""
            cache = self._make_cache()
            result = cache.reconstruct_routing_from_blocks([], 0)
            self.assertIsNone(result)

    class TestHeteroMoERoutingCacheLOC(unittest.TestCase):
        """Tests for LOC spill/promote/fallback in DES-LOC."""

        BLOCK_SIZE = 8
        NUM_LAYERS = 2
        TOPK = 2
        DUMMY = -1

        def _make_cache(self, loc_capacity: int = 16) -> HeteroMoERoutingCache:
            return HeteroMoERoutingCache(
                block_size_tokens=self.BLOCK_SIZE,
                num_layers=self.NUM_LAYERS,
                topk=self.TOPK,
                dummy_block_idx=self.DUMMY,
                loc_capacity=loc_capacity,
            )

        def _rand_routing(self, n: int, seed: int = 0) -> np.ndarray:
            rng = np.random.default_rng(seed)
            return rng.integers(-100, 100, size=(n, self.NUM_LAYERS, self.TOPK), dtype=np.int16)

        def test_spill_to_loc(self):
            """Spill moves routing from hot to LOC."""
            cache = self._make_cache()
            rout = self._rand_routing(self.BLOCK_SIZE)
            cache.store_block_routing(1, np.arange(self.BLOCK_SIZE), rout)
            self.assertIn(1, cache.block_routing)

            spilled = cache.spill_to_loc(1)
            self.assertTrue(spilled)
            self.assertNotIn(1, cache.block_routing)
            self.assertIn(1, cache.loc_routing)

        def test_get_block_routing_loc_fallback(self):
            """get_block_routing falls back to LOC and promotes."""
            cache = self._make_cache()
            rout = self._rand_routing(self.BLOCK_SIZE, seed=77)
            cache.store_block_routing(5, np.arange(self.BLOCK_SIZE), rout)
            cache.spill_to_loc(5)

            # Not in hot
            self.assertNotIn(5, cache.block_routing)

            got = cache.get_block_routing(5)
            self.assertIsNotNone(got)
            np.testing.assert_array_equal(got, rout)

            # Should be promoted back to hot
            self.assertIn(5, cache.block_routing)
            self.assertEqual(cache.stats()["get_hits_loc"], 1)
            self.assertEqual(cache.stats()["loc_promotions"], 1)

        def test_reconstruct_with_loc_fallback(self):
            """Reconstruction succeeds when some blocks are in LOC."""
            cache = self._make_cache()
            bs = self.BLOCK_SIZE

            r1 = self._rand_routing(bs, seed=1)
            r2 = self._rand_routing(bs, seed=2)

            cache.store_block_routing(10, np.arange(bs), r1)
            cache.store_block_routing(20, np.arange(bs), r2)

            # Spill block 10 to LOC (simulates H100→LOC eviction after prefill)
            cache.spill_to_loc(10)

            result = cache.reconstruct_routing_from_blocks([10, 20], 2 * bs)
            self.assertIsNotNone(result)
            self.assertEqual(result.shape, (2 * bs, self.NUM_LAYERS, self.TOPK))
            np.testing.assert_array_equal(result[:bs], r1)
            np.testing.assert_array_equal(result[bs:], r2)

        def test_promote_from_loc(self):
            """promote_from_loc restores routing to hot."""
            cache = self._make_cache()
            rout = self._rand_routing(self.BLOCK_SIZE, seed=99)
            cache.store_block_routing(7, np.arange(self.BLOCK_SIZE), rout)
            cache.spill_to_loc(7)

            promoted = cache.promote_from_loc(7)
            self.assertTrue(promoted)
            self.assertIn(7, cache.block_routing)
            self.assertNotIn(7, cache.loc_routing)

        def test_spill_nonexistent_block(self):
            """Spilling a block with no routing returns False."""
            cache = self._make_cache()
            result = cache.spill_to_loc(999)
            self.assertFalse(result)

    class TestHeteroMoERoutingCacheScatter(unittest.TestCase):
        """Tests for store_routing_per_block scatter logic."""

        BLOCK_SIZE = 4
        NUM_LAYERS = 2
        TOPK = 2
        DUMMY = -1

        def _make_cache(self) -> HeteroMoERoutingCache:
            return HeteroMoERoutingCache(
                block_size_tokens=self.BLOCK_SIZE,
                num_layers=self.NUM_LAYERS,
                topk=self.TOPK,
                dummy_block_idx=self.DUMMY,
            )

        def test_scatter_all_tokens_one_block(self):
            """All tokens in one block → one store call."""
            cache = self._make_cache()
            bs = self.BLOCK_SIZE
            rng = np.random.default_rng(0)
            flat = rng.integers(0, 8, size=(bs, self.NUM_LAYERS, self.TOPK), dtype=np.int16)
            token_to_block = np.full(bs, 42, dtype=np.int64)
            token_to_pos = np.arange(bs, dtype=np.int64)

            cache.store_routing_per_block(flat, token_to_block, token_to_pos, bs)
            stored = cache.get_block_routing(42)

            self.assertIsNotNone(stored)
            np.testing.assert_array_equal(stored[:bs], flat)

        def test_scatter_multi_block(self):
            """Tokens spread across 3 blocks."""
            cache = self._make_cache()
            bs = self.BLOCK_SIZE
            n = 3 * bs
            rng = np.random.default_rng(1)
            flat = rng.integers(0, 16, size=(n, self.NUM_LAYERS, self.TOPK), dtype=np.int16)

            token_to_block = np.repeat([10, 20, 30], bs).astype(np.int64)
            token_to_pos = np.tile(np.arange(bs), 3).astype(np.int64)

            cache.store_routing_per_block(flat, token_to_block, token_to_pos, n)

            for i, bid in enumerate([10, 20, 30]):
                stored = cache.get_block_routing(bid)
                self.assertIsNotNone(stored)
                np.testing.assert_array_equal(stored[:bs], flat[i * bs: (i + 1) * bs])

        def test_scatter_skips_dummy_block(self):
            """Tokens mapped to dummy block are not stored."""
            cache = self._make_cache()
            bs = self.BLOCK_SIZE
            rng = np.random.default_rng(2)
            flat = rng.integers(0, 8, size=(bs, self.NUM_LAYERS, self.TOPK), dtype=np.int16)

            token_to_block = np.full(bs, self.DUMMY, dtype=np.int64)
            token_to_pos = np.arange(bs, dtype=np.int64)

            cache.store_routing_per_block(flat, token_to_block, token_to_pos, bs)
            self.assertEqual(len(cache.block_routing), 0)

        def test_scatter_none_routing_is_noop(self):
            """None flat_routing → no-op, no error."""
            cache = self._make_cache()
            result = cache.store_routing_per_block(
                None, np.array([0]), np.array([0]), 1
            )
            self.assertEqual(result, [])
            self.assertEqual(len(cache.block_routing), 0)

        def test_scatter_cross_device_deferred(self):
            """Cross-device writes are deferred into scatter_plan_buffer."""
            cache = self._make_cache()
            bs = self.BLOCK_SIZE
            rng = np.random.default_rng(3)
            flat = rng.integers(0, 8, size=(bs, self.NUM_LAYERS, self.TOPK), dtype=np.int16)

            token_to_block = np.full(bs, 99, dtype=np.int64)
            token_to_pos = np.arange(bs, dtype=np.int64)

            block_tier_map = {99: HeteroDeviceTier.A6000}

            deferred = cache.store_routing_per_block(
                flat,
                token_to_block,
                token_to_pos,
                bs,
                source_tier=HeteroDeviceTier.H100_NVL,
                block_tier_map=block_tier_map,
                defer_cross_device=True,
            )

            # Should be deferred, not immediately written
            self.assertEqual(len(deferred), 1)
            self.assertEqual(len(cache.scatter_plan_buffer), 1)
            self.assertNotIn(99, cache.block_routing)  # not written yet

            # Flush executes the plan locally
            flushed = cache.flush_scatter_plans()
            self.assertEqual(flushed, 1)
            self.assertIn(99, cache.block_routing)

        def test_scatter_updates_loc_in_place(self):
            """If a block is in LOC (not hot), scatter updates LOC directly."""
            cache = self._make_cache()
            bs = self.BLOCK_SIZE
            rng = np.random.default_rng(4)
            initial = rng.integers(0, 8, size=(bs, self.NUM_LAYERS, self.TOPK), dtype=np.int16)

            # Put block 55 in LOC only
            cache.loc_routing.put(55, initial.copy())

            # New routing for positions 0..1
            new_routing = rng.integers(10, 20, size=(2, self.NUM_LAYERS, self.TOPK), dtype=np.int16)
            token_to_block = np.array([55, 55], dtype=np.int64)
            token_to_pos = np.array([0, 1], dtype=np.int64)

            cache.store_routing_per_block(new_routing, token_to_block, token_to_pos, 2)

            # Block should still NOT be in hot (updated in LOC directly)
            self.assertNotIn(55, cache.block_routing)
            # LOC should have updated values
            loc_val = cache.loc_routing.get(55)
            self.assertIsNotNone(loc_val)
            np.testing.assert_array_equal(loc_val[:2], new_routing)

    class TestDESLOCRoutingSerializer(unittest.TestCase):
        """Tests for DESLOCRoutingSerializer."""

        def _rand(self, shape: tuple, seed: int = 0) -> np.ndarray:
            rng = np.random.default_rng(seed)
            return rng.integers(-100, 100, size=shape, dtype=np.int16)

        def test_serialize_deserialize_no_tier(self):
            arr = self._rand((10, 4, 2))
            d = DESLOCRoutingSerializer.serialize(arr)
            self.assertIn("data", d)
            self.assertIn("dtype", d)
            self.assertNotIn("loc_meta", d)

            got, tier = DESLOCRoutingSerializer.deserialize(d)
            np.testing.assert_array_equal(got, arr)
            self.assertIsNone(tier)

        def test_serialize_deserialize_with_tier(self):
            arr = self._rand((5, 2, 2))
            d = DESLOCRoutingSerializer.serialize(arr, source_tier=HeteroDeviceTier.H100_NVL)
            self.assertIn("loc_meta", d)
            self.assertEqual(d["loc_meta"]["source_tier"], "H100_NVL")

            got, tier = DESLOCRoutingSerializer.deserialize(d)
            np.testing.assert_array_equal(got, arr)
            self.assertEqual(tier, HeteroDeviceTier.H100_NVL)

        def test_megatron_compat_serialize(self):
            """megatron-compatible path: no tier metadata."""
            arr = self._rand((3, 4, 2))
            d = DESLOCRoutingSerializer.serialize_ndarray(arr)
            got = DESLOCRoutingSerializer.deserialize_ndarray(d)
            np.testing.assert_array_equal(got, arr)

        def test_unknown_tier_is_handled(self):
            arr = self._rand((2, 2, 2))
            d = DESLOCRoutingSerializer.serialize(arr, source_tier=HeteroDeviceTier.A6000)
            d["loc_meta"]["source_tier"] = "NONEXISTENT_TIER"
            got, tier = DESLOCRoutingSerializer.deserialize(d)
            np.testing.assert_array_equal(got, arr)
            self.assertIsNone(tier)

    class TestCollectFlatRouting(unittest.TestCase):
        """Tests for HeteroMoERoutingCache.collect_flat_routing()."""

        def _make_tensor(self, n: int, nl: int = 4, topk: int = 2) -> torch.Tensor:
            return torch.randint(0, 8, (n, nl, topk), dtype=torch.int32)

        def test_none_returns_none(self):
            result = HeteroMoERoutingCache.collect_flat_routing(
                None, 10, 16, 1, False, 8
            )
            self.assertIsNone(result)

        def test_no_sp_single_rank(self):
            t = self._make_tensor(8)
            result = HeteroMoERoutingCache.collect_flat_routing(
                t, 8, 8, 1, False, 8,
                source_tier=HeteroDeviceTier.H100_NVL
            )
            self.assertIsNotNone(result)
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, (8, 4, 2))
            self.assertEqual(result.dtype, np.int16)

        def test_padding_stripped(self):
            """Active token count < tensor length → padded entries stripped."""
            t = self._make_tensor(16)
            result = HeteroMoERoutingCache.collect_flat_routing(
                t, active_token_count=10, padded_active_token_count=16,
                tp_size=1, use_sequence_parallel=False, num_experts=8
            )
            self.assertEqual(result.shape[0], 10)

        def test_a6000_forces_int16(self):
            """A6000 tier always uses int16 regardless of expert count."""
            t = self._make_tensor(4)
            result = HeteroMoERoutingCache.collect_flat_routing(
                t, 4, 4, 1, False, num_experts=65536,
                source_tier=HeteroDeviceTier.A6000
            )
            self.assertEqual(result.dtype, np.int16)

        def test_h100_uses_int32_for_large_expert_count(self):
            """H100 tier uses int32 when num_experts > 32768."""
            t = self._make_tensor(4)
            result = HeteroMoERoutingCache.collect_flat_routing(
                t, 4, 4, 1, False, num_experts=40000,
                source_tier=HeteroDeviceTier.H100_NVL
            )
            self.assertEqual(result.dtype, np.int32)

    class TestDeviceRoutingConfig(unittest.TestCase):
        """Tests for DeviceRoutingConfig factory methods."""

        def test_h100_small_experts(self):
            cfg = DeviceRoutingConfig.for_h100(num_experts=8)
            self.assertEqual(cfg.tier, HeteroDeviceTier.H100_NVL)
            self.assertEqual(cfg.routing_dtype, np.dtype(np.int16))

        def test_h100_large_experts(self):
            cfg = DeviceRoutingConfig.for_h100(num_experts=40000)
            self.assertEqual(cfg.routing_dtype, np.dtype(np.int32))

        def test_a6000_always_int16_small(self):
            cfg = DeviceRoutingConfig.for_a6000(num_experts=8)
            self.assertEqual(cfg.routing_dtype, np.dtype(np.int16))

        def test_loc_config(self):
            cfg = DeviceRoutingConfig.for_loc()
            self.assertEqual(cfg.tier, HeteroDeviceTier.CPU_LOC)
            self.assertEqual(cfg.device_index, -1)

    class TestRoutingScatterPlan(unittest.TestCase):
        """Tests for RoutingScatterPlan."""

        def test_cross_device_detection(self):
            plan = RoutingScatterPlan(
                source_tier=HeteroDeviceTier.H100_NVL,
                block_id=1,
                target_tier=HeteroDeviceTier.A6000,
                positions=np.array([0]),
                routing=np.zeros((1, 2, 2), dtype=np.int16),
            )
            self.assertTrue(plan.is_cross_device)

        def test_same_device_not_cross(self):
            plan = RoutingScatterPlan(
                source_tier=HeteroDeviceTier.H100_NVL,
                block_id=2,
                target_tier=HeteroDeviceTier.H100_NVL,
                positions=np.array([0]),
                routing=np.zeros((1, 2, 2), dtype=np.int16),
            )
            self.assertFalse(plan.is_cross_device)

        def test_loc_target_not_cross(self):
            plan = RoutingScatterPlan(
                source_tier=HeteroDeviceTier.H100_NVL,
                block_id=3,
                target_tier=HeteroDeviceTier.CPU_LOC,
                positions=np.array([0]),
                routing=np.zeros((1, 2, 2), dtype=np.int16),
            )
            self.assertFalse(plan.is_cross_device)

    class TestStatsTracking(unittest.TestCase):
        """Verify stats counters increment correctly."""

        def _make_cache(self) -> HeteroMoERoutingCache:
            return HeteroMoERoutingCache(4, 2, 2, -1)

        def test_store_increments_store_calls(self):
            cache = self._make_cache()
            cache.store_block_routing(1, np.array([0]), np.zeros((1, 2, 2), dtype=np.int16))
            self.assertEqual(cache.stats()["store_calls"], 1)

        def test_get_hit_hot_counter(self):
            cache = self._make_cache()
            cache.store_block_routing(1, np.array([0]), np.zeros((1, 2, 2), dtype=np.int16))
            cache.get_block_routing(1)
            self.assertEqual(cache.stats()["get_hits_hot"], 1)

        def test_get_miss_counter(self):
            cache = self._make_cache()
            cache.get_block_routing(999)
            self.assertEqual(cache.stats()["get_misses"], 1)

        def test_reconstruct_ok_counter(self):
            cache = self._make_cache()
            cache.store_block_routing(
                10, np.arange(4), np.zeros((4, 2, 2), dtype=np.int16)
            )
            cache.reconstruct_routing_from_blocks([10], 4)
            self.assertEqual(cache.stats()["reconstruct_ok"], 1)

        def test_reconstruct_miss_counter(self):
            cache = self._make_cache()
            cache.reconstruct_routing_from_blocks([99], 4)
            self.assertEqual(cache.stats()["reconstruct_miss"], 1)

        def test_loc_spill_counter(self):
            cache = self._make_cache()
            cache.store_block_routing(5, np.array([0]), np.zeros((1, 2, 2), dtype=np.int16))
            cache.spill_to_loc(5)
            self.assertEqual(cache.stats()["loc_spills"], 1)

    # Run all tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestLOCRoutingEvictionPolicy,
        TestHeteroMoERoutingCacheBasic,
        TestHeteroMoERoutingCacheLOC,
        TestHeteroMoERoutingCacheScatter,
        TestDESLOCRoutingSerializer,
        TestCollectFlatRouting,
        TestDeviceRoutingConfig,
        TestRoutingScatterPlan,
        TestStatsTracking,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
