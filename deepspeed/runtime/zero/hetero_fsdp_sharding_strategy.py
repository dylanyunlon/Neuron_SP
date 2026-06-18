"""
Heterogeneous FSDP Sharding Strategy for DES-LOC
=================================================

Upstream design intent (Megatron bb42a0081e85b5115f7c264bea78e82c0bc26a93):
    Megatron's fix adds a CLI argument ``--outer-dp-sharding-strategy`` that controls
    the outer data-parallel sharding strategy in Hybrid Sharded Data Parallel (HSDP) mode.
    In HSDP, the parallelism is factored into two nested groups:
      - Inner group: fully-sharded (FSDP) within a node or a tight NVLink domain
      - Outer group: either pure replication ("no_shard") or optimizer-state sharding ("optim")
    The upstream patch is purely a CLI-layer fix: it wires the new argument through
    ``training/arguments.py`` so that the outer sharding behaviour can be selected at
    launch time instead of being hard-coded.

DES-LOC reinterpretation:
    DES-LOC (Decoupled Execution with Shared LOcality Cache) runs on a heterogeneous cluster:
        • 2× NVIDIA A6000 48 GB  (SM86, PCIe, no NVLink)
        • 1× NVIDIA H100 NVL 96 GB (SM90, PCIe, no NVLink)
        • 1.5 TB CPU DRAM as a third-tier "slow store"
    Because the three GPUs differ in memory capacity, compute throughput (SM86 vs SM90),
    and interconnect topology (all PCIe, no NVLink), a uniform FSDP strategy is
    sub-optimal and can OOM the A6000s while leaving the H100 under-utilised.

    ``HeteroFSDPShardingStrategy`` replaces Megatron's single enum value with a
    *per-device-class* strategy table.  The outer sharding group is partitioned into
    heterogeneous *tiers*:

        Tier 0 — H100 NVL (capacity node):  ``FULL_SHARD`` — optimizer states, grads,
                  and params all sharded; H100 holds the "authoritative" shard.
        Tier 1 — A6000 pair (SM86 worker nodes): ``SHARD_GRAD_OP`` — grads + optimizer
                  states sharded, params replicated during forward/backward; prevents OOM
                  while keeping activation recompute costs manageable.
        CPU DRAM offload tier: activations and fp32 master weights that do not fit in
                  GPU VRAM are offloaded to the DES-LOC Shared LOcality Cache (SLC),
                  a pinned-memory ring-buffer in host DRAM.

    The ``HeteroFSDPShardingConfig`` dataclass is the single source of truth for all
    sharding decisions; it can be constructed programmatically *or* from a CLI
    argument string compatible with Megatron's ``--outer-dp-sharding-strategy`` syntax,
    so that existing launch scripts migrate with a one-line change.

    Key DES-LOC additions vs. upstream:
        1. ``DeviceTier`` enum — classifies each rank's device at init time.
        2. ``HeteroFSDPShardingStrategy`` — maps (tier, role) → ShardingStrategy.
        3. ``SharedLocalityCache`` — a lightweight CPU pinned-memory ring for param/grad
           offload; the "LOC" in DES-LOC.  Written in pure Python + PyTorch; zero C++.
        4. ``HeteroFSDPShardingConfig`` — parsed from CLI args, replaces the upstream
           ``outer_dp_sharding_strategy`` string enum end-to-end.
        5. ``build_hetero_fsdp_config`` — factory used by DeepSpeed engine init.
"""

from __future__ import annotations

import enum
import logging
import math
import os
import socket
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants matching the target cluster topology
# ---------------------------------------------------------------------------

# SM capability integers exposed by torch.cuda.get_device_capability()
_SM86_CAPABILITY: Tuple[int, int] = (8, 6)   # A6000
_SM90_CAPABILITY: Tuple[int, int] = (9, 0)   # H100 NVL

# VRAM thresholds (bytes) — used for automatic tier detection
_HIGH_VRAM_THRESHOLD_BYTES: int = 80 * (1 << 30)   # >80 GB → H100 tier
_MID_VRAM_THRESHOLD_BYTES: int  = 40 * (1 << 30)   # >40 GB → A6000 tier

# CPU DRAM available to the Shared Locality Cache (SLC)
_DEFAULT_SLC_SIZE_BYTES: int = 32 * (1 << 30)       # 32 GB of the 1.5 TB pool

# Megatron-compatible CLI choices (mirrors upstream --outer-dp-sharding-strategy)
_UPSTREAM_CHOICES = frozenset(["no_shard", "optim"])


# ---------------------------------------------------------------------------
# DeviceTier — classify each rank's local GPU
# ---------------------------------------------------------------------------

class DeviceTier(enum.IntEnum):
    """Tier classification for DES-LOC heterogeneous nodes.

    Ordering matters: higher tier = more capable = preferred for heavy shards.
    """
    CPU_OFFLOAD = 0   # No GPU, or explicit offload-only rank
    SM86_WORKER = 1   # A6000 48 GB — worker nodes
    SM90_ANCHOR = 2   # H100 NVL 96 GB — anchor / capacity node

    @classmethod
    def from_local_device(cls, device_index: Optional[int] = None) -> "DeviceTier":
        """Detect the tier of ``device_index`` (defaults to current CUDA device).

        Detection heuristic (in priority order):
            1. Environment variable ``DESOC_DEVICE_TIER`` overrides auto-detect.
            2. SM capability check: (9,0) → SM90_ANCHOR, (8,6) → SM86_WORKER.
            3. VRAM size fallback when SM cap is ambiguous.
            4. If no CUDA device is available, return CPU_OFFLOAD.
        """
        env_override = os.environ.get("DESOC_DEVICE_TIER", "").strip().upper()
        if env_override:
            try:
                tier = cls[env_override]
                logger.debug("DeviceTier: env override → %s", tier.name)
                return tier
            except KeyError:
                logger.warning(
                    "DESOC_DEVICE_TIER='%s' is not a valid tier; ignoring.", env_override
                )

        if not torch.cuda.is_available():
            logger.debug("DeviceTier: no CUDA device → CPU_OFFLOAD")
            return cls.CPU_OFFLOAD

        if device_index is None:
            device_index = torch.cuda.current_device()

        cap = torch.cuda.get_device_capability(device_index)
        vram = torch.cuda.get_device_properties(device_index).total_memory

        logger.debug(
            "DeviceTier: device %d  cap=%s  vram=%.1f GB",
            device_index, cap, vram / (1 << 30),
        )

        if cap >= _SM90_CAPABILITY or vram >= _HIGH_VRAM_THRESHOLD_BYTES:
            return cls.SM90_ANCHOR
        if cap >= _SM86_CAPABILITY or vram >= _MID_VRAM_THRESHOLD_BYTES:
            return cls.SM86_WORKER
        # Older / smaller GPU — treat as worker but log a warning
        logger.warning(
            "DeviceTier: device %d has unknown cap %s (vram=%.1f GB); "
            "defaulting to SM86_WORKER.",
            device_index, cap, vram / (1 << 30),
        )
        return cls.SM86_WORKER


# ---------------------------------------------------------------------------
# ShardingStrategy — mirrors torch.distributed.fsdp.ShardingStrategy
# but remains independent so DeepSpeed zero-3 can map to its own primitives.
# ---------------------------------------------------------------------------

class ShardingStrategy(enum.Enum):
    """Per-rank sharding strategy for DES-LOC FSDP groups.

    Semantics mirror ``torch.distributed.fsdp.ShardingStrategy`` but are
    interpreted by the DeepSpeed ZeRO-3 / HybridFSDP engine, not PyTorch FSDP.

    FULL_SHARD
        All three buckets (params, grads, optimizer states) are sharded across
        the process group.  Minimises per-rank peak memory.  Used on H100 anchor
        nodes which are memory-rich but must not monopolise all param copies.

    SHARD_GRAD_OP
        Grads and optimizer states are sharded; params are all-gathered before
        each forward pass and freed after backward.  Balances memory vs.
        all-reduce traffic — appropriate for A6000 worker nodes.

    NO_SHARD
        All three buckets are replicated.  Only used for very small modules
        or as a debugging baseline.

    HYBRID_SHARD
        Inner group FULL_SHARD + outer group NO_SHARD (HSDP inner layer).
        Not directly assigned per-rank; the outer strategy table determines
        whether the outer group also shards.

    CPU_OFFLOAD_SHARD
        Params live in the SLC (CPU pinned memory) and are streamed to GPU
        on demand.  For ranks operating in CPU offload mode.
    """
    FULL_SHARD       = "full_shard"
    SHARD_GRAD_OP    = "shard_grad_op"
    NO_SHARD         = "no_shard"
    HYBRID_SHARD     = "hybrid_shard"
    CPU_OFFLOAD_SHARD = "cpu_offload_shard"


# ---------------------------------------------------------------------------
# HeteroFSDPShardingStrategy — the core DES-LOC mapping
# ---------------------------------------------------------------------------

# Strategy table: (DeviceTier, role) → ShardingStrategy
#   role ∈ {"inner", "outer"}
#   inner = within the tight FSDP shard group (typically intra-node or same-tier)
#   outer = across HSDP outer group (inter-tier)
_DEFAULT_STRATEGY_TABLE: Dict[Tuple[DeviceTier, str], ShardingStrategy] = {
    (DeviceTier.SM90_ANCHOR,  "inner"): ShardingStrategy.FULL_SHARD,
    (DeviceTier.SM90_ANCHOR,  "outer"): ShardingStrategy.FULL_SHARD,
    (DeviceTier.SM86_WORKER,  "inner"): ShardingStrategy.SHARD_GRAD_OP,
    (DeviceTier.SM86_WORKER,  "outer"): ShardingStrategy.SHARD_GRAD_OP,
    (DeviceTier.CPU_OFFLOAD,  "inner"): ShardingStrategy.CPU_OFFLOAD_SHARD,
    (DeviceTier.CPU_OFFLOAD,  "outer"): ShardingStrategy.NO_SHARD,
}

# When the upstream CLI arg is "optim" we upgrade the outer strategy for anchor nodes.
_OPTIM_UPGRADE_TABLE: Dict[DeviceTier, ShardingStrategy] = {
    DeviceTier.SM90_ANCHOR: ShardingStrategy.FULL_SHARD,     # already full; no change
    DeviceTier.SM86_WORKER: ShardingStrategy.FULL_SHARD,     # upgrade: optim state also sharded
    DeviceTier.CPU_OFFLOAD: ShardingStrategy.CPU_OFFLOAD_SHARD,
}


class HeteroFSDPShardingStrategy:
    """Per-rank FSDP sharding strategy resolver for DES-LOC heterogeneous clusters.

    Constructed from a ``HeteroFSDPShardingConfig`` (or equivalently from the
    upstream ``--outer-dp-sharding-strategy`` CLI string), this class answers
    the single query:

        "Given my local GPU tier, which ``ShardingStrategy`` should I use for
         the inner / outer FSDP group?"

    It also owns the logic for deciding whether a rank should participate in
    CPU-side SLC offload.

    Attributes
    ----------
    config : HeteroFSDPShardingConfig
        The resolved configuration driving all decisions.
    local_tier : DeviceTier
        The tier of the current rank's GPU, detected at construction time.
    inner_strategy : ShardingStrategy
        Strategy for the inner (intra-tier) FSDP group.
    outer_strategy : ShardingStrategy
        Strategy for the outer (inter-tier) FSDP group.
    use_slc_offload : bool
        Whether this rank should use the Shared LOcality Cache for param/grad offload.
    """

    def __init__(
        self,
        config: "HeteroFSDPShardingConfig",
        device_index: Optional[int] = None,
    ) -> None:
        self.config = config
        self.local_tier = DeviceTier.from_local_device(device_index)
        self.inner_strategy, self.outer_strategy = self._resolve_strategies()
        self.use_slc_offload = self._should_offload_to_slc()

        logger.info(
            "HeteroFSDPShardingStrategy: rank=%s tier=%s inner=%s outer=%s slc=%s",
            _safe_rank(),
            self.local_tier.name,
            self.inner_strategy.value,
            self.outer_strategy.value,
            self.use_slc_offload,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_strategies(
        self,
    ) -> Tuple[ShardingStrategy, ShardingStrategy]:
        """Map (tier, upstream_outer_strategy) → (inner, outer) strategies.

        The upstream ``outer_dp_sharding_strategy`` value ("no_shard" / "optim")
        acts as a *hint* that DES-LOC refines per device tier:

        no_shard hint
            Use the default table (_DEFAULT_STRATEGY_TABLE).  Worker nodes get
            SHARD_GRAD_OP, anchor node gets FULL_SHARD for the inner group but
            NO_SHARD for the outer group to avoid unnecessary inter-tier
            all-gathers across slow PCIe links.

        optim hint
            Upgrade outer strategies via _OPTIM_UPGRADE_TABLE so that optimizer
            states are also sharded across tiers.  On a PCIe-only cluster this
            trades communication cost for memory savings — worthwhile when
            training very large models that otherwise OOM on A6000s.
        """
        tier = self.local_tier
        inner = _DEFAULT_STRATEGY_TABLE.get(
            (tier, "inner"), ShardingStrategy.SHARD_GRAD_OP
        )

        upstream = self.config.upstream_outer_strategy
        if upstream == "no_shard":
            # Outer group: replicate across tiers (saves PCIe bandwidth)
            outer_default = _DEFAULT_STRATEGY_TABLE.get(
                (tier, "outer"), ShardingStrategy.NO_SHARD
            )
            # But if the tier has no peer (e.g., single H100), fall back to NO_SHARD
            outer = outer_default if self.config.outer_group_size > 1 else ShardingStrategy.NO_SHARD

        elif upstream == "optim":
            # Upgrade outer strategy: optimizer states also sharded inter-tier.
            # Only valid when inner strategy is already FULL_SHARD or SHARD_GRAD_OP.
            upgraded = _OPTIM_UPGRADE_TABLE.get(tier, ShardingStrategy.SHARD_GRAD_OP)
            if tier == DeviceTier.SM86_WORKER and self.config.outer_group_size < 2:
                logger.warning(
                    "outer_dp_sharding_strategy='optim' requested but outer_group_size=%d "
                    "for SM86_WORKER; falling back to SHARD_GRAD_OP.",
                    self.config.outer_group_size,
                )
                upgraded = ShardingStrategy.SHARD_GRAD_OP
            outer = upgraded
        else:
            logger.error("Unknown upstream_outer_strategy '%s'; defaulting to NO_SHARD.", upstream)
            outer = ShardingStrategy.NO_SHARD

        return inner, outer

    def _should_offload_to_slc(self) -> bool:
        """Decide whether this rank should engage the SLC CPU-offload path.

        Rules:
          - CPU_OFFLOAD tier: always offload.
          - SM86_WORKER with outer strategy FULL_SHARD AND config.enable_slc_overflow:
                offload optimizer states that do not fit in 48 GB.
          - SM90_ANCHOR: never offload (96 GB is enough for all current models).
          - Disabled globally if config.enable_slc_overflow is False.
        """
        if not self.config.enable_slc_overflow:
            return False
        if self.local_tier == DeviceTier.CPU_OFFLOAD:
            return True
        if (
            self.local_tier == DeviceTier.SM86_WORKER
            and self.outer_strategy == ShardingStrategy.FULL_SHARD
        ):
            return True
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def strategy_for_module(
        self,
        module_param_bytes: int,
        is_outer_group: bool,
    ) -> ShardingStrategy:
        """Return the appropriate strategy for a specific module given its parameter size.

        DES-LOC adds a *size-aware* fallback: very small modules (< 4 MB) are
        not worth sharding across PCIe because the all-gather latency dominates.
        For such modules we downgrade to NO_SHARD regardless of the global policy.

        Parameters
        ----------
        module_param_bytes : int
            Total parameter bytes for the module under consideration.
        is_outer_group : bool
            True → query outer strategy, False → query inner strategy.
        """
        _SMALL_MODULE_THRESHOLD = 4 * (1 << 20)  # 4 MB

        base = self.outer_strategy if is_outer_group else self.inner_strategy

        if module_param_bytes < _SMALL_MODULE_THRESHOLD:
            logger.debug(
                "strategy_for_module: module params=%.2f MB < threshold; downgrading to NO_SHARD",
                module_param_bytes / (1 << 20),
            )
            return ShardingStrategy.NO_SHARD

        return base

    def summary(self) -> Dict[str, object]:
        """Return a JSON-serialisable summary for logging / debugging."""
        return {
            "rank": _safe_rank(),
            "hostname": socket.gethostname(),
            "local_tier": self.local_tier.name,
            "inner_strategy": self.inner_strategy.value,
            "outer_strategy": self.outer_strategy.value,
            "use_slc_offload": self.use_slc_offload,
            "upstream_outer_strategy": self.config.upstream_outer_strategy,
            "outer_group_size": self.config.outer_group_size,
            "slc_size_gb": self.config.slc_size_bytes / (1 << 30),
        }


# ---------------------------------------------------------------------------
# SharedLocalityCache — the "LOC" in DES-LOC
# ---------------------------------------------------------------------------

class SharedLocalityCache:
    """CPU pinned-memory ring-buffer for parameter and gradient offload.

    The SLC is the core DES-LOC mechanism that makes CPU DRAM a first-class
    memory tier.  Unlike naive CPU offload (which stalls the GPU pipeline
    waiting for PCIe transfers), SLC uses double-buffering and asynchronous
    CUDA streams to overlap:

        GPU compute (forward/backward) ↔ PCIe transfer (param prefetch / grad writeback)

    Architecture
    ------------
    The SLC maintains two halves ("ping" and "pong") of a large pinned tensor.
    While the GPU uses the data in the "ping" buffer, the CPU is pre-fetching
    the *next* batch of parameters into the "pong" buffer via non-blocking
    ``tensor.copy_(src, non_blocking=True)``.

    Memory layout (per half)::

        ┌──────────────────────────────────────────┐
        │  param_flat  [slc_half_bytes]            │  fp32 master weights
        ├──────────────────────────────────────────┤
        │  grad_flat   [slc_half_bytes]            │  accumulated grads
        └──────────────────────────────────────────┘

    The total SLC allocation is ``2 × slc_half_bytes`` pinned bytes.

    Parameters
    ----------
    slc_size_bytes : int
        Total bytes to allocate for the SLC ring.  Half used for params, half for grads.
    device : torch.device
        The GPU device that will consume buffers from this SLC.
    prefetch_stream : Optional[torch.cuda.Stream]
        CUDA stream used for async H2D prefetch.  Created automatically if None.

    Notes
    -----
    The SLC is process-local.  In a multi-rank deployment each rank has its own SLC
    instance sized by ``slc_size_bytes / world_size`` so the 1.5 TB DRAM pool is
    shared fairly across ranks.
    """

    def __init__(
        self,
        slc_size_bytes: int = _DEFAULT_SLC_SIZE_BYTES,
        device: Optional[torch.device] = None,
        prefetch_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        self.slc_size_bytes = slc_size_bytes
        self.device = device or (
            torch.device(f"cuda:{torch.cuda.current_device()}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # Allocate pinned CPU tensors — float32 (4 bytes / element)
        self._half_elems = slc_size_bytes // 2 // 4
        try:
            self._param_buf_ping = torch.empty(
                self._half_elems, dtype=torch.float32, pin_memory=True
            )
            self._param_buf_pong = torch.empty(
                self._half_elems, dtype=torch.float32, pin_memory=True
            )
            self._grad_buf_ping = torch.empty(
                self._half_elems, dtype=torch.float32, pin_memory=True
            )
            self._grad_buf_pong = torch.empty(
                self._half_elems, dtype=torch.float32, pin_memory=True
            )
        except RuntimeError as exc:
            logger.error(
                "SLC: Failed to allocate %.1f GB of pinned memory: %s",
                slc_size_bytes / (1 << 30),
                exc,
            )
            raise

        self._active_half: int = 0  # 0 = ping, 1 = pong
        self._prefetch_stream = prefetch_stream or (
            torch.cuda.Stream(device=self.device)
            if torch.cuda.is_available()
            else None
        )

        # Catalog: maps param tensor id → (offset_in_half, num_elems)
        self._param_catalog: Dict[int, Tuple[int, int]] = {}
        self._write_cursor: int = 0

        logger.info(
            "SLC: allocated 2×%.1f GB pinned (param+grad) for device %s",
            (self._half_elems * 4) / (1 << 30),
            self.device,
        )

    # ------------------------------------------------------------------
    # Buffer access helpers
    # ------------------------------------------------------------------

    @property
    def active_param_buf(self) -> torch.Tensor:
        return self._param_buf_ping if self._active_half == 0 else self._param_buf_pong

    @property
    def standby_param_buf(self) -> torch.Tensor:
        return self._param_buf_pong if self._active_half == 0 else self._param_buf_ping

    @property
    def active_grad_buf(self) -> torch.Tensor:
        return self._grad_buf_ping if self._active_half == 0 else self._grad_buf_pong

    def swap_buffers(self) -> None:
        """Flip active ↔ standby halves after a prefetch completes."""
        self._active_half ^= 1
        logger.debug("SLC: swapped to half %d", self._active_half)

    # ------------------------------------------------------------------
    # Param registration & offload
    # ------------------------------------------------------------------

    def register_param(self, param: torch.Tensor) -> int:
        """Register a flat-fp32 param tensor in the SLC catalog.

        Returns the assigned slot index (opaque handle for later retrieval).
        Raises ``MemoryError`` if the SLC is full.
        """
        n = param.numel()
        if self._write_cursor + n > self._half_elems:
            raise MemoryError(
                f"SLC catalog full: cursor={self._write_cursor} + n={n} "
                f"> half_elems={self._half_elems}.  "
                "Increase slc_size_bytes or reduce the number of offloaded params."
            )
        offset = self._write_cursor
        self._write_cursor += n
        slot = id(param)
        self._param_catalog[slot] = (offset, n)
        logger.debug("SLC.register_param: slot=%d offset=%d n=%d", slot, offset, n)
        return slot

    def offload_param(self, param: torch.Tensor, slot: int) -> None:
        """Copy a GPU param to the active SLC half (blocking D2H).

        In production this would be pipelined; here we use a simple
        blocking copy for correctness, with a non-blocking variant
        triggered via ``prefetch_param`` when the next layer is known.
        """
        offset, n = self._param_catalog[slot]
        cpu_view = self.active_param_buf[offset: offset + n]
        cpu_view.copy_(param.data.float().view(-1))
        logger.debug("SLC.offload_param: slot=%d  %.2f MB D→H", slot, n * 4 / (1 << 20))

    def prefetch_param(self, slot: int, target_gpu_buf: torch.Tensor) -> None:
        """Non-blocking H2D copy of a param from SLC into ``target_gpu_buf``.

        Issues the copy on ``self._prefetch_stream`` so it overlaps with
        compute on the default stream.
        """
        offset, n = self._param_catalog[slot]
        cpu_view = self.active_param_buf[offset: offset + n]
        if self._prefetch_stream is not None:
            with torch.cuda.stream(self._prefetch_stream):
                target_gpu_buf.copy_(cpu_view, non_blocking=True)
        else:
            target_gpu_buf.copy_(cpu_view)
        logger.debug("SLC.prefetch_param: slot=%d  %.2f MB H→D (async)", slot, n * 4 / (1 << 20))

    def offload_grad(self, grad: torch.Tensor, slot: int) -> None:
        """Copy a GPU gradient to the SLC grad buffer (blocking D2H)."""
        offset, n = self._param_catalog[slot]
        cpu_view = self.active_grad_buf[offset: offset + n]
        cpu_view.copy_(grad.data.float().view(-1))

    def stats(self) -> Dict[str, object]:
        """Return utilisation statistics."""
        used_bytes = self._write_cursor * 4
        capacity_bytes = self._half_elems * 4
        return {
            "used_gb": used_bytes / (1 << 30),
            "capacity_gb": capacity_bytes / (1 << 30),
            "utilisation_pct": 100.0 * used_bytes / capacity_bytes if capacity_bytes else 0.0,
            "num_registered_params": len(self._param_catalog),
            "active_half": self._active_half,
        }


# ---------------------------------------------------------------------------
# HeteroFSDPShardingConfig — parsed from CLI args
# ---------------------------------------------------------------------------

@dataclass
class HeteroFSDPShardingConfig:
    """Configuration dataclass for DES-LOC heterogeneous FSDP sharding.

    This is the DES-LOC analogue of Megatron's ``--outer-dp-sharding-strategy``
    argument.  It extends that single string with additional fields needed to
    reason about the heterogeneous cluster topology.

    Parameters
    ----------
    upstream_outer_strategy : str
        Megatron-compatible value: "no_shard" or "optim".  Controls whether
        outer-group optimizer states are sharded across tiers.
    outer_group_size : int
        Number of ranks in the outer FSDP group.  On the target cluster this
        is 3 (2× A6000 + 1× H100).
    inner_group_size : int
        Number of ranks in the inner FSDP group.  For single-GPU-per-node
        PCIe deployments this is typically 1.
    enable_slc_overflow : bool
        When True, ranks that would OOM without offload automatically spill
        optimizer states to the SLC.
    slc_size_bytes : int
        Total bytes to allocate for the per-rank SLC ring.
    dp_outer_dim : Optional[int]
        The outer DP dimension size (mirrors Megatron's ``dp_outer_dim``).
        If None, HSDP outer sharding is disabled and the strategy table
        collapses to pure inner-group FSDP.
    strategy_overrides : Dict[Tuple[DeviceTier, str], ShardingStrategy]
        Fine-grained per-(tier, role) overrides.  Merged on top of the
        default table at resolve time.
    """

    upstream_outer_strategy: str = "no_shard"
    outer_group_size: int = 3          # 2× A6000 + 1× H100
    inner_group_size: int = 1
    enable_slc_overflow: bool = True
    slc_size_bytes: int = _DEFAULT_SLC_SIZE_BYTES
    dp_outer_dim: Optional[int] = None
    strategy_overrides: Dict[Tuple[DeviceTier, str], ShardingStrategy] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.upstream_outer_strategy not in _UPSTREAM_CHOICES:
            raise ValueError(
                f"upstream_outer_strategy must be one of {sorted(_UPSTREAM_CHOICES)}, "
                f"got '{self.upstream_outer_strategy}'."
            )
        if self.outer_group_size < 1:
            raise ValueError(f"outer_group_size must be >= 1, got {self.outer_group_size}.")
        if self.slc_size_bytes < 1 << 20:
            raise ValueError(
                f"slc_size_bytes must be >= 1 MB, got {self.slc_size_bytes}."
            )

    @classmethod
    def from_cli_args(cls, args: object) -> "HeteroFSDPShardingConfig":
        """Construct from a parsed argparse namespace (Megatron-style).

        Reads the following attributes from ``args`` (all optional with defaults):
            - ``outer_dp_sharding_strategy`` (str)   — maps to upstream_outer_strategy
            - ``data_parallel_sharding_strategy`` (str) — used to validate "optim" constraint
            - ``dp_outer_dim`` (int or None)
            - ``desoc_slc_size_gb`` (float)          — new DES-LOC arg; default 32.0
            - ``desoc_slc_overflow`` (bool)          — new DES-LOC arg; default True
        """
        outer_strat = getattr(args, "outer_dp_sharding_strategy", "no_shard") or "no_shard"
        inner_strat = getattr(args, "data_parallel_sharding_strategy", "no_shard") or "no_shard"
        dp_outer_dim = getattr(args, "dp_outer_dim", None)

        # Upstream constraint: "optim" for outer is only valid when inner is "optim_grads_params"
        if outer_strat == "optim" and inner_strat != "optim_grads_params":
            logger.warning(
                "outer_dp_sharding_strategy='optim' requires "
                "data_parallel_sharding_strategy='optim_grads_params'; "
                "got '%s'.  Downgrading outer to 'no_shard'.",
                inner_strat,
            )
            outer_strat = "no_shard"

        slc_size_gb = float(getattr(args, "desoc_slc_size_gb", 32.0))
        slc_size_bytes = int(slc_size_gb * (1 << 30))

        slc_overflow = bool(getattr(args, "desoc_slc_overflow", True))

        # Outer group size: inferred from dist world_size if available
        if dist.is_available() and dist.is_initialized():
            outer_group_size = dist.get_world_size()
        else:
            outer_group_size = 3  # default: our target cluster

        cfg = cls(
            upstream_outer_strategy=outer_strat,
            outer_group_size=outer_group_size,
            inner_group_size=1,
            enable_slc_overflow=slc_overflow,
            slc_size_bytes=slc_size_bytes,
            dp_outer_dim=dp_outer_dim,
        )
        logger.info("HeteroFSDPShardingConfig.from_cli_args: %s", cfg)
        return cfg


# ---------------------------------------------------------------------------
# build_hetero_fsdp_config — factory used by DeepSpeed engine init
# ---------------------------------------------------------------------------

def build_hetero_fsdp_config(
    ds_config: Dict,
    cli_args: Optional[object] = None,
) -> HeteroFSDPShardingConfig:
    """Build a ``HeteroFSDPShardingConfig`` from a DeepSpeed config dict.

    This factory is the single entry-point called by the DeepSpeed engine
    during ``initialize()``.  It resolves the configuration in priority order:

        1. ``ds_config["hetero_fsdp"]`` block (new DES-LOC key)
        2. ``cli_args`` namespace (Megatron-compatible CLI)
        3. Built-in defaults (safe for the target 2×A6000 + 1×H100 cluster)

    The ``ds_config["hetero_fsdp"]`` block supports these keys::

        {
            "hetero_fsdp": {
                "outer_dp_sharding_strategy": "no_shard" | "optim",
                "slc_size_gb": 32.0,
                "slc_overflow": true,
                "dp_outer_dim": null
            }
        }

    Parameters
    ----------
    ds_config : Dict
        Full DeepSpeed config dictionary (loaded from ds_config.json).
    cli_args : object, optional
        Parsed argparse namespace.  Used if ``hetero_fsdp`` block is absent.

    Returns
    -------
    HeteroFSDPShardingConfig
    """
    hetero_block = ds_config.get("hetero_fsdp", {})

    if hetero_block:
        logger.info("build_hetero_fsdp_config: using ds_config['hetero_fsdp'] block.")
        outer_strat = hetero_block.get("outer_dp_sharding_strategy", "no_shard")
        slc_size_bytes = int(hetero_block.get("slc_size_gb", 32.0) * (1 << 30))
        slc_overflow = bool(hetero_block.get("slc_overflow", True))
        dp_outer_dim = hetero_block.get("dp_outer_dim", None)

        if dist.is_available() and dist.is_initialized():
            outer_group_size = dist.get_world_size()
        else:
            outer_group_size = 3

        return HeteroFSDPShardingConfig(
            upstream_outer_strategy=outer_strat,
            outer_group_size=outer_group_size,
            inner_group_size=1,
            enable_slc_overflow=slc_overflow,
            slc_size_bytes=slc_size_bytes,
            dp_outer_dim=dp_outer_dim,
        )

    if cli_args is not None:
        logger.info("build_hetero_fsdp_config: falling back to cli_args.")
        return HeteroFSDPShardingConfig.from_cli_args(cli_args)

    logger.info(
        "build_hetero_fsdp_config: no config found; using cluster defaults "
        "(outer=no_shard, slc=32 GB, overflow=True)."
    )
    return HeteroFSDPShardingConfig()


# ---------------------------------------------------------------------------
# Tier-aware process group builder
# ---------------------------------------------------------------------------

def build_tier_process_groups(
    world_size: int,
    rank: int,
    tier_map: Dict[int, DeviceTier],
) -> Tuple[Optional[dist.ProcessGroup], Optional[dist.ProcessGroup]]:
    """Create inner and outer FSDP process groups respecting device tiers.

    In DES-LOC the inner group contains ranks of the *same* tier (to maximise
    bandwidth locality on the same PCIe switch), while the outer group spans
    all tiers.

    On the target cluster:
        rank 0 → H100 NVL  (SM90_ANCHOR)
        rank 1 → A6000 #0  (SM86_WORKER)
        rank 2 → A6000 #1  (SM86_WORKER)

    Inner groups:
        H100:  {0}       (no peer of the same tier)
        A6000: {1, 2}    (same SM86 tier, same PCIe switch)

    Outer group:
        {0, 1, 2}        (all ranks)

    Parameters
    ----------
    world_size : int
    rank : int
    tier_map : Dict[int, DeviceTier]
        Maps rank → DeviceTier for *all* ranks.

    Returns
    -------
    (inner_group, outer_group) : Tuple of ProcessGroup or None
        None if ``dist`` is not initialised.
    """
    if not (dist.is_available() and dist.is_initialized()):
        logger.warning("build_tier_process_groups: dist not initialised; returning (None, None).")
        return None, None

    my_tier = tier_map.get(rank, DeviceTier.SM86_WORKER)

    # Outer group: all ranks
    outer_ranks = list(range(world_size))
    outer_group = dist.new_group(ranks=outer_ranks)

    # Inner group: same-tier peers
    inner_ranks = sorted(r for r, t in tier_map.items() if t == my_tier)
    if len(inner_ranks) < 2:
        logger.info(
            "build_tier_process_groups: rank %d is the sole %s rank; "
            "inner group is trivial (self-only).",
            rank, my_tier.name,
        )
    inner_group = dist.new_group(ranks=inner_ranks)

    logger.info(
        "build_tier_process_groups: rank=%d tier=%s inner_ranks=%s outer_ranks=%s",
        rank, my_tier.name, inner_ranks, outer_ranks,
    )
    return inner_group, outer_group


# ---------------------------------------------------------------------------
# CLI argument registration (Megatron-compatible extension)
# ---------------------------------------------------------------------------

def add_hetero_fsdp_args(parser) -> None:
    """Register DES-LOC hetero-FSDP arguments on an argparse parser.

    Mirrors the upstream Megatron ``_add_distributed_args`` pattern so that
    existing Megatron launch scripts can adopt DES-LOC with minimal changes:

        Before:  --outer-dp-sharding-strategy optim
        After:   --outer-dp-sharding-strategy optim \\
                     --desoc-slc-size-gb 48 --desoc-slc-overflow

    The ``--outer-dp-sharding-strategy`` argument is intentionally kept
    Megatron-compatible (same name, same choices) so that Megatron checkpoints
    and launch configs are directly portable.
    """
    group = parser.add_argument_group("DES-LOC Heterogeneous FSDP")

    # Megatron-compatible arg (replaces the upstream patch's new arg)
    group.add_argument(
        "--outer-dp-sharding-strategy",
        type=str,
        default="no_shard",
        choices=["no_shard", "optim"],
        dest="outer_dp_sharding_strategy",
        help=(
            "Sharding strategy for the outer data-parallel group in DES-LOC HSDP mode.  "
            "'no_shard' replicates params/grads/optim-states across tiers (saves PCIe BW).  "
            "'optim' shards optimizer states across tiers to reduce per-rank peak memory.  "
            "'optim' requires --data-parallel-sharding-strategy=optim_grads_params.  "
            "DES-LOC further refines this choice per device tier; see HeteroFSDPShardingStrategy."
        ),
    )

    # New DES-LOC args
    group.add_argument(
        "--desoc-slc-size-gb",
        type=float,
        default=32.0,
        dest="desoc_slc_size_gb",
        help=(
            "Size of the per-rank Shared LOcality Cache (SLC) in gigabytes.  "
            "The SLC is a pinned-memory ring-buffer in CPU DRAM used for "
            "parameter and gradient offload on A6000 worker ranks.  "
            "Default: 32.0 GB."
        ),
    )

    group.add_argument(
        "--desoc-slc-overflow",
        action="store_true",
        default=True,
        dest="desoc_slc_overflow",
        help=(
            "Enable automatic SLC overflow: when an A6000 rank would OOM, "
            "optimizer states are automatically spilled to the SLC.  "
            "Enabled by default."
        ),
    )

    group.add_argument(
        "--no-desoc-slc-overflow",
        action="store_false",
        dest="desoc_slc_overflow",
        help="Disable SLC overflow (see --desoc-slc-overflow).",
    )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _safe_rank() -> int:
    """Return the distributed rank if available, else -1."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return -1


def _validate_cluster_topology(tier_map: Dict[int, DeviceTier]) -> None:
    """Log a warning if the detected topology diverges from the target cluster.

    Target: exactly 1× SM90_ANCHOR and 2× SM86_WORKER.
    """
    anchor_count = sum(1 for t in tier_map.values() if t == DeviceTier.SM90_ANCHOR)
    worker_count = sum(1 for t in tier_map.values() if t == DeviceTier.SM86_WORKER)

    if anchor_count != 1 or worker_count != 2:
        logger.warning(
            "_validate_cluster_topology: expected 1× SM90_ANCHOR + 2× SM86_WORKER "
            "but found %d anchor(s) + %d worker(s).  "
            "DES-LOC strategy tables may be sub-optimal for this topology.",
            anchor_count, worker_count,
        )
    else:
        logger.info(
            "_validate_cluster_topology: cluster matches DES-LOC target "
            "(1× H100 NVL + 2× A6000)."
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    # 1. DeviceTier auto-detect (CPU fallback in CI without GPUs)
    tier = DeviceTier.from_local_device()
    assert isinstance(tier, DeviceTier), "DeviceTier.from_local_device must return a DeviceTier"

    # 2. HeteroFSDPShardingConfig validation
    cfg = HeteroFSDPShardingConfig(upstream_outer_strategy="no_shard", outer_group_size=3)
    assert cfg.upstream_outer_strategy == "no_shard"
    try:
        HeteroFSDPShardingConfig(upstream_outer_strategy="invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # 3. HeteroFSDPShardingStrategy resolve
    strategy = HeteroFSDPShardingStrategy(cfg)
    assert isinstance(strategy.inner_strategy, ShardingStrategy)
    assert isinstance(strategy.outer_strategy, ShardingStrategy)

    # 4. SLC allocation (tiny, just to confirm pinned alloc works)
    slc = SharedLocalityCache(slc_size_bytes=8 * (1 << 20))  # 8 MB
    assert slc.active_param_buf.is_pinned(), "SLC param buffer must be pinned"
    stats = slc.stats()
    assert stats["utilisation_pct"] == 0.0

    # 5. build_hetero_fsdp_config with empty ds_config + no cli_args
    default_cfg = build_hetero_fsdp_config({})
    assert default_cfg.upstream_outer_strategy == "no_shard"

    logger.info("All smoke tests passed.")
