"""
DES-LOC Heterogeneous Checkpoint Configuration
===============================================

Upstream Design Intent (Megatron d4f9347):
    Megatron-LM refactored flat argparse checkpoint arguments into a structured
    ``CheckpointConfig`` dataclass via ``ArgumentGroupFactory``, centralizing all
    checkpoint-related settings (save/load paths, intervals, formats, async behavior,
    replication, and optimizer state handling) in one typed, self-documenting config
    object.  The key insight is that checkpoint behavior is complex enough to warrant
    first-class schema ownership rather than scattered ``add_argument`` calls.

DES-LOC Adaptation Points:
    In Neuron_SP's DES-LOC (Decoupled Execution with Shared LOcality Cache) framework,
    checkpoint semantics must account for **three** fundamentally different execution
    domains sharing a single logical training step:

    1. **A6000 x2 (SM86, 48 GB each)**  — "Locality Workers"
       These hold the Shared LOcality Cache (SLC): gradient accumulators, activation
       re-materialization buffers, and optimizer second-moment statistics that are
       too large for H100 VRAM.  Their checkpoint state is *incremental* — only dirty
       cache lines need to be flushed.

    2. **H100 NVL (SM90, 96 GB)**  — "Decoupled Executor"
       Holds model parameters (FP16/BF16) and runs forward/backward.  Its checkpoint
       is the *authoritative* parameter snapshot.  SM90-specific kernels (e.g.,
       warp-specialised attention) produce state that SM86 cannot reconstruct, so the
       H100 checkpoint must be saved before any A6000 flush begins.

    3. **CPU DRAM (1.5 TB)**  — "Spill Arena"
       Optimizer states that overflow both GPU tiers live here.  CPU checkpoint is
       written asynchronously using a persistent background thread to avoid blocking
       the H100 compute stream.

    Additional DES-LOC-specific fields added beyond Megatron's CheckpointConfig:
    - ``slc_flush_policy``    : controls when A6000 SLC is checkpointed relative to H100
    - ``spill_arena_ckpt_dir``: separate path for CPU-resident optimizer state
    - ``h100_saves_first``    : enforce ordering (H100 param save → A6000 SLC flush)
    - ``slc_incremental_save``: only serialize dirty SLC entries (saves ~70% of A6000 I/O)
    - ``cpu_async_worker_threads``: thread-pool size for CPU spill-arena serialization
    - ``sm86_ckpt_dtype``     : allow A6000 to save SLC in lower precision than H100
    - ``replication_topo``    : heterogeneous replication (cross-tier vs. same-tier)

    Fields inherited verbatim from Megatron's CheckpointConfig that remain valid:
    ``save``, ``save_interval``, ``load``, ``ckpt_format``, ``async_save``,
    ``dist_ckpt_strictness``, ``finetune``, ``ckpt_step``, ``auto_detect_ckpt_format``,
    ``ckpt_fully_parallel_save``, ``ckpt_fully_parallel_load``.

    Fields dropped / transformed:
    - ``replication`` / ``replication_jump`` / ``replication_factor`` are replaced by
      the richer ``replication_topo`` dict (heterogeneous tiers cannot share a single
      scalar jump parameter).
    - ``non_persistent_ckpt_type = "in_memory"`` maps to DES-LOC SLC hot-standby.
    - ``distrib_optim_fully_reshardable_mem_efficient`` becomes ``cpu_async_worker_threads``
      because on this PCIe-only topology the bottleneck is CPU↔GPU bandwidth, not NCCL.

PCIe Topology Note:
    All three devices are connected via PCIe with **no NVLink**.  This means:
    - H100↔A6000 parameter copies are bandwidth-limited (~32 GB/s peak, ~20 GB/s realistic).
    - The checkpoint serializer avoids unnecessary cross-tier copies: each tier saves
      its own state independently and only a lightweight metadata manifest is exchanged.
    - ``h100_saves_first=True`` (default) ensures the authoritative parameter snapshot
      exists on stable storage before A6000 SLC data is written, providing a consistent
      recovery point even if an A6000 dies mid-flush.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict, fields
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SLCFlushPolicy(str, Enum):
    """Controls when A6000 Shared LOcality Cache state is checkpointed.

    Megatron has no equivalent — this is pure DES-LOC.

    ``AFTER_H100``:  (default, recommended)
        H100 writes authoritative parameters first; A6000 SLC flush begins only
        after H100 fsync completes.  Guarantees a consistent recovery point.

    ``PARALLEL``:
        H100 and A6000 flush concurrently.  Faster wall-clock time but requires
        both tiers to succeed for a valid checkpoint.  Use only with redundant
        storage (e.g., RAID-backed NAS).

    ``SLC_ONLY``:
        Only A6000 SLC state is saved.  Useful for mid-epoch hot-standby where
        H100 parameters have not changed since the last full save.

    ``SKIP``:
        Do not checkpoint A6000 SLC.  The SLC will be recomputed from H100
        state on resume (costs extra forward passes but saves I/O).
    """
    AFTER_H100 = "after_h100"
    PARALLEL = "parallel"
    SLC_ONLY = "slc_only"
    SKIP = "skip"


class ReplicationTier(str, Enum):
    """Which tier(s) participate in checkpoint replication.

    Megatron's scalar ``replication_factor`` / ``replication_jump`` assume a
    homogeneous GPU pool.  DES-LOC needs per-tier control.

    ``NONE``        : No replication.
    ``H100_ONLY``   : Replicate H100 param shard only (cheapest, most critical).
    ``A6000_ONLY``  : Replicate A6000 SLC only.
    ``CROSS_TIER``  : H100 param shard replicated to CPU DRAM; A6000 SLC
                      replicated to H100 VRAM hot-buffer.  Expensive but gives
                      full redundancy on a single node.
    ``ALL``         : Both H100 and A6000 state replicated independently.
    """
    NONE = "none"
    H100_ONLY = "h100_only"
    A6000_ONLY = "a6000_only"
    CROSS_TIER = "cross_tier"
    ALL = "all"


class CkptFormat(str, Enum):
    """Checkpoint serialization format — mirrors Megatron's ``ckpt_format`` choices
    with an added ``deslock_native`` option for DES-LOC's split-manifest format.

    ``TORCH``         : ``torch.save`` / ``torch.load`` (legacy, monolithic).
    ``TORCH_DIST``    : Megatron distributed checkpoint (sharded, reshardable).
    ``TORCH_DCP``     : ``torch.distributed.checkpoint`` format.
    ``DESLOCK_NATIVE``: DES-LOC's own format: separate H100, A6000, and CPU
                        shard files with a unified JSON manifest.  Required when
                        using ``slc_incremental_save=True``.
    """
    TORCH = "torch"
    TORCH_DIST = "torch_dist"
    TORCH_DCP = "torch_dcp"
    DESLOCK_NATIVE = "deslock_native"


class DistCkptStrictness(str, Enum):
    """Key-mismatch handling during distributed checkpoint load.

    Mirrors Megatron's ``StrictHandling`` enum.  Values are identical so that
    checkpoints written by Megatron can be loaded by DES-LOC with the same
    strictness semantics.
    """
    ASSUME_OK_UNEXPECTED = "assume_ok_unexpected"
    LOG_UNEXPECTED = "log_unexpected"
    LOG_ALL = "log_all"
    RAISE_UNEXPECTED = "raise_unexpected"
    RAISE_ALL = "raise_all"
    RETURN_UNEXPECTED = "return_unexpected"
    RETURN_ALL = "return_all"
    IGNORE_ALL = "ignore_all"


# ---------------------------------------------------------------------------
# DES-LOC Replication Topology
# ---------------------------------------------------------------------------

@dataclass
class HeteroReplicationTopology:
    """Describes per-tier replication for heterogeneous hardware.

    Replaces Megatron's scalar ``replication`` / ``replication_jump`` /
    ``replication_factor`` triplet which assumes uniform GPU ranks.

    In DES-LOC there is (at most) one H100 and two A6000s per node, so
    "rank spacing" is meaningless.  Instead we express replication as a
    destination-tier mapping.

    Attributes:
        tier (ReplicationTier): Which tier(s) to replicate.
        h100_replica_on_cpu (bool): If True, the H100 parameter shard is
            shadowed in CPU DRAM asynchronously after each H100 save.
            Useful as a fast local backup without extra GPU memory.
        a6000_replica_peer (bool): If True, each A6000 replicates its SLC
            partition to the *other* A6000 (peer-to-peer via PCIe).  Requires
            both A6000s to be alive; provides protection against single-GPU
            failure.
        cpu_replica_compression (bool): Compress CPU-resident replicas with
            LZ4 to reduce DRAM footprint.  ~3× compression on typical
            optimizer state; ~15% CPU overhead.
    """
    tier: ReplicationTier = ReplicationTier.NONE
    h100_replica_on_cpu: bool = False
    a6000_replica_peer: bool = False
    cpu_replica_compression: bool = False

    def validate(self) -> None:
        if self.tier == ReplicationTier.CROSS_TIER and not self.h100_replica_on_cpu:
            logger.warning(
                "ReplicationTier.CROSS_TIER selected but h100_replica_on_cpu=False; "
                "H100→CPU replication will be skipped.  Set h100_replica_on_cpu=True "
                "or choose a different tier."
            )
        if self.a6000_replica_peer and self.tier == ReplicationTier.H100_ONLY:
            raise ValueError(
                "a6000_replica_peer=True conflicts with ReplicationTier.H100_ONLY. "
                "Use ReplicationTier.A6000_ONLY or ALL."
            )


# ---------------------------------------------------------------------------
# Core HeteroCheckpointConfig dataclass
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class HeteroCheckpointConfig:
    """Checkpoint configuration for DES-LOC heterogeneous training.

    Mirrors Megatron-LM's ``CheckpointConfig`` (introduced in d4f9347) and extends
    it with fields specific to the A6000 × 2 + H100 NVL topology used by Neuron_SP.

    Field groups
    ------------
    (A) Core save/load paths — identical semantics to Megatron.
    (B) Save/load intervals — identical semantics to Megatron.
    (C) Format & strictness — identical semantics to Megatron.
    (D) Optimizer & RNG state — identical semantics to Megatron.
    (E) Async & parallelism — partially transformed (see docstring above).
    (F) DES-LOC–specific — new fields with no Megatron equivalent.
    """

    # ------------------------------------------------------------------
    # (A) Core paths
    # ------------------------------------------------------------------

    save: Optional[str] = None
    """Output directory for persistent checkpoints (H100 param shard + manifest).
    A6000 SLC data is written to ``{save}/slc/`` unless ``slc_save_dir`` overrides it.
    CPU spill-arena state is written to ``spill_arena_ckpt_dir`` if set, else
    ``{save}/cpu_spill/``.
    """

    load: Optional[str] = None
    """Directory from which to resume.  DES-LOC will load H100 params first,
    then restore A6000 SLC (if present and ``slc_flush_policy != SKIP``),
    then reload CPU spill-arena state.
    """

    pretrained_checkpoint: Optional[str] = None
    """Pretrained checkpoint for fine-tuning.  Only H100 parameters and
    (optionally) embeddings are loaded; A6000 SLC is cold-started.
    """

    # ------------------------------------------------------------------
    # (B) Intervals
    # ------------------------------------------------------------------

    save_interval: Optional[int] = None
    """Number of training iterations between full (H100 + A6000 + CPU) saves."""

    save_wgrads_interval: Optional[int] = None
    """Iterations between wgrad (main_grad) saves.  In DES-LOC, main_grads live
    on A6000 SLC; this interval triggers a partial SLC flush of grad buffers only.
    """

    save_dgrads_interval: Optional[int] = None
    """Iterations between dgrad saves (A6000 SLC partition only)."""

    save_retain_interval: Optional[int] = None
    """Iterations between retained (non-rotating) checkpoints.  All other
    checkpoints except the latest retained one are pruned automatically.
    """

    non_persistent_save_interval: Optional[int] = None
    """Iterations between non-persistent (hot-standby) saves.
    In DES-LOC this maps to an in-SLC snapshot: the A6000's shared cache
    takes a consistent copy without writing to storage.
    """

    ckpt_step: Optional[int] = None
    """Specific training step to load from (overrides latest)."""

    # ------------------------------------------------------------------
    # (C) Format & strictness
    # ------------------------------------------------------------------

    ckpt_format: CkptFormat = CkptFormat.DESLOCK_NATIVE
    """Serialization format.  Defaults to ``deslock_native`` (split-manifest)
    for full heterogeneous support.  Set to ``torch_dist`` for Megatron
    cross-compatibility.
    """

    auto_detect_ckpt_format: bool = False
    """Auto-detect format of existing checkpoint on disk.  Adds one extra
    manifest read on rank-0 at resume time.
    """

    ckpt_convert_format: Optional[Literal["torch", "torch_dist"]] = None
    """Target format when converting an existing checkpoint (offline tool)."""

    ckpt_convert_save: Optional[str] = None
    """Destination directory for converted checkpoint."""

    dist_ckpt_strictness: DistCkptStrictness = DistCkptStrictness.ASSUME_OK_UNEXPECTED
    """Key-mismatch policy when loading distributed checkpoints from storage.
    Does not affect ``load_state_dict`` strictness for model weights.
    """

    ckpt_fully_parallel_save: bool = True
    """Parallelize save across DP ranks.  In DES-LOC this means H100 and the
    two A6000s write their shards simultaneously (PCIe-limited; ~3× throughput
    vs. sequential).
    """

    ckpt_fully_parallel_load: bool = False
    """Parallelize load across DP ranks."""

    ckpt_assume_constant_structure: bool = False
    """Assume model/optimizer state-dict structure is constant across saves.
    Enables caching of shard metadata between saves to reduce manifest overhead.
    """

    # ------------------------------------------------------------------
    # (D) Optimizer & RNG state
    # ------------------------------------------------------------------

    save_optim: bool = True
    """Include optimizer state in checkpoint."""

    save_rng: bool = True
    """Include RNG state in checkpoint."""

    load_optim: bool = True
    """Restore optimizer state from checkpoint."""

    load_rng: bool = True
    """Restore RNG state from checkpoint."""

    load_main_params_from_ckpt: bool = False
    """When resuming without optimizer state, explicitly copy main params (FP32)
    from checkpoint into the H100 parameter buffers.  Required for FP16/BF16
    mixed-precision runs where main params are stored separately.
    """

    finetune: bool = False
    """Fine-tuning mode: load weights only; reset optimizer state and iteration
    counter to zero.  A6000 SLC is cold-started.
    """

    use_checkpoint_args: bool = False
    """Override model-architecture CLI args with values from checkpoint metadata."""

    use_mp_args_from_checkpoint_args: bool = False
    """Copy tensor/pipeline parallelism args from checkpoint."""

    use_tokenizer_model_from_checkpoint_args: bool = True
    """Use tokenizer path recorded in checkpoint metadata."""

    exit_on_missing_checkpoint: bool = False
    """Exit (rather than random-init) if ``load`` path has no checkpoint."""

    # ------------------------------------------------------------------
    # (E) Async & background workers
    # ------------------------------------------------------------------

    async_save: bool = False
    """Async checkpoint save.  In DES-LOC: H100 param shard is serialized on a
    CUDA stream pinned to the H100; A6000 SLC flush runs on a separate CPU
    thread pool.  CPU spill-arena is always async regardless of this flag.
    Works only with ``ckpt_format in {torch_dist, deslock_native}``.
    """

    use_persistent_ckpt_worker: bool = False
    """Keep the async checkpoint worker threads alive between saves (persistent
    thread pool).  When False, worker threads are spawned per-save and torn down
    after fsync.  Persistent workers reduce per-save latency at the cost of
    background CPU utilization.
    """

    cpu_async_worker_threads: int = 4
    """Number of threads in the CPU spill-arena serialization pool.
    Each thread handles one optimizer-state shard from CPU DRAM.
    On a 1.5 TB DRAM system, 4 threads saturate a typical NVMe array without
    starving the compute process.

    DES-LOC replaces Megatron's ``distrib_optim_fully_reshardable_mem_efficient``
    (which uses Gloo to reduce peak GPU memory) with this thread-pool approach
    because on a PCIe-only topology the bottleneck is CPU↔storage bandwidth,
    not GPU↔CPU NCCL collectives.
    """

    dist_ckpt_optim_fully_reshardable: bool = False
    """Enable full TP/PP/EP/DP reshardability for optimizer distributed ckpt."""

    # ------------------------------------------------------------------
    # (F) DES-LOC–specific fields (no Megatron equivalent)
    # ------------------------------------------------------------------

    slc_flush_policy: SLCFlushPolicy = SLCFlushPolicy.AFTER_H100
    """Controls ordering of H100 param save vs A6000 SLC flush.
    See ``SLCFlushPolicy`` docstring for semantics.
    Default ``AFTER_H100`` guarantees a consistent recovery point on PCIe
    topologies where H100↔A6000 copy would be needed to reconstruct SLC.
    """

    slc_save_dir: Optional[str] = None
    """Override directory for A6000 SLC checkpoint files.
    If None, defaults to ``{save}/slc/``.
    Useful when SLC data should go to a local SSD while H100 params go to NAS.
    """

    slc_incremental_save: bool = True
    """Only serialize A6000 SLC entries that have been modified since the last
    checkpoint (dirty-page tracking).  Reduces A6000 I/O by ~70% on typical
    gradient-accumulation workloads.  Requires ``ckpt_format=deslock_native``.
    """

    spill_arena_ckpt_dir: Optional[str] = None
    """Directory for CPU spill-arena optimizer state checkpoints.
    If None, defaults to ``{save}/cpu_spill/``.
    """

    h100_saves_first: bool = True
    """Enforce that H100 parameter shard is fully written and fsynced before
    A6000 SLC flush begins.  When ``slc_flush_policy=PARALLEL`` this flag is
    ignored (both tiers write simultaneously).
    """

    sm86_ckpt_dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"
    """Precision for A6000 (SM86) SLC checkpoint serialization.
    The SLC contains gradient accumulators and optimizer moments; saving in
    BF16 halves checkpoint size with negligible accuracy impact on resume.
    H100 (SM90) parameters are always saved in their native dtype (usually BF16).
    """

    replication_topo: HeteroReplicationTopology = field(
        default_factory=HeteroReplicationTopology
    )
    """Heterogeneous replication topology.  Replaces Megatron's scalar
    ``replication`` / ``replication_jump`` / ``replication_factor`` fields.
    See ``HeteroReplicationTopology`` for full semantics.
    """

    slc_hot_standby_enabled: bool = False
    """Enable in-SLC hot-standby snapshots (maps to Megatron's
    ``non_persistent_ckpt_type="in_memory"``).  A consistent SLC snapshot is
    taken every ``non_persistent_save_interval`` steps without touching storage.
    On failure, the hot-standby allows sub-second recovery for A6000 state.
    """

    manifest_version: int = 2
    """DES-LOC checkpoint manifest schema version.  Version 1 is single-tier
    (H100 only, for Megatron compatibility).  Version 2 is the full tri-tier
    manifest (H100 + A6000 + CPU).
    """

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Cross-field consistency checks with clear error messages."""
        if self.slc_incremental_save and self.ckpt_format != CkptFormat.DESLOCK_NATIVE:
            raise ValueError(
                f"slc_incremental_save=True requires ckpt_format=deslock_native, "
                f"got ckpt_format={self.ckpt_format}.  Either set "
                f"ckpt_format=CkptFormat.DESLOCK_NATIVE or disable slc_incremental_save."
            )

        if (
            self.slc_flush_policy == SLCFlushPolicy.PARALLEL
            and self.h100_saves_first
        ):
            logger.warning(
                "slc_flush_policy=PARALLEL and h100_saves_first=True are contradictory; "
                "PARALLEL policy overrides h100_saves_first.  Set h100_saves_first=False "
                "to suppress this warning when using PARALLEL policy."
            )

        if self.async_save and self.ckpt_format not in (
            CkptFormat.TORCH_DIST, CkptFormat.DESLOCK_NATIVE
        ):
            raise ValueError(
                f"async_save=True requires ckpt_format in {{torch_dist, deslock_native}}, "
                f"got {self.ckpt_format}."
            )

        if self.cpu_async_worker_threads < 1:
            raise ValueError(
                f"cpu_async_worker_threads must be >= 1, got {self.cpu_async_worker_threads}."
            )

        if self.save_interval is not None and self.save_interval < 1:
            raise ValueError(f"save_interval must be >= 1, got {self.save_interval}.")

        if self.slc_hot_standby_enabled and self.non_persistent_save_interval is None:
            raise ValueError(
                "slc_hot_standby_enabled=True requires non_persistent_save_interval to be set."
            )

        self.replication_topo.validate()

        logger.debug(
            "HeteroCheckpointConfig validated: ckpt_format=%s, slc_flush_policy=%s, "
            "async_save=%s, slc_incremental_save=%s, replication=%s",
            self.ckpt_format,
            self.slc_flush_policy,
            self.async_save,
            self.slc_incremental_save,
            self.replication_topo.tier,
        )

    # ------------------------------------------------------------------
    # Derived path helpers
    # ------------------------------------------------------------------

    def resolved_slc_dir(self) -> Optional[Path]:
        """Return the resolved SLC checkpoint directory.

        Returns None if ``save`` is not set (no checkpointing configured).
        """
        if self.slc_save_dir:
            return Path(self.slc_save_dir)
        if self.save:
            return Path(self.save) / "slc"
        return None

    def resolved_cpu_spill_dir(self) -> Optional[Path]:
        """Return the resolved CPU spill-arena checkpoint directory."""
        if self.spill_arena_ckpt_dir:
            return Path(self.spill_arena_ckpt_dir)
        if self.save:
            return Path(self.save) / "cpu_spill"
        return None

    def resolved_h100_dir(self) -> Optional[Path]:
        """Return the H100 parameter shard directory (always under ``save``)."""
        if self.save:
            return Path(self.save) / "h100"
        return None

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to a JSON-serializable dict (for manifest embedding)."""
        d = asdict(self)
        # Enums → string values for JSON
        d["ckpt_format"] = self.ckpt_format.value
        d["slc_flush_policy"] = self.slc_flush_policy.value
        d["dist_ckpt_strictness"] = self.dist_ckpt_strictness.value
        d["replication_topo"]["tier"] = self.replication_topo.tier.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HeteroCheckpointConfig":
        """Reconstruct from a previously serialized dict."""
        d = dict(d)  # shallow copy
        d["ckpt_format"] = CkptFormat(d["ckpt_format"])
        d["slc_flush_policy"] = SLCFlushPolicy(d["slc_flush_policy"])
        d["dist_ckpt_strictness"] = DistCkptStrictness(d["dist_ckpt_strictness"])
        topo_dict = d.pop("replication_topo", {})
        topo_dict["tier"] = ReplicationTier(topo_dict.get("tier", "none"))
        d["replication_topo"] = HeteroReplicationTopology(**topo_dict)
        return cls(**d)

    def save_to_manifest(self, manifest_path: Path) -> None:
        """Write config as JSON into a checkpoint manifest file.

        The manifest is the single source of truth for what tiers were saved
        and with which settings, enabling safe cross-version resume.
        """
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "deslock_manifest_version": self.manifest_version,
            "checkpoint_config": self.to_dict(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        tmp = manifest_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, manifest_path)  # atomic rename
        logger.info("DES-LOC checkpoint manifest written: %s", manifest_path)

    @classmethod
    def load_from_manifest(cls, manifest_path: Path) -> "HeteroCheckpointConfig":
        """Reconstruct config from an existing checkpoint manifest."""
        with open(manifest_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        version = payload.get("deslock_manifest_version", 1)
        if version < 2:
            logger.warning(
                "Manifest version %d detected (expected 2); some DES-LOC fields "
                "will use defaults.  Consider converting the checkpoint.",
                version,
            )
        return cls.from_dict(payload["checkpoint_config"])


# ---------------------------------------------------------------------------
# Tier-aware save-order planner
# ---------------------------------------------------------------------------

class TierSavePlanner:
    """Determines the concrete serialization order for a DES-LOC checkpoint.

    Given a ``HeteroCheckpointConfig``, emits an ordered list of
    ``(tier_name, save_fn_key)`` tuples that the checkpoint engine executes.

    This is the runtime counterpart to the static config: it translates
    ``slc_flush_policy`` and ``h100_saves_first`` into a concrete schedule.

    The planner is intentionally stateless (pure function over config) so it
    can be unit-tested without a live GPU cluster.
    """

    TIER_H100 = "h100"
    TIER_A6000 = "a6000_slc"
    TIER_CPU = "cpu_spill"

    def __init__(self, config: HeteroCheckpointConfig) -> None:
        self.config = config
        self._plan: list[Tuple[str, bool]] = []  # (tier, is_async)

    def build_plan(self) -> list[Tuple[str, bool]]:
        """Return ordered list of (tier_name, run_async) for the current config.

        ``run_async=True`` means the save is dispatched to a background worker.
        The caller is responsible for joining all async workers before returning
        from the checkpoint call.
        """
        cfg = self.config
        policy = cfg.slc_flush_policy
        plan: list[Tuple[str, bool]] = []

        if policy == SLCFlushPolicy.SKIP:
            # H100 + CPU only; A6000 SLC intentionally omitted.
            plan.append((self.TIER_H100, cfg.async_save))
            plan.append((self.TIER_CPU, True))  # CPU is always async
            logger.debug("TierSavePlanner: SLC_SKIP policy — A6000 omitted from plan.")

        elif policy == SLCFlushPolicy.SLC_ONLY:
            # A6000 SLC only; H100 params assumed unchanged.
            plan.append((self.TIER_A6000, cfg.async_save))
            logger.debug("TierSavePlanner: SLC_ONLY policy — H100/CPU omitted.")

        elif policy == SLCFlushPolicy.PARALLEL:
            # All three tiers concurrently (caller must run in thread pool).
            plan.append((self.TIER_H100, True))
            plan.append((self.TIER_A6000, True))
            plan.append((self.TIER_CPU, True))
            logger.debug("TierSavePlanner: PARALLEL policy — all tiers async.")

        else:  # AFTER_H100 (default)
            # H100 first (blocking or async-then-wait), then A6000 + CPU.
            h100_async = cfg.async_save and not cfg.h100_saves_first
            plan.append((self.TIER_H100, h100_async))
            # A6000 and CPU can overlap after H100 fsync.
            plan.append((self.TIER_A6000, cfg.async_save))
            plan.append((self.TIER_CPU, True))
            logger.debug(
                "TierSavePlanner: AFTER_H100 policy — H100 async=%s, "
                "A6000/CPU async=%s.",
                h100_async,
                cfg.async_save,
            )

        self._plan = plan
        return plan

    def describe_plan(self) -> str:
        """Human-readable plan summary for log output."""
        if not self._plan:
            self.build_plan()
        lines = ["DES-LOC checkpoint plan:"]
        for i, (tier, is_async) in enumerate(self._plan, start=1):
            lines.append(f"  step {i}: {tier} [{'async' if is_async else 'sync'}]")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CPU spill-arena async writer
# ---------------------------------------------------------------------------

class CPUSpillArenaWriter:
    """Background thread-pool writer for CPU DRAM optimizer state.

    Megatron's ``distrib_optim_fully_reshardable_mem_efficient`` uses Gloo +
    single-rank serialization to reduce peak GPU memory.  In DES-LOC, optimizer
    state that overflows H100/A6000 VRAM is already in CPU DRAM (the 1.5 TB
    spill arena), so the bottleneck is CPU→storage bandwidth, not NCCL.

    This writer partitions the CPU optimizer-state dict into shards and submits
    each shard to a thread pool.  Threads write to ``spill_arena_ckpt_dir``
    independently; a sentinel file is written only after all shards fsync.

    Thread safety: ``submit`` is called from the main training thread;
    ``wait`` is called before the next checkpoint or before program exit.
    """

    SENTINEL_FILENAME = "cpu_spill_complete.flag"

    def __init__(self, config: HeteroCheckpointConfig) -> None:
        self.config = config
        self._pool: Optional[list[threading.Thread]] = None
        self._errors: list[Exception] = []
        self._lock = threading.Lock()

    def submit(
        self,
        state_shards: Dict[str, Any],
        step: int,
        dest_dir: Path,
    ) -> None:
        """Submit optimizer-state shards for async serialization.

        Args:
            state_shards: Mapping of shard-name → tensor/state dict.
                          Each shard is serialized by one worker thread.
            step: Training step number (used in shard file names).
            dest_dir: Directory to write shards into.
        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        n_threads = self.config.cpu_async_worker_threads
        shard_items = list(state_shards.items())

        if self._pool is not None:
            logger.warning(
                "CPUSpillArenaWriter: previous async write has not been joined; "
                "calling wait() before submitting new shards."
            )
            self.wait()

        self._errors.clear()
        self._pool = []

        # Partition shards across worker threads.
        buckets: list[list] = [[] for _ in range(n_threads)]
        for idx, item in enumerate(shard_items):
            buckets[idx % n_threads].append(item)

        for bucket in buckets:
            if not bucket:
                continue
            t = threading.Thread(
                target=self._write_bucket,
                args=(bucket, step, dest_dir),
                daemon=True,
            )
            t.start()
            self._pool.append(t)

        logger.debug(
            "CPUSpillArenaWriter: submitted %d shards across %d threads for step %d.",
            len(shard_items),
            len(self._pool),
            step,
        )

    def _write_bucket(
        self,
        bucket: list[Tuple[str, Any]],
        step: int,
        dest_dir: Path,
    ) -> None:
        """Worker: serialize one bucket of shards to disk."""
        import torch  # deferred import to avoid top-level torch dep in config module

        for shard_name, shard_data in bucket:
            safe_name = shard_name.replace("/", "_").replace(":", "_")
            shard_path = dest_dir / f"step_{step:08d}_{safe_name}.pt"
            try:
                torch.save(shard_data, shard_path)
                logger.debug("CPUSpillArenaWriter: wrote shard %s", shard_path)
            except Exception as exc:
                with self._lock:
                    self._errors.append(exc)
                logger.error(
                    "CPUSpillArenaWriter: failed to write shard %s: %s",
                    shard_path,
                    exc,
                    exc_info=True,
                )

    def wait(self) -> None:
        """Block until all submitted shards have been written.

        Raises RuntimeError if any worker thread encountered an error.
        """
        if self._pool is None:
            return
        for t in self._pool:
            t.join()
        self._pool = None

        if self._errors:
            raise RuntimeError(
                f"CPUSpillArenaWriter: {len(self._errors)} shard write(s) failed. "
                f"First error: {self._errors[0]}"
            )
        logger.info("CPUSpillArenaWriter: all shards written successfully.")


# ---------------------------------------------------------------------------
# DeepSpeed argument-group factory shim
# ---------------------------------------------------------------------------

def add_hetero_checkpoint_args(parser: Any) -> Any:
    """Register DES-LOC checkpoint arguments with a DeepSpeed/argparse parser.

    This function mirrors Megatron's ``_add_checkpointing_args`` (which was
    refactored in d4f9347 to use ``ArgumentGroupFactory``).  We do not have
    ``ArgumentGroupFactory`` in the DeepSpeed codebase, so we register
    arguments manually while keeping the same logical grouping.

    DES-LOC-specific arguments are registered in a separate sub-group
    ``"deslock_checkpointing"`` to avoid polluting the base checkpoint group.

    Args:
        parser: An ``argparse.ArgumentParser`` or DeepSpeed's wrapper.

    Returns:
        The modified parser (same object).
    """
    group = parser.add_argument_group(title="checkpointing")

    group.add_argument("--save", type=str, default=None,
                       help="Output directory for H100 parameter + manifest checkpoints.")
    group.add_argument("--load", type=str, default=None,
                       help="Directory containing a DES-LOC checkpoint to resume from.")
    group.add_argument("--save-interval", "--persistent-save-interval",
                       type=int, default=None,
                       help="Iterations between full (all-tier) checkpoint saves.")
    group.add_argument("--save-wgrads-interval", type=int, default=None,
                       help="Iterations between wgrad (A6000 SLC grad-buffer) saves.")
    group.add_argument("--save-dgrads-interval", type=int, default=None,
                       help="Iterations between dgrad (A6000 SLC) saves.")
    group.add_argument("--save-retain-interval", type=int, default=None,
                       help="Iterations between retained (non-rotating) checkpoints.")
    group.add_argument("--ckpt-step", type=int, default=None,
                       help="Specific step to load from.")
    group.add_argument("--finetune", action="store_true",
                       help="Fine-tuning mode: load weights only, reset optimizer & iteration.")
    group.add_argument("--pretrained-checkpoint", type=str, default=None,
                       help="Pretrained checkpoint directory for fine-tuning.")
    group.add_argument("--no-save-optim", action="store_true", default=None,
                       help="Do not save optimizer state.")
    group.add_argument("--no-save-rng", action="store_true", default=None,
                       help="Do not save RNG state.")
    group.add_argument("--no-load-optim", action="store_true", default=None,
                       help="Do not load optimizer state on resume.")
    group.add_argument("--no-load-rng", action="store_true", default=None,
                       help="Do not load RNG state on resume.")
    group.add_argument("--use-checkpoint-args", action="store_true",
                       help="Override model arch args from checkpoint metadata.")
    group.add_argument("--exit-on-missing-checkpoint", action="store_true",
                       help="Exit if --load path has no checkpoint.")
    group.add_argument("--ckpt-format",
                       default="deslock_native",
                       choices=["torch", "torch_dist", "torch_dcp", "deslock_native"],
                       help="Checkpoint serialization format.")
    group.add_argument("--auto-detect-ckpt-format", action="store_true",
                       help="Auto-detect format of existing checkpoint on disk.")
    group.add_argument("--async-save", action="store_true", default=None,
                       help="Enable async checkpoint save (torch_dist or deslock_native only).")
    group.add_argument("--ckpt-fully-parallel-save", action="store_true",
                       help="Parallelize save across DP ranks.")
    group.add_argument("--ckpt-fully-parallel-load", action="store_true",
                       help="Parallelize load across DP ranks.")

    deslock = parser.add_argument_group(title="deslock_checkpointing")

    deslock.add_argument("--slc-flush-policy",
                         default="after_h100",
                         choices=[e.value for e in SLCFlushPolicy],
                         help="A6000 SLC flush ordering relative to H100 param save.")
    deslock.add_argument("--slc-save-dir", type=str, default=None,
                         help="Override directory for A6000 SLC checkpoint files.")
    deslock.add_argument("--no-slc-incremental-save", action="store_false",
                         dest="slc_incremental_save",
                         help="Disable dirty-page incremental SLC saves.")
    deslock.add_argument("--spill-arena-ckpt-dir", type=str, default=None,
                         help="Directory for CPU spill-arena optimizer state.")
    deslock.add_argument("--no-h100-saves-first", action="store_false",
                         dest="h100_saves_first",
                         help="Allow A6000 SLC flush before H100 param save completes.")
    deslock.add_argument("--sm86-ckpt-dtype",
                         default="bfloat16",
                         choices=["float32", "bfloat16", "float16"],
                         help="A6000 SLC checkpoint serialization dtype.")
    deslock.add_argument("--cpu-async-worker-threads", type=int, default=4,
                         help="Thread-pool size for CPU spill-arena serialization.")
    deslock.add_argument("--replication-tier",
                         default="none",
                         choices=[e.value for e in ReplicationTier],
                         help="Checkpoint replication tier(s) for DES-LOC.")
    deslock.add_argument("--h100-replica-on-cpu", action="store_true",
                         help="Shadow H100 param shard to CPU DRAM after each save.")
    deslock.add_argument("--a6000-replica-peer", action="store_true",
                         help="Replicate each A6000 SLC partition to the other A6000.")
    deslock.add_argument("--slc-hot-standby", action="store_true",
                         dest="slc_hot_standby_enabled",
                         help="Enable in-SLC hot-standby snapshots.")
    deslock.add_argument("--non-persistent-save-interval", type=int, default=None,
                         help="Iterations between hot-standby SLC snapshots.")

    return parser


def config_from_args(args: Any) -> HeteroCheckpointConfig:
    """Build a ``HeteroCheckpointConfig`` from parsed argparse namespace.

    Handles the ``--no-*`` negation pattern (Megatron convention) and maps
    flat namespace attributes to nested ``HeteroReplicationTopology``.

    Args:
        args: ``argparse.Namespace`` returned by ``parser.parse_args()``.

    Returns:
        Validated ``HeteroCheckpointConfig``.
    """
    def _get(name: str, default: Any = None) -> Any:
        return getattr(args, name, default)

    topo = HeteroReplicationTopology(
        tier=ReplicationTier(_get("replication_tier", "none")),
        h100_replica_on_cpu=bool(_get("h100_replica_on_cpu", False)),
        a6000_replica_peer=bool(_get("a6000_replica_peer", False)),
    )

    # Handle --no-save-optim → save_optim=False
    save_optim = not bool(_get("no_save_optim", False))
    save_rng = not bool(_get("no_save_rng", False))
    load_optim = not bool(_get("no_load_optim", False))
    load_rng = not bool(_get("no_load_rng", False))

    return HeteroCheckpointConfig(
        save=_get("save"),
        load=_get("load"),
        pretrained_checkpoint=_get("pretrained_checkpoint"),
        save_interval=_get("save_interval"),
        save_wgrads_interval=_get("save_wgrads_interval"),
        save_dgrads_interval=_get("save_dgrads_interval"),
        save_retain_interval=_get("save_retain_interval"),
        non_persistent_save_interval=_get("non_persistent_save_interval"),
        ckpt_step=_get("ckpt_step"),
        finetune=bool(_get("finetune", False)),
        save_optim=save_optim,
        save_rng=save_rng,
        load_optim=load_optim,
        load_rng=load_rng,
        use_checkpoint_args=bool(_get("use_checkpoint_args", False)),
        exit_on_missing_checkpoint=bool(_get("exit_on_missing_checkpoint", False)),
        ckpt_format=CkptFormat(_get("ckpt_format", "deslock_native")),
        auto_detect_ckpt_format=bool(_get("auto_detect_ckpt_format", False)),
        async_save=bool(_get("async_save", False)),
        ckpt_fully_parallel_save=bool(_get("ckpt_fully_parallel_save", True)),
        ckpt_fully_parallel_load=bool(_get("ckpt_fully_parallel_load", False)),
        slc_flush_policy=SLCFlushPolicy(_get("slc_flush_policy", "after_h100")),
        slc_save_dir=_get("slc_save_dir"),
        slc_incremental_save=bool(_get("slc_incremental_save", True)),
        spill_arena_ckpt_dir=_get("spill_arena_ckpt_dir"),
        h100_saves_first=bool(_get("h100_saves_first", True)),
        sm86_ckpt_dtype=_get("sm86_ckpt_dtype", "bfloat16"),
        cpu_async_worker_threads=int(_get("cpu_async_worker_threads", 4)),
        replication_topo=topo,
        slc_hot_standby_enabled=bool(_get("slc_hot_standby_enabled", False)),
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

    # 1. Default config constructs and validates without error.
    cfg = HeteroCheckpointConfig(save="/tmp/deslock_ckpt", save_interval=100)
    assert cfg.ckpt_format == CkptFormat.DESLOCK_NATIVE
    assert cfg.slc_flush_policy == SLCFlushPolicy.AFTER_H100
    assert cfg.resolved_slc_dir() == Path("/tmp/deslock_ckpt/slc")

    # 2. TierSavePlanner produces correct ordering for AFTER_H100 policy.
    planner = TierSavePlanner(cfg)
    plan = planner.build_plan()
    assert plan[0][0] == TierSavePlanner.TIER_H100, "H100 must be first in AFTER_H100 plan"
    assert plan[-1][0] == TierSavePlanner.TIER_CPU, "CPU must be last in AFTER_H100 plan"

    # 3. Serialization round-trip preserves all enum values.
    d = cfg.to_dict()
    cfg2 = HeteroCheckpointConfig.from_dict(d)
    assert cfg2.ckpt_format == cfg.ckpt_format
    assert cfg2.slc_flush_policy == cfg.slc_flush_policy
    assert cfg2.replication_topo.tier == cfg.replication_topo.tier

    # 4. slc_incremental_save=True with non-native format raises ValueError.
    try:
        HeteroCheckpointConfig(
            save="/tmp/x",
            ckpt_format=CkptFormat.TORCH_DIST,
            slc_incremental_save=True,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "deslock_native" in str(e)

    # 5. PARALLEL policy plan has all three tiers as async.
    cfg_par = HeteroCheckpointConfig(
        save="/tmp/y",
        slc_flush_policy=SLCFlushPolicy.PARALLEL,
        h100_saves_first=False,
        slc_incremental_save=False,
    )
    par_plan = TierSavePlanner(cfg_par).build_plan()
    assert all(is_async for _, is_async in par_plan), "All tiers must be async in PARALLEL plan"
    assert len(par_plan) == 3

    logger.info("All smoke tests passed.")
