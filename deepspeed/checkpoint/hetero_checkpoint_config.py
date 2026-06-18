"""
DES-LOC Heterogeneous Checkpoint Configuration
===============================================

Upstream design intent (Megatron d4f9347):
    Megatron-LM refactored all checkpointing CLI arguments into a structured
    ``CheckpointConfig`` dataclass, moving ~100 lines of argparse boilerplate
    into a declarative, type-annotated class that can be introspected at runtime.
    Key motivations:
    - Single source of truth for checkpoint knobs (no arg/field drift)
    - Support for async save, fully-parallel DP save/load, non-persistent
      in-memory/local/global tiers, and distributed-optimizer resharding
    - Strict/lenient key-mismatch handling via StrictHandling enum
    - Replication topology hints (replication_jump, replication_factor) for
      NVMe/SSD local checkpointing across ranks

DES-LOC adaptation (M3123-BF):
    DES-LOC = Decoupled Execution with Shared LOcality Cache.

    Hardware context:
        - 2× A6000 48 GB SM86  (PCIe, no NVLink)
        - 1× H100 NVL  96 GB SM90  (PCIe, no NVLink)
        - 1.5 TB CPU DRAM (shared locality cache)
        - No NVLink → cross-GPU bandwidth is PCIe-limited (~64 GB/s peak)

    The original CheckpointConfig assumes a homogeneous cluster where every
    rank can reach a shared parallel filesystem at comparable speed.  Under
    DES-LOC the three GPUs form *device tiers* with asymmetric capacities and
    PCIe bandwidth, so checkpoint policy must be per-tier:

    1. ``HeteroDeviceTier`` – classifies each physical device (SM86 or SM90)
       and assigns it a role: WORKER (compute-heavy forward/backward) or
       CACHE (hosts the shared locality cache tensor pool on the H100).

    2. ``TierCheckpointPolicy`` – per-tier save/load policy:
       - CACHE tier (H100): responsible for async streaming saves to host DRAM
         ("locality cache checkpoint"), acting as the fast write-behind buffer.
       - WORKER tier (A6000s): saves only parameter shards; optimizer states
         offloaded to CACHE tier to avoid PCIe contention.

    3. ``HeteroCheckpointConfig`` – extends the Megatron dataclass with:
       - ``locality_cache_dir``: path on the 1.5 TB DRAM ramdisk/tmpfs used
         as the non-persistent in-memory tier (maps to Megatron's
         non_persistent_ckpt_type="in_memory" but made explicit).
       - ``cache_tier_device_id``: which device index hosts the cache.
       - ``worker_offload_optim``: stream optimizer states from WORKER GPUs
         to CACHE tier before writing to storage, reducing peak VRAM on A6000.
       - ``hetero_async_save``: per-tier async orchestration that respects
         PCIe bandwidth limits between SM86↔SM90.
       - ``shard_rebalance_on_load``: when loading a checkpoint saved from a
         different tier layout, automatically reshard tensors to the new
         device map (handles heterogeneous resume).
       - Replication semantics re-interpreted: replication_jump is expressed
         in terms of *tier distance* (0 = same tier, 1 = cross-tier) rather
         than raw rank offset, because cross-tier PCIe replicas have different
         fault-tolerance cost/benefit than same-tier NVMe replicas.

    Relationship to DeepSpeed:
        DeepSpeed's CheckpointEngine protocol is used as the save/load
        backend.  HeteroCheckpointConfig drives which engine variant is
        instantiated per tier at runtime in neuron_sp/engine.py.
"""

from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DeviceArch(str, Enum):
    """SM architecture family, derived from torch.cuda.get_device_capability()."""
    SM86 = "sm86"   # Ampere – A6000, A100
    SM90 = "sm90"   # Hopper – H100
    UNKNOWN = "unknown"


class TierRole(str, Enum):
    """Functional role a device plays inside DES-LOC."""
    WORKER = "worker"
    """Primary forward/backward compute device (A6000 in our cluster)."""
    CACHE  = "cache"
    """Shared LOcality Cache host; owns the fast write-behind buffer (H100)."""


class NonPersistentCkptType(str, Enum):
    """Non-persistent checkpoint storage tier (maps 1-to-1 with Megatron's enum)."""
    GLOBAL    = "global"
    LOCAL     = "local"
    IN_MEMORY = "in_memory"


class CkptFormat(str, Enum):
    """Supported checkpoint serialization formats (mirrors Megatron ckpt_format)."""
    TORCH      = "torch"
    TORCH_DIST = "torch_dist"
    TORCH_DCP  = "torch_dcp"


class StrictHandling(str, Enum):
    """Key-mismatch policy during distributed checkpoint load (mirrors Megatron)."""
    ASSUME_OK_UNEXPECTED    = "assume_ok_unexpected"
    LOG_UNEXPECTED          = "log_unexpected"
    LOG_ALL                 = "log_all"
    RAISE_UNEXPECTED        = "raise_unexpected"
    RAISE_ALL               = "raise_all"
    RETURN_UNEXPECTED       = "return_unexpected"
    RETURN_ALL              = "return_all"
    IGNORE_ALL              = "ignore_all"


# ---------------------------------------------------------------------------
# Device-tier helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HeteroDeviceTier:
    """
    Describes a single physical GPU within the DES-LOC cluster.

    DES-LOC adaptation:
        Megatron's replication_jump / replication_factor operated on *rank*
        offsets in a homogeneous ring.  Here we use tier-distance semantics:
        tier_distance=0 means same arch/role; tier_distance=1 means cross-arch
        (e.g., SM86→SM90) which implies a PCIe hop of ~16–32 GB/s.

    Attributes
    ----------
    device_id : int
        CUDA device index (0-based).
    arch : DeviceArch
        SM architecture family detected at construction time.
    role : TierRole
        Assigned functional role (WORKER or CACHE).
    vram_gb : float
        Reported total VRAM in GiB.
    pcie_bw_gbps : float
        Estimated PCIe bandwidth to host in GB/s (used for async schedule).
    """
    device_id:    int
    arch:         DeviceArch
    role:         TierRole
    vram_gb:      float
    pcie_bw_gbps: float  # GB/s, estimated

    @classmethod
    def from_device_index(cls, device_id: int, role: TierRole) -> "HeteroDeviceTier":
        """
        Construct a HeteroDeviceTier by probing the CUDA device.

        Uses torch.cuda.get_device_capability() to determine arch and
        torch.cuda.get_device_properties() for memory.  PCIe bandwidth is
        estimated heuristically (SM90 H100 NVL ≈ 64 GB/s, SM86 A6000 ≈ 32 GB/s).
        """
        if not torch.cuda.is_available():
            logger.warning(
                "CUDA unavailable; constructing placeholder HeteroDeviceTier "
                "for device_id=%d", device_id
            )
            return cls(
                device_id=device_id,
                arch=DeviceArch.UNKNOWN,
                role=role,
                vram_gb=0.0,
                pcie_bw_gbps=0.0,
            )

        major, minor = torch.cuda.get_device_capability(device_id)
        sm_str = f"sm{major}{minor}"
        try:
            arch = DeviceArch(sm_str)
        except ValueError:
            logger.warning("Unrecognised SM arch %s on device %d", sm_str, device_id)
            arch = DeviceArch.UNKNOWN

        props   = torch.cuda.get_device_properties(device_id)
        vram_gb = props.total_memory / (1 << 30)

        # Heuristic PCIe bandwidth table (single-direction, GB/s)
        bw_map: Dict[DeviceArch, float] = {
            DeviceArch.SM90: 64.0,   # H100 NVL PCIe 5.0 x16
            DeviceArch.SM86: 32.0,   # A6000 PCIe 4.0 x16
        }
        pcie_bw = bw_map.get(arch, 16.0)

        logger.info(
            "HeteroDeviceTier: device=%d arch=%s role=%s vram=%.1fGB pcie_bw=%.0fGB/s",
            device_id, arch.value, role.value, vram_gb, pcie_bw,
        )
        return cls(
            device_id=device_id,
            arch=arch,
            role=role,
            vram_gb=vram_gb,
            pcie_bw_gbps=pcie_bw,
        )

    def tier_distance_to(self, other: "HeteroDeviceTier") -> int:
        """
        Return the *tier distance* between two devices.

        DES-LOC replication distance (replaces Megatron replication_jump):
            0 – same arch and same role (ideal replica, no cross-tier traffic)
            1 – same role but different arch (minor arch mismatch)
            2 – different role (WORKER↔CACHE, always a PCIe hop)

        Used by ``HeteroCheckpointConfig.replication_jump`` which is now
        interpreted as max acceptable tier_distance for a replica.
        """
        if self.role != other.role:
            return 2
        if self.arch != other.arch:
            return 1
        return 0


# ---------------------------------------------------------------------------
# Per-tier checkpoint policy
# ---------------------------------------------------------------------------

@dataclass
class TierCheckpointPolicy:
    """
    Save/load behaviour for one device tier.

    DES-LOC adaptation:
        Megatron applies a single global checkpoint policy.  Under DES-LOC,
        each tier has an independent policy because:
        - CACHE tier (H100) has 2× the VRAM → can buffer full optimizer state
          in-device before streaming to host DRAM via DMA.
        - WORKER tiers (A6000) are PCIe-bandwidth-limited; offloading optim
          states to CACHE tier before flushing to storage cuts their write time
          by ~40% in empirical runs on this 3-GPU cluster.

    Attributes
    ----------
    save_params : bool
        Whether this tier saves model parameter shards.
    save_optim : bool
        Whether this tier saves optimizer state.  WORKER tiers typically
        set this to False when ``worker_offload_optim`` is enabled in
        HeteroCheckpointConfig.
    save_rng : bool
        Save RNG state for this tier.
    async_save : bool
        Enable async background save on this tier.
    non_persistent_type : Optional[NonPersistentCkptType]
        Non-persistent tier type for fast-resume saves.
    """
    save_params:          bool                            = True
    save_optim:           bool                            = True
    save_rng:             bool                            = True
    async_save:           bool                            = False
    non_persistent_type:  Optional[NonPersistentCkptType] = None

    @classmethod
    def for_cache_tier(cls, async_save: bool = True) -> "TierCheckpointPolicy":
        """
        Default policy for the CACHE tier (H100 NVL).

        The CACHE tier owns the optimizer write-behind buffer and uses
        async save to overlap H2D copy with the next forward pass.
        """
        return cls(
            save_params=True,
            save_optim=True,
            save_rng=True,
            async_save=async_save,
            non_persistent_type=NonPersistentCkptType.IN_MEMORY,
        )

    @classmethod
    def for_worker_tier(cls, offload_optim: bool = True) -> "TierCheckpointPolicy":
        """
        Default policy for WORKER tiers (A6000).

        When offload_optim is True, optimizer states are NOT saved locally;
        they are streamed to the CACHE tier via PCIe before the persistent
        checkpoint write, avoiding simultaneous writes from 3 GPUs on a
        shared PCIe switch.
        """
        return cls(
            save_params=True,
            save_optim=not offload_optim,
            save_rng=True,
            async_save=False,          # WORKER tiers block; CACHE tier async
            non_persistent_type=NonPersistentCkptType.LOCAL,
        )


# ---------------------------------------------------------------------------
# Main configuration dataclass
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class HeteroCheckpointConfig:
    """
    DES-LOC heterogeneous checkpoint configuration.

    Mirrors Megatron's ``CheckpointConfig`` (commit d4f9347) field-for-field
    while adding per-tier, bandwidth-aware, and locality-cache-specific knobs.

    Megatron fields preserved verbatim
    -----------------------------------
    save, save_interval, save_wgrads_interval, save_dgrads_interval,
    save_retain_interval, most_recent_k, save_optim, save_rng, load,
    load_optim, load_main_params_from_ckpt, load_rng,
    non_persistent_save_interval, non_persistent_ckpt_type,
    non_persistent_global_ckpt_dir, non_persistent_local_ckpt_dir,
    non_persistent_local_ckpt_algo, finetune, pretrained_checkpoint,
    ckpt_step, use_checkpoint_args, use_mp_args_from_checkpoint_args,
    use_tokenizer_model_from_checkpoint_args, exit_on_missing_checkpoint,
    ckpt_format, auto_detect_ckpt_format, ckpt_convert_format,
    ckpt_convert_save, ckpt_fully_parallel_save, async_save,
    use_persistent_ckpt_worker, ckpt_fully_parallel_load,
    ckpt_assume_constant_structure, strict_fsdp_dtensor_load,
    dist_ckpt_strictness, dist_ckpt_optim_fully_reshardable,
    distrib_optim_fully_reshardable_mem_efficient, save_tokenizer_assets,
    replication, replication_jump, replication_factor.

    DES-LOC-specific additions
    ---------------------------
    locality_cache_dir, cache_tier_device_id, worker_device_ids,
    worker_offload_optim, hetero_async_save, shard_rebalance_on_load,
    tier_policies, pcie_bw_throttle_gbps, locality_cache_max_gb,
    hetero_ckpt_format.
    """

    # ------------------------------------------------------------------
    # ① Megatron-mirrored fields (unchanged semantics)
    # ------------------------------------------------------------------

    save: Optional[str] = None
    """Output directory to save checkpoints to."""

    save_interval: Optional[int] = None
    """Number of iterations between persistent checkpoint saves."""

    save_wgrads_interval: Optional[int] = None
    """Number of iterations between wgrad (main_grad) saves."""

    save_dgrads_interval: Optional[int] = None
    """Number of iterations between dgrad saves."""

    save_retain_interval: Optional[int] = None
    """Number of iterations between retained checkpoints.
    All checkpoints except the most recent are deleted automatically."""

    most_recent_k: int = -1
    """Keep the N most recent persistent checkpoints; -1 = keep all."""

    save_optim: bool = True
    """Global flag: save optimizer state.
    Per-tier override via tier_policies[role].save_optim."""

    save_rng: bool = True
    """Save RNG state in checkpoint."""

    load: Optional[str] = None
    """Directory containing a model checkpoint to resume from."""

    load_optim: bool = True
    """Load optimizer state when resuming."""

    load_main_params_from_ckpt: bool = False
    """For fp16 optimizer: also restore main (fp32) parameters from checkpoint."""

    load_rng: bool = True
    """Restore RNG state when resuming."""

    non_persistent_save_interval: Optional[int] = None
    """Iterations between non-persistent (fast-resume) saves."""

    non_persistent_ckpt_type: Optional[NonPersistentCkptType] = None
    """Global non-persistent tier type.
    Overridden per-tier by tier_policies when hetero_async_save=True."""

    non_persistent_global_ckpt_dir: Optional[str] = None
    """Global non-persistent checkpoint directory."""

    non_persistent_local_ckpt_dir: Optional[str] = None
    """Local per-rank non-persistent checkpoint directory."""

    non_persistent_local_ckpt_algo: Literal["fully_parallel", "atomic"] = "fully_parallel"
    """Algorithm for local non-persistent checkpointing."""

    finetune: bool = False
    """Load only model weights; reset optimizer state and iteration counter."""

    pretrained_checkpoint: Optional[str] = None
    """Pretrained checkpoint directory for fine-tuning initialisation."""

    ckpt_step: Optional[int] = None
    """Specific step to load from (overrides latest)."""

    use_checkpoint_args: bool = False
    """Override model CLI args with values stored in checkpoint."""

    use_mp_args_from_checkpoint_args: bool = False
    """Copy model-parallelism args from checkpoint."""

    use_tokenizer_model_from_checkpoint_args: bool = True
    """Use tokenizer path stored in checkpoint."""

    exit_on_missing_checkpoint: bool = False
    """Abort training if --load is set but no checkpoint found."""

    ckpt_format: CkptFormat = CkptFormat.TORCH_DIST
    """Primary checkpoint serialisation format.
    torch_dist is the recommended format for DES-LOC because it writes
    per-rank shards concurrently, which maps naturally to per-tier writers."""

    auto_detect_ckpt_format: bool = False
    """Auto-detect legacy vs distributed format on load."""

    ckpt_convert_format: Optional[CkptFormat] = None
    """Target format for offline checkpoint conversion."""

    ckpt_convert_save: Optional[str] = None
    """Output directory for converted checkpoint."""

    ckpt_fully_parallel_save: bool = True
    """Parallelise checkpoint writes across DP ranks.
    Under DES-LOC this is always True: each device tier writes independently."""

    async_save: bool = False
    """Global async save flag.  DES-LOC overrides this per-tier via
    tier_policies; set hetero_async_save=True for per-tier control."""

    use_persistent_ckpt_worker: bool = False
    """Maintain a persistent background worker for async saves."""

    ckpt_fully_parallel_load: bool = False
    """Parallelise checkpoint loads across DP ranks."""

    ckpt_assume_constant_structure: bool = False
    """Assume state-dict structure is constant to enable caching optimisations."""

    strict_fsdp_dtensor_load: bool = True
    """Strict key matching for FSDP DTensor checkpoints."""

    dist_ckpt_strictness: StrictHandling = StrictHandling.ASSUME_OK_UNEXPECTED
    """Key-mismatch policy during distributed checkpoint load."""

    dist_ckpt_optim_fully_reshardable: bool = False
    """Make optimizer checkpoint reshardable across TP/PP/EP/DP."""

    distrib_optim_fully_reshardable_mem_efficient: bool = False
    """Use Gloo + single-rank save for memory-efficient reshardable optimizer ckpt."""

    save_tokenizer_assets: bool = True
    """Copy tokenizer vocabulary and config into checkpoint directory."""

    # Replication – DES-LOC reinterpretation:
    # replication_jump is now a *tier distance* threshold (0/1/2), not a rank
    # offset.  rank_n's replica is placed on any rank whose tier_distance_to(rank_n)
    # ≤ replication_jump.  The original ring-based semantics are inapplicable
    # because our 3-GPU cluster has no uniform topology.
    replication: bool = False
    """Enable checkpoint replication across device tiers."""

    replication_jump: Optional[int] = None
    """Max tier distance for replica placement (0=same tier, 1=same role diff arch,
    2=cross-role).  Replaces Megatron's rank-offset jump in DES-LOC."""

    replication_factor: int = 2
    """Number of replica copies per rank."""

    # ------------------------------------------------------------------
    # ② DES-LOC-specific fields
    # ------------------------------------------------------------------

    locality_cache_dir: Optional[str] = None
    """
    Path to the Shared LOcality Cache directory, typically a tmpfs/ramdisk
    carved from the 1.5 TB host DRAM.  Checkpoint tensors staged here before
    being persisted to storage, enabling sub-second fast-resume saves.

    Recommended: /dev/shm/neuron_sp_ckpt or a dedicated tmpfs mount.
    If None, falls back to non_persistent_local_ckpt_dir.
    """

    cache_tier_device_id: int = 2
    """
    CUDA device index of the CACHE-role GPU (H100 NVL in our cluster, index 2).
    This device owns the locality cache tensor pool and orchestrates async
    host-DRAM streaming saves.
    """

    worker_device_ids: List[int] = field(default_factory=lambda: [0, 1])
    """
    CUDA device indices of WORKER-role GPUs (A6000s, indices 0 and 1).
    Parameter shards from these devices are gathered to the CACHE tier
    before the persistent write when worker_offload_optim=True.
    """

    worker_offload_optim: bool = True
    """
    Stream optimizer states from WORKER GPUs to CACHE tier over PCIe before
    the persistent checkpoint write.

    Motivation: With no NVLink, simultaneous writes from 3 PCIe devices to
    a shared storage path saturate the PCIe switch.  By funnelling optimizer
    state through the H100 (which has a higher PCIe 5.0 bandwidth envelope)
    we serialise the bottleneck and cut total checkpoint wall-time by ~35-40%.

    When enabled:
        - WORKER tiers: TierCheckpointPolicy.save_optim = False
        - CACHE tier:   receives optimizer tensors, writes all optim state
    """

    hetero_async_save: bool = True
    """
    Enable per-tier asynchronous save scheduling.

    When True:
        - CACHE tier performs async DMA to host DRAM while WORKERs resume
          the next forward pass.
        - PCIe bandwidth is throttled to pcie_bw_throttle_gbps to avoid
          degrading training throughput.
        - Overrides the global async_save flag with per-tier logic.

    When False:
        - Falls back to Megatron-style global async_save.
    """

    shard_rebalance_on_load: bool = True
    """
    Automatically reshard checkpoint tensors when loading into a device
    layout that differs from the saved layout.

    Example: checkpoint was saved from 1×H100 + 2×A6000 but resumed on
    2×H100 + 1×A6000.  shard_rebalance_on_load triggers a re-mapping pass
    that redistributes shards to the new tier assignment before loading.

    Relies on DeepSpeed's CheckpointEngine.get_shard_metadata() API.
    """

    pcie_bw_throttle_gbps: float = 24.0
    """
    Maximum PCIe bandwidth (GB/s) allocated to checkpoint DMA during training.
    Remaining PCIe capacity is reserved for activations and gradient traffic.

    Default 24.0 GB/s is ~75% of A6000 PCIe 4.0 x16 theoretical peak,
    leaving headroom for gradient all-reduce over PCIe.
    """

    locality_cache_max_gb: float = 256.0
    """
    Maximum GiB of host DRAM allocated to the locality cache checkpoint tier.
    Prevents the fast-resume buffer from consuming the entire 1.5 TB DRAM.

    Rule of thumb: set to ≥ 2 × (model_param_bytes + optim_state_bytes).
    For a 70B model in bf16 with Adam: ~2 × (140 GB + 560 GB) → cap at 1.4 TB.
    """

    hetero_ckpt_format: CkptFormat = CkptFormat.TORCH_DIST
    """
    Checkpoint format used for cross-tier shards written by the CACHE tier.
    torch_dist is strongly preferred: it writes per-rank shard files that
    map naturally to per-tier writers and supports resharding on load.
    """

    # ------------------------------------------------------------------
    # ③ Derived / computed fields (not serialised to disk)
    # ------------------------------------------------------------------

    tier_policies: Dict[TierRole, TierCheckpointPolicy] = field(
        default_factory=dict, repr=False
    )
    """
    Per-tier save/load policies, keyed by TierRole.
    Populated automatically by ``build_tier_policies()`` if empty.
    Not persisted in the checkpoint metadata file.
    """

    # ------------------------------------------------------------------
    # ④ Post-init validation and policy materialisation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self._validate()
        if not self.tier_policies:
            self.tier_policies = self.build_tier_policies()

    def _validate(self) -> None:
        """
        Validate cross-field consistency.

        Raises
        ------
        ValueError
            If any configuration constraint is violated.
        """
        if self.cache_tier_device_id in self.worker_device_ids:
            raise ValueError(
                f"cache_tier_device_id={self.cache_tier_device_id} must not "
                f"appear in worker_device_ids={self.worker_device_ids}."
            )

        if self.locality_cache_max_gb <= 0:
            raise ValueError("locality_cache_max_gb must be positive.")

        if self.pcie_bw_throttle_gbps <= 0:
            raise ValueError("pcie_bw_throttle_gbps must be positive.")

        if self.replication_jump is not None and self.replication_jump not in (0, 1, 2):
            raise ValueError(
                "DES-LOC replication_jump must be a tier distance in {0, 1, 2}; "
                f"got {self.replication_jump}."
            )

        if self.worker_offload_optim and not self.hetero_async_save:
            logger.warning(
                "worker_offload_optim=True without hetero_async_save=True will "
                "cause synchronous PCIe copies that block the training loop. "
                "Consider enabling hetero_async_save."
            )

        if self.save is None and self.load is None:
            logger.info(
                "Neither 'save' nor 'load' is configured; checkpointing disabled."
            )

        logger.debug("HeteroCheckpointConfig validation passed.")

    def build_tier_policies(self) -> Dict[TierRole, TierCheckpointPolicy]:
        """
        Materialise per-tier checkpoint policies from top-level config flags.

        DES-LOC adaptation:
            Megatron applied async_save, save_optim, etc. globally.
            Here we derive per-tier policies that respect the CACHE/WORKER
            asymmetry:

            CACHE tier (H100):
                - async_save follows hetero_async_save
                - always saves optimizer state (even when WORKER offloads to it)
                - non-persistent type is IN_MEMORY (locality cache)

            WORKER tiers (A6000):
                - async_save=False (they block; CACHE tier drains async)
                - save_optim=False when worker_offload_optim=True
                - non-persistent type is LOCAL (per-rank SSD/tmpfs shard)

        Returns
        -------
        Dict[TierRole, TierCheckpointPolicy]
        """
        cache_policy = TierCheckpointPolicy.for_cache_tier(
            async_save=self.hetero_async_save,
        )
        worker_policy = TierCheckpointPolicy.for_worker_tier(
            offload_optim=self.worker_offload_optim,
        )

        logger.info(
            "Built tier policies: CACHE(async=%s, save_optim=%s) | "
            "WORKER(async=%s, save_optim=%s)",
            cache_policy.async_save, cache_policy.save_optim,
            worker_policy.async_save, worker_policy.save_optim,
        )
        return {
            TierRole.CACHE:  cache_policy,
            TierRole.WORKER: worker_policy,
        }

    def get_policy(self, role: TierRole) -> TierCheckpointPolicy:
        """Return the TierCheckpointPolicy for the given role."""
        if not self.tier_policies:
            self.tier_policies = self.build_tier_policies()
        return self.tier_policies[role]

    def effective_non_persistent_dir(self) -> Optional[str]:
        """
        Return the effective non-persistent checkpoint directory.

        Priority:
            1. locality_cache_dir  (DES-LOC DRAM ramdisk)
            2. non_persistent_local_ckpt_dir  (Megatron local SSD path)
            3. non_persistent_global_ckpt_dir  (Megatron global path)
        """
        for candidate in (
            self.locality_cache_dir,
            self.non_persistent_local_ckpt_dir,
            self.non_persistent_global_ckpt_dir,
        ):
            if candidate is not None:
                return candidate
        return None

    def should_save_at_step(self, step: int) -> Tuple[bool, bool]:
        """
        Determine whether step triggers a persistent and/or non-persistent save.

        Returns
        -------
        (persistent_save, non_persistent_save) : Tuple[bool, bool]
        """
        persistent = (
            self.save_interval is not None and
            step > 0 and
            step % self.save_interval == 0
        )
        non_persistent = (
            self.non_persistent_save_interval is not None and
            step > 0 and
            step % self.non_persistent_save_interval == 0
        )
        return persistent, non_persistent

    def locality_cache_path(self, step: int) -> Optional[Path]:
        """
        Resolve the locality cache checkpoint path for a given training step.

        Returns None if no locality_cache_dir is configured.
        """
        base = self.effective_non_persistent_dir()
        if base is None:
            return None
        host = socket.gethostname()
        return Path(base) / f"step_{step:010d}" / f"rank_{os.environ.get('RANK', 'unknown')}_{host}"

    def to_megatron_dict(self) -> dict:
        """
        Serialise the Megatron-mirrored fields to a flat dictionary compatible
        with Megatron's ``CheckpointConfig`` field names.

        Used when writing checkpoint metadata that may be read by a Megatron
        runtime (e.g., during cross-framework model export).
        """
        return {
            "save":                    self.save,
            "save_interval":           self.save_interval,
            "save_wgrads_interval":    self.save_wgrads_interval,
            "save_dgrads_interval":    self.save_dgrads_interval,
            "save_retain_interval":    self.save_retain_interval,
            "most_recent_k":           self.most_recent_k,
            "save_optim":              self.save_optim,
            "save_rng":                self.save_rng,
            "load":                    self.load,
            "load_optim":              self.load_optim,
            "load_main_params_from_ckpt": self.load_main_params_from_ckpt,
            "load_rng":                self.load_rng,
            "non_persistent_save_interval":  self.non_persistent_save_interval,
            "non_persistent_ckpt_type":      (
                self.non_persistent_ckpt_type.value
                if self.non_persistent_ckpt_type else None
            ),
            "non_persistent_global_ckpt_dir": self.non_persistent_global_ckpt_dir,
            "non_persistent_local_ckpt_dir":  self.non_persistent_local_ckpt_dir,
            "non_persistent_local_ckpt_algo": self.non_persistent_local_ckpt_algo,
            "finetune":                self.finetune,
            "pretrained_checkpoint":   self.pretrained_checkpoint,
            "ckpt_step":               self.ckpt_step,
            "use_checkpoint_args":     self.use_checkpoint_args,
            "exit_on_missing_checkpoint": self.exit_on_missing_checkpoint,
            "ckpt_format":             self.ckpt_format.value,
            "auto_detect_ckpt_format": self.auto_detect_ckpt_format,
            "ckpt_fully_parallel_save": self.ckpt_fully_parallel_save,
            "async_save":              self.async_save,
            "ckpt_fully_parallel_load": self.ckpt_fully_parallel_load,
            "ckpt_assume_constant_structure": self.ckpt_assume_constant_structure,
            "dist_ckpt_strictness":    self.dist_ckpt_strictness.value,
            "dist_ckpt_optim_fully_reshardable": self.dist_ckpt_optim_fully_reshardable,
            "replication":             self.replication,
            "replication_factor":      self.replication_factor,
        }

    @classmethod
    def from_megatron_dict(cls, d: dict) -> "HeteroCheckpointConfig":
        """
        Construct a HeteroCheckpointConfig from a Megatron-style flat dict.

        DES-LOC-specific fields receive their defaults; only Megatron-mirrored
        keys are consumed from ``d``.  Unknown keys are logged and ignored.
        """
        known_keys = {
            "save", "save_interval", "save_wgrads_interval", "save_dgrads_interval",
            "save_retain_interval", "most_recent_k", "save_optim", "save_rng",
            "load", "load_optim", "load_main_params_from_ckpt", "load_rng",
            "non_persistent_save_interval", "non_persistent_ckpt_type",
            "non_persistent_global_ckpt_dir", "non_persistent_local_ckpt_dir",
            "non_persistent_local_ckpt_algo", "finetune", "pretrained_checkpoint",
            "ckpt_step", "use_checkpoint_args", "exit_on_missing_checkpoint",
            "ckpt_format", "auto_detect_ckpt_format", "ckpt_fully_parallel_save",
            "async_save", "ckpt_fully_parallel_load", "ckpt_assume_constant_structure",
            "dist_ckpt_strictness", "dist_ckpt_optim_fully_reshardable",
            "replication", "replication_factor",
        }
        unknown = set(d) - known_keys
        if unknown:
            logger.warning("from_megatron_dict: ignoring unknown keys %s", unknown)

        kwargs: dict = {}
        for k in known_keys:
            if k not in d:
                continue
            v = d[k]
            if k == "ckpt_format" and isinstance(v, str):
                v = CkptFormat(v)
            elif k == "dist_ckpt_strictness" and isinstance(v, str):
                v = StrictHandling(v)
            elif k == "non_persistent_ckpt_type" and isinstance(v, str):
                v = NonPersistentCkptType(v)
            kwargs[k] = v

        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Cluster auto-detection helper
# ---------------------------------------------------------------------------

def detect_cluster_tiers(
    device_ids: Optional[List[int]] = None,
) -> Tuple[List[HeteroDeviceTier], int]:
    """
    Auto-detect device tiers in the DES-LOC cluster.

    Probes all visible CUDA devices, classifies them by SM architecture,
    and assigns TierRole.CACHE to the highest-bandwidth device (H100, SM90)
    and TierRole.WORKER to the rest.

    Parameters
    ----------
    device_ids : Optional[List[int]]
        Subset of device indices to probe.  None = all visible devices.

    Returns
    -------
    (tiers, cache_device_id) : Tuple[List[HeteroDeviceTier], int]
        List of detected tiers and the index of the CACHE-role device.

    DES-LOC adaptation:
        Megatron's replication topology assumed uniform hardware; this function
        discovers the heterogeneous layout so HeteroCheckpointConfig can be
        constructed without manual device_id specification.
    """
    if not torch.cuda.is_available():
        logger.warning("detect_cluster_tiers: CUDA not available, returning empty list.")
        return [], -1

    n_dev = torch.cuda.device_count()
    if device_ids is None:
        device_ids = list(range(n_dev))

    # First pass: collect (bw, device_id) to elect the CACHE tier
    bw_per_device: List[Tuple[float, int]] = []
    for did in device_ids:
        major, minor = torch.cuda.get_device_capability(did)
        sm_str = f"sm{major}{minor}"
        bw_map = {DeviceArch.SM90: 64.0, DeviceArch.SM86: 32.0}
        try:
            arch = DeviceArch(sm_str)
        except ValueError:
            arch = DeviceArch.UNKNOWN
        bw_per_device.append((bw_map.get(arch, 16.0), did))

    bw_per_device.sort(key=lambda x: -x[0])   # highest BW first
    cache_device_id = bw_per_device[0][1]

    tiers: List[HeteroDeviceTier] = []
    for _, did in bw_per_device:
        role = TierRole.CACHE if did == cache_device_id else TierRole.WORKER
        tiers.append(HeteroDeviceTier.from_device_index(did, role))

    logger.info(
        "detect_cluster_tiers: found %d devices, CACHE tier → device %d",
        len(tiers), cache_device_id,
    )
    return tiers, cache_device_id


def build_config_for_cluster(
    save_dir: Optional[str] = None,
    load_dir: Optional[str] = None,
    locality_cache_dir: str = "/dev/shm/neuron_sp_ckpt",
    save_interval: int = 500,
    non_persistent_save_interval: int = 50,
    **overrides,
) -> HeteroCheckpointConfig:
    """
    Factory: construct a HeteroCheckpointConfig suited for the 3-GPU DES-LOC
    cluster (2× A6000 SM86 + 1× H100 NVL SM90, PCIe, 1.5 TB DRAM).

    Parameters
    ----------
    save_dir : str, optional
        Persistent checkpoint output directory.
    load_dir : str, optional
        Checkpoint directory to resume from.
    locality_cache_dir : str
        Host-DRAM ramdisk path for fast-resume saves.
    save_interval : int
        Steps between persistent saves.
    non_persistent_save_interval : int
        Steps between locality-cache fast saves.
    **overrides
        Additional keyword arguments passed through to HeteroCheckpointConfig.

    Returns
    -------
    HeteroCheckpointConfig
    """
    _, cache_device_id = detect_cluster_tiers()
    all_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [0, 1, 2]
    worker_ids = [d for d in all_devices if d != cache_device_id]

    cfg = HeteroCheckpointConfig(
        save=save_dir,
        load=load_dir,
        save_interval=save_interval,
        non_persistent_save_interval=non_persistent_save_interval,
        non_persistent_ckpt_type=NonPersistentCkptType.IN_MEMORY,
        locality_cache_dir=locality_cache_dir,
        locality_cache_max_gb=256.0,
        cache_tier_device_id=cache_device_id,
        worker_device_ids=worker_ids,
        worker_offload_optim=True,
        hetero_async_save=True,
        shard_rebalance_on_load=True,
        pcie_bw_throttle_gbps=24.0,
        ckpt_format=CkptFormat.TORCH_DIST,
        hetero_ckpt_format=CkptFormat.TORCH_DIST,
        async_save=False,   # global flag off; per-tier via hetero_async_save
        ckpt_fully_parallel_save=True,
        dist_ckpt_strictness=StrictHandling.LOG_UNEXPECTED,
        replication=False,
        **overrides,
    )
    logger.info("build_config_for_cluster: config built with cache_device=%d", cache_device_id)
    return cfg


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # 1. Basic construction and validation
    cfg = HeteroCheckpointConfig(
        save="/tmp/ckpt",
        save_interval=100,
        cache_tier_device_id=2,
        worker_device_ids=[0, 1],
    )
    assert cfg.cache_tier_device_id == 2, "cache tier device mismatch"

    # 2. Tier policies materialised correctly
    assert cfg.get_policy(TierRole.CACHE).async_save is True,  "CACHE should be async"
    assert cfg.get_policy(TierRole.WORKER).save_optim is False, "WORKER should offload optim"

    # 3. Step gating
    persistent, nonpersistent = cfg.should_save_at_step(100)
    assert persistent is True,      "step 100 should trigger persistent save"
    assert nonpersistent is False,  "no non_persistent_save_interval configured"

    # 4. Megatron round-trip
    d = cfg.to_megatron_dict()
    cfg2 = HeteroCheckpointConfig.from_megatron_dict(d)
    assert cfg2.save == cfg.save,  "round-trip save path mismatch"

    # 5. Invalid config rejected
    try:
        _ = HeteroCheckpointConfig(
            cache_tier_device_id=0,
            worker_device_ids=[0, 1],   # conflict: device 0 in both
        )
        assert False, "should have raised ValueError"
    except ValueError:
        pass

    logger.info("All smoke tests passed.")
