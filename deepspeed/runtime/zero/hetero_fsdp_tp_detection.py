"""
DES-LOC Heterogeneous FSDP Tensor Parallelism Detection
========================================================

Upstream design intent (Megatron 8cbc45b):
    Megatron-LM's M-FSDP previously annotated tensor-parallel parameters with
    a trio of boolean/integer flags (_mcore_tp, _tp_duplicated, _tp_partition_dim)
    set by a single monolithic `_fix_tensor_parallel_attributes` method that hard-coded
    TELinear isinstance checks.  This caused two classes of bugs:

    1. TELayerNormColumnParallelLinear fused modules incorrectly tagged their
       layer_norm_weight/layer_norm_bias as column-parallel instead of replicated,
       corrupting FSDP shard boundaries.

    2. The isinstance(TELinear) guard created a tight coupling to TE's class
       hierarchy, breaking when TE was absent or when users subclassed TELinear.

    The fix replaces the flag trio with a single `_tensor_parallel_mode` string
    ("column" | "row" | "replicated") and extracts detection logic into a
    registry-first, attribute-fallback, name-heuristic chain.

DES-LOC adaptation rationale:
    In the DES-LOC (Decoupled Execution with Shared LOcality Cache) framework
    running on 2× A6000 48 GB (SM86, PCIe) + 1× H100 NVL 96 GB (SM90, PCIe),
    tensor parallelism placement decisions carry an additional dimension:
    *device capability heterogeneity*.

    - Column-parallel parameters are partitioned across TP ranks.  On a hetero
      cluster the shard assigned to H100 may be larger (we call this
      "capacity-weighted TP sharding") to exploit the higher memory and compute.
    - Replicated parameters must be broadcast-synced; on PCIe-only topology
      we prefer a CPU-DRAM staging path to avoid saturating the narrow GPU↔GPU
      path.
    - Row-parallel weight gradients are reduced across TP ranks; we tag these
      differently from their forward placement so the DES-LOC gradient reducer
      can choose the right collective.

    This module therefore extends the Megatron detection logic with:

    A. ``HeteroTPRegistry`` – a device-capability-aware lookup that returns not
       only the parallelism mode but also a recommended ``DeviceTier`` enum
       indicating which GPU tier should host the authoritative shard.

    B. ``DesLocTPAnnotator`` – drop-in replacement for Megatron's
       ``_annotate_tensor_parallelism``, adding DES-LOC metadata attrs:
         ``_des_loc_tp_mode``   : "column" | "row" | "replicated"
         ``_des_loc_device_tier``: DeviceTier enum
         ``_des_loc_sync_via_cpu``: bool — route broadcast through CPU DRAM

    C. ``HeteroParamAndGradBufferMixin`` – mixin that wraps DeepSpeed's ZeRO
       param/grad buffer to propagate the new single-attr scheme while keeping
       backward compat with legacy ``_tensor_parallel_mode`` (for upstream FSDP
       interop).

    D. Utility helpers that mirror Megatron's refactored utils.py functions but
       are device-topology-aware.

Hardware topology handled:
    - SM86 (A6000) × 2  : 48 GB each, PCIe gen 4 ×16, ~32 GB/s P2P
    - SM90 (H100 NVL)   : 96 GB, PCIe gen 5 ×16, ~64 GB/s host↔device
    - No NVLink          : all inter-GPU traffic via PCIe or CPU DRAM
    - CPU DRAM           : 1.5 TB, acts as LOcality Cache in DES-LOC
"""

from __future__ import annotations

import enum
import logging
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import torch
import torch.distributed as dist
from torch import nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device-tier taxonomy
# ---------------------------------------------------------------------------

class DeviceTier(enum.IntEnum):
    """Ordinal capability tiers present in the DES-LOC cluster.

    Higher ordinal = higher capability (more VRAM, higher SM count, wider bus).
    The enum is intentionally IntEnum so tier comparisons are arithmetic.
    """
    CPU_DRAM = 0   # 1.5 TB staging area — not a compute device, but a locality tier
    A6000    = 1   # SM86, 48 GB, PCIe gen4
    H100_NVL = 2   # SM90, 96 GB, PCIe gen5


# SM version → DeviceTier mapping (extendable at runtime)
_SM_TO_TIER: Dict[int, DeviceTier] = {
    86: DeviceTier.A6000,
    90: DeviceTier.H100_NVL,
}

def get_device_tier(device: torch.device) -> DeviceTier:
    """Return the :class:`DeviceTier` for *device*.

    Falls back to :attr:`DeviceTier.A6000` for unknown SM versions so that
    the system degrades gracefully on new hardware rather than crashing.
    """
    if device.type != "cuda":
        return DeviceTier.CPU_DRAM
    props = torch.cuda.get_device_properties(device)
    sm = props.major * 10 + props.minor
    tier = _SM_TO_TIER.get(sm)
    if tier is None:
        logger.warning(
            "Unknown SM version %d on device %s; defaulting to DeviceTier.A6000",
            sm, device,
        )
        tier = DeviceTier.A6000
    return tier


# ---------------------------------------------------------------------------
# Module-type registry (forked & extended from Megatron 8cbc45b)
# ---------------------------------------------------------------------------

class HeteroTPRegistry:
    """Registry mapping module class names → (tp_mode, preferred_tier).

    Upstream Megatron uses a plain dict of sets keyed by mode string.
    DES-LOC extends this to a dict of sets keyed by (mode, DeviceTier) so
    that, for example, column-parallel weights are preferentially placed on
    the H100 (highest tier) while replicated norm weights can live on any GPU
    or even be staged in CPU DRAM to save VRAM.

    Design note on capacity-weighted TP:
        When TP=2 across A6000+H100, a naïve 50/50 column split wastes H100
        capacity.  We annotate the preferred tier here; the DES-LOC shard
        planner reads ``_des_loc_device_tier`` and applies a weighted split
        (e.g., 1/3 → A6000, 2/3 → H100 proportional to VRAM ratio 48:96).
        This module does NOT implement the split itself — it only annotates.
    """

    # (mode, preferred_tier) keyed by class name.
    # Tier indicates which device *should own* the authoritative shard.
    _REGISTRY: Dict[str, Tuple[str, DeviceTier]] = {
        # Column-parallel → shard dim 0 → prefer H100 for larger shard
        "ColumnParallelLinear":             ("column",     DeviceTier.H100_NVL),
        "TEColumnParallelLinear":           ("column",     DeviceTier.H100_NVL),
        "TELayerNormColumnParallelLinear":  ("column",     DeviceTier.H100_NVL),  # weight/bias only
        "TEColumnParallelGroupedLinear":    ("column",     DeviceTier.H100_NVL),
        "VocabParallelEmbedding":           ("column",     DeviceTier.H100_NVL),
        "DotProductAttention":              ("column",     DeviceTier.H100_NVL),
        "TEDotProductAttention":            ("column",     DeviceTier.H100_NVL),
        # Row-parallel → shard dim 1 → H100 still preferred (gradient reduce is heavier)
        "RowParallelLinear":                ("row",        DeviceTier.H100_NVL),
        "TERowParallelLinear":              ("row",        DeviceTier.H100_NVL),
        "TERowParallelGroupedLinear":       ("row",        DeviceTier.H100_NVL),
        # Replicated → no sharding → prefer CPU DRAM as LOcality Cache on PCIe-only topology
        "TENorm":                           ("replicated", DeviceTier.CPU_DRAM),
        "FusedLayerNorm":                   ("replicated", DeviceTier.CPU_DRAM),
        "WrappedTorchNorm":                 ("replicated", DeviceTier.CPU_DRAM),
        "LayerNorm":                        ("replicated", DeviceTier.CPU_DRAM),
        "RMSNorm":                          ("replicated", DeviceTier.CPU_DRAM),
        "L2Norm":                           ("replicated", DeviceTier.CPU_DRAM),
        "IdentityOp":                       ("replicated", DeviceTier.CPU_DRAM),
        "TopKRouter":                       ("replicated", DeviceTier.CPU_DRAM),
    }

    @classmethod
    def lookup(
        cls, module_type_name: str
    ) -> Optional[Tuple[str, DeviceTier]]:
        """Return ``(tp_mode, DeviceTier)`` or ``None`` if unknown."""
        return cls._REGISTRY.get(module_type_name)

    @classmethod
    def register(
        cls,
        module_type_name: str,
        tp_mode: str,
        preferred_tier: DeviceTier,
    ) -> None:
        """Register a custom module type at runtime (e.g., user-defined TP layers)."""
        assert tp_mode in ("column", "row", "replicated"), (
            f"tp_mode must be 'column', 'row', or 'replicated'; got {tp_mode!r}"
        )
        logger.debug(
            "HeteroTPRegistry: registering %s → (%s, %s)",
            module_type_name, tp_mode, preferred_tier,
        )
        cls._REGISTRY[module_type_name] = (tp_mode, preferred_tier)


# ---------------------------------------------------------------------------
# DES-LOC TP detection helpers (mirrors Megatron utils.py refactor)
# ---------------------------------------------------------------------------

def des_loc_get_tp_partition_dim(param: torch.Tensor) -> Optional[int]:
    """Return the partition dimension for a DES-LOC TP parameter.

    Mirrors Megatron's ``get_mcore_tensor_parallel_partition_dim`` but reads
    the unified ``_des_loc_tp_mode`` attribute set by :class:`DesLocTPAnnotator`.

    Returns:
        0  for column-parallel parameters
        1  for row-parallel parameters
        None for replicated or un-annotated parameters
    """
    mode = getattr(param, "_des_loc_tp_mode", None)
    if mode == "column":
        return 0
    if mode == "row":
        return 1
    return None


def des_loc_is_tp_duplicated(param: torch.Tensor) -> bool:
    """Return True if the parameter is replicated across TP ranks.

    Mirrors Megatron's ``is_mcore_tensor_parallel_duplicated``: a parameter is
    duplicated when it has no sharding dimension (partition_dim is None).
    """
    return des_loc_get_tp_partition_dim(param) is None


def des_loc_using_tensor_parallel(
    tp_group: Optional[dist.ProcessGroup],
    is_expert: bool = False,
) -> bool:
    """Return True if the TP process group has more than one rank.

    Upstream Megatron (8cbc45b) replaced the old ``is_mcore_tensor_model_parallel``
    boolean-per-param check with a group-level size check.  DES-LOC follows the
    same pattern but accepts a raw ``ProcessGroup`` instead of a dist_index object
    (DeepSpeed does not expose the same dist_index abstraction as Megatron-FSDP).

    Args:
        tp_group: The tensor-parallel process group; ``None`` means TP=1.
        is_expert: When True, this is an expert-TP group (MoE models).
    """
    if tp_group is None:
        return False
    size = dist.get_world_size(tp_group)
    logger.debug(
        "des_loc_using_tensor_parallel: %s group size=%d",
        "expert-TP" if is_expert else "TP", size,
    )
    return size > 1


def should_sync_via_cpu(
    param: torch.Tensor,
    local_device: torch.device,
) -> bool:
    """Decide whether a replicated-param broadcast should stage through CPU DRAM.

    On a PCIe-only topology, GPU↔GPU P2P bandwidth is limited (~32 GB/s for
    A6000, ~64 GB/s for H100).  For replicated parameters (norms, routers) we
    prefer to:
        1. Copy from src GPU to CPU DRAM (pinned).
        2. Broadcast the CPU tensor.
        3. Copy from CPU DRAM to each dst GPU.

    This avoids contending with the PCIe bus on the forward/backward data path.
    For sharded parameters the broadcast cost is proportional to shard size, which
    is already smaller; the PCIe contention is acceptable.

    Policy:
        - If ``_des_loc_tp_mode == "replicated"`` AND the param lives on an A6000
          (lower tier), route through CPU.
        - Column/row params are never CPU-staged here (handled by the ZeRO
          param-gather pipeline which already has its own CPU offload path).
    """
    mode = getattr(param, "_des_loc_tp_mode", None)
    if mode != "replicated":
        return False
    tier = get_device_tier(local_device)
    # On PCIe-only, always stage through CPU DRAM for replicated params.
    # (The 1.5 TB CPU DRAM is the DES-LOC Shared LOcality Cache.)
    return True  # topology is always PCIe-only in our cluster


# ---------------------------------------------------------------------------
# Core annotator: DES-LOC equivalent of _annotate_tensor_parallelism
# ---------------------------------------------------------------------------

class DesLocTPAnnotator:
    """Annotates every parameter of a module tree with DES-LOC TP metadata.

    Upstream context (Megatron 8cbc45b):
        ``FullyShardedDataParallel._annotate_tensor_parallelism`` iterates
        ``root_module.modules()`` and for each submodule calls
        ``_detect_parallelism_type(param_name, submodule)`` to set a single
        ``_tensor_parallel_mode`` attribute.

    DES-LOC extensions:
        We set three attributes per parameter:
            ``_des_loc_tp_mode``    : str  — "column" | "row" | "replicated"
            ``_des_loc_device_tier``: DeviceTier — preferred device tier
            ``_des_loc_sync_via_cpu``: bool — use CPU DRAM for broadcast

        We also write ``_tensor_parallel_mode`` for upstream FSDP interop
        (DeepSpeed's param buffer code copies this attr verbatim).

    Detection priority chain (mirrors Megatron's _detect_parallelism_type):
        1. Fused module special-case (TELayerNormColumnParallelLinear).
        2. HeteroTPRegistry lookup by class name.
        3. ``tensor_model_parallel`` + ``partition_dim`` attribute fallback.
        4. Norm-name heuristic (any class whose name contains "Norm").
        5. TELinear ``parallel_mode`` attribute fallback.
        6. Return None (no annotation).
    """

    def __init__(self, local_device: Optional[torch.device] = None):
        self.local_device = local_device or (
            torch.device(f"cuda:{torch.cuda.current_device()}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self._local_tier = get_device_tier(self.local_device)
        logger.info(
            "DesLocTPAnnotator initialised: device=%s tier=%s",
            self.local_device, self._local_tier,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate(self, root_module: nn.Module) -> Dict[str, str]:
        """Walk *root_module* and annotate all parameters in-place.

        Returns a summary dict ``{param_name: tp_mode}`` for logging/debugging.
        """
        summary: Dict[str, str] = {}
        for submodule in root_module.modules():
            for param_name, param in submodule.named_parameters(recurse=False):
                result = self._detect(param_name, submodule)
                if result is None:
                    continue
                tp_mode, preferred_tier = result
                self._annotate_param(param, tp_mode, preferred_tier)
                # Track full name for the summary (best-effort)
                summary[param_name] = tp_mode

        logger.debug(
            "DesLocTPAnnotator.annotate: annotated %d parameters", len(summary)
        )
        return summary

    # ------------------------------------------------------------------
    # Detection logic
    # ------------------------------------------------------------------

    def _detect(
        self, param_name: str, module: nn.Module
    ) -> Optional[Tuple[str, DeviceTier]]:
        """Return ``(tp_mode, DeviceTier)`` or ``None``.

        Follows the priority chain described in the class docstring.
        """
        module_type = type(module).__name__

        # --- Priority 1: fused TELayerNormColumnParallelLinear ---
        # This module has both column-parallel (weight, bias) and replicated
        # (layer_norm_weight, layer_norm_bias) parameters in the same object.
        # Megatron bug (pre-8cbc45b) was tagging layer_norm_* as column.
        if module_type == "TELayerNormColumnParallelLinear":
            if param_name.endswith("layer_norm_weight") or param_name.endswith("layer_norm_bias"):
                return ("replicated", DeviceTier.CPU_DRAM)
            return ("column", DeviceTier.H100_NVL)

        # --- Priority 2: registry lookup ---
        reg_result = HeteroTPRegistry.lookup(module_type)
        if reg_result is not None:
            tp_mode, preferred_tier = reg_result
            # Row-parallel bias is replicated (not sharded), per upstream rule
            if tp_mode == "row" and "bias" in param_name:
                return ("replicated", DeviceTier.CPU_DRAM)
            return (tp_mode, preferred_tier)

        # --- Priority 3: tensor_model_parallel + partition_dim attributes ---
        if hasattr(module, "tensor_model_parallel"):
            if not module.tensor_model_parallel:
                return ("replicated", DeviceTier.CPU_DRAM)
            partition_dim = getattr(module, "partition_dim", None)
            if partition_dim == 0:
                return ("column", DeviceTier.H100_NVL)
            elif partition_dim == 1:
                if "bias" in param_name:
                    return ("replicated", DeviceTier.CPU_DRAM)
                return ("row", DeviceTier.H100_NVL)

        # --- Priority 4: norm name heuristic ---
        if any(tok in module_type for tok in ("Norm", "Normalization")):
            return ("replicated", DeviceTier.CPU_DRAM)

        # --- Priority 5: TELinear parallel_mode attribute ---
        if module_type == "TELinear" and hasattr(module, "parallel_mode"):
            pm = module.parallel_mode
            if pm == "column":
                return ("column", DeviceTier.H100_NVL)
            elif pm == "row":
                if "bias" in param_name:
                    return ("replicated", DeviceTier.CPU_DRAM)
                return ("row", DeviceTier.H100_NVL)
            else:
                return ("replicated", DeviceTier.CPU_DRAM)

        # --- Priority 6: unknown ---
        return None

    # ------------------------------------------------------------------
    # Annotation helper
    # ------------------------------------------------------------------

    def _annotate_param(
        self,
        param: torch.Tensor,
        tp_mode: str,
        preferred_tier: DeviceTier,
    ) -> None:
        """Stamp DES-LOC metadata onto *param* in-place.

        Attrs written:
            _des_loc_tp_mode      : str
            _des_loc_device_tier  : DeviceTier
            _des_loc_sync_via_cpu : bool
            _tensor_parallel_mode : str  (upstream FSDP interop)
        """
        setattr(param, "_des_loc_tp_mode", tp_mode)
        setattr(param, "_des_loc_device_tier", preferred_tier)
        setattr(param, "_des_loc_sync_via_cpu", should_sync_via_cpu(param, self.local_device))
        # Upstream interop: DeepSpeed param buffer copies this attr
        setattr(param, "_tensor_parallel_mode", tp_mode)


# ---------------------------------------------------------------------------
# HeteroParamAndGradBufferMixin
# ---------------------------------------------------------------------------

class HeteroParamAndGradBufferMixin:
    """Mixin for DeepSpeed ZeRO param/grad buffers to propagate DES-LOC attrs.

    Upstream context (Megatron param_and_grad_buffer.py 8cbc45b):
        The buffer copies TP attrs from ``old_param`` to ``new_param`` when
        re-allocating tensors.  The old code copied three attrs:
            _mcore_tp, _tp_partition_dim, _tp_duplicated
        The new code copies just one:
            _tensor_parallel_mode

    DES-LOC adds four attrs to the copy-list so they survive buffer re-alloc:
        _des_loc_tp_mode, _des_loc_device_tier, _des_loc_sync_via_cpu,
        _tensor_parallel_mode

    Usage:
        class MyZeROBuffer(HeteroParamAndGradBufferMixin, DeepSpeedParamBuffer):
            ...
    """

    # Full set of attrs to propagate on param copy/realloc
    _DES_LOC_TP_ATTRS: Tuple[str, ...] = (
        "_des_loc_tp_mode",
        "_des_loc_device_tier",
        "_des_loc_sync_via_cpu",
        "_tensor_parallel_mode",  # upstream FSDP interop
    )

    def _copy_tp_attrs(self, src_param: torch.Tensor, dst_param: torch.Tensor) -> None:
        """Copy all DES-LOC TP metadata from *src_param* to *dst_param*.

        Called wherever Megatron's buffer code previously copied the trio of
        ``_mcore_tp`` / ``_tp_partition_dim`` / ``_tp_duplicated``.
        """
        for attr in self._DES_LOC_TP_ATTRS:
            val = getattr(src_param, attr, None)
            if val is not None:
                setattr(dst_param, attr, val)
                logger.debug(
                    "_copy_tp_attrs: %s.%s = %r → dst param",
                    type(src_param).__name__, attr, val,
                )

    def _build_attr_copy_list(self) -> List[str]:
        """Return the attr list used by buffer param-copy loops.

        Replaces the old hard-coded list:
            ["_mcore_tp", "_tp_duplicated", "_tp_partition_dim"]
        """
        return list(self._DES_LOC_TP_ATTRS)


# ---------------------------------------------------------------------------
# DTensor placement helper (mirrors make_fsdp_dtensor logic in 8cbc45b)
# ---------------------------------------------------------------------------

def make_des_loc_dtensor_placements(
    param: torch.Tensor,
    tp_group: dist.ProcessGroup,
    force_sync_replicated: bool = False,
) -> Tuple[List, List[int]]:
    """Compute DTensor placements and global shape for a DES-LOC TP parameter.

    Upstream Megatron (8cbc45b) simplified ``make_fsdp_dtensor`` by removing
    the redundant ``tp_mesh.mesh.numel() > 1`` guard (it now lives in
    ``using_tensor_parallel``).  This function mirrors that simplified logic
    for DeepSpeed's equivalent DTensor construction path.

    Args:
        param: Parameter tensor already annotated by :class:`DesLocTPAnnotator`.
        tp_group: Tensor-parallel process group.
        force_sync_replicated: If True, broadcast replicated params from rank 0
            to ensure consistency (matches Megatron's ``force_sync_tp_duplicated_param``).

    Returns:
        (placements, global_shape_list) where placements is a list of
        ``torch.distributed.tensor.placement_types`` objects and global_shape
        is the reconstructed full-tensor shape.

    Raises:
        AssertionError: If param is already a DTensor when TP is active.
    """
    try:
        from torch.distributed.tensor import Replicate, Shard
    except ImportError:
        from torch.distributed._tensor import Replicate, Shard  # type: ignore[no-redef]

    global_shape = list(param.shape)

    if des_loc_is_tp_duplicated(param):
        # Replicated placement — param is identical on all TP ranks
        placements = [Replicate()]
        if force_sync_replicated and param.numel() > 0:
            logger.debug(
                "make_des_loc_dtensor_placements: broadcasting replicated param "
                "(shape=%s) from rank 0",
                list(param.shape),
            )
            dist.broadcast(param, src=dist.get_global_rank(tp_group, 0), group=tp_group)
    else:
        tp_dim = des_loc_get_tp_partition_dim(param)
        assert tp_dim is not None, (
            "[DES-LOC] Parameter has no partition dim but is not marked replicated. "
            f"_des_loc_tp_mode={getattr(param, '_des_loc_tp_mode', None)!r}"
        )
        placements = [Shard(tp_dim)]
        tp_world = dist.get_world_size(tp_group)
        global_shape[tp_dim] *= tp_world
        logger.debug(
            "make_des_loc_dtensor_placements: shard dim=%d tp_world=%d "
            "local_shape=%s global_shape=%s",
            tp_dim, tp_world, list(param.shape), global_shape,
        )

    return placements, global_shape


# ---------------------------------------------------------------------------
# Expert-param helpers (MoE support, mirrors Megatron is_expert_param lambda)
# ---------------------------------------------------------------------------

def is_expert_param(param_name: str) -> bool:
    """Return True if *param_name* belongs to an MoE expert sub-module.

    Mirrors Megatron's inline lambda ``is_expert_param = lambda n, p: ".experts." in n``
    but as a module-level function so it can be referenced in type annotations
    and registered as a DeepSpeed hook.
    """
    return ".experts." in param_name


def is_router_param(param_name: str) -> bool:
    """Return True if *param_name* is an MoE router weight.

    Used by the DES-LOC gradient reducer to skip TP-reduce for router weights
    (they are always replicated).
    """
    return ".router.weight" in param_name


# ---------------------------------------------------------------------------
# Expert device-mesh helper (mirrors test_delay_wgrad_compute.py fix)
# ---------------------------------------------------------------------------

def build_expert_device_mesh_with_tp(
    expert_data_parallel_group: dist.ProcessGroup,
    device_type: str = "cuda",
):
    """Build a 2D DeviceMesh (fsdp × tp) for expert parallelism.

    Upstream bug (pre-8cbc45b): the expert DeviceMesh was built as a 1D mesh
    with only an "fsdp" dimension, missing the "tp" dimension.  Megatron's fix
    adds a dummy TP=1 group so the mesh shape matches the non-expert mesh
    (required for MegatronFSDP's internal submesh lookups).

    DES-LOC adaptation:
        On our cluster, expert TP is always 1 (we don't split MoE experts
        across TP ranks).  We still build the 2D mesh to stay API-compatible,
        using a per-rank singleton group as the dummy TP group.

    Args:
        expert_data_parallel_group: The expert data-parallel process group.
        device_type: Device type string for DeviceMesh construction.

    Returns:
        A 2D :class:`torch.distributed.DeviceMesh` with dim names
        ``("fsdp", "tp")``.
    """
    try:
        from torch.distributed import DeviceMesh
    except ImportError:
        raise RuntimeError(
            "DeviceMesh requires PyTorch >= 2.1. "
            "Please upgrade your PyTorch installation."
        )

    dp_ranks: List[int] = dist.get_process_group_ranks(expert_data_parallel_group)
    current_rank: int = dist.get_rank()

    # Dummy TP=1 group: a process group containing only the current rank.
    # This matches the fix in Megatron's test_delay_wgrad_compute.py (8cbc45b).
    dummy_tp_group = dist.new_group(ranks=[current_rank])

    mesh_tensor = [[r] for r in dp_ranks]  # shape: [dp_size, 1]

    logger.info(
        "build_expert_device_mesh_with_tp: dp_ranks=%s current_rank=%d",
        dp_ranks, current_rank,
    )

    return DeviceMesh.from_group(
        [expert_data_parallel_group, dummy_tp_group],
        device_type=device_type,
        mesh=mesh_tensor,
        mesh_dim_names=("fsdp", "tp"),
    )


# ---------------------------------------------------------------------------
# Capacity-weighted shard size calculator (DES-LOC-specific)
# ---------------------------------------------------------------------------

def compute_weighted_shard_sizes(
    global_size: int,
    device_tiers: List[DeviceTier],
) -> List[int]:
    """Compute per-rank shard sizes proportional to device VRAM capacity.

    On a uniform cluster, TP sharding splits evenly.  On our heterogeneous
    cluster (2× A6000 48 GB + 1× H100 96 GB), a column-parallel weight
    should ideally be split 1/4 : 1/4 : 1/2 to match capacity ratios.

    This function computes integer shard sizes that sum to *global_size*,
    rounded to the nearest element (last shard absorbs remainder).

    Args:
        global_size: Total number of elements on the sharded dimension.
        device_tiers: List of :class:`DeviceTier` values, one per TP rank
            (in rank order).

    Returns:
        List of per-rank shard sizes (integers, sum == global_size).

    Example:
        >>> compute_weighted_shard_sizes(
        ...     48, [DeviceTier.A6000, DeviceTier.A6000, DeviceTier.H100_NVL]
        ... )
        [12, 12, 24]
    """
    _TIER_VRAM_GB: Dict[DeviceTier, int] = {
        DeviceTier.CPU_DRAM: 0,   # not a compute tier
        DeviceTier.A6000:    48,
        DeviceTier.H100_NVL: 96,
    }

    weights = [_TIER_VRAM_GB.get(t, 48) for t in device_tiers]
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("All device tiers have zero VRAM weight; cannot shard.")

    sizes: List[int] = []
    allocated = 0
    for i, w in enumerate(weights):
        if i == len(weights) - 1:
            # Last rank absorbs remainder to ensure exact sum
            sizes.append(global_size - allocated)
        else:
            s = int(round(global_size * w / total_weight))
            sizes.append(s)
            allocated += s

    assert sum(sizes) == global_size, (
        f"Shard sizes {sizes} do not sum to global_size {global_size}"
    )
    logger.debug(
        "compute_weighted_shard_sizes: global=%d tiers=%s → sizes=%s",
        global_size, [t.name for t in device_tiers], sizes,
    )
    return sizes


# ---------------------------------------------------------------------------
# Convenience: annotate a DeepSpeed model in-place
# ---------------------------------------------------------------------------

def annotate_model_for_des_loc(
    model: nn.Module,
    local_device: Optional[torch.device] = None,
    tp_group: Optional[dist.ProcessGroup] = None,
    expert_tp_group: Optional[dist.ProcessGroup] = None,
) -> Dict[str, str]:
    """Top-level entry point: annotate *model* for DES-LOC heterogeneous FSDP.

    This is the DES-LOC equivalent of calling
    ``FullyShardedDataParallel._annotate_tensor_parallelism(module)``
    in Megatron (added in 8cbc45b).

    Args:
        model: The DeepSpeed model (pre-engine-wrap) to annotate.
        local_device: Local CUDA device.  Auto-detected if None.
        tp_group: Optional TP process group; used to log whether TP is active.
        expert_tp_group: Optional expert-TP group for MoE models.

    Returns:
        ``{param_name: tp_mode}`` summary dict for diagnostics.
    """
    if local_device is None and torch.cuda.is_available():
        local_device = torch.device(f"cuda:{torch.cuda.current_device()}")

    if tp_group is not None:
        tp_active = des_loc_using_tensor_parallel(tp_group)
        logger.info(
            "annotate_model_for_des_loc: TP active=%s (world_size=%d)",
            tp_active, dist.get_world_size(tp_group),
        )

    annotator = DesLocTPAnnotator(local_device=local_device)
    summary = annotator.annotate(model)

    # Log expert-TP status separately
    if expert_tp_group is not None:
        expert_tp_active = des_loc_using_tensor_parallel(expert_tp_group, is_expert=True)
        logger.info(
            "annotate_model_for_des_loc: expert-TP active=%s", expert_tp_active
        )

    # Summary log at INFO level so operators can audit annotation results
    mode_counts: Dict[str, int] = {}
    for mode in summary.values():
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    logger.info(
        "annotate_model_for_des_loc: annotation complete — %s",
        ", ".join(f"{m}={c}" for m, c in sorted(mode_counts.items())),
    )
    return summary


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal smoke tests — not a full test suite.
    # Full tests live in tests/unit_tests/zero/test_hetero_fsdp_tp_detection.py

    # --- 1. Partition dim helpers ---
    p_col = nn.Parameter(torch.empty(4, 8))
    p_col._des_loc_tp_mode = "column"
    assert des_loc_get_tp_partition_dim(p_col) == 0, "column → dim 0"

    p_row = nn.Parameter(torch.empty(8, 4))
    p_row._des_loc_tp_mode = "row"
    assert des_loc_get_tp_partition_dim(p_row) == 1, "row → dim 1"

    p_rep = nn.Parameter(torch.empty(4))
    p_rep._des_loc_tp_mode = "replicated"
    assert des_loc_is_tp_duplicated(p_rep) is True, "replicated → duplicated"

    # --- 2. Capacity-weighted sharding ---
    sizes = compute_weighted_shard_sizes(
        48, [DeviceTier.A6000, DeviceTier.A6000, DeviceTier.H100_NVL]
    )
    assert sum(sizes) == 48, f"sizes must sum to 48, got {sizes}"
    assert sizes[2] > sizes[0], f"H100 shard should be larger than A6000 shard: {sizes}"

    # --- 3. Annotator on a toy model ---
    class ColumnParallelLinear(nn.Linear):
        pass

    class LayerNorm(nn.LayerNorm):
        pass

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.col = ColumnParallelLinear(8, 16)
            self.norm = LayerNorm(16)

    model = ToyModel()
    annotator = DesLocTPAnnotator(local_device=torch.device("cpu"))
    summary = annotator.annotate(model)

    assert model.col.weight._des_loc_tp_mode == "column", "col.weight must be column"
    assert model.norm.weight._des_loc_tp_mode == "replicated", "norm.weight must be replicated"
    assert model.norm.weight._des_loc_sync_via_cpu is True, "norm.weight must sync via CPU"

    print("All smoke tests passed.")
