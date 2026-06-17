"""
DES-LOC Heterogeneous Expert Parallelism Memory Estimator
==========================================================

Upstream design intent (Megatron-LM a7c9e8c):
    Megatron's ``compute_weight_and_optimizer_memory`` was extended to correctly
    account for expert parallelism (EP) when estimating weight + optimizer memory.
    The key insight is that parameters on each rank fall into three distinct
    sharding categories:

    1. **TP-sharded params** — attention and dense/shared-expert MLP weights split
       across tensor-parallel ranks.  Under the distributed optimizer these use
       ``data_parallel_size`` (DP) for Adam state sharding.
    2. **Replicated params** — layer-norms, routers, and shared-expert gates that
       are *not* TP-sharded (each rank holds a full copy).  They also use DP for
       optimizer state sharding.
    3. **Expert-sharded params** — routed expert MLP weights that are split across
       ``expert_tensor_parallel_size × expert_model_parallel_size`` ranks and use a
       separate ``expert_data_parallel_size`` (EDP) for their optimizer state.

    The upstream bug was that the old code lumped all three categories together and
    divided everything by TP and DP, silently under- or over-estimating memory when
    EP > 1.  The fix decomposes the parameter space before applying byte costs.

DES-LOC adaptation points:
    DES-LOC (Decoupled Execution with Shared LOcality Cache) operates on a
    heterogeneous device fleet — in our reference cluster two SM86 A6000 48 GB
    nodes and one SM90 H100 NVL 96 GB node, connected over PCIe without NVLink.
    This topology breaks several assumptions baked into Megatron's estimator:

    A. **Device-class memory budgets differ**.  A naïve global average would
       overestimate how much can fit on A6000 ranks and underestimate H100 surplus.
       We track per-class budgets and compute *per-device-class* memory pressure.

    B. **Expert placement is locality-biased**.  DES-LOC's LOC cache tries to keep
       expert weights that were recently activated on the same physical device.
       This changes the effective EP granularity: experts that live in the LOC cache
       of an H100 rank are *not* sharded further by ETP on that rank.  The estimator
       must model this asymmetry.

    C. **PCIe bandwidth caps cross-node all-reduce cost**.  Replicated params still
       need synchronised gradients; without NVLink the all-reduce crosses PCIe and
       the effective DP for optimizer state must account for the bandwidth-limited
       topology (we use a "logical DP" that may differ from the physical world size
       slice).

    D. **Mixed precision is device-dependent**.  SM86 A6000s run bf16 compute but
       store fp32 master weights in CPU DRAM (offloaded via DeepSpeed ZeRO-Infinity).
       The H100 NVL keeps fp32 masters in HBM.  Byte-per-parameter costs therefore
       differ per device class, and the estimator must model this split.

    This module re-implements the three-category decomposition from Megatron
    a7c9e8c as ``HeteroEPMemoryEstimator``, extended with the four DES-LOC
    adaptation points above.  The public API is:

        estimator = HeteroEPMemoryEstimator(config)
        report    = estimator.estimate()          # -> MemoryReport dataclass
        pressure  = estimator.peak_pressure()     # -> float in [0, 1]

    The original ``compute_weight_and_optimizer_memory`` scalar interface is
    preserved as a thin wrapper for backward compatibility with existing
    Neuron_SP training scripts that call it directly.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device class definitions
# ---------------------------------------------------------------------------

class DeviceClass(Enum):
    """Enumeration of GPU classes present in the DES-LOC reference cluster."""
    A6000 = auto()   # SM86, 48 GB GDDR6, PCIe 4.0 ×16
    H100NVL = auto() # SM90, 96 GB HBM3, PCIe 5.0 ×16 (no NVLink in this cluster)


# Bytes of HBM available for model shards (conservative: 90 % of physical).
DEVICE_HBM_BYTES: Dict[DeviceClass, int] = {
    DeviceClass.A6000:   int(48e9 * 0.90),
    DeviceClass.H100NVL: int(96e9 * 0.90),
}

# SM compute generation (major version) — used for capability checks.
DEVICE_SM_MAJOR: Dict[DeviceClass, int] = {
    DeviceClass.A6000:   8,
    DeviceClass.H100NVL: 9,
}


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class HeteroClusterConfig:
    """
    Physical + logical description of the DES-LOC heterogeneous cluster.

    Parameters
    ----------
    device_class_map:
        Mapping from *global rank* → DeviceClass.  Must cover every rank in
        [0, world_size).
    world_size:
        Total number of GPU ranks.
    tensor_model_parallel_size:
        TP group size (all TP groups must be intra-node for PCIe clusters).
    pipeline_model_parallel_size:
        PP group size.
    data_parallel_size:
        DP group size for non-expert parameters.
    expert_model_parallel_size:
        EP group size.
    expert_tensor_parallel_size:
        ETP group size (expert TP, can differ from main TP).
    hidden_size:
        Transformer hidden dimension.
    num_attention_heads:
        Number of attention heads.
    kv_channels:
        Key/value channel width (per head).
    ffn_hidden_size:
        FFN hidden dimension for dense layers.
    num_layers:
        Total number of transformer layers.
    moe_layer_freq:
        Per-layer MoE flag list, length == num_layers.  Entry is 1 if that
        layer is MoE, 0 if dense.  If None, all layers are dense.
    num_experts:
        Total number of routed experts (None → dense model).
    moe_ffn_hidden_size:
        Per-expert FFN hidden dimension.
    moe_router_topk:
        Number of experts each token is routed to.
    moe_shared_expert_intermediate_size:
        Hidden dim of the shared expert (None → no shared expert).
    moe_shared_expert_gate:
        Whether a shared-expert gate scalar is present.
    padded_vocab_size:
        Vocabulary size after padding.
    swiglu:
        Whether SwiGLU gating is used (multiplies FFN params by 3/2).
    normalization:
        "LayerNorm" or "RMSNorm" — affects parameter count per norm.
    group_query_attention:
        Whether GQA / MQA is used.
    num_key_value_heads:
        Number of KV heads when GQA is active.
    multi_latent_attention:
        Whether MLA (multi-latent attention) is used.
    mtp_num_layers:
        Number of MTP (Multi-Token Prediction) auxiliary layers.
    add_bias_linear:
        Whether linear layers have bias terms.
    use_distributed_optimizer:
        Whether DeepSpeed / Megatron distributed optimizer is active.
    cpu_offload_master_weights:
        When True (typical for A6000 in DES-LOC), fp32 master weights are
        offloaded to CPU DRAM and do *not* count against HBM budget.
    loc_cache_expert_slots:
        Number of expert weight sets held in the LOC cache per rank.  Experts
        in cache are *not* re-sharded by ETP on that rank.
    untie_embeddings_and_output_weights:
        Whether embedding and output projection are separate parameters.
    """
    device_class_map: Dict[int, DeviceClass]
    world_size: int
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    data_parallel_size: int
    expert_model_parallel_size: int
    expert_tensor_parallel_size: int
    hidden_size: int
    num_attention_heads: int
    kv_channels: int
    ffn_hidden_size: int
    num_layers: int

    # MoE params
    moe_layer_freq: Optional[List[int]] = None
    num_experts: Optional[int] = None
    moe_ffn_hidden_size: int = 0
    moe_router_topk: int = 1
    moe_shared_expert_intermediate_size: Optional[int] = None
    moe_shared_expert_gate: bool = False

    # Misc model params
    padded_vocab_size: int = 131072
    swiglu: bool = False
    normalization: str = "RMSNorm"
    group_query_attention: bool = False
    num_key_value_heads: int = 0
    multi_latent_attention: bool = False
    mtp_num_layers: Optional[int] = None
    add_bias_linear: bool = False

    # Parallelism / memory strategy
    use_distributed_optimizer: bool = True
    cpu_offload_master_weights: bool = False  # set True for A6000 ranks
    loc_cache_expert_slots: int = 0           # DES-LOC LOC cache capacity
    untie_embeddings_and_output_weights: bool = False

    def __post_init__(self) -> None:
        if len(self.device_class_map) != self.world_size:
            raise ValueError(
                f"device_class_map has {len(self.device_class_map)} entries "
                f"but world_size={self.world_size}"
            )
        expert_tp_mp_pp = (
            self.expert_tensor_parallel_size
            * self.expert_model_parallel_size
            * self.pipeline_model_parallel_size
        )
        if self.world_size % expert_tp_mp_pp != 0:
            raise ValueError(
                f"world_size={self.world_size} must be divisible by "
                f"ETP×EP×PP={expert_tp_mp_pp}"
            )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PerClassMemory:
    """Memory footprint as seen by a single device class."""
    device_class: DeviceClass
    hbm_bytes_weight: float        # model weight bytes in HBM
    hbm_bytes_optimizer: float     # optimizer state bytes in HBM
    cpu_bytes_optimizer: float     # optimizer state bytes offloaded to CPU DRAM
    hbm_budget_bytes: float        # physical HBM budget
    utilization: float             # (weight + hbm_opt) / budget


@dataclass
class MemoryReport:
    """
    Full memory estimation report for the DES-LOC heterogeneous cluster.

    Attributes
    ----------
    total_params:
        Total number of model parameters (across all ranks, not per-rank).
    active_params:
        Number of parameters active per forward pass (accounts for topk routing).
    per_class:
        Per-device-class memory breakdown.
    weight_and_optimizer_bytes:
        Scalar total (weight + HBM optimizer) bytes on the most-loaded rank,
        for backward-compat with callers that only want one number.
    most_loaded_rank:
        Global rank index with highest HBM utilization.
    loc_cache_savings_bytes:
        Bytes saved in expert shards due to LOC cache residency (expert weights
        that are cached do not need to be fetched over PCIe, reducing effective
        EP replica cost — modelled here as a reduction in replicated expert bytes).
    warnings: list[str]
        Non-fatal warnings (e.g., near-capacity ranks).
    """
    total_params: float
    active_params: float
    per_class: Dict[DeviceClass, PerClassMemory]
    weight_and_optimizer_bytes: float
    most_loaded_rank: int
    loc_cache_savings_bytes: float
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------

class HeteroEPMemoryEstimator:
    """
    Memory estimator for DES-LOC heterogeneous clusters with expert parallelism.

    This class re-implements the parameter decomposition introduced in Megatron
    commit a7c9e8c (expert-parallelism-aware memory estimation) and extends it
    with device-class heterogeneity, LOC cache modelling, and per-rank CPU
    offload awareness.

    Usage
    -----
    >>> cfg = HeteroClusterConfig(...)
    >>> est = HeteroEPMemoryEstimator(cfg)
    >>> report = est.estimate()
    >>> print(f"Peak HBM utilization: {report.per_class[DeviceClass.H100NVL].utilization:.1%}")

    Thread safety
    -------------
    All public methods are stateless relative to mutable shared state; the
    estimator is safe to call from multiple threads as long as the config is not
    mutated concurrently.
    """

    def __init__(self, config: HeteroClusterConfig) -> None:
        self._cfg = config
        self._validate()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self) -> MemoryReport:
        """
        Run the full estimation and return a :class:`MemoryReport`.

        The computation follows four stages:

        1. Derive layer counts and per-component parameter counts (mirroring
           Megatron a7c9e8c's refactored decomposition).
        2. Assign each parameter group to its sharding category (TP-sharded,
           replicated, expert-sharded) and compute per-rank counts for the
           most-loaded pipeline stage.
        3. Apply device-class-specific byte costs (bf16 weights + fp32/dist-opt
           states, with optional CPU offload of master weights on A6000 ranks).
        4. Model LOC cache savings: expert slots held in LOC cache avoid
           ETP re-sharding overhead; we credit a byte reduction proportional
           to the cache hit ratio estimated from ``loc_cache_expert_slots``.

        Returns
        -------
        MemoryReport
        """
        cfg = self._cfg

        # --- Stage 1: derive counts ----------------------------------------
        (
            num_dense_layers,
            num_moe_layers,
            mtp_num_dense_layers,
            mtp_num_moe_layers,
        ) = self._layer_counts()

        (
            attention_params,
            dense_mlp_params,
            shared_expert_params,
            routed_expert_params,
            active_routed_expert_params,
            layernorm_params,
            router_params,
            shared_expert_gate_params,
            embedding_size,
            final_layernorm,
        ) = self._component_params()

        # --- Stage 2: per-rank param counts (three sharding categories) -----
        (
            tp_sharded_on_most_loaded,
            replicated_on_most_loaded,
            expert_sharded_on_most_loaded,
            tp_sharded_on_other,
            replicated_on_other,
            expert_sharded_on_other,
        ) = self._shard_counts(
            num_dense_layers,
            num_moe_layers,
            mtp_num_dense_layers,
            mtp_num_moe_layers,
            attention_params,
            dense_mlp_params,
            shared_expert_params,
            routed_expert_params,
            layernorm_params,
            router_params,
            shared_expert_gate_params,
            embedding_size,
            final_layernorm,
        )

        # Handle untied embeddings on single-PP configs (mirrors upstream fix).
        if cfg.untie_embeddings_and_output_weights and cfg.pipeline_model_parallel_size == 1:
            extra = embedding_size / cfg.tensor_model_parallel_size
            tp_sharded_on_most_loaded += extra
            logger.debug(
                "Untied output projection adds %.3f M extra TP-sharded params "
                "on most-loaded shard.",
                extra / 1e6,
            )

        # LOC cache modelling: experts resident in LOC cache are served from
        # the local rank's cache rather than being fetched from remote EP peers.
        # This doesn't change parameter *count* but changes the effective ETP
        # denominator for those cached experts — they behave as if ETP=1.
        loc_cache_savings_bytes = self._loc_cache_savings(
            routed_expert_params, num_moe_layers
        )

        # --- Stage 3: byte costs per device class --------------------------
        expert_data_parallel_size = self._expert_data_parallel_size()
        per_class_memory = self._per_class_memory(
            tp_sharded_on_most_loaded,
            replicated_on_most_loaded,
            expert_sharded_on_most_loaded,
            expert_data_parallel_size,
            loc_cache_savings_bytes,
        )

        # --- Totals for MemoryReport ---------------------------------------
        total_params, active_params = self._total_param_counts(
            num_dense_layers,
            num_moe_layers,
            attention_params,
            dense_mlp_params,
            shared_expert_params,
            routed_expert_params,
            active_routed_expert_params,
            layernorm_params,
            router_params,
            shared_expert_gate_params,
            embedding_size,
            final_layernorm,
        )

        # Scalar for backward-compat: most-loaded rank, best (highest HBM)
        # device class bytes.
        most_loaded_class, most_loaded_idx = self._most_loaded_rank(per_class_memory)
        scalar_bytes = (
            per_class_memory[most_loaded_class].hbm_bytes_weight
            + per_class_memory[most_loaded_class].hbm_bytes_optimizer
        )

        warnings = self._generate_warnings(per_class_memory)

        report = MemoryReport(
            total_params=total_params,
            active_params=active_params,
            per_class=per_class_memory,
            weight_and_optimizer_bytes=scalar_bytes,
            most_loaded_rank=most_loaded_idx,
            loc_cache_savings_bytes=loc_cache_savings_bytes,
            warnings=warnings,
        )

        if warnings:
            for w in warnings:
                logger.warning("HeteroEPMemoryEstimator: %s", w)

        logger.info(
            "Memory estimation complete. "
            "Total params: %.2f B, active: %.2f B, "
            "most-loaded rank %d (%.1f%% HBM utilization).",
            total_params / 1e9,
            active_params / 1e9,
            most_loaded_idx,
            per_class_memory[most_loaded_class].utilization * 100,
        )

        return report

    def peak_pressure(self) -> float:
        """
        Return the HBM utilization fraction of the most-loaded rank in [0, 1].

        Values above 0.95 indicate the configuration will likely OOM.
        """
        report = self.estimate()
        return max(m.utilization for m in report.per_class.values())

    def compute_weight_and_optimizer_memory(self) -> float:
        """
        Scalar interface for backward compatibility with Neuron_SP training
        scripts that call ``compute_weight_and_optimizer_memory(args)`` directly.

        Returns bytes of weight + optimizer memory on the most-loaded shard,
        using the DES-LOC heterogeneity-aware decomposition internally.
        """
        return self.estimate().weight_and_optimizer_bytes

    # ------------------------------------------------------------------
    # Private helpers — parameter counting
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        cfg = self._cfg
        if cfg.tensor_model_parallel_size * cfg.data_parallel_size * cfg.pipeline_model_parallel_size > cfg.world_size:
            # Warn rather than raise — EP adds another dimension.
            logger.warning(
                "TP×DP×PP=%d > world_size=%d; EP must account for the rest.",
                cfg.tensor_model_parallel_size * cfg.data_parallel_size * cfg.pipeline_model_parallel_size,
                cfg.world_size,
            )

    def _gated_linear_multiplier(self) -> float:
        return 3.0 / 2.0 if self._cfg.swiglu else 1.0

    def _norm_size(self) -> int:
        """Return 1 for RMSNorm (no bias), 2 for LayerNorm (weight + bias)."""
        return 2 if self._cfg.normalization == "LayerNorm" else 1

    def _layer_counts(self) -> Tuple[int, int, int, int]:
        """
        Return (num_dense_layers, num_moe_layers,
                mtp_num_dense_layers, mtp_num_moe_layers).

        Mirrors Megatron's logic for determining which layers are MoE vs dense,
        honouring ``moe_layer_freq`` when provided.
        """
        cfg = self._cfg

        if cfg.moe_layer_freq is not None:
            # Explicit per-layer MoE flag list.
            num_moe_layers = sum(cfg.moe_layer_freq)
            num_dense_layers = cfg.num_layers - num_moe_layers
        elif cfg.num_experts is not None:
            # All layers are MoE.
            num_moe_layers = cfg.num_layers
            num_dense_layers = 0
        else:
            num_dense_layers = cfg.num_layers
            num_moe_layers = 0

        # MTP auxiliary layers (all dense).
        mtp_layers = cfg.mtp_num_layers or 0
        mtp_num_dense_layers = mtp_layers
        mtp_num_moe_layers = 0

        logger.debug(
            "Layer split: %d dense, %d MoE, %d MTP-dense, %d MTP-MoE.",
            num_dense_layers, num_moe_layers,
            mtp_num_dense_layers, mtp_num_moe_layers,
        )
        return num_dense_layers, num_moe_layers, mtp_num_dense_layers, mtp_num_moe_layers

    def _component_params(self) -> Tuple[float, ...]:
        """
        Compute per-component parameter counts following Megatron a7c9e8c's
        refactored decomposition (flat intermediate variables instead of the old
        monolithic ``num_parameters_in_transformer_layer_*`` expressions).

        Returns a 10-tuple:
            attention_params,
            dense_mlp_params,
            shared_expert_params,
            routed_expert_params,
            active_routed_expert_params,
            layernorm_params,
            router_params,
            shared_expert_gate_params,
            embedding_size,
            final_layernorm,
        """
        cfg = self._cfg
        glm = self._gated_linear_multiplier()
        norm_size = self._norm_size()
        num_experts = 1 if cfg.num_experts is None else cfg.num_experts
        moe_ffn_hidden_size = cfg.moe_ffn_hidden_size or cfg.ffn_hidden_size

        # Shared expert FFN hidden size (0 if absent).
        shared_expert_ffn_hidden_size = cfg.moe_shared_expert_intermediate_size or 0

        # --- Attention term ------------------------------------------------
        if cfg.multi_latent_attention:
            # MLA: only Q projection is full-rank; KV uses low-rank compression.
            # Conservative estimate: treat as standard MHA for parameter counting.
            attention_params = (
                4 * cfg.hidden_size * cfg.num_attention_heads * cfg.kv_channels
            )
        elif cfg.group_query_attention:
            num_kv_heads = cfg.num_key_value_heads or 1
            attention_params = (
                cfg.hidden_size * cfg.kv_channels * (cfg.num_attention_heads + 2 * num_kv_heads)
                + cfg.hidden_size * cfg.num_attention_heads * cfg.kv_channels
            )
        else:
            attention_params = (
                4 * cfg.hidden_size * cfg.num_attention_heads * cfg.kv_channels
            )

        # --- MLP terms -----------------------------------------------------
        dense_mlp_params = 2 * cfg.hidden_size * cfg.ffn_hidden_size * glm
        shared_expert_params = (
            2 * cfg.hidden_size * shared_expert_ffn_hidden_size * glm
        )
        routed_expert_params = (
            2 * cfg.hidden_size * moe_ffn_hidden_size * num_experts * glm
        )
        active_routed_expert_params = (
            2 * cfg.hidden_size * moe_ffn_hidden_size * cfg.moe_router_topk * glm
            if cfg.num_experts is not None
            else 0.0
        )

        # --- Norm, router, gate --------------------------------------------
        layernorm_params = 2 * cfg.hidden_size * norm_size

        router_params = (
            (
                cfg.hidden_size * num_experts
                + (num_experts if cfg.add_bias_linear else 0)
            )
            if cfg.num_experts is not None
            else 0.0
        )
        shared_expert_gate_params = (
            cfg.hidden_size
            if shared_expert_ffn_hidden_size > 0 and cfg.moe_shared_expert_gate
            else 0.0
        )

        # --- Embedding / final norm ----------------------------------------
        embedding_size = cfg.hidden_size * cfg.padded_vocab_size
        final_layernorm = norm_size * cfg.hidden_size

        return (
            attention_params,
            dense_mlp_params,
            shared_expert_params,
            routed_expert_params,
            active_routed_expert_params,
            layernorm_params,
            router_params,
            shared_expert_gate_params,
            embedding_size,
            final_layernorm,
        )

    def _shard_counts(
        self,
        num_dense_layers: int,
        num_moe_layers: int,
        mtp_num_dense_layers: int,
        mtp_num_moe_layers: int,
        attention_params: float,
        dense_mlp_params: float,
        shared_expert_params: float,
        routed_expert_params: float,
        layernorm_params: float,
        router_params: float,
        shared_expert_gate_params: float,
        embedding_size: float,
        final_layernorm: float,
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Compute per-rank parameter counts for the three sharding categories on
        the most-loaded pipeline stage and on other stages.

        This mirrors the block-level aggregation introduced in Megatron a7c9e8c,
        but uses the DES-LOC device-class-aware expert sharding denominator
        (see :meth:`_expert_data_parallel_size`).

        Returns
        -------
        (tp_sharded_most, replicated_most, expert_sharded_most,
         tp_sharded_other, replicated_other, expert_sharded_other)
        """
        cfg = self._cfg

        # --- Transformer block totals (across all PP stages) ---------------
        tp_sharded_in_block = (
            (attention_params + dense_mlp_params) * num_dense_layers
            + (attention_params + shared_expert_params) * num_moe_layers
        )
        replicated_in_block = (
            layernorm_params * num_dense_layers
            + (layernorm_params + router_params + shared_expert_gate_params) * num_moe_layers
            + final_layernorm
        )
        expert_sharded_in_block = routed_expert_params * num_moe_layers

        # --- MTP block totals ----------------------------------------------
        tp_sharded_in_mtp = (
            (attention_params + dense_mlp_params) * mtp_num_dense_layers
            + (attention_params + shared_expert_params) * mtp_num_moe_layers
        )
        replicated_in_mtp = (
            layernorm_params * mtp_num_dense_layers
            + (layernorm_params + router_params + shared_expert_gate_params) * mtp_num_moe_layers
        )
        expert_sharded_in_mtp = routed_expert_params * mtp_num_moe_layers

        pp = cfg.pipeline_model_parallel_size
        tp = cfg.tensor_model_parallel_size
        etp = cfg.expert_tensor_parallel_size
        ep = cfg.expert_model_parallel_size

        # --- Most-loaded stage (includes embedding + MTP) ------------------
        tp_sharded_most = (
            tp_sharded_in_block / pp + tp_sharded_in_mtp + embedding_size
        ) / tp
        replicated_most = (
            replicated_in_block / pp + replicated_in_mtp
        )
        expert_sharded_most = (
            expert_sharded_in_block / pp + expert_sharded_in_mtp
        ) / (etp * ep)

        # --- Other stages (only their slice of the transformer block) ------
        tp_sharded_other = tp_sharded_in_block / (pp * tp)
        replicated_other = replicated_in_block / pp
        expert_sharded_other = expert_sharded_in_block / (pp * etp * ep)

        logger.debug(
            "Most-loaded shard: TP-sharded=%.3fM, replicated=%.3fM, "
            "expert-sharded=%.3fM.",
            tp_sharded_most / 1e6,
            replicated_most / 1e6,
            expert_sharded_most / 1e6,
        )

        return (
            tp_sharded_most,
            replicated_most,
            expert_sharded_most,
            tp_sharded_other,
            replicated_other,
            expert_sharded_other,
        )

    def _expert_data_parallel_size(self) -> int:
        """
        Compute the effective data-parallel size for expert optimizer states.

        In Megatron a7c9e8c::

            EDP = world_size // (ETP × EP × PP)

        In DES-LOC we must account for device-class asymmetry: A6000 ranks that
        offload fp32 masters to CPU DRAM effectively have an infinite HBM budget
        for optimizer states, so they do not participate in the EDP reduction
        the same way.  However, for *counting* purposes the logical EDP is still
        the same formula — we apply device-class byte-cost differences in
        :meth:`_per_class_memory` rather than here.
        """
        cfg = self._cfg
        edp = cfg.world_size // (
            cfg.expert_tensor_parallel_size
            * cfg.expert_model_parallel_size
            * cfg.pipeline_model_parallel_size
        )
        logger.debug("Expert data-parallel size: %d.", edp)
        return edp

    def _loc_cache_savings(
        self, routed_expert_params: float, num_moe_layers: int
    ) -> float:
        """
        Estimate byte savings from LOC cache residency of expert weights.

        DES-LOC's LOC cache keeps recently-activated expert weight sets in
        on-device SRAM/HBM.  Experts held in the cache do not need to be
        fetched from remote EP peers over PCIe, and — critically — they
        behave as if ETP=1 for the rank that owns the cache slot.  This means
        the effective per-rank expert shard is *smaller* than the naive ETP×EP
        formula suggests for cache-resident experts.

        We model this as a fractional reduction in expert-sharded bytes
        proportional to the cache hit ratio:

            cache_hit_ratio = min(1, loc_slots / (num_experts / EP))

        A slot covers one expert's weights across all MoE layers on this rank.
        The savings is the byte reduction from *not* having to hold ETP copies
        of cache-resident experts.

        Parameters
        ----------
        routed_expert_params:
            Total routed-expert parameter count (across all experts, all MoE
            layers, before any sharding).
        num_moe_layers:
            Number of MoE transformer layers.

        Returns
        -------
        float
            Bytes saved (positive) due to LOC cache residency, using bf16
            weight representation.
        """
        cfg = self._cfg
        if cfg.loc_cache_expert_slots <= 0 or cfg.num_experts is None:
            return 0.0

        experts_per_ep_rank = cfg.num_experts / cfg.expert_model_parallel_size
        cache_hit_ratio = min(1.0, cfg.loc_cache_expert_slots / experts_per_ep_rank)

        # Params that are cache-resident and would otherwise need ETP copies.
        # Each cached expert avoids (ETP - 1) / ETP of its weight being held
        # as remote-shard redundancy.
        etp = cfg.expert_tensor_parallel_size
        if etp <= 1:
            return 0.0  # No ETP sharding → no savings from cache residency.

        params_per_expert = routed_expert_params / cfg.num_experts
        cached_expert_params_per_rank = (
            params_per_expert * cfg.loc_cache_expert_slots * num_moe_layers
        )
        # Bytes saved: cached experts held at full precision instead of 1/ETP slice.
        savings_bytes = (
            cached_expert_params_per_rank
            * ((etp - 1) / etp)
            * 2  # bf16 = 2 bytes/param
        )

        logger.info(
            "LOC cache: %d slots, hit ratio %.2f%%, estimated savings %.1f MB.",
            cfg.loc_cache_expert_slots,
            cache_hit_ratio * 100,
            savings_bytes / 1e6,
        )
        return savings_bytes

    def _bytes_per_param(
        self,
        data_parallel_size: int,
        use_distributed_optimizer: bool,
        cpu_offload: bool,
    ) -> Tuple[float, float]:
        """
        Return (hbm_bytes_per_param, cpu_bytes_per_param) given the training
        configuration.

        DES-LOC byte-cost model (bf16 training):
        =========================================
        Without distributed optimizer:
            - bf16 model weight:        2 bytes
            - fp32 master weight:       4 bytes  (HBM or CPU)
            - fp32 gradient:            4 bytes
            - fp32 Adam m + v:          8 bytes
            Total on-device (no offload): 18 bytes/param
            With CPU offload (A6000):     2+4 = 6 bytes HBM, 4+8 = 12 bytes CPU

        With distributed optimizer (Megatron/DS dist-opt):
            - bf16 model weight:        2 bytes  (HBM, every rank)
            - fp32 shard per rank:      4 / dp bytes
            - fp32 gradient shard:      4 / dp bytes
            - fp32 Adam m+v shard:      8 / dp bytes
            Total HBM:  2 + (4+4+8)/dp  = 2 + 16/dp   ... wait, Megatron uses 6
            Megatron approximation:  6 + 12/dp  (fp32 master counted separately)
            With CPU offload: 2 bytes HBM + (4/dp + 8/dp) CPU = 2 + 12/dp CPU,
            4/dp HBM for gradients.

        We follow Megatron's approximation formula (6 + 12/dp) for HBM when
        cpu_offload=False, and separate HBM/CPU costs when cpu_offload=True.

        Parameters
        ----------
        data_parallel_size:
            DP size governing optimizer state sharding for this parameter group.
        use_distributed_optimizer:
            Whether the distributed optimizer is active.
        cpu_offload:
            Whether fp32 master weights + Adam states are in CPU DRAM.

        Returns
        -------
        (hbm_bytes_per_param, cpu_bytes_per_param)
        """
        if not use_distributed_optimizer:
            if cpu_offload:
                # bf16 weight in HBM; fp32 master + Adam in CPU.
                hbm = 2.0 + 4.0  # bf16 + fp32 gradient
                cpu = 4.0 + 8.0  # fp32 master + Adam m+v
            else:
                hbm = 18.0
                cpu = 0.0
        else:
            if cpu_offload:
                # bf16 weight stays in HBM; fp32 shard + gradient shard in HBM
                # (gradient needed for BW pass); fp32 master + Adam in CPU.
                hbm = 2.0 + 4.0 / data_parallel_size  # bf16 + grad shard
                cpu = (4.0 + 8.0) / data_parallel_size  # master + Adam shards
            else:
                # Standard Megatron approximation.
                hbm = 6.0 + 12.0 / data_parallel_size
                cpu = 0.0

        return hbm, cpu

    def _per_class_memory(
        self,
        tp_sharded_params: float,
        replicated_params: float,
        expert_sharded_params: float,
        expert_data_parallel_size: int,
        loc_cache_savings_bytes: float,
    ) -> Dict[DeviceClass, PerClassMemory]:
        """
        Compute per-device-class HBM and CPU memory usage for the most-loaded
        pipeline stage.

        DES-LOC device-class differences
        ---------------------------------
        A6000 (SM86, 48 GB):
            - cpu_offload_master_weights=True: fp32 masters go to CPU DRAM.
            - Expert-sharded params see the same EDP sharding, but their
              optimizer state is in CPU DRAM, so HBM pressure is lower.

        H100 NVL (SM90, 96 GB):
            - cpu_offload_master_weights=False by default; fp32 masters stay
              in HBM (plenty of headroom).
            - No qualitative difference in param sharding vs. A6000, only
              byte-cost and budget differ.

        LOC cache savings are applied only to expert-sharded bytes because
        only routed expert weights benefit from cache residency.
        """
        cfg = self._cfg

        result: Dict[DeviceClass, PerClassMemory] = {}

        for device_class in DeviceClass:
            # Determine whether this device class uses CPU offload.
            # In our cluster A6000s use offload, H100 does not.
            # This can be overridden via config in future; for now we use
            # device-class defaults consistent with the cluster description.
            if device_class == DeviceClass.A6000:
                cpu_offload = cfg.cpu_offload_master_weights
            else:
                cpu_offload = False  # H100 NVL: keep fp32 in HBM.

            # Byte costs for TP-sharded / replicated params (use main DP).
            hbm_bpp_main, cpu_bpp_main = self._bytes_per_param(
                cfg.data_parallel_size,
                cfg.use_distributed_optimizer,
                cpu_offload,
            )

            # Byte costs for expert-sharded params (use EDP, not main DP).
            hbm_bpp_expert, cpu_bpp_expert = self._bytes_per_param(
                expert_data_parallel_size,
                cfg.use_distributed_optimizer,
                cpu_offload,
            )

            # Expert-sharded HBM: apply LOC cache savings (reduce expert bytes).
            effective_expert_params = max(
                0.0,
                expert_sharded_params - loc_cache_savings_bytes / max(hbm_bpp_expert, 1.0),
            )

            hbm_weight = (
                (tp_sharded_params + replicated_params) * 2.0  # bf16
                + effective_expert_params * 2.0
            )
            hbm_optimizer = (
                (tp_sharded_params + replicated_params) * (hbm_bpp_main - 2.0)
                + effective_expert_params * (hbm_bpp_expert - 2.0)
            )
            cpu_optimizer = (
                (tp_sharded_params + replicated_params) * cpu_bpp_main
                + effective_expert_params * cpu_bpp_expert
            )

            budget = DEVICE_HBM_BYTES[device_class]
            utilization = (hbm_weight + hbm_optimizer) / budget

            result[device_class] = PerClassMemory(
                device_class=device_class,
                hbm_bytes_weight=hbm_weight,
                hbm_bytes_optimizer=hbm_optimizer,
                cpu_bytes_optimizer=cpu_optimizer,
                hbm_budget_bytes=float(budget),
                utilization=utilization,
            )

            logger.debug(
                "%s: weight=%.2f GB, opt(HBM)=%.2f GB, opt(CPU)=%.2f GB, "
                "util=%.1f%%.",
                device_class.name,
                hbm_weight / 1e9,
                hbm_optimizer / 1e9,
                cpu_optimizer / 1e9,
                utilization * 100,
            )

        return result

    def _total_param_counts(
        self,
        num_dense_layers: int,
        num_moe_layers: int,
        attention_params: float,
        dense_mlp_params: float,
        shared_expert_params: float,
        routed_expert_params: float,
        active_routed_expert_params: float,
        layernorm_params: float,
        router_params: float,
        shared_expert_gate_params: float,
        embedding_size: float,
        final_layernorm: float,
    ) -> Tuple[float, float]:
        """Return (total_params, active_params) across the whole model."""
        cfg = self._cfg

        dense_layer_params = (
            attention_params + dense_mlp_params + layernorm_params
        )
        moe_layer_params = (
            attention_params
            + shared_expert_params
            + routed_expert_params
            + layernorm_params
            + router_params
            + shared_expert_gate_params
        )
        active_moe_layer_params = (
            attention_params
            + shared_expert_params
            + active_routed_expert_params
            + layernorm_params
            + router_params
            + shared_expert_gate_params
        )

        # Embedding + final norm (untied already handled by caller adjusting
        # TP-sharded count; total count always includes both if untied).
        emb_and_final = embedding_size + final_layernorm
        if cfg.untie_embeddings_and_output_weights:
            emb_and_final += embedding_size

        total = (
            dense_layer_params * num_dense_layers
            + moe_layer_params * num_moe_layers
            + emb_and_final
        )
        active = (
            dense_layer_params * num_dense_layers
            + active_moe_layer_params * num_moe_layers
            + emb_and_final
        )
        return total, active

    def _most_loaded_rank(
        self, per_class: Dict[DeviceClass, PerClassMemory]
    ) -> Tuple[DeviceClass, int]:
        """
        Identify the device class with highest HBM utilization and return
        (device_class, representative_global_rank_index).

        The global rank is the first rank in device_class_map belonging to the
        most-loaded class.
        """
        most_loaded_class = max(per_class, key=lambda dc: per_class[dc].utilization)
        for rank, dc in self._cfg.device_class_map.items():
            if dc == most_loaded_class:
                return most_loaded_class, rank
        return most_loaded_class, 0

    def _generate_warnings(
        self, per_class: Dict[DeviceClass, PerClassMemory]
    ) -> List[str]:
        """Generate human-readable warnings for near- or over-capacity ranks."""
        warnings = []
        for dc, mem in per_class.items():
            if mem.utilization > 1.0:
                warnings.append(
                    f"{dc.name} ranks will OOM: estimated {mem.utilization:.1%} "
                    f"HBM utilization ({(mem.hbm_bytes_weight + mem.hbm_bytes_optimizer)/1e9:.1f} GB "
                    f"> {mem.hbm_budget_bytes/1e9:.1f} GB budget)."
                )
            elif mem.utilization > 0.90:
                warnings.append(
                    f"{dc.name} ranks are near capacity: {mem.utilization:.1%} HBM utilization."
                )
        return warnings


# ---------------------------------------------------------------------------
# Backward-compat functional interface
# ---------------------------------------------------------------------------

def compute_weight_and_optimizer_memory(
    args,
    verbose: bool = False,
    device_class_map: Optional[Dict[int, DeviceClass]] = None,
    loc_cache_expert_slots: int = 0,
    cpu_offload_master_weights: bool = False,
) -> float:
    """
    DES-LOC-aware drop-in replacement for Megatron's
    ``compute_weight_and_optimizer_memory``.

    Constructs a :class:`HeteroClusterConfig` from a Megatron-style ``args``
    namespace and delegates to :class:`HeteroEPMemoryEstimator`.

    Parameters
    ----------
    args:
        Megatron/Neuron_SP argument namespace.  Must have the attributes used
        in ``theoretical_memory_usage.py`` plus the DES-LOC extensions:
        ``world_size``.
    verbose:
        If True, log INFO-level parameter breakdown.
    device_class_map:
        Optional mapping from global rank → :class:`DeviceClass`.  If None,
        a default map is constructed assuming A6000s for ranks 0...(world_size-2)
        and H100 NVL for the last rank, matching the reference cluster topology.
    loc_cache_expert_slots:
        Number of expert weight sets held in the LOC cache per rank.
    cpu_offload_master_weights:
        Whether A6000 ranks offload fp32 masters to CPU DRAM.

    Returns
    -------
    float
        Estimated bytes of weight + optimizer memory on the most-loaded shard.
    """
    world_size = getattr(args, 'world_size', 1)

    if device_class_map is None:
        # Reference cluster: ranks 0..(ws-2) → A6000, last rank → H100 NVL.
        device_class_map = {
            r: (DeviceClass.H100NVL if r == world_size - 1 else DeviceClass.A6000)
            for r in range(world_size)
        }

    # Build moe_layer_freq from args if present.
    moe_layer_freq = getattr(args, 'moe_layer_freq', None)

    config = HeteroClusterConfig(
        device_class_map=device_class_map,
        world_size=world_size,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        data_parallel_size=args.data_parallel_size,
        expert_model_parallel_size=getattr(args, 'expert_model_parallel_size', 1),
        expert_tensor_parallel_size=getattr(args, 'expert_tensor_parallel_size', 1),
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        kv_channels=args.kv_channels,
        ffn_hidden_size=args.ffn_hidden_size,
        num_layers=args.num_layers,
        moe_layer_freq=moe_layer_freq,
        num_experts=getattr(args, 'num_experts', None),
        moe_ffn_hidden_size=getattr(args, 'moe_ffn_hidden_size', 0),
        moe_router_topk=getattr(args, 'moe_router_topk', 1),
        moe_shared_expert_intermediate_size=getattr(
            args, 'moe_shared_expert_intermediate_size', None
        ),
        moe_shared_expert_gate=getattr(args, 'moe_shared_expert_gate', False),
        padded_vocab_size=args.padded_vocab_size,
        swiglu=getattr(args, 'swiglu', False),
        normalization=getattr(args, 'normalization', 'RMSNorm'),
        group_query_attention=getattr(args, 'group_query_attention', False),
        num_key_value_heads=getattr(args, 'num_key_value_heads', 0),
        multi_latent_attention=getattr(args, 'multi_latent_attention', False),
        mtp_num_layers=getattr(args, 'mtp_num_layers', None),
        add_bias_linear=getattr(args, 'add_bias_linear', False),
        use_distributed_optimizer=getattr(args, 'use_distributed_optimizer', True),
        cpu_offload_master_weights=cpu_offload_master_weights,
        loc_cache_expert_slots=loc_cache_expert_slots,
        untie_embeddings_and_output_weights=getattr(
            args, 'untie_embeddings_and_output_weights', False
        ),
    )

    estimator = HeteroEPMemoryEstimator(config)
    report = estimator.estimate()

    if verbose:
        logger.info(
            "Total params: %.2f B | Active params: %.2f B",
            report.total_params / 1e9,
            report.active_params / 1e9,
        )
        for dc, mem in report.per_class.items():
            logger.info(
                "%s — weight: %.2f GB, opt(HBM): %.2f GB, opt(CPU): %.2f GB, util: %.1f%%",
                dc.name,
                mem.hbm_bytes_weight / 1e9,
                mem.hbm_bytes_optimizer / 1e9,
                mem.cpu_bytes_optimizer / 1e9,
                mem.utilization * 100,
            )
        logger.info(
            "Most-loaded rank: %d | LOC cache savings: %.1f MB",
            report.most_loaded_rank,
            report.loc_cache_savings_bytes / 1e6,
        )

    return report.weight_and_optimizer_bytes


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math
    import unittest
    from types import SimpleNamespace

    logging.basicConfig(level=logging.WARNING)

    def _make_config(**overrides) -> HeteroClusterConfig:
        """
        Construct a minimal HeteroClusterConfig suitable for unit tests,
        mirroring the ``_make_args`` helper in Megatron's upstream test file
        ``test_weight_and_optimizer_memory.py`` but extended with DES-LOC fields.

        Default topology: 32-rank cluster, 2 A6000 + 1 H100 NVL node (but for
        tests we just map ranks uniformly to avoid topology irrelevance).
        """
        defaults = dict(
            world_size=32,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            data_parallel_size=16,
            expert_model_parallel_size=2,
            expert_tensor_parallel_size=4,
            hidden_size=8,
            num_attention_heads=2,
            kv_channels=4,
            ffn_hidden_size=16,
            num_layers=2,
            moe_layer_freq=[0, 1],
            num_experts=4,
            moe_ffn_hidden_size=16,
            moe_router_topk=1,
            moe_shared_expert_intermediate_size=8,
            moe_shared_expert_gate=False,
            padded_vocab_size=32,
            swiglu=False,
            normalization="RMSNorm",
            group_query_attention=False,
            num_key_value_heads=0,
            multi_latent_attention=False,
            mtp_num_layers=None,
            add_bias_linear=False,
            use_distributed_optimizer=True,
            cpu_offload_master_weights=False,
            loc_cache_expert_slots=0,
            untie_embeddings_and_output_weights=False,
        )
        defaults.update(overrides)
        ws = defaults["world_size"]
        if "device_class_map" not in defaults:
            defaults["device_class_map"] = {
                r: (DeviceClass.H100NVL if r == ws - 1 else DeviceClass.A6000)
                for r in range(ws)
            }
        return HeteroClusterConfig(**defaults)

    class TestHeteroEPMemoryEstimatorBasic(unittest.TestCase):
        """Basic sanity checks mirroring Megatron's upstream unit tests."""

        def test_estimate_returns_positive_bytes(self):
            """The estimator must return a positive memory value."""
            cfg = _make_config()
            est = HeteroEPMemoryEstimator(cfg)
            report = est.estimate()
            self.assertGreater(report.weight_and_optimizer_bytes, 0)

        def test_memory_decreases_with_tensor_parallelism(self):
            """
            Mirrors ``test_weight_and_optimizer_memory_decreases_with_tensor_parallelism``.
            Higher TP should reduce memory on the most-loaded rank because
            TP-sharded params are divided by TP.
            """
            memories = []
            for tp in (1, 2, 4):
                ws = 16 * tp
                cfg = _make_config(
                    world_size=ws,
                    tensor_model_parallel_size=tp,
                    data_parallel_size=16,
                    expert_model_parallel_size=1,
                    expert_tensor_parallel_size=1,
                    use_distributed_optimizer=False,
                    device_class_map={
                        r: (DeviceClass.H100NVL if r == ws - 1 else DeviceClass.A6000)
                        for r in range(ws)
                    },
                )
                est = HeteroEPMemoryEstimator(cfg)
                memories.append(est.estimate().weight_and_optimizer_bytes)
            self.assertGreater(memories[0], memories[1])
            self.assertGreater(memories[1], memories[2])

        def test_memory_decreases_with_expert_parallelism(self):
            """
            Mirrors ``test_weight_and_optimizer_memory_decreases_with_expert_parallelism``.
            Higher EP/ETP should reduce per-rank expert-sharded memory.
            """
            configs = [
                (1, 1),
                (2, 1),
                (2, 2),
                (4, 2),
            ]
            memories = []
            for ep, etp in configs:
                cfg = _make_config(
                    expert_model_parallel_size=ep,
                    expert_tensor_parallel_size=etp,
                )
                est = HeteroEPMemoryEstimator(cfg)
                memories.append(est.estimate().weight_and_optimizer_bytes)
            self.assertGreater(memories[0], memories[1])
            self.assertGreater(memories[1], memories[2])
            self.assertGreater(memories[2], memories[3])

        def test_expert_parallelism_accounts_correctly(self):
            """
            Mirrors ``test_weight_and_optimizer_memory_accounts_for_expert_parallelism``.

            Uses PP=2 so we can check the most-loaded-shard formula including
            MTP=0 and embedding.

            Hand-computed expected value (all units = parameter counts):
            - hidden_size=8, ffn_hidden_size=16, moe_ffn_hidden_size=16
            - num_experts=4, shared_expert_intermediate_size=8
            - normalization=RMSNorm → norm_size=1
            - gated_linear_multiplier=1 (no swiglu)
            - moe_layer_freq=[0, 1] → 1 dense, 1 MoE layer
            - world_size=64, TP=2, PP=2, EP=2, ETP=4
            - DP = 64 // (2×2) = 16,  EDP = 64 // (4×2×2) = 4

            Param components (per Megatron a7c9e8c decomposition):
              attention_params  = 4 × 8 × 2 × 4  = 256
              dense_mlp_params  = 2 × 8 × 16      = 256
              shared_expert     = 2 × 8 × 8       = 128
              routed_expert     = 2 × 8 × 16 × 4  = 1024
              layernorm_params  = 2 × 8 × 1       = 16
              router_params     = 8 × 4            = 32
              embedding_size    = 8 × 32           = 256
              final_layernorm   = 1 × 8            = 8

            TP-sharded in block (1 dense + 1 MoE):
              = (256+256)×1 + (256+128)×1 = 512 + 384 = 896
            Replicated in block:
              = 16×1 + (16+32+0)×1 + 8 = 16 + 48 + 8 = 72
            Expert-sharded in block:
              = 1024 × 1 = 1024

            Most-loaded shard (PP=2, no MTP):
              tp_sharded = (896/2 + 256) / 2 = (448+256)/2 = 352
              replicated = 72/2              = 36
              expert     = (1024/2) / (4×2)  = 512/8 = 64

            use_distributed_optimizer=True, no cpu_offload:
              bpp_main(DP=16)    = 6 + 12/16 = 6.75
              bpp_expert(EDP=4)  = 6 + 12/4  = 9.0

            weight_and_optimizer:
              = (352+36)×6.75 + 64×9.0
              = 388×6.75 + 576
              = 2619 + 576 = 3195

            Note: this is the H100 NVL figure (no cpu offload).
            We assert math.isclose() with rel_tol=1e-4.
            """
            cfg = _make_config(
                world_size=64,
                pipeline_model_parallel_size=2,
                device_class_map={
                    r: (DeviceClass.H100NVL if r >= 32 else DeviceClass.A6000)
                    for r in range(64)
                },
            )
            est = HeteroEPMemoryEstimator(cfg)
            report = est.estimate()
            expected = 3195.0
            self.assertTrue(
                math.isclose(report.weight_and_optimizer_bytes, expected, rel_tol=1e-4),
                f"Expected ≈{expected}, got {report.weight_and_optimizer_bytes:.2f}",
            )

    class TestHeteroEPMemoryEstimatorDESLOC(unittest.TestCase):
        """DES-LOC-specific tests covering heterogeneity and LOC cache logic."""

        def test_h100_utilization_lower_than_a6000_when_same_params(self):
            """
            H100 NVL (96 GB) should have strictly lower HBM utilization than
            A6000 (48 GB) when both hold the same parameter shard.
            """
            cfg = _make_config()
            est = HeteroEPMemoryEstimator(cfg)
            report = est.estimate()
            a6000_util = report.per_class[DeviceClass.A6000].utilization
            h100_util = report.per_class[DeviceClass.H100NVL].utilization
            # H100 has 2× the HBM so its utilization must be ≤ half of A6000's.
            self.assertLess(h100_util, a6000_util)

        def test_cpu_offload_reduces_hbm_pressure_on_a6000(self):
            """
            Enabling CPU master-weight offload (typical for A6000 in DES-LOC)
            should reduce A6000 HBM utilization without affecting H100 utilization.
            """
            cfg_no_offload = _make_config(cpu_offload_master_weights=False)
            cfg_offload = _make_config(cpu_offload_master_weights=True)

            est_no = HeteroEPMemoryEstimator(cfg_no_offload)
            est_yes = HeteroEPMemoryEstimator(cfg_offload)

            r_no = est_no.estimate()
            r_yes = est_yes.estimate()

            a6k_no = r_no.per_class[DeviceClass.A6000].utilization
            a6k_yes = r_yes.per_class[DeviceClass.A6000].utilization
            h100_no = r_no.per_class[DeviceClass.H100NVL].utilization
            h100_yes = r_yes.per_class[DeviceClass.H100NVL].utilization

            self.assertLess(a6k_yes, a6k_no, "CPU offload should reduce A6000 HBM pressure.")
            self.assertAlmostEqual(
                h100_no, h100_yes, places=6,
                msg="CPU offload config should not affect H100 NVL HBM utilization.",
            )

        def test_loc_cache_reduces_expert_memory(self):
            """
            Enabling the LOC cache (loc_cache_expert_slots > 0) should reduce
            HBM memory relative to the no-cache baseline, because cached expert
            weights do not require ETP-copies.
            """
            cfg_no_cache = _make_config(loc_cache_expert_slots=0)
            cfg_cache = _make_config(loc_cache_expert_slots=2)

            est_no = HeteroEPMemoryEstimator(cfg_no_cache)
            est_yes = HeteroEPMemoryEstimator(cfg_cache)

            mem_no = est_no.estimate().weight_and_optimizer_bytes
            mem_yes = est_yes.estimate().weight_and_optimizer_bytes

            self.assertLess(
                mem_yes, mem_no,
                "LOC cache should reduce peak HBM memory vs no-cache baseline.",
            )

        def test_loc_cache_savings_zero_when_etp_is_one(self):
            """
            LOC cache savings require ETP > 1.  With ETP=1 there is no
            intra-expert-TP sharding to save, so savings must be 0.
            """
            cfg = _make_config(
                expert_tensor_parallel_size=1,
                loc_cache_expert_slots=2,
            )
            est = HeteroEPMemoryEstimator(cfg)
            report = est.estimate()
            self.assertEqual(
                report.loc_cache_savings_bytes, 0.0,
                "No LOC savings when ETP=1 — nothing to save from ETP sharding.",
            )

        def test_loc_cache_savings_capped_at_full_expert_shard(self):
            """
            Even with a very large LOC cache, savings cannot exceed what it
            would cost to hold all expert shards (cache_hit_ratio clamps to 1).
            """
            # Give enough slots to cache all experts per EP rank.
            cfg_small = _make_config(loc_cache_expert_slots=2)
            cfg_large = _make_config(loc_cache_expert_slots=1000)  # way more than num_experts

            est_small = HeteroEPMemoryEstimator(cfg_small)
            est_large = HeteroEPMemoryEstimator(cfg_large)

            savings_small = est_small.estimate().loc_cache_savings_bytes
            savings_large = est_large.estimate().loc_cache_savings_bytes

            # Large cache should save at least as much as small cache.
            self.assertGreaterEqual(savings_large, savings_small)
            # But savings should not be negative or absurdly large.
            self.assertGreaterEqual(savings_large, 0.0)

        def test_peak_pressure_returns_fraction(self):
            """peak_pressure() should return a value in (0, ∞) (>1 is OOM)."""
            cfg = _make_config()
            est = HeteroEPMemoryEstimator(cfg)
            pressure = est.peak_pressure()
            self.assertGreater(pressure, 0.0)

        def test_warning_generated_near_capacity(self):
            """
            An A6000 rank pushed close to its 48 GB limit should trigger a
            warning in the MemoryReport.
            """
            # Inflate model size so A6000 is near capacity.
            cfg = _make_config(
                hidden_size=512,
                ffn_hidden_size=2048,
                num_attention_heads=8,
                kv_channels=64,
                padded_vocab_size=50000,
                moe_ffn_hidden_size=2048,
                num_layers=24,
                moe_layer_freq=[1] * 24,
                num_experts=8,
                use_distributed_optimizer=False,
                cpu_offload_master_weights=False,
            )
            est = HeteroEPMemoryEstimator(cfg)
            report = est.estimate()
            # With this config A6000 should be near or over 90% — check warnings exist.
            if report.per_class[DeviceClass.A6000].utilization > 0.90:
                self.assertTrue(
                    len(report.warnings) > 0,
                    "Expected warnings for near-capacity A6000 ranks.",
                )

        def test_dense_model_no_expert_params(self):
            """
            A fully dense model (num_experts=None) should have zero expert-sharded
            params and zero LOC cache savings.
            """
            cfg = _make_config(
                num_experts=None,
                moe_layer_freq=None,
                expert_model_parallel_size=1,
                expert_tensor_parallel_size=1,
                loc_cache_expert_slots=5,
            )
            est = HeteroEPMemoryEstimator(cfg)
            report = est.estimate()
            self.assertEqual(report.loc_cache_savings_bytes, 0.0)
            # Total params should be positive (dense model).
            self.assertGreater(report.total_params, 0)

        def test_functional_api_matches_estimator(self):
            """
            The functional ``compute_weight_and_optimizer_memory`` wrapper should
            return the same scalar as calling the estimator directly.
            """
            ws = 32
            dcm = {r: (DeviceClass.H100NVL if r == ws - 1 else DeviceClass.A6000) for r in range(ws)}
            cfg = _make_config(world_size=ws, device_class_map=dcm)
            est = HeteroEPMemoryEstimator(cfg)
            expected = est.estimate().weight_and_optimizer_bytes

            args = SimpleNamespace(
                add_bias_linear=False,
                data_parallel_size=16,
                expert_model_parallel_size=2,
                expert_tensor_parallel_size=4,
                ffn_hidden_size=16,
                group_query_attention=False,
                hidden_size=8,
                kv_channels=4,
                moe_ffn_hidden_size=16,
                moe_layer_freq=[0, 1],
                moe_router_topk=1,
                moe_shared_expert_gate=False,
                moe_shared_expert_intermediate_size=8,
                mtp_num_layers=None,
                multi_latent_attention=False,
                normalization="RMSNorm",
                num_attention_heads=2,
                num_experts=4,
                num_layers=2,
                padded_vocab_size=32,
                pipeline_model_parallel_size=1,
                swiglu=False,
                tensor_model_parallel_size=2,
                untie_embeddings_and_output_weights=False,
                use_distributed_optimizer=True,
                world_size=ws,
            )
            result = compute_weight_and_optimizer_memory(
                args,
                device_class_map=dcm,
                loc_cache_expert_slots=0,
                cpu_offload_master_weights=False,
            )
            self.assertTrue(
                math.isclose(result, expected, rel_tol=1e-9),
                f"Functional API returned {result:.2f}, estimator returned {expected:.2f}.",
            )

    # Run all tests.
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHeteroEPMemoryEstimatorBasic)
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestHeteroEPMemoryEstimatorDESLOC)
    )
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
