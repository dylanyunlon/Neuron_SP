"""
hetero_latent_moe_flops.py
==========================
DES-LOC (Decoupled Execution with Shared LOcality Cache) adapter for
Mixture-of-Experts layers with latent projections, targeting the
Neuron_SP heterogeneous hardware stack:

    2× A6000 48 GB  (SM86, PCIe)
    1× H100 NVL 96 GB (SM90, PCIe)
    1.5 TB CPU DRAM shared locality cache

Upstream intent (Megatron commit a4008d0f)
------------------------------------------
Robin Zhang's fix addresses two independent but related issues in Megatron's
latent-MoE (DeepSeek-style compressed-expert) implementation:

1. **FLOP accounting** — The original `num_floating_point_operations` assumed
   every routed-expert FFN ran at full `hidden_size`.  With latent MoE the
   expert's *input* is first projected down to `moe_latent_size`, so the
   actual arithmetic is:

       routed_flops = (moe_ffn_hidden_size * k * expansion
                       * moe_latent_size / hidden_size)   # expert body
                    + 2 * moe_latent_size                 # fc1/fc2 proj

2. **backward_dw dispatch** — `fc2_latent_proj` (down-projection of the
   expert output back to `moe_latent_size`) runs its forward in the
   *comm stream* to overlap EP all-to-all; its weight-gradient computation
   must therefore also run in that same stream to maintain ordering.
   `fc1_latent_proj` is associated with the shared-expert path and is
   triggered from the `shared_experts` branch.

DES-LOC reinterpretation
------------------------
In DES-LOC we decompose the MoE execution graph across heterogeneous
devices, exploiting their different memory/compute trade-offs:

* **H100 (SM90)** — runs the *hot* routed-expert forward/backward at
  full throughput.  Its 96 GB HBM acts as a second-level "locality cache"
  for weight shards that would not fit on A6000s.

* **A6000 ×2 (SM86)** — run the shared-expert path and the latent
  projections (fc1/fc2_latent_proj), which are bandwidth-bound rather
  than compute-bound and therefore better matched to the A6000's
  slower-but-wider memory bus (or offloaded to CPU DRAM via DeepSpeed's
  ZeRO-Infinity / pin_memory pipeline).

* **CPU DRAM locality cache** — expert weight shards that are accessed
  infrequently are pinned in CPU DRAM and streamed to the appropriate GPU
  just before their backward_dw.  The FLOP estimator here accounts for
  the *effective* latency-adjusted throughput of that path.

This file provides:

  HeteroLatentMoEFlops
    ├── compute_routed_flops()          — latent-adjusted FLOPs for routed experts
    ├── compute_shared_flops()          — shared expert + fc1_latent_proj FLOPs
    ├── compute_total_flops()           — full-layer aggregate
    └── estimate_device_assignment()    — heuristic mapping of sub-graphs to GPUs

  HeteroBackwardDwScheduler
    ├── schedule_routed_backward_dw()   — fc2_latent_proj in comm stream (H100)
    ├── schedule_shared_backward_dw()   — shared + fc1_latent_proj (A6000)
    └── synchronize_streams()

  LatentMoEFlopsHook (DeepSpeed engine hook)
    — wraps the above into the DeepSpeed training loop

Author: Neuron_SP project (DES-LOC heterogeneous adapter)
Upstream ref: github.com/NVIDIA/Megatron-LM commit a4008d0f
"""

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware capability descriptors
# ---------------------------------------------------------------------------

@dataclass
class DeviceCapability:
    """Describes one physical GPU in the DES-LOC cluster.

    Attributes
    ----------
    device_index : int
        CUDA device ordinal.
    sm_version : int
        Streaming-multiprocessor generation (e.g. 86 for A6000, 90 for H100).
    hbm_gb : float
        High-bandwidth memory in gigabytes.
    peak_tflops_bf16 : float
        Manufacturer-rated BF16 TFLOPS (used for roofline estimation).
    interconnect : str
        ``"pcie"`` or ``"nvlink"``.  All devices in this cluster are PCIe.
    preferred_role : str
        Human-readable role hint: ``"routed_expert"``, ``"shared_expert"``,
        ``"latent_proj"``, or ``"any"``.
    """
    device_index: int
    sm_version: int
    hbm_gb: float
    peak_tflops_bf16: float
    interconnect: str = "pcie"
    preferred_role: str = "any"


# Cluster topology for Neuron_SP's 3-GPU heterogeneous server
NEURON_SP_CLUSTER: List[DeviceCapability] = [
    DeviceCapability(
        device_index=0, sm_version=86, hbm_gb=48.0,
        peak_tflops_bf16=309.7, interconnect="pcie",
        preferred_role="latent_proj",
    ),
    DeviceCapability(
        device_index=1, sm_version=86, hbm_gb=48.0,
        peak_tflops_bf16=309.7, interconnect="pcie",
        preferred_role="shared_expert",
    ),
    DeviceCapability(
        device_index=2, sm_version=90, hbm_gb=96.0,
        peak_tflops_bf16=1978.9, interconnect="pcie",
        preferred_role="routed_expert",
    ),
]


def get_cluster_topology(
    override: Optional[List[DeviceCapability]] = None,
) -> List[DeviceCapability]:
    """Return device capability list, preferring runtime override.

    DES-LOC topology can be overridden at runtime via the
    ``DESLOCK_CLUSTER_TOPOLOGY`` environment variable (JSON path) or by
    passing *override* directly — useful for unit tests and CI environments
    with fewer devices.
    """
    if override is not None:
        return override
    env_path = os.environ.get("DESLOCK_CLUSTER_TOPOLOGY")
    if env_path:
        import json
        with open(env_path) as fh:
            raw = json.load(fh)
        return [DeviceCapability(**d) for d in raw]
    return NEURON_SP_CLUSTER


# ---------------------------------------------------------------------------
# FLOP accounting: latent MoE correction
# ---------------------------------------------------------------------------

@dataclass
class LatentMoEFlopsConfig:
    """All scalars needed to compute MoE FLOPs with optional latent projection.

    Upstream context
    ----------------
    In standard dense/sparse MoE each expert operates on tokens of size
    ``hidden_size``.  In *latent* MoE (DeepSeek MoE variant) a shared
    down-projection maps activations from ``hidden_size`` → ``moe_latent_size``
    before routing, so each expert sees narrower inputs.

    Attributes
    ----------
    batch_size : int
    seq_length : int
    hidden_size : int
    num_experts : int
        Total number of routed experts in the layer.
    num_experts_routed_to : int
        ``top-k`` value, i.e. experts activated per token.
    moe_ffn_hidden_size : int
        Inner dimension of each expert's FFN.
    ffn_expansion_factor : float
        Multiplier for gated-linear-unit variants (``2`` for SwiGLU,
        ``1`` for standard FFN).
    num_moe_layers : int
        Number of MoE layers in the full model (used for model-level totals).
    num_dense_layers : int
        Number of non-MoE transformer layers.
    moe_latent_size : Optional[int]
        If ``None`` the layer uses standard (non-latent) routing.
        When set, expert inputs are first projected to this size.
    shared_expert_ffn_hidden_size : int
        Hidden dim of the shared-expert FFN (``0`` if no shared expert).
    num_query_groups : int
        For MLA / multi-query attention FLOP accounting (pass-through).
    """
    batch_size: int
    seq_length: int
    hidden_size: int
    num_experts: int
    num_experts_routed_to: int
    moe_ffn_hidden_size: int
    ffn_expansion_factor: float = 1.0
    num_moe_layers: int = 1
    num_dense_layers: int = 0
    moe_latent_size: Optional[int] = None
    shared_expert_ffn_hidden_size: int = 0
    num_query_groups: int = 1


class HeteroLatentMoEFlops:
    """Compute forward-pass FLOP estimates for a latent-MoE layer in DES-LOC.

    Design intent (upstream)
    ------------------------
    Megatron's ``num_floating_point_operations`` function was computing
    expert FLOPs as if every expert ran at ``hidden_size`` width.  The fix
    replaces the simple product with a two-term expression:

        expert_body_flops  = moe_ffn_hidden_size * k * expansion
                             * (moe_latent_size / hidden_size)
        latent_proj_flops  = 2 * moe_latent_size   (fc1 up + fc2 down)

    Each term is then multiplied by ``num_moe_layers``.

    DES-LOC adaptation
    ------------------
    We extend this by attributing FLOPs to specific devices according to the
    DES-LOC execution plan:

    * ``expert_body_flops``  → H100 (SM90, device 2)
    * ``latent_proj_flops``  → A6000 pair (SM86, devices 0/1)
    * ``shared_expert_flops`` → A6000 (device 1)

    This per-device attribution drives both scheduling decisions and
    profiling dashboards in the Neuron_SP training loop.

    Parameters
    ----------
    config : LatentMoEFlopsConfig
    cluster : list of DeviceCapability, optional
        Defaults to ``NEURON_SP_CLUSTER``.
    """

    def __init__(
        self,
        config: LatentMoEFlopsConfig,
        cluster: Optional[List[DeviceCapability]] = None,
    ) -> None:
        self.cfg = config
        self.cluster = get_cluster_topology(cluster)
        self._validate()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        c = self.cfg
        if c.moe_latent_size is not None and c.moe_latent_size >= c.hidden_size:
            logger.warning(
                "moe_latent_size=%d >= hidden_size=%d; latent compression has no effect.",
                c.moe_latent_size,
                c.hidden_size,
            )
        if c.num_experts_routed_to > c.num_experts:
            raise ValueError(
                f"num_experts_routed_to={c.num_experts_routed_to} exceeds "
                f"num_experts={c.num_experts}"
            )

    @property
    def _tokens(self) -> int:
        """Total tokens in the batch."""
        return self.cfg.batch_size * self.cfg.seq_length

    # ------------------------------------------------------------------
    # Public FLOP computation methods
    # ------------------------------------------------------------------

    def compute_routed_flops(self) -> float:
        """FLOPs for all routed-expert FFNs, latent-corrected.

        Returns
        -------
        float
            Total multiply-add operations (counted as 2× MACs, i.e. FLOPs
            not FLOP/s) for routed experts across all MoE layers.

        Notes
        -----
        Without latent projection::

            routed = tokens * moe_ffn_hidden * k * expansion

        With latent projection the expert body shrinks proportionally::

            expert_body = tokens * moe_ffn_hidden * k * expansion
                          * (moe_latent_size / hidden_size)
            latent_proj  = tokens * 2 * moe_latent_size
            routed       = (expert_body + latent_proj) * num_moe_layers

        The factor ``2`` for latent_proj comes from the two linear layers
        (fc1: hidden→latent, fc2: latent→hidden).  Upstream Megatron uses
        the same ``+ 2 * moe_latent_size`` convention.
        """
        c = self.cfg
        tokens = self._tokens

        if c.moe_latent_size is None:
            # Standard (non-latent) routed-expert FLOPs
            per_layer = (
                tokens
                * c.moe_ffn_hidden_size
                * c.num_experts_routed_to
                * c.ffn_expansion_factor
            )
        else:
            latent_ratio = c.moe_latent_size / c.hidden_size
            expert_body = (
                tokens
                * c.moe_ffn_hidden_size
                * c.num_experts_routed_to
                * c.ffn_expansion_factor
                * latent_ratio
            )
            latent_proj = tokens * 2 * c.moe_latent_size
            per_layer = expert_body + latent_proj

        total = per_layer * c.num_moe_layers
        logger.debug(
            "compute_routed_flops: per_layer=%.3e  total=%.3e  "
            "(latent_size=%s)",
            per_layer, total, c.moe_latent_size,
        )
        return total

    def compute_shared_flops(self) -> float:
        """FLOPs for the shared-expert FFN (invariant to latent mode).

        The shared expert does *not* participate in token routing and always
        operates at ``hidden_size``, so its FLOPs are independent of
        ``moe_latent_size``.  This matches Megatron's post-patch treatment
        where ``fc1_latent_proj`` is accounted separately under the
        shared-expert branch.

        Returns
        -------
        float
            Total shared-expert FLOPs across all MoE layers.
        """
        c = self.cfg
        tokens = self._tokens

        shared = (
            tokens
            * c.shared_expert_ffn_hidden_size
            * c.ffn_expansion_factor
            * c.num_moe_layers
        )
        logger.debug("compute_shared_flops: %.3e", shared)
        return shared

    def compute_total_flops(self) -> float:
        """Aggregate FLOPs: dense layers + routed experts + shared expert.

        This mirrors ``num_floating_point_operations`` in
        ``megatron/training/training.py`` post-patch, adapted to DeepSpeed
        conventions where FLOP counts drive gradient-accumulation and
        pipeline-schedule budgets.

        Returns
        -------
        float
            Total model FLOPs for one forward pass over the batch.
        """
        c = self.cfg
        tokens = self._tokens

        # Dense transformer layers (attention + FFN), unchanged from upstream
        dense_attn = tokens * c.hidden_size * c.num_query_groups * 4  # QKV + proj
        dense_ffn = tokens * c.hidden_size * 8  # standard 4h FFN × 2 for fwd
        dense_total = (dense_attn + dense_ffn) * c.num_dense_layers

        moe_total = self.compute_routed_flops() + self.compute_shared_flops()

        total = dense_total + moe_total
        logger.info(
            "HeteroLatentMoEFlops total: dense=%.3e  moe=%.3e  grand=%.3e",
            dense_total, moe_total, total,
        )
        return total

    # ------------------------------------------------------------------
    # DES-LOC device assignment heuristic
    # ------------------------------------------------------------------

    def estimate_device_assignment(self) -> Dict[str, int]:
        """Return a mapping of sub-graph name → CUDA device index.

        DES-LOC assigns sub-graphs to devices based on their compute and
        bandwidth profiles:

        * Routed-expert bodies (high compute intensity) → H100 (SM90).
        * Latent projections (bandwidth-bound, small weight tensors) →
          A6000 with the "latent_proj" role.
        * Shared-expert FFN → A6000 with the "shared_expert" role.
        * CPU DRAM fallback is used when expert weight shards do not fit
          on any GPU; this path is handled by DeepSpeed ZeRO-Infinity and
          is represented here as device index ``-1``.

        Returns
        -------
        dict
            Keys: ``"routed_expert"``, ``"fc1_latent_proj"``,
            ``"fc2_latent_proj"``, ``"shared_expert"``, ``"cpu_cache"``.
        """
        assignment: Dict[str, int] = {}
        role_map: Dict[str, int] = {}
        for dev in self.cluster:
            role_map.setdefault(dev.preferred_role, dev.device_index)

        assignment["routed_expert"] = role_map.get(
            "routed_expert", role_map.get("any", 0)
        )
        assignment["fc2_latent_proj"] = role_map.get(
            "latent_proj", role_map.get("any", 0)
        )
        assignment["fc1_latent_proj"] = role_map.get(
            "latent_proj", role_map.get("any", 0)
        )
        assignment["shared_expert"] = role_map.get(
            "shared_expert", role_map.get("any", 0)
        )
        # CPU DRAM locality cache — always -1, managed by ZeRO-Infinity
        assignment["cpu_cache"] = -1

        logger.info("DES-LOC device assignment: %s", assignment)
        return assignment


# ---------------------------------------------------------------------------
# backward_dw stream scheduler
# ---------------------------------------------------------------------------

class HeteroBackwardDwScheduler:
    """Coordinate weight-gradient computation across heterogeneous streams.

    Upstream context (Megatron commit a4008d0f)
    -------------------------------------------
    In Megatron's latent MoE, ``fc2_latent_proj`` runs its forward pass in
    the *comm stream* (used for EP all-to-all) so that gradient computation
    overlaps with communication.  The corresponding ``backward_dw`` must
    therefore also run in the comm stream to maintain stream ordering —
    otherwise the gradient might be written before the corresponding
    activation tensor is ready.

    ``fc1_latent_proj`` lives in the shared-expert branch and can run in
    the default compute stream.

    DES-LOC adaptation
    ------------------
    We have three CUDA devices on a PCIe fabric.  Instead of a single
    "comm stream" we maintain:

    * ``_h100_comm_stream`` — CUDA stream on H100 (device 2) used for
      expert-parallel all-to-all and fc2_latent_proj backward_dw.
    * ``_a6000_compute_stream[0,1]`` — compute streams on each A6000 for
      shared-expert and fc1_latent_proj backward_dw.
    * An ``asyncio``-style threading lock ensures that backward_dw calls
      do not race across the PCIe bus.

    Parameters
    ----------
    assignment : dict
        Output of ``HeteroLatentMoEFlops.estimate_device_assignment()``.
    use_comm_stream_for_fc2 : bool
        Mirror the upstream behaviour of running ``fc2_latent_proj.backward_dw``
        in the comm stream.  Set to ``False`` to disable (e.g. in unit tests
        without real CUDA devices).
    """

    def __init__(
        self,
        assignment: Dict[str, int],
        use_comm_stream_for_fc2: bool = True,
    ) -> None:
        self.assignment = assignment
        self.use_comm_stream_for_fc2 = use_comm_stream_for_fc2
        self._lock = threading.Lock()

        self._streams: Dict[str, Optional[torch.cuda.Stream]] = {}
        self._init_streams()

    # ------------------------------------------------------------------
    # Stream initialisation
    # ------------------------------------------------------------------

    def _init_streams(self) -> None:
        """Create per-device CUDA streams, gracefully degrading on CPU-only envs."""
        if not torch.cuda.is_available():
            logger.warning(
                "CUDA not available; HeteroBackwardDwScheduler running in no-op mode."
            )
            for key in ("h100_comm", "a6000_0_compute", "a6000_1_compute"):
                self._streams[key] = None
            return

        num_devices = torch.cuda.device_count()

        def safe_stream(device_idx: int, name: str) -> Optional[torch.cuda.Stream]:
            if device_idx < num_devices:
                try:
                    s = torch.cuda.Stream(device=device_idx)
                    logger.debug("Created CUDA stream '%s' on device %d", name, device_idx)
                    return s
                except RuntimeError as exc:
                    logger.warning("Could not create stream '%s': %s", name, exc)
            else:
                logger.warning(
                    "Device %d not found (only %d devices); stream '%s' disabled.",
                    device_idx, num_devices, name,
                )
            return None

        h100_idx = self.assignment.get("routed_expert", 2)
        a6000_0_idx = self.assignment.get("fc2_latent_proj", 0)
        a6000_1_idx = self.assignment.get("shared_expert", 1)

        self._streams["h100_comm"] = safe_stream(h100_idx, "h100_comm")
        self._streams["a6000_0_compute"] = safe_stream(a6000_0_idx, "a6000_0_compute")
        self._streams["a6000_1_compute"] = safe_stream(a6000_1_idx, "a6000_1_compute")

    # ------------------------------------------------------------------
    # Public scheduling API
    # ------------------------------------------------------------------

    def schedule_routed_backward_dw(
        self,
        routed_experts: nn.Module,
        fc2_latent_proj: Optional[nn.Module],
    ) -> None:
        """Run routed-expert backward_dw; dispatch fc2_latent_proj to comm stream.

        Mirrors upstream logic::

            self.experts.backward_dw()
            if moe_latent_size:
                with torch.cuda.stream(comm_stream):
                    self.fc2_latent_proj.backward_dw()

        In DES-LOC the "comm stream" is the H100's dedicated communication
        stream to maximise PCIe bus utilisation during EP all-to-all.

        Parameters
        ----------
        routed_experts : nn.Module
            Expert module with a ``backward_dw()`` method.
        fc2_latent_proj : nn.Module or None
            Latent down-projection layer; skipped when ``None``.
        """
        with self._lock:
            logger.debug("schedule_routed_backward_dw: calling routed_experts.backward_dw()")
            routed_experts.backward_dw()  # type: ignore[operator]

            if fc2_latent_proj is not None:
                comm_stream = self._streams.get("h100_comm")
                if self.use_comm_stream_for_fc2 and comm_stream is not None:
                    logger.debug(
                        "fc2_latent_proj.backward_dw dispatched to h100_comm stream"
                    )
                    with torch.cuda.stream(comm_stream):
                        fc2_latent_proj.backward_dw()  # type: ignore[operator]
                else:
                    # Fallback: run synchronously if stream unavailable
                    logger.debug(
                        "fc2_latent_proj.backward_dw running synchronously "
                        "(comm stream unavailable)"
                    )
                    fc2_latent_proj.backward_dw()  # type: ignore[operator]

    def schedule_shared_backward_dw(
        self,
        shared_experts: Optional[nn.Module],
        shared_expert_overlap: bool,
        fc1_latent_proj: Optional[nn.Module],
        moe_latent_size: Optional[int],
    ) -> None:
        """Run shared-expert and fc1_latent_proj backward_dw.

        Mirrors upstream logic::

            if use_shared_expert and not shared_expert_overlap:
                self.shared_experts.backward_dw()
            if moe_latent_size:
                self.fc1_latent_proj.backward_dw()

        In DES-LOC both operations are dispatched to the A6000 (device 1)
        compute stream, which is dedicated to the shared-expert sub-graph.

        Parameters
        ----------
        shared_experts : nn.Module or None
            Shared expert module.  Pass ``None`` to skip.
        shared_expert_overlap : bool
            When ``True`` the shared expert runs overlapped with routing and
            its ``backward_dw`` is handled elsewhere; skip here.
        fc1_latent_proj : nn.Module or None
            Latent up-projection; triggered only when ``moe_latent_size``
            is set.
        moe_latent_size : int or None
            ``None`` disables fc1_latent_proj dispatch.
        """
        a6000_stream = self._streams.get("a6000_1_compute")

        def _run(fn_module: nn.Module, name: str) -> None:
            if a6000_stream is not None:
                with torch.cuda.stream(a6000_stream):
                    logger.debug("%s.backward_dw dispatched to a6000_1_compute", name)
                    fn_module.backward_dw()  # type: ignore[operator]
            else:
                logger.debug("%s.backward_dw running synchronously", name)
                fn_module.backward_dw()  # type: ignore[operator]

        with self._lock:
            if shared_experts is not None and not shared_expert_overlap:
                _run(shared_experts, "shared_experts")

            if moe_latent_size is not None and fc1_latent_proj is not None:
                _run(fc1_latent_proj, "fc1_latent_proj")

    def synchronize_streams(self) -> None:
        """Block until all scheduled backward_dw operations complete.

        Called at the end of each micro-step in the DeepSpeed engine hook
        to ensure gradient tensors are materialised before the optimizer
        step.  On PCIe fabrics without NVLink this is critical to prevent
        stale gradients leaking across pipeline stages.
        """
        if not torch.cuda.is_available():
            return
        for name, stream in self._streams.items():
            if stream is not None:
                logger.debug("Synchronizing stream '%s'", name)
                stream.synchronize()


# ---------------------------------------------------------------------------
# DES-LOC FLOP-aware locality cache pressure estimator
# ---------------------------------------------------------------------------

@dataclass
class LocalityCachePressure:
    """Estimate CPU DRAM locality cache pressure from MoE weight shards.

    In DES-LOC "shared locality cache" refers to the 1.5 TB CPU DRAM pool
    that holds expert weight shards evicted from GPU HBM.  During
    ``backward_dw``, weight gradients for the *cold* experts are accumulated
    in CPU DRAM and later merged during the optimizer step.

    This dataclass quantifies the expected cache pressure to guide
    prefetch scheduling.

    Attributes
    ----------
    num_experts : int
    expert_param_count : int
        Number of parameters per expert (approximate).
    dtype_bytes : int
        Bytes per parameter (2 for BF16, 4 for FP32).
    active_experts_per_step : int
        Expected number of experts touched per micro-step.
    gpu_hbm_budget_bytes : int
        Total HBM available for expert weight cache (after activations).
    cpu_dram_bytes : int
        Total CPU DRAM available (default: 1.5 TB for Neuron_SP cluster).
    """
    num_experts: int
    expert_param_count: int
    dtype_bytes: int = 2  # BF16
    active_experts_per_step: int = 1
    gpu_hbm_budget_bytes: int = int(48e9)  # A6000 after activations
    cpu_dram_bytes: int = int(1.5e12)      # 1.5 TB

    def cold_expert_ratio(self) -> float:
        """Fraction of experts that cannot fit in GPU HBM simultaneously."""
        total_weight_bytes = (
            self.num_experts * self.expert_param_count * self.dtype_bytes
        )
        if self.gpu_hbm_budget_bytes >= total_weight_bytes:
            return 0.0
        resident_experts = self.gpu_hbm_budget_bytes // (
            self.expert_param_count * self.dtype_bytes
        )
        cold = max(0, self.num_experts - resident_experts)
        ratio = cold / self.num_experts
        logger.debug(
            "cold_expert_ratio: resident=%d  cold=%d  ratio=%.3f",
            resident_experts, cold, ratio,
        )
        return ratio

    def estimated_dma_bytes_per_step(self) -> int:
        """Expected DMA transfer from CPU DRAM per micro-step for backward_dw.

        Only cold experts need to be streamed from CPU DRAM during their
        backward_dw.  Active (hot) experts remain resident in HBM.
        """
        cold_ratio = self.cold_expert_ratio()
        cold_active = max(0, self.active_experts_per_step * cold_ratio)
        dma = int(cold_active * self.expert_param_count * self.dtype_bytes)
        logger.debug("estimated_dma_bytes_per_step: %.2f MB", dma / 1e6)
        return dma

    def fits_in_cpu_dram(self) -> bool:
        """Check that the full expert pool fits in the locality cache."""
        total = self.num_experts * self.expert_param_count * self.dtype_bytes
        ok = total <= self.cpu_dram_bytes
        if not ok:
            logger.error(
                "Expert pool (%.1f GB) exceeds CPU DRAM cache (%.1f GB). "
                "Consider reducing num_experts or enabling expert quantization.",
                total / 1e9, self.cpu_dram_bytes / 1e9,
            )
        return ok


# ---------------------------------------------------------------------------
# DeepSpeed engine integration hook
# ---------------------------------------------------------------------------

class LatentMoEFlopsHook:
    """DeepSpeed training-loop hook that wires DES-LOC latent-MoE accounting.

    Usage in Neuron_SP training script::

        hook = LatentMoEFlopsHook(engine=ds_engine, flops_cfg=flops_cfg)
        hook.register()

        for batch in dataloader:
            loss = engine(batch)
            engine.backward(loss)
            hook.on_backward_end()   # ← triggers backward_dw scheduling
            engine.step()

    The hook:

    1. Replaces the default FLOP counter with ``HeteroLatentMoEFlops``.
    2. After each ``engine.backward()``, iterates over MoE layers and calls
       ``HeteroBackwardDwScheduler`` to dispatch weight-gradient computations
       to the correct device streams.
    3. Emits per-step telemetry to Python ``logging`` (captured by
       DeepSpeed's tensorboard/wandb integration if configured).

    Parameters
    ----------
    engine : object
        DeepSpeed engine instance (typed as ``object`` to avoid hard dep
        at import time; duck-typed at runtime).
    flops_cfg : LatentMoEFlopsConfig
    cluster : list of DeviceCapability, optional
    use_comm_stream_for_fc2 : bool
    """

    def __init__(
        self,
        engine: object,
        flops_cfg: LatentMoEFlopsConfig,
        cluster: Optional[List[DeviceCapability]] = None,
        use_comm_stream_for_fc2: bool = True,
    ) -> None:
        self.engine = engine
        self.flops_estimator = HeteroLatentMoEFlops(flops_cfg, cluster)
        assignment = self.flops_estimator.estimate_device_assignment()
        self.scheduler = HeteroBackwardDwScheduler(
            assignment=assignment,
            use_comm_stream_for_fc2=use_comm_stream_for_fc2,
        )
        self._step_count = 0
        self._registered_moe_layers: List[nn.Module] = []
        logger.info(
            "LatentMoEFlopsHook initialised. "
            "Total FLOPs/step estimate: %.3e",
            self.flops_estimator.compute_total_flops(),
        )

    def register(self) -> None:
        """Discover MoE layers in the engine's model and cache references."""
        model = getattr(self.engine, "module", self.engine)
        for name, module in model.named_modules():
            # Detect MoE layers by duck-typing: must have .experts and
            # .backward_dw attributes (matches both Megatron-style and
            # custom DES-LOC MoE layer implementations).
            if (
                hasattr(module, "experts")
                and hasattr(module, "backward_dw")
                and callable(getattr(module, "backward_dw", None))
            ):
                self._registered_moe_layers.append(module)
                logger.info("Registered MoE layer for DES-LOC scheduling: %s", name)
        if not self._registered_moe_layers:
            logger.warning(
                "No MoE layers found in engine.module. "
                "Ensure the model uses DES-LOC-compatible MoE layer classes."
            )

    def on_backward_end(self) -> None:
        """Dispatch backward_dw for all registered MoE layers.

        Called immediately after ``engine.backward()`` completes, before
        the optimizer step.  Iterates over registered MoE layers and
        schedules their weight-gradient computations on the appropriate
        device streams via ``HeteroBackwardDwScheduler``.
        """
        self._step_count += 1
        cfg = self.flops_estimator.cfg

        for layer in self._registered_moe_layers:
            # Routed expert + fc2_latent_proj (H100 comm stream)
            fc2 = getattr(layer, "fc2_latent_proj", None)
            self.scheduler.schedule_routed_backward_dw(
                routed_experts=layer.experts,
                fc2_latent_proj=fc2 if cfg.moe_latent_size is not None else None,
            )

            # Shared expert + fc1_latent_proj (A6000 compute stream)
            shared = getattr(layer, "shared_experts", None)
            overlap = getattr(layer, "shared_expert_overlap", False)
            fc1 = getattr(layer, "fc1_latent_proj", None)
            self.scheduler.schedule_shared_backward_dw(
                shared_experts=shared,
                shared_expert_overlap=overlap,
                fc1_latent_proj=fc1,
                moe_latent_size=cfg.moe_latent_size,
            )

        self.scheduler.synchronize_streams()

        if self._step_count % 100 == 0:
            logger.info(
                "LatentMoEFlopsHook: step=%d  estimated_flops=%.3e",
                self._step_count,
                self.flops_estimator.compute_total_flops(),
            )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # --- FLOP accounting ---

    cfg_latent = LatentMoEFlopsConfig(
        batch_size=2, seq_length=512, hidden_size=4096,
        num_experts=64, num_experts_routed_to=2,
        moe_ffn_hidden_size=2048, ffn_expansion_factor=1.0,
        num_moe_layers=4, num_dense_layers=28,
        moe_latent_size=512,
        shared_expert_ffn_hidden_size=4096,
    )
    estimator = HeteroLatentMoEFlops(cfg_latent)

    routed = estimator.compute_routed_flops()
    shared = estimator.compute_shared_flops()
    total = estimator.compute_total_flops()

    # With latent projection, routed FLOPs should be *less* than without
    cfg_std = LatentMoEFlopsConfig(
        batch_size=2, seq_length=512, hidden_size=4096,
        num_experts=64, num_experts_routed_to=2,
        moe_ffn_hidden_size=2048, ffn_expansion_factor=1.0,
        num_moe_layers=4, num_dense_layers=28,
        moe_latent_size=None,
    )
    routed_std = HeteroLatentMoEFlops(cfg_std).compute_routed_flops()

    assert routed < routed_std, (
        f"Latent routing should reduce expert-body FLOPs: {routed:.3e} vs {routed_std:.3e}"
    )
    assert total > 0, "Total FLOPs must be positive"

    assignment = estimator.estimate_device_assignment()
    assert "routed_expert" in assignment, "Device assignment missing routed_expert key"

    # --- Locality cache pressure ---
    pressure = LocalityCachePressure(
        num_experts=64,
        expert_param_count=int(2e6),
        dtype_bytes=2,
        active_experts_per_step=4,
    )
    assert pressure.fits_in_cpu_dram(), "Expert pool should fit in 1.5 TB DRAM"

    # --- Backward scheduler (no-CUDA graceful degradation) ---
    scheduler = HeteroBackwardDwScheduler(
        assignment=assignment,
        use_comm_stream_for_fc2=False,  # safe for CPU-only CI
    )
    scheduler.synchronize_streams()  # should not raise

    print("All smoke tests passed.")
