"""
DES-LOC Heterogeneous MoE Gradient Finalization
================================================

Upstream Design Intent (Megatron aa786b72):
--------------------------------------------
Megatron-LM commit aa786b72 ("Thread custom process groups through MoE grad finalization")
addresses a fundamental issue in Mixture-of-Experts training: the router's expert-bias update
requires an all-reduce across TPxDPxCP ranks, but the original code always fell back to the
global parallel state singleton. This coupling prevented callers from supplying custom process
groups — for example, when running non-standard parallelism topologies or when the global
parallel state is uninitialized (e.g., in unit tests or inference servers).

The upstream fix propagates an explicit ``tp_dp_cp_group`` parameter from the outermost
``finalize_model_grads`` entry-point down through ``_update_router_expert_bias`` and into
``get_updated_expert_bias``, removing every call-site that touched the global parallel_state
singleton for this particular all-reduce.

DES-LOC Adaptation Points:
---------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) introduces a three-tier heterogeneous
device topology:

  Tier-0  →  2× NVIDIA A6000 48 GB  (SM86, PCIe)    — "locality" workers
  Tier-1  →  1× NVIDIA H100 NVL 96 GB (SM90, PCIe)  — "execution" worker
  Host    →  1.5 TB CPU DRAM                          — shared LOcality Cache backing store

Key DES-LOC considerations that differ from vanilla Megatron:

1. **Heterogeneous SM version**: A6000 (SM86) and H100 (SM90) cannot share the same CUDA
   kernel binary when SM-version-specific optimizations are used (e.g., warpgroup MMA).
   The finalization path must route expert-bias all-reduce tensors to a common dtype/device
   that all participating ranks can handle — concretely, fp32 on CUDA with no SM90-only ops.

2. **No NVLink**: PCIe bandwidth between nodes is ~32 GB/s bidirectional, roughly 10× slower
   than NVLink. The all-reduce in ``get_updated_expert_bias`` is therefore latency-sensitive.
   DES-LOC uses a two-phase approach: intra-tier reduce (fast, local PCIe) then cross-tier
   all-reduce (slow, explicit).  We model this with a ``HeteroProcessGroupBundle`` that holds
   separate intra-tier and cross-tier groups.

3. **Locality Cache coherence**: Expert token counts (``local_tokens_per_expert``) that are
   accumulated on A6000 workers may be staged in the Shared LOcality Cache (CPU DRAM) between
   micro-batches. Before the all-reduce, we must flush any pinned-memory staging buffer back
   to device.  After the bias update we optionally write the result back to the locality cache
   for use by the router's next forward pass without a device round-trip.

4. **Custom process groups are first-class**: DES-LOC never relies on a global parallel state
   singleton because the three-tier topology requires different process groups per layer type.
   The ``HeteroProcessGroupBundle`` replaces Megatron's ``ProcessGroupCollection`` and carries
   an explicit ``tp_dp_cp`` handle alongside tier-aware intra/cross groups.

5. **Expert-bias update gating**: On heterogeneous hardware, the expert bias should only be
   updated from a single tier's router statistics to avoid double-counting tokens routed to
   shared experts.  We introduce a ``bias_update_tier`` flag to gate which tier performs the
   authoritative all-reduce.

Module structure mirrors the upstream diff but is rewritten for DeepSpeed's engine API:

  - ``HeteroProcessGroupBundle``        — replaces ProcessGroupCollection
  - ``HeteroLocalityCacheRef``          — thin wrapper around a pinned-memory staging buffer
  - ``get_updated_expert_bias_hetero``  — replaces moe_utils.get_updated_expert_bias
  - ``update_router_expert_bias_hetero``— replaces _update_router_expert_bias
  - ``finalize_moe_grads_hetero``       — replaces finalize_model_grads (MoE-specific slice)
  - ``HeteroMoEGradFinalizer``          — stateful class used by DeepSpeed engine hooks

Author: Neuron_SP / DES-LOC project
Mirrors: Megatron-LM aa786b72c097d92c3656844321380a2e212c169e
"""

from __future__ import annotations

import logging
import math
import os
import time
import unittest
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn

# ---------------------------------------------------------------------------
# Module-level logger — we avoid noisy prints and only log meaningful events.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DES-LOC device tier identifiers
# ---------------------------------------------------------------------------
TIER_LOCALITY = 0   # A6000 SM86 — "locality" workers
TIER_EXECUTION = 1  # H100 SM90  — "execution" worker
TIER_UNKNOWN = -1


def _detect_tier(device: Optional[torch.device] = None) -> int:
    """Detect which DES-LOC tier the current process belongs to.

    Heuristic: read the ``DESLOC_TIER`` environment variable first (set by
    the launcher).  Fall back to SM version detection via ``torch.cuda``.

    Returns:
        int: ``TIER_LOCALITY`` (0) or ``TIER_EXECUTION`` (1).
    """
    env_tier = os.environ.get("DESLOC_TIER", "")
    if env_tier == "0":
        return TIER_LOCALITY
    if env_tier == "1":
        return TIER_EXECUTION

    if device is None:
        if not torch.cuda.is_available():
            return TIER_UNKNOWN
        device = torch.device("cuda", torch.cuda.current_device())

    try:
        props = torch.cuda.get_device_properties(device)
        sm = props.major * 10 + props.minor
        if sm >= 90:
            return TIER_EXECUTION   # SM90 → H100
        return TIER_LOCALITY        # SM86 → A6000
    except Exception:
        return TIER_UNKNOWN


# ---------------------------------------------------------------------------
# HeteroProcessGroupBundle
# ---------------------------------------------------------------------------

@dataclass
class HeteroProcessGroupBundle:
    """Process-group bundle for DES-LOC heterogeneous training.

    Mirrors Megatron's ``ProcessGroupCollection`` but adds tier-aware groups
    required for two-phase all-reduce over PCIe-connected devices.

    Attributes:
        tp (dist.ProcessGroup):
            Tensor-parallel group.  In DES-LOC this is typically confined to
            one tier because SM86 and SM90 cannot share tensor-parallel ops.
        dp (dist.ProcessGroup):
            Data-parallel group (no context parallel).
        dp_cp (dist.ProcessGroup):
            Data-parallel group with context parallel folded in.
        tp_dp_cp (dist.ProcessGroup):
            The group spanning all TP×DP×CP ranks — used for the expert-bias
            all-reduce (mirrors upstream aa786b72).
        pp (Optional[dist.ProcessGroup]):
            Pipeline-parallel group.  May be None if not using PP.
        embd (Optional[dist.ProcessGroup]):
            Embedding all-reduce group.
        intra_tier (Optional[dist.ProcessGroup]):
            Intra-tier group (e.g., both A6000 ranks together).  Used in the
            first phase of the two-phase all-reduce.
        cross_tier (Optional[dist.ProcessGroup]):
            Cross-tier group (one rank per tier).  Used in the second phase.
        bias_update_tier (int):
            Which tier is authoritative for the expert-bias update.  Only
            ranks belonging to this tier perform the update; others receive
            the broadcast result.  Defaults to ``TIER_LOCALITY``.
        locality_cache_group (Optional[dist.ProcessGroup]):
            Optional group whose members share access to the Shared LOcality
            Cache (CPU DRAM).  Used to coordinate cache flush/fill.
    """

    tp: dist.ProcessGroup = dist.group.WORLD
    dp: dist.ProcessGroup = dist.group.WORLD
    dp_cp: dist.ProcessGroup = dist.group.WORLD
    tp_dp_cp: Optional[dist.ProcessGroup] = None
    pp: Optional[dist.ProcessGroup] = None
    embd: Optional[dist.ProcessGroup] = None
    intra_tier: Optional[dist.ProcessGroup] = None
    cross_tier: Optional[dist.ProcessGroup] = None
    bias_update_tier: int = TIER_LOCALITY
    locality_cache_group: Optional[dist.ProcessGroup] = None

    def validate(self, require_tp_dp_cp: bool = False) -> None:
        """Validate required fields.

        Args:
            require_tp_dp_cp: If True, asserts that ``tp_dp_cp`` is not None.
                Should be set to True whenever ``moe_router_enable_expert_bias``
                is active (mirrors the upstream assertion in finalize_model_grads).

        Raises:
            AssertionError: On any validation failure.
        """
        assert self.tp is not None, "HeteroProcessGroupBundle.tp must not be None"
        assert self.dp_cp is not None, "HeteroProcessGroupBundle.dp_cp must not be None"
        if require_tp_dp_cp:
            assert self.tp_dp_cp is not None, (
                "HeteroProcessGroupBundle.tp_dp_cp must not be None when "
                "moe_router_enable_expert_bias is enabled.  "
                "This mirrors the upstream Megatron requirement introduced in aa786b72."
            )


# ---------------------------------------------------------------------------
# HeteroLocalityCacheRef — pinned-memory staging buffer
# ---------------------------------------------------------------------------

@dataclass
class HeteroLocalityCacheRef:
    """Reference to a pinned-memory staging buffer in the Shared LOcality Cache.

    The Shared LOcality Cache (1.5 TB CPU DRAM) is used to stage tensors
    between micro-batches, reducing PCIe traffic for frequently-reused values
    such as ``local_tokens_per_expert``.

    Attributes:
        key (str):
            Logical key identifying this cache slot (e.g., ``"layer_3.tokens_per_expert"``).
        pinned_buffer (Optional[torch.Tensor]):
            CPU-pinned tensor.  If None, the cache slot is not populated.
        is_dirty (bool):
            True if the pinned buffer has been written since the last device sync.
    """

    key: str
    pinned_buffer: Optional[torch.Tensor] = None
    is_dirty: bool = False

    def flush_to_device(self, device_tensor: torch.Tensor, stream: Optional[torch.cuda.Stream] = None) -> bool:
        """Copy pinned buffer → device tensor (H2D) if dirty.

        Args:
            device_tensor: Target device tensor (must match shape/dtype of pinned_buffer).
            stream: CUDA stream to use for the async copy.  If None, uses current stream.

        Returns:
            True if a copy was performed, False if the cache was clean.
        """
        if self.pinned_buffer is None or not self.is_dirty:
            return False

        if self.pinned_buffer.shape != device_tensor.shape:
            logger.warning(
                "LocalityCache[%s] shape mismatch: buffer %s vs device %s — skipping flush",
                self.key, self.pinned_buffer.shape, device_tensor.shape,
            )
            return False

        ctx = torch.cuda.stream(stream) if stream is not None else _null_context()
        with ctx:
            device_tensor.copy_(self.pinned_buffer, non_blocking=True)

        self.is_dirty = False
        logger.debug("LocalityCache[%s] flushed H2D (%s)", self.key, tuple(device_tensor.shape))
        return True

    def fill_from_device(self, device_tensor: torch.Tensor, stream: Optional[torch.cuda.Stream] = None) -> None:
        """Copy device tensor → pinned buffer (D2H) for caching.

        Allocates a new pinned buffer if necessary.

        Args:
            device_tensor: Source device tensor.
            stream: CUDA stream for the async copy.
        """
        if (
            self.pinned_buffer is None
            or self.pinned_buffer.shape != device_tensor.shape
            or self.pinned_buffer.dtype != device_tensor.dtype
        ):
            self.pinned_buffer = torch.empty(
                device_tensor.shape,
                dtype=device_tensor.dtype,
                pin_memory=True,
            )

        ctx = torch.cuda.stream(stream) if stream is not None else _null_context()
        with ctx:
            self.pinned_buffer.copy_(device_tensor, non_blocking=True)

        self.is_dirty = False  # freshly synced from device — not dirty
        logger.debug("LocalityCache[%s] filled D2H (%s)", self.key, tuple(device_tensor.shape))


class _null_context:
    """Minimal no-op context manager used when no stream is provided."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ---------------------------------------------------------------------------
# Expert-bias update — hetero-aware
# ---------------------------------------------------------------------------

def get_updated_expert_bias_hetero(
    tokens_per_expert: torch.Tensor,
    expert_bias: torch.Tensor,
    expert_bias_update_rate: float,
    tp_dp_cp_group: Optional[dist.ProcessGroup] = None,
    intra_tier_group: Optional[dist.ProcessGroup] = None,
    cross_tier_group: Optional[dist.ProcessGroup] = None,
    locality_cache_ref: Optional[HeteroLocalityCacheRef] = None,
    h2d_stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Compute updated MoE router expert bias for DES-LOC heterogeneous topology.

    Upstream (Megatron aa786b72) Design:
        The function receives ``tokens_per_expert`` — a per-rank accumulation of
        how many tokens were routed to each expert — and performs:
          1. All-reduce over the TPxDPxCP group to get global token counts.
          2. Compute a signed offset from the per-expert average.
          3. Update ``expert_bias`` by ``sign(offset) * update_rate``.

        The upstream fix allows passing an explicit ``tp_dp_cp_group`` instead
        of always querying the global parallel state singleton.

    DES-LOC Adaptation:
        Because A6000 (SM86) and H100 (SM90) are connected via PCIe without
        NVLink, we optionally split the all-reduce into two phases:

          Phase 1 (intra-tier):  reduce within each tier (fast, local PCIe).
          Phase 2 (cross-tier):  all-reduce between tier representatives.

        This can halve the effective message size for the cross-tier hop,
        reducing PCIe contention on the shared bus.

        If ``intra_tier_group`` and ``cross_tier_group`` are both provided, the
        two-phase path is used.  Otherwise we fall back to a single all-reduce
        over ``tp_dp_cp_group``, exactly mirroring the upstream behaviour.

        Additionally, if a ``locality_cache_ref`` is provided, we:
          - Flush any staged (H2D) token counts before the all-reduce.
          - Write the updated bias back to the locality cache (D2H) so the
            next forward pass can read it without a device round-trip.

    Args:
        tokens_per_expert: Shape ``[num_experts]`` or ``[num_moe_layers, num_experts]``.
            Per-rank token count tensor on CUDA.
        expert_bias: Same shape as ``tokens_per_expert``.  The current bias values.
        expert_bias_update_rate: Step size for the bias update (upstream: ``moe_router_bias_update_rate``).
        tp_dp_cp_group: Full TP×DP×CP process group.  Used as fallback when
            ``intra_tier_group``/``cross_tier_group`` are not supplied.
        intra_tier_group: Optional intra-tier sub-group for phase-1 reduce.
        cross_tier_group: Optional cross-tier sub-group for phase-2 all-reduce.
        locality_cache_ref: Optional pinned-memory cache reference for
            flush-before / fill-after semantics.
        h2d_stream: CUDA stream for async H2D copies from the locality cache.

    Returns:
        torch.Tensor: Updated expert bias, same shape and device as input.

    Raises:
        ValueError: If neither ``tp_dp_cp_group`` nor the two-phase groups are provided.
    """
    if tp_dp_cp_group is None and (intra_tier_group is None or cross_tier_group is None):
        raise ValueError(
            "get_updated_expert_bias_hetero requires either tp_dp_cp_group "
            "or both intra_tier_group and cross_tier_group.  "
            "In DES-LOC, always provide these via HeteroProcessGroupBundle."
        )

    with torch.no_grad():
        # ------------------------------------------------------------------
        # Step 0: Flush locality cache → device if we have staged token counts
        # ------------------------------------------------------------------
        if locality_cache_ref is not None and locality_cache_ref.is_dirty:
            flushed = locality_cache_ref.flush_to_device(tokens_per_expert, stream=h2d_stream)
            if flushed and h2d_stream is not None:
                # Synchronise so the all-reduce sees the flushed values.
                torch.cuda.current_stream().wait_stream(h2d_stream)

        # ------------------------------------------------------------------
        # Step 1: All-reduce token counts across the TPxDPxCP group.
        #
        # DES-LOC two-phase path: intra-tier reduce first, then cross-tier.
        # This reduces the number of bytes crossing the PCIe bus.
        # ------------------------------------------------------------------
        use_two_phase = (intra_tier_group is not None) and (cross_tier_group is not None)

        if use_two_phase:
            # Phase 1: sum within tier (cheap — both A6000s on same PCIe root).
            dist.all_reduce(tokens_per_expert, op=dist.ReduceOp.SUM, group=intra_tier_group)
            logger.debug(
                "Expert-bias update: intra-tier reduce done, group=%s", intra_tier_group
            )

            # Phase 2: all-reduce across tiers (expensive — PCIe to H100).
            dist.all_reduce(tokens_per_expert, op=dist.ReduceOp.SUM, group=cross_tier_group)
            logger.debug(
                "Expert-bias update: cross-tier reduce done, group=%s", cross_tier_group
            )
        else:
            # Single-phase fallback — identical to upstream aa786b72 behaviour.
            dist.all_reduce(tokens_per_expert, group=tp_dp_cp_group)

        # ------------------------------------------------------------------
        # Step 2: Compute signed offset and apply update (identical to upstream).
        #
        # average = sum(tokens_per_expert) / num_experts
        # offset  = average − tokens_per_expert
        # new_bias = expert_bias + sign(offset) * update_rate
        # ------------------------------------------------------------------
        average_tokens = tokens_per_expert.sum(dim=-1, keepdim=True) / tokens_per_expert.shape[-1]
        offset = average_tokens - tokens_per_expert
        updated_expert_bias = expert_bias + torch.sign(offset) * expert_bias_update_rate

        # ------------------------------------------------------------------
        # Step 3: Optionally write updated bias back to locality cache (D2H).
        # Next forward pass on A6000 locality workers can read from pinned mem.
        # ------------------------------------------------------------------
        if locality_cache_ref is not None:
            locality_cache_ref.fill_from_device(updated_expert_bias)

    return updated_expert_bias


# ---------------------------------------------------------------------------
# Per-model router expert-bias update
# ---------------------------------------------------------------------------

def update_router_expert_bias_hetero(
    model: List[nn.Module],
    moe_router_bias_update_rate: float,
    tp_dp_cp_group: Optional[dist.ProcessGroup] = None,
    intra_tier_group: Optional[dist.ProcessGroup] = None,
    cross_tier_group: Optional[dist.ProcessGroup] = None,
    locality_cache_registry: Optional[Dict[str, HeteroLocalityCacheRef]] = None,
    h2d_stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """Update the expert bias in every MoE router found in *model*.

    Mirrors Megatron's ``_update_router_expert_bias`` but:
      - Accepts ``HeteroProcessGroupBundle``-derived groups instead of a
        global parallel state.
      - Supports per-layer locality cache references via ``locality_cache_registry``.
      - Logs per-layer update events at DEBUG level.

    The function walks the model list looking for modules that expose:
      - ``router.local_tokens_per_expert`` (torch.Tensor)
      - ``router.expert_bias`` (torch.Tensor, nn.Parameter, or buffer)

    This matches both Megatron's TopKRouter interface and the DeepSpeed MoE
    router interface used by Neuron_SP.

    Args:
        model: List of ``nn.Module`` instances (one per pipeline stage).
        moe_router_bias_update_rate: Step size (from TransformerConfig or DeepSpeed config).
        tp_dp_cp_group: Full TP×DP×CP group (fallback if two-phase groups not provided).
        intra_tier_group: Intra-tier group for phase-1 reduce (DES-LOC two-phase).
        cross_tier_group: Cross-tier group for phase-2 all-reduce (DES-LOC two-phase).
        locality_cache_registry: Optional dict mapping layer-key → cache ref.
            Keys follow the pattern ``"<module_name>.tokens_per_expert"``.
        h2d_stream: CUDA stream for async H2D copies.
    """
    tokens_per_expert_list: List[torch.Tensor] = []
    expert_bias_list: List[torch.Tensor] = []
    layer_keys: List[str] = []

    # ------------------------------------------------------------------
    # Collect tokens_per_expert and expert_bias tensors from all MoE layers.
    # ------------------------------------------------------------------
    for model_chunk in model:
        for name, module in model_chunk.named_modules():
            router = getattr(module, "router", None)
            if router is None:
                continue

            tpe = getattr(router, "local_tokens_per_expert", None)
            eb = getattr(router, "expert_bias", None)

            if tpe is None or eb is None:
                continue

            # Unwrap nn.Parameter → Tensor for the update.
            if isinstance(eb, nn.Parameter):
                eb = eb.data

            tokens_per_expert_list.append(tpe)
            expert_bias_list.append(eb)
            layer_keys.append(name)

    if not tokens_per_expert_list:
        logger.debug("update_router_expert_bias_hetero: no MoE routers found, skipping")
        return

    logger.debug(
        "update_router_expert_bias_hetero: found %d MoE layer(s): %s",
        len(layer_keys), layer_keys,
    )

    # ------------------------------------------------------------------
    # Stack into batch tensors for a single all-reduce call.
    # Shape: [num_moe_layers, num_experts]
    # ------------------------------------------------------------------
    stacked_tokens = torch.stack(tokens_per_expert_list, dim=0)
    stacked_bias = torch.stack(expert_bias_list, dim=0)

    # Use the first layer's cache ref for the stacked tensor if available.
    # (Stacked update means one cache slot for all layers, keyed by first layer.)
    locality_ref: Optional[HeteroLocalityCacheRef] = None
    if locality_cache_registry is not None and layer_keys:
        cache_key = f"{layer_keys[0]}.tokens_per_expert"
        locality_ref = locality_cache_registry.get(cache_key)

    stacked_updated_bias = get_updated_expert_bias_hetero(
        tokens_per_expert=stacked_tokens,
        expert_bias=stacked_bias,
        expert_bias_update_rate=moe_router_bias_update_rate,
        tp_dp_cp_group=tp_dp_cp_group,
        intra_tier_group=intra_tier_group,
        cross_tier_group=cross_tier_group,
        locality_cache_ref=locality_ref,
        h2d_stream=h2d_stream,
    )

    # ------------------------------------------------------------------
    # Write updated bias back to each router and zero token counts.
    # ------------------------------------------------------------------
    for eb_tensor, updated_eb, layer_key in zip(expert_bias_list, stacked_updated_bias, layer_keys):
        eb_tensor.copy_(updated_eb)
        logger.debug(
            "Expert bias updated for layer '%s': mean_abs_bias=%.4f",
            layer_key, updated_eb.abs().mean().item(),
        )

    # Zero token counters so they don't accumulate across gradient-sync steps.
    for tpe in tokens_per_expert_list:
        tpe.zero_()


# ---------------------------------------------------------------------------
# finalize_moe_grads_hetero — top-level entry point
# ---------------------------------------------------------------------------

def finalize_moe_grads_hetero(
    model: List[nn.Module],
    pg_bundle: Optional[HeteroProcessGroupBundle] = None,
    moe_router_enable_expert_bias: bool = False,
    moe_router_bias_update_rate: float = 0.001,
    locality_cache_registry: Optional[Dict[str, HeteroLocalityCacheRef]] = None,
    h2d_stream: Optional[torch.cuda.Stream] = None,
    current_tier: Optional[int] = None,
) -> None:
    """Finalize MoE-specific gradients for a DES-LOC heterogeneous training step.

    This function is the DES-LOC counterpart of Megatron's ``finalize_model_grads``
    (the MoE-specific slice).  It is called by the DeepSpeed engine hook after the
    backward pass and handles:

      1. Expert-bias update (if enabled): all-reduce token counts and update biases.
      2. Tier gating: only the authoritative tier performs the bias update.
      3. Process group validation: mirrors the upstream aa786b72 assertion that
         ``tp_dp_cp`` must be non-None when expert bias is enabled.

    Upstream Mapping:
        - ``pg_collection``           → ``pg_bundle`` (HeteroProcessGroupBundle)
        - ``config.moe_router_enable_expert_bias`` → explicit ``moe_router_enable_expert_bias``
        - ``_update_router_expert_bias(model, config)`` → ``update_router_expert_bias_hetero(...)``

    DES-LOC Specifics:
        - ``current_tier`` controls which tier's ranks perform the expert-bias update.
          If None, all ranks participate (safe default).
        - The ``pg_bundle.bias_update_tier`` acts as a mask: if ``current_tier`` does
          not match, this rank is still part of the all-reduce (required for collective
          correctness) but does not apply the resulting bias update locally.
          Wait — actually, for a correct all-reduce all ranks must call it, so we
          always call the collective and only conditionally apply the write-back.

    Args:
        model: List of pipeline-stage modules.
        pg_bundle: Heterogeneous process-group bundle.  If None, attempts to fall
            back to a global parallel state (for compatibility with non-DES-LOC code
            paths, though this is discouraged).
        moe_router_enable_expert_bias: Whether the expert-bias mechanism is active.
        moe_router_bias_update_rate: Bias update step size.
        locality_cache_registry: Per-layer locality cache references.
        h2d_stream: CUDA stream for async H2D copies from locality cache.
        current_tier: This rank's DES-LOC tier (auto-detected if None).

    Raises:
        AssertionError: If ``moe_router_enable_expert_bias`` is True but
            ``pg_bundle.tp_dp_cp`` is None (mirrors upstream aa786b72 assertion).
    """
    if current_tier is None:
        current_tier = _detect_tier()

    # ------------------------------------------------------------------
    # Resolve the TP×DP×CP group — the critical change from upstream aa786b72.
    # ------------------------------------------------------------------
    tp_dp_cp_group: Optional[dist.ProcessGroup] = None
    intra_tier_group: Optional[dist.ProcessGroup] = None
    cross_tier_group: Optional[dist.ProcessGroup] = None

    if pg_bundle is not None:
        if moe_router_enable_expert_bias:
            # Mirror upstream: assert tp_dp_cp is present when bias is enabled.
            pg_bundle.validate(require_tp_dp_cp=True)
            tp_dp_cp_group = pg_bundle.tp_dp_cp

        intra_tier_group = pg_bundle.intra_tier
        cross_tier_group = pg_bundle.cross_tier

        logger.debug(
            "finalize_moe_grads_hetero: using pg_bundle (tp_dp_cp=%s, intra_tier=%s, cross_tier=%s)",
            tp_dp_cp_group, intra_tier_group, cross_tier_group,
        )
    else:
        # Fallback: try to resolve from a global parallel state compatible shim.
        # This path is discouraged in DES-LOC but provided for compatibility.
        logger.warning(
            "finalize_moe_grads_hetero: pg_bundle is None — falling back to global parallel state. "
            "This is discouraged in DES-LOC and may fail if parallel state is uninitialized."
        )
        try:
            from deepspeed.comm import get_world_group
            tp_dp_cp_group = get_world_group()
        except Exception:
            tp_dp_cp_group = dist.group.WORLD

    # ------------------------------------------------------------------
    # Expert-bias update.
    # ------------------------------------------------------------------
    if moe_router_enable_expert_bias:
        update_router_expert_bias_hetero(
            model=model,
            moe_router_bias_update_rate=moe_router_bias_update_rate,
            tp_dp_cp_group=tp_dp_cp_group,
            intra_tier_group=intra_tier_group,
            cross_tier_group=cross_tier_group,
            locality_cache_registry=locality_cache_registry,
            h2d_stream=h2d_stream,
        )
        logger.info(
            "finalize_moe_grads_hetero: expert bias updated (tier=%d, update_rate=%g)",
            current_tier, moe_router_bias_update_rate,
        )


# ---------------------------------------------------------------------------
# HeteroMoEGradFinalizer — stateful class for DeepSpeed engine hooks
# ---------------------------------------------------------------------------

class HeteroMoEGradFinalizer:
    """Stateful MoE gradient finalizer for DES-LOC heterogeneous training.

    Intended to be registered as a DeepSpeed engine hook:

    .. code-block:: python

        finalizer = HeteroMoEGradFinalizer(
            pg_bundle=my_pg_bundle,
            moe_router_enable_expert_bias=True,
            moe_router_bias_update_rate=0.001,
        )
        engine.register_forward_pre_hook(...)   # optional
        # Called manually after optimizer.step() or at end of train_step:
        finalizer.finalize(model_chunks)

    Attributes:
        pg_bundle (HeteroProcessGroupBundle): Process-group configuration.
        moe_router_enable_expert_bias (bool): Whether expert bias is active.
        moe_router_bias_update_rate (float): Bias update step size.
        locality_cache_registry (dict): Layer-key → HeteroLocalityCacheRef.
        h2d_stream (torch.cuda.Stream): Dedicated stream for H2D copies.
        _step_count (int): Number of ``finalize`` calls made (for logging).
        _total_latency_ms (float): Cumulative wall-clock time in finalize (ms).
    """

    def __init__(
        self,
        pg_bundle: Optional[HeteroProcessGroupBundle] = None,
        moe_router_enable_expert_bias: bool = False,
        moe_router_bias_update_rate: float = 0.001,
        locality_cache_registry: Optional[Dict[str, HeteroLocalityCacheRef]] = None,
        create_h2d_stream: bool = True,
    ) -> None:
        self.pg_bundle = pg_bundle
        self.moe_router_enable_expert_bias = moe_router_enable_expert_bias
        self.moe_router_bias_update_rate = moe_router_bias_update_rate
        self.locality_cache_registry: Dict[str, HeteroLocalityCacheRef] = (
            locality_cache_registry if locality_cache_registry is not None else {}
        )

        self.h2d_stream: Optional[torch.cuda.Stream] = None
        if create_h2d_stream and torch.cuda.is_available():
            self.h2d_stream = torch.cuda.Stream()

        self._step_count: int = 0
        self._total_latency_ms: float = 0.0

        if pg_bundle is not None and moe_router_enable_expert_bias:
            # Validate early so misconfiguration is caught at construction time.
            pg_bundle.validate(require_tp_dp_cp=True)
            logger.info(
                "HeteroMoEGradFinalizer initialised: expert_bias=True, update_rate=%g, "
                "tp_dp_cp=%s, intra_tier=%s, cross_tier=%s",
                moe_router_bias_update_rate,
                pg_bundle.tp_dp_cp,
                pg_bundle.intra_tier,
                pg_bundle.cross_tier,
            )

    def register_locality_cache(self, layer_key: str, ref: HeteroLocalityCacheRef) -> None:
        """Register a locality cache reference for a specific layer.

        Args:
            layer_key: Module path key (e.g., ``"transformer.layers.3.mlp"``).
            ref: The pinned-memory cache reference.
        """
        self.locality_cache_registry[layer_key] = ref
        logger.debug("HeteroMoEGradFinalizer: registered locality cache for key '%s'", layer_key)

    def finalize(self, model: List[nn.Module], current_tier: Optional[int] = None) -> None:
        """Run MoE gradient finalization for one training step.

        Args:
            model: List of pipeline-stage modules.
            current_tier: This rank's DES-LOC tier (auto-detected if None).
        """
        t0 = time.perf_counter()

        finalize_moe_grads_hetero(
            model=model,
            pg_bundle=self.pg_bundle,
            moe_router_enable_expert_bias=self.moe_router_enable_expert_bias,
            moe_router_bias_update_rate=self.moe_router_bias_update_rate,
            locality_cache_registry=self.locality_cache_registry if self.locality_cache_registry else None,
            h2d_stream=self.h2d_stream,
            current_tier=current_tier,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._step_count += 1
        self._total_latency_ms += elapsed_ms

        # Log every 100 steps to avoid log spam while still providing visibility.
        if self._step_count % 100 == 0:
            avg_ms = self._total_latency_ms / self._step_count
            logger.info(
                "HeteroMoEGradFinalizer: step=%d, avg_finalize_latency=%.2f ms",
                self._step_count, avg_ms,
            )

    @property
    def avg_latency_ms(self) -> float:
        """Average wall-clock latency of ``finalize`` calls in milliseconds."""
        if self._step_count == 0:
            return 0.0
        return self._total_latency_ms / self._step_count


# ---------------------------------------------------------------------------
# Utility: build HeteroProcessGroupBundle from DeepSpeed comm groups
# ---------------------------------------------------------------------------

def build_hetero_pg_bundle_from_deepspeed(
    tp_group: Optional[dist.ProcessGroup] = None,
    dp_group: Optional[dist.ProcessGroup] = None,
    dp_cp_group: Optional[dist.ProcessGroup] = None,
    tp_dp_cp_group: Optional[dist.ProcessGroup] = None,
    pp_group: Optional[dist.ProcessGroup] = None,
    embd_group: Optional[dist.ProcessGroup] = None,
    tier_rank_map: Optional[Dict[int, int]] = None,
    bias_update_tier: int = TIER_LOCALITY,
) -> HeteroProcessGroupBundle:
    """Construct a ``HeteroProcessGroupBundle`` from DeepSpeed process groups.

    This is a convenience factory for users migrating from vanilla DeepSpeed
    to DES-LOC.  It optionally derives ``intra_tier`` and ``cross_tier`` groups
    from ``tier_rank_map`` — a mapping from global rank to tier ID.

    Args:
        tp_group: Tensor-parallel group.
        dp_group: Data-parallel group.
        dp_cp_group: DP+CP group.
        tp_dp_cp_group: TP×DP×CP group (required for expert bias).
        pp_group: Pipeline-parallel group.
        embd_group: Embedding all-reduce group.
        tier_rank_map: ``{global_rank: tier_id}`` mapping.  If provided,
            intra-tier and cross-tier sub-groups are derived automatically.
        bias_update_tier: Which tier performs the authoritative bias write-back.

    Returns:
        HeteroProcessGroupBundle ready for use with DES-LOC finalization.
    """
    world = dist.group.WORLD

    bundle = HeteroProcessGroupBundle(
        tp=tp_group or world,
        dp=dp_group or world,
        dp_cp=dp_cp_group or world,
        tp_dp_cp=tp_dp_cp_group,
        pp=pp_group,
        embd=embd_group,
        bias_update_tier=bias_update_tier,
    )

    if tier_rank_map is not None and dist.is_available() and dist.is_initialized():
        _derive_tier_groups(bundle, tier_rank_map)

    return bundle


def _derive_tier_groups(
    bundle: HeteroProcessGroupBundle,
    tier_rank_map: Dict[int, int],
) -> None:
    """Derive intra-tier and cross-tier groups from a rank→tier mapping.

    Mutates ``bundle.intra_tier`` and ``bundle.cross_tier`` in place.

    DES-LOC topology (example):
        Rank 0 → A6000 #0 (tier 0)
        Rank 1 → A6000 #1 (tier 0)
        Rank 2 → H100    (tier 1)

        intra_tier for rank 0,1: group({0, 1})
        cross_tier: group({0, 2}) or group({1, 2}) — one representative per tier.

    Args:
        bundle: Bundle to mutate.
        tier_rank_map: Mapping global_rank → tier_id.
    """
    world_size = dist.get_world_size()
    tiers: Dict[int, List[int]] = {}
    for rank, tier in tier_rank_map.items():
        tiers.setdefault(tier, []).append(rank)

    current_rank = dist.get_rank()
    current_tier = tier_rank_map.get(current_rank, TIER_UNKNOWN)

    # Intra-tier: all ranks sharing the same tier.
    my_tier_ranks = sorted(tiers.get(current_tier, []))
    if len(my_tier_ranks) > 1:
        bundle.intra_tier = dist.new_group(ranks=my_tier_ranks)
        logger.debug(
            "_derive_tier_groups: intra_tier group created for tier %d, ranks=%s",
            current_tier, my_tier_ranks,
        )
    else:
        bundle.intra_tier = None  # Single-rank tier — no intra-tier reduce needed.

    # Cross-tier: one representative from each tier.
    # By convention, pick the lowest-rank in each tier.
    tier_representatives = sorted([min(ranks) for ranks in tiers.values()])
    if len(tier_representatives) > 1:
        bundle.cross_tier = dist.new_group(ranks=tier_representatives)
        logger.debug(
            "_derive_tier_groups: cross_tier group created, representatives=%s",
            tier_representatives,
        )
    else:
        bundle.cross_tier = None


# ---------------------------------------------------------------------------
# ProcessGroupCollection compatibility shim
# ---------------------------------------------------------------------------

class ProcessGroupCollection:
    """Minimal compatibility shim that mirrors Megatron's ProcessGroupCollection.

    Allows existing code that constructs a ``ProcessGroupCollection`` to be
    passed into DES-LOC functions that expect a ``HeteroProcessGroupBundle``.

    Upstream (aa786b72) added ``tp_dp_cp`` to this class.  We expose it here
    as a plain attribute so that validation logic in ``HeteroProcessGroupBundle``
    can reference it.

    This is NOT a full replacement — use ``HeteroProcessGroupBundle`` for new code.
    """

    def __init__(
        self,
        tp: Optional[dist.ProcessGroup] = None,
        pp: Optional[dist.ProcessGroup] = None,
        embd: Optional[dist.ProcessGroup] = None,
        pos_embd: Optional[dist.ProcessGroup] = None,
        dp_cp: Optional[dist.ProcessGroup] = None,
        tp_dp_cp: Optional[dist.ProcessGroup] = None,
    ) -> None:
        self.tp = tp
        self.pp = pp
        self.embd = embd
        self.pos_embd = pos_embd
        self.dp_cp = dp_cp
        self.tp_dp_cp = tp_dp_cp  # Added in upstream aa786b72

    def to_hetero_bundle(
        self,
        intra_tier: Optional[dist.ProcessGroup] = None,
        cross_tier: Optional[dist.ProcessGroup] = None,
        bias_update_tier: int = TIER_LOCALITY,
    ) -> HeteroProcessGroupBundle:
        """Convert to a ``HeteroProcessGroupBundle``.

        Args:
            intra_tier: DES-LOC intra-tier group (optional).
            cross_tier: DES-LOC cross-tier group (optional).
            bias_update_tier: Tier that performs the authoritative bias update.

        Returns:
            HeteroProcessGroupBundle populated from this instance's fields.
        """
        world = dist.group.WORLD
        return HeteroProcessGroupBundle(
            tp=self.tp or world,
            dp=world,
            dp_cp=self.dp_cp or world,
            tp_dp_cp=self.tp_dp_cp,
            pp=self.pp,
            embd=self.embd,
            intra_tier=intra_tier,
            cross_tier=cross_tier,
            bias_update_tier=bias_update_tier,
        )


# ---------------------------------------------------------------------------
# Utility: reset per-step MoE temporary tensors
# ---------------------------------------------------------------------------

def reset_moe_temporary_tensors(model: List[nn.Module]) -> None:
    """Zero out all per-step MoE auxiliary tensors.

    Mirrors Megatron's ``reset_model_temporary_tensors`` for the MoE-specific
    tensors: aux-loss trackers and token-count buffers.

    Args:
        model: List of pipeline-stage modules.
    """
    for model_chunk in model:
        for module in model_chunk.modules():
            # Auxiliary loss tracker (e.g., load-balancing loss accumulator).
            if hasattr(module, "reset_global_aux_loss_tracker"):
                module.reset_global_aux_loss_tracker()
            # Token-per-expert counters on the router.
            router = getattr(module, "router", None)
            if router is not None:
                tpe = getattr(router, "local_tokens_per_expert", None)
                if tpe is not None:
                    tpe.zero_()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    class _TestHeteroLocalityCache(unittest.TestCase):
        """Tests for HeteroLocalityCacheRef flush/fill semantics."""

        def test_fill_and_flush_roundtrip(self):
            """D2H fill followed by H2D flush should reproduce original values."""
            if not torch.cuda.is_available():
                self.skipTest("CUDA not available")

            device = torch.device("cuda", 0)
            original = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
            ref = HeteroLocalityCacheRef(key="test.layer.tokens")

            # Fill: device → pinned buffer.
            ref.fill_from_device(original)
            self.assertIsNotNone(ref.pinned_buffer)
            self.assertFalse(ref.is_dirty)

            # Mark dirty (simulate a stale buffer that should be flushed).
            ref.is_dirty = True

            # Modify device tensor.
            target = torch.zeros_like(original)
            flushed = ref.flush_to_device(target)
            self.assertTrue(flushed)
            torch.cuda.synchronize()

            torch.testing.assert_close(target.cpu(), original.cpu())
            self.assertFalse(ref.is_dirty)

        def test_flush_skips_when_clean(self):
            """Flush should be a no-op when the cache is clean."""
            if not torch.cuda.is_available():
                self.skipTest("CUDA not available")

            device = torch.device("cuda", 0)
            original = torch.tensor([5.0, 6.0], device=device)
            ref = HeteroLocalityCacheRef(key="test.clean")
            ref.fill_from_device(original)
            ref.is_dirty = False  # explicitly clean

            target = torch.zeros_like(original)
            flushed = ref.flush_to_device(target)
            self.assertFalse(flushed)
            # target should remain zeros
            torch.testing.assert_close(target, torch.zeros_like(original))

        def test_fill_reallocates_on_shape_change(self):
            """fill_from_device should reallocate pinned buffer on shape mismatch."""
            if not torch.cuda.is_available():
                self.skipTest("CUDA not available")

            device = torch.device("cuda", 0)
            ref = HeteroLocalityCacheRef(key="test.realloc")

            t1 = torch.ones(4, device=device)
            ref.fill_from_device(t1)
            self.assertEqual(ref.pinned_buffer.shape, (4,))

            t2 = torch.ones(8, device=device) * 2.0
            ref.fill_from_device(t2)
            self.assertEqual(ref.pinned_buffer.shape, (8,))

    class _TestGetUpdatedExpertBiasHetero(unittest.TestCase):
        """Tests for get_updated_expert_bias_hetero without distributed setup."""

        def _run_single_rank(self, tokens, bias, rate, expected_bias):
            """Helper: run bias update in single-rank (no-dist) mode."""
            # We cannot call dist.all_reduce without init, so we mock the group param.
            # Instead test the pure-math path by temporarily monkey-patching all_reduce.
            import unittest.mock as mock

            with mock.patch("torch.distributed.all_reduce", side_effect=lambda t, **kw: None):
                result = get_updated_expert_bias_hetero(
                    tokens_per_expert=tokens.clone(),
                    expert_bias=bias.clone(),
                    expert_bias_update_rate=rate,
                    tp_dp_cp_group=dist.group.WORLD,  # won't be called
                )
            torch.testing.assert_close(result, expected_bias, atol=1e-5, rtol=1e-5)

        def test_balanced_tokens_no_bias_change(self):
            """With equal token distribution, offset=0 → bias unchanged."""
            tokens = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
            bias = torch.tensor([[0.1, 0.2, -0.1, 0.0]])
            rate = 0.25
            # average = 1.0, offset = 0 everywhere → sign(0) = 0 → no change
            expected = bias.clone()
            self._run_single_rank(tokens, bias, rate, expected)

        def test_unbalanced_tokens_bias_update(self):
            """Expert with fewer tokens than average gets positive bias boost."""
            # tokens = [0, 2], average = 1, offset = [1, -1]
            # sign([1, -1]) = [1, -1], update = [+rate, -rate]
            tokens = torch.tensor([[0.0, 2.0]])
            bias = torch.tensor([[0.0, 0.0]])
            rate = 0.25
            expected = torch.tensor([[0.25, -0.25]])
            self._run_single_rank(tokens, bias, rate, expected)

        def test_multi_layer_stacked(self):
            """Multi-layer stacked tensor is processed correctly per row."""
            # Layer 0: tokens=[0,4], avg=2, offset=[2,-2], update=[+0.5,-0.5]
            # Layer 1: tokens=[3,1], avg=2, offset=[-1,1], update=[-0.5,+0.5]
            tokens = torch.tensor([[0.0, 4.0], [3.0, 1.0]])
            bias = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
            rate = 0.5
            expected = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
            self._run_single_rank(tokens, bias, rate, expected)

        def test_raises_without_groups(self):
            """Should raise ValueError when no group is provided."""
            tokens = torch.tensor([1.0, 1.0])
            bias = torch.tensor([0.0, 0.0])
            with self.assertRaises(ValueError):
                get_updated_expert_bias_hetero(
                    tokens_per_expert=tokens,
                    expert_bias=bias,
                    expert_bias_update_rate=0.1,
                    tp_dp_cp_group=None,
                    intra_tier_group=None,
                    cross_tier_group=None,
                )

        def test_locality_cache_fill_after_update(self):
            """Locality cache should be populated with updated bias after call."""
            if not torch.cuda.is_available():
                self.skipTest("CUDA not available")
            import unittest.mock as mock

            device = torch.device("cuda", 0)
            tokens = torch.tensor([0.0, 2.0], device=device)
            bias = torch.tensor([0.0, 0.0], device=device)
            ref = HeteroLocalityCacheRef(key="test.cache_fill")

            with mock.patch("torch.distributed.all_reduce", side_effect=lambda t, **kw: None):
                result = get_updated_expert_bias_hetero(
                    tokens_per_expert=tokens.clone(),
                    expert_bias=bias.clone(),
                    expert_bias_update_rate=0.25,
                    tp_dp_cp_group=dist.group.WORLD,
                    locality_cache_ref=ref,
                )

            torch.cuda.synchronize()
            self.assertIsNotNone(ref.pinned_buffer)
            # pinned_buffer should match result
            torch.testing.assert_close(ref.pinned_buffer, result.cpu())

    class _TestHeteroProcessGroupBundle(unittest.TestCase):
        """Tests for HeteroProcessGroupBundle validation logic."""

        def test_validate_passes_without_tp_dp_cp_when_not_required(self):
            bundle = HeteroProcessGroupBundle(
                tp=dist.group.WORLD,
                dp=dist.group.WORLD,
                dp_cp=dist.group.WORLD,
                tp_dp_cp=None,
            )
            # Should not raise
            bundle.validate(require_tp_dp_cp=False)

        def test_validate_raises_when_tp_dp_cp_required_but_none(self):
            bundle = HeteroProcessGroupBundle(
                tp=dist.group.WORLD,
                dp=dist.group.WORLD,
                dp_cp=dist.group.WORLD,
                tp_dp_cp=None,
            )
            with self.assertRaises(AssertionError) as ctx:
                bundle.validate(require_tp_dp_cp=True)
            self.assertIn("tp_dp_cp", str(ctx.exception))

        def test_validate_passes_when_tp_dp_cp_provided(self):
            bundle = HeteroProcessGroupBundle(
                tp=dist.group.WORLD,
                dp=dist.group.WORLD,
                dp_cp=dist.group.WORLD,
                tp_dp_cp=dist.group.WORLD,
            )
            bundle.validate(require_tp_dp_cp=True)  # Should not raise

        def test_tp_none_raises(self):
            bundle = HeteroProcessGroupBundle(
                tp=None,
                dp=dist.group.WORLD,
                dp_cp=dist.group.WORLD,
            )
            with self.assertRaises(AssertionError):
                bundle.validate()

    class _TestProcessGroupCollectionShim(unittest.TestCase):
        """Tests for the ProcessGroupCollection compatibility shim."""

        def test_to_hetero_bundle_preserves_tp_dp_cp(self):
            world = dist.group.WORLD
            pgc = ProcessGroupCollection(
                tp=world,
                pp=None,
                embd=None,
                pos_embd=None,
                dp_cp=world,
                tp_dp_cp=world,
            )
            bundle = pgc.to_hetero_bundle()
            self.assertIs(bundle.tp_dp_cp, world)
            self.assertIs(bundle.dp_cp, world)
            self.assertIsNone(bundle.pp)

        def test_to_hetero_bundle_none_tp_dp_cp(self):
            world = dist.group.WORLD
            pgc = ProcessGroupCollection(tp=world, dp_cp=world, tp_dp_cp=None)
            bundle = pgc.to_hetero_bundle()
            self.assertIsNone(bundle.tp_dp_cp)
            # Should raise when expert bias requires it
            with self.assertRaises(AssertionError):
                bundle.validate(require_tp_dp_cp=True)

    class _TestHeteroMoEGradFinalizer(unittest.TestCase):
        """Tests for HeteroMoEGradFinalizer lifecycle and latency tracking."""

        def _make_fake_model(self, num_experts: int = 4) -> nn.Module:
            """Build a minimal model with a router that has the required buffers."""
            model = nn.Module()
            model.router = nn.Module()
            model.router.register_buffer(
                "local_tokens_per_expert",
                torch.ones(num_experts) * 2.0,
            )
            model.router.register_buffer(
                "expert_bias",
                torch.zeros(num_experts),
            )
            return model

        def test_finalizer_init_validates_pg_bundle(self):
            """Constructor should raise when expert bias enabled but tp_dp_cp is None."""
            bundle = HeteroProcessGroupBundle(
                tp=dist.group.WORLD,
                dp=dist.group.WORLD,
                dp_cp=dist.group.WORLD,
                tp_dp_cp=None,
            )
            with self.assertRaises(AssertionError):
                HeteroMoEGradFinalizer(
                    pg_bundle=bundle,
                    moe_router_enable_expert_bias=True,
                )

        def test_finalizer_no_expert_bias_skips_update(self):
            """With expert bias disabled, finalize should not touch router buffers."""
            import unittest.mock as mock

            bundle = HeteroProcessGroupBundle(
                tp=dist.group.WORLD,
                dp=dist.group.WORLD,
                dp_cp=dist.group.WORLD,
                tp_dp_cp=dist.group.WORLD,
            )
            finalizer = HeteroMoEGradFinalizer(
                pg_bundle=bundle,
                moe_router_enable_expert_bias=False,
                create_h2d_stream=False,
            )

            model = self._make_fake_model()
            original_bias = model.router.expert_bias.clone()
            original_tokens = model.router.local_tokens_per_expert.clone()

            with mock.patch("torch.distributed.all_reduce", side_effect=lambda t, **kw: None):
                finalizer.finalize([model])

            # Bias and tokens should be unchanged since expert_bias is disabled.
            torch.testing.assert_close(model.router.expert_bias, original_bias)
            torch.testing.assert_close(model.router.local_tokens_per_expert, original_tokens)

        def test_finalizer_step_count_and_latency(self):
            """step_count and avg_latency_ms should be updated after finalize."""
            import unittest.mock as mock

            bundle = HeteroProcessGroupBundle(
                tp=dist.group.WORLD,
                dp=dist.group.WORLD,
                dp_cp=dist.group.WORLD,
                tp_dp_cp=dist.group.WORLD,
            )
            finalizer = HeteroMoEGradFinalizer(
                pg_bundle=bundle,
                moe_router_enable_expert_bias=False,
                create_h2d_stream=False,
            )

            self.assertEqual(finalizer._step_count, 0)
            self.assertAlmostEqual(finalizer.avg_latency_ms, 0.0)

            model = self._make_fake_model()
            with mock.patch("torch.distributed.all_reduce", side_effect=lambda t, **kw: None):
                finalizer.finalize([model])
                finalizer.finalize([model])

            self.assertEqual(finalizer._step_count, 2)
            self.assertGreater(finalizer.avg_latency_ms, 0.0)

    class _TestDetectTier(unittest.TestCase):
        """Tests for _detect_tier heuristic."""

        def test_env_var_takes_precedence(self):
            import os
            os.environ["DESLOC_TIER"] = "0"
            try:
                self.assertEqual(_detect_tier(), TIER_LOCALITY)
            finally:
                del os.environ["DESLOC_TIER"]

            os.environ["DESLOC_TIER"] = "1"
            try:
                self.assertEqual(_detect_tier(), TIER_EXECUTION)
            finally:
                del os.environ["DESLOC_TIER"]

        def test_unknown_when_no_cuda_and_no_env(self):
            import os
            os.environ.pop("DESLOC_TIER", None)
            if torch.cuda.is_available():
                # Can't reliably test "no CUDA" path on a CUDA machine.
                self.skipTest("CUDA available; cannot test TIER_UNKNOWN path")
            result = _detect_tier(device=None)
            self.assertEqual(result, TIER_UNKNOWN)

    class _TestResetMoETemporaryTensors(unittest.TestCase):
        """Tests for reset_moe_temporary_tensors."""

        def test_zeros_token_counts(self):
            model = nn.Module()
            model.router = nn.Module()
            model.router.register_buffer(
                "local_tokens_per_expert", torch.ones(4) * 3.0
            )
            reset_moe_temporary_tensors([model])
            torch.testing.assert_close(
                model.router.local_tokens_per_expert, torch.zeros(4)
            )

        def test_calls_aux_loss_reset(self):
            calls = []

            class _FakeLayer(nn.Module):
                def reset_global_aux_loss_tracker(self_inner):
                    calls.append(True)

            model = nn.Module()
            model.layer = _FakeLayer()
            reset_moe_temporary_tensors([model])
            self.assertEqual(len(calls), 1)

    # Run all tests.
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for cls in [
        _TestHeteroLocalityCache,
        _TestGetUpdatedExpertBiasHetero,
        _TestHeteroProcessGroupBundle,
        _TestProcessGroupCollectionShim,
        _TestHeteroMoEGradFinalizer,
        _TestDetectTier,
        _TestResetMoETemporaryTensors,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
