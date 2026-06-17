"""
deepspeed/moe/hetero_loss_grad_scale.py
========================================

DES-LOC Heterogeneous MoE Loss Gradient Scaling Adapter
--------------------------------------------------------

Upstream Design Intent (Megatron commit 9ef8a2a4208565d1d68728b864bcd873a7443f8a):
    The original Megatron-LM fix addresses a subtle gradient scaling bug in Mixture-of-Experts
    (MoE) auxiliary losses (aux_loss and z_loss) when Tensor Parallelism (TP) degree > 1 is
    combined with the ``--calculate-per-token-loss`` flag.

    Root Cause:
        Under ``calculate_per_token_loss``, ``finalize_model_grads`` divides every parameter
        gradient by ``total_global_tokens``. Router weights are marked
        ``sequence_parallel=True``, meaning each TP rank processes a local sequence shard and
        ``_allreduce_non_tensor_model_parallel_grads`` SUMS partial gradients across the TP group.
        This sum-then-divide interaction causes aux_loss and z_loss gradients to be
        under-scaled by a factor of ``tp_size * cp_size`` relative to the intended target of
        ``1 / (num_micro_batches * dp_size)``.

    Fix Derivation:
        total_global_tokens
            = num_micro_batches * dp_cp_size * loss_func_local_tokens
            = num_micro_batches * dp_cp_size * tp_size * num_local_tokens
            = num_micro_batches * dp_size * (num_local_tokens * tp_cp_group.size())

        Pre-multiplying aux_loss (and z_loss) by ``num_local_tokens * tp_cp_group.size()``
        cancels that same factor in ``total_global_tokens``, leaving the target scaling
        ``1 / (num_micro_batches * dp_size)`` on the loss gradient.

        The z_loss forward coefficient is additionally divided by ``tp_cp_group.size()``
        because z_loss is computed independently on each TP+CP rank's local logits and must
        be averaged (not summed) across the TP+CP group.

DES-LOC Adaptation Points:
    The DES-LOC (Decoupled Execution with Shared LOcality Cache) framework introduces
    heterogeneous hardware — 2x A6000 48GB (SM86, PCIe) + 1x H100 NVL 96GB (SM90, PCIe) —
    with 1.5TB CPU DRAM as a shared locality cache. This creates several new dimensions of
    complexity beyond what Megatron's uniform-hardware TP/CP group model assumes:

    1. **Heterogeneous TP-group composition**: A single TP group may span devices of different
       compute capability (SM86 vs SM90). The H100 NVL can sustain higher MFU on large
       expert activations, but both A6000 devices share a PCIe root complex and have
       asymmetric bandwidth to each other vs to the H100. The concept of a uniform
       ``tp_cp_group.size()`` must be replaced with a *weighted* group size that reflects
       actual token counts processed per device per step.

    2. **DES-LOC locality cache for aux_loss accumulation**: Under DES-LOC, router logits
       and expert assignments from previous micro-batches are cached in CPU DRAM and reused
       for load-balancing loss computation to amortize the cost of recomputation. The
       gradient scaling must account for the fact that some aux_loss contributions come from
       cached (stale) activations vs live (current) activations.

    3. **Decoupled execution phase alignment**: DES-LOC decouples the forward pass of router
       and expert sub-networks into separate phases that may execute on different devices.
       The gradient scaler must be phase-aware: the router (running on all three GPUs with
       sequence sharding) uses the heterogeneous group correction, while expert networks
       (potentially pinned to the H100 for heavy compute) use standard DP averaging.

    4. **PCIe-only interconnect**: Without NVLink, allreduce across the TP group is
       bandwidth-limited. The scaler is designed to minimize extra communication by computing
       the group-size correction factor locally from the device topology manifest rather than
       via a collective.

Hardware Topology (fixed for this deployment):
    - GPU 0: NVIDIA A6000 48GB, SM86, PCIe gen4 x16
    - GPU 1: NVIDIA A6000 48GB, SM86, PCIe gen4 x16
    - GPU 2: NVIDIA H100 NVL 96GB, SM90, PCIe gen4 x16
    - CPU DRAM: 1.5TB, used as DES-LOC locality cache
    - No NVLink / NVSwitch fabric

Author: Neuron_SP / dylanyunlon
Upstream: megatron/core/transformer/moe/router.py @ 9ef8a2a
"""

from __future__ import annotations

import logging
import math
import os
import time
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardware topology manifest
# ---------------------------------------------------------------------------


class DeviceClass(Enum):
    """Compute capability tier for heterogeneous grouping."""
    SM86_A6000 = auto()   # NVIDIA A6000 48GB
    SM90_H100NVL = auto() # NVIDIA H100 NVL 96GB


@dataclass
class DeviceDescriptor:
    """Per-device static properties used by DES-LOC topology logic."""
    local_rank: int
    device_class: DeviceClass
    vram_gb: float
    sm_count: int
    # Relative token-processing throughput normalised so that A6000 = 1.0.
    # Derived from roofline analysis on the router's linear projection.
    relative_throughput: float
    # PCIe bandwidth to CPU DRAM in GB/s (unidirectional peak).
    pcie_bw_to_cpu_gb: float


# Static manifest for the fixed 3-GPU cluster.
_DEVICE_MANIFEST: Dict[int, DeviceDescriptor] = {
    0: DeviceDescriptor(
        local_rank=0,
        device_class=DeviceClass.SM86_A6000,
        vram_gb=48.0,
        sm_count=84,
        relative_throughput=1.0,
        pcie_bw_to_cpu_gb=32.0,
    ),
    1: DeviceDescriptor(
        local_rank=1,
        device_class=DeviceClass.SM86_A6000,
        vram_gb=48.0,
        sm_count=84,
        relative_throughput=1.0,
        pcie_bw_to_cpu_gb=32.0,
    ),
    2: DeviceDescriptor(
        local_rank=2,
        device_class=DeviceClass.SM90_H100NVL,
        vram_gb=96.0,
        sm_count=132,
        relative_throughput=2.3,  # ~2.3x router throughput vs A6000 at BF16
        pcie_bw_to_cpu_gb=64.0,
    ),
}


def get_device_descriptor(local_rank: Optional[int] = None) -> DeviceDescriptor:
    """Return the :class:`DeviceDescriptor` for *local_rank*.

    Falls back to the current CUDA device if *local_rank* is ``None``.
    Raises ``KeyError`` for unknown ranks so misconfiguration is caught early.
    """
    if local_rank is None:
        local_rank = torch.cuda.current_device()
    if local_rank not in _DEVICE_MANIFEST:
        raise KeyError(
            f"DES-LOC topology manifest has no entry for local_rank={local_rank}. "
            f"Known ranks: {list(_DEVICE_MANIFEST.keys())}"
        )
    return _DEVICE_MANIFEST[local_rank]


# ---------------------------------------------------------------------------
# DES-LOC process group helpers
# ---------------------------------------------------------------------------


@dataclass
class HeteroGroupInfo:
    """Describes a DES-LOC heterogeneous process group that spans multiple device classes.

    Unlike Megatron's uniform TP groups, a DES-LOC group may contain ranks from both
    SM86 (A6000) and SM90 (H100 NVL) devices.  The *effective_size* is a throughput-weighted
    count rather than a simple head-count, used to correct gradient scaling so that faster
    devices' larger token contributions are properly credited.

    Attributes:
        group:              The underlying ``dist.ProcessGroup`` (or ``None`` for single-device).
        world_size:         Raw number of ranks in the group (head-count).
        effective_size:     Throughput-weighted equivalent size, computed as
                            ``sum(desc.relative_throughput for desc in members) /
                            max(desc.relative_throughput for desc in members)``.
                            Used in lieu of ``tp_cp_group.size()`` for gradient scaling.
        local_rank_in_group: This process's rank within the group.
        member_descriptors: Ordered list of :class:`DeviceDescriptor` for each rank.
        is_heterogeneous:   True if the group spans more than one :class:`DeviceClass`.
    """
    group: Optional[dist.ProcessGroup]
    world_size: int
    effective_size: float
    local_rank_in_group: int
    member_descriptors: List[DeviceDescriptor]
    is_heterogeneous: bool


def build_hetero_tp_cp_group(
    tp_size: int,
    cp_size: int,
    global_rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> HeteroGroupInfo:
    """Build (or describe) the TP×CP group for this rank under DES-LOC topology.

    In a standard Megatron setup this would be a uniform group of ``tp_size * cp_size``
    ranks.  Under DES-LOC the group may be *heterogeneous* — for example a 2-way TP group
    spanning one A6000 (rank 0) and the H100 (rank 2).  The effective size accounts for
    their differing throughputs so that gradient scaling remains numerically consistent.

    Args:
        tp_size:     Tensor-parallel degree.
        cp_size:     Context-parallel degree.
        global_rank: This process's global rank.  Defaults to ``dist.get_rank()``.
        world_size:  Total number of processes.  Defaults to ``dist.get_world_size()``.

    Returns:
        A :class:`HeteroGroupInfo` fully describing this rank's TP×CP group.
    """
    if not dist.is_available() or not dist.is_initialized():
        # Single-process / unit-test path: synthesize a trivial group descriptor.
        local_rank = torch.cuda.current_device() if torch.cuda.is_available() else 0
        desc = _DEVICE_MANIFEST.get(local_rank, _DEVICE_MANIFEST[0])
        return HeteroGroupInfo(
            group=None,
            world_size=1,
            effective_size=1.0,
            local_rank_in_group=0,
            member_descriptors=[desc],
            is_heterogeneous=False,
        )

    if global_rank is None:
        global_rank = dist.get_rank()
    if world_size is None:
        world_size = dist.get_world_size()

    group_size = tp_size * cp_size
    # Determine which contiguous group this rank belongs to.
    group_idx = global_rank // group_size
    start = group_idx * group_size
    ranks_in_group = list(range(start, min(start + group_size, world_size)))

    # Map global ranks to local (physical) ranks via the manifest.
    # In DES-LOC the mapping is identity for a 3-GPU single-node setup.
    member_descs = []
    for r in ranks_in_group:
        local_r = r % len(_DEVICE_MANIFEST)
        member_descs.append(_DEVICE_MANIFEST.get(local_r, _DEVICE_MANIFEST[0]))

    device_classes = {d.device_class for d in member_descs}
    is_hetero = len(device_classes) > 1

    # Effective size: normalise so the fastest device counts as 1.0 unit.
    max_tp = max(d.relative_throughput for d in member_descs)
    effective_size = sum(d.relative_throughput for d in member_descs) / max_tp

    local_rank_in_group = global_rank - start

    if is_hetero:
        logger.info(
            "DES-LOC TP×CP group %d is heterogeneous: %s "
            "(effective_size=%.2f, head_count=%d)",
            group_idx,
            [d.device_class.name for d in member_descs],
            effective_size,
            len(ranks_in_group),
        )

    return HeteroGroupInfo(
        group=dist.new_group(ranks=ranks_in_group),
        world_size=len(ranks_in_group),
        effective_size=effective_size,
        local_rank_in_group=local_rank_in_group,
        member_descriptors=member_descs,
        is_heterogeneous=is_hetero,
    )


# ---------------------------------------------------------------------------
# DES-LOC locality cache for aux_loss accumulation
# ---------------------------------------------------------------------------


@dataclass
class LocalityCacheEntry:
    """A single cached aux_loss contribution stored in CPU DRAM.

    DES-LOC caches router outputs from previous micro-batches in CPU DRAM to amortise
    recomputation overhead.  Each entry carries a *staleness_weight* in [0, 1] that
    decays the contribution of old entries when the cache is replayed during loss
    computation.

    Attributes:
        step:              Training step at which this entry was created.
        aux_loss_value:    Scalar aux_loss (detached, CPU tensor).
        num_local_tokens:  Token count on the originating rank at cache time.
        effective_tp_cp:   Effective TP×CP group size at cache time.
        staleness_weight:  Replay weight, decayed by ``DESLOCCacheConfig.staleness_decay``
                           per step.
    """
    step: int
    aux_loss_value: Tensor        # CPU tensor, scalar
    num_local_tokens: int
    effective_tp_cp: float
    staleness_weight: float = 1.0


@dataclass
class DESLOCCacheConfig:
    """Configuration for the DES-LOC locality cache used in aux_loss accumulation.

    Attributes:
        max_entries:       Maximum number of micro-batch entries to cache.
        staleness_decay:   Multiplicative decay applied to each cached entry's weight
                           per training step.  Set to 0.0 to disable cache replay.
        pin_memory:        Whether to pin CPU tensors for faster H2D transfer.
        enable_cache:      Master switch; if False the cache is bypassed entirely.
    """
    max_entries: int = 8
    staleness_decay: float = 0.9
    pin_memory: bool = True
    enable_cache: bool = True


class DESLOCLocalityCache:
    """CPU-DRAM locality cache for MoE aux_loss contributions.

    This cache is the core DES-LOC data structure that enables *decoupled execution*:
    the router can run one or more steps ahead of the expert network, and its aux_loss
    outputs are staged here so that the optimizer step still receives correctly-scaled
    gradient information even when the live forward pass is not yet complete.

    Under the standard Megatron scheme every micro-batch computes and immediately
    backprops the aux_loss.  DES-LOC instead accumulates aux_loss contributions over a
    window of micro-batches (up to ``max_entries``), weighting staler entries by
    ``staleness_decay^(current_step - entry_step)``, then flushes the combined loss into
    the backward pass.

    The cache lives in CPU DRAM (1.5 TB available) so it does not consume precious GPU
    VRAM on either the A6000 or H100 devices.

    Args:
        config: :class:`DESLOCCacheConfig` controlling cache behaviour.
    """

    def __init__(self, config: DESLOCCacheConfig) -> None:
        self.config = config
        self._entries: List[LocalityCacheEntry] = []
        self._current_step: int = 0

    def push(
        self,
        aux_loss: Tensor,
        num_local_tokens: int,
        effective_tp_cp: float,
        step: Optional[int] = None,
    ) -> None:
        """Stage a new aux_loss value from the current micro-batch.

        The tensor is detached and moved to CPU to avoid holding GPU memory across steps.

        Args:
            aux_loss:          Scalar loss tensor (may be on any device).
            num_local_tokens:  Token count on this rank for the current micro-batch.
            effective_tp_cp:   Effective TP×CP group size (from :class:`HeteroGroupInfo`).
            step:              Training step; defaults to the internal counter.
        """
        if not self.config.enable_cache:
            return

        if step is None:
            step = self._current_step

        cpu_tensor = aux_loss.detach().cpu()
        if self.config.pin_memory:
            cpu_tensor = cpu_tensor.pin_memory()

        entry = LocalityCacheEntry(
            step=step,
            aux_loss_value=cpu_tensor,
            num_local_tokens=num_local_tokens,
            effective_tp_cp=effective_tp_cp,
            staleness_weight=1.0,
        )
        self._entries.append(entry)

        # Evict oldest entries beyond the window.
        if len(self._entries) > self.config.max_entries:
            evicted = self._entries.pop(0)
            logger.debug(
                "DES-LOC cache evicted entry from step=%d (staleness=%.3f)",
                evicted.step,
                evicted.staleness_weight,
            )

    def replay(self, current_step: int, device: torch.device) -> Optional[Tensor]:
        """Compute a staleness-weighted aggregate aux_loss from cached entries.

        Each entry's contribution is weighted by
        ``staleness_decay ^ (current_step - entry.step)``.
        Entries with weight < 1e-4 are pruned.

        Args:
            current_step: The current training step (used to compute staleness).
            device:       Device to place the returned tensor on.

        Returns:
            A scalar tensor representing the weighted aggregate aux_loss, or ``None``
            if the cache is empty or disabled.
        """
        if not self.config.enable_cache or not self._entries:
            return None

        self._current_step = current_step
        total_weight = 0.0
        weighted_sum = torch.tensor(0.0, dtype=torch.float32)

        surviving_entries: List[LocalityCacheEntry] = []
        for entry in self._entries:
            age = current_step - entry.step
            weight = (self.config.staleness_decay ** age) * entry.staleness_weight
            if weight < 1e-4:
                continue
            weighted_sum += weight * entry.aux_loss_value.float()
            total_weight += weight
            entry.staleness_weight = weight  # update for next replay call
            surviving_entries.append(entry)

        self._entries = surviving_entries

        if total_weight < 1e-8:
            return None

        result = (weighted_sum / total_weight).to(device)
        logger.debug(
            "DES-LOC cache replay: %d entries, total_weight=%.4f, "
            "aggregate_aux_loss=%.6f",
            len(surviving_entries),
            total_weight,
            result.item(),
        )
        return result

    def clear(self) -> None:
        """Flush all cached entries (e.g. at the start of a new epoch)."""
        self._entries.clear()
        logger.info("DES-LOC locality cache cleared.")

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# MoEAuxLossAutoScaler — heterogeneous-aware
# ---------------------------------------------------------------------------


class HeteroMoEAuxLossAutoScaler(Function):
    """Custom autograd function that injects a scaled auxiliary loss gradient.

    This is the DES-LOC adaptation of Megatron's ``MoEAuxLossAutoScaler``.  The upstream
    version simply stores the (scaled) aux_loss as a scalar and returns it as the gradient
    of the activation in the backward pass.  The DES-LOC version extends this with:

    - **Phase tagging**: distinguishes router-phase gradients (heterogeneous TP×CP
      correction applies) from expert-phase gradients (standard DP averaging).
    - **Cache-augmented loss**: optionally blends a cached replay value from the DES-LOC
      locality cache into the stored loss before backward.
    - **Device-class awareness**: the correction factor is read from the
      :class:`HeteroGroupInfo` rather than from a simple ``tp_cp_group.size()`` integer.

    The gradient injection formula mirrors Megatron's fix:

        scaled_loss = aux_loss * num_local_tokens * effective_tp_cp_size

    where ``effective_tp_cp_size`` is the throughput-weighted group size from
    :class:`HeteroGroupInfo` (equals ``tp_cp_group.size()`` for homogeneous groups).

    Usage::

        activation = HeteroMoEAuxLossAutoScaler.apply(
            activation, scaled_aux_loss
        )
    """

    @staticmethod
    def forward(ctx, activation: Tensor, aux_loss: Tensor) -> Tensor:
        """Store *aux_loss* for gradient injection; return *activation* unchanged."""
        ctx.save_for_backward(aux_loss)
        return activation

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        """Return the stored aux_loss as the gradient of *activation*; None for *aux_loss*."""
        (aux_loss,) = ctx.saved_tensors
        aux_loss_grad = aux_loss * torch.ones_like(grad_output)
        return aux_loss_grad, None


# ---------------------------------------------------------------------------
# Core gradient scaling logic
# ---------------------------------------------------------------------------


class HeteroMoELossGradScale:
    """Heterogeneity-aware MoE auxiliary loss gradient scaler for DES-LOC.

    This class encapsulates the corrected gradient scaling logic from Megatron commit
    9ef8a2a, adapted for the DES-LOC heterogeneous hardware environment.

    The central responsibility is to compute the *pre-scaling factor* that must be
    applied to aux_loss and z_loss tensors before they are stored by the autograd
    function, so that after ``finalize_model_grads`` divides by ``total_global_tokens``
    the net effect on the gradient is exactly ``1 / (num_micro_batches * dp_size)``.

    Under DES-LOC the correction factor is not simply ``num_local_tokens * tp_cp_size``
    (as in Megatron) but rather:

        correction = num_local_tokens * hetero_group.effective_size

    where ``effective_size`` accounts for asymmetric token throughput across SM86 and
    SM90 devices.  Additionally, when the DES-LOC locality cache is active, the scaler
    can augment the live loss with a weighted replay from cached micro-batches.

    Args:
        hetero_group:   :class:`HeteroGroupInfo` describing this rank's TP×CP group.
        cache_config:   Optional :class:`DESLOCCacheConfig`; if ``None`` caching is
                        disabled.
        calculate_per_token_loss: Whether the training run uses per-token loss scaling.
                        Must match the flag passed to the model config.
        current_step:   Mutable reference to the current training step (list of one int).
    """

    def __init__(
        self,
        hetero_group: HeteroGroupInfo,
        cache_config: Optional[DESLOCCacheConfig] = None,
        calculate_per_token_loss: bool = True,
        current_step: Optional[List[int]] = None,
    ) -> None:
        self.hetero_group = hetero_group
        self.calculate_per_token_loss = calculate_per_token_loss
        self._step_ref = current_step if current_step is not None else [0]

        if cache_config is not None:
            self.cache = DESLOCLocalityCache(cache_config)
        else:
            self.cache = DESLOCLocalityCache(DESLOCCacheConfig(enable_cache=False))

        logger.info(
            "HeteroMoELossGradScale initialised: per_token_loss=%s, "
            "group_world_size=%d, group_effective_size=%.2f, is_hetero=%s, "
            "cache_enabled=%s",
            self.calculate_per_token_loss,
            self.hetero_group.world_size,
            self.hetero_group.effective_size,
            self.hetero_group.is_heterogeneous,
            cache_config.enable_cache if cache_config else False,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scale_aux_loss(
        self,
        activation: Tensor,
        aux_loss: Tensor,
        valid_token_count: Optional[int] = None,
    ) -> Tensor:
        """Apply corrected aux_loss gradient scaling and inject via autograd.

        This mirrors Megatron's ``attach_and_log_load_balancing_loss`` fix but uses
        the heterogeneous effective group size in place of ``tp_cp_group.size()``.

        When ``calculate_per_token_loss`` is True, the scaling applied is:

            scaled_aux_loss = aux_loss * num_local_tokens * effective_tp_cp_size

        This ensures that after ``finalize_model_grads`` divides by
        ``total_global_tokens = num_micro_batches * dp_size * num_local_tokens *
        effective_tp_cp_size``, the net gradient scale on aux_loss is
        ``1 / (num_micro_batches * dp_size)``.

        If the DES-LOC locality cache is active, the live aux_loss is blended with a
        cached replay value before scaling so that routing decisions from recent
        micro-batches contribute to the gradient of earlier layers.

        Args:
            activation:        Router activation tensor (returned unchanged in forward).
            aux_loss:          Scalar auxiliary load-balancing loss.
            valid_token_count: Number of non-padding tokens on this rank.  If ``None``,
                               ``activation.shape[0]`` is used.

        Returns:
            The *activation* tensor hooked to inject the scaled aux_loss gradient
            during backward.
        """
        if self.calculate_per_token_loss:
            num_local_tokens = (
                valid_token_count
                if valid_token_count is not None
                else activation.shape[0]
            )
            effective_tp_cp = self.hetero_group.effective_size

            # Optionally blend cached replay into the live aux_loss.
            live_aux_loss = aux_loss
            if self.cache.config.enable_cache:
                step = self._step_ref[0]
                cached = self.cache.replay(step, device=aux_loss.device)
                if cached is not None:
                    # Blend: 0.9 * live + 0.1 * cached (tunable).
                    live_aux_loss = 0.9 * aux_loss + 0.1 * cached

                # Stage this micro-batch's contribution for future replays.
                self.cache.push(
                    aux_loss=aux_loss,
                    num_local_tokens=num_local_tokens,
                    effective_tp_cp=effective_tp_cp,
                    step=step,
                )

            scaled = live_aux_loss * num_local_tokens * effective_tp_cp
            activation = HeteroMoEAuxLossAutoScaler.apply(activation, scaled)
        else:
            activation = HeteroMoEAuxLossAutoScaler.apply(activation, aux_loss)

        return activation

    def scale_z_loss(
        self,
        logits: Tensor,
        z_loss: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply corrected z_loss gradient scaling and inject via autograd.

        Mirrors Megatron's ``_compute_router_z_loss`` fix.  The forward-side coefficient
        division by ``tp_cp_group.size()`` (to average z_loss across TP+CP ranks rather
        than sum) is handled upstream by the caller; here we apply only the backward-side
        correction that mirrors the aux_loss path.

        When ``calculate_per_token_loss`` is True:

            num_local_tokens = sum(~padding_mask) if padding_mask else logits.shape[0]
            scaled_z_loss = z_loss * num_local_tokens * effective_tp_cp_size

        Args:
            logits:       Router logit tensor (shape: [num_local_tokens, num_experts]).
            z_loss:       Scalar z-regularisation loss.
            padding_mask: Boolean mask where True = padding token (excluded from count).

        Returns:
            The *logits* tensor hooked to inject the scaled z_loss gradient during backward.
        """
        if self.calculate_per_token_loss:
            num_local_tokens = (
                int((~padding_mask).sum().item())
                if padding_mask is not None
                else logits.shape[0]
            )
            effective_tp_cp = self.hetero_group.effective_size
            scaled = z_loss * num_local_tokens * effective_tp_cp
            logits = HeteroMoEAuxLossAutoScaler.apply(logits, scaled)
        else:
            logits = HeteroMoEAuxLossAutoScaler.apply(logits, z_loss)

        return logits

    def advance_step(self) -> None:
        """Increment the internal training step counter."""
        self._step_ref[0] += 1


# ---------------------------------------------------------------------------
# z_loss function (ported from Megatron, heterogeneous-topology aware)
# ---------------------------------------------------------------------------


def z_loss_func(
    logits: Tensor,
    z_loss_coeff: float,
    padding_mask: Optional[Tensor] = None,
) -> Tensor:
    """Compute the router z-loss (entropy regularisation).

    The z-loss penalises large router logit magnitudes to encourage numerical stability
    and prevent routing collapse.  The formula is:

        z_loss = z_loss_coeff * mean(log(sum(exp(logits)))^2)

    where the mean is taken over non-padding tokens.

    In Megatron this is computed per TP+CP rank on local logits.  Under DES-LOC the same
    computation runs per device, and the caller is responsible for dividing the coefficient
    by ``effective_tp_cp_size`` before calling this function (matching the forward-side
    normalisation from commit 9ef8a2a).

    Args:
        logits:        Float tensor of shape ``[T, E]`` where T = local tokens, E = experts.
        z_loss_coeff:  Coefficient, already divided by ``effective_tp_cp_size`` by the caller.
        padding_mask:  Boolean tensor of shape ``[T]``; True positions are padding.

    Returns:
        Scalar z-loss tensor on the same device as *logits*.
    """
    if padding_mask is not None:
        valid = ~padding_mask
        log_z = torch.logsumexp(logits[valid], dim=-1)
    else:
        log_z = torch.logsumexp(logits, dim=-1)

    z_loss = z_loss_coeff * (log_z ** 2).mean()
    return z_loss


# ---------------------------------------------------------------------------
# Router integration shim
# ---------------------------------------------------------------------------


class HeteroTopKRouterLossIntegration:
    """Thin integration layer that wires :class:`HeteroMoELossGradScale` into a router.

    In Megatron ``TopKRouter`` directly calls ``MoEAuxLossAutoScaler.apply(...)`` with
    the scaling baked in.  Under DES-LOC we extract that logic into this class so that:

    - The same scaler instance can be shared across pipeline stages.
    - The DES-LOC locality cache persists across micro-batches within a step.
    - The heterogeneous group info is injected once at construction rather than
      re-computed on every forward pass.

    Args:
        scaler:           Pre-configured :class:`HeteroMoELossGradScale` instance.
        moe_z_loss_coeff: Base z-loss coefficient from model config (before TP÷ adjustment).
        num_moe_experts:  Total number of experts (used for logging context).
    """

    def __init__(
        self,
        scaler: HeteroMoELossGradScale,
        moe_z_loss_coeff: float = 0.0,
        num_moe_experts: int = 8,
    ) -> None:
        self.scaler = scaler
        self.moe_z_loss_coeff = moe_z_loss_coeff
        self.num_moe_experts = num_moe_experts

    def attach_load_balancing_loss(
        self,
        activation: Tensor,
        aux_loss: Tensor,
        valid_token_count: Optional[int] = None,
    ) -> Tensor:
        """Inject load-balancing loss gradient into *activation*.

        Drop-in replacement for Megatron's ``MoEAuxLossAutoScaler.apply(activation,
        aux_loss * num_local_tokens)`` with heterogeneous TP×CP correction.

        Args:
            activation:        Router gate activation.
            aux_loss:          Load-balancing auxiliary loss scalar.
            valid_token_count: Non-padding token count; None → activation.shape[0].

        Returns:
            Activation tensor with gradient hook for aux_loss injection.
        """
        return self.scaler.scale_aux_loss(activation, aux_loss, valid_token_count)

    def attach_z_loss(
        self,
        logits: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute and inject z-loss gradient into *logits*.

        The z-loss coefficient is adjusted by ``effective_tp_cp_size`` on the
        forward side (to average across ranks), then the backward correction is
        applied by :meth:`HeteroMoELossGradScale.scale_z_loss`.

        Args:
            logits:       Router logit tensor ``[T, E]``.
            padding_mask: Optional padding mask (True = pad).

        Returns:
            Logit tensor with gradient hook for z-loss injection.
        """
        if self.moe_z_loss_coeff == 0.0:
            return logits

        # Forward-side normalisation: divide by effective_tp_cp to average across ranks.
        adjusted_coeff = self.moe_z_loss_coeff / self.scaler.hetero_group.effective_size
        z_loss = z_loss_func(logits, adjusted_coeff, padding_mask=padding_mask)

        return self.scaler.scale_z_loss(logits, z_loss, padding_mask=padding_mask)


# ---------------------------------------------------------------------------
# Gradient invariance validator (debug / profiling utility)
# ---------------------------------------------------------------------------


class GradientInvarianceValidator:
    """Validates that the per-token aux_loss gradient is invariant to TP×CP configuration.

    This is the DES-LOC equivalent of Megatron's ``TestPerTokenAuxLoss`` regression
    test, adapted to run at training time as a lightweight sanity check.

    The validator computes a *reference gradient* at TP=1, CP=1 (effective_size=1.0) and
    compares it against the gradient produced by the current heterogeneous configuration.
    A relative error above ``tolerance`` triggers a warning.

    Args:
        tolerance: Maximum acceptable relative gradient error (default 1e-3).
    """

    def __init__(self, tolerance: float = 1e-3) -> None:
        self.tolerance = tolerance
        self._reference_grad: Optional[Tensor] = None
        self._reference_num_tokens: Optional[int] = None

    def record_reference(
        self,
        grad: Tensor,
        num_tokens: int,
    ) -> None:
        """Store a reference gradient for future comparisons.

        Should be called once with the gradient obtained at TP=1, CP=1.
        """
        self._reference_grad = grad.detach().cpu().float()
        self._reference_num_tokens = num_tokens

    def validate(
        self,
        grad: Tensor,
        num_tokens: int,
        hetero_group: HeteroGroupInfo,
        step: int,
    ) -> bool:
        """Compare *grad* against the stored reference.

        Args:
            grad:         Gradient tensor from the current TP×CP config.
            num_tokens:   Token count on this rank.
            hetero_group: Group info for logging context.
            step:         Training step (for log messages).

        Returns:
            True if the gradient is within tolerance; False otherwise.
        """
        if self._reference_grad is None:
            logger.debug("GradientInvarianceValidator: no reference recorded yet.")
            return True

        candidate = grad.detach().cpu().float()
        # Scale candidate to account for different token counts across configs.
        if self._reference_num_tokens is not None and num_tokens > 0:
            scale = self._reference_num_tokens / num_tokens
        else:
            scale = 1.0

        rel_err = (
            (candidate * scale - self._reference_grad).abs().mean()
            / (self._reference_grad.abs().mean() + 1e-8)
        ).item()

        if rel_err > self.tolerance:
            logger.warning(
                "Step %d: aux_loss gradient invariance violated! "
                "relative_error=%.4e > tolerance=%.4e. "
                "hetero_group: world_size=%d, effective_size=%.2f, is_hetero=%s. "
                "This may indicate a misconfigured TP×CP effective size.",
                step,
                rel_err,
                self.tolerance,
                hetero_group.world_size,
                hetero_group.effective_size,
                hetero_group.is_heterogeneous,
            )
            return False

        logger.debug(
            "Step %d: aux_loss gradient invariance OK (rel_err=%.2e).",
            step,
            rel_err,
        )
        return True


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def make_des_loc_loss_scaler(
    tp_size: int = 1,
    cp_size: int = 1,
    calculate_per_token_loss: bool = True,
    enable_cache: bool = True,
    cache_max_entries: int = 8,
    cache_staleness_decay: float = 0.9,
    moe_z_loss_coeff: float = 0.0,
    num_moe_experts: int = 8,
    current_step: Optional[List[int]] = None,
) -> Tuple[HeteroMoELossGradScale, HeteroTopKRouterLossIntegration]:
    """Convenience factory for creating a fully-configured DES-LOC loss scaler.

    Builds the :class:`HeteroGroupInfo`, :class:`DESLOCCacheConfig`,
    :class:`HeteroMoELossGradScale`, and :class:`HeteroTopKRouterLossIntegration`
    in one call.

    Args:
        tp_size:                  Tensor-parallel degree.
        cp_size:                  Context-parallel degree.
        calculate_per_token_loss: Whether per-token loss scaling is active.
        enable_cache:             Whether to enable the DES-LOC locality cache.
        cache_max_entries:        Maximum cached micro-batches.
        cache_staleness_decay:    Cache entry decay rate per step.
        moe_z_loss_coeff:         Z-loss coefficient from model config.
        num_moe_experts:          Number of MoE experts.
        current_step:             Shared mutable step counter (list of one int).

    Returns:
        Tuple of (scaler, integration_layer).
    """
    hetero_group = build_hetero_tp_cp_group(tp_size=tp_size, cp_size=cp_size)
    cache_config = DESLOCCacheConfig(
        max_entries=cache_max_entries,
        staleness_decay=cache_staleness_decay,
        pin_memory=torch.cuda.is_available(),
        enable_cache=enable_cache,
    )
    scaler = HeteroMoELossGradScale(
        hetero_group=hetero_group,
        cache_config=cache_config,
        calculate_per_token_loss=calculate_per_token_loss,
        current_step=current_step,
    )
    integration = HeteroTopKRouterLossIntegration(
        scaler=scaler,
        moe_z_loss_coeff=moe_z_loss_coeff,
        num_moe_experts=num_moe_experts,
    )
    return scaler, integration


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    import unittest

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    class TestHeteroMoEAuxLossAutoScaler(unittest.TestCase):
        """Tests for HeteroMoEAuxLossAutoScaler custom autograd function."""

        def test_forward_identity(self):
            """Autograd function must return activation unchanged in forward."""
            act = torch.randn(16, 8, requires_grad=True)
            aux = torch.tensor(0.5)
            out = HeteroMoEAuxLossAutoScaler.apply(act, aux)
            self.assertTrue(torch.allclose(out, act))

        def test_backward_injects_aux_loss(self):
            """Backward pass must inject scaled aux_loss as gradient of activation."""
            act = torch.ones(4, 4, requires_grad=True)
            aux_loss = torch.tensor(2.0)
            out = HeteroMoEAuxLossAutoScaler.apply(act, aux_loss)
            # Fake upstream gradient of all-ones.
            out.sum().backward()
            expected_grad = torch.ones_like(act) * 2.0
            self.assertTrue(
                torch.allclose(act.grad, expected_grad),
                f"Expected grad={expected_grad}, got {act.grad}",
            )

        def test_backward_shape_preserved(self):
            """Gradient shape must match activation shape."""
            act = torch.randn(32, 64, requires_grad=True)
            aux = torch.tensor(0.1)
            out = HeteroMoEAuxLossAutoScaler.apply(act, aux)
            out.mean().backward()
            self.assertEqual(act.grad.shape, act.shape)

    class TestDeviceManifest(unittest.TestCase):
        """Tests for the device topology manifest."""

        def test_known_ranks(self):
            """All three ranks must be present in the manifest."""
            for rank in [0, 1, 2]:
                desc = get_device_descriptor(rank)
                self.assertIsNotNone(desc)
                self.assertEqual(desc.local_rank, rank)

        def test_unknown_rank_raises(self):
            """Unknown rank should raise KeyError."""
            with self.assertRaises(KeyError):
                get_device_descriptor(99)

        def test_h100_faster_than_a6000(self):
            """H100 (rank 2) must have higher relative_throughput than A6000 (rank 0)."""
            a6000 = get_device_descriptor(0)
            h100 = get_device_descriptor(2)
            self.assertGreater(h100.relative_throughput, a6000.relative_throughput)

        def test_a6000_devices_symmetric(self):
            """Both A6000 devices (ranks 0 and 1) must have identical specs."""
            d0 = get_device_descriptor(0)
            d1 = get_device_descriptor(1)
            self.assertEqual(d0.device_class, d1.device_class)
            self.assertAlmostEqual(d0.relative_throughput, d1.relative_throughput)
            self.assertAlmostEqual(d0.vram_gb, d1.vram_gb)

    class TestHeteroGroupInfo(unittest.TestCase):
        """Tests for build_hetero_tp_cp_group in single-process mode."""

        def test_single_process_group(self):
            """In single-process mode, should return trivial group with effective_size=1."""
            group = build_hetero_tp_cp_group(tp_size=1, cp_size=1)
            self.assertEqual(group.world_size, 1)
            self.assertAlmostEqual(group.effective_size, 1.0)
            self.assertFalse(group.is_heterogeneous)

        def test_single_process_not_heterogeneous(self):
            """Single-process group must not be flagged as heterogeneous."""
            group = build_hetero_tp_cp_group(tp_size=2, cp_size=1)
            self.assertFalse(group.is_heterogeneous)

    class TestDESLOCLocalityCache(unittest.TestCase):
        """Tests for the CPU-DRAM locality cache."""

        def _make_cache(self, **kwargs) -> DESLOCLocalityCache:
            cfg = DESLOCCacheConfig(**kwargs)
            return DESLOCLocalityCache(cfg)

        def test_push_and_len(self):
            """Pushed entries should be retrievable and counted."""
            cache = self._make_cache(max_entries=4, enable_cache=True)
            for i in range(3):
                cache.push(
                    aux_loss=torch.tensor(float(i)),
                    num_local_tokens=32,
                    effective_tp_cp=1.0,
                    step=i,
                )
            self.assertEqual(len(cache), 3)

        def test_eviction(self):
            """Cache must not exceed max_entries."""
            cache = self._make_cache(max_entries=3, enable_cache=True)
            for i in range(6):
                cache.push(
                    aux_loss=torch.tensor(1.0),
                    num_local_tokens=32,
                    effective_tp_cp=1.0,
                    step=i,
                )
            self.assertLessEqual(len(cache), 3)

        def test_replay_returns_none_when_empty(self):
            """Replay on empty cache must return None."""
            cache = self._make_cache(enable_cache=True)
            result = cache.replay(current_step=0, device=torch.device("cpu"))
            self.assertIsNone(result)

        def test_replay_weighted_decay(self):
            """Replay result must be between 0 and the pushed value (staleness reduces it)."""
            cache = self._make_cache(
                max_entries=4, staleness_decay=0.5, enable_cache=True
            )
            cache.push(
                aux_loss=torch.tensor(1.0),
                num_local_tokens=32,
                effective_tp_cp=1.0,
                step=0,
            )
            # At step=2, decay = 0.5^2 = 0.25; weighted result = 0.25/0.25 = 1.0 still
            # (only one entry so normalisation cancels).
            result = cache.replay(current_step=2, device=torch.device("cpu"))
            self.assertIsNotNone(result)
            self.assertAlmostEqual(result.item(), 1.0, places=5)

        def test_disabled_cache_push_noop(self):
            """Pushing to a disabled cache must have no effect."""
            cache = self._make_cache(enable_cache=False)
            cache.push(
                aux_loss=torch.tensor(0.5),
                num_local_tokens=8,
                effective_tp_cp=1.0,
            )
            self.assertEqual(len(cache), 0)

        def test_replay_returns_none_when_disabled(self):
            """Replay on disabled cache must return None."""
            cache = self._make_cache(enable_cache=False)
            result = cache.replay(current_step=1, device=torch.device("cpu"))
            self.assertIsNone(result)

        def test_clear(self):
            """clear() must remove all entries."""
            cache = self._make_cache(enable_cache=True)
            for i in range(3):
                cache.push(torch.tensor(1.0), 16, 1.0, step=i)
            cache.clear()
            self.assertEqual(len(cache), 0)

    class TestHeteroMoELossGradScale(unittest.TestCase):
        """Tests for the core gradient scaling logic."""

        def _make_homogeneous_scaler(
            self, calculate_per_token_loss: bool = True
        ) -> HeteroMoELossGradScale:
            group = HeteroGroupInfo(
                group=None,
                world_size=1,
                effective_size=1.0,
                local_rank_in_group=0,
                member_descriptors=[_DEVICE_MANIFEST[0]],
                is_heterogeneous=False,
            )
            return HeteroMoELossGradScale(
                hetero_group=group,
                cache_config=DESLOCCacheConfig(enable_cache=False),
                calculate_per_token_loss=calculate_per_token_loss,
            )

        def _make_hetero_scaler(
            self, effective_size: float = 2.3, calculate_per_token_loss: bool = True
        ) -> HeteroMoELossGradScale:
            group = HeteroGroupInfo(
                group=None,
                world_size=2,
                effective_size=effective_size,
                local_rank_in_group=0,
                member_descriptors=[_DEVICE_MANIFEST[0], _DEVICE_MANIFEST[2]],
                is_heterogeneous=True,
            )
            return HeteroMoELossGradScale(
                hetero_group=group,
                cache_config=DESLOCCacheConfig(enable_cache=False),
                calculate_per_token_loss=calculate_per_token_loss,
            )

        def test_scale_aux_loss_per_token_homogeneous(self):
            """With per_token_loss=True and homogeneous group, gradient must equal
            aux_loss * num_local_tokens * 1.0 (effective_size=1)."""
            scaler = self._make_homogeneous_scaler()
            T, E = 16, 8
            act = torch.ones(T, E, requires_grad=True)
            aux_loss = torch.tensor(0.5)
            out = scaler.scale_aux_loss(act, aux_loss)
            out.sum().backward()
            # Expected grad = aux_loss * T * 1.0 = 0.5 * 16 = 8.0 on every element.
            expected = torch.full((T, E), fill_value=aux_loss.item() * T)
            self.assertTrue(
                torch.allclose(act.grad, expected, atol=1e-5),
                f"Expected {expected[0,0]:.4f}, got {act.grad[0,0]:.4f}",
            )

        def test_scale_aux_loss_no_per_token(self):
            """Without per_token_loss, gradient must equal raw aux_loss (no scaling)."""
            scaler = self._make_homogeneous_scaler(calculate_per_token_loss=False)
            T, E = 8, 4
            act = torch.ones(T, E, requires_grad=True)
            aux_loss = torch.tensor(3.0)
            out = scaler.scale_aux_loss(act, aux_loss)
            out.sum().backward()
            expected = torch.full((T, E), fill_value=3.0)
            self.assertTrue(torch.allclose(act.grad, expected, atol=1e-5))

        def test_scale_aux_loss_hetero_effective_size(self):
            """Hetero group must scale gradient by effective_size instead of world_size."""
            effective_size = 2.3
            scaler = self._make_hetero_scaler(effective_size=effective_size)
            T, E = 16, 8
            act = torch.ones(T, E, requires_grad=True)
            aux_loss = torch.tensor(1.0)
            out = scaler.scale_aux_loss(act, aux_loss)
            out.sum().backward()
            expected_val = 1.0 * T * effective_size
            self.assertAlmostEqual(act.grad[0, 0].item(), expected_val, places=4)

        def test_scale_aux_loss_valid_token_count(self):
            """valid_token_count should override activation.shape[0] for token counting."""
            scaler = self._make_homogeneous_scaler()
            T, E = 32, 8
            valid_tokens = 20
            act = torch.ones(T, E, requires_grad=True)
            aux_loss = torch.tensor(1.0)
            out = scaler.scale_aux_loss(act, aux_loss, valid_token_count=valid_tokens)
            out.sum().backward()
            expected_val = 1.0 * valid_tokens * 1.0
            self.assertAlmostEqual(act.grad[0, 0].item(), expected_val, places=4)

        def test_scale_z_loss_per_token_no_mask(self):
            """With per_token_loss=True and no mask, z_loss gradient must scale by T * eff."""
            scaler = self._make_homogeneous_scaler()
            T, E = 12, 6
            logits = torch.randn(T, E, requires_grad=True)
            z_loss = torch.tensor(0.25)
            out = scaler.scale_z_loss(logits, z_loss)
            out.sum().backward()
            expected_val = 0.25 * T * 1.0
            self.assertAlmostEqual(logits.grad[0, 0].item(), expected_val, places=4)

        def test_scale_z_loss_with_padding_mask(self):
            """Padding mask must reduce the effective token count."""
            scaler = self._make_homogeneous_scaler()
            T, E = 16, 4
            logits = torch.randn(T, E, requires_grad=True)
            z_loss = torch.tensor(1.0)
            # Mask last 4 tokens as padding.
            mask = torch.zeros(T, dtype=torch.bool)
            mask[-4:] = True  # True = padding
            valid = T - 4  # 12 valid tokens
            out = scaler.scale_z_loss(logits, z_loss, padding_mask=mask)
            out.sum().backward()
            expected_val = 1.0 * valid * 1.0
            self.assertAlmostEqual(logits.grad[0, 0].item(), expected_val, places=4)

        def test_advance_step(self):
            """advance_step must increment the shared step counter."""
            step_ref = [0]
            group = build_hetero_tp_cp_group()
            scaler = HeteroMoELossGradScale(
                hetero_group=group,
                calculate_per_token_loss=True,
                current_step=step_ref,
            )
            scaler.advance_step()
            scaler.advance_step()
            self.assertEqual(step_ref[0], 2)

    class TestGradientInvarianceValidator(unittest.TestCase):
        """Tests for GradientInvarianceValidator."""

        def _make_trivial_group(self) -> HeteroGroupInfo:
            return HeteroGroupInfo(
                group=None,
                world_size=1,
                effective_size=1.0,
                local_rank_in_group=0,
                member_descriptors=[_DEVICE_MANIFEST[0]],
                is_heterogeneous=False,
            )

        def test_validate_no_reference(self):
            """Validation without a stored reference must return True (no constraint)."""
            v = GradientInvarianceValidator()
            dummy_grad = torch.ones(4, 4)
            result = v.validate(dummy_grad, 16, self._make_trivial_group(), step=0)
            self.assertTrue(result)

        def test_validate_identical_grads(self):
            """Identical gradient must pass validation."""
            v = GradientInvarianceValidator(tolerance=1e-3)
            ref = torch.ones(4, 4)
            v.record_reference(ref, num_tokens=16)
            result = v.validate(ref.clone(), 16, self._make_trivial_group(), step=1)
            self.assertTrue(result)

        def test_validate_large_error_fails(self):
            """Gradient with large relative error must fail validation."""
            v = GradientInvarianceValidator(tolerance=1e-3)
            ref = torch.ones(4, 4)
            v.record_reference(ref, num_tokens=16)
            bad_grad = ref * 1000.0
            result = v.validate(bad_grad, 16, self._make_trivial_group(), step=2)
            self.assertFalse(result)

    class TestZLossFunc(unittest.TestCase):
        """Tests for z_loss_func."""

        def test_output_shape_scalar(self):
            """z_loss_func must return a scalar tensor."""
            logits = torch.randn(8, 4)
            z = z_loss_func(logits, z_loss_coeff=0.01)
            self.assertEqual(z.shape, torch.Size([]))

        def test_no_mask_vs_all_valid_mask(self):
            """z_loss with no mask must equal z_loss with all-False mask."""
            logits = torch.randn(16, 8)
            mask = torch.zeros(16, dtype=torch.bool)
            z_no_mask = z_loss_func(logits, 0.01)
            z_full_mask = z_loss_func(logits, 0.01, padding_mask=mask)
            self.assertAlmostEqual(z_no_mask.item(), z_full_mask.item(), places=5)

        def test_zero_coeff_zero_loss(self):
            """Zero coefficient must produce zero loss."""
            logits = torch.randn(8, 4)
            z = z_loss_func(logits, z_loss_coeff=0.0)
            self.assertAlmostEqual(z.item(), 0.0, places=8)

        def test_padding_reduces_tokens(self):
            """Masking half the tokens must produce a different loss than no masking."""
            torch.manual_seed(42)
            logits = torch.randn(16, 4)
            mask = torch.zeros(16, dtype=torch.bool)
            mask[8:] = True  # mask second half
            z_full = z_loss_func(logits, 0.1)
            z_masked = z_loss_func(logits, 0.1, padding_mask=mask)
            # The two values should differ because different tokens are averaged.
            self.assertNotAlmostEqual(z_full.item(), z_masked.item(), places=4)

    class TestHeteroTopKRouterLossIntegration(unittest.TestCase):
        """Integration tests for HeteroTopKRouterLossIntegration."""

        def _make_integration(self, moe_z_loss_coeff: float = 0.01) -> HeteroTopKRouterLossIntegration:
            scaler, integration = make_des_loc_loss_scaler(
                tp_size=1,
                cp_size=1,
                calculate_per_token_loss=True,
                enable_cache=False,
                moe_z_loss_coeff=moe_z_loss_coeff,
                num_moe_experts=8,
            )
            return integration

        def test_attach_load_balancing_loss_returns_tensor(self):
            """attach_load_balancing_loss must return a tensor of the same shape."""
            intg = self._make_integration()
            act = torch.randn(16, 8, requires_grad=True)
            aux = torch.tensor(0.3)
            out = intg.attach_load_balancing_loss(act, aux)
            self.assertEqual(out.shape, act.shape)

        def test_attach_z_loss_zero_coeff_identity(self):
            """With z_loss_coeff=0, attach_z_loss must return logits unchanged."""
            intg = self._make_integration(moe_z_loss_coeff=0.0)
            logits = torch.randn(16, 8, requires_grad=True)
            out = intg.attach_z_loss(logits)
            self.assertIs(out, logits)

        def test_attach_z_loss_nonzero_hooks_backward(self):
            """With z_loss_coeff>0, backward must produce a non-zero gradient on logits."""
            intg = self._make_integration(moe_z_loss_coeff=0.1)
            logits = torch.randn(16, 8, requires_grad=True)
            out = intg.attach_z_loss(logits)
            out.sum().backward()
            self.assertIsNotNone(logits.grad)
            self.assertFalse(torch.all(logits.grad == 0))

        def test_gradient_scaling_invariant_to_effective_size(self):
            """Per-token aux_loss gradient (normalised by num_tokens * effective_size)
            must match between a homogeneous group (eff=1) and a hetero group (eff=2.3)
            after the consumer normalises by the same factor."""
            T, E = 16, 8
            aux_val = 0.5

            def run_with_effective_size(eff: float) -> float:
                group = HeteroGroupInfo(
                    group=None,
                    world_size=1,
                    effective_size=eff,
                    local_rank_in_group=0,
                    member_descriptors=[_DEVICE_MANIFEST[0]],
                    is_heterogeneous=eff != 1.0,
                )
                scaler = HeteroMoELossGradScale(
                    hetero_group=group,
                    cache_config=DESLOCCacheConfig(enable_cache=False),
                    calculate_per_token_loss=True,
                )
                act = torch.ones(T, E, requires_grad=True)
                aux = torch.tensor(aux_val)
                out = scaler.scale_aux_loss(act, aux)
                out.sum().backward()
                raw_grad = act.grad[0, 0].item()
                # Normalise by the correction factor to get the "target" gradient.
                return raw_grad / (T * eff)

            target_eff1 = run_with_effective_size(1.0)
            target_eff23 = run_with_effective_size(2.3)
            self.assertAlmostEqual(target_eff1, target_eff23, places=4)

    class TestMakeDesLocLossScaler(unittest.TestCase):
        """Smoke tests for the make_des_loc_loss_scaler factory."""

        def test_returns_correct_types(self):
            """Factory must return (HeteroMoELossGradScale, HeteroTopKRouterLossIntegration)."""
            scaler, intg = make_des_loc_loss_scaler()
            self.assertIsInstance(scaler, HeteroMoELossGradScale)
            self.assertIsInstance(intg, HeteroTopKRouterLossIntegration)

        def test_shared_step_ref(self):
            """Advancing step via scaler must be visible through the shared ref."""
            step_ref = [0]
            scaler, _ = make_des_loc_loss_scaler(current_step=step_ref)
            scaler.advance_step()
            scaler.advance_step()
            self.assertEqual(step_ref[0], 2)

        def test_cache_disabled_by_default_when_requested(self):
            """When enable_cache=False, cache must not accumulate entries."""
            scaler, intg = make_des_loc_loss_scaler(enable_cache=False)
            act = torch.ones(8, 4, requires_grad=True)
            aux = torch.tensor(1.0)
            intg.attach_load_balancing_loss(act, aux)
            self.assertEqual(len(scaler.cache), 0)

    # Run all tests.
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHeteroMoEAuxLossAutoScaler)
    for cls in [
        TestDeviceManifest,
        TestHeteroGroupInfo,
        TestDESLOCLocalityCache,
        TestHeteroMoELossGradScale,
        TestGradientInvarianceValidator,
        TestZLossFunc,
        TestHeteroTopKRouterLossIntegration,
        TestMakeDesLocLossScaler,
    ]:
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
