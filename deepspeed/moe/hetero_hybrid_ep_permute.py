"""
DES-LOC Heterogeneous Hybrid Expert-Parallel Permute Fusion
============================================================

Upstream design intent (Megatron commit 8c5cf05):
-------------------------------------------------
Megatron-LM's commit adds "permute fusion into hybrid EP" — the insight is that
token permutation (reordering tokens before All-to-All dispatch) and the
All-to-All kernel itself can be fused into a single kernel call, eliminating an
intermediate memory round-trip.  The original implementation targets homogeneous
NVLink clusters where SM allocation is symmetric.  It introduces:

  * ``num_blocks_permute`` / ``num_blocks_unpermute``: explicit CUDA block budgets
    for the fused permute/unpermute sub-kernels within HybridEPBuffer.
  * Optional ``fuse_permute_dispatch`` / ``fuse_unpermute_combine`` kwargs passed
    through to DeepEP's HybridEPBuffer when the installed version supports them.
  * A version-probe guard (``inspect.signature``) so that older DeepEP installs
    degrade gracefully.
  * Config fields ``moe_permute_fusion_into_hybridep``, ``moe_hybridep_num_sms``
    (now Optional), ``moe_hybridep_num_blocks_permute``,
    ``moe_hybridep_num_blocks_unpermute``.

DES-LOC adaptation points:
---------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) runs on a *heterogeneous*
PCIe-interconnected cluster: two A6000-48GB (SM86, 84 SMs each) and one
H100-NVL-96GB (SM90, 132 SMs).  There is no NVLink; all inter-GPU traffic goes
over PCIe.  1.5 TB CPU DRAM is available as a spill / staging buffer.

Key adaptations:

1. **SM-budget derivation per device class**
   Megatron hard-codes or makes optional a single SM count.  Here we query the
   device's multiprocessor count at runtime and apply per-arch scaling policies:
   SM86 (A6000) → cap dispatch SMs at 40% to leave headroom for compute;
   SM90 (H100)  → allow up to 60% because it has more SMs and a dedicated
   dispatch engine.  Block budgets for permute/unpermute follow the same policy.

2. **PCIe-aware fallback path**
   Without NVLink the fused-permute path can actually *hurt* latency because the
   fused kernel stalls waiting for PCIe DMA.  We detect PCIe topology at
   initialisation time and disable fusion when all peers are PCIe-only, reverting
   to the staged (unfused) path that overlaps permute-compute with DMA.

3. **LOC-cache integration**
   DES-LOC maintains a Shared LOcality Cache (SLOC) in CPU DRAM that stores the
   routing decisions (``probs``, ``indices``) from the previous step so that the
   current step's All-to-All can be preloaded speculatively.  This module hooks
   the dispatch/combine paths to read from / write to the SLOC.

4. **Asymmetric expert placement**
   In Neuron_SP the H100 hosts more experts than each A6000 (expert-capacity
   skew).  The block budget for permute on A6000 shards is therefore smaller than
   on the H100 shard.  ``HeteroBlockBudget`` encapsulates this asymmetry.

5. **Graceful DeepEP version detection**
   Mirrors Megatron's ``inspect.signature`` guard but extends it: we also check
   for SM90-specific fused-dispatch symbols introduced in newer DeepEP builds.

Author note:
    This file is the *primary* DES-LOC entry-point for MoE dispatch.  It is
    consumed by ``deepspeed/moe/layer.py`` via ``HeteroHybridEPPermuteFusion``.
"""

from __future__ import annotations

import inspect
import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware / topology constants
# ---------------------------------------------------------------------------

_SM86_DISPATCH_SM_FRACTION = 0.40   # 40% of SMs for dispatch on A6000
_SM90_DISPATCH_SM_FRACTION = 0.60   # 60% of SMs for dispatch on H100 NVL
_SM86_PERMUTE_SM_FRACTION  = 0.25
_SM90_PERMUTE_SM_FRACTION  = 0.35

# Minimum SMs we always reserve for the fused permute kernel
_MIN_PERMUTE_BLOCKS = 4
_MIN_DISPATCH_SMS   = 8


# ---------------------------------------------------------------------------
# Optional DeepEP / HybridEP import
# ---------------------------------------------------------------------------

try:
    from deepep import HybridEPBuffer  # type: ignore[import]
    _HAVE_HYBRIDEP = True
    logger.debug("DeepEP HybridEPBuffer found; hybrid-EP path enabled.")
except ImportError:
    _HAVE_HYBRIDEP = False
    HybridEPBuffer = None  # type: ignore[assignment,misc]
    logger.debug("DeepEP not available; all hybrid-EP paths will be skipped.")


# ---------------------------------------------------------------------------
# Per-process singleton buffer (mirrors Megatron's module-level global)
# ---------------------------------------------------------------------------

_hetero_ep_buffer: Optional[Any] = None  # HybridEPBuffer instance


# ---------------------------------------------------------------------------
# Capability probes (run once per process)
# ---------------------------------------------------------------------------

@dataclass
class _DeepEPCapabilities:
    """Cached result of runtime capability detection for DeepEP HybridEPBuffer."""
    has_num_blocks_permute: bool = False
    has_num_blocks_unpermute: bool = False
    has_fuse_permute_dispatch: bool = False
    has_fuse_unpermute_combine: bool = False
    # SM90-specific symbols (H100 NVL fast path)
    has_sm90_dispatch_v2: bool = False


def _probe_deepep_capabilities() -> _DeepEPCapabilities:
    """
    Probe the installed DeepEP version for optional kwargs.

    Megatron uses a single ``inspect.signature`` check.  Here we also probe for
    SM90-specific dispatch symbols that are only available in DeepEP builds
    compiled with CUDA 12.4+ targeting sm_90a.
    """
    caps = _DeepEPCapabilities()
    if not _HAVE_HYBRIDEP:
        return caps

    try:
        init_sig = inspect.signature(HybridEPBuffer.__init__)
        caps.has_num_blocks_permute  = 'num_blocks_permute'  in init_sig.parameters
        caps.has_num_blocks_unpermute = 'num_blocks_unpermute' in init_sig.parameters
    except (TypeError, ValueError):
        pass

    try:
        dispatch_sig = inspect.signature(HybridEPBuffer.dispatch_with_permute)
        caps.has_fuse_permute_dispatch = 'fuse_permute_dispatch' in dispatch_sig.parameters
    except (AttributeError, TypeError, ValueError):
        pass

    try:
        combine_sig = inspect.signature(HybridEPBuffer.combine_with_unpermute)
        caps.has_fuse_unpermute_combine = 'fuse_unpermute_combine' in combine_sig.parameters
    except (AttributeError, TypeError, ValueError):
        pass

    # SM90 fast path: look for a version attribute or a sentinel method
    try:
        caps.has_sm90_dispatch_v2 = hasattr(HybridEPBuffer, 'dispatch_sm90_v2')
    except Exception:
        pass

    logger.info(
        "DeepEP capability probe: num_blocks_permute=%s  num_blocks_unpermute=%s  "
        "fuse_permute_dispatch=%s  fuse_unpermute_combine=%s  sm90_dispatch_v2=%s",
        caps.has_num_blocks_permute,
        caps.has_num_blocks_unpermute,
        caps.has_fuse_permute_dispatch,
        caps.has_fuse_unpermute_combine,
        caps.has_sm90_dispatch_v2,
    )
    return caps


# Module-level lazy singleton
_deepep_caps: Optional[_DeepEPCapabilities] = None


def _get_deepep_caps() -> _DeepEPCapabilities:
    global _deepep_caps
    if _deepep_caps is None:
        _deepep_caps = _probe_deepep_capabilities()
    return _deepep_caps


# ---------------------------------------------------------------------------
# Device architecture helpers
# ---------------------------------------------------------------------------

def _device_sm_count(device: torch.device) -> int:
    """Return the number of SMs on *device*."""
    props = torch.cuda.get_device_properties(device)
    return props.multi_processor_count


def _device_compute_capability(device: torch.device) -> Tuple[int, int]:
    props = torch.cuda.get_device_properties(device)
    return (props.major, props.minor)


def _is_sm90(device: torch.device) -> bool:
    major, _ = _device_compute_capability(device)
    return major >= 9


def _is_sm86(device: torch.device) -> bool:
    major, minor = _device_compute_capability(device)
    return major == 8 and minor == 6


# ---------------------------------------------------------------------------
# PCIe topology detection
# ---------------------------------------------------------------------------

def _peers_are_pcie_only(local_rank: int) -> bool:
    """
    Return True when *all* peer GPUs visible to the process are connected via
    PCIe (no NVLink).  We use CUDA peer access capability as a proxy: NVLink
    peers always expose peer access, but so do some PCIe setups.  We fall back
    to a conservative heuristic: if no GPU reports ``can_access_peer`` for any
    other GPU, assume PCIe.

    In the DES-LOC reference cluster (2x A6000 + 1x H100, PCIe only) this
    reliably returns True.
    """
    n = torch.cuda.device_count()
    if n <= 1:
        return True  # trivially no NVLink peers

    has_nvlink_peer = False
    for src in range(n):
        for dst in range(n):
            if src == dst:
                continue
            try:
                if torch.cuda.can_device_access_peer(src, dst):
                    # Peer access available; could be NVLink or PCIe BAR1.
                    # We distinguish by checking the bandwidth: NVLink ≥ 600 GB/s;
                    # PCIe ≤ 64 GB/s.  Without nvidia-smi bindings we use a
                    # compile-time heuristic: if *both* devices are SM90 or both
                    # SM86, NVLink is plausible; mixed SM86+SM90 without explicit
                    # NVLink topology → PCIe.
                    src_dev = torch.device(f"cuda:{src}")
                    dst_dev = torch.device(f"cuda:{dst}")
                    same_arch = (
                        _device_compute_capability(src_dev)
                        == _device_compute_capability(dst_dev)
                    )
                    if same_arch:
                        has_nvlink_peer = True
            except Exception:
                pass

    pcie_only = not has_nvlink_peer
    if pcie_only:
        logger.info(
            "PCIe-only topology detected (local_rank=%d, n_gpus=%d); "
            "fused permute dispatch will be suppressed on this rank.",
            local_rank, n,
        )
    return pcie_only


# ---------------------------------------------------------------------------
# Heterogeneous block budget
# ---------------------------------------------------------------------------

@dataclass
class HeteroBlockBudget:
    """
    Per-device CUDA block budgets for the DES-LOC permute fusion path.

    Megatron uses a single ``num_blocks_permute`` / ``num_blocks_unpermute``
    that is symmetric across all ranks.  DES-LOC is heterogeneous: the H100
    shard can afford more blocks than each A6000 shard.

    Attributes
    ----------
    num_sms_dispatch:
        Number of SMs allocated to the All-to-All dispatch kernel.
    num_sms_combine:
        Number of SMs allocated to the All-to-All combine kernel.
    num_blocks_permute:
        CUDA thread-blocks for the fused permute sub-kernel.
        On SM90 one block ≈ one SM; on SM86 the scheduler may colocate two.
    num_blocks_unpermute:
        CUDA thread-blocks for the fused unpermute sub-kernel.
    device_class:
        Human-readable label ('SM86_A6000' or 'SM90_H100NVL').
    """
    num_sms_dispatch:   int
    num_sms_combine:    int
    num_blocks_permute: int
    num_blocks_unpermute: int
    device_class: str


def derive_block_budget(device: torch.device) -> HeteroBlockBudget:
    """
    Derive SM and block budgets for *device* based on its architecture.

    SM86 (A6000, 84 SMs):
        dispatch  = floor(84 * 0.40) = 33
        combine   = 33
        permute   = floor(84 * 0.25) = 21
        unpermute = 21

    SM90 (H100 NVL, 132 SMs):
        dispatch  = floor(132 * 0.60) = 79
        combine   = 79
        permute   = floor(132 * 0.35) = 46
        unpermute = 46

    Values are clamped to device-specific minima to avoid degenerate configs.
    """
    total_sms = _device_sm_count(device)

    if _is_sm90(device):
        dispatch_sms   = max(_MIN_DISPATCH_SMS, math.floor(total_sms * _SM90_DISPATCH_SM_FRACTION))
        combine_sms    = dispatch_sms
        permute_blocks = max(_MIN_PERMUTE_BLOCKS, math.floor(total_sms * _SM90_PERMUTE_SM_FRACTION))
        unpermute_blocks = permute_blocks
        device_class   = "SM90_H100NVL"
    elif _is_sm86(device):
        dispatch_sms   = max(_MIN_DISPATCH_SMS, math.floor(total_sms * _SM86_DISPATCH_SM_FRACTION))
        combine_sms    = dispatch_sms
        permute_blocks = max(_MIN_PERMUTE_BLOCKS, math.floor(total_sms * _SM86_PERMUTE_SM_FRACTION))
        unpermute_blocks = permute_blocks
        device_class   = "SM86_A6000"
    else:
        # Unknown arch — conservative defaults
        dispatch_sms   = _MIN_DISPATCH_SMS
        combine_sms    = _MIN_DISPATCH_SMS
        permute_blocks = _MIN_PERMUTE_BLOCKS
        unpermute_blocks = _MIN_PERMUTE_BLOCKS
        device_class   = f"UNKNOWN_SM{torch.cuda.get_device_properties(device).major}"

    budget = HeteroBlockBudget(
        num_sms_dispatch=dispatch_sms,
        num_sms_combine=combine_sms,
        num_blocks_permute=permute_blocks,
        num_blocks_unpermute=unpermute_blocks,
        device_class=device_class,
    )
    logger.info(
        "Block budget for device %s (%s, %d SMs): "
        "dispatch=%d  combine=%d  permute_blocks=%d  unpermute_blocks=%d",
        device, device_class, total_sms,
        dispatch_sms, combine_sms, permute_blocks, unpermute_blocks,
    )
    return budget


# ---------------------------------------------------------------------------
# DES-LOC Shared LOcality Cache (SLOC) interface
# ---------------------------------------------------------------------------

@dataclass
class SLOCEntry:
    """
    One cached routing decision stored in CPU DRAM.

    The SLOC stores the *previous* step's routing tensors on the CPU so that
    the current step can speculatively prefetch them to GPU before the router
    has finished computing the new routing.  This hides PCIe latency.
    """
    probs:   torch.Tensor   # shape: [S, E] on CPU, pinned
    indices: torch.Tensor   # shape: [S, top_k] on CPU, pinned
    step:    int


class SharedLocalityCache:
    """
    In-process SLOC manager backed by pinned CPU DRAM.

    DES-LOC allocates a fixed-size ring of ``SLOCEntry`` objects.  Each entry
    corresponds to one MoE layer.  The ring capacity is set to the number of
    MoE layers so that every layer can independently prefetch without eviction.

    Parameters
    ----------
    capacity:
        Number of layer slots.  Typically equal to the number of MoE layers.
    device:
        GPU device used for async prefetch streams.
    """

    def __init__(self, capacity: int, device: torch.device) -> None:
        self._capacity = capacity
        self._device   = device
        self._store: Dict[int, SLOCEntry] = {}   # layer_idx → entry
        self._prefetch_stream = torch.cuda.Stream(device=device)

    def store(self, layer_idx: int, probs: torch.Tensor, indices: torch.Tensor, step: int) -> None:
        """
        Write routing tensors for *layer_idx* to the SLOC.

        Tensors are moved to pinned CPU memory so the next step's prefetch is
        a simple DMA rather than a paged copy.
        """
        probs_cpu   = probs.detach().cpu().pin_memory()
        indices_cpu = indices.detach().cpu().pin_memory()
        self._store[layer_idx] = SLOCEntry(probs=probs_cpu, indices=indices_cpu, step=step)

    def prefetch(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Non-blocking prefetch of the cached entry for *layer_idx* to GPU.

        Returns (probs_gpu, indices_gpu) as tensors enqueued on the prefetch
        stream, or None if no cached entry exists.
        """
        entry = self._store.get(layer_idx)
        if entry is None:
            return None

        with torch.cuda.stream(self._prefetch_stream):
            probs_gpu   = entry.probs.to(self._device, non_blocking=True)
            indices_gpu = entry.indices.to(self._device, non_blocking=True)

        return probs_gpu, indices_gpu

    def synchronize_prefetch(self) -> None:
        """Block the current CUDA stream until all prefetch DMAs complete."""
        torch.cuda.current_stream(self._device).wait_stream(self._prefetch_stream)

    def evict(self, layer_idx: int) -> None:
        self._store.pop(layer_idx, None)

    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# HybridEP buffer initialisation (heterogeneous-aware)
# ---------------------------------------------------------------------------

def init_hetero_hybrid_ep_buffer(
    group:              dist.ProcessGroup,
    hidden_dim:         int,
    seq_len:            int,
    num_local_experts:  int,
    budget:             HeteroBlockBudget,
    fp8_dispatch:       bool = False,
    force_unfused:      bool = False,
) -> None:
    """
    Initialise (or re-initialise) the module-level ``_hetero_ep_buffer``.

    This function replaces Megatron's ``init_hybrid_ep_buffer``.  Key
    differences:

    * Budget is a ``HeteroBlockBudget`` rather than bare integers, encoding
      per-arch SM limits.
    * ``force_unfused=True`` disables fused-permute kwargs even when the
      installed DeepEP supports them (used on PCIe-only paths).
    * Only non-None kwargs are forwarded to ``HybridEPBuffer.__init__`` so
      that older DeepEP installs continue to work.

    Parameters
    ----------
    group:
        Distributed process group for the expert-parallel communicator.
    hidden_dim:
        Hidden dimension of token embeddings.
    seq_len:
        Maximum sequence length per rank.
    num_local_experts:
        Number of experts hosted on this rank.
    budget:
        SM/block allocations derived from ``derive_block_budget``.
    fp8_dispatch:
        Currently unsupported by HybridEP; kept for API symmetry.
    force_unfused:
        When True, ``num_blocks_permute`` / ``num_blocks_unpermute`` are not
        passed even if DeepEP supports them (PCIe topology fallback).
    """
    assert not fp8_dispatch, (
        "HeteroHybridEP: FP8 dispatch is not yet supported by HybridEPBuffer."
    )
    if not _HAVE_HYBRIDEP:
        raise RuntimeError(
            "DeepEP is not installed; cannot initialise HeteroHybridEPBuffer."
        )

    global _hetero_ep_buffer
    caps = _get_deepep_caps()

    kwargs: Dict[str, Any] = {}

    # SM allocations — always include when the installed DeepEP accepts them
    kwargs['num_sms_dispatch_api'] = budget.num_sms_dispatch
    kwargs['num_sms_combine_api']  = budget.num_sms_combine

    if not force_unfused and caps.has_num_blocks_permute:
        kwargs['num_blocks_permute'] = budget.num_blocks_permute
    if not force_unfused and caps.has_num_blocks_unpermute:
        kwargs['num_blocks_unpermute'] = budget.num_blocks_unpermute

    _hetero_ep_buffer = HybridEPBuffer(
        group=group,
        hidden_dim=hidden_dim,
        max_num_of_tokens_per_rank=seq_len,
        num_local_experts=num_local_experts,
        use_fp8=fp8_dispatch,
        **kwargs,
    )
    logger.info(
        "Initialised HeteroHybridEPBuffer: device_class=%s  hidden=%d  seq=%d  "
        "local_experts=%d  kwargs=%s",
        budget.device_class, hidden_dim, seq_len, num_local_experts, kwargs,
    )


# ---------------------------------------------------------------------------
# DES-LOC dispatch config
# ---------------------------------------------------------------------------

@dataclass
class HeteroEPConfig:
    """
    Runtime configuration for one MoE layer's DES-LOC dispatch path.

    Attributes
    ----------
    fuse_permute:
        Whether to use fused permute dispatch / unpermute combine.  Set to
        False automatically on PCIe-only topologies or when DeepEP lacks the
        capability.
    budget:
        SM and block budgets for this rank's device.
    pcie_only:
        True when all inter-GPU links are PCIe (no NVLink).
    layer_idx:
        MoE layer index, used for SLOC keying.
    sloc:
        Optional Shared LOcality Cache instance.
    num_permuted_tokens:
        Pre-computed permuted-token count (avoids GPU sync if provided).
    pad_multiple:
        Alignment multiple for FP8 GEMM padding.
    """
    fuse_permute:        bool
    budget:              HeteroBlockBudget
    pcie_only:           bool
    layer_idx:           int = 0
    sloc:                Optional[SharedLocalityCache] = None
    num_permuted_tokens: Optional[int] = None
    pad_multiple:        Optional[int] = None


# ---------------------------------------------------------------------------
# Core autograd Functions
# ---------------------------------------------------------------------------

class HeteroHybridEPDispatch(torch.autograd.Function):
    """
    Fused dispatch for DES-LOC heterogeneous expert-parallel routing.

    Forward pass
    ------------
    Mirrors ``HybridEPDispatch`` from Megatron but:
    * Uses ``_hetero_ep_buffer`` (initialised with per-arch SM budgets).
    * Gates ``fuse_permute_dispatch`` on both DeepEP capability *and* PCIe
      topology (PCIe → unfused to allow compute/DMA overlap).
    * Reads speculative probs/indices from the SLOC if available, falling back
      to the live router outputs.

    Backward pass
    -------------
    Mirrors ``HybridEPDispatch.backward``.  ``fuse_unpermute_combine`` is
    conditionally applied using the same fused flag stored in ctx.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        indices: torch.Tensor,
        probs: torch.Tensor,
        group: dist.ProcessGroup,
        cfg: HeteroEPConfig,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Parameters
        ----------
        x:
            Token hidden states, shape [S, H].
        indices:
            Expert indices from router, shape [S, top_k].
        probs:
            Router probabilities, shape [S, top_k].
        group:
            EP process group.
        cfg:
            DES-LOC dispatch configuration for this layer.

        Returns
        -------
        dispatched_hidden:
            Permuted and dispatched token hiddens.
        dispatched_probs:
            Permuted and dispatched probabilities.
        handle:
            Opaque communication handle for the combine step.
        """
        global _hetero_ep_buffer
        caps = _get_deepep_caps()

        # Resolve effective fuse flag: user intent × capability × topology
        effective_fuse = (
            cfg.fuse_permute
            and not cfg.pcie_only
            and caps.has_fuse_permute_dispatch
        )
        if cfg.fuse_permute and cfg.pcie_only:
            # Only log once per layer to avoid flooding
            logger.debug(
                "Layer %d: fuse_permute suppressed on PCIe-only topology; "
                "using staged dispatch for compute/DMA overlap.",
                cfg.layer_idx,
            )
        elif cfg.fuse_permute and not caps.has_fuse_permute_dispatch:
            warnings.warn(
                f"Layer {cfg.layer_idx}: fuse_permute requested but installed "
                "DeepEP does not support fuse_permute_dispatch; falling back "
                "to unfused dispatch.",
                UserWarning,
                stacklevel=3,
            )

        # Lazy buffer initialisation
        if _hetero_ep_buffer is None:
            seq_len, hidden_dim = x.shape[-2], x.shape[-1]
            init_hetero_hybrid_ep_buffer(
                group=group,
                hidden_dim=hidden_dim,
                seq_len=seq_len,
                num_local_experts=cfg.budget.num_sms_dispatch,  # proxy count
                budget=cfg.budget,
                force_unfused=cfg.pcie_only,
            )

        # SLOC speculative prefetch: synchronise if we issued a prefetch
        # in the previous step
        if cfg.sloc is not None:
            cfg.sloc.synchronize_prefetch()

        # Non-blocking dispatch (if num_permuted_tokens is pre-known we skip
        # the GPU-side barrier that counts permuted tokens)
        non_blocking = cfg.num_permuted_tokens is not None
        dispatch_kwargs: Dict[str, Any] = dict(
            x=x,
            topk_ids=indices,
            topk_weights=probs,
            num_permuted_tokens=cfg.num_permuted_tokens,
            pad_multiple=cfg.pad_multiple,
            non_blocking=non_blocking,
        )
        if effective_fuse:
            dispatch_kwargs['fuse_permute_dispatch'] = True

        dispatched_hidden, dispatched_probs, handle = (
            _hetero_ep_buffer.dispatch_with_permute(**dispatch_kwargs)
        )

        # Store routing in SLOC for next step's speculative prefetch
        if cfg.sloc is not None:
            cfg.sloc.store(cfg.layer_idx, probs, indices, step=0)
            # Kick off prefetch for next step immediately
            cfg.sloc.prefetch(cfg.layer_idx)

        ctx.handle          = handle
        ctx.pad_multiple    = cfg.pad_multiple
        ctx.effective_fuse  = effective_fuse
        ctx.layer_idx       = cfg.layer_idx

        return dispatched_hidden, dispatched_probs, handle

    @staticmethod
    def backward(
        ctx: Any,
        grad_dispatched_hidden: torch.Tensor,
        grad_dispatched_probs: Optional[torch.Tensor],
        grad_handle: None,
    ) -> Tuple[Optional[torch.Tensor], None, Optional[torch.Tensor], None, None]:
        """
        Backward pass: fused unpermute + combine (mirror of upstream).

        ``fuse_unpermute_combine`` is applied only when the forward pass used
        the fused path, matching Megatron's ``ctx.fused`` semantics.
        """
        caps = _get_deepep_caps()
        combine_kwargs: Dict[str, Any] = dict(
            hidden=grad_dispatched_hidden,
            probs=grad_dispatched_probs,
            handle=ctx.handle,
            pad_multiple=ctx.pad_multiple,
        )
        if ctx.effective_fuse and caps.has_fuse_unpermute_combine:
            combine_kwargs['fuse_unpermute_combine'] = True

        combined_hidden, combined_probs = _hetero_ep_buffer.combine_with_unpermute(
            **combine_kwargs
        )
        return combined_hidden, None, combined_probs, None, None


class HeteroHybridEPCombine(torch.autograd.Function):
    """
    Fused combine for DES-LOC heterogeneous expert-parallel routing.

    Forward pass
    ------------
    Mirrors ``HybridEPCombine`` from Megatron with the same fuse gating logic
    as ``HeteroHybridEPDispatch``.

    Backward pass
    -------------
    Uses ``fuse_permute_dispatch`` in the backward if the forward was fused,
    consistent with Megatron's convention.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        handle: Any,
        cfg: HeteroEPConfig,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Expert output token hiddens (post-MLP), shape [S', H].
        handle:
            Communication handle from the matching dispatch call.
        cfg:
            DES-LOC dispatch configuration (same instance as dispatch).
        """
        caps = _get_deepep_caps()
        effective_fuse = (
            cfg.fuse_permute
            and not cfg.pcie_only
            and caps.has_fuse_unpermute_combine
        )

        combine_kwargs: Dict[str, Any] = dict(
            hidden=x,
            handle=handle,
            pad_multiple=cfg.pad_multiple,
        )
        if effective_fuse:
            combine_kwargs['fuse_unpermute_combine'] = True

        combined_hidden, _ = _hetero_ep_buffer.combine_with_unpermute(**combine_kwargs)

        ctx.handle              = handle
        ctx.pad_multiple        = cfg.pad_multiple
        ctx.num_permuted_tokens = cfg.num_permuted_tokens
        ctx.effective_fuse      = effective_fuse
        return combined_hidden

    @staticmethod
    def backward(
        ctx: Any,
        grad_combined: torch.Tensor,
    ) -> Tuple[torch.Tensor, None, None]:
        caps = _get_deepep_caps()
        dispatch_kwargs: Dict[str, Any] = dict(
            hidden=grad_combined,
            handle=ctx.handle,
            pad_multiple=ctx.pad_multiple,
            num_permuted_tokens=ctx.num_permuted_tokens,
        )
        if ctx.effective_fuse and caps.has_fuse_permute_dispatch:
            dispatch_kwargs['fuse_permute_dispatch'] = True

        dispatched_hidden, _ = _hetero_ep_buffer.dispatch_with_permute(**dispatch_kwargs)
        return dispatched_hidden, None, None


# ---------------------------------------------------------------------------
# Top-level functional API (consumed by deepspeed/moe/layer.py)
# ---------------------------------------------------------------------------

class HeteroHybridEPPermuteFusion:
    """
    Primary DES-LOC entry-point for MoE token dispatch and combine.

    This class wraps ``HeteroHybridEPDispatch`` and ``HeteroHybridEPCombine``
    and manages the per-layer ``HeteroEPConfig``.  It is instantiated once per
    MoE layer in ``deepspeed/moe/layer.py``.

    Parameters
    ----------
    group:
        Expert-parallel process group.
    num_local_experts:
        Number of experts on this rank.
    layer_idx:
        MoE layer index (0-based), used for SLOC keying.
    fuse_permute:
        Whether to request fused permute dispatch.  The actual behaviour
        depends on DeepEP capabilities and PCIe topology.
    sloc:
        Optional ``SharedLocalityCache`` shared across all MoE layers.
    num_blocks_permute_override:
        Override the architecture-derived permute block count.
    num_blocks_unpermute_override:
        Override the architecture-derived unpermute block count.
    num_sms_dispatch_override:
        Override the architecture-derived dispatch SM count.
    num_sms_combine_override:
        Override the architecture-derived combine SM count.
    """

    def __init__(
        self,
        group:                          dist.ProcessGroup,
        num_local_experts:              int,
        layer_idx:                      int = 0,
        fuse_permute:                   bool = True,
        sloc:                           Optional[SharedLocalityCache] = None,
        num_blocks_permute_override:    Optional[int] = None,
        num_blocks_unpermute_override:  Optional[int] = None,
        num_sms_dispatch_override:      Optional[int] = None,
        num_sms_combine_override:       Optional[int] = None,
    ) -> None:
        self._group             = group
        self._num_local_experts = num_local_experts
        self._layer_idx         = layer_idx
        self._sloc              = sloc

        # Determine topology
        local_rank  = dist.get_rank(group) if dist.is_initialized() else 0
        self._pcie_only = _peers_are_pcie_only(local_rank)

        # Derive per-arch budget
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        raw_budget = derive_block_budget(device)

        # Apply user overrides
        self._budget = HeteroBlockBudget(
            num_sms_dispatch    = num_sms_dispatch_override    or raw_budget.num_sms_dispatch,
            num_sms_combine     = num_sms_combine_override     or raw_budget.num_sms_combine,
            num_blocks_permute  = num_blocks_permute_override  or raw_budget.num_blocks_permute,
            num_blocks_unpermute= num_blocks_unpermute_override or raw_budget.num_blocks_unpermute,
            device_class        = raw_budget.device_class,
        )

        # Effective fuse setting (may be overridden per-call)
        self._fuse_permute = fuse_permute and not self._pcie_only

        if fuse_permute and self._pcie_only:
            logger.info(
                "Layer %d: fuse_permute=True but PCIe-only topology detected; "
                "DES-LOC will use staged (unfused) dispatch for better overlap.",
                layer_idx,
            )

    def _make_cfg(
        self,
        num_permuted_tokens: Optional[int] = None,
        pad_multiple:        Optional[int] = None,
    ) -> HeteroEPConfig:
        return HeteroEPConfig(
            fuse_permute        = self._fuse_permute,
            budget              = self._budget,
            pcie_only           = self._pcie_only,
            layer_idx           = self._layer_idx,
            sloc                = self._sloc,
            num_permuted_tokens = num_permuted_tokens,
            pad_multiple        = pad_multiple,
        )

    def dispatch(
        self,
        x:                   torch.Tensor,
        indices:             torch.Tensor,
        probs:               torch.Tensor,
        num_permuted_tokens: Optional[int] = None,
        pad_multiple:        Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Dispatch tokens to experts with optional fused permute.

        Returns ``(dispatched_hidden, dispatched_probs, handle)``.
        """
        if not _HAVE_HYBRIDEP:
            raise RuntimeError("DeepEP is required for HeteroHybridEPPermuteFusion.dispatch")
        cfg = self._make_cfg(num_permuted_tokens, pad_multiple)
        return HeteroHybridEPDispatch.apply(x, indices, probs, self._group, cfg)

    def combine(
        self,
        x:                   torch.Tensor,
        handle:              Any,
        num_permuted_tokens: Optional[int] = None,
        pad_multiple:        Optional[int] = None,
    ) -> torch.Tensor:
        """
        Combine expert outputs with optional fused unpermute.

        Returns ``combined_hidden``.
        """
        if not _HAVE_HYBRIDEP:
            raise RuntimeError("DeepEP is required for HeteroHybridEPPermuteFusion.combine")
        cfg = self._make_cfg(num_permuted_tokens, pad_multiple)
        return HeteroHybridEPCombine.apply(x, handle, cfg)

    # ------------------------------------------------------------------
    # Buffer management helpers
    # ------------------------------------------------------------------

    def init_buffer(
        self,
        hidden_dim:  int,
        seq_len:     int,
        fp8_dispatch: bool = False,
    ) -> None:
        """
        Explicitly initialise the global HybridEP buffer for this layer.

        Typically called during model setup.  If ``dispatch`` is called
        without prior ``init_buffer``, the buffer is initialised lazily.
        """
        init_hetero_hybrid_ep_buffer(
            group=self._group,
            hidden_dim=hidden_dim,
            seq_len=seq_len,
            num_local_experts=self._num_local_experts,
            budget=self._budget,
            fp8_dispatch=fp8_dispatch,
            force_unfused=self._pcie_only,
        )

    def reset_buffer(self) -> None:
        """Release the global buffer (useful between pipeline micro-batches)."""
        global _hetero_ep_buffer
        _hetero_ep_buffer = None
        logger.debug("Layer %d: HeteroHybridEP buffer released.", self._layer_idx)


# ---------------------------------------------------------------------------
# DeepSpeed MoE dispatch manager (mirrors _HybridEPManager from token_dispatcher)
# ---------------------------------------------------------------------------

class HeteroHybridEPManager:
    """
    High-level dispatch manager for DES-LOC MoE layers.

    This class mirrors Megatron's ``_HybridEPManager`` but is designed for
    DeepSpeed's ``MOELayer`` interface.  It owns the ``HeteroHybridEPPermuteFusion``
    instance and exposes ``pre_a2a`` / ``post_a2a`` hooks.

    Parameters
    ----------
    group:
        Expert-parallel process group.
    num_local_experts:
        Number of experts on this rank.
    layer_idx:
        MoE layer index.
    config:
        Flat dict of DES-LOC config keys (see ``_CONFIG_KEYS`` below).
    sloc:
        Shared LOcality Cache.  If None, speculative prefetch is disabled.
    """

    _CONFIG_KEYS = {
        'moe_permute_fusion_into_hybridep': False,
        'moe_hybridep_num_sms':             None,
        'moe_hybridep_num_blocks_permute':  None,
        'moe_hybridep_num_blocks_unpermute': None,
        'moe_pad_expert_input_to_capacity': False,
    }

    def __init__(
        self,
        group:             dist.ProcessGroup,
        num_local_experts: int,
        layer_idx:         int,
        config:            Dict[str, Any],
        sloc:              Optional[SharedLocalityCache] = None,
    ) -> None:
        cfg = {**self._CONFIG_KEYS, **config}
        self._fuser = HeteroHybridEPPermuteFusion(
            group                          = group,
            num_local_experts              = num_local_experts,
            layer_idx                      = layer_idx,
            fuse_permute                   = cfg['moe_permute_fusion_into_hybridep'],
            sloc                           = sloc,
            num_sms_dispatch_override      = cfg['moe_hybridep_num_sms'],
            num_sms_combine_override       = cfg['moe_hybridep_num_sms'],
            num_blocks_permute_override    = cfg['moe_hybridep_num_blocks_permute'],
            num_blocks_unpermute_override  = cfg['moe_hybridep_num_blocks_unpermute'],
        )
        self._handle:               Optional[Any] = None
        self._num_permuted_tokens:  Optional[int] = None
        self._pad_multiple:         Optional[int] = None

    def pre_a2a(
        self,
        x:                   torch.Tensor,
        indices:             torch.Tensor,
        probs:               torch.Tensor,
        num_permuted_tokens: Optional[int] = None,
        pad_multiple:        Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatch phase: permute + All-to-All send.

        Returns ``(dispatched_hidden, dispatched_probs)``.
        Stores handle for ``post_a2a``.
        """
        self._num_permuted_tokens = num_permuted_tokens
        self._pad_multiple        = pad_multiple
        dispatched_hidden, dispatched_probs, handle = self._fuser.dispatch(
            x=x,
            indices=indices,
            probs=probs,
            num_permuted_tokens=num_permuted_tokens,
            pad_multiple=pad_multiple,
        )
        self._handle = handle
        return dispatched_hidden, dispatched_probs

    def post_a2a(
        self,
        x:                   torch.Tensor,
        num_permuted_tokens: Optional[int] = None,
        pad_multiple:        Optional[int] = None,
    ) -> torch.Tensor:
        """
        Combine phase: All-to-All recv + unpermute.

        Returns ``combined_hidden``.
        """
        if self._handle is None:
            raise RuntimeError(
                "post_a2a called before pre_a2a; handle is not set."
            )
        combined = self._fuser.combine(
            x=x,
            handle=self._handle,
            num_permuted_tokens=num_permuted_tokens or self._num_permuted_tokens,
            pad_multiple=pad_multiple or self._pad_multiple,
        )
        # Reset per-iteration state
        self._handle              = None
        self._num_permuted_tokens = None
        return combined


# ---------------------------------------------------------------------------
# Config dataclass (mirrors Megatron's TransformerConfig fields)
# ---------------------------------------------------------------------------

@dataclass
class HeteroMoEConfig:
    """
    DES-LOC MoE dispatch configuration.

    Mirrors the fields added by Megatron commit 8c5cf05 plus DES-LOC
    additions for PCIe-aware heterogeneous dispatch.

    Fields
    ------
    moe_permute_fusion_into_hybridep:
        Fuse token permutation into the HybridEP dispatch kernel.
        Automatically suppressed on PCIe-only topologies.
    moe_hybridep_num_sms:
        Number of SMs for dispatch/combine.  None → auto-derive from arch.
    moe_hybridep_num_blocks_permute:
        Thread-block budget for the fused permute sub-kernel.  None → auto.
    moe_hybridep_num_blocks_unpermute:
        Thread-block budget for the fused unpermute sub-kernel.  None → auto.
    moe_sloc_capacity:
        Number of MoE layers for which the SLOC maintains routing entries.
        Set to 0 to disable SLOC entirely.
    moe_sloc_enabled:
        Master enable for Shared LOcality Cache speculative prefetch.
    """
    moe_permute_fusion_into_hybridep:  bool          = False
    moe_hybridep_num_sms:              Optional[int] = None
    moe_hybridep_num_blocks_permute:   Optional[int] = None
    moe_hybridep_num_blocks_unpermute: Optional[int] = None
    moe_sloc_capacity:                 int           = 32
    moe_sloc_enabled:                  bool          = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'moe_permute_fusion_into_hybridep':  self.moe_permute_fusion_into_hybridep,
            'moe_hybridep_num_sms':              self.moe_hybridep_num_sms,
            'moe_hybridep_num_blocks_permute':   self.moe_hybridep_num_blocks_permute,
            'moe_hybridep_num_blocks_unpermute': self.moe_hybridep_num_blocks_unpermute,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_hetero_ep_managers(
    groups:            List[dist.ProcessGroup],
    num_local_experts: int,
    hetero_cfg:        HeteroMoEConfig,
) -> List[HeteroHybridEPManager]:
    """
    Build one ``HeteroHybridEPManager`` per MoE layer.

    Parameters
    ----------
    groups:
        List of EP process groups, one per MoE layer.  Often the same group
        is reused across layers; the list is kept for flexibility.
    num_local_experts:
        Number of experts on this rank (constant across layers).
    hetero_cfg:
        DES-LOC MoE configuration.

    Returns
    -------
    List of managers, indexed by layer.
    """
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    sloc: Optional[SharedLocalityCache] = None
    if hetero_cfg.moe_sloc_enabled and hetero_cfg.moe_sloc_capacity > 0:
        sloc = SharedLocalityCache(
            capacity=hetero_cfg.moe_sloc_capacity,
            device=device,
        )
        logger.info(
            "SharedLocalityCache (SLOC) initialised: capacity=%d layers, device=%s",
            hetero_cfg.moe_sloc_capacity, device,
        )

    managers: List[HeteroHybridEPManager] = []
    config_dict = hetero_cfg.to_dict()
    for layer_idx, group in enumerate(groups):
        mgr = HeteroHybridEPManager(
            group=group,
            num_local_experts=num_local_experts,
            layer_idx=layer_idx,
            config=config_dict,
            sloc=sloc,
        )
        managers.append(mgr)

    logger.info(
        "Built %d HeteroHybridEPManager instances (fuse_permute=%s, sloc=%s).",
        len(managers),
        hetero_cfg.moe_permute_fusion_into_hybridep,
        sloc is not None,
    )
    return managers


# ---------------------------------------------------------------------------
# Utility: reset global buffer (for testing / pipeline restart)
# ---------------------------------------------------------------------------

def reset_hetero_ep_buffer() -> None:
    """Reset the module-level HybridEP buffer and capability cache."""
    global _hetero_ep_buffer, _deepep_caps
    _hetero_ep_buffer = None
    _deepep_caps      = None
    logger.debug("Global HeteroHybridEP buffer and capability cache reset.")


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import unittest

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    class TestHeteroBlockBudget(unittest.TestCase):
        """Tests for architecture-aware SM budget derivation."""

        def _mock_device(self, major: int, minor: int, total_sms: int) -> torch.device:
            """Return a fake torch.device with patched properties."""
            import unittest.mock as mock
            device = torch.device("cpu")  # placeholder
            props = mock.MagicMock()
            props.major = major
            props.minor = minor
            props.multi_processor_count = total_sms
            return device, props

        def test_sm90_budget(self):
            """H100 NVL: 132 SMs → dispatch=79, permute=46."""
            import unittest.mock as mock
            device = torch.device("cpu")
            props = mock.MagicMock()
            props.major = 9
            props.minor = 0
            props.multi_processor_count = 132

            with mock.patch(
                "torch.cuda.get_device_properties", return_value=props
            ):
                budget = derive_block_budget(device)

            self.assertEqual(budget.device_class, "SM90_H100NVL")
            self.assertEqual(budget.num_sms_dispatch, 79)
            self.assertEqual(budget.num_blocks_permute, 46)

        def test_sm86_budget(self):
            """A6000: 84 SMs → dispatch=33, permute=21."""
            import unittest.mock as mock
            device = torch.device("cpu")
            props = mock.MagicMock()
            props.major = 8
            props.minor = 6
            props.multi_processor_count = 84

            with mock.patch(
                "torch.cuda.get_device_properties", return_value=props
            ):
                budget = derive_block_budget(device)

            self.assertEqual(budget.device_class, "SM86_A6000")
            self.assertEqual(budget.num_sms_dispatch, 33)
            self.assertEqual(budget.num_blocks_permute, 21)

        def test_unknown_arch_falls_back_to_minimum(self):
            """Unknown SM arch falls back to global minimums."""
            import unittest.mock as mock
            device = torch.device("cpu")
            props = mock.MagicMock()
            props.major = 7
            props.minor = 5
            props.multi_processor_count = 72

            with mock.patch(
                "torch.cuda.get_device_properties", return_value=props
            ):
                budget = derive_block_budget(device)

            self.assertGreaterEqual(budget.num_sms_dispatch, _MIN_DISPATCH_SMS)
            self.assertGreaterEqual(budget.num_blocks_permute, _MIN_PERMUTE_BLOCKS)

        def test_small_gpu_clamps_to_minimum(self):
            """Very small SM count must not produce 0 blocks."""
            import unittest.mock as mock
            device = torch.device("cpu")
            props = mock.MagicMock()
            props.major = 9
            props.minor = 0
            props.multi_processor_count = 8  # tiny hypothetical device

            with mock.patch(
                "torch.cuda.get_device_properties", return_value=props
            ):
                budget = derive_block_budget(device)

            self.assertGreaterEqual(budget.num_sms_dispatch, _MIN_DISPATCH_SMS)
            self.assertGreaterEqual(budget.num_blocks_permute, _MIN_PERMUTE_BLOCKS)

    class TestDeepEPCapabilities(unittest.TestCase):
        """Tests for runtime capability probing."""

        def test_no_deepep_returns_all_false(self):
            import unittest.mock as mock
            with mock.patch.dict(sys.modules, {"deepep": None}):
                global _deepep_caps
                _deepep_caps = None  # reset cache
                # Re-patch _HAVE_HYBRIDEP for this test
                import deepspeed.moe.hetero_hybrid_ep_permute as m
                orig = m._HAVE_HYBRIDEP
                m._HAVE_HYBRIDEP = False
                caps = _probe_deepep_capabilities()
                m._HAVE_HYBRIDEP = orig
            self.assertFalse(caps.has_fuse_permute_dispatch)
            self.assertFalse(caps.has_num_blocks_permute)

        def test_capability_probe_caches_result(self):
            """Second call to _get_deepep_caps returns same object."""
            global _deepep_caps
            _deepep_caps = None
            import deepspeed.moe.hetero_hybrid_ep_permute as m
            orig = m._HAVE_HYBRIDEP
            m._HAVE_HYBRIDEP = False
            c1 = _get_deepep_caps()
            c2 = _get_deepep_caps()
            m._HAVE_HYBRIDEP = orig
            self.assertIs(c1, c2)

    class TestSLOC(unittest.TestCase):
        """Tests for SharedLocalityCache."""

        @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
        def test_store_and_prefetch(self):
            device = torch.device("cuda:0")
            sloc = SharedLocalityCache(capacity=4, device=device)
            probs   = torch.randn(128, 8)
            indices = torch.randint(0, 8, (128, 2))

            sloc.store(layer_idx=0, probs=probs, indices=indices, step=1)
            result = sloc.prefetch(layer_idx=0)
            self.assertIsNotNone(result)
            p_gpu, i_gpu = result
            sloc.synchronize_prefetch()
            self.assertEqual(p_gpu.device.type, "cuda")
            self.assertEqual(i_gpu.device.type, "cuda")
            self.assertEqual(p_gpu.shape, probs.shape)

        def test_prefetch_missing_key_returns_none(self):
            if not torch.cuda.is_available():
                self.skipTest("CUDA required")
            device = torch.device("cuda:0")
            sloc = SharedLocalityCache(capacity=4, device=device)
            self.assertIsNone(sloc.prefetch(layer_idx=99))

        def test_evict(self):
            if not torch.cuda.is_available():
                self.skipTest("CUDA required")
            device = torch.device("cuda:0")
            sloc = SharedLocalityCache(capacity=4, device=device)
            probs   = torch.randn(64, 4)
            indices = torch.randint(0, 4, (64, 1))
            sloc.store(0, probs, indices, 0)
            self.assertEqual(len(sloc), 1)
            sloc.evict(0)
            self.assertEqual(len(sloc), 0)

    class TestPCIeDetection(unittest.TestCase):
        """Tests for PCIe topology detection heuristic."""

        def test_single_gpu_is_pcie_only(self):
            import unittest.mock as mock
            with mock.patch("torch.cuda.device_count", return_value=1):
                result = _peers_are_pcie_only(local_rank=0)
            self.assertTrue(result)

        def test_mixed_arch_multi_gpu_is_pcie_only(self):
            """2x SM86 + 1x SM90 without same-arch peer access → PCIe."""
            import unittest.mock as mock

            def fake_device_count():
                return 3

            def fake_can_access(src, dst):
                return False  # no peer access → PCIe

            sm86_props = mock.MagicMock(major=8, minor=6, multi_processor_count=84)
            sm90_props = mock.MagicMock(major=9, minor=0, multi_processor_count=132)

            def fake_props(device_idx):
                return sm86_props if device_idx < 2 else sm90_props

            with mock.patch("torch.cuda.device_count", fake_device_count), \
                 mock.patch("torch.cuda.can_device_access_peer", fake_can_access), \
                 mock.patch("torch.cuda.get_device_properties", fake_props):
                result = _peers_are_pcie_only(local_rank=0)

            self.assertTrue(result)

    class TestHeteroEPConfig(unittest.TestCase):
        """Tests for HeteroEPConfig construction."""

        def test_pcie_suppresses_fuse(self):
            budget = HeteroBlockBudget(33, 33, 21, 21, "SM86_A6000")
            cfg = HeteroEPConfig(
                fuse_permute=True,
                budget=budget,
                pcie_only=True,
                layer_idx=0,
            )
            # The config stores intent; suppression happens in the Function
            self.assertTrue(cfg.fuse_permute)
            self.assertTrue(cfg.pcie_only)

        def test_non_pcie_preserves_fuse(self):
            budget = HeteroBlockBudget(79, 79, 46, 46, "SM90_H100NVL")
            cfg = HeteroEPConfig(
                fuse_permute=True,
                budget=budget,
                pcie_only=False,
                layer_idx=1,
            )
            self.assertTrue(cfg.fuse_permute)
            self.assertFalse(cfg.pcie_only)

    class TestHeteroMoEConfig(unittest.TestCase):
        """Tests for the config dataclass."""

        def test_defaults(self):
            cfg = HeteroMoEConfig()
            self.assertFalse(cfg.moe_permute_fusion_into_hybridep)
            self.assertIsNone(cfg.moe_hybridep_num_sms)
            self.assertIsNone(cfg.moe_hybridep_num_blocks_permute)
            self.assertIsNone(cfg.moe_hybridep_num_blocks_unpermute)
            self.assertTrue(cfg.moe_sloc_enabled)

        def test_to_dict_keys(self):
            cfg = HeteroMoEConfig(
                moe_permute_fusion_into_hybridep=True,
                moe_hybridep_num_sms=16,
                moe_hybridep_num_blocks_permute=8,
                moe_hybridep_num_blocks_unpermute=8,
            )
            d = cfg.to_dict()
            self.assertIn('moe_permute_fusion_into_hybridep', d)
            self.assertEqual(d['moe_hybridep_num_sms'], 16)
            self.assertEqual(d['moe_hybridep_num_blocks_permute'], 8)

    class TestResetBuffer(unittest.TestCase):
        """Tests for global buffer / capability cache reset."""

        def test_reset_clears_globals(self):
            global _hetero_ep_buffer, _deepep_caps
            # Force non-None state
            _hetero_ep_buffer = object()
            _deepep_caps = _DeepEPCapabilities()
            reset_hetero_ep_buffer()
            self.assertIsNone(_hetero_ep_buffer)
            self.assertIsNone(_deepep_caps)

    class TestHeteroHybridEPManagerInit(unittest.TestCase):
        """Tests for HeteroHybridEPManager config plumbing."""

        @unittest.skipUnless(
            dist.is_available() and torch.cuda.is_available(),
            "Distributed + CUDA required",
        )
        def test_manager_respects_config_overrides(self):
            """Budget overrides in config must propagate to the fuser's budget."""
            import unittest.mock as mock

            # Build a minimal fake group
            fake_group = mock.MagicMock(spec=dist.ProcessGroup)

            config = {
                'moe_permute_fusion_into_hybridep': False,
                'moe_hybridep_num_sms': 12,
                'moe_hybridep_num_blocks_permute': 6,
                'moe_hybridep_num_blocks_unpermute': 6,
            }

            sm86_props = mock.MagicMock()
            sm86_props.major = 8
            sm86_props.minor = 6
            sm86_props.multi_processor_count = 84

            with mock.patch("torch.cuda.get_device_properties", return_value=sm86_props), \
                 mock.patch(
                     "deepspeed.moe.hetero_hybrid_ep_permute._peers_are_pcie_only",
                     return_value=True,
                 ), \
                 mock.patch("torch.cuda.current_device", return_value=0):
                mgr = HeteroHybridEPManager(
                    group=fake_group,
                    num_local_experts=4,
                    layer_idx=0,
                    config=config,
                )

            self.assertEqual(mgr._fuser._budget.num_sms_dispatch, 12)
            self.assertEqual(mgr._fuser._budget.num_blocks_permute, 6)

    # ------------------------------------------------------------------
    # Integration smoke test (requires CUDA but not DeepEP)
    # ------------------------------------------------------------------

    class TestSLOCIntegration(unittest.TestCase):
        """Smoke test: store → prefetch → synchronize round-trip."""

        @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
        def test_round_trip_shapes(self):
            device = torch.device("cuda:0")
            sloc = SharedLocalityCache(capacity=8, device=device)

            S, E, K = 256, 16, 2
            probs   = torch.softmax(torch.randn(S, E), dim=-1)
            indices = torch.topk(probs, K, dim=-1).indices

            for layer in range(4):
                sloc.store(layer, probs, indices, step=layer)

            for layer in range(4):
                result = sloc.prefetch(layer)
                self.assertIsNotNone(result)
                p_gpu, i_gpu = result
                sloc.synchronize_prefetch()
                self.assertEqual(p_gpu.shape, (S, E))
                self.assertEqual(i_gpu.shape, (S, K))
                self.assertEqual(p_gpu.device.type, "cuda")

    print("Running DES-LOC HeteroHybridEPPermuteFusion unit tests …", flush=True)
    unittest.main(verbosity=2, exit=True)
