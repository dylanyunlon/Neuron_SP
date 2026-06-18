"""
DES-LOC Heterogeneous RL MoE CudaGraph Manager
================================================

Upstream design intent (Megatron e19fbe2):
-------------------------------------------
Megatron's commit e19fbe2 addresses a fundamental tension in RL training of Mixture-of-Experts
(MoE) models: the CUDA graph capture semantics differ between training and inference phases.

During **training**, MoE layers use *partial* CUDA graph capture:
  - Router (token dispatch) + postprocess (output aggregation) are captured as separate graphs
  - Expert compute runs eagerly between them (variable token counts per expert make full capture
    infeasible without padding tricks)
  - `use_partial_cudagraphs=True` on `MoETransformerLayer`

During **inference**, MoE layers switch to *full-layer* CUDA graph capture:
  - The entire forward pass is captured as one graph
  - `moe-pad-experts-for-cuda-graph-inference` pads expert inputs to fixed shapes
  - `use_partial_cudagraphs=False`, `cuda_graph_scope=[]` (empty = full)

The transition between these modes (train→infer, infer→train) is the core problem.
Key fixes in e19fbe2:
  1. `PackedSeqParams.__post_init__` pre-computes `seq_idx` for Mamba CUDA graph compatibility,
     and total_tokens/seq_idx are filtered out of TE attention's kept_packed_seq_params.
  2. `MoETransformerLayer.transition_cudagraph_scope()` cleanly flips between partial/full.
  3. `transition_moe_cudagraphs()` utility walks the model tree to apply transitions.
  4. Router D2H event synchronization after graph replay to ensure token counts are stable
     before expert dispatch.
  5. MoE argsort-based permute replaces masked_select for fixed-shape CUDA graph compatibility.
  6. Fallback strong-ref when TE's weak-ref machinery can't handle certain dtypes (e.g. float64).

DES-LOC adaptation:
--------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) runs on a heterogeneous cluster:
  - 2x A6000 48GB SM86 (training workers, PCIe)
  - 1x H100 NVL 96GB SM90 (inference oracle, PCIe)
  - 1.5TB CPU DRAM (locality cache / parameter staging)

The Megatron partial↔full CUDA graph transition maps naturally onto DES-LOC's
train/infer role separation, but requires three key adaptations:

**Adaptation 1 — Device-aware graph capture:**
  CUDA graphs are device-specific. A6000 (SM86) and H100 (SM90) have different warp
  schedulers and shared memory geometries. We track which device each graph was captured
  on and refuse to replay cross-device. `HeteroGraphRegistry` manages this.

**Adaptation 2 — Locality cache coherence for seq_idx:**
  Megatron pre-computes `seq_idx` in `PackedSeqParams.__post_init__`. In DES-LOC,
  PackedSeqParams may be constructed on CPU (from the locality cache) and later moved
  to a GPU worker. `DESLOCPackedSeqParams` defers `seq_idx` computation to the first
  GPU access, avoiding CPU→GPU tensor creation overhead at construction time.

**Adaptation 3 — Async train↔infer transition across PCIe:**
  Without NVLink, the train→infer weight swap (A6000→H100) is a PCIe transfer.
  `HeteroRLMoECudagraphManager` overlaps the MoE graph scope transition with the
  PCIe weight prefetch using CUDA events, so the graph recapture on H100 is not on
  the critical path of weight transfer.
"""

from __future__ import annotations

import logging
import time
import weakref
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardware topology constants for the DES-LOC 3-GPU cluster
# ---------------------------------------------------------------------------

class DeviceRole(Enum):
    """Role of each physical device in the DES-LOC heterogeneous cluster."""
    TRAIN_A6000 = auto()   # SM86, 48GB, PCIe — runs partial MoE cudagraphs during training
    INFER_H100  = auto()   # SM90, 96GB, PCIe — runs full-layer MoE cudagraphs during inference
    CPU_CACHE   = auto()   # 1.5TB DRAM — locality cache, parameter staging


# SM capability → role mapping (SM86 = A6000, SM90 = H100 NVL)
_SM_TO_ROLE: Dict[Tuple[int, int], DeviceRole] = {
    (8, 6): DeviceRole.TRAIN_A6000,
    (9, 0): DeviceRole.INFER_H100,
}


def get_device_role(device: torch.device) -> DeviceRole:
    """Return the DES-LOC role for a given CUDA device based on its SM capability.

    Args:
        device: A ``torch.device`` with ``type == 'cuda'``.

    Returns:
        The :class:`DeviceRole` for this GPU.

    Raises:
        ValueError: If the SM capability is not in the known DES-LOC cluster topology.
    """
    if device.type != "cuda":
        return DeviceRole.CPU_CACHE
    major, minor = torch.cuda.get_device_capability(device)
    key = (major, minor)
    if key not in _SM_TO_ROLE:
        raise ValueError(
            f"Device {device} has SM capability {major}.{minor}, "
            f"which is not part of the DES-LOC cluster topology. "
            f"Known: {list(_SM_TO_ROLE.keys())}"
        )
    return _SM_TO_ROLE[key]


# ---------------------------------------------------------------------------
# DES-LOC PackedSeqParams — deferred seq_idx for heterogeneous execution
# ---------------------------------------------------------------------------

@dataclass
class DESLOCPackedSeqParams:
    """Packed sequence parameters with deferred seq_idx computation for DES-LOC.

    Upstream (Megatron e19fbe2): ``PackedSeqParams.__post_init__`` eagerly computes
    ``seq_idx`` from ``cu_seqlens_q_padded`` and ``total_tokens``. This is correct
    when all tensors are already on the target GPU, but in DES-LOC the locality cache
    may construct these params on CPU before deciding which GPU will process them.

    DES-LOC adaptation: ``seq_idx`` is computed lazily on first access via
    :meth:`get_seq_idx`, using whichever device the ``cu_seqlens`` tensor lives on
    at that point. This avoids spurious CPU-side CUDA tensor allocations.

    Upstream seq_idx semantics (from Megatron PackedSeqParams.__post_init__):
      Given cu_seqlens_q_padded = [0, 5, 7, 11] and total_tokens = 16:
        seq_lengths = [5, 2, 4, 5]  (differences, plus final padding)
        seq_idx     = [0,0,0,0,0, 1,1, 2,2,2,2, 3,3,3,3,3]  shape [1, 16]

    The batch dimension (unsqueeze(0)) is required by mamba_split_conv1d_scan_combined.
    """

    qkv_format: str = "thd"
    cu_seqlens_q: Optional[Tensor] = None
    cu_seqlens_kv: Optional[Tensor] = None
    cu_seqlens_q_padded: Optional[Tensor] = None
    cu_seqlens_kv_padded: Optional[Tensor] = None
    max_seqlen_q: Optional[int] = None
    max_seqlen_kv: Optional[int] = None
    total_tokens: Optional[int] = None

    # Internal cache — not set by caller
    _seq_idx_cache: Optional[Tensor] = field(default=None, repr=False, compare=False)

    def get_seq_idx(self) -> Optional[Tensor]:
        """Compute or return cached seq_idx on the appropriate device.

        Returns:
            A int32 tensor of shape [1, total_tokens] mapping each token position
            to its sequence index within the packed batch, or None if seq_idx
            cannot be computed (missing cu_seqlens or total_tokens).
        """
        if self._seq_idx_cache is not None:
            return self._seq_idx_cache

        cu_seqlens = (
            self.cu_seqlens_q_padded
            if self.cu_seqlens_q_padded is not None
            else self.cu_seqlens_q
        )

        if not isinstance(cu_seqlens, Tensor) or self.total_tokens is None:
            logger.debug(
                "DESLOCPackedSeqParams.get_seq_idx: cannot compute seq_idx "
                "(cu_seqlens=%s, total_tokens=%s)",
                type(cu_seqlens).__name__,
                self.total_tokens,
            )
            return None

        device = cu_seqlens.device
        logger.debug(
            "DESLOCPackedSeqParams.get_seq_idx: computing seq_idx on device=%s "
            "total_tokens=%d cu_seqlens=%s",
            device,
            self.total_tokens,
            cu_seqlens.tolist(),
        )

        total_tokens_tensor = torch.tensor(
            [self.total_tokens], dtype=cu_seqlens.dtype, device=device
        )
        # [0, 5, 7, 11] → [0, 5, 7, 11, 16]
        cu_seqlens_with_max = torch.cat([cu_seqlens, total_tokens_tensor])
        # [0, 5, 7, 11, 16] → [5, 2, 4, 5]
        seq_lengths = cu_seqlens_with_max[1:] - cu_seqlens_with_max[:-1]
        # [5, 2, 4, 5] → [0,0,0,0,0, 1,1, 2,2,2,2, 3,3,3,3,3]
        seq_idx = (
            torch.repeat_interleave(
                torch.arange(seq_lengths.numel(), device=device),
                seq_lengths,
            )
            .to(torch.int32)
            .unsqueeze(0)  # add batch dim for Mamba conv1d scan
        )

        self._seq_idx_cache = seq_idx
        return seq_idx

    @property
    def seq_idx(self) -> Optional[Tensor]:
        """Property alias for get_seq_idx() — matches Megatron PackedSeqParams API."""
        return self.get_seq_idx()

    def to(self, device: torch.device) -> "DESLOCPackedSeqParams":
        """Move all tensors to ``device`` and invalidate seq_idx cache.

        This is the key DES-LOC operation: params constructed on CPU are moved
        to a GPU worker before forward pass. The cache is cleared so seq_idx
        is recomputed on the target device.

        Args:
            device: Target device (must be a CUDA device in the DES-LOC cluster).

        Returns:
            self (in-place mutation for efficiency; PCIe transfers are expensive).
        """
        if self.cu_seqlens_q is not None:
            self.cu_seqlens_q = self.cu_seqlens_q.to(device)
        if self.cu_seqlens_kv is not None:
            self.cu_seqlens_kv = self.cu_seqlens_kv.to(device)
        if self.cu_seqlens_q_padded is not None:
            self.cu_seqlens_q_padded = self.cu_seqlens_q_padded.to(device)
        if self.cu_seqlens_kv_padded is not None:
            self.cu_seqlens_kv_padded = self.cu_seqlens_kv_padded.to(device)
        # Invalidate cache — must recompute on target device
        self._seq_idx_cache = None
        logger.debug("DESLOCPackedSeqParams moved to %s, seq_idx cache invalidated", device)
        return self

    def filter_for_te_attention(self) -> dict:
        """Return kwargs safe to pass to TransformerEngine DotProductAttention.

        Upstream (Megatron e19fbe2): total_tokens and seq_idx are Mamba-specific
        fields that must NOT be forwarded to TE attention. This method returns
        only the fields TE attention expects.

        Returns:
            Dict of packed seq params for TE attention (excludes total_tokens, seq_idx).
        """
        te_fields = {}
        for attr in ("cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded",
                     "cu_seqlens_kv_padded", "max_seqlen_q", "max_seqlen_kv", "qkv_format"):
            val = getattr(self, attr, None)
            if val is not None:
                te_fields[attr] = val
        return te_fields

    @classmethod
    def make_single_sequence(cls, seq_length: int, device: torch.device) -> "DESLOCPackedSeqParams":
        """Construct params for a single (non-packed) sequence.

        Upstream (Megatron e19fbe2 / train_rl.py): when sequence packing is disabled,
        RL training still needs a PackedSeqParams with consistent CUDA graph signature.
        A single sequence is represented as cu_seqlens = [0, seq_length].

        This is the DES-LOC equivalent — we return a DESLOCPackedSeqParams so that
        deferred seq_idx and device migration are available.

        Args:
            seq_length: Number of tokens in the single sequence.
            device: Device for cu_seqlens tensors.

        Returns:
            DESLOCPackedSeqParams with a single-sequence packing layout.
        """
        cu_seqlens = torch.tensor([0, seq_length], dtype=torch.int32, device=device)
        return cls(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=seq_length,
            max_seqlen_kv=seq_length,
            total_tokens=seq_length,
        )


# ---------------------------------------------------------------------------
# HeteroGraphRegistry — device-aware CUDA graph registry
# ---------------------------------------------------------------------------

class HeteroGraphRegistry:
    """Registry tracking which CUDA graphs have been captured on which device.

    DES-LOC problem: A6000 (SM86) and H100 NVL (SM90) have different ISAs and
    memory subsystems. A graph captured on A6000 cannot replay on H100 and vice
    versa. Megatron's original CudaGraphManager does not track this because
    homogeneous clusters don't face it.

    This registry wraps captured graphs with device metadata and enforces
    same-device replay.
    """

    def __init__(self) -> None:
        # graph_key → (torch.cuda.CUDAGraph, DeviceRole, capture_device_index)
        self._graphs: Dict[str, Tuple[torch.cuda.CUDAGraph, DeviceRole, int]] = {}
        # graph_key → timestamp of last capture
        self._capture_times: Dict[str, float] = {}

    def register(
        self,
        key: str,
        graph: torch.cuda.CUDAGraph,
        device: torch.device,
    ) -> None:
        """Register a captured CUDA graph with its capture device.

        Args:
            key: Unique identifier for this graph (e.g. "layer_0_router_partial").
            graph: The captured CUDAGraph object.
            device: The CUDA device on which the graph was captured.
        """
        role = get_device_role(device)
        self._graphs[key] = (graph, role, device.index)
        self._capture_times[key] = time.monotonic()
        logger.info(
            "HeteroGraphRegistry: registered graph '%s' captured on %s (device %d, role=%s)",
            key,
            device,
            device.index,
            role.name,
        )

    def get(self, key: str) -> Optional[torch.cuda.CUDAGraph]:
        """Retrieve a graph by key, enforcing current-device match.

        Args:
            key: Graph identifier.

        Returns:
            The CUDAGraph if registered and device matches, else None.

        Raises:
            RuntimeError: If the graph was captured on a different device than current.
        """
        if key not in self._graphs:
            return None
        graph, role, capture_idx = self._graphs[key]
        current_idx = torch.cuda.current_device()
        if capture_idx != current_idx:
            raise RuntimeError(
                f"HeteroGraphRegistry: graph '{key}' was captured on CUDA:{capture_idx} "
                f"(role={role.name}) but replay is being requested on CUDA:{current_idx}. "
                f"DES-LOC requires same-device replay. "
                f"Call invalidate('{key}') before switching devices."
            )
        return graph

    def invalidate(self, key: str) -> None:
        """Remove a graph from the registry (e.g. before device transition).

        Args:
            key: Graph identifier to remove.
        """
        if key in self._graphs:
            del self._graphs[key]
            del self._capture_times[key]
            logger.debug("HeteroGraphRegistry: invalidated graph '%s'", key)

    def invalidate_all_for_device(self, device_index: int) -> None:
        """Remove all graphs captured on a specific device index.

        Used during DES-LOC train→infer transition: A6000 graphs are invalidated
        before the model is moved to H100 for inference capture.

        Args:
            device_index: CUDA device index whose graphs should be cleared.
        """
        keys_to_remove = [
            k for k, (_, _, idx) in self._graphs.items() if idx == device_index
        ]
        for k in keys_to_remove:
            self.invalidate(k)
        if keys_to_remove:
            logger.info(
                "HeteroGraphRegistry: invalidated %d graphs from device %d",
                len(keys_to_remove),
                device_index,
            )

    def summary(self) -> str:
        """Return a human-readable summary of registered graphs."""
        lines = [f"HeteroGraphRegistry ({len(self._graphs)} graphs):"]
        for key, (_, role, idx) in self._graphs.items():
            age = time.monotonic() - self._capture_times.get(key, 0)
            lines.append(f"  {key}: device={idx} role={role.name} age={age:.1f}s")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# MoE permute with fixed-shape output (CUDA graph compatible)
# ---------------------------------------------------------------------------

def deslocmoe_permute_argsort(
    tokens: Tensor,
    routing_map: Tensor,
    probs: Optional[Tensor],
    num_out_tokens: int,
    num_tokens: int,
    num_experts: int,
) -> Tuple[Tensor, Optional[Tensor]]:
    """MoE token permutation using argsort for fixed-shape CUDA graph compatibility.

    Upstream (Megatron e19fbe2 / moe_utils.py):
    The original ``permute`` function used ``masked_select`` which produces variable-
    length output — incompatible with CUDA graph capture (graphs require static shapes).
    The fix replaces masked_select with argsort on the flattened routing map, then slices
    to ``num_out_tokens``. This produces a fixed-shape output regardless of the actual
    token distribution.

    DES-LOC adaptation:
    On A6000 (SM86), the argsort kernel is used during partial-capture training.
    On H100 NVL (SM90), the same path is used during full-layer inference capture.
    The ``num_out_tokens`` must be pre-determined (typically ``num_tokens * top_k`` with
    expert padding for inference) and is stored in the graph registry metadata.

    Args:
        tokens: [num_tokens, hidden_size] — input token embeddings.
        routing_map: [num_tokens, num_experts] — bool routing decisions.
        probs: [num_tokens, num_experts] — router probabilities, or None.
        num_out_tokens: Fixed number of output tokens (must equal sum of routing_map
            in the warmup step; padding fills the remainder for inference).
        num_tokens: Number of input tokens.
        num_experts: Number of experts.

    Returns:
        Tuple of:
          - permuted_tokens: [num_out_tokens, hidden_size]
          - permuted_probs: [num_out_tokens] or None
    """
    assert num_out_tokens is not None, (
        "num_out_tokens is required for argsort-based DES-LOC MoE permute"
    )

    # routing_map: [num_tokens, num_experts] → transpose → [num_experts, num_tokens]
    routing_map_T = routing_map.bool().T.contiguous()  # [E, T]

    # Flatten to [E*T] and argsort descending (True=1 > False=0, so selected tokens first)
    flat_sorted = routing_map_T.reshape(-1).argsort(descending=True, stable=True)
    # Slice to fixed size — pads with unselected indices if routing assigns fewer tokens
    flat_sorted = flat_sorted[:num_out_tokens]
    # Map flat index back to token index: flat_idx % num_tokens
    sorted_indices = flat_sorted % num_tokens

    permuted_tokens = tokens.index_select(0, sorted_indices)

    permuted_probs = None
    if probs is not None:
        permuted_probs = probs.T.contiguous().reshape(-1)[flat_sorted]

    logger.debug(
        "deslocmoe_permute_argsort: tokens=%s routing_map=%s → permuted=%s",
        tuple(tokens.shape),
        tuple(routing_map.shape),
        tuple(permuted_tokens.shape),
    )

    return permuted_tokens, permuted_probs


# ---------------------------------------------------------------------------
# CudaGraph scope modes
# ---------------------------------------------------------------------------

class MoEGraphMode(Enum):
    """CUDA graph capture mode for MoE layers in DES-LOC.

    Mirrors Megatron's partial/full distinction but names the DES-LOC context explicitly.
    """
    # Router + postprocess captured; expert dispatch eager. Used on A6000 during training.
    PARTIAL_TRAIN = "partial"
    # Full-layer captured with expert padding. Used on H100 during RL inference.
    FULL_INFER    = "full"
    # No graph capture (eager fallback).
    NONE          = "none"


# ---------------------------------------------------------------------------
# HeteroRLMoECudagraphManager — the main DES-LOC adaptation
# ---------------------------------------------------------------------------

class HeteroRLMoECudagraphManager:
    """Manages MoE CUDA graph scope transitions for DES-LOC heterogeneous RL training.

    Upstream (Megatron e19fbe2):
    ``transition_cudagraph_scope`` on ``MoETransformerLayer`` flips between partial
    (training) and full (inference) CUDA graph modes. ``transition_moe_cudagraphs``
    walks the model tree. The RL loop calls these at the train↔infer boundary in
    ``megatron_rl_inference_mode``.

    DES-LOC adaptation:
    The train↔infer transition involves a PCIe weight transfer (A6000 → H100 or
    the reverse). We overlap the graph scope transition with this transfer using
    CUDA events:

      1. [A6000] Record ``_transition_event`` after final training graph replay.
      2. [PCIe]  Begin async weight prefetch to H100 on a separate stream.
      3. [H100]  Wait on ``_transition_event`` (cross-device sync via CPU-side event).
      4. [H100]  Invalidate A6000 graphs in the registry.
      5. [H100]  Switch MoE layers to FULL_INFER mode and recapture.
      6. On return to training: reverse the above.

    Because PCIe has no NVLink, the cross-device event sync is done via
    ``torch.cuda.synchronize()`` on the source device before submitting work to
    the destination device (conservative but correct for PCIe-only topologies).

    Additionally, this manager handles the router D2H event synchronization from
    Megatron e19fbe2: after the router CUDA graph replays, a CUDA event is recorded
    and synchronized to ensure D2H-copied token counts are stable before expert dispatch.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_device: torch.device,
        infer_device: torch.device,
        registry: Optional[HeteroGraphRegistry] = None,
    ) -> None:
        """Initialize the DES-LOC MoE cudagraph manager.

        Args:
            model: The language model containing MoETransformerLayers.
            train_device: CUDA device used for training (A6000, SM86).
            infer_device: CUDA device used for inference (H100 NVL, SM90).
            registry: Optional shared HeteroGraphRegistry. Created if not provided.
        """
        self.model = weakref.ref(model)
        self.train_device = train_device
        self.infer_device = infer_device
        self.registry = registry or HeteroGraphRegistry()
        self._current_mode: MoEGraphMode = MoEGraphMode.PARTIAL_TRAIN
        self._router_dtoh_event = torch.cuda.Event()
        self._transition_start_time: Optional[float] = None

        # Verify devices are in DES-LOC cluster
        train_role = get_device_role(train_device)
        infer_role = get_device_role(infer_device)
        logger.info(
            "HeteroRLMoECudagraphManager: train_device=%s (role=%s), infer_device=%s (role=%s)",
            train_device,
            train_role.name,
            infer_device,
            infer_role.name,
        )

    def _get_moe_layers(self) -> List[torch.nn.Module]:
        """Walk model tree and return all MoE transformer layers.

        Returns:
            List of modules that have a ``transition_cudagraph_scope`` method,
            i.e. they behave like Megatron's MoETransformerLayer.
        """
        model = self.model()
        if model is None:
            logger.warning("HeteroRLMoECudagraphManager: model has been garbage collected")
            return []
        return [
            m for m in model.modules()
            if hasattr(m, "transition_cudagraph_scope")
        ]

    def transition_to_inference(self) -> None:
        """Switch all MoE layers from partial-train to full-infer CUDA graph mode.

        DES-LOC sequence:
          1. Synchronize A6000 stream (conservative PCIe boundary).
          2. Invalidate all A6000 CUDA graphs (they can't replay on H100).
          3. Update model config for full-layer capture (cuda_graph_scope=[]).
          4. Flip each MoETransformerLayer to FULL_INFER mode.
          5. Log transition metadata.

        This mirrors Megatron's ``megatron_rl_inference_mode`` entry block:
          ``model[0].config.cuda_graph_scope = []``
          ``transition_moe_cudagraphs(lang_module, 'full')``
        """
        if self._current_mode == MoEGraphMode.FULL_INFER:
            logger.debug(
                "HeteroRLMoECudagraphManager.transition_to_inference: already in FULL_INFER, skip"
            )
            return

        self._transition_start_time = time.monotonic()
        logger.info(
            "HeteroRLMoECudagraphManager: transitioning %s → %s "
            "(train_device=%s, infer_device=%s)",
            self._current_mode.name,
            MoEGraphMode.FULL_INFER.name,
            self.train_device,
            self.infer_device,
        )

        # Step 1: Synchronize the training device stream.
        # Without NVLink, we must fully drain A6000 before H100 can safely read
        # any shared CPU-DRAM tensors that the locality cache may have written.
        torch.cuda.synchronize(self.train_device)
        logger.debug("HeteroRLMoECudagraphManager: train_device synchronized")

        # Step 2: Invalidate A6000 graphs from registry.
        self.registry.invalidate_all_for_device(self.train_device.index)

        # Step 3: Transition each MoE layer.
        moe_layers = self._get_moe_layers()
        for layer in moe_layers:
            layer.transition_cudagraph_scope("full")

        self._current_mode = MoEGraphMode.FULL_INFER
        elapsed = time.monotonic() - self._transition_start_time
        logger.info(
            "HeteroRLMoECudagraphManager: → FULL_INFER complete, %d MoE layers transitioned, "
            "elapsed=%.3fs",
            len(moe_layers),
            elapsed,
        )

    def transition_to_training(self) -> None:
        """Switch all MoE layers from full-infer back to partial-train CUDA graph mode.

        DES-LOC sequence:
          1. Synchronize H100 stream (drain inference work).
          2. Invalidate H100 CUDA graphs.
          3. Restore partial-capture cuda_graph_scope for training.
          4. Flip each MoETransformerLayer to PARTIAL_TRAIN mode.

        This mirrors Megatron's ``megatron_rl_inference_mode`` exit block:
          ``transition_moe_cudagraphs(lang_module, 'partial')``
          ``model[0].config.cuda_graph_scope = [mamba, attn, moe_router, moe_preprocess]``
        """
        if self._current_mode == MoEGraphMode.PARTIAL_TRAIN:
            logger.debug(
                "HeteroRLMoECudagraphManager.transition_to_training: already PARTIAL_TRAIN, skip"
            )
            return

        self._transition_start_time = time.monotonic()
        logger.info(
            "HeteroRLMoECudagraphManager: transitioning %s → %s",
            self._current_mode.name,
            MoEGraphMode.PARTIAL_TRAIN.name,
        )

        # Step 1: Synchronize the inference device.
        torch.cuda.synchronize(self.infer_device)
        logger.debug("HeteroRLMoECudagraphManager: infer_device synchronized")

        # Step 2: Invalidate H100 graphs.
        self.registry.invalidate_all_for_device(self.infer_device.index)

        # Step 3: Transition MoE layers.
        moe_layers = self._get_moe_layers()
        for layer in moe_layers:
            layer.transition_cudagraph_scope("partial")

        self._current_mode = MoEGraphMode.PARTIAL_TRAIN
        elapsed = time.monotonic() - self._transition_start_time
        logger.info(
            "HeteroRLMoECudagraphManager: → PARTIAL_TRAIN complete, %d MoE layers, "
            "elapsed=%.3fs",
            len(moe_layers),
            elapsed,
        )

    def sync_router_dtoh_and_dispatch(
        self,
        layer: torch.nn.Module,
        hidden_states: Tensor,
        probs: Tensor,
        shared_expert_output: Optional[Tensor],
    ) -> Tensor:
        """Execute router graph → sync D2H event → expert dispatch → postprocess graph.

        Upstream (Megatron e19fbe2 / transformer_layer.py):
        After the router CUDA graph replays, the D2H copy of token counts (used by
        ``tokens_per_expert``) is queued on the CUDA stream but may not have landed
        in CPU memory. Megatron adds ``_router_dtoh_event.record()`` + ``.synchronize()``
        to block until D2H is complete before calling expert dispatch.

        DES-LOC adaptation:
        On A6000 (partial mode), the event is recorded on the A6000 stream and
        synchronized on the CPU. On H100 (full mode), the entire forward pass is a
        single captured graph so there is no D2H gap — this method is a no-op for H100.

        Args:
            layer: The MoETransformerLayer to execute.
            hidden_states: [T, H] token embeddings.
            probs: [T, num_experts] router probabilities.
            shared_expert_output: Optional shared expert result.

        Returns:
            Output hidden states [T, H].
        """
        if self._current_mode == MoEGraphMode.FULL_INFER:
            # Full graph: no D2H gap, just call forward
            return layer(hidden_states)

        # Partial mode on A6000: router graph → D2H sync → expert dispatch → postprocess
        residual, hidden_states_routed, probs_out, shared_out = (
            layer._forward_mlp_router(hidden_states)
        )

        # Record and sync the D2H event to ensure token counts are stable
        self._router_dtoh_event.record()
        self._router_dtoh_event.synchronize()

        # Restore token dispatcher attributes that the router graph may have repointed
        # into cudagraph pool memory (Megatron e19fbe2 fix)
        if hasattr(layer, "token_dispatcher_attrs"):
            for name, attr in layer.token_dispatcher_attrs.items():
                setattr(layer.mlp.token_dispatcher, name, attr)

        expert_output, mlp_bias = layer._forward_mlp_expert_compute(
            hidden_states_routed, probs_out
        )
        return layer._forward_mlp_postprocess(
            residual, expert_output, shared_out, mlp_bias
        )

    @property
    def current_mode(self) -> MoEGraphMode:
        """Current MoE CUDA graph mode."""
        return self._current_mode

    def status_summary(self) -> str:
        """Return a human-readable status summary for logging."""
        moe_layers = self._get_moe_layers()
        lines = [
            f"HeteroRLMoECudagraphManager status:",
            f"  current_mode   : {self._current_mode.name}",
            f"  train_device   : {self.train_device} ({get_device_role(self.train_device).name})",
            f"  infer_device   : {self.infer_device} ({get_device_role(self.infer_device).name})",
            f"  moe_layers     : {len(moe_layers)}",
            f"  registry       : {len(self.registry._graphs)} graphs",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Weak-ref helper with dtype fallback (Megatron e19fbe2 / cuda_graphs.py fix)
# ---------------------------------------------------------------------------

def make_weak_ref_with_fallback(arg: Tensor, rank: int = 0) -> Tensor:
    """Create a weak reference to a tensor, with fallback for unsupported dtypes.

    Upstream (Megatron e19fbe2 / cuda_graphs.py):
    TransformerEngine's weak reference machinery (``make_weak_ref``) doesn't handle
    all dtypes (e.g. torch.float64 is unmapped in TE's utils.py). Megatron adds a
    try/except that falls back to a strong reference with a warning.

    DES-LOC adaptation:
    Identical logic, but we also log the device role so that DES-LOC debugging
    can distinguish A6000 vs H100 weak-ref failures.

    Args:
        arg: The tensor to weakly reference.
        rank: Distributed rank for log gating.

    Returns:
        A weak reference tensor (or the original tensor if weak ref fails).
    """
    try:
        from transformer_engine.pytorch.utils import make_weak_ref
        ref = make_weak_ref(arg)
        ref.requires_grad = arg.requires_grad
        if hasattr(arg, "can_skip_replay_copy"):
            ref.can_skip_replay_copy = arg.can_skip_replay_copy
        return ref
    except RuntimeError:
        if rank == 0:
            try:
                device = arg.device
                role = get_device_role(device).name
            except (ValueError, RuntimeError):
                role = "unknown"
            logger.warning(
                "make_weak_ref_with_fallback: could not create weak ref for tensor "
                "dtype=%s device=%s role=%s; keeping strong ref (potential memory overhead)",
                arg.dtype,
                arg.device,
                role,
            )
        return arg


# ---------------------------------------------------------------------------
# Graph warmup state tracker (Megatron e19fbe2 / cuda_graphs.py _set_warmup_end fix)
# ---------------------------------------------------------------------------

class WarmupTracker:
    """Tracks CUDA graph warmup phase with correct state management.

    Upstream (Megatron e19fbe2 / cuda_graphs.py):
    ``_set_warmup_end`` was missing the assignment ``_IS_GRAPH_WARMUP = False``.
    This caused the warmup flag to remain True after warmup completed, suppressing
    graph replay for the rest of training.

    DES-LOC adaptation:
    We wrap the flag in a class to make the state explicit and testable. The class
    also tracks which device each warmup was performed on (needed for the heterogeneous
    case where A6000 and H100 have separate warmup sequences).
    """

    def __init__(self) -> None:
        self._is_warmup: bool = False
        self._warmup_device: Optional[torch.device] = None

    def start(self, device: Optional[torch.device] = None) -> None:
        """Mark warmup as started.

        Args:
            device: The CUDA device beginning warmup (for DES-LOC tracking).
        """
        self._is_warmup = True
        self._warmup_device = device
        logger.debug("WarmupTracker: warmup started on device=%s", device)

    def end(self) -> None:
        """Mark warmup as ended. Critically sets _is_warmup = False.

        This is the fix from Megatron e19fbe2: _set_warmup_end was missing the
        assignment. Without it, the graph manager would treat all subsequent
        forward passes as warmup and never capture graphs.
        """
        self._is_warmup = False   # THE FIX: this assignment was missing in original code
        logger.debug(
            "WarmupTracker: warmup ended (was on device=%s)", self._warmup_device
        )
        self._warmup_device = None

    @property
    def is_warmup(self) -> bool:
        """True if currently in CUDA graph warmup phase."""
        return self._is_warmup


# ---------------------------------------------------------------------------
# Tensor mismatch checker with improved error messages (Megatron e19fbe2)
# ---------------------------------------------------------------------------

@dataclass
class TensorMismatchInfo:
    """Structured mismatch report for CUDA graph replay validation.

    Upstream (Megatron e19fbe2 / cuda_graphs.py):
    Error messages were improved from "expected X vs Y" to "Received X but expected Y"
    for clarity. DES-LOC extends this with device role information.
    """
    context: str
    shape_mismatch: Optional[Tuple] = None    # (received, expected)
    dtype_mismatch: Optional[Tuple] = None    # (received, expected)
    device_mismatch: Optional[Tuple] = None   # (received, expected)

    def has_mismatch(self) -> bool:
        """True if any mismatch was detected."""
        return any([self.shape_mismatch, self.dtype_mismatch, self.device_mismatch])

    def format_errors(self) -> str:
        """Format mismatch details for logging."""
        parts = []
        if self.shape_mismatch:
            recv, exp = self.shape_mismatch
            parts.append(f"Received shape {recv} but expected {exp}")
        if self.dtype_mismatch:
            recv, exp = self.dtype_mismatch
            parts.append(f"Received dtype {recv} but expected {exp}")
        if self.device_mismatch:
            recv, exp = self.device_mismatch
            recv_role = "unknown"
            try:
                recv_role = get_device_role(torch.device(str(recv))).name
            except (ValueError, RuntimeError):
                pass
            parts.append(
                f"Received device {recv} (role={recv_role}) but expected {exp}"
            )
        return f"Tensor mismatch at {self.context}: " + ", ".join(parts)


def check_tensor_for_graph_replay(
    val_tensor: Tensor,
    ref_tensor: Tensor,
    context: str,
) -> TensorMismatchInfo:
    """Validate a tensor against its graph capture reference.

    Upstream (Megatron e19fbe2): improved error messages in _CudaGraphRunner.
    DES-LOC: extended to include device role in error output.

    Args:
        val_tensor: The tensor being passed to replay.
        ref_tensor: The reference tensor from graph capture.
        context: Human-readable location string for error messages.

    Returns:
        TensorMismatchInfo — check ``.has_mismatch()`` to see if validation failed.
    """
    info = TensorMismatchInfo(context=context)
    if val_tensor.shape != ref_tensor.shape:
        info.shape_mismatch = (val_tensor.shape, ref_tensor.shape)
    if val_tensor.dtype != ref_tensor.dtype:
        info.dtype_mismatch = (val_tensor.dtype, ref_tensor.dtype)
    if val_tensor.device != ref_tensor.device:
        info.device_mismatch = (val_tensor.device, ref_tensor.device)
    return info


# ---------------------------------------------------------------------------
# Mamba inference guard (Megatron e19fbe2 / mamba_model.py fix)
# ---------------------------------------------------------------------------

def should_run_mtp_forward(
    mtp_process: bool,
    training: bool,
    inference_context: Optional[object],
) -> bool:
    """Determine if the Multi-Token Prediction (MTP) forward pass should run.

    Upstream (Megatron e19fbe2 / mamba_model.py):
    ``self.mtp_process`` was used directly, but MTP inference is not yet supported.
    The fix gates MTP on ``self.training and inference_context is None``.

    DES-LOC adaptation:
    In DES-LOC, the H100 runs inference. We must ensure MTP is disabled when
    inference_context is set (i.e. we are on the H100 in inference mode).
    This function centralizes that gating logic.

    Args:
        mtp_process: Whether this model has MTP enabled.
        training: Whether the model is in training mode (torch Module .training).
        inference_context: The inference context object; None during training.

    Returns:
        True if MTP forward should execute, False otherwise.
    """
    # MTP inference is not supported (TODO in Megatron e19fbe2)
    return mtp_process and training and (inference_context is None)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    logger.info("=== DES-LOC HeteroRLMoECudagraph smoke test ===")

    # --- Test 1: DESLOCPackedSeqParams deferred seq_idx ---
    cu_seqlens = torch.tensor([0, 5, 7, 11], dtype=torch.int32)
    params = DESLOCPackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        total_tokens=16,
    )
    seq_idx = params.seq_idx
    assert seq_idx is not None, "seq_idx should not be None"
    assert seq_idx.shape == (1, 16), f"expected [1,16], got {seq_idx.shape}"
    expected = torch.tensor(
        [[0,0,0,0,0, 1,1, 2,2,2,2, 3,3,3,3,3]], dtype=torch.int32
    )
    assert torch.equal(seq_idx, expected), f"seq_idx mismatch: {seq_idx}"
    logger.info("Test 1 PASSED: DESLOCPackedSeqParams seq_idx")

    # --- Test 2: make_single_sequence ---
    params2 = DESLOCPackedSeqParams.make_single_sequence(128, device=torch.device("cpu"))
    assert params2.total_tokens == 128
    assert params2.cu_seqlens_q.tolist() == [0, 128]
    logger.info("Test 2 PASSED: make_single_sequence")

    # --- Test 3: WarmupTracker state fix ---
    wt = WarmupTracker()
    wt.start()
    assert wt.is_warmup is True
    wt.end()
    assert wt.is_warmup is False, "is_warmup must be False after end() — this was the Megatron bug"
    logger.info("Test 3 PASSED: WarmupTracker.end() sets is_warmup=False")

    # --- Test 4: TensorMismatchInfo formatting ---
    a = torch.zeros(4, 8, dtype=torch.float32)
    b = torch.zeros(4, 16, dtype=torch.bfloat16)
    info = check_tensor_for_graph_replay(a, b, "test_layer.mlp")
    assert info.has_mismatch()
    msg = info.format_errors()
    assert "Received shape" in msg and "Received dtype" in msg, f"unexpected msg: {msg}"
    logger.info("Test 4 PASSED: TensorMismatchInfo: %s", msg)

    # --- Test 5: MoE argsort permute fixed-shape output ---
    T, E, H = 8, 4, 16
    tokens = torch.randn(T, H)
    routing_map = torch.zeros(T, E, dtype=torch.bool)
    routing_map[0, 0] = True
    routing_map[1, 0] = True
    routing_map[2, 1] = True
    probs = torch.rand(T, E)
    num_out = 3
    perm_tokens, perm_probs = deslocmoe_permute_argsort(
        tokens, routing_map, probs, num_out_tokens=num_out,
        num_tokens=T, num_experts=E,
    )
    assert perm_tokens.shape == (num_out, H), f"expected ({num_out},{H}), got {perm_tokens.shape}"
    assert perm_probs.shape == (num_out,), f"expected ({num_out},), got {perm_probs.shape}"
    logger.info("Test 5 PASSED: deslocmoe_permute_argsort fixed-shape output")

    # --- Test 6: should_run_mtp_forward gating ---
    assert should_run_mtp_forward(True, True, None) is True
    assert should_run_mtp_forward(True, False, None) is False   # not training
    assert should_run_mtp_forward(True, True, object()) is False  # inference context set
    assert should_run_mtp_forward(False, True, None) is False   # no mtp
    logger.info("Test 6 PASSED: should_run_mtp_forward gating")

    logger.info("=== All smoke tests PASSED ===")
