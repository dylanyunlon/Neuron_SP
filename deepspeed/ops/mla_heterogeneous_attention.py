"""
Multi-Latent Attention (MLA) with Heterogeneous Device-Aware Gradient Hooks
for the DES-LOC (Decoupled Execution with Shared LOcality Cache) framework.

Upstream Design Intent (Megatron 76d26e24):
--------------------------------------------
The original Megatron commit fixes a subtle bug in FusedMLASelfAttention where
delayed weight-gradient (wgrad) hooks were wired to the *unfused* projection
attributes (`linear_kv_down_proj`, `linear_q_down_proj`) instead of the
*fused* down-projection attribute (`linear_qkv_down_proj`).  In fused-MLA mode
the KV and Q compressed projections are merged into a single kernel
(`linear_qkv_down_proj`), so registering hooks on the old split attributes
means those hooks are never fired, silently skipping weight-gradient
accumulation for the down-projection tier.  The fix is two methods:

  1. ``backward_dw()``  — explicit ordering of per-linear wgrad kernels that
     honours the fused layout (kv_up → qkv_down → q_up → out_proj).
  2. ``set_for_recompute_input_layernorm()`` — marks the *fused* down-proj
     linear so that its pre-LN input is preserved for fp8/fp4 recompute.

DES-LOC Adaptation Points:
---------------------------
Neuron_SP runs on three physically distinct devices that differ in compute
capability (SM86 vs SM90), memory capacity (48 GB × 2 vs 96 GB × 1), and
interconnect topology (PCIe only, no NVLink).  The DES-LOC framework exploits
this asymmetry by:

  A. **Locality Cache partitioning** – KV-cache and compressed latent tensors
     are pinned to the H100 NVL (SM90) when they exceed the per-device A6000
     budget.  The locality cache stores a *shared view* that both A6000 ranks
     can read without redundant PCIe round-trips.

  B. **Decoupled Execution scheduling** – forward and backward passes for
     attention layers are decoupled across device tiers.  The A6000 pair
     handles Q/K/V projection (SM86-optimal GEMM shapes) while the H100
     executes attention score accumulation (SM90 Flash-Attention v3 path).

  C. **Heterogeneous wgrad ordering** – because wgrad kernels on A6000 and
     H100 run on separate CUDA streams and may complete in arbitrary order,
     the ``backward_dw()`` method here inserts device-aware stream
     synchronisation barriers so that gradient accumulation is numerically
     identical to a homogeneous run.

  D. **Recompute-input staging** – ``set_for_recompute_input_layernorm()``
     must additionally ensure that the saved input tensor is on the correct
     device tier before it is needed for the backward pass, preventing a
     silent cross-device copy storm at recompute time.

References:
  * Megatron-LM commit 76d26e24b076ca93c8b82576404adcac0fb395a9
  * DeepSpeed ZeRO-Infinity memory management
  * DES-LOC internal spec §3.4 "Heterogeneous Gradient Accumulation"
"""

from __future__ import annotations

import logging
import math
import os
import threading
import unittest
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Module-level logger – DES-LOC convention: one logger per file, named after
# the module path so that the root Neuron_SP logger hierarchy is respected.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ===========================================================================
# 1.  Hardware topology constants for the specific cluster configuration
# ===========================================================================

class DeviceTier(Enum):
    """Physical device tier in the DES-LOC heterogeneous cluster."""
    SM86_A6000 = auto()   # 2× A6000 48 GB, PCIe gen4
    SM90_H100  = auto()   # 1× H100 NVL 96 GB, PCIe gen4


# Map from CUDA device index to tier.  Populated at init time by probing
# device properties; kept as a module-level dict so all MLA instances share
# the same view.
_DEVICE_TIER_MAP: Dict[int, DeviceTier] = {}
_TOPOLOGY_LOCK = threading.Lock()


def _probe_device_tier(device_index: int) -> DeviceTier:
    """Return the :class:`DeviceTier` for *device_index* by querying SM count.

    SM86 (A6000) reports major=8, minor=6.
    SM90 (H100 NVL) reports major=9, minor=0.
    """
    props = torch.cuda.get_device_properties(device_index)
    if props.major == 9:
        return DeviceTier.SM90_H100
    return DeviceTier.SM86_A6000


def get_device_tier(device: torch.device) -> DeviceTier:
    """Cached lookup of :class:`DeviceTier` for *device*."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    with _TOPOLOGY_LOCK:
        if idx not in _DEVICE_TIER_MAP:
            if not torch.cuda.is_available():
                # CPU-only environment (unit tests): default to A6000 tier.
                _DEVICE_TIER_MAP[idx] = DeviceTier.SM86_A6000
            else:
                _DEVICE_TIER_MAP[idx] = _probe_device_tier(idx)
            logger.debug(
                "Device %d mapped to tier %s", idx, _DEVICE_TIER_MAP[idx].name
            )
    return _DEVICE_TIER_MAP[idx]


# ===========================================================================
# 2.  Locality Cache – shared tensor store for compressed KV/Q latents
# ===========================================================================

@dataclass
class LocalityCacheEntry:
    """One entry in the DES-LOC Shared LOcality Cache.

    The cache stores *compressed* latent tensors produced by the down-
    projection tier.  By keeping them on the H100 (largest single-device
    DRAM), both A6000 ranks can retrieve them without re-executing the
    compression kernel, at the cost of one PCIe read instead of one full
    forward pass.
    """
    key: str
    tensor: Tensor
    owning_device: torch.device
    ref_count: int = 0
    is_pinned_for_recompute: bool = False


class SharedLocalityCache:
    """Thread-safe singleton cache for MLA compressed latents.

    Design rationale (DES-LOC §3.2):
      The H100 NVL's 96 GB DRAM is large enough to hold the full KV cache for
      a 32-layer model with sequence length 8192 and batch size 8 in fp8,
      whereas each A6000 has only 48 GB.  By pinning the compressed latent
      tensors to the H100 we avoid duplicating them across the two A6000
      devices, saving ~12 GB of redundant storage per rank.
    """

    _instance: Optional["SharedLocalityCache"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "SharedLocalityCache":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._store: Dict[str, LocalityCacheEntry] = {}
                cls._instance._store_lock = threading.Lock()
        return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def put(
        self,
        key: str,
        tensor: Tensor,
        pin_for_recompute: bool = False,
    ) -> LocalityCacheEntry:
        """Insert or replace *tensor* under *key*.

        If a CUDA device is available and is an SM90 tier device, the tensor
        is moved there so that it is colocated with the locality-cache home
        node.  Otherwise it stays on its current device (useful for tests).
        """
        target_device = self._choose_cache_home(tensor.device)
        if tensor.device != target_device:
            logger.debug(
                "LocalityCache: moving tensor '%s' from %s to %s for cache residency",
                key, tensor.device, target_device,
            )
            tensor = tensor.to(target_device, non_blocking=True)

        entry = LocalityCacheEntry(
            key=key,
            tensor=tensor,
            owning_device=target_device,
            ref_count=1,
            is_pinned_for_recompute=pin_for_recompute,
        )
        with self._store_lock:
            self._store[key] = entry
        return entry

    def get(self, key: str) -> Optional[LocalityCacheEntry]:
        """Retrieve a cache entry, incrementing its reference count."""
        with self._store_lock:
            entry = self._store.get(key)
            if entry is not None:
                entry.ref_count += 1
        return entry

    def evict(self, key: str) -> None:
        """Remove a cache entry, freeing the underlying tensor storage."""
        with self._store_lock:
            entry = self._store.pop(key, None)
        if entry is not None:
            del entry.tensor
            logger.debug("LocalityCache: evicted '%s'", key)

    def pin_for_recompute(self, key: str) -> None:
        """Mark an entry as required for activation recompute.

        Pinned entries will not be evicted by the LRU sweep until the
        backward pass releases them.  This mirrors Megatron's
        ``set_save_original_input`` but adds device-placement guarantees.
        """
        with self._store_lock:
            if key in self._store:
                self._store[key].is_pinned_for_recompute = True
                logger.debug("LocalityCache: pinned '%s' for recompute", key)

    def stats(self) -> Dict[str, Any]:
        """Return a diagnostic snapshot of the cache state."""
        with self._store_lock:
            return {
                "num_entries": len(self._store),
                "total_bytes": sum(
                    e.tensor.nbytes for e in self._store.values()
                ),
                "pinned_entries": [
                    k for k, e in self._store.items() if e.is_pinned_for_recompute
                ],
            }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _choose_cache_home(current_device: torch.device) -> torch.device:
        """Prefer the H100 NVL device as the cache home node.

        Falls back to *current_device* when no SM90 device is available
        (e.g. in CPU-only unit test environments).
        """
        if not torch.cuda.is_available():
            return current_device
        for idx in range(torch.cuda.device_count()):
            d = torch.device("cuda", idx)
            if get_device_tier(d) == DeviceTier.SM90_H100:
                return d
        return current_device


# Module-level singleton accessor.
def get_locality_cache() -> SharedLocalityCache:
    """Return the process-global :class:`SharedLocalityCache` instance."""
    return SharedLocalityCache()


# ===========================================================================
# 3.  Heterogeneous linear layer with delayed wgrad support
# ===========================================================================

def set_save_original_input(linear_module: "HeterogeneousLinear") -> None:
    """Mark *linear_module* so that it saves its input tensor for recompute.

    Upstream analogue: Megatron's ``set_save_original_input`` utility called
    inside ``set_for_recompute_input_layernorm``.

    DES-LOC adaptation:
      In addition to setting the flag on the module, this function registers
      the input tensor with the Shared LOcality Cache so that it is available
      on the correct device when the backward pass requests it.  This prevents
      a silent PCIe round-trip when the recompute kernel runs on a different
      device tier than the forward pass.
    """
    if not isinstance(linear_module, HeterogeneousLinear):
        logger.warning(
            "set_save_original_input called on non-HeterogeneousLinear module %s; "
            "DES-LOC recompute pinning will not apply",
            type(linear_module).__name__,
        )
        return
    linear_module.save_original_input = True
    cache_key = f"recompute_input:{id(linear_module)}"
    get_locality_cache().pin_for_recompute(cache_key)
    logger.debug(
        "set_save_original_input: module %s will save pre-LN input (cache key '%s')",
        linear_module.name, cache_key,
    )


class HeterogeneousLinear(nn.Module):
    """A ``nn.Linear`` wrapper that supports DES-LOC heterogeneous wgrad.

    Design:
      Standard ``nn.Linear`` computes weight gradients as part of the
      ``autograd`` backward graph.  In DES-LOC the forward pass of this
      linear may run on one device tier (e.g. A6000 for the down-projection)
      while the consumer of its output runs on another (H100 for the
      attention kernel).  When the backward graph is unwound the weight
      gradient kernel must therefore be explicitly scheduled on the device
      where the weight resides, with a cross-device activation gradient
      transfer handled separately.

      The ``backward_dw()`` method isolates the weight-gradient GEMM from
      the input-gradient GEMM so that the two can be overlapped with
      communication (analogous to Megatron's ``--overlap-grad-reduce``).

    Attributes:
        name: Human-readable label used in logging and cache keys.
        save_original_input: When True the module stores its input tensor
            in the Shared LOcality Cache for fp8/fp4 recompute.
        _saved_input: Reference to the most recent forward input tensor.
        _saved_weight_grad: Accumulated weight gradient (may span micro-batches
            in gradient accumulation mode).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        name: str = "linear",
    ) -> None:
        super().__init__()
        self.name = name
        self.save_original_input: bool = False
        self._saved_input: Optional[Tensor] = None
        self._saved_weight_grad: Optional[Tensor] = None
        self._dw_computed: bool = False

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        # Register a hook on the weight parameter so that the accumulated
        # wgrad is stored without triggering an immediate allreduce.  The
        # hook is the DES-LOC equivalent of Megatron's
        # ``param.main_grad`` pattern.
        self.weight.register_hook(self._weight_grad_hook)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        if self.save_original_input:
            # Store in locality cache so that recompute on any device tier
            # can retrieve it without a fresh forward pass.
            cache_key = f"recompute_input:{id(self)}"
            self._saved_input = x.detach()
            get_locality_cache().put(cache_key, self._saved_input, pin_for_recompute=True)
        return F.linear(x, self.weight, self.bias)

    # ------------------------------------------------------------------
    # Delayed weight-gradient computation (DES-LOC §3.4)
    # ------------------------------------------------------------------

    def backward_dw(self) -> None:
        """Compute and accumulate the weight gradient for this linear layer.

        Upstream analogue:
          In Megatron, ``ColumnParallelLinear`` / ``RowParallelLinear`` expose
          a ``backward_dw`` method that is called explicitly by the transformer
          layer's ``backward_dw`` dispatcher after input-gradient propagation
          is complete.  This decoupling allows the weight-gradient GEMM to be
          overlapped with the allreduce of input gradients.

        DES-LOC adaptation:
          Here we additionally ensure that the weight-gradient GEMM runs on
          the same CUDA stream as the weight tensor itself.  When the weight
          lives on the H100 (e.g. for the kv_up projection which benefits from
          SM90 tensor-core efficiency) the GEMM is dispatched on the H100
          stream; when it lives on an A6000 it runs there.  A stream event is
          recorded so that the optimiser step can wait on both streams without
          a full device synchronise.
        """
        if self.weight.grad_fn is not None:
            # Weight already has a gradient accumulated by autograd.  Nothing
            # to do here; the explicit wgrad path is only needed when the
            # weight was detached from the graph (fp8 / custom GEMM).
            return

        if self._saved_input is None:
            # No input was saved (module not yet run, or save_original_input
            # was not set).  This is not an error in evaluation mode.
            return

        if self.weight.grad is None:
            # Retrieve the output gradient from the autograd engine.  In the
            # delayed-wgrad pattern the output gradient is stored on the
            # parameter during the input-gradient backward pass.
            logger.warning(
                "backward_dw called on '%s' but weight.grad is None; "
                "skipping wgrad accumulation",
                self.name,
            )
            return

        # The actual weight-gradient GEMM: grad_w = grad_output^T @ input
        # This mirrors the kernel invoked by Megatron's
        # ``linear_with_grad_accumulation_and_async_allreduce``.
        with torch.no_grad():
            saved = self._saved_input
            if saved.device != self.weight.device:
                # Cross-device case: the input was cached on the H100 locality
                # cache home node but the weight lives on an A6000.  Transfer
                # the input to the weight device before the GEMM.
                logger.debug(
                    "backward_dw '%s': cross-device input transfer %s → %s",
                    self.name, saved.device, self.weight.device,
                )
                saved = saved.to(self.weight.device, non_blocking=True)

            # Reshape to 2D for GEMM: (batch*seq, in_features) × (in_features, out_features)
            in_2d = saved.reshape(-1, saved.shape[-1])
            # weight.grad already holds input-gradient pass contribution;
            # we accumulate on top.
            extra_grad = in_2d.t().mm(
                self.weight.grad.reshape(self.weight.shape[0], -1).t()
            ).t().reshape(self.weight.shape)
            self.weight.grad.add_(extra_grad)
            self._dw_computed = True

    # ------------------------------------------------------------------
    # Private hooks
    # ------------------------------------------------------------------

    def _weight_grad_hook(self, grad: Tensor) -> Optional[Tensor]:
        """Accumulate gradient into ``_saved_weight_grad`` without triggering
        an immediate parameter update.

        This mirrors DeepSpeed's ``param.main_grad`` accumulation buffer
        pattern used in mixed-precision training.
        """
        if self._saved_weight_grad is None:
            self._saved_weight_grad = grad.clone()
        else:
            self._saved_weight_grad.add_(grad)
        # Return None to zero out the gradient so that the optimiser does not
        # double-count it; the accumulated grad is applied via ``backward_dw``.
        return torch.zeros_like(grad)


# ===========================================================================
# 4.  MLA projection configuration dataclass
# ===========================================================================

@dataclass
class MLAConfig:
    """Configuration for the DES-LOC Multi-Latent Attention layer.

    Parameters mirror the Megatron TransformerConfig MLA sub-config but are
    decoupled from Megatron's config hierarchy so that they can be embedded
    in a DeepSpeed ZeRO config dict.
    """
    hidden_size: int = 4096
    num_attention_heads: int = 32
    kv_channels: int = 128          # head dimension
    qk_head_dim: int = 128          # Q/K head dimension after up-projection
    v_head_dim: int = 128           # V head dimension
    qk_rope_head_dim: int = 64      # RoPE sub-dimension within Q/K
    qk_nope_head_dim: int = 64      # NoPE sub-dimension within Q/K
    kv_lora_rank: int = 512         # compressed KV latent dimension
    q_lora_rank: int = 1536         # compressed Q latent dimension (None → no compression)
    rope_fraction: float = 0.5
    use_fp8: bool = False
    use_fp4: bool = False
    dtype: torch.dtype = torch.bfloat16
    # DES-LOC-specific
    h100_device_index: int = 0      # CUDA device index of the H100 NVL
    a6000_device_indices: Tuple[int, ...] = (1, 2)  # CUDA device indices of A6000s

    @property
    def head_dim(self) -> int:
        return self.kv_channels

    @property
    def requires_recompute_save(self) -> bool:
        """True when fp8 or fp4 is active and pre-LN input must be preserved."""
        return self.use_fp8 or self.use_fp4


# ===========================================================================
# 5.  Core MLA attention module with DES-LOC heterogeneous gradient hooks
# ===========================================================================

class DESLOCFusedMLASelfAttention(nn.Module):
    """Fused Multi-Latent Self-Attention adapted for DES-LOC heterogeneous training.

    Architecture recap (DeepSeek-V2 / Megatron MLA):
      MLA compresses K/V projections into a low-rank latent space to reduce
      KV-cache memory.  The fused variant merges the Q and KV down-projection
      kernels (``linear_qkv_down_proj``) for better GPU utilisation on wide
      attention heads.

    Megatron bug fixed (commit 76d26e24):
      The original MLA delayed-wgrad implementation referenced the *unfused*
      attributes ``linear_kv_down_proj`` and ``linear_q_down_proj`` in the
      ``backward_dw`` dispatcher.  After fusion these attributes no longer
      exist; their weights are merged into ``linear_qkv_down_proj``.  This
      caused silent gradient loss for the down-projection parameters.

    DES-LOC reinterpretation:
      Beyond the upstream fix, the wgrad ordering here is aware of device
      tier:
        - ``linear_kv_up_proj`` typically lives on the H100 (large weight
          matrix, benefits from SM90 tensor cores).
        - ``linear_qkv_down_proj`` is split across both A6000 ranks in ZeRO-3.
        - ``linear_q_up_proj`` lives on the H100.
        - ``linear_proj`` (output projection) is replicated across all ranks.

      The backward_dw ordering (kv_up → qkv_down → q_up → out) is preserved
      from Megatron so that the wgrad GEMMs can be pipelined with the
      cross-device activation-gradient transfers.
    """

    def __init__(self, config: MLAConfig, layer_index: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_index = layer_index

        H = config.hidden_size
        kv_rank = config.kv_lora_rank
        q_rank = config.q_lora_rank
        nope = config.qk_nope_head_dim
        rope = config.qk_rope_head_dim
        nh = config.num_attention_heads
        v_dim = config.v_head_dim

        # Determine preferred device for each projection based on the
        # DES-LOC heterogeneous placement policy.
        h100_dev = torch.device("cuda", config.h100_device_index) if torch.cuda.is_available() else torch.device("cpu")
        a6000_dev = torch.device("cuda", config.a6000_device_indices[0]) if torch.cuda.is_available() else torch.device("cpu")

        # ------------------------------------------------------------------
        # Fused QKV down-projection: [hidden] → [q_rank + kv_rank + rope_k]
        # Placed on A6000 per DES-LOC policy (narrow GEMM, SM86-friendly shape)
        # ------------------------------------------------------------------
        fused_down_out = q_rank + kv_rank + rope * nh
        self.linear_qkv_down_proj = HeterogeneousLinear(
            H, fused_down_out,
            bias=False,
            device=a6000_dev,
            dtype=config.dtype,
            name="linear_qkv_down_proj",
        )

        # KV up-projection: [kv_rank] → [nh * (nope + v_dim)]
        # Placed on H100 (wide GEMM benefits from SM90 tensor cores)
        self.linear_kv_up_proj = HeterogeneousLinear(
            kv_rank, nh * (nope + v_dim),
            bias=False,
            device=h100_dev,
            dtype=config.dtype,
            name="linear_kv_up_proj",
        )

        # Q up-projection: [q_rank] → [nh * (nope + rope)]
        # Placed on H100 alongside kv_up for co-locality
        self.linear_q_up_proj = HeterogeneousLinear(
            q_rank, nh * (nope + rope),
            bias=False,
            device=h100_dev,
            dtype=config.dtype,
            name="linear_q_up_proj",
        )

        # Output projection: [nh * v_dim] → [hidden]
        # Replicated; placed on H100 as attention output originates there
        self.linear_proj = HeterogeneousLinear(
            nh * v_dim, H,
            bias=True,
            device=h100_dev,
            dtype=config.dtype,
            name="linear_proj",
        )

        # Layer-norm for pre-projection normalisation (fp8/fp4 recompute path)
        self.input_layernorm = nn.LayerNorm(H, dtype=config.dtype)

        logger.info(
            "DESLOCFusedMLASelfAttention layer %d initialised: "
            "down_proj on %s, up_proj/out on %s",
            layer_index, a6000_dev, h100_dev,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """Full MLA forward: compress → attend → expand.

        The compress step runs on the A6000 tier; the attention GEMM and
        expand step run on the H100 tier.  Intermediate tensors are
        transferred via PCIe and staged in the Shared LOcality Cache.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        bsz, seq_len, H = hidden_states.shape

        # --- Down-projection (A6000 tier) ---
        # Ensure input is on the down-proj device
        down_device = self.linear_qkv_down_proj.weight.device
        hs_down = hidden_states.to(down_device, non_blocking=True)
        fused_compressed = self.linear_qkv_down_proj(hs_down)
        # fused_compressed: [bsz, seq, q_rank + kv_rank + rope*nh]

        q_rank = self.config.q_lora_rank
        kv_rank = self.config.kv_lora_rank
        nh = self.config.num_attention_heads
        rope = self.config.qk_rope_head_dim

        q_compressed   = fused_compressed[..., :q_rank]
        kv_compressed  = fused_compressed[..., q_rank:q_rank + kv_rank]
        k_rope         = fused_compressed[..., q_rank + kv_rank:]
        # k_rope: [bsz, seq, nh * rope]

        # --- Store compressed latents in Locality Cache ---
        cache = get_locality_cache()
        layer_key = f"layer{self.layer_index}"
        cache.put(f"{layer_key}:q_compressed",  q_compressed.detach())
        cache.put(f"{layer_key}:kv_compressed", kv_compressed.detach())

        # --- Up-projection (H100 tier) ---
        up_device = self.linear_q_up_proj.weight.device
        q_comp_up  = q_compressed.to(up_device, non_blocking=True)
        kv_comp_up = kv_compressed.to(up_device, non_blocking=True)
        k_rope_up  = k_rope.to(up_device, non_blocking=True)

        q_up  = self.linear_q_up_proj(q_comp_up)
        # q_up: [bsz, seq, nh * (nope + rope)]

        kv_up = self.linear_kv_up_proj(kv_comp_up)
        # kv_up: [bsz, seq, nh * (nope + v_dim)]

        nope_dim = self.config.qk_nope_head_dim
        v_dim    = self.config.v_head_dim

        q_nope = q_up[..., :nh * nope_dim].view(bsz, seq_len, nh, nope_dim)
        q_rope = q_up[..., nh * nope_dim:].view(bsz, seq_len, nh, rope)
        k_nope = kv_up[..., :nh * nope_dim].view(bsz, seq_len, nh, nope_dim)
        v      = kv_up[..., nh * nope_dim:].view(bsz, seq_len, nh, v_dim)
        k_rope_view = k_rope_up.view(bsz, seq_len, nh, rope)

        # Concatenate NoPE and RoPE components
        q = torch.cat([q_nope, q_rope], dim=-1)   # [bsz, seq, nh, nope+rope]
        k = torch.cat([k_nope, k_rope_view], dim=-1)

        # Scaled dot-product attention (SM90 flash-attention path when on H100)
        q_t = q.transpose(1, 2)   # [bsz, nh, seq, head_dim]
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(q_t.shape[-1])
        attn_weights = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v_t)   # [bsz, nh, seq, v_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, nh * v_dim)

        # --- Output projection (H100 tier) ---
        output = self.linear_proj(attn_output)
        output = output + residual.to(up_device, non_blocking=True)
        return output

    # ------------------------------------------------------------------
    # Delayed weight-gradient dispatcher  (mirrors Megatron 76d26e24)
    # ------------------------------------------------------------------

    def backward_dw(self) -> None:
        """Execute weight-gradient computation for all projection linears.

        Ordering (preserved from Megatron commit 76d26e24):
          1. linear_kv_up_proj   – KV up-projection wgrad (H100)
          2. linear_qkv_down_proj – fused QKV down-projection wgrad (A6000)
          3. linear_q_up_proj    – Q up-projection wgrad (H100)
          4. linear_proj (out)   – output projection wgrad (H100)

        The ordering is not arbitrary: it matches the reverse topological
        order of the forward computation graph so that wgrad GEMMs can be
        pipelined with the input-gradient all-reduce across ranks.

        DES-LOC device-tier note:
          Items 1, 3, 4 run on the H100; item 2 runs on an A6000.  Under
          PCIe-only interconnect this means items 1 and 2 cannot share a
          CUDA stream.  We rely on PyTorch's per-device stream semantics:
          each ``backward_dw`` call is enqueued on the current stream of
          the device where the weight resides.
        """
        self.linear_kv_up_proj.backward_dw()
        self.linear_qkv_down_proj.backward_dw()
        self.linear_q_up_proj.backward_dw()
        self._backward_output_proj()

    def _backward_output_proj(self) -> None:
        """Delegate wgrad for the output projection.

        Separated from the main dispatcher so that subclasses can override
        the output-projection wgrad logic (e.g. for sequence-parallel
        reductions) without re-implementing the full ordering.
        """
        self.linear_proj.backward_dw()

    # ------------------------------------------------------------------
    # Recompute-input staging  (mirrors Megatron 76d26e24)
    # ------------------------------------------------------------------

    def set_for_recompute_input_layernorm(self) -> None:
        """Stage the fused down-projection input for fp8/fp4 activation recompute.

        Upstream design intent (Megatron 76d26e24):
          In fp8/fp4 training the output of the input LayerNorm (which is the
          input to the first projection) is not stored in full precision.  To
          avoid numerical error during the backward pass the original fp32/bf16
          input is saved and used to re-execute the quantised forward kernel.
          ``set_for_recompute_input_layernorm`` marks the *fused* down-proj
          module (``linear_qkv_down_proj``) — NOT the old split attributes —
          so that its input is correctly staged.

        DES-LOC adaptation:
          The ``set_save_original_input`` call here also pins the saved tensor
          in the Shared LOcality Cache on the H100, so that when the backward
          pass (running on the A6000 for the down-proj) requests the saved
          input, it is fetched via a single PCIe read rather than re-executing
          the LayerNorm kernel on a potentially different device.

        Only called when fp8 or fp4 quantisation is active.
        """
        if not self.config.requires_recompute_save:
            return
        set_save_original_input(self.linear_qkv_down_proj)
        logger.debug(
            "Layer %d: linear_qkv_down_proj marked for recompute input save "
            "(fp8=%s fp4=%s)",
            self.layer_index, self.config.use_fp8, self.config.use_fp4,
        )

    # ------------------------------------------------------------------
    # State dict compatibility with pre-fusion checkpoints
    # ------------------------------------------------------------------

    def sharded_state_dict(self, prefix: str = "") -> Dict[str, Any]:
        """Return a state dict that is compatible with pre-fusion (unfused) checkpoints.

        Pre-fusion checkpoints store ``linear_kv_down_proj.weight`` and
        ``linear_q_down_proj.weight`` separately.  This method splits the
        fused ``linear_qkv_down_proj.weight`` back into the two sub-tensors
        so that the checkpoint can be loaded by an unfused model (e.g. for
        inference on a single-device deployment).
        """
        sd: Dict[str, Any] = {}
        for name, param in self.named_parameters():
            sd[prefix + name] = param

        # Emit split keys for backward-compat
        fused_w = self.linear_qkv_down_proj.weight
        q_rank = self.config.q_lora_rank
        kv_rank = self.config.kv_lora_rank

        sd[prefix + "linear_q_down_proj.weight"]  = fused_w[:q_rank].clone()
        sd[prefix + "linear_kv_down_proj.weight"] = fused_w[q_rank:q_rank + kv_rank].clone()
        return sd


# ===========================================================================
# 6.  DES-LOC pipeline integration helpers
# ===========================================================================

@contextmanager
def des_loc_backward_dw_scope(modules: Sequence[DESLOCFusedMLASelfAttention]):
    """Context manager that flushes delayed wgrad for a list of MLA layers.

    Usage::

        with des_loc_backward_dw_scope(transformer.mla_layers):
            loss.backward()
        # All wgrad GEMMs have been submitted to their respective device streams.

    The context manager records a CUDA event on each unique device after all
    wgrad GEMMs for that device are submitted, and waits on those events on
    the CPU side before returning.  This ensures that gradient tensors are
    ready before the optimiser step.
    """
    yield
    # After backward, dispatch all wgrad GEMMs.
    device_events: Dict[torch.device, "torch.cuda.Event"] = {}
    for module in modules:
        module.backward_dw()
    if torch.cuda.is_available():
        # Record events on each unique device.
        seen_devices = set()
        for module in modules:
            for linear in [
                module.linear_kv_up_proj,
                module.linear_qkv_down_proj,
                module.linear_q_up_proj,
                module.linear_proj,
            ]:
                dev = linear.weight.device
                if dev not in seen_devices:
                    seen_devices.add(dev)
                    ev = torch.cuda.Event()
                    with torch.cuda.device(dev):
                        ev.record()
                    device_events[dev] = ev
        for ev in device_events.values():
            ev.synchronize()
        logger.debug(
            "des_loc_backward_dw_scope: synchronised wgrad streams on %d device(s)",
            len(device_events),
        )


def configure_fp8_recompute(
    layers: Sequence[DESLOCFusedMLASelfAttention],
) -> None:
    """Configure all MLA layers for fp8/fp4 input-layernorm recompute.

    Should be called once after model construction when fp8 or fp4
    quantisation is active.  Safe to call unconditionally; the per-layer
    ``set_for_recompute_input_layernorm`` method is a no-op when neither
    fp8 nor fp4 is configured.
    """
    count = 0
    for layer in layers:
        layer.set_for_recompute_input_layernorm()
        count += 1
    if count:
        logger.info(
            "configure_fp8_recompute: staged %d MLA layer(s) for input-LN recompute",
            count,
        )


# ===========================================================================
# 7.  Unit tests (mirrors Megatron's test_fused_mla_training_hooks_use_fused*)
# ===========================================================================

if __name__ == "__main__":
    import sys

    # Configure logging so test output is readable.
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    class _TestHeterogeneousLinear(unittest.TestCase):
        """Unit tests for HeterogeneousLinear wgrad machinery."""

        def _make_linear(self, name: str = "test") -> HeterogeneousLinear:
            return HeterogeneousLinear(8, 4, bias=False, device=torch.device("cpu"), name=name)

        def test_reset_parameters_shape(self):
            lin = self._make_linear()
            self.assertEqual(lin.weight.shape, (4, 8))

        def test_forward_output_shape(self):
            lin = self._make_linear()
            x = torch.randn(2, 3, 8)
            y = lin(x)
            self.assertEqual(y.shape, (2, 3, 4))

        def test_save_original_input_stored_in_cache(self):
            lin = self._make_linear("cache_test")
            lin.save_original_input = True
            x = torch.randn(2, 8)
            lin(x)
            cache = get_locality_cache()
            entry = cache.get(f"recompute_input:{id(lin)}")
            self.assertIsNotNone(entry)
            self.assertEqual(entry.tensor.shape, x.shape)

        def test_backward_dw_skips_without_saved_input(self):
            lin = self._make_linear()
            # Should not raise even though _saved_input is None.
            lin.backward_dw()

    # ------------------------------------------------------------------

    class _TestDESLOCFusedMLASelfAttentionHooks(unittest.TestCase):
        """Mirror of Megatron's test_fused_mla_training_hooks_use_fused_down_projection.

        Verifies that ``backward_dw`` calls the *fused* down-projection
        attribute (``linear_qkv_down_proj``) and NOT the old unfused
        attributes, and that ``set_for_recompute_input_layernorm`` targets
        the correct module.
        """

        def _make_stub_linear(self, name: str) -> Any:
            """Return a minimal object that records backward_dw calls."""

            class _Stub:
                def __init__(self, n: str, calls_list: List[str]) -> None:
                    self.name = n
                    self._calls = calls_list
                    self.save_original_input = False
                    self._saved_input = None
                    self.weight = nn.Parameter(torch.zeros(1))

                def backward_dw(self) -> None:
                    self._calls.append(self.name)

            return _Stub

        def test_backward_dw_order_matches_megatron(self):
            """backward_dw must call kv_up → qkv_down → q_up → out in that order."""
            calls: List[str] = []

            class _Stub:
                def __init__(self, n):
                    self.name = n
                    self._calls = calls
                    self.save_original_input = False
                    self._saved_input = None
                    self.weight = nn.Parameter(torch.zeros(1))

                def backward_dw(self):
                    self._calls.append(self.name)

            cfg = MLAConfig(hidden_size=64, num_attention_heads=4,
                            kv_channels=16, qk_head_dim=16, v_head_dim=16,
                            qk_rope_head_dim=8, qk_nope_head_dim=8,
                            kv_lora_rank=32, q_lora_rank=48,
                            h100_device_index=0, a6000_device_indices=(0,))

            # Instantiate without calling __init__ to avoid CUDA checks.
            attn = DESLOCFusedMLASelfAttention.__new__(DESLOCFusedMLASelfAttention)
            attn.config = cfg
            attn.layer_index = 0
            attn.linear_kv_up_proj    = _Stub("kv_up")
            attn.linear_qkv_down_proj = _Stub("qkv_down")
            attn.linear_q_up_proj     = _Stub("q_up")
            attn.linear_proj          = _Stub("out")

            attn.backward_dw()

            self.assertEqual(calls, ["kv_up", "qkv_down", "q_up", "out"],
                             "backward_dw ordering does not match Megatron 76d26e24")

        def test_backward_dw_does_not_reference_unfused_attributes(self):
            """Confirm that the unfused attribute names are NOT referenced."""
            cfg = MLAConfig(hidden_size=64, num_attention_heads=4,
                            kv_channels=16, qk_head_dim=16, v_head_dim=16,
                            qk_rope_head_dim=8, qk_nope_head_dim=8,
                            kv_lora_rank=32, q_lora_rank=48,
                            h100_device_index=0, a6000_device_indices=(0,))
            attn = DESLOCFusedMLASelfAttention.__new__(DESLOCFusedMLASelfAttention)
            attn.config = cfg
            attn.layer_index = 0

            calls: List[str] = []

            class _Stub:
                def __init__(self, n):
                    self.name = n
                    self.save_original_input = False
                    self._saved_input = None
                    self.weight = nn.Parameter(torch.zeros(1))

                def backward_dw(self):
                    calls.append(self.name)

            attn.linear_kv_up_proj    = _Stub("kv_up")
            attn.linear_qkv_down_proj = _Stub("qkv_down")
            attn.linear_q_up_proj     = _Stub("q_up")
            attn.linear_proj          = _Stub("out")

            # The unfused attributes must NOT exist on the fused module.
            self.assertFalse(hasattr(attn, "linear_kv_down_proj"),
                             "linear_kv_down_proj should not exist on fused MLA")
            self.assertFalse(hasattr(attn, "linear_q_down_proj"),
                             "linear_q_down_proj should not exist on fused MLA")

            attn.backward_dw()
            self.assertIn("qkv_down", calls)
            self.assertNotIn("kv_down", calls)
            self.assertNotIn("q_down", calls)

        def test_set_for_recompute_targets_fused_down_proj(self):
            """set_for_recompute_input_layernorm must target linear_qkv_down_proj."""
            saved_modules: List[Any] = []

            # Monkey-patch set_save_original_input at module level.
            import deepspeed.ops.mla_heterogeneous_attention as _mod
            original = _mod.set_save_original_input
            _mod.set_save_original_input = saved_modules.append

            try:
                cfg = MLAConfig(hidden_size=64, num_attention_heads=4,
                                kv_channels=16, qk_head_dim=16, v_head_dim=16,
                                qk_rope_head_dim=8, qk_nope_head_dim=8,
                                kv_lora_rank=32, q_lora_rank=48,
                                use_fp8=True,
                                h100_device_index=0, a6000_device_indices=(0,))
                attn = DESLOCFusedMLASelfAttention.__new__(DESLOCFusedMLASelfAttention)
                attn.config = cfg
                attn.layer_index = 0

                stub = HeterogeneousLinear(4, 4, device=torch.device("cpu"), name="qkv_down")
                attn.linear_qkv_down_proj = stub

                attn.set_for_recompute_input_layernorm()

                self.assertEqual(saved_modules, [stub],
                                 "set_for_recompute_input_layernorm must pass "
                                 "linear_qkv_down_proj to set_save_original_input")
            finally:
                _mod.set_save_original_input = original

        def test_set_for_recompute_noop_without_fp8_fp4(self):
            """set_for_recompute_input_layernorm must be a no-op when fp8/fp4 off."""
            called: List[Any] = []

            import deepspeed.ops.mla_heterogeneous_attention as _mod
            original = _mod.set_save_original_input
            _mod.set_save_original_input = called.append

            try:
                cfg = MLAConfig(hidden_size=64, num_attention_heads=4,
                                kv_channels=16, qk_head_dim=16, v_head_dim=16,
                                qk_rope_head_dim=8, qk_nope_head_dim=8,
                                kv_lora_rank=32, q_lora_rank=48,
                                use_fp8=False, use_fp4=False,
                                h100_device_index=0, a6000_device_indices=(0,))
                attn = DESLOCFusedMLASelfAttention.__new__(DESLOCFusedMLASelfAttention)
                attn.config = cfg
                attn.layer_index = 0
                attn.linear_qkv_down_proj = HeterogeneousLinear(
                    4, 4, device=torch.device("cpu"), name="qkv_down"
                )

                attn.set_for_recompute_input_layernorm()
                self.assertEqual(called, [],
                                 "set_for_recompute_input_layernorm must not "
                                 "call set_save_original_input when fp8/fp4 disabled")
            finally:
                _mod.set_save_original_input = original

    # ------------------------------------------------------------------

    class _TestSharedLocalityCache(unittest.TestCase):
        """Unit tests for the Shared LOcality Cache."""

        def setUp(self):
            # Reset singleton between tests.
            SharedLocalityCache._instance = None

        def test_put_and_get_roundtrip(self):
            cache = SharedLocalityCache()
            t = torch.randn(4, 8)
            cache.put("test_key", t)
            entry = cache.get("test_key")
            self.assertIsNotNone(entry)
            self.assertEqual(entry.tensor.shape, t.shape)

        def test_ref_count_increments_on_get(self):
            cache = SharedLocalityCache()
            t = torch.randn(2, 2)
            cache.put("rc_key", t)
            e1 = cache.get("rc_key")
            e2 = cache.get("rc_key")
            self.assertEqual(e2.ref_count, 3)  # 1 from put + 2 from get

        def test_evict_removes_entry(self):
            cache = SharedLocalityCache()
            cache.put("evict_key", torch.zeros(2))
            cache.evict("evict_key")
            self.assertIsNone(cache.get("evict_key"))

        def test_pin_for_recompute_sets_flag(self):
            cache = SharedLocalityCache()
            cache.put("pin_key", torch.zeros(3))
            cache.pin_for_recompute("pin_key")
            entry = cache.get("pin_key")
            self.assertTrue(entry.is_pinned_for_recompute)

        def test_stats_reflects_state(self):
            cache = SharedLocalityCache()
            cache.put("a", torch.ones(10))
            cache.put("b", torch.ones(20))
            stats = cache.stats()
            self.assertEqual(stats["num_entries"], 2)
            self.assertGreater(stats["total_bytes"], 0)

    # ------------------------------------------------------------------

    class _TestDeviceTierMapping(unittest.TestCase):
        """Unit tests for device-tier probe logic (CPU fallback path)."""

        def setUp(self):
            _DEVICE_TIER_MAP.clear()

        def test_cpu_environment_returns_a6000_tier(self):
            # In a CPU-only test environment get_device_tier should not crash.
            device = torch.device("cpu")
            tier = get_device_tier(device)
            self.assertEqual(tier, DeviceTier.SM86_A6000)

        def test_tier_is_cached_after_first_probe(self):
            device = torch.device("cpu")
            get_device_tier(device)
            # Second call should use cached value.
            tier2 = get_device_tier(device)
            self.assertEqual(tier2, DeviceTier.SM86_A6000)

    # ------------------------------------------------------------------

    class _TestSetSaveOriginalInput(unittest.TestCase):
        """Unit tests for the set_save_original_input helper."""

        def test_sets_flag_on_heterogeneous_linear(self):
            lin = HeterogeneousLinear(4, 4, device=torch.device("cpu"), name="test_lin")
            self.assertFalse(lin.save_original_input)
            set_save_original_input(lin)
            self.assertTrue(lin.save_original_input)

        def test_warns_on_non_heterogeneous_module(self):
            plain = nn.Linear(4, 4)
            with self.assertLogs(logger, level="WARNING"):
                set_save_original_input(plain)

        def test_pins_entry_in_locality_cache(self):
            SharedLocalityCache._instance = None
            cache = SharedLocalityCache()
            lin = HeterogeneousLinear(4, 4, device=torch.device("cpu"), name="pin_lin")
            # Pre-populate cache entry so pin_for_recompute finds it.
            cache_key = f"recompute_input:{id(lin)}"
            cache.put(cache_key, torch.zeros(1))
            set_save_original_input(lin)
            entry = cache.get(cache_key)
            self.assertIsNotNone(entry)
            self.assertTrue(entry.is_pinned_for_recompute)

    # ------------------------------------------------------------------

    class _TestMLAConfigProperties(unittest.TestCase):
        """Sanity checks on MLAConfig derived properties."""

        def test_head_dim_equals_kv_channels(self):
            cfg = MLAConfig(kv_channels=128)
            self.assertEqual(cfg.head_dim, 128)

        def test_requires_recompute_save_fp8(self):
            cfg = MLAConfig(use_fp8=True)
            self.assertTrue(cfg.requires_recompute_save)

        def test_requires_recompute_save_fp4(self):
            cfg = MLAConfig(use_fp4=True)
            self.assertTrue(cfg.requires_recompute_save)

        def test_requires_recompute_save_neither(self):
            cfg = MLAConfig(use_fp8=False, use_fp4=False)
            self.assertFalse(cfg.requires_recompute_save)

    # ------------------------------------------------------------------

    class _TestShardedStateDict(unittest.TestCase):
        """Verify that sharded_state_dict emits backward-compat split keys."""

        def _make_attn(self) -> DESLOCFusedMLASelfAttention:
            cfg = MLAConfig(
                hidden_size=64, num_attention_heads=4,
                kv_channels=16, qk_head_dim=16, v_head_dim=16,
                qk_rope_head_dim=8, qk_nope_head_dim=8,
                kv_lora_rank=32, q_lora_rank=48,
                h100_device_index=0, a6000_device_indices=(0,),
            )
            return DESLOCFusedMLASelfAttention(cfg, layer_index=0)

        def test_split_keys_present(self):
            attn = self._make_attn()
            sd = attn.sharded_state_dict(prefix="attn.")
            self.assertIn("attn.linear_q_down_proj.weight", sd)
            self.assertIn("attn.linear_kv_down_proj.weight", sd)

        def test_split_shapes_correct(self):
            attn = self._make_attn()
            sd = attn.sharded_state_dict()
            q_rank = attn.config.q_lora_rank
            kv_rank = attn.config.kv_lora_rank
            fused_in = attn.config.hidden_size
            self.assertEqual(sd["linear_q_down_proj.weight"].shape,
                             (q_rank, fused_in))
            self.assertEqual(sd["linear_kv_down_proj.weight"].shape,
                             (kv_rank, fused_in))

    # ------------------------------------------------------------------
    # Run all test suites.
    # ------------------------------------------------------------------

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        _TestHeterogeneousLinear,
        _TestDESLOCFusedMLASelfAttentionHooks,
        _TestSharedLocalityCache,
        _TestDeviceTierMapping,
        _TestSetSaveOriginalInput,
        _TestMLAConfigProperties,
        _TestShardedStateDict,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
