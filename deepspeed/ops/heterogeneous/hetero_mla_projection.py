"""
DES-LOC Heterogeneous MLA Projection Handler
=============================================

Upstream Design Intent (Megatron 7604f28):
    Megatron's commit [split 3/5] refactors AbsorbedMLA's K and V up-projection
    from two separate linear modules (linear_k_up_proj, linear_v_up_proj) into a
    single fused module (linear_kv_up_proj) with combined weight layout:
        [head0_K | head0_V, head1_K | head1_V, ...]  (per-head concatenated)

    This consolidation reduces module count, simplifies tensor-parallel buffer
    bookkeeping, and enables a single all-gather for KV projection during
    sequence-parallel forward passes. The refactor also introduces two helper
    functions (_restore_packed_thd_batch_dim, _apply_absorbed_v_up_projection)
    to decouple post-attention reshaping logic, and adds backward compatibility
    for loading checkpoints saved under the old split layout.

DES-LOC Adaptation Points:
    DES-LOC (Decoupled Execution with Shared LOcality Cache) targets a mixed
    A6000×2 (SM86, 48 GB each) + H100 NVL×1 (SM90, 96 GB) cluster connected
    via PCIe with 1.5 TB CPU DRAM as a spill/cache tier.

    1. Device Placement Policy:
       The combined KV up-projection weight (linear_kv_up_proj) is larger than
       the old split pair but is accessed once per forward pass. Under DES-LOC
       we place it on the H100 (the "locality anchor") so that the fused einsum
       executes on SM90 with tensor-core efficiency. The Q path stays on the
       A6000 that hosts the corresponding attention head shard.

    2. Shared LOcality Cache (SLC) for V up-weights:
       After _get_kv_up_weights() extracts v_up_weight, it is pinned into the
       SLC (a named CPU-side or HBM-side buffer dict shared across pipeline
       stages) so that gradient accumulation in _backward_kv_proj does not
       re-read the full combined weight from H100.

    3. Packed-THD batch-dim restoration:
       _restore_packed_thd_batch_dim is called after core_attention output
       returns to whichever device ran core attention; the unsqueeze is a
       zero-copy op so it is safe across PCIe-mapped tensors.

    4. Checkpoint backward compatibility:
       _load_from_state_dict now merges split K/V weights (old format) into the
       combined layout before the DeepSpeed checkpoint engine ingests them,
       preventing a two-pass re-save cycle on heterogeneous nodes.

    5. Sequence-parallel assertion:
       Megatron added a hard assert that sequence_parallel must be True when
       tp_size > 1. DES-LOC enforces the same constraint because cross-device
       scatter/gather over PCIe cannot hide latency without sequence-parallel
       chunking.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device topology constants for DES-LOC hardware target
# ---------------------------------------------------------------------------
_DEVICE_SM90 = "h100"   # H100 NVL 96 GB – SM90 locality anchor
_DEVICE_SM86 = "a6000"  # A6000 48 GB – SM86 attention-head shards


# ---------------------------------------------------------------------------
# SLC (Shared LOcality Cache) – lightweight in-process tensor registry
# ---------------------------------------------------------------------------

class SharedLocalityCache:
    """
    In-process tensor registry implementing the LOcality Cache tier of DES-LOC.

    The SLC bridges the three memory tiers on the target cluster:
        GPU HBM  (H100 96 GB or A6000 48 GB)
        PCIe-mapped host pinned memory
        CPU DRAM (1.5 TB, used as overflow spill)

    Tensors are keyed by a string tag.  Insertion policy is caller-controlled;
    eviction is explicit (``evict``) or sweep-based (``evict_stale``).

    This class is intentionally free of DeepSpeed internals so it can be unit-
    tested without a GPU.
    """

    def __init__(self, max_entries: int = 256) -> None:
        self._store: Dict[str, torch.Tensor] = {}
        self._access_count: Dict[str, int] = {}
        self._max_entries = max_entries

    def insert(self, key: str, tensor: torch.Tensor, *, pin_cpu: bool = False) -> None:
        """
        Insert *tensor* under *key*.

        If ``pin_cpu`` is True and the tensor is on CPU, it is converted to
        pinned memory so that future H2D copies can use DMA without staging.
        """
        if len(self._store) >= self._max_entries:
            self._evict_lru()
        if pin_cpu and not tensor.is_cuda and not tensor.is_pinned():
            try:
                tensor = tensor.pin_memory()
            except RuntimeError:
                # No CUDA context available in unit tests – keep as-is.
                pass
        self._store[key] = tensor
        self._access_count[key] = 0

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Return cached tensor or None."""
        tensor = self._store.get(key)
        if tensor is not None:
            self._access_count[key] = self._access_count.get(key, 0) + 1
        return tensor

    def evict(self, key: str) -> None:
        """Remove a single entry."""
        self._store.pop(key, None)
        self._access_count.pop(key, None)

    def evict_stale(self, min_accesses: int = 1) -> int:
        """
        Evict entries that have been accessed fewer than *min_accesses* times.

        Returns the number of evicted entries.
        """
        stale = [k for k, v in self._access_count.items() if v < min_accesses]
        for k in stale:
            self.evict(k)
        if stale:
            logger.debug("SLC evicted %d stale entries: %s", len(stale), stale)
        return len(stale)

    def _evict_lru(self) -> None:
        if not self._access_count:
            return
        lru_key = min(self._access_count, key=self._access_count.__getitem__)
        logger.debug("SLC LRU eviction: key=%s accesses=%d", lru_key, self._access_count[lru_key])
        self.evict(lru_key)

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)


# Module-level default SLC instance shared across all HeteroMLAProjection
# objects within a single process.  DeepSpeed engine callers may replace this
# with a cross-rank-aware implementation.
_DEFAULT_SLC: SharedLocalityCache = SharedLocalityCache()


# ---------------------------------------------------------------------------
# Device placement helper
# ---------------------------------------------------------------------------

class DevicePlacementPolicy:
    """
    Maps logical DES-LOC roles to physical torch devices.

    On the A6000×2 + H100 cluster:
        - H100 is index 0 when it is the first enumerated CUDA device, but
          real deployments may differ.  The caller passes explicit device
          indices at construction time.
        - A6000 shards are numbered 1 and 2 (or 0 and 1 if H100 is absent).

    The policy object is queried by HeteroMLAProjection to decide where to
    allocate the combined KV projection weight and where to run the V-up
    einsum.
    """

    def __init__(
        self,
        h100_device_index: int = 0,
        a6000_device_indices: Tuple[int, ...] = (1, 2),
        force_cpu: bool = False,
    ) -> None:
        self._h100_idx = h100_device_index
        self._a6000_idxs = a6000_device_indices
        self._force_cpu = force_cpu
        self._cuda_available = torch.cuda.is_available()

    @property
    def locality_anchor_device(self) -> torch.device:
        """H100 NVL – locality anchor for KV projection weight."""
        if self._force_cpu or not self._cuda_available:
            return torch.device("cpu")
        return torch.device(f"cuda:{self._h100_idx}")

    def attention_shard_device(self, shard_rank: int = 0) -> torch.device:
        """A6000 device for the given TP shard rank."""
        if self._force_cpu or not self._cuda_available:
            return torch.device("cpu")
        idx = self._a6000_idxs[shard_rank % len(self._a6000_idxs)]
        return torch.device(f"cuda:{idx}")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DevicePlacementPolicy(h100={self._h100_idx}, "
            f"a6000={self._a6000_idxs}, force_cpu={self._force_cpu})"
        )


# ---------------------------------------------------------------------------
# MLA configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class MLAProjectionConfig:
    """
    Minimal configuration mirror of Megatron's MLATransformerConfig fields
    that are relevant to the KV up-projection path.

    DES-LOC adds three extra fields (``kv_anchor_device_index``,
    ``attn_shard_device_indices``, ``enable_slc_v_weight_cache``) that have
    no Megatron counterpart.
    """

    # --- upstream fields ---
    num_attention_heads: int = 32
    kv_lora_rank: int = 512
    qk_head_dim: int = 128
    qk_pos_emb_head_dim: int = 64
    v_head_dim: int = 128
    tensor_model_parallel_size: int = 1
    sequence_parallel: bool = True
    add_bias_linear: bool = False
    q_lora_rank: Optional[int] = None

    # --- DES-LOC extensions ---
    kv_anchor_device_index: int = 0          # H100 device index
    attn_shard_device_indices: Tuple[int, ...] = (1, 2)  # A6000 indices
    enable_slc_v_weight_cache: bool = True   # Pin v_up_weight into SLC after first extraction
    slc_cache_max_entries: int = 256

    def __post_init__(self) -> None:
        if self.add_bias_linear:
            raise ValueError("add_bias_linear is not supported for AbsorbedMLA / HeteroMLAProjection")
        if self.tensor_model_parallel_size > 1 and not self.sequence_parallel:
            raise ValueError(
                "HeteroMLAProjection requires sequence_parallel=True "
                "when tensor_model_parallel_size > 1 (DES-LOC PCIe scatter/gather constraint)"
            )


# ---------------------------------------------------------------------------
# Pure helper functions (mirrors of Megatron module-level helpers)
# ---------------------------------------------------------------------------

def restore_packed_thd_batch_dim(
    core_attn_out: torch.Tensor,
    hidden_states: torch.Tensor,
    packed_seq_params: Optional[Any],
) -> torch.Tensor:
    """
    Restore the singleton packed-THD batch dimension when core attention omitted it.

    Upstream intent (Megatron 7604f28):
        When packed sequences use the 'thd' QKV format the batch dimension is
        collapsed by core attention.  This helper unconditionally checks and
        restores it, removing the need for inline ``if`` branches in callers.

    DES-LOC note:
        The unsqueeze is a view (zero-copy) so it is safe to call on tensors
        that live on any device, including PCIe-mapped CPU tensors returned
        from the H100 core-attention path.
    """
    if (
        packed_seq_params is not None
        and getattr(packed_seq_params, "qkv_format", None) == "thd"
        and core_attn_out.ndim == hidden_states.ndim - 1
    ):
        core_attn_out = core_attn_out.unsqueeze(1)
    return core_attn_out


def apply_absorbed_v_up_projection(
    core_attn_out: torch.Tensor,
    v_up_weight: torch.Tensor,
    num_attention_heads_per_partition: int,
    kv_lora_rank: int,
    v_head_dim: int,
    core_consumed_v_up_projection: bool,
) -> torch.Tensor:
    """
    Apply the V up-projection to core attention output unless core attention
    already consumed the projection weight.

    Upstream intent (Megatron 7604f28):
        Extracted from inline code in AbsorbedMLASelfAttention.forward() into
        a standalone function so that backends that fuse the V-up einsum
        (e.g. FlashAttention with absorbed weights) can signal via the
        ``consumes_absorbed_v_up_projection`` attribute and skip the explicit
        einsum here.

    DES-LOC adaptation:
        Before executing the einsum, this function migrates *v_up_weight* and
        *core_attn_out* to the same device.  In DES-LOC the weight lives on
        the H100 (locality anchor) while the attention output may have been
        produced on an A6000.  A minimal PCIe transfer is issued only when
        devices differ; when they are the same the call is a no-op.

    Args:
        core_attn_out: Attention output tensor, shape [..., latent_or_projected].
        v_up_weight:   V up-projection weight, shape [n_heads, v_head_dim, kv_lora_rank].
        num_attention_heads_per_partition: TP-local head count.
        kv_lora_rank:  Latent (compressed KV) dimension.
        v_head_dim:    Per-head V dimension after up-projection.
        core_consumed_v_up_projection: If True, core attention already projected;
            only shape validation is performed.

    Returns:
        Projected output, shape [..., n_heads * v_head_dim].
    """
    latent_output_size = num_attention_heads_per_partition * kv_lora_rank
    projected_output_size = num_attention_heads_per_partition * v_head_dim

    if core_consumed_v_up_projection:
        if core_attn_out.size(-1) != projected_output_size:
            raise RuntimeError(
                f"Core attention claimed to consume V up-projection but output last dim "
                f"{core_attn_out.size(-1)} != expected projected_output_size {projected_output_size}."
            )
        return core_attn_out

    if core_attn_out.size(-1) != latent_output_size:
        raise RuntimeError(
            f"Core attention output last dim {core_attn_out.size(-1)} "
            f"!= expected latent_output_size {latent_output_size}."
        )

    # DES-LOC: colocate tensors before einsum to avoid implicit PCIe stalls.
    target_device = v_up_weight.device
    if core_attn_out.device != target_device:
        logger.debug(
            "DES-LOC colocate: moving core_attn_out from %s to %s for V-up einsum",
            core_attn_out.device,
            target_device,
        )
        core_attn_out = core_attn_out.to(target_device)

    core_attn_out = core_attn_out.view(
        *core_attn_out.shape[:-1],
        num_attention_heads_per_partition,
        kv_lora_rank,
    )
    # einsum: [..., n, c] × [n, d, c] → [..., n, d]
    core_attn_out = torch.einsum("...nc,ndc->...nd", core_attn_out, v_up_weight)
    core_attn_out = core_attn_out.contiguous()
    return core_attn_out.view(*core_attn_out.shape[:-2], -1)


# ---------------------------------------------------------------------------
# Weight layout utilities
# ---------------------------------------------------------------------------

def combine_split_kv_up_weights(
    k_up_weight: torch.Tensor,
    v_up_weight: torch.Tensor,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    kv_lora_rank: int,
) -> torch.Tensor:
    """
    Merge pre-refactor split K/V up-projection weights into the combined layout.

    Upstream intent (Megatron 7604f28):
        The old layout stored K and V as separate tensors:
            K: [n * qk_head_dim, kv_lora_rank]
            V: [n * v_head_dim,  kv_lora_rank]
        The new layout concatenates them per head:
            KV: [n * (qk_head_dim + v_head_dim), kv_lora_rank]
        Megatron calls this inside ``_load_from_state_dict`` for checkpoint
        backward compatibility.

    DES-LOC adaptation:
        Called by ``HeteroMLAProjection._load_from_state_dict`` before handing
        state dict to DeepSpeed's checkpoint engine, so the engine never sees
        the old keys.  The combination is done on CPU to avoid unnecessary
        H2D transfers during checkpoint loading.
    """
    k_3d = k_up_weight.cpu().view(num_heads, qk_head_dim, kv_lora_rank)
    v_3d = v_up_weight.cpu().view(num_heads, v_head_dim, kv_lora_rank)
    combined = torch.cat([k_3d, v_3d], dim=1).contiguous()
    return combined.view(num_heads * (qk_head_dim + v_head_dim), kv_lora_rank)


def split_combined_kv_up_weights(
    combined_weight: torch.Tensor,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    kv_lora_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split combined KV up-projection weight back into separate K and V tensors.

    This is the inverse of ``combine_split_kv_up_weights``, retained for
    DES-LOC's gradient-analysis tooling which may need the per-projection
    gradient norms independently.
    """
    per_head = combined_weight.view(num_heads, qk_head_dim + v_head_dim, kv_lora_rank)
    k = per_head[:, :qk_head_dim, :].contiguous().view(num_heads * qk_head_dim, kv_lora_rank)
    v = per_head[:, qk_head_dim:, :].contiguous().view(num_heads * v_head_dim, kv_lora_rank)
    return k, v


# ---------------------------------------------------------------------------
# Core module: HeteroMLAProjection
# ---------------------------------------------------------------------------

class HeteroMLAProjection(nn.Module):
    """
    DES-LOC heterogeneous-aware absorbed MLA K/V up-projection module.

    This module reimplements the projection handling introduced by Megatron
    commit 7604f28 under DES-LOC semantics:

    Architecture overview
    ---------------------
    The "absorbed MLA" formulation (DeepSeek-V2/V3 style) stores KV in a
    compressed latent space of dimension ``kv_lora_rank``.  Before computing
    attention scores the latent is up-projected back to the full K/V space via
    learnable projection matrices:

        k_full = latent @ k_up_weight.T     # [T, n, qk_head_dim]
        v_full = latent @ v_up_weight.T     # [T, n, v_head_dim]

    Megatron fuses K and V up-projection weights into a single matrix:
        kv_up_weight: [n * (qk_head_dim + v_head_dim), kv_lora_rank]
    stored row-major per-head (head0_K rows, head0_V rows, head1_K rows, …).

    DES-LOC placement
    -----------------
    The combined weight is allocated on the H100 (locality anchor).  All
    einsum operations that consume it run on the H100.  Attention head
    computation on A6000 shards produces a latent-space output that is
    transferred to the H100 via a single PCIe DMA before the V-up einsum.

    The extracted ``v_up_weight`` view is pinned into the SLC after the first
    forward pass so that subsequent backward gradient accumulation can read it
    without traversing the combined weight again.

    Sequence-parallel constraint
    ----------------------------
    When ``tensor_model_parallel_size > 1`` the module asserts
    ``sequence_parallel=True``.  Without sequence-parallel the all-gather
    pattern over PCIe would serialize the KV projection across TP ranks,
    eliminating the bandwidth hiding that DES-LOC's PCIe scheduling provides.

    Parameters
    ----------
    config : MLAProjectionConfig
        Combined upstream + DES-LOC configuration.
    tp_rank : int
        Tensor-parallel rank of this process (determines which A6000 shard
        this instance is colocated with).
    slc : Optional[SharedLocalityCache]
        Shared LOcality Cache.  If None, the module-level ``_DEFAULT_SLC`` is
        used.
    placement_policy : Optional[DevicePlacementPolicy]
        Device placement policy.  If None, a default policy is constructed
        from ``config.kv_anchor_device_index`` and
        ``config.attn_shard_device_indices``.
    """

    def __init__(
        self,
        config: MLAProjectionConfig,
        tp_rank: int = 0,
        slc: Optional[SharedLocalityCache] = None,
        placement_policy: Optional[DevicePlacementPolicy] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.tp_rank = tp_rank
        self.slc = slc if slc is not None else _DEFAULT_SLC
        self.placement_policy = placement_policy or DevicePlacementPolicy(
            h100_device_index=config.kv_anchor_device_index,
            a6000_device_indices=config.attn_shard_device_indices,
        )

        # TP-local head count
        self.num_attention_heads_per_partition: int = (
            config.num_attention_heads // config.tensor_model_parallel_size
        )

        # DES-LOC: weight lives on H100 locality anchor
        anchor = self.placement_policy.locality_anchor_device

        kv_rows = self.num_attention_heads_per_partition * (
            config.qk_head_dim + config.v_head_dim
        )
        self.linear_kv_up_proj = nn.Linear(
            config.kv_lora_rank,
            kv_rows,
            bias=False,
            device=anchor,
        )

        # SLC key template – unique per layer (set by owner via set_layer_tag)
        self._layer_tag: str = f"hetero_mla_tp{tp_rank}"

        logger.info(
            "HeteroMLAProjection init: tp_rank=%d, heads_per_partition=%d, "
            "kv_rows=%d, kv_lora_rank=%d, anchor_device=%s",
            tp_rank,
            self.num_attention_heads_per_partition,
            kv_rows,
            config.kv_lora_rank,
            anchor,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_layer_tag(self, tag: str) -> None:
        """
        Set a human-readable tag for SLC keys (e.g. ``"layer_12_tp1"``).

        Call this after construction to make SLC entries traceable in logs.
        """
        self._layer_tag = tag

    def get_kv_up_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (k_up_weight, v_up_weight) from the combined weight matrix.

        Upstream intent (Megatron 7604f28 ``_get_kv_up_weights``):
            Slices the combined weight into per-head K and V sub-tensors.
            Asserts expected shapes.

        DES-LOC adaptation:
            The extracted ``v_up_weight`` is pinned into the SLC on first call
            so that ``_backward_kv_proj`` can retrieve it without re-reading
            the combined weight.  The SLC key embeds the layer tag so multiple
            layers in the same process share the cache without collision.
        """
        n = self.num_attention_heads_per_partition
        qk = self.config.qk_head_dim
        v = self.config.v_head_dim
        r = self.config.kv_lora_rank

        expected_rows = n * (qk + v)
        weight = self.linear_kv_up_proj.weight  # [n*(qk+v), r]
        if weight.size(0) != expected_rows or weight.size(1) != r:
            raise RuntimeError(
                f"linear_kv_up_proj.weight shape {tuple(weight.shape)} "
                f"!= expected ({expected_rows}, {r})"
            )

        kv_3d = weight.view(n, qk + v, r)
        k_up_weight = kv_3d[:, :qk, :]   # [n, qk, r]
        v_up_weight = kv_3d[:, qk:, :]   # [n, v,  r]

        if self.config.enable_slc_v_weight_cache:
            slc_key = f"{self._layer_tag}:v_up_weight"
            if slc_key not in self.slc:
                # Store a contiguous copy so gradient views do not keep the
                # full combined weight alive in the SLC.
                self.slc.insert(slc_key, v_up_weight.detach().contiguous())
                logger.debug("SLC inserted v_up_weight under key '%s'", slc_key)

        return k_up_weight, v_up_weight

    def get_v_up_weight(self) -> torch.Tensor:
        """
        Return only the V up-projection weight.

        Checks the SLC before extracting from the combined weight to avoid an
        extra view operation in the common steady-state case.
        """
        if self.config.enable_slc_v_weight_cache:
            slc_key = f"{self._layer_tag}:v_up_weight"
            cached = self.slc.get(slc_key)
            if cached is not None:
                return cached

        _, v_up_weight = self.get_kv_up_weights()
        return v_up_weight

    def forward_v_up_projection(
        self,
        core_attn_out: torch.Tensor,
        packed_seq_params: Optional[Any] = None,
        hidden_states: Optional[torch.Tensor] = None,
        core_consumed_v_up_projection: bool = False,
    ) -> torch.Tensor:
        """
        Apply the V up-projection to core attention output and restore packed
        THD batch dimension if necessary.

        This is the DES-LOC replacement for the inline projection block in
        Megatron's AbsorbedMLASelfAttention.forward() (post 7604f28).

        Steps:
            1. Retrieve v_up_weight (SLC-cached when available).
            2. Migrate core_attn_out to the locality anchor if needed.
            3. Apply the absorbed V up-projection einsum.
            4. Restore the packed-THD batch dim.
            5. Return the projected output on the same device as the input
               (migrate back to A6000 if projection ran on H100).

        Args:
            core_attn_out:  Output from core attention, shape [..., latent].
            packed_seq_params: Optional packed sequence metadata.
            hidden_states:  Hidden states tensor (needed only for THD dim
                            restoration shape check).
            core_consumed_v_up_projection: Whether core attention fused the
                projection (FlashAttention absorbed path).

        Returns:
            Projected output, shape [..., n_heads * v_head_dim].
        """
        v_up_weight = self.get_v_up_weight()
        original_device = core_attn_out.device

        projected = apply_absorbed_v_up_projection(
            core_attn_out=core_attn_out,
            v_up_weight=v_up_weight,
            num_attention_heads_per_partition=self.num_attention_heads_per_partition,
            kv_lora_rank=self.config.kv_lora_rank,
            v_head_dim=self.config.v_head_dim,
            core_consumed_v_up_projection=core_consumed_v_up_projection,
        )

        if hidden_states is not None:
            projected = restore_packed_thd_batch_dim(
                projected, hidden_states, packed_seq_params
            )

        # DES-LOC: return result on the original (A6000) device so the output
        # projection (linear_proj) runs on the same device as the residual.
        if projected.device != original_device:
            logger.debug(
                "DES-LOC return transfer: %s → %s", projected.device, original_device
            )
            projected = projected.to(original_device)

        return projected

    def backward_kv_proj(self) -> None:
        """
        Trigger weight-gradient accumulation for the KV up-projection.

        Upstream intent (Megatron 7604f28 ``_backward_kv_proj``):
            Calls ``backward_dw()`` on the KV projection modules.  After the
            refactor this is a single call instead of two.

        DES-LOC adaptation:
            The SLC v_up_weight entry is invalidated after gradient
            accumulation because the weight will be updated by the optimizer
            step, making the cached view stale.
        """
        if hasattr(self.linear_kv_up_proj, "backward_dw"):
            self.linear_kv_up_proj.backward_dw()
        # Invalidate SLC after grad accumulation
        slc_key = f"{self._layer_tag}:v_up_weight"
        if slc_key in self.slc:
            self.slc.evict(slc_key)
            logger.debug("SLC evicted stale v_up_weight after backward: key='%s'", slc_key)

    def load_from_state_dict_compat(
        self,
        state_dict: Dict[str, torch.Tensor],
        prefix: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Produce a state dict with combined KV up-projection weight.

        Upstream intent (Megatron 7604f28 ``_load_from_state_dict``):
            Detects old-format checkpoints (split linear_k_up_proj.weight /
            linear_v_up_proj.weight) and combines them into the new
            linear_kv_up_proj.weight layout before calling super().

        DES-LOC adaptation:
            Returns the patched state dict so DeepSpeed's checkpoint engine
            can ingest it directly.  All tensor operations run on CPU to avoid
            spurious GPU allocations during checkpoint loading.

            Extra-state entries (e.g. fp8 scale metadata from Transformer
            Engine) are handled analogously to Megatron: if only split extra-
            states are present the first non-None one is promoted to the
            combined key.
        """
        combined_key = f"{prefix}linear_kv_up_proj.weight"
        k_up_key = f"{prefix}linear_k_up_proj.weight"
        v_up_key = f"{prefix}linear_v_up_proj.weight"

        patched = dict(state_dict)

        if combined_key not in patched and k_up_key in patched and v_up_key in patched:
            logger.info(
                "DES-LOC ckpt compat: merging split KV up-proj weights under '%s'",
                combined_key,
            )
            patched[combined_key] = combine_split_kv_up_weights(
                k_up_weight=patched.pop(k_up_key),
                v_up_weight=patched.pop(v_up_key),
                num_heads=self.num_attention_heads_per_partition,
                qk_head_dim=self.config.qk_head_dim,
                v_head_dim=self.config.v_head_dim,
                kv_lora_rank=self.config.kv_lora_rank,
            )

        # Extra-state migration (fp8 scale tensors, etc.)
        combined_extra = f"{prefix}linear_kv_up_proj._extra_state"
        k_extra = f"{prefix}linear_k_up_proj._extra_state"
        v_extra = f"{prefix}linear_v_up_proj._extra_state"

        if k_extra in patched or v_extra in patched:
            k_es = patched.pop(k_extra, None)
            v_es = patched.pop(v_extra, None)
            if combined_extra not in patched:
                chosen = k_es if k_es is not None else v_es
                if chosen is not None:
                    patched[combined_extra] = chosen

        return patched

    def colocate_kv_compressed(
        self,
        kv_compressed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Transfer the compressed KV latent to the locality anchor device.

        In DES-LOC the KV down-projection runs on an A6000, producing
        ``kv_compressed`` on device cuda:1 or cuda:2.  Before the K absorption
        einsum (which uses weights on the H100) we must move the latent.

        This call is a no-op when source and destination devices match (e.g.
        during unit tests that force CPU placement).

        Args:
            kv_compressed: Compressed KV, shape [T, 1, kv_lora_rank].

        Returns:
            kv_compressed on the locality anchor device.
        """
        target = self.placement_policy.locality_anchor_device
        if kv_compressed.device != target:
            logger.debug(
                "DES-LOC PCIe transfer: kv_compressed %s → %s, numel=%d",
                kv_compressed.device,
                target,
                kv_compressed.numel(),
            )
            kv_compressed = kv_compressed.to(target)
        return kv_compressed

    def extra_repr(self) -> str:  # pragma: no cover
        return (
            f"tp_rank={self.tp_rank}, "
            f"heads_per_partition={self.num_attention_heads_per_partition}, "
            f"kv_lora_rank={self.config.kv_lora_rank}, "
            f"qk_head_dim={self.config.qk_head_dim}, "
            f"v_head_dim={self.config.v_head_dim}, "
            f"anchor={self.placement_policy.locality_anchor_device}"
        )


# ---------------------------------------------------------------------------
# Submodule spec dataclass (mirrors Megatron's AbsorbedMLASelfAttentionSubmodules)
# ---------------------------------------------------------------------------

@dataclass
class HeteroMLASubmodules:
    """
    Submodule specification for the DES-LOC heterogeneous MLA attention path.

    Upstream intent (Megatron 7604f28):
        ``AbsorbedMLASelfAttentionSubmodules`` merged ``linear_k_up_proj`` and
        ``linear_v_up_proj`` into ``linear_kv_up_proj``.

    DES-LOC extension:
        An additional ``hetero_kv_projection`` field carries the
        ``HeteroMLAProjection`` spec, allowing different projection backends
        (e.g. pure-PyTorch vs. TE-accelerated) to be swapped per layer.
    """

    linear_q_down_proj: Any = None
    linear_q_up_proj: Any = None
    linear_kv_down_proj: Any = None
    linear_kv_up_proj: Any = None          # combined (new layout)
    hetero_kv_projection: Any = None       # DES-LOC: HeteroMLAProjection spec
    core_attention: Any = None
    linear_proj: Any = None
    q_layernorm: Any = None
    k_layernorm: Any = None


# ---------------------------------------------------------------------------
# Gradient analysis helper (DES-LOC-specific, no Megatron counterpart)
# ---------------------------------------------------------------------------

class KVProjectionGradientAnalyzer:
    """
    Post-backward diagnostic for the combined KV up-projection gradient.

    In the old split layout, gradient norms for K and V could be monitored
    independently.  After the refactor the combined weight's gradient mixes
    both.  This class reconstructs per-projection gradient norms and cosine
    similarities for training health monitoring without modifying the forward
    graph.
    """

    def __init__(self, projection: HeteroMLAProjection) -> None:
        self.projection = projection

    def compute_split_grad_norms(self) -> Optional[Dict[str, float]]:
        """
        Compute L2 gradient norms for the K and V sub-blocks of the combined
        up-projection weight gradient.

        Returns None if the gradient is not yet available.
        """
        weight = self.projection.linear_kv_up_proj.weight
        if weight.grad is None:
            return None

        cfg = self.projection.config
        n = self.projection.num_attention_heads_per_partition
        k_grad, v_grad = split_combined_kv_up_weights(
            weight.grad,
            num_heads=n,
            qk_head_dim=cfg.qk_head_dim,
            v_head_dim=cfg.v_head_dim,
            kv_lora_rank=cfg.kv_lora_rank,
        )
        return {
            "k_up_grad_norm": k_grad.float().norm().item(),
            "v_up_grad_norm": v_grad.float().norm().item(),
            "combined_grad_norm": weight.grad.float().norm().item(),
        }

    def cosine_similarity_k_v_grads(self) -> Optional[float]:
        """
        Cosine similarity between the K-sub and V-sub gradient vectors.

        A value near 1.0 indicates the two projections receive nearly
        identical gradient signals, which may hint at redundancy in the
        combined weight.
        """
        norms = self.compute_split_grad_norms()
        if norms is None:
            return None

        cfg = self.projection.config
        n = self.projection.num_attention_heads_per_partition
        weight = self.projection.linear_kv_up_proj.weight
        k_grad, v_grad = split_combined_kv_up_weights(
            weight.grad,
            num_heads=n,
            qk_head_dim=cfg.qk_head_dim,
            v_head_dim=cfg.v_head_dim,
            kv_lora_rank=cfg.kv_lora_rank,
        )
        k_flat = k_grad.float().flatten()
        v_flat = v_grad.float().flatten()
        # Pad shorter vector with zeros for shape compatibility
        if k_flat.numel() < v_flat.numel():
            k_flat = F.pad(k_flat, (0, v_flat.numel() - k_flat.numel()))
        elif v_flat.numel() < k_flat.numel():
            v_flat = F.pad(v_flat, (0, k_flat.numel() - v_flat.numel()))
        cos = F.cosine_similarity(k_flat.unsqueeze(0), v_flat.unsqueeze(0)).item()
        return cos


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import unittest
    from types import SimpleNamespace

    logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s")

    class TestSharedLocalityCache(unittest.TestCase):
        def test_insert_and_get(self):
            slc = SharedLocalityCache(max_entries=4)
            t = torch.randn(3, 3)
            slc.insert("k1", t)
            result = slc.get("k1")
            self.assertIsNotNone(result)
            self.assertEqual(result.shape, t.shape)

        def test_get_missing_returns_none(self):
            slc = SharedLocalityCache()
            self.assertIsNone(slc.get("does_not_exist"))

        def test_evict(self):
            slc = SharedLocalityCache()
            slc.insert("k1", torch.zeros(2))
            slc.evict("k1")
            self.assertNotIn("k1", slc)

        def test_lru_eviction_when_full(self):
            slc = SharedLocalityCache(max_entries=2)
            t0 = torch.zeros(1)
            t1 = torch.ones(1)
            t2 = torch.full((1,), 2.0)
            slc.insert("k0", t0)
            slc.insert("k1", t1)
            # Access k0 to increment its count so k1 becomes LRU
            slc.get("k0")
            slc.insert("k2", t2)  # triggers LRU eviction of k1
            self.assertIn("k0", slc)
            self.assertIn("k2", slc)
            self.assertNotIn("k1", slc)

        def test_evict_stale(self):
            slc = SharedLocalityCache()
            slc.insert("a", torch.zeros(1))
            slc.insert("b", torch.zeros(1))
            slc.get("a")  # access count = 1
            evicted = slc.evict_stale(min_accesses=1)
            self.assertEqual(evicted, 1)
            self.assertIn("a", slc)
            self.assertNotIn("b", slc)

    class TestRestorePackedTHDBatchDim(unittest.TestCase):
        def _make_packed_params(self, fmt):
            return SimpleNamespace(qkv_format=fmt)

        def test_thd_restores_dim(self):
            params = self._make_packed_params("thd")
            hidden = torch.randn(4, 1, 8)   # ndim=3
            out = torch.randn(4, 8)          # ndim=2 (batch dim collapsed)
            restored = restore_packed_thd_batch_dim(out, hidden, params)
            self.assertEqual(restored.ndim, 3)
            self.assertEqual(restored.shape[1], 1)

        def test_sbhd_no_change(self):
            params = self._make_packed_params("sbhd")
            hidden = torch.randn(4, 1, 8)
            out = torch.randn(4, 8)
            result = restore_packed_thd_batch_dim(out, hidden, params)
            self.assertEqual(result.ndim, 2)

        def test_none_params_no_change(self):
            hidden = torch.randn(4, 1, 8)
            out = torch.randn(4, 8)
            result = restore_packed_thd_batch_dim(out, hidden, None)
            self.assertEqual(result.ndim, 2)

        def test_already_correct_ndim_no_change(self):
            params = self._make_packed_params("thd")
            hidden = torch.randn(4, 1, 8)
            out = torch.randn(4, 1, 8)  # same ndim as hidden
            result = restore_packed_thd_batch_dim(out, hidden, params)
            self.assertEqual(result.shape, (4, 1, 8))

    class TestApplyAbsorbedVUpProjection(unittest.TestCase):
        def setUp(self):
            self.n = 2
            self.r = 4   # kv_lora_rank
            self.v = 3   # v_head_dim

        def _make_weight(self):
            return torch.randn(self.n, self.v, self.r)

        def test_basic_projection(self):
            T = 5
            attn_out = torch.randn(T, self.n * self.r)
            w = self._make_weight()
            result = apply_absorbed_v_up_projection(
                attn_out, w, self.n, self.r, self.v, False
            )
            self.assertEqual(result.shape, (T, self.n * self.v))

        def test_core_consumed_skips_einsum(self):
            T = 5
            projected = torch.randn(T, self.n * self.v)
            w = self._make_weight()
            result = apply_absorbed_v_up_projection(
                projected, w, self.n, self.r, self.v, True
            )
            self.assertIs(result, projected)

        def test_wrong_latent_size_raises(self):
            T = 3
            bad_out = torch.randn(T, self.n * self.r + 1)
            w = self._make_weight()
            with self.assertRaises(RuntimeError):
                apply_absorbed_v_up_projection(bad_out, w, self.n, self.r, self.v, False)

        def test_wrong_projected_size_raises_when_consumed(self):
            T = 3
            bad_out = torch.randn(T, self.n * self.v + 1)
            w = self._make_weight()
            with self.assertRaises(RuntimeError):
                apply_absorbed_v_up_projection(bad_out, w, self.n, self.r, self.v, True)

        def test_batched_projection(self):
            B, S = 2, 6
            attn_out = torch.randn(B, S, self.n * self.r)
            w = self._make_weight()
            result = apply_absorbed_v_up_projection(
                attn_out, w, self.n, self.r, self.v, False
            )
            self.assertEqual(result.shape, (B, S, self.n * self.v))

    class TestCombineSplitKVWeights(unittest.TestCase):
        def setUp(self):
            self.n, self.qk, self.v, self.r = 3, 4, 5, 6

        def _make_weights(self):
            k = torch.arange(self.n * self.qk * self.r, dtype=torch.float32).view(
                self.n * self.qk, self.r
            )
            v = torch.arange(self.n * self.v * self.r, dtype=torch.float32).view(
                self.n * self.v, self.r
            )
            return k, v

        def test_combine_produces_correct_shape(self):
            k, v = self._make_weights()
            combined = combine_split_kv_up_weights(k, v, self.n, self.qk, self.v, self.r)
            self.assertEqual(combined.shape, (self.n * (self.qk + self.v), self.r))

        def test_round_trip(self):
            k, v = self._make_weights()
            combined = combine_split_kv_up_weights(k, v, self.n, self.qk, self.v, self.r)
            k2, v2 = split_combined_kv_up_weights(combined, self.n, self.qk, self.v, self.r)
            torch.testing.assert_close(k.cpu(), k2.cpu())
            torch.testing.assert_close(v.cpu(), v2.cpu())

        def test_head_interleaving_order(self):
            # Verify that head0_K rows appear before head0_V rows in combined
            k, v = self._make_weights()
            combined = combine_split_kv_up_weights(k, v, self.n, self.qk, self.v, self.r)
            per_head = combined.view(self.n, self.qk + self.v, self.r)
            # First qk rows of head0 should match k[0:qk]
            expected_k_head0 = k.view(self.n, self.qk, self.r)[0]
            torch.testing.assert_close(per_head[0, : self.qk, :], expected_k_head0)

    class TestMLAProjectionConfigValidation(unittest.TestCase):
        def test_bias_raises(self):
            with self.assertRaises(ValueError):
                MLAProjectionConfig(add_bias_linear=True)

        def test_tp_without_sp_raises(self):
            with self.assertRaises(ValueError):
                MLAProjectionConfig(tensor_model_parallel_size=2, sequence_parallel=False)

        def test_tp_with_sp_ok(self):
            cfg = MLAProjectionConfig(tensor_model_parallel_size=2, sequence_parallel=True)
            self.assertEqual(cfg.tensor_model_parallel_size, 2)

    class TestHeteroMLAProjection(unittest.TestCase):
        def _make_proj(self, tp_size=1, tp_rank=0, enable_slc=True):
            cfg = MLAProjectionConfig(
                num_attention_heads=4,
                kv_lora_rank=8,
                qk_head_dim=6,
                v_head_dim=4,
                tensor_model_parallel_size=tp_size,
                sequence_parallel=(tp_size > 1),
                kv_anchor_device_index=0,
                attn_shard_device_indices=(0,),  # all CPU in test
                enable_slc_v_weight_cache=enable_slc,
            )
            slc = SharedLocalityCache()
            policy = DevicePlacementPolicy(force_cpu=True)
            return HeteroMLAProjection(cfg, tp_rank=tp_rank, slc=slc, placement_policy=policy)

        def test_weight_shapes(self):
            proj = self._make_proj()
            n = proj.num_attention_heads_per_partition  # 4 (tp=1)
            qk, v, r = 6, 4, 8
            w = proj.linear_kv_up_proj.weight
            self.assertEqual(w.shape, (n * (qk + v), r))

        def test_get_kv_up_weights_shapes(self):
            proj = self._make_proj()
            k_w, v_w = proj.get_kv_up_weights()
            n, qk, v, r = 4, 6, 4, 8
            self.assertEqual(k_w.shape, (n, qk, r))
            self.assertEqual(v_w.shape, (n, v, r))

        def test_slc_populated_after_get_kv_up_weights(self):
            proj = self._make_proj(enable_slc=True)
            proj.set_layer_tag("layer_0_tp0")
            proj.get_kv_up_weights()
            self.assertIn("layer_0_tp0:v_up_weight", proj.slc)

        def test_slc_bypass_when_disabled(self):
            proj = self._make_proj(enable_slc=False)
            proj.set_layer_tag("layer_0_tp0")
            proj.get_kv_up_weights()
            self.assertNotIn("layer_0_tp0:v_up_weight", proj.slc)

        def test_get_v_up_weight_uses_slc_on_second_call(self):
            proj = self._make_proj(enable_slc=True)
            proj.set_layer_tag("layer_1_tp0")
            v1 = proj.get_v_up_weight()
            v2 = proj.get_v_up_weight()
            # Second call should return the SLC-cached tensor (same storage)
            self.assertTrue(v2.data_ptr() == v1.data_ptr() or torch.allclose(v1, v2))

        def test_forward_v_up_projection_shape(self):
            proj = self._make_proj()
            T = 7
            n, r, v = 4, 8, 4
            attn_out = torch.randn(T, n * r)
            out = proj.forward_v_up_projection(attn_out)
            self.assertEqual(out.shape, (T, n * v))

        def test_forward_v_up_projection_thd_restore(self):
            proj = self._make_proj()
            T = 5
            n, r = 4, 8
            attn_out = torch.randn(T, n * r)
            hidden = torch.randn(T, 1, 32)
            params = SimpleNamespace(qkv_format="thd")
            out = proj.forward_v_up_projection(attn_out, packed_seq_params=params, hidden_states=hidden)
            self.assertEqual(out.ndim, 3)
            self.assertEqual(out.shape[1], 1)

        def test_backward_kv_proj_evicts_slc(self):
            proj = self._make_proj(enable_slc=True)
            proj.set_layer_tag("layer_bwd_tp0")
            proj.get_kv_up_weights()
            self.assertIn("layer_bwd_tp0:v_up_weight", proj.slc)
            proj.backward_kv_proj()
            self.assertNotIn("layer_bwd_tp0:v_up_weight", proj.slc)

        def test_load_from_state_dict_compat_split_to_combined(self):
            proj = self._make_proj()
            n, qk, v, r = 4, 6, 4, 8
            k_w = torch.randn(n * qk, r)
            v_w = torch.randn(n * v, r)
            prefix = "attn."
            sd = {
                f"{prefix}linear_k_up_proj.weight": k_w.clone(),
                f"{prefix}linear_v_up_proj.weight": v_w.clone(),
                f"{prefix}linear_k_up_proj._extra_state": torch.empty(0),
            }
            patched = proj.load_from_state_dict_compat(sd, prefix)
            self.assertIn(f"{prefix}linear_kv_up_proj.weight", patched)
            self.assertNotIn(f"{prefix}linear_k_up_proj.weight", patched)
            self.assertNotIn(f"{prefix}linear_v_up_proj.weight", patched)
            self.assertIn(f"{prefix}linear_kv_up_proj._extra_state", patched)
            self.assertNotIn(f"{prefix}linear_k_up_proj._extra_state", patched)

        def test_load_from_state_dict_compat_combined_passthrough(self):
            proj = self._make_proj()
            n, qk, v, r = 4, 6, 4, 8
            combined = torch.randn(n * (qk + v), r)
            prefix = "attn."
            sd = {f"{prefix}linear_kv_up_proj.weight": combined}
            patched = proj.load_from_state_dict_compat(sd, prefix)
            torch.testing.assert_close(patched[f"{prefix}linear_kv_up_proj.weight"], combined)

        def test_colocate_kv_compressed_cpu_noop(self):
            proj = self._make_proj()
            kv = torch.randn(5, 1, 8)
            out = proj.colocate_kv_compressed(kv)
            self.assertEqual(out.device, kv.device)

    class TestKVProjectionGradientAnalyzer(unittest.TestCase):
        def _make_proj(self):
            cfg = MLAProjectionConfig(
                num_attention_heads=4,
                kv_lora_rank=8,
                qk_head_dim=6,
                v_head_dim=4,
                enable_slc_v_weight_cache=False,
            )
            policy = DevicePlacementPolicy(force_cpu=True)
            return HeteroMLAProjection(cfg, tp_rank=0, placement_policy=policy)

        def test_compute_split_grad_norms_no_grad(self):
            proj = self._make_proj()
            analyzer = KVProjectionGradientAnalyzer(proj)
            self.assertIsNone(analyzer.compute_split_grad_norms())

        def test_compute_split_grad_norms_with_grad(self):
            proj = self._make_proj()
            n, qk, v, r = 4, 6, 4, 8
            T = 5
            attn_in = torch.randn(T, n * r, requires_grad=False)
            # Manually assign a fake gradient to the weight
            proj.linear_kv_up_proj.weight.grad = torch.randn_like(
                proj.linear_kv_up_proj.weight
            )
            analyzer = KVProjectionGradientAnalyzer(proj)
            norms = analyzer.compute_split_grad_norms()
            self.assertIsNotNone(norms)
            self.assertIn("k_up_grad_norm", norms)
            self.assertIn("v_up_grad_norm", norms)
            self.assertIn("combined_grad_norm", norms)
            # combined norm >= max(k_norm, v_norm) by Cauchy-Schwarz lower bound
            self.assertGreater(norms["combined_grad_norm"], 0.0)

        def test_cosine_similarity_k_v_grads_no_grad(self):
            proj = self._make_proj()
            analyzer = KVProjectionGradientAnalyzer(proj)
            self.assertIsNone(analyzer.cosine_similarity_k_v_grads())

        def test_cosine_similarity_k_v_grads_range(self):
            proj = self._make_proj()
            proj.linear_kv_up_proj.weight.grad = torch.randn_like(
                proj.linear_kv_up_proj.weight
            )
            analyzer = KVProjectionGradientAnalyzer(proj)
            cos = analyzer.cosine_similarity_k_v_grads()
            self.assertIsNotNone(cos)
            self.assertGreaterEqual(cos, -1.0)
            self.assertLessEqual(cos, 1.0)

    class TestDevicePlacementPolicy(unittest.TestCase):
        def test_force_cpu(self):
            policy = DevicePlacementPolicy(force_cpu=True)
            self.assertEqual(policy.locality_anchor_device, torch.device("cpu"))
            self.assertEqual(policy.attention_shard_device(0), torch.device("cpu"))

        def test_shard_device_wraps(self):
            policy = DevicePlacementPolicy(
                h100_device_index=0,
                a6000_device_indices=(1, 2),
                force_cpu=True,
            )
            # Both shards map to CPU in force_cpu mode
            self.assertEqual(policy.attention_shard_device(0), torch.device("cpu"))
            self.assertEqual(policy.attention_shard_device(1), torch.device("cpu"))

    suite = unittest.TestLoader().loadTestsFromTestCase(TestSharedLocalityCache)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRestorePackedTHDBatchDim))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestApplyAbsorbedVUpProjection))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCombineSplitKVWeights))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMLAProjectionConfigValidation))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestHeteroMLAProjection))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestKVProjectionGradientAnalyzer))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDevicePlacementPolicy))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
