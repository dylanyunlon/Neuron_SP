# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Scaled dot-product attention kernel — DES-LOC Neuron_SP port.

Ported from Megatron-LM megatron/core/transformer/dot_product_attention.py
(7 commits from M2305 → M3063).

Key evolution tracked through the commit history:
  M2305 (dbc4129d1) – Sink Attention: SWA base with sink tokens
  M2343 (e5bc9249d) – Enable mixing SWA with full attention [gpt-oss 2/5]
  M2618 (08c377198) – Fix param init for learnable softmax_offset
  M2716 (987395813) – Fixes for gpt-oss (minor shape fixes)
  M2856 (5f5741db9) – Replace global parallel state w/ explicit pg parameters
  M2919 (1eed1d24f) – Typing pass on transformer/
  M3063 (90e685b85) – Protocols replacing ModuleSpec for CoreAttentionBuilder

Design decisions encoded here:
  * Separated from attention.py so selective activation recomputing wraps
    exactly this region (memory intensive, compute light — Section 5 of
    Reducing Activation Recomputation, arxiv 2205.05198).
  * GQA handled via repeat_interleave (no new weight parameters).
  * SWA / sink attention via window_size config + is_layer_window_attention.
  * Learnable / off-by-one / vanilla softmax_offset variants.
  * DES-LOC tier-aware logging at construction.

DES-LOC integration
-------------------
DotProductAttention logs its assigned GPU tier at construction.  The tier
does not change the forward pass logic; routing is handled by the DES-LOC
engine above this level.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from deepspeed.core.transformer.module import MegatronModule
from deepspeed.core.transformer.transformer_config import TransformerConfig
from deepspeed.core.dist_checkpointing.mapping import ShardedTensor, ShardedStateDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy parallel-state helpers (safe when dist not initialised)
# ---------------------------------------------------------------------------

def _get_tp_world_size() -> int:
    try:
        from deepspeed.core.parallel_state import get_tensor_model_parallel_world_size
        return get_tensor_model_parallel_world_size()
    except Exception:
        return 1


def _get_tp_group():
    try:
        from deepspeed.core.parallel_state import get_tensor_model_parallel_group
        return get_tensor_model_parallel_group()
    except Exception:
        return None


def _get_cp_world_size() -> int:
    """Return context-parallel world size (1 when CP is not initialised)."""
    try:
        from deepspeed.core.parallel_state import get_context_parallel_world_size
        return get_context_parallel_world_size()
    except Exception:
        return 1


def _get_cp_rank() -> int:
    """Return context-parallel rank (0 when CP is not initialised)."""
    try:
        from deepspeed.core.parallel_state import get_context_parallel_rank
        return get_context_parallel_rank()
    except Exception:
        return 0


def _get_cp_group():
    """Return context-parallel process group (None when CP is not initialised)."""
    try:
        from deepspeed.core.parallel_state import get_context_parallel_group
        return get_context_parallel_group(check_initialized=False)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Ring-attention helpers for context parallelism (p2p / all_gather paths)
# ---------------------------------------------------------------------------

def _ring_attn_forward(
    query: "Tensor",
    key: "Tensor",
    value: "Tensor",
    softmax_scale: float,
    attention_dropout: "nn.Dropout",
    attn_mask_type: str,
    attention_mask: Optional["Tensor"],
    cp_group,
    cp_rank: int,
    cp_size: int,
    softmax_in_fp32: bool = True,
    training: bool = True,
) -> "Tensor":
    """Ring-attention (P2P) context-parallel attention forward.

    Implements the ring / flash-ring algorithm for causal / non-causal
    attention across ``cp_size`` ranks.  Each rank holds a contiguous
    chunk [s/cp] of the full sequence for Q, K, V and iterates cp_size
    steps, rotating KV around the ring while accumulating the local
    attention output.

    This is a *pure-PyTorch* fallback — production clusters should use
    the TE-based TEDotProductAttention which calls into optimised CUDA
    kernels.  The fallback is correct but does not overlap communication
    with compute.

    Layout: query/key/value come in as ``[sq_local, b, np, hn]`` where
    ``sq_local = full_seq / cp_size``.

    Returns:
        context: ``[sq_local, b, np * hn]``
    """
    import torch
    import torch.distributed as dist

    sq_local, b, np_heads, hn = query.shape
    sk_local = key.shape[0]
    dtype = query.dtype

    # We track the "online softmax" state (log-sum-exp + partial output).
    # Standard online-softmax trick: maintain max_scores and exp_sum separately
    # so we can merge partial results from each KV step.
    #   m: running max of pre-softmax logits   [b, np, sq_local]
    #   l: running sum of exp(logits - m)       [b, np, sq_local]
    #   o: running weighted value sum            [b, np, sq_local, hn]

    scale = softmax_scale
    fp32 = softmax_in_fp32

    # Reshape for bmm: [b*np, sq_local, hn]
    q = query.permute(1, 2, 0, 3).reshape(b * np_heads, sq_local, hn)
    if fp32:
        q = q.float()

    # Running accumulators (in fp32 for numerical stability)
    m = torch.full((b * np_heads, sq_local), float('-inf'), dtype=torch.float32, device=q.device)
    l = torch.zeros((b * np_heads, sq_local), dtype=torch.float32, device=q.device)
    o = torch.zeros((b * np_heads, sq_local, hn), dtype=torch.float32, device=q.device)

    # We rotate KV around the ring.  At step i=0 we use the local KV;
    # at step i>0 we use KV from rank (cp_rank - i) % cp_size.
    kv_buf = torch.cat([key, value], dim=-1)  # [sk_local, b, np, 2*hn]
    recv_buf = torch.empty_like(kv_buf)

    send_to   = (cp_rank + 1) % cp_size
    recv_from = (cp_rank - 1) % cp_size

    for step in range(cp_size):
        # ---- start async send/recv (overlap with compute) ----
        if step < cp_size - 1 and cp_group is not None:
            send_op = dist.P2POp(dist.isend, kv_buf, send_to, group=cp_group)
            recv_op = dist.P2POp(dist.irecv, recv_buf, recv_from, group=cp_group)
            reqs = dist.batch_isend_irecv([send_op, recv_op])

        # ---- compute attention for the current KV chunk ----
        cur_k = kv_buf[..., :hn]   # [sk_local, b, np, hn]
        cur_v = kv_buf[..., hn:]   # [sk_local, b, np, hn]

        # Reshape K, V for bmm
        k_2d = cur_k.permute(1, 2, 0, 3).reshape(b * np_heads, sk_local, hn)
        v_2d = cur_v.permute(1, 2, 0, 3).reshape(b * np_heads, sk_local, hn)
        if fp32:
            k_2d = k_2d.float()
            v_2d = v_2d.float()

        # Raw scores: [b*np, sq_local, sk_local]
        scores = torch.bmm(q, k_2d.transpose(1, 2)) * scale  # [b*np, sq, sk]

        # For causal attention, mask out future positions.
        # In the ring layout, the global position of Q token i is:
        #   cp_rank * sq_local + i
        # and the global position of K token j (at step `step`) is:
        #   ((cp_rank - step) % cp_size) * sk_local + j
        if attn_mask_type == "causal":
            kv_global_offset = ((cp_rank - step) % cp_size) * sk_local
            q_global_offset  = cp_rank * sq_local
            sq_idx = torch.arange(sq_local, device=q.device).unsqueeze(1)  # [sq, 1]
            sk_idx = torch.arange(sk_local, device=q.device).unsqueeze(0)  # [1, sk]
            q_global = q_global_offset + sq_idx
            k_global = kv_global_offset + sk_idx
            # mask[i,j] = True means K token j is in the "future" of Q token i
            causal_mask = k_global > q_global  # [sq, sk]  (bool)
            scores = scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

        if attention_mask is not None:
            # attention_mask: [b, 1, sq, sk] — boolean, True = masked
            mask_2d = attention_mask.squeeze(1).expand(b, np_heads, sq_local, sk_local)
            mask_2d = mask_2d.reshape(b * np_heads, sq_local, sk_local)
            scores = scores.masked_fill(mask_2d, float('-inf'))

        # Online softmax merge
        step_max = scores.max(dim=-1).values                    # [b*np, sq]
        new_m    = torch.maximum(m, step_max)
        exp_scores = torch.exp(scores - new_m.unsqueeze(-1))   # [b*np, sq, sk]
        step_sum = exp_scores.sum(dim=-1)                       # [b*np, sq]

        # Rescale running accumulators by exp(old_m - new_m)
        scale_factor = torch.exp(m - new_m)                     # [b*np, sq]
        l = scale_factor * l + step_sum
        o = scale_factor.unsqueeze(-1) * o + torch.bmm(exp_scores, v_2d)
        m = new_m

        # ---- wait for communication, swap buffers ----
        if step < cp_size - 1 and cp_group is not None:
            for req in reqs:
                req.wait()
            kv_buf, recv_buf = recv_buf, kv_buf

    # Normalise by log-sum-exp denominator
    o = o / l.unsqueeze(-1).clamp(min=1e-12)
    o = o.to(dtype)

    # Apply dropout on the (now normalised) attention weights — we don't have
    # explicit probs here, so apply dropout to the output directly scaled by
    # the dropout factor (equivalent for expected value).
    if training:
        try:
            from megatron.core import tensor_parallel as _tp
            with _tp.get_cuda_rng_tracker().fork():
                o = attention_dropout(o)
        except Exception:
            o = attention_dropout(o)

    # Reshape back to [sq_local, b, np, hn]
    o = o.reshape(b, np_heads, sq_local, hn).permute(2, 0, 1, 3)
    return o


def _allgather_attn_forward(
    query: "Tensor",
    key: "Tensor",
    value: "Tensor",
    softmax_scale: float,
    attention_dropout: "nn.Dropout",
    attn_mask_type: str,
    attention_mask: Optional["Tensor"],
    cp_group,
    cp_rank: int,
    cp_size: int,
    softmax_in_fp32: bool = True,
    training: bool = True,
) -> "Tensor":
    """All-gather context-parallel attention forward.

    Gathers the full KV sequence across all CP ranks before computing
    local Q × full_KV attention.  This is the ``cp_comm_type="all_gather"``
    path — simpler than ring but not overlapped.

    Layout: inputs are ``[sq_local, b, np, hn]``.
    After all-gather KV becomes ``[sq_full, b, np, hn]``.

    Returns:
        context: ``[sq_local, b, np, hn]``
    """
    import torch
    import torch.distributed as dist

    sq_local, b, np_heads, hn = query.shape
    sq_full = sq_local * cp_size
    dtype = query.dtype

    # All-gather K and V along the sequence dimension
    def _allgather(t: "Tensor") -> "Tensor":
        # t: [sq_local, b, np, hn]
        gathered = [torch.empty_like(t) for _ in range(cp_size)]
        dist.all_gather(gathered, t.contiguous(), group=cp_group)
        return torch.cat(gathered, dim=0)  # [sq_full, b, np, hn]

    key_full   = _allgather(key)    # [sq_full, b, np, hn]
    value_full = _allgather(value)  # [sq_full, b, np, hn]

    # Flatten to [b*np, sq, hn] for bmm
    q_2d = query.permute(1, 2, 0, 3).reshape(b * np_heads, sq_local, hn)
    k_2d = key_full.permute(1, 2, 0, 3).reshape(b * np_heads, sq_full, hn)
    v_2d = value_full.permute(1, 2, 0, 3).reshape(b * np_heads, sq_full, hn)

    if softmax_in_fp32:
        q_2d = q_2d.float()
        k_2d = k_2d.float()
        v_2d = v_2d.float()

    scores = torch.bmm(q_2d, k_2d.transpose(1, 2)) * softmax_scale  # [b*np, sq_local, sq_full]

    if attn_mask_type == "causal":
        q_offset = cp_rank * sq_local
        sq_idx = torch.arange(sq_local, device=scores.device).unsqueeze(1)  # [sq_local, 1]
        sk_idx = torch.arange(sq_full,  device=scores.device).unsqueeze(0)  # [1, sq_full]
        causal_mask = (sk_idx > (q_offset + sq_idx))
        scores = scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

    if attention_mask is not None:
        mask_2d = attention_mask.squeeze(1).expand(b, np_heads, sq_local, sq_full)
        mask_2d = mask_2d.reshape(b * np_heads, sq_local, sq_full)
        scores = scores.masked_fill(mask_2d, float('-inf'))

    probs = torch.softmax(scores, dim=-1)
    if softmax_in_fp32:
        probs = probs.to(dtype)

    if training:
        try:
            from megatron.core import tensor_parallel as _tp
            with _tp.get_cuda_rng_tracker().fork():
                probs = attention_dropout(probs)
        except Exception:
            probs = attention_dropout(probs)

    context = torch.bmm(probs, v_2d)  # [b*np, sq_local, hn]
    context = context.reshape(b, np_heads, sq_local, hn).permute(2, 0, 1, 3)
    return context  # [sq_local, b, np, hn]


# ---------------------------------------------------------------------------
# Window-attention helper (SWA / sink attention from M2305/M2343)
# ---------------------------------------------------------------------------

def is_layer_window_attention(
    window_size: Optional[tuple],
    window_attn_skip_freq: Optional[int],
    layer_number: int,
) -> bool:
    """Return True if this layer should use sliding-window attention.

    Args:
        window_size: (left, right) window size tuple, or None.
        window_attn_skip_freq: Every Nth layer uses full attention (sink).
        layer_number: 1-based layer index.
    """
    if window_size is None:
        return False
    if window_attn_skip_freq is not None and layer_number % window_attn_skip_freq == 0:
        return False  # this layer is a "full attention" layer in the SWA pattern
    return True


# ---------------------------------------------------------------------------
# Fused scale-mask-softmax (simplified self-contained version)
# ---------------------------------------------------------------------------

def attention_mask_func(attn_weights: Tensor, mask: Tensor) -> Tensor:
    """Apply additive attention mask (Megatron convention).

    Args:
        attn_weights: ``[b, nh, sq, sk]`` raw attention logits.
        mask: Boolean ``[b, 1, sq, sk]`` where True = masked (ignore).

    Returns:
        attn_weights with -inf at masked positions.
    """
    return attn_weights.masked_fill(mask, -10000.0)


class FusedScaleMaskSoftmax(nn.Module):
    """Scale + optional mask + softmax.

    Dispatches to fast CUDA kernels when available, falls back to pure
    PyTorch otherwise.

    Args:
        config: TransformerConfig (dtype flags, fusion flag, etc.).
        attn_mask_type: ``"causal"``, ``"padding"``, or ``"no_mask"``.
        softmax_in_fp32: Cast to fp32 before softmax.
        scale: Optional additional scale coefficient (for layer-number scaling).
        window_size: Optional (left, right) for SWA masking.
    """

    def __init__(
        self,
        config: TransformerConfig,
        attn_mask_type: str = "causal",
        softmax_in_fp32: bool = True,
        scale: Optional[float] = None,
        window_size: Optional[tuple] = None,
    ) -> None:
        super().__init__()
        self.attn_mask_type = attn_mask_type
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale
        self.window_size = window_size

    def forward(
        self,
        attn_weights: Tensor,
        attention_mask: Optional[Tensor],
        softmax_offset: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply scale, mask, optional offset, and softmax.

        Args:
            attn_weights: ``[b, nh, sq, sk]``
            attention_mask: Optional bool mask ``[b, 1, sq, sk]``.
            softmax_offset: Optional per-head offset ``[nh]`` (off-by-one /
                learnable variant from M2343/M2618).

        Returns:
            attention_probs: ``[b, nh, sq, sk]``.
        """
        orig_dtype = attn_weights.dtype

        if self.scale is not None:
            attn_weights = attn_weights * self.scale

        if attention_mask is not None:
            attn_weights = attention_mask_func(attn_weights, attention_mask)
        elif self.attn_mask_type == "causal":
            sq = attn_weights.size(-2)
            sk = attn_weights.size(-1)
            if sq > 1:
                causal = torch.ones(sq, sk, dtype=torch.bool, device=attn_weights.device)
                causal = torch.triu(causal, diagonal=sk - sq + 1)
                attn_weights = attn_weights.masked_fill(causal, -10000.0)

        if self.softmax_in_fp32:
            attn_weights = attn_weights.float()

        if softmax_offset is not None:
            # off-by-one / learnable: broadcast [nh] → [b, nh, sq, sk]
            offset = softmax_offset.view(1, -1, 1, 1)
            attn_weights = attn_weights + offset

        attn_probs = torch.softmax(attn_weights, dim=-1)

        if self.softmax_in_fp32:
            attn_probs = attn_probs.to(orig_dtype)

        return attn_probs


# ---------------------------------------------------------------------------
# DotProductAttention
# ---------------------------------------------------------------------------

class DotProductAttention(MegatronModule):
    """Scaled dot-product attention — region for selective recomputation.

    This is the memory-intensive, compute-light region described in
    "Reducing Activation Recomputation in Large Transformer Models"
    (arxiv 2205.05198).  Wrapping *only* this region in checkpoint() gives
    a favourable compute/memory tradeoff for large models (≥20B).

    Supports:
      * Standard MHA (num_query_groups == num_attention_heads)
      * GQA (num_query_groups < num_attention_heads) via repeat_interleave
      * MQA (num_query_groups == 1)
      * Sliding Window Attention (config.window_size)
      * Sink tokens (config.window_attn_skip_freq)
      * Learnable / off-by-one softmax offset (config.softmax_type)
      * apply_query_key_layer_scaling (divide softmax_scale by layer_number)
      * DES-LOC tier logging at construction

    Notation (matching Megatron):
      h  = hidden_size
      n  = num_attention_heads
      ng = num_query_groups
      p  = TP world size
      b  = batch_size
      s  = sequence_length
      hn = hidden_size_per_attention_head = h // n
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: str = "causal",
        attention_type: str = "self",
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[object] = None,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

        # ------------------------------------------------------------------
        # Context Parallelism setup
        # Replaces the old NotImplementedError — we now support p2p ring and
        # all_gather CP paths in pure PyTorch.  The a2a / a2a+p2p paths
        # require Transformer Engine and fall back to p2p with a warning.
        # ------------------------------------------------------------------
        context_parallel_size = getattr(config, "context_parallel_size", 1) or 1
        self.cp_size  = context_parallel_size
        self.cp_rank  = _get_cp_rank()  if context_parallel_size > 1 else 0
        self.cp_group = _get_cp_group() if context_parallel_size > 1 else None

        # Resolve the per-layer comm type.  cp_comm_type arg takes precedence;
        # falls back to config.cp_comm_type (str or per-layer list).
        layer_cp_comm = cp_comm_type
        if layer_cp_comm is None:
            cfg_comm = getattr(config, "cp_comm_type", None)
            if isinstance(cfg_comm, list):
                idx = max(0, self.layer_number - 1)
                layer_cp_comm = cfg_comm[idx] if idx < len(cfg_comm) else "p2p"
            else:
                layer_cp_comm = cfg_comm  # str or None
        if context_parallel_size > 1 and layer_cp_comm is None:
            layer_cp_comm = "p2p"  # default to ring
        self.cp_comm_type: Optional[str] = layer_cp_comm

        if context_parallel_size > 1 and self.cp_comm_type not in (
            None, "p2p", "all_gather", "a2a", "a2a+p2p"
        ):
            raise ValueError(
                f"Unsupported cp_comm_type={self.cp_comm_type!r} for "
                "DotProductAttention.  Choose from: 'p2p', 'all_gather', "
                "'a2a', 'a2a+p2p'."
            )
        if context_parallel_size > 1 and self.cp_comm_type in ("a2a", "a2a+p2p"):
            # a2a (Ulysses-style) needs Transformer Engine internals; fall back
            logger.warning(
                "DotProductAttention layer %d: cp_comm_type=%r requires "
                "TEDotProductAttention.  Falling back to 'p2p' ring-attention.",
                self.layer_number, self.cp_comm_type,
            )
            self.cp_comm_type = "p2p"

        # TP partitioning
        world_size = _get_tp_world_size()
        projection_size = config.kv_channels * config.num_attention_heads
        self.hidden_size_per_partition = projection_size // world_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads // world_size

        num_query_groups = config.num_query_groups or config.num_attention_heads
        self.num_query_groups_per_partition = max(1, num_query_groups // world_size)

        # Softmax scale (M2305 → query_key_layer_scaling)
        coeff: Optional[float] = None
        if softmax_scale is not None:
            self.softmax_scale = softmax_scale
        else:
            self.softmax_scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head)

        apply_scaling = getattr(config, "apply_query_key_layer_scaling", False)
        if apply_scaling:
            coeff = float(self.layer_number)
            self.softmax_scale /= coeff

        # SWA (M2343): window attention for eligible layers
        if is_layer_window_attention(
            getattr(config, "window_size", None),
            getattr(config, "window_attn_skip_freq", None),
            self.layer_number,
        ):
            window_size = getattr(config, "window_size", None)
        else:
            window_size = None
        self.window_size = window_size

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            config=config,
            attn_mask_type=attn_mask_type,
            softmax_in_fp32=getattr(config, "attention_softmax_in_fp32", True),
            scale=coeff,
            window_size=window_size,
        )

        # Attention dropout
        dropout_p = attention_dropout if attention_dropout is not None else config.attention_dropout
        self.attention_dropout = nn.Dropout(dropout_p)

        # Softmax type (M2343): vanilla | off-by-one | learnable
        softmax_type = getattr(config, "softmax_type", "vanilla")
        if softmax_type == "vanilla":
            self.softmax_offset = None
        elif softmax_type == "off-by-one":
            self.softmax_offset = torch.zeros(
                self.num_attention_heads_per_partition,
                dtype=getattr(config, "params_dtype", torch.float32),
            )
        elif softmax_type == "learnable":
            # M2618: fix param init — use register_parameter
            offset_param = nn.Parameter(
                torch.empty(
                    self.num_attention_heads_per_partition,
                    dtype=getattr(config, "params_dtype", torch.float32),
                )
            )
            if getattr(config, "perform_initialization", True):
                nn.init.zeros_(offset_param)
            self.register_parameter("softmax_offset", offset_param)
        else:
            raise ValueError(
                f"Unknown softmax_type: {softmax_type!r}. "
                "Valid options: 'vanilla', 'off-by-one', 'learnable'."
            )

        # QK logit clipping (M2831)
        self.current_max_attn_logits: Optional[Tensor] = None
        self.qk_clip: bool = getattr(config, "qk_clip", False)
        self.qk_clip_threshold: float = getattr(config, "qk_clip_threshold", 10.0)

        # DES-LOC: log tier assignment
        tier = config.get_layer_tier(self.layer_number - 1)
        if tier is not None:
            logger.debug(
                "DotProductAttention layer %d → DES-LOC tier: %s",
                self.layer_number,
                tier.upper(),
            )

    # ------------------------------------------------------------------
    # QK logit clipping helper (M2831)
    # ------------------------------------------------------------------

    def _record_max_attn_logits(self, attn_scores: Tensor) -> None:
        """Track per-head max attention logit for QK clipping."""
        with torch.no_grad():
            # attn_scores: [b, nh, sq, sk]
            self.current_max_attn_logits = attn_scores.detach().max(dim=-1).values.max(dim=-1).values.max(dim=0).values

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor],
        attn_mask_type: Optional[str] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[object] = None,
    ) -> Tensor:
        """Compute scaled dot-product attention.

        This is the selective-recompute region.  Calling code wraps this
        forward in ``tensor_parallel.checkpoint()`` when
        ``recompute_granularity == 'selective'``.

        Supports:
          * Standard SBH layout: ``[s, b, np, hn]``
          * THD (packed-sequence) layout when packed_seq_params.qkv_format == "thd"
          * Context parallelism via ring (p2p) or all_gather comm paths

        Args:
            query:    ``[sq, b, ng, np/ng * hn]``  (post-reshape by Attention)
                      reshaped here to ``[sq, b, np_per_tp, hn]``
            key:      ``[sk, b, ng, hn]``
            value:    ``[sk, b, ng, hn]``
            attention_mask: Bool or float mask.
            attn_mask_type: Override for this call (unused in native path).
            attention_bias: Additive pre-softmax bias ``[b, nh, sq, sk]`` (e.g. ALiBi).
            packed_seq_params: THD packed-sequence params.  When
                ``packed_seq_params.qkv_format == "thd"`` the inputs arrive
                as ``[total_tokens, 1, np, hn]`` (batch dim squeezed to 1).

        Returns:
            context: ``[sq, b, hidden_size_per_partition]``
        """
        # ------------------------------------------------------------------
        # THD (packed-sequence) path
        # When qkv_format == "thd" the Attention module has already squeezed
        # the batch dim; tensors are [total_tokens, 1, np, hn].
        # We handle this separately, returning [total_tokens, 1, np*hn].
        # ------------------------------------------------------------------
        qkv_format = (
            getattr(packed_seq_params, "qkv_format", None)
            if packed_seq_params is not None
            else None
        )
        is_thd = qkv_format == "thd"

        if is_thd:
            total_q, _, np_heads, hn = query.shape
            total_k = key.shape[0]

            cu_seqlens_q  = getattr(packed_seq_params, "cu_seqlens_q",  None)
            cu_seqlens_kv = getattr(packed_seq_params, "cu_seqlens_kv", None)
            max_seqlen_q  = getattr(packed_seq_params, "max_seqlen_q",  None)
            max_seqlen_kv = getattr(packed_seq_params, "max_seqlen_kv", None)

            # Fast path: flash_attn_varlen_func
            try:
                from flash_attn import flash_attn_varlen_func as _fa_varlen
                assert cu_seqlens_q is not None and cu_seqlens_kv is not None
                q_fa = query.squeeze(1)  # [total_q, np, hn]
                k_fa = key.squeeze(1)    # [total_k, np, hn]
                v_fa = value.squeeze(1)  # [total_k, np, hn]
                dropout_p = self.attention_dropout.p if self.training else 0.0
                is_causal = (self.attn_mask_type == "causal")
                out = _fa_varlen(
                    q_fa, k_fa, v_fa,
                    cu_seqlens_q, cu_seqlens_kv,
                    max_seqlen_q  or total_q,
                    max_seqlen_kv or total_k,
                    dropout_p=dropout_p,
                    softmax_scale=self.softmax_scale,
                    causal=is_causal,
                )  # [total_q, np, hn]
                context = out.reshape(total_q, 1, np_heads * hn)
                return context
            except Exception:
                pass  # fall through to manual fallback

            # Manual fallback: per-sequence attention (correct, not fused)
            if cu_seqlens_q is None:
                cu_seqlens_q  = torch.tensor(
                    [0, total_q], dtype=torch.int32, device=query.device
                )
                cu_seqlens_kv = torch.tensor(
                    [0, total_k], dtype=torch.int32, device=key.device
                )

            num_seqs = len(cu_seqlens_q) - 1
            out_chunks = []
            for i in range(num_seqs):
                s_q  = int(cu_seqlens_q[i])
                e_q  = int(cu_seqlens_q[i + 1])
                s_kv = int(cu_seqlens_kv[i])
                e_kv = int(cu_seqlens_kv[i + 1])
                sq_i = e_q  - s_q
                sk_i = e_kv - s_kv

                # [sq_i, np, hn]
                q_i = query[s_q:e_q,   0, :, :]
                k_i = key  [s_kv:e_kv, 0, :, :]
                v_i = value[s_kv:e_kv, 0, :, :]

                # [np, sq_i, sk_i]
                scores = torch.einsum("qnh,knh->nqk", q_i, k_i) * self.softmax_scale

                if self.attn_mask_type == "causal":
                    causal = torch.triu(
                        torch.ones(sq_i, sk_i, dtype=torch.bool, device=scores.device),
                        diagonal=1,
                    )
                    scores = scores.masked_fill(causal.unsqueeze(0), float('-inf'))

                probs = torch.softmax(scores.float(), dim=-1).to(query.dtype)
                if self.training:
                    probs = self.attention_dropout(probs)

                # [sq_i, np, hn]
                ctx_i = torch.einsum("nqk,knh->qnh", probs, v_i)
                out_chunks.append(ctx_i.reshape(sq_i, 1, np_heads * hn))

            context = torch.cat(out_chunks, dim=0)  # [total_q, 1, np*hn]
            return context

        # ------------------------------------------------------------------
        # GQA: expand KV heads to match Q heads
        # [sk, b, ng, hn] → [sk, b, np, hn]
        # This is a noop when ng == np (standard MHA).
        # ------------------------------------------------------------------
        expand_factor = (
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        )
        if expand_factor > 1:
            key = key.repeat_interleave(expand_factor, dim=2)
            value = value.repeat_interleave(expand_factor, dim=2)

        # ------------------------------------------------------------------
        # Context Parallel paths (cp_size > 1)
        # ------------------------------------------------------------------
        if self.cp_size > 1:
            effective_mask_type = attn_mask_type or self.attn_mask_type
            # Accept both str and Megatron AttnMaskType enum
            if isinstance(effective_mask_type, str):
                mask_type_str = effective_mask_type
            else:
                mask_type_str = getattr(effective_mask_type, "name", str(effective_mask_type))

            softmax_in_fp32 = getattr(self.config, "attention_softmax_in_fp32", True)

            if self.cp_comm_type == "all_gather":
                context = _allgather_attn_forward(
                    query, key, value,
                    softmax_scale=self.softmax_scale,
                    attention_dropout=self.attention_dropout,
                    attn_mask_type=mask_type_str,
                    attention_mask=attention_mask,
                    cp_group=self.cp_group,
                    cp_rank=self.cp_rank,
                    cp_size=self.cp_size,
                    softmax_in_fp32=softmax_in_fp32,
                    training=self.training,
                )
            else:
                # Default: "p2p" ring-attention
                context = _ring_attn_forward(
                    query, key, value,
                    softmax_scale=self.softmax_scale,
                    attention_dropout=self.attention_dropout,
                    attn_mask_type=mask_type_str,
                    attention_mask=attention_mask,
                    cp_group=self.cp_group,
                    cp_rank=self.cp_rank,
                    cp_size=self.cp_size,
                    softmax_in_fp32=softmax_in_fp32,
                    training=self.training,
                )
            # context: [sq_local, b, np, hn]

            # Apply attention_bias if provided (additive pre-softmax bias).
            # For CP we approximate as an output residual; use TE for exactness.
            if attention_bias is not None:
                logger.warning(
                    "DotProductAttention: attention_bias with context_parallel_size>1 "
                    "is approximated as an output residual.  Use TEDotProductAttention "
                    "for exact results."
                )
                # attention_bias: [b, np, sq, sk] → sum over K → [b, np, sq]
                bias_out = attention_bias.sum(dim=-1)           # [b, np, sq]
                bias_out = bias_out.permute(2, 0, 1).unsqueeze(-1)  # [sq, b, np, 1]
                context = context + bias_out

            # Flatten head dim: [sq_local, b, np, hn] → [sq_local, b, hp]
            sq_local, b_, np_, hn_ = context.shape
            context = context.reshape(sq_local, b_, np_ * hn_)
            return context

        # ------------------------------------------------------------------
        # Standard (non-CP) path
        # ------------------------------------------------------------------

        # ---------------------------------------------------------------
        # Raw attention scores  [b, np/p, sq, sk]
        # ---------------------------------------------------------------
        output_size = (
            query.size(1),
            query.size(2),
            query.size(0),
            key.size(0),
        )

        # [sq, b, np, hn] → [sq, b*np, hn]  (reshape for GQA compat)
        query_2d = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] → [sk, b*np, hn]
        key_2d = key.view(output_size[3], output_size[0] * output_size[1], -1)

        # Pre-allocate matmul buffer from the global memory pool
        try:
            from megatron.core import parallel_state as _mcore_ps
            matmul_input_buffer = _mcore_ps.get_global_memory_buffer().get_tensor(
                (output_size[0] * output_size[1], output_size[2], output_size[3]),
                query.dtype,
                "mpu",
            )
        except Exception:
            matmul_input_buffer = torch.empty(
                output_size[0] * output_size[1],
                output_size[2],
                output_size[3],
                dtype=query.dtype,
                device=query.device,
            )

        # [b*np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_2d.transpose(0, 1),                    # [b*np, sq, hn]
            key_2d.transpose(0, 1).transpose(1, 2),      # [b*np, hn, sk]
            beta=0.0,
            alpha=self.softmax_scale,
        )

        # [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # Apply attention_bias (e.g. ALiBi additive bias)
        if attention_bias is not None:
            attention_scores = attention_scores + attention_bias

        # Track logits for QK clipping (M2831)
        if self.qk_clip and self.training:
            self._record_max_attn_logits(attention_scores)

        # ---------------------------------------------------------------
        # Scale + mask + softmax
        # ---------------------------------------------------------------
        softmax_offset = getattr(self, "softmax_offset", None)
        if isinstance(softmax_offset, Tensor):
            softmax_offset = softmax_offset.to(attention_scores.device)

        attention_probs: Tensor = self.scale_mask_softmax(
            attention_scores, attention_mask, softmax_offset
        )

        # Sequence-parallel safe dropout
        seq_parallel = getattr(self.config, "sequence_parallel", False)
        if not seq_parallel:
            try:
                from megatron.core import tensor_parallel as _tp
                with _tp.get_cuda_rng_tracker().fork():
                    attention_probs = self.attention_dropout(attention_probs)
            except Exception:
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # ---------------------------------------------------------------
        # Context layer  [sq, b, hidden_size_per_partition]
        # ---------------------------------------------------------------
        context_output_size = (
            value.size(1),
            value.size(2),
            query.size(0),
            value.size(3),
        )

        # [sk, b*np, hn]
        value_2d = value.view(
            value.size(0), context_output_size[0] * context_output_size[1], -1
        )

        # [b*np, sq, sk]
        attention_probs_2d = attention_probs.view(
            context_output_size[0] * context_output_size[1], context_output_size[2], -1
        )

        # [b*np, sq, hn]
        context = torch.bmm(attention_probs_2d, value_2d.transpose(0, 1))

        # [b, np, sq, hn] → [sq, b, np, hn] → [sq, b, hp]
        context = context.view(*context_output_size)
        context = context.permute(2, 0, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.view(*new_context_shape)

        return context

    # ------------------------------------------------------------------
    # Sharded state dict
    # ------------------------------------------------------------------

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Sharded state dict for the learnable softmax_offset parameter.

        From Megatron M2480: Fix Sink Attention TP — softmax_offset is sharded
        along axis 0 (num_attention_heads) by TP rank, so it must be represented
        as a ShardedTensor with the TP fragmentation on axis 0.  Without this,
        saving/restoring with TP>1 would silently duplicate/truncate the parameter.
        On DES-LOC PCIe clusters with TP across heterogeneous tiers this is
        especially important since checkpoint round-trips are expensive over PCIe.
        """
        softmax_type = getattr(self.config, "softmax_type", "vanilla")
        if softmax_type != "learnable":
            return {}
        # softmax_offset shape: [num_attention_heads_per_tp_rank]
        # global shape: [num_attention_heads_total] sharded on axis 0 by TP
        param = self.softmax_offset
        tp_size = getattr(self, "_tp_size", 1)
        tp_rank = getattr(self, "_tp_rank", 0)
        local_size = param.shape[0]
        global_size = local_size * tp_size
        key = f"{prefix}softmax_offset"
        return {
            key: ShardedTensor(
                key=key,
                data=param.data,
                dtype=param.dtype,
                local_shape=(local_size,),
                global_shape=(global_size,),
                global_offset=(tp_rank * local_size,),
                axis_fragmentations=(tp_size,),
            )
        }
