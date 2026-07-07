# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""
FlashAttentionOp — Python wrapper for the SM-dispatched FA-2 CUDA kernels.

addresses #145: FlashAttention-2 kernel integration for heterogeneous SM.

Architecture
------------
This module provides `FlashAttentionOp`, a thin wrapper that:

1. Calls `FusedAttentionBuilder().load()` to JIT-compile (or load a cached
   build of) the `ds_fused_attention` CUDA extension.
2. Auto-detects the SM version of the current device (86 / 90 / 120) and
   passes it to the kernel launcher.
3. Enforces the BF16 dtype contract expected by the CUDA kernels.
4. Handles the [sq, b, np, hn] ↔ [b, np, sq, hn] layout translation that
   DotProductAttention performs internally in `_run_fused_attention()`.

Kernel dispatch
---------------
The CUDA kernels select tile sizes at launch time via the AttnPolicy template:

  SM 8.6 (A6000):  Br=64,  Bc=64,  128 threads, __ldg() global loads
  SM 9.0 (H100):   Br=128, Bc=128, 256 threads, cp.async for K/V
  SM 12.0 (BW):    Br=128, Bc=64,  persistent kernel, cp.async.bulk

See csrc/attention/fused_attention.cu for the full occupancy analysis.

Usage
-----
    from deepspeed.ops.fused_attention import FlashAttentionOp

    op = FlashAttentionOp()

    # forward: returns (output, lse)
    out, lse = op.forward(query, key, value, softmax_scale=0.125, causal=True)

    # GQA forward (warp-group KV-sharing kernel):
    out = op.gqa_forward(query, key, value, num_kv_heads=8, causal=True)

    # backward (gradients into dq, dk, dv):
    dq, dk, dv = op.backward(d_output, query, key, value, output, lse,
                              softmax_scale=0.125, causal=True)
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def _get_sm_version(device: Optional[torch.device] = None) -> int:
    """Return SM version integer (e.g. 86, 90, 120) for *device*.

    Falls back to the current CUDA device when *device* is None.
    Returns 86 (A6000 / SM 8.6 baseline) when CUDA is not available.
    """
    if not torch.cuda.is_available():
        return 86
    if device is None:
        dev_idx = torch.cuda.current_device()
    else:
        dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(dev_idx)
    return major * 10 + minor


class FlashAttentionOp:
    """
    SM-dispatched FlashAttention-2 operator for heterogeneous GPU clusters.

    Wraps the `ds_fused_attention` CUDA extension compiled by
    `FusedAttentionBuilder`.  The extension is JIT-compiled on first use
    and cached; subsequent instantiations reuse the cached shared object.

    Supports:
      * MHA  (num_kv_heads == num_q_heads)
      * GQA  (num_kv_heads < num_q_heads, must divide evenly)
      * MQA  (num_kv_heads == 1)
      * Causal masking
      * Sliding-Window Attention (SWA) via window_left / window_right
      * Attention dropout (Philox-compatible with PyTorch's CUDARNGTracker)
      * Forward + backward pass
      * Dedicated GQA kernel with warp-group KV-sharing (gqa_forward)
    """

    _op = None           # shared extension handle across all instances
    _load_tried = False  # guard against repeated failed load attempts

    # ------------------------------------------------------------------
    # Construction / lazy load
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self._ensure_loaded()

    @classmethod
    def _ensure_loaded(cls) -> None:
        """JIT-compile and cache the CUDA extension (idempotent)."""
        if cls._op is not None or cls._load_tried:
            return
        cls._load_tried = True
        try:
            # Primary path: pre-built or JIT via op_builder
            from deepspeed.ops.op_builder import FusedAttentionBuilder
            cls._op = FusedAttentionBuilder().load()
            logger.debug("FlashAttentionOp: ds_fused_attention loaded via FusedAttentionBuilder.")
        except Exception as e:
            # Secondary path: extension installed as a standalone wheel
            try:
                import ds_fused_attention as _mod  # noqa: F401
                cls._op = _mod
                logger.debug("FlashAttentionOp: ds_fused_attention loaded as installed package.")
            except ImportError:
                logger.debug(
                    "FlashAttentionOp: ds_fused_attention unavailable (%s). "
                    "FlashAttentionOp.forward() will raise RuntimeError.",
                    e,
                )

    @classmethod
    def is_available(cls) -> bool:
        """Return True if the CUDA extension was successfully loaded."""
        cls._ensure_loaded()
        return cls._op is not None

    def _require_op(self):
        if self._op is None:
            raise RuntimeError(
                "ds_fused_attention CUDA extension is not available. "
                "Run `DS_BUILD_FUSED_ATTENTION=1 pip install -e .` to compile it, "
                "or ensure the package was installed with CUDA support."
            )

    # ------------------------------------------------------------------
    # Input validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_tensor(t: Tensor, name: str, ndim: int = 4) -> None:
        if not t.is_cuda:
            raise TypeError(f"FlashAttentionOp: {name} must be a CUDA tensor.")
        if not t.is_contiguous():
            raise ValueError(f"FlashAttentionOp: {name} must be contiguous.")
        if t.dim() != ndim:
            raise ValueError(
                f"FlashAttentionOp: {name} must be {ndim}-D, got {t.dim()}-D."
            )

    @staticmethod
    def _to_bf16(t: Tensor, name: str) -> Tensor:
        """Cast to BF16 with a warning if the source dtype is unexpected."""
        if t.dtype == torch.bfloat16:
            return t
        if t.dtype in (torch.float16, torch.float32):
            logger.debug(
                "FlashAttentionOp: casting %s from %s to bfloat16 for CUDA kernel.",
                name, t.dtype,
            )
            return t.to(torch.bfloat16)
        raise TypeError(
            f"FlashAttentionOp: {name} has unsupported dtype {t.dtype}. "
            "Use bfloat16, float16, or float32."
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        softmax_scale: Optional[float] = None,
        causal: bool = True,
        window_left: int = -1,
        window_right: int = -1,
        dropout_p: float = 0.0,
        philox_seed: int = 0,
        philox_offset: int = 0,
        sm_version: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Run the SM-dispatched FA-2 forward pass.

        Input tensors must be in **[B, H, S, D]** (BHSD) layout, contiguous,
        on a CUDA device.  BF16 is required by the kernel; other floating-point
        dtypes are silently cast.

        For GQA set ``key``/``value`` to shape ``[B, Hkv, Sk, D]`` with
        ``Hkv < Hq``; the kernel remaps ``kv_head = q_head // (Hq // Hkv)``
        internally (no extra argument needed).

        Args:
            query:         BF16 CUDA tensor [B, Hq, Sq, D]
            key:           BF16 CUDA tensor [B, Hkv, Sk, D]
            value:         BF16 CUDA tensor [B, Hkv, Sk, D]
            softmax_scale: Attention scale.  Defaults to 1/√D.
            causal:        Enable causal (decoder) masking.
            window_left:   SWA left window size (-1 = unbounded).
            window_right:  SWA right window size (-1 = unbounded).
            dropout_p:     Attention dropout probability.
            philox_seed:   Philox RNG seed (match PyTorch CUDARNGTracker).
            philox_offset: Philox RNG offset (per-batch / per-layer counter).
            sm_version:    SM version int (86/90/120); 0 = auto-detect.

        Returns:
            (output, lse):
              output — BF16 tensor [B, Hq, Sq, D]
              lse    — FP32 tensor [B, Hq, Sq] (log-sum-exp for backward)
        """
        self._require_op()

        # Validate shapes
        self._check_tensor(query, "query")
        self._check_tensor(key,   "key")
        self._check_tensor(value, "value")

        B, Hq, Sq, D = query.shape
        Hkv = key.size(1)
        if Hq % Hkv != 0:
            raise ValueError(
                f"FlashAttentionOp.forward: num_q_heads ({Hq}) must be divisible "
                f"by num_kv_heads ({Hkv}) for GQA."
            )
        if D % 2 != 0:
            raise ValueError(
                f"FlashAttentionOp.forward: head_dim must be divisible by 2, got {D}."
            )

        # Dtype — kernel requires BF16
        query = self._to_bf16(query, "query")
        key   = self._to_bf16(key,   "key")
        value = self._to_bf16(value, "value")

        # Default scale
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(D)

        # SM auto-detect
        sm = sm_version if sm_version > 0 else _get_sm_version(query.device)

        return self._op.fused_attention_forward(
            query, key, value,
            softmax_scale,
            causal,
            window_left,
            window_right,
            dropout_p,
            philox_seed,
            philox_offset,
            sm,
        )

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(
        self,
        d_output: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        output: Tensor,
        lse: Tensor,
        softmax_scale: Optional[float] = None,
        causal: bool = True,
        sm_version: int = 0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Run the SM-dispatched FA-2 backward pass.

        Recomputes softmax probabilities from saved Q, K, V, O, and LSE;
        accumulates dQ, dK, dV (initialised to zero internally).

        All tensor arguments must be in **[B, H, S, D]** (BHSD) layout.

        Args:
            d_output:      BF16 upstream gradient [B, Hq, Sq, D]
            query:         BF16 tensor            [B, Hq, Sq, D]
            key:           BF16 tensor            [B, Hkv, Sk, D]
            value:         BF16 tensor            [B, Hkv, Sk, D]
            output:        BF16 forward output    [B, Hq, Sq, D]
            lse:           FP32 log-sum-exp        [B, Hq, Sq]  (from forward)
            softmax_scale: Attention scale.  Defaults to 1/√D.
            causal:        Must match the forward pass causal flag.
            sm_version:    SM version int (86/90/120); 0 = auto-detect.

        Returns:
            (dq, dk, dv): gradient tensors, all BF16.
        """
        self._require_op()

        self._check_tensor(d_output, "d_output")
        self._check_tensor(query,    "query")
        self._check_tensor(key,      "key")
        self._check_tensor(value,    "value")
        self._check_tensor(output,   "output")
        self._check_tensor(lse, "lse", ndim=3)

        D = query.size(3)
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(D)

        d_output = self._to_bf16(d_output, "d_output")
        query    = self._to_bf16(query,    "query")
        key      = self._to_bf16(key,      "key")
        value    = self._to_bf16(value,    "value")
        output   = self._to_bf16(output,   "output")

        sm = sm_version if sm_version > 0 else _get_sm_version(query.device)

        return self._op.fused_attention_backward(
            d_output, query, key, value, output, lse,
            softmax_scale,
            causal,
            sm,
        )

    # ------------------------------------------------------------------
    # GQA warp-group forward (dedicated KV-sharing kernel, #142)
    # ------------------------------------------------------------------

    def gqa_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        num_kv_heads: int = 0,
        causal: bool = True,
        sm_scale: float = 0.0,
        sm_version: int = 0,
    ) -> Tensor:
        """
        Run the warp-group GQA forward kernel.

        This kernel dispatches one CUDA block per *KV-head group* (gqa_ratio
        Q-heads share one block).  K/V tiles are loaded **once** into shared
        memory and used by all gqa_ratio Q-head warps — providing up to
        gqa_ratio× reduction in K/V HBM traffic vs the per-Q-head dispatch
        in `forward()`.

        Optimal for large GQA ratios (Llama-3-70B: ratio=8, i.e. 8× savings).
        For ratio=1 (MHA) the overhead from warp-group coordination makes this
        kernel slightly slower than `forward()`; use `forward()` for MHA.

        Args:
            query:        BF16 CUDA tensor [B, Hq, Sq, D]
            key:          BF16 CUDA tensor [B, Hkv, Sk, D]
            value:        BF16 CUDA tensor [B, Hkv, Sk, D]
            num_kv_heads: Number of KV heads (Hkv).  Pass 0 to infer from key.
            causal:       Enable causal masking.
            sm_scale:     Softmax scale; 0.0 → auto 1/√D.
            sm_version:   SM version int (86/90/120); 0 = auto-detect.

        Returns:
            output: BF16 tensor [B, Hq, Sq, D]
        """
        self._require_op()

        self._check_tensor(query, "query")
        self._check_tensor(key,   "key")
        self._check_tensor(value, "value")

        D = query.size(3)
        if D % 8 != 0:
            raise ValueError(
                f"FlashAttentionOp.gqa_forward: head_dim must be divisible by 8 "
                f"for float4 vectorised loads, got {D}."
            )

        query = self._to_bf16(query, "query")
        key   = self._to_bf16(key,   "key")
        value = self._to_bf16(value, "value")

        sm = sm_version if sm_version > 0 else _get_sm_version(query.device)

        return self._op.fused_gqa_attention_forward(
            query, key, value,
            num_kv_heads,
            causal,
            sm_scale,
            sm,
        )

    # ------------------------------------------------------------------
    # Convenience: auto-select MHA vs GQA kernel
    # ------------------------------------------------------------------

    def auto_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        softmax_scale: Optional[float] = None,
        causal: bool = True,
        window_left: int = -1,
        window_right: int = -1,
        dropout_p: float = 0.0,
        philox_seed: int = 0,
        philox_offset: int = 0,
        sm_version: int = 0,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Auto-select between the GQA warp-group kernel and the standard FA-2
        kernel based on ``Hq / Hkv``.

        * ``gqa_ratio >= 4``  → `gqa_forward()` (warp-group KV-sharing; no LSE)
        * ``gqa_ratio < 4``   → `forward()` (standard FA-2; returns LSE)

        The GQA kernel does not save LSE.  If you need LSE for the backward
        pass (e.g. custom autograd.Function), always use `forward()` directly.

        Returns:
            (output, lse): lse is None when the GQA kernel was chosen.
        """
        Hq  = query.size(1)
        Hkv = key.size(1)
        gqa_ratio = Hq // Hkv if Hkv > 0 else 1

        if gqa_ratio >= 4 and window_left == -1 and window_right == -1 and dropout_p == 0.0:
            # GQA warp-group kernel: more efficient for high ratios, no SWA/dropout
            D = query.size(3)
            scale = softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(D))
            sm = sm_version if sm_version > 0 else _get_sm_version(query.device)
            out = self.gqa_forward(
                query, key, value,
                num_kv_heads=Hkv,
                causal=causal,
                sm_scale=scale,
                sm_version=sm,
            )
            return out, None
        else:
            out, lse = self.forward(
                query, key, value,
                softmax_scale=softmax_scale,
                causal=causal,
                window_left=window_left,
                window_right=window_right,
                dropout_p=dropout_p,
                philox_seed=philox_seed,
                philox_offset=philox_offset,
                sm_version=sm_version,
            )
            return out, lse
