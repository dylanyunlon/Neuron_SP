# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Optimizer configuration dataclass for Neuron_SP distributed optimizer.

Mirrors Megatron's OptimizerConfig with DES-LOC moment-sync extensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class OptimizerConfig:
    """Optimizer configuration.

    Mirrors Megatron's OptimizerConfig with DES-LOC moment sync extensions.
    All training hyper-parameters that control the distributed optimizer live
    here so they can be serialised / passed around independently of the
    optimizer instance itself.

    Attributes:
        lr:                       Peak learning rate (required).
        min_lr:                   Minimum learning rate for LR schedule.
        weight_decay:             L2 regularisation coefficient.
        params_dtype:             dtype used for model parameters (bf16/fp16/fp32).
        loss_scale:               Fixed loss scale. None → dynamic scaling.
        initial_loss_scale:       Starting loss scale for dynamic scaling.
        min_loss_scale:           Floor for dynamic loss scale.
        loss_scale_window:        Steps over which scale is increased if no overflow.
        hysteresis:               Number of consecutive overflow steps before scale drops.
        fp16:                     Enable FP16 training (requires grad_scaler).
        bf16:                     Enable BF16 training (no grad_scaler needed).
        clip_grad:                Max gradient norm (0 → disabled).
        use_distributed_optimizer: Use ZeRO-style distributed optimizer.
        overlap_param_gather:     Overlap all-gather with next forward pass.
        overlap_grad_reduce:      Overlap grad reduce-scatter with backward pass.
        adam_beta1:               Adam first moment decay.
        adam_beta2:               Adam second moment decay.
        adam_eps:                 Adam numerical stability constant.

    DES-LOC extensions:
        desloc_enabled:           Enable DES-LOC Ku/Kv decomposed moment sync.
        ku:                       First-moment sync period (steps).
        kv:                       Second-moment sync period (steps).
        kx:                       Parameter sync period (steps, ≡ Kx in paper).
        heterogeneous_shard_sizing: Scale shard sizes by FLOPS ratio.

    GPU tier FLOPS ratios (used by heterogeneous shard sizing):
        h100_bf16_tflops:         H100 BF16 TFLOPS (default 989).
        a6000_bf16_tflops:        A6000 BF16 TFLOPS (default 309.7).
    """

    # Learning rate
    lr: Optional[float] = None
    min_lr: Optional[float] = None

    # Regularisation
    weight_decay: float = 0.01

    # From Megatron M3042: when True, q_layernorm/k_layernorm params ARE subject to weight
    # decay even though they are 1-D (standard rule would exempt them as bias-like params).
    # Required for Qwen3-Next / models that have explicit QK LayerNorm and want WD on them
    # for training stability. When False (default), all 1-D params and .bias are WD-exempt.
    apply_wd_to_qk_layernorm: bool = False

    # From Megatron M2356: Add setting to support using either Adam or AdamW.
    # If True (default), weight decay is decoupled from the gradient update (AdamW behavior).
    # If False, original Adam update rule is used (weight decay applied via L2 gradient).
    # When using PyTorch Adam: True → torch.optim.AdamW, False → torch.optim.Adam.
    # When using TE/Apex Adam: passed as adam_w_mode kwarg.
    decoupled_weight_decay: bool = True
    """If True, decouple weight decay from the gradient update (AdamW). If False,
    use original Adam update rule. Defaults to True (AdamW behavior)."""

    # Precision
    params_dtype: torch.dtype = torch.float32
    fp16: bool = False
    bf16: bool = True

    # Loss scaling (FP16 only)
    loss_scale: Optional[float] = None
    initial_loss_scale: float = 2 ** 32
    min_loss_scale: float = 1.0
    loss_scale_window: int = 1000
    hysteresis: int = 2

    # Gradient clipping
    clip_grad: float = 1.0

    # Distributed optimizer flags
    use_distributed_optimizer: bool = True
    overlap_param_gather: bool = False
    overlap_grad_reduce: bool = False

    # Adam hyperparameters
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # -----------------------------------------------------------------------
    # DES-LOC extensions
    # -----------------------------------------------------------------------

    # Master switch — set to True to enable decomposed moment synchronisation
    desloc_enabled: bool = False

    # Decomposed sync periods (in optimizer steps)
    ku: int = 32   # First-moment (exp_avg) sync period
    kv: int = 64   # Second-moment (exp_avg_sq) sync period
    kx: int = 8    # Parameter sync period (Kx in DES-LOC paper)

    # Heterogeneous shard sizing: distribute parameters proportionally to
    # each GPU tier's BF16 FLOPS so that compute time is balanced.
    heterogeneous_shard_sizing: bool = False

    # Reference BF16 TFLOPS for the two tiers used in DES-LOC clusters:
    #   H100 SXM5 ≈ 989 BF16 TFLOPS  (dense)
    #   A6000 Ada ≈ 309.7 BF16 TFLOPS (dense, not sparse)
    h100_bf16_tflops: float = 989.0
    a6000_bf16_tflops: float = 309.7

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def flops_ratio(self) -> float:
        """Return H100:A6000 FLOPS ratio used for shard sizing.

        H100 receives ``ratio / (1 + ratio)`` of the total shard; A6000
        receives ``1 / (1 + ratio)``.
        """
        return self.h100_bf16_tflops / self.a6000_bf16_tflops

    def h100_shard_fraction(self) -> float:
        """Fraction of parameters owned by the H100 tier."""
        r = self.flops_ratio()
        return r / (1.0 + r)

    def a6000_shard_fraction(self) -> float:
        """Fraction of parameters owned by the A6000 tier."""
        return 1.0 / (1.0 + self.flops_ratio())

    def is_ku_step(self, step: int) -> bool:
        """Return True if *step* is a first-moment sync step."""
        return (step % self.ku) == 0

    def is_kv_step(self, step: int) -> bool:
        """Return True if *step* is a second-moment sync step."""
        return (step % self.kv) == 0

    def is_kx_step(self, step: int) -> bool:
        """Return True if *step* is a parameter sync step (Kx)."""
        return (step % self.kx) == 0

    # -----------------------------------------------------------------------
    # Insight I6: PCIe-aware overlap (Megatron aa-3.5)
    # -----------------------------------------------------------------------
    # When use_pcie_aware_overlap=True the distributed optimizer recalculates
    # bucket_size and the overlap-trigger threshold using PCIe bandwidth and
    # latency instead of NVLink-tuned defaults.
    #
    # Background: Megatron's default bucket_size of 40M–125M+ elements was
    # calibrated for NVLink (≥600 GB/s), where the per-collective launch
    # overhead is negligible relative to the bandwidth.  Over PCIe (typ.
    # 16 GB/s unidirectional for PCIe 4.0 ×16) the transfer time for a
    # 40M-element bf16 bucket is ≈80 ms — far larger than the backward-pass
    # segment time on typical models, meaning overlap is impossible.  Using
    # smaller buckets allows the collective to finish before the next
    # backward segment completes, restoring meaningful overlap.
    #
    # The overlap-trigger threshold is the minimum bucket size (in elements)
    # at which launching a collective makes sense, i.e. where the transfer
    # time exceeds the NCCL launch overhead (~10 µs).
    use_pcie_aware_overlap: bool = False
    pcie_bw_gbps: float = 16.0    # GB/s — PCIe 4.0 ×16 unidirectional
    pcie_latency_us: float = 10.0  # µs — typical PCIe host↔device round-trip

    def pcie_bucket_size(self, dp_world_size: int = 1) -> int:
        """Compute PCIe-aware bucket_size in elements.

        Returns the smaller of:
         - A size where transfer time ≥ 4× PCIe latency (useful-overlap floor).
         - The NVLink default (so we don't regress on NVLink nodes).

        Args:
            dp_world_size: DP world size; used to scale the floor slightly.
        """
        # Insight I6: PCIe-aware overlap (Megatron aa-3.5)
        bytes_per_elem = 2  # bf16 / fp16
        pcie_bw_bytes = self.pcie_bw_gbps * 1e9
        latency_s = self.pcie_latency_us * 1e-6
        min_bucket_bytes = 4.0 * latency_s * pcie_bw_bytes
        min_bucket_elems = int(min_bucket_bytes / bytes_per_elem)
        pcie_bucket = max(min_bucket_elems, 500_000 * dp_world_size)
        nvlink_default = max(40_000_000, 1_000_000 * dp_world_size)
        return min(pcie_bucket, nvlink_default)

    def pcie_overlap_trigger_elems(self) -> int:
        """Minimum bucket size in elements that justifies a PCIe collective.

        Below this threshold the NCCL launch overhead (~latency) dominates
        transfer time, so launching the collective asynchronously does not
        provide meaningful overlap benefit.
        """
        # Insight I6: PCIe-aware overlap (Megatron aa-3.5)
        bytes_per_elem = 2  # bf16 / fp16
        pcie_bw_bytes = self.pcie_bw_gbps * 1e9
        latency_s = self.pcie_latency_us * 1e-6
        trigger_bytes = latency_s * pcie_bw_bytes
        return max(1, int(trigger_bytes / bytes_per_elem))

