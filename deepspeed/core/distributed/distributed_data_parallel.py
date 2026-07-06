# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""DistributedDataParallel wrapper with DES-LOC Kx-gated gradient synchronization.

Evolution summary (ported from Megatron-LM DDP commit history, 24 commits):
  M2282 (76622edf3): pgs_collection — ProcessGroupCollection unifying tp/dp_cp/pp/embd.
  M2286 (ca9797e95): Revert pgs_collection.
  M2301 (8c1a3f5df): Replay pgs_collection; add pg_collection support to DDP.
  M2352 (c2c36f77c): Fix convergence bug in reuse_grad_buf_for_mxfp8_param_ag.
  M2408 (8301dfda7): Fix duplicate init for self.module in DistributedDataParallel.
  M2459 (576980459): Unify enable/external cudagraph with cuda-graph-impl.
  M2777 (299034c2f): fp8 param cuda-graph support — _post_param_sync extracted
      from start_param_sync; is_graph_capturing() guard in hooks.
  M2974 (670473184): m4 leftover changes (overlap, pp_rank bucketing).
  M2977 (f967176b8): Revert m4 changes.
  M2980 (891876215): Reapply m4 changes.
  M3087 (dbde759da): Add ability to save wgrads and dgrads.
  M3139 (287d2f47c): Fix RL optimizer offload.
  M3140 (3955c49ed): Revert RL offload fix.
  M3146 (36411ddff): Reapply RL offload fix.
  M3442 (f91c4bb37): Fix memory issue in mxfp8 model init.
  M3443 (a2381d800): overlap-param-gather for layer-wise optimizer + unit tests.
  M3616 (c586f6d56): FP32 local gradient accumulation for subset of params.
  M3737 (e1db4a03d): NVFP4 native weights for DDP.
  M3811 (55b8111ad): DDP refactoring — extract param layout into optimizer classmethod;
      full_param_layout / FullParamLayout integration; auto-compute layout warning.
  M3834 (f2dcd421b): Add missing knob reduce_scatter_with_fp32_accumulation.
  M3948 (c1e938b03): Integrate LayerWiseDistributedOptimizer with DDP buffer infra.
  M3998 (0044db1f2): Route non-Muon params through DistributedOptimizer.
  M4020 (08bad7a48): MXFP8/FP4 post-processing after forced param AG in eval.
  M4036 (88e7ab091): Drain predecessor reduce-scatter at dispatch time
      (previous_grad_reduce_bucket_group linkage).

DES-LOC extensions:
  - DistributedDataParallelConfig.allow_skip_grad_sync: enable Kx gating.
  - finish_grad_sync(force_all_reduce): forwards flag for Kx recovery.
  - start_grad_sync(skip_sync): gate on Kx step (called by finalize_model_grads).
  - broadcast_params(): called every Kx step to fix ZeRO-3 shard inconsistency.
  - no_sync(): context manager for gradient accumulation (multi-microbatch).
  - offload_grad_buffers() / restore_grad_buffers(): RL optimizer offload support.
  - Tier-aware gradient bucketing: when a TierMap is available from parallel_state,
    the auto-computed bucket_size is adjusted per GPU hardware tier (A6000 / H100 /
    Blackwell) via compute_tier_bucket_sizes(), then synchronised across the DP group
    with an all-reduce MIN so all ranks use consistent bucket boundaries.
    Tier multipliers: A6000 → 0.5× (PCIe), H100 → 1.5× (NVLink), Blackwell → 2.0×.
    Respects explicit bucket_size overrides — tier adjustment is skipped when
    ddp_config.bucket_size is set by the caller.

Provides:
  DistributedDataParallelConfig, DistributedDataParallel
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

import deepspeed.core.parallel_state as parallel_state
from deepspeed.core.model_parallel_config import ModelParallelConfig
from deepspeed.core.distributed.param_and_grad_buffer import (
    ParamAndGradBuffer,
    ParamAndGradBucketGroup,
    group_params_for_buffers,
    partition_buckets,
    compute_tier_bucket_sizes,
)

logger = logging.getLogger(__name__)

# Guard against graph capturing (M2777: fp8 cuda-graph support).
try:
    from megatron.core.transformer.cuda_graphs import is_graph_capturing
except ImportError:
    def is_graph_capturing() -> bool:
        return False


# ---------------------------------------------------------------------------
# DistributedDataParallelConfig
# ---------------------------------------------------------------------------

@dataclass
class DistributedDataParallelConfig:
    """Configuration for DistributedDataParallel wrapper.

    Mirrors Megatron's DistributedDataParallelConfig with DES-LOC extensions.

    Attributes:
        grad_reduce_in_fp32: Reduce gradients in FP32 regardless of param dtype.
        overlap_grad_reduce: Overlap grad all-reduce with backward computation.
        overlap_param_gather: Overlap param all-gather with forward computation.
        align_param_gather: Align param all-gather across pipeline stages.
        use_distributed_optimizer: Use reduce-scatter instead of all-reduce.
        num_distributed_optimizer_instances: Number of parallel DistOpt instances.
        check_for_nan_in_grad: Assert on NaN/Inf in grad norms before collective.
        check_for_large_grads: Warn on unexpectedly large grad norms.
        average_in_collective: Average (rather than sum) in the collective.
        bucket_size: Target bucket size in elements (None → auto).
        nccl_ub: Use NCCL user-buffer (symmetric) allocation.
        disable_symmetric_registration: Disable symmetric NCCL buffer registration.
        reduce_scatter_with_fp32_accumulation: Use FP32-accumulating reduce-scatter.
        reuse_grad_buf_for_mxfp8_param_ag: Reuse grad buffer for MXFP8 param AG.
        delay_wgrad_compute: Delay weight grad compute until backward_dw() call.
        param_name_patterns_for_fp32_local_accumulation: Per-param FP32 patterns.
        pad_buckets_for_high_nccl_busbw: Pad buckets for NCCL bus-bandwidth.
        use_megatron_fsdp: Use Megatron-FSDP (FSDP integration).
        allow_skip_grad_sync: DES-LOC: allow skipping grad collective on non-Kx steps.
        use_pcie_aware_overlap: Insight I6 — recalculate bucket_size and overlap
            trigger points assuming PCIe bandwidth instead of NVLink.
        pcie_bw_gbps: Effective PCIe bandwidth in GB/s used for bucket sizing
            (Insight I6; default 16 GB/s ≈ PCIe 4.0 ×16 unidirectional).
        pcie_latency_us: PCIe round-trip latency in microseconds used to set
            the overlap trigger threshold (Insight I6; default 10 µs).
    """

    grad_reduce_in_fp32: bool = False
    overlap_grad_reduce: bool = False
    overlap_param_gather: bool = False
    align_param_gather: bool = False
    use_distributed_optimizer: bool = False
    num_distributed_optimizer_instances: int = 1
    check_for_nan_in_grad: bool = False
    check_for_large_grads: bool = False
    average_in_collective: bool = False
    bucket_size: Optional[int] = None
    nccl_ub: bool = False
    disable_symmetric_registration: bool = False
    reduce_scatter_with_fp32_accumulation: bool = False
    reuse_grad_buf_for_mxfp8_param_ag: bool = False
    delay_wgrad_compute: bool = False
    param_name_patterns_for_fp32_local_accumulation: List[str] = field(default_factory=list)
    pad_buckets_for_high_nccl_busbw: bool = False
    use_megatron_fsdp: bool = False

    # From Megatron M3321: all-gather in start_param_sync for better overlap
    fsdp_all_gather_in_start_param_sync: bool = True
    """If True, use all-gather during the initial Megatron-FSDP parameter
    synchronization step to better overlap first param AG with computation."""

    # From Megatron M3574: MFSDP mixed-precision dtype customization
    megatron_fsdp_main_params_dtype: Optional[torch.dtype] = torch.float32
    """Data type for the main weight buffer in Megatron-FSDP distributed
    optimization. If None, compute weights serve as main weights."""

    # Insight I6: PCIe-aware overlap (Megatron aa-3.5)
    # When True, bucket_size and the overlap trigger threshold are computed
    # from PCIe bandwidth/latency rather than NVLink assumptions.
    # PCIe parameters to tune:
    #   pcie_bw_gbps:    effective unidirectional PCIe bandwidth (GB/s).
    #                    PCIe 4.0 ×16 ≈ 16 GB/s; set lower for shared lanes.
    #   pcie_latency_us: round-trip PCIe latency (µs); used to compute the
    #                    minimum bucket size whose transfer time dominates
    #                    latency (overlap threshold).
    use_pcie_aware_overlap: bool = False
    pcie_bw_gbps: float = 16.0    # GB/s — PCIe 4.0 ×16 unidirectional
    pcie_latency_us: float = 10.0  # µs — typical host↔device round-trip

    megatron_fsdp_main_grads_dtype: Optional[torch.dtype] = None
    """Data type for the main gradient buffer in Megatron-FSDP. If None,
    main grads match the model compute parameter dtype."""

    megatron_fsdp_grad_comm_dtype: Optional[torch.dtype] = None
    """Data type for gradient gather/scatter communication in Megatron-FSDP.
    If None, uses main_grads_dtype. Setting to BF16 can reduce comm latency.
    # PCIe-opt: set to torch.bfloat16 on DES-LOC A6000+H100+Blackwell to halve grad comm bandwidth.
    """

    # DES-LOC: allow skipping grad sync on non-Kx steps.
    allow_skip_grad_sync: bool = True

    # From Megatron M3194: when True, offload_grad_buffers() is suppressed.
    # CUDA graphs capture tensor storage addresses at graph-capture time; freeing
    # grad buffer storage (offload) invalidates those addresses and causes silent
    # corruption or crashes on graph replay. Set to True whenever RL training
    # cudagraphs are active to prevent this.
    training_cuda_graphs_enabled: bool = False

    # M4041 (Megatron 67b2f3878): When True, skip ``param.grad = None`` in the
    # backward post-hook so that live tensor addresses captured during CUDA graph
    # recording remain valid on every replay iteration.  Setting grad to None
    # during replay would break the captured compute graph because PyTorch stores
    # the grad pointer at capture time; subsequent replays would dereference a
    # freed (or re-allocated) tensor.
    #
    # Conflict with DES-LOC Kx sync: the Kx non-sync path still calls the same
    # backward hook.  On non-Kx steps the grad accumulates but is not cleared
    # until the next Kx boundary.  This is intentional — the Kx design already
    # relies on grad persistence across micro-steps, so cuda_graph_mode=True is
    # compatible with DES-LOC's non-Kx accumulation.
    #
    # Set to True automatically when cuda_graph_impl == 'full_iteration'.
    cuda_graph_mode: bool = False

    # -----------------------------------------------------------------------
    # Megatron upstream fields (ported from distributed_data_parallel_config.py)
    # These are not yet wired into DES-LOC training paths but are kept for
    # API compatibility with callers that pass Megatron-style configs.
    # -----------------------------------------------------------------------

    num_buckets: Optional[int] = None
    """Number of buckets for data-parallel communication. Mutually exclusive
    with bucket_size. When specified, bucket_size is computed at runtime by
    dividing total parameter count by num_buckets. Only effective when
    overlap_grad_reduce=True (otherwise bucket boundaries don't matter)."""

    fp8_param_gather: bool = False
    """If True, keep the compute param in fp8 (no intermediate dtype) and
    perform the param all-gather in fp8. Requires reuse_grad_buf_for_mxfp8_param_ag
    to be True when using mxfp8 recipe."""

    fp4_param_gather: bool = False
    """If True, keep the compute param in fp4 (no intermediate dtype) and
    perform the param all-gather in fp4 (NVFP4 native weight path, M3737)."""

    gradient_reduce_div_fusion: bool = True
    """If True, fuse gradient division into the reduction collective kernel
    instead of a separate element-wise divide. Reduces kernel launch overhead
    at the cost of a slightly different numerics path (Megatron upstream)."""

    use_custom_fsdp: bool = False
    """Deprecated alias for use_megatron_fsdp. Retained for backward
    compatibility; callers should migrate to use_megatron_fsdp=True."""

    data_parallel_sharding_strategy: str = 'no_shard'
    """Sharding strategy for Megatron-FSDP. Valid values:
      'no_shard'           — standard DDP, no parameter sharding.
      'optim'              — shard optimizer state only.
      'optim_grads'        — shard optimizer state + gradients.
      'optim_grads_params' — full ZeRO-3 (optimizer + grads + params)."""

    suggested_communication_unit_size: Optional[int] = None
    """Number of elements to communicate at once during FSDP operations.
    Affects FSDP all-gather prefetch behavior. Larger values increase the
    communication buffer size; smaller values disable prefetching."""

    keep_fp8_transpose_cache: bool = False
    """If True, keep the fp8 transpose cache when using Megatron FSDP.
    Trades memory for reduced recomputation of transposed fp8 weights."""

    fsdp_double_buffer: bool = False
    """If True, use persistently allocated double buffers for the temporary
    memory needed in Megatron FSDP communications. Required when nccl_ub=True
    (automatically set True in that case)."""

    fsdp_db_use_persist_buf_on_alloc_fail: bool = False
    """Whether to fall back to persistent buffer when a bucket does not fit
    the FSDP double buffer size. When True, enables NCCL UB for that bucket
    at the cost of extra memory; when False uses dynamic allocation."""

    outer_dp_sharding_strategy: str = 'no_shard'
    """Sharding strategy for outer DP group in Hybrid Sharded Data Parallel
    (HSDP) mode. Valid values: 'no_shard', 'optim'. Only effective when
    Hybrid FSDP is enabled."""

    fsdp_manual_registration: bool = False
    """If True, manually register FSDP communication buffers to NCCL user
    buffer. Only effective when use_megatron_fsdp=True and nccl_ub=True.
    Minimises registration call count for large models."""

    megatron_fsdp_use_decoupled_grad: bool = False
    """If True, Megatron-FSDP uses the precision-aware optimizer gradient path
    (decoupled_grad on optimizer params) instead of casting main gradients to
    param dtype for .grad."""

    megatron_fsdp_cuda_graph_mode: bool = False
    """If True, Megatron-FSDP practices CUDA-graph-safe operations, such as
    not dereferencing param.grad after the optimizer step to preserve
    references for CUDA-graph replay."""

    megatron_fsdp_enable_fine_grained_param_gather: bool = False
    """If True, enables fine-grained parameter gathering for Megatron-FSDP,
    increasing overlap between param all-gather and forward computation.
    Particularly useful for MXFP8 activation recomputation memory savings."""

    def __post_init__(self) -> None:
        """Validate configuration after dataclass field initialisation.

        Mirrors Megatron's DistributedDataParallelConfig.__post_init__ with
        additional DES-LOC consistency checks.

        Raises:
            AssertionError: if mutually exclusive options are both set.
            ValueError: if field values are out of range.
        """
        # num_buckets and bucket_size are mutually exclusive.
        if self.num_buckets is not None:
            assert self.bucket_size is None, (
                "Cannot specify both num_buckets and bucket_size in "
                "DistributedDataParallelConfig"
            )
            assert self.num_buckets > 0, (
                f"num_buckets must be > 0, got {self.num_buckets}"
            )

        # fp8_param_gather is required when reusing grad buffer for MXFP8 AG.
        if self.reuse_grad_buf_for_mxfp8_param_ag:
            assert self.fp8_param_gather, (
                "reuse_grad_buf_for_mxfp8_param_ag requires fp8_param_gather=True"
            )

        # FP32 local accumulation patterns are redundant when grad_reduce_in_fp32 is True.
        if len(self.param_name_patterns_for_fp32_local_accumulation) > 0:
            assert not self.grad_reduce_in_fp32, (
                "param_name_patterns_for_fp32_local_accumulation should only be specified "
                "when grad_reduce_in_fp32 is False — otherwise all grads are already FP32"
            )

        # Deprecation shim: use_custom_fsdp → use_megatron_fsdp.
        if self.use_custom_fsdp and not self.use_megatron_fsdp:
            logger.warning(
                "DistributedDataParallelConfig: use_custom_fsdp is deprecated; "
                "setting use_megatron_fsdp=True automatically."
            )
            object.__setattr__(self, 'use_megatron_fsdp', True)

        # DES-LOC: allow_skip_grad_sync=False with overlap_grad_reduce=True is valid
        # (forces every step to sync, disabling Kx gating entirely).
        # allow_skip_grad_sync=True without overlap_grad_reduce is also valid
        # (finalize_model_grads controls skip_sync on the synchronous path).


# ---------------------------------------------------------------------------
# DistributedDataParallel
# ---------------------------------------------------------------------------

class DistributedDataParallel(nn.Module):
    """DDP wrapper storing grads in contiguous buffers with DES-LOC Kx-gating.

    Unlike ``torch.nn.parallel.DistributedDataParallel``, this class:

    - Stores all gradients in a flat contiguous buffer per dtype group.
    - Supports bucketed all-reduce / reduce-scatter with optional overlap.
    - Supports ZeRO-style reduce-scatter via use_distributed_optimizer=True.
    - Supports FSDP integration via use_megatron_fsdp flag (M3948).
    - Supports layer-wise optimizer param all-gather (M3443 / M3948).
    - Provides DES-LOC Kx-gated gradient synchronization.

    Evolution:
      - M2282/M2301: pg_collection unification.
      - M2777: FP8 cuda-graph support; _post_param_sync.
      - M3443: overlap_param_gather for layer-wise optimizer.
      - M3811: full_param_layout pre-computation; extracted param layout.
      - M3834: reduce_scatter_with_fp32_accumulation knob.
      - M3948: LayerWiseDistributedOptimizer integration.
      - M4036: previous_grad_reduce_bucket_group drain linkage.
      - M4041: DDPConfig.cuda_graph_mode — conditional param.grad deref
               in backward hook for full-iteration CUDA graph compatibility.

    Args:
        config: Model parallel configuration.
        ddp_config: DDP-specific configuration.
        module: The model to wrap.
        disable_bucketing: Put all params in one bucket (no overlap).
        pg_collection: Optional unified process-group collection.
        full_param_layout: Pre-computed FullParamLayout for all dtype groups.
    """

    def __init__(
        self,
        config: ModelParallelConfig,
        ddp_config: DistributedDataParallelConfig,
        module: nn.Module,
        disable_bucketing: bool = False,
        pg_collection=None,
        full_param_layout=None,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Resolve process groups (M2301 pgs_collection / M3811 refactoring).
        # ------------------------------------------------------------------
        process_group_dict = _setup_process_groups_for_ddp(
            pg_collection, config, ddp_config
        )

        dp_group = process_group_dict['dp_group']
        dp_cp_group = process_group_dict['dp_cp_group']
        intra_dp_cp_group = process_group_dict['intra_dp_cp_group']
        expt_dp_group = process_group_dict['expt_dp_group']
        intra_expt_dp_group = process_group_dict['intra_expt_dp_group']
        tp_group = process_group_dict['tp_group']
        pp_group = process_group_dict['pp_group']
        ep_group = process_group_dict.get('ep_group')

        self.dp_group = dp_group
        self.dp_cp_group = dp_cp_group
        self.intra_dp_cp_group = intra_dp_cp_group
        self.expt_dp_group = expt_dp_group
        self.intra_expt_dp_group = intra_expt_dp_group
        self.tp_group = tp_group
        self.pp_group = pp_group
        self.ep_group = ep_group

        if ddp_config.num_distributed_optimizer_instances > 1:
            self.inter_dist_opt_group = process_group_dict['inter_dist_opt_group']

        # ------------------------------------------------------------------
        # Bucket size (M2974: scale with dp_size).
        # Insight I6: PCIe-aware overlap (Megatron aa-3.5)
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # Tier-aware bucket sizing (DES-LOC heterogeneous GPU extension).
        # Applied BEFORE the PCIe-aware / NVLink-default selection so that
        # the per-tier multiplier is computed on top of the chosen baseline.
        # After computing the baseline below, compute_tier_bucket_sizes() will
        # be called to apply the multiplier and sync across the DP group.
        # (See deepspeed/core/distributed/param_and_grad_buffer.py for the
        # full rationale and multiplier table.)
        # ------------------------------------------------------------------
        _apply_tier_bucketing = (
            ddp_config.bucket_size is None          # only auto-tune; respect explicit override
            and getattr(ddp_config, 'num_buckets', None) is None  # num_buckets → fixed count
            and parallel_state.is_initialized()
        )

        if ddp_config.bucket_size is None:
            if getattr(ddp_config, 'use_pcie_aware_overlap', False):
                # PCIe-aware bucket sizing.
                #
                # Rationale: The original formula (40M + 1M*dp_size) was tuned
                # for NVLink (≥600 GB/s bidirectional), where large buckets
                # amortise NCCL launch overhead.  Over PCIe (typ. 16 GB/s
                # unidirectional) smaller buckets are better because:
                #
                #  1. Smaller bucket → shorter transfer time → collective
                #     completes sooner → more backward compute can be overlapped.
                #  2. PCIe latency (≈10 µs) is much larger relative to transfer
                #     time than NVLink latency, so the minimum useful bucket is
                #     larger than NVLink latency-amortised minimum but still much
                #     smaller than the NVLink-tuned 40 M default.
                #
                # Formula: choose the bucket large enough that transfer time ≥
                # 4× PCIe round-trip latency (so overlap is worthwhile), but at
                # most the NVLink default so we don't over-buffer.
                #
                #   min_bytes = 4 × latency_s × bw_bytes_per_s
                #   bucket_elements = min_bytes / bytes_per_element (fp16/bf16 → 2 B)
                #
                bytes_per_elem = 2  # assume bf16/fp16 grad dtype as worst case
                pcie_bw_bytes = ddp_config.pcie_bw_gbps * 1e9
                latency_s = ddp_config.pcie_latency_us * 1e-6
                # Minimum bucket to make overlap worthwhile over PCIe.
                min_bucket_bytes = 4.0 * latency_s * pcie_bw_bytes
                min_bucket_elems = int(min_bucket_bytes / bytes_per_elem)
                # Scale slightly with dp_size (more ranks → larger reduce payload).
                pcie_bucket = max(min_bucket_elems, 500_000 * dp_group.size())
                # Cap at NVLink default so we don't regress on NVLink nodes.
                nvlink_default = max(40_000_000, 1_000_000 * dp_group.size())
                ddp_config.bucket_size = min(pcie_bucket, nvlink_default)
                logger.info(
                    # Insight I6: PCIe-aware overlap (Megatron aa-3.5)
                    "I6 PCIe-aware bucket_size=%d elements "
                    "(bw=%.1f GB/s, latency=%.1f µs, dp=%d)",
                    ddp_config.bucket_size,
                    ddp_config.pcie_bw_gbps,
                    ddp_config.pcie_latency_us,
                    dp_group.size(),
                )
            else:
                # Original NVLink-tuned default (Megatron M2974).
                ddp_config.bucket_size = max(40_000_000, 1_000_000 * dp_group.size())

        # ------------------------------------------------------------------
        # Apply tier-aware bucket sizing (DES-LOC heterogeneous GPU clusters).
        # compute_tier_bucket_sizes() queries the TierMap for the local GPU's
        # hardware tier (A6000 / H100 / Blackwell) and applies a per-tier
        # multiplier, then synchronises the result across the DP group via an
        # all-reduce MIN so all ranks use consistent bucket boundaries.
        # Only applied when bucket_size was auto-computed (not overridden).
        # ------------------------------------------------------------------
        if _apply_tier_bucketing and ddp_config.bucket_size is not None:
            ddp_config.bucket_size = compute_tier_bucket_sizes(
                ddp_config.bucket_size, dp_group=dp_group
            )

        if not ddp_config.overlap_grad_reduce:
            ddp_config.bucket_size = None

        self.ddp_config = ddp_config
        self.config = config
        self._module = module

        logger.info(
            "Setting up DistributedDataParallel with config %s", self.ddp_config
        )

        # ------------------------------------------------------------------
        # Disable bucketing on non-first PP stages (M2974 pp_rank logic).
        # ------------------------------------------------------------------
        self.bucket_size = ddp_config.bucket_size
        self.force_all_reduce = False
        pp_rank = pp_group[0].rank() if isinstance(pp_group, list) else pp_group.rank()
        if disable_bucketing or pp_rank > 0:
            self.bucket_size = None

        self.param_to_bucket_group: Dict[torch.nn.Parameter, ParamAndGradBucketGroup] = {}

        # ------------------------------------------------------------------
        # Collect trainable parameters.
        # ------------------------------------------------------------------
        param_to_name: Dict[torch.nn.Parameter, str] = {}
        self.params_with_grad: List[torch.nn.Parameter] = []
        all_params: List[torch.nn.Parameter] = []

        for name, param in self._module.named_parameters():
            if not param.requires_grad:
                continue
            self.params_with_grad.append(param)
            param.grad_added_to_main_grad = False
            param_to_name[param] = name
            all_params.append(param)

        # ------------------------------------------------------------------
        # num_buckets → bucket_size resolution (Megatron upstream).
        # When num_buckets is specified instead of bucket_size, derive the
        # target bucket_size by dividing the total trainable parameter count
        # by the requested number of buckets.  Only effective when
        # overlap_grad_reduce=True; bucket boundaries are irrelevant otherwise.
        # This must happen after all_params is populated and before the
        # tier-aware bucket sizing block that reads ddp_config.bucket_size.
        # ------------------------------------------------------------------
        if getattr(ddp_config, 'num_buckets', None) is not None:
            assert ddp_config.bucket_size is None, (
                "num_buckets and bucket_size are mutually exclusive"
            )
            total_params = sum(p.data.nelement() for p in all_params)
            ddp_config.bucket_size = max(1, total_params // ddp_config.num_buckets)
            logger.info(
                "num_buckets=%d → computed bucket_size=%d "
                "(total_params=%d trainable elements)",
                ddp_config.num_buckets,
                ddp_config.bucket_size,
                total_params,
            )

        # ------------------------------------------------------------------
        # Group parameters by (param_dtype, grad_dtype, is_expert, is_layerwise)
        # (M3811 group_params_for_buffers extraction).
        # ------------------------------------------------------------------
        buffer_groups = group_params_for_buffers(all_params, ddp_config.grad_reduce_in_fp32)

        # ------------------------------------------------------------------
        # Auto-compute full_param_layout when using distributed optimizer
        # (M3811 backward-compat path — callers should pre-compute layouts).
        # ------------------------------------------------------------------
        if full_param_layout is None and ddp_config.use_distributed_optimizer:
            logger.warning(
                "DistributedDataParallel: full_param_layout not provided with "
                "use_distributed_optimizer=True. Auto-computing layout inside DDP. "
                "Callers should pre-compute layouts via "
                "DistributedOptimizer.compute_full_param_layout() and pass them in."
            )
            try:
                from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
                full_param_layout = DistributedOptimizer.compute_full_param_layout(
                    all_params,
                    self.bucket_size,
                    intra_dp_cp_group.size(),
                    ddp_config,
                    expert_data_parallel_world_size=intra_expt_dp_group.size(),
                )
            except ImportError:
                pass

        # Verify layout consistency when provided (M3811).
        if full_param_layout is not None:
            assert set(buffer_groups.keys()) == set(full_param_layout.layouts.keys()), (
                f"Buffer keys {set(buffer_groups.keys())} != "
                f"full_param_layout keys {set(full_param_layout.layouts.keys())}"
            )
            for buffer_key, (params, param_indices) in buffer_groups.items():
                layout = full_param_layout.layouts[buffer_key]
                assert set(params) == set(layout.param_index_map.keys()), \
                    f"Params for {buffer_key} do not match between grouping and layout"
                assert param_indices == layout.param_indices, \
                    f"param_indices for {buffer_key} do not match"

        self.full_param_layout = full_param_layout

        # ------------------------------------------------------------------
        # Gradient scaling factors.
        # ------------------------------------------------------------------
        if getattr(config, 'calculate_per_token_loss', False):
            assert not ddp_config.average_in_collective, \
                "Cannot average in collective when calculating per-token loss"
            gradient_scaling_factor = 1.0
            expert_gradient_scaling_factor = 1.0
        else:
            if ddp_config.average_in_collective:
                gradient_scaling_factor = 1.0
                expert_gradient_scaling_factor = (
                    expt_dp_group.size() / dp_cp_group.size()
                )
            else:
                data_parallel_world_size = dp_cp_group.size()
                gradient_scaling_factor = 1.0 / data_parallel_world_size
                expert_gradient_scaling_factor = 1.0 / data_parallel_world_size

        # ------------------------------------------------------------------
        # Allocate buffers for each dtype group.
        # ------------------------------------------------------------------
        self.buffers: List[ParamAndGradBuffer] = []
        self.expert_parallel_buffers: List[ParamAndGradBuffer] = []

        _pg_collection_for_buf = _make_pg_collection(tp_group, dp_cp_group)

        for buffer_key, (params, param_indices) in buffer_groups.items():
            is_expert = buffer_key[2] if isinstance(buffer_key, tuple) else \
                getattr(buffer_key, 'is_expert_parallel', False)
            if is_expert:
                dp_data_group = intra_expt_dp_group
                scaling_factor = expert_gradient_scaling_factor
            else:
                dp_data_group = intra_dp_cp_group
                scaling_factor = gradient_scaling_factor

            # Validate scaling factor (M3811 assertion).
            if not getattr(config, 'calculate_per_token_loss', False):
                target = 1.0 / dp_cp_group.size()
                if ddp_config.average_in_collective:
                    if ddp_config.num_distributed_optimizer_instances == 1:
                        assert scaling_factor / dp_data_group.size() == target, \
                            f"Scaling factor mismatch: {scaling_factor}/{dp_data_group.size()} != {target}"
                    else:
                        assert scaling_factor == 1 or scaling_factor == (
                            expt_dp_group.size() / dp_cp_group.size()
                        ), f"Unexpected scaling factor {scaling_factor}"
                else:
                    assert scaling_factor == target, \
                        f"Scaling factor mismatch: {scaling_factor} != {target}"

            param_layout = (
                full_param_layout.layouts.get(buffer_key)
                if full_param_layout is not None
                else None
            )
            params_with_names = [(p, param_to_name[p]) for p in params]

            # Resolve param_dtype / grad_dtype from buffer_key.
            if isinstance(buffer_key, tuple):
                param_dtype, grad_dtype = buffer_key[0], buffer_key[1]
            else:
                param_dtype = buffer_key.param_dtype
                grad_dtype = buffer_key.grad_dtype

            buf = ParamAndGradBuffer(
                ddp_config,
                param_dtype,
                grad_dtype,
                params_with_names,
                dp_data_group,
                self.bucket_size,
                param_to_name,
                scaling_factor,
                param_indices,
                ddp_config.nccl_ub,
                _pg_collection_for_buf,
                param_layout=param_layout,
            )
            if is_expert:
                self.expert_parallel_buffers.append(buf)
            else:
                self.buffers.append(buf)

        # ------------------------------------------------------------------
        # Partition buckets into communication groups (M2777 / M3616).
        # ------------------------------------------------------------------
        self.bucket_groups: List[ParamAndGradBucketGroup] = partition_buckets(
            self.buffers,
            force_single_bucket_group=disable_bucketing,
            reduce_scatter_with_fp32_accumulation=ddp_config.reduce_scatter_with_fp32_accumulation,
        )
        self.expert_parallel_bucket_groups: List[ParamAndGradBucketGroup] = partition_buckets(
            self.expert_parallel_buffers,
            force_single_bucket_group=disable_bucketing,
            reduce_scatter_with_fp32_accumulation=ddp_config.reduce_scatter_with_fp32_accumulation,
        )

        # ------------------------------------------------------------------
        # Multi-DistOpt: assign inter-instance group + communication stream (M3561).
        # ------------------------------------------------------------------
        if ddp_config.num_distributed_optimizer_instances > 1:
            assert ddp_config.use_distributed_optimizer, \
                "Partial DistOpt requires use_distributed_optimizer=True"
            for bgs in [self.bucket_groups, self.expert_parallel_bucket_groups]:
                comm_stream = torch.cuda.Stream(device=torch.cuda.current_device())
                for bg in bgs:
                    bg.inter_distributed_optimizer_instance_group = self.inter_dist_opt_group
                    bg.communication_stream = comm_stream

        # ------------------------------------------------------------------
        # Chain bucket groups for async param-gather overlap (M3443).
        # next_param_gather_bucket_group is set in reverse order because
        # all-gathers happen in reverse bucket order during forward.
        # ------------------------------------------------------------------
        if ddp_config.overlap_param_gather:
            for bgs in [self.bucket_groups, self.expert_parallel_bucket_groups]:
                n = len(bgs)
                for i in range(1, n):
                    bgs[n - i].next_param_gather_bucket_group = bgs[n - i - 1]

        # ------------------------------------------------------------------
        # Chain bucket groups for predecessor reduce-scatter drain (M4036).
        # Only needed with reduce_scatter_with_fp32_accumulation + single DistOpt.
        # Forward order: bgs[i]'s predecessor is bgs[i-1].
        # ------------------------------------------------------------------
        if (
            ddp_config.overlap_grad_reduce
            and ddp_config.reduce_scatter_with_fp32_accumulation
            and ddp_config.num_distributed_optimizer_instances == 1
        ):
            for bgs in [self.bucket_groups, self.expert_parallel_bucket_groups]:
                for i in range(1, len(bgs)):
                    bgs[i].previous_grad_reduce_bucket_group = bgs[i - 1]

        # ------------------------------------------------------------------
        # Build param → bucket_group map (used in backward hook).
        # ------------------------------------------------------------------
        for bgs in [self.bucket_groups, self.expert_parallel_bucket_groups]:
            for bg in bgs:
                for bucket in bg.buckets:
                    for param in bucket.params_list:
                        self.param_to_bucket_group[param] = bg

        # ------------------------------------------------------------------
        # Unmap weight_tensor (TE fp8 workaround — M3442 fix).
        # ------------------------------------------------------------------
        if ddp_config.use_distributed_optimizer:
            @torch.no_grad()
            def _unmap(m: nn.Module) -> None:
                if hasattr(m, 'weight_tensor'):
                    m.weight_tensor = None
            self._module.apply(_unmap)

        # ------------------------------------------------------------------
        # Register backward hooks (M2777 is_graph_capturing guard).
        # ------------------------------------------------------------------
        self.grad_accs = []
        for param in self._module.parameters():
            if not param.requires_grad:
                continue
            if ddp_config.delay_wgrad_compute and getattr(param, 'skip_backward_post_hook', False):
                # Delay-wgrad path (M3616): register on parent module instead.
                for mod in self._module.modules():
                    if hasattr(mod, 'register_wgrad_accumulation_and_reduce_hooks'):
                        for pv in mod.parameters():
                            if param is pv:
                                mod.register_wgrad_accumulation_and_reduce_hooks(
                                    self._make_backward_post_hook(param)
                                )
                                break
            else:
                param_tmp = param.expand_as(param)
                grad_acc = param_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_backward_post_hook(param))
                self.grad_accs.append(grad_acc)

        # ------------------------------------------------------------------
        # Forward pre-hooks for overlap_param_gather (M3443).
        # ------------------------------------------------------------------
        self.use_forward_hook = ddp_config.overlap_param_gather
        self.remove_forward_pre_hook_handles: Dict[nn.Module, object] = {}
        if self.use_forward_hook:
            self.enable_forward_pre_hook()
        self.overlap_param_gather_with_optimizer_step: bool = False

    # ------------------------------------------------------------------
    # Properties / forward
    # ------------------------------------------------------------------

    @property
    def module(self) -> nn.Module:
        return self._module

    def forward(self, *args, **kwargs):
        return self._module(*args, **kwargs)

    # ------------------------------------------------------------------
    # Forward pre-hook management (M3443)
    # ------------------------------------------------------------------

    def enable_forward_pre_hook(self) -> None:
        """Register forward pre-hooks for overlap_param_gather (M3443)."""
        assert self.use_forward_hook
        assert len(self.remove_forward_pre_hook_handles) == 0
        for mod in self._module.modules():
            self.remove_forward_pre_hook_handles[mod] = mod.register_forward_pre_hook(
                self._make_forward_pre_hook()
            )

    def disable_forward_pre_hook(self, param_sync: bool = True) -> None:
        """Deregister forward pre-hooks (M3443).

        Args:
            param_sync: If True, force synchronous param all-gather on disable.
        """
        assert self.use_forward_hook
        for mod in list(self.remove_forward_pre_hook_handles):
            self.remove_forward_pre_hook_handles[mod].remove()
            del self.remove_forward_pre_hook_handles[mod]
        assert len(self.remove_forward_pre_hook_handles) == 0
        if param_sync:
            self.start_param_sync(force_sync=True)

    def _make_forward_pre_hook(self):
        """Create forward pre-hook to wait on all-gather handles (M3443)."""

        def hook(module: nn.Module, *unused) -> None:
            assert self.use_forward_hook, \
                "Forward pre-hook should only be active when overlap_param_gather=True"

            if is_graph_capturing():
                return

            for param in module.parameters(recurse=False):
                if param not in self.param_to_bucket_group:
                    continue
                assert param.requires_grad
                skip_next = (
                    self.ddp_config.align_param_gather
                    or self.overlap_param_gather_with_optimizer_step
                )
                self.param_to_bucket_group[param].finish_param_sync(
                    skip_next_bucket_dispatch=skip_next
                )

        return hook

    # ------------------------------------------------------------------
    # Backward hook (M2777 is_graph_capturing guard)
    # ------------------------------------------------------------------

    def _make_backward_post_hook(self, param: torch.nn.Parameter):
        """Create backward post-hook to accumulate grad and trigger async reduce."""

        def hook(*unused) -> None:
            if is_graph_capturing():
                return

            if param in self.param_to_bucket_group:
                assert param.requires_grad
                if self.ddp_config.overlap_grad_reduce:
                    assert param.grad is not None, \
                        "param.grad is None but overlap_grad_reduce=True"
                if param.grad is not None and (
                    not param.grad_added_to_main_grad
                    or getattr(param, 'zero_out_wgrad', False)
                ):
                    param.main_grad.add_(param.grad.data)
                # M4041 (Megatron 67b2f3878): Conditional param.grad deref.
                # In full-iteration CUDA graph mode we must NOT set param.grad
                # to None here: the graph was recorded with a live grad tensor
                # address, and overwriting the attribute would break replay.
                # On non-graph paths (and non-Kx DES-LOC steps) the usual
                # eager deref frees the gradient storage immediately.
                if not self.ddp_config.cuda_graph_mode:
                    param.grad = None
                if self.ddp_config.overlap_grad_reduce:
                    self.param_to_bucket_group[param].register_grad_ready(
                        param, self.force_all_reduce
                    )

        return hook

    # ------------------------------------------------------------------
    # no_sync context manager (gradient accumulation)
    # ------------------------------------------------------------------

    @contextmanager
    def no_sync(self):
        """Disable gradient synchronization for gradient accumulation.

        Within this context, backward passes accumulate gradients locally
        without triggering any cross-rank collectives. On exit, the next
        backward pass is treated as the last microbatch.
        """
        for bg in self.bucket_groups + self.expert_parallel_bucket_groups:
            bg.is_last_microbatch = False
        try:
            yield
        finally:
            for bg in self.bucket_groups + self.expert_parallel_bucket_groups:
                bg.is_last_microbatch = True

    # ------------------------------------------------------------------
    # Internal bucket-group param sync helper (M4020)
    # ------------------------------------------------------------------

    def _start_bucket_group_param_sync(
        self, bucket_group: ParamAndGradBucketGroup, force_sync: bool
    ) -> None:
        """Dispatch param all-gather for one bucket group + run post-processing.

        Factored out so LayerWiseDistributedOptimizer can sync only its own
        buckets without losing the post-all-gather work (M3948).
        """
        bucket_group.start_param_sync(force_sync=force_sync)
        if not self.ddp_config.overlap_param_gather:
            bucket_group._post_param_sync()

    # ------------------------------------------------------------------
    # Param sync
    # ------------------------------------------------------------------

    def start_param_sync(
        self,
        *unused,
        force_sync: bool = False,
        force_dispatch: bool = False,
    ) -> None:
        """Initiate param sync (all-gather) for all model parameters.

        By default when overlap_param_gather=True, dispatches async AGs.
        When overlap_param_gather=False, issues synchronous collective.

        Args:
            force_sync: Force synchronous collective regardless of other settings.
            force_dispatch: Force dispatch even if overlap_param_gather_with_optimizer_step.
        """
        if not force_sync:
            if self.overlap_param_gather_with_optimizer_step and not force_dispatch:
                return
        for bg in self.bucket_groups + self.expert_parallel_bucket_groups:
            self._start_bucket_group_param_sync(bg, force_sync=force_sync)

    # ------------------------------------------------------------------
    # Grad sync
    # ------------------------------------------------------------------

    def start_grad_sync(self, skip_sync: bool = False) -> None:
        """Initiate grad sync (all-reduce or reduce-scatter) for all params.

        DES-LOC extension: on non-Kx steps, skip_sync=True defers the
        collective while still maintaining local gradient accumulation.

        Args:
            skip_sync: DES-LOC Kx gate — skip collective on non-Kx steps.
        """
        for bg in self.bucket_groups + self.expert_parallel_bucket_groups:
            bg._skip_sync = skip_sync
            bg.start_grad_sync(force_all_reduce=False, skip_sync=skip_sync)

    def finish_grad_sync(self, force_all_reduce: Optional[bool] = False) -> None:
        """Finalize grad sync (all-reduce or reduce-scatter) for all params.

        When overlap_grad_reduce=True, waits for any outstanding async ops.
        When overlap_grad_reduce=False, dispatches and waits synchronously.

        Args:
            force_all_reduce: Force all-reduce even with use_distributed_optimizer=True.
        """
        for bg in self.bucket_groups + self.expert_parallel_bucket_groups:
            bg.finish_grad_sync(force_all_reduce=force_all_reduce)

    # ------------------------------------------------------------------
    # Param broadcast (DES-LOC Kx step / ZeRO-3 consistency fix)
    # ------------------------------------------------------------------

    def broadcast_params(self) -> None:
        """Broadcast all parameters from rank 0 across DP replicas.

        Called on every Kx step in DES-LOC to prevent the Kx spike bug
        (ZeRO-3 shard inconsistency after local accumulation).
        """
        for param in self._module.parameters():
            is_expert = not getattr(param, 'allreduce', True)
            dp_grp = self.expt_dp_group if is_expert else self.dp_cp_group
            src_rank = torch.distributed.get_global_rank(dp_grp, 0)
            torch.distributed.broadcast(param.data, src=src_rank, group=dp_grp)

    # ------------------------------------------------------------------
    # Free overlap buffers (M3904 async checkpoint OOM fix)
    # ------------------------------------------------------------------

    def free_overlap_buffers(self) -> None:
        """Free overlap param-gather GPU buffers across all bucket groups (M3904)."""
        for bg in self.bucket_groups + self.expert_parallel_bucket_groups:
            bg.free_overlap_buffers()

    # ------------------------------------------------------------------
    # Gradient scaling
    # ------------------------------------------------------------------

    def scale_gradients(self, scaling_factor: float) -> None:
        """Scale all gradients in all buffers by scaling_factor."""
        for buf in self.buffers + self.expert_parallel_buffers:
            buf.scale_gradients(scaling_factor)

    # ------------------------------------------------------------------
    # Zero grad buffer
    # ------------------------------------------------------------------

    def zero_grad_buffer(self) -> None:
        """Zero out all grad buffers. Call at the beginning of each training step."""
        cuda_graph_impl = getattr(self.config, 'cuda_graph_impl', 'none')
        if cuda_graph_impl != 'transformer_engine':
            for param in self.params_with_grad:
                param.grad_added_to_main_grad = False
        for buf in self.buffers + self.expert_parallel_buffers:
            buf.reset()
        for bg in self.bucket_groups + self.expert_parallel_bucket_groups:
            bg.reset()

    # ------------------------------------------------------------------
    # Offload / restore grad buffers (M3139 RL optimizer offload)
    # ------------------------------------------------------------------

    def offload_grad_buffers(
        self, synchronize: bool = True, empty_cache: bool = True
    ) -> None:
        """Free all grad_data tensors to release GPU memory (M3139).

        Uses storage().resize_(0) to release while keeping tensor views valid.
        All bucket.grad_data and param.main_grad views remain live (but
        accessing them during offload is undefined behavior).

        From Megatron M3194: when training_cuda_graphs_enabled=True, this
        method is a no-op with a warning. CUDA graphs capture storage addresses
        at graph-capture time; freeing and reallocating grad buffers would
        invalidate those addresses, causing silent corruption on replay.

        Args:
            synchronize: Call torch.cuda.synchronize() before freeing.
            empty_cache: Call torch.cuda.empty_cache() after freeing.
        """
        # From Megatron M3194: do not offload when training cudagraphs are active
        if getattr(self.config, 'training_cuda_graphs_enabled', False):
            logging.warning(
                "offload_grad_buffers() called but training_cuda_graphs_enabled=True — "
                "skipping offload to prevent CUDA graph address invalidation (M3194)."
            )
            return
        if synchronize:
            torch.cuda.synchronize()
        for buf in self.buffers + self.expert_parallel_buffers:
            buf.offload_to_cpu(move_params=False, move_grads=True)
        if empty_cache:
            torch.cuda.empty_cache()

    def restore_grad_buffers(self, synchronize: bool = True) -> None:
        """Reallocate grad_data tensors on GPU (M3139).

        All existing views automatically become valid again since they share
        the same storage. The grad_data is zeroed after reallocation.

        Args:
            synchronize: Call torch.cuda.synchronize() after allocation.
        """
        for buf in self.buffers + self.expert_parallel_buffers:
            buf.reload_from_cpu(move_params=False, move_grads=True)
        if synchronize:
            torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Checkpoint param-sync helper
# ---------------------------------------------------------------------------

def force_param_sync(model_chunks: list) -> None:
    """Force synchronous parameter sync (all-gather) for all DDP model chunks.

    From Megatron M2853: simplifies parameter sync for checkpoint save.
    Previously the code called disable_forward_pre_hook/enable_forward_pre_hook
    which had additional side-effects. Using start_param_sync(force_sync=True)
    directly is cleaner and correct.

    Call this before saving a checkpoint when overlap_param_gather is enabled,
    to ensure all parameters are fully gathered before serialisation.
    """
    for model_chunk in model_chunks:
        if isinstance(model_chunk, DistributedDataParallel):
            model_chunk.start_param_sync(force_sync=True)


# ---------------------------------------------------------------------------
# Process-group setup helpers
# ---------------------------------------------------------------------------

def _setup_process_groups_for_ddp(
    pg_collection,
    config: ModelParallelConfig,
    ddp_config: DistributedDataParallelConfig,
) -> Dict[str, torch.distributed.ProcessGroup]:
    """Resolve all DDP process groups from pg_collection or parallel_state.

    Mirrors Megatron's ProcessGroupCollection.setup_process_groups_for_ddp (M2301).
    Falls back to parallel_state accessor hierarchy or world group for non-distributed usage.

    Returns:
        Dict mapping group names to process groups.
    """
    _ps_init = parallel_state.is_initialized()

    def _get(fn_name: str, *args, fallback=None, **kwargs):
        if pg_collection is not None:
            # Try to extract from pg_collection attributes.
            return None  # handled below per-key
        if _ps_init:
            try:
                fn = getattr(parallel_state, fn_name, None)
                if fn is not None:
                    return fn(*args, **kwargs)
            except Exception:
                pass
        return fallback or torch.distributed.GroupMember.WORLD

    # When pg_collection is provided, extract groups directly.
    if pg_collection is not None:
        try:
            from megatron.core.process_groups_config import ProcessGroupCollection as PGC
            return PGC.setup_process_groups_for_ddp(pg_collection, config, ddp_config)
        except (ImportError, AttributeError):
            pass
        # Fallback: extract attrs from pg_collection directly.
        result = {}
        result['dp_group'] = getattr(pg_collection, 'dp', None) or \
            _fallback_dp_group(_ps_init)
        result['dp_cp_group'] = getattr(pg_collection, 'dp_cp', None) or \
            _fallback_dp_group(_ps_init)
        result['intra_dp_cp_group'] = getattr(pg_collection, 'intra_dp_cp', None) or \
            result['dp_cp_group']
        result['expt_dp_group'] = getattr(pg_collection, 'expt_dp', None) or \
            result['dp_group']
        result['intra_expt_dp_group'] = getattr(pg_collection, 'intra_expt_dp', None) or \
            result['expt_dp_group']
        result['tp_group'] = getattr(pg_collection, 'tp', None) or \
            _fallback_tp_group(_ps_init)
        result['pp_group'] = getattr(pg_collection, 'pp', None) or \
            _fallback_pp_group(_ps_init)
        result['ep_group'] = getattr(pg_collection, 'ep', None)
        if ddp_config.num_distributed_optimizer_instances > 1:
            result['inter_dist_opt_group'] = getattr(
                pg_collection, 'inter_dist_opt', None
            ) or result['dp_group']
        return result

    # No pg_collection: fall back to parallel_state.
    result = {}
    result['dp_group'] = _fallback_dp_group(_ps_init)
    result['dp_cp_group'] = _fallback_dp_cp_group(_ps_init)
    result['intra_dp_cp_group'] = _fallback_intra_dp_cp_group(_ps_init, result['dp_cp_group'])
    result['expt_dp_group'] = _fallback_expt_dp_group(_ps_init, result['dp_group'])
    result['intra_expt_dp_group'] = _fallback_intra_expt_dp_group(
        _ps_init, result['expt_dp_group']
    )
    result['tp_group'] = _fallback_tp_group(_ps_init)
    result['pp_group'] = _fallback_pp_group(_ps_init)
    result['ep_group'] = _fallback_ep_group(_ps_init)
    if ddp_config.num_distributed_optimizer_instances > 1:
        result['inter_dist_opt_group'] = result['dp_group']
    return result


def _world() -> torch.distributed.ProcessGroup:
    return torch.distributed.GroupMember.WORLD


def _try_ps(fn_name: str, *args, **kwargs):
    try:
        fn = getattr(parallel_state, fn_name, None)
        if fn is not None:
            return fn(*args, **kwargs)
    except Exception:
        pass
    return None


def _fallback_dp_group(ps_init: bool):
    if ps_init:
        v = _try_ps('get_data_parallel_group', with_context_parallel=False)
        if v is not None:
            return v
    return _world()


def _fallback_dp_cp_group(ps_init: bool):
    if ps_init:
        v = _try_ps('get_data_parallel_group', with_context_parallel=True)
        if v is not None:
            return v
    return _fallback_dp_group(ps_init)


def _fallback_intra_dp_cp_group(ps_init: bool, dp_cp_group):
    if ps_init:
        v = _try_ps('get_intra_distributed_optimizer_instance_group')
        if v is not None:
            return v
    return dp_cp_group


def _fallback_expt_dp_group(ps_init: bool, dp_group):
    if ps_init:
        v = _try_ps('get_expert_data_parallel_group', with_context_parallel=True)
        if v is not None:
            return v
        v = _try_ps('get_data_parallel_group', with_context_parallel=True)
        if v is not None:
            return v
    return dp_group


def _fallback_intra_expt_dp_group(ps_init: bool, expt_dp_group):
    if ps_init:
        v = _try_ps('get_expert_intra_distributed_optimizer_instance_group')
        if v is not None:
            return v
    return expt_dp_group


def _fallback_tp_group(ps_init: bool):
    if ps_init:
        v = _try_ps('get_tensor_model_parallel_group')
        if v is not None:
            return v
    return _world()


def _fallback_pp_group(ps_init: bool):
    if ps_init:
        v = _try_ps('get_pipeline_model_parallel_group')
        if v is not None:
            return v
    return _world()


def _fallback_ep_group(ps_init: bool):
    if ps_init:
        v = _try_ps('get_expert_model_parallel_group')
        return v
    return None


def _make_pg_collection(tp_group, dp_cp_group):
    """Create a minimal pg_collection object with tp and dp_cp attributes."""
    class _PGCollection:
        pass
    obj = _PGCollection()
    obj.tp = tp_group
    obj.dp_cp = dp_cp_group
    return obj
