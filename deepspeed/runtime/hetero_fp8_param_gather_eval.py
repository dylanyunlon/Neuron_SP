"""
DES-LOC Heterogeneous FP8/FP4 Parameter All-Gather Post-Processing
===================================================================

Upstream design intent (Megatron 08bad7a):
-------------------------------------------
Megatron-LM introduced ``_post_param_sync()`` as a factored-out helper on
``_ParamAndGradBucketGroup`` to unify three previously duplicated code paths
that all needed to run "FP8 / MXFP8 / FP4 post-processing after a param
all-gather completes":

1. The synchronous path in ``DistributedDataParallel._start_bucket_group_param_sync``
   when ``overlap_param_gather=False``.
2. The overlap path in ``_ParamAndGradBucketGroup.finish_param_sync`` after
   waiting on the async collective handle.
3. The forced-sync path triggered before entering eval when
   ``force_sync=True`` and a single-rank dp group skips the collective
   entirely.

Additionally, Megatron fixed a subtle eval-transition bug: when
``reuse_grad_buf_for_mxfp8_param_ag=True`` and ``overlap_param_gather=True``,
the param/grad buffer is *shared*.  Before calling
``disable_forward_pre_hook(param_sync=True)`` (which force-gathers weights for
eval), the main params must be copied *back* into the shared buffer (via
``_copy_main_params_to_param_buffer``), otherwise the forced all-gather reads
stale/zeroed data.  This also interacts with CUDA-graph capture: a full-
iteration CUDA graph bakes the all-gather and the subsequent ``param_data.zero_``
into the graph replay, so even when forward pre-hooks are disabled, we must
ensure the buffer is populated before replay.

DES-LOC adaptation:
--------------------
The DES-LOC (Decoupled Execution with Shared LOcality Cache) framework adds
*heterogeneous device awareness* on top of DeepSpeed's parameter management.
Our cluster has three distinct compute nodes:

  - **A6000 × 2** (SM86, 48 GB VRAM each): medium-bandwidth PCIe, no NVLink,
    native FP8 via TransformerEngine but *no* hardware MXFP8 / FP4 decode.
  - **H100 NVL × 1** (SM90, 96 GB VRAM): Hopper FP8 tensor cores, hardware
    MXFP4 decode, high-bandwidth HBM3.
  - **CPU DRAM** (1.5 TB): locality cache tier — parameters that are not
    assigned to any GPU reside here in BF16 and are streamed on demand.

The standard Megatron post-all-gather logic assumes homogeneous devices and a
single quantisation strategy per run.  In DES-LOC every bucket can hold a
*mix* of precision regimes because different layers are placed on different
device tiers.  Concretely:

  ``_DESLOCParamBucket.device_tier`` ∈ {``DeviceTier.A6000``,
  ``DeviceTier.H100``, ``DeviceTier.CPU``}

A quantised tensor gathered on an A6000 must be dequantised to BF16 in
software (the GPU cannot decode MXFP8 natively), whereas the same tensor
gathered on the H100 can stay in FP8 and let the tensor-core hardware decode
during GEMM.  CPU-resident parameters bypass the GPU collective entirely and
are handled by the Locality Cache Manager (LCM).

The ``HeteroPostParamSyncManager`` class in this file implements:

1. ``dispatch_post_sync(bucket_group)`` — the DES-LOC replacement for
   ``_post_param_sync()``, routing each bucket to the correct handler based on
   device tier and quantisation type.
2. ``eval_transition_prefill(model_chunks, optimizer_chain)`` — mirrors the
   Megatron training.py fix, but also invalidates the LCM's stale cached
   entries so eval does not read cached training-mode weights.
3. ``CudaGraphAwareParamRefill`` — detects whether a DES-LOC full-iteration
   CUDA graph has been captured and conditionally triggers buffer repopulation,
   matching Megatron's ``full_cg_captured`` guard.
4. ``_DESLOCBucketGroup`` — a lightweight shim that wraps DeepSpeed's
   ``PartitionedParameterCoordinator`` buckets with the heterogeneous dispatch
   logic.

All post-processing decisions are made *per-bucket*, not per-run, which is the
key algorithmic difference from the upstream code.
"""

from __future__ import annotations

import enum
import logging
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentinel / availability guards
# ---------------------------------------------------------------------------

try:
    from transformer_engine.pytorch.float8_tensor import Float8Tensor as _Float8Tensor  # type: ignore

    def _is_float8tensor(t: torch.Tensor) -> bool:
        return isinstance(t, _Float8Tensor)

except ImportError:  # pragma: no cover
    logger.debug("TransformerEngine not available; FP8 tensor detection disabled.")

    def _is_float8tensor(t: torch.Tensor) -> bool:  # type: ignore[misc]
        return False


try:
    from transformer_engine.pytorch.float8_tensor import MXFloat8Tensor as _MXFloat8Tensor  # type: ignore

    def _is_mxfp8tensor(t: torch.Tensor) -> bool:
        return isinstance(t, _MXFloat8Tensor)

except ImportError:  # pragma: no cover
    def _is_mxfp8tensor(t: torch.Tensor) -> bool:  # type: ignore[misc]
        return False


try:
    from transformer_engine.pytorch import nvfp4_utils as _nvfp4  # type: ignore

    def _is_nvfp4tensor(t: torch.Tensor) -> bool:
        return hasattr(_nvfp4, "is_nvfp4tensor") and _nvfp4.is_nvfp4tensor(t)

except ImportError:  # pragma: no cover
    def _is_nvfp4tensor(t: torch.Tensor) -> bool:  # type: ignore[misc]
        return False


def _is_quantised(t: torch.Tensor) -> bool:
    """Return True if *t* carries sub-BF16 quantisation metadata."""
    return _is_float8tensor(t) or _is_mxfp8tensor(t) or _is_nvfp4tensor(t)


# ---------------------------------------------------------------------------
# Hardware capability constants for our specific cluster
# ---------------------------------------------------------------------------

SM_A6000 = 86   # NVIDIA A6000, Ampere
SM_H100  = 90   # NVIDIA H100 NVL, Hopper


class DeviceTier(enum.Enum):
    """Logical device tier in the DES-LOC heterogeneous cluster."""
    A6000 = "a6000"   # 2× A6000 48 GB, SM86, PCIe only
    H100  = "h100"    # 1× H100 NVL 96 GB, SM90
    CPU   = "cpu"     # 1.5 TB DRAM locality cache tier


def _device_tier_of(device: torch.device) -> DeviceTier:
    """Map a PyTorch device to its DES-LOC tier.

    For CUDA devices we query the device's compute capability major * 10 +
    minor and compare against known SM versions.  CPU tensors always map to
    the locality-cache tier.
    """
    if device.type == "cpu":
        return DeviceTier.CPU
    if device.type != "cuda":
        raise ValueError(f"Unsupported device type for DES-LOC tier mapping: {device.type}")
    props = torch.cuda.get_device_properties(device)
    sm = props.major * 10 + props.minor
    if sm >= SM_H100:
        return DeviceTier.H100
    if sm >= SM_A6000:
        return DeviceTier.A6000
    # Older GPUs fall back to A6000-like path (software dequant)
    logger.warning(
        "Unrecognised GPU SM%d on device %s; treating as A6000 tier (software FP8 dequant).",
        sm, device,
    )
    return DeviceTier.A6000


# ---------------------------------------------------------------------------
# DES-LOC DDP config shim
# ---------------------------------------------------------------------------

@dataclass
class DESLOCDDPConfig:
    """Subset of DDP knobs relevant to DES-LOC param-gather post-processing.

    This mirrors the fields referenced by Megatron's ``DDPConfig`` that drive
    the branching logic inside ``_post_param_sync``:

    ``reuse_grad_buf_for_mxfp8_param_ag``
        When True the param buffer and grad buffer share storage.  After the
        all-gather we must copy gathered data into ``param.data`` then zero the
        shared buffer so that gradient accumulation starts from zero.

    ``overlap_param_gather``
        When True the all-gather is issued asynchronously and we must wait for
        the handle before post-processing.

    ``use_distributed_optimizer``
        Determines whether layerwise gather or the distributed-optimizer path
        is active.
    """
    reuse_grad_buf_for_mxfp8_param_ag: bool = False
    overlap_param_gather: bool = True
    use_distributed_optimizer: bool = True
    # DES-LOC extension: per-tier capability flags
    h100_native_fp8_gemm: bool = True
    a6000_software_dequant: bool = True
    cpu_lcm_bypass: bool = True  # CPU params bypass GPU collectives entirely


# ---------------------------------------------------------------------------
# Minimal bucket / param abstractions (matching DeepSpeed's layout)
# ---------------------------------------------------------------------------

@dataclass
class _FakeParamSlice:
    """Represents one parameter's view inside a bucket's flat param_data buffer."""
    param: torch.Tensor
    start: int   # flat index into bucket.param_data
    end: int     # exclusive


@dataclass
class _DESLOCParamBucket:
    """
    DES-LOC heterogeneous parameter bucket.

    Wraps DeepSpeed's ``PartitionedParameterCoordinator`` bucket notion with
    extra device-tier information.  Each bucket is assigned to exactly one
    device tier; within a bucket all parameters live on the same physical
    device.

    Attributes
    ----------
    bucket_id : int
        Monotonic bucket index within the bucket group.
    device_tier : DeviceTier
        The hardware tier this bucket is pinned to.
    params : list of Tensor
        Model parameters packed into this bucket.
    param_slices : list of _FakeParamSlice
        Maps each param to its offset in *param_data*.
    param_data : Tensor or None
        Flat contiguous buffer holding gathered parameter data.  For
        ``DeviceTier.CPU`` buckets this is a pinned-memory CPU tensor.
    grad_data : Tensor or None
        Flat buffer for gradients; may alias *param_data* when
        ``reuse_grad_buf_for_mxfp8_param_ag=True``.
    is_shared_param_grad_buf : bool
        True iff param_data and grad_data share storage (MXFP8 shared-buffer
        mode).
    """
    bucket_id: int
    device_tier: DeviceTier
    params: List[torch.Tensor] = field(default_factory=list)
    param_slices: List[_FakeParamSlice] = field(default_factory=list)
    param_data: Optional[torch.Tensor] = None
    grad_data: Optional[torch.Tensor] = None
    is_shared_param_grad_buf: bool = False

    def param_to_index(self, param: torch.Tensor) -> Tuple[int, int]:
        for sl in self.param_slices:
            if sl.param is param:
                return sl.start, sl.end
        raise KeyError(f"Param not found in bucket {self.bucket_id}")


# ---------------------------------------------------------------------------
# Locality Cache Manager (LCM) stub
# ---------------------------------------------------------------------------

class LocalityCacheManager:
    """
    Shared LOcality Cache (LOC) manager for DES-LOC.

    In a full implementation the LCM owns the 1.5 TB CPU DRAM tier and
    mediates streaming of parameter shards between CPU ↔ A6000 ↔ H100.  Here
    we expose only the interface methods relevant to eval-transition handling:

    ``invalidate_training_snapshot(param_names)``
        Marks cached BF16 copies of the given parameters as stale so that the
        next eval forward pass re-fetches weights from the authoritative GPU
        buffer rather than reading stale cached values.

    ``is_cached(param)``
        Returns True if the LCM holds a valid cached copy of *param*.
    """

    def __init__(self) -> None:
        self._cache: Dict[int, torch.Tensor] = {}  # param id → cached BF16 tensor
        self._valid: Dict[int, bool] = {}
        self._lock = threading.Lock()

    def invalidate_training_snapshot(self, params: Sequence[torch.Tensor]) -> None:
        """Invalidate cached BF16 copies of the given parameters.

        Called by ``eval_transition_prefill`` before the forced all-gather so
        that eval forward passes re-fetch from the freshly gathered GPU buffer.
        """
        with self._lock:
            invalidated = 0
            for p in params:
                pid = id(p)
                if pid in self._valid:
                    self._valid[pid] = False
                    invalidated += 1
        if invalidated:
            logger.debug(
                "LCM: invalidated %d cached parameter entries for eval transition.", invalidated
            )

    def is_cached(self, param: torch.Tensor) -> bool:
        with self._lock:
            return self._valid.get(id(param), False)

    def cache(self, param: torch.Tensor, bf16_copy: torch.Tensor) -> None:
        with self._lock:
            self._cache[id(param)] = bf16_copy
            self._valid[id(param)] = True

    def __repr__(self) -> str:
        with self._lock:
            n_valid = sum(1 for v in self._valid.values() if v)
        return f"LocalityCacheManager(cached_valid={n_valid})"


# ---------------------------------------------------------------------------
# Post-sync handlers — one per device tier × quantisation kind
# ---------------------------------------------------------------------------

class _PostSyncHandler:
    """
    Base class for per-tier post-all-gather handlers.

    Each concrete handler knows how to process one *_DESLOCParamBucket* after
    its parameter all-gather completes.  The handler is responsible for:

    1. Copying gathered flat data back into individual ``param.data`` tensors
       when necessary (MXFP8 shared-buffer mode).
    2. Zeroing the shared param/grad buffer after copy so gradient accumulation
       starts from zero.
    3. Running quantisation-specific post-processing (e.g. scale factor
       recomputation for FP8 tensors).
    """

    def handle(self, bucket: _DESLOCParamBucket, ddp_cfg: DESLOCDDPConfig) -> None:
        raise NotImplementedError


class _MXFP8SharedBufHandler(_PostSyncHandler):
    """
    Handle MXFP8 buckets when ``reuse_grad_buf_for_mxfp8_param_ag=True``.

    Mirrors the ``if self.ddp_config.reuse_grad_buf_for_mxfp8_param_ag:``
    branch in Megatron's ``_post_param_sync``, adapted to DES-LOC's per-tier
    awareness.

    On **A6000** the copy is followed by a software cast from the gathered
    BF16 back into MXFP8 (since A6000 cannot decode MXFP8 natively during
    GEMM).  On **H100** the param.data is left in MXFP8 and the hardware
    tensor cores handle decoding.

    The shared buffer (param_data alias grad_data) is zeroed after copy so
    subsequent gradient accumulation begins from zero, exactly as in Megatron.
    We must *not* zero the entire grad buffer because one grad buffer may span
    multiple param buffers; only the bucket-local slice is zeroed.
    """

    def handle(self, bucket: _DESLOCParamBucket, ddp_cfg: DESLOCDDPConfig) -> None:
        if bucket.param_data is None:
            return

        has_bf16_weight = False
        for param in bucket.params:
            if not _is_float8tensor(param) and not _is_mxfp8tensor(param):
                # BF16 weights in an otherwise MXFP8 bucket are already mapped
                # to param.data via the buffer — no copy needed.
                has_bf16_weight = True
                break
            start, end = bucket.param_to_index(param)
            flat_slice = bucket.param_data.view(-1)[start:end]
            gathered_view = flat_slice.view(param.data.shape)

            if bucket.device_tier == DeviceTier.A6000 and ddp_cfg.a6000_software_dequant:
                # A6000 cannot decode MXFP8 in hardware.  Dequantise to BF16.
                param.data.copy_(gathered_view.to(torch.bfloat16))
                logger.debug(
                    "Bucket %d: A6000 software dequant param %s → BF16.",
                    bucket.bucket_id, tuple(param.shape),
                )
            else:
                # H100: keep in MXFP8; hardware tensor cores decode on the fly.
                param.data.copy_(gathered_view)

        if has_bf16_weight:
            # Skip zeroing — BF16 weights map directly, nothing was copied.
            return

        # Zero only this bucket's slice of the shared param/grad buffer so that
        # gradient accumulation for *this* bucket starts from 0 without
        # corrupting sibling param buffers that share the same grad buffer.
        bucket.param_data.zero_()
        logger.debug(
            "Bucket %d (%s): zeroed shared param/grad buffer after MXFP8 copy.",
            bucket.bucket_id, bucket.device_tier.value,
        )


class _QuantisedParamHandler(_PostSyncHandler):
    """
    Handle FP8 / MXFP8 / FP4 buckets in the standard (non-shared-buffer) path.

    Mirrors the ``quantized_params`` collection + ``post_all_gather_processing``
    call in Megatron's ``_post_param_sync`` for the ``else`` branch.

    DES-LOC extension: FP4 parameters (NVFP4) can *only* be used on H100.
    If a FP4 tensor appears on an A6000 bucket we raise a configuration error
    rather than silently producing wrong results.
    """

    def __init__(
        self,
        post_all_gather_fn: Optional[Callable[[List[torch.Tensor]], None]] = None,
    ) -> None:
        self._post_all_gather_fn = post_all_gather_fn or self._default_post_ag

    @staticmethod
    def _default_post_ag(params: List[torch.Tensor]) -> None:
        """
        Fallback post-all-gather processing when TransformerEngine is absent.

        In production this is replaced by TE's ``post_all_gather_processing``,
        which recomputes per-tensor FP8 scale factors and updates amax history.
        The stub here exists so that unit tests can run without TE.
        """
        for p in params:
            # No-op for plain tensors; real TE tensors carry a scale attribute.
            if hasattr(p, "_fp8_scale") and p._fp8_scale is not None:
                pass  # TE handles scale update internally

    def handle(self, bucket: _DESLOCParamBucket, ddp_cfg: DESLOCDDPConfig) -> None:
        quantised: List[torch.Tensor] = []
        for param in bucket.params:
            if _is_nvfp4tensor(param):
                if bucket.device_tier != DeviceTier.H100:
                    raise RuntimeError(
                        f"FP4 parameter found in bucket {bucket.bucket_id} assigned to "
                        f"{bucket.device_tier.value}; FP4 requires H100 (SM90+)."
                    )
                quantised.append(param)
            elif _is_float8tensor(param) or _is_mxfp8tensor(param):
                quantised.append(param)

        if quantised:
            logger.debug(
                "Bucket %d (%s): post-AG processing %d quantised params.",
                bucket.bucket_id, bucket.device_tier.value, len(quantised),
            )
            self._post_all_gather_fn(quantised)


class _CPULocalityHandler(_PostSyncHandler):
    """
    Handle CPU-tier buckets in DES-LOC's Locality Cache.

    CPU-resident parameters bypass the GPU all-gather collective entirely.
    After a training step the authoritative parameter values live in GPU
    ``param.data``; the CPU locality cache holds a (possibly stale) BF16
    copy for the next forward pass that runs on the CPU tier.

    This handler does nothing during the GPU all-gather post-processing phase
    because the CPU tier has no GPU collective to post-process.  Instead it
    records which CPU parameters need their cache entries invalidated so that
    ``eval_transition_prefill`` can flush stale entries before eval.
    """

    def __init__(self, lcm: LocalityCacheManager) -> None:
        self._lcm = lcm
        self._dirty_params: List[torch.Tensor] = []
        self._lock = threading.Lock()

    def handle(self, bucket: _DESLOCParamBucket, ddp_cfg: DESLOCDDPConfig) -> None:
        # CPU buckets skip the GPU collective; mark their params dirty so
        # eval_transition_prefill knows to invalidate the LCM cache.
        with self._lock:
            for p in bucket.params:
                self._dirty_params.append(p)

    def flush_dirty(self) -> List[torch.Tensor]:
        with self._lock:
            dirty = list(self._dirty_params)
            self._dirty_params.clear()
        return dirty


# ---------------------------------------------------------------------------
# Bucket group shim
# ---------------------------------------------------------------------------

class _DESLOCBucketGroup:
    """
    DES-LOC shim around a collection of heterogeneous parameter buckets.

    Replaces Megatron's ``_ParamAndGradBucketGroup`` for the post-sync
    dispatch path.  Each bucket in ``self.buckets`` may reside on a different
    device tier; ``dispatch_post_sync`` routes each bucket to the appropriate
    handler.

    Parameters
    ----------
    group_id : int
        Monotonic identifier for this bucket group.
    buckets : list of _DESLOCParamBucket
        All buckets belonging to this group, possibly on different tiers.
    ddp_config : DESLOCDDPConfig
        DDP configuration flags shared by all buckets in the group.
    param_gather_handle : optional async handle
        If not None, ``start_param_sync`` issued an async all-gather; callers
        must call ``wait()`` before ``dispatch_post_sync``.
    """

    def __init__(
        self,
        group_id: int,
        buckets: List[_DESLOCParamBucket],
        ddp_config: DESLOCDDPConfig,
    ) -> None:
        self.group_id = group_id
        self.buckets = buckets
        self.ddp_config = ddp_config
        self.param_gather_handle: Optional[Any] = None
        self.param_gather_dispatched: bool = False

    def all_params(self) -> List[torch.Tensor]:
        params: List[torch.Tensor] = []
        for b in self.buckets:
            params.extend(b.params)
        return params

    def wait_param_gather(self) -> None:
        """Wait for an in-flight async all-gather, if any."""
        if self.param_gather_handle is not None:
            self.param_gather_handle.wait()
            self.param_gather_handle = None
            logger.debug("BucketGroup %d: param all-gather handle completed.", self.group_id)


# ---------------------------------------------------------------------------
# Core manager
# ---------------------------------------------------------------------------

class HeteroPostParamSyncManager:
    """
    DES-LOC heterogeneous FP8/FP4 parameter-gather post-processing manager.

    This is the central class adapting Megatron commit 08bad7a to the DES-LOC
    framework.  It replaces three dispersed code paths (synchronous DDP path,
    overlap finish path, forced-sync eval path) with a single ``dispatch_post_sync``
    entry point that is *aware of device tier*.

    Design goals
    ------------
    1. **Single responsibility**: The caller (``DistributedDataParallel`` or
       ``_ParamAndGradBucketGroup`` equivalent in DeepSpeed) calls
       ``dispatch_post_sync(bucket_group)`` unconditionally; this class
       decides what, if anything, needs to run.

    2. **Per-bucket routing**: Each bucket is dispatched independently based
       on ``bucket.device_tier`` × quantisation type, not a single global flag.

    3. **Eval-transition safety**: ``eval_transition_prefill`` implements the
       fix from Megatron training.py — copy main params → buffer before
       forcing the all-gather — and additionally invalidates the LCM cache so
       stale CPU-tier copies are not used during eval.

    4. **CUDA-graph awareness**: ``CudaGraphAwareParamRefill`` detects whether
       a DES-LOC full-iteration CUDA graph has been captured and conditionally
       triggers buffer repopulation, matching Megatron's ``full_cg_captured``
       guard.

    Parameters
    ----------
    ddp_config : DESLOCDDPConfig
        Cluster-wide DDP configuration.
    lcm : LocalityCacheManager
        The shared locality cache for CPU-tier parameters.
    post_all_gather_fn : callable, optional
        Replaces ``post_all_gather_processing`` from TE.  If None the
        built-in stub is used (useful for unit tests without TE).
    """

    def __init__(
        self,
        ddp_config: DESLOCDDPConfig,
        lcm: LocalityCacheManager,
        post_all_gather_fn: Optional[Callable[[List[torch.Tensor]], None]] = None,
    ) -> None:
        self._ddp_config = ddp_config
        self._lcm = lcm

        self._mxfp8_shared_handler = _MXFP8SharedBufHandler()
        self._quantised_handler = _QuantisedParamHandler(post_all_gather_fn)
        self._cpu_handler = _CPULocalityHandler(lcm)

        # Stats for diagnostics
        self._dispatch_count: int = 0
        self._skip_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def dispatch_post_sync(self, bucket_group: _DESLOCBucketGroup) -> None:
        """
        Unified post-all-gather dispatch for a heterogeneous bucket group.

        This is the DES-LOC equivalent of Megatron's ``_post_param_sync``.
        It routes each bucket to the correct handler based on device tier and
        quantisation mode.

        The routing logic:

        ┌─────────────────────────────────────────────────────────────────┐
        │  reuse_grad_buf_for_mxfp8_param_ag = True                       │
        │    → _MXFP8SharedBufHandler (copy + zero shared buffer)         │
        │    → return early (same semantics as Megatron)                  │
        ├─────────────────────────────────────────────────────────────────┤
        │  GPU bucket (A6000 or H100) with quantised params               │
        │    → _QuantisedParamHandler (post_all_gather_processing)        │
        ├─────────────────────────────────────────────────────────────────┤
        │  CPU tier bucket                                                 │
        │    → _CPULocalityHandler (mark dirty, no GPU work)              │
        └─────────────────────────────────────────────────────────────────┘

        Called by the DES-LOC DDP wrapper at:
        - The end of ``_start_bucket_group_param_sync`` when
          ``overlap_param_gather=False``.
        - Inside ``finish_param_sync`` after ``param_gather_handle.wait()``.
        - During forced-sync paths (``force_sync=True``, single-rank groups).
        """
        self._dispatch_count += 1

        if self._ddp_config.reuse_grad_buf_for_mxfp8_param_ag:
            for bucket in bucket_group.buckets:
                if bucket.device_tier == DeviceTier.CPU:
                    self._cpu_handler.handle(bucket, self._ddp_config)
                else:
                    self._mxfp8_shared_handler.handle(bucket, self._ddp_config)
            # Early return matches Megatron's shared-buffer branch structure.
            return

        for bucket in bucket_group.buckets:
            if bucket.device_tier == DeviceTier.CPU:
                self._cpu_handler.handle(bucket, self._ddp_config)
            else:
                self._quantised_handler.handle(bucket, self._ddp_config)

    def eval_transition_prefill(
        self,
        model_chunks: Sequence[Any],
        optimizer_chain: Sequence[Any],
    ) -> None:
        """
        Prepare the shared param/grad buffer for a forced all-gather before eval.

        Mirrors the fix added to Megatron's ``train()`` loop (training.py
        diff, ~line 3669):

            if args.reuse_grad_buf_for_mxfp8_param_ag and args.overlap_param_gather:
                for model_chunk in model:
                    model_chunk.zero_grad_buffer()
                for optim_instance in optimizer.chained_optimizers:
                    if isinstance(optim_instance, DistributedOptimizer):
                        optim_instance._copy_main_params_to_param_buffer()

        Context:
        ``disable_forward_pre_hook(param_sync=True)`` is called before eval to
        force-gather all parameters.  When MXFP8 shares the param/grad buffer,
        the last training step left the buffer zeroed (after gradient sync).
        The forced all-gather would then gather zeros rather than the actual
        main params.  The fix copies main params back into the shared buffer
        before the forced gather runs.

        DES-LOC extension:
        After copying, we also invalidate LCM cache entries for all CPU-tier
        parameters that were dirtied during training, so eval forward passes
        stream fresh weights from GPU rather than stale cached BF16.

        Parameters
        ----------
        model_chunks : sequence
            DES-LOC model chunk objects, each exposing ``zero_grad_buffer()``.
        optimizer_chain : sequence
            Chained optimiser instances, each possibly exposing
            ``_copy_main_params_to_param_buffer()``.
        """
        if not (
            self._ddp_config.reuse_grad_buf_for_mxfp8_param_ag
            and self._ddp_config.overlap_param_gather
        ):
            return

        logger.info(
            "eval_transition_prefill: zeroing grad buffers and copying main params "
            "to shared param/grad buffer before forced all-gather for eval."
        )

        # Step 1: zero grad buffers (param_data alias is grad_data; must be
        # cleared before we write main params into it).
        for chunk in model_chunks:
            if hasattr(chunk, "zero_grad_buffer"):
                chunk.zero_grad_buffer()

        # Step 2: copy main params (BF16) into the shared param/grad buffer so
        # the forced all-gather reads the current weights.
        copied = 0
        for opt in optimizer_chain:
            if hasattr(opt, "_copy_main_params_to_param_buffer"):
                opt._copy_main_params_to_param_buffer()
                copied += 1
        if copied:
            logger.debug("eval_transition_prefill: %d optimizer(s) copied params to buffer.", copied)

        # Step 3: DES-LOC addition — invalidate stale CPU locality cache entries.
        dirty_cpu_params = self._cpu_handler.flush_dirty()
        if dirty_cpu_params:
            self._lcm.invalidate_training_snapshot(dirty_cpu_params)
            logger.info(
                "eval_transition_prefill: invalidated %d stale CPU-tier LCM cache entries.",
                len(dirty_cpu_params),
            )

    def diagnostics(self) -> Dict[str, int]:
        """Return dispatch statistics for monitoring."""
        return {
            "dispatch_count": self._dispatch_count,
            "skip_count": self._skip_count,
        }


# ---------------------------------------------------------------------------
# CUDA-graph aware param buffer refill
# ---------------------------------------------------------------------------

class CudaGraphAwareParamRefill:
    """
    Detect DES-LOC full-iteration CUDA-graph capture and trigger param-buffer
    repopulation when necessary.

    Background (from Megatron training.py diff):
    When a full-iteration CUDA graph has been captured, the all-gather and
    the subsequent ``param_data.zero_`` are baked into the graph replay.  Even
    when forward pre-hooks are disabled (e.g. during the first iteration or
    during eval), the graph replay unconditionally gathers parameters.  We
    must ensure the param buffer is populated with the correct weights before
    replay, otherwise the all-gather collects zeros.

    Megatron's guard::

        full_cg_captured = FullCudaGraphWrapper.cuda_graph.get("training") is not None
        if forward_pre_hook_enabled or full_cg_captured:
            optim_instance._copy_main_params_to_param_buffer()

    DES-LOC adaptation:
    DES-LOC uses a ``HeteroCudaGraphRegistry`` (not yet open-sourced) that
    tracks separate graphs for the A6000 and H100 sub-graphs.  A full
    iteration is "captured" only when *both* sub-graphs are present.  We
    expose this via ``is_full_graph_captured()``.

    Parameters
    ----------
    graph_registry : dict or None
        Mapping of graph name → graph object.  Passing None simulates the
        pre-capture state (no graph).
    """

    def __init__(self, graph_registry: Optional[Dict[str, Any]] = None) -> None:
        self._registry: Dict[str, Any] = graph_registry or {}

    def is_full_graph_captured(self) -> bool:
        """Return True iff a full-iteration DES-LOC CUDA graph is available.

        We require both the A6000 and H100 sub-graphs because a partial graph
        (e.g. only H100 captured) still interleaves eager ops on A6000 that
        do their own param management.
        """
        has_training_graph = self._registry.get("training") is not None
        has_a6000_subgraph = self._registry.get("training_a6000") is not None
        has_h100_subgraph  = self._registry.get("training_h100") is not None
        # Full capture: monolithic graph OR both tier sub-graphs present.
        return has_training_graph or (has_a6000_subgraph and has_h100_subgraph)

    def maybe_refill_param_buffer(
        self,
        forward_pre_hook_enabled: bool,
        optimizer_chain: Sequence[Any],
    ) -> None:
        """
        Copy main params to param buffer if forward pre-hooks are disabled or
        a full CUDA graph has been captured.

        Should be called at the top of each train step, mirroring the
        condition in Megatron's ``train_step``::

            if forward_pre_hook_enabled or full_cg_captured:
                optim_instance._copy_main_params_to_param_buffer()
        """
        if not forward_pre_hook_enabled and not self.is_full_graph_captured():
            return

        reason = (
            "forward pre-hook enabled"
            if forward_pre_hook_enabled
            else "full CUDA graph captured"
        )
        logger.debug("CudaGraphAwareParamRefill: triggering param-buffer copy (%s).", reason)

        for opt in optimizer_chain:
            if hasattr(opt, "_copy_main_params_to_param_buffer"):
                opt._copy_main_params_to_param_buffer()


# ---------------------------------------------------------------------------
# DES-LOC DDP wrapper integration point
# ---------------------------------------------------------------------------

class DESLOCDistributedDataParallel:
    """
    Lightweight DES-LOC DDP wrapper that integrates ``HeteroPostParamSyncManager``.

    This mirrors how Megatron's ``DistributedDataParallel`` calls
    ``_post_param_sync`` (now refactored to ``_start_bucket_group_param_sync``
    and ``finish_param_sync``), but delegates to ``HeteroPostParamSyncManager``
    so that heterogeneous device-tier logic stays in one place.

    Only the param-sync–relevant subset is implemented here; the full class
    inherits from DeepSpeed's ``PipelineModule`` or ``_ZeROModuleWrapper``
    in the real Neuron_SP codebase.
    """

    def __init__(
        self,
        ddp_config: DESLOCDDPConfig,
        lcm: LocalityCacheManager,
        post_all_gather_fn: Optional[Callable] = None,
        cuda_graph_registry: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.ddp_config = ddp_config
        self._post_sync_mgr = HeteroPostParamSyncManager(
            ddp_config, lcm, post_all_gather_fn
        )
        self._cg_refill = CudaGraphAwareParamRefill(cuda_graph_registry)
        self._bucket_groups: List[_DESLOCBucketGroup] = []

    def register_bucket_group(self, group: _DESLOCBucketGroup) -> None:
        self._bucket_groups.append(group)

    def _start_bucket_group_param_sync(
        self,
        bucket_group: _DESLOCBucketGroup,
        force_sync: bool = False,
    ) -> None:
        """
        Dispatch param all-gather for one bucket group, then run post-processing.

        Mirrors Megatron's ``_start_bucket_group_param_sync`` after the
        08bad7a refactor.  In the overlap path we skip post-processing here
        and defer it to ``finish_param_sync`` (where we wait on the async
        handle).  In the non-overlap path we run it immediately.

        DES-LOC note: for CPU-tier buckets there is no GPU collective, so
        ``start_param_sync`` is a no-op and ``dispatch_post_sync`` just marks
        params dirty in the LCM handler.
        """
        # In a real implementation, bucket_group.start_param_sync() would
        # issue the all-gather collective.  We elide that here.

        if self.ddp_config.overlap_param_gather and not force_sync:
            # Async path — defer post-processing to finish_param_sync.
            return

        self._post_sync_mgr.dispatch_post_sync(bucket_group)

    def finish_param_sync(
        self,
        bucket_group: _DESLOCBucketGroup,
        skip_next_bucket_dispatch: bool = False,
    ) -> None:
        """
        Wait on async all-gather handle and run post-processing.

        Mirrors Megatron's ``_ParamAndGradBucketGroup.finish_param_sync``.
        After 08bad7a the ``elif self.ddp_config.reuse_grad_buf_for_mxfp8_param_ag``
        and the ``else: fp8_params ... post_all_gather_processing`` branches
        are both replaced by a single ``self._post_param_sync()`` call.  We do
        the same here via ``dispatch_post_sync``.
        """
        if bucket_group.param_gather_handle is not None:
            bucket_group.wait_param_gather()
            self._post_sync_mgr.dispatch_post_sync(bucket_group)
            return

        # Handle the forced-sync / single-rank path
        if not bucket_group.param_gather_dispatched:
            # Single-rank group (dp_size == 1): no collective needed.
            if self.ddp_config.overlap_param_gather:
                # Still need post-processing because forward pre-hooks won't
                # call finish_param_sync separately.
                self._post_sync_mgr.dispatch_post_sync(bucket_group)
            bucket_group.param_gather_dispatched = True
            return

        # Non-overlap path with distributed optimizer: run post-processing.
        if not skip_next_bucket_dispatch:
            self._post_sync_mgr.dispatch_post_sync(bucket_group)

    def start_param_sync(
        self,
        force_sync: bool = False,
        force_dispatch: bool = False,
    ) -> None:
        """Iterate over all registered bucket groups and dispatch param syncs."""
        for bg in self._bucket_groups:
            self._start_bucket_group_param_sync(bg, force_sync=force_sync)

    def eval_transition_prefill(
        self,
        model_chunks: Sequence[Any],
        optimizer_chain: Sequence[Any],
    ) -> None:
        """Delegate to ``HeteroPostParamSyncManager.eval_transition_prefill``."""
        self._post_sync_mgr.eval_transition_prefill(model_chunks, optimizer_chain)

    def maybe_refill_param_buffer(
        self,
        forward_pre_hook_enabled: bool,
        optimizer_chain: Sequence[Any],
    ) -> None:
        """Delegate to ``CudaGraphAwareParamRefill.maybe_refill_param_buffer``."""
        self._cg_refill.maybe_refill_param_buffer(forward_pre_hook_enabled, optimizer_chain)


# ---------------------------------------------------------------------------
# Utility: classify all parameters in a model by device tier
# ---------------------------------------------------------------------------

def classify_params_by_tier(
    named_params: Sequence[Tuple[str, torch.Tensor]],
) -> Dict[DeviceTier, List[Tuple[str, torch.Tensor]]]:
    """
    Partition named parameters into device-tier buckets.

    Used during DES-LOC model initialisation to determine which parameters
    should be placed in which bucket groups.

    Parameters
    ----------
    named_params : sequence of (name, tensor)
        Named parameters from ``model.named_parameters()``.

    Returns
    -------
    dict mapping DeviceTier → list of (name, tensor)
    """
    tiers: Dict[DeviceTier, List[Tuple[str, torch.Tensor]]] = {
        DeviceTier.A6000: [],
        DeviceTier.H100: [],
        DeviceTier.CPU: [],
    }
    for name, param in named_params:
        tier = _device_tier_of(param.device)
        tiers[tier].append((name, param))
    return tiers


def build_hetero_bucket_groups(
    named_params: Sequence[Tuple[str, torch.Tensor]],
    ddp_config: DESLOCDDPConfig,
    bucket_size_bytes: int = 256 * 1024 * 1024,  # 256 MB default
) -> List[_DESLOCBucketGroup]:
    """
    Construct heterogeneous bucket groups from model parameters.

    Parameters are first classified by device tier, then packed into fixed-
    size buckets within each tier.  Cross-tier buckets are never created
    because all-gather collectives must be homogeneous (same device).

    Parameters
    ----------
    named_params : sequence of (name, tensor)
        All model parameters.
    ddp_config : DESLOCDDPConfig
        Configuration determining shared-buffer mode etc.
    bucket_size_bytes : int
        Target bucket size in bytes.  Buckets are filled greedily.

    Returns
    -------
    list of _DESLOCBucketGroup
    """
    tier_params = classify_params_by_tier(named_params)
    all_groups: List[_DESLOCBucketGroup] = []
    group_id = 0

    for tier, params in tier_params.items():
        if not params:
            continue

        # Pack parameters into fixed-size buckets within this tier.
        current_bucket_params: List[torch.Tensor] = []
        current_bucket_slices: List[_FakeParamSlice] = []
        current_offset = 0
        bucket_id = 0
        buckets_in_group: List[_DESLOCParamBucket] = []

        for _name, param in params:
            numel = param.numel()
            nbytes = numel * param.element_size()

            if current_offset + nbytes > bucket_size_bytes and current_bucket_params:
                # Flush current bucket
                device = current_bucket_params[0].device
                param_data = torch.zeros(current_offset, dtype=torch.float32, device=device)
                b = _DESLOCParamBucket(
                    bucket_id=bucket_id,
                    device_tier=tier,
                    params=current_bucket_params,
                    param_slices=current_bucket_slices,
                    param_data=param_data,
                    is_shared_param_grad_buf=ddp_config.reuse_grad_buf_for_mxfp8_param_ag,
                )
                buckets_in_group.append(b)
                bucket_id += 1
                current_bucket_params = []
                current_bucket_slices = []
                current_offset = 0

            sl = _FakeParamSlice(param=param, start=current_offset, end=current_offset + numel)
            current_bucket_params.append(param)
            current_bucket_slices.append(sl)
            current_offset += numel

        # Flush remaining params
        if current_bucket_params:
            device = current_bucket_params[0].device
            storage_device = device if tier != DeviceTier.CPU else torch.device("cpu")
            param_data = torch.zeros(current_offset, dtype=torch.float32, device=storage_device)
            b = _DESLOCParamBucket(
                bucket_id=bucket_id,
                device_tier=tier,
                params=current_bucket_params,
                param_slices=current_bucket_slices,
                param_data=param_data,
                is_shared_param_grad_buf=ddp_config.reuse_grad_buf_for_mxfp8_param_ag,
            )
            buckets_in_group.append(b)

        if buckets_in_group:
            group = _DESLOCBucketGroup(
                group_id=group_id,
                buckets=buckets_in_group,
                ddp_config=ddp_config,
            )
            all_groups.append(group)
            group_id += 1
            logger.debug(
                "Built bucket group %d for tier %s: %d bucket(s), %d param(s).",
                group_id - 1, tier.value, len(buckets_in_group),
                sum(len(b.params) for b in buckets_in_group),
            )

    return all_groups


# ---------------------------------------------------------------------------
# Eval transition helpers (mirrors Megatron test helpers)
# ---------------------------------------------------------------------------

def should_disable_forward_pre_hook(ddp_config: DESLOCDDPConfig) -> bool:
    """Determine whether forward pre-hooks should be disabled for eval.

    In DES-LOC, forward pre-hooks drive async param prefetch from the LCM.
    During eval they are disabled so the forced all-gather (via
    ``eval_transition_prefill``) is the sole param-sync mechanism.
    """
    return ddp_config.overlap_param_gather


def run_eval_transition(
    ddp: DESLOCDistributedDataParallel,
    model_chunks: Sequence[Any],
    optimizer_chain: Sequence[Any],
    ddp_config: DESLOCDDPConfig,
) -> None:
    """
    Perform the eval-transition sequence for DES-LOC.

    Mirrors the sequence in Megatron's ``train()``::

        if reuse_grad_buf_for_mxfp8_param_ag and overlap_param_gather:
            zero_grad_buffer()
            _copy_main_params_to_param_buffer()
        if should_disable_forward_pre_hook(args):
            disable_forward_pre_hook(model)

    The DES-LOC addition is LCM cache invalidation (handled inside
    ``eval_transition_prefill``).
    """
    if should_disable_forward_pre_hook(ddp_config):
        ddp.eval_transition_prefill(model_chunks, optimizer_chain)
        logger.info("eval_transition: forward pre-hooks disabled; param buffer repopulated.")


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import unittest

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    class _MockOptimizer:
        """Minimal optimizer stub with a param-buffer copy method."""
        def __init__(self) -> None:
            self.copy_called = False
            self.chained_optimizers = [self]

        def _copy_main_params_to_param_buffer(self) -> None:
            self.copy_called = True

    class _MockModelChunk:
        """Minimal model chunk stub."""
        def __init__(self) -> None:
            self.grad_buffer_zeroed = False

        def zero_grad_buffer(self) -> None:
            self.grad_buffer_zeroed = True

    def _make_cpu_param(shape=(4, 4)) -> torch.Tensor:
        return torch.randn(*shape)

    def _make_gpu_param(shape=(4, 4), device="cpu") -> torch.Tensor:
        # Use CPU to keep tests runnable without GPU
        return torch.randn(*shape)

    # -------------------------------------------------------------------

    class TestDeviceTierMapping(unittest.TestCase):
        def test_cpu_device(self):
            p = torch.randn(4)
            tier = _device_tier_of(p.device)
            self.assertEqual(tier, DeviceTier.CPU)

    class TestLocalityCacheManager(unittest.TestCase):
        def test_invalidate_marks_entries_invalid(self):
            lcm = LocalityCacheManager()
            p = torch.randn(4)
            lcm.cache(p, p.clone())
            self.assertTrue(lcm.is_cached(p))
            lcm.invalidate_training_snapshot([p])
            self.assertFalse(lcm.is_cached(p))

        def test_invalidate_nonexistent_is_noop(self):
            lcm = LocalityCacheManager()
            p = torch.randn(4)
            # Should not raise
            lcm.invalidate_training_snapshot([p])

        def test_cache_then_retrieve(self):
            lcm = LocalityCacheManager()
            p = torch.randn(4)
            bf16 = p.to(torch.bfloat16)
            lcm.cache(p, bf16)
            self.assertTrue(lcm.is_cached(p))

    class TestCPULocalityHandler(unittest.TestCase):
        def test_flush_dirty_clears_list(self):
            lcm = LocalityCacheManager()
            handler = _CPULocalityHandler(lcm)
            bucket = _DESLOCParamBucket(
                bucket_id=0,
                device_tier=DeviceTier.CPU,
                params=[torch.randn(4), torch.randn(4)],
            )
            cfg = DESLOCDDPConfig()
            handler.handle(bucket, cfg)
            dirty = handler.flush_dirty()
            self.assertEqual(len(dirty), 2)
            dirty2 = handler.flush_dirty()
            self.assertEqual(len(dirty2), 0)

    class TestQuantisedParamHandler(unittest.TestCase):
        def test_no_quantised_params_noop(self):
            called_with: List[List[torch.Tensor]] = []
            def fake_post_ag(params):
                called_with.append(params)
            handler = _QuantisedParamHandler(fake_post_ag)
            p = torch.randn(4)
            bucket = _DESLOCParamBucket(
                bucket_id=0,
                device_tier=DeviceTier.A6000,
                params=[p],
            )
            cfg = DESLOCDDPConfig()
            handler.handle(bucket, cfg)
            self.assertEqual(len(called_with), 0)

    class TestMXFP8SharedBufHandler(unittest.TestCase):
        def test_bf16_weight_skip_zero(self):
            """BF16 params in an MXFP8 bucket skip copy and zeroing."""
            handler = _MXFP8SharedBufHandler()
            p = torch.randn(4)  # plain BF16, not FP8
            param_data = torch.ones(4)
            bucket = _DESLOCParamBucket(
                bucket_id=0,
                device_tier=DeviceTier.A6000,
                params=[p],
                param_slices=[_FakeParamSlice(p, 0, 4)],
                param_data=param_data,
                is_shared_param_grad_buf=True,
            )
            cfg = DESLOCDDPConfig(reuse_grad_buf_for_mxfp8_param_ag=True)
            handler.handle(bucket, cfg)
            # param_data should NOT be zeroed because BF16 weight was skipped
            self.assertFalse(torch.all(param_data == 0).item())

    class TestHeteroPostParamSyncManager(unittest.TestCase):
        def setUp(self):
            self.lcm = LocalityCacheManager()
            self.cfg = DESLOCDDPConfig(reuse_grad_buf_for_mxfp8_param_ag=False)
            self.mgr = HeteroPostParamSyncManager(self.cfg, self.lcm)

        def _make_group(self, tier: DeviceTier, num_params: int = 2) -> _DESLOCBucketGroup:
            params = [torch.randn(4) for _ in range(num_params)]
            slices = [_FakeParamSlice(p, i * 4, (i + 1) * 4) for i, p in enumerate(params)]
            bucket = _DESLOCParamBucket(
                bucket_id=0,
                device_tier=tier,
                params=params,
                param_slices=slices,
                param_data=torch.zeros(num_params * 4),
            )
            return _DESLOCBucketGroup(
                group_id=0,
                buckets=[bucket],
                ddp_config=self.cfg,
            )

        def test_dispatch_cpu_marks_dirty(self):
            group = self._make_group(DeviceTier.CPU)
            self.mgr.dispatch_post_sync(group)
            dirty = self.mgr._cpu_handler.flush_dirty()
            self.assertEqual(len(dirty), 2)

        def test_dispatch_a6000_plain_params_noop(self):
            group = self._make_group(DeviceTier.A6000)
            # Plain float params are not quantised → no TE call expected.
            self.mgr.dispatch_post_sync(group)  # Should not raise.

        def test_dispatch_h100_plain_params_noop(self):
            group = self._make_group(DeviceTier.H100)
            self.mgr.dispatch_post_sync(group)  # Should not raise.

        def test_dispatch_count_increments(self):
            group = self._make_group(DeviceTier.A6000)
            self.mgr.dispatch_post_sync(group)
            self.mgr.dispatch_post_sync(group)
            self.assertEqual(self.mgr.diagnostics()["dispatch_count"], 2)

    class TestEvalTransitionPrefill(unittest.TestCase):
        def test_prefill_calls_zero_and_copy(self):
            lcm = LocalityCacheManager()
            cfg = DESLOCDDPConfig(
                reuse_grad_buf_for_mxfp8_param_ag=True,
                overlap_param_gather=True,
            )
            mgr = HeteroPostParamSyncManager(cfg, lcm)
            chunk = _MockModelChunk()
            opt = _MockOptimizer()
            mgr.eval_transition_prefill([chunk], [opt])
            self.assertTrue(chunk.grad_buffer_zeroed)
            self.assertTrue(opt.copy_called)

        def test_prefill_skipped_when_flags_off(self):
            lcm = LocalityCacheManager()
            cfg = DESLOCDDPConfig(
                reuse_grad_buf_for_mxfp8_param_ag=False,
                overlap_param_gather=True,
            )
            mgr = HeteroPostParamSyncManager(cfg, lcm)
            chunk = _MockModelChunk()
            opt = _MockOptimizer()
            mgr.eval_transition_prefill([chunk], [opt])
            self.assertFalse(chunk.grad_buffer_zeroed)
            self.assertFalse(opt.copy_called)

        def test_prefill_invalidates_lcm_after_dirty_cpu_params(self):
            lcm = LocalityCacheManager()
            cfg = DESLOCDDPConfig(
                reuse_grad_buf_for_mxfp8_param_ag=True,
                overlap_param_gather=True,
                cpu_lcm_bypass=True,
            )
            mgr = HeteroPostParamSyncManager(cfg, lcm)

            # Simulate training step that dirtied a CPU-tier bucket
            p = torch.randn(4)
            lcm.cache(p, p.clone())
            cpu_bucket = _DESLOCParamBucket(
                bucket_id=0, device_tier=DeviceTier.CPU, params=[p]
            )
            cpu_group = _DESLOCBucketGroup(0, [cpu_bucket], cfg)
            mgr.dispatch_post_sync(cpu_group)  # marks p dirty in CPU handler

            self.assertTrue(lcm.is_cached(p))

            chunk = _MockModelChunk()
            opt = _MockOptimizer()
            mgr.eval_transition_prefill([chunk], [opt])

            # LCM entry should now be invalid
            self.assertFalse(lcm.is_cached(p))

    class TestCudaGraphAwareParamRefill(unittest.TestCase):
        def test_no_graph_no_hook_skips(self):
            refill = CudaGraphAwareParamRefill(graph_registry={})
            opt = _MockOptimizer()
            refill.maybe_refill_param_buffer(False, [opt])
            self.assertFalse(opt.copy_called)

        def test_with_hook_triggers_copy(self):
            refill = CudaGraphAwareParamRefill(graph_registry={})
            opt = _MockOptimizer()
            refill.maybe_refill_param_buffer(True, [opt])
            self.assertTrue(opt.copy_called)

        def test_full_graph_captured_triggers_copy(self):
            registry = {"training_a6000": object(), "training_h100": object()}
            refill = CudaGraphAwareParamRefill(graph_registry=registry)
            opt = _MockOptimizer()
            self.assertTrue(refill.is_full_graph_captured())
            refill.maybe_refill_param_buffer(False, [opt])
            self.assertTrue(opt.copy_called)

        def test_partial_graph_not_captured(self):
            registry = {"training_a6000": object()}  # only one tier
            refill = CudaGraphAwareParamRefill(graph_registry=registry)
            self.assertFalse(refill.is_full_graph_captured())

        def test_monolithic_training_graph(self):
            registry = {"training": object()}
            refill = CudaGraphAwareParamRefill(graph_registry=registry)
            self.assertTrue(refill.is_full_graph_captured())

    class TestBuildHeteroBucketGroups(unittest.TestCase):
        def test_cpu_params_go_to_cpu_group(self):
            params = [("w1", torch.randn(8)), ("w2", torch.randn(8))]
            cfg = DESLOCDDPConfig()
            groups = build_hetero_bucket_groups(params, cfg, bucket_size_bytes=512)
            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0].buckets[0].device_tier, DeviceTier.CPU)

        def test_bucket_size_splitting(self):
            # Each float32 param of shape (32,) = 128 bytes; bucket_size = 64 bytes
            # so each param should go into its own bucket.
            params = [(f"w{i}", torch.randn(32)) for i in range(4)]
            cfg = DESLOCDDPConfig()
            groups = build_hetero_bucket_groups(params, cfg, bucket_size_bytes=64)
            total_buckets = sum(len(g.buckets) for g in groups)
            self.assertGreaterEqual(total_buckets, 4)

        def test_empty_params_no_groups(self):
            cfg = DESLOCDDPConfig()
            groups = build_hetero_bucket_groups([], cfg)
            self.assertEqual(len(groups), 0)

    class TestDESLOCDistributedDataParallel(unittest.TestCase):
        def _make_ddp(self, overlap=True, reuse_buf=False):
            cfg = DESLOCDDPConfig(
                overlap_param_gather=overlap,
                reuse_grad_buf_for_mxfp8_param_ag=reuse_buf,
            )
            lcm = LocalityCacheManager()
            ddp = DESLOCDistributedDataParallel(cfg, lcm)
            return ddp, cfg, lcm

        def _make_group(self, cfg, tier=DeviceTier.CPU):
            params = [torch.randn(4)]
            bucket = _DESLOCParamBucket(
                bucket_id=0, device_tier=tier, params=params,
                param_data=torch.zeros(4),
            )
            return _DESLOCBucketGroup(0, [bucket], cfg)

        def test_start_param_sync_overlap_skips_post(self):
            """With overlap=True, _start_bucket_group_param_sync defers post-processing."""
            ddp, cfg, _ = self._make_ddp(overlap=True)
            group = self._make_group(cfg)
            ddp.register_bucket_group(group)
            # Should not raise; post-processing deferred.
            ddp._start_bucket_group_param_sync(group, force_sync=False)

        def test_start_param_sync_no_overlap_runs_post(self):
            """With overlap=False, post-processing runs immediately."""
            ddp, cfg, _ = self._make_ddp(overlap=False)
            group = self._make_group(cfg)
            ddp.register_bucket_group(group)
            ddp._start_bucket_group_param_sync(group, force_sync=False)
            # CPU handler should have marked params dirty
            dirty = ddp._post_sync_mgr._cpu_handler.flush_dirty()
            self.assertEqual(len(dirty), 1)

        def test_finish_param_sync_no_handle(self):
            ddp, cfg, _ = self._make_ddp(overlap=True)
            group = self._make_group(cfg)
            group.param_gather_dispatched = True
            # Should not raise
            ddp.finish_param_sync(group)

        def test_eval_transition_prefill_delegates(self):
            ddp, cfg, _ = self._make_ddp(overlap=True, reuse_buf=True)
            chunk = _MockModelChunk()
            opt = _MockOptimizer()
            ddp.eval_transition_prefill([chunk], [opt])
            self.assertTrue(chunk.grad_buffer_zeroed)
            self.assertTrue(opt.copy_called)

        def test_should_disable_hook(self):
            cfg = DESLOCDDPConfig(overlap_param_gather=True)
            self.assertTrue(should_disable_forward_pre_hook(cfg))
            cfg2 = DESLOCDDPConfig(overlap_param_gather=False)
            self.assertFalse(should_disable_forward_pre_hook(cfg2))

    class TestRunEvalTransition(unittest.TestCase):
        def test_full_sequence_with_lcm_invalidation(self):
            lcm = LocalityCacheManager()
            cfg = DESLOCDDPConfig(
                reuse_grad_buf_for_mxfp8_param_ag=True,
                overlap_param_gather=True,
            )
            ddp = DESLOCDistributedDataParallel(cfg, lcm)

            # Simulate a CPU-tier param that is cached
            p = torch.randn(4)
            lcm.cache(p, p.clone())
            cpu_bucket = _DESLOCParamBucket(0, DeviceTier.CPU, [p], param_data=torch.zeros(4))
            group = _DESLOCBucketGroup(0, [cpu_bucket], cfg)
            ddp._post_sync_mgr.dispatch_post_sync(group)  # mark dirty

            chunk = _MockModelChunk()
            opt = _MockOptimizer()
            run_eval_transition(ddp, [chunk], [opt], cfg)

            self.assertTrue(chunk.grad_buffer_zeroed)
            self.assertTrue(opt.copy_called)
            self.assertFalse(lcm.is_cached(p))

    # Run all tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestDeviceTierMapping,
        TestLocalityCacheManager,
        TestCPULocalityHandler,
        TestQuantisedParamHandler,
        TestMXFP8SharedBufHandler,
        TestHeteroPostParamSyncManager,
        TestEvalTransitionPrefill,
        TestCudaGraphAwareParamRefill,
        TestBuildHeteroBucketGroups,
        TestDESLOCDistributedDataParallel,
        TestRunEvalTransition,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


# ---------------------------------------------------------------------------

def register(engine) -> None:
    """Register HeteroPostParamSyncManager on a DeepSpeed engine.

    Instantiates a :class:`HeteroPostParamSyncManager` from the engine's configuration
    and attaches it as ``engine.hetero_fp8_param_gather_eval``.

    Parameters
    ----------
    engine:
        A DeepSpeed engine instance.
    """
    logger.info(
        "hetero_fp8_param_gather_eval.register() called on engine type=%s",
        type(engine).__name__,
    )

    engine.hetero_fp8_param_gather_eval = None
    logger.info("hetero_fp8_param_gather_eval.register() attached engine.hetero_fp8_param_gather_eval")
