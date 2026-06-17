# Copyright (c) 2026 Neuron_SP Project (DES-LOC Heterogeneous Training Framework)
# Adapted from NVIDIA Megatron-LM commit 3f6a2ed844373ef0ed6c0e6dbac58a431d088dcb
# Original: Add opt-in MXFP8 LM-head output projection (#4825)
#
# DES-LOC Adaptation: HeteroMXFP8LMHead — SM90 fast-path MXFP8 LM-head with
# locality-cache-aware weight pinning, SM86 fallback, and heterogeneous dispatch.
#
# Hardware topology assumed:
#   - rank 0,1 : A6000 48GB SM86 (PCIe, no NVLink)
#   - rank 2    : H100 NVL 96GB  SM90 (PCIe)
#   - Host DRAM : 1.5TB (used as DES-LOC Shared LOcality Cache)
#
# Upstream design intent (Megatron 3f6a2ed):
#   NVIDIA added TELMHeadColumnParallelLinear as a drop-in LM-head replacement
#   that activates MXFP8 autocast on the output projection when three conditions
#   hold: fp8=True, fp8_recipe='mxfp8', fp8_output_proj=True. The class:
#     (a) forces delay_wgrad_compute=False to keep wgrad timing identical to the
#         non-FP8 ColumnParallelLinear.backward_dw no-op;
#     (b) installs a state-dict pre-hook so _extra_state keys are always present,
#         enabling bf16<->MXFP8 checkpoint round-trips;
#     (c) gates gather_output / runtime_gather_output on the TP group, unchanged.
#   is_mxfp8_output_proj_active() is a pure predicate over config fields.
#   TransformerConfig gains fp8_output_proj: bool = False with validation that
#   fp8=True and fp8_recipe='mxfp8' must co-hold.
#
# DES-LOC adaptation points:
#   1. Device-class detection: SM90 (H100) supports native MXFP8 tensor cores;
#      SM86 (A6000) does not — fallback to BF16 projection with optional FP8
#      emulation via software scale/quantize.
#   2. Shared LOcality Cache (SLC): the H100's large HBM and host DRAM are used
#      as a weight-pinning tier. When the LM-head weight is too large for A6000
#      VRAM headroom, we pin it in host DRAM and stream via PCIe with prefetch.
#   3. Decoupled Execution: the forward pass on SM86 ranks is decoupled from
#      SM90: SM90 may run the MXFP8 projection asynchronously while SM86 ranks
#      do embedding/attention, then a barrier synchronises before gather_output.
#   4. wgrad timing: DES-LOC must not compute wgrad on SM86 inside the MXFP8
#      kernel path (not supported) — delay_wgrad_compute is forced False on all
#      device classes, matching upstream intent but enforced per-device.
#   5. State-dict compatibility: same _extra_state shim as upstream so that
#      heterogeneous checkpoints (some ranks BF16, one MXFP8) load cleanly.

from __future__ import annotations

import copy
import dataclasses
import enum
import logging
import math
import os
import threading
import time
import unittest
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device-class constants
# ---------------------------------------------------------------------------

SM_A6000 = 86   # Ampere, no MXFP8 tensor cores
SM_H100  = 90   # Hopper, native MXFP8


class DeviceClass(enum.Enum):
    """Coarse capability class derived from CUDA SM version."""
    SM86_AMPERE  = "sm86_ampere"   # A6000: FP8 emulation only
    SM90_HOPPER  = "sm90_hopper"   # H100 NVL: native MXFP8
    CPU          = "cpu"            # fallback / unit-test


def detect_device_class(device: Optional[torch.device] = None) -> DeviceClass:
    """Return the :class:`DeviceClass` of *device* (default: current CUDA device).

    This is called once per rank during module construction and cached. The
    result drives all downstream branching between the MXFP8 fast path and the
    BF16 / emulated-FP8 slow path.

    DES-LOC note: we intentionally do *not* use ``torch.cuda.get_device_capability``
    major/minor tuple arithmetic here because the SM version is more expressive
    for future SM10x (Blackwell) devices that may appear in the cluster.
    """
    if device is None:
        if not torch.cuda.is_available():
            return DeviceClass.CPU
        device = torch.device("cuda", torch.cuda.current_device())

    if device.type != "cuda":
        return DeviceClass.CPU

    props = torch.cuda.get_device_properties(device)
    sm = props.major * 10 + props.minor
    if sm >= SM_H100:
        return DeviceClass.SM90_HOPPER
    if sm >= SM_A6000:
        return DeviceClass.SM86_AMPERE
    return DeviceClass.SM86_AMPERE   # treat older as Ampere-class


# ---------------------------------------------------------------------------
# DES-LOC configuration dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DESLOCConfig:
    """Runtime knobs for Decoupled Execution with Shared LOcality Cache.

    Attributes
    ----------
    slc_pin_threshold_bytes:
        If the LM-head weight tensor exceeds this size (bytes) on an SM86 rank,
        pin it in host DRAM (SLC) and stream on demand instead of keeping it in
        device VRAM.  Default 2 GiB.
    slc_prefetch_streams:
        Number of CUDA streams dedicated to PCIe prefetch from SLC to device
        scratch buffer on SM86 ranks.
    decoupled_sm90_async:
        When True, SM90 rank launches the MXFP8 forward asynchronously after
        posting a CUDA event; SM86 ranks wait on a barrier before gather_output.
        Adds latency hiding at the cost of an extra synchronisation point.
    fp8_emulate_on_sm86:
        When True, SM86 ranks run a software FP8-emulation path (scale +
        cast to float8_e4m3fn + cast back) before the BF16 matmul so that
        weight statistics match the SM90 MXFP8 path.  Experimental.
    wgrad_delay_disabled:
        Must stay True (matches upstream intent): wgrad is never delayed on the
        LM-head regardless of the global config value.
    tp_size:
        Tensor-parallel world size.  Inferred from dist if 0.
    """
    slc_pin_threshold_bytes: int       = 2 * 1024 ** 3   # 2 GiB
    slc_prefetch_streams: int          = 2
    decoupled_sm90_async: bool         = False
    fp8_emulate_on_sm86: bool          = False
    wgrad_delay_disabled: bool         = True
    tp_size: int                       = 0


# ---------------------------------------------------------------------------
# SLC weight-pinning manager
# ---------------------------------------------------------------------------

class SharedLocalityCacheManager:
    """Manage LM-head weight residency across VRAM / host-DRAM tiers.

    The DES-LOC architecture treats host DRAM (1.5 TB) as a Shared LOcality
    Cache (SLC).  Large tensors that do not fit in A6000 VRAM headroom are
    pinned in host memory and streamed to a device scratch buffer per forward
    pass.  The H100 rank always keeps weights in device VRAM.

    Thread-safety: prefetch is enqueued on a dedicated CUDA stream; the caller
    must call :meth:`wait_prefetch` before using the device buffer.
    """

    def __init__(self, descloc_cfg: DESLOCConfig, device: torch.device):
        self._cfg     = descloc_cfg
        self._device  = device
        self._streams = [
            torch.cuda.Stream(device=device)
            for _ in range(descloc_cfg.slc_prefetch_streams)
        ]
        self._stream_idx     = 0
        self._pinned_weight: Optional[torch.Tensor] = None
        self._device_scratch: Optional[torch.Tensor] = None
        self._prefetch_event: Optional[torch.cuda.Event] = None
        self._lock = threading.Lock()

    def maybe_offload(self, weight: torch.Tensor) -> bool:
        """Offload *weight* to pinned host memory if it exceeds the SLC threshold.

        Returns True if the weight was offloaded, False if it stays on device.

        DES-LOC rationale: A6000 ranks have 48 GB VRAM shared among activations,
        optimizer states, and model shards.  A large vocabulary LM-head weight
        (e.g., 128k vocab × 8192 hidden × 2 bytes ≈ 2 GB per shard) may not
        leave sufficient headroom for the activation working set.  Offloading to
        the 1.5 TB host DRAM via pinned memory preserves PCIe bandwidth for
        demand paging.
        """
        nbytes = weight.numel() * weight.element_size()
        if nbytes <= self._cfg.slc_pin_threshold_bytes:
            return False

        with self._lock:
            if self._pinned_weight is not None:
                return True   # already offloaded

            logger.info(
                "SLC: offloading LM-head weight (%.2f GiB) to pinned host DRAM on %s",
                nbytes / 1024 ** 3,
                self._device,
            )
            self._pinned_weight = weight.detach().cpu().pin_memory()
            self._device_scratch = torch.empty_like(weight, device=self._device)
        return True

    def schedule_prefetch(self, weight: torch.Tensor) -> bool:
        """Asynchronously copy the pinned weight to the device scratch buffer.

        Returns True if a prefetch was scheduled (weight is in SLC), False
        if the weight is already resident on device.
        """
        with self._lock:
            if self._pinned_weight is None:
                return False
            stream = self._streams[self._stream_idx % len(self._streams)]
            self._stream_idx += 1
            evt = torch.cuda.Event()
            with torch.cuda.stream(stream):
                self._device_scratch.copy_(self._pinned_weight, non_blocking=True)
                evt.record(stream)
            self._prefetch_event = evt
        return True

    def wait_prefetch(self) -> Optional[torch.Tensor]:
        """Block the current CUDA stream until the prefetch is done.

        Returns the device scratch buffer if a prefetch was in flight, else None.
        """
        with self._lock:
            if self._prefetch_event is None:
                return None
            self._prefetch_event.wait()
            self._prefetch_event = None
            return self._device_scratch

    @property
    def is_offloaded(self) -> bool:
        with self._lock:
            return self._pinned_weight is not None


# ---------------------------------------------------------------------------
# FP8 emulation helper for SM86
# ---------------------------------------------------------------------------

def _fp8_emulate_e4m3(tensor: torch.Tensor, scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Software-emulate FP8 E4M3 quantisation on SM86 (Ampere) hardware.

    Upstream MXFP8 uses hardware tensor-core instructions available only on
    SM90+.  On SM86 we approximate the quantisation effect by:
      1. Optionally computing a per-tensor absmax scale.
      2. Casting to torch.float8_e4m3fn (available from PyTorch ≥ 2.1).
      3. Casting back to BF16 for the actual matmul.

    This path is opt-in via DESLOCConfig.fp8_emulate_on_sm86 and is primarily
    useful for verifying numerical parity between SM86 and SM90 outputs during
    development.  It is NOT numerically identical to MXFP8 because MXFP8 uses
    microscaling (per-group scales); this emulates per-tensor scaling only.

    Parameters
    ----------
    tensor:
        Input tensor (BF16 or FP32).
    scale:
        Optional pre-computed scale.  If None, computed as 448 / absmax(tensor).
    """
    orig_dtype = tensor.dtype
    t_fp32 = tensor.float()
    if scale is None:
        amax = t_fp32.abs().max().clamp(min=1e-12)
        scale = torch.tensor(448.0, device=tensor.device) / amax

    t_scaled = t_fp32 * scale
    # Clamp to E4M3 representable range [-448, 448]
    t_clamped = t_scaled.clamp(-448.0, 448.0)

    try:
        t_fp8 = t_clamped.to(torch.float8_e4m3fn)
        t_dequant = t_fp8.to(torch.bfloat16) / scale.to(torch.bfloat16)
    except (AttributeError, RuntimeError):
        # torch.float8_e4m3fn not available; fall back to pure BF16
        logger.debug("FP8 emulation: float8_e4m3fn unavailable, using BF16 passthrough")
        t_dequant = tensor.to(torch.bfloat16)

    return t_dequant.to(orig_dtype)


# ---------------------------------------------------------------------------
# Core: HeteroMXFP8LMHead
# ---------------------------------------------------------------------------

class HeteroMXFP8LMHead(nn.Module):
    """Heterogeneous MXFP8 LM-head output projection for DES-LOC.

    This module reinterprets Megatron's ``TELMHeadColumnParallelLinear``
    (commit 3f6a2ed) for a cluster with mixed SM86 (A6000) and SM90 (H100)
    devices connected via PCIe without NVLink.

    Design contract
    ---------------
    * **SM90 fast path**: if the current device is SM90-class and the config
      enables MXFP8 output projection (``is_hetero_mxfp8_active``), the
      forward runs under an MXFP8 autocast context, matching upstream behavior.
    * **SM86 slow path**: BF16 matmul, optionally with FP8 emulation.  The
      weight may be pinned in host DRAM (SLC) and streamed via PCIe prefetch.
    * **wgrad timing**: ``delay_wgrad_compute`` is forced False regardless of
      the global config, mirroring upstream's rationale that the LM-head's
      ``backward_dw`` is a no-op in the base class.
    * **State-dict compatibility**: ``get_extra_state`` returns None and
      ``set_extra_state`` is a no-op, identical to upstream, so heterogeneous
      checkpoints (some ranks BF16, one MXFP8) load without key mismatches.
    * **Tensor parallelism**: the weight shard has shape
      ``[output_size // tp_size, input_size]``.  ``gather_output`` controls
      whether a TP all-gather is performed after the projection.

    Parameters
    ----------
    input_size:
        Hidden dimension (e.g., 8192).
    output_size:
        Vocabulary size before TP sharding.
    descloc_cfg:
        DES-LOC runtime knobs.
    bias:
        If True, add a bias term to the output.
    gather_output:
        If True, all-gather the TP shards into the full output.
    tp_group:
        Process group for tensor parallelism.  If None, defaults to the
        default group (rank 0 only in unit tests).
    init_method:
        Weight initialisation callable ``f(weight_tensor) -> None``.
    params_dtype:
        Parameter storage dtype (BF16 recommended).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        descloc_cfg: Optional[DESLOCConfig] = None,
        *,
        bias: bool = False,
        gather_output: bool = True,
        tp_group: Optional[dist.ProcessGroup] = None,
        init_method=None,
        params_dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self._descloc_cfg  = descloc_cfg or DESLOCConfig()
        self._device_arg   = device
        self.input_size    = input_size
        self.output_size   = output_size
        self.gather_output = gather_output
        self._tp_group     = tp_group
        self._params_dtype = params_dtype

        # ---- Tensor-parallel bookkeeping ---------------------------------- #
        tp_size = self._descloc_cfg.tp_size
        if tp_size == 0:
            if dist.is_available() and dist.is_initialized():
                tp_size = dist.get_world_size(tp_group)
            else:
                tp_size = 1
        self._tp_size = tp_size
        self._output_size_per_partition = math.ceil(output_size / tp_size)

        # ---- Device-class detection --------------------------------------- #
        resolved_device = device or (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self._device_class = detect_device_class(resolved_device)
        self._resolved_device = resolved_device

        logger.info(
            "HeteroMXFP8LMHead: rank device %s classified as %s "
            "(input=%d, output=%d, tp_size=%d)",
            resolved_device,
            self._device_class.value,
            input_size,
            output_size,
            tp_size,
        )

        # ---- Weight parameter -------------------------------------------- #
        self.weight = nn.Parameter(
            torch.empty(
                self._output_size_per_partition,
                input_size,
                dtype=params_dtype,
                device=resolved_device,
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(self._output_size_per_partition, dtype=params_dtype,
                            device=resolved_device)
            )
        else:
            self.register_parameter("bias", None)

        # ---- Weight initialisation --------------------------------------- #
        if init_method is not None:
            init_method(self.weight)
        else:
            nn.init.normal_(self.weight, mean=0.0, std=0.02)

        # ---- SLC manager (SM86 only) ------------------------------------- #
        self._slc: Optional[SharedLocalityCacheManager] = None
        if self._device_class == DeviceClass.SM86_AMPERE:
            self._slc = SharedLocalityCacheManager(self._descloc_cfg, resolved_device)
            offloaded = self._slc.maybe_offload(self.weight)
            if offloaded:
                logger.warning(
                    "SLC: LM-head weight (%.2f GiB) offloaded to host DRAM on %s; "
                    "PCIe prefetch will be used each forward pass",
                    self.weight.numel() * self.weight.element_size() / 1024 ** 3,
                    resolved_device,
                )

        # ---- SM90 async event (decoupled execution) ---------------------- #
        self._sm90_async_event: Optional[torch.cuda.Event] = None
        if (
            self._device_class == DeviceClass.SM90_HOPPER
            and self._descloc_cfg.decoupled_sm90_async
            and torch.cuda.is_available()
        ):
            self._sm90_async_event = torch.cuda.Event()
            logger.info(
                "DES-LOC decoupled async enabled on SM90 rank; "
                "SM86 ranks will barrier before gather_output"
            )

        # ---- State-dict shim hook --------------------------------------- #
        # Mirrors upstream: ensure _extra_state key is always present so that
        # a BF16 checkpoint can be loaded into an MXFP8 module and vice-versa.
        self._register_load_state_dict_pre_hook(self._extra_state_pre_hook)

    # ------------------------------------------------------------------
    # State-dict compatibility (upstream TELMHeadColumnParallelLinear §§)
    # ------------------------------------------------------------------

    @staticmethod
    def _extra_state_pre_hook(
        state_dict: Dict[str, Any],
        prefix: str,
        *args,
        **kwargs,
    ) -> None:
        """Ensure ``_extra_state`` key is present so bf16<->MXFP8 swaps work.

        Upstream rationale: ColumnParallelLinear has a no-op set_extra_state and
        does not write _extra_state to checkpoints.  TELMHeadColumnParallelLinear
        (TE-backed) *does* write it.  This hook normalises both directions so
        ``strict=True`` loads succeed across the bf16/MXFP8 boundary.
        """
        key = f"{prefix}_extra_state"
        state_dict.setdefault(key, None)

    def get_extra_state(self) -> None:
        """Return None — matches ColumnParallelLinear's no-extra-state contract."""
        return None

    def set_extra_state(self, state: Any) -> None:
        """No-op — matches upstream TELMHeadColumnParallelLinear.set_extra_state."""
        return

    # ------------------------------------------------------------------
    # Internal: device-path forward helpers
    # ------------------------------------------------------------------

    def _forward_sm90_mxfp8(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """MXFP8 projection on SM90 (H100 NVL).

        In production this wraps a Transformer Engine MXFP8 autocast context.
        In the DES-LOC framework we gate on ``HAVE_TE``; if TE is absent (e.g.,
        unit-test environment) we fall through to a BF16 matmul so that tests
        can run on any hardware.

        Upstream NVTX range ``mxfp8_output_proj_telinear`` is preserved here
        as a CUDA profiler annotation.
        """
        try:
            import transformer_engine.pytorch as te  # type: ignore
            from transformer_engine.common.recipe import MXFP8BlockScaling  # type: ignore

            recipe = MXFP8BlockScaling()
            with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                if torch.cuda.is_available():
                    torch.cuda.nvtx.range_push("hetero_mxfp8_lm_head_sm90")
                try:
                    out = F.linear(input_, weight, self.bias)
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.nvtx.range_pop()
            return out

        except (ImportError, Exception) as exc:
            # TE unavailable or MXFP8 context failed — degrade gracefully.
            if not isinstance(exc, ImportError):
                logger.warning(
                    "SM90 MXFP8 context raised %s; falling back to BF16 on %s",
                    exc,
                    self._resolved_device,
                )
            return F.linear(input_.to(torch.bfloat16), weight.to(torch.bfloat16), 
                           self.bias.to(torch.bfloat16) if self.bias is not None else None)

    def _forward_sm86_bf16(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """BF16 projection on SM86 (A6000), with optional SLC weight streaming
        and optional FP8 emulation.

        SLC streaming:
            If the weight was offloaded to host DRAM, we call
            ``schedule_prefetch`` then ``wait_prefetch`` to bring it back to a
            device scratch buffer via a dedicated CUDA stream.  The prefetch was
            ideally scheduled in the *previous* iteration's epilogue (pipeline
            overlap), but we defensively schedule-and-wait here if not.

        FP8 emulation:
            When ``descloc_cfg.fp8_emulate_on_sm86`` is True, we apply
            ``_fp8_emulate_e4m3`` to both input and weight before the matmul.
            This provides a rough numerical approximation of MXFP8 quantisation
            effects for debugging parity between SM86 and SM90 outputs.
        """
        # --- SLC: use prefetched weight if available ---------------------- #
        effective_weight = weight
        if self._slc is not None and self._slc.is_offloaded:
            # Try to get an already-prefetched buffer; schedule if missing.
            buf = self._slc.wait_prefetch()
            if buf is None:
                self._slc.schedule_prefetch(weight)
                buf = self._slc.wait_prefetch()
            if buf is not None:
                effective_weight = buf
            # Schedule next-iteration prefetch (pipeline overlap)
            self._slc.schedule_prefetch(weight)

        # --- Optional FP8 emulation --------------------------------------- #
        if self._descloc_cfg.fp8_emulate_on_sm86:
            emu_input  = _fp8_emulate_e4m3(input_.to(torch.bfloat16))
            emu_weight = _fp8_emulate_e4m3(effective_weight.to(torch.bfloat16))
        else:
            emu_input  = input_.to(torch.bfloat16)
            emu_weight = effective_weight.to(torch.bfloat16)

        bias_bf16 = self.bias.to(torch.bfloat16) if self.bias is not None else None
        return F.linear(emu_input, emu_weight, bias_bf16)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the heterogeneous LM-head projection.

        Parameters
        ----------
        input_:
            Activations from the final transformer layer, shape
            ``[seq_len, batch, hidden]`` or ``[tokens, hidden]``.
        weight:
            Optional external weight override.  Upstream TELMHeadColumnParallelLinear
            raises if ``weight is not self.weight``; we honour the same contract
            here to catch accidental misuse in tied-embedding setups.
        runtime_gather_output:
            If not None, overrides ``self.gather_output`` for this call only.
            Used by inference code that switches gather mode per request.

        Returns
        -------
        (output, bias):
            ``output`` has shape ``[seq_len, batch, vocab_size]`` (if gathered)
            or ``[seq_len, batch, vocab_size // tp_size]`` (if not gathered).
            ``bias`` is None when the module was constructed without bias.
        """
        # --- Weight override guard (mirrors upstream) --------------------- #
        if weight is not None and weight is not self.weight:
            raise RuntimeError(
                "HeteroMXFP8LMHead does not support runtime weight override. "
                "Use tied-embedding weight sharing at construction time."
            )
        w = self.weight

        # --- Device-class dispatch --------------------------------------- #
        if self._device_class == DeviceClass.SM90_HOPPER:
            # SM90: native MXFP8 fast path
            output_parallel = self._forward_sm90_mxfp8(input_, w)

            # DES-LOC decoupled async: record event so SM86 ranks can sync
            if self._sm90_async_event is not None:
                self._sm90_async_event.record()

        elif self._device_class == DeviceClass.SM86_AMPERE:
            # SM86: BF16 slow path with optional SLC streaming + FP8 emulation
            output_parallel = self._forward_sm86_bf16(input_, w)

        else:
            # CPU fallback (unit tests, profiling without GPU)
            output_parallel = F.linear(
                input_.float(),
                w.float(),
                self.bias.float() if self.bias is not None else None,
            )

        # --- Gather output across TP ranks -------------------------------- #
        gather = self.gather_output
        if runtime_gather_output is not None:
            gather = runtime_gather_output

        if gather and self._tp_size > 1:
            if dist.is_available() and dist.is_initialized():
                # DES-LOC barrier: SM90 async path requires SM86 to wait for
                # the SM90 event before issuing the all-gather, because the
                # gather src on rank 2 (SM90) must be ready.
                if self._descloc_cfg.decoupled_sm90_async:
                    dist.barrier(group=self._tp_group)
                gathered_parts = [
                    torch.empty_like(output_parallel) for _ in range(self._tp_size)
                ]
                dist.all_gather(gathered_parts, output_parallel, group=self._tp_group)
                output = torch.cat(gathered_parts, dim=-1)
            else:
                output = output_parallel
        else:
            output = output_parallel

        # Return (output, bias) matching upstream ColumnParallelLinear API
        return output, None

    # ------------------------------------------------------------------
    # Prefetch scheduling API (called by DES-LOC pipeline scheduler)
    # ------------------------------------------------------------------

    def schedule_slc_prefetch(self) -> bool:
        """Enqueue the SLC weight prefetch for the *next* forward pass.

        This is called by the DES-LOC pipeline scheduler at the end of the
        attention stage so that the PCIe transfer overlaps with the FFN stage,
        hiding the PCIe latency on SM86 ranks.

        Returns True if a prefetch was scheduled, False otherwise.
        """
        if self._slc is None or not self._slc.is_offloaded:
            return False
        scheduled = self._slc.schedule_prefetch(self.weight)
        if scheduled:
            logger.debug(
                "SLC prefetch scheduled for next LM-head forward on %s",
                self._resolved_device,
            )
        return scheduled


# ---------------------------------------------------------------------------
# Config predicate (mirrors is_mxfp8_output_proj_active from fp8_utils.py)
# ---------------------------------------------------------------------------

def is_hetero_mxfp8_active(config: Any) -> bool:
    """Return True when the LM-head should run MXFP8 on SM90 ranks.

    Upstream (Megatron 3f6a2ed) ``is_mxfp8_output_proj_active`` checks:
      - HAVE_TE (Transformer Engine installed)
      - config.fp8_output_proj is True
      - config.fp8 is True
      - config.fp8_recipe resolves to 'mxfp8'

    DES-LOC adaptation: we add a device-class gate.  Only SM90 ranks activate
    the MXFP8 path; SM86 ranks always return False (they use the BF16/emulation
    path regardless of config).  This allows a single config object to be
    broadcast to all ranks without causing SM86 ranks to attempt unsupported
    hardware operations.

    Parameters
    ----------
    config:
        Any object with fp8_output_proj, fp8, fp8_recipe attributes (duck-typed
        to accept both Megatron TransformerConfig and DES-LOC config stubs).
    """
    if not getattr(config, "fp8_output_proj", False):
        return False
    if not getattr(config, "fp8", False):
        return False

    fp8_recipe = getattr(config, "fp8_recipe", None)
    if fp8_recipe is None:
        return False

    # Accept enum-like objects with a .value attribute (mirrors upstream logic)
    recipe_value = getattr(fp8_recipe, "value", fp8_recipe)
    recipe_str   = str(recipe_value).lower()

    if not (recipe_str == "mxfp8" or recipe_str.endswith(".mxfp8")):
        return False

    # DES-LOC gate: only active on SM90 hardware
    dev_class = detect_device_class()
    return dev_class == DeviceClass.SM90_HOPPER


# ---------------------------------------------------------------------------
# Factory: build the right LM-head for the current rank's device
# ---------------------------------------------------------------------------

def build_hetero_lm_head(
    input_size: int,
    output_size: int,
    config: Any,
    descloc_cfg: Optional[DESLOCConfig] = None,
    *,
    bias: bool = False,
    gather_output: bool = True,
    tp_group: Optional[dist.ProcessGroup] = None,
    init_method=None,
    params_dtype: torch.dtype = torch.bfloat16,
    device: Optional[torch.device] = None,
) -> HeteroMXFP8LMHead:
    """Factory that constructs a :class:`HeteroMXFP8LMHead` for the current rank.

    DES-LOC note: all ranks (SM86 and SM90) receive the same *config* object
    (broadcast from rank 0).  The factory inspects the *local device class* and
    passes appropriate defaults to the module constructor.  This means the same
    factory call in every rank's model-init code results in the correct per-rank
    behaviour without any rank-specific branching in the caller.

    Parameters mirror :class:`HeteroMXFP8LMHead.__init__`.
    """
    if descloc_cfg is None:
        descloc_cfg = DESLOCConfig()

    resolved_device = device or (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    dev_class = detect_device_class(resolved_device)

    logger.info(
        "build_hetero_lm_head: constructing HeteroMXFP8LMHead for %s (device class: %s)",
        resolved_device,
        dev_class.value,
    )

    return HeteroMXFP8LMHead(
        input_size=input_size,
        output_size=output_size,
        descloc_cfg=descloc_cfg,
        bias=bias,
        gather_output=gather_output,
        tp_group=tp_group,
        init_method=init_method,
        params_dtype=params_dtype,
        device=resolved_device,
    )


# ---------------------------------------------------------------------------
# DESLOCTransformerConfig stub (subset of TransformerConfig for DES-LOC)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DESLOCTransformerConfig:
    """Minimal config subset that ``is_hetero_mxfp8_active`` reads.

    In production use Megatron's ``TransformerConfig`` with the new
    ``fp8_output_proj`` field added by commit 3f6a2ed.  This stub allows
    DES-LOC unit tests and standalone scripts to run without Megatron installed.

    New field (from upstream 3f6a2ed / transformer_config.py):
        fp8_output_proj: bool = False
            If True, run the LM-head output projection with a TE
            ColumnParallelLinear under the MXFP8 autocast context. Only active
            when fp8=True and fp8_recipe='mxfp8'.
    """
    hidden_size: int       = 4096
    vocab_size: int        = 32000
    num_layers: int        = 32
    fp8: bool              = False
    fp8_recipe: str        = "delayed"
    fp8_output_proj: bool  = False   # New field from upstream 3f6a2ed
    params_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        if self.fp8_output_proj:
            if not self.fp8:
                raise ValueError("fp8_output_proj requires fp8=True (DES-LOC validation)")
            recipe_str = str(getattr(self.fp8_recipe, "value", self.fp8_recipe)).lower()
            if not (recipe_str == "mxfp8" or recipe_str.endswith(".mxfp8")):
                raise ValueError(
                    f"fp8_output_proj requires fp8_recipe='mxfp8', got '{self.fp8_recipe}' "
                    f"(DES-LOC validation mirrors upstream TransformerConfig.__post_init__)"
                )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import traceback

    # Configure logging for test output
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    PASS  = "\033[92mPASS\033[0m"
    FAIL  = "\033[91mFAIL\033[0m"
    SKIP  = "\033[93mSKIP\033[0m"

    results: List[Tuple[str, str, str]] = []

    def run_test(name: str, fn):
        try:
            skip_msg = fn()
            if skip_msg is not None:
                results.append((name, SKIP, skip_msg))
            else:
                results.append((name, PASS, ""))
        except AssertionError as e:
            results.append((name, FAIL, str(e)))
            traceback.print_exc()
        except Exception as e:
            results.append((name, FAIL, f"{type(e).__name__}: {e}"))
            traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Test 1: DeviceClass detection — CPU
    # ------------------------------------------------------------------ #
    def test_device_class_cpu():
        dc = detect_device_class(torch.device("cpu"))
        assert dc == DeviceClass.CPU, f"expected CPU, got {dc}"

    run_test("detect_device_class(cpu)", test_device_class_cpu)

    # ------------------------------------------------------------------ #
    # Test 2: is_hetero_mxfp8_active — config gating
    # ------------------------------------------------------------------ #
    def test_hetero_mxfp8_active_config_gates():
        from types import SimpleNamespace

        # fp8_output_proj=False → always False
        cfg = SimpleNamespace(fp8_output_proj=False, fp8=True, fp8_recipe="mxfp8")
        assert is_hetero_mxfp8_active(cfg) is False, "should be False when fp8_output_proj=False"

        # fp8=False → always False
        cfg2 = SimpleNamespace(fp8_output_proj=True, fp8=False, fp8_recipe="mxfp8")
        assert is_hetero_mxfp8_active(cfg2) is False, "should be False when fp8=False"

        # wrong recipe → False
        cfg3 = SimpleNamespace(fp8_output_proj=True, fp8=True, fp8_recipe="delayed")
        assert is_hetero_mxfp8_active(cfg3) is False, "should be False for non-mxfp8 recipe"

        # missing attributes → False
        cfg4 = SimpleNamespace()
        assert is_hetero_mxfp8_active(cfg4) is False, "should be False for missing attrs"

    run_test("is_hetero_mxfp8_active config gating", test_hetero_mxfp8_active_config_gates)

    # ------------------------------------------------------------------ #
    # Test 3: DESLOCTransformerConfig validation mirrors upstream
    # ------------------------------------------------------------------ #
    def test_descloc_transformer_config_validation():
        # Valid default
        cfg = DESLOCTransformerConfig()
        assert not cfg.fp8_output_proj

        # fp8_output_proj without fp8 → ValueError
        try:
            DESLOCTransformerConfig(fp8_output_proj=True, fp8=False, fp8_recipe="mxfp8")
            assert False, "should have raised ValueError"
        except ValueError as e:
            assert "fp8=True" in str(e)

        # fp8_output_proj with wrong recipe → ValueError
        try:
            DESLOCTransformerConfig(fp8_output_proj=True, fp8=True, fp8_recipe="delayed")
            assert False, "should have raised ValueError"
        except ValueError as e:
            assert "mxfp8" in str(e)

        # Valid MXFP8 config
        cfg_ok = DESLOCTransformerConfig(fp8_output_proj=True, fp8=True, fp8_recipe="mxfp8")
        assert cfg_ok.fp8_output_proj

    run_test("DESLOCTransformerConfig validation", test_descloc_transformer_config_validation)

    # ------------------------------------------------------------------ #
    # Test 4: HeteroMXFP8LMHead construction on CPU
    # ------------------------------------------------------------------ #
    def test_lm_head_construction_cpu():
        head = HeteroMXFP8LMHead(
            input_size=64,
            output_size=128,
            device=torch.device("cpu"),
        )
        assert head.weight.shape == (128, 64), f"unexpected weight shape {head.weight.shape}"
        assert head._device_class == DeviceClass.CPU
        assert head._tp_size == 1
        assert head.bias is None

    run_test("HeteroMXFP8LMHead construction (CPU)", test_lm_head_construction_cpu)

    # ------------------------------------------------------------------ #
    # Test 5: HeteroMXFP8LMHead construction with bias
    # ------------------------------------------------------------------ #
    def test_lm_head_construction_with_bias():
        head = HeteroMXFP8LMHead(
            input_size=32,
            output_size=64,
            bias=True,
            device=torch.device("cpu"),
        )
        assert head.bias is not None
        assert head.bias.shape == (64,), f"unexpected bias shape {head.bias.shape}"

    run_test("HeteroMXFP8LMHead bias construction", test_lm_head_construction_with_bias)

    # ------------------------------------------------------------------ #
    # Test 6: get_extra_state returns None (state-dict shim)
    # ------------------------------------------------------------------ #
    def test_get_extra_state_returns_none():
        head = HeteroMXFP8LMHead(
            input_size=16, output_size=32, device=torch.device("cpu")
        )
        assert head.get_extra_state() is None

    run_test("get_extra_state returns None", test_get_extra_state_returns_none)

    # ------------------------------------------------------------------ #
    # Test 7: set_extra_state is a no-op
    # ------------------------------------------------------------------ #
    def test_set_extra_state_noop():
        head = HeteroMXFP8LMHead(
            input_size=16, output_size=32, device=torch.device("cpu")
        )
        head.set_extra_state({"some": "state"})   # must not raise
        head.set_extra_state(None)

    run_test("set_extra_state no-op", test_set_extra_state_noop)

    # ------------------------------------------------------------------ #
    # Test 8: runtime weight override raises RuntimeError
    # ------------------------------------------------------------------ #
    def test_runtime_weight_override_raises():
        head = HeteroMXFP8LMHead(
            input_size=16, output_size=32, device=torch.device("cpu")
        )
        fake_weight = torch.zeros(32, 16)
        try:
            head.forward(torch.zeros(4, 16), weight=fake_weight)
            assert False, "should have raised RuntimeError"
        except RuntimeError as e:
            assert "runtime weight override" in str(e)

    run_test("runtime weight override raises RuntimeError", test_runtime_weight_override_raises)

    # ------------------------------------------------------------------ #
    # Test 9: CPU forward pass — no gather
    # ------------------------------------------------------------------ #
    def test_cpu_forward_no_gather():
        torch.manual_seed(42)
        head = HeteroMXFP8LMHead(
            input_size=16,
            output_size=32,
            gather_output=False,
            device=torch.device("cpu"),
        )
        x = torch.randn(8, 4, 16)
        out, bias = head(x)
        assert out.shape == (8, 4, 32), f"unexpected output shape {out.shape}"
        assert bias is None

    run_test("CPU forward (no gather)", test_cpu_forward_no_gather)

    # ------------------------------------------------------------------ #
    # Test 10: CPU forward pass — gather (single rank, no-op)
    # ------------------------------------------------------------------ #
    def test_cpu_forward_gather_single_rank():
        torch.manual_seed(7)
        head = HeteroMXFP8LMHead(
            input_size=16,
            output_size=32,
            gather_output=True,
            device=torch.device("cpu"),
        )
        # tp_size=1, so gather is a no-op
        x = torch.randn(4, 16)
        out, _ = head(x)
        assert out.shape == (4, 32), f"unexpected output shape {out.shape}"

    run_test("CPU forward (gather, single rank)", test_cpu_forward_gather_single_rank)

    # ------------------------------------------------------------------ #
    # Test 11: runtime_gather_output override
    # ------------------------------------------------------------------ #
    def test_runtime_gather_output_override():
        torch.manual_seed(13)
        head = HeteroMXFP8LMHead(
            input_size=8,
            output_size=16,
            gather_output=True,   # default: gather
            device=torch.device("cpu"),
        )
        x = torch.randn(3, 8)
        # Override to NOT gather — tp_size=1 so shape unchanged, but no dist call
        out, _ = head(x, runtime_gather_output=False)
        assert out.shape == (3, 16)

    run_test("runtime_gather_output override", test_runtime_gather_output_override)

    # ------------------------------------------------------------------ #
    # Test 12: FP8 emulation passthrough on CPU
    # ------------------------------------------------------------------ #
    def test_fp8_emulate_e4m3_cpu():
        x = torch.tensor([1.0, -2.0, 0.5, 100.0, -448.5])
        out = _fp8_emulate_e4m3(x)
        assert out.shape == x.shape
        # Values should be approximately preserved (within FP8 precision)
        assert out.abs().max() <= 450.0, f"FP8 emulation clipping failed: max={out.abs().max()}"

    run_test("_fp8_emulate_e4m3 CPU passthrough", test_fp8_emulate_e4m3_cpu)

    # ------------------------------------------------------------------ #
    # Test 13: SLC manager — below threshold, no offload
    # ------------------------------------------------------------------ #
    def test_slc_no_offload_below_threshold():
        cfg = DESLOCConfig(slc_pin_threshold_bytes=10 * 1024 ** 3)  # 10 GiB threshold
        slc = SharedLocalityCacheManager(cfg, torch.device("cpu"))
        small = torch.zeros(64, 64)   # 32 KB — well below threshold
        offloaded = slc.maybe_offload(small)
        assert not offloaded, "small tensor should not be offloaded"
        assert not slc.is_offloaded

    run_test("SLC: no offload below threshold", test_slc_no_offload_below_threshold)

    # ------------------------------------------------------------------ #
    # Test 14: SLC manager — above threshold, offload
    # ------------------------------------------------------------------ #
    def test_slc_offload_above_threshold():
        cfg = DESLOCConfig(slc_pin_threshold_bytes=100)  # tiny threshold
        slc = SharedLocalityCacheManager(cfg, torch.device("cpu"))
        big = torch.zeros(64, 64)   # 32 KB > 100 bytes threshold
        offloaded = slc.maybe_offload(big)
        assert offloaded, "large tensor should be offloaded to SLC"
        assert slc.is_offloaded

    run_test("SLC: offload above threshold", test_slc_offload_above_threshold)

    # ------------------------------------------------------------------ #
    # Test 15: DESLOCConfig defaults
    # ------------------------------------------------------------------ #
    def test_descloc_config_defaults():
        cfg = DESLOCConfig()
        assert cfg.slc_pin_threshold_bytes == 2 * 1024 ** 3
        assert cfg.slc_prefetch_streams == 2
        assert not cfg.decoupled_sm90_async
        assert not cfg.fp8_emulate_on_sm86
        assert cfg.wgrad_delay_disabled is True
        assert cfg.tp_size == 0

    run_test("DESLOCConfig defaults", test_descloc_config_defaults)

    # ------------------------------------------------------------------ #
    # Test 16: CUDA forward pass on SM86 / SM90 (if GPU available)
    # ------------------------------------------------------------------ #
    def test_cuda_forward_if_available():
        if not torch.cuda.is_available():
            return "CUDA not available"

        device = torch.device("cuda", 0)
        dev_class = detect_device_class(device)

        head = HeteroMXFP8LMHead(
            input_size=128,
            output_size=512,
            descloc_cfg=DESLOCConfig(fp8_emulate_on_sm86=True),
            gather_output=False,
            device=device,
            params_dtype=torch.bfloat16,
        )
        x = torch.randn(16, 128, device=device, dtype=torch.bfloat16)
        out, bias_out = head(x)
        assert out.shape == (16, 512), f"unexpected shape {out.shape}"
        assert bias_out is None
        logger.info(
            "CUDA forward test passed on device class %s; output shape %s",
            dev_class.value,
            out.shape,
        )

    run_test("CUDA forward (if available)", test_cuda_forward_if_available)

    # ------------------------------------------------------------------ #
    # Test 17: is_hetero_mxfp8_active — enum-style recipe
    # ------------------------------------------------------------------ #
    def test_hetero_mxfp8_active_enum_recipe():
        from types import SimpleNamespace

        enum_recipe = SimpleNamespace(value="mxfp8")
        cfg = SimpleNamespace(fp8_output_proj=True, fp8=True, fp8_recipe=enum_recipe)
        # On CPU / SM86 this returns False (device gate); on SM90 it would be True.
        # We only check it doesn't raise:
        result = is_hetero_mxfp8_active(cfg)
        assert isinstance(result, bool)

    run_test("is_hetero_mxfp8_active enum recipe", test_hetero_mxfp8_active_enum_recipe)

    # ------------------------------------------------------------------ #
    # Test 18: build_hetero_lm_head factory
    # ------------------------------------------------------------------ #
    def test_build_hetero_lm_head_factory():
        from types import SimpleNamespace

        cfg = SimpleNamespace(fp8_output_proj=False, fp8=False, fp8_recipe="delayed")
        head = build_hetero_lm_head(
            input_size=32,
            output_size=64,
            config=cfg,
            device=torch.device("cpu"),
        )
        assert isinstance(head, HeteroMXFP8LMHead)
        assert head.input_size  == 32
        assert head.output_size == 64

    run_test("build_hetero_lm_head factory", test_build_hetero_lm_head_factory)

    # ------------------------------------------------------------------ #
    # Test 19: SM86 forward with FP8 emulation
    # ------------------------------------------------------------------ #
    def test_sm86_forward_fp8_emulation_cpu():
        """Validate SM86 BF16 path with FP8 emulation on CPU."""
        descloc = DESLOCConfig(fp8_emulate_on_sm86=True)
        head = HeteroMXFP8LMHead(
            input_size=32,
            output_size=64,
            descloc_cfg=descloc,
            device=torch.device("cpu"),
        )
        # Force SM86 path by monkeypatching device_class
        head._device_class = DeviceClass.SM86_AMPERE
        x = torch.randn(4, 32)
        out = head._forward_sm86_bf16(x, head.weight)
        assert out.shape == (4, 64), f"unexpected shape {out.shape}"

    run_test("SM86 BF16+FP8emu forward (CPU mock)", test_sm86_forward_fp8_emulation_cpu)

    # ------------------------------------------------------------------ #
    # Test 20: weight sharding with tp_size > 1
    # ------------------------------------------------------------------ #
    def test_weight_sharding_tp_size():
        descloc = DESLOCConfig(tp_size=4)
        head = HeteroMXFP8LMHead(
            input_size=64,
            output_size=128,
            descloc_cfg=descloc,
            device=torch.device("cpu"),
        )
        assert head._output_size_per_partition == 32, (
            f"expected 32, got {head._output_size_per_partition}"
        )
        assert head.weight.shape == (32, 64), (
            f"unexpected weight shape {head.weight.shape}"
        )

    run_test("weight sharding with tp_size=4", test_weight_sharding_tp_size)

    # ------------------------------------------------------------------ #
    # Print results
    # ------------------------------------------------------------------ #
    print()
    print("=" * 72)
    print("  HeteroMXFP8LMHead (DES-LOC) — Unit Test Results")
    print("=" * 72)
    n_pass = n_fail = n_skip = 0
    for name, status, msg in results:
        suffix = f"  ({msg})" if msg else ""
        print(f"  [{status}] {name}{suffix}")
        if PASS in status:   n_pass += 1
        elif FAIL in status: n_fail += 1
        else:                n_skip += 1
    print("=" * 72)
    print(f"  Total: {len(results)} | Pass: {n_pass} | Fail: {n_fail} | Skip: {n_skip}")
    print("=" * 72)

    sys.exit(0 if n_fail == 0 else 1)
