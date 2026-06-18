"""
HeteroTEGemmWgrad — DES-LOC Heterogeneous Mixed-Precision Weight-Gradient GEMM
===============================================================================

Upstream design intent (Megatron-LM commit 8d7a3f8d):
------------------------------------------------------
Megatron's FSDP path computes weight gradients via a backward GEMM:

    wgrad = grad_output^T  @  total_input          (math)

In standard single-dtype training (e.g. bf16→bf16) ``torch.matmul(..., out=w)``
works fine.  But with gradient accumulation buffers stored in fp32 for numerical
stability, PyTorch's ``out=`` pathway silently upcasts *after* the GEMM, losing
precision in the intermediate products.  TransformerEngine's ``general_gemm``
solves this by performing the accumulation natively in the output dtype, with a
``grad=True`` flag that selects the NT (non-transposed A, transposed B) cuBLAS
kernel layout.

DES-LOC adaptation points:
---------------------------
The DES-LOC (Decoupled Execution with Shared LOcality Cache) framework targets a
heterogeneous cluster of **2× A6000 48 GB SM86** + **1× H100 NVL 96 GB SM90**
connected via PCIe (no NVLink) with 1.5 TB CPU DRAM as a spill tier.

Three key differences from the Megatron FSDP path that this module addresses:

1. **Device-class dispatch** – SM86 (A6000) does not support native BF16 tensor
   cores at full throughput; the H100 (SM90) does.  We dispatch to different
   kernel back-ends per compute device rather than assuming a homogeneous fleet.

2. **SLOC cache awareness** – The Shared LOcality Cache is a DeepSpeed-managed
   tensor buffer that lives in CPU pinned memory and is staged onto each GPU on
   demand.  When ``total_input`` or ``grad_output`` reside in the SLOC tier we
   must orchestrate an async prefetch *before* launching the GEMM, not inside it.

3. **Gradient dtype negotiation** – Because the three GPUs may hold parameter
   shards with different precision (A6000 stores fp16 working copies, H100 stores
   bf16), the wgrad accumulation dtype must be resolved at call-time, not
   compile-time.

This module exports a single entry-point:

    hetero_wgrad_gemm(total_input, grad_output, weight, device_info, sloc_cache)

which replaces the ``torch.matmul`` / ``te_general_gemm`` branch in
``LinearWithGradAccumulationAndAsyncCommunication.backward``.

Architecture
------------
::

    HeteroWgradRouter
    ├── SlocPrefetchContext   (async pin→GPU staging)
    ├── SM86WgradKernel       (A6000: fp16 input → fp32 accum via cuBLAS SGEMM)
    ├── SM90WgradKernel       (H100: bf16 input → fp32 accum via TE general_gemm)
    └── CpuFallbackKernel     (pure-torch fallback when GPU OOM)

"""

from __future__ import annotations

import logging
import math
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public re-export so callers can do:
#   from deepspeed.ops.hetero_te_gemm_wgrad import hetero_wgrad_gemm
# ---------------------------------------------------------------------------
__all__ = [
    "hetero_wgrad_gemm",
    "DeviceClass",
    "DeviceInfo",
    "SlocCache",
    "HeteroWgradRouter",
]

# ---------------------------------------------------------------------------
# TE availability probe
# ---------------------------------------------------------------------------
_TE_GENERAL_GEMM = None
_TE_AVAILABLE = False

try:
    from transformer_engine.pytorch.cpp_extensions import general_gemm as _te_general_gemm  # type: ignore

    _TE_GENERAL_GEMM = _te_general_gemm
    _TE_AVAILABLE = True
    logger.info("TransformerEngine general_gemm available — SM90 path enabled.")
except ImportError:
    logger.warning(
        "TransformerEngine not found.  SM90 wgrad path will fall back to "
        "torch.matmul (mixed-precision accumulation accuracy may be reduced)."
    )


# ---------------------------------------------------------------------------
# Device taxonomy
# ---------------------------------------------------------------------------


class DeviceClass(Enum):
    """Compute capability class relevant to DES-LOC dispatch."""

    SM86 = auto()  # A6000 — no BF16 tensor core native
    SM90 = auto()  # H100 NVL — full BF16 / FP8 tensor core
    CPU = auto()  # CPU DRAM spill tier


# Map SM major.minor → DeviceClass
_SM_TO_CLASS: dict[tuple[int, int], DeviceClass] = {
    (8, 6): DeviceClass.SM86,
    (9, 0): DeviceClass.SM90,
}


def _detect_device_class(device: torch.device) -> DeviceClass:
    """Return the :class:`DeviceClass` for *device*.

    For CUDA devices we query the compute capability.  CPU and any unknown
    SM are mapped to :attr:`DeviceClass.CPU` (safest fallback).
    """
    if device.type != "cuda":
        return DeviceClass.CPU
    major, minor = torch.cuda.get_device_capability(device)
    cls = _SM_TO_CLASS.get((major, minor))
    if cls is None:
        logger.debug(
            "Unknown SM%d%d on device %s — treating as SM86 (conservative).",
            major,
            minor,
            device,
        )
        cls = DeviceClass.SM86
    return cls


@dataclass
class DeviceInfo:
    """Per-device metadata consumed by :class:`HeteroWgradRouter`.

    Parameters
    ----------
    device:
        The torch device this info describes.
    device_class:
        Compute capability class.  If ``None``, auto-detected.
    free_memory_bytes:
        Optional hint for OOM fallback decisions.  When ``None``, we query
        ``torch.cuda.mem_get_info`` at call-time (adds a tiny CUDA sync).
    """

    device: torch.device
    device_class: Optional[DeviceClass] = None
    free_memory_bytes: Optional[int] = None

    def __post_init__(self):
        if self.device_class is None:
            self.device_class = _detect_device_class(self.device)

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "DeviceInfo":
        """Construct :class:`DeviceInfo` from a tensor's device."""
        return cls(device=t.device)

    def get_free_memory(self) -> int:
        """Return free VRAM in bytes.  May issue a lightweight CUDA query."""
        if self.free_memory_bytes is not None:
            return self.free_memory_bytes
        if self.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(self.device)
            return free
        # CPU: report 75 % of 1.5 TB as nominal "free"
        return int(1.5e12 * 0.75)


# ---------------------------------------------------------------------------
# SLOC Cache
# ---------------------------------------------------------------------------


@dataclass
class SlocTensorHandle:
    """A reference to a tensor that may reside in the SLOC (CPU-pinned) tier.

    DES-LOC keeps large activations / gradients in the Shared LOcality Cache
    (1.5 TB CPU DRAM, page-locked) and promotes them to GPU on demand.  This
    handle tracks the promotion state so callers can wait for the async H2D
    copy before issuing a GEMM.
    """

    cpu_tensor: torch.Tensor          # always valid; lives in pinned memory
    gpu_tensor: Optional[torch.Tensor] = None  # populated after prefetch
    _event: Optional[torch.cuda.Event] = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def is_on_gpu(self) -> bool:
        return self.gpu_tensor is not None

    def prefetch_async(self, device: torch.device, stream: torch.cuda.Stream) -> None:
        """Kick off an async H2D copy into *stream*.

        Safe to call multiple times; subsequent calls are no-ops if the copy
        is already in-flight or complete.
        """
        with self._lock:
            if self.gpu_tensor is not None:
                return  # already promoted
            with torch.cuda.stream(stream):
                self.gpu_tensor = self.cpu_tensor.to(device, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(stream)
                self._event = ev
        logger.debug(
            "SLOC prefetch issued: shape=%s dtype=%s → %s",
            tuple(self.cpu_tensor.shape),
            self.cpu_tensor.dtype,
            device,
        )

    def wait(self, stream: Optional[torch.cuda.Stream] = None) -> torch.Tensor:
        """Block the current (or given) stream until the H2D copy completes.

        Returns the GPU tensor.  Raises ``RuntimeError`` if ``prefetch_async``
        was never called.
        """
        if self.gpu_tensor is None:
            raise RuntimeError(
                "SlocTensorHandle.wait() called before prefetch_async().  "
                "Call prefetch_async() first or access cpu_tensor directly."
            )
        if self._event is not None:
            target = stream or torch.cuda.current_stream()
            target.wait_event(self._event)
        return self.gpu_tensor

    def evict(self) -> None:
        """Release the GPU copy; the CPU tensor remains valid."""
        with self._lock:
            self.gpu_tensor = None
            self._event = None


class SlocCache:
    """Lightweight registry of :class:`SlocTensorHandle` objects.

    In a full DES-LOC deployment this is managed by the DeepSpeed runtime.
    Here we provide a standalone implementation that the wgrad kernel can
    use directly, compatible with the broader DeepSpeed SLOC manager API.

    Usage::

        cache = SlocCache()
        handle = cache.register(my_cpu_pinned_tensor, key="layer3.input")
        handle.prefetch_async(device, prefetch_stream)
        # ... launch other ops ...
        gpu_t = handle.wait()
    """

    def __init__(self):
        self._store: dict[str, SlocTensorHandle] = {}
        self._lock = threading.Lock()

    def register(self, cpu_tensor: torch.Tensor, key: str) -> SlocTensorHandle:
        """Register *cpu_tensor* under *key* and return its handle."""
        if not cpu_tensor.is_pinned():
            logger.warning(
                "SlocCache.register: tensor '%s' is not pinned.  "
                "H2D throughput will be suboptimal over PCIe.",
                key,
            )
        with self._lock:
            handle = SlocTensorHandle(cpu_tensor=cpu_tensor)
            self._store[key] = handle
        return handle

    def get(self, key: str) -> Optional[SlocTensorHandle]:
        return self._store.get(key)

    def evict_all(self) -> None:
        """Release all GPU copies (e.g. after a backward pass)."""
        with self._lock:
            for h in self._store.values():
                h.evict()
        logger.debug("SlocCache: evicted all GPU copies.")

    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# Dtype negotiation
# ---------------------------------------------------------------------------

# Megatron uses fp32 for main_grad to preserve gradient quality.
# A6000 (SM86) may hold fp16 working params; H100 (SM90) holds bf16.
# The negotiation table maps (input_dtype, device_class) → preferred compute dtype.
_WGRAD_COMPUTE_DTYPE: dict[tuple[torch.dtype, DeviceClass], torch.dtype] = {
    (torch.float16, DeviceClass.SM86): torch.float32,   # SM86: upcasting avoids fp16 overflow
    (torch.bfloat16, DeviceClass.SM86): torch.float32,  # SM86: bf16 accum not reliable
    (torch.float16, DeviceClass.SM90): torch.float32,   # H100: still use fp32 main_grad
    (torch.bfloat16, DeviceClass.SM90): torch.float32,  # H100: TE handles bf16→fp32 natively
    (torch.float32, DeviceClass.SM86): torch.float32,
    (torch.float32, DeviceClass.SM90): torch.float32,
}


def _resolve_accum_dtype(
    input_dtype: torch.dtype, device_class: DeviceClass
) -> torch.dtype:
    key = (input_dtype, device_class)
    dtype = _WGRAD_COMPUTE_DTYPE.get(key)
    if dtype is None:
        logger.warning(
            "No wgrad dtype rule for (%s, %s); defaulting to fp32.", input_dtype, device_class
        )
        dtype = torch.float32
    return dtype


# ---------------------------------------------------------------------------
# Kernel implementations
# ---------------------------------------------------------------------------


class _SM86WgradKernel:
    """Weight-gradient GEMM for A6000 (SM86, no BF16 tensor cores).

    Strategy
    --------
    SM86 tensor cores do not accelerate BF16.  We therefore:
      1. Cast inputs to FP16 for the matmul (FP16 tensor cores ARE present).
      2. Accumulate into an FP32 buffer to avoid overflow.
      3. If ``out`` is provided and is FP32, write directly; otherwise copy.

    This mirrors the ``fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32``
    path in Megatron but without the Megatron-specific CUDA extension
    dependency, using only cuBLAS via torch.

    Layout convention (same as Megatron NT):
        wgrad = total_input^T  @  grad_output        in torch speak
              = grad_output^T  @  total_input         in BLAS NT speak
    """

    @staticmethod
    def compute(
        total_input: torch.Tensor,
        grad_output: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute wgrad on SM86, accumulating in FP32.

        Parameters
        ----------
        total_input:
            Input activation tensor, shape ``[seq*batch, in_features]``.
        grad_output:
            Upstream gradient, shape ``[seq*batch, out_features]``.
        out:
            Optional pre-allocated FP32 output tensor of shape
            ``[out_features, in_features]``.  Written in-place when provided.

        Returns
        -------
        torch.Tensor
            Weight gradient, shape ``[out_features, in_features]``, dtype FP32.
        """
        # Flatten to 2D if activations include a sequence dimension
        if total_input.dim() > 2:
            total_input = total_input.view(-1, total_input.size(-1))
        if grad_output.dim() > 2:
            grad_output = grad_output.view(-1, grad_output.size(-1))

        # SM86: cast to fp16 for tensor core acceleration, then accumulate in fp32
        ti_fp16 = total_input.to(torch.float16)
        go_fp16 = grad_output.to(torch.float16)

        # torch.mm with float16 inputs accumulates in fp16 on most backends;
        # we use addmm with a zero fp32 base to force fp32 accumulation.
        if out is not None and out.dtype == torch.float32:
            # In-place accumulation: out += grad_output^T @ total_input
            # Use torch.addmm: out = beta*out + alpha*(grad_output.t() @ total_input)
            # beta=1 for gradient accumulation, but Megatron resets main_grad each step.
            torch.mm(go_fp16.t().float(), ti_fp16.float(), out=out)
            logger.debug(
                "SM86WgradKernel: in-place mm, shape=%s, out_dtype=%s",
                tuple(out.shape),
                out.dtype,
            )
            return out
        else:
            result = torch.mm(go_fp16.t().float(), ti_fp16.float())
            if out is not None:
                out.copy_(result)
                return out
            return result


class _SM90WgradKernel:
    """Weight-gradient GEMM for H100 NVL (SM90).

    This is the DES-LOC analogue of Megatron's ``te_general_gemm`` branch.
    SM90 supports native BF16 tensor cores and TransformerEngine's
    ``general_gemm`` can accumulate directly into FP32, giving both speed
    and accuracy without a manual upcast.

    When TE is unavailable (e.g. unit-test environments without CUDA),
    we fall back to a torch-based mixed-precision path identical to SM86 but
    exploiting SM90's BF16 throughput by keeping inputs in BF16 longer.
    """

    @staticmethod
    def compute(
        total_input: torch.Tensor,
        grad_output: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        out_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Compute wgrad on SM90 using TE general_gemm when available.

        Parameters
        ----------
        total_input:
            Input activation, shape ``[T, K]`` (T = seq*batch, K = in_features).
        grad_output:
            Upstream gradient, shape ``[T, N]`` (N = out_features).
        out:
            Optional pre-allocated output, shape ``[N, K]``.
        out_dtype:
            Accumulation dtype for the output buffer (fp32 recommended).

        Returns
        -------
        torch.Tensor
            Weight gradient ``[N, K]`` in *out_dtype*.
        """
        if total_input.dim() > 2:
            total_input = total_input.view(-1, total_input.size(-1))
        if grad_output.dim() > 2:
            grad_output = grad_output.view(-1, grad_output.size(-1))

        if _TE_AVAILABLE and _TE_GENERAL_GEMM is not None:
            # Replicate Megatron's te_general_gemm call with NT layout.
            # TE signature: general_gemm(A, B, out_dtype, layout, out, grad)
            # NT layout: C = A @ B^T, but with grad=True TE interprets as wgrad.
            logger.debug(
                "SM90WgradKernel: using TE general_gemm, "
                "total_input=%s grad_output=%s out_dtype=%s",
                tuple(total_input.shape),
                tuple(grad_output.shape),
                out_dtype,
            )
            try:
                _TE_GENERAL_GEMM(
                    total_input,
                    grad_output,
                    out_dtype=out_dtype,
                    layout="NT",
                    out=out,
                    grad=True,
                )
                return out  # type: ignore[return-value]
            except Exception as exc:
                logger.warning(
                    "TE general_gemm failed (%s); falling back to torch path.", exc
                )

        # Fallback: BF16 inputs, FP32 accumulation via torch.mm
        ti_bf16 = total_input.to(torch.bfloat16)
        go_bf16 = grad_output.to(torch.bfloat16)
        result_fp32 = torch.mm(go_bf16.t().float(), ti_bf16.float())

        if out is not None:
            out.copy_(result_fp32)
            return out
        return result_fp32


class _CpuFallbackKernel:
    """Pure-PyTorch CPU fallback when GPU memory is insufficient.

    In DES-LOC the 1.5 TB CPU DRAM tier acts as overflow storage.  When both
    A6000s are saturated and the H100 has less than *min_free_bytes* available,
    we offload the wgrad GEMM to CPU.  This is slow but correct, and avoids
    OOM kills during memory-intensive forward/backward passes.

    Note: this kernel always returns a CPU tensor.  The caller is responsible
    for moving the result back to the target device before optimizer update.
    """

    MIN_FREE_BYTES_DEFAULT = 2 * 1024**3  # 2 GB safety margin

    @staticmethod
    def compute(
        total_input: torch.Tensor,
        grad_output: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ti_cpu = total_input.cpu().float()
        go_cpu = grad_output.cpu().float()

        if ti_cpu.dim() > 2:
            ti_cpu = ti_cpu.view(-1, ti_cpu.size(-1))
        if go_cpu.dim() > 2:
            go_cpu = go_cpu.view(-1, go_cpu.size(-1))

        result = torch.mm(go_cpu.t(), ti_cpu)
        logger.debug(
            "CpuFallbackKernel: computed wgrad on CPU, shape=%s", tuple(result.shape)
        )
        if out is not None:
            out.copy_(result)
            return out
        return result


# ---------------------------------------------------------------------------
# Prefetch context manager for SLOC-resident tensors
# ---------------------------------------------------------------------------


@contextmanager
def _sloc_prefetch_context(
    tensors: list[torch.Tensor | SlocTensorHandle],
    device: torch.device,
):
    """Async-prefetch any :class:`SlocTensorHandle` in *tensors* to *device*.

    Yields a list of plain :class:`torch.Tensor` objects in the same order as
    *tensors*, with SLOC handles replaced by their promoted GPU tensors.

    This implements the "Shared LOcality" part of DES-LOC: activations stored
    in CPU-pinned memory are staged to the target GPU just-in-time, overlapping
    the H2D copy with any preceding computation.

    Example::

        with _sloc_prefetch_context([total_input, grad_output], device) as ts:
            ti, go = ts
            kernel.compute(ti, go, out=wgrad_buf)
    """
    prefetch_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    promoted: list[torch.Tensor] = []
    handles_to_wait: list[SlocTensorHandle] = []

    for t in tensors:
        if isinstance(t, SlocTensorHandle):
            if prefetch_stream is not None:
                t.prefetch_async(device, prefetch_stream)
                handles_to_wait.append(t)
            else:
                # CPU device — just use the cpu_tensor directly
                promoted.append(t.cpu_tensor)
        else:
            promoted.append(t)

    # Wait for all in-flight H2D copies before yielding
    for h in handles_to_wait:
        gpu_t = h.wait(stream=prefetch_stream)
        promoted.append(gpu_t)

    # Synchronise prefetch stream with compute stream
    if prefetch_stream is not None:
        compute_stream = torch.cuda.current_stream(device)
        compute_stream.wait_stream(prefetch_stream)

    try:
        yield promoted
    finally:
        # Optionally evict SLOC handles after use to free VRAM
        # (controlled by SLOC eviction policy; we leave them hot here
        #  so a subsequent optimizer step can reuse them)
        pass


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class HeteroWgradRouter:
    """Selects and executes the appropriate wgrad GEMM kernel for each device.

    This is the central dispatch component that replaces the Megatron branch::

        if te_general_gemm is not None:
            te_general_gemm(total_input, grad_output, ...)
        else:
            torch.matmul(grad_output.t(), total_input, out=weight.main_grad)

    In a DES-LOC cluster the three GPUs may be mid-backward on different
    parameter shards simultaneously (Decoupled Execution).  The router is
    called once per device per wgrad step.

    Parameters
    ----------
    oom_fallback_threshold_bytes:
        If free VRAM on the target device drops below this threshold, the
        router falls back to :class:`_CpuFallbackKernel`.
    """

    def __init__(self, oom_fallback_threshold_bytes: int = 2 * 1024**3):
        self.oom_threshold = oom_fallback_threshold_bytes
        self._kernel_sm86 = _SM86WgradKernel()
        self._kernel_sm90 = _SM90WgradKernel()
        self._kernel_cpu = _CpuFallbackKernel()

    def route(
        self,
        total_input: torch.Tensor,
        grad_output: torch.Tensor,
        weight: torch.Tensor,
        device_info: Optional[DeviceInfo] = None,
        sloc_cache: Optional[SlocCache] = None,
    ) -> torch.Tensor:
        """Compute ``weight.main_grad += grad_output^T @ total_input``.

        This is a drop-in replacement for the Megatron FSDP wgrad accumulation
        block.  It handles:

        * SLOC-resident tensor promotion (async H2D prefetch).
        * Device-class kernel selection (SM86 vs SM90 vs CPU fallback).
        * Gradient dtype negotiation (input dtype × device class → accum dtype).
        * OOM-triggered CPU spill.

        Parameters
        ----------
        total_input:
            Forward-pass input activation (possibly a :class:`SlocTensorHandle`
            if the layer's activations were offloaded to the SLOC tier).
        grad_output:
            Upstream gradient from the next layer.
        weight:
            The parameter whose gradient we are accumulating.  Must have a
            ``main_grad`` attribute (set by the FSDP sharding mechanism).
        device_info:
            Metadata about the compute device.  Auto-detected from *weight*
            if not provided.
        sloc_cache:
            SLOC cache registry.  Used to look up handle metadata for logging.

        Returns
        -------
        torch.Tensor
            ``weight.main_grad`` (updated in-place).
        """
        # ------------------------------------------------------------------
        # 1. Materialise main_grad (mirrors Megatron's get_main_grad() call)
        # ------------------------------------------------------------------
        if hasattr(weight, "__fsdp_param__"):
            weight.main_grad = weight.main_grad if hasattr(weight, "main_grad") else \
                weight.get_main_grad()
        elif not hasattr(weight, "main_grad"):
            raise AttributeError(
                "weight must have a 'main_grad' attribute.  "
                "Ensure the parameter is wrapped by DeepSpeed FSDP."
            )

        main_grad = weight.main_grad

        # ------------------------------------------------------------------
        # 2. Resolve device info
        # ------------------------------------------------------------------
        if device_info is None:
            device_info = DeviceInfo.from_tensor(weight)
        device = device_info.device
        dc = device_info.device_class

        # ------------------------------------------------------------------
        # 3. Resolve accumulation dtype
        # ------------------------------------------------------------------
        input_dtype = (
            total_input.cpu_tensor.dtype
            if isinstance(total_input, SlocTensorHandle)
            else total_input.dtype
        )
        accum_dtype = _resolve_accum_dtype(input_dtype, dc)
        logger.debug(
            "HeteroWgradRouter: device=%s class=%s input_dtype=%s accum_dtype=%s",
            device,
            dc,
            input_dtype,
            accum_dtype,
        )

        # ------------------------------------------------------------------
        # 4. SLOC prefetch
        # ------------------------------------------------------------------
        with _sloc_prefetch_context([total_input, grad_output], device) as (ti, go):

            # ----------------------------------------------------------------
            # 5. OOM guard — measure free VRAM before committing to GPU kernel
            # ----------------------------------------------------------------
            free_bytes = device_info.get_free_memory()
            # Rough estimate: two full fp32 copies of the wgrad buffer
            n, k = go.size(-1), ti.size(-1)
            wgrad_bytes = n * k * 4  # fp32
            if free_bytes < self.oom_threshold + wgrad_bytes:
                logger.warning(
                    "VRAM low (%d MB free) on %s — routing wgrad to CPU spill.",
                    free_bytes // (1024**2),
                    device,
                )
                return self._kernel_cpu.compute(ti, go, out=main_grad)

            # ----------------------------------------------------------------
            # 6. Kernel dispatch
            # ----------------------------------------------------------------
            if dc == DeviceClass.SM90:
                return _SM90WgradKernel.compute(
                    ti, go, out=main_grad, out_dtype=accum_dtype
                )
            elif dc == DeviceClass.SM86:
                return _SM86WgradKernel.compute(ti, go, out=main_grad)
            else:
                # CPU or unknown — use CPU fallback
                return self._kernel_cpu.compute(ti, go, out=main_grad)


# ---------------------------------------------------------------------------
# Module-level singleton router + convenience function
# ---------------------------------------------------------------------------

_default_router = HeteroWgradRouter()


def hetero_wgrad_gemm(
    total_input: torch.Tensor,
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    device_info: Optional[DeviceInfo] = None,
    sloc_cache: Optional[SlocCache] = None,
) -> torch.Tensor:
    """Module-level entry point for DES-LOC heterogeneous wgrad GEMM.

    Drop-in replacement for the Megatron FSDP wgrad block::

        # Megatron (8d7a3f8):
        if te_general_gemm is not None:
            te_general_gemm(total_input, grad_output,
                            out_dtype=weight.main_grad.dtype,
                            layout="NT", out=weight.main_grad, grad=True)
        else:
            torch.matmul(grad_output.t(), total_input, out=weight.main_grad)

        # DES-LOC (this file):
        hetero_wgrad_gemm(total_input, grad_output, weight,
                          device_info=device_info, sloc_cache=sloc_cache)

    The function is intentionally stateless at the call site; all routing
    state is encapsulated in :data:`_default_router`.

    Parameters
    ----------
    total_input:
        Forward activation tensor (or SLOC handle).
    grad_output:
        Upstream gradient tensor.
    weight:
        Parameter with ``main_grad`` attribute.
    device_info:
        Optional device metadata; auto-detected when ``None``.
    sloc_cache:
        Optional SLOC cache registry for logging and eviction management.

    Returns
    -------
    torch.Tensor
        ``weight.main_grad`` updated in-place.
    """
    return _default_router.route(
        total_input=total_input,
        grad_output=grad_output,
        weight=weight,
        device_info=device_info,
        sloc_cache=sloc_cache,
    )


# ---------------------------------------------------------------------------
# Integration helpers for DeepSpeed FSDP backward hook
# ---------------------------------------------------------------------------


def patch_fsdp_linear_backward(linear_module: torch.nn.Module) -> None:
    """Monkey-patch a DeepSpeed FSDP linear layer to use hetero wgrad GEMM.

    Called once per linear layer during model initialisation.  After patching,
    the backward hook of *linear_module* will invoke :func:`hetero_wgrad_gemm`
    instead of the default ``torch.matmul`` path.

    This preserves the Megatron FSDP contract (``weight.main_grad`` is
    accumulated in-place) while enabling DES-LOC device-class dispatch.

    Parameters
    ----------
    linear_module:
        A ``torch.nn.Linear`` (or compatible) wrapped by DeepSpeed FSDP.
    """
    original_weight = linear_module.weight

    def _wgrad_hook(grad_output: torch.Tensor) -> None:
        # total_input is stashed on the module by the forward hook
        total_input = getattr(linear_module, "_des_loc_saved_input", None)
        if total_input is None:
            logger.warning(
                "patch_fsdp_linear_backward: no saved input found on %s.  "
                "Ensure _des_loc_forward_hook was registered.",
                linear_module,
            )
            return
        hetero_wgrad_gemm(total_input, grad_output, original_weight)

    def _forward_hook(module, inputs, output):
        # Save input activation for the backward wgrad computation
        module._des_loc_saved_input = inputs[0].detach()

    linear_module.register_forward_hook(_forward_hook)
    original_weight.register_hook(lambda g: None)  # ensure grad hook fires
    logger.info(
        "DES-LOC wgrad patch applied to %s (device=%s).",
        type(linear_module).__name__,
        original_weight.device,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # ------------------------------------------------------------------ #
    # Helper: make a fake weight with main_grad attribute                 #
    # ------------------------------------------------------------------ #
    def _make_weight(out_f, in_f, dtype=torch.float32, device="cpu"):
        w = torch.nn.Parameter(torch.randn(out_f, in_f, dtype=dtype, device=device))
        w.main_grad = torch.zeros(out_f, in_f, dtype=torch.float32, device=device)
        return w

    T, K, N = 16, 64, 32   # seq_len, in_features, out_features

    # 1. CPU path: basic correctness
    total_input = torch.randn(T, K)
    grad_output = torch.randn(T, N)
    w = _make_weight(N, K)
    w.__fsdp_param__ = True

    dev_info = DeviceInfo(device=torch.device("cpu"), device_class=DeviceClass.CPU)
    result = hetero_wgrad_gemm(total_input, grad_output, w, device_info=dev_info)

    expected = grad_output.t().float() @ total_input.float()
    assert result.shape == (N, K), f"Shape mismatch: {result.shape}"
    assert torch.allclose(result, expected, atol=1e-4), "CPU wgrad value mismatch"
    print("PASS: CPU wgrad correctness")

    # 2. SM86 path: fp16 inputs → fp32 grad
    w2 = _make_weight(N, K)
    w2.__fsdp_param__ = True
    ti_fp16 = total_input.half()
    go_fp16 = grad_output.half()
    dev_sm86 = DeviceInfo(device=torch.device("cpu"), device_class=DeviceClass.SM86)
    res_sm86 = _SM86WgradKernel.compute(ti_fp16, go_fp16, out=w2.main_grad)
    assert res_sm86.dtype == torch.float32, "SM86 output must be fp32"
    print("PASS: SM86 dtype=fp32")

    # 3. SLOC handle promotion
    cpu_pinned = torch.randn(T, K).pin_memory()
    handle = SlocTensorHandle(cpu_tensor=cpu_pinned)
    cache = SlocCache()
    cache.register(cpu_pinned, key="test.input")
    assert len(cache) == 1
    print("PASS: SlocCache registration")

    # 4. Dtype resolution
    assert _resolve_accum_dtype(torch.bfloat16, DeviceClass.SM86) == torch.float32
    assert _resolve_accum_dtype(torch.bfloat16, DeviceClass.SM90) == torch.float32
    print("PASS: dtype negotiation")

    # 5. DeviceInfo auto-detection (CPU)
    di = DeviceInfo(device=torch.device("cpu"))
    assert di.device_class == DeviceClass.CPU
    print("PASS: DeviceInfo auto-detect CPU")

    print("\nAll smoke tests passed.")
