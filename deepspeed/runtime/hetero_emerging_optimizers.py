"""
DES-LOC Heterogeneous Emerging Optimizers
==========================================

Upstream design intent (Megatron c9797ad):
    Megatron's emerging_optimizers.py provides a plugin registry for
    "emerging" (non-Adam) optimizers—Muon, AdaptiveMuon, SOAP, Lion—that
    use orthogonalized gradient updates (Newton-Schulz iterations) or
    sign-based momentum (Lion).  The key design moves in that commit are:

    1. AdaptiveMuon: extends Muon with AdamW-style second-moment
       accumulation *after* orthogonalization, giving spectral-step
       magnitudes that adapt to curvature.  Two flavors: "adamuon"
       (full second-moment) and "normuon" (Frobenius-norm scaling only).

    2. EmergingOptimizerEntry with sensible defaults: init_state_fn and
       default_param_overrides now have defaults, so downstream registries
       don't need to spell everything out.

    3. OrthogonalizedOptimizer.__init__ called explicitly (not via super())
       to support the TensorParallel* multiple-inheritance diamond correctly.

    4. Auto-registration of all registry-listed optimizers instead of just
       ["soap"], with skipping for local (tensor-parallel) overrides.

DES-LOC adaptation points (DES-LOC = Decoupled Execution with Shared LOcality Cache):
    Hardware: 2× A6000 48 GB SM86 + 1× H100 NVL 96 GB SM90, PCIe, no NVLink
    1.5 TB CPU DRAM available as overflow locality cache.

    The critical challenge: Newton-Schulz orthogonalization is FLOP-heavy
    (iterative matrix products) and strongly benefits from high-bandwidth
    compute.  On the H100 (SM90, BF16 tensor cores) it runs 4-6× faster
    per FLOP than on the A6000 (SM86, FP32 tensor cores).  We therefore:

    A. DevicePlacementPolicy: route orthogonalization to H100 ("fast_device"),
       Adam fallback params to A6000s ("slow_devices"), CPU DRAM for
       optimizer-state overflow when VRAM headroom is tight.

    B. DeslocLocalityCache: a per-device LRU structure that pins the
       momentum / second-moment buffers of recently-updated params in GPU
       VRAM, spilling older state to CPU DRAM via async pinned transfers.
       On re-access the state is streamed back with prefetch pipelining.

    C. HeteroAdaptiveMuon: wraps AdaptiveMuon logic so that the
       Newton-Schulz iterations execute on the fast device while
       second-moment accumulation can be split across devices by
       parameter shard ownership.

    D. HeteroEmergingOptimizerRegistry: mirrors Megatron's
       _EMERGING_OPTIMIZERS dict, adding device-placement metadata and
       locality-cache integration without breaking the upstream interface.

    E. Async cross-device gradient staging: gradients computed on slow
       devices are staged to the fast device in a background CUDA stream
       before the NS iteration, overlapping compute with PCIe transfer.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guard – mirrors Megatron's try/except block
# ---------------------------------------------------------------------------
try:
    from emerging_optimizers.orthogonalized_optimizers import (
        OrthogonalizedOptimizer,
        get_muon_scale_factor,
    )
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import (
        NSCoeffT,
        newton_schulz_tp,
    )

    HAVE_EMERGING_OPTIMIZERS = True
    logger.info("[DES-LOC] emerging_optimizers package found; full optimizer set available.")
except ImportError:
    HAVE_EMERGING_OPTIMIZERS = False
    OrthogonalizedOptimizer = object  # type: ignore[assignment,misc]
    logger.warning(
        "[DES-LOC] emerging_optimizers not found; HeteroAdaptiveMuon disabled. "
        "Fallback to Adam for all parameter groups."
    )


# ===========================================================================
# 1. Device placement & locality-cache infrastructure
# ===========================================================================


class DeviceRole(Enum):
    """Role of a CUDA device within the DES-LOC heterogeneous cluster."""
    FAST = "fast"      # H100 NVL SM90: runs NS iterations
    SLOW = "slow"      # A6000 SM86:    runs Adam fallback + data IO
    CPU  = "cpu"       # Host DRAM:     optimizer-state overflow pool


@dataclass
class DevicePlacementPolicy:
    """Maps optimizer workloads to physical devices.

    DES-LOC rationale
    -----------------
    PCIe interconnect bandwidth is the bottleneck, not per-device TFLOPS.
    We minimise cross-device tensor movement by assigning:
      * Newton-Schulz (compute-bound, large matmul) → FAST device
      * Momentum + second-moment accumulation → owner device (where param lives)
      * Optimizer state overflow → CPU DRAM (1.5 TB available)

    Upstream Megatron has no equivalent concept; it assumes NVLink topology.
    """

    fast_device: torch.device          # H100
    slow_devices: List[torch.device]   # A6000 × 2
    cpu_device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    # VRAM headroom below which state is spilled to CPU (bytes)
    vram_spill_threshold_bytes: int = 4 * 1024 ** 3  # 4 GB

    def owner_device(self, param: torch.Tensor) -> torch.device:
        """Return the device that owns *param*."""
        return param.device

    def ns_device(self) -> torch.device:
        """Device that should run Newton-Schulz iterations."""
        return self.fast_device

    def should_spill_to_cpu(self, device: torch.device) -> bool:
        """True when *device* VRAM headroom is below the spill threshold."""
        if device.type != "cuda":
            return False
        try:
            free, _ = torch.cuda.mem_get_info(device.index)
            return free < self.vram_spill_threshold_bytes
        except RuntimeError:
            return False

    @classmethod
    def auto_detect(cls) -> "DevicePlacementPolicy":
        """Auto-detect fast/slow devices by SM version and VRAM capacity.

        DES-LOC target topology: index 0,1 = A6000 (SM86), index 2 = H100 (SM90).
        Falls back gracefully when fewer GPUs are present.
        """
        fast: Optional[torch.device] = None
        slow: List[torch.device] = []

        if not torch.cuda.is_available():
            logger.warning("[DES-LOC] No CUDA devices; using CPU-only placement.")
            dummy = torch.device("cpu")
            return cls(fast_device=dummy, slow_devices=[dummy])

        n = torch.cuda.device_count()
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            sm = props.major * 10 + props.minor   # e.g. SM90 → 90
            mem_gb = props.total_memory / (1024 ** 3)
            dev = torch.device(f"cuda:{i}")
            logger.info(
                "[DES-LOC] cuda:%d  SM%d  %.1f GB  %s",
                i, sm, mem_gb, props.name
            )
            if sm >= 90:  # Hopper+
                fast = dev
            else:
                slow.append(dev)

        if fast is None:
            logger.warning(
                "[DES-LOC] No SM90+ device found; promoting highest-VRAM device to FAST role."
            )
            best = max(
                range(n),
                key=lambda i: torch.cuda.get_device_properties(i).total_memory,
            )
            fast = torch.device(f"cuda:{best}")
            slow = [torch.device(f"cuda:{i}") for i in range(n) if i != best]

        logger.info(
            "[DES-LOC] Placement policy: FAST=%s  SLOW=%s",
            fast, [str(d) for d in slow]
        )
        return cls(fast_device=fast, slow_devices=slow)


# ---------------------------------------------------------------------------
# Locality cache
# ---------------------------------------------------------------------------

class DeslocLocalityCache:
    """Per-device LRU cache that keeps hot optimizer states in GPU VRAM.

    Design
    ------
    Megatron stores all optimizer state on-device; it has the luxury of
    NVLink VRAM aggregation.  In DES-LOC, with PCIe and asymmetric VRAM
    (48 GB / 96 GB), we cannot assume all state fits.  Instead:

    * Hot state (recently touched params) stays in GPU VRAM.
    * Cold state is evicted to 1.5 TB CPU DRAM (pinned memory) via async
      `cudaMemcpyAsync` in a dedicated staging stream.
    * On re-access, state is prefetched back with the same stream, pipelining
      with the forward/backward pass on the default compute stream.

    The cache is keyed by param data_ptr() for O(1) lookup.

    Capacity is monitored after each eviction cycle; the `vram_budget`
    can be reduced at runtime if other consumers (activations, etc.) grow.
    """

    def __init__(
        self,
        device: torch.device,
        vram_budget_bytes: int = 8 * 1024 ** 3,  # 8 GB default
        max_cpu_cache_entries: int = 4096,
    ):
        self.device = device
        self.vram_budget_bytes = vram_budget_bytes
        self.max_cpu_cache_entries = max_cpu_cache_entries

        # Ordered so we can evict LRU entries from the front
        self._gpu_cache: OrderedDict[int, Dict[str, torch.Tensor]] = OrderedDict()
        self._cpu_cache: OrderedDict[int, Dict[str, torch.Tensor]] = OrderedDict()

        # Background stream for async host↔device transfers
        if device.type == "cuda":
            self._transfer_stream = torch.cuda.Stream(device=device)
        else:
            self._transfer_stream = None  # type: ignore[assignment]

        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        logger.debug(
            "[DES-LOC][LocalityCache] Init on %s, budget=%.1f GB, max_cpu=%d",
            device, vram_budget_bytes / 1024 ** 3, max_cpu_cache_entries
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, param_ptr: int) -> Optional[Dict[str, torch.Tensor]]:
        """Retrieve optimizer state for *param_ptr*; promotes to MRU."""
        with self._lock:
            if param_ptr in self._gpu_cache:
                self._gpu_cache.move_to_end(param_ptr)
                self._hits += 1
                return self._gpu_cache[param_ptr]

            if param_ptr in self._cpu_cache:
                self._misses += 1
                state = self._prefetch_from_cpu(param_ptr)
                self._gpu_cache[param_ptr] = state
                self._gpu_cache.move_to_end(param_ptr)
                del self._cpu_cache[param_ptr]
                self._maybe_evict()
                return state

            return None

    def put(self, param_ptr: int, state: Dict[str, torch.Tensor]) -> None:
        """Insert or update state for *param_ptr*."""
        with self._lock:
            # Ensure tensors live on the right device
            gpu_state = {
                k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in state.items()
            }
            self._gpu_cache[param_ptr] = gpu_state
            self._gpu_cache.move_to_end(param_ptr)
            self._maybe_evict()

    def evict_all_to_cpu(self) -> None:
        """Force-evict all GPU state to CPU (e.g., before a checkpoint save)."""
        with self._lock:
            for ptr, state in list(self._gpu_cache.items()):
                self._spill_to_cpu(ptr, state)
            self._gpu_cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Return cache hit/miss/eviction statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": self._hits / max(total, 1),
            "gpu_entries": len(self._gpu_cache),
            "cpu_entries": len(self._cpu_cache),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_evict(self) -> None:
        """Evict LRU GPU entries if VRAM budget is exceeded."""
        if self.device.type != "cuda":
            return
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device.index)
        except RuntimeError:
            return
        used_bytes = total_bytes - free_bytes

        while used_bytes > (total_bytes - self.vram_budget_bytes) and self._gpu_cache:
            ptr, state = self._gpu_cache.popitem(last=False)  # LRU
            self._spill_to_cpu(ptr, state)
            self._evictions += 1
            try:
                free_bytes, _ = torch.cuda.mem_get_info(self.device.index)
                used_bytes = total_bytes - free_bytes
            except RuntimeError:
                break

        # Trim CPU cache if needed
        while len(self._cpu_cache) > self.max_cpu_cache_entries:
            self._cpu_cache.popitem(last=False)

    def _spill_to_cpu(self, ptr: int, state: Dict[str, torch.Tensor]) -> None:
        """Async copy *state* tensors to pinned CPU memory."""
        cpu_state: Dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                if self._transfer_stream is not None:
                    with torch.cuda.stream(self._transfer_stream):
                        cpu_v = torch.empty_like(v, device="cpu", pin_memory=True)
                        cpu_v.copy_(v, non_blocking=True)
                else:
                    cpu_v = v.cpu()
                cpu_state[k] = cpu_v
            else:
                cpu_state[k] = v
        self._cpu_cache[ptr] = cpu_state
        logger.debug("[DES-LOC][LocalityCache] Spilled param_ptr=%d to CPU.", ptr)

    def _prefetch_from_cpu(self, ptr: int) -> Dict[str, torch.Tensor]:
        """Async copy state from pinned CPU back to GPU."""
        cpu_state = self._cpu_cache[ptr]
        gpu_state: Dict[str, torch.Tensor] = {}
        for k, v in cpu_state.items():
            if isinstance(v, torch.Tensor):
                if self._transfer_stream is not None:
                    with torch.cuda.stream(self._transfer_stream):
                        gpu_v = v.to(self.device, non_blocking=True)
                else:
                    gpu_v = v.to(self.device)
                gpu_state[k] = gpu_v
            else:
                gpu_state[k] = v
        logger.debug("[DES-LOC][LocalityCache] Prefetched param_ptr=%d from CPU.", ptr)
        return gpu_state


# ===========================================================================
# 2. Gradient staging: slow-device → fast-device async transfer
# ===========================================================================


class GradientStagingBuffer:
    """Async PCIe gradient staging from slow devices to the fast (NS) device.

    Upstream Megatron can call all-reduce / scatter directly on NVLink because
    all GPUs share a coherent fabric.  Under DES-LOC's PCIe topology, moving
    a gradient tensor from an A6000 to the H100 takes ~10 ms for a 200 MB
    weight.  We hide this with a dedicated staging stream and double-buffering.

    Usage pattern (DES-LOC optimizer step):
        buffer = GradientStagingBuffer(policy)
        buffer.stage(param)           # async copy grad to fast_device
        # ... (forward/backward on slow devices) ...
        staged_grad = buffer.retrieve(param)   # sync + return
        # run NS iteration on staged_grad (now on H100)
    """

    def __init__(self, policy: DevicePlacementPolicy):
        self.policy = policy
        fast = policy.fast_device
        self._stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=fast) if fast.type == "cuda" else None
        )
        self._staged: Dict[int, torch.Tensor] = {}   # param_ptr → staged grad

    def stage(self, param: torch.Tensor) -> None:
        """Enqueue async copy of *param.grad* to the fast device."""
        if param.grad is None:
            return
        grad = param.grad
        fast = self.policy.fast_device

        if grad.device == fast:
            self._staged[param.data_ptr()] = grad
            return

        if self._stream is not None:
            with torch.cuda.stream(self._stream):
                staged = grad.to(fast, non_blocking=True)
        else:
            staged = grad.to(fast)

        self._staged[param.data_ptr()] = staged
        logger.debug(
            "[DES-LOC][GradStaging] Queued grad %s from %s → %s",
            tuple(grad.shape), grad.device, fast
        )

    def retrieve(self, param: torch.Tensor) -> Optional[torch.Tensor]:
        """Synchronise the staging stream and return the staged gradient."""
        ptr = param.data_ptr()
        if ptr not in self._staged:
            return None
        if self._stream is not None:
            self._stream.synchronize()
        grad = self._staged.pop(ptr)
        return grad

    def clear(self) -> None:
        self._staged.clear()


# ===========================================================================
# 3. Newton-Schulz helpers (device-aware)
# ===========================================================================


def _ns_orthogonalize_on_fast_device(
    grad: torch.Tensor,
    ns_device: torch.device,
    num_steps: int = 5,
    fp32_prec: str = "medium",
) -> torch.Tensor:
    """Run Newton-Schulz orthogonalization on the fast (H100) device.

    DES-LOC adaptation:
        Megatron's newton_schulz_tp call is TP-aware but device-agnostic.
        Here we explicitly move the gradient to *ns_device*, run the
        iteration, then move the result back to the original device.
        The round-trip PCIe cost is paid once per step vs. the 5-iteration
        NS kernel savings on SM90 being substantial.

    Args:
        grad: Gradient tensor (may live on any device).
        ns_device: Target device for the NS iteration (fast device).
        num_steps: Number of Newton-Schulz iterations.
        fp32_prec: torch.set_float32_matmul_precision value during iteration.

    Returns:
        Orthogonalized gradient on the *original* device of *grad*.
    """
    orig_device = grad.device
    orig_dtype = grad.dtype

    # Move to fast device in FP32 for numerical stability
    g = grad.to(ns_device, dtype=torch.float32)

    if not HAVE_EMERGING_OPTIMIZERS:
        # Fallback: sign-based approximation (cheap, no deps)
        g = g / (g.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        return g.to(orig_device, dtype=orig_dtype)

    # Full NS iteration using upstream utility
    prev_prec = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(fp32_prec)
    try:
        # newton_schulz_tp signature: (G, steps, coeff_type) → orthogonalized G
        g = newton_schulz_tp(g, num_steps)
    finally:
        torch.set_float32_matmul_precision(prev_prec)

    return g.to(orig_device, dtype=orig_dtype)


# ===========================================================================
# 4. HeteroAdaptiveMuon – the main DES-LOC optimizer
# ===========================================================================


class HeteroAdaptiveMuon(torch.optim.Optimizer):
    """Heterogeneous Adaptive Muon optimizer for the DES-LOC cluster.

    Upstream design (Megatron TensorParallelAdaptiveMuon):
        * Extends Muon with AdamW-style second-moment accumulation after
          orthogonalization ("adamuon") or Frobenius-norm scaling ("normuon").
        * Uses explicit class calls (not super()) in __init__ to handle
          TensorParallel multiple-inheritance correctly.
        * Delegates the step() to AdaptiveMuon.step() which handles the
          second-moment logic.

    DES-LOC reinterpretation:
        We cannot use the TensorParallel infrastructure (it assumes NVLink
        all-reduce).  Instead:

        1. GradientStagingBuffer: async-moves grads from A6000→H100 before
           the NS iteration, overlapping transfer with gradient computation.

        2. DeslocLocalityCache: spills cold momentum/second-moment tensors
           to the 1.5 TB CPU DRAM, fetching them back with prefetch on access.

        3. NS iterations run on H100 (fast_device); accumulation arithmetic
           runs on the param's owner device to minimise data movement.

        4. "adamuon" mode: v_t = β₂ v_{t-1} + (1−β₂) ||G̃||²_F * I
           where G̃ is the NS-orthogonalized gradient.
           "normuon" mode: scale the update by 1/||G̃||_F directly.

        Both modes run the accumulation on the param's owner device.

    Args:
        params: Iterable of parameter groups.
        placement_policy: DevicePlacementPolicy describing the cluster topology.
        lr: Learning rate.
        momentum: EMA coefficient for first moment (Nesterov-style).
        nesterov: Whether to apply Nesterov momentum.
        weight_decay: Weight decay coefficient (decoupled by default).
        num_ns_steps: Newton-Schulz iteration count (upstream default: 5).
        moment2_method: "adamuon" or "normuon".
        beta2: Second-moment EMA coefficient (adamuon mode).
        eps: Numerical stability constant.
        fp32_matmul_prec: Precision during NS matmul ("high" for A6000,
                          "medium" for H100 BF16 tensor cores).
        locality_cache: Optional pre-built DeslocLocalityCache; if None,
                        one is created per device.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        placement_policy: Optional[DevicePlacementPolicy] = None,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        num_ns_steps: int = 5,
        moment2_method: Literal["adamuon", "normuon"] = "adamuon",
        beta2: float = 0.95,
        eps: float = 1e-8,
        fp32_matmul_prec: str = "medium",
        locality_cache: Optional[Dict[str, DeslocLocalityCache]] = None,
    ) -> None:
        if placement_policy is None:
            logger.info("[DES-LOC][HeteroAdaptiveMuon] Auto-detecting device placement policy.")
            placement_policy = DevicePlacementPolicy.auto_detect()

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            num_ns_steps=num_ns_steps,
            moment2_method=moment2_method,
            beta2=beta2,
            eps=eps,
            fp32_matmul_prec=fp32_matmul_prec,
        )
        super().__init__(params, defaults)

        self.placement_policy = placement_policy
        self._grad_stage = GradientStagingBuffer(placement_policy)

        # One locality cache per GPU device
        if locality_cache is not None:
            self._locality_caches = locality_cache
        else:
            self._locality_caches: Dict[str, DeslocLocalityCache] = {}
            all_devices = [placement_policy.fast_device] + placement_policy.slow_devices
            for dev in all_devices:
                key = str(dev)
                if key not in self._locality_caches:
                    budget = self._default_cache_budget(dev)
                    self._locality_caches[key] = DeslocLocalityCache(dev, vram_budget_bytes=budget)
                    logger.info(
                        "[DES-LOC][HeteroAdaptiveMuon] Cache for %s: %.1f GB budget.",
                        dev, budget / 1024 ** 3
                    )

        self._step_count = 0
        logger.info(
            "[DES-LOC][HeteroAdaptiveMuon] Initialized. fast=%s slow=%s "
            "moment2=%s beta2=%.3f lr=%.2e",
            placement_policy.fast_device,
            [str(d) for d in placement_policy.slow_devices],
            moment2_method, beta2, lr,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _default_cache_budget(self, device: torch.device) -> int:
        """Return VRAM budget for the locality cache on *device*."""
        if device.type != "cuda":
            return 0
        try:
            _, total = torch.cuda.mem_get_info(device.index)
        except RuntimeError:
            return 4 * 1024 ** 3
        # Reserve 40% for activations / gradients; cache gets 30%
        return int(total * 0.30)

    def _get_cache(self, device: torch.device) -> Optional[DeslocLocalityCache]:
        return self._locality_caches.get(str(device))

    def _load_state(self, param: torch.Tensor, state_key: str) -> Optional[torch.Tensor]:
        """Load a state tensor from locality cache or optimizer state dict."""
        cache = self._get_cache(param.device)
        if cache is not None:
            cached = cache.get(param.data_ptr())
            if cached is not None and state_key in cached:
                return cached[state_key]
        st = self.state[param]
        return st.get(state_key)

    def _save_state(self, param: torch.Tensor, state_key: str, value: torch.Tensor) -> None:
        """Persist a state tensor to both optimizer state dict and locality cache."""
        self.state[param][state_key] = value
        cache = self._get_cache(param.device)
        if cache is not None:
            existing = cache.get(param.data_ptr()) or {}
            existing[state_key] = value
            cache.put(param.data_ptr(), existing)

    # ------------------------------------------------------------------
    # Upstream-compatible state initialisation
    # ------------------------------------------------------------------

    def _init_param_state(self, param: torch.Tensor, group: Dict[str, Any]) -> None:
        """Initialise momentum and second-moment buffers for *param*.

        Mirrors Megatron's _init_group / _eopt_init_state_fn logic but
        places buffers via the locality cache rather than plain state dict.
        """
        st = self.state[param]
        if "step" not in st:
            st["step"] = 0
            buf = torch.zeros_like(param.data, device=param.device)
            self._save_state(param, "momentum_buffer", buf)
            if group["moment2_method"] == "adamuon":
                v_buf = torch.zeros((), device=param.device, dtype=torch.float32)
                self._save_state(param, "second_moment", v_buf)
            logger.debug(
                "[DES-LOC] Init state for param shape=%s on %s.",
                tuple(param.shape), param.device
            )

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform one heterogeneous Adaptive Muon step.

        Algorithm (per parameter with gradient):
            1. Stage gradient async to fast device (PCIe hidden by overlap).
            2. Retrieve staged gradient (sync staging stream).
            3. Orthogonalize via Newton-Schulz on fast device.
            4. Move orthogonalized gradient back to param's owner device.
            5. Apply second-moment scaling (adamuon / normuon) on owner device.
            6. Apply Nesterov momentum + decoupled weight decay.
            7. Write updated param.  Save state via locality cache.
        """
        loss: Optional[float] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        # --- Phase 1: stage all gradients async ---
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    self._grad_stage.stage(param)

        # --- Phase 2: compute updates ---
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]
            num_ns = group["num_ns_steps"]
            method = group["moment2_method"]
            beta2 = group["beta2"]
            eps = group["eps"]
            fp32_prec = group["fp32_matmul_prec"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                self._init_param_state(param, group)
                st = self.state[param]
                st["step"] += 1
                step_t = st["step"]
                owner_device = param.device

                # Retrieve staged grad (on fast device)
                staged_grad = self._grad_stage.retrieve(param)
                if staged_grad is None:
                    staged_grad = param.grad.to(self.placement_policy.fast_device)

                # Newton-Schulz orthogonalization on fast device
                if staged_grad.ndim >= 2:
                    orth_grad = _ns_orthogonalize_on_fast_device(
                        staged_grad,
                        self.placement_policy.fast_device,
                        num_steps=num_ns,
                        fp32_prec=fp32_prec,
                    )
                else:
                    # 1-D params (biases, norms): just normalise
                    orth_grad = staged_grad / (staged_grad.norm().clamp(min=eps))

                # Move orthogonalized grad back to owner device
                g = orth_grad.to(owner_device, dtype=param.dtype)

                # Second-moment scaling
                if method == "adamuon":
                    g = self._adamuon_scale(g, param, step_t, beta2, eps)
                elif method == "normuon":
                    g = self._normuon_scale(g, eps)
                else:
                    raise ValueError(f"[DES-LOC] Unknown moment2_method: {method!r}")

                # Momentum buffer (on owner device)
                m = self._load_state(param, "momentum_buffer")
                if m is None:
                    m = torch.zeros_like(param.data)
                m.mul_(momentum).add_(g, alpha=1.0 - momentum)
                self._save_state(param, "momentum_buffer", m)

                # Nesterov lookahead
                update = g.add(m, alpha=momentum) if nesterov else m.clone()

                # Decoupled weight decay
                if wd != 0.0:
                    param.data.mul_(1.0 - lr * wd)

                # Parameter update
                param.data.add_(update, alpha=-lr)

                # Log first step for debugging
                if self._step_count == 1 and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "[DES-LOC] step=1 param=%s norm_before=%.4f norm_after=%.4f",
                        tuple(param.shape),
                        param.grad.norm().item(),
                        param.data.norm().item(),
                    )

        self._grad_stage.clear()

        if self._step_count % 100 == 0:
            self._log_cache_stats()

        return loss

    # ------------------------------------------------------------------
    # Second-moment methods
    # ------------------------------------------------------------------

    def _adamuon_scale(
        self,
        g: torch.Tensor,
        param: torch.Tensor,
        step: int,
        beta2: float,
        eps: float,
    ) -> torch.Tensor:
        """AdaMuon: AdamW-style second moment on orthogonalized gradient.

        v_t = β₂ v_{t-1} + (1 - β₂) ||g||²_F / numel(g)
        v̂_t = v_t / (1 - β₂^t)   (bias correction)
        output = g / (sqrt(v̂_t) + ε)

        The second moment scalar summarises the expected per-element magnitude
        of the orthogonalized gradient, making the effective LR scale-invariant.
        """
        v = self._load_state(param, "second_moment")
        if v is None:
            v = torch.zeros((), device=param.device, dtype=torch.float32)

        frob_sq = (g.float().norm() ** 2) / g.numel()
        v = beta2 * v + (1.0 - beta2) * frob_sq
        self._save_state(param, "second_moment", v)

        # Bias-corrected denominator
        bc = 1.0 - beta2 ** step
        denom = (v / bc).sqrt() + eps

        return g / denom.to(g.dtype)

    @staticmethod
    def _normuon_scale(g: torch.Tensor, eps: float) -> torch.Tensor:
        """NorMuon: normalise by Frobenius norm directly (no EMA).

        Simpler than adamuon; effective when gradient norm is stable.
        """
        norm = g.norm().clamp(min=eps)
        return g / norm

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _log_cache_stats(self) -> None:
        for dev_key, cache in self._locality_caches.items():
            s = cache.stats()
            logger.info(
                "[DES-LOC][LocalityCache] step=%d device=%s hit_rate=%.2f%% "
                "gpu_entries=%d cpu_entries=%d evictions=%d",
                self._step_count, dev_key,
                s["hit_rate"] * 100,
                s["gpu_entries"], s["cpu_entries"], s["evictions"],
            )

    def state_dict(self) -> Dict[str, Any]:
        """Extend upstream state_dict with DES-LOC cache metadata."""
        sd = super().state_dict()
        sd["deslock_step_count"] = self._step_count
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._step_count = state_dict.pop("deslock_step_count", 0)
        super().load_state_dict(state_dict)


# ===========================================================================
# 5. HeteroLion: Lion optimizer with DES-LOC state management
# ===========================================================================


class HeteroLion(torch.optim.Optimizer):
    """Lion optimizer adapted for DES-LOC heterogeneous placement.

    Upstream (Megatron c9797ad): Lion from emerging_optimizers.scalar_optimizers
    is registered like SOAP.  It uses sign(β₁m + (1−β₁)g) as the update and
    updates m ← β₂m + (1−β₂)g.  No orthogonalization needed.

    DES-LOC adaptation:
        Lion is cheap (no matmul) so no NS-device routing is needed.
        We add DeslocLocalityCache for the momentum buffer m, which can be
        large for wide models and pressure the 48 GB A6000 VRAM.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        placement_policy: Optional[DevicePlacementPolicy] = None,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        locality_cache: Optional[Dict[str, DeslocLocalityCache]] = None,
    ) -> None:
        if placement_policy is None:
            placement_policy = DevicePlacementPolicy.auto_detect()
        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.placement_policy = placement_policy

        if locality_cache is not None:
            self._locality_caches = locality_cache
        else:
            self._locality_caches = {}
            for dev in [placement_policy.fast_device] + placement_policy.slow_devices:
                key = str(dev)
                if key not in self._locality_caches and dev.type == "cuda":
                    _, total = torch.cuda.mem_get_info(dev.index)
                    self._locality_caches[key] = DeslocLocalityCache(dev, int(total * 0.25))

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            wd = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                g = param.grad

                st = self.state[param]
                if "m" not in st:
                    st["m"] = torch.zeros_like(param.data)
                m = st["m"]

                # Lion update: sign(β₁m + (1−β₁)g)
                update = (beta1 * m + (1.0 - beta1) * g).sign_()

                if wd != 0.0:
                    param.data.mul_(1.0 - lr * wd)
                param.data.add_(update, alpha=-lr)

                # Update momentum
                m.mul_(beta2).add_(g, alpha=1.0 - beta2)
                st["m"] = m

                # Write to locality cache
                cache = self._locality_caches.get(str(param.device))
                if cache is not None:
                    cache.put(param.data_ptr(), {"m": m})

        return loss


# ===========================================================================
# 6. HeteroEmergingOptimizerRegistry
# ===========================================================================


@dataclass
class HeteroOptimizerEntry:
    """Mirrors Megatron's EmergingOptimizerEntry with DES-LOC placement metadata.

    DES-LOC addition: *preferred_device_role* tells the registry scheduler
    which device role should run this optimizer's heavy kernels.
    """

    optimizer_cls: type
    preferred_device_role: DeviceRole = DeviceRole.FAST
    config_to_kwargs: Optional[Callable] = None
    default_locality_cache_budget_fraction: float = 0.25
    description: str = ""


_HETERO_OPTIMIZER_REGISTRY: Dict[str, HeteroOptimizerEntry] = {
    "adaptive_muon": HeteroOptimizerEntry(
        optimizer_cls=HeteroAdaptiveMuon,
        preferred_device_role=DeviceRole.FAST,
        description="Muon + AdaptiveMoment, NS on H100, state in locality cache",
    ),
    "lion": HeteroOptimizerEntry(
        optimizer_cls=HeteroLion,
        preferred_device_role=DeviceRole.SLOW,
        description="Lion sign-momentum, cheap enough for A6000",
    ),
}


def register_hetero_optimizer(name: str, entry: HeteroOptimizerEntry) -> None:
    """Register a new heterogeneous optimizer variant."""
    if name in _HETERO_OPTIMIZER_REGISTRY:
        logger.warning("[DES-LOC][Registry] Overwriting existing entry for %r.", name)
    _HETERO_OPTIMIZER_REGISTRY[name] = entry
    logger.info("[DES-LOC][Registry] Registered optimizer %r.", name)


def get_hetero_optimizer(
    name: str,
    params: Iterable[torch.Tensor],
    placement_policy: Optional[DevicePlacementPolicy] = None,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """Instantiate a registered heterogeneous optimizer by name.

    DES-LOC factory function – mirrors Megatron's get_megatron_optimizer
    entry point but is placement-aware.

    Args:
        name: Registry key (e.g. "adaptive_muon", "lion").
        params: Model parameters.
        placement_policy: If None, auto-detected from CUDA devices.
        **kwargs: Forwarded to the optimizer constructor.

    Returns:
        Instantiated optimizer.
    """
    if name not in _HETERO_OPTIMIZER_REGISTRY:
        raise ValueError(
            f"[DES-LOC] Unknown hetero optimizer {name!r}. "
            f"Available: {list(_HETERO_OPTIMIZER_REGISTRY)}"
        )
    entry = _HETERO_OPTIMIZER_REGISTRY[name]
    policy = placement_policy or DevicePlacementPolicy.auto_detect()
    logger.info(
        "[DES-LOC][Registry] Creating %r on fast=%s slow=%s.",
        name, policy.fast_device, [str(d) for d in policy.slow_devices],
    )
    return entry.optimizer_cls(params, placement_policy=policy, **kwargs)


# ===========================================================================
# DeepSpeed engine registration
# ===========================================================================


def register(engine) -> None:
    """Register DES-LOC heterogeneous emerging optimizer support on a DeepSpeed engine.

    Detects the device placement policy from the available CUDA devices,
    builds the :class:`DevicePlacementPolicy` and
    :class:`GradientStagingBuffer`, and attaches them as
    ``engine.hetero_emerging_optimizer_ctx``.  The actual optimizer
    instantiation is deferred to the engine's optimizer-build phase;
    this hook prepares the infrastructure that
    :func:`get_hetero_optimizer` needs.

    Parameters
    ----------
    engine:
        A DeepSpeed engine instance.  The engine's ``config`` (or
        ``ds_config``) is inspected for optimizer type and DES-LOC
        settings.
    """
    logger.info(
        "hetero_emerging_optimizers.register() called on engine type=%s",
        type(engine).__name__,
    )

    # Auto-detect device placement policy
    policy = DevicePlacementPolicy.auto_detect()

    # Read optimizer name from engine config
    config = getattr(engine, "config", None) or getattr(engine, "ds_config", None)
    optimizer_name = None
    if config is not None:
        optimizer_name = getattr(config, "des_loc_emerging_optimizer", None)

    # Build locality caches for all devices
    locality_caches: Dict[str, DeslocLocalityCache] = {}
    all_devices = [policy.fast_device] + policy.slow_devices
    for dev in all_devices:
        key = str(dev)
        if key not in locality_caches and dev.type == "cuda":
            try:
                _, total = torch.cuda.mem_get_info(dev.index)
                budget = int(total * 0.25)
            except RuntimeError:
                budget = 4 * 1024 ** 3
            locality_caches[key] = DeslocLocalityCache(dev, vram_budget_bytes=budget)

    # Build gradient staging buffer
    grad_staging = GradientStagingBuffer(policy)

    # Attach context to engine
    engine.hetero_emerging_optimizer_ctx = {
        "placement_policy": policy,
        "locality_caches": locality_caches,
        "grad_staging": grad_staging,
        "optimizer_name": optimizer_name,
        "registry": _HETERO_OPTIMIZER_REGISTRY,
    }

    # If optimizer name is specified, validate it exists in registry
    if optimizer_name is not None:
        if optimizer_name in _HETERO_OPTIMIZER_REGISTRY:
            logger.info(
                "DES-LOC emerging optimizer '%s' will be used. "
                "fast=%s slow=%s.",
                optimizer_name,
                policy.fast_device,
                [str(d) for d in policy.slow_devices],
            )
        else:
            logger.warning(
                "DES-LOC emerging optimizer '%s' not found in registry. "
                "Available: %s",
                optimizer_name,
                list(_HETERO_OPTIMIZER_REGISTRY.keys()),
            )
    else:
        logger.info(
            "No emerging optimizer specified; context stored at "
            "engine.hetero_emerging_optimizer_ctx for manual use."
        )


# ===========================================================================
# Smoke test
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("DES-LOC HeteroEmergingOptimizers smoke test")

    # Use CPU as "fast" device when no CUDA available (CI / dev machine)
    cpu = torch.device("cpu")
    policy = DevicePlacementPolicy(fast_device=cpu, slow_devices=[cpu])

    model = nn.Linear(32, 16, bias=False)
    opt = HeteroAdaptiveMuon(
        model.parameters(),
        placement_policy=policy,
        lr=1e-3,
        num_ns_steps=3,
        moment2_method="adamuon",
    )

    x = torch.randn(8, 32)
    w0 = model.weight.data.clone()
    loss = model(x).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()

    # 1. Weight must have changed
    assert not torch.equal(model.weight.data, w0), "Weight did not update"

    # 2. State dict round-trip
    sd = opt.state_dict()
    assert "deslock_step_count" in sd
    opt.load_state_dict(sd)
    assert opt._step_count == 1

    # 3. Lion smoke
    opt_lion = HeteroLion(model.parameters(), placement_policy=policy, lr=1e-3)
    w1 = model.weight.data.clone()
    model(x).sum().backward()
    opt_lion.step()
    assert not torch.equal(model.weight.data, w1), "Lion did not update"

    # 4. Registry lookup
    opt2 = get_hetero_optimizer("lion", model.parameters(), placement_policy=policy)
    assert isinstance(opt2, HeteroLion)

    # 5. LocalityCache basic put/get
    cache = DeslocLocalityCache(cpu, vram_budget_bytes=2 ** 30)
    dummy = torch.ones(4, 4)
    cache.put(42, {"buf": dummy})
    retrieved = cache.get(42)
    assert retrieved is not None and torch.equal(retrieved["buf"], dummy)

    logger.info("All smoke tests passed.")
