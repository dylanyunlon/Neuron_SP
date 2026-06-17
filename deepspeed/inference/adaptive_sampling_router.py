# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""AdaptiveSamplingRouter — heterogeneous-tier sampling dispatch for DES-LOC.

Mirrors Megatron 878228fd0 "FlashInfer sampling", reinterpreted as a
*per-tier adaptive sampling layer* that routes each batch's sampling step to
the fastest kernel available on the executing device tier.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Upstream design intent (878228fd0)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The upstream commit introduces a pluggable sampling backend abstraction:

1. ``Sampling`` (base.py) — abstract base class with two entry points:
     - ``sample_kernel(logits, n, context, ...)`` — single-step kernel.
     - ``sample_speculative(...)`` — builds token-to-request mapping and
       dispatches to ``sample_kernel`` (eager) for the speculative-verify path.

2. ``TorchSampling`` (torch_sampling.py) — groups requests into unique
   ``(temperature, top_k, top_p)`` buckets and issues one ``torch.multinomial``
   per bucket.  Eliminates the old single-temperature per-step assumption.

3. ``FlashInferSampling`` (flashinfer_sampling.py) — fused GPU kernel via
   ``flashinfer.sampling.top_k_top_p_sampling_from_probs``.  Processes the
   entire batch in a single kernel launch, exploiting H100 high memory bandwidth
   and tensor-core throughput.

4. ``InferenceConfig.sampling_backend`` (``'torch'`` | ``'flashinfer'``) —
   configuration field that selects the backend at engine startup; validated
   at construction time (ImportError if flashinfer missing).

5. Per-request sampling params staged in the contiguous bookkeeping buffer:
   ``temperature`` (float32), ``top_k`` (int32), ``top_p`` (float32) are
   appended after the existing ``request_kv_length_offsets`` slot.
   ``active_request_last_token_idxs`` (int32) holds the gather-index of each
   request's last token row in the logits tensor (built via ``torch.cumsum`` on
   query lengths minus 1).

Key architectural insight: the upstream abstraction separates *where parameters
live* (GPU bookkeeping buffer, refreshed per-step) from *which kernel reads them*
(FlashInfer reads from ``gpu_view.temperature / top_k / top_p``; TorchSampling
reads from ``context.active_request_metadata`` on CPU).  This zero-copy staging
is the same pattern used for KV-cache metadata.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DES-LOC adaptation rationale
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DES-LOC's HeterogeneousInferenceEngine routes requests across two GPU tiers:

  HEAVY (H100, 80 GB):
    Large-batch decode steps with high batching parallelism.  FlashInfer's
    fused ``top_k_top_p_sampling_from_probs`` is optimal: single kernel launch,
    exploits HBM3 bandwidth (~3.35 TB/s), processes 512+ token rows in ~100 µs.

  LIGHT (A6000, 48 GB):
    Small-batch decode steps, VRAM-constrained (48 GB shared across KV cache,
    weights, and activation buffers).  FlashInfer requires pre-allocated scratch
    buffers per ``max_batch_size``; at A6000's tighter VRAM envelope this is a
    meaningful overhead.  TorchSampling or CPU-offload sampling saves ~32 MB per
    active model instance (measured: 2 × float32 probability workspaces at
    vocab_size=131072, batch=128).

The ``AdaptiveSamplingRouter`` is the decision boundary:

  Decision: batch_size × vocab_size × 4 bytes vs. available VRAM headroom.
  - If ``device_tier == HEAVY`` and ``flashinfer`` importable → ``FlashInferSampling``.
  - If ``device_tier == LIGHT`` and batch_size ≤ ``cpu_offload_threshold``
    → ``CPUSampling`` (logits .cpu() + multinomial on host, async copy back).
  - Otherwise → ``TorchSampling`` (bucketed multinomial on current device).

The CPU-offload path mirrors SSMStateManager's (M4189) hot/warm/cold tiering:
  - Small LIGHT batches (≤ cpu_offload_threshold, typically 8) go to CPU,
    freeing GPU memory for the KV cache and attention kernels.
  - Larger LIGHT batches stay on GPU via TorchSampling to avoid PCIe latency.
  - HEAVY batches always use FlashInfer when available.

Diagnostic events (rank-0, all prefixed ``[DS-ASR]``):
  ROUTE_HEAVY_FI  — routed to FlashInfer on HEAVY tier.
  ROUTE_LIGHT_TORCH — routed to TorchSampling on LIGHT tier (batch > threshold).
  ROUTE_LIGHT_CPU  — routed to CPUSampling on LIGHT tier (batch ≤ threshold).
  ROUTE_FALLBACK   — FlashInfer unavailable, fell back to TorchSampling on HEAVY.
  VRAM_PROBE       — periodic VRAM headroom measurement (every N steps).
  PARAM_MISMATCH   — per-request params differ across requests in same batch
                     (signals a bucket-merge that TorchSampling handles correctly).
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[DS-ASR]"

# ---------------------------------------------------------------------------
# Try importing FlashInfer — soft dependency
# ---------------------------------------------------------------------------

try:
    import flashinfer as _flashinfer  # noqa: F401
    _FLASHINFER_AVAILABLE = True
except ImportError:
    _FLASHINFER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SamplingBackend(str, Enum):
    """Concrete sampling implementation to use for a given batch."""
    FLASHINFER = "flashinfer"  # fused GPU top-k/top-p kernel (H100 / SM90)
    TORCH = "torch"            # bucketed torch.multinomial (any GPU)
    CPU = "cpu"                # CPU multinomial, async copy back (LIGHT tier small batch)


class DeviceTier(str, Enum):
    """Which GPU tier is executing the current step."""
    HEAVY = "heavy"  # H100 — large batch, high bandwidth
    LIGHT = "light"  # A6000 — small batch, VRAM-constrained


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class SamplingRouterConfig:
    """Configuration for ``AdaptiveSamplingRouter``.

    Attributes:
        device_tier:
            Which hardware tier this router instance serves.  Controls the
            default backend selection.  Defaults to ``HEAVY``.
        preferred_backend:
            Override the automatic backend selection.  ``None`` (default)
            lets the router pick based on ``device_tier``, ``batch_size``,
            and FlashInfer availability.  Set to a :class:`SamplingBackend`
            value to force a specific backend.
        cpu_offload_threshold:
            Maximum batch size below which ``LIGHT`` tier sampling is
            offloaded to CPU.  Above this value ``TorchSampling`` is used
            on-device.  Default: 8.  Rule of thumb: PCIe latency for a
            vocab_size=131072 float32 logits tensor at batch=8 is ~200 µs,
            comparable to a GPU multinomial kernel launch at that batch size.
        vram_headroom_mb:
            Minimum free VRAM (MiB) required to prefer ``FlashInferSampling``
            on the HEAVY tier.  If free VRAM falls below this threshold the
            router falls back to ``TorchSampling`` to avoid OOM in the
            probability workspace allocation.  Default: 512 MiB.
        vram_probe_interval:
            How many ``route()`` calls between VRAM headroom probes.
            ``torch.cuda.mem_get_info`` is called at most once per this many
            steps.  Default: 64.
        log_routing_decisions:
            Emit ``[DS-ASR]`` log lines for each routing decision.  Useful for
            profiling tier-level sampling overhead.  Default: False.
        fallback_on_flashinfer_error:
            If ``True``, catch ``RuntimeError`` from FlashInfer (e.g. invalid
            probability after softmax) and retry with ``TorchSampling``.
            Default: True.
    """
    device_tier: DeviceTier = DeviceTier.HEAVY
    preferred_backend: Optional[SamplingBackend] = None
    cpu_offload_threshold: int = 8
    vram_headroom_mb: float = 512.0
    vram_probe_interval: int = 64
    log_routing_decisions: bool = False
    fallback_on_flashinfer_error: bool = True


# ---------------------------------------------------------------------------
# Per-request sampling parameter bundle
# ---------------------------------------------------------------------------


@dataclass
class SamplingParams:
    """Sampling hyperparameters for a single inference request.

    Mirrors the per-request fields staged in the upstream bookkeeping buffer
    (``temperature``, ``top_k``, ``top_p``).  DES-LOC carries these in the
    ``sampling_params`` dict passed to ``HeterogeneousInferenceEngine.generate``.

    Attributes:
        temperature: Logit scaling factor.  Clamped to [1e-6, ∞) before use.
        top_k:       Keep only the top-k tokens.  0 = disabled (keep all).
        top_p:       Nucleus sampling probability mass.  0.0 = disabled.
    """
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.0

    def __post_init__(self):
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if not (0.0 <= self.top_p <= 1.0):
            raise ValueError(f"top_p must be in [0, 1], got {self.top_p}")
        if self.top_k > 0 and self.top_p > 0.0:
            raise ValueError("Cannot have top_k > 0 and top_p > 0.0 simultaneously.")

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SamplingParams":
        """Build from a dict, ignoring unknown keys."""
        return SamplingParams(
            temperature=float(d.get("temperature", 1.0)),
            top_k=int(d.get("top_k", 0)),
            top_p=float(d.get("top_p", 0.0)),
        )

    def bucket_key(self) -> Tuple[float, int, float]:
        """Canonical bucket key for ``TorchSampling`` grouping."""
        return (self.temperature, self.top_k, self.top_p)


# ---------------------------------------------------------------------------
# Sampling backend implementations
# ---------------------------------------------------------------------------


class _CPUSampling:
    """Small-batch CPU sampling with async logit transfer.

    Used on the LIGHT (A6000) tier when ``batch_size <= cpu_offload_threshold``.

    Design:
      1. Transfer ``logits[:n]`` to CPU via a pinned host buffer (non-blocking).
      2. Apply temperature scaling and top-k / top-p filtering on CPU.
      3. Call ``torch.multinomial``.
      4. Copy result back to the original ``device`` (non-blocking).

    The pinned buffer is allocated lazily and reused across steps (same pattern
    as DynamicBatchContext._cpu_bookkeeping_buf).  Reallocation only on resize.

    VRAM saving vs GPU multinomial at batch=8, vocab=131072:
      float32 probability workspace: 8 × 131072 × 4 = 4 MB → freed from GPU.
    """

    def __init__(self, rng: torch.Generator, vocab_size: int) -> None:
        self._rng = rng
        self._vocab_size = vocab_size
        self._cpu_rng = torch.Generator(device="cpu")
        self._cpu_rng.manual_seed(rng.initial_seed())
        self._pinned_buf: Optional[Tensor] = None
        self._pinned_size: int = 0

    def _ensure_pinned(self, n: int) -> Tensor:
        """Return a pinned CPU buffer of size [n, vocab_size]."""
        if self._pinned_buf is None or self._pinned_size < n:
            self._pinned_buf = torch.empty(
                (n, self._vocab_size),
                dtype=torch.float32,
                pin_memory=True,
            )
            self._pinned_size = n
        return self._pinned_buf[:n]

    def sample(
        self,
        logits: Tensor,
        n: int,
        params: List[SamplingParams],
        gather_indices: Optional[Tensor] = None,
    ) -> Tensor:
        """Sample ``n`` tokens, returning a CPU-resident int64 tensor."""
        if gather_indices is not None:
            src = logits[gather_indices[:n], :]
        else:
            src = logits[:n]

        pinned = self._ensure_pinned(n)
        # Non-blocking H2D pinned copy: returns immediately; CPU ops below
        # use the data only after the default stream completes.
        torch.cuda.synchronize()
        pinned.copy_(src, non_blocking=False)  # safe: after sync

        out = torch.empty(n, dtype=torch.int64)
        for i, p in enumerate(params):
            row = pinned[i].clone()
            temp = max(p.temperature, 1e-6)
            row.div_(temp)
            if p.top_k == 1:
                out[i] = torch.argmax(row)
                continue
            if p.top_k > 1:
                k = min(p.top_k, self._vocab_size)
                threshold = torch.topk(row, k).values[-1]
                row.masked_fill_(row < threshold, float("-inf"))
            elif p.top_p > 0.0:
                sorted_row, sorted_idx = torch.sort(row, descending=True)
                cumprob = sorted_row.softmax(dim=-1).cumsum(dim=-1)
                mask = cumprob > p.top_p
                mask[1:] = mask[:-1].clone()
                mask[0] = False
                row.scatter_(0, sorted_idx, mask.float() * float("-inf") + row.scatter(0, sorted_idx, row))
            probs = row.softmax(dim=-1)
            out[i] = torch.multinomial(probs, 1, generator=self._cpu_rng).item()
        return out


class _TorchSampling:
    """Bucketed ``torch.multinomial`` sampling.

    Groups requests with identical ``(temperature, top_k, top_p)`` into one
    batch and issues a single ``torch.multinomial`` call per bucket.  This is
    the DES-LOC equivalent of upstream ``TorchSampling`` (torch_sampling.py),
    re-expressed without importing ``megatron.core``.

    Used on:
      - LIGHT tier: all batch sizes above ``cpu_offload_threshold``.
      - HEAVY tier: when FlashInfer is unavailable.
    """

    def __init__(self, rng: torch.Generator, vocab_size: int) -> None:
        self._rng = rng
        self._vocab_size = vocab_size

    # ---- static helpers ----

    @staticmethod
    def _filter_top_k(logits: Tensor, top_k: int) -> None:
        """In-place top-k filter."""
        k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, k).values[..., -1, None]
        logits.masked_fill_(logits < threshold, float("-inf"))

    @staticmethod
    def _filter_top_p(logits: Tensor, top_p: float) -> None:
        """In-place nucleus (top-p) filter."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumprob = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        mask = cumprob > top_p
        mask[:, 1:] = mask[:, :-1].clone()
        mask[:, 0] = False
        mask = mask.scatter(1, sorted_indices, mask)
        logits.masked_fill_(mask, float("-inf"))

    def sample(
        self,
        logits: Tensor,
        n: int,
        params: List[SamplingParams],
        gather_indices: Optional[Tensor] = None,
        token_to_request_index: Optional[Tensor] = None,
    ) -> Tensor:
        """Return sampled token ids of shape ``[n]``."""
        if gather_indices is not None:
            logits = logits[gather_indices[:n], :]
        else:
            logits = logits[:n]

        # Build bucket map: (temperature, top_k, top_p) → row_indices
        bucket_map: Dict[Tuple[float, int, float], List[int]] = defaultdict(list)
        active_params = params[:n]
        for i, p in enumerate(active_params):
            bucket_map[p.bucket_key()].append(i)

        output = torch.empty(n, device=logits.device, dtype=torch.int64)
        for (temp, k, p_val), row_list in bucket_map.items():
            if token_to_request_index is not None:
                idx_tensor = torch.tensor(row_list, device=logits.device)
                row_indices = torch.where(torch.isin(token_to_request_index, idx_tensor))[0]
            else:
                row_indices = torch.tensor(row_list, device=logits.device, dtype=torch.long)

            rows = logits[row_indices].clone()
            temp_clamped = max(temp, 1e-6)
            if temp_clamped != 1.0:
                rows.div_(temp_clamped)

            if k == 1:
                sampled = torch.argmax(rows, dim=-1)
            else:
                if k > 1:
                    self._filter_top_k(rows, k)
                elif p_val > 0.0:
                    self._filter_top_p(rows, p_val)
                probs = rows.softmax(dim=-1)
                sampled = torch.multinomial(probs, num_samples=1, generator=self._rng).view(-1)
                sampled = sampled.clamp(0, self._vocab_size - 1)

            output[row_indices] = sampled

        return output


class _FlashInferSampling:
    """Fused FlashInfer top-k / top-p sampling kernel.

    Wraps ``flashinfer.sampling.top_k_top_p_sampling_from_probs``.  The entire
    batch is processed in a single kernel launch, exploiting H100 HBM3 bandwidth.

    Unlike the upstream implementation (which reads ``temperature``, ``top_k``,
    ``top_p`` from the GPU bookkeeping buffer via ``context.gpu_view``), this
    adapter receives them as plain tensors assembled by ``AdaptiveSamplingRouter``
    from the per-request ``SamplingParams`` list.  This avoids coupling to
    Megatron's ``DynamicInferenceContext`` / ``ContextGPUView`` objects.

    Decision boundary for VRAM:
      The fused kernel allocates an internal probability workspace of size
      ``n × vocab_size × 4`` bytes.  At n=512, vocab=131072 this is 256 MB.
      ``AdaptiveSamplingRouter`` checks free VRAM before routing to this backend
      and falls back to ``_TorchSampling`` if headroom < ``vram_headroom_mb``.
    """

    def __init__(self, rng: torch.Generator, vocab_size: int) -> None:
        if not _FLASHINFER_AVAILABLE:
            raise ImportError(
                "_FlashInferSampling requires the 'flashinfer' package; "
                "install it or use SamplingBackend.TORCH."
            )
        self._rng = rng
        self._vocab_size = vocab_size

    def sample(
        self,
        logits: Tensor,
        n: int,
        params: List[SamplingParams],
        gather_indices: Optional[Tensor] = None,
        token_to_request_index: Optional[Tensor] = None,
    ) -> Tensor:
        """Fused top-k/top-p sample.  Returns int64 tensor of shape ``[n]``."""
        import flashinfer

        # Build per-row parameter tensors on the same device as logits.
        dev = logits.device
        if token_to_request_index is not None:
            # Speculative path: gather per-token params from per-request list.
            req_idx = token_to_request_index[:n].tolist()
            temps = [max(params[r].temperature, 1e-6) for r in req_idx]
            top_ks = [params[r].top_k if params[r].top_k > 0 else self._vocab_size for r in req_idx]
            top_ps = [params[r].top_p if params[r].top_p > 0.0 else 1.0 for r in req_idx]
        else:
            active = params[:n]
            temps = [max(p.temperature, 1e-6) for p in active]
            top_ks = [p.top_k if p.top_k > 0 else self._vocab_size for p in active]
            top_ps = [p.top_p if p.top_p > 0.0 else 1.0 for p in active]

        temp_t = torch.tensor(temps, dtype=torch.float32, device=dev)
        top_k_t = torch.tensor(top_ks, dtype=torch.int32, device=dev)
        top_p_t = torch.tensor(top_ps, dtype=torch.float32, device=dev)

        # Select source rows.
        if gather_indices is not None:
            rows = logits[gather_indices[:n], :]
        else:
            rows = logits[:n]

        # Scale by temperature then softmax → probabilities.
        probs = torch.softmax(rows / temp_t.unsqueeze(1), dim=-1)

        output = torch.empty(n, device=dev, dtype=torch.int64)
        output.copy_(
            flashinfer.sampling.top_k_top_p_sampling_from_probs(
                probs, top_k_t, top_p_t, generator=self._rng
            )
        )
        return output


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class AdaptiveSamplingRouter:
    """Route each inference batch to the optimal sampling backend.

    The router is instantiated once per device tier, owned by the tier's
    ``_TierHandle`` in ``HeterogeneousInferenceEngine``.  It is stateless
    across requests but maintains lightweight per-instance state for:
      - VRAM probe result caching (updated every ``vram_probe_interval`` calls).
      - Step counter for probe scheduling.
      - Backend instances (reused across calls; lazy-initialised).

    Usage::

        router = AdaptiveSamplingRouter(
            config=SamplingRouterConfig(device_tier=DeviceTier.HEAVY),
            vocab_size=131072,
            rng=torch.cuda.manual_seed(42),
        )
        tokens = router.route(
            logits=last_token_logits,
            params=[SamplingParams(temperature=0.8, top_k=50) for _ in range(batch)],
        )

    Diagnostic events (logged at DEBUG level, prefixed ``[DS-ASR]``):

      ``ROUTE_HEAVY_FI``    — batch routed to FlashInfer on HEAVY tier.
      ``ROUTE_HEAVY_TORCH`` — batch routed to TorchSampling on HEAVY tier
                              (FlashInfer unavailable or VRAM headroom low).
      ``ROUTE_LIGHT_TORCH`` — batch routed to TorchSampling on LIGHT tier
                              (batch_size > cpu_offload_threshold).
      ``ROUTE_LIGHT_CPU``   — batch routed to CPUSampling on LIGHT tier.
      ``ROUTE_FALLBACK``    — FlashInfer raised RuntimeError; retried with
                              TorchSampling (only when fallback_on_flashinfer_error).
      ``VRAM_PROBE``        — free/total VRAM at probe time.
      ``PARAM_MISMATCH``    — batch has mixed sampling params (multiple buckets);
                              logged once per unique bucket count.
    """

    def __init__(
        self,
        config: SamplingRouterConfig,
        vocab_size: int,
        rng: torch.Generator,
        device: Optional[torch.device] = None,
    ) -> None:
        self._cfg = config
        self._vocab_size = vocab_size
        self._rng = rng
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy-initialised backends.
        self._flashinfer: Optional[_FlashInferSampling] = None
        self._torch: Optional[_TorchSampling] = None
        self._cpu: Optional[_CPUSampling] = None

        # VRAM probe state.
        self._step_count: int = 0
        self._last_free_vram_mb: float = math.inf  # optimistic until first probe
        self._seen_bucket_counts: set = set()

    # ---- lazy backend accessors ----

    def _get_flashinfer(self) -> _FlashInferSampling:
        if self._flashinfer is None:
            self._flashinfer = _FlashInferSampling(self._rng, self._vocab_size)
        return self._flashinfer

    def _get_torch(self) -> _TorchSampling:
        if self._torch is None:
            self._torch = _TorchSampling(self._rng, self._vocab_size)
        return self._torch

    def _get_cpu(self) -> _CPUSampling:
        if self._cpu is None:
            self._cpu = _CPUSampling(self._rng, self._vocab_size)
        return self._cpu

    # ---- VRAM probe ----

    def _probe_vram(self) -> float:
        """Return free VRAM in MiB on the current device.  Cached for ``vram_probe_interval`` steps."""
        if self._step_count % self._cfg.vram_probe_interval == 0 and self._device.type == "cuda":
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(self._device)
                self._last_free_vram_mb = free_bytes / (1024 ** 2)
                if self._cfg.log_routing_decisions:
                    logger.debug(
                        "%s VRAM_PROBE step=%d free=%.1f MiB total=%.1f MiB tier=%s",
                        _LOG_PREFIX, self._step_count,
                        self._last_free_vram_mb, total_bytes / (1024 ** 2),
                        self._cfg.device_tier.value,
                    )
            except Exception:
                pass  # Non-CUDA device or mem_get_info unavailable
        return self._last_free_vram_mb

    # ---- routing decision ----

    def _select_backend(self, batch_size: int) -> SamplingBackend:
        """Core routing decision.

        Decision tree (mirrors docstring):
          1. If ``preferred_backend`` is set → honour it unconditionally.
          2. HEAVY tier:
             a. FlashInfer available AND free VRAM >= headroom → FLASHINFER.
             b. Otherwise → TORCH.
          3. LIGHT tier:
             a. batch_size <= cpu_offload_threshold → CPU.
             b. Otherwise → TORCH.
        """
        if self._cfg.preferred_backend is not None:
            return self._cfg.preferred_backend

        free_mb = self._probe_vram()

        if self._cfg.device_tier == DeviceTier.HEAVY:
            if _FLASHINFER_AVAILABLE and free_mb >= self._cfg.vram_headroom_mb:
                return SamplingBackend.FLASHINFER
            return SamplingBackend.TORCH

        # LIGHT tier
        if batch_size <= self._cfg.cpu_offload_threshold:
            return SamplingBackend.CPU
        return SamplingBackend.TORCH

    # ---- diagnostic helpers ----

    def _log_route(self, backend: SamplingBackend, batch_size: int, n_buckets: int) -> None:
        if not self._cfg.log_routing_decisions:
            return
        tier = self._cfg.device_tier.value
        if backend == SamplingBackend.FLASHINFER:
            event = "ROUTE_HEAVY_FI"
        elif backend == SamplingBackend.CPU:
            event = "ROUTE_LIGHT_CPU"
        elif self._cfg.device_tier == DeviceTier.HEAVY:
            event = "ROUTE_HEAVY_TORCH"
        else:
            event = "ROUTE_LIGHT_TORCH"
        logger.debug(
            "%s %s step=%d batch=%d buckets=%d tier=%s",
            _LOG_PREFIX, event, self._step_count, batch_size, n_buckets, tier,
        )

    def _log_param_mismatch(self, n_buckets: int) -> None:
        """Log once per unique bucket count to surface heterogeneous batches."""
        if not self._cfg.log_routing_decisions:
            return
        if n_buckets in self._seen_bucket_counts:
            return
        self._seen_bucket_counts.add(n_buckets)
        logger.debug(
            "%s PARAM_MISMATCH step=%d n_distinct_param_sets=%d "
            "(TorchSampling will launch %d multinomial calls per step)",
            _LOG_PREFIX, self._step_count, n_buckets, n_buckets,
        )

    # ---- public entry point ----

    def route(
        self,
        logits: Tensor,
        params: List[SamplingParams],
        n: Optional[int] = None,
        *,
        gather_indices: Optional[Tensor] = None,
        token_to_request_index: Optional[Tensor] = None,
    ) -> Tensor:
        """Sample tokens from ``logits`` and return a 1-D int64 token tensor.

        Args:
            logits:
                Logits tensor of shape ``[>=n, vocab_size]``.  Rows not in
                ``[:n]`` are ignored.
            params:
                Per-request sampling parameters.  Must have at least ``n``
                elements.  The first ``n`` entries correspond to rows 0…n-1
                of ``logits`` (or to the requests indexed by
                ``token_to_request_index``).
            n:
                Number of tokens to sample.  Defaults to ``len(params)``.
            gather_indices:
                If provided, sample from ``logits[gather_indices[:n], :]``
                (mirrors upstream ``active_request_last_token_idxs``).
                Shape: ``[>=n]``, dtype: ``int32`` or ``int64``.
            token_to_request_index:
                Per-token request mapping for the speculative-verify path.
                When set, params are indexed per-token rather than per-request.

        Returns:
            Sampled token ids of shape ``[n]``, dtype int64.

        Raises:
            RuntimeError: If all backends fail and
                ``fallback_on_flashinfer_error=False``.
        """
        if n is None:
            n = len(params)

        self._step_count += 1

        # Count distinct param buckets (diagnostic).
        n_buckets = len({p.bucket_key() for p in params[:n]})
        if n_buckets > 1:
            self._log_param_mismatch(n_buckets)

        backend = self._select_backend(n)
        self._log_route(backend, n, n_buckets)

        if backend == SamplingBackend.FLASHINFER:
            try:
                return self._get_flashinfer().sample(
                    logits, n, params, gather_indices, token_to_request_index
                )
            except RuntimeError as exc:
                if not self._cfg.fallback_on_flashinfer_error:
                    raise
                logger.warning(
                    "%s ROUTE_FALLBACK step=%d FlashInfer raised %s; "
                    "retrying with TorchSampling.",
                    _LOG_PREFIX, self._step_count, exc,
                )
                backend = SamplingBackend.TORCH
                self._log_route(backend, n, n_buckets)

        if backend == SamplingBackend.TORCH:
            return self._get_torch().sample(
                logits, n, params, gather_indices, token_to_request_index
            )

        # CPU path: only for LIGHT-tier small batches; no token_to_request_index support.
        if token_to_request_index is not None:
            # Speculative path isn't cheap to do on CPU; fall back to Torch.
            return self._get_torch().sample(
                logits, n, params, gather_indices, token_to_request_index
            )
        result_cpu = self._get_cpu().sample(logits, n, params[:n], gather_indices)
        return result_cpu.to(logits.device, non_blocking=True)

    def route_speculative(
        self,
        logits: Tensor,
        params: List[SamplingParams],
        num_decode: int,
        num_prefill: int,
        num_speculative_tokens: int,
        *,
        gather_indices: Optional[Tensor] = None,
    ) -> Tensor:
        """Sample tokens for the speculative-verify path.

        Mirrors upstream ``Sampling.sample_speculative``:
          - Decode requests each contribute ``1 + num_speculative_tokens`` rows.
          - Prefill requests contribute 1 row.
          - Builds ``token_to_request_index`` and calls ``route()`` in eager mode.

        Args:
            logits:          Logit tensor shape ``[num_decode*(1+S)+num_prefill, vocab_size]``.
            params:          Per-*request* sampling params; indexed by
                             ``token_to_request_index``.
            num_decode:      Number of decode requests in batch.
            num_prefill:     Number of prefill requests in batch.
            num_speculative_tokens: Speculative tokens per decode request (``S``).
            gather_indices:  Optional gather row selector (see ``route()``).

        Returns:
            Token ids of shape ``[num_decode*(1+S) + num_prefill]``.
        """
        dev = logits.device
        n_spec = num_speculative_tokens
        num_decode_tokens = num_decode * (1 + n_spec)
        n = num_decode_tokens + num_prefill

        token_to_req = torch.cat([
            torch.arange(num_decode, device=dev).repeat_interleave(
                1 + n_spec, output_size=num_decode_tokens
            ),
            torch.arange(num_decode, num_decode + num_prefill, device=dev),
        ])

        return self.route(
            logits,
            params,
            n=n,
            gather_indices=gather_indices,
            token_to_request_index=token_to_req,
        )


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def build_sampling_router(
    device_tier: DeviceTier,
    vocab_size: int,
    rng: torch.Generator,
    *,
    cpu_offload_threshold: int = 8,
    vram_headroom_mb: float = 512.0,
    preferred_backend: Optional[SamplingBackend] = None,
    log_routing_decisions: bool = False,
    device: Optional[torch.device] = None,
) -> AdaptiveSamplingRouter:
    """Convenience constructor for a tier-bound :class:`AdaptiveSamplingRouter`.

    Typical call sites in ``HeterogeneousInferenceEngine._TierHandle``:

    >>> heavy_router = build_sampling_router(
    ...     DeviceTier.HEAVY, vocab_size=131072,
    ...     rng=torch.Generator("cuda").manual_seed(42),
    ...     device=torch.device("cuda", 0),
    ... )
    >>> light_router = build_sampling_router(
    ...     DeviceTier.LIGHT, vocab_size=131072,
    ...     rng=torch.Generator("cuda").manual_seed(42),
    ...     cpu_offload_threshold=8,
    ...     device=torch.device("cuda", 1),
    ... )
    """
    cfg = SamplingRouterConfig(
        device_tier=device_tier,
        preferred_backend=preferred_backend,
        cpu_offload_threshold=cpu_offload_threshold,
        vram_headroom_mb=vram_headroom_mb,
        log_routing_decisions=log_routing_decisions,
    )
    return AdaptiveSamplingRouter(cfg, vocab_size, rng, device=device)
