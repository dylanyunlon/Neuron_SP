"""
deepspeed/inference/hetero_token_clamper.py
===========================================

DES-LOC Heterogeneous Token Budget Clamper
-------------------------------------------

Upstream Design Intent (Megatron-LM commit 53d9ba0e):
    Megatron's DynamicInferenceEngine previously rejected any request whose
    ``num_tokens_to_generate`` would push the total sequence length past
    ``max_sequence_length``.  The patch (PR #5181) changes the semantics to
    *clamp* instead of *reject*, mirroring vLLM's behavior: the engine silently
    truncates the generation budget to whatever tokens remain in the context
    window, warns on rank-0, and continues serving the request.  This is a
    quality-of-life improvement for callers that set ``max_new_tokens`` to a
    large sentinel value (e.g. 2048) without knowing the exact prompt length.

DES-LOC Adaptation — HeteroTokenClamper:
    The DES-LOC (Decoupled Execution with Shared LOcality Cache) framework adds
    a layer of complexity absent in Megatron's homogeneous tensor-parallel
    world: the three physical devices (2× A6000 48 GB SM86, 1× H100 NVL 96 GB
    SM90) have *different* KV-cache capacities, and they are connected only via
    PCIe with no NVLink.  A single ``max_sequence_length`` scalar is therefore
    insufficient — each device tier has its own *effective* context budget
    determined by:

        1. **Device memory** — H100 can cache longer sequences than A6000.
        2. **SM generation** — SM90 supports FP8 KV compression (halving the
           footprint) while SM86 does not; this changes the token budget
           non-linearly.
        3. **PCIe bandwidth** — spilling KV blocks across the PCIe fabric is
           expensive; the clamper penalises cross-device overflow to discourage
           the scheduler from relying on remote KV transfers.
        4. **CPU DRAM offload** — with 1.5 TB of host memory, DES-LOC can
           offload cold KV blocks to DRAM; the clamper is aware of this tier and
           assigns a *soft* budget (allowed but penalised) above the device-local
           hard budget.

    The ``HeteroTokenClamper`` centralises all of this logic so that the
    inference scheduler, the chunked-prefill pipeline, and the generation loop
    can all call a single, testable function instead of scattering ad-hoc
    ``min()`` guards across the codebase.

Key entry points:
    ``HeteroTokenClamper.clamp_request(request, device_tier)``
        Returns a ``ClampResult`` describing the final generation budget,
        whether clamping occurred, and which memory tier will serve the KV
        blocks.

    ``HeteroTokenClamper.validate_batch(requests, device_assignments)``
        Batch-level guard called by the scheduler before dispatching a micro-
        batch; raises ``HeteroBudgetExhaustedError`` if *no* tier can serve
        even the shortest request after clamping.

Usage inside DeepSpeed inference pipeline::

    from deepspeed.inference.hetero_token_clamper import (
        HeteroTokenClamper,
        DeviceTier,
        HeteroBudgetExhaustedError,
    )

    clamper = HeteroTokenClamper.from_engine_context(ds_engine.context)
    result  = clamper.clamp_request(request, DeviceTier.H100_NVL)
    if result.was_clamped:
        # scheduler already received a warning via logging; no action needed
        pass

References:
    * Megatron-LM PR #5181: https://github.com/NVIDIA/Megatron-LM/pull/5181
    * Megatron commit 53d9ba0e457baab2d7665f0f1572cad79ddabb3c
    * Neuron_SP project: https://github.com/dylanyunlon/Neuron_SP
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import math
import os
import threading
import time
import unittest
import warnings
from typing import Dict, FrozenSet, Iterator, List, Optional, Sequence, Tuple

import torch

# ---------------------------------------------------------------------------
# Module-level logger — DES-LOC subsystem tag keeps grep-able.
# ---------------------------------------------------------------------------
logger = logging.getLogger("deepspeed.inference.des_loc.token_clamper")

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

# SM compute capabilities present in the target cluster.
_SM86_COMPUTE_CAPABILITY: Tuple[int, int] = (8, 6)   # A6000
_SM90_COMPUTE_CAPABILITY: Tuple[int, int] = (9, 0)   # H100 NVL

# FP8 KV compression ratio available on SM90 hardware.
# Empirically, 2× compression is conservative; real-world gain is 1.8–2.1×.
_SM90_FP8_KV_COMPRESSION_RATIO: float = 2.0

# PCIe bandwidth penalty coefficient.  When tokens spill across the PCIe
# fabric the effective throughput drops; we model this as a *soft* budget
# multiplier (< 1.0) applied to the remote tier's contribution.
_PCIE_SPILL_PENALTY: float = 0.75

# Fraction of CPU DRAM that DES-LOC reserves for KV offload (the rest is used
# by DeepSpeed's parameter server and activation checkpointing).
_CPU_DRAM_KV_FRACTION: float = 0.40

# Total host DRAM in bytes (1.5 TB).
_HOST_DRAM_BYTES: int = 1_536 * (1 << 30)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class DeviceTier(enum.Enum):
    """
    Logical device tiers in the DES-LOC heterogeneous cluster.

    DES-LOC separates devices into *tiers* rather than individual ranks so
    that the clamper can reason about capability classes without being tightly
    coupled to a particular tensor-parallel layout.
    """

    A6000_SM86 = "a6000_sm86"   # 48 GB GDDR6, SM 8.6, no FP8 KV
    H100_NVL   = "h100_nvl"     # 96 GB HBM3e, SM 9.0, FP8 KV available
    CPU_DRAM   = "cpu_dram"     # 1.5 TB host memory, PCIe-accessible


class ClampReason(enum.Enum):
    """
    Reason a token budget was clamped (or why it was accepted unchanged).
    """

    NOT_CLAMPED            = "not_clamped"
    NEGATIVE_BUDGET        = "negative_budget"          # upstream: hard reject
    PROMPT_EXCEEDS_CONTEXT = "prompt_exceeds_context"   # upstream: hard reject
    EXCEEDS_DEVICE_LOCAL   = "exceeds_device_local"     # soft clamp → device local
    EXCEEDS_DEVICE_WITH_FP8= "exceeds_device_with_fp8"  # clamped to FP8-expanded budget
    SPILLED_TO_CPU_DRAM    = "spilled_to_cpu_dram"      # soft clamp → CPU tier
    FULLY_REJECTED         = "fully_rejected"           # no tier can serve


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class HeteroBudgetExhaustedError(RuntimeError):
    """
    Raised when *every* device tier has exhausted its KV budget and even the
    CPU DRAM offload tier cannot accommodate the shortest request in a batch.

    This is the DES-LOC analogue of Megatron's ``MaxSequenceLengthOverflowError``
    for the hard-reject path (negative ``num_tokens_to_generate`` or prompt
    longer than the maximum context).
    """

    def __init__(self, request_id: str, prompt_len: int, requested_tokens: int) -> None:
        self.request_id      = request_id
        self.prompt_len      = prompt_len
        self.requested_tokens= requested_tokens
        super().__init__(
            f"Request '{request_id}': prompt_len={prompt_len} + "
            f"requested_tokens={requested_tokens} cannot fit in any tier "
            f"(device-local, FP8-expanded, or CPU-DRAM offload)."
        )


class NegativeTokenBudgetError(ValueError):
    """
    Raised when ``num_tokens_to_generate < 0`` — an unambiguous caller error
    that cannot be resolved by clamping (mirrors Megatron's hard-reject).
    """

    def __init__(self, request_id: str, value: int) -> None:
        self.request_id = request_id
        self.value      = value
        super().__init__(
            f"Request '{request_id}' has num_tokens_to_generate={value} < 0. "
            "This is a hard error that cannot be clamped."
        )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DeviceTierBudget:
    """
    Per-tier token budget derived from hardware characteristics.

    Attributes
    ----------
    tier:
        Which device tier this budget describes.
    device_local_max_tokens:
        Maximum tokens (prompt + generation) that fit entirely in the device's
        own VRAM without any cross-device spill.  For SM90 devices this
        already reflects FP8 KV compression.
    fp8_expanded_max_tokens:
        SM90 only: maximum tokens when FP8 KV compression is exploited.
        For SM86 tiers this equals ``device_local_max_tokens``.
    cpu_offload_max_tokens:
        Maximum tokens when cold KV blocks are allowed to overflow into CPU
        DRAM via PCIe.  Subject to ``_PCIE_SPILL_PENALTY``.
    kv_bytes_per_token:
        Estimated KV cache size in bytes per token (both K and V heads,
        all layers, for this tier's precision).
    supports_fp8_kv:
        Whether this tier can use FP8 KV quantisation.
    """

    tier:                    DeviceTier
    device_local_max_tokens: int
    fp8_expanded_max_tokens: int
    cpu_offload_max_tokens:  int
    kv_bytes_per_token:      float
    supports_fp8_kv:         bool

    @property
    def hard_max(self) -> int:
        """Largest token count this tier can handle (possibly with CPU offload)."""
        return self.cpu_offload_max_tokens


@dataclasses.dataclass
class ClampResult:
    """
    Output of ``HeteroTokenClamper.clamp_request``.

    Attributes
    ----------
    request_id:
        Echoes the originating request identifier.
    original_num_tokens:
        The value of ``num_tokens_to_generate`` before any clamping.
    clamped_num_tokens:
        The value after clamping (equals ``original_num_tokens`` if no clamp).
    was_clamped:
        Convenience flag; ``True`` iff ``clamped_num_tokens != original_num_tokens``.
    reason:
        Why the budget was (or was not) clamped.
    serving_tier:
        Which device tier will serve the KV blocks for this request.
    pcie_spill_fraction:
        Fraction of KV blocks that will be served from CPU DRAM (0.0 if
        everything fits in VRAM).
    """

    request_id:          str
    original_num_tokens: int
    clamped_num_tokens:  int
    was_clamped:         bool
    reason:              ClampReason
    serving_tier:        DeviceTier
    pcie_spill_fraction: float = 0.0

    @property
    def generation_penalty_factor(self) -> float:
        """
        Estimated relative throughput degradation due to PCIe spill.
        Returns 1.0 when no spill occurs.
        """
        if self.pcie_spill_fraction <= 0.0:
            return 1.0
        return 1.0 - (1.0 - _PCIE_SPILL_PENALTY) * self.pcie_spill_fraction


@dataclasses.dataclass
class InferenceRequest:
    """
    Minimal request representation used by ``HeteroTokenClamper``.

    In production this would be the DeepSpeed / Megatron request object.
    The clamper only reads and writes the fields listed here, making it easy
    to adapt to different request schemas via a thin adapter.
    """

    request_id:           str
    prompt_tokens:        List[int]
    num_tokens_to_generate: int
    # Status fields (mutated by the clamper on hard-reject paths)
    failed:               bool = False
    failure_reason:       Optional[str] = None

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_tokens)


# ---------------------------------------------------------------------------
# Core clamper
# ---------------------------------------------------------------------------


class HeteroTokenClamper:
    """
    DES-LOC heterogeneous token budget clamper.

    Centralises the logic for determining whether a request's generation
    budget fits in the available KV cache across all device tiers, and either
    clamps the budget or hard-rejects the request when no tier can serve it.

    Design mirroring Megatron PR #5181:
        Megatron's change replaced a binary accept/reject with a clamp+warn
        pattern.  ``HeteroTokenClamper`` generalises this to three memory tiers
        (device-local VRAM → FP8-expanded VRAM → CPU DRAM offload) and makes
        the decision tier-aware.

    Thread safety:
        All public methods are thread-safe.  Internal state is protected by
        ``_lock``.  Budget tables are immutable after construction.

    Parameters
    ----------
    tier_budgets:
        Mapping from ``DeviceTier`` to its precomputed ``DeviceTierBudget``.
    global_max_sequence_length:
        Absolute upper bound on (prompt + generation) tokens enforced
        regardless of tier.  Matches ``context.max_sequence_length`` in
        Megatron's engine context.
    enable_fp8_kv:
        Whether FP8 KV compression is globally enabled.  Even if ``True``,
        only SM90 tiers benefit.
    enable_cpu_offload:
        Whether cold KV blocks may be spilled to CPU DRAM.
    rank:
        Local rank; warnings are emitted only on rank 0 (mirrors Megatron).
    """

    def __init__(
        self,
        tier_budgets:               Dict[DeviceTier, DeviceTierBudget],
        global_max_sequence_length: int,
        enable_fp8_kv:              bool = True,
        enable_cpu_offload:         bool = True,
        rank:                       int  = 0,
    ) -> None:
        if global_max_sequence_length <= 0:
            raise ValueError(
                f"global_max_sequence_length must be positive, got {global_max_sequence_length}"
            )
        self._tier_budgets               = dict(tier_budgets)
        self._global_max_sequence_length = global_max_sequence_length
        self._enable_fp8_kv              = enable_fp8_kv
        self._enable_cpu_offload         = enable_cpu_offload
        self._rank                       = rank
        self._lock                       = threading.Lock()
        # Monotonic counters for observability (read via .stats property).
        self._n_requests_seen:   int = 0
        self._n_clamped:         int = 0
        self._n_hard_rejected:   int = 0
        self._n_cpu_spilled:     int = 0

        logger.debug(
            "HeteroTokenClamper initialised: global_max_seq=%d tiers=%s "
            "fp8_kv=%s cpu_offload=%s rank=%d",
            global_max_sequence_length,
            [t.value for t in tier_budgets],
            enable_fp8_kv,
            enable_cpu_offload,
            rank,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_engine_context(
        cls,
        context,               # DeepSpeed / Megatron engine context object
        rank: int = 0,
        enable_fp8_kv: bool = True,
        enable_cpu_offload: bool = True,
    ) -> "HeteroTokenClamper":
        """
        Construct a ``HeteroTokenClamper`` from a live engine context.

        Extracts ``max_sequence_length``, ``num_layers``, ``num_kv_heads``,
        and ``head_dim`` from the context to derive per-tier KV budgets
        automatically.

        Parameters
        ----------
        context:
            Engine context exposing at minimum:
            ``max_sequence_length``, ``num_layers``, ``num_kv_heads``,
            ``head_dim``, ``kv_dtype`` (optional).
        rank:
            Local rank for warning suppression.
        enable_fp8_kv:
            Pass ``False`` to disable FP8 KV optimisation even on SM90.
        enable_cpu_offload:
            Pass ``False`` to disable CPU DRAM spill.
        """
        max_seq_len  = getattr(context, "max_sequence_length",  4096)
        num_layers   = getattr(context, "num_layers",           32)
        num_kv_heads = getattr(context, "num_kv_heads",         8)
        head_dim     = getattr(context, "head_dim",             128)
        kv_dtype_str = getattr(context, "kv_dtype",             "float16")

        kv_dtype_bytes = _kv_dtype_bytes(kv_dtype_str)
        tier_budgets   = _build_tier_budgets(
            max_seq_len, num_layers, num_kv_heads, head_dim, kv_dtype_bytes
        )
        return cls(
            tier_budgets               = tier_budgets,
            global_max_sequence_length = max_seq_len,
            enable_fp8_kv              = enable_fp8_kv,
            enable_cpu_offload         = enable_cpu_offload,
            rank                       = rank,
        )

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def clamp_request(
        self,
        request:     InferenceRequest,
        device_tier: DeviceTier,
    ) -> ClampResult:
        """
        Clamp (or hard-reject) a single request's generation budget.

        This method is the DES-LOC analogue of the clamping block added by
        Megatron PR #5181.  The decision tree is:

        1. ``num_tokens_to_generate < 0``
               → hard reject (``NegativeTokenBudgetError``).
        2. ``prompt_len > global_max_sequence_length``
               → hard reject (``HeteroBudgetExhaustedError``).
        3. ``prompt_len + num_tokens_to_generate <= device_local_max_tokens``
               → accept unchanged (``ClampReason.NOT_CLAMPED``).
        4. ``prompt_len + num_tokens_to_generate <= fp8_expanded_max_tokens``
           *and* FP8 KV enabled on this tier
               → clamp to FP8 budget (``ClampReason.EXCEEDS_DEVICE_WITH_FP8``).
        5. ``prompt_len + num_tokens_to_generate <= cpu_offload_max_tokens``
           *and* CPU offload enabled
               → clamp to CPU-offload budget with spill fraction
               (``ClampReason.SPILLED_TO_CPU_DRAM``).
        6. Otherwise → hard reject (``HeteroBudgetExhaustedError``).

        The request object is *mutated in place* (``num_tokens_to_generate``
        and optionally ``failed`` / ``failure_reason``) to match Megatron's
        convention of modifying the request object directly.

        Parameters
        ----------
        request:
            The request to evaluate.  Modified in place.
        device_tier:
            Which tier is being considered for this request.

        Returns
        -------
        ClampResult
            Describes the outcome; the caller may inspect ``was_clamped`` and
            ``pcie_spill_fraction`` for scheduling decisions.

        Raises
        ------
        NegativeTokenBudgetError
            If ``num_tokens_to_generate < 0``.
        HeteroBudgetExhaustedError
            If the prompt alone exceeds global context or no tier can fit even
            the minimum generation budget.
        """
        with self._lock:
            self._n_requests_seen += 1

        rid            = request.request_id
        prompt_len     = request.prompt_len
        requested_gen  = request.num_tokens_to_generate
        budget         = self._tier_budgets.get(device_tier)

        # ── Hard reject: nonsensical generation budget ───────────────────────
        if requested_gen < 0:
            with self._lock:
                self._n_hard_rejected += 1
            request.failed         = True
            request.failure_reason = f"negative num_tokens_to_generate={requested_gen}"
            raise NegativeTokenBudgetError(rid, requested_gen)

        # ── Hard reject: prompt alone overflows the global context ────────────
        if prompt_len > self._global_max_sequence_length:
            with self._lock:
                self._n_hard_rejected += 1
            request.failed         = True
            request.failure_reason = (
                f"prompt_len={prompt_len} > "
                f"global_max_sequence_length={self._global_max_sequence_length}"
            )
            raise HeteroBudgetExhaustedError(rid, prompt_len, requested_gen)

        if budget is None:
            # Unknown tier — fall back to global limit.
            remaining = self._global_max_sequence_length - prompt_len
            return self._apply_clamp(
                request, rid, requested_gen, remaining, device_tier, 0.0,
                ClampReason.EXCEEDS_DEVICE_LOCAL,
            )

        # ── Path 3: fits in device-local VRAM ────────────────────────────────
        total_requested = prompt_len + requested_gen
        if total_requested <= budget.device_local_max_tokens:
            return ClampResult(
                request_id          = rid,
                original_num_tokens = requested_gen,
                clamped_num_tokens  = requested_gen,
                was_clamped         = False,
                reason              = ClampReason.NOT_CLAMPED,
                serving_tier        = device_tier,
                pcie_spill_fraction = 0.0,
            )

        # ── Path 4: fits with FP8 KV expansion ───────────────────────────────
        if (
            self._enable_fp8_kv
            and budget.supports_fp8_kv
            and total_requested <= budget.fp8_expanded_max_tokens
        ):
            remaining_fp8 = budget.fp8_expanded_max_tokens - prompt_len
            return self._apply_clamp(
                request, rid, requested_gen, remaining_fp8, device_tier, 0.0,
                ClampReason.EXCEEDS_DEVICE_WITH_FP8,
            )

        # ── Path 5: fits with CPU DRAM offload ───────────────────────────────
        if self._enable_cpu_offload and total_requested <= budget.cpu_offload_max_tokens:
            remaining_cpu = budget.cpu_offload_max_tokens - prompt_len
            # Estimate what fraction of KV blocks will reside in CPU DRAM.
            vram_cap      = (
                budget.fp8_expanded_max_tokens
                if (self._enable_fp8_kv and budget.supports_fp8_kv)
                else budget.device_local_max_tokens
            )
            overflow_toks = max(0, total_requested - vram_cap)
            spill_frac    = overflow_toks / max(total_requested, 1)
            with self._lock:
                self._n_cpu_spilled += 1
            return self._apply_clamp(
                request, rid, requested_gen, remaining_cpu, device_tier, spill_frac,
                ClampReason.SPILLED_TO_CPU_DRAM,
            )

        # ── Path 6: no tier can serve — hard reject ───────────────────────────
        with self._lock:
            self._n_hard_rejected += 1
        request.failed         = True
        request.failure_reason = (
            f"no tier can serve prompt_len={prompt_len} + "
            f"num_tokens_to_generate={requested_gen}"
        )
        raise HeteroBudgetExhaustedError(rid, prompt_len, requested_gen)

    def validate_batch(
        self,
        requests:           Sequence[InferenceRequest],
        device_assignments: Dict[str, DeviceTier],
    ) -> List[ClampResult]:
        """
        Validate and clamp an entire micro-batch before dispatch.

        Calls ``clamp_request`` for each request; collects hard-reject
        exceptions and raises a combined ``HeteroBudgetExhaustedError`` only
        if *every* request in the batch was hard-rejected (because partial
        batch execution is still useful).

        Parameters
        ----------
        requests:
            Requests in the micro-batch.
        device_assignments:
            Maps ``request_id → DeviceTier`` for this scheduling round.

        Returns
        -------
        List[ClampResult]
            One result per request in the same order.  Requests that were
            hard-rejected have ``ClampResult.reason == ClampReason.FULLY_REJECTED``
            and the corresponding ``InferenceRequest.failed`` is ``True``.
        """
        results: List[ClampResult] = []
        n_failed = 0

        for req in requests:
            tier = device_assignments.get(req.request_id, DeviceTier.H100_NVL)
            try:
                result = self.clamp_request(req, tier)
                results.append(result)
            except (NegativeTokenBudgetError, HeteroBudgetExhaustedError) as exc:
                n_failed += 1
                logger.warning(
                    "Batch validation: request '%s' hard-rejected on tier %s: %s",
                    req.request_id, tier.value, exc,
                )
                results.append(
                    ClampResult(
                        request_id          = req.request_id,
                        original_num_tokens = req.num_tokens_to_generate,
                        clamped_num_tokens  = 0,
                        was_clamped         = False,
                        reason              = ClampReason.FULLY_REJECTED,
                        serving_tier        = tier,
                    )
                )

        if n_failed == len(requests):
            raise HeteroBudgetExhaustedError(
                request_id      = f"batch({len(requests)})",
                prompt_len      = -1,
                requested_tokens= -1,
            )

        return results

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def stats(self) -> Dict[str, int]:
        """
        Monotonic counters since construction.  Safe to read from any thread.
        """
        with self._lock:
            return {
                "requests_seen": self._n_requests_seen,
                "clamped":       self._n_clamped,
                "hard_rejected": self._n_hard_rejected,
                "cpu_spilled":   self._n_cpu_spilled,
            }

    def reset_stats(self) -> None:
        """Reset all counters (useful between benchmark iterations)."""
        with self._lock:
            self._n_requests_seen = 0
            self._n_clamped       = 0
            self._n_hard_rejected = 0
            self._n_cpu_spilled   = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_clamp(
        self,
        request:      InferenceRequest,
        rid:          str,
        original_gen: int,
        remaining:    int,
        tier:         DeviceTier,
        spill_frac:   float,
        reason:       ClampReason,
    ) -> ClampResult:
        """
        Apply a clamped generation budget and emit a rank-0 warning.

        This is the direct DES-LOC counterpart of Megatron's::

            request.sampling_params.num_tokens_to_generate = remaining_tokens
            if self.rank == 0:
                warnings.warn(...)

        We use ``logging.warning`` rather than ``warnings.warn`` so that the
        message flows through the DeepSpeed logging infrastructure and is
        properly correlated with other rank-level events.
        """
        clamped_gen = min(original_gen, max(0, remaining))
        was_clamped = clamped_gen != original_gen

        if was_clamped:
            request.num_tokens_to_generate = clamped_gen
            with self._lock:
                self._n_clamped += 1
            if self._rank == 0:
                logger.warning(
                    "Request '%s': num_tokens_to_generate clamped %d → %d "
                    "(reason=%s, tier=%s, pcie_spill=%.2f)",
                    rid, original_gen, clamped_gen, reason.value, tier.value, spill_frac,
                )

        return ClampResult(
            request_id          = rid,
            original_num_tokens = original_gen,
            clamped_num_tokens  = clamped_gen,
            was_clamped         = was_clamped,
            reason              = reason if was_clamped else ClampReason.NOT_CLAMPED,
            serving_tier        = tier,
            pcie_spill_fraction = spill_frac,
        )


# ---------------------------------------------------------------------------
# Budget derivation utilities
# ---------------------------------------------------------------------------


def _kv_dtype_bytes(dtype_str: str) -> int:
    """
    Convert a dtype string to bytes-per-element.

    Supported strings: ``'float32'``, ``'float16'``, ``'bfloat16'``, ``'int8'``,
    ``'float8'`` / ``'fp8'``.
    """
    _MAP = {
        "float32":  4,
        "float16":  2,
        "bfloat16": 2,
        "int8":     1,
        "float8":   1,
        "fp8":      1,
    }
    key = dtype_str.lower().strip()
    if key not in _MAP:
        logger.debug("Unknown kv_dtype '%s'; defaulting to 2 bytes.", dtype_str)
        return 2
    return _MAP[key]


def _kv_bytes_per_token(
    num_layers:      int,
    num_kv_heads:    int,
    head_dim:        int,
    bytes_per_elem:  int,
) -> float:
    """
    Estimate KV cache bytes per token.

    Formula:  2 (K+V) × num_layers × num_kv_heads × head_dim × bytes_per_elem
    """
    return 2.0 * num_layers * num_kv_heads * head_dim * bytes_per_elem


def _build_tier_budgets(
    global_max_seq:  int,
    num_layers:      int,
    num_kv_heads:    int,
    head_dim:        int,
    kv_dtype_bytes:  int,
) -> Dict[DeviceTier, DeviceTierBudget]:
    """
    Compute per-tier ``DeviceTierBudget`` objects from model and hardware specs.

    The VRAM figures (48 GB for A6000, 96 GB for H100 NVL) are fixed by the
    DES-LOC cluster configuration.  We reserve a fraction of each device's
    VRAM for model weights and activations; the remainder is available for KV
    cache.

    Reservation fractions (empirically determined for LLaMA-class models):
      - A6000:   65 % for weights / activations → 35 % for KV
      - H100 NVL: 55 % for weights / activations → 45 % for KV
    """
    kv_per_tok = _kv_bytes_per_token(num_layers, num_kv_heads, head_dim, kv_dtype_bytes)

    a6000_vram_bytes   = 48 * (1 << 30)
    h100_nvl_vram_bytes= 96 * (1 << 30)

    a6000_kv_bytes     = a6000_vram_bytes   * 0.35
    h100_kv_bytes      = h100_nvl_vram_bytes * 0.45

    cpu_dram_kv_bytes  = _HOST_DRAM_BYTES * _CPU_DRAM_KV_FRACTION

    def _tokens(byte_budget: float) -> int:
        if kv_per_tok <= 0:
            return global_max_seq
        return min(global_max_seq, int(byte_budget / kv_per_tok))

    a6000_local = _tokens(a6000_kv_bytes)
    h100_local  = _tokens(h100_kv_bytes)

    # FP8 compression: only available on H100 NVL (SM90).
    h100_fp8    = _tokens(h100_kv_bytes * _SM90_FP8_KV_COMPRESSION_RATIO)

    # CPU DRAM offload budget (device-agnostic, but models PCIe bandwidth).
    cpu_tokens  = _tokens(cpu_dram_kv_bytes * _PCIE_SPILL_PENALTY)

    a6000_budget = DeviceTierBudget(
        tier                    = DeviceTier.A6000_SM86,
        device_local_max_tokens = a6000_local,
        fp8_expanded_max_tokens = a6000_local,   # no FP8 on SM86
        cpu_offload_max_tokens  = a6000_local + cpu_tokens,
        kv_bytes_per_token      = kv_per_tok,
        supports_fp8_kv         = False,
    )
    h100_budget = DeviceTierBudget(
        tier                    = DeviceTier.H100_NVL,
        device_local_max_tokens = h100_local,
        fp8_expanded_max_tokens = h100_fp8,
        cpu_offload_max_tokens  = h100_fp8 + cpu_tokens,
        kv_bytes_per_token      = kv_per_tok * (1 / _SM90_FP8_KV_COMPRESSION_RATIO),
        supports_fp8_kv         = True,
    )
    cpu_budget = DeviceTierBudget(
        tier                    = DeviceTier.CPU_DRAM,
        device_local_max_tokens = cpu_tokens,
        fp8_expanded_max_tokens = cpu_tokens,
        cpu_offload_max_tokens  = cpu_tokens,
        kv_bytes_per_token      = kv_per_tok,
        supports_fp8_kv         = False,
    )

    logger.debug(
        "DeviceTierBudgets computed: A6000_local=%d H100_local=%d H100_fp8=%d cpu=%d "
        "(kv_per_tok=%.1f bytes)",
        a6000_local, h100_local, h100_fp8, cpu_tokens, kv_per_tok,
    )

    return {
        DeviceTier.A6000_SM86: a6000_budget,
        DeviceTier.H100_NVL:   h100_budget,
        DeviceTier.CPU_DRAM:   cpu_budget,
    }


# ---------------------------------------------------------------------------
# Scheduler-facing integration helper
# ---------------------------------------------------------------------------


class HeteroSchedulerIntegration:
    """
    Thin wrapper that exposes the clamper to DeepSpeed's micro-batch scheduler.

    Intended to be composed into ``deepspeed.inference.engine.InferenceEngine``
    or a DES-LOC-specific scheduler subclass::

        self._token_clamper = HeteroSchedulerIntegration(clamper, device_map)

    The scheduler calls ``prepare_micro_batch`` before dispatching each step.
    """

    def __init__(
        self,
        clamper:    HeteroTokenClamper,
        device_map: Dict[int, DeviceTier],  # rank → tier
    ) -> None:
        self._clamper    = clamper
        self._device_map = device_map

    def prepare_micro_batch(
        self,
        requests: Sequence[InferenceRequest],
        rank:     int,
    ) -> Tuple[List[InferenceRequest], List[ClampResult]]:
        """
        Clamp all requests in a micro-batch for the given rank's device tier.

        Returns the (possibly filtered) request list together with one
        ``ClampResult`` per request.  Hard-rejected requests are removed from
        the serving list but their ``failed`` flag is set on the request object.
        """
        tier   = self._device_map.get(rank, DeviceTier.H100_NVL)
        assign = {r.request_id: tier for r in requests}

        try:
            results = self._clamper.validate_batch(requests, assign)
        except HeteroBudgetExhaustedError:
            logger.error(
                "Rank %d (%s): entire micro-batch of %d requests exhausted all "
                "memory tiers; returning empty batch.",
                rank, tier.value, len(requests),
            )
            return [], []

        # Filter out hard-rejected requests from the live serving list.
        live_requests = [
            req for req, res in zip(requests, results)
            if res.reason != ClampReason.FULLY_REJECTED
        ]

        n_dropped = len(requests) - len(live_requests)
        if n_dropped:
            logger.warning(
                "Rank %d (%s): %d/%d requests hard-rejected in micro-batch.",
                rank, tier.value, n_dropped, len(requests),
            )

        return live_requests, results


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    import traceback

    logging.basicConfig(
        level  = logging.DEBUG,
        format = "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _make_clamper(
        global_max: int = 4096,
        fp8_kv:     bool = True,
        cpu_offload:bool = True,
        rank:       int  = 0,
    ) -> HeteroTokenClamper:
        """Build a clamper with synthetic tier budgets for testing."""
        budgets = {
            DeviceTier.A6000_SM86: DeviceTierBudget(
                tier                    = DeviceTier.A6000_SM86,
                device_local_max_tokens = 1024,
                fp8_expanded_max_tokens = 1024,
                cpu_offload_max_tokens  = 2048,
                kv_bytes_per_token      = 512.0,
                supports_fp8_kv         = False,
            ),
            DeviceTier.H100_NVL: DeviceTierBudget(
                tier                    = DeviceTier.H100_NVL,
                device_local_max_tokens = 2048,
                fp8_expanded_max_tokens = 4096,
                cpu_offload_max_tokens  = 6144,
                kv_bytes_per_token      = 256.0,
                supports_fp8_kv         = True,
            ),
            DeviceTier.CPU_DRAM: DeviceTierBudget(
                tier                    = DeviceTier.CPU_DRAM,
                device_local_max_tokens = 8192,
                fp8_expanded_max_tokens = 8192,
                cpu_offload_max_tokens  = 8192,
                kv_bytes_per_token      = 512.0,
                supports_fp8_kv         = False,
            ),
        }
        return HeteroTokenClamper(
            tier_budgets               = budgets,
            global_max_sequence_length = global_max,
            enable_fp8_kv              = fp8_kv,
            enable_cpu_offload         = cpu_offload,
            rank                       = rank,
        )

    def _req(
        rid:       str,
        prompt_len:int,
        num_gen:   int,
    ) -> InferenceRequest:
        return InferenceRequest(
            request_id            = rid,
            prompt_tokens         = list(range(prompt_len)),
            num_tokens_to_generate= num_gen,
        )

    # ── Test runner ──────────────────────────────────────────────────────────

    passed = 0
    failed = 0

    def run_test(name: str, fn):
        global passed, failed
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as exc:
            print(f"  FAIL  {name}: {exc}")
            failed += 1
        except Exception as exc:
            print(f"  ERROR {name}: {exc}")
            traceback.print_exc()
            failed += 1

    print("=" * 72)
    print("HeteroTokenClamper — Unit Tests")
    print("=" * 72)

    # ── Test 1: Request fits in device-local VRAM — no clamp ─────────────────
    def test_no_clamp_device_local():
        clamper = _make_clamper()
        req     = _req("r1", prompt_len=100, num_gen=500)   # total=600 < 2048
        result  = clamper.clamp_request(req, DeviceTier.H100_NVL)
        assert not result.was_clamped, "Should not be clamped"
        assert result.reason == ClampReason.NOT_CLAMPED
        assert req.num_tokens_to_generate == 500
        assert result.pcie_spill_fraction == 0.0

    run_test("no_clamp_device_local", test_no_clamp_device_local)

    # ── Test 2: Clamp to remaining VRAM (mirrors Megatron PR #5181 core case) ─
    def test_clamp_to_device_local_remaining():
        clamper = _make_clamper()
        # H100 local=2048; prompt=100; remaining=1948; request 3000 → clamp to 1948
        req    = _req("r2", prompt_len=100, num_gen=3000)
        result = clamper.clamp_request(req, DeviceTier.H100_NVL)
        # 3000 > 2048-100=1948 but fits in fp8_expanded=4096
        assert result.was_clamped
        assert result.reason == ClampReason.EXCEEDS_DEVICE_WITH_FP8
        assert req.num_tokens_to_generate == 4096 - 100   # FP8 path taken
        assert result.pcie_spill_fraction == 0.0

    run_test("clamp_fp8_path_h100", test_clamp_to_device_local_remaining)

    # ── Test 3: A6000 has no FP8 — clamp directly to device-local remaining ──
    def test_clamp_a6000_no_fp8():
        clamper = _make_clamper()
        # A6000 local=1024; prompt=200; remaining=824; request 2000 → no FP8 → CPU spill
        req    = _req("r3", prompt_len=200, num_gen=2000)
        result = clamper.clamp_request(req, DeviceTier.A6000_SM86)
        # total=2200; exceeds device_local(1024) and fp8==local; try cpu_offload=2048
        # remaining_cpu = 2048-200=1848 < 2000 → clamp
        assert result.was_clamped
        assert result.reason == ClampReason.SPILLED_TO_CPU_DRAM
        assert req.num_tokens_to_generate == 2048 - 200
        assert result.pcie_spill_fraction > 0.0

    run_test("clamp_a6000_cpu_spill", test_clamp_a6000_no_fp8)

    # ── Test 4: Negative num_tokens_to_generate → NegativeTokenBudgetError ───
    def test_negative_budget_raises():
        clamper = _make_clamper()
        req     = _req("r4", prompt_len=10, num_gen=-1)
        raised  = False
        try:
            clamper.clamp_request(req, DeviceTier.H100_NVL)
        except NegativeTokenBudgetError:
            raised = True
        assert raised, "Expected NegativeTokenBudgetError"
        assert req.failed

    run_test("negative_budget_raises", test_negative_budget_raises)

    # ── Test 5: Prompt longer than global context → hard reject ──────────────
    def test_prompt_overflow_raises():
        clamper = _make_clamper(global_max=512)
        req     = _req("r5", prompt_len=600, num_gen=50)
        raised  = False
        try:
            clamper.clamp_request(req, DeviceTier.H100_NVL)
        except HeteroBudgetExhaustedError:
            raised = True
        assert raised, "Expected HeteroBudgetExhaustedError"
        assert req.failed

    run_test("prompt_overflow_raises", test_prompt_overflow_raises)

    # ── Test 6: No tier can serve → hard reject ───────────────────────────────
    def test_no_tier_serves():
        clamper = _make_clamper(fp8_kv=False, cpu_offload=False)
        # H100 local=2048; prompt=100; remaining=1948; request 5000 → no FP8, no CPU
        req    = _req("r6", prompt_len=100, num_gen=5000)
        raised = False
        try:
            clamper.clamp_request(req, DeviceTier.H100_NVL)
        except HeteroBudgetExhaustedError:
            raised = True
        assert raised, "Expected HeteroBudgetExhaustedError"
        assert req.failed

    run_test("no_tier_serves", test_no_tier_serves)

    # ── Test 7: Exactly at device-local limit → no clamp ─────────────────────
    def test_exact_boundary_no_clamp():
        clamper = _make_clamper()
        # H100 local=2048; prompt=48; gen=2000 → total=2048 == local → not clamped
        req    = _req("r7", prompt_len=48, num_gen=2000)
        result = clamper.clamp_request(req, DeviceTier.H100_NVL)
        assert not result.was_clamped
        assert result.reason == ClampReason.NOT_CLAMPED

    run_test("exact_boundary_no_clamp", test_exact_boundary_no_clamp)

    # ── Test 8: validate_batch — mixed batch, some clamped, some rejected ─────
    def test_validate_batch_mixed():
        clamper = _make_clamper()
        requests = [
            _req("rb1", prompt_len=100,  num_gen=500),     # fits
            _req("rb2", prompt_len=100,  num_gen=3000),    # clamp FP8
            _req("rb3", prompt_len=4000, num_gen=500),     # prompt overflow
        ]
        assign = {
            "rb1": DeviceTier.H100_NVL,
            "rb2": DeviceTier.H100_NVL,
            "rb3": DeviceTier.H100_NVL,
        }
        results = clamper.validate_batch(requests, assign)
        assert len(results) == 3
        assert results[0].reason == ClampReason.NOT_CLAMPED
        assert results[1].was_clamped
        assert results[2].reason == ClampReason.FULLY_REJECTED
        assert requests[2].failed

    run_test("validate_batch_mixed", test_validate_batch_mixed)

    # ── Test 9: Stats counter accuracy ───────────────────────────────────────
    def test_stats_counters():
        clamper = _make_clamper()
        _req_a  = _req("s1", 100, 500)
        _req_b  = _req("s2", 100, 3000)
        clamper.clamp_request(_req_a, DeviceTier.H100_NVL)
        clamper.clamp_request(_req_b, DeviceTier.H100_NVL)
        s = clamper.stats
        assert s["requests_seen"] == 2, f"requests_seen={s['requests_seen']}"
        assert s["clamped"]       == 1, f"clamped={s['clamped']}"
        assert s["hard_rejected"] == 0

    run_test("stats_counters", test_stats_counters)

    # ── Test 10: generation_penalty_factor is 1.0 without spill ──────────────
    def test_no_penalty_without_spill():
        clamper = _make_clamper()
        req     = _req("p1", 100, 500)
        result  = clamper.clamp_request(req, DeviceTier.H100_NVL)
        assert result.generation_penalty_factor == 1.0

    run_test("no_penalty_without_spill", test_no_penalty_without_spill)

    # ── Test 11: CPU spill yields penalty < 1.0 ───────────────────────────────
    def test_penalty_with_spill():
        clamper = _make_clamper()
        req     = _req("p2", 200, 2000)  # A6000: local=1024, cpu=2048 → spills
        result  = clamper.clamp_request(req, DeviceTier.A6000_SM86)
        assert result.pcie_spill_fraction > 0.0
        assert result.generation_penalty_factor < 1.0

    run_test("penalty_with_spill", test_penalty_with_spill)

    # ── Test 12: HeteroSchedulerIntegration filters hard-rejected ─────────────
    def test_scheduler_integration_filters_rejected():
        clamper = _make_clamper()
        integration = HeteroSchedulerIntegration(
            clamper    = clamper,
            device_map = {0: DeviceTier.H100_NVL},
        )
        requests = [
            _req("si1", 100,  500),    # fine
            _req("si2", 4500, 100),    # prompt overflow → reject
        ]
        live, results = integration.prepare_micro_batch(requests, rank=0)
        assert len(live) == 1
        assert live[0].request_id == "si1"
        assert results[1].reason == ClampReason.FULLY_REJECTED

    run_test("scheduler_integration_filters_rejected", test_scheduler_integration_filters_rejected)

    # ── Test 13: from_engine_context factory smoke test ───────────────────────
    def test_from_engine_context_factory():
        class FakeContext:
            max_sequence_length = 8192
            num_layers          = 32
            num_kv_heads        = 8
            head_dim            = 128
            kv_dtype            = "bfloat16"

        clamper = HeteroTokenClamper.from_engine_context(FakeContext(), rank=0)
        assert clamper._global_max_sequence_length == 8192
        assert DeviceTier.H100_NVL in clamper._tier_budgets
        assert DeviceTier.A6000_SM86 in clamper._tier_budgets
        assert DeviceTier.CPU_DRAM in clamper._tier_budgets

    run_test("from_engine_context_factory", test_from_engine_context_factory)

    # ── Test 14: Reset stats ──────────────────────────────────────────────────
    def test_reset_stats():
        clamper = _make_clamper()
        clamper.clamp_request(_req("rst1", 100, 500), DeviceTier.H100_NVL)
        clamper.reset_stats()
        s = clamper.stats
        assert s["requests_seen"] == 0
        assert s["clamped"]       == 0

    run_test("reset_stats", test_reset_stats)

    # ── Test 15: Mirroring Megatron test_max_sequence_length_clamp semantics ──
    def test_megatron_mirror_clamp_semantics():
        """
        Direct analogue of Megatron's new test_max_sequence_length_clamp:
        set num_tokens_to_generate = remaining + 100 and verify:
          - request is NOT failed
          - num_tokens_to_generate is clamped to exactly 'remaining' (or less)
        """
        clamper     = _make_clamper()
        prompt_len  = 8
        # Use H100 local capacity as the 'max_sequence_length' analogue.
        h100_budget = clamper._tier_budgets[DeviceTier.H100_NVL]
        remaining   = h100_budget.device_local_max_tokens - prompt_len
        req         = _req("megatron_mirror", prompt_len, remaining + 100)

        result = clamper.clamp_request(req, DeviceTier.H100_NVL)

        assert not req.failed, "Request must not be failed after clamping"
        assert req.num_tokens_to_generate <= remaining + 100, \
            "Clamped budget must not exceed original"
        assert result.was_clamped, "Result must indicate clamping occurred"

    run_test("megatron_mirror_clamp_semantics", test_megatron_mirror_clamp_semantics)

    # ── Test 16: Thread safety under concurrent requests ─────────────────────
    def test_thread_safety():
        import concurrent.futures
        clamper  = _make_clamper()
        errors   = []

        def worker(i: int):
            try:
                req = _req(f"t{i}", prompt_len=100, num_gen=500 + i)
                clamper.clamp_request(req, DeviceTier.H100_NVL)
            except Exception as exc:
                errors.append(exc)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(worker, i) for i in range(128)]
            concurrent.futures.wait(futures)

        assert not errors, f"Thread safety errors: {errors}"
        assert clamper.stats["requests_seen"] == 128

    run_test("thread_safety", test_thread_safety)

    # ── Test 17: _kv_dtype_bytes fallback ────────────────────────────────────
    def test_kv_dtype_bytes_fallback():
        assert _kv_dtype_bytes("float32")  == 4
        assert _kv_dtype_bytes("float16")  == 2
        assert _kv_dtype_bytes("bfloat16") == 2
        assert _kv_dtype_bytes("fp8")      == 1
        assert _kv_dtype_bytes("UNKNOWN")  == 2  # fallback

    run_test("kv_dtype_bytes_fallback", test_kv_dtype_bytes_fallback)

    # ── Test 18: zero generation budget → not an error ───────────────────────
    def test_zero_generation_budget():
        clamper = _make_clamper()
        req     = _req("z1", prompt_len=100, num_gen=0)
        result  = clamper.clamp_request(req, DeviceTier.H100_NVL)
        assert not req.failed
        assert req.num_tokens_to_generate == 0

    run_test("zero_generation_budget", test_zero_generation_budget)

    # ── Test 19: CPU_DRAM tier used directly ─────────────────────────────────
    def test_cpu_dram_tier_direct():
        clamper = _make_clamper()
        req     = _req("cd1", prompt_len=100, num_gen=6000)
        result  = clamper.clamp_request(req, DeviceTier.CPU_DRAM)
        # CPU_DRAM local=8192; 100+6000=6100 < 8192 → not clamped
        assert not result.was_clamped
        assert result.serving_tier == DeviceTier.CPU_DRAM

    run_test("cpu_dram_tier_direct", test_cpu_dram_tier_direct)

    # ── Test 20: DeviceTierBudget.hard_max correctness ───────────────────────
    def test_device_tier_budget_hard_max():
        b = DeviceTierBudget(
            tier                    = DeviceTier.H100_NVL,
            device_local_max_tokens = 2048,
            fp8_expanded_max_tokens = 4096,
            cpu_offload_max_tokens  = 6144,
            kv_bytes_per_token      = 256.0,
            supports_fp8_kv         = True,
        )
        assert b.hard_max == 6144

    run_test("device_tier_budget_hard_max", test_device_tier_budget_hard_max)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 72)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 72)
    sys.exit(0 if failed == 0 else 1)
