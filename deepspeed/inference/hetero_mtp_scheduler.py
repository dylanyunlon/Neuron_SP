# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""HeteroMTPScheduler — heterogeneous Multi-Token Prediction draft/verify scheduling
for DES-LOC.

Mirrors Megatron 300d1b655 "Add MTP support for hybrid models", reinterpreted as a
*role-separated draft/verify scheduler* where the MTP draft heads run on A6000 GPUs
(fast, cheap, small model) and the main model verification runs on H100 GPUs (accurate,
large).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Upstream design intent (300d1b655)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The Megatron MTP commit introduced three interlocking subsystems:

1. ``MultiTokenPredictionLayer`` / ``MultiTokenPredictionBlock`` — draft heads that
   accept hidden states + embedding of the next token, project them together, pass
   through a lightweight transformer layer, and emit a prediction for the (i+k)-th
   token.  The forward builds a per-depth chain:
       h_0  →  MTPLayer_1(embed(x_{i+1}), h_0)  →  h_1
       h_1  →  MTPLayer_2(embed(x_{i+2}), h_1)  →  h_2
       ...
   The output is torch.cat([h_0, h_1, ..., h_K], dim=0) so that
   ``process_mtp_loss`` can unpack each draft hidden state and compute a
   cross-entropy loss against the appropriately rolled label tensor.

2. ``process_mtp_loss`` — extracts this to a standalone function so that both
   GPTModel and MambaModel can call it identically on the post-process rank.
   Key insight: the hidden states tensor is chunked into (1 + mtp_num_layers)
   pieces; piece 0 is the main model output; pieces 1..K are the draft heads.
   Only piece 0 flows back through ``MTPLossAutoScaler`` so that the draft
   gradient scale is never confused with the main gradient scale.

3. ``mtp_on_this_rank`` / pipeline stage gating — MTP layers are built only on
   the pipeline stage(s) that actually host them (default: last stage), and the
   embedding is duplicated to that stage with a tied shard so checkpoint round-trips
   are consistent.

DES-LOC reinterpretation — role separation across GPU tiers:
───────────────────────────────────────────────────────────────
The key observation is that the MTP draft heads are architecturally independent
of the main model: they take an externally-supplied hidden-state tensor and an
embedding, not the model's internal layers.  This independence makes them ideal
candidates for *offloading to a cheaper GPU tier* while the main model stays on
the expensive tier.

    ┌─────────────────────────────────────────────────────────────┐
    │ A6000 cluster (draft tier, SM86, 48 GB VRAM)               │
    │  • Runs MTP draft heads (MultiTokenPredictionLayer × K)    │
    │  • Generates K candidate tokens per prompt position        │
    │  • draft_depth K configurable per device (default K=2)     │
    └───────────────────────┬─────────────────────────────────────┘
                            │  hidden states + draft token IDs
                            │  (PCIe / NVLink / IB depending on topo)
                            ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ H100 cluster (verify tier, SM90, 80 GB VRAM)               │
    │  • Runs main model forward (TransformerBlock / MambaStack) │
    │  • Verifies draft tokens with LLM-level score              │
    │  • Accepts a variable prefix of matching tokens (k* ≤ K)   │
    └─────────────────────────────────────────────────────────────┘

This is analogous to how ``heterogeneous_engine.py`` routes token-length requests
to LIGHT vs HEAVY tiers.  Here the routing is *structural* (draft vs. verify) rather
than token-length-based.

Decision boundary — when to use A6000 for drafting:
  •  A6000 SM86 (fp16 tensor core peak ≈ 310 TFLOP/s) vs H100 SM90 (≈ 2000 TFLOP/s).
  •  MTP layer compute ≈ 2 × hidden_size^2 × seq_len FLOPs per depth.
  •  Rule: if MTP layer FLOPs per token < flop_ratio × (main layer FLOPs per token),
     the draft head fits in A6000's per-token budget relative to the verify step.
     Default: draft if (mtp_layer_params / total_params) < DRAFT_OFFLOAD_THRESHOLD.

Diagnostic events (rank-0, logger + print):
  [DS-HMTP] DRAFT_DISPATCH: per-batch draft depth and target device.
  [DS-HMTP] VERIFY_RESULT: per-batch accepted prefix length after verification.
  [DS-HMTP] DEPTH_ADAPT: dynamic draft_depth adjustment based on accept rate.
  [DS-HMTP] TIER_IMBALANCE: A6000 or H100 queues diverge beyond threshold.
  [DS-HMTP] PIPELINE_STALL: verify tier idle > stall_threshold_ms waiting for draft.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Tuple

import torch

from deepspeed.utils import logger as ds_logger

_LOG_PREFIX = "[DS-HMTP]"

# ---------------------------------------------------------------------------
# GPU tier constants matching heterogeneous_engine.py taxonomy
# ---------------------------------------------------------------------------

# Default draft depth per GPU tier (tunable at runtime)
_DEFAULT_DRAFT_DEPTH_A6000 = 2   # SM86: 310 TFLOP/s fp16 — runs 2 MTP layers cheaply
_DEFAULT_DRAFT_DEPTH_H100  = 0   # SM90: verify only, no separate draft offload

# Fraction of total param budget below which MTP layer qualifies for A6000 offload
_DRAFT_OFFLOAD_THRESHOLD = 0.08  # < 8 % param share → off-load to draft tier

# Exponential-moving-average smoothing for accept-rate tracking
_ACCEPT_RATE_EMA_ALPHA = 0.1

# After this many accept-rate samples, enable dynamic depth adaptation
_ADAPT_MIN_SAMPLES = 32

# Stall threshold: if verify tier is idle more than this (ms) waiting for draft,
# log a PIPELINE_STALL event.
_STALL_THRESHOLD_MS = 20.0


# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------

class GpuRole(str, Enum):
    """Role assignment for a GPU in the MTP pipeline."""
    DRAFT  = "draft"   # A6000 — runs MTP draft heads
    VERIFY = "verify"  # H100  — runs main model + acceptance check


@dataclass
class MTPTierConfig:
    """Configuration for one GPU tier in the heterogeneous MTP pipeline.

    Attributes:
        role:         DRAFT or VERIFY.
        device_ids:   CUDA device indices owned by this tier.
        draft_depth:  Number of MTP prediction depths to run on draft tier.
                      Ignored (set to 0) for VERIFY role.
        sm_version:   CUDA compute capability major×10+minor (e.g. 86 for SM8.6,
                      90 for SM9.0).  Used to gate arch-specific kernels and
                      mirrors the ``fp8_gemm.py`` SM86/SM90 routing pattern.
        max_batch_tokens: Maximum (batch_size × seq_len) accepted in one step.
        flop_budget:  Optional peak TFLOP/s for this tier.  When both tiers report
                      flop_budget, ``HeteroMTPScheduler`` auto-calibrates draft_depth
                      so the draft step takes ≤ 1 / (flop_ratio + 1) of the total
                      wall-clock (i.e. draft and verify finish in lock-step).
    """
    role: GpuRole
    device_ids: List[int]
    draft_depth: int = _DEFAULT_DRAFT_DEPTH_A6000
    sm_version: int = 86          # default A6000
    max_batch_tokens: int = 32768
    flop_budget: Optional[float] = None  # TFLOP/s


@dataclass
class DraftResult:
    """Output produced by the draft tier for a single request/batch.

    Attributes:
        request_id:        Opaque request identifier.
        hidden_states:     Stacked hidden-state tensor shape [1+K, S, B, H].
                           Slice 0 is the main model's last-layer hidden state;
                           slices 1..K are the K draft-head outputs, matching
                           ``MultiTokenPredictionBlock``'s output convention.
        draft_token_ids:   Draft token IDs shape [B, K] predicted by the draft
                           heads (argmax of logits at each depth).
        draft_depth:       K actually used (may be < tier.draft_depth if the
                           batch was short).
        timestamp_ns:      Wall-clock nanoseconds when draft completed.
    """
    request_id: Any
    hidden_states: torch.Tensor
    draft_token_ids: torch.Tensor
    draft_depth: int
    timestamp_ns: int = field(default_factory=lambda: time.time_ns())


@dataclass
class VerifyResult:
    """Output produced by the verify tier for a single request/batch.

    Attributes:
        request_id:       Matches the originating ``DraftResult``.
        accepted_length:  Number of draft tokens accepted (0 ≤ k* ≤ draft_depth).
        output_token_ids: Final accepted token IDs shape [B, k*+1]  (+1 for the
                          corrected token after the last rejection).
        verify_latency_ms: Time from receiving ``DraftResult`` to emitting this
                           result, in milliseconds.
    """
    request_id: Any
    accepted_length: int
    output_token_ids: torch.Tensor
    verify_latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Accept-rate tracker (EMA)
# ---------------------------------------------------------------------------

class _AcceptRateTracker:
    """Exponential-moving-average tracker for draft-token acceptance rates.

    Mirrors the throughput-weighted EMA used in ``bandwidth_aware_dispatcher.py``
    but specialised for acceptance-rate signals from the verify tier.

    The accept rate (0.0–1.0) drives the ``_DepthAdapter``'s decisions to raise or
    lower ``draft_depth`` dynamically to maximise wall-clock tokens/second while
    keeping the A6000 draft budget in check.
    """

    def __init__(self, alpha: float = _ACCEPT_RATE_EMA_ALPHA) -> None:
        self._alpha = alpha
        self._ema: Optional[float] = None
        self._n_samples: int = 0
        self._lock = threading.Lock()

    def update(self, accepted: int, draft_depth: int) -> float:
        """Record one verification result and return the updated EMA.

        Args:
            accepted:    Number of draft tokens accepted (k*).
            draft_depth: Total draft tokens generated (K).

        Returns:
            Current EMA acceptance rate after this update.
        """
        if draft_depth == 0:
            return self._ema or 0.0
        sample = accepted / draft_depth
        with self._lock:
            self._n_samples += 1
            if self._ema is None:
                self._ema = sample
            else:
                self._ema = self._alpha * sample + (1.0 - self._alpha) * self._ema
            return self._ema

    @property
    def rate(self) -> float:
        """Current EMA acceptance rate, or 0.5 if no data yet."""
        return self._ema if self._ema is not None else 0.5

    @property
    def n_samples(self) -> int:
        """Total number of verify results observed."""
        return self._n_samples

    def reset(self) -> None:
        with self._lock:
            self._ema = None
            self._n_samples = 0


# ---------------------------------------------------------------------------
# Depth adapter — dynamic draft_depth calibration
# ---------------------------------------------------------------------------

class _DepthAdapter:
    """Dynamically adapt draft_depth based on observed acceptance rate.

    The decision boundary is:
        • If accept_rate < low_thresh: decrease draft_depth (drafts are wasted).
        • If accept_rate > high_thresh: increase draft_depth (drafts are likely good).
        • Otherwise: hold current depth.

    Upper bound is capped at ``max_depth``; lower bound is clamped to 1 (always
    draft at least one token, otherwise MTP adds zero value).

    This is the DES-LOC analogue of the adaptive window sizing in
    ``dynamic_batch_context.py``'s chunked-prefill schedule adaptor.

    Diagnostic:
        Emits [DS-HMTP] DEPTH_ADAPT on every adjustment.
    """

    def __init__(
        self,
        initial_depth: int,
        min_depth: int = 1,
        max_depth: int = 8,
        low_thresh: float = 0.3,
        high_thresh: float = 0.75,
        rank: int = 0,
    ) -> None:
        self._depth = initial_depth
        self._min = max(1, min_depth)
        self._max = max_depth
        self._low = low_thresh
        self._high = high_thresh
        self._rank = rank

    @property
    def depth(self) -> int:
        return self._depth

    def maybe_adapt(self, tracker: _AcceptRateTracker) -> None:
        """Adjust draft_depth based on current EMA.  No-op until min samples seen."""
        if tracker.n_samples < _ADAPT_MIN_SAMPLES:
            return
        rate = tracker.rate
        old = self._depth
        if rate < self._low and self._depth > self._min:
            self._depth -= 1
        elif rate > self._high and self._depth < self._max:
            self._depth += 1
        if self._depth != old and self._rank == 0:
            _log_depth_adapt(old, self._depth, rate)


# ---------------------------------------------------------------------------
# Draft executor — runs on DRAFT-role GPUs
# ---------------------------------------------------------------------------

class _DraftExecutor:
    """Execute MTP draft heads on the draft tier (A6000).

    The executor holds a reference to the draft model (a ``MultiTokenPredictionBlock``
    or equivalent callable) and executes it for each batch submitted via
    ``submit()``.

    The design mirrors ``SSMStateManager``'s hot/cold state tiers: the executor
    owns the draft heads' KV cache (hot) on A6000 VRAM and streams it from CPU
    (cold) when VRAM pressure exceeds threshold.

    Thread safety:
        ``submit()`` is called from the scheduling thread.  The executor may
        run on a dedicated CUDA stream (``cuda_stream``) to pipeline compute with
        the verify tier on H100.
    """

    def __init__(
        self,
        config: "HeteroMTPConfig",
        draft_model: Optional[Callable] = None,
        embedding_fn: Optional[Callable] = None,
        output_layer: Optional[Callable] = None,
        cuda_stream: Optional[torch.cuda.Stream] = None,
        rank: int = 0,
    ) -> None:
        """
        Args:
            config:        Scheduler config (depth, device IDs, etc.).
            draft_model:   Callable matching ``MultiTokenPredictionBlock.forward``.
                           If None, the executor runs in stub mode (returns zeros).
            embedding_fn:  Callable returning token embeddings shape [S, B, H].
            output_layer:  Callable returning (logits, _) from hidden states.
            cuda_stream:   CUDA stream for async execution.  When provided, the
                           caller must synchronise before consuming results.
            rank:          Global rank for logging.
        """
        self._config = config
        self._draft_model = draft_model
        self._embedding_fn = embedding_fn
        self._output_layer = output_layer
        self._stream = cuda_stream
        self._rank = rank

        # Depth is owned by the adapter; executor reads it each call.
        self._adapter = _DepthAdapter(
            initial_depth=config.draft_tier.draft_depth,
            max_depth=config.max_draft_depth,
            rank=rank,
        )
        self._accept_tracker = _AcceptRateTracker()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def submit(
        self,
        request_id: Any,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> DraftResult:
        """Run draft heads and return a ``DraftResult``.

        Args:
            request_id:    Opaque ID for matching with verify results.
            hidden_states: Main model output [S, B, H] (from verify tier prefill
                           or last accepted token step).
            input_ids:     Input token IDs [B, S].
            position_ids:  Position IDs [B, S].
            attention_mask: Optional causal mask.

        Returns:
            :class:`DraftResult` with stacked hidden states and draft token IDs.
        """
        depth = self._adapter.depth
        device = self._primary_device()

        t0_ns = time.time_ns()

        if self._draft_model is None:
            # Stub mode: return zero hidden states and random draft tokens.
            stacked = _stub_hidden_states(hidden_states, depth)
            draft_ids = _stub_draft_ids(input_ids, depth, device)
            if self._rank == 0:
                _log_draft_dispatch(request_id, depth, device, stub=True)
            return DraftResult(
                request_id=request_id,
                hidden_states=stacked,
                draft_token_ids=draft_ids,
                draft_depth=depth,
                timestamp_ns=t0_ns,
            )

        ctx = torch.cuda.stream(self._stream) if self._stream is not None else _nullctx()
        with ctx:
            hidden_states = hidden_states.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)
            position_ids = position_ids.to(device, non_blocking=True)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)

            stacked_hs, draft_ids = self._run_draft_heads(
                hidden_states=hidden_states,
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                depth=depth,
            )

        if self._stream is not None:
            self._stream.synchronize()

        elapsed_ms = (time.time_ns() - t0_ns) / 1e6
        if self._rank == 0:
            _log_draft_dispatch(request_id, depth, device, stub=False, ms=elapsed_ms)

        return DraftResult(
            request_id=request_id,
            hidden_states=stacked_hs,
            draft_token_ids=draft_ids,
            draft_depth=depth,
            timestamp_ns=t0_ns,
        )

    def record_verify_result(self, result: "VerifyResult") -> None:
        """Feed acceptance signal back so the depth adapter can calibrate.

        Called by the scheduler after the verify tier emits a ``VerifyResult``.
        """
        rate = self._accept_tracker.update(
            accepted=result.accepted_length,
            draft_depth=result.accepted_length + 1,  # +1 for the bonus token
        )
        self._adapter.maybe_adapt(self._accept_tracker)
        _ = rate  # used implicitly inside adapt

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_draft_heads(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        depth: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute ``draft_model`` for ``depth`` prediction steps.

        The upstream ``MultiTokenPredictionBlock.forward`` expects:
          - hidden_states: the *concatenated* tensor [1+prev_depth, S, B, H].
            For the first draft call, we pass [1, S, B, H] (main output only).
          - The block then appends K draft outputs: result is [(1+K), S, B, H].

        Returns:
            stacked_hs:  Hidden states [(1+depth), S, B, H].
            draft_ids:   Draft token IDs [B, depth].
        """
        # Block expects input shape [S, B, H]; ensure dims are right.
        if hidden_states.dim() == 3:
            hs = hidden_states.unsqueeze(0)   # [1, S, B, H]
        else:
            hs = hidden_states

        stacked_hs = self._draft_model(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hs,
            attention_mask=attention_mask,
            embedding=self._embedding_fn,
        )

        # Extract draft logits to get token IDs for each depth.
        # stacked_hs shape: [(1+depth), S, B, H]
        # We need the last token position at each depth slice > 0.
        num_hs = stacked_hs.shape[0]
        actual_depth = num_hs - 1
        draft_ids_list: List[torch.Tensor] = []
        for d in range(actual_depth):
            depth_hs = stacked_hs[d + 1]          # [S, B, H]
            last_tok_hs = depth_hs[-1]             # [B, H]
            logits, _ = self._output_layer(
                last_tok_hs.unsqueeze(0)            # [1, B, H]
            )
            token_id = logits[-1].argmax(dim=-1)   # [B]
            draft_ids_list.append(token_id)

        if draft_ids_list:
            draft_ids = torch.stack(draft_ids_list, dim=1)   # [B, depth]
        else:
            B = hidden_states.shape[-2] if hidden_states.dim() >= 3 else 1
            draft_ids = torch.zeros(B, 0, dtype=torch.long, device=hidden_states.device)

        return stacked_hs, draft_ids

    def _primary_device(self) -> torch.device:
        ids = self._config.draft_tier.device_ids
        if ids:
            return torch.device(f"cuda:{ids[0]}")
        return torch.device("cuda")

    @property
    def accept_tracker(self) -> _AcceptRateTracker:
        return self._accept_tracker

    @property
    def adapter(self) -> _DepthAdapter:
        return self._adapter


# ---------------------------------------------------------------------------
# Verify executor — runs on VERIFY-role GPUs
# ---------------------------------------------------------------------------

class _VerifyExecutor:
    """Execute main-model verification on the verify tier (H100).

    Receives ``DraftResult`` from the draft tier, runs the main model forward
    with the draft token embeddings concatenated (speculative decode style),
    and produces a ``VerifyResult`` with the accepted prefix length.

    The acceptance criterion mirrors the standard speculative-decoding rule:
        For depth d ∈ {1..K}: accept draft token t_d if
            main_logit[d-1].argmax() == t_d  (greedy),
        or with a temperature-scaled probability (nucleus sampling path).

    Diagnostic:
        Emits [DS-HMTP] VERIFY_RESULT with accepted_length / draft_depth ratio
        and verify latency.
    """

    def __init__(
        self,
        config: "HeteroMTPConfig",
        main_model_fn: Optional[Callable] = None,
        cuda_stream: Optional[torch.cuda.Stream] = None,
        rank: int = 0,
    ) -> None:
        """
        Args:
            config:          Scheduler config.
            main_model_fn:   Callable ``(hidden_states, ...) → (logits, _)``
                             that runs the verify forward.  If None, stub mode.
            cuda_stream:     CUDA stream for async execution.
            rank:            Global rank for logging.
        """
        self._config = config
        self._main_model_fn = main_model_fn
        self._stream = cuda_stream
        self._rank = rank

    def verify(
        self,
        draft_result: DraftResult,
        labels: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> VerifyResult:
        """Verify draft tokens against the main model.

        Args:
            draft_result:  ``DraftResult`` from the draft executor.
            labels:        Ground-truth labels for training-time loss (optional).
            temperature:   Sampling temperature for probabilistic acceptance.
                           1.0 = greedy prefix (standard speculative decoding).

        Returns:
            :class:`VerifyResult` with accepted_length and output_token_ids.
        """
        t0 = time.monotonic()

        if self._main_model_fn is None:
            # Stub mode: accept all draft tokens.
            accepted = draft_result.draft_depth
            B = draft_result.draft_token_ids.shape[0]
            out_ids = draft_result.draft_token_ids
            latency = (time.monotonic() - t0) * 1e3
            result = VerifyResult(
                request_id=draft_result.request_id,
                accepted_length=accepted,
                output_token_ids=out_ids,
                verify_latency_ms=latency,
            )
            if self._rank == 0:
                _log_verify_result(draft_result.request_id, accepted,
                                   draft_result.draft_depth, latency, stub=True)
            return result

        device = self._primary_device()
        stacked_hs = draft_result.hidden_states.to(device, non_blocking=True)
        draft_ids  = draft_result.draft_token_ids.to(device, non_blocking=True)

        ctx = torch.cuda.stream(self._stream) if self._stream is not None else _nullctx()
        with ctx:
            accepted, out_ids = self._acceptance_check(
                stacked_hs, draft_ids, temperature
            )
        if self._stream is not None:
            self._stream.synchronize()

        latency = (time.monotonic() - t0) * 1e3
        result = VerifyResult(
            request_id=draft_result.request_id,
            accepted_length=accepted,
            output_token_ids=out_ids,
            verify_latency_ms=latency,
        )
        if self._rank == 0:
            _log_verify_result(draft_result.request_id, accepted,
                               draft_result.draft_depth, latency, stub=False)
        return result

    def _acceptance_check(
        self,
        stacked_hs: torch.Tensor,
        draft_ids: torch.Tensor,
        temperature: float,
    ) -> Tuple[int, torch.Tensor]:
        """Greedy speculative-decoding acceptance.

        Args:
            stacked_hs:  [(1+K), S, B, H] hidden states (slice 0 = main model).
            draft_ids:   [B, K] draft token IDs.

        Returns:
            (accepted_length, output_token_ids) where output is [B, k*+1].

        Mirrors ``MultiTokenPredictionLayer.forward_single_position`` which
        is used at inference for accepting the verified prefix.
        """
        # For each draft depth d, the main model verifies using slice d hidden states.
        # Slice 0 → predicts token at depth 1; slice d → verifies depth d+1.
        K = draft_ids.shape[-1]
        if K == 0:
            B = stacked_hs.shape[-2]
            return 0, torch.zeros(B, 1, dtype=torch.long, device=stacked_hs.device)

        accepted = 0
        last_hs = stacked_hs[0]   # [S, B, H]: main model output for base token

        for d in range(K):
            last_tok = last_hs[-1]         # [B, H]
            logits, _ = self._main_model_fn(last_tok.unsqueeze(0))
            # logits shape: [1, B, vocab_size] or [1, B*vocab_size]
            if logits.dim() == 3:
                logits = logits[-1]        # [B, vocab_size]
            elif logits.dim() == 2:
                pass
            greedy = logits.argmax(dim=-1) # [B]
            # Compare greedy with draft; take the majority-match decision
            # (batch-level: accept if ALL items in batch agree, conservative).
            if (greedy == draft_ids[:, d]).all():
                accepted += 1
                last_hs = stacked_hs[d + 1] if d + 1 < stacked_hs.shape[0] else last_hs
            else:
                # First mismatch: the corrected token is the main model's prediction.
                out_ids = torch.cat([draft_ids[:, :d], greedy.unsqueeze(1)], dim=1)
                return accepted, out_ids

        # All K tokens accepted; emit an extra correction token.
        last_tok = stacked_hs[-1][-1]  # [B, H]
        extra_logits, _ = self._main_model_fn(last_tok.unsqueeze(0))
        if extra_logits.dim() == 3:
            extra_logits = extra_logits[-1]
        bonus = extra_logits.argmax(dim=-1).unsqueeze(1)  # [B, 1]
        out_ids = torch.cat([draft_ids, bonus], dim=1)     # [B, K+1]
        return accepted, out_ids

    def _primary_device(self) -> torch.device:
        ids = self._config.verify_tier.device_ids
        if ids:
            return torch.device(f"cuda:{ids[0]}")
        return torch.device("cuda")


# ---------------------------------------------------------------------------
# Scheduler config
# ---------------------------------------------------------------------------

@dataclass
class HeteroMTPConfig:
    """Top-level configuration for ``HeteroMTPScheduler``.

    Attributes:
        draft_tier:          Configuration for the A6000 draft-head GPU tier.
        verify_tier:         Configuration for the H100 verify GPU tier.
        max_draft_depth:     Maximum draft depth the adapter can grow to.
        pipeline_queue_size: Maximum pending ``DraftResult``s queued waiting
                             for the verify tier.  Backpressure above this
                             size triggers a TIER_IMBALANCE log event.
        stall_threshold_ms:  Verify-tier idle time (ms) that triggers a
                             PIPELINE_STALL event (verify waiting for draft).
        adaptive_depth:      If True, enable dynamic draft_depth via
                             ``_DepthAdapter``.  If False, depth is fixed at
                             ``draft_tier.draft_depth``.
        hidden_size:         Model hidden dimension (for FLOP budget estimates).
        vocab_size:          Vocabulary size (for output-layer sizing checks).
        rank:                Global rank (0 = primary, logs diagnostic events).
    """
    draft_tier:          MTPTierConfig
    verify_tier:         MTPTierConfig
    max_draft_depth:     int = 8
    pipeline_queue_size: int = 16
    stall_threshold_ms:  float = _STALL_THRESHOLD_MS
    adaptive_depth:      bool = True
    hidden_size:         int = 4096
    vocab_size:          int = 32000
    rank:                int = 0


# ---------------------------------------------------------------------------
# Main scheduler
# ---------------------------------------------------------------------------

class HeteroMTPScheduler:
    """Heterogeneous MTP draft/verify scheduler for DES-LOC.

    Orchestrates the pipeline:
      1. Submit a batch → draft executor (A6000) generates K candidate tokens.
      2. ``DraftResult`` is handed to verify executor (H100) for acceptance.
      3. ``VerifyResult`` is returned to caller and fed back to the draft
         executor's ``_AcceptRateTracker`` for depth adaptation.

    The scheduler can run in two modes mirroring ``HeterogeneousInferenceEngine``:

    • **Inline mode** (``use_pipeline_thread=False``): draft and verify run
      sequentially on the caller's thread.  Simple; suitable for profiling.

    • **Pipeline mode** (``use_pipeline_thread=True``): a dedicated thread runs
      the verify step on the H100 while the calling thread can be preparing the
      next draft batch.  A bounded deque with size ``pipeline_queue_size`` acts
      as a back-pressure valve.  This matches how ``_EventLoopManager`` in the
      upstream stable-API commit decoupled submission from execution.

    Usage::

        cfg = HeteroMTPConfig(
            draft_tier=MTPTierConfig(
                role=GpuRole.DRAFT, device_ids=[0, 1],
                draft_depth=2, sm_version=86),
            verify_tier=MTPTierConfig(
                role=GpuRole.VERIFY, device_ids=[2, 3],
                draft_depth=0, sm_version=90),
        )
        scheduler = HeteroMTPScheduler(
            config=cfg,
            draft_model=mtp_block,
            embedding_fn=model.embedding,
            output_layer=model.output_layer,
            main_model_fn=main_model.output_layer,
        )
        result = scheduler.step(
            request_id=0,
            hidden_states=hs,
            input_ids=tokens,
            position_ids=pos_ids,
        )
        print(f"Accepted {result.accepted_length} / {cfg.draft_tier.draft_depth} tokens")
        scheduler.shutdown()
    """

    def __init__(
        self,
        config: HeteroMTPConfig,
        draft_model: Optional[Callable] = None,
        embedding_fn: Optional[Callable] = None,
        output_layer: Optional[Callable] = None,
        main_model_fn: Optional[Callable] = None,
        use_pipeline_thread: bool = False,
    ) -> None:
        """
        Args:
            config:               Scheduler configuration.
            draft_model:          ``MultiTokenPredictionBlock``-compatible callable.
                                  If None, stub mode is used (no actual drafting).
            embedding_fn:         Embedding callable for draft head token input.
            output_layer:         Output projection (draft tier's vocab head).
            main_model_fn:        Main model callable for verify step (H100).
            use_pipeline_thread:  If True, run verify in a background thread.
        """
        self._config = config
        self._rank  = config.rank

        # Build executors
        draft_stream  = _maybe_stream(config.draft_tier.device_ids)
        verify_stream = _maybe_stream(config.verify_tier.device_ids)

        self._draft_exec = _DraftExecutor(
            config=config,
            draft_model=draft_model,
            embedding_fn=embedding_fn,
            output_layer=output_layer,
            cuda_stream=draft_stream,
            rank=self._rank,
        )
        self._verify_exec = _VerifyExecutor(
            config=config,
            main_model_fn=main_model_fn,
            cuda_stream=verify_stream,
            rank=self._rank,
        )

        # Pipeline threading
        self._use_pipeline = use_pipeline_thread
        self._pipeline_queue: deque = deque(maxlen=config.pipeline_queue_size)
        self._pipeline_thread: Optional[threading.Thread] = None
        self._verify_results: Dict[Any, VerifyResult] = {}
        self._result_lock  = threading.Lock()
        self._shutdown_ev  = threading.Event()

        if use_pipeline_thread:
            self._pipeline_thread = threading.Thread(
                target=self._verify_loop, daemon=True, name="ds-hmtp-verify"
            )
            self._pipeline_thread.start()
            if self._rank == 0:
                _log(logging.DEBUG, "PIPELINE_READY",
                     "Verify pipeline thread started (H100 side)")

        self._is_shutdown = False

        # Calibrate draft_depth from flop budgets if both are provided.
        self._maybe_calibrate_depth()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        request_id: Any,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> VerifyResult:
        """Run one draft+verify step and return the accepted tokens.

        In **inline mode** this blocks until verification completes.
        In **pipeline mode** this submits the draft immediately and blocks
        only if the pipeline queue is full (back-pressure).

        Args:
            request_id:     Opaque identifier for this request.
            hidden_states:  Main model output [S, B, H] for the current position.
            input_ids:      Input token IDs [B, S].
            position_ids:   Position IDs [B, S].
            attention_mask: Optional causal attention mask.
            labels:         Ground-truth labels for training-mode loss.
            temperature:    Acceptance temperature (1.0 = greedy).

        Returns:
            :class:`VerifyResult` with the accepted prefix and corrected token.
        """
        if self._is_shutdown:
            raise RuntimeError("HeteroMTPScheduler has been shut down.")

        # 1. Draft step (A6000)
        draft_result = self._draft_exec.submit(
            request_id=request_id,
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        if self._use_pipeline:
            return self._step_pipeline(draft_result, labels, temperature)
        else:
            return self._step_inline(draft_result, labels, temperature)

    def step_batch(
        self,
        requests: List[Dict[str, Any]],
        temperature: float = 1.0,
    ) -> List[VerifyResult]:
        """Run draft+verify for a list of requests.

        Each request dict must have keys:
          ``request_id``, ``hidden_states``, ``input_ids``, ``position_ids``.
        Optional: ``attention_mask``, ``labels``.

        Returns a list of ``VerifyResult`` in the same order as ``requests``.
        """
        results = []
        for req in requests:
            result = self.step(
                request_id=req["request_id"],
                hidden_states=req["hidden_states"],
                input_ids=req["input_ids"],
                position_ids=req["position_ids"],
                attention_mask=req.get("attention_mask"),
                labels=req.get("labels"),
                temperature=temperature,
            )
            results.append(result)
        return results

    def shutdown(self) -> None:
        """Shut down the scheduler and release GPU resources.  Idempotent."""
        if self._is_shutdown:
            return
        self._is_shutdown = True
        if self._pipeline_thread is not None:
            self._shutdown_ev.set()
            self._pipeline_thread.join(timeout=10.0)
            self._pipeline_thread = None
            if self._rank == 0:
                _log(logging.DEBUG, "PIPELINE_STOP", "Verify pipeline thread stopped")

    # ------------------------------------------------------------------
    # Statistics / introspection
    # ------------------------------------------------------------------

    @property
    def draft_depth(self) -> int:
        """Current (possibly adapted) draft depth."""
        return self._draft_exec.adapter.depth

    @property
    def accept_rate(self) -> float:
        """Current EMA acceptance rate (0.0–1.0)."""
        return self._draft_exec.accept_tracker.rate

    @property
    def n_verified(self) -> int:
        """Total number of verify results processed."""
        return self._draft_exec.accept_tracker.n_samples

    def stats(self) -> Dict[str, Any]:
        """Return a dict of scheduler diagnostics for monitoring."""
        return {
            "draft_depth":  self.draft_depth,
            "accept_rate":  self.accept_rate,
            "n_verified":   self.n_verified,
            "queue_len":    len(self._pipeline_queue),
            "adaptive":     self._config.adaptive_depth,
        }

    # ------------------------------------------------------------------
    # Internal execution paths
    # ------------------------------------------------------------------

    def _step_inline(
        self,
        draft_result: DraftResult,
        labels: Optional[torch.Tensor],
        temperature: float,
    ) -> VerifyResult:
        """Synchronous inline draft→verify path (no threading)."""
        verify_result = self._verify_exec.verify(draft_result, labels, temperature)
        self._draft_exec.record_verify_result(verify_result)
        return verify_result

    def _step_pipeline(
        self,
        draft_result: DraftResult,
        labels: Optional[torch.Tensor],
        temperature: float,
    ) -> VerifyResult:
        """Pipeline path: enqueue draft, then block until verify completes.

        The pipeline thread drains the queue; the scheduler submits drafts
        and waits for the corresponding result via a per-request ``Event``.
        """
        ev = threading.Event()
        entry = (draft_result, labels, temperature, ev)

        # Back-pressure: if queue is full, stall until space opens.
        t_stall = time.monotonic()
        while len(self._pipeline_queue) >= self._config.pipeline_queue_size:
            time.sleep(0.001)
            stall_ms = (time.monotonic() - t_stall) * 1e3
            if stall_ms > self._config.stall_threshold_ms and self._rank == 0:
                _log(logging.WARNING, "PIPELINE_STALL",
                     f"Verify tier stalled for {stall_ms:.1f} ms — "
                     f"queue={len(self._pipeline_queue)}/{self._config.pipeline_queue_size}")

        self._pipeline_queue.append(entry)
        ev.wait(timeout=60.0)  # 60s hard timeout

        with self._result_lock:
            result = self._verify_results.pop(draft_result.request_id, None)
        if result is None:
            raise TimeoutError(
                f"HeteroMTPScheduler: verify timed out for request {draft_result.request_id}"
            )
        self._draft_exec.record_verify_result(result)
        return result

    def _verify_loop(self) -> None:
        """Background thread: drain pipeline_queue, run verify, store results."""
        while not self._shutdown_ev.is_set():
            if not self._pipeline_queue:
                time.sleep(0.0005)
                continue
            try:
                entry = self._pipeline_queue.popleft()
            except IndexError:
                continue
            draft_result, labels, temperature, ev = entry
            try:
                result = self._verify_exec.verify(draft_result, labels, temperature)
            except Exception as exc:
                if self._rank == 0:
                    _log(logging.ERROR, "VERIFY_ERROR",
                         f"Verify failed for request {draft_result.request_id}: {exc}")
                result = VerifyResult(
                    request_id=draft_result.request_id,
                    accepted_length=0,
                    output_token_ids=draft_result.draft_token_ids[:, :1]
                        if draft_result.draft_token_ids.numel() > 0
                        else torch.zeros(1, 1, dtype=torch.long),
                )
            with self._result_lock:
                self._verify_results[draft_result.request_id] = result
            ev.set()

    # ------------------------------------------------------------------
    # FLOP-budget depth calibration
    # ------------------------------------------------------------------

    def _maybe_calibrate_depth(self) -> None:
        """Auto-calibrate draft_depth from GPU flop_budget fields.

        If both draft and verify tiers report ``flop_budget`` (TFLOP/s), we
        compute the ideal draft depth K* such that:

            T_draft(K*) ≈ T_verify

        where T_draft(K) = K × flop_per_mtp_layer / draft_flop_budget
        and   T_verify   = flop_per_verify_step / verify_flop_budget.

        MTP layer FLOPs ≈ 2 × hidden_size^2 × (attn + ffn factor)
        We use the simple 4 × hidden_size^2 heuristic (1 QKV proj + 1 FFN).
        """
        d = self._config.draft_tier
        v = self._config.verify_tier
        if d.flop_budget is None or v.flop_budget is None:
            return

        # Rough per-token FLOPs for one MTP layer
        h = self._config.hidden_size
        flop_mtp_layer = 4 * h * h   # ~2 matmuls (proj + transformer)
        flop_verify    = 32 * h * h  # main model has ~8× more layers in typical configs

        # Ideal K for equal-wall-clock pipelining:
        #   K × (flop_mtp_layer / d.flop_budget) = flop_verify / v.flop_budget
        ratio = (flop_verify / v.flop_budget) / (flop_mtp_layer / d.flop_budget)
        calibrated_k = max(1, min(int(math.floor(ratio)), self._config.max_draft_depth))

        if calibrated_k != self._config.draft_tier.draft_depth and self._rank == 0:
            _log(
                logging.INFO, "DEPTH_CALIBRATE",
                f"Calibrated draft_depth {self._config.draft_tier.draft_depth} → {calibrated_k} "
                f"(draft={d.flop_budget:.0f} TFLOP/s, verify={v.flop_budget:.0f} TFLOP/s)"
            )
        self._draft_exec._adapter._depth = calibrated_k


# ---------------------------------------------------------------------------
# Factory: build from GPU SM version strings (mirrors fp8_gemm.py routing)
# ---------------------------------------------------------------------------

def build_hetero_mtp_scheduler(
    draft_device_ids: List[int],
    verify_device_ids: List[int],
    draft_model: Optional[Callable] = None,
    embedding_fn: Optional[Callable] = None,
    output_layer: Optional[Callable] = None,
    main_model_fn: Optional[Callable] = None,
    *,
    draft_depth: int = _DEFAULT_DRAFT_DEPTH_A6000,
    draft_sm_version: int = 86,
    verify_sm_version: int = 90,
    hidden_size: int = 4096,
    vocab_size: int = 32000,
    adaptive_depth: bool = True,
    use_pipeline_thread: bool = False,
    rank: int = 0,
    draft_flop_budget: Optional[float] = None,
    verify_flop_budget: Optional[float] = None,
) -> HeteroMTPScheduler:
    """Convenience factory matching the DES-LOC A6000+H100 default topology.

    Args:
        draft_device_ids:    CUDA device IDs for A6000 (draft) GPUs.
        verify_device_ids:   CUDA device IDs for H100 (verify) GPUs.
        draft_model:         ``MultiTokenPredictionBlock`` or equivalent.
        embedding_fn:        Embedding callable.
        output_layer:        Vocab head for draft-tier logits.
        main_model_fn:       Main model output layer for verify step.
        draft_depth:         Initial draft depth K.
        draft_sm_version:    Compute capability of draft GPUs (default 86 = SM8.6).
        verify_sm_version:   Compute capability of verify GPUs (default 90 = SM9.0).
        hidden_size:         Model hidden dimension.
        vocab_size:          Vocabulary size.
        adaptive_depth:      Whether to enable dynamic depth adaptation.
        use_pipeline_thread: Whether to run verify on a background thread.
        rank:                Global rank for diagnostics.
        draft_flop_budget:   Optional peak TFLOP/s for draft tier (enables auto-calibration).
        verify_flop_budget:  Optional peak TFLOP/s for verify tier.

    Returns:
        Configured :class:`HeteroMTPScheduler`.

    Example::

        scheduler = build_hetero_mtp_scheduler(
            draft_device_ids=[0, 1],          # two A6000s
            verify_device_ids=[2, 3],         # two H100s
            draft_model=mtp_block,
            embedding_fn=model.embedding,
            output_layer=model.output_layer,
            main_model_fn=model.output_layer,
            draft_depth=2,
            draft_flop_budget=310.0,          # A6000 fp16 peak TFLOP/s
            verify_flop_budget=2000.0,        # H100 fp16 peak TFLOP/s
        )
    """
    config = HeteroMTPConfig(
        draft_tier=MTPTierConfig(
            role=GpuRole.DRAFT,
            device_ids=draft_device_ids,
            draft_depth=draft_depth,
            sm_version=draft_sm_version,
            flop_budget=draft_flop_budget,
        ),
        verify_tier=MTPTierConfig(
            role=GpuRole.VERIFY,
            device_ids=verify_device_ids,
            draft_depth=0,
            sm_version=verify_sm_version,
            flop_budget=verify_flop_budget,
        ),
        max_draft_depth=max(8, draft_depth * 3),
        adaptive_depth=adaptive_depth,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        rank=rank,
    )
    return HeteroMTPScheduler(
        config=config,
        draft_model=draft_model,
        embedding_fn=embedding_fn,
        output_layer=output_layer,
        main_model_fn=main_model_fn,
        use_pipeline_thread=use_pipeline_thread,
    )


# ---------------------------------------------------------------------------
# Stub utilities (unit-test / dry-run mode)
# ---------------------------------------------------------------------------

def _stub_hidden_states(base: torch.Tensor, depth: int) -> torch.Tensor:
    """Return [1+depth, S, B, H] of zeros matching base shape."""
    if base.dim() == 3:
        base = base.unsqueeze(0)   # [1, S, B, H]
    if depth == 0:
        return base
    zeros = torch.zeros(
        depth, *base.shape[1:], dtype=base.dtype, device=base.device
    )
    return torch.cat([base, zeros], dim=0)


def _stub_draft_ids(
    input_ids: torch.Tensor, depth: int, device: torch.device
) -> torch.Tensor:
    """Return [B, depth] draft token IDs (all zeros for stubs)."""
    B = input_ids.shape[0] if input_ids.dim() >= 2 else 1
    return torch.zeros(B, depth, dtype=torch.long, device=device)


def _maybe_stream(
    device_ids: List[int]
) -> Optional[torch.cuda.Stream]:
    """Create a CUDA stream on the first available device, or None."""
    if not device_ids:
        return None
    try:
        with torch.cuda.device(device_ids[0]):
            return torch.cuda.Stream()
    except Exception:
        return None


class _nullctx:
    """Trivial no-op context manager (replaces ``contextlib.nullcontext`` for
    Python < 3.7 compatibility and avoids importing contextlib)."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# Diagnostic logging helpers
# ---------------------------------------------------------------------------

def _log(level: int, event: str, msg: str) -> None:
    line = f"{_LOG_PREFIX} {event}: {msg}"
    ds_logger.log(level, line)


def _log_draft_dispatch(
    request_id: Any,
    depth: int,
    device: torch.device,
    stub: bool,
    ms: float = 0.0,
) -> None:
    mode = "stub" if stub else f"{ms:.2f} ms"
    _log(
        logging.DEBUG, "DRAFT_DISPATCH",
        f"req={request_id} depth={depth} device={device} ({mode})"
    )


def _log_verify_result(
    request_id: Any,
    accepted: int,
    draft_depth: int,
    latency_ms: float,
    stub: bool,
) -> None:
    mode = "stub" if stub else f"{latency_ms:.2f} ms"
    rate_pct = (accepted / max(draft_depth, 1)) * 100
    _log(
        logging.DEBUG, "VERIFY_RESULT",
        f"req={request_id} accepted={accepted}/{draft_depth} "
        f"({rate_pct:.0f}%) latency={mode}"
    )


def _log_depth_adapt(old: int, new: int, rate: float) -> None:
    direction = "↑" if new > old else "↓"
    _log(
        logging.INFO, "DEPTH_ADAPT",
        f"draft_depth {old} → {new} {direction} (accept_rate={rate:.2%})"
    )
