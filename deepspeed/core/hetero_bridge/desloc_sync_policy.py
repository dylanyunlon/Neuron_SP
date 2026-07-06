# SPDX-License-Identifier: Apache-2.0
"""desloc_sync_policy.py — DES-LOC decomposed-local-SGD per-parameter sync schedule.

Implements Algorithm 1 from the DES-LOC paper: each parameter is classified
into one of three sync classes (x / u / v) based on gradient variance, and
synchronized only every Kx / Ku / Kv steps respectively.

Gradient variance heuristic
----------------------------
We track a per-parameter exponential moving average of gradient L2-norm
squared.  High-variance parameters (attention / embedding layers) are tagged
'x' (sync every Kx steps).  Low-variance parameters (feed-forward layers) are
tagged 'u' or 'v'.  The exact thresholds mirror the DES-LOC paper's Table 2.

Classification algorithm
------------------------
1. Compute EMA-smoothed gradient variance per param (α=0.1).
2. Sort params by variance descending.
3. Top 1/3 by count → 'x' (fast sync, Kx steps).
4. Mid 1/3 → 'u' (medium sync, Ku steps).
5. Bottom 1/3 → 'v' (slow sync, Kv steps).

Name-based overrides (applied before variance ranking):
  - '*embed*', '*wte*', '*wpe*'  → always 'x' (embeddings are highly variable)
  - 'lm_head*'                   → always 'x'
  - '*norm*', '*ln*'             → always 'v' (layer norms converge quickly)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# Smoothing factor for the gradient variance EMA.
_EMA_ALPHA = 0.1

# Name patterns → forced class. Applied in order; first match wins.
_NAME_OVERRIDES: List[Tuple[str, str]] = [
    ("embed", "x"),
    ("wte", "x"),
    ("wpe", "x"),
    ("lm_head", "x"),
    ("norm", "v"),
    ("ln_", "v"),
    ("ln_f", "v"),
]


@dataclass
class SyncPeriods:
    kx: int  # fast params (e.g. attention) sync period
    ku: int  # slow params
    kv: int  # very-slow params


class DesLocSyncPolicy:
    """Per-parameter DES-LOC sync class assignment and step-level predicate.

    Usage
    -----
    >>> policy = DesLocSyncPolicy(SyncPeriods(kx=32, ku=96, kv=192))
    >>> classes = policy.classify(list(model.named_parameters()))
    >>> if policy.should_sync(param_id=42, step=128):
    ...     # launch allreduce for param 42
    """

    def __init__(self, periods: "SyncPeriods") -> None:
        self.periods = periods
        # param_id → sync class ('x', 'u', 'v')
        self._classes: Dict[int, str] = {}
        # param_id → EMA of gradient variance (L2-norm^2 / numel)
        self._var_ema: Dict[int, float] = {}
        # Whether classify() has been called at least once
        self._classified = False

    # ------------------------------------------------------------------
    # Public API (frozen per ARCHITECTURE.md)
    # ------------------------------------------------------------------

    def classify(
        self,
        named_params: "list[tuple[str, torch.Tensor]]",
    ) -> Dict[int, str]:
        """Classify each parameter into sync class 'x', 'u', or 'v'.

        Args:
            named_params: List of (name, parameter) tuples from
                          ``model.named_parameters()``.

        Returns:
            Dict mapping ``id(param)`` → sync class string.
        """
        import torch

        result: Dict[int, str] = {}

        # ── Step 1: name-based overrides ────────────────────────────
        forced: Dict[int, str] = {}
        for name, param in named_params:
            name_lower = name.lower()
            for pattern, cls in _NAME_OVERRIDES:
                if pattern in name_lower:
                    forced[id(param)] = cls
                    break

        # ── Step 2: compute gradient variance per param ────────────
        # Use the existing EMA state if available (warm restart), else 0.
        variance_scores: List[Tuple[int, float]] = []
        for name, param in named_params:
            pid = id(param)
            if pid in forced:
                continue  # already classified
            if param.grad is not None:
                g = param.grad.detach().float()
                # Variance proxy: mean squared gradient magnitude (L2² / numel)
                var_proxy = (g * g).mean().item()
            else:
                var_proxy = 0.0

            # EMA update
            prev = self._var_ema.get(pid, var_proxy)
            ema = (1.0 - _EMA_ALPHA) * prev + _EMA_ALPHA * var_proxy
            self._var_ema[pid] = ema
            variance_scores.append((pid, ema))

        # ── Step 3: rank remaining params by variance descending ────
        variance_scores.sort(key=lambda t: t[1], reverse=True)
        n = len(variance_scores)
        top_k = max(1, n // 3)
        mid_k = max(1, n // 3)

        for i, (pid, _) in enumerate(variance_scores):
            if i < top_k:
                result[pid] = "x"
            elif i < top_k + mid_k:
                result[pid] = "u"
            else:
                result[pid] = "v"

        # ── Step 4: merge forced overrides ──────────────────────────
        result.update(forced)
        for pid, cls in forced.items():
            result[pid] = cls

        # ── Step 5: persist and log ─────────────────────────────────
        self._classes = result
        self._classified = True

        n_x = sum(1 for c in result.values() if c == "x")
        n_u = sum(1 for c in result.values() if c == "u")
        n_v = sum(1 for c in result.values() if c == "v")
        logger.info(
            "[DesLocSyncPolicy] classified %d params — "
            "x=%d (Kx=%d), u=%d (Ku=%d), v=%d (Kv=%d)",
            len(result), n_x, self.periods.kx,
            n_u, self.periods.ku,
            n_v, self.periods.kv,
        )
        return result

    def should_sync(self, param_id: int, step: int) -> bool:
        """Return True if *param_id* should be synced at *step*.

        If classify() has not been called yet the policy falls back to Kx
        (sync every Kx steps) for all params — a safe over-approximation.

        Args:
            param_id: ``id(parameter)`` as returned by classify().
            step:     Current training step (0-indexed).

        Returns:
            True if this param should participate in an allreduce this step.
        """
        if not self._classified:
            # Pre-classification fallback: sync every Kx steps.
            return (step + 1) % self.periods.kx == 0

        cls = self._classes.get(param_id, "x")
        if cls == "x":
            period = self.periods.kx
        elif cls == "u":
            period = self.periods.ku
        else:  # 'v'
            period = self.periods.kv

        return (step + 1) % period == 0

    # ------------------------------------------------------------------
    # Extra helpers (used by AutoSPHook)
    # ------------------------------------------------------------------

    def get_class(self, param_id: int) -> str:
        """Return the sync class for *param_id*, defaulting to 'x'."""
        return self._classes.get(param_id, "x")

    def update_variance_ema(
        self,
        named_params: "list[tuple[str, torch.Tensor]]",
    ) -> None:
        """Incrementally update gradient variance EMAs without re-classifying.

        Call this every step (or every N steps) to keep the EMA current so
        that the next classify() call reflects recent gradient behaviour.
        """
        for _, param in named_params:
            if param.grad is None:
                continue
            pid = id(param)
            g = param.grad.detach().float()
            var_proxy = (g * g).mean().item()
            prev = self._var_ema.get(pid, var_proxy)
            self._var_ema[pid] = (1.0 - _EMA_ALPHA) * prev + _EMA_ALPHA * var_proxy
