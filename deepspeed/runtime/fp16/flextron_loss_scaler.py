# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""flextron_loss_scaler.py — Budget-aware loss scaler for DES-LOC Flextron training.

Mirrors Megatron 2d862fe0c (Flextron commit) loss_func.py ``_mask_loss`` and
``loss_func``, reinterpreted as a DeepSpeed DynamicLossScaler extension that:

1. Decomposes the total loss into LM loss + param_budget_loss components —
   matching the upstream ``param_loss_item`` split in ``_mask_loss``.

2. Accumulates per-budget loss statistics (lm_loss, param_loss, total_loss)
   across micro-batches, mirroring the upstream ``report`` dict with per-budget
   breakdown for logging distillation loss per budget level.

3. Emits structured diagnostics at three event boundaries (M451 pattern —
   one log per event, not per-iteration noise):
     [DS-FlextronScale] OVERFLOW:   scale dropped due to gradient overflow.
     [DS-FlextronScale] GREW:       scale grown after stable_interval iters
                                    (extends M451 DynamicLossScaler GREW event
                                    with per-budget budget contribution context).
     [DS-FlextronScale] BUDGET_LOG: per-budget loss breakdown at configurable
                                    log_interval, mirrors upstream per-budget
                                    report dict reduction.

Design intent (upstream 2d862fe0c)
------------------------------------
The FlextronRouter selects one budget ∈ budget_list per forward pass.  The
total loss is:

    L_total = L_lm  +  alpha * L_param_budget

where L_param_budget penalises deviations from the target budget (measured
in parameter count or memory footprint).  During training, different micro-
batches may use different budgets (sampled from budget_probs).  The upstream
``loss_func`` builds a ``report`` dict with per-budget breakdown so the
trainer can log how each sub-network is converging.

DES-LOC reinterpretation
-------------------------
DeepSpeed's training loop does not have direct access to the Megatron
``report`` dict.  We replicate the budget-breakdown accumulation inside the
loss scaler so it can be queried at each optimizer step without modifying the
engine's main training loop.  The scaler accepts an optional
``(lm_loss, param_loss, budget)`` tuple per micro-batch call and aggregates
it until ``update_scale`` is called (matching one gradient accumulation cycle).

Checkpoint replay
-----------------
``load_state_dict`` follows M451 discipline: ``_iter`` and budget counters
are restored from the checkpoint but ``_budget_lm_accum`` / ``_budget_param_accum``
are reset to zero (accumulated across micro-batches, not persisted).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

from deepspeed.runtime.fp16.loss_scaler import DynamicLossScaler
from deepspeed.utils import logger as ds_logger

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[DS-FlextronScale]"


class FlextronAwareLossScaler(DynamicLossScaler):
    """DynamicLossScaler extended with per-budget loss decomposition.

    Wraps DynamicLossScaler to additionally track, per micro-batch, how much
    of the total loss comes from the LM objective vs the Flextron budget
    penalty term.  Produces a per-budget report dict at each optimizer step
    that mirrors the upstream Megatron ``loss_func`` ``report`` structure.

    Parameters
    ----------
    budget_list : list of float
        All possible budget values; used to initialise per-budget accumulators.
        Must match the budget_list used by FlextronRouter / HeteroFlextronConfig.
    loss_alpha : float
        Coefficient on the param_budget loss term (upstream ``args.loss_alpha``).
    log_interval : int
        Emit ``[DS-FlextronScale] BUDGET_LOG`` every this many optimizer steps.
        Set to 0 to disable budget logging.
    All other parameters are forwarded verbatim to DynamicLossScaler.
    """

    def __init__(
        self,
        init_scale: float,
        scale_window: int,
        min_scale: float,
        delayed_shift: int,
        consecutive_hysteresis: bool,
        *,
        budget_list: List[float],
        loss_alpha: float = 1.0,
        log_interval: int = 100,
        raise_error_at_min_scale: bool = True,
        dtype=torch.half,
    ):
        super().__init__(
            init_scale=init_scale,
            scale_window=scale_window,
            min_scale=min_scale,
            delayed_shift=delayed_shift,
            consecutive_hysteresis=consecutive_hysteresis,
            raise_error_at_min_scale=raise_error_at_min_scale,
            dtype=dtype,
        )
        # Budget tracking
        self.budget_list  = sorted(set(budget_list), reverse=True)
        self.loss_alpha   = loss_alpha
        self.log_interval = log_interval

        # Per-budget accumulators: reset at each optimizer step.
        # Structure: {budget: [lm_loss_sum, param_loss_sum, token_count]}
        self._budget_lm_accum:    Dict[float, float] = defaultdict(float)
        self._budget_param_accum: Dict[float, float] = defaultdict(float)
        self._budget_token_count: Dict[float, int]   = defaultdict(int)
        self._microbatch_count = 0

        # Step counter for log_interval gating (persisted through checkpoints)
        self._opt_step = 0

    # ------------------------------------------------------------------
    # Per-micro-batch loss registration
    # ------------------------------------------------------------------

    def register_microbatch_loss(
        self,
        *,
        lm_loss: float,
        param_loss: float,
        budget: float,
        num_tokens: int = 1,
    ) -> None:
        """Record LM + param losses for one micro-batch.

        Called by the training loop (or engine forward hook) for each
        micro-batch before ``update_scale``.  Mirrors upstream ``_mask_loss``
        splitting the output tensor into LM loss and budget penalty.

        Parameters
        ----------
        lm_loss : float
            Language-model loss contribution for this micro-batch.
        param_loss : float
            Param-budget penalty contribution (alpha * L_param_budget).
        budget : float
            Budget fraction used for this micro-batch (selected by router).
        num_tokens : int
            Number of unmasked tokens (for weighted averaging in report).
        """
        # Snap budget to nearest entry in budget_list to handle float precision
        snapped = self._snap_budget(budget)
        self._budget_lm_accum[snapped]    += lm_loss    * num_tokens
        self._budget_param_accum[snapped] += param_loss * num_tokens
        self._budget_token_count[snapped] += num_tokens
        self._microbatch_count += 1

    def _snap_budget(self, budget: float) -> float:
        """Snap budget to nearest value in self.budget_list (float precision guard)."""
        if not self.budget_list:
            return budget
        return min(self.budget_list, key=lambda b: abs(b - budget))

    # ------------------------------------------------------------------
    # Report dict (mirrors upstream Megatron loss_func report structure)
    # ------------------------------------------------------------------

    def build_report(self) -> Dict[str, Tuple[float, int]]:
        """Return per-budget loss breakdown accumulated since last optimizer step.

        Mirrors the upstream ``report`` dict returned by ``loss_func``:
          'lm loss'          : (weighted_lm_loss_sum, total_tokens)
          'param loss item'  : (weighted_param_sum,   total_tokens)
          'lm loss (b=X.XXX)': per-budget breakdown for each budget in budget_list

        Returns
        -------
        dict mapping str -> (loss_value, token_count)
        """
        report: Dict[str, Tuple[float, int]] = {}
        total_lm    = 0.0
        total_param = 0.0
        total_tok   = 0

        for budget in self.budget_list:
            tok = self._budget_token_count.get(budget, 0)
            if tok == 0:
                continue
            lm_s  = self._budget_lm_accum.get(budget, 0.0)
            par_s = self._budget_param_accum.get(budget, 0.0)
            total_lm    += lm_s
            total_param += par_s
            total_tok   += tok
            report[f"lm loss (b={budget:.4f})"]    = (lm_s, tok)
            report[f"param loss (b={budget:.4f})"] = (par_s, tok)

        report["lm loss"]        = (total_lm,    max(total_tok, 1))
        report["param loss item"] = (total_param, max(total_tok, 1))
        return report

    # ------------------------------------------------------------------
    # Override update_scale to emit budget diagnostics
    # ------------------------------------------------------------------

    def update_scale(self, overflow: bool) -> None:
        """Extend DynamicLossScaler.update_scale with budget-aware diagnostics.

        Mirrors M451's structured GREW diagnostic (single event at the growth
        boundary) and adds an OVERFLOW event.  Also emits BUDGET_LOG at
        configurable intervals with the per-budget loss breakdown.
        """
        try:
            import torch.distributed as tdist
            _rank = tdist.get_rank() if tdist.is_initialized() else 0
        except Exception:
            _rank = 0

        prev_scale = self.cur_scale
        prev_iter  = self.cur_iter

        # Call parent — modifies self.cur_scale and self.cur_iter
        super().update_scale(overflow)

        scale_grew    = (not overflow) and (self.cur_scale > prev_scale)
        scale_dropped = overflow

        if _rank == 0:
            if scale_dropped:
                # OVERFLOW event (mirrors M451 loss_scaler scale-drop implicit event)
                stable = prev_iter - self.last_overflow_iter
                msg = (
                    f"{_LOG_PREFIX} OVERFLOW at opt_step={self._opt_step}  "
                    f"scale: {prev_scale:.1f} -> {self.cur_scale:.1f}  "
                    f"stable_iters_before_overflow={stable}  "
                    f"microbatches_this_step={self._microbatch_count}  "
                    f"budgets_seen={sorted(self._budget_token_count.keys())}"
                )
                print(msg, flush=True)
                ds_logger.warning(msg)

            elif scale_grew:
                # GREW event — extends M451 DynamicLossScaler GREW with budget context
                stable = prev_iter - self.last_overflow_iter
                msg = (
                    f"{_LOG_PREFIX} GREW at opt_step={self._opt_step}  "
                    f"scale: {prev_scale:.1f} -> {self.cur_scale:.1f}  "
                    f"stable_interval={stable}  scale_window={self.scale_window}  "
                    f"budgets_this_step={sorted(self._budget_token_count.keys())}  "
                    f"alpha={self.loss_alpha}"
                )
                print(msg, flush=True)
                ds_logger.info(msg)

            # BUDGET_LOG at configurable interval
            if (
                self.log_interval > 0
                and self._opt_step % self.log_interval == 0
                and self._microbatch_count > 0
            ):
                report = self.build_report()
                # Format per-budget breakdown compactly
                per_budget = {
                    k: f"{v:.4f}/{n}"
                    for k, (v, n) in report.items()
                    if k.startswith("lm loss (b=") and n > 0
                }
                msg = (
                    f"{_LOG_PREFIX} BUDGET_LOG opt_step={self._opt_step}  "
                    f"total_lm={report.get('lm loss', (0.0, 0))[0]:.4f}  "
                    f"total_param={report.get('param loss item', (0.0, 0))[0]:.4f}  "
                    f"per_budget={per_budget}  "
                    f"scale={self.cur_scale:.1f}"
                )
                print(msg, flush=True)
                ds_logger.info(msg)

        # Advance step counter and reset micro-batch accumulators
        self._opt_step += 1
        self._reset_accumulators()

    def _reset_accumulators(self) -> None:
        """Clear per-step accumulators after each optimizer step."""
        self._budget_lm_accum.clear()
        self._budget_param_accum.clear()
        self._budget_token_count.clear()
        self._microbatch_count = 0

    # ------------------------------------------------------------------
    # Checkpoint support (M451 replay discipline)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Extend parent state_dict with budget tracking state."""
        d = {
            "cur_scale":       self.cur_scale,
            "cur_iter":        self.cur_iter,
            "last_overflow_iter": self.last_overflow_iter,
            "cur_hysteresis":  self.cur_hysteresis,
            # FlextronAwareLossScaler extras
            "opt_step":        self._opt_step,
            "budget_list":     self.budget_list,
            "loss_alpha":      self.loss_alpha,
            "log_interval":    self.log_interval,
        }
        return d

    def load_state_dict(self, d: dict) -> None:
        """Restore scaler state, then emit replay diagnostic (M451 pattern).

        Accumulator dicts are reset (they are transient, not persisted).
        ``_opt_step`` is restored from checkpoint so log_interval parity
        is maintained across checkpoint restarts.
        """
        prev_scale = self.cur_scale
        prev_step  = self._opt_step

        self.cur_scale           = d.get("cur_scale",           self.cur_scale)
        self.cur_iter            = d.get("cur_iter",            self.cur_iter)
        self.last_overflow_iter  = d.get("last_overflow_iter",  self.last_overflow_iter)
        self.cur_hysteresis      = d.get("cur_hysteresis",      self.cur_hysteresis)
        self._opt_step           = d.get("opt_step",            self._opt_step)
        self.loss_alpha          = d.get("loss_alpha",          self.loss_alpha)
        self.log_interval        = d.get("log_interval",        self.log_interval)

        restored_budget_list = d.get("budget_list", self.budget_list)
        if restored_budget_list != self.budget_list:
            logger.warning(
                f"{_LOG_PREFIX} LOAD_STATE_DICT: budget_list changed "
                f"{restored_budget_list} -> {self.budget_list} (keeping current)"
            )
        # Do NOT restore accumulators — they are transient
        self._reset_accumulators()

        try:
            import torch.distributed as tdist
            _rank = tdist.get_rank() if tdist.is_initialized() else 0
        except Exception:
            _rank = 0

        if _rank == 0:
            msg = (
                f"{_LOG_PREFIX} LOAD_STATE_DICT: restored scale={self.cur_scale:.1f}  "
                f"(was {prev_scale:.1f})  opt_step={self._opt_step}  "
                f"(was {prev_step})  budget_list={self.budget_list}  "
                f"accumulators reset"
            )
            print(msg, flush=True)
            ds_logger.info(msg)
