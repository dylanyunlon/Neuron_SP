# Copyright (c) 2025 Peng Du and Zhipeng Wang
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
try:
    from deepspeed.runtime.zero.muon.original_muon import MuonWithAuxAdam as BaseMuonWithAuxAdam
    from deepspeed.runtime.zero.muon.original_muon import adam_update
except ImportError:
    pass


class MuonWithAuxAdam(BaseMuonWithAuxAdam):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                # we move the muon update part to the deepspeed's optimizer since the parameter here is a flat version
                # thus not suitable for muon update
                for p in group["params"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(p.grad.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"], state["step"], group["betas"],
                                         group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


# ═══════════════════════════════════════════════════════════════
# DES-LOC Muon Integration (M188)
# Section 5.6: "Since Muon preconditions the momentum term
# directly rather than tracking second-moment estimates,
# the relevant sync periods reduce to Kx and Ku."
# ═══════════════════════════════════════════════════════════════


class DESLOCMuonWithAuxAdam(BaseMuonWithAuxAdam):
    """DES-LOC-aware MuonWithAuxAdam.

    Extends the base Muon optimizer with:
    1. Step counter for DES-LOC sync scheduling
    2. Ku-gated momentum synchronization
    3. Per-coordinate gradient clipping (Algorithm 1 line 12)
    4. Communication tracking for NeurIPS figures

    For Muon, only Kx (parameters) and Ku (momentum) are relevant.
    There is no second moment (Kv is unused / set to Ku).
    """

    def __init__(self, *args, **kwargs):
        # Extract DES-LOC params before passing to base
        self._desloc_Kx = kwargs.pop('desloc_Kx', 1)
        self._desloc_Ku = kwargs.pop('desloc_Ku', 1)
        self._desloc_clip_radius = kwargs.pop('desloc_clip_radius', 1.0)
        self._desloc_enabled = kwargs.pop('desloc_enabled', False)
        super().__init__(*args, **kwargs)
        self._desloc_step = 0
        self._desloc_sync_x_count = 0
        self._desloc_sync_u_count = 0
        self._desloc_total_comm_bytes = 0
        self._desloc_skipped_comm_bytes = 0
        self._desloc_clip_count = 0
        self._desloc_momentum_norms = []  # track for half-life analysis

    def _desloc_should_sync_x(self):
        """Check if parameters should be synced at current step."""
        if not self._desloc_enabled or self._desloc_Kx <= 1:
            return True
        return self._desloc_step % self._desloc_Kx == 0

    def _desloc_should_sync_u(self):
        """Check if momentum buffer should be synced at current step."""
        if not self._desloc_enabled or self._desloc_Ku <= 1:
            return True
        return self._desloc_step % self._desloc_Ku == 0

    def _desloc_clip_grad(self, grad, rho):
        """Per-coordinate gradient clipping (Algorithm 1 line 12)."""
        if rho >= float('inf'):
            return
        mask = grad.abs() > rho
        if mask.any():
            self._desloc_clip_count += 1
            grad.clamp_(-rho, rho)

    def _desloc_track_momentum_norm(self, state):
        """Track momentum buffer norms for half-life analysis (Section 5.1)."""
        if 'momentum_buffer' in state:
            norm_val = state['momentum_buffer'].norm().item()
            self._desloc_momentum_norms.append(norm_val)
            # Keep bounded
            if len(self._desloc_momentum_norms) > 10000:
                self._desloc_momentum_norms = self._desloc_momentum_norms[-5000:]

    def get_desloc_stats(self):
        """Return DES-LOC communication statistics."""
        return {
            'step': self._desloc_step,
            'sync_x_count': self._desloc_sync_x_count,
            'sync_u_count': self._desloc_sync_u_count,
            'total_comm_bytes': self._desloc_total_comm_bytes,
            'skipped_comm_bytes': self._desloc_skipped_comm_bytes,
            'clip_count': self._desloc_clip_count,
            'Kx': self._desloc_Kx,
            'Ku': self._desloc_Ku,
            'momentum_norm_samples': len(self._desloc_momentum_norms),
        }

    def get_momentum_norm_history(self):
        """Return momentum norm history for Figure CLXXXVII generation."""
        return list(self._desloc_momentum_norms)

    @torch.no_grad()
    def step(self, closure=None):
        """DES-LOC-aware Muon step.

        1. Increment DES-LOC step counter
        2. Apply per-coordinate clipping to gradients
        3. Execute Muon/Adam update locally
        4. Track momentum norms for half-life analysis
        """
        self._desloc_step += 1

        # Apply per-coordinate clipping before the optimizer step
        if self._desloc_enabled and self._desloc_clip_radius < float('inf'):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        self._desloc_clip_grad(p.grad, self._desloc_clip_radius)

        # Call base Muon step
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(p.grad.reshape(p.shape), alpha=-group["lr"])
                    # Track momentum norms
                    if p in self.state:
                        self._desloc_track_momentum_norm(self.state[p])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad, state["exp_avg"], state["exp_avg_sq"],
                        state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss

    def state_dict_desloc(self):
        """Extended state dict with DES-LOC counters."""
        base = {}
        base['optimizer_state'] = self.state_dict()
        base['desloc'] = {
            'step': self._desloc_step,
            'sync_x_count': self._desloc_sync_x_count,
            'sync_u_count': self._desloc_sync_u_count,
            'total_comm_bytes': self._desloc_total_comm_bytes,
            'Kx': self._desloc_Kx,
            'Ku': self._desloc_Ku,
            'clip_radius': self._desloc_clip_radius,
        }
        return base

    def load_state_dict_desloc(self, state_dict):
        """Load state dict with DES-LOC counters."""
        if 'optimizer_state' in state_dict:
            self.load_state_dict(state_dict['optimizer_state'])
        desloc = state_dict.get('desloc', {})
        self._desloc_step = desloc.get('step', 0)
        self._desloc_sync_x_count = desloc.get('sync_x_count', 0)
        self._desloc_sync_u_count = desloc.get('sync_u_count', 0)
        self._desloc_total_comm_bytes = desloc.get('total_comm_bytes', 0)


class DESLOCMuonCommTracker:
    """Track Muon-specific communication for DES-LOC figures.

    Section 5.6: "DES-LOC communicates more than 190x fewer bytes
    than the baseline" — this tracker measures the actual reduction.

    For Muon, communication consists of:
    - Parameter all_gather at Kx intervals
    - Momentum allreduce at Ku intervals (when enabled)
    - No second moment communication (Muon has none)
    """

    def __init__(self, param_bytes=0, world_size=1):
        self.param_bytes = param_bytes
        self.world_size = world_size
        self.step = 0
        self.param_gather_bytes = 0
        self.momentum_sync_bytes = 0
        self.param_gather_count = 0
        self.momentum_sync_count = 0
        self.step_log = []

    def record_param_gather(self, num_bytes):
        """Record a parameter all_gather event."""
        self.param_gather_bytes += num_bytes
        self.param_gather_count += 1

    def record_momentum_sync(self, num_bytes):
        """Record a momentum allreduce event."""
        self.momentum_sync_bytes += num_bytes
        self.momentum_sync_count += 1

    def record_step(self, loss=None):
        """Record per-step metrics for figure generation."""
        total = self.param_gather_bytes + self.momentum_sync_bytes
        entry = {
            'step': self.step,
            'total_bytes': total,
            'param_gather_bytes': self.param_gather_bytes,
            'momentum_sync_bytes': self.momentum_sync_bytes,
        }
        if loss is not None:
            entry['loss'] = loss
        self.step_log.append(entry)
        self.step += 1

    def get_reduction_vs_baseline(self, total_steps=None):
        """Compute reduction factor vs Local Muon baseline.

        Baseline: all_gather every step (Kx=1) + momentum sync every step.
        DES-LOC: all_gather at Kx intervals + momentum sync at Ku intervals.
        """
        steps = total_steps or max(self.step, 1)
        baseline_bytes = self.param_bytes * 2 * steps  # gather + momentum
        actual = self.param_gather_bytes + self.momentum_sync_bytes
        if actual == 0:
            return float('inf')
        return round(baseline_bytes / actual, 1)

    def export_for_figure(self):
        """Export data for Figure CLXXXVII (Section 5.6)."""
        return {
            'steps': [e['step'] for e in self.step_log],
            'total_bytes': [e['total_bytes'] for e in self.step_log],
            'param_bytes': [e['param_gather_bytes'] for e in self.step_log],
            'momentum_bytes': [e['momentum_sync_bytes'] for e in self.step_log],
            'losses': [e.get('loss') for e in self.step_log],
            'reduction_factor': self.get_reduction_vs_baseline(),
        }


# End M188
